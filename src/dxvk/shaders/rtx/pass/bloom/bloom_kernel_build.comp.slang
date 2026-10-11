/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

//!variant bloom_kernel_build_near.comp KERNEL_PART=0
//!variant bloom_kernel_build_coarse.comp KERNEL_PART=1
//!variant bloom_kernel_build_sun.comp KERNEL_PART=2
//!end-variants

// Writes the observer's point spread function as each texel's share of the light. The near and coarse parts are in
// wrap around layout, the kernel's centre at texel (0, 0) and offsets past half the buffer negative, and the near part
// leaves the centre texel for the combine pass. Both windows end before the padding runs out, so the convolution cannot
// wrap light around into the image.
// The sun part is the diffraction alone, centred on its texture, in on axis output pixels.

#include "rtx/pass/bloom/bloom.h"
#include "rtx/utility/math.slangh"

layout(binding = BLOOM_KERNEL_BUILD_CONSTANTS_INPUT)
ConstantBuffer<BloomKernelArgs> kernelArgs;

// The aperture's normalised power spectrum, sampled bilinearly and wrapping around.
layout(binding = BLOOM_KERNEL_BUILD_APERTURE_POWER_INPUT)
Sampler2D<float> InAperturePower;

layout(rgba32f, binding = BLOOM_KERNEL_BUILD_OUTPUT)
RWTexture2D<float4> OutKernel;

#include "rtx/pass/bloom/bloom_kernel.slangh"

static const uint kMaxNearFieldSamples = 32;

// Power of the aperture's spectrum at frequency k, in spectrum texels, as a share of its total. Zero past Nyquist.
float aperturePower(vec2 k)
{
  const float size = float(kernelArgs.apertureSize);

  if (any(abs(k) >= vec2(0.5f * size)))
  {
    return 0.0f;
  }

  return InAperturePower.SampleLevel((k + 0.5f) / size, 0.0f);
}

// The share of the aperture's light that lands in a square of side texelWidth at offset, both in near texels: its
// spectrum integrated over the square.
float nearField(vec2 offset, float texelWidth, float spectrumTexelsPerTexel)
{
  const float spectrumTexelsPerSquare = spectrumTexelsPerTexel * texelWidth;
  const uint count = uint(clamp(ceil(spectrumTexelsPerSquare), 1.0f, float(kMaxNearFieldSamples)));
  const float step = 1.0f / float(count);
  float sum = 0.0f;

  for (uint j = 0; j < count; j++)
  {
    for (uint i = 0; i < count; i++)
    {
      const vec2 position = offset + ((vec2(float(i), float(j)) + 0.5f) * step - 0.5f) * texelWidth;

      sum += aperturePower(position * spectrumTexelsPerTexel);
    }
  }

  const float spectrumTexelsPerSample = spectrumTexelsPerSquare * step;

  return sum * spectrumTexelsPerSample * spectrumTexelsPerSample;
}

// Fraunhofer diffraction of the aperture across the visible spectrum, near field and far field, as the share of the
// light in a square of side texelWidth at offset, both in near texels.
vec3 diffraction(vec2 offset, float texelWidth)
{
  const float radius = length(offset);
  const float angle = atan2(-offset.y, offset.x);
  const float nyquistTexelsPerSigma = 0.5f * float(kernelArgs.apertureSize) / kernelArgs.spectrumTexelsPerSigma;
  vec3 result = vec3(0.0f);

  for (uint i = 0; i < BLOOM_KERNEL_SPECTRAL_SAMPLES; i++)
  {
    const BloomKernelSpectralSample spectralSample = kernelArgs.spectralSamples[i];
    const float sigma = kernelArgs.sigmaTexels / spectralSample.scale;
    const float nyquist = nyquistTexelsPerSigma * sigma;
    const float farShare = smoothstep(kernelArgs.farFieldStart * nyquist, kernelArgs.farFieldEnd * nyquist, radius);
    float share = 0.0f;

    if (farShare < 1.0f)
    {
      share += (1.0f - farShare) * nearField(offset, texelWidth, kernelArgs.spectrumTexelsPerSigma / sigma);
    }

    if (farShare > 0.0f)
    {
      share += farShare * diffractionFarField(max(radius, 1e-3f), angle, sigma, texelWidth * texelWidth / 12.0f) * texelWidth * texelWidth;
    }

    result += spectralSample.weight * share;
  }

  return result * spikeBoostAt(radius);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 texel : SV_DispatchThreadID)
{
#if KERNEL_PART == 2
  const uint size = kernelArgs.sunPsfSize;

  if (any(texel >= uint2(size)))
  {
    return;
  }

  // The offset is in on axis output pixels from the texture's centre, then in near texels.
  const float pixelsPerTexel = kernelArgs.sunPsfPixelsPerTexel;
  const vec2 offsetPixels = (vec2(texel) + 0.5f - 0.5f * float(size)) * pixelsPerTexel;
  const float texelWidth = pixelsPerTexel * kernelArgs.nearTexelsPerPixel;
  const vec3 share = diffractionShare() * diffraction(offsetPixels * kernelArgs.nearTexelsPerPixel, texelWidth);

  // The texture holds the share per output pixel.
  OutKernel[texel] = float4(max(share, vec3(0.0f)) / (pixelsPerTexel * pixelsPerTexel), 0.0f);
#else
#if KERNEL_PART == 1
  const uint2 bufferSize = kernelArgs.coarseBufferSize;
  const float texelScale = kernelArgs.nearTexelsPerCoarseTexel;
#else
  const uint2 bufferSize = kernelArgs.nearBufferSize;
  const float texelScale = 1.0f;
#endif

  if (any(texel >= bufferSize))
  {
    return;
  }

  const int2 size = int2(bufferSize);
  const int2 t = int2(texel);
  const vec2 offset = texelScale * vec2(
    float(t.x <= size.x / 2 ? t.x : t.x - size.x),
    float(t.y <= size.y / 2 ? t.y : t.y - size.y));
  const float radius = length(offset);
  const float nearWindow = kernelWindow(radius, kernelArgs.nearWindowRadius);
  vec3 kernel = vec3(0.0f);

#if KERNEL_PART == 1
  // Beyond the near window, where the kernel is smooth enough to sample at the coarse texels' centres.
  const float weight = (1.0f - nearWindow) * kernelWindow(radius, kernelArgs.coarseWindowRadius);

  if (weight > 0.0f && radius < kernelArgs.coarseWindowRadius)
  {
    kernel = kernelFarPart(offset, texelScale) * (weight * texelScale * texelScale);
  }
#else
  if (radius > 0.0f && radius < kernelArgs.nearWindowRadius)
  {
    kernel = (diffractionShare() * diffraction(offset, 1.0f) + observerScatterTexel(offset)) * nearWindow;
  }
#endif

  OutKernel[texel] = float4(max(kernel, vec3(0.0f)), 0.0f);
#endif
}
