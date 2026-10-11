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

// Adds the convolved image to the scene. The buffers hold only the light the kernel spreads away from each pixel, and
// each pixel keeps the centre tap's share of its own at full resolution, so the result is the image convolved with the
// kernel without the buffers' blur. The sun's glare follows in the same pass.

#include "rtx/pass/bloom/bloom.h"

// Both buffers are sampled bilinearly and clamped to their edges.
layout(binding = BLOOM_FFT_COMPOSITE_BLOOM_INPUT)
Sampler2D<float4> InBloom;

layout(binding = BLOOM_FFT_COMPOSITE_CENTER_TAP_INPUT)
Texture2D<float4> InCenterTap;

layout(binding = BLOOM_FFT_COMPOSITE_COARSE_BLOOM_INPUT)
Sampler2D<float4> InCoarseBloom;

layout(binding = BLOOM_FFT_COMPOSITE_KERNEL_CONSTANTS_INPUT)
ConstantBuffer<BloomKernelArgs> kernelArgs;

layout(binding = BLOOM_FFT_COMPOSITE_SUN_GLARE_CONSTANTS_INPUT)
ConstantBuffer<BloomSunGlareArgs> sunGlare;

layout(binding = BLOOM_FFT_COMPOSITE_SUN_PSF_INPUT)
Sampler2D<float4> InSunPsf;

layout(binding = BLOOM_FFT_COMPOSITE_SUN_VISIBILITY_INPUT)
Texture2D<float4> InSunVisibility;

layout(binding = BLOOM_FFT_COMPOSITE_COLOR_INPUT_OUTPUT)
RWTexture2D<float4> InOutColor;

layout(push_constant)
ConstantBuffer<BloomFftCompositeArgs> cb;

#include "rtx/pass/bloom/bloom_kernel.slangh"
#include "rtx/pass/bloom/bloom_sun_glare.slangh"

// Uniform cubic B-spline over 4x4 texels, which smooths away the buffer's texels without ringing. Each pair of taps
// along an axis is one bilinear tap between them, weighted by their sum, so four bilinear taps cover all sixteen
// (Sigg and Hadwiger 2005, "Fast Third-Order Texture Filtering"). position is in continuous texel coordinates, with
// texel centres at + 0.5.
float3 sampleBSpline(Sampler2D<float4> buffer, uint2 bufferSize, float2 position)
{
  const float2 coordinate = position - 0.5f;
  const float2 base = floor(coordinate);
  const float2 t = coordinate - base;
  const float2 t2 = t * t;
  const float2 t3 = t2 * t;

  const float2 w0 = (1.0f - 3.0f * t + 3.0f * t2 - t3) / 6.0f;
  const float2 w1 = (3.0f * t3 - 6.0f * t2 + 4.0f) / 6.0f;
  const float2 w2 = (-3.0f * t3 + 3.0f * t2 + 3.0f * t + 1.0f) / 6.0f;
  const float2 w3 = t3 / 6.0f;
  const float2 g0 = w0 + w1;
  const float2 g1 = w2 + w3;
  const float2 inverseSize = 1.0f / float2(bufferSize);
  const float2 p0 = (base - 0.5f + w1 / g0) * inverseSize;
  const float2 p1 = (base + 1.5f + w3 / g1) * inverseSize;

  return g0.x * g0.y * buffer.SampleLevel(p0, 0.0f).rgb
       + g1.x * g0.y * buffer.SampleLevel(float2(p1.x, p0.y), 0.0f).rgb
       + g0.x * g1.y * buffer.SampleLevel(float2(p0.x, p1.y), 0.0f).rgb
       + g1.x * g1.y * buffer.SampleLevel(p1, 0.0f).rgb;
}

// The spatial kernel, centred on the screen, on a log scale from 1e-10 to 1 of its normalised weights.
float3 displayKernel(uint2 pixel)
{
  const float2 offset = (float2(pixel) + 0.5f - 0.5f * float2(cb.imageSize)) * cb.texelsPerPixel;
  const int2 size = int2(cb.bufferSize);
  const int2 texel = (int2(floor(offset)) % size + size) % size;
  const float3 weight = max(InBloom.Load(int3(texel, 0)).rgb, float3(1e-12f));

  return saturate((log10(weight) + 10.0f) / 10.0f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 pixel : SV_DispatchThreadID)
{
  if (any(pixel >= cb.imageSize))
  {
    return;
  }

  float4 color = InOutColor[pixel];

  if (cb.debugView == BLOOM_FFT_DEBUG_VIEW_KERNEL)
  {
    InOutColor[pixel] = float4(displayKernel(pixel), color.a);
    return;
  }

  if (any(isnan(color)) || any(isinf(color)))
  {
    return;
  }

  const float2 bufferPosition = (float2(pixel) + 0.5f) * cb.texelsPerPixel + cb.bufferOffset;
  float3 bloom = max(sampleBSpline(InBloom, cb.bufferSize, bufferPosition), float3(0.0f));

  if (cb.coarseEnabled)
  {
    const float2 coarsePosition = (float2(pixel) + 0.5f) * cb.coarseTexelsPerPixel + cb.coarseBufferOffset;
    bloom += max(sampleBSpline(InCoarseBloom, cb.coarseBufferSize, coarsePosition), float3(0.0f));
  }

  const float3 kept = cb.subtractCenter ? InCenterTap.Load(int3(0, 0, 0)).rgb : float3(0.0f);

  color.rgb = cb.debugView == BLOOM_FFT_DEBUG_VIEW_BLOOM_ONLY
    ? bloom
    : color.rgb + cb.intensity * (bloom - (1.0f - kept) * color.rgb);

  if (cb.sunGlareEnabled && !any(isnan(color.rgb)) && !any(isinf(color.rgb)))
  {
    const vec3 radiancePerShare = sunGlare.radiancePerShare * InSunVisibility.Load(int3(0, 0, 0)).rgb;

    // An occluded sun casts no glare. The branch is uniform across the dispatch.
    if (any(radiancePerShare > vec3(0.0f)))
    {
      color.rgb = addSunGlare(pixel, color.rgb, sunGlare, radiancePerShare, InSunPsf);
    }
  }

  InOutColor[pixel] = color;
}
