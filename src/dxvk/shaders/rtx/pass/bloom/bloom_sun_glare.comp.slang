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

// The glare of the sun's light that its rendered disc lacks, at full resolution and with no limit on its reach, on or
// off the screen. Near the sun the diffraction comes from the PSF texture filtered by the sun's disc, and past it from
// the analytic far field, spread across by the disc. The scatter is analytic throughout.

#include "rtx/pass/bloom/bloom.h"
#include "rtx/utility/math.slangh"

layout(binding = BLOOM_SUN_GLARE_CONSTANTS_INPUT)
ConstantBuffer<BloomKernelArgs> kernelArgs;

layout(binding = BLOOM_SUN_GLARE_PSF_INPUT)
Sampler2D<float4> InSunPsf;

layout(binding = BLOOM_SUN_GLARE_SUN_VISIBILITY_INPUT)
Texture2D<float4> InSunVisibility;

layout(binding = BLOOM_SUN_GLARE_COLOR_INPUT_OUTPUT)
RWTexture2D<float4> InOutColor;

layout(push_constant)
ConstantBuffer<BloomSunGlareArgs> cb;

#include "rtx/pass/bloom/bloom_kernel.slangh"

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 pixel : SV_DispatchThreadID)
{
  if (any(pixel >= cb.imageSize))
  {
    return;
  }

  float4 color = InOutColor[pixel];

  if (any(isnan(color.rgb)) || any(isinf(color.rgb)))
  {
    return;
  }

  const vec3 radiancePerShare = cb.radiancePerShare * InSunVisibility.Load(int3(0, 0, 0)).rgb;

  if (all(radiancePerShare <= vec3(0.0f)))
  {
    return;
  }

  // The pixel's true angle from the sun, as a radius in near texels and on axis pixels.
  const vec2 ndc = vec2(
    (float(pixel.x) + 0.5f) / float(cb.imageSize.x) * 2.0f - 1.0f,
    1.0f - (float(pixel.y) + 0.5f) / float(cb.imageSize.y) * 2.0f);
  const vec3 pixelDirection = normalize(vec3(ndc * cb.tanHalfFov, 1.0f));
  const vec3 sunDirection = normalize(vec3(cb.sunTan, 1.0f));
  const float theta = atan2(length(cross(pixelDirection, sunDirection)), dot(pixelDirection, sunDirection));
  const float radius = max(theta / cb.anglePerNearTexel, 1e-3f);
  const float radiusPixels = radius / cb.nearTexelsPerPixel;

  const vec2 offset = vec2(pixel) + 0.5f - cb.sunPixel;
  const float offsetLength = length(offset);
  const vec2 onAxis = (offsetLength > 1e-4f ? offset / offsetLength : vec2(1.0f, 0.0f)) * radiusPixels;
  // An output pixel's area in near texels.
  const float pixelArea = cb.nearTexelsPerPixel * cb.nearTexelsPerPixel;

  vec3 diffraction = vec3(0.0f);
  const float farBlend = cb.psfEnabled ? smoothstep(0.8f * cb.psfRadiusPixels, cb.psfRadiusPixels, radiusPixels) : 1.0f;

  if (farBlend < 1.0f)
  {
    const vec2 uv = onAxis / (kernelArgs.sunPsfPixelsPerTexel * float(kernelArgs.sunPsfSize)) + 0.5f;
    diffraction += (1.0f - farBlend) * InSunPsf.SampleLevel(uv, 0.0f).rgb;
  }

  if (farBlend > 0.0f)
  {
    // The spikes spread across over the sun's disc and the output pixel.
    const float angle = atan2(-onAxis.y, onAxis.x);
    const float spreadVariance = cb.discVarianceTexels + pixelArea / 12.0f;
    const vec3 farField = diffractionShare() * diffractionFarFieldChannels(radius, angle, spreadVariance) * spikeBoostAt(radius);

    diffraction += farBlend * farField * pixelArea;
  }

  // Per steradian at the pixel's true angle, over an on axis pixel's solid angle.
  const vec3 scatter = observerScatterDensity(theta, cb.veilScale) * (cb.anglePerNearTexel * cb.anglePerNearTexel * pixelArea);

  // Kept within the output's range, which the glare at the disc's centre far exceeds.
  color.rgb = max(color.rgb, min(color.rgb + radiancePerShare * (diffraction + scatter), vec3(cb.maxRadiance)));
  InOutColor[pixel] = color;
}
