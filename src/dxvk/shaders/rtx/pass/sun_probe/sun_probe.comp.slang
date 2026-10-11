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

// Measures the visible fraction of the sun disc, per channel, in the linear HDR image. The disc is sampled inside
// 0.8 of its radius, clear of the limb that antialiasing blends with the sky, and an annulus around it gives the
// light in front of the sun (sky, fog, a cloud's own glow), which is subtracted. What remains, against the disc's
// expected clamped radiance, is the share of the sun that gets through. Samples off the screen do not count.

#include "rtx/pass/sun_probe/sun_probe.h"

layout(binding = SUN_PROBE_COLOR_INPUT)
Texture2D<float4> InColor;

layout(rgba32f, binding = SUN_PROBE_VISIBILITY_OUTPUT)
RWTexture2D<float4> OutVisibility;

layout(push_constant)
ConstantBuffer<SunProbeArgs> cb;

static const uint kGroupSize = SUN_PROBE_GROUP_SIZE;
static const uint kRegionSamples = kGroupSize / 2;
static const float kGoldenAngle = 2.39996323f;

// RGB is the sum of the observed radiance and A the number of samples on the screen.
groupshared vec4 g_observed[kGroupSize];
groupshared vec3 g_expected[kGroupSize];

vec3 loadBilinear(vec2 position)
{
  const vec2 coordinate = position - 0.5f;
  const vec2 base = floor(coordinate);
  const vec2 f = coordinate - base;
  const int2 maxTexel = int2(cb.imageSize) - 1;

  vec3 result = vec3(0.0f);

  [unroll]
  for (int j = 0; j < 2; j++)
  {
    [unroll]
    for (int i = 0; i < 2; i++)
    {
      const int2 texel = clamp(int2(base) + int2(i, j), int2(0), maxTexel);
      vec3 value = InColor.Load(int3(texel, 0)).rgb;

      if (any(isnan(value)) || any(isinf(value)))
      {
        value = vec3(0.0f);
      }

      result += (i == 0 ? 1.0f - f.x : f.x) * (j == 0 ? 1.0f - f.y : f.y) * value;
    }
  }

  return result;
}

[shader("compute")]
[numthreads(kGroupSize, 1, 1)]
void main(uint threadIndex : SV_GroupIndex)
{
  const bool disc = threadIndex < kRegionSamples;
  const uint index = disc ? threadIndex : threadIndex - kRegionSamples;
  const float u = (float(index) + 0.5f) / float(kRegionSamples);

  // The samples lie on equal area spirals over the disc and the annulus.
  const float annulusInner = cb.annulusInnerRadiusPixels;
  const float annulusOuter = 1.25f * annulusInner;
  const float radius = disc
    ? 0.8f * cb.discRadiusPixels * sqrt(u)
    : sqrt(lerp(annulusInner * annulusInner, annulusOuter * annulusOuter, u));
  const float angle = float(index) * kGoldenAngle;
  const vec2 position = cb.sunPixel + radius * vec2(cos(angle), sin(angle));

  vec4 observed = vec4(0.0f);
  vec3 expected = vec3(0.0f);

  if (all(position >= vec2(0.0f)) && all(position < vec2(cb.imageSize)))
  {
    observed = vec4(loadBilinear(position), 1.0f);

    if (disc)
    {
      // Hestroffer and Magnan limb darkening, as evalSunDisk draws it.
      const float normalizedRadius = radius / max(cb.discRadiusPixels, 1e-6f);
      const float mu = sqrt(max(1.0f - normalizedRadius * normalizedRadius, 1e-8f));

      expected = min(cb.centerRadiance * pow(vec3(mu), cb.limbDarkeningExponent), vec3(cb.maxDiscRadiance));
    }
  }

  g_observed[threadIndex] = observed;
  g_expected[threadIndex] = expected;
  GroupMemoryBarrierWithGroupSync();

  // The two halves reduce side by side.
  for (uint stride = kRegionSamples / 2; stride > 0; stride /= 2)
  {
    if (index < stride)
    {
      g_observed[threadIndex] += g_observed[threadIndex + stride];
      g_expected[threadIndex] += g_expected[threadIndex + stride];
    }

    GroupMemoryBarrierWithGroupSync();
  }

  if (threadIndex != 0)
  {
    return;
  }

  const vec4 discSum = g_observed[0];
  const vec4 annulusSum = g_observed[kRegionSamples];

  if (discSum.w <= 0.0f)
  {
    return;
  }

  const vec3 discMean = discSum.rgb / discSum.w;
  const vec3 expectedMean = g_expected[0] / discSum.w;
  const vec3 background = annulusSum.w > 0.0f ? annulusSum.rgb / annulusSum.w : vec3(0.0f);

  const vec3 measured = vec3(
    expectedMean.r > 0.0f ? saturate((discMean.r - background.r) / expectedMean.r) : 0.0f,
    expectedMean.g > 0.0f ? saturate((discMean.g - background.g) / expectedMean.g) : 0.0f,
    expectedMean.b > 0.0f ? saturate((discMean.b - background.b) / expectedMean.b) : 0.0f);

  OutVisibility[uint2(0, 0)] = vec4(measured, 1.0f);
}
