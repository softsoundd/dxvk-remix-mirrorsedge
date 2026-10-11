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

// The sun's point spread function over its disc: each point of the disc casts the pattern, weighted by the disc's
// limb darkened radiance, mu^alpha with mu the cosine on the sun's surface (as evalSunDisk). The disc's edge is
// antialiased by its coverage of each texel.

#include "rtx/pass/bloom/bloom.h"

layout(binding = BLOOM_SUN_PSF_FILTER_INPUT)
Texture2D<float4> InPsf;

layout(rgba32f, binding = BLOOM_SUN_PSF_FILTER_OUTPUT)
RWTexture2D<float4> OutPsf;

layout(push_constant)
ConstantBuffer<BloomSunPsfFilterArgs> cb;

static const int kMaxRadius = 48;

vec3 discWeight(vec2 offset)
{
  const float radius = cb.discRadiusTexels;
  const float distance = length(offset);
  // Coverage of a texel by the disc's edge.
  const float coverage = saturate(radius + 0.5f - distance);

  if (coverage <= 0.0f)
  {
    return vec3(0.0f);
  }

  const float rho = min(distance / radius, 1.0f);
  const float mu = sqrt(max(1.0f - rho * rho, 0.0f));

  return coverage * pow(vec3(max(mu, 1e-3f)), cb.limbDarkeningExponent);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 texel : SV_DispatchThreadID)
{
  if (any(texel >= uint2(cb.size)))
  {
    return;
  }

  const int reach = min(int(ceil(cb.discRadiusTexels + 0.5f)), kMaxRadius);
  const int size = int(cb.size);
  vec3 sum = vec3(0.0f);
  vec3 weightSum = vec3(0.0f);

  for (int y = -reach; y <= reach; y++)
  {
    for (int x = -reach; x <= reach; x++)
    {
      const vec3 weight = discWeight(vec2(float(x), float(y)));

      if (all(weight <= vec3(0.0f)))
      {
        continue;
      }

      weightSum += weight;

      const int2 source = int2(texel) - int2(x, y);

      if (any(source < int2(0)) || any(source >= int2(size)))
      {
        continue;
      }

      sum += weight * InPsf.Load(int3(source, 0)).rgb;
    }
  }

  OutPsf[texel] = float4(sum / max(weightSum, vec3(1e-12f)), 0.0f);
}
