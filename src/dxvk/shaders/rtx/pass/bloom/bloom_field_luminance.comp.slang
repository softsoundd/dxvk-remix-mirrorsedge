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

// The luminance of the visual field for the eye's pupil: the mean over the image in the coarse buffer, plus the
// restored sun's illuminance spread over the field's solid angle, so that the product with the field's area is the
// corneal flux density Watson and Yellott's formula takes. One workgroup.

#include "rtx/pass/bloom/bloom.h"

layout(binding = BLOOM_FIELD_LUMINANCE_INPUT)
Texture2D<float4> InCoarseImage;

layout(binding = BLOOM_FIELD_LUMINANCE_SUN_VISIBILITY_INPUT)
Texture2D<float4> InSunVisibility;

layout(rgba32f, binding = BLOOM_FIELD_LUMINANCE_OUTPUT)
RWTexture2D<float4> OutFieldLuminance;

layout(push_constant)
ConstantBuffer<BloomFieldLuminanceArgs> cb;

static const uint kGroupSize = 256;

groupshared float g_sums[kGroupSize];

[shader("compute")]
[numthreads(kGroupSize, 1, 1)]
void main(uint threadIndex : SV_GroupIndex)
{
  const uint2 minTexel = uint2(max(cb.imageMin, vec2(0.0f)));
  const uint2 maxTexel = min(uint2(ceil(cb.imageMax)), cb.bufferSize);
  const uint width = maxTexel.x - minTexel.x;
  const uint count = width * (maxTexel.y - minTexel.y);
  float sum = 0.0f;

  for (uint i = threadIndex; i < count; i += kGroupSize)
  {
    const uint2 texel = minTexel + uint2(i % width, i / width);
    sum += dot(InCoarseImage.Load(int3(texel, 0)).rgb, vec3(0.2126f, 0.7152f, 0.0722f));
  }

  g_sums[threadIndex] = sum;
  GroupMemoryBarrierWithGroupSync();

  for (uint stride = kGroupSize / 2; stride > 0; stride /= 2)
  {
    if (threadIndex < stride)
    {
      g_sums[threadIndex] += g_sums[threadIndex + stride];
    }

    GroupMemoryBarrierWithGroupSync();
  }

  if (threadIndex == 0)
  {
    float luminance = count > 0 ? g_sums[0] / float(count) : 0.0f;

    if (cb.sunEnabled)
    {
      const vec3 visibility = InSunVisibility.Load(int3(0, 0, 0)).rgb;
      luminance += cb.sunLuminance * dot(visibility, vec3(0.2126f, 0.7152f, 0.0722f));
    }

    OutFieldLuminance[uint2(0, 0)] = float4(luminance, 0.0f, 0.0f, 0.0f);
  }
}
