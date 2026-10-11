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

// Fits the linear HDR image into the centre of an FFT buffer, near or coarse, leaving the padding black so that the
// circular convolution does not wrap glow around to the opposite edge. An FFT spreads every input texel over the
// whole frame, so the non-finite pixels are dropped here.

#include "rtx/pass/bloom/bloom.h"
#include "rtx/utility/math.slangh"

layout(binding = BLOOM_FFT_SETUP_COLOR_INPUT)
Texture2D<float4> InColor;

layout(rgba32f, binding = BLOOM_FFT_SETUP_SPECTRUM_OUTPUT)
RWTexture2D<float4> OutSpectrum;

layout(push_constant)
ConstantBuffer<BloomFftSetupArgs> cb;

static const int kMaxFilterRadius = 16;

// Resamples the image at center (in pixel index space, so integer coordinates are pixel centres) with a tent one
// buffer texel wide. Its weights over the buffer sum to one at every phase, so a small bright source keeps its
// energy and its position as it moves across the texels, which keeps its glow from pulsing. Taps off the image or
// not finite carry no weight.
vec3 tentResample(vec2 center, float footprint)
{
  const int radius = min(int(ceil(footprint)), kMaxFilterRadius);
  const vec2 centerFloor = floor(center);

  vec3 sum = vec3(0.0f);
  float weightSum = 0.0f;

  for (int y = -radius; y <= radius; y++)
  {
    const float sourceY = centerFloor.y + float(y);
    const float weightY = max(1.0f - abs(sourceY - center.y) / footprint, 0.0f);

    for (int x = -radius; x <= radius; x++)
    {
      const float sourceX = centerFloor.x + float(x);
      const float weight = max(1.0f - abs(sourceX - center.x) / footprint, 0.0f) * weightY;
      const int2 source = int2(int(sourceX), int(sourceY));

      if (weight <= 0.0f || any(source < int2(0)) || any(source >= int2(cb.imageSize)))
      {
        continue;
      }

      const vec3 value = InColor.Load(int3(source, 0)).rgb;

      if (any(isnan(value)) || any(isinf(value)))
      {
        continue;
      }

      sum += weight * min(value, vec3(cb.maxInputRadiance));
      weightSum += weight;
    }
  }

  return weightSum > 0.0f ? max(sum / weightSum, vec3(0.0f)) : vec3(0.0f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 texel : SV_DispatchThreadID)
{
  if (any(texel >= cb.bufferSize))
  {
    return;
  }

  const float pixelsPerTexel = 1.0f / cb.texelsPerPixel;
  const float2 pixelCenter = (float2(texel) + 0.5f - cb.bufferOffset) * pixelsPerTexel;
  float3 color = float3(0.0f);

  if (all(pixelCenter >= float2(0.0f)) && all(pixelCenter < float2(cb.imageSize)))
  {
    // The filter spans at least a pixel, so the buffer still interpolates when it is finer than the image.
    color = tentResample(pixelCenter - 0.5f, max(pixelsPerTexel, 1.0f));
  }

  OutSpectrum[texel] = float4(color, 0.0f);
}
