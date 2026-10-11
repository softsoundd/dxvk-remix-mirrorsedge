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

// Adds the traced ghosts drawn at half and quarter resolution to the output. Only ghosts whose edges the sun's disc
// blurs over several of a layer's pixels are drawn there, so the layers are sampled bilinearly. Each layer pixel holds
// the light over its whole area, which bilinear sampling keeps.

#include "rtx/pass/lens_flare/lens_flare.h"

// Both layers are sampled bilinearly and clamped to their edges.
layout(binding = LENS_FLARE_COMPOSITE_HALF_INPUT)
Sampler2D<float4> InHalf;

layout(binding = LENS_FLARE_COMPOSITE_QUARTER_INPUT)
Sampler2D<float4> InQuarter;

layout(binding = LENS_FLARE_COMPOSITE_COLOR_INPUT_OUTPUT)
RWTexture2D<float4> InOutColor;

layout(push_constant)
ConstantBuffer<LensFlareCompositeArgs> cb;

[shader("compute")]
[numthreads(LENS_FLARE_TILE_SIZE, LENS_FLARE_TILE_SIZE, 1)]
void main(uint2 pixel : SV_DispatchThreadID)
{
  if (any(pixel >= cb.imageSize))
  {
    return;
  }

  // The layers span the output's extent, so a pixel's centre has the same coordinate in each.
  const float2 uv = (float2(pixel) + 0.5f) * cb.invImageSize;
  float3 light = float3(0.0f);

  if (cb.drawHalf)
  {
    light += InHalf.SampleLevel(uv, 0.0f).rgb;
  }

  if (cb.drawQuarter)
  {
    light += InQuarter.SampleLevel(uv, 0.0f).rgb;
  }

  float4 color = InOutColor[pixel];
  color.rgb += light;
  InOutColor[pixel] = color;
}
