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

// Cinematic motion blur, pass 2 of 4: TileMax.
//
// Reduces the per pixel velocity field to the dominant velocity per r x r tile, where r is
// the maximum blur radius. Run twice, once per axis - the paper notes in section 3 that
// splitting the reduction this way is substantially faster than a single 2D pass for the
// cost of one intermediate.
//
// Output layout, shared with NeighborMax:
//   xy = dominant velocity, display pixels
//   z  = its magnitude
//   w  = unused here, angular variance once NeighborMax has run

#include "rtx/pass/post_fx/post_fx.h"
#include "rtx/pass/post_fx/post_fx_motion_blur_cine_common.slangh"

[[vk::binding(POST_FX_MB_CINE_TILEMAX_INPUT)]]
Texture2D<float4> TileMaxInput;

[[vk::binding(POST_FX_MB_CINE_TILEMAX_OUTPUT)]]
RWTexture2D<float4> TileMaxOutput;

layout(push_constant)
ConstantBuffer<PostFxMotionBlurCineTileArgs> cb;

[shader("compute")]
[numthreads(POST_FX_TILE_SIZE, POST_FX_TILE_SIZE, 1)]
void main(in uint2 tilePos : SV_DispatchThreadID)
{
  if (any(tilePos >= cb.dstSize))
  {
    return;
  }

  // Along the reduction axis the destination indexes whole tiles, across it the destination
  // still indexes single texels, so the stride is 1 on the axis that is not being reduced.
  const int2 stride = int2(1, 1) + cb.pixelStep * int(cb.tileSize - 1u);
  const int2 sourceStart = int2(tilePos) * stride;
  const int2 maxSource = int2(cb.srcSize) - 1;

  float2 dominantVelocity = float2(0.0f, 0.0f);
  float dominantSpeedSquared = 0.0f;

  for (uint i = 0u; i < cb.tileSize; ++i)
  {
    const int2 samplePos = clamp(sourceStart + int2(i) * cb.pixelStep, int2(0, 0), maxSource);
    const float2 velocity = TileMaxInput[samplePos].xy;
    const float speedSquared = dot(velocity, velocity);

    if (speedSquared > dominantSpeedSquared)
    {
      dominantSpeedSquared = speedSquared;
      dominantVelocity = velocity;
    }
  }

  TileMaxOutput[tilePos] = float4(dominantVelocity, sqrt(dominantSpeedSquared), 0.0f);
}
