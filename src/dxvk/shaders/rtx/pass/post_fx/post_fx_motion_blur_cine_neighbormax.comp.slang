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

// Cinematic motion blur, pass 3 of 4: NeighborMax.
//
// Dilates TileMax over the one ring neighbourhood so a tile knows the fastest motion that
// can reach into it. This is the stage that lets a moving object smear onto stationary
// background - without it a background pixel has no velocity of its own and stays sharp
// no matter what passes in front of it.
//
// Two refinements over McGuire 2012, both from Guertin 2014:
//   - section 4.4, off axis culling: a diagonal neighbour only contributes if its trail
//     actually leans toward this tile, which tightens the worst case overestimate.
//   - Eq. 2, angular variance: how much the neighbourhood disagrees about direction, used
//     by the gather to decide how many samples to spend on the second direction.

#include "rtx/pass/post_fx/post_fx.h"
#include "rtx/pass/post_fx/post_fx_motion_blur_cine_common.slangh"

[[vk::binding(POST_FX_MB_CINE_NEIGHBORMAX_INPUT)]]
Texture2D<float4> TileMax;

[[vk::binding(POST_FX_MB_CINE_NEIGHBORMAX_OUTPUT)]]
RWTexture2D<float4> NeighborMaxOutput;

layout(push_constant)
ConstantBuffer<PostFxMotionBlurCineTileArgs> cb;

// Diagonal neighbours sit sqrt(2) tiles away, so their trail has to be roughly aligned with
// the diagonal to overlap this tile at all; a trail leaning off that axis sweeps into one of
// the edge neighbours instead. Axis aligned neighbours are always considered, since even a
// small deviation there still overlaps. The alignment test takes an absolute value because a
// trail extends both ways from its pixel, so pointing away reaches this tile just as well.
bool neighborCanReachTile(const int2 offset, const float2 neighborVelocity, const float neighborSpeed)
{
  if (offset.x == 0 || offset.y == 0)
  {
    return true;
  }

  if (neighborSpeed < POST_FX_MB_MIN_SPEED)
  {
    return false;
  }

  const float2 towardCenter = normalize(-float2(offset));
  const float2 direction = neighborVelocity / neighborSpeed;

  // cos(60 degrees). Deliberately permissive: over inclusion just reproduces the unmodified
  // algorithm, whereas excluding a neighbour that really does reach drops visible smear.
  return abs(dot(direction, towardCenter)) > 0.5f;
}

[shader("compute")]
[numthreads(POST_FX_TILE_SIZE, POST_FX_TILE_SIZE, 1)]
void main(in uint2 tilePos : SV_DispatchThreadID)
{
  if (any(tilePos >= cb.dstSize))
  {
    return;
  }

  const int2 maxTile = int2(cb.srcSize) - 1;

  const float4 centerTile = TileMax[clamp(int2(tilePos), int2(0, 0), maxTile)];
  const float2 centerDirection = motionBlurSafeNormalize(centerTile.xy, float2(0.0f, 0.0f));
  const bool centerHasMotion = centerTile.z >= POST_FX_MB_MIN_SPEED;

  float2 dominantVelocity = float2(0.0f, 0.0f);
  float dominantSpeed = 0.0f;

  float alignmentSum = 0.0f;
  float alignmentCount = 0.0f;

  [unroll]
  for (int y = -1; y <= 1; ++y)
  {
    [unroll]
    for (int x = -1; x <= 1; ++x)
    {
      const int2 offset = int2(x, y);
      const int2 samplePos = clamp(int2(tilePos) + offset, int2(0, 0), maxTile);
      const float4 neighborTile = TileMax[samplePos];
      const float neighborSpeed = neighborTile.z;

      if (neighborSpeed > dominantSpeed && neighborCanReachTile(offset, neighborTile.xy, neighborSpeed))
      {
        dominantSpeed = neighborSpeed;
        dominantVelocity = neighborTile.xy;
      }

      // Eq. 2. A neighbour with no motion does not disagree about direction, so it counts as
      // fully aligned rather than skewing the average toward spurious variance.
      float alignment = 1.0f;
      if (centerHasMotion && neighborSpeed >= POST_FX_MB_MIN_SPEED)
      {
        alignment = abs(dot(centerDirection, neighborTile.xy / neighborSpeed));
      }

      alignmentSum += alignment;
      alignmentCount += 1.0f;
    }
  }

  const float variance = saturate(1.0f - alignmentSum / max(alignmentCount, 1.0f));

  NeighborMaxOutput[tilePos] = float4(dominantVelocity, dominantSpeed, variance);
}
