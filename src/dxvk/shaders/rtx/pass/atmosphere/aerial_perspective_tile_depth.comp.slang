/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

// Farthest primary hit (forward distance) under each screen tile of the aerial perspective volume, which
// bounds how far the tile's froxel ray is integrated. One thread group per tile visits every pixel of
// it, so a single sky or window pixel keeps the whole tile open, as it must: a primary miss is written
// as the depth range, which no froxel of the tile exceeds.

#include "rtx/pass/atmosphere/atmosphere_common.slangh"

layout(binding = 0)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

layout(binding = 1)
Texture2D<float> primaryLinearViewZInput;

layout(binding = 2, r32f)
RWTexture2D<float> tileDepthOutput;

static const uint kGroupWidth = 16;
static const uint kGroupSize = kGroupWidth * kGroupWidth;

groupshared float s_maxDepth[kGroupSize];

[shader("compute")]
[numthreads(kGroupWidth, kGroupWidth, 1)]
void main(uint2 tile : SV_GroupID, uint2 localId : SV_GroupThreadID, uint localIndex : SV_GroupIndex) {
  const uint size = atmosphereArgs.aerialPerspectiveLutSize;

  uint2 resolution;
  primaryLinearViewZInput.GetDimensions(resolution.x, resolution.y);

  const uint2 begin = (tile * resolution) / size;
  const uint2 end = max(((tile + 1u) * resolution) / size, begin + 1u);

  // Linear view Z is negative for a right handed camera; the miss sentinel is positive for either.
  float maxDepth = 0.0f;
  for (uint y = begin.y + localId.y; y < end.y; y += kGroupWidth) {
    for (uint x = begin.x + localId.x; x < end.x; x += kGroupWidth) {
      maxDepth = max(maxDepth, abs(primaryLinearViewZInput.Load(int3(x, y, 0))));
    }
  }

  s_maxDepth[localIndex] = maxDepth;
  GroupMemoryBarrierWithGroupSync();

  for (uint stride = kGroupSize / 2; stride > 0; stride /= 2) {
    if (localIndex < stride) {
      s_maxDepth[localIndex] = max(s_maxDepth[localIndex], s_maxDepth[localIndex + stride]);
    }
    GroupMemoryBarrierWithGroupSync();
  }

  if (localIndex == 0 && all(tile < uint2(size, size))) {
    const bool unbounded = s_maxDepth[0] >= atmosphereArgs.aerialPerspectiveMissLinearViewZ;
    tileDepthOutput[tile] = unbounded ? atmosphereArgs.aerialPerspectiveDepthRange : s_maxDepth[0];
  }
}
