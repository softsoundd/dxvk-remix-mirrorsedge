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

// Jump flooding over the occupancy grid (Rong and Tan 2006). Mode 0 seeds every boundary voxel (occupied with an
// empty face neighbour; outside the slab counts as empty, so a body touching the base or top has its surface
// there); mode 1 is a jump pass, each voxel adopting the nearest seed among its own and its 26 neighbours' at
// the jump distance. Seeding the boundary alone, with the sign recovered from the occupancy at resolve time,
// needs one field rather than the inside and outside pair. Distances compare in km, wrap aware.

#include "rtx/pass/clouds/cloud_binding_indices.h"
#include "rtx/pass/clouds/cloud_nvdf_common.slangh"

layout(binding = CLOUD_BINDING_CONSTANTS)
ConstantBuffer<CloudConstants> constants;

layout(binding = CLOUD_BINDING_BAKE_INPUT)
Texture3D<float> occupancyInput;

layout(binding = CLOUD_BINDING_BAKE_INPUT2)
Texture3D<uint> seedsInput;

layout(binding = CLOUD_BINDING_BAKE_OUTPUT, r32ui)
RWTexture3D<uint> seedsOutput;

layout(push_constant)
ConstantBuffer<CloudNvdfJfaArgs> pushArgs;

bool isOccupied(int3 voxel) {
  if (voxel.y < 0 || voxel.y >= int(CLOUD_NVDF_SIZE_Y)) {
    return false;
  }
  return occupancyInput.Load(int4(voxel, 0)) > 0.5f;
}

[shader("compute")]
[numthreads(8, 4, 8)]
void main(uint3 threadId : SV_DispatchThreadID) {
  if (any(threadId >= uint3(CLOUD_NVDF_SIZE_XZ, CLOUD_NVDF_SIZE_Y, CLOUD_NVDF_SIZE_XZ))) {
    return;
  }

  if (pushArgs.mode == 0u) {
    uint seed = kCloudNvdfSeedInvalid;
    if (isOccupied(int3(threadId))) {
      const int3 offsets[6] = {
        int3(1, 0, 0), int3(-1, 0, 0),
        int3(0, 1, 0), int3(0, -1, 0),
        int3(0, 0, 1), int3(0, 0, -1),
      };
      [unroll]
      for (int i = 0; i < 6; ++i) {
        int3 neighbour = int3(threadId) + offsets[i];
        neighbour.x = cloudNvdfWrapXZ(neighbour.x);
        neighbour.z = cloudNvdfWrapXZ(neighbour.z);
        if (!isOccupied(neighbour)) {
          seed = cloudNvdfPackSeed(threadId);
          break;
        }
      }
    }
    seedsOutput[threadId] = seed;
    return;
  }

  const CloudArgs cloud = constants.cloud;
  const int jump = int(pushArgs.jumpSizeVoxels);

  uint bestSeed = seedsInput.Load(int4(int3(threadId), 0));
  float bestDistance = bestSeed == kCloudNvdfSeedInvalid
    ? 1e30f
    : cloudNvdfVoxelDistanceKm(threadId, cloudNvdfUnpackSeed(bestSeed), cloud.tileKm, cloud.thicknessKm);

  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0) {
          continue;
        }
        int3 neighbour = int3(threadId) + int3(dx, dy, dz) * jump;
        if (neighbour.y < 0 || neighbour.y >= int(CLOUD_NVDF_SIZE_Y)) {
          continue;
        }
        neighbour.x = cloudNvdfWrapXZ(neighbour.x);
        neighbour.z = cloudNvdfWrapXZ(neighbour.z);

        const uint candidate = seedsInput.Load(int4(neighbour, 0));
        if (candidate == kCloudNvdfSeedInvalid) {
          continue;
        }
        const float distance = cloudNvdfVoxelDistanceKm(threadId, cloudNvdfUnpackSeed(candidate), cloud.tileKm, cloud.thicknessKm);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestSeed = candidate;
        }
      }
    }
  }

  seedsOutput[threadId] = bestSeed;
}
