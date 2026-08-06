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

/*
 * Cloud NVDF jump-flooding pass (fork — Nubis3 conversion Phase A).
 *
 * Two modes, selected by push constant (CloudNvdfJfaArgs.mode):
 *
 *  mode 0 — SEED INIT. An occupied voxel with at least one EMPTY 6-neighbor
 *    is a cloud-body BOUNDARY voxel; it seeds itself. Everything else starts
 *    with no seed. Single-field boundary seeding (instead of the classic
 *    inside/outside two-field JFA) halves the pass count; the SIGN of the
 *    final distance is recovered from the occupancy grid at resolve time.
 *    XZ neighbors wrap (the field is tile-periodic); Y neighbors outside the
 *    slab count as EMPTY, so a body touching the slab floor/ceiling has its
 *    surface there — matching the density samplers' hard base/top gates.
 *
 *  mode 1 — JUMP PASS at jumpSizeVoxels. Standard 3^3-neighborhood JFA step:
 *    each voxel adopts the closest seed among its own and its 26 neighbors'
 *    at offset +-jumpSizeVoxels per axis. Distances compare in PHYSICAL km
 *    (wrap-aware, anisotropic-voxel-aware — see cloudNvdfVoxelDistanceKm);
 *    comparing in voxel units would bias seeds along the cheaper axis
 *    whenever tile/thickness leave the voxels non-cubic.
 *
 * Pass schedule (driven by RtxAtmosphere): 128, 64, 32, 16, 8, 4, 2, 1, then
 * one extra 1-refinement pass (JFA+1 — kills most of the rare far-seed
 * misses plain JFA can leave). Ping-pong between the two R32_UINT seed
 * buffers; the CPU swaps bindings per pass.
 *
 * Dispatch: 8x4x8 threads, (32, 16, 32) groups -> 256x64x256 voxels.
 */

#include "rtx/pass/atmosphere/atmosphere_common.slangh"
#include "rtx/pass/atmosphere/cloud_nvdf_common.slangh"

layout(binding = CLOUD_NVDF_JFA_BINDING_CONSTANTS)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

// Occupancy (read in mode 0 only; bound in both modes for a stable layout).
layout(binding = CLOUD_NVDF_JFA_BINDING_OCCUPANCY_INPUT)
Texture3D<float> occupancySrv;

// Seed ping-pong (read in mode 1 only).
layout(binding = CLOUD_NVDF_JFA_BINDING_SEEDS_INPUT)
Texture3D<uint> seedsInput;

layout(binding = CLOUD_NVDF_JFA_BINDING_SEEDS_OUTPUT, r32ui)
RWTexture3D<uint> seedsOutput;

layout(push_constant)
ConstantBuffer<CloudNvdfJfaArgs> pushArgs;

bool isOccupied(int3 vox) {
  // Y out of range = empty (see header). XZ are pre-wrapped by callers.
  if (vox.y < 0 || vox.y >= int(CLOUD_NVDF_SIZE_Y)) {
    return false;
  }
  return occupancySrv.Load(int4(vox, 0)).r > 0.5f;
}

[shader("compute")]
[numthreads(8, 4, 8)]
void main(uint3 threadId : SV_DispatchThreadID) {
  if (any(threadId >= uint3(CLOUD_NVDF_SIZE_XZ, CLOUD_NVDF_SIZE_Y, CLOUD_NVDF_SIZE_XZ))) {
    return;
  }

  if (pushArgs.mode == 0u) {
    // --- Seed init -------------------------------------------------------
    uint seed = kCloudNvdfSeedInvalid;
    if (isOccupied(int3(threadId))) {
      // 6-neighborhood boundary test (face neighbors only — a voxel whose
      // face neighbors are all occupied is interior even if a diagonal is
      // empty; face-adjacency matches the marching resolution of the field).
      const int3 offsets[6] = {
        int3( 1, 0, 0), int3(-1, 0, 0),
        int3( 0, 1, 0), int3( 0,-1, 0),
        int3( 0, 0, 1), int3( 0, 0,-1),
      };
      [unroll]
      for (int i = 0; i < 6; ++i) {
        int3 n = int3(threadId) + offsets[i];
        n.x = cloudNvdfWrapXZ(n.x);
        n.z = cloudNvdfWrapXZ(n.z);
        if (!isOccupied(n)) {
          seed = cloudNvdfPackSeed(threadId);
          break;
        }
      }
    }
    seedsOutput[threadId] = seed;
    return;
  }

  // --- Jump pass ---------------------------------------------------------
  AtmosphereArgs args = atmosphereArgs;
  const float tileKm      = args.cloudNoiseTileKm;
  const float thicknessKm = args.cloudThickness;
  const int   jump        = int(pushArgs.jumpSizeVoxels);

  uint  bestSeed = seedsInput.Load(int4(int3(threadId), 0)).r;
  float bestDist = (bestSeed == kCloudNvdfSeedInvalid)
                 ? 1e30f
                 : cloudNvdfVoxelDistanceKm(threadId, cloudNvdfUnpackSeed(bestSeed),
                                            tileKm, thicknessKm);

  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0) {
          continue;
        }
        int3 n = int3(threadId) + int3(dx, dy, dz) * jump;
        // Y clamps out (no vertical wrap); XZ wrap onto the tile torus.
        if (n.y < 0 || n.y >= int(CLOUD_NVDF_SIZE_Y)) {
          continue;
        }
        n.x = cloudNvdfWrapXZ(n.x);
        n.z = cloudNvdfWrapXZ(n.z);

        uint cand = seedsInput.Load(int4(n, 0)).r;
        if (cand == kCloudNvdfSeedInvalid) {
          continue;
        }
        float d = cloudNvdfVoxelDistanceKm(threadId, cloudNvdfUnpackSeed(cand),
                                           tileKm, thicknessKm);
        if (d < bestDist) {
          bestDist = d;
          bestSeed = cand;
        }
      }
    }
  }

  seedsOutput[threadId] = bestSeed;
}
