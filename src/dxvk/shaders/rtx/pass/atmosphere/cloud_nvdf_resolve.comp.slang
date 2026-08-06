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
 * Cloud NVDF signed resolve (fork — Nubis3 conversion Phase A).
 *
 * Converts the final JFA seed buffer into the published SDF: for each voxel,
 * distance to its closest boundary seed in km, negated when the voxel is
 * inside the body (sign from the occupancy grid — the reason single-field
 * boundary-seeded JFA suffices).
 *
 * Precision notes:
 *  - Output is raw signed km in R16F. R16F ulp is ~4 m at 6 km and ~6 cm at
 *    0.1 km — always below the ~47 m voxel size, so no normalization pass is
 *    needed (Nubis3 stores [0,1] only for BC1 compression, which we skip).
 *  - Distances are voxel-CENTER to voxel-center; the true zero crossing lies
 *    between a boundary voxel and its empty neighbor, so the field carries a
 *    half-voxel (~23 m) outward bias at the surface. Both the profile remap
 *    and the (safety-scaled) sphere-trace stepping absorb this; a sub-voxel
 *    refinement pass is deliberately deferred until something visibly needs it.
 *  - Voxels with no seed at all (a sky with zero cloud bodies at the baked
 *    nominal coverage) resolve to +100 km: unambiguously "empty air" for
 *    both the profile early-out and stepping.
 *
 * Writes the BACK SDF buffer; RtxAtmosphere publish-swaps front/back after
 * this pass completes so consumers never see a half-baked field.
 *
 * Dispatch: 8x4x8 threads, (32, 16, 32) groups -> 256x64x256 voxels.
 */

#include "rtx/pass/atmosphere/atmosphere_common.slangh"
#include "rtx/pass/atmosphere/cloud_nvdf_common.slangh"

layout(binding = CLOUD_NVDF_RESOLVE_BINDING_CONSTANTS)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

layout(binding = CLOUD_NVDF_RESOLVE_BINDING_OCCUPANCY_INPUT)
Texture3D<float> occupancySrv;

layout(binding = CLOUD_NVDF_RESOLVE_BINDING_SEEDS_INPUT)
Texture3D<uint> seedsInput;

layout(binding = CLOUD_NVDF_RESOLVE_BINDING_SDF_OUTPUT, r16f)
RWTexture3D<float> sdfOutput;

// "No cloud anywhere" distance — far above any real in-slab distance (the
// wrapped tile caps XZ distances at tileKm/sqrt(2) ~ 8.5 km) yet far below
// R16F max (65504), so downstream math can't overflow.
static const float kCloudNvdfEmptyKm = 100.0f;

[shader("compute")]
[numthreads(8, 4, 8)]
void main(uint3 threadId : SV_DispatchThreadID) {
  if (any(threadId >= uint3(CLOUD_NVDF_SIZE_XZ, CLOUD_NVDF_SIZE_Y, CLOUD_NVDF_SIZE_XZ))) {
    return;
  }

  AtmosphereArgs args = atmosphereArgs;

  uint seed = seedsInput.Load(int4(int3(threadId), 0)).r;

  float sdfKm;
  if (seed == kCloudNvdfSeedInvalid) {
    sdfKm = kCloudNvdfEmptyKm;
  } else {
    float d = cloudNvdfVoxelDistanceKm(threadId, cloudNvdfUnpackSeed(seed),
                                       args.cloudNoiseTileKm, args.cloudThickness);
    bool inside = occupancySrv.Load(int4(int3(threadId), 0)).r > 0.5f;
    sdfKm = inside ? -d : d;
  }

  sdfOutput[threadId] = sdfKm;
}
