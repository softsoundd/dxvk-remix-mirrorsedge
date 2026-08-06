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
 * Cloud NVDF occupancy voxelize (fork — Nubis3 conversion Phase A).
 *
 * Rasterizes the procedural cloud BODY — the placement map + per-column
 * model (formerly the legacy view sampler's gate) — into a tile-periodic
 * 256x64x256 R8 occupancy grid. This is the "Frankencloudscape" input the
 * JFA distance transform runs over; the resulting SDF gives Nubis3's
 * dimensional profile ("gradient from outside to inside") and, later,
 * sphere-traced march stepping.
 *
 * Deliberate exclusions:
 *  - hexOn = false: the NVDF is baked RAW tile-periodic; hex de-tiling
 *    offsets are applied at SAMPLE time (Phase B), exactly the treatment
 *    the placement map gets today. Baking the de-tiled field is impossible
 *    anyway — the hex lattice is world-space and breaks the tile period.
 *  - No DETAIL-noise taps: per Nubis3, the SDF describes the cloud BODY; the
 *    detail noise stays a sample-time value-erosion and must not shift the
 *    baked iso-surface. (The body-erosion carve below is different in kind:
 *    it is part of the BODY GENERATOR — Nubis3's authored frankenclouds are
 *    not convex, and a placement blob x column band is. See the carve note
 *    at the sample site.)
 *  - Coverage is evaluated at a fixed NOMINAL value (nvdfNominalCoverage),
 *    not the live weather coverage: at sample time the live coverage
 *    becomes a level-set offset on the SDF (clouds grow/shrink with zero
 *    rebake). Only when the live coverage drifts far from the baked nominal
 *    does the dirty key trigger a re-bake at a re-quantized nominal.
 *
 * Dispatch: 8x4x8 threads, (32, 16, 32) groups -> 256x64x256 voxels.
 * Axis convention: see cloud_nvdf.h (u = world X, v = VERTICAL, w = world Z).
 */

#include "rtx/pass/atmosphere/atmosphere_common.slangh"
#include "rtx/pass/atmosphere/cloud_nvdf_common.slangh"

layout(binding = CLOUD_NVDF_OCCUPANCY_BINDING_CONSTANTS)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

layout(binding = CLOUD_NVDF_OCCUPANCY_BINDING_OUTPUT, r8)
RWTexture3D<float> occupancyOutput;

// Same placement map resource the density samplers consume; the column model
// here MUST match computeCloudColumn's inputs so the SDF describes the same
// bodies the (legacy and Nubis3) samplers shade.
layout(binding = CLOUD_NVDF_OCCUPANCY_BINDING_PLACEMENT_INPUT)
Texture2D<float4> cloudPlacementMapSrv;

layout(binding = CLOUD_NVDF_OCCUPANCY_BINDING_SAMPLER)
SamplerState placementSamplerState;

[shader("compute")]
[numthreads(8, 4, 8)]
void main(uint3 threadId : SV_DispatchThreadID) {
  if (any(threadId >= uint3(CLOUD_NVDF_SIZE_XZ, CLOUD_NVDF_SIZE_Y, CLOUD_NVDF_SIZE_XZ))) {
    return;
  }

  AtmosphereArgs args = atmosphereArgs;

  // Placement tap at the voxel-center tile UV — the hexOn=false path of
  // the retired sampleCloudPlacement helper (a single linear tap; the map tiles at
  // cloudNoiseTileKm, same period as this grid's horizontal domain).
  vec2 tileUV = cloudNvdfVoxelToTileUV(threadId);
  vec4 placement = cloudPlacementMapSrv.SampleLevel(placementSamplerState, tileUV, 0);
  float hf = cloudNvdfVoxelToHeightFraction(threadId);

  // Body-erosion carve (fork — anti-blobby pass, 2026-07-16). The placement
  // blob x column band bakes CONVEX bodies; every downstream detail lever
  // (erosion, wobble, HF noise) is surface-deep and can't fix a smooth hull
  // — the deck reads as blobs. Shift the placement waterline field with a
  // tile-periodic 3D FBM before the column model: because the shift varies
  // with HEIGHT, one column is wide at one altitude and narrow at another —
  // overhangs, notches and lumps bake into the SDF itself (Nubis3's authored
  // frankenclouds are not convex). Zero-mean, so net coverage is preserved
  // and the nominal-coverage level-set semantics stay centered. Bake-time
  // only; nvdfBodyErosionStrength is an NVDF re-bake dirty key.
  if (args.nvdfBodyErosionStrength > 0.0f) {
    // ~2 km lumps at the default 12 km tile, 4 octaves down to ~250 m.
    // Y cycles keep the features roughly isotropic in km (integer for
    // lattice periodicity; the grid never wrap-samples vertically anyway).
    const float kCarveCyclesXZ = 6.0f;
    float carveCyclesY = max(1.0f, round(kCarveCyclesXZ * args.cloudThickness
                                         / max(args.cloudNoiseTileKm, 1e-3f)));
    vec3 carvePeriod = vec3(kCarveCyclesXZ, carveCyclesY, kCarveCyclesXZ);
    vec3 carveP = vec3(tileUV.x, hf, tileUV.y) * carvePeriod;
    float carve = fbmNoise3DPeriodic(carveP, 4, 67.3f, carvePeriod);
    placement.r = saturate(placement.r + carve * 0.25f * args.nvdfBodyErosionStrength);
  }

  // Column model at the NOMINAL coverage (see header). computeCloudColumn is
  // the same helper the density samplers call — one body definition.
  CloudColumn col = computeCloudColumn(placement, saturate(args.nvdfNominalCoverage), args);

  // Same gates as the legacy sampler's column early-outs: presence
  // epsilon, then the sample's height fraction against the column's own
  // [base, top] band.
  bool occupied = (col.presence > 0.001f)
               && (hf > col.baseFrac)
               && (hf < col.topFrac);

  occupancyOutput[threadId] = occupied ? 1.0f : 0.0f;
}
