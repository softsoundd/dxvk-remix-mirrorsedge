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
 * Cloud placement map bake (fork — 2026-06-11, column-shaping rework).
 *
 * Fills the 512x512 RGBA8 cloud placement map that drives the per-column
 * cloud model in sampleCloudDensityTextured / sampleCloudDensityForShadow:
 *
 *   R — cluster field F [0,1]: Worley-cell cloud clusters modulated by a
 *       value-noise FBM. Cell centers are cloud cores (F high); cell
 *       boundaries are gaps (F low). The runtime coverage remap turns this
 *       into per-column cloud "presence" — it is the horizontal silhouette
 *       source for individual clouds, replacing the role the thresholded
 *       3D noise used to play (which placed mass independently per
 *       altitude and produced vertically disconnected stacked puffs).
 *   G — top-height jitter J [0,1]: smooth decorrelated FBM sampled at
 *       cluster scale. Maps to a per-cloud tower-height multiplier so
 *       equal-presence clusters still get different tops (kills the
 *       uniform "every cloud ends at the slab top" plane).
 *   B — base lift [0,1]: low-frequency undulation of the local cloud base
 *       (small fraction of the slab — cloud decks are flat-ish but not
 *       machined flat).
 *   A — unused (1.0).
 *
 * The map tiles at cloudNoiseTileKm — the SAME period as the 3D noise
 * volume — so the runtime's hex de-tiling lattice (computeCloudHexTile)
 * applies to both with the same per-cell offsets: a cloud cluster and the
 * 3D noise that textures it travel together across lattice cells.
 *
 * All lattices are wrapped to integer periods so the map tiles seamlessly.
 * The cluster cell count derives from cloudCellSizeKm (rounded to an
 * integer cells-per-tile); re-baked live when cloudCellSizeKm or
 * cloudNoiseTileKm changes via RtxAtmosphere::needsCloudPlacementRebake.
 *
 * Dispatch: 64x64 thread groups, 8x8 threads each -> 512x512 texels.
 */

#include "rtx/pass/atmosphere/atmosphere_common.slangh"

layout(binding = 0)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

layout(binding = 1, rgba8)
RWTexture2D<float4> cloudPlacementMapOutput;

// Must match RtxAtmosphere::kCloudPlacementMapSize.
static const float kPlacementMapSize = 512.0f;

// Design mean of the R-channel cluster field. The runtime hex-blend
// (sampleCloudPlacement in atmosphere_common.slangh) anchors its
// variance-preserving blend on this value — keep both sites in lockstep
// with any retune of the field construction below.
// (Shared constant kCloudPlacementFieldMean lives in atmosphere_common.slangh.)

// ---------------------------------------------------------------------------
// Periodic 2D value noise + FBM. hash31 (atmosphere_common.slangh) hashes the
// wrapped integer lattice so the pattern tiles exactly at `period` cells.
// Returns [0,1], mean ~0.5.
// ---------------------------------------------------------------------------
float valueNoise2DPeriodic(vec2 p, float period, float seed) {
  vec2 cell = floor(p);
  vec2 f = p - cell;
  f = f * f * (3.0f - 2.0f * f);

  // Wrap lattice corners to the period (positive modulo).
  vec2 c00 = cell - period * floor(cell / period);
  vec2 c10 = (cell + vec2(1.0f, 0.0f)); c10 -= period * floor(c10 / period);
  vec2 c01 = (cell + vec2(0.0f, 1.0f)); c01 -= period * floor(c01 / period);
  vec2 c11 = (cell + vec2(1.0f, 1.0f)); c11 -= period * floor(c11 / period);

  float v00 = hash31(vec3(c00, seed));
  float v10 = hash31(vec3(c10, seed));
  float v01 = hash31(vec3(c01, seed));
  float v11 = hash31(vec3(c11, seed));

  return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
}

float fbm2DPeriodic(vec2 p, int octaves, float period, float seed) {
  float sum = 0.0f;
  float amp = 0.5f;
  float ampSum = 0.0f;
  float freq = 1.0f;
  for (int i = 0; i < octaves; ++i) {
    sum    += amp * valueNoise2DPeriodic(p * freq, period * freq, seed + float(i) * 13.7f);
    ampSum += amp;
    amp    *= 0.5f;
    freq   *= 2.0f;
  }
  return sum / max(ampSum, 1e-4f);
}

// ---------------------------------------------------------------------------
// Periodic 2D Worley (min feature-point distance over the 3x3 neighborhood).
// Returns ~[0,1]: 0 at a feature point (cluster core), 1 toward the farthest
// cell-boundary corners.
// ---------------------------------------------------------------------------
float worleyNoise2DPeriodic(vec2 p, float period, float seed) {
  vec2 cell = floor(p);
  vec2 f = p - cell;

  float minDist = 8.0f;
  for (int y = -1; y <= 1; ++y) {
    for (int x = -1; x <= 1; ++x) {
      vec2 neighbor = vec2(float(x), float(y));
      vec2 wrapped = cell + neighbor;
      wrapped -= period * floor(wrapped / period);
      vec2 featurePoint = vec2(hash31(vec3(wrapped, seed)),
                               hash31(vec3(wrapped, seed + 31.7f)));
      vec2 diff = neighbor + featurePoint - f;
      minDist = min(minDist, dot(diff, diff));
    }
  }
  return saturate(sqrt(minDist));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
  AtmosphereArgs args = atmosphereArgs;

  if (any(threadId.xy >= uint2(512u, 512u))) {
    return;
  }

  vec2 uv = (vec2(threadId.xy) + 0.5f) / kPlacementMapSize;

  // Integer cells-per-tile so the Worley lattice wraps seamlessly at the
  // tile boundary. cloudCellSizeKm is the user-facing knob ("average cloud
  // cluster footprint"); the rounding means the EFFECTIVE cell size snaps
  // to tileKm / round(tileKm / cellSizeKm).
  float cells = clamp(round(args.cloudNoiseTileKm / max(args.cloudCellSizeKm, 0.25f)),
                      3.0f, 32.0f);

  // R — cluster field. Worley cell cores carry the clusters; an FBM at
  // sub-cluster scale roughens the radial falloff so silhouettes read as
  // organic cloud banks rather than circular stamps. The post-saturation
  // mean of this construction is ~0.60 (Monte-Carlo) — recorded as
  // kCloudPlacementFieldMean in atmosphere_common.slangh for the
  // variance-preserving hex blend; keep both in lockstep on retune.
  float w = worleyNoise2DPeriodic(uv * cells, cells, 11.3f);
  float clusterCore = 1.0f - w;  // 1 at cluster cores, 0 at boundaries
  float shapeFbm = fbm2DPeriodic(uv * cells, 3, cells, 53.1f);
  float F = saturate(0.5f + (clusterCore - 0.45f) * 0.95f + (shapeFbm - 0.5f) * 0.65f);

  // G — top-height jitter. Smooth FBM near cluster frequency, decorrelated
  // seed: neighboring clusters get visibly different tower heights but the
  // field stays continuous (no per-cell plateau seams).
  float cellsJ = max(round(cells * 0.5f), 2.0f);
  float J = fbm2DPeriodic(uv * cellsJ, 2, cellsJ, 71.9f);

  // B — base lift. Low frequency (3 cycles per tile): the deck's ceiling
  // undulates over multi-km scales.
  float baseLift = fbm2DPeriodic(uv * 3.0f, 2, 3.0f, 113.7f);

  cloudPlacementMapOutput[threadId.xy] = vec4(F, J, baseLift, 1.0f);
}
