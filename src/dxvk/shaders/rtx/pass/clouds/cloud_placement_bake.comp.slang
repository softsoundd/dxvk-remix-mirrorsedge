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

// The placement map the per-column body model reads, tiling at the field's tile:
//   R: cluster field, Worley cell cores (clouds) and boundaries (gaps) shaped by value noise FBM.
//   G: per-cloud top height jitter, so equal presence clusters still reach different tops.
//   B: low frequency lift of the local cloud base.
// Every lattice wraps to an integer period. Adjacent integer cell counts blend, so the cell size is continuous.

#include "rtx/pass/clouds/cloud_binding_indices.h"
#include "rtx/pass/clouds/cloud_noise.slangh"
#include "rtx/pass/clouds/cloud_args.h"

layout(binding = CLOUD_BINDING_CONSTANTS)
ConstantBuffer<CloudConstants> constants;

layout(binding = CLOUD_BINDING_BAKE_OUTPUT, rgba8)
RWTexture2D<float4> placementOutput;

float valueNoise2DPeriodic(vec2 p, float period, float seed) {
  const vec2 cell = floor(p);
  vec2 f = p - cell;
  f = f * f * (3.0f - 2.0f * f);

  vec2 c00 = cell;
  vec2 c10 = cell + vec2(1.0f, 0.0f);
  vec2 c01 = cell + vec2(0.0f, 1.0f);
  vec2 c11 = cell + vec2(1.0f, 1.0f);
  c00 -= period * floor(c00 / period);
  c10 -= period * floor(c10 / period);
  c01 -= period * floor(c01 / period);
  c11 -= period * floor(c11 / period);

  const float v00 = cloudLatticeHash(c00, seed);
  const float v10 = cloudLatticeHash(c10, seed);
  const float v01 = cloudLatticeHash(c01, seed);
  const float v11 = cloudLatticeHash(c11, seed);
  return lerp(lerp(v00, v10, f.x), lerp(v01, v11, f.x), f.y);
}

float fbm2DPeriodic(vec2 p, int octaves, float period, float seed) {
  float sum = 0.0f;
  float amplitude = 0.5f;
  float amplitudeSum = 0.0f;
  float frequency = 1.0f;
  for (int i = 0; i < octaves; ++i) {
    sum += amplitude * valueNoise2DPeriodic(p * frequency, period * frequency, seed + float(i) * 13.7f);
    amplitudeSum += amplitude;
    amplitude *= 0.5f;
    frequency *= 2.0f;
  }
  return sum / max(amplitudeSum, 1e-4f);
}

float worleyNoise2DPeriodic(vec2 p, float period, float seed) {
  const vec2 cell = floor(p);
  const vec2 f = p - cell;
  float minDistanceSq = 8.0f;
  for (int y = -1; y <= 1; ++y) {
    for (int x = -1; x <= 1; ++x) {
      const vec2 neighbour = vec2(float(x), float(y));
      vec2 wrapped = cell + neighbour;
      wrapped -= period * floor(wrapped / period);
      const vec2 featurePoint = vec2(cloudLatticeHash(wrapped, seed), cloudLatticeHash(wrapped, seed + 31.7f));
      const vec2 d = neighbour + featurePoint - f;
      minDistanceSq = min(minDistanceSq, dot(d, d));
    }
  }
  return saturate(sqrt(minDistanceSq));
}

float clusterFieldAt(vec2 uv, float cells) {
  const float clusterCore = 1.0f - worleyNoise2DPeriodic(uv * cells, cells, 11.3f);
  const float shapeFbm = fbm2DPeriodic(uv * cells, 3, cells, 53.1f);
  return saturate(0.5f + (clusterCore - 0.45f) * 0.95f + (shapeFbm - 0.5f) * 0.65f);
}

float topJitterAt(vec2 uv, float cells) {
  const float jitterCells = max(round(cells * 0.5f), 2.0f);
  return fbm2DPeriodic(uv * jitterCells, 2, jitterCells, 71.9f);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
  if (any(threadId.xy >= uint2(CLOUD_PLACEMENT_MAP_SIZE))) {
    return;
  }

  const CloudArgs cloud = constants.cloud;
  const vec2 uv = (vec2(threadId.xy) + 0.5f) / float(CLOUD_PLACEMENT_MAP_SIZE);

  const float cells = clamp(cloud.tileKm / max(cloud.cellSizeKm, 0.25f), 1.0f, 32.0f);
  const float cellsLo = floor(cells);
  const float cellsHi = min(cellsLo + 1.0f, 32.0f);
  const float cellsBlend = smoothstep(0.0f, 1.0f, cells - cellsLo);

  const float cluster = lerp(clusterFieldAt(uv, cellsLo), clusterFieldAt(uv, cellsHi), cellsBlend);
  const float topJitter = lerp(topJitterAt(uv, cellsLo), topJitterAt(uv, cellsHi), cellsBlend);
  const float baseLift = fbm2DPeriodic(uv * 3.0f, 2, 3.0f, 113.7f);

  placementOutput[threadId.xy] = vec4(cluster, topJitter, baseLift, 1.0f);
}
