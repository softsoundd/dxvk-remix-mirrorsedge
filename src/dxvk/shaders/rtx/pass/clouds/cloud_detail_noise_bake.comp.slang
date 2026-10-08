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

// The periodic detail volume the density sampler erodes with:
//   R / G: low / high frequency wispy, a curl warped inversion of the alligator approximation (web-like strands).
//   B / A: low / high frequency billowy, the alligator approximation itself (layered lumps with creases).
// The alligator approximation keeps one jittered kernel of random amplitude per lattice cell and takes the
// strongest minus the runner up, so lumps of different sizes contest with creases between them rather than
// packing like inverted Worley's equal spheres. All lattices have integer periods and the curl potentials are
// periodic, so the volume tiles. A fixed pattern baked once.

#include "rtx/pass/clouds/cloud_binding_indices.h"
#include "rtx/pass/clouds/cloud_noise.slangh"
#include "rtx/pass/clouds/cloud_args.h"

layout(binding = CLOUD_BINDING_BAKE_OUTPUT, rgba8)
RWTexture3D<float4> detailOutput;

// Lattice frequencies in cycles per volume. With the volume repeating every few km, these put the erosion's
// content between about 470 and 90 m.
static const float kLowFrequencyCycles = 6.0f;
static const float kHighFrequencyCycles = 16.0f;

float alligator3DPeriodic(vec3 p, float seed, vec3 period) {
  const vec3 cell = floor(p);
  const vec3 f = p - cell;
  float m1 = 0.0f;
  float m2 = 0.0f;
  [unroll]
  for (int dz = -1; dz <= 1; ++dz) {
    [unroll]
    for (int dy = -1; dy <= 1; ++dy) {
      [unroll]
      for (int dx = -1; dx <= 1; ++dx) {
        const vec3 offset = vec3(float(dx), float(dy), float(dz));
        vec3 wrapped = cell + offset;
        wrapped -= period * floor(wrapped / period);
        const vec3 feature = vec3(
          cloudHash31(wrapped + vec3(seed + 11.317f, seed, seed)),
          cloudHash31(wrapped + vec3(seed, seed + 47.113f, seed)),
          cloudHash31(wrapped + vec3(seed, seed, seed + 73.951f)));
        const float amplitude = 0.25f + 0.75f * cloudHash31(wrapped + vec3(seed + 5.771f, seed + 9.043f, seed + 1.229f));
        const float d = length(offset + feature - f);
        const float k = amplitude * pow(saturate(1.0f - d * 0.7071f), 1.6f);
        m2 = max(m2, min(m1, k));
        m1 = max(m1, k);
      }
    }
  }
  return saturate((m1 - m2) * 1.25f);
}

float alligatorFbm3DPeriodic(vec3 p, int octaves, float seed, vec3 basePeriod) {
  float value = 0.0f;
  float amplitude = 0.5f;
  float total = 0.0f;
  vec3 period = basePeriod;
  for (int i = 0; i < octaves; ++i) {
    value += amplitude * alligator3DPeriodic(p, seed + 13.7f * float(i), period);
    total += amplitude;
    amplitude *= 0.5f;
    p *= 2.0f;
    period *= 2.0f;
  }
  return value / max(total, 1e-4f);
}

float curlPotentialDerivative(vec3 p, vec3 axis, float seed, vec3 period) {
  const float e = 0.25f;
  return (cloudFbmPeriodic(p + axis * e, 2, seed, period) - cloudFbmPeriodic(p - axis * e, 2, seed, period)) / (2.0f * e);
}

// Curl of a vector potential of three decorrelated periodic FBMs: divergence free, so it warps without
// bunching, and periodic like its potentials.
vec3 curlNoise3DPeriodic(vec3 p, float seed, vec3 period) {
  const vec3 ex = vec3(1.0f, 0.0f, 0.0f);
  const vec3 ey = vec3(0.0f, 1.0f, 0.0f);
  const vec3 ez = vec3(0.0f, 0.0f, 1.0f);
  const float dCdy = curlPotentialDerivative(p, ey, seed + 67.1f, period);
  const float dBdz = curlPotentialDerivative(p, ez, seed + 31.7f, period);
  const float dAdz = curlPotentialDerivative(p, ez, seed + 12.9f, period);
  const float dCdx = curlPotentialDerivative(p, ex, seed + 67.1f, period);
  const float dBdx = curlPotentialDerivative(p, ex, seed + 31.7f, period);
  const float dAdy = curlPotentialDerivative(p, ey, seed + 12.9f, period);
  return vec3(dCdy - dBdz, dAdz - dCdx, dBdx - dAdy);
}

[shader("compute")]
[numthreads(8, 8, 8)]
void main(uint3 threadId : SV_DispatchThreadID) {
  if (any(threadId >= uint3(CLOUD_DETAIL_NOISE_SIZE))) {
    return;
  }

  const vec3 p = (vec3(threadId) + 0.5f) / float(CLOUD_DETAIL_NOISE_SIZE);

  // Wisps are squeezed to half height (the volume's y is vertical), so the edge cuts read as sheared streaks
  // rather than round webs.
  const vec3 wispSqueeze = vec3(1.0f, 2.0f, 1.0f);
  const vec3 lowPeriodWisp = vec3(kLowFrequencyCycles) * wispSqueeze;
  const vec3 highPeriodWisp = vec3(kHighFrequencyCycles) * wispSqueeze;
  const float curlWarpCells = 0.55f;
  const vec3 warpLow = curlNoise3DPeriodic(p * lowPeriodWisp, 101.3f, lowPeriodWisp) * curlWarpCells;
  const vec3 warpHigh = curlNoise3DPeriodic(p * highPeriodWisp, 227.9f, highPeriodWisp) * curlWarpCells;
  const float wispyLow = pow(saturate(1.0f - alligatorFbm3DPeriodic(p * lowPeriodWisp + warpLow, 3, 23.1f, lowPeriodWisp)), 2.0f);
  const float wispyHigh = pow(saturate(1.0f - alligatorFbm3DPeriodic(p * highPeriodWisp + warpHigh, 2, 41.7f, highPeriodWisp)), 2.0f);

  const float billowyLow = saturate(alligatorFbm3DPeriodic(p * kLowFrequencyCycles, 3, 7.9f, vec3(kLowFrequencyCycles)));
  const float billowyHigh = saturate(alligatorFbm3DPeriodic(p * kHighFrequencyCycles, 2, 13.3f, vec3(kHighFrequencyCycles)));

  // Every channel recentred to a mean of 0.5, so the silhouette wobble, the interior texture and the de-tiling's
  // variance restoration share one centre. The means are the kernels' above, measured by Monte Carlo (the curl
  // warp preserves measure, so it cannot shift them); a change to a kernel needs them measured again.
  const vec4 rawChannelMean = vec4(0.7224f, 0.7267f, 0.1555f, 0.1542f);
  const vec4 raw = vec4(wispyLow, wispyHigh, billowyLow, billowyHigh);
  detailOutput[threadId] = saturate(raw + (vec4(0.5f) - rawChannelMean));
}
