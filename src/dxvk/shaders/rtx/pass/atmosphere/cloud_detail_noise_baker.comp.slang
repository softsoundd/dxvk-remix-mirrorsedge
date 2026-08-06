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
 * Nubis3 detail-noise bake (fork — Nubis3 conversion Phase B; noise-character
 * rewrite 2026-07-16).
 *
 * Fills the 128^3 RGBA8 wispy/billowy detail volume the Nubis3 up-rez
 * function composites into its value-erosion field (Nubis Cubed 2023,
 * Sampling Density section, p.95-99):
 *
 *   R — low-frequency WISPY   ("Curly-Alligator": curl-warped inverted
 *                              alligator — web-like strands with curly
 *                              distortion).
 *   G — high-frequency WISPY  (same at higher frequency, decorrelated seed).
 *   B — low-frequency BILLOWY ("Alligator": layered rounded lumps separated
 *                              by creases).
 *   A — high-frequency BILLOWY (same at higher frequency).
 *
 * Noise character matters (rev 2): the first bake used Perlin FBM (wispy)
 * and inverted Worley (billowy) — the EXACT noises the Nubis3 talk calls out
 * as insufficient: plain inverted Worley reads as "packed spheres" and
 * Perlin(-Worley) is "not purely wispy enough". Their replacements are
 * Houdini's Alligator noise and a curl-distorted inversion of it. The
 * alligator3DPeriodic below is a sparse-bump approximation of Alligator:
 * one jittered feature point per lattice cell with a random amplitude, value
 * = strongest kernel minus runner-up — amplitude-layered lumps with dark
 * creases where kernels contest, instead of equal-sized packed spheres.
 *
 * The sampler blends R->G and B->A over the dimensional profile (deeper into
 * the body = finer structure) and wispy->billowy over the cloud type — see
 * sampleCloudDensityNubis3 in cloud_nubis3_common.slangh.
 *
 * The volume is periodic in its own [0,1)^3 domain (all lattice periods are
 * integers in cycles-per-volume, and the curl-warp potential field is itself
 * periodic), so the runtime's frac()-wrapped sample at any frequency scale
 * tiles seamlessly. It is a FIXED pattern — no live bake inputs — baked once
 * at init (cost is irrelevant; nothing here runs per frame). The legacy
 * 256^3 R8 volume is untouched (the legacy density path keeps reading it for
 * the A/B toggle).
 *
 * Dispatch: 8x8x8 threads, (16, 16, 16) groups -> 128^3 voxels.
 */

#include "rtx/pass/atmosphere/atmosphere_common.slangh"

layout(binding = 0)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

layout(binding = 1, rgba8)
RWTexture3D<float4> cloudDetailNoiseOutput;

// Keep in lockstep with RtxAtmosphere::kCloudDetailNoise3DSize.
static const float kDetailVolumeSize = 128.0f;

// Lattice frequencies in integer cycles-per-volume so every channel tiles.
//
// Frequency choice (anti-blobby pass, 2026-07-16 — was 4/8 with 3 octaves):
// Nubis's own detail texture repeats every ~100 m of world (sample pos
// * 0.01), so ALL of its erosion content is sub-100 m — the crispness lives
// there. Our volume repeats every cloudNoiseTileKm / cloudDetailScale
// (~2.8 km at defaults), so 4/8 cycles put the erosion at 700/350 m: gentle
// undulation, not cauliflower. 6/16 cycles + octave stacks give
// 465..116 m (wispy) and 174..87 m (billowy-hi) content near the 128-texel
// Nyquist. The overall scale stays live-tunable via cloudDetailScale.
static const float kLowFreqCycles  = 6.0f;
static const float kHighFreqCycles = 16.0f;

// ---------------------------------------------------------------------------
// Alligator-noise approximation (see file header). p is in lattice-cell
// units; period (integer per axis) wraps the CELL IDS so the field tiles.
// Value = strongest sparse bump minus the runner-up: amplitude-layered lumps
// with creases, in ~[0, 1).
// ---------------------------------------------------------------------------
float alligator3DPeriodic(vec3 p, float seed, vec3 period) {
  vec3 cell = floor(p);
  vec3 f = p - cell;
  float m1 = 0.0f;
  float m2 = 0.0f;
  [unroll]
  for (int dz = -1; dz <= 1; ++dz) {
    [unroll]
    for (int dy = -1; dy <= 1; ++dy) {
      [unroll]
      for (int dx = -1; dx <= 1; ++dx) {
        vec3 offs = vec3(float(dx), float(dy), float(dz));
        vec3 cw = cell + offs;
        cw -= period * floor(cw / period);  // positive wrap of the cell id
        // Feature point + amplitude from three decorrelated hashes.
        vec3 feat = vec3(hash31(cw + vec3(seed + 11.317f, seed, seed)),
                         hash31(cw + vec3(seed, seed + 47.113f, seed)),
                         hash31(cw + vec3(seed, seed, seed + 73.951f)));
        // Wide amplitude range = lump-size variety (uniform lumps are the
        // strongest "synthetic" tell). Softened kernel exponent for gentler
        // shoulders (was 2.0 — read as hard plateaus).
        float amp = 0.25f + 0.75f * hash31(cw + vec3(seed + 5.771f, seed + 9.043f, seed + 1.229f));
        float d = length(offs + feat - f);
        // Smooth bump with ~1.4-cell reach so neighboring kernels contest.
        float k = amp * pow(saturate(1.0f - d * 0.7071f), 1.6f);
        m2 = max(m2, min(m1, k));  // runner-up before m1 updates
        m1 = max(m1, k);
      }
    }
  }
  // Gain stretches the typical contrast range toward [0,1] (heuristic; the
  // erosion / interior-texture knobs absorb residual calibration). 1.25
  // (was 1.8): the harder gain saturated lumps into flat plateaus with
  // razor creases — the "intense/fake" read.
  return saturate((m1 - m2) * 1.25f);
}

float alligatorFbm3DPeriodic(vec3 p, int octaves, float seed, vec3 basePeriod) {
  float value = 0.0f;
  float amplitude = 0.5f;
  float total = 0.0f;
  vec3 pp = p;
  vec3 period = basePeriod;
  for (int i = 0; i < octaves; ++i) {
    value += amplitude * alligator3DPeriodic(pp, seed + 13.7f * float(i), period);
    total += amplitude;
    amplitude *= 0.5f;
    pp *= 2.0f;
    period *= 2.0f;
  }
  return value / max(total, 1e-4f);
}

// Central difference of one periodic Perlin FBM potential along one axis.
float cloudCurlPotentialDiff(vec3 p, vec3 axis, float seed, vec3 period) {
  const float e = 0.25f;
  return (fbmNoise3DPeriodic(p + axis * e, 2, seed, period)
        - fbmNoise3DPeriodic(p - axis * e, 2, seed, period)) / (2.0f * e);
}

// Periodic curl field for the wispy channels' domain warp: curl of a vector
// potential (A, B, C) built from three decorrelated periodic Perlin FBMs —
// divergence-free by construction, periodic because the potentials are.
// One-shot bake: the 12 FBM evals per voxel are irrelevant.
vec3 curlNoise3DPeriodic(vec3 p, float seed, vec3 period) {
  const float sA = 12.9f;
  const float sB = 31.7f;
  const float sC = 67.1f;
  const vec3 ex = vec3(1, 0, 0);
  const vec3 ey = vec3(0, 1, 0);
  const vec3 ez = vec3(0, 0, 1);
  float dC_dy = cloudCurlPotentialDiff(p, ey, seed + sC, period);
  float dB_dz = cloudCurlPotentialDiff(p, ez, seed + sB, period);
  float dA_dz = cloudCurlPotentialDiff(p, ez, seed + sA, period);
  float dC_dx = cloudCurlPotentialDiff(p, ex, seed + sC, period);
  float dB_dx = cloudCurlPotentialDiff(p, ex, seed + sB, period);
  float dA_dy = cloudCurlPotentialDiff(p, ey, seed + sA, period);
  return vec3(dC_dy - dB_dz, dA_dz - dC_dx, dB_dx - dA_dy);
}

[shader("compute")]
[numthreads(8, 8, 8)]
void main(uint3 threadId : SV_DispatchThreadID) {
  if (any(threadId >= uint3(128u, 128u, 128u))) {
    return;
  }

  // Voxel center in the volume's own [0,1) periodic domain.
  vec3 p = (vec3(threadId) + 0.5f) / kDetailVolumeSize;

  const vec3 loPeriod = vec3(kLowFreqCycles);
  const vec3 hiPeriod = vec3(kHighFreqCycles);

  // Wispy: "Curly-Alligator" — INVERTED alligator (bright web-like strands
  // where kernels contest) sampled through a curl-noise domain warp so the
  // strands pick up curly distortion (Nubis p.95: wisps form where density
  // decreases and the surrounding air billows INTO the vapor). The warp
  // amplitude is in lattice cells; the curl field is periodic at the same
  // lattice, so the warped sample still tiles.
  //
  // Anisotropy (edge-wisp-cut pass, 2026-07-16): wispy features are
  // stretched 2x horizontally (doubled VERTICAL cycle count = features half
  // as tall as they are wide — the volume's y axis is world-vertical), so
  // the edge-erosion cuts read as sheared trailing streaks instead of round
  // webs. Integer per-axis periods keep the tiling exact.
  const vec3 kWispSqueeze = vec3(1.0f, 2.0f, 1.0f);
  const vec3 loPeriodW = loPeriod * kWispSqueeze;
  const vec3 hiPeriodW = hiPeriod * kWispSqueeze;
  const float kCurlWarpCells = 0.55f;
  vec3 warpLo = curlNoise3DPeriodic(p * kLowFreqCycles * kWispSqueeze, 101.3f, loPeriodW) * kCurlWarpCells;
  vec3 warpHi = curlNoise3DPeriodic(p * kHighFreqCycles * kWispSqueeze, 227.9f, hiPeriodW) * kCurlWarpCells;
  // The pow(2) deepens the web contrast (the mean recentering happens in the
  // normalization below — the old claim that the pow pulled the mean to ~0.5
  // was wrong; measured, the shaped wispy channels run BRIGHT, ~0.72).
  float wispyLo = pow(saturate(1.0f - alligatorFbm3DPeriodic(p * kLowFreqCycles * kWispSqueeze + warpLo, 3, 23.1f, loPeriodW)), 2.0f);
  float wispyHi = pow(saturate(1.0f - alligatorFbm3DPeriodic(p * kHighFreqCycles * kWispSqueeze + warpHi, 2, 41.7f, hiPeriodW)), 2.0f);

  // Billowy: alligator FBM straight — amplitude-layered rounded lumps with
  // creases ("layered billows"), replacing the packed-sphere read of the old
  // inverted-Worley channel.
  float billowyLo = saturate(alligatorFbm3DPeriodic(p * kLowFreqCycles,  3, 7.9f,  loPeriod));
  float billowyHi = saturate(alligatorFbm3DPeriodic(p * kHighFreqCycles, 2, 13.3f, hiPeriod));

  // Zero-mean channel normalization (GT7 p.228; fork — detail round
  // 2026-07-16). Recenter every channel to mean 0.5 AT BAKE TIME so all
  // sampler consumers (the zero-mean silhouette wobble, the interior-texture
  // density factor, the two-scale variance re-expansion) share one true
  // center — GT7: "we normalized the texture so that the average value is
  // zero [so] the amount of clouds does not vary with the intensity of the
  // noise". Before this, the wobble pushed wispy silhouettes outward
  // (~+0.22 x amplitude) and billowy inward (~-0.10), and the interior
  // texture hid a large density cut on billowy cores. Raw means MEASURED via
  // a lattice Monte-Carlo port of the kernel math above (scratchpad harness
  // measure_channel_means.py, 2026-07-16; the curl warp is divergence-free =
  // measure-preserving, so skipping it cannot shift the mean). Pure bias:
  // tails clip slightly at 0/1 and the sampler's variance re-expansion
  // absorbs the residual contrast change. RE-MEASURE if the kernel gain,
  // exponent, amp range, octave counts, or channel transforms change.
  const vec4 kRawChannelMean = vec4(0.7224f, 0.7267f, 0.1555f, 0.1542f);
  vec4 raw = vec4(wispyLo, wispyHi, billowyLo, billowyHi);
  cloudDetailNoiseOutput[threadId] = saturate(raw + (vec4(0.5f) - kRawChannelMean));
}
