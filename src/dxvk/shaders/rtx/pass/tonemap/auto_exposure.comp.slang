/*
* Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx/pass/tonemap/tonemapping.slangh"
#include "rtx/pass/tonemap/adaptation_v1.slangh"

layout(binding = AUTO_EXPOSURE_HISTOGRAM_INPUT_OUTPUT)
RWTexture1D<uint> InOutHistogram;

layout(binding = AUTO_EXPOSURE_EXPOSURE_INPUT_OUTPUT)
RWTexture1D<float> OutExposure;

layout(binding = AUTO_EXPOSURE_DEBUG_VIEW_OUTPUT)
RWTexture2D<float4> DebugView;

layout(push_constant)
ConstantBuffer<ToneMappingAutoExposureArgs> cb;

groupshared float g_sumLogYf[EXPOSURE_HISTOGRAM_SIZE];
groupshared float g_count[EXPOSURE_HISTOGRAM_SIZE];

// ---------------------------------------------------------------------------
// Perceptual auto-exposure (post Naka-Rushton).
//
// The histogram pass bins per-pixel Yf (Stockman-Sharpe CIE 170-2
// luminosity) into a log2-Yf histogram. This pass collapses that
// histogram into a single adaptation level and converts it to an
// exposure scale using a first-site cone-contrast law, matching the
// observer-model assumptions used by the psycho17 tonemap operator.
//
// Why this shape instead of the previous Naka-Rushton-in-resolve?
// - Spatial pooling for adaptation is geometric, not arithmetic. The
//   neural response to luminance is approximately logarithmic
//   (Stockman & Brainard 2010), so a log-mean of Yf is the appropriate
//   first-moment statistic. Center-weighted metering (a spatial
//   Gaussian on pixel position applied in the histogram pass) biases
//   bin counts toward the mid-tone subject the eye actually fixates
//   on rather than the unweighted scene mean.
// - The "what exposure scale do I apply" decision is a separate
//   question from the tonemap shoulder/toe shape. The previous shader
//   conflated the two by running Naka-Rushton at the resolve step. The
//   psycho17 operator already does Naka-Rushton in cone space; the
//   auto-exposure stage only needs to deliver a scene whose adapted
//   mean lands at a perceptual mid-gray. A bare first-site contrast
//   law (Y_target / (Y_adapt + Y0)) does exactly that and is the form
//   any non-Naka-Rushton operator also wants.
// - Yf is the operator's native achromatic axis, so the exposure scale
//   is computed in the same space as the cone responses that consume
//   it. No basis change between pipeline stages.
// ---------------------------------------------------------------------------

// Stockman-Brainard first-site cone-noise floor. Caps the dark-scene
// exposure boost so a near-black frame doesn't blow up to infinity:
//   exposure = targetAdaptedYf / (Y_adapt + Y_noise)
// With Y_noise = 0.0032 (~mid-mesopic), max boost ~= targetAdaptedYf/0.0032.
// Reference: Stockman & Brainard (2010), "Color Vision Mechanisms".
//
// adaptedYf comes from the histogram pass in Yf-normalised-to-white
// space (Yf rescaled so BT.709 (1,1,1) maps to ~1.0; see kYfRefWhite in
// auto_exposure_histogram.comp.slang). The Stockman 0.0032 reference is
// in raw cone-Yf, so applying it here is implicitly rescaled by 1/~1.05.
// The shift is ~5% and well below the explicit `rtx.autoExposure.maxExposure`
// cap that gates dark-scene boost in practice — left as-is so the
// citation is meaningful for retuning; do not chase the rescale.
static const float kConeNoiseFloorYf = 0.0032f;

// Secondary floor for the adapted Yf. The primary dark-scene guard is
// kConeNoiseFloorYf (0.0032), which dominates at all realistic adaptation
// levels — this only activates if exp2(sumLogYf / totalWeight) underflows
// below 1e-6 (e.g. heavily clipped histogram).
static const float kAdaptedYfFloor = 1e-6f;

[shader("compute")]
[numthreads(EXPOSURE_HISTOGRAM_SIZE, 1, 1)]
void main(uint2 threadId : SV_DispatchThreadID, uint linearIndex : SV_GroupIndex)
{
  // Per-bin contribution to a center-weighted log-Yf geometric mean.
  // Spatial Gaussian center-weighting is applied in the histogram pass,
  // so bin counts already reflect that weighting. Each bin contributes
  // its center Yf scaled by its (spatially-weighted) pixel count.
  //   sum_wLogYf = sum_i ( count_i * log2(Yf_i) )
  //   sum_w      = sum_i ( count_i )
  //   geomMean   = exp2( sum_wLogYf / sum_w )
  // Bin 0 is reserved for near-black pixels and excluded.
  const uint pixelCount = InOutHistogram[linearIndex];
  float sumWeightedLogYf = 0.0f;
  float sumWeight        = 0.0f;
  if (linearIndex >= 1 && pixelCount > 0)
  {
    const float binYf    = histogramBinToLuminance(linearIndex);
    const float logBinYf = log2(binYf);
    sumWeightedLogYf     = float(pixelCount) * logBinYf;
    sumWeight            = float(pixelCount);
  }
  g_sumLogYf[linearIndex] = sumWeightedLogYf;
  g_count[linearIndex]    = sumWeight;
  GroupMemoryBarrierWithGroupSync();

  // Parallel tree reduction across all EXPOSURE_HISTOGRAM_SIZE (power-of-two)
  // lanes. Bin 0 is excluded by the `linearIndex >= 1` guard above, so
  // summing the full array drops it implicitly.
  [unroll]
  for (uint stride = EXPOSURE_HISTOGRAM_SIZE / 2; stride > 0; stride >>= 1)
  {
    if (linearIndex < stride)
    {
      g_sumLogYf[linearIndex] += g_sumLogYf[linearIndex + stride];
      g_count[linearIndex]    += g_count[linearIndex + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }

  if (linearIndex == 0)
  {
    const float totalSumLogYf = g_sumLogYf[0];
    const float totalWeight   = g_count[0];

    // Center-weighted geometric mean of Yf. The spatial Gaussian from
    // the histogram pass already biases counts toward the mid-tone
    // subject at the screen centre; this reduces to a standard
    // count-weighted log mean.
    const float adaptedYf = (totalWeight > 0.0f)
        ? max(exp2(totalSumLogYf / totalWeight), kAdaptedYfFloor)
        : cb.targetAdaptedYf;  // Empty histogram => neutral exposure.

    const float rawTargetExposure = cb.targetAdaptedYf / (adaptedYf + kConeNoiseFloorYf);

    // Clamp the target before the temporal blend, not after. Clamping the
    // smoothed output would let the integrator wind up against the cap and
    // produce a sluggish recovery when the scene brightens again.
    const float targetExposure    = min(rawTargetExposure, cb.maxExposure);

    // Asymmetric exponential adaptation in log-exposure space. Human
    // eyes adapt faster to bright scenes (cone bleaching) than to dark
    // (rod recovery), so the time-constant flips with the direction of
    // change. Log-space blending makes the time-constant invariant to
    // absolute scene level — a 4-stop change adapts in the same wall
    // time whether it happens at noon or twilight.
    const float prevExposure    = max(OutExposure[0], 1e-6f);
    const float logPrev         = log2(prevExposure);
    const float logTarget       = log2(max(targetExposure, 1e-6f));
    const bool  sceneBrightened = (targetExposure < prevExposure);
    const float tau             = sceneBrightened ? cb.lightAdaptTau : cb.darkAdaptTau;
    const float logNew          = adaptation::v1::ExponentialBlend(
        logPrev, logTarget, cb.deltaTime, tau);

    OutExposure[0] = max(exp2(logNew), 1e-6f);
  }

  if (cb.debugMode)
  {
    // Debug view: spatially-weighted bin counts normalised by total pixel
    // count (R) and per-bin Yf in [0, 1] reflectance (B).
    const float binYf = (linearIndex >= 1) ? histogramBinToLuminance(linearIndex) : 0.0f;
    DebugView[uint2(linearIndex, 0)] = vec4(
        float(pixelCount) / float(cb.numPixels),
        0.0f,
        saturate(binYf),
        1.0f);
  }

  // Reset the histogram for the next frame.
  InOutHistogram[linearIndex] = 0;
}
