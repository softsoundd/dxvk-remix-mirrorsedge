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
#include "rtx/pass/tonemap/psycho17.slangh"

layout(binding = AUTO_EXPOSURE_HISTOGRAM_INPUT_OUTPUT)
RWTexture1D<uint> OutHistogram;

layout(binding = AUTO_EXPOSURE_COLOR_INPUT)
RWTexture2D<float4> InColorBuffer;

layout(push_constant)
ConstantBuffer<ToneMappingAutoExposureArgs> cb;

groupshared uint g_localData[EXPOSURE_HISTOGRAM_SIZE];

// Map per-pixel CIE 170-2 luminosity Yf to a log-Yf histogram bucket.
//
// Switched from BT.709 photometric luminance to Stockman-Sharpe Yf
// (PSYCHO17_MB_WEIGHTS . LMS, exposed as renodx::yf::from_BT709) so the
// auto-exposure stage and the psycho17 observer model share a single
// physiological achromatic measure. Exposure scale and any downstream
// adaptive state therefore live in the same space.
//
//   Bucket 0       => "near-black" pixels, excluded from the average.
//   Buckets 1..N-1 => log2-Yf evenly spaced over
//                     [kAutoExposureLogLumMin, kAutoExposureLogLumMax].
// The bin helpers in tonemapping.slangh are unit-agnostic, so re-using
// them for Yf is correct.
//
// Yf is rescaled by Yf-of-BT.709-white so that 18% linear reflectance
// bins at Yf == 0.18 exactly — making `rtx.autoExposure.targetAdaptedYf
// = 0.18` actually mean "18% mid-gray" as documented. Without this
// normalisation 0.18 reflectance would land at ~0.189 cone-Yf with the
// Stockman-Sharpe MB weights, and the target would silently drift dark.
// `histogramBinToLuminance` in the resolve returns bin centres in this
// same normalised space, so the symmetry holds end-to-end.
static const float kYfRefWhite = 1.05040443f; // renodx::tonemap::psycho::yf::from_BT709(1.0) precomputed.

uint inputToHistogramBucket(const float3 inputColor)
{
  const float yf = renodx::tonemap::psycho::yf::from_BT709(inputColor) * (1.0f / kYfRefWhite);

  // Negative or NaN inputs (e.g. denoiser ringing) collapse to bin 0
  // via the !(yf >= eps) test so they are excluded from the resolve,
  // matching the original near-black handling.
  const float kEpsilon = 1e-5f;
  if (!(yf >= kEpsilon))
  {
    return 0;
  }

  return luminanceToHistogramBin(yf);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 threadId : SV_DispatchThreadID, uint linearIndex : SV_GroupIndex)
{
  g_localData[linearIndex] = 0;

  GroupMemoryBarrierWithGroupSync();

  uint2 dimensions;
  InColorBuffer.GetDimensions(dimensions.x, dimensions.y);

  if (all(threadId < dimensions))
  {
    const float3 inputColor = InColorBuffer[threadId].xyz;
    const uint bucketIdx = inputToHistogramBucket(inputColor);

    // Gaussian center-weighted metering. Pixels near the screen
    // centre contribute more to the auto-exposure measurement than
    // pixels at the edges — matching the "center-weighted" metering
    // pattern of real cameras and giving stable, intuitive
    // exposure when the player is looking around a scene with
    // bright sky / dark indoors at the periphery.
    //
    // Distances are normalised by the SHORTER screen axis (height
    // on widescreen), so the Gaussian is circular in pixel space
    // and the horizontal periphery on a 16:9 / 21:9 frame falls
    // off properly instead of contributing the same as the vertical
    // periphery. At sigma = 0.25 of the short axis:
    //   * vertical edge midpoint  (|dy| = 0.5)   -> exp(-2)  ~= 0.135
    //   * horizontal edge midpoint on 16:9       -> exp(-6.3)~= 0.002
    //   * horizontal edge midpoint on 21:9       -> exp(-12) ~= 6e-6
    const float invScale = 1.0f / float(min(dimensions.x, dimensions.y));
    const float2 delta   = (float2(threadId) + 0.5f - 0.5f * float2(dimensions)) * invScale;
    const float  rSq     = dot(delta, delta);
    const float  kSigma  = 0.25f;
    const float  gauss   = exp(-0.5f * rSq / (kSigma * kSigma));

    // Fixed-point weight (scale 256). The resolve pass divides
    // sum(weight * logYf) by sum(weight) so the absolute scale cancels.
    // Any pixel that contributes at all gets at least 1 count, so
    // the dimmest periphery still nudges the histogram.
    const uint weight = max(uint(gauss * 256.0f + 0.5f), 1u);
    InterlockedAdd(g_localData[bucketIdx], weight);
  }

  GroupMemoryBarrierWithGroupSync();

  uint localBinValue = g_localData[linearIndex];
  if (localBinValue != 0)
  {
    InterlockedAdd(OutHistogram[linearIndex], localBinValue);
  }
}
