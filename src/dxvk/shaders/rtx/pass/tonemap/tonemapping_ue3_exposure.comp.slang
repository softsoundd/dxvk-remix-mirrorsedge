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

// Mirror's Edge (TdToneMapExposure) auto exposure, run on Remix's linear radiance.
//
// The game meters a 1x1 arithmetic mean of its scene colour, produced by a fixed-point
// downsample chain that clamps every channel at 1.0 before averaging, takes its luminosity with
// the 0.3 / 0.59 / 0.11 weights, and exposes it to 0.25: sqrt(E) = clamp(sqrt(0.25 / L), Low,
// High). Low and High are authored per PostProcessVolume and are what places a bright exterior
// at its floor and lets an interior rise to its ceiling; they arrive here from the live capture
// of the game's (skipped) exposure pass. Remix's radiance is scaled into the game's scene units
// by one calibration factor first.
//
// The meter, key and clamps are the shipped ones. The adaptation is Faithful Luma's: movement in
// stops at a fixed speed per second, easing in exponentially inside transitionStops of the
// target. A cleared state (0) or a reset starts on the target.

#include "rtx/pass/tonemap/tonemapping_ue3_exposure.h"
#include "rtx/utility/math.slangh"

[[vk::binding(TONEMAPPING_UE3_EXPOSURE_COLOR_INPUT)]]
RWTexture2D<float4> InColorBuffer;

[[vk::binding(TONEMAPPING_UE3_EXPOSURE_OUTPUT)]]
RWTexture1D<float> InOutExposure;

layout(push_constant)
ConstantBuffer<ToneMappingUe3ExposureArgs> cb;

static const uint kThreadCount = UE3_EXPOSURE_METER_THREADS_PER_AXIS * UE3_EXPOSURE_METER_THREADS_PER_AXIS;
static const float kSampleCount = float(UE3_EXPOSURE_METER_GRID * UE3_EXPOSURE_METER_GRID);
static const float3 kShippedLuminosityWeights = float3(0.3f, 0.59f, 0.11f);

groupshared float3 g_sum[kThreadCount];

[shader("compute")]
[numthreads(UE3_EXPOSURE_METER_THREADS_PER_AXIS, UE3_EXPOSURE_METER_THREADS_PER_AXIS, 1)]
void main(uint2 threadId : SV_GroupThreadID, uint linearIndex : SV_GroupIndex)
{
  uint2 dims;
  InColorBuffer.GetDimensions(dims.x, dims.y);
  const float2 dimsF = float2(max(dims, uint2(1, 1)));

  // Each thread averages a 16x16 block of a 256x256 grid of sample points laid over the image.
  float3 sum = float3(0.0f);
  for (uint j = 0; j < UE3_EXPOSURE_METER_SAMPLES_PER_THREAD; ++j)
  {
    for (uint i = 0; i < UE3_EXPOSURE_METER_SAMPLES_PER_THREAD; ++i)
    {
      const uint2 gridPos = threadId * UE3_EXPOSURE_METER_SAMPLES_PER_THREAD + uint2(i, j);
      const float2 uv = (float2(gridPos) + 0.5f) / float(UE3_EXPOSURE_METER_GRID);
      const uint2 pixel = min(uint2(uv * dimsF), dims - uint2(1, 1));

      float3 c = InColorBuffer[pixel].rgb * cb.sceneScale;
      c = (any(isnan(c)) || any(isinf(c))) ? float3(0.0f) : c;
      // The shipped downsample chain is fixed point: every channel is clamped at 1.0 before it
      // is averaged, so a bright source cannot dominate the mean.
      sum += saturate(c);
    }
  }

  g_sum[linearIndex] = sum;
  GroupMemoryBarrierWithGroupSync();

  for (uint stride = kThreadCount / 2; stride > 0; stride >>= 1)
  {
    if (linearIndex < stride)
    {
      g_sum[linearIndex] += g_sum[linearIndex + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }

  if (linearIndex != 0)
  {
    return;
  }

  const float3 mean = g_sum[0] / kSampleCount;
  float luminosity = dot(mean, kShippedLuminosityWeights);
  // The shipped pass replaces a NaN, and an exact black, mean with 0.25 (a target of E = 1).
  if (!(luminosity > 0.0f) && !(luminosity < 0.0f))
  {
    luminosity = 0.25f;
  }

  // Shipped key and clamps; the clamps apply to sqrt(E)
  const float targetSqrt = sqrt(0.25f / clamp(luminosity, 1e-7f, 5000.0f));
  const float low = max(cb.exposureLow, 1e-3f);
  const float high = max(cb.exposureHigh, low);
  const float clampedSqrt = clamp(targetSqrt, low, high);
  const float target = clampedSqrt * clampedSqrt;

  // The previous exposure is clamped into the current range as shipped, so entering an area
  // with a different range snaps into it and then adapts toward the target.
  const float previous = InOutExposure[0];
  float current = target;
  if (cb.resetHistory == 0 && previous > 0.0f)
  {
    current = clamp(previous, low * low, high * high);
  }

  // Faithful Luma adaptation: stops per second toward the target, easing in exponentially with
  // the same slope inside transitionStops. A zero speed (an authored fixed exposure) holds.
  const float dt = max(cb.deltaTimeSeconds, 0.0f);
  const bool toLight = target < current;
  const float speed = toLight ? cb.speedToLight : cb.speedToDark;
  const float gapStops = abs(log2(max(target, 1e-6f) / max(current, 1e-6f)));
  const float transition = max(cb.transitionStops, 1e-3f);
  const float linearStops = speed * dt;
  const float easedStops = gapStops * (1.0f - exp(-speed * dt / transition));
  const float stepStops = min((gapStops > transition) ? linearStops : easedStops, gapStops);
  const float direction = toLight ? -1.0f : 1.0f;

  InOutExposure[0] = current * exp2(direction * stepStops);
}
