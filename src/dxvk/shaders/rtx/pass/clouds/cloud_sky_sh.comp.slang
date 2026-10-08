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

// L2 spherical harmonics of the light reaching the cloud layer from everything but the sun's beam, at its base,
// middle and top: the sky (marched through the atmosphere from that altitude, so the sky above a cloud top is
// the darker, bluer sky of less air) and, below the horizon, the ground under the clouds. Upper and lower
// hemispheres project separately, since the cloud above and below a sample filter them differently.
//
// The ground's response is the atmosphere's diffuse planet ground, lit by the sun through the clouds' mean
// shadow over the sun grid's window plus the share that still arrives diffusely through them. The light is linear
// in that ground light, so the full bake (level 0) keeps the sky's harmonics and the ground's under unit light
// apart, and when only the grids changed (level 1) the ground light alone is re-weighted.
//
// One workgroup per altitude, each of its 256 threads integrating 4 of 1024 Fibonacci directions; the sums
// reduce across each wave, then across the waves.

#include "rtx/pass/clouds/cloud_binding_indices.h"
#include "rtx/pass/clouds/cloud_common.slangh"

layout(binding = CLOUD_BINDING_CONSTANTS)
ConstantBuffer<CloudConstants> constants;

layout(binding = CLOUD_BINDING_VOLUME_SAMPLER)
SamplerState volumeSampler;

layout(binding = CLOUD_BINDING_SUN_GRID)
Texture3D<float4> sunGrid;

layout(binding = CLOUD_BINDING_TRANSMITTANCE_LUT)
Texture2D transmittanceLut;

layout(binding = CLOUD_BINDING_MULTISCATTERING_LUT)
Texture2D multiscatteringLut;

layout(binding = CLOUD_BINDING_AEROSOL_PHASE_LUT)
Texture2D aerosolPhaseLut;

layout(binding = CLOUD_BINDING_BAKE_OUTPUT, rgba32f)
RWTexture2D<float4> shOutput;

// The sky's harmonics in shOutput's rows, then each altitude's ground under unit light (lower hemisphere only).
layout(binding = CLOUD_BINDING_BAKE_OUTPUT2, rgba32f)
RWTexture2D<float4> shParts;

layout(push_constant)
ConstantBuffer<CloudPassArgs> passArgs;

static const uint kThreadCount = 256u;
static const uint kDirectionsPerThread = 4u;
static const uint kDirectionCount = kThreadCount * kDirectionsPerThread;
// Waves of 32 lanes or more.
static const uint kMaxWaves = kThreadCount / 32u;
static const uint kCoefficients = CLOUD_SKY_SH_COEFFICIENTS;
// Both hemispheres' sky, then the ground's.
static const uint kSums = 3u * kCoefficients;

groupshared vec3 waveSums[kMaxWaves][kSums];
groupshared float waveGroundTransmittance[kMaxWaves];

vec3 fibonacciDirection(uint index, uint count) {
  const float goldenAngle = 2.39996323f;
  const float y = 1.0f - 2.0f * (float(index) + 0.5f) / float(count);
  const float r = sqrt(max(1.0f - y * y, 0.0f));
  const float phi = goldenAngle * float(index);
  return vec3(r * cos(phi), y, r * sin(phi));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID) {
  const CloudArgs cloud = constants.cloud;
  const AtmosphereArgs atmosphere = constants.atmosphere;
  const uint thread = groupThreadId.x;
  const uint altitudeIndex = groupId.x;
  const uint laneCount = WaveGetLaneCount();
  const uint wave = thread / laneCount;
  const uint waveCount = kThreadCount / laneCount;

  // The clouds' mean shadow on the ground: the sun grid's base row holds each column's whole optical depth.
  {
    const vec2 cell = (vec2(float(thread % 16u), float(thread / 16u)) + 0.5f) / 16.0f;
    const float od = cloud.enabled ? sunGrid.SampleLevel(volumeSampler, vec3(cell.x, 0.5f / float(CLOUD_GRID_SIZE_Y), cell.y), 0).x : 0.0f;
    const float waveSum = WaveActiveSum(exp(-od * cloud.shadowExtinctionKm));
    if (WaveIsFirstLane()) {
      waveGroundTransmittance[wave] = waveSum;
    }
  }
  GroupMemoryBarrierWithGroupSync();
  float groundTransmittance = 0.0f;
  for (uint i = 0u; i < waveCount; ++i) {
    groundTransmittance += waveGroundTransmittance[i];
  }
  groundTransmittance /= float(kThreadCount);
  const float groundLight = groundTransmittance + (1.0f - groundTransmittance) * saturate(cloud.groundDiffuseShare);

  const uint groundRow = 2u * CLOUD_SKY_SH_ALTITUDES + altitudeIndex;
  if (passArgs.level != 0u) {
    if (thread < 2u * kCoefficients) {
      const uint row = altitudeIndex * 2u + thread / kCoefficients;
      const uint coefficient = thread % kCoefficients;
      vec3 total = shParts[uint2(coefficient, row)].rgb;
      if (thread >= kCoefficients) {
        total += groundLight * shParts[uint2(coefficient, groundRow)].rgb;
      }
      shOutput[uint2(coefficient, row)] = vec4(total, 0.0f);
    }
    return;
  }

  const float altitudeKm = cloud.baseAltitudeKm + cloud.thicknessKm * float(altitudeIndex) / float(CLOUD_SKY_SH_ALTITUDES - 1);
  const vec3 origin = vec3(0.0f, max(altitudeKm, kPlanetRadiusOffset), 0.0f);
  const vec3 sunDir = normalize(atmosphere.sunDirection);
  const vec3 planetCenter = getPlanetCenter(atmosphere);

  vec3 sums[kSums];
  [unroll]
  for (uint m = 0u; m < kSums; ++m) {
    sums[m] = vec3(0.0f);
  }

  const float weight = 4.0f * kPi / float(kDirectionCount);
  for (uint j = 0u; j < kDirectionsPerThread; ++j) {
    const vec3 direction = fibonacciDirection(thread * kDirectionsPerThread + j, kDirectionCount);

    float tGround0, tGround1;
    const bool hitsGround = intersectSphere(origin, direction, planetCenter, atmosphere.planetRadius, tGround0, tGround1) && tGround0 > 0.0f;

    vec3 transmittance;
    const vec3 sky = evalAtmosphereRadianceSegment(
      origin, direction, sunDir, atmosphere, transmittanceLut, multiscatteringLut, aerosolPhaseLut, 24,
      hitsGround ? tGround0 : 0.0f, 0.0f, transmittance);

    // Only directions below the horizon reach the ground, so it lies in the lower hemisphere.
    vec3 ground = vec3(0.0f);
    if (hitsGround) {
      const vec3 groundPosition = origin + direction * tGround0;
      const vec3 groundNormal = normalize(groundPosition - planetCenter);
      const float cosSun = dot(groundNormal, sunDir);
      if (cosSun > 0.0f) {
        const vec3 sunShadowing = evalSunShadowing(groundPosition, sunDir, transmittanceLut, atmosphere);
        ground = transmittance * atmosphere.sunDiscIlluminance * sunShadowing * cosSun * (atmosphere.groundAlbedo / kPi)
          * cloud.groundBounceStrength;
      }
    }

    float basis[CLOUD_SKY_SH_COEFFICIENTS];
    cloudShBasis(direction, basis);
    const uint hemisphere = direction.y > 0.0f ? 0u : 1u;
    [unroll]
    for (uint m = 0u; m < kCoefficients; ++m) {
      sums[hemisphere * kCoefficients + m] += sky * (basis[m] * weight);
      sums[2u * kCoefficients + m] += ground * (basis[m] * weight);
    }
  }

  [unroll]
  for (uint m = 0u; m < kSums; ++m) {
    const vec3 waveSum = WaveActiveSum(sums[m]);
    if (WaveIsFirstLane()) {
      waveSums[wave][m] = waveSum;
    }
  }
  GroupMemoryBarrierWithGroupSync();

  if (thread < 2u * kCoefficients) {
    const uint row = altitudeIndex * 2u + thread / kCoefficients;
    const uint coefficient = thread % kCoefficients;
    vec3 sky = vec3(0.0f);
    vec3 ground = vec3(0.0f);
    for (uint i = 0u; i < waveCount; ++i) {
      sky += waveSums[i][thread];
      ground += waveSums[i][2u * kCoefficients + coefficient];
    }
    shParts[uint2(coefficient, row)] = vec4(sky, 0.0f);
    if (thread >= kCoefficients) {
      shParts[uint2(coefficient, groundRow)] = vec4(ground, 0.0f);
      sky += groundLight * ground;
    }
    shOutput[uint2(coefficient, row)] = vec4(sky, 0.0f);
  }
}
