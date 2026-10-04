/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

// Multiple scattering LUT: Hillaire's Psi_ms (EGSR 2020 Section 5.5), made direction aware.
//
// One workgroup per (sun zenith angle cosine, altitude) texel integrates the radiance arriving at the texel's
// sample point from a grid of directions and writes what the atmosphere marches sample for each species: the
// field's second moments for Rayleigh scattering and its phase weighted average about each view direction for
// the aerosol (layout in atmosphere_args.h). Every order past the first arrives far more from the horizon and
// the sun than from the zenith, so the light scattered toward a viewer depends on the view direction. The bake
// runs in passes, each reading the previous one's result:
//   isotropic           the paper's estimate: the second order from the real phase functions and every order
//                       beyond it through the series F_ms = 1 / (1 - f_ms) of Equations 9-10, with the ground's
//                       reflection counted in f_ms;
//   from isotropic and  the radiance of every order arriving from each direction: the sun scattered once at the
//   from directional    remote point, the remote point's own multiple scattering from the previous pass and the
//                       ground lit by the sun and the sky.
//
// Psi_ms has units of sr^-1: multiplying it by a directional light's illuminance and the local sigma_s yields that
// light's complete multiple scattering contribution (Equation 11). Neither the illuminance nor sigma_s is baked in.

#include "rtx/pass/atmosphere/atmosphere_common.slangh"

layout(binding = 0)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

layout(binding = 1)
Texture2D<float4> transmittanceLutInput;

layout(binding = 2)
Texture2D<float4> aerosolPhaseLutInput;

layout(binding = 3)
Texture2D<float4> multiscatteringInput;

layout(binding = 4, rgba16f)
RWTexture2D<float4> multiscatteringOutput;

layout(push_constant)
ConstantBuffer<AtmosphereMultiscatteringBakeArgs> bakeArgs;

// Directions uniform in zenith cosine and in azimuth from the sun. The field is symmetric about the sun's vertical
// plane, so each thread marches one direction on the positive z side, which also stands for its mirror image.
#define ZENITH_COUNT 16
#define AZIMUTH_COUNT 32
#define THREAD_COUNT (ZENITH_COUNT * AZIMUTH_COUNT / 2)
static const float kDirectionSolidAngle = 4.0f * kPi / float(ZENITH_COUNT * AZIMUTH_COUNT);
static const int kMinMarchStepCount = 4;
static const int kMaxMarchStepCount = 128;

groupshared vec3 gDirection[THREAD_COUNT];
groupshared vec3 gLuminance[THREAD_COUNT];
// The isotropic pass's transfer factor; the later passes' throughput to the ground.
groupshared vec3 gTransfer[THREAD_COUNT];
// Upward radiance on the ground beneath the sample, cosine weighted. The first half of the grid points up.
groupshared vec3 gSkyIrradiance[THREAD_COUNT / 2];
groupshared vec3 gGroundIrradiance;

vec3 gridDirection(uint index) {
  const uint zenith = index / (AZIMUTH_COUNT / 2);
  const uint azimuth = index % (AZIMUTH_COUNT / 2);
  const float cosZenith = 1.0f - 2.0f * (float(zenith) + 0.5f) / float(ZENITH_COUNT);
  const float sinZenith = sqrt(max(0.0f, 1.0f - cosZenith * cosZenith));
  const float phi = 2.0f * kPi * (float(azimuth) + 0.5f) / float(AZIMUTH_COUNT);
  return vec3(sinZenith * cos(phi), cosZenith, sinZenith * sin(phi));
}

vec3 mirrorDirection(vec3 direction) {
  return vec3(direction.x, direction.y, -direction.z);
}

// View direction a texel of the aerosol tiles stands for, with the sun at azimuth zero. Inverse of the lookup in
// evalMultipleScatteringPsi().
vec3 aerosolTileViewDirection(uint viewZenithIndex, uint azimuthIndex) {
  const float s = 2.0f * float(viewZenithIndex) / float(ATMOSPHERE_MS_VIEW_ZENITH_COUNT - 1) - 1.0f;
  const float cosZenith = s * abs(s);
  const float sinZenith = sqrt(max(0.0f, 1.0f - cosZenith * cosZenith));
  const float phi = kPi * float(azimuthIndex) / float(ATMOSPHERE_MS_RELATIVE_AZIMUTH_COUNT - 1);
  return vec3(sinZenith * cos(phi), cosZenith, sinZenith * sin(phi));
}

struct FieldSample {
  vec3 luminance;         // Radiance arriving against the direction, per unit sun illuminance
  vec3 transfer;          // Fraction of the light leaving along the direction that interacts again
  vec3 groundThroughput;  // Transmittance to the ground, zero when the direction misses it
};

// Radiance arriving at samplePos from the rayDir side. The ground's response to the sky is left to the caller,
// which only knows the sky irradiance once every direction is marched.
FieldSample integrateField(vec3 samplePos, vec3 rayDir, vec3 sunDir, uint passIndex, AtmosphereArgs args) {
  FieldSample result;
  result.luminance = vec3(0.0f);
  result.transfer = vec3(0.0f);
  result.groundThroughput = vec3(0.0f);

  AtmosphereMarchSegment segment;
  if (!clipAtmosphereMarch(samplePos, rayDir, args, 0.0f, 0.0f, segment)) {
    return result;
  }

  const vec3 planetCenter = getPlanetCenter(args);
  const float cosTheta = dot(rayDir, sunDir);
  const float phaseR = rayleighPhase(cosTheta, args.rayleighPhaseGamma);
  const vec3 phaseM = miePhase(cosTheta, args, aerosolPhaseLutInput);

  const int stepCount = clamp(int(args.multiscatteringStepCount), kMinMarchStepCount, kMaxMarchStepCount);
  const float invSteps = 1.0f / float(stepCount);
  const float span = segment.t1 - segment.t0;
  vec3 throughput = vec3(1.0f);
  float segmentStart = 0.0f;

  for (int i = 0; i < stepCount; ++i) {
    const float segmentEnd = atmosphereStepFraction(float(i + 1) * invSteps);
    const float dt = (segmentEnd - segmentStart) * span;
    const float t = segment.t0 + (segmentStart + (segmentEnd - segmentStart) * 0.5f) * span;
    segmentStart = segmentEnd;

    if (dt <= 0.0f) {
      continue;
    }

    const vec3 marchPos = samplePos + rayDir * t;
    const vec3 toMarchPos = marchPos - planetCenter;
    const float radius = max(length(toMarchPos), kAtmosphereEpsilon);
    const float altitude = clamp(radius - args.planetRadius, 0.0f, args.atmosphereThickness);
    const AtmosphereMedium medium = sampleAtmosphereMedium(altitude, args);

    // The sun scattered once at the remote point toward the sample, then the remote point's own multiple
    // scattering, which leaves it against rayDir as seen from the sample.
    const vec3 sunShadowing = evalSunShadowing(marchPos, sunDir, transmittanceLutInput, args);
    vec3 source = sunShadowing * (medium.rayleighScattering * phaseR + medium.mieScattering * phaseM);

    const vec3 up = toMarchPos / radius;
    if (passIndex == ATMOSPHERE_MS_PASS_FROM_ISOTROPIC) {
      source += medium.scattering * sampleMultiscatteringIsotropic(multiscatteringInput, altitude, dot(up, sunDir), args);
    } else if (passIndex == ATMOSPHERE_MS_PASS_FROM_DIRECTIONAL) {
      vec3 rayleighPsi;
      vec3 aerosolPsi;
      evalMultipleScatteringPsi(multiscatteringInput, altitude, up, rayDir, sunDir, args, rayleighPsi, aerosolPsi);
      source += medium.rayleighScattering * rayleighPsi + medium.mieScattering * aerosolPsi;
    }

    // Energy conserving analytical integration (Equation 17).
    const vec3 stepTransmittance = clamp(exp(-medium.extinction * dt), vec3(0.0f), vec3(1.0f));
    const vec3 weight = throughput * (vec3(1.0f) - stepTransmittance) / max(medium.extinction, vec3(1e-7f));
    result.luminance += weight * source;
    result.transfer += weight * medium.scattering;
    throughput *= stepTransmittance;
  }

  // The ground's diffuse response to the sun (the L_o term of Equation 6). Its reflection is one more interaction
  // for the transfer factor too.
  if (segment.hitsGround) {
    const vec3 groundPos = samplePos + rayDir * segment.t1;
    const float NdotL = dot(normalize(groundPos - planetCenter), sunDir);
    if (NdotL > 0.0f) {
      const vec3 sunShadowing = evalSunShadowing(groundPos, sunDir, transmittanceLutInput, args);
      result.luminance += throughput * sunShadowing * NdotL * (args.groundAlbedo / kPi);
    }
    result.transfer += throughput * args.groundAlbedo;
    result.groundThroughput = throughput;
  }

  return result;
}

int2 atlasTexel(int2 tile, uint2 texel, AtmosphereArgs args) {
  return tile * int(args.multiscatteringLutSize) + int2(texel);
}

[shader("compute")]
[numthreads(THREAD_COUNT, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint threadIndex : SV_GroupIndex) {
  const AtmosphereArgs args = atmosphereArgs;
  const uint passIndex = bakeArgs.passIndex;
  // Parameterization matching multiscatteringTileCoordinate().
  const uint2 texel = groupId.xy;
  const vec2 uv = (vec2(texel) + 0.5f) / float(args.multiscatteringLutSize);
  const float sunZenithCosAngle = textureCoordToUnitRange(uv.x, args.multiscatteringLutSize) * 2.0f - 1.0f;
  const float altitude = multiscatteringLutVToAltitude(textureCoordToUnitRange(uv.y, args.multiscatteringLutSize), args);
  const vec3 samplePos = vec3(0.0f, clamp(altitude, kPlanetRadiusOffset, args.atmosphereThickness), 0.0f);
  const float sunZenithSinAngle = sqrt(max(0.0f, 1.0f - sunZenithCosAngle * sunZenithCosAngle));
  const vec3 sunDir = normalize(vec3(sunZenithSinAngle, sunZenithCosAngle, 0.0f));

  const vec3 direction = gridDirection(threadIndex);
  const FieldSample field = integrateField(samplePos, direction, sunDir, passIndex, args);
  gDirection[threadIndex] = direction;
  gLuminance[threadIndex] = field.luminance;
  gTransfer[threadIndex] = passIndex == ATMOSPHERE_MS_PASS_ISOTROPIC ? field.transfer : field.groundThroughput;

  if (passIndex != ATMOSPHERE_MS_PASS_ISOTROPIC && threadIndex < THREAD_COUNT / 2) {
    const vec3 groundSamplePos = vec3(0.0f, kPlanetRadiusOffset, 0.0f);
    gSkyIrradiance[threadIndex] = integrateField(groundSamplePos, direction, sunDir, passIndex, args).luminance * direction.y;
  }
  GroupMemoryBarrierWithGroupSync();

  if (passIndex == ATMOSPHERE_MS_PASS_ISOTROPIC) {
    if (threadIndex == 0) {
      // Every direction stands for its mirror too, so the means over the half are those over the sphere.
      vec3 luminanceSum = vec3(0.0f);
      vec3 transferSum = vec3(0.0f);
      for (uint i = 0; i < THREAD_COUNT; ++i) {
        luminanceSum += gLuminance[i];
        transferSum += gTransfer[i];
      }
      // Equations 9 and 10. f_ms stays below one for any physical medium; the clamp guards pathological setups.
      const vec3 transfer = clamp(transferSum / float(THREAD_COUNT), vec3(0.0f), vec3(0.999f));
      const vec3 psi = luminanceSum / float(THREAD_COUNT) / (vec3(1.0f) - transfer);
      multiscatteringOutput[atlasTexel(int2(0, 0), texel, args)] = float4(max(psi, vec3(0.0f)), 1.0f);
    }
    return;
  }

  if (threadIndex == 0) {
    vec3 irradiance = vec3(0.0f);
    for (uint i = 0; i < THREAD_COUNT / 2; ++i) {
      irradiance += gSkyIrradiance[i];
    }
    gGroundIrradiance = irradiance * (2.0f * kDirectionSolidAngle);
  }
  GroupMemoryBarrierWithGroupSync();

  gLuminance[threadIndex] += gTransfer[threadIndex] * (args.groundAlbedo / kPi) * gGroundIrradiance;
  GroupMemoryBarrierWithGroupSync();

  const uint aerosolTexelCount = ATMOSPHERE_MS_VIEW_ZENITH_COUNT * ATMOSPHERE_MS_RELATIVE_AZIMUTH_COUNT;
  if (threadIndex < aerosolTexelCount) {
    // The field weighted by the aerosol phase function about the view direction. The weights are renormalised so
    // the forward peak, narrower than the grid, conserves energy.
    const uint viewZenithIndex = threadIndex / ATMOSPHERE_MS_RELATIVE_AZIMUTH_COUNT;
    const uint azimuthIndex = threadIndex % ATMOSPHERE_MS_RELATIVE_AZIMUTH_COUNT;
    const vec3 viewDir = aerosolTileViewDirection(viewZenithIndex, azimuthIndex);
    vec3 weighted = vec3(0.0f);
    vec3 weightSum = vec3(0.0f);
    for (uint i = 0; i < THREAD_COUNT; ++i) {
      const vec3 d = gDirection[i];
      const vec3 weight = miePhase(dot(d, viewDir), args, aerosolPhaseLutInput)
                        + miePhase(dot(mirrorDirection(d), viewDir), args, aerosolPhaseLutInput);
      weighted += weight * gLuminance[i];
      weightSum += weight;
    }
    multiscatteringOutput[atlasTexel(int2(viewZenithIndex, 1 + azimuthIndex), texel, args)] =
      float4(weighted / max(weightSum, vec3(1e-12f)), 1.0f);
  } else if (threadIndex < aerosolTexelCount + 3) {
    // The field's second moments in the frame (toward the sun, across, vertical), for one channel's Rayleigh
    // response; mirror images share them.
    const uint channel = threadIndex - aerosolTexelCount;
    vec4 moments = vec4(0.0f);
    for (uint i = 0; i < THREAD_COUNT; ++i) {
      const vec3 d = gDirection[i];
      moments += gLuminance[i][channel] * vec4(d.x * d.x, d.z * d.z, d.y * d.y, d.x * d.y);
    }
    multiscatteringOutput[atlasTexel(int2(1 + channel, 0), texel, args)] = moments * (2.0f * kDirectionSolidAngle);
  }
}
