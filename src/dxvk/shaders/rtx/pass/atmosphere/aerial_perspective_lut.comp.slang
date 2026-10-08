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

// Aerial perspective volume (paper Section 5.4): per froxel slice, the in-scattered luminance toward the
// camera (RGB) and the mean transmittance (A). The march starts at aerialPerspectiveStartDistance, where
// Remix's global volumetrics froxel grid hands off.

#include "rtx/pass/atmosphere/atmosphere_common.slangh"
#include "rtx/pass/clouds/cloud_common.slangh"

layout(binding = 0)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

layout(binding = 1)
Texture2D<float4> transmittanceLutInput;

layout(binding = 2)
Texture2D<float4> multiscatteringLutInput;

layout(binding = 3, rgba16f)
RWTexture3D<float4> aerialPerspectiveLutOutput;

layout(binding = 4)
Texture2D<float> aerialPerspectiveTileDepthInput;

layout(binding = 5)
Texture2D<float4> aerosolPhaseLutInput;

// The cloud layer, whose shadow falls on the air beneath it (rtx_clouds.cpp): the leading CloudArgs of its
// constants buffer.
layout(binding = 6)
ConstantBuffer<CloudArgs> cloudConstants;

layout(binding = 7)
Texture3D<float4> cloudSunGrid;

layout(binding = 8)
Texture2D<float> cloudShadowMap;

layout(binding = 9)
SamplerState cloudSampler;

// evalAtmosphereRadianceSegment's march with the cloud layer's shadow on each step's sun.
vec3 evalCloudShadowedSegment(
  vec3 rayOrigin, vec3 rayDir, vec3 rayDirWorld, AtmosphereArgs args, CloudArgs cloud, float maxDistanceKm, float startDistanceKm,
  out vec3 outTransmittance)
{
  outTransmittance = vec3(1.0f);
  AtmosphereMarchSegment segment;
  if (!clipAtmosphereMarch(rayOrigin, rayDir, args, maxDistanceKm, startDistanceKm, segment)) {
    return vec3(0.0f);
  }

  const int stepCount = 12;
  const float span = segment.t1 - segment.t0;
  const float cosTheta = dot(rayDir, args.sunDirection);
  const float phaseR = rayleighPhase(cosTheta, args.rayleighPhaseGamma);
  const vec3 phaseM = miePhase(cosTheta, args, aerosolPhaseLutInput);

  vec3 inScatter = vec3(0.0f);
  float segmentStart = 0.0f;
  for (int i = 0; i < stepCount; ++i) {
    const float segmentEnd = atmosphereStepFraction(float(i + 1) / float(stepCount));
    const float dt = (segmentEnd - segmentStart) * span;
    const float t = segment.t0 + (segmentStart + (segmentEnd - segmentStart) * 0.5f) * span;
    segmentStart = segmentEnd;
    if (dt <= 0.0f) {
      continue;
    }

    const vec3 samplePos = rayOrigin + rayDir * t;
    vec3 sunShadowing = evalSunShadowing(samplePos, args.sunDirection, transmittanceLutInput, args);
    if (any(sunShadowing > vec3(0.0f))) {
      const vec3 worldPosition = args.cameraPosition + rayDirWorld * (t * args.worldUnitsPerKilometer);
      sunShadowing *= cloudSunTransmittance(cloudSunGrid, cloudShadowMap, cloudSampler, worldPosition, cloud, args);
    }

    vec3 stepTransmittance;
    const vec3 contribution = outTransmittance * integrateAtmosphereStep(
      samplePos, rayDir, dt, args.sunDirection, phaseR, phaseM, sunShadowing, vec3(1.0f), args, multiscatteringLutInput, stepTransmittance);
    if (all(isfinite(contribution))) {
      inScatter += contribution;
    }
    outTransmittance *= stepTransmittance;
  }
  return max(inScatter, vec3(0.0f));
}

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 threadId : SV_DispatchThreadID) {
  const AtmosphereArgs args = aerialPerspectiveMediumArgs(atmosphereArgs);

  const uint3 size = uint3(args.aerialPerspectiveLutSize, args.aerialPerspectiveLutSize, args.aerialPerspectiveLutDepth);
  if (any(threadId >= size)) {
    return;
  }

  const vec3 uvw = (vec3(threadId) + 0.5f) / vec3(size);

  const vec3 rayVectorWorld = aerialPerspectiveUvwToWorldPosition(uvw, args) - args.cameraPosition;
  const float rayLengthWorld = length(rayVectorWorld);
  if (rayLengthWorld <= 0.0f) {
    aerialPerspectiveLutOutput[threadId] = float4(0.0f, 0.0f, 0.0f, 1.0f);
    return;
  }

  const vec3 rayDirWorld = rayVectorWorld / rayLengthWorld;
  const vec3 rayDir = aerialPerspectiveWorldToAtmosphereDirection(rayDirWorld, args);

  const float kmPerWorldUnit = 1.0f / max(args.worldUnitsPerKilometer, 1e-6f);
  const vec3 rayOrigin = aerialPerspectiveRayOrigin(args);

  // Only the portion beyond the global volumetrics range and short of the tile's farthest surface
  // belongs to this froxel.
  const float maxRayLengthWorld = aerialPerspectiveMaxRayLength(
    aerialPerspectiveTileDepthInput.Load(int3(threadId.xy, 0)), uvw.xy, args);
  const float startDistanceKm = args.aerialPerspectiveStartDistance * kmPerWorldUnit;
  const float endDistanceKm = min(rayLengthWorld, maxRayLengthWorld) * kmPerWorldUnit;

  vec3 transmittance = vec3(1.0f);
  vec3 inScatter = vec3(0.0f, 0.0f, 0.0f);

  const CloudArgs cloud = cloudConstants;
  if (endDistanceKm > startDistanceKm) {
    if (cloud.enabled && (cloud.flags & CLOUD_FLAG_FOG_SHADOWS) != 0) {
      inScatter = evalCloudShadowedSegment(rayOrigin, rayDir, rayDirWorld, args, cloud, endDistanceKm, startDistanceKm, transmittance);
    } else {
      // Fewer steps than the sky, since each froxel covers a short segment and neighbouring slices
      // are filtered together on lookup.
      inScatter = evalAtmosphereRadianceSegment(
        rayOrigin, rayDir, args.sunDirection, args,
        transmittanceLutInput, multiscatteringLutInput, aerosolPhaseLutInput,
        12, endDistanceKm, startDistanceKm, transmittance);
    }
  }

  // A stores the mean of the wavelength dependent transmittance, per paper Section 5.4.
  const float meanTransmittance = (transmittance.x + transmittance.y + transmittance.z) * (1.0f / 3.0f);

  aerialPerspectiveLutOutput[threadId] = float4(inScatter, saturate(meanTransmittance));
}
