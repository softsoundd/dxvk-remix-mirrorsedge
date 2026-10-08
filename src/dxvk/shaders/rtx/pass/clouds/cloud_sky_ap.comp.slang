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

// Aerial perspective from the camera to any distance in any direction, for the air in front of the clouds:
// Hillaire 2020's aerial perspective volume, parameterised like the sky-view LUT rather than the frustum, since
// clouds fill the sky well beyond the frustum volume's 32 km. Each texel marches its direction once and stores
// the in-scatter and transmittance accumulated to every slice: logarithmic in distance, from the froxel fog's
// hand-off out to CLOUD_SKY_AP_MAX_DISTANCE_KM. The air is shadowed by the clouds (crepuscular rays beneath a
// broken deck) where the sun grid or the far field shadow map covers it.

#include "rtx/pass/clouds/cloud_binding_indices.h"
#include "rtx/pass/clouds/cloud_common.slangh"

layout(binding = CLOUD_BINDING_CONSTANTS)
ConstantBuffer<CloudConstants> constants;

layout(binding = CLOUD_BINDING_VOLUME_SAMPLER)
SamplerState volumeSampler;

layout(binding = CLOUD_BINDING_SUN_GRID)
Texture3D<float4> sunGrid;

layout(binding = CLOUD_BINDING_SHADOW_MAP)
Texture2D<float> shadowMap;

layout(binding = CLOUD_BINDING_TRANSMITTANCE_LUT)
Texture2D transmittanceLut;

layout(binding = CLOUD_BINDING_MULTISCATTERING_LUT)
Texture2D multiscatteringLut;

layout(binding = CLOUD_BINDING_AEROSOL_PHASE_LUT)
Texture2D aerosolPhaseLut;

layout(binding = CLOUD_BINDING_BAKE_OUTPUT, rgba16f)
RWTexture3D<float4> inScatterOutput;

layout(binding = CLOUD_BINDING_BAKE_OUTPUT2, rgba16f)
RWTexture3D<float4> transmittanceOutput;

// Columns interleave over frames (period / phase): the sky and the clouds' shadows in it change slowly.
layout(push_constant)
ConstantBuffer<CloudPassArgs> passArgs;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
  const uint2 lutTexel = uint2(threadId.x * max(passArgs.period, 1u) + passArgs.phase, threadId.y);
  if (any(lutTexel >= uint2(CLOUD_SKY_AP_LUT_WIDTH, CLOUD_SKY_AP_LUT_HEIGHT))) {
    return;
  }

  const CloudArgs cloud = constants.cloud;
  const AtmosphereArgs atmosphere = constants.atmosphere;
  const AtmosphereArgs medium = aerialPerspectiveMediumArgs(atmosphere);

  const vec2 uv = (vec2(lutTexel) + 0.5f) / vec2(CLOUD_SKY_AP_LUT_WIDTH, CLOUD_SKY_AP_LUT_HEIGHT);
  const vec3 direction = cloudSkyUvToDirection(uv);
  const vec3 origin = vec3(0.0f, max(cloud.cameraPositionKm.y, kPlanetRadiusOffset), 0.0f);
  const vec3 sunDir = normalize(atmosphere.sunDirection);
  const vec3 planetCenter = getPlanetCenter(atmosphere);

  AtmosphereMarchSegment segment;
  const bool marchable = clipAtmosphereMarch(origin, direction, atmosphere, CLOUD_SKY_AP_MAX_DISTANCE_KM, cloud.apStartDistanceKm, segment);

  const float cosTheta = dot(direction, sunDir);
  const float phaseR = rayleighPhase(cosTheta, atmosphere.rayleighPhaseGamma);
  const vec3 phaseM = miePhase(cosTheta, atmosphere, aerosolPhaseLut);
  const bool cloudShadows = cloud.enabled && (cloud.flags & CLOUD_FLAG_FOG_SHADOWS) != 0;

  vec3 inScatter = vec3(0.0f);
  vec3 transmittance = vec3(1.0f);
  float t = marchable ? segment.t0 : 0.0f;

  for (uint slice = 0u; slice < uint(CLOUD_SKY_AP_LUT_DEPTH); ++slice) {
    const float sliceEnd = marchable ? min(cloudSkyApWToDistance((float(slice) + 0.5f) / float(CLOUD_SKY_AP_LUT_DEPTH)), segment.t1) : 0.0f;
    if (sliceEnd > t) {
      // Three steps per slice: the slices are a few percent of their distance deep.
      const int steps = 3;
      const float dt = (sliceEnd - t) / float(steps);
      for (int i = 0; i < steps; ++i) {
        const float ts = t + (float(i) + 0.5f) * dt;
        const vec3 samplePosition = origin + direction * ts;
        vec3 sunShadowing = evalSunShadowing(samplePosition, sunDir, transmittanceLut, atmosphere);

        if (cloudShadows && any(sunShadowing > vec3(0.0f))) {
          const vec3 positionKm = vec3(
            cloud.cameraPositionKm.x + direction.x * ts,
            length(samplePosition - planetCenter) - atmosphere.planetRadius,
            cloud.cameraPositionKm.z + direction.z * ts);
          sunShadowing *= cloudShadowTransmittance(cloudSunOpticalDepth(sunGrid, shadowMap, volumeSampler, positionKm, sunDir, cloud), cloud);
        }

        vec3 stepTransmittance;
        const vec3 contribution = transmittance * integrateAtmosphereStep(
          samplePosition, direction, dt, sunDir, phaseR, phaseM, sunShadowing, vec3(1.0f), medium, multiscatteringLut, stepTransmittance);
        if (all(isfinite(contribution))) {
          inScatter += contribution;
        }
        transmittance *= stepTransmittance;
      }
      t = sliceEnd;
    }

    const uint3 texel = uint3(lutTexel, slice);
    inScatterOutput[texel] = vec4(inScatter, 0.0f);
    transmittanceOutput[texel] = vec4(transmittance, 0.0f);
  }
}
