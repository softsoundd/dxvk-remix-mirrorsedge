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

// Ray traced aerial perspective volume.
//
// Same froxel layout and output as aerial_perspective_lut.comp.slang, but every sample also traces
// visibility rays through the scene: one toward the sun for the single scattering and one toward the sky
// for the multiple scattering, so air inside a building or a street canyon is lit by neither. Sample
// positions and sky directions are jittered per frame and the previous frame's volume is reprojected
// and blended in, so the estimate converges over a few frames.

#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/atmosphere/aerial_perspective_binding_indices.h"

#include "rtx/utility/common.slangh"
#include "rtx/utility/color.slangh"
#include "rtx/utility/noise.slangh"
#include "rtx/utility/sampling.slangh"
#include "rtx/concept/ray/ray.slangh"
#include "rtx/algorithm/visibility.slangh"
#include "rtx/pass/atmosphere/atmosphere_common.slangh"
#include "rtx/pass/clouds/cloud_common.slangh"

layout(binding = AERIAL_PERSPECTIVE_BINDING_PREV_LUT_INPUT)
Texture3D<float4> PrevAerialPerspectiveLut;

layout(binding = AERIAL_PERSPECTIVE_BINDING_TILE_DEPTH_INPUT)
Texture2D<float> AerialPerspectiveTileDepth;

layout(binding = AERIAL_PERSPECTIVE_BINDING_PREV_TILE_DEPTH_INPUT)
Texture2D<float> PrevAerialPerspectiveTileDepth;

layout(binding = AERIAL_PERSPECTIVE_BINDING_LUT_OUTPUT, rgba16f)
RWTexture3D<float4> AerialPerspectiveLutOutput;

static const uint kMaxShadowSteps = 32;

// Fraction of light from a direction that the scene lets through to a world position.
vec3 evalSceneVisibility(vec3 worldPosition, vec3 directionWorld, AtmosphereArgs args)
{
  Ray visibilityRay = rayCreateEmpty();
  visibilityRay.origin = worldPosition;
  visibilityRay.direction = directionWorld;
  visibilityRay.tMax = max(args.aerialPerspectiveShadowMaxDistance, 1.0f);

  // Culling disabled as in the froxel fog: a one sided wall must shadow the air behind it from either side.
  // The volume lives in the main portal space like the froxel grid.
  const uint8_t rayMask = OBJECT_MASK_OPAQUE | OBJECT_MASK_ALPHA_BLEND;
  const PortalSpace2BitsType portalSpace = PORTAL_SPACE_NONE;
  const VisibilityResult visibility = traceVisibilityRay<visibilityModeDisableCulling>(
    visibilityRay, rayMask, invalidRayPortalIndex, portalSpace);

  return visibility.hasOpaqueHit ? vec3(0.0f) : saturate(visibility.attenuation);
}

// How much brighter (or darker) the sky is along a world direction with the cloud layer over it, from the
// dome: the multiple scattering term stands for the sky's light, which an overcast deck replaces with the
// light of its base.
float evalCloudySkyRatio(vec3 directionWorld, AtmosphereArgs args)
{
  if (!cb.cloudArgs.enabled)
  {
    return 1.0f;
  }
  const vec3 dirYUp = normalize(cloudWorldToYUp(directionWorld, args.isZUp != 0));
  const vec3 sky = sampleSkyAmbientForVolume(dirYUp, args, AtmosphereSkyViewLut);
  const vec4 layer = cloudSampleDome(CloudDome, CloudSampler, dirYUp, cb.cloudArgs);
  const float skyLuminance = calcBt709Luminance(sky);
  return skyLuminance > 1e-6f ? clamp(calcBt709Luminance(cloudCompositeOverSky(layer, sky)) / skyLuminance, 0.0f, 4.0f) : 1.0f;
}

// Cosine weighted direction over the sky hemisphere, i.e. around the world up axis.
vec3 sampleSkyDirection(vec2 rndSample, AtmosphereArgs args)
{
  float unusedPdf;
  const vec3 direction = calcCosineHemisphereDirectionSample(rndSample, unusedPdf);

  return args.isZUp != 0 ? direction : vec3(direction.x, direction.z, direction.y);
}

// Previous frame's froxel coordinate of a world position, or false when it fell outside that frustum
// or its previous march covered a different stretch of air: a froxel behind a wall in either frame holds
// the integral to that wall, so its history and its estimate describe different paths. endDistance is
// the length this frame's march covers, in world units.
bool reprojectToPreviousVolume(vec3 worldPosition, float endDistance, AtmosphereArgs args, out vec3 prevUvw)
{
  prevUvw = vec3(0.0f);

  const vec3 offset = worldPosition - args.prevCameraPosition;
  const float forwardDistance = dot(offset, args.prevCameraForward);
  if (forwardDistance <= 0.0f) {
    return false;
  }

  // cameraRight / cameraUp are pre-scaled by the frustum half extents at unit forward distance.
  const float rightScale = max(dot(args.prevCameraRight, args.prevCameraRight), 1e-8f);
  const float upScale = max(dot(args.prevCameraUp, args.prevCameraUp), 1e-8f);
  const vec2 ndc = vec2(
    dot(offset, args.prevCameraRight) / (forwardDistance * rightScale),
    dot(offset, args.prevCameraUp) / (forwardDistance * upScale));

  prevUvw = vec3((ndc.x + 1.0f) * 0.5f, (1.0f - ndc.y) * 0.5f, aerialPerspectiveForwardDistanceToSlice(forwardDistance, args));
  if (any(prevUvw < vec3(0.0f)) || any(prevUvw > vec3(1.0f))) {
    return false;
  }

  // The frustum extents are shared between the frames, so the current basis serves for the ray length.
  const float prevTileDepth = PrevAerialPerspectiveTileDepth.Load(int3(aerialPerspectiveTileCoordinate(prevUvw.xy, args), 0));
  const float prevEndDistance = min(length(offset), aerialPerspectiveMaxRayLength(prevTileDepth, prevUvw.xy, args));

  // Camera motion alone shifts the length by the frame's displacement; only a change of bound is rejected.
  const float tolerance = max(0.05f * endDistance, 0.002f * args.worldUnitsPerKilometer);

  return abs(prevEndDistance - endDistance) <= tolerance;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 threadId : SV_DispatchThreadID) {
  const AtmosphereArgs args = aerialPerspectiveMediumArgs(cb.atmosphereArgs);

  const uint3 size = uint3(args.aerialPerspectiveLutSize, args.aerialPerspectiveLutSize, args.aerialPerspectiveLutDepth);
  if (any(threadId >= size)) {
    return;
  }

  const vec3 uvw = (vec3(threadId) + 0.5f) / vec3(size);

  const vec3 froxelWorldPosition = aerialPerspectiveUvwToWorldPosition(uvw, args);
  const vec3 rayVectorWorld = froxelWorldPosition - args.cameraPosition;
  const float rayLengthWorld = length(rayVectorWorld);
  if (rayLengthWorld <= 0.0f) {
    AerialPerspectiveLutOutput[threadId] = float4(0.0f, 0.0f, 0.0f, 1.0f);
    return;
  }

  const vec3 rayDirWorld = rayVectorWorld / rayLengthWorld;
  const vec3 rayDir = aerialPerspectiveWorldToAtmosphereDirection(rayDirWorld, args);
  const vec3 sunDirectionWorld = normalize(args.isZUp != 0
    ? vec3(args.sunDirection.x, args.sunDirection.z, args.sunDirection.y)
    : args.sunDirection);

  const float kmPerWorldUnit = 1.0f / max(args.worldUnitsPerKilometer, 1e-6f);
  const vec3 rayOrigin = aerialPerspectiveRayOrigin(args);

  // Only the portion beyond the global volumetrics range and short of the tile's farthest surface
  // belongs to this froxel.
  const float endDistanceWorld = min(rayLengthWorld, aerialPerspectiveMaxRayLength(
    AerialPerspectiveTileDepth.Load(int3(threadId.xy, 0)), uvw.xy, args));
  const float startDistanceKm = args.aerialPerspectiveStartDistance * kmPerWorldUnit;
  const float endDistanceKm = endDistanceWorld * kmPerWorldUnit;

  vec3 transmittance = vec3(1.0f);
  vec3 inScatter = vec3(0.0f);

  AtmosphereMarchSegment segment;
  if (clipAtmosphereMarch(rayOrigin, rayDir, args, endDistanceKm, startDistanceKm, segment)) {
    const uint stepCount = clamp(args.aerialPerspectiveShadowSteps, 1u, kMaxShadowSteps);
    const float span = segment.t1 - segment.t0;
    const float invSteps = 1.0f / float(stepCount);

    const float cosTheta = dot(rayDir, args.sunDirection);
    const float phaseR = rayleighPhase(cosTheta, args.rayleighPhaseGamma);
    const vec3 phaseM = miePhase(cosTheta, args, AtmosphereAerosolPhaseLut);

    // One jitter per froxel and frame, shared by its steps so the march stays monotonic.
    const float jitter = randomFloat(uint4(threadId, args.aerialPerspectiveFrameIndex));

    float segmentStart = 0.0f;
    for (uint i = 0; i < stepCount; ++i) {
      const float segmentEnd = atmosphereStepFraction(float(i + 1) * invSteps);
      const float dt = (segmentEnd - segmentStart) * span;
      const float t = segment.t0 + (segmentStart + (segmentEnd - segmentStart) * jitter) * span;
      segmentStart = segmentEnd;

      if (dt <= 0.0f) {
        continue;
      }

      const vec3 samplePos = rayOrigin + rayDir * t;
      const vec3 sampleWorldPosition = args.cameraPosition + rayDirWorld * (t * args.worldUnitsPerKilometer);

      // Atmospheric shadowing of the sun, then the scene's and the cloud layer's.
      vec3 sunShadowing = evalSunShadowing(samplePos, args.sunDirection, AtmosphereTransmittanceLut, args);
      if (any(sunShadowing > vec3(0.0f))) {
        sunShadowing *= evalSceneVisibility(sampleWorldPosition, sunDirectionWorld, args);
        if ((cb.cloudArgs.flags & CLOUD_FLAG_FOG_SHADOWS) != 0) {
          sunShadowing *= cloudSunTransmittance(CloudSunGrid, CloudShadowMap, CloudSampler, sampleWorldPosition, cb.cloudArgs, args);
        }
      }

      // One sky direction per step, so the accumulation converges to the sky fraction the step sees, through
      // the clouds over it.
      const vec2 skySample = vec2(
        randomFloat(uint4(threadId, args.aerialPerspectiveFrameIndex * kMaxShadowSteps + i)),
        randomFloat(uint4(threadId.yzx, args.aerialPerspectiveFrameIndex * kMaxShadowSteps + i + 1u)));
      const vec3 skyDirection = sampleSkyDirection(skySample, args);
      const vec3 skyVisibility = evalSceneVisibility(sampleWorldPosition, skyDirection, args) * evalCloudySkyRatio(skyDirection, args);

      vec3 sampleTransmittance;
      const vec3 contribution = transmittance * integrateAtmosphereStep(
        samplePos, rayDir, dt, args.sunDirection, phaseR, phaseM, sunShadowing, skyVisibility, args, AtmosphereMultiscatteringLut, sampleTransmittance);

      if (any(isnan(contribution)) || any(isinf(contribution))) {
        continue;
      }

      inScatter += contribution;
      transmittance *= sampleTransmittance;

      if (all(transmittance < vec3(1e-4f))) {
        break;
      }
    }
  }

  // A stores the mean of the wavelength dependent transmittance, per paper Section 5.4.
  const float meanTransmittance = (transmittance.x + transmittance.y + transmittance.z) * (1.0f / 3.0f);
  vec4 result = float4(max(inScatter, vec3(0.0f)), saturate(meanTransmittance));

  // Temporal accumulation against the reprojected previous volume.
  vec3 prevUvw;
  if (args.aerialPerspectiveHistoryValid != 0 && args.aerialPerspectiveTemporalBlend > 0.0f &&
      reprojectToPreviousVolume(froxelWorldPosition, endDistanceWorld, args, prevUvw)) {
    const vec4 history = sampleAerialPerspectiveLut(PrevAerialPerspectiveLut, prevUvw, args);
    if (!any(isnan(history)) && !any(isinf(history))) {
      result = lerp(result, history, saturate(args.aerialPerspectiveTemporalBlend));
    }
  }

  AerialPerspectiveLutOutput[threadId] = result;
}
