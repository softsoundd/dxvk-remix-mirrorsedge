/*
* Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx/pass/composite/composite_bindings.slangh"

#include "rtx/pass/composite/composite_args.h"
#include "rtx/pass/composite/composite.slangh"

#include "rtx/utility/common.slangh"
#include "rtx/utility/packing.slangh"
#include "rtx/utility/noise.slangh"
#include "rtx/utility/color.slangh"
#include "rtx/utility/froxel.slangh"
#include "rtx/utility/brdf.slangh"
#include "rtx/utility/demodulate_helpers.slangh"
#include "rtx/utility/debug_view_helpers.slangh"
#include "rtx/concept/camera/camera.slangh"
#include "rtx/concept/surface/alpha_blend_surface.slangh"
#include "rtx/utility/geometry_flags.slangh"
#include "rtx/pass/post_fx/post_fx_motion_blur_geometry_flags.slangh"

#include "rtx/algorithm/volume_composite_helpers.slangh"
#include "rtx/algorithm/accumulate.slangh"
#include "rtx/pass/atmosphere/atmosphere_common.slangh"
#include "rtx/pass/sparse_rendering/sparse_rendering.slangh"
#include "rtx/pass/sparse_rendering/compact_active_pixels.slangh"

float3 sampleDomeLightTexture(Sampler2D<float3> DomeLight, float3 worldDirection, float4x4 worldToDomeLightTransform)
{
  float3 domeSampleDirection = mul(worldToDomeLightTransform, float4(worldDirection, 0.0f)).xyz;
  float2 sampleUV = cartesianDirectionToLatLongSphere(domeSampleDirection);
  return DomeLight.SampleLevel(sampleUV, 0).xyz;
}

void preCompositeDebugView(uint2 pixelCoordinate, uint2 compactedCoordinate, bool pixelIsActive)
{
  // Debug views that need to run before the active early-out so they cover every pixel.
  switch(cb.debugViewIdx)
  {
  case DEBUG_VIEW_SPARSE_RENDERING_PIXEL_SAMPLING_RATE:
    storeInDebugView(pixelCoordinate, SparseRenderingPixelSamplingRate[pixelCoordinate]);
    break;
  case DEBUG_VIEW_DENOISED_PRIMARY_INDIRECT_DIFFUSE_RADIANCE:
  case DEBUG_VIEW_DENOISED_PRIMARY_INDIRECT_DIFFUSE_HIT_T:
    // Inactive pixels have no compacted radiance and keep the cleared value.
    if (cb.enableSeparatedDenoisers && pixelIsActive)
    {
      storeInDebugView(pixelCoordinate, PrimaryIndirectDiffuseRadianceHitDistance[compactedCoordinate]);
    }
    break;
  case DEBUG_VIEW_DENOISED_PRIMARY_INDIRECT_SPECULAR_RADIANCE:
  case DEBUG_VIEW_DENOISED_PRIMARY_INDIRECT_SPECULAR_HIT_T:
    if (cb.enableSeparatedDenoisers && pixelIsActive)
    {
      storeInDebugView(pixelCoordinate, PrimaryIndirectSpecularRadianceHitDistance[compactedCoordinate]);
    }
    break;

  case DEBUG_VIEW_SPARSE_RENDERING_ACTIVE_PIXELS_OUTPUT_SCALE:
  {
    const float rate = SparseRenderingPixelSamplingRate[pixelCoordinate];
    storeInDebugView(pixelCoordinate, rate > 0.f ? 1.0 / rate : 0.f);
    break;
  }

  case DEBUG_VIEW_SPARSE_RENDERING_ACTIVE_PIXELS_MASK:
  {
    if (isActivePixelFromMask(SparseRenderingActivePixelMask, pixelCoordinate))
    {
      storeInDebugView(pixelCoordinate, 1.0);
    }
    break;
  }

  case DEBUG_VIEW_NAN:
  {
    bool isValid = true;
    // DEBUG_VIEW_SPARSE_RENDERING_PIXEL_SAMPLING_RATE / _ACTIVE_PIXELS_OUTPUT_SCALE
    isValid &= isValidValue(SparseRenderingPixelSamplingRate[pixelCoordinate]);
    // DEBUG_VIEW_SPARSE_RENDERING_ACTIVE_PIXELS_MASK
    isValid &= isValidValue(float(isActivePixelFromMask(SparseRenderingActivePixelMask, pixelCoordinate)));

    accumulateInDebugViewAnd(pixelCoordinate, isValid);
    break;
  }
  }
}

void updateAlphaBlendDebugView(
  uint2 pixelCoordinate,
  AlphaBlendSurface surface)
{
  // Output Debug Information
  switch(cb.debugViewIdx)
  {
  case DEBUG_VIEW_STOCHASTIC_ALPHA_BLEND_COLOR:
    storeInDebugView(pixelCoordinate, surface.isValid() ? surface.color.rgb : f16vec3(1,0,1));
    break;
  case DEBUG_VIEW_STOCHASTIC_ALPHA_BLEND_NORMAL:
    storeInDebugView(pixelCoordinate, surface.isValid() ? surface.normal : f16vec3(0));
    break;
  case DEBUG_VIEW_STOCHASTIC_ALPHA_BLEND_GEOMETRY_HASH:
    storeInDebugView(pixelCoordinate, r5g6b5ToColor(surface.geometryHash));
    break;    
  case DEBUG_VIEW_STOCHASTIC_ALPHA_BLEND_BACKGROUND_TRANSPARENCY:
    storeInDebugView(pixelCoordinate, surface.backgroundTransparency);
    break;
  }
}

void postCompositeDebugView(
  uint2 pixelCoordinate,
  GeometryFlags geometryFlags,
  bool pixelIsActive)
{
  // Output Debug Information
  switch(cb.debugViewIdx)
  {
  case DEBUG_VIEW_GEOMETRY_FLAGS_FIRST_SAMPLED_LOBE_IS_SPECULAR:
    storeInDebugView(pixelCoordinate, geometryFlags.firstSampledLobeIsSpecular);
    break;
  case DEBUG_VIEW_RAY_RECONSTRUCTION_HIT_DISTANCE:
    if (cb.writeRayReconstructionHitDistance)
    {
      storeInDebugView(pixelCoordinate, RayReconstructionHitDistanceOutput[pixelCoordinate] / 1000.0);
    }
    break;
  case DEBUG_VIEW_SPARSE_RENDERING_ACTIVE_THREADS:
    if (pixelIsActive)
    {
      storeInDebugView(pixelCoordinate, 1.0);
    }
    break;

  case DEBUG_VIEW_NAN:
  {
    bool isValid = true;
    // DEBUG_VIEW_GEOMETRY_FLAGS_FIRST_SAMPLED_LOBE_IS_SPECULAR
    isValid &= isValidValue(float(geometryFlags.firstSampledLobeIsSpecular));
    // DEBUG_VIEW_RAY_RECONSTRUCTION_HIT_DISTANCE
    if (cb.writeRayReconstructionHitDistance)
    {
      isValid &= isValidValue(RayReconstructionHitDistanceOutput[pixelCoordinate]);
    }

    accumulateInDebugViewAnd(pixelCoordinate, isValid);
    break;
  }
  }
}

// Loads the secondary (PSR) radiance contribution. RR consumes the raw combined signal directly;
// other upscalers retain the NRD unpack and remodulation path.
vec3 loadSecondaryRadiance(
  uint2 compactedCoordinate,
  GeometryFlags geometryFlags,
  vec3 secondaryAlbedo,
  vec3 secondarySpecularAlbedo,
  vec3 secondaryAttenuation)
{
  if (!geometryFlags.secondarySurfaceMask)
  {
    return vec3(0);
  }

  vec3 secondaryCombinedDiffuseRadiance = SecondaryCombinedDiffuseRadianceHitDistance[compactedCoordinate].xyz;
  vec3 secondaryCombinedSpecularRadiance = SecondaryCombinedSpecularRadianceHitDistance[compactedCoordinate].xyz;

  if (!cb.compositeSecondaryCombinedDiffuse)
  {
    secondaryCombinedDiffuseRadiance = vec3(0);
  }
  if (!cb.compositeSecondaryCombinedSpecular)
  {
    secondaryCombinedSpecularRadiance = vec3(0);
  }

  if (cb.useRayReconstruction)
  {
    return (secondaryCombinedDiffuseRadiance + secondaryCombinedSpecularRadiance) * secondaryAttenuation;
  }

  unpackDenoiserRadiance(secondaryCombinedDiffuseRadiance, cb.secondaryCombinedDenoiser);
  unpackDenoiserRadiance(secondaryCombinedSpecularRadiance, cb.secondaryCombinedDenoiser);

  return (secondaryCombinedDiffuseRadiance * secondaryAlbedo +
          secondaryCombinedSpecularRadiance * secondarySpecularAlbedo) * secondaryAttenuation;
}

// Loads, combines, and remodulates the primary direct + indirect radiance contribution.
// Caller must gate this on pixelIsActive — the radiance inputs are only valid on active pixels.
vec3 loadPrimaryRadiance(
  uint2 pixelCoordinate,
  uvec2 compactedCoordinate,
  vec3 primaryAlbedo,
  vec3 primarySpecularAlbedo,
  vec3 primaryAttenuation,
  float perceptualRoughness,
  bool primaryMiss)
{
  // In combined-denoiser mode the indirect radiance is combined in the direct buffer.
  vec3 primaryDirectDiffuseRadiance = PrimaryDirectDiffuseRadianceHitDistance[compactedCoordinate].xyz;
  vec3 primaryDirectSpecularRadiance = PrimaryDirectSpecularRadianceHitDistance[compactedCoordinate].xyz;

  unpackDenoiserRadiance(primaryDirectDiffuseRadiance, cb.primaryDirectDenoiser);
  unpackDenoiserRadiance(primaryDirectSpecularRadiance, cb.primaryDirectDenoiser);

  vec3 primaryIndirectDiffuseRadiance = vec3(0);
  vec3 primaryIndirectSpecularRadiance = vec3(0);

  if (cb.enableSeparatedDenoisers)
  {
    if (cb.usePostFilter)
    {
      primaryIndirectDiffuseRadiance = applyPostFilter(PrimaryIndirectDiffuseRadianceHitDistance, pixelCoordinate, cb.primaryIndirectDenoiser);
      primaryIndirectSpecularRadiance = applyPostFilter(PrimaryIndirectSpecularRadianceHitDistance, pixelCoordinate, cb.primaryIndirectDenoiser);
    }
    else
    {
      primaryIndirectDiffuseRadiance = PrimaryIndirectDiffuseRadianceHitDistance[compactedCoordinate].xyz;
      primaryIndirectSpecularRadiance = PrimaryIndirectSpecularRadianceHitDistance[compactedCoordinate].xyz;

      unpackDenoiserRadiance(primaryIndirectDiffuseRadiance, cb.primaryIndirectDenoiser);
      unpackDenoiserRadiance(primaryIndirectSpecularRadiance, cb.primaryIndirectDenoiser);
    }
  }

  if (!cb.compositePrimaryDirectDiffuse)
  {
    primaryDirectDiffuseRadiance = vec3(0);
  }
  if (!cb.compositePrimaryDirectSpecular)
  {
    primaryDirectSpecularRadiance = vec3(0);
  }
  if (!cb.compositePrimaryIndirectDiffuse)
  {
    primaryIndirectDiffuseRadiance = vec3(0);
  }
  if (!cb.compositePrimaryIndirectSpecular)
  {
    primaryIndirectSpecularRadiance = vec3(0);
  }

  vec2 bsdfFactor = vec2(1);
  if (cb.enableRtxdi)
  {
    bsdfFactor = BSDFFactor[compactedCoordinate].xy;
  }

  vec2 bsdfFactor2 = vec2(1);
  if (cb.enableReSTIRGI)
  {
    bsdfFactor2 = BSDFFactor2[pixelCoordinate].xy;
  }

  if (cb.enhanceAlbedo && cb.enableReSTIRGI)
  {
    // Note: this is not a reachable path when Sparse Rendering is enabled (bsdfFactor2.x == 1.0f),
    // since only NRC path is supported. Otherwise, this would have to be done for inactive pixels as well.
    PrimaryAlbedo[compactedCoordinate].xyz = primaryAlbedo * sqrt(bsdfFactor2.x);
  }

  vec3 primaryCombinedDiffuseRadiance;
  vec3 primaryCombinedSpecularRadiance;

  if (cb.enableSeparatedDenoisers)
  {
    primaryCombinedDiffuseRadiance = primaryDirectDiffuseRadiance * bsdfFactor.x + primaryIndirectDiffuseRadiance * bsdfFactor2.x;
    primaryCombinedSpecularRadiance = primaryDirectSpecularRadiance * bsdfFactor.y + primaryIndirectSpecularRadiance * bsdfFactor2.y;
  }
  else
  {
    bsdfFactor.x = clamp(bsdfFactor.x, 0.7f, 1.3f);
    bsdfFactor = 0.5f * (bsdfFactor + bsdfFactor2);
    primaryCombinedDiffuseRadiance = primaryDirectDiffuseRadiance * bsdfFactor.x;
    primaryCombinedSpecularRadiance = primaryDirectSpecularRadiance * bsdfFactor.y;
  }

  float roughnessFactor = 1.0f;
  if (cb.demodulateRoughness)
  {
    roughnessFactor = getRoughnessDemodulationFactor(perceptualRoughness, cb.roughnessDemodulationOffset);
  }

  vec3 remodulatedTotalPrimaryRadiance =
    (primaryCombinedDiffuseRadiance * primaryAlbedo +
     primaryCombinedSpecularRadiance * primarySpecularAlbedo / roughnessFactor) * primaryAttenuation;

  if (cb.enableReSTIRGI)
  {
    float specularWeight = pow(perceptualRoughness, cb.pixelHighlightReuseStrength);
    LastFinalOutput[pixelCoordinate] = primaryMiss
      ? vec4(0)
      : vec4(primaryIndirectDiffuseRadiance, calcBt709Luminance(primaryIndirectSpecularRadiance) / roughnessFactor * specularWeight);
  }

  return primaryMiss ? cb.clearColorFinalColor : remodulatedTotalPrimaryRadiance;
}

void loadInputRadiance(
  uint2 pixelCoordinate,
  uvec2 compactedCoordinate,
  GeometryFlags geometryFlags,
  bool pixelIsActive,
  vec3 primaryAlbedo,
  vec3 primarySpecularAlbedo,
  vec3 primaryAttenuation,
  vec3 secondaryAlbedo,
  vec3 secondarySpecularAlbedo,
  vec3 secondaryAttenuation,
  float perceptualRoughness,
  bool primaryMiss,
  out vec3 sharedRadiance,
  out vec3 remodulatedTotalPrimaryRadiance,
  out vec3 remodulatedTotalSecondaryRadiance)
{
  // Shared radiance (emissive + pre-attenuated layers) is addressed by pixel in every mode.
  sharedRadiance = vec3(SharedRadianceRG[pixelCoordinate].xy, SharedRadianceB[pixelCoordinate]);

  // Inactive pixels take default flags with no secondary surface, so this returns zero there.
  remodulatedTotalSecondaryRadiance = loadSecondaryRadiance(
    compactedCoordinate, geometryFlags,
    secondaryAlbedo, secondarySpecularAlbedo, secondaryAttenuation);

  remodulatedTotalPrimaryRadiance = vec3(0);
  if (pixelIsActive)
  {
    remodulatedTotalPrimaryRadiance = loadPrimaryRadiance(
      pixelCoordinate,
      compactedCoordinate,
      primaryAlbedo, primarySpecularAlbedo, primaryAttenuation,
      perceptualRoughness, primaryMiss);
  }
}

// LUT hemisphere mean * 0.5 * strength matches froxel SH evaluateHG(*, 0) (2π / 4π).
float3 evalReflectionSkyFloor()
{
  if (!cb.volumeArgs.physicalAtmosphereEnabled ||
      cb.volumeArgs.skyAmbientStrength <= 0.0f)
  {
    return cb.volumeArgs.multiScatteringEstimate;
  }

  const float3 lutMean = SkyHemisphereMean.Load(int3(0, 0, 0)).rgb;
  return lutMean * (cb.volumeArgs.skyAmbientStrength * 0.5f);
}

// Whether the aerial perspective volume applies this frame at all.
bool aerialPerspectiveActive()
{
  return bool(cb.volumeArgs.physicalAtmosphereEnabled) &&
    cb.atmosphereArgs.aerialPerspectiveLutSize != 0 &&
    !cb.domeLightArgs.active;
}

// Camera segment of the aerial perspective for a pixel, up to a distance along its view ray.
void evalCameraSegmentAerialPerspective(uint2 pixelCoordinate, float hitDistance, out vec3 inScatter, out float transmittance)
{
  const vec2 screenUV = cameraPixelCoordinateToScreenUV(cb.camera, pixelCoordinate);
  const vec3 uvw = aerialPerspectiveScreenUvToUvw(screenUV, hitDistance, cb.atmosphereArgs);
  const vec4 aerialPerspective = sampleAerialPerspectiveLut(AtmosphereAerialPerspectiveLut, uvw, cb.atmosphereArgs);

  inScatter = max(aerialPerspective.rgb, vec3(0.0f));
  transmittance = saturate(aerialPerspective.a);
}

vec3 integrateVolumetricTransmittance(
  inout RNG randomState,
  const VolumeArgs volumeArgs,
  const float timeMS,
  const bool primaryMiss,
  const vec3 startPosition,
  const float hitDistance,
  const vec3 viewDir)
{
  if (!volumeArgs.enable)
  {
    return vec3(1.0f);
  }

  const VolumeDefinitionCamera camera = volumeArgs.cameras[froxelVolumeMain];
  float maxPathDistance = primaryMiss
    ? volumeArgs.froxelMaxDistance
    : hitDistance - (volumeArgs.froxelMaxDistance / volumeArgs.froxelDepthSlices) * volumeArgs.depthOffset;
  maxPathDistance = clamp(maxPathDistance, 0.0f, volumeArgs.froxelMaxDistance);

  const vec3 targetPosition = startPosition + viewDir * maxPathDistance;
  const vec3 startPositionTranslated = worldToTranslatedWorld(camera, startPosition);
  const vec3 targetPositionTranslated = worldToTranslatedWorld(camera, targetPosition);
  const vec3 targetVector = targetPositionTranslated - startPositionTranslated;
  const uint numSteps = volumeArgs.froxelDepthSlices / 4;

  vec3 previousTranslatedWorldPosition = startPositionTranslated;
  vec3 accumulatedTransmittance = vec3(1.0f);

  for (uint z = 0; z < numSteps; z++)
  {
    const float t = (float(z + 1) + getNextSampleBlueNoise(randomState)) / float(numSteps + 1);
    const float nonlinearT = pow(t, volumeArgs.froxelDepthSliceDistributionExponent);
    const vec3 currentTranslatedWorldPosition = startPositionTranslated + targetVector * nonlinearT;
    const float stepSize = length(currentTranslatedWorldPosition - previousTranslatedWorldPosition);
    const float density = sampleDensityField(
      volumeArgs,
      camera,
      previousTranslatedWorldPosition,
      currentTranslatedWorldPosition,
      stepSize,
      timeMS);

    accumulatedTransmittance *= calcVolumetricAttenuation(
      volumeArgs,
      camera,
      false,
      stepSize,
      previousTranslatedWorldPosition,
      viewDir,
      density);
    previousTranslatedWorldPosition = currentTranslatedWorldPosition;
  }

  if (primaryMiss)
  {
    accumulatedTransmittance *= calcVolumetricAttenuation(
      volumeArgs, camera, true, 0.0f, previousTranslatedWorldPosition, viewDir);
  }
  else if (hitDistance > volumeArgs.froxelMaxDistance)
  {
    accumulatedTransmittance *= calcVolumetricAttenuation(
      volumeArgs,
      camera,
      false,
      hitDistance - volumeArgs.froxelMaxDistance,
      previousTranslatedWorldPosition,
      viewDir);
  }

  return accumulatedTransmittance;
}

void applyVolumetricLighting(
  uint2 pixelCoordinate,
  inout RNG randomState,
  CameraDirections primaryRayDirections,
  bool pixelIsActive,
  bool primaryMiss,
  float viewDistance,
  float radianceScale,
  float volumetricRadianceScale,
  vec3 remodulatedTotalSecondaryRadiance,
  vec3 sharedRadiance,
  // Reflection one-bounce fog: firstHit > 0 enables the dual froxel integrate, whether PSR replaced the
  // surface or the indirect integrator produced the reflection and stashed its segment.
  float psrFirstHitDistance,
  float psrReflectionSegmentSigned,
  float3 psrReflectionDirection,
  // Reflectance the fog beyond the surface is seen through. PSR content already carries it; an indirect
  // reflection is remodulated by specular albedo after denoising, so its haze must be scaled to match.
  vec3 segmentAttenuation,
  // PSR content takes the segment's extinction from here. An indirect reflection already had it applied in
  // the path by RESOLVER_USE_VOLUMETRIC_ATTENUATION, and that radiance buffer also holds the surface's own
  // direct lighting, which the segment must not attenuate.
  bool segmentIsPsr,
  out vec3 volumetricPreintegratedRadiance,
  out vec3 volumeAttenuation,
  out vec3 controlMaskVolumeAttenuation,
  out vec3 alphaBlendOutput,
  out float backgroundAlpha,
  out vec3 radianceOutput)
{
  // Override the default volume sampling RNG seed offset
  setVolumeSamplingRngSeed(1);

  // Calculate volumetric radiance and attenuation if volumetric lighting is enabled

  volumetricPreintegratedRadiance = vec3(0);
  volumeAttenuation = vec3(1);
  controlMaskVolumeAttenuation = vec3(1);
  alphaBlendOutput = vec3(0);
  backgroundAlpha = 1.0f;

  vec3 ignoredMeanInScatter = vec3(0.0f);

  // Note: This is only evaluated when volumetrics are enabled, no external check is needed.
  // Todo: Using Primary linear view Z for now, not the best option as this does not work with PSR (among other effects like
  // Portals and particles), but fine for initial testing.
  // Skipped at union-inactive pixels: the integration is expensive and its result only contributes
  // through paths gated on remodulated radiance, which is zero at those pixels anyway.
  // When sparse volumetrics is disabled for this pixel's primary classification, volumetrics run at
  // every pixel (full rate) and are not scaled below.
  const bool fullRateVolumetrics = primaryMiss
    ? !bool(cb.sparseRenderingArgs.enableSparseVolumetricsPrimaryMiss)
    : !bool(cb.sparseRenderingArgs.enableSparseVolumetricsPrimaryHit);
  if (pixelIsActive || fullRateVolumetrics)
  {
    vec2 volumeUV = cameraPixelCoordinateToScreenUV(cb.camera, pixelCoordinate);
    volumeUV.x /= cb.volumeArgs.numFroxelVolumes;

    const bool psrReflectionFog = psrFirstHitDistance > 0.0f
      && abs(psrReflectionSegmentSigned) > 0.0f
      && dot(psrReflectionDirection, psrReflectionDirection) > 0.0f;

    if (psrReflectionFog)
    {
      // Camera→mirror, then along the reflection ray. T = T0*T1; S = S0 + T0*S1.
      const float3 cameraPos = cameraGetWorldPosition(cb.camera);
      float3 t0 = vec3(1.0f);
      float3 s0 = vec3(0.0f);
      integrateVolumetricNEE(
        randomState, VolumeFilteredRadianceAge, VolumeFilteredRadianceY, VolumeFilteredRadianceCoCg,
        cb.volumeArgs, cb.timeSinceStartMS, pixelCoordinate, volumeUV, false,
        cameraPos, psrFirstHitDistance, primaryRayDirections.worldDirection,
        s0, t0, ignoredMeanInScatter);

      const float3 mirrorPos = cameraPos + primaryRayDirections.worldDirection * psrFirstHitDistance;
      const bool reflMiss = psrReflectionSegmentSigned < 0.0f;
      const float reflSeg = abs(psrReflectionSegmentSigned);
      float3 t1 = vec3(1.0f);
      float3 s1 = vec3(0.0f);
      const float3 reflectionInScatter = evalUnoccludedReflectionInScatter(
        cb.volumeArgs, psrReflectionDirection, evalReflectionSkyFloor(), mirrorPos,
        cb.unoccludedSunRadiance, cb.unoccludedSunDirection);
      integrateVolumetricNEE(
        randomState, VolumeFilteredRadianceAge, VolumeFilteredRadianceY, VolumeFilteredRadianceCoCg,
        cb.volumeArgs, cb.timeSinceStartMS, pixelCoordinate, volumeUV, reflMiss,
        mirrorPos, reflSeg, psrReflectionDirection,
        s1, t1, ignoredMeanInScatter, false, reflectionInScatter, true);

      volumeAttenuation = segmentIsPsr ? (t0 * t1) : t0;
      volumetricPreintegratedRadiance = s0 + t0 * s1 * segmentAttenuation;
    }
    else
    {
      integrateVolumetricNEE(
        randomState, VolumeFilteredRadianceAge, VolumeFilteredRadianceY, VolumeFilteredRadianceCoCg,
        cb.volumeArgs, cb.timeSinceStartMS, pixelCoordinate, volumeUV, primaryMiss, cameraGetWorldPosition(cb.camera), viewDistance, primaryRayDirections.worldDirection,
        volumetricPreintegratedRadiance, volumeAttenuation, ignoredMeanInScatter);
    }
    controlMaskVolumeAttenuation = volumeAttenuation;
  }
  else if (cb.enableDlssNrVolumetricControlMask)
  {
    // The control mask remains dense when volumetric radiance integration uses sparse pixels.
    RNG controlMaskRandomState = randomState;
    controlMaskVolumeAttenuation = integrateVolumetricTransmittance(
      controlMaskRandomState,
      cb.volumeArgs,
      cb.timeSinceStartMS,
      primaryMiss,
      cameraGetWorldPosition(cb.camera),
      viewDistance,
      primaryRayDirections.worldDirection);
  }

  // Stochastic alpha blend

  if (cb.enableStochasticAlphaBlend)
  {    
    AlphaBlendSurface surface = AlphaBlendSurface.createFromPacked(AlphaBlendGBuffer[pixelCoordinate], cb.alphaBlendSurfacePackMult);
    vec3 cameraPosition = cameraGetWorldPosition(cb.camera);

    // composite_alpha_blend only writes the radiance where the surface is valid.
    vec4 centerLight = vec4(0.0f);
    if (surface.isValid())
    {
      centerLight = AlphaBlendRadiance[pixelCoordinate];
    }

    bool discardPixel = cb.stochasticAlphaBlendDiscardBlackPixel && all(centerLight.xyz == 0);

    if (!discardPixel && surface.isValid())
    {
      vec3 particleNormal = surface.normal;
      backgroundAlpha = surface.backgroundTransparency;

      vec4 totalLight = vec4(centerLight.xyz, 1);

      if (cb.stochasticAlphaBlendEnableFilter)
      {
        float minHitT = surface.hitT;
        float maxHitT = surface.hitT;
        float4 multilayerLight = totalLight;
        for (int i = 0; i < 25; ++i)
        {
          int2 offset = int2(i % 5, i / 5) - 2;
          if (all(offset == 0))
          {
            continue;
          }
          int2 neighborPixel = clamp(int2(pixelCoordinate) + offset * 2, int2(0), cb.camera.resolution - 1);

          AlphaBlendSurface neighborSurface = AlphaBlendSurface.createFromPacked(AlphaBlendGBuffer[neighborPixel], cb.alphaBlendSurfacePackMult);
          if (neighborSurface.isValid())
          {
            const CameraDirections neighborRayDirections = cameraPixelCoordinateToDirection(cb.camera, neighborPixel.xy);
            vec3 neighborPosition = cameraPosition + neighborSurface.hitT * neighborRayDirections.worldDirection;

            const float transparencyBandwidth = 0.05;
            const float distanceWeight = 0.5;
            float weight = exp(
              -abs(neighborSurface.backgroundTransparency - surface.backgroundTransparency) / transparencyBandwidth +
              // Note: Convert offset to float to avoid need for integer dot product Vulkan feature to be enabled.
              -distanceWeight * dot(float2(offset), float2(offset))
            );

            minHitT = min(minHitT, neighborSurface.hitT);
            maxHitT = max(maxHitT, neighborSurface.hitT);

            vec4 neighborLight = AlphaBlendRadiance[neighborPixel];
            multilayerLight += vec4(neighborLight.xyz * weight, weight);
          }
        }
        multilayerLight.rgb /= multilayerLight.w;

        float depthRange = maxHitT - minHitT;
        float depthBandwidth = 0.05;
        totalLight.xyz = lerp(totalLight.xyz, multilayerLight.xyz, saturate(depthRange / surface.hitT / depthBandwidth));
      }
      alphaBlendOutput = totalLight.xyz;

      // Volumetrics transmittance in front of the surface, which its aerial perspective is seen through.
      vec3 alphaVolumetricAttenuation = vec3(1.0f, 1.0f, 1.0f);

      if (cb.volumeArgs.enable)
      {
        float alphaBlendSurfaceLinearViewZ = surface.hitT * primaryRayDirections.viewDirection.z;

        vec2 volumeUV = cameraPixelCoordinateToScreenUV(cb.camera, pixelCoordinate);
        volumeUV.x /= cb.volumeArgs.numFroxelVolumes;
        
        vec3 alphaVolumetricPreintegratedRadiance = vec3(0.0f, 0.0f, 0.0f);
        integrateVolumetricNEE(
          randomState, VolumeFilteredRadianceAge, VolumeFilteredRadianceY, VolumeFilteredRadianceCoCg,
          cb.volumeArgs, cb.timeSinceStartMS, pixelCoordinate, volumeUV, false, cameraGetWorldPosition(cb.camera), surface.hitT, primaryRayDirections.worldDirection,
          alphaVolumetricPreintegratedRadiance, alphaVolumetricAttenuation, ignoredMeanInScatter);

        alphaBlendOutput *= alphaVolumetricAttenuation * (1 - backgroundAlpha);
        alphaBlendOutput += alphaVolumetricPreintegratedRadiance * (1 - backgroundAlpha);
      }
      else
      {
        const vec4 fogColor = calculateFog(surface.hitT);
        alphaBlendOutput = alphaBlendOutput * (1.0f - fogColor.a) + fogColor.rgb * fogColor.a;
        alphaBlendOutput *= (1 - backgroundAlpha);
      }

      // Distant glass and particles haze over like the opaque surfaces behind them. The in-scatter
      // covers the surface's share of the pixel.
      if (aerialPerspectiveActive() && surface.hitT > 0.0f)
      {
        vec3 alphaInScatter;
        float alphaTransmittance;
        evalCameraSegmentAerialPerspective(pixelCoordinate, surface.hitT, alphaInScatter, alphaTransmittance);
        alphaBlendOutput = alphaBlendOutput * alphaTransmittance
          + alphaInScatter * alphaVolumetricAttenuation * (1 - backgroundAlpha);
      }

      // No sparse compensation scaling: alphaBlendOutput is already a dense per-pixel signal.
      // composite_alpha_blend writes AlphaBlendRadiance at every valid surface,
      // and it undoes the sparse scale of the neighbor radiance it samples, so there are no sparse holes to compensate for.
      // Applying radianceScale here would over-count active pixels and amplify surfaces seen
      // through alpha-blended particles.

      if (surface.hasEmissive)
      {
        alphaBlendOutput += sharedRadiance;
      }
    }

    updateAlphaBlendDebugView(pixelCoordinate, surface);
  }

  radianceOutput = volumetricPreintegratedRadiance * volumetricRadianceScale;
  radianceOutput += remodulatedTotalSecondaryRadiance * volumeAttenuation;
}

void applySkyContribution(
  uint2 pixelCoordinate,
  CameraDirections primaryRayDirections,
  bool primaryMiss,
  float radianceScale,
  vec3 volumeAttenuation,
  inout vec3 radianceOutput)
{
  // A zero scale marks an inactive sparse miss, whose sky comes from ray reconstruction.
  if (!primaryMiss || radianceScale == 0.f)
  {
    return;
  }

  // Physical Atmosphere sky is already in SharedRadiance — skip SkyMatte/SkyLight. An external
  // dome light still takes priority over the atmosphere, and is composited from here as usual.
  if (cb.volumeArgs.physicalAtmosphereEnabled && !cb.domeLightArgs.active)
  {
    return;
  }

  vec3 skyContribution;
  if (cb.domeLightArgs.active)
  {
    skyContribution = cb.domeLightArgs.radiance * sampleDomeLightTexture(SkyLight, primaryRayDirections.worldDirection, cb.domeLightArgs.worldToLightTransform) * volumeAttenuation;
  }
  else
  {
    vec2 screenUV = cameraPixelCoordinateToScreenUV(cb.camera, pixelCoordinate);
    skyContribution = cb.skyBrightness * SkyLight.SampleLevel(screenUV, 0) * volumeAttenuation;
  }

  if (cb.sparseRenderingArgs.mode != SparseRenderingMode::Off &&
      cb.sparseRenderingArgs.enableSparsePrimaryRayMissComposition)
  {
    skyContribution *= radianceScale;
  }

  radianceOutput += skyContribution;
}

// Aerial perspective of the segment a mirror (PSR) pixel reflects. The froxel volume is fitted to the
// camera frustum, which the reflected ray leaves, so this marches the atmosphere inline from the mirror's
// altitude. Like the camera segment it starts past the global volumetrics range, whose reflection fog
// already covers the near part of the segment. Identity when the segment ends inside that range.
void evalReflectedSegmentAerialPerspective(
  float3 mirrorPosition,
  float3 reflectionDirection,
  float segmentLength,
  out vec3 inScatter,
  out vec3 transmittance)
{
  inScatter = vec3(0.0f);
  transmittance = vec3(1.0f);

  const AtmosphereArgs args = aerialPerspectiveMediumArgs(cb.atmosphereArgs);
  const float kmPerWorldUnit = 1.0f / max(args.worldUnitsPerKilometer, 1e-6f);

  const float startDistanceKm = args.aerialPerspectiveStartDistance * kmPerWorldUnit;
  const float endDistanceKm = segmentLength * kmPerWorldUnit;
  if (endDistanceKm <= startDistanceKm)
  {
    return;
  }

  // Mirror altitude relative to the camera, along the world up axis.
  const float mirrorHeightKm = (args.isZUp
    ? mirrorPosition.z - args.cameraPosition.z
    : mirrorPosition.y - args.cameraPosition.y) * kmPerWorldUnit;
  const vec3 origin = vec3(0.0f, max(args.aerialPerspectiveViewAltitude + mirrorHeightKm, kPlanetRadiusOffset), 0.0f);
  const vec3 direction = aerialPerspectiveWorldToAtmosphereDirection(reflectionDirection, args);

  inScatter = evalAtmosphereRadianceSegment(
    origin, direction, args.sunDirection, args, AtmosphereTransmittanceLut, AtmosphereMultiscatteringLut,
    AtmosphereAerosolPhaseLut, 8, endDistanceKm, startDistanceKm, transmittance);
}

// Aerial perspective (paper Section 5.4). Skipped on primary misses, whose sky-view LUT radiance already
// carries the whole column. PSR pixels apply their mirror's reflected segment first, then the camera's.
// The volume starts past the froxel grid and is attenuated by the grid's transmittance; that the near
// in-scatter picks it up too is negligible at the grid's scale of tens of metres.
void applyAerialPerspective(
  uint2 pixelCoordinate,
  bool primaryMiss,
  float hitDistance,
  vec3 volumeAttenuation,
  // Reflected segment of a PSR pixel: 0 length when there is none (including reflected sky, which
  // already carries its full column). The throughput is the mirror reflectance its in-scatter is seen through.
  float3 mirrorPosition,
  float3 reflectionDirection,
  float reflectedSegmentLength,
  vec3 reflectedSegmentThroughput,
  inout vec3 radianceOutput,
  // Of the camera's segment, which all light from past the PSR surface is seen through.
  out float cameraSegmentTransmittance)
{
  vec3 inScatter = vec3(0.0f);
  float transmittance = 1.0f;
  cameraSegmentTransmittance = 1.0f;

  if (aerialPerspectiveActive() && !primaryMiss && hitDistance > 0.0f)
  {
    if (reflectedSegmentLength > 0.0f)
    {
      vec3 reflectedInScatter;
      vec3 reflectedTransmittance;
      evalReflectedSegmentAerialPerspective(
        mirrorPosition, reflectionDirection, reflectedSegmentLength, reflectedInScatter, reflectedTransmittance);

      const vec3 reflectedContribution = max(reflectedInScatter, vec3(0.0f)) * reflectedSegmentThroughput * volumeAttenuation;
      radianceOutput = radianceOutput * reflectedTransmittance + reflectedContribution;

      inScatter += reflectedContribution;
      transmittance *= (reflectedTransmittance.x + reflectedTransmittance.y + reflectedTransmittance.z) * (1.0f / 3.0f);
    }

    vec3 cameraInScatter;
    float cameraTransmittance;
    evalCameraSegmentAerialPerspective(pixelCoordinate, hitDistance, cameraInScatter, cameraTransmittance);
    cameraInScatter *= volumeAttenuation;
    cameraSegmentTransmittance = cameraTransmittance;

    radianceOutput = radianceOutput * cameraTransmittance + cameraInScatter;

    inScatter = inScatter * cameraTransmittance + cameraInScatter;
    transmittance *= cameraTransmittance;
  }

  switch (cb.debugViewIdx)
  {
  case DEBUG_VIEW_ATMOSPHERE_AERIAL_PERSPECTIVE_INSCATTER:
    storeInDebugView(pixelCoordinate, inScatter);
    break;
  case DEBUG_VIEW_ATMOSPHERE_AERIAL_PERSPECTIVE_TRANSMITTANCE:
    storeInDebugView(pixelCoordinate, transmittance);
    break;
  }
}

float applyFog(
  float viewDistance,
  float radianceScale,
  inout vec3 radianceOutput)
{
  if (cb.volumeArgs.enable)
  {
    return 1.0f; // Volumetric fog is applied in volume integration
  }

  // Todo: Using view distance is a bit of a hack here, the primary emissive contribution will not be fully weighted correctly
  // as emissive hits will be over-attenuated.
  vec4 fogColor = calculateFog(viewDistance);

  if (cb.sparseRenderingArgs.mode != SparseRenderingMode::Off)
  {
    fogColor.xyz *= radianceScale;
  }

  // Blend fog over output
  radianceOutput = lerp(radianceOutput, fogColor.rgb, fogColor.a);

  return 1.0f - fogColor.a;
}

// Sparse rendering scale for sky and fog in-scatter: 1 / rate at active sparse misses, 0 at inactive ones, otherwise 1.
float calculateRadianceScale(uint2 pixelCoordinate, bool pixelIsActive, bool primaryMiss)
{
  // Hits blend fog at every pixel, so only sparse misses need a scale.
  if (cb.sparseRenderingArgs.mode == SparseRenderingMode::Off ||
      !primaryMiss ||
      !cb.sparseRenderingArgs.enableSparsePrimaryRayMissComposition)
  {
    return 1.f;
  }

  if (!pixelIsActive)
  {
    // Inactive misses get no sky or fog in-scatter; active misses carry both at 1 / rate for ray reconstruction.
    return 0.f;
  }

  return calculateRadianceScaleFromSamplingRate(SparseRenderingPixelSamplingRate[pixelCoordinate]);
}

// Volumetric in-scatter is integrated at active pixels independently of sparse sky composition; full-rate volumetrics need no scale.
float calculateVolumetricRadianceScale(uint2 pixelCoordinate, bool pixelIsActive, bool primaryMiss)
{
  const bool sparseVolumetrics = primaryMiss
    ? bool(cb.sparseRenderingArgs.enableSparseVolumetricsPrimaryMiss)
    : bool(cb.sparseRenderingArgs.enableSparseVolumetricsPrimaryHit);
  if (!pixelIsActive || cb.sparseRenderingArgs.mode == SparseRenderingMode::Off || !sparseVolumetrics)
  {
    return 1.f;
  }

  return calculateRadianceScaleFromSamplingRate(SparseRenderingPixelSamplingRate[pixelCoordinate]);
}

// What a cloud layer along a direction makes of the sky a PSR miss took there, the sky-view LUT's alone
// (geometry_resolver.slangh): its own light and the sun's through it, less the sky it covers.
vec3 evalCloudOverMissSky(vec4 cloudLayer, vec3 worldDirection)
{
  const bool isZUp = cb.atmosphereArgs.isZUp != 0;
  const float transmittance = saturate(cloudLayer.a);
  const vec3 sky = sampleSkyViewLutRadiance(AtmosphereSkyViewLut, worldDirection, cb.atmosphereArgs, isZUp);
  const vec3 sun = evalSunDisk(worldDirection, AtmosphereTransmittanceLut, cb.atmosphereArgs, isZUp);
  return max(cloudLayer.rgb, vec3(0.0f)) + sun * transmittance - sky * (1.0f - transmittance);
}

vec4 compositeResult(
  uint2 pixelCoordinate,
  uvec2 compactedCoordinate,
  bool pixelIsActive,
  GeometryFlags geometryFlags,
  float primaryLinearViewZ,
  bool primaryMiss,
  out float controlMaskTransmittance)
{
  RNG randomState = createRNG(pixelCoordinate, cb.frameIdx, 0);

  // Load GBuffer Information.
  // Inactive pixels have no compacted slot.
  // The primary reads are therefore gated on activity, and the default flags of inactive pixels skip the secondary reads.
  vec3 primaryAttenuation = vec3(0);
  vec3 primaryAlbedo = vec3(0);
  vec3 primarySpecularAlbedo = vec3(0);
  float perceptualRoughness = 0.0f;
  vec3 secondaryAttenuation = vec3(0);
  vec3 secondaryAlbedo = vec3(0);
  vec3 secondarySpecularAlbedo = vec3(0);
  if (geometryFlags.secondarySurfaceMask)
  {
    secondaryAttenuation = r11g11b10ToColor(imageLoad(SecondaryAttenuation, ivec2(compactedCoordinate)));
    secondaryAlbedo = SecondaryAlbedo[compactedCoordinate].xyz;
    secondarySpecularAlbedo = SecondarySpecularAlbedo[compactedCoordinate].xyz;
  }
  if (pixelIsActive)
  {
    primaryAttenuation = r11g11b10ToColor(PrimaryAttenuation[compactedCoordinate]);
    primaryAlbedo = PrimaryAlbedo[compactedCoordinate].xyz;
    primarySpecularAlbedo = PrimarySpecularAlbedo[compactedCoordinate].xyz;
    perceptualRoughness = PrimaryVirtualWorldNormalPerceptualRoughness[compactedCoordinate].w;
  }

  // Reconstruct the primary ray direction and view hit distance for the current pixel

  const CameraDirections primaryRayDirections = cameraPixelCoordinateToDirection(cb.camera, pixelCoordinate);
  const vec3 viewPosition = cameraReconstructViewPosition(cb.camera, primaryRayDirections.viewDirection, primaryLinearViewZ);
  // Note: Should be equal to the virtual hit distance just without needing to read it (since we need the linear view Z in world space for
  // other calculations here anyways and can just use it to reconstruct this).
  // PSR replaces PrimaryLinearViewZ with virtual path length — do not use it for froxel distance.
  // Reflection PSR: dual-integrate using stashed first-bounce segment + direction.
  // Transmission-only: fog along camera→first-hit (no reflection stash).
  float viewDistance = length(viewPosition);
  float volumeHitDistance = viewDistance;
  float psrFirstHitDistance = -1.0f;
  float psrReflectionSegmentSigned = 0.0f;
  float3 psrReflectionDirection = float3(0.0f, 0.0f, 0.0f);
  const bool performedPsr = geometryFlags.performedAnyPSR() && !geometryFlags.secondarySurfaceMask;
  if (performedPsr)
  {
    const float firstHit = PsrFirstHitDistance[pixelCoordinate];
    if (firstHit > 0.0f)
    {
      volumeHitDistance = firstHit;
      viewDistance = firstHit;
      // Dual integrate only when reflection PSR stashed segment/direction (PSTR leaves those buffers uncleared).
      if (geometryFlags.performPSRR)
      {
        psrFirstHitDistance = firstHit;
        psrReflectionSegmentSigned = PsrReflectionSegmentLength[pixelCoordinate];
        psrReflectionDirection = signedOctahedralToSphereDirection(PsrReflectionDirection[pixelCoordinate]);
      }
    }
  }
  else if (pixelIsActive && !primaryMiss)
  {
    // A reflection PSR declined arrives through the indirect integrator, which stashed its first bounce
    // length. Only the length is stashed; since the segment is lit by a single value its direction enters
    // only through the density field, so the mirror direction serves a rough lobe.
    // The term is scaled by the primary specular albedo below, so the second froxel integration is skipped
    // where that reflectance is too small for the fog to show (most dielectrics away from grazing angles).
    // Only stashed with volumetrics on, the one case the fog needs it.
    const float indirectSegment = cb.volumeArgs.enable ? IndirectReflectionSegment[pixelCoordinate] : 0.0f;
    const float reflectionFogReflectance = max(primarySpecularAlbedo.x, max(primarySpecularAlbedo.y, primarySpecularAlbedo.z));
    if (indirectSegment != 0.0f && reflectionFogReflectance >= cb.volumeArgs.reflectionFogMinSpecularAlbedo)
    {
      const float3 shadingNormal =
        PrimaryVirtualWorldNormalPerceptualRoughness[pixelCoordinate].xyz * 2.0f - 1.0f;

      psrFirstHitDistance = viewDistance;
      psrReflectionSegmentSigned = indirectSegment;
      psrReflectionDirection = reflect(primaryRayDirections.worldDirection, normalize(shadingNormal));
    }
  }

  vec3 sharedRadiance;
  vec3 remodulatedTotalPrimaryRadiance;
  vec3 remodulatedTotalSecondaryRadiance;

  vec3 volumetricPreintegratedRadiance;
  vec3 volumeAttenuation;
  vec3 controlMaskVolumeAttenuation;
  vec3 alphaBlendOutput;
  float backgroundAlpha;
  vec3 radianceOutput;
  vec3 sharedFinalOutput;

  const float radianceScale = calculateRadianceScale(pixelCoordinate, pixelIsActive, primaryMiss);
  const float volumetricRadianceScale = calculateVolumetricRadianceScale(pixelCoordinate, pixelIsActive, primaryMiss);

  // Load Input Radiance
  loadInputRadiance(
    pixelCoordinate, compactedCoordinate, geometryFlags, pixelIsActive,
    primaryAlbedo, primarySpecularAlbedo, primaryAttenuation,
    secondaryAlbedo, secondarySpecularAlbedo, secondaryAttenuation,
    perceptualRoughness, primaryMiss,
    sharedRadiance,
    remodulatedTotalPrimaryRadiance, remodulatedTotalSecondaryRadiance);

  // Calculate volumetric lighting
  applyVolumetricLighting(
    pixelCoordinate, randomState, primaryRayDirections, pixelIsActive, primaryMiss, volumeHitDistance,
    radianceScale, volumetricRadianceScale, remodulatedTotalSecondaryRadiance, sharedRadiance,
    psrFirstHitDistance, psrReflectionSegmentSigned, psrReflectionDirection,
    performedPsr ? vec3(1.0f, 1.0f, 1.0f) : primarySpecularAlbedo, performedPsr,
    volumetricPreintegratedRadiance, volumeAttenuation, controlMaskVolumeAttenuation,
    alphaBlendOutput, backgroundAlpha, radianceOutput);

  // Calculate final combined radiance output
  // Note: The primary hit accumulated radiance value is not affected by the throughput (rather takes attenuation into account
  // while accumulating already), so it can be added in here without issue.
  radianceOutput += remodulatedTotalPrimaryRadiance * volumeAttenuation;

  // SharedRadiance on a primary HIT is pre-attenuated. On a MISS it holds sky without
  // froxel volumetrics yet — apply volumeAttenuation so sky and geometry share fog.
  // Soften under Physical Atmosphere as OD grows (fogSkyAttenuationPower).
  // PSR forces primaryMiss=false even for reflection sky; treat that like a miss here.
  const bool psrReflectionSkyMiss = geometryFlags.performPSRR
    && !geometryFlags.secondarySurfaceMask
    && psrReflectionSegmentSigned < 0.0f;
  // Primary misses only carry sky in SharedRadiance when the atmosphere supplied it; skybox and
  // dome light misses are composited later from the sky matte and must stay pre-attenuated.
  const bool primarySkyMiss = primaryMiss
    && cb.volumeArgs.physicalAtmosphereEnabled
    && !cb.domeLightArgs.active;
  const bool attenuateSharedAsSky = primarySkyMiss || psrReflectionSkyMiss;
  vec3 skyVolumeAttenuation = volumeAttenuation;
  if (attenuateSharedAsSky && cb.volumeArgs.physicalAtmosphereEnabled)
  {
    const float3 T = max(volumeAttenuation, vec3(1e-6f));
    const float avgT = (T.x + T.y + T.z) * (1.0f / 3.0f);
    const float opticalDepth = -log(avgT);
    const float softAmount = saturate(opticalDepth / 2.0f);
    const float skyPower = lerp(1.0f, max(cb.volumeArgs.fogSkyAttenuationPower, 1e-3f), softAmount);
    skyVolumeAttenuation = pow(T, skyPower);
  }
  radianceOutput += attenuateSharedAsSky ? (sharedRadiance * skyVolumeAttenuation) : sharedRadiance;

  // Applied before the sky so it only affects geometry; the sky already carries its full column.
  // Only PSR content is entirely reflected, so only it gets the reflected segment; a reflection the
  // indirect integrator produced shares its buffers with the surface's own lighting.
  const bool psrReflectedGeometry = performedPsr && geometryFlags.performPSRR && psrReflectionSegmentSigned > 0.0f;
  const float3 psrMirrorPosition = cameraGetWorldPosition(cb.camera) + primaryRayDirections.worldDirection * max(psrFirstHitDistance, 0.0f);
  float aerialPerspectiveTransmittance;
  applyAerialPerspective(
    pixelCoordinate, primaryMiss, volumeHitDistance, volumeAttenuation,
    psrMirrorPosition, psrReflectionDirection, psrReflectedGeometry ? psrReflectionSegmentSigned : 0.0f, primaryAttenuation,
    radianceOutput, aerialPerspectiveTransmittance);

  // The cloud layer lies behind the froxel fog and in front of whatever the pixel shows, the screen pass having
  // stopped each ray there, so the fog's in-scatter F stays on top: F + T_cloud (radiance - F) + T_fog C.
  // A PSR chain whose miss the screen pass marched instead (a mirror's or glass's first reflection, the sky straight
  // through glass) has its sky replaced in place: by the clouds and the sun through them, at the chain's
  // attenuation the PSR resolver stored and through the fog and air its sky was seen through.
  float cloudLayerTransmittance = 1.0f;
  if (cb.cloudArgs.enabled && cb.volumeArgs.physicalAtmosphereEnabled)
  {
    const bool cloudsMarchedPerPixel = (cb.cloudArgs.flags & CLOUD_FLAG_PSR_REFLECTIONS) != 0 && !cb.domeLightArgs.active;
    const bool psrTransmittedSky = cloudsMarchedPerPixel && geometryFlags.performPSTR
      && primaryLinearViewZ == cb.primaryDirectMissLinearViewZ;
    const vec4 cloudLayer = CloudLayer[pixelCoordinate];
    const vec3 fogInScatter = volumetricPreintegratedRadiance * volumetricRadianceScale;
    const vec3 fogTransmittance = attenuateSharedAsSky ? skyVolumeAttenuation : volumeAttenuation;

    if (cloudsMarchedPerPixel && psrReflectionSkyMiss)
    {
      radianceOutput += fogTransmittance * aerialPerspectiveTransmittance * primaryAttenuation
        * evalCloudOverMissSky(CloudReflection[pixelCoordinate], psrReflectionDirection);
    }

    // Glass's reflected sky, a miss of its secondary surface, which the fog leaves alone like the sky through it.
    if (cloudsMarchedPerPixel && geometryFlags.secondarySurfaceMask
      && PsrReflectionSegmentLength[pixelCoordinate] == cb.secondaryCombinedMissLinearViewZ)
    {
      const vec3 glassReflectionDirection = signedOctahedralToSphereDirection(PsrReflectionDirection[pixelCoordinate]);
      radianceOutput += aerialPerspectiveTransmittance * secondaryAttenuation
        * evalCloudOverMissSky(CloudReflection[pixelCoordinate], glassReflectionDirection);
    }

    if (psrTransmittedSky)
    {
      // The layer is the clouds behind the glass, which take no fog, as the sky the glass shows took none.
      radianceOutput += aerialPerspectiveTransmittance * primaryAttenuation
        * evalCloudOverMissSky(cloudLayer, primaryRayDirections.worldDirection);
    }
    else
    {
      cloudLayerTransmittance = saturate(cloudLayer.a);
      // Sparse misses carry their sky at the sampling rate's scale; the clouds over it follow.
      const float cloudRadianceScale = primaryMiss ? radianceScale : 1.0f;
      radianceOutput = fogInScatter + (radianceOutput - fogInScatter) * cloudLayerTransmittance
        + fogTransmittance * max(cloudLayer.rgb, vec3(0.0f)) * cloudRadianceScale;
    }
  }

  applySkyContribution(pixelCoordinate, primaryRayDirections, primaryMiss, radianceScale, volumeAttenuation, radianceOutput);

  // Apply fog
  const float fogTransmittance = applyFog(viewDistance, radianceScale, radianceOutput);
  controlMaskTransmittance = calcBt709Luminance(saturate(controlMaskVolumeAttenuation));
  controlMaskTransmittance *= fogTransmittance * backgroundAlpha * cloudLayerTransmittance;

  // Combine the alpha-blended foreground with the background before upscaling.
  return vec4(radianceOutput * backgroundAlpha + alphaBlendOutput, 1.f);
}

[shader("compute")]
[numthreads(COMPOSITE_THREAD_GROUP_WIDTH, COMPOSITE_THREAD_GROUP_HEIGHT, 1)]
void main(uint2 threadIndex : SV_DispatchThreadID, uint2 localIndex : SV_GroupThreadID)
{
  // Launched over the screen grid even with sparse rendering on,
  // because inactive pixels are still composited and the debug views below need every pixel.
  if (any(threadIndex >= cb.camera.resolution))
  {
    return;
  }

  uvec2 pixelCoordinate;
  uvec2 compactedCoordinate;
  const bool pixelIsActive = loadPixelCoordinate(
    threadIndex, ActiveLocalPixelCoords, TileActiveCounts,
    cb.camera.resolution, cb.sparseRenderingArgs,
    pixelCoordinate, compactedCoordinate);

  // Takes the resolved coordinates rather than the dispatch index, because the compaction table permutes pixels within each tile.
  preCompositeDebugView(pixelCoordinate, compactedCoordinate, pixelIsActive);

  // Load GBuffer Information.
  // Linear view Z is per pixel.
  // SharedFlags is compacted, so inactive pixels take default flags,
  // which have no PSR and no secondary surface and therefore also skip the secondary reads below.
  GeometryFlags geometryFlags = geometryFlagsDecode(uint16_t(0));
  if (pixelIsActive)
  {
    geometryFlags = geometryFlagsReadFromGBuffer(compactedCoordinate, SharedFlags);
  }
  const float primaryLinearViewZ = PrimaryLinearViewZ[pixelCoordinate];
  // Note: Relying on the same sentinel value outputted to the linear view Z which NRD relies on, so this is safe (and lets us not have to read
  // in the cone radius texture which typically contains the flag we use to determine this).
  bool primaryMiss = primaryLinearViewZ == cb.primaryDirectMissLinearViewZ
                     && !geometryFlags.performedAnyPSR();  // must not be PSR

  // Inactive pixels take default flags, so a PSR lobe that missed would pass the check above.
  if (primaryMiss && !pixelIsActive && usesPrimaryRayMissFlag(cb.sparseRenderingArgs))
  {
    primaryMiss = packedFlagGet(PrimarySurfaceFlags[pixelCoordinate], kPrimarySurfaceFlagPrimaryRayMiss);
  }

  // Written from composite (rather than the prepare RR pass) so the value is gated by sparse activity
  if (cb.writeRayReconstructionHitDistance)
  {
    float hitDistance = 0;
    if (pixelIsActive && !primaryMiss &&
        (cb.enableReSTIRGI || geometryFlags.firstSampledLobeIsSpecular))
    {
      hitDistance = PrimaryIndirectSpecularRadianceHitDistance[compactedCoordinate].a;
    }
    RayReconstructionHitDistanceOutput[pixelCoordinate] = hitDistance;
  }

  float controlMaskTransmittance;
  vec4 finalOutput = compositeResult(
    pixelCoordinate, compactedCoordinate, pixelIsActive, geometryFlags, primaryLinearViewZ, primaryMiss,
    controlMaskTransmittance);

  if (cb.enableDlssNrVolumetricControlMask)
  {
    // Fog and alpha-blended volumetrics carry no stable surface detail for DLSS-NR to enhance.
    // Transmittance preserves the surface controls in clear regions and suppresses them as fog becomes opaque.
    const vec4 surfaceControlMask = DlssNrControlMask[pixelCoordinate];
    DlssNrControlMask[pixelCoordinate] = vec4(surfaceControlMask.rgb * controlMaskTransmittance, 1.f);
  }

  FinalOutput[pixelCoordinate] = accumulate(pixelCoordinate, finalOutput, cb.accumulationArgs, true, AccumulatedFinalOutput);

  // Note: we only debug output for active pixels
  postCompositeDebugView(pixelCoordinate, geometryFlags, pixelIsActive);
}
