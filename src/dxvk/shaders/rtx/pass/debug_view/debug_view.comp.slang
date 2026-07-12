/*
* Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
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

//!variant debug_view.comp
//!>       ENABLE_DEBUG_VIEW_OPTIONAL_FEATURES=0

//!variant debug_view_using_optional_extensions.comp
//!>       ENABLE_DEBUG_VIEW_OPTIONAL_FEATURES=1

//!end-variants

// Loads debug view inputs, applies any on-load transformations
// and accumulates the result

#include "rtx/pass/debug_view/debug_view_args.h"
#include "rtx/pass/debug_view/debug_view_binding_indices.h"
#include "rtx/utility/debug_view_indices.h"
#include "rtx/utility/geometry_flags.slangh"
#include "rtx/utility/noise.slangh"
#include "rtx/utility/procedural_noise.slangh"
#include "rtx/algorithm/volume_composite_helpers.slangh"

#include "rtxdi/Reservoir.slangh"
#include "rtx/pass/nrc/nrc_utilities.slangh"
#include "rtx/algorithm/accumulate.slangh"

#include "rtx/concept/surface/minimal_surface_interaction.slangh"

uint32_t RAB_EncodeNormal(float3 normal)
{
  return float2x32ToSnorm2x16(sphereDirectionToSignedOctahedral(normal));
}

float3 RAB_DecodeNormal(uint encodedNormal)
{
  return signedOctahedralToSphereDirection(snorm2x16ToFloat2x32(encodedNormal));
}

// Inputs

[[vk::binding(DEBUG_VIEW_BINDING_CONSTANTS_INPUT)]]
ConstantBuffer<DebugViewArgs> cb;

[[vk::binding(DEBUG_VIEW_BINDING_DENOISED_PRIMARY_DIRECT_DIFFUSE_RADIANCE_HIT_T_INPUT)]]
Texture2D<float4> DenoisedPrimaryDirectDiffuseRadianceHitT;
[[vk::binding(DEBUG_VIEW_BINDING_DENOISED_PRIMARY_DIRECT_SPECULAR_RADIANCE_HIT_T_INPUT)]]
Texture2D<float4> DenoisedPrimaryDirectSpecularRadianceHitT;
[[vk::binding(DEBUG_VIEW_BINDING_DENOISED_SECONDARY_COMBINED_DIFFUSE_RADIANCE_HIT_T_INPUT)]]
Texture2D<float4> DenoisedSecondaryCombinedDiffuseRadianceHitT;
[[vk::binding(DEBUG_VIEW_BINDING_DENOISED_SECONDARY_COMBINED_SPECULAR_RADIANCE_HIT_T_INPUT)]]
Texture2D<float4> DenoisedSecondaryCombinedSpecularRadianceHitT;
[[vk::binding(DEBUG_VIEW_BINDING_SHARED_FLAGS_INPUT)]]
Texture2D<uint16_t> SharedFlags;
[[vk::binding(DEBUG_VIEW_BINDING_PRIMARY_LINEAR_VIEW_Z_INPUT)]]
Texture2D<float> PrimaryLinearViewZ;
[[vk::binding(DEBUG_VIEW_BINDING_PRIMARY_VIRTUAL_WORLD_SHADING_NORMAL_PERCEPTUAL_ROUGHNESS_INPUT)]]
Texture2D<float4> PrimaryVirtualWorldNormalPerceptualRoughness;
[[vk::binding(DEBUG_VIEW_BINDING_PRIMARY_SCREEN_SPACE_MOTION_VECTOR_INPUT)]]
Texture2D<float2> PrimaryScreenSpaceMotionVector;
[[vk::binding(DEBUG_VIEW_BINDING_RTXDI_CONFIDENCE_INPUT)]]
Texture2D<float4> RtxdiConfidence;
[[vk::binding(DEBUG_VIEW_BINDING_RENDER_OUTPUT_INPUT)]]
Texture2D<float4> RenderOutput;
[[vk::binding(DEBUG_VIEW_BINDING_INSTRUMENTATION_INPUT)]]
Texture2D<uint> Instrumentation;
[[vk::binding(DEBUG_VIEW_BINDING_TERRAIN_INPUT)]]
Texture2D<float4> TerrainTexture;

// Fork: cloud-occluded sky-ambient transmittance LUT (32x16 R16F).
// Visualized by DEBUG_VIEW_CLOUD_SKY_TRANSMITTANCE_LUT.
[[vk::binding(DEBUG_VIEW_BINDING_CLOUD_SKY_TRANSMITTANCE_LUT_INPUT)]]
Texture2D<float> CloudSkyTransmittanceLut;

// Fork: Nubis Cubed sun-direction optical-depth voxel grid (R16F 3D).
// Visualized by DEBUG_VIEW_CLOUD_D_SUN.
[[vk::binding(DEBUG_VIEW_BINDING_CLOUD_D_SUN_INPUT)]]
Texture3D<float> DebugViewCloudDSun;

// Fork: Nubis Cubed zenith optical-depth voxel grid (R16F 3D).
// Visualized by DEBUG_VIEW_CLOUD_D_AMBIENT.
[[vk::binding(DEBUG_VIEW_BINDING_CLOUD_D_AMBIENT_INPUT)]]
Texture3D<float> DebugViewCloudDAmbient;

// Fork: Nubis Cubed screen-space cloud render RT (RGBA16F, premultiplied
// radiance in rgb + cloud transmittance in alpha). Visualized by
// DEBUG_VIEW_CLOUD_RENDER_RT (enum 876).
[[vk::binding(DEBUG_VIEW_BINDING_CLOUD_RENDER_RT_INPUT)]]
Texture2D<float4> DebugViewCloudRenderRT;

// Fork: the post-denoise cumulus shadow factor texture and its debug view
// (enum 878) were removed 2026-06-19 with the screen-space cloud-shadow system.
// The cloud shadow now folds onto the sun radiance in the NEE; enum 875/877
// read the D_sun grid directly for cloud-on-terrain shadow diagnostics.

[[vk::binding(DEBUG_VIEW_BINDING_VOLUME_RESERVOIRS_INPUT)]]
Texture3D<uint4> VolumeReservoirs;

[[vk::binding(DEBUG_VIEW_BINDING_VOLUME_AGE_INPUT)]]
Sampler3D<float> VolumeFilteredRadianceAge;
[[vk::binding(DEBUG_VIEW_BINDING_VOLUME_RADIANCE_Y_INPUT)]]
Sampler3D<float4> VolumeFilteredRadianceY;
[[vk::binding(DEBUG_VIEW_BINDING_VOLUME_RADIANCE_COCG_INPUT)]]
Sampler3D<float2> VolumeFilteredRadianceCoCg;

[[vk::binding(DEBUG_VIEW_BINDING_VALUE_NOISE_SAMPLER)]]
Sampler3D<float4> ValueNoiseSampler;

[[vk::binding(DEBUG_VIEW_BINDING_BLUE_NOISE_TEXTURE)]]
Texture2DArray BlueNoise;

[[vk::binding(DEBUG_VIEW_BINDING_DEBUG_VIEW_INPUT)]]
Texture2D<float4> DebugView;

[[vk::binding(DEBUG_VIEW_BINDING_NRD_VALIDATION_LAYER_INPUT)]]
Texture2D<float4> NrdValidationLayer;

[[vk::binding(DEBUG_VIEW_BINDING_COMPOSITE_INPUT)]]
Texture2D<float4> Composite;

[[vk::binding(DEBUG_VIEW_BINDING_ALTERNATE_DISOCCLUSION_THRESHOLD_INPUT)]]
Texture2D<float> PrimaryDisocclusionThreshold;

layout(binding = DEBUG_VIEW_BINDING_PREV_WORLD_POSITION_INPUT)
Texture2D<float4> PreviousWorldPosition_WorldTriangleNormal;

[[vk::binding(DEBUG_VIEW_BINDING_SHARED_TERMINATOR_FIX_INPUT)]]
Texture2D<float4> SharedShadowTerminatorFix;

// Inputs / Outputs

[[vk::binding(DEBUG_VIEW_BINDING_ACCUMULATED_DEBUG_VIEW_INPUT_OUTPUT)]]
RWTexture2D<float4> AccumulatedDebugView; 

// Outputs

[[vk::binding(DEBUG_VIEW_BINDING_STATISTICS_BUFFER_OUTPUT)]]
RWStructuredBuffer<float> DebugViewStatistics;

// Samplers


// Samplers


// Samplers

[[vk::binding(DEBUG_VIEW_BINDING_NEAREST_SAMPLER)]]
SamplerState NearestSampler;

[[vk::binding(DEBUG_VIEW_BINDING_LINEAR_SAMPLER)]]
SamplerState LinearSampler;

#include "rtx/utility/common.slangh"
#include "rtx/utility/color.slangh"
#include "rtx/utility/packing.slangh"
#include "rtx/utility/noise.slangh"
#include "rtx/external/NRD.slangh"

// Fork: atmosphere helpers for cloud-diagnostic debug views (enum 875).
// Pulled in WITHOUT defining ATMOSPHERE_AVAILABLE — the helpers we need
// (sampleCloudGroundShadow_OptionB_explicit) take their D_sun texture +
// sampler as explicit parameters, so they don't rely on common-bindings
// globals. The ATMOSPHERE_AVAILABLE-gated sampleDSun / sampleDAmbient
// fall back to safe no-op stubs in this translation unit, which is
// correct — the enum-875 case uses the explicit-parameter variant.
#include "rtx/pass/atmosphere/atmosphere_common.slangh"

float4 sampleTexture(Texture2D<float4> texture, ivec2 threadId)
{
  const float2 uv = (threadId + 0.5f) / float2(cb.debugViewResolution);
  float4 sampledColor = float4(0.0f);

  // Note: sampledColor variable written to rather than returning from each of these cases directly as for
  // some reason directly causes a crash when attempting to execute this shader on AMD's (current) drivers
  // (may be fixed at some point though).
  switch(cb.samplerType)
  {
  case DebugViewSamplerType::Nearest:
    sampledColor = texture[threadId];
  case DebugViewSamplerType::NormalizedNearest:
    sampledColor = texture.SampleLevel(NearestSampler, uv, 0.0f);
  case DebugViewSamplerType::NormalizedLinear:
    sampledColor = texture.SampleLevel(LinearSampler, uv, 0.0f);
  }

  return sampledColor;
}

float nrdGetHitT(float normalizedHitT, float primaryHitPerceptualRoughness, float linearViewZ)
{
  if (cb.nrd.isReblurEnabled > 0)
  {
    return REBLUR_GetHitDist(normalizedHitT, linearViewZ, cb.nrd.hitDistanceParams, primaryHitPerceptualRoughness);
  }
  else
  {
    return normalizedHitT;
  }
}

vec4 nrdDenoisedHitTtoColor(ivec2 threadId)
{
  float normalizedHitT;
  float primaryHitPerceptualRoughness;

  switch (cb.debugViewIdx)
  {
  case DEBUG_VIEW_DENOISED_PRIMARY_DIRECT_DIFFUSE_HIT_T:
    normalizedHitT = DenoisedPrimaryDirectDiffuseRadianceHitT[threadId].w;
    primaryHitPerceptualRoughness = 1.0;
    break;
  
  case DEBUG_VIEW_DENOISED_PRIMARY_DIRECT_SPECULAR_HIT_T:
    normalizedHitT = DenoisedPrimaryDirectSpecularRadianceHitT[threadId].w;
    primaryHitPerceptualRoughness = PrimaryVirtualWorldNormalPerceptualRoughness[threadId].w;
    break;

  case DEBUG_VIEW_DENOISED_PRIMARY_INDIRECT_DIFFUSE_HIT_T:
    normalizedHitT = DebugView[threadId].w;
    primaryHitPerceptualRoughness = 1.0;
    break;

  case DEBUG_VIEW_DENOISED_PRIMARY_INDIRECT_SPECULAR_HIT_T:
    normalizedHitT = DebugView[threadId].w;
    primaryHitPerceptualRoughness = PrimaryVirtualWorldNormalPerceptualRoughness[threadId].w;
    break;
  }

  if (cb.nrd.isReblurEnabled > 0)
  {
    float linearViewZ = PrimaryLinearViewZ[threadId].x;
    return vec4(vec3(REBLUR_GetHitDist(normalizedHitT, linearViewZ, cb.nrd.hitDistanceParams, primaryHitPerceptualRoughness)), 1.0);
  }
  else
  {
    return vec4(vec3(normalizedHitT), 1.0);
  }
}

vec4 unormVectorToColor(vec3 unormVector)
{
  return vec4(unormVector * 2.0 - 1.0, 1.0); 
}

vec4 loadInput(inout RNG randomState, ivec2 threadId)
{
  vec4 value = vec4(0);
  const GeometryFlags geometryFlags = geometryFlagsReadFromGBuffer(threadId, SharedFlags);

  // Compute the coordinate with respect to the upscaled output
  // Note: This is used so textures (primarily just the final output) using the upscaled extent
  // rather than the downscaled extent can be displayed through the current debug view system.
  // This does mean that the image will be downscaled in a fairly poor nearest neighbor
  // way into the debug view texture, meaning when viewing the final output through the debug
  // view it is recommended to disable DLSS (or whatever other upscaler) until this is changed
  // to something better (the debug view texture ideally should be at the full output resolution).

  uvec2 upscaledResolution;
  RenderOutput.GetDimensions(upscaledResolution.x, upscaledResolution.y);

  const ivec2 upscaledPixelCoordinate = (threadId * upscaledResolution) / cb.debugViewResolution;

  // Generate noise values
  // Note: This is done here to avoid duplicating this logic for NaN/Inf checks. Compiler ideally should
  // optimize to only evaluate this when one of those two debug views is in use, but the debug view isn't
  // performance critical anyways so it's not a big deal if it doesn't.

  const float whiteNoiseValue = randomFloat(uvec3(threadId, cb.frameIdx));
  const float blueNoiseValue = getNextSampleBlueNoise(randomState);
  const float valueNoiseValue = ValueNoiseSource<false>.evaluateSignedNoise(float4(threadId, cb.animationTimeSec, cb.animationTimeSec + 100.0f) / 10.0f);
  const float fractalValueNoiseValue = fractalBrownianNoise<4, ValueNoiseSource<false>>(
    float4(threadId, cb.animationTimeSec, cb.animationTimeSec + 100.0f),
    3,
    0.05f,
    2.0f,
    0.5f,
    true
  );
  const float simplexNoiseValue = SimplexNoiseSource.evaluateSignedNoise(float3(threadId, cb.animationTimeSec) / 10.0f);
  const float fractalSimplexNoiseValue = fractalBrownianNoise<3, SimplexNoiseSource>(
    float3(threadId, cb.animationTimeSec * 5.0f),
    3,
    0.01f,
    2.0f,
    0.5f,
    true
  );

  // Get the respective input based on the debug index

  switch (cb.debugViewIdx)
  {
  case DEBUG_VIEW_VIRTUAL_SHADING_NORMAL:
    value = unormVectorToColor(PrimaryVirtualWorldNormalPerceptualRoughness[threadId].xyz);
    break;
  case DEBUG_VIEW_SCREEN_SPACE_MOTION_VECTOR:
    value = vec4(abs(PrimaryScreenSpaceMotionVector[threadId].xy), 0, 1);
    break;
  case DEBUG_VIEW_WHITE_NOISE:
    value = vec4(vec3(whiteNoiseValue), 1.0f);
    break;
  case DEBUG_VIEW_BLUE_NOISE:
    value = vec4(vec3(blueNoiseValue), 1.0f);
    break;
  case DEBUG_VIEW_VALUE_NOISE:
    value = vec4(vec3(snormF32ToUnormF32(valueNoiseValue)), 1.0f);
    break;
  case DEBUG_VIEW_FRACTAL_VALUE_NOISE:
    value = vec4(vec3(fractalValueNoiseValue), 1.0f);
    break;
  case DEBUG_VIEW_SIMPLEX_NOISE:
    value = vec4(vec3(snormF32ToUnormF32(simplexNoiseValue)), 1.0f);
    break;
  case DEBUG_VIEW_FRACTAL_SIMPLEX_NOISE:
    value = vec4(vec3(fractalSimplexNoiseValue), 1.0f);
    break;
  case DEBUG_VIEW_DENOISED_PRIMARY_DIRECT_DIFFUSE_RADIANCE:
    value = vec4(DenoisedPrimaryDirectDiffuseRadianceHitT[threadId].xyz, 1);
    break;
  case DEBUG_VIEW_DENOISED_PRIMARY_DIRECT_SPECULAR_RADIANCE:
    value = vec4(DenoisedPrimaryDirectSpecularRadianceHitT[threadId].xyz, 1);
    break;
  case DEBUG_VIEW_DENOISED_PRIMARY_INDIRECT_DIFFUSE_RADIANCE:
    value = vec4(DebugView[threadId].xyz, 1);
    break;
  case DEBUG_VIEW_DENOISED_PRIMARY_INDIRECT_SPECULAR_RADIANCE:
    value = vec4(DebugView[threadId].xyz, 1);
    break;
  case DEBUG_VIEW_DENOISED_SECONDARY_COMBINED_DIFFUSE_RADIANCE:
    value = vec4(DenoisedSecondaryCombinedDiffuseRadianceHitT[threadId].xyz, 1);
    break;
  case DEBUG_VIEW_DENOISED_SECONDARY_COMBINED_SPECULAR_RADIANCE:
    value = vec4(DenoisedSecondaryCombinedSpecularRadianceHitT[threadId].xyz, 1);
    break;
  case DEBUG_VIEW_DENOISED_PRIMARY_DIRECT_DIFFUSE_HIT_T:
  case DEBUG_VIEW_DENOISED_PRIMARY_DIRECT_SPECULAR_HIT_T:
  case DEBUG_VIEW_DENOISED_PRIMARY_INDIRECT_DIFFUSE_HIT_T:
  case DEBUG_VIEW_DENOISED_PRIMARY_INDIRECT_SPECULAR_HIT_T:
    value = nrdDenoisedHitTtoColor(threadId);
    break;
  case DEBUG_VIEW_PRIMARY_ALTERNATE_DISOCCLUSION_THRESHOLD:
    value = vec4(PrimaryDisocclusionThreshold[threadId].xxx, 1);
    break;
  case DEBUG_VIEW_PRE_TONEMAP_OUTPUT:
  case DEBUG_VIEW_POST_TONEMAP_OUTPUT:
  case DEBUG_VIEW_LOCAL_TONEMAPPER_FINAL_COMBINE_OUTPUT:
  case DEBUG_VIEW_LOCAL_TONEMAPPER_LUMINANCE_OUTPUT:
  case DEBUG_VIEW_LOCAL_TONEMAPPER_EXPOSURE_OUTPUT:
  case DEBUG_VIEW_LOCAL_TONEMAPPER_BLEND_OUTPUT:
    value = RenderOutput[upscaledPixelCoordinate];
    break;
  case DEBUG_VIEW_COMPOSITE_OUTPUT:
    value = Composite[threadId];
    break;
  case DEBUG_VIEW_VIEW_MODEL:
    if (geometryFlags.isViewModel)
    {
      value = RenderOutput[upscaledPixelCoordinate];
    }
    else
    {
      // Checkerboard background
      const bool state1 = ((threadId.x >> 3) & 1) ^ ((threadId.y >> 3) & 1);
      value = vec4(0.25, 0.25, 0.25, 1) + state1 * vec4(0.25, 0.25, 0.25, 0);
    }
    break;
  case DEBUG_VIEW_PSR_PRIMARY_SECONDARY_SURFACE_MASK:
    value = geometryFlags.primarySelectedIntegrationSurface ? vec4(1, 1, 1, 1) : vec4(0, 0, 0, 1);
    break;
  case DEBUG_VIEW_TERRAIN_MAP:
    value = vec4(sampleTexture(TerrainTexture, threadId).rgb, 1);
    break;
  case DEBUG_VIEW_TERRAIN_MAP_OPACITY:
    value = vec4(sampleTexture(TerrainTexture, threadId).aaa, 1);
    break;
  case DEBUG_VIEW_CLOUD_SKY_TRANSMITTANCE_LUT:
    {
      // Stretch the 32x16 LUT to fill the screen via linear sampling.
      // Heatmap: black = clear sky (transmittance 1.0), red = full cloud
      // occlusion (transmittance 0.0). X axis = azimuth, Y axis = elevation.
      const vec2 uv = (vec2(threadId) + 0.5f) / vec2(cb.debugViewResolution);
      const float trans = CloudSkyTransmittanceLut.SampleLevel(LinearSampler, uv, 0).r;
      const float occlusion = saturate(1.0f - trans);
      value = vec4(occlusion, 0.0f, 0.0f, 1.0f);
    }
    break;
  case DEBUG_VIEW_CLOUD_D_SUN:
    {
      // Sample the Nubis Cubed D_sun voxel grid at the mid-Z slice (w = 0.5);
      // map pixel UV directly to the horizontal grid UV. Grayscale: bright =
      // high sun-direction optical depth (deep cloud cover above this voxel),
      // dark = clear sky between voxel and sun. Cumulus cells should be
      // visible as cellular patterns.
      const vec2 uv = (vec2(threadId) + 0.5f) / vec2(cb.debugViewResolution);
      const vec3 uvw = vec3(uv, 0.5f);
      const float opticalDepth = DebugViewCloudDSun.SampleLevel(LinearSampler, uvw, 0).r;
      // Scale for visibility: typical optical depth in cumulus is 0..5.
      const float intensity = saturate(opticalDepth * 0.2f);
      value = vec4(intensity, intensity, intensity, 1.0f);
    }
    break;
  case DEBUG_VIEW_CLOUD_D_AMBIENT:
    {
      // Sample the Nubis Cubed D_ambient (zenith) voxel grid at the mid-Z
      // slice. Mostly uniform brightness expected: the zenith path is mostly
      // empty air above the cloud slab. Some banding where the slab is dense.
      const vec2 uv = (vec2(threadId) + 0.5f) / vec2(cb.debugViewResolution);
      const vec3 uvw = vec3(uv, 0.5f);
      const float opticalDepth = DebugViewCloudDAmbient.SampleLevel(LinearSampler, uvw, 0).r;
      const float intensity = saturate(opticalDepth * 0.2f);
      value = vec4(intensity, intensity, intensity, 1.0f);
    }
    break;
  case DEBUG_VIEW_CLOUD_GROUND_SHADOW_PRODSHAPE:
    {
      // Production-call-shape diagnostic for sampleCloudGroundShadow_OptionB.
      // This case calls the helper with the EXACT per-pixel call shape that
      // the upcoming NEE wiring (Task 6) will use: worldPos from the
      // G-buffer hit position, sunDir from atmosphereArgs.sunDirection,
      // and cb.isZUp != 0u for the coordinate-frame flag. The D_sun
      // texture and its sampler are plumbing parameters (the same pointer
      // for every pixel) — they are passed explicitly because the debug
      // view binds D_sun at a different slot from the production global,
      // but the call shape that varies per pixel (worldPos, sunDir, args,
      // isZUp) is identical to what production will use.
      //
      // Visual gate: compare against DEBUG_VIEW_CLOUD_D_SUN (enum 873) at
      // the same scene. If the cumulus patterns spatially agree, the
      // isZUp handling is correct. If they disagree, isZUp is mismatched
      // between the debug call path and the production call path — fix
      // before wiring NEE.

      // G-buffer worldPos at this pixel — same source the production NEE
      // entry points use (minimalSurfaceInteraction.position).
      GBufferMemoryMinimalSurfaceInteraction memory;
      memory.encodedWorldPositionWorldTriangleTBN =
          PreviousWorldPosition_WorldTriangleNormal[threadId];
      memory.positionError = 0;
      const MinimalSurfaceInteraction si = minimalSurfaceInteractionCreate(memory);
      const vec3 worldPos = si.position;

      // Sun direction. CRITICAL FRAME NOTE: cb.atmosphereArgs.sunDirection
      // is stored in CANONICAL Y-UP space (rtx_atmosphere.cpp:321-323:
      // sunDirection.y = sin(elevation)). Production NEE at
      // sampleAtmosphereSunLight pre-swaps Y-up → Z-up caller frame
      // (atmosphere_common.slangh:941) BEFORE calling the helper, which
      // re-swaps back to Y-up via its isZUp flag. Pre-swap to match;
      // skipping this was the cause of this case painting solid-white
      // in-game (helper bailed at slab-bottom intersect because a Y-up
      // vector was being treated as Z-up and double-swapped).
      const bool isZUpFlag = (cb.isZUp != 0u);
      const vec3 sunDirCallerFrame = isZUpFlag
          ? vec3(cb.atmosphereArgs.sunDirection.x,
                 cb.atmosphereArgs.sunDirection.z,
                 cb.atmosphereArgs.sunDirection.y)
          : cb.atmosphereArgs.sunDirection;
      const vec3 sunDir = normalize(sunDirCallerFrame);

      // CRITICAL: match production NEE call shape exactly.
      //   (worldPos, sunDir, atmosphereArgs, isZUp != 0u)
      // Any divergence here defeats the purpose of the diagnostic.
      const float shadow = sampleCloudGroundShadow_OptionB_explicit(
          worldPos, sunDir, cb.atmosphereArgs, isZUpFlag,
          DebugViewCloudDSun, LinearSampler);

      value = vec4(shadow, shadow, shadow, 1.0f);
    }
    break;
  case DEBUG_VIEW_CLOUD_RENDER_RT:
    {
      // Fork diagnostic. Visualizes the Nubis Cubed cloud render RT
      // produced by cloud_render.comp.slang (C4 of the 2026-05-12
      // workstream). The RT is at downscale extent; map pixel UV directly
      // to the texture via Load. Premultiplied radiance is shown as-is;
      // alpha (cloud transmittance) is ignored for the debug case — what
      // matters here is the per-pixel cloud color from the Nubis Cubed
      // lighting equations (top-bright / bottom-dark cumulus gradient,
      // silver lining at backlit edges, less-flat shadow side).
      //
      // Compare against the live sky (debug view off) to inspect the raw
      // cloud-render output before compositing. The cloud-render RT value is
      // the same scene as the live sky's cloud layer minus the sky-miss
      // composite.
      vec4 cloudSample = DebugViewCloudRenderRT.Load(int3(threadId, 0));
      value = vec4(cloudSample.rgb, 1.0f);
    }
    break;
  case DEBUG_VIEW_CLOUD_GROUND_SHADOW_RAW_OD:
    {
      // Fork diagnostic (2026-05-17). Sibling of enum 875 — same per-pixel
      // call shape (G-buffer worldPos + atmosphereArgs.sunDirection + cb.isZUp)
      // but the math stops at the dSunTex.SampleLevel call. NO exp(), NO mix.
      // Exists because 875 paints solid white in-game, and the post-exp/mix
      // collapse hides which of {OD~=0, cloudShadowStrength=0, sun below horizon,
      // surface above slab} is responsible. RGB encodes three independent
      // signals so one screenshot answers the question.
      //
      // Discrimination key (compare on flat terrain at midday):
      //   R uniform + B uniform        → scene-scale: every pixel reads same UVW
      //                                  (fix cloudVoxelGridExtentKm or
      //                                  worldPosKm → UVW conversion)
      //   R patterned, matches 873     → consumer works; exp/mix is killing it
      //                                  (raise cloudShadowMarchStrength or
      //                                  the bake's OD scale)
      //   R patterned but unrelated    → world-anchoring / slab-bottom bug
      //                                  (probe intersectCloudSlabBottomFromBelow
      //                                  and cloudVoxelWorldToUVW)
      //   R dark, G even darker        → magnitude underflow: bake OD tiny
      //
      // Math is duplicated (not refactored from sampleCloudGroundShadow_OptionB_impl)
      // so the diagnostic carries zero risk to the production helper it inspects.

      // G-buffer worldPos at this pixel — same source the production NEE
      // entry points use (minimalSurfaceInteraction.position).
      GBufferMemoryMinimalSurfaceInteraction memory;
      memory.encodedWorldPositionWorldTriangleTBN =
          PreviousWorldPosition_WorldTriangleNormal[threadId];
      memory.positionError = 0;
      const MinimalSurfaceInteraction si = minimalSurfaceInteractionCreate(memory);
      const vec3 worldPos = si.position;

      // Sun direction sourced from the atmosphere CB. CRITICAL FRAME NOTE:
      // args.sunDirection is stored in CANONICAL Y-UP space
      // (rtx_atmosphere.cpp:321-323: sunDirection.y = sin(elevation)). The
      // production NEE call at sampleAtmosphereSunLight pre-swaps Y-up →
      // Z-up caller frame (line 941) BEFORE passing it to the helper, which
      // then re-swaps back to Y-up via its isZUp flag. We skip that
      // round-trip here: sunDir is consumed in Y-up directly, so the
      // isZUp swap below applies to worldPos ONLY. Mishandling this was
      // the cause of enum 875 painting solid-white in-game (helper bailed
      // at the slab-bottom intersection because a Y-up vector was being
      // re-interpreted as Z-up and double-swapped).
      const bool isZUpFlag = (cb.isZUp != 0u);
      const vec3 sunDirYUp = normalize(cb.atmosphereArgs.sunDirection);

      // Replicates sampleCloudGroundShadow_OptionB_impl up through the
      // dSunTex.SampleLevel call. worldPos comes from the G-buffer in
      // caller's frame; swap to Y-up only if engine is Z-up.
      const vec3 worldPosYUp = isZUpFlag
          ? vec3(worldPos.x, worldPos.z, worldPos.y)
          : worldPos;
      const float kmPerWorldUnit = 1.0f / max(cb.atmosphereArgs.worldUnitsPerKm, 1e-6f);
      const vec3 worldPosKm = worldPosYUp * kmPerWorldUnit;
      const float slabTopY = cb.atmosphereArgs.cloudAltitude
                            + cb.atmosphereArgs.cloudVoxelGridVerticalKm;

      vec3 outColor;
      if (worldPosKm.y > slabTopY) {
        // Surface above slab — paint magenta so this branch is visually
        // distinct from "lookup ran but returned ~zero."
        outColor = vec3(1.0f, 0.0f, 1.0f);
      } else {
        float tToEntry;
        const bool reached = intersectCloudSlabBottomFromBelow(
            worldPosKm, sunDirYUp, cb.atmosphereArgs, tToEntry);
        if (!reached) {
          // Sun below horizon or otherwise unreachable — paint blue.
          outColor = vec3(0.0f, 0.0f, 1.0f);
        } else {
          const vec3 cloudEntryPosKm = worldPosKm + tToEntry * sunDirYUp;
          const vec3 uvw = cloudVoxelWorldToUVW(cloudEntryPosKm,
                                                 cb.atmosphereArgs,
                                                 /*isZUp=*/false);
          const float opticalDepth = DebugViewCloudDSun.SampleLevel(LinearSampler, uvw, 0).r;
          outColor.r = saturate(opticalDepth * 0.2f);
          outColor.g = saturate(opticalDepth);
          outColor.b = uvw.x;
        }
      }
      value = vec4(outColor, 1.0f);
    }
    break;
  // DEBUG_VIEW_CLOUD_SHADOW_FACTOR_RAW (enum 878) removed 2026-06-19 with the
  // screen-space cloud-shadow system. Use enum 875/877 (D_sun grid reads) for
  // cloud-on-terrain shadow diagnostics.
  case DEBUG_VIEW_NRC_UPDATE_RADIANCE:
  case DEBUG_VIEW_NRC_UPDATE_THROUGHPUT:
  case DEBUG_VIEW_NRC_UPDATE_RADIANCE_MULTIPLIED_BY_THROUGHPUT:
  case DEBUG_VIEW_NRC_UPDATE_IS_UNBIASED:
  case DEBUG_VIEW_NRC_UPDATE_NUMBER_OF_BOUNCES:
  case DEBUG_VIEW_NRC_UPDATE_NUMBER_OF_PATH_SEGMENTS:
  case DEBUG_VIEW_NRC_UPDATE_NUMBER_OF_INDIRECT_PATH_SEGMENTS:
    value = DebugView[Nrc::calculateTrainingPixelCoordinate(threadId, cb.nrcArgs)];
    break;

  case DEBUG_VIEW_EXPOSURE_HISTOGRAM:
    // Show histogram or weight as a bar across the top on the screen, technically not safe, but the first handful of threads should be fine.
    if (threadId.y < 32 && threadId.x < cb.debugViewResolution.x * 0.5)
    {
      const uint histogramBucket = (float(threadId.x) / (cb.debugViewResolution.x * 0.5)) * 256;
      vec4 histogramOrWeight = DebugView[uint2(histogramBucket, 0)];
      if (threadId.y < 16)
      {
        // Show weight
        value = float4(0,histogramOrWeight.yz,0);
      }
      else
      {
        // Show weighted histogram
        value = histogramOrWeight.y > 0 ? float4(0,1,0,0) : histogramOrWeight.xxxx;
      }
      
      if (histogramBucket == 0 && value.x > 0)
      {
        value = float4(1, 0, 0, 0); // Red, we have pixels that are in the ignore bucket.
      }
    }
    else
    {
      value = RenderOutput[upscaledPixelCoordinate];
    }
    break;
  case DEBUG_VIEW_INSTRUMENTATION_THREAD_DIVERGENCE:
    value = float4(Instrumentation[threadId], 0, 0, 0);
    break;
  case DEBUG_VIEW_RTXDI_CONFIDENCE:
    if (cb.isRTXDIConfidenceValid) 
    {
      const float confidence = saturate(1.0 - RtxdiConfidence[threadId].x);
      const float confidenceAlpha = pow(confidence, 0.5);
      const float3 confidenceColor = turboColormap(confidence);
      const float3 sceneColor = RenderOutput[upscaledPixelCoordinate].rgb;
      value.rgb = lerp(sceneColor, confidenceColor, confidenceAlpha);
      value.a = 1.0;
    }
    break;
  case DEBUG_VIEW_VOLUME_RESERVOIR_DEPTH_LAYERS: {
    const Camera camera = cb.camera;
    VolumeArgs volumeArgs = cb.volumeArgs;
    const uvec3 froxelDimensions = uvec3(volumeArgs.restirFroxelGridDimensions, volumeArgs.restirFroxelDepthSlices);
      
    vec2 screenUV = cameraPixelCoordinateToScreenUV(camera, threadId.xy);
    screenUV.x *= cb.volumeArgs.numFroxelVolumes;
    const float froxelVolume = floor(screenUV.x);
    screenUV.x = frac(screenUV.x);
    const uint widthTiles = ceil(sqrt(float(volumeArgs.restirFroxelDepthSlices)));
    const uint heightTiles = ceil(float(volumeArgs.restirFroxelDepthSlices) / widthTiles);
    const uint previewLevel = uint(screenUV.x * widthTiles) + uint(screenUV.y * heightTiles) * widthTiles;

    VolumeReSTIR_Reservoir reservoir = VolumeReSTIR_Reservoir::createEmpty();

    if (previewLevel < volumeArgs.restirFroxelDepthSlices)
    {
      vec2 previewUV = mod(screenUV, vec2(1.0f / widthTiles, 1.0f / heightTiles)) * vec2(widthTiles, heightTiles);
      previewUV.x = clamp(previewUV.x, volumeArgs.minFilteredRadianceU, volumeArgs.maxFilteredRadianceU);
      previewUV.x = (previewUV.x + froxelVolume) * volumeArgs.inverseNumFroxelVolumes;
      const float previewW = float(previewLevel) / float(volumeArgs.restirFroxelDepthSlices);

      VolumeReSTIR_PackedReservoir packedReservoir;
      packedReservoir.data0 = VolumeReservoirs[uint3(vec3(previewUV, previewW) * froxelDimensions)];
      reservoir = VolumeReSTIR_Reservoir::createFromPacked(packedReservoir);
      if(cb.debugKnob.x <= 1.f)
      {
        value = reservoir.sampleCount;
      }
      else if(cb.debugKnob.x <= 2.f)
      {
        value = reservoir.lightSample.lightIndex;
      }
      else if(cb.debugKnob.x <= 3.f)
      {
        value = reservoir.weightSum;
      }
      else if(cb.debugKnob.x <= 4.f)
      {
        value = reservoir.lightSample.targetPdf;
      }
      else if(cb.debugKnob.x <= 5.f)
      {
        value = reservoir.knownVisible ? float4(0, 1, 0, 1) : float4(1, 0, 0, 1);
      }
      else 
      {
        value.xy = float2(reservoir.lightSample.lightSampleCoordinates);
      }
    }

    break;
  }
  case DEBUG_VIEW_VOLUME_PREINTEGRATION: {
    const Camera camera = cb.camera;
    const VolumeArgs volumeArgs = cb.volumeArgs;
      
    const vec2 screenUV = cameraPixelCoordinateToScreenUV(camera, threadId.xy);
    const CameraDirections primaryRayDirections = cameraPixelCoordinateToDirection(camera, threadId.xy);
    
    // Calculate volumetric radiance and attenuation if volumetric lighting is enabled

    vec3 volumetricPreintegratedRadiance = vec3(0.0f, 0.0f, 0.0f);
    vec3 volumeAttenuation = vec3(1.0f, 1.0f, 1.0f);
    float volumetricIntegrationOffset = 0.f;
    integrateVolumetricNEE(
      randomState, VolumeFilteredRadianceAge, VolumeFilteredRadianceY, VolumeFilteredRadianceCoCg,
      volumeArgs, cb.animationTimeSec * 1000, threadId.xy, screenUV, false, cameraGetWorldPosition(camera), volumeArgs.froxelMaxDistance, primaryRayDirections.worldDirection,
      volumetricPreintegratedRadiance, volumeAttenuation);
    value.xyz = volumetricPreintegratedRadiance;
    break;
  }
  case DEBUG_VIEW_SCROLLING_LINE:
    value = RenderOutput[upscaledPixelCoordinate];
    if (threadId.x == (cb.frameIdx + 10) % cb.debugViewResolution.x)
    {
      value = float4(0, 1, 1, 1);
    }
    break;
  case DEBUG_VIEW_PREV_WORLD_POSITION_AND_TBN:
    {
      GBufferMemoryMinimalSurfaceInteraction memory;
      memory.encodedWorldPositionWorldTriangleTBN = PreviousWorldPosition_WorldTriangleNormal[threadId];
      memory.positionError = 0;

      MinimalSurfaceInteraction si = minimalSurfaceInteractionCreate(memory);
      
      if(cb.debugKnob.x <= 1.f)
      {
        value.xyz = si.position;
      }
      else if(cb.debugKnob.x <= 2.f)
      {
        value.xyz = normalize(float3(si.geometryNormal));
      }
      else if(cb.debugKnob.x <= 3.f)
      {
        value.xyz = normalize(float3(si.geometryTangent));
      }
      else if(cb.debugKnob.x <= 4.f)
      {
        value.xyz = normalize(float3(si.geometryBitangent));
      }
      break;
    }
  case DEBUG_VIEW_SHADOW_TERMINATOR_OFFSET:
    value = SharedShadowTerminatorFix[threadId];
    break;
  case DEBUG_VIEW_NAN:
    bool isValid = DebugView[threadId].x != 0.0;

    // DEBUG_VIEW_VIRTUAL_SHADING_NORMAL
    isValid &= isValidValue(unormVectorToColor(PrimaryVirtualWorldNormalPerceptualRoughness[threadId].xyz));
    // DEBUG_VIEW_SCREEN_SPACE_MOTION_VECTOR
    isValid &= isValidValue(PrimaryScreenSpaceMotionVector[threadId].xy);

    // DEBUG_VIEW_WHITE_NOISE
    isValid &= isValidValue(whiteNoiseValue);
    // DEBUG_VIEW_BLUE_NOISE
    isValid &= isValidValue(blueNoiseValue);
    // DEBUG_VIEW_FRACTAL_VALUE_NOISE
    isValid &= isValidValue(fractalValueNoiseValue);
    // DEBUG_VIEW_FRACTAL_SIMPLEX_NOISE
    isValid &= isValidValue(fractalSimplexNoiseValue);
    // Skipped (Should be covered by fractal noise valid value checks)
    // DEBUG_VIEW_SIMPLEX_NOISE
    // DEBUG_VIEW_VALUE_NOISE

    // DEBUG_VIEW_DENOISED_PRIMARY_DIRECT_DIFFUSE_RADIANCE
    isValid &= isValidValue(DenoisedPrimaryDirectDiffuseRadianceHitT[threadId].xyz);
    // DEBUG_VIEW_DENOISED_PRIMARY_DIRECT_SPECULAR_RADIANCE
    isValid &= isValidValue(DenoisedPrimaryDirectSpecularRadianceHitT[threadId].xyz);
    // DEBUG_VIEW_DENOISED_PRIMARY_INDIRECT_DIFFUSE_RADIANCE
    // DEBUG_VIEW_DENOISED_PRIMARY_INDIRECT_SPECULAR_RADIANCE
    isValid &= isValidValue(DebugView[threadId].xyz);
    // DEBUG_VIEW_DENOISED_SECONDARY_COMBINED_DIFFUSE_RADIANCE
    isValid &= isValidValue(DenoisedSecondaryCombinedDiffuseRadianceHitT[threadId].xyz);
    // DEBUG_VIEW_DENOISED_SECONDARY_COMBINED_SPECULAR_RADIANCE
    isValid &= isValidValue(DenoisedSecondaryCombinedSpecularRadianceHitT[threadId].xyz);

    float linearViewZ = PrimaryLinearViewZ[threadId].x;
    // DEBUG_VIEW_DENOISED_PRIMARY_DIRECT_DIFFUSE_HIT_T
    isValid &= isValidValue(nrdGetHitT(DenoisedPrimaryDirectDiffuseRadianceHitT[threadId].w, 1.0, linearViewZ));
    // DEBUG_VIEW_DENOISED_PRIMARY_DIRECT_SPECULAR_HIT_T
    isValid &= isValidValue(nrdGetHitT(DenoisedPrimaryDirectSpecularRadianceHitT[threadId].w, PrimaryVirtualWorldNormalPerceptualRoughness[threadId].w, linearViewZ));
    // DEBUG_VIEW_DENOISED_PRIMARY_INDIRECT_DIFFUSE_HIT_T
    // DEBUG_VIEW_DENOISED_PRIMARY_INDIRECT_SPECULAR_HIT_T
    isValid &= isValidValue(nrdGetHitT(DebugView[threadId].w, 1.0, linearViewZ));
    // DEBUG_VIEW_PRIMARY_ALTERNATE_DISOCCLUSION_THRESHOLD:
    isValid &= isValidValue(PrimaryDisocclusionThreshold[threadId]);
    // DEBUG_VIEW_PRE_TONEMAP_OUTPUT
    // DEBUG_VIEW_POST_TONEMAP_OUTPUT
    // DEBUG_VIEW_VIEW_MODEL
    // DEBUG_VIEW_LOCAL_TONEMAPPER_FINAL_COMBINE_OUTPUT
    // DEBUG_VIEW_LOCAL_TONEMAPPER_LUMINANCE_OUTPUT
    // DEBUG_VIEW_LOCAL_TONEMAPPER_EXPOSURE_OUTPUT
    // DEBUG_VIEW_LOCAL_TONEMAPPER_BLEND_OUTPUT
    isValid &= isValidValue(RenderOutput[upscaledPixelCoordinate]);
    // DEBUG_VIEW_COMPOSITE_OUTPUT
    isValid &= isValidValue(Composite[threadId]);

    // DEBUG_VIEW_SHADOW_TERMINATOR_OFFSET
    isValid &= isValidValue(SharedShadowTerminatorFix[threadId]);

    // Skipped
    // DEBUG_VIEW_VIEW_MODEL
    // DEBUG_VIEW_PSR_PRIMARY_SECONDARY_SURFACE_MASK
    // DEBUG_VIEW_TERRAIN_MAP
    // DEBUG_VIEW_TERRAIN_MAP_OPACITY

    // Skipped - since they read in data from DebugView
    // DEBUG_VIEW_NRC_UPDATE_RADIANCE
    // DEBUG_VIEW_NRC_UPDATE_THROUGHPUT
    // DEBUG_VIEW_NRC_UPDATE_RADIANCE_MULTIPLIED_BY_THROUGHPUT
    // DEBUG_VIEW_NRC_UPDATE_IS_UNBIASED
    // DEBUG_VIEW_NRC_UPDATE_NUMBER_OF_BOUNCES
    // DEBUG_VIEW_NRC_UPDATE_NUMBER_OF_PATH_SEGMENTS
    // DEBUG_VIEW_NRC_UPDATE_NUMBER_OF_INDIRECT_PATH_SEGMENTS

    // Skipped
    // DEBUG_VIEW_EXPOSURE_HISTOGRAM

    // DEBUG_VIEW_INSTRUMENTATION_THREAD_DIVERGENCE
    isValid &= isValidValue(Instrumentation[threadId]);
    
    // DEBUG_VIEW_RTXDI_CONFIDENCE
    if (cb.isRTXDIConfidenceValid) 
    {
      isValid &= isValidValue(RtxdiConfidence[threadId].x);
    }

    // Skipped
    // DEBUG_VIEW_VOLUME_RESERVOIR_DEPTH_LAYERS
    // DEBUG_VIEW_VOLUME_PREINTEGRATION
    // DEBUG_VIEW_SCROLLING_LINE
    
    // Set value to true when it is invalid so that 0 represents valid value.
    // This allows the overlay to print over the invalid pixels only.
    value = isValid ? 0 : 1;

    break;

  case DEBUG_VIEW_SSS_DIFFUSION_PROFILE_SAMPLING:
  {
    value = DebugView[threadId] + RenderOutput[upscaledPixelCoordinate];
    break;
  }
  case DEBUG_VIEW_NRD_INSTANCE_0_VALIDATION_LAYER:
  case DEBUG_VIEW_NRD_INSTANCE_1_VALIDATION_LAYER:
  case DEBUG_VIEW_NRD_INSTANCE_2_VALIDATION_LAYER:
  {
    float4 validation = NrdValidationLayer[threadId];
    value.xyz = lerp( RenderOutput[upscaledPixelCoordinate].xyz, validation.xyz, validation.w );

    break;
  }
  default:
    value = DebugView[threadId];
    break;
  }

  // Quantize the input value to the requested step size if enabled
  // Note: This is done to allow for simulation of what the result roughly would look like if written to a quantized texture (e.g. to simulate 8 bits per channel,
  // quantize with a step size of 1/255) primarily to visualize banding artifacts better without the often times larger floating point precision of the input
  // textures hiding them.

  if (cb.enableInputQuantization)
  {
    value = round(value * cb.quantizationInverseStepSize) * cb.quantizationStepSize;
  }

  return value;
}

static const uint32_t kDebugViewThreadGroupSize = DEBUG_VIEW_THREAD_GROUP_SIZE_WIDTH * DEBUG_VIEW_THREAD_GROUP_SIZE_HEIGHT;
static const uint32_t kDebugViewMinWaveSize = 8; // Value to support all hardware. Wave size can be as low as 8 on some hardware
static const uint32_t kDebugViewMaxWavesPerGroup = (kDebugViewThreadGroupSize + kDebugViewMinWaveSize - 1) / kDebugViewMinWaveSize;

// Workgroup-level partial sums for statistics reduction (avoids per-pixel global atomics)
#if ENABLE_DEBUG_VIEW_OPTIONAL_FEATURES
// Note: storing float4 for cleaner implementation because reducing the bank conflicts due to it makes zero difference
groupshared float4 s_wavePartials[kDebugViewMaxWavesPerGroup];
#endif

void storeDebugViewOutputStatistics(vec4 value, uint groupIndex)
{
  if (!cb.calculateStatistics)
  {
    return;
  }

#if ENABLE_DEBUG_VIEW_OPTIONAL_FEATURES
  // First level: reduce within each wave using intrinsics (no atomics needed)
  float4 waveSum = float4(WaveActiveSum(value.x), WaveActiveSum(value.y),
                          WaveActiveSum(value.z), WaveActiveSum(value.w));

  if (WaveIsFirstLane())
  {
    const uint waveIndex = groupIndex / WaveGetLaneCount();
    s_wavePartials[waveIndex] = waveSum;
  }

  GroupMemoryBarrierWithGroupSync();

  // Second level: thread 0 sums the per-wave partials and does a single global atomic
  if (groupIndex == 0)
  {
    float4 groupSum = float4(0.0f);
    const uint numWaves = (kDebugViewThreadGroupSize + WaveGetLaneCount() - 1) / WaveGetLaneCount();
    for (uint i = 0; i < numWaves; i++)
    {
      groupSum += s_wavePartials[i];
    }

    InterlockedAddFloat(DebugViewStatistics[0], groupSum.x);
    InterlockedAddFloat(DebugViewStatistics[1], groupSum.y);
    InterlockedAddFloat(DebugViewStatistics[2], groupSum.z);
    InterlockedAddFloat(DebugViewStatistics[3], groupSum.w);
  }
#endif
}

[shader("compute")]
[numthreads(DEBUG_VIEW_THREAD_GROUP_SIZE_WIDTH, DEBUG_VIEW_THREAD_GROUP_SIZE_HEIGHT, 1)]
void main(uint2 threadId : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
  // Out-of-bounds threads must still participate in the workgroup reduction barrier
  vec4 value = vec4(0.0f);

  if (all(threadId < cb.debugViewResolution))
  {
    RNG randomState = createRNG(threadId, cb.frameIdx, 0);
    value = loadInput(randomState, threadId);

    const bool storeInAccumulationBuffer = true;
    value = accumulate(threadId, value, cb.accumulationArgs, storeInAccumulationBuffer, AccumulatedDebugView);
  }

  storeDebugViewOutputStatistics(value, groupIndex);
}
