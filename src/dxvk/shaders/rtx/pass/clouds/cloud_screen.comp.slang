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

// The cloud layer along every camera ray, at render resolution, after the path tracer: each ray is marched only
// up to the scene it meets (on mirrors, the mirror), so composite lays the layer over the scene exactly, with no
// depth test of its own. Glass the sky is seen straight through lets the ray on, and composite lays those clouds
// into the light the glass transmits. Output is the premultiplied layer (rgb) and transmittance (a), see
// cloud_march. Mirrors' and glass's reflections into the sky get a layer of their own along the reflected ray.
//
// The march jitters by spatio-temporal blue noise; a reprojected history, clamped to the 3x3 neighbourhood of
// the current frame, resolves it before DLSS sees it: evenly over the frames a young history holds, then at the
// tier's blend weight. That weight keeps a long history: a sunlit face's light falls off within its first optical
// depth, where the march's steps are long, and under drifting clouds the upscaler no longer averages that grain
// away but shows it as a slow fizzle. A pixel whose ray the scene stops short of every cloud takes part in
// neither, or its empty layer would outline a facade's silhouette with sky. The history marks such pixels
// (transmittance -1), and composite reads a copy of the layer without the marks. Where clouds cover the sky the
// pass also writes the upscalers' motion vectors with the clouds' own motion.

#include "rtx/pass/raytrace_args.h"
#include "rtx/pass/clouds/cloud_march.slangh"
#include "rtx/pass/sparse_rendering/compact_active_pixels.slangh"
#include "rtx/utility/geometry_flags.slangh"
#include "rtx/utility/packing.slangh"
#include "rtx/utility/debug_view_indices.h"

layout(binding = CLOUD_BINDING_CAMERA)
ConstantBuffer<RaytraceArgs> cb;

layout(binding = CLOUD_BINDING_BLUE_NOISE)
Texture2DArray BlueNoise;

layout(r32f, binding = CLOUD_BINDING_PRIMARY_LINEAR_VIEW_Z)
Texture2D<float> PrimaryLinearViewZ;

layout(r32f, binding = CLOUD_BINDING_PSR_FIRST_HIT_DISTANCE)
Texture2D<float> PsrFirstHitDistance;

// A mirror's first reflection, which the PSR resolver stashes for the fog: its length (negative when it reached
// the sky) and its direction.
layout(binding = CLOUD_BINDING_PSR_REFLECTION_SEGMENT)
Texture2D<float> PsrReflectionSegmentLength;

layout(binding = CLOUD_BINDING_PSR_REFLECTION_DIRECTION)
Texture2D<float2> PsrReflectionDirection;

// Composite scales a mirror's reflected clouds by the primary attenuation and glass's by the secondary.
layout(binding = CLOUD_BINDING_PRIMARY_ATTENUATION)
Texture2D<uint> PrimaryAttenuation;

layout(binding = CLOUD_BINDING_SECONDARY_ATTENUATION)
Texture2D<uint> SecondaryAttenuation;

// Under sparse rendering the PSR outputs above sit at an active pixel's compacted slot, which these resolve.
layout(binding = CLOUD_BINDING_COMPACTED_PIXEL_INDICES)
Texture2D<uint16_t> CompactedPixelIndices;

layout(binding = CLOUD_BINDING_TILE_ACTIVE_COUNTS)
Texture2D<uint2> TileActiveCounts;

// Read by pixel: the compacted GBuffer's dense copy, which inactive pixels have too.
layout(r16ui, binding = CLOUD_BINDING_SHARED_FLAGS)
Texture2D<uint16_t> SharedFlags;

layout(binding = CLOUD_BINDING_HISTORY)
Texture2D<float4> CloudLayerHistory;

layout(binding = CLOUD_BINDING_HISTORY_AGE)
Texture2D<float> CloudLayerHistoryAge;

layout(binding = CLOUD_BINDING_DOME_LUT)
Texture2D<float4> CloudDome;

layout(binding = CLOUD_BINDING_SHADOW_MAP)
Texture2D<float> CloudShadowMap;

layout(binding = CLOUD_BINDING_LAYER_OUTPUT, rgba16f)
RWTexture2D<float4> CloudLayerOutput;

layout(binding = CLOUD_BINDING_COMPOSITE_LAYER_OUTPUT, rgba16f)
RWTexture2D<float4> CloudCompositeLayerOutput;

layout(binding = CLOUD_BINDING_HISTORY_AGE_OUTPUT, r16f)
RWTexture2D<float> CloudLayerHistoryAgeOutput;

layout(binding = CLOUD_BINDING_REFLECTION_OUTPUT, rgba16f)
RWTexture2D<float4> CloudReflectionOutput;

layout(binding = CLOUD_BINDING_REFLECTION_HISTORY)
Texture2D<float4> CloudReflectionHistory;

layout(binding = CLOUD_BINDING_DEBUG_VIEW)
RWTexture2D<float4> CloudDebugView;

// The G-buffer's motion vectors, which over the sky are the camera's alone.
layout(binding = CLOUD_BINDING_MOTION_VECTOR_OUTPUT, rg16f)
RWTexture2D<float2> PrimaryScreenSpaceMotionVector;

layout(binding = CLOUD_BINDING_MOTION_VECTOR_RR_OUTPUT, rg16f)
RWTexture2D<float2> PrimaryScreenSpaceMotionVectorDLSSRR;

groupshared vec4 tileLayer[8][8];
// Each pixel's mean cloud depth this frame, weighted by the cloud it met (0 where it met none).
groupshared vec2 tileDepth[8][8];
// The layer along each pixel's reflection into the sky, marked where it reflects none.
groupshared vec4 tileReflection[8][8];

// Transmittance that marks a pixel the scene hid the clouds behind, in the history and the tile.
static const float kHiddenMark = -1.0f;

// Frames of history past which the age is not counted: well past where the tiers' blend weights take over.
static const float kMaxHistoryAge = 64.0f;

// A reflection whose throughput is at most this marches at cloudReflectionMarchSettings,
// which near-mirrors use at full weight in cloud_glossy.comp.slang.
// Composite scales the layer by the throughput, which dims the coarser march's grain with it.
// Glass short of grazing falls below this, while mirrors keep the view's settings.
static const float kDimReflectionThroughput = 0.5f;

// A history at uv, bilinearly over the texels that carry a layer, and with withAge the frames it holds there.
// weight: those texels' share of the footprint.
vec4 sampleHistory(Texture2D<float4> history, bool withAge, vec2 uv, out float weight, out float age) {
  vec2 size;
  history.GetDimensions(size.x, size.y);
  const vec2 position = uv * size - 0.5f;
  const int2 origin = int2(floor(position));
  const vec2 f = position - vec2(origin);
  vec4 sum = vec4(0.0f);
  float ageSum = 0.0f;
  weight = 0.0f;
  [unroll]
  for (int i = 0; i < 4; ++i) {
    const int2 offset = int2(i & 1, i >> 1);
    const int2 texel = clamp(origin + offset, int2(0), int2(size) - 1);
    const vec4 value = history[texel];
    const float texelWeight = (offset.x != 0 ? f.x : 1.0f - f.x) * (offset.y != 0 ? f.y : 1.0f - f.y);
    if (value.a >= 0.0f) {
      sum += value * texelWeight;
      if (withAge) {
        ageSum += CloudLayerHistoryAge[texel] * texelWeight;
      }
      weight += texelWeight;
    }
  }
  age = weight > 0.0f ? ageSum / weight : 0.0f;
  return weight > 0.0f ? sum / weight : vec4(0.0f);
}

// Current frame weight of the reflections' history, which takes the march's jitter out before the upscaler,
// whose guides on glass describe what the glass shows through it rather than the reflection.
static const float kReflectionHistoryBlend = 0.1f;

// Distance a pixel whose ray found no cloud reprojects at: the middle of the cloud layer's typical range.
static const float kReprojectionFallbackKm = 12.0f;

// Where the cloud at distanceKm along a pixel's unjittered ray was a frame ago, in world units: the clouds move
// by the wind, and their detail drifts on it by the convective rise and, growing with height, the shear.
vec3 cloudPreviousWorldPosition(vec3 pixelDirectionWorld, float distanceKm, CloudArgs cloud, AtmosphereArgs atmosphere) {
  const bool isZUp = atmosphere.isZUp != 0;
  const float heightFraction = cloudHeightFraction(
    cloudRayAltitudeKm(normalize(cloudWorldToYUp(pixelDirectionWorld, isZUp)), distanceKm, cloud, atmosphere.planetRadius), cloud);
  const vec3 stepYUp = vec3(cloud.windStepKm.x, cloud.riseStepKm, cloud.windStepKm.y)
    + vec3(cloud.windDirection.x, 0.0f, cloud.windDirection.y) * (cloud.shearStepKm * heightFraction);
  return cameraGetWorldPosition(cb.camera) + (pixelDirectionWorld * distanceKm - cloudWorldToYUp(stepYUp, isZUp)) * cloud.worldUnitsPerKm;
}

void writeDebugView(
  uint2 pixel, CloudMarchResult result, vec4 filtered, float historyWeight, vec3 dirYUp, vec3 dirWorld, float sceneDistanceKm,
  CloudArgs cloud, AtmosphereArgs atmosphere)
{
  vec4 value = vec4(0.0f, 0.0f, 0.0f, 1.0f);
  switch (cloud.debugView) {
  case DEBUG_VIEW_CLOUD_SCENE_SHADOW:
    if (sceneDistanceKm < 1e8f && (cloud.flags & CLOUD_FLAG_SUN_SHADOWS) != 0) {
      const vec3 worldPosition = cameraGetWorldPosition(cb.camera) + dirWorld * (sceneDistanceKm * cloud.worldUnitsPerKm);
      value = vec4(vec3(cloudSunTransmittance(CloudSunGrid, CloudShadowMap, CloudVolumeSampler, worldPosition, cloud, atmosphere)), 1.0f);
    } else {
      value = vec4(1.0f);
    }
    break;
  case DEBUG_VIEW_CLOUD_LAYER_RADIANCE:
    value = vec4(filtered.rgb, 1.0f);
    break;
  case DEBUG_VIEW_CLOUD_LAYER_TRANSMITTANCE:
    value = vec4(vec3(filtered.a), 1.0f);
    break;
  case DEBUG_VIEW_CLOUD_FIRST_HIT_DISTANCE:
    value = vec4(vec3(max(result.firstHitKm, 0.0f)), 1.0f);
    break;
  case DEBUG_VIEW_CLOUD_MEAN_DEPTH:
    value = vec4(vec3(max(result.meanDepthKm, 0.0f)), 1.0f);
    break;
  case DEBUG_VIEW_CLOUD_SAMPLE_COUNT:
    value = vec4(vec3(result.sampleCount), 1.0f);
    break;
  case DEBUG_VIEW_CLOUD_DOME:
    value = vec4(cloudSampleDome(CloudDome, CloudVolumeSampler, dirYUp, cloud).rgb, 1.0f);
    break;
  case DEBUG_VIEW_CLOUD_SKY_AP_INSCATTER:
  case DEBUG_VIEW_CLOUD_SKY_AP_TRANSMITTANCE:
  {
    vec2 apUv = cloudDirectionToSkyUv(dirYUp);
    apUv.y = cloudClampToTexelCentres(apUv.y, float(CLOUD_SKY_AP_LUT_HEIGHT));
    const float distanceKm = result.meanDepthKm > 0.0f ? result.meanDepthKm : kReprojectionFallbackKm;
    const CloudAirSegment air = cloudSampleSkyAp(apUv, distanceKm, cloud);
    value = vec4(cloud.debugView == DEBUG_VIEW_CLOUD_SKY_AP_INSCATTER ? air.inScatter : air.transmittance, 1.0f);
    break;
  }
  case DEBUG_VIEW_CLOUD_SUN_OPTICAL_DEPTH:
    if (result.firstHitKm > 0.0f) {
      const vec3 rel = dirYUp * result.firstHitKm;
      const vec3 positionKm = vec3(
        cloud.cameraPositionKm.x + rel.x,
        cloudRayAltitudeKm(dirYUp, result.firstHitKm, cloud, atmosphere.planetRadius),
        cloud.cameraPositionKm.z + rel.z);
      const float opticalDepth = cloudGridContains(positionKm.xz, cloud)
        ? CloudSunGrid.SampleLevel(CloudVolumeSampler, cloudGridUvw(positionKm, cloud), 0).x
        : (cloudFarGridContains(positionKm.xz, cloud) ? CloudSunGridFar.SampleLevel(CloudVolumeSampler, cloudFarGridUvw(positionKm, cloud), 0).x : 0.0f);
      value = vec4(vec3(opticalDepth), 1.0f);
    }
    break;
  case DEBUG_VIEW_CLOUD_DIFFUSION_FLUENCE:
    if (result.meanDepthKm > 0.0f) {
      const vec3 rel = dirYUp * result.meanDepthKm;
      const vec3 positionKm = vec3(
        cloud.cameraPositionKm.x + rel.x,
        cloudRayAltitudeKm(dirYUp, result.meanDepthKm, cloud, atmosphere.planetRadius),
        cloud.cameraPositionKm.z + rel.z);
      const float fluence = cloudGridContains(positionKm.xz, cloud)
        ? cloudDiffusionFluence(CloudDiffusionGrid, cloudGridUvw(positionKm, cloud))
        : (cloudFarGridContains(positionKm.xz, cloud) ? cloudDiffusionFluence(CloudDiffusionGridFar, cloudFarGridUvw(positionKm, cloud)) : 0.0f);
      value = vec4(vec3(fluence), 1.0f);
    }
    break;
  case DEBUG_VIEW_CLOUD_HISTORY_WEIGHT:
    value = vec4(vec3(historyWeight), 1.0f);
    break;
  default:
    return;
  }
  CloudDebugView[pixel] = value;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 threadId : SV_DispatchThreadID, uint2 localId : SV_GroupThreadID) {
  const bool inside = all(threadId < cb.camera.resolution);
  const uint2 pixel = min(threadId, cb.camera.resolution - 1u);
  const CloudArgs cloud = CloudCb.cloud;
  const AtmosphereArgs atmosphere = CloudCb.atmosphere;

  const CameraDirections directions = cameraPixelCoordinateToDirection(cb.camera, ivec2(pixel));
  const vec3 dirYUp = normalize(cloudWorldToYUp(directions.worldDirection, atmosphere.isZUp != 0));

  // How far the ray goes before the scene: mirrors (PSR) stop it at the mirror, while glass the sky is seen through
  // lets it on, where the PSR resolver leaves the clouds behind it to this march.
  const GeometryFlags geometryFlags = geometryFlagsReadFromGBuffer(ivec2(pixel), SharedFlags);
  const float linearViewZ = PrimaryLinearViewZ[pixel];
  const bool performedPsr = geometryFlags.performedAnyPSR() && !geometryFlags.secondarySurfaceMask;
  const bool primaryMiss = linearViewZ == cb.primaryDirectMissLinearViewZ && !geometryFlags.performedAnyPSR();
  const bool perPixelPsr = (cloud.flags & CLOUD_FLAG_PSR_REFLECTIONS) != 0 && !cb.domeLightArgs.active;
  const bool transmittedSky = perPixelPsr && geometryFlags.performPSTR && linearViewZ == cb.primaryDirectMissLinearViewZ;
  const float kmPerWorldUnit = 1.0f / max(cloud.worldUnitsPerKm, 1e-6f);
  uvec2 psrCoordinate;
  const bool hasPsrOutputs = compactedCoordinateFromPixelIfActive(
    CompactedPixelIndices, TileActiveCounts, pixel, cb.sparseRenderingArgs, psrCoordinate);
  float sceneDistanceKm = 1e9f;
  if (performedPsr && !transmittedSky) {
    const float firstHit = hasPsrOutputs ? PsrFirstHitDistance[psrCoordinate] : 0.0f;
    if (firstHit > 0.0f) {
      sceneDistanceKm = firstHit * kmPerWorldUnit;
    } else if (!hasPsrOutputs) {
      // A mirror at an inactive pixel is at no known distance, so it takes no layer and ray reconstruction fills it in.
      sceneDistanceKm = 0.0f;
    }
  } else if (!primaryMiss && !transmittedSky) {
    sceneDistanceKm = length(cameraReconstructViewPosition(cb.camera, directions.viewDirection, linearViewZ)) * kmPerWorldUnit;
  }

  CloudMarchSettings settings;
  settings.maxSamples = max(cloud.viewSamplesMax, 2u);
  settings.stepKm = max(cloud.viewStepKm, 1e-3f);
  settings.minStepKm = max(cloud.adaptiveStepKm, 1e-3f);
  settings.maxDistanceKm = sceneDistanceKm;
  settings.footprintPerKm = cloud.pixelAngle;
  settings.fineShadowTaps = cloud.shadowTaps;
  settings.cameraDetail = true;
  settings.fringeStepScale = 0.5f;

  const float jitter = BlueNoise[uint3(pixel & 127u, cloud.frameIndex & 63u)].x;

  // Half the rays into the layer march each frame, alternating in a checkerboard. The others keep their history,
  // which is the filtered layer of their last march, until their next march clamps it to the current frame again.
  // A ray with no history, such as one into sky just uncovered, marches anyway.
  // The cloud debug views show the march's own result, so every ray marches while one is active.
  const bool historyEnabled = (cloud.flags & CLOUD_FLAG_HISTORY_VALID) != 0 && cloud.historyBlend < 1.0f;
  const bool cloudDebugView = cloud.debugView >= DEBUG_VIEW_CLOUD_MIN && cloud.debugView <= DEBUG_VIEW_CLOUD_MAX;
  const bool rests = (cloud.flags & CLOUD_FLAG_CHECKERBOARD) != 0 && historyEnabled && !cloudDebugView
    && ((pixel.x + pixel.y + cloud.frameIndex) & 1u) != 0u
    && cloudRayEntersLayer(dirYUp, sceneDistanceKm, cloud, atmosphere);
  CloudMarchResult result = cloudEmptyMarchResult();
  if (!rests) {
    result = cloudMarch(dirYUp, settings, jitter, cloud, atmosphere);
  }
  vec4 current = vec4(result.radiance, result.transmittance);
  bool hidden = !rests && sceneDistanceKm < 1e8f && result.firstHitKm < 0.0f;

  // Neighbourhood of the current frame within the tile, the range the history is clamped to.
  tileLayer[localId.y][localId.x] = hidden || rests ? vec4(0.0f, 0.0f, 0.0f, kHiddenMark) : current;
  tileDepth[localId.y][localId.x] = vec2(result.meanDepthKm, result.meanDepthKm > 0.0f ? 1.0f - result.transmittance : 0.0f);
  GroupMemoryBarrierWithGroupSync();
  vec4 neighbourhoodMin = current;
  vec4 neighbourhoodMax = current;
  vec2 neighbourhoodDepth = vec2(0.0f);
  [unroll]
  for (int dy = -1; dy <= 1; ++dy) {
    [unroll]
    for (int dx = -1; dx <= 1; ++dx) {
      const int2 neighbour = clamp(int2(localId) + int2(dx, dy), int2(0), int2(7));
      const vec4 value = tileLayer[neighbour.y][neighbour.x];
      if (value.a >= 0.0f) {
        neighbourhoodMin = min(neighbourhoodMin, value);
        neighbourhoodMax = max(neighbourhoodMax, value);
      }
      const vec2 depth = tileDepth[neighbour.y][neighbour.x];
      neighbourhoodDepth += vec2(depth.x * depth.y, depth.y);
    }
  }
  const float neighbourhoodDepthKm = neighbourhoodDepth.y > 1e-4f
    ? neighbourhoodDepth.x / neighbourhoodDepth.y
    : kReprojectionFallbackKm;
  const vec3 pixelDirection = cameraPixelCoordinateToDirection(cb.camera, ivec2(pixel), false).worldDirection;

  vec4 filtered = current;
  float historyWeight = 0.0f;
  float age = 1.0f;
  if (rests) {
    // The history is reprojected at the marching neighbours' cloud depth.
    const float reprojectionKm = min(neighbourhoodDepthKm, sceneDistanceKm);
    const vec3 worldPosition = cloudPreviousWorldPosition(pixelDirection, reprojectionKm, cloud, atmosphere);
    const vec2 previousUv = cameraNDCToScreenUV(cb.camera, cameraWorldPosToPrevNDC(cb.camera, worldPosition));
    float footprint = 0.0f;
    float historyAge = 0.0f;
    const vec4 history = sampleHistory(CloudLayerHistory, true, previousUv, footprint, historyAge);
    if (all(previousUv > vec2(0.0f)) && all(previousUv < vec2(1.0f)) && footprint > 0.0f) {
      filtered = history;
      historyWeight = 1.0f;
      age = historyAge;
    } else {
      result = cloudMarch(dirYUp, settings, jitter, cloud, atmosphere);
      current = vec4(result.radiance, result.transmittance);
      hidden = sceneDistanceKm < 1e8f && result.firstHitKm < 0.0f;
      filtered = current;
    }
  }

  // A ray that met no cloud has exactly no layer, which no history may blur onto it.
  const bool reachedClouds = result.firstHitKm >= 0.0f;

  if (!rests && historyEnabled && reachedClouds) {
    // The pixel's centre, without this frame's jitter, at the cloud's depth, back where the cloud was a frame ago.
    const float reprojectionKm = min(result.meanDepthKm > 0.0f ? result.meanDepthKm : kReprojectionFallbackKm, sceneDistanceKm);
    const vec3 worldPosition = cloudPreviousWorldPosition(pixelDirection, reprojectionKm, cloud, atmosphere);
    const vec2 previousUv = cameraNDCToScreenUV(cb.camera, cameraWorldPosToPrevNDC(cb.camera, worldPosition));
    float footprint = 0.0f;
    float historyAge = 0.0f;
    const vec4 history = sampleHistory(CloudLayerHistory, true, previousUv, footprint, historyAge);
    if (all(previousUv > vec2(0.0f)) && all(previousUv < vec2(1.0f)) && footprint > 0.0f) {
      // A young history (sky a facade just uncovered) is averaged evenly with this frame, so that its first,
      // noisiest frames do not linger at the tier's long history weight.
      const float blend = max(saturate(cloud.historyBlend), 1.0f / (historyAge + 1.0f));
      historyWeight = 1.0f - blend;
      filtered = lerp(clamp(history, neighbourhoodMin, neighbourhoodMax), current, blend);
      age = min(historyAge + 1.0f, kMaxHistoryAge);
    }
  }

  // A mirror or glass (its reflection being its secondary surface) whose first reflection reached the sky takes
  // that sky's clouds along the exact reflected ray rather than from the dome, which the PSR miss left out. Only
  // glass's first reflection carries the throughput composite applies; its later ones carry none.
  const bool mirrorReflectedSky = performedPsr && geometryFlags.performPSRR && PsrFirstHitDistance[psrCoordinate] > 0.0f
    && PsrReflectionSegmentLength[psrCoordinate] < 0.0f;
  const bool glassReflectedSky = geometryFlags.secondarySurfaceMask
    && PsrReflectionSegmentLength[psrCoordinate] == cb.secondaryCombinedNrd.missLinearViewZ;
  const bool reflectsSky = perPixelPsr && hasPsrOutputs && (mirrorReflectedSky || glassReflectedSky);
  vec4 reflection = vec4(0.0f, 0.0f, 0.0f, 1.0f);
  float reflectionDepthKm = kReprojectionFallbackKm;
  if (reflectsSky)
  {
    const vec3 reflectedWorld = signedOctahedralToSphereDirection(PsrReflectionDirection[psrCoordinate]);
    const uint packedThroughput = glassReflectedSky
      ? SecondaryAttenuation[psrCoordinate]
      : PrimaryAttenuation[psrCoordinate];
    const vec3 throughput = r11g11b10ToColor(packedThroughput);
    CloudMarchSettings reflectionSettings = cloudReflectionMarchSettings(cloud);
    if (max(throughput.r, max(throughput.g, throughput.b)) > kDimReflectionThroughput) {
      reflectionSettings = settings;
      reflectionSettings.maxDistanceKm = 1e9f;
      reflectionSettings.cameraDetail = false;
    }
    const CloudMarchResult reflected = cloudMarch(
      normalize(cloudWorldToYUp(reflectedWorld, atmosphere.isZUp != 0)), reflectionSettings, jitter, cloud, atmosphere);
    reflection = vec4(reflected.radiance, reflected.transmittance);
    if (reflected.meanDepthKm > 0.0f) {
      reflectionDepthKm = reflected.meanDepthKm;
    }
  }

  tileReflection[localId.y][localId.x] = reflectsSky ? reflection : vec4(0.0f, 0.0f, 0.0f, kHiddenMark);
  GroupMemoryBarrierWithGroupSync();
  vec4 reflectionMin = reflection;
  vec4 reflectionMax = reflection;
  [unroll]
  for (int ry = -1; ry <= 1; ++ry) {
    [unroll]
    for (int rx = -1; rx <= 1; ++rx) {
      const vec4 value = tileReflection[clamp(int(localId.y) + ry, 0, 7)][clamp(int(localId.x) + rx, 0, 7)];
      if (value.a >= 0.0f) {
        reflectionMin = min(reflectionMin, value);
        reflectionMax = max(reflectionMax, value);
      }
    }
  }

  // The sky a planar mirror or pane reflects lies, seen from the camera, along the camera's own ray, mirrored, so
  // its history is where that ray's direction looked a frame ago, as for the open sky.
  vec4 filteredReflection = reflection;
  if (reflectsSky && (cloud.flags & CLOUD_FLAG_HISTORY_VALID) != 0) {
    const vec3 virtualPosition = cameraGetWorldPosition(cb.camera) + pixelDirection * (reflectionDepthKm * cloud.worldUnitsPerKm);
    const vec2 previousUv = cameraNDCToScreenUV(cb.camera, cameraWorldPosToPrevNDC(cb.camera, virtualPosition));
    float footprint = 0.0f;
    float unusedAge = 0.0f;
    const vec4 history = sampleHistory(CloudReflectionHistory, false, previousUv, footprint, unusedAge);
    if (all(previousUv > vec2(0.0f)) && all(previousUv < vec2(1.0f)) && footprint > 0.0f) {
      filteredReflection = lerp(clamp(history, reflectionMin, reflectionMax), reflection, kReflectionHistoryBlend);
    }
  }

  if (!inside) {
    return;
  }

  CloudReflectionOutput[pixel] = reflectsSky ? filteredReflection : vec4(0.0f, 0.0f, 0.0f, kHiddenMark);

  CloudLayerOutput[pixel] = hidden ? vec4(0.0f, 0.0f, 0.0f, kHiddenMark) : filtered;
  CloudLayerHistoryAgeOutput[pixel] = hidden ? 0.0f : age;
  CloudCompositeLayerOutput[pixel] = filtered;

  // The upscaler's motion where clouds cover the sky: theirs rather than the sky's, or it takes drifting clouds
  // for still ones and its history slides off their edges. At their depth over the neighbourhood, weighted by the
  // cloud each sample met, which a thin edge's own sample often misses; faded in over the first tenth of cover.
  const float motionWeight = smoothstep(0.0f, 0.1f, 1.0f - saturate(filtered.a));
  if ((primaryMiss || transmittedSky) && motionWeight > 0.0f) {
    const vec3 previousPosition = cloudPreviousWorldPosition(pixelDirection, neighbourhoodDepthKm, cloud, atmosphere);
    const vec2 previousNdc = cameraWorldPosToPrevNDC(cb.camera, previousPosition);
    const vec2 cloudMotion = cameraCalcMotionVectorPixel(cb.camera, cameraPixelCoordinateToNDC(cb.camera, ivec2(pixel)), previousNdc);
    PrimaryScreenSpaceMotionVector[pixel] = lerp(PrimaryScreenSpaceMotionVector[pixel], cloudMotion, motionWeight);
    PrimaryScreenSpaceMotionVectorDLSSRR[pixel] = lerp(PrimaryScreenSpaceMotionVectorDLSSRR[pixel], cloudMotion, motionWeight);
  }

  if (cloudDebugView) {
    writeDebugView(pixel, result, filtered, historyWeight, dirYUp, directions.worldDirection, sceneDistanceKm, cloud, atmosphere);
  }
}
