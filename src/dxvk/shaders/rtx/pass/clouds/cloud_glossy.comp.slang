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

// The clouds of near-mirror reflections into the sky. The indirect integrator composites them from the dome, whose
// texels are wider than a narrow lobe and flicker whole as its interleaved refreshes land, and stashes each such
// ray (stashCloudGlossyRay). A march along the ray replaces the dome's clouds in the pixel's indirect radiance
// before the NEE pass carries it on, so everything downstream takes them as it takes the sky. ReSTIR GI, which may
// share the ray with other pixels, keeps the dome's.
//
// A reprojected history resolves the march's jitter, which ray reconstruction alone shows while the camera moves.
// The history is taken only from texels marched along nearly the pixel's ray.
// A planar mirror keeps its rays as the camera moves, while a curved, bumpy or moving reflector changes them.
// Clamping the history to the current frame's neighbourhood would drag it back into the jitter,
// because a block's rays share one march and leave too few marches nearby to bound it.
// Each pixel carries its history on through the frames in which its path does not sample the specular lobe.

#include "rtx/pass/raytrace_args.h"
#include "rtx/pass/clouds/cloud_march.slangh"
#include "rtx/pass/sparse_rendering/compact_active_pixels.slangh"
#include "rtx/utility/packing.slangh"

layout(binding = CLOUD_BINDING_CAMERA)
ConstantBuffer<RaytraceArgs> cb;

layout(binding = CLOUD_BINDING_BLUE_NOISE)
Texture2DArray BlueNoise;

layout(binding = CLOUD_BINDING_DOME_LUT)
Texture2D<float4> CloudDome;

layout(binding = CLOUD_BINDING_SKY_VIEW_LUT)
Texture2D SkyViewLut;

// The throughput (R11G11B10) and direction (signed octahedral) of each pixel's ray, zero where there is none. Rays
// are zeroed as they are consumed, so the stash needs no clearing.
layout(binding = CLOUD_BINDING_GLOSSY_RAY, rg32ui)
RWTexture2D<uint2> CloudGlossyRay;

// Under sparse rendering at the pixel's compacted slot, which the two maps below resolve.
layout(binding = CLOUD_BINDING_INDIRECT_RADIANCE, rgba16f)
RWTexture2D<float4> IndirectRadianceHitDistance;

layout(binding = CLOUD_BINDING_COMPACTED_PIXEL_INDICES)
Texture2D<uint16_t> CompactedPixelIndices;

layout(binding = CLOUD_BINDING_TILE_ACTIVE_COUNTS)
Texture2D<uint2> TileActiveCounts;

// These hold each pixel's history from a frame ago and this frame's, packed by glossyHistoryPack.
// Every pixel is written each frame.
layout(binding = CLOUD_BINDING_GLOSSY_HISTORY)
Texture2D<uint4> CloudGlossyHistory;

layout(binding = CLOUD_BINDING_GLOSSY_HISTORY_OUTPUT, rgba32ui)
RWTexture2D<uint4> CloudGlossyHistoryOutput;

static const uint kTileSize = 16;
static const uint kBlocksPerSide = kTileSize / 2;
// Each thread scans one 2x2 block of the tile.
static const uint kThreads = kBlocksPerSide * kBlocksPerSide;
static const uint kNoLeader = 4u;

// Stands in for the transmittance of a tile pixel that has no layer.
static const float kNoLayer = -1.0f;

// A history texel is taken when its ray lies within the larger of these two angles of the pixel's ray.
// The camera's jitter moves a still reflection's ray by about a pixel.
// The path state's half precision direction moves it by up to about a milliradian.
static const float kHistoryAnglePixels = 4.0f;
static const float kHistoryMinAngleRadians = 0.002f;

// The tile's rays that march, packed so that no thread idles beside another's march: the rays lie scattered over
// about every other pixel of a glossy surface.
groupshared uint tileRayCount;
groupshared uint2 tileRays[kTileSize * kTileSize];

// These hold each pixel's layer this frame, or kNoLayer where it has none, and the distance its history reprojects at.
groupshared vec4 tileLayer[kTileSize][kTileSize];
groupshared float tileDepthKm[kTileSize][kTileSize];

// Holds a pixel's filtered layer, the ray it was marched along and the frames it has accumulated.
// Its age is 0 when there is no history.
struct GlossyHistory {
  vec4 layer;
  vec3 direction;
  float age;
};

GlossyHistory glossyHistoryCreateEmpty() {
  GlossyHistory history;
  history.layer = vec4(0.0f);
  history.direction = vec3(0.0f);
  history.age = 0.0f;
  return history;
}

// Packs the layer as four halves, the ray as signed octahedral and the age as a float.
uint4 glossyHistoryPack(GlossyHistory history) {
  return uint4(
    f32tof16(history.layer.r) | (f32tof16(history.layer.g) << 16),
    f32tof16(history.layer.b) | (f32tof16(history.layer.a) << 16),
    sphereDirectionToSnorm2x16(history.direction),
    asuint(history.age));
}

GlossyHistory glossyHistoryUnpack(uint4 texel) {
  GlossyHistory history;
  history.layer = vec4(f16tof32(texel.x), f16tof32(texel.x >> 16), f16tof32(texel.y), f16tof32(texel.y >> 16));
  history.direction = snorm2x16ToSphereDirection(texel.z);
  history.age = asfloat(texel.w);
  return history;
}

// Returns the history at uv, bilinearly over the texels marched along a ray within the cosine of the reference ray.
// The reference ray is the pixel's own when it has one.
// Otherwise, it is the ray of the footprint's largest texel that holds a history.
GlossyHistory sampleHistory(vec2 uv, bool hasRay, vec3 direction, float cosine) {
  vec2 size;
  CloudGlossyHistory.GetDimensions(size.x, size.y);
  const vec2 position = uv * size - 0.5f;
  const int2 origin = int2(floor(position));
  const vec2 f = position - vec2(origin);

  GlossyHistory texels[4];
  float texelWeights[4];
  vec3 reference = direction;
  float largestWeight = 0.0f;
  [unroll]
  for (int i = 0; i < 4; ++i) {
    const int2 offset = int2(i & 1, i >> 1);
    texels[i] = glossyHistoryUnpack(CloudGlossyHistory[clamp(origin + offset, int2(0), int2(size) - 1)]);
    texelWeights[i] = (offset.x != 0 ? f.x : 1.0f - f.x) * (offset.y != 0 ? f.y : 1.0f - f.y);
    if (!hasRay && texels[i].age > 0.0f && texelWeights[i] > largestWeight) {
      largestWeight = texelWeights[i];
      reference = texels[i].direction;
    }
  }

  GlossyHistory history = glossyHistoryCreateEmpty();
  history.direction = reference;
  float weight = 0.0f;
  [unroll]
  for (int i = 0; i < 4; ++i) {
    if (texels[i].age > 0.0f && dot(texels[i].direction, reference) >= cosine) {
      history.layer += texels[i].layer * texelWeights[i];
      history.age += texels[i].age * texelWeights[i];
      weight += texelWeights[i];
    }
  }
  if (weight > 0.0f) {
    history.layer /= weight;
    history.age /= weight;
  }
  return history;
}

// Returns the pixel's history from where the camera's ray looked a frame ago at the clouds' depth.
// A planar mirror shows its clouds along the camera's own ray, so they reproject as the open sky does.
GlossyHistory reprojectHistory(
  uint2 pixel, float depthKm, bool hasRay, vec3 direction, float cosine, CloudArgs cloud)
{
  if ((cloud.flags & CLOUD_FLAG_GLOSSY_HISTORY_VALID) == 0u) {
    return glossyHistoryCreateEmpty();
  }

  const vec3 pixelDirection = cameraPixelCoordinateToDirection(cb.camera, ivec2(pixel), false).worldDirection;
  const vec3 virtualPosition = cameraGetWorldPosition(cb.camera) + pixelDirection * (depthKm * cloud.worldUnitsPerKm);
  const vec2 previousUv = cameraNDCToScreenUV(cb.camera, cameraWorldPosToPrevNDC(cb.camera, virtualPosition));
  if (any(previousUv <= vec2(0.0f)) || any(previousUv >= vec2(1.0f))) {
    return glossyHistoryCreateEmpty();
  }
  return sampleHistory(previousUv, hasRay, direction, cosine);
}

// Returns the layer over the sky along the ray minus the dome's clouds that the integrator composited over it.
vec3 cloudCorrection(vec3 worldDirection, vec4 layer, CloudArgs cloud, AtmosphereArgs atmosphere) {
  const vec3 dirYUp = normalize(cloudWorldToYUp(worldDirection, atmosphere.isZUp != 0));
  const vec3 sky = sampleSkyViewLutRadiance(SkyViewLut, worldDirection, atmosphere, atmosphere.isZUp != 0);
  const vec3 dome = cloudCompositeOverSky(cloudSampleDome(CloudDome, CloudVolumeSampler, dirYUp, cloud), sky);
  return cloudCompositeOverSky(layer, sky) - dome;
}

// Replaces the dome's clouds in the pixel's indirect radiance, by its ray's throughput times the correction.
void applyCloudCorrection(uvec2 radianceCoordinate, uint2 ray, vec3 correction) {
  vec4 radiance = IndirectRadianceHitDistance[radianceCoordinate];
  radiance.rgb = max(radiance.rgb + r11g11b10ToColor(ray.x) * correction, vec3(0.0f));
  IndirectRadianceHitDistance[radianceCoordinate] = radiance;
}

[shader("compute")]
[numthreads(kThreads, 1, 1)]
void main(uint2 groupId : SV_GroupID, uint threadIndex : SV_GroupIndex) {
  if (threadIndex == 0u) {
    tileRayCount = 0u;
  }
  GroupMemoryBarrierWithGroupSync();

  const CloudArgs cloud = CloudCb.cloud;
  const AtmosphereArgs atmosphere = CloudCb.atmosphere;

  // A block's first ray marches, and so does any other ray further than the sharing angle from it.
  // The rest take the first ray's layer once it has marched.
  const uint2 tileOrigin = groupId * kTileSize;
  const uint2 blockLocal = 2u * uint2(threadIndex % kBlocksPerSide, threadIndex / kBlocksPerSide);
  uint2 rays[4];
  uint followers = 0u;
  uint leader = kNoLeader;
  vec3 leaderDirection = vec3(0.0f);
  [unroll]
  for (uint i = 0u; i < 4u; ++i) {
    const uint2 local = blockLocal + uint2(i & 1u, i >> 1u);
    tileLayer[local.y][local.x] = vec4(0.0f, 0.0f, 0.0f, kNoLayer);

    const uint2 pixel = tileOrigin + local;
    rays[i] = all(pixel < cb.camera.resolution) ? CloudGlossyRay[pixel] : uint2(0u, 0u);
    if (rays[i].x == 0u) {
      continue;
    }

    const vec3 direction = snorm2x16ToSphereDirection(rays[i].y);
    if (leader != kNoLeader && dot(direction, leaderDirection) >= cloud.glossyShareCosine) {
      followers |= 1u << i;
      continue;
    }

    if (leader == kNoLeader) {
      leader = i;
      leaderDirection = direction;
    }
    uint slot;
    InterlockedAdd(tileRayCount, 1u, slot);
    tileRays[slot] = pixel;
  }
  GroupMemoryBarrierWithGroupSync();

  for (uint i = threadIndex; i < tileRayCount; i += kThreads) {
    const uint2 pixel = tileRays[i];
    const uint2 ray = CloudGlossyRay[pixel];
    CloudGlossyRay[pixel] = uint2(0u, 0u);

    uvec2 radianceCoordinate;
    if (!compactedCoordinateFromPixelIfActive(CompactedPixelIndices, TileActiveCounts, pixel, cb.sparseRenderingArgs, radianceCoordinate)) {
      continue;
    }

    const vec3 dirYUp = normalize(cloudWorldToYUp(snorm2x16ToSphereDirection(ray.y), atmosphere.isZUp != 0));
    // BlueNoise is single channel. The layer offset keeps this jitter apart from cloud_screen.comp.slang's.
    const float jitter = BlueNoise[uint3(pixel & 127u, (cloud.frameIndex + 32u) & 63u)].x;
    const CloudMarchResult marched = cloudMarch(dirYUp, cloudReflectionMarchSettings(cloud), jitter, cloud, atmosphere);

    const uint2 local = pixel - tileOrigin;
    tileLayer[local.y][local.x] = vec4(marched.radiance, marched.transmittance);
    tileDepthKm[local.y][local.x] = marched.meanDepthKm > 0.0f ? marched.meanDepthKm : kReprojectionFallbackKm;
  }
  GroupMemoryBarrierWithGroupSync();

  // Each pixel blends its layer into its history, or carries the history on when it has no layer.
  const float historyCosine = cos(max(kHistoryAnglePixels * cloud.pixelAngle, kHistoryMinAngleRadians));
  [unroll]
  for (uint i = 0u; i < 4u; ++i) {
    const uint2 local = blockLocal + uint2(i & 1u, i >> 1u);
    const uint2 pixel = tileOrigin + local;
    if (any(pixel >= cb.camera.resolution)) {
      continue;
    }

    const bool follower = (followers & (1u << i)) != 0u;
    if (follower) {
      CloudGlossyRay[pixel] = uint2(0u, 0u);
    }
    const uint2 source = follower ? blockLocal + uint2(leader & 1u, leader >> 1u) : local;
    const vec4 current = tileLayer[source.y][source.x];
    const bool hasLayer = current.a >= 0.0f;
    const vec3 direction = snorm2x16ToSphereDirection(rays[i].y);
    const float depthKm = hasLayer ? tileDepthKm[source.y][source.x] : kReprojectionFallbackKm;
    GlossyHistory history = reprojectHistory(pixel, depthKm, hasLayer, direction, historyCosine, cloud);
    if (!hasLayer) {
      CloudGlossyHistoryOutput[pixel] = history.age > 0.0f ? glossyHistoryPack(history) : uint4(0u);
      continue;
    }

    // A young history is averaged evenly with this frame,
    // so that its first, noisiest frames do not linger at the tier's long history weight.
    const float blend = max(saturate(cloud.historyBlend), 1.0f / (history.age + 1.0f));
    history.layer = lerp(history.layer, current, blend);
    history.direction = direction;
    history.age = min(history.age + 1.0f, kMaxHistoryAge);
    CloudGlossyHistoryOutput[pixel] = glossyHistoryPack(history);

    uvec2 radianceCoordinate;
    if (compactedCoordinateFromPixelIfActive(CompactedPixelIndices, TileActiveCounts, pixel, cb.sparseRenderingArgs, radianceCoordinate)) {
      applyCloudCorrection(radianceCoordinate, rays[i], cloudCorrection(direction, history.layer, cloud, atmosphere));
    }
  }
}
