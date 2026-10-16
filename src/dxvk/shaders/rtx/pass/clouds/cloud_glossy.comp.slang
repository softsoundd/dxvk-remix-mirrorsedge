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

static const uint kTileSize = 16;
static const uint kBlocksPerSide = kTileSize / 2;
// Each thread scans one 2x2 block of the tile.
static const uint kThreads = kBlocksPerSide * kBlocksPerSide;
static const uint kNoLeader = 4u;

// The tile's rays that march, packed so that no thread idles beside another's march: the rays lie scattered over
// about every other pixel of a glossy surface.
groupshared uint tileRayCount;
groupshared uint2 tileRays[kTileSize * kTileSize];

// Each 2x2 block's first ray, as x + 2y within the block, and the clouds it marched against the dome's, which the
// block's other rays along nearly the same direction take instead of marching their own.
groupshared uint blockLeader[kBlocksPerSide * kBlocksPerSide];
groupshared vec3 blockCorrection[kBlocksPerSide * kBlocksPerSide];

// Returns the clouds marched along the ray minus the dome's clouds that the integrator composited, both over the sky.
vec3 marchCloudCorrection(uint2 pixel, vec3 worldDirection, CloudArgs cloud, AtmosphereArgs atmosphere) {
  const vec3 dirYUp = normalize(cloudWorldToYUp(worldDirection, atmosphere.isZUp != 0));

  const CloudMarchSettings settings = cloudReflectionMarchSettings(cloud);
  const float jitter = BlueNoise[uint3(pixel & 127u, cloud.frameIndex & 63u)].y;
  const CloudMarchResult marched = cloudMarch(dirYUp, settings, jitter, cloud, atmosphere);

  // The sky the integrator's miss took, under the dome's clouds as it composited them and under the march's.
  const vec3 sky = sampleSkyViewLutRadiance(SkyViewLut, worldDirection, atmosphere, atmosphere.isZUp != 0);
  const vec3 dome = cloudCompositeOverSky(cloudSampleDome(CloudDome, CloudVolumeSampler, dirYUp, cloud), sky);
  const vec3 clouds = cloudCompositeOverSky(vec4(marched.radiance, marched.transmittance), sky);
  return clouds - dome;
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

  // A block's first ray marches, and so does any other ray further than the sharing angle from it. The rest take the
  // first ray's clouds once it has marched.
  const uint2 tileOrigin = groupId * kTileSize;
  const uint2 blockOrigin = tileOrigin + 2u * uint2(threadIndex % kBlocksPerSide, threadIndex / kBlocksPerSide);
  uint2 rays[4];
  uint followers = 0u;
  uint leader = kNoLeader;
  vec3 leaderDirection = vec3(0.0f);
  [unroll]
  for (uint i = 0u; i < 4u; ++i) {
    const uint2 pixel = blockOrigin + uint2(i & 1u, i >> 1u);
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
  blockLeader[threadIndex] = leader;
  blockCorrection[threadIndex] = vec3(0.0f);
  GroupMemoryBarrierWithGroupSync();

  for (uint i = threadIndex; i < tileRayCount; i += kThreads) {
    const uint2 pixel = tileRays[i];
    const uint2 ray = CloudGlossyRay[pixel];
    CloudGlossyRay[pixel] = uint2(0u, 0u);

    uvec2 radianceCoordinate;
    if (!compactedCoordinateFromPixelIfActive(CompactedPixelIndices, TileActiveCounts, pixel, cb.sparseRenderingArgs, radianceCoordinate)) {
      continue;
    }

    const vec3 correction = marchCloudCorrection(pixel, snorm2x16ToSphereDirection(ray.y), cloud, atmosphere);
    applyCloudCorrection(radianceCoordinate, ray, correction);

    const uint2 local = pixel - tileOrigin;
    const uint block = (local.y / 2u) * kBlocksPerSide + local.x / 2u;
    if ((local.x & 1u) + 2u * (local.y & 1u) == blockLeader[block]) {
      blockCorrection[block] = correction;
    }
  }

  GroupMemoryBarrierWithGroupSync();

  [unroll]
  for (uint i = 0u; i < 4u; ++i) {
    if ((followers & (1u << i)) == 0u) {
      continue;
    }

    const uint2 pixel = blockOrigin + uint2(i & 1u, i >> 1u);
    CloudGlossyRay[pixel] = uint2(0u, 0u);

    uvec2 radianceCoordinate;
    if (compactedCoordinateFromPixelIfActive(CompactedPixelIndices, TileActiveCounts, pixel, cb.sparseRenderingArgs, radianceCoordinate)) {
      applyCloudCorrection(radianceCoordinate, rays[i], blockCorrection[threadIndex]);
    }
  }
}
