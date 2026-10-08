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

layout(binding = CLOUD_BINDING_INDIRECT_RADIANCE, rgba16f)
RWTexture2D<float4> IndirectRadianceHitDistance;

static const uint kTileSize = 16;
static const uint kThreads = 64;

// The tile's pixels that carry a ray, packed so that no thread idles beside another's march: the rays lie
// scattered over about every other pixel of a glossy surface.
groupshared uint tileRayCount;
groupshared uint2 tileRays[kTileSize * kTileSize];

void replaceDomeClouds(uint2 pixel, uint2 ray, CloudArgs cloud, AtmosphereArgs atmosphere) {
  const vec3 worldDirection = snorm2x16ToSphereDirection(ray.y);
  const vec3 dirYUp = normalize(cloudWorldToYUp(worldDirection, atmosphere.isZUp != 0));

  // From the eye, like the mirrors' march, at the dome's settings for reflections but the pixel's footprint.
  CloudMarchSettings settings;
  settings.maxSamples = max(cloud.viewSamplesMax / 2u, 32u);
  settings.stepKm = max(cloud.viewStepKm * 2.0f, 0.2f);
  settings.minStepKm = max(cloud.adaptiveStepKm * 4.0f, 0.1f);
  settings.maxDistanceKm = 1e9f;
  settings.footprintPerKm = cloud.pixelAngle;
  settings.fineShadowTaps = min(cloud.shadowTaps, 1u);
  settings.cameraDetail = false;
  settings.fringeStepScale = 1.0f;
  const float jitter = BlueNoise[uint3(pixel & 127u, cloud.frameIndex & 63u)].y;
  const CloudMarchResult marched = cloudMarch(dirYUp, settings, jitter, cloud, atmosphere);

  // The sky the integrator's miss took, under the dome's clouds as it composited them and under the march's.
  const vec3 sky = sampleSkyViewLutRadiance(SkyViewLut, worldDirection, atmosphere, atmosphere.isZUp != 0);
  const vec3 dome = cloudCompositeOverSky(cloudSampleDome(CloudDome, CloudVolumeSampler, dirYUp, cloud), sky);
  const vec3 clouds = cloudCompositeOverSky(vec4(marched.radiance, marched.transmittance), sky);

  vec4 radiance = IndirectRadianceHitDistance[pixel];
  radiance.rgb = max(radiance.rgb + r11g11b10ToColor(ray.x) * (clouds - dome), vec3(0.0f));
  IndirectRadianceHitDistance[pixel] = radiance;
}

[shader("compute")]
[numthreads(kThreads, 1, 1)]
void main(uint2 groupId : SV_GroupID, uint threadIndex : SV_GroupIndex) {
  if (threadIndex == 0u) {
    tileRayCount = 0u;
  }
  GroupMemoryBarrierWithGroupSync();

  const uint2 tileOrigin = groupId * kTileSize;
  for (uint i = threadIndex; i < kTileSize * kTileSize; i += kThreads) {
    const uint2 pixel = tileOrigin + uint2(i % kTileSize, i / kTileSize);
    if (all(pixel < cb.camera.resolution) && CloudGlossyRay[pixel].x != 0u) {
      uint slot;
      InterlockedAdd(tileRayCount, 1u, slot);
      tileRays[slot] = pixel;
    }
  }
  GroupMemoryBarrierWithGroupSync();

  const CloudArgs cloud = CloudCb.cloud;
  const AtmosphereArgs atmosphere = CloudCb.atmosphere;
  for (uint i = threadIndex; i < tileRayCount; i += kThreads) {
    const uint2 pixel = tileRays[i];
    replaceDomeClouds(pixel, CloudGlossyRay[pixel], cloud, atmosphere);
    CloudGlossyRay[pixel] = uint2(0u, 0u);
  }
}
