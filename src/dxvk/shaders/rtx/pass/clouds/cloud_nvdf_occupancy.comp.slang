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

// Voxelises the cloud bodies (the placement map through the column model, at the nominal coverage) into the
// tile periodic occupancy grid the jump flood turns into the body distance field. The detail volume stays out:
// it erodes at sample time and must not move the baked surface. The live coverage becomes a level set offset
// at sample time, so only a large drift from the nominal needs a re-bake.

#include "rtx/pass/clouds/cloud_binding_indices.h"
#include "rtx/pass/clouds/cloud_field.slangh"
#include "rtx/pass/clouds/cloud_nvdf_common.slangh"

layout(binding = CLOUD_BINDING_CONSTANTS)
ConstantBuffer<CloudConstants> constants;

layout(binding = CLOUD_BINDING_PLACEMENT_MAP)
Texture2D<float4> placementMap;

layout(binding = CLOUD_BINDING_VOLUME_SAMPLER)
SamplerState placementSampler;

layout(binding = CLOUD_BINDING_BAKE_OUTPUT, r8)
RWTexture3D<float> occupancyOutput;

[shader("compute")]
[numthreads(8, 4, 8)]
void main(uint3 threadId : SV_DispatchThreadID) {
  if (any(threadId >= uint3(CLOUD_NVDF_SIZE_XZ, CLOUD_NVDF_SIZE_Y, CLOUD_NVDF_SIZE_XZ))) {
    return;
  }

  const CloudArgs cloud = constants.cloud;

  const vec2 tileUv = cloudNvdfVoxelToTileUv(threadId);
  vec4 placement = placementMap.SampleLevel(placementSampler, tileUv, 0);
  const float heightFraction = cloudNvdfVoxelToHeightFraction(threadId);

  // A placement blob through a column band is convex, and the detail erosion is only surface deep. Shifting the
  // placement's waterline with a height varying periodic FBM bakes overhangs, notches and lumps into the body
  // itself. Zero mean, so the coverage is unchanged.
  if (cloud.nvdfBodyErosion > 0.0f) {
    const float carveCyclesXZ = 6.0f;
    const float carveCyclesY = max(1.0f, round(carveCyclesXZ * cloud.thicknessKm / max(cloud.tileKm, 1e-3f)));
    const vec3 carvePeriod = vec3(carveCyclesXZ, carveCyclesY, carveCyclesXZ);
    const float carve = cloudFbmPeriodic(vec3(tileUv.x, heightFraction, tileUv.y) * carvePeriod, 4, 67.3f, carvePeriod);
    placement.r = saturate(placement.r + carve * 0.25f * cloud.nvdfBodyErosion);
  }

  const CloudColumn column = cloudComputeColumn(placement, saturate(cloud.nvdfNominalCoverage), cloud);
  const bool occupied = column.presence > 0.001f && heightFraction > column.baseFrac && heightFraction < column.topFrac;

  occupancyOutput[threadId] = occupied ? 1.0f : 0.0f;
}
