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

// Optical depth grids over the camera's neighbourhood, in extinction weighted density x km, so any extinction
// applies afterwards. Per voxel:
//   sun grid:     x toward the sun to where the ray leaves the slab; y the voxel's own extinction weighted density,
//                 filtered to the voxel: the medium the diffusion of the scattered sunlight runs through
//                 (cloud_diffusion).
//   ambient grid: x straight up to the top, y straight down to the base.
// Rays follow the curved shells: a low sun leaves the top after a bounded path, so there is no grazing
// extrapolation. The march samples the same field the view march renders, so the clouds and their shadows
// agree. The horizontal axes are toroidal in the field's frame, which the wind carries, so a bake stays true as
// the clouds drift: columns interleave over frames (period / phase) while the field evolves, and the strips a
// moving window brings in bake whole (period 1, from phase in x and offset in z). Two cascades share this bake:
// the near window around the camera and a coarser far one for distant clouds.

#include "rtx/pass/clouds/cloud_binding_indices.h"
#include "rtx/pass/clouds/cloud_field.slangh"

layout(binding = CLOUD_BINDING_CONSTANTS)
ConstantBuffer<CloudConstants> constants;

layout(binding = CLOUD_BINDING_VOLUME_SAMPLER)
SamplerState volumeSampler;

layout(binding = CLOUD_BINDING_NVDF)
Texture3D<float> nvdf;

layout(binding = CLOUD_BINDING_DETAIL_NOISE)
Texture3D<float4> detailNoise;

layout(binding = CLOUD_BINDING_BAKE_OUTPUT, rg16f)
RWTexture3D<float2> sunGridOutput;

layout(binding = CLOUD_BINDING_BAKE_OUTPUT2, rg16f)
RWTexture3D<float2> ambientGridOutput;

layout(push_constant)
ConstantBuffer<CloudPassArgs> passArgs;

groupshared vec2 columnControls[32];

float bakeDensity(vec3 positionKm, vec2 controls, CloudArgs cloud) {
  return cloudSampleExtinctionDensity(positionKm, controls.x, controls.y, false, cloud, nvdf, detailNoise, volumeSampler);
}

// The same, and how far from the sample the field is provably empty, at the safety factor the march's sphere
// tracing takes the body field's distances at. footprintKm filters the detail to that size, zero for none.
float bakeDensity(vec3 positionKm, vec2 controls, CloudArgs cloud, out float emptyKm, float footprintKm = 0.0f) {
  const float density = cloudSampleExtinctionDensity(
    positionKm, controls.x, controls.y, false, footprintKm, cloud, nvdf, detailNoise, volumeSampler, emptyKm);
  emptyKm *= cloud.nvdfStepScale;
  return density;
}

vec3 linePoint(vec3 positionKm, vec3 direction, float t, float planetRadiusKm) {
  return vec3(positionKm.x + direction.x * t, cloudAltitudeAlong(positionKm.y, direction.y, t, planetRadiusKm), positionKm.z + direction.z * t);
}

// The trapezoid rule along a line from a voxel. emptyUntilKm: the line is provably empty up to there (from the
// last empty sample's distance to the nearest cloud), so samples short of it are zero without reading the field.
struct CloudLine {
  float t;
  float density;
  float opticalDepth;
  float emptyUntilKm;
};

// The line's first sample is the voxel's own density, filtered to voxelFootprintKm.
CloudLine lineBegin(vec3 positionKm, vec2 controls, float voxelFootprintKm, CloudArgs cloud) {
  CloudLine line;
  line.t = 0.0f;
  line.density = bakeDensity(positionKm, controls, cloud, line.emptyUntilKm, voxelFootprintKm);
  line.opticalDepth = 0.0f;
  return line;
}

// A sample of the line at t: the field's density, or zero where the line is known to be empty. The far section
// reads the coverage field at each sample, since it varies over the section's tens of km, which no sample's empty
// radius bounds.
float lineDensity(inout CloudLine line, vec3 positionKm, vec3 direction, float t, vec2 controls, bool far, CloudArgs cloud, float planetRadiusKm) {
  if (!far && t < line.emptyUntilKm) {
    return 0.0f;
  }
  const vec3 p = linePoint(positionKm, direction, t, planetRadiusKm);
  if (far) {
    return bakeDensity(p, cloudControls(p.xz, cloud), cloud);
  }
  float emptyKm;
  const float density = bakeDensity(p, controls, cloud, emptyKm);
  line.emptyUntilKm = t + emptyKm;
  return density;
}

void lineAdd(inout CloudLine line, float t, float density) {
  line.opticalDepth += 0.5f * (line.density + density) * (t - line.t);
  line.t = t;
  line.density = density;
}

// Samples the line up to tEnd on planes perpendicular to it fixed in the field spacingKm apart, then at tEnd:
// neighbouring voxels integrate the same points of the field, so the grids carry no per-voxel noise.
void lineSection(
  inout CloudLine line, vec3 positionKm, vec3 direction, float along, float tEnd, float spacingKm, vec2 controls,
  bool far, CloudArgs cloud, float planetRadiusKm)
{
  for (float t = (floor((along + line.t) / spacingKm) + 1.0f) * spacingKm - along; t < tEnd; t += spacingKm) {
    lineAdd(line, t, lineDensity(line, positionKm, direction, t, controls, far, cloud, planetRadiusKm));
  }
  if (tEnd > line.t) {
    lineAdd(line, tEnd, lineDensity(line, positionKm, direction, tEnd, controls, far, cloud, planetRadiusKm));
  }
}

// Optical depth from a voxel toward the sun to the slab's exit, finer near the voxel and coarse over the long
// paths of a low sun; density is the voxel's own.
float marchTowardSun(
  vec3 positionKm, vec3 sunDir, vec2 controls, float voxelFootprintKm, CloudArgs cloud, float planetRadiusKm, out float density)
{
  CloudLine line = lineBegin(positionKm, controls, voxelFootprintKm, cloud);
  density = line.density;
  const float tExit = cloudSlabExitDistance(positionKm.y, sunDir.y, cloud, planetRadiusKm);
  if (tExit <= 0.0f) {
    return kCloudOccludedOpticalDepth;
  }
  const vec2 fieldXz = cloudFieldXz(positionKm.xz, cloud);
  const float along = dot(vec3(fieldXz.x, positionKm.y, fieldXz.y), sunDir);
  lineSection(line, positionKm, sunDir, along, min(tExit, 3.0f), 0.3f, controls, false, cloud, planetRadiusKm);
  if (tExit > 3.0f && line.opticalDepth < kCloudOccludedOpticalDepth) {
    lineSection(line, positionKm, sunDir, along, min(tExit, 12.0f), 0.6f, controls, false, cloud, planetRadiusKm);
  }
  if (tExit > 12.0f && line.opticalDepth < kCloudOccludedOpticalDepth) {
    lineSection(line, positionKm, sunDir, along, min(tExit, 80.0f), 8.5f, controls, true, cloud, planetRadiusKm);
  }
  return min(line.opticalDepth, kCloudOccludedOpticalDepth);
}

// Optical depth straight up or down to the slab's top or base.
float marchVerticalOpticalDepth(vec3 positionKm, float sign, vec2 controls, CloudArgs cloud) {
  const int taps = 6;
  const float boundary = sign > 0.0f ? cloud.baseAltitudeKm + cloud.thicknessKm : cloud.baseAltitudeKm;
  const float distance = abs(boundary - positionKm.y);
  if (distance <= 0.0f) {
    return 0.0f;
  }
  const float stepLength = distance / float(taps);
  float opticalDepth = 0.0f;
  float emptyUntilKm = 0.0f;
  [unroll]
  for (int i = 0; i < taps; ++i) {
    const float t = (float(i) + 0.5f) * stepLength;
    if (t >= emptyUntilKm) {
      float emptyKm;
      opticalDepth += bakeDensity(positionKm + vec3(0.0f, sign * t, 0.0f), controls, cloud, emptyKm) * stepLength;
      emptyUntilKm = t + emptyKm;
    }
  }
  return opticalDepth;
}

[shader("compute")]
[numthreads(8, 8, 4)]
void main(uint3 dispatchId : SV_DispatchThreadID, uint3 groupThreadId : SV_GroupThreadID) {
  const CloudArgs cloud = constants.cloud;
  const AtmosphereArgs atmosphere = constants.atmosphere;

  // passArgs.level picks the cascade: 0 the near window, 1 the far one.
  const bool far = bool(passArgs.level);
  const uint sizeXZ = cloudGridSizeXZ(passArgs.level);
  const uint period = max(passArgs.period, 1u);
  const uint column = cloudInterleavedTexel(dispatchId.x, period, passArgs.phase) % sizeXZ;
  const uint3 voxel = uint3(column, dispatchId.y, (dispatchId.z + passArgs.offset) % sizeXZ);
  const vec3 uvw = (vec3(voxel) + 0.5f) / vec3(float(sizeXZ), float(CLOUD_GRID_SIZE_Y), float(sizeXZ));

  const vec2 windowOriginKm = far ? cloud.farGridOriginKm : cloud.gridOriginKm.xy;
  const float windowExtentKm = far ? cloud.farGridExtentKm : cloud.gridExtentKm;
  const vec3 positionKm = vec3(
    cloudWindowTexelToFieldKm(uvw.x, windowOriginKm.x, windowExtentKm) + cloud.windOffsetKm.x,
    cloud.baseAltitudeKm + uvw.y * cloud.thicknessKm,
    cloudWindowTexelToFieldKm(uvw.z, windowOriginKm.y, windowExtentKm) + cloud.windOffsetKm.y);

  // Point sampled, detail finer than the voxel would flip the diffusion's cells across the medium's threshold as
  // the field evolves. The filter is the voxel's size, undoing the sharpening the view's detail LOD bias adds.
  const float voxelKm = max(windowExtentKm / float(sizeXZ), cloud.thicknessKm / float(CLOUD_GRID_SIZE_Y));
  const float voxelFootprintKm = voxelKm * exp2(-cloud.detailLodBias);

  // The coverage and type fields are the same for every voxel of a column.
  const uint columnIndex = groupThreadId.z * 8u + groupThreadId.x;
  if (groupThreadId.y == 0u) {
    columnControls[columnIndex] = cloudControls(positionKm.xz, cloud);
  }
  GroupMemoryBarrierWithGroupSync();
  const vec2 controls = columnControls[columnIndex];

  float density;
  const float towardSun = marchTowardSun(
    positionKm, normalize(atmosphere.sunDirection), controls, voxelFootprintKm, cloud, atmosphere.planetRadius, density);
  const vec2 vertical = vec2(
    marchVerticalOpticalDepth(positionKm, 1.0f, controls, cloud),
    marchVerticalOpticalDepth(positionKm, -1.0f, controls, cloud));

  sunGridOutput[voxel] = vec2(towardSun, density);
  ambientGridOutput[voxel] = vertical;
}
