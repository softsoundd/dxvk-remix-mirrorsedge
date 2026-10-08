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

// The diffusion's cells from a cascade's sun grid, (Q, density): Q the energy each cell takes from the sun's
// beam per unit sun irradiance and km of beam, the beam's delta-M scaled transmittance where it enters the cell on
// the sun's side less what the cell's own medium passes; density the cell's extinction weighted density, zero
// for clear air. A transmittance differenced across the cell would take in its denser neighbours' extinction
// through the grid's filtering and spike the fluence of a thin cell beside a dense one.

#include "rtx/pass/clouds/cloud_binding_indices.h"
#include "rtx/pass/clouds/cloud_common.slangh"

layout(binding = CLOUD_BINDING_CONSTANTS)
ConstantBuffer<CloudConstants> constants;

layout(binding = CLOUD_BINDING_VOLUME_SAMPLER)
SamplerState volumeSampler;

layout(binding = CLOUD_BINDING_BAKE_INPUT)
Texture3D<float2> sunGrid;

layout(binding = CLOUD_BINDING_BAKE_OUTPUT, rg32f)
RWTexture3D<float2> cellsOutput;

// level picks the cascade: 0 the near window, 1 the far one.
layout(push_constant)
ConstantBuffer<CloudPassArgs> passArgs;

float sunTransmittance(vec3 uvw, CloudArgs cloud) {
  uvw.y = cloudClampToTexelCentres(uvw.y, float(CLOUD_GRID_SIZE_Y));
  return exp(-cloud.extinctionKm * (1.0f - cloud.forwardPeakFraction) * sunGrid.SampleLevel(volumeSampler, uvw, 0).x);
}

[shader("compute")]
[numthreads(8, 4, 8)]
void main(uint3 voxel : SV_DispatchThreadID) {
  if (any(voxel >= uint3(CLOUD_GRID_SIZE_XZ, CLOUD_GRID_SIZE_Y, CLOUD_GRID_SIZE_XZ))) {
    return;
  }
  const CloudArgs cloud = constants.cloud;
  const float density = sunGrid.Load(int4(voxel, 0)).y;
  if (density < kCloudDiffusionMinDensity) {
    cellsOutput[voxel] = vec2(0.0f);
    return;
  }
  const vec3 gridSize = vec3(float(CLOUD_GRID_SIZE_XZ), float(CLOUD_GRID_SIZE_Y), float(CLOUD_GRID_SIZE_XZ));
  const float windowExtentKm = bool(passArgs.level) ? cloud.farGridExtentKm : cloud.gridExtentKm;
  const vec3 extentKm = vec3(windowExtentKm, cloud.thicknessKm, windowExtentKm);
  const float h = min(extentKm.x / gridSize.x, extentKm.y / gridSize.y);
  const vec3 uvw = (vec3(voxel) + 0.5f) / gridSize;
  const vec3 halfStep = normalize(constants.atmosphere.sunDirection) * (0.5f * h) / extentKm;
  const float tau = cloud.extinctionKm * (1.0f - cloud.forwardPeakFraction) * density * h;
  const float q = sunTransmittance(uvw + halfStep, cloud) * (1.0f - exp(-tau)) / h;
  cellsOutput[voxel] = vec2(q, density);
}
