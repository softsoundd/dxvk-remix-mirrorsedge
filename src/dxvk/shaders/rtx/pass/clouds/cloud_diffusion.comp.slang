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

// One red-black SOR half sweep of the P1 diffusion of the sunlight a cascade's clouds scatter (the diffusion
// floor of cloud_lighting): -div(D grad phi) = Q - div(3 D Q1), D = 1 / (3 sigma_tr), from the cells' Q (the beam's
// energy each removes, per unit sun irradiance) and Q1 = g' Q along the beam, the first moment of the delta-M
// scaled scattering, which carries the light forward. The medium's light escapes into clear air by Marshak
// boundaries, phi + 2 D d(phi)/dn = 0, and nothing crosses the window's edges. Cells of clear air next to the medium
// hold the boundary fluence extrapolated a cell on, so trilinear filtering meets it at the face. passArgs: level
// the cascade, phase the parity of the cells this half sweep updates.

#include "rtx/pass/clouds/cloud_binding_indices.h"
#include "rtx/pass/clouds/cloud_common.slangh"

layout(binding = CLOUD_BINDING_CONSTANTS)
ConstantBuffer<CloudConstants> constants;

// The cells: (Q, density), density zero for clear air.
layout(binding = CLOUD_BINDING_BAKE_INPUT)
Texture3D<float2> cells;

layout(binding = CLOUD_BINDING_BAKE_OUTPUT, r32f)
RWTexture3D<float> fluence;

layout(push_constant)
ConstantBuffer<CloudPassArgs> passArgs;

// A little under the optimum for clouds some tens of cells across; any factor below 2 converges.
static const float kOverRelaxation = 1.8f;

[shader("compute")]
[numthreads(4, 4, 8)]
void main(uint3 threadId : SV_DispatchThreadID) {
  const uint3 voxel = uint3(threadId.x * 2u + ((threadId.y + threadId.z + passArgs.phase) & 1u), threadId.y, threadId.z);
  const uint sizeXZ = cloudGridSizeXZ(passArgs.level);
  if (any(voxel >= uint3(sizeXZ, CLOUD_GRID_SIZE_Y, sizeXZ))) {
    return;
  }
  const CloudArgs cloud = constants.cloud;
  const bool far = bool(passArgs.level);
  const vec2 windowOriginKm = far ? cloud.farGridOriginKm : cloud.gridOriginKm.xy;
  const float windowExtentKm = far ? cloud.farGridExtentKm : cloud.gridExtentKm;
  const vec3 gridSize = vec3(float(sizeXZ), float(CLOUD_GRID_SIZE_Y), float(sizeXZ));
  const vec3 cellKm = vec3(windowExtentKm, cloud.thicknessKm, windowExtentKm) / gridSize;
  const float transport = (1.0f - cloud.asymmetry) * cloud.extinctionKm;
  const vec3 sunDir = normalize(constants.atmosphere.sunDirection);
  const float forwardWeight = 3.0f * cloud.truncatedAsymmetry;

  // The window's horizontal edges are a seam of the toroidal grid: no light crosses them.
  const vec3 uvw = (vec3(voxel) + 0.5f) / gridSize;
  const vec2 fieldKm = vec2(
    cloudWindowTexelToFieldKm(uvw.x, windowOriginKm.x, windowExtentKm),
    cloudWindowTexelToFieldKm(uvw.z, windowOriginKm.y, windowExtentKm));
  const vec2 windowMin = windowOriginKm - 0.5f * windowExtentKm;
  const vec2 windowMax = windowOriginKm + 0.5f * windowExtentKm;

  const vec2 cell = cells.Load(int4(voxel, 0));
  const bool medium = cell.y > 0.0f;
  const float d = medium ? 1.0f / (3.0f * transport * cell.y) : 0.0f;
  float numerator = cell.x;
  float denominator = 0.0f;
  float extrapolated = 0.0f;
  float extrapolatedCount = 0.0f;

  [unroll]
  for (int n = 0; n < 6; ++n) {
    const int axis = n >> 1;
    const int offset = (n & 1) != 0 ? 1 : -1;
    const float h = cellKm[axis];
    int3 neighbour = int3(voxel);
    neighbour[axis] += offset;
    bool outside = false;
    if (axis == 1) {
      outside = neighbour.y < 0 || neighbour.y >= CLOUD_GRID_SIZE_Y;
    } else {
      const float coordinate = (axis == 0 ? fieldKm.x : fieldKm.y) + float(offset) * h;
      if (coordinate < (axis == 0 ? windowMin.x : windowMin.y) || coordinate > (axis == 0 ? windowMax.x : windowMax.y)) {
        continue;
      }
      neighbour[axis] = (neighbour[axis] + int(sizeXZ)) % int(sizeXZ);
    }
    const vec2 other = outside ? vec2(0.0f) : cells.Load(int4(neighbour, 0));

    if (!medium) {
      // Clear air: each medium neighbour's Marshak boundary fluence toward this cell, extrapolated to its centre.
      if (other.y > 0.0f) {
        const float a = 2.0f / (3.0f * transport * other.y * h);
        extrapolated += max(2.0f * a / (0.5f + a) - 1.0f, 0.0f) * fluence[neighbour];
        extrapolatedCount += 1.0f;
      }
      continue;
    }

    const float sunAlongNormal = sunDir[axis] * float(offset);
    if (outside || other.y <= 0.0f) {
      // Marshak: the face's net outflow is 2 D / (h + 4 D) phi.
      denominator += 2.0f * d / ((h + 4.0f * d) * h);
      numerator += forwardWeight * d * 0.5f * cell.x * sunAlongNormal / h;
      continue;
    }
    const float dOther = 1.0f / (3.0f * transport * other.y);
    const float dFace = 2.0f * d * dOther / (d + dOther);
    const float k = dFace / (h * h);
    numerator += k * fluence[neighbour] + forwardWeight * dFace * 0.5f * (cell.x + other.x) * sunAlongNormal / h;
    denominator += k;
  }

  if (!medium) {
    fluence[voxel] = extrapolatedCount > 0.0f ? extrapolated / extrapolatedCount : 0.0f;
    return;
  }
  const float gaussSeidel = max(numerator, 0.0f) / max(denominator, 1e-6f);
  fluence[voxel] = max(lerp(fluence[voxel], gaussSeidel, kOverRelaxation), 0.0f);
}
