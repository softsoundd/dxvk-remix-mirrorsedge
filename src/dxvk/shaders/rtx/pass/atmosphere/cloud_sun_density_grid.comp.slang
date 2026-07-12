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

// Cloud D_sun voxel grid bake (Nubis Cubed 2023, fork — 2026-05-12).
//
// Bakes summed optical depth along the sun direction at each voxel of a
// camera-centered 256x256x32 R16F grid. Consumed at shade time via
// sampleDSun() for both the future Nubis Cubed view-march multi-scatter
// rewrite and the cloud-on-terrain shadow NEE path (C4-C6 of the
// 2026-05-12 workstream).
//
// Cadence: rebaked every 8 frames at offset 0 (round-robin with D_ambient
// at offset 4). Driven from RtxAtmosphere::computeLuts.

#include "rtx/pass/atmosphere/atmosphere_common.slangh"

layout(binding = 0)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

layout(binding = 1, r16f)
RWTexture3D<float> cloudDSunGridOutput;

// 2D Texture3D + linear/REPEAT sampler for the prebaked Perlin FBM cloud
// noise volume — same resource that sampleCloudDensityTextured consumes
// elsewhere. Both are bound by the dispatch site in RtxAtmosphere; the
// REPEAT sampler matches the horizontal tile-wrap of the voxel grid.
layout(binding = 2)
Texture3D<float> cloudNoise3DSrv;

layout(binding = 3)
SamplerState cloudNoise3DSamplerState;

// Cloud placement map (fork — 2026-06-11, column-shaping rework). Same
// resource the view march samples; the column model inside
// sampleCloudDensityForShadow reads it so the baked shadows match the
// per-column cloud shapes overhead.
layout(binding = 4)
Texture2D<float4> cloudPlacementMapSrv;

// Grid dimensions kept in lockstep with RtxAtmosphere::kCloudVoxelGridX/Y/Z.
static const uint kGridX = 256u;
static const uint kGridY = 256u;
static const uint kGridZ = 32u;

[shader("compute")]
[numthreads(8, 8, 4)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
  if (any(dispatchId >= uint3(kGridX, kGridY, kGridZ))) {
    return;
  }

  AtmosphereArgs args = atmosphereArgs;

  // Voxel center in UVW. The +0.5 puts the sample at the voxel center so
  // adjacent voxels integrate offset positions (no "ringing" between voxels).
  vec3 uvw = (vec3(dispatchId) + 0.5f) / vec3(float(kGridX), float(kGridY), float(kGridZ));

  // UVW -> world (Y-up, km), camera-centered.
  vec3 worldPosYUp = cloudVoxelUVWToWorld(uvw, args);

  // Sun direction is authored in Y-up space.
  vec3 sunDirYUp = normalize(args.sunDirection);

  // March integral. Returns 0 if the ray never reaches the slab top.
  float opticalDepth = sampleCloudSunOpticalDepthAtWorld(
      worldPosYUp, sunDirYUp, args, cloudNoise3DSrv, cloudNoise3DSamplerState,
      cloudPlacementMapSrv);

  cloudDSunGridOutput[dispatchId] = opticalDepth;
}
