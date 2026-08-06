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
// Nubis3 density model + the bake integrals themselves (moved there,
// fork — Nubis3 conversion Phase B).
#include "rtx/pass/atmosphere/cloud_nubis3_common.slangh"

layout(binding = 0)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

layout(binding = 1, r16f)
RWTexture3D<float> cloudDSunGridOutput;

// Linear/REPEAT sampler for the Nubis3 volume taps; the REPEAT wrap matches
// the horizontal tile-wrap of the voxel grid. (Bindings 2/4 held the legacy
// 256^3 noise volume + placement map — retired with the legacy model.)
layout(binding = 3)
SamplerState cloudNoise3DSamplerState;

// Nubis3 NVDF SDF + detail volume (fork — Nubis3 conversion Phase B). The
// bake integrand samples the same iso-surface the view march renders — the
// terrain cloud-shadow NEE fold reads this grid with zero integrator
// changes.
layout(binding = 5)
Texture3D<float> cloudNvdfSdfSrv;

layout(binding = 6)
Texture3D<float4> cloudDetailNoiseSrv;

// Grid dimensions kept in lockstep with RtxAtmosphere::kCloudVoxelGridX/Y/Z.
// Axis fix 2026-07-16: uvw.y = VERTICAL gets the 32-texel axis, uvw.z =
// world Z gets 256 (see the note on RtxAtmosphere::kCloudVoxelGridX/Y/Z —
// the old Y=256/Z=32 allocation made world-Z shadow texels 375 m wide: the
// blocky "squares" cast into the deck).
static const uint kGridX = 256u;
static const uint kGridY = 32u;
static const uint kGridZ = 256u;

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
      worldPosYUp, sunDirYUp, args, cloudNoise3DSamplerState,
      cloudNvdfSdfSrv, cloudDetailNoiseSrv);

  cloudDSunGridOutput[dispatchId] = opticalDepth;
}
