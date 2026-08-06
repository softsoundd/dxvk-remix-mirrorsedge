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

// Cloud D_ambient voxel grid bake (Nubis Cubed 2023, fork — 2026-05-12).
//
// Bakes summed optical depth straight up (zenith) at each voxel of a
// camera-centered 256x256x32 R16F grid. Consumed at shade time via
// sampleDAmbient() for the Nubis Cubed page-142 ambient attenuation term
// landing in C4-C6 of the 2026-05-12 workstream.
//
// Cadence: rebaked every 8 frames at offset 4 (round-robin with D_sun
// at offset 0). Driven from RtxAtmosphere::computeLuts.

#include "rtx/pass/atmosphere/atmosphere_common.slangh"
// Nubis3 density model + the bake integrals themselves (moved there,
// fork — Nubis3 conversion Phase B).
#include "rtx/pass/atmosphere/cloud_nubis3_common.slangh"

layout(binding = 0)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

layout(binding = 1, r16f)
RWTexture3D<float> cloudDAmbientGridOutput;

// Linear/REPEAT sampler for the Nubis3 volume taps; horizontal REPEAT
// matches the tile-wrap of the voxel grid. (Bindings 2/4 held the legacy
// 256^3 noise volume + placement map — retired with the legacy model.)
layout(binding = 3)
SamplerState cloudNoise3DSamplerState;

// Nubis3 NVDF SDF + detail volume (fork — Nubis3 conversion Phase B). Same
// integrand as cloud_sun_density_grid.comp.slang.
layout(binding = 5)
Texture3D<float> cloudNvdfSdfSrv;

layout(binding = 6)
Texture3D<float4> cloudDetailNoiseSrv;

// Grid dimensions kept in lockstep with RtxAtmosphere::kCloudVoxelGridX/Y/Z.
// Axis fix 2026-07-16: uvw.y = VERTICAL gets the 32-texel axis, uvw.z =
// world Z gets 256 (see cloud_sun_density_grid.comp.slang).
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

  vec3 uvw = (vec3(dispatchId) + 0.5f) / vec3(float(kGridX), float(kGridY), float(kGridZ));

  // UVW -> world (Y-up, km).
  vec3 worldPosYUp = cloudVoxelUVWToWorld(uvw, args);

  // Zenith integral up to the slab top.
  float opticalDepth = sampleCloudAmbientOpticalDepthAtWorld(
      worldPosYUp, args, cloudNoise3DSamplerState,
      cloudNvdfSdfSrv, cloudDetailNoiseSrv);

  cloudDAmbientGridOutput[dispatchId] = opticalDepth;
}
