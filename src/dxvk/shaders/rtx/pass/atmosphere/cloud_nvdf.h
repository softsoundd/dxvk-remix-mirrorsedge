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

#ifndef CLOUD_NVDF_H
#define CLOUD_NVDF_H

#include "rtx/utility/shader_types.h"

// Cloud NVDF (Nubis Voxel Data Field) — fork, Nubis3 conversion Phase A.
//
// Shared CPU/GPU constants for the cloud-body SDF bake chain:
//   occupancy voxelize -> JFA seed init -> JFA jump passes -> signed resolve
//
// The NVDF voxelizes the procedural cloud BODY (placement map + column
// model — the "Frankencloudscape" in Nubis3 terms) into a tile-periodic
// grid, then a wrap-aware jump-flooding pass chain turns it into a true
// signed distance field. Detail noise never enters the SDF — it stays a
// sample-time erosion, exactly as in Nubis3 (SIGGRAPH 2023).
//
// Texture axis convention (explicit — do NOT confuse with the D_sun voxel
// grids, whose comments and UVW mapping disagree):
//   texture x (u) = world X, CLOUD_NVDF_SIZE_XZ texels, tile-wrapped
//   texture y (v) = VERTICAL (slab height fraction), CLOUD_NVDF_SIZE_Y texels, clamped
//   texture z (w) = world Z, CLOUD_NVDF_SIZE_XZ texels, tile-wrapped
// Horizontal domain is one cloudNoiseTileKm period (world-anchored, wind
// applied at sample time); vertical domain is [cloudAltitude,
// cloudAltitude + cloudThickness]. At the 12 km / 3 km defaults a voxel is
// ~47 m in XZ and ~47 m in Y — the body model has no content finer than the
// >= 1 km placement cells, so this is ample; erosion owns everything finer.
//
// Keep in lockstep with RtxAtmosphere::kCloudNvdfSizeXZ / kCloudNvdfSizeY.
#define CLOUD_NVDF_SIZE_XZ 256
#define CLOUD_NVDF_SIZE_Y  64

// Pass-local binding slots. Each NVDF pass owns its tiny descriptor set;
// nothing here touches the common atmosphere binding range (200-216) — the
// path tracer never samples the NVDF (terrain shadows keep reading D_sun).
// Keep each block in lockstep with the BEGIN_PARAMETER block of the matching
// ManagedShader class in rtx_atmosphere.cpp.
#define CLOUD_NVDF_OCCUPANCY_BINDING_CONSTANTS        0
#define CLOUD_NVDF_OCCUPANCY_BINDING_OUTPUT           1
#define CLOUD_NVDF_OCCUPANCY_BINDING_PLACEMENT_INPUT  2
#define CLOUD_NVDF_OCCUPANCY_BINDING_SAMPLER          3

#define CLOUD_NVDF_JFA_BINDING_CONSTANTS        0
#define CLOUD_NVDF_JFA_BINDING_OCCUPANCY_INPUT  1
#define CLOUD_NVDF_JFA_BINDING_SEEDS_INPUT      2
#define CLOUD_NVDF_JFA_BINDING_SEEDS_OUTPUT     3

#define CLOUD_NVDF_RESOLVE_BINDING_CONSTANTS        0
#define CLOUD_NVDF_RESOLVE_BINDING_OCCUPANCY_INPUT  1
#define CLOUD_NVDF_RESOLVE_BINDING_SEEDS_INPUT      2
#define CLOUD_NVDF_RESOLVE_BINDING_SDF_OUTPUT       3

// JFA pass push constants. mode 0 = seed init (reads occupancy, writes
// boundary-voxel seeds), mode 1 = jump pass at jumpSizeVoxels (reads seeds,
// writes refined seeds). Push constants instead of AtmosphereArgs fields so
// the 9-pass chain never churns the shared constants buffer mid-frame.
struct CloudNvdfJfaArgs {
  uint jumpSizeVoxels;
  uint mode;
  uint pad0;
  uint pad1;
};

#endif  // CLOUD_NVDF_H
