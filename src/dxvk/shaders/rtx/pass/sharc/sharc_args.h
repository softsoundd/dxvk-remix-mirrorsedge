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
#pragma once

#include "rtx/utility/shader_types.h"

// Update paths also deposit their camera-visible primary vertex, valued as the direct pass's
// RTXDI lighting plus the sampled continuation.
#define SHARC_FLAG_UPDATE_PRIMARY_VERTEX 0x1u
// Cache insertion and reuse are allowed at rough opaque surfaces reached by a non-diffuse lobe.
#define SHARC_FLAG_ALLOW_SPECULAR_PATHS  0x2u
// How many times an update path whose first bounce escaped to the sky re-samples that bounce.
#define SHARC_SKY_RETRY_SHIFT            2u
#define SHARC_SKY_RETRY_MASK             0x7u

struct SharcArgs {
  vec3 cameraPosition;
  uint capacity;

  vec3 cameraPositionPrev;
  float gridScale;

  uint accumulationFrames;
  uint staleFrames;
  uint updateTileSize;
  uint updateBounces;

  float radianceScale;
  // Isotropic roughness (GGX alpha) a surface must reach to hold a cell. A cell carries one
  // non-directional radiance value, so a surface below this reflects more sharply than the cell
  // can represent.
  float minRoughness;
  // The cache stores reflected radiance and the path adds emission separately, so an emissive
  // surface is excluded. Comparing luminance against a threshold rather than testing for any
  // emission at all keeps surfaces carrying a faint emissive map eligible.
  float maxEmissiveLuminance;
  // Cells answer a query once accumulatedSampleNum exceeds this. At zero a single sample reads
  // back as converged radiance, which makes newly revealed geometry glow.
  uint minSampleCount;

  uint flags;
  // Ceiling on one deposit into a cell, as a multiple of what the cell already holds; 0 disables it.
  // A cell is a mean of its samples, so an outlier is divided by the sample count rather than
  // averaged away. Scaling the limit by the cell's own value lets a bright cell keep accepting
  // bright deposits while a dark one refuses spikes.
  float maxDepositRatio;
  // Absolute luminance floor under that ceiling, so a cell near black can still brighten when the
  // lighting changes.
  float minDepositCeiling;
  // Isotropic roughness (GGX alpha) floor applied to a material during the update stage only,
  // before the continuation is sampled and before NEE is evaluated; 0 disables it. Separate from
  // minRoughness and applied after it, so it never feeds query eligibility: it changes what a cell
  // stores, not which surfaces have one.
  float updateRoughnessClamp;
};

#ifdef __cplusplus
static_assert(sizeof(SharcArgs) == 80);
// Every struct in the RaytraceArgs block must be a whole number of 16B rows. The shader lays the
// block out with scalar rules (-fvk-use-scalar-layout) and pads nothing, while the C++ struct it is
// memcpy'd from carries alignas(16) on vec4 and mat4. A struct whose size is not a multiple of 16
// moves renderTargetCamera, the first alignas(16) member after the block, in the C++ view alone,
// and with it the path-tracer loop bounds that follow it.
static_assert(sizeof(SharcArgs) % 16 == 0,
              "SharcArgs must be a whole number of 16B rows.");
#endif
