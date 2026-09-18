// Copyright (c) 2026, NVIDIA CORPORATION. SPDX-License-Identifier: MIT
#pragma once
#include "rtx/utility/shader_types.h"

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
  float minRoughness;
  // Bit 0 is set whenever the cache is active and is read by nothing. The name is kept so
  // that no shader's debug names change. Bit 1: update paths also deposit their primary
  // vertex (RTXDI direct light plus the camera-side continuation), so a path that exits to
  // the sky at its first bounce still feeds one cell. Bits 2..4: how many times a
  // first-bounce sky miss is re-sampled from a cosine lobe about the primary normal.
  uint enabled;
  uint allowSpecularPaths;
  // Emissive surfaces are excluded because the cache stores reflected radiance and the
  // path adds emission separately. A strict "any emission at all" test disqualifies every
  // surface carrying a faint emissive map, which in Portal RTX is most of them, so compare
  // luminance against a threshold instead. Zero reproduces the original behaviour.
  float maxEmissiveLuminance;
  // A path that arrived by a specular lobe only reaches the cache because allowSpecularPaths
  // waived the lobe check. Serving one from an isotropic cache on a smooth surface glows, so
  // those paths get their own, stricter roughness floor.
  float minRoughnessSpecular;
  // Cells are readable once accumulatedSampleNum exceeds this. The SDK default of 0 lets a
  // cell answer from a single sample, which is what makes newly revealed geometry glow: one
  // bright path lands in an empty cell and is read back as if it had converged.
  uint minSampleCount;
  // Query-side gate for specular arrivals: test the footprint of the lobe that launched the
  // segment against the voxel, NVIDIA's prescribed check, instead of the hit material's
  // roughness. While set, minRoughnessSpecular is unused and both lobe classes share minRoughness.
  uint footprintGate;
  // The brightest single deposit an update path may write into a cell, as luminance; 0 lets
  // any value through, as rtx.fireflyFilteringLuminanceThreshold's own convention does. A cell
  // is a mean of accumulated samples, so an outlier is not averaged away, only divided by the
  // sample count -- and that count falls with render resolution because the update dispatch is
  // one path per tile of the render target. Bounding the deposit is the only remedy that costs
  // no coverage: it refuses no query, rejects no surface and creates no cell that did not exist.
  // Applies to the deferred update backend, the shipping one.
  float maxDepositLuminance;
  // Every struct in RaytraceArgs must be a whole number of 16B rows, and this pad is what keeps
  // that true here. The shader lays the block out with scalar rules (-fvk-use-scalar-layout) and
  // pads nothing; the C++ struct it is memcpy'd from carries alignas(16) on vec4 and mat4. The
  // two agree only while no C++ padding appears: at 84 bytes this struct pushed
  // renderTargetCamera, the first alignas(16) member after it, 12 bytes past where every shader
  // reads it, and with it frameIdx, pathMaxBounces, secondaryRayMaxInteractions and
  // numActiveRayPortals -- path-tracer loop bounds read as garbage, which hangs the GPU.
  uint pad0;
  uint pad1;
  uint pad2;
};
#define SHARC_UPDATE_FLAG_PRIMARY_VERTEX 2u
#define SHARC_UPDATE_SKY_RETRY_SHIFT 2u
#define SHARC_UPDATE_SKY_RETRY_MASK 7u
#ifdef __cplusplus
static_assert(sizeof(SharcArgs) == 96);
// The invariant the size above exists to hold. Keep both: the first catches an unintended
// change, the second explains which change is never allowed.
static_assert(sizeof(SharcArgs) % 16 == 0,
              "SharcArgs must be a whole number of 16B rows; see the comment on pad0.");
#endif
