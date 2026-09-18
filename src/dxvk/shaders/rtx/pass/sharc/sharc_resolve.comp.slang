// Copyright (c) 2026, NVIDIA CORPORATION. SPDX-License-Identifier: MIT
#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/sharc/sharc_bindings.slangh"

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 index : SV_DispatchThreadID) {
  if (index.x >= cb.sharcArgs.capacity) {
    return;
  }
  SharcResolveParameters p;
  p.cameraPositionPrev = cb.sharcArgs.cameraPositionPrev;
  p.accumulationFrameNum = cb.sharcArgs.accumulationFrames;
  p.responsiveFrameNum = 1;
  p.staleFrameNumMax = cb.sharcArgs.staleFrames;
  p.frameIndex = cb.frameIdx;
  SharcResolveEntry(index.x, makeSharcParameters(), p);
}
