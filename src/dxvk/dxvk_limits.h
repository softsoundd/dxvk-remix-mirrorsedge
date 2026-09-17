#pragma once

#include "dxvk_include.h"

namespace dxvk {
  
  enum DxvkLimits : size_t {
    MaxNumRenderTargets         =     8,
    MaxNumVertexAttributes      =    32,
    MaxNumVertexBindings        =    32,
    MaxNumXfbBuffers            =     4,
    MaxNumXfbStreams            =     4,
    MaxNumViewports             =    16,
    MaxNumResourceSlots         =  1216,
    MaxNumActiveBindings        =   384,
    // NV-DXVK start: Remix records well over a dozen command lists per frame and the GPU-heavy injectRTX
    // lists come last, so with room for barely one frame the CS thread blocks in DxvkSubmissionQueue::submit
    // whenever the application runs a frame ahead. Sized for about three frames of lists.
    MaxNumQueuedCommandBuffers  =    48,
    // NV-DXVK end
    MaxNumQueryCountPerPool     =   128,
    MaxNumSpecConstants         =    14,
    MaxUniformBufferSize        = 65536,
    MaxVertexBindingStride      =  2048,
    MaxPushConstantSize         =   128,
  };
  
}