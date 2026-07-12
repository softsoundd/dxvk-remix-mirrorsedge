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
    MaxNumQueuedCommandBuffers  =    18,
    MaxNumQueryCountPerPool     =   128,
    MaxNumSpecConstants         =    14,
    MaxUniformBufferSize        = 65536,
    MaxVertexBindingStride      =  2048,
    // NV-DXVK start: increase push constant budget for fork tonemap operator args.
    //
    // Bumped 128 -> 256 because the fork-side tonemap apply args grew past
    // 128 bytes once the new operator parameter blocks were added.
    // `ToneMappingApplyToneMappingArgs` is currently 176 bytes — see the
    // `static_assert` at the bottom of
    // `shaders/rtx/pass/tonemap/tonemapping.h`. The Vulkan spec minimum
    // guarantee is 128, but every RTX-class GPU Remix targets reports
    // `maxPushConstantsSize >= 256`.
    //
    // With the old 128-byte cap, `pushConstants()` overflowed
    // `m_state.pc.data[bank]` on every tonemap dispatch, corrupting
    // adjacent bank state and crashing the NVIDIA driver later inside
    // `nvoglv64.dll` (release builds compile out the size assert in
    // `dxvk_context.cpp`, so the overflow was silent). `dxvk_context.cpp`'s
    // memcpy + this storage size are the only consumers; pipeline layouts
    // derive push-constant ranges from per-shader reflection, so
    // increasing this is harmless to existing shaders.
    MaxPushConstantSize         =   256,
    // NV-DXVK end
  };
  
}