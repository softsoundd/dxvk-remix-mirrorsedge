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

// Messages between the runtime and the 32-bit bridge client for the patches the client applies to the
// game executable. See documentation/UE3Compatibility.md, "Game executable patches".

#include <cmath>
#include <cstdint>

namespace dxvk {
  // Runtime -> bridge client. wParam: GamePatchBits of the patches that should be active.
  // lParam: the frustum bypass limits, see packFrustumBypassLimits.
  static constexpr const char* kGamePatchRequestMsgName = "UWM_REMIX_GAME_PATCH_REQUEST";
  // Bridge client -> runtime, answering every request. wParam: GamePatchBits currently active.
  // lParam: GamePatchBits whose patch sites the executable lacks.
  static constexpr const char* kGamePatchStatusMsgName = "UWM_REMIX_GAME_PATCH_STATUS";

  enum GamePatchBits : uint32_t {
    kGamePatchDisableFrustumCulling = 1u << 0,
    kGamePatchShowThirdPersonModel = 1u << 1,
    // Only together with kGamePatchDisableFrustumCulling: bypass culling only within the request's limits.
    kGamePatchLimitFrustumBypass = 1u << 2,
    kGamePatchDisableOcclusionQueries = 1u << 3,
    kGamePatchDisableSceneCaptures = 1u << 4,
    kGamePatchDisableDynamicShadows = 1u << 5,
    kGamePatchDisableDynamicLighting = 1u << 6,
    kGamePatchDisableVelocityPass = 1u << 7,
  };

  // Limits travel as two 16-bit counts of this many game units: the distance within which an off-screen
  // primitive is kept in the low half, and the bounding radius from which it is kept at any distance in the
  // high half, where 0 keeps none by size.
  static constexpr float kFrustumBypassLimitUnit = 16.0f;

  inline uint32_t packFrustumBypassLimits(const float maxDistance, const float minRadius) {
    const auto quantize = [](const float value) {
      const float steps = std::ceil(value / kFrustumBypassLimitUnit);
      return steps <= 0.0f ? 0u : steps >= 65535.0f ? 0xFFFFu : static_cast<uint32_t>(steps);
    };
    return quantize(maxDistance) | (quantize(minRadius) << 16);
  }

  inline float unpackFrustumBypassMaxDistance(const uint32_t limits) {
    return static_cast<float>(limits & 0xFFFFu) * kFrustumBypassLimitUnit;
  }

  inline float unpackFrustumBypassMinRadius(const uint32_t limits) {
    return static_cast<float>(limits >> 16) * kFrustumBypassLimitUnit;
  }
}
