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

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <vector>

// Holds the parts of the AMD FSR integration that need neither a device nor the FSR runtime,
// so the unit tests can cover them.

namespace dxvk {

  // The numeric values are stored in the config files.
  enum class FSRPreset : int {
    UltraPerformance = 0,
    Performance,
    Balanced,
    Quality,
    NativeAA,
  };

  namespace fsr {

    // Returns the per-dimension upscale ratio AMD defines for the preset.
    inline float presetUpscaleRatio(FSRPreset preset) {
      switch (preset) {
      case FSRPreset::UltraPerformance: return 3.0f;
      case FSRPreset::Performance:      return 2.0f;
      case FSRPreset::Balanced:         return 1.7f;
      case FSRPreset::Quality:          return 1.5f;
      case FSRPreset::NativeAA:         return 1.0f;
      }

      return 1.0f;
    }

    // Returns the render resolution as a percentage of the display resolution, per dimension.
    inline float presetScreenPercentage(FSRPreset preset) {
      return 100.0f / presetUpscaleRatio(preset);
    }

    inline const char* presetName(FSRPreset preset) {
      switch (preset) {
      case FSRPreset::UltraPerformance: return "Ultra Performance";
      case FSRPreset::Performance:      return "Performance";
      case FSRPreset::Balanced:         return "Balanced";
      case FSRPreset::Quality:          return "Quality";
      case FSRPreset::NativeAA:         return "Native AA";
      }

      return "Unknown";
    }

    // Returns the texture LOD bias for content rendered at renderWidth and upscaled to displayWidth.
    // AMD recommends log2(render / display) - 1, and the offset carries that last term.
    inline float mipBias(uint32_t renderWidth, uint32_t displayWidth, float offset) {
      if (renderWidth == 0 || displayWidth == 0) {
        return offset;
      }

      return std::log2(float(renderWidth) / float(displayWidth)) + offset;
    }

    // Stands in for the frame time while it is not known, as on the first frame.
    constexpr float kDefaultFrameTimeMs = 1000.0f / 60.0f;

    // FSR's debug checker treats anything shorter as a time in the wrong unit.
    constexpr float kMinFrameTimeMs = 1.0f;

    // Returns the frame time to give FSR, in milliseconds.
    // The time is measured on the render thread, where two frames that arrive back to back read as almost none.
    inline float frameTimeMs(float measuredFrameTimeMs) {
      if (!std::isfinite(measuredFrameTimeMs) || measuredFrameTimeMs <= 0.0f) {
        return kDefaultFrameTimeMs;
      }

      return std::max(measuredFrameTimeMs, kMinFrameTimeMs);
    }

    struct DepthParameters {
      // Set when depth maps the near plane to 1 and the far plane to 0.
      bool inverted = false;
      // Set when depth never reaches its far value, which puts the far plane at infinity.
      bool infinite = false;
      float nearDistance = 0.0f;
      // Holds the far plane's distance, which is unused while the far plane is at infinity.
      float farDistance = 0.0f;
    };

    // A far plane this many times further than the near plane is as good as infinite in 32-bit depth.
    constexpr float kInfiniteFarPlaneRatio = 1.0e6f;

    // Reads the planes from a perspective projection with depth in [0, 1].
    // The projection gives clip z = zScale * z + zOffset and clip w = wScale * z, for a view-space z.
    // Depth is then a + b / t at view distance t, with a = zScale / wScale and b = zOffset / |wScale|.
    // A projection whose depth at infinity stays inside [0, 1] has no far plane, which is how UE3 builds its own.
    // fallbackNear is returned with an infinite far plane when the projection is not a perspective one.
    inline DepthParameters depthParametersFromProjection(float zScale, float zOffset, float wScale,
                                                         bool inverted, float fallbackNear) {
      DepthParameters parameters;
      parameters.inverted = inverted;
      parameters.infinite = true;
      parameters.nearDistance = fallbackNear;

      if (std::abs(wScale) < 1.0e-6f || !std::isfinite(zScale) || !std::isfinite(zOffset)) {
        return parameters;
      }

      const float depthAtInfinity = zScale / wScale;
      const float depthSlope = zOffset / std::abs(wScale);

      // Standard depth is 0 at the near plane and 1 at the far plane, and inverted depth swaps the two values.
      const float nearDistance = inverted
        ? depthSlope / (1.0f - depthAtInfinity)
        : -depthSlope / depthAtInfinity;

      if (!std::isfinite(nearDistance) || nearDistance <= 0.0f) {
        return parameters;
      }

      parameters.nearDistance = nearDistance;

      const bool reachesFarValue = inverted ? depthAtInfinity < 0.0f : depthAtInfinity > 1.0f;

      if (reachesFarValue) {
        const float farDistance = inverted
          ? -depthSlope / depthAtInfinity
          : depthSlope / (1.0f - depthAtInfinity);

        if (std::isfinite(farDistance) && farDistance > nearDistance &&
            farDistance < nearDistance * kInfiniteFarPlaneRatio) {
          parameters.infinite = false;
          parameters.farDistance = farDistance;
        }
      }

      return parameters;
    }

    // Writes the planes the way AMD's samples pass them.
    // Inverted depth swaps the two, and an infinite far plane is FLT_MAX.
    inline void cameraPlanes(const DepthParameters& parameters, float& cameraNear, float& cameraFar) {
      const float farDistance = parameters.infinite ? FLT_MAX : parameters.farDistance;

      cameraNear = parameters.inverted ? farDistance : parameters.nearDistance;
      cameraFar = parameters.inverted ? parameters.nearDistance : farDistance;
    }

    // These match the VkQueueFlagBits values.
    constexpr uint32_t kQueueGraphics = 0x1;
    constexpr uint32_t kQueueCompute = 0x2;
    constexpr uint32_t kQueueTransfer = 0x4;

    struct QueueFamily {
      uint32_t flags = 0;
      // Counts the queues of the family that no other role uses yet.
      uint32_t queueCount = 0;
    };

    // Returns the queue families to try for the frame generation image acquire queue, best first.
    // That queue only ever takes empty submissions that wait on and signal semaphores, so any family serves.
    // The families such submissions are ordinary on come first, starting with the one the renderer needs least.
    // A family without a queue to spare is left out.
    inline std::vector<uint32_t> imageAcquireFamilyPreference(const std::vector<QueueFamily>& families) {
      const auto rank = [] (uint32_t flags) -> uint32_t {
        const uint32_t workFlags = flags & (kQueueGraphics | kQueueCompute | kQueueTransfer);

        if (workFlags == kQueueTransfer) {
          return 0;
        } else if ((workFlags & kQueueGraphics) == 0 && (workFlags & kQueueCompute) != 0) {
          return 1;
        } else if ((workFlags & kQueueGraphics) != 0) {
          return 2;
        }

        // A family with none of the three is typically a video queue.
        return 3;
      };

      std::vector<uint32_t> order;
      for (uint32_t i = 0; i < families.size(); i++) {
        if (families[i].queueCount > 0) {
          order.push_back(i);
        }
      }

      std::stable_sort(order.begin(), order.end(), [&] (uint32_t a, uint32_t b) {
        return rank(families[a].flags) < rank(families[b].flags);
      });

      return order;
    }

    struct QueueCandidate {
      // Holds the VkQueue.
      // Holds zero when the device has no such queue.
      uintptr_t handle = 0;
      uint32_t family = 0;
      uint32_t flags = 0;
      bool canPresent = false;
    };

    struct FrameGenQueues {
      QueueCandidate present;
      QueueCandidate imageAcquire;
      // Stays empty when no compute queue is left, which only costs asynchronous interpolation.
      QueueCandidate asyncCompute;
      bool valid = false;
    };

    // Picks the queues the frame generation swapchain needs next to the game's own queue.
    // The swapchain submits to them from its own threads, so each has to be a queue nothing else uses,
    // and all of them have to be distinct.
    // Candidates are listed best first, and an entry with a zero handle is skipped.
    inline FrameGenQueues selectFrameGenQueues(const QueueCandidate& game,
                                               const std::vector<QueueCandidate>& presentCandidates,
                                               const std::vector<QueueCandidate>& imageAcquireCandidates,
                                               const std::vector<QueueCandidate>& asyncComputeCandidates) {
      FrameGenQueues queues;

      std::array<uintptr_t, 4> taken = { game.handle, 0, 0, 0 };
      uint32_t takenCount = 1;

      const auto isTaken = [&] (uintptr_t handle) {
        return std::find(taken.begin(), taken.begin() + takenCount, handle) != taken.begin() + takenCount;
      };

      const auto pick = [&] (const std::vector<QueueCandidate>& candidates, bool needsPresent, bool needsCompute) {
        for (const QueueCandidate& candidate : candidates) {
          if (candidate.handle == 0 || isTaken(candidate.handle)) {
            continue;
          }

          if (needsPresent && !candidate.canPresent) {
            continue;
          }

          if (needsCompute && (candidate.flags & kQueueCompute) == 0) {
            continue;
          }

          taken[takenCount++] = candidate.handle;
          return candidate;
        }

        return QueueCandidate();
      };

      if (game.handle == 0) {
        return queues;
      }

      queues.present = pick(presentCandidates, true, false);
      queues.imageAcquire = pick(imageAcquireCandidates, false, false);

      if (queues.present.handle == 0 || queues.imageAcquire.handle == 0) {
        return queues;
      }

      queues.asyncCompute = pick(asyncComputeCandidates, false, true);
      queues.valid = true;

      return queues;
    }

    // The frame generation swapchain creates this many images for the game to render into,
    // and asks the surface for at least as many real ones.
    // Two presents per rendered frame need three, and the swapchain has room for six.
    constexpr uint32_t kMinProxyImageCount = 3;
    constexpr uint32_t kMaxProxyImageCount = 6;

    // surfaceMaxImageCount is zero when the surface sets no upper limit.
    inline uint32_t proxyImageCount(uint32_t requestedImageCount, uint32_t surfaceMinImageCount, uint32_t surfaceMaxImageCount) {
      uint32_t count = std::clamp(std::max(requestedImageCount, surfaceMinImageCount), kMinProxyImageCount, kMaxProxyImageCount);

      if (surfaceMaxImageCount != 0) {
        count = std::min(count, surfaceMaxImageCount);
      }

      return count;
    }

  } // namespace fsr

} // namespace dxvk
