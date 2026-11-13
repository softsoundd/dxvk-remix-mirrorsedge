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
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../test_utils.h"
#include "../../../src/dxvk/rtx_render/rtx_fsr_helpers.h"

using namespace dxvk;

namespace {

  void expect(bool condition, const std::string& message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  // The tolerance is relative to the expected value, and absolute below 1.
  void expectNear(double actual, double expected, double tolerance, const std::string& label) {
    if (std::abs(actual - expected) > tolerance * std::max(std::abs(expected), 1.0)) {
      std::ostringstream message;
      message << label << ": expected " << expected << ", got " << actual;
      throw std::runtime_error(message.str());
    }
  }

  // The ratios are the ones AMD defines for the quality modes, and the config files store the preset's number.
  void testPresets() {
    std::cout << "  testPresets" << std::endl;

    expect(int(FSRPreset::UltraPerformance) == 0 && int(FSRPreset::Performance) == 1 &&
           int(FSRPreset::Balanced) == 2 && int(FSRPreset::Quality) == 3 && int(FSRPreset::NativeAA) == 4,
           "the preset numbers stored in the config files changed");

    expectNear(fsr::presetUpscaleRatio(FSRPreset::NativeAA), 1.0, 1e-6, "Native AA ratio");
    expectNear(fsr::presetUpscaleRatio(FSRPreset::Quality), 1.5, 1e-6, "Quality ratio");
    expectNear(fsr::presetUpscaleRatio(FSRPreset::Balanced), 1.7, 1e-6, "Balanced ratio");
    expectNear(fsr::presetUpscaleRatio(FSRPreset::Performance), 2.0, 1e-6, "Performance ratio");
    expectNear(fsr::presetUpscaleRatio(FSRPreset::UltraPerformance), 3.0, 1e-6, "Ultra Performance ratio");

    expectNear(fsr::presetScreenPercentage(FSRPreset::NativeAA), 100.0, 1e-5, "Native AA screen percentage");
    expectNear(fsr::presetScreenPercentage(FSRPreset::Quality), 66.6667, 1e-5, "Quality screen percentage");
    expectNear(fsr::presetScreenPercentage(FSRPreset::Balanced), 58.8235, 1e-5, "Balanced screen percentage");
    expectNear(fsr::presetScreenPercentage(FSRPreset::Performance), 50.0, 1e-5, "Performance screen percentage");
    expectNear(fsr::presetScreenPercentage(FSRPreset::UltraPerformance), 33.3333, 1e-5, "Ultra Performance screen percentage");

    expect(std::string(fsr::presetName(FSRPreset::NativeAA)) == "Native AA", "Native AA name");
    expect(std::string(fsr::presetName(FSRPreset::UltraPerformance)) == "Ultra Performance", "Ultra Performance name");
  }

  // AMD recommends a texture LOD bias of log2(render / display) - 1.
  void testMipBias() {
    std::cout << "  testMipBias" << std::endl;

    expectNear(fsr::mipBias(3840, 3840, -1.0f), -1.0, 1e-5, "Native AA mip bias");
    expectNear(fsr::mipBias(2560, 3840, -1.0f), -1.58496, 1e-5, "Quality mip bias");
    expectNear(fsr::mipBias(2259, 3840, -1.0f), -1.76534, 1e-4, "Balanced mip bias");
    expectNear(fsr::mipBias(1920, 3840, -1.0f), -2.0, 1e-5, "Performance mip bias");
    expectNear(fsr::mipBias(1280, 3840, -1.0f), -2.58496, 1e-5, "Ultra Performance mip bias");

    expectNear(fsr::mipBias(1920, 3840, 0.0f), -1.0, 1e-5, "mip bias without the offset");
    expectNear(fsr::mipBias(0, 3840, -1.0f), -1.0, 1e-6, "mip bias without a render size");
    expectNear(fsr::mipBias(1920, 0, -1.0f), -1.0, 1e-6, "mip bias without a display size");
  }

  // FSR treats a frame time below a millisecond as an error, and the first frame has none to give.
  void testFrameTime() {
    std::cout << "  testFrameTime" << std::endl;

    expectNear(fsr::frameTimeMs(8.3f), 8.3, 1e-6, "a measured frame time");
    expectNear(fsr::frameTimeMs(0.25f), fsr::kMinFrameTimeMs, 1e-6, "a frame time below a millisecond");
    expectNear(fsr::frameTimeMs(0.0f), fsr::kDefaultFrameTimeMs, 1e-6, "a frame time of zero");
    expectNear(fsr::frameTimeMs(-4.0f), fsr::kDefaultFrameTimeMs, 1e-6, "a negative frame time");
    expectNear(fsr::frameTimeMs(std::nanf("")), fsr::kDefaultFrameTimeMs, 1e-6, "a frame time that is not a number");
    expectNear(fsr::frameTimeMs(HUGE_VALF), fsr::kDefaultFrameTimeMs, 1e-6, "an infinite frame time");
  }

  struct ProjectionDepthTerms {
    float zScale;
    float zOffset;
    float wScale;
  };

  // Returns the depth terms of UE3's player projection, FPerspectiveMatrix(FOV, Width, Height, MinZ).
  // Its far plane is infinite, and depth approaches 1 - Z_PRECISION with distance.
  ProjectionDepthTerms ue3Projection(float nearPlane) {
    constexpr float kZPrecision = 0.001f;

    return { 1.0f - kZPrecision, -nearPlane * (1.0f - kZPrecision), 1.0f };
  }

  ProjectionDepthTerms finiteProjection(float nearPlane, float farPlane) {
    return { farPlane / (farPlane - nearPlane), -nearPlane * farPlane / (farPlane - nearPlane), 1.0f };
  }

  // Returns the terms with clip z = w - z, as a reversed depth patch makes them, which maps depth d to 1 - d.
  ProjectionDepthTerms reverseDepth(const ProjectionDepthTerms& terms) {
    return { terms.wScale - terms.zScale, -terms.zOffset, terms.wScale };
  }

  fsr::DepthParameters depthParameters(const ProjectionDepthTerms& terms, bool inverted, float fallbackNear = 1.0f) {
    return fsr::depthParametersFromProjection(terms.zScale, terms.zOffset, terms.wScale, inverted, fallbackNear);
  }

  void testDepthParameters() {
    std::cout << "  testDepthParameters" << std::endl;

    for (const float nearPlane : { 10.0f, 3.0f, 0.5f }) {
      const std::string name = "near plane " + std::to_string(nearPlane);

      const fsr::DepthParameters standard = depthParameters(ue3Projection(nearPlane), false);
      expect(!standard.inverted && standard.infinite, name + ": UE3's projection is not read as standard infinite depth");
      expectNear(standard.nearDistance, nearPlane, 1e-4, name + ": standard near distance");

      const fsr::DepthParameters reversed = depthParameters(reverseDepth(ue3Projection(nearPlane)), true);
      expect(reversed.inverted && reversed.infinite, name + ": the reversed projection is not read as inverted infinite depth");
      expectNear(reversed.nearDistance, nearPlane, 1e-4, name + ": reversed near distance");
    }

    const fsr::DepthParameters finite = depthParameters(finiteProjection(1.0f, 1000.0f), false);
    expect(!finite.inverted && !finite.infinite, "the finite projection is read as infinite");
    expectNear(finite.nearDistance, 1.0, 1e-4, "finite near distance");
    expectNear(finite.farDistance, 1000.0, 1e-3, "finite far distance");

    const fsr::DepthParameters finiteReversed = depthParameters(reverseDepth(finiteProjection(1.0f, 1000.0f)), true);
    expect(finiteReversed.inverted && !finiteReversed.infinite, "the reversed finite projection is read as infinite");
    expectNear(finiteReversed.nearDistance, 1.0, 1e-4, "reversed finite near distance");
    expectNear(finiteReversed.farDistance, 1000.0, 1e-3, "reversed finite far distance");

    // A right-handed projection looks down negative z, and has the same planes.
    const fsr::DepthParameters rightHanded = fsr::depthParametersFromProjection(1000.0f / -999.0f, 1000.0f / -999.0f, -1.0f, false, 7.0f);
    expect(!rightHanded.infinite, "the right-handed projection is read as infinite");
    expectNear(rightHanded.nearDistance, 1.0, 1e-4, "right-handed near distance");
    expectNear(rightHanded.farDistance, 1000.0, 1e-3, "right-handed far distance");

    // A far plane this distant is beyond what 32-bit depth resolves.
    const fsr::DepthParameters distant = depthParameters(finiteProjection(0.1f, 1.0e7f), false);
    expect(distant.infinite, "a far plane beyond depth precision is read as finite");
    expectNear(distant.nearDistance, 0.1, 1e-3, "distant near distance");

    // An orthographic projection has no perspective divide to read the planes from.
    const fsr::DepthParameters orthographic = fsr::depthParametersFromProjection(0.001f, 0.0f, 0.0f, false, 7.0f);
    expect(orthographic.infinite, "the fallback has a far plane");
    expectNear(orthographic.nearDistance, 7.0, 1e-6, "fallback near distance");

    // Reading a standard projection as inverted puts the near plane behind the camera.
    const fsr::DepthParameters mismatched = fsr::depthParametersFromProjection(1.5f, 10.0f, 1.0f, true, 7.0f);
    expect(mismatched.infinite, "the mismatched projection has a far plane");
    expectNear(mismatched.nearDistance, 7.0, 1e-6, "mismatched near distance");
  }

  // AMD's samples pass the planes swapped for inverted depth, and FLT_MAX for a far plane at infinity.
  void testCameraPlanes() {
    std::cout << "  testCameraPlanes" << std::endl;

    float cameraNear = 0.0f;
    float cameraFar = 0.0f;

    fsr::DepthParameters parameters;
    parameters.nearDistance = 10.0f;
    parameters.farDistance = 5000.0f;

    fsr::cameraPlanes(parameters, cameraNear, cameraFar);
    expect(cameraNear == 10.0f && cameraFar == 5000.0f, "standard finite planes");

    parameters.infinite = true;
    fsr::cameraPlanes(parameters, cameraNear, cameraFar);
    expect(cameraNear == 10.0f && cameraFar == FLT_MAX, "standard infinite planes");

    parameters.inverted = true;
    fsr::cameraPlanes(parameters, cameraNear, cameraFar);
    expect(cameraNear == FLT_MAX && cameraFar == 10.0f, "inverted infinite planes");

    parameters.infinite = false;
    fsr::cameraPlanes(parameters, cameraNear, cameraFar);
    expect(cameraNear == 5000.0f && cameraFar == 10.0f, "inverted finite planes");
  }

  fsr::QueueCandidate queue(uintptr_t handle, uint32_t family, uint32_t flags, bool canPresent) {
    fsr::QueueCandidate candidate;
    candidate.handle = handle;
    candidate.family = family;
    candidate.flags = flags;
    candidate.canPresent = canPresent;
    return candidate;
  }

  void testQueueSelection() {
    std::cout << "  testQueueSelection" << std::endl;

    constexpr uint32_t kGraphicsFamily = fsr::kQueueGraphics | fsr::kQueueCompute | fsr::kQueueTransfer;
    constexpr uint32_t kComputeFamily = fsr::kQueueCompute | fsr::kQueueTransfer;
    constexpr uint32_t kTransferFamily = fsr::kQueueTransfer;

    const fsr::QueueCandidate none;
    const fsr::QueueCandidate game = queue(1, 0, kGraphicsFamily, true);

    // A graphics family with spare queues gives frame generation a present queue that can also compose.
    {
      const fsr::QueueCandidate fsrPresent = queue(2, 0, kGraphicsFamily, true);
      const fsr::QueueCandidate imageAcquire = queue(3, 1, kTransferFamily, false);
      const fsr::QueueCandidate asyncCompute = queue(4, 2, kComputeFamily, true);
      const fsr::QueueCandidate present = queue(5, 2, kComputeFamily, true);

      const fsr::FrameGenQueues queues = fsr::selectFrameGenQueues(game,
        { fsrPresent, present }, { imageAcquire, asyncCompute, present }, { asyncCompute, present });

      expect(queues.valid, "spare graphics queues: no queues selected");
      expect(queues.present.handle == 2, "spare graphics queues: present queue");
      expect(queues.imageAcquire.handle == 3, "spare graphics queues: image acquire queue");
      expect(queues.asyncCompute.handle == 4, "spare graphics queues: async compute queue");
    }

    // A single graphics queue leaves the compute family to present from.
    {
      const fsr::QueueCandidate imageAcquire = queue(3, 2, kTransferFamily, false);
      const fsr::QueueCandidate asyncCompute = queue(4, 1, kComputeFamily, true);
      const fsr::QueueCandidate present = queue(5, 1, kComputeFamily, true);

      const fsr::FrameGenQueues queues = fsr::selectFrameGenQueues(game,
        { none, present }, { imageAcquire, asyncCompute, present }, { asyncCompute, present });

      expect(queues.valid, "single graphics queue: no queues selected");
      expect(queues.present.handle == 5, "single graphics queue: present queue");
      expect(queues.imageAcquire.handle == 3, "single graphics queue: image acquire queue");
      expect(queues.asyncCompute.handle == 4, "single graphics queue: async compute queue");
    }

    // Without a dedicated image acquire queue, the compute queue takes that role and interpolation stays on the game queue.
    {
      const fsr::QueueCandidate asyncCompute = queue(4, 1, kComputeFamily, true);
      const fsr::QueueCandidate present = queue(5, 1, kComputeFamily, true);

      const fsr::FrameGenQueues queues = fsr::selectFrameGenQueues(game,
        { none, present }, { none, asyncCompute, present }, { asyncCompute, present });

      expect(queues.valid, "no image acquire queue: no queues selected");
      expect(queues.present.handle == 5, "no image acquire queue: present queue");
      expect(queues.imageAcquire.handle == 4, "no image acquire queue: image acquire queue");
      expect(queues.asyncCompute.handle == 0, "no image acquire queue: async compute queue");
    }

    // A compute family with a single queue hands out that queue under both names.
    {
      const fsr::QueueCandidate shared = queue(4, 1, kComputeFamily, true);

      const fsr::FrameGenQueues queues = fsr::selectFrameGenQueues(game,
        { none, shared }, { none, shared, shared }, { shared, shared });

      expect(!queues.valid, "one compute queue: the same queue was selected twice");
    }

    // The present queue has to be able to present to the surface.
    {
      const fsr::QueueCandidate imageAcquire = queue(3, 2, kTransferFamily, false);
      const fsr::QueueCandidate asyncCompute = queue(4, 1, kComputeFamily, false);
      const fsr::QueueCandidate present = queue(5, 1, kComputeFamily, false);

      const fsr::FrameGenQueues queues = fsr::selectFrameGenQueues(game,
        { none, present }, { imageAcquire, asyncCompute, present }, { asyncCompute, present });

      expect(!queues.valid, "no present support: a present queue was selected");
    }

    // A device with nothing but its graphics queue cannot run frame generation.
    {
      const fsr::FrameGenQueues queues = fsr::selectFrameGenQueues(game, { none, none }, { none, none, none }, { none, none });

      expect(!queues.valid, "graphics queue only: queues were selected");
    }

    // The game's own queue is never handed out again.
    {
      const fsr::QueueCandidate imageAcquire = queue(3, 2, kTransferFamily, false);

      const fsr::FrameGenQueues queues = fsr::selectFrameGenQueues(game, { game }, { imageAcquire }, { game });

      expect(!queues.valid, "game queue reused: queues were selected");
    }

    // The async compute queue has to be able to run compute work.
    {
      const fsr::QueueCandidate fsrPresent = queue(2, 0, kGraphicsFamily, true);
      const fsr::QueueCandidate imageAcquire = queue(3, 1, kTransferFamily, false);
      const fsr::QueueCandidate transferOnly = queue(6, 1, kTransferFamily, false);

      const fsr::FrameGenQueues queues = fsr::selectFrameGenQueues(game, { fsrPresent }, { imageAcquire }, { transferOnly });

      expect(queues.valid, "transfer-only async candidate: no queues selected");
      expect(queues.asyncCompute.handle == 0, "transfer-only async candidate: async compute queue");
    }
  }

  void testImageAcquireFamilyPreference() {
    std::cout << "  testImageAcquireFamilyPreference" << std::endl;

    constexpr uint32_t kGraphicsFamily = fsr::kQueueGraphics | fsr::kQueueCompute | fsr::kQueueTransfer;
    constexpr uint32_t kComputeFamily = fsr::kQueueCompute | fsr::kQueueTransfer;
    constexpr uint32_t kTransferFamily = fsr::kQueueTransfer;
    constexpr uint32_t kVideoFamily = 0x20;

    // The families are listed the way an NVIDIA driver reports them.
    {
      const std::vector<fsr::QueueFamily> families = {
        { kGraphicsFamily, 16 }, { kTransferFamily, 2 }, { kComputeFamily, 8 }, { kVideoFamily, 1 },
      };

      const std::vector<uint32_t> order = fsr::imageAcquireFamilyPreference(families);

      expect(order == std::vector<uint32_t>({ 1, 2, 0, 3 }), "the families are not ordered transfer, compute, graphics, other");
    }

    // A family without queues is left out.
    {
      const std::vector<fsr::QueueFamily> families = {
        { kGraphicsFamily, 1 }, { kComputeFamily, 2 }, { kTransferFamily, 0 },
      };

      const std::vector<uint32_t> order = fsr::imageAcquireFamilyPreference(families);

      expect(order == std::vector<uint32_t>({ 1, 0 }), "an empty family is offered");
    }
  }

  void testProxyImageCount() {
    std::cout << "  testProxyImageCount" << std::endl;

    expect(fsr::proxyImageCount(2, 2, 8) == 3, "a double-buffered request does not get three images");
    expect(fsr::proxyImageCount(3, 2, 8) == 3, "a triple-buffered request changes");
    expect(fsr::proxyImageCount(8, 2, 16) == 6, "the count exceeds what the frame generation swapchain holds");
    expect(fsr::proxyImageCount(2, 4, 8) == 4, "the count is below the surface's minimum");
    expect(fsr::proxyImageCount(3, 2, 0) == 3, "a surface without an upper limit changes the count");
    expect(fsr::proxyImageCount(5, 2, 4) == 4, "the count exceeds the surface's maximum");
  }

}

int main() {
  try {
    std::cout << std::endl << "Begin FSR helper tests" << std::endl;
    testPresets();
    testMipBias();
    testFrameTime();
    testDepthParameters();
    testCameraPlanes();
    testQueueSelection();
    testImageAcquireFamilyPreference();
    testProxyImageCount();
    std::cout << "All FSR helper tests passed" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
