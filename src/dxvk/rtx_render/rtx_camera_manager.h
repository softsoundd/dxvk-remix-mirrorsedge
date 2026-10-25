/*
* Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "rtx_camera.h"
#include "rtx_common_object.h"
#include "rtx_matrix_helpers.h"
#include "../../util/util_fast_cache.h"

namespace dxvk {
  class DxvkDevice;

  class CameraManager : public CommonDeviceObject {
  public:
    explicit CameraManager(DxvkDevice* device);
    ~CameraManager() override = default;

    CameraManager(const CameraManager& other) = delete;
    CameraManager(CameraManager&& other) noexcept = delete;
    CameraManager& operator=(const CameraManager& other) = delete;
    CameraManager& operator=(CameraManager&& other) noexcept = delete;

    const RtCamera& getCamera(CameraType::Enum cameraType) const { return accessCamera(*this, cameraType); }
    RtCamera& getCamera(CameraType::Enum cameraType) { return accessCamera(*this, cameraType); }

    const RtCamera& getMainCamera() const { return getCamera(CameraType::Main); }
    RtCamera& getMainCamera() { return getCamera(CameraType::Main); }

    bool isCameraValid(CameraType::Enum cameraType) const;

    void onFrameEnd();

    void processExternalCamera(CameraType::Enum type, const Matrix4& worldToView, const Matrix4& viewToProjection);

  private:
    template<
      typename T,
      std::enable_if_t<std::is_same_v<T, CameraManager> || std::is_same_v<T, const CameraManager>, bool> = true>
    static auto& accessCamera(T& cameraManager, CameraType::Enum cameraType) {
      assert(cameraType < CameraType::Count);
      // Unknown aliases the Main camera object; it is never updated directly.
      if (cameraType == CameraType::Unknown) {
        return cameraManager.m_cameras[CameraType::Main];
      }
      return cameraManager.m_cameras[cameraType];
    }

  private:
    std::array<RtCamera, CameraType::Count> m_cameras;
    fast_unordered_cache<DecomposeProjectionParams> m_decompositionCache;

    DecomposeProjectionParams getOrDecomposeProjection(const Matrix4& viewToProjection);
  };
}  // namespace dxvk
