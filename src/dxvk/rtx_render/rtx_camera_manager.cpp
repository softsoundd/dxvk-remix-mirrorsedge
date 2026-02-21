/*
* Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_camera_manager.h"

#include "dxvk_device.h"

namespace {
  constexpr float kFovToleranceRadians = 0.001f;
}

namespace dxvk {

  CameraManager::CameraManager(DxvkDevice* device) : CommonDeviceObject(device) {
    for (int i = 0; i < CameraType::Count; i++) {
      m_cameras[i].setCameraType(CameraType::Enum(i));
    }
  }

  bool CameraManager::isCameraValid(CameraType::Enum cameraType) const {
    assert(cameraType < CameraType::Enum::Count);
    const uint32_t frameId = m_device->getCurrentFrameId();
    const RtCamera& camera = accessCamera(*this, cameraType);
    if (camera.isValid(frameId))
      return true;

    // keep main camera usable for a rejection frame so raytracing doesn't disengage
    // while we wait for a suspicious jump candidate to prove persistence todo: fix this later
    if (cameraType == CameraType::Main &&
        guardMainCameraFromOutliers() &&
        m_pendingMainJumpCandidate.valid &&
        m_pendingMainJumpCandidate.frameId == frameId &&
        frameId > 0 &&
        camera.isValid(frameId - 1)) {
      return true;
    }

    return false;
  }

  void CameraManager::onFrameEnd() {
    m_lastSetCameraType = CameraType::Unknown;
    m_decompositionCache.clear();
  }

  CameraType::Enum CameraManager::processCameraData(const DrawCallState& input) {
    // If theres no real camera data here - bail
    if (isIdentityExact(input.getTransformData().viewToProjection)) {
      return input.testCategoryFlags(InstanceCategories::Sky) ? CameraType::Sky : CameraType::Unknown;
    }

    switch (RtxOptions::fusedWorldViewMode()) {
    case FusedWorldViewMode::None:
      if (input.getTransformData().objectToView == input.getTransformData().objectToWorld && !isIdentityExact(input.getTransformData().objectToView)) {
        return input.testCategoryFlags(InstanceCategories::Sky) ? CameraType::Sky : CameraType::Unknown;
      }
      break;
    case FusedWorldViewMode::View:
      if (Logger::logLevel() >= LogLevel::Warn) {
        // Check if World is identity
        ONCE_IF_FALSE(isIdentityExact(input.getTransformData().objectToWorld),
                      Logger::warn("[RTX-Compatibility] Fused world-view tranform set to View but World transform is not identity!"));
      }
      break;
    case FusedWorldViewMode::World:
      if (Logger::logLevel() >= LogLevel::Warn) {
        // Check if View is identity
        ONCE_IF_FALSE(isIdentityExact(input.getTransformData().objectToView),
                      Logger::warn("[RTX-Compatibility] Fused world-view tranform set to World but View transform is not identity!"));
      }
      break;
    }

    // Get camera params
    DecomposeProjectionParams decomposeProjectionParams = getOrDecomposeProjection(input.getTransformData().viewToProjection);

    // Filter invalid cameras, extreme shearing
    static auto isFovValid = [](float fovA) {
      return fovA >= kFovToleranceRadians;
    };
    static auto areFovsClose = [](float fovA, const RtCamera& cameraB) {
      return std::abs(fovA - cameraB.getFov()) < kFovToleranceRadians;
    };

    if (std::abs(decomposeProjectionParams.shearX) > 0.01f || !isFovValid(decomposeProjectionParams.fov)) {
      ONCE(Logger::warn("[RTX] CameraManager: rejected an invalid camera"));
      return input.getCategoryFlags().test(InstanceCategories::Sky) ? CameraType::Sky : CameraType::Unknown;
    }


    auto isViewModel = [this](float fov, float maxZ, uint32_t frameId) {
      if (RtxOptions::ViewModel::enable()) {
        // Note: max Z check is the top-priority
        if (maxZ <= RtxOptions::ViewModel::maxZThreshold()) {
          return true;
        }
        if (getCamera(CameraType::Main).isValid(frameId)) {
          // FOV is different from Main camera => assume that it's a ViewModel one
          if (!areFovsClose(fov, getCamera(CameraType::Main))) {
            return true;
          }
        }
      }
      return false;
    };

    const uint32_t frameId = m_device->getCurrentFrameId();

    auto cameraType = CameraType::Main;
    if (input.isDrawingToRaytracedRenderTarget) {
      cameraType = CameraType::RenderToTexture;
    } else if (input.testCategoryFlags(InstanceCategories::Sky)) {
      cameraType = CameraType::Sky;
    } else if (isViewModel(decomposeProjectionParams.fov, input.maxZ, frameId)) {
      cameraType = CameraType::ViewModel;
    }
    
    // Check fov consistency across frames
    if (frameId > 0) {
      if (getCamera(cameraType).isValid(frameId - 1) && !areFovsClose(decomposeProjectionParams.fov, getCamera(cameraType))) {
        ONCE(Logger::warn("[RTX] CameraManager: FOV of a camera changed between frames"));
      }
    }

    auto& camera = getCamera(cameraType);
    auto cameraSequence = RtCameraSequence::getInstance();
    bool shouldUpdateMainCamera = cameraType == CameraType::Main && camera.getLastUpdateFrame() != frameId;
    bool isPlaying = RtCameraSequence::mode() == RtCameraSequence::Mode::Playback;
    bool isBrowsing = RtCameraSequence::mode() == RtCameraSequence::Mode::Browse;
    bool isCameraCut = false;
    Matrix4 worldToView = input.getTransformData().worldToView;
    Matrix4 viewToProjection = input.getTransformData().viewToProjection;

    if (guardMainCameraFromOutliers() && shouldUpdateMainCamera && frameId > 0 && getCamera(CameraType::Main).isValid(frameId - 1)) {
      const RtCamera& prevMain = getCamera(CameraType::Main);

      const float prevAspect = prevMain.getAspectRatio();
      const float prevFov = prevMain.getFov();

      const float aspectRelDiff =
        prevAspect > 1e-6f
          ? std::abs(decomposeProjectionParams.aspectRatio - prevAspect) / prevAspect
          : 0.0f;

      Vector3 candDir(0.0f);
      Vector3 candPos(0.0f);
      {
        const Matrix4 viewToWorld = inverseAffine(worldToView);
        candDir = Vector3 { viewToWorld[2].xyz() };
        candPos = Vector3 { viewToWorld[3].xyz() };
        if (!decomposeProjectionParams.isLHS)
          candDir = -candDir;

        const float len = length(candDir);
        if (len > 0.0f)
          candDir /= len;
      }

      Vector3 prevDir = prevMain.getDirection(false);
      const float prevLen = length(prevDir);
      if (prevLen > 0.0f)
        prevDir /= prevLen;

      const float dirDot = dot(candDir, prevDir);
      const float fovDiff = std::abs(decomposeProjectionParams.fov - prevFov);

      // reject if aspect ratio is way off or if both fov and direction are strong outliers
      constexpr float kAspectRelDiffThreshold = 0.10f;
      constexpr float kFovDiffThreshold = 0.35f;
      constexpr float kDirDotThreshold = 0.20f;

      const bool rejectedByAspectOrDirection =
        aspectRelDiff > kAspectRelDiffThreshold ||
        (fovDiff > kFovDiffThreshold && dirDot < kDirDotThreshold);

      if (rejectedByAspectOrDirection) {
        m_pendingMainJumpCandidate.valid = false;
        ONCE(Logger::warn("[RTX-Compatibility] CameraManager: rejected outlier Main camera candidate (likely shadow/utility camera)."));
        return CameraType::Unknown;
      }

      // reject the bad camera frame spikes but accept persistent jumps
      // on the next frame so real teleports/loads still converge to the new camera
      {
        const float posJumpDistSqr = lengthSqr(candPos - prevMain.getPosition(false));
        const float suspiciousJumpThresholdSqr = RtxOptions::getUniqueObjectDistanceSqr() * 4.0f;
        const bool isSuspiciousJump = posJumpDistSqr > suspiciousJumpThresholdSqr;

        if (isSuspiciousJump) {
          bool isPersistentJump = false;
          if (m_pendingMainJumpCandidate.valid &&
              (m_pendingMainJumpCandidate.frameId == frameId ||
               m_pendingMainJumpCandidate.frameId + 1 == frameId)) {
            const float repeatPosDistSqr = lengthSqr(candPos - m_pendingMainJumpCandidate.position);
            const float repeatDirDot = dot(candDir, m_pendingMainJumpCandidate.direction);
            const float repeatFovDiff = std::abs(decomposeProjectionParams.fov - m_pendingMainJumpCandidate.fov);
            const float repeatAspectRelDiff =
              m_pendingMainJumpCandidate.aspectRatio > 1e-6f
                ? std::abs(decomposeProjectionParams.aspectRatio - m_pendingMainJumpCandidate.aspectRatio) / m_pendingMainJumpCandidate.aspectRatio
                : 0.0f;

            isPersistentJump =
              repeatPosDistSqr <= RtxOptions::getUniqueObjectDistanceSqr() &&
              repeatDirDot > 0.85f &&
              repeatFovDiff < 0.10f &&
              repeatAspectRelDiff < 0.05f;
          }

          if (!isPersistentJump) {
            m_pendingMainJumpCandidate.valid = true;
            m_pendingMainJumpCandidate.frameId = frameId;
            m_pendingMainJumpCandidate.position = candPos;
            m_pendingMainJumpCandidate.direction = candDir;
            m_pendingMainJumpCandidate.fov = decomposeProjectionParams.fov;
            m_pendingMainJumpCandidate.aspectRatio = decomposeProjectionParams.aspectRatio;
            ONCE(Logger::warn("[RTX-Compatibility] CameraManager: rejected provisional large-jump Main camera candidate."));
            return CameraType::Unknown;
          }
        }

        m_pendingMainJumpCandidate.valid = false;
      }
    }

    if (isPlaying || isBrowsing) {
      if (shouldUpdateMainCamera) {
        RtCamera::RtCameraSetting setting;
        cameraSequence->getRecord(cameraSequence->currentFrame(), setting);
        isCameraCut = camera.updateFromSetting(frameId, setting, 0);

        if (isPlaying) {
          cameraSequence->goToNextFrame();
        }
      }
    } else {
      isCameraCut = camera.update(
        frameId,
        worldToView,
        viewToProjection,
        decomposeProjectionParams.fov,
        decomposeProjectionParams.aspectRatio,
        decomposeProjectionParams.nearPlane,
        decomposeProjectionParams.farPlane,
        decomposeProjectionParams.isLHS
      );
    }


    if (shouldUpdateMainCamera && RtCameraSequence::mode() == RtCameraSequence::Mode::Record) {
      auto& setting = camera.getSetting();
      cameraSequence->addRecord(setting);
    }

    // Register camera cut when there are significant interruptions to the view (like changing level, or opening a menu)
    if (isCameraCut && cameraType == CameraType::Main) {
      m_lastCameraCutFrameId = m_device->getCurrentFrameId();
    }
    m_lastSetCameraType = cameraType;

    return cameraType;
  }

  bool CameraManager::isCameraCutThisFrame() const {
    return m_lastCameraCutFrameId == m_device->getCurrentFrameId();
  }

  void CameraManager::processExternalCamera(CameraType::Enum type,
                                            const Matrix4& worldToView,
                                            const Matrix4& viewToProjection) {
    DecomposeProjectionParams decomposeProjectionParams = getOrDecomposeProjection(viewToProjection);

    getCamera(type).update(
      m_device->getCurrentFrameId(),
      worldToView,
      viewToProjection,
      decomposeProjectionParams.fov,
      decomposeProjectionParams.aspectRatio,
      decomposeProjectionParams.nearPlane,
      decomposeProjectionParams.farPlane,
      decomposeProjectionParams.isLHS);
  }

    DecomposeProjectionParams CameraManager::getOrDecomposeProjection(const Matrix4& viewToProjection) {
      XXH64_hash_t projectionHash = XXH64(&viewToProjection, sizeof(viewToProjection), 0);
      auto iter = m_decompositionCache.find(projectionHash);
      if (iter != m_decompositionCache.end()) {
        return iter->second;
      }

      DecomposeProjectionParams decomposeProjectionParams;
      decomposeProjection(viewToProjection, decomposeProjectionParams);
      m_decompositionCache.emplace(projectionHash, decomposeProjectionParams);
      return decomposeProjectionParams;
    }
}  // namespace dxvk
