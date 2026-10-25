/*
* Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
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

#include "rtx_constants.h"
#include "rtx_option.h"

#include "../util/util_vector.h"
#include "../util/util_matrix.h"

namespace dxvk
{
  namespace CameraType {
    enum Enum : uint32_t {
      Main = 0,        // Main camera
      ViewModel,       // Camera for view model rendering
      Portal0,         // Camera associated with rendering portal 0
      Portal1,         // Camera associated with rendering portal 1
      Sky,             // Some renderers have separate world / sky cameras
      RenderToTexture, // Camera used to replace a render target texture that is being raytraced.
      Unknown,         // Unset camera state, used mainly for state tracking. Its camera object is aliased 
                       // with the Main camera object, so on access it retrieves the Main camera

      Count
    };
  }

  // Returns a 2D <-0.5, 0.5> Halton jitter sample 
  Vector2 calculateHaltonJitter(uint32_t currentFrame, uint32_t jitterSequenceLength);

  class RtCamera
  {
    RTX_OPTION("rtx.camera", bool, correctProjectionYFlip, false,
               "Some games (typically Unity titles) use a projection matrix with a negative Y scale, causing the "
               "ray-traced scene to render upside down. Enable this to automatically detect and correct the Y-flip "
               "in the projection matrix at runtime.");

    uint32_t m_renderResolution[2] = { 0, 0 };
    uint32_t m_finalResolution[2] = { 0, 0 };
    float m_jitter[2] = { 0, 0 };
    bool m_firstUpdate = true;
    CameraType::Enum m_type;

    enum MatrixType {
      WorldToView = 0,
      PreviousWorldToView,
      ViewToWorld,
      PreviousViewToWorld,
      PreviousPreviousViewToWorld,

      ViewToProjection,
      PreviousViewToProjection,
      ProjectionToView,
      PreviousProjectionToView,

      ViewToProjectionJittered,
      PreviousViewToProjectionJittered,
      ProjectionToViewJittered,
      PreviousProjectionToViewJittered,

      Count
    };

    // Note: All camera matricies stored as double precision. While this does not do much for some matricies (which were provided
    // by the application in floating point precision), it does help for preserving matrix stability on those which have been inverted,
    // as well as in code using these matrices which may do further inversions or combination operations. If such precision is not needed
    // the matrices retreived from the various getter functions can be casted to float matrices for a minor upfront performance cost.
    Matrix4d m_matCache[MatrixType::Count] = {};

    // Projection parameters passed to the last update()
    float m_fov = 0.0f;
    float m_aspectRatio = 0.0f;
    float m_nearPlane = 0.0f;
    float m_farPlane = 0.0f;
    bool m_isLHS = false;

    uint32_t m_lastViewHistoryInvalidationFrameId = kInvalidFrameIndex;

    // Note: Start the camera off as invalid until it is set properly.
    uint32_t m_frameLastTouched = kInvalidFrameIndex;

  public:
    RtCamera() = default;
    ~RtCamera() = default;
    // Note: Copying functionality required for DLFG implementation (to store a copy of a camera used to render a frame for DLFG evaluation later).
    // Avoid copies otherwise as this is a fairly large structure. If needed only the data required by DLFG could be copied and this copy functionality
    // could be removed for slightly better performance.
    RtCamera(const RtCamera& other) = default;
    RtCamera& operator=(const RtCamera& other) = default;

    // Gets the Y axis (vertical) FoV of the camera's projection matrix in radians. Note this value will be positive always (even with strange camera types).
    float getFov() const { return m_fov; }
    float getAspectRatio() const { return m_aspectRatio; }

    const Matrix4d& getWorldToView() const { return m_matCache[MatrixType::WorldToView]; }
    const Matrix4d& getPreviousWorldToView() const { return m_matCache[MatrixType::PreviousWorldToView]; }
    const Matrix4d& getViewToWorld() const { return m_matCache[MatrixType::ViewToWorld]; }
    const Matrix4d& getPreviousViewToWorld() const { return m_matCache[MatrixType::PreviousViewToWorld]; }
    const Matrix4d& getPreviousPreviousViewToWorld() const { return m_matCache[MatrixType::PreviousPreviousViewToWorld]; }

    const Matrix4d& getViewToProjection() const { return m_matCache[MatrixType::ViewToProjection]; }
    const Matrix4d& getPreviousViewToProjection() const { return m_matCache[MatrixType::PreviousViewToProjection]; }
    const Matrix4d& getProjectionToView() const { return m_matCache[MatrixType::ProjectionToView]; }
    const Matrix4d& getPreviousProjectionToView() const { return m_matCache[MatrixType::PreviousProjectionToView]; }

    bool isValid(const uint32_t frameIdx) const { return m_frameLastTouched == frameIdx; }
    uint32_t getLastUpdateFrame() const { return m_frameLastTouched; }

    Vector3 getPosition() const;
    Vector3 getDirection() const;
    Vector3 getUp() const;
    Vector3 getRight() const;

    // Note: getNearPlane() / getFarPlane() return values
    // corresponding to the viewToProjection matrix passed into update(..),
    // and NOT to the viewToProjection in the m_matCache, because of 'enableNearPlaneOverride' option.
    // If need actual near/far planes corresponding to the current matrix, use calculateNearFarPlanes().
    float getNearPlane() const { return m_nearPlane; }
    float getFarPlane() const { return m_farPlane; }
    std::pair<float, float> calculateNearFarPlanes() const;

    void setCameraType(CameraType::Enum type) {
      m_type = type;
    }

    void invalidateViewHistory(uint32_t frameIdx);
    bool isViewHistoryInvalidated(uint32_t frameIdx) const;
    bool isCameraCut() const;

    const Matrix4d& getViewToProjectionJittered() const { return m_matCache[MatrixType::ViewToProjectionJittered]; }
    const Matrix4d& getPreviousViewToProjectionJittered() const { return m_matCache[MatrixType::PreviousViewToProjectionJittered]; }
    const Matrix4d& getProjectionToViewJittered() const { return m_matCache[MatrixType::ProjectionToViewJittered]; }
    const Matrix4d& getPreviousProjectionToViewJittered() const { return m_matCache[MatrixType::PreviousProjectionToViewJittered]; }

    void setResolution(const uint32_t renderResolution[2], const uint32_t finalResolution[2]);
    bool update(
      uint32_t frameIdx, const Matrix4& newWorldToView, const Matrix4& newViewToProjection,
      float fov, float aspectRatio, float nearPlane, float farPlane, bool isLHS
    );
    void getJittering(float jitter[2]) const;

    // Overrides the jitter reported for the current frame. Used by the NGX passthrough mode,
    // where the sub-pixel jitter is applied on the D3D9 side (viewport offset) rather than
    // through the ray tracing projection matrix, so the camera must report exactly the value
    // the frame was rasterized with (consumed by DLSS and DLFG).
    void setExternalJitter(const float jitter[2]) {
      m_jitter[0] = jitter[0];
      m_jitter[1] = jitter[1];
    }

    bool isLHS() const { return m_isLHS; }

  private:
    Vector2 calcPixelJitter(uint32_t jitterFrameIdx) const;
    Vector2 calcClipSpaceJitter(const Vector2& pixelJitter, float ratioX, float ratioY) const;
    void applyAndGetJitter(Matrix4d& inoutProjection, float (&outPixelJitter)[2], uint32_t jitterFrameIdx) const;
    Matrix4d overrideNearPlane(const Matrix4d& modifiedViewToProj);
  };
}
