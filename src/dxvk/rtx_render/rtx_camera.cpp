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
#include "../../util/util_math.h"
#include "../../util/util_vector.h"
#include "../../util/util_matrix.h"
#include "rtx_camera.h"
#include "rtx_options.h"
#include "rtx_matrix_helpers.h"
#include "rtx_xess.h"

namespace dxvk
{
  Vector3 RtCamera::getPosition() const {
    return Vector3{ getViewToWorld()[3].xyz() };
  }

  Vector3 RtCamera::getDirection() const {
    // Note: To get a "forward" direction in world space, the basis vector mapping world to view space's Z axis
    // is used, but unlike the up/right vectors we must consider which direction the projection matrix is treating as
    // forwards. With left handed projection matrices this is the +Z axis, but with right handed matrices this is -Z.
    if (m_isLHS) {
      return Vector3{ getViewToWorld()[2].xyz() };
    } else {
      return -Vector3{ getViewToWorld()[2].xyz() };
    }
  }

  Vector3 RtCamera::getUp() const {
    return Vector3{ getViewToWorld()[1].xyz() };
  }

  Vector3 RtCamera::getRight() const {
    return Vector3{ getViewToWorld()[0].xyz() };
  }

  std::pair<float, float> RtCamera::calculateNearFarPlanes() const {
    // Note: Converted to floats to interface with MathLib. Ideally this should be a double still.
    Matrix4 floatViewToProj{ getViewToProjection() };

    // Check size since struct padding can impact this memcpy
    static_assert(sizeof(float4x4) == sizeof(floatViewToProj));

    uint32_t flags;
    float    cameraParams[PROJ_NUM];
    DecomposeProjection(
      NDC_D3D,
      NDC_D3D,
      *reinterpret_cast<float4x4*>(&floatViewToProj),
      &flags,
      cameraParams,
      nullptr,
      nullptr,
      nullptr,
      nullptr);

    return { cameraParams[PROJ_ZNEAR], cameraParams[PROJ_ZFAR] };
  }

  void RtCamera::invalidateViewHistory(uint32_t frameIdx) {
    m_lastViewHistoryInvalidationFrameId = frameIdx;
  }

  bool RtCamera::isViewHistoryInvalidated(uint32_t frameIdx) const {
    return m_lastViewHistoryInvalidationFrameId == frameIdx
        || isCameraCut();
  }

  bool RtCamera::isCameraCut() const {
    return lengthSqr(getViewToWorld()[3] - getPreviousViewToWorld()[3]) > RtxOptions::getUniqueObjectDistanceSqr();
  }

  Vector2 calculateHaltonJitter(uint32_t currentFrame, uint32_t jitterSequenceLength) {
    // Halton jitter
    Vector2 result(0.0f, 0.0f);

    uint32_t frameIndex = currentFrame % jitterSequenceLength;

    constexpr uint32_t baseX = 2;
    uint32_t Index = frameIndex + 1;
    float invBase = 1.0f / baseX;
    float fraction = invBase;
    while (Index > 0) {
      result.x += (Index % baseX) * fraction;
      Index /= baseX;
      fraction *= invBase;
    }

    constexpr uint32_t baseY = 3;
    Index = frameIndex + 1;
    invBase = 1.0f / baseY;
    fraction = invBase;
    while (Index > 0) {
      result.y += (Index % baseY) * fraction;
      Index /= baseY;
      fraction *= invBase;
    }

    result.x -= 0.5f;
    result.y -= 0.5f;
    return result;
  }

  void RtCamera::setResolution(const uint32_t renderResolution[2], const uint32_t finalResolution[2]) {
    m_renderResolution[0] = renderResolution[0];
    m_renderResolution[1] = renderResolution[1];
    m_finalResolution[0] = finalResolution[0];
    m_finalResolution[1] = finalResolution[1];
  }

  Matrix4d RtCamera::overrideNearPlane(const Matrix4d& modifiedViewToProj) {
    // Note: Converted to floats to interface with MathLib. Ideally this should be a double still.
    Matrix4 floatModifiedViewToProj { modifiedViewToProj };

    // Check size since struct padding can impact this memcpy
    static_assert(sizeof(float4x4) == sizeof(floatModifiedViewToProj));

    uint32_t flags;
    float cameraParams[PROJ_NUM];
    DecomposeProjection(NDC_D3D, NDC_D3D, *reinterpret_cast<float4x4*>(&floatModifiedViewToProj), &flags, cameraParams, nullptr, nullptr, nullptr, nullptr);

    // Prevent user controls exceeding the near plane distance from original projection
    const float minNearPlane = std::min(RtxOptions::nearPlaneOverride(), cameraParams[PROJ_ZNEAR]);

    float4x4 newProjection;
    newProjection.SetupByAngles(cameraParams[PROJ_ANGLEMINX], cameraParams[PROJ_ANGLEMAXX], cameraParams[PROJ_ANGLEMINY], cameraParams[PROJ_ANGLEMAXY], minNearPlane, cameraParams[PROJ_ZFAR], flags);
    memcpy(&floatModifiedViewToProj, &newProjection, sizeof(float4x4));

    return Matrix4d { floatModifiedViewToProj };
  }

  bool RtCamera::update(
    uint32_t frameIdx, const Matrix4& newWorldToView, const Matrix4& newViewToProjection,
    float fov, float aspectRatio, float nearPlane, float farPlane, bool isLHS) {
    if (m_frameLastTouched == frameIdx) {
      return false;
    }

    m_fov = fov;
    m_aspectRatio = aspectRatio;
    m_nearPlane = nearPlane;
    m_farPlane = farPlane;
    m_isLHS = isLHS;

    // Setup World/View Matrix Data

    m_matCache[MatrixType::PreviousPreviousViewToWorld] = m_matCache[MatrixType::PreviousViewToWorld];
    m_matCache[MatrixType::PreviousWorldToView] = m_matCache[MatrixType::WorldToView];
    m_matCache[MatrixType::PreviousViewToWorld] = m_matCache[MatrixType::ViewToWorld];
    m_matCache[MatrixType::WorldToView] = Matrix4d{ newWorldToView };
    m_matCache[MatrixType::ViewToWorld] = inverse(m_matCache[MatrixType::WorldToView]);

    // Setup View/Projection Matrix Data

    m_matCache[MatrixType::PreviousViewToProjection] = m_matCache[MatrixType::ViewToProjection];
    m_matCache[MatrixType::PreviousProjectionToView] = m_matCache[MatrixType::ProjectionToView];

    auto modifiedViewToProj = Matrix4d{ newViewToProjection };

    // Correct for games that use a negative Y scale in the projection matrix
    // (e.g., certain Unity titles) which causes the ray-traced scene to render upside-down.
    if (correctProjectionYFlip()) {
      modifiedViewToProj[1][1] = -modifiedViewToProj[1][1];
    }

    if (RtxOptions::enableNearPlaneOverride()) {
      modifiedViewToProj = overrideNearPlane(modifiedViewToProj);
    }
    
    m_matCache[MatrixType::ViewToProjection] = modifiedViewToProj;
    m_matCache[MatrixType::ProjectionToView] = inverse(modifiedViewToProj);

    // Apply jittering

    auto newViewToProjectionJittered = modifiedViewToProj;
    applyAndGetJitter(
      newViewToProjectionJittered,
      m_jitter,
      frameIdx);

    m_matCache[MatrixType::PreviousViewToProjectionJittered] = m_matCache[MatrixType::ViewToProjectionJittered];
    m_matCache[MatrixType::PreviousProjectionToViewJittered] = m_matCache[MatrixType::ProjectionToViewJittered];
    m_matCache[MatrixType::ViewToProjectionJittered] = newViewToProjectionJittered;
    m_matCache[MatrixType::ProjectionToViewJittered] = inverse(newViewToProjectionJittered);

    m_frameLastTouched = frameIdx;

    // For our first update, we should init both previous and current to the same value
    if (m_firstUpdate) {
      m_matCache[MatrixType::PreviousWorldToView] = m_matCache[MatrixType::WorldToView];
      m_matCache[MatrixType::PreviousViewToWorld] = m_matCache[MatrixType::ViewToWorld];
      m_matCache[MatrixType::PreviousPreviousViewToWorld] = m_matCache[MatrixType::ViewToWorld];

      m_matCache[MatrixType::PreviousViewToProjection] = m_matCache[MatrixType::ViewToProjection];
      m_matCache[MatrixType::PreviousProjectionToView] = m_matCache[MatrixType::ProjectionToView];

      m_matCache[MatrixType::PreviousViewToProjectionJittered] = m_matCache[MatrixType::ViewToProjectionJittered];
      m_matCache[MatrixType::PreviousProjectionToViewJittered] = m_matCache[MatrixType::ProjectionToViewJittered];

      // Never do this again
      m_firstUpdate = false;
    }

    return isCameraCut();
  }

  void RtCamera::getJittering(float jitter[2]) const {
    jitter[0] = m_jitter[0];
    jitter[1] = m_jitter[1];
  }
  
  Vector2 RtCamera::calcPixelJitter(uint32_t jitterFrameIdx) const {
    // Only apply jittering when DLSS/XeSS/TAA is enabled, or if forced by settings
    if (!RtxOptions::isDLSSEnabled() &&
        !RtxOptions::isXeSSEnabled() &&
        !RtxOptions::isTAAEnabled() &&
        !RtxOptions::forceCameraJitter()) {
      return Vector2{ 0, 0 };
    }

    uint32_t jitterSequenceLength = RtxOptions::cameraJitterSequenceLength();

    if (RtxOptions::isXeSSEnabled() && DxvkXeSS::XessOptions::useRecommendedJitterSequenceLength()) {
      float upscaleFactor = static_cast<float>(m_finalResolution[1]) / m_renderResolution[1];

      // XeSS 2.1 formula: ceil(upscale_factor^2 * 8)
      // The 8.0 multiplier ensures sufficient temporal samples for higher upscaling factors
      uint32_t xessLength = static_cast<uint32_t>(std::ceil(upscaleFactor * upscaleFactor * 8.0f));
      
      // Apply minimum jitter sequence length
      uint32_t minLength = DxvkXeSS::XessOptions::minJitterSequenceLength();
      xessLength = std::max(xessLength, minLength);
      
      jitterSequenceLength = xessLength;
    }
    
    return calculateHaltonJitter(jitterFrameIdx, jitterSequenceLength);
  }

  Vector2 RtCamera::calcClipSpaceJitter(
    const Vector2& pixelJitter, 
    float ratioX,
    float ratioY) const {
    if (m_renderResolution[0] == 0 || m_renderResolution[1] == 0) {
      return Vector2{ 0, 0 };
    }
    return Vector2{
      pixelJitter[0] / static_cast<float>(m_renderResolution[0]) * ratioX * 2.f,
      pixelJitter[1] / static_cast<float>(m_renderResolution[1]) * ratioY * 2.f,
    };
  }

  void RtCamera::applyAndGetJitter(
    Matrix4d& inoutProjection, 
    float (&outPixelJitter)[2], 
    uint32_t jitterFrameIdx) const {
    Vector2 pixelJitter = calcPixelJitter(jitterFrameIdx);
    float ratioX = Sign(inoutProjection[2][3]);
    float ratioY = -Sign(inoutProjection[2][3]);
    Vector2 clipSpaceJitter = calcClipSpaceJitter(pixelJitter, ratioX, ratioY);
    {
      outPixelJitter[0] = pixelJitter[0];
      outPixelJitter[1] = pixelJitter[1];
    }
    if (std::abs(clipSpaceJitter[0]) < std::numeric_limits<float>::min() &&
        std::abs(clipSpaceJitter[1]) < std::numeric_limits<float>::min()) {
      return;
    }
    inoutProjection[2][0] += clipSpaceJitter[0];
    inoutProjection[2][1] += clipSpaceJitter[1];
  }
}
