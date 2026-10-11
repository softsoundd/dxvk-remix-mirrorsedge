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

#include "rtx_resources.h"
#include "rtx_common_object.h"
#include "rtx_option.h"

namespace dxvk {

  class RtxContext;

  enum class SunVisibilitySource : int {
    RayTraced = 0,
    Image,
  };

  /**
   * \brief The Physical Atmosphere sun as the camera sees it this frame.
   *
   * Projects the sun onto the screen and measures how much of its disc the camera sees, into a 1x1 texture on the GPU
   * that the sun's glare and the lens flare read.
   */
  class RtxSunProbe : public CommonDeviceObject {
  public:
    struct State {
      // The sun is above the horizon and in front of the camera, and its disc is drawn.
      bool active = false;
      // Some of the disc is on the screen this frame.
      bool onScreen = false;
      // The visibility texture holds a measurement of the current sun.
      bool measured = false;
      // Disc centre in output pixels, with a top left origin.
      Vector2 pixel = Vector2(0.0f);
      // Disc centre in NDC with x scaled by the aspect ratio, y up.
      Vector2 ndcAspect = Vector2(0.0f);
      // The projected disc's semi-axis across the view. Along it, from the screen's centre, it is 1 / cosOffAxis
      // longer.
      float discRadiusPixels = 0.0f;
      // Cosine of the angle between the sun and the view direction.
      float cosOffAxis = 1.0f;
      float angularRadius = 0.0f;
      float tanHalfFovY = 0.0f;
      float aspectRatio = 1.0f;
      // The share of the disc's area on the screen.
      float onScreenFraction = 0.0f;
      // Sun illuminance at the camera through the atmosphere, unclamped.
      Vector3 illuminance = Vector3(0.0f);
      // The part of the illuminance the rendered disc carries under its radiance clamp, all of it on the screen.
      Vector3 renderedIlluminance = Vector3(0.0f);
      Vector3 limbDarkeningExponent = Vector3(0.0f);
    };

    explicit RtxSunProbe(DxvkDevice* device);

    RTX_OPTION("rtx.sunVisibility", SunVisibilitySource, source, SunVisibilitySource::RayTraced,
               "How the sun's visibility is measured for its glare and the lens flare: 0: Ray Traced (256 rays from the camera\n"
               "to the sun's disc through the scene, the cloud layer and the volumetric fog, on or off the screen), 1: Image\n"
               "(the disc's pixels in the rendered image while it is on the screen, for debugging).");

    // Projects the sun and, when measure is set, measures its visibility. Must run before any pass that changes the
    // disc's pixels, such as bloom, and after the scene's acceleration structure is built.
    void dispatch(RtxContext& ctx, const Resources::RaytracingOutput& rtOutput, bool measure);

    const State& getState() const { return m_state; }

    // The illuminance the rendered disc lacks before its visibility: what its radiance clamp holds back, and the part
    // of the clamped disc off the screen.
    Vector3 getRestoredIlluminance() const;

    // 1x1 RGBA32F texture whose RGB is the visible fraction of the disc.
    // Returns null until the probe first measures.
    Rc<DxvkImageView> getVisibilityView() const { return m_visibility.view; }

    void showImguiSettings();

  private:
    void updateState(RtxContext& ctx, const Resources::RaytracingOutput& rtOutput);
    void dispatchRayTraced(RtxContext& ctx, const Resources::RaytracingOutput& rtOutput);
    void dispatchImage(RtxContext& ctx, const Resources::RaytracingOutput& rtOutput);

    State m_state;
    Vector3 m_sunDirection = Vector3(0.0f);
    // Clear of the disc's radial stretch off the axis, which exceeds discRadiusPixels.
    float m_annulusInnerRadiusPixels = 0.0f;
    Resources::Resource m_visibility;
  };

}
