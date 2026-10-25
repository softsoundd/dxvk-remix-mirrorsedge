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

#include "../dxvk_context.h"
#include "rtx_resources.h"
#include "rtx_camera_manager.h"
#include "rtx_ngx_passthrough.h"

#include <cstdint>
#include <chrono>
#include <memory>
#include <array>
#include "rtx_options.h"

namespace dxvk {
  class DxvkContext;
  class SceneManager;

  struct DrawParameters {
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    uint32_t instanceCount = 0;
    uint32_t firstIndex = 0;
    uint32_t vertexOffset = 0;
  };
  /** 
   * \brief RTX context
   * 
   * Tracks pipeline state and records command lists.
   * This is where the actual rendering commands are
   * recorded.
   */

  class RtxContext : public DxvkContext {

  public:
    
    RtxContext(const Rc<DxvkDevice>& device);
    ~RtxContext();

    float getGpuIdleTimeSinceLastCall();

    /**
      * \brief Reset screen resolution, and resize all screen
      *        buffers to specified resolution if required.
      * 
      * \param [in] upscaleExtent: New desired resolution.
      */
    void resetScreenResolution(const VkExtent3D& upscaleExtent);

    /**
      * \brief Triggers the RTX renderer.  Writes to targetImage (if specified) 
      *        and the currently bound render target if not.
      *        Will flush the scene and perform other non rendering tasks.
      * 
      * \param [in] cachedReflexFrameId: The Reflex frame ID at the time of calling, cached so Reflex can have
      * consistent frame IDs throughout the dispatches of an application frame.
      * \param [in] targetImage: Image to store raytraced result in
      */
    void injectRTX(std::uint64_t cachedReflexFrameId, Rc<DxvkImage> targetImage = nullptr);
    void endFrame(std::uint64_t cachedReflexFrameId, Rc<DxvkImage> targetImage = nullptr, bool callInjectRtx = true);

    // NGX passthrough mode: per-frame data handed over from the D3D9 layer - the game's scene
    // depth image, the pre-post-process color target (the game's scene color when the
    // injection point is before the game's post chain, null when DLSS runs on the final
    // output), the suppressed ScreenPercentage-upscale source (null when the game renders
    // at full resolution), the scene subrect, the captured dynamic object draws with their
    // capture diagnostics, and the sub-pixel viewport jitter the frame was rasterized with
    void setNgxPassthroughFrameData(const Rc<DxvkImage>& sceneDepthImage,
                                    const Rc<DxvkImage>& colorTargetImage,
                                    const Rc<DxvkImage>& colorMirrorImage,
                                    const Rc<DxvkImage>& upscaleSourceImage,
                                    const VkRect2D& sourceSubrect,
                                    const VkOffset2D& colorSourceOffset,
                                    const NgxOutputTransform& outputTransform,
                                    std::vector<NgxVelocityDraw>&& velocityDraws,
                                    const NgxVelocityCaptureStats& velocityStats,
                                    const Vector3& sceneTransformOffset,
                                    float jitterX, float jitterY,
                                    bool cameraMatricesValid,
                                    const Matrix4& worldToView,
                                    const Matrix4& viewToProjection);

    // Copies the pre-UI backbuffer into the HUD-less slot. Emitted by the D3D9 layer.
    void captureNgxPassthroughHudless(const Rc<DxvkImage>& backbufferImage);

    // Runs DLSS-NR on the pre-UI backbuffer of a pre-post frame. Emitted by the D3D9 layer.
    void applyNgxPassthroughNeuralRendering(const Rc<DxvkImage>& backbufferImage);

    // Copies that slot onto the backbuffer for the developer preview.
    void previewNgxPassthroughHudless(const Rc<DxvkImage>& backbufferImage);

    // NGX passthrough mode: copies the game's depth buffer into a runtime-owned snapshot.
    // Emitted by the D3D9 layer right before the game's first mid-scene depth clear (UE3
    // clears the depth buffer ahead of its foreground DPG, which would otherwise destroy
    // the world depth needed for motion vector generation).
    void snapshotNgxPassthroughDepth(const Rc<DxvkImage>& sceneDepthImage);


    void onPresent(Rc<DxvkImage> targetImage = nullptr);

    virtual void flushCommandList() override;

    SceneManager& getSceneManager();
    Resources& getResourceManager();
  
    static void triggerScreenshot() { s_triggerScreenshot = true; }

#ifdef REMIX_DEVELOPMENT
    /** When crash hotkeys are armed, checks if CPU or GPU crash hotkey was pressed; returns true if injectRTX should return immediately (e.g. after GPU crash). */
    bool handleCrashHotkeys();
#endif

  protected:
    virtual void updateComputeShaderResources() override;

  private:
    void reportCpuSimdSupport();

    void takeScreenshot(std::string imageName, Rc<DxvkImage> image);

    VkExtent3D setDownscaleExtent(const VkExtent3D& upscaleExtent);

    void onInjectRtxFrameEnd(bool raytracedThisFrame);

    void dispatchNgxPassthrough(Rc<DxvkImage> targetImage);
    void updateMetrics(const float gpuIdleTimeMilliseconds);

    uint32_t m_frameLastInjected = kInvalidFrameIndex;
    uint32_t m_lastMetricsFrameId = kInvalidFrameIndex;
    float m_metricsGpuIdleTimeMs = 0.0f;

    bool shouldUseDLSS() const;
    bool shouldUseNIS() const;
    bool shouldUseTAA() const;
    bool shouldUseXeSS() const;

    inline static bool s_triggerScreenshot = false;

    bool m_dlssSupported;
    bool m_submitContainsInjectRtx = false;
    uint64_t m_cachedReflexFrameId = 0;

    bool m_resetHistory = true;    // Discards use of temporal data in passes

    // NGX passthrough mode inputs from the D3D9 layer (valid for the current frame only)
    Rc<DxvkImage> m_ngxPassthroughSceneDepth;
    Rc<DxvkImage> m_ngxPassthroughColorTarget;
    Rc<DxvkImage> m_ngxPassthroughColorMirror;
    Rc<DxvkImage> m_ngxPassthroughUpscaleSource;
    VkRect2D m_ngxPassthroughSubrect = { { 0, 0 }, { 0, 0 } };
    VkOffset2D m_ngxPassthroughColorSubrectOffset = { 0, 0 };
    NgxOutputTransform m_ngxPassthroughOutputTransform;
    std::vector<NgxVelocityDraw> m_ngxPassthroughVelocityDraws;
    float m_ngxPassthroughJitter[2] = { 0.0f, 0.0f };

    std::chrono::time_point<std::chrono::steady_clock> m_prevRunningTime;
    uint64_t m_prevGpuIdleTicks = 0;
    bool m_prevGpuIdleTicksInitialized = false;

    bool m_screenshotFrameEnabled = false;
    bool m_triggerDelayedTerminate = false;
    uint32_t m_screenshotFrameNum = -1;
    uint32_t m_terminateAppFrameNum = -1;
  };
} // namespace dxvk
