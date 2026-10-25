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
#include <cstring>
#include <cmath>
#include <cassert>

#include "dxvk_device.h"
#include "dxvk_gpu_query.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_shader_manager.h"
#include "dxvk_adapter.h"
#include "rtx_context.h"
#include "rtx_asset_exporter.h"
#include "rtx_options.h"
#include "rtx_xess.h"
#include "rtx_ngx_passthrough.h"
#include "rtx_scene_manager.h"

#include "../util/log/metrics.h"
#include "../util/util_defer.h"
#include "../util/util_global_time.h"
#include "../util/util_sentry.h"

#include "rtx_imgui.h"
#include "dxvk_scoped_annotation.h"
#include "imgui/dxvk_imgui.h"

#include <ctime>
#include <nvapi.h>

#include <NvLowLatencyVk.h>
#include <pclstats.h>

#include "rtx_matrix_helpers.h"
#include "../util/util_fastops.h"

namespace dxvk {

  Metrics Metrics::s_instance;

  void RtxContext::takeScreenshot(std::string imageName, Rc<DxvkImage> image) {
    // NOTE: Improve this, I'd like all textures from the same frame to have the same time code...  Currently sampling the time on each "dump op" results in different timecodes.
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::string path = env::getEnvVar("DXVK_SCREENSHOT_PATH");

    if (path.empty()) {
      path = "./Screenshots/";
    } else if (*path.rbegin() != '/') {
      path += '/';
    }

    auto& exporter = getCommonObjects()->metaExporter();
    exporter.dumpImageToFile(this, path, str::format(imageName, "_", tm.tm_mday, tm.tm_mon, tm.tm_year, "-", tm.tm_hour, tm.tm_min, tm.tm_sec, ".dds"), image);
  }

  RtxContext::RtxContext(const Rc<DxvkDevice>& device)
    : DxvkContext(device) {
    m_dlssSupported = (m_device->extensions().nvxBinaryImport &&
                       m_device->extensions().nvxImageViewHandle &&
                       m_device->extensions().khrPushDescriptor);


    if (env::getEnvVar("DXVK_DUMP_SCREENSHOT_FRAME") != "") {
      m_screenshotFrameNum = stoul(env::getEnvVar("DXVK_DUMP_SCREENSHOT_FRAME"));
      m_screenshotFrameEnabled = true;
    }

    if (env::getEnvVar("DXVK_TERMINATE_APP_FRAME") != "") {
      m_terminateAppFrameNum = stoul(env::getEnvVar("DXVK_TERMINATE_APP_FRAME"));
      m_triggerDelayedTerminate = true;
    }

    Metrics::TestTraceConfig testTraceConfig;
    testTraceConfig.enabled = RtxOptions::Automation::enableTestTrace();
    testTraceConfig.screenshotFrameEnabled = m_screenshotFrameEnabled;
    testTraceConfig.screenshotFrameNum = m_screenshotFrameNum;
    testTraceConfig.terminateAppFrameNum = m_terminateAppFrameNum;
    Metrics::configureTestTrace(testTraceConfig);

    m_prevRunningTime = std::chrono::steady_clock::now();

    reportCpuSimdSupport();

    GlobalTime::get().init(RtxOptions::timeDeltaBetweenFrames() * 0.001f);
    GlobalTime::get().setAdvanceTime(RtxOptions::advanceTime());
  }

  RtxContext::~RtxContext() {
    getCommonObjects()->metaExporter().waitForAllExportsToComplete();

    if (m_screenshotFrameNum != -1 || m_terminateAppFrameNum != -1) {
      Metrics::serialize();
    }

  }

  SceneManager& RtxContext::getSceneManager() {
    return getCommonObjects()->getSceneManager();
  }
  Resources& RtxContext::getResourceManager() {
    return getCommonObjects()->getResources();
  }

  // Returns GPU idle time between calls to this in milliseconds
  float RtxContext::getGpuIdleTimeSinceLastCall() {
    uint64_t currGpuIdleTicks = m_device->getStatCounters().getCtr(DxvkStatCounter::GpuIdleTicks);
    if (!m_prevGpuIdleTicksInitialized) {
      // DxvkSubmissionQueue::gpuIdleTicks() is a monotonic accumulator, so the
      // only invalid sample here is the first one before we've established a baseline.
      m_prevGpuIdleTicks = currGpuIdleTicks;
      m_prevGpuIdleTicksInitialized = true;
      return 0.0f;
    }

    uint64_t delta = currGpuIdleTicks - m_prevGpuIdleTicks;
    m_prevGpuIdleTicks = currGpuIdleTicks;

    return static_cast<float>(delta) * 0.001f; // to milliseconds
  }

  VkExtent3D RtxContext::setDownscaleExtent(const VkExtent3D& upscaleExtent) {
    ScopedCpuProfileZone();
    VkExtent3D downscaleExtent;
    if (shouldUseDLSS()) {
      DxvkDLSS& dlss = m_common->metaDLSS();
      uint32_t displaySize[2] = { upscaleExtent.width, upscaleExtent.height };
      uint32_t renderSize[2];
      dlss.setSetting(displaySize, RtxOptions::qualityDLSS(), renderSize);
      downscaleExtent.width = renderSize[0];
      downscaleExtent.height = renderSize[1];
      downscaleExtent.depth = 1;
    } else if (shouldUseXeSS()) {
      DxvkXeSS& xess = m_common->metaXeSS();
      uint32_t displaySize[2] = { upscaleExtent.width, upscaleExtent.height };
      uint32_t renderSize[2];
      xess.setSetting(displaySize, DxvkXeSS::XessOptions::preset(), renderSize);
      downscaleExtent.width = renderSize[0];
      downscaleExtent.height = renderSize[1];
      downscaleExtent.depth = 1;
      
      // XeSS: Apply recommended jitter sequence length if enabled
      if (DxvkXeSS::XessOptions::useRecommendedJitterSequenceLength() && xess.isActive()) {
        uint32_t recommendedJitterLength = xess.calcRecommendedJitterSequenceLength();
        uint32_t currentJitterLength = RtxOptions::cameraJitterSequenceLength();
      }
    } else if (shouldUseNIS() || shouldUseTAA()) {
      auto resolutionScale = RtxOptions::resolutionScale();
      downscaleExtent.width = uint32_t(std::roundf(upscaleExtent.width * resolutionScale));
      downscaleExtent.height = uint32_t(std::roundf(upscaleExtent.height * resolutionScale));
      downscaleExtent.depth = 1;
    } else {
      downscaleExtent = upscaleExtent;
    }
    downscaleExtent.width = std::max(downscaleExtent.width, 1u);
    downscaleExtent.height = std::max(downscaleExtent.height, 1u);

    return downscaleExtent;
  }

  void RtxContext::resetScreenResolution(const VkExtent3D& upscaleExtent) {
    // Calculate extents based on if DLSS is enabled or not
    const VkExtent3D downscaleExtent = setDownscaleExtent(upscaleExtent);

    uint32_t renderSize[] = { downscaleExtent.width, downscaleExtent.height };
    uint32_t displaySize[] = { upscaleExtent.width, upscaleExtent.height };

    DlssNeuralRendering& dlssnr = m_common->metaDlssNeuralRendering();
    dlssnr.setDlssNeuralRenderingSettings(displaySize);

    // Set resolution to cameras for jittering
    for (int i = 0; i < CameraType::Count; i++) {
      if (i == CameraType::Unknown) {
        continue;
      }
      RtCamera& camera = getSceneManager().getCameraManager().getCamera(static_cast<CameraType::Enum>(i));
      camera.setResolution(renderSize, displaySize);
    }

    // Note: Ensure the rendering resolution is not more than 2^14 - 1. This is due to assuming only
    // 14 of the 16 bits of an integer will be used for these pixel coordinates to pack additional data
    // into the free bits in memory payload structures on the GPU.
    assert((renderSize[0] < (1 << 14)) && (renderSize[1] < (1 << 14)));
  }

  void RtxContext::onInjectRtxFrameEnd(bool rayTracedThisFrame) {
    getSceneManager().onFrameEnd(this, rayTracedThisFrame);
  }
  
#ifdef REMIX_DEVELOPMENT
  bool RtxContext::handleCrashHotkeys() {
    // Crash Hotkey Feature: When armed via the Development tab checkbox, pressing the crash hotkey
    // triggers a deliberate null pointer dereference crash. This is useful for testing crash handling,
    // crash dumps, and crash reporting systems.
    static bool crashHotkeyStartupLogged = false;
    if (!crashHotkeyStartupLogged && RtxOptions::enableCrashHotkey()) {
      const auto crashHotkeyStr = buildKeyBindDescriptorStringForDisplay(RtxOptions::crashHotkey());
      const auto gpuCrashHotkeyStr = buildKeyBindDescriptorStringForDisplay(RtxOptions::gpuCrashHotkey());
      Logger::warn(str::format("Crash hotkeys ARMED at startup - ", crashHotkeyStr, " = CPU crash, ", gpuCrashHotkeyStr, " = GPU crash"));
      crashHotkeyStartupLogged = true;
    }

    if (RtxOptions::enableCrashHotkey() && ImGUI::checkHotkeyState(RtxOptions::crashHotkey(), false)) {
      const auto crashHotkeyStr = buildKeyBindDescriptorStringForDisplay(RtxOptions::crashHotkey());
      Logger::err(str::format("Deliberate crash triggered via crash hotkey (", crashHotkeyStr, ")"));
      // Trigger a null pointer dereference to cause a crash
      volatile int* nullPtr = nullptr;
      *nullPtr = 0xDEAD;
    }

    if (RtxOptions::enableCrashHotkey() && ImGUI::checkHotkeyState(RtxOptions::gpuCrashHotkey(), false)) {
      const auto gpuCrashHotkeyStr = buildKeyBindDescriptorStringForDisplay(RtxOptions::gpuCrashHotkey());
      Logger::warn(str::format("GPU crash triggered via hotkey (", gpuCrashHotkeyStr, ")"));
      commitGraphicsState<true, false>();
      getCommonObjects()->metaGpuCrash().dispatch(this);
      return true;
    }
    return false;
  }
#endif

  // Hooked into D3D9 presentImage (same place HUD rendering is)
  void RtxContext::injectRTX(std::uint64_t cachedReflexFrameId, Rc<DxvkImage> targetImage) {
    ScopedCpuProfileZone();
    // NV-DXVK start: CPU frame breakdown for the built-in pass timer
    RtxGpuPassTimer::CpuScope injectCpuScope(RtxGpuPassTimer::isEnabled() ? &getCommonObjects()->metaGpuPassTimer() : nullptr,
                                             RtxGpuPassTimer::CpuCounter::CsInjectRtx);
    // NV-DXVK end

    if (RtxOptions::enableBreakIntoDebuggerOnPressingB() && ImGUI::checkHotkeyState({VirtualKey{ 'B' }}, true)) {
      while (!::IsDebuggerPresent()) {
        ::Sleep(100);
      }
      __debugbreak();
    }

#ifdef REMIX_DEVELOPMENT
    if (handleCrashHotkeys()) {
      return;
    }
#endif

    commitGraphicsState<true, false>();

    auto common = getCommonObjects();

    // NV-DXVK start: built-in GPU pass timings
    // Resolve timestamp queries of completed frames before this frame records new zones.
    common->metaGpuPassTimer().onFrameBegin(this);
    // NV-DXVK end

    const auto asyncShaderCompilationActive = RtxOptions::Shader::enableAsyncCompilation() && common->pipelineManager().remixShaderCompilationCount() > 0;

    const bool isCameraValid = getSceneManager().getCamera().isValid(m_device->getCurrentFrameId());

    // Note: This must be done before the early out return below so the present throttle delay is always set.
    m_device->setPresentThrottleDelay(RtxOptions::enablePresentThrottle() ? RtxOptions::presentThrottleDelay() : 0);

    // Early out if Remix has already been injected this frame
    if (m_frameLastInjected == m_device->getCurrentFrameId()) {
      return;
    }

    if (RtxOptions::upscalerType() == UpscalerType::DLSS && !common->metaDLSS().supportsDLSS()) {
      RtxOptions::upscalerType.setDeferred(UpscalerType::TAAU);
    }

    if (DxvkDLFG::enable() && !common->metaDLFG().supportsDLFG()) {
      DxvkDLFG::enable.setDeferred(false);
    }
    
#ifdef REMIX_DEVELOPMENT
    // Update the Shader Manager

    ShaderManager::getInstance()->update();
#endif

    const float gpuIdleTimeMilliseconds = getGpuIdleTimeSinceLastCall();
    Metrics::TestTraceSample testTraceSample;
    testTraceSample.frameId = m_device->getCurrentFrameId();
    testTraceSample.effectiveDeltaMs = GlobalTime::get().deltaTimeMs();
    testTraceSample.realWallDeltaMs = GlobalTime::get().realDeltaTimeMs();
    testTraceSample.gpuIdleTimeMs = gpuIdleTimeMilliseconds;
    testTraceSample.shaderCompileInflightCount = getCommonObjects()->pipelineManager().remixShaderCompilationCount();
    testTraceSample.cameraValid = isCameraValid;
    testTraceSample.asyncShaderPrewarming = RtxInitializer::asyncShaderPrewarming();
    testTraceSample.asyncCompilationEnabled = RtxOptions::Shader::enableAsyncCompilation();
    testTraceSample.asyncCompilationActive = asyncShaderCompilationActive;
    Metrics::recordTestTrace(testTraceSample);


    // The game's raster frame, with the selected upscaler, frame generation, and Reflex.
    // Not gated on async shader compilation: that flag tracks the path tracer's shader set.
    const bool ngxPassthroughCanDispatch =
      isCameraValid || m_ngxPassthroughSceneDepth != nullptr;

    if (ngxPassthroughCanDispatch) {
      if (targetImage == nullptr) {
        targetImage = m_state.om.renderTargets.color[0].view->image();
      }

      // Signal Reflex rendering start for this frame's submit
      RtxReflex& reflex = m_common->metaReflex();
      reflex.updateMode();

      m_submitContainsInjectRtx = true;
      m_cachedReflexFrameId = cachedReflexFrameId;

      dispatchNgxPassthrough(targetImage);

      m_frameLastInjected = m_device->getCurrentFrameId();
    } else {
      m_common->metaNgxPassthrough().noteCameraInvalidFrame();
    }

    onInjectRtxFrameEnd(false);

    // apply changes to RtxOptions after the frame has ended
    RtxOptionManager::applyPendingValues(m_device.ptr(), /* forceOnChange */ false);

    // Update stats
    updateMetrics(gpuIdleTimeMilliseconds);

    m_resetHistory = false;
  }

  void RtxContext::endFrame(std::uint64_t cachedReflexFrameId, Rc<DxvkImage> targetImage, bool callInjectRtx) {

    if (callInjectRtx) {
      // Second chance for frames that never hit the scene-end trigger. injectRTX returns
      // immediately when this frame was already dispatched.
      injectRTX(cachedReflexFrameId, targetImage);
    }

    // Update time on the frame end so all other systems can benefit from a global time
    GlobalTime::get().update();
  }

  // Called right before D3D9 present
  void RtxContext::onPresent(Rc<DxvkImage> targetImage) {
    {
      static bool s_firstFrameDone = false;
      if (!s_firstFrameDone) {
        s_firstFrameDone = true;
        sentry::onFirstFrame();
      }
    }
    // Screenshots capture the present image, UI included.
    {
      const bool captureTestScreenshot = (m_screenshotFrameEnabled && m_device->getCurrentFrameId() == m_screenshotFrameNum);
      if (s_triggerScreenshot || captureTestScreenshot) {
        if (targetImage == nullptr) {
          targetImage = m_state.om.renderTargets.color[0].view->image();
        }
        takeScreenshot("rtxImageDxvkView", targetImage);
      }
    }
    s_triggerScreenshot = false;

    // Some time in the future kill process
    if (m_triggerDelayedTerminate &&
        (m_device->getCurrentFrameId() > m_terminateAppFrameNum)) {
      Logger::info(str::format("RTX: Terminating application"));
      Metrics::serialize();
      getCommonObjects()->metaExporter().waitForAllExportsToComplete();

      env::killProcess();
    }

    // This needs to happen at the end of frame, after ImGUI rendering
    GpuMemoryTracker::onFrameEnd();
  }

  void RtxContext::updateMetrics(const float gpuIdleTimeMilliseconds) {
    ScopedCpuProfileZone();
    const uint32_t frameId = m_device->getCurrentFrameId();

    // injectRTX can run twice in a frame, so sample once. GPU idle time from a repeat call carries into the next sample.
    m_metricsGpuIdleTimeMs += gpuIdleTimeMilliseconds;
    if (m_lastMetricsFrameId == frameId) {
      return;
    }
    m_lastMetricsFrameId = frameId;
    const float frameGpuIdleTimeMs = m_metricsGpuIdleTimeMs;
    m_metricsGpuIdleTimeMs = 0.0f;

    const float timeSinceStartMs = static_cast<float>(GlobalTime::get().realTimeSinceStartMs());
    Metrics::logFloat(Metric::dxvk_total_time_ms, timeSinceStartMs);
    Metrics::logFloat(Metric::dxvk_frame_count, static_cast<float>(frameId));

    const uint32_t warmupFrames = RtxOptions::Automation::metricsWarmupFrames();
    const bool isWarmup = warmupFrames > 0 && frameId <= warmupFrames;
    if (isWarmup) {
      Metrics::logFloat(Metric::dxvk_warmup_time_ms, timeSinceStartMs);
      return;
    }

    Metrics::logAverage(Metric::dxvk_average_frame_time_ms, GlobalTime::get().realDeltaTimeMs()); // In milliseconds
    Metrics::logAverage(Metric::dxvk_gpu_idle_time_ms, frameGpuIdleTimeMs); // In milliseconds
    uint64_t vidUsageMib = 0;
    uint64_t sysUsageMib = 0;
    const VkPhysicalDeviceMemoryProperties memprops = m_device->adapter()->memoryProperties();
    for (uint32_t i = 0; i < memprops.memoryHeapCount; i++) {
      bool isDeviceLocal = memprops.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;

      if (isDeviceLocal) {
        vidUsageMib += m_device->getMemoryStats(i).totalUsed() >> 20;
      }
      else {
        sysUsageMib += m_device->getMemoryStats(i).totalUsed() >> 20;
      }
    }
    Metrics::logAverage(Metric::dxvk_vid_memory_usage_mb, static_cast<float>(vidUsageMib)); // In MB
    Metrics::logAverage(Metric::dxvk_sys_memory_usage_mb, static_cast<float>(sysUsageMib)); // In MB
  }

  void RtxContext::snapshotNgxPassthroughDepth(const Rc<DxvkImage>& sceneDepthImage) {
    ScopedGpuProfileZone(this, "NGX Passthrough Depth Snapshot");
    m_common->metaNgxPassthrough().captureDepthSnapshot(this, sceneDepthImage);
  }

  void RtxContext::captureNgxPassthroughHudless(const Rc<DxvkImage>& backbufferImage) {
    ScopedGpuProfileZone(this, "NGX Passthrough HUD-less Capture");
    spillRenderPass(false);
    m_common->metaNgxPassthrough().captureHudless(this, backbufferImage);
  }

  void RtxContext::applyNgxPassthroughNeuralRendering(const Rc<DxvkImage>& backbufferImage) {
    spillRenderPass(false);
    m_execBarriers.recordCommands(m_cmd);
    m_common->metaNgxPassthrough().applyNeuralRendering(this, m_execBarriers, backbufferImage);
  }

  void RtxContext::previewNgxPassthroughHudless(const Rc<DxvkImage>& backbufferImage) {
    ScopedGpuProfileZone(this, "NGX Passthrough HUD-less Preview");
    spillRenderPass(false);
    m_common->metaNgxPassthrough().previewHudless(this, backbufferImage);
  }


  void RtxContext::setNgxPassthroughFrameData(const Rc<DxvkImage>& sceneDepthImage,
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
                                              const Matrix4& viewToProjection) {
    if (cameraMatricesValid) {
      getSceneManager().getCameraManager().processExternalCamera(
        CameraType::Main, worldToView, viewToProjection);
    }

    m_ngxPassthroughSceneDepth = sceneDepthImage;
    m_ngxPassthroughColorTarget = colorTargetImage;
    m_ngxPassthroughColorMirror = colorMirrorImage;
    m_ngxPassthroughUpscaleSource = upscaleSourceImage;
    m_ngxPassthroughSubrect = sourceSubrect;
    m_ngxPassthroughColorSubrectOffset = colorSourceOffset;
    m_ngxPassthroughOutputTransform = outputTransform;
    m_ngxPassthroughVelocityDraws = std::move(velocityDraws);
    m_ngxPassthroughJitter[0] = jitterX;
    m_ngxPassthroughJitter[1] = jitterY;

    m_common->metaNgxPassthrough().setVelocityCaptureStats(velocityStats);
    m_common->metaNgxPassthrough().setSceneTransformOffset(sceneTransformOffset);
    m_common->metaNgxPassthrough().setSceneCameraFresh(cameraMatricesValid);
  }

  void RtxContext::dispatchNgxPassthrough(Rc<DxvkImage> targetImage) {
    ScopedCpuProfileZone();

    // Flush any pending game rasterization out of the active render pass before compute work
    this->spillRenderPass(false);
    m_execBarriers.recordCommands(m_cmd);

    RtCamera& camera = getSceneManager().getCameraManager().getCamera(CameraType::Main);

    const bool resetHistory = m_resetHistory || camera.isViewHistoryInvalidated(m_device->getCurrentFrameId());

    // Pre-post-process injection: DLSS reads and writes the game's scene color instead of
    // the render target bound at the trigger draw (the first post pass's own output target)
    const bool prePostProcess = m_ngxPassthroughColorTarget != nullptr;
    if (prePostProcess) {
      targetImage = m_ngxPassthroughColorTarget;
    }

    m_common->metaNgxPassthrough().dispatch(this, m_state, m_execBarriers, targetImage,
                                            m_ngxPassthroughColorMirror,
                                            m_ngxPassthroughSceneDepth,
                                            m_ngxPassthroughUpscaleSource, m_ngxPassthroughSubrect,
                                            m_ngxPassthroughColorSubrectOffset,
                                            m_ngxPassthroughOutputTransform,
                                            prePostProcess,
                                            m_ngxPassthroughVelocityDraws,
                                            m_ngxPassthroughJitter, resetHistory);

    // Do not hold references to the game's resources across frames
    m_ngxPassthroughSceneDepth = nullptr;
    m_ngxPassthroughColorTarget = nullptr;
    m_ngxPassthroughColorMirror = nullptr;
    m_ngxPassthroughUpscaleSource = nullptr;
    m_ngxPassthroughSubrect = { { 0, 0 }, { 0, 0 } };
    m_ngxPassthroughColorSubrectOffset = { 0, 0 };
    m_ngxPassthroughOutputTransform = NgxOutputTransform();
    m_ngxPassthroughVelocityDraws.clear();
  }


  void RtxContext::flushCommandList() {
    ScopedCpuProfileZone();

    m_device->submitCommandList(
      this->endRecording(),
      VK_NULL_HANDLE,
      VK_NULL_HANDLE,
      m_submitContainsInjectRtx,
      m_cachedReflexFrameId);
    
    // Reset this now that we've completed the submission
    m_submitContainsInjectRtx = false;
    
    this->beginRecording(
      m_device->createCommandList());
  }

  void RtxContext::updateComputeShaderResources() {
    ScopedCpuProfileZone();
    DxvkContext::updateComputeShaderResources();
  }

  bool RtxContext::shouldUseDLSS() const {
    // Note: m_dlssSupported only checks for the presence of some basic extensions, the actual DLSS context needs to be queried to see
    // if a given platform supports DLSS (as this will depend on if it was actually initialized successfully or not). Cases where m_dlssSupported
    // is true but supportsDLSS() is not are for example when the DLSS DLL is missing.
    return RtxOptions::isDLSSEnabled() && m_dlssSupported && m_common->metaDLSS().supportsDLSS();
  }

  bool RtxContext::shouldUseNIS() const {
    return RtxOptions::isNISEnabled();
  }

  bool RtxContext::shouldUseTAA() const {
    return RtxOptions::isTAAEnabled();
  }

  bool RtxContext::shouldUseXeSS() const {
    return RtxOptions::upscalerType() == UpscalerType::XeSS;
  }

  void RtxContext::reportCpuSimdSupport() {
    switch (fast::getSimdSupportLevel()) {
    case fast::AVX512:
      dxvk::Logger::info("CPU supports SIMD: AVX512");
      break;
    case fast::AVX2:
      dxvk::Logger::info("CPU supports SIMD: AVX2");
      break;
    case fast::SSE4_1:
      dxvk::Logger::info("CPU supports SIMD: SSE 4.1");
      break;
    case fast::SSE3:
      dxvk::Logger::info("CPU supports SIMD: SSE 3");
      break;
    case fast::SSE2:
      dxvk::Logger::info("CPU supports SIMD: SSE 2");
      break;
    case fast::None:
      dxvk::Logger::info("CPU doesn't support SIMD");
      break;
    default:
      Logger::err("Invalid SIMD state");
      break;
    }
  }
} // namespace dxvk
