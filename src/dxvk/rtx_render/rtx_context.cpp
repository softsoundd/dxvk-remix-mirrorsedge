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
#include "rtx_bindless_resource_manager.h"
#include "rtx_opacity_micromap_manager.h"
#include "rtx_asset_replacer.h"
#include "rtx_terrain_baker.h"
#include "rtx_texture_manager.h"
#include "rtx_neural_radiance_cache.h"
#include "rtx_ray_reconstruction.h"
#include "rtx_xess.h"
#include "rtx_rtxdi_rayquery.h"
#include "rtx_restir_gi_rayquery.h"
#include "rtx_composite.h"
#include "rtx_debug_view.h"
#include "rtx_ngx_passthrough.h"

#include "rtx/pass/sparse_rendering/sparse_rendering.h"

#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/raytrace_args.h"
#include "rtx/pass/volume_args.h"
#include "rtx/utility/debug_view_indices.h"
#include "rtx/utility/gpu_printing.h"
#include "rtx_nrd_settings.h"
#include "rtx_scene_manager.h"
#include "rtx_sparse_rendering.h"

#include "../d3d9/d3d9_state.h"
#include "../d3d9/d3d9_spec_constants.h"

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

// Destructor requires the struct definitions
#include "rtx_sky.h"

namespace dxvk {

  Metrics Metrics::s_instance;

  bool g_allowSrgbConversionForOutput = true;
  bool g_forceKeepObjectPickingImage = false;

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

  void RtxContext::blitImageHelper(Rc<DxvkContext> ctx, const Rc<DxvkImage>& srcImage, const Rc<DxvkImage>& dstImage, VkFilter filter) {
    const DxvkFormatInfo* dstFormatInfo = imageFormatInfo(dstImage->info().format);
    const DxvkFormatInfo* srcFormatInfo = imageFormatInfo(srcImage->info().format);

    const VkImageSubresource dstSubresource = { dstFormatInfo->aspectMask, 0, 0 };
    const VkImageSubresource srcSubresource = { srcFormatInfo->aspectMask, 0, 0 };

    VkExtent3D srcExtent = srcImage->mipLevelExtent(srcSubresource.mipLevel);
    VkExtent3D dstExtent = dstImage->mipLevelExtent(dstSubresource.mipLevel);

    VkImageSubresourceLayers dstSubresourceLayers = {
      dstSubresource.aspectMask,
      dstSubresource.mipLevel,
      dstSubresource.arrayLayer, 1 };

    VkImageSubresourceLayers srcSubresourceLayers = {
      srcSubresource.aspectMask,
      srcSubresource.mipLevel,
      srcSubresource.arrayLayer, 1 };

    VkImageBlit blitInfo;

    blitInfo.dstSubresource = dstSubresourceLayers;
    blitInfo.srcSubresource = srcSubresourceLayers;

    blitInfo.dstOffsets[0] = VkOffset3D{ 0,                        0,                       0 };
    blitInfo.dstOffsets[1] = VkOffset3D{ int32_t(dstExtent.width),  int32_t(dstExtent.height),  1 };

    blitInfo.srcOffsets[0] = VkOffset3D{ 0,                          0,                         0 };
    blitInfo.srcOffsets[1] = VkOffset3D{ int32_t(srcExtent.width),    int32_t(srcExtent.height),    1 };

    VkComponentMapping swizzle = {
      VK_COMPONENT_SWIZZLE_IDENTITY,
      VK_COMPONENT_SWIZZLE_IDENTITY,
      VK_COMPONENT_SWIZZLE_IDENTITY,
      VK_COMPONENT_SWIZZLE_IDENTITY,
    };

    ctx->blitImage(dstImage, swizzle, srcImage, swizzle, blitInfo, filter);
  }

  RtxContext::RtxContext(const Rc<DxvkDevice>& device)
    : DxvkContext(device) {
    // Note: This may not be the best place to check for these features/properties, they ideally would be specified as
    // required upfront, but there's no good place to do that for this RTX extension (the D3D9 stuff does it before device
    // creation), so instead we just check for what is needed.
    // Note: When adding new extensions update DxvkAdapter::createDevice as it is what brings these features over.
    m_rayTracingSupported = (m_device->features().core.features.shaderInt16 &&
                             m_device->features().vulkan11Features.storageBuffer16BitAccess &&
                             m_device->features().vulkan11Features.uniformAndStorageBuffer16BitAccess &&
                             m_device->features().vulkan12Features.bufferDeviceAddress &&
                             m_device->features().vulkan12Features.descriptorIndexing &&
                             m_device->features().vulkan12Features.runtimeDescriptorArray &&
                             m_device->features().vulkan12Features.descriptorBindingPartiallyBound &&
                             m_device->features().vulkan12Features.shaderStorageBufferArrayNonUniformIndexing &&
                             m_device->features().vulkan12Features.shaderSampledImageArrayNonUniformIndexing &&
                             m_device->features().vulkan12Features.descriptorBindingVariableDescriptorCount &&
                             m_device->features().vulkan12Features.shaderInt8 &&
                             m_device->features().vulkan12Features.shaderFloat16 &&
                             m_device->features().vulkan12Features.uniformAndStorageBuffer8BitAccess &&
                             m_device->features().khrAccelerationStructureFeatures.accelerationStructure &&
                             m_device->features().khrRayQueryFeatures.rayQuery &&
                             m_device->features().khrDeviceRayTracingPipelineFeatures.rayTracingPipeline &&
                             m_device->extensions().khrShaderInt8Float16Types &&
                             m_device->properties().coreSubgroup.subgroupSize >= 1 &&
                             m_device->properties().coreSubgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT &&
                             m_device->properties().coreSubgroup.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);

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

    checkOpacityMicromapSupport();
    checkShaderExecutionReorderingSupport();
    checkNeuralRadianceCacheSupport();
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
    } else if (shouldUseRayReconstruction()) {
      DxvkRayReconstruction& rayReconstruction = m_common->metaRayReconstruction();
      uint32_t displaySize[2] = { upscaleExtent.width, upscaleExtent.height };
      uint32_t renderSize[2];
      rayReconstruction.setSettings(displaySize, RtxOptions::qualityDLSS(), renderSize);
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

    // With reloadTextureWhenResolutionChanged ON, textures will get reloaded when resolution is changed,
    // which may cause long wait when changing DLSS-RR or other upscalers' settings.
    // Therefore reloadTextureWhenResolutionChanged is set to OFF by default to improve performance. 
    if (RtxOptions::reloadTextureWhenResolutionChanged()) {
      getSceneManager().requestTextureVramFree();
    }
  }

  bool RtxContext::useRayReconstruction() const {
    return m_common->metaRayReconstruction().useRayReconstruction();
  }

  RtxContext::InternalUpscaler RtxContext::getCurrentFrameUpscaler() {
    if (shouldUseDLSS() && m_common->metaDLSS().isActive()) {
      return InternalUpscaler::DLSS;
    } else if (shouldUseRayReconstruction() && m_common->metaRayReconstruction().isActive()) {
      return InternalUpscaler::DLSS_RR;
    } else if (shouldUseXeSS() && m_common->metaXeSS().isActive()) {
      return InternalUpscaler::XeSS;
    } else if (shouldUseNIS()) {
      return InternalUpscaler::NIS;
    } else if (shouldUseTAA()) {
      return InternalUpscaler::TAAU;
    } else {
      return InternalUpscaler::None;
    }
  }


  void RtxContext::onInjectRtxFrameEnd(bool rayTracedThisFrame) {
    if (rayTracedThisFrame) {
      Resources::RaytracingOutput& rtOutput = getResourceManager().getRaytracingOutput();

      m_common->metaNeuralRadianceCache().onFrameEnd(rtOutput);
      rtOutput.onFrameEnd();
    }

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
#ifdef REMIX_DEVELOPMENT
    m_currentPassStage = RtxFramePassStage::FrameBegin;
#endif

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

    const auto isRaytracingEnabled = RtxOptions::enableRaytracing();
    const auto asyncShaderCompilationActive = RtxOptions::Shader::enableAsyncCompilation() && common->pipelineManager().remixShaderCompilationCount() > 0;

    // Whether a ray-traced scene is actually being rendered this frame.
    const bool isCameraValid = getSceneManager().getCamera().isValid(m_device->getCurrentFrameId());

    // Determine and set present throttle delay
    // Note: This must be done before the early out returns below which is why some logic here is redundant (e.g. checking if ray tracing is supported again)
    // just to ensure the present throttle delay is always being set properly.

    const auto requestedPresentThrottleDelay = RtxOptions::enablePresentThrottle() ? RtxOptions::presentThrottleDelay() : 0;
    std::uint32_t requestedAsyncShaderCompilationDelay = 0U;

    // Note: Only use the async shader compilation throttle delay when rendering which uses Remix shaders would actually take place. As such this delay is not
    // needed when ray tracing is not supported or enabled, or when there is no valid camera yet (e.g. a 2D main menu) as Remix shaders will not be
    // used in that case.
    if (m_rayTracingSupported && isRaytracingEnabled && asyncShaderCompilationActive && isCameraValid) {
      requestedAsyncShaderCompilationDelay = RtxOptions::Shader::asyncCompilationThrottleMilliseconds();
    }

    // Note: Determine the throttle delay to use based on the larger of the two requested delay values as the larger should satisfy the requests of both.
    // A sum is also potentially a valid way of going about this, but a maximum makes more sense in that these delays aren't expected to stack but rather
    // are just requests for some minimum amount of time to spend waiting per frame.
    const auto computedPresentThrottleDelay = std::max(requestedPresentThrottleDelay, requestedAsyncShaderCompilationDelay);

    m_device->setPresentThrottleDelay(computedPresentThrottleDelay);

    // Early out if ray tracing is not supported or if Remix has already been injected

    if (!m_rayTracingSupported) {
      ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Raytracing doesn't appear to be supported on this HW.")));
      return;
    }

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

    common->getTextureManager().processAllHotReloadRequests();

    const float gpuIdleTimeMilliseconds = getGpuIdleTimeSinceLastCall();
    Metrics::TestTraceSample testTraceSample;
    testTraceSample.frameId = m_device->getCurrentFrameId();
    testTraceSample.effectiveDeltaMs = GlobalTime::get().deltaTimeMs();
    testTraceSample.realWallDeltaMs = GlobalTime::get().realDeltaTimeMs();
    testTraceSample.gpuIdleTimeMs = gpuIdleTimeMilliseconds;
    testTraceSample.surfaceCount = getSceneManager().getAccelManager().getSurfaceCount();
    testTraceSample.shaderCompileInflightCount = getCommonObjects()->pipelineManager().remixShaderCompilationCount();
    testTraceSample.debugViewMode = m_common->metaDebugView().getDebugViewIndex();
    testTraceSample.compositeDebugViewMode = m_common->metaDebugView().getCompositeDebugViewIndex();
    testTraceSample.raytracingEnabled = isRaytracingEnabled;
    testTraceSample.cameraValid = isCameraValid;
    testTraceSample.asyncShaderPrewarming = RtxInitializer::asyncShaderPrewarming();
    testTraceSample.asyncCompilationEnabled = RtxOptions::Shader::enableAsyncCompilation();
    testTraceSample.asyncCompilationActive = asyncShaderCompilationActive;
    testTraceSample.surfaceBufferAvailable = getSceneManager().getSurfaceBuffer() != nullptr;
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
      m_framesWithoutValidScene = 0;
    } else {
      m_common->metaNgxPassthrough().noteCameraInvalidFrame();
      m_framesWithoutValidScene++;
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

#ifdef REMIX_DEVELOPMENT
    queryAvailableResourceAliasing();
    analyzeResourceAliasing();
    clearResourceAliasingCache();
#endif

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
    // If injectRTX couldn't screenshot a final image or a pre-present screenshot is requested,
    // take a screenshot of a present image (with UI and others)
    {
      const bool isRaytracingEnabled = RtxOptions::enableRaytracing();
      const bool isCameraValid = getSceneManager().getCameraManager().isCameraValid(CameraType::Main);

      if (!isRaytracingEnabled || !isCameraValid || s_capturePrePresentTestScreenshot) {
        const bool captureTestScreenshot = (m_screenshotFrameEnabled && m_device->getCurrentFrameId() == m_screenshotFrameNum);
        const bool captureDxvkScreenImage = s_triggerScreenshot || captureTestScreenshot;
        if (captureDxvkScreenImage) {
          if (targetImage == nullptr) {
            targetImage = m_state.om.renderTargets.color[0].view->image();
          }
          takeScreenshot("rtxImageDxvkView", targetImage);
        }
      }
    }
    s_triggerScreenshot = false;

    // Some time in the future kill process
    if (m_triggerDelayedTerminate &&
        (m_device->getCurrentFrameId() > m_terminateAppFrameNum) &&
        m_common->capturer()->isIdle()) {
      Logger::info(str::format("RTX: Terminating application"));
      Metrics::serialize();
      getCommonObjects()->metaExporter().waitForAllExportsToComplete();

      env::killProcess();
    }

    // This needs to happen at the end of frame, after ImGUI rendering
    GpuMemoryTracker::onFrameEnd();
  }

  void RtxContext::updateMetrics(const float gpuIdleTimeMilliseconds) const {
    ScopedCpuProfileZone();
    Metrics::logRollingAverage(Metric::dxvk_average_frame_time_ms, GlobalTime::get().realDeltaTimeMs()); // In milliseconds
    Metrics::logRollingAverage(Metric::dxvk_gpu_idle_time_ms, gpuIdleTimeMilliseconds); // In milliseconds
    uint64_t vidUsageMib = 0;
    uint64_t sysUsageMib = 0;
    const VkPhysicalDeviceMemoryProperties memprops = m_device->adapter()->memoryProperties();
    // Calc memory usage
    for (uint32_t i = 0; i < memprops.memoryHeapCount; i++) {
      bool isDeviceLocal = memprops.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;

      if (isDeviceLocal) {
        vidUsageMib += m_device->getMemoryStats(i).totalUsed() >> 20;
      }
      else {
        sysUsageMib += m_device->getMemoryStats(i).totalUsed() >> 20;
      }
    }
    Metrics::logRollingAverage(Metric::dxvk_vid_memory_usage_mb, static_cast<float>(vidUsageMib)); // In MB
    Metrics::logRollingAverage(Metric::dxvk_sys_memory_usage_mb, static_cast<float>(sysUsageMib)); // In MB
    Metrics::logFloat(Metric::dxvk_total_time_ms, static_cast<float>(GlobalTime::get().realTimeSinceStartMs()));
    Metrics::logFloat(Metric::dxvk_frame_count, static_cast<float>(m_device->getCurrentFrameId()));
  }

  void RtxContext::setConstantBuffers(const uint32_t vsFixedFunctionConstants, const uint32_t psSharedStateConstants, Rc<DxvkBuffer> vertexCaptureCB) {
    m_rtState.vsFixedFunctionCB = m_rc[vsFixedFunctionConstants].bufferSlice.buffer();
    m_rtState.psSharedStateCB = m_rc[psSharedStateConstants].bufferSlice.buffer();
    m_rtState.vertexCaptureCB = vertexCaptureCB;
  }

  void RtxContext::addLights(const D3DLIGHT9* pLights, const uint32_t numLights) {
    for (uint32_t i = 0; i < numLights; i++) {
      getSceneManager().addLight(pLights[i]);
    }
  }

  void RtxContext::commitGeometryToRT(const DrawParameters& params, DrawCallState& drawCallState){
    ScopedCpuProfileZone();

    const RasterGeometry& geoData = drawCallState.getGeometryData();
    DrawCallTransforms& transformData = drawCallState.modifyTransformData();

    // NV-DXVK start: static geometry hash memoization can pre-populate hashes without a future
    assert(geoData.futureGeometryHashes.valid() || geoData.hashes[HashComponents::VertexPosition] != kEmptyHash);
    // NV-DXVK end
    assert(geoData.positionBuffer.defined());

    const auto fusedMode = RtxOptions::fusedWorldViewMode();
    if (unlikely(fusedMode != FusedWorldViewMode::None)) {
      if (fusedMode == FusedWorldViewMode::View) {
        // Set World from WorldView transform
        transformData.objectToWorld = transformData.objectToView;
        // Set camera to identity
        transformData.worldToView = Matrix4();
      } else if (fusedMode == FusedWorldViewMode::World) {
        // Nothing to do...
      }
    }

    auto& cameraManager = getSceneManager().getCameraManager();

    // TODO: a last camera is used to finalize skinning...
    // processCameraData can be called only after finalizePendingFutures,
    // as we need geometry hash to check sky geometries
    const RtCamera* lastCamera =
      cameraManager.isCameraValid(cameraManager.getLastSetCameraType())
        ? &cameraManager.getCamera(cameraManager.getLastSetCameraType())
        : nullptr;

    // Sync any pending work with geometry processing threads
    if (drawCallState.finalizePendingFutures(lastCamera)) {
      drawCallState.cameraType = cameraManager.processCameraData(drawCallState);

      if (drawCallState.cameraType == CameraType::Unknown) {
        if (RtxOptions::skipObjectsWithUnknownCamera()) {
          return;
        }
        // fallback
        drawCallState.cameraType = CameraType::Enum::Main;
      }

      if (tryHandleSky(&params, &drawCallState) == TryHandleSkyResult::SkipSubmit) {
        return;
      }

      // Bake the terrain
      const MaterialData* overrideMaterialData = nullptr;
      bakeTerrain(params, drawCallState, &overrideMaterialData);

    
      // An attempt to resolve cases where games pre-combine view and world matrices
      if (RtxOptions::resolvePreCombinedMatrices() &&
        isIdentityExact(drawCallState.getTransformData().worldToView)) {
        const auto* referenceCamera = &cameraManager.getCamera(drawCallState.cameraType);
        // Note: we may accept a data even from a prev frame, as we need any information to restore;
        // but if camera data is stale, it introduces an scene object transform's lag
        if (!referenceCamera->isValid(m_device->getCurrentFrameId()) &&
          !referenceCamera->isValid(m_device->getCurrentFrameId() - 1)) {
          referenceCamera = &cameraManager.getCamera(CameraType::Main);
        }
        transformData.objectToWorld = referenceCamera->getViewToWorld(false) * drawCallState.getTransformData().objectToView;
        transformData.worldToView = referenceCamera->getWorldToView(false);
      }
      
      // Apply free camera transform when view space texGenMode is used.
      // Note: TerrainBaking already applies this transform for TexGenMode::CascadedViewPositions 
      if ((transformData.texgenMode == TexGenMode::ViewPositions
           || transformData.texgenMode == TexGenMode::ViewNormals)
          && RtCamera::enableFreeCamera()) {
        if (cameraManager.isCameraValid(CameraType::Main)) {
          const RtCamera& camera = cameraManager.getMainCamera();
          // Revert the main camera's viewToWorld transform and then apply the free camera's one
          transformData.textureTransform *= camera.getViewToWorldToFreeCamViewToWorld();
        } else {
          ONCE(Logger::warn(str::format("[RTX] Tried to update surface transform with Free Camera's transform "
                                        "but main camera has not been processed this frame yet. Skipping the transform update")));
        }
      }

      getSceneManager().submitDrawState(this, drawCallState, overrideMaterialData);
    }
  }

  void RtxContext::commitExternalGeometryToRT(std::unique_ptr<ExternalDrawState> state) {
    getSceneManager().submitExternalDraw(this, std::move(state));
  }

  static uint32_t jenkinsHash(uint32_t a) {
    // http://burtleburtle.net/bob/hash/integer.html
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
  }

  void RtxContext::getDenoiseArgs(NrdArgs& outPrimaryDirectNrdArgs, NrdArgs& outPrimaryIndirectNrdArgs, NrdArgs& outSecondaryNrdArgs) {
    const bool realtimeDenoiserEnabled = RtxOptions::useDenoiser() && !RtxOptions::useDenoiserReferenceMode();
    const bool separateDenoiserEnabled = RtxOptions::denoiseDirectAndIndirectLightingSeparately();

    auto& denoiser0 = (separateDenoiserEnabled ? m_common->metaPrimaryDirectLightDenoiser() : m_common->metaPrimaryCombinedLightDenoiser());
    auto& denoiser1 = (separateDenoiserEnabled ? m_common->metaPrimaryIndirectLightDenoiser() : m_common->metaPrimaryCombinedLightDenoiser());
    auto& denoiser2 = m_common->metaSecondaryCombinedLightDenoiser();

    outPrimaryDirectNrdArgs = denoiser0.getNrdArgs();
    outPrimaryIndirectNrdArgs = denoiser1.getNrdArgs();
    outSecondaryNrdArgs = denoiser2.getNrdArgs();

    // Disable ReBLUR when RR is on because ReBLUR uses a different buffer encoding
    bool useRR = useRayReconstruction();
    if (useRR) {
      outPrimaryDirectNrdArgs.isReblurEnabled = false;
      outPrimaryIndirectNrdArgs.isReblurEnabled = false;
    }
  }

  void RtxContext::updateRaytraceArgsConstantBuffer(Resources::RaytracingOutput& rtOutput,
                                                    const VkExtent3D& downscaledExtent, const VkExtent3D& targetExtent) {
    ScopedCpuProfileZone();
    // Prepare shader arguments
    RaytraceArgs &constants = rtOutput.m_raytraceArgs;
    constants = {}; 

    auto const& camera{ getSceneManager().getCamera() };
    const uint32_t frameIdx = m_device->getCurrentFrameId();

    constants.camera = camera.getShaderConstants();

    // Set the Raytraced Render Target camera matrices
    auto const& renderTargetCamera { getSceneManager().getCameraManager().getCamera(CameraType::RenderToTexture) };
    constants.renderTargetCamera = renderTargetCamera.getShaderConstants(/*freecam =*/ false);
    constants.enableRaytracedRenderTarget = renderTargetCamera.isValid(m_device->getCurrentFrameId()) && RtxOptions::RaytracedRenderTarget::enable();
    const CameraManager& cameraManager = getSceneManager().getCameraManager();

    const bool enablePortalVolumes = RtxGlobalVolumetrics::enableInPortals() &&
      cameraManager.isCameraValid(CameraType::Portal0) &&
      cameraManager.isCameraValid(CameraType::Portal1);
    
    // Note: Ensure the number of lights can fit into the ray tracing args.
    assert(getSceneManager().getLightManager().getActiveCount() <= std::numeric_limits<uint16_t>::max());
    bool useRR = shouldUseRayReconstruction();

    constants.frameIdx = RtxOptions::rngSeedWithFrameIndex() ? m_device->getCurrentFrameId() : 0;
    constants.lightCount = static_cast<uint16_t>(getSceneManager().getLightManager().getActiveCount());

    constants.fireflyFilteringLuminanceThreshold = RtxOptions::fireflyFilteringLuminanceThreshold();
    constants.secondarySpecularFireflyFilteringThreshold = RtxOptions::secondarySpecularFireflyFilteringThreshold();
    constants.primaryRayMaxInteractions = RtxOptions::primaryRayMaxInteractions();
    constants.psrRayMaxInteractions = RtxOptions::psrRayMaxInteractions();
    constants.secondaryRayMaxInteractions = RtxOptions::secondaryRayMaxInteractions();

    // Todo: Potentially move this to the volume manager in the future to be more organized.
    constants.volumeTemporalReuseMaxSampleCount = RtxGlobalVolumetrics::temporalReuseMaxSampleCount();
    
    constants.russianRouletteMode = RtxOptions::russianRouletteMode();
    constants.russianRouletteDiffuseContinueProbability = RtxOptions::russianRouletteDiffuseContinueProbability();
    constants.russianRouletteSpecularContinueProbability = RtxOptions::russianRouletteSpecularContinueProbability();
    constants.russianRouletteDistanceFactor = RtxOptions::russianRouletteDistanceFactor();
    constants.russianRouletteMaxContinueProbability = RtxOptions::russianRouletteMaxContinueProbability();
    constants.russianRoulette1stBounceMinContinueProbability = RtxOptions::russianRoulette1stBounceMinContinueProbability();
    constants.russianRoulette1stBounceMaxContinueProbability = RtxOptions::russianRoulette1stBounceMaxContinueProbability();
    constants.pathMinBounces = RtxOptions::pathMinBounces();
    constants.pathMaxBounces = RtxOptions::pathMaxBounces();
    // Note: Probability adjustments always in the 0-1 range and therefore less than FLOAT16_MAX.
    constants.opaqueDiffuseLobeSamplingProbabilityZeroThreshold =
      glm::packHalf1x16(RtxOptions::opaqueDiffuseLobeSamplingProbabilityZeroThreshold());
    constants.minOpaqueDiffuseLobeSamplingProbability =
      glm::packHalf1x16(RtxOptions::minOpaqueDiffuseLobeSamplingProbability());
    constants.opaqueSpecularLobeSamplingProbabilityZeroThreshold =
      glm::packHalf1x16(RtxOptions::opaqueSpecularLobeSamplingProbabilityZeroThreshold());
    constants.minOpaqueSpecularLobeSamplingProbability =
      glm::packHalf1x16(RtxOptions::minOpaqueSpecularLobeSamplingProbability());
    constants.opaqueOpacityTransmissionLobeSamplingProbabilityZeroThreshold =
      glm::packHalf1x16(RtxOptions::opaqueOpacityTransmissionLobeSamplingProbabilityZeroThreshold());
    constants.minOpaqueOpacityTransmissionLobeSamplingProbability =
      glm::packHalf1x16(RtxOptions::minOpaqueOpacityTransmissionLobeSamplingProbability());
    constants.opaqueDiffuseTransmissionLobeSamplingProbabilityZeroThreshold =
      glm::packHalf1x16(RtxOptions::opaqueDiffuseTransmissionLobeSamplingProbabilityZeroThreshold());
    constants.minOpaqueDiffuseTransmissionLobeSamplingProbability =
      glm::packHalf1x16(RtxOptions::minOpaqueDiffuseTransmissionLobeSamplingProbability());
    constants.translucentSpecularLobeSamplingProbabilityZeroThreshold =
      glm::packHalf1x16(RtxOptions::translucentSpecularLobeSamplingProbabilityZeroThreshold());
    constants.minTranslucentSpecularLobeSamplingProbability =
      glm::packHalf1x16(RtxOptions::minTranslucentSpecularLobeSamplingProbability());
    constants.translucentTransmissionLobeSamplingProbabilityZeroThreshold =
      glm::packHalf1x16(RtxOptions::translucentTransmissionLobeSamplingProbabilityZeroThreshold());
    constants.minTranslucentTransmissionLobeSamplingProbability =
      glm::packHalf1x16(RtxOptions::minTranslucentTransmissionLobeSamplingProbability());
    constants.indirectRaySpreadAngleFactor = RtxOptions::indirectRaySpreadAngleFactor();

    // Note: Emissibe blend override emissive intensity always clamped to FLOAT16_MAX, so this packing is fine.
    constants.emissiveBlendOverrideEmissiveIntensity = glm::packHalf1x16(RtxOptions::emissiveBlendOverrideEmissiveIntensity());
    constants.emissiveIntensity = glm::packHalf1x16(RtxOptions::emissiveIntensity());
    constants.particleSoftnessFactor = glm::packHalf1x16(RtxOptions::particleSoftnessFactor());

    constants.psrrMaxBounces = RtxOptions::psrrMaxBounces();
    constants.pstrMaxBounces = RtxOptions::pstrMaxBounces();

    auto& rtxdi = m_common->metaRtxdiRayQuery();
    constants.enableEmissiveBlendEmissiveOverride = RtxOptions::enableEmissiveBlendEmissiveOverride();
    constants.enableRtxdi = RtxOptions::useRTXDI();
    constants.enableSecondaryBounces = RtxOptions::enableSecondaryBounces();
    constants.enableSeparatedDenoisers = RtxOptions::denoiseDirectAndIndirectLightingSeparately();
    constants.enableCalculateVirtualShadingNormals = RtxOptions::useVirtualShadingNormalsForDenoising();
    constants.enableViewModelVirtualInstances = RtxOptions::ViewModel::enableVirtualInstances();
    constants.enablePSRR = RtxOptions::enablePSRR();
    constants.enablePSTR = RtxOptions::enablePSTR();
    constants.enablePSTROutgoingSplitApproximation = RtxOptions::enablePSTROutgoingSplitApproximation();
    constants.enablePSTRSecondaryIncidentSplitApproximation = RtxOptions::enablePSTRSecondaryIncidentSplitApproximation();
    constants.psrrNormalDetailThreshold = RtxOptions::psrrNormalDetailThreshold();
    constants.pstrNormalDetailThreshold = RtxOptions::pstrNormalDetailThreshold();
    const float meterToWorld = RtxOptions::getMeterToWorldUnitScale();
    constants.psrMaxDistance = RtxOptions::psrMaxDistanceMeters() * meterToWorld;
    constants.psrMaxDistanceFade = std::min(
      RtxOptions::psrMaxDistanceFadeMeters(), RtxOptions::psrMaxDistanceMeters()) * meterToWorld;
    constants.enableDirectLighting = RtxOptions::enableDirectLighting();
    constants.enableStochasticAlphaBlend = m_common->metaComposite().enableStochasticAlphaBlend();
    constants.enableSeparateUnorderedApproximations = RtxOptions::enableSeparateUnorderedApproximations() && getResourceManager().getTLAS(Tlas::Unordered).accelStructure != nullptr;
    constants.enableDirectTranslucentShadows = RtxOptions::enableDirectTranslucentShadows();
    constants.enableDirectAlphaBlendShadows = RtxOptions::enableDirectAlphaBlendShadows();
    constants.enableIndirectTranslucentShadows = RtxOptions::enableIndirectTranslucentShadows();
    constants.enableIndirectAlphaBlendShadows = RtxOptions::enableIndirectAlphaBlendShadows();
    constants.enableRussianRoulette = RtxOptions::enableRussianRoulette();
    constants.enableDemodulateRoughness = m_common->metaDemodulate().demodulateRoughness();
    constants.enableReplaceDirectSpecularHitTWithIndirectSpecularHitT = RtxOptions::replaceDirectSpecularHitTWithIndirectSpecularHitT();
    constants.enablePortalFadeInEffect = RtxOptions::enablePortalFadeInEffect();
    constants.enableEnhanceBSDFDetail = (shouldUseDLSS() || useRR || shouldUseTAA()) && m_common->metaComposite().enableDLSSEnhancement();
    constants.enhanceBSDFIndirectMode = (uint32_t)m_common->metaComposite().dlssEnhancementMode();
    constants.enhanceBSDFDirectLightPower = useRR ? 0.0 : m_common->metaComposite().dlssEnhancementDirectLightPower();
    constants.enhanceBSDFIndirectLightPower = m_common->metaComposite().dlssEnhancementIndirectLightPower();
    constants.enhanceBSDFDirectLightMaxValue = m_common->metaComposite().dlssEnhancementDirectLightMaxValue();
    constants.enhanceBSDFIndirectLightMaxValue = m_common->metaComposite().dlssEnhancementIndirectLightMaxValue();
    constants.enhanceBSDFIndirectLightMinRoughness = m_common->metaComposite().dlssEnhancementIndirectLightMinRoughness();
    constants.enableFirstBounceLobeProbabilityDithering = RtxOptions::enableFirstBounceLobeProbabilityDithering();
    constants.enableUnorderedResolveInIndirectRays = RtxOptions::enableUnorderedResolveInIndirectRays();
    constants.enableProbabilisticUnorderedResolveInIndirectRays = RtxOptions::enableProbabilisticUnorderedResolveInIndirectRays();
    constants.enableTransmissionApproximationInIndirectRays = RtxOptions::enableTransmissionApproximationInIndirectRays();
    constants.enableUnorderedEmissiveParticlesInIndirectRays = RtxOptions::enableUnorderedEmissiveParticlesInIndirectRays();
    constants.enableDecalMaterialBlending = RtxOptions::enableDecalMaterialBlending();
    constants.enableLegacyRectLightConeShaping = LightManager::enableLegacyRectLightConeShaping();
    constants.enableRectLightConeShapingRatioScaling = LightManager::enableRectLightConeShapingRatioScaling();
    constants.enableBillboardOrientationCorrection = RtxOptions::enableBillboardOrientationCorrection() && RtxOptions::enableSeparateUnorderedApproximations();
    constants.useIntersectionBillboardsOnPrimaryRays = RtxOptions::useIntersectionBillboardsOnPrimaryRays() && constants.enableBillboardOrientationCorrection;
    constants.enableDirectLightBoilingFilter = m_common->metaDemodulate().enableDirectLightBoilingFilter() && RtxOptions::useRTXDI();

    auto& sparseRendering = m_common->metaSparseRendering();
    if (constants.enableDirectLightBoilingFilter && sparseRendering.isActive()) {
      // RR path disables direct light boiling filter, but in case someone manually enables it.
      // The filter is group cooperative - it accumulates into groupshared memory across
      // GroupMemoryBarrierWithGroupSync() and zero initializes that memory from a single thread.
      // Sparse rendering runs it on direct active threads only, so part of the group would skip
      // the barriers and the accumulators could stay uninitialized. Supporting it needs every
      // thread in the group to reach the barriers, not just spatial locality of remapped pixels.
      ONCE(Logger::warn("[RTX] Direct Light Boiling Filter is not supported with Sparse Rendering enabled."));
      constants.enableDirectLightBoilingFilter = false;
    }

    constants.directLightBoilingThreshold = m_common->metaDemodulate().directLightBoilingThreshold();
    constants.translucentDecalAlbedoFactor = RtxOptions::translucentDecalAlbedoFactor();
    constants.enablePlayerModelInPrimarySpace =
      getSceneManager().getInstanceManager().isExternalCameraRegime();
    constants.enablePlayerModelPrimaryShadows = RtxOptions::PlayerModel::enablePrimaryShadows();
    constants.enablePreviousTLAS = RtxOptions::enablePreviousTLAS() && m_common->getSceneManager().isPreviousFrameSceneAvailable();

    constants.pomMode = getSceneManager().getActivePOMCount() > 0 ? RtxOptions::Displacement::mode() : DisplacementMode::Off;
    if (constants.pomMode == DisplacementMode::Off) {
      constants.pomEnableDirectLighting = false;
      constants.pomEnableIndirectLighting = false;
      constants.pomEnableNEECache = false;
      constants.pomEnableReSTIRGI = false;
      constants.pomEnablePSR = true; // enable PSR for materials with heightmaps if POM is completely disabled.
    } else {
      constants.pomEnableDirectLighting = RtxOptions::Displacement::enableDirectLighting();
      constants.pomEnableIndirectLighting = RtxOptions::Displacement::enableIndirectLighting();
      constants.pomEnableNEECache = RtxOptions::Displacement::enableNEECache();
      constants.pomEnableReSTIRGI = RtxOptions::Displacement::enableReSTIRGI();
      constants.pomEnablePSR = RtxOptions::Displacement::enablePSR();
    }
    constants.pomMaxIterations = RtxOptions::Displacement::maxIterations();

    constants.totalMipBias = getSceneManager().getTotalMipBias(); 
    constants.hairCardMipBias = RtxOptions::hairCardMipBias();
    constants.hairCardRoughnessScale = RtxOptions::hairCardRoughnessScale();

    constants.upscaleFactor = float2 {
      rtOutput.m_compositeOutputExtent.width / static_cast<float>(rtOutput.m_finalOutputExtent.width),
      rtOutput.m_compositeOutputExtent.height / static_cast<float>(rtOutput.m_finalOutputExtent.height) };

    constants.terrainArgs = getSceneManager().getTerrainBaker().getTerrainArgs();

    constants.sssArgs.enableThinOpaque = RtxOptions::SubsurfaceScattering::enableThinOpaque();
    constants.sssArgs.enableDiffusionProfile = RtxOptions::SubsurfaceScattering::enableDiffusionProfile();
    constants.sssArgs.diffusionProfileScale = std::max(RtxOptions::SubsurfaceScattering::diffusionProfileScale(), 0.001f);
    constants.enableSssTransmission = RtxOptions::SubsurfaceScattering::enableTransmission();
    constants.enableSssTransmissionSingleScattering = RtxOptions::SubsurfaceScattering::enableTransmissionSingleScattering();
    constants.sssTransmissionBsdfSampleCount = RtxOptions::SubsurfaceScattering::transmissionBsdfSampleCount();
    constants.sssTransmissionSingleScatteringSampleCount = RtxOptions::SubsurfaceScattering::transmissionSingleScatteringSampleCount();
    constants.enableTransmissionDiffusionProfileCorrection = RtxOptions::SubsurfaceScattering::enableTransmissionDiffusionProfileCorrection();
    constants.metersToWorldUnitScale = RtxOptions::getMeterToWorldUnitScale();
    constants.enableHeuristicSingleScatteringTransmission = RtxOptions::SubsurfaceScattering::enableHeuristicSingleScatteringTransmission();
    constants.sssArgs.diffusionProfileDebuggingPixel = u16vec2 {
      static_cast<uint16_t>(RtxOptions::SubsurfaceScattering::diffusionProfileDebugPixelPosition().x),
      static_cast<uint16_t>(RtxOptions::SubsurfaceScattering::diffusionProfileDebugPixelPosition().y) };

    auto& restirGI = m_common->metaReSTIRGIRayQuery();
    ReSTIRGISampleStealing restirGISampleStealingMode = restirGI.useSampleStealing();
    // Stealing pixels requires indirect light stored in separated buffers instead of combined with direct light,
    // steal samples if separated denoiser is disabled.
    if (restirGISampleStealingMode == ReSTIRGISampleStealing::StealPixel 
        && !RtxOptions::denoiseDirectAndIndirectLightingSeparately()) {
      restirGISampleStealingMode = ReSTIRGISampleStealing::StealSample;
    }
    constants.enableReSTIRGI = restirGI.isActive();
    constants.enableReSTIRGITemporalReuse = restirGI.useTemporalReuse();
    constants.enableReSTIRGISpatialReuse = restirGI.useSpatialReuse();
    constants.reSTIRGIMISMode = (uint32_t)restirGI.misMode();
    constants.enableReSTIRGIFinalVisibility = restirGI.useFinalVisibility();
    constants.enableReSTIRGIReflectionReprojection = restirGI.useReflectionReprojection();
    constants.restirGIReflectionMinParallax = restirGI.reflectionMinParallax();
    constants.enableReSTIRGIVirtualSample = restirGI.useVirtualSample();
    constants.reSTIRGIMISModePairwiseMISCentralWeight = restirGI.pairwiseMISCentralWeight();
    constants.reSTIRGIVirtualSampleLuminanceThreshold = restirGI.virtualSampleLuminanceThreshold();
    constants.reSTIRGIVirtualSampleRoughnessThreshold = restirGI.virtualSampleRoughnessThreshold();
    constants.reSTIRGIVirtualSampleSpecularThreshold = restirGI.virtualSampleSpecularThreshold();
    constants.reSTIRGIVirtualSampleMaxDistanceRatio = restirGI.virtualSampleMaxDistanceRatio();
    constants.reSTIRGIBiasCorrectionMode = (uint32_t) restirGI.biasCorrectionMode();
    constants.enableReSTIRGIPermutationSampling = restirGI.usePermutationSampling();
    constants.enableReSTIRGISampleStealing = (uint32_t)restirGISampleStealingMode;
    constants.reSTIRGISampleStealingJitter = restirGI.sampleStealingJitter();
    constants.enableReSTIRGIStealBoundaryPixelSamplesWhenOutsideOfScreen = (uint32_t)restirGI.stealBoundaryPixelSamplesWhenOutsideOfScreen();
    constants.enableReSTIRGIBoilingFilter = restirGI.useBoilingFilter();
    constants.boilingFilterLowerThreshold = restirGI.boilingFilterMinThreshold();
    constants.boilingFilterHigherThreshold = restirGI.boilingFilterMaxThreshold();
    constants.boilingFilterRemoveReservoirThreshold = restirGI.boilingFilterRemoveReservoirThreshold();
    constants.temporalHistoryLength = restirGI.getTemporalHistoryLength(GlobalTime::get().deltaTimeMs());
    constants.permutationSamplingSize = restirGI.permutationSamplingSize();
    constants.enableReSTIRGIDLSSRRCompatibilityMode = useRR ? restirGI.useDLSSRRCompatibilityMode() : 0;
    constants.reSTIRGIDLSSRRTemporalRandomizationRadius = constants.camera.resolution.x / 960.0f * restirGI.DLSSRRTemporalRandomizationRadius();
    constants.enableReSTIRGITemporalBiasCorrection = restirGI.useTemporalBiasCorrection();
    constants.enableReSTIRGIDiscardEnlargedPixels = restirGI.useDiscardEnlargedPixels();
    constants.reSTIRGIHistoryDiscardStrength = restirGI.historyDiscardStrength();
    constants.enableReSTIRGITemporalJacobian = restirGI.useTemporalJacobian();
    constants.reSTIRGIFireflyThreshold = restirGI.fireflyThreshold();
    constants.reSTIRGIRoughnessClamp = restirGI.roughnessClamp();
    constants.reSTIRGIMISRoughness = restirGI.misRoughness();
    constants.reSTIRGIMISParallaxAmount = restirGI.parallaxAmount();
    constants.enableReSTIRGIDemodulatedTargetFunction = restirGI.useDemodulatedTargetFunction();
    constants.enableReSTIRGILightingValidation = RtxOptions::useRTXDI() && rtxdi.getEnableDenoiserGradient(*this) && restirGI.validateLightingChange();
    constants.reSTIRGISampleValidationThreshold = restirGI.lightingValidationThreshold();
    constants.enableReSTIRGIVisibilityValidation = restirGI.validateVisibilityChange();
    constants.reSTIRGIVisibilityValidationRange = 1.0f + restirGI.visibilityValidationRange();

    // Neural Radiance Cache
    NeuralRadianceCache& nrc = m_common->metaNeuralRadianceCache();
    constants.enableNrc = nrc.isActive();
    constants.allowNrcTraining = NeuralRadianceCache::NrcOptions::trainCache();
    nrc.setRaytraceArgs(constants);

    m_common->metaNeeCache().setRaytraceArgs(constants, m_resetHistory);
    constants.surfaceCount = getSceneManager().getAccelManager().getSurfaceCount();

    m_common->metaSparseRendering().setSparseRenderingArgs(*this, constants.sparseRenderingArgs);

    auto* cameraTeleportDirectionInfo = getSceneManager().getRayPortalManager().getCameraTeleportationRayPortalDirectionInfo();
    constants.teleportationPortalIndex = cameraTeleportDirectionInfo ? cameraTeleportDirectionInfo->entryPortalInfo.portalIndex + 1 : 0;

    // Note: Use half of the vertical FoV for the main camera in radians divided by the vertical resolution to get the effective half angle of a single pixel.
    constants.screenSpacePixelSpreadHalfAngle = getSceneManager().getCamera().getFov() / 2.0f / constants.camera.resolution.y;

    // Note: This value is assumed to be positive (specifically not have the sign bit set) as otherwise it will break Ray Interaction encoding.
    assert(std::signbit(constants.screenSpacePixelSpreadHalfAngle) == false);

    // Enable object picking only when resource was created
    // TODO: should be a spec.const
    constants.enableObjectPicking = bool { rtOutput.m_primaryObjectPicking.isValid() };

    // Debug View
    {
      const DebugView& debugView = m_common->metaDebugView();
      constants.debugView = debugView.debugViewIdx();
      constants.debugKnob = debugView.debugKnob();
      constants.forceFirstHitInGBufferPass = debugView.showFirstGBufferHit();
      constants.enableDlssNrControlMask =
        m_common->metaDlssNeuralRendering().useDlssNeuralRendering() &&
        !DlssNeuralRendering::useAutoMask();
      constants.enableDlssNrVolumetricControlMask =
        constants.enableDlssNrControlMask &&
        DlssNeuralRendering::enableVolumetricControlMask();

      constants.gpuPrintThreadIndex = u16vec2 { kInvalidThreadIndex, kInvalidThreadIndex };
      constants.gpuPrintElementIndex = frameIdx % kMaxFramesInFlight;

      if (debugView.gpuPrint.enable() && ImGui::IsKeyDown(ImGuiKey_ModCtrl)) {
        if (debugView.gpuPrint.useMousePosition()) {
          Vector2 toDownscaledExtentScale{
            downscaledExtent.width / static_cast<float>(targetExtent.width),
            downscaledExtent.height / static_cast<float>(targetExtent.height)
          };

          const ImVec2 mousePos = ImGui::GetMousePos();
          constants.gpuPrintThreadIndex = u16vec2 {
            static_cast<uint16_t>(mousePos.x * toDownscaledExtentScale.x),
            static_cast<uint16_t>(mousePos.y * toDownscaledExtentScale.y)
          };
        } else {
          constants.gpuPrintThreadIndex = u16vec2 {
            static_cast<uint16_t>(debugView.gpuPrint.pixelIndex().x), 
            static_cast<uint16_t>(debugView.gpuPrint.pixelIndex().y) 
          };
        }
      }
    }

    getDenoiseArgs(constants.primaryDirectNrd, constants.primaryIndirectNrd, constants.secondaryCombinedNrd);

    RayPortalManager::SceneData portalData = getSceneManager().getRayPortalManager().getRayPortalInfoSceneData();
    constants.numActiveRayPortals = portalData.numActiveRayPortals;
    constants.virtualInstancePortalIndex = getSceneManager().getInstanceManager().getVirtualInstancePortalIndex() & 0xff;

    memcpy(&constants.rayPortalHitInfos[0], &portalData.rayPortalHitInfos, sizeof(portalData.rayPortalHitInfos));
    memcpy(&constants.rayPortalHitInfos[maxRayPortalCount], &portalData.previousRayPortalHitInfos, sizeof(portalData.previousRayPortalHitInfos));

    constants.uniformRandomNumber = jenkinsHash(constants.frameIdx);
    constants.vertexColorStrength = RtxOptions::vertexColorStrength();
    constants.viewModelRayTMax = RtxOptions::ViewModel::rangeMeters() * RtxOptions::getMeterToWorldUnitScale();
    constants.roughnessDemodulationOffset = m_common->metaDemodulate().demodulateRoughnessOffset();
    
    const RtxGlobalVolumetrics& globalVolumetrics = getCommonObjects()->metaGlobalVolumetrics();
    constants.volumeArgs = globalVolumetrics.getVolumeArgs(cameraManager, getSceneManager().getFogState(), enablePortalVolumes);
    constants.startInMediumMaterialIndex = getSceneManager().getStartInMediumMaterialIndex();
    OpaqueMaterialOptions::fillShaderParams(constants.opaqueMaterialArgs);
    TranslucentMaterialOptions::fillShaderParams(constants.translucentMaterialArgs);
    ViewDistanceOptions::fillShaderParams(constants.viewDistanceArgs, RtxOptions::getMeterToWorldUnitScale());
    constants.alphaBlendSurfacePackMult = RtxOptions::getMeterToWorldUnitScale();

    // We are going to use this value to perform some animations on GPU, to mitigate precision related issues loop time
    // at the 24 bit boundary (as we use a 8 bit scalar on top of this time which we want to fit into 32 bits without issues,
    // plus we also convert this value to a floating point value at some point as well which has 23 bits of precision).
    // Bitwise and used rather than modulus as well for slightly better performance.
    constants.timeSinceStartSeconds = (static_cast<uint32_t>(GlobalTime::get().absoluteTimeMs()) & ((1U << 24U) - 1U)) / 1000.f;

    m_common->metaRtxdiRayQuery().setRaytraceArgs(rtOutput);
    getSceneManager().getLightManager().setRaytraceArgs(
      constants,
      m_common->metaRtxdiRayQuery().initialSampleCount(),
      RtxGlobalVolumetrics::initialRISSampleCount(),
      RtxOptions::risLightSampleCount());

    constants.resolveTransparencyThreshold = RtxOptions::resolveTransparencyThreshold();
    constants.resolveOpaquenessThreshold = RtxOptions::resolveOpaquenessThreshold();
    constants.resolveStochasticAlphaBlendThreshold = m_common->metaComposite().stochasticAlphaBlendOpacityThreshold();

    constants.skyBrightness = RtxOptions::skyBrightness();
    constants.isLastCompositeOutputValid = restirGI.isActive() && restirGI.getLastCompositeOutput().matchesWriteFrameIdx(frameIdx - 1);
    constants.isZUp = RtxOptions::zUp();
    constants.enableCullingSecondaryRays = RtxOptions::enableCullingInSecondaryRays();
    constants.enableShadowBackfaceSkip = getSceneManager().getAccelManager().hasShadowBackfaceSkipInstances();

    constants.domeLightArgs = getSceneManager().getLightManager().getDomeLightArgs();

    // Ray miss value handling
    constants.clearColorDepth = getSceneManager().getGlobals().clearColorDepth;
    constants.clearColorPicking = getSceneManager().getGlobals().clearColorPicking;
    constants.clearColorNormal = getSceneManager().getGlobals().clearColorNormal;

    // DLSS-RR
    constants.enableDLSSRR = useRR;
    constants.setLogValueForDisocclusionMaskForDLSSRR = DxvkRayReconstruction::enableDisocclusionMaskBlur();
    constants.invalidateHistoryForAnimatedWater = DxvkRayReconstruction::invalidateHistoryForAnimatedWater();

    // The denoising normals are consumed only by NRD. dispatchDenoise reads these back rather than
    // recomputing the condition, so the GBuffer never writes a guide nothing will read.
    {
      // The guides are allocated exactly when NRD is the effective denoiser, and Resources::onFrameBegin has
      // already reconciled that this frame, so the flag is the condition.
      const bool willNrdDenoise = getResourceManager().areNrdDenoisingGuideResourcesAllocated();
      constants.writeSecondaryDenoisingGuides = willNrdDenoise;
      constants.writePrimaryDenoisingNormal = willNrdDenoise;

      // The primary virtual motion vector is not an NRD guide alone - RTXDI temporal reuse and gradients,
      // ReSTIR GI temporal reuse and the worldMotion debug screenshot all read it.
      constants.writePrimaryVirtualMotionVector =
        willNrdDenoise || RtxOptions::useRTXDI() || restirGI.isActive() || RtxOptions::captureDebugImage();
    }

    // Force static-scene primary motion vectors (camera motion still accounted for), avoiding
    // world-space position precision drift on static geometry such as view models.
    constants.forceStaticSceneMotionVectors = RtxOptions::forceStaticSceneMotionVectors();

    NrdArgs primaryDirectNrdArgs;
    NrdArgs primaryIndirectNrdArgs;
    NrdArgs secondaryNrdArgs;
    getDenoiseArgs(primaryDirectNrdArgs, primaryIndirectNrdArgs, secondaryNrdArgs);

    constants.primaryDirectMissLinearViewZ = primaryDirectNrdArgs.missLinearViewZ;

    constants.wboitEnergyLossCompensation = RtxOptions::wboitEnergyLossCompensation();
    constants.wboitDepthWeightTuning = RtxOptions::wboitDepthWeightTuning();
    constants.wboitEnabled = RtxOptions::wboitEnabled();

    constants.eyeArgs.enableEyes = RtxOptions::Eye::enable();
    constants.eyeArgs.normalBendingEyeball = RtxOptions::Eye::eyeballSphereOffset();
    constants.eyeArgs.normalBendingCornea = RtxOptions::Eye::corneaSphereOffset();
    constants.eyeArgs.whitesAlbedoScale = RtxOptions::Eye::eyeWhitesAlbedoScale();
    constants.eyeArgs.irisRadius = RtxOptions::Eye::irisRadius();
    constants.eyeArgs.irisDepth = RtxOptions::Eye::irisDepth();

    constants.shadowTerminatorArgs.soften = RtxOptions::ShadowTerminator::soften();

    // Note: shadow terminator image are allocated/freed based on this RtxOption
    constants.shadowTerminatorArgs.enableOffset = RtxOptions::ShadowTerminator::enableOffset();
    constants.shadowTerminatorArgs.maxArea = std::max(0.f, RtxOptions::ShadowTerminator::maxArea() * RtxOptions::getMeterToWorldUnitScale() * RtxOptions::getMeterToWorldUnitScale());
    constants.shadowTerminatorArgs.maxLength = std::max(0.f, RtxOptions::ShadowTerminator::maxLength() * RtxOptions::getMeterToWorldUnitScale());

    // Upload the constants to the GPU
    {
      Rc<DxvkBuffer> cb = getResourceManager().getConstantsBuffer();

      writeToBuffer(cb, 0, sizeof(constants), &constants);

      m_cmd->trackResource<DxvkAccess::Write>(cb);
    }
  }

  void RtxContext::bindCommonRayTracingResources(const Resources::RaytracingOutput& rtOutput) {
    ScopedCpuProfileZone();

    Rc<DxvkBuffer> constantsBuffer = getResourceManager().getConstantsBuffer();
    Rc<DxvkBuffer> surfaceBuffer = getSceneManager().getSurfaceBuffer();
    Rc<DxvkBuffer> surfaceMappingBuffer = getSceneManager().getSurfaceMappingBuffer();
    Rc<DxvkBuffer> billboardsBuffer = getSceneManager().getBillboardsBuffer();
    Rc<DxvkBuffer> surfaceMaterialBuffer = getSceneManager().getSurfaceMaterialBuffer();
    Rc<DxvkBuffer> surfaceMaterialExtensionBuffer = getSceneManager().getSurfaceMaterialExtensionBuffer();
    Rc<DxvkBuffer> volumeMaterialBuffer = getSceneManager().getVolumeMaterialBuffer();
    Rc<DxvkBuffer> lightBuffer = getSceneManager().getLightManager().getLightBuffer();
    Rc<DxvkBuffer> previousLightBuffer = getSceneManager().getLightManager().getPreviousLightBuffer();
    Rc<DxvkBuffer> lightMappingBuffer = getSceneManager().getLightManager().getLightMappingBuffer();
    Rc<DxvkBuffer> lightIdentityBuffer = getSceneManager().getLightManager().getLightIdentityBuffer();
    Rc<DxvkBuffer> gpuPrintBuffer = getResourceManager().getRaytracingOutput().m_gpuPrintBuffer;
    Rc<DxvkImageView> valueNoiseLut = getResourceManager().getValueNoiseLut(this);
    Rc<DxvkSampler> linearSampler = getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    Rc<DxvkBuffer> samplerFeedbackBuffer = getResourceManager().getRaytracingOutput().m_samplerFeedbackDevice;

    DebugView& debugView = getCommonObjects()->metaDebugView();

    bindAccelerationStructure(BINDING_ACCELERATION_STRUCTURE, getResourceManager().getTLAS(Tlas::Opaque).accelStructure);
    bindAccelerationStructure(BINDING_ACCELERATION_STRUCTURE_PREVIOUS, getResourceManager().getTLAS(Tlas::Opaque).previousAccelStructure.ptr() ? getResourceManager().getTLAS(Tlas::Opaque).previousAccelStructure : getResourceManager().getTLAS(Tlas::Opaque).accelStructure);
    bindAccelerationStructure(BINDING_ACCELERATION_STRUCTURE_UNORDERED, getResourceManager().getTLAS(Tlas::Unordered).accelStructure);
    bindAccelerationStructure(BINDING_ACCELERATION_STRUCTURE_SSS, getResourceManager().getTLAS(Tlas::SSS).accelStructure);
    bindResourceBuffer(BINDING_SURFACE_DATA_BUFFER, DxvkBufferSlice(surfaceBuffer, 0, surfaceBuffer->info().size));
    bindResourceBuffer(BINDING_SURFACE_MAPPING_BUFFER, DxvkBufferSlice(surfaceMappingBuffer, 0, surfaceMappingBuffer.ptr() ? surfaceMappingBuffer->info().size : 0));
    bindResourceBuffer(BINDING_SURFACE_MATERIAL_DATA_BUFFER, DxvkBufferSlice(surfaceMaterialBuffer, 0, surfaceMaterialBuffer->info().size));
    bindResourceBuffer(BINDING_SURFACE_MATERIAL_EXT_DATA_BUFFER, surfaceMaterialExtensionBuffer.ptr() ? DxvkBufferSlice(surfaceMaterialExtensionBuffer, 0, surfaceMaterialExtensionBuffer->info().size) : DxvkBufferSlice());
    bindResourceBuffer(BINDING_VOLUME_MATERIAL_DATA_BUFFER, volumeMaterialBuffer.ptr() ? DxvkBufferSlice(volumeMaterialBuffer, 0, volumeMaterialBuffer->info().size) : DxvkBufferSlice());
    bindResourceBuffer(BINDING_LIGHT_DATA_BUFFER, DxvkBufferSlice(lightBuffer, 0, lightBuffer.ptr() ? lightBuffer->info().size : 0));
    bindResourceBuffer(BINDING_PREVIOUS_LIGHT_DATA_BUFFER, DxvkBufferSlice(previousLightBuffer, 0, previousLightBuffer.ptr() ? previousLightBuffer->info().size : 0));
    bindResourceBuffer(BINDING_LIGHT_MAPPING, DxvkBufferSlice(lightMappingBuffer, 0, lightMappingBuffer.ptr() ? lightMappingBuffer->info().size : 0));
    bindResourceBuffer(BINDING_LIGHT_IDENTITY_BUFFER, lightIdentityBuffer.ptr() ? DxvkBufferSlice(lightIdentityBuffer, 0, lightIdentityBuffer->info().size) : DxvkBufferSlice());
    bindResourceBuffer(BINDING_BILLBOARDS_BUFFER, DxvkBufferSlice(billboardsBuffer, 0, billboardsBuffer.ptr() ? billboardsBuffer->info().size : 0));
    bindResourceView(BINDING_BLUE_NOISE_TEXTURE, getResourceManager().getBlueNoiseTexture(this), nullptr);
    bindResourceBuffer(BINDING_CONSTANTS, DxvkBufferSlice(constantsBuffer, 0, constantsBuffer->info().size));
    bindResourceView(BINDING_DEBUG_VIEW_TEXTURE, debugView.getDebugOutput(), nullptr);
    bindResourceBuffer(BINDING_GPU_PRINT_BUFFER, DxvkBufferSlice(gpuPrintBuffer, 0, gpuPrintBuffer.ptr() ? gpuPrintBuffer->info().size : 0));
    bindResourceView(BINDING_VALUE_NOISE_SAMPLER, valueNoiseLut, nullptr);
    bindResourceSampler(BINDING_VALUE_NOISE_SAMPLER, linearSampler);
    bindResourceBuffer(BINDING_SAMPLER_READBACK_BUFFER, DxvkBufferSlice(samplerFeedbackBuffer, 0, samplerFeedbackBuffer.ptr() ? samplerFeedbackBuffer->info().size : 0));
  }

  void RtxContext::bindResourceView(const uint32_t slot, const Rc<DxvkImageView>& imageView, const Rc<DxvkBufferView>& bufferView)
  {
    DxvkContext::bindResourceView(slot, imageView, bufferView);

#ifdef REMIX_DEVELOPMENT
    // Cache resources for aliasing
    cacheResourceAliasingImageView(imageView);
#endif
  }

  void RtxContext::checkOpacityMicromapSupport() {
    bool isOpacityMicromapSupported = OpacityMicromapManager::checkIsOpacityMicromapSupported(*m_device);

    RtxOptions::setIsOpacityMicromapSupported(isOpacityMicromapSupported);

    Logger::info(str::format("[RTX info] Opacity Micromap: ", isOpacityMicromapSupported ? "supported" : "not supported"));
  }

  bool RtxContext::checkIsShaderExecutionReorderingSupported(DxvkDevice& device) {
    if (!RtxOptions::isShaderExecutionReorderingSupported()) {
      return false;
    }

    // SER Extension support check
    const bool isSERExtensionSupported = device.extensions().nvRayTracingInvocationReorder;
    const bool isSERReorderingEnabled =
      VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_NV == device.properties().nvRayTracingInvocationReorderProperties.rayTracingInvocationReorderReorderingHint;
      
    return isSERExtensionSupported && isSERReorderingEnabled;
  }

  void RtxContext::checkShaderExecutionReorderingSupport() {
    const bool isSERSupported = checkIsShaderExecutionReorderingSupported(*m_device);
    
    RtxOptions::enableShaderExecutionReordering = isSERSupported;

    const VkPhysicalDeviceProperties& props = m_device->adapter()->deviceProperties();
    const NV_GPU_ARCHITECTURE_ID archId = RtxOptions::getNvidiaArch();

    Logger::info(str::format("[RTX info] Shader Execution Reordering: ", isSERSupported ? "supported" : "not supported"));

    bool isShaderExecutionReorderingEnabled = RtxOptions::isShaderExecutionReorderingInPathtracerGbufferEnabled() ||
      RtxOptions::isShaderExecutionReorderingInPathtracerIntegrateIndirectEnabled();

    Logger::info(str::format("[RTX info] Shader Execution Reordering: ", isShaderExecutionReorderingEnabled ? "enabled" : "disabled"));
  }

  void RtxContext::checkNeuralRadianceCacheSupport() {
    // Update RtxOption selection if Neural Radiance Cache was selected but it's not supported
    if (RtxOptions::integrateIndirectMode() == IntegrateIndirectMode::NeuralRadianceCache &&
        !NeuralRadianceCache::checkIsSupported(m_device.ptr())) {

      // Fallback to ReSTIRGI
      Logger::warn(str::format("[RTX] Neural Radiance Cache is not supported. Switching indirect illumination mode to ReSTIR GI."));
      // TODO[REMIX-4105] trying to use NRC for a frame when it isn't supported will cause a crash, so this needs to be setImmediately.
      // Should refactor this to use a separate global for the final state, and indicate user preference with the option.
      // Use Quality layer to ensure this overrides the Environment layer (where env vars are stored).
      RtxOptions::integrateIndirectMode.setImmediately(IntegrateIndirectMode::ReSTIRGI, RtxOptionLayer::getQualityLayer());
    }
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

    // flush the residue
    tryHandleSky(nullptr, nullptr);

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

    getCommonObjects()->metaGeometryUtils().flushCommandList();
  }

  void RtxContext::updateComputeShaderResources() {
    ScopedCpuProfileZone();
    DxvkContext::updateComputeShaderResources();

    auto&& layout = m_state.cp.pipeline->layout();
    if (layout->requiresExtraDescriptorSet()) {
      m_cmd->cmdBindDescriptorSet(VK_PIPELINE_BIND_POINT_COMPUTE, layout->pipelineLayout(), 
                                  getSceneManager().getBindlessResourceManager().getGlobalBindlessTableSet(BindlessResourceManager::Textures),
                                  BINDING_SET_BINDLESS_TEXTURE2D);

      m_cmd->cmdBindDescriptorSet(VK_PIPELINE_BIND_POINT_COMPUTE, layout->pipelineLayout(), 
                                  getSceneManager().getBindlessResourceManager().getGlobalBindlessTableSet(BindlessResourceManager::Buffers),
                                  BINDING_SET_BINDLESS_RAW_BUFFER);

      m_cmd->cmdBindDescriptorSet(VK_PIPELINE_BIND_POINT_COMPUTE, layout->pipelineLayout(),
                                  getSceneManager().getBindlessResourceManager().getGlobalBindlessTableSet(BindlessResourceManager::Samplers),
                                  BINDING_SET_BINDLESS_SAMPLER);
    }
  }

  void RtxContext::updateRaytracingShaderResources() {
    ScopedCpuProfileZone();
    DxvkContext::updateRaytracingShaderResources();

    auto&& layout = m_state.rp.pipeline->layout();
    if (layout->requiresExtraDescriptorSet()) {
      m_cmd->cmdBindDescriptorSet(VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, layout->pipelineLayout(), 
                                  getSceneManager().getBindlessResourceManager().getGlobalBindlessTableSet(BindlessResourceManager::Textures),
                                  BINDING_SET_BINDLESS_TEXTURE2D);

      m_cmd->cmdBindDescriptorSet(VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, layout->pipelineLayout(), 
                                  getSceneManager().getBindlessResourceManager().getGlobalBindlessTableSet(BindlessResourceManager::Buffers),
                                  BINDING_SET_BINDLESS_RAW_BUFFER);

      m_cmd->cmdBindDescriptorSet(VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, layout->pipelineLayout(), 
                                  getSceneManager().getBindlessResourceManager().getGlobalBindlessTableSet(BindlessResourceManager::Samplers),
                                  BINDING_SET_BINDLESS_SAMPLER);
    }
  }

  bool RtxContext::shouldUseDLSS() const {
    // Note: m_dlssSupported only checks for the presence of some basic extensions, the actual DLSS context needs to be queried to see
    // if a given platform supports DLSS (as this will depend on if it was actually initialized successfully or not). Cases where m_dlssSupported
    // is true but supportsDLSS() is not are for example when the DLSS DLL is missing.
    return RtxOptions::isDLSSEnabled() && m_dlssSupported && m_common->metaDLSS().supportsDLSS();
  }

  bool RtxContext::shouldUseRayReconstruction() const {
    return useRayReconstruction();
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

  D3D9RtxVertexCaptureData& RtxContext::allocAndMapVertexCaptureConstantBuffer() {
    DxvkBufferSliceHandle slice = m_rtState.vertexCaptureCB->allocSlice();
    invalidateBuffer(m_rtState.vertexCaptureCB, slice);

    return *static_cast<D3D9RtxVertexCaptureData*>(slice.mapPtr);
  }

  D3D9FixedFunctionVS& RtxContext::allocAndMapFixedFunctionVSConstantBuffer() {
    DxvkBufferSliceHandle slice = m_rtState.vsFixedFunctionCB->allocSlice();
    invalidateBuffer(m_rtState.vsFixedFunctionCB, slice);

    return *static_cast<D3D9FixedFunctionVS*>(slice.mapPtr);
  }
  D3D9SharedPS& RtxContext::allocAndMapPSSharedStateConstantBuffer() {
    DxvkBufferSliceHandle slice = m_rtState.psSharedStateCB->allocSlice();
    invalidateBuffer(m_rtState.psSharedStateCB, slice);

    return *static_cast<D3D9SharedPS*>(slice.mapPtr);
  }

  void RtxContext::rasterizeToSkyMatte(const DrawParameters& params, const DrawCallState& drawCallState) {
    ScopedGpuProfileZone(this, "rasterizeToSkyMatte");

    const RtCamera& camera = getSceneManager().getCamera();
    const uint32_t* renderResolution = camera.m_renderResolution;

    union UnifiedCB {
      D3D9RtxVertexCaptureData programmablePipeline;
      D3D9FixedFunctionVS fixedFunction;

      UnifiedCB() { }
    };

    UnifiedCB prevCB;

    if (drawCallState.usesVertexShader) {
      prevCB.programmablePipeline = *static_cast<D3D9RtxVertexCaptureData*>(m_rtState.vertexCaptureCB->mapPtr(0));
    } else {
      prevCB.fixedFunction = *static_cast<D3D9FixedFunctionVS*>(m_rtState.vsFixedFunctionCB->mapPtr(0));
    }

    auto skyMatteView = getResourceManager().getSkyMatte(this, m_skyColorFormat).view;
    const auto skyMatteExt = skyMatteView->mipLevelExtent(0);

    // Update spec constants
    int prevClipSpaceJitterEnabled = -1;
    {
      if (drawCallState.usesVertexShader) {
        prevClipSpaceJitterEnabled = getSpecConstantsInfo(VK_PIPELINE_BIND_POINT_GRAPHICS)
          .specConstants[D3D9SpecConstantId::ClipSpaceJitterEnabled]
            ? 1
            : 0;
        // Enable, to use clipSpaceJitter, see notes below
        setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::ClipSpaceJitterEnabled, true);
      }
    }

    {
      VkViewport viewport { 0.5f, static_cast<float>(skyMatteExt.height) + 0.5f,
       static_cast<float>(skyMatteExt.width),
       -static_cast<float>(skyMatteExt.height),
       drawCallState.minZ,
       drawCallState.maxZ
      };
      VkRect2D scissor {
        { 0, 0 },
        { skyMatteExt.width, skyMatteExt.height }
      };
      setViewports(1, &viewport, &scissor);
    }

    if (drawCallState.usesVertexShader) {
      D3D9RtxVertexCaptureData modified = prevCB.programmablePipeline;
      {
        // Jittered clip space for DLSS
        // Note: we can't jitter the projection matrix, as a game might calculate
        // its gl_Position by different methods (e.g. without projection matrix at all);
        // so apply jitter directly on gl_Position
        float ratioX = Sign(drawCallState.getTransformData().viewToProjection[2][3]);
        float ratioY = -Sign(drawCallState.getTransformData().viewToProjection[2][3]);
        Vector2 clipSpaceJitter = camera.calcClipSpaceJitter(camera.calcPixelJitter(m_device->getCurrentFrameId()), ratioX, ratioY);
        modified.jitterX = clipSpaceJitter.x;
        modified.jitterY = clipSpaceJitter.y;
      }

      // Ensure that memcpy can be used for fewer memory interactions
      static_assert(std::is_trivially_copyable_v<D3D9RtxVertexCaptureData>);
      allocAndMapVertexCaptureConstantBuffer() = modified;
    } else {
      D3D9FixedFunctionVS modified = prevCB.fixedFunction;
      {
        // Jittered projection for DLSS
        camera.applyJitterTo(modified.Projection,
                             m_device->getCurrentFrameId());
      }
      // Ensure that memcpy can be used for fewer memory interactions
      static_assert(std::is_trivially_copyable_v<D3D9FixedFunctionVS>);
      allocAndMapFixedFunctionVSConstantBuffer() = modified;
    }

    DxvkRenderTargets skyRt;
    skyRt.color[0].view = getResourceManager().getCompatibleViewForView(skyMatteView, m_skyRtColorFormat);
    skyRt.color[0].layout = VK_IMAGE_LAYOUT_GENERAL;
    bindRenderTargets(skyRt);

    if (m_skyClearDirty) {
      DxvkContext::clearRenderTarget(skyMatteView, VK_IMAGE_ASPECT_COLOR_BIT, m_skyClearValue);
    }

    if (params.indexCount == 0) {
      DxvkContext::draw(params.vertexCount, params.instanceCount, params.vertexOffset, 0);
    } else {
      DxvkContext::drawIndexed(params.indexCount, params.instanceCount, params.firstIndex, params.vertexOffset, 0);
    }

    // Restore state
    if (prevClipSpaceJitterEnabled >= 0) {
      assert(prevClipSpaceJitterEnabled == 0 || prevClipSpaceJitterEnabled == 1);
      setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::ClipSpaceJitterEnabled, prevClipSpaceJitterEnabled);
    }
    if (drawCallState.usesVertexShader) {
      allocAndMapVertexCaptureConstantBuffer() = prevCB.programmablePipeline;
    } else {
      allocAndMapFixedFunctionVSConstantBuffer() = prevCB.fixedFunction;
    }
  }

  void RtxContext::initSkyProbe() {
    auto skyProbeImage = getResourceManager().getSkyProbe(this, m_skyColorFormat).image;

    if (m_skyProbeImage == skyProbeImage)
      return;

    m_skyProbeImage = skyProbeImage;

    DxvkImageViewCreateInfo viewInfo;
    viewInfo.type = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = m_skyRtColorFormat;
    viewInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.minLevel = 0;
    viewInfo.numLevels = 1;
    viewInfo.minLayer = 0;
    viewInfo.numLayers = 1;

    for (uint32_t n = 0; n < 6; n++) {
      viewInfo.minLayer = n;
      m_skyProbeCubePlanes[n] = m_device->createImageView(m_skyProbeImage, viewInfo);
    }
  }

  void RtxContext::rasterizeToSkyProbe(const DrawParameters& params, const DrawCallState& drawCallState) {
    ScopedGpuProfileZone(this, "rasterizeToSkyProbe");

    // Lazy init
    initSkyProbe();

    // Grab transforms
    
    union UnifiedCB {
      D3D9RtxVertexCaptureData programmablePipeline;
      D3D9FixedFunctionVS fixedFunction;

      UnifiedCB() { }
    };

    UnifiedCB prevCB;

    if (drawCallState.usesVertexShader) {
      prevCB.programmablePipeline = *static_cast<D3D9RtxVertexCaptureData*>(m_rtState.vertexCaptureCB->mapPtr(0));
    } else {
      prevCB.fixedFunction = *static_cast<D3D9FixedFunctionVS*>(m_rtState.vsFixedFunctionCB->mapPtr(0));
    }

    const Matrix4& worldToView = drawCallState.usesVertexShader ? drawCallState.getTransformData().worldToView : prevCB.fixedFunction.View;
    const Matrix4& viewToProj  = drawCallState.usesVertexShader ? drawCallState.getTransformData().viewToProjection : prevCB.fixedFunction.Projection;

    // Figure out camera position
    const Vector3 camPos = inverse(worldToView).data[3].xyz();

    const DxvkRsInfo &ri = m_state.gp.state.rs;

    DxvkRasterizerState prevRasterizerState {};
    {
      DxvkRasterizerState newRs;
      {
        newRs.depthClipEnable = ri.depthClipEnable();
        newRs.depthBiasEnable = ri.depthBiasEnable();
        newRs.polygonMode = ri.polygonMode();
        newRs.cullMode = ri.cullMode();
        newRs.frontFace = ri.frontFace();
        newRs.sampleCount = ri.sampleCount();
        newRs.conservativeMode = ri.conservativeMode();
      }
      prevRasterizerState = newRs;

      // Set cull mode to none
      newRs.cullMode = VK_CULL_MODE_NONE;
      setRasterizerState(newRs);
    }


    // Update spec constants
    int prevCustomVertexTransformEnabled = -1;
    {
      if (drawCallState.usesVertexShader) {
        prevCustomVertexTransformEnabled = getSpecConstantsInfo(VK_PIPELINE_BIND_POINT_GRAPHICS)
          .specConstants[D3D9SpecConstantId::CustomVertexTransformEnabled]
            ? 1
            : 0;
        setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::CustomVertexTransformEnabled, true);
      }
    }

    const auto& skyProbeExt = m_skyProbeImage->info().extent;

    VkViewport viewport { 0, static_cast<float>(skyProbeExt.height),
      static_cast<float>(skyProbeExt.width),
      -static_cast<float>(skyProbeExt.height),
      0.f, 1.f
    };

    VkRect2D scissor {
      { 0, 0 },
      { skyProbeExt.width, skyProbeExt.height }
    };

    setViewports(1, &viewport, &scissor);

    // Go over sky probe views and rasterize to each plane.
    // NOTE: Ideally sky probe should be rendered in single pass using
    // multiple views, however this would require multiview support
    // plumbing to dxvk side.
    // TODO: add multiview rendering in future.
    for (uint32_t plane = 0; plane < 6; plane++) {
      Rc<DxvkImageView>* skyRenderTarget = &m_skyProbeCubePlanes[plane];

      if (drawCallState.usesVertexShader) {
        D3D9RtxVertexCaptureData& newState = allocAndMapVertexCaptureConstantBuffer();
        newState = prevCB.programmablePipeline;

        // Create cube plane projection
        Matrix4 proj = viewToProj;
        proj[0][0] = 1.f;
        proj[1][1] = 1.f;
        proj[2][2] = 1.f;
        proj[2][3] = 1.f;

        newState.customWorldToProjection = proj * makeViewMatrixForCubePlane(plane, camPos);
      } else {
        // Push new state to the fixed function constants
        D3D9FixedFunctionVS& newState = allocAndMapFixedFunctionVSConstantBuffer();
        newState = prevCB.fixedFunction;
        const Matrix4 view = makeViewMatrixForCubePlane(plane, camPos);

        // Set to identity, as we use custom matrices that transform from world to cube side projection
        newState.View      = view;
        newState.WorldView = view * prevCB.fixedFunction.World;
        // And cube plane projection
        newState.Projection[0][0] = 1.f;
        newState.Projection[1][1] = 1.f;
        newState.Projection[2][2] = 1.f;
        newState.Projection[2][3] = 1.f;
      }

      DxvkRenderTargets skyRt;
      skyRt.color[0].view   = *skyRenderTarget;
      skyRt.color[0].layout = VK_IMAGE_LAYOUT_GENERAL;

      bindRenderTargets(skyRt);

      if (m_skyClearDirty) {
        DxvkContext::clearRenderTarget(*skyRenderTarget, VK_IMAGE_ASPECT_COLOR_BIT, m_skyClearValue);
      }

      if (params.indexCount > 0) {
        DxvkContext::drawIndexed(params.indexCount, params.instanceCount, params.firstIndex, params.vertexOffset, 0);
      } else {
        DxvkContext::draw(params.vertexCount, params.instanceCount, params.vertexOffset, 0);
      }
    }

    // Restore state
    setRasterizerState(prevRasterizerState);
    if (prevCustomVertexTransformEnabled >= 0) {
      assert(prevCustomVertexTransformEnabled == 0 || prevCustomVertexTransformEnabled == 1);
      setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::CustomVertexTransformEnabled, prevCustomVertexTransformEnabled);
    }
    if (drawCallState.usesVertexShader) {
      allocAndMapVertexCaptureConstantBuffer() = prevCB.programmablePipeline;
    } else {
      allocAndMapFixedFunctionVSConstantBuffer() = prevCB.fixedFunction;
    }
  }

  void RtxContext::bakeTerrain(const DrawParameters& params, DrawCallState& drawCallState, const MaterialData** outOverrideMaterialData) {
    if (!getSceneManager().getTerrainBaker().enableBaking() ||
        !drawCallState.testCategoryFlags(InstanceCategories::Terrain)) {
      return;
    }

    DrawCallTransforms& transformData = drawCallState.modifyTransformData();

    // Terrain Baker (may) update bound color textures, so preserve the views
    Rc<DxvkImageView> previousColorView;
    Rc<DxvkImageView> previousSecondaryColorView;

    OpaqueMaterialData* opaqueReplacementMaterial = nullptr;
    TerrainBaker& terrainBaker = getSceneManager().getTerrainBaker();

    if (!TerrainBaker::debugDisableBaking()) {

      // Retrieve the replacement material
      std::shared_ptr<MaterialData> replacementMaterial = getSceneManager().getAssetReplacer()->getReplacementMaterial(drawCallState.getMaterialData().getHash());

      if (replacementMaterial) {
        if (replacementMaterial->getType() == MaterialDataType::Opaque) {
          opaqueReplacementMaterial = &replacementMaterial->getOpaqueMaterialData();

          // Original 0th colour texture slot
          const uint32_t colorTextureSlot = drawCallState.getMaterialData().colorTextureSlot[0];

          // Save current color texture first
          if (colorTextureSlot < m_rc.size() && m_rc[colorTextureSlot].imageView != nullptr) {
            previousColorView = m_rc[colorTextureSlot].imageView;
          }          
          
        } else {
          ONCE(Logger::warn(str::format("[RTX Texture Baker] Only opaque replacement materials are supported for terrain baking. Texture hash ",
                                        drawCallState.getMaterialData().getHash(),
                                        " has a non-opaque replacement material set. Baking the texture with legacy material instead.")));
        }
      }
    }

    // Bake the material
    const bool isBaked = terrainBaker.bakeDrawCall(this, m_state, m_rtState, params, drawCallState, opaqueReplacementMaterial, transformData.textureTransform);

    if (isBaked) {
      // Bind the baked terrain texture to the mesh
      if (!TerrainBaker::debugDisableBinding()) {

        // Set the terrain's baked material data
        *outOverrideMaterialData = terrainBaker.getMaterialData();

        // Generate texcoords in the RT shader
        transformData.texgenMode = TexGenMode::CascadedViewPositions;

        // Update the legacy material data with legacy value defaults as well as set the color textur since some of its data 
        // is still used through the Rt pipeline even though overrideMaterialData is specifide. 
        // Also SceneManager uses sampler associated with the color texture to patch samplers for the textures in the opaque material.
        LegacyMaterialData overrideMaterial;
        overrideMaterial.colorTextures[0] = (*outOverrideMaterialData)->getOpaqueMaterialData().getAlbedoOpacityTexture();
        overrideMaterial.samplers[0] = terrainBaker.getTerrainSampler();
        overrideMaterial.updateCachedHash();
        drawCallState.modifyMaterialData() = overrideMaterial;
      }

      // Restore state modified during baking
      if (!TerrainBaker::debugDisableBaking()) {

        // Restore bound color texture views
        if (previousColorView != nullptr) {
          bindResourceView(drawCallState.getMaterialData().colorTextureSlot[0], previousColorView, nullptr);
        }
      }
    }
  }

  void RtxContext::rasterizeSky(const DrawParameters& params, const DrawCallState& drawCallState) {
    // Grab and apply replacement texture if any
    // NOTE: only the original color texture will be replaced with albedo-opacity texture
    std::shared_ptr<MaterialData> replacementMaterial = getSceneManager().getAssetReplacer()->getReplacementMaterial(drawCallState.getMaterialData().getHash());
    bool replacemenIsLDR = false;
    Rc<DxvkImageView> replacementTexture = {};
    uint32_t replacementTextureSlot = UINT32_MAX;

    if (replacementMaterial && replacementMaterial->getType() == MaterialDataType::Opaque) {
      // Must pull a ref because we will modify it for loading purposes below.
      TextureRef& albedoOpacity = replacementMaterial->getOpaqueMaterialData().getAlbedoOpacityTexture();

      if (albedoOpacity.isValid()) {
        uint32_t textureIndex;
        getSceneManager().trackTexture(albedoOpacity, textureIndex, true, false);

        if (!albedoOpacity.isImageEmpty()) {
          replacementTextureSlot = drawCallState.getMaterialData().colorTextureSlot[0];
          replacementTexture = albedoOpacity.getImageView();
          replacemenIsLDR = TextureUtils::isLDR(albedoOpacity.getImageView()->info().format);
        } else {
          ONCE(Logger::warn("A replacement texture for sky was specified, but it could not be loaded."));
        }
      }
    }
    
    Rc<DxvkImageView> curColorView = {};
    if (replacementTextureSlot < m_rc.size())
    {
      if (m_rc[replacementTextureSlot].imageView != nullptr && replacementTexture != nullptr) {
        // Save currently bound texture to restore later
        curColorView = m_rc[replacementTextureSlot].imageView;
        // Bind a replacement texture
        bindResourceView(replacementTextureSlot, replacementTexture, nullptr);
      }
    }

    // Save current RTs
    DxvkRenderTargets curRts = m_state.om.renderTargets;

    if (!TextureUtils::isLDR(m_skyRtColorFormat) && (!replacementMaterial || replacemenIsLDR)) {
      ONCE(Logger::warn("Sky may not appear correct: sky intermediate format has been forced to HDR "
                        "while the original sky is LDR and no HDR sky replacement has been found!"));
    }

    // Save viewports
    const uint32_t curViewportCount = m_state.gp.state.rs.viewportCount();
    const DxvkViewportState curVp = m_state.vp;

    rasterizeToSkyMatte(params, drawCallState);
    rasterizeToSkyProbe(params, drawCallState);

    m_skyClearDirty = false;

    // Restore VPs
    setViewports(curViewportCount, curVp.viewports.data(), curVp.scissorRects.data());

    // Restore RTs
    bindRenderTargets(curRts);

    // Restore color texture
    if (curColorView != nullptr) {
      bindResourceView(drawCallState.getMaterialData().colorTextureSlot[0], curColorView, nullptr);
    }
  }

  void RtxContext::clearRenderTarget(const Rc<DxvkImageView>& imageView,
                                     VkImageAspectFlags clearAspects, VkClearValue clearValue) {
    // Capture color for skybox clear
    if (clearAspects & VK_IMAGE_ASPECT_COLOR_BIT) {
      m_skyClearValue = clearValue;

      // Set dirty flag so that next skyprobe rasterize will clear the views.
      // We assume that skybox drawcalls will immediately follow the clear. The logic would
      // need to be revisited if this is not true for some game.
      m_skyClearDirty = true;
    }

    DxvkContext::clearRenderTarget(imageView, clearAspects, clearValue);
  }

  void RtxContext::clearImageView(const Rc<DxvkImageView>& imageView, VkOffset3D offset,
                                  VkExtent3D extent, VkImageAspectFlags aspect, VkClearValue value) {
    // Capture color for skybox clear
    if (aspect & VK_IMAGE_ASPECT_COLOR_BIT) {
      m_skyClearValue = value;

      // Set dirty flag so that next skyprobe rasterize will clear the views.
      // We assume that skybox drawcalls will immediately follow the clear. The logic would
      // need to be revisited if this is not true for some game.
      m_skyClearDirty = true;
    }

    DxvkContext::clearImageView(imageView, offset, extent, aspect, value);
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

  const DxvkScInfo& RtxContext::getSpecConstantsInfo(VkPipelineBindPoint pipeline) const {
    return
      pipeline == VK_PIPELINE_BIND_POINT_GRAPHICS
      ? m_state.gp.state.sc
      : pipeline == VK_PIPELINE_BIND_POINT_COMPUTE
      ? m_state.cp.state.sc
      : m_state.rp.state.sc;
  }

  void RtxContext::setSpecConstantsInfo(
    VkPipelineBindPoint pipeline,
    const DxvkScInfo& newSpecConstantInfo) {
    DxvkScInfo& specConstantInfo =
      pipeline == VK_PIPELINE_BIND_POINT_GRAPHICS
      ? m_state.gp.state.sc
      : pipeline == VK_PIPELINE_BIND_POINT_COMPUTE
      ? m_state.cp.state.sc
      : m_state.rp.state.sc;

    if (specConstantInfo != newSpecConstantInfo) {
      specConstantInfo = newSpecConstantInfo;

      m_flags.set(
        pipeline == VK_PIPELINE_BIND_POINT_GRAPHICS
        ? DxvkContextFlag::GpDirtyPipelineState
        : pipeline == VK_PIPELINE_BIND_POINT_COMPUTE
          ? DxvkContextFlag::CpDirtyPipelineState
          : DxvkContextFlag::RpDirtyPipelineState);
    }
  }

#ifdef REMIX_DEVELOPMENT
  void RtxContext::cacheResourceAliasingImageView(const Rc<DxvkImageView>& imageView) {
    if (imageView.ptr()) {
      // Determine the format compatibility category for the image view
      const auto formatCategory = Resources::getFormatCompatibilityCategory(imageView->info().format);
      const auto categoryIndex = static_cast<uint32_t>(formatCategory);
      const auto& underlyingImage = imageView->image();

      // Proceed only if the category is valid and the image view is tracked in the resource view map
      if (formatCategory != RtxTextureFormatCompatibilityCategory::InvalidFormatCompatibilityCategory &&
          Resources::s_resourcesViewMap.find(imageView.ptr()) != Resources::s_resourcesViewMap.end()) {
        bool aliasingMatchFound = false;
        // Search the cache for an existing aliased resource with the same underlying image
        for (auto& compatibleResource : m_resourceCacheTable[categoryIndex]) {
          if (compatibleResource.view->image() == underlyingImage) {
            // Match found: update the begin and end pass stages to expand their range
            compatibleResource.beginPassStage = std::min(compatibleResource.beginPassStage, m_currentPassStage);
            compatibleResource.endPassStage = std::max(compatibleResource.endPassStage, m_currentPassStage);

            // Add the current resource name to the set of names for this aliased group
            compatibleResource.names.insert(Resources::s_resourcesViewMap[imageView.ptr()]);
            aliasingMatchFound = true;
            break;
          }
        }

        if (!aliasingMatchFound) {
          // No match found: cache this as a new aliased resource entry
          m_resourceCacheTable[categoryIndex].push_back({ imageView, m_currentPassStage, m_currentPassStage, { Resources::s_resourcesViewMap[imageView.ptr()] } });
        }
      }
    }
  }

  void RtxContext::queryAvailableResourceAliasing() {
    // Check if aliasing query is enabled through user settings
    if (!Resources::s_queryAliasing) {
      return;
    }

    // Set the aliasing resource dimensions based on user options
    const VkExtent3D extent = { RtxOptions::Aliasing::width(), RtxOptions::Aliasing::height(), RtxOptions::Aliasing::depth() };
    // Get the start and end frame pass stages from user settings
    const RtxFramePassStage beginPass = RtxOptions::Aliasing::beginPass();
    const RtxFramePassStage endPass = RtxOptions::Aliasing::endPass();

    std::string newResourceAliasingQueryResult;
    // Check if the begin pass is before the end pass
    if (beginPass > endPass) {
      // Set an error message if the begin pass is invalid
      Resources::s_resourceAliasingQueryText = "Begin Pass must be before the End Pass";
      return;
    }

    // Lambda function to check if a resource matches the aliasing criteria
    auto isResourceMatches = [&](const Rc<DxvkImageView>& view) {
      const auto& imageInfo = view->image()->info();
      const auto& viewInfo = view->info();
      uint32_t aliasingWidth = RtxOptions::Aliasing::width();
      uint32_t aliasingHeight = RtxOptions::Aliasing::height();

      // Adjust dimensions if the aliasing extent type is DownScaledExtent or TargetExtent
      if (RtxOptions::Aliasing::extentType() == RtxTextureExtentType::DownScaledExtent) {
        aliasingWidth = getResourceManager().getDownscaleDimensions().width;
        aliasingHeight = getResourceManager().getDownscaleDimensions().height;
      } else if (RtxOptions::Aliasing::extentType() == RtxTextureExtentType::TargetExtent) {
        aliasingWidth = getResourceManager().getTargetDimensions().width;
        aliasingHeight = getResourceManager().getTargetDimensions().height;
      }

      // Check if the resource's dimensions and format match the aliasing query settings
      return imageInfo.extent.width == aliasingWidth &&
              imageInfo.extent.height == aliasingHeight &&
              imageInfo.extent.depth == RtxOptions::Aliasing::depth() &&
              imageInfo.numLayers == RtxOptions::Aliasing::layer() &&
              imageInfo.type == RtxOptions::Aliasing::imageType() &&
              viewInfo.type == RtxOptions::Aliasing::imageViewType();
    };

    std::string manualSolveResources;
    uint32_t matchedIndex = 0;

    bool aliasingMatchFound = false;
    const auto category = RtxOptions::Aliasing::formatCategory();
    if (category == RtxTextureFormatCompatibilityCategory::InvalidFormatCompatibilityCategory) {
      // If the format category is invalid, no aliasing can be done
      Resources::s_resourceAliasingQueryText = "Please select aliasing compatible texture format.";
      return;
    }

    // Map category to index for cache lookup
    const uint32_t index = static_cast<uint32_t>(category);

    // Loop through the resource cache table for the corresponding format category
    for (auto& compatibleResource : m_resourceCacheTable[index]) {
      // Check if the resource is compatible with the aliasing query (based on pass stages and matching criteria)
      if ((endPass < compatibleResource.beginPassStage || beginPass > compatibleResource.endPassStage ||
            (beginPass != endPass && compatibleResource.beginPassStage != compatibleResource.endPassStage &&
            (endPass == compatibleResource.beginPassStage || beginPass == compatibleResource.endPassStage))) &&
          (isResourceMatches(compatibleResource.view))) {

        // Loop through names of matching resources and prepare result string
        for (const auto& name : compatibleResource.names) {
          if (Resources::s_dynamicAliasingResourcesSet.find(compatibleResource.view.ptr()) == Resources::s_dynamicAliasingResourcesSet.end()) {
            ++matchedIndex;
            newResourceAliasingQueryResult += std::to_string(matchedIndex) + ". " + name + "\n";
            if (matchedIndex > 10) {
              break; // Limit to 10 results
            }
          } else {
            if (manualSolveResources.empty()) {
              // Give notification for users who want to do aliasing for dynamic resources
              manualSolveResources = "[WARNING] Use caution when aliasing dynamic resources. Ensure aliasing is handled every frame in Resources::onFrameBegin.\n";
            }
            manualSolveResources += name + "\n";
          }
        }
        aliasingMatchFound = true;
      }
    }

    // Set the result of the aliasing query, either showing available resources or a no-match message
    Resources::s_resourceAliasingQueryText =
      aliasingMatchFound ? (newResourceAliasingQueryResult + manualSolveResources) : "No available resources that can be aliased, please create a new resource.";
  }

  void RtxContext::clearResourceAliasingCache() {
    // Clean up caches
    for (auto& resourceCaches : m_resourceCacheTable) {
      resourceCaches.clear();
    }
    m_currentPassStage = RtxFramePassStage::FrameEnd;
  }

  void RtxContext::analyzeResourceAliasing() {
    // Early exit if the aliasing analyzer option is not enabled
    if (!Resources::s_startAliasingAnalyzer) {
      return;
    }

    // Lambda to check if two image views are compatible for aliasing
    auto isResourceCompatible = [](const Rc<DxvkImageView>& view, const Rc<DxvkImageView>& matchedView) {
      const auto& imageInfo = view->image()->info();
      const auto& matchedImageInfo = matchedView->image()->info();
      const auto& viewInfo = view->info();
      return imageInfo.extent == matchedImageInfo.extent &&
             imageInfo.numLayers == matchedImageInfo.numLayers &&
             imageInfo.type == matchedImageInfo.type;
    };

    std::string availableAliasingText;

    // Iterate over all format compatibility categories
    for (uint32_t index = 0; index < static_cast<uint32_t>(RtxTextureFormatCompatibilityCategory::Count); ++index) {
      const std::vector<ResourceCache>& cacheList = m_resourceCacheTable[index];

      // Compare each pair of resources within the same format category
      for (size_t i = 0; i < cacheList.size(); ++i) {
        for (size_t j = i + 1; j < cacheList.size(); ++j) {
          // Check for non-overlapping lifetimes (safe for aliasing)
          if ((cacheList[i].endPassStage < cacheList[j].beginPassStage || cacheList[i].beginPassStage > cacheList[j].endPassStage ||
               (cacheList[i].beginPassStage != cacheList[i].endPassStage && cacheList[j].beginPassStage != cacheList[j].endPassStage &&
                (cacheList[i].endPassStage == cacheList[j].beginPassStage || cacheList[i].beginPassStage == cacheList[j].endPassStage))) &&
              isResourceCompatible(cacheList[i].view, cacheList[j].view)) {
            // Add the resource names to the output text
            availableAliasingText += *cacheList[i].names.begin() + " <-> " + *cacheList[j].names.begin() + "\n";
          }
        }
      }
    }

    // Output the results to the GUI field
    Resources::s_aliasingAnalyzerResultText = !availableAliasingText.empty() ? availableAliasingText : "Can't find any resources that can be aliased.\n";
  }
#endif
} // namespace dxvk
