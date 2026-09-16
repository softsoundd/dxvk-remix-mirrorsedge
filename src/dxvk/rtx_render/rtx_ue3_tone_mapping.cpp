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
#include "rtx_ue3_tone_mapping.h"

#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_render/rtx_shader_manager.h"
#include "rtx_context.h"
#include "rtx_imgui.h"
#include "rtx/pass/tonemap/tonemapping_ue3.h"
#include "rtx/pass/tonemap/tonemapping_ue3_exposure.h"

#include <rtx_shaders/tonemapping_ue3.h>
#include <rtx_shaders/tonemapping_ue3_exposure.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace dxvk {
  // Defined within an unnamed namespace to ensure unique definition across binary
  namespace {
    class Ue3ToneMappingShader : public ManagedShader {
      SHADER_SOURCE(Ue3ToneMappingShader, VK_SHADER_STAGE_COMPUTE_BIT, tonemapping_ue3)

      BEGIN_PARAMETER()
        RW_TEXTURE2D(TONEMAPPING_UE3_COLOR_INPUT)
        RW_TEXTURE1D_READONLY(TONEMAPPING_UE3_EXPOSURE_INPUT)
        SAMPLER1D(TONEMAPPING_UE3_CURVE_K_INPUT)
        SAMPLER1D(TONEMAPPING_UE3_CURVE_M_INPUT)
        CONSTANT_BUFFER(TONEMAPPING_UE3_CONSTANTS_INPUT)
        RW_TEXTURE2D(TONEMAPPING_UE3_COLOR_OUTPUT)
      END_PARAMETER()
    };

    class Ue3ExposureMeterShader : public ManagedShader {
      SHADER_SOURCE(Ue3ExposureMeterShader, VK_SHADER_STAGE_COMPUTE_BIT, tonemapping_ue3_exposure)

      PUSH_CONSTANTS(ToneMappingUe3ExposureArgs)

      BEGIN_PARAMETER()
        RW_TEXTURE2D(TONEMAPPING_UE3_EXPOSURE_COLOR_INPUT)
        RW_TEXTURE1D(TONEMAPPING_UE3_EXPOSURE_OUTPUT)
      END_PARAMETER()
    };

    constexpr VkExtent3D kCurveTextureExtent = { kUe3CurveSegmentCount, 1, 1 };
    constexpr VkExtent3D kMeterTextureExtent = { 1, 1, 1 };

    // The engine uploads dt * min(Scene_ExposureSpeedUp, 2.5) and dt * min(Scene_ExposureSpeedDown, 3.0);
    // dividing by our own frame time and the cap recovers the level's speed relative to the cap.
    constexpr float kEngineSpeedUpCap = 2.5f;
    constexpr float kEngineSpeedDownCap = 3.0f;

    float levelSpeedFactor(const float upload, const float cap, const float deltaTimeSeconds) {
      return std::clamp(upload / (deltaTimeSeconds * cap), 0.f, 1.f);
    }

    // Frame time for the meter's adaptation; falls back to the configured constant frame time (or
    // 60 FPS) when the per-frame delta is 0, as the auto exposure pass does.
    float meterDeltaTimeSeconds(const float frameTimeMilliseconds) {
      const float fallbackMs = RtxOptions::timeDeltaBetweenFrames() > 0.f ? RtxOptions::timeDeltaBetweenFrames() : 16.6f;
      return (frameTimeMilliseconds > 0.f ? frameTimeMilliseconds : fallbackMs) * 0.001f;
    }

    // Identity segments (y = x) in the game's K/M texel layout
    std::array<Vector4, kUe3CurveSegmentCount> identityCurveK() {
      std::array<Vector4, kUe3CurveSegmentCount> texels;
      texels.fill(Vector4(1.f, 0.f, 1.f, 0.f)); // [Ms.r, Bs.r, Ms.g, Bs.g]
      return texels;
    }
    std::array<Vector4, kUe3CurveSegmentCount> identityCurveM() {
      std::array<Vector4, kUe3CurveSegmentCount> texels;
      texels.fill(Vector4(1.f, 0.f, 0.f, 0.f)); // [Ms.b, Bs.b, -, -]
      return texels;
    }
  }

  DxvkUe3ToneMapping::DxvkUe3ToneMapping(DxvkDevice* device)
  : CommonDeviceObject(device) {
  }

  DxvkUe3ToneMapping::~DxvkUe3ToneMapping() { }

  void DxvkUe3ToneMapping::prewarmShaders(DxvkPipelineManager& pipelineManager) const {
    if (RtxOptions::tonemappingMode() != TonemappingMode::MirrorsEdge) {
      return;
    }

    Ue3ToneMappingShader::getShader();
    Ue3ExposureMeterShader::getShader();
  }

  void DxvkUe3ToneMapping::createResources(Rc<RtxContext> ctx) {
    Rc<DxvkContext> baseCtx = ctx;

    m_curveK = Resources::createImageResource(baseCtx, "ue3 tonemap curve K", kCurveTextureExtent, VK_FORMAT_R32G32B32A32_SFLOAT,
                                              1, VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D);
    m_curveM = Resources::createImageResource(baseCtx, "ue3 tonemap curve M", kCurveTextureExtent, VK_FORMAT_R32G32B32A32_SFLOAT,
                                              1, VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D);
    // Cleared to 0, which the meter reads as "no state" and starts on its target
    m_meterExposure = Resources::createImageResource(baseCtx, "ue3 tonemap exposure meter", kMeterTextureExtent, VK_FORMAT_R32_SFLOAT,
                                                     1, VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D);
    m_meterResetPending = true;

    uploadMsBsTexels(ctx, identityCurveK(), identityCurveM());
  }

  void DxvkUe3ToneMapping::onCapture(const Ue3ToneMapCapture& capture, uint32_t frameId) {
    m_capture = capture;
    m_hasCapture = true;
    m_captureFrameId = frameId;
  }

  void DxvkUe3ToneMapping::uploadMsBsTexels(Rc<RtxContext> ctx,
                                            const std::array<Vector4, kUe3CurveSegmentCount>& texelsK,
                                            const std::array<Vector4, kUe3CurveSegmentCount>& texelsM) {
    const bool changed = !m_hasUploadedMsBs ||
                         std::memcmp(texelsK.data(), m_uploadedK.data(), sizeof(m_uploadedK)) != 0 ||
                         std::memcmp(texelsM.data(), m_uploadedM.data(), sizeof(m_uploadedM)) != 0;
    if (!changed) {
      return;
    }

    const VkImageSubresourceLayers subresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    const VkDeviceSize pitch = kUe3CurveSegmentCount * sizeof(Vector4);

    ctx->updateImage(m_curveK.image, subresource, VkOffset3D { 0, 0, 0 }, kCurveTextureExtent, texelsK.data(), pitch, pitch);
    ctx->updateImage(m_curveM.image, subresource, VkOffset3D { 0, 0, 0 }, kCurveTextureExtent, texelsM.data(), pitch, pitch);

    m_uploadedK = texelsK;
    m_uploadedM = texelsM;
    m_hasUploadedMsBs = true;
  }

  void DxvkUe3ToneMapping::dispatch(
    Rc<RtxContext> ctx,
    Rc<DxvkSampler> linearSampler,
    Rc<DxvkImageView> exposureView,
    const Resources::RaytracingOutput& rtOutput,
    bool autoExposureEnabled,
    float frameTimeMilliseconds,
    bool resetHistory) {

    ScopedGpuProfileZone(ctx, "Mirror's Edge Tone Mapping");

    if (m_curveK.image == nullptr) {
      createResources(ctx);
    }
    m_meterResetPending |= resetHistory;

    const uint32_t currentFrame = device()->getCurrentFrameId();
    const uint32_t captureAge = (m_hasCapture && currentFrame >= m_captureFrameId) ? (currentFrame - m_captureFrameId) : ~0u;
    const bool captureFresh = m_hasCapture && captureAge <= uint32_t(std::max(0, captureStaleFrames()));
    const bool captureUsable = captureFresh || (m_hasCapture && holdStaleCapture());

    const bool usingCapturedCurves = captureUsable && m_capture.hasCurves;
    if (usingCapturedCurves) {
      uploadMsBsTexels(ctx, m_capture.curveK, m_capture.curveM);
    } else {
      uploadMsBsTexels(ctx, identityCurveK(), identityCurveM());
    }

    ToneMappingUe3Args args = {};

    const bool usingCapturedConstants = useCapturedConstants() && captureUsable && m_capture.hasConstants;
    if (usingCapturedConstants) {
      args.sceneShadowsAndDesaturation = m_capture.sceneShadowsAndDesaturation;
      args.sceneInverseHighLights = m_capture.sceneInverseHighLights;
      args.sceneMidTones = m_capture.sceneMidTones;
      args.sceneScaledLuminanceWeights = m_capture.sceneScaledLuminanceWeights;
      args.gammaColorScaleAndInverse = m_capture.gammaColorScaleAndInverse;
      args.gammaOverlayColor = applyCapturedOverlayColor() ? m_capture.gammaOverlayColor : Vector4(0.f, 0.f, 0.f, 0.f);
    } else {
      const Vector3 shadows = manualSceneShadows();
      const Vector3 highLights = manualSceneHighLights();
      const Vector3 midTones = manualSceneMidTones();
      const float desaturation = std::clamp(manualSceneDesaturation(), 0.f, 1.f);
      // The shipped TdToneMapping desaturation weights; the game's grade was authored against them
      const Vector3 lumaWeights = Vector3(0.3f, 0.59f, 0.11f);
      const Vector3 gammaColorScale = manualGammaColorScale();

      args.sceneShadowsAndDesaturation = Vector4(shadows.x, shadows.y, shadows.z, 1.f - desaturation);
      args.sceneInverseHighLights = Vector4(1.f / std::max(highLights.x, 1e-4f),
                                            1.f / std::max(highLights.y, 1e-4f),
                                            1.f / std::max(highLights.z, 1e-4f), 1.f);
      args.sceneMidTones = Vector4(midTones.x, midTones.y, midTones.z, 1.f);
      args.sceneScaledLuminanceWeights = Vector4(lumaWeights.x * desaturation,
                                                 lumaWeights.y * desaturation,
                                                 lumaWeights.z * desaturation, 0.f);
      args.gammaColorScaleAndInverse = Vector4(gammaColorScale.x, gammaColorScale.y, gammaColorScale.z,
                                               1.f / std::max(manualDisplayGamma(), 0.01f));
      args.gammaOverlayColor = Vector4(0.f, 0.f, 0.f, 0.f);
    }

    const Resources::Resource& inputColorBuffer = rtOutput.m_finalOutput.resource(Resources::AccessType::Read);
    const Resources::Resource& outputColorBuffer = rtOutput.m_finalOutput.resource(Resources::AccessType::Write);

    // Scene calibration: Remix radiance times this is in the game's scene units
    const float sceneScale = exp2f(exposureBias());
    const float userBrightness = exp2f(RtxOptions::calcUserEVBias());

    // --- Exposure: the game's meter on Remix's radiance, or Remix's auto exposure ---
    const bool usingMeter = exposureModel() == Ue3ExposureModel::MirrorsEdgeMeter;
    Rc<DxvkImageView> exposureViewToBind = exposureView;
    bool meterClampsFromCapture = false;
    float meterLow = manualExposureLow();
    float meterHigh = manualExposureHigh();
    float meterManual = manualExposureManual();
    float meterSpeedFactorUp = 1.f;
    float meterSpeedFactorDown = 1.f;

    if (usingMeter) {
      const float deltaTimeSeconds = meterDeltaTimeSeconds(frameTimeMilliseconds);

      if (meterUseCapturedSettings() && captureUsable && m_capture.hasExposureSettings) {
        meterClampsFromCapture = true;
        meterLow = m_capture.exposureSettings.z;
        meterHigh = m_capture.exposureSettings.w;
        meterManual = m_capture.exposureSettings.x;
        if (meterHonourLevelSpeeds()) {
          meterSpeedFactorUp = levelSpeedFactor(m_capture.exposureSettings.y, kEngineSpeedUpCap, deltaTimeSeconds);
          meterSpeedFactorDown = levelSpeedFactor(m_capture.maxDeltaDown, kEngineSpeedDownCap, deltaTimeSeconds);
        }
      }
      meterLow = std::max(meterLow, 1e-3f);
      meterHigh = std::max(meterHigh, meterLow);
      meterManual = std::max(meterManual, 1e-3f);

      ToneMappingUe3ExposureArgs meterArgs = {};
      meterArgs.sceneScale = sceneScale;
      meterArgs.deltaTimeSeconds = deltaTimeSeconds;
      meterArgs.exposureLow = meterLow;
      meterArgs.exposureHigh = meterHigh;
      // Into light the exposure falls, governed by the level's SpeedDown; into dark it rises, by SpeedUp
      meterArgs.speedToLight = std::max(meterSpeedToLight(), 0.f) * meterSpeedFactorDown;
      meterArgs.speedToDark = std::max(meterSpeedToDark(), 0.f) * meterSpeedFactorUp;
      meterArgs.transitionStops = std::max(meterTransitionStops(), 1e-3f);
      meterArgs.resetHistory = m_meterResetPending ? 1u : 0u;
      m_meterResetPending = false;

      {
        ScopedGpuProfileZone(ctx, "Mirror's Edge Exposure Meter");
        ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
        ctx->pushConstants(0, sizeof(meterArgs), &meterArgs);
        ctx->bindResourceView(TONEMAPPING_UE3_EXPOSURE_COLOR_INPUT, inputColorBuffer.view, nullptr);
        ctx->bindResourceView(TONEMAPPING_UE3_EXPOSURE_OUTPUT, m_meterExposure.view, nullptr);
        ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, Ue3ExposureMeterShader::getShader());
        ctx->dispatch(1, 1, 1);
      }

      exposureViewToBind = m_meterExposure.view;
    }

    // The meter's texel is E as the game's shader stores it; Scene_ExposureManual, the scene
    // calibration and the user brightness multiply it here.
    args.enableAutoExposure = usingMeter ? 1u : (autoExposureEnabled ? 1u : 0u);
    args.exposureFactor = sceneScale * userBrightness * (usingMeter ? meterManual : 1.f);
    args.applyColorCurves = usingCapturedCurves;
    // Match the sampler filter the game bound for the curve LUTs (Mirror's
    // Edge point-filters them, i.e. exact piecewise-segment evaluation)
    args.curvePointSampling = usingCapturedCurves && m_capture.curvePointFiltering;

    // Faithful Luma; the clamps keep every curve monotonic (knee below display white, white
    // points at or above it)
    args.faithfulLuma = faithfulLuma();
    args.rangeCompression = uint32_t(rangeCompression());
    args.neutwoWhiteClip = std::max(neutwoWhiteClip(), 1.f);
    args.neutwoContrast = std::clamp(neutwoContrast(), 0.1f, 4.f);
    args.softClipKnee = std::clamp(softClipKnee(), 0.05f, 0.99f);
    args.softClipWhite = std::max(softClipWhite(), 1.01f);
    args.huePreservation = std::clamp(huePreservation(), 0.f, 1.f);
    args.bezoldBruckePerStop = bezoldBruckePerStop();
    args.hueReference = uint32_t(hueReference());
    args.highlightDesaturation = std::clamp(highlightDesaturation(), 0.f, 1.f);

    if (m_constants == nullptr) {
      DxvkBufferCreateInfo info;
      info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT;
      info.size = sizeof(ToneMappingUe3Args);
      m_constants = device()->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "UE3 Tone Mapping Constant Buffer");
    }

    ctx->writeToBuffer(m_constants, 0, sizeof(ToneMappingUe3Args), &args);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constants);

    const VkExtent3D workgroups = util::computeBlockCount(outputColorBuffer.view->imageInfo().extent, VkExtent3D { 16, 16, 1 });

    ctx->bindResourceView(TONEMAPPING_UE3_COLOR_INPUT, inputColorBuffer.view, nullptr);
    ctx->bindResourceView(TONEMAPPING_UE3_EXPOSURE_INPUT, exposureViewToBind, nullptr);
    ctx->bindResourceView(TONEMAPPING_UE3_CURVE_K_INPUT, m_curveK.view, nullptr);
    ctx->bindResourceSampler(TONEMAPPING_UE3_CURVE_K_INPUT, linearSampler);
    ctx->bindResourceView(TONEMAPPING_UE3_CURVE_M_INPUT, m_curveM.view, nullptr);
    ctx->bindResourceSampler(TONEMAPPING_UE3_CURVE_M_INPUT, linearSampler);
    ctx->bindResourceBuffer(TONEMAPPING_UE3_CONSTANTS_INPUT, DxvkBufferSlice(m_constants, 0, m_constants->info().size));
    ctx->bindResourceView(TONEMAPPING_UE3_COLOR_OUTPUT, outputColorBuffer.view, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, Ue3ToneMappingShader::getShader());
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);

    // Publish a status snapshot for the settings UI
    {
      std::lock_guard<std::mutex> lock(m_statusMutex);
      m_status.usingCapturedCurves = usingCapturedCurves;
      m_status.captureSeen = m_hasCapture;
      m_status.captureFresh = captureFresh;
      m_status.captureHasCurves = m_capture.hasCurves;
      m_status.captureAgeFrames = m_hasCapture ? captureAge : 0;
      m_status.usingCapturedConstants = usingCapturedConstants;
      m_status.usingMeter = usingMeter;
      m_status.meterClampsFromCapture = meterClampsFromCapture;
      m_status.meterLow = meterLow;
      m_status.meterHigh = meterHigh;
      m_status.meterManual = meterManual;
      m_status.meterSpeedFactorUp = meterSpeedFactorUp;
      m_status.meterSpeedFactorDown = meterSpeedFactorDown;
      m_status.capturedState = m_capture;
    }
  }

  void DxvkUe3ToneMapping::showImguiSettings() {
    Status status;
    {
      std::lock_guard<std::mutex> lock(m_statusMutex);
      status = m_status;
    }

    ImGui::Text("Active curves: %s", status.usingCapturedCurves ? "live capture (game TdTonemapping pass)" : "identity (no curves)");

    if (status.captureSeen) {
      ImGui::Text("Capture: %s (age %u frame%s)%s",
                  status.captureFresh ? "fresh" : (holdStaleCapture() ? "stale (held)" : "stale"),
                  status.captureAgeFrames,
                  status.captureAgeFrames == 1 ? "" : "s",
                  status.captureHasCurves ? "" : " [constants only - no curve textures resolved]");
    } else {
      ImGui::TextWrapped("Capture: none seen. Enable the game's native tonemapper (remove 'scale set TdTonemapping false' or set it to true) so its pass can be captured; the draw itself stays skipped.");
    }
    ImGui::Text("Grade constants: %s", status.usingCapturedConstants ? "captured from game" : "manual options");

    RemixGui::Separator();

    // --- Exposure ---
    RemixGui::Combo("Exposure Model", &exposureModelObject(), "Remix Auto Exposure\0Mirror's Edge Meter (game model, captured per-volume clamps)\0");
    const bool meterSelected = exposureModel() == Ue3ExposureModel::MirrorsEdgeMeter;
    RemixGui::DragFloat(meterSelected ? "Scene Calibration (EV)" : "Exposure Bias (EV)", &exposureBiasObject(), 0.01f, -6.f, 6.f);
    if (meterSelected) {
      ImGui::Indent();
      if (status.usingMeter) {
        ImGui::Text("Clamps: Low %.3f  High %.3f  ->  E in [%.3f, %.3f]  (%s)  Manual %.3f",
                    status.meterLow, status.meterHigh,
                    status.meterLow * status.meterLow, status.meterHigh * status.meterHigh,
                    status.meterClampsFromCapture ? "captured from the current volume" : "manual",
                    status.meterManual);
        if (status.meterClampsFromCapture && meterHonourLevelSpeeds()) {
          ImGui::Text("Level speed factors: up %.2f  down %.2f", status.meterSpeedFactorUp, status.meterSpeedFactorDown);
        }
      }
      RemixGui::Checkbox("Use Captured Exposure Settings", &meterUseCapturedSettingsObject());
      RemixGui::DragFloat("Manual Scene_ExposureLow", &manualExposureLowObject(), 0.005f, 0.05f, 4.f);
      RemixGui::DragFloat("Manual Scene_ExposureHigh", &manualExposureHighObject(), 0.005f, 0.05f, 4.f);
      RemixGui::DragFloat("Manual Scene_ExposureManual", &manualExposureManualObject(), 0.005f, 0.05f, 4.f);
      RemixGui::DragFloat("Adaptation To Light (stops/s)", &meterSpeedToLightObject(), 0.1f, 0.f, 60.f);
      RemixGui::DragFloat("Adaptation To Dark (stops/s)", &meterSpeedToDarkObject(), 0.1f, 0.f, 60.f);
      RemixGui::DragFloat("Transition (stops)", &meterTransitionStopsObject(), 0.05f, 0.05f, 6.f);
      RemixGui::Checkbox("Honour Level Exposure Speeds", &meterHonourLevelSpeedsObject());
      ImGui::Unindent();
    }

    RemixGui::Separator();

    RemixGui::Checkbox("Use Captured Grade Constants", &useCapturedConstantsObject());
    if (useCapturedConstants()) {
      ImGui::Indent();
      RemixGui::Checkbox("Apply Captured Overlay Color (engine fades)", &applyCapturedOverlayColorObject());
      ImGui::Unindent();
    }
    RemixGui::DragInt("Capture Stale Frames", &captureStaleFramesObject(), 1.f, 1, 600);
    RemixGui::Checkbox("Hold Stale Capture", &holdStaleCaptureObject());

    RemixGui::Separator();

    // --- Faithful Luma ---
    RemixGui::Checkbox("Faithful Luma", &faithfulLumaObject());
    if (faithfulLuma()) {
      ImGui::Indent();
      ImGui::TextWrapped("Shipped grade, gamma and curves; the per-channel clip at exposed 1.0 becomes a range compression curve with its hue solved in OKLab, and black reaches code 0.");

      RemixGui::Combo("Range Compression", &rangeCompressionObject(), "Neutwo (compresses from mid grey, asymptotic headroom)\0Faithful Luma Shoulder (identity to the knee, white at 3x)\0");
      ImGui::Indent();
      if (rangeCompression() == Ue3RangeCompression::Neutwo) {
        RemixGui::DragFloat("White Clip", &neutwoWhiteClipObject(), 0.1f, 1.f, 100.f, "%.1f", ImGuiSliderFlags_Logarithmic);
        RemixGui::DragFloat("Contrast", &neutwoContrastObject(), 0.005f, 0.5f, 2.f);
      } else {
        RemixGui::DragFloat("Soft Clip Knee", &softClipKneeObject(), 0.005f, 0.05f, 0.99f);
        RemixGui::DragFloat("Soft Clip White", &softClipWhiteObject(), 0.01f, 1.01f, 16.f);
      }
      ImGui::Unindent();

      RemixGui::DragFloat("Hue Preservation", &huePreservationObject(), 0.01f, 0.f, 1.f);
      RemixGui::DragFloat("Bezold-Brucke Per Stop (deg)", &bezoldBruckePerStopObject(), 0.05f, 0.f, 10.f);
      RemixGui::Combo("Hue Reference", &hueReferenceObject(), "Scene\0Approved Look (share of the clip's hue)\0");
      RemixGui::DragFloat("Highlight Desaturation", &highlightDesaturationObject(), 0.01f, 0.f, 1.f);
      ImGui::Unindent();
    } else {
      ImGui::TextWrapped("Faithful Luma disabled: verbatim shipped TdToneMapping (per-channel clip at exposed 1.0 with its hue shifts, black floor at #020202).");
    }

    RemixGui::Separator();

    // --- Manual grade constants ---
    if (RemixGui::CollapsingHeader("Manual Grade Constants (used when capture is unavailable or disabled)", ImGuiTreeNodeFlags_None)) {
      ImGui::Indent();
      RemixGui::DragFloat3("Scene Shadows", &manualSceneShadowsObject(), 0.005f, -1.f, 1.f);
      RemixGui::DragFloat3("Scene HighLights", &manualSceneHighLightsObject(), 0.005f, 0.05f, 4.f);
      RemixGui::DragFloat3("Scene MidTones", &manualSceneMidTonesObject(), 0.005f, 0.1f, 4.f);
      RemixGui::DragFloat("Scene Desaturation", &manualSceneDesaturationObject(), 0.005f, 0.f, 1.f);
      RemixGui::DragFloat("Display Gamma", &manualDisplayGammaObject(), 0.01f, 1.f, 3.f);
      RemixGui::DragFloat3("Gamma Color Scale", &manualGammaColorScaleObject(), 0.005f, 0.f, 2.f);
      ImGui::TextWrapped("Desaturation uses the shipped 0.3 / 0.59 / 0.11 luminance weights.");
      ImGui::Unindent();
    }

    // --- Captured values readout ---
    if (status.captureSeen && RemixGui::CollapsingHeader("Captured Game State", ImGuiTreeNodeFlags_None)) {
      const Ue3ToneMapCapture& c = status.capturedState;
      ImGui::Indent();
      ImGui::Text("SceneShadows: %.4f %.4f %.4f  (1-Desat): %.4f",
                  c.sceneShadowsAndDesaturation.x, c.sceneShadowsAndDesaturation.y,
                  c.sceneShadowsAndDesaturation.z, c.sceneShadowsAndDesaturation.w);
      ImGui::Text("SceneInverseHighLights: %.4f %.4f %.4f",
                  c.sceneInverseHighLights.x, c.sceneInverseHighLights.y, c.sceneInverseHighLights.z);
      ImGui::Text("SceneMidTones: %.4f %.4f %.4f",
                  c.sceneMidTones.x, c.sceneMidTones.y, c.sceneMidTones.z);
      ImGui::Text("SceneScaledLuminanceWeights: %.4f %.4f %.4f",
                  c.sceneScaledLuminanceWeights.x, c.sceneScaledLuminanceWeights.y, c.sceneScaledLuminanceWeights.z);
      ImGui::Text("GammaColorScale: %.4f %.4f %.4f  GammaInverse: %.4f (gamma %.3f)",
                  c.gammaColorScaleAndInverse.x, c.gammaColorScaleAndInverse.y, c.gammaColorScaleAndInverse.z,
                  c.gammaColorScaleAndInverse.w,
                  c.gammaColorScaleAndInverse.w != 0.f ? 1.f / c.gammaColorScaleAndInverse.w : 0.f);
      ImGui::Text("GammaOverlayColor: %.4f %.4f %.4f",
                  c.gammaOverlayColor.x, c.gammaOverlayColor.y, c.gammaOverlayColor.z);
      if (c.hasExposureSettings) {
        ImGui::Text("TdToneMapExposure: Manual %.3f  Low %.3f  High %.3f  dt*min(SpeedUp,2.5) %.5f  dt*min(SpeedDown,3.0) %.5f",
                    c.exposureSettings.x, c.exposureSettings.z, c.exposureSettings.w,
                    c.exposureSettings.y, c.maxDeltaDown);
      } else {
        ImGui::Text("TdToneMapExposure: not seen");
      }
      ImGui::Text("Curve textures: %s", c.hasCurves ? "captured" : "not resolved");
      if (c.hasCurves) {
        ImGui::Text("Curve LUT filtering: %s", c.curvePointFiltering ? "point" : "linear");
        ImGui::Text("K[0]:  Ms.r %.3f Bs.r %.3f Ms.g %.3f Bs.g %.3f",
                    c.curveK[0].x, c.curveK[0].y, c.curveK[0].z, c.curveK[0].w);
        ImGui::Text("M[0]:  Ms.b %.3f Bs.b %.3f",
                    c.curveM[0].x, c.curveM[0].y);
        ImGui::Text("K[14]: Ms.r %.3f Bs.r %.3f Ms.g %.3f Bs.g %.3f",
                    c.curveK[14].x, c.curveK[14].y, c.curveK[14].z, c.curveK[14].w);
        ImGui::Text("K[15]: Ms.r %.3f Bs.r %.3f Ms.g %.3f Bs.g %.3f",
                    c.curveK[15].x, c.curveK[15].y, c.curveK[15].z, c.curveK[15].w);
        ImGui::Text("M[14]: Ms.b %.3f Bs.b %.3f | M[15]: Ms.b %.3f Bs.b %.3f",
                    c.curveM[14].x, c.curveM[14].y, c.curveM[15].x, c.curveM[15].y);
      }
      ImGui::Unindent();
    }
  }
}
