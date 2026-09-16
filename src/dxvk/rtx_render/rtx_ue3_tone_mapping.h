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

#include "dxvk_format.h"
#include "dxvk_include.h"
#include "dxvk_context.h"
#include "rtx_resources.h"
#include "rtx_options.h"

#include "../util/util_vector.h"

#include <array>
#include <mutex>

namespace dxvk {

  class DxvkDevice;
  class DxvkPipelineManager;

  // The game bakes its per-map colour curves into 16 piecewise linear
  // segments (FCurveInfo.Ms[16]/Bs[16]) uploaded as two 16x1 LUT textures.
  constexpr uint32_t kUe3CurveSegmentCount = 16;

  // Snapshot of the game's TdToneMapping pass state, captured by the D3D9
  // layer when the (skipped) native tonemap fullscreen draw is seen. Grade
  // constants are the pixel shader constants verbatim; curve texels are the
  // game's baked/blended ColorCurvesK/ColorCurvesM 16x1 LUT contents.
  struct Ue3ToneMapCapture {
    Vector4 sceneShadowsAndDesaturation = Vector4(0.f, 0.f, 0.f, 1.f); // rgb: shadows, a: (1 - desaturation)
    Vector4 sceneInverseHighLights = Vector4(1.f, 1.f, 1.f, 1.f);
    Vector4 sceneMidTones = Vector4(1.f, 1.f, 1.f, 1.f);
    Vector4 sceneScaledLuminanceWeights = Vector4(0.f, 0.f, 0.f, 0.f);
    Vector4 gammaColorScaleAndInverse = Vector4(1.f, 1.f, 1.f, 0.5f);  // a: 1 / DisplayGamma (ME default 2.0)
    Vector4 gammaOverlayColor = Vector4(0.f, 0.f, 0.f, 0.f);
    bool hasConstants = false;
    bool hasCurves = false;
    // Sampler filter the game bound for the curve LUTs; matched by the shader
    bool curvePointFiltering = false;
    std::array<Vector4, kUe3CurveSegmentCount> curveK = {};
    std::array<Vector4, kUe3CurveSegmentCount> curveM = {};

    // The (skipped) TdToneMapExposure pass's constants for the current PostProcessVolume.
    // ExposureSettings = (Scene_ExposureManual, dt * min(Scene_ExposureSpeedUp, 2.5),
    // Scene_ExposureLow, Scene_ExposureHigh); the clamps apply to sqrt(E).
    // MaxDeltaDown = dt * min(Scene_ExposureSpeedDown, 3.0).
    bool hasExposureSettings = false;
    Vector4 exposureSettings = Vector4(1.f, 0.f, 0.85f, 1.65f); // PostProcessVolume defaults
    float maxDeltaDown = 0.f;
  };

  // Which hue the Faithful Luma range compression aims an over-range colour at.
  enum class Ue3HueReference : int {
    Scene = 0,         // the scene's hue
    ApprovedLook = 1,  // turned toward the shipped clip's hue by the share of luminance the clip could not show
  };

  // Which curve limits the graded colour to the display range under Faithful Luma.
  enum class Ue3RangeCompression : int {
    Neutwo = 0,                // x / sqrt(x^2 + 1) family (RenoDX form), compresses from mid grey, asymptotic headroom
    FaithfulLumaShoulder = 1,  // identity below the knee, extended Reinhard to the white point (the reference shaders' curve)
  };

  // Where the exposure the display transform applies comes from.
  enum class Ue3ExposureModel : int {
    RemixAutoExposure = 0,  // rtx.autoExposure, as for the other tonemappers
    MirrorsEdgeMeter = 1,   // the game's TdToneMapExposure model on Remix's radiance, with the captured per-volume clamps
  };

  // Mirror's Edge (UE3 / TdToneMapping) display transform pass. Replaces the
  // Global/Local tonemappers when rtx.tonemappingMode selects it. Outputs
  // display-encoded color (gamma + colour curves applied), so the final
  // srgb_dither pass must skip its linear -> sRGB conversion for this mode.
  class DxvkUe3ToneMapping : public CommonDeviceObject {
  public:
    explicit DxvkUe3ToneMapping(DxvkDevice* device);
    ~DxvkUe3ToneMapping();

    void dispatch(
      Rc<RtxContext> ctx,
      Rc<DxvkSampler> linearSampler,
      Rc<DxvkImageView> exposureView,
      const Resources::RaytracingOutput& rtOutput,
      bool autoExposureEnabled,
      float frameTimeMilliseconds,
      bool resetHistory);

    // Called on the CS thread (via RtxContext::setUe3ToneMapCapture).
    void onCapture(const Ue3ToneMapCapture& capture, uint32_t frameId);

    void prewarmShaders(DxvkPipelineManager& pipelineManager) const;

    void showImguiSettings();

  private:
    void createResources(Rc<RtxContext> ctx);
    void uploadMsBsTexels(Rc<RtxContext> ctx,
                          const std::array<Vector4, kUe3CurveSegmentCount>& texelsK,
                          const std::array<Vector4, kUe3CurveSegmentCount>& texelsM);

    Resources::Resource m_curveK;
    Resources::Resource m_curveM;
    // 1x1 R32F state of the Mirror's Edge exposure meter: E as the game stores it, persisting
    // across frames for the adaptation.
    Resources::Resource m_meterExposure;
    bool m_meterResetPending = true;
    Rc<DxvkBuffer> m_constants;

    // Last texel payloads uploaded to the GPU curve textures (change detection)
    std::array<Vector4, kUe3CurveSegmentCount> m_uploadedK = {};
    std::array<Vector4, kUe3CurveSegmentCount> m_uploadedM = {};
    bool m_hasUploadedMsBs = false;

    // Captured game state (CS thread only)
    Ue3ToneMapCapture m_capture;
    bool m_hasCapture = false;
    uint32_t m_captureFrameId = 0;

    // Status snapshot for the ImGui thread
    struct Status {
      bool usingCapturedCurves = false;
      bool captureSeen = false;
      bool captureFresh = false;
      bool captureHasCurves = false;
      uint32_t captureAgeFrames = 0;
      bool usingCapturedConstants = false;
      // Exposure meter as dispatched this frame
      bool usingMeter = false;
      bool meterClampsFromCapture = false;
      float meterLow = 0.f;
      float meterHigh = 0.f;
      float meterManual = 1.f;
      float meterSpeedFactorUp = 1.f;
      float meterSpeedFactorDown = 1.f;
      Ue3ToneMapCapture capturedState;
    };
    mutable std::mutex m_statusMutex;
    Status m_status;

    RTX_OPTION("rtx.tonemap.ue3", float, exposureBias, 0.f,
               "Scene calibration in EV: exp2 of this scales Remix's linear radiance into the game's scene units (exposed 1.0 = display white) before the meter and the grade, so one value serves every area and the per-volume clamps do the rest. With Remix auto exposure it is a bias on top of the auto exposure result, or the exposure itself when auto exposure is disabled.");
    RTX_OPTION("rtx.tonemap.ue3", Ue3ExposureModel, exposureModel, Ue3ExposureModel::MirrorsEdgeMeter,
               "Valid values: <RemixAutoExposure=0, MirrorsEdgeMeter=1>. "
               "MirrorsEdgeMeter runs the game's TdToneMapExposure model on Remix's radiance: an arithmetic mean with every channel clamped at 1.0 (the shipped fixed-point downsample chain), the shipped 0.3/0.59/0.11 luminosity, key 0.25, and sqrt(E) clamped to the Scene_ExposureLow/High captured from the current PostProcessVolume. Those clamps are authored per area and are what holds a bright exterior at its floor while an interior may rise to its ceiling, so the authored per-area exposure carries over. Adaptation follows Faithful Luma's stops-per-second law. "
               "RemixAutoExposure uses rtx.autoExposure as the other tonemappers do.");
    RTX_OPTION("rtx.tonemap.ue3", bool, meterUseCapturedSettings, true,
               "Mirror's Edge meter: take Scene_ExposureLow/High/Manual from the live capture of the game's exposure pass. When disabled, or while no capture is available, the manual values apply.");
    RTX_OPTION("rtx.tonemap.ue3", float, manualExposureLow, 0.85f,
               "Mirror's Edge meter: Scene_ExposureLow when no capture is used; clamps sqrt(E) from below (0.85 is the PostProcessVolume default).");
    RTX_OPTION("rtx.tonemap.ue3", float, manualExposureHigh, 1.65f,
               "Mirror's Edge meter: Scene_ExposureHigh when no capture is used; clamps sqrt(E) from above (1.65 is the PostProcessVolume default).");
    RTX_OPTION("rtx.tonemap.ue3", float, manualExposureManual, 1.0f,
               "Mirror's Edge meter: Scene_ExposureManual when no capture is used; multiplies the metered exposure.");
    RTX_OPTION("rtx.tonemap.ue3", float, meterSpeedToLight, 12.0f,
               "Mirror's Edge meter: adaptation speed in stops per second while the exposure falls (into light). 12 / 6 feel like an eye; 3 / 1 are typical engine defaults and feel like a camera.");
    RTX_OPTION("rtx.tonemap.ue3", float, meterSpeedToDark, 6.0f,
               "Mirror's Edge meter: adaptation speed in stops per second while the exposure rises (into dark).");
    RTX_OPTION("rtx.tonemap.ue3", float, meterTransitionStops, 1.5f,
               "Mirror's Edge meter: distance from the target, in stops, inside which the adaptation eases in exponentially with the same slope (time constant = transition / speed).");
    RTX_OPTION("rtx.tonemap.ue3", bool, meterHonourLevelSpeeds, true,
               "Mirror's Edge meter: scale the adaptation speeds by the level's Scene_ExposureSpeedUp/Down relative to the engine caps (2.5 / 3.0), recovered from the captured uploads. The PostProcessVolume defaults exceed the caps, so most areas run at full speed; a level that zeroes a speed holds the exposure, as shipped.");
    RTX_OPTION("rtx.tonemap.ue3", int, captureStaleFrames, 8,
               "Number of frames a live capture of the game's tonemap pass stays fresh. When exceeded (e.g. the game's TdTonemapping is disabled), behavior follows rtx.tonemap.ue3.holdStaleCapture.");
    RTX_OPTION("rtx.tonemap.ue3", bool, holdStaleCapture, true,
               "Keep using the last captured curves and grade constants when the capture goes stale (e.g. loading screens, or the game's tonemap pass temporarily stopping) instead of snapping back to identity curves and manual constants. New captures always replace the held state.");
    RTX_OPTION("rtx.tonemap.ue3", bool, useCapturedConstants, true,
               "Use the grade constants (shadows/highlights/midtones/desaturation/display gamma) captured live from the game's tonemap pass when available. When disabled (or stale), the manual rtx.tonemap.ue3.manual* options are used instead.");
    RTX_OPTION("rtx.tonemap.ue3", bool, applyCapturedOverlayColor, true,
               "Apply the captured GammaOverlayColor engine constant (engine-side screen fades) in the tone map. Disable if fades are double-applied by replayed overlay draws.");

    // Faithful Luma: the shipped shader changed only where it loses information.
    RTX_OPTION("rtx.tonemap.ue3", bool, faithfulLuma, true,
               "Faithful Luma: keeps the shipped grade, gamma and curve math and replaces the per-channel hard clip at exposed 1.0 with a range compression curve (rtx.tonemap.ue3.rangeCompression) whose hue shift is corrected in OKLab, so bright surfaces keep their texture and saturated colours keep their hue while light sources still blow out to white; the shipped pow() guard is dropped so black reaches code 0. "
               "When disabled the transform is the shipped TdToneMapping verbatim: clipped channels shift hue and black floors at #020202.");
    RTX_OPTION("rtx.tonemap.ue3", Ue3RangeCompression, rangeCompression, Ue3RangeCompression::Neutwo,
               "Faithful Luma: the curve that limits the graded colour to the display range, applied per channel before the OKLab hue solve. Valid values: <Neutwo=0, FaithfulLumaShoulder=1>. "
               "Neutwo is the x / sqrt(x^2 + 1) family in RenoDX's display-peak / white-clip form: slope 1 at black, near-identity at mid grey, compressing gradually from there with headroom many stops past display white, which suits path-traced radiance. "
               "FaithfulLumaShoulder is the reference shaders' curve: identity below Soft Clip Knee and an extended Reinhard reaching white at Soft Clip White, so below the knee the image is the shipped image. It is sized for the game's own scene range.");
    RTX_OPTION("rtx.tonemap.ue3", float, neutwoWhiteClip, 100.0f,
               "Faithful Luma, Neutwo: graded value that lands exactly on display white; everything above is white. Large values approach the asymptotic curve (100 is RenoDX's neutral SDR default); lower values give hot sources a true white sooner; 1 is the shipped hard clip.");
    RTX_OPTION("rtx.tonemap.ue3", float, neutwoContrast, 1.0f,
               "Faithful Luma, Neutwo: power around mid grey (0.18) applied to luminance before the curve, chromaticity kept (RenoDRT's contrast). 1 = none.");
    RTX_OPTION("rtx.tonemap.ue3", float, softClipKnee, 0.8f,
               "Faithful Luma, shoulder: graded value where the shoulder starts; below it the image is the shipped image. 0.8 puts exposed white at 242/255; lower buys highlight room with content the shipped clip showed unclipped.");
    RTX_OPTION("rtx.tonemap.ue3", float, softClipWhite, 3.0f,
               "Faithful Luma, shoulder: graded value that reaches display white; everything above is white, as the clip made it. 3.0 is the 99th percentile of sunlit diffuse white measured in the game.");
    RTX_OPTION("rtx.tonemap.ue3", float, huePreservation, 1.0f,
               "Faithful Luma: how far the curve's per-channel hue shift is replaced by the target hue (the scene's OKLab hue, adjusted by Bezold-Brucke Per Stop and Hue Reference). Only the hue moves; the curve's peak channel and saturation are kept. 0 = shipped-style shifts, smoothed; 1 = the target hue.");
    RTX_OPTION("rtx.tonemap.ue3", float, bezoldBruckePerStop, 1.5f,
               "Faithful Luma: degrees of OKLab hue the target turns per stop the curve darkened the colour, toward yellow (between the purplish-red and green invariant hues) or blue (between green and purplish red), the way brightness turns perceived hue. 0 matches the input hue exactly.");
    RTX_OPTION("rtx.tonemap.ue3", Ue3HueReference, hueReference, Ue3HueReference::ApprovedLook,
               "Faithful Luma: which hue an over-range colour aims at. Valid values: <Scene=0, ApprovedLook=1>. Scene aims at the scene's hue. ApprovedLook turns the target toward the hue the shipped clip gave the colour by the share of its luminance that lay above display white, discounted by the chroma the clip kept: the levels were lit and graded through the clip, so its hue is part of the approved look. Displayable colours, lights blown to white and skies are barely affected; strongly over-range saturated paint takes a fraction of the clip's turn.");
    RTX_OPTION("rtx.tonemap.ue3", float, highlightDesaturation, 0.0f,
               "Faithful Luma: desaturate over-range colours to the chroma the shipped clip left them (the saturation of min(graded, 1)), toward the peak channel so hue and the curve's brightness are kept. 0 keeps the light's tint; 1 makes shipped whites white. Saturated colours and anything at or below 1.0 are unaffected.");

    // Manual grade constants, used when captured constants are unavailable or disabled
    RTX_OPTION("rtx.tonemap.ue3", Vector3, manualSceneShadows, Vector3(0.f, 0.f, 0.f),
               "Manual Scene_Shadows grade (subtracted after the highlights scale). UE3 default is 0.");
    RTX_OPTION("rtx.tonemap.ue3", Vector3, manualSceneHighLights, Vector3(1.f, 1.f, 1.f),
               "Manual Scene_HighLights grade (scene color is scaled by its reciprocal). UE3 default is 1.");
    RTX_OPTION("rtx.tonemap.ue3", Vector3, manualSceneMidTones, Vector3(1.f, 1.f, 1.f),
               "Manual Scene_MidTones grade (per-channel pow exponents). UE3 default is 1.");
    RTX_OPTION("rtx.tonemap.ue3", float, manualSceneDesaturation, 0.f,
               "Manual Scene_Desaturation grade in [0, 1]. UE3 default is 0 (no desaturation).");
    RTX_OPTION("rtx.tonemap.ue3", float, manualDisplayGamma, 2.0f,
               "Manual display gamma for the final encode. Mirror's Edge maps its brightness slider midpoint to 2.0 (not stock UE3's 2.2).");
    RTX_OPTION("rtx.tonemap.ue3", Vector3, manualGammaColorScale, Vector3(1.f, 1.f, 1.f),
               "Manual GammaColorScale applied before the display gamma encode.");
  };

}
