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
      bool autoExposureEnabled);

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
      Ue3ToneMapCapture capturedState;
    };
    mutable std::mutex m_statusMutex;
    Status m_status;

    RTX_OPTION("rtx.tonemap.ue3", float, exposureBias, 0.f,
               "Exposure bias (in EV) applied on top of the auto exposure result (or used directly when auto exposure is disabled) before the Mirror's Edge tone map grade.");
    RTX_OPTION("rtx.tonemap.ue3", int, captureStaleFrames, 8,
               "Number of frames a live capture of the game's tonemap pass stays fresh. When exceeded (e.g. the game's TdTonemapping is disabled), behavior follows rtx.tonemap.ue3.holdStaleCapture.");
    RTX_OPTION("rtx.tonemap.ue3", bool, holdStaleCapture, true,
               "Keep using the last captured curves and grade constants when the capture goes stale (e.g. loading screens, or the game's tonemap pass temporarily stopping) instead of snapping back to identity curves and manual constants. New captures always replace the held state.");
    RTX_OPTION("rtx.tonemap.ue3", bool, useCapturedConstants, true,
               "Use the grade constants (shadows/highlights/midtones/desaturation/display gamma) captured live from the game's tonemap pass when available. When disabled (or stale), the manual rtx.tonemap.ue3.manual* options are used instead.");
    RTX_OPTION("rtx.tonemap.ue3", bool, applyCapturedOverlayColor, true,
               "Apply the captured GammaOverlayColor engine constant (engine-side screen fades) in the tone map. Disable if fades are double-applied by replayed overlay draws.");

    // FaithfulLuma modernizations (default on: hue-stable handling of Remix's
    // unbounded path-traced radiance where the original renderer hard-clipped)
    RTX_OPTION("rtx.tonemap.ue3", bool, huePreservingShoulder, true,
               "Tone map through a luminance-anchored extended Reinhard shoulder and reconstruct RGB around it (FaithfulLuma), keeping hue stable through highlights. When disabled, uses the shipped per-channel hard clip at scene white (blown highlights clip per channel with the original hue shifts).");
    RTX_OPTION("rtx.tonemap.ue3", float, linearWhite, 4.0f,
               "Scene luminance mapped to display white by the hue-preserving shoulder. The shipped pipeline clamps scene color at MAX_SCENE_COLOR = 4.0; lower values brighten highlights overall.");
    RTX_OPTION("rtx.tonemap.ue3", bool, highlightDesaturation, true,
               "With the hue-preserving shoulder: desaturate only genuinely hot sources toward display white (quadratic weight engaging above half of Linear White).");
    RTX_OPTION("rtx.tonemap.ue3", float, highlightDesaturationStrength, 0.55f,
               "Maximum highlight desaturation toward white.");
    RTX_OPTION("rtx.tonemap.ue3", bool, gradePreserveBlend, true,
               "With the hue-preserving shoulder: preserve the shipped per-channel midtone grade in shadows and lower midtones, fading to a hue-stable neutral luminance grade in highlights (FaithfulLuma). When disabled, the per-channel midtone pow applies everywhere (shipped behavior).");
    RTX_OPTION("rtx.tonemap.ue3", float, gradePreservePivot, 0.16f,
               "Tonemapped luminance where the per-channel grade starts fading toward the neutral grade.");
    RTX_OPTION("rtx.tonemap.ue3", float, gradePreserveSlope, 1.75f,
               "How quickly the per-channel grade fades above the pivot.");
    RTX_OPTION("rtx.tonemap.ue3", bool, whiteNeutrality, true,
               "Neutralize bright, near-neutral display whites toward their own luminance (removes warm casts revealed in clipped sunlit whites) with display-energy protection (FaithfulLuma).");
    RTX_OPTION("rtx.tonemap.ue3", float, whiteLumaStart, 0.72f,
               "Display luminance where the whites correction starts fading in.");
    RTX_OPTION("rtx.tonemap.ue3", float, whiteLumaRange, 0.28f,
               "Display luminance range over which the whites correction reaches full strength.");
    RTX_OPTION("rtx.tonemap.ue3", float, whiteChromaStart, 0.04f,
               "Relative chroma below which a bright pixel counts as fully white.");
    RTX_OPTION("rtx.tonemap.ue3", float, whiteChromaRange, 0.30f,
               "Relative chroma range over which the whites correction fades out.");
    RTX_OPTION("rtx.tonemap.ue3", float, whiteNeutralityStrength, 0.85f,
               "Maximum blend toward luminance-neutral white.");

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
    RTX_OPTION("rtx.tonemap.ue3", bool, manualUseRec709LumaWeights, true,
               "Use Rec.709 luminance weights (0.2126/0.7152/0.0722) for the manual desaturation grade like FaithfulLuma; disable for the shipped 0.3/0.59/0.11 weights. Only affects the manual constants path - captured constants carry the game's own weights verbatim.");
  };

}
