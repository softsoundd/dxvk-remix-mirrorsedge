/*
* Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_objectpicking.h"
#include "rtx_resources.h"

#include "../spirv/spirv_code_buffer.h"
#include "../util/util_matrix.h"
#include "rtx_options.h"

namespace dxvk {

  class DxvkDevice;
  class DxvkPipelineManager;
  class RtCamera;

  enum class MotionBlurMode : int {
    // The original per-pixel gather. Kept bit-for-bit so it stays available for comparison.
    Legacy = 0,
    // Tile-based feature-aware reconstruction filter, Guertin/McGuire/Nowrouzezahrai HPG 2014.
    Cinematic = 1,
  };

  enum class MotionBlurDirectionSplit : int {
    Dominant = 0,
    Even = 1,
    Variance = 2,
  };

  enum class MotionBlurDebugView : int {
    Off = 0,
    Velocity = 1,
    TileMax = 2,
    NeighborMax = 3,
    Variance = 4,
    BlurAmount = 5,
  };

  class DxvkPostFx {
  public:
    DxvkPostFx(DxvkDevice* device);
    ~DxvkPostFx();

    // Registers the frame path's post FX compute shaders for pipeline prewarming. The
    // highlighting shader is a development picking feature and is left to compile on first use.
    void prewarmShaders(DxvkPipelineManager& pipelineManager) const;

    // Individual motion blur inputs; the path traced pipeline packs these from the
    // raytracing output bundle, the NGX passthrough mode from its synthesized equivalents.
    // All pointers must be valid; color extents may differ from the G-buffer inputs
    // (mainCameraResolution describes the motion vector / flags / view-Z resolution).
    // Screen space positions of a surface point half a frame either side of now, expressed
    // as matrices mapping current unjittered NDC to clip space at those instants. The
    // cinematic filter fits its curved sample path through them; when the camera history is
    // not usable this frame the path degrades to a straight line.
    struct MotionBlurCurveMatrices {
      Matrix4 toPrevHalfClip = Matrix4();
      Matrix4 toNextHalfClip = Matrix4();
      bool valid = false;
    };

    struct MotionBlurInputs {
      const Resources::Resource* inOutColor;              // blurred in place (via intermediateColor)
      const Resources::Resource* intermediateColor;       // scratch, same extent/format as inOutColor
      const Resources::Resource* screenSpaceMotionVector; // RG float, pixels, current -> previous
      const Resources::Resource* surfaceFlags;            // R8_UINT motion blur surface flags
      const Resources::AliasedResource* surfaceFlagsScratch1; // R8_UINT, prefilter ping
      const Resources::AliasedResource* surfaceFlagsScratch2; // R8_UINT, prefilter pong
      const Resources::Resource* linearViewZ;             // R32F linear view-space Z

      // Cinematic mode intermediates. All must be present for the mode to engage; when any
      // is missing the dispatch falls back to Legacy rather than failing.
      const Resources::Resource* cineVelocityDepth = nullptr;  // RGBA16F @ display res
      const Resources::Resource* cineCurvature = nullptr;      // RG16F @ display res
      const Resources::Resource* cineTileMaxX = nullptr;       // RGBA16F, tiles x height
      const Resources::Resource* cineTileMax = nullptr;        // RGBA16F, tiles x tiles
      const Resources::Resource* cineNeighborMax = nullptr;    // RGBA16F, tiles x tiles

      // Previous frame's motion vectors, used to recover per-object curvature. Optional.
      const Resources::Resource* previousScreenSpaceMotionVector = nullptr;

      MotionBlurCurveMatrices curves;
      float nearPlane = 0.0f;
      float farPlane = 0.0f;
    };

    // Motion blur phase. Runs before tonemapping while the image is still in linear HDR space.
    // Reads m_finalOutput, writes back to m_finalOutput (via intermediate texture).
    void dispatchMotionBlur(
      Rc<RtxContext> ctx,
      Rc<DxvkSampler> nearestSampler,
      Rc<DxvkSampler> linearSampler,
      const uvec2& mainCameraResolution,
      const uint32_t frameIdx,
      const Resources::RaytracingOutput& rtOutput,
      const bool cameraCutDetected);

    // Motion blur on explicitly provided inputs (see MotionBlurInputs)
    void dispatchMotionBlur(
      Rc<RtxContext> ctx,
      Rc<DxvkSampler> nearestSampler,
      Rc<DxvkSampler> linearSampler,
      const uvec2& mainCameraResolution,
      const uint32_t frameIdx,
      const MotionBlurInputs& inputs,
      const bool cameraCutDetected);

    // Lens effects phase (chromatic aberration + vignette). Runs after tonemapping
    // so it operates on post-tonemap LDR data — these are display-space lens artifacts.
    // Reads and writes m_finalOutput in place.
    void dispatchLensEffects(
      Rc<RtxContext> ctx,
      Rc<DxvkSampler> linearSampler,
      const uvec2& mainCameraResolution,
      const uint32_t frameIdx,
      const Resources::RaytracingOutput& rtOutput);

    // Lens effects on an explicitly provided color (blurred in place via the intermediate)
    void dispatchLensEffects(
      Rc<RtxContext> ctx,
      Rc<DxvkSampler> linearSampler,
      const uvec2& mainCameraResolution,
      const uint32_t frameIdx,
      const Resources::Resource& inOutColor,
      const Resources::Resource& intermediateColor);

    void dispatchHighlighting(
      Rc<RtxContext> ctx,
      const Resources::RaytracingOutput& rtOutput,
      std::vector<uint32_t>&& objectPickingValuesToHighlight,
      const std::optional<Vector2i>& pixelToHighlight,
      HighlightColor color);

    void showImguiSettings();

    inline bool isPostFxEnabled() const { return enable(); }
    inline bool isMotionBlurEnabled() const {
      if (!enable() || !enableMotionBlur()) {
        return false;
      }
      if (motionBlurMode() == MotionBlurMode::Cinematic) {
        return motionBlurCineSampleCount() > 0 && motionBlurShutterAngle() > 0.0f;
      }
      return motionBlurSampleCount() > 0 && exposureFraction() > 0.0f;
    }
    inline bool isChromaticAberrationEnabled() const { return enable() && enableChromaticAberration() && chromaticAberrationAmount() > 0.0f; }
    inline bool isVignetteEnabled() const { return enable() && enableVignette() && vignetteIntensity() > 0.0f; }

    RTX_OPTION_ARGS("rtx.postfx", bool, enable, true, "Enables post-processing effects.",
                    args.environment = "RTX_POST_FX_ENABLE",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.postfx", bool, enableMotionBlur, true, "Enables motion blur post-processing effect.",
                    args.environment = "RTX_POST_FX_MOTION_BLUR_ENABLE",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.postfx", bool, enableChromaticAberration, true, "Enables chromatic aberration post-processing effect.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.postfx", bool, enableVignette, true, "Enables vignette post-processing effect.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION("rtx.postfx", bool, desaturateOthersOnHighlight, true, "If true, desaturare all objects that are not highlighted.");
    RTX_OPTION_ARGS("rtx.postfx", MotionBlurMode, motionBlurMode, MotionBlurMode::Legacy,
                    "Motion blur algorithm. Legacy is the original per-pixel gather. Cinematic is a tile based "
                    "feature-aware reconstruction filter (Guertin, McGuire & Nowrouzezahrai, HPG 2014) which "
                    "dilates velocity across tiles so moving objects smear onto stationary background, samples "
                    "two directions to resolve overlapping motion, and can follow curved sample paths.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.postfx", float, motionBlurShutterAngle, 180.0f,
                    "Cinematic motion blur: shutter angle in degrees, following the film convention where 180 "
                    "means the shutter is open for half of each frame. Because motion vectors already describe "
                    "one frame of displacement, the blur length tracks the current frame rate: the same real "
                    "motion blurs half as far at twice the frame rate, which is the motion actually lost between "
                    "those two frames. Replaces exposureFraction, which only applies to Legacy mode.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION("rtx.postfx", float, motionBlurShutterReferenceFps, 0.0f,
               "Cinematic motion blur: when above zero, scale the shutter as though the frame rate were this "
               "value, holding the blur at a fixed length instead of letting it track the current frame rate "
               "(equivalent to Unreal's r.MotionBlur.TargetFPS). Off by default, since a fixed reference makes "
               "the blur overshoot the actual frame-to-frame displacement at higher frame rates and trail "
               "unnaturally. Useful where a frame-rate-invariant look is wanted, such as cutscenes.");

  private:
    // Builds the argument blocks and runs setup -> TileMax -> NeighborMax -> gather. The
    // surface flag prefilter has already run when this is called.
    void dispatchMotionBlurCinematic(
      Rc<RtxContext> ctx,
      Rc<DxvkSampler> linearSampler,
      const uvec2& mainCameraResolution,
      const uint32_t frameIdx,
      const MotionBlurInputs& inputs);

    Rc<vk::DeviceFn> m_vkd;
    Rc<DxvkBuffer> m_highlightingValues;
    // Setup pass arguments live in a uniform buffer: the two curve matrices alone fill the
    // 128 byte push constant budget (MaxPushConstantSize).
    Rc<DxvkBuffer> m_motionBlurCineConstants;

    RTX_OPTION("rtx.postfx", bool,  enableMotionBlurNoiseSample, true, "Enable random distance sampling for every step along the motion vector. The random pattern is generated with interleaved gradient noise.");
    RTX_OPTION("rtx.postfx", bool,  enableMotionBlurEmissive, true, "Enable Motion Blur for Emissive surfaces. Disable this when the motion blur on emissive surfaces cause severe artifacts.");
    RTX_OPTION("rtx.postfx", bool,  enableMotionBlurViewModel, false,
               "Enable Motion Blur for view-model (first-person) surfaces.\n"
               "Excluded by default: a held item that tracks the camera has near-zero real screen motion and blur on it usually looks wrong.\n"
               "Games with strongly animated first-person meshes (e.g. sprinting arms) benefit from enabling this; use rtx.postfx.motionBlurMaskOutTextures to opt out specific materials.");
    RTX_OPTION("rtx.postfx", uint,  motionBlurSampleCount, 4, "The number of samples along the motion vector. More samples could help to reduce motion blur noise.");
    RTX_OPTION("rtx.postfx", float, exposureFraction, 0.4f, "Simulate the camera exposure, the longer exposure will cause stronger motion blur.");
    RTX_OPTION("rtx.postfx", float, blurDiameterFraction, 0.02f, "The diameter of the circle that motion blur samplings occur. Motion vectors beyond this circle will be clamped.");
    RTX_OPTION("rtx.postfx", float, motionBlurMinimumVelocityThresholdInPixel, 1.0f, "The minimum motion vector distance that enable the motion blur. The unit is pixel size.");
    RTX_OPTION("rtx.postfx", float, motionBlurDynamicDeduction, 1.0f, "The deduction of motion blur for dynamic objects.");
    RTX_OPTION("rtx.postfx", float, motionBlurJitterStrength, 0.6f, "The jitter strength of every sample along the motion vector.");

    // Cinematic motion blur. Paper symbols are noted against each parameter; the reference
    // configuration is {N, r, tau, k, phi, gamma} = {35, 40, 1, 40, 27, 1.5}.
    //
    // N is 32 rather than the paper's 35 only to keep it a round number: with the samples
    // split evenly between two directions this is 16 per direction, which is where a
    // single-direction filter of this family stops showing banding.
    RTX_OPTION("rtx.postfx", uint, motionBlurCineSampleCount, 32,
               "Cinematic motion blur: samples per pixel (N), split between the two sampling directions. "
               "Higher trades performance for less noise.");
    RTX_OPTION("rtx.postfx", float, motionBlurCineMaxRadiusFraction, 0.021f,
               "Cinematic motion blur: maximum blur radius as a fraction of screen width, which also sets the "
               "tile size. The reference 40 pixels at 1920 wide is 0.021.");
    RTX_OPTION("rtx.postfx", float, motionBlurCineCenterWeightBias, 40.0f,
               "Cinematic motion blur: centre tap weight bias (k). Lower keeps unblurred detail stronger; the "
               "term is normalised by sample count so thin features survive at any N.");
    RTX_OPTION("rtx.postfx", float, motionBlurCineJitterScale, 27.0f,
               "Cinematic motion blur: sample jitter scale (phi). Scaled down by the sample count, so raising N "
               "reduces noise without reintroducing banding.");
    RTX_OPTION("rtx.postfx", float, motionBlurCineTileBlendSlope, 1.0f,
               "Cinematic motion blur: tile boundary blend slope (tau). Controls how far from a tile border the "
               "dominant velocity lookup starts being stochastically borrowed from the neighbour.");
    RTX_OPTION("rtx.postfx", float, motionBlurCineGammaThreshold, 1.5f,
               "Cinematic motion blur: velocity threshold (gamma) below which the second sampling direction "
               "falls back to the perpendicular of the dominant velocity.");
    RTX_OPTION("rtx.postfx", MotionBlurDirectionSplit, motionBlurCineDirectionSplit, MotionBlurDirectionSplit::Even,
               "Cinematic motion blur: how samples are split between the dominant and centre directions. "
               "Dominant matches single-direction filters, Even splits evenly, Variance allocates by how much "
               "the tile neighbourhood disagrees about direction.");
    // The three below extend past the published filter. Each estimates something the
    // available data does not directly supply, so each is separately toggleable and the base
    // mode with all of them off is exactly Guertin 2014.
    RTX_OPTION("rtx.postfx", bool, motionBlurCineCurvedPaths, true,
               "Cinematic motion blur: follow curved sample paths derived from the camera, so fast turns arc "
               "instead of smearing along a straight line.");
    RTX_OPTION("rtx.postfx", bool, motionBlurCineObjectCurvature, true,
               "Cinematic motion blur: additionally recover per-object curvature from the previous frame's "
               "motion vectors, covering rotating and swinging geometry rather than camera motion alone. "
               "Requires curved sample paths.");
    RTX_OPTION("rtx.postfx", MotionBlurDebugView, motionBlurCineDebugView, MotionBlurDebugView::Off,
               "Cinematic motion blur: visualise an intermediate of the filter instead of the blurred image.");
    RTX_OPTION("rtx.postfx", float, chromaticAberrationAmount, 0.02f, "The strength of chromatic aberration.");
    RTX_OPTION("rtx.postfx", float, chromaticCenterAttenuationAmount, 0.975f, "Control the amount of chromatic aberration effect that attunuated when close to the center of screen.");
    RTX_OPTION("rtx.postfx", float, vignetteIntensity, 0.6f, "The darkness of vignette effect.");
    RTX_OPTION("rtx.postfx", float, vignetteRadius, 0.8f, "The radius that vignette effect starts. The unit is normalized screen space, 0 represents the center, 1 means the edge of the short edge of the rendering window. So, this setting can larger than 1 until reach to the long edge of the rendering window.");
    RTX_OPTION("rtx.postfx", float, vignetteSoftness, 0.2f, "The gradient that the color drop to black from the vignetteRadius to the edge of rendering window.");
  };

  // Builds the pair of matrices the cinematic filter fits its curved sample path through, by
  // interpolating and extrapolating the camera's own pose history. previousSpaceFromCurrent
  // corrects for games whose world space shifts between frames; pass identity otherwise.
  DxvkPostFx::MotionBlurCurveMatrices buildMotionBlurCurveMatrices(
    const RtCamera& camera,
    const Matrix4d& previousSpaceFromCurrent = Matrix4d());

}
