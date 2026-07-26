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

#include <memory>
#include <vector>

#include "rtx_resources.h"
#include "rtx_common_object.h"
#include "rtx_options.h"
#include "../util/util_matrix.h"

namespace dxvk {
  class DxvkDevice;
  class DxvkBarrierSet;
  class DxvkPipelineManager;
  class RtxContext;
  class RtCamera;
  class NGXDLSSContext;
  struct DxvkContextState;

  // Skinned velocity limits, shared by the D3D9 capture and the raster pass: UE3's D3D9
  // GPU skin palette spans at most 75 bones x 3 registers (the capture pads every palette
  // to this count so the raster's palette layout is uniform); the per-frame draw cap
  // bounds the bone upload buffer.
  constexpr uint32_t kNgxVelocityBonePaletteRegisters = 225;
  // A crowd puts far more skinned draws on screen than there are characters, since each one
  // renders as several sections (body, head, attachments). A cap of 64 was overrun by roughly
  // half as much again with a handful of characters in view, and a skinned draw that misses the
  // cap is a character that loses its velocity for the frame.
  constexpr uint32_t kNgxVelocityMaxSkinnedDraws = 192;

  // CPU-modified mesh limits (UE3 CPU-skins morph/cloth-augmented skeletal meshes into
  // dedicated dynamic buffers, e.g. the first person arms): per-frame draw cap and
  // per-draw vertex cap bound the previous-position upload buffer.
  // A character's cloth is several draws on its own, and a scene holds more than one character.
  constexpr uint32_t kNgxVelocityMaxDynamicDraws = 16;
  constexpr uint32_t kNgxVelocityMaxDynamicVertices = 32768;

  // One dynamic object draw captured by the D3D9 layer for the object velocity pass: the
  // geometry references plus clip transforms for the current and previous frame, composed
  // from the reconstructed camera and the disambiguated LocalToWorld in the same
  // convention the motion vector pass's reprojection uses. Skinned draws (UE3 GPU skin)
  // additionally carry both frames' bone palettes and the blend attribute layout; their
  // motion is the composition of skinning delta and rigid transform delta.
  struct NgxVelocityDraw {
    DxvkBufferSlice vertexBuffer;
    uint32_t vertexStride = 0;
    uint32_t positionOffset = 0;
    VkFormat positionFormat = VK_FORMAT_R32G32B32_SFLOAT;

    DxvkBufferSlice indexBuffer;
    VkIndexType indexType = VK_INDEX_TYPE_UINT16;

    uint32_t indexCount = 0;
    uint32_t firstIndex = 0;
    int32_t vertexOffset = 0;

    Matrix4 clipFromLocal;
    Matrix4 prevClipFromLocal;

    // Skinned draws: bone palettes (registers as uploaded, 3 per bone: 3x4 rows with the
    // translation in .w), both padded to kNgxVelocityBonePaletteRegisters. Empty = rigid
    // draw. Blend attributes live on stream 0 (UBYTE4 indices, UBYTE4N weights - the UE3
    // D3D9 GPU skin layout). boneIndexScale: 3 for raw bone indices (the game's shader
    // scales them), 1 for indices pre-scaled at mesh build (detected from the bytecode).
    std::vector<Vector4> bonesPrevious;
    std::vector<Vector4> bonesCurrent;
    uint32_t blendIndicesOffset = 0;
    uint32_t blendWeightsOffset = 0;
    uint32_t boneIndexScale = 3;
    // Influences the game's shader variant consumes (1 = rigid-skin, implicit weight 1;
    // 2/4 = soft variants); the replay must not blend components the game ignores
    uint32_t skinInfluenceCount = 4;

    // Rendered after the game's mid-scene depth clear (UE3 foreground DPG: first person
    // meshes): depth-tested against the live foreground depth and phase-marked in the
    // velocity target so consumption cannot cross scene phases
    bool foregroundPhase = false;

    // The draw-time viewport depth range: the foreground DPG renders with a squashed
    // range in some configurations (occlusion culling disabled; keeps first person meshes
    // out of the world's depth precision), and the raster must map its depth identically
    // for the two-sided depth match
    float viewportMinZ = 0.0f;
    float viewportMaxZ = 1.0f;

    // CPU-modified meshes (UE3 CPU skinning: bone-less shader, dedicated dynamic vertex
    // buffer rewritten per frame): the motion lives in the vertex positions, so the
    // capture snapshots the previous frame's positions (tightly packed float3 per
    // vertex, baseVertexIndex-relative) and the raster reads them as a second stream.
    // Empty = the current positions serve both sides (rigid/skinned draws).
    std::vector<Vector3> previousPositions;
  };

  // The colour transform of the engine composite that Super Resolution replaces, captured from
  // that draw's own constants (UE3 GammaCorrectionPixelShader). Identity when the engine's
  // composite was already a plain copy.
  struct NgxOutputTransform {
    float colorScale[3] = { 1.0f, 1.0f, 1.0f };
    float overlayColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float inverseGamma = 1.0f;
    bool enabled = false;
  };

  // Per-frame diagnostics of the D3D9-side dynamic draw capture, shown in the developer
  // menu. Healthy steady state: captured tracks the number of moving objects on screen
  // (capturedSkinned of them skeletal, capturedForeground of them in the foreground DPG
  // phase - must be nonzero whenever first person meshes render in SDPG_Foreground);
  // newRegistrations ~0 (first sightings and claims without emission - spikes during
  // motion mean sightings are failing to pair with their history); the skip counters ~0
  // (each names the gate that rejected otherwise-qualifying scene draws); transpose
  // flips 0 (the camera packing-convention tiebreaker must be stable).
  struct NgxVelocityCaptureStats {
    uint32_t captured = 0;
    uint32_t capturedSkinned = 0;
    uint32_t capturedDynamic = 0;
    uint32_t capturedForeground = 0;
    uint32_t exactMatches = 0;
    // Sightings that found nothing to pair with, so produced no velocity: a one-frame dropout each.
    // Broken down by why - no sighting of that placement last frame at all, or one that was there
    // but too far in translation or in rotation to be believed the same object having moved.
    uint32_t newRegistrations = 0;
    // Of those, the skinned ones. The total is dominated by instanced level geometry, where a
    // placement's nearest sibling is far enough away to refuse and emitting nothing is the right
    // answer anyway; a skinned miss is a character or an attachment losing a frame of velocity.
    uint32_t newRegistrationsSkinned = 0;
    uint32_t missNoLastFrameSighting = 0;
    uint32_t missBeyondTranslation = 0;
    uint32_t missBeyondRotation = 0;
    // Sightings that paired but whose motion the emission gate would not vouch for
    uint32_t claimedWithoutVelocity = 0;
    // Sightings paired only by elimination, beyond the per-frame motion bounds: a fast mover
    // sustaining its delta, or one placement of an asset arriving as another leaves. Held to
    // the repeat-confirmation path, so a steady nonzero count with content that visibly moves
    // means real movers are being made to wait a frame; spikes as the camera turns are swaps
    // being refused.
    uint32_t pairedBeyondBounds = 0;
    // Sightings left untracked because every instance slot for their identity was already in
    // use by a placement seen this frame or last. Nonzero means an asset has more copies on
    // screen than kMaxInstancesPerIdentity holds: harmless for the static geometry that is
    // usually instanced that heavily, but a mover among them would go without velocity.
    uint32_t skippedInstanceCap = 0;
    // Scene draws on dynamic buffers that did not match the CPU-modified-mesh shape, so nothing was
    // captured for them - ring-pool geometry, whose allocations move every frame. A mesh that
    // carries its motion in its vertex positions and lands here goes without velocity entirely.
    uint32_t skippedDynamicBuffer = 0;
    // Skinned draws whose bone palette does not fit kNgxVelocityBonePaletteRegisters. Nonzero means
    // a rig richer than the cap allows, and every mesh using it goes without velocity.
    uint32_t skippedBonePalette = 0;
    uint32_t skippedNoCamera = 0;
    uint32_t skippedBudget = 0;
    // Scene draws with the CPU-modified-mesh buffer shape (dedicated dynamic VB, static
    // IB) rejected by the depth-test-enable gate: nonzero means such meshes render with
    // z-test disabled in some game state and are invisible to the capture
    uint32_t skippedZDisabled = 0;
    bool frameCameraValid = false;
    // Qualifying mid-scene depth clears this frame: must be nonzero whenever first person
    // meshes render in SDPG_Foreground (or their draws get misclassified as world-phase).
    // 1 with occlusion culling disabled (the foreground DPG boundary); 2 with it enabled
    // (the engine issues an extra mid-scene clear ahead of the foreground DPG).
    uint32_t depthClears = 0;
    uint32_t cameraTransposeFlips = 0;
  };

  /**
   * \brief NGX passthrough mode
   *
   * Presents the game's own rasterized rendering (the path tracer is bypassed entirely)
   * while still running DLSS Super Resolution / DLAA, DLSS Frame Generation and Reflex.
   *
   * The injection point is preferably before the game's post-process chain (DLSS consumes
   * the linear HDR scene color right when the first post pass would sample it, and the post
   * chain then operates on the anti-aliased result - see rtx.ngxPassthrough.prePostProcess),
   * falling back to scene end (post-processed LDR output, before UI). This pass:
   *  1. merges the game's hardware depth (pre-clear world snapshot + foreground depth, the
   *     latter live or itself a snapshot when a second mid-scene clear wiped it) into an
   *     R32F texture,
   *  2. synthesizes screen space motion vectors by reprojecting that depth through the
   *     current and previous frame cameras, overridden with true object motion (rigid,
   *     skinned or CPU-modified) where the object velocity raster captured a dynamic
   *     object (see rasterizeObjectVelocities),
   *  3. evaluates DLSS on the game's color target using those inputs plus the sub-pixel
   *     jitter that the D3D9 layer applied to the game's viewport (see D3D9Rtx, including
   *     the ScreenPositionScaleBias compensation that keeps the game's own screen space
   *     lookups aligned with the jittered content),
   *  4. writes the result back to the game's color target - re-merging the game's original
   *     alpha at the pre-post point, where UE3's D3D9 path stores the scene depth that
   *     shadow projections and depth-based post effects read back - and mirrors it into the
   *     scene color surface when the post chain consumes a resolve copy, and
   *  5. hands the depth/motion vectors plus camera to DLSS Frame Generation for
   *     interpolation at present time.
   *
   * Reflex markers are placed by the existing present/submit logic and need no work here.
   */
  class RtxNgxPassthrough : public CommonDeviceObject {
  public:
    explicit RtxNgxPassthrough(DxvkDevice* device);
    ~RtxNgxPassthrough() override;

    void onDestroy() override;

    // Registers the passthrough compute shaders (motion vector synthesis, alpha merge) for
    // pipeline prewarming. The velocity raster's vertex/fragment shaders are graphics-stage
    // and compile with their full pipeline state on first use instead.
    void prewarmShaders(DxvkPipelineManager& pipelineManager) const;

    // Runs the passthrough pipeline at the injection point. targetImage is the game's color
    // target DLSS reads from and writes back to, sceneDepthImage the depth-stencil image
    // identified by the D3D9 layer as holding the scene depth this frame (may be null, in
    // which case DLSS/DLFG are skipped and the frame passes through untouched).
    //
    // prePostProcess: the injection fired at the first post-process pass sampling the scene
    // color; targetImage is the game's linear (typically HDR) scene color and sourceSubrect
    // carries the scene viewport rect inside it. DLSS runs as DLAA within that rect and the
    // game's post chain consumes the result. When false, targetImage is the final output
    // (scene end, pre-UI) holding the post-processed LDR scene.
    //
    // upscaleSourceImage/sourceSubrect describe the ScreenPercentage upscaling path: when
    // the game renders its scene into a reduced subrect (UE3 ScreenPercentage < 100), the
    // D3D9 layer suppresses the engine's bilinear stretch onto the primary target and hands
    // over the stretch's source texture plus the scene subrect; DLSS then performs the
    // upscale to the target resolution instead (true Super Resolution). When
    // upscaleSourceImage is null the mode operates as DLAA at the target resolution.
    // colorSourceOffset is where that reduced colour sits inside upscaleSourceImage. It matches
    // sourceSubrect's offset whenever both live in scene color space, but when the runtime owns
    // the upscale the source is the backbuffer, where the engine centred the view rect while the
    // depth subrect stayed where the scene rasterized.
    // mirrorTargetImage (optional, pre-post-process only): a second image that receives the
    // DLSS output as well - the scene color render surface when the post chain consumes a
    // resolved copy, so any later surface->texture re-resolves propagate the result.
    void dispatch(RtxContext* ctx,
                  DxvkContextState& dxvkCtxState,
                  DxvkBarrierSet& barriers,
                  const Rc<DxvkImage>& targetImage,
                  const Rc<DxvkImage>& mirrorTargetImage,
                  const Rc<DxvkImage>& sceneDepthImage,
                  const Rc<DxvkImage>& upscaleSourceImage,
                  const VkRect2D& sourceSubrect,
                  const VkOffset2D& colorSourceOffset,
                  const NgxOutputTransform& outputTransform,
                  bool prePostProcess,
                  const std::vector<NgxVelocityDraw>& velocityDraws,
                  const float jitter[2],
                  bool resetHistory);

    void showImguiSettings();

    static float screenPercentageForUpscaler(uint32_t displayWidth, uint32_t displayHeight);
    static const char* upscalerModeLabel();

    // Whether the selected passthrough upscaler needs sub-pixel viewport jitter (temporal upscalers only).
    static bool needsViewportJitter(DxvkDevice* device);
    // renderHeight/displayHeight scale the phase count with the upscale ratio, as DLSS requires
    // and as RtCamera already does on the path traced path. Pass 0 when they are not known yet.
    static uint32_t viewportJitterSequenceLength(DxvkDevice* device,
                                                 uint32_t renderHeight = 0,
                                                 uint32_t displayHeight = 0);

    // Multisampled depth cannot be resolved for the NGX inputs, so the D3D9 layer refuses MSAA
    // outright (capability queries and resource creation) rather than negotiating it with the
    // game's own settings.
    static bool forceGameMsaaOff() {
      return ngxPassthroughMode() && disableGameMsaa();
    }

    /**
      * \brief: Draws the frame's held-back debug view over the presented image.
      *
      * Neither injection point can host the view in place: at the pre-post point the game's post
      * chain still has to run over that image and would mangle it, and at the late point the game
      * keeps drawing to the backbuffer afterwards and overwrites it. Overlaying at present also
      * leaves the frame rendering exactly as it normally does, upscaler included, so the view
      * shows the configuration being diagnosed rather than one altered to make room for it.
      *
      * Called from the swapchain on its own context - going through the device's command stream
      * puts the blit after the present flush, where it never reaches the presented image - and
      * ahead of the present blit, so the Remix UI stays on top and usable.
      */
    void blitDebugOverlayToPresent(DxvkContext* ctx, const Rc<DxvkImage>& targetImage);

    // Like screenPercentageForUpscaler(), but uses cached XeSS optimal input size when available
    // so the game's ScreenPercentage tracks the SDK rather than hardcoded fallback factors.
    float screenPercentageForDisplay(uint32_t displayWidth, uint32_t displayHeight);

    // Keeps DxvkXeSS::m_inputSize in sync for the UI when the path-traced frame path is not
    // running (NGX passthrough). Falls back to cached passthrough extents when display size is 0.
    void syncXeSSInputResolution(uint32_t displayWidth, uint32_t displayHeight);

    // syncXeSSInputResolution + getInputSize, for ImGui panels.
    void getXeSSInputResolution(uint32_t displayWidth, uint32_t displayHeight,
                                uint32_t& inputWidth, uint32_t& inputHeight);

    // Per-frame capture diagnostics from the D3D9 layer, stored for the developer menu
    void setVelocityCaptureStats(const NgxVelocityCaptureStats& stats) {
      m_lastCaptureStats = stats;
    }

    // How far the space the game's transforms are expressed in moved since the previous frame,
    // measured by the D3D9 capture. Titles uploading true world space report zero; where it is
    // nonzero the two frames' cameras describe different spaces, and the reprojection has to
    // cross between them (see generateMotionVectorsAndDepth).
    void setSceneTransformOffset(const Vector3& offset) {
      m_sceneTransformOffset = offset;
    }

    // Single status line (upscaler state + resolutions), shared between the developer panel and
    // the user menu (the stock DLSS object's state is meaningless while this mode is active)
    void showImguiStatusLine(bool includeInjectionPoint = true);

    // Copies the game's depth buffer into a runtime-owned snapshot. Called (via RtxContext)
    // right before the game's first mid-scene depth clear: UE3 clears the depth buffer
    // ahead of its foreground DPG (with occlusion culling enabled, an extra clear precedes
    // that one), which would otherwise destroy the world depth needed for motion vector
    // generation.
    void captureDepthSnapshot(RtxContext* ctx, const Rc<DxvkImage>& sceneDepthImage);

    // Copies the pre-UI backbuffer into this frame's HUD-less slot for frame generation
    // (see dlfgHudlessInput). Called (via RtxContext) at the scene-end/UI boundary: from
    // dispatch() directly at the late injection point, from the D3D9 layer's first
    // UI-classified backbuffer draw when the injection ran pre-post-process, or at frame
    // end when no UI was drawn at all (the presented frame is its own HUD-less copy then).
    void captureHudless(RtxContext* ctx, const Rc<DxvkImage>& backbufferImage);

    // Diagnostic: counts frames where the injection ran but the main camera was not valid,
    // so the passthrough dispatch was skipped entirely (feeds the periodic summary log)
    void noteCameraInvalidFrame() {
      m_statCameraInvalidCount++;
    }

    enum class DebugVisualization : int {
      Off = 0,
      MotionVectors = 1,
      Depth = 2,
      ObjectVelocityCoverage = 3,
    };

    // Path tracing is incompatible with NGX passthrough; keep rtx.enableRaytracing off.
    static void enforceRaytracingDisabledForPassthrough();
    static void ngxPassthroughModeOnChange(DxvkDevice* device);

    RTX_OPTION_ARGS("rtx", bool, ngxPassthroughMode, false,
               "Master toggle for the NGX passthrough mode: the game's original rasterized rendering is presented (no path tracing,\n"
               "no scene capture) while the selected upscaler (DLSS, NIS, TAA-U or XeSS), DLSS Frame Generation and Reflex run on top.\n"
               "Screen space motion vectors are synthesized from the game's depth buffer via camera reprojection, and sub-pixel\n"
               "camera jitter is injected through the game's viewport so temporal upscalers can accumulate detail.\n"
               "Takes precedence over rtx.enableRaytracing (which is forced off while this mode is active). Should be set at launch (game depth\n"
               "buffers are created with a shader-readable layout only when this mode is active).",
               args.onChangeCallback = &RtxNgxPassthrough::ngxPassthroughModeOnChange);
    RTX_OPTION("rtx.ngxPassthrough", bool, enableJitter, true,
               "Applies a sub-pixel Halton jitter offset to the game's viewport for all draws targeting the detected scene render\n"
               "target. Required for temporal upscalers (DLSS, TAA-U, XeSS) to anti-alias; only disable for debugging.");
    RTX_OPTION("rtx.ngxPassthrough", bool, prePostProcess, true,
               "Runs the upscaler on the game's linear scene color before the game's post-process chain instead of on the final\n"
               "post-processed output, matching a native engine integration: bloom, tonemapping and color grading then operate\n"
               "on the anti-aliased, unjittered image rather than baking into the upscaler input (post effects sampling a jittered\n"
               "scene wobble sub-pixel per frame, which temporal upscalers otherwise have to soften out). Also enables DLSS's HDR\n"
               "input mode when the scene color is a floating point target and DLSS is selected. The injection triggers on the\n"
               "first post-process pass sampling the identified scene color; frames where that never happens (e.g. the game's post\n"
               "chain is disabled) fall back to the late injection point automatically.");
    RTX_OPTION("rtx.ngxPassthrough", bool, objectVelocities, true,
               "Rasterizes true motion vectors for dynamic objects into the synthesized motion vector inputs, replacing the\n"
               "static-world camera reprojection where they land. Covers rigid movers (doors, elevators - draws whose transform\n"
               "changed since the previous frame), skinned meshes (first person arms/legs, characters - UE3 GPU skinning\n"
               "replayed with the current and previous frame's bone palettes) and CPU-modified meshes (tracked through vertex\n"
               "position snapshots). DLSS, Frame Generation and Remix motion blur then see real screen space motion instead\n"
               "of treating everything as static world. First person meshes are handled in both their world/intermediate and\n"
               "foreground render phases.");
    RTX_OPTION("rtx.ngxPassthrough", bool, motionBlurFirstPerson, true,
               "Applies Remix motion blur (rtx.postfx) to the camera-locked first person meshes like the rest of the scene:\n"
               "during sprints and camera motion the hands smear with their true motion, matching the vanilla game's camera\n"
               "motion blur. Disable to exclude them from blur like the path traced pipeline's view model exclusion\n"
               "(crisp hands).");
    RTX_OPTION("rtx.ngxPassthrough", bool, dlfgHudlessInput, true,
               "Provides DLSS Frame Generation with a HUD-less copy of the frame (captured between the end of the game's\n"
               "rendering and the first UI draw) in addition to the presented backbuffer. The interpolator then knows exactly\n"
               "which pixels are UI instead of detecting them heuristically, which strongly reduces UI warping over busy\n"
               "moving backgrounds.");
    RTX_OPTION("rtx.ngxPassthrough", bool, driveGameScreenPercentage, true,
               "Lets the Remix upscaler quality preset choose the game's effective render resolution.\n"
               "Full Resolution renders natively (DLAA for DLSS, 100% for NIS/TAA-U/XeSS); other tiers use the upscaler's\n"
               "standard scaling factors. In-memory only (saved game settings unchanged); when disabled, follows live\n"
               "game values. Requires rtx.ngxPassthroughMode and locatable game readers.");
    RTX_OPTION("rtx.ngxPassthrough", bool, disableGameMsaa, true,
               "Reports multisampling as unsupported to the game and creates every surface single-sampled, so no\n"
               "multisampled resource can exist. Required because multisampled depth cannot be resolved for NGX inputs.\n"
               "The game's saved MSAA setting is not modified; it simply has nothing to select. Requires\n"
               "rtx.ngxPassthroughMode.");
    RTX_OPTION("rtx.ngxPassthrough", int, jitterSequenceLength, 0,
               "Number of frames in the sub-pixel jitter loop. 0 derives it from the upscale ratio as\n"
               "8 * (display/render)^2, the value DLSS documents, and uses it unchanged up to a 2.5x ratio -\n"
               "so Quality, Balanced and Performance are unaffected. Beyond that it is clamped to 8 frames,\n"
               "which is a deliberate divergence: the documented rule assumes jitter in the projection, where\n"
               "it reaches every sample, while this mode offsets the viewport and whatever the offset misses\n"
               "is displaced when the upscaler removes it. A long loop makes the deepest offsets rare enough\n"
               "to escape accumulation and show as a shift with the loop's period, scaled into display pixels\n"
               "by the ratio, which is why only the most aggressive preset needs it. Set a value to override.");
    RTX_OPTION("rtx.ngxPassthrough", int, screenPercentageUpscaleOwner, 0,
               "Who turns the reduced render resolution back into a full resolution frame.\n"
               "0: Auto - let the engine do it when its own NeedsUpscale() readers could be redirected\n"
               "(the verified path), otherwise fall back to the runtime. 1: Engine. 2: Runtime, which\n"
               "upscales the reduced subrect the engine composited into the backbuffer and leaves the\n"
               "engine believing it renders natively.");
    RTX_OPTION("rtx.ngxPassthrough", int, systemSettingsScreenPercentageRva, 0,
               "Escape hatch for titles where GSystemSettings cannot be located automatically: the RVA of\n"
               "FSystemSettings::ScreenPercentage within the game's main module. 0 leaves discovery\n"
               "automatic. The value is still proven by the behavioural probe before it is used.");
    RTX_OPTION("rtx.ngxPassthrough", int, dlssRenderPreset, 10,
               "DLSS render preset (model selection) hint for the NGX feature. 0: Default (snippet/driver decides, typically an\n"
               "older CNN model), 1-6: presets A-F (CNN models), 10: preset J (transformer model - noticeably better detail\n"
               "preservation and temporal stability, slightly higher GPU cost). Values outside the known range fall back to\n"
               "Default. Changing this recreates the DLSS feature.");
    RTX_OPTION("rtx.ngxPassthrough", bool, objectVelocityDebugFreeze, false,
               "Diagnostic for the object velocity pass: composes every captured draw's previous-frame side from the\n"
               "current frame's transforms and bone palettes (forcing the rasterized velocity to zero) and disables the\n"
               "depth test (exposing the raw rasterized footprint). Coverage in the motion vector debug view then shows\n"
               "exactly where the replayed geometry lands with current-frame data alone: neutral coverage hugging the\n"
               "object silhouettes means the replay (transforms, skinning) is correct; any artifact seen without the\n"
               "freeze then comes from the depth-based visibility or the previous-frame history pairing.");
    RTX_OPTION("rtx.ngxPassthrough", bool, bypassUpscaler, false,
               "Diagnostic: keeps driving the game's render resolution from the selected upscaler's preset but does not\n"
               "run the upscaler, bringing the reduced render up to display resolution with a plain filtered stretch\n"
               "instead - the same reduced input, with and without the reconstruction. Setting rtx.upscalerType to None\n"
               "instead returns the game to rendering at full resolution.")
    RTX_OPTION("rtx.ngxPassthrough", float, objectVelocityDepthTolerance, 1.0f,
               "Diagnostic multiplier on the object velocity raster's depth match, which is 0.1% of view distance at\n"
               "1.0. An object that is captured and rasterized in the right place but never appears in the Object\n"
               "Velocity Coverage view is being discarded by that match; raising this until it appears measures how\n"
               "far its depth actually is from the game's, which distinguishes float rounding from a wrong transform.\n"
               "Values below 1 are ignored. Leave at 1 outside diagnosis: a loose match lets an occluded object's\n"
               "motion seep through whatever is in front of it.")
    RTX_OPTION("rtx.ngxPassthrough", int, debugVisualization, 0,
               "Debug visualization for the synthesized DLSS inputs. 0: Off, 1: Motion Vectors, 2: Depth,\n"
               "3: Object Velocity Coverage (the raw velocity raster output: green = world-phase coverage, red =\n"
               "foreground-phase coverage, dark = no rasterized velocity).\n"
               "In the motion vector mode the blue channel encodes each pixel's motion source, independent of motion\n"
               "magnitude: 0 = per-object velocity override (dynamic object coverage), 0.25 = foreground object velocity\n"
               "(first person meshes with true skinned motion), 0.5 = world camera reprojection, 1 = camera-locked\n"
               "foreground without object velocity.");
    RTX_OPTION("rtx.ngxPassthrough", int, dumpVelocityCaptureFrames, 0,
               "Diagnostic: when set to a value N > 0, every scene draw the object velocity capture examines over the\n"
               "next N frames is written to the log with the state its gates decide on - primitive count, bone palette\n"
               "size, buffer usage, depth write and compare, and the viewport against the scene's - then the value\n"
               "resets to 0 automatically. The developer menu has a button for this. Arm it while the object under\n"
               "investigation is on screen: an object whose velocity never reaches the screen is told from one whose\n"
               "does by which gate its draw fails.")
    RTX_OPTION("rtx.ngxPassthrough", int, dumpPostChainFrames, 0,
               "Diagnostic: when set to a value N > 0, the non-scene draw flow (post-process passes, composites, UI, resolve\n"
               "copies, render state) of the next N frames is written to the log, then the value resets to 0 automatically.\n"
               "The developer menu has a button for this. Used to validate the pre-post-process injection point against the\n"
               "game's real compositing chain.");

  private:
    void createResources(Rc<DxvkContext> ctx, const VkExtent2D& renderExtent, const VkExtent2D& displayExtent);
    bool generateMotionVectorsAndDepth(RtxContext* ctx,
                                       DxvkContextState& dxvkCtxState,
                                       DxvkBarrierSet& barriers,
                                       const Rc<DxvkImage>& sceneDepthImage,
                                       const VkOffset2D& subrectOffset,
                                       const RtCamera& camera,
                                       const std::vector<NgxVelocityDraw>& velocityDraws,
                                       const float jitter[2]);

    // Rasterizes the captured dynamic object draws (rigid and skinned) as NDC deltas over
    // a sentinel clear into m_objectVelocity, manually depth-tested: world-phase draws
    // against the world depth, foreground-phase draws against the live foreground depth,
    // marked via the blue-channel phase marker. Returns a bitmask of the phases that
    // rendered (bit 0 world, bit 1 foreground); the texture must not be sampled when zero.
    // subrectOffset/jitter: where the scene subrect sits in the depth buffer, and the
    // sub-pixel viewport jitter the game's own rasterization carried this frame - both are
    // replicated so the raster lands on the exact pixels (and depths) the game produced.
    uint32_t rasterizeObjectVelocities(RtxContext* ctx,
                                       DxvkContextState& dxvkCtxState,
                                       const std::vector<NgxVelocityDraw>& velocityDraws,
                                       const Rc<DxvkImageView>& worldDepthView,
                                       const Rc<DxvkImageView>& foregroundDepthView,
                                       const VkOffset2D& subrectOffset,
                                       const float jitter[2]);
    // preserveTargetAlpha: combine the DLSS RGB output with the color source's original
    // alpha before writing back (pre-post-process injection: UE3's D3D9 path stores the
    // scene depth in the scene color alpha channel, which shadow projection and depth-based
    // post effects read; DLSS output alpha is undefined).
    bool evaluateDlss(RtxContext* ctx,
                      DxvkBarrierSet& barriers,
                      const Rc<DxvkImage>& colorSourceImage,
                      const VkOffset2D& colorSourceOffset,
                      const Rc<DxvkImage>& targetImage,
                      const VkOffset2D& targetOffset,
                      bool preserveTargetAlpha,
                      const float jitter[2],
                      bool resetHistory);

    // Plain filtered stretch in the upscaler's place, for rtx.ngxPassthrough.bypassUpscaler
    bool evaluateUpscalerBypass(RtxContext* ctx,
                                DxvkBarrierSet& barriers,
                                const Rc<DxvkImage>& colorSourceImage,
                                const VkOffset2D& colorSourceOffset,
                                const Rc<DxvkImage>& targetImage,
                                const VkOffset2D& targetOffset,
                                bool preserveTargetAlpha,
                                bool resetHistory);

    bool evaluateNis(RtxContext* ctx,
                     DxvkBarrierSet& barriers,
                     const Rc<DxvkImage>& colorSourceImage,
                     const VkOffset2D& colorSourceOffset,
                     const Rc<DxvkImage>& targetImage,
                     const VkOffset2D& targetOffset,
                     bool preserveTargetAlpha,
                     bool resetHistory);

    bool evaluateTaau(RtxContext* ctx,
                      DxvkBarrierSet& barriers,
                      const Rc<DxvkImage>& colorSourceImage,
                      const VkOffset2D& colorSourceOffset,
                      const Rc<DxvkImage>& targetImage,
                      const VkOffset2D& targetOffset,
                      bool preserveTargetAlpha,
                      const float jitter[2],
                      bool resetHistory);

#ifndef _M_ARM64
    bool evaluateXess(RtxContext* ctx,
                      DxvkBarrierSet& barriers,
                      const Rc<DxvkImage>& colorSourceImage,
                      const VkOffset2D& colorSourceOffset,
                      const Rc<DxvkImage>& targetImage,
                      const VkOffset2D& targetOffset,
                      bool preserveTargetAlpha,
                      bool resetHistory);
#endif

    void snapshotColorInput(RtxContext* ctx,
                            const Rc<DxvkImage>& colorSourceImage,
                            const VkOffset2D& colorSourceOffset);

    bool applyPostFxAndWriteback(RtxContext* ctx,
                                 DxvkBarrierSet& barriers,
                                 const Rc<DxvkImage>& targetImage,
                                 const VkOffset2D& targetOffset,
                                 bool preserveTargetAlpha,
                                 bool resetHistory);

    // Runs Remix postfx on the scene color when no upscaler is selected or available.
    bool evaluatePostFxOnly(RtxContext* ctx,
                            DxvkBarrierSet& barriers,
                            const Rc<DxvkImage>& colorSourceImage,
                            const VkOffset2D& colorSourceOffset,
                            const Rc<DxvkImage>& targetImage,
                            const VkOffset2D& targetOffset,
                            bool preserveTargetAlpha,
                            bool resetHistory);

    void copyColorInputToOutput(RtxContext* ctx);

    // Combines m_dlssOutput RGB with m_colorInput alpha into m_mergedOutput
    void dispatchAlphaMerge(RtxContext* ctx, DxvkBarrierSet& barriers);

    // One feature per injection point (indexed by HDR content), so switching between them costs
    // nothing and neither loses its temporal history. Sharing one meant every switch rebuilt the
    // feature and reset the history, which the scene-dependent injection point does constantly.
    struct DlssFeature {
      std::unique_ptr<NGXDLSSContext> context;
      bool needsInitialize = true;
      int initializedRenderPreset = -1;
      uint32_t initializedInput[2] = { 0, 0 };
      uint32_t initializedOutput[2] = { 0, 0 };
    };
    DlssFeature m_dlssFeatures[2];
    int m_dlssLastContentHDR = -1;

    // Whether the selected upscaler evaluated successfully last frame, for the status line
    bool m_upscalerActive = false;
    bool m_postFxActive = false;

    // Per-dispatch write-back configuration, consumed by the merge pass.
    NgxOutputTransform m_outputTransform;
    bool m_preserveOriginalAlpha = false;

    // Status/diagnostics for the developer menu (written on the CS thread, read for display)
    const char* m_statusReason = "not dispatched yet";
    uint32_t m_lastDispatchFrameId = 0;
    uint32_t m_dlssInitCount = 0;
    float m_lastJitter[2] = { 0.0f, 0.0f };
    bool m_lastDispatchPrePost = false;

    // Rolling counters for the periodic diagnostic summary log
    uint32_t m_statDispatchCount = 0;
    uint32_t m_statUpscalerActiveCount = 0;
    uint32_t m_statCameraInvalidCount = 0;
    uint32_t m_statUpscalerOffCount = 0;
    uint32_t m_statNoInputsCount = 0;
    uint32_t m_statEvaluateFailedCount = 0;
    uint32_t m_statDebugVisCount = 0;
    uint32_t m_statResetHistoryCount = 0;
    uint32_t m_statPrePostCount = 0;
    // How often the colour source changed between HDR and LDR within the window. Not the same as
    // the injection point moving, which is what m_statPrePostCount against m_statDispatchCount
    // shows: both injection points can feed HDR content, so this can read zero while the point is
    // relocating on a third of frames.
    uint32_t m_statContentHdrFlips = 0;
    // Largest deviation of the camera reprojection matrix from identity in the window. Zero while
    // the view moves means camera reprojection is producing no motion at all.
    float m_statReprojectionFromIdentityMax = 0.0f;
    uint32_t m_statWindowStartFrameId = 0;

    VkExtent2D m_renderExtent = { 0, 0 };
    VkExtent2D m_displayExtent = { 0, 0 };

    Resources::Resource m_colorInput;                 // copy of the game color target fed to DLSS
    Resources::Resource m_dlssOutput;                 // DLSS output, blitted back onto the game target
    Resources::Resource m_mergedOutput;               // DLSS RGB + the game's original alpha (pre-post
                                                      // injection: UE3 D3D9 stores scene depth in alpha)
    Resources::Resource m_debugPresentImage;          // debug view held back for the present-time overlay
    bool m_debugPresentValid = false;                 // holds across frames that capture no new view
    Resources::ResourceQueue m_depthQueue;            // R32F depth, one slot per DLFG frame in flight
    Resources::ResourceQueue m_motionVectorQueue;     // RG16F pixel-space motion vectors

    // HUD-less copies of the pre-UI backbuffer for frame generation (see dlfgHudlessInput).
    // Created lazily to match the backbuffer format/extent; the current slot's view is
    // handed to the DLFG presenter at dispatch and filled later the same frame.
    Resources::ResourceQueue m_hudlessQueue;

    // Remix post-processing (rtx.postfx) inputs synthesized alongside the motion vectors:
    // linear view-space Z and the motion blur surface flags (camera-locked foreground =
    // view model), plus the prefilter scratch pair the motion blur pass ping-pongs
    Resources::Resource m_linearViewZ;
    Resources::Resource m_surfaceFlags;
    Resources::AliasedResource m_surfaceFlagsScratch1;
    Resources::AliasedResource m_surfaceFlagsScratch2;

    // Per-object velocity raster target (RGBA16F: RG = NDC delta over a sentinel clear,
    // B = phase ownership marker separating world and foreground DPG draws)
    Resources::Resource m_objectVelocity;

    // Bone palettes for the skinned velocity draws, re-uploaded per frame (per-draw
    // blocks: header + previous palette + current palette), and the identity-ramp slot
    // buffer each draw binds as an instance-rate attribute to select its block. Sized
    // for the per-frame skinned draw cap.
    Rc<DxvkBuffer> m_boneBuffer;
    Rc<DxvkBuffer> m_boneSlotsBuffer;
    bool m_boneSlotsInitialized = false;

    // Previous-frame position snapshots for CPU-modified mesh draws (second vertex stream)
    Rc<DxvkBuffer> m_dynamicPositionsBuffer;

    // Diagnostics: dynamic draws rasterized last frame + D3D9-side capture stats (imgui)
    uint32_t m_lastVelocityDrawCount = 0;
    NgxVelocityCaptureStats m_lastCaptureStats;
    Vector3 m_sceneTransformOffset = Vector3(0.0f, 0.0f, 0.0f);

    Rc<DxvkBuffer> m_constantsBuffer;
    Rc<DxvkBuffer> m_velocityRasterConstants;

    // Sampled depth-aspect view over the game's depth-stencil image
    Rc<DxvkImageView> m_gameDepthView;
    const DxvkImage* m_gameDepthViewImage = nullptr;

    // World depth snapshot, captured right before the game's first mid-scene depth clear
    // (see captureDepthSnapshot). Consumed by the next dispatch on the CS timeline; the
    // pending flag rather than a frame id comparison keeps consumption robust against
    // frame counter increments racing the CS thread.
    Rc<DxvkImage> m_depthSnapshotImage;
    Rc<DxvkImageView> m_depthSnapshotView;
    bool m_depthSnapshotPending = false;
  };
}
