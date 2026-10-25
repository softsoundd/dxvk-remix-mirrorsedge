#pragma once

#include "d3d9_state.h"
#include "../dxvk/dxvk_buffer.h"
#include "../util/util_threadpool.h"

#include <array>
#include <atomic>
#include <memory>
#include <vector>
#include <optional>
#include <unordered_map>

namespace dxvk {
  struct D3D9BufferSlice;
  class DxvkDevice;
  class D3D9CommonTexture;

  namespace PrepareDrawFlag {
    enum {
      Ignore                      = 0,
      ApplyDrawState              = 1 << 1, // Submit draw state to dxvk-cs, so it can be used to issue original or non-original draw calls
      OriginalDrawCall            = 1 << 2, // Issue the original draw call
      PreserveDrawCallAndItsState = ApplyDrawState | OriginalDrawCall,
    };
  }
  using PrepareDrawFlags = uint32_t;

  //This class handles all of the RTX operations that are required from the D3D9 side.
  struct D3D9Rtx {
    friend class ImGUI; // <-- we want to modify these values directly.

    D3D9Rtx(D3D9DeviceEx* d3d9Device);
    ~D3D9Rtx();

    RTX_OPTION("rtx", bool, orthographicIsUI, true, "When enabled, draw calls that are orthographic will be considered as UI.");
    RTX_OPTION("rtx", bool, preTransformedVerticesIsUI, false, "When enabled, draw calls using pre-transformed (screen-space) vertices will be considered as UI. This is typical for D3D8/D3D9 games that render UI with RHW vertices.");
    RTX_OPTION("rtx.d3d9", bool, ue3EngineMode, false,
               "Master toggle for Unreal Engine 3 D3D9 compatibility.");
    RTX_OPTION("rtx.d3d9", bool, ue3SkipSceneCapturePasses, false,
               "UE3 multi-pass compat: classify UE3 SceneCapture rendering (SceneCapture2D/Reflect/Portal actors: security "
               "monitors, mirrors, scripted window reflections) as capture passes, so a probe's camera never replaces the "
               "Main camera and its world draws are not treated as the main scene. A world draw is a capture pass when its "
               "viewport is strictly smaller than half the backbuffer in both dimensions, or when a sub-full viewport's "
               "aspect does not match the backbuffer (exact-half splitscreen viewports are never matched). "
               "Implicitly enabled by rtx.d3d9.ue3EngineMode.");
    RTX_OPTION_ARGS("rtx.d3d9", bool, sequenceTrackedLockWaits, true,
                    "When a Lock has to wait for a resource, drain the command stream thread only up to the last command "
                    "that touched it (tracked per buffer and texture subresource) instead of everything queued, which "
                    "includes the previous frame's injectRTX recording once the game runs ahead of the GPU. Same model as "
                    "current upstream DXVK. Disable to fall back to the full drain.",
                    args.flags = RtxOptionFlags::UserSetting);

    // Copy of the parameters issued to D3D9 on DrawXXX
    struct DrawContext {
      D3DPRIMITIVETYPE PrimitiveType;
      INT              BaseVertexIndex;
      UINT             MinVertexIndex;
      UINT             NumVertices;
      UINT             StartIndex;
      UINT             PrimitiveCount;
      BOOL             Indexed;
    };
    static_assert(sizeof(DrawContext) == 28, "Please, recheck initializer usages if this changes.");

    // rtx.d3d9.sequenceTrackedLockWaits
    bool SequenceTrackedLockWaitsEnabled() const {
      return m_frameOptions.sequenceTrackedLockWaits;
    }

    // True once this frame's ray tracing has been injected; later draws are UI / post work.
    bool IsRtxInjectTriggered() const {
      return m_rtxInjectTriggered;
    }

    /**
      * \brief: This function is responsible for preparing the geometry for rendering in Direct3D 9.
      *
      * \param [in] indexed: A boolean value indicating whether or not the geometry to be rendered is indexed.
      * \param [in] state : An object of type Direct3DState9 that contains the current state of the Direct3D pipeline.
      * \param [in] context : An object of type Draw that contains the context for the draw call.
      *
      * Returns false if this drawcall should be removed from further processing, returns true otherwise.
      */
    PrepareDrawFlags PrepareDrawGeometryForRT(const bool indexed, const DrawContext& context);

    /**
      * \brief: This function is responsible for preparing the geometry for rendering in Direct3D 9 
      *         when the vertex and index data is packed into a single buffer: ||VERTICES|INDICES||
      * 
      * \param [in] indexed: A boolean value indicating whether or not the geometry to be rendered is indexed.
      * \param [in] buffer : An object of type D3D9BufferSlice that contains the packed vertex and index data.
      * \param [in] indexFormat : The format of the indices in the buffer.
      * \param [in] indexOffset : The offset of the index data in bytes
      * \param [in] indexSize : The size of the index data in bytes.
      * \param [in] vertexSize : The size of the vertex data in bytes.
      * \param [in] vertexStride : The stride of the vertex data in bytes.
      * \param [in] drawContext : An object of type Draw that contains the context for the draw call.
      *
      * Returns false if this drawcall should be removed from further processing, returns true otherwise.
      */
    PrepareDrawFlags PrepareDrawUPGeometryForRT(const bool indexed,
                                                const D3D9BufferSlice& buffer,
                                                const D3DFORMAT indexFormat,
                                                const uint32_t indexSize,
                                                const uint32_t indexOffset,
                                                const uint32_t vertexSize,
                                                const uint32_t vertexStride,
                                                const DrawContext& context);

    /**
      * \brief: Signal that a swapchain has been resized or reconfigured.
      * 
      * \param [in] presentationParameters: A reference to the D3D present params.
      */
    void ResetSwapChain(const D3DPRESENT_PARAMETERS& presentationParameters);

    /**
      * \brief: Signal that we've reached the end of the frame.
      *
      * \param [in] countAsFrame: When false, per-frame accumulators are still reset but the
      * passthrough frame counter does not advance. Minimized presents use this so the
      * object-velocity cache does not age out while the window is iconic.
      */
    void EndFrame(const Rc<DxvkImage>& targetImage, bool callInjectRtx = true, bool countAsFrame = true);

    /**
      * \brief: While suspended, draws stay on the game's raster path. DLSS injection, jitter,
      * velocity capture, and depth snapshots do not run.
      */
    void SetPassthroughSuspended(bool suspended) {
      m_passthroughSuspended = suspended;
    }

    /**
      * \brief: Signal that we're about to present the image.
      */
    void OnPresent(const Rc<DxvkImage>& targetImage);

    /**
      * \brief: Increments the Reflex frame ID. Should be called after presentation and only after every Reflex related marker
      * call for the current frame (this typically means other threads running in parallel will need to cache this value from the
      * frame they were dispatched on).
      */
    void IncrementReflexFrameId() {
      ++m_reflexFrameId;
    }

    /**
      * \brief: Gets the Reflex frame ID for the current frame on the main thread. This is incremented after each present.
      * Only intended for use with Reflex, other methods for getting a frame ID exist which may make more sense for other systems.
      */
    uint64_t GetReflexFrameId() const {
      return m_reflexFrameId;
    }

    /**
      * \brief: NGX passthrough mode: returns the sub-pixel jitter offset (in pixels) to apply to
      * the Vulkan viewport of the current draw, or false when the draw must not be jittered.
      * Called from D3D9DeviceEx::BindViewportAndScissor on the application thread. Draws
      * rendering into the detected scene color target - and full-size screen space passes
      * depth/stencil-tested against the scene depth (light attenuation shadow projections,
      * distortion accumulation) - are jittered during the pre-injection (scene) phase of the
      * frame; UI and other offscreen passes are left untouched.
      */
    bool GetNgxPassthroughViewportJitter(float* pJitterX, float* pJitterY) const;

    /**
      * \brief: NGX passthrough mode: patches the shader stage's ScreenPositionScaleBias
      * constant in the mapped constant buffer copy (staged game state is never modified) so
      * clip-space-derived screen texture UVs follow the viewport-jittered content. Called
      * from D3D9DeviceEx::UploadConstantSet with the destination float constant array.
      */
    void PatchNgxScreenPositionScaleBias(DxsoProgramType stage, void* floatConstants, uint32_t floatConstantCount) const;

    // Viewports and scissors within this many pixels of the scene rect are treated as that rect.
    static constexpr uint32_t kNgxViewportMatchPixels = 4;
    // A scene viewport this close to backbuffer * ScreenPercentage / 100 is that rect.
    static constexpr int32_t kNgxDrivenViewportTolerance = 16;

    // Expands a viewport that matches the scene rect out to the bound render target.
    // The game's D3D viewport is left unchanged.
    bool GetNgxPassthroughExpandedViewport(D3DVIEWPORT9* viewport) const;

    // Rewrites a four-vertex user-pointer quad laid out as float4 position, then float2 UV.
    // The DOF gather keeps its filter-buffer quad and reads the full upscaled scene.
    // The separable blur stays on the downsampled region.
    // A blend quad on the scene rect grows to the bound target, and its filter UVs stay put.
    // A composite at the centred view rect on the backbuffer grows to the whole backbuffer.
    void RetargetNgxUpscaledPostQuad(void* vertices, uint32_t vertexCount, uint32_t stride);
    // True when the bound declaration starts with a float4 position followed by a float2 UV.
    // UE3's FFilterVertex and its FBatchedElements resolve vertex both match.
    bool ngxBoundDeclIsFilterVertex() const;

    // Called at the end of the draw that deferred injection.
    // A queued draw is followed by DLSS on the image it wrote.
    // A dropped draw cancels the deferral.
    // A later draw then cannot inject against a stale target.
    void FinishDeferredNgxPrePostInjection(bool drawWasIssued);

    /**
      * \brief: NGX passthrough mode: returns the texture LOD bias to fold into the current
      * draw's sampler keys, or 0 when out of scope. DLSS Super Resolution renders the scene
      * at a reduced resolution, so scene material sampling must bias mip selection by
      * log2(render / display) to match the upscaled output's texel density (the standard
      * DLSS integration bias). Scene-color draws during the pre-injection phase only; UI
      * and post passes sample unbiased. Called from D3D9DeviceEx::CreateSamplerKey.
      */
    float GetNgxPassthroughSamplerLodBias() const;

    /**
      * \brief: NGX passthrough mode: called from D3D9DeviceEx::Clear before the clear executes.
      * A mid-scene depth clear on the scene depth buffer (UE3 clears depth ahead of its
      * foreground DPG) destroys the world depth needed for motion vector generation, so the
      * depth buffer is snapshotted just before the first such clear each frame.
      */
    void NotifyClear(DWORD clearFlags);

    /**
      * \brief: NGX passthrough mode: called from D3D9DeviceEx::StretchRect. UE3's D3D9 RHI
      * gives the scene color a dedicated render surface and resolves it into a separate
      * texture that the post-process chain samples; tracking these copies lets the
      * pre-post-process injection recognize the first post pass (it samples a resolve
      * destination, never the render surface itself).
      */
    void NotifyStretchRect(const Rc<DxvkImage>& sourceImage, const Rc<DxvkImage>& destImage);

    /**
      * \brief: NGX passthrough mode: one-time XeSS input-resolution sync once the display
      * size is known (preset sync runs earlier in RtxInitializer).
      */
    void bootstrapNgxPassthroughUpscaler(uint32_t displayWidth, uint32_t displayHeight);

  private:
    D3D9DeviceEx* m_parent;

    std::optional<D3DPRESENT_PARAMETERS> m_activePresentParams;

    uint32_t m_drawCallID = 0;
    // Note: A frame identifier the the main thread holds on to passed down into thread invocations such that
    // Reflex markers have a consistent ID despite executing in parallel (as typical methods of getting a frame ID
    // in DXVK depend on say when the submit thread's present happens which is unpredictable).
    uint64_t m_reflexFrameId = 0;

    bool m_passthroughSuspended = false;
    bool m_rtxInjectTriggered = false;

    fast_unordered_cache<Rc<DxvkSampler>> m_samplerCache;

    enum class Ue3VertexFactoryType : uint8_t {
      Unknown = 0,
      Local,
      GPUSkin,
      GPUSkinMorph,
      Terrain,
      TerrainMorph,
      Particle,
      ParticleBeamTrail,
      SpeedTree,
      Foliage,
      // FParticleInstancedMeshVertexFactory: hardware-instanced static meshes used by mesh
      // particle emitters, including the PhysX/NxFluid debris ones.
      ParticleInstancedMesh,
      LocalDecal,
      LensFlare,
      PositionOnly,
    };

    Ue3VertexFactoryType m_currentUe3VertexFactory = Ue3VertexFactoryType::Unknown;
    enum class Ue3PassType : uint8_t {
      Unknown = 0,
      Material,
      DepthPrepass,
      ShadowDepth,
      Velocity,
      Lighting,
      ModulatedShadowProjection,
      FullscreenPostProcess,
      UiComposite,
      FogOrDistortion,
      VideoCinematic,
      VideoSurface,
      SceneCapture,
    };

    Ue3PassType m_currentUe3PassType = Ue3PassType::Unknown;
    fast_unordered_cache<Ue3VertexFactoryType> m_ue3VertexFactoryCache;


    static Ue3VertexFactoryType classifyUe3VertexFactory(const D3D9VertexElements& elements);
    static bool isUe3WorldGeometryVertexFactory(Ue3VertexFactoryType type);

    // Aspect-matched to the backbuffer so square SceneCapture RTs cannot pass as main-view-sized on widescreen.
    static bool ue3ViewportAspectMatchesBackbuffer(uint32_t vpW, uint32_t vpH, uint32_t bbW, uint32_t bbH);

    struct Ue3ShaderFeatureInfo {
      bool initialized = false;
      bool hasMaterialSampler = false;
      bool hasEngineAuxSampler = false;
      bool hasSceneColorSampler = false;
      bool hasSceneDepthSampler = false;
      bool hasLightAttenuationSampler = false;
      bool hasShadowSampler = false;
      bool hasVelocitySampler = false;
      bool hasExposureOrToneSampler = false;
      bool hasBlurredImageSampler = false;
      bool hasFilterTextureSampler = false;
      bool hasUiSampler = false;
      bool hasDistortionSampler = false;
      bool hasBinkConstants = false;
      bool hasPrevViewProjection = false;
      bool hasVelocityConstants = false;
      bool hasMotionBlurConstants = false;
      bool hasDynamicLightingConstants = false;
      bool hasLightFunctionConstants = false;
      bool hasSphericalHarmonicLightingConstants = false;
      bool hasScreenToShadowMatrix = false;
      bool hasShadowModulateConstants = false;
      bool hasToneMapConstants = false;
      bool hasGammaConstants = false;
      bool hasFogConstants = false;
      bool hasHazeConstants = false;
      bool hasUiCompositeConstants = false;
      // Constant registers of UE3's GammaCorrectionPixelShader (kNgxNoRegister when absent).
      // That shader is the engine's final composite, which the Super Resolution upscale
      // replaces - so its transform has to be carried over rather than dropped.
      static constexpr uint32_t kNgxNoRegister = UINT32_MAX;
      uint32_t gammaInverseReg = kNgxNoRegister;
      uint32_t gammaColorScaleReg = kNgxNoRegister;
      uint32_t gammaOverlayColorReg = kNgxNoRegister;
      // DOFAndBloom / UberPostProcess CTAB tokens (DepthOfFieldCommon / FilterPixelShader).
      bool hasDofPackedParameters = false;
      bool hasDofMinMaxBlurClamp = false;
      bool hasFilterSampleWeights = false;
      // Only the DOF/bloom gather declares BloomScale.
      bool hasBloomScale = false;
      // The sampler register of the motion-blur velocity buffer.
      // It stays 0xFF when the shader has none.
      uint8_t velocitySamplerIndex = 0xFF;

      bool looksLikeDofAndBloomPostProcess() const {
        return hasBlurredImageSampler ||
               hasBloomScale ||
               (hasDofPackedParameters && hasDofMinMaxBlurClamp) ||
               (hasFilterTextureSampler && hasFilterSampleWeights);
      }

      // GammaColorScaleAndInverse float register (-1 when not present in the shader's CTAB).
      int16_t toneMapGammaColorScaleReg = -1;
    };

    fast_unordered_cache<Ue3ShaderFeatureInfo> m_ue3ShaderFeatureCache;
    Ue3ShaderFeatureInfo getUe3ShaderFeatureInfo(const D3D9CommonShader* shader);
    Ue3PassType classifyUe3Pass(const DrawContext& drawContext);

    // Minimal CTAB scan for the camera symbols plus what the object velocity capture needs.
    struct NgxCameraCtabRegs {
      bool ctabVerified = false;   // shader declares both ViewProjectionMatrix and CameraPosition
      uint32_t viewProjRegister = 0;
      uint32_t viewOriginRegister = 0;

      // ViewProjection presence independent of the full verification (ctabVerified
      // additionally requires the camera position symbol; the velocity capture accepts
      // CPU-modified-mesh draws on ViewProjection + LocalToWorld alone)
      bool hasViewProjection = false;
      bool hasLocalToWorld = false;
      uint32_t localToWorldRegister = 0;
      bool hasWorldToLocal = false;
      uint32_t worldToLocalRegister = 0;
      bool hasBoneMatrices = false;
      uint32_t boneMatricesRegister = 0;
      uint32_t boneMatricesRegisterCount = 0;
      // UE3 GPU skin index addressing: false = the shader multiplies BLENDINDICES by 3
      // before the address register (raw bone indices in the vertex data), true = the
      // indices are pre-scaled at mesh build and feed the address register directly.
      // Detected from the bytecode (a mul/mad reading the BLENDINDICES input).
      bool boneIndicesPreScaled = false;
      // Bone influences the shader actually consumes (UE3 builds 1/2/4-influence
      // variants): the count of distinct BLENDWEIGHT components read by arithmetic
      // instructions; 1 when only indices are read (rigid-skin, implicit weight 1)
      uint32_t skinInfluenceCount = 0;
    };
    fast_unordered_cache<NgxCameraCtabRegs> m_ngxCameraCtabCache;
    NgxCameraCtabRegs scanNgxCameraCtabRegs(const std::vector<uint8_t>& bytecode) const;

    // Object velocity capture (see RtxNgxPassthrough::objectVelocities): per draw-identity
    // transform state from previous sightings. The identity hash covers geometry and draw
    // parameters only, so every placement of the same mesh asset in the level aliases onto
    // one identity - each placement is tracked as an instance and draws are matched against
    // the previous frame's instances (same placement = static, near = the same object moved,
    // far = a different placement). The disambiguated objectToWorld feeds the previous-frame
    // clip transform, the same-placement test and the distance metric for the near-match.
    struct NgxVelocityObjectInstance {
      Matrix4 objectToWorld;
      // The packed ViewProjection at the sighting's draw (oriented): scene phases render
      // with their own projections (UE3's foreground DPG uses a first-person FOV), so
      // both sides of a velocity delta must compose with the projection their draw
      // actually used
      Matrix4 worldToProjection;
      uint32_t lastSeenFrame = 0;
      // Movement history gating velocity emission: wasMoving latches once the instance
      // is a confirmed mover (decaying after a period at rest - see the exact-match
      // path), the last per-frame deltas feed the two-frame consistency confirmation
      // for objects first sighted with non-gentle motion, lastEmitFrame drives the
      // decay. Together they ensure static placements swapping visibility (culling
      // pop-in/out) cannot fabricate velocity.
      bool wasMoving = false;
      Vector3 lastMoveDelta = Vector3(0.0f, 0.0f, 0.0f);
      float lastRotScaleDelta = 0.0f;
      uint32_t lastEmitFrame = 0;
      // This frame's emitted draw (index into m_ngxVelocityDraws) and the phase
      // duplication guard: a same-frame repeat draw in the other scene phase (UE3
      // renders the first person meshes into both the intermediate and foreground DPGs
      // in some movestates) re-emits the recorded draw for its own phase, once. The
      // previous-side state is preserved at emit time so the duplicate can recompose
      // with its own phase's projection.
      uint32_t lastEmitDrawIndex = 0;
      uint32_t lastPhaseDuplicateFrame = 0;
      Matrix4 lastEmitPrevObjectToWorld;
      Matrix4 lastEmitPrevWorldToProjection;
      // Skinned instances: the bone palette from the last sighting (raw registers,
      // shader-declared count); empty for rigid draws. Animation with a static
      // LocalToWorld is detected and emitted through palette deltas.
      std::vector<Vector4> bones;
      // CPU-modified mesh instances: the vertex positions from the last sighting
      // (snapshotted from the dynamic buffer's CPU mapping); empty otherwise
      std::vector<Vector3> dynamicPositions;
    };
    struct NgxVelocityObjectState {
      std::vector<NgxVelocityObjectInstance> instances;
      // Frame of the last brand-new registration (a sighting pairing with nothing):
      // recent pop-ins mark the identity as churning, distrusting motion-consistency
      // confirmation (grids of instanced meshes produce repeating pop-in deltas under
      // steady camera movement, indistinguishable from consistent object motion)
      uint32_t lastNewRegistrationFrame = 0;
      // Frame this identity last contributed to the global transform offset measurement: it
      // gets one say per frame however many copies of it are on screen
      uint32_t lastOffsetVoteFrame = UINT32_MAX;
    };
    fast_unordered_cache<NgxVelocityObjectState> m_ngxVelocityObjectCache;
    std::vector<NgxVelocityDraw> m_ngxVelocityDraws;

    // The frame's accepted main camera validity (plus the previous frame's), gating the
    // velocity capture. The clip transforms themselves compose with the packed
    // ViewProjection captured per draw (see NgxVelocityObjectInstance::
    // worldToProjection) - the game's exact vertex transform, which the velocity
    // raster's depth compare depends on. The reconstruction's transpose flag
    // disambiguates LocalToWorld packing.
    bool m_ngxFrameCameraUsedTranspose = false;
    bool m_ngxFrameCameraValid = false;
    bool m_ngxPrevCameraValid = false;
    bool m_ngxPrevCameraUsedTranspose = false;

    // Capture diagnostics for the developer menu (reset per frame; the transpose flip
    // count is cumulative and detects an unstable packing-convention tiebreaker)
    NgxVelocityCaptureStats m_ngxVelocityStats;
    uint32_t m_ngxCameraTransposeFlips = 0;

    // Some UE3 titles upload LocalToWorld in a space carrying a per-frame global translation
    // rather than true world space, recognizable in that every placement in a frame shares one
    // fractional offset and it moves frame to frame with the camera. A static placement's
    // matrix is then not identical across frames the way it is in Mirror's Edge. This is the
    // offset the scene as a whole moved by, which a static placement's
    // translation delta reproduces exactly. Zero for titles uploading true world space, which
    // reduces the static test back to plain equality.
    Vector3 m_ngxGlobalTransformOffset = Vector3(0.0f, 0.0f, 0.0f);
    // Translation deltas of this frame's basis-preserving pairings, resolved to the offset
    // above at frame end by picking the value the most sightings agree on
    struct NgxTranslationDeltaVote {
      Vector3 delta = Vector3(0.0f, 0.0f, 0.0f);
      uint32_t votes = 0;
    };
    static constexpr uint32_t kNgxTranslationDeltaVoteSlots = 8;
    std::array<NgxTranslationDeltaVote, kNgxTranslationDeltaVoteSlots> m_ngxTranslationDeltaVotes;
    // Votes behind the offset currently in force, so a better supported candidate can take
    // over mid-frame while a single dissenting sighting cannot
    uint32_t m_ngxAdoptedOffsetVotes = 0;

    // On-demand velocity capture dump (rtx.ngxPassthrough.dumpVelocityCaptureFrames), armed while
    // the object under investigation is on screen, since a log that fires by itself samples startup
    // instead. Budgeted per draw kind rather than per frame: world geometry is hundreds of draws
    // and comes first, so a single budget is spent before a character is ever reached.
    static constexpr uint32_t kNgxVelocityDumpMaxSkinnedPerFrame = 40;
    static constexpr uint32_t kNgxVelocityDumpMaxDynamicPerFrame = 20;
    static constexpr uint32_t kNgxVelocityDumpMaxRigidPerFrame = 12;
    uint32_t m_ngxVelocityDumpFramesLeft = 0;
    uint32_t m_ngxVelocityDumpSkinnedThisFrame = 0;
    uint32_t m_ngxVelocityDumpDynamicThisFrame = 0;
    uint32_t m_ngxVelocityDumpRigidThisFrame = 0;

    // Counters accumulated over a window and logged, because the health of the capture is a
    // ratio - sightings that paired against sightings that had to register anew - and the
    // per-sighting miss lines are too heavily rate limited to show one. Only the per-frame
    // counters that mean something summed live here; the rest of NgxVelocityCaptureStats is
    // frame state (the frame's camera validity) or already cumulative (the transpose flips).
    struct NgxVelocityWindowTotals {
      uint32_t frames = 0;
      uint32_t captured = 0;
      uint32_t capturedSkinned = 0;
      uint32_t capturedDynamic = 0;
      uint32_t capturedForeground = 0;
      uint32_t exactMatches = 0;
      uint32_t newRegistrations = 0;
      uint32_t newRegistrationsSkinned = 0;
      uint32_t missNoLastFrameSighting = 0;
      uint32_t missBeyondTranslation = 0;
      uint32_t missBeyondRotation = 0;
      uint32_t claimedWithoutVelocity = 0;
      uint32_t pairedBeyondBounds = 0;
      uint32_t skippedNoCamera = 0;
      uint32_t skippedBudget = 0;
      uint32_t skippedZDisabled = 0;
      uint32_t skippedInstanceCap = 0;
      uint32_t skippedDynamicBuffer = 0;
      uint32_t skippedBonePalette = 0;
      uint32_t depthClears = 0;
      // Frames whose depth was cleared more than once mid-scene. Only two depth sources exist -
      // the snapshot taken at the first clear and the live buffer after the last - so a draw
      // between two clears is in neither, and the raster discards its velocity entirely.
      uint32_t framesWithOrphanedDepthPhase = 0;
    };
    static constexpr uint32_t kNgxVelocityWindowFrames = 600;
    NgxVelocityWindowTotals m_ngxVelocityWindow;

    // Skinned / CPU-modified draws captured this frame (bound their upload buffers;
    // reset per frame)
    uint32_t m_ngxVelocitySkinnedDraws = 0;
    uint32_t m_ngxVelocityDynamicDraws = 0;

    // Self-triggering pairing-miss dump: when sightings fail to pair with their history,
    // a few lines with the exact comparison values go to the log (rate limited; the burst
    // frame latch starts out of range of any real frame counter value)
    uint32_t m_ngxVelocityPairingLogFrame = UINT32_MAX;
    uint32_t m_ngxVelocityPairingLogLines = 0;
    uint32_t m_ngxVelocityPairingLogNextAllowedFrame = 0;

    void tryCaptureNgxVelocityDraw(const DrawContext& drawContext);

    // UE3 LocalToWorld extraction (transpose/packing disambiguation + memo cache), used by velocity capture.
    Matrix4 extractUe3ObjectToWorld(uint32_t reg, bool hasWorldToLocal, uint32_t w2lReg, bool cameraUsedTranspose);

    // NGX passthrough mode: ScreenPositionScaleBias register per shader (UE3's shared
    // clip-space-to-scene-texture-UV constant, used by every screen space lookup: shadow
    // projections reading the scene depth from the scene color alpha, translucency reading
    // the resolved scene, distortion, fog). The viewport jitter shifts rendered content
    // physically but clip-space-derived UVs do not follow it, so the bias components (w for
    // U, z for V) are shifted by the frame jitter at constant upload time; consumers then
    // sample the jittered content aligned. Cached per shader bytecode hash (VS and PS).
    struct NgxSpsbCtabReg {
      bool present = false;
      uint32_t reg = 0;
      // UE3's motion blur keeps its taps inside the view rect with RenderTargetClampParameter.
      bool clampPresent = false;
      uint32_t clampReg = 0;
    };
    // Returns a key for the registers a patch occupies, so a change between shaders forces a fresh upload.
    // Returns -1 when the shader has nothing to patch.
    static int32_t ngxPatchRegisterKey(const NgxSpsbCtabReg& patch, bool replaceFull);
    fast_unordered_cache<NgxSpsbCtabReg> m_ngxSpsbCtabCache;
    NgxSpsbCtabReg scanNgxSpsbCtabReg(const std::vector<uint8_t>& bytecode) const;
    NgxSpsbCtabReg lookupNgxSpsbCtabReg(const D3D9CommonShader* shader);

    // Returns the scale-bias UE3 would upload for a view covering the whole scene-colour buffer.
    // The layout is (scale U, scale V, offset V, offset U),
    // as in ScreenPositionScaleBias and SceneCoordinateScaleBias.
    // The offsets include UE3's D3D9 half-pixel centre.
    Vector4 ngxFullBufferScaleBias() const;

    // Per-draw ScreenPositionScaleBias patch, decided in prepareDrawForNgxPassthrough
    // and consumed when the device uploads constants.
    // Cleared when injection commits.
    // Later draws sample the unjittered output.
    // m_ngxSpsbReplaceFull installs the full-buffer mapping and suppresses the jitter add.
    bool m_ngxSpsbPatchActive = false;
    bool m_ngxSpsbReplaceFull = false;
    NgxSpsbCtabReg m_ngxSpsbPatchVs;
    NgxSpsbCtabReg m_ngxSpsbPatchPs;
    float m_ngxSpsbPatchAdd[2] = { 0.0f, 0.0f };

    // This frame's pre-post evaluate upscaled the scene subrect onto the whole scene-colour target.
    // Post draws stay retargeted until the first UI draw.
    bool m_ngxPrePostUpscaled = false;
    bool m_ngxLoggedPrePostUpscale = false;

    // The bound viewport was expanded to the render target of that draw.
    // The extent is remembered.
    // A later expandable target of a different size rebinds.
    bool m_ngxViewportExpandApplied = false;
    uint32_t m_ngxExpandedViewportWidth = 0;
    uint32_t m_ngxExpandedViewportHeight = 0;

    // TdSunHaze matched the pre-post read.
    // The draw is issued, then DLSS runs on the image it wrote.
    bool m_ngxDeferPrePostInjection = false;
    Rc<DxvkImage> m_ngxDeferredPrePostTarget;
    D3DPRIMITIVETYPE m_ngxDeferredPrimitiveType = D3DPT_TRIANGLELIST;

    // Where this frame's injection was taken.
    enum class NgxInjectionSite : uint8_t {
      None,
      Haze,
      Gather,
      AimedRead,
      FinalComposite,
      Stretch,
      Late,
    };
    NgxInjectionSite m_ngxFrameInjectionSite = NgxInjectionSite::None;
    // True once this frame's pre-post injection upscaled the scene subrect in place.
    bool m_ngxFrameUpscaledInPlace = false;

    // Scene geometry drawn after an offscreen injection means the injection landed inside the scene.
    // Consecutive frames like that suspend the haze, gather and aimed-read points.
    bool m_ngxSceneDrawAfterInjection = false;
    uint32_t m_ngxSceneAfterInjectionStrikes = 0;
    uint32_t m_ngxEarlyInjectionSuspensions = 0;
    uint32_t m_ngxEarlyInjectionSuspendedUntil = 0;
    bool ngxEarlyInjectionSuspended() const;

    // Set when a post pass stays on the reduced view rect after an in-place upscale.
    // Consecutive frames like that suspend sub-full pre-post until ScreenPercentage or the scene extent changes.
    bool m_ngxReducedRectDrawThisFrame = false;
    uint32_t m_ngxReducedRectStrikes = 0;
    bool m_ngxSubrectPrePostSuspended = false;
    float m_ngxSuspendedScreenPercentage = 0.0f;
    VkExtent3D m_ngxSuspendedSceneExtent = { 0, 0, 0 };

    // A profile is logged once it holds for two frames, so frames that alternate do not flood the log.
    std::string m_ngxPendingProfile;
    std::string m_ngxLoggedProfile;
    uint32_t m_ngxPendingProfileFrames = 0;
    uint32_t m_ngxProfileLogsRemaining = 100;
    void logNgxUe3Profile();

    // These hold the first draw that tripped each fallback this frame, which its warning quotes.
    std::string m_ngxSceneDrawEvidence;
    std::string m_ngxReducedRectEvidence;

    std::string describeNgxDraw(const DrawContext& drawContext) const;
    void checkNgxPostInjectionDraw(const DrawContext& drawContext);
    void noteNgxReducedRectDraw(const char* reason);
    void updateNgxInjectionFallbacks();

    // Constant uploads only happen when the game dirties them; when the bound shader's
    // patch register differs from what the last upload patched (shader switch without a
    // constant change - the bridge's redundant-setter filtering makes this reachable), a
    // fresh upload is forced so the patch lands at the right register (-1 = no patch)
    int32_t m_ngxSpsbLastVsReg = -1;
    int32_t m_ngxSpsbLastPsReg = -1;

    // The Super Resolution sampler LOD bias folded into the last draw's sampler keys
    // (see GetNgxPassthroughSamplerLodBias); samplers are only re-created when dirtied,
    // so every change forces a full sampler re-bind
    float m_ngxAppliedSamplerLodBias = 0.0f;

    // The jitter state the currently bound Vulkan viewport carries. The per-draw jitter decision
    // follows the bound targets, which can change without D3D9 dirtying the viewport, so the
    // rebind has to be forced whenever this stops matching the current draw.
    bool m_ngxViewportJitterApplied = false;
    float m_ngxAppliedViewportJitter[2] = { 0.0f, 0.0f };

    // Last logged jitter loop length, so the line is emitted only when it changes
    uint32_t m_ngxLoggedJitterSequenceLength = 0;

    // Which injection requirements the current frame met, for attributing a relocation to the
    // late point, plus the running relocation count and its last reported value
    bool m_ngxDiagSawPostQuad = false;
    bool m_ngxDiagSceneColorReady = false;
    uint32_t m_ngxPrePostMissCount = 0;
    uint32_t m_ngxPrePostMissLastReportedCount = 0;

    // Last frame whose scene draws donated the view rect. The per-frame validity flag needs a
    // camera-reconstructed scene draw ahead of the post chain in the same frame, which a frame
    // can miss for reasons unrelated to the view rect; the rect itself only changes when
    // ScreenPercentage or the resolution does, so a recent one stays usable.
    uint32_t m_ngxSceneViewportLastValidFrame = 0;

    // Where this frame's scene depth clears, scene color reads, last depth-writing draw, injection
    // and first backbuffer draw fell. The injection has to go at the first read past the end of the
    // scene, which is only knowable once the frame is over, so the shape is recorded and the next
    // frame's position derived from it. Also logged periodically, which is how a placement is
    // verified in a title whose frame shape has not been seen before.
    static constexpr uint32_t kNgxFrameShapeSlots = 12;
    uint32_t m_ngxFrameClearDraws[kNgxFrameShapeSlots] = {};
    uint32_t m_ngxFrameReadDraws[kNgxFrameShapeSlots] = {};
    uint32_t m_ngxFrameClearCount = 0;
    uint32_t m_ngxFrameReadCount = 0;
    uint32_t m_ngxFrameLastGeometryDraw = 0;
    uint32_t m_ngxFrameInjectionDraw = 0;
    uint32_t m_ngxFrameFirstBackbufferDraw = 0;
    uint32_t m_ngxFrameShapeLogsRemaining = 10;

    // Reads seen so far this frame, and how many of them fell inside the scene last time. Counted
    // from the frame's most recent scene depth clear where there is one, because that is the form
    // that holds still: frames differ in how many reads land before the clear, so a whole-frame
    // count inherits that variation and alternates, while the stretch from the clear to the end of
    // the scene does not. The whole-frame pair covers frames with no detected clear.
    uint32_t m_ngxPrePostCandidatesThisFrame = 0;
    uint32_t m_ngxPrePostReadsInsideScene = 0;
    uint32_t m_ngxPrevReadsInsideScene = UINT32_MAX;
    uint32_t m_ngxReadsSinceLastClear = 0;
    uint32_t m_ngxReadsAfterClearInsideScene = 1;
    uint32_t m_ngxPrevReadsAfterClearInsideScene = UINT32_MAX;

    // Set once the game has bound the backbuffer this frame, which marks the end of the scene and
    // its post chain. Scene color reads after that belong to UI compositing and must not count
    // towards the injection ordinal, or the upscaler ends up resolving the UI.
    bool m_ngxBackbufferDrawSeenThisFrame = false;

    void countNgxPostInjectionSceneColorConsumer(const DrawContext& drawContext);
    void updateNgxPrePostInjectionAim();

    // Isolated renderer shadow (GSystemSettings untouched). Bridge path owns the parent handle.
    bool m_ngxScreenPercentageScanDone = false;
    bool m_ngxScreenPercentageDriven = false;
    bool m_ngxGameProcessOwned = false;
    HANDLE m_ngxGameProcess = nullptr;
    DWORD m_ngxGameProcessId = 0;
    uintptr_t m_ngxScreenPercentageRemoteAddr = 0;
    uintptr_t m_ngxGameSettingsShadowRemoteAddr = 0;
    bool m_ngxGameSettingsRedirectsValid = false;
    float m_ngxScreenPercentageLastLogged = 0.0f;
    uint32_t m_ngxScreenPercentageScanAttempts = 0;
    uint64_t m_ngxScreenPercentageNextScanMs = 0;
    struct NgxGameSettingsCodePatch {
      uintptr_t operandAddress = 0;
      uint32_t originalOperand = 0;
      uint32_t redirectedOperand = 0;
      DWORD originalProtection = 0;
      bool originalProtectionKnown = false;
      bool forceRestore = false;
    };
    std::vector<NgxGameSettingsCodePatch> m_ngxGameSettingsCodePatches;

    // Shape alone never commits a candidate: the installed redirects are proven by driving a
    // known percentage and watching the game's own scene viewport respond. A disproven
    // candidate is rolled back, remembered, and the scan runs again for the next best match.
    enum class NgxSettingsProbeState {
      Idle,
      Pending,
      Confirmed,
      Failed,
    };
    NgxSettingsProbeState m_ngxSettingsProbeState = NgxSettingsProbeState::Idle;
    uint32_t m_ngxSettingsProbeStartFrame = 0;
    float m_ngxSettingsProbeValue = 0.0f;
    std::vector<uintptr_t> m_ngxRejectedSettingsCandidates;

    // FSystemSettings::NeedsUpscale() reads both fields through `this`, so the engine can only
    // keep performing its own upscale when those relative readers were redirected too;
    // otherwise the runtime upscales the reduced subrect itself.
    bool m_ngxEngineUpscaleAvailable = false;
    bool m_ngxRuntimeOwnedUpscale = false;
    uint32_t m_ngxMissingEngineUpscaleFrames = 0;

    // Scene-camera viewport gate. 0 means unknown.
    float m_ngxGameScreenPercentage = 0.0f;

    // True after this frame records a viewport that matches the driven ScreenPercentage.
    // A full-surface replay on that scene-colour image matches primitive count and vertex shader.
    // The replay leaves the recorded rect in place.
    // Any other full-surface view replaces the rect.
    bool m_ngxReducedSceneViewportThisFrame = false;
    struct NgxReducedSceneDraw {
      uint32_t primitiveCount = 0;
      XXH64_hash_t vertexShaderHash = 0;
    };
    std::vector<NgxReducedSceneDraw> m_ngxReducedSceneDraws;

    bool m_ngxPassthroughBootstrapped = false;

    // Latest UE3 camera matrices accepted this frame; replayed into the CS camera at injection
    // time so dispatch does not depend on earlier async processExternalCamera ordering.
    Matrix4 m_ngxFrameWorldToView;
    Matrix4 m_ngxFrameViewToProjection;
    bool m_ngxFrameCameraMatricesValid = false;

    // Cache of the current vertex-shader camera constants so repeated draws skip re-extraction.
    bool tryGetUe3CameraFromConstantsCached(uint32_t viewProjReg,
                                            uint32_t viewOriginReg,
                                            Matrix4& outWorldToView,
                                            Matrix4& outViewToProjection,
                                            bool& outUsedTranspose,
                                            float& outReconstructionError,
                                            XXH64_hash_t* outConstantsHash = nullptr);

    struct Ue3CameraConstantsCache {
      XXH64_hash_t hash = 0;
      bool valid = false;
      // Failed extractions are cached too: engine utility shaders whose declared camera
      // registers never decompose would otherwise re-run the full extraction every draw.
      // Extraction is deterministic on the register contents (the key), so a cached
      // failure can never mask a would-be success.
      bool extractionFailed = false;
      bool usedTranspose = false;
      Matrix4 worldToView;
      Matrix4 viewToProjection;
      float reconstructionError = 0.0f;
    };

    // Small N-way cache: the main view, capture probes and engine utility shaders carry
    // distinct camera constant blocks (including cached failures) that interleave within
    // a frame, so a single slot thrashes and re-runs the heavy matrix extraction (4x4
    // inverse, two projection decompositions) once per draw instead of once per unique
    // camera.
    static constexpr uint32_t kUe3CameraConstantsCacheSlots = 8;
    std::array<Ue3CameraConstantsCache, kUe3CameraConstantsCacheSlots> m_ue3CameraConstantsCache;
    uint32_t m_ue3CameraConstantsCacheNextSlot = 0;

    // ObjectToWorld extraction memo: the LocalToWorld (+ optional WorldToLocal) constant
    // block resolves to the same transpose/affinity/inverse disambiguation result whenever
    // the register contents repeat (static placements re-upload identical matrices every
    // frame). All inputs are part of the key, so entries can never go stale; the map is
    // cleared wholesale when it exceeds a size cap.
    static constexpr size_t kUe3ObjectToWorldCacheMaxEntries = 32768;
    fast_unordered_cache<Matrix4> m_ue3ObjectToWorldCache;

    struct IndexContext {
      VkIndexType indexType = VK_INDEX_TYPE_NONE_KHR;
      D3D9CommonBuffer* ibo = nullptr;
      DxvkBufferSliceHandle indexBuffer;
    };

    struct VertexContext {
      uint32_t stride = 0;
      uint32_t offset = 0;
      DxvkBufferSlice buffer;
      DxvkBufferSliceHandle mappedSlice;
      D3D9CommonBuffer* pVBO = nullptr;
      bool canUseBuffer;
    };



    // Frame counter used by NGX velocity pairing and the settings probe.
    uint32_t m_ue3FrameCounter = 0;

    const Direct3DState9& d3d9State() const;

    void triggerInjectRTX();

    bool checkBoundTextureCategory(const fast_unordered_set& textureCategory) const;

    // --- NGX passthrough mode (see RtxNgxPassthrough) ---
    // All draws rasterize as-is; the per-draw work reduces to UE3 camera extraction, scene
    // color/depth target identification (for viewport jitter and DLSS inputs) and the thin
    // RTX injection trigger at scene end.
    PrepareDrawFlags prepareDrawForNgxPassthrough(const DrawContext& drawContext);
    void tryNgxPassthroughCameraCapture(uint32_t primitiveCount);
    void rememberNgxReducedSceneDraw(uint32_t primitiveCount, XXH64_hash_t vertexShaderHash);
    bool ngxFullSurfaceDrawReplaysReducedScene(uint32_t primitiveCount, XXH64_hash_t vertexShaderHash) const;
    void emitNgxPassthroughFrameData();

    void applyNgxPassthroughScreenPercentage();
    void locateNgxPassthroughGameSettings();
    void restoreNgxGameSettingsRedirects();
    bool writeNgxScreenPercentageShadow(float screenPercentage);
    void mirrorNgxLiveScreenPercentage();
    void updateNgxSettingsProbe();

    // True while the runtime, rather than the engine, owns the final reduced-subrect upscale.
    bool ngxRuntimeOwnsUpscale() const;

    // Scene targets identified from CTAB-verified camera draws with depth writes. Kept across
    // frames (UE3 render targets are stable between resolution changes) so the jitter can
    // engage from the first draw of a frame; dropped when unseen for a while.
    Rc<DxvkImage> m_ngxSceneColorImage;
    Rc<DxvkImage> m_ngxSceneDepthImage;
    // Holds a buffer's screen-percentage subrect while it is scaled onto the whole image.
    Rc<DxvkImage> m_ngxSubrectStretchCopy;
    uint32_t m_ngxSceneTargetsLastSeenFrame = 0;

    // Per-frame sub-pixel jitter (pixels), decided on the app thread at the first draw of the
    // frame and forwarded to the CS side so DLSS/DLFG report exactly the rasterized value
    float m_ngxFrameJitter[2] = { 0.0f, 0.0f };
    bool m_ngxFrameJitterValid = false;
    bool m_ngxFrameDataEmitted = false;
    bool m_ngxDepthSnapshotTakenThisFrame = false;
    // Qualifying mid-scene depth clears seen this frame (first = foreground DPG boundary,
    // second = occlusion query pass wiping the foreground depth)
    uint32_t m_ngxDepthClearsThisFrame = 0;
    // The late upscale runs after the uber pass is queued, before the HUD.
    bool m_ngxDeferLateSubrectUpscale = false;

    // A title that clears depth mid-scene draws its foreground after that clear.
    // The frame of the latest such clear tells whether the title still clears there.
    bool m_ngxMidSceneClearSeen = false;
    uint32_t m_ngxLastMidSceneClearFrame = 0;
    // True once this frame is past its foreground depth clear.
    // True as well when the title has not cleared depth mid-scene recently.
    bool ngxPastForegroundClear() const;

    XXH64_hash_t m_ngxLastCameraConstantsHash = 0;

    // The swapchain backbuffer image for this frame (backbuffers rotate every present).
    // The scene-end trigger only accepts draws rendering to THIS image: description-based
    // primary checks also match offscreen buffers allocated at backbuffer size (e.g. ME's
    // TdUI compositing buffers), which would make the injection target the wrong image.
    Rc<DxvkImage> m_ngxFrameBackbufferImage;

    // ScreenPercentage upscaling (Super Resolution): the viewport of this frame's scene draws
    // (the subrect the game rendered into) and, once the engine's stretch onto the primary
    // target was detected and suppressed, its source texture plus the scene subrect
    D3DVIEWPORT9 m_ngxSceneViewport = {};
    bool m_ngxSceneViewportValid = false;
    Rc<DxvkImage> m_ngxUpscaleSourceImage;
    VkRect2D m_ngxSubrect = { { 0, 0 }, { 0, 0 } };
    // Where the reduced colour sits inside the upscale source. Equal to m_ngxSubrect's offset
    // for the engine-owned path (both live in scene color space), but the runtime-owned path
    // reads from the backbuffer, where the engine centred the view rect instead.
    VkOffset2D m_ngxColorSubrectOffset = { 0, 0 };

    // Runtime-owned upscale: with NeedsUpscale() reporting no upscale, the engine's post chain
    // composites its reduced view rect straight into the backbuffer and stops there.
    D3DVIEWPORT9 m_ngxRuntimeUpscaleRect = {};
    bool m_ngxRuntimeUpscaleRectValid = false;

    // Super Resolution replaces UE3's FinishRenderViewTarget composite, so whatever that draw
    // was doing to the colour has to be reproduced on the upscaled result. Read from the
    // suppressed draw's own constants, so a title where the composite is a plain copy (its
    // post chain already gamma corrected into LDR scene colour) yields the identity.
    NgxOutputTransform m_ngxOutputTransform;
    void captureNgxOutputTransform();

    // The uber pass encodes its output with GammaColorScaleAndInverse.
    // These track that output and plain copies of it, so a late upscale that reads one is linearized first.
    std::array<Rc<DxvkImage>, 4> m_ngxUberEncodedImages;
    uint32_t m_ngxUberEncodedImageCount = 0;
    Vector4 m_ngxUberGammaColorScaleAndInverse = Vector4(1.0f, 1.0f, 1.0f, 1.0f);
    bool ngxImageIsUberEncoded(const DxvkImage* image) const;
    void rememberNgxUberEncodedImage(const Rc<DxvkImage>& image);
    void noteNgxUberEncode(const DrawContext& drawContext);
    // Reads GammaColorScaleAndInverse from the bound pixel shader.
    // requireBlurredImage limits it to the uber blend, since a tonemap-only pass has no BlurredImage.
    bool readNgxBoundUberGamma(Vector4& gammaColorScaleAndInverse, bool requireBlurredImage);
    // Returns the transform that linearizes an image the uber pass encoded and encodes it again after DLSS.
    // Returns a disabled transform when the values do not describe a real encode.
    static NgxOutputTransform ngxLinearizingTransform(const Vector4& gammaColorScaleAndInverse);

    // Pre-post-process injection: when the scene end trigger was the first post-process pass
    // sampling the scene color, DLSS reads and writes the image that pass consumes - the
    // scene color render surface or one of its resolve destinations (m_ngxSubrect carries
    // the scene viewport rect) - and the game's post chain picks up the result. The mirror
    // is the scene color surface when the target is a resolve destination, so later
    // re-resolves of the surface propagate the anti-aliased content too.
    Rc<DxvkImage> m_ngxColorTargetImage;
    Rc<DxvkImage> m_ngxColorMirrorImage;

    // These are this frame's same-size copies of the scene colour surface.
    // The post chain samples them in place of the surface.
    // Mirror's Edge copies with StretchRect.
    // Stock UE3 D3D9 copies with an opaque two-triangle draw.
    std::array<Rc<DxvkImage>, 8> m_ngxSceneColorResolves;
    uint32_t m_ngxSceneColorResolveCount = 0;
    bool m_ngxResolvesFromStretchRect = false;
    bool m_ngxResolvesFromDraws = false;
    void rememberNgxSceneColorResolve(const Rc<DxvkImage>& image);
    void learnNgxDrawResolve(const DrawContext& drawContext);

    // Where this frame's HUD-less copy was taken.
    enum class NgxHudlessCaptureSite {
      AfterSceneComposite,
      LateInjection,
      FrameEnd,
    };

    // True after this frame's HUD-less copy has been recorded.
    bool m_ngxHudlessCapturedThisFrame = false;
    // True after this frame's pre-post DLSS-NR has been recorded.
    bool m_ngxNeuralRenderingAppliedThisFrame = false;
    // True after the scene blit has targeted the backbuffer.
    // A copy before that blit reads the previous frame.
    bool m_ngxHudlessSceneCompositeSeen = false;

    bool m_ngxHudlessCaptureLogValid = false;
    NgxHudlessCaptureSite m_ngxHudlessLoggedSite = NgxHudlessCaptureSite::AfterSceneComposite;
    uint32_t m_ngxHudlessLastLogFrame = 0;

    // Acts on the first backbuffer draw after the scene blit, while the backbuffer holds the finished pre-UI image.
    // It records a pre-post frame's DLSS-NR, then the HUD-less copy.
    void maybeFinishNgxScene(const DrawContext& drawContext);
    bool ngxDrawIsBackbufferSceneComposite(const DrawContext& drawContext) const;
    // True for a backbuffer draw after this frame's scene composite that is not itself composite-shaped.
    // Nothing from there on needs retargeting.
    bool ngxDrawFollowsSceneComposite(const DrawContext& drawContext) const;
    void noteNgxHudlessCapture(NgxHudlessCaptureSite site);

    // Diagnostic dump of the post-chain flow (draws sampling render targets, StretchRect
    // copies, render state): armed for the first frames after scene target identification
    // and on demand (rtx.ngxPassthrough.dumpPostChainFrames); validates the pre-post
    // injection against the game's real compositing
    uint32_t m_ngxPostChainDumpFramesLeft = 0;

    // Session budget for the automatic dump. The on-demand option disarms itself after one use;
    // the automatic one is driven by scene target changes, which are not guaranteed to be rare.
    uint32_t m_ngxAutoDumpArmsRemaining = 4;
    uint32_t m_ngxPostChainDumpLinesThisFrame = 0;

    void dumpNgxPostChainDraw(const DrawContext& drawContext);
    const char* classifyNgxImageForDump(const DxvkImage* image) const;
    // Names the pass the bound pixel shader belongs to for the dump.
    // The names are haze, gather, blur, uber, DOF blend, tonemap, gamma, motion blur, resolve and other.
    const char* classifyNgxPassForDump(uint32_t rtSamplerMask, const DxvkImage* renderTargetImage);
    void reportNgxPrePostMiss();

    bool ngxPrePostInjectionAllowed() const;
    Rc<DxvkImage> matchNgxSceneColorSample(const DxvkImage* sampledImage) const;
    void recordNgxPrePostSceneColorRead();
    void engageNgxPrePostInjection(const Rc<DxvkImage>& matchedTarget);
    Rc<DxvkImage> findNgxPrePostSceneColorFromSamplers(bool applyOrdinalGate, bool recordRead);
    // Returns the image the bound pixel shader reads as scene colour.
    // A known scene-colour image wins.
    // Otherwise returns the largest sampled render target that covers the scene viewport, float formats first.
    // Returns null when nothing qualifies.
    Rc<DxvkImage> findNgxSceneColorSource() const;
    static bool ngxSceneViewportIsSubrect(const D3DVIEWPORT9& sceneViewport,
                                          uint32_t backBufferWidth, uint32_t backBufferHeight);
    // True when the viewport is the rect produced by the driven ScreenPercentage.
    bool ngxViewportMatchesDrivenScreenPercentage(const D3DVIEWPORT9& viewport) const;
    static bool ngxShaderIsFinishRenderViewTargetGamma(const Ue3ShaderFeatureInfo& psInfo);
    static bool ngxShaderIsFinalBackbufferComposite(const Ue3ShaderFeatureInfo& psInfo);
    // True when the shader declares any name of a scene, post-process or UI pass.
    // A plain copy declares none.
    static bool ngxShaderHasPostProcessNames(const Ue3ShaderFeatureInfo& psInfo);
    bool ngxDrawIsUi(const DrawContext& drawContext);
    bool ngxOffscreenPrePostShape(const DrawContext& drawContext);
    // True for UE3's filter color target.
    // Its size is the scene buffer divided by FilterDownsampleFactor (2, 4 or 8), plus a two-pixel border.
    bool ngxImageIsDofFilterBuffer(const DxvkImage* image) const;
    bool ngxBoundImageIsDofFilterBuffer() const;
    bool tryGetBoundPixelFeatures(Ue3ShaderFeatureInfo& info);
    // True for the DOF/bloom gather.
    // It samples scene colour and declares BloomScale or the DepthOfFieldCommon parameters.
    // Blur weights, BlurredImage and the uber grade belong to the passes after it.
    bool ngxBoundPixelIsDofGather();
    bool ngxBoundPixelIsDofBlend();
    bool ngxDepthTestDisabled() const;
    // True for haze, depth of field, bloom, tonemap, and gamma.
    // Fog and other scene-colour quads come earlier.
    // Upscaling on those quads leaves the distortion stencil in the screen-percentage rect.
    bool ngxBoundPixelIsUpscaleBoundary();
    // Scales the screen-percentage subrect of a screen-space buffer onto the whole image.
    // Passes after an in-place upscale then read it on the same coordinates as the upscaled scene.
    bool stretchNgxSubrectToFull(const Rc<DxvkImage>& image);
    // Scales the velocity buffer the bound motion-blur shader samples, once per frame.
    void stretchNgxVelocityBuffer();
    bool m_ngxVelocityStretchedThisFrame = false;
    void refreshNgxUe3VertexFactory();
    void armNgxFullBufferScreenPositionScaleBias();
    void noteNgxPostUpscaleViewport();
    void commitNgxPrePostInjection(D3DPRIMITIVETYPE primitiveType);

    // Per-draw snapshot of the bound texture slots, used by UI classification.
    // Invalidated when the bound set changes; bindings cannot change within a draw.
    struct BoundTextureSnapshot {
      uint32_t mask = 0; // bound slots with a valid common texture
      std::array<XXH64_hash_t, SamplerCount> imageHashes;
    };
    mutable BoundTextureSnapshot m_boundTextureSnapshot;
    mutable bool m_boundTextureSnapshotValid = false;
    const BoundTextureSnapshot& ensureBoundTextureSnapshot() const;

    // Per-frame snapshot of the scalar options read on the passthrough draw path.
    // Refreshed in EndFrame and lazily on the first frame's draw.
    struct FrameOptionCache {
      bool valid = false;

      bool orthographicIsUI = false;
      bool preTransformedVerticesIsUI = false;
      bool ue3EngineMode = false;
      bool ue3SkipSceneCapturePasses = false;
      bool sequenceTrackedLockWaits = true;
      bool ngxPassthroughJitter = false;
      bool ngxPrePostProcess = false;
      bool ngxDlfgHudless = false;
      bool ngxDlfgHudlessPreview = false;
      bool ngxNeuralRendering = false;
      bool ngxObjectVelocities = false;
      int ngxDebugVisualization = 0;

      const fast_unordered_set* uiTextures = nullptr;
    };
    FrameOptionCache m_frameOptions;
    void refreshFrameOptionCache();

    bool isRenderingUI();
  };
}
