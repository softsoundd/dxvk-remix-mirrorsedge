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

  enum class D3D9RtxFlag : uint32_t {
    DirtyLights,
    DirtyClipPlanes,
  };

  using D3D9RtxFlags = Flags<D3D9RtxFlag>;

  namespace PrepareDrawFlag {
    enum {
      Ignore                      = 0,
      CommitToRayTracing          = 1 << 0, // Process the current state as a part of ray tracing
      ApplyDrawState              = 1 << 1, // Submit draw state to dxvk-cs, so it can be used to issue original or non-original draw calls (e.g. terrain, sky)
      OriginalDrawCall            = 1 << 2, // Issue the original draw call
      PreserveDrawCallAndItsState = ApplyDrawState | OriginalDrawCall,
    };
  }
  using PrepareDrawFlags = uint32_t;

  // Exact UV dataflow analysis types (shared between the PS coordinate-origin resolver,
  // the VS interpolant->IA trace, and the per-draw UV decision in D3D9Rtx).
  //
  // A UvAffineTerm models one term of `value' = value * scale + offset` as a small sum:
  //   term = imm + consts[constReg][constComp] * factor + consts[constReg2][constComp2] * factor2
  // where each part is optional (immValid / constReg >= 0 / constReg2 >= 0; constReg2 is only
  // used when constReg is). When no part is set the term is absent (identity for scale, zero
  // for offset). Scale terms only ever carry a single part (immediate or one constant ref -
  // products of two draw-time constants are not representable); offset terms may legitimately
  // sum an immediate and up to two constant refs (UE3 materials add uniform-expression tile
  // offsets and literal centering biases to one coordinate). `inexact` marks terms that
  // encountered math not representable in this model (the origin may still be provable).
  struct UvAffineTerm {
    bool immValid = false;
    float imm = 0.0f;
    int16_t constReg = -1;
    uint8_t constComp = 0;
    float factor = 1.0f;
    int16_t constReg2 = -1;
    uint8_t constComp2 = 0;
    float factor2 = 1.0f;
    bool inexact = false;
  };

  // One coordinate component as `value * scale + offset`, or - for a UV matrix such as UE3's
  // Rotator - as a combination of two components of the same interpolant:
  //   value' = scale * uv[scaleComponent] + cross * uv[crossComponent] + offset
  // Center biases fold into `offset` as an immediate alongside the two matrix-row constants,
  // which is what the two constant slots of an offset term are for. A coefficient that would
  // become the product of two draw-time constants (a Panner feeding a Rotator) is not
  // representable and marks the term inexact.
  struct UvComponentAffine {
    UvAffineTerm scale;   // absent => 1.0
    UvAffineTerm offset;  // absent => 0.0
    UvAffineTerm cross;   // absent => 0.0, so an unmixed component is unaffected by it
    bool hasCross = false;
    uint8_t scaleComponent = 0;  // interpolant component `scale` multiplies when hasCross
    uint8_t crossComponent = 0;  // interpolant component `cross` multiplies when hasCross
  };

  // Deterministic resolution of the coordinate a pixel shader feeds into a sampler:
  // proves (or fails to prove) that both the U and V components of every sample site
  // originate from components of a single TEXCOORD interpolant, with an affine chain.
  struct PsSamplerUvOrigin {
    bool originValid = false;    // U/V proven to originate from one TEXCOORD interpolant
    bool sitesAgree = true;      // all valid sample sites agreed on origin + affine
    bool affineExact = false;    // affine chain fully representable for both components
    // disagreeing static-tiling sites resolved by keeping the highest-frequency one
    // (UE3 distance-fade anti-tiling idiom) instead of first-in-bytecode order
    bool preferredHighestFrequencySite = false;
    uint8_t semanticIndex = 0;   // TEXCOORD usage index of the source interpolant
    uint8_t compU = 0;           // interpolant component feeding sample U
    uint8_t compV = 1;           // interpolant component feeding sample V
    UvComponentAffine affineU;
    UvComponentAffine affineV;
    uint16_t validSiteCount = 0;
    uint16_t invalidSiteCount = 0;
  };

  // Classification of the VS-side path from an output TEXCOORD interpolant back to the IA.
  enum class Ue3VsUvTraceKind : uint8_t {
    Invalid = 0,     // origin could not be proven (procedural UVs, mixed inputs, unsupported ops)
    PureMove,        // interpolant components == IA texcoord set `.xy` exactly
    AffineConst,     // interpolant == IA texcoord `.xy` * scale + offset (constants/immediates)
    OriginOnly,      // origin proven but the VS math is not representable as an affine transform
  };


  //This class handles all of the RTX operations that are required from the D3D9 side.
  struct D3D9Rtx {
    friend class ImGUI; // <-- we want to modify these values directly.

    D3D9Rtx(D3D9DeviceEx* d3d9Device, bool enableDrawCallConversion = true);
    ~D3D9Rtx();

    RTX_OPTION("rtx", bool, orthographicIsUI, true, "When enabled, draw calls that are orthographic will be considered as UI.");
    RTX_OPTION("rtx", bool, preTransformedVerticesIsUI, false, "When enabled, draw calls using pre-transformed (screen-space) vertices will be considered as UI. This is typical for D3D8/D3D9 games that render UI with RHW vertices.");
    RTX_OPTION("rtx", bool, allowCubemaps, false, "When enabled, cubemaps from the game are processed through Remix, but they may not render correctly.");
    RTX_OPTION("rtx", bool, useVertexCapture, true, "When enabled, injects code into the original vertex shader to capture final shaded vertex positions.  Is useful for games using simple vertex shaders, that still also set the fixed function transform matrices.");
    RTX_OPTION("rtx", bool, useVertexCapturedNormals, true, "When enabled, vertex normals are read from the input assembler and used in raytracing.  This doesn't always work as normals can be in any coordinate space, but can help sometimes.");
    RTX_OPTION("rtx", bool, useVertexCapturedTexcoords, false, "When enabled, vertex shader output texcoords always override input texcoords from the vertex declaration. Enable for games where the vertex shader applies meaningful UV transformations that should be used for ray tracing (e.g. animated UVs via shader constants).");
    RTX_OPTION("rtx", bool, useWorldMatricesForShaders, true, "When enabled, Remix will utilize the world matrices being passed from the game via D3D9 fixed function API, even when running with shaders.  Sometimes games pass these matrices and they are useful, however for some games they are very unreliable, and should be filtered out.  If you're seeing precision related issues with shader vertex capture, try disabling this setting.");
    RTX_OPTION("rtx.d3d9", bool, ue3EngineMode, false,
               "Master toggle for Unreal Engine 3 D3D9 compatibility. Also defaults rtx.zUp to True, UE3 being a "
               "Z-up engine, unless a config file sets it explicitly.");
    RTX_OPTION("rtx.d3d9", bool, ue3CameraFromShaderConstants, false,
               "UE3 compat: derive World/View and View/Projection matrices from UE3 reserved shader constants. "
               "Implicitly enabled by rtx.d3d9.ue3EngineMode.");
    RTX_OPTION("rtx.d3d9", bool, ue3ObjectToWorldFromShaderConstants, false,
               "UE3 compat: extract LocalToWorld from vertex shader constants using shader CTAB and use it for object transforms. "
               "Implicitly enabled by rtx.d3d9.ue3EngineMode.");
    RTX_OPTION("rtx.d3d9", bool, autoRaytracedRenderTargetFromFullscreenComposite, false,
               "D3D9 compat: auto-detect an offscreen render target being used as the main scene (sampled in a fullscreen composite pass) "
               "and treat it as a raytraced render target to capture the correct geometry in games that upscale/composite to the backbuffer.");
    RTX_OPTION("rtx.d3d9", bool, rasterizeFullscreenCompositeToPrimary, false,
               "D3D9 compat: rasterise likely fullscreen composite/postprocess passes to the primary render target. Helps avoid raytracing a fullscreen quad.");
    RTX_OPTION("rtx.d3d9", bool, shaderPathTexcoordIndexFromPixelShader, false,
               "Shader-path compat: infer TEXCOORD set used by pixel shader rather than trusting D3DTSS_TEXCOORDINDEX. "
               "Helps UE3 games where fixed-function stage state is stale or incorrect when shaders are active. "
               "Implicitly enabled by rtx.d3d9.ue3EngineMode.");
    RTX_OPTION("rtx.d3d9", fast_unordered_set, vsTexcoordCaptureOutlierTextures, {},
               "Texture hashes for which VS-captured texcoords should be overridden with IA (input assembler) texcoords. "
               "Useful as a compatibility fallback when certain textures appear stretched due to incorrect VS texcoord capture.");
    RTX_OPTION("rtx.d3d9", bool, ue3SkipDepthPrepass, false,
               "UE3 multi-pass compat: skip draw calls using a position-only vertex declaration (no texcoords/colors), "
               "which are characteristic of UE3 depth prepass draws. The geometry will be captured during the base pass instead. "
               "Implicitly enabled by rtx.d3d9.ue3EngineMode.");
    RTX_OPTION("rtx.d3d9", bool, ue3SkipShadowDepthPasses, false,
               "UE3 multi-pass compat: skip draw calls targeting small square render targets (typical of shadow depth maps). "
               "Prevents shadow-pass geometry from being incorrectly captured as scene geometry. "
               "Implicitly enabled by rtx.d3d9.ue3EngineMode.");
    RTX_OPTION("rtx.d3d9", bool, ue3SkipDepthTestDisabledTranslucency, false,
               "UE3 translucency compat: skip alpha-blended draw calls that have depth test and depth write both disabled. "
               "These are typically UE3 NeedsDepthTestDisabled materials (e.g. fullscreen overlays, fog volume composites) "
               "that should not create RT geometry. Implicitly enabled by rtx.d3d9.ue3EngineMode.");
    RTX_OPTION("rtx.d3d9", bool, ue3SkipSceneCapturePasses, false,
               "UE3 multi-pass compat: fully ignore UE3 SceneCapture rendering. SceneCapture probes "
               "(SceneCapture2D/Reflect/Portal actors: security monitors, mirrors, scripted window reflections) "
               "re-render the world from their own camera before the main view into the shared SceneColor render "
               "target, with genuine ViewProjectionMatrix/CameraPosition constants - without this option the "
               "capture camera can steal the Main camera for a frame (momentary camera flips) and capture "
               "geometry is ingested into the ray-traced scene through mirrored or oblique-clipped views, "
               "corrupting it. World-geometry draws are dropped when any capture signal matches: viewport "
               "strictly smaller than half the backbuffer in both dimensions (probe-sized targets; exact-half "
               "splitscreen viewports are never matched), a sub-full viewport whose aspect does not match the "
               "backbuffer (e.g. square probe RTs on widescreen), a mirrored view-projection (negative 3x3 "
               "determinant, reflect probes' FMirrorMatrix), or a CTAB-declared camera that fails plausibility "
               "extraction (reflect/portal probes' oblique FClipProjectionMatrix near-plane clip; the main "
               "view always extracts). Sub-main-view-sized world draws also cannot update the Main camera. "
               "Skipped draws are removed entirely while ray tracing, so probe target textures show "
               "their last resolved content - visually equivalent to running with 'show scenecapture' toggled "
               "off. Implicitly enabled by rtx.d3d9.ue3EngineMode.");
    RTX_OPTION("rtx.d3d9", bool, conservativeOcclusionQueries, false,
               "Answer hardware occlusion queries with conservative results suited to path tracing instead "
               "of GPU-measured raster visibility: readbacks return immediately with full-backbuffer "
               "coverage ('assume unoccluded'), and the bounding-box test draws inside query brackets are "
               "ignored since nothing consumes a measurement. Raster-measured results misbehave under ray "
               "tracing: the depth buffer they test against is never populated (scene draws are consumed "
               "for RT instead of rasterized), and even a correct zero - an off-frustum, sub-pixel, edge-on "
               "or camera-enclosing bounding box - only means invisible to a rasterizer, while that geometry "
               "still drives reflections, GI and emissive lighting. Engines that hide meshes on 0-sample "
               "results (e.g. UE3) otherwise flicker mesh visibility and drop geometry from reflections, and "
               "their blocking readbacks stall the render thread on GPU completion (spinning synchronous "
               "bridge round trips for 32-bit games). Net effect is equivalent to disabling the game's "
               "occlusion culling, with no game-side console access, ini edits or patches required; "
               "pixel-count consumers (e.g. UE3 lens flare fading) see fully-visible. Only active while ray "
               "tracing is enabled. Implicitly enabled by rtx.d3d9.ue3EngineMode.");
    RTX_OPTION_ARGS("rtx.d3d9", bool, eventQueryCsCompletion, false,
                    "Report D3DQUERYTYPE_EVENT queries as complete once Remix's command stream thread has consumed the "
                    "game's commands up to the event, instead of once the GPU has executed them. Engines issue an event "
                    "after Present and poll it at the end of the next frame to stay at most one frame ahead of the GPU "
                    "(UE3's FrameSyncEvent). Under Remix a frame's GPU work is only submitted after Present, once "
                    "injectRTX has been recorded, so that throttle makes the game wait for the GPU frame plus the "
                    "injectRTX CPU time every frame and the GPU idles for the latter. Completing the event once the "
                    "draws have been captured lets the GPU keep a frame queued; frame time becomes the larger of the "
                    "CPU and GPU frame times. Costs up to one frame of input latency when the CPU is faster than the "
                    "GPU (Reflex re-paces the CPU against the GPU queue). Games that rely on the event to protect "
                    "D3DLOCK_NOOVERWRITE buffer reuse need GPU completion; Mirror's Edge does not use that lock mode. "
                    "Only active while ray tracing is enabled.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.d3d9", bool, skipRenderTargetCopies, true,
                    "Drop StretchRect copies out of a render target of 512x512 pixels or more that are issued before the "
                    "frame's ray tracing is injected and do not target the back buffer. UE3 copies its full-resolution "
                    "scene colour into resolve textures several times per frame for its post-processing chain, whose "
                    "output the ray-traced image replaces; the game does not read these targets back on the CPU. The "
                    "resolve textures keep whatever they last held. Only active while ray tracing is enabled.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.d3d9", bool, sequenceTrackedLockWaits, true,
                    "When a Lock has to wait for a resource, drain the command stream thread only up to the last command "
                    "that touched it (tracked per buffer and texture subresource: uploads, readbacks and the draws whose "
                    "geometry is captured for ray tracing) instead of everything queued, which includes the previous "
                    "frame's injectRTX recording once the game runs ahead of the GPU. Same model as current upstream "
                    "DXVK. Disable to fall back to the full drain.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION("rtx.d3d9", bool, ue3RequireCtabCameraConstants, false,
               "UE3 compat: only allow a draw call to update the Main camera when its vertex shader CTAB explicitly "
               "names both ViewProjectionMatrix and CameraPosition constants. Engine utility shaders (shadow depth, "
               "filters, etc.) do not declare these, so whatever data happens to live in the fallback camera registers "
               "(c0..c4) can otherwise be misinterpreted as a one-frame Main camera (e.g. a light-space matrix during "
               "UE3 light environment updates). Geometry from unverified draws is still rendered normally. "
               "Implicitly enabled by rtx.d3d9.ue3EngineMode.");
    RTX_OPTION("rtx.d3d9", bool, ue3StreamingStableTextureHashing, true,
               "UE3 compat: derive Remix texture hashes from the small-mip tail (mips at or below 64px, plus format and "
               "aspect ratio) instead of the top mip. UE3 texture streaming creates a new D3D9 texture object per "
               "mip-count change, so a top-mip hash differs per streamed variant of one logical texture and everything "
               "keyed on texture hashes (replacements, categories, tags, material identity) stops matching while a "
               "lower-mip variant is bound. The tail mips are present in every variant and copied byte-identically "
               "between them by the engine, so this hash is stable across streaming and texture LOD settings by "
               "construction - stateless and deterministic, no runtime learning. Unmipped textures and render targets "
               "keep the standard top-mip hash. "
               "Note: the two schemes produce different hashes, so texture tags and replacements only match under the "
               "setting they were authored with. "
               "Only active when rtx.d3d9.ue3EngineMode is enabled.");
    RTX_OPTION("rtx.d3d9", bool, ue3LogTextureHashProvenance, false,
               "UE3 compat: log how each Remix image hash was derived - the mip 0 content hash on its "
               "own, the mip count, whether the streaming-stable mip tail or the top mip produced the "
               "value, how many tail mips were hashed, the D3D9 usage and pool, and which subresources "
               "still had a pending upload when the hash was latched. One line per (texture shape, image "
               "hash) pair, so a texture whose hash moves between runs shows every value it took rather "
               "than only the first.\n"
               "The mip 0 hash is what makes this diagnostic worth reading. A hash is latched once and "
               "never recomputed, so when a material's descriptor hash is constant while its image hash "
               "changes per level load, comparing mip 0 separates the two possible causes: repeating "
               "while the full-chain hash moves means the picture is the same and the instability is in "
               "the smaller mips, which identity has no reason to depend on; moving as well means the "
               "material is genuinely binding a different texture.");
    RTX_OPTION("rtx.d3d9", bool, ue3LogUvResolution, false,
               "UE3 compat: log the deterministic UV resolution decision (proven IA set / captured interpolant / legacy fallback) "
               "once per unique pixel shader + stage combination, including ambiguity diagnostics.");
    RTX_OPTION("rtx.d3d9", fast_unordered_set, ue3UvTraceShaderHashes, {},
               "UE3 compat diagnostics: pixel shader bytecode hashes whose UV dataflow analysis is traced "
               "instruction by instruction to the log ([RTX-UV-TRACE] lines: opcode, operands, and the "
               "per-component origin/affine/constant-expression verdict after each write, plus every sampler "
               "site decision). The trace runs once when the shader is first analyzed. Use together with "
               "rtx.d3d9.ue3LogUvAffineDetail to root-cause atlas/tiling materials whose affine chain "
               "resolves inexactly.");
    RTX_OPTION("rtx.d3d9", bool, ue3LogUvAffineDetail, false,
               "UE3 compat diagnostics: log the full UV affine chain behind the deterministic UV resolution of "
               "shader-path draws: per-component scale, cross and offset terms (immediate or constant-register "
               "component with factor, and inexactness), the live resolved 2x2 plus translation, the gates that "
               "allowed or rejected writing the texture transform, and the pixel shader CTAB names/values of "
               "referenced constant registers. On the first sighting of a pixel shader it also dumps every "
               "sampler's UV origin and affine chain with the currently bound textures. Logs once per distinct "
               "resolved transform, capped per shader+stage. Use this to diagnose texture-atlas materials whose "
               "tile offset is not applied, or a UV matrix (UE3 Rotator) that resolves inexactly.");
    RTX_OPTION("rtx.d3d9", bool, ue3LogOcclusionQueries, false,
               "Occlusion query diagnostics: log bracketed test draws (fragment-test state, viewport, world "
               "AABB, whether the view origin sits inside it; capped), 0-sample completions with their "
               "recorded bracket contents (capped), and a periodic summary (bracket/draw/result counts, "
               "empty brackets, zero-result causes, pending readbacks, min/max sample counts). Diagnoses "
               "games whose hardware occlusion culling misbehaves under ray tracing - meshes hidden or "
               "flickering, render thread stalling on readbacks - by showing what was queried and what the "
               "game read back.");
    RTX_OPTION("rtx.d3d9", bool, deferredUiReplay, true,
               "Replay behavior for deferred UI overlay draws (rtx.deferredUiTextures / rtx.d3d9.deferredUiPixelShaders): "
               "when enabled, each tagged draw is snapshotted (vertex/index data, shaders, constants, textures, blend "
               "state) and re-issued on top of the ray-traced image immediately after RTX injection, below any UI the "
               "game rasterizes afterwards. This preserves the overlay's visual contribution without ending the "
               "ray-traced scene at the overlay's mid-frame draw position. When disabled, tagged draws are simply "
               "suppressed before injection (overlays become invisible while ray tracing, but still never break the scene).");
    RTX_OPTION("rtx.d3d9", fast_unordered_set, deferredUiPixelShaders, {},
               "Pixel shader bytecode hashes whose draws are treated as deferred UI overlays (see rtx.deferredUiTextures).\n"
               "Prefer this over texture tagging when the overlay samples a render target (e.g. UE3's scene color copy): "
               "RT image hashes change on recreation. Prefer a pixel shader tag, or one of the RT descriptor hashes the "
               "texture picker offers via rtx.deferredUiTextures. Whenever a "
               "tagged texture matches a draw, the runtime logs that draw's pixel shader hash ([RTX-DeferredUI] log lines) "
               "so the tag can be moved into this option.\n"
               "A pixel shader tag is treated as explicit intent: unlike texture tags it is not subject to the engine "
               "post-process shader exclusion, and it alone outranks the UE3 fullscreen post-process / fog-distortion "
               "classification, so an overlay whose shader looks like a screen-space contribution pass is still "
               "deferred rather than dropped (the world-geometry and depth-write guards still apply).");
    RTX_OPTION("rtx.d3d9", bool, deferredUiRefreshSceneColor, true,
               "When replaying deferred UI overlay draws, first copy the ray-traced output into any scene-color "
               "render-target texture the overlay samples (e.g. UE3's resolved SceneColorTexture). Overlay materials "
               "that read the scene (fade lerps, scope distortion, damage effects) then composite over the ray-traced "
               "image instead of the stale rasterized scene. The copy runs before each replayed draw, so chained "
               "effects see the previous overlay's output. Disable if a replayed overlay shows artifacts.");
    RTX_OPTION("rtx", bool, enableIndexBufferMemoization, true, "CPU performance optimization, should generally be enabled.  Will reduce main thread time by caching processIndexBuffer operations and reusing when possible, this will come at the expense of some CPU RAM.");
    RTX_OPTION("rtx", uint32_t, numGeometryProcessingThreads, 2, "The desired number of CPU threads to dedicate to geometry processing  Will be limited by the number of CPU cores.  There may be some advantage to lowering this number in games which are fairly simple and use a low number of draw calls per frame.  The default was determined by looking at a game with around 2000 draw calls per frame, and with a reasonably high average triangle count per draw.");

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

    /**
      * \brief: Initialize the D3D9 RTX interface
      */
    void Initialize();

    /**
      * \brief: Signal that an occlusion query has started for the current device
      */
    void BeginOcclusionQuery() {
      ++m_activeOcclusionQueries;
      ++m_oqBracketCounter;
      if (m_frameOptions.ue3LogOcclusionQueries) {
        ++m_oqDiag.brackets;
        m_oqDiag.drawsInCurrentBracket = 0;
        auto& record = m_oqRecords[m_oqBracketCounter & (kOcclusionQueryRecordCount - 1)];
        record = {};
        record.bracketId = m_oqBracketCounter;
      }
    }

    /**
      * \brief: Signal that an occlusion query has ended for the current device
      */
    void EndOcclusionQuery() {
      --m_activeOcclusionQueries;
      assert(m_activeOcclusionQueries >= 0);
      if (m_frameOptions.ue3LogOcclusionQueries &&
          m_activeOcclusionQueries == 0 &&
          m_oqDiag.drawsInCurrentBracket == 0) {
        // A measured empty bracket is guaranteed to report 0 samples passed.
        ++m_oqDiag.emptyBrackets;
      }
    }

    /**
      * \brief: Diagnostics: count an occlusion query readback that found the result not yet
      * available (see rtx.d3d9.ue3LogOcclusionQueries).
      */
    void TrackOcclusionQueryPendingRead() {
      if (m_frameOptions.ue3LogOcclusionQueries) {
        ++m_oqDiag.pendingReads;
      }
    }

    /**
      * \brief: Identifier of the most recently begun occlusion query bracket; stamped onto the
      * query object so its eventual result can be correlated with the recorded bracket contents.
      */
    uint32_t GetCurrentOcclusionBracketId() const {
      return m_oqBracketCounter;
    }

    /**
      * \brief: Diagnostics: record an occlusion query result as it becomes available to the
      * application (see rtx.d3d9.ue3LogOcclusionQueries).
      */
    void TrackOcclusionQueryResult(DWORD samplesPassed, uint32_t bracketId);

    /**
      * \brief: True when conservative occlusion query behaviour is active: occlusion query
      * readbacks are answered immediately with the synthesized "unoccluded" result and the
      * bracketed test draws are ignored. See rtx.d3d9.conservativeOcclusionQueries.
      */
    bool ConservativeOcclusionQueriesEnabled() const {
      return m_frameOptions.enableRaytracing &&
             (m_frameOptions.conservativeOcclusionQueries || m_frameOptions.ue3EngineMode);
    }

    // rtx.d3d9.eventQueryCsCompletion
    bool EventQueryCsCompletionEnabled() const {
      return m_frameOptions.enableRaytracing && m_frameOptions.eventQueryCsCompletion;
    }

    // rtx.d3d9.sequenceTrackedLockWaits
    bool SequenceTrackedLockWaitsEnabled() const {
      return m_frameOptions.sequenceTrackedLockWaits;
    }

    // True once this frame's ray tracing has been injected; later draws are UI / post work.
    bool IsRtxInjectTriggered() const {
      return m_rtxInjectTriggered;
    }

    // rtx.d3d9.skipRenderTargetCopies
    bool ShouldSkipRenderTargetCopy(const VkExtent3D& srcExtent, bool srcIsRenderTarget, bool dstIsBackBuffer) const {
      constexpr uint32_t kMinLargeTargetPixels = 512 * 512;
      return m_frameOptions.enableRaytracing &&
             m_frameOptions.skipRenderTargetCopies &&
             !m_rtxInjectTriggered &&
             srcIsRenderTarget &&
             !dstIsBackBuffer &&
             srcExtent.width * srcExtent.height >= kMinLargeTargetPixels;
    }

    void NoteRenderTargetCopySkipped() {
      ++m_drawDispositionStats.renderTargetCopiesSkipped;
    }

    /**
      * \brief: True while draws are being issued inside an occlusion query bracket and
      * conservative occlusion query behaviour applies to them.
      */
    bool ShouldApplyConservativeOcclusionQueryState() const {
      return m_activeOcclusionQueries > 0 && ConservativeOcclusionQueriesEnabled();
    }

    /**
      * \brief: The synthesized occlusion query result: full backbuffer coverage ("assume
      * unoccluded"), which keeps pixel-count consumers (e.g. UE3's LastPixelsPercentage
      * feeding lens flare fading) at fully-visible.
      */
    DWORD GetConservativeOcclusionQueryResult() const {
      if (m_activePresentParams.has_value()) {
        const DWORD area = DWORD(m_activePresentParams->BackBufferWidth) *
                           DWORD(m_activePresentParams->BackBufferHeight);
        if (area != 0) {
          return area;
        }
      }
      return 1u << 20;
    }

    /**
      * \brief: Signal that a parameter needs to be updated for RTX
      *
      * \param [in] flag: parameter that requires updating
      */
    void SetDirty(D3D9RtxFlag flag) {
      m_flags.set(flag);
    }

    /**
      * \brief: Signal that a transform has updated
      *
      * \param [in] idx: index of transform
      */
    void SetTransformDirty(const uint32_t transformIdx) {
      if (transformIdx > GetTransformIndex(D3DTS_WORLD)) {
        m_maxBone = std::max(m_maxBone, transformIdx - GetTransformIndex(D3DTS_WORLD));
      }
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
      */
    void EndFrame(const Rc<DxvkImage>& targetImage, bool callInjectRtx = true);

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

    static bool isLightmapTexture(XXH64_hash_t textureHash) {
      return lookupHash(RtxOptions::lightmapTextures(), textureHash);
    }

  private:
    D3D9DeviceEx* m_parent;

    std::optional<D3DPRESENT_PARAMETERS> m_activePresentParams;

    D3D9RtxFlags m_flags = 0xFFFFffff;

    uint32_t m_drawCallID = 0;
    // Note: A frame identifier the the main thread holds on to passed down into thread invocations such that
    // Reflex markers have a consistent ID despite executing in parallel (as typical methods of getting a frame ID
    // in DXVK depend on say when the submit thread's present happens which is unpredictable).
    uint64_t m_reflexFrameId = 0;

    uint32_t m_maxBone = 0;

    const bool m_enableDrawCallConversion;
    bool m_rtxInjectTriggered = false;
    DWORD m_texcoordIndex = 0;
    DWORD m_iaTexcoordIndex = 0;
    uint8_t m_texcoordCompU = 0;
    uint8_t m_texcoordCompV = 1;

    // Per-draw result of the deterministic UV resolution:
    // - LegacyTss: no provable resolution, behave like upstream (TSS index + optional capture fallbacks)
    // - ProvenIa: PS origin + VS trace proved an exact IA texcoord set; use IA texcoords
    // - CaptureInterpolant: PS origin proven but the VS path is procedural/unprovable;
    //   capture the exact interpolant components from the VS output instead of guessing an IA set
    enum class UvResolutionMode : uint8_t {
      LegacyTss = 0,
      ProvenIa,
      CaptureInterpolant,
    };
    UvResolutionMode m_uvResolutionMode = UvResolutionMode::LegacyTss;

    // two pass translucency dedup - track previous draw's shader/texture/geometry state
    // to detect UE3 back+front face translucency passes on the same mesh. The geometry
    // identity is required: UE3 sorts translucent prims back-to-front per camera, so
    // consecutive draws of different meshes sharing one material are common and must
    // never dedup against each other. Reset at frame end (EndFrame).
    XXH64_hash_t m_prevDrawVsPsHash = 0;
    XXH64_hash_t m_prevDrawTextureHash = 0;
    XXH64_hash_t m_prevDrawGeometryHash = 0;
    DWORD m_prevDrawCullMode = 0;

    int m_activeOcclusionQueries = 0;
    uint32_t m_oqBracketCounter = 0;

    // Diagnostics for rtx.d3d9.ue3LogOcclusionQueries: bracket/result statistics flushed
    // periodically from EndFrame, plus capped one-off detail logs.
    struct OcclusionQueryDiagnostics {
      uint32_t brackets = 0;
      uint32_t emptyBrackets = 0;
      uint32_t bracketedDraws = 0;
      uint32_t drawsInCurrentBracket = 0;
      uint32_t results = 0;
      uint32_t zeroResults = 0;
      uint32_t zeroCameraInside = 0;
      uint32_t zeroSmallViewport = 0;
      uint32_t pendingReads = 0;
      DWORD minResult = ~0u;
      DWORD maxResult = 0;
      uint32_t framesSinceSummary = 0;
      uint32_t stateSnapshotLogsRemaining = 12;
      uint32_t zeroResultLogsRemaining = 24;
    };
    OcclusionQueryDiagnostics m_oqDiag;

    // Per-bracket record of what was drawn inside an occlusion query, so a query result that
    // arrives frames later can be correlated with the geometry that produced it. Ring-indexed
    // by bracket id; sized for several frames of query traffic.
    struct OcclusionQueryBracketRecord {
      uint32_t bracketId = 0;
      uint16_t drawCount = 0;
      uint16_t primCount = 0;
      uint16_t viewportW = 0;
      uint16_t viewportH = 0;
      bool conservativeActive = false;
      bool cameraInsideBox = false;
      float cameraToBoxDistance = 0.0f;
      Vector3 boxMin = Vector3(0.0f);
      Vector3 boxMax = Vector3(0.0f);
      Vector3 cameraPos = Vector3(0.0f);
    };
    static constexpr uint32_t kOcclusionQueryRecordCount = 4096; // power of two
    std::array<OcclusionQueryBracketRecord, kOcclusionQueryRecordCount> m_oqRecords = {};

    Rc<DxvkBuffer> m_vsVertexCaptureData;

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

    // D3D9 hardware instancing state of the current draw. UE3 places foliage and NxFluid mesh
    // particles with one instanced draw whose per-instance transform lives in a vertex stream
    // (InstanceOffset + InstanceXAxis/YAxis/ZAxis as TEXCOORD1..4), not in a shader constant, so
    // both the placement and the capture-safety decisions have to come from the declaration.
    struct Ue3InstancingInfo {
      uint32_t instanceCount = 1;
      // Streams marked D3DSTREAMSOURCE_INSTANCEDATA: advanced per instance rather than per vertex,
      // so nothing in them may ever be read as a per-vertex attribute.
      uint32_t instanceDataStreamMask = 0;
      // Set when TEXCOORD1..4 form a complete FLOAT3 instance basis on one instance-data stream.
      bool hasInstanceTransform = false;
      uint32_t transformStream = 0;
      uint32_t offsetByteOffset = 0;
      uint32_t axisByteOffsets[3] = {};

      bool isInstanced() const { return instanceCount > 1 || instanceDataStreamMask != 0; }
    };

    static Ue3InstancingInfo resolveUe3Instancing(const D3D9VertexElements& elements,
                                                  const std::array<UINT, caps::MaxStreams>& streamFreq,
                                                  uint32_t instanceCount);

    Ue3InstancingInfo m_currentUe3Instancing;


    // Keeps UE3's vertex lightmap coefficient streams and per-instance transform streams from
    // being mistaken for the surface's UVs.
    uint32_t resolveIaTexcoordAvoidingNonUvElements(uint32_t iaTexcoordIdx) const;

    // Half-or-larger in both dims (allows ScreenPercentage >50%) and aspect-matched to the
    // backbuffer so square SceneCapture RTs cannot pass as main-view-sized on widescreen.
    static bool ue3ViewportAspectMatchesBackbuffer(uint32_t vpW, uint32_t vpH, uint32_t bbW, uint32_t bbH);
    static bool ue3ViewportIsMainViewSized(uint32_t vpW, uint32_t vpH, uint32_t bbW, uint32_t bbH);

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
      bool hasVideoSampler = false;
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

      bool looksLikeDofAndBloomPostProcess() const {
        return hasBlurredImageSampler ||
               (hasDofPackedParameters && hasDofMinMaxBlurClamp) ||
               (hasFilterTextureSampler && hasFilterSampleWeights);
      }

      // TdToneMapping capture support: sampler indices of the baked colour
      // curve LUT textures and float register indices of the grade constants
      // (-1 / 0xFF when not present in the shader's CTAB).
      uint8_t colorCurvesKSamplerIndex = 0xFF;
      uint8_t colorCurvesMSamplerIndex = 0xFF;
      int16_t toneMapSceneShadowsReg = -1;         // SceneShadowsAndDesaturation
      int16_t toneMapInverseHighLightsReg = -1;    // SceneInverseHighLights
      int16_t toneMapMidTonesReg = -1;             // SceneMidTones
      int16_t toneMapScaledLumaWeightsReg = -1;    // SceneScaledLuminanceWeights
      int16_t toneMapGammaColorScaleReg = -1;      // GammaColorScaleAndInverse
      int16_t toneMapGammaOverlayReg = -1;         // GammaOverlayColor
    };

    fast_unordered_cache<Ue3ShaderFeatureInfo> m_ue3ShaderFeatureCache;
    Ue3ShaderFeatureInfo getUe3ShaderFeatureInfo(const D3D9CommonShader* shader);
    Ue3PassType classifyUe3Pass(const DrawContext& drawContext);
    static const char* describeUe3VertexFactory(Ue3VertexFactoryType type);
    static const char* describeUe3PassType(Ue3PassType type);
    bool trackUe3MovieTextureRenderTarget(const char* reason);
    bool isUe3MovieTextureDescHash(XXH64_hash_t descHash) const;

    // TdToneMapping capture: the game uploads its baked/blended colour curves
    // as two 16x1 float textures (ColorCurvesK/ColorCurvesM) each frame; the
    // texel payloads are snooped at upload/unlock time (keyed by destination
    // texture) and joined with the tonemap pass's pixel shader constants when
    // the fullscreen tonemap draw is classified.
    static constexpr uint32_t kUe3CurveTexelCount = 16;

    // Captures the TdToneMapping grade constants + curve texels once per
    // frame at the (not raytraced) tonemap draw and forwards them to the
    // renderer.
    void maybeCaptureUe3ToneMapState();

    struct Ue3VsShaderCtabInfo {
      bool initialized = false;
      bool hasLocalToWorld = false;
      uint32_t localToWorldRegisterIndex = 0;
      uint32_t localToWorldRegisterCount = 0;

      bool hasWorldToLocal = false;
      uint32_t worldToLocalRegisterIndex = 0;
      uint32_t worldToLocalRegisterCount = 0;

      bool hasViewProjectionMatrix = false;
      uint32_t viewProjectionMatrixRegisterIndex = 0;
      uint32_t viewProjectionMatrixRegisterCount = 0;

      bool hasCameraPosition = false;
      uint32_t cameraPositionRegisterIndex = 0;
      uint32_t cameraPositionRegisterCount = 0;

      bool hasBoneMatrices = false;
      uint32_t boneMatricesRegisterIndex = 0;
      uint32_t boneMatricesRegisterCount = 0;

      // vertex-factory style hints inferred from VS CTAB constant names
      // these are used to relax/adjust shader-path UV heuristics where packed UVs
      // and UV offsets are expected (decal/terrain/speedtree paths)
      bool hasDecalTransform = false;
      bool hasDecalLocation = false;
      bool hasDecalOffset = false;
      bool hasTextureCoordinateScaleBias = false;
      bool hasLightMapCoordinateScaleBias = false;
      bool hasShadowCoordinateScaleBias = false;
      bool hasViewToLocal = false;
      bool hasWindMatrices = false;
    };

    // Full CTAB symbol table per vertex shader. Deliberately kept out of Ue3VsShaderCtabInfo,
    // which is copied by value into m_currentUe3CtabInfo on every draw - putting strings in there
    // would allocate per draw. Two consumers: naming registers for the constant-churn diagnostic,
    // and resolving which registers are shading-only for the hash exclusions below. Filled
    // unconditionally at parse time (once per unique shader) so enabling the diagnostic
    // mid-session still names registers for shaders already seen.
    struct Ue3VsConstantSymbol {
      uint32_t registerIndex = 0;
      uint32_t registerCount = 0;
      std::string name;
    };
    fast_unordered_cache<std::vector<Ue3VsConstantSymbol>> m_ue3VsConstantSymbols;
    // Names the register, e.g. "c12 (LocalToWorld[1])", falling back to the bare register.
    std::string describeUe3VsConstantRegister(XXH64_hash_t vsBytecodeHash, uint32_t reg) const;

    // Merged register ranges the stable VS hash skips, resolved once per vertex shader because
    // they are a pure function of its CTAB. Both variants are precomputed so the per-draw path
    // never builds, sorts or merges anything - it just hashes the gaps. Kept in a side map rather
    // than in Ue3VsShaderCtabInfo, which is copied by value on every draw.
    struct Ue3VsHashExclusions {
      // Six come from fixed sources (two reserved camera registers, their CTAB overrides, and the
      // two transform matrices); the rest are one per shading-only CTAB symbol. Overflowing only
      // leaves a register in the hash that would have been excluded, so a generous cap is enough.
      static constexpr uint32_t kMaxRanges = 16;
      struct Range {
        uint32_t begin = 0;
        uint32_t end = 0;
      };
      std::array<Range, kMaxRanges> cameraOnly = {};
      uint32_t cameraOnlyCount = 0;
      // Camera registers plus the shading-only constants (LightMapScale, lightmap/shadow
      // coordinate scale-bias). LightMapScale holds a different element count and different
      // values per lightmap policy, so leaving it in makes the geometry hash move with the
      // DirectionalLightmaps setting. Unlike the object transform these registers cost nothing
      // to drop - they never reach a vertex position - so UE3 mode always uses this variant.
      std::array<Range, kMaxRanges> cameraAndShading = {};
      uint32_t cameraAndShadingCount = 0;
      // Camera registers plus the object transform and shading-only constants.
      std::array<Range, kMaxRanges> withPlacement = {};
      uint32_t withPlacementCount = 0;
    };

    // keyed by XXH3 hash of the vertex shader DXSO bytecode
    fast_unordered_cache<Ue3VsShaderCtabInfo> m_ue3VsShaderCtabCache;
    std::optional<Ue3VsShaderCtabInfo> m_currentUe3CtabInfo;

    // NGX passthrough mode: minimal CTAB scan for the camera symbols plus what the object
    // velocity capture needs. Kept as a separate cache from m_ue3VsShaderCtabCache so the
    // passthrough path never populates partial entries into the full ray tracing parse cache.
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
    };
    fast_unordered_cache<NgxSpsbCtabReg> m_ngxSpsbCtabCache;
    NgxSpsbCtabReg scanNgxSpsbCtabReg(const std::vector<uint8_t>& bytecode) const;

    // Per-draw ScreenPositionScaleBias patch state, decided in prepareDrawForNgxPassthrough
    // and consumed by the device's constant upload (cleared before the injection trigger
    // draw: everything after the injection samples un-jittered DLSS output)
    bool m_ngxSpsbPatchActive = false;
    NgxSpsbCtabReg m_ngxSpsbPatchVs;
    NgxSpsbCtabReg m_ngxSpsbPatchPs;
    float m_ngxSpsbPatchAdd[2] = { 0.0f, 0.0f };

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

    // Scene-camera viewport gate; 0 = unknown.
    float m_ngxGameScreenPercentage = 0.0f;
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
    fast_unordered_cache<Ue3VsHashExclusions> m_ue3VsHashExclusionCache;
    // Borrowed from the cache above for the duration of the draw; never owned.
    const Ue3VsHashExclusions* m_currentUe3VsHashExclusions = nullptr;
    static Ue3VsHashExclusions buildUe3VsHashExclusions(const Ue3VsShaderCtabInfo& ctabInfo,
                                                        const std::vector<Ue3VsConstantSymbol>* symbols);

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

    // pixel shader texcoord inference cache (for shader-path UV selection)
    struct PsSamplerTexcoordEntry {
      bool initialized = false;
      // exact per-sampler coordinate origin resolution (authoritative for the UV decision)
      std::array<PsSamplerUvOrigin, caps::MaxTexturesPS> samplerUvOrigin;
      // statistical inference below is used for diffuse-sampler *scoring* only
      std::array<int8_t, caps::MaxTexturesPS> samplerToTexcoord;
      std::array<uint8_t, caps::MaxTexturesPS> samplerCoordCompValid;
      std::array<uint8_t, caps::MaxTexturesPS> samplerCoordCompU;
      std::array<uint8_t, caps::MaxTexturesPS> samplerCoordCompV;
      std::array<uint8_t, caps::MaxTexturesPS> samplerSemanticFlags;
      std::array<uint16_t, caps::MaxTexturesPS> samplerExpressionFlags;
      std::array<uint16_t, caps::MaxTexturesPS> samplerSampleCount;
      std::array<int16_t, caps::MaxTexturesPS> samplerScaleConstReg;
      std::array<uint8_t, caps::MaxTexturesPS> samplerScaleConstCompU;
      std::array<uint8_t, caps::MaxTexturesPS> samplerScaleConstCompV;
      std::array<float, caps::MaxTexturesPS> samplerScaleFactorU;
      std::array<float, caps::MaxTexturesPS> samplerScaleFactorV;
      std::array<uint8_t, caps::MaxTexturesPS> samplerScaleImmediateValid;
      std::array<float, caps::MaxTexturesPS> samplerScaleImmediateU;
      std::array<float, caps::MaxTexturesPS> samplerScaleImmediateV;
      std::array<int16_t, caps::MaxTexturesPS> samplerOffsetConstReg;
      std::array<uint8_t, caps::MaxTexturesPS> samplerOffsetConstCompU;
      std::array<uint8_t, caps::MaxTexturesPS> samplerOffsetConstCompV;
      std::array<float, caps::MaxTexturesPS> samplerOffsetFactorU;
      std::array<float, caps::MaxTexturesPS> samplerOffsetFactorV;
      std::array<uint8_t, caps::MaxTexturesPS> samplerOffsetImmediateValid;
      std::array<float, caps::MaxTexturesPS> samplerOffsetImmediateU;
      std::array<float, caps::MaxTexturesPS> samplerOffsetImmediateV;
      // Sampler registers holding UE3 lightmap machinery (LightMapTextures[] coefficients and
      // the bicubic B-spline weight LUT). Their count is a function of the DirectionalLightmaps
      // setting - 3 coefficients vs 1 - so every draw-time decision that reads the bound texture
      // set has to subtract them or it varies with a setting Remix does not care about.
      uint32_t lightmapSamplerMask = 0;
    };

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
    const char* m_ue3LastDrawDecision = "";

    struct DrawDispositionStats {
      uint32_t frames = 0;
      uint64_t draws = 0;
      uint64_t rayTraced = 0;
      uint64_t rasterized = 0;              // draws whose original draw call executes on the GPU
      uint64_t rasterizedPrims = 0;
      uint64_t rasterizedForCapture = 0;    // ray-traced draws kept only so the vertex shader runs for vertex capture
      uint64_t rasterizedForCapturePrims = 0;
      uint64_t rasterizedPostInjection = 0; // UI and other draws after injectRTX
      uint64_t ignored = 0;
      uint64_t renderTargetCopiesSkipped = 0; // StretchRects dropped by rtx.d3d9.skipRenderTargetCopies
    };
    DrawDispositionStats m_drawDispositionStats;

    static bool isPrimitiveSupported(const D3DPRIMITIVETYPE PrimitiveType) {
      return (PrimitiveType == D3DPT_TRIANGLELIST || PrimitiveType == D3DPT_TRIANGLEFAN || PrimitiveType == D3DPT_TRIANGLESTRIP);
    }

    const Direct3DState9& d3d9State() const;

    void flushOcclusionQueryDiagnostics();

    void triggerInjectRTX();

    bool checkBoundTextureCategory(const fast_unordered_set& textureCategory) const;

    // --- NGX passthrough mode (see RtxNgxPassthrough) ---
    // All draws rasterize as-is; the per-draw work reduces to UE3 camera extraction, scene
    // color/depth target identification (for viewport jitter and DLSS inputs) and the thin
    // RTX injection trigger at scene end.
    PrepareDrawFlags prepareDrawForNgxPassthrough(const DrawContext& drawContext);
    void tryNgxPassthroughCameraCapture();
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

    // Set when the engine's upscaling draw turned out to also be its post-process composite, so
    // neither replacing it (the grade is lost) nor leaving it (it upscales itself, bilinearly)
    // is correct. Such a title only supports full resolution DLAA.
    bool m_ngxEngineCompositeUnsafe = false;

    // Pre-post-process injection: when the scene end trigger was the first post-process pass
    // sampling the scene color, DLSS reads and writes the image that pass consumes - the
    // scene color render surface or one of its resolve destinations (m_ngxSubrect carries
    // the scene viewport rect) - and the game's post chain picks up the result. The mirror
    // is the scene color surface when the target is a resolve destination, so later
    // re-resolves of the surface propagate the anti-aliased content too.
    Rc<DxvkImage> m_ngxColorTargetImage;
    Rc<DxvkImage> m_ngxColorMirrorImage;

    // Same-size StretchRect destinations copied from the scene color surface this frame
    // (UE3 D3D9 scene color resolves); sampled by the post chain in place of the surface
    std::array<Rc<DxvkImage>, 4> m_ngxSceneColorResolves;
    uint32_t m_ngxSceneColorResolveCount = 0;

    // HUD-less capture for frame generation: when the injection ran pre-post-process, the
    // pre-UI backbuffer state is captured at the first UI-classified backbuffer draw after
    // the injection (or at frame end when no UI is drawn)
    bool m_ngxHudlessCapturedThisFrame = false;

    void maybeCaptureNgxHudless(const DrawContext& drawContext);

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
    void reportNgxPrePostMiss();

    bool ngxPrePostInjectionAllowed() const;
    Rc<DxvkImage> matchNgxSceneColorSample(const DxvkImage* sampledImage) const;
    void recordNgxPrePostSceneColorRead();
    void engageNgxPrePostInjection(const Rc<DxvkImage>& matchedTarget);
    Rc<DxvkImage> findNgxPrePostSceneColorFromSamplers(bool applyOrdinalGate, bool recordRead);
    static bool ngxSceneViewportIsFullSize(const D3DVIEWPORT9& sceneViewport,
                                           uint32_t backBufferWidth, uint32_t backBufferHeight);
    static bool ngxSceneViewportIsSubrect(const D3DVIEWPORT9& sceneViewport,
                                          uint32_t backBufferWidth, uint32_t backBufferHeight);
    static bool ngxShaderIsFinishRenderViewTargetGamma(const Ue3ShaderFeatureInfo& psInfo);

    // Per-draw snapshot of the bound texture slots, used by UI classification.
    // Invalidated when the bound set changes; bindings cannot change within a draw.
    struct BoundTextureSnapshotEntry {
      D3D9CommonTexture* texture = nullptr;
      XXH64_hash_t imageHash = kEmptyHash;
      // RT-only absolute descriptor hash (includes Width/Height); also copied into
      // descriptorHash below for MIC / identity paths that need a size-dependent key.
      XXH64_hash_t rtDescriptorHash = 0;
      // RT-only aspect-normalized descriptor hash used by deferred-UI / raytraced-RT tagging.
      XXH64_hash_t rtResolutionAgnosticDescriptorHash = 0;
      // Descriptor hash of every image (stable across recreation, unlike imageHash for
      // GPU-written or session-composited textures).
      XXH64_hash_t descriptorHash = 0;
      bool hasImage = false;
      bool hasSampleView = false;
      bool isRenderTarget = false;
    };
    struct BoundTextureSnapshot {
      uint32_t mask = 0; // bound slots with a valid common texture
      std::array<BoundTextureSnapshotEntry, SamplerCount> entries;
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
      bool conservativeOcclusionQueries = false;
      bool eventQueryCsCompletion = false;
      bool sequenceTrackedLockWaits = true;
      bool skipRenderTargetCopies = true;
      bool ue3LogOcclusionQueries = false;
      bool ngxPassthroughMode = false;
      bool ngxPassthroughJitter = false;
      bool ngxPrePostProcess = false;
      bool ngxDlfgHudless = false;
      bool ngxObjectVelocities = false;
      int ngxDebugVisualization = 0;

      bool enableRaytracing = false;
      bool logReplacementResolution = false;

      const fast_unordered_set* uiTextures = nullptr;
    };
    FrameOptionCache m_frameOptions;
    void refreshFrameOptionCache();

    bool isRenderingUI();

    void submitActiveDrawCallState();
  };
}
