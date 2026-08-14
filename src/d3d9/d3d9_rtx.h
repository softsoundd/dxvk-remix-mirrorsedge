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

  struct UvComponentAffine {
    UvAffineTerm scale;   // absent => 1.0
    UvAffineTerm offset;  // absent => 0.0
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

  // Where vertex capture reads a draw's positions from. Both exact sources are independent
  // of view depth; ClipReconstruction inverts the projection, whose error grows with the
  // square of view depth over the near plane and shears distant geometry.
  enum class Ue3CapturePositionSource : uint8_t {
    ClipReconstruction = 0, // unproject the vertex shader's clip-space output (inexact)
    InputAssembler,         // IA object-space positions, for meshes the shader only moves rigidly
    PreProjectionRegister,  // the world position the shader itself computed, read back directly
  };

  enum class Ue3CapturePositionSourceOverride : int {
    Auto = 0,
    ForcePreProjection,
    ForceInputAssembler,
    ForceReconstruction,
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
    RTX_OPTION("rtx.d3d9", bool, ue3MaterialInstanceConstantHash, false,
               "UE3 MaterialInstanceConstant support: deterministic child-level material identity composed of the "
               "pixel shader bytecode hash, the ordered set of textures bound to the shader's material samplers "
               "(CTAB names Texture2D_*/TextureCube_*), and the shader's material constants (CTAB UniformVector_*/"
               "UniformScalar_* registers). Distinguishes material instances by their TextureParameterValues, "
               "StaticSwitchParameters and VectorParameterValues/ScalarParameterValues, enabling tagging at the "
               "child level instead of broadly at the parent level. Shaders whose constant registers carry "
               "frame-varying expression values (Time, fades, sub-UV frames) must be listed in "
               "rtx.d3d9.ue3MicConstantIdentityExcludedShaders or their hashes churn every frame. "
               "Implicitly enabled by rtx.d3d9.ue3EngineMode.");
    RTX_OPTION("rtx.d3d9", bool, ue3LightmapPermutationInvariantHash, false,
               "UE3 MaterialInstanceConstant support: make material identity invariant to the UE3 lightmap "
               "shader permutations selected by system settings. UE3 recompiles every lightmapped base-pass "
               "pixel shader per lightmap policy (DirectionalLightmaps=True: 3-coefficient TEXTURE_LIGHTMAP, "
               "False: 1-coefficient SIMPLE_TEXTURE_LIGHTMAP; Mirror's Edge TdBicubicFiltering adds a bicubic "
               "filtering permutation with LightMapResolution/BSplineTexture symbols), so the bytecode hash "
               "seeding material identity - and with it every material hash used for tagging and replacements - "
               "changes when those settings flip. With this option, draws whose shader CTAB declares lightmap "
               "policy symbols - in the pixel shader (texture lightmaps: LightMapTextures et al) or only in the "
               "vertex shader (vertex lightmaps: LightMapScale; their lightmap reaches the pixel shader through "
               "interpolators) - use permutation-stable identity inputs instead: the seed is a canonical "
               "signature of the material sampler declarations (Texture2D_*/TextureCube_*/Texture3D_* names), "
               "and the texture set and Uniform* constants are keyed by symbol name rather than by register, "
               "since register assignments and per-permutation unreferenced elements shift between permutations. "
               "Shaders without lightmap symbols on either stage keep their bytecode-seeded hashes. Note: "
               "enabling this changes the material hashes of lightmapped materials once (existing "
               "material-hash-keyed work must be redone). Residual variance remains for materials with "
               "parameters referenced by only one permutation: the simple-lightmap compile strips samplers and "
               "uniforms used exclusively by specular/two-sided-lighting expressions, so such materials keep "
               "one stable identity per lightmap policy state. rtx.d3d9.ue3LightmapPermutationBridgeLookup "
               "bridges replacement lookups for them (author under DirectionalLightmaps=False); the "
               "constants-free shader+textures identity tier also still matches across permutations unless a "
               "sampler itself was stripped. "
               "Implicitly enabled by rtx.d3d9.ue3EngineMode.");
    RTX_OPTION("rtx.d3d9", bool, ue3LightmapPermutationBridgeLookup, false,
               "UE3 MaterialInstanceConstant support: bridge replacement lookups across lightmap policy "
               "permutations for materials that cannot share one identity. The simple-lightmap compile "
               "(DirectionalLightmaps=False) strips material samplers and uniforms referenced only by "
               "specular/two-sided-lighting expressions, so such materials genuinely bind different data per "
               "state and rtx.d3d9.ue3LightmapPermutationInvariantHash alone cannot unify them. With this "
               "option, draws running under the richer permutation also compute the identity hashes they "
               "would have produced with small symbol subsets removed (up to 2 samplers and 2 uniforms, plus "
               "a constants-free variant), and the replacement lookup tries those alternates after its normal "
               "tiers. Dropping exactly the stripped symbols reproduces the simpler state's hash bit-for-bit, "
               "so replacements authored/captured under DirectionalLightmaps=False apply under =True with no "
               "aliasing heuristics - an alternate either reconstructs a captured identity exactly or misses. "
               "The reverse direction is impossible (the simpler compile lacks the stripped data), so author "
               "material replacements against the simple-lightmap state. "
               "Implicitly enabled by rtx.d3d9.ue3EngineMode.");
    RTX_OPTION("rtx.d3d9", fast_unordered_set, ue3MicConstantIdentityExcludedShaders, {},
               "UE3 MaterialInstanceConstant support: pixel shader bytecode hashes whose UniformVector_*/"
               "UniformScalar_* constants are excluded from material identity hashing. UE3 evaluates material "
               "uniform expressions on the CPU every draw, so shaders using Time/panner/fade/sub-UV expressions "
               "receive frame-varying values in the same constant registers as stable material instance "
               "parameters; folding those into the hash would mint a new material identity every frame. "
               "Such shaders announce themselves as an endless stream of new materialHash lines when "
               "rtx.d3d9.ue3LogMaterialInstanceHash is enabled (a churn warning names the shader once a "
               "threshold is crossed) - add the reported hash here. For lightmap-bearing shaders under "
               "rtx.d3d9.ue3LightmapPermutationInvariantHash the reported value is the canonical shader "
               "identity; both it and the raw bytecode hash are honoured. Excluded shaders fall back to "
               "pixel shader + material texture set identity.");
    RTX_OPTION("rtx.d3d9", bool, ue3LogMaterialInstanceHash, false,
               "UE3 MaterialInstanceConstant support: log a one-shot per-material breakdown of the material "
               "identity hash (pixel shader hash, material texture set with per-sampler image hashes, constant "
               "ranges and hash, and the final hash), plus a churn warning naming any shader that mints an "
               "abnormal number of distinct hashes (a sign its constants are frame-varying and it belongs in "
               "rtx.d3d9.ue3MicConstantIdentityExcludedShaders).");
    RTX_OPTION("rtx.d3d9", bool, ue3MicExcludeRenderTargetsFromIdentity, true,
               "UE3 MaterialInstanceConstant support: exclude render-target-backed textures from the material "
               "texture-set identity hash. Render targets bound as material samplers (scene captures, "
               "reflection buffers - e.g. the capture Mirror's Edge routes into the first-person body "
               "materials) receive a new image hash every time the game recreates them (respawn, checkpoint "
               "reload, level load). Folding that hash into material identity re-mints the material hash on "
               "every recreation, silently breaking texture tags and asset replacements anchored on the "
               "identity: they only match again when the render target happens to reproduce its capture-time "
               "hash. Excluding render targets (treating them like hashless textures, which were always "
               "skipped) keeps material identity stable across recreations. Note: identities that previously "
               "included a render-target hash change once when this option turns on - re-anchor affected "
               "replacements (rtx.logReplacementResolution logs the new hashes).");
    RTX_OPTION("rtx.d3d9", fast_unordered_set, ue3MicIdentityExcludedTextureDescHashes, {},
               "UE3 MaterialInstanceConstant support: descriptor hashes of textures to exclude from the "
               "material texture-set identity hash. Some engine-composited textures are re-uploaded with "
               "different contents every session, so their content-based image hash is session-unique and "
               "any material identity that includes them changes across sessions - replacements anchored on "
               "such identities silently stop matching (they were authored against one session's hash). A "
               "texture's *descriptor* hash is stable across recreations: find it in the "
               "rtx.d3d9.ue3LogMaterialInstanceHash breakdown (textures=[sN:0x<image>(desc:0x<descriptor>)]) "
               "or in [RTX-MicDrift] sampler diffs, add it here, then re-anchor the affected material once - "
               "its identity is stable from then on. Note descriptor hashes derive from texture properties "
               "(dimensions/format/usage), so identically-shaped textures share one and the exclusion "
               "applies to all of them - usually desirable for the engine-composited textures this option "
               "targets. UE3-streamed textures recreate at a different size per resident mip level and so "
               "carry one descriptor hash per size; the composited/dynamic textures this option is meant "
               "for are fixed-size.");
    RTX_OPTION("rtx.d3d9", bool, ue3MicPersistAutoExcludedConstantGroups, true,
               "UE3 MaterialInstanceConstant support: persist the constants-churn auto-exclusion set "
               "(see rtx.d3d9.ue3MicAutoExcludeFrameVaryingConstants) across sessions in "
               "rtx-remix/ue3MicAutoExcludedGroups.cache. Without persistence a churning material group is "
               "only excluded after it re-mints enough distinct hashes within the session, so its material "
               "identity flips mid-session at an unpredictable point - replacements anchored on either side "
               "of the flip only match part of the time. With persistence the exclusion applies from the "
               "first frame of every later session, making such identities deterministic. Delete the cache "
               "file to reset learned exclusions.");
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
    RTX_OPTION("rtx.d3d9", bool, ue3ForegroundDpgIsViewModel, true,
               "UE3 compat: classify SDPG_Foreground draws as view-model (first-person overlay) geometry. "
               "UE3 renders the foreground depth priority group (first-person arms, held weapon, muzzle flash) "
               "after the world DPG behind a mid-scene depth-only clear so foreground meshes never depth-clash "
               "with the world. Draws after that boundary receive the ViewModel category and override any "
               "player-model tag, so a weapon mesh shared between first- and third-person components can be "
               "tagged as Player Model Geometry to control the third-person copy while the first-person copy "
               "stays a view model. Only active in rtx.d3d9.ue3EngineMode; promotion to the ViewModel camera "
               "additionally requires rtx.viewModel.enable.");
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
    RTX_OPTION("rtx.d3d9", bool, ue3StaticLocalMeshVertexCaptureCache, false,
               "UE3 compat: for static draws captured through an exact position source, reuse the vertex shader "
               "output captured on an earlier frame rather than preserving a new vertex-capture draw. Only exact "
               "sources qualify: a clip-space reconstruction depends on where the camera was when it was taken, so "
               "reusing one across frames would freeze that frame's reconstruction error into the mesh. Cache keys "
               "cover the object transform and the shader's non-camera constants, so only genuinely static draws "
               "repeat a key; skinned draws and moving objects are refused outright or never admitted, and the "
               "cache is bounded by rtx.d3d9.ue3StaticLocalMeshVertexCaptureCacheBudgetMiB.");
    RTX_OPTION("rtx.d3d9", uint32_t, ue3StaticLocalMeshVertexCaptureCacheWarmupFrames, 2,
               "UE3 compat: number of distinct frames a static draw's cache key must be seen on before its captured "
               "vertex data is retained for reuse. Until then the key is tracked by a few bytes of CPU-side "
               "bookkeeping only, so a key that never repeats never retains a device-local capture buffer.");
    RTX_OPTION("rtx.d3d9", uint32_t, ue3StaticLocalMeshVertexCaptureCacheBudgetMiB, 256,
               "UE3 compat: upper bound in MiB on the device-local vertex-capture buffers retained by the static "
               "vertex-capture cache. Least recently used entries are evicted once the budget is exceeded, so an "
               "unexpectedly high key cardinality costs cache hit rate rather than VRAM. 0 disables the cache's "
               "retention tier entirely.");
    RTX_OPTION("rtx.d3d9", uint32_t, ue3StaticLocalMeshVertexCaptureCacheMaxEntries, 16384,
               "UE3 compat: upper bound on the number of retained static vertex-capture entries, as a backstop "
               "against many tiny captures exhausting the entry map before the byte budget is reached.");
    RTX_OPTION("rtx.d3d9", uint32_t, ue3StaticLocalMeshVertexCaptureCacheRetentionFrames, 600,
               "UE3 compat: number of frames a retained static vertex-capture entry survives without being reused "
               "before it is dropped. This is a staleness bound, not the memory bound - "
               "rtx.d3d9.ue3StaticLocalMeshVertexCaptureCacheBudgetMiB caps the bytes and evicts least recently used "
               "entries, which is the mechanism that actually keeps VRAM in check. Keep this generous: a mesh that "
               "leaves the view and comes back has to be recaptured once the entry expires, and at a high frame rate "
               "a short window expires entries during ordinary camera movement.");
    RTX_OPTION("rtx.d3d9", uint32_t, ue3StaticLocalMeshVertexCaptureCacheMinReusePercent, 10,
               "UE3 compat: percentage of eligible draws that must be served from the static vertex-capture cache for "
               "it to keep running. A title whose draws carry a camera-dependent vertex shader constant mints a fresh "
               "cache key every frame the camera moves, so the cache can never hit and its bookkeeping is pure "
               "overhead; below this rate it goes dormant, releasing its retained buffers and key records, and "
               "re-tests itself every rtx.d3d9.ue3StaticLocalMeshVertexCaptureCacheReuseProbeFrames frames in case a "
               "later scene is cacheable. 0 disables the guard and lets the cache run unconditionally.");
    RTX_OPTION("rtx.d3d9", uint32_t, ue3StaticLocalMeshVertexCaptureCacheReuseProbeFrames, 1800,
               "UE3 compat: number of frames the static vertex-capture cache stays dormant before briefly re-enabling "
               "itself to re-measure its reuse rate. Lower values notice a newly cacheable scene sooner; higher values "
               "spend less time re-measuring in a title where the cache can never hit.");
    RTX_OPTION("rtx.d3d9", bool, ue3ExcludePlacementFromVertexShaderHash, false,
               "UE3 compat: leave the object transform (LocalToWorld/WorldToLocal) and the shading-only constants "
               "(LightMapScale, lightmap/shadow coordinate scale-bias) out of the vertex-shader constant hash that "
               "feeds HashComponents::VertexShader, and so rules::FullGeometryHash.\n"
               "Enable it only for titles that recompute LocalToWorld every frame for geometry that is not moving. "
               "There, the transform moves that hash every frame, DrawCallCache::exactMatch never matches across "
               "frames, and a fresh BlasEntry is allocated for every draw of every frame; excluding the transform "
               "restores cross-frame matching. Use rtx.d3d9.ue3LogVertexConstantChurn to identify such a title: its "
               "level 2 reports the raw LocalToWorld registers differing on nearly every comparison.\n"
               "It is off by default because it is a trade-off, not a pure win. That same hash is also what separates "
               "one placement of a mesh from another, so excluding the transform collapses every instance of a mesh "
               "into a single BlasEntry that must then disambiguate them internally. A title with stable transforms "
               "and dense instancing is already in the good case and measurably loses throughput from the collapse - "
               "Mirror's Edge loses roughly a tenth of its frame rate in a heavy scene.\n"
               "Skinned meshes churn the hash through their bone registers either way, which stay included, so "
               "genuine vertex changes are never lost. Asset and replacement hashes are unaffected regardless, "
               "because rtx.geometryAssetHashRuleString excludes vertexshader.");
    RTX_OPTION("rtx.d3d9", bool, ue3LogVertexConstantChurn, false,
               "UE3 compat diagnostics: for draws that should be cacheable static geometry, report what actually "
               "changes between consecutive frames, as three separately attributed levels - the input-assembler "
               "identity, the set of instance transforms placed with the mesh, and any other vertex shader constant. "
               "Constant changes are named by their CTAB symbol and correlated against whether the view moved. Use "
               "this when the static vertex-capture cache reports a low reuse rate, to find which of the three is "
               "responsible; only the third is beyond the reach of "
               "rtx.d3d9.ue3ExcludePlacementFromVertexShaderHash.");
    RTX_OPTION("rtx.d3d9", uint32_t, ue3VertexConstantChurnMaxTrackedDraws, 256,
               "UE3 compat diagnostics: how many distinct input-assembler identities the constant-churn diagnostic "
               "samples. A sample is enough to characterise a title, and each tracked mesh retains a snapshot of the "
               "shader's used float constants for comparison against the next frame.");
    RTX_OPTION("rtx.d3d9", bool, ue3LogStaticVertexCaptureCacheStats, false,
               "UE3 compat diagnostics: log the static vertex-capture cache's retained entry count, retained bytes "
               "and per-frame reuse rate roughly once per second, plus a one-time warning when the byte budget or "
               "entry cap first forces an eviction. Use this to confirm the cache is hitting rather than just "
               "accumulating, and to size rtx.d3d9.ue3StaticLocalMeshVertexCaptureCacheBudgetMiB.");
    RTX_OPTION("rtx.d3d9", bool, ue3StaticGeometryHashMemoization, true,
               "UE3 CPU optimization: reuse geometry hash and bounding box results across frames for draws whose "
               "input-assembler vertex/index buffers are static (any vertex factory, including the bind-pose buffers "
               "of GPU-skinned meshes) instead of re-hashing the full vertex/index data every draw. Entries are keyed "
               "purely on the IA identity (buffer handles, offsets, draw parameters, per-buffer content generation "
               "counters), so one entry serves every instance of a mesh and results can never go stale; the per-draw "
               "vertex-shader-constants hash component is recombined live so served hashes are bit-identical to a "
               "fresh compute.");
    RTX_OPTION("rtx.d3d9", bool, ue3ExactVertexCapture, true,
               "UE3 compat: capture positions from the register the vertex shader multiplies by ViewProjectionMatrix "
               "rather than by unprojecting its clip-space output. Unprojecting divides by a quantity built from the "
               "difference of two numbers the size of view depth, so its error grows with the square of distance and "
               "makes distant meshes shear and swim as the camera moves; reading back the untransformed value the "
               "shader computed carries no such term. Requires the shader's oPos transform to be recognised and its "
               "matrix register to match the CTAB ViewProjectionMatrix symbol, which together prove the register "
               "holds a world position; draws failing either check fall back to unprojecting "
               "(see rtx.d3d9.ue3RequireExactVertexCapture).");
    RTX_OPTION("rtx.d3d9", bool, ue3RequireExactVertexCapture, false,
               "UE3 compat: drop draws from the ray-traced scene when neither exact position source applies, rather "
               "than falling back to clip-space reconstruction. This makes distance-dependent vertex distortion "
               "impossible rather than merely rare, at the cost of losing any geometry the exact paths do not cover. "
               "Enable rtx.d3d9.ue3LogCapturePrecision for a session first to enumerate which draws would be dropped.");
    RTX_OPTION("rtx.d3d9", Ue3CapturePositionSourceOverride, ue3VertexCaptureSourceOverride, Ue3CapturePositionSourceOverride::Auto,
               "UE3 compat diagnostics: force every vertex-capture draw onto one position source. 0 picks the most "
               "accurate applicable source (default), 1 forces the shader's pre-projection register, 2 forces "
               "input-assembler positions, 3 forces clip-space reconstruction. Toggling between 0 and 3 on a long "
               "outdoor view is the direct A/B for distance-dependent distortion. Forcing a source a draw does not "
               "qualify for falls back to reconstruction rather than producing wrong geometry.");
    RTX_OPTION("rtx.d3d9", bool, ue3NativeLocalMeshVertexCapture, true,
               "UE3 compat: allow input-assembler object-space positions to be used directly for conservative static "
               "LocalVertexFactory draws. This is the second-choice exact source, used for shaders whose oPos "
               "transform rtx.d3d9.ue3ExactVertexCapture could not recognise.");
    RTX_OPTION("rtx.d3d9", bool, ue3RequireCtabCameraConstants, false,
               "UE3 compat: only allow a draw call to update the Main camera when its vertex shader CTAB explicitly "
               "names both ViewProjectionMatrix and CameraPosition constants. Engine utility shaders (shadow depth, "
               "filters, etc.) do not declare these, so whatever data happens to live in the fallback camera registers "
               "(c0..c4) can otherwise be misinterpreted as a one-frame Main camera (e.g. a light-space matrix during "
               "UE3 light environment updates). Geometry from unverified draws is still rendered normally. "
               "Implicitly enabled by rtx.d3d9.ue3EngineMode.");
    RTX_OPTION("rtx.d3d9", bool, ue3StableDiffuseSelection, false,
               "Shader-path compat: cache the diffuse/albedo sampler selection per (pixel shader, bound texture set, "
               "sRGB states, vertex factory) so the same material always resolves to the same albedo texture. "
               "Without this, selection heuristics that read live shader constants (UE3 rewrites uniform expression "
               "registers per draw for panner/time/view-driven materials) can flip the chosen sampler between frames "
               "or with camera position, making a surface's albedo switch to an unrelated texture. "
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
    RTX_OPTION("rtx.d3d9", bool, ue3MicAutoExcludeFrameVaryingConstants, true,
               "UE3 MaterialInstanceConstant support: automatically detect materials whose UniformVector_*/"
               "UniformScalar_* constant registers are frame-varying (Time/panner/fade/sub-UV expressions) and "
               "exclude their constants from material identity hashing at runtime, as if they were listed in "
               "rtx.d3d9.ue3MicConstantIdentityExcludedShaders. Without this, such materials mint a new material "
               "hash every frame, churning instance identity (visible as temporal instability/flicker and per-frame "
               "BLAS rebuilds) until each is excluded manually. Detection and exclusion are scoped to a single "
               "(identity seed, texture set) group - one churning material family never widens to other materials "
               "sharing its shader or signature - and re-seen sibling hashes decay the churn count, so legitimate "
               "constant-differentiated sibling sets of any size do not trip it. Only active when material "
               "instance hashing is enabled (rtx.d3d9.ue3MaterialInstanceConstantHash or rtx.d3d9.ue3EngineMode).");
    RTX_OPTION("rtx.d3d9", bool, ue3LogClassification, false,
               "UE3 compat: log pass/vertex-factory classification decisions for draw routing. "
               "Also emits once-per-identity [UE3-Particle] lines (hashes, albedo, category bits, blend) "
               "for Particle / ParticleBeamTrail / LensFlare draws.");
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
               "shader-path draws: per-component scale/offset terms (immediate or constant-register component with "
               "factor, and inexactness), the live resolved scale/offset values, the gates that allowed or rejected "
               "writing the texture transform, and the pixel shader CTAB names/values of referenced constant "
               "registers. On the first sighting of a pixel shader it also dumps every sampler's UV origin and "
               "affine chain with the currently bound textures. Logs once per distinct resolved transform, capped "
               "per shader+stage. Use this to diagnose texture-atlas materials whose tile offset is not applied.");
    RTX_OPTION("rtx.d3d9", bool, ue3LogAlbedoSelection, false,
               "UE3 compat diagnostics: log a per-sampler score breakdown of the shader-path albedo selection once per "
               "(pixel shader, bound texture set, sRGB states, vertex factory) key: texture hash, dimensions, sRGB state, "
               "sample count, semantic/expression flags, final score, and the winning stages. Use this to diagnose draws "
               "where the wrong texture (e.g. a normal or specular map) is chosen as the ray-traced albedo.");
    RTX_OPTION("rtx.d3d9", bool, ue3LogCapturePrecision, false,
               "UE3 compat: log vertex capture precision diagnostics - the resolved position source per unique "
               "(vertex shader, vertex factory) with the reason any draw fell back to clip-space reconstruction, "
               "plus hash, cache and camera matrix details. The fallback lines are the list to work through before "
               "enabling rtx.d3d9.ue3RequireExactVertexCapture.");
    RTX_OPTION("rtx.d3d9", bool, ue3LogDrawStatusFlaps, false,
               "UE3 compat diagnostics: detect draws whose raytracing status (raytraced/rasterized/ignored) changes "
               "between nearby frames and log the transition with pass classification and shader hashes. A draw whose "
               "status flaps frame-to-frame manifests as geometry flickering in and out of the raytraced scene; this "
               "probe identifies which submission-side decision is responsible. Logs are capped per draw identity.");
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
      * \brief: Sends the pending drawcall geometry/state for raytracing, if nothing pending, does nothing.
      *
      * \param [in] drawContext : An object of type Draw that contains the context for the draw call.
      */
    void CommitGeometryToRT(const DrawContext& drawContext);

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
      * \brief: Signal a device Clear call. Used to detect the UE3 foreground DPG boundary
      * (mid-scene depth-only clear before first-person arms/weapon draws).
      */
    void OnClear(DWORD flags);

    /**
      * \brief: Called from texture upload/unlock paths with the sysmem source
      * of the data; stashes 16x1 float RGBA payloads (the game's baked tonemap
      * colour curve LUTs) for the Mirror's Edge tonemapping mode's live
      * capture. Partial-rect updates are merged; offsets/counts are in texels.
      */
    void onUe3CurveTextureUpload(const D3D9CommonTexture* dstTexture, D3D9CommonTexture* srcTexture, uint32_t srcSubresource,
                                 uint32_t srcTexelOffsetX, uint32_t dstTexelOffsetX,
                                 uint32_t texelWidth, uint32_t texelHeight);

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

  private: 
    // Reused fixed-size blocks: once allocated, a block is never reallocated, so background
    // skinning can keep raw Matrix4* into a prior copy. m_blocks can grow, but the heap
    // Block objects and their `m_matrices` array storage stay pinned. The write cursor
    // rewinds each frame; old blocks are retained to avoid per-frame allocs.
    struct SkinningMatrixPool {
      static constexpr size_t kMatricesPerBlock = 256 * 256;
      struct Block {
        std::array<Matrix4, kMatricesPerBlock> m_matrices;
      };
      std::vector<std::unique_ptr<Block>> m_blocks;
      size_t m_blockIndex = 0;          // which block the next write will use
      size_t m_nextIndexInBlock = 0;    // next free slot in m_blocks[m_blockIndex]

      void clear();
      const Matrix4* stageBones(const Matrix4* source, size_t matrixCount);
    } m_stagedBones;

    inline static const uint32_t kMaxConcurrentDraws = 6 * 1024; // some games issuing >3000 draw calls per frame...  account for some consumer thread lag with x2
    using GeometryProcessor = WorkerThreadPool<kMaxConcurrentDraws>;
    const std::unique_ptr<GeometryProcessor> m_pGeometryWorkers;
    AtomicQueue<DrawCallState, kMaxConcurrentDraws> m_drawCallStateQueue;

    DrawCallState m_activeDrawCallState;

    RtxStagingDataAlloc m_rtStagingData;
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
    bool m_forceGeometryCopy = false;
    bool m_forceIaTexcoordForOutlier = false;
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

    // UE3 SDPG_Foreground tracking: the foreground DPG (first-person arms/weapon) renders
    // after the world DPG behind a mid-scene depth-only clear. Both reset in EndFrame.
    bool m_ue3SeenMainViewWorldDraw = false;
    bool m_ue3ForegroundDpgActive = false;

    static Ue3VertexFactoryType classifyUe3VertexFactory(const D3D9VertexElements& elements);
    static bool isUe3WorldGeometryVertexFactory(Ue3VertexFactoryType type);

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
    static const char* describeGeometryStatus(RtxGeometryStatus status);
    void logUe3Classification(const DrawContext& drawContext,
                              Ue3PassType passType,
                              RtxGeometryStatus status,
                              const char* reason);
    bool trackUe3MovieTextureRenderTarget(const char* reason);
    bool isUe3MovieTextureDescHash(XXH64_hash_t descHash) const;

    // TdToneMapping capture: the game uploads its baked/blended colour curves
    // as two 16x1 float textures (ColorCurvesK/ColorCurvesM) each frame; the
    // texel payloads are snooped at upload/unlock time (keyed by destination
    // texture) and joined with the tonemap pass's pixel shader constants when
    // the fullscreen tonemap draw is classified.
    static constexpr uint32_t kUe3CurveTexelCount = 16;
    struct Ue3CurveTexels {
      std::array<Vector4, kUe3CurveTexelCount> texels = {};
    };
    std::unordered_map<const D3D9CommonTexture*, Ue3CurveTexels> m_ue3CurveTexelCache;
    bool m_ue3ToneMapCapturedThisFrame = false;

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

      // any CTAB name containing "lightmap" (LightMapScale, LightMapCoordinateScaleBias...):
      // the shader pair is recompiled per UE3 lightmap policy permutation. Vertex-lightmap
      // policies declare lightmap symbols only in the vertex shader, so this flag extends
      // lightmap-permutation-invariant material identity to their pixel shaders.
      bool hasLightmapSymbols = false;

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
      // Camera registers plus the object transform and shading-only constants.
      std::array<Range, kMaxRanges> withPlacement = {};
      uint32_t withPlacementCount = 0;
    };

    // keyed by XXH3 hash of the vertex shader DXSO bytecode
    fast_unordered_cache<Ue3VsShaderCtabInfo> m_ue3VsShaderCtabCache;
    std::optional<Ue3VsShaderCtabInfo> m_currentUe3CtabInfo;

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
    };
    fast_unordered_cache<PsSamplerTexcoordEntry> m_psSamplerTexcoordCache;
    fast_unordered_set m_loggedUvResolutions;

    // rtx.d3d9.ue3LogUvAffineDetail state: per-shader one-shot sampler dump, per distinct
    // resolved transform dedup, and a per (shader, stage) cap so panner/frame-varying
    // transforms cannot flood the log
    fast_unordered_set m_loggedUvAffineShaderDumps;
    fast_unordered_set m_loggedUvAffineDetails;
    fast_unordered_cache<uint16_t> m_uvAffineDetailLogCounts;

    // Deterministic diffuse selection: the winning sampler stages for a given
    // (pixel shader, ordered bound texture set, sRGB states, vertex factory) key.
    // Reusing the first decision keeps the albedo pick stable when scoring inputs
    // read live shader constants that UE3 rewrites per draw.
    // decisionAreaSum is the total bound texel area the decision was scored against:
    // streamed mip variants share this key (streaming-stable hashes), and a set bound
    // with more area re-scores and supersedes a decision made on streamed-down mips.
    struct Ue3DiffuseSelectionEntry {
      uint8_t chosenStages[2] = { 0xFF, 0xFF };
      uint8_t cubemapFallbackStage = 0xFF;
      uint64_t decisionAreaSum = 0;
      // Runtime only, not serialised: only entries read from the cache file are worth auditing.
      bool fromDisk = false;
    };
    fast_unordered_cache<Ue3DiffuseSelectionEntry> m_ue3DiffuseSelectionCache;
    // Persisted to rtx-remix/ue3DiffuseSelection.cache. The pin survives a level reload in memory
    // but not a relaunch, and a decision first made while a material's textures were still
    // streamed down can differ from the settled one, so the file is what makes every session
    // start from the same pick.
    bool m_ue3DiffuseSelectionLoaded = false;
    bool m_ue3DiffuseSelectionDirty = false;
    bool m_ue3DiffuseSelectionSaveBlocked = false;
    uint32_t m_ue3DiffuseSelectionLastSaveFrame = 0;
    // A stored pick records a decision, not the inputs behind it, and is consulted whenever the
    // bound texel area has not grown past the recorded peak - which after a settled run is
    // essentially always. Changed scoring would therefore be invisible wherever a cache exists,
    // so a bounded sample of loaded picks is re-scored and any disagreement reported once.
    uint32_t m_ue3DiffuseSelectionAuditsRemaining = 0;
    bool m_ue3DiffuseSelectionAuditWarned = false;
    void loadUe3DiffuseSelectionCache();
    void saveUe3DiffuseSelectionCache();
    // scoring reads the user-taggable lightmap/never-albedo/preferred-albedo texture sets; drop
    // cached decisions when those sets change so texture tagging in the UI takes effect live
    size_t m_ue3DiffuseSelectionLightmapSetSize = 0;
    size_t m_ue3DiffuseSelectionNeverAlbedoSetSize = 0;
    size_t m_ue3DiffuseSelectionPreferredAlbedoSetSize = 0;

    // selection cache keys already dumped by rtx.d3d9.ue3LogAlbedoSelection
    fast_unordered_set m_loggedAlbedoSelections;

    // draw identities already dumped as [UE3-Particle] by rtx.d3d9.ue3LogClassification
    fast_unordered_set m_loggedUe3ParticleDraws;

    // per-texture spread over distinct pixel shaders: textures sampled by many unrelated
    // materials are shared detail/pattern/tint overlays rather than surface identity albedo.
    // Persisted across sessions (rtx-remix/ue3TextureSpread.cache). `count` accumulates every
    // shader seen, but scoring reads only `scoringCount` - the value loaded from disk - so the
    // penalty is a fixed input for the whole session. Discoveries made now apply from the next
    // session, which is what keeps a material's albedo pick from changing meaning mid-run.
    struct Ue3TextureMaterialSpread {
      std::array<XXH64_hash_t, 12> psHashes = {};
      uint8_t count = 0;
      uint8_t scoringCount = 0;
    };
    fast_unordered_cache<Ue3TextureMaterialSpread> m_ue3TextureMaterialSpread;
    bool m_ue3TextureSpreadLoaded = false;
    bool m_ue3TextureSpreadDirty = false;
    // Set when the cache file existed but could not be read in full. Saving rewrites the
    // file from the map, so a partial load must never be allowed to publish itself.
    bool m_ue3TextureSpreadSaveBlocked = false;
    uint32_t m_ue3TextureSpreadLastSaveFrame = 0;
    void loadUe3TextureSpreadCache();
    void saveUe3TextureSpreadCache();

    // Diagnostic state for rtx.d3d9.ue3LogDrawStatusFlaps: per draw identity, the
    // prepare-flags outcome of the previous sighting, to detect frame-to-frame flapping
    struct Ue3DrawStatusEntry {
      uint32_t lastFlags = 0;
      uint32_t lastFrame = 0;
      const char* lastDecision = "";
      uint8_t lastPassType = 0;
      uint8_t logCount = 0;
    };
    fast_unordered_cache<Ue3DrawStatusEntry> m_ue3DrawStatusCache;
    uint32_t m_ue3FrameCounter = 0;
    const char* m_ue3LastDrawDecision = "";

    XXH64_hash_t mixUe3InstanceTransformConstants(XXH64_hash_t seed) const;
    void trackUe3DrawStatusFlap(const DrawContext& drawContext, PrepareDrawFlags flags);
    void logUe3UnboundAlbedoOnce(const D3D9CommonShader* pixelShader,
                                 XXH64_hash_t psHash,
                                 uint32_t usedSamplerMask,
                                 uint32_t usedTextureMask,
                                 const PsSamplerTexcoordEntry* inferredEntry);
    void logUe3ParticleDrawOnce();

    // UE3 albedo selection refuses render targets: soft-particle and scene-colour buffers carry a
    // content hash and near-backbuffer area, so at high resolutions they outscore the real material
    // texture. Movie surfaces and explicitly tagged targets stay eligible. Only meaningful under
    // rtx.d3d9.ue3EngineMode, which callers gate on.
    bool isUe3RenderTargetRefusedAsAlbedo(D3D9CommonTexture* texture,
                                          uint32_t stage,
                                          const PsSamplerTexcoordEntry* inferredEntry) const;

    struct Ue3VsTexcoordTraceEntry {
      bool initialized = false;
      Ue3VsUvTraceKind kind = Ue3VsUvTraceKind::Invalid;
      uint8_t iaTexcoordIndex = 0;
      uint8_t inputReg = 0;
      UvComponentAffine affineU;
      UvComponentAffine affineV;
    };
    fast_unordered_cache<Ue3VsTexcoordTraceEntry> m_ue3VsTexcoordTraceCache;
    fast_unordered_set m_autoRaytracedRenderTargetDescHashes;
    fast_unordered_set m_ue3MovieTextureDescHashes;

    // True if the image is tagged in rtx.raytracedRenderTargetTextures (and optionally
    // present in the auto-detected set).
    bool isTaggedRaytracedRenderTarget(const Rc<DxvkImage>& image, bool includeAutoDetected) const;

    // Authored render-target tags accept the aspect-normalized descriptor hash (registered by
    // the texture picker, stable across resolution changes) or the absolute one. Returns the
    // hash that matched, so diagnostics can name the identity the author actually tagged.
    static XXH64_hash_t matchAuthoredRenderTargetTag(const fast_unordered_set& tags,
                                                     XXH64_hash_t descriptorHash,
                                                     XXH64_hash_t resolutionAgnosticDescriptorHash);
    bool isBackBufferSizedImage(const Rc<DxvkImage>& image) const;

    // NOTE: to avoid calculating matrix inverse,
    //       m_seenCameraPositions doesn't contain the actual positions,
    //       but only relative values, see USE_TRUE_CAMERA_POSITION_FOR_COMPARISON
    std::vector<Vector3> m_seenCameraPositions;
    std::vector<Vector3> m_seenCameraPositionsPrev;

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

    // Retention tier of the static vertex-capture cache. An entry pins the whole
    // device-local capture buffer that its four views alias, so keys only reach this tier
    // once the admission tier below has proven they repeat across frames.
    struct Ue3VertexCaptureCacheEntry {
      RasterBuffer positionBuffer;
      RasterBuffer normalBuffer;
      RasterBuffer texcoordBuffer;
      RasterBuffer color0Buffer;
      uint32_t vertexCount = 0;
      uint32_t lastFrameTouched = 0;
      VkDeviceSize byteSize = 0;
    };

    fast_unordered_cache<Ue3VertexCaptureCacheEntry> m_ue3VertexCaptureCache;
    // Sum of byteSize over the retention tier, maintained incrementally so the budget can be
    // enforced without walking the map every frame.
    VkDeviceSize m_ue3VertexCaptureCacheBytes = 0;

    // Admission tier: a CPU-only sighting record that holds no GPU resources. A key has to
    // be seen on this many distinct frames before the tier above starts holding its capture
    // buffer, so a key that never repeats - an animating skinned pose, or any object whose
    // transform moves, both of which change the key every frame - costs a few bytes of
    // bookkeeping rather than a retained device-local buffer per draw per frame.
    struct Ue3VertexCaptureAdmissionEntry {
      uint32_t vertexCount = 0;
      uint32_t sightings = 0;
      uint32_t lastFrameSeen = 0;
    };

    fast_unordered_cache<Ue3VertexCaptureAdmissionEntry> m_ue3VertexCaptureAdmission;

    // Cache effectiveness counters. The per-frame pair is folded at end of frame into the
    // dormancy evaluation window (always) and the diagnostic interval (when logging is on).
    uint32_t m_ue3VertexCaptureCacheFrameReuses = 0;
    uint32_t m_ue3VertexCaptureCacheFrameCaptures = 0;
    uint64_t m_ue3VertexCaptureCacheStatReuses = 0;
    uint64_t m_ue3VertexCaptureCacheStatCaptures = 0;
    uint32_t m_ue3VertexCaptureCacheStatFrames = 0;
    uint32_t m_ue3VertexCaptureCacheStatFrameStamp = 0;
    uint64_t m_ue3VertexCaptureCacheEvictions = 0;

    // Dormancy guard. Some titles recompute a draw's object transform every frame even for
    // geometry that is not moving, which mints a fresh cache key per draw per frame. The cache
    // then cannot hit at all, and the admission tier alone accumulates a record per draw per
    // frame for no benefit. Measuring the reuse rate over a window and standing the whole cache
    // down when it is hopeless keeps the option safe to leave enabled in any UE3 title.
    bool m_ue3VertexCaptureCacheDormant = false;
    uint32_t m_ue3VertexCaptureCacheProbeCountdown = 0;
    uint32_t m_ue3VertexCaptureWindowFrames = 0;
    uint64_t m_ue3VertexCaptureWindowReuses = 0;
    uint64_t m_ue3VertexCaptureWindowCaptures = 0;

    // Constant-churn diagnostic (rtx.d3d9.ue3LogVertexConstantChurn). One entry per sampled
    // *mesh*, keyed on input-assembler identity, holding the multiset of instance transforms seen
    // for it each frame. Keying per mesh rather than per instance is what makes the three things
    // that can break a cache key separable, because a key that folds them together cannot say
    // which one moved:
    //   1. the IA identity itself (buffer handles, draw range, content generation counters)
    //   2. the set of instance transforms placed with that mesh
    //   3. any other shader constant folded into the stable VS hash
    // The transform registers are skipped in 3 precisely so it isolates the third cause; without
    // that, 2 and 3 would both fire on the same underlying change and neither would be diagnostic.
    struct Ue3ChurnMeshEntry {
      uint32_t lastFrameSeen = 0;
      // Frame the transform multiset below is accumulating for; rotated lazily on first touch of
      // a new frame so meshes that stop being drawn simply age out.
      uint32_t currentSetFrame = 0;
      // Extracted objectToWorld per placement, and the raw LocalToWorld/WorldToLocal register
      // block those were derived from. Tracking both is what separates a game that genuinely moves
      // its transforms from an objectToWorld extraction that is not reproducible: identical raw
      // registers with differing extracted matrices can only be the latter.
      std::vector<XXH64_hash_t> transformsThisFrame;
      std::vector<XXH64_hash_t> rawTransformsThisFrame;
      // Hashes and size of the last *completed* frame's multisets, so two complete sets are
      // compared rather than a complete set against a partially accumulated one.
      uint32_t completedSetFrame = 0;
      XXH64_hash_t completedSetHash = 0;
      XXH64_hash_t completedRawSetHash = 0;
      uint32_t completedSetSize = 0;
      XXH64_hash_t vsBytecodeHash = 0;
      Matrix4 worldToView;
      std::vector<Vector4> floatConsts;
      // Registers the level 3 diff skips: the camera registers because the stable VS hash already
      // omits them, and the object transform because that is level 2's question and including it
      // would make the result depend on which placement was drawn first each frame.
      uint32_t viewProjReg = 0;
      uint32_t viewProjRegCount = 0;
      uint32_t cameraPosReg = 0;
      uint32_t cameraPosRegCount = 0;
      uint32_t localToWorldReg = 0;
      uint32_t localToWorldRegCount = 0;
      uint32_t worldToLocalReg = 0;
      uint32_t worldToLocalRegCount = 0;
    };
    fast_unordered_cache<Ue3ChurnMeshEntry> m_ue3ConstantChurn;

    // Aggregates over the report window. Ordered map so the per-register report comes out in
    // register order. The shader hash is carried alongside the count purely so the report can
    // resolve the register's CTAB name without searching for a shader that declares it.
    struct Ue3ChurnRegisterTally {
      uint64_t count = 0;
      XXH64_hash_t vsBytecodeHash = 0;
    };
    std::map<uint32_t, Ue3ChurnRegisterTally> m_ue3ConstantChurnByRegister;
    // Level 1: did the mesh's IA identity come back on the next frame at all?
    uint64_t m_ue3ChurnMeshKeysCreated = 0;
    uint64_t m_ue3ChurnMeshRevisited = 0;
    // Level 2: for meshes that did come back, was the set of placements identical? Tracked for the
    // extracted matrices and for the raw registers, so the two can be cross-tabulated.
    uint64_t m_ue3ChurnTransformSetIdentical = 0;
    uint64_t m_ue3ChurnTransformSetDiffered = 0;
    uint64_t m_ue3ChurnTransformSetSizeChanged = 0;
    uint64_t m_ue3ChurnRawTransformSetIdentical = 0;
    // The diagnosis: extracted matrices moved while the registers they came from did not.
    uint64_t m_ue3ChurnExtractionUnstable = 0;
    // Level 3: did any constant move that is neither camera-derived nor part of the transform?
    uint64_t m_ue3ConstantChurnComparisons = 0;
    uint64_t m_ue3ChurnOtherConstantsChanged = 0;
    uint64_t m_ue3ConstantChurnWhileViewMoved = 0;
    uint64_t m_ue3ConstantChurnWhileViewStill = 0;
    uint32_t m_ue3ConstantChurnDetailDumps = 0;
    uint32_t m_ue3ConstantChurnReportFrameStamp = 0;

    void trackUe3ConstantChurn(XXH64_hash_t iaKey, const RasterGeometry& geoData);
    void reportUe3ConstantChurn();

    // Cross-frame memo of geometry hash + bounding box results for draws whose IA
    // vertex/index buffers are static (any vertex factory - a skinned mesh's bind-pose
    // buffers are as immutable as a static mesh's; only its bone constants animate).
    // Keyed purely on the IA identity (buffers, offsets, generations, draw range, decl,
    // texcoord selection), so one entry serves every instance of a mesh regardless of
    // transform. The entry holds the hash components computed from that identity; the
    // per-draw VertexShader component (stable VS-constant hash, position source) is
    // recombined live by the consumer, making served hashes bit-identical to a fresh
    // compute. Entries are heap-pinned via shared_ptr: a geometry worker publishes
    // results into the entry (release store on the ready flag) while the main thread
    // owns the map and serves published results on later frames (acquire load).
    struct Ue3GeometryMemoEntry {
      std::atomic<bool> hashesReady { false };
      std::atomic<bool> aabbReady { false };
      // per-component hashes; the VertexShader slot is intentionally left empty
      std::array<XXH64_hash_t, size_t(HashComponents::Count)> componentHashes = {};
      AxisAlignedBoundingBox boundingBox;
      uint32_t lastFrameTouched = 0;
    };
    fast_unordered_cache<std::shared_ptr<Ue3GeometryMemoEntry>> m_ue3GeometryMemoCache;
    void pruneUe3GeometryMemoCache();

    bool canMemoizeUe3IaGeometryHashes(const IndexContext& indexContext,
                                       const VertexContext vertexContext[caps::MaxStreams],
                                       const RasterGeometry& geoData) const;
    XXH64_hash_t computeUe3IaGeometryMemoKey(const IndexContext& indexContext,
                                             const VertexContext vertexContext[caps::MaxStreams],
                                             const DrawContext& drawContext,
                                             const RasterGeometry& geoData) const;
    // The geometry-hash VertexShader component for the current draw (stable VS-constant
    // hash plus position-source/outlier folds); shared by computeHash and the memo hit path.
    XXH64_hash_t computeLiveGeometryVertexShaderHashComponent();

    // Position source resolved once per draw, before the vertex-capture cache key is built
    // (the cache is only valid for exact sources) and before capture flags are uploaded.
    Ue3CapturePositionSource m_activeCapturePositionSource = Ue3CapturePositionSource::ClipReconstruction;

    Ue3CapturePositionSource resolveUe3CapturePositionSource(const IndexContext& indexContext,
                                                             const VertexContext vertexContext[caps::MaxStreams],
                                                             const RasterGeometry& geoData,
                                                             const char** outReason) const;
    static bool isUe3ExactCapturePositionSource(Ue3CapturePositionSource source) {
      return source != Ue3CapturePositionSource::ClipReconstruction;
    }
    static const char* describeUe3CapturePositionSource(Ue3CapturePositionSource source);
    void logUe3CapturePositionSource(Ue3CapturePositionSource source, const char* reason) const;

    bool canUseUe3StaticVertexCaptureCache(const IndexContext& indexContext,
                                           const VertexContext vertexContext[caps::MaxStreams],
                                           const RasterGeometry& geoData) const;
    bool isUe3StaticVertexCaptureCacheEligible(const IndexContext& indexContext,
                                               const VertexContext vertexContext[caps::MaxStreams],
                                               const RasterGeometry& geoData) const;
    bool canUseUe3PreProjectionVertexCapture(const char** outReason) const;
    bool canUseUe3NativeLocalVertexCapture(const IndexContext& indexContext,
                                           const VertexContext vertexContext[caps::MaxStreams],
                                           const RasterGeometry& geoData) const;
    XXH64_hash_t computeUe3StableVertexShaderHash(bool* outHashedFloatConstsWithExclusions = nullptr) const;

    // Per-draw memo of computeUe3StableVertexShaderHash (VS bytecode + camera-excluded
    // constants). The same value feeds both the geometry hash (computeHash) and the static
    // vertex-capture cache key, so it is computed once per draw in internalPrepareDraw
    // instead of hashing up to 4KB of shader constants twice.
    XXH64_hash_t m_activeStableVsHash = 0;
    XXH64_hash_t computeUe3StaticVertexCaptureCacheKey(const IndexContext& indexContext,
                                                       const VertexContext vertexContext[caps::MaxStreams],
                                                       const DrawContext& drawContext,
                                                       const RasterGeometry& geoData) const;
    bool tryReuseUe3StaticVertexCapture(XXH64_hash_t cacheKey, RasterGeometry& geoData);
    void updateUe3StaticVertexCaptureCache(XXH64_hash_t cacheKey, const RasterGeometry& geoData);
    void pruneUe3StaticVertexCaptureCache();
    void enforceUe3StaticVertexCaptureCacheBudget();
    void eraseUe3StaticVertexCaptureCacheEntry(XXH64_hash_t cacheKey);
    void clearUe3StaticVertexCaptureCache();
    // Called once per frame after pruning: folds the frame's reuse/capture counts into the
    // dormancy window and the diagnostic interval, then runs both.
    void updateUe3StaticVertexCaptureCacheState();
    void evaluateUe3StaticVertexCaptureCacheDormancy();
    void reportUe3StaticVertexCaptureCacheStats();

    static bool isPrimitiveSupported(const D3DPRIMITIVETYPE PrimitiveType) {
      return (PrimitiveType == D3DPT_TRIANGLELIST || PrimitiveType == D3DPT_TRIANGLEFAN || PrimitiveType == D3DPT_TRIANGLESTRIP);
    }

    const Direct3DState9& d3d9State() const;

    template<typename T>
    static void copyIndices(const uint32_t indexCount, T*& pIndicesDst, T* pIndices, uint32_t& minIndex, uint32_t& maxIndex);

    template<typename T>
    DxvkBufferSlice processIndexBuffer(const uint32_t indexCount, const uint32_t startIndex, const IndexContext& indexCtx, uint32_t& minIndex, uint32_t& maxIndex);

    bool prepareVertexCapture(const int vertexIndexOffset, Ue3CapturePositionSource positionSource);

    void processVertices(const VertexContext vertexContext[caps::MaxStreams], int vertexIndexOffset, RasterGeometry& geoData);

    bool processRenderState(const DrawContext& drawContext);

    template<bool FixedFunction>
    bool processTextures();

    PrepareDrawFlags internalPrepareDraw(const IndexContext& indexContext, const VertexContext vertexContext[caps::MaxStreams], const DrawContext& drawContext);

    void recordOcclusionQueryBracketedDraw(const VertexContext vertexContext[caps::MaxStreams],
                                           const DrawContext& drawContext);

    void flushOcclusionQueryDiagnostics();

    void triggerInjectRTX();

    // rtx.deferredUiTextures support: self-contained snapshots of overlay draws captured
    // mid-scene and replayed on top of the ray-traced image once RTX injection has fired.
    // The snapshot copies the referenced vertex/index ranges to CPU memory (immune to the
    // game re-locking its dynamic buffers between capture and replay) and is re-issued
    // through the regular D3D9 UP draw path.
    static constexpr std::array<D3DRENDERSTATETYPE, 25> kDeferredUiRenderStates = {
      D3DRS_ALPHABLENDENABLE, D3DRS_SRCBLEND, D3DRS_DESTBLEND, D3DRS_BLENDOP,
      D3DRS_SEPARATEALPHABLENDENABLE, D3DRS_SRCBLENDALPHA, D3DRS_DESTBLENDALPHA, D3DRS_BLENDOPALPHA,
      D3DRS_BLENDFACTOR,
      D3DRS_ALPHATESTENABLE, D3DRS_ALPHAREF, D3DRS_ALPHAFUNC,
      D3DRS_CULLMODE, D3DRS_FILLMODE, D3DRS_SHADEMODE,
      D3DRS_COLORWRITEENABLE,
      D3DRS_FOGENABLE,
      D3DRS_SRGBWRITEENABLE,
      D3DRS_SCISSORTESTENABLE,
      D3DRS_CLIPPING, D3DRS_CLIPPLANEENABLE,
      // captured for save/restore symmetry; forced off while replaying (overlays composite
      // over the final image, depth/stencil contents at replay time are meaningless)
      D3DRS_ZENABLE, D3DRS_ZWRITEENABLE, D3DRS_ZFUNC, D3DRS_STENCILENABLE,
    };

    static constexpr uint32_t kMaxDeferredUiDrawsPerFrame = 16;
    static constexpr uint32_t kMaxDeferredUiVertexBytes = 1024 * 1024;      // per draw, across all streams
    static constexpr uint32_t kMaxDeferredUiFrameVertexBytes = 8 * 1024 * 1024; // per frame, across all draws
    static constexpr uint32_t kMaxDeferredUiIndices = 256 * 1024;

    struct DeferredUiDraw {
      D3DPRIMITIVETYPE primitiveType = D3DPT_TRIANGLELIST;
      UINT primitiveCount = 0;
      bool indexed = false;
      uint32_t vertexCount = 0;

      // vertex data for the window [firstVertex, firstVertex + vertexCount), with all
      // referenced streams interleaved into a single stream-0 layout for UP replay
      uint32_t vertexStride = 0;
      std::vector<uint8_t> vertexData;

      // rebased onto the copied vertex window, widened to 32-bit
      std::vector<uint32_t> indexData;

      // the declaration to replay with: the original when it only references stream 0,
      // otherwise an internally created remap of every element onto the interleaved stream 0
      Com<IDirect3DVertexDeclaration9> replayDecl;
      Com<D3D9VertexShader, false> vertexShader;
      Com<D3D9PixelShader, false> pixelShader;

      std::vector<Vector4> vsFloatConsts;
      std::vector<Vector4> psFloatConsts;
      std::vector<Vector4i> vsIntConsts;
      std::vector<Vector4i> psIntConsts;
      std::vector<uint32_t> vsBoolConsts;
      std::vector<uint32_t> psBoolConsts;

      struct TextureBinding {
        uint32_t slot = 0;
        Com<IDirect3DBaseTexture9> texture;
        std::array<DWORD, SamplerStateCount> samplerStates = {};
        // non-null when the bound texture is a render target (scene color candidate for
        // the rtx.d3d9.deferredUiRefreshSceneColor blit)
        Rc<DxvkImage> renderTargetImage;
      };
      std::vector<TextureBinding> textures;

      std::array<DWORD, kDeferredUiRenderStates.size()> renderStates = {};
      D3DVIEWPORT9 viewport = {};
      RECT scissorRect = {};
      uint32_t sourceRenderTargetWidth = 0;
      uint32_t sourceRenderTargetHeight = 0;
    };

    std::vector<DeferredUiDraw> m_deferredUiDraws;
    uint32_t m_deferredUiFrameVertexBytes = 0;
    bool m_replayingDeferredUiDraws = false;

    // one-shot log keys (pixel shader hash mixed with the defer/refuse decision) for the
    // [RTX-DeferredUI] tag diagnostics
    fast_unordered_set m_deferredUiLoggedDecisions;

    // Checks whether the draw matches rtx.deferredUiTextures (image hash for non-RTs, or
    // resolution-agnostic RT descriptor hash) or rtx.d3d9.deferredUiPixelShaders.
    bool isDeferredUiTaggedDraw(XXH64_hash_t* pMatchedTextureHash = nullptr) const;

    bool captureDeferredUiDraw(const IndexContext& indexContext,
                               const VertexContext vertexContext[caps::MaxStreams],
                               const DrawContext& drawContext);
    // pOverrideRenderTarget: bind this surface as RT0 for the replay (EndFrame fallback path,
    // where the app's current RT0 is unrelated); nullptr replays onto the currently bound RT0
    // (mid-frame injection path). injectionTargetImage: the image the ray-traced result was
    // blitted to, used as the source for the scene-color refresh blit (may be null to skip).
    void replayDeferredUiDraws(IDirect3DSurface9* pOverrideRenderTarget,
                               const Rc<DxvkImage>& injectionTargetImage);
    Rc<DxvkImage> getCurrentRenderTargetImage() const;

    struct DrawCallType {
      RtxGeometryStatus status;
      bool triggerRtxInjection;
      // rtx.deferredUiTextures: rasterize on top of the ray-traced image without triggering
      // injection - the draw is captured and replayed after injection fires later in the frame
      bool deferUntilInjection = false;
    };
    DrawCallType makeDrawCallType(const DrawContext& drawContext);

    bool checkBoundTextureCategory(const fast_unordered_set& textureCategory) const;

    // Per-draw snapshot of the bound texture slots (common texture pointer, cached image
    // hash, render-target descriptor hash), lazily built and shared by the per-draw
    // consumers that would otherwise each re-walk the texture stages: UI/deferred-UI tag
    // checks, the MIC texture-set hash, the diffuse-selection cache key and the
    // two-sided-translucency dedup.
    // Invalidated at the top of internalPrepareDraw; bindings cannot change within a draw.
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

    // Per-frame snapshot of the scalar options read on the per-draw hot path. Every
    // RtxOption read acquires the global option update mutex; the per-draw pipeline
    // (makeDrawCallType, classifyUe3Pass, processRenderState, processTextures, the
    // geometry identity keys) reads dozens of options per draw, which at UE3 draw
    // counts (~2400/frame) is >100k mutex acquisitions per frame. Option values only
    // resolve once per frame anyway, so a per-frame value snapshot is exactly as fresh
    // as the underlying resolution model. Refreshed in EndFrame (the same cadence as
    // DrawCallState::refreshCategoryLookupTable) and lazily on the first frame's draw.
    // Set-typed options are intentionally not snapshotted: their accessors return
    // references to stable storage and are read far less often per draw.
    // Field names mirror the option accessors they cache.
    struct FrameOptionCache {
      bool valid = false;

      // D3D9Rtx options
      bool orthographicIsUI = false;
      bool preTransformedVerticesIsUI = false;
      bool allowCubemaps = false;
      bool useVertexCapture = false;
      bool useVertexCapturedNormals = false;
      bool useWorldMatricesForShaders = false;
      bool ue3EngineMode = false;
      bool ue3CameraFromShaderConstants = false;
      bool ue3ObjectToWorldFromShaderConstants = false;
      bool autoRaytracedRenderTargetFromFullscreenComposite = false;
      bool rasterizeFullscreenCompositeToPrimary = false;
      bool shaderPathTexcoordIndexFromPixelShader = false;
      bool ue3MaterialInstanceConstantHash = false;
      bool ue3LightmapPermutationInvariantHash = false;
      bool ue3LightmapPermutationBridgeLookup = false;
      bool ue3MicExcludeRenderTargetsFromIdentity = true;
      bool ue3MicPersistAutoExcludedConstantGroups = true;
      bool ue3LogMaterialInstanceHash = false;
      bool ue3SkipDepthPrepass = false;
      bool ue3SkipShadowDepthPasses = false;
      bool ue3SkipDepthTestDisabledTranslucency = false;
      bool ue3SkipSceneCapturePasses = false;
      bool ue3ForegroundDpgIsViewModel = false;
      bool conservativeOcclusionQueries = false;
      bool ue3StaticLocalMeshVertexCaptureCache = false;
      uint32_t ue3StaticLocalMeshVertexCaptureCacheWarmupFrames = 0;
      uint32_t ue3StaticLocalMeshVertexCaptureCacheBudgetMiB = 0;
      uint32_t ue3StaticLocalMeshVertexCaptureCacheMaxEntries = 0;
      uint32_t ue3StaticLocalMeshVertexCaptureCacheRetentionFrames = 0;
      uint32_t ue3StaticLocalMeshVertexCaptureCacheMinReusePercent = 0;
      uint32_t ue3StaticLocalMeshVertexCaptureCacheReuseProbeFrames = 0;
      bool ue3LogStaticVertexCaptureCacheStats = false;
      bool ue3ExcludePlacementFromVertexShaderHash = false;
      bool ue3LogVertexConstantChurn = false;
      uint32_t ue3VertexConstantChurnMaxTrackedDraws = 0;
      bool ue3StaticGeometryHashMemoization = false;
      bool ue3ExactVertexCapture = false;
      bool ue3RequireExactVertexCapture = false;
      Ue3CapturePositionSourceOverride ue3VertexCaptureSourceOverride = Ue3CapturePositionSourceOverride::Auto;
      bool ue3NativeLocalMeshVertexCapture = false;
      bool ue3RequireCtabCameraConstants = false;
      bool ue3StableDiffuseSelection = false;
      bool ue3MicAutoExcludeFrameVaryingConstants = false;
      bool ue3LogClassification = false;
      bool ue3LogUvResolution = false;
      bool ue3LogUvAffineDetail = false;
      bool ue3LogAlbedoSelection = false;
      bool ue3LogCapturePrecision = false;
      bool ue3LogDrawStatusFlaps = false;
      bool ue3LogOcclusionQueries = false;
      bool deferredUiReplay = false;
      bool deferredUiRefreshSceneColor = false;
      bool enableIndexBufferMemoization = false;

      // upstream RtxOptions (raytracedRenderTargetEnable caches
      // RtxOptions::RaytracedRenderTarget::enable, needsMeshBoundingBox the
      // derived RtxOptions::needsMeshBoundingBox result)
      bool enableRaytracing = false;
      bool enableAlphaTest = false;
      bool enableAlphaBlend = false;
      bool raytracedRenderTargetEnable = false;
      bool skipDrawCallsPostRTXInjection = false;
      bool useBuffersDirectly = false;
      bool fogIgnoreSky = false;
      bool needsMeshBoundingBox = false;
      bool validateCPUIndexData = false;
      bool alwaysCopyDecalGeometries = false;
      bool terrainAsDecalsEnabledIfNoBaker = false;
      bool terrainAsDecalsAllowOverModulate = false;
      bool enableMultiStageTextureFactorBlending = false;
      bool ignoreAllVertexColorBakedLighting = false;
      bool vertexColorIsBakedLighting = false;
      bool logReplacementResolution = false;
      Vector2i drawCallRange = Vector2i(0, 0);

      // Set-typed options, cached as pointers: each option's resolved hash set is
      // allocated once at construction and only mutated in place during option
      // resolution, so a per-frame pointer is exactly as safe as the per-call
      // reference the locked accessor hands out - both are read outside the option
      // mutex between resolution points.
      const fast_unordered_set* uiTextures = nullptr;
      const fast_unordered_set* deferredUiTextures = nullptr;
      const fast_unordered_set* deferredUiPixelShaders = nullptr;
      const fast_unordered_set* lightmapTextures = nullptr;
      const fast_unordered_set* neverAlbedoTextures = nullptr;
      const fast_unordered_set* preferredAlbedoTextures = nullptr;
      const fast_unordered_set* smoothNormalsTextures = nullptr;
      const fast_unordered_set* ignoreBakedLightingTextures = nullptr;
      const fast_unordered_set* raytracedRenderTargetTextures = nullptr;
      const fast_unordered_set* vsTexcoordCaptureOutlierTextures = nullptr;
      const fast_unordered_set* ue3MicConstantIdentityExcludedShaders = nullptr;
      const fast_unordered_set* ue3MicIdentityExcludedTextureDescHashes = nullptr;
      const fast_unordered_set* replacementDebugHashes = nullptr;
    };
    FrameOptionCache m_frameOptions;
    void refreshFrameOptionCache();

    // Material hashes tracked this frame for SceneManager::trackReplacementMaterialHash,
    // flushed as one CS command in EndFrame instead of one EmitCs per draw. The only
    // consumers (graph components via getReplacementMaterialHashUsageCount) read the
    // per-frame map during SceneManager::onFrameEnd, which executes after the flush on
    // the CS timeline, so batching is invisible to them.
    std::vector<XXH64_hash_t> m_pendingReplacementMaterialHashes;

    bool isRenderingUI();

    Future<SkinningData> processSkinning(const RasterGeometry& geoData);

    // When publishTo is non-null, the worker additionally publishes the computed result
    // into the memo entry so later frames can reuse it without recomputing.
    Future<AxisAlignedBoundingBox> computeAxisAlignedBoundingBox(const RasterGeometry& geoData,
                                                                 const std::shared_ptr<Ue3GeometryMemoEntry>& publishTo = {});

    Future<GeometryHashes> computeHash(const RasterGeometry& geoData, const uint32_t maxIndexValue,
                                       const std::shared_ptr<Ue3GeometryMemoEntry>& publishTo = {});

    void submitActiveDrawCallState();
  };
}
