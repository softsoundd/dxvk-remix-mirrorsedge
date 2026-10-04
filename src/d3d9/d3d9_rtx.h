#pragma once

#include "d3d9_state.h"
#include "d3d9_rtx_ue3.h"
#include "../dxvk/dxvk_buffer.h"
#include "../util/util_threadpool.h"

#include <array>
#include <atomic>
#include <chrono>
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
    RTX_OPTION("rtx.d3d9", bool, autoRaytracedRenderTargetFromFullscreenComposite, false,
               "D3D9 compat: auto-detect an offscreen render target being used as the main scene (sampled in a fullscreen composite pass) "
               "and treat it as a raytraced render target to capture the correct geometry in games that upscale/composite to the backbuffer.");
    RTX_OPTION("rtx.d3d9", bool, rasterizeFullscreenCompositeToPrimary, false,
               "D3D9 compat: rasterise likely fullscreen composite/postprocess passes to the primary render target. Helps avoid raytracing a fullscreen quad.");
    RTX_OPTION("rtx.d3d9", bool, ue3AutoDetectLightmapTextures, true,
               "UE3 compat: treat every texture bound to a lightmap sampler as if it had been listed in "
               "rtx.lightmapTextures. UE3 declares its baked lighting under fixed CTAB sampler names "
               "(LightMapTextures, and Mirror's Edge's BSplineTexture filtering LUT), so the runtime can "
               "recognise them without the per-level hashes ever being tagged by hand. Discovered hashes are "
               "held for the session only and are never written to a config. Remix supplies the lighting, so "
               "these textures are excluded from albedo selection, material identity and the texture picker's "
               "taggable set. Requires rtx.d3d9.ue3EngineMode.");
    RTX_OPTION("rtx.d3d9", float, ue3ConstantAlbedoTintGain, 8.0f,
               "UE3 compat: how strongly a textureless material's UniformVector_* colour pulls its albedo away "
               "from rtx.legacyMaterial.albedoConstant. UE3 keeps such a material's tint in a colour register and "
               "relies on the baked lightmap for brightness, so the raw value is far too dark to use as an albedo "
               "once Remix has removed the lightmap and relit the surface. The register's brightest channel times "
               "this gain, clamped to 1, is the weight blending from the legacy albedo constant to the register's "
               "fully saturated hue, so a register at zero leaves the surface at the legacy constant and a tint "
               "ramping up fades smoothly to its colour rather than stepping to it. The default reaches full "
               "saturation at 0.125, the peak Mirror's Edge's menu highlight reaches. Set to 0 to use the register "
               "value directly instead, which is faithful to the constant but renders these surfaces very dark.");
    RTX_OPTION("rtx.d3d9", bool, ue3HighlightTints, true,
               "UE3 compat: reproduce the tint a game fades in through a material strength parameter, such as Mirror's "
               "Edge's Runner Vision (LOI_Strength). Remix samples the albedo texture itself, so a tint the pixel shader "
               "applies from its constants never reaches the path tracer. The pixel shader is analysed to prove which "
               "UniformScalar_* tints the colour output as lerp(X, X * V, S), and towards which UniformVector_*, or "
               "only adds an unlit glow of a texture to one channel, as enemy weapons flash red. The "
               "tint is carried per surface, so it follows the fade frame by frame, leaves material hashes alone and "
               "applies over replacement materials too. rtx.d3d9.ue3HighlightTintRequireMotion decides which proven "
               "tints count as a highlight. Requires rtx.d3d9.ue3EngineMode; see documentation/UE3Compatibility.md, "
               "\"Runner Vision\".");
    RTX_OPTION("rtx.d3d9", bool, ue3HighlightTintRequireMotion, true,
               "UE3 compat: apply a proven tint, and its glow, only once its strength has been seen moving. Runner "
               "Vision fades LOI_Strength on the material instances it creates, while an object authored in the tint "
               "colour holds its strength still, and Remix leaves that colour to its material. Tracked per object - "
               "material instance and placement - so an object painted in the highlight colour stays untinted when an "
               "identical one is highlighted; an object at a placement not seen before, as a moving one is every "
               "frame, follows its material instance. Disable to apply every proven tint and its glow, authored "
               "colours included.");
    RTX_OPTION("rtx.d3d9", float, ue3HighlightGlowIntensity, 1.0f,
               "UE3 compat: scale on the glow Runner Vision adds to a highlighted surface. Most Runner Vision materials "
               "add an unlit copy of the tinted colour, weighted by the strength and a coefficient read from the shader "
               "(0.1 on Mirror's Edge), which keeps highlighted objects readable in shadow; enemy weapons instead glow "
               "one of their textures red before a strike, at several times the strength. It is reproduced as "
               "emission, per channel, of that fraction of the surface's tinted albedo, or of the glowing texture, "
               "which the surface's material carries as its emissive texture (a replacement material only when it "
               "authors no emission or emissive texture); 1 keeps the game's coefficients, and 0 disables the glow "
               "while keeping the tint. Also scaled by rtx.emissiveIntensity.");
    RTX_OPTION("rtx.d3d9", fast_unordered_set, ue3HighlightTintExcludedMaterials, {},
               "UE3 compat: surfaces rtx.d3d9.ue3HighlightTints never tints, matched against the draw's material hash, "
               "its textureSet+shader hash and its primary colour texture hash. For a tint the game uses for something "
               "other than a highlight that rtx.d3d9.ue3HighlightTintRequireMotion still lets through; "
               "rtx.d3d9.ue3LogHighlightTints reports all three hashes the first time a tint applies to a material.");
    RTX_OPTION("rtx.d3d9", bool, ue3LogHighlightTints, false,
               "UE3 compat diagnostics: log what rtx.d3d9.ue3HighlightTints finds. Once per pixel shader: the proven "
               "strength and colour pairs with their CTAB names and glow coefficients, the material scalars that reach "
               "the colour output without proving a tint, and why a shader could not be analysed. Once per material: "
               "the first time a tint applies, with each pair's live strength and colour, the resulting tint and glow, "
               "and the hashes rtx.d3d9.ue3HighlightTintExcludedMaterials takes; and when a tint is held back because "
               "its strength has not moved.");
    RTX_OPTION("rtx.d3d9", Vector3, ue3HighlightDebugForceTint, Vector3(1.f, 1.f, 1.f),
               "UE3 compat diagnostic: force this tint onto every UE3 material, whatever the analysis found, to check "
               "the per-surface tint reaches the shader independently of detection. Identity (1, 1, 1) disables the "
               "override.");
    RTX_OPTION("rtx.d3d9", bool, ue3ParticleVertexColor, true,
               "UE3 compat: reproduce the per-particle colour and alpha of UE3 sprite, SubUV and beam/trail particles, such "
               "as a ParticleModuleColorOverLife fade. UE3 hands the particle colour to the pixel shader as a TEXCOORD "
               "interpolant (TEXCOORD3 on SubUV sprites, TEXCOORD1 otherwise), which Remix never treats as a vertex colour. "
               "The pixel shader is analysed for how the colour reaches its output - a per-channel tint, an opacity scale, "
               "or the colour scale of UE3's additive blend mode - and those uses are reproduced through the captured "
               "vertex colour and the texture stage operations, on the draws whose constants allow them. Requires "
               "rtx.d3d9.ue3EngineMode; see documentation/UE3Compatibility.md, \"Opacity-driven fades\".");
    RTX_OPTION("rtx.d3d9", bool, ue3MaterialFades, true,
               "UE3 compat: fade a draw in and out with the material parameter its pixel shader fades it by, such as a "
               "modulate decal lerping from white to its texture, or a translucent opacity multiplied by a parameter. The "
               "pixel shader is analysed for the UniformScalar_* and UniformVector_* components its output is affine in; "
               "each draw checks with its own constants whether one brings the whole output to the value its blend leaves "
               "the framebuffer unchanged by, and scales the surface's opacity and emission by how far from there it is. "
               "Carried per surface, so it follows the parameter frame by frame, leaves material hashes alone and applies "
               "over blended replacement materials. Requires rtx.d3d9.ue3EngineMode; see documentation/UE3Compatibility.md, "
               "\"Opacity-driven fades\".");
    RTX_OPTION("rtx.d3d9", fast_unordered_set, ue3MaterialFadeExcludedMaterials, {},
               "UE3 compat: surfaces rtx.d3d9.ue3MaterialFades never fades, matched against the draw's material hash, its "
               "textureSet+shader hash and its primary colour texture hash, all of which rtx.d3d9.ue3LogMaterialFades "
               "reports.");
    RTX_OPTION("rtx.d3d9", bool, ue3LogMaterialFades, false,
               "UE3 compat diagnostics: log what rtx.d3d9.ue3MaterialFades and rtx.d3d9.ue3ParticleVertexColor find. Once "
               "per pixel shader: the fade candidates, the particle colour uses, the material scalars the output is not "
               "affine in, and why a shader could not be analysed. Once per colour texture, shader and outcome: the blend, "
               "the particle colour uses applied, each candidate's live value and coverage, and the hashes "
               "rtx.d3d9.ue3MaterialFadeExcludedMaterials takes.");
    RTX_OPTION("rtx.d3d9", float, ue3MaterialFadeDebugForceCoverage, -1.0f,
               "UE3 compat diagnostic: force this coverage onto every blended UE3 surface, whatever the analysis found, to "
               "check the per-surface fade reaches the shader independently of detection. Negative disables the override.");
    RTX_OPTION("rtx.d3d9", bool, ue3MicConstantIdentity, true,
               "UE3 MaterialInstanceConstant support: fold the material's Uniform* constants into its "
               "identity, distinguishing instances that share a parent and its textures but differ in "
               "VectorParameterValues/ScalarParameterValues. Mirror's Edge relies on this - its colour "
               "variants (RooftopPropsClusters and its Blue/Orange/Yellow siblings, and many others) are "
               "one texture set with a different tint parameter, and without this tier they collapse onto "
               "a single anchor.\n"
               "Only UniformVector_* parameters contribute. The scalars are where the two lightmap "
               "compiles genuinely disagree - DiffusePower exponents the lightmap under SIMPLE_LIGHTING "
               "and a LightMapBasis-derived transfer coefficient otherwise, and SpecularPower is its "
               "structural twin present in only one compile - so the whole scalar class is left out "
               "rather than trying to tell those two apart, which the bytecode does not allow. Material "
               "instances separated only by a scalar parameter therefore share an anchor.\n"
               "Turn it off for content whose material instances are told apart by their textures alone; "
               "identity then becomes shader + material texture set, which is fully independent of the "
               "lightmap policy. Either way, changing this re-mints the material hashes of every material "
               "carrying constants, so anchors keyed on the old ones stop matching.");
    RTX_OPTION("rtx.d3d9", bool, ue3MicVolatileConstantDetection, true,
               "UE3 MaterialInstanceConstant support: leave frame-varying UniformVector_* registers (panners, "
               "rotators, flipbook frames, time-driven fades) out of material identity. A register is volatile when "
               "the shader's dataflow shows a sampler's coordinate depending on it; tints reaching the output colour "
               "are kept. Identity is decided before the first draw and never changes. Turning this off re-mints the "
               "hashes of every material carrying a volatile register. See \"Material identity and replacement anchor "
               "stability\" in documentation/UE3Compatibility.md.");
    RTX_OPTION("rtx.d3d9", fast_unordered_set, ue3MicConstantIdentityExcludedShaders, {},
               "UE3 MaterialInstanceConstant support: pixel shader hashes whose UniformVector_* constants are "
               "excluded from material identity hashing wholesale. Reach for this only when every material on "
               "a shader is frame-varying in a way rtx.d3d9.ue3MicVolatileConstantDetection cannot see: one "
               "UE3 base-pass shader commonly serves dozens of material families, and listing it drops the "
               "constants tier for all of them, merging instances that differ only by a colour parameter. "
               "Under rtx.d3d9.ue3EngineMode both the canonical shader identity and the raw bytecode hash are "
               "honoured. Prefer rtx.d3d9.ue3MicConstantIdentityExcludedMaterials, which is scoped to a single "
               "material family.");
    RTX_OPTION("rtx.d3d9", fast_unordered_set, ue3MicConstantIdentityExcludedMaterials, {},
               "UE3 MaterialInstanceConstant support: textureSet+shader hashes whose constants are excluded "
               "from material identity hashing. That hash names one material family - a shader together with "
               "the exact set of images its material samplers bind - so this excludes precisely the family that "
               "churns and leaves every other material on the same shader with its constants tier intact. It is "
               "also the second replacement lookup tier, so the same value can anchor the family's override.\n"
               "List a hash here when [RTX-MicChurn] reports a family still minting more than one identity, "
               "which happens when a frame-varying register reaches the output colour rather than a UV "
               "coordinate and so cannot be told from an authored tint. The warning prints the value ready to "
               "paste; it is also reported as 'textureSetShader=0x...' by rtx.d3d9.ue3LogMaterialInstanceHash.");
    RTX_OPTION("rtx.d3d9", bool, ue3LogMaterialInstanceHash, false,
               "UE3 MaterialInstanceConstant support: log a one-shot per-material breakdown of the material "
               "identity hash (pixel shader hash, material texture set with per-sampler image hashes, which "
               "UniformVector_* registers were kept versus dropped as volatile, the constants hash, and the "
               "final hash).");
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
               "its identity is stable from then on. The sampler is identified by that descriptor hash rather "
               "than dropped from the texture set: a texture whose contents are not reproducible still has a "
               "reproducible shape, and dropping it would achieve nothing on a material whose only sampler it "
               "is, since an empty texture set falls back to the primary colour texture's image hash - the "
               "value being excluded.\n"
               "Note descriptor hashes derive from texture properties (dimensions/format/usage), so "
               "identically-shaped textures share one and the exclusion applies to all of them; materials "
               "distinguished only by which of those they bind will merge. That is usually acceptable for the "
               "engine-composited textures this option targets. UE3-streamed textures recreate at a different "
               "size per resident mip level and so carry one descriptor hash per size; the composited/dynamic "
               "textures this option is meant for are fixed-size.");
    RTX_OPTION("rtx.d3d9", fast_unordered_set, vsTexcoordCaptureOutlierTextures, {},
               "Texture hashes for which VS-captured texcoords should be overridden with IA (input assembler) texcoords. "
               "Useful as a compatibility fallback when certain textures appear stretched due to incorrect VS texcoord capture.");
    RTX_OPTION("rtx.d3d9", bool, ue3AutoCullEnclosingMeshShadowBackfaces, false,
               "UE3 compat: on shadow/NEE visibility rays, ignore inward backfaces of one-sided opaque meshes whose object AABB contains the camera.\n"
               "For wrapping building shells around BSP interiors. Gated by rtx.d3d9.ue3AutoCullEnclosingMeshMinExtentMeters / MaxExtentMeters. Requires rtx.d3d9.ue3EngineMode. Off by default; geometry tagging is more precise.");
    RTX_OPTION("rtx.d3d9", float, ue3AutoCullEnclosingMeshMinExtentMeters, 2.0f,
               "Minimum world-space mesh extent (meters) for rtx.d3d9.ue3AutoCullEnclosingMeshShadowBackfaces. "
               "Keeps thin floor slabs and billboard cards from being treated as enclosing shells.");
    RTX_OPTION("rtx.d3d9", float, ue3AutoCullEnclosingMeshMaxExtentMeters, 150.0f,
               "Maximum world-space mesh extent (meters) for rtx.d3d9.ue3AutoCullEnclosingMeshShadowBackfaces. "
               "Keeps whole-level BSP models from being treated as enclosing shells.");
    RTX_OPTION("rtx.d3d9", bool, ue3ForegroundDpgIsViewModel, true,
               "UE3 compat: classify SDPG_Foreground draws as view-model (first-person overlay) geometry. "
               "UE3 renders the foreground depth priority group (first-person arms, held weapon, muzzle flash) "
               "after the world DPG behind a mid-scene depth-only clear so foreground meshes never depth-clash "
               "with the world. Draws after that boundary receive the ViewModel category and override any "
               "player-model tag, so a weapon mesh shared between first- and third-person components can be "
               "tagged as Player Model Geometry to control the third-person copy while the first-person copy "
               "stays a view model. Only active in rtx.d3d9.ue3EngineMode; promotion to the ViewModel camera "
               "additionally requires rtx.viewModel.enable.");
    RTX_OPTION("rtx.d3d9", bool, ue3DisableFrustumCulling, false,
               "Mirror's Edge: patch the game so its renderer skips the per-primitive view frustum test and draws "
               "every primitive within its cull distance, keeping geometry outside the camera's view in the ray "
               "traced scene for shadows, reflections and indirect light. Costs CPU time in both the game and Remix; "
               "rtx.d3d9.ue3FrustumBypassMaxDistanceMeters limits it. "
               "Applied inside the game process by the bridge client, only while rtx.d3d9.ue3EngineMode and ray "
               "tracing are enabled; see documentation/UE3Compatibility.md, \"Game executable patches\".");
    RTX_OPTION_ARGS("rtx.d3d9", float, ue3FrustumBypassMaxDistanceMeters, 0.f,
               "Mirror's Edge: with rtx.d3d9.ue3DisableFrustumCulling, keep primitives outside the camera's view "
               "only within this distance of the camera, or 0 to keep all of them. Measured like UE3's cull "
               "distances, which scale with the field of view. Saves CPU time in the game and Remix at the cost of "
               "far off-screen geometry, which rtx.antiCulling.object.enable can retain once it has been seen.",
               args.minValue = 0.f);
    RTX_OPTION_ARGS("rtx.d3d9", float, ue3FrustumBypassMinRadiusMeters, 0.f,
               "Mirror's Edge: with rtx.d3d9.ue3FrustumBypassMaxDistanceMeters set, also keep primitives outside "
               "the camera's view whose bounding sphere has at least this radius at any distance, such as the "
               "buildings that fill distant reflections, or 0 to keep none by size.",
               args.minValue = 0.f);
    RTX_OPTION("rtx.d3d9", bool, ue3ShowThirdPersonModel, false,
               "Mirror's Edge: patch the game so the third-person body and weapon meshes (Mesh3p) also draw in the "
               "first-person view. The game otherwise hides them from the player's own camera and keeps them only "
               "for its shadows and reflections; drawn, they become Remix player-model geometry (rtx.playerModel*). "
               "Applied inside the game process by the bridge client, only while rtx.d3d9.ue3EngineMode and ray "
               "tracing are enabled; see documentation/UE3Compatibility.md, \"Game executable patches\".");
    RTX_OPTION("rtx.d3d9", bool, ue3DisableOcclusionQueries, false,
               "Mirror's Edge: patch the game to stop issuing hardware occlusion queries, as its toggleocclusion "
               "console command does. Remix answers them as unoccluded while ray tracing, so they only cost a "
               "bounding box draw per tested primitive, and with DirectionalLightmaps a depth prepass. "
               "Applied inside the game process by the bridge client, only while rtx.d3d9.ue3EngineMode and ray "
               "tracing are enabled; see documentation/UE3Compatibility.md, \"Game executable patches\".");
    RTX_OPTION("rtx.d3d9", bool, ue3DisableSceneCaptures, false,
               "Mirror's Edge: patch the game to stop updating scene capture probes, as \"show scenecapture\" does. "
               "Each capture renders the scene again into a texture Remix ignores, the whole level with "
               "rtx.d3d9.ue3DisableFrustumCulling. "
               "Applied inside the game process by the bridge client, only while rtx.d3d9.ue3EngineMode and ray "
               "tracing are enabled; see documentation/UE3Compatibility.md, \"Game executable patches\".");
    RTX_OPTION("rtx.d3d9", bool, ue3DisableDynamicShadows, false,
               "Mirror's Edge: patch the game to stop rendering its dynamic shadows, the shadow depths and their "
               "projections, as \"show dynamicshadows\" does. Remix ignores these draws. "
               "Applied inside the game process by the bridge client, only while rtx.d3d9.ue3EngineMode and ray "
               "tracing are enabled; see documentation/UE3Compatibility.md, \"Game executable patches\".");
    RTX_OPTION("rtx.d3d9", bool, ue3DisableDynamicLighting, false,
               "Mirror's Edge: patch the game to skip its per-light passes, modulated shadows and lighting-only post "
               "process effects, which \"viewmode unlit\" also skips; base pass shaders are unchanged. Remix ignores "
               "these draws. "
               "Applied inside the game process by the bridge client, only while rtx.d3d9.ue3EngineMode and ray "
               "tracing are enabled; see documentation/UE3Compatibility.md, \"Game executable patches\".");
    RTX_OPTION("rtx.d3d9", bool, ue3DisableVelocityPass, false,
               "Mirror's Edge: patch the game to skip the velocity pass its motion blur reads, as MotionBlur=False "
               "does. Remix computes its own motion vectors and ignores these draws. "
               "Applied inside the game process by the bridge client, only while rtx.d3d9.ue3EngineMode and ray "
               "tracing are enabled; see documentation/UE3Compatibility.md, \"Game executable patches\".");
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
    RTX_OPTION_ARGS("rtx.d3d9", bool, discardCaptureOnlyDrawFragments, true,
                    "Draws that are ray traced keep their original draw call only when the vertex shader has to run for "
                    "vertex capture; nothing reads what they rasterize, since the ray-traced image replaces the scene "
                    "render target and occlusion queries are answered conservatively. With this on those draws run with "
                    "an empty scissor rectangle, so the vertex shader (and the capture) runs but no fragments are shaded "
                    "or blended. Saves the fill cost of large dynamic geometry, e.g. particle sprites covering the screen "
                    "at the output resolution.",
                    args.flags = RtxOptionFlags::UserSetting);
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
               "UE3 compat: leave the object transform (LocalToWorld/WorldToLocal) out of the vertex-shader constant "
               "hash that feeds HashComponents::VertexShader, and so rules::FullGeometryHash. The shading-only "
               "constants (LightMapScale, lightmap/shadow coordinate scale-bias) are excluded by "
               "rtx.d3d9.ue3EngineMode regardless of this option, since they never reach a vertex position and "
               "LightMapScale otherwise moves the hash with the DirectionalLightmaps setting.\n"
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
    RTX_OPTION("rtx.d3d9", uint32_t, ue3GeometryMemoSelfCheckFrames, 0,
               "UE3 compat diagnostics: correctness check for rtx.d3d9.ue3StaticGeometryHashMemoization. Every N "
               "frames, draws that would be served from the geometry hash / bounding box memo are hashed in full "
               "instead; a geometry worker compares each hash component and the bounding box with the memoized "
               "entry and logs [GeometryHashMemoCheck] for the first 20 mismatches, naming the differing components. "
               "A mismatch means a buffer write reached the geometry without refreshing the buffer's content "
               "generation. The fresh result replaces the entry. 0 = off.");
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
    RTX_OPTION("rtx.d3d9", bool, ue3DecomposeInstancedDraws, true,
               "UE3 compat: split a D3D9 hardware-instanced draw into one ray-traced instance per hardware instance, "
               "reading each placement out of the instance-data stream (UE3's InstanceOffset/InstanceXAxis/"
               "InstanceYAxis/InstanceZAxis) rather than from a vertex shader constant.\n"
               "UE3 places foliage and PhysX/NxFluid mesh particles this way: one draw call, one identity LocalToWorld, "
               "and every placement in a dynamic vertex stream. Without this the batch collapses onto that single "
               "identity transform, and because vertex capture indexes its output buffer by vertex alone every hardware "
               "instance writes the same slots - the surviving positions are an arbitrary mix of placements, so the mesh "
               "renders as an exploded cluster of stretched triangles that reshuffles on every capture.\n"
               "Enabled, such draws take their object-space positions from the input assembler (no capture, nothing to "
               "race) and each instance is submitted with its own object-to-world transform, matching what the engine's "
               "non-instanced fallback path produces. A draw whose placements cannot be recovered is dropped, with the "
               "reason logged, rather than rendered at the world origin.\n"
               "Disabling it is a complete bypass, racing included, so it is a direct A/B. See "
               "rtx.d3d9.ue3LogInstancedDraws.");
    RTX_OPTION("rtx.d3d9", uint32_t, ue3MaxDecomposedInstances, 4096,
               "UE3 compat: upper bound on the hardware instances rtx.d3d9.ue3DecomposeInstancedDraws expands from one "
               "draw. Each instance becomes its own ray-traced submission, so a pathological batch would otherwise cost "
               "unbounded CPU time; the excess is dropped with a one-shot warning naming the draw. The default clears "
               "Mirror's Edge's densest debris scatters with headroom and stays well inside the draw call state queue's "
               "capacity.\n"
               "Cost is linear in instances, so lowering this is the reliable way to trade density for frame time. What "
               "it keeps is a fixed subset - the instances the game lists first, which is spawn order, so a batch thins "
               "out roughly evenly rather than losing one side of itself. Deliberately not the instances nearest the "
               "camera: that set changes as the view moves, so instances would appear and disappear, and since each is "
               "named by its position in the game's buffer a view-dependent selection also renames them and costs them "
               "their history.");
    RTX_OPTION_ARGS("rtx.d3d9", float, ue3DecomposedInstanceCullDistance, 0.f,
               "UE3 compat: drop decomposed hardware instances farther than this many world units from the camera, "
               "or 0 to keep every instance the game submitted.\n"
               "Each instance costs a ray-traced submission and a TLAS entry, and the game's own culling only removes "
               "what leaves the frustum, so this bounds what a far-off scatter costs while it is still on screen.\n"
               "It cannot help with a cluster you are standing in, where every instance is at much the same distance "
               "and the bound becomes all-or-nothing; use rtx.d3d9.ue3MaxDecomposedInstances for that. Being "
               "view-dependent it can also pop at its boundary, so keep the distance far enough out that the popping "
               "is not what you are looking at.",
               args.minValue = 0.f);
    RTX_OPTION("rtx.d3d9", bool, ue3StableDecomposedInstanceIdentity, true,
               "UE3 compat: give each instance produced by rtx.d3d9.ue3DecomposeInstancedDraws an identity that "
               "survives it moving, so Remix recognises it frame to frame by a hash lookup.\n"
               "Instance identity normally includes the object transform, which is ideal for the static geometry that "
               "dominates a scene but means anything moving misses the exact-identity lookup every frame and falls "
               "through to a spatial nearest-neighbour search. That search scans a cell neighbourhood sized from "
               "rtx.uniqueObjectDistance, so a dense cluster of moving objects lands its whole batch in one cell and "
               "the search becomes quadratic in the batch's size.\n"
               "Decomposed instances are the one case with a better answer available: they arrive in a stable order in "
               "the game's instance stream, so instance N of a batch can be named directly. Pairing is then exact "
               "rather than a proximity guess, which also makes their motion vectors correct. Lowering "
               "rtx.uniqueObjectDistance is not a substitute - it is global, and dropping it far enough to subdivide "
               "a cluster also stops camera-attached geometry matching during fast turns.");
    RTX_OPTION("rtx.d3d9", bool, ue3LogInstancedDrawStats, false,
               "UE3 compat diagnostics: log roughly once a second what hardware-instanced draws are costing - "
               "instanced draws and hardware instances per frame, how many were submitted, how many each bound "
               "dropped, and the wall time the submitting thread spent expanding them - which separates submission "
               "cost from the per-instance work the consumer thread and the GPU do.\n"
               "Also reports the instance-order stability rtx.d3d9.ue3StableDecomposedInstanceIdentity depends on: how "
               "far index-paired instances moved between frames, and how often a batch could not be compared because "
               "its instance count changed. A mean displacement on the scale of a batch's own extent would mean the "
               "game reorders its instance buffer, making that option unsound.");
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
    RTX_OPTION("rtx.d3d9", bool, ue3ReportMicIdentityChurn, true,
               "UE3 MaterialInstanceConstant support: warn once per material family that mints enough "
               "distinct identity hashes to make an animating register the only plausible explanation, "
               "naming the UniformVector_* registers whose values moved and printing the "
               "rtx.d3d9.ue3MicConstantIdentityExcludedMaterials entry that pins it. The threshold sits well "
               "above the size of a genuine colour-variant sibling set, which a frame-varying register "
               "passes within a second, so the two do not have to be told apart by hand.\n"
               "Identity is never altered as a result. The report exists so a family whose frame-varying "
               "register could not be recognised from dataflow - a fade or tint the bytecode cannot "
               "distinguish from an authored parameter - announces itself instead of quietly breaking the "
               "replacements anchored on it. It costs nothing until a family actually churns, so unlike "
               "rtx.logReplacementResolution it is on by default. Only active when rtx.d3d9.ue3EngineMode is enabled.");
    RTX_OPTION("rtx.d3d9", bool, ue3LogUvResolution, false,
               "UE3 compat: log the deterministic UV resolution decision (proven IA set / captured interpolant / legacy fallback) "
               "once per unique pixel shader + stage combination, including ambiguity diagnostics.");
    RTX_OPTION("rtx.d3d9", bool, ue3LogUvAffineDetail, false,
               "UE3 compat diagnostics: log the full UV affine chain behind the deterministic UV resolution of "
               "shader-path draws: per-component scale, cross and offset terms (immediate or constant-register "
               "component with factor, and inexactness), the live resolved 2x2 plus translation, the gates that "
               "allowed or rejected writing the texture transform, and the pixel shader CTAB names/values of "
               "referenced constant registers. On the first sighting of a pixel shader it also dumps every "
               "sampler's UV origin and affine chain with the currently bound textures. Logs once per distinct "
               "resolved transform, capped per shader+stage. Use this to diagnose texture-atlas materials whose "
               "tile offset is not applied, or a UV matrix (UE3 Rotator) that resolves inexactly.");
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
    RTX_OPTION("rtx.d3d9", bool, ue3LogInstancedDraws, false,
               "UE3 compat diagnostics: log a one-shot [UE3-Instanced] line per draw identity for every draw that "
               "uses D3D9 hardware instancing (a stream frequency above one, or a D3DSTREAMSOURCE_INSTANCEDATA "
               "stream): instance count, per-stream frequency/stride/dynamic usage, the vertex declaration, the "
               "classified vertex factory and pass, the resolved capture position source, whether a per-instance "
               "transform was recovered, the vertex/index counts, and the draw's shader/material/texture hashes. Use "
               "it to confirm which geometry a title places through hardware instancing - under UE3 that is foliage "
               "and PhysX/NxFluid mesh particles - and that rtx.d3d9.ue3DecomposeInstancedDraws is handling it.");
    RTX_OPTION("rtx.d3d9", fast_unordered_set, ue3TraceDrawTextureHashes, {},
               "UE3 compat diagnostics: texture image hashes whose draws are dumped as a full [UE3-DrawTrace] "
               "dossier - shader hashes, vertex factory, pass, capture position source, hardware instancing state with "
               "the first recovered instance transforms, the vertex declaration, per-stream buffer identity, the "
               "object-to-world transform, and the asset and full geometry hashes alongside the material hash. The "
               "asset geometry hash is what replacements anchor on, so this is also how to confirm a mesh's hash is "
               "stable across sessions. Logged once per draw identity, but resolving the geometry hash synchronously "
               "costs a worker sync - leave the list empty in normal use.");
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
    RTX_OPTION("rtx.d3d9", bool, deferredUiHdrReplay, true,
               "Replay deferred UI overlays the game drew into a floating-point render target (e.g. UE3 MaterialEffects, "
               "drawn into the HDR scene colour before the game's display transform) on Remix's linear HDR image, in the "
               "game's scene units, before tone mapping. Overlays drawn into 8-bit targets replay on the tone-mapped output "
               "either way. When disabled, every overlay replays on the tone-mapped output, where linear overlay maths runs "
               "on display-encoded colour: tints brighten the image and clip highlights.");
    RTX_OPTION("rtx", bool, enableIndexBufferMemoization, true, "CPU performance optimization, should generally be enabled.  Will reduce main thread time by caching processIndexBuffer operations and reusing when possible, this will come at the expense of some CPU RAM.");
    RTX_OPTION("rtx", bool, poolVertexCaptureBuffers, true,
               "CPU performance optimization (shader vertex capture). Transient capture buffers (those the UE3 static "
               "vertex-capture cache does not retain) are pooled in power-of-two size classes and reused once nothing refers "
               "to them and the GPU is done with them, instead of a new device-local buffer per draw; buffers idle for ~300 "
               "frames are released. Captures the static cache retains keep exact-size allocations so its byte budget stays "
               "accurate. Off: a new buffer per capture.");
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
    }

    /**
      * \brief: Signal that an occlusion query has ended for the current device
      */
    void EndOcclusionQuery() {
      --m_activeOcclusionQueries;
      assert(m_activeOcclusionQueries >= 0);
    }

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

    // rtx.d3d9.discardCaptureOnlyDrawFragments
    bool DiscardCaptureOnlyDrawFragmentsEnabled() const {
      return m_frameOptions.enableRaytracing && m_frameOptions.discardCaptureOnlyDrawFragments;
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
      * \brief: While suspended, draws take the raster-only path and nothing is captured for ray tracing.
      */
    void SetSceneCaptureSuspended(bool suspended) {
      m_sceneCaptureSuspended = suspended;
    }

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
      * \brief: Whether a texture hash is a UE3 lightmap discovered this session from a pixel
      * shader's CTAB sampler names (rtx.d3d9.ue3AutoDetectLightmapTextures). Process-wide and
      * append-only, so the texture paths outside the draw-call setup - which have no D3D9Rtx
      * to hand - can honour the discovery the same way they honour rtx.lightmapTextures.
      */
    static bool isAutoDetectedLightmapTexture(XXH64_hash_t textureHash);

    /**
      * \brief: Whether a texture is baked lighting Remix must leave alone - either tagged in
      * rtx.lightmapTextures or auto-detected this session.
      */
    static bool isLightmapTexture(XXH64_hash_t textureHash) {
      return lookupHash(RtxOptions::lightmapTextures(), textureHash) ||
             isAutoDetectedLightmapTexture(textureHash);
    }

    static Ue3GamePatchStatus getUe3GamePatchStatus() {
      const uint64_t status = s_ue3GamePatchStatus.load(std::memory_order_relaxed);
      Ue3GamePatchStatus result;
      result.answered = (status & kUe3GamePatchAnswered) != 0;
      result.active = uint32_t(status & 0xFFFFu);
      result.notFound = uint32_t(status >> 16) & 0xFFFFu;
      return result;
    }

  private: 
    // Written on the bridge message channel's thread: the active mask in bits 0-15, the not-found
    // mask in bits 16-31, and whether any answer has arrived.
    static constexpr uint64_t kUe3GamePatchAnswered = 1ull << 63;
    inline static std::atomic<uint64_t> s_ue3GamePatchStatus { 0 };

    static void registerAutoDetectedLightmapTexture(XXH64_hash_t textureHash);

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
    bool m_sceneCaptureSuspended = false;
    bool m_rtxInjectTriggered = false;
    bool m_forceGeometryCopy = false;
    bool m_forceIaTexcoordForOutlier = false;
    DWORD m_texcoordIndex = 0;
    DWORD m_iaTexcoordIndex = 0;
    uint8_t m_texcoordCompU = 0;
    uint8_t m_texcoordCompV = 1;

    UvResolutionMode m_uvResolutionMode = UvResolutionMode::LegacyTss;

    // Per draw: the TEXCOORD usage index carrying the UE3 particle colour the draw takes, and the
    // kVertexCaptureFlag_Color* flags prepareVertexCapture captures it with. UINT32_MAX for none.
    uint32_t m_ue3ParticleColorTexcoordIndex = UINT32_MAX;
    uint32_t m_ue3ParticleColorCaptureFlags = 0;

    // The previous draw's identity, for isUe3SecondTwoSidedTranslucentPass. Reset in EndFrame.
    XXH64_hash_t m_prevDrawVsPsHash = 0;
    XXH64_hash_t m_prevDrawTextureHash = 0;
    XXH64_hash_t m_prevDrawGeometryHash = 0;
    DWORD m_prevDrawCullMode = 0;
    bool isUe3SecondTwoSidedTranslucentPass(const DrawContext& drawContext);

    int m_activeOcclusionQueries = 0;

    Rc<DxvkBuffer> m_vsVertexCaptureData;

    fast_unordered_cache<Rc<DxvkSampler>> m_samplerCache;

    Ue3VertexFactoryType m_currentUe3VertexFactory = Ue3VertexFactoryType::Unknown;

    Ue3PassType m_currentUe3PassType = Ue3PassType::Unknown;
    fast_unordered_cache<Ue3VertexFactoryType> m_ue3VertexFactoryCache;

    // UE3 SDPG_Foreground tracking: the foreground DPG (first-person arms/weapon) renders
    // after the world DPG behind a mid-scene depth-only clear. Both reset in EndFrame.
    bool m_ue3SeenMainViewWorldDraw = false;
    bool m_ue3ForegroundDpgActive = false;

    static Ue3VertexFactoryType classifyUe3VertexFactory(const D3D9VertexElements& elements);
    static bool isUe3WorldGeometryVertexFactory(Ue3VertexFactoryType type);

    static Ue3InstancingInfo resolveUe3Instancing(const D3D9VertexElements& elements,
                                                  const std::array<UINT, caps::MaxStreams>& streamFreq,
                                                  uint32_t instanceCount);

    Ue3InstancingInfo m_currentUe3Instancing;

    // Placements for the current draw, filled in internalPrepareDraw and consumed by
    // CommitGeometryToRT. Empty for ordinary draws.
    std::vector<Ue3DecomposedInstance> m_ue3DecomposedInstances;

    // Why the placements could not be recovered for an instanced draw, or null when they were.
    // Refuses the draw rather than placing its object-space geometry at the world origin.
    const char* m_ue3InstanceTransformReadFailure = nullptr;

    void cullAndClampUe3InstanceTransforms(std::vector<Ue3DecomposedInstance>& instances,
                                           uint32_t& outCulledByDistance,
                                           uint32_t& outCulledByBudget) const;

    fast_unordered_cache<Ue3InstanceOrderProbe> m_ue3InstanceOrderProbes;
    void trackUe3InstanceOrderStability(XXH64_hash_t batchKey,
                                        const std::vector<Ue3DecomposedInstance>& instances);

    std::unordered_map<XXH64_hash_t, std::vector<Ue3InstancedBatchRecord>> m_ue3InstancedBatches;
    uint64_t m_ue3NextInstancedBatchId = 1;
    XXH64_hash_t resolveUe3InstancedBatchKey(const RasterGeometry& geoData,
                                             const std::vector<Ue3DecomposedInstance>& instances);
    XXH64_hash_t m_ue3DecomposedBatchKey = kEmptyHash;

    // rtx.d3d9.ue3LogInstancedDrawStats accumulators, reported and reset about once a second.
    void reportUe3InstancedDrawStats();
    uint32_t m_ue3InstancedStatFrames = 0;
    uint32_t m_ue3InstancedStatFrameStamp = 0;
    uint64_t m_ue3InstancedStatDraws = 0;
    uint64_t m_ue3InstancedStatInstancesSeen = 0;
    uint64_t m_ue3InstancedStatInstancesSubmitted = 0;
    uint64_t m_ue3InstancedStatCulledDistance = 0;
    uint64_t m_ue3InstancedStatCulledBudget = 0;
    uint64_t m_ue3InstancedStatSubmitNs = 0;
    uint64_t m_ue3InstancedStatOrderPairs = 0;
    uint64_t m_ue3InstancedStatOrderStablePairs = 0;
    uint64_t m_ue3InstancedStatOrderSizeChanges = 0;
    uint64_t m_ue3InstancedStatOrderComparableBatches = 0;
    double m_ue3InstancedStatOrderDisplacementSum = 0.0;
    float m_ue3InstancedStatOrderDisplacementMax = 0.f;

    // Keeps UE3's vertex lightmap coefficient streams and per-instance transform streams from
    // being mistaken for the surface's UVs.
    uint32_t resolveIaTexcoordIndex(uint32_t iaTexcoordIdx) const;

    // Half-or-larger in both dims (allows ScreenPercentage >50%) and aspect-matched to the
    // backbuffer so square SceneCapture RTs cannot pass as main-view-sized on widescreen.
    static bool ue3ViewportAspectMatchesBackbuffer(uint32_t vpW, uint32_t vpH, uint32_t bbW, uint32_t bbH);
    static bool ue3ViewportIsMainViewSized(uint32_t vpW, uint32_t vpH, uint32_t bbW, uint32_t bbH);

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

    // TdToneMapping's colour curves arrive as two 16x1 textures (ColorCurvesK/M) uploaded every frame. Their
    // texels are kept at unlock and joined with the tone-map pass constants when that draw is classified.
    std::unordered_map<const D3D9CommonTexture*, Ue3CurveTexels> m_ue3CurveTexelCache;
    bool m_ue3ToneMapCapturedThisFrame = false;

    // TdToneMapExposure constants from the most recent exposure draw, joined into the capture
    bool m_ue3HasExposureSettings = false;
    Vector4 m_ue3ExposureSettings = Vector4(0.f, 0.f, 0.f, 0.f);
    float m_ue3MaxDeltaDown = 0.f;

    // Captures the TdToneMapping grade constants + curve texels once per
    // frame at the (not raytraced) tonemap draw and forwards them to the
    // renderer; records the TdToneMapExposure constants when that pass is
    // the current draw.
    void maybeCaptureUe3ToneMapState();

    fast_unordered_cache<std::vector<Ue3VsConstantSymbol>> m_ue3VsConstantSymbols;
    // Names the register, e.g. "c12 (LocalToWorld[1])", falling back to the bare register.
    std::string describeUe3VsConstantRegister(XXH64_hash_t vsBytecodeHash, uint32_t reg) const;

    // keyed by XXH3 hash of the vertex shader DXSO bytecode
    fast_unordered_cache<Ue3VsShaderCtabInfo> m_ue3VsShaderCtabCache;
    std::optional<Ue3VsShaderCtabInfo> m_currentUe3CtabInfo;

    fast_unordered_cache<Ue3VsHashExclusions> m_ue3VsHashExclusionCache;
    // Borrowed from the cache above for the duration of the draw; never owned.
    const Ue3VsHashExclusions* m_currentUe3VsHashExclusions = nullptr;
    static Ue3VsHashExclusions buildUe3VsHashExclusions(const Ue3VsShaderCtabInfo& ctabInfo,
                                                        const std::vector<Ue3VsConstantSymbol>* symbols);

    // N-way: the main view, capture probes and utility shaders interleave distinct camera blocks within a
    // frame, so a single slot would rerun the matrix extraction per draw.
    static constexpr uint32_t kUe3CameraConstantsCacheSlots = 8;
    std::array<Ue3CameraConstantsCache, kUe3CameraConstantsCacheSlots> m_ue3CameraConstantsCache;
    uint32_t m_ue3CameraConstantsCacheNextSlot = 0;

    // Every input to the object-to-world disambiguation is in the key, so entries never go stale; the map
    // is cleared when it exceeds the cap.
    static constexpr size_t kUe3ObjectToWorldCacheMaxEntries = 32768;
    fast_unordered_cache<Matrix4> m_ue3ObjectToWorldCache;

    bool applyUe3ShaderConstantTransforms(const DrawContext& drawContext, DrawCallTransforms& transformData);

    fast_unordered_cache<PsSamplerTexcoordEntry> m_psSamplerTexcoordCache;
    fast_unordered_set m_loggedUvResolutions;

    // Lightmap hashes this device has already published to the session registry, so the
    // shared registry is only locked the first time each one is seen.
    fast_unordered_set m_ue3SeenLightmapTextures;

    // rtx.d3d9.ue3LogUvAffineDetail state: per-shader one-shot sampler dump, per distinct
    // resolved transform dedup, and a per (shader, stage) cap so panner/frame-varying
    // transforms cannot flood the log
    fast_unordered_set m_loggedUvAffineShaderDumps;
    fast_unordered_set m_loggedUvAffineDetails;
    fast_unordered_cache<uint16_t> m_uvAffineDetailLogCounts;

    Ue3DiffuseSelectionCache m_ue3DiffuseSelectionCache;
    // selection cache keys already dumped by rtx.d3d9.ue3LogAlbedoSelection
    fast_unordered_set m_loggedAlbedoSelections;

    Ue3TextureSpreadCache m_ue3TextureSpreadCache;

    uint32_t m_ue3FrameCounter = 0;

    XXH64_hash_t mixUe3InstanceTransformConstants(XXH64_hash_t seed) const;
    void logUe3UnboundAlbedoOnce(const D3D9CommonShader* pixelShader,
                                 XXH64_hash_t psHash,
                                 uint32_t usedSamplerMask,
                                 uint32_t usedTextureMask,
                                 const PsSamplerTexcoordEntry* inferredEntry);
    // rtx.d3d9.ue3ParticleVertexColor and rtx.d3d9.ue3MaterialFades for the draw being prepared.
    void applyUe3MaterialFades(XXH64_hash_t psHash, const std::vector<uint8_t>& bytecode,
                               const D3D9CommonShader* pixelShader, XXH64_hash_t textureSetShaderHash);

    // UE3 albedo selection refuses render targets: soft-particle and scene-colour buffers carry a
    // content hash and near-backbuffer area, so at high resolutions they outscore the real material
    // texture. Movie surfaces and explicitly tagged targets stay eligible. Only meaningful under
    // rtx.d3d9.ue3EngineMode, which callers gate on.
    bool isUe3RenderTargetRefusedAsAlbedo(D3D9CommonTexture* texture,
                                          uint32_t stage,
                                          const PsSamplerTexcoordEntry* inferredEntry) const;

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

    Ue3StaticVertexCaptureCache m_ue3StaticVertexCaptureCache;

    fast_unordered_cache<Ue3ChurnMeshEntry> m_ue3ConstantChurn;

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

    Ue3GeometryMemo m_ue3GeometryMemo;

    bool canMemoizeUe3IaGeometryHashes(const IndexContext& indexContext,
                                       const VertexContext vertexContext[caps::MaxStreams],
                                       const RasterGeometry& geoData) const;
    XXH64_hash_t computeUe3IaGeometryMemoKey(const IndexContext& indexContext,
                                             const VertexContext vertexContext[caps::MaxStreams],
                                             const DrawContext& drawContext,
                                             const RasterGeometry& geoData) const;
    // The geometry-hash VertexShader component for the current draw (stable VS-constant
    // hash plus position-source/outlier folds); shared by computeHash and the memo hit path.
    XXH64_hash_t computeGeometryVertexShaderHash();

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
    bool canUseUe3InstancedMeshVertexPositions(const RasterGeometry& geoData,
                                               const char** outReason = nullptr) const;

    // Reads m_currentUe3Instancing's per-instance basis out of the instance-data stream. Degenerate
    // placements are dropped: PhysX leaves unused slots in the emitter's instance buffer untouched.
    bool readUe3InstanceTransforms(const VertexContext vertexContext[caps::MaxStreams],
                                   std::vector<Ue3DecomposedInstance>& instances,
                                   const char** outReason = nullptr) const;

    // Diagnostics: rtx.d3d9.ue3LogInstancedDraws / rtx.d3d9.ue3TraceDrawTextureHashes.
    std::string describeUe3DrawInstancing(const VertexContext vertexContext[caps::MaxStreams]) const;
    std::string describeUe3VertexDeclaration() const;
    std::string describeUe3DrawIdentity() const;
    void logUe3InstancedDrawOnce(const DrawContext& drawContext,
                                 const VertexContext vertexContext[caps::MaxStreams],
                                 const RasterGeometry& geoData);
    void logUe3TracedDrawOnce(const DrawContext& drawContext,
                              const VertexContext vertexContext[caps::MaxStreams],
                              RasterGeometry& geoData);

    // draw identities already dumped by the instancing / texture-hash draw probes
    fast_unordered_set m_loggedUe3InstancedDraws;
    fast_unordered_set m_loggedUe3TracedDraws;
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
    // NV-DXVK start: draw disposition statistics (logged every 300 frames while the pass timer is on)
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
    void reportDrawDispositionStats();
    // NV-DXVK end

    static bool isPrimitiveSupported(const D3DPRIMITIVETYPE PrimitiveType) {
      return (PrimitiveType == D3DPT_TRIANGLELIST || PrimitiveType == D3DPT_TRIANGLEFAN || PrimitiveType == D3DPT_TRIANGLESTRIP);
    }

    const Direct3DState9& d3d9State() const;

    template<typename T>
    static void copyIndices(const uint32_t indexCount, T*& pIndicesDst, T* pIndices, uint32_t& minIndex, uint32_t& maxIndex);

    template<typename T>
    DxvkBufferSlice processIndexBuffer(const uint32_t indexCount, const uint32_t startIndex, const IndexContext& indexCtx, uint32_t& minIndex, uint32_t& maxIndex);

    struct VertexCapturePlan {
      bool captureTexcoords = false;
      bool captureNormals = false;
      bool captureColor = false;
      uint32_t texcoordOutputRegister = UINT32_MAX;
      uint32_t colorOutputRegister = UINT32_MAX;
      uint32_t flags = 0;   // kVertexCaptureFlag_*
      uint32_t boneMatricesBaseReg = 0;
      uint32_t boneCount = 0;
    };
    VertexCapturePlan planUe3VertexCapture(const D3D9CommonShader* vertexShader, Ue3CapturePositionSource positionSource) const;

    // allowPooledBuffer: the capture is transient (not retained by the UE3 static vertex-capture
    // cache), so rtx.poolVertexCaptureBuffers may serve it from the capture buffer pool.
    bool prepareVertexCapture(const int vertexIndexOffset, Ue3CapturePositionSource positionSource, bool allowPooledBuffer);

    // rtx.poolVertexCaptureBuffers: capture buffers in power-of-two size classes, reused once the
    // pool holds the last reference and no command list still uses them. App thread only.
    struct PooledCaptureBuffer {
      Rc<DxvkBuffer> buffer;
      uint32_t lastUsedFrame = 0;
    };
    struct CaptureBufferBucket {
      std::vector<PooledCaptureBuffer> buffers;
      size_t cursor = 0;
    };
    static constexpr VkDeviceSize kMinCaptureBufferClass = 4096;
    static constexpr size_t kCaptureBufferProbes = 16;
    static constexpr size_t kMaxPooledCaptureBuffersPerClass = 4096;
    static constexpr uint32_t kCaptureBufferMaxIdleFrames = 300;
    std::unordered_map<VkDeviceSize, CaptureBufferBucket> m_captureBufferPool;
    DxvkBufferSlice allocVertexCaptureBuffer(const VkDeviceSize size, bool allowPooledBuffer);
    void trimVertexCaptureBufferPool();

    void processVertices(const VertexContext vertexContext[caps::MaxStreams], int vertexIndexOffset, RasterGeometry& geoData);

    bool processRenderState(const DrawContext& drawContext);

    template<bool FixedFunction>
    bool processTextures();
    static Rc<DxvkImageView> getRemixSampleView(D3D9CommonTexture* texture, bool srgb);
    Ue3TextureState beginUe3TextureState(bool programmablePs);
    bool resolveInferredSamplerOffset(const PsSamplerTexcoordEntry* entry, uint32_t stage, float& outU, float& outV) const;
    bool hasNonZeroInferredSamplerOffset(const PsSamplerTexcoordEntry* entry, uint32_t stage) const;
    PsSamplerTexcoordEntry* getOrInitPsSamplerTexcoordEntry(const D3D9CommonShader* ps, XXH64_hash_t& outHash);
    void selectUe3BoundTextures(Ue3TextureState& ue3, uint32_t& firstStage);
    void computeUe3MaterialIdentity(Ue3TextureState& ue3, uint32_t firstStage);
    void registerUe3TexturelessMaterial();
    void markUe3MovieTextureMaterial(const Ue3TextureState& ue3, XXH64_hash_t materialHash, XXH64_hash_t textureHash);
    void resolveUe3Texcoords(Ue3TextureState& ue3, uint32_t firstStage, uint32_t stageStateIdx, uint32_t& texcoordIdx, uint32_t& iaTexcoordIdx);

    PrepareDrawFlags internalPrepareDraw(const IndexContext& indexContext, const VertexContext vertexContext[caps::MaxStreams], const DrawContext& drawContext);

    struct Ue3StaticVertexCaptureKey {
      bool canUseCache = false;
      XXH64_hash_t key = kEmptyHash;
    };

    // A draw's geometry hash memo entry: served from it, or the entry the computed hashes publish
    // to (and, during a self-check, verify against).
    struct Ue3GeometryMemoLookup {
      bool served = false;
      XXH64_hash_t key = kEmptyHash;
      std::shared_ptr<Ue3GeometryMemoEntry> publishTo;
      std::shared_ptr<const Ue3GeometryMemoEntry> verifyAgainst;
    };

    void classifyUe3DrawVertexFactory();
    void resolveUe3DrawInstancing();
    void readUe3DrawInstances(const VertexContext vertexContext[caps::MaxStreams], const DrawContext& drawContext,
                              const RasterGeometry& geoData);
    void updateUe3SkinnedDrawIdentity();
    bool resolveUe3DrawCaptureSource(const IndexContext& indexContext, const VertexContext vertexContext[caps::MaxStreams],
                                     const RasterGeometry& geoData);
    Ue3StaticVertexCaptureKey computeUe3DrawStaticVertexCaptureKey(const IndexContext& indexContext,
                                                                   const VertexContext vertexContext[caps::MaxStreams],
                                                                   const DrawContext& drawContext,
                                                                   const RasterGeometry& geoData);
    Ue3GeometryMemoLookup lookupUe3GeometryMemo(const IndexContext& indexContext,
                                                const VertexContext vertexContext[caps::MaxStreams],
                                                const DrawContext& drawContext,
                                                RasterGeometry& geoData);
    bool tryReuseUe3DrawStaticVertexCapture(const Ue3StaticVertexCaptureKey& key, RasterGeometry& geoData);
    void recordUe3StaticVertexCapture(const Ue3StaticVertexCaptureKey& key, bool reused, const RasterGeometry& geoData);

    // Occlusion-test draws whose query result is synthesized are ignored by the draw entry points
    // before the draw contexts are built.
    bool ignoreOcclusionTestDrawEarly();

    // A null targetImage injects into the backend's bound RT0. An hdrCanvas stages the injection
    // (see RtxContext::injectRTX), and finishInjectRTX must follow.
    void triggerInjectRTX(const Rc<DxvkImage>& targetImage = nullptr, const Rc<DxvkImage>& hdrCanvas = nullptr);

    // Deferred UI draws (see "Deferred overlays" in UE3Compatibility.md), snapshotted with their vertex and index ranges
    // because the game may re-lock its buffers before the replay.
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
      bool sceneLinear = false;
    };

    std::vector<DeferredUiDraw> m_deferredUiDraws;
    uint32_t m_deferredUiFrameVertexBytes = 0;
    bool m_replayingDeferredUiDraws = false;

    // The injection target's size; scene-linear overlays replay onto it before tone mapping.
    // Private ref: a public one would keep the device alive.
    Com<D3D9Surface, false> m_deferredUiHdrCanvas;

    // one-shot log keys (pixel shader hash mixed with the defer/refuse decision) for the
    // [RTX-DeferredUI] tag diagnostics
    fast_unordered_set m_deferredUiLoggedDecisions;

    // Checks whether the draw matches rtx.deferredUiTextures (image hash for non-RTs, or
    // resolution-agnostic RT descriptor hash) or rtx.d3d9.deferredUiPixelShaders.
    bool isDeferredUiTaggedDraw(XXH64_hash_t* pMatchedTextureHash = nullptr) const;

    bool captureDeferredUiDraw(const IndexContext& indexContext,
                               const VertexContext vertexContext[caps::MaxStreams],
                               const DrawContext& drawContext);
    // pOverrideRenderTarget: bind this surface as RT0 for the replay (the EndFrame backbuffer, or
    // the HDR canvas); nullptr replays onto the currently bound RT0 (mid-frame injection path).
    // sceneSourceImage: the image the overlays composite over, used as the source for the
    // scene-color refresh blit (may be null to skip).
    void replayDeferredUiDraws(std::vector<DeferredUiDraw> draws,
                               IDirect3DSurface9* pOverrideRenderTarget,
                               const Rc<DxvkImage>& sceneSourceImage);
    // Injects RTX into targetImage and replays the captured deferred overlays: scene-linear ones
    // onto the HDR canvas between the two injection stages, the rest afterwards onto
    // pDisplayOverlayTarget (the bound RT0 when null).
    void injectRtxWithOverlays(const Rc<DxvkImage>& targetImage, IDirect3DSurface9* pDisplayOverlayTarget);
    bool ensureDeferredUiHdrCanvas(const VkExtent3D& extent);
    Rc<DxvkImage> getCurrentRenderTargetImage() const;
    D3D9Format getCurrentRenderTargetFormat() const;
    // RT0 has a floating-point format: the game draws linear scene colour into it
    bool isSceneLinearRenderTarget() const;

    struct DrawCallType {
      RtxGeometryStatus status;
      bool triggerRtxInjection;
      // rtx.deferredUiTextures: rasterize on top of the ray-traced image without triggering
      // injection - the draw is captured and replayed after injection fires later in the frame
      bool deferUntilInjection = false;
    };

    // A draw's deferred-UI tag match (isDeferredUiTaggedDraw), evaluated at most once and only when
    // asked. A matched hash of zero means a pixel shader tag matched, which the option documents as
    // explicit intent, unlike a texture tag that a shared texture can trigger on any draw.
    class DeferredUiTagQuery {
    public:
      explicit DeferredUiTagQuery(const D3D9Rtx& rtx) : m_rtx(rtx) { }
      bool isTagged();
      bool isPixelShaderTagged();
      const XXH64_hash_t& matchedTextureHash() const { return m_matchedTextureHash; }
    private:
      const D3D9Rtx& m_rtx;
      XXH64_hash_t m_matchedTextureHash = 0;
      int m_state = -1;
    };

    bool isUe3DepthTestDisabledTranslucency(DeferredUiTagQuery& deferredUiTag) const;
    std::optional<DrawCallType> classifyUe3DrawPass(const DrawContext& drawContext, DeferredUiTagQuery& deferredUiTag);
    std::optional<DrawCallType> decideDeferredUiDraw(const DrawContext& drawContext, DeferredUiTagQuery& deferredUiTag);
    bool isUe3ShadowDepthPass() const;
    void logNonPrimaryRenderTargetOnce() const;
    std::optional<DrawCallType> classifyRenderTargetSamplingDraw(const DrawContext& drawContext);
    DrawCallType makeDrawCallType(const DrawContext& drawContext);

    bool checkBoundTextureCategory(const fast_unordered_set& textureCategory) const;

    // The bound texture slots, built lazily and shared by the draw's consumers. Invalidated at the top of
    // internalPrepareDraw; bindings cannot change within a draw.
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

    // Options read per draw, snapshotted once per frame: every RtxOption read takes the global option mutex,
    // which at UE3 draw counts is over 100k acquisitions a frame. Refreshed in EndFrame and on the first
    // draw. Fields are named after their options.
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
      bool autoRaytracedRenderTargetFromFullscreenComposite = false;
      bool rasterizeFullscreenCompositeToPrimary = false;
      bool ue3MicConstantIdentity = false;
      bool ue3MicExcludeRenderTargetsFromIdentity = true;
      bool ue3MicVolatileConstantDetection = true;
      bool ue3ReportMicIdentityChurn = true;
      bool ue3LogMaterialInstanceHash = false;
      bool ue3ForegroundDpgIsViewModel = false;
      bool conservativeOcclusionQueries = false;
      bool eventQueryCsCompletion = false;
      bool sequenceTrackedLockWaits = true;
      bool discardCaptureOnlyDrawFragments = true;
      bool skipRenderTargetCopies = true;
      Ue3StaticVertexCaptureCache::Settings ue3StaticVertexCaptureCache;
      bool ue3ExcludePlacementFromVertexShaderHash = false;
      bool ue3LogVertexConstantChurn = false;
      uint32_t ue3VertexConstantChurnMaxTrackedDraws = 0;
      bool ue3StaticGeometryHashMemoization = false;
      uint32_t ue3GeometryMemoSelfCheckFrames = 0;
      bool ue3ExactVertexCapture = false;
      bool ue3RequireExactVertexCapture = false;
      Ue3CapturePositionSourceOverride ue3VertexCaptureSourceOverride = Ue3CapturePositionSourceOverride::Auto;
      bool ue3NativeLocalMeshVertexCapture = false;
      bool ue3DecomposeInstancedDraws = false;
      uint32_t ue3MaxDecomposedInstances = 0;
      float ue3DecomposedInstanceCullDistance = 0.f;
      bool ue3StableDecomposedInstanceIdentity = false;
      bool ue3LogInstancedDrawStats = false;
      bool ue3AutoDetectLightmapTextures = false;
      float ue3ConstantAlbedoTintGain = 0.f;
      bool ue3HighlightTints = false;
      bool ue3HighlightTintRequireMotion = true;
      float ue3HighlightGlowIntensity = 0.f;
      bool ue3LogHighlightTints = false;
      Vector3 ue3HighlightDebugForceTint = Vector3(1.f, 1.f, 1.f);
      bool ue3ParticleVertexColor = false;
      bool ue3MaterialFades = false;
      bool ue3LogMaterialFades = false;
      float ue3MaterialFadeDebugForceCoverage = -1.f;
      bool ue3LogUvResolution = false;
      bool ue3LogUvAffineDetail = false;
      bool ue3LogAlbedoSelection = false;
      bool ue3LogCapturePrecision = false;
      bool ue3LogInstancedDraws = false;
      bool deferredUiReplay = false;
      bool deferredUiRefreshSceneColor = false;
      bool deferredUiHdrReplay = false;
      bool enableIndexBufferMemoization = false;
      bool poolVertexCaptureBuffers = false;

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

      // Set-typed options: pointers into m_frameOptionSets' copies. The options' own storage
      // cannot be read per draw without the option mutex (the CS thread assigns whole sets
      // under it when it resolves pending option changes); the copies are refreshed only when
      // g_rtxOptionResolveGeneration moved.
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
      const fast_unordered_set* ue3MicConstantIdentityExcludedMaterials = nullptr;
      const fast_unordered_set* ue3MicIdentityExcludedTextureDescHashes = nullptr;
      const fast_unordered_set* ue3HighlightTintExcludedMaterials = nullptr;
      const fast_unordered_set* ue3MaterialFadeExcludedMaterials = nullptr;
      const fast_unordered_set* ue3TraceDrawTextureHashes = nullptr;
      const fast_unordered_set* replacementDebugHashes = nullptr;
    };
    FrameOptionCache m_frameOptions;
    void refreshFrameOptionCache();

    // Storage behind FrameOptionCache's set pointers (see refreshFrameOptionSets).
    struct FrameOptionSets {
      uint64_t generation = ~0ull;
      fast_unordered_set uiTextures;
      fast_unordered_set deferredUiTextures;
      fast_unordered_set deferredUiPixelShaders;
      fast_unordered_set lightmapTextures;
      fast_unordered_set neverAlbedoTextures;
      fast_unordered_set preferredAlbedoTextures;
      fast_unordered_set smoothNormalsTextures;
      fast_unordered_set ignoreBakedLightingTextures;
      fast_unordered_set raytracedRenderTargetTextures;
      fast_unordered_set vsTexcoordCaptureOutlierTextures;
      fast_unordered_set ue3MicConstantIdentityExcludedShaders;
      fast_unordered_set ue3MicConstantIdentityExcludedMaterials;
      fast_unordered_set ue3MicIdentityExcludedTextureDescHashes;
      fast_unordered_set ue3HighlightTintExcludedMaterials;
      fast_unordered_set ue3MaterialFadeExcludedMaterials;
      fast_unordered_set ue3TraceDrawTextureHashes;
      fast_unordered_set replacementDebugHashes;
      Ue3AlbedoTagDigests albedoTagDigests;
    };
    FrameOptionSets m_frameOptionSets;
    void refreshFrameOptionSets();

    // Flushed to SceneManager::trackReplacementMaterialHash as one CS command in EndFrame. Its consumers read
    // the map in SceneManager::onFrameEnd, which runs after the flush on the CS timeline.
    std::vector<XXH64_hash_t> m_pendingReplacementMaterialHashes;

    // GamePatchBits and frustum bypass limits last sent to the bridge client by updateUe3GamePatchRequest.
    uint32_t m_ue3GamePatchRequest = 0;
    uint32_t m_ue3GamePatchLimits = 0;
    std::chrono::steady_clock::time_point m_ue3GamePatchRequestTime;
    void updateUe3GamePatchRequest();

    bool isRenderingUI();

    Future<SkinningData> processSkinning(const RasterGeometry& geoData);

    // When publishTo is non-null, the worker additionally publishes the computed result
    // into the memo entry so later frames can reuse it without recomputing.
    // When verifyAgainst is non-null (rtx.d3d9.ue3GeometryMemoSelfCheckFrames), the worker
    // compares its result with that published entry and logs [GeometryHashMemoCheck] on a
    // mismatch; the entry is only read (later draws may be served from it meanwhile).
    Future<AxisAlignedBoundingBox> computeAxisAlignedBoundingBox(const RasterGeometry& geoData,
                                                                 const std::shared_ptr<Ue3GeometryMemoEntry>& publishTo = {},
                                                                 const std::shared_ptr<const Ue3GeometryMemoEntry>& verifyAgainst = {});

    Future<GeometryHashes> computeHash(const RasterGeometry& geoData, const uint32_t maxIndexValue,
                                       const std::shared_ptr<Ue3GeometryMemoEntry>& publishTo = {},
                                       const std::shared_ptr<const Ue3GeometryMemoEntry>& verifyAgainst = {});

    void submitActiveDrawCallState();

    // Expands m_ue3DecomposedInstances into one ray-traced submission per hardware instance, each
    // with its own object-to-world transform.
    void submitUe3DecomposedInstances(const DrawParameters& params);
  };
}
