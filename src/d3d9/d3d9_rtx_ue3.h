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

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "d3d9_state.h"
#include "../dxso/dxso_material_fades.h"
#include "../dxso/dxso_sampler_inference.h"
#include "../dxso/dxso_ue3_material_identity.h"
#include "../dxso/dxso_uv_dataflow.h"

namespace dxvk {

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

  // The bridge client's latest answer to the game patch request, as masks of GamePatchBits.
  struct Ue3GamePatchStatus {
    bool answered = false;
    uint32_t active = 0;
    uint32_t notFound = 0;
  };

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

  // One placement recovered from the instance-data stream. sourceIndex is the instance's position
  // in the game's buffer, which is what names it across frames - not its position in this vector,
  // which culling shifts.
  struct Ue3DecomposedInstance {
    Matrix4 instanceToObject;
    uint32_t sourceIndex = 0;
  };

  // Instance-order stability probe for rtx.d3d9.ue3StableDecomposedInstanceIdentity, whose pairing
  // of instance i with instance i of the previous frame is only sound while the game keeps its
  // instance buffer in a stable order. Measures how far index-paired instances moved between
  // consecutive frames: small means the order held, displacements on the scale of the batch's own
  // extent mean the pairing is meaningless.
  struct Ue3InstanceOrderProbe {
    std::vector<Vector3> translations;
    uint32_t lastFrame = 0;
  };

  // Transform-free identity of the instanced batch the current draw belongs to, combined with an
  // instance's index to name it across frames. kEmptyHash outside a decomposed draw.
  //
  // Deliberately not derived from the instance buffer: UE3 hands out a fresh or pooled instance
  // buffer per frame, so its handle is not an identity. The mesh streams are stable, so a batch is
  // identified by its mesh plus continuity of its own centroid - there are only a handful of
  // instanced batches per frame, so matching them is trivially cheap.
  struct Ue3InstancedBatchRecord {
    XXH64_hash_t id = kEmptyHash;
    Vector3 centroid = Vector3(0.f, 0.f, 0.f);
    uint32_t lastFrame = 0;
    uint32_t claimedFrame = 0;
  };

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
    // TdToneMapExposure pass
    int16_t exposureSettingsReg = -1;            // ExposureSettings (Manual, dt*SpeedUp, Low, High)
    int16_t maxDeltaDownReg = -1;                // MaxDeltaDown (dt*SpeedDown)
  };

  constexpr uint32_t kUe3CurveTexelCount = 16;
  struct Ue3CurveTexels {
    std::array<Vector4, kUe3CurveTexelCount> texels = {};
  };

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

  // pixel shader texcoord inference cache (for shader-path UV selection)
  struct PsSamplerTexcoordEntry {
    bool initialized = false;
    // exact per-sampler coordinate origin resolution (authoritative for the UV decision)
    std::array<PsSamplerUvOrigin, caps::MaxTexturesPS> samplerUvOrigin;
    // statistical inference below is used for diffuse-sampler *scoring* only
    std::array<PsSamplerTexcoordInference, caps::MaxTexturesPS> samplers;
    // expression flags the register-granular inference derived that the lane-precise
    // coordinate analysis showed to be impossible on the sampler's own lanes (diagnostic)
    std::array<uint16_t, caps::MaxTexturesPS> samplerExpressionFlagsCleared = {};
    // Sampler registers holding UE3 lightmap machinery (LightMapTextures[] coefficients and
    // the bicubic B-spline weight LUT). Their count is a function of the DirectionalLightmaps
    // setting - 3 coefficients vs 1 - so every draw-time decision that reads the bound texture
    // set has to subtract them or it varies with a setting Remix does not care about.
    uint32_t lightmapSamplerMask = 0;
  };

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

  struct Ue3VsTexcoordTraceEntry {
    bool initialized = false;
    Ue3VsUvTraceKind kind = Ue3VsUvTraceKind::Invalid;
    uint8_t iaTexcoordIndex = 0;
    uint8_t inputReg = 0;
    UvComponentAffine affineU;
    UvComponentAffine affineV;
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

  // Aggregates over the report window. Ordered map so the per-register report comes out in
  // register order. The shader hash is carried alongside the count purely so the report can
  // resolve the register's CTAB name without searching for a shader that declares it.
  struct Ue3ChurnRegisterTally {
    uint64_t count = 0;
    XXH64_hash_t vsBytecodeHash = 0;
  };

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

}
