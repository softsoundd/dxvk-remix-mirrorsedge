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
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "d3d9_state.h"
#include "../util/util_fast_cache.h"
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

  // How a draw's UV set was resolved.
  enum class UvResolutionMode : uint8_t {
    LegacyTss = 0,      // nothing proven: the TSS texcoord index, with the capture fallbacks
    ProvenIa,           // the pixel and vertex shaders prove one IA texcoord set
    CaptureInterpolant, // the origin is proven but the VS math is not: capture the interpolant
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

  // How far index-paired instances moved between frames (rtx.d3d9.ue3LogInstancedDrawStats): the
  // pairing ue3StableDecomposedInstanceIdentity relies on holds only while this stays small.
  struct Ue3InstanceOrderProbe {
    std::vector<Vector3> translations;
    uint32_t lastFrame = 0;
  };

  // Transform-free identity of the current draw's instanced batch, combined with an instance's index
  // to name it across frames; kEmptyHash outside a decomposed draw.
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

  // Kept out of Ue3VsShaderCtabInfo, which is copied per draw. Filled for every shader at parse time, so
  // enabling the constant-churn diagnostic mid-session still names the registers of shaders already seen.
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
    // Camera registers plus the shading-only constants, which never reach a position (see "Placement
    // constants and the geometry hash" in UE3Compatibility.md).
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

  // The albedo pick for a (pixel shader, bound texture set, sRGB, vertex factory) key, pinned because
  // scoring reads constants UE3 rewrites per draw. A set bound with more texel area than
  // decisionAreaSum (more mips streamed in) re-scores and supersedes it.
  struct Ue3DiffuseSelectionEntry {
    uint8_t chosenStages[2] = { 0xFF, 0xFF };
    uint8_t cubemapFallbackStage = 0xFF;
    uint64_t decisionAreaSum = 0;
    // Runtime only, not serialised: only entries read from the cache file are worth auditing.
    bool fromDisk = false;
  };

  // Order-independent digests of the texture tag sets albedo scoring reads.
  struct Ue3AlbedoTagDigests {
    uint32_t lightmap = 0;
    uint32_t neverAlbedo = 0;
    uint32_t preferredAlbedo = 0;

    bool operator==(const Ue3AlbedoTagDigests& other) const {
      return lightmap == other.lightmap && neverAlbedo == other.neverAlbedo && preferredAlbedo == other.preferredAlbedo;
    }
    bool operator!=(const Ue3AlbedoTagDigests& other) const {
      return !(*this == other);
    }
  };

  // Persisted to rtx-remix/ue3DiffuseSelection.cache. The pin survives a level reload in memory
  // but not a relaunch, and a decision first made while a material's textures were still
  // streamed down can differ from the settled one, so the file is what makes every session
  // start from the same pick.
  class Ue3DiffuseSelectionCache {
  public:
    // Loads the file on first use. True when the tags changed since the stored picks were scored,
    // which drops them all.
    bool updateTags(const Ue3AlbedoTagDigests& tags);

    const Ue3DiffuseSelectionEntry* find(XXH64_hash_t key) const {
      const auto it = m_selections.find(key);
      return it != m_selections.end() ? &it->second : nullptr;
    }

    void erase(XXH64_hash_t key) {
      m_selections.erase(key);
    }

    void store(XXH64_hash_t key, const Ue3DiffuseSelectionEntry& entry) {
      m_selections[key] = entry;
      m_dirty = true;
    }

    // A stored pick records a decision, not the inputs behind it, and is consulted whenever the
    // bound texel area has not grown past the recorded peak - which after a settled run is
    // essentially always. Changed scoring would therefore be invisible wherever a cache exists,
    // so a bounded sample of loaded picks is re-scored and any disagreement reported once.
    bool takeAudit(const Ue3DiffuseSelectionEntry& entry);
    void reportAudit(XXH64_hash_t key, const uint8_t (&storedStages)[2], const uint8_t (&scoredStages)[2]);

    void saveIfDue(uint64_t frameId);
    void save();

  private:
    void load(const Ue3AlbedoTagDigests& tags);

    fast_unordered_cache<Ue3DiffuseSelectionEntry> m_selections;
    bool m_loaded = false;
    bool m_dirty = false;
    bool m_saveBlocked = false;
    uint32_t m_lastSaveFrame = 0;
    uint32_t m_auditsRemaining = 0;
    bool m_auditWarned = false;
    // The tags the stored picks were scored under.
    Ue3AlbedoTagDigests m_tags;
  };

  // Distinct pixel shaders seen sampling a texture. Scoring reads only scoringCount, the value loaded
  // from disk (see "Albedo selection and the texture spread cache" in UE3Compatibility.md).
  struct Ue3TextureMaterialSpread {
    std::array<XXH64_hash_t, 12> psHashes = {};
    uint8_t count = 0;
    uint8_t scoringCount = 0;
  };

  class Ue3TextureSpreadCache {
  public:
    // Loads the file on first use and records the shader for the next session. Returns the
    // spread to score with, which is the one loaded from disk.
    uint32_t recordShader(XXH64_hash_t textureHash, XXH64_hash_t psHash);

    void saveIfDue(uint64_t frameId);
    void save();

  private:
    void load();

    fast_unordered_cache<Ue3TextureMaterialSpread> m_spreads;
    bool m_loaded = false;
    bool m_dirty = false;
    // Set when the cache file existed but could not be read in full. Saving rewrites the
    // file from the map, so a partial load must never be allowed to publish itself.
    bool m_saveBlocked = false;
    uint32_t m_lastSaveFrame = 0;
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

  // CPU-only sighting record: a key must be seen on enough distinct frames before its capture buffer
  // is retained, so a key that never repeats (a moving or animating draw) costs no device memory.
  struct Ue3VertexCaptureAdmissionEntry {
    uint32_t vertexCount = 0;
    uint32_t sightings = 0;
    uint32_t lastFrameSeen = 0;
  };

  // rtx.d3d9.ue3StaticLocalMeshVertexCaptureCache: vertex captures of static meshes, served again
  // on later frames while the draw's key repeats.
  class Ue3StaticVertexCaptureCache {
  public:
    struct Settings {
      bool enabled = false;
      uint32_t warmupFrames = 0;
      uint32_t budgetMiB = 0;
      uint32_t maxEntries = 0;
      uint32_t retentionFrames = 0;
      uint32_t minReusePercent = 0;
      uint32_t reuseProbeFrames = 0;
      bool logStats = false;
    };

    bool isDormant() const {
      return m_dormant;
    }

    bool tryReuse(XXH64_hash_t key, uint32_t currentFrame, RasterGeometry& geoData);
    void recordCapture(XXH64_hash_t key, uint32_t currentFrame, const Settings& settings, const RasterGeometry& geoData);
    void endFrame(uint32_t currentFrame, const Settings& settings);

  private:
    void prune(uint32_t currentFrame, const Settings& settings);
    void enforceBudget(const Settings& settings);
    void erase(XXH64_hash_t key);
    void clear();
    void evaluateDormancy(const Settings& settings);
    void reportStats(uint32_t currentFrame, const Settings& settings);

    fast_unordered_cache<Ue3VertexCaptureCacheEntry> m_entries;
    // Sum of byteSize over m_entries, so the budget is enforced without walking the map every frame.
    VkDeviceSize m_bytes = 0;

    fast_unordered_cache<Ue3VertexCaptureAdmissionEntry> m_admission;
    uint32_t m_lastSweepFrame = 0;

    // The per-frame pair is folded at end of frame into the dormancy window (always) and the
    // logging interval (when rtx.d3d9.ue3LogStaticVertexCaptureCacheStats is on).
    uint32_t m_frameReuses = 0;
    uint32_t m_frameCaptures = 0;
    uint64_t m_statReuses = 0;
    uint64_t m_statCaptures = 0;
    uint32_t m_statFrames = 0;
    uint32_t m_statFrameStamp = 0;
    uint64_t m_evictions = 0;

    // Some titles recompute a draw's object transform every frame even for geometry that is not
    // moving, which mints a fresh key per draw per frame. Standing the cache down when its measured
    // reuse rate is hopeless keeps the option safe to leave enabled in any UE3 title.
    bool m_dormant = false;
    uint32_t m_probeCountdown = 0;
    uint32_t m_windowFrames = 0;
    uint64_t m_windowReuses = 0;
    uint64_t m_windowCaptures = 0;
  };

  // rtx.d3d9.ue3LogVertexConstantChurn: one entry per mesh (IA identity) rather than per placement, so
  // churn can be attributed to the IA identity, the set of placements, or another constant (see
  // "Diagnosing a cache that never hits" in UE3Compatibility.md).
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

  // Geometry hash components and bounds of a draw with static IA buffers, keyed on IA identity alone so
  // one entry serves every placement and pose; the VertexShader component is recombined live. A
  // geometry worker publishes into the entry (release on the ready flags) while the app thread owns
  // the map and serves it on later frames (acquire).
  struct Ue3GeometryMemoEntry {
    std::atomic<bool> hashesReady { false };
    std::atomic<bool> aabbReady { false };
    // per-component hashes; the VertexShader slot is intentionally left empty
    std::array<XXH64_hash_t, size_t(HashComponents::Count)> componentHashes = {};
    AxisAlignedBoundingBox boundingBox;
    uint32_t lastFrameTouched = 0;
  };

  class Ue3GeometryMemo {
  public:
    struct Lookup {
      // Published hashes to serve the draw from; null when it hashes in full.
      const Ue3GeometryMemoEntry* ready = nullptr;
      // Where a full hash publishes, and during a self-check the entry it must reproduce.
      std::shared_ptr<Ue3GeometryMemoEntry> publishTo;
      std::shared_ptr<const Ue3GeometryMemoEntry> verifyAgainst;
    };

    Lookup lookup(XXH64_hash_t key, uint32_t currentFrame, bool selfCheck);

    void erase(XXH64_hash_t key) {
      m_entries.erase(key);
    }

    void prune(uint32_t currentFrame);

  private:
    fast_unordered_cache<std::shared_ptr<Ue3GeometryMemoEntry>> m_entries;
  };

  // Per-draw state shared by the UE3 texture hooks of processTextures.
  struct Ue3TextureState {
    const D3D9CommonShader* inferredPs = nullptr;
    XXH64_hash_t inferredPsHash = 0;
    PsSamplerTexcoordEntry* inferredPsEntry = nullptr;
    Ue3VertexFactoryType vfType = Ue3VertexFactoryType::Unknown;
    bool isUe3GpuSkinVF = false;
    bool isUe3TerrainVF = false;
    bool isUe3ParticleVF = false;
    bool isUe3FoliageVF = false;
    bool isUe3SpeedTreeVF = false;
    bool isUe3LocalDecalVF = false;
    bool isUe3MorphVF = false;
    bool likelyGpuSkinnedMesh = false;
    const Ue3VsShaderCtabInfo* ue3VsHints = nullptr;
    bool likelyUe3DecalUvSpace = false;
    bool likelyUe3TerrainUvSpace = false;
    bool likelyUe3BillboardUvSpace = false;
    bool likelyUe3FlexiblePackedUvPath = false;
    bool likelyPackedUvConventions = false;
    bool selectedUe3MovieTexture = false;
  };

}
