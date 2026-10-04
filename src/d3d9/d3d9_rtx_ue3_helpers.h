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

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "d3d9_rtx.h"
#include "d3d9_device.h"
#include "../dxso/dxso_material_fades.h"
#include "../dxso/dxso_ue3_material_identity.h"
#include "../util/xxHash/xxhash.h"

// Shared by the d3d9_rtx*.cpp files.

namespace dxvk {

  // We only look at RT 0 currently.
  const uint32_t kRenderTargetIndex = 0;

  // Defined here so the calls from every d3d9_rtx*.cpp file inline; the release build has no LTO.
  inline const Direct3DState9& D3D9Rtx::d3d9State() const {
    return *m_parent->GetRawState();
  }

  // Static in the cross-frame sense: content only changes through an explicit upload,
  // which bumps the buffer's remixContentGeneration counter (part of every cache key).
  inline bool isStaticD3D9Buffer(D3D9CommonBuffer* buffer) {
    if (buffer == nullptr || buffer->Desc() == nullptr) {
      return false;
    }

    if ((buffer->Desc()->Usage & D3DUSAGE_DYNAMIC) != 0 || buffer->WasWrittenByGPU()) {
      return false;
    }

    return !buffer->NeedsUpload();
  }

  // UE3 reserved D3D9 vertex shader registers (see UE3 engine Shaders/Common.usf and D3D9Drv/Src/D3D9Commands.cpp)
  constexpr uint32_t kUe3VsrViewProjMatrixRegister = 0; // c0..c3

  constexpr uint32_t kUe3VsrViewOriginRegister     = 4; // c4

  struct VDeclSignature {
    bool hasPosition = false;
    uint8_t positionType = 0;
    uint8_t positionStream = 0xFF;
    bool hasTangent = false;
    uint8_t tangentType = 0;
    uint8_t tangentStream = 0xFF;
    bool hasNormal = false;
    uint8_t normalType = 0;
    uint8_t normalStream = 0xFF;
    bool hasBinormal = false;
    uint8_t binormalType = 0;
    bool hasBlendWeight = false;
    uint8_t blendWeightType = 0;
    bool hasBlendIndices = false;
    bool hasColor0 = false;
    bool hasColor1 = false;
    uint8_t texcoordCount = 0;
    uint8_t maxTexcoordIndex = 0;
    uint8_t texcoordTypes[8] = {};
    bool hasTexcoord6 = false;
    uint8_t texcoord6Type = 0;
    bool hasTexcoord7 = false;
    uint8_t texcoord7Type = 0;
    uint8_t totalElements = 0;
  };

  // [RTX-MicDrift] (rtx.logReplacementResolution / rtx.replacementDebugHashes): the identity tiers behind
  // each material family's last minted hash, keyed by (shader identity seed, primary colour texture
  // hash), so a new mint can be attributed to the tier that moved.
  constexpr uint32_t kMicDriftMaxTrackedSamplers = 16;  // caps::MaxTexturesPS

  constexpr uint32_t kMicDriftMaxTrackedConstants = 16;

  // Per-family log caps: a family whose constants animate in a way volatile-register
  // classification cannot see would otherwise emit a drift warning per draw.
  constexpr uint32_t kMicDriftMaxLogsPerFamily = 16;

  constexpr uint32_t kMicDriftMaxLogsPerTrackedFamily = 64;

  struct Ue3MicIdentitySample {
    XXH64_hash_t materialHash = kEmptyHash;
    XXH64_hash_t textureSetHash = kEmptyHash;
    XXH64_hash_t constantsHash = kEmptyHash;
    bool constantsExcluded = false;
    bool valid = false;
    uint32_t driftLogsEmitted = 0;

    struct SamplerRecord {
      uint8_t reg = 0xFF;
      bool isRenderTarget = false;
      XXH64_hash_t imageHash = kEmptyHash;
      XXH64_hash_t descriptorHash = kEmptyHash;
    };
    std::array<SamplerRecord, kMicDriftMaxTrackedSamplers> samplers = {};
    uint32_t samplerCount = 0;

    struct ConstantRecord {
      uint16_t reg = 0xFFFF;
      Vector4 value;
    };
    std::array<ConstantRecord, kMicDriftMaxTrackedConstants> constants = {};
    uint32_t constantCount = 0;
  };

  // Runner Vision (rtx.d3d9.ue3HighlightTints), per draw. The pairs are proven once per shader;
  // the strength and colour are read live, because the game fades the strength on the material
  // instance it creates for each highlighted mesh element.
  struct Ue3HighlightTintDraw {
    const Vector4* fConsts = nullptr;
    // The material instance, whatever its strength (the material hash leaves UniformScalar_*
    // constants out), and the object drawn: that instance at this placement.
    XXH64_hash_t   materialHash = kEmptyHash;
    XXH64_hash_t   objectHash = kEmptyHash;
    bool           requireMotion = true;
    float          glowIntensity = 0.0f;
    bool           log = false;
  };

  struct Ue3IaMemoKeyHeader {
    D3D9Rtx::DrawContext drawContext; // 28 bytes, static_assert'd in d3d9_rtx.h
    uint32_t vertexCount;
    uint32_t indexCount;
    uint32_t topology;
    uint32_t texcoordIndex;
    uint32_t iaTexcoordIndex;
    uint32_t texcoordCompU;
    uint32_t texcoordCompV;
    uint32_t uvResolutionMode;
    uint32_t forceIaTexcoordForOutlier;
    uint32_t indexType;
    uint32_t explicitPad0;
    uint64_t ibo;
    uint64_t indexSliceHandle;
    uint64_t indexSliceOffset;
    uint64_t indexSliceLength;
    uint64_t indexContentGeneration;
  };

  static_assert(sizeof(Ue3IaMemoKeyHeader) == 28 + 11 * sizeof(uint32_t) + 5 * sizeof(uint64_t),
                "Ue3IaMemoKeyHeader must have no implicit padding (it is hashed by memory).");

  // shader may be null; the view then carries only the bytecode.
  DxsoShaderView makeDxsoShaderView(const std::vector<uint8_t>& bytecode, const D3D9CommonShader* shader);

  extern fast_unordered_set s_loggedNonPrimaryRtDescHashes;

  extern fast_unordered_set s_loggedSampledRtDescHashes;

  VDeclSignature buildVDeclSignature(const D3D9VertexElements& elements);

  // CTAB sampler register -> declared name. Cached per shader hash.
  const std::map<uint32_t, std::string>& getUe3PsSamplerNames(
      const XXH64_hash_t psHash,
      const std::vector<uint8_t>& bytecode);

  // CTAB float-constant register -> declared name (UniformScalar_*/UniformVector_*/engine
  // constants), for rtx.d3d9.ue3LogUvAffineDetail diagnostics. Cached per shader hash.
  const std::map<uint32_t, std::string>& getUe3PsFloatConstantNames(
      const XXH64_hash_t psHash,
      const std::vector<uint8_t>& bytecode);

  uint32_t digestTextureTags(const fast_unordered_set& tags);

  extern fast_unordered_cache<Ue3MicIdentitySample> s_ue3MicIdentityByFamily;

  extern fast_unordered_set s_ue3MicRtPoisonWarnedFamilies;

  // Bounded, for the per-family drift sample that is retained for every family while replacement
  // diagnostics are on.
  uint32_t snapshotUe3MicIdentityConstants(
      const Vector4* fConsts,
      const Ue3PsMaterialIdentityInfo& identityInfo,
      const bool useNameOrderStream,
      Ue3MicIdentitySample::ConstantRecord* out,
      const uint32_t outCapacity);

  // Unbounded: each family takes at most two, and a cap would silently drop the register that
  // moved on any shader declaring more uniforms than the cap - leaving a report that says a
  // family churned but not which value did, which is the only part worth reading.
  std::vector<Ue3MicIdentitySample::ConstantRecord> snapshotUe3MicIdentityConstants(
      const Vector4* fConsts,
      const Ue3PsMaterialIdentityInfo& identityInfo,
      const bool useNameOrderStream);

  void reportUe3MicIdentityChurnOnce(
      const XXH64_hash_t textureSetShaderHash,
      const XXH64_hash_t psHash,
      const XXH64_hash_t shaderIdentitySeed,
      const XXH64_hash_t primaryTextureHash,
      const XXH64_hash_t constantsHash,
      const Vector4* fConsts,
      const Ue3PsMaterialIdentityInfo& identityInfo,
      const bool useNameOrderStream,
      const std::vector<uint8_t>& bytecode);

  void logUe3MaterialInstanceHashBreakdownOnce(
      const XXH64_hash_t materialHash,
      const XXH64_hash_t psHash,
      const XXH64_hash_t shaderIdentitySeed,
      const XXH64_hash_t textureSetHash,
      const XXH64_hash_t textureSetShaderHash,
      const XXH64_hash_t constantsHash,
      const Ue3PsMaterialIdentityInfo& identityInfo,
      const bool constantsExcluded,
      const std::string& textureList);

  // Reused XXH3 streaming state: created once per thread instead of heap
  // allocating/freeing a state per hash operation on the per-draw path. The state is
  // fully reset before each use, so digests are identical to a fresh state.
  XXH3_state_t* getThreadLocalXxh3State();

  // Returns kEmptyHash when ranges is empty: without CTAB info, a raw register-range
  // fallback would fold per-view/per-mesh constants into the hash.
  XXH64_hash_t hashUe3MaterialConstants(
      const Vector4* fConsts,
      const Ue3MaterialConstRanges& ranges);

  // Hashes (name key, leading element) of each named Uniform* constant in name order: fxc trims arrays to
  // the elements a permutation references, so only the leading element is comparable across lightmap
  // policies.
  XXH64_hash_t hashUe3MaterialConstantsByNameOrder(
      const Vector4* fConsts,
      const std::vector<std::pair<XXH64_hash_t, uint32_t>>& namedUniformFirstRegistersByNameOrder);

  const Ue3PsMaterialIdentityInfo& getOrParseUe3PsMaterialIdentityInfo(
      const XXH64_hash_t psHash,
      const std::vector<uint8_t>& bytecode,
      const D3D9CommonShader* pixelShader,
      const bool detectVolatileConstants);

  // Multiplies each applying pair's tint into `tint` and adds its glow to `glow`; returns whether
  // any pair applied. appliedLog, when given, receives the live values of those that did.
  bool evaluateUe3HighlightTints(const Ue3PsMaterialIdentityInfo& info, const Ue3HighlightTintDraw& draw,
                                 Vector3& tint, Vector3& glow, std::string* appliedLog);

  void logUe3HighlightPairsOnce(const XXH64_hash_t psHash, const XXH64_hash_t shaderIdentitySeed,
                                const std::vector<uint8_t>& bytecode, const Ue3PsMaterialIdentityInfo& info);

  bool extractUe3CameraMatrices(
    const D3D9ShaderConstantsVSSoftware& vsConsts,
    const uint32_t viewProjRegisterBase,
    const uint32_t viewOriginRegister,
    Matrix4& outWorldToView,
    Matrix4& outViewToProjection,
    bool* outUsedTranspose = nullptr,
    float* outReconstructionError = nullptr);

  // Flat, padding-free records so the per-draw geometry identity keys hash in 2-3
  // XXH3 calls rather than one tiny seed-chained call per field: each
  // XXH3_64bits_withSeed call pays fixed setup/finalize costs that dwarf the mixing.
  struct Ue3KeyStreamRecord {
    uint64_t pVBO;
    uint64_t sliceHandle;
    uint64_t sliceOffset;
    uint64_t sliceLength;
    uint64_t contentGeneration;
    uint32_t stride;
    uint32_t offset;
  };

  static_assert(sizeof(Ue3KeyStreamRecord) == 48, "Ue3KeyStreamRecord must have no implicit padding (it is hashed by memory).");

  // D3D9 vertex declarations are bounded by MAXD3DDECLLENGTH (64); records are hashed
  // in bounded chunks anyway so larger inputs would simply chain across chunks.
  constexpr size_t kUe3KeyStreamRecordChunk = 64;

  // templated so the private D3D9Rtx::VertexContext type is deduced rather than named
  template<typename VertexContextT>
  Ue3KeyStreamRecord makeUe3KeyStreamRecord(const VertexContextT& ctx) {
    Ue3KeyStreamRecord record = {};
    record.pVBO = uint64_t(reinterpret_cast<uintptr_t>(ctx.pVBO));
    record.sliceHandle = uint64_t(reinterpret_cast<uintptr_t>(ctx.mappedSlice.handle));
    record.sliceOffset = uint64_t(ctx.mappedSlice.offset);
    record.sliceLength = uint64_t(ctx.mappedSlice.length);
    record.contentGeneration = ctx.pVBO != nullptr ? ctx.pVBO->remixContentGeneration : 0ull;
    record.stride = ctx.stride;
    record.offset = ctx.offset;
    return record;
  }

  // excludeStreamMask drops streams whose contents are not part of the identity being keyed.
  // Instance-data streams are the case that matters: they are reallocated and rewritten every
  // frame, so including them would mint a key that can never repeat, while none of their
  // contents reaches the geometry the key describes.
  template<typename ElementsT, typename VertexContextT>
  XXH64_hash_t hashUe3KeyStreamRecords(const ElementsT& elements,
                                       const VertexContextT* vertexContext,
                                       XXH64_hash_t seed,
                                       const uint32_t excludeStreamMask = 0) {
    // decl layout itself (semantics, formats, offsets, stream assignment)
    XXH64_hash_t hash = XXH3_64bits_withSeed(elements.data(), elements.size() * sizeof(D3DVERTEXELEMENT9), seed);

    // per-element stream identity
    std::array<Ue3KeyStreamRecord, kUe3KeyStreamRecordChunk> records;
    size_t recordCount = 0;
    for (const auto& element : elements) {
      // stream bounds are pre-validated by the canUse/canMemoize eligibility gates
      assert(element.Stream < caps::MaxStreams);
      if ((excludeStreamMask & (1u << element.Stream)) != 0) {
        continue;
      }
      records[recordCount++] = makeUe3KeyStreamRecord(vertexContext[element.Stream]);
      if (recordCount == records.size()) {
        hash = XXH3_64bits_withSeed(records.data(), recordCount * sizeof(Ue3KeyStreamRecord), hash);
        recordCount = 0;
      }
    }
    if (recordCount > 0) {
      hash = XXH3_64bits_withSeed(records.data(), recordCount * sizeof(Ue3KeyStreamRecord), hash);
    }

    return hash;
  }

  // templated so the private D3D9Rtx::IndexContext type is deduced rather than named
  template<typename IndexContextT>
  Ue3IaMemoKeyHeader makeUe3IaMemoKeyHeader(const IndexContextT& indexContext,
                                            const D3D9Rtx::DrawContext& drawContext,
                                            const RasterGeometry& geoData,
                                            const uint32_t texcoordIndex,
                                            const uint32_t iaTexcoordIndex,
                                            const uint32_t texcoordCompU,
                                            const uint32_t texcoordCompV,
                                            const uint32_t uvResolutionMode,
                                            const bool forceIaTexcoordForOutlier) {
    Ue3IaMemoKeyHeader header = {};
    header.drawContext = drawContext;
    header.vertexCount = geoData.vertexCount;
    header.indexCount = geoData.indexCount;
    header.topology = uint32_t(geoData.topology);
    header.texcoordIndex = texcoordIndex;
    header.iaTexcoordIndex = iaTexcoordIndex;
    header.texcoordCompU = texcoordCompU;
    header.texcoordCompV = texcoordCompV;
    header.uvResolutionMode = uvResolutionMode;
    header.forceIaTexcoordForOutlier = forceIaTexcoordForOutlier ? 1u : 0u;
    header.indexType = uint32_t(indexContext.indexType);
    header.ibo = uint64_t(reinterpret_cast<uintptr_t>(indexContext.ibo));
    header.indexSliceHandle = uint64_t(reinterpret_cast<uintptr_t>(indexContext.indexBuffer.handle));
    header.indexSliceOffset = uint64_t(indexContext.indexBuffer.offset);
    header.indexSliceLength = uint64_t(indexContext.indexBuffer.length);
    // content generation makes stale reuse impossible if the game rewrites the buffer
    header.indexContentGeneration = indexContext.ibo != nullptr ? indexContext.ibo->remixContentGeneration : 0ull;
    return header;
  }

}
