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
#include "d3d9_rtx.h"
#include "d3d9_rtx_ue3_helpers.h"

#include "d3d9_include.h"
#include "d3d9_state.h"
#include "d3d9_util.h"
#include "d3d9_buffer.h"
#include "d3d9_device.h"
#include "d3d9_initializer.h"
#include "../util/util_fastops.h"
#include "../util/util_game_patches.h"
#include "../util/util_math.h"
#include "d3d9_rtx_utils.h"
#include "d3d9_texture.h"
#include "../dxso/dxso_color_terms.h"
#include "../dxso/dxso_highlight_tints.h"
#include "../dxso/dxso_material_fades.h"
#include "../dxso/dxso_sampler_inference.h"
#include "../dxso/dxso_ue3_material_identity.h"
#include "../dxso/dxso_uv_dataflow.h"
#include "../dxso/dxso_tables.h"
#include "../dxvk/rtx_render/rtx_bridge_message_channel.h"
#include "../dxvk/rtx_render/rtx_terrain_baker.h"
#include "../dxvk/rtx_render/rtx_ue3_tone_mapping.h"
#include "../dxvk/rtx_render/rtx_gpu_pass_timer.h"
#include "../dxvk/imgui/dxvk_imgui.h"
#include <algorithm>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <shared_mutex>
#include <sstream>
#include <system_error>

namespace dxvk {

  namespace {
    struct Ue3VertexCaptureKeyHeader {
      Ue3IaMemoKeyHeader iaHeader; // shared IA identity block
      uint32_t cullMode;
      uint32_t frontFace;
      uint32_t positionSource;
      uint32_t explicitPad0;
      uint64_t stableVsHash;
      Matrix4 objectToWorld;
    };

    static_assert(sizeof(Ue3VertexCaptureKeyHeader) ==
                    sizeof(Ue3IaMemoKeyHeader) + 4 * sizeof(uint32_t) + sizeof(uint64_t) + sizeof(Matrix4),
                  "Ue3VertexCaptureKeyHeader must have no implicit padding (it is hashed by memory).");
  }

  // Requires the vertex shader's oPos transform to have been recognised at compile time AND
  // its matrix register to be the one the CTAB names ViewProjectionMatrix. The analyzer only
  // proves "this register is multiplied by a matrix to make oPos"; the CTAB match is what
  // proves the register holds a world position rather than some factory-local space.
  bool D3D9Rtx::canUseUe3PreProjectionVertexCapture(const char** outReason) const {
    auto fail = [&](const char* reason) {
      if (outReason != nullptr) {
        *outReason = reason;
      }
      return false;
    };

    if (!m_frameOptions.ue3ExactVertexCapture) {
      return fail("exact capture disabled");
    }
    if (!m_frameOptions.ue3EngineMode) {
      return fail("UE3 engine mode disabled");
    }

    const D3D9CommonShader* vertexShader = GetCommonShader(d3d9State().vertexShader);
    if (vertexShader == nullptr) {
      return fail("no vertex shader");
    }

    const DxsoPreProjectionPositionInfo& preProj = vertexShader->GetPreProjectionPositionInfo();
    if (!preProj.valid) {
      // The analyzer's own reason is more specific than anything this layer could say.
      return fail(preProj.failureReason);
    }

    if (!m_currentUe3CtabInfo.has_value() || !m_currentUe3CtabInfo->hasViewProjectionMatrix) {
      return fail("CTAB declares no ViewProjectionMatrix");
    }
    if (preProj.matrixConstBase != m_currentUe3CtabInfo->viewProjectionMatrixRegisterIndex) {
      return fail("oPos matrix is not ViewProjectionMatrix");
    }

    if (outReason != nullptr) {
      *outReason = "";
    }
    return true;
  }

  Ue3CapturePositionSource D3D9Rtx::resolveUe3CapturePositionSource(
      const IndexContext& indexContext,
      const VertexContext vertexContext[caps::MaxStreams],
      const RasterGeometry& geoData,
      const char** outReason) const {
    const char* preProjReason = "";
    const bool canPreProj = canUseUe3PreProjectionVertexCapture(&preProjReason);
    const bool canInputAssembler = canUseUe3NativeLocalVertexCapture(indexContext, vertexContext, geoData);

    auto pick = [&](Ue3CapturePositionSource source, const char* reason) {
      if (outReason != nullptr) {
        *outReason = reason;
      }
      return source;
    };

    // Vertex capture cannot describe a hardware-instanced draw at all: its output buffer is
    // indexed by vertex, so every instance writes the same slots and the survivors are an
    // arbitrary mix of placements. The input assembler holds one object-space copy of the mesh,
    // which is the only self-consistent answer, and the placements come from the instance stream
    // instead (see readUe3InstanceTransforms). Overrides are deliberately not honoured here.
    if (m_currentUe3Instancing.instanceCount > 1) {
      const char* instancedReason = "";
      if (canUseUe3InstancedMeshVertexPositions(geoData, &instancedReason)) {
        return pick(Ue3CapturePositionSource::InputAssembler, "hardware-instanced draw");
      }
      return pick(Ue3CapturePositionSource::ClipReconstruction, instancedReason);
    }

    switch (m_frameOptions.ue3VertexCaptureSourceOverride) {
    case Ue3CapturePositionSourceOverride::ForcePreProjection:
      // A forced source that does not apply falls through rather than producing geometry in
      // the wrong space, so the override stays safe to leave on while investigating.
      if (canPreProj) {
        return pick(Ue3CapturePositionSource::PreProjectionRegister, "forced");
      }
      return pick(Ue3CapturePositionSource::ClipReconstruction, preProjReason);

    case Ue3CapturePositionSourceOverride::ForceInputAssembler:
      if (canInputAssembler) {
        return pick(Ue3CapturePositionSource::InputAssembler, "forced");
      }
      return pick(Ue3CapturePositionSource::ClipReconstruction, "not a conservative static local mesh");

    case Ue3CapturePositionSourceOverride::ForceReconstruction:
      return pick(Ue3CapturePositionSource::ClipReconstruction, "forced");

    case Ue3CapturePositionSourceOverride::Auto:
      break;
    }

    if (canPreProj) {
      return pick(Ue3CapturePositionSource::PreProjectionRegister, "");
    }
    if (canInputAssembler) {
      return pick(Ue3CapturePositionSource::InputAssembler, preProjReason);
    }
    return pick(Ue3CapturePositionSource::ClipReconstruction, preProjReason);
  }

  void D3D9Rtx::logUe3CapturePositionSource(const Ue3CapturePositionSource source, const char* reason) const {
    if (!m_frameOptions.ue3LogCapturePrecision) {
      return;
    }

    const D3D9CommonShader* vertexShader = GetCommonShader(d3d9State().vertexShader);
    const XXH64_hash_t vsHash = vertexShader != nullptr ? vertexShader->GetBytecodeHash() : kEmptyHash;

    // One line per shader per outcome: enough to enumerate everything that would be dropped
    // by ue3RequireExactVertexCapture without the volume of a per-draw log.
    static fast_unordered_set s_loggedSources;
    const XXH64_hash_t logKey =
      vsHash ^ (XXH64_hash_t(source) << 48) ^ (XXH64_hash_t(m_currentUe3VertexFactory) << 56);
    if (!s_loggedSources.insert(logKey).second) {
      return;
    }

    Logger::info(str::format(
      "[RTX-Compatibility][UE3-Capture] position source=", describeUe3CapturePositionSource(source),
      ", vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
      ", vs=0x", std::hex, vsHash, std::dec,
      (reason != nullptr && reason[0] != '\0') ? ", reason=" : "",
      (reason != nullptr) ? reason : ""));

    // On a fallback, follow up with what the shader's position math actually looked like -
    // a compiler is free to emit the same matrix multiply many ways, and this is the only
    // way to tell an unsupported shape apart from a detector bug without a debugger.
    if (source == Ue3CapturePositionSource::ClipReconstruction && vertexShader != nullptr) {
      const DxsoPreProjectionPositionInfo& preProj = vertexShader->GetPreProjectionPositionInfo();
      if (!preProj.positionDefinitions.empty()) {
        Logger::info(str::format(
          "[RTX-Compatibility][UE3-Capture]   vs=0x", std::hex, vsHash, std::dec,
          " oPos math: ", preProj.positionDefinitions));
      }
    }
  }

  bool D3D9Rtx::canUseUe3StaticVertexCaptureCache(const IndexContext& indexContext,
                                                  const VertexContext vertexContext[caps::MaxStreams],
                                                  const RasterGeometry& geoData) const {
    if (!m_frameOptions.ue3StaticLocalMeshVertexCaptureCache) {
      return false;
    }
    // Refusing here rather than at insertion also skips building the cache key, whose per-draw
    // stream hashing is most of the cost the cache exists to save.
    if (m_ue3VertexCaptureCacheDormant) {
      return false;
    }
    return isUe3StaticVertexCaptureCacheEligible(indexContext, vertexContext, geoData);
  }

  // The structural half of the test above: whether this draw is the kind the cache is for,
  // independent of whether the cache is enabled or has stood itself down. The constant-churn
  // diagnostic keys off this so it can explain a cache that is off or dormant.
  bool D3D9Rtx::isUe3StaticVertexCaptureCacheEligible(const IndexContext& indexContext,
                                                     const VertexContext vertexContext[caps::MaxStreams],
                                                     const RasterGeometry& geoData) const {
    if (!m_frameOptions.ue3EngineMode) {
      return false;
    }
    if (!m_parent->UseProgrammableVS()) {
      return false;
    }
    if (!m_frameOptions.useVertexCapture) {
      return false;
    }
    // Only exact sources are camera-independent, so only they can be reused across frames.
    // A cached clip-space reconstruction would pin the mesh to the reconstruction error of
    // whichever frame captured it, which is worse than recapturing every frame.
    if (!isUe3ExactCapturePositionSource(m_activeCapturePositionSource)) {
      return false;
    }
    // The cache key deliberately omits the camera constant registers, so it can only serve
    // factories whose world position is a function of the input assembler plus object and
    // bone constants. Camera-facing factories (sprites, billboards, leaf cards) and morphing
    // terrain move their vertices with the view and would be served a stale pose. Terrain is
    // also left out because a cache hit suppresses the original draw, which the terrain baker
    // needs to rasterize.
    switch (m_currentUe3VertexFactory) {
    case Ue3VertexFactoryType::Local:
    case Ue3VertexFactoryType::LocalDecal:
    case Ue3VertexFactoryType::GPUSkin:
    case Ue3VertexFactoryType::GPUSkinMorph:
    case Ue3VertexFactoryType::Foliage:
      break;
    default:
      return false;
    }
    // The skinned factories are on the list above because a skinned mesh held in a fixed pose is as
    // cacheable as a static one, but an animating one never is: its bone matrices are in the key's
    // stable VS-constant hash, so every pose mints a key that can never be hit again. Admission
    // would catch that, but a character animates essentially always, so refuse them up front rather
    // than pay a key and an admission record per draw per frame.
    if (geoData.blendWeightBuffer.defined() || geoData.blendIndicesBuffer.defined()) {
      return false;
    }
    // Instanced draws never capture in the first place; caching one would only pin a buffer that
    // is never consulted again.
    if (m_currentUe3Instancing.instanceCount > 1) {
      return false;
    }
    if (!m_currentUe3CtabInfo.has_value()) {
      return false;
    }
    if (!m_currentUe3CtabInfo->hasLocalToWorld) {
      return false;
    }
    if (!geoData.positionBuffer.defined()) {
      return false;
    }
    if (m_activeDrawCallState.testCategoryFlags(InstanceCategories::WorldUI) ||
        m_activeDrawCallState.testCategoryFlags(InstanceCategories::Terrain)) {
      return false;
    }
    if (d3d9State().vertexDecl == nullptr) {
      return false;
    }

    const auto& elements = d3d9State().vertexDecl->GetElements();

    if (indexContext.indexType != VK_INDEX_TYPE_NONE_KHR && !isStaticD3D9Buffer(indexContext.ibo)) {
      return false;
    }

    for (const auto& element : elements) {
      if (element.Stream >= caps::MaxStreams) {
        return false;
      }

      const VertexContext& ctx = vertexContext[element.Stream];
      if (ctx.mappedSlice.handle == VK_NULL_HANDLE || !isStaticD3D9Buffer(ctx.pVBO)) {
        return false;
      }
    }

    return true;
  }

  XXH64_hash_t D3D9Rtx::computeUe3StaticVertexCaptureCacheKey(const IndexContext& indexContext,
                                                              const VertexContext vertexContext[caps::MaxStreams],
                                                              const DrawContext& drawContext,
                                                              const RasterGeometry& geoData) const {
    constexpr uint64_t kSeed = 0x92F367D3E2F391A5ull;

    Ue3VertexCaptureKeyHeader header = {};
    header.iaHeader = makeUe3IaMemoKeyHeader(
        indexContext, drawContext, geoData,
        uint32_t(m_texcoordIndex), uint32_t(m_iaTexcoordIndex),
        uint32_t(m_texcoordCompU), uint32_t(m_texcoordCompV),
        uint32_t(m_uvResolutionMode), m_forceIaTexcoordForOutlier);
    header.cullMode = uint32_t(geoData.cullMode);
    header.frontFace = uint32_t(geoData.frontFace);
    // Positions from different sources are not interchangeable, so switching source (e.g. via
    // ue3VertexCaptureSourceOverride) must not serve an entry captured through the other one.
    header.positionSource = uint32_t(m_activeCapturePositionSource);
    // computed once per draw in internalPrepareDraw and shared with computeHash
    header.stableVsHash = m_activeStableVsHash;
    header.objectToWorld = m_activeDrawCallState.transformData.objectToWorld;

    const XXH64_hash_t headerHash = XXH3_64bits_withSeed(&header, sizeof(header), kSeed);
    return hashUe3KeyStreamRecords(d3d9State().vertexDecl->GetElements(), vertexContext, headerHash,
                                   m_currentUe3Instancing.instanceDataStreamMask);
  }

  // Second-choice exact source, for shaders whose oPos transform was not recognised. Only
  // safe where the shader does nothing to the position but move it rigidly by LocalToWorld,
  // so the conditions here stay deliberately narrow: static LocalVertexFactory meshes with
  // no skinning, decal, wind or view-space CTAB constants in play.
  bool D3D9Rtx::canUseUe3NativeLocalVertexCapture(const IndexContext& indexContext,
                                                  const VertexContext vertexContext[caps::MaxStreams],
                                                  const RasterGeometry& geoData) const {
    if (!m_frameOptions.ue3NativeLocalMeshVertexCapture || !m_frameOptions.ue3EngineMode) {
      return false;
    }
    if (!m_parent->UseProgrammableVS() || !m_frameOptions.useVertexCapture) {
      return false;
    }
    if (m_currentUe3VertexFactory != Ue3VertexFactoryType::Local) {
      return false;
    }
    if (!m_currentUe3CtabInfo.has_value() || !m_currentUe3CtabInfo->hasLocalToWorld) {
      return false;
    }
    const Ue3VsShaderCtabInfo& ctabInfo = *m_currentUe3CtabInfo;
    if (ctabInfo.hasDecalTransform ||
        ctabInfo.hasDecalLocation ||
        ctabInfo.hasDecalOffset ||
        ctabInfo.hasTextureCoordinateScaleBias ||
        ctabInfo.hasViewToLocal ||
        ctabInfo.hasWindMatrices) {
      return false;
    }
    if (!geoData.positionBuffer.defined() ||
        geoData.blendWeightBuffer.defined() ||
        geoData.blendIndicesBuffer.defined()) {
      return false;
    }
    if (m_activeDrawCallState.testCategoryFlags(InstanceCategories::WorldUI) ||
        m_activeDrawCallState.testCategoryFlags(InstanceCategories::Terrain)) {
      return false;
    }
    if (d3d9State().vertexDecl == nullptr) {
      return false;
    }

    const auto& elements = d3d9State().vertexDecl->GetElements();
    const VDeclSignature sig = buildVDeclSignature(elements);
    if (sig.positionStream == sig.tangentStream || sig.positionStream == sig.normalStream) {
      return false;
    }

    if (indexContext.indexType != VK_INDEX_TYPE_NONE_KHR && !isStaticD3D9Buffer(indexContext.ibo)) {
      return false;
    }

    for (const auto& element : elements) {
      if (element.Stream >= caps::MaxStreams) {
        return false;
      }

      const VertexContext& ctx = vertexContext[element.Stream];
      if (ctx.mappedSlice.handle == VK_NULL_HANDLE || !isStaticD3D9Buffer(ctx.pVBO)) {
        return false;
      }
    }

    return true;
  }

  bool D3D9Rtx::tryReuseUe3StaticVertexCapture(XXH64_hash_t cacheKey, RasterGeometry& geoData) {
    auto it = m_ue3VertexCaptureCache.find(cacheKey);
    if (it == m_ue3VertexCaptureCache.end()) {
      return false;
    }

    Ue3VertexCaptureCacheEntry& entry = it->second;
    const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();
    if (entry.vertexCount != geoData.vertexCount ||
        !entry.positionBuffer.defined() ||
        entry.lastFrameTouched == currentFrame) {
      return false;
    }

    geoData.positionBuffer = entry.positionBuffer;
    geoData.normalBuffer = entry.normalBuffer;
    geoData.texcoordBuffer = entry.texcoordBuffer;
    geoData.color0Buffer = entry.color0Buffer;
    entry.lastFrameTouched = currentFrame;
    ++m_ue3VertexCaptureCacheFrameReuses;

    return true;
  }

  // Two-tier admission. A fresh capture is recorded in the CPU-only admission map first and
  // only promoted to the retention tier once its key has been seen on enough distinct frames,
  // because the key covers the object transform and the shader's non-camera constants: any
  // draw that animates or moves mints a new key every frame, so retaining on first sighting
  // would pin a device-local capture buffer per such draw per frame with no possible reuse.
  void D3D9Rtx::updateUe3StaticVertexCaptureCache(XXH64_hash_t cacheKey, const RasterGeometry& geoData) {
    const VkDeviceSize budgetBytes =
      VkDeviceSize(m_frameOptions.ue3StaticLocalMeshVertexCaptureCacheBudgetMiB) * 1024ull * 1024ull;
    if (budgetBytes == 0) {
      return;
    }

    // The four views alias one capture buffer, so the position slice's length is the entry's
    // whole VRAM cost.
    const VkDeviceSize byteSize = geoData.positionBuffer.length();
    if (byteSize > budgetBytes) {
      // a single capture larger than the whole budget would evict everything and then itself
      return;
    }

    const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();

    // An already-retained key reaching here was recaptured rather than reused - drawn a second
    // time in one frame, or its vertex count changed - so refresh it in place instead of
    // sending it back through admission.
    auto it = m_ue3VertexCaptureCache.find(cacheKey);
    if (it == m_ue3VertexCaptureCache.end()) {
      const uint32_t warmupFrames = std::max(1u, m_frameOptions.ue3StaticLocalMeshVertexCaptureCacheWarmupFrames);

      Ue3VertexCaptureAdmissionEntry& admission = m_ue3VertexCaptureAdmission[cacheKey];
      if (admission.vertexCount != geoData.vertexCount) {
        // same key over different geometry: restart, the previous sightings prove nothing
        admission.vertexCount = geoData.vertexCount;
        admission.sightings = 0;
      }
      if (admission.lastFrameSeen != currentFrame || admission.sightings == 0) {
        // count one sighting per frame, so a mesh drawn many times within a single frame does
        // not look like a key that has proven itself across frames
        admission.sightings = std::min(admission.sightings + 1, 0xFFFFu);
        admission.lastFrameSeen = currentFrame;
      }

      if (admission.sightings < warmupFrames) {
        return;
      }

      it = m_ue3VertexCaptureCache.emplace(cacheKey, Ue3VertexCaptureCacheEntry {}).first;
      // the admission record has served its purpose; if the budget later evicts this entry
      // the key simply serves its warmup again
      m_ue3VertexCaptureAdmission.erase(cacheKey);
    }

    Ue3VertexCaptureCacheEntry& entry = it->second;
    m_ue3VertexCaptureCacheBytes -= std::min(m_ue3VertexCaptureCacheBytes, entry.byteSize);
    entry.positionBuffer = geoData.positionBuffer;
    entry.normalBuffer = geoData.normalBuffer;
    entry.texcoordBuffer = geoData.texcoordBuffer;
    entry.color0Buffer = geoData.color0Buffer;
    entry.vertexCount = geoData.vertexCount;
    entry.lastFrameTouched = currentFrame;
    entry.byteSize = byteSize;
    m_ue3VertexCaptureCacheBytes += byteSize;
  }

  void D3D9Rtx::eraseUe3StaticVertexCaptureCacheEntry(XXH64_hash_t cacheKey) {
    auto it = m_ue3VertexCaptureCache.find(cacheKey);
    if (it == m_ue3VertexCaptureCache.end()) {
      return;
    }
    m_ue3VertexCaptureCacheBytes -= std::min(m_ue3VertexCaptureCacheBytes, it->second.byteSize);
    m_ue3VertexCaptureCache.erase(it);
  }

  void D3D9Rtx::clearUe3StaticVertexCaptureCache() {
    m_ue3VertexCaptureCache.clear();
    m_ue3VertexCaptureCacheBytes = 0;
    m_ue3VertexCaptureAdmission.clear();
  }

  void D3D9Rtx::pruneUe3StaticVertexCaptureCache() {
    const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();
    const uint32_t retentionFrames =
      std::max(1u, m_frameOptions.ue3StaticLocalMeshVertexCaptureCacheRetentionFrames);

    // Expiry is a staleness bound hundreds of frames long and the budget below is what bounds
    // memory, so both maps are swept every 1/16 of the window rather than every frame.
    const uint32_t sweepIntervalFrames = std::max(1u, retentionFrames / 16);
    if (currentFrame - m_ue3VertexCaptureLastSweepFrame >= sweepIntervalFrames) {
      m_ue3VertexCaptureLastSweepFrame = currentFrame;

      m_ue3VertexCaptureCache.erase_if([&](auto it) {
        if (currentFrame - it->second.lastFrameTouched <= retentionFrames) {
          return false;
        }
        m_ue3VertexCaptureCacheBytes -= std::min(m_ue3VertexCaptureCacheBytes, it->second.byteSize);
        return true;
      });

      // Admission records are pure bookkeeping, but a churning key mints one per draw per frame,
      // so they need the same expiry to stay bounded. A key still warming up is re-seen every
      // frame, so the retention window is more than enough to keep it alive.
      m_ue3VertexCaptureAdmission.erase_if([&](auto it) {
        return currentFrame - it->second.lastFrameSeen > retentionFrames;
      });
    }

    enforceUe3StaticVertexCaptureCacheBudget();
  }

  // Backstop for the admission policy: if a game still manages to push more repeating keys
  // than expected (many placements of the same mesh each key on their own transform), the
  // cache gives up hit rate rather than VRAM.
  void D3D9Rtx::enforceUe3StaticVertexCaptureCacheBudget() {
    const VkDeviceSize budgetBytes =
      VkDeviceSize(m_frameOptions.ue3StaticLocalMeshVertexCaptureCacheBudgetMiB) * 1024ull * 1024ull;
    const size_t maxEntries = m_frameOptions.ue3StaticLocalMeshVertexCaptureCacheMaxEntries;

    auto overBudget = [&]() {
      return m_ue3VertexCaptureCacheBytes > budgetBytes ||
             (maxEntries > 0 && m_ue3VertexCaptureCache.size() > maxEntries);
    };

    if (!overBudget()) {
      return;
    }

    // A once-per-session note: from here on the cache trades hit rate for a VRAM ceiling, which
    // is worth knowing when diagnosing why captures are not being reused.
    ONCE(Logger::warn(str::format(
      "[RTX-Compatibility][UE3-Capture] static vertex capture cache reached its limit (",
      m_ue3VertexCaptureCacheBytes / (1024 * 1024), " MiB over ",
      m_ue3VertexCaptureCache.size(), " entries, budget ",
      budgetBytes / (1024 * 1024), " MiB / ", maxEntries, " entries); evicting least recently "
      "used entries from here on. Raise rtx.d3d9.ue3StaticLocalMeshVertexCaptureCacheBudgetMiB "
      "if reuse suffers.")));

    if (budgetBytes == 0) {
      clearUe3StaticVertexCaptureCache();
      return;
    }

    std::vector<std::pair<uint32_t, XXH64_hash_t>> byLastTouched;
    byLastTouched.reserve(m_ue3VertexCaptureCache.size());
    for (const auto& [key, entry] : m_ue3VertexCaptureCache) {
      byLastTouched.emplace_back(entry.lastFrameTouched, key);
    }
    std::sort(byLastTouched.begin(), byLastTouched.end());

    for (const auto& candidate : byLastTouched) {
      if (!overBudget()) {
        break;
      }
      eraseUe3StaticVertexCaptureCacheEntry(candidate.second);
      ++m_ue3VertexCaptureCacheEvictions;
    }
  }

  void D3D9Rtx::updateUe3StaticVertexCaptureCacheState() {
    const uint32_t reuses = m_ue3VertexCaptureCacheFrameReuses;
    const uint32_t captures = m_ue3VertexCaptureCacheFrameCaptures;
    m_ue3VertexCaptureCacheFrameReuses = 0;
    m_ue3VertexCaptureCacheFrameCaptures = 0;

    // Toggling the cache off in the dev UI should not leave a stale dormancy verdict behind,
    // and nothing should stay resident while it is off.
    if (!m_frameOptions.ue3StaticLocalMeshVertexCaptureCache) {
      if (m_ue3VertexCaptureCacheDormant || !m_ue3VertexCaptureCache.empty() || !m_ue3VertexCaptureAdmission.empty()) {
        clearUe3StaticVertexCaptureCache();
        m_ue3VertexCaptureCacheDormant = false;
        m_ue3VertexCaptureCacheProbeCountdown = 0;
      }
      return;
    }

    m_ue3VertexCaptureWindowReuses += reuses;
    m_ue3VertexCaptureWindowCaptures += captures;
    ++m_ue3VertexCaptureWindowFrames;

    m_ue3VertexCaptureCacheStatReuses += reuses;
    m_ue3VertexCaptureCacheStatCaptures += captures;
    ++m_ue3VertexCaptureCacheStatFrames;

    evaluateUe3StaticVertexCaptureCacheDormancy();
    reportUe3StaticVertexCaptureCacheStats();
  }

  void D3D9Rtx::evaluateUe3StaticVertexCaptureCacheDormancy() {
    auto resetWindow = [this]() {
      m_ue3VertexCaptureWindowFrames = 0;
      m_ue3VertexCaptureWindowReuses = 0;
      m_ue3VertexCaptureWindowCaptures = 0;
    };

    const uint32_t minReusePercent = m_frameOptions.ue3StaticLocalMeshVertexCaptureCacheMinReusePercent;
    if (minReusePercent == 0) {
      // Guard disabled: never stand the cache down, and keep the window empty so re-enabling the
      // guard later judges the cache on fresh frames rather than accumulated history.
      m_ue3VertexCaptureCacheDormant = false;
      m_ue3VertexCaptureCacheProbeCountdown = 0;
      resetWindow();
      return;
    }

    if (m_ue3VertexCaptureCacheDormant) {
      // Nothing is eligible while dormant, so the window cannot measure anything. Wait out the
      // probe interval, then wake up and let the window below judge the cache afresh - a later
      // level or a different shader set may well be cacheable.
      if (m_ue3VertexCaptureCacheProbeCountdown > 0) {
        --m_ue3VertexCaptureCacheProbeCountdown;
        resetWindow();
        return;
      }
      m_ue3VertexCaptureCacheDormant = false;
      resetWindow();
      if (m_frameOptions.ue3LogStaticVertexCaptureCacheStats) {
        Logger::info("[RTX-Compatibility][UE3-Capture] static vertex capture cache waking to re-measure its reuse rate.");
      }
      return;
    }

    // Long enough to span a camera pause without being fooled by one, short enough that a
    // hopeless title is stood down within a few seconds of gameplay.
    constexpr uint32_t kReuseEvaluationFrames = 120;
    if (m_ue3VertexCaptureWindowFrames < kReuseEvaluationFrames) {
      return;
    }

    const uint64_t considered = m_ue3VertexCaptureWindowReuses + m_ue3VertexCaptureWindowCaptures;
    const uint32_t reusePercent = considered > 0
      ? uint32_t((m_ue3VertexCaptureWindowReuses * 100ull) / considered)
      : 100u;
    resetWindow();

    if (considered == 0) {
      // no eligible draws in the window (menu, loading screen): no evidence either way
      return;
    }
    if (reusePercent >= minReusePercent) {
      return;
    }

    m_ue3VertexCaptureCacheDormant = true;
    m_ue3VertexCaptureCacheProbeCountdown = m_frameOptions.ue3StaticLocalMeshVertexCaptureCacheReuseProbeFrames;
    const size_t releasedEntries = m_ue3VertexCaptureCache.size();
    const VkDeviceSize releasedBytes = m_ue3VertexCaptureCacheBytes;
    const size_t releasedKeys = m_ue3VertexCaptureAdmission.size();
    clearUe3StaticVertexCaptureCache();

    // Worth one line per session even without diagnostics on: it explains why an enabled
    // option stopped doing anything, and points at the option that turns the guard off.
    ONCE(Logger::info(str::format(
      "[RTX-Compatibility][UE3-Capture] static vertex capture cache stood down: only ", reusePercent,
      "% of eligible draws were reused (threshold ", minReusePercent, "%), so its keys are not repeating in this "
      "title - most likely a camera-dependent vertex shader constant or object transform. Released ",
      releasedBytes / (1024 * 1024), " MiB over ", releasedEntries, " entries and ", releasedKeys,
      " pending keys. Re-tested every ", m_ue3VertexCaptureCacheProbeCountdown, " frames; set "
      "rtx.d3d9.ue3StaticLocalMeshVertexCaptureCacheMinReusePercent = 0 to keep it running regardless.")));

    if (m_frameOptions.ue3LogStaticVertexCaptureCacheStats) {
      Logger::info(str::format(
        "[RTX-Compatibility][UE3-Capture] static vertex capture cache going dormant at ", reusePercent,
        "% reuse; released ", releasedBytes / (1024 * 1024), " MiB / ", releasedEntries, " entries / ",
        releasedKeys, " pending keys."));
    }
  }

}
