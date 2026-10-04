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

    // Capture cannot separate hardware instances, so instanced draws read IA positions whatever the
    // override (see "Hardware-instanced mesh particles and foliage" in UE3Compatibility.md).
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
    if (!m_frameOptions.ue3StaticVertexCaptureCache.enabled) {
      return false;
    }
    // Refusing here rather than at insertion also skips building the cache key, whose per-draw
    // stream hashing is most of the cost the cache exists to save.
    if (m_ue3StaticVertexCaptureCache.isDormant()) {
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
    // The key omits camera registers, so only factories whose positions ignore the view qualify. Terrain
    // is out as well: a hit suppresses the draw the terrain baker rasterizes.
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
    // An animating skinned mesh mints a new key every pose, so it is refused up front rather than paying
    // for a key and an admission record per draw per frame.
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

  bool Ue3StaticVertexCaptureCache::tryReuse(const XXH64_hash_t key, const uint32_t currentFrame, RasterGeometry& geoData) {
    auto it = m_entries.find(key);
    if (it == m_entries.end()) {
      return false;
    }

    Ue3VertexCaptureCacheEntry& entry = it->second;
    if (entry.vertexCount != geoData.vertexCount || !entry.positionBuffer.defined()) {
      return false;
    }

    geoData.positionBuffer = entry.positionBuffer;
    geoData.normalBuffer = entry.normalBuffer;
    geoData.texcoordBuffer = entry.texcoordBuffer;
    geoData.color0Buffer = entry.color0Buffer;
    entry.lastFrameTouched = currentFrame;
    ++m_frameReuses;

    return true;
  }

  // A capture is retained only once admitted: a draw that moves or animates mints a new key every frame
  // and would otherwise pin a device-local buffer per draw per frame.
  void Ue3StaticVertexCaptureCache::recordCapture(const XXH64_hash_t key, const uint32_t currentFrame,
                                                  const Settings& settings, const RasterGeometry& geoData) {
    ++m_frameCaptures;

    const VkDeviceSize budgetBytes =
      VkDeviceSize(settings.budgetMiB) * 1024ull * 1024ull;
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

    // An already-retained key reaching here was recaptured because its vertex count changed, so
    // refresh it in place instead of sending it back through admission.
    auto it = m_entries.find(key);
    if (it == m_entries.end()) {
      const uint32_t warmupFrames = std::max(1u, settings.warmupFrames);

      Ue3VertexCaptureAdmissionEntry& admission = m_admission[key];
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

      it = m_entries.emplace(key, Ue3VertexCaptureCacheEntry {}).first;
      // the admission record has served its purpose; if the budget later evicts this entry
      // the key simply serves its warmup again
      m_admission.erase(key);
    }

    Ue3VertexCaptureCacheEntry& entry = it->second;
    m_bytes -= std::min(m_bytes, entry.byteSize);
    entry.positionBuffer = geoData.positionBuffer;
    entry.normalBuffer = geoData.normalBuffer;
    entry.texcoordBuffer = geoData.texcoordBuffer;
    entry.color0Buffer = geoData.color0Buffer;
    entry.vertexCount = geoData.vertexCount;
    entry.lastFrameTouched = currentFrame;
    entry.byteSize = byteSize;
    m_bytes += byteSize;
  }

  void Ue3StaticVertexCaptureCache::erase(const XXH64_hash_t key) {
    auto it = m_entries.find(key);
    if (it == m_entries.end()) {
      return;
    }
    m_bytes -= std::min(m_bytes, it->second.byteSize);
    m_entries.erase(it);
  }

  void Ue3StaticVertexCaptureCache::clear() {
    m_entries.clear();
    m_bytes = 0;
    m_admission.clear();
  }

  void Ue3StaticVertexCaptureCache::prune(const uint32_t currentFrame, const Settings& settings) {
    const uint32_t retentionFrames =
      std::max(1u, settings.retentionFrames);

    // Expiry is a staleness bound hundreds of frames long and the budget below is what bounds
    // memory, so both maps are swept every 1/16 of the window rather than every frame.
    const uint32_t sweepIntervalFrames = std::max(1u, retentionFrames / 16);
    if (currentFrame - m_lastSweepFrame >= sweepIntervalFrames) {
      m_lastSweepFrame = currentFrame;

      m_entries.erase_if([&](auto it) {
        if (currentFrame - it->second.lastFrameTouched <= retentionFrames) {
          return false;
        }
        m_bytes -= std::min(m_bytes, it->second.byteSize);
        return true;
      });

      // Admission records are pure bookkeeping, but a churning key mints one per draw per frame,
      // so they need the same expiry to stay bounded. A key still warming up is re-seen every
      // frame, so the retention window is more than enough to keep it alive.
      m_admission.erase_if([&](auto it) {
        return currentFrame - it->second.lastFrameSeen > retentionFrames;
      });
    }

    enforceBudget(settings);
  }

  // Backstop for the admission policy: if a game still manages to push more repeating keys
  // than expected (many placements of the same mesh each key on their own transform), the
  // cache gives up hit rate rather than VRAM.
  void Ue3StaticVertexCaptureCache::enforceBudget(const Settings& settings) {
    const VkDeviceSize budgetBytes =
      VkDeviceSize(settings.budgetMiB) * 1024ull * 1024ull;
    const size_t maxEntries = settings.maxEntries;

    auto overBudget = [&]() {
      return m_bytes > budgetBytes ||
             (maxEntries > 0 && m_entries.size() > maxEntries);
    };

    if (!overBudget()) {
      return;
    }

    // A once-per-session note: from here on the cache trades hit rate for a VRAM ceiling, which
    // is worth knowing when diagnosing why captures are not being reused.
    ONCE(Logger::warn(str::format(
      "[RTX-Compatibility][UE3-Capture] static vertex capture cache reached its limit (",
      m_bytes / (1024 * 1024), " MiB over ",
      m_entries.size(), " entries, budget ",
      budgetBytes / (1024 * 1024), " MiB / ", maxEntries, " entries); evicting least recently "
      "used entries from here on. Raise rtx.d3d9.ue3StaticLocalMeshVertexCaptureCacheBudgetMiB "
      "if reuse suffers.")));

    if (budgetBytes == 0) {
      clear();
      return;
    }

    std::vector<std::pair<uint32_t, XXH64_hash_t>> byLastTouched;
    byLastTouched.reserve(m_entries.size());
    for (const auto& [key, entry] : m_entries) {
      byLastTouched.emplace_back(entry.lastFrameTouched, key);
    }
    std::sort(byLastTouched.begin(), byLastTouched.end());

    for (const auto& candidate : byLastTouched) {
      if (!overBudget()) {
        break;
      }
      erase(candidate.second);
      ++m_evictions;
    }
  }

  void Ue3StaticVertexCaptureCache::endFrame(const uint32_t currentFrame, const Settings& settings) {
    prune(currentFrame, settings);

    const uint32_t reuses = m_frameReuses;
    const uint32_t captures = m_frameCaptures;
    m_frameReuses = 0;
    m_frameCaptures = 0;

    // Toggling the cache off in the dev UI should not leave a stale dormancy verdict behind,
    // and nothing should stay resident while it is off.
    if (!settings.enabled) {
      if (m_dormant || !m_entries.empty() || !m_admission.empty()) {
        clear();
        m_dormant = false;
        m_probeCountdown = 0;
      }
      return;
    }

    m_windowReuses += reuses;
    m_windowCaptures += captures;
    ++m_windowFrames;

    m_statReuses += reuses;
    m_statCaptures += captures;
    ++m_statFrames;

    evaluateDormancy(settings);
    reportStats(currentFrame, settings);
  }

  void Ue3StaticVertexCaptureCache::evaluateDormancy(const Settings& settings) {
    auto resetWindow = [this]() {
      m_windowFrames = 0;
      m_windowReuses = 0;
      m_windowCaptures = 0;
    };

    const uint32_t minReusePercent = settings.minReusePercent;
    if (minReusePercent == 0) {
      // Guard disabled: never stand the cache down, and keep the window empty so re-enabling the
      // guard later judges the cache on fresh frames rather than accumulated history.
      m_dormant = false;
      m_probeCountdown = 0;
      resetWindow();
      return;
    }

    if (m_dormant) {
      // Nothing is eligible while dormant, so the window cannot measure anything. Wait out the
      // probe interval, then wake up and let the window below judge the cache afresh - a later
      // level or a different shader set may well be cacheable.
      if (m_probeCountdown > 0) {
        --m_probeCountdown;
        resetWindow();
        return;
      }
      m_dormant = false;
      resetWindow();
      if (settings.logStats) {
        Logger::info("[RTX-Compatibility][UE3-Capture] static vertex capture cache waking to re-measure its reuse rate.");
      }
      return;
    }

    // Long enough to span a camera pause without being fooled by one, short enough that a
    // hopeless title is stood down within a few seconds of gameplay.
    constexpr uint32_t kReuseEvaluationFrames = 120;
    if (m_windowFrames < kReuseEvaluationFrames) {
      return;
    }

    const uint64_t considered = m_windowReuses + m_windowCaptures;
    const uint32_t reusePercent = considered > 0
      ? uint32_t((m_windowReuses * 100ull) / considered)
      : 100u;
    resetWindow();

    if (considered == 0) {
      // no eligible draws in the window (menu, loading screen): no evidence either way
      return;
    }
    if (reusePercent >= minReusePercent) {
      return;
    }

    m_dormant = true;
    m_probeCountdown = settings.reuseProbeFrames;
    const size_t releasedEntries = m_entries.size();
    const VkDeviceSize releasedBytes = m_bytes;
    const size_t releasedKeys = m_admission.size();
    clear();

    // Worth one line per session even without diagnostics on: it explains why an enabled
    // option stopped doing anything, and points at the option that turns the guard off.
    ONCE(Logger::info(str::format(
      "[RTX-Compatibility][UE3-Capture] static vertex capture cache stood down: only ", reusePercent,
      "% of eligible draws were reused (threshold ", minReusePercent, "%), so its keys are not repeating in this "
      "title - most likely a camera-dependent vertex shader constant or object transform. Released ",
      releasedBytes / (1024 * 1024), " MiB over ", releasedEntries, " entries and ", releasedKeys,
      " pending keys. Re-tested every ", m_probeCountdown, " frames; set "
      "rtx.d3d9.ue3StaticLocalMeshVertexCaptureCacheMinReusePercent = 0 to keep it running regardless.")));

    if (settings.logStats) {
      Logger::info(str::format(
        "[RTX-Compatibility][UE3-Capture] static vertex capture cache going dormant at ", reusePercent,
        "% reuse; released ", releasedBytes / (1024 * 1024), " MiB / ", releasedEntries, " entries / ",
        releasedKeys, " pending keys."));
    }
  }

  void Ue3StaticVertexCaptureCache::reportStats(const uint32_t currentFrame, const Settings& settings) {
    if (!settings.logStats) {
      return;
    }

    // roughly once a second at any plausible frame rate; this is a diagnostic, not a metric
    constexpr uint32_t kStatIntervalFrames = 60;
    if (currentFrame - m_statFrameStamp < kStatIntervalFrames) {
      return;
    }
    m_statFrameStamp = currentFrame;

    if (m_dormant) {
      Logger::info(str::format(
        "[RTX-Compatibility][UE3-Capture] static vertex capture cache: dormant, re-testing in ",
        m_probeCountdown, " frames."));
      m_statReuses = 0;
      m_statCaptures = 0;
      m_statFrames = 0;
      return;
    }

    const uint64_t considered = m_statReuses + m_statCaptures;
    const uint32_t reusePercent = considered > 0
      ? uint32_t((m_statReuses * 100ull) / considered)
      : 0u;

    Logger::info(str::format(
      "[RTX-Compatibility][UE3-Capture] static vertex capture cache: ",
      m_entries.size(), " retained entries, ",
      m_bytes / (1024 * 1024), " MiB, ",
      m_admission.size(), " keys awaiting admission, ",
      reusePercent, "% of eligible draws reused over the last ",
      m_statFrames, " frames, ",
      m_evictions, " total evictions."));

    m_statReuses = 0;
    m_statCaptures = 0;
    m_statFrames = 0;
  }

  // Resolve where this draw's positions come from before anything keys off it: the capture
  // cache is only valid for exact sources, and the strict gate below refuses the inexact one.
  // False when the draw has no acceptable position source and is ignored.
  bool D3D9Rtx::resolveUe3DrawCaptureSource(const IndexContext& indexContext,
                                            const VertexContext vertexContext[caps::MaxStreams],
                                            const RasterGeometry& geoData) {
    m_activeCapturePositionSource = Ue3CapturePositionSource::ClipReconstruction;
    if (m_parent->UseProgrammableVS() && m_frameOptions.useVertexCapture) {
      const char* positionSourceReason = "";
      m_activeCapturePositionSource = resolveUe3CapturePositionSource(
        indexContext, vertexContext, geoData, &positionSourceReason);
      logUe3CapturePositionSource(m_activeCapturePositionSource, positionSourceReason);

      if (m_frameOptions.ue3RequireExactVertexCapture &&
          !isUe3ExactCapturePositionSource(m_activeCapturePositionSource)) {
        ONCE(Logger::info("[RTX-Compatibility-Info] Ignoring draw without an exact vertex capture position source (rtx.d3d9.ue3RequireExactVertexCapture)."));
        return false;
      }

      // An instanced draw that did not land on input-assembler positions has no correct answer
      // left: capture would have every hardware instance overwrite one another's slots, producing
      // a cloud of triangles built from mixed placements that changes every time it is captured.
      // Dropping the draw is the honest outcome.
      if (m_currentUe3Instancing.instanceCount > 1 &&
          m_activeCapturePositionSource != Ue3CapturePositionSource::InputAssembler) {
        ONCE(Logger::warn(str::format(
          "[RTX-Compatibility] Ignoring hardware-instanced draw: its positions would have to come from "
          "vertex capture, which is indexed per vertex and so cannot separate instances (vf=",
          describeUe3VertexFactory(m_currentUe3VertexFactory),
          ", reason='", positionSourceReason, "') ",
          describeUe3DrawIdentity(), " ",
          describeUe3DrawInstancing(vertexContext),
          " decl=[", describeUe3VertexDeclaration(), "]")));
        return false;
      }
    }

    return true;
  }

  D3D9Rtx::Ue3StaticVertexCaptureKey D3D9Rtx::computeUe3DrawStaticVertexCaptureKey(const IndexContext& indexContext,
                                                                                   const VertexContext vertexContext[caps::MaxStreams],
                                                                                   const DrawContext& drawContext,
                                                                                   const RasterGeometry& geoData) {
    Ue3StaticVertexCaptureKey key;
    // Also read by the geometry hash, so it is computed once per draw here.
    m_activeStableVsHash = (m_parent->UseProgrammableVS() && m_frameOptions.useVertexCapture)
      ? computeUe3StableVertexShaderHash()
      : kEmptyHash;

    key.canUseCache =
      canUseUe3StaticVertexCaptureCache(indexContext, vertexContext, geoData);
    key.key =
      key.canUseCache
        ? computeUe3StaticVertexCaptureCacheKey(indexContext, vertexContext, drawContext, geoData)
        : kEmptyHash;
    return key;
  }

  bool D3D9Rtx::tryReuseUe3DrawStaticVertexCapture(const Ue3StaticVertexCaptureKey& key, RasterGeometry& geoData) {
    const bool reused = key.canUseCache &&
      m_ue3StaticVertexCaptureCache.tryReuse(key.key, m_parent->GetDXVKDevice()->getCurrentFrameId(), geoData);
    if (m_frameOptions.ue3LogCapturePrecision &&
        Logger::logLevel() <= LogLevel::Debug &&
        key.canUseCache) {
      static fast_unordered_set s_loggedCacheReuse;
      const XXH64_hash_t logKey = key.key ^ (reused ? 0x9E3779B97F4A7C15ull : 0xD1B54A32D192ED03ull);
      if (s_loggedCacheReuse.insert(logKey).second) {
        Logger::debug(str::format(
          "[RTX-Compatibility][UE3-Capture] static vertex capture cache ",
          reused ? "reused" : "recapturing",
          ", key=0x", std::hex, key.key, std::dec,
          ", vertices=", geoData.vertexCount));
      }
    }
    return reused;
  }

  void D3D9Rtx::recordUe3StaticVertexCapture(const Ue3StaticVertexCaptureKey& key, bool reused, const RasterGeometry& geoData) {
    if (key.canUseCache && !reused) {
      m_ue3StaticVertexCaptureCache.recordCapture(key.key, m_parent->GetDXVKDevice()->getCurrentFrameId(),
                                                  m_frameOptions.ue3StaticVertexCaptureCache, geoData);
    }
  }

  // What vertex capture writes for this draw. Reads the IA buffers processVertices bound, so it runs
  // before prepareVertexCapture replaces any of them with captured data.
  D3D9Rtx::VertexCapturePlan D3D9Rtx::planUe3VertexCapture(const D3D9CommonShader* vertexShader,
                                                           const Ue3CapturePositionSource positionSource) const {
    auto BoundShaderHas = [&](const D3D9CommonShader* shader, DxsoUsage usage, bool inOut)-> bool {
      if (shader == nullptr) {
        return false;
      }

      const auto& sgn = inOut ? shader->GetIsgn() : shader->GetOsgn();
      for (uint32_t i = 0; i < sgn.elemCount; i++) {
        const auto& decl = sgn.elems[i];
        if (decl.semantic.usageIndex == 0 && decl.semantic.usage == usage) {
          return true;
        }
      }
      return false;
    };

    auto BoundShaderHasAnyUsageIndex = [&](const D3D9CommonShader* shader, DxsoUsage usage, bool inOut)-> bool {
      if (shader == nullptr) {
        return false;
      }

      const auto& sgn = inOut ? shader->GetIsgn() : shader->GetOsgn();
      for (uint32_t i = 0; i < sgn.elemCount; i++) {
        const auto& decl = sgn.elems[i];
        if (decl.semantic.usage == usage) {
          return true;
        }
      }
      return false;
    };

    auto FindVsTexcoordOutputRegister = [&](const D3D9CommonShader* shader, uint32_t usageIndex) -> uint32_t {
      if (shader == nullptr) {
        return std::numeric_limits<uint32_t>::max();
      }

      const auto& osgn = shader->GetOsgn();
      for (uint32_t i = 0; i < osgn.elemCount; i++) {
        const auto& decl = osgn.elems[i];
        if (decl.semantic.usage == DxsoUsage::Texcoord && decl.semantic.usageIndex == usageIndex) {
          return decl.regNumber;
        }
      }
      return std::numeric_limits<uint32_t>::max();
    };

    auto FindVsTexcoordOutputRegisterByRegNumber = [&](const D3D9CommonShader* shader, uint32_t regNumber) -> uint32_t {
      if (shader == nullptr) {
        return std::numeric_limits<uint32_t>::max();
      }

      const auto& osgn = shader->GetOsgn();
      for (uint32_t i = 0; i < osgn.elemCount; i++) {
        const auto& decl = osgn.elems[i];
        if (decl.semantic.usage == DxsoUsage::Texcoord && decl.regNumber == regNumber) {
          return decl.regNumber;
        }
      }
      return std::numeric_limits<uint32_t>::max();
    };

    auto FindUniqueVsTexcoordOutputRegister = [&](const D3D9CommonShader* shader) -> uint32_t {
      if (shader == nullptr) {
        return std::numeric_limits<uint32_t>::max();
      }

      const auto& osgn = shader->GetOsgn();
      uint32_t foundReg = std::numeric_limits<uint32_t>::max();
      uint32_t foundCount = 0;
      for (uint32_t i = 0; i < osgn.elemCount; i++) {
        const auto& decl = osgn.elems[i];
        if (decl.semantic.usage != DxsoUsage::Texcoord) {
          continue;
        }

        foundReg = decl.regNumber;
        foundCount++;
        if (foundCount > 1) {
          return std::numeric_limits<uint32_t>::max();
        }
      }

      return foundCount == 1
        ? foundReg
        : std::numeric_limits<uint32_t>::max();
    };

    const RasterGeometry& geoData = m_activeDrawCallState.geometryData;

    const bool hasIaTexcoord = geoData.texcoordBuffer.defined();
    const bool forceIaTexcoordForOutlier = m_forceIaTexcoordForOutlier && hasIaTexcoord;

    VertexCapturePlan plan;
    uint32_t& vertexCaptureFlags = plan.flags;

    // ProvenIa reads IA texcoords, CaptureInterpolant captures the proven interpolant, and LegacyTss keeps
    // the legacy capture cascade.
    uint32_t capturedTexcoordOutputRegister = std::numeric_limits<uint32_t>::max();
    switch (m_uvResolutionMode) {
    case UvResolutionMode::ProvenIa:
      if (!hasIaTexcoord) {
        // proven IA set is missing from this draw's declaration (degenerate);
        // the captured interpolant is still exact
        capturedTexcoordOutputRegister = FindVsTexcoordOutputRegister(vertexShader, m_texcoordIndex);
      }
      break;
    case UvResolutionMode::CaptureInterpolant:
      capturedTexcoordOutputRegister = FindVsTexcoordOutputRegister(vertexShader, m_texcoordIndex);
      if (capturedTexcoordOutputRegister == std::numeric_limits<uint32_t>::max()) {
        capturedTexcoordOutputRegister = FindUniqueVsTexcoordOutputRegister(vertexShader);
      }
      break;
    case UvResolutionMode::LegacyTss:
    default:
      capturedTexcoordOutputRegister = FindVsTexcoordOutputRegister(vertexShader, m_texcoordIndex);
      if (capturedTexcoordOutputRegister == std::numeric_limits<uint32_t>::max()) {
        if (!m_frameOptions.ue3EngineMode) {
          capturedTexcoordOutputRegister = FindVsTexcoordOutputRegisterByRegNumber(vertexShader, m_texcoordIndex);
        }
      }
      if (capturedTexcoordOutputRegister == std::numeric_limits<uint32_t>::max()) {
        capturedTexcoordOutputRegister = FindUniqueVsTexcoordOutputRegister(vertexShader);
      }
      break;
    }
    if (forceIaTexcoordForOutlier) {
      capturedTexcoordOutputRegister = std::numeric_limits<uint32_t>::max();
    }

    if (useVertexCapturedTexcoords()
        && BoundShaderHasAnyUsageIndex(vertexShader, DxsoUsage::Texcoord, false)
        && capturedTexcoordOutputRegister == std::numeric_limits<uint32_t>::max()) {
      capturedTexcoordOutputRegister = FindUniqueVsTexcoordOutputRegister(vertexShader);
    }

    // Valid IA texcoords are only replaced by the VS output when opted in (useVertexCapturedTexcoords), as
    // what a VS writes to TEXCOORD is not always a UV. CaptureInterpolant is exempt, being proven.
    const bool captureVsTexcoords =
      BoundShaderHasAnyUsageIndex(vertexShader, DxsoUsage::Texcoord, false)
      && capturedTexcoordOutputRegister != std::numeric_limits<uint32_t>::max()
      && (m_uvResolutionMode == UvResolutionMode::CaptureInterpolant
          || useVertexCapturedTexcoords()
          || !geoData.texcoordBuffer.defined()
          || !RtxGeometryUtils::isTexcoordFormatValid(geoData.texcoordBuffer.vertexFormat()));

    plan.captureTexcoords = captureVsTexcoords;
    plan.texcoordOutputRegister = capturedTexcoordOutputRegister;

    // Captured positions are post-skinning, so a skinned draw takes its normal from the VS NORMAL output,
    // then a COLOR0-encoded normal, then bone-skinned IA normals, and the bind-pose IA normal last.
    const bool vsOutputsNormal = BoundShaderHas(vertexShader, DxsoUsage::Normal, false);
    const bool vsHasNormalInput = BoundShaderHas(vertexShader, DxsoUsage::Normal, true);
    const bool normalFromDecl = geoData.normalBuffer.defined();
    const bool isGpuSkinned = geoData.blendWeightBuffer.defined() || geoData.blendIndicesBuffer.defined();

    const bool vsOutputsColor0 = BoundShaderHas(vertexShader, DxsoUsage::Color, false);
    const bool hasBlendWeights = geoData.blendWeightBuffer.defined();
    const bool hasBlendIndices = geoData.blendIndicesBuffer.defined();
    const bool hasAnyBlendStream = hasBlendWeights || hasBlendIndices;
    const Ue3VsShaderCtabInfo* ue3CtabInfo = m_currentUe3CtabInfo.has_value() ? &(*m_currentUe3CtabInfo) : nullptr;
    const bool hasBoneMatricesInCtab = ue3CtabInfo != nullptr &&
                                       ue3CtabInfo->hasBoneMatrices &&
                                       ue3CtabInfo->boneMatricesRegisterCount >= 3;
    const bool canUseBoneSkinnedNormalCapture =
      isGpuSkinned &&
      !vsOutputsNormal &&
      !vsOutputsColor0 &&
      vsHasNormalInput &&
      normalFromDecl &&
      hasAnyBlendStream &&
      hasBoneMatricesInCtab;

    if (Logger::logLevel() <= LogLevel::Debug) {
      const uint32_t normalDiagKey =
        (vsOutputsNormal ? 1u : 0u) | (vsHasNormalInput ? 2u : 0u) |
        (normalFromDecl ? 4u : 0u) | (isGpuSkinned ? 8u : 0u) |
        (vsOutputsColor0 ? 16u : 0u) |
        (hasBlendWeights ? 32u : 0u) |
        (hasBlendIndices ? 64u : 0u) |
        (hasBoneMatricesInCtab ? 128u : 0u) |
        (canUseBoneSkinnedNormalCapture ? 256u : 0u) |
        (normalFromDecl ? (uint32_t(geoData.normalBuffer.vertexFormat()) << 8) : 0u);
      static fast_unordered_set s_loggedNormalDiag;
      if (s_loggedNormalDiag.insert(normalDiagKey).second) {
        const char* caseLabel = "Case3-keepIA";
        if (vsOutputsNormal) caseLabel = "Case1-vsOutputNormal";
        else if (isGpuSkinned && vsOutputsColor0) caseLabel = "Case2a-COLOR0skinned";
        else if (canUseBoneSkinnedNormalCapture) caseLabel = "Case2b-boneSkinCapture";
        else if (isGpuSkinned && normalFromDecl) caseLabel = "Case2c-bindPoseFallback";
        else if (isGpuSkinned) caseLabel = "Case2d-noNormals";

        Logger::debug(str::format(
          "[RTX-Compatibility] Vertex capture normal [", caseLabel, "]: vsOutputsNormal=", vsOutputsNormal,
          ", vsHasNormalInput=", vsHasNormalInput,
          ", normalFromDecl=", normalFromDecl,
          normalFromDecl ? str::format(", declFmt=", geoData.normalBuffer.vertexFormat()).c_str() : "",
          ", isGpuSkinned=", isGpuSkinned,
          ", vsOutputsColor0=", vsOutputsColor0,
          ", hasBlendWeights=", hasBlendWeights,
          ", hasBlendIndices=", hasBlendIndices,
          ", hasBoneMatricesInCtab=", hasBoneMatricesInCtab,
          ", vertexCount=", geoData.vertexCount));
      }
    }

    
    switch (positionSource) {
    case Ue3CapturePositionSource::InputAssembler:
      vertexCaptureFlags |= kVertexCaptureFlag_PositionFromInput;
      break;
    case Ue3CapturePositionSource::PreProjectionRegister:
      vertexCaptureFlags |= kVertexCaptureFlag_PositionFromPreProjection;
      break;
    case Ue3CapturePositionSource::ClipReconstruction:
      break;
    }

    if (vsOutputsNormal && (m_frameOptions.useVertexCapturedNormals || m_frameOptions.ue3EngineMode)) {
      // 1: VS outputs NORMAL - use vertex-captured normals (they match the captured positions)
      plan.captureNormals = true;
    } else if (isGpuSkinned && vsOutputsColor0) {
      // 2: GPU-skinned mesh, VS doesn't output NORMAL but has COLOR0 output
      // UE3 GpuSkinVertexFactory outputs the bone-transformed world-space tangent basis normal
      // through COLOR0 as (normal * 0.5 + 0.5), so we tell the vertex capture shader to decode COLOR0
      // as the normal source instead of using the bind-pose IA NORMAL
      vertexCaptureFlags |= kVertexCaptureFlag_NormalFromColor0;
      plan.captureNormals = true;
      ONCE(Logger::info("[RTX-Compatibility] UE3 GPU-skinned mesh: capturing normal from VS COLOR0 output (skinned tangent basis)."));
    } else if (canUseBoneSkinnedNormalCapture) {
      // 2b: GPU-skinned mesh without VS NORMAL/COLOR0 output
      // reconstruct skinned normals in the vertex capture shader using BoneMatrices from VS constants
      vertexCaptureFlags |= kVertexCaptureFlag_NormalBoneSkinning;
      if (geoData.normalBuffer.vertexFormat() == VK_FORMAT_R8G8B8A8_UNORM) {
        vertexCaptureFlags |= kVertexCaptureFlag_NormalInputEncodedUByte4;
      }
      if (hasBlendIndices && geoData.blendIndicesBuffer.vertexFormat() == VK_FORMAT_R8G8B8A8_UNORM) {
        vertexCaptureFlags |= kVertexCaptureFlag_BlendIndicesInputNormalized;
      }
      if (hasBlendWeights && geoData.blendWeightBuffer.vertexFormat() == VK_FORMAT_R8G8B8A8_USCALED) {
        vertexCaptureFlags |= kVertexCaptureFlag_BlendWeightsInputUnnormalized;
      }

      plan.captureNormals = true;
      ONCE(Logger::info(str::format(
        "[RTX-Compatibility] UE3 GPU-skinned mesh: reconstructing normals via BoneMatrices in vertex capture shader (boneBaseReg=c",
        ue3CtabInfo->boneMatricesRegisterIndex, ", boneRegCount=", ue3CtabInfo->boneMatricesRegisterCount,
        ", hasBlendWeights=", hasBlendWeights, ", hasBlendIndices=", hasBlendIndices, ").")));
    } else if (isGpuSkinned && normalFromDecl) {
      // 2c: GPU-skinned but no COLOR0 output and no usable bone CTAB metadata
      // keep bind-pose IA normals as smooth fallback, they're in the wrong orientation for animated poses (missing bone transform), but provide smooth per-vertex interpolation
      ONCE(Logger::info("[RTX-Compatibility] UE3 GPU-skinned mesh without COLOR0 output: using bind-pose IA normals as smooth fallback."));
    }
    // 3 : non-skinned, VS doesn't output NORMAL, just keep IA normals from processVertices they're in the same object space as the captured positions anyway..

    // Check if we should/can get colors. UE3 passes its particle colour through the vertex shader
    // as a TEXCOORD interpolant (ParticleSpriteVertexFactory.usf), which takes the place of COLOR0.
    const uint32_t capturedColorOutputRegister = m_ue3ParticleColorTexcoordIndex != UINT32_MAX
      ? FindVsTexcoordOutputRegister(vertexShader, m_ue3ParticleColorTexcoordIndex)
      : std::numeric_limits<uint32_t>::max();
    if (capturedColorOutputRegister != std::numeric_limits<uint32_t>::max() ||
        (BoundShaderHas(vertexShader, DxsoUsage::Color, false) && d3d9State().pixelShader.ptr() == nullptr)) {
      plan.captureColor = true;
    }
    if (capturedColorOutputRegister != std::numeric_limits<uint32_t>::max()) {
      vertexCaptureFlags |= m_ue3ParticleColorCaptureFlags;
    }
    if (m_ue3ParticleColorTexcoordIndex != UINT32_MAX && m_frameOptions.ue3LogMaterialFades && vertexShader != nullptr) {
      static fast_unordered_set s_loggedColorCaptures;
      const XXH64_hash_t vsHash = vertexShader->GetBytecodeHash();
      if (s_loggedColorCaptures.insert(XXH3_64bits_withSeed(&m_ue3ParticleColorTexcoordIndex, sizeof(uint32_t), vsHash)).second) {
        Logger::info(str::format(
          "[RTX-Compatibility][UE3-Fade] Capture: vs=0x", std::hex, vsHash, std::dec, " particle colour TEXCOORD",
          m_ue3ParticleColorTexcoordIndex,
          capturedColorOutputRegister != std::numeric_limits<uint32_t>::max()
            ? str::format(" from o", capturedColorOutputRegister)
            : std::string(" is not a vertex shader output, so the particle colour stays white")));
      }
    }

    plan.colorOutputRegister = capturedColorOutputRegister;
    if ((vertexCaptureFlags & kVertexCaptureFlag_NormalBoneSkinning) != 0 && ue3CtabInfo != nullptr) {
      plan.boneMatricesBaseReg = ue3CtabInfo->boneMatricesRegisterIndex;
      plan.boneCount = std::min(ue3CtabInfo->boneMatricesRegisterCount / 3u, 256u);
    }
    return plan;
  }

}
