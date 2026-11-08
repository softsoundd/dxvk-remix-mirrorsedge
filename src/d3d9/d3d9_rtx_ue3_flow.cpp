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

// Decides on the application thread which draws carry a flow layer (rtx.d3d9.ue3FlowLayers): the reads that move a
// material's normal detail, where they sit relative to the surface's texture coordinates, and how fast they move.
// See "Flow layers" in UE3Compatibility.md.

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <mutex>

#include "d3d9_rtx.h"
#include "d3d9_rtx_ue3_helpers.h"
#include "d3d9_device.h"
#include "d3d9_sampler.h"
#include "d3d9_texture.h"
#include "../dxvk/dxvk_scoped_annotation.h"
#include "../util/util_string.h"

namespace dxvk {

  namespace {

    constexpr char kUe3FlowSpeedCachePath[] = "rtx-remix/ue3FlowLayers.cache";
    constexpr char kUe3FlowSpeedCacheTempPath[] = "rtx-remix/ue3FlowLayers.cache.tmp";
    constexpr uint64_t kUe3FlowSpeedCacheMagic = 0x31574F4C46334555ull; // "UE3FLOW1"
    constexpr uint32_t kUe3FlowSpeedCacheMaxEntries = 1u << 16;
    constexpr uint32_t kUe3FlowSpeedCacheSaveIntervalFrames = 600;

    // A speed is measured over this many frames in which the read moved, and settles once the two halves of the window
    // agree this closely. Frame pacing averages out over the window, while slow motion or a pause partway through shows
    // as a disagreement and waits for the next window.
    constexpr size_t kUe3FlowWindowFrames = 32;
    constexpr double kUe3FlowWindowAgreement = 0.05;
    // A read that has not moved for this long is static.
    constexpr double kUe3FlowStillSeconds = 3.0;
    // Frame steps outside this range are a hitch or a stall, which say nothing about the speed.
    constexpr double kUe3FlowMinStepSeconds = 1.0e-3;
    constexpr double kUe3FlowMaxStepSeconds = 0.25;
    // Settled speeds are rounded to this, so a speed measured twice gives one surface material.
    constexpr double kUe3FlowSpeedQuantum = 1.0 / 8192.0;
    // A read's tiling repeats from draw to draw unless something rotates or scales it over time.
    constexpr float kUe3FlowLinearTolerance = 1.0e-3f;

    using Affine = UvAffine;

    bool linearMatches(const std::array<float, 4>& a, const Affine& b) {
      const float scale = std::max({ std::abs(a[0]), std::abs(a[1]), std::abs(a[2]), std::abs(a[3]), 1.0e-6f });
      for (uint32_t i = 0; i < 4; i++) {
        if (std::abs(a[i] - b[i]) > kUe3FlowLinearTolerance * scale) {
          return false;
        }
      }
      return true;
    }

    float quantizeSpeed(const double speed) {
      return float(std::round(speed / kUe3FlowSpeedQuantum) * kUe3FlowSpeedQuantum);
    }

    // Materials that compile to one pixel shader can pan at different speeds, so their layers are tracked apart.
    XXH64_hash_t flowStateKey(const XXH64_hash_t psHash, const XXH64_hash_t materialHash) {
      return XXH3_64bits_withSeed(&materialHash, sizeof(materialHash), psHash);
    }

    // A material can read one normal map several times, panned differently, so speeds are kept per read.
    XXH64_hash_t flowSpeedKey(const XXH64_hash_t stateKey, const uint32_t site) {
      return XXH3_64bits_withSeed(&site, sizeof(site), stateKey);
    }

    // Streamed mips bind another D3D9 texture under the same streaming-stable image hash, which the layer has to rebind.
    XXH64_hash_t textureVariantHash(const TextureRef& texture) {
      const DxvkImageView* view = texture.getImageView();
      if (view == nullptr) {
        return kEmptyHash;
      }
      const DxvkImageCreateInfo& info = view->image()->info();
      const std::array<uint32_t, 5> key = { info.extent.width, info.extent.height, info.mipLevels, uint32_t(info.format),
                                            uint32_t(view->info().format) };
      return XXH3_64bits_withSeed(key.data(), sizeof(key), texture.getImageHash());
    }

    std::string describeSamplerStage(const uint32_t stage, const std::map<uint32_t, std::string>& names) {
      const auto it = names.find(stage);
      return str::format("s", stage, "(", it != names.end() ? it->second.c_str() : "?", ")");
    }

    // Names a sample's sampler, and which of its reads it is when the layer reads the sampler more than once.
    std::string describeFlowSample(const DxsoUe3FlowShaderInfo& info, const uint32_t index,
                                   const std::map<uint32_t, std::string>& names) {
      const DxsoUe3FlowSample& sample = info.samples[index];
      uint32_t reads = 0;
      uint32_t earlierReads = 0;
      for (const DxsoUe3FlowSample& other : info.samples) {
        if (other.sampler == sample.sampler) {
          reads++;
          earlierReads += other.site < sample.site ? 1 : 0;
        }
      }
      const std::string stage = describeSamplerStage(sample.sampler, names);
      return reads > 1 ? str::format(stage, " read ", earlierReads + 1) : stage;
    }

  }

  void D3D9Rtx::publishUe3FlowLayerStatus(const Ue3FlowLayerStatus& status) {
    std::lock_guard<std::mutex> lock(s_ue3FlowStatusMutex);
    s_ue3FlowStatus[status.materialHash] = status;
  }

  const Vector2* Ue3FlowSpeedCache::find(const XXH64_hash_t key) {
    if (!m_loaded) {
      load();
    }
    const auto it = m_speeds.find(key);
    return it != m_speeds.end() ? &it->second : nullptr;
  }

  void Ue3FlowSpeedCache::store(const XXH64_hash_t key, const Vector2& velocity) {
    if (!m_loaded) {
      load();
    }
    m_speeds[key] = velocity;
    m_dirty = true;
  }

  void Ue3FlowSpeedCache::load() {
    m_loaded = true;
    const bool fileExists = ue3CacheFileExists(kUe3FlowSpeedCachePath);
    std::ifstream file(kUe3FlowSpeedCachePath, std::ios::binary);
    if (!file.is_open()) {
      if (fileExists) {
        m_saveBlocked = true;
        Logger::warn(str::format("[RTX-Compatibility][UE3-Flow] ", kUe3FlowSpeedCachePath,
                                 " exists but could not be opened. Flow speeds are measured again this session."));
      }
      return;
    }
    uint64_t magic = 0;
    uint32_t entryCount = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&entryCount), sizeof(entryCount));
    if (!file || magic != kUe3FlowSpeedCacheMagic || entryCount > kUe3FlowSpeedCacheMaxEntries) {
      m_saveBlocked = true;
      Logger::warn(str::format("[RTX-Compatibility][UE3-Flow] ", kUe3FlowSpeedCachePath,
                               " is not recognised and is left untouched. Delete it to start a fresh one."));
      return;
    }
    for (uint32_t i = 0; i < entryCount; i++) {
      XXH64_hash_t key = 0;
      Vector2 velocity;
      file.read(reinterpret_cast<char*>(&key), sizeof(key));
      file.read(reinterpret_cast<char*>(&velocity.x), sizeof(velocity.x));
      file.read(reinterpret_cast<char*>(&velocity.y), sizeof(velocity.y));
      if (!file) {
        m_saveBlocked = true;
        Logger::warn(str::format("[RTX-Compatibility][UE3-Flow] ", kUe3FlowSpeedCachePath, " is truncated: ", i, " of ",
                                 entryCount, " entries readable. It is left untouched for this session."));
        return;
      }
      m_speeds[key] = velocity;
    }
    Logger::info(str::format("[RTX-Compatibility][UE3-Flow] Loaded flow speed cache: ", entryCount, " speeds"));
  }

  void Ue3FlowSpeedCache::save() {
    if (!m_dirty || m_saveBlocked) {
      return;
    }
    {
      std::ofstream file(kUe3FlowSpeedCacheTempPath, std::ios::binary | std::ios::trunc);
      if (!file.is_open()) {
        return;
      }
      const uint32_t entryCount = uint32_t(std::min<size_t>(m_speeds.size(), kUe3FlowSpeedCacheMaxEntries));
      file.write(reinterpret_cast<const char*>(&kUe3FlowSpeedCacheMagic), sizeof(kUe3FlowSpeedCacheMagic));
      file.write(reinterpret_cast<const char*>(&entryCount), sizeof(entryCount));
      uint32_t written = 0;
      for (const auto& [key, velocity] : m_speeds) {
        if (written++ >= entryCount) {
          break;
        }
        file.write(reinterpret_cast<const char*>(&key), sizeof(key));
        file.write(reinterpret_cast<const char*>(&velocity.x), sizeof(velocity.x));
        file.write(reinterpret_cast<const char*>(&velocity.y), sizeof(velocity.y));
      }
      file.close();
      if (!file) {
        return;
      }
    }
    if (ue3CommitCacheFile(kUe3FlowSpeedCacheTempPath, kUe3FlowSpeedCachePath)) {
      m_dirty = false;
    }
  }

  void Ue3FlowSpeedCache::saveIfDue(const uint64_t frameId) {
    if (m_dirty && frameId >= m_lastSaveFrame + kUe3FlowSpeedCacheSaveIntervalFrames) {
      m_lastSaveFrame = uint32_t(frameId);
      save();
    }
  }

  Rc<DxvkSampler> D3D9Rtx::getOrCreateStageSampler(const uint32_t stage) {
    const D3D9SamplerKey key = m_parent->CreateSamplerKey(stage);
    const XXH64_hash_t samplerHash = D3D9SamplerKeyHash {}(key);
    const auto it = m_samplerCache.find(samplerHash);
    if (it != m_samplerCache.end()) {
      return it->second;
    }
    Rc<DxvkSampler> sampler = m_parent->GetDXVKDevice()->createSampler(m_parent->DecodeSamplerKey(key));
    m_samplerCache.insert(std::make_pair(samplerHash, sampler));
    return sampler;
  }

  const DxsoUe3FlowShaderInfo& D3D9Rtx::getOrAnalyzeUe3FlowShader(const XXH64_hash_t psHash, const D3D9CommonShader* pixelShader) {
    auto it = m_ue3FlowShaders.find(psHash);
    if (it != m_ue3FlowShaders.end()) {
      return it->second;
    }
    it = m_ue3FlowShaders.emplace(psHash, analyzeDxsoUe3FlowLayers(makeDxsoShaderView(pixelShader->GetBytecode(), pixelShader))).first;
    const DxsoUe3FlowShaderInfo& info = it->second;
    if (m_frameOptions.ue3LogFlowLayers && info.refusal != DxsoUe3FlowRefusal::NotAnalyzed &&
        info.refusal != DxsoUe3FlowRefusal::NoPannedSampler) {
      if (info.refusal != DxsoUe3FlowRefusal::None) {
        Logger::info(str::format("[RTX-Compatibility][UE3-Flow] ps=0x", std::hex, psHash, std::dec,
                                 " pans a texture but has no flow layer: ", describeDxsoUe3FlowRefusal(info.refusal),
                                 info.refusalDetail.empty() ? "" : " ", info.refusalDetail));
      } else {
        std::string samples;
        for (const DxsoUe3FlowSample& sample : info.samples) {
          samples += str::format(samples.empty() ? "" : ", ", "s", uint32_t(sample.sampler), "(", sample.name, ") at U=[",
                                 formatUvComponentAffine(sample.uv.affineU), "] V=[", formatUvComponentAffine(sample.uv.affineV),
                                 "] weight ", sample.weight);
        }
        std::string offsets;
        for (const uint32_t reg : info.offsetRegisters) {
          offsets += str::format(offsets.empty() ? "c" : " c", reg);
        }
        const std::string blend = info.blend.present
          ? str::format(", blended from the first towards the second by ", info.blend.scale, " * s", uint32_t(info.blend.sampler),
                        "(", info.blend.name, ").", "xyzw"[info.blend.channel & 3u], " at U=[",
                        formatUvComponentAffine(info.blend.uv.affineU), "] V=[", formatUvComponentAffine(info.blend.uv.affineV), "]")
          : std::string();
        Logger::info(str::format(
          "[RTX-Compatibility][UE3-Flow] ps=0x", std::hex, psHash, std::dec, " has a flow layer: samples [", samples, "]", blend,
          " offset by [", offsets, "], ",
          info.mask.present
            ? str::format("mask s", uint32_t(info.mask.sampler), "(", info.mask.name, ").", "xyzw"[info.mask.channel & 3u],
                          " as ", info.mask.bias, " + ", info.mask.scale, " * texel")
            : std::string("no mask"),
          info.ignoresOtherFactors ? ", leaving out other textures that also weigh the samples" : ""));
      }
    }
    return info;
  }

  bool D3D9Rtx::isUe3FlowLayerExcluded(const LegacyMaterialData& material) const {
    const fast_unordered_set& excluded = *m_frameOptions.ue3FlowLayerExcludedTextures;
    return !excluded.empty() &&
           (lookupHash(excluded, material.getHash()) || lookupHash(excluded, material.getTextureSetAndShaderHash()) ||
            lookupHash(excluded, material.getColorTexture().getImageHash()));
  }

  const DxsoUe3FlowShaderInfo* D3D9Rtx::getUe3FlowShaderForBake(const Ue3TextureState& ue3) {
    if (!m_frameOptions.ue3FlowLayers || ue3.inferredPs == nullptr || isUe3FlowLayerExcluded(m_activeDrawCallState.materialData)) {
      return nullptr;
    }
    const DxsoUe3FlowShaderInfo& info = getOrAnalyzeUe3FlowShader(ue3.inferredPsHash, ue3.inferredPs);
    if (info.refusal != DxsoUe3FlowRefusal::None || info.samples.empty()) {
      return nullptr;
    }
    // A layer whose reads never move is not rendered, so the bake keeps them as the game draws them.
    const auto state = m_ue3FlowMaterials.find(flowStateKey(ue3.inferredPsHash, m_activeDrawCallState.materialData.getHash()));
    if (state != m_ue3FlowMaterials.end() && (state->second.refused || state->second.still)) {
      return nullptr;
    }
    return &info;
  }

  void D3D9Rtx::resolveUe3FlowLayers(const Ue3TextureState& ue3, const Vector2* compositePeriod) {
    LegacyMaterialData& material = m_activeDrawCallState.materialData;
    material.ue3FlowLayer = Ue3FlowLayer {};
    if (ue3.inferredPs == nullptr || ue3.inferredPsEntry == nullptr) {
      return;
    }
    const XXH64_hash_t materialHash = material.getHash();
    if (materialHash == kEmptyHash) {
      return;
    }

    const DxsoUe3FlowShaderInfo& info = getOrAnalyzeUe3FlowShader(ue3.inferredPsHash, ue3.inferredPs);
    if (info.refusal != DxsoUe3FlowRefusal::None || info.samples.empty() || (info.blend.present && info.samples.size() != 2)) {
      return;
    }

    ScopedCpuProfileZoneN("UE3 flow layer");
    const XXH64_hash_t pickHash = material.getColorTexture().getImageHash();
    if (isUe3FlowLayerExcluded(material)) {
      return;
    }

    const std::map<uint32_t, std::string>& samplerNames = getUe3PsSamplerNames(ue3.inferredPsHash, ue3.inferredPs->GetBytecode());
    const XXH64_hash_t stateKey = flowStateKey(ue3.inferredPsHash, materialHash);
    Ue3FlowMaterialState& state = m_ue3FlowMaterials[stateKey];

    // Lists the reads the layer renders, which are its samples, or with a blend the normal map it blends towards and the
    // blend's weight. The normal map a blend starts from stands still, and the surface's own normal stands in for it.
    struct LayerRead {
      const PsSamplerUvOrigin* uv = nullptr;
      uint32_t site = 0;
      uint32_t sampler = 0;
      float weight = 1.0f;
    };
    std::array<LayerRead, Ue3FlowLayer::kMaxSamples> reads;
    uint32_t readCount = 0;
    if (info.blend.present) {
      reads[0] = { &info.samples[1].uv, info.samples[1].site, info.samples[1].sampler, 1.0f };
      reads[1] = { &info.blend.uv, info.blend.site, info.blend.sampler, info.blend.scale };
      readCount = 2;
    } else {
      readCount = std::min<uint32_t>(uint32_t(info.samples.size()), Ue3FlowLayer::kMaxSamples);
      for (uint32_t i = 0; i < readCount; i++) {
        const DxsoUe3FlowSample& sample = info.samples[i];
        reads[i] = { &sample.uv, sample.site, sample.sampler, sample.weight };
      }
    }
    auto readName = [&](const uint32_t i) {
      return info.blend.present && i == 1
        ? str::format("the blend's weight ", describeSamplerStage(info.blend.sampler, samplerNames))
        : describeFlowSample(info, info.blend.present ? 1 : i, samplerNames);
    };

    Ue3FlowLayerStatus status;
    status.materialHash = materialHash;
    status.pickHash = pickHash;
    status.sampleCount = readCount;
    status.masked = info.mask.present;
    status.blend = info.blend.present;
    for (uint32_t i = 0; i < readCount; i++) {
      status.weights[i] = reads[i].weight;
    }

    // Logged and published once per material and reason.
    auto refuse = [&](const std::string& reason) {
      status.state = reason;
      const XXH64_hash_t key = XXH3_64bits_withSeed(reason.data(), reason.size(), materialHash);
      static fast_unordered_set s_loggedRefusals;
      if (s_loggedRefusals.insert(key).second) {
        publishUe3FlowLayerStatus(status);
        if (m_frameOptions.ue3LogFlowLayers) {
          Logger::warn(str::format("[RTX-Compatibility][UE3-Flow] materialHash=0x", std::hex, materialHash, " pick=0x", pickHash,
                                   " ps=0x", ue3.inferredPsHash, std::dec, " has a flow layer that is not rendered: ", reason, "."));
        }
      }
    };

    // Where each read's coordinates come from: the surface's interpolant, or an IA texcoord set the vertex shader passes
    // through, which the geometry then carries as its second UV set. This is decided from the shaders alone, ahead of
    // anything that changes from draw to draw, so a geometry carries the same vertex data from its first draw on.
    auto readsSurfaceInterpolant = [&](const PsSamplerUvOrigin& origin) {
      return origin.originValid && origin.sitesAgree && origin.affineExact && origin.semanticIndex == m_ue3UvFacts.semanticIndex &&
             origin.compU == m_ue3UvFacts.compU && origin.compV == m_ue3UvFacts.compV;
    };
    const D3D9CommonShader* vs = m_parent->UseProgrammableVS() && d3d9State().vertexShader.ptr() != nullptr
      ? d3d9State().vertexShader->GetCommonShader()
      : nullptr;
    uint32_t secondarySet = UINT32_MAX;
    std::string uvRefusal;
    // Returns the trace of a read on the second UV set, or null for one on the surface's.
    auto traceUvSet = [&](const PsSamplerUvOrigin& origin, const auto& name) -> const Ue3VsTexcoordTraceEntry* {
      if (readsSurfaceInterpolant(origin)) {
        return nullptr;
      }
      const Ue3VsTexcoordTraceEntry* trace = vs != nullptr && origin.originValid && origin.affineExact
        ? getOrTraceUe3VsTexcoord(vs, findVsTexcoordOutputRegister(vs->GetOsgn(), origin.semanticIndex), origin.compU, origin.compV)
        : nullptr;
      if (trace == nullptr || (trace->kind != Ue3VsUvTraceKind::PureMove && trace->kind != Ue3VsUvTraceKind::AffineConst)) {
        if (uvRefusal.empty()) {
          uvRefusal = str::format(name(), " reads another UV set than the surface, and not one the vertex shader passes through");
        }
        return nullptr;
      }
      if (secondarySet != UINT32_MAX && secondarySet != trace->iaTexcoordIndex) {
        if (uvRefusal.empty()) {
          uvRefusal = str::format(name(), " reads a third UV set");
        }
        return nullptr;
      }
      secondarySet = trace->iaTexcoordIndex;
      return trace;
    };
    std::array<const Ue3VsTexcoordTraceEntry*, Ue3FlowLayer::kMaxSamples> readTraces = {};
    for (uint32_t i = 0; i < readCount; i++) {
      readTraces[i] = traceUvSet(*reads[i].uv, [&]() { return readName(i); });
    }
    auto maskName = [&]() { return str::format("the mask ", describeSamplerStage(info.mask.sampler, samplerNames)); };
    const Ue3VsTexcoordTraceEntry* maskTrace = info.mask.present ? traceUvSet(info.mask.uv, maskName) : nullptr;
    if (!uvRefusal.empty()) {
      refuse(uvRefusal);
      return;
    }
    m_ue3FlowSecondaryTexcoordSet = secondarySet;
    status.secondaryUv = secondarySet != UINT32_MAX;

    if (state.refused) {
      refuse(state.refusal);
      return;
    }

    // A read on the surface's interpolant has its own affine of it, so its coordinate is one of the surface's coordinates
    // too. A bake's coordinates are the interpolant over the bake's repeat.
    bool readsSurface = info.mask.present && maskTrace == nullptr;
    for (uint32_t i = 0; i < readCount; i++) {
      readsSurface |= readTraces[i] == nullptr;
    }
    Affine surfaceInverse = { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f };
    if (readsSurface) {
      Affine surface;
      if (compositePeriod != nullptr) {
        surface = { 1.0f / compositePeriod->x, 0.0f, 0.0f, 1.0f / compositePeriod->y, 0.0f, 0.0f };
      } else if (m_ue3UvFacts.surfaceAffineResolved) {
        surface = m_ue3UvFacts.psAffine;
      } else {
        refuse("the surface's texture coordinates are not resolved from the shader");
        return;
      }
      if (!invertUvAffine(surface, surfaceInverse)) {
        refuse("the surface's texture transform is degenerate");
        return;
      }
    }

    // Resolves a sampler's coordinate as an affine of the interpolant. With staticOffsets, the constants its offset reads are
    // left out: they are the panners, whose motion the velocity carries.
    auto resolveSamplerAffine = [&](const PsSamplerUvOrigin& origin, const bool staticOffsets, Affine& out) -> bool {
      auto resolveRow = [&](const UvComponentAffine& row, const bool rowIsU, float& coeffU, float& coeffV, float& translation) -> bool {
        float scale = 1.0f;
        float cross = 0.0f;
        if (!resolveUe3PsAffineTerm(row.scale, 1.0f, scale) || (row.hasCross && !resolveUe3PsAffineTerm(row.cross, 0.0f, cross))) {
          return false;
        }
        if (staticOffsets) {
          translation = row.offset.immValid ? row.offset.imm : 0.0f;
        } else if (!resolveUe3PsAffineTerm(row.offset, 0.0f, translation)) {
          return false;
        }
        coeffU = 0.0f;
        coeffV = 0.0f;
        if (!row.hasCross) {
          (rowIsU ? coeffU : coeffV) = scale;
          return true;
        }
        auto accumulate = [&](const uint8_t component, const float coefficient) {
          if (component == origin.compU) {
            coeffU += coefficient;
            return true;
          }
          if (component == origin.compV) {
            coeffV += coefficient;
            return true;
          }
          return false;
        };
        return accumulate(row.scaleComponent, scale) && accumulate(row.crossComponent, cross);
      };
      return resolveRow(origin.affineU, true, out[0], out[1], out[4]) && resolveRow(origin.affineV, false, out[2], out[3], out[5]) &&
             std::all_of(out.begin(), out.end(), [](const float v) { return std::isfinite(v); });
    };

    // Maps a read's interpolant affine onto the surface's texture coordinates, or onto the second UV set through the vertex
    // shader's affine of it.
    auto placeRead = [&](const Affine& readAffine, const Ue3VsTexcoordTraceEntry* trace, Affine& out) -> bool {
      if (trace == nullptr) {
        out = composeUvAffine(readAffine, surfaceInverse);
        return true;
      }
      Affine vertexAffine = { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f };
      if (trace->kind == Ue3VsUvTraceKind::AffineConst &&
          !(resolveUe3VsAffineTerm(trace->affineU.scale, 1.0f, vertexAffine[0]) &&
            resolveUe3VsAffineTerm(trace->affineU.offset, 0.0f, vertexAffine[4]) &&
            resolveUe3VsAffineTerm(trace->affineV.scale, 1.0f, vertexAffine[3]) &&
            resolveUe3VsAffineTerm(trace->affineV.offset, 0.0f, vertexAffine[5]))) {
        return false;
      }
      out = composeUvAffine(readAffine, vertexAffine);
      return std::all_of(out.begin(), out.end(), [](const float v) { return std::isfinite(v); });
    };

    if (!state.initialized) {
      state.initialized = true;
      for (uint32_t i = 0; i < readCount; i++) {
        Ue3FlowMaterialState::SampleTrack& track = state.samples[i];
        const PsSamplerUvOrigin& origin = *reads[i].uv;
        auto addContributions = [&](const UvAffineTerm& term, const uint8_t axis) {
          if (term.constReg >= 0 && uint32_t(term.constReg) < caps::MaxFloatConstantsPS) {
            track.contributions.push_back({ uint16_t(term.constReg), uint8_t(term.constComp & 3u), axis, term.factor, 0.0f });
          }
          if (term.constReg2 >= 0 && uint32_t(term.constReg2) < caps::MaxFloatConstantsPS) {
            track.contributions.push_back({ uint16_t(term.constReg2), uint8_t(term.constComp2 & 3u), axis, term.factor2, 0.0f });
          }
        };
        addContributions(origin.affineU.offset, 0);
        addContributions(origin.affineV.offset, 1);
        for (Ue3FlowMaterialState::SampleTrack::Contribution& contribution : track.contributions) {
          contribution.lastValue = d3d9State().psConsts.fConsts[contribution.reg][contribution.component];
        }
        // A read whose offset reads no constant, such as the normal map a blend leans towards, cannot pan.
        if (track.contributions.empty()) {
          track.latched = true;
        } else if (const Vector2* cached = m_ue3FlowSpeedCache.find(flowSpeedKey(stateKey, reads[i].site))) {
          track.latched = true;
          track.velocity = *cached;
        }
      }
      state.lastObservedFrame = m_ue3FrameCounter;
      state.lastObservedTime = std::chrono::steady_clock::now();
    } else if (state.lastObservedFrame != m_ue3FrameCounter) {
      // UE3 uploads a panner as frac(time * speed), so a frame's step is the difference wrapped into [-0.5, 0.5].
      const auto now = std::chrono::steady_clock::now();
      const double step = std::chrono::duration<double>(now - state.lastObservedTime).count();
      const bool measurable = m_ue3FrameCounter == state.lastObservedFrame + 1 && step >= kUe3FlowMinStepSeconds &&
                              step <= kUe3FlowMaxStepSeconds;
      for (uint32_t i = 0; i < readCount; i++) {
        Ue3FlowMaterialState::SampleTrack& track = state.samples[i];
        Vector2 displacement(0.0f);
        for (Ue3FlowMaterialState::SampleTrack::Contribution& contribution : track.contributions) {
          const float value = d3d9State().psConsts.fConsts[contribution.reg][contribution.component];
          float delta = value - contribution.lastValue;
          delta -= std::round(delta);
          displacement[contribution.axis] += delta * contribution.factor;
          contribution.lastValue = value;
        }
        if (track.latched || !measurable) {
          continue;
        }
        if (std::abs(displacement.x) + std::abs(displacement.y) <= 1.0e-7f) {
          track.stillSeconds += step;
          continue;
        }
        track.stillSeconds = 0.0;
        track.windowDisplacement.push_back(displacement);
        track.windowSeconds.push_back(step);
        if (track.windowDisplacement.size() < kUe3FlowWindowFrames) {
          continue;
        }
        const size_t half = kUe3FlowWindowFrames / 2;
        auto velocityOf = [&](const size_t begin, const size_t end) {
          Vector2 sum(0.0f);
          double seconds = 0.0;
          for (size_t f = begin; f < end; f++) {
            sum += track.windowDisplacement[f];
            seconds += track.windowSeconds[f];
          }
          return Vector2(float(sum.x / seconds), float(sum.y / seconds));
        };
        const Vector2 first = velocityOf(0, half);
        const Vector2 second = velocityOf(half, kUe3FlowWindowFrames);
        const float scale = std::max(length(first), length(second));
        if (length(first - second) <= float(kUe3FlowWindowAgreement) * scale) {
          const Vector2 velocity = velocityOf(0, kUe3FlowWindowFrames);
          track.latched = true;
          track.velocity = Vector2(quantizeSpeed(velocity.x), quantizeSpeed(velocity.y));
          m_ue3FlowSpeedCache.store(flowSpeedKey(stateKey, reads[i].site), track.velocity);
          if (m_frameOptions.ue3LogFlowLayers) {
            Logger::info(str::format("[RTX-Compatibility][UE3-Flow] materialHash=0x", std::hex, materialHash, " ps=0x",
                                     ue3.inferredPsHash, std::dec, " ", readName(i), " pans at (", track.velocity.x,
                                     ", ", track.velocity.y, ") per second."));
          }
        } else {
          track.windowDisplacement.erase(track.windowDisplacement.begin(), track.windowDisplacement.begin() + half);
          track.windowSeconds.erase(track.windowSeconds.begin(), track.windowSeconds.begin() + half);
        }
      }
      state.lastObservedFrame = m_ue3FrameCounter;
      state.lastObservedTime = now;
    }

    // A read settles once it has a speed or has stood still long enough to call it static.
    bool settled = true;
    bool moving = false;
    for (uint32_t i = 0; i < readCount; i++) {
      const Ue3FlowMaterialState::SampleTrack& track = state.samples[i];
      const bool still = !track.latched && track.stillSeconds >= kUe3FlowStillSeconds;
      settled &= track.latched || still;
      moving |= track.latched && (track.velocity.x != 0.0f || track.velocity.y != 0.0f);
      status.velocities[i] = track.velocity;
    }
    if (!settled) {
      status.state = "measuring the speed";
      static fast_unordered_set s_publishedMeasuring;
      if (s_publishedMeasuring.insert(materialHash).second) {
        publishUe3FlowLayerStatus(status);
      }
      return;
    }
    if (!moving) {
      if (!state.still) {
        state.still = true;
        if (m_frameOptions.ue3LogFlowLayers) {
          Logger::info(str::format("[RTX-Compatibility][UE3-Flow] materialHash=0x", std::hex, materialHash, " ps=0x",
                                   ue3.inferredPsHash, std::dec,
                                   " never moves its flow layer's reads, which leaves its normal maps to the Toolkit as any "
                                   "normal map is."));
        }
      }
      refuse("the layer's reads stand still");
      return;
    }

    Ue3FlowLayer layer;
    layer.sampleCount = uint8_t(readCount);
    layer.blend = info.blend.present;
    for (uint32_t i = 0; i < readCount; i++) {
      const LayerRead& read = reads[i];
      Ue3FlowMaterialState::SampleTrack& track = state.samples[i];
      Affine readAffine;
      Affine transform;
      if (!resolveSamplerAffine(*read.uv, true, readAffine) || !placeRead(readAffine, readTraces[i], transform)) {
        refuse(str::format(readName(i), "'s coordinate does not resolve against the draw's constants"));
        return;
      }
      if (!track.hasLinear) {
        track.hasLinear = true;
        track.linear = { readAffine[0], readAffine[1], readAffine[2], readAffine[3] };
      } else if (!linearMatches(track.linear, readAffine)) {
        state.refused = true;
        state.refusal = str::format(readName(i), " changes its tiling or rotation from draw to draw, which a scrolling layer ",
                                    "cannot follow");
        refuse(state.refusal);
        return;
      }
      if (read.sampler >= SamplerCount || d3d9State().textures[read.sampler] == nullptr) {
        refuse(str::format(readName(i), " has no texture bound"));
        return;
      }
      // A blend's weight is decoded as the game samples it, as a mask is.
      const bool isWeight = info.blend.present && i == 1;
      D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[read.sampler]);
      const bool srgb = d3d9State().samplerStates[read.sampler][D3DSAMP_SRGBTEXTURE] & 0x1;
      const Rc<DxvkImageView> view = texture != nullptr && texture->GetType() == D3DRTYPE_TEXTURE
        ? getRemixSampleView(texture, srgb && !isWeight)
        : nullptr;
      if (view == nullptr) {
        refuse(str::format(readName(i), " is not a sampleable 2D texture"));
        return;
      }
      Ue3FlowLayer::Sample& out = layer.samples[i];
      out.texture = TextureRef(view);
      out.transform = transform;
      out.velocity = track.velocity;
      out.weight = read.weight;
      out.secondaryUv = readTraces[i] != nullptr;
      if (isWeight) {
        out.components = { uint8_t(info.blend.channel & 3u), Ue3FlowLayer::kNoComponent, Ue3FlowLayer::kNoComponent };
        layer.blendIsSrgb = srgb;
      } else {
        const DxsoUe3FlowSample& sample = info.samples[info.blend.present ? 1 : i];
        for (uint32_t lane = 0; lane < 3; lane++) {
          out.components[lane] = sample.components[lane] == kDxsoUe3FlowNoComponent
            ? Ue3FlowLayer::kNoComponent
            : sample.components[lane];
        }
      }
    }
    layer.sampler = getOrCreateStageSampler(reads[0].sampler);

    if (info.mask.present) {
      const uint32_t stage = info.mask.sampler;
      Affine maskAffine;
      Affine relative;
      if (!resolveSamplerAffine(info.mask.uv, false, maskAffine) || !placeRead(maskAffine, maskTrace, relative)) {
        refuse(str::format(maskName(), "'s coordinate does not resolve against the draw's constants"));
        return;
      }
      const float diagonal = std::abs(relative[0]) + std::abs(relative[3]);
      if (std::abs(relative[1]) > 1.0e-4f * diagonal || std::abs(relative[2]) > 1.0e-4f * diagonal) {
        refuse(str::format(maskName(), " is rotated against its UV set"));
        return;
      }
      D3D9CommonTexture* texture = stage < SamplerCount && d3d9State().textures[stage] != nullptr
        ? GetCommonTexture(d3d9State().textures[stage])
        : nullptr;
      const Rc<DxvkImageView> view = texture != nullptr && texture->GetType() == D3DRTYPE_TEXTURE
        ? getRemixSampleView(texture, false)
        : nullptr;
      if (view == nullptr) {
        refuse(str::format(maskName(), " has no sampleable texture bound"));
        return;
      }
      layer.mask = TextureRef(view);
      layer.maskTransform = Vector4(relative[0], relative[3], relative[4], relative[5]);
      layer.maskChannel = info.mask.channel;
      layer.maskIsSrgb = d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1;
      layer.maskSecondaryUv = maskTrace != nullptr;
      layer.maskScale = info.mask.scale;
      layer.maskBias = info.mask.bias;
    }
    layer.strength = m_frameOptions.ue3FlowLayerStrength;
    // A blend keeps the surface's own roughness.
    layer.roughness = layer.blend ? -1.0f : m_frameOptions.ue3FlowLayerRoughness;

    XXH3_state_t* const hashState = getThreadLocalXxh3State();
    if (hashState != nullptr) {
      XXH3_64bits_reset(hashState);
      for (uint32_t i = 0; i < readCount; i++) {
        const Ue3FlowLayer::Sample& sample = layer.samples[i];
        const XXH64_hash_t textureHash = textureVariantHash(sample.texture);
        XXH3_64bits_update(hashState, &textureHash, sizeof(textureHash));
        XXH3_64bits_update(hashState, sample.transform.data(), sizeof(sample.transform));
        XXH3_64bits_update(hashState, &sample.velocity, sizeof(sample.velocity));
        XXH3_64bits_update(hashState, &sample.weight, sizeof(sample.weight));
        XXH3_64bits_update(hashState, sample.components.data(), sizeof(sample.components));
        const uint8_t secondaryUv = sample.secondaryUv ? 1u : 0u;
        XXH3_64bits_update(hashState, &secondaryUv, sizeof(secondaryUv));
      }
      const XXH64_hash_t samplerHash = layer.sampler->info().calculateHash();
      const XXH64_hash_t maskHash = textureVariantHash(layer.mask);
      XXH3_64bits_update(hashState, &samplerHash, sizeof(samplerHash));
      XXH3_64bits_update(hashState, &maskHash, sizeof(maskHash));
      XXH3_64bits_update(hashState, &layer.maskTransform, sizeof(layer.maskTransform));
      const std::array<float, 4> scalars = { layer.maskScale, layer.maskBias, layer.strength, layer.roughness };
      const std::array<uint8_t, 5> flags = {
        layer.maskChannel, uint8_t(layer.maskIsSrgb ? 1u : 0u), uint8_t(layer.maskSecondaryUv ? 1u : 0u),
        uint8_t(layer.blend ? 1u : 0u), uint8_t(layer.blendIsSrgb ? 1u : 0u)
      };
      XXH3_64bits_update(hashState, scalars.data(), sizeof(scalars));
      XXH3_64bits_update(hashState, flags.data(), sizeof(flags));
      layer.identity = XXH3_64bits_digest(hashState);
    }
    material.ue3FlowLayer = std::move(layer);

    if (!state.loggedAttach) {
      state.loggedAttach = true;
      if (m_frameOptions.ue3LogFlowLayers) {
        const Ue3FlowLayer& attached = material.ue3FlowLayer;
        auto uvSetName = [&](const bool secondaryUv) {
          return secondaryUv ? str::format("UV set ", secondarySet) : std::string("the surface's coordinates");
        };
        std::string samples;
        for (uint32_t i = 0; i < readCount; i++) {
          const Ue3FlowLayer::Sample& sample = attached.samples[i];
          samples += str::format(samples.empty() ? "" : ", ", readName(i),
                                 " tex=0x", std::hex, sample.texture.getImageHash(), std::dec, " weight ", sample.weight,
                                 " at (", sample.transform[0], ",", sample.transform[1], ",", sample.transform[2], ",",
                                 sample.transform[3], ",", sample.transform[4], ",", sample.transform[5], ") of ",
                                 uvSetName(sample.secondaryUv), " scrolling (", sample.velocity.x, ", ", sample.velocity.y, ") per second");
        }
        Logger::info(str::format(
          "[RTX-Compatibility][UE3-Flow] materialHash=0x", std::hex, materialHash, " pick=0x", pickHash, " ps=0x", ue3.inferredPsHash,
          std::dec, attached.blend ? " renders a flow layer blending the surface's normal: [" : " renders a flow layer: [", samples, "]",
          attached.mask.isValid()
            ? str::format(" masked by tex=0x", std::hex, attached.mask.getImageHash(), std::dec, ".", "xyzw"[attached.maskChannel & 3u],
                          " at (", attached.maskTransform.x, ",", attached.maskTransform.y, ",", attached.maskTransform.z, ",",
                          attached.maskTransform.w, ") of ", uvSetName(attached.maskSecondaryUv))
            : std::string(" with no mask"),
          compositePeriod != nullptr ? " over the composite bake's repeat" : "", "."));
      }
    }
    static fast_unordered_set s_publishedAttached;
    if (s_publishedAttached.insert(materialHash).second) {
      publishUe3FlowLayerStatus(status);
    }
  }

  void D3D9Rtx::attachUe3FlowSecondaryTexcoords(RasterGeometry& geoData) {
    if (m_ue3FlowSecondaryTexcoordSet == UINT32_MAX) {
      return;
    }
    // UE3 draws leave color0 free, unless the vertex capture fills it with a particle's colour.
    if (m_ue3FlowSecondaryTexcoords.defined() && !geoData.color0Buffer.defined()) {
      geoData.color0Buffer = m_ue3FlowSecondaryTexcoords;
      geoData.color0IsTexcoord = true;
      return;
    }
    Ue3FlowLayer& layer = m_activeDrawCallState.materialData.ue3FlowLayer;
    if (!layer.isValid() || !layer.readsSecondaryUv()) {
      return;
    }
    layer = Ue3FlowLayer {};
    const XXH64_hash_t materialHash = m_activeDrawCallState.materialData.getHash();
    static fast_unordered_set s_loggedMissingSets;
    if (m_frameOptions.ue3LogFlowLayers && s_loggedMissingSets.insert(materialHash).second) {
      const uint32_t set = m_ue3FlowSecondaryTexcoordSet;
      Logger::warn(str::format("[RTX-Compatibility][UE3-Flow] materialHash=0x", std::hex, materialHash, std::dec,
                               " has a flow layer that is not rendered: ",
                               geoData.color0Buffer.defined()
                                 ? str::format("the draw has a vertex colour where its UV set ", set, " would go")
                                 : str::format("the draw holds no UV set ", set, " the layer can read"),
                               "."));
    }
  }

}
