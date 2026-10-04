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
    // A save rewrites the whole file from memory, so a load that read less than the file holds must block
    // saving for the session. "Absent" is told apart from "unreadable" so that a first run still saves.
    bool ue3CacheFileExists(const char* path) {
      std::error_code error;
      return std::filesystem::exists(path, error);
    }

    // Publishes a fully written temp file over the real one, so a process that dies mid-write
    // leaves the previous cache intact rather than a truncated file that the next load would
    // read as a short but well-formed one.
    bool ue3CommitCacheFile(const char* tempPath, const char* path) {
      std::error_code error;
      std::filesystem::rename(tempPath, path, error);
      if (error) {
        std::filesystem::remove(tempPath, error);
        return false;
      }
      return true;
    }

    // Both learned caches are rewritten whole, so they are saved on an interval rather than on
    // every change; the destructor flushes whatever the last interval missed.
    constexpr uint32_t kUe3CacheSaveIntervalFrames = 600;

    constexpr char kUe3TextureSpreadCachePath[] = "rtx-remix/ue3TextureSpread.cache";

    constexpr char kUe3TextureSpreadCacheTempPath[] = "rtx-remix/ue3TextureSpread.cache.tmp";

    constexpr uint64_t kUe3TextureSpreadCacheMagic = 0x3144525053334555ull; // "UE3SPRD1"

    constexpr uint32_t kUe3TextureSpreadCacheMaxEntries = 1u << 20;

    constexpr char kUe3DiffuseSelectionCachePath[] = "rtx-remix/ue3DiffuseSelection.cache";

    constexpr char kUe3DiffuseSelectionCacheTempPath[] = "rtx-remix/ue3DiffuseSelection.cache.tmp";

    constexpr uint64_t kUe3DiffuseSelectionCacheMagic = 0x324C455344334555ull; // "UE3DSEL2"

    constexpr uint32_t kUe3DiffuseSelectionCacheMaxEntries = 1u << 20;

    // Stored picks are only meaningful under the scoring that produced them. Bump this whenever
    // the albedo score or the header changes, so a build re-derives instead of serving decisions
    // its own scoring would no longer make.
    constexpr uint32_t kUe3DiffuseSelectionScoringVersion = 4;

    // How many loaded picks to re-score and check against current scoring per session. A scoring
    // change disagrees broadly, so a small sample finds one; the cost is bounded to that many draws.
    constexpr uint32_t kUe3DiffuseSelectionAuditCount = 32;
  }

  fast_unordered_set s_loggedNonPrimaryRtDescHashes;

  fast_unordered_set s_loggedSampledRtDescHashes;

  uint32_t digestTextureTags(const fast_unordered_set& tags) {
    uint64_t sum = tags.size();
    for (const XXH64_hash_t tag : tags) {
      sum += XXH3_64bits(&tag, sizeof(tag));
    }
    return uint32_t(sum ^ (sum >> 32));
  }

  // Once per pixel shader: the sampler picture behind a draw with no albedo, plus the constant table and
  // UniformVector values of shaders that sample nothing.
  void D3D9Rtx::logUe3UnboundAlbedoOnce(const D3D9CommonShader* pixelShader,
                                        const XXH64_hash_t psHash,
                                        const uint32_t usedSamplerMask,
                                        const uint32_t usedTextureMask,
                                        const PsSamplerTexcoordEntry* inferredEntry) {
    if (pixelShader == nullptr || psHash == kEmptyHash) {
      return;
    }

    static fast_unordered_set s_loggedShaders;
    if (!s_loggedShaders.insert(psHash).second) {
      return;
    }

    const auto& samplerNames = getUe3PsSamplerNames(psHash, pixelShader->GetBytecode());
    std::string samplerLog;
    for (uint32_t stage : bit::BitMask(usedSamplerMask)) {
      if (stage >= caps::MaxTexturesPS) {
        continue;
      }
      samplerLog += samplerLog.empty() ? "s" : ", s";
      samplerLog += str::format(stage);
      const auto nameIt = samplerNames.find(stage);
      samplerLog += str::format("(", nameIt != samplerNames.end() ? nameIt->second.c_str() : "?", ")");
      if (d3d9State().textures[stage] == nullptr) {
        samplerLog += "=unbound";
        continue;
      }
      D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[stage]);
      if (texture == nullptr || texture->GetImage() == nullptr) {
        samplerLog += "=noimage";
        continue;
      }
      samplerLog += str::format(
        "=hash:0x", std::hex, texture->GetImage()->getHash(),
        ",desc:0x", texture->GetImage()->getDescriptorHash(), std::dec,
        texture->IsRenderTarget() ? ",RT" : "",
        isLightmapTexture(texture->GetImage()->getHash()) ? ",lightmap" : "",
        ",type:", uint32_t(texture->GetType()),
        ",samples:", inferredEntry != nullptr ? inferredEntry->samplers[stage].sampleCount : 0);
    }

    std::string constantLog;
    if (usedSamplerMask == 0) {
      const auto& bytecode = pixelShader->GetBytecode();
      if (bytecode.size() >= sizeof(uint32_t) && (bytecode.size() % sizeof(uint32_t)) == 0) {
        const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
        if ((tokens[0] & 0xffff0000u) == 0xffff0000u) {
          DxsoProgramInfo programInfo(DxsoProgramTypes::PixelShader, tokens[0] & 0xffu, (tokens[0] >> 8) & 0xffu);
          DxsoDecodeContext decoder(programInfo);
          DxsoCodeIter iter(tokens + 1);
          while (decoder.decodeInstruction(iter)) {
            if (decoder.getCtabInfo().m_size != 0) {
              break;
            }
          }
          for (const DxsoCtab::Constant& c : decoder.getCtabInfo().m_constantData) {
            constantLog += constantLog.empty() ? "" : ", ";
            constantLog += str::format(c.name, "@", c.registerSet == kD3dxRegisterSetSampler ? "s" : "c", c.registerIndex);
          }
        }
      }
      const Ue3PsMaterialIdentityInfo& identityInfo = getOrParseUe3PsMaterialIdentityInfo(
        psHash, bytecode, pixelShader, m_frameOptions.ue3MicVolatileConstantDetection);
      for (const uint32_t reg : identityInfo.uniformVectorRegisters) {
        if (reg >= caps::MaxFloatConstantsPS) {
          continue;
        }
        const Vector4& v = d3d9State().psConsts.fConsts[reg];
        constantLog += str::format(" | c", reg, "=(", v.x, ",", v.y, ",", v.z, ",", v.w, ")");
      }
    }
    if (!constantLog.empty()) {
      constantLog = str::format(" ctab=[", constantLog, "]");
    }

    Logger::warn(str::format(
      "[RTX-Compatibility][UE3-NoAlbedo] Draw has no bindable albedo texture: ps=0x", std::hex, psHash, std::dec,
      " vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
      " usedSamplerMask=0x", std::hex, usedSamplerMask,
      " boundUsedMask=0x", usedTextureMask, std::dec,
      " samplers=[", samplerLog.empty() ? "none" : samplerLog, "]",
      " alphaBlend=", d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] ? 1 : 0,
      " srcBlend=", d3d9State().renderStates[D3DRS_SRCBLEND],
      " dstBlend=", d3d9State().renderStates[D3DRS_DESTBLEND],
      constantLog));
  }

  bool D3D9Rtx::isUe3RenderTargetRefusedAsAlbedo(D3D9CommonTexture* texture,
                                                 const uint32_t stage,
                                                 const PsSamplerTexcoordEntry* inferredEntry) const {
    if (texture == nullptr || !texture->IsRenderTarget() || texture->GetImage() == nullptr) {
      return false;
    }

    const Rc<DxvkImage>& image = texture->GetImage();
    if (isUe3MovieTextureDescHash(image->getDescriptorHash())) {
      return false;
    }
    if (inferredEntry != nullptr && stage < caps::MaxTexturesPS &&
        (inferredEntry->samplers[stage].semanticFlags & kPsSamplerSemanticMovieTexture) != 0) {
      return false;
    }
    // Author intent wins: a preferred-albedo tag, or a screen whose material needs the target
    // itself bound as the albedo.
    if (lookupHash(*m_frameOptions.preferredAlbedoTextures, image->getHash())) {
      return false;
    }
    return !isTaggedRaytracedRenderTarget(image, true);
  }

  bool Ue3DiffuseSelectionCache::updateTags(const Ue3AlbedoTagDigests& tags) {
    if (!m_loaded) {
      load(tags);
    }
    if (tags == m_tags) {
      return false;
    }
    m_selections.clear();
    // Tagging invalidates every stored decision, so the persisted set has to shrink with
    // the in-memory one rather than keep serving picks the tags have just overruled.
    m_dirty = true;
    m_tags = tags;
    return true;
  }

  bool Ue3DiffuseSelectionCache::takeAudit(const Ue3DiffuseSelectionEntry& entry) {
    if (!entry.fromDisk || m_auditsRemaining == 0) {
      return false;
    }
    --m_auditsRemaining;
    return true;
  }

  void Ue3DiffuseSelectionCache::reportAudit(const XXH64_hash_t key, const uint8_t (&storedStages)[2],
                                             const uint8_t (&scoredStages)[2]) {
    if (m_auditWarned || (scoredStages[0] == storedStages[0] && scoredStages[1] == storedStages[1])) {
      return;
    }
    m_auditWarned = true;
    Logger::warn(str::format(
      "[RTX-Compatibility][UE3] Albedo selection cache disagrees with current scoring "
      "(material key 0x", std::hex, key, ": stored [s",
      uint32_t(storedStages[0]), ",s", uint32_t(storedStages[1]),
      "], scored [s", uint32_t(scoredStages[0]), ",s", uint32_t(scoredStages[1]), "])", std::dec,
      ". Stored picks are being used, so a scoring change will not take effect: bump "
      "kUe3DiffuseSelectionScoringVersion, or delete ", kUe3DiffuseSelectionCachePath, "."));
  }

  void Ue3DiffuseSelectionCache::load(const Ue3AlbedoTagDigests& tags) {
    m_loaded = true;

    auto refuseFutureSaves = [this](const std::string& reason) {
      m_saveBlocked = true;
      Logger::warn(str::format(
        "[RTX-Compatibility][UE3] Albedo selection cache ", reason,
        ". Leaving the file untouched for this session; albedo picks re-derive as materials are "
        "first drawn. Delete ", kUe3DiffuseSelectionCachePath, " to start a fresh one."));
    };

    const bool fileExists = ue3CacheFileExists(kUe3DiffuseSelectionCachePath);

    std::ifstream file(kUe3DiffuseSelectionCachePath, std::ios::binary);
    if (!file.is_open()) {
      if (fileExists) {
        refuseFutureSaves("exists but could not be opened");
      }
      return;
    }

    // A stored pick is only valid under the tags that produced it. Adopting the current digests
    // here also stops the first scoring draw's tag-change check from clearing everything just loaded.
    m_tags = tags;

    uint64_t magic = 0;
    uint32_t scoringVersion = 0;
    uint32_t lightmapDigest = 0;
    uint32_t neverAlbedoDigest = 0;
    uint32_t preferredAlbedoDigest = 0;
    uint32_t entryCount = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&scoringVersion), sizeof(scoringVersion));
    file.read(reinterpret_cast<char*>(&lightmapDigest), sizeof(lightmapDigest));
    file.read(reinterpret_cast<char*>(&neverAlbedoDigest), sizeof(neverAlbedoDigest));
    file.read(reinterpret_cast<char*>(&preferredAlbedoDigest), sizeof(preferredAlbedoDigest));
    file.read(reinterpret_cast<char*>(&entryCount), sizeof(entryCount));
    if (!file || magic != kUe3DiffuseSelectionCacheMagic || entryCount > kUe3DiffuseSelectionCacheMaxEntries) {
      refuseFutureSaves("header is unreadable or not recognised");
      return;
    }
    if (scoringVersion != kUe3DiffuseSelectionScoringVersion) {
      // Our own file, written by different scoring: discard and rewrite rather than refuse.
      Logger::info(str::format(
        "[RTX-Compatibility][UE3] Albedo selection cache was written by scoring version ",
        scoringVersion, " (this build is ", kUe3DiffuseSelectionScoringVersion,
        "); re-deriving picks and rewriting it."));
      m_dirty = true;
      return;
    }
    if (lightmapDigest != tags.lightmap ||
        neverAlbedoDigest != tags.neverAlbedo ||
        preferredAlbedoDigest != tags.preferredAlbedo) {
      Logger::info(
        "[RTX-Compatibility][UE3] Albedo selection cache was written under different lightmap, never-albedo or "
        "preferred-albedo texture tags; re-deriving picks and rewriting it.");
      m_dirty = true;
      return;
    }

    for (uint32_t i = 0; i < entryCount; i++) {
      XXH64_hash_t key = 0;
      Ue3DiffuseSelectionEntry entry;
      file.read(reinterpret_cast<char*>(&key), sizeof(key));
      file.read(reinterpret_cast<char*>(&entry.chosenStages[0]), sizeof(entry.chosenStages[0]));
      file.read(reinterpret_cast<char*>(&entry.chosenStages[1]), sizeof(entry.chosenStages[1]));
      file.read(reinterpret_cast<char*>(&entry.cubemapFallbackStage), sizeof(entry.cubemapFallbackStage));
      file.read(reinterpret_cast<char*>(&entry.decisionAreaSum), sizeof(entry.decisionAreaSum));
      if (!file) {
        refuseFutureSaves(str::format("is truncated: ", i, " of ", entryCount, " entries readable"));
        return;
      }
      entry.fromDisk = true;
      m_selections[key] = entry;
    }

    m_auditsRemaining = kUe3DiffuseSelectionAuditCount;

    Logger::info(str::format(
      "[RTX-Compatibility][UE3] Loaded albedo selection cache: ", entryCount, " materials"));
  }

  void Ue3DiffuseSelectionCache::save() {
    if (!m_dirty || m_saveBlocked) {
      return;
    }

    {
      std::ofstream file(kUe3DiffuseSelectionCacheTempPath, std::ios::binary | std::ios::trunc);
      if (!file.is_open()) {
        return;
      }

      const uint32_t entryCount =
        uint32_t(std::min<size_t>(m_selections.size(), kUe3DiffuseSelectionCacheMaxEntries));
      // The tracked digests rather than the live sets: these are what the stored decisions were
      // scored against, and they are readable from the destructor, where the frame options a
      // draw would have populated may never have existed.
      file.write(reinterpret_cast<const char*>(&kUe3DiffuseSelectionCacheMagic), sizeof(kUe3DiffuseSelectionCacheMagic));
      file.write(reinterpret_cast<const char*>(&kUe3DiffuseSelectionScoringVersion), sizeof(kUe3DiffuseSelectionScoringVersion));
      file.write(reinterpret_cast<const char*>(&m_tags.lightmap), sizeof(m_tags.lightmap));
      file.write(reinterpret_cast<const char*>(&m_tags.neverAlbedo), sizeof(m_tags.neverAlbedo));
      file.write(reinterpret_cast<const char*>(&m_tags.preferredAlbedo), sizeof(m_tags.preferredAlbedo));
      file.write(reinterpret_cast<const char*>(&entryCount), sizeof(entryCount));

      uint32_t written = 0;
      for (const auto& entry : m_selections) {
        if (written >= entryCount) {
          break;
        }
        file.write(reinterpret_cast<const char*>(&entry.first), sizeof(entry.first));
        file.write(reinterpret_cast<const char*>(&entry.second.chosenStages[0]), sizeof(entry.second.chosenStages[0]));
        file.write(reinterpret_cast<const char*>(&entry.second.chosenStages[1]), sizeof(entry.second.chosenStages[1]));
        file.write(reinterpret_cast<const char*>(&entry.second.cubemapFallbackStage), sizeof(entry.second.cubemapFallbackStage));
        file.write(reinterpret_cast<const char*>(&entry.second.decisionAreaSum), sizeof(entry.second.decisionAreaSum));
        written++;
      }

      file.close();
      if (!file) {
        return;
      }
    }

    if (ue3CommitCacheFile(kUe3DiffuseSelectionCacheTempPath, kUe3DiffuseSelectionCachePath)) {
      m_dirty = false;
    }
  }

  void Ue3DiffuseSelectionCache::saveIfDue(const uint64_t frameId) {
    if (m_dirty && frameId >= m_lastSaveFrame + kUe3CacheSaveIntervalFrames) {
      m_lastSaveFrame = uint32_t(frameId);
      save();
    }
  }

  uint32_t Ue3TextureSpreadCache::recordShader(const XXH64_hash_t textureHash, const XXH64_hash_t psHash) {
    if (!m_loaded) {
      load();
    }
    Ue3TextureMaterialSpread& spread = m_spreads[textureHash];
    bool psKnown = false;
    for (uint8_t i = 0; i < spread.count; i++) {
      if (spread.psHashes[i] == psHash) {
        psKnown = true;
        break;
      }
    }
    if (!psKnown && spread.count < spread.psHashes.size()) {
      spread.psHashes[spread.count++] = psHash;
      m_dirty = true;
    }
    return spread.scoringCount;
  }

  void Ue3TextureSpreadCache::load() {
    m_loaded = true;

    auto refuseFutureSaves = [this](const std::string& reason) {
      m_saveBlocked = true;
      Logger::warn(str::format(
        "[RTX-Compatibility][UE3] Texture material-spread cache ", reason,
        ". Leaving the file untouched for this session so it is not overwritten with less "
        "than it holds; albedo scoring falls back to no spread data. Delete ",
        kUe3TextureSpreadCachePath, " to start a fresh one."));
    };

    const bool fileExists = ue3CacheFileExists(kUe3TextureSpreadCachePath);

    std::ifstream file(kUe3TextureSpreadCachePath, std::ios::binary);
    if (!file.is_open()) {
      // No file at all is an ordinary first run, and saving must stay available to create one.
      if (fileExists) {
        refuseFutureSaves("exists but could not be opened");
      }
      return;
    }

    uint64_t magic = 0;
    uint32_t entryCount = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&entryCount), sizeof(entryCount));
    if (!file || magic != kUe3TextureSpreadCacheMagic || entryCount > kUe3TextureSpreadCacheMaxEntries) {
      refuseFutureSaves("header is unreadable or not recognised");
      return;
    }

    for (uint32_t i = 0; i < entryCount; i++) {
      XXH64_hash_t texHash = 0;
      uint8_t count = 0;
      file.read(reinterpret_cast<char*>(&texHash), sizeof(texHash));
      file.read(reinterpret_cast<char*>(&count), sizeof(count));
      Ue3TextureMaterialSpread spread;
      if (!file || count > spread.psHashes.size()) {
        refuseFutureSaves(str::format("is truncated: ", i, " of ", entryCount, " entries readable"));
        return;
      }
      for (uint8_t p = 0; p < count; p++) {
        file.read(reinterpret_cast<char*>(&spread.psHashes[p]), sizeof(XXH64_hash_t));
      }
      if (!file) {
        refuseFutureSaves(str::format("is truncated: ", i, " of ", entryCount, " entries readable"));
        return;
      }
      spread.count = count;
      spread.scoringCount = count;
      m_spreads[texHash] = spread;
    }

    Logger::info(str::format(
      "[RTX-Compatibility][UE3] Loaded texture material-spread cache: ", entryCount, " textures"));
  }

  void Ue3TextureSpreadCache::save() {
    if (!m_dirty || m_saveBlocked) {
      return;
    }

    {
      std::ofstream file(kUe3TextureSpreadCacheTempPath, std::ios::binary | std::ios::trunc);
      if (!file.is_open()) {
        return;
      }

      const uint32_t entryCount =
        uint32_t(std::min<size_t>(m_spreads.size(), kUe3TextureSpreadCacheMaxEntries));
      file.write(reinterpret_cast<const char*>(&kUe3TextureSpreadCacheMagic), sizeof(kUe3TextureSpreadCacheMagic));
      file.write(reinterpret_cast<const char*>(&entryCount), sizeof(entryCount));

      uint32_t written = 0;
      for (const auto& entry : m_spreads) {
        if (written >= entryCount) {
          break;
        }
        file.write(reinterpret_cast<const char*>(&entry.first), sizeof(entry.first));
        file.write(reinterpret_cast<const char*>(&entry.second.count), sizeof(entry.second.count));
        for (uint8_t p = 0; p < entry.second.count; p++) {
          file.write(reinterpret_cast<const char*>(&entry.second.psHashes[p]), sizeof(XXH64_hash_t));
        }
        written++;
      }

      file.close();
      if (!file) {
        return;
      }
    }

    // Cleared only once the new file is in place, so a failed write is retried next interval.
    if (ue3CommitCacheFile(kUe3TextureSpreadCacheTempPath, kUe3TextureSpreadCachePath)) {
      m_dirty = false;
    }
  }

  void Ue3TextureSpreadCache::saveIfDue(const uint64_t frameId) {
    if (m_dirty && frameId >= m_lastSaveFrame + kUe3CacheSaveIntervalFrames) {
      m_lastSaveFrame = uint32_t(frameId);
      save();
    }
  }

  // The shader path binds the textures the pixel shader actually reads, scoring them for the albedo
  // slot (see "Albedo selection and the texture spread cache" in UE3Compatibility.md).
  void D3D9Rtx::selectUe3BoundTextures(Ue3TextureState& ue3, uint32_t& firstStage) {
    const uint8_t kInvalidStage = 0xFF;
    const D3D9CommonShader*& inferredPs = ue3.inferredPs;
    XXH64_hash_t& inferredPsHash = ue3.inferredPsHash;
    PsSamplerTexcoordEntry*& inferredPsEntry = ue3.inferredPsEntry;
    const Ue3VertexFactoryType vfType = ue3.vfType;
    const bool isUe3MorphVF = ue3.isUe3MorphVF;
    const bool likelyGpuSkinnedMesh = ue3.likelyGpuSkinnedMesh;
    const bool likelyUe3FlexiblePackedUvPath = ue3.likelyUe3FlexiblePackedUvPath;
    const bool likelyPackedUvConventions = ue3.likelyPackedUvConventions;
    bool& selectedUe3MovieTexture = ue3.selectedUe3MovieTexture;
    // for the shader path we pick the most relevant textures actually used by the pixel shader
    // we prefer sRGB textures for the first slot since normal maps/masks are usually sampled in linear space
    uint8_t chosenStages[LegacyMaterialData::kMaxSupportedTextures] = { kInvalidStage, kInvalidStage };
    int64_t chosenScore[LegacyMaterialData::kMaxSupportedTextures] = { std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::min() };
    uint8_t strictCubemapFallbackStage = kInvalidStage;
    int32_t strictCubemapFallbackScore = std::numeric_limits<int32_t>::min();

    const uint32_t usedSamplerMask = m_parent->m_psShaderMasks.samplerMask;
    // Lightmaps leave the texture set before anything reads it; a score penalty would still let
    // DirectionalLightmaps flip the pick (see "UE3 lightmaps are bypassed" in UE3Compatibility.md).
    const uint32_t lightmapStageMask =
      (m_frameOptions.ue3EngineMode && inferredPsEntry != nullptr) ? inferredPsEntry->lightmapSamplerMask : 0u;
    // Lighting-only inputs (normal, specular, transfer masks) leave the pool too, since the simple lightmap
    // compile never declares them. Opacity masks and coordinate drivers stay eligible.
    uint32_t lightingInputStageMask = 0;
    if (m_frameOptions.ue3EngineMode && inferredPs != nullptr && inferredPsHash != kEmptyHash) {
      lightingInputStageMask = getOrParseUe3PsMaterialIdentityInfo(
        inferredPsHash, inferredPs->GetBytecode(), inferredPs,
        m_frameOptions.ue3MicVolatileConstantDetection).lightingInputSamplerMask;
    }
    const uint32_t usedTextureMask =
      m_parent->m_activeTextures & usedSamplerMask & ~lightmapStageMask & ~lightingInputStageMask;

    // Publish what was bound at those samplers so the texture paths that cannot see a pixel
    // shader - hash preservation on CPU writes, the terrain baker's stage filter, the texture
    // picker - recognise the same images. First sight per hash only; the session set is
    // append-only so the repeat check stays local.
    if (m_frameOptions.ue3AutoDetectLightmapTextures) {
      for (const uint32_t stage : bit::BitMask(lightmapStageMask & m_parent->m_activeTextures)) {
        if (stage >= SamplerCount || d3d9State().textures[stage] == nullptr) {
          continue;
        }
        D3D9CommonTexture* lightmap = GetCommonTexture(d3d9State().textures[stage]);
        if (lightmap == nullptr || lightmap->GetImage() == nullptr) {
          continue;
        }
        const XXH64_hash_t lightmapHash = lightmap->GetImage()->getHash();
        if (lightmapHash != kEmptyHash && m_ue3SeenLightmapTextures.insert(lightmapHash).second) {
          registerAutoDetectedLightmapTexture(lightmapHash);
        }
      }
    }

    // Scoring reads live constants UE3 rewrites per draw, so the first decision per key is pinned.
    XXH64_hash_t selectionCacheKey = kEmptyHash;
    bool selectionCacheUsable = false;
    bool selectionFromCache = false;
    uint64_t selectionBoundAreaSum = 0;
    bool auditingPinnedSelection = false;
    uint8_t auditedPinnedStages[2] = { kInvalidStage, kInvalidStage };
    uint8_t auditedPinnedCubemapStage = kInvalidStage;
    if (m_frameOptions.ue3EngineMode &&
        inferredPsEntry != nullptr && inferredPsHash != kEmptyHash) {
      ScopedCpuProfileZoneN("UE3 diffuse selection lookup");
      if (m_ue3DiffuseSelectionCache.updateTags(m_frameOptionSets.albedoTagDigests)) {
        // re-log re-scored selections so tag effects are visible in ue3LogAlbedoSelection output
        m_loggedAlbedoSelections.clear();
      }

      struct SelectionKeyTuple {
        uint32_t stage;
        uint32_t srgb;
        XXH64_hash_t texHash;
      };
      static_assert(sizeof(SelectionKeyTuple) == 16, "SelectionKeyTuple must have no implicit padding (it is hashed by memory).");
      // one tuple per bound texture plus a trailing vertex-factory context tuple,
      // whose out-of-range stage index cannot collide with a real texture tuple
      std::array<SelectionKeyTuple, SamplerCount + 1> keyTuples;
      uint32_t keyTupleCount = 0;
      const BoundTextureSnapshot& boundTextures = ensureBoundTextureSnapshot();
      for (const uint32_t stage : bit::BitMask(usedTextureMask & boundTextures.mask)) {
        const BoundTextureSnapshotEntry& entry = boundTextures.entries[stage];
        if (!entry.hasImage) {
          continue;
        }
        keyTuples[keyTupleCount++] = {
          stage,
          d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1u,
          entry.imageHash,
        };

        const auto* desc = entry.texture->Desc();
        if (desc != nullptr) {
          selectionBoundAreaSum += uint64_t(desc->Width) * uint64_t(desc->Height);
        }
      }
      // score-relevant vertex factory context (packed UV biases differ per factory)
      const uint32_t vfContext =
        uint32_t(vfType) |
        (likelyGpuSkinnedMesh ? 1u << 8 : 0u) |
        (likelyUe3FlexiblePackedUvPath ? 1u << 9 : 0u) |
        (likelyPackedUvConventions ? 1u << 10 : 0u);
      keyTuples[keyTupleCount++] = { uint32_t(SamplerCount), vfContext, kEmptyHash };
      selectionCacheKey = XXH3_64bits_withSeed(keyTuples.data(), keyTupleCount * sizeof(SelectionKeyTuple), inferredPsHash);
      selectionCacheUsable = true;

      // Streaming-stable hashes give every mip variant of a material the same key, so a
      // decision scored against streamed-down mips (smaller bound texel area) is only
      // authoritative for equal or smaller sets; a larger set re-scores and supersedes it.
      const Ue3DiffuseSelectionEntry* cachedSelection = m_ue3DiffuseSelectionCache.find(selectionCacheKey);
      if (cachedSelection != nullptr) {
        if (selectionBoundAreaSum <= cachedSelection->decisionAreaSum) {
          chosenStages[0] = cachedSelection->chosenStages[0];
          chosenStages[1] = cachedSelection->chosenStages[1];
          strictCubemapFallbackStage = cachedSelection->cubemapFallbackStage;
          selectionFromCache = true;

          // Drop cached picks that landed on a refused render target so a real material
          // sampler can re-compete. Reusing the scoring-loop predicate keeps a legitimately
          // tagged pick from being discarded and re-scored every frame.
          const auto cachedPickRefused = [&](const uint8_t stage) {
            if (stage == kInvalidStage || stage >= SamplerCount ||
                d3d9State().textures[stage] == nullptr) {
              return false;
            }
            return isUe3RenderTargetRefusedAsAlbedo(
              GetCommonTexture(d3d9State().textures[stage]), stage, inferredPsEntry);
          };

          if (m_frameOptions.ue3EngineMode &&
              (cachedPickRefused(chosenStages[0]) || cachedPickRefused(chosenStages[1]))) {
            chosenStages[0] = kInvalidStage;
            chosenStages[1] = kInvalidStage;
            strictCubemapFallbackStage = kInvalidStage;
            selectionFromCache = false;
            m_ue3DiffuseSelectionCache.erase(selectionCacheKey);
            m_loggedAlbedoSelections.erase(selectionCacheKey);
          } else if (m_ue3DiffuseSelectionCache.takeAudit(*cachedSelection)) {
            // Score this one anyway and compare. The pinned pick still wins - re-scoring at an
            // arbitrary moment is exactly the transient the pin exists to avoid - so the audit
            // only reports.
            auditingPinnedSelection = true;
            auditedPinnedStages[0] = chosenStages[0];
            auditedPinnedStages[1] = chosenStages[1];
            auditedPinnedCubemapStage = strictCubemapFallbackStage;
            chosenStages[0] = kInvalidStage;
            chosenStages[1] = kInvalidStage;
            strictCubemapFallbackStage = kInvalidStage;
            selectionFromCache = false;
          }
        } else {
          // superseding re-score: let ue3LogAlbedoSelection dump the authoritative decision
          m_loggedAlbedoSelections.erase(selectionCacheKey);
        }
      }
    }

    // per-stage score breakdown for rtx.d3d9.ue3LogAlbedoSelection, dumped once per selection key.
    // Audited draws are excluded: their scoring is discarded, so dumping it would report a
    // decision the draw did not use.
    const bool logAlbedoSelection =
      m_frameOptions.ue3LogAlbedoSelection &&
      selectionCacheUsable &&
      !selectionFromCache &&
      !auditingPinnedSelection &&
      m_loggedAlbedoSelections.find(selectionCacheKey) == m_loggedAlbedoSelections.end();
    std::string albedoSelectionLog;

    // Size credit in mip steps: one doubling of texel area is worth a fixed amount, so a
    // texture that has streamed one mip further gains a fixed, small advantage instead of
    // one proportional to the texel count.
    auto albedoAreaScore = [](const uint64_t area) -> int64_t {
      constexpr int64_t kScorePerMipStep = 40'000;
      constexpr int64_t kMaxMipSteps = 24;  // 4096x4096
      int64_t steps = 0;
      for (uint64_t remaining = area >> 1; remaining != 0 && steps < kMaxMipSteps; remaining >>= 1) {
        ++steps;
      }
      return steps * kScorePerMipStep;
    };

    // The unprovable-origin penalty only means something where the resolved UV transform is
    // actually consumed; the gate mirrors the one guarding that resolution below.
    const bool uvOriginPenaltyActive =
      m_frameOptions.ue3EngineMode &&
      inferredPsEntry != nullptr;

    const uint32_t scoringTextureMask = selectionFromCache ? 0u : usedTextureMask;
    for (uint32_t stage : bit::BitMask(scoringTextureMask)) {
      if (stage >= SamplerCount || d3d9State().textures[stage] == nullptr) {
        continue;
      }

      D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[stage]);
      if (!texture) {
        continue;
      }

      const D3DRESOURCETYPE textureType = texture->GetType();
      const bool is2DTexture = textureType == D3DRTYPE_TEXTURE;
      const bool isCubeTexture = textureType == D3DRTYPE_CUBETEXTURE;
      if (!is2DTexture && !isCubeTexture) {
        continue;
      }

      const XXH64_hash_t texHash = texture->GetSampleView(false)->image()->getHash();
      if (isLightmapTexture(texHash)) {
        continue;
      }
      const bool isNeverAlbedo = lookupHash(*m_frameOptions.neverAlbedoTextures, texHash);
      const bool isPreferredAlbedo = lookupHash(*m_frameOptions.preferredAlbedoTextures, texHash);

      // Scored from the spread loaded from disk, never the live count (see "Albedo selection and the
      // texture spread cache" in UE3Compatibility.md).
      uint32_t materialSpread = 0;
      if (inferredPsHash != kEmptyHash && texHash != kEmptyHash) {
        materialSpread = m_ue3TextureSpreadCache.recordShader(texHash, inferredPsHash);
      }

      const bool srgb = (d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1) != 0;
      const bool isRenderTarget = texture->IsRenderTarget();
      const XXH64_hash_t texDescHash =
        (texture->GetImage() != nullptr)
          ? texture->GetImage()->getDescriptorHash()
          : kEmptyHash;
      const auto* desc = texture->Desc();
      const uint64_t area = desc ? uint64_t(desc->Width) * uint64_t(desc->Height) : 0;
      uint16_t sampleCount = 0;
      bool hasInferredTexcoord = false;
      int8_t inferredTexcoordIdx = -1;
      uint8_t inferredSamplerSemanticFlags = 0;
      uint16_t inferredSamplerExpressionFlags = 0;
      bool inferredSamplerLooksEngineAuxiliary = false;
      bool inferredSamplerLooksMaterialTexture = false;
      bool inferredSamplerLooksLightmap = false;
      bool inferredSamplerLooksNonDiffuse = false;
      bool inferredSamplerLooksVideo = false;
      bool inferredSamplerLooksMovieTexture = false;
      bool inferredSamplerExprUvTransform = false;
      bool inferredSamplerExprUvOffset = false;
      bool inferredSamplerExprUvAnimated = false;
      bool inferredSamplerExprUvTimeDriven = false;
      bool inferredSamplerExprViewDependent = false;
      bool inferredSamplerExprMaskControl = false;
      bool inferredSamplerExprColorContribution = false;
      bool inferredSamplerExprBlendMath = false;
      bool inferredSamplerExprNormalDecode = false;
      bool inferredSamplerExprReachesOutputColor = false;
      bool inferredSamplerExprDiffuseAnchor = false;
      bool inferredUsesZw = false;
      bool inferredUsesWz = false;
      bool inferredUsesXy = false;
      bool inferredUsesPackedSecondary = false;
      bool hasNonZeroInferredOffset = false;
      // Whether this sampler's UV origin was proven back to a single interpolant. The
      // surface's texture transform is derived from the winning stage alone, so a stage
      // without a provable origin cannot carry one: winning the slot costs the surface
      // its whole UV transform and leaves the texture at raw interpolant scale.
      bool inferredSamplerUvOriginProvable = false;
      bool inferredSamplerReadsPrimaryUvPair = false;
      if (inferredPsEntry != nullptr && stage < caps::MaxTexturesPS) {
        const PsSamplerUvOrigin& stageUvOrigin = inferredPsEntry->samplerUvOrigin[stage];
        inferredSamplerUvOriginProvable = stageUvOrigin.originValid;
        // UE3 packs UV set 0 into an interpolant's .xy and set 1 into its .zw, so a proven
        // origin on the .xy pair names the mesh's primary channel.
        inferredSamplerReadsPrimaryUvPair =
          stageUvOrigin.originValid && stageUvOrigin.sitesAgree &&
          stageUvOrigin.compU == 0 && stageUvOrigin.compV == 1;
        sampleCount = inferredPsEntry->samplers[stage].sampleCount;
        inferredTexcoordIdx = inferredPsEntry->samplers[stage].texcoord;
        // GPUSkinMorphVF - TEXCOORD6/7 are morph delta streams, treat as noninferable UV
        if (isUe3MorphVF && inferredTexcoordIdx >= 6) {
          inferredTexcoordIdx = -1;
        }
        inferredSamplerSemanticFlags = inferredPsEntry->samplers[stage].semanticFlags;
        inferredSamplerExpressionFlags = inferredPsEntry->samplers[stage].expressionFlags;
        inferredSamplerLooksEngineAuxiliary = (inferredSamplerSemanticFlags & kPsSamplerSemanticEngineAuxiliary) != 0;
        inferredSamplerLooksMaterialTexture = (inferredSamplerSemanticFlags & kPsSamplerSemanticMaterialTexture) != 0;
        inferredSamplerLooksLightmap = (inferredSamplerSemanticFlags & kPsSamplerSemanticLightmap) != 0;
        inferredSamplerLooksNonDiffuse = (inferredSamplerSemanticFlags & kPsSamplerSemanticNonDiffuse) != 0;
        inferredSamplerLooksVideo = (inferredSamplerSemanticFlags & kPsSamplerSemanticVideo) != 0;
        inferredSamplerLooksMovieTexture = (inferredSamplerSemanticFlags & kPsSamplerSemanticMovieTexture) != 0;
        inferredSamplerExprUvTransform = (inferredSamplerExpressionFlags & kPsSamplerExprUvTransform) != 0;
        inferredSamplerExprUvOffset = (inferredSamplerExpressionFlags & kPsSamplerExprUvOffset) != 0;
        inferredSamplerExprUvAnimated = (inferredSamplerExpressionFlags & kPsSamplerExprUvAnimated) != 0;
        inferredSamplerExprUvTimeDriven = (inferredSamplerExpressionFlags & kPsSamplerExprUvTimeDriven) != 0;
        inferredSamplerExprViewDependent = (inferredSamplerExpressionFlags & kPsSamplerExprViewDependent) != 0;
        inferredSamplerExprMaskControl = (inferredSamplerExpressionFlags & kPsSamplerExprMaskControl) != 0;
        inferredSamplerExprColorContribution = (inferredSamplerExpressionFlags & kPsSamplerExprColorContribution) != 0;
        inferredSamplerExprBlendMath = (inferredSamplerExpressionFlags & kPsSamplerExprBlendMath) != 0;
        inferredSamplerExprNormalDecode = (inferredSamplerExpressionFlags & kPsSamplerExprNormalDecode) != 0;
        inferredSamplerExprReachesOutputColor = (inferredSamplerExpressionFlags & kPsSamplerExprReachesOutputColor) != 0;
        inferredSamplerExprDiffuseAnchor = (inferredSamplerExpressionFlags & kPsSamplerExprDiffuseAnchor) != 0;
        hasInferredTexcoord = inferredTexcoordIdx >= 0;
        inferredUsesZw =
          inferredPsEntry->samplers[stage].coordCompValid &&
          inferredPsEntry->samplers[stage].coordCompU == 2 &&
          inferredPsEntry->samplers[stage].coordCompV == 3;
        inferredUsesWz =
          inferredPsEntry->samplers[stage].coordCompValid &&
          inferredPsEntry->samplers[stage].coordCompU == 3 &&
          inferredPsEntry->samplers[stage].coordCompV == 2;
        inferredUsesXy =
          inferredPsEntry->samplers[stage].coordCompValid &&
          inferredPsEntry->samplers[stage].coordCompU == 0 &&
          inferredPsEntry->samplers[stage].coordCompV == 1;
        inferredUsesPackedSecondary = inferredUsesWz || inferredUsesZw;
        hasNonZeroInferredOffset = hasNonZeroInferredSamplerOffset(inferredPsEntry, stage);
      }
      const bool isMovieTexture =
        isUe3MovieTextureDescHash(texDescHash) ||
        (isRenderTarget && inferredSamplerLooksMovieTexture);

      if (isCubeTexture && !m_frameOptions.allowCubemaps) {
        const bool looksMaterialCubemap =
          sampleCount > 0 &&
          (!isRenderTarget || isMovieTexture) &&
          !inferredSamplerLooksEngineAuxiliary &&
          !inferredSamplerLooksLightmap &&
          !inferredSamplerLooksNonDiffuse &&
          !inferredSamplerExprViewDependent &&
          !inferredSamplerExprMaskControl &&
          !isNeverAlbedo &&
          (inferredSamplerLooksMaterialTexture || inferredSamplerSemanticFlags == 0);

        if (looksMaterialCubemap) {
          int32_t cubeScore = 0;
          cubeScore += int32_t(std::min<uint16_t>(sampleCount, 16u)) * 64;
          cubeScore += hasInferredTexcoord ? 256 : 0;
          cubeScore -= hasNonZeroInferredOffset ? 96 : 0;
          cubeScore -= int32_t(stage);

          if (cubeScore > strictCubemapFallbackScore) {
            strictCubemapFallbackScore = cubeScore;
            strictCubemapFallbackStage = uint8_t(stage);
          }
        }

        continue;
      }

      // hashless textures (typically render targets) can never be bound as legacy
      // albedo; letting one win a slot starves the real diffuse and leaves the
      // surface white with no clickable texture hash
      if (texHash == kEmptyHash) {
        continue;
      }

      // Non-UE3 keeps the score penalties below instead.
      if (m_frameOptions.ue3EngineMode &&
          isUe3RenderTargetRefusedAsAlbedo(texture, stage, inferredPsEntry)) {
        continue;
      }

      // A tiled texture covers its tile count times its area, but only tiny authored tiles get the credit:
      // tiled detail and dirt overlays are usually 256x256 or larger and must not win on size.
      constexpr uint64_t kTilingCreditMaxRawArea = 128ull * 128ull;
      constexpr uint64_t kTilingCreditMaxEffectiveArea = 2ull * 1024ull * 1024ull;
      uint64_t effectiveArea = area;
      if (inferredPsEntry != nullptr && stage < caps::MaxTexturesPS &&
          area > 0 && area <= kTilingCreditMaxRawArea) {
        float tilingU = 1.0f;
        float tilingV = 1.0f;
        bool tilingKnown = false;
        if (inferredPsEntry->samplers[stage].scaleImmediateValid) {
          tilingU = std::abs(inferredPsEntry->samplers[stage].scaleImmediateU);
          tilingV = std::abs(inferredPsEntry->samplers[stage].scaleImmediateV);
          tilingKnown = true;
        } else if (inferredPsEntry->samplers[stage].scaleConstReg >= 0 &&
                   uint32_t(inferredPsEntry->samplers[stage].scaleConstReg) < caps::MaxFloatConstantsPS) {
          const Vector4& scaleConst =
            d3d9State().psConsts.fConsts[uint32_t(inferredPsEntry->samplers[stage].scaleConstReg)];
          tilingU = std::abs(scaleConst[inferredPsEntry->samplers[stage].scaleConstCompU & 0x3u] *
                             inferredPsEntry->samplers[stage].scaleFactorU);
          tilingV = std::abs(scaleConst[inferredPsEntry->samplers[stage].scaleConstCompV & 0x3u] *
                             inferredPsEntry->samplers[stage].scaleFactorV);
          tilingKnown = std::isfinite(tilingU) && std::isfinite(tilingV);
        }
        if (tilingKnown) {
          const float tiles = std::min(std::max(tilingU * tilingV, 1.0f), 1024.0f);
          effectiveArea = std::min(uint64_t(double(area) * double(tiles)), kTilingCreditMaxEffectiveArea);
        }
      }
      // UV-math bonuses only apply where tiling resolved to an actual repeat factor: a tiled
      // small texture is an identity albedo, a plain UV transform on one is an overlay tell
      const bool hasResolvedTiling = effectiveArea > area;

      int64_t score = 0;
      score += srgb ? 1'000'000 : 0;
      score += int64_t(std::min<uint16_t>(sampleCount, 16u)) * 120'000ll;
      score += hasInferredTexcoord ? 250'000 : -150'000;
      score += hasResolvedTiling ? 80'000 : 0;
      score -= (isRenderTarget && !isMovieTexture) ? 500'000 : 0;
      score += isMovieTexture ? 4'000'000 : 0;
      score += inferredSamplerLooksMaterialTexture ? 230'000 : 0;
      score -= inferredSamplerLooksEngineAuxiliary ? 420'000 : 0;
      score -= inferredSamplerLooksLightmap ? 280'000 : 0;
      score -= inferredSamplerLooksNonDiffuse ? 220'000 : 0;
      score -= inferredSamplerExprViewDependent ? 260'000 : 0;
      score -= inferredSamplerExprMaskControl ? 320'000 : 0;
      score -= inferredSamplerLooksVideo ? 8'000'000 : 0;
      score -= isNeverAlbedo ? 6'000'000 : 0;
      score += isPreferredAlbedo ? 8'000'000 : 0;
      // bytecode-proven tangent-space normal decode (t * 2 - 1 into normalize/dot chains):
      // decisive penalty - must outweigh typical size advantages. Only applies to linear
      // (non-sRGB) samplers: UE3 imports normal maps with SRGB=0, while gamma-decoded color
      // textures can pick up the flag spuriously in lit shaders full of *2-1 remap math.
      const bool normalDecodeActive = inferredSamplerExprNormalDecode && !srgb;
      score -= normalDecodeActive ? 1'500'000 : 0;
      // only a high spread is directional: mid spreads (3-7) are just as often a legitimately
      // reused diffuse (e.g. common city wall/plaster sheets) as a shared overlay
      if (materialSpread >= 8) {
        score -= 2'600'000;
      }
      // tiny ramp/tint lookups (gradients, palettes) are material parameters, not albedo
      score -= (area > 0 && area <= 1'024) ? 350'000 : 0;
      if (m_frameOptions.ue3EngineMode) {
        // UE3 binds many scene buffers, shadow maps, exposure/color curves, and UI/video
        // surfaces alongside material samplers. Keep these out of legacy albedo slots.
        score -= ((isRenderTarget && !isMovieTexture) || inferredSamplerLooksEngineAuxiliary) ? 450'000 : 0;
        score -= (inferredSamplerLooksEngineAuxiliary && !inferredSamplerLooksMaterialTexture) ? 350'000 : 0;
        score -= (isRenderTarget && !srgb && !isMovieTexture) ? 250'000 : 0;
        score -= (!hasInferredTexcoord && inferredSamplerLooksEngineAuxiliary) ? 220'000 : 0;
      }
      const bool looksExpressionDrivenMaterial =
        !inferredSamplerLooksEngineAuxiliary &&
        !inferredSamplerLooksLightmap &&
        !inferredSamplerLooksNonDiffuse &&
        !inferredSamplerExprViewDependent &&
        !inferredSamplerExprMaskControl &&
        !isNeverAlbedo &&
        (inferredSamplerLooksMaterialTexture || inferredSamplerSemanticFlags == 0);
      // On a static mesh UV set 1 is the lightmap channel, so the sampler reading the primary .xy pair is the
      // surface's own diffuse. It must win outright: the winner also fixes the surface's texcoord set, which
      // the near-tie tiebreaks below could otherwise hand to a second layer.
      score += (looksExpressionDrivenMaterial && inferredSamplerReadsPrimaryUvPair) ? 150'000 : 0;
      score += (looksExpressionDrivenMaterial && inferredSamplerExprUvTransform && hasResolvedTiling) ? 95'000 : 0;
      score += (looksExpressionDrivenMaterial && inferredSamplerExprUvOffset) ? 52'000 : 0;
      score += (looksExpressionDrivenMaterial && inferredSamplerExprUvAnimated) ? 115'000 : 0;
      score += (looksExpressionDrivenMaterial && inferredSamplerExprUvTimeDriven && sampleCount >= 2u) ? 140'000 : 0;
      score += (looksExpressionDrivenMaterial && inferredSamplerExprColorContribution) ? 65'000 : 0;
      score += (looksExpressionDrivenMaterial && inferredSamplerExprBlendMath && sampleCount >= 2u) ? 60'000 : 0;
      // sampled value provably reaches oC0.rgb as color - the defining trait of a diffuse/emissive
      // texture, which normal/lighting-input samplers lack (their values collapse in dot products)
      score += (looksExpressionDrivenMaterial && inferredSamplerExprReachesOutputColor) ? 200'000 : 0;
      // Only the diffuse expression is multiplied by the lightmap or the ambient and sky constants. Render
      // targets are excluded: light attenuation buffers multiply into those terms too.
      score += (looksExpressionDrivenMaterial && !normalDecodeActive && !isRenderTarget &&
                inferredSamplerExprDiffuseAnchor) ? 2'500'000 : 0;
      // A sampler whose UV origin is unprovable must not win on size: the transform is taken
      // from the winner only, so promoting one drops the surface to raw interpolant UVs.
      // Flat, so a shader where no sampler resolves is left unaffected.
      score -= (uvOriginPenaltyActive && !inferredSamplerUvOriginProvable) ? 1'200'000 : 0;
      // normal maps get no size credit - resolution advantage must not offset the decode penalty.
      // Under UE3 the same applies to render targets that only survived the refusal above
      // through an explicit tag: their near-backbuffer area would dominate every other signal.
      const bool renderTargetSizeCreditDenied =
        m_frameOptions.ue3EngineMode && isRenderTarget && !isMovieTexture;
      // Size counts in mip steps, not texels: streaming rewrites the bound dimensions as a
      // level pages in, so raw area would let whichever candidate had paged in further win
      // outright. One doubling is worth less than any single structural signal, which orders
      // equally-classified candidates by size without letting residency overturn class.
      score += (normalDecodeActive || renderTargetSizeCreditDenied)
        ? 0
        : int64_t(albedoAreaScore(effectiveArea));
      // near-exact ties among color-chain candidates (magnitudes stay below any real signal):
      // prefer a UV transform (the tiled base material - overlays sample raw UVs), then the
      // LATER sampler (the material translator assigns Texture2D_N slots in property compile
      // order Normal -> Emissive -> Diffuse). Everywhere else prefer the earlier stage.
      const bool diffuseChainCandidate =
        looksExpressionDrivenMaterial && !normalDecodeActive && !isRenderTarget &&
        inferredSamplerExprReachesOutputColor;
      if (diffuseChainCandidate) {
        score += inferredSamplerExprUvTransform ? 200 : 0;
        score += int64_t(stage) * 4;
      } else {
        score -= int64_t(stage);
      }
      if (likelyPackedUvConventions) {
        // UE3-style shader paths (skinned and non-skinned vertex factories) frequently pack
        // secondary UVs into non-`.xy` components, so we bias slot 0 toward diffuse-like UV usage
        const bool packedPairSuspicious =
          hasNonZeroInferredOffset ||
          inferredSamplerLooksEngineAuxiliary ||
          (likelyGpuSkinnedMesh && !inferredSamplerLooksMaterialTexture);

        const bool skinnedHighConfidence =
          likelyGpuSkinnedMesh && inferredSamplerLooksMaterialTexture && sampleCount >= 3u;
        const int64_t uv0Bonus = likelyGpuSkinnedMesh ? 300'000ll : 60'000ll;
        const int64_t nonUv0Penalty = likelyGpuSkinnedMesh ? 180'000ll : 20'000ll;
        const int64_t xyBonus = likelyGpuSkinnedMesh ? 120'000ll : 35'000ll;
        const int64_t wzPenalty = likelyGpuSkinnedMesh
          ? (skinnedHighConfidence ? 160'000ll : 320'000ll)
          : (packedPairSuspicious ? 120'000ll : 8'000ll);
        const int64_t zwPenalty = likelyGpuSkinnedMesh
          ? (skinnedHighConfidence ? 125'000ll : 250'000ll)
          : (packedPairSuspicious ? 100'000ll : 8'000ll);
        const int64_t packedSecondaryPenalty = likelyGpuSkinnedMesh
          ? (skinnedHighConfidence ? 40'000ll : 80'000ll)
          : (packedPairSuspicious ? 28'000ll : 2'000ll);
        const int64_t offsetPenalty = likelyGpuSkinnedMesh
          ? 220'000ll
          : (inferredSamplerLooksEngineAuxiliary ? 180'000ll
             : (inferredSamplerLooksMaterialTexture ? 12'000ll : 45'000ll));
        const int64_t auxiliaryPenalty = likelyGpuSkinnedMesh ? 120'000ll : 70'000ll;
        const bool looksUvAnimatedMaterialTexture =
          inferredSamplerLooksMaterialTexture &&
          !inferredSamplerLooksEngineAuxiliary &&
          !inferredSamplerLooksNonDiffuse &&
          sampleCount >= 2u;
        const int64_t adjustedOffsetPenalty = looksUvAnimatedMaterialTexture
          ? std::max<int64_t>(offsetPenalty / 6ll, 2'000ll)
          : offsetPenalty;

        if (inferredTexcoordIdx == 0) {
          score += uv0Bonus;
        } else if (inferredTexcoordIdx > 0) {
          score -= nonUv0Penalty;
        }

        if (inferredUsesXy) {
          score += xyBonus;
        }
        if (inferredUsesWz) {
          score -= wzPenalty;
        } else if (inferredUsesZw) {
          score -= zwPenalty;
        }

        if (inferredUsesPackedSecondary) {
          score -= packedSecondaryPenalty;
        }

        if (hasNonZeroInferredOffset) {
          score -= adjustedOffsetPenalty;
        }

        if (inferredSamplerLooksEngineAuxiliary) {
          score -= auxiliaryPenalty;
        }

        const bool highConfidenceMaterialTexture =
          inferredSamplerLooksMaterialTexture && sampleCount >= 3u;
        if (inferredSamplerLooksMaterialTexture &&
            inferredUsesPackedSecondary &&
            !packedPairSuspicious &&
            (likelyUe3FlexiblePackedUvPath || (likelyGpuSkinnedMesh && highConfidenceMaterialTexture))) {
          score += 45'000ll;
        }
      }

      if (logAlbedoSelection) {
        std::string flagList;
        auto appendFlag = [&](const bool set, const char* name) {
          if (!set) {
            return;
          }
          if (!flagList.empty()) {
            flagList += "|";
          }
          flagList += name;
        };
        appendFlag(inferredSamplerLooksMaterialTexture, "MAT");
        appendFlag(inferredSamplerLooksEngineAuxiliary, "AUX");
        appendFlag(inferredSamplerLooksLightmap, "LIGHTMAP");
        appendFlag(inferredSamplerLooksNonDiffuse, "NONDIFFUSE");
        appendFlag(inferredSamplerLooksVideo, "VIDEO");
        appendFlag(inferredSamplerLooksMovieTexture, "MOVIE");
        appendFlag(inferredSamplerExprUvTransform, "UVXFORM");
        appendFlag(inferredSamplerExprUvOffset, "UVOFS");
        appendFlag(inferredSamplerExprUvAnimated, "UVANIM");
        appendFlag(inferredSamplerExprUvTimeDriven, "UVTIME");
        appendFlag(inferredSamplerExprViewDependent, "VIEWDEP");
        appendFlag(inferredSamplerExprMaskControl, "MASKCTL");
        appendFlag(inferredSamplerExprColorContribution, "COLORCONTRIB");
        appendFlag(inferredSamplerExprBlendMath, "BLEND");
        appendFlag(normalDecodeActive, "NORMALDECODE");
        appendFlag(inferredSamplerExprNormalDecode && !normalDecodeActive, "NORMALDECODE-SRGBVETO");
        appendFlag(inferredSamplerExprReachesOutputColor, "REACHESOC0");
        appendFlag(inferredSamplerExprDiffuseAnchor, "ANCHOR");
        appendFlag(isNeverAlbedo, "TAG:NEVERALBEDO");
        appendFlag(isPreferredAlbedo, "TAG:PREFERALBEDO");
        appendFlag(isRenderTarget, "RT");
        appendFlag(uvOriginPenaltyActive && !inferredSamplerUvOriginProvable, "NOUVORIGIN");
        appendFlag(inferredSamplerReadsPrimaryUvPair, "UV0XY");
        // what the register-granular inference had derived that the coordinate lanes rule out
        if (inferredPsEntry != nullptr && inferredPsEntry->samplerExpressionFlagsCleared[stage] != 0) {
          const uint16_t cleared = inferredPsEntry->samplerExpressionFlagsCleared[stage];
          std::string clearedList;
          if (cleared & kPsSamplerExprUvTransform) clearedList += "UVXFORM ";
          if (cleared & kPsSamplerExprUvOffset)    clearedList += "UVOFS ";
          if (cleared & kPsSamplerExprUvAnimated)  clearedList += "UVANIM ";
          if (cleared & kPsSamplerExprBlendMath)   clearedList += "BLEND ";
          appendFlag(true, str::format("LANECLEARED:", clearedList).c_str());
        }

        albedoSelectionLog += str::format(
          "\n  s", stage,
          " tex=0x", std::hex, texHash, std::dec,
          " ", desc ? desc->Width : 0u, "x", desc ? desc->Height : 0u,
          " srgb=", srgb ? 1 : 0,
          " samples=", sampleCount,
          " tc=", int32_t(inferredTexcoordIdx),
          " effArea=", effectiveArea,
          " spread=", materialSpread,
          " flags=[", flagList.empty() ? "-" : flagList, "]",
          " score=", score);
      }

      // insert into top-2 (simple selection sort)
      for (uint32_t slot = 0; slot < LegacyMaterialData::kMaxSupportedTextures; slot++) {
        if (stage == chosenStages[slot]) {
          break;
        }

        if (score > chosenScore[slot]) {
          for (uint32_t s = LegacyMaterialData::kMaxSupportedTextures - 1; s > slot; s--) {
            chosenStages[s] = chosenStages[s - 1];
            chosenScore[s] = chosenScore[s - 1];
          }
          chosenStages[slot] = uint8_t(stage);
          chosenScore[slot] = score;
          break;
        }
      }
    }

    if (chosenStages[0] == kInvalidStage && strictCubemapFallbackStage != kInvalidStage) {
      chosenStages[0] = strictCubemapFallbackStage;
    }

    // Nothing scored: fall back to the lowest used sampler with a hashed texture, as stage 0 may hold a
    // stale one. The second pass admits refused render targets when they are all the shader samples.
    for (uint32_t pass = 0; pass < 2 && chosenStages[0] == kInvalidStage; pass++) {
      const bool allowRefusedRenderTargets = pass == 1;
      for (uint32_t stage : bit::BitMask(usedTextureMask)) {
        if (stage >= SamplerCount || d3d9State().textures[stage] == nullptr) {
          continue;
        }
        D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[stage]);
        if (texture == nullptr || texture->GetImage() == nullptr ||
            texture->GetImage()->getHash() == kEmptyHash) {
          continue;
        }
        if (!allowRefusedRenderTargets && m_frameOptions.ue3EngineMode &&
            isUe3RenderTargetRefusedAsAlbedo(texture, stage, inferredPsEntry)) {
          continue;
        }
        chosenStages[0] = uint8_t(stage);
        break;
      }
    }

    // compat fix for UE3-like packed UV conventions:
    // if primary stage selection is suspicious (packed UV components, non-UV0, or atlas offset)
    // prefer a sibling sampler that references the same texture but looks more diffuse-like
    if (!selectionFromCache &&
        likelyPackedUvConventions &&
        inferredPsEntry != nullptr &&
        chosenStages[0] != kInvalidStage &&
        chosenStages[0] < caps::MaxTexturesPS &&
        d3d9State().textures[chosenStages[0]] != nullptr) {
      const uint8_t primaryStage = chosenStages[0];
      const bool primaryUsesZw = inferredPsEntry->samplers[primaryStage].coordCompValid &&
                                 inferredPsEntry->samplers[primaryStage].coordCompU == 2 &&
                                 inferredPsEntry->samplers[primaryStage].coordCompV == 3;
      const bool primaryUsesWz = inferredPsEntry->samplers[primaryStage].coordCompValid &&
                                 inferredPsEntry->samplers[primaryStage].coordCompU == 3 &&
                                 inferredPsEntry->samplers[primaryStage].coordCompV == 2;
      const bool primaryUsesPackedSecondary = primaryUsesWz || primaryUsesZw;
      const int8_t primaryTexcoord = inferredPsEntry->samplers[primaryStage].texcoord;
      const bool primaryHasNonZeroOffset = hasNonZeroInferredSamplerOffset(inferredPsEntry, primaryStage);
      const bool primaryLooksEngineAuxiliary =
        (inferredPsEntry->samplers[primaryStage].semanticFlags & kPsSamplerSemanticEngineAuxiliary) != 0;
      const bool primaryLooksMaterialTexture =
        (inferredPsEntry->samplers[primaryStage].semanticFlags & kPsSamplerSemanticMaterialTexture) != 0;
      const bool primaryLooksViewDependent =
        (inferredPsEntry->samplers[primaryStage].expressionFlags & kPsSamplerExprViewDependent) != 0;
      const bool primaryLooksMaskControl =
        (inferredPsEntry->samplers[primaryStage].expressionFlags & kPsSamplerExprMaskControl) != 0;
      const bool primaryPackedPairSuspicious =
        primaryUsesPackedSecondary &&
        (primaryLooksEngineAuxiliary ||
         primaryHasNonZeroOffset ||
         (likelyGpuSkinnedMesh && !primaryLooksMaterialTexture));
      const bool primaryOffsetSuspicious =
        primaryHasNonZeroOffset &&
        (likelyGpuSkinnedMesh || primaryLooksEngineAuxiliary);
      const bool primarySuspicious =
        primaryLooksEngineAuxiliary ||
        primaryLooksViewDependent ||
        primaryLooksMaskControl ||
        primaryPackedPairSuspicious ||
        primaryOffsetSuspicious ||
        (likelyGpuSkinnedMesh && primaryTexcoord > 0);

      if (primarySuspicious) {
        D3D9CommonTexture* primaryTexture = GetCommonTexture(d3d9State().textures[primaryStage]);
        const XXH64_hash_t primaryHash =
          (primaryTexture != nullptr && primaryTexture->GetImage() != nullptr)
            ? primaryTexture->GetImage()->getHash()
            : kEmptyHash;

        auto scoreDiffuseCandidate = [&](const uint32_t stage) -> int32_t {
          if (stage >= caps::MaxTexturesPS) {
            return std::numeric_limits<int32_t>::min();
          }

          if (inferredPsEntry->samplers[stage].sampleCount == 0) {
            return std::numeric_limits<int32_t>::min();
          }

          int32_t score = 0;
          const int32_t uv0Bonus = likelyGpuSkinnedMesh ? 420 : 110;
          const int32_t nonUv0Penalty = likelyGpuSkinnedMesh ? 240 : 40;
          const int32_t xyBonus = likelyGpuSkinnedMesh ? 320 : 80;

          const int8_t tc = inferredPsEntry->samplers[stage].texcoord;
          if (tc == 0) {
            score += uv0Bonus;
          } else if (tc > 0) {
            score -= nonUv0Penalty;
          }

          const uint8_t semanticFlags = inferredPsEntry->samplers[stage].semanticFlags;
          const uint16_t expressionFlags = inferredPsEntry->samplers[stage].expressionFlags;
          const bool looksMaterialTexture = (semanticFlags & kPsSamplerSemanticMaterialTexture) != 0;
          const bool looksEngineAuxiliary = (semanticFlags & kPsSamplerSemanticEngineAuxiliary) != 0;
          const bool looksNonDiffuse = (semanticFlags & kPsSamplerSemanticNonDiffuse) != 0;
          const bool looksVideo = (semanticFlags & kPsSamplerSemanticVideo) != 0;
          const bool looksExprUvTransform = (expressionFlags & kPsSamplerExprUvTransform) != 0;
          const bool looksExprUvOffset = (expressionFlags & kPsSamplerExprUvOffset) != 0;
          const bool looksExprUvAnimated = (expressionFlags & kPsSamplerExprUvAnimated) != 0;
          const bool looksExprUvTimeDriven = (expressionFlags & kPsSamplerExprUvTimeDriven) != 0;
          const bool looksExprViewDependent = (expressionFlags & kPsSamplerExprViewDependent) != 0;
          const bool looksExprMaskControl = (expressionFlags & kPsSamplerExprMaskControl) != 0;
          const bool looksExprColorContribution = (expressionFlags & kPsSamplerExprColorContribution) != 0;
          const bool looksExprBlendMath = (expressionFlags & kPsSamplerExprBlendMath) != 0;
          const bool looksExprNormalDecode = (expressionFlags & kPsSamplerExprNormalDecode) != 0;
          const bool looksExprReachesOutputColor = (expressionFlags & kPsSamplerExprReachesOutputColor) != 0;
          const bool looksExprDiffuseAnchor = (expressionFlags & kPsSamplerExprDiffuseAnchor) != 0;
          if ((semanticFlags & kPsSamplerSemanticMaterialTexture) != 0) {
            score += 180;
          }
          if ((semanticFlags & kPsSamplerSemanticEngineAuxiliary) != 0) {
            score -= 360;
          }
          if ((semanticFlags & kPsSamplerSemanticLightmap) != 0) {
            score -= 260;
          }
          if (m_frameOptions.ue3EngineMode && looksEngineAuxiliary) {
            score -= looksMaterialTexture ? 160 : 320;
          }
          if (looksNonDiffuse) {
            score -= 220;
          }
          if (looksExprViewDependent) {
            score -= 240;
          }
          if (looksExprMaskControl) {
            score -= 280;
          }
          if (looksVideo) {
            score -= 8000;
          }
          const bool candidateSrgb =
            (d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1) != 0;
          const bool candidateNormalDecode = looksExprNormalDecode && !candidateSrgb;
          if (candidateNormalDecode) {
            score -= 3000;
          }
          const bool looksExpressionDrivenMaterial =
            !looksEngineAuxiliary &&
            !looksNonDiffuse &&
            !looksExprViewDependent &&
            !looksExprMaskControl &&
            (looksMaterialTexture || semanticFlags == 0);
          if (looksExpressionDrivenMaterial && looksExprUvTransform) {
            score += 95;
          }
          if (looksExpressionDrivenMaterial && looksExprUvOffset) {
            score += 55;
          }
          if (looksExpressionDrivenMaterial && looksExprUvAnimated) {
            score += 125;
          }
          if (looksExpressionDrivenMaterial && looksExprUvTimeDriven &&
              inferredPsEntry->samplers[stage].sampleCount >= 2u) {
            score += 70;
          }
          if (looksExpressionDrivenMaterial && looksExprColorContribution) {
            score += 30;
          }
          if (looksExpressionDrivenMaterial && looksExprBlendMath &&
              inferredPsEntry->samplers[stage].sampleCount >= 2u) {
            score += 45;
          }
          if (looksExpressionDrivenMaterial && looksExprReachesOutputColor) {
            score += 80;
          }
          if (looksExpressionDrivenMaterial && !candidateNormalDecode && looksExprDiffuseAnchor) {
            score += 600;
          }

          const bool candidateHasNonZeroOffset = hasNonZeroInferredSamplerOffset(inferredPsEntry, stage);
          const bool candidatePackedPairSuspicious =
            candidateHasNonZeroOffset ||
            looksEngineAuxiliary ||
            (likelyGpuSkinnedMesh && !looksMaterialTexture);
          const int32_t wzPenalty = likelyGpuSkinnedMesh
            ? 380
            : (candidatePackedPairSuspicious ? 180 : 20);
          const int32_t zwPenalty = likelyGpuSkinnedMesh
            ? 360
            : (candidatePackedPairSuspicious ? 160 : 20);
          const int32_t offsetPenalty = likelyGpuSkinnedMesh
            ? 320
            : (looksEngineAuxiliary ? 220 : (looksMaterialTexture ? 20 : 60));
          const bool candidateLooksUvAnimatedMaterialTexture =
            looksMaterialTexture &&
            !looksEngineAuxiliary &&
            !looksNonDiffuse &&
            inferredPsEntry->samplers[stage].sampleCount >= 2u;
          const int32_t adjustedOffsetPenalty = candidateLooksUvAnimatedMaterialTexture
            ? std::max(offsetPenalty / 6, 8)
            : offsetPenalty;

          if (inferredPsEntry->samplers[stage].coordCompValid) {
            const uint8_t compU = inferredPsEntry->samplers[stage].coordCompU & 0x3u;
            const uint8_t compV = inferredPsEntry->samplers[stage].coordCompV & 0x3u;
            if (compU == 0u && compV == 1u) {
              score += xyBonus;
            } else if (compU == 3u && compV == 2u) {
              score -= wzPenalty;
            } else if (compU == 2u && compV == 3u) {
              score -= zwPenalty;
            }
          }

          if (candidateHasNonZeroOffset) {
            score -= adjustedOffsetPenalty;
          }

          if (!likelyGpuSkinnedMesh &&
              likelyUe3FlexiblePackedUvPath &&
              looksMaterialTexture &&
              inferredPsEntry->samplers[stage].coordCompValid) {
            const uint8_t compU = inferredPsEntry->samplers[stage].coordCompU & 0x3u;
            const uint8_t compV = inferredPsEntry->samplers[stage].coordCompV & 0x3u;
            const bool usesPackedSecondaryPair =
              (compU == 2u && compV == 3u) ||
              (compU == 3u && compV == 2u);
            if (usesPackedSecondaryPair && !candidatePackedPairSuspicious) {
              score += 60;
            }
          }

          score += int32_t(std::min<uint16_t>(inferredPsEntry->samplers[stage].sampleCount, 8u)) * 16;
          score -= int32_t(stage);

          return score;
        };

        if (primaryHash != kEmptyHash) {
          int32_t bestScore = scoreDiffuseCandidate(primaryStage);
          uint8_t promotedStage = kInvalidStage;

          for (uint32_t stage : bit::BitMask(usedTextureMask)) {
            if (stage >= SamplerCount ||
                stage == primaryStage ||
                d3d9State().textures[stage] == nullptr ||
                stage >= caps::MaxTexturesPS) {
              continue;
            }

            D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[stage]);
            if (texture == nullptr || texture->GetImage() == nullptr) {
              continue;
            }

            if (texture->GetImage()->getHash() != primaryHash) {
              continue;
            }

            const int32_t candidateScore = scoreDiffuseCandidate(stage);
            if (candidateScore > bestScore + 40) {
              bestScore = candidateScore;
              promotedStage = uint8_t(stage);
            }
          }

          if (promotedStage != kInvalidStage) {
            if (chosenStages[1] == promotedStage) {
              std::swap(chosenStages[0], chosenStages[1]);
            } else {
              chosenStages[0] = promotedStage;
            }
          }
        }
      }
    }

    // remember the final (post-fallback, post-promotion) decision for this material key,
    // overwriting any decision made against a smaller (streamed-down) texel area
    if (auditingPinnedSelection) {
      m_ue3DiffuseSelectionCache.reportAudit(selectionCacheKey, auditedPinnedStages, chosenStages);
      // The pin stands regardless - the audit reports, it does not re-decide.
      chosenStages[0] = auditedPinnedStages[0];
      chosenStages[1] = auditedPinnedStages[1];
      strictCubemapFallbackStage = auditedPinnedCubemapStage;
    } else if (selectionCacheUsable && !selectionFromCache) {
      Ue3DiffuseSelectionEntry cacheEntry;
      cacheEntry.chosenStages[0] = chosenStages[0];
      cacheEntry.chosenStages[1] = chosenStages[1];
      cacheEntry.cubemapFallbackStage = strictCubemapFallbackStage;
      cacheEntry.decisionAreaSum = selectionBoundAreaSum;
      m_ue3DiffuseSelectionCache.store(selectionCacheKey, cacheEntry);
    }

    if (logAlbedoSelection) {
      m_loggedAlbedoSelections.insert(selectionCacheKey);
      auto stageName = [&](const uint8_t stage) {
        return stage == kInvalidStage ? std::string("-") : str::format("s", uint32_t(stage));
      };
      Logger::info(str::format(
        "[RTX-Compatibility][UE3-AlbedoSelection] ps=0x", std::hex, inferredPsHash,
        " key=0x", selectionCacheKey, std::dec,
        " chosen=[", stageName(chosenStages[0]), ",", stageName(chosenStages[1]), "]",
        albedoSelectionLog.empty() ? "\n  (no scoreable candidates)" : albedoSelectionLog.c_str()));
    }

    uint32_t textureID = 0;
    for (uint32_t stageIdx = 0; stageIdx < LegacyMaterialData::kMaxSupportedTextures && textureID < LegacyMaterialData::kMaxSupportedTextures; stageIdx++) {
      const uint8_t stage = chosenStages[stageIdx];
      if (stage == kInvalidStage || stage >= SamplerCount || d3d9State().textures[stage] == nullptr) {
        continue;
      }

      D3D9CommonTexture* pTexInfo = GetCommonTexture(d3d9State().textures[stage]);
      assert(pTexInfo != nullptr);
      const XXH64_hash_t texHash = pTexInfo->GetImage()->getHash();
      const XXH64_hash_t texDescHash =
        pTexInfo->GetImage() != nullptr
          ? pTexInfo->GetImage()->getDescriptorHash()
          : kEmptyHash;
      const bool allowHashlessCubemap =
        pTexInfo->GetType() == D3DRTYPE_CUBETEXTURE && stage == strictCubemapFallbackStage;

      if (texHash == kEmptyHash && !allowHashlessCubemap) {
        continue;
      }

      if (textureID == 0) {
        firstStage = stage;
      }

      D3D9SamplerKey key = m_parent->CreateSamplerKey(stage);
      XXH64_hash_t samplerHash = D3D9SamplerKeyHash{}(key);

      Rc<DxvkSampler> sampler;
      auto samplerIt = m_samplerCache.find(samplerHash);
      if (samplerIt != m_samplerCache.end()) {
        sampler = samplerIt->second;
      } else {
        const auto samplerInfo = m_parent->DecodeSamplerKey(key);
        sampler = m_parent->GetDXVKDevice()->createSampler(samplerInfo);
        m_samplerCache.insert(std::make_pair(samplerHash, sampler));
      }

      const bool srgb = d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1;
      Rc<DxvkImageView> sampleView = getRemixSampleView(pTexInfo, srgb);
      if (sampleView == nullptr) {
        continue;
      }
      m_activeDrawCallState.materialData.colorTextures[textureID] = TextureRef(sampleView);
      m_activeDrawCallState.materialData.samplers[textureID] = sampler;
      selectedUe3MovieTexture |=
        isUe3MovieTextureDescHash(texDescHash) ||
        (stage < caps::MaxTexturesPS &&
         inferredPsEntry != nullptr &&
         (inferredPsEntry->samplers[stage].semanticFlags & kPsSamplerSemanticMovieTexture) != 0 &&
         pTexInfo->IsRenderTarget());
      if (textureID == 0) {
        m_activeDrawCallState.materialData.colorTextureIsSrgb = srgb;
      }

      auto shaderSampler = RemapStateSamplerShader(stage);
      m_activeDrawCallState.materialData.colorTextureSlot[textureID] = computeResourceSlotId(shaderSampler.first, DxsoBindingType::Image, uint32_t(shaderSampler.second));
      ++textureID;
    }

    if (textureID == 0 &&
        strictCubemapFallbackStage != kInvalidStage &&
        strictCubemapFallbackStage < SamplerCount &&
        d3d9State().textures[strictCubemapFallbackStage] != nullptr) {
      D3D9CommonTexture* pTexInfo = GetCommonTexture(d3d9State().textures[strictCubemapFallbackStage]);
      if (pTexInfo != nullptr && pTexInfo->GetImage() != nullptr) {
        firstStage = strictCubemapFallbackStage;

        D3D9SamplerKey key = m_parent->CreateSamplerKey(firstStage);
        XXH64_hash_t samplerHash = D3D9SamplerKeyHash{}(key);

        Rc<DxvkSampler> sampler;
        auto samplerIt = m_samplerCache.find(samplerHash);
        if (samplerIt != m_samplerCache.end()) {
          sampler = samplerIt->second;
        } else {
          const auto samplerInfo = m_parent->DecodeSamplerKey(key);
          sampler = m_parent->GetDXVKDevice()->createSampler(samplerInfo);
          m_samplerCache.insert(std::make_pair(samplerHash, sampler));
        }

        const bool srgb = d3d9State().samplerStates[firstStage][D3DSAMP_SRGBTEXTURE] & 0x1;
        Rc<DxvkImageView> sampleView = getRemixSampleView(pTexInfo, srgb);
        if (sampleView != nullptr) {
          m_activeDrawCallState.materialData.colorTextures[0] = TextureRef(sampleView);
          m_activeDrawCallState.materialData.samplers[0] = sampler;
          selectedUe3MovieTexture |= isUe3MovieTextureDescHash(pTexInfo->GetImage()->getDescriptorHash());
          m_activeDrawCallState.materialData.colorTextureIsSrgb = srgb;

          auto shaderSampler = RemapStateSamplerShader(firstStage);
          m_activeDrawCallState.materialData.colorTextureSlot[0] =
            computeResourceSlotId(shaderSampler.first, DxsoBindingType::Image, uint32_t(shaderSampler.second));
        }
      }
    }

    if (m_frameOptions.ue3EngineMode && !m_activeDrawCallState.materialData.colorTextures[0].isValid()) {
      logUe3UnboundAlbedoOnce(inferredPs, inferredPsHash, usedSamplerMask, usedTextureMask, inferredPsEntry);
    }
  }

}
