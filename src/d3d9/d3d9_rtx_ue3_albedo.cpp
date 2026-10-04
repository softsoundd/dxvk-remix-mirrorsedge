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
    // Shared by the UE3 on-disk caches. Each rewrites its whole file from an in-memory
    // container, so a save is only safe when that container holds everything the file held:
    // a load that ends up with less must take the file out of reach for the session, or one
    // unreadable read becomes permanent loss of data that took a playthrough to accumulate.
    // Distinguishing "absent" from "unreadable" is what makes a first run still able to save.
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

    constexpr char kUe3TextureSpreadCachePath[] = "rtx-remix/ue3TextureSpread.cache";

    constexpr char kUe3TextureSpreadCacheTempPath[] = "rtx-remix/ue3TextureSpread.cache.tmp";

    constexpr uint64_t kUe3TextureSpreadCacheMagic = 0x3144525053334555ull; // "UE3SPRD1"

    constexpr uint32_t kUe3TextureSpreadCacheMaxEntries = 1u << 20;

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

  // A raytraced draw whose material ends up with no albedo texture renders as an untextured
  // surface with nothing to click in the texture UI. Log the full sampler picture once per
  // pixel shader so the cause is attributable (shader samples no textures at all vs. all
  // candidates unbindable/skipped); for sampler-less shaders also dump the constant table
  // and live UniformVector values to identify the shader's role.
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
      " alphaBlend=", d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] != FALSE ? 1 : 0,
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

  void D3D9Rtx::loadUe3DiffuseSelectionCache() {
    m_ue3DiffuseSelectionLoaded = true;

    auto refuseFutureSaves = [this](const std::string& reason) {
      m_ue3DiffuseSelectionSaveBlocked = true;
      Logger::warn(str::format(
        "[RTX-Compatibility][UE3] Albedo selection cache ", reason,
        ". Leaving the file untouched for this session; albedo picks re-derive as materials are "
        "first drawn. Delete ", kUe3DiffuseSelectionCachePath, " to start a fresh one."));
    };

    const bool fileExists = ue3CacheFileExists(kUe3DiffuseSelectionCachePath);

    std::ifstream file(kUe3DiffuseSelectionCachePath, std::ios::binary);
    if (!file.is_open()) {
      if (fileExists)
        refuseFutureSaves("exists but could not be opened");
      return;
    }

    // A stored pick is only valid under the tags that produced it. Adopting the current digests
    // here also stops the first scoring draw's tag-change check from clearing everything just loaded.
    const FrameOptionSets& tagSets = m_frameOptionSets;
    m_ue3DiffuseSelectionLightmapTagDigest = tagSets.lightmapTextureDigest;
    m_ue3DiffuseSelectionNeverAlbedoTagDigest = tagSets.neverAlbedoTextureDigest;
    m_ue3DiffuseSelectionPreferredAlbedoTagDigest = tagSets.preferredAlbedoTextureDigest;

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
      m_ue3DiffuseSelectionDirty = true;
      return;
    }
    if (lightmapDigest != tagSets.lightmapTextureDigest ||
        neverAlbedoDigest != tagSets.neverAlbedoTextureDigest ||
        preferredAlbedoDigest != tagSets.preferredAlbedoTextureDigest) {
      Logger::info(
        "[RTX-Compatibility][UE3] Albedo selection cache was written under different lightmap, never-albedo or "
        "preferred-albedo texture tags; re-deriving picks and rewriting it.");
      m_ue3DiffuseSelectionDirty = true;
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
      m_ue3DiffuseSelectionCache[key] = entry;
    }

    m_ue3DiffuseSelectionAuditsRemaining = kUe3DiffuseSelectionAuditCount;

    Logger::info(str::format(
      "[RTX-Compatibility][UE3] Loaded albedo selection cache: ", entryCount, " materials"));
  }

  void D3D9Rtx::saveUe3DiffuseSelectionCache() {
    if (!m_ue3DiffuseSelectionDirty || m_ue3DiffuseSelectionSaveBlocked)
      return;

    {
      std::ofstream file(kUe3DiffuseSelectionCacheTempPath, std::ios::binary | std::ios::trunc);
      if (!file.is_open())
        return;

      const uint32_t entryCount =
        uint32_t(std::min<size_t>(m_ue3DiffuseSelectionCache.size(), kUe3DiffuseSelectionCacheMaxEntries));
      // The tracked digests rather than the live sets: these are what the stored decisions were
      // scored against, and they are readable from the destructor, where the frame options a
      // draw would have populated may never have existed.
      file.write(reinterpret_cast<const char*>(&kUe3DiffuseSelectionCacheMagic), sizeof(kUe3DiffuseSelectionCacheMagic));
      file.write(reinterpret_cast<const char*>(&kUe3DiffuseSelectionScoringVersion), sizeof(kUe3DiffuseSelectionScoringVersion));
      file.write(reinterpret_cast<const char*>(&m_ue3DiffuseSelectionLightmapTagDigest), sizeof(m_ue3DiffuseSelectionLightmapTagDigest));
      file.write(reinterpret_cast<const char*>(&m_ue3DiffuseSelectionNeverAlbedoTagDigest), sizeof(m_ue3DiffuseSelectionNeverAlbedoTagDigest));
      file.write(reinterpret_cast<const char*>(&m_ue3DiffuseSelectionPreferredAlbedoTagDigest), sizeof(m_ue3DiffuseSelectionPreferredAlbedoTagDigest));
      file.write(reinterpret_cast<const char*>(&entryCount), sizeof(entryCount));

      uint32_t written = 0;
      for (const auto& entry : m_ue3DiffuseSelectionCache) {
        if (written >= entryCount)
          break;
        file.write(reinterpret_cast<const char*>(&entry.first), sizeof(entry.first));
        file.write(reinterpret_cast<const char*>(&entry.second.chosenStages[0]), sizeof(entry.second.chosenStages[0]));
        file.write(reinterpret_cast<const char*>(&entry.second.chosenStages[1]), sizeof(entry.second.chosenStages[1]));
        file.write(reinterpret_cast<const char*>(&entry.second.cubemapFallbackStage), sizeof(entry.second.cubemapFallbackStage));
        file.write(reinterpret_cast<const char*>(&entry.second.decisionAreaSum), sizeof(entry.second.decisionAreaSum));
        written++;
      }

      file.close();
      if (!file)
        return;
    }

    if (ue3CommitCacheFile(kUe3DiffuseSelectionCacheTempPath, kUe3DiffuseSelectionCachePath))
      m_ue3DiffuseSelectionDirty = false;
  }

  void D3D9Rtx::loadUe3TextureSpreadCache() {
    m_ue3TextureSpreadLoaded = true;

    auto refuseFutureSaves = [this](const std::string& reason) {
      m_ue3TextureSpreadSaveBlocked = true;
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
      if (fileExists)
        refuseFutureSaves("exists but could not be opened");
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
      for (uint8_t p = 0; p < count; p++)
        file.read(reinterpret_cast<char*>(&spread.psHashes[p]), sizeof(XXH64_hash_t));
      if (!file) {
        refuseFutureSaves(str::format("is truncated: ", i, " of ", entryCount, " entries readable"));
        return;
      }
      spread.count = count;
      spread.scoringCount = count;
      m_ue3TextureMaterialSpread[texHash] = spread;
    }

    Logger::info(str::format(
      "[RTX-Compatibility][UE3] Loaded texture material-spread cache: ", entryCount, " textures"));
  }

  void D3D9Rtx::saveUe3TextureSpreadCache() {
    if (!m_ue3TextureSpreadDirty || m_ue3TextureSpreadSaveBlocked)
      return;

    {
      std::ofstream file(kUe3TextureSpreadCacheTempPath, std::ios::binary | std::ios::trunc);
      if (!file.is_open())
        return;

      const uint32_t entryCount =
        uint32_t(std::min<size_t>(m_ue3TextureMaterialSpread.size(), kUe3TextureSpreadCacheMaxEntries));
      file.write(reinterpret_cast<const char*>(&kUe3TextureSpreadCacheMagic), sizeof(kUe3TextureSpreadCacheMagic));
      file.write(reinterpret_cast<const char*>(&entryCount), sizeof(entryCount));

      uint32_t written = 0;
      for (const auto& entry : m_ue3TextureMaterialSpread) {
        if (written >= entryCount)
          break;
        file.write(reinterpret_cast<const char*>(&entry.first), sizeof(entry.first));
        file.write(reinterpret_cast<const char*>(&entry.second.count), sizeof(entry.second.count));
        for (uint8_t p = 0; p < entry.second.count; p++)
          file.write(reinterpret_cast<const char*>(&entry.second.psHashes[p]), sizeof(XXH64_hash_t));
        written++;
      }

      file.close();
      if (!file)
        return;
    }

    // Cleared only once the new file is in place, so a failed write is retried next interval.
    if (ue3CommitCacheFile(kUe3TextureSpreadCacheTempPath, kUe3TextureSpreadCachePath))
      m_ue3TextureSpreadDirty = false;
  }

}
