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
    // rtx.d3d9.ue3ReportMicIdentityChurn reports a family once it has minted this many identities: colour
    // variant siblings run to about eight, while an animating register passes any bound within a second
    // (see "What the runtime cannot recognise, and how it tells you" in UE3Compatibility.md).
    constexpr uint32_t kUe3MicChurnReportThreshold = 16;

    struct Ue3MicChurnReport {
      XXH64_hash_t firstConstantsHash = kEmptyHash;
      std::vector<Ue3MicIdentitySample::ConstantRecord> firstConstants;
      fast_unordered_set seenConstantsHashes;
      bool warned = false;
    };

    static fast_unordered_cache<Ue3MicChurnReport> s_ue3MicChurnByMaterialFamily;

    // Walks the registers in the order the constants hash streams them, so a report can name the
    // ones whose values moved. `emit` returns false to stop early, which is how the fixed-size
    // drift snapshot applies its bound.
    template <typename EmitFn>
    static void forEachUe3MicIdentityConstant(
        const Ue3PsMaterialIdentityInfo& identityInfo,
        const bool useNameOrderStream,
        EmitFn&& emit) {
      if (useNameOrderStream) {
        for (const auto& [uniformNameKey, uniformRegister] : identityInfo.namedUniformFirstRegistersByNameOrder) {
          if (uniformRegister < caps::MaxFloatConstantsPS && !emit(uniformRegister)) {
            return;
          }
        }
        return;
      }

      for (const auto& [rangeStart, rangeCount] : identityInfo.constRanges) {
        for (uint32_t r = rangeStart; r < rangeStart + rangeCount; r++) {
          if (r < caps::MaxFloatConstantsPS && !emit(r)) {
            return;
          }
        }
      }
    }

    static fast_unordered_cache<Ue3PsMaterialIdentityInfo> s_ue3PsMaterialIdentityCache;

    // The parse bakes in whether volatile registers were filtered, so a mid-session toggle of
    // rtx.d3d9.ue3MicVolatileConstantDetection has to invalidate it.
    static bool s_ue3PsMaterialIdentityCacheDetectedVolatile = true;

    // Whether the strength a key shows has changed since it first showed one above 0.
    class Ue3StrengthMotion {
    public:
      bool record(const XXH64_hash_t key, const float strength, bool* seenBefore = nullptr) {
        if (lookupHash(m_moved, key)) {
          if (seenBefore != nullptr) {
            *seenBefore = true;
          }
          return true;
        }
        // Bounded, as a moving object shows a new placement, and so a new key, every frame.
        if (m_first.size() >= kMaxStillKeys) {
          m_first.clear();
        }
        const auto [first, inserted] = m_first.emplace(key, strength);
        if (seenBefore != nullptr) {
          *seenBefore = !inserted;
        }
        if (inserted || first->second == strength) {
          return false;
        }
        m_first.erase(first);
        m_moved.insert(key);
        return true;
      }

    private:
      static constexpr size_t kMaxStillKeys = 1u << 16;
      fast_unordered_cache<float> m_first;
      fast_unordered_set m_moved;
    };

    static std::string describeUe3HighlightPair(const Ue3PsMaterialIdentityInfo::HighlightPair& pair) {
      const std::array<float, 3>& glow = pair.tint.glowCoefficient;
      const std::string glowText = glow[0] == glow[1] && glow[1] == glow[2]
        ? str::format(glow[0])
        : str::format("(", glow[0], ",", glow[1], ",", glow[2], ")");
      return pair.tint.glowOnly
        ? str::format(pair.strengthName, "@c", pair.tint.scalarReg, " glow-only s", pair.tint.glowSampler, ".",
                      "rgba"[pair.tint.glowComponent], " glow=", glowText)
        : str::format(pair.strengthName, "@c", pair.tint.scalarReg, " -> ", pair.colorName, "@c", pair.tint.colorReg[0],
                      " glow=", glowText);
    }

    // Opacity-driven fades (rtx.d3d9.ue3MaterialFades, rtx.d3d9.ue3ParticleVertexColor). Analysed once
    // per pixel shader and particle colour register; each draw evaluates the result with its constants.
    static const DxsoMaterialFadeResult& getOrAnalyzeUe3MaterialFades(const XXH64_hash_t psHash,
                                                                      const std::vector<uint8_t>& bytecode,
                                                                      const int32_t particleColorInputRegister) {
      static fast_unordered_cache<DxsoMaterialFadeResult> s_cache;
      const XXH64_hash_t key = XXH3_64bits_withSeed(&particleColorInputRegister, sizeof(particleColorInputRegister), psHash);
      const auto it = s_cache.find(key);
      if (it != s_cache.end()) {
        return it->second;
      }

      DxsoMaterialFadeResult result;
      result.failure = DxsoHighlightFailure::NotPixelShader;
      if (bytecode.size() >= 2 * sizeof(uint32_t) && (bytecode.size() % sizeof(uint32_t)) == 0) {
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
          result = analyzeDxsoMaterialFades(tokens, bytecode.size() / sizeof(uint32_t),
                                            dxsoHighlightInputsFromUe3Ctab(decoder.getCtabInfo()), particleColorInputRegister);
        }
      }
      return s_cache.emplace(key, std::move(result)).first->second;
    }

    static std::string describeUe3MaterialFade(const DxsoMaterialFade& fade, const std::map<uint32_t, std::string>& names) {
      const auto it = names.find(fade.reg);
      return str::format(it != names.end() ? it->second : std::string("?"), "@c", fade.reg, ".", "xyzw"[fade.component & 3u],
                         fade.laneMask == kDxsoFadeAlphaLane ? " oC0.a" : " oC0.rgb");
    }

    // The particle colour uses the shader proves, marked where they also depend on the draw's constants.
    static std::string describeUe3ParticleColorUses(const DxsoParticleColorUse& use) {
      auto anyTerms = [](const auto& polys) {
        return std::any_of(polys.begin(), polys.end(), [](const DxsoFadePoly& p) { return !p.empty(); });
      };
      std::string uses;
      uses += use.tintsColor ? (anyTerms(use.tintResidual) ? " tint(conditional)" : " tint") : "";
      uses += use.scalesColor ? (anyTerms(use.scalesColorResidual) ? " scalesColor(conditional)" : " scalesColor") : "";
      uses += use.scalesOpacity ? (use.scalesOpacityResidual.empty() ? " scalesOpacity" : " scalesOpacity(conditional)") : "";
      return uses.empty() ? std::string(" unused") : uses;
    }

    static void logUe3MaterialFadesOnce(const XXH64_hash_t psHash, const std::vector<uint8_t>& bytecode,
                                        const int32_t particleColorInputRegister, const DxsoMaterialFadeResult& result) {
      static fast_unordered_set s_logged;
      if (!s_logged.insert(XXH3_64bits_withSeed(&particleColorInputRegister, sizeof(particleColorInputRegister), psHash)).second) {
        return;
      }

      const std::map<uint32_t, std::string>& names = getUe3PsFloatConstantNames(psHash, bytecode);
      std::string fades;
      for (const DxsoMaterialFade& fade : result.fades) {
        fades += str::format(fades.empty() ? "" : ", ", describeUe3MaterialFade(fade, names));
      }
      std::string unproven;
      for (const uint32_t reg : result.unprovenScalarRegs) {
        const auto it = names.find(reg);
        unproven += str::format(unproven.empty() ? "" : ", ", it != names.end() ? it->second : std::string("?"), "@c", reg);
      }
      std::string particleColor;
      if (particleColorInputRegister >= 0) {
        particleColor = str::format(" particleColor=v", particleColorInputRegister, ":", describeUe3ParticleColorUses(result.particleColor));
      }
      Logger::info(str::format(
        "[RTX-Compatibility][UE3-Fade] ps=0x", std::hex, psHash, std::dec,
        " fades=[", fades, "]", particleColor, " unprovenScalars=[", unproven, "]",
        !result.analyzed ? str::format(" not analysed: ", dxsoHighlightFailureName(result.failure)) : std::string()));
    }

    // The value each oC0 lane must hold for the draw's colour blend to leave the framebuffer as it was:
    // a fade to that value fades the whole draw out. Opaque and masked draws have no such value.
    struct Ue3BlendRest {
      bool  alphaFades = false;
      float alphaRest = 0.0f;
      bool  colorFades = false;
      float colorRest = 0.0f;
    };

    static Ue3BlendRest ue3BlendRest(const DxvkBlendMode& blend) {
      Ue3BlendRest rest;
      if (!blend.enableBlending || blend.colorBlendOp != VK_BLEND_OP_ADD) {
        return rest;
      }
      const VkBlendFactor src = blend.colorSrcFactor;
      const VkBlendFactor dst = blend.colorDstFactor;
      if (src == VK_BLEND_FACTOR_SRC_ALPHA && (dst == VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA || dst == VK_BLEND_FACTOR_ONE)) {
        // UE3 Translucent, and its emissive variant
        rest.alphaFades = true;
        rest.alphaRest = 0.0f;
      } else if (src == VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA && dst == VK_BLEND_FACTOR_SRC_ALPHA) {
        rest.alphaFades = true;
        rest.alphaRest = 1.0f;
      }
      if ((src == VK_BLEND_FACTOR_ONE || src == VK_BLEND_FACTOR_SRC_ALPHA) && dst == VK_BLEND_FACTOR_ONE) {
        // UE3 Additive
        rest.colorFades = true;
        rest.colorRest = 0.0f;
      } else if ((src == VK_BLEND_FACTOR_DST_COLOR && dst == VK_BLEND_FACTOR_ZERO) ||
                 (src == VK_BLEND_FACTOR_ZERO && dst == VK_BLEND_FACTOR_SRC_COLOR)) {
        // UE3 Modulate
        rest.colorFades = true;
        rest.colorRest = 1.0f;
      } else if (src == VK_BLEND_FACTOR_DST_COLOR && dst == VK_BLEND_FACTOR_SRC_COLOR) {
        // modulate 2x
        rest.colorFades = true;
        rest.colorRest = 0.5f;
      }
      return rest;
    }
  }

  DxsoShaderView makeDxsoShaderView(const std::vector<uint8_t>& bytecode, const D3D9CommonShader* shader) {
    DxsoShaderView view;
    if (bytecode.size() >= sizeof(uint32_t) && (bytecode.size() % sizeof(uint32_t)) == 0) {
      view.tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
      view.tokenCount = bytecode.size() / sizeof(uint32_t);
    }
    if (shader != nullptr) {
      view.info = &shader->GetInfo();
      view.isgn = &shader->GetIsgn();
      view.osgn = &shader->GetOsgn();
    }
    return view;
  }

  const std::map<uint32_t, std::string>& getUe3PsSamplerNames(
      const XXH64_hash_t psHash,
      const std::vector<uint8_t>& bytecode) {
    static fast_unordered_cache<std::map<uint32_t, std::string>> s_ue3PsSamplerNameCache;

    auto it = s_ue3PsSamplerNameCache.find(psHash);
    if (it != s_ue3PsSamplerNameCache.end()) {
      return it->second;
    }

    std::map<uint32_t, std::string> names;
    if (bytecode.size() >= sizeof(uint32_t) && (bytecode.size() % sizeof(uint32_t)) == 0) {
      const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
      const uint32_t headerToken = tokens[0];
      if ((headerToken & 0xffff0000u) == 0xffff0000u) {
        const uint32_t majorVersion = (headerToken >> 8) & 0xffu;
        const uint32_t minorVersion = headerToken & 0xffu;
        DxsoProgramInfo programInfo(DxsoProgramTypes::PixelShader, minorVersion, majorVersion);

        DxsoDecodeContext decoder(programInfo);
        DxsoCodeIter iter(tokens + 1);
        while (decoder.decodeInstruction(iter)) {
          if (decoder.getCtabInfo().m_size != 0) {
            break;
          }
        }

        const DxsoCtab& ctab = decoder.getCtabInfo();
        for (const DxsoCtab::Constant& c : ctab.m_constantData) {
          if (c.registerSet != kD3dxRegisterSetSampler || c.registerCount == 0) {
            continue;
          }
          const uint32_t end = std::min<uint32_t>(c.registerIndex + c.registerCount, caps::MaxTexturesPS);
          for (uint32_t s = c.registerIndex; s < end; s++) {
            names[s] = c.name;
          }
        }
      }
    }

    return s_ue3PsSamplerNameCache.emplace(psHash, std::move(names)).first->second;
  }

  const std::map<uint32_t, std::string>& getUe3PsFloatConstantNames(
      const XXH64_hash_t psHash,
      const std::vector<uint8_t>& bytecode) {
    static fast_unordered_cache<std::map<uint32_t, std::string>> s_ue3PsFloatConstantNameCache;

    auto it = s_ue3PsFloatConstantNameCache.find(psHash);
    if (it != s_ue3PsFloatConstantNameCache.end()) {
      return it->second;
    }

    std::map<uint32_t, std::string> names;
    if (bytecode.size() >= sizeof(uint32_t) && (bytecode.size() % sizeof(uint32_t)) == 0) {
      const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
      const uint32_t headerToken = tokens[0];
      if ((headerToken & 0xffff0000u) == 0xffff0000u) {
        const uint32_t majorVersion = (headerToken >> 8) & 0xffu;
        const uint32_t minorVersion = headerToken & 0xffu;
        DxsoProgramInfo programInfo(DxsoProgramTypes::PixelShader, minorVersion, majorVersion);

        DxsoDecodeContext decoder(programInfo);
        DxsoCodeIter iter(tokens + 1);
        while (decoder.decodeInstruction(iter)) {
          if (decoder.getCtabInfo().m_size != 0) {
            break;
          }
        }

        const DxsoCtab& ctab = decoder.getCtabInfo();
        for (const DxsoCtab::Constant& c : ctab.m_constantData) {
          if (c.registerSet != kD3dxRegisterSetFloat4 || c.registerCount == 0) {
            continue;
          }
          const uint32_t end = std::min<uint32_t>(c.registerIndex + c.registerCount, caps::MaxFloatConstantsPS);
          for (uint32_t r = c.registerIndex; r < end; r++) {
            names[r] = c.registerCount > 1u
              ? str::format(c.name, "[", r - c.registerIndex, "]")
              : c.name;
          }
        }
      }
    }

    return s_ue3PsFloatConstantNameCache.emplace(psHash, std::move(names)).first->second;
  }

  fast_unordered_cache<Ue3MicIdentitySample> s_ue3MicIdentityByFamily;

  fast_unordered_set s_ue3MicRtPoisonWarnedFamilies;

  uint32_t snapshotUe3MicIdentityConstants(
      const Vector4* fConsts,
      const Ue3PsMaterialIdentityInfo& identityInfo,
      const bool useNameOrderStream,
      Ue3MicIdentitySample::ConstantRecord* out,
      const uint32_t outCapacity) {
    uint32_t count = 0;
    forEachUe3MicIdentityConstant(identityInfo, useNameOrderStream, [&](const uint32_t reg) {
      out[count++] = Ue3MicIdentitySample::ConstantRecord { uint16_t(reg), fConsts[reg] };
      return count < outCapacity;
    });
    return count;
  }

  std::vector<Ue3MicIdentitySample::ConstantRecord> snapshotUe3MicIdentityConstants(
      const Vector4* fConsts,
      const Ue3PsMaterialIdentityInfo& identityInfo,
      const bool useNameOrderStream) {
    std::vector<Ue3MicIdentitySample::ConstantRecord> records;
    forEachUe3MicIdentityConstant(identityInfo, useNameOrderStream, [&](const uint32_t reg) {
      records.push_back({ uint16_t(reg), fConsts[reg] });
      return true;
    });
    return records;
  }

  void reportUe3MicIdentityChurnOnce(
      const XXH64_hash_t textureSetShaderHash,
      const XXH64_hash_t psHash,
      const XXH64_hash_t shaderIdentitySeed,
      const XXH64_hash_t primaryTextureHash,
      const XXH64_hash_t constantsHash,
      const Vector4* fConsts,
      const Ue3PsMaterialIdentityInfo& identityInfo,
      const bool useNameOrderStream,
      const std::vector<uint8_t>& bytecode) {
    if (textureSetShaderHash == kEmptyHash || constantsHash == kEmptyHash) {
      return;
    }

    Ue3MicChurnReport& report = s_ue3MicChurnByMaterialFamily[textureSetShaderHash];
    if (report.warned) {
      return;
    }

    if (report.firstConstantsHash == kEmptyHash) {
      report.firstConstantsHash = constantsHash;
      report.firstConstants = snapshotUe3MicIdentityConstants(fConsts, identityInfo, useNameOrderStream);
      report.seenConstantsHashes.insert(constantsHash);
      return;
    }

    if (!report.seenConstantsHashes.insert(constantsHash).second ||
        report.seenConstantsHashes.size() < kUe3MicChurnReportThreshold) {
      return;
    }

    report.warned = true;
    report.seenConstantsHashes.clear();

    const std::vector<Ue3MicIdentitySample::ConstantRecord> current =
      snapshotUe3MicIdentityConstants(fConsts, identityInfo, useNameOrderStream);

    const std::map<uint32_t, std::string>& constantNames = getUe3PsFloatConstantNames(psHash, bytecode);
    std::string detail;
    for (const Ue3MicIdentitySample::ConstantRecord& now : current) {
      for (const Ue3MicIdentitySample::ConstantRecord& first : report.firstConstants) {
        if (first.reg != now.reg) {
          continue;
        }
        if (first.value.x != now.value.x || first.value.y != now.value.y ||
            first.value.z != now.value.z || first.value.w != now.value.w) {
          const auto nameIt = constantNames.find(uint32_t(now.reg));
          detail += str::format(
            "\n    c", uint32_t(now.reg),
            nameIt != constantNames.end() ? str::format(" (", nameIt->second, ")") : std::string(),
            ": (", first.value.x, ",", first.value.y, ",", first.value.z, ",", first.value.w,
            ") -> (", now.value.x, ",", now.value.y, ",", now.value.z, ",", now.value.w, ")");
        }
        break;
      }
    }

    Logger::warn(str::format(
      "[RTX-MicChurn] Material family textureSetShader=0x", std::hex, textureSetShaderHash,
      " (ps=0x", psHash, " seed=0x", shaderIdentitySeed, " tex=0x", primaryTextureHash, std::dec,
      ") has minted ", kUe3MicChurnReportThreshold,
      " identities from its constants tier, so a register is very likely animating:",
      // No register differing while the hash did means the two mints saw different register
      // *sets*, which for a fixed shader only happens when a permutation trimmed a uniform.
      detail.empty()
        ? str::format("\n    (constants hash 0x", std::hex, report.firstConstantsHash, " -> 0x", constantsHash,
                      std::dec, " with no differing register among the ", current.size(), " compared)")
        : detail,
      "\n  Every material hash it mints is unanchorable. Pin it with:"
      "\n    rtx.d3d9.ue3MicConstantIdentityExcludedMaterials = 0x", std::hex, textureSetShaderHash, std::dec,
      "\n  then re-anchor the material once. Identity is unchanged for this session."));

    // Nothing reads these once the family has reported; the record stays only for `warned`.
    report.firstConstants.clear();
    report.firstConstants.shrink_to_fit();
  }

  void logUe3MaterialInstanceHashBreakdownOnce(
      const XXH64_hash_t materialHash,
      const XXH64_hash_t psHash,
      const XXH64_hash_t shaderIdentitySeed,
      const XXH64_hash_t textureSetHash,
      const XXH64_hash_t textureSetShaderHash,
      const XXH64_hash_t constantsHash,
      const Ue3PsMaterialIdentityInfo& identityInfo,
      const bool constantsExcluded,
      const std::string& textureList) {
    static fast_unordered_set s_loggedMaterialInstanceHashes;
    if (!s_loggedMaterialInstanceHashes.insert(materialHash).second) {
      return;
    }

    std::string ranges;
    for (const auto& [start, count] : identityInfo.constRanges) {
      ranges += str::format(ranges.empty() ? "c" : ",c", start, "+", count);
    }

    std::string volatileRegisters;
    for (const uint32_t reg : identityInfo.volatileUniformRegisters) {
      volatileRegisters += str::format(volatileRegisters.empty() ? "c" : ",c", reg);
    }

    const bool usedCanonicalSeed = shaderIdentitySeed != psHash;
    Logger::info(str::format(
      "[RTX-Compatibility][UE3-MIC] materialHash=0x", std::hex, materialHash,
      " ps=0x", psHash,
      " seed=0x", shaderIdentitySeed, std::dec,
      usedCanonicalSeed
        ? (identityInfo.texturelessSignature ? " (canonical textureless, lightmap-permutation invariant)"
                                              : " (canonical, lightmap-permutation invariant)")
        : " (bytecode)",
      std::hex,
      " textureSet=0x", textureSetHash,
      " textureSetShader=0x", textureSetShaderHash,
      " consts=0x", constantsHash, std::dec,
      " textures=[", textureList, "]",
      " constRanges=[", ranges, "]",
      volatileRegisters.empty() ? std::string() : str::format(" volatileRegs=[", volatileRegisters, "]"),
      constantsExcluded ? " constsExcluded=1" : "",
      " ctab=", identityInfo.hasCtab ? 1 : 0));
  }

  XXH3_state_t* getThreadLocalXxh3State() {
    static thread_local XXH3_state_t* const state = XXH3_createState();
    return state;
  }

  XXH64_hash_t hashUe3MaterialConstants(
      const Vector4* fConsts,
      const Ue3MaterialConstRanges& ranges) {
    if (ranges.empty()) {
      return kEmptyHash;
    }

    XXH3_state_t* const state = getThreadLocalXxh3State();
    if (state == nullptr) {
      return kEmptyHash;
    }
    XXH3_64bits_reset(state);

    bool anyRegisterHashed = false;
    for (const auto& [start, count] : ranges) {
      if (start + count > caps::MaxFloatConstantsPS) {
        continue;
      }
      XXH3_64bits_update(state, &fConsts[start], count * sizeof(Vector4));
      anyRegisterHashed = true;
    }

    return anyRegisterHashed ? XXH3_64bits_digest(state) : kEmptyHash;
  }

  XXH64_hash_t hashUe3MaterialConstantsByNameOrder(
      const Vector4* fConsts,
      const std::vector<std::pair<XXH64_hash_t, uint32_t>>& namedUniformFirstRegistersByNameOrder) {
    if (namedUniformFirstRegistersByNameOrder.empty()) {
      return kEmptyHash;
    }

    XXH3_state_t* const state = getThreadLocalXxh3State();
    if (state == nullptr) {
      return kEmptyHash;
    }
    XXH3_64bits_reset(state);

    bool anyRegisterHashed = false;
    for (const auto& [nameKey, reg] : namedUniformFirstRegistersByNameOrder) {
      if (reg >= caps::MaxFloatConstantsPS) {
        continue;
      }
      XXH3_64bits_update(state, &nameKey, sizeof(nameKey));
      XXH3_64bits_update(state, &fConsts[reg], sizeof(Vector4));
      anyRegisterHashed = true;
    }

    return anyRegisterHashed ? XXH3_64bits_digest(state) : kEmptyHash;
  }

  const Ue3PsMaterialIdentityInfo& getOrParseUe3PsMaterialIdentityInfo(
      const XXH64_hash_t psHash,
      const std::vector<uint8_t>& bytecode,
      const D3D9CommonShader* pixelShader,
      const bool detectVolatileConstants) {
    if (s_ue3PsMaterialIdentityCacheDetectedVolatile != detectVolatileConstants) {
      s_ue3PsMaterialIdentityCacheDetectedVolatile = detectVolatileConstants;
      s_ue3PsMaterialIdentityCache.clear();
    }

    auto it = s_ue3PsMaterialIdentityCache.find(psHash);
    if (it == s_ue3PsMaterialIdentityCache.end()) {
      it = s_ue3PsMaterialIdentityCache.emplace(
        psHash, parseUe3PsMaterialIdentityFromCtab(makeDxsoShaderView(bytecode, pixelShader), detectVolatileConstants)).first;
    }
    return it->second;
  }

  bool evaluateUe3HighlightTints(const Ue3PsMaterialIdentityInfo& info, const Ue3HighlightTintDraw& draw,
                                 Vector3& tint, Vector3& glow, std::string* appliedLog) {
    static Ue3StrengthMotion s_objectMotion;
    static Ue3StrengthMotion s_instanceMotion;
    static fast_unordered_set s_heldBackLogged;

    bool applied = false;
    for (const Ue3PsMaterialIdentityInfo::HighlightPair& pair : info.highlightPairs) {
      const uint32_t strengthReg = pair.tint.scalarReg;
      if (strengthReg >= caps::MaxFloatConstantsPS) {
        continue;
      }
      const float strength = draw.fConsts[strengthReg].x;
      if (!std::isfinite(strength) || strength <= 0.0f) {
        continue;
      }

      // Only a highlight whose strength moves counts; one held still is authored in the tint colour. Tracked
      // per object as well as per instance (see "Runner Vision" in UE3Compatibility.md).
      if (draw.requireMotion) {
        const XXH64_hash_t instanceKey = XXH3_64bits_withSeed(&pair.nameKey, sizeof(pair.nameKey), draw.materialHash);
        const XXH64_hash_t objectKey = XXH3_64bits_withSeed(&pair.nameKey, sizeof(pair.nameKey), draw.objectHash);
        bool objectSeen = false;
        const bool objectMoved = s_objectMotion.record(objectKey, strength, &objectSeen);
        const bool instanceMoved = s_instanceMotion.record(instanceKey, strength);
        if (!objectMoved && (objectSeen || !instanceMoved)) {
          if (draw.log && s_heldBackLogged.insert(instanceKey).second) {
            Logger::info(str::format(
              "[RTX-Compatibility][UE3-Highlight] Held back on materialHash=0x", std::hex, draw.materialHash, std::dec,
              ": ", describeUe3HighlightPair(pair), " holds strength ", strength, " and has not moved."));
          }
          continue;
        }
      }

      const float s = std::min(strength, 1.0f);
      Vector3 color(1.0f, 1.0f, 1.0f);
      for (uint32_t lane = 0; lane < 3; lane++) {
        const int32_t reg = pair.tint.colorReg[lane];
        if (reg < 0 || uint32_t(reg) >= caps::MaxFloatConstantsPS) {
          continue;
        }
        const float value = draw.fConsts[reg][pair.tint.colorComponent[lane]];
        color[lane] = std::isfinite(value) ? std::max(value, 0.0f) : 1.0f;
        tint[lane] *= 1.0f + s * (color[lane] - 1.0f);
      }
      for (uint32_t lane = 0; lane < 3; lane++) {
        glow[lane] += pair.tint.glowCoefficient[lane] * s * draw.glowIntensity;
      }
      applied = true;

      if (appliedLog != nullptr) {
        *appliedLog += str::format(appliedLog->empty() ? "" : "; ", describeUe3HighlightPair(pair),
                                   " strength=", strength, " colour=(", color.x, ",", color.y, ",", color.z, ")");
      }
    }
    return applied;
  }

  void logUe3HighlightPairsOnce(const XXH64_hash_t psHash, const XXH64_hash_t shaderIdentitySeed,
                                const std::vector<uint8_t>& bytecode, const Ue3PsMaterialIdentityInfo& info) {
    if (info.highlightPairs.empty() && info.highlightUnprovenScalars.empty() &&
        info.highlightFailure == DxsoHighlightFailure::None) {
      return;
    }
    static fast_unordered_set s_logged;
    if (!s_logged.insert(psHash).second) {
      return;
    }

    std::string tints;
    for (const Ue3PsMaterialIdentityInfo::HighlightPair& pair : info.highlightPairs) {
      tints += str::format(tints.empty() ? "" : ", ", describeUe3HighlightPair(pair));
    }
    std::string unproven;
    const std::map<uint32_t, std::string>& names = getUe3PsFloatConstantNames(psHash, bytecode);
    for (const uint32_t reg : info.highlightUnprovenScalars) {
      const auto it = names.find(reg);
      unproven += str::format(unproven.empty() ? "" : ", ", it != names.end() ? it->second : std::string("?"), "@c", reg);
    }
    Logger::info(str::format(
      "[RTX-Compatibility][UE3-Highlight] ps=0x", std::hex, psHash, " seed=0x", shaderIdentitySeed, std::dec,
      " tints=[", tints, "] unprovenScalars=[", unproven, "]",
      info.highlightFailure != DxsoHighlightFailure::None
        ? str::format(" not analysed: ", dxsoHighlightFailureName(info.highlightFailure))
        : std::string()));
  }

  void D3D9Rtx::applyUe3MaterialFades(const XXH64_hash_t psHash, const std::vector<uint8_t>& bytecode,
                                      const D3D9CommonShader* pixelShader, const XXH64_hash_t textureSetShaderHash) {
    LegacyMaterialData& materialData = m_activeDrawCallState.materialData;
    const bool logFades = m_frameOptions.ue3LogMaterialFades;

    // The particle colour is TEXCOORD3 on SubUV sprites, whose TEXCOORD1 and 2 carry the second
    // sub-image and the blend between them, and TEXCOORD1 on plain sprites and beams/trails.
    uint32_t particleColorTexcoord = UINT32_MAX;
    int32_t particleColorInputRegister = -1;
    if (m_frameOptions.ue3ParticleVertexColor && pixelShader->GetInfo().majorVersion() >= 3 &&
        (m_currentUe3VertexFactory == Ue3VertexFactoryType::Particle ||
         m_currentUe3VertexFactory == Ue3VertexFactoryType::ParticleBeamTrail)) {
      particleColorTexcoord = 1;
      if (m_currentUe3VertexFactory == Ue3VertexFactoryType::Particle && d3d9State().vertexDecl != nullptr) {
        for (const auto& element : d3d9State().vertexDecl->GetElements()) {
          if (element.Usage == D3DDECLUSAGE_TEXCOORD && element.UsageIndex == 3 && element.Type == D3DDECLTYPE_FLOAT4) {
            particleColorTexcoord = 3;
          }
        }
      }
      const DxsoIsgn& isgn = pixelShader->GetIsgn();
      for (uint32_t i = 0; i < isgn.elemCount; i++) {
        if (isgn.elems[i].semantic.usage == DxsoUsage::Texcoord && isgn.elems[i].semantic.usageIndex == particleColorTexcoord) {
          particleColorInputRegister = int32_t(isgn.elems[i].regNumber);
        }
      }
    }

    const Ue3BlendRest blendRest = ue3BlendRest(materialData.blendMode);
    const bool canFade = m_frameOptions.ue3MaterialFades && (blendRest.alphaFades || blendRest.colorFades);
    if (particleColorInputRegister < 0 && !canFade && !logFades) {
      return;
    }

    const DxsoMaterialFadeResult& result = getOrAnalyzeUe3MaterialFades(psHash, bytecode, particleColorInputRegister);
    if (logFades) {
      logUe3MaterialFadesOnce(psHash, bytecode, particleColorInputRegister, result);
    }
    if (!result.analyzed) {
      return;
    }

    const float* constants = reinterpret_cast<const float*>(d3d9State().psConsts.fConsts);
    const uint32_t constantCount = caps::MaxFloatConstantsPS;
    auto vanishes = [&](const auto& polys) {
      return std::all_of(polys.begin(), polys.end(),
                         [&](const DxsoFadePoly& p) { return dxsoFadePolyVanishes(p, constants, constantCount); });
    };

    // The particle colour reaches Remix as the captured vertex colour, through the stage operations
    // the shader's own use of it maps onto. It is a material tint, not baked lighting.
    const DxsoParticleColorUse& particleColor = result.particleColor;
    const bool hasParticleColor = particleColorInputRegister >= 0;
    const bool tintsColor = hasParticleColor && particleColor.tintsColor && vanishes(particleColor.tintResidual);
    const bool scalesColor = hasParticleColor && particleColor.scalesColor && vanishes(particleColor.scalesColorResidual);
    const bool scalesOpacity = hasParticleColor && particleColor.scalesOpacity &&
                               dxsoFadePolyVanishes(particleColor.scalesOpacityResidual, constants, constantCount);
    if (tintsColor || scalesColor || scalesOpacity) {
      m_ue3ParticleColorTexcoordIndex = particleColorTexcoord;
      m_ue3ParticleColorCaptureFlags =
        (tintsColor ? 0u : kVertexCaptureFlag_ColorWhiteRgb) |
        (scalesColor ? kVertexCaptureFlag_ColorPremultiplyAlpha : 0u);
      materialData.isVertexColorBakedLighting = false;
      if (tintsColor || scalesColor) {
        materialData.textureColorArg1Source = RtTextureArgSource::Texture;
        materialData.textureColorArg2Source = RtTextureArgSource::VertexColor0;
        materialData.textureColorOperation = DxvkRtTextureOperation::Modulate;
      }
      if (scalesOpacity) {
        materialData.textureAlphaArg1Source = RtTextureArgSource::Texture;
        materialData.textureAlphaArg2Source = RtTextureArgSource::VertexColor0;
        materialData.textureAlphaOperation = DxvkRtTextureOperation::Modulate;
        materialData.ue3AnimatedVertexOpacity = true;
      }
    }

    const fast_unordered_set& excluded = *m_frameOptions.ue3MaterialFadeExcludedMaterials;
    const XXH64_hash_t colorTextureHash = materialData.getColorTexture().getImageHash();
    bool fadesExcluded = false;
    if (logFades || !excluded.empty()) {
      materialData.updateCachedHash();
      fadesExcluded = lookupHash(excluded, materialData.getHash()) || lookupHash(excluded, textureSetShaderHash) ||
                      lookupHash(excluded, colorTextureHash);
    }

    // Each modulator fades the draw once, whether its alpha, its colour or both come to rest. The
    // analysis lists a modulator's candidates together.
    float coverage = 1.0f;
    const DxsoMaterialFade* lastApplied = nullptr;
    std::string fadeLog;
    if (canFade && !fadesExcluded) {
      const std::map<uint32_t, std::string>* names = logFades ? &getUe3PsFloatConstantNames(psHash, bytecode) : nullptr;
      for (const DxsoMaterialFade& fade : result.fades) {
        const bool alphaLane = fade.laneMask == kDxsoFadeAlphaLane;
        if (alphaLane ? !blendRest.alphaFades : !blendRest.colorFades) {
          continue;
        }
        if (lastApplied != nullptr && lastApplied->reg == fade.reg && lastApplied->component == fade.component) {
          continue;
        }
        const std::optional<float> fadeCoverage =
          dxsoMaterialFadeCoverage(fade, alphaLane ? blendRest.alphaRest : blendRest.colorRest, constants, constantCount);
        if (names != nullptr) {
          fadeLog += str::format(fadeLog.empty() ? "" : "; ", describeUe3MaterialFade(fade, *names),
                                 " value=", fade.reg < constantCount ? constants[fade.reg * 4u + (fade.component & 3u)] : 0.0f,
                                 fadeCoverage ? str::format(" coverage=", *fadeCoverage) : std::string(" not at rest"));
        }
        if (!fadeCoverage) {
          continue;
        }
        lastApplied = &fade;
        coverage *= *fadeCoverage;
      }
      materialData.ue3FadeCoverage = coverage;
    }

    if (logFades && (hasParticleColor || !result.fades.empty() || !result.unprovenScalarRegs.empty())) {
      // One line per texture and state, so a fade's start, middle and end each show once.
      const uint32_t srcBlend = d3d9State().renderStates[D3DRS_SRCBLEND];
      const uint32_t dstBlend = d3d9State().renderStates[D3DRS_DESTBLEND];
      const uint32_t state = (tintsColor ? 1u : 0u) | (scalesColor ? 2u : 0u) | (scalesOpacity ? 4u : 0u) |
                             (lastApplied == nullptr ? 0u : coverage <= 0.0f ? 8u : coverage < 1.0f ? 16u : 32u) |
                             (fadesExcluded ? 64u : 0u) | (srcBlend << 8) | (dstBlend << 16) |
                             (uint32_t(m_currentUe3VertexFactory) << 24);
      XXH64_hash_t key = XXH3_64bits_withSeed(&colorTextureHash, sizeof(colorTextureHash), psHash);
      key = XXH3_64bits_withSeed(&state, sizeof(state), key);
      static fast_unordered_set s_loggedFadeDraws;
      if (s_loggedFadeDraws.insert(key).second) {
        std::string particleLog;
        if (hasParticleColor) {
          std::string uses;
          uses += tintsColor ? " tint" : "";
          uses += scalesColor ? " scalesColor" : "";
          uses += scalesOpacity ? " scalesOpacity" : "";
          particleLog = str::format(" particleColor=v", particleColorInputRegister, " applied=[", uses.empty() ? uses : uses.substr(1),
                                    "] proven=[", describeUe3ParticleColorUses(particleColor).substr(1), "]");
        }
        Logger::info(str::format(
          "[RTX-Compatibility][UE3-Fade] Draw: texture=0x", std::hex, colorTextureHash, " ps=0x", psHash,
          " materialHash=0x", materialData.getHash(), " textureSetShader=0x", textureSetShaderHash, std::dec,
          " vf=", describeUe3VertexFactory(m_currentUe3VertexFactory), " srcBlend=", srcBlend, " dstBlend=", dstBlend,
          particleLog, fadesExcluded ? " excluded" : "",
          canFade ? str::format(" coverage=", coverage, " fades=[", fadeLog, "]") : std::string(" blend does not fade")));
      }
    }
  }

  // See "Material identity and replacement anchor stability" in UE3Compatibility.md. Runs before
  // setupCategoriesForTexture, whose lookups use the full material hash.
  void D3D9Rtx::computeUe3MaterialIdentity(Ue3TextureState& ue3, const uint32_t firstStage) {
    if (m_frameOptions.ue3EngineMode && m_parent->UseProgrammablePS() && d3d9State().pixelShader.ptr() != nullptr) {
      ScopedCpuProfileZoneN("UE3 material identity");
      const D3D9CommonShader* psCommonShader = d3d9State().pixelShader->GetCommonShader();
      const auto& bytecode = psCommonShader->GetBytecode();
      const XXH64_hash_t psHash = psCommonShader->GetBytecodeHash();
      if (psHash != 0) {
        const Ue3PsMaterialIdentityInfo& identityInfo = getOrParseUe3PsMaterialIdentityInfo(
          psHash, bytecode, psCommonShader, m_frameOptions.ue3MicVolatileConstantDetection);

        // Seeded from the canonical CTAB signature, including permutations without lightmap symbols, so a
        // material keeps one identity across lightmap policies (see "Identity" in UE3Compatibility.md).
        const bool useInvariantShaderIdentity =
          m_frameOptions.ue3EngineMode &&
          identityInfo.canonicalShaderSignature != kEmptyHash;
        const XXH64_hash_t shaderIdentitySeed =
          useInvariantShaderIdentity ? identityInfo.canonicalShaderSignature : psHash;

        m_activeDrawCallState.materialData.setPixelShaderHashForMaterialInstance(shaderIdentitySeed);

        // Every texture bound to a material sampler, keyed by CTAB name for invariant-identity shaders:
        // lightmap sampler counts shift the register assignments between permutations.
        const bool logMicHash = m_frameOptions.ue3LogMaterialInstanceHash;
        if (logMicHash && !identityInfo.identitySummary.empty()) {
          static fast_unordered_set s_loggedIdentitySummaries;
          if (s_loggedIdentitySummaries.insert(psHash).second) {
            Logger::info(str::format(
              "[RTX-Ue3Identity] ps=0x", std::hex, psHash,
              " seed=0x", identityInfo.canonicalShaderSignature, std::dec,
              " ", identityInfo.identitySummary));
          }
        }
        std::string micTextureListLog;
        // Replacement identity drift diagnostics: record the (register, image hash, RT flag)
        // tuples that feed textureSetHash so tier drift can be attributed per sampler.
        Ue3MicIdentitySample::SamplerRecord micDiagSamplers[kMicDriftMaxTrackedSamplers];
        uint32_t micDiagSamplerCount = 0;
        const BoundTextureSnapshot& identityBoundTextures = ensureBoundTextureSnapshot();

        XXH64_hash_t textureSetHash = kEmptyHash;
        if (identityInfo.materialSamplerMask != 0) {
          XXH3_state_t* const state = getThreadLocalXxh3State();
          if (state != nullptr) {
            XXH3_64bits_reset(state);
            bool anyTextureHashed = false;
            const BoundTextureSnapshot& boundTextures = identityBoundTextures;
            auto hashMaterialSamplerTexture = [&](const void* samplerKey, const size_t samplerKeySize, const uint32_t samplerRegister, const char* samplerLogName) -> XXH64_hash_t {
              if (samplerRegister >= SamplerCount || (boundTextures.mask & (1u << samplerRegister)) == 0) {
                return kEmptyHash;
              }
              const BoundTextureSnapshotEntry& entry = boundTextures.entries[samplerRegister];
              if (!entry.hasImage) {
                return kEmptyHash;
              }
              const XXH64_hash_t imageHash = entry.imageHash;
              if (imageHash == kEmptyHash) {
                return kEmptyHash; // hashless (e.g. render target bound as a material texture)
              }
              if (m_frameOptions.ue3MicExcludeRenderTargetsFromIdentity && entry.isRenderTarget) {
                // RT image hashes change on every recreation (respawn/checkpoint/level
                // load) and would re-mint the identity each time; treat as hashless.
                if (m_frameOptions.logReplacementResolution || m_frameOptions.ue3LogMaterialInstanceHash) {
                  static fast_unordered_set s_loggedRtIdentityExclusions;
                  const XXH64_hash_t exclusionLogKey = XXH3_64bits_withSeed(&samplerRegister, sizeof(samplerRegister), psHash);
                  if (s_loggedRtIdentityExclusions.insert(exclusionLogKey).second) {
                    Logger::info(str::format(
                      "[RTX-Compatibility][UE3-MIC] Excluded render-target image 0x", std::hex, imageHash,
                      " (RT descriptor hash 0x", entry.rtDescriptorHash,
                      ") at material sampler s", std::dec, samplerRegister,
                      " of pixel shader 0x", std::hex, psHash, std::dec,
                      " from material identity (rtx.d3d9.ue3MicExcludeRenderTargetsFromIdentity)."));
                  }
                }
                return kEmptyHash;
              }
              if (!m_frameOptions.ue3MicIdentityExcludedTextureDescHashes->empty() &&
                  lookupHash(*m_frameOptions.ue3MicIdentityExcludedTextureDescHashes, entry.descriptorHash)) {
                // Identified by its descriptor hash rather than dropped: an empty texture set falls back to the
                // primary colour texture's image hash, the very value being excluded.
                if (m_frameOptions.logReplacementResolution || m_frameOptions.ue3LogMaterialInstanceHash) {
                  static fast_unordered_set s_loggedDescIdentityExclusions;
                  const XXH64_hash_t exclusionLogKey = XXH3_64bits_withSeed(&samplerRegister, sizeof(samplerRegister), psHash);
                  if (s_loggedDescIdentityExclusions.insert(exclusionLogKey).second) {
                    Logger::info(str::format(
                      "[RTX-Compatibility][UE3-MIC] Identifying material sampler s", std::dec, samplerRegister,
                      " of pixel shader 0x", std::hex, psHash,
                      " by its descriptor hash 0x", entry.descriptorHash,
                      " instead of its image hash 0x", imageHash, std::dec,
                      " (rtx.d3d9.ue3MicIdentityExcludedTextureDescHashes)."));
                  }
                }
                XXH3_64bits_update(state, samplerKey, samplerKeySize);
                XXH3_64bits_update(state, &entry.descriptorHash, sizeof(entry.descriptorHash));
                anyTextureHashed = true;
                if (micDiagSamplerCount < kMicDriftMaxTrackedSamplers) {
                  micDiagSamplers[micDiagSamplerCount++] = Ue3MicIdentitySample::SamplerRecord {
                    uint8_t(samplerRegister), entry.isRenderTarget, entry.descriptorHash, entry.descriptorHash };
                }
                if (logMicHash) {
                  micTextureListLog += str::format(
                    micTextureListLog.empty() ? "s" : ",s", samplerRegister,
                    samplerLogName != nullptr ? str::format("(", samplerLogName, ")") : std::string(),
                    ":desc0x", std::hex, entry.descriptorHash, std::dec);
                }
                return kEmptyHash;
              }
              XXH3_64bits_update(state, samplerKey, samplerKeySize);
              XXH3_64bits_update(state, &imageHash, sizeof(imageHash));
              anyTextureHashed = true;
              if (micDiagSamplerCount < kMicDriftMaxTrackedSamplers) {
                micDiagSamplers[micDiagSamplerCount++] = Ue3MicIdentitySample::SamplerRecord {
                  uint8_t(samplerRegister), entry.isRenderTarget, imageHash, entry.descriptorHash };
              }
              if (logMicHash) {
                micTextureListLog += str::format(
                  micTextureListLog.empty() ? "s" : ",s", samplerRegister,
                  samplerLogName != nullptr ? str::format("(", samplerLogName, ")") : std::string(),
                  ":0x", std::hex, imageHash, "(desc:0x", entry.descriptorHash, ")", std::dec);
              }
              return imageHash;
            };
            if (useInvariantShaderIdentity) {
              // name-keyed: register assignments shift between lightmap policy permutations
              for (const auto& [samplerName, samplerNameKey, samplerRegister] : identityInfo.materialSamplersByNameOrder) {
                if (samplerRegister < caps::MaxTexturesPS) {
                  hashMaterialSamplerTexture(&samplerNameKey, sizeof(samplerNameKey), samplerRegister, samplerName.c_str());
                }
              }
            } else {
              // register-keyed
              for (uint32_t s = 0; s < caps::MaxTexturesPS; s++) {
                if ((identityInfo.materialSamplerMask & (1u << s)) == 0) {
                  continue;
                }
                hashMaterialSamplerTexture(&s, sizeof(s), s, nullptr);
              }
            }
            if (anyTextureHashed) {
              textureSetHash = XXH3_64bits_digest(state);
            }
          }
        }
        m_activeDrawCallState.materialData.setMaterialTextureSetHashForMaterialInstance(textureSetHash);
        // A canonical textureless signature already names the material; whatever albedo
        // scoring bound for display must not leak into its identity.
        const bool textureSetIsComplete = useInvariantShaderIdentity && identityInfo.materialSamplerMask == 0;
        m_activeDrawCallState.materialData.setMaterialTextureSetIsComplete(textureSetIsComplete);

        // Volatile registers are already out of identityInfo, so the constants tier cannot change mid-session.
        // Both exclusion lists are keyed on values that outlive a session.
        const XXH64_hash_t identityTextureSetHash =
          (textureSetHash != kEmptyHash || textureSetIsComplete)
            ? textureSetHash
            : m_activeDrawCallState.materialData.getColorTexture().getImageHash();
        const XXH64_hash_t textureSetShaderHash =
          XXH3_64bits_withSeed(&identityTextureSetHash, sizeof(identityTextureSetHash), shaderIdentitySeed);
        const bool constantsExcluded =
          // The tier that cannot be made lightmap-policy independent; opt-in only.
          !m_frameOptions.ue3MicConstantIdentity ||
          lookupHash(*m_frameOptions.ue3MicConstantIdentityExcludedShaders, psHash) ||
          (useInvariantShaderIdentity && lookupHash(*m_frameOptions.ue3MicConstantIdentityExcludedShaders, shaderIdentitySeed)) ||
          lookupHash(*m_frameOptions.ue3MicConstantIdentityExcludedMaterials, textureSetShaderHash);
        // Invariant-identity shaders hash constants by uniform name, since lightmap permutations shift and trim
        // uniform registers. Other shaders hash the register range.
        XXH64_hash_t psConstsHash = kEmptyHash;
        if (!constantsExcluded) {
          psConstsHash = useInvariantShaderIdentity
            ? hashUe3MaterialConstantsByNameOrder(d3d9State().psConsts.fConsts, identityInfo.namedUniformFirstRegistersByNameOrder)
            : hashUe3MaterialConstants(d3d9State().psConsts.fConsts, identityInfo.constRanges);
        }
        m_activeDrawCallState.materialData.setPixelShaderConstantsHashForMaterialInstance(psConstsHash);

        // Report - never act on - a family that still mints more than one identity, so a
        // frame-varying register the bytecode could not see announces itself instead of
        // quietly breaking the replacements anchored on it.
        if (m_frameOptions.ue3ReportMicIdentityChurn && !constantsExcluded && psConstsHash != kEmptyHash) {
          reportUe3MicIdentityChurnOnce(
            textureSetShaderHash, psHash, shaderIdentitySeed,
            m_activeDrawCallState.materialData.getColorTexture().getImageHash(),
            psConstsHash, d3d9State().psConsts.fConsts, identityInfo, useInvariantShaderIdentity, bytecode);
        }

        // Replacement identity drift diagnostics: attribute a changed material hash
        // to the tier that moved and flag RT-poisoned identities.
        const bool replacementDiagActive =
          m_frameOptions.logReplacementResolution ||
          (m_frameOptions.replacementDebugHashes != nullptr && !m_frameOptions.replacementDebugHashes->empty());
        if (replacementDiagActive) {
          m_activeDrawCallState.materialData.updateCachedHash();
          const XXH64_hash_t materialHash = m_activeDrawCallState.materialData.getHash();
          const XXH64_hash_t primaryTexHash = m_activeDrawCallState.materialData.getColorTexture().getImageHash();

          bool tracked = false;
          if (m_frameOptions.replacementDebugHashes != nullptr && !m_frameOptions.replacementDebugHashes->empty()) {
            const fast_unordered_set& dbg = *m_frameOptions.replacementDebugHashes;
            tracked = lookupHash(dbg, primaryTexHash) || lookupHash(dbg, materialHash) || lookupHash(dbg, textureSetShaderHash);
            for (uint32_t i = 0; !tracked && i < micDiagSamplerCount; i++) {
              tracked = lookupHash(dbg, micDiagSamplers[i].imageHash);
            }
          }

          if (m_frameOptions.logReplacementResolution || tracked) {
            const XXH64_hash_t familyKey = XXH3_64bits_withSeed(&primaryTexHash, sizeof(primaryTexHash), shaderIdentitySeed);

            // Snapshot the constant registers feeding the identity so drift can name the
            // register(s) whose values moved.
            Ue3MicIdentitySample::ConstantRecord curConstants[kMicDriftMaxTrackedConstants];
            uint32_t curConstantCount = 0;
            if (!constantsExcluded && psConstsHash != kEmptyHash) {
              curConstantCount = snapshotUe3MicIdentityConstants(
                d3d9State().psConsts.fConsts, identityInfo, useInvariantShaderIdentity,
                curConstants, kMicDriftMaxTrackedConstants);
            }

            Ue3MicIdentitySample& prev = s_ue3MicIdentityByFamily[familyKey];
            if (prev.valid && prev.materialHash != materialHash &&
                prev.driftLogsEmitted < (tracked ? kMicDriftMaxLogsPerTrackedFamily : kMicDriftMaxLogsPerFamily)) {
              ++prev.driftLogsEmitted;
              std::string detail;
              if (prev.textureSetHash != textureSetHash) {
                detail += str::format("\n  textureSet 0x", std::hex, prev.textureSetHash, " -> 0x", textureSetHash, std::dec, ":");
                for (uint32_t i = 0; i < micDiagSamplerCount; i++) {
                  const Ue3MicIdentitySample::SamplerRecord& cur = micDiagSamplers[i];
                  const Ue3MicIdentitySample::SamplerRecord* old = nullptr;
                  for (uint32_t j = 0; j < prev.samplerCount; j++) {
                    if (prev.samplers[j].reg == cur.reg) {
                      old = &prev.samplers[j];
                      break;
                    }
                  }
                  if (old == nullptr) {
                    detail += str::format(" s", uint32_t(cur.reg), " added=0x", std::hex, cur.imageHash,
                                          "(desc:0x", cur.descriptorHash, ")", std::dec, cur.isRenderTarget ? "(RT)" : "");
                  } else if (old->imageHash != cur.imageHash) {
                    detail += str::format(" s", uint32_t(cur.reg), " 0x", std::hex, old->imageHash, "->0x", cur.imageHash,
                                          "(desc:0x", cur.descriptorHash, ")", std::dec, cur.isRenderTarget ? "(RT)" : "");
                  }
                }
                for (uint32_t j = 0; j < prev.samplerCount; j++) {
                  bool stillPresent = false;
                  for (uint32_t i = 0; i < micDiagSamplerCount; i++) {
                    if (micDiagSamplers[i].reg == prev.samplers[j].reg) {
                      stillPresent = true;
                      break;
                    }
                  }
                  if (!stillPresent) {
                    detail += str::format(" s", uint32_t(prev.samplers[j].reg), " removed=0x", std::hex, prev.samplers[j].imageHash, std::dec,
                                          prev.samplers[j].isRenderTarget ? "(RT)" : "");
                  }
                }
              }
              if (prev.constantsExcluded != constantsExcluded) {
                detail += str::format("\n  constantsExcluded ", prev.constantsExcluded ? 1 : 0, " -> ", constantsExcluded ? 1 : 0);
              }
              if (prev.constantsHash != psConstsHash) {
                detail += str::format("\n  consts 0x", std::hex, prev.constantsHash, " -> 0x", psConstsHash, std::dec, ":");
                for (uint32_t i = 0; i < curConstantCount; i++) {
                  const Ue3MicIdentitySample::ConstantRecord& cur = curConstants[i];
                  for (uint32_t j = 0; j < prev.constantCount; j++) {
                    const Ue3MicIdentitySample::ConstantRecord& old = prev.constants[j];
                    if (old.reg == cur.reg) {
                      if (old.value.x != cur.value.x || old.value.y != cur.value.y ||
                          old.value.z != cur.value.z || old.value.w != cur.value.w) {
                        detail += str::format(" c", uint32_t(cur.reg),
                                              " (", old.value.x, ",", old.value.y, ",", old.value.z, ",", old.value.w,
                                              ")->(", cur.value.x, ",", cur.value.y, ",", cur.value.z, ",", cur.value.w, ")");
                      }
                      break;
                    }
                  }
                }
              }
              Logger::warn(str::format(
                "[RTX-MicDrift] Material identity changed for family tex=0x", std::hex, primaryTexHash,
                " seed=0x", shaderIdentitySeed,
                ": materialHash 0x", prev.materialHash, " -> 0x", materialHash,
                " (textureSet+shader tier 0x", textureSetShaderHash, ")", std::dec,
                detail.empty() ? "\n  (no attributable tier diff captured)" : detail.c_str(),
                "\n  Replacements anchored on the previous hash no longer match this draw."));
            }

            prev.valid = true;
            prev.materialHash = materialHash;
            prev.textureSetHash = textureSetHash;
            prev.constantsHash = psConstsHash;
            prev.constantsExcluded = constantsExcluded;
            prev.samplerCount = micDiagSamplerCount;
            for (uint32_t i = 0; i < micDiagSamplerCount; i++) {
              prev.samplers[i] = micDiagSamplers[i];
            }
            prev.constantCount = curConstantCount;
            for (uint32_t i = 0; i < curConstantCount; i++) {
              prev.constants[i] = curConstants[i];
            }

            // RT-poisoning sweep: a render-target-backed image hash inside the identity
            // makes it unstable across RT recreation (respawn / level load).
            for (uint32_t i = 0; i < micDiagSamplerCount; i++) {
              if (micDiagSamplers[i].isRenderTarget) {
                if (s_ue3MicRtPoisonWarnedFamilies.insert(familyKey).second) {
                  Logger::warn(str::format(
                    "[RTX-MicRtPoisoning] Material identity for family tex=0x", std::hex, primaryTexHash,
                    " seed=0x", shaderIdentitySeed,
                    " includes render-target image hash 0x", micDiagSamplers[i].imageHash,
                    " at material sampler s", std::dec, uint32_t(micDiagSamplers[i].reg),
                    std::hex, " (stable RT descriptor hash 0x", micDiagSamplers[i].descriptorHash,
                    "): materialHash 0x", materialHash, std::dec,
                    " will change whenever the game recreates this render target (respawn/level load),"
                    " breaking replacements anchored on it."));
                }
                break;
              }
            }
          }
        }

        // Constant-colour materials take the first plausible colour among the kept vectors in CTAB name order,
        // which every lightmap compile shares. The lowest name often holds a zero vector.
        if (identityInfo.materialSamplerMask == 0 &&
            !identityInfo.uniformVectorRegisters.empty() &&
            !m_activeDrawCallState.materialData.colorTextures[0].isValid()) {
          const float tintGain = m_frameOptions.ue3ConstantAlbedoTintGain;
          auto tryConstantAlbedo = [&](const uint32_t reg) {
            if (reg >= caps::MaxFloatConstantsPS) {
              return false;
            }
            // A highlight colour holds its value whether the highlight is on or not, so taking
            // it would leave the surface permanently tinted.
            if (std::find(identityInfo.highlightColorRegisters.begin(), identityInfo.highlightColorRegisters.end(), reg) !=
                identityInfo.highlightColorRegisters.end()) {
              return false;
            }
            const Vector4& uniformColor = d3d9State().psConsts.fConsts[reg];
            if (!std::isfinite(uniformColor.x) || !std::isfinite(uniformColor.y) ||
                !std::isfinite(uniformColor.z) || !std::isfinite(uniformColor.w)) {
              return false;
            }
            const float maxComp = std::max({ uniformColor.x, uniformColor.y, uniformColor.z });
            const float minComp = std::min({ uniformColor.x, uniformColor.y, uniformColor.z });
            // reject blacks/negatives (not visible albedo) and HDR-scale values (intensities).
            // Rejecting black also lets a register holding the material's switched-off colour
            // fall through to whichever one holds its real tint.
            if (minComp < 0.0f || maxComp <= 0.01f || maxComp > 8.0f) {
              return false;
            }

            Vector4 albedo = uniformColor;
            if (tintGain > 0.0f) {
              // The register holds a tint UE3 multiplied against baked lighting for brightness,
              // so raw it is near-black once the lightmap is gone. Blend the legacy constant
              // towards the fully saturated hue by the register's own strength, which keeps a
              // ramping tint continuous instead of stepping away from the unlit surface.
              const Vector3 base = LegacyMaterialDefaults::albedoConstant();
              const float weight = std::min(maxComp * tintGain, 1.0f);
              albedo.x = base.x + (uniformColor.x / maxComp - base.x) * weight;
              albedo.y = base.y + (uniformColor.y / maxComp - base.y) * weight;
              albedo.z = base.z + (uniformColor.z / maxComp - base.z) * weight;
            }
            m_activeDrawCallState.materialData.ue3ConstantAlbedo = albedo;
            m_activeDrawCallState.materialData.hasUe3ConstantAlbedo = true;
            return true;
          };
          if (!identityInfo.namedUniformFirstRegistersByNameOrder.empty()) {
            for (const auto& [uniformNameKey, uniformRegister] : identityInfo.namedUniformFirstRegistersByNameOrder) {
              if (tryConstantAlbedo(uniformRegister)) {
                break;
              }
            }
          } else {
            for (const uint32_t reg : identityInfo.uniformVectorRegisters) {
              if (tryConstantAlbedo(reg)) {
                break;
              }
            }
          }
        }

        // Runner Vision: the game fades a strength parameter up on the surfaces it highlights
        // and tints them through constants Remix never evaluates. Carried per surface rather
        // than in the material, so the fade reaches the renderer frame by frame without
        // re-minting the material or moving its identity.
        if (m_frameOptions.ue3HighlightTints) {
          const bool logHighlight = m_frameOptions.ue3LogHighlightTints;
          if (logHighlight) {
            logUe3HighlightPairsOnce(psHash, shaderIdentitySeed, bytecode, identityInfo);
          }

          LegacyMaterialData& materialData = m_activeDrawCallState.materialData;
          const auto& pairs = identityInfo.highlightPairs;
          const Vector4* fConsts = d3d9State().psConsts.fConsts;
          const float glowIntensity = std::max(m_frameOptions.ue3HighlightGlowIntensity, 0.0f);

          // A glow-only pair glows one of the material's textures. It is bound whatever the
          // strength, so the material stays the same through a highlight.
          const auto glowPair = std::find_if(pairs.begin(), pairs.end(), [](const auto& pair) { return pair.tint.glowOnly; });
          if (glowPair != pairs.end() && glowIntensity > 0.0f) {
            const uint32_t stage = uint32_t(glowPair->tint.glowSampler);
            D3D9CommonTexture* const glowTexture =
              stage < caps::MaxTexturesPS ? GetCommonTexture(d3d9State().textures[stage]) : nullptr;
            if (glowTexture != nullptr && glowTexture->GetImage() != nullptr &&
                glowTexture->GetImage()->getHash() != kEmptyHash) {
              if (const Rc<DxvkImageView> view = getRemixSampleView(glowTexture, false); view != nullptr) {
                materialData.ue3HighlightGlowTexture = TextureRef(view);
                materialData.ue3HighlightGlowTextureIsSrgb = d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1;
                materialData.ue3HighlightGlowTextureChannel = glowPair->tint.glowComponent;
              }
            }
          }

          // Nearly every draw holds its strengths at 0, which leaves the surface as it is.
          const bool active = std::any_of(pairs.begin(), pairs.end(), [&](const auto& pair) {
            return pair.tint.scalarReg < caps::MaxFloatConstantsPS && fConsts[pair.tint.scalarReg].x > 0.0f;
          });
          if (active) {
            materialData.updateCachedHash();
            const fast_unordered_set& excluded = *m_frameOptions.ue3HighlightTintExcludedMaterials;
            const XXH64_hash_t colorTextureHash = materialData.getColorTexture().getImageHash();
            const bool isExcluded = !excluded.empty() &&
              (lookupHash(excluded, materialData.getHash()) || lookupHash(excluded, textureSetShaderHash) ||
               lookupHash(excluded, colorTextureHash));
            if (!isExcluded) {
              Ue3HighlightTintDraw draw;
              draw.fConsts = fConsts;
              draw.materialHash = materialData.getHash();
              draw.requireMotion = m_frameOptions.ue3HighlightTintRequireMotion;
              if (draw.requireMotion) {
                const Matrix4& objectToWorld = m_activeDrawCallState.transformData.objectToWorld;
                draw.objectHash = XXH3_64bits_withSeed(&objectToWorld, sizeof(objectToWorld), draw.materialHash);
              }
              draw.glowIntensity = glowIntensity;
              draw.log = logHighlight;

              static fast_unordered_set s_loggedHighlightMaterials;
              const bool logApplied = logHighlight && !lookupHash(s_loggedHighlightMaterials, materialData.getHash());
              std::string appliedLog;
              const bool applied = evaluateUe3HighlightTints(identityInfo, draw, materialData.ue3HighlightTint,
                                                             materialData.ue3HighlightGlow, logApplied ? &appliedLog : nullptr);
              if (applied && logApplied) {
                s_loggedHighlightMaterials.insert(materialData.getHash());
                const Vector3& tint = materialData.ue3HighlightTint;
                const Vector3& glow = materialData.ue3HighlightGlow;
                const TextureRef& glowTexture = materialData.ue3HighlightGlowTexture;
                Logger::info(str::format(
                  "[RTX-Compatibility][UE3-Highlight] Tint applied: materialHash=0x", std::hex, materialData.getHash(),
                  " textureSetShader=0x", textureSetShaderHash, " texture=0x", colorTextureHash, " ps=0x", psHash,
                  glowTexture.isValid() ? str::format(" glowTexture=0x", std::hex, glowTexture.getImageHash()) : std::string(),
                  std::dec, " tint=(", tint.x, ",", tint.y, ",", tint.z, ") glow=(", glow.x, ",", glow.y, ",", glow.z, ") ",
                  appliedLog));
              }
            }
          }
        }

        const Vector3& forcedHighlightTint = m_frameOptions.ue3HighlightDebugForceTint;
        if (forcedHighlightTint.x != 1.0f || forcedHighlightTint.y != 1.0f || forcedHighlightTint.z != 1.0f) {
          m_activeDrawCallState.materialData.ue3HighlightTint = forcedHighlightTint;
        }

        // Opacity-driven fades: UE3 hands its particle colour to the shader as an interpolant Remix
        // never treats as a vertex colour, and fades draws through material constants it never
        // evaluates. Both come from the shader analysis and are carried per draw, like the highlight.
        if (m_frameOptions.ue3ParticleVertexColor || m_frameOptions.ue3MaterialFades) {
          applyUe3MaterialFades(psHash, bytecode, psCommonShader, textureSetShaderHash);
        }

        const float forcedCoverage = m_frameOptions.ue3MaterialFadeDebugForceCoverage;
        if (forcedCoverage >= 0.0f && m_activeDrawCallState.materialData.blendMode.enableBlending) {
          m_activeDrawCallState.materialData.ue3FadeCoverage = std::min(forcedCoverage, 1.0f);
        }

        if (logMicHash) {
          m_activeDrawCallState.materialData.updateCachedHash();
          logUe3MaterialInstanceHashBreakdownOnce(
            m_activeDrawCallState.materialData.getHash(), psHash, shaderIdentitySeed, textureSetHash,
            textureSetShaderHash, psConstsHash, identityInfo, constantsExcluded, micTextureListLog);
        }

      }
    }
  }

  // Texture-less materials have no presence in the texture selection UI; register
  // their material hash as an entry (white thumbnail) so they can be clicked/tagged.
  // Category and replacement lookups already accept material hashes.
  void D3D9Rtx::registerUe3TexturelessMaterial() {
    if (!m_activeDrawCallState.materialData.colorTextures[0].isValid()) {
      const XXH64_hash_t texturelessMaterialHash = m_activeDrawCallState.materialData.getHash();
      static fast_unordered_set s_registeredTexturelessMaterials;
      if (texturelessMaterialHash != kEmptyHash &&
          s_registeredTexturelessMaterials.insert(texturelessMaterialHash).second) {
        m_parent->EmitCs([texturelessMaterialHash](DxvkContext* ctx) {
          const Rc<DxvkImageView> whiteView =
            static_cast<RtxContext*>(ctx)->getResourceManager().getWhiteTexture(ctx);
          if (whiteView != nullptr) {
            ImGUI::AddTexture(texturelessMaterialHash, whiteView, ImGUI::kTextureFlagsDefault);
          }
        });
      }
    }
  }

}
