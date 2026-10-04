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
    // ===== Residual identity churn reporting =====
    // (rtx.d3d9.ue3ReportMicIdentityChurn)
    //
    // Volatile-register classification is a property of the shader, so a material's identity is
    // decided before its first draw is hashed and cannot move afterwards. What the bytecode
    // cannot see is a frame-varying expression that reaches the output colour rather than a
    // coordinate - a time-driven fade or tint is indistinguishable from an authored
    // VectorParameterValue there. Those families still mint more than one identity, so they are
    // reported rather than left to break their anchors quietly. Nothing is excluded as a result:
    // an identity that changes mid-session is exactly what this whole design exists to avoid,
    // and the fix belongs in a config the next session starts from.
    //
    // Keyed on textureSetShaderHash, which already folds in the identity seed and is both the
    // authoring handle (rtx.d3d9.ue3MicConstantIdentityExcludedMaterials) and the second
    // replacement lookup tier.
    //
    // A second identity on one texture set is also what a colour variant looks like the first time
    // its sibling is drawn, so reporting at two would bury the families that matter under benign
    // ones. Measured constant-differentiated sibling sets run to about eight; a frame-varying
    // register passes any bound within a second, so a threshold well clear of the former separates
    // them without needing to understand the values.
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
          if (uniformRegister < caps::MaxFloatConstantsPS && !emit(uniformRegister))
            return;
        }
        return;
      }

      for (const auto& [rangeStart, rangeCount] : identityInfo.constRanges) {
        for (uint32_t r = rangeStart; r < rangeStart + rangeCount; r++) {
          if (r < caps::MaxFloatConstantsPS && !emit(r))
            return;
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
          if (seenBefore != nullptr)
            *seenBefore = true;
          return true;
        }
        // Bounded, as a moving object shows a new placement, and so a new key, every frame.
        if (m_first.size() >= kMaxStillKeys)
          m_first.clear();
        const auto [first, inserted] = m_first.emplace(key, strength);
        if (seenBefore != nullptr)
          *seenBefore = !inserted;
        if (inserted || first->second == strength)
          return false;
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
      if (it != s_cache.end())
        return it->second;

      DxsoMaterialFadeResult result;
      result.failure = DxsoHighlightFailure::NotPixelShader;
      if (bytecode.size() >= 2 * sizeof(uint32_t) && (bytecode.size() % sizeof(uint32_t)) == 0) {
        const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
        if ((tokens[0] & 0xffff0000u) == 0xffff0000u) {
          DxsoProgramInfo programInfo(DxsoProgramTypes::PixelShader, tokens[0] & 0xffu, (tokens[0] >> 8) & 0xffu);
          DxsoDecodeContext decoder(programInfo);
          DxsoCodeIter iter(tokens + 1);
          while (decoder.decodeInstruction(iter)) {
            if (decoder.getCtabInfo().m_size != 0)
              break;
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
      if (!s_logged.insert(XXH3_64bits_withSeed(&particleColorInputRegister, sizeof(particleColorInputRegister), psHash)).second)
        return;

      const std::map<uint32_t, std::string>& names = getUe3PsFloatConstantNames(psHash, bytecode);
      std::string fades;
      for (const DxsoMaterialFade& fade : result.fades)
        fades += str::format(fades.empty() ? "" : ", ", describeUe3MaterialFade(fade, names));
      std::string unproven;
      for (const uint32_t reg : result.unprovenScalarRegs) {
        const auto it = names.find(reg);
        unproven += str::format(unproven.empty() ? "" : ", ", it != names.end() ? it->second : std::string("?"), "@c", reg);
      }
      std::string particleColor;
      if (particleColorInputRegister >= 0)
        particleColor = str::format(" particleColor=v", particleColorInputRegister, ":", describeUe3ParticleColorUses(result.particleColor));
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
      if (!blend.enableBlending || blend.colorBlendOp != VK_BLEND_OP_ADD)
        return rest;
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

  // shader may be null; the view then carries only the bytecode.
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

  // CTAB sampler register -> declared name. Cached per shader hash.
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
          if (decoder.getCtabInfo().m_size != 0)
            break;
        }

        const DxsoCtab& ctab = decoder.getCtabInfo();
        for (const DxsoCtab::Constant& c : ctab.m_constantData) {
          if (c.registerSet != kD3dxRegisterSetSampler || c.registerCount == 0)
            continue;
          const uint32_t end = std::min<uint32_t>(c.registerIndex + c.registerCount, caps::MaxTexturesPS);
          for (uint32_t s = c.registerIndex; s < end; s++) {
            names[s] = c.name;
          }
        }
      }
    }

    return s_ue3PsSamplerNameCache.emplace(psHash, std::move(names)).first->second;
  }

  // CTAB float-constant register -> declared name (UniformScalar_*/UniformVector_*/engine
  // constants), for rtx.d3d9.ue3LogUvAffineDetail diagnostics. Cached per shader hash.
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
          if (decoder.getCtabInfo().m_size != 0)
            break;
        }

        const DxsoCtab& ctab = decoder.getCtabInfo();
        for (const DxsoCtab::Constant& c : ctab.m_constantData) {
          if (c.registerSet != kD3dxRegisterSetFloat4 || c.registerCount == 0)
            continue;
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

  // Bounded, for the per-family drift sample that is retained for every family while replacement
  // diagnostics are on.
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

  // Unbounded: each family takes at most two, and a cap would silently drop the register that
  // moved on any shader declaring more uniforms than the cap - leaving a report that says a
  // family churned but not which value did, which is the only part worth reading.
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
    if (textureSetShaderHash == kEmptyHash || constantsHash == kEmptyHash)
      return;

    Ue3MicChurnReport& report = s_ue3MicChurnByMaterialFamily[textureSetShaderHash];
    if (report.warned)
      return;

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
        if (first.reg != now.reg)
          continue;
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
    if (!s_loggedMaterialInstanceHashes.insert(materialHash).second)
      return;

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

  // Reused XXH3 streaming state: created once per thread instead of heap
  // allocating/freeing a state per hash operation on the per-draw path. The state is
  // fully reset before each use, so digests are identical to a fresh state.
  XXH3_state_t* getThreadLocalXxh3State() {
    static thread_local XXH3_state_t* const state = XXH3_createState();
    return state;
  }

  // Returns kEmptyHash when ranges is empty: without CTAB info, a raw register-range
  // fallback would fold per-view/per-mesh constants into the hash.
  XXH64_hash_t hashUe3MaterialConstants(
      const Vector4* fConsts,
      const Ue3MaterialConstRanges& ranges) {
    if (ranges.empty())
      return kEmptyHash;

    XXH3_state_t* const state = getThreadLocalXxh3State();
    if (state == nullptr)
      return kEmptyHash;
    XXH3_64bits_reset(state);

    bool anyRegisterHashed = false;
    for (const auto& [start, count] : ranges) {
      if (start + count > caps::MaxFloatConstantsPS)
        continue;
      XXH3_64bits_update(state, &fConsts[start], count * sizeof(Vector4));
      anyRegisterHashed = true;
    }

    return anyRegisterHashed ? XXH3_64bits_digest(state) : kEmptyHash;
  }

  // Permutation-invariant constants identity: streams (name key, leading element value) of
  // each named Uniform* constant, in name order. fxc trims each uniform array to the
  // elements the permutation actually references (the directional lightmap path can
  // reference more expression elements than the simple path, e.g. an unreferenced specular
  // expression), so higher elements are not comparable across lightmap policy permutations.
  // The leading element is always within the reported range and the engine uploads the same
  // expression value to it in every permutation. Name keys keep the stream aligned even if
  // a permutation strips an entire unreferenced uniform.
  XXH64_hash_t hashUe3MaterialConstantsByNameOrder(
      const Vector4* fConsts,
      const std::vector<std::pair<XXH64_hash_t, uint32_t>>& namedUniformFirstRegistersByNameOrder) {
    if (namedUniformFirstRegistersByNameOrder.empty())
      return kEmptyHash;

    XXH3_state_t* const state = getThreadLocalXxh3State();
    if (state == nullptr)
      return kEmptyHash;
    XXH3_64bits_reset(state);

    bool anyRegisterHashed = false;
    for (const auto& [nameKey, reg] : namedUniformFirstRegistersByNameOrder) {
      if (reg >= caps::MaxFloatConstantsPS)
        continue;
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

  // Multiplies each applying pair's tint into `tint` and adds its glow to `glow`; returns whether
  // any pair applied. appliedLog, when given, receives the live values of those that did.
  bool evaluateUe3HighlightTints(const Ue3PsMaterialIdentityInfo& info, const Ue3HighlightTintDraw& draw,
                                        Vector3& tint, Vector3& glow, std::string* appliedLog) {
    static Ue3StrengthMotion s_objectMotion;
    static Ue3StrengthMotion s_instanceMotion;
    static fast_unordered_set s_heldBackLogged;

    bool applied = false;
    for (const Ue3PsMaterialIdentityInfo::HighlightPair& pair : info.highlightPairs) {
      const uint32_t strengthReg = pair.tint.scalarReg;
      if (strengthReg >= caps::MaxFloatConstantsPS)
        continue;
      const float strength = draw.fConsts[strengthReg].x;
      if (!std::isfinite(strength) || strength <= 0.0f)
        continue;

      // Runner Vision fades the strength on the instances it creates (TdLOIAddOnObject), so a
      // highlight shows it moving; an object that holds it still is authored in the tint colour,
      // which Remix leaves to its material. Tracked per object, so one painted in the highlight
      // colour stays untinted when an identical one is highlighted. An object at a placement not
      // seen before - a moving one, every frame - follows its material instance.
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
        if (reg < 0 || uint32_t(reg) >= caps::MaxFloatConstantsPS)
          continue;
        const float value = draw.fConsts[reg][pair.tint.colorComponent[lane]];
        color[lane] = std::isfinite(value) ? std::max(value, 0.0f) : 1.0f;
        tint[lane] *= 1.0f + s * (color[lane] - 1.0f);
      }
      for (uint32_t lane = 0; lane < 3; lane++)
        glow[lane] += pair.tint.glowCoefficient[lane] * s * draw.glowIntensity;
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
        info.highlightFailure == DxsoHighlightFailure::None)
      return;
    static fast_unordered_set s_logged;
    if (!s_logged.insert(psHash).second)
      return;

    std::string tints;
    for (const Ue3PsMaterialIdentityInfo::HighlightPair& pair : info.highlightPairs)
      tints += str::format(tints.empty() ? "" : ", ", describeUe3HighlightPair(pair));
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
          if (element.Usage == D3DDECLUSAGE_TEXCOORD && element.UsageIndex == 3 && element.Type == D3DDECLTYPE_FLOAT4)
            particleColorTexcoord = 3;
        }
      }
      const DxsoIsgn& isgn = pixelShader->GetIsgn();
      for (uint32_t i = 0; i < isgn.elemCount; i++) {
        if (isgn.elems[i].semantic.usage == DxsoUsage::Texcoord && isgn.elems[i].semantic.usageIndex == particleColorTexcoord)
          particleColorInputRegister = int32_t(isgn.elems[i].regNumber);
      }
    }

    const Ue3BlendRest blendRest = ue3BlendRest(materialData.blendMode);
    const bool canFade = m_frameOptions.ue3MaterialFades && (blendRest.alphaFades || blendRest.colorFades);
    if (particleColorInputRegister < 0 && !canFade && !logFades)
      return;

    const DxsoMaterialFadeResult& result = getOrAnalyzeUe3MaterialFades(psHash, bytecode, particleColorInputRegister);
    if (logFades)
      logUe3MaterialFadesOnce(psHash, bytecode, particleColorInputRegister, result);
    if (!result.analyzed)
      return;

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
        if (alphaLane ? !blendRest.alphaFades : !blendRest.colorFades)
          continue;
        if (lastApplied != nullptr && lastApplied->reg == fade.reg && lastApplied->component == fade.component)
          continue;
        const std::optional<float> fadeCoverage =
          dxsoMaterialFadeCoverage(fade, alphaLane ? blendRest.alphaRest : blendRest.colorRest, constants, constantCount);
        if (names != nullptr) {
          fadeLog += str::format(fadeLog.empty() ? "" : "; ", describeUe3MaterialFade(fade, *names),
                                 " value=", fade.reg < constantCount ? constants[fade.reg * 4u + (fade.component & 3u)] : 0.0f,
                                 fadeCoverage ? str::format(" coverage=", *fadeCoverage) : std::string(" not at rest"));
        }
        if (!fadeCoverage)
          continue;
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

}
