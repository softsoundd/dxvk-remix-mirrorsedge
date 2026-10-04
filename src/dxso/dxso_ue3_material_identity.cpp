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
#include "dxso_ue3_material_identity.h"

#include <algorithm>
#include <array>
#include <map>

#include "dxso_code.h"
#include "dxso_color_terms.h"
#include "dxso_ctab.h"
#include "dxso_sampler_inference.h"
#include "../util/util_string.h"

namespace dxvk {

  // Names and register classes only: register indices and fxc's used element counts both vary
  // between the lightmap-policy compiles of a material.
  static std::string makeUe3SignatureEntry(const std::string& lowerName, const uint16_t registerSet) {
    return str::format(lowerName, "\x01", registerSet);
  }

  // Sorts entries in place.
  static XXH64_hash_t hashUe3SignatureEntries(std::vector<std::string>& entries) {
    if (entries.empty()) {
      return kEmptyHash;
    }
    std::sort(entries.begin(), entries.end());
    std::string serialized;
    for (const std::string& entry : entries) {
      serialized += entry;
      serialized += '\x02';
    }
    return XXH3_64bits(serialized.data(), serialized.size());
  }

  Ue3PsMaterialIdentityInfo parseUe3PsMaterialIdentityFromCtab(
      const DxsoShaderView& pixelShader,
      const bool detectVolatileConstants) {
    Ue3PsMaterialIdentityInfo info;
    if (pixelShader.tokens == nullptr) {
      return info;
    }

    const uint32_t* tokens = pixelShader.tokens;
    const size_t tokenCount = pixelShader.tokenCount;
    const uint32_t headerToken = tokens[0];
    const uint32_t headerTypeMask = headerToken & 0xffff0000u;
    if (headerTypeMask != 0xffff0000u) {
      return info;
    }

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
    if (ctab.m_size == 0 || ctab.m_constantData.empty()) {
      return info;
    }

    info.hasCtab = true;

    auto startsWith = [](const std::string& s, const char* prefix) {
      return s.rfind(prefix, 0) == 0;
    };

    struct DeclaredSampler {
      std::string name;
      uint32_t registerIndex = 0;
      uint32_t registerCount = 0;
    };
    struct DeclaredUniform {
      std::string name;
      uint16_t registerSet = 0;
      uint32_t registerIndex = 0;
      uint32_t registerCount = 0;
    };

    std::vector<DeclaredSampler> declaredSamplers;
    std::vector<DeclaredUniform> declaredUniforms;
    // Material and engine samplers alike: a frame-varying uniform leaves identity whichever
    // sampler's coordinate it drives.
    std::vector<uint32_t> allDeclaredSamplerRegisters;

    // See "Identity" in UE3Compatibility.md.
    DxsoColorTermInputs colorTermInputs;
    colorTermInputs.trackLiterals = true;

    for (const DxsoCtab::Constant& c : ctab.m_constantData) {
      const std::string name = toLowerAscii(c.name);

      if (c.registerSet == kD3dxRegisterSetSampler) {
        if (c.registerCount != 0) {
          const uint32_t declaredEnd = std::min<uint32_t>(c.registerIndex + c.registerCount, kDxsoMaxPsSamplers);
          for (uint32_t s = c.registerIndex; s < declaredEnd; s++) {
            allDeclaredSamplerRegisters.push_back(s);
            if (name.find("lightmap") != std::string::npos) {
              colorTermInputs.lightmapSamplerMask |= 1u << s;
            }
          }
        }
        // Only the material translator's numbered names are material parameters; lightmap and
        // engine samplers are named otherwise.
        if (c.registerCount != 0 &&
            (startsWith(name, "texture2d_") || startsWith(name, "texturecube_") || startsWith(name, "texture3d_"))) {
          declaredSamplers.push_back({ name, c.registerIndex, c.registerCount });
          const uint32_t declaredEnd = std::min<uint32_t>(c.registerIndex + c.registerCount, kDxsoMaxPsSamplers);
          for (uint32_t s = c.registerIndex; s < declaredEnd; s++) {
            colorTermInputs.trackedSamplerMask |= 1u << s;
          }
        }
        continue;
      }

      if (c.registerSet > 2u || c.registerCount == 0) {
        continue;
      }
      if (c.registerIndex + c.registerCount > kDxsoMaxPsFloatConstants) {
        continue;
      }

      // BasePassPixelShader.usf scales unlit and dynamic diffuse by AmbientColorAndSkyFactor.rgb,
      // and sky-lit diffuse by UpperSkyColor and LowerSkyColor.
      if (name.find("ambientcolorandskyfactor") != std::string::npos ||
          name.find("upperskycolor") != std::string::npos ||
          name.find("lowerskycolor") != std::string::npos) {
        for (uint32_t r = c.registerIndex; r < c.registerIndex + c.registerCount && r < kDxsoColorTermMaxConstRegs; r++) {
          colorTermInputs.lightingConstRegs.set(r);
        }
        continue;
      }

      const bool isUniformVector = name.find("uniformvector_") != std::string::npos;
      const bool isUniformScalar = name.find("uniformscalar_") != std::string::npos;
      if (!isUniformVector && !isUniformScalar) {
        continue;
      }
      // Every vector, kept in identity or not: a material without textures reads its colour from one.
      if (isUniformVector) {
        info.uniformVectorRegisters.push_back(c.registerIndex);
        if (c.registerIndex < kDxsoColorTermMaxConstRegs) {
          colorTermInputs.trackedConstRegs.set(c.registerIndex);
        }
      }
      declaredUniforms.push_back({ name, c.registerSet, c.registerIndex, c.registerCount });
    }

    const DxsoColorTermResult colorTerms =
      analyzeDxsoColorTerms(tokens, tokenCount, colorTermInputs);

    std::vector<std::pair<std::string, uint32_t>> uniformsByName;
    std::vector<std::string> samplerSignatureEntries;
    std::vector<std::string> keptSamplerNames, excludedSamplerNames;

    // Shared by the lighting-input fallback and the volatile-register fallback below.
    std::map<uint32_t, PsSamplerTexcoordInference> samplerInference;
    if (pixelShader.info != nullptr && pixelShader.isgn != nullptr) {
      for (const uint32_t samplerRegister : allDeclaredSamplerRegisters) {
        samplerInference.emplace(samplerRegister, inferPixelShaderTexcoordForSampler(pixelShader, samplerRegister));
      }
    }

    // Registers driving a sampler's coordinate: texture transforms, often animated, rather than
    // authored parameters. See the frame-varying constants under "Material identity and
    // replacement anchor stability" in UE3Compatibility.md.
    std::vector<uint32_t> volatileRegisters;
    if (detectVolatileConstants) {
      auto markVolatile = [&](const int32_t reg) {
        if (reg < 0 || uint32_t(reg) >= kDxsoMaxPsFloatConstants) {
          return;
        }
        const uint32_t value = uint32_t(reg);
        if (std::find(volatileRegisters.begin(), volatileRegisters.end(), value) == volatileRegisters.end()) {
          volatileRegisters.push_back(value);
        }
      };
      // The colour-term analysis tracks the dependency set per register lane. The inference's
      // per-register set, which also counts whatever fxc packed into spare lanes, is the
      // fallback for bytecode the analysis cannot read.
      if (colorTerms.analyzed) {
        for (const uint32_t samplerRegister : allDeclaredSamplerRegisters) {
          if (samplerRegister >= kDxsoColorTermMaxSamplers) {
            continue;
          }
          const auto& regs = colorTerms.samplerCoordConstRegs[samplerRegister];
          for (uint32_t reg = 0; reg < kDxsoMaxPsFloatConstants && reg < kDxsoColorTermMaxConstRegs; reg++) {
            if (regs.test(reg)) {
              markVolatile(int32_t(reg));
            }
          }
        }
      } else {
        for (const auto& [samplerRegister, inferred] : samplerInference) {
          for (const uint32_t reg : inferred.coordConstRegs) {
            markVolatile(int32_t(reg));
          }
        }
      }
    }

    // Samplers set aside by the role rule below, kept only when nothing else identifies the
    // material.
    uint32_t fallbackSamplerMask = 0;
    std::vector<std::tuple<std::string, XXH64_hash_t, uint32_t>> fallbackSamplersByNameOrder;
    std::vector<std::string> fallbackSignatureEntries, fallbackSamplerNames;

    for (const DeclaredSampler& sampler : declaredSamplers) {
      const uint32_t end = std::min<uint32_t>(sampler.registerIndex + sampler.registerCount, kDxsoMaxPsSamplers);
      bool anyKept = false, anyFallback = false;
      for (uint32_t s = sampler.registerIndex; s < end; s++) {
        // Array elements get names of their own; material samplers are scalar in practice.
        const std::string samplerName =
          sampler.registerCount > 1u ? str::format(sampler.name, "[", s - sampler.registerIndex, "]") : sampler.name;
        const XXH64_hash_t samplerNameKey = XXH3_64bits(samplerName.data(), samplerName.size());

        bool keep = true, keepIfNothingElse = false;
        if (colorTerms.analyzed) {
          const DxsoColorTermRole role = classifyDxsoColorTermSource(
            colorTerms.samplerTerms[s], colorTerms.samplerReach[s], colorTerms.hasLightConstTerm);
          // See the Opacity and Coordinate roles under "Identity" in UE3Compatibility.md.
          keep = role == DxsoColorTermRole::Color || role == DxsoColorTermRole::Opacity;
          keepIfNothingElse = role == DxsoColorTermRole::Coordinate;
          if (role == DxsoColorTermRole::LightingOnly || role == DxsoColorTermRole::Unused) {
            info.lightingInputSamplerMask |= (1u << s);
          }
        } else {
          const auto inferenceIt = samplerInference.find(s);
          keep = inferenceIt == samplerInference.end() || !isUe3LightingInputSampler(inferenceIt->second);
          keepIfNothingElse = !keep;
        }

        if (!keep) {
          excludedSamplerNames.push_back(samplerName);
          if (keepIfNothingElse) {
            anyFallback = true;
            fallbackSamplerMask |= (1u << s);
            fallbackSamplersByNameOrder.emplace_back(samplerName, samplerNameKey, s);
            fallbackSamplerNames.push_back(samplerName);
          }
          continue;
        }
        keptSamplerNames.push_back(samplerName);
        anyKept = true;
        info.materialSamplerMask |= (1u << s);
        info.materialSamplersByNameOrder.emplace_back(samplerName, samplerNameKey, s);
      }
      if (anyKept) {
        samplerSignatureEntries.push_back(makeUe3SignatureEntry(sampler.name, kD3dxRegisterSetSampler));
      }
      if (anyFallback) {
        fallbackSignatureEntries.push_back(makeUe3SignatureEntry(sampler.name, kD3dxRegisterSetSampler));
      }
    }

    // An empty set would merge every such material onto one identity.
    if (info.materialSamplerMask == 0 && fallbackSamplerMask != 0) {
      info.materialSamplerMask = fallbackSamplerMask;
      info.materialSamplersByNameOrder = std::move(fallbackSamplersByNameOrder);
      samplerSignatureEntries = std::move(fallbackSignatureEntries);
      std::vector<std::string> stillExcluded;
      for (const std::string& name : excludedSamplerNames) {
        if (std::find(fallbackSamplerNames.begin(), fallbackSamplerNames.end(), name) == fallbackSamplerNames.end()) {
          stillExcluded.push_back(name);
        }
      }
      excludedSamplerNames = std::move(stillExcluded);
      keptSamplerNames = std::move(fallbackSamplerNames);
    }

    std::vector<std::string> keptUniformNames, volatileUniformNames, roleExcludedUniformNames;
    std::vector<uint32_t> droppedUniformRegisters;

    // Constants are all that tells textureless materials apart, and merging them is worse than
    // an identity that moves.
    const bool keepEveryUniform = info.materialSamplerMask == 0;

    // The hash streams a uniform's leading register, but the animated element can be anywhere
    // in its range.
    auto uniformRangeIsVolatile = [&](const DeclaredUniform& uniform) {
      const uint32_t end = uniform.registerIndex + uniform.registerCount;
      for (uint32_t r = uniform.registerIndex; r < end; r++) {
        if (std::find(volatileRegisters.begin(), volatileRegisters.end(), r) != volatileRegisters.end()) {
          return true;
        }
      }
      return false;
    };

    for (const DeclaredUniform& uniform : declaredUniforms) {
      // Vectors only; see "Constants tier" in UE3Compatibility.md.
      if (uniform.name.find("uniformscalar_") != std::string::npos) {
        continue;
      }

      if (!keepEveryUniform && uniformRangeIsVolatile(uniform)) {
        volatileUniformNames.push_back(uniform.name);
        // Only what left identity: an engine constant driving a coordinate is flagged too.
        droppedUniformRegisters.push_back(uniform.registerIndex);
        continue;
      }

      // A lighting-only vector exists in the directional compile alone, and a coordinate-only one
      // is a texture transform whatever the volatile scan found.
      if (colorTerms.analyzed && uniform.registerIndex < kDxsoColorTermMaxConstRegs) {
        const DxsoColorTermRole role = classifyDxsoColorTermSource(
          colorTerms.constTerms[uniform.registerIndex], colorTerms.constReach[uniform.registerIndex],
          colorTerms.hasLightConstTerm);
        if (role == DxsoColorTermRole::LightingOnly || role == DxsoColorTermRole::Unused ||
            role == DxsoColorTermRole::Coordinate) {
          roleExcludedUniformNames.push_back(uniform.name);
          droppedUniformRegisters.push_back(uniform.registerIndex);
          continue;
        }
      }

      keptUniformNames.push_back(uniform.name);
      uniformsByName.emplace_back(uniform.name, uniform.registerIndex);
      info.constRanges.emplace_back(uniform.registerIndex, uniform.registerCount);
    }

    // `def` components reaching the colour output unlit, which every compile shares.
    std::vector<std::string> signatureLiteralEntries;
    if (colorTerms.analyzed) {
      for (const DxsoColorTermLiteral& literal : colorTerms.literals) {
        if (dxsoColorTermSetHasUnlitTerm(literal.terms)) {
          signatureLiteralEntries.push_back(str::format("literal:", std::hex, literal.bits));
        }
      }
    }
    std::sort(signatureLiteralEntries.begin(), signatureLiteralEntries.end());

    if (!excludedSamplerNames.empty() || !volatileUniformNames.empty() || !roleExcludedUniformNames.empty()) {
      auto join = [](const std::vector<std::string>& names) {
        std::string out;
        for (const std::string& name : names) {
          if (!out.empty()) {
            out += ',';
          }
          out += name;
        }
        return out.empty() ? std::string("-") : out;
      };
      info.identitySummary = str::format(
        "samplers kept=[", join(keptSamplerNames), "] excludedByRole=[",
        join(excludedSamplerNames), "] uniforms kept=[", join(keptUniformNames),
        "] excludedAsVolatile=[", join(volatileUniformNames),
        "] excludedByRole=[", join(roleExcludedUniformNames), "]",
        colorTerms.analyzed ? "" : " colorTerms=unanalyzed");
    }

    std::sort(droppedUniformRegisters.begin(), droppedUniformRegisters.end());
    info.volatileUniformRegisters = std::move(droppedUniformRegisters);

    std::sort(info.uniformVectorRegisters.begin(), info.uniformVectorRegisters.end());

    std::sort(uniformsByName.begin(), uniformsByName.end());
    info.namedUniformFirstRegistersByNameOrder.reserve(uniformsByName.size());
    for (const auto& [uniformName, uniformRegister] : uniformsByName) {
      info.namedUniformFirstRegistersByNameOrder.emplace_back(
        XXH3_64bits(uniformName.data(), uniformName.size()), uniformRegister);
    }
    // Name order is the same in every compile, including one that strips an unreferenced symbol.
    std::sort(info.materialSamplersByNameOrder.begin(), info.materialSamplersByNameOrder.end());

    // See "Identity" in UE3Compatibility.md. Engine symbols differ between the compiles, so only
    // material names count.
    if (!samplerSignatureEntries.empty()) {
      info.canonicalShaderSignature = hashUe3SignatureEntries(samplerSignatureEntries);
    } else if (colorTerms.analyzed) {
      std::vector<std::string> texturelessEntries;
      for (const std::string& name : keptUniformNames) {
        texturelessEntries.push_back(makeUe3SignatureEntry(name, kD3dxRegisterSetFloat4));
      }
      texturelessEntries.insert(texturelessEntries.end(), signatureLiteralEntries.begin(), signatureLiteralEntries.end());
      if (!texturelessEntries.empty()) {
        texturelessEntries.push_back("ue3:textureless");
        info.canonicalShaderSignature = hashUe3SignatureEntries(texturelessEntries);
        info.texturelessSignature = true;
      }
    }

    auto mergeConstRanges = [](Ue3MaterialConstRanges& ranges) {
      if (ranges.empty()) {
        return;
      }
      std::sort(ranges.begin(), ranges.end());
      Ue3MaterialConstRanges merged;
      for (const auto& r : ranges) {
        if (merged.empty() || r.first > merged.back().first + merged.back().second) {
          merged.push_back(r);
        } else {
          const uint32_t end = std::max(merged.back().first + merged.back().second, r.first + r.second);
          merged.back().second = end - merged.back().first;
        }
      }
      ranges = std::move(merged);
    };
    mergeConstRanges(info.constRanges);

    // Runner Vision highlight tints. A shader without a material scalar cannot carry one.
    const DxsoHighlightInputs highlightInputs = dxsoHighlightInputsFromUe3Ctab(ctab);
    if (highlightInputs.scalarRegs.any()) {
      const DxsoHighlightResult highlight =
        analyzeDxsoHighlightTints(tokens, tokenCount, highlightInputs);
      info.highlightFailure = highlight.failure;
      info.highlightUnprovenScalars = highlight.unprovenScalarRegs;
      auto constantName = [&ctab](const uint32_t reg) {
        for (const DxsoCtab::Constant& c : ctab.m_constantData) {
          if (c.registerSet == kD3dxRegisterSetFloat4 && reg >= c.registerIndex && reg < c.registerIndex + c.registerCount) {
            return c.name;
          }
        }
        return std::string();
      };
      for (const DxsoHighlightPair& tint : highlight.pairs) {
        Ue3PsMaterialIdentityInfo::HighlightPair pair;
        pair.tint = tint;
        pair.strengthName = constantName(tint.scalarReg);
        pair.colorName = constantName(uint32_t(tint.colorReg[0]));
        const std::string key = pair.strengthName + '\x01' + pair.colorName;
        pair.nameKey = XXH3_64bits(key.data(), key.size());
        info.highlightPairs.push_back(std::move(pair));
        for (const int32_t reg : tint.colorReg) {
          if (std::find(info.highlightColorRegisters.begin(), info.highlightColorRegisters.end(), uint32_t(reg)) ==
              info.highlightColorRegisters.end()) {
            info.highlightColorRegisters.push_back(uint32_t(reg));
          }
        }
      }
    }
    return info;
  }

}
