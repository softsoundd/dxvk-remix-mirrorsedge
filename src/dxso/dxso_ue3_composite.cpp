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
#include "dxso_ue3_composite.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>

#include "dxso_code.h"
#include "dxso_color_terms.h"
#include "dxso_ctab.h"
#include "dxso_decoder.h"
#include "dxso_symbolic_eval.h"
#include "../util/util_string.h"

namespace dxvk {

  const char* describeDxsoUe3CompositeRefusal(const DxsoUe3CompositeRefusal refusal) {
    switch (refusal) {
    case DxsoUe3CompositeRefusal::None: return "none";
    case DxsoUe3CompositeRefusal::NotAnalyzed: return "not an analysable ps_2_0+ shader with a constant table";
    case DxsoUe3CompositeRefusal::NoColorLayer: return "no material texture reaches the colour output";
    case DxsoUe3CompositeRefusal::EngineSampler: return "an engine texture reaches the output";
    case DxsoUe3CompositeRefusal::VolumeLayer: return "a volume texture reaches the output";
    case DxsoUe3CompositeRefusal::VertexLightmap:
      return "directional vertex lightmap compile, whose diffuse arrives through several lighting interpolants";
    case DxsoUe3CompositeRefusal::VertexColor: return "the colour reads a vertex colour";
    case DxsoUe3CompositeRefusal::MaskedWithoutCoverage:
      return "clip()s, but its alpha output is not an interpolant the bake can mark coverage with";
    }
    return "unknown";
  }

  const char* describeDxsoUe3CompositeLighting(const DxsoUe3CompositeLighting lighting) {
    switch (lighting) {
    case DxsoUe3CompositeLighting::AmbientConstant: return "ambient constant";
    case DxsoUe3CompositeLighting::SimpleLightmap: return "simple lightmap";
    case DxsoUe3CompositeLighting::SimpleVertexLightmap: return "simple vertex lightmap";
    case DxsoUe3CompositeLighting::Unlit: return "unlit";
    }
    return "unknown";
  }

  namespace {

    bool setHasFactor(const DxsoColorTermSet set, const uint8_t factors) {
      for (uint32_t signature = 0; signature < kDxsoColorTermSignatureCount; signature++) {
        if (((set >> signature) & 1u) != 0 && (signature & factors) != 0) {
          return true;
        }
      }
      return false;
    }

    // Returns whether a term carries the lighting constant and no other factor, as UE3's DiffuseColor * AmbientColor does.
    // Only the compiles that are not simple-lightmapped emit one. Sky light terms carry the view as well.
    bool setHasPureAmbientTerm(const DxsoColorTermSet set) {
      return ((set >> uint32_t(DxsoColorTermFactor_LightConst)) & 1u) != 0;
    }

    bool setHasLitTerm(const DxsoColorTermSet set) {
      constexpr uint8_t kLit = DxsoColorTermFactor_Lightmap | DxsoColorTermFactor_LightConst | DxsoColorTermFactor_Interpolant;
      return setHasFactor(set, kLit);
    }

    // Returns whether the opcode's destination token names the register it writes.
    bool writesDestination(const DxsoOpcode op) {
      switch (op) {
      case DxsoOpcode::If:
      case DxsoOpcode::Ifc:
      case DxsoOpcode::BreakC:
      case DxsoOpcode::BreakP:
      case DxsoOpcode::Call:
      case DxsoOpcode::CallNz:
      case DxsoOpcode::Rep:
      case DxsoOpcode::Loop:
      case DxsoOpcode::TexKill:
      case DxsoOpcode::Dcl:
      case DxsoOpcode::Def:
      case DxsoOpcode::DefI:
      case DxsoOpcode::DefB:
        return false;
      default:
        return getDxsoSourceOperandCount(op) > 0;
      }
    }

    bool startsWith(const std::string& s, const char* prefix) {
      return s.rfind(prefix, 0) == 0;
    }

    // Returns the sampler as a mask layer when the colour output reads it only through one channel of one
    // read, as a polynomial of degree 1 or 2. Nothing the evaluation cannot expand may be computed from it,
    // and none of it may reach the alpha.
    std::optional<DxsoUe3CompositeMaskLayer> findMask(const DxsoShaderView& pixelShader, const DxsoProgramInfo& programInfo,
                                                      const uint32_t sampler) {
      using dxso_symbolic::SymbolKind;
      using dxso_symbolic::Term;

      // With no material parameters or lighting declared, everything that does not read the sampler folds.
      const DxsoHighlightInputs inputs;
      dxso_symbolic::EvaluatorOptions options;
      options.modulatorSamplerMask = 1u << sampler;
      const auto evaluator = std::make_unique<dxso_symbolic::Evaluator>(inputs, options, programInfo);
      evaluator->run(pixelShader.tokens, pixelShader.tokenCount);
      if (evaluator->failure() != DxsoHighlightFailure::None) {
        return std::nullopt;
      }

      std::vector<uint16_t> samples;
      for (size_t s = 0; s < evaluator->symbolCount(); s++) {
        const dxso_symbolic::Symbol& symbol = evaluator->symbol(uint16_t(s));
        if (symbol.kind == SymbolKind::Sample && symbol.reg == sampler) {
          samples.push_back(uint16_t(s));
        }
      }

      int32_t mask = -1;
      uint32_t maxDegree = 0;
      for (uint32_t lane = 0; lane < 4; lane++) {
        if (!evaluator->outputWritten(lane)) {
          continue;
        }
        for (const Term& t : evaluator->output(lane)) {
          uint32_t degree = 0;
          for (uint32_t i = 0; i < t.count; i++) {
            const uint16_t s = t.syms[i];
            const dxso_symbolic::Symbol& symbol = evaluator->symbol(s);
            if (symbol.kind == SymbolKind::Sample && symbol.reg == sampler) {
              if (lane == 3 || (mask >= 0 && s != uint16_t(mask))) {
                return std::nullopt;
              }
              mask = s;
              degree++;
            } else if (symbol.kind == SymbolKind::Atom &&
                       std::any_of(samples.begin(), samples.end(),
                                   [&](const uint16_t sample) { return evaluator->atomDependsOn(s, sample); })) {
              return std::nullopt;
            }
          }
          if (degree > 2) {
            return std::nullopt;
          }
          maxDegree = std::max(maxDegree, degree);
        }
      }
      if (mask < 0) {
        return std::nullopt;
      }
      DxsoUe3CompositeMaskLayer layer;
      layer.sampler = uint8_t(sampler);
      layer.channel = evaluator->symbol(uint16_t(mask)).component;
      layer.degree = uint8_t(maxDegree);
      return layer;
    }

  }

  DxsoUe3CompositeShaderInfo analyzeDxsoUe3CompositeShader(const DxsoShaderView& pixelShader) {
    DxsoUe3CompositeShaderInfo info;
    if (pixelShader.tokens == nullptr || pixelShader.tokenCount < 2) {
      return info;
    }
    const uint32_t headerToken = pixelShader.tokens[0];
    if ((headerToken & 0xffff0000u) != 0xffff0000u) {
      return info;
    }
    const uint32_t majorVersion = (headerToken >> 8) & 0xffu;
    const uint32_t minorVersion = headerToken & 0xffu;
    if (majorVersion < 2) {
      return info;
    }
    const DxsoProgramInfo programInfo(DxsoProgramTypes::PixelShader, minorVersion, majorVersion);

    std::array<DxsoTextureType, kDxsoColorTermMaxSamplers> samplerTypes;
    samplerTypes.fill(DxsoTextureType::Texture2D);
    uint32_t declaredSamplerMask = 0;
    bool readsColor0 = false;
    bool readsColor1 = false;
    int32_t alphaInputRegister = -1;

    DxsoDecodeContext decoder(programInfo);
    DxsoCodeIter iter(pixelShader.tokens + 1);
    while (decoder.decodeInstruction(iter)) {
      const DxsoInstructionContext& ctx = decoder.getInstructionContext();
      const DxsoOpcode op = ctx.instruction.opcode;
      if (op == DxsoOpcode::Dcl) {
        if (ctx.dst.id.type == DxsoRegisterType::Sampler) {
          if (ctx.dst.id.num < kDxsoColorTermMaxSamplers) {
            samplerTypes[ctx.dst.id.num] = ctx.dcl.textureType;
            declaredSamplerMask |= 1u << ctx.dst.id.num;
          }
        } else if (ctx.dst.id.type == DxsoRegisterType::Input && ctx.dcl.semantic.usage == DxsoUsage::Color) {
          readsColor0 |= ctx.dcl.semantic.usageIndex == 0;
          readsColor1 |= ctx.dcl.semantic.usageIndex == 1;
        }
        continue;
      }
      if (op == DxsoOpcode::TexKill) {
        info.hasTexkill = true;
        continue;
      }
      if (!writesDestination(op) ||
          ctx.dst.id.type != DxsoRegisterType::ColorOut || ctx.dst.id.num != 0 || !ctx.dst.mask[3]) {
        continue;
      }
      // The last write to oC0.w decides what the alpha output carries.
      const DxsoRegister& source = ctx.src[0];
      const bool plainInputRead =
        op == DxsoOpcode::Mov && !ctx.dst.saturate && ctx.dst.shift == 0 &&
        source.id.type == DxsoRegisterType::Input && !source.hasRelative &&
        source.modifier == DxsoRegModifier::None && source.swizzle[3] == 3u;
      alphaInputRegister = plainInputRead ? int32_t(source.id.num) : -1;
    }

    const DxsoCtab& ctab = decoder.getCtabInfo();
    if (ctab.m_size == 0 || ctab.m_constantData.empty()) {
      return info;
    }

    // See "Identity" in UE3Compatibility.md for the classes the colour-term analysis assigns.
    DxsoColorTermInputs inputs;
    inputs.trackLiterals = true;
    uint32_t materialSamplerMask = 0;
    uint32_t lightmapLikeSamplerMask = 0;
    std::array<std::string, kDxsoColorTermMaxSamplers> samplerNames;

    for (const DxsoCtab::Constant& c : ctab.m_constantData) {
      if (c.registerCount == 0) {
        continue;
      }
      const std::string name = toLowerAscii(c.name);
      if (c.registerSet == kD3dxRegisterSetSampler) {
        const uint32_t end = std::min<uint32_t>(c.registerIndex + c.registerCount, kDxsoColorTermMaxSamplers);
        for (uint32_t s = c.registerIndex; s < end; s++) {
          samplerNames[s] = c.registerCount > 1 ? str::format(c.name, "[", s - c.registerIndex, "]") : c.name;
          if (startsWith(name, "texture2d_") || startsWith(name, "texturecube_") || startsWith(name, "texture3d_")) {
            materialSamplerMask |= 1u << s;
          } else if (containsToken(name, "lightmap") || containsToken(name, "bsplinetexture")) {
            lightmapLikeSamplerMask |= 1u << s;
            if (containsToken(name, "lightmaptextures")) {
              info.lightmapTextureSamplerMask |= 1u << s;
            }
          }
        }
        continue;
      }
      if (c.registerSet != kD3dxRegisterSetFloat4 || c.registerIndex >= kDxsoColorTermMaxConstRegs) {
        continue;
      }
      if (containsToken(name, "ambientcolorandskyfactor")) {
        info.ambientColorAndSkyFactorRegister = int32_t(c.registerIndex);
        inputs.lightingConstRegs.set(c.registerIndex);
      } else if (containsToken(name, "upperskycolor")) {
        info.upperSkyColorRegister = int32_t(c.registerIndex);
        inputs.lightingConstRegs.set(c.registerIndex);
      } else if (containsToken(name, "lowerskycolor")) {
        info.lowerSkyColorRegister = int32_t(c.registerIndex);
        inputs.lightingConstRegs.set(c.registerIndex);
      } else if (containsToken(name, "lightmapscale")) {
        info.lightMapScaleRegister = int32_t(c.registerIndex);
        info.lightMapScaleCount = c.registerCount;
      } else if (containsToken(name, "localtoworldmatrix") || containsToken(name, "worldtoviewmatrix")) {
        info.worldCoords = true;
      } else if (startsWith(name, "uniformvector_") || startsWith(name, "uniformscalar_")) {
        for (uint32_t r = c.registerIndex; r < c.registerIndex + c.registerCount && r < kDxsoColorTermMaxConstRegs; r++) {
          info.materialConstantRegisters.push_back(r);
          inputs.trackedConstRegs.set(r);
        }
      }
    }
    // LocalVertexFactory only passes TangentBasisTangent (COLOR1) under WORLD_COORDS.
    info.worldCoords |= readsColor1;
    std::sort(info.materialConstantRegisters.begin(), info.materialConstantRegisters.end());
    info.materialConstantRegisters.erase(
      std::unique(info.materialConstantRegisters.begin(), info.materialConstantRegisters.end()),
      info.materialConstantRegisters.end());

    inputs.lightmapSamplerMask = lightmapLikeSamplerMask;
    inputs.trackedSamplerMask = declaredSamplerMask & ~lightmapLikeSamplerMask;

    const DxsoColorTermResult terms =
      analyzeDxsoColorTerms(pixelShader.tokens, pixelShader.tokenCount, inputs);
    if (!terms.analyzed) {
      return info;
    }
    if (pixelShader.isgn != nullptr) {
      for (uint32_t i = 0; i < pixelShader.isgn->elemCount; i++) {
        const DxsoIsgnEntry& element = pixelShader.isgn->elems[i];
        if (element.semantic.usage == DxsoUsage::Texcoord && element.regNumber < 16 &&
            element.semantic.usageIndex < info.colorTexcoordLanes.size()) {
          info.colorTexcoordLanes[element.semantic.usageIndex] |=
            uint8_t((terms.colorInterpolantLanes >> (4u * element.regNumber)) & 0xFu);
        }
      }
    }
    // A register that reaches no output cannot change a bake, and one that animates would otherwise
    // mint a bake every frame.
    info.materialConstantRegisters.erase(
      std::remove_if(info.materialConstantRegisters.begin(), info.materialConstantRegisters.end(),
                     [&](const uint32_t r) { return !terms.constReach[r]; }),
      info.materialConstantRegisters.end());

    bool anyPureAmbientTerm = false;
    bool anyVertexLightmapTerm = false;
    bool anyColorLayer = false;
    auto noteColorSet = [&](const DxsoColorTermSet set) {
      info.hasEmissive |= dxsoColorTermSetHasUnlitTerm(set);
      info.viewDependent |= setHasFactor(set, DxsoColorTermFactor_View);
      anyPureAmbientTerm |= setHasPureAmbientTerm(set);
      anyVertexLightmapTerm |= setHasFactor(set, DxsoColorTermFactor_Interpolant);
    };

    for (uint32_t s = 0; s < kDxsoColorTermMaxSamplers; s++) {
      if (((inputs.trackedSamplerMask >> s) & 1u) == 0) {
        continue;
      }
      const DxsoColorTermSet set = terms.samplerTerms[s];
      const uint8_t reach = terms.samplerReach[s];
      const DxsoColorTermRole role = classifyDxsoColorTermSource(set, reach, terms.hasLightConstTerm);
      if (role == DxsoColorTermRole::Unused || role == DxsoColorTermRole::LightingOnly) {
        if (((materialSamplerMask >> s) & 1u) != 0 && samplerTypes[s] == DxsoTextureType::Texture2D &&
            terms.samplerSampleCount[s] > 0) {
          info.lightingLayers.push_back({ uint8_t(s), samplerNames[s] });
        }
        continue;
      }
      if (role != DxsoColorTermRole::Color && role != DxsoColorTermRole::Opacity &&
          role != DxsoColorTermRole::Coordinate) {
        continue;
      }
      if (((materialSamplerMask >> s) & 1u) == 0) {
        info.refusal = DxsoUe3CompositeRefusal::EngineSampler;
        info.refusalDetail = str::format("s", s, "(", samplerNames[s].empty() ? "?" : samplerNames[s].c_str(), ")");
        return info;
      }
      if (samplerTypes[s] == DxsoTextureType::Texture3D) {
        info.refusal = DxsoUe3CompositeRefusal::VolumeLayer;
        info.refusalDetail = str::format("s", s, "(", samplerNames[s], ")");
        return info;
      }

      DxsoUe3CompositeLayer layer;
      layer.sampler = uint8_t(s);
      layer.isCube = samplerTypes[s] == DxsoTextureType::TextureCube;
      layer.name = samplerNames[s];
      if (role == DxsoColorTermRole::Color) {
        layer.emissive = dxsoColorTermSetHasUnlitTerm(set);
        layer.diffuse = setHasLitTerm(set);
        anyColorLayer |= layer.emissive || layer.diffuse;
        noteColorSet(set);
      }
      layer.opacity = (reach & DxsoColorTermReach_Opacity) != 0;
      info.layers.push_back(std::move(layer));
    }

    for (uint32_t r = 0; r < kDxsoColorTermMaxConstRegs; r++) {
      if (inputs.trackedConstRegs.test(r)) {
        noteColorSet(terms.constTerms[r]);
      }
    }
    for (const DxsoColorTermLiteral& literal : terms.literals) {
      noteColorSet(literal.terms);
    }

    if (!anyColorLayer) {
      info.refusal = DxsoUe3CompositeRefusal::NoColorLayer;
      return info;
    }

    if (anyPureAmbientTerm && info.ambientColorAndSkyFactorRegister >= 0) {
      info.lighting = DxsoUe3CompositeLighting::AmbientConstant;
    } else if (info.lightmapTextureSamplerMask) {
      info.lighting = DxsoUe3CompositeLighting::SimpleLightmap;
    } else if (anyVertexLightmapTerm) {
      const uint32_t factorInputs = terms.interpolantFactorInputs;
      if (!factorInputs || (factorInputs & (factorInputs - 1)) != 0) {
        info.refusal = DxsoUe3CompositeRefusal::VertexLightmap;
        return info;
      }
      info.lighting = DxsoUe3CompositeLighting::SimpleVertexLightmap;
      info.vertexLightmapInputMask = factorInputs;
    } else {
      info.lighting = DxsoUe3CompositeLighting::Unlit;
    }

    if (readsColor0 && !info.worldCoords) {
      info.refusal = DxsoUe3CompositeRefusal::VertexColor;
      return info;
    }

    info.alphaInputRegister = alphaInputRegister;
    if (info.hasTexkill && alphaInputRegister < 0) {
      info.refusal = DxsoUe3CompositeRefusal::MaskedWithoutCoverage;
      return info;
    }

    info.viewDependent |= std::any_of(info.layers.begin(), info.layers.end(),
                                      [](const DxsoUe3CompositeLayer& layer) { return layer.isCube; });

    // The bake with the mask at 1 only replaces the first pass, so the mask must not change the emission
    // the second pass takes out of it.
    for (const DxsoUe3CompositeLayer& layer : info.layers) {
      const uint32_t s = layer.sampler;
      if (layer.isCube || !layer.contributesColor() || layer.opacity ||
          (terms.samplerReach[s] & DxsoColorTermReach_Coordinate) != 0 || terms.samplerSampleCount[s] != 1 ||
          (layer.emissive && info.lighting != DxsoUe3CompositeLighting::Unlit)) {
        continue;
      }
      if (const std::optional<DxsoUe3CompositeMaskLayer> mask = findMask(pixelShader, programInfo, s)) {
        info.maskLayers.push_back(*mask);
      }
    }

    info.refusal = DxsoUe3CompositeRefusal::None;
    return info;
  }

  float solveUe3CompositePeriod(const float* scales, const uint32_t count, const uint32_t maxMultiple, const float tolerance) {
    if (scales == nullptr || count == 0) {
      return 0.f;
    }
    double minScale = std::numeric_limits<double>::infinity();
    for (uint32_t i = 0; i < count; i++) {
      const double scale = std::abs(double(scales[i]));
      if (!std::isfinite(scale) || scale < 1e-6) {
        return 0.f;
      }
      minScale = std::min(minScale, scale);
    }
    for (uint32_t k = 1; k <= maxMultiple; k++) {
      const double period = double(k) / minScale;
      bool whole = true;
      for (uint32_t i = 0; i < count && whole; i++) {
        const double repeats = std::abs(double(scales[i])) * period;
        const double nearest = std::round(repeats);
        whole = nearest >= 1.0 && std::abs(repeats - nearest) <= double(tolerance);
      }
      if (whole) {
        return float(period);
      }
    }
    return 0.f;
  }

  uint32_t solveUe3CompositeResolution(const float* texelsPerPeriod, const uint32_t count,
                                       const uint32_t minResolution, const uint32_t maxResolution) {
    double needed = double(std::max(minResolution, 1u));
    for (uint32_t i = 0; i < count; i++) {
      const double texels = double(texelsPerPeriod[i]);
      if (std::isfinite(texels)) {
        needed = std::max(needed, texels);
      }
    }
    uint32_t resolution = 1;
    while (double(resolution) < needed && resolution < (1u << 15)) {
      resolution <<= 1;
    }
    return std::max(std::min(resolution, maxResolution), 1u);
  }

}
