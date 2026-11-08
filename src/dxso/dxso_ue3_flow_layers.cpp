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
#include "dxso_ue3_flow_layers.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <utility>

#include "dxso_code.h"
#include "dxso_color_terms.h"
#include "dxso_ctab.h"
#include "dxso_decoder.h"
#include "dxso_highlight_tints.h"
#include "dxso_symbolic_eval.h"
#include "dxso_uv_dataflow.h"
#include "../util/util_string.h"

namespace dxvk {

  const char* describeDxsoUe3FlowRefusal(const DxsoUe3FlowRefusal refusal) {
    switch (refusal) {
    case DxsoUe3FlowRefusal::None: return "none";
    case DxsoUe3FlowRefusal::NotAnalyzed: return "not an analysable ps_2_0+ shader with a constant table";
    case DxsoUe3FlowRefusal::NoPannedSampler:
      return "no material texture outside the colour has a coordinate offset by a constant";
    case DxsoUe3FlowRefusal::EvaluationFailed: return "the symbolic evaluation failed";
    case DxsoUe3FlowRefusal::NotNormalized: return "no panned texture reaches a normalised vector";
    case DxsoUe3FlowRefusal::NotLinear: return "a panned texture reaches the normal other than as a weighted vector";
    case DxsoUe3FlowRefusal::NotProportional: return "the panned textures are weighted by different masks";
    case DxsoUe3FlowRefusal::MaskNotSingleChannel: return "the panned textures are weighted by more than one texture channel";
    case DxsoUe3FlowRefusal::NoWeight: return "the panned textures carry no positive weight";
    case DxsoUe3FlowRefusal::MaskUnresolved: return "the mask's texture coordinate cannot be resolved";
    case DxsoUe3FlowRefusal::FoldedBlend:
      return "a later normalisation reaches the panned textures through math the evaluation had to fold";
    }
    return "unknown";
  }

  namespace {

    using dxso_symbolic::Evaluator;
    using dxso_symbolic::NormalizeSite;
    using dxso_symbolic::Poly;
    using dxso_symbolic::SymbolKind;
    using dxso_symbolic::Term;

    bool startsWith(const std::string& s, const char* prefix) {
      return s.rfind(prefix, 0) == 0;
    }

    bool termReadsConstant(const UvAffineTerm& t) {
      return t.constReg >= 0 || t.constReg2 >= 0;
    }

    void collectOffsetRegisters(const UvAffineTerm& t, std::vector<uint32_t>& regs) {
      if (t.constReg >= 0) {
        regs.push_back(uint32_t(t.constReg));
      }
      if (t.constReg2 >= 0) {
        regs.push_back(uint32_t(t.constReg2));
      }
    }

    bool termReadsRegister(const UvAffineTerm& t, const uint32_t reg) {
      return t.constReg == int16_t(reg) || t.constReg2 == int16_t(reg);
    }

    bool componentReadsRegister(const UvComponentAffine& component, const uint32_t reg) {
      return termReadsRegister(component.scale, reg) || termReadsRegister(component.offset, reg) ||
             (component.hasCross && termReadsRegister(component.cross, reg));
    }

    Poly sortedPoly(std::vector<Term> terms) {
      std::sort(terms.begin(), terms.end(), dxso_symbolic::monomialLess);
      Poly result;
      for (const Term& t : terms) {
        if (!result.empty() && dxso_symbolic::compareMonomial(result.back(), t) == 0) {
          result.back().coef += t.coef;
        } else {
          result.push_back(t);
        }
      }
      result.erase(std::remove_if(result.begin(), result.end(),
                                  [](const Term& t) { return std::abs(t.coef) <= dxso_symbolic::kZeroCoefficient; }),
                   result.end());
      return result;
    }

    // Returns the ratio when a = ratio * b, monomial by monomial.
    std::optional<double> polyRatio(const Poly& a, const Poly& b) {
      if (a.size() != b.size() || b.empty()) {
        return std::nullopt;
      }
      const double ratio = a[0].coef / b[0].coef;
      for (size_t i = 0; i < a.size(); i++) {
        if (dxso_symbolic::compareMonomial(a[i], b[i]) != 0 ||
            !dxso_symbolic::approxEqual(a[i].coef, ratio * b[i].coef)) {
          return std::nullopt;
        }
      }
      return ratio;
    }

    std::optional<Poly> multiplyPolys(const Poly& a, const Poly& b) {
      std::vector<Term> products;
      products.reserve(a.size() * b.size());
      for (const Term& x : a) {
        for (const Term& y : b) {
          if (uint32_t(x.count) + y.count > dxso_symbolic::kMaxTermSymbols) {
            return std::nullopt;
          }
          Term t;
          t.coef = x.coef * y.coef;
          t.count = uint8_t(x.count + y.count);
          std::merge(x.syms.begin(), x.syms.begin() + x.count, y.syms.begin(), y.syms.begin() + y.count, t.syms.begin());
          products.push_back(t);
        }
      }
      return sortedPoly(std::move(products));
    }

    enum class Fit : uint8_t {
      Absent,
      // The vector is a sum in which the read appears once per term, channel for lane, times one shared polynomial.
      Linear,
      // The read appears in the vector some other way.
      NotLinear,
      // Only atoms computed from the read appear, as when the read was normalised on its own first.
      Indirect,
    };

    // Describes a read of a material texture whose coordinate a constant offsets.
    struct CandidateRead {
      uint32_t sampler = 0;
      uint32_t site = 0;
      PsSamplerUvOrigin uv;
      std::vector<uint16_t> symbols;
      // Counts the sampler's candidate reads before this one, to tell them apart in the log.
      uint32_t ordinal = 0;
    };

    // Describes how a read reaches a normalised vector: per lane, the texel channel and the polynomial that channel is
    // multiplied by. Every lane has to be multiplied by the same one, so the sample enters as a vector with one weight.
    struct SampleFit {
      Fit fit = Fit::Absent;
      std::array<uint8_t, 3> components = { kDxsoUe3FlowNoComponent, kDxsoUe3FlowNoComponent, kDxsoUe3FlowNoComponent };
      Poly coefficient;
    };

    // Splits the lanes of a site into the terms holding one of `symbols` and what multiplies it there. Each lane may
    // hold one symbol, the same in every term, at most once per term, and nothing `dependsOn` reports alongside it.
    // Returns false when a lane breaks that.
    template<typename IsSymbol, typename DependsOn>
    bool splitLanes(const NormalizeSite& site, IsSymbol&& isSymbol, DependsOn&& dependsOn,
                    std::array<int32_t, 3>& laneSymbols, std::array<Poly, 3>& laneCoefficients, bool& indirect) {
      indirect = false;
      for (uint32_t lane = 0; lane < 3; lane++) {
        laneSymbols[lane] = -1;
        std::vector<Term> terms;
        for (const Term& t : site.lanes[lane]) {
          int32_t found = -1;
          uint32_t occurrences = 0;
          bool dependent = false;
          for (uint32_t i = 0; i < t.count; i++) {
            if (isSymbol(t.syms[i])) {
              occurrences++;
              found = t.syms[i];
            } else if (dependsOn(t.syms[i])) {
              dependent = true;
            }
          }
          if (occurrences == 0) {
            indirect |= dependent;
            continue;
          }
          if (occurrences > 1 || dependent || (laneSymbols[lane] >= 0 && laneSymbols[lane] != found)) {
            return false;
          }
          laneSymbols[lane] = found;
          Term divisor;
          divisor.coef = 1.0;
          divisor.count = 1;
          divisor.syms[0] = uint16_t(found);
          Term remainder;
          dxso_symbolic::divideMonomial(t, divisor, remainder);
          remainder.coef = t.coef;
          terms.push_back(remainder);
        }
        laneCoefficients[lane] = sortedPoly(std::move(terms));
      }
      return true;
    }

    // Returns the shared coefficient when every lane present carries the same one and x and y are present.
    std::optional<Poly> sharedCoefficient(const std::array<int32_t, 3>& laneSymbols, const std::array<Poly, 3>& laneCoefficients) {
      if (laneSymbols[0] < 0 || laneSymbols[1] < 0) {
        return std::nullopt;
      }
      for (uint32_t lane = 1; lane < 3; lane++) {
        if (laneSymbols[lane] < 0) {
          continue;
        }
        const std::optional<double> ratio = polyRatio(laneCoefficients[lane], laneCoefficients[0]);
        if (!ratio || !dxso_symbolic::approxEqual(*ratio, 1.0)) {
          return std::nullopt;
        }
      }
      return laneCoefficients[0];
    }

    SampleFit fitSample(const Evaluator& evaluator, const NormalizeSite& site, const CandidateRead& read) {
      SampleFit result;
      std::array<int32_t, 3> laneSymbols;
      std::array<Poly, 3> laneCoefficients;
      bool indirect = false;
      const bool split = splitLanes(
        site,
        [&](const uint16_t s) {
          const dxso_symbolic::Symbol& symbol = evaluator.symbol(s);
          return symbol.kind == SymbolKind::Sample && symbol.reg == read.sampler && symbol.site == uint16_t(read.site);
        },
        [&](const uint16_t s) {
          return evaluator.symbol(s).kind == SymbolKind::Atom &&
                 std::any_of(read.symbols.begin(), read.symbols.end(),
                             [&](const uint16_t sample) { return evaluator.atomDependsOn(s, sample); });
        },
        laneSymbols, laneCoefficients, indirect);
      const bool reached = std::any_of(laneSymbols.begin(), laneSymbols.end(), [](const int32_t s) { return s >= 0; });
      if (!split) {
        result.fit = Fit::NotLinear;
        return result;
      }
      if (!reached) {
        result.fit = indirect ? Fit::Indirect : Fit::Absent;
        return result;
      }

      result.fit = Fit::NotLinear;
      for (uint32_t lane = 0; lane < 3; lane++) {
        if (laneSymbols[lane] >= 0) {
          result.components[lane] = evaluator.symbol(uint16_t(laneSymbols[lane])).component;
        }
      }
      // x and y carry the perturbation. A missing z is reconstructed, as a two-channel normal map's is.
      const std::array<uint8_t, 3>& c = result.components;
      if (indirect || c[0] == c[1] || (c[2] != kDxsoUe3FlowNoComponent && (c[2] == c[0] || c[2] == c[1]))) {
        return result;
      }
      std::optional<Poly> coefficient = sharedCoefficient(laneSymbols, laneCoefficients);
      if (!coefficient) {
        return result;
      }
      result.fit = Fit::Linear;
      result.coefficient = std::move(*coefficient);
      return result;
    }

    // Returns the coefficient a later site multiplies an earlier site's normalised vector by, lane for lane, when it
    // blends that vector in linearly. A use that is not linear, such as the reflection vector built from it, gives none.
    std::optional<Poly> fitNormalized(const Evaluator& evaluator, const NormalizeSite& site, const NormalizeSite& earlier,
                                      const std::vector<uint16_t>& candidateSamples) {
      if (earlier.outputs[0] == dxso_symbolic::kNoSymbol || earlier.outputs[1] == dxso_symbolic::kNoSymbol) {
        return std::nullopt;
      }
      std::array<int32_t, 3> laneSymbols;
      std::array<Poly, 3> laneCoefficients;
      bool indirect = false;
      const bool split = splitLanes(
        site,
        [&](const uint16_t s) {
          return s == earlier.outputs[0] || s == earlier.outputs[1] || s == earlier.outputs[2];
        },
        [&](const uint16_t s) {
          if (evaluator.symbol(s).kind != SymbolKind::Atom) {
            return std::binary_search(candidateSamples.begin(), candidateSamples.end(), s);
          }
          return std::any_of(candidateSamples.begin(), candidateSamples.end(),
                             [&](const uint16_t sample) { return evaluator.atomDependsOn(s, sample); });
        },
        laneSymbols, laneCoefficients, indirect);
      if (!split || indirect) {
        return std::nullopt;
      }
      for (uint32_t lane = 0; lane < 3; lane++) {
        if (laneSymbols[lane] >= 0 && laneSymbols[lane] != int32_t(earlier.outputs[lane])) {
          return std::nullopt;
        }
      }
      return sharedCoefficient(laneSymbols, laneCoefficients);
    }

    struct ShaderFacts {
      // Holds the samplers with a candidate read.
      uint32_t candidateMask = 0;
      uint32_t material2dMask = 0;
      std::array<std::string, kDxsoColorTermMaxSamplers> samplerNames;
      // Lists every read, in program order.
      std::vector<PsSampleSiteUv> sites;
      std::vector<CandidateRead> reads;
    };

    std::string describeSampler(const ShaderFacts& facts, const uint32_t s) {
      return str::format("s", s, "(", facts.samplerNames[s].empty() ? "?" : facts.samplerNames[s].c_str(), ")");
    }

    std::string describeRead(const ShaderFacts& facts, const CandidateRead& read) {
      const bool several = std::count_if(facts.reads.begin(), facts.reads.end(),
                                         [&](const CandidateRead& other) { return other.sampler == read.sampler; }) > 1;
      return several
        ? str::format(describeSampler(facts, read.sampler), " read ", read.ordinal + 1)
        : describeSampler(facts, read.sampler);
    }

    const PsSampleSiteUv* findSite(const ShaderFacts& facts, const uint32_t site) {
      const auto it = std::find_if(facts.sites.begin(), facts.sites.end(),
                                   [&](const PsSampleSiteUv& s) { return uint16_t(s.instruction) == uint16_t(site); });
      return it != facts.sites.end() ? &*it : nullptr;
    }

    // Collects the samplers and the offset registers of a layer's reads, and the registers no other read's coordinate
    // reads. A read whose coordinate the dataflow cannot resolve is not seen here, but a composite bake refuses such a layer.
    void collectLayerRegisters(const ShaderFacts& facts, DxsoUe3FlowShaderInfo& info) {
      std::vector<uint32_t> layerSites;
      auto addRead = [&](const uint8_t sampler, const uint32_t site, const PsSamplerUvOrigin& uv) {
        info.flowSamplerMask |= 1u << sampler;
        layerSites.push_back(site);
        collectOffsetRegisters(uv.affineU.offset, info.offsetRegisters);
        collectOffsetRegisters(uv.affineV.offset, info.offsetRegisters);
      };
      for (const DxsoUe3FlowSample& sample : info.samples) {
        addRead(sample.sampler, sample.site, sample.uv);
      }
      if (info.blend.present) {
        addRead(info.blend.sampler, info.blend.site, info.blend.uv);
      }
      std::sort(info.offsetRegisters.begin(), info.offsetRegisters.end());
      info.offsetRegisters.erase(std::unique(info.offsetRegisters.begin(), info.offsetRegisters.end()),
                                 info.offsetRegisters.end());
      for (const uint32_t reg : info.offsetRegisters) {
        const bool shared = std::any_of(facts.sites.begin(), facts.sites.end(), [&](const PsSampleSiteUv& site) {
          return std::find(layerSites.begin(), layerSites.end(), site.instruction) == layerSites.end() && site.uv.originValid &&
                 (componentReadsRegister(site.uv.affineU, reg) || componentReadsRegister(site.uv.affineV, reg));
        });
        if (!shared) {
          info.exclusiveOffsetRegisters.push_back(reg);
        }
      }
    }

    // Fits a normalised vector whose lanes are a * X[k] + b + w * c * (Y[k] - X[k]): the normal map X blended towards Y by
    // w, one channel of the panned read, as a tarp's wind blends its normal map towards a wrinkled one. c is a constant,
    // or that times one channel of another texture, which the layer keeps as its mask.
    std::optional<DxsoUe3FlowShaderInfo> fitBlend(const Evaluator& evaluator, const ShaderFacts& facts,
                                                  const NormalizeSite& site, const CandidateRead& weightRead) {
      auto isWeight = [&](const uint16_t s) {
        const dxso_symbolic::Symbol& symbol = evaluator.symbol(s);
        return symbol.kind == SymbolKind::Sample && symbol.reg == weightRead.sampler && symbol.site == uint16_t(weightRead.site);
      };
      auto foldsWeight = [&](const uint16_t s) {
        return evaluator.symbol(s).kind == SymbolKind::Atom &&
               std::any_of(weightRead.symbols.begin(), weightRead.symbols.end(),
                           [&](const uint16_t weight) { return evaluator.atomDependsOn(s, weight); });
      };
      auto sameRead = [&](const uint16_t a, const uint16_t b) {
        const dxso_symbolic::Symbol& x = evaluator.symbol(a);
        const dxso_symbolic::Symbol& y = evaluator.symbol(b);
        return x.reg == y.reg && x.site == y.site;
      };

      // Each lane is P + w * Q.
      int32_t weightSymbol = -1;
      std::array<std::vector<Term>, 3> rest;
      std::array<std::vector<Term>, 3> weighted;
      for (uint32_t lane = 0; lane < 3; lane++) {
        for (const Term& t : site.lanes[lane]) {
          int32_t found = -1;
          uint32_t occurrences = 0;
          for (uint32_t i = 0; i < t.count; i++) {
            if (isWeight(t.syms[i])) {
              occurrences++;
              found = t.syms[i];
            } else if (foldsWeight(t.syms[i])) {
              return std::nullopt;
            }
          }
          if (occurrences == 0) {
            rest[lane].push_back(t);
            continue;
          }
          if (occurrences > 1 || (weightSymbol >= 0 && weightSymbol != found)) {
            return std::nullopt;
          }
          weightSymbol = found;
          Term divisor;
          divisor.coef = 1.0;
          divisor.count = 1;
          divisor.syms[0] = uint16_t(found);
          Term remainder;
          dxso_symbolic::divideMonomial(t, divisor, remainder);
          remainder.coef = t.coef;
          weighted[lane].push_back(remainder);
        }
      }
      if (weightSymbol < 0) {
        return std::nullopt;
      }

      // P is a * X + b, one channel of the normal map blended from per lane.
      std::array<int32_t, 3> fromSymbols = { -1, -1, -1 };
      std::array<double, 3> fromCoefficients = {};
      for (uint32_t lane = 0; lane < 3; lane++) {
        for (const Term& t : rest[lane]) {
          if (t.count == 0) {
            continue;
          }
          if (t.count != 1 || evaluator.symbol(t.syms[0]).kind != SymbolKind::Sample || fromSymbols[lane] >= 0) {
            return std::nullopt;
          }
          fromSymbols[lane] = t.syms[0];
          fromCoefficients[lane] = t.coef;
        }
      }
      if (fromSymbols[0] < 0 || fromSymbols[1] < 0 || !(std::abs(fromCoefficients[0]) > 1e-6)) {
        return std::nullopt;
      }
      for (uint32_t lane = 0; lane < 3; lane++) {
        if (fromSymbols[lane] >= 0 &&
            (!sameRead(uint16_t(fromSymbols[lane]), uint16_t(fromSymbols[0])) ||
             !dxso_symbolic::approxEqual(fromCoefficients[lane], fromCoefficients[0]))) {
          return std::nullopt;
        }
      }

      // Q is c * (Y - X): what multiplies X is -c, the same in every lane, and the rest is c times one channel of the
      // normal map blended to.
      std::array<int32_t, 3> toSymbols = { -1, -1, -1 };
      Poly step;
      for (uint32_t lane = 0; lane < 3; lane++) {
        std::vector<Term> fromTerms;
        std::vector<Term> toTerms;
        for (const Term& t : weighted[lane]) {
          if (fromSymbols[lane] >= 0 && dxso_symbolic::symbolOccurrences(t, uint16_t(fromSymbols[lane])) == 1) {
            Term divisor;
            divisor.coef = 1.0;
            divisor.count = 1;
            divisor.syms[0] = uint16_t(fromSymbols[lane]);
            Term remainder;
            dxso_symbolic::divideMonomial(t, divisor, remainder);
            remainder.coef = -t.coef;
            fromTerms.push_back(remainder);
          } else {
            toTerms.push_back(t);
          }
        }
        const Poly c = sortedPoly(std::move(fromTerms));
        if (c.empty() && toTerms.empty()) {
          continue;
        }
        if (c.empty() || toTerms.empty()) {
          return std::nullopt;
        }
        if (step.empty()) {
          step = c;
        } else {
          const std::optional<double> ratio = polyRatio(c, step);
          if (!ratio || !dxso_symbolic::approxEqual(*ratio, 1.0)) {
            return std::nullopt;
          }
        }
        // The first of c's monomials divides the first term blended to, leaving its channel of Y.
        Term toSymbol;
        if (!dxso_symbolic::divideMonomial(sortedPoly(toTerms)[0], step[0], toSymbol) || toSymbol.count != 1 ||
            evaluator.symbol(toSymbol.syms[0]).kind != SymbolKind::Sample) {
          return std::nullopt;
        }
        toSymbols[lane] = toSymbol.syms[0];
        const std::optional<Poly> expected = multiplyPolys(step, dxso_symbolic::polySymbol(toSymbol.syms[0]));
        const std::optional<double> toRatio = expected ? polyRatio(sortedPoly(toTerms), *expected) : std::nullopt;
        if (!toRatio || !dxso_symbolic::approxEqual(*toRatio, 1.0)) {
          return std::nullopt;
        }
      }
      if (toSymbols[0] < 0 || toSymbols[1] < 0 || step.size() != 1 || step[0].count > 1) {
        return std::nullopt;
      }
      for (uint32_t lane = 0; lane < 3; lane++) {
        if (toSymbols[lane] >= 0 && !sameRead(uint16_t(toSymbols[lane]), uint16_t(toSymbols[0]))) {
          return std::nullopt;
        }
      }
      if (sameRead(uint16_t(toSymbols[0]), uint16_t(fromSymbols[0]))) {
        return std::nullopt;
      }

      // The reads have to be ones the runtime can place.
      const dxso_symbolic::Symbol& from = evaluator.symbol(uint16_t(fromSymbols[0]));
      const dxso_symbolic::Symbol& to = evaluator.symbol(uint16_t(toSymbols[0]));
      const PsSampleSiteUv* fromSite = findSite(facts, from.site);
      const PsSampleSiteUv* toSite = findSite(facts, to.site);
      const PsSampleSiteUv* weightSite = findSite(facts, weightRead.site);
      auto placeable = [](const PsSampleSiteUv* s) { return s != nullptr && s->uv.originValid && s->uv.affineExact; };
      if (!placeable(fromSite) || !placeable(toSite) || !placeable(weightSite)) {
        return std::nullopt;
      }
      const PsSampleSiteUv* factorSite = nullptr;
      int32_t factorSymbol = -1;
      if (step[0].count == 1) {
        factorSymbol = step[0].syms[0];
        const dxso_symbolic::Symbol& factor = evaluator.symbol(uint16_t(factorSymbol));
        const bool otherRead = !sameRead(uint16_t(factorSymbol), uint16_t(fromSymbols[0])) &&
                               !sameRead(uint16_t(factorSymbol), uint16_t(toSymbols[0])) && !isWeight(uint16_t(factorSymbol));
        factorSite = otherRead && factor.kind == SymbolKind::Sample && factor.reg < kDxsoMaxPsSamplers &&
                     ((facts.material2dMask >> factor.reg) & 1u) != 0
          ? findSite(facts, factor.site)
          : nullptr;
        if (!placeable(factorSite)) {
          return std::nullopt;
        }
      }

      DxsoUe3FlowShaderInfo info;
      auto makeSample = [&](const PsSampleSiteUv& readSite, const std::array<int32_t, 3>& symbols) {
        DxsoUe3FlowSample sample;
        sample.sampler = readSite.sampler;
        sample.site = readSite.instruction;
        sample.uv = readSite.uv;
        for (uint32_t lane = 0; lane < 3; lane++) {
          sample.components[lane] = symbols[lane] >= 0 ? evaluator.symbol(uint16_t(symbols[lane])).component : kDxsoUe3FlowNoComponent;
        }
        sample.weight = 1.f;
        sample.name = facts.samplerNames[readSite.sampler];
        return sample;
      };
      info.samples.push_back(makeSample(*fromSite, fromSymbols));
      info.samples.push_back(makeSample(*toSite, toSymbols));
      for (const DxsoUe3FlowSample& sample : info.samples) {
        if (sample.components[0] == sample.components[1] ||
            (sample.components[2] != kDxsoUe3FlowNoComponent &&
             (sample.components[2] == sample.components[0] || sample.components[2] == sample.components[1]))) {
          return std::nullopt;
        }
      }
      info.blend.present = true;
      info.blend.sampler = uint8_t(weightRead.sampler);
      info.blend.site = weightRead.site;
      info.blend.uv = weightSite->uv;
      info.blend.channel = evaluator.symbol(uint16_t(weightSymbol)).component;
      info.blend.scale = float(step[0].coef / fromCoefficients[0]);
      info.blend.name = facts.samplerNames[weightRead.sampler];
      if (factorSite != nullptr) {
        info.mask.present = true;
        info.mask.sampler = factorSite->sampler;
        info.mask.uv = factorSite->uv;
        info.mask.channel = evaluator.symbol(uint16_t(factorSymbol)).component;
        info.mask.name = facts.samplerNames[factorSite->sampler];
      }
      collectLayerRegisters(facts, info);
      info.refusal = DxsoUe3FlowRefusal::None;
      return info;
    }

    // Looks for a blend at the normalisations from the last, which is where the material normal is decided.
    std::optional<DxsoUe3FlowShaderInfo> analyzeBlend(const Evaluator& evaluator, const ShaderFacts& facts) {
      const std::vector<NormalizeSite>& sites = evaluator.normalizeSites();
      for (int32_t i = int32_t(sites.size()) - 1; i >= 0; i--) {
        for (const CandidateRead& read : facts.reads) {
          if (std::optional<DxsoUe3FlowShaderInfo> info = fitBlend(evaluator, facts, sites[i], read)) {
            return info;
          }
        }
      }
      return std::nullopt;
    }

    // With every material texture a modulator, a texture that blends the flow vector in keeps the blend a polynomial,
    // so a later normalisation that reaches the vector only through folded math is a use of it, such as a normalised
    // reflection vector. With only the samples kept, the blend may be in what folded.
    DxsoUe3FlowShaderInfo analyzeSites(const Evaluator& evaluator, const ShaderFacts& facts, const bool blendsMayFold) {
      DxsoUe3FlowShaderInfo info;
      const std::vector<NormalizeSite>& sites = evaluator.normalizeSites();

      // The flow vector is the last one the samples reach linearly and only that way. The material's own expressions
      // may normalise it before CalcMaterialParameters normalises the material normal, so a later normalisation that
      // blends it in linearly is where its weight is decided.
      int32_t flowSite = -1;
      std::vector<std::pair<size_t, SampleFit>> fits;
      info.refusal = DxsoUe3FlowRefusal::NotNormalized;
      for (int32_t i = int32_t(sites.size()) - 1; i >= 0 && flowSite < 0; i--) {
        std::vector<std::pair<size_t, SampleFit>> siteFits;
        bool clean = true;
        for (size_t r = 0; r < facts.reads.size() && clean; r++) {
          SampleFit fit = fitSample(evaluator, sites[i], facts.reads[r]);
          if (fit.fit == Fit::NotLinear && info.refusal == DxsoUe3FlowRefusal::NotNormalized) {
            info.refusal = DxsoUe3FlowRefusal::NotLinear;
            info.refusalDetail = describeRead(facts, facts.reads[r]);
          }
          clean = fit.fit != Fit::NotLinear && fit.fit != Fit::Indirect;
          if (fit.fit == Fit::Linear) {
            siteFits.emplace_back(r, std::move(fit));
          }
        }
        if (clean && !siteFits.empty()) {
          flowSite = i;
          fits = std::move(siteFits);
        }
      }
      if (flowSite < 0) {
        // A panned read can instead weigh a blend between two normal maps.
        if (std::optional<DxsoUe3FlowShaderInfo> blend = analyzeBlend(evaluator, facts)) {
          return *blend;
        }
        return info;
      }
      info = DxsoUe3FlowShaderInfo {};

      // Each sample's coefficient is a multiple of one shared weight, which is what confines them all.
      Poly weight = fits[0].second.coefficient;
      std::vector<double> ratios(fits.size(), 1.0);
      for (size_t i = 1; i < fits.size(); i++) {
        const std::optional<double> ratio = polyRatio(fits[i].second.coefficient, weight);
        if (!ratio) {
          info.refusal = DxsoUe3FlowRefusal::NotProportional;
          info.refusalDetail = str::format(describeRead(facts, facts.reads[fits[i].first]), " against ",
                                           describeRead(facts, facts.reads[fits[0].first]));
          return info;
        }
        ratios[i] = *ratio;
      }

      std::vector<uint16_t> candidateSamples;
      for (const CandidateRead& read : facts.reads) {
        candidateSamples.insert(candidateSamples.end(), read.symbols.begin(), read.symbols.end());
      }
      std::sort(candidateSamples.begin(), candidateSamples.end());
      bool blended = false;
      for (int32_t j = int32_t(sites.size()) - 1; j > flowSite; j--) {
        if (const std::optional<Poly> blend = fitNormalized(evaluator, sites[j], sites[flowSite], candidateSamples)) {
          const std::optional<Poly> combined = multiplyPolys(weight, *blend);
          if (!combined) {
            info.refusal = DxsoUe3FlowRefusal::EvaluationFailed;
            info.refusalDetail = "the weight outgrew the term budget";
            return info;
          }
          weight = *combined;
          blended = true;
          break;
        }
      }
      if (!blended && blendsMayFold) {
        for (size_t j = size_t(flowSite) + 1; j < sites.size(); j++) {
          for (const auto& [readIndex, fit] : fits) {
            if (fitSample(evaluator, sites[j], facts.reads[readIndex]).fit != Fit::Absent) {
              info.refusal = DxsoUe3FlowRefusal::FoldedBlend;
              info.refusalDetail = describeRead(facts, facts.reads[readIndex]);
              return info;
            }
          }
        }
      }

      // The weight is constant + linear * mask, where the mask is one channel of one other material texture.
      // Anything else in it, such as another texture's channel overriding the samples or a product of several,
      // is left out of the layer.
      double constant = 0.0;
      double linear = 0.0;
      int32_t maskSymbol = -1;
      for (const Term& t : weight) {
        if (t.count == 0) {
          constant += t.coef;
          continue;
        }
        if (t.count == 1) {
          const dxso_symbolic::Symbol& symbol = evaluator.symbol(t.syms[0]);
          if (symbol.kind == SymbolKind::Sample && symbol.reg < kDxsoMaxPsSamplers &&
              ((facts.material2dMask >> symbol.reg) & 1u) != 0 && ((facts.candidateMask >> symbol.reg) & 1u) == 0) {
            if (maskSymbol >= 0 && maskSymbol != int32_t(t.syms[0])) {
              info.refusal = DxsoUe3FlowRefusal::MaskNotSingleChannel;
              info.refusalDetail = str::format(describeSampler(facts, evaluator.symbol(uint16_t(maskSymbol)).reg), " and ",
                                               describeSampler(facts, symbol.reg));
              return info;
            }
            maskSymbol = int32_t(t.syms[0]);
            linear += t.coef;
            continue;
          }
        }
        info.ignoresOtherFactors = true;
      }

      const double peak = maskSymbol >= 0 ? std::max(constant, constant + linear) : constant;
      if (!(peak > 1e-6)) {
        info.refusal = DxsoUe3FlowRefusal::NoWeight;
        return info;
      }

      if (maskSymbol >= 0) {
        const dxso_symbolic::Symbol& symbol = evaluator.symbol(uint16_t(maskSymbol));
        const PsSampleSiteUv* maskSite = findSite(facts, symbol.site);
        if (maskSite == nullptr || !maskSite->uv.originValid || !maskSite->uv.affineExact) {
          info.refusal = DxsoUe3FlowRefusal::MaskUnresolved;
          info.refusalDetail = describeSampler(facts, symbol.reg);
          return info;
        }
        info.mask.present = true;
        info.mask.sampler = uint8_t(symbol.reg);
        info.mask.uv = maskSite->uv;
        info.mask.channel = symbol.component;
        info.mask.scale = float(linear / peak);
        info.mask.bias = float(constant / peak);
        info.mask.name = facts.samplerNames[symbol.reg];
      }

      // A texel t is the decoded normal (2t - 1) / 2 plus a bias, so its weight is half its coefficient.
      for (size_t i = 0; i < fits.size(); i++) {
        const CandidateRead& read = facts.reads[fits[i].first];
        const double sampleWeight = ratios[i] * peak * 0.5;
        if (!(sampleWeight > 1e-6)) {
          info.refusal = DxsoUe3FlowRefusal::NoWeight;
          info.refusalDetail = describeRead(facts, read);
          return info;
        }
        DxsoUe3FlowSample sample;
        sample.sampler = uint8_t(read.sampler);
        sample.site = read.site;
        sample.uv = read.uv;
        sample.components = fits[i].second.components;
        sample.weight = float(sampleWeight);
        sample.name = facts.samplerNames[read.sampler];
        info.samples.push_back(std::move(sample));
      }
      std::stable_sort(info.samples.begin(), info.samples.end(),
                       [](const DxsoUe3FlowSample& a, const DxsoUe3FlowSample& b) { return a.weight > b.weight; });
      if (info.samples.size() > kDxsoUe3FlowMaxSamples) {
        info.samples.resize(kDxsoUe3FlowMaxSamples);
      }

      collectLayerRegisters(facts, info);
      info.refusal = DxsoUe3FlowRefusal::None;
      return info;
    }

  }

  DxsoUe3FlowShaderInfo analyzeDxsoUe3FlowLayers(const DxsoShaderView& pixelShader) {
    DxsoUe3FlowShaderInfo info;
    if (pixelShader.tokens == nullptr || pixelShader.info == nullptr || pixelShader.isgn == nullptr) {
      return info;
    }
    const std::optional<DxsoProgramInfo> programInfo =
      dxso_symbolic::pixelShaderProgramInfo(pixelShader.tokens, pixelShader.tokenCount);
    if (!programInfo) {
      return info;
    }

    std::array<DxsoTextureType, kDxsoColorTermMaxSamplers> samplerTypes;
    samplerTypes.fill(DxsoTextureType::Texture2D);
    uint32_t declaredSamplerMask = 0;
    DxsoDecodeContext decoder(*programInfo);
    DxsoCodeIter iter(pixelShader.tokens + 1);
    while (decoder.decodeInstruction(iter)) {
      const DxsoInstructionContext& ctx = decoder.getInstructionContext();
      if (ctx.instruction.opcode == DxsoOpcode::Dcl && ctx.dst.id.type == DxsoRegisterType::Sampler &&
          ctx.dst.id.num < kDxsoColorTermMaxSamplers) {
        samplerTypes[ctx.dst.id.num] = ctx.dcl.textureType;
        declaredSamplerMask |= 1u << ctx.dst.id.num;
      }
    }
    const DxsoCtab& ctab = decoder.getCtabInfo();
    if (ctab.m_size == 0 || ctab.m_constantData.empty()) {
      return info;
    }

    ShaderFacts facts;
    DxsoColorTermInputs inputs;
    uint32_t lightmapLikeSamplerMask = 0;
    for (const DxsoCtab::Constant& c : ctab.m_constantData) {
      if (c.registerCount == 0) {
        continue;
      }
      const std::string name = toLowerAscii(c.name);
      if (c.registerSet == kD3dxRegisterSetSampler) {
        const uint32_t end = std::min<uint32_t>(c.registerIndex + c.registerCount, kDxsoColorTermMaxSamplers);
        for (uint32_t s = c.registerIndex; s < end; s++) {
          facts.samplerNames[s] = c.registerCount > 1 ? str::format(c.name, "[", s - c.registerIndex, "]") : c.name;
          if (startsWith(name, "texture2d_")) {
            facts.material2dMask |= 1u << s;
          } else if (containsToken(name, "lightmap") || containsToken(name, "bsplinetexture")) {
            lightmapLikeSamplerMask |= 1u << s;
          }
        }
        continue;
      }
      if (c.registerSet != kD3dxRegisterSetFloat4 || c.registerIndex >= kDxsoColorTermMaxConstRegs) {
        continue;
      }
      if (containsToken(name, "ambientcolorandskyfactor") || containsToken(name, "upperskycolor") ||
          containsToken(name, "lowerskycolor")) {
        inputs.lightingConstRegs.set(c.registerIndex);
      }
    }
    facts.material2dMask &= declaredSamplerMask;
    for (uint32_t s = 0; s < kDxsoColorTermMaxSamplers; s++) {
      if (samplerTypes[s] != DxsoTextureType::Texture2D) {
        facts.material2dMask &= ~(1u << s);
      }
    }

    inputs.lightmapSamplerMask = lightmapLikeSamplerMask;
    inputs.trackedSamplerMask = declaredSamplerMask & ~lightmapLikeSamplerMask;
    const DxsoColorTermResult terms = analyzeDxsoColorTerms(pixelShader.tokens, pixelShader.tokenCount, inputs);
    if (!terms.analyzed) {
      return info;
    }
    std::array<PsSamplerUvOrigin, kDxsoMaxPsSamplers> origins;
    analyzePsSamplerUvOrigins(pixelShader, origins, nullptr, &facts.sites);

    // A flow sample shades through the normal, so its texture never reaches the colour as a colour of its own, and it is
    // read through a coordinate a constant offsets. Whether that constant animates is for the draws to show.
    std::array<uint32_t, kDxsoMaxPsSamplers> candidateReadsPerSampler = {};
    for (const PsSampleSiteUv& site : facts.sites) {
      const uint32_t s = site.sampler;
      if (s >= kDxsoMaxPsSamplers || ((facts.material2dMask >> s) & 1u) == 0) {
        continue;
      }
      const DxsoColorTermRole role =
        classifyDxsoColorTermSource(terms.samplerTerms[s], terms.samplerReach[s], terms.hasLightConstTerm);
      if (role == DxsoColorTermRole::Color || role == DxsoColorTermRole::Opacity) {
        continue;
      }
      if (!site.uv.originValid || !site.uv.affineExact ||
          (!termReadsConstant(site.uv.affineU.offset) && !termReadsConstant(site.uv.affineV.offset))) {
        continue;
      }
      CandidateRead read;
      read.sampler = s;
      read.site = site.instruction;
      read.uv = site.uv;
      read.ordinal = candidateReadsPerSampler[s]++;
      facts.reads.push_back(std::move(read));
      facts.candidateMask |= 1u << s;
    }
    if (facts.reads.empty()) {
      info.refusal = DxsoUe3FlowRefusal::NoPannedSampler;
      return info;
    }

    // With no material parameters or lighting declared, everything no modulator reaches folds. Every material texture
    // is a modulator first, so a mask stays visible however the material combines it. A shader too complex for that is
    // evaluated again with only the candidates kept.
    const DxsoHighlightInputs evaluationInputs;
    for (const bool everyTexture : { true, false }) {
      dxso_symbolic::EvaluatorOptions options;
      options.modulatorSamplerMask = everyTexture ? facts.material2dMask : facts.candidateMask;
      options.recordNormalizeSites = true;
      const auto evaluator = std::make_unique<Evaluator>(evaluationInputs, options, *programInfo);
      evaluator->run(pixelShader.tokens, pixelShader.tokenCount);
      if (evaluator->failure() != DxsoHighlightFailure::None) {
        info.refusal = DxsoUe3FlowRefusal::EvaluationFailed;
        info.refusalDetail = dxsoHighlightFailureName(evaluator->failure());
        continue;
      }
      for (CandidateRead& read : facts.reads) {
        read.symbols.clear();
      }
      for (size_t s = 0; s < evaluator->symbolCount(); s++) {
        const dxso_symbolic::Symbol& symbol = evaluator->symbol(uint16_t(s));
        if (symbol.kind != SymbolKind::Sample) {
          continue;
        }
        for (CandidateRead& read : facts.reads) {
          if (symbol.reg == read.sampler && symbol.site == uint16_t(read.site)) {
            read.symbols.push_back(uint16_t(s));
          }
        }
      }
      return analyzeSites(*evaluator, facts, !everyTexture);
    }
    return info;
  }

}
