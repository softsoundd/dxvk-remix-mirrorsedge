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
#include "dxso_material_fades.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <utility>

#include "dxso_symbolic_eval.h"

#include "../util/xxHash/xxhash.h"

namespace dxvk {

  namespace {

    using dxso_symbolic::Evaluator;
    using dxso_symbolic::Poly;
    using dxso_symbolic::SymbolKind;
    using dxso_symbolic::Term;

    static_assert(DxsoFadeTerm::kMaxParameters >= dxso_symbolic::kMaxTermSymbols);

    // A lane that outgrows this is a lit network the fades never come from, and evaluating it for
    // every draw would cost more than it could gain.
    constexpr size_t kMaxFadeLaneTerms = 64;
    // Coefficients are products of float constants, so equal ones differ by rounding.
    constexpr double kRelativeTolerance = 1.0e-3;
    // A colour or opacity term this small changes nothing on screen.
    constexpr double kVanishTolerance = 1.0e-4;

    Term withoutOneOccurrence(const Term& t, const uint16_t symbol) {
      Term reduced;
      reduced.coef = t.coef;
      bool removed = false;
      for (uint32_t i = 0; i < t.count; i++) {
        if (!removed && t.syms[i] == symbol) {
          removed = true;
          continue;
        }
        reduced.syms[reduced.count++] = t.syms[i];
      }
      return reduced;
    }

    class FadeAnalysis {
    public:
      explicit FadeAnalysis(const Evaluator& evaluator)
        : m_eval(evaluator) { }

      void finish(DxsoMaterialFadeResult& result) const {
        DxsoHighlightFailure failure = m_eval.failure();
        bool anyOutput = false;
        for (uint32_t lane = 0; lane < 4; lane++) {
          anyOutput |= m_eval.outputWritten(lane);
        }
        if (failure == DxsoHighlightFailure::None && !anyOutput) {
          failure = DxsoHighlightFailure::NoColorOutput;
        }
        result.failure = failure;
        result.analyzed = failure == DxsoHighlightFailure::None;
        if (!result.analyzed) {
          return;
        }

        findFadeCandidates(result);
        findParticleColorUses(result);
      }

    private:
      // The terms with their constant registers split off; false past the term budget.
      bool toFadePoly(const Poly& p, DxsoFadePoly& out) const {
        if (p.size() > kMaxFadeLaneTerms) {
          return false;
        }
        out.clear();
        out.reserve(p.size());
        for (const Term& t : p) {
          DxsoFadeTerm term;
          term.coef = t.coef;
          std::array<uint16_t, dxso_symbolic::kMaxTermSymbols> rest;
          uint32_t restCount = 0;
          for (uint32_t i = 0; i < t.count; i++) {
            const dxso_symbolic::Symbol& s = m_eval.symbol(t.syms[i]);
            if (s.liveConstant) {
              term.parameters[term.parameterCount++] = uint16_t(s.reg * 4u + s.component);
            } else {
              rest[restCount++] = t.syms[i];
            }
          }
          if (restCount != 0) {
            term.monomial = std::max<uint64_t>(XXH3_64bits(rest.data(), restCount * sizeof(uint16_t)), 1);
          }
          out.push_back(term);
        }
        std::stable_sort(out.begin(), out.end(),
                         [](const DxsoFadeTerm& a, const DxsoFadeTerm& b) { return a.monomial < b.monomial; });
        return true;
      }

      bool laneFoldsModulator(const uint32_t lane, const uint16_t modulator) const {
        for (const Term& t : m_eval.output(lane)) {
          for (uint32_t i = 0; i < t.count; i++) {
            if (m_eval.symbol(t.syms[i]).kind == SymbolKind::Atom && m_eval.atomDependsOn(t.syms[i], modulator)) {
              return true;
            }
          }
        }
        return false;
      }

      // The lane as offset + M * slope. A power of M above 1, or M also inside something the
      // evaluation did not expand, makes it something other than affine in M.
      bool splitLane(const uint32_t lane, const uint16_t modulator, DxsoFadePoly& offset, DxsoFadePoly& slope) const {
        if (laneFoldsModulator(lane, modulator)) {
          return false;
        }
        Poly modulatorFree, modulatorLinear;
        for (const Term& t : m_eval.output(lane)) {
          const uint32_t occurrences = dxso_symbolic::symbolOccurrences(t, modulator);
          if (occurrences == 0) {
            modulatorFree.push_back(t);
          } else if (occurrences == 1) {
            modulatorLinear.push_back(withoutOneOccurrence(t, modulator));
          } else {
            return false;
          }
        }
        return toFadePoly(modulatorFree, offset) && toFadePoly(modulatorLinear, slope);
      }

      void findFadeCandidates(DxsoMaterialFadeResult& result) const {
        std::set<uint16_t> modulators;
        for (uint32_t lane = 0; lane < 4; lane++) {
          if (!m_eval.outputWritten(lane)) {
            continue;
          }
          for (const Term& t : m_eval.output(lane)) {
            for (uint32_t i = 0; i < t.count; i++) {
              const SymbolKind kind = m_eval.symbol(t.syms[i]).kind;
              if (kind == SymbolKind::MaterialScalar || kind == SymbolKind::MaterialVector) {
                modulators.insert(t.syms[i]);
              }
            }
          }
        }

        std::set<uint32_t> scalarsReachingOutput;
        std::set<uint32_t> scalarsWithFade;
        for (const uint16_t modulator : modulators) {
          const dxso_symbolic::Symbol& symbol = m_eval.symbol(modulator);
          const bool isScalar = symbol.kind == SymbolKind::MaterialScalar;
          if (isScalar) {
            scalarsReachingOutput.insert(symbol.reg);
          }

          DxsoMaterialFade fade;
          fade.kind = isScalar ? DxsoFadeModulatorKind::MaterialScalar : DxsoFadeModulatorKind::MaterialVector;
          fade.reg = symbol.reg;
          fade.component = symbol.component;

          DxsoMaterialFade alpha = fade;
          if (m_eval.outputWritten(3) && splitLane(3, modulator, alpha.offset[3], alpha.slope[3]) && !alpha.slope[3].empty()) {
            alpha.laneMask = kDxsoFadeAlphaLane;
            result.fades.push_back(std::move(alpha));
            if (isScalar) {
              scalarsWithFade.insert(symbol.reg);
            }
          }

          // A lane M never reaches has to be a literal, which the draw then checks against rest: one
          // that varies across the surface would leave it showing, so M is a tint of the lanes it
          // does reach rather than a fade of the colour.
          DxsoMaterialFade color = fade;
          bool colorSplits = true;
          bool colorDepends = false;
          for (uint32_t lane = 0; lane < 3 && colorSplits; lane++) {
            colorSplits = m_eval.outputWritten(lane) && splitLane(lane, modulator, color.offset[lane], color.slope[lane]);
            if (colorSplits && color.slope[lane].empty()) {
              colorSplits = std::all_of(color.offset[lane].begin(), color.offset[lane].end(),
                                        [](const DxsoFadeTerm& t) { return t.monomial == 0 && t.parameterCount == 0; });
            }
            colorDepends |= colorSplits && !color.slope[lane].empty();
          }
          if (colorSplits && colorDepends) {
            color.laneMask = kDxsoFadeColorLanes;
            result.fades.push_back(std::move(color));
            if (isScalar) {
              scalarsWithFade.insert(symbol.reg);
            }
          }
        }

        for (const uint32_t reg : scalarsReachingOutput) {
          if (scalarsWithFade.count(reg) == 0) {
            result.unprovenScalarRegs.push_back(reg);
          }
        }
      }

      // The vertex colour reaches Remix through its texture stage operations, which multiply the
      // sampled colour by one vertex colour channel per output channel and the opacity by its alpha.
      // Fog and other colourless terms never reach Remix's albedo, and are left out of the colour.
      void findParticleColorUses(DxsoMaterialFadeResult& result) const {
        std::array<int32_t, 4> channel = { -1, -1, -1, -1 };
        for (size_t s = 0; s < m_eval.symbolCount(); s++) {
          const dxso_symbolic::Symbol& symbol = m_eval.symbol(uint16_t(s));
          if (symbol.kind == SymbolKind::VertexColor && symbol.component < 4) {
            channel[symbol.component] = int32_t(s);
          }
        }

        auto occurrences = [&](const Term& t, const uint32_t c) {
          return channel[c] < 0 ? 0u : dxso_symbolic::symbolOccurrences(t, uint16_t(channel[c]));
        };
        auto foldsChannel = [&](const uint32_t lane, const uint32_t c) {
          return channel[c] >= 0 && laneFoldsModulator(lane, uint16_t(channel[c]));
        };
        auto carriesColor = [&](const Term& t) {
          if (m_eval.isColorBearing(t)) {
            return true;
          }
          for (uint32_t c = 0; c < 4; c++) {
            if (occurrences(t, c) != 0) {
              return true;
            }
          }
          return false;
        };

        DxsoParticleColorUse& use = result.particleColor;
        bool tintsColor = channel[0] >= 0 || channel[1] >= 0 || channel[2] >= 0;
        bool scalesColor = channel[3] >= 0;
        for (uint32_t lane = 0; lane < 3; lane++) {
          if (!m_eval.outputWritten(lane)) {
            tintsColor = scalesColor = false;
            break;
          }
          bool laneTinted = channel[lane] >= 0;
          bool laneScaled = !foldsChannel(lane, 3);
          for (uint32_t c = 0; c < 3; c++) {
            laneTinted &= !foldsChannel(lane, c);
          }
          bool anyTinted = false;
          bool anyScaled = false;
          Poly tintResidual, scaleResidual;
          for (const Term& t : m_eval.output(lane)) {
            if (!carriesColor(t)) {
              continue;
            }
            if (occurrences(t, lane) == 0) {
              tintResidual.push_back(t);
            } else {
              anyTinted = true;
              for (uint32_t c = 0; c < 3; c++) {
                laneTinted &= occurrences(t, c) == (c == lane ? 1u : 0u);
              }
            }
            if (occurrences(t, 3) == 0) {
              scaleResidual.push_back(t);
            } else {
              anyScaled = true;
              laneScaled &= occurrences(t, 3) == 1;
            }
          }
          tintsColor &= laneTinted && anyTinted && toFadePoly(tintResidual, use.tintResidual[lane]);
          scalesColor &= laneScaled && anyScaled && toFadePoly(scaleResidual, use.scalesColorResidual[lane]);
        }
        use.tintsColor = tintsColor;
        use.scalesColor = scalesColor;

        // The opacity carries no fog, so every term of it counts. Other channels of the colour may
        // shape it too; only the alpha has to scale it linearly.
        bool scalesOpacity = m_eval.outputWritten(3) && channel[3] >= 0 && !foldsChannel(3, 3);
        bool anyScaled = false;
        Poly opacityResidual;
        if (scalesOpacity) {
          for (const Term& t : m_eval.output(3)) {
            const uint32_t n = occurrences(t, 3);
            if (n == 0) {
              opacityResidual.push_back(t);
            } else {
              anyScaled = true;
              scalesOpacity &= n == 1;
            }
          }
        }
        use.scalesOpacity = scalesOpacity && anyScaled && toFadePoly(opacityResidual, use.scalesOpacityResidual);
      }

      const Evaluator& m_eval;
    };

    // A lane's coefficient per monomial key for one draw's constants, in key order. Slot 0 always
    // holds the constant monomial, whose key sorts first.
    struct LaneValues {
      std::array<uint64_t, kMaxFadeLaneTerms + 1> monomial;
      std::array<double, kMaxFadeLaneTerms + 1> coef;
      uint32_t count = 0;
    };

    bool evaluate(const DxsoFadePoly& poly, const float* constants, const uint32_t registerCount, LaneValues& values) {
      if (poly.size() > kMaxFadeLaneTerms) {
        return false;
      }
      values.monomial[0] = 0;
      values.coef[0] = 0.0;
      values.count = 1;
      for (const DxsoFadeTerm& t : poly) {
        double coef = t.coef;
        for (uint32_t i = 0; i < t.parameterCount; i++) {
          if (t.parameters[i] >= registerCount * 4u) {
            return false;
          }
          const float value = constants[t.parameters[i]];
          if (!std::isfinite(value)) {
            return false;
          }
          coef *= value;
        }
        if (values.monomial[values.count - 1] == t.monomial) {
          values.coef[values.count - 1] += coef;
        } else {
          values.monomial[values.count] = t.monomial;
          values.coef[values.count] = coef;
          values.count++;
        }
      }
      return true;
    }

  }

  DxsoMaterialFadeResult analyzeDxsoMaterialFades(
    const uint32_t*             tokens,
    const size_t                tokenCount,
    const DxsoHighlightInputs&  inputs,
    const int32_t               particleColorInputRegister) {
    DxsoMaterialFadeResult result;
    result.failure = DxsoHighlightFailure::NotPixelShader;
    const std::optional<DxsoProgramInfo> programInfo = dxso_symbolic::pixelShaderProgramInfo(tokens, tokenCount);
    if (!programInfo) {
      return result;
    }

    dxso_symbolic::EvaluatorOptions options;
    options.vertexColorInputRegister = particleColorInputRegister;
    options.rangeClampsAreIdentity = true;
    const auto evaluator = std::make_unique<Evaluator>(inputs, options, *programInfo);
    evaluator->run(tokens, tokenCount);
    FadeAnalysis(*evaluator).finish(result);
    return result;
  }

  std::optional<float> dxsoMaterialFadeCoverage(
    const DxsoMaterialFade& fade,
    const float             restValue,
    const float*            constants,
    const uint32_t          registerCount) {
    if (fade.reg >= registerCount) {
      return std::nullopt;
    }
    const float modulator = constants[fade.reg * 4u + (fade.component & 3u)];
    if (!std::isfinite(modulator)) {
      return std::nullopt;
    }

    // Each lane deviates from rest by offset - rest + M * slope. For the draw to fade as one
    // surface, every deviation has to be a multiple of its slope - (ratio + M) * slope - with the
    // same ratio in every lane that depends on M, and the lanes that do not have to sit at rest.
    // When none depends on M for these constants, every lane is at rest and so is the draw.
    std::optional<double> ratio;
    for (uint32_t lane = 0; lane < 4; lane++) {
      if ((fade.laneMask & (1u << lane)) == 0) {
        continue;
      }
      LaneValues deviation, slope;
      if (!evaluate(fade.offset[lane], constants, registerCount, deviation) ||
          !evaluate(fade.slope[lane], constants, registerCount, slope)) {
        return std::nullopt;
      }
      deviation.coef[0] -= restValue;

      // Both are in key order, so one merge pairs their coefficients.
      double deviationNorm = 0.0;
      double slopeNorm = 0.0;
      double projection = 0.0;
      for (uint32_t i = 0, j = 0; i < deviation.count || j < slope.count;) {
        const bool takeDeviation = j == slope.count || (i < deviation.count && deviation.monomial[i] <= slope.monomial[j]);
        const bool takeSlope = i == deviation.count || (j < slope.count && slope.monomial[j] <= deviation.monomial[i]);
        const double u = takeDeviation ? deviation.coef[i++] : 0.0;
        const double v = takeSlope ? slope.coef[j++] : 0.0;
        deviationNorm += u * u;
        slopeNorm += v * v;
        projection += u * v;
      }
      if (slopeNorm <= kVanishTolerance * kVanishTolerance) {
        if (deviationNorm > kVanishTolerance * kVanishTolerance) {
          return std::nullopt;
        }
        continue;
      }

      // What is left of the deviation once its multiple of the slope is taken out.
      const double laneRatio = projection / slopeNorm;
      const double residual = std::max(deviationNorm - laneRatio * projection, 0.0);
      const double scale = std::max(deviationNorm, laneRatio * laneRatio * slopeNorm);
      if (residual > kRelativeTolerance * kRelativeTolerance * scale + kVanishTolerance * kVanishTolerance) {
        return std::nullopt;
      }
      if (ratio && std::abs(*ratio - laneRatio) > kRelativeTolerance * std::max(1.0, std::abs(laneRatio))) {
        return std::nullopt;
      }
      ratio = laneRatio;
    }
    if (!ratio) {
      return 0.0f;
    }

    // The draw is at rest where M = -ratio. A fade that crosses rest inside the range is something
    // else: the colour passes through the framebuffer's own and out the other side.
    const double rest = -*ratio;
    if (rest > kRelativeTolerance && rest < 1.0 - kRelativeTolerance) {
      return std::nullopt;
    }
    const double full = std::abs(*ratio + 1.0) >= std::abs(*ratio) ? 1.0 : 0.0;
    const double span = *ratio + full;
    if (std::abs(span) <= kVanishTolerance) {
      return std::nullopt;
    }
    return float(std::clamp((*ratio + double(modulator)) / span, 0.0, 1.0));
  }

  bool dxsoFadePolyVanishes(
    const DxsoFadePoly&     poly,
    const float*            constants,
    const uint32_t          registerCount) {
    LaneValues values;
    if (!evaluate(poly, constants, registerCount, values)) {
      return false;
    }
    return std::all_of(values.coef.begin(), values.coef.begin() + values.count,
                       [](const double c) { return std::abs(c) <= kVanishTolerance; });
  }

}
