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
#include "dxso_highlight_tints.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <memory>
#include <string>

#include "dxso_ctab.h"
#include "dxso_symbolic_eval.h"

namespace dxvk {

  namespace {

    using dxso_symbolic::Evaluator;
    using dxso_symbolic::Poly;
    using dxso_symbolic::SymbolKind;
    using dxso_symbolic::Term;

    class HighlightProof {
    public:
      explicit HighlightProof(const Evaluator& evaluator)
        : m_eval(evaluator) { }

      void finish(DxsoHighlightResult& result) const {
        DxsoHighlightFailure failure = m_eval.failure();
        if (failure == DxsoHighlightFailure::None &&
            !m_eval.outputWritten(0) && !m_eval.outputWritten(1) && !m_eval.outputWritten(2)) {
          failure = DxsoHighlightFailure::NoColorOutput;
        }
        result.failure = failure;
        result.analyzed = failure == DxsoHighlightFailure::None;
        if (!result.analyzed) {
          return;
        }

        struct PairBuild {
          DxsoHighlightPair pair;
          double glowReference = 0.0;
          float glow = 0.0f;
        };
        std::map<uint32_t, PairBuild> builds;
        std::map<uint32_t, uint16_t> scalarsReachingColor;

        for (uint32_t lane = 0; lane < 3; lane++) {
          if (!m_eval.outputWritten(lane)) {
            continue;
          }
          const Poly& output = m_eval.output(lane);

          std::vector<uint16_t> scalars;
          for (const Term& t : output) {
            for (uint32_t i = 0; i < t.count; i++) {
              const uint16_t s = t.syms[i];
              if (m_eval.symbol(s).kind == SymbolKind::MaterialScalar &&
                  std::find(scalars.begin(), scalars.end(), s) == scalars.end()) {
                scalars.push_back(s);
              }
            }
          }

          for (const uint16_t scalar : scalars) {
            const uint32_t reg = m_eval.symbol(scalar).reg;
            scalarsReachingColor.emplace(reg, scalar);

            Poly strengthFree, strengthLinear;
            splitByStrength(output, scalar, strengthFree, strengthLinear);
            const LaneProof proof = proveLane(strengthFree, strengthLinear);
            if (!proof.proven) {
              continue;
            }

            PairBuild& build = builds[reg];
            build.pair.scalarReg = reg;
            build.pair.colorReg[lane] = int32_t(m_eval.symbol(proof.color).reg);
            build.pair.colorComponent[lane] = m_eval.symbol(proof.color).component;
            if (proof.glowReference > build.glowReference) {
              build.glowReference = proof.glowReference;
              build.glow = float(proof.glow);
            }
          }
        }

        // A tint colours the whole surface. A strength that lerps one channel and does something
        // else to the others is some other effect, whatever that one channel looks like.
        for (const auto& [reg, scalar] : scalarsReachingColor) {
          const auto it = builds.find(reg);
          const bool everyChannel = it != builds.end() &&
            it->second.pair.colorReg[0] >= 0 && it->second.pair.colorReg[1] >= 0 && it->second.pair.colorReg[2] >= 0;
          if (everyChannel) {
            DxsoHighlightPair pair = it->second.pair;
            pair.glowCoefficient.fill(it->second.glow);
            result.pairs.push_back(pair);
          } else if (DxsoHighlightPair pair; proveGlowOnly(scalar, pair)) {
            result.pairs.push_back(pair);
          } else {
            result.unprovenScalarRegs.push_back(reg);
          }
        }
      }

    private:
      struct LaneProof {
        bool     proven = false;
        uint16_t color = 0;
        double   glow = 0.0;
        double   glowReference = 0.0;
      };

      // The output as P0 + S * Q + (higher powers of S, which only the glow's own lerp produces).
      static void splitByStrength(const Poly& output, const uint16_t strength, Poly& strengthFree, Poly& strengthLinear) {
        for (const Term& t : output) {
          const uint32_t occurrences = dxso_symbolic::symbolOccurrences(t, strength);
          if (occurrences == 0) {
            strengthFree.push_back(t);
          } else if (occurrences == 1) {
            Term reduced;
            reduced.coef = t.coef;
            for (uint32_t i = 0; i < t.count; i++) {
              if (t.syms[i] != strength) {
                reduced.syms[reduced.count++] = t.syms[i];
              }
            }
            strengthLinear.push_back(reduced);
          }
        }
        // removing the strength can reorder monomials; it cannot merge two, which would have
        // been one term of the output already
        std::sort(strengthLinear.begin(), strengthLinear.end(), dxso_symbolic::monomialLess);
      }

      // lerp(X, X * V, S) = X + S * (X * V - X). For each colour-bearing monomial m of X - a term
      // of P0 with coefficient p - Q must carry -p * m and p * m * v, with v one component of a
      // material vector. Every such m in the channel has to agree on v.
      //
      // A monomial can admit more than one v. Under a second tint lerp, the first lerp's own
      // terms are monomials of X times its colour, so one of X's monomials also matches that
      // colour: the channel's colour is the one candidate every tinted monomial shares.
      //
      // Every colour-bearing monomial the strength fades out has to be tinted. A layer blend
      // under UE3's (1 - Emissive) diffuse factor, lerp(A, B, S) * (1 - V0), fades A * V0 as well
      // as A, and A alone would otherwise pass as a tint towards V0.
      LaneProof proveLane(const Poly& strengthFree, const Poly& strengthLinear) const {
        LaneProof proof;
        std::vector<uint16_t> shared;
        std::vector<uint16_t> candidates;
        std::vector<const Term*> tinted;

        for (const Term& m : strengthFree) {
          if (!m_eval.isColorBearing(m)) {
            continue;
          }
          const Term* removed = dxso_symbolic::findMonomial(strengthLinear, m);
          if (removed == nullptr || !dxso_symbolic::approxEqual(removed->coef, -m.coef)) {
            continue;
          }
          candidates.clear();
          for (const Term& q : strengthLinear) {
            if (q.count != m.count + 1 || !dxso_symbolic::approxEqual(q.coef, m.coef)) {
              continue;
            }
            Term rest;
            if (!dxso_symbolic::divideMonomial(q, m, rest) || rest.count != 1) {
              continue;
            }
            if (m_eval.symbol(rest.syms[0]).kind == SymbolKind::MaterialVector) {
              candidates.push_back(rest.syms[0]);
            }
          }
          if (candidates.empty()) {
            // faded towards nothing: a blend or a fade, not a tint
            return LaneProof {};
          }
          if (tinted.empty()) {
            shared = candidates;
          } else {
            shared.erase(std::remove_if(shared.begin(), shared.end(), [&](const uint16_t s) {
              return std::find(candidates.begin(), candidates.end(), s) == candidates.end();
            }), shared.end());
          }
          tinted.push_back(&m);
        }

        if (tinted.empty() || shared.size() != 1) {
          return LaneProof {};
        }
        proof.proven = true;
        proof.color = shared[0];

        // The glow: an unlit S * k * m' in Q, where m' is a tinted monomial with only lighting and
        // other colourless factors taken out. Several lighting paths can carry the tinted colour
        // with different literal weights (lightmap transfer coefficients); the least attenuated
        // one, usually AmbientColor or a plain lightmap at weight 1, is the colour's own scale.
        for (const Term& glow : strengthLinear) {
          if (glow.coef <= 0.0 || !m_eval.isColorBearing(glow) || m_eval.hasLighting(glow) ||
              dxso_symbolic::containsSymbol(glow, proof.color)) {
            continue;
          }
          for (const Term* m : tinted) {
            Term rest;
            if (m->count <= glow.count || !dxso_symbolic::divideMonomial(*m, glow, rest) || !m_eval.isFactorOnly(rest)) {
              continue;
            }
            const double k = glow.coef / m->coef;
            if (k > 0.0 && std::abs(m->coef) > proof.glowReference) {
              proof.glowReference = std::abs(m->coef);
              proof.glow = k;
            }
          }
        }
        return proof;
      }

      // One channel gets k * S * T added, with k > 0 and T a channel of a material texture, and nothing
      // else involves the strength: an unlit flash of colour. A strength that scales a texture into
      // several channels is the material's own emission animating, which Remix leaves to the material.
      // The strength squared, or inside a value the evaluation did not expand, is something else.
      bool proveGlowOnly(const uint16_t strength, DxsoHighlightPair& pair) const {
        const Term* glow = nullptr;
        uint32_t glowLane = 0;
        for (uint32_t lane = 0; lane < 3; lane++) {
          if (!m_eval.outputWritten(lane)) {
            continue;
          }
          for (const Term& t : m_eval.output(lane)) {
            for (uint32_t i = 0; i < t.count; i++) {
              if (m_eval.symbol(t.syms[i]).kind == SymbolKind::Atom && m_eval.atomDependsOn(t.syms[i], strength)) {
                return false;
              }
            }
            const uint32_t occurrences = dxso_symbolic::symbolOccurrences(t, strength);
            if (occurrences == 0) {
              continue;
            }
            if (occurrences > 1 || glow != nullptr || t.count != 2 || t.coef <= 0.0) {
              return false;
            }
            glow = &t;
            glowLane = lane;
          }
        }
        if (glow == nullptr) {
          return false;
        }
        const dxso_symbolic::Symbol& sample = m_eval.symbol(glow->syms[glow->syms[0] == strength ? 1 : 0]);
        if (sample.kind != SymbolKind::Sample || sample.lighting) {
          return false;
        }
        pair.scalarReg = m_eval.symbol(strength).reg;
        pair.glowCoefficient[glowLane] = float(glow->coef);
        pair.glowOnly = true;
        pair.glowSampler = sample.reg;
        pair.glowComponent = sample.component;
        return true;
      }

      const Evaluator& m_eval;
    };

  }

  DxsoHighlightResult analyzeDxsoHighlightTints(
    const uint32_t*             tokens,
    const size_t                tokenCount,
    const DxsoHighlightInputs&  inputs) {
    DxsoHighlightResult result;
    result.failure = DxsoHighlightFailure::NotPixelShader;
    const std::optional<DxsoProgramInfo> programInfo = dxso_symbolic::pixelShaderProgramInfo(tokens, tokenCount);
    if (!programInfo) {
      return result;
    }

    const auto evaluator = std::make_unique<Evaluator>(inputs, dxso_symbolic::EvaluatorOptions {}, *programInfo);
    evaluator->run(tokens, tokenCount);
    HighlightProof(*evaluator).finish(result);
    return result;
  }

  DxsoHighlightInputs dxsoHighlightInputsFromUe3Ctab(const DxsoCtab& ctab) {
    constexpr uint16_t kRegisterSetFloat4 = 2u;
    constexpr uint16_t kRegisterSetSampler = 3u;

    DxsoHighlightInputs inputs;
    for (const DxsoCtab::Constant& c : ctab.m_constantData) {
      if (c.registerCount == 0) {
        continue;
      }
      std::string name = c.name;
      for (char& ch : name) {
        ch = char(std::tolower(static_cast<unsigned char>(ch)));
      }
      const uint32_t end = c.registerIndex + c.registerCount;

      if (c.registerSet == kRegisterSetSampler) {
        if (name.find("lightmap") != std::string::npos || name.find("bsplinetexture") != std::string::npos) {
          for (uint32_t s = c.registerIndex; s < end && s < 32u; s++) {
            inputs.lightingSamplerMask |= 1u << s;
          }
        }
        continue;
      }
      if (c.registerSet != kRegisterSetFloat4) {
        continue;
      }

      std::bitset<kDxsoHighlightMaxConstRegs>* regs = nullptr;
      if (name.find("uniformscalar_") != std::string::npos) {
        regs = &inputs.scalarRegs;
      } else if (name.find("uniformvector_") != std::string::npos) {
        regs = &inputs.vectorRegs;
      } else if (name.find("ambientcolorandskyfactor") != std::string::npos ||
                 name.find("upperskycolor") != std::string::npos ||
                 name.find("lowerskycolor") != std::string::npos ||
                 name.find("lightmapscale") != std::string::npos) {
        regs = &inputs.lightingConstRegs;
      }
      if (regs == nullptr) {
        continue;
      }
      for (uint32_t r = c.registerIndex; r < end && r < kDxsoHighlightMaxConstRegs; r++) {
        regs->set(r);
      }
    }
    return inputs;
  }

  const char* dxsoHighlightFailureName(const DxsoHighlightFailure failure) {
    switch (failure) {
    case DxsoHighlightFailure::None:               return "none";
    case DxsoHighlightFailure::NotPixelShader:     return "not a ps_2_0+ pixel shader";
    case DxsoHighlightFailure::FlowControl:        return "flow control";
    case DxsoHighlightFailure::RelativeAddressing: return "relative constant addressing";
    case DxsoHighlightFailure::TooComplex:         return "expression too complex";
    case DxsoHighlightFailure::NoColorOutput:      return "no colour output";
    default:                                       return "?";
    }
  }

}
