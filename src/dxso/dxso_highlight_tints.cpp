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
#include <initializer_list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

#include "dxso_code.h"
#include "dxso_common.h"
#include "dxso_ctab.h"
#include "dxso_decoder.h"

#include "../util/xxHash/xxhash.h"

namespace dxvk {

  namespace {

    constexpr uint32_t kMaxTemps = 64;
    // Runner Vision networks fit well inside these budgets; the shaders that exceed them are rim
    // and fresnel networks with many scalar-weighted sums.
    constexpr uint32_t kMaxTermSymbols = 16;
    constexpr size_t   kMaxPolyTerms = 8192;
    constexpr size_t   kMaxSymbols = 0xFFFF;
    constexpr double   kZeroCoefficient = 1.0e-9;
    // Coefficients are products of the shader's literals, so matching ones differ only by float
    // rounding.
    constexpr double   kMatchTolerance = 1.0e-4;

    // Atom hash seeds, so different operations on the same inputs give different atoms.
    constexpr uint32_t kTagCollapse   = 0x10000u;
    constexpr uint32_t kTagPredicated = 0x20000u;
    constexpr uint32_t kTagModifier   = 0x30000u;
    constexpr uint32_t kTagAllLanes   = 0xFFu;

    enum class SymbolKind : uint8_t {
      MaterialScalar,  // the x component of a UniformScalar_* register
      MaterialVector,  // a component of a UniformVector_* register
      Constant,        // any other constant register component
      Interpolant,     // an input register component (texcoords, colours, vPos, vFace)
      Sample,          // one component of one texture read
      Atom,            // a value the evaluation does not expand
    };

    struct Symbol {
      SymbolKind kind = SymbolKind::Atom;
      uint16_t   reg = 0;
      uint8_t    component = 0;
      // a lightmap sample or lighting constant, or derived from one
      bool       lighting = false;
      // a material texture sample, or derived from one
      bool       color = false;
      // a material parameter, or derived from one
      bool       material = false;
    };

    // One monomial: a coefficient times a sorted multiset of symbols.
    struct Term {
      double   coef = 0.0;
      uint8_t  count = 0;
      std::array<uint16_t, kMaxTermSymbols> syms = {};
    };

    // Terms sorted by monomial, each monomial once, no zero coefficients.
    using Poly = std::vector<Term>;

    int compareMonomial(const Term& a, const Term& b) {
      if (a.count != b.count) {
        return a.count < b.count ? -1 : 1;
      }
      for (uint32_t i = 0; i < a.count; i++) {
        if (a.syms[i] != b.syms[i]) {
          return a.syms[i] < b.syms[i] ? -1 : 1;
        }
      }
      return 0;
    }

    bool monomialLess(const Term& a, const Term& b) {
      return compareMonomial(a, b) < 0;
    }

    bool approxEqual(const double a, const double b) {
      return std::abs(a - b) <= kMatchTolerance * std::max(std::abs(a), std::abs(b));
    }

    Poly polyConstant(const double value) {
      Poly p;
      if (std::abs(value) > kZeroCoefficient) {
        Term t;
        t.coef = value;
        p.push_back(t);
      }
      return p;
    }

    Poly polySymbol(const uint16_t symbol) {
      Term t;
      t.coef = 1.0;
      t.count = 1;
      t.syms[0] = symbol;
      return Poly { t };
    }

    const Term* findMonomial(const Poly& p, const Term& key) {
      const auto it = std::lower_bound(p.begin(), p.end(), key, monomialLess);
      return (it != p.end() && compareMonomial(*it, key) == 0) ? &*it : nullptr;
    }

    // The symbols of `dividend` left once those of `divisor` are taken out, when it divides.
    bool divideMonomial(const Term& dividend, const Term& divisor, Term& remainder) {
      remainder = Term {};
      uint32_t j = 0;
      for (uint32_t i = 0; i < dividend.count; i++) {
        if (j < divisor.count && dividend.syms[i] == divisor.syms[j]) {
          j++;
          continue;
        }
        if (j < divisor.count && divisor.syms[j] < dividend.syms[i]) {
          return false;
        }
        remainder.syms[remainder.count++] = dividend.syms[i];
      }
      return j == divisor.count;
    }

    bool containsSymbol(const Term& t, const uint16_t symbol) {
      return std::binary_search(t.syms.begin(), t.syms.begin() + t.count, symbol);
    }

    uint64_t hashPoly(uint64_t seed, const Poly& p) {
      const uint32_t size = uint32_t(p.size());
      seed = XXH3_64bits_withSeed(&size, sizeof(size), seed);
      for (const Term& t : p) {
        seed = XXH3_64bits_withSeed(&t.coef, sizeof(t.coef), seed);
        seed = XXH3_64bits_withSeed(t.syms.data(), t.count * sizeof(uint16_t), seed);
      }
      return seed;
    }

    uint32_t sourceOperandCount(const DxsoOpcode op) {
      switch (op) {
      case DxsoOpcode::Mov:
      case DxsoOpcode::Rcp:
      case DxsoOpcode::Rsq:
      case DxsoOpcode::Exp:
      case DxsoOpcode::Log:
      case DxsoOpcode::ExpP:
      case DxsoOpcode::LogP:
      case DxsoOpcode::Frc:
      case DxsoOpcode::Abs:
      case DxsoOpcode::Nrm:
      case DxsoOpcode::Sgn:
      case DxsoOpcode::Lit:
      case DxsoOpcode::Mova:
      case DxsoOpcode::DsX:
      case DxsoOpcode::DsY:
      case DxsoOpcode::SinCos:
        return 1;
      case DxsoOpcode::Add:
      case DxsoOpcode::Sub:
      case DxsoOpcode::Mul:
      case DxsoOpcode::Dp3:
      case DxsoOpcode::Dp4:
      case DxsoOpcode::Min:
      case DxsoOpcode::Max:
      case DxsoOpcode::Slt:
      case DxsoOpcode::Sge:
      case DxsoOpcode::Dst:
      case DxsoOpcode::Pow:
      case DxsoOpcode::Crs:
      case DxsoOpcode::M4x4:
      case DxsoOpcode::M4x3:
      case DxsoOpcode::M3x4:
      case DxsoOpcode::M3x3:
      case DxsoOpcode::M3x2:
      case DxsoOpcode::Bem:
        return 2;
      case DxsoOpcode::Mad:
      case DxsoOpcode::Lrp:
      case DxsoOpcode::Cmp:
      case DxsoOpcode::Cnd:
      case DxsoOpcode::Dp2Add:
        return 3;
      default:
        return 0;
      }
    }

    class HighlightAnalyzer {
    public:
      HighlightAnalyzer(const DxsoHighlightInputs& inputs, const DxsoProgramInfo& programInfo)
        : m_inputs(inputs), m_programInfo(programInfo) {
        m_isDef.fill(0);
      }

      void run(const uint32_t* tokens, const size_t tokenCount) {
        DxsoDecodeContext decoder(m_programInfo);
        DxsoCodeIter iter(tokens + 1);
        for (size_t guard = 0; guard < tokenCount && m_failure == DxsoHighlightFailure::None &&
                               decoder.decodeInstruction(iter); guard++) {
          processInstruction(decoder.getInstructionContext());
        }
      }

      void finish(DxsoHighlightResult& result) {
        if (m_failure == DxsoHighlightFailure::None &&
            !m_outputWritten[0] && !m_outputWritten[1] && !m_outputWritten[2]) {
          m_failure = DxsoHighlightFailure::NoColorOutput;
        }
        result.failure = m_failure;
        result.analyzed = m_failure == DxsoHighlightFailure::None;
        if (!result.analyzed) {
          return;
        }

        struct PairBuild {
          DxsoHighlightPair pair;
          double glowReference = 0.0;
        };
        std::map<uint32_t, PairBuild> builds;
        std::set<uint32_t> scalarsReachingColor;

        for (uint32_t lane = 0; lane < 3; lane++) {
          if (!m_outputWritten[lane]) {
            continue;
          }
          const Poly& output = m_output[lane];

          std::vector<uint16_t> scalars;
          for (const Term& t : output) {
            for (uint32_t i = 0; i < t.count; i++) {
              const uint16_t s = t.syms[i];
              if (m_symbols[s].kind == SymbolKind::MaterialScalar &&
                  std::find(scalars.begin(), scalars.end(), s) == scalars.end()) {
                scalars.push_back(s);
              }
            }
          }

          for (const uint16_t scalar : scalars) {
            const uint32_t reg = m_symbols[scalar].reg;
            scalarsReachingColor.insert(reg);

            Poly strengthFree, strengthLinear;
            splitByStrength(output, scalar, strengthFree, strengthLinear);
            const LaneProof proof = proveLane(strengthFree, strengthLinear);
            if (!proof.proven) {
              continue;
            }

            PairBuild& build = builds[reg];
            build.pair.scalarReg = reg;
            build.pair.colorReg[lane] = int32_t(m_symbols[proof.color].reg);
            build.pair.colorComponent[lane] = m_symbols[proof.color].component;
            if (proof.glowReference > build.glowReference) {
              build.glowReference = proof.glowReference;
              build.pair.glowCoefficient = float(proof.glow);
            }
          }
        }

        // A tint colours the whole surface. A strength that lerps one channel and does something
        // else to the others is some other effect, whatever that one channel looks like.
        for (const uint32_t reg : scalarsReachingColor) {
          const auto it = builds.find(reg);
          const bool everyChannel = it != builds.end() &&
            it->second.pair.colorReg[0] >= 0 && it->second.pair.colorReg[1] >= 0 && it->second.pair.colorReg[2] >= 0;
          if (everyChannel) {
            result.pairs.push_back(it->second.pair);
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

      void fail(const DxsoHighlightFailure failure) {
        if (m_failure == DxsoHighlightFailure::None) {
          m_failure = failure;
        }
      }

      // --- symbols ------------------------------------------------------------------------

      static uint64_t baseKey(const SymbolKind kind, const uint32_t space, const uint32_t num, const uint32_t component) {
        return (uint64_t(kind) << 56) | (uint64_t(space & 0xFFu) << 48) |
               (uint64_t(num & 0xFFFFFFFFu) << 8) | uint64_t(component & 0xFFu);
      }

      // Base keys keep the kind in the top byte, which never reaches the top bit.
      static uint64_t atomKey(const uint64_t contentHash) {
        return (1ull << 63) | (contentHash >> 1);
      }

      uint16_t makeSymbol(const uint64_t key, const Symbol& proto) {
        const auto it = m_symbolIndex.find(key);
        if (it != m_symbolIndex.end()) {
          return it->second;
        }
        if (m_symbols.size() >= kMaxSymbols) {
          fail(DxsoHighlightFailure::TooComplex);
          return 0;
        }
        const uint16_t id = uint16_t(m_symbols.size());
        m_symbols.push_back(proto);
        m_symbolIndex.emplace(key, id);
        return id;
      }

      void flagsOf(const Poly& p, bool& lighting, bool& color, bool& material) const {
        for (const Term& t : p) {
          for (uint32_t i = 0; i < t.count; i++) {
            const Symbol& s = m_symbols[t.syms[i]];
            lighting |= s.lighting;
            color |= s.color;
            material |= s.material;
          }
        }
      }

      // A value the evaluation does not expand. Equal inputs give the same atom, so a value
      // computed once and read twice stays one symbol.
      Poly opaque(const uint32_t tag, std::initializer_list<const Poly*> inputs) {
        uint64_t hash = tag;
        Symbol s;
        s.kind = SymbolKind::Atom;
        for (const Poly* p : inputs) {
          hash = hashPoly(hash, *p);
          flagsOf(*p, s.lighting, s.color, s.material);
        }
        return polySymbol(makeSymbol(atomKey(hash), s));
      }

      bool isColorBearing(const Term& t) const {
        for (uint32_t i = 0; i < t.count; i++) {
          const Symbol& s = m_symbols[t.syms[i]];
          if (s.color || s.kind == SymbolKind::MaterialVector) {
            return true;
          }
        }
        return false;
      }

      bool hasLighting(const Term& t) const {
        for (uint32_t i = 0; i < t.count; i++) {
          if (m_symbols[t.syms[i]].lighting) {
            return true;
          }
        }
        return false;
      }

      // Only lighting, interpolants and other colourless, parameter-free factors.
      bool isFactorOnly(const Term& t) const {
        for (uint32_t i = 0; i < t.count; i++) {
          const Symbol& s = m_symbols[t.syms[i]];
          if (s.color || s.material) {
            return false;
          }
        }
        return true;
      }

      // --- polynomial arithmetic, failing the analysis past the term budget -----------------

      Poly add(const Poly& a, const Poly& b, const double bScale = 1.0) {
        Poly result;
        result.reserve(a.size() + b.size());
        size_t i = 0, j = 0;
        while (i < a.size() || j < b.size()) {
          const int cmp = i == a.size() ? 1 : j == b.size() ? -1 : compareMonomial(a[i], b[j]);
          Term t;
          if (cmp < 0) {
            t = a[i++];
          } else if (cmp > 0) {
            t = b[j++];
            t.coef *= bScale;
          } else {
            t = a[i++];
            t.coef += bScale * b[j++].coef;
          }
          if (std::abs(t.coef) > kZeroCoefficient) {
            result.push_back(t);
          }
        }
        if (result.size() > kMaxPolyTerms) {
          fail(DxsoHighlightFailure::TooComplex);
          return {};
        }
        return result;
      }

      Poly mul(const Poly& a, const Poly& b) {
        if (a.empty() || b.empty()) {
          return {};
        }
        if (a.size() * b.size() > kMaxPolyTerms * 16) {
          fail(DxsoHighlightFailure::TooComplex);
          return {};
        }
        Poly products;
        products.reserve(a.size() * b.size());
        for (const Term& x : a) {
          for (const Term& y : b) {
            if (uint32_t(x.count) + y.count > kMaxTermSymbols) {
              fail(DxsoHighlightFailure::TooComplex);
              return {};
            }
            Term t;
            t.coef = x.coef * y.coef;
            t.count = uint8_t(x.count + y.count);
            std::merge(x.syms.begin(), x.syms.begin() + x.count,
                       y.syms.begin(), y.syms.begin() + y.count, t.syms.begin());
            products.push_back(t);
          }
        }
        std::sort(products.begin(), products.end(), monomialLess);
        Poly result;
        result.reserve(products.size());
        for (const Term& t : products) {
          if (!result.empty() && compareMonomial(result.back(), t) == 0) {
            result.back().coef += t.coef;
          } else {
            result.push_back(t);
          }
        }
        result.erase(std::remove_if(result.begin(), result.end(),
                                    [](const Term& t) { return std::abs(t.coef) <= kZeroCoefficient; }),
                     result.end());
        if (result.size() > kMaxPolyTerms) {
          fail(DxsoHighlightFailure::TooComplex);
          return {};
        }
        return result;
      }

      // scale * p + offset
      Poly affine(Poly p, const double scale, const double offset) {
        for (Term& t : p) {
          t.coef *= scale;
        }
        return add(p, polyConstant(offset));
      }

      // A value with no material parameter in it is replaced by one atom: the lightmap filter,
      // normal and reflection math never need expanding, and folding them keeps the rest small.
      // A colour still being multiplied by lighting stays expanded, because the glow match has
      // to see the lighting factor apart from the colour it scales.
      Poly collapse(Poly value) {
        if (value.empty() ||
            (value.size() == 1 && value[0].count == 0) ||
            (value.size() == 1 && value[0].count == 1 && value[0].coef == 1.0)) {
          return value;
        }
        bool materialSymbol = false;
        Symbol s;
        s.kind = SymbolKind::Atom;
        for (const Term& t : value) {
          for (uint32_t i = 0; i < t.count; i++) {
            const Symbol& sym = m_symbols[t.syms[i]];
            materialSymbol |= sym.kind == SymbolKind::MaterialScalar || sym.kind == SymbolKind::MaterialVector;
            s.lighting |= sym.lighting;
            s.color |= sym.color;
            s.material |= sym.material;
          }
        }
        if (materialSymbol || (s.color && s.lighting)) {
          return value;
        }
        return polySymbol(makeSymbol(atomKey(hashPoly(kTagCollapse, value)), s));
      }

      // --- operands -------------------------------------------------------------------------

      Poly readConstant(const uint32_t reg, const uint32_t component) {
        if (reg < kDxsoHighlightMaxConstRegs && m_isDef[reg]) {
          return polyConstant(m_defValues[reg][component]);
        }
        const bool inRange = reg < kDxsoHighlightMaxConstRegs;
        Symbol s;
        s.reg = uint16_t(reg);
        s.component = uint8_t(component);
        if (inRange && m_inputs.scalarRegs.test(reg) && component == 0) {
          // A UniformScalar_* is declared float, so only its x component is the parameter.
          s.kind = SymbolKind::MaterialScalar;
          s.material = true;
        } else if (inRange && m_inputs.vectorRegs.test(reg)) {
          s.kind = SymbolKind::MaterialVector;
          s.material = true;
        } else {
          s.kind = SymbolKind::Constant;
          s.lighting = inRange && m_inputs.lightingConstRegs.test(reg);
        }
        return polySymbol(makeSymbol(baseKey(s.kind, uint32_t(DxsoRegisterType::Const), reg, component), s));
      }

      Poly applySourceModifier(const DxsoRegModifier modifier, Poly value) {
        switch (modifier) {
        case DxsoRegModifier::None:    return value;
        case DxsoRegModifier::Neg:     return affine(std::move(value), -1.0, 0.0);
        case DxsoRegModifier::Bias:    return affine(std::move(value), 1.0, -0.5);
        case DxsoRegModifier::BiasNeg: return affine(std::move(value), -1.0, 0.5);
        case DxsoRegModifier::Sign:    return affine(std::move(value), 2.0, -1.0);
        case DxsoRegModifier::SignNeg: return affine(std::move(value), -2.0, 1.0);
        case DxsoRegModifier::Comp:    return affine(std::move(value), -1.0, 1.0);
        case DxsoRegModifier::X2:      return affine(std::move(value), 2.0, 0.0);
        case DxsoRegModifier::X2Neg:   return affine(std::move(value), -2.0, 0.0);
        case DxsoRegModifier::AbsNeg:
          return affine(opaque(kTagModifier | uint32_t(DxsoRegModifier::Abs), { &value }), -1.0, 0.0);
        default:
          // abs, and the ps_1_x projective and boolean modifiers
          return opaque(kTagModifier | uint32_t(modifier), { &value });
        }
      }

      Poly readSourceLane(const DxsoRegister& reg, const uint32_t component) {
        if (reg.hasRelative) {
          fail(DxsoHighlightFailure::RelativeAddressing);
          return {};
        }
        const uint32_t num = reg.id.num;
        Poly value;
        switch (reg.id.type) {
        case DxsoRegisterType::Temp:
        case DxsoRegisterType::TempFloat16:
          if (num < kMaxTemps) {
            value = m_temps[num][component & 3u];
          }
          break;
        case DxsoRegisterType::Const:
          value = readConstant(num, component & 3u);
          break;
        case DxsoRegisterType::Input:
        case DxsoRegisterType::Texture:
        case DxsoRegisterType::PixelTexcoord:
        case DxsoRegisterType::MiscType: {
          Symbol s;
          s.kind = SymbolKind::Interpolant;
          s.reg = uint16_t(num);
          s.component = uint8_t(component);
          value = polySymbol(makeSymbol(baseKey(s.kind, uint32_t(reg.id.type), num, component), s));
          break;
        }
        default: {
          // integer, boolean, loop and predicate registers: colourless engine inputs
          Symbol s;
          s.kind = SymbolKind::Constant;
          s.reg = uint16_t(num);
          s.component = uint8_t(component);
          value = polySymbol(makeSymbol(baseKey(s.kind, uint32_t(reg.id.type), num, component), s));
          break;
        }
        }
        return applySourceModifier(reg.modifier, std::move(value));
      }

      Poly readLane(const DxsoInstructionContext& ctx, const uint32_t srcIndex, const uint32_t lane) {
        const DxsoRegister& src = ctx.src[srcIndex];
        return readSourceLane(src, src.swizzle[lane]);
      }

      // --- instructions ---------------------------------------------------------------------

      void processInstruction(const DxsoInstructionContext& ctx) {
        const DxsoOpcode op = ctx.instruction.opcode;
        switch (op) {
        case DxsoOpcode::Def:
          if (ctx.dst.id.type == DxsoRegisterType::Const && ctx.dst.id.num < kDxsoHighlightMaxConstRegs) {
            m_isDef[ctx.dst.id.num] = 1;
            for (uint32_t c = 0; c < 4; c++) {
              m_defValues[ctx.dst.id.num][c] = ctx.def.float32[c];
            }
          }
          return;
        case DxsoOpcode::DefI:
        case DxsoOpcode::DefB:
        case DxsoOpcode::Dcl:
        case DxsoOpcode::Nop:
        case DxsoOpcode::Comment:
        case DxsoOpcode::End:
        case DxsoOpcode::Phase:
        case DxsoOpcode::SetP:
          return;
        case DxsoOpcode::TexKill:
          // the tested register is encoded as the destination but is only read
          return;
        case DxsoOpcode::If:
        case DxsoOpcode::Ifc:
        case DxsoOpcode::Else:
        case DxsoOpcode::EndIf:
        case DxsoOpcode::Rep:
        case DxsoOpcode::EndRep:
        case DxsoOpcode::Loop:
        case DxsoOpcode::EndLoop:
        case DxsoOpcode::Break:
        case DxsoOpcode::BreakC:
        case DxsoOpcode::BreakP:
        case DxsoOpcode::Call:
        case DxsoOpcode::CallNz:
        case DxsoOpcode::Ret:
        case DxsoOpcode::Label:
          fail(DxsoHighlightFailure::FlowControl);
          return;
        case DxsoOpcode::Tex:
        case DxsoOpcode::TexLdl:
        case DxsoOpcode::TexLdd:
          processSample(ctx);
          return;
        default:
          processArithmetic(ctx);
          return;
        }
      }

      void processSample(const DxsoInstructionContext& ctx) {
        // ps_2_0+: dst temp, src0 coordinate, src1 sampler. Every read is a fresh value.
        const DxsoRegister& samplerReg = ctx.src[1];
        const uint32_t sampler = samplerReg.id.num;
        const bool lighting = sampler < 32u && ((m_inputs.lightingSamplerMask >> sampler) & 1u) != 0;
        const uint32_t instance = m_sampleCount++;
        std::array<Poly, 4> results;
        for (uint32_t lane = 0; lane < 4; lane++) {
          if (!ctx.dst.mask[lane]) {
            continue;
          }
          Symbol s;
          s.kind = SymbolKind::Sample;
          s.reg = uint16_t(sampler);
          s.component = uint8_t(samplerReg.swizzle[lane]);
          s.lighting = lighting;
          s.color = !lighting;
          results[lane] = polySymbol(makeSymbol(baseKey(s.kind, sampler, instance, s.component), s));
        }
        writeResult(ctx, results);
      }

      void processArithmetic(const DxsoInstructionContext& ctx) {
        const DxsoOpcode op = ctx.instruction.opcode;
        const uint32_t srcCount = sourceOperandCount(op);
        const DxsoRegMask mask = ctx.dst.mask;
        std::array<Poly, 4> results;

        auto forEachLane = [&](auto&& fn) {
          for (uint32_t lane = 0; lane < 4 && m_failure == DxsoHighlightFailure::None; lane++) {
            if (mask[lane]) {
              results[lane] = fn(lane);
            }
          }
        };
        auto replicate = [&](const Poly& value) {
          for (uint32_t lane = 0; lane < 4; lane++) {
            if (mask[lane]) {
              results[lane] = value;
            }
          }
        };
        auto dot = [&](const uint32_t components) {
          Poly sum;
          for (uint32_t c = 0; c < components && m_failure == DxsoHighlightFailure::None; c++) {
            sum = add(sum, mul(readLane(ctx, 0, c), readLane(ctx, 1, c)));
          }
          return sum;
        };

        switch (op) {
        case DxsoOpcode::Mov:
          forEachLane([&](uint32_t lane) { return readLane(ctx, 0, lane); });
          break;
        case DxsoOpcode::Add:
          forEachLane([&](uint32_t lane) { return add(readLane(ctx, 0, lane), readLane(ctx, 1, lane)); });
          break;
        case DxsoOpcode::Sub:
          forEachLane([&](uint32_t lane) { return add(readLane(ctx, 0, lane), readLane(ctx, 1, lane), -1.0); });
          break;
        case DxsoOpcode::Mul:
          forEachLane([&](uint32_t lane) { return mul(readLane(ctx, 0, lane), readLane(ctx, 1, lane)); });
          break;
        case DxsoOpcode::Mad:
          forEachLane([&](uint32_t lane) {
            return add(mul(readLane(ctx, 0, lane), readLane(ctx, 1, lane)), readLane(ctx, 2, lane));
          });
          break;
        case DxsoOpcode::Lrp:
          // src0 * src1 + (1 - src0) * src2
          forEachLane([&](uint32_t lane) {
            const Poly weight = readLane(ctx, 0, lane);
            return add(mul(weight, readLane(ctx, 1, lane)), mul(affine(weight, -1.0, 1.0), readLane(ctx, 2, lane)));
          });
          break;
        case DxsoOpcode::Dp2Add:
          replicate(add(dot(2), readLane(ctx, 2, 0)));
          break;
        case DxsoOpcode::Dp3:
          replicate(dot(3));
          break;
        case DxsoOpcode::Dp4:
          replicate(dot(4));
          break;
        case DxsoOpcode::Rcp:
        case DxsoOpcode::Rsq:
        case DxsoOpcode::Exp:
        case DxsoOpcode::Log:
        case DxsoOpcode::ExpP:
        case DxsoOpcode::LogP: {
          // scalar ops read the first swizzle component and replicate
          const Poly in = readLane(ctx, 0, 0);
          replicate(opaque((uint32_t(op) << 8) | kTagAllLanes, { &in }));
          break;
        }
        case DxsoOpcode::Pow: {
          const Poly base = readLane(ctx, 0, 0);
          const Poly exponent = readLane(ctx, 1, 0);
          replicate(opaque((uint32_t(op) << 8) | kTagAllLanes, { &base, &exponent }));
          break;
        }
        case DxsoOpcode::Min:
        case DxsoOpcode::Max:
        case DxsoOpcode::Slt:
        case DxsoOpcode::Sge:
        case DxsoOpcode::Cmp:
        case DxsoOpcode::Cnd:
        case DxsoOpcode::Frc:
        case DxsoOpcode::Abs:
        case DxsoOpcode::Sgn:
        case DxsoOpcode::DsX:
        case DxsoOpcode::DsY:
          // component-wise, but not polynomial
          forEachLane([&](uint32_t lane) {
            std::array<Poly, 3> in;
            for (uint32_t i = 0; i < srcCount && i < 3; i++) {
              in[i] = readLane(ctx, i, lane);
            }
            return opaque((uint32_t(op) << 8) | kTagAllLanes, { &in[0], &in[1], &in[2] });
          });
          break;
        default: {
          // normalise, cross product, matrix transforms and anything unmodelled: every result
          // lane depends on every source component
          std::array<Poly, 12> in;
          for (uint32_t i = 0; i < srcCount && i < 3; i++) {
            for (uint32_t c = 0; c < 4; c++) {
              in[i * 4 + c] = readLane(ctx, i, c);
            }
          }
          forEachLane([&](uint32_t lane) {
            return opaque((uint32_t(op) << 8) | lane, {
              &in[0], &in[1], &in[2], &in[3], &in[4], &in[5], &in[6], &in[7], &in[8], &in[9], &in[10], &in[11] });
          });
          break;
        }
        }

        if (m_failure != DxsoHighlightFailure::None) {
          return;
        }

        // _sat is treated as the identity: the colour math it clamps stays in range, and the lerp
        // it would wrap is still the lerp.
        for (uint32_t lane = 0; lane < 4; lane++) {
          if (!mask[lane]) {
            continue;
          }
          if (ctx.instruction.predicated) {
            // either the new value or the old one, depending on the predicate
            const Poly previous = currentValue(ctx.dst, lane);
            results[lane] = opaque(kTagPredicated | lane, { &results[lane], &previous });
          }
          results[lane] = collapse(std::move(results[lane]));
        }
        writeResult(ctx, results);
      }

      Poly currentValue(const DxsoRegister& dst, const uint32_t lane) const {
        if ((dst.id.type == DxsoRegisterType::Temp || dst.id.type == DxsoRegisterType::TempFloat16) && dst.id.num < kMaxTemps) {
          return m_temps[dst.id.num][lane];
        }
        if (dst.id.type == DxsoRegisterType::ColorOut && dst.id.num == 0 && lane < 3) {
          return m_output[lane];
        }
        return {};
      }

      void writeResult(const DxsoInstructionContext& ctx, std::array<Poly, 4>& results) {
        const DxsoRegister& dst = ctx.dst;
        for (uint32_t lane = 0; lane < 4; lane++) {
          if (!dst.mask[lane]) {
            continue;
          }
          if (dst.id.type == DxsoRegisterType::Temp || dst.id.type == DxsoRegisterType::TempFloat16) {
            if (dst.id.num < kMaxTemps) {
              m_temps[dst.id.num][lane] = std::move(results[lane]);
            }
          } else if (dst.id.type == DxsoRegisterType::ColorOut && dst.id.num == 0 && lane < 3) {
            m_output[lane] = std::move(results[lane]);
            m_outputWritten[lane] = true;
          }
        }
      }

      // --- the proof ------------------------------------------------------------------------

      // The output as P0 + S * Q + (higher powers of S, which only the glow's own lerp produces).
      void splitByStrength(const Poly& output, const uint16_t strength, Poly& strengthFree, Poly& strengthLinear) const {
        for (const Term& t : output) {
          uint32_t occurrences = 0;
          for (uint32_t i = 0; i < t.count; i++) {
            occurrences += t.syms[i] == strength ? 1u : 0u;
          }
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
        std::sort(strengthLinear.begin(), strengthLinear.end(), monomialLess);
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
          if (!isColorBearing(m)) {
            continue;
          }
          const Term* removed = findMonomial(strengthLinear, m);
          if (removed == nullptr || !approxEqual(removed->coef, -m.coef)) {
            continue;
          }
          candidates.clear();
          for (const Term& q : strengthLinear) {
            if (q.count != m.count + 1 || !approxEqual(q.coef, m.coef)) {
              continue;
            }
            Term rest;
            if (!divideMonomial(q, m, rest) || rest.count != 1) {
              continue;
            }
            if (m_symbols[rest.syms[0]].kind == SymbolKind::MaterialVector) {
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
          if (glow.coef <= 0.0 || !isColorBearing(glow) || hasLighting(glow) || containsSymbol(glow, proof.color)) {
            continue;
          }
          for (const Term* m : tinted) {
            Term rest;
            if (m->count <= glow.count || !divideMonomial(*m, glow, rest) || !isFactorOnly(rest)) {
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

      const DxsoHighlightInputs& m_inputs;
      DxsoProgramInfo            m_programInfo;
      DxsoHighlightFailure       m_failure = DxsoHighlightFailure::None;

      std::vector<Symbol>                    m_symbols;
      std::unordered_map<uint64_t, uint16_t> m_symbolIndex;
      uint32_t                               m_sampleCount = 0;

      std::array<std::array<Poly, 4>, kMaxTemps> m_temps;
      std::array<Poly, 3>                        m_output;
      std::array<bool, 3>                        m_outputWritten = { false, false, false };

      std::array<uint8_t, kDxsoHighlightMaxConstRegs>                m_isDef;
      std::array<std::array<float, 4>, kDxsoHighlightMaxConstRegs>   m_defValues = {};
    };

  }

  DxsoHighlightResult analyzeDxsoHighlightTints(
    const uint32_t*             tokens,
    const size_t                tokenCount,
    const DxsoHighlightInputs&  inputs) {
    DxsoHighlightResult result;
    result.failure = DxsoHighlightFailure::NotPixelShader;
    if (tokens == nullptr || tokenCount < 2) {
      return result;
    }
    const uint32_t headerToken = tokens[0];
    if ((headerToken & 0xffff0000u) != 0xffff0000u) {
      return result;
    }
    const uint32_t majorVersion = (headerToken >> 8) & 0xffu;
    const uint32_t minorVersion = headerToken & 0xffu;
    // ps_1_x has no colour output register and stage-style texture ops; not analysed.
    if (majorVersion < 2) {
      return result;
    }

    const DxsoProgramInfo programInfo(DxsoProgramTypes::PixelShader, minorVersion, majorVersion);
    const auto analyzer = std::make_unique<HighlightAnalyzer>(inputs, programInfo);
    analyzer->run(tokens, tokenCount);
    analyzer->finish(result);
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
