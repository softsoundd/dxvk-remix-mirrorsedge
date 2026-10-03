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
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <unordered_map>
#include <vector>

#include "dxso_decoder.h"
#include "dxso_highlight_tints.h"

namespace dxvk {

  /**
   * \brief Symbolic evaluation of a pixel shader's colour output
   *
   * Every register component holds a polynomial over the shader's inputs, with texture samples,
   * material parameters and interpolants as symbols. A value that involves no material parameter
   * is folded into a single opaque atom, as is the result of anything non-polynomial, which keeps
   * the expressions small. The highlight tint and material fade proofs both read what oC0 ends up
   * holding.
   */
  namespace dxso_symbolic {

    // Runner Vision networks fit well inside these budgets; the shaders that exceed them are rim
    // and fresnel networks with many scalar-weighted sums.
    constexpr uint32_t kMaxTermSymbols = 16;
    constexpr size_t   kMaxPolyTerms = 8192;
    constexpr size_t   kMaxSymbols = 0xFFFF;
    constexpr double   kZeroCoefficient = 1.0e-9;
    // Coefficients are products of the shader's literals, so matching ones differ only by float
    // rounding.
    constexpr double   kMatchTolerance = 1.0e-4;

    enum class SymbolKind : uint8_t {
      MaterialScalar,  // the x component of a UniformScalar_* register
      MaterialVector,  // a component of a UniformVector_* register
      Constant,        // any other constant register component
      Interpolant,     // an input register component (texcoords, colours, vPos, vFace)
      VertexColor,     // a component of the input register the caller names as the vertex colour
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
      // a component of a float constant register the shader does not define, so a draw can read it
      bool       liveConstant = false;
    };

    // One monomial: a coefficient times a sorted multiset of symbols.
    struct Term {
      double   coef = 0.0;
      uint8_t  count = 0;
      std::array<uint16_t, kMaxTermSymbols> syms = {};
    };

    // Terms sorted by monomial, each monomial once, no zero coefficients.
    using Poly = std::vector<Term>;

    int         compareMonomial(const Term& a, const Term& b);
    bool        monomialLess(const Term& a, const Term& b);
    bool        approxEqual(double a, double b);
    Poly        polyConstant(double value);
    Poly        polySymbol(uint16_t symbol);
    const Term* findMonomial(const Poly& p, const Term& key);
    // The symbols of `dividend` left once those of `divisor` are taken out, when it divides.
    bool        divideMonomial(const Term& dividend, const Term& divisor, Term& remainder);
    bool        containsSymbol(const Term& t, uint16_t symbol);
    uint32_t    symbolOccurrences(const Term& t, uint16_t symbol);
    // a + bScale * b, without the term budget the evaluation enforces.
    Poly        addPolys(const Poly& a, const Poly& b, double bScale = 1.0);

    // Program info of ps_2_0+ bytecode, the only kind the evaluation models.
    std::optional<DxsoProgramInfo> pixelShaderProgramInfo(const uint32_t* tokens, size_t tokenCount);

    struct EvaluatorOptions {
      // A ps_3_0 input register (v#) whose components become VertexColor symbols rather than
      // interpolants, or -1.
      int32_t vertexColorInputRegister = -1;
      // Treat min and max against a literal bound outside [0, 1] as the identity, as _sat already is:
      // such a clamp leaves an opacity or a colour within that range unchanged.
      bool rangeClampsAreIdentity = false;
    };

    class Evaluator {
    public:
      Evaluator(const DxsoHighlightInputs& inputs, const EvaluatorOptions& options, const DxsoProgramInfo& programInfo);

      void run(const uint32_t* tokens, size_t tokenCount);

      DxsoHighlightFailure failure() const {
        return m_failure;
      }

      // oC0 lanes 0-2 are the colour, lane 3 the alpha.
      bool outputWritten(const uint32_t lane) const {
        return m_outputWritten[lane];
      }

      const Poly& output(const uint32_t lane) const {
        return m_output[lane];
      }

      const Symbol& symbol(const uint16_t id) const {
        return m_symbols[id];
      }

      size_t symbolCount() const {
        return m_symbols.size();
      }

      // A material texture sample or material vector, or derived from one.
      bool isColorBearing(const Term& t) const;
      bool hasLighting(const Term& t) const;
      // Only lighting, interpolants and other colourless, parameter-free factors.
      bool isFactorOnly(const Term& t) const;
      // Whether an atom was computed from `modulator` - a material parameter or vertex colour
      // symbol - directly or through the atoms it folds.
      bool atomDependsOn(uint16_t atom, uint16_t modulator) const;

    private:
      void fail(DxsoHighlightFailure failure);

      uint16_t makeSymbol(uint64_t key, const Symbol& proto);
      uint16_t makeAtom(uint64_t key, const Symbol& proto, std::initializer_list<const Poly*> inputs);
      void collectModulators(const Poly& p, std::vector<uint16_t>& modulators) const;
      void flagsOf(const Poly& p, bool& lighting, bool& color, bool& material) const;
      Poly opaque(uint32_t tag, std::initializer_list<const Poly*> inputs);

      Poly add(const Poly& a, const Poly& b, double bScale = 1.0);
      Poly mul(const Poly& a, const Poly& b);
      Poly affine(Poly p, double scale, double offset);
      Poly collapse(Poly value);

      Poly readConstant(uint32_t reg, uint32_t component);
      Poly applySourceModifier(DxsoRegModifier modifier, Poly value);
      Poly readSourceLane(const DxsoRegister& reg, uint32_t component);
      Poly readLane(const DxsoInstructionContext& ctx, uint32_t srcIndex, uint32_t lane);

      void processInstruction(const DxsoInstructionContext& ctx);
      void processSample(const DxsoInstructionContext& ctx);
      void processArithmetic(const DxsoInstructionContext& ctx);
      Poly currentValue(const DxsoRegister& dst, uint32_t lane) const;
      void writeResult(const DxsoInstructionContext& ctx, std::array<Poly, 4>& results);

      static constexpr uint32_t kMaxTemps = 64;

      const DxsoHighlightInputs& m_inputs;
      EvaluatorOptions           m_options;
      DxsoProgramInfo            m_programInfo;
      DxsoHighlightFailure       m_failure = DxsoHighlightFailure::None;

      std::vector<Symbol>                    m_symbols;
      std::unordered_map<uint64_t, uint16_t> m_symbolIndex;
      // Sorted modulator symbols per atom that folds any.
      std::unordered_map<uint16_t, std::vector<uint16_t>> m_atomModulators;
      uint32_t                               m_sampleCount = 0;

      std::array<std::array<Poly, 4>, kMaxTemps> m_temps;
      std::array<Poly, 4>                        m_output;
      std::array<bool, 4>                        m_outputWritten = { false, false, false, false };

      std::array<uint8_t, kDxsoHighlightMaxConstRegs>                m_isDef;
      std::array<std::array<float, 4>, kDxsoHighlightMaxConstRegs>   m_defValues = {};
    };

  }

}
