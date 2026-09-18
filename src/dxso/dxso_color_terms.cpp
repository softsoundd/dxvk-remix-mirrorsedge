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
#include "dxso_color_terms.h"

#include <cstring>
#include <memory>

#include "dxso_code.h"
#include "dxso_common.h"
#include "dxso_decoder.h"

namespace dxvk {

  namespace {

    constexpr uint32_t kMaxTemps          = 64;
    // v# registers use bits 0..15, ps_2_x t# registers bits 16..31
    constexpr uint32_t kMaxInputs         = 32;
    constexpr uint32_t kTexcoordInputBase = 16;
    constexpr uint32_t kMaxConstSources   = 32;
    constexpr uint32_t kMaxLiteralSources = 64;
    constexpr uint32_t kConstSourceBase   = kDxsoColorTermMaxSamplers;
    constexpr uint32_t kLiteralSourceBase = kConstSourceBase + kMaxConstSources;
    constexpr uint32_t kMaxSources        = kLiteralSourceBase + kMaxLiteralSources;

    using TermRow = std::array<DxsoColorTermSet, kMaxSources>;
    // one bit per source
    using TouchSet = std::array<uint32_t, (kMaxSources + 31) / 32>;
    // one bit per float constant register
    using ConstDepSet = std::array<uint32_t, (kDxsoColorTermMaxConstRegs + 31) / 32>;

    // What one register component holds: the lighting factors its value derives from, the
    // interpolants it derives from, every tracked source's colour terms it carries, every
    // source that fed it at all (through dot products and normalisation too), and every
    // non-literal constant register that fed it.
    struct Value {
      uint8_t  factor = 0;
      // a `def` literal: `basisLiteral` when it equals one of UE3's LightMapBasis components
      bool     literal = false;
      bool     basisLiteral = false;
      // derived from a texture sample (any sampler, tracked or not)
      bool     sampled = false;
      // DxsoCoordExpr bits: the arithmetic this value has been through
      uint8_t  coordExpr = 0;
      uint32_t interpDeps = 0;
      TermRow  terms = {};
      TouchSet touch = {};
      ConstDepSet constDeps = {};

      bool hasConstDeps() const {
        for (const uint32_t w : constDeps) {
          if (w != 0) {
            return true;
          }
        }
        return false;
      }

      void merge(const Value& other) {
        factor |= other.factor;
        literal |= other.literal;
        basisLiteral |= other.basisLiteral;
        sampled |= other.sampled;
        coordExpr |= other.coordExpr;
        interpDeps |= other.interpDeps;
        for (uint32_t i = 0; i < kMaxSources; i++) {
          terms[i] |= other.terms[i];
        }
        for (uint32_t w = 0; w < touch.size(); w++) {
          touch[w] |= other.touch[w];
        }
        mergeConstDeps(other.constDeps);
      }

      void mergeConstDeps(const ConstDepSet& other) {
        for (uint32_t w = 0; w < constDeps.size(); w++) {
          constDeps[w] |= other[w];
        }
      }

      void touchSource(const uint32_t source) {
        touch[source / 32u] |= 1u << (source % 32u);
      }

      void dependOnConst(const uint32_t reg) {
        if (reg < kDxsoColorTermMaxConstRegs) {
          constDeps[reg / 32u] |= 1u << (reg % 32u);
        }
      }
    };

    // BasePassPixelShader.usf's LightMapBasis rows: (0, -1/sqrt2, 1/sqrt2), (sqrt6/3, -1/sqrt6,
    // -1/sqrt6), (1/sqrt3, 1/sqrt3, 1/sqrt3). fxc keeps them as literals in every directional
    // compile and they appear in no other UE3 material math.
    bool isBasisLiteral(const uint32_t bits) {
      float f;
      static_assert(sizeof(f) == sizeof(bits), "float bit pattern");
      std::memcpy(&f, &bits, sizeof(f));
      if (f < 0.0f) {
        f = -f;
      }
      constexpr float kBasis[] = { 0.70710678f, 0.81649658f, 0.40824829f, 0.57735027f };
      for (const float b : kBasis) {
        const float d = f - b;
        if (d < 1.0e-4f && d > -1.0e-4f) {
          return true;
        }
      }
      return false;
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

    // Rows a matrix op reads starting at its second operand's register.
    uint32_t matrixRowCount(const DxsoOpcode op) {
      switch (op) {
      case DxsoOpcode::M4x4:
      case DxsoOpcode::M3x4:
        return 4;
      case DxsoOpcode::M4x3:
      case DxsoOpcode::M3x3:
        return 3;
      case DxsoOpcode::M3x2:
        return 2;
      default:
        return 1;
      }
    }

    // Ops whose result collapses their inputs across components (dot products, normalise,
    // matrix transforms) rather than mapping lane to lane.
    bool isReductionOp(const DxsoOpcode op) {
      switch (op) {
      case DxsoOpcode::Dp2Add:
      case DxsoOpcode::Dp3:
      case DxsoOpcode::Dp4:
      case DxsoOpcode::Nrm:
      case DxsoOpcode::Crs:
      case DxsoOpcode::M4x4:
      case DxsoOpcode::M4x3:
      case DxsoOpcode::M3x4:
      case DxsoOpcode::M3x3:
      case DxsoOpcode::M3x2:
      case DxsoOpcode::Bem:
      case DxsoOpcode::Dst:
      case DxsoOpcode::Lit:
        return true;
      default:
        return false;
      }
    }

    // Scalar ops read one component (the first swizzle lane) and replicate the result.
    bool isScalarOp(const DxsoOpcode op) {
      switch (op) {
      case DxsoOpcode::Pow:
      case DxsoOpcode::Rcp:
      case DxsoOpcode::Rsq:
      case DxsoOpcode::Exp:
      case DxsoOpcode::Log:
      case DxsoOpcode::ExpP:
      case DxsoOpcode::LogP:
      case DxsoOpcode::SinCos:
        return true;
      default:
        return false;
      }
    }

    // Register components an operand read consumes for a reduction op, as a 4-bit mask.
    uint8_t reductionReadComponents(const DxsoOpcode op, const uint32_t srcIndex, const DxsoRegister& src) {
      uint32_t laneCount = 4;
      switch (op) {
      case DxsoOpcode::Dp3:
      case DxsoOpcode::Nrm:
      case DxsoOpcode::Crs:
        laneCount = 3;
        break;
      case DxsoOpcode::Dp2Add:
        laneCount = srcIndex < 2 ? 2 : 1;
        break;
      default:
        break;
      }
      uint8_t comps = 0;
      for (uint32_t lane = 0; lane < laneCount; lane++) {
        comps |= uint8_t(1u << src.swizzle[lane]);
      }
      return comps;
    }

    uint32_t popCount4(const uint8_t mask) {
      return uint32_t((mask & 1u) + ((mask >> 1) & 1u) + ((mask >> 2) & 1u) + ((mask >> 3) & 1u));
    }

    bool isTempRegister(const DxsoRegisterType type) {
      return type == DxsoRegisterType::Temp || type == DxsoRegisterType::TempFloat16;
    }

    // Multiplies every term in `row` by a value carrying `factor`.
    void shiftRow(TermRow& row, const uint8_t factor) {
      if (factor == 0) {
        return;
      }
      for (DxsoColorTermSet& set : row) {
        if (set == 0) {
          continue;
        }
        DxsoColorTermSet shifted = 0;
        for (uint32_t s = 0; s < kDxsoColorTermSignatureCount; s++) {
          if ((set >> s) & 1u) {
            shifted |= DxsoColorTermSet(1u << (s | factor));
          }
        }
        set = shifted;
      }
    }

    void orRow(TermRow& dst, const TermRow& src) {
      for (uint32_t i = 0; i < kMaxSources; i++) {
        dst[i] |= src[i];
      }
    }

    class ColorTermAnalyzer {
    public:
      ColorTermAnalyzer(const DxsoColorTermInputs& inputs, const DxsoProgramInfo& programInfo)
        : m_inputs(inputs), m_programInfo(programInfo) {
        m_constSourceIndex.fill(-1);
        uint32_t next = 0;
        for (uint32_t reg = 0; reg < kDxsoColorTermMaxConstRegs && next < kMaxConstSources; reg++) {
          if (inputs.trackedConstRegs.test(reg)) {
            m_constSourceIndex[reg] = int32_t(kConstSourceBase + next++);
          }
        }
      }

      // Pass 1 learns which interpolants are texture coordinates and which are normalised;
      // pass 2 tracks the terms. Both walk the same token stream.
      void runPass(const uint32_t* tokens, const size_t tokenCount, const uint32_t pass) {
        m_pass = pass;
        m_flowDepth = 0;
        for (auto& lanes : m_temps) {
          lanes.fill(Value {});
        }
        m_isDef.fill(0);

        DxsoDecodeContext decoder(m_programInfo);
        DxsoCodeIter iter(tokens + 1);
        for (size_t guard = 0; guard < tokenCount && decoder.decodeInstruction(iter); guard++) {
          processInstruction(decoder.getInstructionContext());
        }
        if (pass == 1) {
          m_nonUvTexcoordInputs = m_texcoordInputs & ~m_uvOrigins;
        }
      }

      void finish(DxsoColorTermResult& result) const {
        result.analyzed = true;
        result.hasLightConstTerm = (m_outputFactor & DxsoColorTermFactor_LightConst) != 0;
        for (uint32_t s = 0; s < kDxsoColorTermMaxSamplers; s++) {
          result.samplerTerms[s] = m_colorTerms[s];
          result.samplerReach[s] = m_reach[s];
        }
        for (uint32_t reg = 0; reg < kDxsoColorTermMaxConstRegs; reg++) {
          const int32_t idx = m_constSourceIndex[reg];
          result.constTerms[reg] = idx >= 0 ? m_colorTerms[size_t(idx)] : 0;
          result.constReach[reg] = idx >= 0 ? m_reach[size_t(idx)] : 0;
        }
        result.literals.reserve(m_literalBits.size());
        for (uint32_t i = 0; i < m_literalBits.size(); i++) {
          result.literals.push_back({ m_literalBits[i], m_colorTerms[kLiteralSourceBase + i] });
        }
        for (uint32_t s = 0; s < kDxsoColorTermMaxSamplers; s++) {
          for (uint32_t reg = 0; reg < kDxsoColorTermMaxConstRegs; reg++) {
            if ((m_samplerCoordConstDeps[s][reg / 32u] >> (reg % 32u)) & 1u) {
              result.samplerCoordConstRegs[s].set(reg);
            }
          }
          result.samplerCoordExpr[s] = m_samplerCoordExpr[s];
          result.samplerSampleCount[s] = m_samplerSampleCount[s];
        }
      }

    private:
      uint32_t inputBit(const DxsoRegister& r) const {
        switch (r.id.type) {
        case DxsoRegisterType::Input:
          return r.id.num < kTexcoordInputBase ? (1u << r.id.num) : 0u;
        case DxsoRegisterType::Texture:
        case DxsoRegisterType::PixelTexcoord:
          return r.id.num < (kMaxInputs - kTexcoordInputBase) ? (1u << (kTexcoordInputBase + r.id.num)) : 0u;
        default:
          return 0u;
        }
      }

      int32_t literalSource(const uint32_t bits) {
        for (uint32_t i = 0; i < m_literalBits.size(); i++) {
          if (m_literalBits[i] == bits) {
            return int32_t(kLiteralSourceBase + i);
          }
        }
        if (m_literalBits.size() >= kMaxLiteralSources) {
          return -1;
        }
        m_literalBits.push_back(bits);
        return int32_t(kLiteralSourceBase + m_literalBits.size() - 1);
      }

      // One component of an operand. `readComps` is every component the whole read consumes,
      // which decides whether an interpolant read is a vector (vertex lightmap) or a scalar (fog).
      Value readComponent(const DxsoRegister& r, const uint32_t comp, const uint8_t readComps) {
        Value v;
        switch (r.id.type) {
        case DxsoRegisterType::Temp:
        case DxsoRegisterType::TempFloat16:
          if (r.id.num < kMaxTemps) {
            v = m_temps[r.id.num][comp & 3u];
          }
          break;
        case DxsoRegisterType::Input:
        case DxsoRegisterType::Texture:
        case DxsoRegisterType::PixelTexcoord: {
          const uint32_t bit = inputBit(r);
          v.interpDeps = bit;
          if (m_pass == 2 && bit != 0 &&
              (m_nonUvTexcoordInputs & bit) != 0 &&
              (m_normalizedInputs & bit) == 0 &&
              popCount4(readComps) >= 2) {
            v.factor |= DxsoColorTermFactor_Interpolant;
          }
          break;
        }
        case DxsoRegisterType::Const: {
          if (r.hasRelative || r.id.num >= kDxsoColorTermMaxConstRegs) {
            break;
          }
          const uint32_t reg = r.id.num;
          if (m_isDef[reg]) {
            const uint32_t bits = m_defBits[reg][comp & 3u];
            v.literal = true;
            v.basisLiteral = isBasisLiteral(bits);
            if (m_pass == 2 && m_inputs.trackLiterals) {
              const int32_t idx = literalSource(bits);
              if (idx >= 0) {
                v.terms[size_t(idx)] |= 1u;
                v.touchSource(uint32_t(idx));
              }
            }
            break;
          }
          if (m_inputs.lightingConstRegs.test(reg)) {
            v.factor |= DxsoColorTermFactor_LightConst;
          }
          if (m_constSourceIndex[reg] >= 0) {
            v.terms[size_t(m_constSourceIndex[reg])] |= 1u;
            v.touchSource(uint32_t(m_constSourceIndex[reg]));
          }
          v.dependOnConst(reg);
          break;
        }
        default:
          break;
        }
        return v;
      }

      Value readUnion(const DxsoRegister& r, const uint8_t comps) {
        Value v;
        for (uint32_t comp = 0; comp < 4; comp++) {
          if ((comps >> comp) & 1u) {
            v.merge(readComponent(r, comp, comps));
          }
        }
        return v;
      }

      void writeLane(const uint32_t num, const uint32_t lane, const Value& value) {
        if (num >= kMaxTemps) {
          return;
        }
        if (m_flowDepth == 0) {
          m_temps[num][lane] = value;
        } else {
          // Inside control flow either branch may have produced the register: keep both.
          m_temps[num][lane].merge(value);
        }
      }

      void writeColorOutput(const Value& value) {
        m_outputFactor |= value.factor;
        orRow(m_colorTerms, value.terms);
        for (uint32_t i = 0; i < kMaxSources; i++) {
          if (value.terms[i] != 0) {
            m_reach[i] |= DxsoColorTermReach_Color;
          }
        }
      }

      // Opacity and coordinate reach count any influence, colour term or not: a normal map that
      // only steers a reflection lookup still exists in every compile.
      void markReach(const Value& value, const uint8_t reach) {
        for (uint32_t i = 0; i < kMaxSources; i++) {
          if ((value.touch[i / 32u] >> (i % 32u)) & 1u) {
            m_reach[i] |= reach;
          }
        }
      }

      void writeResult(const DxsoRegister& dst, const uint32_t lane, const Value& value) {
        if (isTempRegister(dst.id.type)) {
          writeLane(dst.id.num, lane, value);
        } else if (dst.id.type == DxsoRegisterType::ColorOut && m_pass == 2) {
          // only the colour lanes count as colour; the alpha lane is opacity
          if (lane < 3) {
            writeColorOutput(value);
          } else {
            markReach(value, DxsoColorTermReach_Opacity);
          }
        }
      }

      void processInstruction(const DxsoInstructionContext& ctx) {
        const DxsoOpcode op = ctx.instruction.opcode;
        switch (op) {
        case DxsoOpcode::Def:
          if (ctx.dst.id.type == DxsoRegisterType::Const && ctx.dst.id.num < kDxsoColorTermMaxConstRegs) {
            m_isDef[ctx.dst.id.num] = 1;
            for (uint32_t c = 0; c < 4; c++) {
              m_defBits[ctx.dst.id.num][c] = ctx.def.uint32[c];
            }
          }
          return;
        case DxsoOpcode::Dcl: {
          if (ctx.dst.id.type == DxsoRegisterType::Sampler) {
            if (ctx.dst.id.num < kDxsoColorTermMaxSamplers) {
              m_samplerCoordLanes[ctx.dst.id.num] = ctx.dcl.textureType == DxsoTextureType::Texture2D ? 2u : 3u;
            }
            return;
          }
          const uint32_t bit = inputBit(ctx.dst);
          if (bit != 0) {
            const bool texcoord =
              ctx.dst.id.type != DxsoRegisterType::Input ||
              ctx.dcl.semantic.usage == DxsoUsage::Texcoord;
            if (texcoord) {
              m_texcoordInputs |= bit;
            }
          }
          return;
        }
        case DxsoOpcode::If:
        case DxsoOpcode::Ifc:
        case DxsoOpcode::Rep:
        case DxsoOpcode::Loop:
          m_flowDepth++;
          return;
        case DxsoOpcode::EndIf:
        case DxsoOpcode::EndRep:
        case DxsoOpcode::EndLoop:
          if (m_flowDepth > 0) {
            m_flowDepth--;
          }
          return;
        case DxsoOpcode::Tex:
        case DxsoOpcode::TexLdl:
        case DxsoOpcode::TexLdd:
          processSample(ctx);
          return;
        case DxsoOpcode::TexKill:
          // the tested register is encoded as the destination
          if (m_pass == 2) {
            markReach(readUnion(ctx.dst, 0xFu), DxsoColorTermReach_Opacity);
          }
          return;
        case DxsoOpcode::DefI:
        case DxsoOpcode::DefB:
        case DxsoOpcode::Nop:
        case DxsoOpcode::Comment:
        case DxsoOpcode::End:
        case DxsoOpcode::Phase:
        case DxsoOpcode::Else:
        case DxsoOpcode::Break:
        case DxsoOpcode::BreakC:
        case DxsoOpcode::BreakP:
        case DxsoOpcode::Call:
        case DxsoOpcode::CallNz:
        case DxsoOpcode::Ret:
        case DxsoOpcode::Label:
        case DxsoOpcode::SetP:
        case DxsoOpcode::TexCoord:
        case DxsoOpcode::TexBem:
        case DxsoOpcode::TexBemL:
        case DxsoOpcode::TexReg2Ar:
        case DxsoOpcode::TexReg2Gb:
        case DxsoOpcode::TexM3x2Pad:
        case DxsoOpcode::TexM3x2Tex:
        case DxsoOpcode::TexM3x3Pad:
        case DxsoOpcode::TexM3x3Tex:
        case DxsoOpcode::TexM3x3Spec:
        case DxsoOpcode::TexM3x3VSpec:
        case DxsoOpcode::TexReg2Rgb:
        case DxsoOpcode::TexDp3Tex:
        case DxsoOpcode::TexM3x2Depth:
        case DxsoOpcode::TexDp3:
        case DxsoOpcode::TexM3x3:
        case DxsoOpcode::TexDepth:
          // no colour effect, or ps_1_x texture-stage opcodes that carry the coordinate in
          // the destination and are not analysed
          return;
        default:
          processArithmetic(ctx);
          return;
        }
      }

      void processSample(const DxsoInstructionContext& ctx) {
        // ps_2_0+: dst temp, src0 coordinate, src1 sampler.
        const DxsoRegister& coord = ctx.src[0];
        const uint32_t sampler = ctx.src[1].id.num;

        // Only the lanes the sampler consumes: the compiler leaves earlier values in the rest.
        uint32_t coordLaneCount = 4;
        if (sampler < kDxsoColorTermMaxSamplers && m_samplerCoordLanes[sampler] != 0) {
          coordLaneCount = m_samplerCoordLanes[sampler];
        }
        if (ctx.instruction.specificData.texld == DxsoTexLdMode::Project) {
          coordLaneCount = 4;
        }
        uint8_t coordComps = 0;
        for (uint32_t lane = 0; lane < coordLaneCount; lane++) {
          coordComps |= uint8_t(1u << coord.swizzle[lane]);
        }
        const Value coordValue = readUnion(coord, coordComps);
        if (m_pass == 1) {
          m_uvOrigins |= coordValue.interpDeps;
        } else {
          markReach(coordValue, DxsoColorTermReach_Coordinate);
          if (sampler < kDxsoColorTermMaxSamplers) {
            for (uint32_t w = 0; w < coordValue.constDeps.size(); w++) {
              m_samplerCoordConstDeps[sampler][w] |= coordValue.constDeps[w];
            }
            m_samplerCoordExpr[sampler] |= coordValue.coordExpr;
            if (m_samplerSampleCount[sampler] < 0xffffu) {
              m_samplerSampleCount[sampler]++;
            }
          }
        }

        // A sample is a fresh value: nothing about its coordinate is a colour factor. Its
        // constant dependencies and coordinate arithmetic do carry over: a dependent read's
        // result moves with whatever moved its coordinate.
        Value v;
        v.constDeps = coordValue.constDeps;
        v.coordExpr = coordValue.coordExpr;
        v.sampled = true;
        if (sampler < kDxsoColorTermMaxSamplers) {
          if ((m_inputs.lightmapSamplerMask >> sampler) & 1u) {
            v.factor |= DxsoColorTermFactor_Lightmap;
          }
          if ((m_inputs.trackedSamplerMask >> sampler) & 1u) {
            v.terms[sampler] |= 1u;
            v.touchSource(sampler);
          }
        }
        for (uint32_t lane = 0; lane < 4; lane++) {
          if (ctx.dst.mask[lane]) {
            writeResult(ctx.dst, lane, v);
          }
        }
      }

      // Combines already-read operands according to the opcode's dataflow.
      Value combine(const DxsoOpcode op, std::array<Value, 3>& src, const uint32_t srcCount, const bool normalizesInterpolant) {
        Value result;
        for (uint32_t i = 0; i < srcCount && i < 3; i++) {
          result.factor |= src[i].factor;
          result.interpDeps |= src[i].interpDeps;
          result.sampled |= src[i].sampled;
          result.coordExpr |= src[i].coordExpr;
          for (uint32_t w = 0; w < result.touch.size(); w++) {
            result.touch[w] |= src[i].touch[w];
          }
          result.mergeConstDeps(src[i].constDeps);
        }
        if (normalizesInterpolant) {
          result.factor |= DxsoColorTermFactor_View;
        }

        // Coordinate arithmetic. An additive term that carries no interpolant is an offset;
        // one that is neither a literal nor a plain constant is one the UV resolver cannot
        // express (a sampled or derived value).
        auto noteOffsetTerm = [&](const Value& term) {
          if (term.interpDeps != 0) {
            return;
          }
          result.coordExpr |= DxsoCoordExpr_Offset;
          const bool known = term.literal || (!term.sampled && term.hasConstDeps());
          if (!known) {
            result.coordExpr |= DxsoCoordExpr_UnknownOffset;
          }
        };
        switch (op) {
        case DxsoOpcode::Mov:
          break;
        case DxsoOpcode::Add:
        case DxsoOpcode::Sub:
          result.coordExpr |= DxsoCoordExpr_Arith;
          if (srcCount >= 2 && (src[0].interpDeps != 0) != (src[1].interpDeps != 0)) {
            noteOffsetTerm(src[0].interpDeps != 0 ? src[1] : src[0]);
          }
          break;
        case DxsoOpcode::Mad:
        case DxsoOpcode::Dp2Add:
          result.coordExpr |= DxsoCoordExpr_Arith;
          if (srcCount >= 3) {
            noteOffsetTerm(src[2]);
          }
          break;
        case DxsoOpcode::Frc:
        case DxsoOpcode::SinCos:
          result.coordExpr |= DxsoCoordExpr_Arith | DxsoCoordExpr_Wrap;
          break;
        default:
          result.coordExpr |= DxsoCoordExpr_Arith;
          break;
        }

        switch (op) {
        case DxsoOpcode::Mov:
        case DxsoOpcode::Abs:
          result.literal = src[0].literal;
          result.basisLiteral = src[0].basisLiteral;
          break;
        case DxsoOpcode::Mul:
        case DxsoOpcode::Mad:
        case DxsoOpcode::Dp2Add:
        case DxsoOpcode::Dp3:
        case DxsoOpcode::Dp4: {
          // mul(TangentReflectionVector, LightMapBasis): a view-derived vector against the
          // basis literals is the specular transfer; the diffuse transfer dots the normal
          bool viewOperand = false, basisOperand = false;
          for (uint32_t i = 0; i < 2; i++) {
            viewOperand |= (src[i].factor & DxsoColorTermFactor_View) != 0 && !src[i].basisLiteral;
            basisOperand |= src[i].basisLiteral;
          }
          if (viewOperand && basisOperand) {
            result.factor |= DxsoColorTermFactor_SpecularTransfer;
          }
          break;
        }
        default:
          break;
        }

        switch (op) {
        case DxsoOpcode::Mul:
          shiftRow(src[0].terms, src[1].factor);
          shiftRow(src[1].terms, src[0].factor);
          orRow(result.terms, src[0].terms);
          orRow(result.terms, src[1].terms);
          break;
        case DxsoOpcode::Mad:
          shiftRow(src[0].terms, src[1].factor);
          shiftRow(src[1].terms, src[0].factor);
          orRow(result.terms, src[0].terms);
          orRow(result.terms, src[1].terms);
          orRow(result.terms, src[2].terms);
          break;
        case DxsoOpcode::Lrp:
        case DxsoOpcode::Cmp:
        case DxsoOpcode::Cnd:
          // the first operand selects or weights the other two; each side is scaled by it
          shiftRow(src[0].terms, uint8_t(src[1].factor | src[2].factor));
          shiftRow(src[1].terms, src[0].factor);
          shiftRow(src[2].terms, src[0].factor);
          orRow(result.terms, src[0].terms);
          orRow(result.terms, src[1].terms);
          orRow(result.terms, src[2].terms);
          break;
        case DxsoOpcode::Min:
        case DxsoOpcode::Max:
        case DxsoOpcode::Slt:
        case DxsoOpcode::Sge:
          // clamps and comparisons gate each operand by the other
          shiftRow(src[0].terms, src[1].factor);
          shiftRow(src[1].terms, src[0].factor);
          orRow(result.terms, src[0].terms);
          orRow(result.terms, src[1].terms);
          break;
        case DxsoOpcode::Pow:
          // the exponent scales the base's contribution; the base keeps its own terms
          shiftRow(src[1].terms, src[0].factor);
          orRow(result.terms, src[0].terms);
          orRow(result.terms, src[1].terms);
          break;
        case DxsoOpcode::Dp2Add:
        case DxsoOpcode::Dp3:
        case DxsoOpcode::Dp4: {
          // A dot product against plain literal weights is a weighted channel sum (UE3's
          // Desaturation), so the colour survives; against the basis, another vector or a
          // normal it is direction or coefficient data and stops being any source's colour.
          const bool weights =
            (src[0].literal && !src[0].basisLiteral) || (src[1].literal && !src[1].basisLiteral);
          if (weights) {
            orRow(result.terms, src[0].terms);
            orRow(result.terms, src[1].terms);
            if (op == DxsoOpcode::Dp2Add) {
              orRow(result.terms, src[2].terms);
            }
          }
          break;
        }
        case DxsoOpcode::Nrm:
        case DxsoOpcode::Crs:
        case DxsoOpcode::M4x4:
        case DxsoOpcode::M4x3:
        case DxsoOpcode::M3x4:
        case DxsoOpcode::M3x3:
        case DxsoOpcode::M3x2:
        case DxsoOpcode::Bem:
        case DxsoOpcode::Dst:
        case DxsoOpcode::Lit:
          // normalisation and matrix transforms turn a colour into direction data
          break;
        default:
          // additive and unary ops pass every term through unchanged; unknown opcodes too, so
          // a missed opcode keeps a term alive rather than dropping the source from identity
          for (uint32_t i = 0; i < srcCount && i < 3; i++) {
            orRow(result.terms, src[i].terms);
          }
          break;
        }
        return result;
      }

      void processArithmetic(const DxsoInstructionContext& ctx) {
        const DxsoOpcode op = ctx.instruction.opcode;
        const uint32_t srcCount = sourceOperandCount(op);
        if (srcCount == 0) {
          return;
        }

        // `nrm v`, or the `dp3 r, v, v; rsq; mul` sequence older compilers emit for normalize()
        const bool selfDot = op == DxsoOpcode::Dp3 && ctx.src[0].id == ctx.src[1].id;
        const bool normalizes = op == DxsoOpcode::Nrm || selfDot;

        if (isReductionOp(op)) {
          std::array<Value, 3> src;
          for (uint32_t i = 0; i < srcCount && i < 3; i++) {
            src[i] = readUnion(ctx.src[i], reductionReadComponents(op, i, ctx.src[i]));
          }
          bool normalizesInterpolant = false;
          if (normalizes) {
            if (m_pass == 1) {
              m_normalizedInputs |= src[0].interpDeps;
            } else {
              normalizesInterpolant = (src[0].interpDeps & m_nonUvTexcoordInputs) != 0;
            }
          }
          Value result = combine(op, src, srcCount, normalizesInterpolant);
          if (normalizes) {
            // A normalised vector is a direction, not the interpolant it came from: a
            // reflection vector built from it that ends up as a cube-map coordinate must not
            // turn the camera vector into a texture coordinate.
            result.interpDeps = 0;
          }
          // A matrix transform reads the rows after the one the operand names.
          const uint32_t matrixRows = matrixRowCount(op);
          if (matrixRows > 1 && srcCount >= 2 &&
              ctx.src[1].id.type == DxsoRegisterType::Const && !ctx.src[1].hasRelative) {
            for (uint32_t row = 1; row < matrixRows; row++) {
              const uint32_t reg = ctx.src[1].id.num + row;
              if (reg < kDxsoColorTermMaxConstRegs && !m_isDef[reg]) {
                result.dependOnConst(reg);
              }
            }
          }
          for (uint32_t lane = 0; lane < 4; lane++) {
            if (ctx.dst.mask[lane]) {
              writeResult(ctx.dst, lane, result);
            }
          }
          return;
        }

        // Component-wise: lane L of the result reads lane swizzle[L] of every operand (or the
        // first swizzle lane for scalar ops), so a scalar the compiler packed into a spare lane
        // of a live register never inherits the other lanes' terms.
        const bool scalar = isScalarOp(op);
        std::array<uint8_t, 3> readComps = {};
        for (uint32_t i = 0; i < srcCount && i < 3; i++) {
          for (uint32_t lane = 0; lane < 4; lane++) {
            if (ctx.dst.mask[lane]) {
              readComps[i] |= uint8_t(1u << ctx.src[i].swizzle[scalar ? 0 : lane]);
            }
          }
        }
        for (uint32_t lane = 0; lane < 4; lane++) {
          if (!ctx.dst.mask[lane]) {
            continue;
          }
          std::array<Value, 3> src;
          for (uint32_t i = 0; i < srcCount && i < 3; i++) {
            src[i] = readComponent(ctx.src[i], ctx.src[i].swizzle[scalar ? 0 : lane], readComps[i]);
          }
          writeResult(ctx.dst, lane, combine(op, src, srcCount, false));
        }
      }

      const DxsoColorTermInputs& m_inputs;
      DxsoProgramInfo            m_programInfo;
      uint32_t                   m_pass = 1;
      uint32_t                   m_flowDepth = 0;

      std::array<std::array<Value, 4>, kMaxTemps>                     m_temps = {};
      std::array<uint8_t, kDxsoColorTermMaxConstRegs>                 m_isDef = {};
      std::array<std::array<uint32_t, 4>, kDxsoColorTermMaxConstRegs> m_defBits = {};
      std::array<int32_t, kDxsoColorTermMaxConstRegs>                 m_constSourceIndex = {};
      std::array<uint8_t, kDxsoColorTermMaxSamplers>                  m_samplerCoordLanes = {};
      std::vector<uint32_t>                                           m_literalBits;

      // pass 1 products
      uint32_t m_texcoordInputs = 0;
      uint32_t m_uvOrigins = 0;
      uint32_t m_normalizedInputs = 0;
      uint32_t m_nonUvTexcoordInputs = 0;

      // pass 2 products
      TermRow                            m_colorTerms = {};
      std::array<uint8_t, kMaxSources>   m_reach = {};
      uint8_t                            m_outputFactor = 0;
      std::array<ConstDepSet, kDxsoColorTermMaxSamplers> m_samplerCoordConstDeps = {};
      std::array<uint8_t, kDxsoColorTermMaxSamplers>     m_samplerCoordExpr = {};
      std::array<uint16_t, kDxsoColorTermMaxSamplers>    m_samplerSampleCount = {};
    };

  }

  DxsoColorTermResult analyzeDxsoColorTerms(
    const uint32_t*             tokens,
    const size_t                tokenCount,
    const DxsoColorTermInputs&  inputs) {
    DxsoColorTermResult result;
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
    // the per-lane temp state runs to over 100 KB; keep it off the caller's stack
    const auto analyzer = std::make_unique<ColorTermAnalyzer>(inputs, programInfo);
    analyzer->runPass(tokens, tokenCount, 1);
    analyzer->runPass(tokens, tokenCount, 2);
    analyzer->finish(result);
    return result;
  }

  bool isDxsoColorTermSetSpecularOnly(const DxsoColorTermSet set) {
    if (set == 0) {
      return false;
    }
    constexpr uint8_t kLit = DxsoColorTermFactor_Lightmap | DxsoColorTermFactor_LightConst | DxsoColorTermFactor_Interpolant;
    for (uint32_t s = 0; s < kDxsoColorTermSignatureCount; s++) {
      if (((set >> s) & 1u) == 0) {
        continue;
      }
      if ((s & DxsoColorTermFactor_SpecularTransfer) == 0 || (s & kLit) == 0) {
        return false;
      }
    }
    return true;
  }

  bool isDxsoColorTermSetTransferOnly(const DxsoColorTermSet set) {
    if (set == 0) {
      return false;
    }
    constexpr uint8_t kLightmapLike = DxsoColorTermFactor_Lightmap | DxsoColorTermFactor_Interpolant;
    for (uint32_t s = 0; s < kDxsoColorTermSignatureCount; s++) {
      if (((set >> s) & 1u) == 0) {
        continue;
      }
      if ((s & kLightmapLike) == 0 || (s & DxsoColorTermFactor_LightConst) != 0) {
        return false;
      }
    }
    return true;
  }

  bool dxsoColorTermSetHasUnlitTerm(const DxsoColorTermSet set) {
    return (set & 1u) != 0;
  }

  DxsoColorTermRole classifyDxsoColorTermSource(const DxsoColorTermSet set, const uint8_t reach, const bool shaderHasLightConstTerm) {
    const bool lightingOnly =
      set != 0 &&
      (isDxsoColorTermSetSpecularOnly(set) || (shaderHasLightConstTerm && isDxsoColorTermSetTransferOnly(set)));
    if (set != 0 && !lightingOnly) {
      return DxsoColorTermRole::Color;
    }
    // No policy-stable colour path. The simple-lightmap compile of the same material sees no
    // colour path at all here, so both compiles land on the same reach-based role.
    if (reach & DxsoColorTermReach_Opacity) {
      return DxsoColorTermRole::Opacity;
    }
    if (reach & DxsoColorTermReach_Coordinate) {
      return DxsoColorTermRole::Coordinate;
    }
    return lightingOnly ? DxsoColorTermRole::LightingOnly : DxsoColorTermRole::Unused;
  }

}
