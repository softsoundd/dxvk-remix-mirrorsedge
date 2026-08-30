#include "dxso_analysis.h"

// NV-DXVK start: exact vertex capture
#include "../util/util_string.h"

#include <algorithm>
#include <limits>
#include <sstream>
// NV-DXVK end

namespace dxvk {

  // NV-DXVK start: exact vertex capture
  namespace {

    // Zero-length opcodes leave the decoder's destination register holding the previous
    // instruction's operand, and label/constant operands are not writes, so anything
    // walking destinations has to skip both.
    bool opcodeHasDestination(DxsoOpcode opcode) {
      switch (opcode) {
        case DxsoOpcode::Nop:
        case DxsoOpcode::If:
        case DxsoOpcode::Ifc:
        case DxsoOpcode::Else:
        case DxsoOpcode::EndIf:
        case DxsoOpcode::Loop:
        case DxsoOpcode::EndLoop:
        case DxsoOpcode::Rep:
        case DxsoOpcode::EndRep:
        case DxsoOpcode::Break:
        case DxsoOpcode::BreakC:
        case DxsoOpcode::BreakP:
        case DxsoOpcode::Ret:
        case DxsoOpcode::Call:
        case DxsoOpcode::CallNz:
        case DxsoOpcode::Label:
        case DxsoOpcode::Dcl:
        case DxsoOpcode::Def:
        case DxsoOpcode::DefI:
        case DxsoOpcode::DefB:
        case DxsoOpcode::Comment:
        case DxsoOpcode::Phase:
        case DxsoOpcode::End:
          return false;
        default:
          return true;
      }
    }

    // Note: partialPrecision/centroid are decoded from bits that source parameter tokens
    // spend on the swizzle, so they are only meaningful on a destination and are not
    // inspected here.
    bool isPlainSource(const DxsoRegister& reg) {
      return !reg.hasRelative
          && reg.modifier == DxsoRegModifier::None;
    }

    bool isIdentitySource(const DxsoRegister& reg) {
      return isPlainSource(reg) && reg.swizzle == IdentitySwizzle;
    }

    // A scalar source such as rA.x, which D3D9 encodes as that component replicated
    // across all four swizzle slots.
    bool isBroadcastSource(const DxsoRegister& reg, uint32_t& outComponent) {
      if (!isPlainSource(reg))
        return false;

      const uint32_t component = reg.swizzle[0];
      for (uint32_t i = 1; i < 4; i++) {
        if (reg.swizzle[i] != component)
          return false;
      }

      outComponent = component;
      return true;
    }

    bool isPlainDst(const DxsoRegister& reg) {
      return !reg.hasRelative
          && !reg.saturate
          && !reg.partialPrecision
          && reg.shift == 0;
    }

    // A register the capture code can read back: the input assembler value or a temp the
    // shader computed. Anything else is either write-only or not a per-vertex value.
    bool isReadableSourceRegister(const DxsoRegisterId& id) {
      return id.type == DxsoRegisterType::Temp
          || id.type == DxsoRegisterType::Input;
    }

    // fxc emits the operands of a scalar-by-matrix-row product in either order, so accept
    // both and report which side is which.
    bool splitBroadcastAndConst(
      const DxsoRegister&  a,
      const DxsoRegister&  b,
      const DxsoRegister** outValue,
      const DxsoRegister** outConst,
            uint32_t&      outComponent) {
      const DxsoRegister* orders[2][2] = { { &a, &b }, { &b, &a } };

      for (const auto& order : orders) {
        const DxsoRegister* value = order[0];
        const DxsoRegister* konst = order[1];

        uint32_t component = 0;
        if (isReadableSourceRegister(value->id)
         && isBroadcastSource(*value, component)
         && konst->id.type == DxsoRegisterType::Const
         && isIdentitySource(*konst)) {
          *outValue = value;
          *outConst = konst;
          outComponent = component;
          return true;
        }
      }

      return false;
    }

    bool splitIdentityAndConst(
      const DxsoRegister&  a,
      const DxsoRegister&  b,
      const DxsoRegister** outValue,
      const DxsoRegister** outConst) {
      const DxsoRegister* orders[2][2] = { { &a, &b }, { &b, &a } };

      for (const auto& order : orders) {
        const DxsoRegister* value = order[0];
        const DxsoRegister* konst = order[1];

        if (isReadableSourceRegister(value->id)
         && isIdentitySource(*value)
         && konst->id.type == DxsoRegisterType::Const
         && isIdentitySource(*konst)) {
          *outValue = value;
          *outConst = konst;
          return true;
        }
      }

      return false;
    }

    // Assembly-style names for the register files a vertex shader's position dataflow can
    // involve; anything else falls back to its numeric type so it remains identifiable.
    std::string registerName(const DxsoRegisterId& id) {
      const char* prefix = nullptr;
      switch (id.type) {
        case DxsoRegisterType::Temp:          prefix = "r"; break;
        case DxsoRegisterType::Input:         prefix = "v"; break;
        case DxsoRegisterType::Const:         prefix = "c"; break;
        case DxsoRegisterType::ConstInt:      prefix = "i"; break;
        case DxsoRegisterType::ConstBool:     prefix = "b"; break;
        case DxsoRegisterType::Addr:          prefix = "a"; break;
        case DxsoRegisterType::RasterizerOut: prefix = "rast"; break;
        case DxsoRegisterType::AttributeOut:  prefix = "oD"; break;
        case DxsoRegisterType::Output:        prefix = "o"; break;
        case DxsoRegisterType::Predicate:     prefix = "p"; break;
        default:
          return str::format("type", uint32_t(id.type), "_", id.num);
      }

      return str::format(prefix, id.num);
    }

    std::string describeSource(const DxsoRegister& reg) {
      static const char kComponents[4] = { 'x', 'y', 'z', 'w' };

      std::string result = registerName(reg.id);
      if (reg.swizzle != IdentitySwizzle) {
        result += '.';
        for (uint32_t i = 0; i < 4; i++)
          result += kComponents[reg.swizzle[i]];
      }
      if (reg.modifier != DxsoRegModifier::None)
        result += str::format("(mod", uint32_t(reg.modifier), ")");
      if (reg.hasRelative)
        result += "[rel]";

      return result;
    }

    std::string describeDestination(const DxsoRegister& reg) {
      static const char kComponents[4] = { 'x', 'y', 'z', 'w' };

      std::string result = registerName(reg.id);
      if (reg.mask != IdentityWriteMask) {
        result += '.';
        for (uint32_t i = 0; i < 4; i++) {
          if (reg.mask[i])
            result += kComponents[i];
        }
      }

      return result;
    }

    // The decoder keeps its operand array between instructions, so printing more sources
    // than an opcode takes would show stale registers from an earlier instruction.
    uint32_t sourceCountForOpcode(DxsoOpcode opcode) {
      switch (opcode) {
        case DxsoOpcode::Mov:
        case DxsoOpcode::Mova:
        case DxsoOpcode::Rcp:
        case DxsoOpcode::Rsq:
        case DxsoOpcode::Exp:
        case DxsoOpcode::ExpP:
        case DxsoOpcode::Log:
        case DxsoOpcode::LogP:
        case DxsoOpcode::Frc:
        case DxsoOpcode::Abs:
        case DxsoOpcode::Nrm:
        case DxsoOpcode::Sgn:
        case DxsoOpcode::Lit:
        case DxsoOpcode::SinCos:
          return 1;

        case DxsoOpcode::Mad:
        case DxsoOpcode::Lrp:
        case DxsoOpcode::Cmp:
        case DxsoOpcode::Cnd:
        case DxsoOpcode::Dp2Add:
          return 3;

        default:
          return 2;
      }
    }

    std::string describeInstruction(const DxsoInstructionContext& ctx) {
      std::stringstream stream;
      stream << ctx.instruction.opcode;

      std::string result = stream.str();
      if (!opcodeHasDestination(ctx.instruction.opcode))
        return result;

      result += " " + describeDestination(ctx.dst);

      const uint32_t sourceCount = sourceCountForOpcode(ctx.instruction.opcode);
      for (uint32_t i = 0; i < sourceCount; i++)
        result += ", " + describeSource(ctx.src[i]);

      return result;
    }

  }
  // NV-DXVK end

  // NV-DXVK start: exact vertex capture - programInfo added
  DxsoAnalyzer::DxsoAnalyzer(
    const DxsoProgramInfo&  programInfo,
          DxsoAnalysisInfo& analysis)
    : m_analysis(&analysis)
    , m_programInfo(programInfo) { }
  // NV-DXVK end

  void DxsoAnalyzer::processInstruction(
    const DxsoInstructionContext& ctx) {
    DxsoOpcode opcode = ctx.instruction.opcode;

    // Co-issued CNDs are issued before their parents,
    // except when the parent is a CND.
    if (opcode == DxsoOpcode::Cnd &&
        m_parentOpcode != DxsoOpcode::Cnd &&
        ctx.instruction.coissue) {
      m_analysis->coissues.push_back(ctx);
    }

    if (opcode == DxsoOpcode::TexKill)
      m_analysis->usesKill = true;

    if (opcode == DxsoOpcode::DsX
     || opcode == DxsoOpcode::DsY

     || opcode == DxsoOpcode::Tex
     || opcode == DxsoOpcode::TexCoord
     || opcode == DxsoOpcode::TexBem
     || opcode == DxsoOpcode::TexBemL
     || opcode == DxsoOpcode::TexReg2Ar
     || opcode == DxsoOpcode::TexReg2Gb
     || opcode == DxsoOpcode::TexM3x2Pad
     || opcode == DxsoOpcode::TexM3x2Tex
     || opcode == DxsoOpcode::TexM3x3Pad
     || opcode == DxsoOpcode::TexM3x3Tex
     || opcode == DxsoOpcode::TexM3x3Spec
     || opcode == DxsoOpcode::TexM3x3VSpec
     || opcode == DxsoOpcode::TexReg2Rgb
     || opcode == DxsoOpcode::TexDp3Tex
     || opcode == DxsoOpcode::TexM3x2Depth
     || opcode == DxsoOpcode::TexDp3
     || opcode == DxsoOpcode::TexM3x3
     //  Explicit LOD.
     //|| opcode == DxsoOpcode::TexLdd
     //|| opcode == DxsoOpcode::TexLdl
     || opcode == DxsoOpcode::TexDepth)
      m_analysis->usesDerivatives = true;

    // NV-DXVK start: exact vertex capture
    if (m_programInfo.type() == DxsoProgramTypes::VertexShader)
      this->recordInstruction(ctx);
    // NV-DXVK end

    m_parentOpcode = ctx.instruction.opcode;
  }

  // NV-DXVK start: exact vertex capture
  void DxsoAnalyzer::recordInstruction(
    const DxsoInstructionContext& ctx) {
    switch (ctx.instruction.opcode) {
      case DxsoOpcode::Call:
      case DxsoOpcode::CallNz:
      case DxsoOpcode::Label:
      case DxsoOpcode::Ret:
        m_usesSubroutines = true;
        break;

      case DxsoOpcode::If:
      case DxsoOpcode::Ifc:
      case DxsoOpcode::Loop:
      case DxsoOpcode::Rep:
        m_controlFlowDepth++;
        break;

      case DxsoOpcode::EndIf:
      case DxsoOpcode::EndLoop:
      case DxsoOpcode::EndRep:
        if (m_controlFlowDepth != 0)
          m_controlFlowDepth--;
        break;

      case DxsoOpcode::Dcl:
        if (ctx.dst.id.type == DxsoRegisterType::Output
         && ctx.dst.id.num < 32
         && ctx.dcl.semantic.usage == DxsoUsage::Position
         && ctx.dcl.semantic.usageIndex == 0)
          m_declaredPositionOutputs |= 1u << ctx.dst.id.num;
        break;

      default:
        break;
    }

    RecordedInstruction recorded;
    recorded.ctx = ctx;
    recorded.controlFlowDepth = m_controlFlowDepth;
    m_instructions.push_back(std::move(recorded));
  }

  bool DxsoAnalyzer::isPositionOutput(
    const DxsoBaseRegister& reg) const {
    if (reg.id.type == DxsoRegisterType::RasterizerOut)
      return reg.id.num == RasterOutPosition;

    if (reg.id.type == DxsoRegisterType::Output)
      return reg.id.num < 32 && (m_declaredPositionOutputs & (1u << reg.id.num)) != 0;

    return false;
  }

  DxsoAnalyzer::Definition DxsoAnalyzer::resolveDefinition(
    const DxsoRegisterId& reg,
          uint32_t        component,
          int32_t         beforeIdx) const {
    DxsoRegisterId currentReg = reg;
    int32_t searchFrom = beforeIdx;

    // Bounded so a pathological mov chain cannot spin.
    for (uint32_t hop = 0; hop < 16; hop++) {
      Definition found = -1;

      for (int32_t i = searchFrom - 1; i >= 0; i--) {
        const RecordedInstruction& recorded = m_instructions[size_t(i)];
        const DxsoInstructionContext& ctx = recorded.ctx;

        if (!opcodeHasDestination(ctx.instruction.opcode))
          continue;
        if (ctx.dst.id != currentReg || !ctx.dst.mask[component])
          continue;

        // A conditional or predicated write means the value reaching the rasterizer is
        // not determined by this instruction alone.
        if (recorded.controlFlowDepth != 0 || ctx.instruction.predicated)
          return -1;

        found = i;
        break;
      }

      if (found < 0)
        return -1;

      const DxsoInstructionContext& ctx = m_instructions[size_t(found)].ctx;
      if (ctx.instruction.opcode != DxsoOpcode::Mov
       || !isPlainDst(ctx.dst)
       || !isPlainSource(ctx.src[0]))
        return found;

      // Only follow a straight copy. A swizzling mov would mean position carries a
      // permutation of the transform's result, which the matchers cannot see because they
      // compare instructions rather than components.
      if (ctx.src[0].swizzle[component] != component)
        return -1;

      currentReg = ctx.src[0].id;
      searchFrom = found;
    }

    return -1;
  }

  bool DxsoAnalyzer::findPositionDefinitions(
          PositionDefinitions& outDefs) const {
    DxsoRegisterId positionReg = { DxsoRegisterType::RasterizerOut, RasterOutPosition };
    bool havePositionReg = false;

    for (int32_t i = int32_t(m_instructions.size()) - 1; i >= 0; i--) {
      const DxsoInstructionContext& ctx = m_instructions[size_t(i)].ctx;

      if (!opcodeHasDestination(ctx.instruction.opcode) || !this->isPositionOutput(ctx.dst))
        continue;

      if (!havePositionReg) {
        positionReg = ctx.dst.id;
        havePositionReg = true;
      } else if (ctx.dst.id != positionReg) {
        // Two registers claim to be position; there is no single answer.
        return false;
      }
    }

    if (!havePositionReg)
      return false;

    for (uint32_t component = 0; component < 4; component++) {
      outDefs[component] = this->resolveDefinition(positionReg, component, int32_t(m_instructions.size()));
      if (outDefs[component] < 0)
        return false;
    }

    return true;
  }

  // dp4 dst.c, rA, c[K+c] for each of the four components, in any order and at any
  // positions in the shader - fxc interleaves them with unrelated work.
  bool DxsoAnalyzer::matchDp4Group(
    const PositionDefinitions& defs,
          MatrixTransform&     outTransform) const {
    MatrixTransform transform;
    transform.firstIdx = std::numeric_limits<int32_t>::max();

    for (uint32_t component = 0; component < 4; component++) {
      const DxsoInstructionContext& ctx = m_instructions[size_t(defs[component])].ctx;

      if (ctx.instruction.opcode != DxsoOpcode::Dp4)
        return false;
      if (!isPlainDst(ctx.dst) || ctx.dst.mask.popCount() != 1)
        return false;

      const DxsoRegister* value = nullptr;
      const DxsoRegister* konst = nullptr;
      if (!splitIdentityAndConst(ctx.src[0], ctx.src[1], &value, &konst))
        return false;

      if (component == 0) {
        transform.source = value->id;
        transform.constBase = konst->id.num;
      } else if (value->id != transform.source
              || konst->id.num != transform.constBase + component) {
        return false;
      }

      transform.firstIdx = std::min(transform.firstIdx, defs[component]);
      transform.lastIdx = std::max(transform.lastIdx, defs[component]);
    }

    outTransform = transform;
    return true;
  }

  // mul rT, rA.x, c[K] then mad rT, rA.y, c[K+1], rT and so on, which is what fxc emits
  // for mul(matrix, vector) with column-major packing. The accumulator may hop between
  // temps, the component order is not fixed, and the steps need not be adjacent.
  bool DxsoAnalyzer::matchAccumulatorChain(
    const PositionDefinitions& defs,
          MatrixTransform&     outTransform) const {
    // The final step writes all four components at once, so every component must resolve
    // to the same instruction.
    for (uint32_t component = 1; component < 4; component++) {
      if (defs[component] != defs[0])
        return false;
    }

    MatrixTransform transform;
    transform.lastIdx = defs[0];

    bool haveSource = false;
    uint8_t componentMask = 0;
    int32_t currentIdx = defs[0];

    for (uint32_t step = 0; step < 4; step++) {
      const DxsoInstructionContext& ctx = m_instructions[size_t(currentIdx)].ctx;
      const DxsoOpcode opcode = ctx.instruction.opcode;

      if (opcode != DxsoOpcode::Mad && opcode != DxsoOpcode::Mul)
        return false;
      if (!isPlainDst(ctx.dst) || ctx.dst.mask != IdentityWriteMask)
        return false;

      const DxsoRegister* value = nullptr;
      const DxsoRegister* konst = nullptr;
      uint32_t component = 0;
      if (!splitBroadcastAndConst(ctx.src[0], ctx.src[1], &value, &konst, component))
        return false;

      if (!haveSource) {
        if (konst->id.num < component)
          return false;
        transform.source = value->id;
        transform.constBase = konst->id.num - component;
        haveSource = true;
      } else if (value->id != transform.source
              || konst->id.num != transform.constBase + component) {
        return false;
      }

      if ((componentMask & (1u << component)) != 0)
        return false;
      componentMask |= uint8_t(1u << component);

      if (opcode == DxsoOpcode::Mul) {
        // A mul starts the chain: by here every component must have been accounted for.
        if (componentMask != 0xF)
          return false;

        transform.firstIdx = currentIdx;
        outTransform = transform;
        return true;
      }

      // Follow the accumulator. It must be a whole vector produced by one instruction.
      if (!isReadableSourceRegister(ctx.src[2].id) || !isIdentitySource(ctx.src[2]))
        return false;

      const Definition accum = this->resolveDefinition(ctx.src[2].id, 0, currentIdx);
      if (accum < 0)
        return false;

      for (uint32_t i = 1; i < 4; i++) {
        if (this->resolveDefinition(ctx.src[2].id, i, currentIdx) != accum)
          return false;
      }

      currentIdx = accum;
    }

    return false;
  }

  // m4x4 dst, rA, c[K] - dst.i = dot(rA, c[K + i])
  bool DxsoAnalyzer::matchMatrixOp(
    const PositionDefinitions& defs,
          MatrixTransform&     outTransform) const {
    for (uint32_t component = 1; component < 4; component++) {
      if (defs[component] != defs[0])
        return false;
    }

    const DxsoInstructionContext& ctx = m_instructions[size_t(defs[0])].ctx;
    if (ctx.instruction.opcode != DxsoOpcode::M4x4)
      return false;
    if (!isPlainDst(ctx.dst) || ctx.dst.mask != IdentityWriteMask)
      return false;
    if (!isReadableSourceRegister(ctx.src[0].id) || !isIdentitySource(ctx.src[0]))
      return false;
    if (ctx.src[1].id.type != DxsoRegisterType::Const || !isIdentitySource(ctx.src[1]))
      return false;

    outTransform.source = ctx.src[0].id;
    outTransform.constBase = ctx.src[1].id.num;
    outTransform.firstIdx = defs[0];
    outTransform.lastIdx = defs[0];
    return true;
  }

  bool DxsoAnalyzer::isSourceStableAcross(
    const MatrixTransform& transform) const {
    // Input registers are never written.
    if (transform.source.type == DxsoRegisterType::Input)
      return true;

    // The snapshot is taken before firstIdx, so the transform only describes a single
    // position if nothing rewrites the source while the transform is still reading it.
    //
    // The final step is exempt: fxc routinely reuses the source register as the result of
    // the last multiply (`mad r0, c3, r0.w, r1` where r0 held the world position), which is
    // harmless because every read of the source has already happened by then.
    for (int32_t i = transform.firstIdx; i < transform.lastIdx; i++) {
      const DxsoInstructionContext& ctx = m_instructions[size_t(i)].ctx;

      if (!opcodeHasDestination(ctx.instruction.opcode))
        continue;
      if (ctx.dst.id == transform.source)
        return false;
    }

    return true;
  }

  std::string DxsoAnalyzer::describePositionDefinitions(
    const PositionDefinitions& defs) const {
    static const char kComponents[4] = { 'x', 'y', 'z', 'w' };

    std::string result;
    Definition previousIdx = -1;

    for (uint32_t component = 0; component < 4; component++) {
      if (!result.empty())
        result += " ";

      result += str::format("oPos.", kComponents[component], "<-");

      if (defs[component] < 0) {
        result += "unresolved";
        continue;
      }

      // The common case writes all four components with one instruction; only print it once.
      if (defs[component] == previousIdx) {
        result += "same";
        continue;
      }

      previousIdx = defs[component];
      result += str::format("[", defs[component], "] ",
                            describeInstruction(m_instructions[size_t(defs[component])].ctx));
    }

    return result;
  }

  void DxsoAnalyzer::resolvePreProjectionPosition() {
    DxsoPreProjectionPositionInfo& out = m_analysis->preProjPosition;

    if (m_programInfo.type() != DxsoProgramTypes::VertexShader) {
      out.failureReason = "not a vertex shader";
      return;
    }
    if (m_usesSubroutines) {
      out.failureReason = "shader uses subroutines";
      return;
    }

    PositionDefinitions defs = { -1, -1, -1, -1 };
    if (!this->findPositionDefinitions(defs)) {
      out.failureReason = "oPos components have no single unconditional producer";
      out.positionDefinitions = this->describePositionDefinitions(defs);
      return;
    }

    MatrixTransform transform;
    if (!this->matchAccumulatorChain(defs, transform)
     && !this->matchDp4Group(defs, transform)
     && !this->matchMatrixOp(defs, transform)) {
      out.failureReason = "oPos is not a constant matrix times one register";
      out.positionDefinitions = this->describePositionDefinitions(defs);
      return;
    }

    if (!this->isSourceStableAcross(transform)) {
      out.failureReason = "transform source is rewritten mid-transform";
      out.positionDefinitions = this->describePositionDefinitions(defs);
      return;
    }

    out.valid = true;
    out.sourceReg = transform.source;
    out.matrixConstBase = transform.constBase;
    out.snapshotInstructionIdx = m_instructions[size_t(transform.firstIdx)].ctx.instructionIdx;
    out.failureReason = "";
  }
  // NV-DXVK end

  void DxsoAnalyzer::finalize(size_t tokenCount) {
    m_analysis->bytecodeByteLength = tokenCount * sizeof(uint32_t);

    // NV-DXVK start: exact vertex capture
    this->resolvePreProjectionPosition();
    // NV-DXVK end
  }

}
