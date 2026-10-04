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
#include "dxso_opcode_util.h"

#include "../util/util_string.h"

namespace dxvk {

  bool isFloatConstantRegisterType(const DxsoRegisterType type) {
    return type == DxsoRegisterType::Const
        || type == DxsoRegisterType::Const2
        || type == DxsoRegisterType::Const3
        || type == DxsoRegisterType::Const4;
  }

  int32_t getFloatConstantRegisterIndex(const DxsoRegister& r) {
    switch (r.id.type) {
    case DxsoRegisterType::Const:
      return int32_t(r.id.num);
    case DxsoRegisterType::Const2:
      return 2048 + int32_t(r.id.num);
    case DxsoRegisterType::Const3:
      return 4096 + int32_t(r.id.num);
    case DxsoRegisterType::Const4:
      return 6144 + int32_t(r.id.num);
    default:
      return -1;
    }
  }

  bool decodeConstantModifierScale(const DxsoRegModifier modifier, float& outScale) {
    switch (modifier) {
    case DxsoRegModifier::None:
      outScale = 1.0f;
      return true;
    case DxsoRegModifier::Neg:
      outScale = -1.0f;
      return true;
    default:
      outScale = 1.0f;
      return false;
    }
  }

  // How many of DxsoInstructionContext::src an opcode actually reads. The decoder overwrites
  // that array in order and leaves the rest holding whatever the previous instruction put
  // there, so anything reading a slot the opcode does not use gets a stale register. The
  // existing coordinate heuristics tolerate that (an extra provenance bit only nudges
  // scoring); constant-dependency tracking cannot, because a stale constant would be dropped
  // from material identity and merge materials that a tint tells apart.
  //
  // Unknown opcodes report 0 so the tracking fails closed: a missed dependency leaves an
  // identity that churns and says so through [RTX-MicChurn], whereas an invented one silently
  // collapses two anchors into one.
  uint32_t getDxsoSourceOperandCount(const DxsoOpcode op) {
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
    case DxsoOpcode::SinCos:  // ps_3_0 folds away the two constant operands ps_2_0 required
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
    // Sampling ops report none. Their destination holds a sampled value, whose dependency on the
    // coordinate's constants matters only for a dependent texture read, and their operand layout
    // varies by shader model (ps_1_x carries the coordinate in the destination). Under-reporting
    // costs a dependent read's taint and is caught by [RTX-MicChurn]; guessing wrong would taint
    // from a stale slot and merge two anchors.
    default:
      return 0;
    }
  }

  // Matrix multiplies name only the first row's constant register; the rest of the matrix
  // occupies the registers immediately after it. Returns 0 for everything else.
  uint32_t getDxsoMatrixRowCount(const DxsoOpcode op) {
    switch (op) {
    case DxsoOpcode::M4x4: return 4;
    case DxsoOpcode::M4x3:
    case DxsoOpcode::M3x4:
    case DxsoOpcode::M3x3: return 3;
    case DxsoOpcode::M3x2: return 2;
    default:               return 0;
    }
  }

  static const char* dxsoRegisterTypePrefix(const DxsoRegisterType type) {
    switch (type) {
    case DxsoRegisterType::Temp:          return "r";
    case DxsoRegisterType::Input:         return "v";
    case DxsoRegisterType::Const:         return "c";
    case DxsoRegisterType::Texture:       return "t";   // Addr in VS
    case DxsoRegisterType::RasterizerOut: return "rast";
    case DxsoRegisterType::AttributeOut:  return "oD";
    case DxsoRegisterType::Output:        return "o";   // TexcoordOut pre-3.0
    case DxsoRegisterType::ConstInt:      return "i";
    case DxsoRegisterType::ColorOut:      return "oC";
    case DxsoRegisterType::DepthOut:      return "oDepth";
    case DxsoRegisterType::Sampler:       return "s";
    case DxsoRegisterType::Const2:        return "c2_";
    case DxsoRegisterType::Const3:        return "c3_";
    case DxsoRegisterType::Const4:        return "c4_";
    case DxsoRegisterType::ConstBool:     return "b";
    case DxsoRegisterType::Loop:          return "aL";
    case DxsoRegisterType::TempFloat16:   return "rh";
    case DxsoRegisterType::MiscType:      return "misc";
    case DxsoRegisterType::Label:         return "label";
    case DxsoRegisterType::Predicate:     return "p";
    case DxsoRegisterType::PixelTexcoord: return "t";
    default:                              return "reg";
    }
  }

  static const char* dxsoRegModifierName(const DxsoRegModifier modifier) {
    switch (modifier) {
    case DxsoRegModifier::None:    return "";
    case DxsoRegModifier::Neg:     return "_neg";
    case DxsoRegModifier::Bias:    return "_bias";
    case DxsoRegModifier::BiasNeg: return "_biasneg";
    case DxsoRegModifier::Sign:    return "_bx2";
    case DxsoRegModifier::SignNeg: return "_bx2neg";
    case DxsoRegModifier::Comp:    return "_comp";
    case DxsoRegModifier::X2:      return "_x2";
    case DxsoRegModifier::X2Neg:   return "_x2neg";
    case DxsoRegModifier::Dz:      return "_dz";
    case DxsoRegModifier::Dw:      return "_dw";
    case DxsoRegModifier::Abs:     return "_abs";
    case DxsoRegModifier::AbsNeg:  return "_absneg";
    case DxsoRegModifier::Not:     return "_not";
    default:                       return "_mod?";
    }
  }

  std::string formatDxsoSrcRegister(const DxsoRegister& r) {
    std::string s = str::format(dxsoRegisterTypePrefix(r.id.type), r.id.num);
    if (r.hasRelative) {
      s += "[rel]";
    }
    std::string swizzle;
    bool identitySwizzle = true;
    for (uint32_t c = 0; c < 4u; c++) {
      const uint32_t sourceComponent = r.swizzle[c];
      swizzle += "xyzw"[sourceComponent & 0x3u];
      identitySwizzle &= sourceComponent == c;
    }
    if (!identitySwizzle) {
      s += str::format(".", swizzle);
    }
    s += dxsoRegModifierName(r.modifier);
    return s;
  }

  std::string formatDxsoDstRegister(const DxsoRegister& r) {
    std::string s = str::format(dxsoRegisterTypePrefix(r.id.type), r.id.num);
    if (r.mask != IdentityWriteMask) {
      s += ".";
      for (uint32_t c = 0; c < 4u; c++) {
        if (r.mask[c]) {
          s += "xyzw"[c];
        }
      }
    }
    if (r.saturate) {
      s += "_sat";
    }
    if (r.shift != 0) {
      s += str::format("_shift", int32_t(r.shift));
    }
    return s;
  }

}
