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
#include "dxso_uv_dataflow.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "dxso_code.h"
#include "dxso_tables.h"
#include "../util/util_string.h"
#include "../util/util_vector.h"

namespace dxvk {

  uint32_t findVsTexcoordOutputRegister(const DxsoIsgn& osgn, uint32_t usageIndex) {
    for (uint32_t i = 0; i < osgn.elemCount; i++) {
      const auto& decl = osgn.elems[i];
      if (decl.semantic.usage == DxsoUsage::Texcoord && decl.semantic.usageIndex == usageIndex) {
        return decl.regNumber;
      }
    }

    return std::numeric_limits<uint32_t>::max();
  }

  // The tracer proves, per register component, which interpolant (PS) or IA texcoord input (VS)
  // component a value originates from, and accumulates the affine chain applied to it. Math
  // outside the UvComponentAffine model keeps the origin but marks the affine inexact.

  bool uvAffineTermPresent(const UvAffineTerm& t) {
    return t.immValid || t.constReg >= 0;
  }

  static bool uvAffineTermsEqual(const UvAffineTerm& a, const UvAffineTerm& b) {
    if (a.inexact != b.inexact || a.immValid != b.immValid ||
        a.constReg != b.constReg || a.constReg2 != b.constReg2) {
      return false;
    }
    if (a.immValid && a.imm != b.imm) {
      return false;
    }
    if (a.constReg >= 0 && (a.constComp != b.constComp || a.factor != b.factor)) {
      return false;
    }
    if (a.constReg2 >= 0 && (a.constComp2 != b.constComp2 || a.factor2 != b.factor2)) {
      return false;
    }
    return true;
  }

  static bool uvComponentAffinesEqual(const UvComponentAffine& a, const UvComponentAffine& b) {
    if (a.hasCross != b.hasCross) {
      return false;
    }
    if (!uvAffineTermsEqual(a.scale, b.scale) || !uvAffineTermsEqual(a.offset, b.offset)) {
      return false;
    }
    if (!a.hasCross) {
      return true;
    }
    return a.scaleComponent == b.scaleComponent &&
           a.crossComponent == b.crossComponent &&
           uvAffineTermsEqual(a.cross, b.cross);
  }

  bool uvComponentAffineExact(const UvComponentAffine& a) {
    if (a.scale.inexact || a.offset.inexact) {
      return false;
    }
    return !a.hasCross || !a.cross.inexact;
  }

  static bool uvComponentAffineIsIdentity(const UvComponentAffine& a) {
    if (a.hasCross) {
      return false;
    }
    return uvComponentAffineExact(a) &&
           (!uvAffineTermPresent(a.scale) ||
            (a.scale.immValid && a.scale.imm == 1.0f && a.scale.constReg < 0)) &&
           (!uvAffineTermPresent(a.offset) ||
            (a.offset.immValid && a.offset.imm == 0.0f && a.offset.constReg < 0));
  }

  // Compile-time scale magnitude of a static-tiling sample site. Identity scales do not qualify,
  // since a plain base layer under a tiled detail layer is not dual tiling, and neither do panner
  // offsets, since scrolling layers are visible together.
  static bool uvSiteStaticTilingMagnitude(const UvComponentAffine& affU,
                                          const UvComponentAffine& affV,
                                          float& outMagnitude) {
    auto explicitImmediateScale = [](const UvAffineTerm& term, float& outValue) {
      if (term.inexact || !term.immValid || term.constReg >= 0) {
        return false;
      }
      outValue = term.imm;
      return true;
    };
    auto compileTimeOffset = [](const UvAffineTerm& term) {
      return !term.inexact && term.constReg < 0;
    };

    if (affU.hasCross || affV.hasCross) {
      return false;
    }

    if (!compileTimeOffset(affU.offset) || !compileTimeOffset(affV.offset)) {
      return false;
    }

    float scaleU = 1.0f;
    float scaleV = 1.0f;
    if (!explicitImmediateScale(affU.scale, scaleU) || !explicitImmediateScale(affV.scale, scaleV)) {
      return false;
    }

    outMagnitude = std::abs(scaleU * scaleV);
    return std::isfinite(outMagnitude) && outMagnitude > 0.0f;
  }

  static void uvAffineMarkInexact(UvComponentAffine& a) {
    a.scale.inexact = true;
    a.offset.inexact = true;
    if (a.hasCross) {
      a.cross.inexact = true;
    }
  }

  static void uvAffineTermScaleMulImmediate(UvAffineTerm& t, const float value) {
    if (t.constReg >= 0) {
      t.factor *= value;
    } else if (t.immValid) {
      t.imm *= value;
    } else {
      t.immValid = true;
      t.imm = value;
    }
  }

  // An offset term is a sum, so multiplying it distributes over every part.
  static void uvAffineTermOffsetMulImmediate(UvAffineTerm& t, const float value) {
    if (t.immValid) {
      t.imm *= value;
    }
    if (t.constReg >= 0) {
      t.factor *= value;
    }
    if (t.constReg2 >= 0) {
      t.factor2 *= value;
    }
  }

  static void uvAffineTermScaleMulConstant(UvAffineTerm& t, const int16_t reg, const uint8_t comp, const float factor) {
    if (t.constReg >= 0) {
      // the term would become a product of two draw-time constants - not representable
      t.inexact = true;
    } else if (t.immValid) {
      const float staticScale = t.imm;
      t.immValid = false;
      t.imm = 0.0f;
      t.constReg = reg;
      t.constComp = comp;
      t.factor = staticScale * factor;
    } else {
      t.constReg = reg;
      t.constComp = comp;
      t.factor = factor;
    }
  }

  static void uvAffineTermOffsetMulConstant(UvAffineTerm& t, const int16_t reg, const uint8_t comp, const float factor) {
    if (t.constReg >= 0) {
      // any existing constant part times a new draw-time constant is a product of two
      // draw-time constants - not representable
      t.inexact = true;
    } else if (t.immValid) {
      if (t.imm != 0.0f) {
        const float staticOffset = t.imm;
        t.immValid = false;
        t.imm = 0.0f;
        t.constReg = reg;
        t.constComp = comp;
        t.factor = staticOffset * factor;
      } else {
        t.immValid = false;
      }
    }
  }

  static void uvAffineTermAddImmediate(UvAffineTerm& t, const float value) {
    if (value == 0.0f) {
      return;
    }
    t.immValid = true;
    t.imm += value;
  }

  static void uvAffineTermAddConstant(UvAffineTerm& t, const int16_t reg, const uint8_t comp, const float factor) {
    if (t.immValid && t.imm == 0.0f) {
      t.immValid = false;
    }

    if (t.constReg >= 0 && t.constComp == comp && t.constReg == reg) {
      // same component referenced twice: fold into the first part's factor
      t.factor += factor;
      if (t.factor == 0.0f) {
        // cancelled out: promote the second part into the first slot to keep the
        // "constReg2 only set when constReg is" invariant
        t.constReg = t.constReg2;
        t.constComp = t.constComp2;
        t.factor = t.factor2;
        t.constReg2 = -1;
        t.constComp2 = 0;
        t.factor2 = 1.0f;
      }
      return;
    }
    if (t.constReg2 >= 0 && t.constComp2 == comp && t.constReg2 == reg) {
      t.factor2 += factor;
      if (t.factor2 == 0.0f) {
        t.constReg2 = -1;
        t.constComp2 = 0;
        t.factor2 = 1.0f;
      }
      return;
    }

    if (t.constReg < 0) {
      t.constReg = reg;
      t.constComp = comp;
      t.factor = factor;
    } else if (t.constReg2 < 0) {
      t.constReg2 = reg;
      t.constComp2 = comp;
      t.factor2 = factor;
    } else {
      // more than two distinct constant parts - not representable
      t.inexact = true;
    }
  }

  // value' = value * m for a compile-time immediate m. `cross` scales like `scale` does:
  // both multiply an interpolant component, so both take the coefficient.
  static void uvAffineMulImmediate(UvComponentAffine& a, const float value) {
    uvAffineTermScaleMulImmediate(a.scale, value);
    uvAffineTermOffsetMulImmediate(a.offset, value);
    if (a.hasCross) {
      uvAffineTermScaleMulImmediate(a.cross, value);
    }
  }

  // value' = value * consts[reg][comp] * factor
  static void uvAffineMulConstant(UvComponentAffine& a, const int16_t reg, const uint8_t comp, const float factor) {
    uvAffineTermScaleMulConstant(a.scale, reg, comp, factor);
    uvAffineTermOffsetMulConstant(a.offset, reg, comp, factor);
    if (a.hasCross) {
      uvAffineTermScaleMulConstant(a.cross, reg, comp, factor);
    }
  }

  static void uvAffineAddImmediate(UvComponentAffine& a, const float value) {
    uvAffineTermAddImmediate(a.offset, value);
  }

  static void uvAffineAddConstant(UvComponentAffine& a, const int16_t reg, const uint8_t comp, const float factor) {
    uvAffineTermAddConstant(a.offset, reg, comp, factor);
  }

  // Applies a DXSO source modifier to the affine chain; returns false when not representable.
  static bool uvAffineApplySourceModifier(UvComponentAffine& a, const DxsoRegModifier modifier) {
    switch (modifier) {
    case DxsoRegModifier::None:
      return true;
    case DxsoRegModifier::Neg:
      uvAffineMulImmediate(a, -1.0f);
      return true;
    case DxsoRegModifier::Bias: // r - 0.5
      uvAffineAddImmediate(a, -0.5f);
      return true;
    case DxsoRegModifier::BiasNeg: // -(r - 0.5)
      uvAffineAddImmediate(a, -0.5f);
      uvAffineMulImmediate(a, -1.0f);
      return true;
    case DxsoRegModifier::Sign: // r * 2 - 1
      uvAffineMulImmediate(a, 2.0f);
      uvAffineAddImmediate(a, -1.0f);
      return true;
    case DxsoRegModifier::SignNeg: // -(r * 2 - 1)
      uvAffineMulImmediate(a, -2.0f);
      uvAffineAddImmediate(a, 1.0f);
      return true;
    case DxsoRegModifier::Comp: // 1 - r
      uvAffineMulImmediate(a, -1.0f);
      uvAffineAddImmediate(a, 1.0f);
      return true;
    case DxsoRegModifier::X2:
      uvAffineMulImmediate(a, 2.0f);
      return true;
    case DxsoRegModifier::X2Neg:
      uvAffineMulImmediate(a, -2.0f);
      return true;
    default:
      return false; // Dz/Dw/Abs/AbsNeg/Not: origin survives, affine does not
    }
  }

  struct UvExactComponentOrigin {
    bool valid = false;
    uint8_t reg = 0;                 // PS pass: TEXCOORD usage index; VS pass: input register number
    uint8_t component = 0;           // primary source component (meaningful when componentMask has one bit)
    uint8_t componentMask = 0;       // all origin components contributing to this value (bit per component)
    UvComponentAffine affine;
  };

  static bool uvIsSingleComponentMask(const uint8_t mask) {
    return mask != 0u && (mask & (mask - 1u)) == 0u;
  }

  static bool uvOriginSameRegister(const UvExactComponentOrigin& a, const UvExactComponentOrigin& b) {
    return a.valid && b.valid && a.reg == b.reg;
  }

  static bool uvAffineScaleNonIdentity(const UvComponentAffine& a) {
    return uvAffineTermPresent(a.scale) &&
           !(a.scale.immValid && a.scale.imm == 1.0f && a.scale.constReg < 0);
  }

  static void uvAffineTermAddTerm(UvAffineTerm& dest, const UvAffineTerm& src) {
    if (src.inexact) {
      dest.inexact = true;
      return;
    }
    if (src.immValid) {
      uvAffineTermAddImmediate(dest, src.imm);
    }
    if (src.constReg >= 0) {
      uvAffineTermAddConstant(dest, src.constReg, src.constComp, src.factor);
    }
    if (src.constReg2 >= 0) {
      uvAffineTermAddConstant(dest, src.constReg2, src.constComp2, src.factor2);
    }
  }

  // Fold sibling into dest as the cross coefficient: dest = dest + sibling, with dest.scale
  // multiplying dest.component and dest.cross multiplying sibling.component. Fails (and
  // leaves dest unchanged) when the result would leave the scale/cross/offset model.
  static bool uvAffineTryAttachCross(UvExactComponentOrigin& dest, const UvExactComponentOrigin& sibling) {
    if (!uvOriginSameRegister(dest, sibling)) {
      return false;
    }
    if (dest.affine.hasCross || sibling.affine.hasCross) {
      return false;
    }
    if (!uvComponentAffineExact(dest.affine) || !uvComponentAffineExact(sibling.affine)) {
      return false;
    }
    if (!uvIsSingleComponentMask(dest.componentMask) || !uvIsSingleComponentMask(sibling.componentMask)) {
      return false;
    }
    if (dest.component == sibling.component) {
      return false;
    }

    UvComponentAffine combined = dest.affine;
    combined.hasCross = true;
    combined.scaleComponent = dest.component;
    combined.crossComponent = sibling.component;
    if (uvAffineTermPresent(sibling.affine.scale)) {
      combined.cross = sibling.affine.scale;
    } else {
      combined.cross.immValid = true;
      combined.cross.imm = 1.0f;
    }
    uvAffineTermAddTerm(combined.offset, sibling.affine.offset);
    if (!uvComponentAffineExact(combined)) {
      return false;
    }

    dest.componentMask = uint8_t(dest.componentMask | sibling.componentMask);
    dest.affine = combined;
    return true;
  }

  static bool uvAffineMixUsesPair(const UvComponentAffine& a, const uint8_t compU, const uint8_t compV) {
    if (!a.hasCross) {
      return true;
    }
    const auto isPair = [&](const uint8_t c) {
      return c == compU || c == compV;
    };
    return isPair(a.scaleComponent) && isPair(a.crossComponent);
  }

  static bool uvAffineIsExactLinearPair(const UvComponentAffine& u,
                                        const UvComponentAffine& v,
                                        const uint8_t compU,
                                        const uint8_t compV) {
    if (!u.hasCross && !v.hasCross) {
      return false;
    }
    return uvComponentAffineExact(u) && uvComponentAffineExact(v) &&
           uvAffineMixUsesPair(u, compU, compV) &&
           uvAffineMixUsesPair(v, compU, compV);
  }

  // Merges two value origins that both contribute to one result component (blend endpoints,
  // clamps, dot products, etc.). Keeping the origin with an inexact affine when one contributor
  // is unknown deliberately models "base UV plus per-pixel detail" patterns (bump offset,
  // distortion). Component mixing (e.g. UV rotators) unions the component mask so the sample
  // site can still be attributed to the packed UV half it reads from.
  static UvExactComponentOrigin uvMergeOrigins(const UvExactComponentOrigin& a, const UvExactComponentOrigin& b) {
    if (!a.valid && !b.valid) {
      return UvExactComponentOrigin{};
    }

    if (a.valid != b.valid) {
      UvExactComponentOrigin merged = a.valid ? a : b;
      uvAffineMarkInexact(merged.affine);
      return merged;
    }

    if (a.reg != b.reg) {
      return UvExactComponentOrigin{};
    }

    UvExactComponentOrigin merged = a;
    merged.componentMask = uint8_t(a.componentMask | b.componentMask);
    if (!uvIsSingleComponentMask(merged.componentMask)) {
      uvAffineMarkInexact(merged.affine);
    } else if (!uvComponentAffinesEqual(a.affine, b.affine)) {
      uvAffineMarkInexact(merged.affine);
    }
    return merged;
  }

  struct UvConstComponentRef {
    bool valid = false;
    bool isImmediate = false;
    float immediate = 0.0f;
    int16_t constReg = -1;
    uint8_t constComp = 0;
    float factor = 1.0f;
  };

  static void uvAffineMulConstRef(UvComponentAffine& a, const UvConstComponentRef& ref) {
    if (ref.isImmediate) {
      uvAffineMulImmediate(a, ref.immediate);
    } else {
      uvAffineMulConstant(a, ref.constReg, ref.constComp, ref.factor);
    }
  }

  static void uvAffineAddConstRef(UvComponentAffine& a, const UvConstComponentRef& ref, const float sign) {
    if (ref.isImmediate) {
      uvAffineAddImmediate(a, sign * ref.immediate);
    } else {
      uvAffineAddConstant(a, ref.constReg, ref.constComp, sign * ref.factor);
    }
  }

  static void uvAffineTermCollectConstRegs(const UvAffineTerm& term,
                                           int32_t* regs,
                                           uint32_t& count,
                                           const uint32_t cap) {
    auto push = [&](const int16_t r) {
      if (r >= 0 && count < cap) {
        regs[count++] = int32_t(r);
      }
    };
    push(term.constReg);
    push(term.constReg2);
  }

  void uvComponentAffineCollectConstRegs(const UvComponentAffine& a,
                                         int32_t* regs,
                                         uint32_t& count,
                                         const uint32_t cap) {
    uvAffineTermCollectConstRegs(a.scale, regs, count, cap);
    uvAffineTermCollectConstRegs(a.offset, regs, count, cap);
    if (a.hasCross) {
      uvAffineTermCollectConstRegs(a.cross, regs, count, cap);
    }
  }

  // -------------------------------------------------------------------------
  // Formatting helpers for rtx.d3d9.ue3LogUvAffineDetail / ue3UvTraceShaderHashes
  // -------------------------------------------------------------------------

  // Renders one UV affine term (sum of a `def` immediate and up to two draw-time
  // constant register components times static factors).
  static std::string formatUvAffineTerm(const UvAffineTerm& term, const char* identity) {
    std::string s;
    auto appendConstPart = [&s](const int16_t reg, const uint8_t comp, const float factor) {
      if (!s.empty()) {
        s += "+";
      }
      s += str::format("c", reg, ".", "xyzw"[comp & 0x3u]);
      if (factor != 1.0f) {
        s += str::format("*", factor);
      }
    };
    // suppress a redundant "0+" in front of constant parts
    if (term.immValid && (term.imm != 0.0f || term.constReg < 0)) {
      s = str::format(term.imm);
    }
    if (term.constReg >= 0) {
      appendConstPart(term.constReg, term.constComp, term.factor);
    }
    if (term.constReg2 >= 0) {
      appendConstPart(term.constReg2, term.constComp2, term.factor2);
    }
    if (s.empty()) {
      s = identity;
    }
    if (term.inexact) {
      s += "(INEXACT)";
    }
    return s;
  }

  std::string formatUvComponentAffine(const UvComponentAffine& affine) {
    if (!affine.hasCross) {
      return str::format("uv*", formatUvAffineTerm(affine.scale, "1"), "+", formatUvAffineTerm(affine.offset, "0"));
    }
    return str::format(
      "uv.", "xyzw"[affine.scaleComponent & 0x3u], "*", formatUvAffineTerm(affine.scale, "1"),
      "+uv.", "xyzw"[affine.crossComponent & 0x3u], "*", formatUvAffineTerm(affine.cross, "0"),
      "+", formatUvAffineTerm(affine.offset, "0"));
  }

  static std::string formatUvConstExpr(const UvConstComponentRef& ref) {
    if (!ref.valid) {
      return "-";
    }
    if (ref.isImmediate) {
      return str::format(ref.immediate);
    }
    std::string s = str::format("c", ref.constReg, ".", "xyzw"[ref.constComp & 0x3u]);
    if (ref.factor != 1.0f) {
      s += str::format("*", ref.factor);
    }
    return s;
  }

  class UvDataflowTracer {
  public:
    UvDataflowTracer(const std::array<int8_t, 2 * DxsoMaxInterfaceRegs>& inputRegToTexcoord,
                     const bool texcoordRegsAreInterpolants,
                     const bool originIsSemanticIndex)
      : m_inputRegToTexcoord(inputRegToTexcoord)
      , m_texcoordRegsAreInterpolants(texcoordRegsAreInterpolants)
      , m_originIsSemanticIndex(originIsSemanticIndex) {
      m_defConstValid.fill(0);
    }

    void setTrackedOutputRegister(const uint32_t reg) {
      m_trackedOutputReg = reg;
    }

    const UvExactComponentOrigin& outputOrigin(const uint32_t component) const {
      return m_outputOrigins[component & 0x3u];
    }

    // diagnostics accessors for the instruction-level UV trace
    const UvExactComponentOrigin& tempOrigin(const uint32_t reg, const uint32_t component) const {
      return m_tempOrigins[reg % m_tempOrigins.size()][component & 0x3u];
    }

    const UvConstComponentRef& tempConstExpr(const uint32_t reg, const uint32_t component) const {
      return m_tempConstExprs[reg % m_tempConstExprs.size()][component & 0x3u];
    }

    UvConstComponentRef readConst(const DxsoRegister& r, const uint32_t dstComponent) const {
      UvConstComponentRef ref;
      if (r.hasRelative) {
        return ref;
      }

      // temps holding tracked pure-constant expressions (fxc hoists constant
      // subexpressions into temps before applying them to UVs) read like constants
      if (r.id.type == DxsoRegisterType::Temp || r.id.type == DxsoRegisterType::TempFloat16) {
        if (r.id.num >= m_tempConstExprs.size()) {
          return ref;
        }
        const uint8_t comp = uint8_t(r.swizzle[dstComponent & 0x3u] & 0x3u);
        const UvConstComponentRef& tracked = m_tempConstExprs[r.id.num][comp];
        if (!tracked.valid) {
          return ref;
        }
        float modifierScale = 1.0f;
        if (!decodeConstantModifierScale(r.modifier, modifierScale)) {
          return ref;
        }
        ref = tracked;
        if (ref.isImmediate) {
          ref.immediate *= modifierScale;
        } else {
          ref.factor *= modifierScale;
        }
        return ref;
      }

      if (!isFloatConstantRegisterType(r.id.type)) {
        return ref;
      }

      float modifierScale = 1.0f;
      if (!decodeConstantModifierScale(r.modifier, modifierScale)) {
        return ref;
      }

      const int32_t reg = getFloatConstantRegisterIndex(r);
      if (reg < 0 || reg > int32_t(std::numeric_limits<int16_t>::max())) {
        return ref;
      }

      const uint8_t comp = uint8_t(r.swizzle[dstComponent & 0x3u] & 0x3u);
      ref.valid = true;
      if (reg < int32_t(m_defConstValid.size()) && m_defConstValid[reg]) {
        ref.isImmediate = true;
        ref.immediate = modifierScale * m_defConsts[reg][comp];
      } else {
        ref.constReg = int16_t(reg);
        ref.constComp = comp;
        ref.factor = modifierScale;
      }
      return ref;
    }

    UvExactComponentOrigin readOrigin(const DxsoRegister& r, const uint32_t dstComponent) const {
      const uint8_t srcComponent = uint8_t(r.swizzle[dstComponent & 0x3u] & 0x3u);
      UvExactComponentOrigin origin;

      switch (r.id.type) {
      case DxsoRegisterType::Texture: // == Addr in VS; only meaningful for PS
      case DxsoRegisterType::PixelTexcoord:
        if (m_texcoordRegsAreInterpolants) {
          origin.valid = true;
          origin.reg = m_originIsSemanticIndex ? mapRegToSemantic(r.id.num, true) : uint8_t(r.id.num);
          origin.component = srcComponent;
          origin.componentMask = uint8_t(1u << srcComponent);
        }
        break;
      case DxsoRegisterType::Input:
        if (r.id.num < m_inputRegToTexcoord.size() && m_inputRegToTexcoord[r.id.num] >= 0) {
          origin.valid = true;
          origin.reg = m_originIsSemanticIndex ? uint8_t(m_inputRegToTexcoord[r.id.num]) : uint8_t(r.id.num);
          origin.component = srcComponent;
          origin.componentMask = uint8_t(1u << srcComponent);
        }
        break;
      case DxsoRegisterType::Temp:
      case DxsoRegisterType::TempFloat16:
        if (r.id.num < m_tempOrigins.size()) {
          origin = m_tempOrigins[r.id.num][srcComponent];
        }
        break;
      default:
        break;
      }

      if (origin.valid && r.modifier != DxsoRegModifier::None) {
        if (!uvAffineApplySourceModifier(origin.affine, r.modifier)) {
          uvAffineMarkInexact(origin.affine);
        }
      }

      return origin;
    }

    // Merges the origins of the first `componentCount` components of a source operand.
    // Used for dot-product style consumption where several components feed one result.
    UvExactComponentOrigin readOriginUnion(const DxsoRegister& r, const uint32_t componentCount) const {
      UvExactComponentOrigin merged;
      bool first = true;
      for (uint32_t c = 0; c < componentCount && c < 4u; c++) {
        const UvExactComponentOrigin componentOrigin = readOrigin(r, c);
        if (first) {
          merged = componentOrigin;
          first = false;
        } else {
          merged = uvMergeOrigins(merged, componentOrigin);
        }
      }
      return merged;
    }

    void processInstruction(const DxsoInstructionContext& ctx) {
      const DxsoOpcode op = ctx.instruction.opcode;

      if (op == DxsoOpcode::Def &&
          ctx.dst.id.type == DxsoRegisterType::Const &&
          ctx.dst.id.num < m_defConsts.size()) {
        m_defConstValid[ctx.dst.id.num] = 1;
        m_defConsts[ctx.dst.id.num] = Vector4(
          ctx.def.float32[0],
          ctx.def.float32[1],
          ctx.def.float32[2],
          ctx.def.float32[3]);
        return;
      }

      // Flow control makes a linear trace unsound (conditional writes, loop-carried values):
      // conservatively drop all derived provenance at every flow-control boundary. Values read
      // directly from interpolants/inputs after the boundary remain provable.
      switch (op) {
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
      case DxsoOpcode::Call:
      case DxsoOpcode::CallNz:
      case DxsoOpcode::Label:
        for (auto& tempComponents : m_tempOrigins) {
          tempComponents = {};
        }
        for (auto& tempConstExprComponents : m_tempConstExprs) {
          tempConstExprComponents = {};
        }
        for (auto& outputComponent : m_outputOrigins) {
          outputComponent = UvExactComponentOrigin{};
        }
        return;
      // note: a trailing `ret` in main must not clear already-written output origins
      case DxsoOpcode::Ret:
        for (auto& tempComponents : m_tempOrigins) {
          tempComponents = {};
        }
        for (auto& tempConstExprComponents : m_tempConstExprs) {
          tempConstExprComponents = {};
        }
        return;
      default:
        break;
      }

      // non-arithmetic opcodes whose dst token is not a value write
      if (op == DxsoOpcode::Dcl || op == DxsoOpcode::Def ||
          op == DxsoOpcode::DefI || op == DxsoOpcode::DefB ||
          op == DxsoOpcode::TexKill || op == DxsoOpcode::Comment) {
        return;
      }

      const bool dstIsTemp =
        ctx.dst.id.type == DxsoRegisterType::Temp ||
        ctx.dst.id.type == DxsoRegisterType::TempFloat16;
      const bool dstIsTrackedOutput =
        m_trackedOutputReg != std::numeric_limits<uint32_t>::max() &&
        ctx.dst.id.type == DxsoRegisterType::Output &&
        ctx.dst.id.num == m_trackedOutputReg;

      if (!dstIsTemp && !dstIsTrackedOutput) {
        return;
      }
      if (dstIsTemp && ctx.dst.id.num >= m_tempOrigins.size()) {
        return;
      }

      for (uint32_t c = 0; c < 4u; c++) {
        if (!ctx.dst.mask[c]) {
          continue;
        }

        UvExactComponentOrigin origin;
        UvConstComponentRef constExpr;
        // predicated writes are conditional - the resulting value origin is unknowable
        if (!ctx.instruction.predicated) {
          origin = traceComponent(ctx, c);
          if (dstIsTemp && !origin.valid) {
            constExpr = traceConstExpr(ctx, c);
          }
        }

        if (origin.valid) {
          if (ctx.dst.shift != 0) {
            uvAffineMulImmediate(origin.affine, std::exp2(float(ctx.dst.shift)));
          }
          if (ctx.dst.saturate) {
            uvAffineMarkInexact(origin.affine);
          }
        }

        if (constExpr.valid) {
          if (ctx.dst.shift != 0) {
            const float shiftScale = std::exp2(float(ctx.dst.shift));
            if (constExpr.isImmediate) {
              constExpr.immediate *= shiftScale;
            } else {
              constExpr.factor *= shiftScale;
            }
          }
          if (ctx.dst.saturate) {
            if (constExpr.isImmediate) {
              constExpr.immediate = std::clamp(constExpr.immediate, 0.0f, 1.0f);
            } else {
              // clamped draw-time value: not representable as const * factor
              constExpr = UvConstComponentRef{};
            }
          }
        }

        if (dstIsTemp) {
          m_tempOrigins[ctx.dst.id.num][c] = origin;
          m_tempConstExprs[ctx.dst.id.num][c] = constExpr;
        } else {
          m_outputOrigins[c] = origin;
        }
      }
    }

    uint8_t mapRegToSemantic(const uint32_t regNum, const bool allowLegacyFallback) const {
      if (regNum < m_inputRegToTexcoord.size() && m_inputRegToTexcoord[regNum] >= 0) {
        return uint8_t(m_inputRegToTexcoord[regNum]);
      }
      return allowLegacyFallback ? uint8_t(regNum & 0b111) : uint8_t(regNum);
    }

  private:
    UvExactComponentOrigin traceComponent(const DxsoInstructionContext& ctx, const uint32_t c) const {
      switch (ctx.instruction.opcode) {
      case DxsoOpcode::Mov:
      case DxsoOpcode::Frc: // fract() is UV-equivalent under wrap addressing
      case DxsoOpcode::TexCoord: // ps_1_4 texcrd: moves a texcoord register into a temp
        return readOrigin(ctx.src[0], c);

      case DxsoOpcode::Abs: {
        UvExactComponentOrigin origin = readOrigin(ctx.src[0], c);
        if (origin.valid) {
          uvAffineMarkInexact(origin.affine);
        }
        return origin;
      }

      case DxsoOpcode::Mul: {
        const UvExactComponentOrigin o0 = readOrigin(ctx.src[0], c);
        const UvExactComponentOrigin o1 = readOrigin(ctx.src[1], c);
        const UvConstComponentRef k0 = readConst(ctx.src[0], c);
        const UvConstComponentRef k1 = readConst(ctx.src[1], c);

        if (o0.valid && !o1.valid && k1.valid) {
          UvExactComponentOrigin origin = o0;
          uvAffineMulConstRef(origin.affine, k1);
          return origin;
        }
        if (o1.valid && !o0.valid && k0.valid) {
          UvExactComponentOrigin origin = o1;
          uvAffineMulConstRef(origin.affine, k0);
          return origin;
        }
        if (o0.valid && o1.valid) {
          UvExactComponentOrigin merged = uvMergeOrigins(o0, o1);
          if (merged.valid) {
            uvAffineMarkInexact(merged.affine); // value * value is never affine
          }
          return merged;
        }
        if (o0.valid || o1.valid) {
          // origin times an unknown factor: keep base provenance, scale unknown
          UvExactComponentOrigin origin = o0.valid ? o0 : o1;
          origin.affine.scale.inexact = true;
          if (uvAffineTermPresent(origin.affine.offset)) {
            origin.affine.offset.inexact = true;
          }
          return origin;
        }
        return UvExactComponentOrigin{};
      }

      case DxsoOpcode::Add:
      case DxsoOpcode::Sub: {
        const bool isSub = ctx.instruction.opcode == DxsoOpcode::Sub;
        const UvExactComponentOrigin o0 = readOrigin(ctx.src[0], c);
        UvExactComponentOrigin o1 = readOrigin(ctx.src[1], c);
        if (o1.valid && isSub) {
          uvAffineMulImmediate(o1.affine, -1.0f);
        }
        const UvConstComponentRef k0 = readConst(ctx.src[0], c);
        const UvConstComponentRef k1 = readConst(ctx.src[1], c);

        if (o0.valid && !o1.valid && k1.valid) {
          UvExactComponentOrigin origin = o0;
          uvAffineAddConstRef(origin.affine, k1, isSub ? -1.0f : 1.0f);
          return origin;
        }
        if (o1.valid && !o0.valid && k0.valid) {
          UvExactComponentOrigin origin = o1; // already negated when subtracting
          uvAffineAddConstRef(origin.affine, k0, 1.0f);
          return origin;
        }
        if (o0.valid && o1.valid) {
          // x + x with identical origin and affine is an exact doubling: fxc strength-reduces
          // `uv * 2` (e.g. UTiling/VTiling = 2 literals) into a self-add
          if (!isSub &&
              uvOriginSameRegister(o0, o1) &&
              o0.componentMask == o1.componentMask &&
              uvIsSingleComponentMask(o0.componentMask) &&
              uvComponentAffinesEqual(o0.affine, o1.affine)) {
            UvExactComponentOrigin origin = o0;
            uvAffineMulImmediate(origin.affine, 2.0f);
            return origin;
          }
          // UV matrix / rotator in the form fxc emits when it expands a dot into mul/mad:
          // a sum of two interpolant components, at least one already carrying a
          // non-identity scale. Requiring that scale is what keeps a bare U+V - which is
          // not a coordinate transform - out of the model.
          if (uvAffineScaleNonIdentity(o0.affine) || uvAffineScaleNonIdentity(o1.affine)) {
            UvExactComponentOrigin combined = o0;
            if (uvAffineTryAttachCross(combined, o1)) {
              return combined;
            }
          }
          UvExactComponentOrigin merged = uvMergeOrigins(o0, o1);
          if (merged.valid) {
            uvAffineMarkInexact(merged.affine); // sum of two interpolant terms
          }
          return merged;
        }
        if (o0.valid || o1.valid) {
          // origin plus an unknown term: base UV plus per-pixel detail, offset unknown
          UvExactComponentOrigin origin = o0.valid ? o0 : o1;
          origin.affine.offset.inexact = true;
          return origin;
        }
        return UvExactComponentOrigin{};
      }

      case DxsoOpcode::Mad: {
        const UvExactComponentOrigin o0 = readOrigin(ctx.src[0], c);
        const UvExactComponentOrigin o1 = readOrigin(ctx.src[1], c);
        const UvExactComponentOrigin o2 = readOrigin(ctx.src[2], c);
        const UvConstComponentRef k0 = readConst(ctx.src[0], c);
        const UvConstComponentRef k1 = readConst(ctx.src[1], c);
        const UvConstComponentRef k2 = readConst(ctx.src[2], c);

        UvExactComponentOrigin product;
        if (o0.valid && !o1.valid && k1.valid) {
          product = o0;
          uvAffineMulConstRef(product.affine, k1);
        } else if (o1.valid && !o0.valid && k0.valid) {
          product = o1;
          uvAffineMulConstRef(product.affine, k0);
        } else if (o0.valid && o1.valid) {
          product = uvMergeOrigins(o0, o1);
          if (product.valid) {
            uvAffineMarkInexact(product.affine);
          }
        } else if (o0.valid || o1.valid) {
          product = o0.valid ? o0 : o1;
          product.affine.scale.inexact = true;
          if (uvAffineTermPresent(product.affine.offset)) {
            product.affine.offset.inexact = true;
          }
        }

        if (product.valid) {
          if (!o2.valid && k2.valid) {
            uvAffineAddConstRef(product.affine, k2, 1.0f);
            return product;
          }
          if (o2.valid) {
            if (uvAffineScaleNonIdentity(product.affine) || uvAffineScaleNonIdentity(o2.affine)) {
              UvExactComponentOrigin combined = product;
              if (uvAffineTryAttachCross(combined, o2)) {
                return combined;
              }
            }
            UvExactComponentOrigin merged = uvMergeOrigins(product, o2);
            if (merged.valid) {
              uvAffineMarkInexact(merged.affine);
            }
            return merged;
          }
          product.affine.offset.inexact = true; // unknown additive term
          return product;
        }

        // const * const + origin (e.g. precomputed pan offsets added to a UV)
        if (o2.valid && !o0.valid && !o1.valid) {
          UvExactComponentOrigin origin = o2;
          if (k0.valid && k1.valid) {
            if (k0.isImmediate && k1.isImmediate) {
              uvAffineAddImmediate(origin.affine, k0.immediate * k1.immediate);
            } else if (k0.isImmediate) {
              uvAffineAddConstant(origin.affine, k1.constReg, k1.constComp, k1.factor * k0.immediate);
            } else if (k1.isImmediate) {
              uvAffineAddConstant(origin.affine, k0.constReg, k0.constComp, k0.factor * k1.immediate);
            } else {
              origin.affine.offset.inexact = true;
            }
          } else {
            origin.affine.offset.inexact = true;
          }
          return origin;
        }
        return UvExactComponentOrigin{};
      }

      case DxsoOpcode::Min:
      case DxsoOpcode::Max: {
        const UvExactComponentOrigin o0 = readOrigin(ctx.src[0], c);
        const UvExactComponentOrigin o1 = readOrigin(ctx.src[1], c);
        if (o0.valid && o1.valid) {
          return uvMergeOrigins(o0, o1); // min(x,x) stays exact; differing affines go inexact
        }
        if (o0.valid || o1.valid) {
          // clamping an affine UV against a provable constant bound (atlas tile
          // anti-bleed clamping: min/max against the tile edge) is identity for
          // in-range coordinates - keep the affine exact. A bound that is arbitrary
          // math keeps the origin but the affine is unknowable.
          UvExactComponentOrigin origin = o0.valid ? o0 : o1;
          const UvConstComponentRef bound = readConst(o0.valid ? ctx.src[1] : ctx.src[0], c);
          if (!bound.valid) {
            uvAffineMarkInexact(origin.affine);
          }
          return origin;
        }
        return UvExactComponentOrigin{};
      }

      case DxsoOpcode::Lrp:
        // lerp(src2, src1, src0): src0 is the blend factor, not a value contributor
        return uvMergeOrigins(readOrigin(ctx.src[1], c), readOrigin(ctx.src[2], c));

      case DxsoOpcode::Cmp:
      case DxsoOpcode::Cnd: {
        // per-component select between src1/src2; src0 is the condition
        const UvExactComponentOrigin o1 = readOrigin(ctx.src[1], c);
        const UvExactComponentOrigin o2 = readOrigin(ctx.src[2], c);
        if (o1.valid != o2.valid) {
          // selecting between an affine UV and a provable constant is the
          // conditional-move form of tile clamping (fxc emits cmp for the lower
          // bound of clamp()): keep the affine branch exact
          UvExactComponentOrigin origin = o1.valid ? o1 : o2;
          const UvConstComponentRef bound = readConst(o1.valid ? ctx.src[2] : ctx.src[1], c);
          if (!bound.valid) {
            uvAffineMarkInexact(origin.affine);
          }
          return origin;
        }
        return uvMergeOrigins(o1, o2);
      }

      // single-source value-mangling math: register provenance survives, affine does not
      case DxsoOpcode::Rcp:
      case DxsoOpcode::Rsq:
      case DxsoOpcode::Exp:
      case DxsoOpcode::Log: {
        UvExactComponentOrigin origin = readOrigin(ctx.src[0], c);
        if (origin.valid) {
          uvAffineMarkInexact(origin.affine);
        }
        return origin;
      }

      // Dot products read several source components, so the origins are unioned to keep the
      // packed UV half being read. A dp2add of a UV pair against a constant row stays exact as
      // scale + cross + offset; other dots keep the origin with an inexact affine.
      case DxsoOpcode::Dp2Add: {
        const UvExactComponentOrigin o0x = readOrigin(ctx.src[0], 0);
        const UvExactComponentOrigin o0y = readOrigin(ctx.src[0], 1);
        const UvExactComponentOrigin o1x = readOrigin(ctx.src[1], 0);
        const UvExactComponentOrigin o1y = readOrigin(ctx.src[1], 1);
        const UvConstComponentRef k0x = readConst(ctx.src[0], 0);
        const UvConstComponentRef k0y = readConst(ctx.src[0], 1);
        const UvConstComponentRef k1x = readConst(ctx.src[1], 0);
        const UvConstComponentRef k1y = readConst(ctx.src[1], 1);
        const UvConstComponentRef k2 = readConst(ctx.src[2], c);
        const UvExactComponentOrigin o2 = readOrigin(ctx.src[2], c);

        // dest = src0.xy · src1.xy + src2, with the UV pair in either operand and the matrix
        // row constants in the other. Folds into scale + cross + offset by multiplying each
        // UV component by its row coefficient and attaching the sibling as `cross`. src2 is
        // the translation (Center, or Origin - R·Origin); an absent addend contributes 0.
        auto tryDp2AddRow = [&](const UvExactComponentOrigin& uvx,
                                const UvExactComponentOrigin& uvy,
                                const UvConstComponentRef& kx,
                                const UvConstComponentRef& ky,
                                const UvConstComponentRef& addend) -> UvExactComponentOrigin {
          if (!kx.valid || !ky.valid) {
            return UvExactComponentOrigin{};
          }
          UvExactComponentOrigin primary = uvx;
          uvAffineMulConstRef(primary.affine, kx);
          UvExactComponentOrigin sibling = uvy;
          uvAffineMulConstRef(sibling.affine, ky);
          if (!uvAffineTryAttachCross(primary, sibling)) {
            return UvExactComponentOrigin{};
          }
          if (addend.valid) {
            uvAffineAddConstRef(primary.affine, addend, 1.0f);
          }
          if (!uvComponentAffineExact(primary.affine)) {
            return UvExactComponentOrigin{};
          }
          return primary;
        };

        if (!o1x.valid && !o1y.valid && !o2.valid) {
          const UvExactComponentOrigin row = tryDp2AddRow(o0x, o0y, k1x, k1y, k2);
          if (row.valid) {
            return row;
          }
        }
        if (!o0x.valid && !o0y.valid && !o2.valid) {
          const UvExactComponentOrigin row = tryDp2AddRow(o1x, o1y, k0x, k0y, k2);
          if (row.valid) {
            return row;
          }
        }

        UvExactComponentOrigin merged = uvMergeOrigins(
          readOriginUnion(ctx.src[0], 2u),
          readOriginUnion(ctx.src[1], 2u));
        merged = uvMergeOrigins(merged, o2);
        if (merged.valid) {
          uvAffineMarkInexact(merged.affine);
        }
        return merged;
      }

      case DxsoOpcode::Dp3:
      case DxsoOpcode::Dp4: {
        const uint32_t count = ctx.instruction.opcode == DxsoOpcode::Dp3 ? 3u : 4u;
        UvExactComponentOrigin merged = uvMergeOrigins(
          readOriginUnion(ctx.src[0], count),
          readOriginUnion(ctx.src[1], count));
        if (merged.valid) {
          uvAffineMarkInexact(merged.affine);
        }
        return merged;
      }

      case DxsoOpcode::M3x2:
      case DxsoOpcode::M3x3:
      case DxsoOpcode::M3x4:
      case DxsoOpcode::M4x3:
      case DxsoOpcode::M4x4: {
        const uint32_t count =
          (ctx.instruction.opcode == DxsoOpcode::M4x3 || ctx.instruction.opcode == DxsoOpcode::M4x4) ? 4u : 3u;
        // src1 is a matrix-row register sequence (typically constants; readOrigin yields
        // invalid for those, so the union is driven by the vector operand)
        UvExactComponentOrigin merged = uvMergeOrigins(
          readOriginUnion(ctx.src[0], count),
          readOriginUnion(ctx.src[1], count));
        if (merged.valid) {
          uvAffineMarkInexact(merged.affine);
        }
        return merged;
      }

      default:
        // anything else (transcendental math, samples, ...) destroys provenance
        return UvExactComponentOrigin{};
      }
    }

    std::array<int8_t, 2 * DxsoMaxInterfaceRegs> m_inputRegToTexcoord;
    bool m_texcoordRegsAreInterpolants = false;
    bool m_originIsSemanticIndex = false;
    uint32_t m_trackedOutputReg = std::numeric_limits<uint32_t>::max();

    // Temp components holding a constant expression, which fxc hoists before applying it to a UV
    // (a `def` tiling literal times a uniform scalar). Only an immediate, or one constant
    // component times a static factor, is representable.
    UvConstComponentRef traceConstExpr(const DxsoInstructionContext& ctx, const uint32_t c) const {
      auto mulRefs = [](const UvConstComponentRef& a, const UvConstComponentRef& b) -> UvConstComponentRef {
        UvConstComponentRef result;
        if (!a.valid || !b.valid) {
          return result;
        }
        if (a.isImmediate && b.isImmediate) {
          result.valid = true;
          result.isImmediate = true;
          result.immediate = a.immediate * b.immediate;
          return result;
        }
        if (a.isImmediate != b.isImmediate) {
          const UvConstComponentRef& constPart = a.isImmediate ? b : a;
          const float immediatePart = a.isImmediate ? a.immediate : b.immediate;
          result = constPart;
          result.factor *= immediatePart;
          return result;
        }
        return result; // product of two draw-time constants: not representable
      };

      switch (ctx.instruction.opcode) {
      case DxsoOpcode::Mov:
        return readConst(ctx.src[0], c);

      case DxsoOpcode::Mul:
        return mulRefs(readConst(ctx.src[0], c), readConst(ctx.src[1], c));

      case DxsoOpcode::Add:
      case DxsoOpcode::Sub: {
        const UvConstComponentRef k0 = readConst(ctx.src[0], c);
        const UvConstComponentRef k1 = readConst(ctx.src[1], c);
        const float sign = ctx.instruction.opcode == DxsoOpcode::Sub ? -1.0f : 1.0f;
        UvConstComponentRef result;
        if (k0.valid && k1.valid && k0.isImmediate && k1.isImmediate) {
          result.valid = true;
          result.isImmediate = true;
          result.immediate = k0.immediate + sign * k1.immediate;
        }
        // immediate + draw-time constant sums exceed the single-part model
        return result;
      }

      case DxsoOpcode::Mad: {
        const UvConstComponentRef product = mulRefs(readConst(ctx.src[0], c), readConst(ctx.src[1], c));
        const UvConstComponentRef k2 = readConst(ctx.src[2], c);
        UvConstComponentRef result;
        if (product.valid && k2.valid) {
          if (product.isImmediate && k2.isImmediate) {
            result.valid = true;
            result.isImmediate = true;
            result.immediate = product.immediate + k2.immediate;
          } else if (!product.isImmediate && k2.isImmediate && k2.immediate == 0.0f) {
            result = product;
          } else if (product.isImmediate && product.immediate == 0.0f && !k2.isImmediate) {
            result = k2;
          }
        }
        return result;
      }

      case DxsoOpcode::Min:
      case DxsoOpcode::Max: {
        const UvConstComponentRef k0 = readConst(ctx.src[0], c);
        const UvConstComponentRef k1 = readConst(ctx.src[1], c);
        UvConstComponentRef result;
        if (k0.valid && k1.valid && k0.isImmediate && k1.isImmediate) {
          result.valid = true;
          result.isImmediate = true;
          result.immediate = ctx.instruction.opcode == DxsoOpcode::Min
            ? std::min(k0.immediate, k1.immediate)
            : std::max(k0.immediate, k1.immediate);
        }
        return result;
      }

      // unary math on compile-time immediates stays computable
      case DxsoOpcode::Frc:
      case DxsoOpcode::Abs:
      case DxsoOpcode::Rcp:
      case DxsoOpcode::Rsq:
      case DxsoOpcode::Exp:
      case DxsoOpcode::Log: {
        const UvConstComponentRef k0 = readConst(ctx.src[0], c);
        UvConstComponentRef result;
        if (k0.valid && k0.isImmediate && std::isfinite(k0.immediate)) {
          float value = k0.immediate;
          switch (ctx.instruction.opcode) {
          case DxsoOpcode::Frc: value = value - std::floor(value); break;
          case DxsoOpcode::Abs: value = std::abs(value); break;
          case DxsoOpcode::Rcp:
            if (value == 0.0f) {
              return result;
            }
            value = 1.0f / value;
            break;
          case DxsoOpcode::Rsq:
            if (value <= 0.0f) {
              return result;
            }
            value = 1.0f / std::sqrt(value);
            break;
          case DxsoOpcode::Exp: value = std::exp2(value); break;
          case DxsoOpcode::Log:
            if (value == 0.0f) {
              return result;
            }
            value = std::log2(std::abs(value));
            break;
          default: break;
          }
          if (std::isfinite(value)) {
            result.valid = true;
            result.isImmediate = true;
            result.immediate = value;
          }
        }
        return result;
      }

      default:
        return UvConstComponentRef{};
      }
    }

    std::array<std::array<UvExactComponentOrigin, 4>, 64> m_tempOrigins = {};
    std::array<std::array<UvConstComponentRef, 4>, 64> m_tempConstExprs = {};
    std::array<UvExactComponentOrigin, 4> m_outputOrigins = {};

    std::array<uint8_t, 256> m_defConstValid = {};
    std::array<Vector4, 256> m_defConsts = {};
  };

  // Resolves the exact coordinate origin for every sampler of a pixel shader in one pass.
  void analyzePsSamplerUvOrigins(const DxsoShaderView& pixelShader,
                                 std::array<PsSamplerUvOrigin, kDxsoMaxPsSamplers>& outOrigins,
                                 std::vector<std::string>* trace) {
    for (auto& origin : outOrigins) {
      origin = PsSamplerUvOrigin{};
    }

    if (pixelShader.tokens == nullptr || pixelShader.info == nullptr || pixelShader.isgn == nullptr) {
      return;
    }

    const auto& info = *pixelShader.info;
    if (info.type() != DxsoProgramTypes::PixelShader) {
      return;
    }

    const bool traceInstructions = trace != nullptr;
    if (traceInstructions) {
      trace->push_back(str::format(
        "[RTX-UV-TRACE] begin version=", info.majorVersion(), ".", info.minorVersion(),
        " tokens=", pixelShader.tokenCount));
    }

    std::array<int8_t, 2 * DxsoMaxInterfaceRegs> inputRegToTexcoord = {};
    inputRegToTexcoord.fill(-1);
    {
      const auto& isgn = *pixelShader.isgn;
      for (uint32_t i = 0; i < isgn.elemCount; i++) {
        const auto& e = isgn.elems[i];
        if (e.semantic.usage == DxsoUsage::Texcoord && e.regNumber < inputRegToTexcoord.size()) {
          inputRegToTexcoord[e.regNumber] = int8_t(e.semantic.usageIndex & 0b111);
        }
      }
    }

    UvDataflowTracer tracer(inputRegToTexcoord, true /*texcoordRegsAreInterpolants*/, true /*originIsSemanticIndex*/);

    struct SiteRecord {
      bool valid = false;
      uint8_t semanticIndex = 0;
      uint8_t compU = 0;
      uint8_t compV = 1;
      bool affineExact = false;
      UvComponentAffine affineU;
      UvComponentAffine affineV;
    };

    const uint32_t* tokens = pixelShader.tokens;
    DxsoDecodeContext decoder(info);
    DxsoCodeIter iter(tokens + 1);

    while (decoder.decodeInstruction(iter)) {
      const DxsoInstructionContext& ctx = decoder.getInstructionContext();
      const DxsoOpcode op = ctx.instruction.opcode;

      // identify sample sites before the write invalidates the destination temp
      uint32_t sampledSampler = ~0u;
      const DxsoRegister* coordReg = nullptr;
      bool directTexcoordSite = false;
      bool projected = false;
      bool untraceableSite = false;

      switch (op) {
      case DxsoOpcode::Tex:
        if (info.majorVersion() >= 2) {
          sampledSampler = ctx.src[1].id.num;
          coordReg = &ctx.src[0];
          projected = ctx.instruction.specificData.texld == DxsoTexLdMode::Project;
        } else if (info.majorVersion() == 1 && info.minorVersion() == 4) {
          sampledSampler = ctx.dst.id.num;
          coordReg = &ctx.src[0];
        } else {
          // ps_1_0..1_3: `tex t#` samples sampler # at texcoord # directly
          sampledSampler = ctx.dst.id.num;
          directTexcoordSite = true;
        }
        break;
      case DxsoOpcode::TexLdd:
      case DxsoOpcode::TexLdl:
        sampledSampler = ctx.src[1].id.num;
        coordReg = &ctx.src[0];
        break;
      case DxsoOpcode::TexBem:
      case DxsoOpcode::TexBemL:
      case DxsoOpcode::TexReg2Ar:
      case DxsoOpcode::TexReg2Gb:
      case DxsoOpcode::TexReg2Rgb:
      case DxsoOpcode::TexM3x2Tex:
      case DxsoOpcode::TexM3x3Tex:
      case DxsoOpcode::TexDp3Tex:
      case DxsoOpcode::TexM3x3Spec:
      case DxsoOpcode::TexM3x3VSpec:
        sampledSampler = ctx.dst.id.num;
        untraceableSite = true;
        break;
      default:
        break;
      }

      if (sampledSampler < kDxsoMaxPsSamplers &&
          (coordReg != nullptr || directTexcoordSite || untraceableSite)) {
        PsSamplerUvOrigin& agg = outOrigins[sampledSampler];

        SiteRecord site;
        if (directTexcoordSite) {
          site.valid = true;
          site.semanticIndex = tracer.mapRegToSemantic(ctx.dst.id.num, true);
          site.compU = 0;
          site.compV = 1;
          site.affineExact = true;
        } else if (coordReg != nullptr) {
          const UvExactComponentOrigin uOrigin = tracer.readOrigin(*coordReg, 0);
          const UvExactComponentOrigin vOrigin = tracer.readOrigin(*coordReg, 1);
          if (uOrigin.valid && vOrigin.valid && uOrigin.reg == vOrigin.reg) {
            site.valid = true;
            site.semanticIndex = uOrigin.reg;
            const bool cleanComponents =
              uvIsSingleComponentMask(uOrigin.componentMask) &&
              uvIsSingleComponentMask(vOrigin.componentMask);
            if (cleanComponents) {
              site.compU = uOrigin.component;
              site.compV = vOrigin.component;
            } else {
              // component-mixing math (rotators, UV matrices): the exact source pair is not
              // recoverable, so attribute the packed interpolant half being consumed
              const uint8_t unionMask = uint8_t(uOrigin.componentMask | vOrigin.componentMask);
              const bool onlySecondaryHalf =
                (unionMask & 0b0011u) == 0u && (unionMask & 0b1100u) != 0u;
              site.compU = onlySecondaryHalf ? 3u : 0u;
              site.compV = onlySecondaryHalf ? 2u : 1u;
            }
            site.affineU = uOrigin.affine;
            site.affineV = vOrigin.affine;
            site.affineExact =
              !projected &&
              uvComponentAffineExact(uOrigin.affine) &&
              uvComponentAffineExact(vOrigin.affine) &&
              (cleanComponents ||
               uvAffineIsExactLinearPair(uOrigin.affine, vOrigin.affine, site.compU, site.compV));
          }
        }

        if (traceInstructions) {
          std::string siteDetail;
          if (site.valid) {
            siteDetail = str::format(
              " sem=", uint32_t(site.semanticIndex),
              " comps=(", uint32_t(site.compU), ",", uint32_t(site.compV), ")",
              " exact=", site.affineExact ? 1 : 0,
              " U=[", formatUvComponentAffine(site.affineU), "]",
              " V=[", formatUvComponentAffine(site.affineV), "]");
          }
          trace->push_back(str::format(
            "[RTX-UV-TRACE] #", ctx.instructionIdx, " SAMPLE s", sampledSampler,
            coordReg != nullptr ? str::format(" coord=", formatDxsoSrcRegister(*coordReg)).c_str() : "",
            directTexcoordSite ? " direct-texcoord" : "",
            untraceableSite ? " untraceable-legacy-op" : "",
            projected ? " projected" : "",
            " => valid=", site.valid ? 1 : 0,
            siteDetail));
        }

        if (site.valid) {
          if (agg.validSiteCount < std::numeric_limits<uint16_t>::max()) {
            agg.validSiteCount++;
          }

          if (!agg.originValid) {
            // first valid site seeds the aggregate; later disagreements clear sitesAgree
            // and may supersede the affine via the static-tiling frequency preference
            agg.originValid = true;
            agg.semanticIndex = site.semanticIndex;
            agg.compU = site.compU;
            agg.compV = site.compV;
            agg.affineU = site.affineU;
            agg.affineV = site.affineV;
            agg.affineExact = site.affineExact;
          } else {
            const bool sameOriginPair =
              agg.semanticIndex == site.semanticIndex &&
              agg.compU == site.compU &&
              agg.compV == site.compV;
            const bool matches =
              sameOriginPair &&
              agg.affineExact == site.affineExact &&
              uvComponentAffinesEqual(agg.affineU, site.affineU) &&
              uvComponentAffinesEqual(agg.affineV, site.affineV);
            if (!matches) {
              agg.sitesAgree = false;

              // UE3's distance-fade anti-tiling lerps two literal tilings by a depth fade that is
              // 0 near the camera, so the highest-frequency site is the surface's mapping.
              float aggMagnitude = 0.0f;
              float siteMagnitude = 0.0f;
              if (sameOriginPair &&
                  agg.affineExact && site.affineExact &&
                  uvSiteStaticTilingMagnitude(agg.affineU, agg.affineV, aggMagnitude) &&
                  uvSiteStaticTilingMagnitude(site.affineU, site.affineV, siteMagnitude)) {
                agg.preferredHighestFrequencySite = true;
                if (siteMagnitude > aggMagnitude) {
                  agg.affineU = site.affineU;
                  agg.affineV = site.affineV;
                }
              }
            }
          }
        } else {
          if (agg.invalidSiteCount < std::numeric_limits<uint16_t>::max()) {
            agg.invalidSiteCount++;
          }
          if (agg.originValid) {
            agg.sitesAgree = false;
          }
        }
      }

      tracer.processInstruction(ctx);

      if (traceInstructions) {
        std::ostringstream opName;
        opName << op;
        std::string line = str::format("[RTX-UV-TRACE] #", ctx.instructionIdx, " ", opName.str());

        const bool isFlowOrMeta =
          op == DxsoOpcode::Nop || op == DxsoOpcode::Comment || op == DxsoOpcode::End ||
          op == DxsoOpcode::Phase || op == DxsoOpcode::If || op == DxsoOpcode::Ifc ||
          op == DxsoOpcode::Else || op == DxsoOpcode::EndIf || op == DxsoOpcode::Loop ||
          op == DxsoOpcode::EndLoop || op == DxsoOpcode::Rep || op == DxsoOpcode::EndRep ||
          op == DxsoOpcode::Break || op == DxsoOpcode::BreakC || op == DxsoOpcode::BreakP ||
          op == DxsoOpcode::Call || op == DxsoOpcode::CallNz || op == DxsoOpcode::Label ||
          op == DxsoOpcode::Ret;

        if (op == DxsoOpcode::Def) {
          line += str::format(" c", ctx.dst.id.num,
                              " = (", ctx.def.float32[0], ",", ctx.def.float32[1], ",",
                              ctx.def.float32[2], ",", ctx.def.float32[3], ")");
        } else if (op == DxsoOpcode::Dcl || op == DxsoOpcode::DefI || op == DxsoOpcode::DefB) {
          line += str::format(" ", formatDxsoDstRegister(ctx.dst));
        } else if (!isFlowOrMeta) {
          line += str::format(" ", formatDxsoDstRegister(ctx.dst));

          const uint32_t opcodeLength = DxsoGetDefaultOpcodeLength(op);
          const uint32_t sourceCount = (opcodeLength != InvalidOpcodeLength && opcodeLength > 0u)
            ? std::min<uint32_t>(opcodeLength - 1u, uint32_t(ctx.src.size()))
            : 0u;
          for (uint32_t s = 0; s < sourceCount; s++) {
            line += str::format(", ", formatDxsoSrcRegister(ctx.src[s]));
          }
          if (ctx.instruction.predicated) {
            line += " [predicated]";
          }

          // post-instruction provenance verdict for every written temp component
          const bool dstIsTempForTrace =
            ctx.dst.id.type == DxsoRegisterType::Temp ||
            ctx.dst.id.type == DxsoRegisterType::TempFloat16;
          if (dstIsTempForTrace) {
            line += " =>";
            for (uint32_t c = 0; c < 4u; c++) {
              if (!ctx.dst.mask[c]) {
                continue;
              }
              const UvExactComponentOrigin& origin = tracer.tempOrigin(ctx.dst.id.num, c);
              const UvConstComponentRef& constExpr = tracer.tempConstExpr(ctx.dst.id.num, c);
              line += str::format(" ", "xyzw"[c]);
              if (origin.valid) {
                line += str::format("{org=tc", uint32_t(origin.reg),
                                    " srcmask=0x", std::hex, uint32_t(origin.componentMask), std::dec,
                                    " aff=", formatUvComponentAffine(origin.affine), "}");
              } else if (constExpr.valid) {
                line += str::format("{cexpr=", formatUvConstExpr(constExpr), "}");
              } else {
                line += "{-}";
              }
            }
          }
        }

        trace->push_back(line);
      }
    }

    if (traceInstructions) {
      for (uint32_t s = 0; s < kDxsoMaxPsSamplers; s++) {
        const PsSamplerUvOrigin& origin = outOrigins[s];
        if (origin.validSiteCount == 0 && origin.invalidSiteCount == 0) {
          continue;
        }
        trace->push_back(str::format(
          "[RTX-UV-TRACE] result s", s,
          " origin=", origin.originValid ? 1 : 0,
          " sem=", uint32_t(origin.semanticIndex),
          " comps=(", uint32_t(origin.compU), ",", uint32_t(origin.compV), ")",
          " sites=", origin.validSiteCount, "/", origin.invalidSiteCount,
          " agree=", origin.sitesAgree ? 1 : 0,
          " preferHF=", origin.preferredHighestFrequencySite ? 1 : 0,
          " exact=", origin.affineExact ? 1 : 0,
          " U=[", formatUvComponentAffine(origin.affineU), "]",
          " V=[", formatUvComponentAffine(origin.affineV), "]"));
      }
      trace->push_back("[RTX-UV-TRACE] end");
    }
  }

  // Traces a VS output TEXCOORD interpolant (components compU/compV) back to the IA:
  // proves which IA texcoord set feeds it and classifies the math along the way.
  Ue3VsTexcoordTraceResult traceVsOutputTexcoordToInputUsageIndex(
    const DxsoShaderView& vertexShader,
    const uint32_t outputRegNumber,
    const uint8_t compU,
    const uint8_t compV) {
    Ue3VsTexcoordTraceResult result;

    if (vertexShader.tokens == nullptr ||
        vertexShader.info == nullptr ||
        vertexShader.isgn == nullptr ||
        outputRegNumber >= DxsoMaxInterfaceRegs ||
        compU >= 4u ||
        compV >= 4u) {
      return result;
    }

    const auto& info = *vertexShader.info;
    if (info.type() != DxsoProgramTypes::VertexShader) {
      return result;
    }

    std::array<int8_t, 2 * DxsoMaxInterfaceRegs> inputRegToTexcoord = {};
    inputRegToTexcoord.fill(-1);
    {
      const auto& isgn = *vertexShader.isgn;
      for (uint32_t i = 0; i < isgn.elemCount; i++) {
        const auto& e = isgn.elems[i];
        if (e.semantic.usage == DxsoUsage::Texcoord && e.regNumber < inputRegToTexcoord.size()) {
          inputRegToTexcoord[e.regNumber] = int8_t(e.semantic.usageIndex & 0b111);
        }
      }
    }

    UvDataflowTracer tracer(inputRegToTexcoord, false /*texcoordRegsAreInterpolants*/, false /*originIsSemanticIndex*/);
    tracer.setTrackedOutputRegister(outputRegNumber);

    const uint32_t* tokens = vertexShader.tokens;
    DxsoDecodeContext decoder(info);
    DxsoCodeIter iter(tokens + 1);

    while (decoder.decodeInstruction(iter)) {
      tracer.processInstruction(decoder.getInstructionContext());
    }

    const UvExactComponentOrigin& uOrigin = tracer.outputOrigin(compU);
    const UvExactComponentOrigin& vOrigin = tracer.outputOrigin(compV);

    if (!uOrigin.valid || !vOrigin.valid || uOrigin.reg != vOrigin.reg) {
      return result;
    }

    if (uOrigin.reg >= inputRegToTexcoord.size() || inputRegToTexcoord[uOrigin.reg] < 0) {
      return result;
    }

    result.inputReg = uOrigin.reg;
    result.iaTexcoordIndex = uint8_t(inputRegToTexcoord[uOrigin.reg]);
    result.affineU = uOrigin.affine;
    result.affineV = vOrigin.affine;

    // IA texcoord sets are float2 streams: an exact IA representation requires the
    // interpolant components to be exactly U <- .x and V <- .y of a single set
    const bool pureComponentMapping =
      uvIsSingleComponentMask(uOrigin.componentMask) &&
      uvIsSingleComponentMask(vOrigin.componentMask) &&
      uOrigin.component == 0u && vOrigin.component == 1u;

    if (!pureComponentMapping) {
      result.kind = Ue3VsUvTraceKind::OriginOnly;
      return result;
    }

    const bool affinesExact =
      uvComponentAffineExact(uOrigin.affine) &&
      uvComponentAffineExact(vOrigin.affine);
    if (!affinesExact) {
      result.kind = Ue3VsUvTraceKind::OriginOnly;
      return result;
    }

    const bool affinesIdentity =
      uvComponentAffineIsIdentity(uOrigin.affine) &&
      uvComponentAffineIsIdentity(vOrigin.affine);
    result.kind = affinesIdentity ? Ue3VsUvTraceKind::PureMove : Ue3VsUvTraceKind::AffineConst;
    return result;
  }

}
