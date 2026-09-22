#pragma once

#include "d3d9_include.h"
#include "d3d9_state.h"

#include "d3d9_util.h"
#include "d3d9_buffer.h"

#include "d3d9_rtx.h"
#include "d3d9_device.h"

#include "../util/util_fastops.h"
#include "../util/util_math.h"
#include "d3d9_rtx_utils.h"
#include "d3d9_texture.h"
#include "../dxso/dxso_tables.h"
#include "../dxvk/rtx_render/rtx_ngx_passthrough.h"
#include "../dxvk/rtx_render/rtx_options.h"
#include "../dxvk/rtx_render/rtx_dlfg.h"
#include "../dxvk/rtx_render/rtx_camera.h"
// NV-DXVK start: draw disposition statistics
#include "../dxvk/rtx_render/rtx_gpu_pass_timer.h"
// NV-DXVK end
#include "../dxvk/imgui/dxvk_imgui.h"

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <shared_mutex>
#include <sstream>
#include <system_error>

// Toolhelp snapshots for cross-process game-settings driving.
#include <tlhelp32.h>

namespace dxvk {
  static const bool s_isDxvkResolutionEnvVarSet = (env::getEnvVar("DXVK_RESOLUTION_WIDTH") != "") || (env::getEnvVar("DXVK_RESOLUTION_HEIGHT") != "");
  
  // We only look at RT 0 currently.
  const uint32_t kRenderTargetIndex = 0;

  #define CATEGORIES_REQUIRE_DRAW_CALL_STATE  InstanceCategories::Sky, InstanceCategories::Terrain
  #define CATEGORIES_REQUIRE_GEOMETRY_COPY    InstanceCategories::Terrain, InstanceCategories::WorldUI

  namespace {
    // UE3 reserved D3D9 vertex shader registers (see UE3 engine Shaders/Common.usf and D3D9Drv/Src/D3D9Commands.cpp)
    constexpr uint32_t kUe3VsrViewProjMatrixRegister = 0; // c0..c3
    constexpr uint32_t kUe3VsrViewOriginRegister     = 4; // c4

    fast_unordered_set s_loggedNonPrimaryRtDescHashes;
    fast_unordered_set s_loggedSampledRtDescHashes;

    struct VDeclSignature {
      bool hasPosition = false;
      uint8_t positionType = 0;
      uint8_t positionStream = 0xFF;
      bool hasTangent = false;
      uint8_t tangentType = 0;
      uint8_t tangentStream = 0xFF;
      bool hasNormal = false;
      uint8_t normalType = 0;
      uint8_t normalStream = 0xFF;
      bool hasBinormal = false;
      uint8_t binormalType = 0;
      bool hasBlendWeight = false;
      uint8_t blendWeightType = 0;
      bool hasBlendIndices = false;
      bool hasColor0 = false;
      bool hasColor1 = false;
      uint8_t texcoordCount = 0;
      uint8_t maxTexcoordIndex = 0;
      uint8_t texcoordTypes[8] = {};
      bool hasTexcoord6 = false;
      uint8_t texcoord6Type = 0;
      bool hasTexcoord7 = false;
      uint8_t texcoord7Type = 0;
      uint8_t totalElements = 0;
    };

    static VDeclSignature buildVDeclSignature(const D3D9VertexElements& elements) {
      VDeclSignature sig;
      sig.totalElements = uint8_t(std::min<size_t>(elements.size(), 255u));
      for (const auto& e : elements) {
        switch (e.Usage) {
        case D3DDECLUSAGE_POSITION:
        case D3DDECLUSAGE_POSITIONT:
          sig.hasPosition = true;
          sig.positionType = e.Type;
          sig.positionStream = e.Stream;
          break;
        case D3DDECLUSAGE_TANGENT:
          sig.hasTangent = true;
          sig.tangentType = e.Type;
          sig.tangentStream = e.Stream;
          break;
        case D3DDECLUSAGE_NORMAL:
          sig.hasNormal = true;
          sig.normalType = e.Type;
          sig.normalStream = e.Stream;
          break;
        case D3DDECLUSAGE_BINORMAL:
          sig.hasBinormal = true;
          sig.binormalType = e.Type;
          break;
        case D3DDECLUSAGE_BLENDWEIGHT:
          sig.hasBlendWeight = true;
          sig.blendWeightType = e.Type;
          break;
        case D3DDECLUSAGE_BLENDINDICES:
          sig.hasBlendIndices = true;
          break;
        case D3DDECLUSAGE_COLOR:
          if (e.UsageIndex == 0) sig.hasColor0 = true;
          if (e.UsageIndex == 1) sig.hasColor1 = true;
          break;
        case D3DDECLUSAGE_TEXCOORD:
          if (e.UsageIndex < 8) {
            sig.texcoordCount++;
            if (e.UsageIndex > sig.maxTexcoordIndex) {
              sig.maxTexcoordIndex = e.UsageIndex;
            }
            sig.texcoordTypes[e.UsageIndex] = e.Type;
          }
          if (e.UsageIndex == 6) {
            sig.hasTexcoord6 = true;
            sig.texcoord6Type = e.Type;
          }
          if (e.UsageIndex == 7) {
            sig.hasTexcoord7 = true;
            sig.texcoord7Type = e.Type;
          }
          break;
        default:
          break;
        }
      }
      return sig;
    }

    static std::string toLowerAscii(const std::string& s) {
      std::string out;
      out.reserve(s.size());
      for (const char c : s)
        out.push_back(char(std::tolower(static_cast<unsigned char>(c))));
      return out;
    }

    static bool containsToken(const std::string& s, const char* token) {
      return s.find(token) != std::string::npos;
    }

    static uint32_t findVsTexcoordOutputRegister(const D3D9CommonShader* shader, uint32_t usageIndex) {
      if (shader == nullptr)
        return std::numeric_limits<uint32_t>::max();

      const auto& osgn = shader->GetOsgn();
      for (uint32_t i = 0; i < osgn.elemCount; i++) {
        const auto& decl = osgn.elems[i];
        if (decl.semantic.usage == DxsoUsage::Texcoord && decl.semantic.usageIndex == usageIndex)
          return decl.regNumber;
      }

      return std::numeric_limits<uint32_t>::max();
    }

    static bool isFloatConstantRegisterType(const DxsoRegisterType type) {
      return type == DxsoRegisterType::Const
          || type == DxsoRegisterType::Const2
          || type == DxsoRegisterType::Const3
          || type == DxsoRegisterType::Const4;
    }

    static int32_t getFloatConstantRegisterIndex(const DxsoRegister& r) {
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

    static bool decodeConstantModifierScale(const DxsoRegModifier modifier, float& outScale) {
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

    // ---------------------------------------------------------------------------
    // Exact UV dataflow tracing
    //
    // Shared per-component dataflow machinery used by:
    //  - the PS sampler coordinate-origin resolver (analyzePsSamplerUvOrigins)
    //  - the VS interpolant->IA texcoord trace (traceVsOutputTexcoordToInputUsageIndex)
    //
    // The tracer proves, per register component, which interpolant (PS) or IA texcoord
    // input (VS) component a value originates from, and accumulates the affine chain
    // (scale/cross/offset from draw-time constants or def'd immediates) applied along the
    // way - see UvComponentAffine for the term model. Math outside that model keeps the
    // origin but marks the affine inexact; genuinely ambiguous origins are invalidated.
    // ---------------------------------------------------------------------------

    static bool uvAffineTermPresent(const UvAffineTerm& t) {
      return t.immValid || t.constReg >= 0;
    }

    static bool uvAffineTermsEqual(const UvAffineTerm& a, const UvAffineTerm& b) {
      if (a.inexact != b.inexact || a.immValid != b.immValid ||
          a.constReg != b.constReg || a.constReg2 != b.constReg2)
        return false;
      if (a.immValid && a.imm != b.imm)
        return false;
      if (a.constReg >= 0 && (a.constComp != b.constComp || a.factor != b.factor))
        return false;
      if (a.constReg2 >= 0 && (a.constComp2 != b.constComp2 || a.factor2 != b.factor2))
        return false;
      return true;
    }

    static bool uvComponentAffinesEqual(const UvComponentAffine& a, const UvComponentAffine& b) {
      if (a.hasCross != b.hasCross)
        return false;
      if (!uvAffineTermsEqual(a.scale, b.scale) || !uvAffineTermsEqual(a.offset, b.offset))
        return false;
      if (!a.hasCross)
        return true;
      return a.scaleComponent == b.scaleComponent &&
             a.crossComponent == b.crossComponent &&
             uvAffineTermsEqual(a.cross, b.cross);
    }

    static bool uvComponentAffineExact(const UvComponentAffine& a) {
      if (a.scale.inexact || a.offset.inexact)
        return false;
      return !a.hasCross || !a.cross.inexact;
    }

    static bool uvComponentAffineIsIdentity(const UvComponentAffine& a) {
      if (a.hasCross)
        return false;
      return uvComponentAffineExact(a) &&
             (!uvAffineTermPresent(a.scale) ||
              (a.scale.immValid && a.scale.imm == 1.0f && a.scale.constReg < 0)) &&
             (!uvAffineTermPresent(a.offset) ||
              (a.offset.immValid && a.offset.imm == 0.0f && a.offset.constReg < 0));
    }

    // Compile-time scale magnitude of a static-tiling sample site. Identity scales do
    // not qualify: a plain-uv base layer blended with a tiled detail layer is not the
    // dual-tiling idiom. Neither do panner offsets (constant registers): scrolling
    // layers are simultaneously visible, so no site is more authoritative than another.
    static bool uvSiteStaticTilingMagnitude(const UvComponentAffine& affU,
                                            const UvComponentAffine& affV,
                                            float& outMagnitude) {
      auto explicitImmediateScale = [](const UvAffineTerm& term, float& outValue) {
        if (term.inexact || !term.immValid || term.constReg >= 0)
          return false;
        outValue = term.imm;
        return true;
      };
      auto compileTimeOffset = [](const UvAffineTerm& term) {
        return !term.inexact && term.constReg < 0;
      };

      if (affU.hasCross || affV.hasCross)
        return false;

      if (!compileTimeOffset(affU.offset) || !compileTimeOffset(affV.offset))
        return false;

      float scaleU = 1.0f;
      float scaleV = 1.0f;
      if (!explicitImmediateScale(affU.scale, scaleU) || !explicitImmediateScale(affV.scale, scaleV))
        return false;

      outMagnitude = std::abs(scaleU * scaleV);
      return std::isfinite(outMagnitude) && outMagnitude > 0.0f;
    }

    static void uvAffineMarkInexact(UvComponentAffine& a) {
      a.scale.inexact = true;
      a.offset.inexact = true;
      if (a.hasCross)
        a.cross.inexact = true;
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
      if (value == 0.0f)
        return;
      t.immValid = true;
      t.imm += value;
    }

    static void uvAffineTermAddConstant(UvAffineTerm& t, const int16_t reg, const uint8_t comp, const float factor) {
      if (t.immValid && t.imm == 0.0f)
        t.immValid = false;

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
      if (a.hasCross)
        uvAffineTermScaleMulImmediate(a.cross, value);
    }

    // value' = value * consts[reg][comp] * factor
    static void uvAffineMulConstant(UvComponentAffine& a, const int16_t reg, const uint8_t comp, const float factor) {
      uvAffineTermScaleMulConstant(a.scale, reg, comp, factor);
      uvAffineTermOffsetMulConstant(a.offset, reg, comp, factor);
      if (a.hasCross)
        uvAffineTermScaleMulConstant(a.cross, reg, comp, factor);
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
      if (src.immValid)
        uvAffineTermAddImmediate(dest, src.imm);
      if (src.constReg >= 0)
        uvAffineTermAddConstant(dest, src.constReg, src.constComp, src.factor);
      if (src.constReg2 >= 0)
        uvAffineTermAddConstant(dest, src.constReg2, src.constComp2, src.factor2);
    }

    // Fold sibling into dest as the cross coefficient: dest = dest + sibling, with dest.scale
    // multiplying dest.component and dest.cross multiplying sibling.component. Fails (and
    // leaves dest unchanged) when the result would leave the scale/cross/offset model.
    static bool uvAffineTryAttachCross(UvExactComponentOrigin& dest, const UvExactComponentOrigin& sibling) {
      if (!uvOriginSameRegister(dest, sibling))
        return false;
      if (dest.affine.hasCross || sibling.affine.hasCross)
        return false;
      if (!uvComponentAffineExact(dest.affine) || !uvComponentAffineExact(sibling.affine))
        return false;
      if (!uvIsSingleComponentMask(dest.componentMask) || !uvIsSingleComponentMask(sibling.componentMask))
        return false;
      if (dest.component == sibling.component)
        return false;

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
      if (!uvComponentAffineExact(combined))
        return false;

      dest.componentMask = uint8_t(dest.componentMask | sibling.componentMask);
      dest.affine = combined;
      return true;
    }

    static bool uvAffineMixUsesPair(const UvComponentAffine& a, const uint8_t compU, const uint8_t compV) {
      if (!a.hasCross)
        return true;
      const auto isPair = [&](const uint8_t c) {
        return c == compU || c == compV;
      };
      return isPair(a.scaleComponent) && isPair(a.crossComponent);
    }

    static bool uvAffineIsExactLinearPair(const UvComponentAffine& u,
                                          const UvComponentAffine& v,
                                          const uint8_t compU,
                                          const uint8_t compV) {
      if (!u.hasCross && !v.hasCross)
        return false;
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
      if (!a.valid && !b.valid)
        return UvExactComponentOrigin{};

      if (a.valid != b.valid) {
        UvExactComponentOrigin merged = a.valid ? a : b;
        uvAffineMarkInexact(merged.affine);
        return merged;
      }

      if (a.reg != b.reg)
        return UvExactComponentOrigin{};

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
      if (ref.isImmediate)
        uvAffineMulImmediate(a, ref.immediate);
      else
        uvAffineMulConstant(a, ref.constReg, ref.constComp, ref.factor);
    }

    static void uvAffineAddConstRef(UvComponentAffine& a, const UvConstComponentRef& ref, const float sign) {
      if (ref.isImmediate)
        uvAffineAddImmediate(a, sign * ref.immediate);
      else
        uvAffineAddConstant(a, ref.constReg, ref.constComp, sign * ref.factor);
    }

    static void uvAffineTermCollectConstRegs(const UvAffineTerm& term,
                                             int32_t* regs,
                                             uint32_t& count,
                                             const uint32_t cap) {
      auto push = [&](const int16_t r) {
        if (r >= 0 && count < cap)
          regs[count++] = int32_t(r);
      };
      push(term.constReg);
      push(term.constReg2);
    }

    static void uvComponentAffineCollectConstRegs(const UvComponentAffine& a,
                                                  int32_t* regs,
                                                  uint32_t& count,
                                                  const uint32_t cap) {
      uvAffineTermCollectConstRegs(a.scale, regs, count, cap);
      uvAffineTermCollectConstRegs(a.offset, regs, count, cap);
      if (a.hasCross)
        uvAffineTermCollectConstRegs(a.cross, regs, count, cap);
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

    static std::string formatUvComponentAffine(const UvComponentAffine& affine) {
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

    static std::string formatDxsoSrcRegister(const DxsoRegister& r) {
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

    static std::string formatDxsoDstRegister(const DxsoRegister& r) {
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
        if (r.hasRelative)
          return ref;

        // temps holding tracked pure-constant expressions (fxc hoists constant
        // subexpressions into temps before applying them to UVs) read like constants
        if (r.id.type == DxsoRegisterType::Temp || r.id.type == DxsoRegisterType::TempFloat16) {
          if (r.id.num >= m_tempConstExprs.size())
            return ref;
          const uint8_t comp = uint8_t(r.swizzle[dstComponent & 0x3u] & 0x3u);
          const UvConstComponentRef& tracked = m_tempConstExprs[r.id.num][comp];
          if (!tracked.valid)
            return ref;
          float modifierScale = 1.0f;
          if (!decodeConstantModifierScale(r.modifier, modifierScale))
            return ref;
          ref = tracked;
          if (ref.isImmediate) {
            ref.immediate *= modifierScale;
          } else {
            ref.factor *= modifierScale;
          }
          return ref;
        }

        if (!isFloatConstantRegisterType(r.id.type))
          return ref;

        float modifierScale = 1.0f;
        if (!decodeConstantModifierScale(r.modifier, modifierScale))
          return ref;

        const int32_t reg = getFloatConstantRegisterIndex(r);
        if (reg < 0 || reg > int32_t(std::numeric_limits<int16_t>::max()))
          return ref;

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
          if (r.id.num < m_tempOrigins.size())
            origin = m_tempOrigins[r.id.num][srcComponent];
          break;
        default:
          break;
        }

        if (origin.valid && r.modifier != DxsoRegModifier::None) {
          if (!uvAffineApplySourceModifier(origin.affine, r.modifier))
            uvAffineMarkInexact(origin.affine);
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
          for (auto& tempComponents : m_tempOrigins)
            tempComponents = {};
          for (auto& tempConstExprComponents : m_tempConstExprs)
            tempConstExprComponents = {};
          for (auto& outputComponent : m_outputOrigins)
            outputComponent = UvExactComponentOrigin{};
          return;
        // note: a trailing `ret` in main must not clear already-written output origins
        case DxsoOpcode::Ret:
          for (auto& tempComponents : m_tempOrigins)
            tempComponents = {};
          for (auto& tempConstExprComponents : m_tempConstExprs)
            tempConstExprComponents = {};
          return;
        default:
          break;
        }

        // non-arithmetic opcodes whose dst token is not a value write
        if (op == DxsoOpcode::Dcl || op == DxsoOpcode::Def ||
            op == DxsoOpcode::DefI || op == DxsoOpcode::DefB ||
            op == DxsoOpcode::TexKill || op == DxsoOpcode::Comment)
          return;

        const bool dstIsTemp =
          ctx.dst.id.type == DxsoRegisterType::Temp ||
          ctx.dst.id.type == DxsoRegisterType::TempFloat16;
        const bool dstIsTrackedOutput =
          m_trackedOutputReg != std::numeric_limits<uint32_t>::max() &&
          ctx.dst.id.type == DxsoRegisterType::Output &&
          ctx.dst.id.num == m_trackedOutputReg;

        if (!dstIsTemp && !dstIsTrackedOutput)
          return;
        if (dstIsTemp && ctx.dst.id.num >= m_tempOrigins.size())
          return;

        for (uint32_t c = 0; c < 4u; c++) {
          if (!ctx.dst.mask[c])
            continue;

          UvExactComponentOrigin origin;
          UvConstComponentRef constExpr;
          // predicated writes are conditional - the resulting value origin is unknowable
          if (!ctx.instruction.predicated) {
            origin = traceComponent(ctx, c);
            if (dstIsTemp && !origin.valid)
              constExpr = traceConstExpr(ctx, c);
          }

          if (origin.valid) {
            if (ctx.dst.shift != 0)
              uvAffineMulImmediate(origin.affine, std::exp2(float(ctx.dst.shift)));
            if (ctx.dst.saturate)
              uvAffineMarkInexact(origin.affine);
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
        if (regNum < m_inputRegToTexcoord.size() && m_inputRegToTexcoord[regNum] >= 0)
          return uint8_t(m_inputRegToTexcoord[regNum]);
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
          if (origin.valid)
            uvAffineMarkInexact(origin.affine);
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
            if (merged.valid)
              uvAffineMarkInexact(merged.affine); // value * value is never affine
            return merged;
          }
          if (o0.valid || o1.valid) {
            // origin times an unknown factor: keep base provenance, scale unknown
            UvExactComponentOrigin origin = o0.valid ? o0 : o1;
            origin.affine.scale.inexact = true;
            if (uvAffineTermPresent(origin.affine.offset))
              origin.affine.offset.inexact = true;
            return origin;
          }
          return UvExactComponentOrigin{};
        }

        case DxsoOpcode::Add:
        case DxsoOpcode::Sub: {
          const bool isSub = ctx.instruction.opcode == DxsoOpcode::Sub;
          const UvExactComponentOrigin o0 = readOrigin(ctx.src[0], c);
          UvExactComponentOrigin o1 = readOrigin(ctx.src[1], c);
          if (o1.valid && isSub)
            uvAffineMulImmediate(o1.affine, -1.0f);
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
              if (uvAffineTryAttachCross(combined, o1))
                return combined;
            }
            UvExactComponentOrigin merged = uvMergeOrigins(o0, o1);
            if (merged.valid)
              uvAffineMarkInexact(merged.affine); // sum of two interpolant terms
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
            if (product.valid)
              uvAffineMarkInexact(product.affine);
          } else if (o0.valid || o1.valid) {
            product = o0.valid ? o0 : o1;
            product.affine.scale.inexact = true;
            if (uvAffineTermPresent(product.affine.offset))
              product.affine.offset.inexact = true;
          }

          if (product.valid) {
            if (!o2.valid && k2.valid) {
              uvAffineAddConstRef(product.affine, k2, 1.0f);
              return product;
            }
            if (o2.valid) {
              if (uvAffineScaleNonIdentity(product.affine) || uvAffineScaleNonIdentity(o2.affine)) {
                UvExactComponentOrigin combined = product;
                if (uvAffineTryAttachCross(combined, o2))
                  return combined;
              }
              UvExactComponentOrigin merged = uvMergeOrigins(product, o2);
              if (merged.valid)
                uvAffineMarkInexact(merged.affine);
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
          if (o0.valid && o1.valid)
            return uvMergeOrigins(o0, o1); // min(x,x) stays exact; differing affines go inexact
          if (o0.valid || o1.valid) {
            // clamping an affine UV against a provable constant bound (atlas tile
            // anti-bleed clamping: min/max against the tile edge) is identity for
            // in-range coordinates - keep the affine exact. A bound that is arbitrary
            // math keeps the origin but the affine is unknowable.
            UvExactComponentOrigin origin = o0.valid ? o0 : o1;
            const UvConstComponentRef bound = readConst(o0.valid ? ctx.src[1] : ctx.src[0], c);
            if (!bound.valid)
              uvAffineMarkInexact(origin.affine);
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
            if (!bound.valid)
              uvAffineMarkInexact(origin.affine);
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
          if (origin.valid)
            uvAffineMarkInexact(origin.affine);
          return origin;
        }

        // dot products consume multiple components of their sources: union the origins so
        // register provenance (and the packed-UV half being read) survives. UV rotators /
        // matrix transforms (dp2add of a UV pair against a constant row) stay exact as
        // scale+cross+offset; other dots (m3x2..m4x4, lighting) keep origin but mark
        // the affine inexact.
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
            if (!kx.valid || !ky.valid)
              return UvExactComponentOrigin{};
            UvExactComponentOrigin primary = uvx;
            uvAffineMulConstRef(primary.affine, kx);
            UvExactComponentOrigin sibling = uvy;
            uvAffineMulConstRef(sibling.affine, ky);
            if (!uvAffineTryAttachCross(primary, sibling))
              return UvExactComponentOrigin{};
            if (addend.valid)
              uvAffineAddConstRef(primary.affine, addend, 1.0f);
            if (!uvComponentAffineExact(primary.affine))
              return UvExactComponentOrigin{};
            return primary;
          };

          if (!o1x.valid && !o1y.valid && !o2.valid) {
            const UvExactComponentOrigin row = tryDp2AddRow(o0x, o0y, k1x, k1y, k2);
            if (row.valid)
              return row;
          }
          if (!o0x.valid && !o0y.valid && !o2.valid) {
            const UvExactComponentOrigin row = tryDp2AddRow(o1x, o1y, k0x, k0y, k2);
            if (row.valid)
              return row;
          }

          UvExactComponentOrigin merged = uvMergeOrigins(
            readOriginUnion(ctx.src[0], 2u),
            readOriginUnion(ctx.src[1], 2u));
          merged = uvMergeOrigins(merged, o2);
          if (merged.valid)
            uvAffineMarkInexact(merged.affine);
          return merged;
        }

        case DxsoOpcode::Dp3:
        case DxsoOpcode::Dp4: {
          const uint32_t count = ctx.instruction.opcode == DxsoOpcode::Dp3 ? 3u : 4u;
          UvExactComponentOrigin merged = uvMergeOrigins(
            readOriginUnion(ctx.src[0], count),
            readOriginUnion(ctx.src[1], count));
          if (merged.valid)
            uvAffineMarkInexact(merged.affine);
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
          if (merged.valid)
            uvAffineMarkInexact(merged.affine);
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

      // Tracks temp components holding pure constant expressions (no interpolant input).
      // fxc hoists constant subexpressions - e.g. a def'd tiling literal times a uniform
      // scalar - into temps before multiplying/adding them onto UVs; without this, such
      // multiplies degrade to "origin times unknown factor" and the affine goes inexact.
      // Only single-part values (immediate, or one constant-register component times a
      // static factor) are representable; anything else invalidates the tracked value.
      UvConstComponentRef traceConstExpr(const DxsoInstructionContext& ctx, const uint32_t c) const {
        auto mulRefs = [](const UvConstComponentRef& a, const UvConstComponentRef& b) -> UvConstComponentRef {
          UvConstComponentRef result;
          if (!a.valid || !b.valid)
            return result;
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
              if (value == 0.0f)
                return result;
              value = 1.0f / value;
              break;
            case DxsoOpcode::Rsq:
              if (value <= 0.0f)
                return result;
              value = 1.0f / std::sqrt(value);
              break;
            case DxsoOpcode::Exp: value = std::exp2(value); break;
            case DxsoOpcode::Log:
              if (value == 0.0f)
                return result;
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
    static void analyzePsSamplerUvOrigins(const D3D9CommonShader* pixelShader,
                                          const XXH64_hash_t psHash,
                                          std::array<PsSamplerUvOrigin, caps::MaxTexturesPS>& outOrigins) {
      for (auto& origin : outOrigins)
        origin = PsSamplerUvOrigin{};

      if (pixelShader == nullptr)
        return;

      const auto& info = pixelShader->GetInfo();
      if (info.type() != DxsoProgramTypes::PixelShader)
        return;

      const auto& bytecode = pixelShader->GetBytecode();
      if (bytecode.size() < sizeof(uint32_t) || (bytecode.size() % sizeof(uint32_t)) != 0)
        return;

      // rtx.d3d9.ue3UvTraceShaderHashes: instruction-level trace of the UV dataflow analysis
      const bool traceInstructions =
        psHash != 0 &&
        !D3D9Rtx::ue3UvTraceShaderHashes().empty() &&
        lookupHash(D3D9Rtx::ue3UvTraceShaderHashes(), psHash);
      if (traceInstructions) {
        Logger::info(str::format(
          "[RTX-UV-TRACE] begin ps=0x", std::hex, psHash, std::dec,
          " version=", info.majorVersion(), ".", info.minorVersion(),
          " bytes=", bytecode.size()));
      }

      std::array<int8_t, 2 * DxsoMaxInterfaceRegs> inputRegToTexcoord = {};
      inputRegToTexcoord.fill(-1);
      {
        const auto& isgn = pixelShader->GetIsgn();
        for (uint32_t i = 0; i < isgn.elemCount; i++) {
          const auto& e = isgn.elems[i];
          if (e.semantic.usage == DxsoUsage::Texcoord && e.regNumber < inputRegToTexcoord.size())
            inputRegToTexcoord[e.regNumber] = int8_t(e.semantic.usageIndex & 0b111);
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

      const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
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

        if (sampledSampler < caps::MaxTexturesPS &&
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
            Logger::info(str::format(
              "[RTX-UV-TRACE] #", ctx.instructionIdx, " SAMPLE s", sampledSampler,
              coordReg != nullptr ? str::format(" coord=", formatDxsoSrcRegister(*coordReg)).c_str() : "",
              directTexcoordSite ? " direct-texcoord" : "",
              untraceableSite ? " untraceable-legacy-op" : "",
              projected ? " projected" : "",
              " => valid=", site.valid ? 1 : 0,
              siteDetail));
          }

          if (site.valid) {
            if (agg.validSiteCount < std::numeric_limits<uint16_t>::max())
              agg.validSiteCount++;

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

                // UE3 distance-fade anti-tiling materials sample the same texture at two
                // literal tilings and lerp by a saturated depth fade that is 0 near the
                // camera: the highest-frequency static-tiling site is the surface's
                // ground-truth mapping, not whichever site fxc emitted first.
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
            if (agg.invalidSiteCount < std::numeric_limits<uint16_t>::max())
              agg.invalidSiteCount++;
            if (agg.originValid)
              agg.sitesAgree = false;
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

          Logger::info(line);
        }
      }

      if (traceInstructions) {
        for (uint32_t s = 0; s < caps::MaxTexturesPS; s++) {
          const PsSamplerUvOrigin& origin = outOrigins[s];
          if (origin.validSiteCount == 0 && origin.invalidSiteCount == 0) {
            continue;
          }
          Logger::info(str::format(
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
        Logger::info(str::format("[RTX-UV-TRACE] end ps=0x", std::hex, psHash, std::dec));
      }
    }

    struct Ue3VsTexcoordTraceResult {
      Ue3VsUvTraceKind kind = Ue3VsUvTraceKind::Invalid;
      uint8_t iaTexcoordIndex = 0;
      uint8_t inputReg = 0;
      UvComponentAffine affineU;
      UvComponentAffine affineV;
    };

    // Traces a VS output TEXCOORD interpolant (components compU/compV) back to the IA:
    // proves which IA texcoord set feeds it and classifies the math along the way.
    static Ue3VsTexcoordTraceResult traceVsOutputTexcoordToInputUsageIndex(
      const D3D9CommonShader* vertexShader,
      const uint32_t outputRegNumber,
      const uint8_t compU,
      const uint8_t compV) {
      Ue3VsTexcoordTraceResult result;

      if (vertexShader == nullptr ||
          outputRegNumber >= DxsoMaxInterfaceRegs ||
          compU >= 4u ||
          compV >= 4u) {
        return result;
      }

      const auto& info = vertexShader->GetInfo();
      if (info.type() != DxsoProgramTypes::VertexShader)
        return result;

      const auto& bytecode = vertexShader->GetBytecode();
      if (bytecode.size() < sizeof(uint32_t) || (bytecode.size() % sizeof(uint32_t)) != 0)
        return result;

      std::array<int8_t, 2 * DxsoMaxInterfaceRegs> inputRegToTexcoord = {};
      inputRegToTexcoord.fill(-1);
      {
        const auto& isgn = vertexShader->GetIsgn();
        for (uint32_t i = 0; i < isgn.elemCount; i++) {
          const auto& e = isgn.elems[i];
          if (e.semantic.usage == DxsoUsage::Texcoord && e.regNumber < inputRegToTexcoord.size())
            inputRegToTexcoord[e.regNumber] = int8_t(e.semantic.usageIndex & 0b111);
        }
      }

      UvDataflowTracer tracer(inputRegToTexcoord, false /*texcoordRegsAreInterpolants*/, false /*originIsSemanticIndex*/);
      tracer.setTrackedOutputRegister(outputRegNumber);

      const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
      DxsoDecodeContext decoder(info);
      DxsoCodeIter iter(tokens + 1);

      while (decoder.decodeInstruction(iter))
        tracer.processInstruction(decoder.getInstructionContext());

      const UvExactComponentOrigin& uOrigin = tracer.outputOrigin(compU);
      const UvExactComponentOrigin& vOrigin = tracer.outputOrigin(compV);

      if (!uOrigin.valid || !vOrigin.valid || uOrigin.reg != vOrigin.reg)
        return result;

      if (uOrigin.reg >= inputRegToTexcoord.size() || inputRegToTexcoord[uOrigin.reg] < 0)
        return result;

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

    constexpr uint16_t kD3dxRegisterSetFloat4 = 2u;
    constexpr uint16_t kD3dxRegisterSetSampler = 3u;

    // CTAB sampler register -> declared name, for diagnostics. Cached per shader hash.
    static const std::map<uint32_t, std::string>& getUe3PsSamplerNames(
        const XXH64_hash_t psHash,
        const std::vector<uint8_t>& bytecode) {
      static fast_unordered_cache<std::map<uint32_t, std::string>> s_ue3PsSamplerNameCache;

      auto it = s_ue3PsSamplerNameCache.find(psHash);
      if (it != s_ue3PsSamplerNameCache.end()) {
        return it->second;
      }

      std::map<uint32_t, std::string> names;
      if (bytecode.size() >= sizeof(uint32_t) && (bytecode.size() % sizeof(uint32_t)) == 0) {
        const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
        const uint32_t headerToken = tokens[0];
        if ((headerToken & 0xffff0000u) == 0xffff0000u) {
          const uint32_t majorVersion = (headerToken >> 8) & 0xffu;
          const uint32_t minorVersion = headerToken & 0xffu;
          DxsoProgramInfo programInfo(DxsoProgramTypes::PixelShader, minorVersion, majorVersion);

          DxsoDecodeContext decoder(programInfo);
          DxsoCodeIter iter(tokens + 1);
          while (decoder.decodeInstruction(iter)) {
            if (decoder.getCtabInfo().m_size != 0)
              break;
          }

          const DxsoCtab& ctab = decoder.getCtabInfo();
          for (const DxsoCtab::Constant& c : ctab.m_constantData) {
            if (c.registerSet != kD3dxRegisterSetSampler || c.registerCount == 0)
              continue;
            const uint32_t end = std::min<uint32_t>(c.registerIndex + c.registerCount, caps::MaxTexturesPS);
            for (uint32_t s = c.registerIndex; s < end; s++) {
              names[s] = c.name;
            }
          }
        }
      }

      return s_ue3PsSamplerNameCache.emplace(psHash, std::move(names)).first->second;
    }

    // CTAB float-constant register -> declared name (UniformScalar_*/UniformVector_*/engine
    // constants), for rtx.d3d9.ue3LogUvAffineDetail diagnostics. Cached per shader hash.
    static const std::map<uint32_t, std::string>& getUe3PsFloatConstantNames(
        const XXH64_hash_t psHash,
        const std::vector<uint8_t>& bytecode) {
      static fast_unordered_cache<std::map<uint32_t, std::string>> s_ue3PsFloatConstantNameCache;

      auto it = s_ue3PsFloatConstantNameCache.find(psHash);
      if (it != s_ue3PsFloatConstantNameCache.end()) {
        return it->second;
      }

      std::map<uint32_t, std::string> names;
      if (bytecode.size() >= sizeof(uint32_t) && (bytecode.size() % sizeof(uint32_t)) == 0) {
        const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
        const uint32_t headerToken = tokens[0];
        if ((headerToken & 0xffff0000u) == 0xffff0000u) {
          const uint32_t majorVersion = (headerToken >> 8) & 0xffu;
          const uint32_t minorVersion = headerToken & 0xffu;
          DxsoProgramInfo programInfo(DxsoProgramTypes::PixelShader, minorVersion, majorVersion);

          DxsoDecodeContext decoder(programInfo);
          DxsoCodeIter iter(tokens + 1);
          while (decoder.decodeInstruction(iter)) {
            if (decoder.getCtabInfo().m_size != 0)
              break;
          }

          const DxsoCtab& ctab = decoder.getCtabInfo();
          for (const DxsoCtab::Constant& c : ctab.m_constantData) {
            if (c.registerSet != kD3dxRegisterSetFloat4 || c.registerCount == 0)
              continue;
            const uint32_t end = std::min<uint32_t>(c.registerIndex + c.registerCount, caps::MaxFloatConstantsPS);
            for (uint32_t r = c.registerIndex; r < end; r++) {
              names[r] = c.registerCount > 1u
                ? str::format(c.name, "[", r - c.registerIndex, "]")
                : c.name;
            }
          }
        }
      }

      return s_ue3PsFloatConstantNameCache.emplace(psHash, std::move(names)).first->second;
    }


    constexpr uint8_t kPsSamplerSemanticEngineAuxiliary = 1u << 0;
    constexpr uint8_t kPsSamplerSemanticLightmap        = 1u << 1;
    constexpr uint8_t kPsSamplerSemanticMaterialTexture = 1u << 2;
    constexpr uint8_t kPsSamplerSemanticNonDiffuse      = 1u << 3;
    constexpr uint8_t kPsSamplerSemanticVideo           = 1u << 4;
    constexpr uint8_t kPsSamplerSemanticMovieTexture    = 1u << 5;

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
    static uint32_t getDxsoSourceOperandCount(const DxsoOpcode op) {
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
    static uint32_t getDxsoMatrixRowCount(const DxsoOpcode op) {
      switch (op) {
      case DxsoOpcode::M4x4: return 4;
      case DxsoOpcode::M4x3:
      case DxsoOpcode::M3x4:
      case DxsoOpcode::M3x3: return 3;
      case DxsoOpcode::M3x2: return 2;
      default:               return 0;
      }
    }

    // expression-level hints inferred from shader opcode/dataflow around a sampler's UV path
    constexpr uint16_t kPsSamplerExprUvTransform = 1u << 0;
    constexpr uint16_t kPsSamplerExprUvOffset    = 1u << 1;
    constexpr uint16_t kPsSamplerExprUvAnimated  = 1u << 2;
    constexpr uint16_t kPsSamplerExprBlendMath   = 1u << 3;
    constexpr uint16_t kPsSamplerExprUvTimeDriven = 1u << 4;
    constexpr uint16_t kPsSamplerExprViewDependent = 1u << 5;
    constexpr uint16_t kPsSamplerExprMaskControl = 1u << 6;
    constexpr uint16_t kPsSamplerExprColorContribution = 1u << 7;
    // sampled value is decoded as a tangent-space normal (sign-expanded `t * 2 - 1` and/or normalized
    // via nrm / self dot-product) - the UE3 material compiler emits this for every Normal-input texture
    constexpr uint16_t kPsSamplerExprNormalDecode = 1u << 8;
    // sampled value arithmetically reaches the output color register (oC0.rgb) as a color term,
    // rather than only feeding coordinate/lighting math (dot products, kill tests, mask controls)
    constexpr uint16_t kPsSamplerExprReachesOutputColor = 1u << 9;
    // deterministic UE3 base-pass structure signal: the sampled value is multiplied (directly or
    // transitively) with a lightmap sample (static geometry) or with the AmbientColorAndSkyFactor /
    // sky color constants (dynamic geometry, unlit viewmode). In the generated base pass only the
    // material's DIFFUSE expression is modulated this way - emissive adds directly, specular
    // multiplies dot-product transfer chains, normal maps never contribute as color.
    constexpr uint16_t kPsSamplerExprDiffuseAnchor = 1u << 10;

    static uint8_t classifyPixelSamplerSemanticFlags(const std::string& samplerName) {
      std::string lowerName;
      lowerName.reserve(samplerName.size());
      for (const char c : samplerName)
        lowerName.push_back(char(std::tolower(static_cast<unsigned char>(c))));

      auto contains = [&](const char* token) {
        return lowerName.find(token) != std::string::npos;
      };

      uint8_t flags = 0;
      const bool isBinkPlaneName =
        contains("ycrcb") || contains("yuv") || contains("bink");
      const bool isMovieTextureName =
        contains("texturesampleparametermovie") || contains("movie");

      if (!isBinkPlaneName &&
          (contains("texture2d_") || contains("texturecube_") || contains("texture3d_") ||
          contains("materialtexture") || contains("materialsampler") ||
          contains("textureparameter") || contains("texturesample") ||
          contains("texturesampleparameter2d") || contains("texturesampleparametercube") ||
          contains("texturesampleparametermovie") || contains("texturesampleparametersubuv") ||
          contains("fontsampleparameter") || contains("movie") ||
          contains("albedo") || contains("diffuse") || contains("basecolor") ||
          contains("base_color") || contains("billboard") || contains("advert") ||
          contains("poster") || contains("decal") || contains("fontsample") ||
          contains("subuv") || contains("flipbook") || contains("particlesubuv") ||
          contains("meshsubuv") || contains("cubemap"))) {
        flags |= kPsSamplerSemanticMaterialTexture;
      }

      if (contains("scenecolor") || contains("scenedepth") || contains("lightattenuation") ||
          contains("previouslighting") || contains("exposuretexture") ||
          contains("previousexposure") || contains("scenedownsampled") ||
          contains("saturationmasktexture") || contains("randomangletexture") ||
          contains("bsplinetexture") || contains("colorcurvesktexture") ||
          contains("colorcurvesmtexture") || contains("blurredimage") ||
          contains("shadowdepth") || contains("shadowvariance") ||
          contains("shadowtexture") || contains("velocitybuffer") ||
          contains("ambientocclusiontexture") || contains("aohistorytexture") ||
          contains("randomnormaltexture") || contains("filtertexture") ||
          contains("scenecolorscratchtexture") || contains("ldrtranslucencytexture") ||
          contains("accumulateddistortiontexture") || contains("accumulatedfrontfaceslineintegraltexture") ||
          contains("accumulatedbackfaceslineintegraltexture") || contains("scenecoloruitexture") ||
          contains("uibuffer") || contains("uitexture") ||
          contains("blurreduitexture") || contains("sceneblurtexture") ||
          contains("colorcurves") || contains("exposure") ||
          contains("ycrcb") || contains("yuv") || contains("bink") ||
          contains("destdepth") || contains("destcolor") ||
          contains("pixeldepth") || contains("scenetexture") ||
          contains("depthbias") || contains("depthbiased") ||
          contains("lensflare") || contains("lens_flare") ||
          contains("screenposition") || contains("screen_position") ||
          contains("masktexture")) {
        flags |= kPsSamplerSemanticEngineAuxiliary;
      }

      // UE3 lightmap machinery: the LightMapTextures[] coefficient array declared by every
      // texture lightmap permutation, plus Mirror's Edge's BSplineTexture weight LUT, which
      // exists only to filter those coefficients (TdBicubicFiltering).
      if (contains("lightmap") || contains("bspline")) {
        flags |= kPsSamplerSemanticLightmap;
        flags |= kPsSamplerSemanticEngineAuxiliary;
      }

      if (isBinkPlaneName) {
        flags |= kPsSamplerSemanticVideo;
        flags |= kPsSamplerSemanticEngineAuxiliary;
        flags |= kPsSamplerSemanticNonDiffuse;
      }
      if (!isBinkPlaneName && isMovieTextureName) {
        flags |= kPsSamplerSemanticMovieTexture;
      }

      if (contains("normal") || contains("specular") || contains("roughness") ||
          contains("gloss") || contains("metallic") || contains("metalness") ||
          contains("ambientocclusion") || contains("_ao") || contains("ao_") ||
          contains("heightmap") || contains("height") || contains("bump") ||
          contains("opacity") || contains("alphamask") || contains("mask") ||
          contains("dirt") || contains("grunge") || contains("detailmask") ||
          contains("blendmask") || contains("lookup") ||
          contains("reflection") || contains("reflectionvector") ||
          contains("fresnel") || contains("cameravector") ||
          contains("lightvector") || contains("envmap") ||
          contains("specularcube")) {
        flags |= kPsSamplerSemanticNonDiffuse;
      }

      return flags;
    }

    static uint16_t classifyPixelSamplerExpressionFlagsFromName(const std::string& samplerName) {
      std::string lowerName;
      lowerName.reserve(samplerName.size());
      for (const char c : samplerName)
        lowerName.push_back(char(std::tolower(static_cast<unsigned char>(c))));

      auto contains = [&](const char* token) {
        return lowerName.find(token) != std::string::npos;
      };

      uint16_t flags = 0;

      if (contains("time") || contains("gametime") || contains("realtime") ||
          contains("panner") || contains("rotator") || contains("rotation") ||
          contains("flipbook") || contains("subuv") || contains("phase") ||
          contains("sine") || contains("cosine")) {
        flags |= kPsSamplerExprUvTimeDriven;
        flags |= kPsSamplerExprUvAnimated;
        flags |= kPsSamplerExprUvOffset;
      }

      if (contains("reflection") || contains("reflectionvector") ||
          contains("fresnel") || contains("cameravector") ||
          contains("lightvector") || contains("envmap") ||
          contains("specularcube")) {
        flags |= kPsSamplerExprViewDependent;
      }

      if (contains("mask") || contains("alphamask") || contains("opacity") ||
          contains("componentmask") || contains("staticcomponentmask") ||
          contains("switch") || contains("staticswitch") ||
          contains("blendmask") || contains("lookup") ||
          contains("dirt") || contains("grunge")) {
        flags |= kPsSamplerExprMaskControl;
      }

      if (contains("lerp") || contains("blend") || contains("interpolate"))
        flags |= kPsSamplerExprBlendMath;

      if (contains("albedo") || contains("diffuse") || contains("basecolor") ||
          contains("base_color") || contains("billboard") || contains("advert") ||
          contains("poster") || contains("decal") || contains("fontsample")) {
        flags |= kPsSamplerExprColorContribution;
      }

      return flags;
    }

    struct PsSamplerTexcoordInference {
      int32_t texcoord = -1;
      bool coordCompValid = false;
      uint8_t coordCompU = 0;
      uint8_t coordCompV = 1;
      uint16_t sampleCount = 0;
      uint8_t semanticFlags = 0;
      uint16_t expressionFlags = 0;
      int32_t scaleConstReg = -1;
      uint8_t scaleConstCompU = 0;
      uint8_t scaleConstCompV = 1;
      float scaleFactorU = 1.0f;
      float scaleFactorV = 1.0f;
      bool scaleImmediateValid = false;
      float scaleImmediateU = 1.0f;
      float scaleImmediateV = 1.0f;
      int32_t offsetConstReg = -1;
      uint8_t offsetConstCompU = 0;
      uint8_t offsetConstCompV = 1;
      float offsetFactorU = 1.0f;
      float offsetFactorV = 1.0f;
      bool offsetImmediateValid = false;
      float offsetImmediateU = 0.0f;
      float offsetImmediateV = 0.0f;
      // Every float constant register this sampler's coordinate depends on, ascending. Unlike
      // scaleConstReg/offsetConstReg - which are outputs of the affine resolver and so only
      // exist for the shapes it can express - this is the transitive dataflow, so it also
      // covers a rotator's 2x2 matrix, a matrix multiply, and any chain through temps. Excludes
      // `def` literals, which are bytecode rather than draw state.
      std::vector<uint32_t> coordConstRegs;
    };

    struct PsTexcoordScaleHint {
      int32_t constReg = -1;
      uint8_t compU = 0;
      uint8_t compV = 1;
      float scaleFactorU = 1.0f;
      float scaleFactorV = 1.0f;
      bool immediateValid = false;
      float immediateU = 1.0f;
      float immediateV = 1.0f;
      int32_t offsetConstReg = -1;
      uint8_t offsetCompU = 0;
      uint8_t offsetCompV = 1;
      float offsetFactorU = 1.0f;
      float offsetFactorV = 1.0f;
      bool offsetImmediateValid = false;
      float offsetImmediateU = 0.0f;
      float offsetImmediateV = 0.0f;
    };

    // Infer which TEXCOORD set (usageIndex) a pixel shader uses for a given sampler by analyzing shader bytecode,
    // also captures UE3-style TextureCoordinate scaling (UTiling/VTiling) when emitted as `TexCoord * cN`
    static PsSamplerTexcoordInference inferPixelShaderTexcoordForSampler(const D3D9CommonShader* pixelShader, const uint32_t samplerIdx) {
      PsSamplerTexcoordInference result;

      if (pixelShader == nullptr || samplerIdx >= caps::MaxTexturesPS)
        return result;

      const auto& info = pixelShader->GetInfo();
      if (info.type() != DxsoProgramTypes::PixelShader)
        return result;

      const auto& bytecode = pixelShader->GetBytecode();
      if (bytecode.size() < sizeof(uint32_t) || (bytecode.size() % sizeof(uint32_t)) != 0)
        return result;

      const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
      DxsoDecodeContext decoder(info);
      DxsoCodeIter iter(tokens + 1);

      std::array<int8_t, 2 * DxsoMaxInterfaceRegs> inputRegToTexcoord = {};
      inputRegToTexcoord.fill(-1);
      {
        const auto& isgn = pixelShader->GetIsgn();
        for (uint32_t i = 0; i < isgn.elemCount; i++) {
          const auto& e = isgn.elems[i];
          if (e.semantic.usage == DxsoUsage::Texcoord && e.regNumber < inputRegToTexcoord.size())
            inputRegToTexcoord[e.regNumber] = int8_t(e.semantic.usageIndex & 0b111);
        }
      }

      std::array<int8_t, 64> tempToTexcoord = {};
      tempToTexcoord.fill(-1);

      std::array<PsTexcoordScaleHint, 64> tempToScaleHint = {};
      std::array<uint8_t, 64> tempCoordProvenance = {};
      tempCoordProvenance.fill(0);
      std::array<uint16_t, 64> tempCoordExpressionFlags = {};
      tempCoordExpressionFlags.fill(0);
      std::array<uint8_t, 64> tempSamplerValueRole = {};
      tempSamplerValueRole.fill(0);

      // per-temp decode state of this sampler's value, used to recognize the UE3 normal-map
      // unpack `sample * (UnpackMax-UnpackMin) + UnpackMin` = `t * 2 - 1` and its split forms
      constexpr uint8_t kValueStateSignExpanded = 1u << 0; // t * 2 - 1 applied
      constexpr uint8_t kValueStateScaledX2     = 1u << 1; // t * 2 applied
      constexpr uint8_t kValueStateBiasedHalf   = 1u << 2; // t - 0.5 applied
      std::array<uint8_t, 64> tempSamplerValueState = {};
      tempSamplerValueState.fill(0);

      // diffuse anchor provenance: values derived from a lightmap sample or from the
      // UE3 ambient/sky lighting constants (see kPsSamplerExprDiffuseAnchor)
      constexpr uint8_t kAnchorLightmapValue = 1u << 0;
      constexpr uint8_t kAnchorLightingConst = 1u << 1;
      std::array<uint8_t, 64> tempAnchorBits = {};
      tempAnchorBits.fill(0);
      bool anchorInfoInitialized = false;
      uint32_t anchorLightmapSamplerMask = 0;
      std::array<uint8_t, caps::MaxFloatConstantsPS> anchorLightingConstRegs = {};
      anchorLightingConstRegs.fill(0);

      constexpr uint8_t kCoordProvTexcoord = 1u << 0;
      constexpr uint8_t kCoordProvNonTexcoord = 1u << 1;

      // D3D9 literal constants emitted by `def cN, ...` are part of shader bytecode and arent draw call constant state
      std::array<uint8_t, caps::MaxFloatConstantsPS> defFloatConstValid = {};
      defFloatConstValid.fill(0);
      std::array<Vector4, caps::MaxFloatConstantsPS> defFloatConsts = {};

      // Which float constant registers each temp's value transitively depends on, as a bitset.
      // Read at a sample to learn what a coordinate is built from, which is what tells a texture
      // transform apart from an authored parameter no matter what shape the expression takes.
      constexpr uint32_t kConstDepWords = (caps::MaxFloatConstantsPS + 63u) / 64u;
      using ConstDepSet = std::array<uint64_t, kConstDepWords>;
      std::array<ConstDepSet, 64> tempConstDeps = {};

      auto getTexcoordFromRegister = [&](const DxsoRegister& r) -> int32_t {
        auto mapPsInputRegToTexcoordUsage = [&](const uint32_t regNum) -> int32_t {
          if (regNum < inputRegToTexcoord.size()) {
            const int32_t mapped = inputRegToTexcoord[regNum];
            if (mapped >= 0)
              return mapped;
          }
          // fallback for older/atypical signatures where we preserve legacy register-index behavior
          return int32_t(regNum & 0b111);
        };

        switch (r.id.type) {
        case DxsoRegisterType::Texture:
        case DxsoRegisterType::PixelTexcoord:
          return mapPsInputRegToTexcoordUsage(r.id.num);
        case DxsoRegisterType::Input:
          return (r.id.num < inputRegToTexcoord.size()) ? inputRegToTexcoord[r.id.num] : -1;
        case DxsoRegisterType::Temp:
        case DxsoRegisterType::TempFloat16:
          return (r.id.num < tempToTexcoord.size()) ? tempToTexcoord[r.id.num] : -1;
        default:
          return -1;
        }
      };

      auto getCoordProvenanceFromRegister = [&](const DxsoRegister& r) -> uint8_t {
        switch (r.id.type) {
        case DxsoRegisterType::Texture:
        case DxsoRegisterType::PixelTexcoord:
          return kCoordProvTexcoord;
        case DxsoRegisterType::Input:
          if (r.id.num < inputRegToTexcoord.size())
            return inputRegToTexcoord[r.id.num] >= 0 ? kCoordProvTexcoord : kCoordProvNonTexcoord;
          return kCoordProvNonTexcoord;
        case DxsoRegisterType::Temp:
        case DxsoRegisterType::TempFloat16:
          return (r.id.num < tempCoordProvenance.size())
            ? tempCoordProvenance[r.id.num]
            : 0u;
        default:
          return 0u;
        }
      };

      // A `def` literal is baked into the bytecode, so it is already part of the shader identity
      // seed and can never vary between draws.
      auto markConstDep = [&](const int32_t reg, ConstDepSet& inOut) {
        if (reg >= 0 && reg < int32_t(caps::MaxFloatConstantsPS) && !defFloatConstValid[reg])
          inOut[uint32_t(reg) / 64u] |= 1ull << (uint32_t(reg) % 64u);
      };

      auto orConstDepsFromRegister = [&](const DxsoRegister& r, ConstDepSet& inOut) {
        switch (r.id.type) {
        case DxsoRegisterType::Temp:
        case DxsoRegisterType::TempFloat16:
          if (r.id.num < tempConstDeps.size()) {
            const ConstDepSet& deps = tempConstDeps[r.id.num];
            for (uint32_t w = 0; w < kConstDepWords; w++)
              inOut[w] |= deps[w];
          }
          break;
        default:
          if (isFloatConstantRegisterType(r.id.type) && !r.hasRelative)
            markConstDep(getFloatConstantRegisterIndex(r), inOut);
          break;
        }
      };

      auto getCoordExpressionFlagsFromRegister = [&](const DxsoRegister& r) -> uint16_t {
        switch (r.id.type) {
        case DxsoRegisterType::Temp:
        case DxsoRegisterType::TempFloat16:
          return (r.id.num < tempCoordExpressionFlags.size())
            ? tempCoordExpressionFlags[r.id.num]
            : uint16_t(0u);
        default:
          return uint16_t(0u);
        }
      };

      constexpr uint8_t kSamplerValueRoleColor = 1u << 0;
      constexpr uint8_t kSamplerValueRoleControl = 1u << 1;
      auto getSamplerValueRoleFromRegister = [&](const DxsoRegister& r) -> uint8_t {
        switch (r.id.type) {
        case DxsoRegisterType::Temp:
        case DxsoRegisterType::TempFloat16:
          return (r.id.num < tempSamplerValueRole.size())
            ? tempSamplerValueRole[r.id.num]
            : 0u;
        default:
          return 0u;
        }
      };
      auto isScalarSwizzle = [](const DxsoRegister& r) {
        const uint8_t c0 = r.swizzle[0] & 0x3u;
        return c0 == (r.swizzle[1] & 0x3u) &&
               c0 == (r.swizzle[2] & 0x3u) &&
               c0 == (r.swizzle[3] & 0x3u);
      };
      auto isAlphaScalarSwizzle = [&](const DxsoRegister& r) {
        return isScalarSwizzle(r) && ((r.swizzle[0] & 0x3u) == 3u);
      };
      auto classifySamplerValueRoleWithSwizzle = [&](const DxsoRegister& r) -> uint8_t {
        const uint8_t baseRole = getSamplerValueRoleFromRegister(r);
        if (baseRole == 0u)
          return 0u;

        if (isAlphaScalarSwizzle(r))
          return uint8_t(baseRole | kSamplerValueRoleControl);

        return baseRole;
      };

      auto getSamplerValueStateFromRegister = [&](const DxsoRegister& r) -> uint8_t {
        switch (r.id.type) {
        case DxsoRegisterType::Temp:
        case DxsoRegisterType::TempFloat16:
          return (r.id.num < tempSamplerValueState.size())
            ? tempSamplerValueState[r.id.num]
            : 0u;
        default:
          return 0u;
        }
      };

      auto getAnchorBitsFromRegister = [&](const DxsoRegister& r) -> uint8_t {
        switch (r.id.type) {
        case DxsoRegisterType::Temp:
        case DxsoRegisterType::TempFloat16:
          return (r.id.num < tempAnchorBits.size())
            ? tempAnchorBits[r.id.num]
            : 0u;
        default:
          if (isFloatConstantRegisterType(r.id.type) && !r.hasRelative) {
            const int32_t reg = getFloatConstantRegisterIndex(r);
            if (reg >= 0 && reg < int32_t(anchorLightingConstRegs.size()) && anchorLightingConstRegs[reg])
              return kAnchorLightingConst;
          }
          return 0u;
        }
      };

      // true when the source is a `def` literal constant whose swizzled rgb components all equal
      // `target` (within tolerance) - used to recognize the folded normal unpack constants (2, -1)
      auto isDefConstNearRgb = [&](const DxsoRegister& r, const float target) -> bool {
        if (!isFloatConstantRegisterType(r.id.type) || r.hasRelative)
          return false;

        float modifierScale = 1.0f;
        if (!decodeConstantModifierScale(r.modifier, modifierScale))
          return false;

        const int32_t reg = getFloatConstantRegisterIndex(r);
        if (reg < 0 || reg >= int32_t(defFloatConstValid.size()) || !defFloatConstValid[reg])
          return false;

        constexpr float kTolerance = 0.01f;
        for (uint32_t comp = 0; comp < 3; comp++) {
          const float value = modifierScale * defFloatConsts[reg][r.swizzle[comp] & 0x3u];
          if (std::abs(value - target) > kTolerance)
            return false;
        }
        return true;
      };

      auto getScaleHintFromRegister = [&](const DxsoRegister& r, PsTexcoordScaleHint& outHint) -> bool {
        if ((r.id.type == DxsoRegisterType::Temp || r.id.type == DxsoRegisterType::TempFloat16) &&
            r.id.num < tempToScaleHint.size()) {
          const auto& hint = tempToScaleHint[r.id.num];
          if (hint.constReg >= 0 || hint.immediateValid ||
              hint.offsetConstReg >= 0 || hint.offsetImmediateValid) {
            outHint = hint;
            return true;
          }
        }
        return false;
      };

      auto loadScaleHintFromConstant = [&](const DxsoRegister& r, PsTexcoordScaleHint& outHint) -> bool {
        if (!isFloatConstantRegisterType(r.id.type))
          return false;

        float modifierScale = 1.0f;
        if (!decodeConstantModifierScale(r.modifier, modifierScale))
          return false;

        const int32_t reg = getFloatConstantRegisterIndex(r);
        if (reg < 0)
          return false;

        outHint.constReg = reg;
        outHint.compU = r.swizzle[0] & 0x3;
        outHint.compV = r.swizzle[1] & 0x3;
        outHint.scaleFactorU = modifierScale;
        outHint.scaleFactorV = modifierScale;

        if (reg < int32_t(defFloatConstValid.size()) && defFloatConstValid[reg]) {
          const Vector4& c = defFloatConsts[reg];
          outHint.immediateValid = true;
          outHint.immediateU = modifierScale * c[outHint.compU];
          outHint.immediateV = modifierScale * c[outHint.compV];
        }

        return true;
      };

      auto applyOffsetFromConstant = [&](const DxsoRegister& r, PsTexcoordScaleHint& inOutHint, const float sign) -> bool {
        PsTexcoordScaleHint temp;
        if (!loadScaleHintFromConstant(r, temp))
          return false;

        inOutHint.offsetConstReg = temp.constReg;
        inOutHint.offsetCompU = temp.compU;
        inOutHint.offsetCompV = temp.compV;
        inOutHint.offsetFactorU = sign * temp.scaleFactorU;
        inOutHint.offsetFactorV = sign * temp.scaleFactorV;
        inOutHint.offsetImmediateValid = temp.immediateValid;
        if (temp.immediateValid) {
          inOutHint.offsetImmediateU = sign * temp.immediateU;
          inOutHint.offsetImmediateV = sign * temp.immediateV;
        } else {
          inOutHint.offsetImmediateU = 0.0f;
          inOutHint.offsetImmediateV = 0.0f;
        }
        return true;
      };

      auto isPureScaleHint = [](const PsTexcoordScaleHint& h) {
        return h.offsetConstReg < 0 && !h.offsetImmediateValid;
      };

      auto combineMulHints = [&](const PsTexcoordScaleHint& a, const PsTexcoordScaleHint& b, PsTexcoordScaleHint& out) -> bool {
        out = PsTexcoordScaleHint{};

        const bool aConst = (a.constReg >= 0) && !a.immediateValid && isPureScaleHint(a);
        const bool bConst = (b.constReg >= 0) && !b.immediateValid && isPureScaleHint(b);
        const bool aImm = a.immediateValid && isPureScaleHint(a);
        const bool bImm = b.immediateValid && isPureScaleHint(b);

        if (aConst && bImm) {
          out = a;
          out.scaleFactorU = a.scaleFactorU * b.immediateU;
          out.scaleFactorV = a.scaleFactorV * b.immediateV;
          out.immediateValid = false;
          return true;
        }
        if (bConst && aImm) {
          out = b;
          out.scaleFactorU = b.scaleFactorU * a.immediateU;
          out.scaleFactorV = b.scaleFactorV * a.immediateV;
          out.immediateValid = false;
          return true;
        }
        if (aImm && bImm) {
          out.immediateValid = true;
          out.immediateU = a.immediateU * b.immediateU;
          out.immediateV = a.immediateV * b.immediateV;
          return true;
        }

        return false;
      };

      auto tryLoadConstantLikeHint = [&](const DxsoRegister& r, PsTexcoordScaleHint& outHint) -> bool {
        if (loadScaleHintFromConstant(r, outHint))
          return true;

        if (getScaleHintFromRegister(r, outHint) &&
            (outHint.constReg >= 0 || outHint.immediateValid))
          return true;

        return false;
      };

      auto mergeSingle = [](std::initializer_list<int32_t> vals) -> int32_t {
        int32_t v = -1;
        for (int32_t x : vals) {
          if (x < 0) continue;
          if (v < 0) v = x;
          else if (v != x) return -1;
        }
        return v;
      };

      auto isSameScaleHint = [](const PsTexcoordScaleHint& a, const PsTexcoordScaleHint& b) -> bool {
        if (a.constReg != b.constReg ||
            a.compU != b.compU ||
            a.compV != b.compV ||
            a.scaleFactorU != b.scaleFactorU ||
            a.scaleFactorV != b.scaleFactorV ||
            a.immediateValid != b.immediateValid ||
            a.offsetConstReg != b.offsetConstReg ||
            a.offsetCompU != b.offsetCompU ||
            a.offsetCompV != b.offsetCompV ||
            a.offsetFactorU != b.offsetFactorU ||
            a.offsetFactorV != b.offsetFactorV ||
            a.offsetImmediateValid != b.offsetImmediateValid)
          return false;

        if (a.immediateValid)
          if (a.immediateU != b.immediateU || a.immediateV != b.immediateV)
            return false;

        if (a.offsetImmediateValid)
          if (a.offsetImmediateU != b.offsetImmediateU || a.offsetImmediateV != b.offsetImmediateV)
            return false;

        return true;
      };

      std::array<uint32_t, 8> texcoordUseCount = {};
      texcoordUseCount.fill(0);
      std::array<std::array<uint16_t, 16>, 8> texcoordCoordPairUseCount = {};
      for (auto& coordPairUseCount : texcoordCoordPairUseCount)
        coordPairUseCount.fill(0);

      struct ScaleHintAgg {
        bool valid = false;
        bool conflict = false;
        PsTexcoordScaleHint hint;
      };
      std::array<ScaleHintAgg, 8> perTexcoordScaleHints = {};
      uint32_t texcoordDerivedSampleCount = 0;
      uint32_t nonTexcoordDerivedSampleCount = 0;
      uint16_t sampledCoordExpressionFlags = 0;
      ConstDepSet sampledCoordConstDeps = {};  // -> result.coordConstRegs
      bool normalDecodeDetected = false;   // -> kPsSamplerExprNormalDecode
      bool reachesOutputColor = false;     // -> kPsSamplerExprReachesOutputColor
      bool diffuseAnchorDetected = false;  // -> kPsSamplerExprDiffuseAnchor

      auto maskWritesRgb = [](const DxsoRegMask& mask) {
        return mask.popCount() == 0 || mask[0] || mask[1] || mask[2];
      };

      while (decoder.decodeInstruction(iter)) {
        const auto& ctx = decoder.getInstructionContext();
        const DxsoOpcode op = ctx.instruction.opcode;

        // the CTAB comment token precedes all instructions; collect diffuse-anchor sources
        // (lightmap samplers, UE3 ambient/sky lighting constants) as soon as it is decoded
        if (!anchorInfoInitialized && decoder.getCtabInfo().m_size != 0) {
          anchorInfoInitialized = true;
          for (const DxsoCtab::Constant& c : decoder.getCtabInfo().m_constantData) {
            if (c.registerCount == 0)
              continue;
            const std::string lowerName = toLowerAscii(c.name);
            if (c.registerSet == kD3dxRegisterSetSampler) {
              if (lowerName.find("lightmap") != std::string::npos) {
                const uint32_t end = std::min<uint32_t>(c.registerIndex + c.registerCount, 32u);
                for (uint32_t reg = c.registerIndex; reg < end; reg++)
                  anchorLightmapSamplerMask |= 1u << reg;
              }
            } else if (c.registerSet == kD3dxRegisterSetFloat4) {
              // BasePassPixelShader.usf: unlit/dynamic diffuse is multiplied by
              // AmbientColorAndSkyFactor.rgb; sky-lit diffuse by Upper/LowerSkyColor
              if (lowerName.find("ambientcolorandskyfactor") != std::string::npos ||
                  lowerName.find("upperskycolor") != std::string::npos ||
                  lowerName.find("lowerskycolor") != std::string::npos) {
                const uint32_t end = std::min<uint32_t>(c.registerIndex + c.registerCount,
                                                        uint32_t(anchorLightingConstRegs.size()));
                for (uint32_t reg = c.registerIndex; reg < end; reg++)
                  anchorLightingConstRegs[reg] = 1;
              }
            }
          }
        }

        if (op == DxsoOpcode::Def &&
            ctx.dst.id.type == DxsoRegisterType::Const &&
            ctx.dst.id.num < defFloatConsts.size()) {
          defFloatConstValid[ctx.dst.id.num] = 1;
          defFloatConsts[ctx.dst.id.num] = Vector4(
            ctx.def.float32[0],
            ctx.def.float32[1],
            ctx.def.float32[2],
            ctx.def.float32[3]);
        }

        // diffuse anchor detection: this sampler's color value multiplied with a lightmap-derived
        // value or a UE3 lighting constant. Only the product operands count (mad src2 is additive).
        // Checked before the write below so same-register products still see pre-write state.
        if (!diffuseAnchorDetected && (op == DxsoOpcode::Mul || op == DxsoOpcode::Mad)) {
          const bool roleA = (getSamplerValueRoleFromRegister(ctx.src[0]) & kSamplerValueRoleColor) != 0;
          const bool roleB = (getSamplerValueRoleFromRegister(ctx.src[1]) & kSamplerValueRoleColor) != 0;
          const uint8_t anchorA = getAnchorBitsFromRegister(ctx.src[0]);
          const uint8_t anchorB = getAnchorBitsFromRegister(ctx.src[1]);
          if ((roleA && anchorB != 0) || (roleB && anchorA != 0))
            diffuseAnchorDetected = true;
        }

        // Constant dependency propagation, for every temp write rather than only the ones the
        // coordinate tracking recognises: a rotator reaches its coordinate through intermediate
        // temps that are not themselves coordinates. Sources are read before the destination is
        // assigned, so an in-place update (dst == src) sees its own prior dependencies. Only the
        // set belonging to a register actually sampled from is ever read back.
        if ((ctx.dst.id.type == DxsoRegisterType::Temp || ctx.dst.id.type == DxsoRegisterType::TempFloat16) &&
            ctx.dst.id.num < tempConstDeps.size() &&
            op != DxsoOpcode::Def && op != DxsoOpcode::DefI && op != DxsoOpcode::DefB) {
          ConstDepSet deps = {};
          const uint32_t srcCount = std::min<uint32_t>(getDxsoSourceOperandCount(op), uint32_t(ctx.src.size()));
          for (uint32_t s = 0; s < srcCount; s++)
            orConstDepsFromRegister(ctx.src[s], deps);

          const uint32_t matrixRows = getDxsoMatrixRowCount(op);
          if (matrixRows > 1u && srcCount >= 2u &&
              isFloatConstantRegisterType(ctx.src[1].id.type) && !ctx.src[1].hasRelative) {
            const int32_t base = getFloatConstantRegisterIndex(ctx.src[1]);
            for (uint32_t row = 1; row < matrixRows; row++)
              markConstDep(base + int32_t(row), deps);
          }

          tempConstDeps[ctx.dst.id.num] = deps;
        }

        if ((ctx.dst.id.type == DxsoRegisterType::Temp || ctx.dst.id.type == DxsoRegisterType::TempFloat16) &&
            ctx.dst.id.num < tempToTexcoord.size()) {
          int32_t derived = -1;
          bool writesTrackedTemp = false;
          PsTexcoordScaleHint derivedScaleHint;
          uint16_t derivedExpressionFlags = uint16_t(
            getCoordExpressionFlagsFromRegister(ctx.src[0]) |
            getCoordExpressionFlagsFromRegister(ctx.src[1]) |
            getCoordExpressionFlagsFromRegister(ctx.src[2]));
          uint8_t derivedSamplerValueRole = 0;
          bool hasConstOnlyHint = false;
          PsTexcoordScaleHint constOnlyHint;

          auto assignFromSource = [&](const DxsoRegister& srcReg) {
            derived = getTexcoordFromRegister(srcReg);
            derivedExpressionFlags |= getCoordExpressionFlagsFromRegister(srcReg);
            derivedSamplerValueRole |= classifySamplerValueRoleWithSwizzle(srcReg);
            if (derived < 0)
              return;

            if (!getScaleHintFromRegister(srcReg, derivedScaleHint))
              derivedScaleHint = PsTexcoordScaleHint{};
          };

          switch (op) {
          case DxsoOpcode::Mov:
          case DxsoOpcode::Rcp:
          case DxsoOpcode::Rsq:
          case DxsoOpcode::Exp:
          case DxsoOpcode::Log:
          case DxsoOpcode::Frc:
          case DxsoOpcode::SinCos:
          case DxsoOpcode::Abs:
          case DxsoOpcode::Nrm:
          case DxsoOpcode::DsX:
          case DxsoOpcode::DsY:
            writesTrackedTemp = true;
            assignFromSource(ctx.src[0]);
            if (op != DxsoOpcode::Mov)
              derivedExpressionFlags |= kPsSamplerExprBlendMath;
            if (op == DxsoOpcode::Rcp || op == DxsoOpcode::Rsq ||
                op == DxsoOpcode::Exp || op == DxsoOpcode::Log ||
                op == DxsoOpcode::Abs || op == DxsoOpcode::Nrm) {
              derivedExpressionFlags |= kPsSamplerExprUvTransform;
            }
            if (op == DxsoOpcode::Frc || op == DxsoOpcode::SinCos)
              derivedExpressionFlags |= kPsSamplerExprUvAnimated;
            if (op == DxsoOpcode::Nrm) {
              const uint8_t src0Prov = getCoordProvenanceFromRegister(ctx.src[0]);
              if ((src0Prov & kCoordProvNonTexcoord) != 0 &&
                  (src0Prov & kCoordProvTexcoord) == 0) {
                derivedExpressionFlags |= kPsSamplerExprViewDependent;
              }
              // normalizing a sampled value = direction data, not color (normal/vector map)
              if (getSamplerValueRoleFromRegister(ctx.src[0]) != 0)
                normalDecodeDetected = true;
            }
            if (derived < 0) {
              hasConstOnlyHint = loadScaleHintFromConstant(ctx.src[0], constOnlyHint);
              if (!hasConstOnlyHint) {
                PsTexcoordScaleHint tempHint;
                if (getScaleHintFromRegister(ctx.src[0], tempHint) &&
                    (tempHint.constReg >= 0 || tempHint.immediateValid)) {
                  hasConstOnlyHint = true;
                  constOnlyHint = tempHint;
                }
              }
            }
            break;
          case DxsoOpcode::Add:
          case DxsoOpcode::Sub: {
            writesTrackedTemp = true;
            derivedExpressionFlags |= kPsSamplerExprBlendMath;
            const int32_t tc0 = getTexcoordFromRegister(ctx.src[0]);
            const int32_t tc1 = getTexcoordFromRegister(ctx.src[1]);

            if (tc0 >= 0 && tc1 < 0) {
              assignFromSource(ctx.src[0]);
              if (derived >= 0) {
                derivedExpressionFlags |= kPsSamplerExprUvOffset;
                bool hasKnownOffsetHint = false;
                const float sign = (op == DxsoOpcode::Sub) ? -1.0f : 1.0f;
                if (!applyOffsetFromConstant(ctx.src[1], derivedScaleHint, sign)) {
                  PsTexcoordScaleHint tempHint;
                  if (getScaleHintFromRegister(ctx.src[1], tempHint)) {
                    if (tempHint.constReg >= 0 || tempHint.immediateValid) {
                      hasKnownOffsetHint = true;
                      derivedScaleHint.offsetConstReg = tempHint.constReg;
                      derivedScaleHint.offsetCompU = tempHint.compU;
                      derivedScaleHint.offsetCompV = tempHint.compV;
                      derivedScaleHint.offsetFactorU = sign * tempHint.scaleFactorU;
                      derivedScaleHint.offsetFactorV = sign * tempHint.scaleFactorV;
                      derivedScaleHint.offsetImmediateValid = tempHint.immediateValid;
                      if (tempHint.immediateValid) {
                        derivedScaleHint.offsetImmediateU = sign * tempHint.immediateU;
                        derivedScaleHint.offsetImmediateV = sign * tempHint.immediateV;
                      }
                    } else if (tempHint.offsetConstReg >= 0 || tempHint.offsetImmediateValid) {
                      hasKnownOffsetHint = true;
                      derivedScaleHint.offsetConstReg = tempHint.offsetConstReg;
                      derivedScaleHint.offsetCompU = tempHint.offsetCompU;
                      derivedScaleHint.offsetCompV = tempHint.offsetCompV;
                      derivedScaleHint.offsetFactorU = sign * tempHint.offsetFactorU;
                      derivedScaleHint.offsetFactorV = sign * tempHint.offsetFactorV;
                      derivedScaleHint.offsetImmediateValid = tempHint.offsetImmediateValid;
                      if (tempHint.offsetImmediateValid) {
                        derivedScaleHint.offsetImmediateU = sign * tempHint.offsetImmediateU;
                        derivedScaleHint.offsetImmediateV = sign * tempHint.offsetImmediateV;
                      }
                    }
                  }
                } else {
                  hasKnownOffsetHint = true;
                }

                const bool rhsLooksNonTexcoord =
                  (getCoordProvenanceFromRegister(ctx.src[1]) & kCoordProvNonTexcoord) != 0;
                if (!hasKnownOffsetHint && rhsLooksNonTexcoord)
                  derivedExpressionFlags |= kPsSamplerExprUvAnimated;
                else if (!hasKnownOffsetHint)
                  derivedExpressionFlags |= kPsSamplerExprBlendMath;
              }
            } else if (tc1 >= 0 && tc0 < 0) {
              assignFromSource(ctx.src[1]);
              if (derived >= 0) {
                derivedExpressionFlags |= kPsSamplerExprUvOffset;
                bool hasKnownOffsetHint = false;
                if (op == DxsoOpcode::Add) {
                  if (!applyOffsetFromConstant(ctx.src[0], derivedScaleHint, +1.0f)) {
                    PsTexcoordScaleHint tempHint;
                    if (getScaleHintFromRegister(ctx.src[0], tempHint)) {
                      if (tempHint.constReg >= 0 || tempHint.immediateValid) {
                        hasKnownOffsetHint = true;
                        derivedScaleHint.offsetConstReg = tempHint.constReg;
                        derivedScaleHint.offsetCompU = tempHint.compU;
                        derivedScaleHint.offsetCompV = tempHint.compV;
                        derivedScaleHint.offsetFactorU = tempHint.scaleFactorU;
                        derivedScaleHint.offsetFactorV = tempHint.scaleFactorV;
                        derivedScaleHint.offsetImmediateValid = tempHint.immediateValid;
                        if (tempHint.immediateValid) {
                          derivedScaleHint.offsetImmediateU = tempHint.immediateU;
                          derivedScaleHint.offsetImmediateV = tempHint.immediateV;
                        }
                      } else if (tempHint.offsetConstReg >= 0 || tempHint.offsetImmediateValid) {
                        hasKnownOffsetHint = true;
                        derivedScaleHint.offsetConstReg = tempHint.offsetConstReg;
                        derivedScaleHint.offsetCompU = tempHint.offsetCompU;
                        derivedScaleHint.offsetCompV = tempHint.offsetCompV;
                        derivedScaleHint.offsetFactorU = tempHint.offsetFactorU;
                        derivedScaleHint.offsetFactorV = tempHint.offsetFactorV;
                        derivedScaleHint.offsetImmediateValid = tempHint.offsetImmediateValid;
                        if (tempHint.offsetImmediateValid) {
                          derivedScaleHint.offsetImmediateU = tempHint.offsetImmediateU;
                          derivedScaleHint.offsetImmediateV = tempHint.offsetImmediateV;
                        }
                      }
                    }
                  } else {
                    hasKnownOffsetHint = true;
                  }
                } else {
                  // handle `const - uv` as mirrored UV with offset
                  if (derivedScaleHint.constReg < 0 && !derivedScaleHint.immediateValid) {
                    derivedExpressionFlags |= kPsSamplerExprUvTransform;
                    derivedScaleHint.immediateValid = true;
                    derivedScaleHint.immediateU = -1.0f;
                    derivedScaleHint.immediateV = -1.0f;
                    applyOffsetFromConstant(ctx.src[0], derivedScaleHint, +1.0f);
                  } else {
                    // avoid producing incorrect transforms when an existing scale needs negationn
                    derivedScaleHint = PsTexcoordScaleHint{};
                  }
                }

                const bool lhsLooksNonTexcoord =
                  (getCoordProvenanceFromRegister(ctx.src[0]) & kCoordProvNonTexcoord) != 0;
                if (!hasKnownOffsetHint && lhsLooksNonTexcoord)
                  derivedExpressionFlags |= kPsSamplerExprUvAnimated;
                else if (!hasKnownOffsetHint)
                  derivedExpressionFlags |= kPsSamplerExprBlendMath;
              }
            } else {
              derived = mergeSingle({ tc0, tc1 });
            }
            break;
          }
          case DxsoOpcode::Min:
          case DxsoOpcode::Max:
          case DxsoOpcode::Slt:
          case DxsoOpcode::Sge:
          case DxsoOpcode::Dp3:
          case DxsoOpcode::Dp4:
          case DxsoOpcode::Pow:
          case DxsoOpcode::Crs:
          case DxsoOpcode::M4x4:
          case DxsoOpcode::M4x3:
          case DxsoOpcode::M3x4:
          case DxsoOpcode::M3x3:
          case DxsoOpcode::M3x2: {
            writesTrackedTemp = true;
            derivedExpressionFlags |= kPsSamplerExprBlendMath;
            const int32_t tc0 = getTexcoordFromRegister(ctx.src[0]);
            const int32_t tc1 = getTexcoordFromRegister(ctx.src[1]);
            if (tc0 >= 0 && tc1 < 0) {
              assignFromSource(ctx.src[0]);
            } else if (tc1 >= 0 && tc0 < 0) {
              assignFromSource(ctx.src[1]);
            } else {
              derived = mergeSingle({ tc0, tc1 });
            }
            if (derived >= 0 &&
                (op == DxsoOpcode::Dp3 || op == DxsoOpcode::Dp4 ||
                 op == DxsoOpcode::Pow || op == DxsoOpcode::Crs ||
                 op == DxsoOpcode::M4x4 || op == DxsoOpcode::M4x3 ||
                 op == DxsoOpcode::M3x4 || op == DxsoOpcode::M3x3 ||
                 op == DxsoOpcode::M3x2)) {
              derivedExpressionFlags |= kPsSamplerExprUvTransform;
            }
            if (op == DxsoOpcode::Dp3 || op == DxsoOpcode::Dp4 ||
                op == DxsoOpcode::Pow || op == DxsoOpcode::Crs) {
              const uint8_t srcProv =
                getCoordProvenanceFromRegister(ctx.src[0]) |
                getCoordProvenanceFromRegister(ctx.src[1]);
              if ((srcProv & kCoordProvNonTexcoord) != 0 &&
                  (srcProv & kCoordProvTexcoord) == 0) {
                derivedExpressionFlags |= kPsSamplerExprViewDependent;
              }
            }
            if (op == DxsoOpcode::Dp3 || op == DxsoOpcode::Dp4) {
              // normal-map decode signatures:
              //  - self dot-product of a sampled value (the `normalize()` emitted by UE3's
              //    CalcMaterialParameters compiles to `dp3 r.w, n, n; rsq; mul`)
              //  - dot-product consumption of a sign-expanded (`t * 2 - 1`) sampled value
              auto isSignExpandedUse = [&](const DxsoRegister& r) {
                if ((getSamplerValueStateFromRegister(r) & kValueStateSignExpanded) != 0)
                  return true;
                // ps_1_x `_bx2` applies the expansion as a source modifier at the use site
                return r.modifier == DxsoRegModifier::Sign || r.modifier == DxsoRegModifier::SignNeg;
              };
              const uint8_t role0 = getSamplerValueRoleFromRegister(ctx.src[0]);
              const uint8_t role1 = getSamplerValueRoleFromRegister(ctx.src[1]);
              const bool selfDot =
                ctx.src[0].id.type == ctx.src[1].id.type &&
                ctx.src[0].id.num == ctx.src[1].id.num;
              if ((selfDot && role0 != 0) ||
                  (role0 != 0 && isSignExpandedUse(ctx.src[0])) ||
                  (role1 != 0 && isSignExpandedUse(ctx.src[1]))) {
                normalDecodeDetected = true;
              }
            }
            break;
          }
          case DxsoOpcode::Mul: {
            writesTrackedTemp = true;
            derivedExpressionFlags |= kPsSamplerExprBlendMath;
            const int32_t tc0 = getTexcoordFromRegister(ctx.src[0]);
            const int32_t tc1 = getTexcoordFromRegister(ctx.src[1]);
            const bool src0Const = isFloatConstantRegisterType(ctx.src[0].id.type);
            const bool src1Const = isFloatConstantRegisterType(ctx.src[1].id.type);

            if (tc0 >= 0 && tc1 < 0 && src1Const) {
              assignFromSource(ctx.src[0]);
              if (derived >= 0) {
                derivedExpressionFlags |= kPsSamplerExprUvTransform;
                loadScaleHintFromConstant(ctx.src[1], derivedScaleHint);
              }
            } else if (tc1 >= 0 && tc0 < 0 && src0Const) {
              assignFromSource(ctx.src[1]);
              if (derived >= 0) {
                derivedExpressionFlags |= kPsSamplerExprUvTransform;
                loadScaleHintFromConstant(ctx.src[0], derivedScaleHint);
              }
            } else {
              derived = mergeSingle({ tc0, tc1 });
              if (derived < 0) {
                PsTexcoordScaleHint h0, h1;
                if (tryLoadConstantLikeHint(ctx.src[0], h0) &&
                    tryLoadConstantLikeHint(ctx.src[1], h1) &&
                    combineMulHints(h0, h1, constOnlyHint)) {
                  hasConstOnlyHint = true;
                }
              }
            }
            if (derived >= 0)
              derivedExpressionFlags |= kPsSamplerExprUvTransform;
            break;
          }
          case DxsoOpcode::Mad: {
            writesTrackedTemp = true;
            derivedExpressionFlags |= kPsSamplerExprBlendMath;
            const int32_t tc0 = getTexcoordFromRegister(ctx.src[0]);
            const int32_t tc1 = getTexcoordFromRegister(ctx.src[1]);
            const int32_t tc2 = getTexcoordFromRegister(ctx.src[2]);
            const bool src0Const = isFloatConstantRegisterType(ctx.src[0].id.type);
            const bool src1Const = isFloatConstantRegisterType(ctx.src[1].id.type);
            const bool src2Const = isFloatConstantRegisterType(ctx.src[2].id.type);

            if (tc0 >= 0 && tc1 < 0 && src1Const) {
              assignFromSource(ctx.src[0]);
              if (derived >= 0) {
                derivedExpressionFlags |= kPsSamplerExprUvTransform;
                loadScaleHintFromConstant(ctx.src[1], derivedScaleHint);
                if (src2Const) {
                  derivedExpressionFlags |= kPsSamplerExprUvOffset;
                  applyOffsetFromConstant(ctx.src[2], derivedScaleHint, +1.0f);
                }
              }
            } else if (tc1 >= 0 && tc0 < 0 && src0Const) {
              assignFromSource(ctx.src[1]);
              if (derived >= 0) {
                derivedExpressionFlags |= kPsSamplerExprUvTransform;
                loadScaleHintFromConstant(ctx.src[0], derivedScaleHint);
                if (src2Const) {
                  derivedExpressionFlags |= kPsSamplerExprUvOffset;
                  applyOffsetFromConstant(ctx.src[2], derivedScaleHint, +1.0f);
                }
              }
            } else if (tc2 >= 0 && tc0 < 0 && tc1 < 0) {
              assignFromSource(ctx.src[2]);
              if (derived >= 0) {
                PsTexcoordScaleHint mulHint;
                PsTexcoordScaleHint h0, h1;
                if (tryLoadConstantLikeHint(ctx.src[0], h0) &&
                    tryLoadConstantLikeHint(ctx.src[1], h1) &&
                    combineMulHints(h0, h1, mulHint)) {
                  if (mulHint.constReg >= 0 || mulHint.immediateValid) {
                    derivedExpressionFlags |= kPsSamplerExprUvOffset;
                    derivedScaleHint.offsetConstReg = mulHint.constReg;
                    derivedScaleHint.offsetCompU = mulHint.compU;
                    derivedScaleHint.offsetCompV = mulHint.compV;
                    derivedScaleHint.offsetFactorU = mulHint.scaleFactorU;
                    derivedScaleHint.offsetFactorV = mulHint.scaleFactorV;
                    derivedScaleHint.offsetImmediateValid = mulHint.immediateValid;
                    if (mulHint.immediateValid) {
                      derivedScaleHint.offsetImmediateU = mulHint.immediateU;
                      derivedScaleHint.offsetImmediateV = mulHint.immediateV;
                    }
                  }
                }
              }
            } else {
              derived = mergeSingle({ tc0, tc1, tc2 });
            }
            if (derived >= 0)
              derivedExpressionFlags |= kPsSamplerExprUvTransform;
            break;
          }
          case DxsoOpcode::Lrp:
          case DxsoOpcode::Cmp:
          case DxsoOpcode::Dp2Add: {
            writesTrackedTemp = true;
            derived = mergeSingle({
              getTexcoordFromRegister(ctx.src[0]),
              getTexcoordFromRegister(ctx.src[1]),
              getTexcoordFromRegister(ctx.src[2]) });
            if (derived >= 0)
              derivedExpressionFlags |= kPsSamplerExprBlendMath;
            break;
          }
          default:
            break;
          }

          const uint8_t srcSamplerRole0 = classifySamplerValueRoleWithSwizzle(ctx.src[0]);
          const uint8_t srcSamplerRole1 = classifySamplerValueRoleWithSwizzle(ctx.src[1]);
          const uint8_t srcSamplerRole2 = classifySamplerValueRoleWithSwizzle(ctx.src[2]);
          const bool src0UsesSamplerValue = srcSamplerRole0 != 0;
          const bool src1UsesSamplerValue = srcSamplerRole1 != 0;
          const bool src2UsesSamplerValue = srcSamplerRole2 != 0;
          const bool src0Control = (srcSamplerRole0 & kSamplerValueRoleControl) != 0;
          const bool src1Control = (srcSamplerRole1 & kSamplerValueRoleControl) != 0;
          const bool src2Control = (srcSamplerRole2 & kSamplerValueRoleControl) != 0;
          const bool src0Color = (srcSamplerRole0 & kSamplerValueRoleColor) != 0;
          const bool src1Color = (srcSamplerRole1 & kSamplerValueRoleColor) != 0;
          const bool src2Color = (srcSamplerRole2 & kSamplerValueRoleColor) != 0;
          if (src0UsesSamplerValue || src1UsesSamplerValue || src2UsesSamplerValue) {
            switch (op) {
            case DxsoOpcode::Lrp:
            case DxsoOpcode::Cmp:
            case DxsoOpcode::Cnd:
              if (src0UsesSamplerValue &&
                  (src0Control || isScalarSwizzle(ctx.src[0]) || isAlphaScalarSwizzle(ctx.src[0]))) {
                derivedExpressionFlags |= kPsSamplerExprMaskControl;
                derivedSamplerValueRole |= kSamplerValueRoleControl;
              }
              if ((src1UsesSamplerValue && src1Color) || (src2UsesSamplerValue && src2Color)) {
                derivedExpressionFlags |= kPsSamplerExprColorContribution;
                derivedSamplerValueRole |= kSamplerValueRoleColor;
              }
              if ((src1UsesSamplerValue && !src1Color) || (src2UsesSamplerValue && !src2Color)) {
                derivedExpressionFlags |= kPsSamplerExprMaskControl;
                derivedSamplerValueRole |= kSamplerValueRoleControl;
              }
              break;
            case DxsoOpcode::Slt:
            case DxsoOpcode::Sge:
            case DxsoOpcode::TexKill:
              if (src0Control || src1Control || src2Control ||
                  isAlphaScalarSwizzle(ctx.src[0]) ||
                  isAlphaScalarSwizzle(ctx.src[1]) ||
                  isAlphaScalarSwizzle(ctx.src[2])) {
                derivedExpressionFlags |= kPsSamplerExprMaskControl;
                derivedSamplerValueRole |= kSamplerValueRoleControl;
              } else {
                derivedExpressionFlags |= kPsSamplerExprColorContribution;
                derivedSamplerValueRole |= kSamplerValueRoleColor;
              }
              break;
            case DxsoOpcode::Mul:
            case DxsoOpcode::Mad:
            case DxsoOpcode::Add:
            case DxsoOpcode::Sub: {
              const bool hasControl = src0Control || src1Control || src2Control;
              const bool hasColor = src0Color || src1Color || src2Color;
              if (hasControl && !hasColor) {
                derivedExpressionFlags |= kPsSamplerExprMaskControl;
                derivedSamplerValueRole |= kSamplerValueRoleControl;
              } else if (hasControl && hasColor) {
                derivedExpressionFlags |= kPsSamplerExprMaskControl;
                derivedExpressionFlags |= kPsSamplerExprColorContribution;
                derivedSamplerValueRole |= uint8_t(kSamplerValueRoleControl | kSamplerValueRoleColor);
              } else {
                derivedExpressionFlags |= kPsSamplerExprColorContribution;
                derivedSamplerValueRole |= kSamplerValueRoleColor;
              }
              break;
            }
            case DxsoOpcode::Dp3:
            case DxsoOpcode::Dp4:
            case DxsoOpcode::Dp2Add:
            case DxsoOpcode::Crs:
            case DxsoOpcode::Nrm:
            case DxsoOpcode::M4x4:
            case DxsoOpcode::M4x3:
            case DxsoOpcode::M3x4:
            case DxsoOpcode::M3x3:
            case DxsoOpcode::M3x2:
              // dot products / normalizes / matrix transforms collapse a sampled color vector
              // into direction or coefficient data. Terminate value tracking here: the result is
              // neither this sampler's color (no output-color credit) nor a mask of it (no
              // mask-control penalty for innocent downstream mixing).
              derivedSamplerValueRole = 0;
              break;
            default:
              derivedExpressionFlags |= kPsSamplerExprColorContribution;
              derivedSamplerValueRole |= kSamplerValueRoleColor;
              break;
            }
          }

          // normal-map unpack state for this sampler's value: recognize `t * 2 - 1`
          // (UE3's TextureSample UnpackMin/UnpackMax expansion) and its split forms
          uint8_t derivedValueState = 0;
          switch (op) {
          case DxsoOpcode::Mov:
            derivedValueState = getSamplerValueStateFromRegister(ctx.src[0]);
            break;
          case DxsoOpcode::Mul: {
            const bool role0 = getSamplerValueRoleFromRegister(ctx.src[0]) != 0;
            const bool role1 = getSamplerValueRoleFromRegister(ctx.src[1]) != 0;
            if (role0 && isDefConstNearRgb(ctx.src[1], 2.0f)) {
              derivedValueState |= kValueStateScaledX2;
              if ((getSamplerValueStateFromRegister(ctx.src[0]) & kValueStateBiasedHalf) != 0)
                derivedValueState |= kValueStateSignExpanded; // (t - 0.5) * 2
            } else if (role1 && isDefConstNearRgb(ctx.src[0], 2.0f)) {
              derivedValueState |= kValueStateScaledX2;
              if ((getSamplerValueStateFromRegister(ctx.src[1]) & kValueStateBiasedHalf) != 0)
                derivedValueState |= kValueStateSignExpanded;
            }
            break;
          }
          case DxsoOpcode::Add:
          case DxsoOpcode::Sub: {
            const bool isSub = op == DxsoOpcode::Sub;
            const uint8_t state0 = getSamplerValueStateFromRegister(ctx.src[0]);
            const uint8_t state1 = getSamplerValueStateFromRegister(ctx.src[1]);
            const bool role0 = getSamplerValueRoleFromRegister(ctx.src[0]) != 0;
            const bool role1 = getSamplerValueRoleFromRegister(ctx.src[1]) != 0;
            // t * 2 - 1 completing a sign expansion
            if (role0 && (state0 & kValueStateScaledX2) != 0 &&
                isDefConstNearRgb(ctx.src[1], isSub ? 1.0f : -1.0f))
              derivedValueState |= kValueStateSignExpanded;
            // commuted / mirrored forms: (-1) + t * 2, 1 - t * 2
            if (role1 && (state1 & kValueStateScaledX2) != 0 &&
                isDefConstNearRgb(ctx.src[0], isSub ? 1.0f : -1.0f))
              derivedValueState |= kValueStateSignExpanded;
            // t - 0.5 halfway through a (t - 0.5) * 2 expansion
            if (role0 && isDefConstNearRgb(ctx.src[1], isSub ? 0.5f : -0.5f))
              derivedValueState |= kValueStateBiasedHalf;
            if (role1 && isDefConstNearRgb(ctx.src[0], isSub ? 0.5f : -0.5f))
              derivedValueState |= kValueStateBiasedHalf;
            break;
          }
          case DxsoOpcode::Mad: {
            const bool role0 = getSamplerValueRoleFromRegister(ctx.src[0]) != 0;
            const bool role1 = getSamplerValueRoleFromRegister(ctx.src[1]) != 0;
            const bool offsetIsMinusOne = isDefConstNearRgb(ctx.src[2], -1.0f);
            if (offsetIsMinusOne &&
                ((role0 && isDefConstNearRgb(ctx.src[1], 2.0f)) ||
                 (role1 && isDefConstNearRgb(ctx.src[0], 2.0f))))
              derivedValueState |= kValueStateSignExpanded; // mad(t, 2, -1)
            break;
          }
          default:
            break;
          }

          // lightmap/lighting-constant provenance flows through all tracked arithmetic
          const uint8_t derivedAnchorBits = uint8_t(
            getAnchorBitsFromRegister(ctx.src[0]) |
            getAnchorBitsFromRegister(ctx.src[1]) |
            getAnchorBitsFromRegister(ctx.src[2]));

          // the HLSL compiler packs scalar results into spare lanes of live registers
          // (e.g. `dp3 r0.w, n, n` while r0.xyz still holds a tracked color value); a write
          // that touches no rgb lane merges role/anchor and leaves the rgb unpack state alone
          auto writeSamplerValueTracking = [&] {
            if (maskWritesRgb(ctx.dst.mask)) {
              tempSamplerValueRole[ctx.dst.id.num] = derivedSamplerValueRole;
              tempSamplerValueState[ctx.dst.id.num] = derivedValueState;
              tempAnchorBits[ctx.dst.id.num] = derivedAnchorBits;
            } else {
              tempSamplerValueRole[ctx.dst.id.num] |= derivedSamplerValueRole;
              tempAnchorBits[ctx.dst.id.num] |= derivedAnchorBits;
            }
          };

          if (derived >= 0) {
            tempToTexcoord[ctx.dst.id.num] = int8_t(derived);
            tempCoordExpressionFlags[ctx.dst.id.num] = derivedExpressionFlags;
            writeSamplerValueTracking();
            tempCoordProvenance[ctx.dst.id.num] =
              getCoordProvenanceFromRegister(ctx.src[0]) |
              getCoordProvenanceFromRegister(ctx.src[1]) |
              getCoordProvenanceFromRegister(ctx.src[2]) |
              kCoordProvTexcoord;
            if (derivedScaleHint.constReg >= 0 || derivedScaleHint.immediateValid ||
                derivedScaleHint.offsetConstReg >= 0 || derivedScaleHint.offsetImmediateValid)
              tempToScaleHint[ctx.dst.id.num] = derivedScaleHint;
            else
              tempToScaleHint[ctx.dst.id.num] = PsTexcoordScaleHint{};
          } else if (writesTrackedTemp) {
            tempToTexcoord[ctx.dst.id.num] = -1;
            tempCoordExpressionFlags[ctx.dst.id.num] = derivedExpressionFlags;
            writeSamplerValueTracking();
            tempCoordProvenance[ctx.dst.id.num] =
              getCoordProvenanceFromRegister(ctx.src[0]) |
              getCoordProvenanceFromRegister(ctx.src[1]) |
              getCoordProvenanceFromRegister(ctx.src[2]);
            if (hasConstOnlyHint &&
                (constOnlyHint.constReg >= 0 || constOnlyHint.immediateValid)) {
              tempToScaleHint[ctx.dst.id.num] = constOnlyHint;
            } else {
              tempToScaleHint[ctx.dst.id.num] = PsTexcoordScaleHint{};
            }
          }
        }

        // values that only feed lighting/coordinate math (dot products, kill tests, masks)
        // carry a control-only role by the time they reach the output and are not counted
        if (!reachesOutputColor &&
            ctx.dst.id.type == DxsoRegisterType::ColorOut &&
            ctx.dst.id.num == 0 &&
            maskWritesRgb(ctx.dst.mask)) {
          for (const DxsoRegister& srcReg : { ctx.src[0], ctx.src[1], ctx.src[2] }) {
            if ((classifySamplerValueRoleWithSwizzle(srcReg) & kSamplerValueRoleColor) != 0) {
              reachesOutputColor = true;
              break;
            }
          }
        }

        uint32_t sampledSampler = ~0u;
        uint8_t sampleOpSemanticFlags = 0;
        uint16_t sampleOpExpressionFlags = 0;
        DxsoRegister coordRegStorage;
        const DxsoRegister* coordReg = nullptr;
        switch (op) {
        case DxsoOpcode::Tex:
          if (info.majorVersion() >= 2) {
            sampledSampler = ctx.src[1].id.num;
            coordReg = &ctx.src[0];
          } else if (info.majorVersion() == 1 && info.minorVersion() == 4) {
            sampledSampler = ctx.dst.id.num;
            coordReg = &ctx.src[0];
          } else {
            sampledSampler = ctx.dst.id.num;
            coordRegStorage = ctx.dst;
            coordRegStorage.id.type = DxsoRegisterType::PixelTexcoord;
            coordRegStorage.id.num = ctx.dst.id.num;
            coordRegStorage.swizzle = DxsoRegSwizzle(0, 1, 2, 3);
            coordReg = &coordRegStorage;
          }
          break;
        case DxsoOpcode::TexLdd:
        case DxsoOpcode::TexLdl:
          sampledSampler = ctx.src[1].id.num;
          coordReg = &ctx.src[0];
          sampleOpExpressionFlags |= kPsSamplerExprBlendMath;
          break;
        case DxsoOpcode::TexBem:
        case DxsoOpcode::TexBemL:
          sampledSampler = ctx.dst.id.num;
          coordRegStorage = ctx.dst;
          coordRegStorage.id.type = DxsoRegisterType::PixelTexcoord;
          coordRegStorage.id.num = ctx.dst.id.num;
          coordRegStorage.swizzle = DxsoRegSwizzle(0, 1, 2, 3);
          coordReg = &coordRegStorage;
          sampleOpSemanticFlags |= kPsSamplerSemanticNonDiffuse;
          sampleOpExpressionFlags |= kPsSamplerExprUvOffset;
          break;
        case DxsoOpcode::TexReg2Ar:
        case DxsoOpcode::TexReg2Gb:
        case DxsoOpcode::TexReg2Rgb:
          sampledSampler = ctx.dst.id.num;
          coordReg = &ctx.src[0];
          sampleOpExpressionFlags |= kPsSamplerExprUvTransform;
          break;
        case DxsoOpcode::TexM3x2Tex:
        case DxsoOpcode::TexM3x3Tex:
        case DxsoOpcode::TexDp3Tex:
          sampledSampler = ctx.dst.id.num;
          coordReg = &ctx.src[0];
          sampleOpExpressionFlags |= kPsSamplerExprUvTransform;
          break;
        case DxsoOpcode::TexM3x3Spec:
        case DxsoOpcode::TexM3x3VSpec:
          sampledSampler = ctx.dst.id.num;
          coordReg = &ctx.src[0];
          sampleOpSemanticFlags |= kPsSamplerSemanticNonDiffuse;
          sampleOpExpressionFlags |= kPsSamplerExprUvTransform;
          sampleOpExpressionFlags |= kPsSamplerExprViewDependent;
          break;
        case DxsoOpcode::TexM3x2Depth:
        case DxsoOpcode::TexDepth:
          sampledSampler = ctx.dst.id.num;
          coordReg = &ctx.src[0];
          sampleOpSemanticFlags |= kPsSamplerSemanticEngineAuxiliary;
          break;
        default:
          break;
        }

        // a sample write replaces the destination temp's contents: drop stale tracking, or the
        // heavy r# register reuse would misattribute another sampler's usage (e.g. a normal-map
        // unpack) to the sampler being analyzed
        if (coordReg != nullptr &&
            (ctx.dst.id.type == DxsoRegisterType::Temp || ctx.dst.id.type == DxsoRegisterType::TempFloat16) &&
            ctx.dst.id.num < tempSamplerValueRole.size()) {
          if (sampledSampler != samplerIdx) {
            tempSamplerValueRole[ctx.dst.id.num] = 0;
            tempSamplerValueState[ctx.dst.id.num] = 0;
          }
          tempAnchorBits[ctx.dst.id.num] =
            (sampledSampler < 32u && ((anchorLightmapSamplerMask >> sampledSampler) & 1u) != 0)
              ? kAnchorLightmapValue
              : 0u;
        }

        if (coordReg != nullptr && sampledSampler == samplerIdx) {
          result.sampleCount = result.sampleCount < std::numeric_limits<uint16_t>::max()
            ? uint16_t(result.sampleCount + 1u)
            : std::numeric_limits<uint16_t>::max();

          // Accumulated across every sample of this sampler, since a material can read the same
          // texture through more than one coordinate (a sub-UV blend reads two atlas frames).
          orConstDepsFromRegister(*coordReg, sampledCoordConstDeps);

          if ((ctx.dst.id.type == DxsoRegisterType::Temp || ctx.dst.id.type == DxsoRegisterType::TempFloat16) &&
              ctx.dst.id.num < tempSamplerValueRole.size()) {
            uint8_t sampleRole = kSamplerValueRoleColor;
            if ((sampleOpSemanticFlags & (kPsSamplerSemanticEngineAuxiliary | kPsSamplerSemanticNonDiffuse)) != 0)
              sampleRole |= kSamplerValueRoleControl;
            tempSamplerValueRole[ctx.dst.id.num] = sampleRole;
            tempSamplerValueState[ctx.dst.id.num] = 0;
          }

          const DxsoRegister* swizzleReg = coordReg;
          int32_t tc = getTexcoordFromRegister(*swizzleReg);
          uint8_t coordProvenance = getCoordProvenanceFromRegister(*coordReg);
          uint16_t coordExpressionFlags =
            uint16_t(getCoordExpressionFlagsFromRegister(*coordReg) | sampleOpExpressionFlags);
          if ((sampleOpSemanticFlags & kPsSamplerSemanticNonDiffuse) != 0)
            coordExpressionFlags |= kPsSamplerExprMaskControl;
          if ((sampleOpSemanticFlags & kPsSamplerSemanticEngineAuxiliary) == 0)
            coordExpressionFlags |= kPsSamplerExprColorContribution;
          result.semanticFlags |= sampleOpSemanticFlags;

          const std::array<DxsoRegister, 4> fallbackRegs = {
            ctx.src[0], ctx.src[1], ctx.src[2], ctx.dst
          };

          if ((coordProvenance & kCoordProvTexcoord) == 0 || coordExpressionFlags == 0) {
            for (const DxsoRegister& fallbackReg : fallbackRegs) {
              coordProvenance |= getCoordProvenanceFromRegister(fallbackReg);
              coordExpressionFlags |= getCoordExpressionFlagsFromRegister(fallbackReg);
              if ((coordProvenance & kCoordProvTexcoord) != 0)
                break;
            }
          }

          if (tc < 0) {
            for (const DxsoRegister& fallbackReg : fallbackRegs) {
              const int32_t fallbackTc = getTexcoordFromRegister(fallbackReg);
              if (fallbackTc >= 0) {
                tc = fallbackTc;
                swizzleReg = &fallbackReg;
                break;
              }
            }
          }

          if ((coordProvenance & kCoordProvTexcoord) != 0)
            texcoordDerivedSampleCount++;
          else if ((coordProvenance & kCoordProvNonTexcoord) != 0)
            nonTexcoordDerivedSampleCount++;

          const bool coordUsesTexcoord = (coordProvenance & kCoordProvTexcoord) != 0;
          const bool coordUsesNonTexcoord = (coordProvenance & kCoordProvNonTexcoord) != 0;
          if (coordUsesNonTexcoord && !coordUsesTexcoord)
            coordExpressionFlags |= kPsSamplerExprViewDependent;

          if (tc >= 0) {
            if (uint32_t(tc) < texcoordUseCount.size()) {
              texcoordUseCount[uint32_t(tc)]++;

              const uint32_t compU = swizzleReg->swizzle[0] & 0x3;
              const uint32_t compV = swizzleReg->swizzle[1] & 0x3;
              const uint32_t tcIdx = uint32_t(tc);
              const uint32_t pairIdx = ((compU & 0x3u) << 2) | (compV & 0x3u);
              uint16_t& pairCount = texcoordCoordPairUseCount[tcIdx][pairIdx];
              if (pairCount < std::numeric_limits<uint16_t>::max())
                pairCount = uint16_t(pairCount + 1u);
            }
          }

          PsTexcoordScaleHint sampleScaleHint;
          if (tc >= 0 && uint32_t(tc) < perTexcoordScaleHints.size() &&
              getScaleHintFromRegister(*coordReg, sampleScaleHint)) {
            if (sampleScaleHint.constReg >= 0 || sampleScaleHint.immediateValid)
              coordExpressionFlags |= kPsSamplerExprUvTransform;
            if (sampleScaleHint.offsetConstReg >= 0 || sampleScaleHint.offsetImmediateValid)
              coordExpressionFlags |= kPsSamplerExprUvOffset;
            auto& agg = perTexcoordScaleHints[uint32_t(tc)];
            if (!agg.valid) {
              agg.valid = true;
              agg.hint = sampleScaleHint;
            } else if (!isSameScaleHint(agg.hint, sampleScaleHint)) {
              agg.conflict = true;
            }
          }

          sampledCoordExpressionFlags |= coordExpressionFlags;
        }
      }

      int32_t foundTexcoord = -1;
      uint32_t bestCount = 0;
      bool countTie = false;
      for (uint32_t i = 0; i < texcoordUseCount.size(); i++) {
        const uint32_t c = texcoordUseCount[i];
        if (c > bestCount) {
          bestCount = c;
          foundTexcoord = int32_t(i);
          countTie = false;
        } else if (c > 0 && c == bestCount) {
          countTie = true;
        }
      }
      if (countTie) {
        int32_t hintedCandidate = -1;
        for (uint32_t i = 0; i < texcoordUseCount.size(); i++) {
          if (texcoordUseCount[i] != bestCount || bestCount == 0)
            continue;

          const auto& agg = perTexcoordScaleHints[i];
          if (!(agg.valid && !agg.conflict))
            continue;

          if (hintedCandidate < 0) {
            hintedCandidate = int32_t(i);
          } else if (hintedCandidate != int32_t(i)) {
            hintedCandidate = -1;
            break;
          }
        }

        if (hintedCandidate >= 0) {
          foundTexcoord = hintedCandidate;
          countTie = false;
        } else {
          foundTexcoord = -1;
        }
      }

      result.texcoord = foundTexcoord;
      if (foundTexcoord >= 0 && uint32_t(foundTexcoord) < perTexcoordScaleHints.size()) {
        const auto& agg = perTexcoordScaleHints[uint32_t(foundTexcoord)];
        const uint32_t tcIdx = uint32_t(foundTexcoord);
        {
          uint32_t totalPairCount = 0;
          uint32_t bestPairCount = 0;
          uint32_t secondBestPairCount = 0;
          uint32_t bestPairIdx = 0;
          for (uint32_t pairIdx = 0; pairIdx < texcoordCoordPairUseCount[tcIdx].size(); pairIdx++) {
            const uint32_t pairCount = texcoordCoordPairUseCount[tcIdx][pairIdx];
            totalPairCount += pairCount;
            if (pairCount > bestPairCount) {
              secondBestPairCount = bestPairCount;
              bestPairCount = pairCount;
              bestPairIdx = pairIdx;
            } else if (pairCount > secondBestPairCount) {
              secondBestPairCount = pairCount;
            }
          }

          if (bestPairCount > 0) {
            const bool hasClearWinner = bestPairCount > secondBestPairCount;
            const bool hasStrongMajority = (bestPairCount * 3u) >= (totalPairCount * 2u);
            if (hasClearWinner || hasStrongMajority) {
              result.coordCompValid = true;
              result.coordCompU = uint8_t((bestPairIdx >> 2) & 0x3u);
              result.coordCompV = uint8_t(bestPairIdx & 0x3u);
            }
          }
        }

        if (agg.valid && !agg.conflict) {
          result.scaleConstReg = agg.hint.constReg;
          result.scaleConstCompU = agg.hint.compU;
          result.scaleConstCompV = agg.hint.compV;
          result.scaleFactorU = agg.hint.scaleFactorU;
          result.scaleFactorV = agg.hint.scaleFactorV;
          result.scaleImmediateValid = agg.hint.immediateValid;
          result.scaleImmediateU = agg.hint.immediateU;
          result.scaleImmediateV = agg.hint.immediateV;
          result.offsetConstReg = agg.hint.offsetConstReg;
          result.offsetConstCompU = agg.hint.offsetCompU;
          result.offsetConstCompV = agg.hint.offsetCompV;
          result.offsetFactorU = agg.hint.offsetFactorU;
          result.offsetFactorV = agg.hint.offsetFactorV;
          result.offsetImmediateValid = agg.hint.offsetImmediateValid;
          result.offsetImmediateU = agg.hint.offsetImmediateU;
          result.offsetImmediateV = agg.hint.offsetImmediateV;
        }
      }

      for (uint32_t reg = 0; reg < caps::MaxFloatConstantsPS; reg++) {
        if ((sampledCoordConstDeps[reg / 64u] >> (reg % 64u)) & 1ull)
          result.coordConstRegs.push_back(reg);
      }

      result.expressionFlags = sampledCoordExpressionFlags;
      if (result.scaleConstReg >= 0 || result.scaleImmediateValid)
        result.expressionFlags |= kPsSamplerExprUvTransform;
      if (result.offsetConstReg >= 0 || result.offsetImmediateValid)
        result.expressionFlags |= kPsSamplerExprUvOffset;
      if (result.coordCompValid && (result.coordCompU != 0 || result.coordCompV != 1))
        result.expressionFlags |= kPsSamplerExprUvTransform;
      if ((result.expressionFlags & (kPsSamplerExprUvTransform | kPsSamplerExprUvOffset)) != 0 &&
          result.sampleCount >= 2) {
        result.expressionFlags |= kPsSamplerExprUvAnimated;
      }
      if ((result.expressionFlags & kPsSamplerExprMaskControl) != 0 &&
          (result.semanticFlags & (kPsSamplerSemanticEngineAuxiliary | kPsSamplerSemanticLightmap)) == 0) {
        result.semanticFlags |= kPsSamplerSemanticNonDiffuse;
      }
      if ((result.expressionFlags & kPsSamplerExprViewDependent) != 0 &&
          (result.semanticFlags & (kPsSamplerSemanticEngineAuxiliary | kPsSamplerSemanticLightmap)) == 0) {
        result.semanticFlags |= kPsSamplerSemanticNonDiffuse;
      }
      // note: the decode flag intentionally does not imply kPsSamplerSemanticNonDiffuse here.
      // Complex lit shaders remap many color terms by *2-1 (fresnel/rim/mask math), producing
      // false positives on gamma-decoded textures; scoring applies the flag only for samplers
      // bound without D3DSAMP_SRGBTEXTURE (UE3 always imports normal maps with SRGB=0).
      if (normalDecodeDetected)
        result.expressionFlags |= kPsSamplerExprNormalDecode;
      // ps_1_x has no oC0 register; the final value of r0 is the output color
      if (!reachesOutputColor && info.majorVersion() == 1 &&
          (tempSamplerValueRole[0] & kSamplerValueRoleColor) != 0) {
        reachesOutputColor = true;
      }
      if (reachesOutputColor)
        result.expressionFlags |= kPsSamplerExprReachesOutputColor;
      if (diffuseAnchorDetected)
        result.expressionFlags |= kPsSamplerExprDiffuseAnchor;

      {
        const auto& ctab = decoder.getCtabInfo();
        if (ctab.m_size != 0) {
          for (const auto& c : ctab.m_constantData) {
            if (c.registerSet != kD3dxRegisterSetSampler || c.registerCount == 0)
              continue;

            const uint64_t regBegin = c.registerIndex;
            const uint64_t regEnd = regBegin + c.registerCount;
            if (uint64_t(samplerIdx) >= regBegin && uint64_t(samplerIdx) < regEnd) {
              const uint8_t semanticFlagsFromName = classifyPixelSamplerSemanticFlags(c.name);
              result.semanticFlags |= semanticFlagsFromName;
              result.expressionFlags |= classifyPixelSamplerExpressionFlagsFromName(c.name);
              // a generic material-texture name (Texture2D_*) deliberately does not imply color
              // contribution; opcode dataflow decides that (see kPsSamplerExprReachesOutputColor)
              if ((semanticFlagsFromName & (kPsSamplerSemanticEngineAuxiliary | kPsSamplerSemanticNonDiffuse)) != 0)
                result.expressionFlags |= kPsSamplerExprMaskControl;
            }
          }

          if (result.scaleConstReg >= 0 || result.offsetConstReg >= 0) {
            auto toLower = [](const std::string& s) {
              std::string out;
              out.reserve(s.size());
              for (const char c : s)
                out.push_back(char(std::tolower(static_cast<unsigned char>(c))));
              return out;
            };
            auto contains = [](const std::string& s, const char* token) {
              return s.find(token) != std::string::npos;
            };
            auto isLikelyTimeDrivenName = [&](const std::string& lowerName) {
              return
                contains(lowerName, "time") ||
                contains(lowerName, "gametime") ||
                contains(lowerName, "realtime") ||
                contains(lowerName, "sine") ||
                contains(lowerName, "cosine") ||
                contains(lowerName, "panner") ||
                contains(lowerName, "rotator") ||
                contains(lowerName, "rotation") ||
                contains(lowerName, "flipbook") ||
                contains(lowerName, "subuv") ||
                contains(lowerName, "phase") ||
                contains(lowerName, "oscillat") ||
                contains(lowerName, "wind");
            };
            auto constantRangeContainsRegister = [](const DxsoCtab::Constant& c, const int32_t reg) {
              if (reg < 0 || c.registerCount == 0 || c.registerSet > 2u)
                return false;
              const int64_t begin = int64_t(c.registerIndex);
              const int64_t end = begin + int64_t(c.registerCount);
              const int64_t r = int64_t(reg);
              return r >= begin && r < end;
            };

            for (const auto& c : ctab.m_constantData) {
              if (!isLikelyTimeDrivenName(toLower(c.name)))
                continue;
              if (constantRangeContainsRegister(c, result.scaleConstReg) ||
                  constantRangeContainsRegister(c, result.offsetConstReg)) {
                result.expressionFlags |= kPsSamplerExprUvTimeDriven;
                result.expressionFlags |= kPsSamplerExprUvAnimated;
                break;
              }
            }
          }
        }
      }

      // expression fallback tagging for generic/stripped sampler names
      //if the coordinate path clearly looks like UV math and originates from TEXCOORD
      // treat it as a material texture path unless already identified as aux/lightmap
      const bool hasUvExpression =
        (result.expressionFlags &
         (kPsSamplerExprUvTransform | kPsSamplerExprUvOffset | kPsSamplerExprUvAnimated)) != 0;
      if (result.sampleCount > 0 &&
          result.texcoord >= 0 &&
          hasUvExpression &&
          (result.semanticFlags & (kPsSamplerSemanticEngineAuxiliary | kPsSamplerSemanticLightmap)) == 0) {
        result.semanticFlags |= kPsSamplerSemanticMaterialTexture;
      }

      // fallback semantic tagging for shaders that use generic/stripped sampler names
      // if a sampler is sampled only from non-texcoord source, treat it as an engine-auxiliary path
      // this helps deprioritise screen/postprocess samplers that do not flow from mesh UVs
      if (result.sampleCount > 0 &&
          result.texcoord < 0 &&
          texcoordDerivedSampleCount == 0 &&
          nonTexcoordDerivedSampleCount > 0 &&
          (result.semanticFlags & kPsSamplerSemanticMaterialTexture) == 0) {
        result.semanticFlags |= kPsSamplerSemanticEngineAuxiliary;
      }

      return result;
    }


    bool tryExtractUe3WorldToViewAndProjectionFromShaderConstants(
      const D3D9ShaderConstantsVSSoftware& vsConsts,
      const uint32_t viewProjRegisterBase,
      const uint32_t viewOriginRegister,
      Matrix4& outWorldToView,
      Matrix4& outViewToProjection,
      bool* outUsedTranspose = nullptr,
      float* outReconstructionError = nullptr) {

      // UE3 uploads ViewProjectionMatrix via SetVertexShaderConstantF as 4 consecutive float4 registers
      // These values are the raw shader constants and in UE3/HLSL this matrix is typically treated as column-major
      // We derive a stable (world to view, view to projection) pair by
      // 1 treating the provided matrix as a world to projection transform
      // 2 unprojecting a few NDC points to recover view orientation (using camera position)
      // 3 deriving a pure projection matrix and validating it via projection decomposition

      if (viewProjRegisterBase + 3 >= caps::MaxFloatConstantsSoftware)
        return false;
      if (viewOriginRegister >= caps::MaxFloatConstantsSoftware)
        return false;

      Matrix4 worldToProjection;
      worldToProjection[0] = vsConsts.fConsts[viewProjRegisterBase + 0];
      worldToProjection[1] = vsConsts.fConsts[viewProjRegisterBase + 1];
      worldToProjection[2] = vsConsts.fConsts[viewProjRegisterBase + 2];
      worldToProjection[3] = vsConsts.fConsts[viewProjRegisterBase + 3];

      const Vector3 camPos = vsConsts.fConsts[viewOriginRegister].xyz();

      // Quick reject: all-zero matrices show up during some initialization paths
      constexpr float kEps = 1e-6f;
      if (lengthSqr(worldToProjection[0].xyz()) < kEps &&
          lengthSqr(worldToProjection[1].xyz()) < kEps &&
          lengthSqr(worldToProjection[2].xyz()) < kEps &&
          lengthSqr(worldToProjection[3].xyz()) < kEps) {
        return false;
      }

      auto validateProjection = [](const Matrix4& viewToProjection, DecomposeProjectionParams& outParams) {
        decomposeProjection(viewToProjection, outParams);

        const bool finite =
          std::isfinite(outParams.fov) &&
          std::isfinite(outParams.aspectRatio) &&
          std::isfinite(outParams.nearPlane) &&
          std::isfinite(outParams.farPlane) &&
          std::isfinite(outParams.shearX) &&
          std::isfinite(outParams.shearY);
        if (!finite)
          return false;

        if (outParams.fov < 0.001f)
          return false;
        if (std::abs(outParams.shearX) > 0.01f)
          return false;
        if (outParams.nearPlane <= 0.0f)
          return false;
        if (outParams.farPlane <= outParams.nearPlane)
          return false;

        return true;
      };

      auto matrixL1Error = [](const Matrix4& a, const Matrix4& b) {
        float err = 0.0f;
        for (uint32_t c = 0; c < 4; c++) {
          for (uint32_t r = 0; r < 4; r++) {
            err += std::abs(a[c][r] - b[c][r]);
          }
        }
        return err;
      };

      auto tryBuild = [&](const Matrix4& candidateWorldToProjection, Matrix4& outWorldToViewLocal, Matrix4& outViewToProjectionLocal) -> bool {
        // avoid attempting to invert singular matrices
        {
          constexpr double kDetEps = 1e-24;
          const double det = determinant(candidateWorldToProjection);
          if (!std::isfinite(det) || std::abs(det) <= kDetEps)
            return false;
        }

        // invert world to projection to unproject a few NDC points
        Matrix4 invWorldToProjection;
        invWorldToProjection = inverse(candidateWorldToProjection);

        auto unprojectNdc = [&](float ndcX, float ndcY, float ndcZ, Vector3& outWorldPos) -> bool {
          const Vector4 clip(ndcX, ndcY, ndcZ, 1.0f);
          const Vector4 worldH = invWorldToProjection * clip;
          if (!std::isfinite(worldH.w) || std::abs(worldH.w) < kEps)
            return false;
          const float invW = 1.0f / worldH.w;
          outWorldPos = worldH.xyz() * invW;
          return std::isfinite(outWorldPos.x) && std::isfinite(outWorldPos.y) && std::isfinite(outWorldPos.z);
        };

        // reference - D3D NDC: x/y in [-1, 1], z in [0, 1]
        constexpr float ndcZ = 0.5f;
        Vector3 worldCenter, worldUp, worldRight;
        if (!unprojectNdc(0.0f, 0.0f, ndcZ, worldCenter))
          return false;
        if (!unprojectNdc(0.0f, 1.0f, ndcZ, worldUp))
          return false;
        if (!unprojectNdc(1.0f, 0.0f, ndcZ, worldRight))
          return false;

        Vector3 forward = worldCenter - camPos;
        if (lengthSqr(forward) < kEps)
          return false;
        forward = normalize(forward);

        Vector3 upHint = worldUp - worldCenter;
        if (lengthSqr(upHint) < kEps)
          upHint = Vector3(0.0f, 1.0f, 0.0f);
        else
          upHint = normalize(upHint);

        // construct an orthonormal basis, we use the unprojected "up" direction as a hint to fix roll
        Vector3 right = cross(upHint, forward);
        if (lengthSqr(right) < kEps)
          return false;
        right = normalize(right);
        Vector3 up = normalize(cross(forward, right));

        auto buildViewToWorld = [&](const Vector3& r, const Vector3& u, const Vector3& f) {
          Matrix4 viewToWorld;
          viewToWorld[0] = Vector4(r.x, r.y, r.z, 0.0f);
          viewToWorld[1] = Vector4(u.x, u.y, u.z, 0.0f);
          viewToWorld[2] = Vector4(f.x, f.y, f.z, 0.0f);
          viewToWorld[3] = Vector4(camPos.x, camPos.y, camPos.z, 1.0f);
          return viewToWorld;
        };

        // 1st attempt - assume +Z is forward in view space but if the unprojected point ends up behind the camera, flip the forward axis and rebuild
        Matrix4 viewToWorld = buildViewToWorld(right, up, forward);
        Matrix4 worldToView = inverseAffine(viewToWorld);
        const Vector4 centerInView = worldToView * Vector4(worldCenter.x, worldCenter.y, worldCenter.z, 1.0f);
        if (std::isfinite(centerInView.z) && centerInView.z < 0.0f) {
          forward = -forward;
          right = cross(upHint, forward);
          if (lengthSqr(right) < kEps)
            return false;
          right = normalize(right);
          up = normalize(cross(forward, right));
          viewToWorld = buildViewToWorld(right, up, forward);
          worldToView = inverseAffine(viewToWorld);
        }

        // candidate projections - handle possible multiplication order conventions by picking the one that
        // yields a valid projection on decomposition and best reconstructs the original matrix
        Matrix4 bestViewToProjection;
        float bestErr = std::numeric_limits<float>::infinity();
        bool found = false;

        {
          Matrix4 viewToProjection = candidateWorldToProjection * viewToWorld;
          DecomposeProjectionParams projParams {};
          if (validateProjection(viewToProjection, projParams)) {
            const Matrix4 recon = viewToProjection * worldToView;
            const float err = matrixL1Error(recon, candidateWorldToProjection);
            if (err < bestErr) {
              bestErr = err;
              bestViewToProjection = viewToProjection;
              found = true;
            }
          }
        }

        {
          Matrix4 viewToProjection = viewToWorld * candidateWorldToProjection;
          DecomposeProjectionParams projParams {};
          if (validateProjection(viewToProjection, projParams)) {
            const Matrix4 recon = worldToView * viewToProjection;
            const float err = matrixL1Error(recon, candidateWorldToProjection);
            if (err < bestErr) {
              bestErr = err;
              bestViewToProjection = viewToProjection;
              found = true;
            }
          }
        }

        if (!found)
          return false;

        outWorldToViewLocal = worldToView;
        outViewToProjectionLocal = bestViewToProjection;
        return true;
      };

      // try both the raw matrix and its transpose, prefer the one that best reconstructs the original world to projection
      struct CandidateResult {
        bool ok = false;
        float error = std::numeric_limits<float>::infinity();
        Matrix4 worldToView;
        Matrix4 viewToProjection;
      };

      auto evalCandidate = [&](const Matrix4& candidateWorldToProjection) -> CandidateResult {
        CandidateResult r;
        Matrix4 w2v;
        Matrix4 v2p;
        if (!tryBuild(candidateWorldToProjection, w2v, v2p))
          return r;
        r.ok = true;
        r.worldToView = w2v;
        r.viewToProjection = v2p;
        r.error = matrixL1Error(v2p * w2v, candidateWorldToProjection);
        return r;
      };

      const CandidateResult raw = evalCandidate(worldToProjection);
      const CandidateResult transposed = evalCandidate(transpose(worldToProjection));

      const CandidateResult* best = nullptr;
      if (raw.ok && transposed.ok) {
        best = (raw.error <= transposed.error) ? &raw : &transposed;
      } else if (raw.ok) {
        best = &raw;
      } else if (transposed.ok) {
        best = &transposed;
      } else {
        return false;
      }

      outWorldToView = best->worldToView;
      outViewToProjection = best->viewToProjection;
      if (outUsedTranspose != nullptr)
        *outUsedTranspose = (best == &transposed);
      if (outReconstructionError != nullptr)
        *outReconstructionError = best->error;
      return true;
    }
  }

  D3D9Rtx::Ue3VertexFactoryType D3D9Rtx::classifyUe3VertexFactory(const D3D9VertexElements& elements) {
    if (elements.empty()) {
      return Ue3VertexFactoryType::Unknown;
    }

    const VDeclSignature sig = buildVDeclSignature(elements);

    if (!sig.hasPosition) {
      return Ue3VertexFactoryType::Unknown;
    }

    // terrain = POSITION(UBYTE4) + BLENDWEIGHT(FLOAT1) + TANGENT(SHORT2)
    // packed UBYTE4 position is unique to terrain
    if (sig.positionType == D3DDECLTYPE_UBYTE4 &&
        sig.hasBlendWeight && sig.blendWeightType == D3DDECLTYPE_FLOAT1 &&
        sig.hasTangent && sig.tangentType == D3DDECLTYPE_SHORT2) {
      if (sig.hasNormal) {
        return Ue3VertexFactoryType::TerrainMorph;
      }
      return Ue3VertexFactoryType::Terrain;
    }

    // particle = POSITION(FLOAT3) + NORMAL(FLOAT3) + TANGENT(FLOAT3) + TEXCOORD0(FLOAT2) + BLENDWEIGHT(FLOAT1) + TEXCOORD1(FLOAT4)
    // NORMAL is FLOAT3 (not UBYTE4), TANGENT is FLOAT3, plus BLENDWEIGHT(FLOAT1)
    if ((sig.positionType == D3DDECLTYPE_FLOAT3 || sig.positionType == D3DDECLTYPE_FLOAT4) &&
        !sig.hasNormal &&
        sig.hasTangent && sig.tangentType == D3DDECLTYPE_FLOAT4 &&
        sig.hasBlendWeight && sig.blendWeightType == D3DDECLTYPE_FLOAT1 &&
        sig.texcoordCount >= 4 &&
        !sig.hasBlendIndices) {
      return Ue3VertexFactoryType::LensFlare;
    }

    if ((sig.positionType == D3DDECLTYPE_FLOAT3 || sig.positionType == D3DDECLTYPE_FLOAT4) &&
        sig.hasNormal && sig.normalType == D3DDECLTYPE_FLOAT4 &&
        sig.hasTangent && sig.tangentType == D3DDECLTYPE_FLOAT3 &&
        sig.hasBlendWeight && sig.blendWeightType == D3DDECLTYPE_FLOAT1 &&
        sig.texcoordCount >= 2 &&
        !sig.hasBlendIndices) {
      return Ue3VertexFactoryType::ParticleBeamTrail;
    }

    if (sig.positionType == D3DDECLTYPE_FLOAT3 &&
        sig.hasNormal && sig.normalType == D3DDECLTYPE_FLOAT3 &&
        sig.hasTangent && sig.tangentType == D3DDECLTYPE_FLOAT3 &&
        sig.hasBlendWeight && sig.blendWeightType == D3DDECLTYPE_FLOAT1 &&
        !sig.hasBlendIndices) {
      return Ue3VertexFactoryType::Particle;
    }

    // SpeedTree variants carry an explicit binormal and wind info through BLENDINDICES,
    // but they do not have skinning blend weights.
    if (sig.positionType == D3DDECLTYPE_FLOAT3 &&
        sig.hasBinormal &&
        sig.hasBlendIndices &&
        !sig.hasBlendWeight &&
        sig.hasTangent &&
        sig.hasNormal) {
      return Ue3VertexFactoryType::SpeedTree;
    }

    // position only (depth prepass) = only POSITION, no tangent/normal/texcoord/color
    if (sig.positionType == D3DDECLTYPE_FLOAT3 &&
        !sig.hasTangent && !sig.hasNormal && sig.texcoordCount == 0 &&
        !sig.hasColor0 && !sig.hasColor1 &&
        !sig.hasBlendWeight && !sig.hasBlendIndices) {
      return Ue3VertexFactoryType::PositionOnly;
    }

    // GPUSkin variants - must have BLENDINDICES + BLENDWEIGHT (bone data)
    // normal/tangent are UBYTE4 (PackedNormal)
    if (sig.positionType == D3DDECLTYPE_FLOAT3 &&
        sig.hasBlendIndices &&
        sig.hasBlendWeight &&
        sig.hasTangent && sig.tangentType == D3DDECLTYPE_UBYTE4 &&
        sig.hasNormal && sig.normalType == D3DDECLTYPE_UBYTE4) {
      // morph variant adds TEXCOORD6(FLOAT3) for delta position + TEXCOORD7(UBYTE4) for delta normal
      if (sig.hasTexcoord6 && sig.texcoord6Type == D3DDECLTYPE_FLOAT3 &&
          sig.hasTexcoord7) {
        return Ue3VertexFactoryType::GPUSkinMorph;
      }
      return Ue3VertexFactoryType::GPUSkin;
    }

    // Instanced mesh particles share FoliageVertexFactory.usf and the instancing axes in
    // TEXCOORD1..4, but their declaration has no NORMAL: FParticleInstancedMeshVertexFactory
    // walks {VEU_Tangent, VEU_Binormal, VEU_Normal} while only filling components 0 and 1, so
    // the mesh's TangentZ arrives under the BINORMAL semantic. (The stock game's shader still
    // declares it as NORMAL and therefore never reads it - so UE3 renders these unlit.)
    if (sig.positionType == D3DDECLTYPE_FLOAT3 &&
        sig.hasTangent && sig.tangentType == D3DDECLTYPE_UBYTE4 &&
        sig.hasBinormal && sig.binormalType == D3DDECLTYPE_UBYTE4 &&
        !sig.hasNormal &&
        !sig.hasBlendIndices && !sig.hasBlendWeight &&
        sig.texcoordCount >= 5 &&
        sig.texcoordTypes[1] == D3DDECLTYPE_FLOAT3 &&
        sig.texcoordTypes[2] == D3DDECLTYPE_FLOAT3 &&
        sig.texcoordTypes[3] == D3DDECLTYPE_FLOAT3 &&
        sig.texcoordTypes[4] == D3DDECLTYPE_FLOAT3) {
      return Ue3VertexFactoryType::ParticleInstancedMesh;
    }

    // local (static mesh) = POSITION(FLOAT3) + TANGENT(UBYTE4) + NORMAL(UBYTE4) + TEXCOORDs, no BLENDINDICES
    // foliage extends the local layout with instancing axes in TEXCOORD1..4.
    if (sig.positionType == D3DDECLTYPE_FLOAT3 &&
        sig.hasTangent &&
        sig.hasNormal &&
        !sig.hasBlendIndices &&
        sig.texcoordCount >= 5 &&
        sig.texcoordTypes[1] == D3DDECLTYPE_FLOAT3 &&
        sig.texcoordTypes[2] == D3DDECLTYPE_FLOAT3 &&
        sig.texcoordTypes[3] == D3DDECLTYPE_FLOAT3 &&
        sig.texcoordTypes[4] == D3DDECLTYPE_FLOAT3) {
      return Ue3VertexFactoryType::Foliage;
    }

    if (sig.positionType == D3DDECLTYPE_FLOAT3 &&
        sig.hasTangent && sig.tangentType == D3DDECLTYPE_UBYTE4 &&
        sig.hasNormal && sig.normalType == D3DDECLTYPE_UBYTE4 &&
        !sig.hasBlendIndices) {
      return Ue3VertexFactoryType::Local;
    }

    return Ue3VertexFactoryType::Unknown;
  }

  D3D9Rtx::Ue3ShaderFeatureInfo D3D9Rtx::getUe3ShaderFeatureInfo(const D3D9CommonShader* shader) {
    Ue3ShaderFeatureInfo empty;
    empty.initialized = true;

    if (shader == nullptr)
      return empty;

    const auto& bytecode = shader->GetBytecode();
    const XXH64_hash_t shaderHash = shader->GetBytecodeHash();
    if (shaderHash == 0)
      return empty;

    auto it = m_ue3ShaderFeatureCache.find(shaderHash);
    if (it != m_ue3ShaderFeatureCache.end())
      return it->second;

    Ue3ShaderFeatureInfo info;
    info.initialized = true;

    try {
      if (bytecode.size() < sizeof(uint32_t) || (bytecode.size() % sizeof(uint32_t)) != 0) {
        m_ue3ShaderFeatureCache.emplace(shaderHash, info);
        return info;
      }

      const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
      DxsoDecodeContext decoder(shader->GetInfo());
      DxsoCodeIter iter(tokens + 1);
      while (decoder.decodeInstruction(iter)) {
        if (decoder.getCtabInfo().m_size != 0)
          break;
      }

      const DxsoCtab& ctab = decoder.getCtabInfo();
      if (ctab.m_size == 0 || ctab.m_constantData.empty()) {
        m_ue3ShaderFeatureCache.emplace(shaderHash, info);
        return info;
      }

      auto markName = [&](const std::string& lowerName, const bool isSampler, const bool isFloat4, const uint32_t registerIndex) {
        if (isSampler) {
          const uint8_t semanticFlags = classifyPixelSamplerSemanticFlags(lowerName);
          info.hasMaterialSampler |= (semanticFlags & kPsSamplerSemanticMaterialTexture) != 0;
          info.hasEngineAuxSampler |= (semanticFlags & kPsSamplerSemanticEngineAuxiliary) != 0;
          info.hasVideoSampler |= (semanticFlags & kPsSamplerSemanticVideo) != 0;

          // TdToneMapping capture: record the exact sampler indices of the
          // baked colour curve LUT textures.
          if (registerIndex < 0xFF) {
            if (lowerName == "colorcurvesktexture") {
              info.colorCurvesKSamplerIndex = uint8_t(registerIndex);
            } else if (lowerName == "colorcurvesmtexture") {
              info.colorCurvesMSamplerIndex = uint8_t(registerIndex);
            }
          }

          info.hasSceneColorSampler |= containsToken(lowerName, "scenecolor");
          info.hasSceneDepthSampler |= containsToken(lowerName, "scenedepth") ||
                                       containsToken(lowerName, "destdepth") ||
                                       containsToken(lowerName, "pixeldepth");
          info.hasLightAttenuationSampler |= containsToken(lowerName, "lightattenuation");
          info.hasShadowSampler |= containsToken(lowerName, "shadowdepth") ||
                                   containsToken(lowerName, "shadowtexture") ||
                                   containsToken(lowerName, "shadowvariance");
          info.hasVelocitySampler |= containsToken(lowerName, "velocitybuffer") ||
                                     containsToken(lowerName, "velocitytexture");
          info.hasBlurredImageSampler |= containsToken(lowerName, "blurredimage");
          info.hasFilterTextureSampler |= containsToken(lowerName, "filtertexture");
          info.hasExposureOrToneSampler |= containsToken(lowerName, "exposure") ||
                                           containsToken(lowerName, "colorcurves") ||
                                           containsToken(lowerName, "saturationmask") ||
                                           info.hasBlurredImageSampler ||
                                           info.hasFilterTextureSampler;
          info.hasUiSampler |= containsToken(lowerName, "scenecoloruitexture") ||
                               containsToken(lowerName, "blurredui") ||
                               containsToken(lowerName, "uitexture") ||
                               containsToken(lowerName, "uibuffer");
          info.hasDistortionSampler |= containsToken(lowerName, "distortion") ||
                                       containsToken(lowerName, "lineintegral");
        }

        info.hasBinkConstants |= containsToken(lowerName, "ycrcb") ||
                                 containsToken(lowerName, "yuv") ||
                                 containsToken(lowerName, "bink") ||
                                 lowerName == "tor" ||
                                 lowerName == "tog" ||
                                 lowerName == "tob";
        info.hasPrevViewProjection |= containsToken(lowerName, "prevviewprojectionmatrix") ||
                                      containsToken(lowerName, "previousviewprojectionmatrix") ||
                                      containsToken(lowerName, "prev_view_projection_matrix");
        info.hasVelocityConstants |= containsToken(lowerName, "velocityscaleoffset") ||
                                     containsToken(lowerName, "individualvelocityscale") ||
                                     containsToken(lowerName, "stretchtimescale");
        info.hasMotionBlurConstants |= containsToken(lowerName, "motionpacked") ||
                                       containsToken(lowerName, "staticvelocityparameters") ||
                                       containsToken(lowerName, "motionblur");
        info.hasDynamicLightingConstants |= lowerName == "lightcolor" ||
                                            containsToken(lowerName, "lightcolorandfalloffexponent") ||
                                            containsToken(lowerName, "lightpositionandinvradius") ||
                                            containsToken(lowerName, "lightdirection") ||
                                            containsToken(lowerName, "spotdirection") ||
                                            containsToken(lowerName, "tangentlightvector") ||
                                            containsToken(lowerName, "worldlightvector");
        info.hasLightFunctionConstants |= containsToken(lowerName, "screentolight") ||
                                          containsToken(lowerName, "screen_to_light");
        info.hasSphericalHarmonicLightingConstants |= containsToken(lowerName, "worldincidentlighting") ||
                                                     containsToken(lowerName, "shbasiscubetextures") ||
                                                     containsToken(lowerName, "shbasis");
        info.hasScreenToShadowMatrix |= containsToken(lowerName, "screentoshadowmatrix") ||
                                        containsToken(lowerName, "screen_to_shadow");
        info.hasShadowModulateConstants |= containsToken(lowerName, "shadowmodulatecolor") ||
                                           containsToken(lowerName, "shadowattenuation") ||
                                           containsToken(lowerName, "invmaxsubjectdepth");
        info.hasToneMapConstants |= containsToken(lowerName, "sceneshadowsanddesaturation") ||
                                    containsToken(lowerName, "sceneinversehighlights") ||
                                    containsToken(lowerName, "scenemidtones") ||
                                    containsToken(lowerName, "scenescaledluminanceweights") ||
                                    containsToken(lowerName, "exposuresettings");
        info.hasGammaConstants |= containsToken(lowerName, "gammacolorscaleandinverse") ||
                                  containsToken(lowerName, "gammaoverlaycolor") ||
                                  containsToken(lowerName, "inversegamma") ||
                                  containsToken(lowerName, "colorscale");
        info.hasFogConstants |= containsToken(lowerName, "fog") ||
                                containsToken(lowerName, "heightfog") ||
                                containsToken(lowerName, "lineintegral");
        info.hasHazeConstants |= containsToken(lowerName, "haze") ||
                                 containsToken(lowerName, "sunvector");
        info.hasUiCompositeConstants |= lowerName == "fade" ||
                                        containsToken(lowerName, "scenecolorui") ||
                                        containsToken(lowerName, "bluramount");
        info.hasDofPackedParameters |= containsToken(lowerName, "packedparameters");
        info.hasDofMinMaxBlurClamp |= containsToken(lowerName, "minmaxblurclamp");
        info.hasFilterSampleWeights |= containsToken(lowerName, "sampleweights");

        // TdToneMapping capture: record the exact float register indices of
        // the grade constants so the (skipped) tonemap pass's pixel shader
        // constants can be read at draw time.
        if (isFloat4 && registerIndex <= 0x7FFF) {
          if (lowerName == "sceneshadowsanddesaturation") {
            info.toneMapSceneShadowsReg = int16_t(registerIndex);
          } else if (lowerName == "sceneinversehighlights") {
            info.toneMapInverseHighLightsReg = int16_t(registerIndex);
          } else if (lowerName == "scenemidtones") {
            info.toneMapMidTonesReg = int16_t(registerIndex);
          } else if (lowerName == "scenescaledluminanceweights") {
            info.toneMapScaledLumaWeightsReg = int16_t(registerIndex);
          } else if (lowerName == "gammacolorscaleandinverse") {
            info.toneMapGammaColorScaleReg = int16_t(registerIndex);
          } else if (lowerName == "gammaoverlaycolor") {
            info.toneMapGammaOverlayReg = int16_t(registerIndex);
          }
        }
      };

      for (const DxsoCtab::Constant& c : ctab.m_constantData) {
        const bool isSampler = c.registerSet == kD3dxRegisterSetSampler;
        const bool isFloat4 = c.registerSet == kD3dxRegisterSetFloat4;
        const std::string lowerName = toLowerAscii(c.name);
        markName(lowerName, isSampler, isFloat4, c.registerIndex);

        // Exact names, so UberPostProcessBlend's GammaColorScaleAndInverse/GammaOverlayColor
        // (a different shader, whose output the composite still has to gamma correct) cannot
        // be mistaken for the final composite's own constants.
        if (!isSampler) {
          if (lowerName == "inversegamma")
            info.gammaInverseReg = c.registerIndex;
          else if (lowerName == "colorscale")
            info.gammaColorScaleReg = c.registerIndex;
          else if (lowerName == "overlaycolor")
            info.gammaOverlayColorReg = c.registerIndex;
        }
      }
    } catch (...) {
    }

    m_ue3ShaderFeatureCache.emplace(shaderHash, info);
    return info;
  }

  bool D3D9Rtx::isUe3WorldGeometryVertexFactory(const Ue3VertexFactoryType type) {
    switch (type) {
    case Ue3VertexFactoryType::Local:
    case Ue3VertexFactoryType::LocalDecal:
    case Ue3VertexFactoryType::GPUSkin:
    case Ue3VertexFactoryType::GPUSkinMorph:
    case Ue3VertexFactoryType::Terrain:
    case Ue3VertexFactoryType::TerrainMorph:
    case Ue3VertexFactoryType::SpeedTree:
    case Ue3VertexFactoryType::Foliage:
    case Ue3VertexFactoryType::ParticleInstancedMesh:
    case Ue3VertexFactoryType::Particle:
    case Ue3VertexFactoryType::ParticleBeamTrail:
    case Ue3VertexFactoryType::LensFlare:
      return true;
    default:
      return false;
    }
  }

  // UE3 instances a mesh by leaving its placement out of the shader constants entirely: the mesh
  // streams carry D3DSTREAMSOURCE_INDEXEDDATA | instanceCount, one further stream is tagged
  // D3DSTREAMSOURCE_INSTANCEDATA, and the vertex factory reads InstanceOffset (TEXCOORD1) plus the
  // three basis axes (TEXCOORD2..4) out of it. See FParticleInstancedMeshVertexFactory::InitRHI and
  // FFoliageVertexFactory::InitRHI in the UE3 engine, and GetInstanceToWorld in
  // FoliageVertexFactory.usf which both compile down to that layout.


  // Whatever these bounds drop, the kept set has to be the same set next frame: an instance that
  // comes and goes as the camera moves flickers, and a selection that reorders the survivors also
  // renames them (see Ue3DecomposedInstance::sourceIndex). Hence the count clamp keeps the lowest
  // source indices rather than the nearest to the camera. Distance culling is view-dependent by
  // definition and can pop at its boundary, which is why it is opt-in.

  // Names the batch an instanced draw belongs to, without involving any transform.
  //
  // The mesh streams say which mesh is being instanced but not which component is instancing it, and
  // two piles of the same debris share them. The instance buffer would distinguish those but cannot
  // serve as an identity - RenderNxFluidInstanced creates a fresh one per frame unless its pool hands
  // one back - so batches are matched to the previous frame's by continuity of their own centroid,
  // which holds because a batch as a whole barely moves even while its instances do.



  bool D3D9Rtx::ue3ViewportAspectMatchesBackbuffer(
      const uint32_t vpW, const uint32_t vpH, const uint32_t bbW, const uint32_t bbH) {
    if (vpW == 0 || vpH == 0 || bbW == 0 || bbH == 0) {
      return false;
    }
    const double a = double(vpW) * double(bbH);
    const double b = double(vpH) * double(bbW);
    const double denom = std::max(a, b);
    return denom > 0.0 && (std::abs(a - b) / denom) < 0.05;
  }

  bool D3D9Rtx::ue3ViewportIsMainViewSized(
      const uint32_t vpW, const uint32_t vpH, const uint32_t bbW, const uint32_t bbH) {
    return bbW != 0 && bbH != 0 &&
           vpW * 2 >= bbW && vpH * 2 >= bbH &&
           ue3ViewportAspectMatchesBackbuffer(vpW, vpH, bbW, bbH);
  }

  D3D9Rtx::Ue3PassType D3D9Rtx::classifyUe3Pass(const DrawContext& drawContext) {
    if (!m_frameOptions.ue3EngineMode)
      return Ue3PassType::Unknown;

    const bool depthEnabled = d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_TRUE;
    const bool zWriteEnabled = d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE;
    const bool samplesRenderTarget = m_parent->GetActiveRTTextures() != 0;
    // UE3 binds textures lazily, leaving render targets in slots the current shader never
    // reads; classification keyed on raw bound-RT state varies with draw order (which UE3
    // re-sorts per camera for translucency) and flickers. Only count render targets bound
    // to samplers the active pixel shader actually uses.
    const bool psSamplesRenderTarget =
      (m_parent->GetActiveRTTextures() & m_parent->m_psShaderMasks.samplerMask) != 0;
    const bool likelyFullscreen = !depthEnabled && !zWriteEnabled && drawContext.PrimitiveCount <= 4;

    const bool isWorldGeometry = isUe3WorldGeometryVertexFactory(m_currentUe3VertexFactory);

    // Cheap early-outs first: depth prepass and shadow depth draws are the most common
    // skipped passes and need no shader feature information.
    if (m_currentUe3VertexFactory == Ue3VertexFactoryType::PositionOnly)
      return Ue3PassType::DepthPrepass;

    if (m_activePresentParams.has_value() &&
        d3d9State().renderTargets[kRenderTargetIndex] != nullptr) {
      const auto& rtExt = d3d9State().renderTargets[kRenderTargetIndex]->GetSurfaceExtent();
      const uint32_t bbW = m_activePresentParams->BackBufferWidth;
      if (rtExt.width == rtExt.height &&
          rtExt.width <= 2048 &&
          rtExt.width < bbW / 2 &&
          zWriteEnabled) {
        return Ue3PassType::ShadowDepth;
      }
    }

    const D3D9CommonShader* vertexShaderCommon =
      m_parent->UseProgrammableVS() && d3d9State().vertexShader.ptr() != nullptr
        ? d3d9State().vertexShader->GetCommonShader()
        : nullptr;
    const D3D9CommonShader* pixelShaderCommon =
      m_parent->UseProgrammablePS() && d3d9State().pixelShader.ptr() != nullptr
        ? d3d9State().pixelShader->GetCommonShader()
        : nullptr;

    const Ue3ShaderFeatureInfo vsInfo = getUe3ShaderFeatureInfo(vertexShaderCommon);
    const Ue3ShaderFeatureInfo psInfo = getUe3ShaderFeatureInfo(pixelShaderCommon);

    if (psInfo.hasBinkConstants) {
      if (likelyFullscreen && !isWorldGeometry) {
        return Ue3PassType::VideoCinematic;
      }
      return Ue3PassType::VideoSurface;
    }

    if (psInfo.hasUiSampler || psInfo.hasUiCompositeConstants)
      return Ue3PassType::UiComposite;

    // UE3 SceneCapture probes re-render the world before the main view; viewport size/aspect
    // (and later mirrored/undecomposable camera checks) keep them from stealing Main.
    if ((m_frameOptions.ue3SkipSceneCapturePasses || m_frameOptions.ue3EngineMode) &&
        isWorldGeometry &&
        m_activePresentParams.has_value()) {
      const D3DVIEWPORT9& vp = d3d9State().viewport;
      const uint32_t bbW = m_activePresentParams->BackBufferWidth;
      const uint32_t bbH = m_activePresentParams->BackBufferHeight;
      if (bbW != 0 && bbH != 0 && vp.Width != 0 && vp.Height != 0) {
        if (vp.Width * 2 < bbW && vp.Height * 2 < bbH) {
          return Ue3PassType::SceneCapture;
        }
        // Exact-half splitscreen must not be treated as a mismatched-aspect probe.
        const bool exactHalfSplit =
          (vp.Width * 2 == bbW && vp.Height == bbH) ||
          (vp.Height * 2 == bbH && vp.Width == bbW);
        if (!exactHalfSplit &&
            (vp.Width < bbW || vp.Height < bbH) &&
            !ue3ViewportAspectMatchesBackbuffer(vp.Width, vp.Height, bbW, bbH)) {
          return Ue3PassType::SceneCapture;
        }
      }
    }

    if (psInfo.hasScreenToShadowMatrix ||
        psInfo.hasShadowModulateConstants ||
        (psInfo.hasShadowSampler && psInfo.hasSceneDepthSampler)) {
      return Ue3PassType::ModulatedShadowProjection;
    }

    if ((vsInfo.hasPrevViewProjection || psInfo.hasVelocityConstants) &&
        !samplesRenderTarget) {
      return Ue3PassType::Velocity;
    }

    const bool hasDynamicLightPassSignals =
      (psInfo.hasDynamicLightingConstants || vsInfo.hasDynamicLightingConstants) &&
      (psInfo.hasLightAttenuationSampler ||
       psInfo.hasShadowSampler ||
       vsInfo.hasDynamicLightingConstants);
    // UE3 folds light-environment lighting into the BASE pass (SH WorldIncidentLighting /
    // TangentLightVector constants), so a draw with material samplers is the primary
    // textured draw, not a per-light pass - unless it uses additive light-pass blending
    const bool isAdditiveLightBlend =
      d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] != FALSE &&
      d3d9State().renderStates[D3DRS_SRCBLEND] == D3DBLEND_ONE &&
      d3d9State().renderStates[D3DRS_DESTBLEND] == D3DBLEND_ONE;
    const bool looksLikeLitBasePassMaterial = psInfo.hasMaterialSampler && !isAdditiveLightBlend;
    if (!looksLikeLitBasePassMaterial &&
        (hasDynamicLightPassSignals ||
         psInfo.hasLightFunctionConstants ||
         psInfo.hasSphericalHarmonicLightingConstants ||
         (psInfo.hasLightAttenuationSampler && psInfo.hasShadowSampler))) {
      return Ue3PassType::Lighting;
    }

    if ((psInfo.hasVelocitySampler || psInfo.hasMotionBlurConstants) &&
        (samplesRenderTarget || likelyFullscreen)) {
      return Ue3PassType::FullscreenPostProcess;
    }

    // Explicit DOFAndBloom/Uber signals (also covered by the catch-all below for typical
    // ME shaders); kept so FilterColor blur is classified even if z-write state is atypical.
    if (!isWorldGeometry &&
        (likelyFullscreen || psSamplesRenderTarget) &&
        psInfo.looksLikeDofAndBloomPostProcess()) {
      return Ue3PassType::FullscreenPostProcess;
    }

    // The two screen-space catch-alls below must never swallow world geometry:
    // translucent world meshes render with depth writes off and legitimately declare
    // scene color/depth samplers (DepthBiasedAlpha, DestColor/refraction) or fog inputs
    if (!isWorldGeometry &&
        (likelyFullscreen || psSamplesRenderTarget || !zWriteEnabled) &&
        (psInfo.hasFogConstants || psInfo.hasHazeConstants || psInfo.hasDistortionSampler)) {
      return Ue3PassType::FogOrDistortion;
    }

    if (!isWorldGeometry &&
        !zWriteEnabled &&
        (likelyFullscreen || psSamplesRenderTarget) &&
        (psInfo.hasSceneColorSampler ||
         psInfo.hasSceneDepthSampler ||
         psInfo.hasLightAttenuationSampler ||
         psInfo.hasExposureOrToneSampler ||
         psInfo.hasToneMapConstants ||
         psInfo.hasGammaConstants ||
         psInfo.hasEngineAuxSampler)) {
      return Ue3PassType::FullscreenPostProcess;
    }

    if (psInfo.hasMaterialSampler ||
        m_currentUe3VertexFactory == Ue3VertexFactoryType::Local ||
        m_currentUe3VertexFactory == Ue3VertexFactoryType::LocalDecal ||
        m_currentUe3VertexFactory == Ue3VertexFactoryType::GPUSkin ||
        m_currentUe3VertexFactory == Ue3VertexFactoryType::GPUSkinMorph ||
        m_currentUe3VertexFactory == Ue3VertexFactoryType::Terrain ||
        m_currentUe3VertexFactory == Ue3VertexFactoryType::TerrainMorph ||
        m_currentUe3VertexFactory == Ue3VertexFactoryType::SpeedTree ||
        m_currentUe3VertexFactory == Ue3VertexFactoryType::Foliage ||
        m_currentUe3VertexFactory == Ue3VertexFactoryType::ParticleInstancedMesh) {
      return Ue3PassType::Material;
    }

    return Ue3PassType::Unknown;
  }


  D3D9Rtx::D3D9Rtx(D3D9DeviceEx* d3d9Device, bool enableDrawCallConversion)
    : m_parent(d3d9Device)
    , m_enableDrawCallConversion(enableDrawCallConversion) {
  }

  D3D9Rtx::~D3D9Rtx() {
    restoreNgxGameSettingsRedirects();

    // Owned bridge parent handle only; GetCurrentProcess() is a pseudo-handle.
    if (m_ngxGameProcessOwned && m_ngxGameProcess != nullptr)
      ::CloseHandle(m_ngxGameProcess);
  }


  void D3D9Rtx::Initialize() {
    m_vsVertexCaptureData = m_parent->CreateConstantBuffer(false,
                                        sizeof(D3D9RtxVertexCaptureData),
                                        DxsoProgramType::VertexShader,
                                        DxsoConstantBuffers::VSVertexCaptureData);

    // Get constant buffer bindings from D3D9
    m_parent->EmitCs([vertexCaptureCB = m_vsVertexCaptureData](DxvkContext* ctx) {
      const uint32_t vsFixedFunctionConstants = computeResourceSlotId(DxsoProgramType::VertexShader, DxsoBindingType::ConstantBuffer, DxsoConstantBuffers::VSFixedFunction);
      const uint32_t psSharedStateConstants = computeResourceSlotId(DxsoProgramType::PixelShader, DxsoBindingType::ConstantBuffer, DxsoConstantBuffers::PSShared);
      static_cast<RtxContext*>(ctx)->setConstantBuffers(vsFixedFunctionConstants, psSharedStateConstants, vertexCaptureCB);
    });
  }

  const Direct3DState9& D3D9Rtx::d3d9State() const {
    return *m_parent->GetRawState();
  }

  void D3D9Rtx::refreshFrameOptionCache() {
    FrameOptionCache& o = m_frameOptions;

    // Reads use the xObject().get() accessor form (the same locked read as x()) so
    // the plain x() spelling never appears in this function: any plain option
    // accessor found in per-draw code is then, by construction, an un-snapshotted
    // per-draw locked read that should be moved into this cache.
    o.orthographicIsUI = orthographicIsUIObject().get();
    o.preTransformedVerticesIsUI = preTransformedVerticesIsUIObject().get();
    o.ue3EngineMode = ue3EngineModeObject().get();
    o.ue3SkipSceneCapturePasses = ue3SkipSceneCapturePassesObject().get();
    o.conservativeOcclusionQueries = conservativeOcclusionQueriesObject().get();
    o.eventQueryCsCompletion = eventQueryCsCompletionObject().get();
    o.sequenceTrackedLockWaits = sequenceTrackedLockWaitsObject().get();
    o.skipRenderTargetCopies = skipRenderTargetCopiesObject().get();
    o.ue3LogOcclusionQueries = ue3LogOcclusionQueriesObject().get();
    o.ngxPassthroughMode = RtxNgxPassthrough::ngxPassthroughMode();
    o.ngxPassthroughJitter = RtxNgxPassthrough::enableJitter();
    o.ngxPrePostProcess = RtxNgxPassthrough::prePostProcess();
    o.ngxDlfgHudless = RtxNgxPassthrough::dlfgHudlessInput() && DxvkDLFG::enable() &&
                       m_parent->GetDXVKDevice()->getCommon()->metaNGXContext().supportsDLFG();
    o.ngxObjectVelocities = RtxNgxPassthrough::objectVelocities();
    o.ngxDebugVisualization = RtxNgxPassthrough::debugVisualization();

    // On-demand post-chain dump (one-shot: consumed here, option resets itself)
    if (o.ngxPassthroughMode && RtxNgxPassthrough::dumpPostChainFrames() > 0 && m_ngxPostChainDumpFramesLeft == 0) {
      m_ngxPostChainDumpFramesLeft = uint32_t(RtxNgxPassthrough::dumpPostChainFrames());
      RtxNgxPassthrough::dumpPostChainFramesObject().setDeferred(0);
      Logger::info(str::format("[RTX NGX Passthrough][dump] On-demand dump armed for ", m_ngxPostChainDumpFramesLeft, " frames."));
    }

    if (o.ngxPassthroughMode && RtxNgxPassthrough::dumpVelocityCaptureFrames() > 0 &&
        m_ngxVelocityDumpFramesLeft == 0) {
      m_ngxVelocityDumpFramesLeft = uint32_t(RtxNgxPassthrough::dumpVelocityCaptureFrames());
      RtxNgxPassthrough::dumpVelocityCaptureFramesObject().setDeferred(0);
      Logger::info(str::format("[RTX NGX Passthrough][velocity dump] Armed for ",
                               m_ngxVelocityDumpFramesLeft, " frames."));
    }
    o.enableRaytracing = RtxOptions::enableRaytracingObject().get();
    o.logReplacementResolution = RtxOptions::logReplacementResolutionObject().get();
    o.uiTextures = &RtxOptions::uiTexturesObject().get();

    o.valid = true;
  }







  bool D3D9Rtx::tryGetUe3CameraFromConstantsCached(uint32_t viewProjReg,
                                                   uint32_t viewOriginReg,
                                                   Matrix4& outWorldToView,
                                                   Matrix4& outViewToProjection,
                                                   bool& outUsedTranspose,
                                                   float& outReconstructionError,
                                                   XXH64_hash_t* outConstantsHash) {
    if (viewProjReg + 3 >= caps::MaxFloatConstantsSoftware || viewOriginReg >= caps::MaxFloatConstantsSoftware)
      return false;

    // cache by raw constant values to avoid repeated heavy extraction work per draw call
    struct Ue3CameraConstsKey {
      uint32_t viewProjReg;
      uint32_t viewOriginReg;
      Vector4 regs[5];
    };

    Ue3CameraConstsKey key {};
    key.viewProjReg = viewProjReg;
    key.viewOriginReg = viewOriginReg;
    key.regs[0] = d3d9State().vsConsts.fConsts[viewProjReg + 0];
    key.regs[1] = d3d9State().vsConsts.fConsts[viewProjReg + 1];
    key.regs[2] = d3d9State().vsConsts.fConsts[viewProjReg + 2];
    key.regs[3] = d3d9State().vsConsts.fConsts[viewProjReg + 3];
    key.regs[4] = d3d9State().vsConsts.fConsts[viewOriginReg];

    const XXH64_hash_t constantsHash = XXH3_64bits(&key, sizeof(key));

    if (outConstantsHash != nullptr) {
      *outConstantsHash = constantsHash;
    }

    for (const Ue3CameraConstantsCache& slot : m_ue3CameraConstantsCache) {
      if (slot.valid && slot.hash == constantsHash) {
        if (slot.extractionFailed) {
          return false;
        }
        outWorldToView = slot.worldToView;
        outViewToProjection = slot.viewToProjection;
        outUsedTranspose = slot.usedTranspose;
        outReconstructionError = slot.reconstructionError;
        return true;
      }
    }

    Matrix4 ue3WorldToView;
    Matrix4 ue3ViewToProjection;
    bool usedTranspose = false;
    float reconstructionError = 0.0f;
    const bool extracted = tryExtractUe3WorldToViewAndProjectionFromShaderConstants(
        d3d9State().vsConsts, viewProjReg, viewOriginReg, ue3WorldToView, ue3ViewToProjection, &usedTranspose, &reconstructionError);

    Ue3CameraConstantsCache& slot = m_ue3CameraConstantsCache[m_ue3CameraConstantsCacheNextSlot];
    m_ue3CameraConstantsCacheNextSlot = (m_ue3CameraConstantsCacheNextSlot + 1u) % kUe3CameraConstantsCacheSlots;
    slot.hash = constantsHash;
    slot.valid = true;
    slot.extractionFailed = !extracted;
    slot.usedTranspose = usedTranspose;
    slot.worldToView = ue3WorldToView;
    slot.viewToProjection = ue3ViewToProjection;
    slot.reconstructionError = reconstructionError;

    if (!extracted) {
      return false;
    }

    outWorldToView = ue3WorldToView;
    outViewToProjection = ue3ViewToProjection;
    outUsedTranspose = usedTranspose;
    outReconstructionError = reconstructionError;
    return true;
  }

  Matrix4 D3D9Rtx::extractUe3ObjectToWorld(uint32_t reg, bool hasWorldToLocal, uint32_t w2lReg, bool cameraUsedTranspose) {
    // Memo lookup: every input to the transpose/affinity/inverse disambiguation below
    // (register contents and the camera transpose convention tiebreaker) is folded into
    // the key, so a hit returns exactly what the computation would produce. Static
    // placements re-upload identical matrices every frame, making this a per-draw
    // matrix-inverse saving.
    XXH64_hash_t o2wKeyHash = XXH3_64bits(&d3d9State().vsConsts.fConsts[reg], 4 * sizeof(Vector4));
    if (hasWorldToLocal) {
      o2wKeyHash = XXH3_64bits_withSeed(&d3d9State().vsConsts.fConsts[w2lReg], 3 * sizeof(Vector4), o2wKeyHash);
    }
    const uint32_t o2wKeyFlags = (hasWorldToLocal ? 1u : 0u) | (cameraUsedTranspose ? 2u : 0u);
    o2wKeyHash = XXH3_64bits_withSeed(&o2wKeyFlags, sizeof(o2wKeyFlags), o2wKeyHash);

    const auto o2wIt = m_ue3ObjectToWorldCache.find(o2wKeyHash);
    if (o2wIt != m_ue3ObjectToWorldCache.end()) {
      return o2wIt->second;
    }

    const Matrix4 localToWorldRaw = [&] {
      Matrix4 m;
      m[0] = d3d9State().vsConsts.fConsts[reg + 0];
      m[1] = d3d9State().vsConsts.fConsts[reg + 1];
      m[2] = d3d9State().vsConsts.fConsts[reg + 2];
      m[3] = d3d9State().vsConsts.fConsts[reg + 3];
      return m;
    }();

    const Matrix4 localToWorldTransposed = transpose(localToWorldRaw);

    auto isAffineColumnVector = [](const Matrix4& m) {
      constexpr float kEps = 1e-3f;
      return std::abs(m[0].w) < kEps &&
             std::abs(m[1].w) < kEps &&
             std::abs(m[2].w) < kEps &&
             std::abs(m[3].w - 1.0f) < kEps;
    };

    const bool rawAffine = isAffineColumnVector(localToWorldRaw);
    const bool transAffine = isAffineColumnVector(localToWorldTransposed);

    // optionally use WorldToLocal (if present) to disambiguate transpose/packing
    Matrix4 worldToLocalRaw;
    Matrix4 worldToLocalTransposed;
    if (hasWorldToLocal) {
      const Vector4 c0 = d3d9State().vsConsts.fConsts[w2lReg + 0];
      const Vector4 c1 = d3d9State().vsConsts.fConsts[w2lReg + 1];
      const Vector4 c2 = d3d9State().vsConsts.fConsts[w2lReg + 2];

      worldToLocalRaw = Matrix4();
      worldToLocalRaw[0] = Vector4(c0.x, c0.y, c0.z, 0.0f);
      worldToLocalRaw[1] = Vector4(c1.x, c1.y, c1.z, 0.0f);
      worldToLocalRaw[2] = Vector4(c2.x, c2.y, c2.z, 0.0f);
      worldToLocalRaw[3] = Vector4(0.0f, 0.0f, 0.0f, 1.0f);
      worldToLocalTransposed = transpose(worldToLocalRaw);
    }

    auto l1Error3x3 = [](const Matrix4& a, const Matrix4& b) {
      float err = 0.0f;
      for (uint32_t c = 0; c < 3; c++) {
        for (uint32_t r = 0; r < 3; r++) {
          err += std::abs(a[c][r] - b[c][r]);
        }
      }
      return err;
    };

    Matrix4 localToWorld = localToWorldRaw;
    if (hasWorldToLocal && rawAffine && transAffine) {
      // both candidates look affine, so we choose the one whose inverse best matches the provided WorldToLocal basis
      const Matrix4 invRaw = inverseAffine(localToWorldRaw);
      const Matrix4 invTrans = inverseAffine(localToWorldTransposed);

      float bestErr = std::numeric_limits<float>::infinity();
      bool bestIsTransposed = false;

      const float errRaw0 = l1Error3x3(invRaw, worldToLocalRaw);
      const float errRaw1 = l1Error3x3(invRaw, worldToLocalTransposed);
      const float errTrans0 = l1Error3x3(invTrans, worldToLocalRaw);
      const float errTrans1 = l1Error3x3(invTrans, worldToLocalTransposed);

      bestErr = errRaw0;
      bestIsTransposed = false;
      if (errRaw1 < bestErr) { bestErr = errRaw1; bestIsTransposed = false; }
      if (errTrans0 < bestErr) { bestErr = errTrans0; bestIsTransposed = true; }
      if (errTrans1 < bestErr) { bestErr = errTrans1; bestIsTransposed = true; }

      constexpr float kMaxWorldToLocalMatchError = 0.25f;
      if (std::isfinite(bestErr) && bestErr <= kMaxWorldToLocalMatchError) {
        localToWorld = bestIsTransposed ? localToWorldTransposed : localToWorldRaw;
      } else {
        localToWorld = cameraUsedTranspose ? localToWorldTransposed : localToWorldRaw;
      }
    } else if (rawAffine && transAffine) {
      localToWorld = cameraUsedTranspose ? localToWorldTransposed : localToWorldRaw;
    } else if (!rawAffine && transAffine) {
      localToWorld = localToWorldTransposed;
    } else {
      localToWorld = localToWorldRaw;
    }

    if (m_ue3ObjectToWorldCache.size() >= kUe3ObjectToWorldCacheMaxEntries) {
      m_ue3ObjectToWorldCache.clear();
    }
    m_ue3ObjectToWorldCache.emplace(o2wKeyHash, localToWorld);

    return localToWorld;
  }


  const D3D9Rtx::BoundTextureSnapshot& D3D9Rtx::ensureBoundTextureSnapshot() const {
    if (m_boundTextureSnapshotValid) {
      return m_boundTextureSnapshot;
    }

    m_boundTextureSnapshot.mask = 0;

    const uint32_t boundMask = m_parent->m_activeTextures & ((1u << SamplerCount) - 1u);
    for (const uint32_t idx : bit::BitMask(boundMask)) {
      if (d3d9State().textures[idx] == nullptr) {
        continue;
      }

      D3D9CommonTexture* const texture = GetCommonTexture(d3d9State().textures[idx]);
      if (texture == nullptr) {
        continue;
      }

      BoundTextureSnapshotEntry& entry = m_boundTextureSnapshot.entries[idx];
      entry.texture = texture;
      entry.hasSampleView = texture->GetSampleView(false) != nullptr;
      DxvkImage* const image = texture->GetImage().ptr();
      entry.hasImage = image != nullptr;
      entry.imageHash = entry.hasImage ? image->getHash() : kEmptyHash;
      entry.isRenderTarget = texture->IsRenderTarget();
      entry.rtDescriptorHash = (entry.isRenderTarget && entry.hasImage) ? image->getDescriptorHash() : 0;
      entry.rtResolutionAgnosticDescriptorHash =
        (entry.isRenderTarget && entry.hasImage) ? image->getResolutionAgnosticDescriptorHash() : 0;
      // Non-RT images carry no descriptor hash on the DxvkImage; compute one only when
      // replacement diagnostics actually consume it.
      const bool wantDescriptorHashes = m_frameOptions.logReplacementResolution;
      if (entry.rtDescriptorHash != 0) {
        entry.descriptorHash = entry.rtDescriptorHash;
      } else if (wantDescriptorHashes && entry.hasImage && texture->Desc() != nullptr) {
        entry.descriptorHash = texture->Desc()->CalculateHash();
      } else {
        entry.descriptorHash = 0;
      }

      m_boundTextureSnapshot.mask |= (1u << idx);
    }

    m_boundTextureSnapshotValid = true;
    return m_boundTextureSnapshot;
  }

  bool D3D9Rtx::checkBoundTextureCategory(const fast_unordered_set& textureCategory) const {
    const uint32_t usedSamplerMask = m_parent->m_psShaderMasks.samplerMask | m_parent->m_vsShaderMasks.samplerMask;
    const BoundTextureSnapshot& boundTextures = ensureBoundTextureSnapshot();
    const uint32_t usedTextureMask = boundTextures.mask & usedSamplerMask;
    for (const uint32_t idx : bit::BitMask(usedTextureMask)) {
      const XXH64_hash_t texHash = boundTextures.entries[idx].imageHash;
      if (textureCategory.find(texHash) != textureCategory.end()) {
        return true;
      }
    }

    return false;
  }

  bool D3D9Rtx::isRenderingUI() {
    if (!m_parent->UseProgrammableVS() && m_frameOptions.orthographicIsUI) {
      // Here we assume drawcalls with an orthographic projection are UI calls (as this pattern is common, and we can't raytrace these objects).
      const bool isOrthographic = (d3d9State().transforms[GetTransformIndex(D3DTS_PROJECTION)][3][3] == 1.0f);
      const bool zWriteEnabled = d3d9State().renderStates[D3DRS_ZWRITEENABLE];
      if (isOrthographic && !zWriteEnabled) {
        return true;
      }
    }

    // Check if UI texture bound
    return checkBoundTextureCategory(*m_frameOptions.uiTextures);
  }



  void D3D9Rtx::triggerInjectRTX() {
    // Flush any pending game and RTX work
    m_parent->Flush();

    // Send command to inject RTX. Pass the backbuffer when known: the CS thread's bound render
    // target is often the scene color surface at the injection trigger, and falling back to
    // that would upscale the wrong image while the presented backbuffer stays native.
    m_parent->EmitCs([cReflexFrameId = GetReflexFrameId(), cTargetImage = m_ngxFrameBackbufferImage](DxvkContext* ctx) {
      if (cTargetImage != nullptr) {
        static_cast<RtxContext*>(ctx)->injectRTX(cReflexFrameId, cTargetImage);
      } else {
        static_cast<RtxContext*>(ctx)->injectRTX(cReflexFrameId);
      }
    });
  }

  D3D9Rtx::NgxCameraCtabRegs D3D9Rtx::scanNgxCameraCtabRegs(const std::vector<uint8_t>& bytecode) const {
    // Compact variant of the UE3 CTAB parse used by the ray traced path: only the two camera
    // symbols matter here, and only shaders declaring BOTH may steer the main camera
    // (fallback-register extraction can pick up light-space matrices from utility shaders).
    NgxCameraCtabRegs result;

    try {
      if (bytecode.size() < sizeof(uint32_t) || (bytecode.size() % sizeof(uint32_t)) != 0)
        return result;

      const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
      const uint32_t headerToken = tokens[0];
      const uint32_t headerTypeMask = headerToken & 0xffff0000u;

      DxsoProgramType programType;
      if (headerTypeMask == 0xffff0000u)
        programType = DxsoProgramTypes::PixelShader;
      else if (headerTypeMask == 0xfffe0000u)
        programType = DxsoProgramTypes::VertexShader;
      else
        return result;

      const uint32_t majorVersion = (headerToken >> 8) & 0xffu;
      const uint32_t minorVersion = headerToken & 0xffu;
      DxsoProgramInfo programInfo { programType, minorVersion, majorVersion };

      DxsoDecodeContext decoder(programInfo);
      DxsoCodeIter iter(tokens + 1);

      // Full instruction walk (the CTAB comment arrives early; the rest determines the
      // GPU skin replay parameters): a mul/mad reading the BLENDINDICES input means the
      // shader scales raw bone indices by the 3-registers-per-bone stride itself; without
      // one the vertex data is pre-scaled and feeds the address register directly. The
      // BLENDWEIGHT swizzles reveal how many influences the shader variant blends.
      uint32_t blendIndicesInputRegister = UINT32_MAX;
      uint32_t blendWeightsInputRegister = UINT32_MAX;
      bool blendIndicesScaledInShader = false;
      bool blendIndicesRead = false;
      uint32_t weightComponentsRead = 0;  // bitmask of BLENDWEIGHT components consumed

      // The decode context reuses its source array across instructions, so only the
      // operands the opcode actually consumes may be inspected
      const auto sourceOperandCount = [](DxsoOpcode op) -> uint32_t {
        switch (op) {
          case DxsoOpcode::Mov:
          case DxsoOpcode::Rcp:
          case DxsoOpcode::Rsq:
          case DxsoOpcode::Exp:
          case DxsoOpcode::Log:
          case DxsoOpcode::Frc:
            return 1;
          case DxsoOpcode::Mad:
          case DxsoOpcode::Lrp:
            return 3;
          default:
            return 2;
        }
      };

      while (decoder.decodeInstruction(iter)) {
        const DxsoInstructionContext& instructionCtx = decoder.getInstructionContext();
        const DxsoOpcode opcode = instructionCtx.instruction.opcode;

        if (opcode == DxsoOpcode::Dcl &&
            instructionCtx.dst.id.type == DxsoRegisterType::Input &&
            instructionCtx.dcl.semantic.usageIndex == 0) {
          if (instructionCtx.dcl.semantic.usage == DxsoUsage::BlendIndices) {
            blendIndicesInputRegister = instructionCtx.dst.id.num;
          } else if (instructionCtx.dcl.semantic.usage == DxsoUsage::BlendWeight) {
            blendWeightsInputRegister = instructionCtx.dst.id.num;
          }
        }

        if ((blendIndicesInputRegister != UINT32_MAX || blendWeightsInputRegister != UINT32_MAX) &&
            opcode != DxsoOpcode::Dcl && opcode != DxsoOpcode::Def && opcode != DxsoOpcode::Comment) {
          const uint32_t sourceCount = std::min(sourceOperandCount(opcode), uint32_t(instructionCtx.src.size()));
          for (uint32_t s = 0; s < sourceCount; s++) {
            const DxsoRegister& source = instructionCtx.src[s];
            if (source.id.type != DxsoRegisterType::Input) {
              continue;
            }

            if (source.id.num == blendIndicesInputRegister) {
              blendIndicesRead = true;
              if (opcode == DxsoOpcode::Mul || opcode == DxsoOpcode::Mad) {
                blendIndicesScaledInShader = true;
              }
            }

            if (source.id.num == blendWeightsInputRegister) {
              for (uint32_t component = 0; component < 4; component++) {
                weightComponentsRead |= 1u << source.swizzle[component];
              }
            }
          }
        }
      }

      result.boneIndicesPreScaled = blendIndicesInputRegister != UINT32_MAX && !blendIndicesScaledInShader;

      // Influence count: distinct weight components consumed; indices without weights =
      // the rigid-skin variant (single bone, implicit weight 1)
      if (blendIndicesRead) {
        uint32_t weightCount = 0;
        for (uint32_t component = 0; component < 4; component++) {
          weightCount += (weightComponentsRead >> component) & 1u;
        }
        result.skinInfluenceCount = std::max(weightCount, 1u);
      }

      const DxsoCtab& ctab = decoder.getCtabInfo();
      if (ctab.m_size == 0 || ctab.m_constantData.empty())
        return result;

      auto lower = [](const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (const char c : s)
          out.push_back(char(std::tolower(static_cast<unsigned char>(c))));
        return out;
      };

      auto contains = [](const std::string& s, const char* needle) {
        return s.find(needle) != std::string::npos;
      };

      bool hasViewProjectionMatrix = false;
      bool hasCameraPosition = false;
      uint32_t viewProjRegister = 0;
      uint32_t viewOriginRegister = 0;

      for (const DxsoCtab::Constant& c : ctab.m_constantData) {
        const std::string name = lower(c.name);

        if (!hasViewProjectionMatrix && c.registerCount >= 4) {
          const bool looksLikeViewProj =
            contains(name, "viewprojectionmatrix") ||
            contains(name, "viewprojmatrix") ||
            contains(name, "view_projection_matrix") ||
            contains(name, "view_proj_matrix");
          const bool isPreviousViewProj =
            contains(name, "prevviewprojectionmatrix") ||
            contains(name, "prevviewprojmatrix") ||
            contains(name, "previousviewprojectionmatrix") ||
            contains(name, "previousviewprojmatrix") ||
            contains(name, "prev_view_projection_matrix") ||
            contains(name, "prev_view_proj_matrix");
          if (looksLikeViewProj && !isPreviousViewProj) {
            hasViewProjectionMatrix = true;
            viewProjRegister = c.registerIndex;
          }
        }

        if (!hasCameraPosition && c.registerCount >= 1) {
          const bool looksLikeCameraPosition =
            contains(name, "cameraposition") ||
            contains(name, "vieworigin") ||
            contains(name, "cameraworldpos") ||
            contains(name, "cameraworldposition") ||
            contains(name, "camerapos") ||
            contains(name, "eyeposition");
          const bool isPreviousCameraPosition =
            contains(name, "prevcameraposition") ||
            contains(name, "prevvieworigin") ||
            contains(name, "previouscameraposition") ||
            contains(name, "previousvieworigin") ||
            contains(name, "prevcameraworldposition") ||
            contains(name, "previouseyeposition");
          if (looksLikeCameraPosition && !isPreviousCameraPosition) {
            hasCameraPosition = true;
            viewOriginRegister = c.registerIndex;
          }
        }

        // Object velocity capture inputs: rigid LocalToWorld (4 registers), the optional
        // WorldToLocal basis (transpose disambiguation), and the skinning marker that
        // excludes a draw from rigid capture
        if (!result.hasLocalToWorld && c.registerCount >= 4 &&
            (contains(name, "localtoworld") || contains(name, "local_to_world")) &&
            !contains(name, "prev")) {
          result.hasLocalToWorld = true;
          result.localToWorldRegister = c.registerIndex;
        }

        if (!result.hasWorldToLocal && c.registerCount >= 3 &&
            (contains(name, "worldtolocal") || contains(name, "world_to_local"))) {
          result.hasWorldToLocal = true;
          result.worldToLocalRegister = c.registerIndex;
        }

        if (!result.hasBoneMatrices && c.registerCount >= 3 && contains(name, "bone")) {
          result.hasBoneMatrices = true;
          result.boneMatricesRegister = c.registerIndex;
          result.boneMatricesRegisterCount = c.registerCount;
        }
      }

      result.hasViewProjection = hasViewProjectionMatrix;
      if (hasViewProjectionMatrix) {
        result.viewProjRegister = viewProjRegister;
      }

      if (hasViewProjectionMatrix && hasCameraPosition) {
        result.ctabVerified = true;
        result.viewOriginRegister = viewOriginRegister;
      }
    } catch (...) {
      return result;
    }

    return result;
  }

  void D3D9Rtx::tryNgxPassthroughCameraCapture() {
    const D3D9CommonShader* vertexShaderCommon = d3d9State().vertexShader->GetCommonShader();
    if (vertexShaderCommon == nullptr)
      return;

    const XXH64_hash_t shaderHash = vertexShaderCommon->GetBytecodeHash();
    if (shaderHash == 0)
      return;

    auto it = m_ngxCameraCtabCache.find(shaderHash);
    if (it == m_ngxCameraCtabCache.end()) {
      it = m_ngxCameraCtabCache.emplace(shaderHash, scanNgxCameraCtabRegs(vertexShaderCommon->GetBytecode())).first;
    }

    const NgxCameraCtabRegs& ctabRegs = it->second;
    if (!ctabRegs.ctabVerified)
      return;

    const uint32_t viewProjReg = ctabRegs.viewProjRegister;
    const uint32_t viewOriginReg = ctabRegs.viewOriginRegister;

    if (viewProjReg + 3 >= caps::MaxFloatConstantsSoftware || viewOriginReg >= caps::MaxFloatConstantsSoftware)
      return;

    // Mirrored view rejection (SceneCapture reflection/portal probes premultiply a mirror
    // matrix into the view, flipping the 3x3 determinant sign; the main view is always
    // positive). Checked on the raw registers - the sign is transpose-invariant.
    {
      const Vector4& vpRow0 = d3d9State().vsConsts.fConsts[viewProjReg + 0];
      const Vector4& vpRow1 = d3d9State().vsConsts.fConsts[viewProjReg + 1];
      const Vector4& vpRow2 = d3d9State().vsConsts.fConsts[viewProjReg + 2];
      const float vpDet3 =
        vpRow0.x * (vpRow1.y * vpRow2.z - vpRow1.z * vpRow2.y) -
        vpRow0.y * (vpRow1.x * vpRow2.z - vpRow1.z * vpRow2.x) +
        vpRow0.z * (vpRow1.x * vpRow2.y - vpRow1.y * vpRow2.x);
      if (!std::isfinite(vpDet3) || vpDet3 < 0.0f)
        return;
    }

    // Only the main scene view may steer the main camera and scene targets. It renders at
    // backbuffer * ScreenPercentage / 100 (full at 100), while auxiliary camera passes -
    // shadow cascades, SceneCapture probes, velocity/blur - use unrelated sizes. Matching the
    // viewport against the known ScreenPercentage separates them; a viewport-fraction gate
    // cannot, since a low ScreenPercentage main view is as small as an auxiliary pass.
    if (m_activePresentParams.has_value()) {
      const D3DVIEWPORT9& vp = d3d9State().viewport;
      const uint32_t bbW = m_activePresentParams->BackBufferWidth;
      const uint32_t bbH = m_activePresentParams->BackBufferHeight;

      if (bbW != 0 && bbH != 0) {
        const float sp = m_ngxGameScreenPercentage;
        if (sp > 0.0f && sp <= 100.0f) {
          const int32_t expectedW = int32_t(float(bbW) * sp / 100.0f);
          const int32_t expectedH = int32_t(float(bbH) * sp / 100.0f);
          const int32_t sizeTolerance = 16;
          const bool viewportMatchesExpected =
            int32_t(vp.Width)  >= expectedW - sizeTolerance && int32_t(vp.Width)  <= expectedW + sizeTolerance &&
            int32_t(vp.Height) >= expectedH - sizeTolerance && int32_t(vp.Height) <= expectedH + sizeTolerance;
          if (!viewportMatchesExpected) {
            // The D3D9 layer may have updated the isolated ScreenPercentage shadow before the
            // game has consumed it (typically one frame). Accept the main view at full
            // backbuffer size during that window so scene targets and the camera stay valid;
            // once the viewport shrinks the gate enforces the reduced size and Super
            // Resolution engages.
            const bool viewportNearFullSize =
              int32_t(vp.Width) >= int32_t(bbW) - sizeTolerance &&
              int32_t(vp.Height) >= int32_t(bbH) - sizeTolerance;
            const bool expectingReducedResolution = sp < 99.5f;
            if (!(expectingReducedResolution && viewportNearFullSize))
              return;
          }
        } else if (vp.Width * 2 < bbW || vp.Height * 2 < bbH) {
          // ScreenPercentage not resolved yet (first frame): fall back to the near-backbuffer
          // heuristic, safe for >= 50% and self-correcting once it resolves.
          return;
        }
      }
    }

    Matrix4 worldToView;
    Matrix4 viewToProjection;
    bool usedTranspose = false;
    float reconstructionError = 0.0f;
    XXH64_hash_t constantsHash = 0;

    if (!tryGetUe3CameraFromConstantsCached(viewProjReg, viewOriginReg, worldToView, viewToProjection,
                                            usedTranspose, reconstructionError, &constantsHash))
      return;

    // Record the scene color/depth targets from depth-writing scene draws: the depth image
    // feeds the motion vector pass, the color image anchors the viewport jitter scope, the
    // resolve tracking, and the pre-post-process trigger
    if (d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE &&
        d3d9State().renderTargets[kRenderTargetIndex] != nullptr &&
        d3d9State().depthStencil != nullptr) {
      D3D9CommonTexture* renderTargetTexture = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
      D3D9CommonTexture* depthStencilTexture = d3d9State().depthStencil->GetCommonTexture();

      if (renderTargetTexture != nullptr && depthStencilTexture != nullptr &&
          renderTargetTexture->GetImage() != nullptr && depthStencilTexture->GetImage() != nullptr) {
        // The frame's accepted camera (validity gate for the velocity capture, transpose
        // flag for LocalToWorld disambiguation). Only depth-writing draws with bound
        // color+depth targets donate it: the velocity raster depth-tests against the depth
        // these very draws produce, so their constants are by construction the camera that
        // buffer was rasterized with. Draws outside the scene (utility passes early in the
        // frame) may carry stale view constants from the previous frame.
        if (!m_ngxFrameCameraValid) {
          m_ngxFrameCameraValid = true;
          m_ngxFrameCameraUsedTranspose = usedTranspose;
        }

        const bool sceneTargetsChanged = m_ngxSceneColorImage != renderTargetTexture->GetImage();

        m_ngxSceneColorImage = renderTargetTexture->GetImage();
        m_ngxSceneDepthImage = depthStencilTexture->GetImage();
        m_ngxSceneTargetsLastSeenFrame = m_ue3FrameCounter;

        // The viewport of the scene draws is the subrect the game renders into; when the
        // game runs with a reduced ScreenPercentage this is smaller than the backbuffer
        m_ngxSceneViewport = d3d9State().viewport;
        m_ngxSceneViewportValid = true;
        m_ngxSceneViewportLastValidFrame = m_ue3FrameCounter;

        if (sceneTargetsChanged) {
          ONCE(Logger::info(str::format("[RTX NGX Passthrough] Scene color/depth targets identified. color=0x",
                                        std::hex, uintptr_t(m_ngxSceneColorImage.ptr()),
                                        " depth=0x", uintptr_t(m_ngxSceneDepthImage.ptr()), std::dec,
                                        " extent=", m_ngxSceneColorImage->info().extent.width, "x",
                                        m_ngxSceneColorImage->info().extent.height,
                                        " format=", int(m_ngxSceneColorImage->info().format))));
          // Automatic dump on target changes (level transitions): validates the pre-post
          // injection point against the game's compositing without user action. Budgeted, because
          // a title that alternates between two scene color surfaces changes target every frame
          // and would re-arm this forever - Mirror's Edge does exactly that.
          if (m_frameOptions.ngxPrePostProcess && m_ngxAutoDumpArmsRemaining > 0) {
            m_ngxAutoDumpArmsRemaining--;
            m_ngxPostChainDumpFramesLeft = 2;
          }
          // Rebind the viewport so this draw already gets the sub-pixel jitter
          m_parent->m_flags.set(D3D9DeviceFlag::DirtyViewportScissor);
        }
      }
    }

    // Feed the main camera once per unique constants; RtCamera keeps the first update per frame
    if (constantsHash == m_ngxLastCameraConstantsHash)
      return;
    m_ngxLastCameraConstantsHash = constantsHash;

    ONCE(Logger::info(str::format("[RTX NGX Passthrough] UE3 camera captured from shader constants (viewProjReg=c",
                                  viewProjReg, "..c", viewProjReg + 3, ", viewOriginReg=c", viewOriginReg, ").")));

    m_ngxFrameWorldToView = worldToView;
    m_ngxFrameViewToProjection = viewToProjection;
    m_ngxFrameCameraMatricesValid = true;

    m_parent->EmitCs([cWorldToView = worldToView, cViewToProjection = viewToProjection](DxvkContext* ctx) {
      static_cast<RtxContext*>(ctx)->getSceneManager().getCameraManager().processExternalCamera(
        CameraType::Main, cWorldToView, cViewToProjection);
    });
  }

  void D3D9Rtx::tryCaptureNgxVelocityDraw(const DrawContext& drawContext) {
    // Dynamic object draws for the velocity raster: indexed triangle lists into the scene
    // color with depth writes and a known camera, in both scene phases (world/intermediate
    // and, after the mid-scene depth clear, the foreground DPG). Rigid draws are captured
    // when their LocalToWorld moved believably since the previous frame; skinned draws
    // (UE3 GPU skin) additionally when their bone palette animated (see the emission
    // gates below). Sightings are matched against per-identity instance history.
    constexpr size_t kMaxVelocityDrawsPerFrame = 384;

    if (!m_frameOptions.ngxObjectVelocities) {
      return;
    }

    if (!drawContext.Indexed || drawContext.PrimitiveType != D3DPT_TRIANGLELIST || drawContext.PrimitiveCount == 0) {
      return;
    }

    if (m_ngxSceneColorImage == nullptr || d3d9State().renderTargets[kRenderTargetIndex] == nullptr) {
      return;
    }

    D3D9CommonTexture* renderTargetTexture = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
    if (renderTargetTexture == nullptr || renderTargetTexture->GetImage().ptr() != m_ngxSceneColorImage.ptr()) {
      return;
    }

    // A draw into the scene color through a viewport that cannot hold the scene is not the visible
    // scene. Some titles run proxy passes over the same geometry into that target through a
    // degenerate viewport, and those satisfy every other gate here. Capturing them costs twice
    // over: the sighting claims the identity's history before the real draw arrives, so the real
    // one finds nothing from the previous frame to pair with and loses its velocity, and whatever
    // velocity the proxy does emit carries that pass's transforms into the raster and lands on
    // unrelated pixels.
    if (m_activePresentParams.has_value()) {
      const uint32_t backBufferWidth = m_activePresentParams->BackBufferWidth;
      const uint32_t backBufferHeight = m_activePresentParams->BackBufferHeight;
      const D3DVIEWPORT9& vp = d3d9State().viewport;

      // A quarter of the backbuffer in each axis: below the most aggressive ScreenPercentage the
      // scene is ever driven to, and far above a proxy or probe pass.
      if (uint64_t(vp.Width) * 4 < backBufferWidth || uint64_t(vp.Height) * 4 < backBufferHeight) {
        ONCE(Logger::info(str::format(
          "[RTX NGX Passthrough] Ignoring scene color draws through a ", vp.Width, "x", vp.Height,
          " viewport for object velocity; the scene cannot be rendered through it.")));
        return;
      }
    }

    // Depth-tested draws only; depth WRITES are re-checked after classification (the
    // CPU-modified meshes' shaded pass composites without z-writes during motion blur,
    // with their depth coming from a prepass - the raster's two-sided depth test anchors
    // visibility either way). Scene draws with the CPU-modified-mesh buffer signature
    // (large dedicated dynamic VB) are counted when rejected here: a nonzero counter
    // means such meshes render z-test-disabled in some game state.
    if (d3d9State().renderStates[D3DRS_ZENABLE] != D3DZB_TRUE) {
      D3D9CommonBuffer* zGateVertexBuffer = GetCommonBuffer(d3d9State().vertexBuffers[0].vertexBuffer);
      D3D9CommonBuffer* zGateIndexBuffer = GetCommonBuffer(d3d9State().indices);
      if (zGateVertexBuffer != nullptr && zGateIndexBuffer != nullptr &&
          (zGateVertexBuffer->Desc()->Usage & D3DUSAGE_DYNAMIC) != 0 &&
          (zGateIndexBuffer->Desc()->Usage & D3DUSAGE_DYNAMIC) == 0 &&
          d3d9State().vertexBuffers[0].offset == 0 &&
          zGateVertexBuffer->Desc()->Size > 100000) {
        m_ngxVelocityStats.skippedZDisabled++;
      }
      return;
    }
    const bool zWriteEnabled = d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE;

    // Draws after the game's mid-scene depth clear are the UE3 foreground DPG (first
    // person meshes and their attachments); they depth-test against the live foreground
    // depth and carry the foreground phase marker in the velocity raster
    const bool foregroundPhase = m_ngxDepthSnapshotTakenThisFrame;

    // The remaining gates run after draw qualification so their skip counters describe
    // scene draws that would otherwise have been considered
    if (m_ngxVelocityDraws.size() >= kMaxVelocityDrawsPerFrame) {
      m_ngxVelocityStats.skippedBudget++;
      return;
    }

    if (!m_ngxFrameCameraValid || !m_ngxPrevCameraValid) {
      m_ngxVelocityStats.skippedNoCamera++;
      return;
    }

    if (!m_parent->UseProgrammableVS() || d3d9State().vertexShader.ptr() == nullptr || d3d9State().vertexDecl == nullptr) {
      return;
    }

    const D3D9CommonShader* vertexShaderCommon = d3d9State().vertexShader->GetCommonShader();
    if (vertexShaderCommon == nullptr) {
      return;
    }

    const XXH64_hash_t shaderHash = vertexShaderCommon->GetBytecodeHash();
    if (shaderHash == 0) {
      return;
    }

    auto ctabIt = m_ngxCameraCtabCache.find(shaderHash);
    if (ctabIt == m_ngxCameraCtabCache.end()) {
      ctabIt = m_ngxCameraCtabCache.emplace(shaderHash, scanNgxCameraCtabRegs(vertexShaderCommon->GetBytecode())).first;
    }

    const NgxCameraCtabRegs& ctabRegs = ctabIt->second;

    // Buffers and the CPU-modified-mesh shape, established early: the gate relaxations
    // below depend on them
    D3D9CommonBuffer* vertexBufferCommon = GetCommonBuffer(d3d9State().vertexBuffers[0].vertexBuffer);
    D3D9CommonBuffer* indexBufferCommon = GetCommonBuffer(d3d9State().indices);
    if (vertexBufferCommon == nullptr || indexBufferCommon == nullptr) {
      return;
    }

    const bool vbDynamic = (vertexBufferCommon->Desc()->Usage & D3DUSAGE_DYNAMIC) != 0;
    const bool ibDynamic = (indexBufferCommon->Desc()->Usage & D3DUSAGE_DYNAMIC) != 0;
    // UE3 gives a CPU-modified mesh a dedicated dynamic vertex buffer, and a dedicated dynamic
    // index buffer as well wherever cloth tearing is enabled, since torn triangles need new
    // indices. Requiring a static index buffer therefore rejects exactly the meshes whose motion
    // this path exists to carry. A dedicated buffer is told from a shared ring pool by the draw
    // covering a large part of it: a ring-pool draw takes a small slice of a big buffer, and it is
    // the shifting allocations within one that make that geometry untrackable to begin with.
    const uint32_t indexStride =
      static_cast<D3D9Format>(indexBufferCommon->Desc()->Format) == D3D9Format::INDEX32 ? 4u : 2u;
    const uint64_t drawIndexBytes = uint64_t(drawContext.PrimitiveCount) * 3u * indexStride;
    const bool indexBufferDedicated =
      !ibDynamic || drawIndexBytes * 4 >= uint64_t(indexBufferCommon->Desc()->Size);

    const bool dynamicMeshShape = vbDynamic && indexBufferDedicated &&
                                  !ctabRegs.hasBoneMatrices &&
                                  d3d9State().vertexBuffers[0].offset == 0;

    if (!ctabRegs.hasLocalToWorld) {
      return;
    }

    // The full camera verification (ViewProjection + CameraPosition) guards camera
    // steering; the velocity capture itself only needs the per-draw ViewProjection and
    // LocalToWorld. CPU-modified-mesh-shaped draws are accepted on that weaker contract
    // (their motion-blur shader variants may lack the camera position symbol).
    if (!ctabRegs.ctabVerified && !(dynamicMeshShape && ctabRegs.hasViewProjection)) {
      return;
    }

    if (ctabRegs.viewProjRegister + 3 >= caps::MaxFloatConstantsSoftware) {
      return;
    }

    // Skinned draws (UE3 GPU skin): motion is bone palette + rigid transform combined.
    // The velocity raster replays the skinning with both frames' palettes.
    const bool skinned = ctabRegs.hasBoneMatrices;

    if (skinned) {
      if (ctabRegs.boneMatricesRegisterCount == 0 ||
          ctabRegs.boneMatricesRegisterCount > kNgxVelocityBonePaletteRegisters ||
          ctabRegs.boneMatricesRegister + ctabRegs.boneMatricesRegisterCount > caps::MaxFloatConstantsSoftware) {
        // The palette cap follows stock UE3's 75 bones per chunk, but a licensee that needed a
        // richer rig may have raised it, and a mesh past the cap is dropped here while simpler
        // ones around it are captured - one character without velocity, everything else with.
        m_ngxVelocityStats.skippedBonePalette++;

        ONCE(Logger::info(str::format(
          "[RTX NGX Passthrough] Velocity capture rejecting a skinned draw over the bone palette "
          "cap: ", ctabRegs.boneMatricesRegisterCount, " registers (",
          ctabRegs.boneMatricesRegisterCount / 3, " bones) at register ",
          ctabRegs.boneMatricesRegister, ", cap ", kNgxVelocityBonePaletteRegisters, " (",
          kNgxVelocityBonePaletteRegisters / 3, " bones).")));
        return;
      }
      if (m_ngxVelocitySkinnedDraws >= kNgxVelocityMaxSkinnedDraws) {
        m_ngxVelocityStats.skippedBudget++;
        return;
      }
    }

    const uint32_t l2wReg = ctabRegs.localToWorldRegister;
    if (l2wReg + 3 >= caps::MaxFloatConstantsSoftware) {
      return;
    }

    const bool hasWorldToLocal = ctabRegs.hasWorldToLocal &&
                                 ctabRegs.worldToLocalRegister + 2 < caps::MaxFloatConstantsSoftware;

    // The packed ViewProjection at THIS draw (oriented): scene phases render with their
    // own projections (the foreground DPG uses a first-person FOV), so every velocity
    // draw composes with its draw-time matrix - the game's exact vertex transform.
    Matrix4 drawWorldToProjection;
    drawWorldToProjection[0] = d3d9State().vsConsts.fConsts[ctabRegs.viewProjRegister + 0];
    drawWorldToProjection[1] = d3d9State().vsConsts.fConsts[ctabRegs.viewProjRegister + 1];
    drawWorldToProjection[2] = d3d9State().vsConsts.fConsts[ctabRegs.viewProjRegister + 2];
    drawWorldToProjection[3] = d3d9State().vsConsts.fConsts[ctabRegs.viewProjRegister + 3];
    if (m_ngxFrameCameraUsedTranspose) {
      drawWorldToProjection = transpose(drawWorldToProjection);
    }

    // On-demand per-draw record of everything the gates below decide on, so a draw whose velocity
    // never reaches the screen can be told from one whose does. Armed by the user while the object
    // in question is on screen, because a one-shot log fires during startup instead and describes
    // menu frames.
    const bool dumpThisDraw = [&] {
      if (likely(m_ngxVelocityDumpFramesLeft == 0)) {
        return false;
      }
      if (skinned) {
        return m_ngxVelocityDumpSkinnedThisFrame++ < kNgxVelocityDumpMaxSkinnedPerFrame;
      }
      if (vbDynamic || ibDynamic) {
        return m_ngxVelocityDumpDynamicThisFrame++ < kNgxVelocityDumpMaxDynamicPerFrame;
      }
      return m_ngxVelocityDumpRigidThisFrame++ < kNgxVelocityDumpMaxRigidPerFrame;
    }();

    if (unlikely(dumpThisDraw)) {
      const D3DVIEWPORT9& vp = d3d9State().viewport;
      Logger::info(str::format(
        "[RTX NGX Passthrough][velocity dump] draw=", m_drawCallID,
        " prims=", drawContext.PrimitiveCount,
        " skinned=", skinned ? 1 : 0,
        " boneRegs=", skinned ? ctabRegs.boneMatricesRegisterCount : 0,
        " vbDynamic=", vbDynamic ? 1 : 0, " ibDynamic=", ibDynamic ? 1 : 0,
        " vbBytes=", vertexBufferCommon->Desc()->Size,
        " streamOffset=", d3d9State().vertexBuffers[0].offset,
        " zwrite=", d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE ? 1 : 0,
        " zfunc=", int(d3d9State().renderStates[D3DRS_ZFUNC]),
        " vp=", vp.X, ",", vp.Y, " ", vp.Width, "x", vp.Height,
        " z=", vp.MinZ, "..", vp.MaxZ,
        " sceneVp=", m_ngxSceneViewport.X, ",", m_ngxSceneViewport.Y, " ",
        m_ngxSceneViewport.Width, "x", m_ngxSceneViewport.Height));
    }

    // Vertex layout on stream 0: position always; skinned draws additionally need the
    // UE3 GPU skin blend attributes (UBYTE4 indices, UBYTE4N weights)
    const auto& vertexElements = d3d9State().vertexDecl->GetElements();
    uint32_t positionOffset = 0;
    VkFormat positionFormat = VK_FORMAT_UNDEFINED;
    uint32_t blendIndicesOffset = 0;
    uint32_t blendWeightsOffset = 0;
    bool hasBlendIndices = false;
    bool hasBlendWeights = false;

    for (const D3DVERTEXELEMENT9& element : vertexElements) {
      if (element.Stream != 0 || element.UsageIndex != 0) {
        continue;
      }

      if (element.Usage == D3DDECLUSAGE_POSITION) {
        if (element.Type == D3DDECLTYPE_FLOAT3) {
          positionFormat = VK_FORMAT_R32G32B32_SFLOAT;
        } else if (element.Type == D3DDECLTYPE_FLOAT4) {
          positionFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
        }
        positionOffset = element.Offset;
      } else if (element.Usage == D3DDECLUSAGE_BLENDINDICES && element.Type == D3DDECLTYPE_UBYTE4) {
        hasBlendIndices = true;
        blendIndicesOffset = element.Offset;
      } else if (element.Usage == D3DDECLUSAGE_BLENDWEIGHT && element.Type == D3DDECLTYPE_UBYTE4N) {
        hasBlendWeights = true;
        blendWeightsOffset = element.Offset;
      }
    }

    if (positionFormat == VK_FORMAT_UNDEFINED) {
      return;
    }

    if (skinned && (!hasBlendIndices || !hasBlendWeights)) {
      ONCE(Logger::info("[RTX NGX Passthrough] Skinned draw with an unsupported blend attribute layout; not captured."));
      return;
    }

    // Dynamic-buffer draws split into two kinds. CPU-modified meshes (UE3 CPU-skins
    // morph/cloth-augmented skeletal meshes into DEDICATED dynamic buffers, e.g. the
    // first person arms) carry their motion in the vertex positions and are captured
    // with a position snapshot below; they are recognizable by a bone-less shader, a
    // whole-buffer stream (offset 0) and a buffer the draw largely fills. Everything
    // else on dynamic buffers is ring-pool geometry (particles, trails, canvas) whose
    // allocation offsets shift every frame - untrackable, and skipped.
    const bool dynamicMesh = dynamicMeshShape && !skinned;

    if ((vbDynamic || ibDynamic) && !dynamicMesh) {
      // Ring-pool geometry: a dynamic buffer shared between draws whose allocations move every
      // frame, so nothing here can be matched against a previous sighting. Counted because a mesh
      // that carries its motion in its vertex positions is silently left without velocity if it
      // ever lands in this shape rather than being recognised above.
      m_ngxVelocityStats.skippedDynamicBuffer++;

      ONCE(Logger::info(str::format(
        "[RTX NGX Passthrough] Velocity capture rejecting a dynamic-buffer draw: vb dynamic=",
        vbDynamic ? 1 : 0, ", ib dynamic=", ibDynamic ? 1 : 0, ", bone palette=", skinned ? 1 : 0,
        ", stream offset=", d3d9State().vertexBuffers[0].offset,
        ", vb bytes=", vertexBufferCommon->Desc()->Size,
        ". Motion carried in vertex positions is invisible to the capture in this shape.")));
      return;
    }

    // Depth-writing draws only, except CPU-modified meshes (see the z gate above)
    if (!zWriteEnabled && !dynamicMesh) {
      return;
    }

    // Snapshot the CPU-modified mesh's current positions from the dynamic buffer's CPU
    // mapping (tightly packed, whole buffer: the index/base-vertex ranges then apply to
    // the snapshot exactly as to the live stream)
    std::vector<Vector3> currentPositions;

    if (dynamicMesh) {
      if (m_ngxVelocityDynamicDraws >= kNgxVelocityMaxDynamicDraws) {
        m_ngxVelocityStats.skippedBudget++;
        return;
      }

      const uint32_t stride = d3d9State().vertexBuffers[0].stride;
      const uint32_t vertexCount = stride != 0 ? uint32_t(vertexBufferCommon->Desc()->Size / stride) : 0;
      const DxvkBufferSliceHandle mappedSlice = vertexBufferCommon->GetMappedSlice();

      if (vertexCount == 0 || vertexCount > kNgxVelocityMaxDynamicVertices || mappedSlice.mapPtr == nullptr) {
        return;
      }

      currentPositions.resize(vertexCount);
      const uint8_t* positionBytes = reinterpret_cast<const uint8_t*>(mappedSlice.mapPtr) + positionOffset;
      for (uint32_t vertex = 0; vertex < vertexCount; vertex++) {
        std::memcpy(&currentPositions[vertex], positionBytes + size_t(vertex) * stride, sizeof(Vector3));
      }
    }

    // Draw identity: geometry references plus draw parameters. Stable for UE3 static
    // meshes; buffer reallocation across level loads simply re-registers the object.
    // CPU-modified meshes swap between vertex buffer objects when re-skinned (double
    // buffering), so their identity anchors on the buffer SIZE instead of its address -
    // the static index buffer and the draw parameters still separate mesh sections.
    // Hashed from an explicitly packed array: hashing a struct with pointer members would
    // include compiler tail padding, which aggregate initialization leaves uninitialized,
    // making identities jitter with incidental stack contents and breaking the pairing.
    const uint64_t identityData[5] = {
      dynamicMesh ? uint64_t(vertexBufferCommon->Desc()->Size)
                  : uint64_t(reinterpret_cast<uintptr_t>(vertexBufferCommon)),
      uint64_t(reinterpret_cast<uintptr_t>(indexBufferCommon)),
      (uint64_t(drawContext.StartIndex) << 32) | uint64_t(uint32_t(drawContext.PrimitiveCount)),
      (uint64_t(uint32_t(drawContext.BaseVertexIndex)) << 32) | uint64_t(d3d9State().vertexBuffers[0].stride),
      uint64_t(positionOffset),
    };
    const XXH64_hash_t identity = XXH3_64bits(identityData, sizeof(identityData));

    const Vector4* boneRegisters = skinned ? &d3d9State().vsConsts.fConsts[ctabRegs.boneMatricesRegister] : nullptr;
    const uint32_t boneRegisterCount = skinned ? ctabRegs.boneMatricesRegisterCount : 0;

    NgxVelocityObjectState& objectState = m_ngxVelocityObjectCache[identity];

    // Shared draw construction for the emission sites below. previousBones is the
    // instance's cached palette; when unusable (first skinned sighting of the identity),
    // the current palette stands in for both sides - transform-only motion that frame.
    const auto appendVelocityDraw = [&](const Matrix4& objectToWorldCurrent,
                                        const Matrix4& worldToProjectionPrevious,
                                        const Matrix4& objectToWorldPrevious,
                                        const std::vector<Vector4>& previousBones,
                                        const std::vector<Vector3>& previousPositions) {
      NgxVelocityDraw velocityDraw;
      velocityDraw.vertexBuffer = vertexBufferCommon->GetBufferSlice<D3D9_COMMON_BUFFER_TYPE_REAL>(d3d9State().vertexBuffers[0].offset);
      velocityDraw.vertexStride = d3d9State().vertexBuffers[0].stride;
      velocityDraw.positionOffset = positionOffset;
      velocityDraw.positionFormat = positionFormat;
      velocityDraw.indexBuffer = indexBufferCommon->GetBufferSlice<D3D9_COMMON_BUFFER_TYPE_REAL>();
      velocityDraw.indexType = DecodeIndexType(static_cast<D3D9Format>(indexBufferCommon->Desc()->Format));
      velocityDraw.indexCount = drawContext.PrimitiveCount * 3;
      velocityDraw.firstIndex = drawContext.StartIndex;
      velocityDraw.vertexOffset = drawContext.BaseVertexIndex;

      velocityDraw.clipFromLocal = drawWorldToProjection * objectToWorldCurrent;
      velocityDraw.prevClipFromLocal = worldToProjectionPrevious * objectToWorldPrevious;
      velocityDraw.foregroundPhase = foregroundPhase;
      velocityDraw.viewportMinZ = d3d9State().viewport.MinZ;
      velocityDraw.viewportMaxZ = d3d9State().viewport.MaxZ;

      if (foregroundPhase && (d3d9State().viewport.MinZ != 0.0f || d3d9State().viewport.MaxZ != 1.0f)) {
        ONCE(Logger::info(str::format("[RTX NGX Passthrough] Foreground DPG renders with a squashed viewport depth range: [",
                                      d3d9State().viewport.MinZ, ", ", d3d9State().viewport.MaxZ, "].")));
      }

      if (skinned) {
        velocityDraw.blendIndicesOffset = blendIndicesOffset;
        velocityDraw.blendWeightsOffset = blendWeightsOffset;
        velocityDraw.boneIndexScale = ctabRegs.boneIndicesPreScaled ? 1u : 3u;
        velocityDraw.skinInfluenceCount = std::clamp(ctabRegs.skinInfluenceCount, 1u, 4u);

        ONCE(Logger::info(str::format("[RTX NGX Passthrough] Skinned velocity capture active (bone index addressing: ",
                                      ctabRegs.boneIndicesPreScaled ? "pre-scaled" : "shader-scaled",
                                      ", influences: ", velocityDraw.skinInfluenceCount, ").")));

        velocityDraw.bonesPrevious.resize(kNgxVelocityBonePaletteRegisters, Vector4(0.0f));
        velocityDraw.bonesCurrent.resize(kNgxVelocityBonePaletteRegisters, Vector4(0.0f));

        const Vector4* previousPalette =
          previousBones.size() == boneRegisterCount ? previousBones.data() : boneRegisters;
        std::memcpy(velocityDraw.bonesPrevious.data(), previousPalette, boneRegisterCount * sizeof(Vector4));
        std::memcpy(velocityDraw.bonesCurrent.data(), boneRegisters, boneRegisterCount * sizeof(Vector4));

        m_ngxVelocitySkinnedDraws++;
        m_ngxVelocityStats.capturedSkinned++;
      }

      if (dynamicMesh) {
        velocityDraw.previousPositions =
          previousPositions.size() == currentPositions.size() ? previousPositions : currentPositions;
        m_ngxVelocityDynamicDraws++;
        m_ngxVelocityStats.capturedDynamic++;
      }

      if (foregroundPhase) {
        m_ngxVelocityStats.capturedForeground++;
      }

      m_ngxVelocityDraws.push_back(std::move(velocityDraw));
      m_ngxVelocityStats.captured++;
    };

    // Tolerances around the scene-wide transform offset, all in world units and all sized
    // against the same thing: how exactly a translation that did not move can be expected to
    // reproduce that offset. The cluster width is the floor, set by float precision on a
    // difference of two level-scale positions. A bone's translation is compared at the same
    // width, being the same kind of quantity. A placement is held to a wider bound because the
    // offset it is judged against was itself only measured to within the cluster width, and
    // because a residual has to stay far below the spacing between two copies of an asset to
    // keep them apart.
    constexpr float kOffsetClusterTolerance = 0.05f;
    constexpr float kBoneTranslationTolerance = kOffsetClusterTolerance;
    constexpr float kSamePlacementTolerance = 0.5f;

    // Position delta against a cached snapshot (CPU-modified meshes animate their
    // vertex data with a typically static LocalToWorld)
    const auto dynamicPositionsChangedFrom = [&](const std::vector<Vector3>& cachedPositions) {
      if (!dynamicMesh) {
        return false;
      }
      if (cachedPositions.size() != currentPositions.size()) {
        return true;
      }
      return std::memcmp(cachedPositions.data(), currentPositions.data(),
                         cachedPositions.size() * sizeof(Vector3)) != 0;
    };

    // Bone palette delta against a cached palette, net of a translation the whole palette is
    // expected to have taken. The palette holds bone-to-world matrices, so where that space
    // shifts every frame every bone's translation shifts with it, and a skinned mesh that never
    // moved reports its entire palette as animated - which emits velocity unconditionally,
    // ahead of every motion gate. Destructible and instanced scenery is commonly drawn through
    // skinned vertex factories, so that reads as level geometry moving under its own power.
    // Only the translation is affected, which the palette carries in .w, one axis per row of
    // each bone's three; the rotation rows are compared as they are.
    const auto bonesChangedFrom = [&](const std::vector<Vector4>& cachedBones,
                                      const Vector3& expectedTranslationShift) {
      if (!skinned || cachedBones.size() != boneRegisterCount) {
        return false;
      }
      for (uint32_t reg = 0; reg < boneRegisterCount; reg++) {
        const Vector4 delta = boneRegisters[reg] - cachedBones[reg];
        if (std::abs(delta.x) > 1e-5f || std::abs(delta.y) > 1e-5f || std::abs(delta.z) > 1e-5f) {
          return true;
        }

        const uint32_t axis = reg % 3;
        const float axisShift = axis == 0 ? expectedTranslationShift.x
                              : axis == 1 ? expectedTranslationShift.y
                                          : expectedTranslationShift.z;
        // Looser than the rotation rows: a bone's translation is a world position, and the
        // shift being subtracted was itself measured to within a fraction of a unit
        if (std::abs(delta.w - axisShift) > kBoneTranslationTolerance) {
          return true;
        }
      }
      return false;
    };

    // The furthest anything is credited with travelling in one frame, applied both to an
    // object pairing with its own past and to the scene's own coordinate space shifting.
    // Generous: a frame hitch multiplies every per-frame delta, and a delta pushed past the
    // bound costs two frames of velocity.
    constexpr float kMaxFrameTranslation = 250.0f;  // world units per frame

    const Matrix4 sightingObjectToWorld =
      extractUe3ObjectToWorld(l2wReg, hasWorldToLocal, ctabRegs.worldToLocalRegister, m_ngxFrameCameraUsedTranspose);

    // Every placement that held still moved by the same global offset, so the offset is whatever
    // the most sightings agree on. Movers disagree with each other as much as with the static
    // scene, which is what keeps them out of the winning slot.
    auto voteForGlobalTransformOffset = [&](const Vector3& delta) {
      // One vote per identity per frame. Copies of a single instanced asset sit on a shared
      // grid, so when the visible set shifts they pair with their neighbours and agree, in
      // bulk, on a delta that is one grid step rather than the scene's: counted per sighting
      // that bloc outvotes the real offset outright. Counted per identity it is worth one
      // voice, and the offset is instead carried by how many unrelated assets report it.
      if (objectState.lastOffsetVoteFrame == m_ue3FrameCounter) {
        return;
      }

      // No camera crosses a level in a frame. A delta this large is a pairing across two
      // placements, and adopting it would put the same error into every reprojected pixel.
      if (lengthSqr(delta) > kMaxFrameTranslation * kMaxFrameTranslation) {
        return;
      }

      objectState.lastOffsetVoteFrame = m_ue3FrameCounter;

      NgxTranslationDeltaVote* slot = nullptr;
      NgxTranslationDeltaVote* freeSlot = nullptr;
      for (NgxTranslationDeltaVote& vote : m_ngxTranslationDeltaVotes) {
        if (vote.votes == 0) {
          freeSlot = freeSlot != nullptr ? freeSlot : &vote;
          continue;
        }
        // Grouped loosely enough to survive float precision at level-scale coordinates, where
        // a difference of two positions in the tens of thousands already carries thousandths
        // of error: too tight and the far reaches of a level fragment one true offset across
        // several slots, none of them carrying enough votes to be believed.
        if (lengthSqr(vote.delta - delta) <= kOffsetClusterTolerance * kOffsetClusterTolerance) {
          slot = &vote;
          break;
        }
      }

      if (slot != nullptr) {
        slot->votes++;
      } else if (freeSlot != nullptr) {
        freeSlot->delta = delta;
        freeSlot->votes = 1;
        slot = freeSlot;
      } else {
        // Every slot is held by a candidate this one disagrees with; nothing to record it in
        return;
      }

      // Follow the leader from its opening vote rather than waiting for a quorum: until the
      // frame has adopted an offset of its own the placement test is judging against the
      // previous frame's, which differs by however much the camera accelerated, and a static
      // placement that fails that test emits a velocity it should not. An opening vote may
      // come from a mover, so it only stands until two sightings agree on something else.
      if (slot->votes > m_ngxAdoptedOffsetVotes) {
        m_ngxGlobalTransformOffset = slot->delta;
        m_ngxAdoptedOffsetVotes = slot->votes;
      }
    };

    // Same placement: the basis is untouched and the translation moved by exactly what the rest
    // of the scene moved by. A repeat draw within the frame has not moved at all; one carried
    // over from the previous frame moved by the global offset, which is zero wherever
    // LocalToWorld is true world space and the comparison is then plain equality.
    auto isSamePlacement = [&](const NgxVelocityObjectInstance& instance) {
      for (uint32_t col = 0; col < 3; col++) {
        const Vector4 delta = sightingObjectToWorld[col] - instance.objectToWorld[col];
        if (std::abs(delta.x) > 1e-4f || std::abs(delta.y) > 1e-4f || std::abs(delta.z) > 1e-4f) {
          return false;
        }
      }

      Vector3 expectedMove(0.0f, 0.0f, 0.0f);
      if (instance.lastSeenFrame == m_ue3FrameCounter - 1) {
        expectedMove = m_ngxGlobalTransformOffset;
      } else if (instance.lastSeenFrame != m_ue3FrameCounter) {
        // Older history spans an unknown number of global offsets, so it cannot be judged here;
        // the near match below pairs it if the motion is believable
        return false;
      }

      // A placement that held still reproduces the offset to float precision, so this only has
      // to absorb rounding at level-scale coordinates - staying far below the slowest motion
      // worth a velocity, and far below the spacing between two copies of an instanced asset
      const Vector3 residual = (sightingObjectToWorld[3].xyz() - instance.objectToWorld[3].xyz()) - expectedMove;
      return lengthSqr(residual) <= kSamePlacementTolerance * kSamePlacementTolerance;
    };

    // Static placement re-uploading the same matrix every frame, or a repeat draw of an instance
    // already handled this frame (depth prepass + base pass, multi-pass lighting). Static
    // placements get no velocity draw - the camera reprojection is exact for them. Skinned
    // instances with a changed palette are animation in place: same placement, no swap
    // ambiguity, always emitted.
    for (NgxVelocityObjectInstance& instance : objectState.instances) {
      if (isSamePlacement(instance)) {
        // Static placement, but the content may animate in place: skinned palettes or
        // CPU-modified positions changing under an unchanged LocalToWorld - no swap
        // ambiguity, always emitted
        // A repeat draw within the frame has taken no shift; one carried over from the previous
        // frame has taken the scene's, exactly as isSamePlacement judged its transform
        const Vector3 expectedBoneShift = instance.lastSeenFrame == m_ue3FrameCounter
                                            ? Vector3(0.0f, 0.0f, 0.0f)
                                            : m_ngxGlobalTransformOffset;
        const bool contentAnimated = (skinned && bonesChangedFrom(instance.bones, expectedBoneShift)) ||
                                     (dynamicMesh && dynamicPositionsChangedFrom(instance.dynamicPositions));

        if (contentAnimated) {
          appendVelocityDraw(sightingObjectToWorld, instance.worldToProjection, instance.objectToWorld,
                             instance.bones, instance.dynamicPositions);
          instance.lastEmitFrame = m_ue3FrameCounter;
          instance.lastEmitDrawIndex = uint32_t(m_ngxVelocityDraws.size() - 1);
          instance.lastEmitPrevObjectToWorld = instance.objectToWorld;
          instance.lastEmitPrevWorldToProjection = instance.worldToProjection;
        }

        if (skinned) {
          instance.bones.assign(boneRegisters, boneRegisters + boneRegisterCount);
        }
        if (dynamicMesh) {
          // Last use of the snapshot (appendVelocityDraw above copied what it needed)
          instance.dynamicPositions = std::move(currentPositions);
        }

        // Same-frame repeat draw in the other scene phase: re-emit this frame's draw for
        // this phase so both velocity targets receive the object, with both clip
        // transforms recomposed for this phase's projection (the foreground DPG renders
        // with its own first-person projection; each target depth-tests against its own
        // phase's depth, discarding any copy that does not belong)
        if (instance.lastEmitFrame == m_ue3FrameCounter &&
            instance.lastPhaseDuplicateFrame != m_ue3FrameCounter &&
            instance.lastEmitDrawIndex < m_ngxVelocityDraws.size() &&
            m_ngxVelocityDraws.size() < kMaxVelocityDrawsPerFrame) {
          const NgxVelocityDraw& emittedDraw = m_ngxVelocityDraws[instance.lastEmitDrawIndex];

          if (emittedDraw.foregroundPhase != foregroundPhase &&
              (emittedDraw.bonesCurrent.empty() || m_ngxVelocitySkinnedDraws < kNgxVelocityMaxSkinnedDraws) &&
              (emittedDraw.previousPositions.empty() || m_ngxVelocityDynamicDraws < kNgxVelocityMaxDynamicDraws)) {
            NgxVelocityDraw duplicateDraw = emittedDraw;
            duplicateDraw.foregroundPhase = foregroundPhase;
            duplicateDraw.clipFromLocal = drawWorldToProjection * instance.objectToWorld;
            duplicateDraw.prevClipFromLocal = instance.lastEmitPrevWorldToProjection * instance.lastEmitPrevObjectToWorld;
            duplicateDraw.viewportMinZ = d3d9State().viewport.MinZ;
            duplicateDraw.viewportMaxZ = d3d9State().viewport.MaxZ;

            if (!duplicateDraw.bonesCurrent.empty()) {
              m_ngxVelocitySkinnedDraws++;
              m_ngxVelocityStats.capturedSkinned++;
            }
            if (!duplicateDraw.previousPositions.empty()) {
              m_ngxVelocityDynamicDraws++;
              m_ngxVelocityStats.capturedDynamic++;
            }
            if (duplicateDraw.foregroundPhase) {
              m_ngxVelocityStats.capturedForeground++;
            }

            m_ngxVelocityDraws.push_back(std::move(duplicateDraw));
            m_ngxVelocityStats.captured++;
            instance.lastPhaseDuplicateFrame = m_ue3FrameCounter;
          }
        }

        // The placement is the same one, but where LocalToWorld carries a global offset its
        // matrix is not: it has to be carried forward, or the stored transform ages by a frame
        // every frame and the offset the next sighting has to reproduce compounds out of reach
        if (instance.lastSeenFrame != m_ue3FrameCounter) {
          voteForGlobalTransformOffset(sightingObjectToWorld[3].xyz() - instance.objectToWorld[3].xyz());
        }
        instance.objectToWorld = sightingObjectToWorld;
        instance.lastSeenFrame = m_ue3FrameCounter;
        instance.worldToProjection = drawWorldToProjection;

        // Confirmed-mover status decays after a couple seconds at rest: a stale latch
        // is an instant-emit backdoor for visibility swaps pairing against this
        // instance, while a real mover restarting re-confirms through the gentle-onset
        // path with zero frames lost (motion from rest is always gentle at first)
        constexpr uint32_t kWasMovingDecayFrames = 120;
        if (instance.wasMoving && m_ue3FrameCounter - instance.lastEmitFrame > kWasMovingDecayFrames) {
          instance.wasMoving = false;
        }

        m_ngxVelocityStats.exactMatches++;
        return;
      }
    }

    // Disambiguated transform in the same convention the camera reconstruction uses;
    // composed exactly like the motion vector pass composes its reprojection chain
    const Matrix4& objectToWorld = sightingObjectToWorld;

    // Near match against instances seen exactly one frame ago: the same object having
    // moved. Static placements are consumed by the exact match above, so the bounds only
    // arbitrate between simultaneously moving identical movers - the nearest-score pairs
    // each with its own history even when both fit the bounds (double door leaves).
    constexpr float kMaxFrameRotScaleL1 = 3.0f;     // L1 delta over the 3x3 rotation/scale
    const uint32_t previousFrame = m_ue3FrameCounter - 1;

    NgxVelocityObjectInstance* matchedInstance = nullptr;
    float matchedScore = 0.0f;
    float matchedTranslationDelta = 0.0f;
    float matchedRotScaleDelta = 0.0f;

    // Nearest last-frame candidate regardless of bounds: consumed by the single-candidate
    // acceptance below and by the pairing-miss diagnostics
    NgxVelocityObjectInstance* nearestCandidate = nullptr;
    float nearestScore = 0.0f;
    float nearestTranslationDelta = 0.0f;
    float nearestRotScaleDelta = 0.0f;
    uint32_t lastFrameCandidates = 0;

    // Placements sighted within the last couple frames: the cache also holds stale
    // instances from previously visited areas (pruned lazily), which must not count
    // against the identity-stability assessment below
    uint32_t activeInstances = 0;

    // Frames since the freshest sighting of this identity, for the pairing-miss diagnostics.
    // With no last-frame candidate the question is whether the history is one frame too old -
    // the capture skipping frames - or absent entirely, which is a different fault.
    uint32_t freshestSightingAge = UINT32_MAX;

    for (NgxVelocityObjectInstance& instance : objectState.instances) {
      freshestSightingAge = std::min(freshestSightingAge, m_ue3FrameCounter - instance.lastSeenFrame);

      if (instance.lastSeenFrame + 2 >= m_ue3FrameCounter) {
        activeInstances++;
      }

      if (instance.lastSeenFrame != previousFrame) {
        continue;
      }
      lastFrameCandidates++;

      // Motion is what is left after the offset the scene as a whole moved by, so a placement
      // that held still reads as motionless here however far the space its transform is
      // expressed in travelled. Measured raw, every static placement that misses the
      // same-placement test above - through an evicted slot, an ambiguity between copies, or
      // simply never having been seen before - instead reads as moving by the camera's own
      // travel. That is gentle enough to pass the onset gate below, and emitting once sets
      // wasMoving, which keeps it emitting from then on.
      const Vector3 rawTranslationDelta = objectToWorld[3].xyz() - instance.objectToWorld[3].xyz();
      const float translationDelta = length(rawTranslationDelta - m_ngxGlobalTransformOffset);

      float rotScaleDelta = 0.0f;
      for (uint32_t col = 0; col < 3; col++) {
        const Vector4 delta = objectToWorld[col] - instance.objectToWorld[col];
        rotScaleDelta += std::abs(delta.x) + std::abs(delta.y) + std::abs(delta.z);
      }

      // The offset itself is measured raw, and from here rather than only from the matches
      // above, so an estimate that has gone stale can still be re-measured: were it fed only
      // by sightings the estimate itself admitted, a wrong estimate would reject every
      // sighting and never be corrected
      if (rotScaleDelta <= 1e-4f) {
        voteForGlobalTransformOffset(rawTranslationDelta);
      }

      const float score = translationDelta / kMaxFrameTranslation + rotScaleDelta / kMaxFrameRotScaleL1;

      if (nearestCandidate == nullptr || score < nearestScore) {
        nearestCandidate = &instance;
        nearestScore = score;
        nearestTranslationDelta = translationDelta;
        nearestRotScaleDelta = rotScaleDelta;
      }

      if (translationDelta > kMaxFrameTranslation || rotScaleDelta > kMaxFrameRotScaleL1) {
        continue;
      }

      if (matchedInstance == nullptr || score < matchedScore) {
        matchedInstance = &instance;
        matchedScore = score;
        matchedTranslationDelta = translationDelta;
        matchedRotScaleDelta = rotScaleDelta;
      }
    }

    // The bounds exist to disambiguate between multiple placements sharing an identity;
    // with exactly one unclaimed last-frame instance the pairing is unambiguous by
    // elimination, so faster-than-bounds motion (trains cover hundreds of units per
    // frame; frame hitches multiply every delta) may pair too.
    bool pairedBeyondBounds = false;
    if (matchedInstance == nullptr && lastFrameCandidates == 1) {
      matchedInstance = nearestCandidate;
      matchedTranslationDelta = nearestTranslationDelta;
      matchedRotScaleDelta = nearestRotScaleDelta;
      pairedBeyondBounds = matchedInstance != nullptr;
      if (pairedBeyondBounds) {
        m_ngxVelocityStats.pairedBeyondBounds++;
      }
    }

    if (matchedInstance != nullptr) {
      // Net of the scene-wide offset, matching matchedTranslationDelta and the stored history
      // this is compared against on the next sighting
      const Vector3 moveDelta =
        (objectToWorld[3].xyz() - matchedInstance->objectToWorld[3].xyz()) - m_ngxGlobalTransformOffset;

      // Emission gate: a pairing only produces velocity when the motion is believable.
      //  - Negligible deltas (one-time transform settles) are claimed silently: sub-pixel
      //    motion is served equally well by camera reprojection.
      //  - Confirmed movers (wasMoving) emit on sight.
      //  - Unconfirmed instances emit immediately only for gentle motion, plausible for
      //    an object accelerating from rest. Larger first deltas - what a visibility swap
      //    between two placements of the same asset looks like - must repeat consistently
      //    for one frame first: a real fast mover sustains its per-frame delta, a swap
      //    does not repeat.
      //  - Pairings taken by elimination take that confirmation path unconditionally.
      constexpr float kNegligibleTranslation = 0.05f;
      constexpr float kNegligibleRotScale = 1e-3f;
      constexpr float kOnsetTranslationTrust = 8.0f;   // ~480 units/s at 60 fps
      constexpr float kOnsetRotScaleTrust = 0.35f;     // ~3 degrees/frame

      // Animating content (skinned palettes / CPU-modified positions changing) rules out a
      // swap between two static copies of an asset, whose vertex data never animates, so a
      // pairing that stayed within the bounds skips the anti-swap gates below (first person
      // meshes pair through camera-attached transform deltas that routinely exceed them).
      // It cannot rule out a swap between two animated copies - see pairedBeyondBounds.
      // Paired against a last-frame instance by construction, so the scene's shift applies
      const bool contentAnimated = bonesChangedFrom(matchedInstance->bones, m_ngxGlobalTransformOffset) ||
                                   dynamicPositionsChangedFrom(matchedInstance->dynamicPositions);

      const bool negligibleMotion = !contentAnimated &&
                                    matchedTranslationDelta <= kNegligibleTranslation &&
                                    matchedRotScaleDelta <= kNegligibleRotScale;

      bool emitVelocity = false;

      // A pairing taken by elimination sits beyond the per-frame bounds, so it is as much the
      // shape of one placement leaving view as another arrives as it is of real motion, and
      // nothing about the sighting itself tells them apart. Only the delta repeating does -
      // a fast mover sustains it, a visibility swap does not - so these take the confirmation
      // path below regardless of what else vouches for them. Animating content does not: two
      // copies of one skeletal asset both animate, and pairing across them emits the distance
      // between two different characters as a single frame of motion. Neither does wasMoving,
      // which describes the instance's own past and not this sighting's claim to it.
      if (contentAnimated && !pairedBeyondBounds) {
        emitVelocity = true;
      } else if (!negligibleMotion) {
        if (matchedInstance->wasMoving && !pairedBeyondBounds) {
          emitVelocity = true;
        } else if (!pairedBeyondBounds &&
                   matchedTranslationDelta <= kOnsetTranslationTrust &&
                   matchedRotScaleDelta <= kOnsetRotScaleTrust) {
          emitVelocity = true;
        } else {
          // Consistency confirmation is only trusted for identities with few placements
          // currently in view: real movers have one or two, while grids of instanced
          // meshes produce repeating pop-in deltas under steady camera movement that
          // pass any repetition test (those may only confirm through gentle onset, which
          // pop-in distances can never satisfy). A single active placement is
          // unambiguous and skips the churn-recency requirement - its own first
          // registration would otherwise defer its confirmation.
          const bool identityStable = activeInstances <= 4 &&
                                      (activeInstances <= 1 ||
                                       m_ue3FrameCounter > objectState.lastNewRegistrationFrame + 2);

          const bool hadMotion = lengthSqr(matchedInstance->lastMoveDelta) > 0.0f ||
                                 matchedInstance->lastRotScaleDelta > 0.0f;
          const bool translationConsistent =
            length(moveDelta - matchedInstance->lastMoveDelta) <= std::max(0.25f * matchedTranslationDelta, 2.0f);
          const bool rotScaleConsistent =
            std::abs(matchedRotScaleDelta - matchedInstance->lastRotScaleDelta) <= std::max(0.25f * matchedRotScaleDelta, 0.05f);

          emitVelocity = identityStable && hadMotion && translationConsistent && rotScaleConsistent;
        }
      }

      if (emitVelocity) {
        appendVelocityDraw(objectToWorld, matchedInstance->worldToProjection, matchedInstance->objectToWorld,
                           matchedInstance->bones, matchedInstance->dynamicPositions);
        matchedInstance->lastEmitPrevObjectToWorld = matchedInstance->objectToWorld;
        matchedInstance->lastEmitPrevWorldToProjection = matchedInstance->worldToProjection;
      } else {
        // Claimed without velocity (negligible motion or deferred onset confirmation)
        m_ngxVelocityStats.claimedWithoutVelocity++;
      }

      // Claim the instance: repeat draws this frame match the updated transform, other
      // instances cannot pair with it anymore. The movement history feeds the emission
      // gate above on the next sighting.
      matchedInstance->objectToWorld = objectToWorld;
      matchedInstance->worldToProjection = drawWorldToProjection;
      matchedInstance->lastSeenFrame = m_ue3FrameCounter;
      matchedInstance->lastMoveDelta = moveDelta;
      matchedInstance->lastRotScaleDelta = matchedRotScaleDelta;
      if (skinned) {
        matchedInstance->bones.assign(boneRegisters, boneRegisters + boneRegisterCount);
      } else {
        matchedInstance->bones.clear();
      }
      if (dynamicMesh) {
        // Last use of the snapshot (appendVelocityDraw above copied what it needed)
        matchedInstance->dynamicPositions = std::move(currentPositions);
      } else {
        matchedInstance->dynamicPositions.clear();
      }
      if (emitVelocity) {
        matchedInstance->wasMoving = true;
        matchedInstance->lastEmitFrame = m_ue3FrameCounter;
        matchedInstance->lastEmitDrawIndex = uint32_t(m_ngxVelocityDraws.size() - 1);
      }
      return;
    }

    // Self-triggering pairing-miss dump: a moving object failing to pair with its own
    // one-frame-ago history is the exact failure mode behind velocity dropouts, and the
    // log line carries everything needed to tell apart the possible causes (no last-frame
    // sighting at all vs a candidate rejected by the bounds, and by how much).
    {
      constexpr uint32_t kPairingLogMaxLinesPerFrame = 6;
      constexpr uint32_t kPairingLogCooldownFrames = 120;

      // One burst of lines per cooldown window: the burst frame is latched and its
      // remaining lines stay allowed, further frames wait out the cooldown
      const bool inActiveBurst = m_ngxVelocityPairingLogFrame == m_ue3FrameCounter;

      if (inActiveBurst || m_ue3FrameCounter >= m_ngxVelocityPairingLogNextAllowedFrame) {
        if (!inActiveBurst) {
          m_ngxVelocityPairingLogFrame = m_ue3FrameCounter;
          m_ngxVelocityPairingLogLines = 0;
          m_ngxVelocityPairingLogNextAllowedFrame = m_ue3FrameCounter + kPairingLogCooldownFrames;
        }

        if (m_ngxVelocityPairingLogLines < kPairingLogMaxLinesPerFrame) {
          m_ngxVelocityPairingLogLines++;

          const Vector3 translation = objectToWorld[3].xyz();
          std::string line = str::format(
            "[RTX NGX Passthrough][pairing miss] frame=", m_ue3FrameCounter,
            " identity=0x", std::hex, identity, std::dec,
            " instances=", objectState.instances.size(),
            " lastFrameCandidates=", lastFrameCandidates,
            " freshestSightingAge=", (freshestSightingAge == UINT32_MAX
                                        ? std::string("never") : std::to_string(freshestSightingAge)),
            " o2wPos=(", translation.x, ",", translation.y, ",", translation.z, ")",
            " camTranspose=", int(m_ngxFrameCameraUsedTranspose));

          if (nearestCandidate != nullptr) {
            const Vector3 nearestTranslation = nearestCandidate->objectToWorld[3].xyz();
            line += str::format(
              " nearest: dPos=", nearestTranslationDelta,
              " dRotScale=", nearestRotScaleDelta,
              " pos=(", nearestTranslation.x, ",", nearestTranslation.y, ",", nearestTranslation.z, ")");
          } else {
            line += " nearest: none seen last frame";
          }

          Logger::info(line);
        }
      }
    }

    // No usable history: a new placement of this mesh (or a sighting after a visibility
    // gap, where a one-frame velocity cannot be derived). Register it without velocity;
    // replace the stalest slot when the identity is heavily instanced.
    // Modular level geometry puts far more placements of one asset on screen than a small cap
    // can hold: hundreds of copies of a wall or railing section, all sharing this identity and
    // separated only by their transform.
    constexpr size_t kMaxInstancesPerIdentity = 256;

    NgxVelocityObjectInstance* targetInstance = nullptr;
    if (objectState.instances.size() >= kMaxInstancesPerIdentity) {
      for (NgxVelocityObjectInstance& instance : objectState.instances) {
        // Never recycle a slot the last two frames are still using. Over the cap, the stalest
        // slot is one this frame just claimed, so recycling it blindly makes an identity spend
        // the frame overwriting the history its remaining placements are about to pair against:
        // none of them can then match their own placement, and those that pair at all pair with
        // a neighbouring copy and emit the distance between two of them as motion. Leaving the
        // surplus untracked instead costs nothing, since static geometry is served exactly by
        // camera reprojection.
        if (instance.lastSeenFrame + 1 >= m_ue3FrameCounter) {
          continue;
        }
        if (targetInstance == nullptr || instance.lastSeenFrame < targetInstance->lastSeenFrame) {
          targetInstance = &instance;
        }
      }

      if (targetInstance == nullptr) {
        m_ngxVelocityStats.skippedInstanceCap++;
        return;
      }
    } else {
      targetInstance = &objectState.instances.emplace_back();
    }

    // Fresh occupancy: recycled slots must not inherit the previous occupant's movement
    // history (a stale confirmed-mover latch would instant-emit for the next pairing)
    targetInstance->objectToWorld = objectToWorld;
    targetInstance->worldToProjection = drawWorldToProjection;
    targetInstance->lastSeenFrame = m_ue3FrameCounter;
    targetInstance->wasMoving = false;
    targetInstance->lastMoveDelta = Vector3(0.0f, 0.0f, 0.0f);
    targetInstance->lastRotScaleDelta = 0.0f;
    targetInstance->lastEmitFrame = 0;
    targetInstance->lastEmitDrawIndex = 0;
    targetInstance->lastPhaseDuplicateFrame = 0;
    if (skinned) {
      targetInstance->bones.assign(boneRegisters, boneRegisters + boneRegisterCount);
    } else {
      targetInstance->bones.clear();
    }
    if (dynamicMesh) {
      targetInstance->dynamicPositions = std::move(currentPositions);
    } else {
      targetInstance->dynamicPositions.clear();
    }

    objectState.lastNewRegistrationFrame = m_ue3FrameCounter;
    m_ngxVelocityStats.newRegistrations++;
    if (skinned) {
      m_ngxVelocityStats.newRegistrationsSkinned++;
    }

    // Why the sighting had nothing to pair with. A one-frame velocity dropout is exactly this
    // outcome, so the population behind it has to be readable as a breakdown - the per-sighting
    // miss lines are rate limited to a burst every couple of seconds and cannot characterise it.
    if (lastFrameCandidates == 0) {
      m_ngxVelocityStats.missNoLastFrameSighting++;
    } else if (nearestTranslationDelta > kMaxFrameTranslation) {
      m_ngxVelocityStats.missBeyondTranslation++;
    } else {
      m_ngxVelocityStats.missBeyondRotation++;
    }
  }

  namespace {
    // UE3 declares FSystemSettingsData's sub-structs in a fixed order (Engine/Inc/SystemSettings.h),
    // so however the engine version sizes the earlier ones (FExposedTextureLODSettings grows with
    // TEXTUREGROUP_MAX), the tail is always:
    //   FLOAT ScreenPercentage; UBOOL bUpscaleScreenPercentage;
    //   INT ResX; INT ResY; UBOOL bFullscreen; INT MaxMultiSamples;
    // ResX/ResY/bFullscreen are known exactly from the D3D9 present parameters, which is what the
    // whole layout is discovered from.
    constexpr uint32_t kUe3UpscaleFromScreenPercentage = 0x04;
    constexpr uint32_t kUe3ResXFromScreenPercentage = 0x08;
    constexpr uint32_t kUe3ResYFromScreenPercentage = 0x0c;
    constexpr uint32_t kUe3FullscreenFromScreenPercentage = 0x10;
    constexpr uint32_t kUe3MaxMultisamplesFromScreenPercentage = 0x14;
    constexpr uint32_t kUe3SettingsDataTailSize = 0x18;

    // A record opens with FSystemSettingsDataWorldDetail: INT DetailMode then twenty UBOOLs.
    // Recognising that shape is how the offset of ScreenPercentage within the record - and so
    // the address of GSystemSettings itself - is recovered without knowing the engine version.
    constexpr int32_t kUe3DetailModeMax = 3;
    constexpr uint32_t kUe3HeadBooleanRun = 12;
    constexpr uint32_t kUe3MinSettingsDataSize = 0x40;
    constexpr uint32_t kUe3MaxSettingsDataSize = 0x2000;

    // Relaxed pass: how far back from the resolution anchor ScreenPercentage may sit when a
    // build does not use the stock tail adjacency.
    constexpr uint32_t kUe3MaxRelaxedScreenPercentageBacktrack = 0x40;

    // FSystemSettings embeds Defaults[FSL_LevelCount]; finding those repeats at a constant
    // stride both confirms the hit and recovers sizeof(FSystemSettingsData).
    constexpr uint32_t kUe3DefaultsSearchWindow = 0x8000;
    constexpr uint32_t kUe3MinDefaultsRepeats = 2;

    constexpr int32_t kUe3MaxMultisamplesPlausibleLimit = 64;
    constexpr float kUe3MaxPlausibleScreenPercentage = 400.0f;

    // A wrong candidate must never be trusted on shape alone, so the redirect set is proven by
    // driving this value and watching the game's own scene viewport respond. Above the 50%
    // floor the camera gate falls back to while ScreenPercentage is still unknown.
    constexpr float kNgxScreenPercentageProbeValue = 75.0f;
    constexpr uint32_t kNgxScreenPercentageProbeFrames = 240;
    constexpr int32_t kNgxScreenPercentageProbeTolerance = 16;

    // How long an engine-owned upscale may stay missing before the runtime takes over.
    constexpr uint32_t kNgxMissingEngineUpscaleFrameLimit = 30;

    // Bound retries for transient process/module read failures.
    constexpr uint32_t kNgxSettingsScanMaxAttempts = 30;
    constexpr uint64_t kNgxSettingsScanRetryIntervalMs = 1000;

    // Cap accepted operand sites so a pathological match set can never be patched wholesale.
    constexpr size_t kNgxMaxOperandSites = 64;
    constexpr size_t kNgxMaxRawOperandSites = 256;

    // Every renderer-facing use of ScreenPercentage in UE3 converts it to a scale factor:
    // ScaleScreenCoords compares it against 100.0f before dividing by it, UnScaleScreenCoords
    // multiplies by the reciprocal, NeedsUpscale compares against 100.0f. Reads that merely
    // copy the field elsewhere - settings mirrors, script exposure - have no such constant
    // beside them, and leaving those alone is what keeps the player's saved configuration out
    // of this. The surrounding cluster then picks up same-function reads the compiler shared.
    constexpr size_t kNgxPercentScaleConstantWindow = 24;
    constexpr uintptr_t kNgxSettingsCodeClusterRadius = 0x800;

    // Renderer-only values; GSystemSettings remains authoritative for game configuration.
    struct NgxGameSettingsShadow {
      float screenPercentage;
      int32_t upscaleScreenPercentage;
    };
    static_assert(sizeof(NgxGameSettingsShadow) == 8);

    constexpr uintptr_t kNgxShadowScreenPercentageOffset =
      offsetof(NgxGameSettingsShadow, screenPercentage);
    constexpr uintptr_t kNgxShadowUpscaleScreenPercentageOffset =
      offsetof(NgxGameSettingsShadow, upscaleScreenPercentage);

    struct NgxOperandSite {
      uintptr_t address = 0;
      uint32_t original = 0;
    };

    // Which field a site reads, and how it addresses it. The relative pair is what
    // FSystemSettings::NeedsUpscale() uses, so it decides whether the engine can still perform
    // its own upscale.
    enum class NgxOperandCategory : uint8_t {
      AbsoluteScreenPercentage,
      AbsoluteUpscale,
      RelativeScreenPercentage,
      RelativeUpscale,
    };

    struct NgxOperandRedirect {
      uintptr_t address = 0;
      uint32_t original = 0;
      uint32_t redirected = 0;
      DWORD originalProtection = 0;
      bool originalProtectionKnown = false;
      bool forceRestore = false;
      NgxOperandCategory category = NgxOperandCategory::AbsoluteScreenPercentage;
    };

    struct NgxLocatedGameSettings {
      uintptr_t screenPercentageAddress = 0;
      uintptr_t systemSettingsAddress = 0;
      float currentScreenPercentage = 100.0f;
      int32_t currentUpscaleScreenPercentage = 1;
      int32_t currentMaxMultisamples = 0;
      // Offset of ScreenPercentage within FSystemSettingsData; 0 when the relative readers
      // could not be resolved, in which case only the absolute readers are redirected.
      uint32_t screenPercentageStructOffset = 0;
      uint32_t settingsDataStride = 0;
      uint32_t defaultsRepeats = 0;
      bool strictTailLayout = false;
      bool iniConfirmed = false;
      // Which stage rejected the module, so a failure is readable straight from the log.
      const char* missReason = "no reason recorded";
      std::vector<NgxOperandSite> directScreenPercentageSites;
      std::vector<NgxOperandSite> directUpscaleScreenPercentageSites;
      std::vector<NgxOperandSite> relativeScreenPercentageSites;
      std::vector<NgxOperandSite> relativeUpscaleScreenPercentageSites;
    };

    template <typename T>
    bool ngxReadProcessExact(HANDLE process, uintptr_t address, T& value) {
      SIZE_T bytesRead = 0;
      return ::ReadProcessMemory(process, reinterpret_cast<LPCVOID>(address),
                                 &value, sizeof(value), &bytesRead) &&
             bytesRead == sizeof(value);
    }

    template <typename T>
    bool ngxWriteProcessExact(HANDLE process, uintptr_t address, const T& value) {
      SIZE_T bytesWritten = 0;
      return ::WriteProcessMemory(process, reinterpret_cast<LPVOID>(address),
                                  &value, sizeof(value), &bytesWritten) &&
             bytesWritten == sizeof(value);
    }

    template <typename T>
    bool ngxWriteProcessExactRetry(HANDLE process, uintptr_t address,
                                   const T& value) {
      for (uint32_t attempt = 0; attempt < 3; attempt++) {
        if (ngxWriteProcessExact(process, address, value))
          return true;
      }
      return false;
    }

    bool ngxUpdateScreenPercentageShadow(HANDLE process, uintptr_t shadowAddress,
                                         float screenPercentage, int32_t upscale,
                                         bool* outRollbackVerified = nullptr) {
      if (outRollbackVerified != nullptr)
        *outRollbackVerified = true;
      float oldScreenPercentage = 0.0f;
      int32_t oldUpscale = 0;
      if (!ngxReadProcessExact(
            process, shadowAddress + kNgxShadowScreenPercentageOffset,
            oldScreenPercentage) ||
          !ngxReadProcessExact(
            process, shadowAddress + kNgxShadowUpscaleScreenPercentageOffset,
            oldUpscale))
        return false;

      const auto writeScreenPercentage = [&](float value) {
        return ngxWriteProcessExactRetry(
          process, shadowAddress + kNgxShadowScreenPercentageOffset, value);
      };
      const auto writeUpscale = [&](int32_t value) {
        return ngxWriteProcessExactRetry(
          process, shadowAddress + kNgxShadowUpscaleScreenPercentageOffset,
          value);
      };
      const auto pairEquals = [&](float expectedScreenPercentage,
                                  int32_t expectedUpscale) {
        float verifiedScreenPercentage = 0.0f;
        int32_t verifiedUpscale = 0;
        return
          ngxReadProcessExact(
            process, shadowAddress + kNgxShadowScreenPercentageOffset,
            verifiedScreenPercentage) &&
          ngxReadProcessExact(
            process, shadowAddress + kNgxShadowUpscaleScreenPercentageOffset,
            verifiedUpscale) &&
          std::memcmp(&verifiedScreenPercentage, &expectedScreenPercentage,
                      sizeof(expectedScreenPercentage)) == 0 &&
          verifiedUpscale == expectedUpscale;
      };
      const auto writePairSafely = [&](float targetScreenPercentage,
                                       int32_t targetUpscale) {
        bool firstRestored = false;
        bool secondRestored = false;
        if (targetUpscale != 0) {
          firstRestored = writeUpscale(targetUpscale);
          if (firstRestored)
            secondRestored = writeScreenPercentage(targetScreenPercentage);
        } else {
          firstRestored = writeScreenPercentage(targetScreenPercentage);
          if (firstRestored)
            secondRestored = writeUpscale(targetUpscale);
        }
        return firstRestored && secondRestored;
      };
      const auto rollbackOldPair = [&]() {
        return writePairSafely(oldScreenPercentage, oldUpscale) &&
               pairEquals(oldScreenPercentage, oldUpscale);
      };

      if (!writePairSafely(screenPercentage, upscale)) {
        const bool rollbackVerified = rollbackOldPair();
        if (outRollbackVerified != nullptr)
          *outRollbackVerified = rollbackVerified;
        return false;
      }

      const bool verified = pairEquals(screenPercentage, upscale);
      if (!verified) {
        const bool rollbackVerified = rollbackOldPair();
        if (outRollbackVerified != nullptr)
          *outRollbackVerified = rollbackVerified;
      }
      return verified;
    }

    bool ngxRestoreCodeProtection(HANDLE process, uintptr_t address,
                                  DWORD protection) {
      for (uint32_t attempt = 0; attempt < 3; attempt++) {
        DWORD ignoredProtection = 0;
        if (::VirtualProtectEx(
              process, reinterpret_cast<LPVOID>(address), sizeof(uint32_t),
              protection, &ignoredProtection))
          return true;
      }
      return false;
    }

    bool ngxCodeProtectionMatches(HANDLE process, uintptr_t address,
                                  DWORD protection) {
      MEMORY_BASIC_INFORMATION memoryInfo = {};
      return ::VirtualQueryEx(
               process, reinterpret_cast<LPCVOID>(address),
               &memoryInfo, sizeof(memoryInfo)) == sizeof(memoryInfo) &&
             memoryInfo.Protect == protection;
    }

    // Restore the exact pre-write Protect (WRITECOPY pages cannot always regain it).
    bool ngxWriteCodeOperand(HANDLE process, NgxOperandRedirect& redirect) {
      if (!redirect.originalProtectionKnown) {
        MEMORY_BASIC_INFORMATION memoryInfo = {};
        if (::VirtualQueryEx(
              process, reinterpret_cast<LPCVOID>(redirect.address),
              &memoryInfo, sizeof(memoryInfo)) != sizeof(memoryInfo))
          return false;
        redirect.originalProtection = memoryInfo.Protect;
        redirect.originalProtectionKnown = true;
      }

      DWORD oldProtection = 0;
      if (!::VirtualProtectEx(
            process, reinterpret_cast<LPVOID>(redirect.address),
            sizeof(redirect.redirected), PAGE_EXECUTE_READWRITE,
            &oldProtection))
        return false;

      if (oldProtection != redirect.originalProtection) {
        redirect.originalProtection = oldProtection;
        ngxRestoreCodeProtection(process, redirect.address, oldProtection);
        return false;
      }

      const bool wrote =
        ngxWriteProcessExact(process, redirect.address, redirect.redirected);
      const bool flushed = wrote &&
        ::FlushInstructionCache(
          process, reinterpret_cast<LPCVOID>(redirect.address),
          sizeof(redirect.redirected));
      const bool restoredProtection =
        ngxRestoreCodeProtection(
          process, redirect.address, redirect.originalProtection) &&
        ngxCodeProtectionMatches(
          process, redirect.address, redirect.originalProtection);
      return wrote && flushed && restoredProtection;
    }

    bool ngxRestoreCodeOperand(HANDLE process,
                               const NgxOperandRedirect& redirect) {
      if (!redirect.originalProtectionKnown)
        return false;

      DWORD ignoredProtection = 0;
      if (!::VirtualProtectEx(
            process, reinterpret_cast<LPVOID>(redirect.address),
            sizeof(uint32_t), PAGE_EXECUTE_READWRITE, &ignoredProtection))
        return false;

      const bool wrote =
        ngxWriteProcessExactRetry(
          process, redirect.address, redirect.original);
      const bool flushed = wrote &&
        ::FlushInstructionCache(
          process, reinterpret_cast<LPCVOID>(redirect.address),
          sizeof(uint32_t));
      const bool protectionRestored =
        ngxRestoreCodeProtection(
          process, redirect.address, redirect.originalProtection) &&
        ngxCodeProtectionMatches(
          process, redirect.address, redirect.originalProtection);
      uint32_t verified = 0;
      return wrote && flushed && protectionRestored &&
             ngxReadProcessExact(process, redirect.address, verified) &&
             verified == redirect.original;
    }

    // Quiesce target threads while unaligned x86 operands are changed.
    class NgxScopedThreadSuspension {
    public:
      explicit NgxScopedThreadSuspension(DWORD processId) {
        const DWORD currentProcessId = ::GetCurrentProcessId();
        const DWORD currentThreadId = ::GetCurrentThreadId();

        // Repeat snapshots to close the thread-creation race. Fixed storage avoids allocating
        // from a heap that a suspended single-process thread may own.
        uint32_t stablePasses = 0;
        for (uint32_t pass = 0; pass < 8; pass++) {
          HANDLE snapshot =
            ::CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
          if (snapshot == INVALID_HANDLE_VALUE)
            return;

          THREADENTRY32 entry = {};
          entry.dwSize = sizeof(entry);
          if (!::Thread32First(snapshot, &entry)) {
            ::CloseHandle(snapshot);
            return;
          }

          bool discoveredThread = false;
          bool transientThreadExit = false;
          DWORD enumerationError = ERROR_SUCCESS;
          for (;;) {
            if (entry.th32OwnerProcessID == processId &&
                !(processId == currentProcessId &&
                  entry.th32ThreadID == currentThreadId) &&
                !isTracked(entry.th32ThreadID)) {
              discoveredThread = true;
              if (m_threadCount >= m_threads.size()) {
                ::CloseHandle(snapshot);
                return;
              }

              HANDLE thread =
                ::OpenThread(THREAD_SUSPEND_RESUME, FALSE,
                             entry.th32ThreadID);
              if (thread == nullptr) {
                if (::GetLastError() == ERROR_INVALID_PARAMETER) {
                  transientThreadExit = true;
                } else {
                  ::CloseHandle(snapshot);
                  return;
                }
              } else if (::SuspendThread(thread) == DWORD(-1)) {
                transientThreadExit = true;
                ::CloseHandle(thread);
              } else {
                m_threadIds[m_threadCount] = entry.th32ThreadID;
                m_threads[m_threadCount] = thread;
                m_threadCount++;
              }
            }

            ::SetLastError(ERROR_SUCCESS);
            if (!::Thread32Next(snapshot, &entry)) {
              enumerationError = ::GetLastError();
              break;
            }
          }

          ::CloseHandle(snapshot);
          if (enumerationError != ERROR_NO_MORE_FILES)
            return;

          if (!discoveredThread && !transientThreadExit) {
            if (++stablePasses >= 2) {
              m_complete = true;
              return;
            }
          } else {
            stablePasses = 0;
          }
        }
      }

      ~NgxScopedThreadSuspension() {
        for (size_t i = m_threadCount; i > 0; i--)
          ::ResumeThread(m_threads[i - 1]);
        for (size_t i = 0; i < m_threadCount; i++)
          ::CloseHandle(m_threads[i]);
      }

      bool complete() const {
        return m_complete;
      }

    private:
      bool isTracked(DWORD threadId) const {
        for (size_t i = 0; i < m_threadCount; i++) {
          if (m_threadIds[i] == threadId)
            return true;
        }
        return false;
      }

      bool m_complete = false;
      size_t m_threadCount = 0;
      std::array<HANDLE, 256> m_threads = {};
      std::array<DWORD, 256> m_threadIds = {};
    };

    struct NgxOperandApplyResult {
      size_t applied = 0;
      size_t skipped = 0;
      size_t unrestorable = 0;
      size_t appliedByCategory[4] = {};
    };

    // Per-site tolerance: a site that cannot be written, or that cannot regain its original page
    // protection afterwards (WRITECOPY pages do not always come back), is rolled back and left
    // stock instead of failing the whole set. Whether the sites that did land are sufficient is
    // settled by the behavioural probe rather than guessed at here.
    NgxOperandApplyResult ngxApplyOperandRedirects(
        HANDLE process,
        std::vector<NgxOperandRedirect>& redirects,
        std::vector<NgxOperandRedirect>& outActive) {
      NgxOperandApplyResult result;
      outActive.clear();
      // Caller must reserve before threads are suspended (no heap growth while suspended).
      if (outActive.capacity() < redirects.size()) {
        result.skipped = redirects.size();
        return result;
      }

      for (NgxOperandRedirect& redirect : redirects) {
        uint32_t current = 0;
        MEMORY_BASIC_INFORMATION memoryInfo = {};
        if (!ngxReadProcessExact(process, redirect.address, current) ||
            current != redirect.original ||
            ::VirtualQueryEx(
              process, reinterpret_cast<LPCVOID>(redirect.address),
              &memoryInfo, sizeof(memoryInfo)) != sizeof(memoryInfo)) {
          result.skipped++;
          continue;
        }
        redirect.originalProtection = memoryInfo.Protect;
        redirect.originalProtectionKnown = true;

        uint32_t verified = 0;
        if (ngxWriteCodeOperand(process, redirect) &&
            ngxReadProcessExact(process, redirect.address, verified) &&
            verified == redirect.redirected) {
          result.applied++;
          result.appliedByCategory[size_t(redirect.category)]++;
          outActive.push_back(redirect);
          continue;
        }

        // A failed write may still have landed before protection/cache restoration failed.
        if (!ngxRestoreCodeOperand(process, redirect)) {
          redirect.forceRestore = true;
          result.unrestorable++;
          outActive.push_back(redirect);
        } else {
          result.skipped++;
        }
      }

      return result;
    }

    // Bridge host: parent PID is the game process.
    DWORD ngxGetParentPid() {
      const DWORD selfPid = ::GetCurrentProcessId();
      HANDLE snapshot = ::CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
      if (snapshot == INVALID_HANDLE_VALUE)
        return 0;

      PROCESSENTRY32W entry = {};
      entry.dwSize = sizeof(entry);
      DWORD parentPid = 0;
      if (::Process32FirstW(snapshot, &entry)) {
        do {
          if (entry.th32ProcessID == selfPid) {
            parentPid = entry.th32ParentProcessID;
            break;
          }
        } while (::Process32NextW(snapshot, &entry));
      }

      ::CloseHandle(snapshot);
      return parentPid;
    }

    // SNAPMODULE32 so the 64-bit bridge host can see the 32-bit game module.
    bool ngxGetMainModule(DWORD pid, uintptr_t& outBase, uint32_t& outSize,
                          std::wstring& outPath) {
      HANDLE snapshot = ::CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid);
      if (snapshot == INVALID_HANDLE_VALUE)
        return false;

      MODULEENTRY32W module = {};
      module.dwSize = sizeof(module);
      bool ok = false;
      if (::Module32FirstW(snapshot, &module)) {
        outBase = reinterpret_cast<uintptr_t>(module.modBaseAddr);
        outSize = module.modBaseSize;
        outPath = module.szExePath;
        ok = true;
      }

      ::CloseHandle(snapshot);
      return ok;
    }

    enum class NgxLocateResult {
      Found,
      Miss,
      Incomplete,
    };

    // ---------------------------------------------------------------------------------------
    // Optional independent validator: the game's own [SystemSettings] ini section.
    // UE3's property-name table is fixed across engine versions, so agreement between the ini
    // and a candidate record is a known-plaintext confirmation of both base and layout. It only
    // ever raises confidence - a user ini that lags the live values must not reject a candidate.
    // ---------------------------------------------------------------------------------------
    struct NgxIniSystemSettings {
      bool found = false;
      bool hasScreenPercentage = false;
      bool hasUpscale = false;
      bool hasResolution = false;
      bool hasFullscreen = false;
      bool hasMaxMultisamples = false;
      float screenPercentage = 0.0f;
      int32_t upscale = 0;
      int32_t resX = 0;
      int32_t resY = 0;
      int32_t fullscreen = 0;
      int32_t maxMultisamples = 0;
    };

    std::string ngxLowerAscii(const std::string& value) {
      std::string out = value;
      for (char& c : out)
        c = char(std::tolower(uint8_t(c)));
      return out;
    }

    std::string ngxTrimAscii(const std::string& value) {
      size_t first = value.find_first_not_of(" \t\r\n");
      if (first == std::string::npos)
        return std::string();
      size_t last = value.find_last_not_of(" \t\r\n");
      return value.substr(first, last - first + 1);
    }

    bool ngxParseIniBool(const std::string& value, int32_t& out) {
      const std::string lowered = ngxLowerAscii(value);
      if (lowered == "true" || lowered == "1") {
        out = 1;
        return true;
      }
      if (lowered == "false" || lowered == "0") {
        out = 0;
        return true;
      }
      return false;
    }

    bool ngxReadFileText(const std::wstring& path, std::string& out) {
      HANDLE file = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                  nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
      if (file == INVALID_HANDLE_VALUE)
        return false;

      LARGE_INTEGER size = {};
      constexpr LONGLONG kMaxIniBytes = 1 << 20;
      if (!::GetFileSizeEx(file, &size) || size.QuadPart <= 0 || size.QuadPart > kMaxIniBytes) {
        ::CloseHandle(file);
        return false;
      }

      std::vector<uint8_t> raw(size_t(size.QuadPart));
      DWORD bytesRead = 0;
      const bool ok = ::ReadFile(file, raw.data(), DWORD(raw.size()), &bytesRead, nullptr) &&
                      bytesRead == raw.size();
      ::CloseHandle(file);
      if (!ok)
        return false;

      // UE3 writes user configs as UTF-16LE; the shipped defaults are plain ASCII.
      if (raw.size() >= 2 && raw[0] == 0xFF && raw[1] == 0xFE) {
        out.clear();
        out.reserve(raw.size() / 2);
        for (size_t i = 2; i + 1 < raw.size(); i += 2) {
          const uint16_t unit = uint16_t(raw[i]) | uint16_t(uint16_t(raw[i + 1]) << 8);
          out.push_back(unit < 0x80 ? char(unit) : '?');
        }
      } else {
        out.assign(reinterpret_cast<const char*>(raw.data()), raw.size());
      }

      return true;
    }

    bool ngxParseIniSystemSettings(const std::string& text, NgxIniSystemSettings& out) {
      std::istringstream stream(text);
      std::string line;
      bool inSection = false;
      NgxIniSystemSettings parsed;

      while (std::getline(stream, line)) {
        const std::string trimmed = ngxTrimAscii(line);
        if (trimmed.empty() || trimmed[0] == ';')
          continue;

        if (trimmed[0] == '[') {
          if (inSection)
            break;
          inSection = ngxLowerAscii(trimmed) == "[systemsettings]";
          continue;
        }

        if (!inSection)
          continue;

        const size_t separator = trimmed.find('=');
        if (separator == std::string::npos)
          continue;

        const std::string key = ngxLowerAscii(ngxTrimAscii(trimmed.substr(0, separator)));
        const std::string value = ngxTrimAscii(trimmed.substr(separator + 1));
        if (value.empty())
          continue;

        try {
          if (key == "screenpercentage") {
            parsed.screenPercentage = std::stof(value);
            parsed.hasScreenPercentage = std::isfinite(parsed.screenPercentage);
          } else if (key == "upscalescreenpercentage") {
            parsed.hasUpscale = ngxParseIniBool(value, parsed.upscale);
          } else if (key == "fullscreen") {
            parsed.hasFullscreen = ngxParseIniBool(value, parsed.fullscreen);
          } else if (key == "resx") {
            parsed.resX = std::stoi(value);
          } else if (key == "resy") {
            parsed.resY = std::stoi(value);
          } else if (key == "maxmultisamples") {
            parsed.maxMultisamples = std::stoi(value);
            parsed.hasMaxMultisamples = true;
          }
        } catch (const std::exception&) {
          // A malformed entry only costs us that one cross-check.
        }
      }

      parsed.hasResolution = parsed.resX > 0 && parsed.resY > 0;
      parsed.found = parsed.hasScreenPercentage || parsed.hasResolution ||
                     parsed.hasMaxMultisamples;
      if (!parsed.found)
        return false;

      out = parsed;
      return true;
    }

    void ngxCollectEngineInis(const std::wstring& directory,
                              std::vector<std::wstring>& out) {
      constexpr size_t kMaxIniCandidates = 16;
      if (out.size() >= kMaxIniCandidates)
        return;

      WIN32_FIND_DATAW find = {};
      HANDLE search = ::FindFirstFileW((directory + L"\\Config\\*Engine.ini").c_str(), &find);
      if (search == INVALID_HANDLE_VALUE)
        return;

      do {
        if (find.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
          continue;
        out.push_back(directory + L"\\Config\\" + find.cFileName);
      } while (out.size() < kMaxIniCandidates && ::FindNextFileW(search, &find));

      ::FindClose(search);
    }

    // UE3 keeps its configs at <root>\<X>Game\Config\<Y>Engine.ini, with the live user copy
    // often under Documents. Only immediate subdirectories are enumerated, so this stays cheap.
    void ngxCollectEngineIniRoots(const std::wstring& root, uint32_t remainingDepth,
                                  std::vector<std::wstring>& out) {
      constexpr size_t kMaxIniCandidates = 16;
      ngxCollectEngineInis(root, out);
      if (remainingDepth == 0 || out.size() >= kMaxIniCandidates)
        return;

      WIN32_FIND_DATAW find = {};
      HANDLE search = ::FindFirstFileW((root + L"\\*").c_str(), &find);
      if (search == INVALID_HANDLE_VALUE)
        return;

      constexpr uint32_t kMaxSubdirectories = 64;
      uint32_t visited = 0;
      do {
        if (!(find.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
          continue;
        const std::wstring name = find.cFileName;
        if (name == L"." || name == L"..")
          continue;
        ngxCollectEngineIniRoots(root + L"\\" + name, remainingDepth - 1, out);
      } while (++visited < kMaxSubdirectories && out.size() < kMaxIniCandidates &&
               ::FindNextFileW(search, &find));

      ::FindClose(search);
    }

    NgxIniSystemSettings ngxReadIniSystemSettings(const std::wstring& exePath) {
      NgxIniSystemSettings result;
      if (exePath.empty())
        return result;

      std::vector<std::wstring> candidates;

      // The install tree: the exe lives under <root>\Binaries[\Win32], so walk up a few levels.
      std::wstring directory = exePath;
      for (uint32_t level = 0; level < 4; level++) {
        const size_t separator = directory.find_last_of(L"\\/");
        if (separator == std::wstring::npos || separator == 0)
          break;
        directory = directory.substr(0, separator);
        ngxCollectEngineIniRoots(directory, 1, candidates);
      }

      wchar_t userProfile[MAX_PATH] = {};
      if (::GetEnvironmentVariableW(L"USERPROFILE", userProfile, MAX_PATH) != 0) {
        ngxCollectEngineIniRoots(std::wstring(userProfile) + L"\\Documents\\My Games", 2,
                                 candidates);
      }

      for (const std::wstring& candidate : candidates) {
        std::string text;
        NgxIniSystemSettings parsed;
        if (ngxReadFileText(candidate, text) && ngxParseIniSystemSettings(text, parsed)) {
          // Prefer the copy that carries a resolution: that is the one the game writes back to.
          if (!result.found || (parsed.hasResolution && !result.hasResolution))
            result = parsed;
        }
      }

      return result;
    }

    // ---------------------------------------------------------------------------------------
    // Data-side recognition of an FSystemSettingsData record
    // ---------------------------------------------------------------------------------------

    bool ngxIsPlausibleScreenPercentage(float value) {
      return std::isfinite(value) && value > 0.0f &&
             value <= kUe3MaxPlausibleScreenPercentage;
    }

    // UE3 reads MaxMultisamples straight out of the ini as a plain INT and never constrains it,
    // so real installs carry values the D3D9 sample counts do not (Mirror's Edge ships 10).
    bool ngxIsPlausibleMaxMultisamples(int32_t value) {
      return value >= 0 && value <= kUe3MaxMultisamplesPlausibleLimit;
    }

    bool ngxIsPlausibleResolution(int32_t x, int32_t y) {
      return (x == 0 && y == 0) ||
             (x >= 320 && x <= 16384 && y >= 240 && y <= 16384);
    }

    int32_t ngxReadInt32(const uint8_t* data) {
      int32_t value = 0;
      std::memcpy(&value, data, sizeof(value));
      return value;
    }

    float ngxReadFloat(const uint8_t* data) {
      float value = 0.0f;
      std::memcpy(&value, data, sizeof(value));
      return value;
    }

    // FSystemSettingsDataWorldDetail opens the record: INT DetailMode then a long run of UBOOLs.
    bool ngxLooksLikeSettingsRecordHead(const uint8_t* data, size_t available) {
      if (available < sizeof(int32_t) * (1 + kUe3HeadBooleanRun))
        return false;

      const int32_t detailMode = ngxReadInt32(data);
      if (detailMode < 0 || detailMode > kUe3DetailModeMax)
        return false;

      for (uint32_t i = 1; i <= kUe3HeadBooleanRun; i++) {
        const int32_t value = ngxReadInt32(data + i * sizeof(int32_t));
        if (value != 0 && value != 1)
          return false;
      }

      return true;
    }

    // The record tail, with the resolution left loose so the Defaults[] copies (populated from
    // the compat ini rather than the live mode) match as well as the live record does.
    bool ngxLooksLikeSettingsRecordTail(const uint8_t* tail, size_t available) {
      if (available < kUe3SettingsDataTailSize)
        return false;

      const float screenPercentage = ngxReadFloat(tail);
      const int32_t upscale = ngxReadInt32(tail + kUe3UpscaleFromScreenPercentage);
      const int32_t resX = ngxReadInt32(tail + kUe3ResXFromScreenPercentage);
      const int32_t resY = ngxReadInt32(tail + kUe3ResYFromScreenPercentage);
      const int32_t fullscreen = ngxReadInt32(tail + kUe3FullscreenFromScreenPercentage);
      const int32_t maxMultisamples = ngxReadInt32(tail + kUe3MaxMultisamplesFromScreenPercentage);

      return ngxIsPlausibleScreenPercentage(screenPercentage) &&
             (upscale == 0 || upscale == 1) &&
             (fullscreen == 0 || fullscreen == 1) &&
             ngxIsPlausibleMaxMultisamples(maxMultisamples) &&
             ngxIsPlausibleResolution(resX, resY);
    }

    // ---------------------------------------------------------------------------------------
    // x86 operand classification
    //
    // Sites are accepted on instruction form rather than on surrounding byte context: only the
    // encodings that *read* one of these fields are patched, so the engine's own writes (and the
    // config-facing consumers, which all reach the fields through `this`) keep the real values.
    // ---------------------------------------------------------------------------------------

    bool ngxIsAbsoluteModrm(uint8_t modrm) {
      return (modrm & 0xC7) == 0x05;
    }

    bool ngxIsBaseDisp32Modrm(uint8_t modrm) {
      return (modrm & 0xC0) == 0x80 && (modrm & 0x07) != 0x04;
    }

    uint8_t ngxModrmReg(uint8_t modrm) {
      return uint8_t((modrm >> 3) & 0x07);
    }

    uint8_t ngxModrmRm(uint8_t modrm) {
      return uint8_t(modrm & 0x07);
    }

    bool ngxIsAbsoluteFloatReadSite(const uint8_t* bytes, size_t operandOffset) {
      if (operandOffset >= 4) {
        const uint8_t modrm = bytes[operandOffset - 1];
        // movss/addss/mulss/subss/divss xmm, m32. 0x11 (movss m32, xmm) is a store.
        if (bytes[operandOffset - 4] == 0xF3 && bytes[operandOffset - 3] == 0x0F &&
            ngxIsAbsoluteModrm(modrm)) {
          const uint8_t op = bytes[operandOffset - 2];
          if (op == 0x10 || op == 0x58 || op == 0x59 || op == 0x5C || op == 0x5E)
            return true;
        }
      }

      if (operandOffset >= 3) {
        const uint8_t modrm = bytes[operandOffset - 1];
        const uint8_t op = bytes[operandOffset - 2];
        // ucomiss/comiss xmm, m32; a 0x66 prefix would make these the f64 variants.
        if (bytes[operandOffset - 3] == 0x0F && (op == 0x2E || op == 0x2F) &&
            ngxIsAbsoluteModrm(modrm) &&
            (operandOffset < 4 || bytes[operandOffset - 4] != 0x66))
          return true;
      }

      if (operandOffset >= 2) {
        const uint8_t modrm = bytes[operandOffset - 1];
        const uint8_t op = bytes[operandOffset - 2];
        if (ngxIsAbsoluteModrm(modrm)) {
          // x87 arithmetic/compare against m32; D9 /0 is fld, D9 /3 (fstp) is a store.
          if (op == 0xD8)
            return true;
          if (op == 0xD9 && ngxModrmReg(modrm) == 0)
            return true;
        }
      }

      return false;
    }

    bool ngxIsAbsoluteIntReadSite(const uint8_t* bytes, size_t operandOffset) {
      if (operandOffset >= 1 && bytes[operandOffset - 1] == 0xA1)
        return true;  // mov eax, [disp32]

      if (operandOffset >= 2) {
        const uint8_t modrm = bytes[operandOffset - 1];
        const uint8_t op = bytes[operandOffset - 2];
        if (!ngxIsAbsoluteModrm(modrm))
          return false;

        // mov r32, m32 / cmp / test / xor / or, all reading the memory operand.
        if (op == 0x8B || op == 0x3B || op == 0x39 || op == 0x85 || op == 0x33 || op == 0x0B)
          return true;
        // Group 1 immediate forms: only /7 (cmp) leaves the memory operand untouched.
        if ((op == 0x83 || op == 0x81) && ngxModrmReg(modrm) == 7)
          return true;
        // test m32, imm32.
        if (op == 0xF7 && ngxModrmReg(modrm) == 0)
          return true;
      }

      return false;
    }

    // Distinguishes a renderer-facing read from a read that just copies the field somewhere.
    bool ngxReadsPercentScaleConstantNearby(HANDLE proc, const uint8_t* bytes,
                                            size_t operandOffset, size_t length) {
      if (length < sizeof(uint32_t))
        return false;

      const size_t begin = operandOffset > kNgxPercentScaleConstantWindow
        ? operandOffset - kNgxPercentScaleConstantWindow
        : 0;
      const size_t end = std::min(length - sizeof(uint32_t),
                                  operandOffset + kNgxPercentScaleConstantWindow);

      for (size_t offset = begin; offset < end; offset++) {
        if (offset == operandOffset || !ngxIsAbsoluteFloatReadSite(bytes, offset))
          continue;

        uint32_t constantAddress = 0;
        std::memcpy(&constantAddress, bytes + offset, sizeof(constantAddress));

        float constantValue = 0.0f;
        if (ngxReadProcessExact(proc, uintptr_t(constantAddress), constantValue) &&
            (constantValue == 100.0f || constantValue == 0.01f))
          return true;
      }

      return false;
    }

    struct NgxRelativeOperand {
      size_t operandOffset = 0;
      uint32_t displacement = 0;
      uint8_t baseRegister = 0;
      bool isFloat = false;
      bool valid = false;
    };

    // Decodes only the [reg+disp32] read forms UE3 emits for these two fields; anything else
    // simply does not match, which is what keeps the paired-window search precise.
    NgxRelativeOperand ngxDecodeRelativeRead(const uint8_t* bytes, size_t offset,
                                             size_t available) {
      NgxRelativeOperand out;

      const auto emit = [&](size_t operandOffset, bool isFloat, uint8_t modrm) {
        if (operandOffset + sizeof(uint32_t) > available)
          return;
        std::memcpy(&out.displacement, bytes + operandOffset, sizeof(out.displacement));
        out.operandOffset = operandOffset;
        out.baseRegister = ngxModrmRm(modrm);
        out.isFloat = isFloat;
        out.valid = true;
      };

      if (offset + 3 <= available && bytes[offset] == 0x0F &&
          (bytes[offset + 1] == 0x2E || bytes[offset + 1] == 0x2F) &&
          ngxIsBaseDisp32Modrm(bytes[offset + 2]) &&
          (offset == 0 || bytes[offset - 1] != 0x66)) {
        emit(offset + 3, true, bytes[offset + 2]);
      } else if (offset + 4 <= available && bytes[offset] == 0xF3 &&
                 bytes[offset + 1] == 0x0F && bytes[offset + 2] == 0x10 &&
                 ngxIsBaseDisp32Modrm(bytes[offset + 3])) {
        emit(offset + 4, true, bytes[offset + 3]);
      } else if (offset + 2 <= available &&
                 (bytes[offset] == 0x8B || bytes[offset] == 0x3B ||
                  bytes[offset] == 0x39 || bytes[offset] == 0x85) &&
                 ngxIsBaseDisp32Modrm(bytes[offset + 1])) {
        emit(offset + 2, false, bytes[offset + 1]);
      } else if (offset + 2 <= available && bytes[offset] == 0x83 &&
                 ngxIsBaseDisp32Modrm(bytes[offset + 1]) &&
                 ngxModrmReg(bytes[offset + 1]) == 7) {
        emit(offset + 2, false, bytes[offset + 1]);
      }

      return out;
    }

    struct NgxGameSettingsScanInputs {
      uint32_t backBufferWidth = 0;
      uint32_t backBufferHeight = 0;
      bool windowed = false;
      // Escape hatch: when set, discovery is skipped in favour of this module-relative address
      // (still subject to the shape checks and the behavioural probe).
      uint32_t forcedScreenPercentageRva = 0;
      NgxIniSystemSettings ini;
    };

    // The filesystem search is worth doing once per process, not once per scan retry.
    const NgxIniSystemSettings& ngxCachedIniSystemSettings(const std::wstring& exePath) {
      static const NgxIniSystemSettings cached = ngxReadIniSystemSettings(exePath);
      return cached;
    }

    // Finds and validates the renderer-facing settings operands in a 32-bit game module.
    // Candidates the behavioural probe has already disproven are skipped, so a failed probe can
    // simply rescan for the next best match.
    NgxLocateResult ngxLocateGameSettings(HANDLE proc, uintptr_t base, uint32_t moduleSize,
                                          const NgxGameSettingsScanInputs& inputs,
                                          const std::vector<uintptr_t>& rejectedCandidates,
                                          NgxLocatedGameSettings& outSettings) {
      outSettings = {};

      if (inputs.backBufferWidth == 0 || inputs.backBufferHeight == 0)
        return NgxLocateResult::Incomplete;

      uint8_t headers[0x1000];
      SIZE_T bytesRead = 0;
      if (!::ReadProcessMemory(proc, reinterpret_cast<LPCVOID>(base), headers, sizeof(headers), &bytesRead) ||
          bytesRead < sizeof(IMAGE_DOS_HEADER))
        return NgxLocateResult::Incomplete;

      outSettings.missReason = "not a 32-bit PE module";

      const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(headers);
      if (dos->e_magic != IMAGE_DOS_SIGNATURE)
        return NgxLocateResult::Miss;

      const uint32_t ntOffset = uint32_t(dos->e_lfanew);
      if (ntOffset + sizeof(IMAGE_NT_HEADERS32) > sizeof(headers))
        return NgxLocateResult::Miss;

      const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS32*>(headers + ntOffset);
      if (nt->Signature != IMAGE_NT_SIGNATURE ||
          nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR32_MAGIC)
        return NgxLocateResult::Miss;

      outSettings.missReason = "malformed PE section table";

      // IMAGE_NT_HEADERS32 = DWORD Signature; IMAGE_FILE_HEADER; IMAGE_OPTIONAL_HEADER32.
      // The section table follows the optional header (whose size is declared, so this is
      // robust to layout differences).
      const uint32_t sectionTableOffset =
        ntOffset + uint32_t(sizeof(uint32_t) + sizeof(IMAGE_FILE_HEADER)) + nt->FileHeader.SizeOfOptionalHeader;
      const uint32_t sectionCount = nt->FileHeader.NumberOfSections;
      if (sectionCount == 0 || sectionCount > 96)
        return NgxLocateResult::Miss;

      struct SectionRange {
        uintptr_t address = 0;
        uint32_t size = 0;
        bool executable = false;
      };
      std::vector<SectionRange> sectionRanges;

      for (uint32_t i = 0; i < sectionCount; i++) {
        const uint32_t entryOffset = sectionTableOffset + i * uint32_t(sizeof(IMAGE_SECTION_HEADER));
        if (entryOffset + sizeof(IMAGE_SECTION_HEADER) > sizeof(headers))
          return NgxLocateResult::Incomplete;

        const auto* section = reinterpret_cast<const IMAGE_SECTION_HEADER*>(headers + entryOffset);
        const DWORD characteristics = section->Characteristics;
        const bool executable = (characteristics & IMAGE_SCN_MEM_EXECUTE) != 0;
        // GSystemSettings has a constructor, so it lands in an initialised or zero-filled
        // writable data section; both are committed and readable at this point.
        const bool writableData = !executable &&
                                  (characteristics & IMAGE_SCN_MEM_WRITE) != 0 &&
                                  (characteristics & IMAGE_SCN_MEM_READ) != 0;
        if (!executable && !writableData)
          continue;

        uint32_t size = section->Misc.VirtualSize != 0 ? section->Misc.VirtualSize : section->SizeOfRawData;
        if (moduleSize != 0) {
          if (section->VirtualAddress >= moduleSize)
            return NgxLocateResult::Incomplete;
          size = std::min(size,
                          moduleSize - uint32_t(section->VirtualAddress));
        }
        if (size < kUe3SettingsDataTailSize)
          continue;

        sectionRanges.push_back({ base + section->VirtualAddress, size, executable });
      }

      struct SectionCopy {
        uintptr_t address = 0;
        std::vector<uint8_t> bytes;
      };

      const auto readSections = [&](bool executable, std::vector<SectionCopy>& out) {
        for (const SectionRange& range : sectionRanges) {
          if (range.executable != executable)
            continue;

          SectionCopy copy;
          copy.address = range.address;
          copy.bytes.resize(range.size);
          SIZE_T sectionRead = 0;
          if (!::ReadProcessMemory(proc, reinterpret_cast<LPCVOID>(range.address),
                                   copy.bytes.data(), range.size, &sectionRead) ||
              sectionRead < range.size)
            return false;
          out.push_back(std::move(copy));
        }
        return !out.empty();
      };

      // The code sections are two orders of magnitude larger than the data ones, so they are
      // only read once the data side has something worth looking for. Scans repeat until the
      // engine has populated GSystemSettings, and paying for the code copy every retry would
      // be a visible hitch each second.
      std::vector<SectionCopy> dataSections;
      if (!readSections(false, dataSections))
        return NgxLocateResult::Incomplete;

      // ---- Resolution-anchored candidate search -------------------------------------------
      struct Candidate {
        const SectionCopy* section = nullptr;
        size_t offset = 0;              // ScreenPercentage, within the section
        bool strictTail = false;
        bool fullscreenAgrees = false;
        int32_t score = 0;
        float screenPercentage = 100.0f;
        int32_t upscale = 1;
        int32_t maxMultisamples = 0;
        uint32_t defaultsStride = 0;
        uint32_t defaultsRepeats = 0;
      };
      // Relaxed hits are only consulted when nothing matched the stock tail layout, so an
      // unusual build cannot bury the obvious answer under near-misses.
      std::vector<Candidate> candidates;
      std::vector<Candidate> relaxedCandidates;
      constexpr size_t kMaxCandidates = 64;

      const int32_t expectedResX = int32_t(inputs.backBufferWidth);
      const int32_t expectedResY = int32_t(inputs.backBufferHeight);
      const int32_t expectedFullscreen = inputs.windowed ? 0 : 1;

      const auto isRejected = [&](uintptr_t address) {
        return std::find(rejectedCandidates.begin(), rejectedCandidates.end(), address) !=
               rejectedCandidates.end();
      };

      for (const SectionCopy& section : dataSections) {
        const uint8_t* bytes = section.bytes.data();
        const size_t length = section.bytes.size();

        if (inputs.forcedScreenPercentageRva != 0) {
          const uintptr_t forcedAddress = base + inputs.forcedScreenPercentageRva;
          if (forcedAddress < section.address ||
              forcedAddress + kUe3UpscaleFromScreenPercentage + sizeof(int32_t) >
                section.address + length)
            continue;

          const size_t offset = size_t(forcedAddress - section.address);
          // A pinned address the probe already disproved is not worth retrying.
          if (isRejected(forcedAddress))
            continue;

          Candidate candidate;
          candidate.section = &section;
          candidate.offset = offset;
          candidate.strictTail = ngxLooksLikeSettingsRecordTail(bytes + offset, length - offset);
          candidate.screenPercentage = ngxReadFloat(bytes + offset);
          candidate.upscale = ngxReadInt32(bytes + offset + kUe3UpscaleFromScreenPercentage);
          if (candidate.strictTail) {
            candidate.maxMultisamples =
              ngxReadInt32(bytes + offset + kUe3MaxMultisamplesFromScreenPercentage);
          }
          candidates.push_back(candidate);
          continue;
        }

        for (size_t resOffset = 0; resOffset + 2 * sizeof(int32_t) <= length;
             resOffset += sizeof(int32_t)) {
          if (ngxReadInt32(bytes + resOffset) != expectedResX ||
              ngxReadInt32(bytes + resOffset + sizeof(int32_t)) != expectedResY)
            continue;

          // Strict stock layout first: ScreenPercentage sits two dwords ahead of ResX, with
          // bFullscreen and MaxMultiSamples behind ResY.
          if (resOffset >= kUe3ResXFromScreenPercentage) {
            const size_t offset = resOffset - kUe3ResXFromScreenPercentage;
            if (ngxLooksLikeSettingsRecordTail(bytes + offset, length - offset) &&
                !isRejected(section.address + offset)) {
              Candidate candidate;
              candidate.section = &section;
              candidate.offset = offset;
              candidate.strictTail = true;
              candidate.screenPercentage = ngxReadFloat(bytes + offset);
              candidate.upscale = ngxReadInt32(bytes + offset + kUe3UpscaleFromScreenPercentage);
              candidate.maxMultisamples =
                ngxReadInt32(bytes + offset + kUe3MaxMultisamplesFromScreenPercentage);
              // Borderless windowed makes the engine's own idea of fullscreen disagree with
              // D3D9's, so agreement raises confidence rather than gating the candidate.
              candidate.fullscreenAgrees =
                ngxReadInt32(bytes + offset + kUe3FullscreenFromScreenPercentage) ==
                expectedFullscreen;
              if (candidates.size() < kMaxCandidates)
                candidates.push_back(candidate);
              continue;
            }
          }

          // Relaxed pass for builds that do not use the stock tail adjacency: only the
          // ScreenPercentage/bUpscaleScreenPercentage pair itself is required, and the probe
          // carries the burden of proof.
          for (uint32_t back = kUe3ResXFromScreenPercentage + sizeof(int32_t);
               back <= kUe3ResXFromScreenPercentage + kUe3MaxRelaxedScreenPercentageBacktrack &&
               relaxedCandidates.size() < kMaxCandidates;
               back += sizeof(int32_t)) {
            if (resOffset < back)
              break;
            const size_t offset = resOffset - back;
            const int32_t upscale = ngxReadInt32(bytes + offset + kUe3UpscaleFromScreenPercentage);
            if (!ngxIsPlausibleScreenPercentage(ngxReadFloat(bytes + offset)) ||
                (upscale != 0 && upscale != 1) ||
                isRejected(section.address + offset))
              continue;

            Candidate candidate;
            candidate.section = &section;
            candidate.offset = offset;
            candidate.screenPercentage = ngxReadFloat(bytes + offset);
            candidate.upscale = upscale;
            relaxedCandidates.push_back(candidate);
          }
        }
      }

      if (candidates.empty())
        candidates = std::move(relaxedCandidates);

      if (candidates.empty()) {
        outSettings.missReason =
          "no FSystemSettings record in writable data matched the live resolution";
        return NgxLocateResult::Miss;
      }

      std::vector<SectionCopy> executableSections;
      if (!readSections(true, executableSections))
        return NgxLocateResult::Incomplete;

      // ---- Scoring: structural repeats and the optional ini cross-check --------------------
      for (Candidate& candidate : candidates) {
        const uint8_t* bytes = candidate.section->bytes.data();
        const size_t length = candidate.section->bytes.size();

        candidate.score = (candidate.strictTail ? 100 : 0) +
                          (candidate.fullscreenAgrees ? 20 : 0);

        // FSystemSettings embeds Defaults[FSL_LevelCount] behind the live record, so the tail
        // shape repeats at a constant stride equal to sizeof(FSystemSettingsData).
        uint32_t bestStride = 0;
        uint32_t bestRepeats = 0;
        if (candidate.strictTail) {
          // Bounded: the pairwise stride search below is quadratic in the number of hits.
          constexpr size_t kMaxTailHits = 64;
          std::vector<size_t> tailHits;
          const size_t searchEnd =
            std::min(length, candidate.offset + kUe3DefaultsSearchWindow);
          for (size_t offset = candidate.offset + sizeof(int32_t);
               offset + kUe3SettingsDataTailSize <= searchEnd && tailHits.size() < kMaxTailHits;
               offset += sizeof(int32_t)) {
            if (ngxLooksLikeSettingsRecordTail(bytes + offset, length - offset))
              tailHits.push_back(offset);
          }

          for (size_t i = 0; i < tailHits.size(); i++) {
            for (size_t j = i + 1; j < tailHits.size(); j++) {
              const size_t stride = tailHits[j] - tailHits[i];
              if (stride < kUe3MinSettingsDataSize || stride > kUe3MaxSettingsDataSize)
                continue;

              uint32_t repeats = 1;
              for (size_t next = tailHits[j] + stride;
                   next + kUe3SettingsDataTailSize <= searchEnd; next += stride) {
                if (!ngxLooksLikeSettingsRecordTail(bytes + next, length - next))
                  break;
                repeats++;
              }

              if (repeats > bestRepeats) {
                bestRepeats = repeats;
                bestStride = uint32_t(stride);
              }
            }
          }
        }

        if (bestRepeats >= kUe3MinDefaultsRepeats) {
          candidate.defaultsStride = bestStride;
          candidate.defaultsRepeats = bestRepeats;
          candidate.score += 20 * int32_t(std::min<uint32_t>(bestRepeats, 5u));
        }

        const NgxIniSystemSettings& ini = inputs.ini;
        if (ini.found) {
          if (ini.hasScreenPercentage &&
              std::fabs(ini.screenPercentage - candidate.screenPercentage) < 0.01f)
            candidate.score += 50;
          if (ini.hasUpscale && ini.upscale == candidate.upscale)
            candidate.score += 15;
          if (candidate.strictTail && ini.hasMaxMultisamples &&
              ini.maxMultisamples == candidate.maxMultisamples)
            candidate.score += 25;
        }
      }

      std::stable_sort(candidates.begin(), candidates.end(),
                       [](const Candidate& a, const Candidate& b) {
                         return a.score > b.score;
                       });

      const Candidate& chosen = candidates.front();
      const uint8_t* dataBytes = chosen.section->bytes.data();
      const size_t dataLength = chosen.section->bytes.size();
      const uintptr_t screenPercentageAddress = chosen.section->address + chosen.offset;

      outSettings.screenPercentageAddress = screenPercentageAddress;
      outSettings.currentScreenPercentage = chosen.screenPercentage;
      outSettings.currentUpscaleScreenPercentage = chosen.upscale;
      outSettings.currentMaxMultisamples = chosen.maxMultisamples;
      outSettings.settingsDataStride = chosen.defaultsStride;
      outSettings.defaultsRepeats = chosen.defaultsRepeats;
      outSettings.strictTailLayout = chosen.strictTail;
      outSettings.iniConfirmed =
        inputs.ini.found && inputs.ini.hasScreenPercentage &&
        std::fabs(inputs.ini.screenPercentage - chosen.screenPercentage) < 0.01f;

      // ---- Code scan ----------------------------------------------------------------------
      const uint32_t screenPercentageOperand = uint32_t(screenPercentageAddress);
      const uint32_t upscaleOperand =
        uint32_t(screenPercentageAddress + kUe3UpscaleFromScreenPercentage);

      struct AbsoluteSite {
        uintptr_t address = 0;
        uint32_t original = 0;
        bool upscale = false;
      };
      std::vector<AbsoluteSite> absoluteSites;
      std::vector<uintptr_t> seedAddresses;

      struct RelativeWindow {
        uint32_t structOffset = 0;
        NgxOperandSite floatSite;
        NgxOperandSite intSite;
      };
      std::vector<RelativeWindow> relativeWindows;
      constexpr size_t kRelativePairWindow = 48;

      // Two float reads in one function can pair with the same integer read, so sites are
      // deduplicated by address rather than trusting one entry per match.
      const auto appendUniqueSite = [](std::vector<NgxOperandSite>& sites,
                                       const NgxOperandSite& site) {
        for (const NgxOperandSite& existing : sites) {
          if (existing.address == site.address)
            return;
        }
        sites.push_back(site);
      };

      for (const SectionCopy& section : executableSections) {
        const uint8_t* bytes = section.bytes.data();
        const size_t length = section.bytes.size();

        for (size_t offset = 0; offset + sizeof(uint32_t) <= length; offset++) {
          uint32_t operand = 0;
          std::memcpy(&operand, bytes + offset, sizeof(operand));

          if (operand == screenPercentageOperand &&
              ngxIsAbsoluteFloatReadSite(bytes, offset)) {
            if (ngxReadsPercentScaleConstantNearby(proc, bytes, offset, length))
              seedAddresses.push_back(section.address + offset);
            if (absoluteSites.size() < kNgxMaxRawOperandSites)
              absoluteSites.push_back({ section.address + offset, operand, false });
          } else if (operand == upscaleOperand && ngxIsAbsoluteIntReadSite(bytes, offset)) {
            if (absoluteSites.size() < kNgxMaxRawOperandSites)
              absoluteSites.push_back({ section.address + offset, operand, true });
          }
        }
      }

      // Without a scale-factor conversion there is no renderer-facing reader to redirect, and
      // patching the copies alone would corrupt whatever they feed.
      if (seedAddresses.empty()) {
        outSettings.missReason =
          "the record was found but no code converts ScreenPercentage into a scale factor";
        return NgxLocateResult::Miss;
      }

      const auto nearSeed = [&](uintptr_t address) {
        for (const uintptr_t seed : seedAddresses) {
          const uintptr_t distance = address > seed ? address - seed : seed - address;
          if (distance <= kNgxSettingsCodeClusterRadius)
            return true;
        }
        return false;
      };

      for (const AbsoluteSite& site : absoluteSites) {
        if (!nearSeed(site.address))
          continue;
        appendUniqueSite(site.upscale ? outSettings.directUpscaleScreenPercentageSites
                                      : outSettings.directScreenPercentageSites,
                         { site.address, site.original });
      }

      // Without ScaleScreenCoords' absolute reader nothing can shrink the view.
      if (outSettings.directScreenPercentageSites.empty() ||
          outSettings.directScreenPercentageSites.size() > kNgxMaxOperandSites ||
          outSettings.directUpscaleScreenPercentageSites.size() > kNgxMaxOperandSites) {
        outSettings.missReason = "no usable absolute ScreenPercentage readers";
        return NgxLocateResult::Miss;
      }

      // FSystemSettings::NeedsUpscale() reads both fields through `this`, and
      // UnScaleScreenCoords provably inlines it right beside an absolute ScreenPercentage read.
      // So the engine's own displacement is simply the one that appears next to a seed - which
      // is what identifies GSystemSettings without having to recognise the *head* of the
      // record. The head is the part that genuinely varies: the FSystemSettingsData sub-struct
      // split (DetailMode followed by a run of UBOOLs) only exists in later UE3 versions, and
      // Mirror's Edge predates it.
      for (const SectionCopy& section : executableSections) {
        const uint8_t* bytes = section.bytes.data();
        const size_t length = section.bytes.size();

        for (const uintptr_t seed : seedAddresses) {
          if (seed < section.address || seed - section.address >= length)
            continue;

          const size_t seedOffset = size_t(seed - section.address);
          const size_t begin = seedOffset > kNgxSettingsCodeClusterRadius
            ? seedOffset - kNgxSettingsCodeClusterRadius
            : 0;
          const size_t end = std::min(length, seedOffset + kNgxSettingsCodeClusterRadius);

          for (size_t offset = begin;
               offset < end && relativeWindows.size() < kNgxMaxOperandSites; offset++) {
            const NgxRelativeOperand decoded = ngxDecodeRelativeRead(bytes, offset, length);
            if (!decoded.valid || !decoded.isFloat ||
                decoded.displacement < kUe3MinSettingsDataSize ||
                decoded.displacement > kUe3MaxSettingsDataSize ||
                decoded.displacement > chosen.offset)
              continue;

            const uintptr_t floatSiteAddress = section.address + decoded.operandOffset;
            bool alreadyKnown = false;
            for (const RelativeWindow& window : relativeWindows)
              alreadyKnown |= window.floatSite.address == floatSiteAddress;
            if (alreadyKnown)
              continue;

            const size_t pairBegin = offset > kRelativePairWindow ? offset - kRelativePairWindow : 0;
            const size_t pairEnd = std::min(length, offset + kRelativePairWindow);
            for (size_t pairOffset = pairBegin; pairOffset < pairEnd; pairOffset++) {
              const NgxRelativeOperand paired = ngxDecodeRelativeRead(bytes, pairOffset, length);
              if (!paired.valid || paired.isFloat ||
                  paired.baseRegister != decoded.baseRegister ||
                  paired.displacement != decoded.displacement + kUe3UpscaleFromScreenPercentage)
                continue;

              relativeWindows.push_back({
                decoded.displacement,
                { floatSiteAddress, decoded.displacement },
                { section.address + paired.operandOffset, paired.displacement },
              });
              break;
            }
          }
        }
      }

      // The displacement backed by the most windows wins; the Defaults[] stride and the record
      // head (when this engine version has one) are independent derivations that break ties.
      const uint32_t strideDerivedOffset =
        chosen.defaultsStride > kUe3SettingsDataTailSize
          ? chosen.defaultsStride - kUe3SettingsDataTailSize
          : 0;
      uint32_t bestStructOffset = 0;
      size_t bestWindowCount = 0;
      int32_t bestTieBreak = -1;
      for (const RelativeWindow& candidateWindow : relativeWindows) {
        const uint32_t structOffset = candidateWindow.structOffset;
        size_t count = 0;
        for (const RelativeWindow& window : relativeWindows)
          count += window.structOffset == structOffset ? 1 : 0;

        const size_t headOffset = chosen.offset - structOffset;
        const int32_t tieBreak =
          (structOffset == strideDerivedOffset ? 2 : 0) +
          (ngxLooksLikeSettingsRecordHead(dataBytes + headOffset, dataLength - headOffset) ? 1 : 0);

        if (count > bestWindowCount ||
            (count == bestWindowCount && tieBreak > bestTieBreak)) {
          bestWindowCount = count;
          bestTieBreak = tieBreak;
          bestStructOffset = structOffset;
        }
      }

      if (bestWindowCount != 0 && bestStructOffset <= chosen.offset) {
        outSettings.screenPercentageStructOffset = bestStructOffset;
        outSettings.systemSettingsAddress = screenPercentageAddress - bestStructOffset;
        for (const RelativeWindow& window : relativeWindows) {
          if (window.structOffset != bestStructOffset)
            continue;
          appendUniqueSite(outSettings.relativeScreenPercentageSites, window.floatSite);
          appendUniqueSite(outSettings.relativeUpscaleScreenPercentageSites, window.intSite);
        }
      }

      return NgxLocateResult::Found;
    }

  }

  void D3D9Rtx::bootstrapNgxPassthroughUpscaler(uint32_t displayWidth, uint32_t displayHeight) {
    if (m_ngxPassthroughBootstrapped) {
      return;
    }

    if (!m_frameOptions.valid) {
      refreshFrameOptionCache();
    }

    if (!m_frameOptions.ngxPassthroughMode) {
      return;
    }

    // Preset sync already runs in RtxInitializer; here we only prime XeSS input resolution
    // (what the Remix UI does via getXeSSInputResolution) once the display size is known.
    if (displayWidth > 0 && displayHeight > 0) {
      m_parent->GetDXVKDevice()->getCommon()->metaNgxPassthrough()
        .syncXeSSInputResolution(displayWidth, displayHeight);
    }

    m_ngxPassthroughBootstrapped = true;
  }

  bool D3D9Rtx::ngxRuntimeOwnsUpscale() const {
    return m_ngxGameSettingsRedirectsValid && m_ngxRuntimeOwnedUpscale;
  }

  void D3D9Rtx::mirrorNgxLiveScreenPercentage() {
    float liveScreenPercentage = 0.0f;
    int32_t liveUpscale = 0;
    bool rollbackVerified = true;
    const bool mirrored =
      ngxReadProcessExact(m_ngxGameProcess, m_ngxScreenPercentageRemoteAddr,
                          liveScreenPercentage) &&
      ngxReadProcessExact(m_ngxGameProcess,
                          m_ngxScreenPercentageRemoteAddr + sizeof(float),
                          liveUpscale) &&
      ngxUpdateScreenPercentageShadow(
        m_ngxGameProcess, m_ngxGameSettingsShadowRemoteAddr,
        liveScreenPercentage, liveUpscale, &rollbackVerified);

    if (!mirrored) {
      ONCE(Logger::warn("[RTX NGX Passthrough] Could not mirror the game's live "
                        "ScreenPercentage/UpscaleScreenPercentage into the renderer shadow; "
                        "retaining the last effective values."));
      if (!rollbackVerified) {
        ONCE(Logger::err("[RTX NGX Passthrough] ScreenPercentage shadow rollback could "
                         "not be verified after the failed update."));
      }
    } else {
      if (m_ngxScreenPercentageDriven) {
        Logger::info(str::format(
          "[RTX NGX Passthrough] ScreenPercentage override released; renderer follows the "
          "game's live value ", liveScreenPercentage, " (UpscaleScreenPercentage ",
          liveUpscale, ")."));
      }
      m_ngxScreenPercentageDriven = false;
      m_ngxScreenPercentageLastLogged = 0.0f;
    }

    m_ngxGameScreenPercentage =
      (mirrored && m_frameOptions.ngxPassthroughMode &&
       std::isfinite(liveScreenPercentage) && liveScreenPercentage > 0.0f)
        ? liveScreenPercentage
        : 0.0f;
  }

  bool D3D9Rtx::writeNgxScreenPercentageShadow(float screenPercentage) {
    // Engine-owned upscale needs NeedsUpscale() to agree with the reduced view. When the
    // relative readers are unavailable the flag is pinned off instead, so any inlined copy of
    // NeedsUpscale() agrees with the out-of-line one - which still reads the game's own,
    // untouched values and therefore reports no upscale.
    const int32_t upscale = m_ngxRuntimeOwnedUpscale ? 0 : 1;

    bool rollbackVerified = true;
    if (ngxUpdateScreenPercentageShadow(
          m_ngxGameProcess, m_ngxGameSettingsShadowRemoteAddr,
          screenPercentage, upscale, &rollbackVerified))
      return true;

    ONCE(Logger::warn("[RTX NGX Passthrough] Could not transactionally update the "
                      "renderer-isolated ScreenPercentage shadow; it will be retried."));
    if (!rollbackVerified) {
      ONCE(Logger::err("[RTX NGX Passthrough] ScreenPercentage shadow rollback could "
                       "not be verified after the failed update."));
    }
    return false;
  }

  void D3D9Rtx::updateNgxSettingsProbe() {
    if (m_ngxSettingsProbeState != NgxSettingsProbeState::Pending)
      return;

    const uint32_t backBufferWidth =
      m_activePresentParams.has_value() ? m_activePresentParams->BackBufferWidth : 0;
    const uint32_t backBufferHeight =
      m_activePresentParams.has_value() ? m_activePresentParams->BackBufferHeight : 0;

    if (backBufferWidth != 0 && backBufferHeight != 0 && m_ngxSceneViewportValid) {
      const int32_t expectedWidth =
        int32_t(float(backBufferWidth) * m_ngxSettingsProbeValue / 100.0f);
      const int32_t expectedHeight =
        int32_t(float(backBufferHeight) * m_ngxSettingsProbeValue / 100.0f);
      const int32_t widthError = int32_t(m_ngxSceneViewport.Width) - expectedWidth;
      const int32_t heightError = int32_t(m_ngxSceneViewport.Height) - expectedHeight;

      if (widthError >= -kNgxScreenPercentageProbeTolerance &&
          widthError <= kNgxScreenPercentageProbeTolerance &&
          heightError >= -kNgxScreenPercentageProbeTolerance &&
          heightError <= kNgxScreenPercentageProbeTolerance) {
        m_ngxSettingsProbeState = NgxSettingsProbeState::Confirmed;
        Logger::info(str::format(
          "[RTX NGX Passthrough] Game settings redirects confirmed: driving ",
          m_ngxSettingsProbeValue, "% produced a ", m_ngxSceneViewport.Width, "x",
          m_ngxSceneViewport.Height, " scene viewport. Upscale owner: ",
          (m_ngxRuntimeOwnedUpscale ? "runtime." : "engine.")));
        return;
      }
    }

    if (m_ue3FrameCounter - m_ngxSettingsProbeStartFrame < kNgxScreenPercentageProbeFrames)
      return;

    Logger::warn(str::format(
      "[RTX NGX Passthrough] The candidate ScreenPercentage at 0x", std::hex,
      m_ngxScreenPercentageRemoteAddr, std::dec, " did not move the game's scene viewport within ",
      kNgxScreenPercentageProbeFrames, " frames; rejecting it."));

    m_ngxSettingsProbeState = NgxSettingsProbeState::Failed;
    m_ngxRejectedSettingsCandidates.push_back(m_ngxScreenPercentageRemoteAddr);
    restoreNgxGameSettingsRedirects();

    // Only a clean rollback leaves the process in a state where another candidate can be tried.
    constexpr size_t kMaxRejectedCandidates = 4;
    if (m_ngxGameSettingsShadowRemoteAddr == 0 &&
        m_ngxRejectedSettingsCandidates.size() < kMaxRejectedCandidates) {
      if (m_ngxGameProcessOwned && m_ngxGameProcess != nullptr)
        ::CloseHandle(m_ngxGameProcess);
      m_ngxGameProcess = nullptr;
      m_ngxGameProcessOwned = false;
      m_ngxGameProcessId = 0;
      m_ngxScreenPercentageRemoteAddr = 0;
      m_ngxGameScreenPercentage = 0.0f;
      m_ngxScreenPercentageScanDone = false;
      m_ngxSettingsProbeState = NgxSettingsProbeState::Idle;
    } else {
      m_ngxScreenPercentageScanDone = true;
    }
  }

  void D3D9Rtx::applyNgxPassthroughScreenPercentage() {
    const bool driving = m_frameOptions.ngxPassthroughMode &&
                         RtxNgxPassthrough::driveGameScreenPercentage();

    // The camera gate needs ScreenPercentage even when automatic driving is disabled.
    if (m_ngxScreenPercentageRemoteAddr == 0 && !m_ngxScreenPercentageScanDone &&
        m_frameOptions.ngxPassthroughMode) {
      const uint64_t nowMs = ::GetTickCount64();
      if (nowMs >= m_ngxScreenPercentageNextScanMs) {
        m_ngxScreenPercentageNextScanMs = nowMs + kNgxSettingsScanRetryIntervalMs;
        locateNgxPassthroughGameSettings();
      }
    }

    if (m_ngxScreenPercentageRemoteAddr == 0 ||
        m_ngxGameSettingsShadowRemoteAddr == 0 ||
        m_ngxGameProcess == nullptr) {
      m_ngxGameScreenPercentage = 0.0f;
      return;
    }

    // A retained partial transaction is mirrored but never driven.
    if (!m_ngxGameSettingsRedirectsValid || !driving) {
      // A probe interrupted by driving being switched off would otherwise run out its frame
      // budget while nothing is being driven, and reject a candidate it never tested.
      if (m_ngxSettingsProbeState == NgxSettingsProbeState::Pending)
        m_ngxSettingsProbeState = NgxSettingsProbeState::Idle;

      mirrorNgxLiveScreenPercentage();
      return;
    }

    uint32_t displayHeight = 0;
    uint32_t displayWidth = 0;
    if (m_activePresentParams.has_value()) {
      displayHeight = m_activePresentParams->BackBufferHeight;
      displayWidth = m_activePresentParams->BackBufferWidth;
    }

    bootstrapNgxPassthroughUpscaler(displayWidth, displayHeight);

    float screenPercentage = m_parent->GetDXVKDevice()->getCommon()->metaNgxPassthrough()
      .screenPercentageForDisplay(displayWidth, displayHeight);

    // A title whose composite cannot be separated from its upscale is served by the runtime-owned
    // path, where the engine's pass runs untouched into the reduced rect. Without that path a
    // reduced resolution either loses the game's grade or leaves a bordered image, so the only
    // correct answer left is full resolution.
    if (m_ngxEngineCompositeUnsafe && !ngxRuntimeOwnsUpscale()) {
      screenPercentage = 100.0f;
    }

    if (!std::isfinite(screenPercentage) || screenPercentage <= 0.0f) {
      ONCE(Logger::warn("[RTX NGX Passthrough] Upscaler returned an invalid "
                        "ScreenPercentage; retaining the previous effective value."));
      return;
    }

    // Nothing is trusted until the game's own scene viewport has been seen responding to a value
    // we drove, so the probe has to drive something the game must visibly react to. There is
    // nothing to prove while the request matches what the game is already using - and probing
    // then would cost a visible resolution blip for no gain, so keep mirroring until the
    // selected preset actually asks for something different.
    if (m_ngxSettingsProbeState == NgxSettingsProbeState::Idle) {
      if (m_ngxGameScreenPercentage > 0.0f &&
          std::fabs(screenPercentage - m_ngxGameScreenPercentage) < 0.5f) {
        mirrorNgxLiveScreenPercentage();
        return;
      }

      m_ngxSettingsProbeState = NgxSettingsProbeState::Pending;
      m_ngxSettingsProbeStartFrame = m_ue3FrameCounter;
      // Probing at the target itself avoids a blip whenever the selected preset is reduced.
      m_ngxSettingsProbeValue = screenPercentage < 90.0f
        ? screenPercentage
        : kNgxScreenPercentageProbeValue;
      Logger::info(str::format(
        "[RTX NGX Passthrough] Verifying the game-settings redirects by driving ",
        m_ngxSettingsProbeValue, "%."));
    }

    if (m_ngxSettingsProbeState == NgxSettingsProbeState::Pending) {
      updateNgxSettingsProbe();
      if (m_ngxSettingsProbeState == NgxSettingsProbeState::Pending) {
        if (writeNgxScreenPercentageShadow(m_ngxSettingsProbeValue))
          m_ngxGameScreenPercentage = m_ngxSettingsProbeValue;
        return;
      }
    }

    if (m_ngxSettingsProbeState != NgxSettingsProbeState::Confirmed)
      return;

    if (!writeNgxScreenPercentageShadow(screenPercentage))
      return;

    m_ngxScreenPercentageDriven = true;
    m_ngxGameScreenPercentage = screenPercentage;

    if (screenPercentage != m_ngxScreenPercentageLastLogged) {
      m_ngxScreenPercentageLastLogged = screenPercentage;
      Logger::info(str::format("[RTX NGX Passthrough] ", RtxNgxPassthrough::upscalerModeLabel(),
                               " mode -> effective ScreenPercentage ", screenPercentage,
                               (screenPercentage >= 99.5f ? " (native)." : " (Super Resolution).")));
    }
  }

  void D3D9Rtx::locateNgxPassthroughGameSettings() {
    // The live resolution is the anchor the whole layout is discovered from, so there is
    // nothing to look for until the device's present parameters are known.
    if (!m_activePresentParams.has_value())
      return;

    NgxGameSettingsScanInputs scanInputs;
    scanInputs.backBufferWidth = m_activePresentParams->BackBufferWidth;
    scanInputs.backBufferHeight = m_activePresentParams->BackBufferHeight;
    scanInputs.windowed = m_activePresentParams->Windowed != FALSE;
    scanInputs.forcedScreenPercentageRva =
      uint32_t(std::max(0, RtxNgxPassthrough::systemSettingsScreenPercentageRva()));
    if (scanInputs.backBufferWidth == 0 || scanInputs.backBufferHeight == 0)
      return;

    // Current process + bridge parent; Incomplete results retry within the budget.
    struct Candidate { HANDLE handle; DWORD pid; bool ownsHandle; };
    std::vector<Candidate> candidates;
    candidates.push_back({ ::GetCurrentProcess(), ::GetCurrentProcessId(), false });

    bool anyIncomplete = false;

    const DWORD parentPid = ngxGetParentPid();
    if (parentPid != 0) {
      HANDLE parent = ::OpenProcess(PROCESS_VM_READ | PROCESS_VM_WRITE | PROCESS_VM_OPERATION | PROCESS_QUERY_INFORMATION,
                                    FALSE, parentPid);
      if (parent != nullptr) {
        candidates.push_back({ parent, parentPid, true });
      } else {
        anyIncomplete = true;
      }
    } else {
      anyIncomplete = true;
    }

    for (const Candidate& candidate : candidates) {
      uintptr_t moduleBase = 0;
      uint32_t moduleSize = 0;
      std::wstring modulePath;
      if (!ngxGetMainModule(candidate.pid, moduleBase, moduleSize, modulePath)) {
        anyIncomplete = true;
        continue;
      }

      scanInputs.ini = ngxCachedIniSystemSettings(modulePath);

      NgxLocatedGameSettings located;
      const NgxLocateResult result =
        ngxLocateGameSettings(candidate.handle, moduleBase, moduleSize, scanInputs,
                              m_ngxRejectedSettingsCandidates, located);
      if (result == NgxLocateResult::Incomplete)
        anyIncomplete = true;
      if (result != NgxLocateResult::Found) {
        if (result == NgxLocateResult::Miss && candidate.ownsHandle) {
          ONCE(Logger::info(str::format(
            "[RTX NGX Passthrough] Game-settings scan of the parent game process found nothing: ",
            located.missReason, ". Retrying while the engine finishes starting up.")));
        }
        continue;
      }

      void* shadowMemory =
        ::VirtualAllocEx(candidate.handle, nullptr, sizeof(NgxGameSettingsShadow),
                         MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
      const uintptr_t shadowAddress = reinterpret_cast<uintptr_t>(shadowMemory);
      if (shadowMemory == nullptr ||
          shadowAddress > uintptr_t(UINT32_MAX) - sizeof(NgxGameSettingsShadow)) {
        if (shadowMemory != nullptr)
          ::VirtualFreeEx(candidate.handle, shadowMemory, 0, MEM_RELEASE);
        anyIncomplete = true;
        continue;
      }

      NgxGameSettingsShadow shadow = {
        located.currentScreenPercentage,
        located.currentUpscaleScreenPercentage,
      };

      if (!ngxWriteProcessExact(candidate.handle, shadowAddress, shadow)) {
        ::VirtualFreeEx(candidate.handle, shadowMemory, 0, MEM_RELEASE);
        anyIncomplete = true;
        continue;
      }

      const uint32_t shadowScreenPercentage =
        uint32_t(shadowAddress + kNgxShadowScreenPercentageOffset);
      const uint32_t shadowUpscaleScreenPercentage =
        uint32_t(shadowAddress + kNgxShadowUpscaleScreenPercentageOffset);
      // Relative sites keep their [reg+disp32] form; only the displacement moves, so they still
      // depend on the base register holding GSystemSettings.
      const uint32_t relativeShadowScreenPercentage =
        uint32_t((shadowAddress + kNgxShadowScreenPercentageOffset) -
                 located.systemSettingsAddress);
      const uint32_t relativeShadowUpscaleScreenPercentage =
        uint32_t((shadowAddress + kNgxShadowUpscaleScreenPercentageOffset) -
                 located.systemSettingsAddress);

      std::vector<NgxOperandRedirect> redirects;
      redirects.reserve(
        located.directScreenPercentageSites.size() +
        located.directUpscaleScreenPercentageSites.size() +
        located.relativeScreenPercentageSites.size() +
        located.relativeUpscaleScreenPercentageSites.size());

      for (const NgxOperandSite& site : located.directScreenPercentageSites)
        redirects.push_back({ site.address, site.original, shadowScreenPercentage,
                              0, false, false,
                              NgxOperandCategory::AbsoluteScreenPercentage });
      for (const NgxOperandSite& site : located.directUpscaleScreenPercentageSites)
        redirects.push_back({ site.address, site.original, shadowUpscaleScreenPercentage,
                              0, false, false,
                              NgxOperandCategory::AbsoluteUpscale });
      if (located.systemSettingsAddress != 0) {
        for (const NgxOperandSite& site : located.relativeScreenPercentageSites)
          redirects.push_back({ site.address, site.original, relativeShadowScreenPercentage,
                                0, false, false,
                                NgxOperandCategory::RelativeScreenPercentage });
        for (const NgxOperandSite& site : located.relativeUpscaleScreenPercentageSites)
          redirects.push_back({ site.address, site.original, relativeShadowUpscaleScreenPercentage,
                                0, false, false,
                                NgxOperandCategory::RelativeUpscale });
      }

      std::vector<NgxOperandRedirect> activeRedirects;
      activeRedirects.reserve(redirects.size());
      NgxOperandApplyResult applyResult;
      {
        NgxScopedThreadSuspension suspended(candidate.pid);
        if (suspended.complete())
          applyResult = ngxApplyOperandRedirects(candidate.handle, redirects, activeRedirects);
        else
          applyResult.skipped = redirects.size();
      }

      // Without ScaleScreenCoords' reader nothing can shrink the view, so a set that lost it is
      // worthless; roll the rest back rather than let the probe spend frames disproving it.
      const bool redirectsApplied =
        applyResult.appliedByCategory[size_t(NgxOperandCategory::AbsoluteScreenPercentage)] != 0;
      if (!redirectsApplied) {
        for (NgxOperandRedirect& redirect : activeRedirects) {
          if (!redirect.forceRestore && !ngxRestoreCodeOperand(candidate.handle, redirect))
            redirect.forceRestore = true;
        }
        activeRedirects.erase(
          std::remove_if(activeRedirects.begin(), activeRedirects.end(),
                         [](const NgxOperandRedirect& redirect) {
                           return !redirect.forceRestore;
                         }),
          activeRedirects.end());
      }

      if (!redirectsApplied) {
        if (activeRedirects.empty()) {
          ::VirtualFreeEx(candidate.handle, shadowMemory, 0, MEM_RELEASE);
          anyIncomplete = true;
          continue;
        }

        // Keep memory referenced by an operand that rollback could not restore.
        m_ngxGameProcess = candidate.handle;
        m_ngxGameProcessOwned = candidate.ownsHandle;
        m_ngxGameProcessId = candidate.pid;
        m_ngxScreenPercentageRemoteAddr =
          located.screenPercentageAddress;
        m_ngxGameSettingsShadowRemoteAddr = shadowAddress;
        for (const NgxOperandRedirect& redirect : activeRedirects) {
          m_ngxGameSettingsCodePatches.push_back({
            redirect.address, redirect.original, redirect.redirected,
            redirect.originalProtection,
            redirect.originalProtectionKnown, true
          });
          uint32_t currentOperand = 0;
          const bool currentReadable =
            ngxReadProcessExact(candidate.handle, redirect.address,
                                currentOperand);
          MEMORY_BASIC_INFORMATION memoryInfo = {};
          const bool protectionReadable =
            ::VirtualQueryEx(
              candidate.handle,
              reinterpret_cast<LPCVOID>(redirect.address),
              &memoryInfo, sizeof(memoryInfo)) == sizeof(memoryInfo);
          Logger::err(str::format(
            "[RTX NGX Passthrough] Outstanding operand at 0x", std::hex,
            redirect.address, ": original=0x", redirect.original,
            " redirected=0x", redirect.redirected,
            " current=",
            (currentReadable
              ? str::format("0x", std::hex, currentOperand)
              : std::string("<unreadable>")),
            " originalProtect=0x", redirect.originalProtection,
            " currentProtect=",
            (protectionReadable
              ? str::format("0x", std::hex, memoryInfo.Protect)
              : std::string("<unreadable>")),
            std::dec, "."));
        }
        m_ngxScreenPercentageScanDone = true;
        Logger::err("[RTX NGX Passthrough] Game-settings operand redirection failed and "
                    "could not be fully rolled back. Effective values remain mirrored from "
                    "the game, but automatic ScreenPercentage driving is disabled.");
        break;
      }

      m_ngxGameProcess = candidate.handle;
      m_ngxGameProcessOwned = candidate.ownsHandle;
      m_ngxGameProcessId = candidate.pid;
      m_ngxScreenPercentageRemoteAddr =
        located.screenPercentageAddress;
      m_ngxGameSettingsShadowRemoteAddr = shadowAddress;
      m_ngxGameSettingsRedirectsValid = true;
      m_ngxGameScreenPercentage = located.currentScreenPercentage;
      m_ngxGameSettingsCodePatches.reserve(activeRedirects.size());
      for (const NgxOperandRedirect& redirect : activeRedirects) {
        m_ngxGameSettingsCodePatches.push_back({
          redirect.address, redirect.original, redirect.redirected,
          redirect.originalProtection,
          redirect.originalProtectionKnown, redirect.forceRestore
        });
      }
      m_ngxScreenPercentageScanDone = true;

      // FSystemSettings::NeedsUpscale() reads both fields through `this`, so the engine only
      // keeps its own upscale when that relative pair was redirected as well.
      m_ngxEngineUpscaleAvailable =
        applyResult.appliedByCategory[size_t(NgxOperandCategory::RelativeScreenPercentage)] != 0 &&
        applyResult.appliedByCategory[size_t(NgxOperandCategory::RelativeUpscale)] != 0;

      const int upscaleOwner = RtxNgxPassthrough::screenPercentageUpscaleOwner();
      m_ngxRuntimeOwnedUpscale = upscaleOwner == 2 || !m_ngxEngineUpscaleAvailable;
      if (upscaleOwner == 1 && !m_ngxEngineUpscaleAvailable) {
        Logger::warn(
          "[RTX NGX Passthrough] screenPercentageUpscaleOwner requests the engine, but its "
          "NeedsUpscale() readers could not be redirected; the runtime will upscale instead.");
      }

      m_ngxSettingsProbeState = NgxSettingsProbeState::Idle;

      Logger::info(str::format(
        "[RTX NGX Passthrough] Game settings redirects installed in ",
        (candidate.ownsHandle
          ? "the parent game process (RTX Remix bridge)"
          : "the current process"),
        ": ScreenPercentage=0x", std::hex, located.screenPercentageAddress,
        ", GSystemSettings=0x", located.systemSettingsAddress,
        ", shadow=0x", shadowAddress, std::dec,
        ", live=", located.currentScreenPercentage, "/",
        located.currentUpscaleScreenPercentage, "."));
      Logger::info(str::format(
        "[RTX NGX Passthrough] Layout: ",
        (located.strictTailLayout ? "stock tail" : "relaxed tail"),
        ", ScreenPercentage offset 0x", std::hex, located.screenPercentageStructOffset,
        std::dec, ", FSystemSettingsData stride ", located.settingsDataStride,
        " (", located.defaultsRepeats, " Defaults repeats), ini cross-check ",
        (located.iniConfirmed ? "agrees" : "unavailable"),
        ". Operands: ", applyResult.applied, " patched, ", applyResult.skipped,
        " left stock, ", applyResult.unrestorable, " unrestorable. Upscale owner: ",
        (m_ngxRuntimeOwnedUpscale ? "runtime." : "engine.")));
      break;
    }

    for (const Candidate& candidate : candidates) {
      if (candidate.ownsHandle && candidate.handle != m_ngxGameProcess)
        ::CloseHandle(candidate.handle);
    }

    // Unlike a code-signature scan, this one reads engine state: GSystemSettings is only
    // populated once FSystemSettings::Initialize has run and the resolution has been applied,
    // so an early miss says nothing about whether the record exists. Keep retrying on the
    // shared budget instead of giving up on the first look.
    if (m_ngxScreenPercentageRemoteAddr == 0 &&
        m_ngxGameSettingsShadowRemoteAddr == 0 &&
        ++m_ngxScreenPercentageScanAttempts >= kNgxSettingsScanMaxAttempts) {
      m_ngxScreenPercentageScanDone = true;
      Logger::warn(str::format(
        "[RTX NGX Passthrough] No usable FSystemSettings record was found in the game's main "
        "module after ", kNgxSettingsScanMaxAttempts,
        (anyIncomplete ? " attempts (the process or module was not fully readable); "
                       : " attempts; "),
        "the DLSS mode selector will not drive render resolution. Set "
        "rtx.ngxPassthrough.systemSettingsScreenPercentageRva to pin the address, or use the "
        "in-game 'scale set ScreenPercentage <value>' console command."));
    }
  }

  void D3D9Rtx::restoreNgxGameSettingsRedirects() {
    if (m_ngxGameProcess == nullptr ||
        m_ngxGameSettingsShadowRemoteAddr == 0)
      return;

    // Mirror live values first so any unrestorable redirect stays coherent.
    bool screenShadowMirrored = false;
    if (m_ngxScreenPercentageRemoteAddr != 0) {
      float liveScreenPercentage = 0.0f;
      int32_t liveUpscale = 0;
      if (ngxReadProcessExact(m_ngxGameProcess,
                              m_ngxScreenPercentageRemoteAddr,
                              liveScreenPercentage) &&
          ngxReadProcessExact(m_ngxGameProcess,
                              m_ngxScreenPercentageRemoteAddr + sizeof(float),
                              liveUpscale)) {
        bool rollbackVerified = true;
        screenShadowMirrored =
          ngxUpdateScreenPercentageShadow(
            m_ngxGameProcess, m_ngxGameSettingsShadowRemoteAddr,
            liveScreenPercentage, liveUpscale,
            &rollbackVerified) &&
          rollbackVerified;
      }
    }

    bool allRedirectsReleased = true;
    bool suspensionComplete = false;
    bool shadowReleased = false;
    std::vector<uintptr_t> externallyChangedOperands;
    externallyChangedOperands.reserve(m_ngxGameSettingsCodePatches.size());
    {
      NgxScopedThreadSuspension suspended(m_ngxGameProcessId);
      suspensionComplete = suspended.complete();
      if (suspensionComplete) {
        for (auto it = m_ngxGameSettingsCodePatches.rbegin();
             it != m_ngxGameSettingsCodePatches.rend(); ++it) {
          NgxOperandRedirect redirect = {
            it->operandAddress,
            it->originalOperand,
            it->redirectedOperand,
            it->originalProtection,
            it->originalProtectionKnown,
            it->forceRestore,
          };
          uint32_t current = 0;
          if (!ngxReadProcessExact(m_ngxGameProcess, it->operandAddress,
                                   current)) {
            allRedirectsReleased = false;
            continue;
          }

          if (it->forceRestore ||
              current == it->redirectedOperand) {
            if (!ngxRestoreCodeOperand(m_ngxGameProcess, redirect))
              allRedirectsReleased = false;
          } else if (current != it->originalOperand) {
            // External owner; do not clobber. Safe to free our shadow for this site.
            externallyChangedOperands.push_back(it->operandAddress);
          }
        }

        // Confirm no live operand still references the shadow before free.
        for (const NgxGameSettingsCodePatch& patch :
             m_ngxGameSettingsCodePatches) {
          uint32_t current = 0;
          if (!ngxReadProcessExact(m_ngxGameProcess, patch.operandAddress,
                                   current) ||
              current == patch.redirectedOperand ||
              (patch.forceRestore && current != patch.originalOperand))
            allRedirectsReleased = false;
        }

        // Free while still suspended so resumed threads cannot race the release.
        if (allRedirectsReleased) {
          shadowReleased =
            ::VirtualFreeEx(
              m_ngxGameProcess,
              reinterpret_cast<LPVOID>(
                m_ngxGameSettingsShadowRemoteAddr),
              0, MEM_RELEASE) != FALSE;
        }
      }
    }

    if (!suspensionComplete) {
      Logger::warn(
        "[RTX NGX Passthrough] Could not quiesce the game threads to restore settings "
        "reader operands; retaining the renderer shadow allocation.");
      if (!screenShadowMirrored) {
        Logger::err(
          "[RTX NGX Passthrough] The retained renderer shadow could not be fully returned "
          "to the game's live values before teardown.");
      }
      return;
    }

    for (const uintptr_t address : externallyChangedOperands) {
      Logger::warn(str::format(
        "[RTX NGX Passthrough] Settings reader operand at 0x", std::hex,
        address, std::dec,
        " changed by another component; leaving its value intact."));
    }

    if (allRedirectsReleased) {
      if (!shadowReleased) {
        Logger::warn(
          "[RTX NGX Passthrough] Settings reader operands were restored, but the renderer "
          "shadow allocation could not be released.");
      }
      m_ngxGameSettingsCodePatches.clear();
      m_ngxGameSettingsShadowRemoteAddr = 0;
    } else {
      Logger::warn(
        "[RTX NGX Passthrough] One or more settings reader operands could not be restored; "
        "retaining the renderer shadow allocation so no game code points to freed memory.");
      if (!screenShadowMirrored) {
        Logger::err(
          "[RTX NGX Passthrough] The retained renderer shadow could not be fully returned "
          "to the game's live values.");
      }
    }

    m_ngxScreenPercentageDriven = false;
    m_ngxGameSettingsRedirectsValid = false;
    m_ngxEngineUpscaleAvailable = false;
    m_ngxRuntimeOwnedUpscale = false;
    m_ngxMissingEngineUpscaleFrames = 0;
  }

  // Reads the constants of the draw about to be suppressed. UE3's GammaCorrectionPixelShader is
  // `pow(saturate(lerp(scene * ColorScale, OverlayColor.rgb, OverlayColor.a)), InverseGamma)`;
  // when the post chain already gamma corrected into LDR scene colour the engine sets
  // InverseGamma to 1 and this reduces to a copy, which is why Mirror's Edge never needed it.
  void D3D9Rtx::captureNgxOutputTransform() {
    m_ngxOutputTransform = NgxOutputTransform();

    if (!m_parent->UseProgrammablePS() || d3d9State().pixelShader.ptr() == nullptr)
      return;

    const D3D9CommonShader* pixelShaderCommon = d3d9State().pixelShader->GetCommonShader();
    const Ue3ShaderFeatureInfo psInfo = getUe3ShaderFeatureInfo(pixelShaderCommon);

    if (psInfo.gammaInverseReg == Ue3ShaderFeatureInfo::kNgxNoRegister) {
      // Not UE3's GammaCorrectionPixelShader. Report what the suppressed draw actually is:
      // if it carries the Uber post-process constants then the engine composited its whole
      // post chain here, and replacing that draw costs far more than a gamma curve.
      ONCE(Logger::warn(str::format(
        "[RTX NGX Passthrough] The suppressed composite is not UE3's GammaCorrectionPixelShader "
        "(ps=0x", std::hex, pixelShaderCommon->GetBytecodeHash(), std::dec,
        ", gammaConstants=", psInfo.hasGammaConstants ? 1 : 0,
        ", toneMapConstants=", psInfo.hasToneMapConstants ? 1 : 0,
        ", exposureOrToneSampler=", psInfo.hasExposureOrToneSampler ? 1 : 0,
        ", sceneColorSampler=", psInfo.hasSceneColorSampler ? 1 : 0,
        ", colorScaleReg=", psInfo.gammaColorScaleReg != Ue3ShaderFeatureInfo::kNgxNoRegister ? 1 : 0,
        ", overlayColorReg=", psInfo.gammaOverlayColorReg != Ue3ShaderFeatureInfo::kNgxNoRegister ? 1 : 0,
        "); its colour transform cannot be carried over to the upscaled result.")));
      return;
    }

    const auto& constants = d3d9State().psConsts.fConsts;
    const auto readRegister = [&](uint32_t reg) -> const Vector4* {
      return reg < std::size(constants) ? &constants[reg] : nullptr;
    };

    const Vector4* inverseGamma = readRegister(psInfo.gammaInverseReg);
    if (inverseGamma == nullptr || !std::isfinite(inverseGamma->x) ||
        inverseGamma->x <= 0.0f || inverseGamma->x > 4.0f)
      return;

    m_ngxOutputTransform.inverseGamma = inverseGamma->x;

    if (const Vector4* colorScale = readRegister(psInfo.gammaColorScaleReg)) {
      m_ngxOutputTransform.colorScale[0] = colorScale->x;
      m_ngxOutputTransform.colorScale[1] = colorScale->y;
      m_ngxOutputTransform.colorScale[2] = colorScale->z;
    }

    if (const Vector4* overlayColor = readRegister(psInfo.gammaOverlayColorReg)) {
      m_ngxOutputTransform.overlayColor[0] = overlayColor->x;
      m_ngxOutputTransform.overlayColor[1] = overlayColor->y;
      m_ngxOutputTransform.overlayColor[2] = overlayColor->z;
      m_ngxOutputTransform.overlayColor[3] = overlayColor->w;
    }

    const bool isIdentity =
      m_ngxOutputTransform.inverseGamma == 1.0f &&
      m_ngxOutputTransform.colorScale[0] == 1.0f &&
      m_ngxOutputTransform.colorScale[1] == 1.0f &&
      m_ngxOutputTransform.colorScale[2] == 1.0f &&
      m_ngxOutputTransform.overlayColor[3] == 0.0f;
    m_ngxOutputTransform.enabled = !isIdentity;

    ONCE(Logger::info(str::format(
      "[RTX NGX Passthrough] Suppressed composite transform: InverseGamma ",
      m_ngxOutputTransform.inverseGamma, ", ColorScale (",
      m_ngxOutputTransform.colorScale[0], ", ", m_ngxOutputTransform.colorScale[1], ", ",
      m_ngxOutputTransform.colorScale[2], "), OverlayColor alpha ",
      m_ngxOutputTransform.overlayColor[3], " -> ",
      (m_ngxOutputTransform.enabled ? "reapplied to the upscaled result."
                                    : "identity, nothing to reapply."))));
  }

  void D3D9Rtx::emitNgxPassthroughFrameData() {
    if (m_ngxFrameDataEmitted)
      return;
    m_ngxFrameDataEmitted = true;

    Rc<DxvkImage> sceneDepth = m_ngxSceneDepthImage;

    // Runtime-owned upscale: the engine composited its reduced view rect into the backbuffer and
    // performed no upscale of its own, so DLSS reads that subrect and writes the full frame. The
    // depth subrect stays where the scene rasterized (scene color space), which is why the two
    // offsets are tracked separately.
    if (m_ngxUpscaleSourceImage == nullptr && m_ngxRuntimeUpscaleRectValid &&
        m_ngxColorTargetImage == nullptr && m_ngxFrameBackbufferImage != nullptr &&
        m_ngxSceneViewportValid) {
      m_ngxUpscaleSourceImage = m_ngxFrameBackbufferImage;
      m_ngxSubrect.offset = { int32_t(m_ngxSceneViewport.X), int32_t(m_ngxSceneViewport.Y) };
      m_ngxSubrect.extent = { m_ngxSceneViewport.Width, m_ngxSceneViewport.Height };
      m_ngxColorSubrectOffset = { int32_t(m_ngxRuntimeUpscaleRect.X),
                                  int32_t(m_ngxRuntimeUpscaleRect.Y) };
    }

    // A scene rendered into a reduced subrect (both dimensions - a single reduced dimension
    // is letterboxing, which full-rect DLAA handles fine) without any Super Resolution
    // interception means the depth subrect does not correspond to the full-resolution color
    // the injection sees; skip the DLSS inputs rather than feeding mismatched data.
    if (m_ngxUpscaleSourceImage == nullptr && m_ngxSceneViewportValid && m_activePresentParams.has_value()) {
      const uint32_t backBufferWidth = m_activePresentParams->BackBufferWidth;
      const uint32_t backBufferHeight = m_activePresentParams->BackBufferHeight;

      if (backBufferWidth != 0 && backBufferHeight != 0 &&
          uint64_t(m_ngxSceneViewport.Width) * 100 <= uint64_t(backBufferWidth) * 97 &&
          uint64_t(m_ngxSceneViewport.Height) * 100 <= uint64_t(backBufferHeight) * 97) {
        ONCE(Logger::warn("[RTX NGX Passthrough] Scene rendered into a reduced subrect without Super Resolution interception; DLSS is skipped."));
        sceneDepth = nullptr;

        // Neither the engine's upscale stretch nor its subrect composite was found. If the
        // engine was supposed to be doing the upscale, hand ownership to the runtime rather
        // than keep presenting a reduced image; the next frame renders through that path.
        if (!m_ngxRuntimeOwnedUpscale && m_ngxGameSettingsRedirectsValid &&
            RtxNgxPassthrough::screenPercentageUpscaleOwner() != 1 &&
            ++m_ngxMissingEngineUpscaleFrames >= kNgxMissingEngineUpscaleFrameLimit) {
          m_ngxRuntimeOwnedUpscale = true;
          m_ngxMissingEngineUpscaleFrames = 0;
          Logger::warn("[RTX NGX Passthrough] The engine's own upscale never appeared; taking "
                       "ownership of the Super Resolution upscale in the runtime instead.");
        }
      }
    } else {
      m_ngxMissingEngineUpscaleFrames = 0;
    }

    m_ngxVelocityStats.frameCameraValid = m_ngxFrameCameraValid;
    m_ngxVelocityStats.depthClears = m_ngxDepthClearsThisFrame;
    m_ngxVelocityStats.cameraTransposeFlips = m_ngxCameraTransposeFlips;

    // Accumulated across the window and logged as a rate. The per-sighting pairing-miss lines are
    // rate limited to a burst every couple of seconds, which shows what a miss looks like but not
    // how often one happens - and whether the capture is healthy is entirely a question of the
    // ratio between sightings that paired and sightings that had to register anew.
    m_ngxVelocityWindow.captured += m_ngxVelocityStats.captured;
    m_ngxVelocityWindow.capturedSkinned += m_ngxVelocityStats.capturedSkinned;
    m_ngxVelocityWindow.capturedDynamic += m_ngxVelocityStats.capturedDynamic;
    m_ngxVelocityWindow.capturedForeground += m_ngxVelocityStats.capturedForeground;
    m_ngxVelocityWindow.exactMatches += m_ngxVelocityStats.exactMatches;
    m_ngxVelocityWindow.newRegistrations += m_ngxVelocityStats.newRegistrations;
    m_ngxVelocityWindow.newRegistrationsSkinned += m_ngxVelocityStats.newRegistrationsSkinned;
    m_ngxVelocityWindow.missNoLastFrameSighting += m_ngxVelocityStats.missNoLastFrameSighting;
    m_ngxVelocityWindow.missBeyondTranslation += m_ngxVelocityStats.missBeyondTranslation;
    m_ngxVelocityWindow.missBeyondRotation += m_ngxVelocityStats.missBeyondRotation;
    m_ngxVelocityWindow.claimedWithoutVelocity += m_ngxVelocityStats.claimedWithoutVelocity;
    m_ngxVelocityWindow.pairedBeyondBounds += m_ngxVelocityStats.pairedBeyondBounds;
    m_ngxVelocityWindow.skippedNoCamera += m_ngxVelocityStats.skippedNoCamera;
    m_ngxVelocityWindow.skippedBudget += m_ngxVelocityStats.skippedBudget;
    m_ngxVelocityWindow.skippedZDisabled += m_ngxVelocityStats.skippedZDisabled;
    m_ngxVelocityWindow.skippedInstanceCap += m_ngxVelocityStats.skippedInstanceCap;
    m_ngxVelocityWindow.skippedDynamicBuffer += m_ngxVelocityStats.skippedDynamicBuffer;
    m_ngxVelocityWindow.skippedBonePalette += m_ngxVelocityStats.skippedBonePalette;
    m_ngxVelocityWindow.depthClears += m_ngxVelocityStats.depthClears;
    if (m_ngxVelocityStats.depthClears > 1) {
      m_ngxVelocityWindow.framesWithOrphanedDepthPhase++;
    }

    if (++m_ngxVelocityWindow.frames >= kNgxVelocityWindowFrames) {
      Logger::info(str::format(
        "[RTX NGX Passthrough][velocity] over ", m_ngxVelocityWindow.frames, " frames: ",
        m_ngxVelocityWindow.captured, " captured (", m_ngxVelocityWindow.capturedSkinned, " skinned, ",
        m_ngxVelocityWindow.capturedDynamic, " CPU-modified, ",
        m_ngxVelocityWindow.capturedForeground, " foreground-phase), ",
        m_ngxVelocityWindow.exactMatches, " paired to their own history, ",
        m_ngxVelocityWindow.newRegistrations, " registered anew (",
        m_ngxVelocityWindow.newRegistrationsSkinned, " of them skinned; ",
        m_ngxVelocityWindow.missNoLastFrameSighting, " unseen last frame / ",
        m_ngxVelocityWindow.missBeyondTranslation, " beyond translation / ",
        m_ngxVelocityWindow.missBeyondRotation, " beyond rotation), ",
        m_ngxVelocityWindow.claimedWithoutVelocity, " claimed without velocity, ",
        m_ngxVelocityWindow.pairedBeyondBounds, " paired beyond the motion bounds, skipped: ",
        m_ngxVelocityWindow.skippedNoCamera, " no camera / ",
        m_ngxVelocityWindow.skippedBudget, " over budget / ",
        m_ngxVelocityWindow.skippedZDisabled, " depth test off / ",
        m_ngxVelocityWindow.skippedInstanceCap, " over the instance cap / ",
        m_ngxVelocityWindow.skippedDynamicBuffer, " on unrecognised dynamic buffers / ",
        m_ngxVelocityWindow.skippedBonePalette, " over the bone palette cap; tracking ",
        m_ngxVelocityObjectCache.size(), " identities; ", m_ngxVelocityWindow.depthClears,
        " mid-scene depth clears over ", m_ngxVelocityWindow.framesWithOrphanedDepthPhase,
        " frames cleared more than once; scene-wide transform offset (",
        m_ngxGlobalTransformOffset.x, ",", m_ngxGlobalTransformOffset.y, ",",
        m_ngxGlobalTransformOffset.z, ")."));

      m_ngxVelocityWindow = NgxVelocityWindowTotals();
    }

    m_parent->EmitCs([cSceneDepth = sceneDepth,
                      cColorTarget = m_ngxColorTargetImage,
                      cColorMirror = m_ngxColorMirrorImage,
                      cUpscaleSource = m_ngxUpscaleSourceImage,
                      cSubrect = m_ngxSubrect,
                      cColorSubrectOffset = m_ngxColorSubrectOffset,
                      cOutputTransform = m_ngxOutputTransform,
                      cVelocityDraws = std::move(m_ngxVelocityDraws),
                      cVelocityStats = m_ngxVelocityStats,
                      cSceneTransformOffset = m_ngxGlobalTransformOffset,
                      cJitterX = m_ngxFrameJitter[0],
                      cJitterY = m_ngxFrameJitter[1],
                      cCameraMatricesValid = m_ngxFrameCameraMatricesValid,
                      cWorldToView = m_ngxFrameWorldToView,
                      cViewToProjection = m_ngxFrameViewToProjection](DxvkContext* ctx) mutable {
      static_cast<RtxContext*>(ctx)->setNgxPassthroughFrameData(cSceneDepth, cColorTarget, cColorMirror, cUpscaleSource, cSubrect,
                                                                cColorSubrectOffset, cOutputTransform,
                                                                std::move(cVelocityDraws), cVelocityStats, cSceneTransformOffset,
                                                                cJitterX, cJitterY,
                                                                cCameraMatricesValid, cWorldToView, cViewToProjection);
    });

    m_ngxVelocityDraws.clear();
  }

  const char* D3D9Rtx::classifyNgxImageForDump(const DxvkImage* image) const {
    if (image == nullptr)
      return "null";
    if (m_ngxSceneColorImage != nullptr && image == m_ngxSceneColorImage.ptr())
      return "sceneColor";
    if (m_ngxSceneDepthImage != nullptr && image == m_ngxSceneDepthImage.ptr())
      return "sceneDepth";
    if (m_ngxFrameBackbufferImage != nullptr && image == m_ngxFrameBackbufferImage.ptr())
      return "backbuffer";
    for (uint32_t i = 0; i < m_ngxSceneColorResolveCount; i++) {
      if (m_ngxSceneColorResolves[i].ptr() == image)
        return "sceneColorResolve";
    }
    return "other";
  }

  // Line cap shared by the per-draw and StretchRect dump paths (heavy frames would
  // otherwise flood the log)
  static constexpr uint32_t kNgxPostChainDumpMaxLinesPerFrame = 160;

  // Frames a donated scene view rect stays usable for the injection decision
  static constexpr uint32_t kNgxSceneViewportStaleFrameLimit = 8;

  // Frames between the bounded frame-shape log lines
  static constexpr uint32_t kNgxFrameShapeLogInterval = 900;

  // A frame reaching the late injection point while the pre-post one is enabled has relocated
  // the injection, which costs the upscaler its history. Attribute it to the requirement that
  // failed; rate limited because what matters is which reason dominates, not each instance.
  void D3D9Rtx::reportNgxPrePostMiss() {
    if (!ngxPrePostInjectionAllowed())
      return;

    // A frame that never identified a scene color has no scene to inject into - a menu, a loading
    // screen, the frames either side of a level transition. Nothing relocated there and no history
    // is lost, so counting it here would misattribute a normal transition as a fault.
    if (m_ngxSceneColorImage == nullptr) {
      return;
    }

    m_ngxPrePostMissCount++;

    if (m_ngxPrePostMissCount != 1 &&
        m_ngxPrePostMissCount - m_ngxPrePostMissLastReportedCount < 60)
      return;

    m_ngxPrePostMissLastReportedCount = m_ngxPrePostMissCount;

    const char* reason =
      !m_ngxDiagSceneColorReady ? "no scene color/view rect was established" :
      !m_ngxDiagSawPostQuad     ? "no pass matched the post-process quad shape" :
      m_ngxPrePostCandidatesThisFrame == 0
                                ? "a qualifying pass ran but sampled neither the scene color nor a resolve of it" :
                                  "the frame's scene color reads ran out before the one the injection was aimed at";

    Logger::info(str::format("[RTX NGX Passthrough] Injection relocated to the late point on ",
                             m_ngxPrePostMissCount, " frames so far; most recently because ", reason,
                             ". DLSS keeps a separate history per injection point, so relocating "
                             "costs the moving content its accumulation."));
  }

  void D3D9Rtx::dumpNgxPostChainDraw(const DrawContext& drawContext) {
    if (m_ngxPostChainDumpLinesThisFrame >= kNgxPostChainDumpMaxLinesPerFrame)
      return;

    D3D9CommonTexture* renderTargetTexture = d3d9State().renderTargets[kRenderTargetIndex] != nullptr
      ? d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture() : nullptr;
    const DxvkImage* renderTargetImage = (renderTargetTexture != nullptr && renderTargetTexture->GetImage() != nullptr)
      ? renderTargetTexture->GetImage().ptr() : nullptr;

    const uint32_t rtSamplerMask = m_parent->GetActiveRTTextures() & m_parent->m_psShaderMasks.samplerMask;

    // Plain scene geometry (into the scene color, no render-target textures sampled) would
    // flood the log; the interesting flow is everything else: post passes, composites, UI
    const bool rtIsSceneColor = m_ngxSceneColorImage != nullptr && renderTargetImage == m_ngxSceneColorImage.ptr();
    if (rtIsSceneColor && rtSamplerMask == 0)
      return;

    m_ngxPostChainDumpLinesThisFrame++;

    std::string line = str::format(
      "[RTX NGX Passthrough][dump] draw=", m_drawCallID,
      (m_rtxInjectTriggered ? " (post-inject)" : ""),
      " rt=", classifyNgxImageForDump(renderTargetImage), "(0x", std::hex, uintptr_t(renderTargetImage), std::dec);

    if (renderTargetImage != nullptr) {
      line += str::format(",", renderTargetImage->info().extent.width, "x", renderTargetImage->info().extent.height,
                          ",fmt=", int(renderTargetImage->info().format));
    }

    const D3DVIEWPORT9& vp = d3d9State().viewport;
    line += str::format(") vp=", vp.Width, "x", vp.Height,
                        " prims=", drawContext.PrimitiveCount,
                        " z=", int(d3d9State().renderStates[D3DRS_ZENABLE]),
                        " zw=", int(d3d9State().renderStates[D3DRS_ZWRITEENABLE]),
                        " st=", int(d3d9State().renderStates[D3DRS_STENCILENABLE]),
                        " ab=", int(d3d9State().renderStates[D3DRS_ALPHABLENDENABLE]));

    for (const uint32_t i : bit::BitMask(rtSamplerMask)) {
      D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[i]);
      const DxvkImage* sampledImage = (texture != nullptr && texture->GetImage() != nullptr) ? texture->GetImage().ptr() : nullptr;

      line += str::format(" s", i, "=", classifyNgxImageForDump(sampledImage), "(0x", std::hex, uintptr_t(sampledImage), std::dec);
      if (sampledImage != nullptr) {
        line += str::format(",", sampledImage->info().extent.width, "x", sampledImage->info().extent.height);
      }
      line += ")";
    }

    Logger::info(line);
  }

  void D3D9Rtx::NotifyStretchRect(const Rc<DxvkImage>& sourceImage, const Rc<DxvkImage>& destImage) {
    if (!m_frameOptions.ngxPassthroughMode)
      return;

    if (m_ngxPostChainDumpFramesLeft > 0 && m_ngxPostChainDumpLinesThisFrame < kNgxPostChainDumpMaxLinesPerFrame) {
      m_ngxPostChainDumpLinesThisFrame++;
      Logger::info(str::format(
        "[RTX NGX Passthrough][dump] StretchRect",
        (m_rtxInjectTriggered ? " (post-inject)" : ""),
        " src=", classifyNgxImageForDump(sourceImage.ptr()), "(0x", std::hex, uintptr_t(sourceImage.ptr()), std::dec,
        (sourceImage != nullptr ? str::format(",", sourceImage->info().extent.width, "x", sourceImage->info().extent.height) : ""),
        ") dst=", classifyNgxImageForDump(destImage.ptr()), "(0x", std::hex, uintptr_t(destImage.ptr()), std::dec,
        (destImage != nullptr ? str::format(",", destImage->info().extent.width, "x", destImage->info().extent.height) : ""), ")"));
    }

    if (!m_frameOptions.ngxPrePostProcess || m_rtxInjectTriggered)
      return;

    if (m_ngxSceneColorImage == nullptr || sourceImage == nullptr || destImage == nullptr ||
        sourceImage.ptr() != m_ngxSceneColorImage.ptr())
      return;

    // Only full-size copies qualify as scene color resolves (UE3's CopyToResolveTarget for
    // a dedicated scene color surface); reduced-size copies are downsamples
    if (destImage->info().extent.width != sourceImage->info().extent.width ||
        destImage->info().extent.height != sourceImage->info().extent.height)
      return;

    for (uint32_t i = 0; i < m_ngxSceneColorResolveCount; i++) {
      if (m_ngxSceneColorResolves[i].ptr() == destImage.ptr())
        return;
    }

    if (m_ngxSceneColorResolveCount < m_ngxSceneColorResolves.size()) {
      m_ngxSceneColorResolves[m_ngxSceneColorResolveCount++] = destImage;
    }
  }

  void D3D9Rtx::NotifyClear(DWORD clearFlags) {
    if (!m_frameOptions.ngxPassthroughMode || m_rtxInjectTriggered)
      return;

    if ((clearFlags & D3DCLEAR_ZBUFFER) == 0)
      return;

    // Only depth clears AFTER world geometry was drawn matter (frame-start clears and
    // SceneCapture probe clears precede any accepted scene camera draw)
    if (!m_ngxSceneViewportValid || m_ngxSceneDepthImage == nullptr)
      return;

    if (d3d9State().depthStencil == nullptr)
      return;

    D3D9CommonTexture* depthStencilTexture = d3d9State().depthStencil->GetCommonTexture();
    if (depthStencilTexture == nullptr || depthStencilTexture->GetImage().ptr() != m_ngxSceneDepthImage.ptr())
      return;

    // The world depth is about to be destroyed (UE3 clears depth ahead of its foreground
    // DPG; with occlusion culling enabled an extra clear precedes that one); snapshot it
    // at the FIRST clear for this frame's motion vector generation. Emitted before the
    // clear itself is recorded, so the copy is ordered ahead of it on the CS timeline.
    // Later clears only advance the counter (the world depth is already gone; the
    // foreground DPG following the last clear stays live in the depth buffer through the
    // injection point).
    m_ngxDepthClearsThisFrame++;

    if (m_ngxFrameClearCount < kNgxFrameShapeSlots) {
      m_ngxFrameClearDraws[m_ngxFrameClearCount++] = m_drawCallID;
    }

    // The injection counts reads from the clear, so a later clear restarts the count
    m_ngxReadsSinceLastClear = 0;

    if (m_ngxDepthClearsThisFrame > 1)
      return;

    m_ngxDepthSnapshotTakenThisFrame = true;

    m_parent->EmitCs([cSceneDepth = m_ngxSceneDepthImage](DxvkContext* ctx) {
      static_cast<RtxContext*>(ctx)->snapshotNgxPassthroughDepth(cSceneDepth);
    });
  }

  bool D3D9Rtx::GetNgxPassthroughViewportJitter(float* pJitterX, float* pJitterY) const {
    if (!m_frameOptions.ngxPassthroughMode || !m_ngxFrameJitterValid || m_rtxInjectTriggered)
      return false;

    if (m_ngxFrameJitter[0] == 0.0f && m_ngxFrameJitter[1] == 0.0f)
      return false;

    if (m_ngxSceneColorImage == nullptr || d3d9State().renderTargets[kRenderTargetIndex] == nullptr)
      return false;

    D3D9CommonTexture* renderTargetTexture = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();

    bool jitterThisDraw = renderTargetTexture != nullptr &&
                          renderTargetTexture->GetImage().ptr() == m_ngxSceneColorImage.ptr();

    // Screen space passes that render into other full-size targets while testing against
    // the scene depth-stencil (dynamic shadow projections and their stencil marking into
    // the light attenuation buffer, distortion accumulation) must shift with the scene:
    // their output is consumed at scene-aligned coordinates and their stencil/depth tests
    // run against the jittered scene depth. Reduced-size targets (bloom, AO) and the UI/
    // post chain (no scene depth bound) stay untouched.
    if (!jitterThisDraw &&
        m_ngxSceneDepthImage != nullptr &&
        renderTargetTexture != nullptr && renderTargetTexture->GetImage() != nullptr &&
        d3d9State().depthStencil != nullptr) {
      D3D9CommonTexture* depthStencilTexture = d3d9State().depthStencil->GetCommonTexture();

      if (depthStencilTexture != nullptr && depthStencilTexture->GetImage() != nullptr &&
          depthStencilTexture->GetImage().ptr() == m_ngxSceneDepthImage.ptr()) {
        const VkExtent3D& rtExtent = renderTargetTexture->GetImage()->info().extent;
        const VkExtent3D& sceneExtent = m_ngxSceneColorImage->info().extent;

        jitterThisDraw = rtExtent.width == sceneExtent.width && rtExtent.height == sceneExtent.height;
      }
    }

    if (!jitterThisDraw)
      return false;

    *pJitterX = m_ngxFrameJitter[0];
    *pJitterY = m_ngxFrameJitter[1];
    return true;
  }

  float D3D9Rtx::GetNgxPassthroughSamplerLodBias() const {
    if (!m_frameOptions.ngxPassthroughMode || m_rtxInjectTriggered)
      return 0.0f;

    if (!m_ngxSceneViewportValid || m_ngxSceneColorImage == nullptr || !m_activePresentParams.has_value())
      return 0.0f;

    // Scene-color draws only: the same reduced-resolution content DLSS upscales. UI and
    // post passes (different targets, or post-injection) stay unbiased.
    if (d3d9State().renderTargets[kRenderTargetIndex] == nullptr)
      return 0.0f;

    D3D9CommonTexture* renderTargetTexture = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
    if (renderTargetTexture == nullptr || renderTargetTexture->GetImage().ptr() != m_ngxSceneColorImage.ptr())
      return 0.0f;

    // Meaningfully reduced in both dimensions = the Super Resolution subrect configuration
    // (the same 97% test the stretch interception uses); DLAA yields log2(1) = 0 naturally
    const uint32_t backBufferWidth = m_activePresentParams->BackBufferWidth;
    const uint32_t backBufferHeight = m_activePresentParams->BackBufferHeight;

    if (backBufferWidth == 0 || backBufferHeight == 0 ||
        uint64_t(m_ngxSceneViewport.Width) * 100 > uint64_t(backBufferWidth) * 97 ||
        uint64_t(m_ngxSceneViewport.Height) * 100 > uint64_t(backBufferHeight) * 97)
      return 0.0f;

    // The standard DLSS integration bias: mip selection matches the upscaled output's
    // texel density instead of the reduced render resolution
    const float bias = std::log2(float(m_ngxSceneViewport.Width) / float(backBufferWidth));
    return std::clamp(bias, -4.0f, 0.0f);
  }

  D3D9Rtx::NgxSpsbCtabReg D3D9Rtx::scanNgxSpsbCtabReg(const std::vector<uint8_t>& bytecode) const {
    NgxSpsbCtabReg result;

    try {
      if (bytecode.size() < sizeof(uint32_t) || (bytecode.size() % sizeof(uint32_t)) != 0)
        return result;

      const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
      const uint32_t headerToken = tokens[0];
      const uint32_t headerTypeMask = headerToken & 0xffff0000u;

      DxsoProgramType programType;
      if (headerTypeMask == 0xffff0000u)
        programType = DxsoProgramTypes::PixelShader;
      else if (headerTypeMask == 0xfffe0000u)
        programType = DxsoProgramTypes::VertexShader;
      else
        return result;

      const uint32_t majorVersion = (headerToken >> 8) & 0xffu;
      const uint32_t minorVersion = headerToken & 0xffu;
      DxsoProgramInfo programInfo { programType, minorVersion, majorVersion };

      DxsoDecodeContext decoder(programInfo);
      DxsoCodeIter iter(tokens + 1);

      while (decoder.decodeInstruction(iter)) {
        if (decoder.getCtabInfo().m_size != 0)
          break;
      }

      const DxsoCtab& ctab = decoder.getCtabInfo();
      if (ctab.m_size == 0 || ctab.m_constantData.empty())
        return result;

      for (const DxsoCtab::Constant& c : ctab.m_constantData) {
        if (c.registerCount < 1)
          continue;

        std::string name;
        name.reserve(c.name.size());
        for (const char ch : c.name)
          name.push_back(char(std::tolower(static_cast<unsigned char>(ch))));

        if (name.find("screenpositionscalebias") != std::string::npos) {
          result.present = true;
          result.reg = c.registerIndex;
          break;
        }
      }
    } catch (...) {
      return result;
    }

    return result;
  }

  void D3D9Rtx::PatchNgxScreenPositionScaleBias(DxsoProgramType stage, void* floatConstants, uint32_t floatConstantCount) const {
    if (!m_ngxSpsbPatchActive)
      return;

    const NgxSpsbCtabReg& patch = stage == DxsoProgramTypes::VertexShader ? m_ngxSpsbPatchVs : m_ngxSpsbPatchPs;
    if (!patch.present || patch.reg >= floatConstantCount)
      return;

    // UV.x = clip.x * SPSB.x + SPSB.w; UV.y = clip.y * SPSB.y + SPSB.z (UE3 samples with
    // ".xy * SPSB.xy + SPSB.wz"): shift the bias so lookups follow the jittered content
    float* constant = reinterpret_cast<float*>(floatConstants) + size_t(patch.reg) * 4;
    constant[3] += m_ngxSpsbPatchAdd[0];
    constant[2] += m_ngxSpsbPatchAdd[1];
  }

  // Detects the first UI-classified draw on the backbuffer after a pre-post-process
  // injection and captures the backbuffer as this frame's HUD-less frame generation input
  // (the game's post chain has written its final output by then, the UI has not). Uses the
  // same UI classification as the late injection trigger.
  void D3D9Rtx::maybeCaptureNgxHudless(const DrawContext& drawContext) {
    if (m_ngxHudlessCapturedThisFrame || !m_frameOptions.ngxDlfgHudless ||
        m_ngxFrameBackbufferImage == nullptr) {
      return;
    }

    if (drawContext.PrimitiveCount == 0 || d3d9State().renderTargets[kRenderTargetIndex] == nullptr) {
      return;
    }

    D3D9CommonTexture* renderTargetTexture = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
    if (renderTargetTexture == nullptr || renderTargetTexture->GetImage().ptr() != m_ngxFrameBackbufferImage.ptr()) {
      return;
    }

    // Per-draw state the UI classification depends on (the post-injection path skips the
    // regular per-draw setup)
    m_boundTextureSnapshotValid = false;

    m_currentUe3VertexFactory = Ue3VertexFactoryType::Unknown;
    if (m_frameOptions.ue3EngineMode && d3d9State().vertexDecl != nullptr) {
      const auto& elements = d3d9State().vertexDecl->GetElements();
      XXH64_hash_t declKey = XXH3_64bits(elements.data(), elements.size() * sizeof(D3DVERTEXELEMENT9));
      auto it = m_ue3VertexFactoryCache.find(declKey);
      if (it != m_ue3VertexFactoryCache.end()) {
        m_currentUe3VertexFactory = it->second;
      } else {
        m_currentUe3VertexFactory = classifyUe3VertexFactory(elements);
        m_ue3VertexFactoryCache.emplace(declKey, m_currentUe3VertexFactory);
      }
    }

    const bool isUiDraw =
      classifyUe3Pass(drawContext) == Ue3PassType::UiComposite ||
      isRenderingUI() ||
      (m_frameOptions.preTransformedVerticesIsUI &&
       d3d9State().vertexDecl != nullptr &&
       d3d9State().vertexDecl->TestFlag(D3D9VertexDeclFlag::HasPositionT));

    if (!isUiDraw) {
      return;
    }

    m_ngxHudlessCapturedThisFrame = true;

    m_parent->EmitCs([cBackbuffer = m_ngxFrameBackbufferImage](DxvkContext* ctx) {
      static_cast<RtxContext*>(ctx)->captureNgxPassthroughHudless(cBackbuffer);
    });
  }

  // Positions the next frame's injection from the shape this one turned out to have, and logs that
  // shape a bounded number of times so a placement can be checked in a title whose frame shape has
  // not been seen before. Called once per frame, before the shape is cleared.
  void D3D9Rtx::updateNgxPrePostInjectionAim() {
    if (m_ngxFrameShapeLogsRemaining > 0 && m_ngxFrameReadCount > 0 &&
        (m_ue3FrameCounter % kNgxFrameShapeLogInterval) == 0) {
      m_ngxFrameShapeLogsRemaining--;

      std::string clears;
      for (uint32_t i = 0; i < m_ngxFrameClearCount; i++) {
        clears += (i != 0 ? ", " : "") + std::to_string(m_ngxFrameClearDraws[i]);
      }

      std::string reads;
      for (uint32_t i = 0; i < m_ngxFrameReadCount; i++) {
        reads += (i != 0 ? ", " : "") + std::to_string(m_ngxFrameReadDraws[i]);
      }

      // A correct placement has injectedAtDraw equal to the first read past lastDepthWritingDraw
      Logger::info(str::format(
        "[RTX NGX Passthrough][frame shape] frame=", m_ue3FrameCounter,
        " sceneDepthClearsAtDraw={", clears, "}",
        " lastDepthWritingDraw=", m_ngxFrameLastGeometryDraw,
        " sceneColorReadsAtDraw={", reads, "}",
        " injectedAtDraw=", m_ngxFrameInjectionDraw,
        " firstBackbufferDraw=", m_ngxFrameFirstBackbufferDraw));
    }

    if (m_ngxFrameLastGeometryDraw != 0 && m_ngxFrameReadCount != 0) {
      uint32_t readsInsideScene = 0;
      while (readsInsideScene < m_ngxFrameReadCount &&
             m_ngxFrameReadDraws[readsInsideScene] < m_ngxFrameLastGeometryDraw) {
        readsInsideScene++;
      }

      // Both counts are adopted only when two frames running agree. A title may alternate between
      // frames whose scene holds two scene color reads and frames whose scene holds three, and
      // correcting after every frame that proves the aim wrong is wrong on both: aim at the later
      // read and the shorter frame never reaches it, aim at the earlier one and the longer frame
      // injects inside its scene. Requiring agreement declines to chase the alternation.
      if (readsInsideScene == m_ngxPrevReadsInsideScene &&
          readsInsideScene != m_ngxPrePostReadsInsideScene) {
        m_ngxPrePostReadsInsideScene = readsInsideScene;
      }
      m_ngxPrevReadsInsideScene = readsInsideScene;

      if (m_ngxFrameClearCount > 0) {
        const uint32_t lastClearDraw = m_ngxFrameClearDraws[m_ngxFrameClearCount - 1];
        uint32_t readsAfterClearInsideScene = 0;

        for (uint32_t i = 0; i < m_ngxFrameReadCount; i++) {
          if (m_ngxFrameReadDraws[i] > lastClearDraw &&
              m_ngxFrameReadDraws[i] < m_ngxFrameLastGeometryDraw) {
            readsAfterClearInsideScene++;
          }
        }

        if (readsAfterClearInsideScene == m_ngxPrevReadsAfterClearInsideScene &&
            readsAfterClearInsideScene != m_ngxReadsAfterClearInsideScene) {
          Logger::info(str::format(
            "[RTX NGX Passthrough] Pre-post-process injection aimed at scene color read ",
            readsAfterClearInsideScene + 1, " after the scene depth clear: ",
            readsAfterClearInsideScene, " of them fall inside the scene."));

          m_ngxReadsAfterClearInsideScene = readsAfterClearInsideScene;
        }
        m_ngxPrevReadsAfterClearInsideScene = readsAfterClearInsideScene;
      }
    }

    m_ngxReadsSinceLastClear = 0;
    m_ngxFrameClearCount = 0;
    m_ngxFrameReadCount = 0;
    m_ngxFrameLastGeometryDraw = 0;
    m_ngxFrameInjectionDraw = 0;
    m_ngxFrameFirstBackbufferDraw = 0;
    m_ngxPrePostCandidatesThisFrame = 0;
  }

  bool D3D9Rtx::ngxPrePostInjectionAllowed() const {
    if (!m_frameOptions.ngxPrePostProcess)
      return false;

    const int debugVis = m_frameOptions.ngxDebugVisualization;
    return debugVis == 0 ||
           debugVis == int(dxvk::RtxNgxPassthrough::DebugVisualization::SceneColor);
  }

  Rc<DxvkImage> D3D9Rtx::matchNgxSceneColorSample(const DxvkImage* sampledImage) const {
    if (sampledImage == nullptr || m_ngxSceneColorImage == nullptr)
      return nullptr;

    if (sampledImage == m_ngxSceneColorImage.ptr())
      return m_ngxSceneColorImage;

    for (uint32_t r = 0; r < m_ngxSceneColorResolveCount; r++) {
      if (m_ngxSceneColorResolves[r].ptr() == sampledImage)
        return m_ngxSceneColorResolves[r];
    }

    return nullptr;
  }

  void D3D9Rtx::recordNgxPrePostSceneColorRead() {
    m_ngxPrePostCandidatesThisFrame++;
    m_ngxReadsSinceLastClear++;

    if (m_ngxFrameReadCount < kNgxFrameShapeSlots)
      m_ngxFrameReadDraws[m_ngxFrameReadCount++] = m_drawCallID;
  }

  void D3D9Rtx::engageNgxPrePostInjection(const Rc<DxvkImage>& matchedTarget) {
    m_ngxColorTargetImage = matchedTarget;
    // When the consumed image is a resolve copy, mirror the DLSS output into the scene color
    // surface as well: later post passes may re-resolve from it (UE3 scene color resolves are
    // surface -> texture copies)
    m_ngxColorMirrorImage = (matchedTarget.ptr() != m_ngxSceneColorImage.ptr()) ? m_ngxSceneColorImage : nullptr;
    m_ngxSubrect.offset = { int32_t(m_ngxSceneViewport.X), int32_t(m_ngxSceneViewport.Y) };
    m_ngxSubrect.extent = { m_ngxSceneViewport.Width, m_ngxSceneViewport.Height };
    m_ngxColorSubrectOffset = m_ngxSubrect.offset;
    m_ngxFrameInjectionDraw = m_drawCallID;
  }

  Rc<DxvkImage> D3D9Rtx::findNgxPrePostSceneColorFromSamplers(bool applyOrdinalGate, bool recordRead) {
    const uint32_t rtSamplerMask = m_parent->GetActiveRTTextures() & m_parent->m_psShaderMasks.samplerMask;

    for (const uint32_t i : bit::BitMask(rtSamplerMask)) {
      D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[i]);
      if (texture == nullptr || texture->GetImage() == nullptr)
        continue;

      const Rc<DxvkImage> matchedTarget = matchNgxSceneColorSample(texture->GetImage().ptr());
      if (matchedTarget == nullptr)
        continue;

      if (recordRead)
        recordNgxPrePostSceneColorRead();

      if (applyOrdinalGate) {
        const bool haveClearThisFrame = m_ngxFrameClearCount > 0;

        const bool atInjectionPoint = haveClearThisFrame
          ? m_ngxReadsSinceLastClear >= m_ngxReadsAfterClearInsideScene + 1
          : m_ngxPrePostCandidatesThisFrame >= m_ngxPrePostReadsInsideScene + 1;

        if (!atInjectionPoint)
          break;
      }

      return matchedTarget;
    }

    return nullptr;
  }

  bool D3D9Rtx::ngxSceneViewportIsFullSize(const D3DVIEWPORT9& sceneViewport,
                                           uint32_t backBufferWidth, uint32_t backBufferHeight) {
    return backBufferWidth != 0 && backBufferHeight != 0 &&
           uint64_t(sceneViewport.Width) * 100 >= uint64_t(backBufferWidth) * 97 &&
           uint64_t(sceneViewport.Height) * 100 >= uint64_t(backBufferHeight) * 97;
  }

  bool D3D9Rtx::ngxSceneViewportIsSubrect(const D3DVIEWPORT9& sceneViewport,
                                          uint32_t backBufferWidth, uint32_t backBufferHeight) {
    return backBufferWidth != 0 && backBufferHeight != 0 &&
           uint64_t(sceneViewport.Width) * 100 <= uint64_t(backBufferWidth) * 97 &&
           uint64_t(sceneViewport.Height) * 100 <= uint64_t(backBufferHeight) * 97;
  }

  bool D3D9Rtx::ngxShaderIsFinishRenderViewTargetGamma(const Ue3ShaderFeatureInfo& psInfo) {
    return psInfo.gammaInverseReg != Ue3ShaderFeatureInfo::kNgxNoRegister &&
           psInfo.hasSceneColorSampler &&
           !psInfo.hasToneMapConstants &&
           !psInfo.hasExposureOrToneSampler;
  }

  void D3D9Rtx::countNgxPostInjectionSceneColorConsumer(const DrawContext& drawContext) {
    // The injection ordinal is only useful if the count it comes from covers the whole frame, and
    // the passes that decide whether this frame's injection was the last one are precisely the ones
    // that come after it. Same shape test as the pre-injection path, deliberately: a mismatch
    // between the two would bias the count and walk the injection point away from the end.
    if (!ngxPrePostInjectionAllowed() || m_ngxSceneColorImage == nullptr ||
        m_ngxBackbufferDrawSeenThisFrame ||
        !m_activePresentParams.has_value() || d3d9State().renderTargets[kRenderTargetIndex] == nullptr) {
      return;
    }

    const uint32_t backBufferWidth = m_activePresentParams->BackBufferWidth;
    const uint32_t backBufferHeight = m_activePresentParams->BackBufferHeight;

    // These have to match the pre-injection test exactly. Any difference makes the frame's read
    // total depend on where the injection landed, and since that total aims the next frame, the
    // aim then feeds back into itself and never settles.
    const bool depthTestDisabled = d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_FALSE ||
                                   d3d9State().renderStates[D3DRS_ZFUNC] == D3DCMP_ALWAYS;

    if (drawContext.PrimitiveCount > 4 ||
        !depthTestDisabled ||
        d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE ||
        d3d9State().renderStates[D3DRS_STENCILENABLE] != FALSE ||
        d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] != FALSE ||
        d3d9State().viewport.Width + 1 < backBufferWidth ||
        d3d9State().viewport.Height + 1 < backBufferHeight) {
      return;
    }

    D3D9CommonTexture* renderTargetTexture = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
    if (renderTargetTexture != nullptr &&
        (renderTargetTexture->GetImage().ptr() == m_ngxSceneColorImage.ptr() ||
         (m_ngxFrameBackbufferImage != nullptr &&
          renderTargetTexture->GetImage().ptr() == m_ngxFrameBackbufferImage.ptr()))) {
      return;
    }

    const uint32_t rtSamplerMask = m_parent->GetActiveRTTextures() & m_parent->m_psShaderMasks.samplerMask;

    for (const uint32_t i : bit::BitMask(rtSamplerMask)) {
      D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[i]);
      if (texture == nullptr || texture->GetImage() == nullptr) {
        continue;
      }

      const DxvkImage* sampledImage = texture->GetImage().ptr();
      if (matchNgxSceneColorSample(sampledImage) != nullptr) {
        m_ngxPrePostCandidatesThisFrame++;

        if (m_ngxFrameReadCount < kNgxFrameShapeSlots) {
          m_ngxFrameReadDraws[m_ngxFrameReadCount++] = m_drawCallID;
        }
        break;
      }
    }
  }

  PrepareDrawFlags D3D9Rtx::prepareDrawForNgxPassthrough(const DrawContext& drawContext) {
    ScopedCpuProfileZone();

    // Per-frame draw index for the diagnostics (nothing else advances it in this mode)
    m_drawCallID++;

    if (unlikely(m_ngxPostChainDumpFramesLeft > 0)) {
      dumpNgxPostChainDraw(drawContext);
    }

    // Super Resolution texture LOD bias transitions: the bias is folded into the sampler
    // keys at bind time (see GetNgxPassthroughSamplerLodBias), but samplers are only
    // (re)created when their stage is dirtied - so whenever this draw's effective bias
    // differs from the last draw's (scene <-> UI target switches, the injection trigger),
    // every stage must re-bind or draws would sample with the previous scope's bias.
    {
      const float samplerLodBias = GetNgxPassthroughSamplerLodBias();
      if (samplerLodBias != m_ngxAppliedSamplerLodBias) {
        m_ngxAppliedSamplerLodBias = samplerLodBias;
        m_parent->m_dirtySamplerStates = (1u << uint32_t(d3d9State().samplerStates.size())) - 1u;
      }
    }

    // The same hazard applies to the viewport jitter. Whether a draw is jittered depends on its
    // render target and depth-stencil (see GetNgxPassthroughViewportJitter), but the Vulkan
    // viewport is only rebuilt when D3D9 marks the viewport dirty, which binding a different
    // target need not do - so a draw can inherit the previous draw's jitter state and land on a
    // different sub-pixel grid to the rest of the frame. Force the rebind whenever it flips.
    {
      float jitterX = 0.0f;
      float jitterY = 0.0f;
      const bool jitterThisDraw = GetNgxPassthroughViewportJitter(&jitterX, &jitterY);

      if (jitterThisDraw != m_ngxViewportJitterApplied ||
          (jitterThisDraw && (jitterX != m_ngxAppliedViewportJitter[0] ||
                              jitterY != m_ngxAppliedViewportJitter[1]))) {
        m_ngxViewportJitterApplied = jitterThisDraw;
        m_ngxAppliedViewportJitter[0] = jitterX;
        m_ngxAppliedViewportJitter[1] = jitterY;
        m_parent->m_flags.set(D3D9DeviceFlag::DirtyViewportScissor);
      }
    }

    // Where the scene ends. Deliberately independent of camera reconstruction: the last depth
    // priority group is where the first person mesh and held weapon are, they are skinned, and
    // skinned draws are exactly the ones whose constants the camera extraction rejects - reading
    // this from that path would make the frame look as if its last group never drew anything.
    if (drawContext.PrimitiveCount > 4 &&
        d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE &&
        d3d9State().depthStencil != nullptr) {
      m_ngxFrameLastGeometryDraw = m_drawCallID;
    }

    // Once the game has bound the backbuffer it is finished with the scene and its post chain, and
    // what follows is the UI being composited. Tracked across the whole frame, including past the
    // injection, because the shape that positions the next frame spans the whole frame.
    if (!m_ngxBackbufferDrawSeenThisFrame && m_ngxFrameBackbufferImage != nullptr &&
        d3d9State().renderTargets[kRenderTargetIndex] != nullptr) {
      D3D9CommonTexture* drawTarget = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();

      if (drawTarget != nullptr && drawTarget->GetImage().ptr() == m_ngxFrameBackbufferImage.ptr()) {
        m_ngxBackbufferDrawSeenThisFrame = true;
        m_ngxFrameFirstBackbufferDraw = m_drawCallID;
      }
    }

    // Every draw executes as plain rasterization in this mode. The remaining per-draw work:
    // UE3 camera extraction, scene target identification, and the thin injection trigger.
    if (m_rtxInjectTriggered) {
      countNgxPostInjectionSceneColorConsumer(drawContext);
      maybeCaptureNgxHudless(drawContext);
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    m_boundTextureSnapshotValid = false;

    // Decide this frame's sub-pixel jitter before the first draw that may consume it
    if (!m_ngxFrameJitterValid) {
      if (unlikely(!m_frameOptions.valid)) {
        refreshFrameOptionCache();
      }

      // Drive ScreenPercentage before scene draws so the game can apply the reduced render
      // resolution this frame (EndFrame alone is one frame late for the first scene pass).
      applyNgxPassthroughScreenPercentage();

      m_ngxFrameJitterValid = true;
      m_ngxFrameJitter[0] = 0.0f;
      m_ngxFrameJitter[1] = 0.0f;

      // Temporal upscaler selected and usable on this system
      const bool temporalUpscalerUsable = RtxNgxPassthrough::needsViewportJitter(m_parent->GetDXVKDevice().ptr());

      if (m_frameOptions.ngxPassthroughJitter && temporalUpscalerUsable) {
        // The render height the game is about to draw at, so the phase count matches the
        // upscale ratio DLSS will see this frame.
        uint32_t renderHeight = 0;
        uint32_t displayHeight = 0;
        if (m_activePresentParams.has_value()) {
          displayHeight = m_activePresentParams->BackBufferHeight;
          if (m_ngxGameScreenPercentage > 0.0f && m_ngxGameScreenPercentage <= 100.0f) {
            renderHeight = uint32_t(float(displayHeight) * m_ngxGameScreenPercentage / 100.0f);
          } else {
            renderHeight = displayHeight;
          }
        }

        const uint32_t jitterSequenceLength = RtxNgxPassthrough::viewportJitterSequenceLength(
          m_parent->GetDXVKDevice().ptr(), renderHeight, displayHeight);

        // An artefact with the period of this loop is the signature of a phase that recurs too
        // rarely to accumulate, so the length has to be readable rather than re-derived by hand
        if (jitterSequenceLength != m_ngxLoggedJitterSequenceLength) {
          m_ngxLoggedJitterSequenceLength = jitterSequenceLength;
          Logger::info(str::format("[RTX NGX Passthrough] Viewport jitter loop: ", jitterSequenceLength,
                                   " phases at ", renderHeight, "p render / ", displayHeight, "p display."));
        }
        const Vector2 jitter = calculateHaltonJitter(m_parent->GetDXVKDevice()->getCurrentFrameId(),
                                                     jitterSequenceLength);
        m_ngxFrameJitter[0] = jitter.x;
        m_ngxFrameJitter[1] = jitter.y;
      }

      // The viewport carries the previous frame's jitter offset until rebound
      m_parent->m_flags.set(D3D9DeviceFlag::DirtyViewportScissor);

      // The GPU constant buffers may carry the previous frame's ScreenPositionScaleBias
      // patch (a different jitter): force one fresh upload per stage this frame
      m_parent->m_consts[DxsoProgramTypes::VertexShader].dirty = true;
      m_parent->m_consts[DxsoProgramTypes::PixelShader].dirty = true;
      m_ngxSpsbLastVsReg = -1;
      m_ngxSpsbLastPsReg = -1;

      // Resolve this frame's backbuffer image (backbuffers rotate every present) for the
      // scene-end trigger: description-based primary checks also match offscreen buffers
      // allocated at backbuffer size (e.g. ME's TdUI compositing targets), which would make
      // the injection run against - and DLSS write into - the wrong image on some frames.
      m_ngxFrameBackbufferImage = nullptr;

      Com<IDirect3DSurface9> backBuffer;
      if (SUCCEEDED(m_parent->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &backBuffer)) && backBuffer != nullptr) {
        if (D3D9CommonTexture* backBufferTexture = static_cast<D3D9Surface*>(backBuffer.ptr())->GetCommonTexture()) {
          m_ngxFrameBackbufferImage = backBufferTexture->GetImage();
        }
      }
    }

    // ScreenPositionScaleBias jitter compensation for this draw's shaders: screen space
    // lookups (shadow projections reading scene depth from the scene color alpha,
    // translucency, distortion, fog) compute UVs from clip-space varyings, which do not
    // follow the viewport jitter; the bias constant is shifted by the jitter at constant
    // upload so those lookups sample the jittered content aligned. Without this, passes
    // whose result depends sharply on the sampled depth (dynamic shadow projections with
    // tight bias) oscillate with the per-frame jitter sign.
    //
    // Decided before any early-out below: the patch must always describe the shaders bound
    // for whatever the device uploads next, including degenerate draws that skip the rest
    // of the per-draw work.
    m_ngxSpsbPatchActive = false;

    if ((m_ngxFrameJitter[0] != 0.0f || m_ngxFrameJitter[1] != 0.0f) && m_ngxSceneColorImage != nullptr) {
      auto lookupSpsbReg = [&](const D3D9CommonShader* shader) -> NgxSpsbCtabReg {
        if (shader == nullptr)
          return NgxSpsbCtabReg();

        const XXH64_hash_t shaderHash = shader->GetBytecodeHash();
        if (shaderHash == 0)
          return NgxSpsbCtabReg();

        auto it = m_ngxSpsbCtabCache.find(shaderHash);
        if (it == m_ngxSpsbCtabCache.end()) {
          m_ngxSpsbCtabCache.emplace(shaderHash, scanNgxSpsbCtabReg(shader->GetBytecode()));
          it = m_ngxSpsbCtabCache.find(shaderHash);
        }
        return it->second;
      };

      m_ngxSpsbPatchVs = lookupSpsbReg(m_parent->UseProgrammableVS() && d3d9State().vertexShader.ptr() != nullptr
                                         ? d3d9State().vertexShader->GetCommonShader() : nullptr);
      m_ngxSpsbPatchPs = lookupSpsbReg(d3d9State().pixelShader.ptr() != nullptr
                                         ? d3d9State().pixelShader->GetCommonShader() : nullptr);

      if (m_ngxSpsbPatchVs.present || m_ngxSpsbPatchPs.present) {
        const VkExtent3D& sceneExtent = m_ngxSceneColorImage->info().extent;

        if (sceneExtent.width != 0 && sceneExtent.height != 0) {
          m_ngxSpsbPatchAdd[0] = m_ngxFrameJitter[0] / float(sceneExtent.width);
          m_ngxSpsbPatchAdd[1] = m_ngxFrameJitter[1] / float(sceneExtent.height);
          m_ngxSpsbPatchActive = true;
        }
      }
    }

    // Force a fresh constant upload when the patch register moved (shader switch without a
    // constant change): the buffered copy carries the previous shader's patch layout
    {
      const int32_t vsReg = (m_ngxSpsbPatchActive && m_ngxSpsbPatchVs.present) ? int32_t(m_ngxSpsbPatchVs.reg) : -1;
      const int32_t psReg = (m_ngxSpsbPatchActive && m_ngxSpsbPatchPs.present) ? int32_t(m_ngxSpsbPatchPs.reg) : -1;

      if (vsReg != m_ngxSpsbLastVsReg) {
        m_parent->m_consts[DxsoProgramTypes::VertexShader].dirty = true;
        m_ngxSpsbLastVsReg = vsReg;
      }
      if (psReg != m_ngxSpsbLastPsReg) {
        m_parent->m_consts[DxsoProgramTypes::PixelShader].dirty = true;
        m_ngxSpsbLastPsReg = psReg;
      }
    }

    if (drawContext.PrimitiveCount == 0 || d3d9State().renderTargets[kRenderTargetIndex] == nullptr) {
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    // UE3 vertex factory classification (cached by vertex declaration) feeds classifyUe3Pass
    m_currentUe3VertexFactory = Ue3VertexFactoryType::Unknown;
    m_currentUe3PassType = Ue3PassType::Unknown;
    if (m_frameOptions.ue3EngineMode && d3d9State().vertexDecl != nullptr) {
      const auto& elements = d3d9State().vertexDecl->GetElements();
      XXH64_hash_t declKey = XXH3_64bits(elements.data(), elements.size() * sizeof(D3DVERTEXELEMENT9));
      auto it = m_ue3VertexFactoryCache.find(declKey);
      if (it != m_ue3VertexFactoryCache.end()) {
        m_currentUe3VertexFactory = it->second;
      } else {
        m_currentUe3VertexFactory = classifyUe3VertexFactory(elements);
        m_ue3VertexFactoryCache.emplace(declKey, m_currentUe3VertexFactory);
      }
    }

    // Camera extraction from UE3 reserved shader constants (CTAB-verified draws only)
    if (m_parent->UseProgrammableVS() && d3d9State().vertexShader.ptr() != nullptr) {
      tryNgxPassthroughCameraCapture();
    }

    // Dynamic object capture for the velocity raster pass
    tryCaptureNgxVelocityDraw(drawContext);

    // Scene end detection: the first UI draw on the primary render target ends the scene and
    // triggers the thin injection (DLSS on the game's post-processed output, pre-UI). When
    // the game renders its scene into a reduced ScreenPercentage subrect, the engine's final
    // bilinear stretch onto the primary target ends the scene instead: that draw is
    // suppressed and DLSS performs the upscale (true Super Resolution).
    bool triggerInjection = false;
    bool suppressDraw = false;

    if (m_activePresentParams.has_value()) {
      const uint32_t backBufferWidth = m_activePresentParams->BackBufferWidth;
      const uint32_t backBufferHeight = m_activePresentParams->BackBufferHeight;

      D3D9CommonTexture* renderTargetTexture = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();

      // Pre-post-process injection point: the first fullscreen composite quad that samples
      // the scene color (the render surface directly, or a same-size resolve copy of it -
      // UE3's D3D9 RHI gives the scene color a dedicated render surface and resolves it
      // into a texture for sampling) while rendering into a different target is the start
      // of the game's post-process chain (UE3 renders all scene DPGs - world and foreground
      // - before its post chain reads scene color). DLSS runs on the linear scene color
      // here and writes the result back to the image the pass consumes, so bloom/
      // tonemapping/dynamic contrast operate on the anti-aliased, unjittered image the way
      // a native engine integration would.
      //
      // The quad requirements (depth disabled, no z-write, no stencil, no blending, few
      // primitives, fullscreen viewport) exist because mid-scene lighting passes also
      // sample the scene color resolve into other targets: dynamic shadow projections read
      // the scene depth from the resolve alpha while rendering into the light attenuation
      // buffer or modulating the scene color as fullscreen-viewport frustum geometry, with
      // depth testing frequently disabled - but always stencil-masked, blended, and more
      // than 4 primitives (observed in ME via the post-chain dump). Triggering on one of
      // those would run DLSS before the scene is complete.
      //
      // Only engages for a full-size scene (a ScreenPercentage subrect is handled by the
      // stretch-replacement path below).
      const bool likelyPostProcessQuad =
        drawContext.PrimitiveCount <= 4 &&
        (d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_FALSE ||
         d3d9State().renderStates[D3DRS_ZFUNC] == D3DCMP_ALWAYS) &&
        d3d9State().renderStates[D3DRS_ZWRITEENABLE] == FALSE &&
        d3d9State().renderStates[D3DRS_STENCILENABLE] == FALSE &&
        d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] == FALSE &&
        d3d9State().viewport.Width + 1 >= backBufferWidth &&
        d3d9State().viewport.Height + 1 >= backBufferHeight;

      // The view rect comes from the frame's scene draws, and a frame whose geometry all failed
      // camera reconstruction donates none. Losing the injection over a rectangle that had not
      // changed sends the frame to the late point, so a recently donated rect stays usable.
      const bool sceneViewportUsable =
        m_ngxSceneViewportValid ||
        (m_ngxSceneViewportLastValidFrame != 0 &&
         m_ue3FrameCounter - m_ngxSceneViewportLastValidFrame <= kNgxSceneViewportStaleFrameLimit);

      // Which requirements the frame met, so one that ends up at the late point can say why
      m_ngxDiagSawPostQuad |= likelyPostProcessQuad;
      m_ngxDiagSceneColorReady |= (m_ngxSceneColorImage != nullptr && sceneViewportUsable);

      // A pass with the backbuffer bound is the final composite or the UI being drawn over it, and
      // the pre-post point is mid-chain by definition. Letting those count walks the injection past
      // the end of the post chain onto a pass with UI in it, which the upscaler then resolves and
      // softens; that territory belongs to the late injection point.
      const bool targetsBackbuffer =
        renderTargetTexture != nullptr && m_ngxFrameBackbufferImage != nullptr &&
        renderTargetTexture->GetImage().ptr() == m_ngxFrameBackbufferImage.ptr();

      const bool sceneIsFullRes =
        m_ngxSceneColorImage != nullptr && sceneViewportUsable &&
        ngxSceneViewportIsFullSize(m_ngxSceneViewport, backBufferWidth, backBufferHeight);

      const bool candidateShape =
        ngxPrePostInjectionAllowed() &&
        likelyPostProcessQuad &&
        !targetsBackbuffer && !m_ngxBackbufferDrawSeenThisFrame &&
        sceneIsFullRes &&
        (renderTargetTexture == nullptr || renderTargetTexture->GetImage().ptr() != m_ngxSceneColorImage.ptr());

      if (candidateShape) {
        const Rc<DxvkImage> matchedTarget = findNgxPrePostSceneColorFromSamplers(true, true);

        if (matchedTarget != nullptr) {
          engageNgxPrePostInjection(matchedTarget);
          triggerInjection = true;

          ONCE(Logger::info(str::format("[RTX NGX Passthrough] Pre-post-process injection engaged: DLSS runs on the ",
                                        (m_ngxColorMirrorImage != nullptr ? "resolved scene color" : "scene color"),
                                        " before the game's post-process chain.")));
        }
      }

      // The trigger requires the actual backbuffer image, not just a backbuffer-sized
      // description: offscreen composition buffers (e.g. ME's TdUI targets) share the
      // description and would otherwise receive the injection on some frames, making DLSS
      // alternate between the real backbuffer and an offscreen image
      bool isPrimary;

      if (m_ngxFrameBackbufferImage != nullptr) {
        isPrimary = renderTargetTexture != nullptr &&
                    renderTargetTexture->GetImage().ptr() == m_ngxFrameBackbufferImage.ptr();
      } else {
        isPrimary = s_isDxvkResolutionEnvVarSet ||
          isRenderTargetPrimary(*m_activePresentParams, renderTargetTexture->Desc());
      }

      if (!triggerInjection && isPrimary) {
        // ScreenPercentage upscale replacement: only engages when this frame's scene provably
        // rendered into a subrect meaningfully smaller than the backbuffer in both dimensions
        // (a reduced height alone would match letterboxed cinematics)
        const bool sceneIsSubrect =
          m_ngxSceneViewportValid &&
          ngxSceneViewportIsSubrect(m_ngxSceneViewport, backBufferWidth, backBufferHeight);

        // The composite has to execute before DLSS can read what it wrote, so the draw that
        // produced it must never be the one that triggers the injection.
        bool recordedRuntimeComposite = false;

        if (sceneIsSubrect && ngxRuntimeOwnsUpscale()) {
          // The engine believes it renders natively, so there is no upscale stretch to replace:
          // its post chain composites the reduced view rect straight into the backbuffer (at the
          // centred position ScaleScreenCoords produced) and stops. Note where that landed; the
          // upscale happens once the scene is finished, in emitNgxPassthroughFrameData.
          //
          // The composite's own viewport is not a usable signature for finding it. UE3 may leave
          // the viewport at full size and size the quad instead, so requiring the viewport to
          // match the scene rect misses the draw entirely in those titles, and the reduced image
          // is then presented unstretched with borders around it. Either shape is accepted.
          const D3DVIEWPORT9& vp = d3d9State().viewport;
          const bool depthTestDisabled = d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_FALSE ||
                                         d3d9State().renderStates[D3DRS_ZFUNC] == D3DCMP_ALWAYS;
          const bool compositeViewportIsSceneSized =
            vp.Width == m_ngxSceneViewport.Width && vp.Height == m_ngxSceneViewport.Height;
          const bool viewportIsSceneOrFull =
            compositeViewportIsSceneSized ||
            (vp.Width + 1 >= backBufferWidth && vp.Height + 1 >= backBufferHeight);

          const bool likelySubrectComposite =
            drawContext.PrimitiveCount <= 4 &&
            depthTestDisabled &&
            d3d9State().renderStates[D3DRS_ZWRITEENABLE] == FALSE &&
            viewportIsSceneOrFull &&
            (m_parent->GetActiveRTTextures() & m_parent->m_psShaderMasks.samplerMask) != 0;

          if (likelySubrectComposite) {
            // Where the scene landed in the backbuffer, which is not where it landed in scene
            // color. A composite that sets a reduced viewport states its position directly; one
            // that leaves the viewport full size does not, and the scene viewport's own offset
            // cannot stand in for it - that is a scene color space position, and a title may
            // render at its origin while compositing centred. UE3's ScaleScreenCoords centres
            // the reduced rect in the target, so the remaining case is half the difference.
            m_ngxRuntimeUpscaleRect = m_ngxSceneViewport;

            if (compositeViewportIsSceneSized) {
              m_ngxRuntimeUpscaleRect.X = vp.X;
              m_ngxRuntimeUpscaleRect.Y = vp.Y;
            } else {
              m_ngxRuntimeUpscaleRect.X = backBufferWidth > m_ngxSceneViewport.Width
                ? (backBufferWidth - m_ngxSceneViewport.Width) / 2 : 0;
              m_ngxRuntimeUpscaleRect.Y = backBufferHeight > m_ngxSceneViewport.Height
                ? (backBufferHeight - m_ngxSceneViewport.Height) / 2 : 0;
            }

            m_ngxRuntimeUpscaleRectValid = true;
            recordedRuntimeComposite = true;

            ONCE(Logger::info(str::format(
              "[RTX NGX Passthrough] Runtime-owned Super Resolution: the engine composited its ",
              m_ngxRuntimeUpscaleRect.Width, "x", m_ngxRuntimeUpscaleRect.Height, " view rect at (",
              m_ngxRuntimeUpscaleRect.X, ", ", m_ngxRuntimeUpscaleRect.Y,
              ") in the backbuffer, ", (compositeViewportIsSceneSized ? "stated by its viewport" : "centred by ScaleScreenCoords"),
              "; DLSS will upscale it to ", backBufferWidth, "x", backBufferHeight, ".")));
          }
        } else if (sceneIsSubrect) {
          const D3DVIEWPORT9& vp = d3d9State().viewport;
          const bool depthTestDisabled = d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_FALSE ||
                                         d3d9State().renderStates[D3DRS_ZFUNC] == D3DCMP_ALWAYS;
          const bool zWriteEnabled = d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE;

          // The stretch is the first fullscreen-viewport composite quad targeting the primary
          // render target after the scene (the post chain writes to offscreen targets while
          // an upscale is pending)
          const bool likelyUpscaleStretch =
            drawContext.PrimitiveCount <= 4 &&
            depthTestDisabled && !zWriteEnabled &&
            vp.Width + 1 >= backBufferWidth && vp.Height + 1 >= backBufferHeight;

          // Only a plain composite may be stood in for. Some UE3 licensee builds fold the whole
          // post chain into the upscaling draw - depth of field, bloom, the material grade and
          // atmospheric fog on the way out; replacing that discards all of it, so the runtime
          // takes ownership of the upscale instead and lets the engine's own pass composite
          // into the reduced rect untouched.
          bool compositeSafeToReplace = true;
          if (likelyUpscaleStretch && m_parent->UseProgrammablePS() &&
              d3d9State().pixelShader.ptr() != nullptr) {
            const Ue3ShaderFeatureInfo psInfo =
              getUe3ShaderFeatureInfo(d3d9State().pixelShader->GetCommonShader());
            compositeSafeToReplace =
              psInfo.gammaInverseReg != Ue3ShaderFeatureInfo::kNgxNoRegister ||
              (!psInfo.hasToneMapConstants && !psInfo.hasExposureOrToneSampler);

            if (!compositeSafeToReplace && !m_ngxEngineCompositeUnsafe) {
              m_ngxEngineCompositeUnsafe = true;

              // Hand the upscale to the runtime rather than give up on Super Resolution: the
              // engine's pass then composites its reduced view rect into the backbuffer with its
              // whole chain intact, and the runtime upscales that rect. Only when the settings
              // redirects are unavailable - so the runtime cannot own the upscale either - does
              // this leave full resolution as the only correct option.
              m_ngxRuntimeOwnedUpscale = m_ngxGameSettingsRedirectsValid;

              Logger::info(str::format(
                "[RTX NGX Passthrough] The engine's upscaling draw also runs its post-process chain "
                "(DOF, bloom, colour grading), so it cannot be replaced without discarding all of it. ",
                m_ngxRuntimeOwnedUpscale
                  ? "The runtime will upscale the reduced rect the engine composites instead, leaving "
                    "that pass untouched."
                  : "Without the settings redirects the runtime cannot upscale either, so this title "
                    "is limited to full resolution DLAA."));
            }
          }

          if (likelyUpscaleStretch && compositeSafeToReplace) {
            // The stretch source: the largest render-target texture the pixel shader samples
            // that covers the scene subrect
            Rc<DxvkImage> sourceImage;
            const uint32_t rtSamplerMask = m_parent->GetActiveRTTextures() & m_parent->m_psShaderMasks.samplerMask;

            for (const uint32_t i : bit::BitMask(rtSamplerMask)) {
              D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[i]);
              if (texture == nullptr || texture->GetImage() == nullptr) {
                continue;
              }

              const VkExtent3D& sourceExtent = texture->GetImage()->info().extent;
              if (sourceExtent.width < m_ngxSceneViewport.X + m_ngxSceneViewport.Width ||
                  sourceExtent.height < m_ngxSceneViewport.Y + m_ngxSceneViewport.Height) {
                continue;
              }

              if (sourceImage == nullptr ||
                  uint64_t(sourceExtent.width) * sourceExtent.height >
                  uint64_t(sourceImage->info().extent.width) * sourceImage->info().extent.height) {
                sourceImage = texture->GetImage();
              }
            }

            if (sourceImage != nullptr) {
              m_ngxUpscaleSourceImage = sourceImage;
              m_ngxSubrect.offset = { int32_t(m_ngxSceneViewport.X), int32_t(m_ngxSceneViewport.Y) };
              m_ngxSubrect.extent = { m_ngxSceneViewport.Width, m_ngxSceneViewport.Height };
              m_ngxColorSubrectOffset = m_ngxSubrect.offset;
              captureNgxOutputTransform();

              triggerInjection = true;
              suppressDraw = true;

              ONCE(Logger::info(str::format(
                "[RTX NGX Passthrough] ScreenPercentage upscale detected and replaced with DLSS Super Resolution (",
                m_ngxSceneViewport.Width, "x", m_ngxSceneViewport.Height, " -> ",
                backBufferWidth, "x", backBufferHeight, ").")));
            }
          }
        }

        if (!triggerInjection && !recordedRuntimeComposite && sceneIsFullRes &&
            ngxPrePostInjectionAllowed() &&
            likelyPostProcessQuad && targetsBackbuffer &&
            m_parent->UseProgrammablePS() && d3d9State().pixelShader.ptr() != nullptr) {
          const Ue3ShaderFeatureInfo psInfo =
            getUe3ShaderFeatureInfo(d3d9State().pixelShader->GetCommonShader());

          if (ngxShaderIsFinishRenderViewTargetGamma(psInfo)) {
            const Rc<DxvkImage> matchedTarget = findNgxPrePostSceneColorFromSamplers(false, true);

            if (matchedTarget != nullptr) {
              engageNgxPrePostInjection(matchedTarget);
              triggerInjection = true;

              ONCE(Logger::info(str::format(
                "[RTX NGX Passthrough] Post-process disabled: pre-post-process injection engaged at ",
                "FinishRenderViewTarget (DLSS runs on the ",
                (m_ngxColorMirrorImage != nullptr ? "resolved scene color" : "scene color"),
                " before gamma composite to the backbuffer).")));
            }
          }
        }

        if (!triggerInjection && !recordedRuntimeComposite) {
          m_currentUe3PassType = classifyUe3Pass(drawContext);

          if (m_currentUe3PassType == Ue3PassType::UiComposite) {
            reportNgxPrePostMiss();
            triggerInjection = true;
          } else if (isRenderingUI()) {
            reportNgxPrePostMiss();
            triggerInjection = true;
          } else if (m_frameOptions.preTransformedVerticesIsUI &&
                     d3d9State().vertexDecl != nullptr &&
                     d3d9State().vertexDecl->TestFlag(D3D9VertexDeclFlag::HasPositionT)) {
            triggerInjection = true;
          }
        }
      }
    }

    if (triggerInjection) {
      // Everything from the trigger draw on samples the un-jittered DLSS output: stop the
      // ScreenPositionScaleBias patch and force fresh (unpatched) constant uploads
      m_ngxSpsbPatchActive = false;
      m_parent->m_consts[DxsoProgramTypes::VertexShader].dirty = true;
      m_parent->m_consts[DxsoProgramTypes::PixelShader].dirty = true;
      m_ngxSpsbLastVsReg = -1;
      m_ngxSpsbLastPsReg = -1;

      // Late/Super Resolution injection targets the pre-UI backbuffer, so the dispatch
      // captures the HUD-less copy inline; only the pre-post-process injection needs the
      // UI-boundary capture later in the frame
      if (m_ngxColorTargetImage == nullptr) {
        m_ngxHudlessCapturedThisFrame = true;
      }

      // Bind all resources required for this drawcall to context first (i.e. render targets)
      m_parent->PrepareDraw(drawContext.PrimitiveType);

      emitNgxPassthroughFrameData();

      triggerInjectRTX();

      m_rtxInjectTriggered = true;

      // The UI must not inherit the scene's sub-pixel jitter
      m_ngxViewportJitterApplied = false;
      m_ngxAppliedViewportJitter[0] = 0.0f;
      m_ngxAppliedViewportJitter[1] = 0.0f;
      m_parent->m_flags.set(D3D9DeviceFlag::DirtyViewportScissor);
    }

    return suppressDraw ? PrepareDrawFlag::Ignore : PrepareDrawFlag::PreserveDrawCallAndItsState;
  }

  PrepareDrawFlags D3D9Rtx::PrepareDrawGeometryForRT(const bool indexed, const DrawContext& context) {
    (void)indexed;

    if (unlikely(!m_frameOptions.valid)) {
      refreshFrameOptionCache();
    }

    if (!m_enableDrawCallConversion) {
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    return prepareDrawForNgxPassthrough(context);
  }

  PrepareDrawFlags D3D9Rtx::PrepareDrawUPGeometryForRT(const bool indexed,
                                                       const D3D9BufferSlice& buffer,
                                                       const D3DFORMAT indexFormat,
                                                       const uint32_t indexSize,
                                                       const uint32_t indexOffset,
                                                       const uint32_t vertexSize,
                                                       const uint32_t vertexStride,
                                                       const DrawContext& drawContext) {
    (void)indexed;
    (void)buffer;
    (void)indexFormat;
    (void)indexSize;
    (void)indexOffset;
    (void)vertexSize;
    (void)vertexStride;

    if (unlikely(!m_frameOptions.valid)) {
      refreshFrameOptionCache();
    }

    if (!m_enableDrawCallConversion) {
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    return prepareDrawForNgxPassthrough(drawContext);
  }

  void D3D9Rtx::ResetSwapChain(const D3DPRESENT_PARAMETERS& presentationParameters) {
    // Early out if the cached present parameters are not out of date

    if (m_activePresentParams.has_value()) {
      if (
        m_activePresentParams->BackBufferWidth == presentationParameters.BackBufferWidth &&
        m_activePresentParams->BackBufferHeight == presentationParameters.BackBufferHeight &&
        m_activePresentParams->BackBufferFormat == presentationParameters.BackBufferFormat &&
        m_activePresentParams->BackBufferCount == presentationParameters.BackBufferCount &&
        m_activePresentParams->MultiSampleType == presentationParameters.MultiSampleType &&
        m_activePresentParams->MultiSampleQuality == presentationParameters.MultiSampleQuality &&
        m_activePresentParams->SwapEffect == presentationParameters.SwapEffect &&
        m_activePresentParams->hDeviceWindow == presentationParameters.hDeviceWindow &&
        m_activePresentParams->Windowed == presentationParameters.Windowed &&
        m_activePresentParams->EnableAutoDepthStencil == presentationParameters.EnableAutoDepthStencil &&
        m_activePresentParams->AutoDepthStencilFormat == presentationParameters.AutoDepthStencilFormat &&
        m_activePresentParams->Flags == presentationParameters.Flags &&
        m_activePresentParams->FullScreen_RefreshRateInHz == presentationParameters.FullScreen_RefreshRateInHz &&
        m_activePresentParams->PresentationInterval == presentationParameters.PresentationInterval
      ) {
        return;
      }
    }

    // Cache the present parameters
    m_activePresentParams = presentationParameters;

    // Prime settings before the first scene pass.
    if (RtxNgxPassthrough::ngxPassthroughMode()) {
      if (!m_frameOptions.valid) {
        refreshFrameOptionCache();
      }
      applyNgxPassthroughScreenPercentage();
    }

    // Inform the backend about potential presenter update
    m_parent->EmitCs([cWidth = m_activePresentParams->BackBufferWidth,
                      cHeight = m_activePresentParams->BackBufferHeight](DxvkContext* ctx) {
      static_cast<RtxContext*>(ctx)->resetScreenResolution({ cWidth, cHeight , 1 });
    });
  }





  void D3D9Rtx::TrackOcclusionQueryResult(DWORD samplesPassed, uint32_t bracketId) {
    if (!m_frameOptions.ue3LogOcclusionQueries) {
      return;
    }

    ++m_oqDiag.results;
    m_oqDiag.minResult = std::min(m_oqDiag.minResult, samplesPassed);
    m_oqDiag.maxResult = std::max(m_oqDiag.maxResult, samplesPassed);

    if (samplesPassed != 0) {
      return;
    }
    ++m_oqDiag.zeroResults;

    const auto& record = m_oqRecords[bracketId & (kOcclusionQueryRecordCount - 1)];
    const bool haveRecord = bracketId != 0 && record.bracketId == bracketId;

    if (haveRecord) {
      if (record.cameraInsideBox) {
        ++m_oqDiag.zeroCameraInside;
      }
      if (m_activePresentParams.has_value() &&
          record.viewportW < uint16_t(m_activePresentParams->BackBufferWidth / 2) &&
          record.viewportH < uint16_t(m_activePresentParams->BackBufferHeight / 2)) {
        ++m_oqDiag.zeroSmallViewport;
      }
    }

    if (m_oqDiag.zeroResultLogsRemaining == 0) {
      return;
    }
    --m_oqDiag.zeroResultLogsRemaining;

    if (haveRecord) {
      Logger::info(str::format(
        "[UE3-OQ] 0 samples passed: bracket=", bracketId,
        " draws=", record.drawCount,
        " prims=", record.primCount,
        " viewport=", record.viewportW, "x", record.viewportH,
        " conservative=", record.conservativeActive ? 1 : 0,
        " cameraInside=", record.cameraInsideBox ? 1 : 0,
        " camDistToBox=", record.cameraToBoxDistance,
        " boxMin=(", record.boxMin.x, ",", record.boxMin.y, ",", record.boxMin.z, ")",
        " boxMax=(", record.boxMax.x, ",", record.boxMax.y, ",", record.boxMax.z, ")",
        " camera=(", record.cameraPos.x, ",", record.cameraPos.y, ",", record.cameraPos.z, ")"));
    } else {
      Logger::info(str::format(
        "[UE3-OQ] 0 samples passed: bracket=", bracketId,
        " (no record: bracket predates logging or record evicted; ",
        record.drawCount == 0 ? "possibly an empty bracket)" : "ring overwritten)"));
    }
  }

  void D3D9Rtx::flushOcclusionQueryDiagnostics() {
    auto& d = m_oqDiag;
    if (++d.framesSinceSummary < 120) {
      return;
    }

    if (d.brackets != 0 || d.results != 0) {
      Logger::info(str::format(
        "[UE3-OQ] summary over ", d.framesSinceSummary, " frames:",
        " brackets=", d.brackets,
        " emptyBrackets=", d.emptyBrackets,
        " bracketedDraws=", d.bracketedDraws,
        " results=", d.results,
        " zeroResults=", d.zeroResults,
        " zeroCameraInside=", d.zeroCameraInside,
        " zeroSmallViewport=", d.zeroSmallViewport,
        " pendingReads=", d.pendingReads,
        " minResult=", d.results != 0 ? d.minResult : 0,
        " maxResult=", d.maxResult));
    }

    // Reset aggregates but keep the remaining one-off log budgets.
    const uint32_t stateLogs = d.stateSnapshotLogsRemaining;
    const uint32_t zeroLogs = d.zeroResultLogsRemaining;
    d = {};
    d.stateSnapshotLogsRemaining = stateLogs;
    d.zeroResultLogsRemaining = zeroLogs;
  }

  void D3D9Rtx::EndFrame(const Rc<DxvkImage>& targetImage, bool callInjectRtx) {
    refreshFrameOptionCache();

    // Update the effective settings before the next frame.
    applyNgxPassthroughScreenPercentage();

    if (m_frameOptions.ue3LogOcclusionQueries) {
      flushOcclusionQueryDiagnostics();
    }

    const auto currentReflexFrameId = GetReflexFrameId();

    // NGX passthrough mode: no scene end trigger fired this frame (e.g. no UI drawn), so the
    // fallback injection below runs on the backbuffer; hand over this frame's data first
    if (m_frameOptions.ngxPassthroughMode && !m_rtxInjectTriggered && callInjectRtx) {
      emitNgxPassthroughFrameData();
    }

    // HUD-less fallback: the injection ran (pre-post-process) but no UI-classified draw
    // followed, so the presented frame is its own HUD-less copy
    if (m_frameOptions.ngxPassthroughMode && m_rtxInjectTriggered && callInjectRtx &&
        m_frameOptions.ngxDlfgHudless && !m_ngxHudlessCapturedThisFrame &&
        m_ngxFrameBackbufferImage != nullptr) {
      m_ngxHudlessCapturedThisFrame = true;

      m_parent->EmitCs([cBackbuffer = m_ngxFrameBackbufferImage](DxvkContext* ctx) {
        static_cast<RtxContext*>(ctx)->captureNgxPassthroughHudless(cBackbuffer);
      });
    }
    m_parent->Flush();

    // Inform backend of end-frame
    m_parent->EmitCs([currentReflexFrameId, targetImage, callInjectRtx](DxvkContext* ctx) { 
      static_cast<RtxContext*>(ctx)->endFrame(currentReflexFrameId, targetImage, callInjectRtx); 
    });

    // Wait for injectRTX to finish upscaling the backbuffer before the external presenter
    // (or deferred UI replay) touches it on the same thread.
    if (callInjectRtx) {
      m_parent->Flush();
    }

    DrawCallState::refreshCategoryLookupTable();

    // The per-draw profile zones in this file are only meaningful against the draw count, which
    // a UE3 title moves by an order of magnitude depending on its occlusion and frustum culling.
    ProfilerPlotValue("D3D9 Draw Calls", int64_t(m_drawCallID));

    // Reset for the next frame
    m_rtxInjectTriggered = false;
    m_drawCallID = 0;
    ++m_ue3FrameCounter;

    // NGX passthrough per-frame state
    m_ngxSpsbPatchActive = false;
    m_ngxFrameJitterValid = false;
    // The new frame's jitter differs, so the bound viewport no longer matches anything
    m_ngxViewportJitterApplied = false;
    m_ngxAppliedViewportJitter[0] = 0.0f;
    m_ngxAppliedViewportJitter[1] = 0.0f;
    m_ngxFrameDataEmitted = false;
    m_ngxDepthSnapshotTakenThisFrame = false;
    m_ngxDepthClearsThisFrame = 0;
    m_ngxDiagSawPostQuad = false;
    m_ngxDiagSceneColorReady = false;
    m_ngxBackbufferDrawSeenThisFrame = false;

    updateNgxPrePostInjectionAim();

    m_ngxHudlessCapturedThisFrame = false;

    // Object velocity per-frame state: rotate the accepted camera, drop uncaptured
    // leftovers, and bound the identity cache (level transitions leave stale entries
    // behind; re-registration costs one frame of velocity)
    if (m_ngxFrameCameraValid && m_ngxPrevCameraValid &&
        m_ngxFrameCameraUsedTranspose != m_ngxPrevCameraUsedTranspose) {
      m_ngxCameraTransposeFlips++;
    }
    m_ngxPrevCameraValid = m_ngxFrameCameraValid;
    m_ngxPrevCameraUsedTranspose = m_ngxFrameCameraUsedTranspose;
    m_ngxFrameCameraValid = false;
    m_ngxFrameCameraMatricesValid = false;
    // Settle on the best supported offset of the frame, which may be better supported than
    // whichever candidate led at the moment it was adopted. A frame that produced no evidence
    // at all leaves the standing offset alone rather than dropping it: zeroing it would put a
    // step the size of the camera's travel through the next frame's reprojection, and a frame
    // with nothing to measure from is equally a frame with nothing to misjudge.
    {
      const NgxTranslationDeltaVote* winner = nullptr;
      for (const NgxTranslationDeltaVote& vote : m_ngxTranslationDeltaVotes) {
        if (vote.votes > 0 && (winner == nullptr || vote.votes > winner->votes)) {
          winner = &vote;
        }
      }
      if (winner != nullptr) {
        m_ngxGlobalTransformOffset = winner->delta;
      }
      m_ngxTranslationDeltaVotes.fill(NgxTranslationDeltaVote());
      m_ngxAdoptedOffsetVotes = 0;
    }

    m_ngxVelocityStats = NgxVelocityCaptureStats();
    m_ngxVelocitySkinnedDraws = 0;
    m_ngxVelocityDynamicDraws = 0;
    m_ngxVelocityDraws.clear();
    // Instances left behind by level streaming and visibility changes accumulate, so the cache
    // drops the ones not sighted for a while: as routine upkeep, and as soon as it grows past
    // the ceiling. The ceiling bounds memory, not correctness - discarding the cache wholesale
    // takes every object's previous transform with it, so a level dense enough to sit above the
    // ceiling loses its history on every frame and the scene renders with no object velocities
    // at all. Pruning by age instead only ever drops what could not have paired anyway, and
    // tightens the age until the cache fits.
    constexpr size_t kVelocityCacheCeiling = 32768;
    constexpr uint32_t kVelocityCacheRoutinePruneAge = 1024;

    auto pruneVelocityInstancesOlderThan = [this](const uint32_t maxAge) {
      m_ngxVelocityObjectCache.erase_if([frame = m_ue3FrameCounter, maxAge](auto it) {
        auto& instances = it->second.instances;
        instances.erase(std::remove_if(instances.begin(), instances.end(),
                                       [frame, maxAge](const NgxVelocityObjectInstance& instance) {
                                         return instance.lastSeenFrame + maxAge < frame;
                                       }),
                        instances.end());
        return instances.empty();
      });
    };

    if ((m_ue3FrameCounter & 0xFF) == 0 || m_ngxVelocityObjectCache.size() > kVelocityCacheCeiling) {
      uint32_t maxAge = kVelocityCacheRoutinePruneAge;
      pruneVelocityInstancesOlderThan(maxAge);

      // Pairing only ever reaches one frame back, so tightening the age surrenders nothing that
      // was going to be used. At the floor the cache holds only what the last couple of frames
      // sighted, which the per-frame draw count bounds on its own.
      while (m_ngxVelocityObjectCache.size() > kVelocityCacheCeiling && maxAge > 2) {
        maxAge /= 4;
        pruneVelocityInstancesOlderThan(maxAge);
      }
    }
    m_ngxLastCameraConstantsHash = 0;
    m_ngxSceneViewportValid = false;
    m_ngxUpscaleSourceImage = nullptr;
    m_ngxColorTargetImage = nullptr;
    m_ngxColorMirrorImage = nullptr;
    m_ngxSubrect = { { 0, 0 }, { 0, 0 } };
    m_ngxColorSubrectOffset = { 0, 0 };
    m_ngxRuntimeUpscaleRectValid = false;
    m_ngxOutputTransform = NgxOutputTransform();
    for (uint32_t i = 0; i < m_ngxSceneColorResolveCount; i++) {
      m_ngxSceneColorResolves[i] = nullptr;
    }
    m_ngxSceneColorResolveCount = 0;
    m_ngxVelocityDumpSkinnedThisFrame = 0;
    m_ngxVelocityDumpDynamicThisFrame = 0;
    m_ngxVelocityDumpRigidThisFrame = 0;
    if (m_ngxVelocityDumpFramesLeft > 0) {
      m_ngxVelocityDumpFramesLeft--;
    }
    if (m_ngxPostChainDumpFramesLeft > 0) {
      m_ngxPostChainDumpFramesLeft--;
      Logger::info(str::format("[RTX NGX Passthrough][dump] ---- end of frame ", m_ue3FrameCounter, " ----"));
    }
    m_ngxPostChainDumpLinesThisFrame = 0;
    // Drop scene target references when unseen for a while (level transitions recreate them)
    if (m_ngxSceneColorImage != nullptr && m_ue3FrameCounter - m_ngxSceneTargetsLastSeenFrame > 60) {
      m_ngxSceneColorImage = nullptr;
      m_ngxSceneDepthImage = nullptr;
    }

    // two-pass translucency dedup state must not span frames
    m_prevDrawVsPsHash = 0;
    m_prevDrawTextureHash = 0;
    m_prevDrawGeometryHash = 0;
    m_prevDrawCullMode = 0;
  }

  void D3D9Rtx::OnPresent(const Rc<DxvkImage>& targetImage) {
    // Inform backend of present
    m_parent->EmitCs([targetImage](DxvkContext* ctx) { static_cast<RtxContext*>(ctx)->onPresent(targetImage); });
  }
}
