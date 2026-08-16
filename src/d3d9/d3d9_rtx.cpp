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
#include "../dxvk/rtx_render/rtx_terrain_baker.h"
#include "../dxvk/rtx_render/rtx_ue3_tone_mapping.h"
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
    // (scale/offset from draw-time constants or def'd immediates) applied along the way.
    // Math outside the affine model keeps the origin but marks the affine inexact;
    // genuinely ambiguous origins are invalidated.
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
      return uvAffineTermsEqual(a.scale, b.scale) && uvAffineTermsEqual(a.offset, b.offset);
    }

    static bool uvComponentAffineExact(const UvComponentAffine& a) {
      return !a.scale.inexact && !a.offset.inexact;
    }

    static bool uvComponentAffineIsIdentity(const UvComponentAffine& a) {
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
    }

    // value' = value * m for a compile-time immediate m
    static void uvAffineMulImmediate(UvComponentAffine& a, const float value) {
      if (a.scale.constReg >= 0) {
        a.scale.factor *= value;
      } else if (a.scale.immValid) {
        a.scale.imm *= value;
      } else {
        a.scale.immValid = true;
        a.scale.imm = value;
      }

      // the offset is a sum: multiplying by an immediate distributes over every part
      if (a.offset.immValid) {
        a.offset.imm *= value;
      }
      if (a.offset.constReg >= 0) {
        a.offset.factor *= value;
      }
      if (a.offset.constReg2 >= 0) {
        a.offset.factor2 *= value;
      }
    }

    // value' = value * consts[reg][comp] * factor
    static void uvAffineMulConstant(UvComponentAffine& a, const int16_t reg, const uint8_t comp, const float factor) {
      if (a.scale.constReg >= 0) {
        // scale would become a product of two draw-time constants - not representable
        a.scale.inexact = true;
      } else if (a.scale.immValid) {
        const float staticScale = a.scale.imm;
        a.scale.immValid = false;
        a.scale.imm = 0.0f;
        a.scale.constReg = reg;
        a.scale.constComp = comp;
        a.scale.factor = staticScale * factor;
      } else {
        a.scale.constReg = reg;
        a.scale.constComp = comp;
        a.scale.factor = factor;
      }

      if (a.offset.constReg >= 0) {
        // any existing constant part times a new draw-time constant is a product of two
        // draw-time constants - not representable
        a.offset.inexact = true;
      } else if (a.offset.immValid) {
        if (a.offset.imm != 0.0f) {
          const float staticOffset = a.offset.imm;
          a.offset.immValid = false;
          a.offset.imm = 0.0f;
          a.offset.constReg = reg;
          a.offset.constComp = comp;
          a.offset.factor = staticOffset * factor;
        } else {
          a.offset.immValid = false;
        }
      }
    }

    static void uvAffineAddImmediate(UvComponentAffine& a, const float value) {
      if (value == 0.0f)
        return;

      a.offset.immValid = true;
      a.offset.imm += value;
    }

    static void uvAffineAddConstant(UvComponentAffine& a, const int16_t reg, const uint8_t comp, const float factor) {
      if (a.offset.immValid && a.offset.imm == 0.0f)
        a.offset.immValid = false;

      if (a.offset.constReg >= 0 && a.offset.constComp == comp && a.offset.constReg == reg) {
        // same component referenced twice: fold into the first part's factor
        a.offset.factor += factor;
        if (a.offset.factor == 0.0f) {
          // cancelled out: promote the second part into the first slot to keep the
          // "constReg2 only set when constReg is" invariant
          a.offset.constReg = a.offset.constReg2;
          a.offset.constComp = a.offset.constComp2;
          a.offset.factor = a.offset.factor2;
          a.offset.constReg2 = -1;
          a.offset.constComp2 = 0;
          a.offset.factor2 = 1.0f;
        }
        return;
      }
      if (a.offset.constReg2 >= 0 && a.offset.constComp2 == comp && a.offset.constReg2 == reg) {
        a.offset.factor2 += factor;
        if (a.offset.factor2 == 0.0f) {
          a.offset.constReg2 = -1;
          a.offset.constComp2 = 0;
          a.offset.factor2 = 1.0f;
        }
        return;
      }

      if (a.offset.constReg < 0) {
        a.offset.constReg = reg;
        a.offset.constComp = comp;
        a.offset.factor = factor;
      } else if (a.offset.constReg2 < 0) {
        a.offset.constReg2 = reg;
        a.offset.constComp2 = comp;
        a.offset.factor2 = factor;
      } else {
        // more than two distinct constant parts - not representable
        a.offset.inexact = true;
      }
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
      return str::format("uv*", formatUvAffineTerm(affine.scale, "1"), "+", formatUvAffineTerm(affine.offset, "0"));
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
        // register provenance (and the packed-UV half being read) survives. This is what keeps
        // UV rotators / matrix transforms (dp2add pairs, m3x2..m4x4) attributable.
        case DxsoOpcode::Dp2Add: {
          UvExactComponentOrigin merged = uvMergeOrigins(
            readOriginUnion(ctx.src[0], 2u),
            readOriginUnion(ctx.src[1], 2u));
          merged = uvMergeOrigins(merged, readOrigin(ctx.src[2], c));
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
                // component-mixing math (rotators, UV matrices): the register is proven but the
                // exact pair is not - attribute the packed interpolant half being consumed
                const uint8_t unionMask = uint8_t(uOrigin.componentMask | vOrigin.componentMask);
                const bool onlySecondaryHalf =
                  (unionMask & 0b0011u) == 0u && (unionMask & 0b1100u) != 0u;
                site.compU = onlySecondaryHalf ? 3u : 0u;
                site.compV = onlySecondaryHalf ? 2u : 1u;
              }
              site.affineU = uOrigin.affine;
              site.affineV = vOrigin.affine;
              site.affineExact =
                cleanComponents &&
                !projected &&
                uvComponentAffineExact(uOrigin.affine) &&
                uvComponentAffineExact(vOrigin.affine);
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

    // Material texture params (TextureParameterValues) live in samplers the UE3 material
    // translator names Texture2D_* / TextureCube_*, and material constants (VectorParameterValues
    // e.g. DiffuseColor, ScalarParameterValues) in UniformVector_* / UniformScalar_* registers -
    // the shader CTAB is parsed for both.
    // Note: UE3 also writes frame-varying uniform expression values (Time, fades, sub-UV frames)
    // into the same constant registers; the two are indistinguishable at the D3D9 level, so
    // shaders doing that must be opted out of constants-based identity via
    // rtx.d3d9.ue3MicConstantIdentityExcludedShaders.
    constexpr uint16_t kD3dxRegisterSetFloat4 = 2u;
    constexpr uint16_t kD3dxRegisterSetSampler = 3u;

    // merged (start, count) register ranges of UniformVector_* / UniformScalar_* constants
    using Ue3MaterialConstRanges = std::vector<std::pair<uint32_t, uint32_t>>;

    // Canonical shader signature serialization. Entries carry names and register classes
    // only: register indices shift with lightmap sampler counts and register counts are
    // fxc's *used* element counts, which vary per permutation - neither may leak into the
    // permutation-invariant signature.
    static std::string makeUe3SignatureEntry(const std::string& lowerName, const uint16_t registerSet) {
      return str::format(lowerName, "\x01", registerSet);
    }

    // sorts entries in place; callers discard them afterwards
    static XXH64_hash_t hashUe3SignatureEntries(std::vector<std::string>& entries) {
      if (entries.empty())
        return kEmptyHash;
      std::sort(entries.begin(), entries.end());
      std::string serialized;
      for (const std::string& entry : entries) {
        serialized += entry;
        serialized += '\x02';
      }
      return XXH3_64bits(serialized.data(), serialized.size());
    }

    struct Ue3PsMaterialIdentityInfo {
      Ue3MaterialConstRanges constRanges;
      // bit per sampler index: CTAB sampler strictly named texture2d_* / texturecube_* / texture3d_*
      uint32_t materialSamplerMask = 0;
      // UniformVector_* float registers in ascending register order - for constant-color
      // materials (no material texture samplers) one of these holds the material's color
      std::vector<uint32_t> uniformVectorRegisters;
      // (name, name key, sampler register) per material sampler, ordered by CTAB name. The UE3
      // material translator assigns Texture2D_N/TextureCube_N names once per material, so
      // name-keyed streaming aligns the texture set across lightmap policy permutations even
      // when lightmap sampler counts shift the register assignments.
      std::vector<std::tuple<std::string, XXH64_hash_t, uint32_t>> materialSamplersByNameOrder;
      // (name key, first register) per Uniform* constant, ordered by name. Name-keyed,
      // leading-register-only streaming keeps the constants identity aligned across lightmap
      // policy permutations, which shift uniform registers.
      std::vector<std::pair<XXH64_hash_t, uint32_t>> namedUniformFirstRegistersByNameOrder;
      // XXH3 over the name-sorted material sampler declarations (names and register class
      // only): a shader identity that is stable across the UE3 lightmap policy permutations
      // (directional vs simple texture lightmaps, Mirror's Edge bicubic lightmap filtering)
      // compiled from the same material. Register indices, element counts, and Uniform*
      // constants are permutation-dependent (fxc strips or trims whatever a permutation does
      // not reference) and deliberately excluded.
      XXH64_hash_t canonicalShaderSignature = kEmptyHash;
      bool hasCtab = false;

      // Set only when a sampler was left out, for rtx.d3d9.ue3LogMaterialInstanceHash: comparing
      // it between a DirectionalLightmaps=True and =False run is the direct way to see whether a
      // material's two compiles agree on their identity inputs.
      std::string identitySummary;
    };

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

    // Churn threshold for warning about frame-varying constant registers: genuine
    // constant-differentiated material instance siblings form small groups (measured 2-8 per
    // shader+texture set), while frame-varying values sweep unbounded hashes within a single
    // group. A widely reused shader legitimately produces many hashes spread across many
    // texture sets, so the count must be per group, not per shader.
    constexpr uint32_t kUe3MicChurnWarnThreshold = 32;

    // Runtime auto-exclusion of frame-varying constants from material identity
    // (rtx.d3d9.ue3MicAutoExcludeFrameVaryingConstants): per (identity seed, texture set)
    // group, remember the most recent distinct constants hashes in a bounded ring. Re-seeing
    // a sibling's hash is a ring hit and DECAYS the distinct count: stable siblings redraw
    // every frame, so legitimate constant-differentiated families - however many siblings
    // accumulate across levels - hit far more than they miss and never trip. Frame-varying
    // constants (Time/panner/fade/sub-UV expressions) mint a new hash every draw, never hit
    // the ring, and cross the threshold within a second - after which the GROUP's constants
    // are dropped from identity for the session. Exclusion is keyed per group, never per
    // seed: the canonical seed is only a sampler-name signature shared by many unrelated
    // materials, and excluding it wholesale would collapse the identity - and break the
    // replacement matching - of every material that shares it.
    constexpr uint32_t kUe3MicConstantHashRingSize = kUe3MicChurnWarnThreshold;

    struct Ue3MicConstantChurnEntry {
      std::array<XXH64_hash_t, kUe3MicConstantHashRingSize> recentHashes = {};
      uint32_t ringCursor = 0;
      uint32_t distinctCount = 0;
    };

    static fast_unordered_cache<Ue3MicConstantChurnEntry> s_ue3MicConstantChurnPerGroup;
    static fast_unordered_set s_ue3MicAutoExcludedGroups;

    // Shared by the UE3 on-disk caches. Each rewrites its whole file from an in-memory
    // container, so a save is only safe when that container holds everything the file held:
    // a load that ends up with less must take the file out of reach for the session, or one
    // unreadable read becomes permanent loss of data that took a playthrough to accumulate.
    // Distinguishing "absent" from "unreadable" is what makes a first run still able to save.
    bool ue3CacheFileExists(const char* path) {
      std::error_code error;
      return std::filesystem::exists(path, error);
    }

    // Publishes a fully written temp file over the real one, so a process that dies mid-write
    // leaves the previous cache intact rather than a truncated file that the next load would
    // read as a short but well-formed one.
    bool ue3CommitCacheFile(const char* tempPath, const char* path) {
      std::error_code error;
      std::filesystem::rename(tempPath, path, error);
      if (error) {
        std::filesystem::remove(tempPath, error);
        return false;
      }
      return true;
    }

    // Cross-session persistence for the auto-exclusion set; rationale in the
    // rtx.d3d9.ue3MicPersistAutoExcludedConstantGroups option documentation.
    constexpr char kUe3MicAutoExcludedGroupsCachePath[] = "rtx-remix/ue3MicAutoExcludedGroups.cache";
    constexpr char kUe3MicAutoExcludedGroupsCacheTempPath[] = "rtx-remix/ue3MicAutoExcludedGroups.cache.tmp";
    constexpr uint64_t kUe3MicAutoExcludedGroupsCacheMagic = 0x315843494D334555ull; // "UE3MICX1"
    constexpr uint32_t kUe3MicAutoExcludedGroupsCacheMaxEntries = 1u << 16;

    static bool s_ue3MicAutoExcludedGroupsLoaded = false;
    // See the spread cache: a save rewrites the file from the set, so a load that could not
    // read all of it must not be published. Losing entries here silently un-excludes a
    // frame-varying group, which changes its material hash and breaks anchors authored
    // against the excluded identity.
    static bool s_ue3MicAutoExcludedGroupsSaveBlocked = false;

    static void loadUe3MicAutoExcludedGroupsCache() {
      s_ue3MicAutoExcludedGroupsLoaded = true;

      auto refuseFutureSaves = [](const std::string& reason) {
        s_ue3MicAutoExcludedGroupsSaveBlocked = true;
        Logger::warn(str::format(
          "[RTX-Compatibility][UE3-MIC] Constants auto-exclusion cache ", reason,
          ". Leaving the file untouched for this session; groups re-derive their exclusions "
          "from scratch. Delete ", kUe3MicAutoExcludedGroupsCachePath, " to start a fresh one."));
      };

      const bool fileExists = ue3CacheFileExists(kUe3MicAutoExcludedGroupsCachePath);

      std::ifstream file(kUe3MicAutoExcludedGroupsCachePath, std::ios::binary);
      if (!file.is_open()) {
        if (fileExists)
          refuseFutureSaves("exists but could not be opened");
        return;
      }

      uint64_t magic = 0;
      uint32_t entryCount = 0;
      file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
      file.read(reinterpret_cast<char*>(&entryCount), sizeof(entryCount));
      if (!file || magic != kUe3MicAutoExcludedGroupsCacheMagic || entryCount > kUe3MicAutoExcludedGroupsCacheMaxEntries) {
        refuseFutureSaves("header is unreadable or not recognised");
        return;
      }

      for (uint32_t i = 0; i < entryCount; i++) {
        XXH64_hash_t groupKey = 0;
        file.read(reinterpret_cast<char*>(&groupKey), sizeof(groupKey));
        if (!file) {
          refuseFutureSaves(str::format("is truncated: ", i, " of ", entryCount, " groups readable"));
          return;
        }
        s_ue3MicAutoExcludedGroups.insert(groupKey);
      }

      Logger::info(str::format(
        "[RTX-Compatibility][UE3-MIC] Loaded constants auto-exclusion cache: ", entryCount, " material groups"));
    }

    static void saveUe3MicAutoExcludedGroupsCache() {
      // Merge any not-yet-loaded persisted entries first so a save never drops them.
      if (!s_ue3MicAutoExcludedGroupsLoaded)
        loadUe3MicAutoExcludedGroupsCache();

      if (s_ue3MicAutoExcludedGroupsSaveBlocked)
        return;

      {
        std::ofstream file(kUe3MicAutoExcludedGroupsCacheTempPath, std::ios::binary | std::ios::trunc);
        if (!file.is_open())
          return;

        const uint32_t entryCount =
          uint32_t(std::min<size_t>(s_ue3MicAutoExcludedGroups.size(), kUe3MicAutoExcludedGroupsCacheMaxEntries));
        file.write(reinterpret_cast<const char*>(&kUe3MicAutoExcludedGroupsCacheMagic), sizeof(kUe3MicAutoExcludedGroupsCacheMagic));
        file.write(reinterpret_cast<const char*>(&entryCount), sizeof(entryCount));

        uint32_t written = 0;
        for (const XXH64_hash_t groupKey : s_ue3MicAutoExcludedGroups) {
          if (written >= entryCount)
            break;
          file.write(reinterpret_cast<const char*>(&groupKey), sizeof(groupKey));
          written++;
        }

        file.close();
        if (!file)
          return;
      }

      ue3CommitCacheFile(kUe3MicAutoExcludedGroupsCacheTempPath, kUe3MicAutoExcludedGroupsCachePath);
    }

    // ===== Replacement identity drift diagnostics =====
    // (rtx.logReplacementResolution / rtx.replacementDebugHashes)
    //
    // Remembers, per material family, the identity tiers behind the last minted material
    // hash so a mint with a different hash can be attributed to the tier that moved
    // (texture set, constants, or exclusion state). A family is keyed by (shader identity
    // seed, primary color texture hash) - both stable across sessions, unlike any of the
    // minted hashes themselves.
    constexpr uint32_t kMicDriftMaxTrackedSamplers = 16;  // caps::MaxTexturesPS
    constexpr uint32_t kMicDriftMaxTrackedConstants = 16;
    // Per-family log caps: without them an A/B-oscillating constants hash (which the churn
    // ring buffer intentionally does not auto-exclude) would emit a drift warning per draw.
    constexpr uint32_t kMicDriftMaxLogsPerFamily = 16;
    constexpr uint32_t kMicDriftMaxLogsPerTrackedFamily = 64;

    struct Ue3MicIdentitySample {
      XXH64_hash_t materialHash = kEmptyHash;
      XXH64_hash_t textureSetHash = kEmptyHash;
      XXH64_hash_t constantsHash = kEmptyHash;
      bool constantsExcluded = false;
      bool valid = false;
      uint32_t driftLogsEmitted = 0;

      struct SamplerRecord {
        uint8_t reg = 0xFF;
        bool isRenderTarget = false;
        XXH64_hash_t imageHash = kEmptyHash;
        XXH64_hash_t descriptorHash = kEmptyHash;
      };
      std::array<SamplerRecord, kMicDriftMaxTrackedSamplers> samplers = {};
      uint32_t samplerCount = 0;

      struct ConstantRecord {
        uint16_t reg = 0xFFFF;
        Vector4 value;
      };
      std::array<ConstantRecord, kMicDriftMaxTrackedConstants> constants = {};
      uint32_t constantCount = 0;
    };

    static fast_unordered_cache<Ue3MicIdentitySample> s_ue3MicIdentityByFamily;
    static fast_unordered_set s_ue3MicRtPoisonWarnedFamilies;

    static XXH64_hash_t makeUe3MicChurnGroupKey(const XXH64_hash_t shaderIdentitySeed,
                                                const XXH64_hash_t textureSetHash) {
      return XXH3_64bits_withSeed(&textureSetHash, sizeof(textureSetHash), shaderIdentitySeed);
    }

    static bool isUe3MicGroupAutoExcluded(const XXH64_hash_t churnGroupKey, const bool persistAcrossSessions) {
      if (persistAcrossSessions && !s_ue3MicAutoExcludedGroupsLoaded)
        loadUe3MicAutoExcludedGroupsCache();
      return s_ue3MicAutoExcludedGroups.find(churnGroupKey) != s_ue3MicAutoExcludedGroups.end();
    }

    // Returns true when the group was newly auto-excluded on this call.
    static bool trackUe3MicConstantChurn(const XXH64_hash_t churnGroupKey,
                                         const XXH64_hash_t psHash,
                                         const XXH64_hash_t shaderIdentitySeed,
                                         const XXH64_hash_t textureSetHash,
                                         const XXH64_hash_t constantsHash,
                                         const bool persistAcrossSessions) {
      Ue3MicConstantChurnEntry& entry = s_ue3MicConstantChurnPerGroup[churnGroupKey];

      for (const XXH64_hash_t seenHash : entry.recentHashes) {
        if (seenHash == constantsHash) {
          if (entry.distinctCount > 0) {
            --entry.distinctCount;
          }
          return false;
        }
      }

      entry.recentHashes[entry.ringCursor] = constantsHash;
      entry.ringCursor = (entry.ringCursor + 1u) % kUe3MicConstantHashRingSize;
      ++entry.distinctCount;

      if (entry.distinctCount >= kUe3MicChurnWarnThreshold &&
          s_ue3MicAutoExcludedGroups.insert(churnGroupKey).second) {
        if (persistAcrossSessions) {
          saveUe3MicAutoExcludedGroupsCache();
        }
        Logger::warn(str::format(
          "[RTX-Compatibility][UE3-MIC] Material group (seed=0x", std::hex, shaderIdentitySeed,
          ", ps=0x", psHash,
          ", textureSet=0x", textureSetHash, std::dec,
          ") minted ", kUe3MicChurnWarnThreshold,
          "+ distinct constant hashes - its constant registers are frame-varying "
          "(Time/panner/fade/sub-UV expressions). Excluding this group's constants from "
          "material identity",
          persistAcrossSessions
            ? " and persisting the exclusion to rtx-remix/ue3MicAutoExcludedGroups.cache (applies "
              "from the start of future sessions; delete the file to reset)."
            : " for this session; add the shader hash or seed to "
              "rtx.d3d9.ue3MicConstantIdentityExcludedShaders to exclude it permanently."));
        return true;
      }

      return false;
    }

    static void logUe3MaterialInstanceHashBreakdownOnce(
        const XXH64_hash_t materialHash,
        const XXH64_hash_t psHash,
        const XXH64_hash_t shaderIdentitySeed,
        const XXH64_hash_t textureSetHash,
        const XXH64_hash_t constantsHash,
        const Ue3PsMaterialIdentityInfo& identityInfo,
        const bool constantsExcluded,
        const std::string& textureList) {
      static fast_unordered_set s_loggedMaterialInstanceHashes;
      if (!s_loggedMaterialInstanceHashes.insert(materialHash).second)
        return;

      std::string ranges;
      for (const auto& [start, count] : identityInfo.constRanges) {
        ranges += str::format(ranges.empty() ? "c" : ",c", start, "+", count);
      }

      const bool usedCanonicalSeed = shaderIdentitySeed != psHash;
      Logger::info(str::format(
        "[RTX-Compatibility][UE3-MIC] materialHash=0x", std::hex, materialHash,
        " ps=0x", psHash,
        " seed=0x", shaderIdentitySeed, std::dec,
        usedCanonicalSeed ? " (canonical, lightmap-permutation invariant)" : " (bytecode)",
        std::hex,
        " textureSet=0x", textureSetHash,
        " consts=0x", constantsHash,
        " group=0x", makeUe3MicChurnGroupKey(shaderIdentitySeed, textureSetHash), std::dec,
        " textures=[", textureList, "]",
        " constRanges=[", ranges, "]",
        constantsExcluded ? " constsExcluded=1" : "",
        " ctab=", identityInfo.hasCtab ? 1 : 0));

      static fast_unordered_cache<uint32_t> s_distinctHashCountPerGroup;
      static fast_unordered_set s_churnWarnedGroups;
      const XXH64_hash_t groupKey = makeUe3MicChurnGroupKey(shaderIdentitySeed, textureSetHash);
      const uint32_t distinctCount = ++s_distinctHashCountPerGroup[groupKey];
      // Warn per group rather than per shader: one shader serves many material families, and
      // warning once for the first to churn hides every other family behind it.
      if (distinctCount == kUe3MicChurnWarnThreshold && s_churnWarnedGroups.insert(groupKey).second) {
        Logger::warn(str::format(
          "[RTX-Compatibility][UE3-MIC] Pixel shader 0x", std::hex, psHash,
          " has minted ", std::dec, distinctCount, "+ distinct material hashes for a single texture set (0x",
          std::hex, textureSetHash, std::dec, ") - its constant registers are likely frame-varying ",
          "(Time/panner/fade/sub-UV expressions). If it is not excluded automatically, add group 0x",
          std::hex, groupKey, std::dec, " to rtx.d3d9.ue3MicConstantIdentityExcludedGroups to stabilize "
          "this material's identity without affecting others on the same shader."));
      }
    }

    constexpr uint8_t kPsSamplerSemanticEngineAuxiliary = 1u << 0;
    constexpr uint8_t kPsSamplerSemanticLightmap        = 1u << 1;
    constexpr uint8_t kPsSamplerSemanticMaterialTexture = 1u << 2;
    constexpr uint8_t kPsSamplerSemanticNonDiffuse      = 1u << 3;
    constexpr uint8_t kPsSamplerSemanticVideo           = 1u << 4;
    constexpr uint8_t kPsSamplerSemanticMovieTexture    = 1u << 5;

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

    // A material sampler the base pass only consumes as a lighting *input* rather than as
    // colour. UE3's simple-lightmap compile does not reference those inputs, so fxc strips them
    // and the two compiles declare different sampler sets for the same material; leaving them
    // in identity is what makes a material's hash move with the DirectionalLightmaps setting.
    //
    // Excluding them makes the sets agree without knowing which compile is running: where the
    // sampler is declared it is recognised and dropped, and where fxc already stripped it there
    // is nothing to drop. The test is deliberately a property of the sampler's own use, so it
    // cannot key off a symbol that exists in one compile and not the other.
    static bool isUe3LightingInputSampler(const PsSamplerTexcoordInference& inferred) {
      // Same signal albedo selection treats as decisive. Guards against the whole rule
      // misfiring on a material's own base texture.
      if ((inferred.expressionFlags & kPsSamplerExprDiffuseAnchor) != 0)
        return false;
      if ((inferred.expressionFlags & kPsSamplerExprNormalDecode) != 0)
        return true;
      if ((inferred.semanticFlags & kPsSamplerSemanticNonDiffuse) != 0)
        return true;
      // Sampled but never reaching oC0 as colour: specular and mask inputs collapse into dot
      // products and comparisons, which is what separates them from a diffuse or emissive map.
      if (inferred.sampleCount > 0 &&
          (inferred.expressionFlags & kPsSamplerExprReachesOutputColor) == 0) {
        return true;
      }
      return false;
    }

    static Ue3PsMaterialIdentityInfo parseUe3PsMaterialIdentityFromCtab(
        const std::vector<uint8_t>& bytecode,
        const D3D9CommonShader* pixelShader) {
      Ue3PsMaterialIdentityInfo info;
      if (bytecode.size() < sizeof(uint32_t) || (bytecode.size() % sizeof(uint32_t)) != 0)
        return info;

      const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
      const uint32_t headerToken = tokens[0];
      const uint32_t headerTypeMask = headerToken & 0xffff0000u;
      if (headerTypeMask != 0xffff0000u)
        return info;

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
      if (ctab.m_size == 0 || ctab.m_constantData.empty())
        return info;

      info.hasCtab = true;

      auto startsWith = [](const std::string& s, const char* prefix) {
        return s.rfind(prefix, 0) == 0;
      };

      struct DeclaredSampler {
        std::string name;
        uint32_t registerIndex = 0;
        uint32_t registerCount = 0;
      };
      struct DeclaredUniform {
        std::string name;
        uint16_t registerSet = 0;
        uint32_t registerIndex = 0;
        uint32_t registerCount = 0;
      };

      std::vector<DeclaredSampler> declaredSamplers;
      std::vector<DeclaredUniform> declaredUniforms;

      for (const DxsoCtab::Constant& c : ctab.m_constantData) {
        const std::string name = toLowerAscii(c.name);

        if (c.registerSet == kD3dxRegisterSetSampler) {
          // strict prefix rule: only the numbered sampler names emitted by the UE3 material
          // translator count as material texture parameters; lightmaps/scene/shadow samplers
          // use other names and must stay out of the material identity
          if (c.registerCount != 0 &&
              (startsWith(name, "texture2d_") || startsWith(name, "texturecube_") || startsWith(name, "texture3d_"))) {
            declaredSamplers.push_back({ name, c.registerIndex, c.registerCount });
          }
          continue;
        }

        if (c.registerSet > 2u || c.registerCount == 0)
          continue;
        if (c.registerIndex + c.registerCount > caps::MaxFloatConstantsPS)
          continue;

        const bool isUniformVector = name.find("uniformvector_") != std::string::npos;
        const bool isUniformScalar = name.find("uniformscalar_") != std::string::npos;
        if (!isUniformVector && !isUniformScalar)
          continue;
        // Constant-colour materials read their colour straight out of these, so the list has to
        // stay complete even though only the vectors reach the identity hash.
        if (isUniformVector) {
          info.uniformVectorRegisters.push_back(c.registerIndex);
        }
        declaredUniforms.push_back({ name, c.registerSet, c.registerIndex, c.registerCount });
      }

      std::vector<std::pair<std::string, uint32_t>> uniformsByName;
      std::vector<std::string> samplerSignatureEntries;
      std::vector<std::string> uniformSignatureEntries;
      std::vector<std::string> keptSamplerNames, excludedSamplerNames;

      // Only needed to fall back on if every sampler classifies as a lighting input.
      uint32_t allSamplerMask = 0;
      std::vector<std::tuple<std::string, XXH64_hash_t, uint32_t>> allSamplersByNameOrder;
      std::vector<std::string> allSamplerSignatureEntries;

      for (const DeclaredSampler& sampler : declaredSamplers) {
        const uint32_t end = std::min<uint32_t>(sampler.registerIndex + sampler.registerCount, caps::MaxTexturesPS);
        bool anyKept = false;
        for (uint32_t s = sampler.registerIndex; s < end; s++) {
          // arrays get per-register names so name keys stay unambiguous (material samplers
          // are scalar in practice, this is defensive)
          const std::string samplerName =
            sampler.registerCount > 1u ? str::format(sampler.name, "[", s - sampler.registerIndex, "]") : sampler.name;
          const XXH64_hash_t samplerNameKey = XXH3_64bits(samplerName.data(), samplerName.size());
          allSamplerMask |= (1u << s);
          allSamplersByNameOrder.emplace_back(samplerName, samplerNameKey, s);

          if (pixelShader != nullptr && isUe3LightingInputSampler(inferPixelShaderTexcoordForSampler(pixelShader, s))) {
            excludedSamplerNames.push_back(samplerName);
            continue;
          }
          keptSamplerNames.push_back(samplerName);
          anyKept = true;
          info.materialSamplerMask |= (1u << s);
          info.materialSamplersByNameOrder.emplace_back(samplerName, samplerNameKey, s);
        }
        allSamplerSignatureEntries.push_back(makeUe3SignatureEntry(sampler.name, kD3dxRegisterSetSampler));
        if (anyKept)
          samplerSignatureEntries.push_back(makeUe3SignatureEntry(sampler.name, kD3dxRegisterSetSampler));
      }

      // A material that classified as nothing but lighting inputs has been misread; an empty set
      // would collapse every such material onto one identity. Keep what was declared instead:
      // that costs invariance for this shader but never merges distinct materials.
      if (info.materialSamplerMask == 0 && allSamplerMask != 0) {
        info.materialSamplerMask = allSamplerMask;
        info.materialSamplersByNameOrder = std::move(allSamplersByNameOrder);
        samplerSignatureEntries = std::move(allSamplerSignatureEntries);
        keptSamplerNames.swap(excludedSamplerNames);
        excludedSamplerNames.clear();
      }

      for (const DeclaredUniform& uniform : declaredUniforms) {
        // Vectors only. Scalars are where the two lightmap compiles genuinely disagree:
        // DiffusePower exponents the lightmap under SIMPLE_LIGHTING and a LightMapBasis-derived
        // transfer coefficient otherwise, and SpecularPower is its structural twin present in
        // only one compile - nothing in the bytecode separates them, so keeping either keeps
        // both. The vectors carry the tint that tells UE3's colour variants apart.
        if (uniform.name.find("uniformscalar_") != std::string::npos)
          continue;

        uniformsByName.emplace_back(uniform.name, uniform.registerIndex);
        uniformSignatureEntries.push_back(makeUe3SignatureEntry(uniform.name, uniform.registerSet));
        info.constRanges.emplace_back(uniform.registerIndex, uniform.registerCount);
      }

      if (!excludedSamplerNames.empty()) {
        auto join = [](const std::vector<std::string>& names) {
          std::string out;
          for (const std::string& name : names) {
            if (!out.empty())
              out += ',';
            out += name;
          }
          return out.empty() ? std::string("-") : out;
        };
        info.identitySummary = str::format(
          "samplers kept=[", join(keptSamplerNames), "] excludedAsLightingInputs=[",
          join(excludedSamplerNames), "]");
      }

      std::sort(info.uniformVectorRegisters.begin(), info.uniformVectorRegisters.end());

      std::sort(uniformsByName.begin(), uniformsByName.end());
      info.namedUniformFirstRegistersByNameOrder.reserve(uniformsByName.size());
      for (const auto& [uniformName, uniformRegister] : uniformsByName) {
        info.namedUniformFirstRegistersByNameOrder.emplace_back(
          XXH3_64bits(uniformName.data(), uniformName.size()), uniformRegister);
      }

      // name order is deterministic and identical across permutations, and name keys keep the
      // streamed identity aligned even if a permutation strips an unreferenced symbol
      std::sort(info.materialSamplersByNameOrder.begin(), info.materialSamplersByNameOrder.end());

      // canonical shader signature: name-sorted material sampler declarations, minus the
      // lighting inputs the simple-lightmap compile would not have declared. Engine symbols are
      // excluded wholesale (the lightmap policy permutations reference different engine
      // constants, e.g. AmbientColorAndSkyFactor only outside SIMPLE_LIGHTING), and register
      // indices and counts are excluded because both shift with the lightmap sampler count and
      // with fxc's per-permutation element trimming. Constant-color materials have no material
      // samplers, so their signature falls back to the uniform declarations.
      info.canonicalShaderSignature = hashUe3SignatureEntries(
        !samplerSignatureEntries.empty() ? samplerSignatureEntries : uniformSignatureEntries);

      auto mergeConstRanges = [](Ue3MaterialConstRanges& ranges) {
        if (ranges.empty())
          return;
        std::sort(ranges.begin(), ranges.end());
        Ue3MaterialConstRanges merged;
        for (const auto& r : ranges) {
          if (merged.empty() || r.first > merged.back().first + merged.back().second) {
            merged.push_back(r);
          } else {
            const uint32_t end = std::max(merged.back().first + merged.back().second, r.first + r.second);
            merged.back().second = end - merged.back().first;
          }
        }
        ranges = std::move(merged);
      };
      mergeConstRanges(info.constRanges);
      return info;
    }

    // Reused XXH3 streaming state: created once per thread instead of heap
    // allocating/freeing a state per hash operation on the per-draw path. The state is
    // fully reset before each use, so digests are identical to a fresh state.
    static XXH3_state_t* getThreadLocalXxh3State() {
      static thread_local XXH3_state_t* const state = XXH3_createState();
      return state;
    }

    // Returns kEmptyHash when ranges is empty: without CTAB info, a raw register-range
    // fallback would fold per-view/per-mesh constants into the hash.
    static XXH64_hash_t hashUe3MaterialConstants(
        const Vector4* fConsts,
        const Ue3MaterialConstRanges& ranges) {
      if (ranges.empty())
        return kEmptyHash;

      XXH3_state_t* const state = getThreadLocalXxh3State();
      if (state == nullptr)
        return kEmptyHash;
      XXH3_64bits_reset(state);

      bool anyRegisterHashed = false;
      for (const auto& [start, count] : ranges) {
        if (start + count > caps::MaxFloatConstantsPS)
          continue;
        XXH3_64bits_update(state, &fConsts[start], count * sizeof(Vector4));
        anyRegisterHashed = true;
      }

      return anyRegisterHashed ? XXH3_64bits_digest(state) : kEmptyHash;
    }

    // Permutation-invariant constants identity: streams (name key, leading element value) of
    // each named Uniform* constant, in name order. fxc trims each uniform array to the
    // elements the permutation actually references (the directional lightmap path can
    // reference more expression elements than the simple path, e.g. an unreferenced specular
    // expression), so higher elements are not comparable across lightmap policy permutations.
    // The leading element is always within the reported range and the engine uploads the same
    // expression value to it in every permutation. Name keys keep the stream aligned even if
    // a permutation strips an entire unreferenced uniform.
    static XXH64_hash_t hashUe3MaterialConstantsByNameOrder(
        const Vector4* fConsts,
        const std::vector<std::pair<XXH64_hash_t, uint32_t>>& namedUniformFirstRegistersByNameOrder) {
      if (namedUniformFirstRegistersByNameOrder.empty())
        return kEmptyHash;

      XXH3_state_t* const state = getThreadLocalXxh3State();
      if (state == nullptr)
        return kEmptyHash;
      XXH3_64bits_reset(state);

      bool anyRegisterHashed = false;
      for (const auto& [nameKey, reg] : namedUniformFirstRegistersByNameOrder) {
        if (reg >= caps::MaxFloatConstantsPS)
          continue;
        XXH3_64bits_update(state, &nameKey, sizeof(nameKey));
        XXH3_64bits_update(state, &fConsts[reg], sizeof(Vector4));
        anyRegisterHashed = true;
      }

      return anyRegisterHashed ? XXH3_64bits_digest(state) : kEmptyHash;
    }

    static fast_unordered_cache<Ue3PsMaterialIdentityInfo> s_ue3PsMaterialIdentityCache;

    static const Ue3PsMaterialIdentityInfo& getOrParseUe3PsMaterialIdentityInfo(
        const XXH64_hash_t psHash,
        const std::vector<uint8_t>& bytecode,
        const D3D9CommonShader* pixelShader) {
      auto it = s_ue3PsMaterialIdentityCache.find(psHash);
      if (it == s_ue3PsMaterialIdentityCache.end()) {
        it = s_ue3PsMaterialIdentityCache.emplace(
          psHash, parseUe3PsMaterialIdentityFromCtab(bytecode, pixelShader)).first;
      }
      return it->second;
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
        markName(toLowerAscii(c.name), isSampler, isFloat4, c.registerIndex);
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
  D3D9Rtx::Ue3InstancingInfo D3D9Rtx::resolveUe3Instancing(
      const D3D9VertexElements& elements,
      const std::array<UINT, caps::MaxStreams>& streamFreq,
      const uint32_t instanceCount) {
    Ue3InstancingInfo info;

    // D3D9 only honours a stream-0 instance count when some stream the declaration reads is
    // tagged as instance data; GenerateDrawInfo gates the replayed draw the same way, so the
    // scene has to agree or it would decompose a draw the hardware runs once.
    uint32_t usedStreamMask = 0;
    for (const auto& element : elements) {
      if (element.Stream < caps::MaxStreams) {
        usedStreamMask |= 1u << element.Stream;
      }
    }

    for (const uint32_t s : bit::BitMask(usedStreamMask)) {
      if ((streamFreq[s] & D3DSTREAMSOURCE_INSTANCEDATA) != 0) {
        info.instanceDataStreamMask |= 1u << s;
      }
    }

    if (info.instanceDataStreamMask == 0) {
      return info;
    }
    info.instanceCount = std::max(instanceCount, 1u);

    // A complete basis is all four elements, FLOAT3, on one and the same instance-data stream.
    // Anything short of that is some other instancing scheme and must not be decomposed.
    const D3DVERTEXELEMENT9* basis[4] = {};
    for (const auto& element : elements) {
      if (element.Usage != D3DDECLUSAGE_TEXCOORD ||
          element.UsageIndex < 1 || element.UsageIndex > 4 ||
          element.Type != D3DDECLTYPE_FLOAT3 ||
          element.Stream >= caps::MaxStreams ||
          (info.instanceDataStreamMask & (1u << element.Stream)) == 0) {
        continue;
      }
      basis[element.UsageIndex - 1] = &element;
    }

    if (basis[0] == nullptr || basis[1] == nullptr || basis[2] == nullptr || basis[3] == nullptr) {
      return info;
    }
    if (basis[1]->Stream != basis[0]->Stream ||
        basis[2]->Stream != basis[0]->Stream ||
        basis[3]->Stream != basis[0]->Stream) {
      return info;
    }

    info.hasInstanceTransform = true;
    info.transformStream = basis[0]->Stream;
    info.offsetByteOffset = basis[0]->Offset;
    info.axisByteOffsets[0] = basis[1]->Offset;
    info.axisByteOffsets[1] = basis[2]->Offset;
    info.axisByteOffsets[2] = basis[3]->Offset;
    return info;
  }

  bool D3D9Rtx::readUe3InstanceTransforms(const VertexContext vertexContext[caps::MaxStreams],
                                          std::vector<Ue3DecomposedInstance>& instances,
                                          const char** outReason) const {
    instances.clear();

    auto fail = [&](const char* reason) {
      if (outReason != nullptr) {
        *outReason = reason;
      }
      return false;
    };
    if (outReason != nullptr) {
      *outReason = "";
    }

    const Ue3InstancingInfo& info = m_currentUe3Instancing;
    if (!info.hasInstanceTransform || info.instanceCount == 0) {
      return fail("no per-instance transform basis in the declaration");
    }

    const VertexContext& ctx = vertexContext[info.transformStream];
    if (ctx.stride == 0) {
      return fail("instance stream has no stride");
    }
    if (ctx.mappedSlice.mapPtr == nullptr) {
      return fail("instance stream is not host-visible");
    }

    // The four FLOAT3 reads have to stay inside the record, and the whole array inside the slice.
    uint32_t maxByteOffset = info.offsetByteOffset;
    for (const uint32_t axisOffset : info.axisByteOffsets) {
      maxByteOffset = std::max(maxByteOffset, axisOffset);
    }
    if (maxByteOffset + sizeof(float) * 3 > ctx.stride) {
      return fail("instance basis does not fit the instance stream's stride");
    }

    const size_t firstByte = size_t(ctx.offset);
    const size_t requiredBytes = size_t(ctx.stride) * info.instanceCount;
    if (firstByte + requiredBytes > ctx.mappedSlice.length) {
      return fail("instance stream is shorter than the draw's instance count");
    }

    const uint8_t* records = static_cast<const uint8_t*>(ctx.mappedSlice.mapPtr) + firstByte;
    auto readVector3 = [](const uint8_t* record, const uint32_t byteOffset) {
      float v[3];
      std::memcpy(v, record + byteOffset, sizeof(v));
      return Vector3(v[0], v[1], v[2]);
    };

    instances.reserve(info.instanceCount);
    for (uint32_t i = 0; i < info.instanceCount; i++) {
      const uint8_t* record = records + size_t(ctx.stride) * i;
      const Vector3 offset = readVector3(record, info.offsetByteOffset);
      const Vector3 xAxis = readVector3(record, info.axisByteOffsets[0]);
      const Vector3 yAxis = readVector3(record, info.axisByteOffsets[1]);
      const Vector3 zAxis = readVector3(record, info.axisByteOffsets[2]);

      // PhysX only writes the live prefix of an NxFluid emitter's instance buffer, so trailing
      // records hold whatever the previous frame left - or nothing at all on a fresh allocation.
      // A zero or non-finite basis is not a placement.
      const bool finite =
        std::isfinite(offset.x) && std::isfinite(offset.y) && std::isfinite(offset.z) &&
        std::isfinite(xAxis.x) && std::isfinite(xAxis.y) && std::isfinite(xAxis.z) &&
        std::isfinite(yAxis.x) && std::isfinite(yAxis.y) && std::isfinite(yAxis.z) &&
        std::isfinite(zAxis.x) && std::isfinite(zAxis.y) && std::isfinite(zAxis.z);
      if (!finite) {
        continue;
      }
      if (std::abs(dot(xAxis, cross(yAxis, zAxis))) <= 1e-12f) {
        continue;
      }

      // Row-vector layout to match Remix's Matrix4 (basis in rows 0..2, translation in row 3),
      // identical to the FMatrix(XAxis, YAxis, ZAxis, Location) the engine's non-instanced
      // NxFluid mesh path hands to FMeshElement::LocalToWorld.
      Ue3DecomposedInstance instance;
      instance.instanceToObject[0] = Vector4(xAxis.x, xAxis.y, xAxis.z, 0.0f);
      instance.instanceToObject[1] = Vector4(yAxis.x, yAxis.y, yAxis.z, 0.0f);
      instance.instanceToObject[2] = Vector4(zAxis.x, zAxis.y, zAxis.z, 0.0f);
      instance.instanceToObject[3] = Vector4(offset.x, offset.y, offset.z, 1.0f);
      // The buffer position, not the position in this vector: a degenerate record earlier in the
      // buffer must not shift the identity of everything after it.
      instance.sourceIndex = i;
      instances.push_back(instance);
    }

    if (instances.empty()) {
      return fail("every instance's basis was degenerate");
    }
    return true;
  }

  // Whatever these bounds drop, the kept set has to be the same set next frame: an instance that
  // comes and goes as the camera moves flickers, and a selection that reorders the survivors also
  // renames them (see Ue3DecomposedInstance::sourceIndex). Hence the count clamp keeps the lowest
  // source indices rather than the nearest to the camera. Distance culling is view-dependent by
  // definition and can pop at its boundary, which is why it is opt-in.
  void D3D9Rtx::cullAndClampUe3InstanceTransforms(std::vector<Ue3DecomposedInstance>& instances,
                                                  uint32_t& outCulledByDistance,
                                                  uint32_t& outCulledByBudget) const {
    outCulledByDistance = 0;
    outCulledByBudget = 0;

    const uint32_t budget = std::max(m_frameOptions.ue3MaxDecomposedInstances, 1u);
    const float cullDistance = m_frameOptions.ue3DecomposedInstanceCullDistance;
    if (cullDistance <= 0.0f && instances.size() <= budget) {
      return;
    }

    if (cullDistance > 0.0f) {
      const DrawCallTransforms& transformData = m_activeDrawCallState.transformData;

      // worldToView is affine, so its inverse's translation row is the view origin in world space.
      // A camera that cannot be inverted leaves distances meaningless; skip to the count clamp.
      const double det = determinant(transformData.worldToView);
      if (std::isfinite(det) && std::abs(det) > 1e-24) {
        const Matrix4 viewToWorld = inverseAffine(transformData.worldToView);
        const Vector3 cameraPosition(viewToWorld[3].x, viewToWorld[3].y, viewToWorld[3].z);
        if (std::isfinite(cameraPosition.x) && std::isfinite(cameraPosition.y) &&
            std::isfinite(cameraPosition.z)) {
          const float cullDistanceSq = cullDistance * cullDistance;
          const size_t before = instances.size();
          // Order-preserving, so the surviving instances keep their relative buffer order.
          instances.erase(
            std::remove_if(instances.begin(), instances.end(),
                           [&](const Ue3DecomposedInstance& instance) {
                             // Carried through the draw's object transform, which is identity for
                             // UE3's instanced factories but composed anyway to stay general.
                             const Vector4 world = transformData.objectToWorld *
                               Vector4(instance.instanceToObject[3].x,
                                       instance.instanceToObject[3].y,
                                       instance.instanceToObject[3].z, 1.0f);
                             const Vector3 delta = Vector3(world.x, world.y, world.z) - cameraPosition;
                             return dot(delta, delta) > cullDistanceSq;
                           }),
            instances.end());
          outCulledByDistance = uint32_t(before - instances.size());
        }
      }
    }

    if (instances.size() > budget) {
      outCulledByBudget = uint32_t(instances.size() - budget);
      instances.resize(budget);
    }
  }

  // Names the batch an instanced draw belongs to, without involving any transform.
  //
  // The mesh streams say which mesh is being instanced but not which component is instancing it, and
  // two piles of the same debris share them. The instance buffer would distinguish those but cannot
  // serve as an identity - RenderNxFluidInstanced creates a fresh one per frame unless its pool hands
  // one back - so batches are matched to the previous frame's by continuity of their own centroid,
  // which holds because a batch as a whole barely moves even while its instances do.
  XXH64_hash_t D3D9Rtx::resolveUe3InstancedBatchKey(const RasterGeometry& geoData,
                                                    const std::vector<Ue3DecomposedInstance>& instances) {
    if (instances.empty()) {
      return kEmptyHash;
    }

    struct MeshKey {
      const void* pVertexDecl;
      const void* pVertexBuffer0;
      const void* pIndexBuffer;
      uint32_t vertexCount;
      uint32_t indexCount;
    };
    const MeshKey meshKey = {
      d3d9State().vertexDecl.ptr(),
      d3d9State().vertexBuffers[0].vertexBuffer.ptr(),
      d3d9State().indices.ptr(),
      geoData.vertexCount,
      geoData.indexCount,
    };
    const XXH64_hash_t meshHash = XXH3_64bits(&meshKey, sizeof(meshKey));

    Vector3 centroid(0.f, 0.f, 0.f);
    for (const Ue3DecomposedInstance& instance : instances) {
      centroid += Vector3(instance.instanceToObject[3].x,
                          instance.instanceToObject[3].y,
                          instance.instanceToObject[3].z);
    }
    centroid *= 1.0f / float(instances.size());

    const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();
    std::vector<Ue3InstancedBatchRecord>& records = m_ue3InstancedBatches[meshHash];

    // Retire batches that have not been drawn for a while so a level change cannot leave a stale
    // record that a new batch of the same mesh would latch onto.
    constexpr uint32_t kRetireAfterFrames = 120;
    records.erase(
      std::remove_if(records.begin(), records.end(),
                     [&](const Ue3InstancedBatchRecord& record) {
                       return currentFrame - record.lastFrame > kRetireAfterFrames;
                     }),
      records.end());

    // A whole batch travelling this far in one frame is a different batch, not the same one moved.
    constexpr float kMaxCentroidDriftSqr = 1000.f * 1000.f;
    Ue3InstancedBatchRecord* nearest = nullptr;
    float nearestDistSqr = kMaxCentroidDriftSqr;
    for (Ue3InstancedBatchRecord& record : records) {
      if (record.claimedFrame == currentFrame) {
        continue; // another draw already matched this batch this frame
      }
      const Vector3 delta = record.centroid - centroid;
      const float distSqr = dot(delta, delta);
      if (distSqr < nearestDistSqr) {
        nearestDistSqr = distSqr;
        nearest = &record;
      }
    }

    if (nearest == nullptr) {
      Ue3InstancedBatchRecord record;
      record.id = XXH3_64bits_withSeed(&m_ue3NextInstancedBatchId, sizeof(m_ue3NextInstancedBatchId), meshHash);
      ++m_ue3NextInstancedBatchId;
      records.push_back(record);
      nearest = &records.back();
    }

    nearest->centroid = centroid;
    nearest->lastFrame = currentFrame;
    nearest->claimedFrame = currentFrame;
    return nearest->id;
  }

  void D3D9Rtx::trackUe3InstanceOrderStability(const XXH64_hash_t batchKey,
                                               const std::vector<Ue3DecomposedInstance>& instances) {
    const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();

    // One entry per batch identity, each holding a translation per instance. Batch identities are
    // minted afresh across a level change, so drop the lot rather than grow without bound.
    constexpr size_t kMaxProbes = 64;
    if (m_ue3InstanceOrderProbes.size() > kMaxProbes) {
      m_ue3InstanceOrderProbes.clear();
    }

    Ue3InstanceOrderProbe& probe = m_ue3InstanceOrderProbes[batchKey];
    const bool consecutiveFrames = probe.lastFrame != 0 && probe.lastFrame + 1 == currentFrame;

    if (consecutiveFrames) {
      if (probe.translations.size() != instances.size()) {
        // A length change shifts every index past the first added or removed instance, so this
        // batch cannot be index-paired at all this frame.
        ++m_ue3InstancedStatOrderSizeChanges;
      } else {
        ++m_ue3InstancedStatOrderComparableBatches;
        const float stableThreshold = RtxOptions::uniqueObjectDistance();
        for (size_t i = 0; i < instances.size(); i++) {
          const Matrix4& m = instances[i].instanceToObject;
          const Vector3 current(m[3].x, m[3].y, m[3].z);
          const Vector3 delta = current - probe.translations[i];
          const float displacement = std::sqrt(dot(delta, delta));
          ++m_ue3InstancedStatOrderPairs;
          m_ue3InstancedStatOrderDisplacementSum += double(displacement);
          m_ue3InstancedStatOrderDisplacementMax =
            std::max(m_ue3InstancedStatOrderDisplacementMax, displacement);
          if (displacement <= stableThreshold) {
            ++m_ue3InstancedStatOrderStablePairs;
          }
        }
      }
    }

    probe.lastFrame = currentFrame;
    probe.translations.resize(instances.size());
    for (size_t i = 0; i < instances.size(); i++) {
      const Matrix4& m = instances[i].instanceToObject;
      probe.translations[i] = Vector3(m[3].x, m[3].y, m[3].z);
    }
  }

  void D3D9Rtx::reportUe3InstancedDrawStats() {
    if (!m_frameOptions.ue3LogInstancedDrawStats) {
      return;
    }

    ++m_ue3InstancedStatFrames;

    const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();

    // roughly once a second at any plausible frame rate; this is a diagnostic, not a metric
    constexpr uint32_t kStatIntervalFrames = 60;
    if (currentFrame - m_ue3InstancedStatFrameStamp < kStatIntervalFrames) {
      return;
    }
    m_ue3InstancedStatFrameStamp = currentFrame;

    const uint32_t frames = std::max(m_ue3InstancedStatFrames, 1u);
    const double perFrame = 1.0 / double(frames);

    Logger::info(str::format(
      "[RTX-Compatibility][UE3-Instanced] per frame over ", frames, " frames: ",
      double(m_ue3InstancedStatDraws) * perFrame, " instanced draws, ",
      double(m_ue3InstancedStatInstancesSeen) * perFrame, " hardware instances, ",
      double(m_ue3InstancedStatInstancesSubmitted) * perFrame, " submitted, ",
      double(m_ue3InstancedStatCulledDistance) * perFrame, " distance-culled, ",
      double(m_ue3InstancedStatCulledBudget) * perFrame, " budget-culled; ",
      double(m_ue3InstancedStatSubmitNs) * perFrame / 1.0e6, " ms/frame expanding them on the "
      "submitting thread (the rest of an instance's cost lands on the consumer thread and the GPU)."));

    // ue3StableDecomposedInstanceIdentity rests on the game keeping its instance order, so report how
    // far index-paired instances actually moved rather than taking that on trust.
    if (m_ue3InstancedStatOrderPairs > 0 || m_ue3InstancedStatOrderSizeChanges > 0) {
      const double meanDisplacement = m_ue3InstancedStatOrderPairs > 0
        ? m_ue3InstancedStatOrderDisplacementSum / double(m_ue3InstancedStatOrderPairs)
        : 0.0;
      const uint32_t stablePercent = m_ue3InstancedStatOrderPairs > 0
        ? uint32_t((m_ue3InstancedStatOrderStablePairs * 100ull) / m_ue3InstancedStatOrderPairs)
        : 0u;

      Logger::info(str::format(
        "[RTX-Compatibility][UE3-Instanced] instance order: ",
        m_ue3InstancedStatOrderComparableBatches, " batch comparisons over ", frames, " frames (",
        m_ue3InstancedStatOrderSizeChanges, " skipped on an instance count change), ",
        m_ue3InstancedStatOrderPairs, " index-paired instances, mean displacement ",
        meanDisplacement, " units, max ", m_ue3InstancedStatOrderDisplacementMax, ", ",
        stablePercent, "% within rtx.uniqueObjectDistance. A mean on the scale of the batch's own "
        "extent means the game reorders its instance buffer and index pairing is meaningless."));
    }

    m_ue3InstancedStatFrames = 0;
    m_ue3InstancedStatDraws = 0;
    m_ue3InstancedStatInstancesSeen = 0;
    m_ue3InstancedStatInstancesSubmitted = 0;
    m_ue3InstancedStatCulledDistance = 0;
    m_ue3InstancedStatCulledBudget = 0;
    m_ue3InstancedStatSubmitNs = 0;
    m_ue3InstancedStatOrderPairs = 0;
    m_ue3InstancedStatOrderStablePairs = 0;
    m_ue3InstancedStatOrderSizeChanges = 0;
    m_ue3InstancedStatOrderComparableBatches = 0;
    m_ue3InstancedStatOrderDisplacementSum = 0.0;
    m_ue3InstancedStatOrderDisplacementMax = 0.f;
  }

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

  const char* D3D9Rtx::describeUe3VertexFactory(const Ue3VertexFactoryType type) {
    switch (type) {
    case Ue3VertexFactoryType::Unknown: return "Unknown";
    case Ue3VertexFactoryType::Local: return "Local";
    case Ue3VertexFactoryType::GPUSkin: return "GPUSkin";
    case Ue3VertexFactoryType::GPUSkinMorph: return "GPUSkinMorph";
    case Ue3VertexFactoryType::Terrain: return "Terrain";
    case Ue3VertexFactoryType::TerrainMorph: return "TerrainMorph";
    case Ue3VertexFactoryType::Particle: return "Particle";
    case Ue3VertexFactoryType::ParticleBeamTrail: return "ParticleBeamTrail";
    case Ue3VertexFactoryType::SpeedTree: return "SpeedTree";
    case Ue3VertexFactoryType::Foliage: return "Foliage";
    case Ue3VertexFactoryType::ParticleInstancedMesh: return "ParticleInstancedMesh";
    case Ue3VertexFactoryType::LocalDecal: return "LocalDecal";
    case Ue3VertexFactoryType::LensFlare: return "LensFlare";
    case Ue3VertexFactoryType::PositionOnly: return "PositionOnly";
    }
    return "Unknown";
  }

  const char* D3D9Rtx::describeUe3PassType(const Ue3PassType type) {
    switch (type) {
    case Ue3PassType::Unknown: return "Unknown";
    case Ue3PassType::Material: return "Material";
    case Ue3PassType::DepthPrepass: return "DepthPrepass";
    case Ue3PassType::ShadowDepth: return "ShadowDepth";
    case Ue3PassType::Velocity: return "Velocity";
    case Ue3PassType::Lighting: return "Lighting";
    case Ue3PassType::ModulatedShadowProjection: return "ModulatedShadowProjection";
    case Ue3PassType::FullscreenPostProcess: return "FullscreenPostProcess";
    case Ue3PassType::UiComposite: return "UiComposite";
    case Ue3PassType::FogOrDistortion: return "FogOrDistortion";
    case Ue3PassType::VideoCinematic: return "VideoCinematic";
    case Ue3PassType::VideoSurface: return "VideoSurface";
    case Ue3PassType::SceneCapture: return "SceneCapture";
    }
    return "Unknown";
  }

  namespace {
    // IEEE 754 half -> float (no denormal flush; curve data never relies on
    // half denormals, but decode them correctly anyway)
    float ue3HalfToFloat(const uint16_t half) {
      const uint32_t sign = (half & 0x8000u) << 16;
      uint32_t exponent = (half & 0x7C00u) >> 10;
      uint32_t mantissa = half & 0x03FFu;

      if (exponent == 0) {
        if (mantissa == 0) {
          const uint32_t bits = sign;
          float result;
          std::memcpy(&result, &bits, sizeof(result));
          return result;
        }
        // subnormal half: normalize
        while ((mantissa & 0x0400u) == 0) {
          mantissa <<= 1;
          exponent--;
        }
        exponent++;
        mantissa &= ~0x0400u;
      } else if (exponent == 0x1F) {
        const uint32_t bits = sign | 0x7F800000u | (mantissa << 13);
        float result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
      }

      const uint32_t bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
      float result;
      std::memcpy(&result, &bits, sizeof(result));
      return result;
    }
  }

  void D3D9Rtx::onUe3CurveTextureUpload(const D3D9CommonTexture* dstTexture, D3D9CommonTexture* srcTexture, const uint32_t srcSubresource,
                                        const uint32_t srcTexelOffsetX, const uint32_t dstTexelOffsetX,
                                        const uint32_t texelWidth, const uint32_t texelHeight) {
    if (!m_frameOptions.ue3EngineMode || dstTexture == nullptr || srcTexture == nullptr) {
      return;
    }

    // The curve LUTs are 16x1 float RGBA textures (one texel per curve segment)
    const auto* desc = dstTexture->Desc();
    if (desc->Width != kUe3CurveTexelCount || desc->Height != 1) {
      return;
    }

    const D3D9Format format = desc->Format;
    if (format != D3D9Format::A32B32G32R32F && format != D3D9Format::A16B16G16R16F) {
      // Near-miss diagnostic: a 16x1 destination in an unexpected format would
      // mean the game's curve textures need another decode path.
      ONCE(Logger::info(str::format("[RTX-UE3-Tonemap] Ignoring 16x1 texture upload in unsupported format ",
                                    uint32_t(format), " (expected A32B32G32R32F/A16B16G16R16F).")));
      return;
    }

    if (texelHeight != 1 || texelWidth == 0 || dstTexelOffsetX >= kUe3CurveTexelCount) {
      return;
    }

    const void* srcData = srcTexture->GetMappedSlice(srcSubresource).mapPtr;
    if (srcData == nullptr) {
      return;
    }

    // Bounded cache: keys are only ever compared against currently-bound
    // textures (never dereferenced), so stale entries are harmless; still keep
    // the map tiny since only a handful of curve textures ever exist.
    if (m_ue3CurveTexelCache.size() > 16 && m_ue3CurveTexelCache.find(dstTexture) == m_ue3CurveTexelCache.end()) {
      m_ue3CurveTexelCache.clear();
    }
    Ue3CurveTexels& payload = m_ue3CurveTexelCache[dstTexture];

    const uint32_t count = std::min(texelWidth, kUe3CurveTexelCount - dstTexelOffsetX);

    if (format == D3D9Format::A32B32G32R32F) {
      const float* texels = reinterpret_cast<const float*>(srcData) + srcTexelOffsetX * 4;
      for (uint32_t i = 0; i < count; i++) {
        payload.texels[dstTexelOffsetX + i] = Vector4(texels[i * 4 + 0], texels[i * 4 + 1], texels[i * 4 + 2], texels[i * 4 + 3]);
      }
    } else {
      const uint16_t* texels = reinterpret_cast<const uint16_t*>(srcData) + srcTexelOffsetX * 4;
      for (uint32_t i = 0; i < count; i++) {
        payload.texels[dstTexelOffsetX + i] = Vector4(ue3HalfToFloat(texels[i * 4 + 0]),
                                                      ue3HalfToFloat(texels[i * 4 + 1]),
                                                      ue3HalfToFloat(texels[i * 4 + 2]),
                                                      ue3HalfToFloat(texels[i * 4 + 3]));
      }
    }

    ONCE(Logger::info(str::format("[RTX-UE3-Tonemap] Snooping curve LUT texture uploads (",
                                  format == D3D9Format::A32B32G32R32F ? "fp32" : "fp16",
                                  ", rect x=", dstTexelOffsetX, " w=", count, ").")));
  }

  void D3D9Rtx::maybeCaptureUe3ToneMapState() {
    if (m_ue3ToneMapCapturedThisFrame || !m_frameOptions.ue3EngineMode) {
      return;
    }

    // Only spend effort when the Mirror's Edge tonemapper consumes the capture
    if (RtxOptions::tonemappingMode() != TonemappingMode::MirrorsEdge) {
      return;
    }

    if (!m_parent->UseProgrammablePS() || d3d9State().pixelShader == nullptr) {
      return;
    }

    const Ue3ShaderFeatureInfo psInfo = getUe3ShaderFeatureInfo(d3d9State().pixelShader->GetCommonShader());

    // Only the TdToneMapping pass declares ColorCurvesK/M; require those sampler
    // indices so UberPostProcessBlend (grade+gamma only) cannot lock out capture.
    if (!psInfo.hasToneMapConstants || !psInfo.hasGammaConstants ||
        psInfo.toneMapSceneShadowsReg < 0 || psInfo.toneMapGammaColorScaleReg < 0 ||
        psInfo.toneMapMidTonesReg < 0 ||
        psInfo.colorCurvesKSamplerIndex >= caps::MaxTexturesPS ||
        psInfo.colorCurvesMSamplerIndex >= caps::MaxTexturesPS) {
      return;
    }

    Ue3ToneMapCapture capture;

    const auto readConstant = [&](const int16_t reg, Vector4& target) {
      if (reg >= 0 && reg < int16_t(caps::MaxFloatConstantsPS)) {
        target = d3d9State().psConsts.fConsts[reg];
      }
    };
    readConstant(psInfo.toneMapSceneShadowsReg, capture.sceneShadowsAndDesaturation);
    readConstant(psInfo.toneMapInverseHighLightsReg, capture.sceneInverseHighLights);
    readConstant(psInfo.toneMapMidTonesReg, capture.sceneMidTones);
    readConstant(psInfo.toneMapScaledLumaWeightsReg, capture.sceneScaledLuminanceWeights);
    readConstant(psInfo.toneMapGammaColorScaleReg, capture.gammaColorScaleAndInverse);
    readConstant(psInfo.toneMapGammaOverlayReg, capture.gammaOverlayColor);
    capture.hasConstants = true;

    const auto resolveCurveTexels = [&](const uint8_t samplerIndex, std::array<Vector4, kUe3CurveTexelCount>& target) {
      if (samplerIndex >= caps::MaxTexturesPS) {
        return false;
      }
      IDirect3DBaseTexture9* texture = d3d9State().textures[samplerIndex];
      if (texture == nullptr) {
        return false;
      }
      const auto it = m_ue3CurveTexelCache.find(GetCommonTexture(texture));
      if (it == m_ue3CurveTexelCache.end()) {
        return false;
      }
      target = it->second.texels;
      return true;
    };

    capture.hasCurves = resolveCurveTexels(psInfo.colorCurvesKSamplerIndex, capture.curveK) &&
                        resolveCurveTexels(psInfo.colorCurvesMSamplerIndex, capture.curveM);

    // Capture the sampler filter bound for the curve LUTs so the tonemapper's
    // lookup matches the game exactly (point snaps to one segment texel,
    // bilinear blends adjacent ones).
    if (psInfo.colorCurvesKSamplerIndex < caps::MaxTexturesPS) {
      const DWORD magFilter = d3d9State().samplerStates[psInfo.colorCurvesKSamplerIndex][D3DSAMP_MAGFILTER];
      capture.curvePointFiltering = (magFilter == D3DTEXF_POINT || magFilter == D3DTEXF_NONE);
      if (capture.curvePointFiltering) {
        ONCE(Logger::info("[RTX-UE3-Tonemap] Game samples its curve LUTs with point filtering; matching in the tonemapper."));
      }
    }

    m_ue3ToneMapCapturedThisFrame = true;

    ONCE(Logger::info(str::format("[RTX-UE3-Tonemap] Capturing TdToneMapping pass state (curve textures resolved: ",
                                  capture.hasCurves ? "yes" : "no", ")")));

    if (!capture.hasCurves) {
      // Pinpoint why curve resolution failed: missing CTAB sampler indices,
      // nothing bound at the slots, or no snooped upload for the bound texture.
      const auto describeCurveSampler = [&](const uint8_t samplerIndex) -> std::string {
        if (samplerIndex >= caps::MaxTexturesPS) {
          return "sampler not in CTAB";
        }
        IDirect3DBaseTexture9* texture = d3d9State().textures[samplerIndex];
        if (texture == nullptr) {
          return str::format("slot ", uint32_t(samplerIndex), ": no texture bound");
        }
        const auto* common = GetCommonTexture(texture);
        const auto* desc = common->Desc();
        const bool snooped = m_ue3CurveTexelCache.find(common) != m_ue3CurveTexelCache.end();
        return str::format("slot ", uint32_t(samplerIndex), ": ", desc->Width, "x", desc->Height,
                           " fmt=", uint32_t(desc->Format), snooped ? " [snooped]" : " [no snooped upload]");
      };
      ONCE(Logger::warn(str::format("[RTX-UE3-Tonemap] Curve textures unresolved at tonemap pass. K={",
                                    describeCurveSampler(psInfo.colorCurvesKSamplerIndex), "} M={",
                                    describeCurveSampler(psInfo.colorCurvesMSamplerIndex), "}")));
    }

    m_parent->EmitCs([capture](DxvkContext* ctx) {
      static_cast<RtxContext*>(ctx)->setUe3ToneMapCapture(capture);
    });
  }

  const char* D3D9Rtx::describeGeometryStatus(const RtxGeometryStatus status) {
    switch (status) {
    case RtxGeometryStatus::Ignored: return "Ignored";
    case RtxGeometryStatus::Rasterized: return "Rasterized";
    case RtxGeometryStatus::RayTraced: return "RayTraced";
    }
    return "Unknown";
  }

  void D3D9Rtx::logUe3Classification(const DrawContext& drawContext,
                                     const Ue3PassType passType,
                                     const RtxGeometryStatus status,
                                     const char* reason) {
    // remember the routing decision for the draw-status flap probe regardless of log level
    m_ue3LastDrawDecision = reason;

    if (!m_frameOptions.ue3LogClassification && Logger::logLevel() > LogLevel::Debug)
      return;

    XXH64_hash_t vsHash = 0;
    if (m_parent->UseProgrammableVS() && d3d9State().vertexShader.ptr() != nullptr) {
      vsHash = d3d9State().vertexShader->GetCommonShader()->GetBytecodeHash();
    }
    XXH64_hash_t psHash = 0;
    if (m_parent->UseProgrammablePS() && d3d9State().pixelShader.ptr() != nullptr) {
      psHash = d3d9State().pixelShader->GetCommonShader()->GetBytecodeHash();
    }

    uint32_t rtWidth = 0;
    uint32_t rtHeight = 0;
    if (d3d9State().renderTargets[kRenderTargetIndex] != nullptr) {
      const auto& rtExt = d3d9State().renderTargets[kRenderTargetIndex]->GetSurfaceExtent();
      rtWidth = rtExt.width;
      rtHeight = rtExt.height;
    }

    Logger::debug(str::format(
      "[RTX-Compatibility][UE3] draw=", m_activeDrawCallState.drawCallID,
      " vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
      " pass=", describeUe3PassType(passType),
      " status=", describeGeometryStatus(status),
      " prims=", drawContext.PrimitiveCount,
      " vsHash=0x", std::hex, vsHash,
      " psHash=0x", psHash, std::dec,
      " rt=", rtWidth, "x", rtHeight,
      " fg=", m_ue3ForegroundDpgActive ? 1 : 0,
      " reason=", reason));
  }

  namespace {
    // Session registry of UE3 lightmap textures recognised from pixel shader CTAB sampler
    // names. Append-only, and read from texture paths that run on other threads than the
    // draw-call setup that discovers them.
    std::shared_mutex g_autoLightmapTexturesMutex;
    fast_unordered_set g_autoLightmapTextures;
    // Fast negative for the common case of a title that never discovers any.
    std::atomic<bool> g_autoLightmapTexturesPopulated { false };
  }

  bool D3D9Rtx::isAutoDetectedLightmapTexture(const XXH64_hash_t textureHash) {
    if (textureHash == kEmptyHash || !g_autoLightmapTexturesPopulated.load(std::memory_order_acquire))
      return false;

    std::shared_lock lock(g_autoLightmapTexturesMutex);
    return g_autoLightmapTextures.find(textureHash) != g_autoLightmapTextures.end();
  }

  void D3D9Rtx::registerAutoDetectedLightmapTexture(const XXH64_hash_t textureHash) {
    if (textureHash == kEmptyHash)
      return;

    bool inserted = false;
    {
      std::unique_lock lock(g_autoLightmapTexturesMutex);
      inserted = g_autoLightmapTextures.insert(textureHash).second;
      if (inserted)
        g_autoLightmapTexturesPopulated.store(true, std::memory_order_release);
    }

    if (inserted)
      ImGUI::AddAutoTaggedLightmapTexture(textureHash);
  }

  void D3D9Rtx::OnClear(DWORD flags) {
    if (!m_frameOptions.ue3EngineMode || !m_frameOptions.ue3ForegroundDpgIsViewModel) {
      return;
    }

    // UE3 renders SDPG_Foreground (first-person arms, held weapon, muzzle flash) after the
    // world DPG behind a depth-only clear so foreground meshes never depth-clash with the
    // world. A z-only clear on a main-view-sized viewport after this frame's world draws
    // marks every subsequent draw as foreground until EndFrame.
    if ((flags & D3DCLEAR_TARGET) != 0 || (flags & D3DCLEAR_ZBUFFER) == 0) {
      return;
    }
    if (m_ue3ForegroundDpgActive || !m_ue3SeenMainViewWorldDraw) {
      return;
    }
    if (d3d9State().depthStencil == nullptr || !m_activePresentParams.has_value()) {
      return;
    }

    // Shadow / SceneCapture depth clears use sub-main-view viewports; ignore those.
    const D3DVIEWPORT9& vp = d3d9State().viewport;
    const uint32_t bbW = m_activePresentParams->BackBufferWidth;
    const uint32_t bbH = m_activePresentParams->BackBufferHeight;
    if (!ue3ViewportIsMainViewSized(vp.Width, vp.Height, bbW, bbH)) {
      return;
    }

    m_ue3ForegroundDpgActive = true;
    ONCE(Logger::info("[RTX-Compatibility-Info] UE3 foreground DPG boundary detected (mid-scene depth-only clear); subsequent draws classify as ViewModel."));
    if (m_frameOptions.ue3LogClassification) {
      Logger::debug(str::format("[RTX-Compatibility][UE3] Foreground DPG boundary after draw=", m_activeDrawCallState.drawCallID));
    }
  }

  bool D3D9Rtx::trackUe3MovieTextureRenderTarget(const char* reason) {
    if (d3d9State().renderTargets[kRenderTargetIndex] == nullptr ||
        !m_activePresentParams.has_value()) {
      return false;
    }

    D3D9CommonTexture* rtTexture = GetCommonTexture(d3d9State().renderTargets[kRenderTargetIndex]->GetBaseTexture());
    if (rtTexture == nullptr || rtTexture->GetImage() == nullptr || rtTexture->Desc() == nullptr) {
      return false;
    }

    if (isRenderTargetPrimary(*m_activePresentParams, rtTexture->Desc())) {
      return false;
    }

    const XXH64_hash_t descHash = rtTexture->GetImage()->getDescriptorHash();
    if (descHash == kEmptyHash) {
      return false;
    }

    const bool inserted = m_ue3MovieTextureDescHashes.insert(descHash).second;
    if (inserted && Logger::logLevel() <= LogLevel::Debug) {
      const auto* desc = rtTexture->Desc();
      Logger::debug(str::format(
        "[RTX-Compatibility][UE3] Tracked movie texture render target: ",
        desc->Width, "x", desc->Height,
        ", descHash=0x", std::hex, descHash, std::dec,
        ", reason=", reason));
    }
    return true;
  }

  bool D3D9Rtx::isUe3MovieTextureDescHash(const XXH64_hash_t descHash) const {
    return descHash != kEmptyHash &&
           m_ue3MovieTextureDescHashes.find(descHash) != m_ue3MovieTextureDescHashes.end();
  }

  D3D9Rtx::D3D9Rtx(D3D9DeviceEx* d3d9Device, bool enableDrawCallConversion)
    : m_rtStagingData(d3d9Device->GetDXVKDevice(), "RtxStagingDataAlloc: D3D9", (VkMemoryPropertyFlagBits) (VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
    , m_parent(d3d9Device)
    , m_enableDrawCallConversion(enableDrawCallConversion)
    , m_pGeometryWorkers(enableDrawCallConversion ? std::make_unique<GeometryProcessor>(numGeometryProcessingThreads(), "geometry-processing") : nullptr) {
  }

  D3D9Rtx::~D3D9Rtx() {
    // EndFrame saves on an interval, so the tail of a session would otherwise be lost.
    saveUe3TextureSpreadCache();
    saveUe3DiffuseSelectionCache();
  }

  void D3D9Rtx::SkinningMatrixPool::clear() {
    m_blockIndex = 0;
    m_nextIndexInBlock = 0;
  }

  const Matrix4* D3D9Rtx::SkinningMatrixPool::stageBones(const Matrix4* source, size_t matrixCount) {
    assert(matrixCount <= kMatricesPerBlock);
    if (matrixCount == 0) {
      return nullptr;
    }
    for (;;) {
      if (m_blocks.empty()) {
        m_blocks.push_back(std::make_unique<Block>());
        m_blockIndex = 0;
        m_nextIndexInBlock = 0;
      } else if (m_nextIndexInBlock + matrixCount > kMatricesPerBlock) {
        ++m_blockIndex;
        m_nextIndexInBlock = 0;
        if (m_blockIndex == m_blocks.size()) {
          m_blocks.push_back(std::make_unique<Block>());
        }
        continue;
      }
      break;
    }
    Matrix4* const dest = m_blocks[m_blockIndex]->m_matrices.data() + m_nextIndexInBlock;
    memcpy(dest, source, matrixCount * sizeof(Matrix4));
    m_nextIndexInBlock += matrixCount;
    return dest;
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
    o.allowCubemaps = allowCubemapsObject().get();
    o.useVertexCapture = useVertexCaptureObject().get();
    o.useVertexCapturedNormals = useVertexCapturedNormalsObject().get();
    o.useWorldMatricesForShaders = useWorldMatricesForShadersObject().get();
    o.ue3EngineMode = ue3EngineModeObject().get();
    o.ue3CameraFromShaderConstants = ue3CameraFromShaderConstantsObject().get();
    o.ue3ObjectToWorldFromShaderConstants = ue3ObjectToWorldFromShaderConstantsObject().get();
    o.autoRaytracedRenderTargetFromFullscreenComposite = autoRaytracedRenderTargetFromFullscreenCompositeObject().get();
    o.rasterizeFullscreenCompositeToPrimary = rasterizeFullscreenCompositeToPrimaryObject().get();
    o.shaderPathTexcoordIndexFromPixelShader = shaderPathTexcoordIndexFromPixelShaderObject().get();
    o.ue3MaterialInstanceConstantHash = ue3MaterialInstanceConstantHashObject().get();
    o.ue3MicConstantIdentity = ue3MicConstantIdentityObject().get();
    o.ue3MicExcludeRenderTargetsFromIdentity = ue3MicExcludeRenderTargetsFromIdentityObject().get();
    o.ue3MicPersistAutoExcludedConstantGroups = ue3MicPersistAutoExcludedConstantGroupsObject().get();
    o.ue3LogMaterialInstanceHash = ue3LogMaterialInstanceHashObject().get();
    o.ue3SkipDepthPrepass = ue3SkipDepthPrepassObject().get();
    o.ue3SkipShadowDepthPasses = ue3SkipShadowDepthPassesObject().get();
    o.ue3SkipDepthTestDisabledTranslucency = ue3SkipDepthTestDisabledTranslucencyObject().get();
    o.ue3SkipSceneCapturePasses = ue3SkipSceneCapturePassesObject().get();
    o.ue3ForegroundDpgIsViewModel = ue3ForegroundDpgIsViewModelObject().get();
    o.conservativeOcclusionQueries = conservativeOcclusionQueriesObject().get();
    o.ue3StaticLocalMeshVertexCaptureCache = ue3StaticLocalMeshVertexCaptureCacheObject().get();
    o.ue3StaticLocalMeshVertexCaptureCacheWarmupFrames = ue3StaticLocalMeshVertexCaptureCacheWarmupFramesObject().get();
    o.ue3StaticLocalMeshVertexCaptureCacheBudgetMiB = ue3StaticLocalMeshVertexCaptureCacheBudgetMiBObject().get();
    o.ue3StaticLocalMeshVertexCaptureCacheMaxEntries = ue3StaticLocalMeshVertexCaptureCacheMaxEntriesObject().get();
    o.ue3StaticLocalMeshVertexCaptureCacheRetentionFrames = ue3StaticLocalMeshVertexCaptureCacheRetentionFramesObject().get();
    o.ue3StaticLocalMeshVertexCaptureCacheMinReusePercent = ue3StaticLocalMeshVertexCaptureCacheMinReusePercentObject().get();
    o.ue3StaticLocalMeshVertexCaptureCacheReuseProbeFrames = ue3StaticLocalMeshVertexCaptureCacheReuseProbeFramesObject().get();
    o.ue3LogStaticVertexCaptureCacheStats = ue3LogStaticVertexCaptureCacheStatsObject().get();
    o.ue3ExcludePlacementFromVertexShaderHash = ue3ExcludePlacementFromVertexShaderHashObject().get();
    o.ue3LogVertexConstantChurn = ue3LogVertexConstantChurnObject().get();
    o.ue3VertexConstantChurnMaxTrackedDraws = ue3VertexConstantChurnMaxTrackedDrawsObject().get();
    o.ue3StaticGeometryHashMemoization = ue3StaticGeometryHashMemoizationObject().get();
    o.ue3ExactVertexCapture = ue3ExactVertexCaptureObject().get();
    o.ue3RequireExactVertexCapture = ue3RequireExactVertexCaptureObject().get();
    o.ue3VertexCaptureSourceOverride = ue3VertexCaptureSourceOverrideObject().get();
    o.ue3NativeLocalMeshVertexCapture = ue3NativeLocalMeshVertexCaptureObject().get();
    o.ue3DecomposeInstancedDraws = ue3DecomposeInstancedDrawsObject().get();
    o.ue3MaxDecomposedInstances = ue3MaxDecomposedInstancesObject().get();
    o.ue3DecomposedInstanceCullDistance = ue3DecomposedInstanceCullDistanceObject().get();
    o.ue3StableDecomposedInstanceIdentity = ue3StableDecomposedInstanceIdentityObject().get();
    o.ue3LogInstancedDrawStats = ue3LogInstancedDrawStatsObject().get();
    o.ue3RequireCtabCameraConstants = ue3RequireCtabCameraConstantsObject().get();
    o.ue3StableDiffuseSelection = ue3StableDiffuseSelectionObject().get();
    o.ue3AutoDetectLightmapTextures = ue3AutoDetectLightmapTexturesObject().get() && o.ue3EngineMode;
    o.ue3MicAutoExcludeFrameVaryingConstants = ue3MicAutoExcludeFrameVaryingConstantsObject().get();
    o.ue3LogClassification = ue3LogClassificationObject().get();
    o.ue3LogUvResolution = ue3LogUvResolutionObject().get();
    o.ue3LogUvAffineDetail = ue3LogUvAffineDetailObject().get();
    o.ue3LogAlbedoSelection = ue3LogAlbedoSelectionObject().get();
    o.ue3LogCapturePrecision = ue3LogCapturePrecisionObject().get();
    o.ue3LogInstancedDraws = ue3LogInstancedDrawsObject().get();
    o.ue3LogDrawStatusFlaps = ue3LogDrawStatusFlapsObject().get();
    o.ue3LogOcclusionQueries = ue3LogOcclusionQueriesObject().get();
    o.deferredUiReplay = deferredUiReplayObject().get();
    o.deferredUiRefreshSceneColor = deferredUiRefreshSceneColorObject().get();
    o.enableIndexBufferMemoization = enableIndexBufferMemoizationObject().get();

    o.enableRaytracing = RtxOptions::enableRaytracingObject().get();
    o.enableAlphaTest = RtxOptions::enableAlphaTestObject().get();
    o.enableAlphaBlend = RtxOptions::enableAlphaBlendObject().get();
    o.raytracedRenderTargetEnable = RtxOptions::RaytracedRenderTarget::enableObject().get();
    o.skipDrawCallsPostRTXInjection = RtxOptions::skipDrawCallsPostRTXInjectionObject().get();
    o.useBuffersDirectly = RtxOptions::useBuffersDirectlyObject().get();
    o.fogIgnoreSky = RtxOptions::fogIgnoreSkyObject().get();
    o.needsMeshBoundingBox = RtxOptions::needsMeshBoundingBox(); // derived helper, not an RtxOption
    o.validateCPUIndexData = RtxOptions::validateCPUIndexDataObject().get();
    o.alwaysCopyDecalGeometries = RtxOptions::alwaysCopyDecalGeometriesObject().get();
    o.terrainAsDecalsEnabledIfNoBaker = RtxOptions::terrainAsDecalsEnabledIfNoBakerObject().get();
    o.terrainAsDecalsAllowOverModulate = RtxOptions::terrainAsDecalsAllowOverModulateObject().get();
    o.enableMultiStageTextureFactorBlending = RtxOptions::enableMultiStageTextureFactorBlendingObject().get();
    o.ignoreAllVertexColorBakedLighting = RtxOptions::ignoreAllVertexColorBakedLightingObject().get();
    o.vertexColorIsBakedLighting = RtxOptions::vertexColorIsBakedLightingObject().get();
    o.logReplacementResolution = RtxOptions::logReplacementResolutionObject().get();
    o.drawCallRange = RtxOptions::drawCallRangeObject().get();

    o.uiTextures = &RtxOptions::uiTexturesObject().get();
    o.deferredUiTextures = &RtxOptions::deferredUiTexturesObject().get();
    o.deferredUiPixelShaders = &deferredUiPixelShadersObject().get();
    o.lightmapTextures = &RtxOptions::lightmapTexturesObject().get();
    o.neverAlbedoTextures = &RtxOptions::neverAlbedoTexturesObject().get();
    o.preferredAlbedoTextures = &RtxOptions::preferredAlbedoTexturesObject().get();
    o.smoothNormalsTextures = &RtxOptions::smoothNormalsTexturesObject().get();
    o.ignoreBakedLightingTextures = &RtxOptions::ignoreBakedLightingTexturesObject().get();
    o.raytracedRenderTargetTextures = &RtxOptions::raytracedRenderTargetTexturesObject().get();
    o.vsTexcoordCaptureOutlierTextures = &vsTexcoordCaptureOutlierTexturesObject().get();
    o.ue3MicConstantIdentityExcludedShaders = &ue3MicConstantIdentityExcludedShadersObject().get();
    o.ue3MicConstantIdentityExcludedGroups = &ue3MicConstantIdentityExcludedGroupsObject().get();
    o.ue3MicIdentityExcludedTextureDescHashes = &ue3MicIdentityExcludedTextureDescHashesObject().get();
    o.ue3TraceDrawTextureHashes = &ue3TraceDrawTextureHashesObject().get();
    o.replacementDebugHashes = &RtxOptions::replacementDebugHashesObject().get();

    o.valid = true;
  }

  const char* D3D9Rtx::describeUe3CapturePositionSource(const Ue3CapturePositionSource source) {
    switch (source) {
    case Ue3CapturePositionSource::ClipReconstruction: return "ClipReconstruction";
    case Ue3CapturePositionSource::InputAssembler: return "InputAssembler";
    case Ue3CapturePositionSource::PreProjectionRegister: return "PreProjectionRegister";
    }
    return "Unknown";
  }

  // Requires the vertex shader's oPos transform to have been recognised at compile time AND
  // its matrix register to be the one the CTAB names ViewProjectionMatrix. The analyzer only
  // proves "this register is multiplied by a matrix to make oPos"; the CTAB match is what
  // proves the register holds a world position rather than some factory-local space.
  bool D3D9Rtx::canUseUe3PreProjectionVertexCapture(const char** outReason) const {
    auto fail = [&](const char* reason) {
      if (outReason != nullptr) {
        *outReason = reason;
      }
      return false;
    };

    if (!m_frameOptions.ue3ExactVertexCapture) {
      return fail("exact capture disabled");
    }
    if (!m_frameOptions.ue3EngineMode) {
      return fail("UE3 engine mode disabled");
    }

    const D3D9CommonShader* vertexShader = GetCommonShader(d3d9State().vertexShader);
    if (vertexShader == nullptr) {
      return fail("no vertex shader");
    }

    const DxsoPreProjectionPositionInfo& preProj = vertexShader->GetPreProjectionPositionInfo();
    if (!preProj.valid) {
      // The analyzer's own reason is more specific than anything this layer could say.
      return fail(preProj.failureReason);
    }

    if (!m_currentUe3CtabInfo.has_value() || !m_currentUe3CtabInfo->hasViewProjectionMatrix) {
      return fail("CTAB declares no ViewProjectionMatrix");
    }
    if (preProj.matrixConstBase != m_currentUe3CtabInfo->viewProjectionMatrixRegisterIndex) {
      return fail("oPos matrix is not ViewProjectionMatrix");
    }

    if (outReason != nullptr) {
      *outReason = "";
    }
    return true;
  }

  Ue3CapturePositionSource D3D9Rtx::resolveUe3CapturePositionSource(
      const IndexContext& indexContext,
      const VertexContext vertexContext[caps::MaxStreams],
      const RasterGeometry& geoData,
      const char** outReason) const {
    const char* preProjReason = "";
    const bool canPreProj = canUseUe3PreProjectionVertexCapture(&preProjReason);
    const bool canInputAssembler = canUseUe3NativeLocalVertexCapture(indexContext, vertexContext, geoData);

    auto pick = [&](Ue3CapturePositionSource source, const char* reason) {
      if (outReason != nullptr) {
        *outReason = reason;
      }
      return source;
    };

    // Vertex capture cannot describe a hardware-instanced draw at all: its output buffer is
    // indexed by vertex, so every instance writes the same slots and the survivors are an
    // arbitrary mix of placements. The input assembler holds one object-space copy of the mesh,
    // which is the only self-consistent answer, and the placements come from the instance stream
    // instead (see readUe3InstanceTransforms). Overrides are deliberately not honoured here.
    if (m_currentUe3Instancing.instanceCount > 1) {
      const char* instancedReason = "";
      if (canUseUe3InstancedMeshVertexPositions(geoData, &instancedReason)) {
        return pick(Ue3CapturePositionSource::InputAssembler, "hardware-instanced draw");
      }
      return pick(Ue3CapturePositionSource::ClipReconstruction, instancedReason);
    }

    switch (m_frameOptions.ue3VertexCaptureSourceOverride) {
    case Ue3CapturePositionSourceOverride::ForcePreProjection:
      // A forced source that does not apply falls through rather than producing geometry in
      // the wrong space, so the override stays safe to leave on while investigating.
      if (canPreProj) {
        return pick(Ue3CapturePositionSource::PreProjectionRegister, "forced");
      }
      return pick(Ue3CapturePositionSource::ClipReconstruction, preProjReason);

    case Ue3CapturePositionSourceOverride::ForceInputAssembler:
      if (canInputAssembler) {
        return pick(Ue3CapturePositionSource::InputAssembler, "forced");
      }
      return pick(Ue3CapturePositionSource::ClipReconstruction, "not a conservative static local mesh");

    case Ue3CapturePositionSourceOverride::ForceReconstruction:
      return pick(Ue3CapturePositionSource::ClipReconstruction, "forced");

    case Ue3CapturePositionSourceOverride::Auto:
      break;
    }

    if (canPreProj) {
      return pick(Ue3CapturePositionSource::PreProjectionRegister, "");
    }
    if (canInputAssembler) {
      return pick(Ue3CapturePositionSource::InputAssembler, preProjReason);
    }
    return pick(Ue3CapturePositionSource::ClipReconstruction, preProjReason);
  }

  void D3D9Rtx::logUe3CapturePositionSource(const Ue3CapturePositionSource source, const char* reason) const {
    if (!m_frameOptions.ue3LogCapturePrecision) {
      return;
    }

    const D3D9CommonShader* vertexShader = GetCommonShader(d3d9State().vertexShader);
    const XXH64_hash_t vsHash = vertexShader != nullptr ? vertexShader->GetBytecodeHash() : kEmptyHash;

    // One line per shader per outcome: enough to enumerate everything that would be dropped
    // by ue3RequireExactVertexCapture without the volume of a per-draw log.
    static fast_unordered_set s_loggedSources;
    const XXH64_hash_t logKey =
      vsHash ^ (XXH64_hash_t(source) << 48) ^ (XXH64_hash_t(m_currentUe3VertexFactory) << 56);
    if (!s_loggedSources.insert(logKey).second) {
      return;
    }

    Logger::info(str::format(
      "[RTX-Compatibility][UE3-Capture] position source=", describeUe3CapturePositionSource(source),
      ", vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
      ", vs=0x", std::hex, vsHash, std::dec,
      (reason != nullptr && reason[0] != '\0') ? ", reason=" : "",
      (reason != nullptr) ? reason : ""));

    // On a fallback, follow up with what the shader's position math actually looked like -
    // a compiler is free to emit the same matrix multiply many ways, and this is the only
    // way to tell an unsupported shape apart from a detector bug without a debugger.
    if (source == Ue3CapturePositionSource::ClipReconstruction && vertexShader != nullptr) {
      const DxsoPreProjectionPositionInfo& preProj = vertexShader->GetPreProjectionPositionInfo();
      if (!preProj.positionDefinitions.empty()) {
        Logger::info(str::format(
          "[RTX-Compatibility][UE3-Capture]   vs=0x", std::hex, vsHash, std::dec,
          " oPos math: ", preProj.positionDefinitions));
      }
    }
  }

  // Static in the cross-frame sense: content only changes through an explicit upload,
  // which bumps the buffer's remixContentGeneration counter (part of every cache key).
  static bool isStaticD3D9Buffer(D3D9CommonBuffer* buffer) {
    if (buffer == nullptr || buffer->Desc() == nullptr) {
      return false;
    }

    if ((buffer->Desc()->Usage & D3DUSAGE_DYNAMIC) != 0 || buffer->WasWrittenByGPU()) {
      return false;
    }

    return !buffer->NeedsUpload();
  }

  bool D3D9Rtx::canUseUe3StaticVertexCaptureCache(const IndexContext& indexContext,
                                                  const VertexContext vertexContext[caps::MaxStreams],
                                                  const RasterGeometry& geoData) const {
    if (!m_frameOptions.ue3StaticLocalMeshVertexCaptureCache) {
      return false;
    }
    // Refusing here rather than at insertion also skips building the cache key, whose per-draw
    // stream hashing is most of the cost the cache exists to save.
    if (m_ue3VertexCaptureCacheDormant) {
      return false;
    }
    return isUe3StaticVertexCaptureCacheEligible(indexContext, vertexContext, geoData);
  }

  // The structural half of the test above: whether this draw is the kind the cache is for,
  // independent of whether the cache is enabled or has stood itself down. The constant-churn
  // diagnostic keys off this so it can explain a cache that is off or dormant.
  bool D3D9Rtx::isUe3StaticVertexCaptureCacheEligible(const IndexContext& indexContext,
                                                     const VertexContext vertexContext[caps::MaxStreams],
                                                     const RasterGeometry& geoData) const {
    if (!m_frameOptions.ue3EngineMode) {
      return false;
    }
    if (!m_parent->UseProgrammableVS()) {
      return false;
    }
    if (!m_frameOptions.useVertexCapture) {
      return false;
    }
    // Only exact sources are camera-independent, so only they can be reused across frames.
    // A cached clip-space reconstruction would pin the mesh to the reconstruction error of
    // whichever frame captured it, which is worse than recapturing every frame.
    if (!isUe3ExactCapturePositionSource(m_activeCapturePositionSource)) {
      return false;
    }
    // The cache key deliberately omits the camera constant registers, so it can only serve
    // factories whose world position is a function of the input assembler plus object and
    // bone constants. Camera-facing factories (sprites, billboards, leaf cards) and morphing
    // terrain move their vertices with the view and would be served a stale pose. Terrain is
    // also left out because a cache hit suppresses the original draw, which the terrain baker
    // needs to rasterize.
    switch (m_currentUe3VertexFactory) {
    case Ue3VertexFactoryType::Local:
    case Ue3VertexFactoryType::LocalDecal:
    case Ue3VertexFactoryType::GPUSkin:
    case Ue3VertexFactoryType::GPUSkinMorph:
    case Ue3VertexFactoryType::Foliage:
      break;
    default:
      return false;
    }
    // The skinned factories are on the list above because a skinned mesh held in a fixed pose is as
    // cacheable as a static one, but an animating one never is: its bone matrices are in the key's
    // stable VS-constant hash, so every pose mints a key that can never be hit again. Admission
    // would catch that, but a character animates essentially always, so refuse them up front rather
    // than pay a key and an admission record per draw per frame.
    if (geoData.blendWeightBuffer.defined() || geoData.blendIndicesBuffer.defined()) {
      return false;
    }
    // Instanced draws never capture in the first place; caching one would only pin a buffer that
    // is never consulted again.
    if (m_currentUe3Instancing.instanceCount > 1) {
      return false;
    }
    if (!m_currentUe3CtabInfo.has_value()) {
      return false;
    }
    if (!m_currentUe3CtabInfo->hasLocalToWorld) {
      return false;
    }
    if (!geoData.positionBuffer.defined()) {
      return false;
    }
    if (m_activeDrawCallState.testCategoryFlags(InstanceCategories::WorldUI) ||
        m_activeDrawCallState.testCategoryFlags(InstanceCategories::Terrain)) {
      return false;
    }
    if (d3d9State().vertexDecl == nullptr) {
      return false;
    }

    const auto& elements = d3d9State().vertexDecl->GetElements();

    if (indexContext.indexType != VK_INDEX_TYPE_NONE_KHR && !isStaticD3D9Buffer(indexContext.ibo)) {
      return false;
    }

    for (const auto& element : elements) {
      if (element.Stream >= caps::MaxStreams) {
        return false;
      }

      const VertexContext& ctx = vertexContext[element.Stream];
      if (ctx.mappedSlice.handle == VK_NULL_HANDLE || !isStaticD3D9Buffer(ctx.pVBO)) {
        return false;
      }
    }

    return true;
  }

  // Geometry hash/AABB memo eligibility: unlike the vertex-capture cache above, this only
  // requires the IA vertex/index content to be immutable across frames - any vertex factory
  // qualifies (a GPU-skinned mesh's bind-pose buffers are as static as a Local mesh's; only
  // its bone constants animate, and those live in the per-draw VertexShader hash component
  // which is recombined live rather than memoized).
  bool D3D9Rtx::canMemoizeUe3IaGeometryHashes(const IndexContext& indexContext,
                                              const VertexContext vertexContext[caps::MaxStreams],
                                              const RasterGeometry& geoData) const {
    if (!m_frameOptions.ue3StaticGeometryHashMemoization) {
      return false;
    }
    if (!m_frameOptions.ue3EngineMode) {
      return false;
    }
    // staging copies get a fresh physical slice every draw, so their keys never repeat and
    // memo entries would be dead weight
    if (m_forceGeometryCopy) {
      return false;
    }
    if (d3d9State().vertexDecl == nullptr) {
      return false;
    }
    if (!geoData.positionBuffer.defined()) {
      return false;
    }

    if (indexContext.indexType != VK_INDEX_TYPE_NONE_KHR && !isStaticD3D9Buffer(indexContext.ibo)) {
      return false;
    }

    for (const auto& element : d3d9State().vertexDecl->GetElements()) {
      if (element.Stream >= caps::MaxStreams) {
        return false;
      }

      // Instance-data streams are dynamic by nature and contribute nothing to the geometry the
      // memo describes; computeUe3IaGeometryMemoKey leaves them out of the key for the same reason.
      if ((m_currentUe3Instancing.instanceDataStreamMask & (1u << element.Stream)) != 0) {
        continue;
      }

      const VertexContext& ctx = vertexContext[element.Stream];
      if (ctx.mappedSlice.handle == VK_NULL_HANDLE || !isStaticD3D9Buffer(ctx.pVBO)) {
        return false;
      }
    }

    return true;
  }

  namespace {
    // Flat, padding-free records so the per-draw geometry identity keys hash in 2-3
    // XXH3 calls rather than one tiny seed-chained call per field: each
    // XXH3_64bits_withSeed call pays fixed setup/finalize costs that dwarf the mixing.
    struct Ue3KeyStreamRecord {
      uint64_t pVBO;
      uint64_t sliceHandle;
      uint64_t sliceOffset;
      uint64_t sliceLength;
      uint64_t contentGeneration;
      uint32_t stride;
      uint32_t offset;
    };
    static_assert(sizeof(Ue3KeyStreamRecord) == 48, "Ue3KeyStreamRecord must have no implicit padding (it is hashed by memory).");

    // D3D9 vertex declarations are bounded by MAXD3DDECLLENGTH (64); records are hashed
    // in bounded chunks anyway so larger inputs would simply chain across chunks.
    constexpr size_t kUe3KeyStreamRecordChunk = 64;

    // templated so the private D3D9Rtx::VertexContext type is deduced rather than named
    template<typename VertexContextT>
    Ue3KeyStreamRecord makeUe3KeyStreamRecord(const VertexContextT& ctx) {
      Ue3KeyStreamRecord record = {};
      record.pVBO = uint64_t(reinterpret_cast<uintptr_t>(ctx.pVBO));
      record.sliceHandle = uint64_t(reinterpret_cast<uintptr_t>(ctx.mappedSlice.handle));
      record.sliceOffset = uint64_t(ctx.mappedSlice.offset);
      record.sliceLength = uint64_t(ctx.mappedSlice.length);
      record.contentGeneration = ctx.pVBO != nullptr ? ctx.pVBO->remixContentGeneration : 0ull;
      record.stride = ctx.stride;
      record.offset = ctx.offset;
      return record;
    }

    // excludeStreamMask drops streams whose contents are not part of the identity being keyed.
    // Instance-data streams are the case that matters: they are reallocated and rewritten every
    // frame, so including them would mint a key that can never repeat, while none of their
    // contents reaches the geometry the key describes.
    template<typename ElementsT, typename VertexContextT>
    XXH64_hash_t hashUe3KeyStreamRecords(const ElementsT& elements,
                                         const VertexContextT* vertexContext,
                                         XXH64_hash_t seed,
                                         const uint32_t excludeStreamMask = 0) {
      // decl layout itself (semantics, formats, offsets, stream assignment)
      XXH64_hash_t hash = XXH3_64bits_withSeed(elements.data(), elements.size() * sizeof(D3DVERTEXELEMENT9), seed);

      // per-element stream identity
      std::array<Ue3KeyStreamRecord, kUe3KeyStreamRecordChunk> records;
      size_t recordCount = 0;
      for (const auto& element : elements) {
        // stream bounds are pre-validated by the canUse/canMemoize eligibility gates
        assert(element.Stream < caps::MaxStreams);
        if ((excludeStreamMask & (1u << element.Stream)) != 0) {
          continue;
        }
        records[recordCount++] = makeUe3KeyStreamRecord(vertexContext[element.Stream]);
        if (recordCount == records.size()) {
          hash = XXH3_64bits_withSeed(records.data(), recordCount * sizeof(Ue3KeyStreamRecord), hash);
          recordCount = 0;
        }
      }
      if (recordCount > 0) {
        hash = XXH3_64bits_withSeed(records.data(), recordCount * sizeof(Ue3KeyStreamRecord), hash);
      }

      return hash;
    }

    struct Ue3IaMemoKeyHeader {
      D3D9Rtx::DrawContext drawContext; // 28 bytes, static_assert'd in d3d9_rtx.h
      uint32_t vertexCount;
      uint32_t indexCount;
      uint32_t topology;
      uint32_t texcoordIndex;
      uint32_t iaTexcoordIndex;
      uint32_t texcoordCompU;
      uint32_t texcoordCompV;
      uint32_t uvResolutionMode;
      uint32_t forceIaTexcoordForOutlier;
      uint32_t indexType;
      uint32_t explicitPad0;
      uint64_t ibo;
      uint64_t indexSliceHandle;
      uint64_t indexSliceOffset;
      uint64_t indexSliceLength;
      uint64_t indexContentGeneration;
    };
    static_assert(sizeof(Ue3IaMemoKeyHeader) == 28 + 11 * sizeof(uint32_t) + 5 * sizeof(uint64_t),
                  "Ue3IaMemoKeyHeader must have no implicit padding (it is hashed by memory).");

    // templated so the private D3D9Rtx::IndexContext type is deduced rather than named
    template<typename IndexContextT>
    Ue3IaMemoKeyHeader makeUe3IaMemoKeyHeader(const IndexContextT& indexContext,
                                              const D3D9Rtx::DrawContext& drawContext,
                                              const RasterGeometry& geoData,
                                              const uint32_t texcoordIndex,
                                              const uint32_t iaTexcoordIndex,
                                              const uint32_t texcoordCompU,
                                              const uint32_t texcoordCompV,
                                              const uint32_t uvResolutionMode,
                                              const bool forceIaTexcoordForOutlier) {
      Ue3IaMemoKeyHeader header = {};
      header.drawContext = drawContext;
      header.vertexCount = geoData.vertexCount;
      header.indexCount = geoData.indexCount;
      header.topology = uint32_t(geoData.topology);
      header.texcoordIndex = texcoordIndex;
      header.iaTexcoordIndex = iaTexcoordIndex;
      header.texcoordCompU = texcoordCompU;
      header.texcoordCompV = texcoordCompV;
      header.uvResolutionMode = uvResolutionMode;
      header.forceIaTexcoordForOutlier = forceIaTexcoordForOutlier ? 1u : 0u;
      header.indexType = uint32_t(indexContext.indexType);
      header.ibo = uint64_t(reinterpret_cast<uintptr_t>(indexContext.ibo));
      header.indexSliceHandle = uint64_t(reinterpret_cast<uintptr_t>(indexContext.indexBuffer.handle));
      header.indexSliceOffset = uint64_t(indexContext.indexBuffer.offset);
      header.indexSliceLength = uint64_t(indexContext.indexBuffer.length);
      // content generation makes stale reuse impossible if the game rewrites the buffer
      header.indexContentGeneration = indexContext.ibo != nullptr ? indexContext.ibo->remixContentGeneration : 0ull;
      return header;
    }
  }

  // IA-only identity for the geometry hash/AABB memo: draw range, decl, texcoord selection
  // (it decides which stream feeds the hashed texcoord region) and per-buffer physical
  // identity + content generation. Deliberately excludes the stable VS-constant hash and
  // the object transform, so every instance of a mesh - and every animation pose of a
  // skinned mesh - shares one entry.
  XXH64_hash_t D3D9Rtx::computeUe3IaGeometryMemoKey(const IndexContext& indexContext,
                                                    const VertexContext vertexContext[caps::MaxStreams],
                                                    const DrawContext& drawContext,
                                                    const RasterGeometry& geoData) const {
    constexpr uint64_t kSeed = 0x7BD5C66E91A3D4F1ull;

    const Ue3IaMemoKeyHeader header = makeUe3IaMemoKeyHeader(
        indexContext, drawContext, geoData,
        uint32_t(m_texcoordIndex), uint32_t(m_iaTexcoordIndex),
        uint32_t(m_texcoordCompU), uint32_t(m_texcoordCompV),
        uint32_t(m_uvResolutionMode), m_forceIaTexcoordForOutlier);

    const XXH64_hash_t headerHash = XXH3_64bits_withSeed(&header, sizeof(header), kSeed);
    return hashUe3KeyStreamRecords(d3d9State().vertexDecl->GetElements(), vertexContext, headerHash,
                                   m_currentUe3Instancing.instanceDataStreamMask);
  }

  namespace {
    struct Ue3VertexCaptureKeyHeader {
      Ue3IaMemoKeyHeader iaHeader; // shared IA identity block
      uint32_t cullMode;
      uint32_t frontFace;
      uint32_t positionSource;
      uint32_t explicitPad0;
      uint64_t stableVsHash;
      Matrix4 objectToWorld;
    };
    static_assert(sizeof(Ue3VertexCaptureKeyHeader) ==
                    sizeof(Ue3IaMemoKeyHeader) + 4 * sizeof(uint32_t) + sizeof(uint64_t) + sizeof(Matrix4),
                  "Ue3VertexCaptureKeyHeader must have no implicit padding (it is hashed by memory).");
  }

  XXH64_hash_t D3D9Rtx::computeUe3StaticVertexCaptureCacheKey(const IndexContext& indexContext,
                                                              const VertexContext vertexContext[caps::MaxStreams],
                                                              const DrawContext& drawContext,
                                                              const RasterGeometry& geoData) const {
    constexpr uint64_t kSeed = 0x92F367D3E2F391A5ull;

    Ue3VertexCaptureKeyHeader header = {};
    header.iaHeader = makeUe3IaMemoKeyHeader(
        indexContext, drawContext, geoData,
        uint32_t(m_texcoordIndex), uint32_t(m_iaTexcoordIndex),
        uint32_t(m_texcoordCompU), uint32_t(m_texcoordCompV),
        uint32_t(m_uvResolutionMode), m_forceIaTexcoordForOutlier);
    header.cullMode = uint32_t(geoData.cullMode);
    header.frontFace = uint32_t(geoData.frontFace);
    // Positions from different sources are not interchangeable, so switching source (e.g. via
    // ue3VertexCaptureSourceOverride) must not serve an entry captured through the other one.
    header.positionSource = uint32_t(m_activeCapturePositionSource);
    // computed once per draw in internalPrepareDraw and shared with computeHash
    header.stableVsHash = m_activeStableVsHash;
    header.objectToWorld = m_activeDrawCallState.transformData.objectToWorld;

    const XXH64_hash_t headerHash = XXH3_64bits_withSeed(&header, sizeof(header), kSeed);
    return hashUe3KeyStreamRecords(d3d9State().vertexDecl->GetElements(), vertexContext, headerHash,
                                   m_currentUe3Instancing.instanceDataStreamMask);
  }

  // Second-choice exact source, for shaders whose oPos transform was not recognised. Only
  // safe where the shader does nothing to the position but move it rigidly by LocalToWorld,
  // so the conditions here stay deliberately narrow: static LocalVertexFactory meshes with
  // no skinning, decal, wind or view-space CTAB constants in play.
  bool D3D9Rtx::canUseUe3NativeLocalVertexCapture(const IndexContext& indexContext,
                                                  const VertexContext vertexContext[caps::MaxStreams],
                                                  const RasterGeometry& geoData) const {
    if (!m_frameOptions.ue3NativeLocalMeshVertexCapture || !m_frameOptions.ue3EngineMode) {
      return false;
    }
    if (!m_parent->UseProgrammableVS() || !m_frameOptions.useVertexCapture) {
      return false;
    }
    if (m_currentUe3VertexFactory != Ue3VertexFactoryType::Local) {
      return false;
    }
    if (!m_currentUe3CtabInfo.has_value() || !m_currentUe3CtabInfo->hasLocalToWorld) {
      return false;
    }
    const Ue3VsShaderCtabInfo& ctabInfo = *m_currentUe3CtabInfo;
    if (ctabInfo.hasDecalTransform ||
        ctabInfo.hasDecalLocation ||
        ctabInfo.hasDecalOffset ||
        ctabInfo.hasTextureCoordinateScaleBias ||
        ctabInfo.hasViewToLocal ||
        ctabInfo.hasWindMatrices) {
      return false;
    }
    if (!geoData.positionBuffer.defined() ||
        geoData.blendWeightBuffer.defined() ||
        geoData.blendIndicesBuffer.defined()) {
      return false;
    }
    if (m_activeDrawCallState.testCategoryFlags(InstanceCategories::WorldUI) ||
        m_activeDrawCallState.testCategoryFlags(InstanceCategories::Terrain)) {
      return false;
    }
    if (d3d9State().vertexDecl == nullptr) {
      return false;
    }

    const auto& elements = d3d9State().vertexDecl->GetElements();
    const VDeclSignature sig = buildVDeclSignature(elements);
    if (sig.positionStream == sig.tangentStream || sig.positionStream == sig.normalStream) {
      return false;
    }

    auto isStaticBuffer = [](D3D9CommonBuffer* buffer) {
      if (buffer == nullptr || buffer->Desc() == nullptr) {
        return false;
      }
      if ((buffer->Desc()->Usage & D3DUSAGE_DYNAMIC) != 0 || buffer->WasWrittenByGPU()) {
        return false;
      }
      return !buffer->NeedsUpload();
    };

    if (indexContext.indexType != VK_INDEX_TYPE_NONE_KHR && !isStaticBuffer(indexContext.ibo)) {
      return false;
    }

    for (const auto& element : elements) {
      if (element.Stream >= caps::MaxStreams) {
        return false;
      }

      const VertexContext& ctx = vertexContext[element.Stream];
      if (ctx.mappedSlice.handle == VK_NULL_HANDLE || !isStaticBuffer(ctx.pVBO)) {
        return false;
      }
    }

    return true;
  }

  // An instanced mesh factory's world position is GetInstanceToWorld(Input) * Input.Position
  // (FoliageVertexFactory.usf, shared by FFoliageVertexFactory and
  // FParticleInstancedMeshVertexFactory), so the declaration's POSITION is object space by
  // construction - there is no shader constant involved and nothing to prove about the transform.
  // That makes the conservatism canUseUe3NativeLocalVertexCapture applies to Local draws, where
  // reading the input assembler is a guess about what the shader does, unnecessary here.
  bool D3D9Rtx::canUseUe3InstancedMeshVertexPositions(const RasterGeometry& geoData,
                                                     const char** outReason) const {
    auto fail = [&](const char* reason) {
      if (outReason != nullptr) {
        *outReason = reason;
      }
      return false;
    };

    if (!m_frameOptions.ue3EngineMode) {
      return fail("UE3 engine mode disabled");
    }
    if (!m_currentUe3Instancing.hasInstanceTransform) {
      return fail("no per-instance transform basis in the declaration");
    }
    if (m_currentUe3VertexFactory != Ue3VertexFactoryType::Foliage &&
        m_currentUe3VertexFactory != Ue3VertexFactoryType::ParticleInstancedMesh) {
      return fail("not a recognised instanced mesh vertex factory");
    }
    // Object-space positions are only meaningful alongside the transforms that place them.
    if (m_ue3InstanceTransformReadFailure != nullptr) {
      return fail(m_ue3InstanceTransformReadFailure);
    }
    if (!geoData.positionBuffer.defined()) {
      return fail("no input-assembler position buffer");
    }
    // Skinning would move the vertices after the input assembler; instanced factories never skin.
    if (geoData.blendWeightBuffer.defined() || geoData.blendIndicesBuffer.defined()) {
      return fail("draw carries skinning data");
    }

    if (outReason != nullptr) {
      *outReason = "";
    }
    return true;
  }

  // Resolved once per vertex shader. The camera registers are always excluded: hashing them would
  // make every draw's identity move with the view. Stock UE3 reserves two (VSR_ViewProjMatrix at c0,
  // VSR_ViewOrigin at c4), which the defaults encode. The second variant additionally drops the
  // object transform and the shading-only constants, which describe where a mesh is and how it is
  // lit rather than what its geometry is.
  D3D9Rtx::Ue3VsHashExclusions D3D9Rtx::buildUe3VsHashExclusions(
      const Ue3VsShaderCtabInfo& ctabInfo,
      const std::vector<Ue3VsConstantSymbol>* symbols) {
    using Range = Ue3VsHashExclusions::Range;
    std::array<Range, Ue3VsHashExclusions::kMaxRanges> raw = {};
    uint32_t rawCount = 0;
    auto add = [&](const uint32_t begin, const uint32_t count) {
      if (count == 0 || rawCount >= Ue3VsHashExclusions::kMaxRanges) {
        return;
      }
      raw[rawCount++] = Range { begin, begin + count };
    };

    // Sorts, merges overlaps, and writes the result out. Merging means the per-draw path can walk
    // the ranges once in order and hash the gaps between them.
    auto finalize = [](std::array<Range, Ue3VsHashExclusions::kMaxRanges>& ranges,
                       const uint32_t count,
                       std::array<Range, Ue3VsHashExclusions::kMaxRanges>& out,
                       uint32_t& outCount) {
      outCount = 0;
      if (count == 0) {
        return;
      }
      std::sort(ranges.begin(), ranges.begin() + count,
                [](const Range& a, const Range& b) { return a.begin < b.begin; });
      out[0] = ranges[0];
      outCount = 1;
      for (uint32_t i = 1; i < count; i++) {
        if (ranges[i].begin <= out[outCount - 1].end) {
          out[outCount - 1].end = std::max(out[outCount - 1].end, ranges[i].end);
        } else {
          out[outCount++] = ranges[i];
        }
      }
    };

    // Where the CTAB names them, those locations replace the reserved defaults rather than adding
    // to them. Excluding both would drop c0..c4 from the hash for a shader that keeps real
    // per-draw state there, letting draws that differ collide on the same hash.
    uint32_t viewProjReg = kUe3VsrViewProjMatrixRegister;
    uint32_t viewProjRegCount = 4;
    uint32_t viewOriginReg = kUe3VsrViewOriginRegister;
    uint32_t viewOriginRegCount = 1;
    if (ctabInfo.hasViewProjectionMatrix && ctabInfo.viewProjectionMatrixRegisterCount > 0) {
      viewProjReg = ctabInfo.viewProjectionMatrixRegisterIndex;
      viewProjRegCount = ctabInfo.viewProjectionMatrixRegisterCount;
    }
    if (ctabInfo.hasCameraPosition && ctabInfo.cameraPositionRegisterCount > 0) {
      viewOriginReg = ctabInfo.cameraPositionRegisterIndex;
      viewOriginRegCount = ctabInfo.cameraPositionRegisterCount;
    }
    add(viewProjReg, viewProjRegCount);
    add(viewOriginReg, viewOriginRegCount);

    Ue3VsHashExclusions result;
    std::array<Range, Ue3VsHashExclusions::kMaxRanges> cameraRaw = raw;
    const uint32_t cameraRawCount = rawCount;
    finalize(cameraRaw, cameraRawCount, result.cameraOnly, result.cameraOnlyCount);

    if (symbols != nullptr) {
      auto contains = [](const std::string& haystack, const char* needle) {
        return haystack.find(needle) != std::string::npos;
      };
      for (const Ue3VsConstantSymbol& symbol : *symbols) {
        const std::string name = toLowerAscii(symbol.name);
        // Shading-only: these reach interpolators, never vertex positions.
        if (contains(name, "lightmapscale") ||
            contains(name, "light_map_scale") ||
            contains(name, "lightmapcoordinatescalebias") ||
            contains(name, "light_map_coordinate_scale_bias") ||
            contains(name, "shadowcoordinatescalebias") ||
            contains(name, "shadow_coordinate_scale_bias")) {
          add(symbol.registerIndex, symbol.registerCount);
        }
      }
    }

    std::array<Range, Ue3VsHashExclusions::kMaxRanges> shadingRaw = raw;
    const uint32_t shadingRawCount = rawCount;
    finalize(shadingRaw, shadingRawCount, result.cameraAndShading, result.cameraAndShadingCount);

    if (ctabInfo.hasLocalToWorld) {
      add(ctabInfo.localToWorldRegisterIndex, ctabInfo.localToWorldRegisterCount);
    }
    if (ctabInfo.hasWorldToLocal) {
      add(ctabInfo.worldToLocalRegisterIndex, ctabInfo.worldToLocalRegisterCount);
    }
    finalize(raw, rawCount, result.withPlacement, result.withPlacementCount);

    return result;
  }

  XXH64_hash_t D3D9Rtx::computeUe3StableVertexShaderHash(bool* outHashedFloatConstsWithExclusions) const {
    if (outHashedFloatConstsWithExclusions != nullptr) {
      *outHashedFloatConstsWithExclusions = false;
    }

    if (d3d9State().vertexShader.ptr() == nullptr) {
      return kEmptyHash;
    }

    const D3D9ConstantSets& cb = m_parent->m_consts[DxsoProgramTypes::VertexShader];
    XXH64_hash_t hash = d3d9State().vertexShader->GetCommonShader()->GetBytecodeHash();

    const uint32_t floatConstRegCount = cb.meta.maxConstIndexF;
    const uint8_t* const floatConstBase = reinterpret_cast<const uint8_t*>(&d3d9State().vsConsts.fConsts[0]);

    auto hashFloatConstRange = [&](uint32_t beginReg, uint32_t endReg) {
      beginReg = std::min(beginReg, floatConstRegCount);
      endReg = std::min(endReg, floatConstRegCount);
      if (beginReg >= endReg) {
        return;
      }

      const size_t offsetBytes = size_t(beginReg) * sizeof(Vector4);
      const size_t sizeBytes = size_t(endReg - beginReg) * sizeof(Vector4);
      hash = XXH3_64bits_withSeed(floatConstBase + offsetBytes, sizeBytes, hash);
    };

    // The exclusion ranges are a pure function of the shader's CTAB, so they are resolved once per
    // shader and borrowed here. This path runs for every draw; it must not build or sort anything.
    bool hashedFloatConstsWithExclusions = false;
    if ((m_frameOptions.ue3CameraFromShaderConstants || m_frameOptions.ue3EngineMode) &&
        floatConstRegCount > 0) {
      // A shader with no resolved CTAB still must not hash the reserved camera registers, or its
      // identity would move with the view.
      static const Ue3VsHashExclusions s_reservedOnly = buildUe3VsHashExclusions(Ue3VsShaderCtabInfo {}, nullptr);
      const Ue3VsHashExclusions& exclusions =
        m_currentUe3VsHashExclusions != nullptr ? *m_currentUe3VsHashExclusions : s_reservedOnly;
      const bool excludePlacement = m_frameOptions.ue3ExcludePlacementFromVertexShaderHash;
      // Dropping the shading-only constants is free and keeps the hash off the lightmap policy,
      // so UE3 mode takes it unconditionally; the placement registers stay opt-in for their cost.
      const bool excludeShading = m_frameOptions.ue3EngineMode;
      const Ue3VsHashExclusions::Range* ranges = exclusions.cameraOnly.data();
      uint32_t rangeCount = exclusions.cameraOnlyCount;
      if (excludePlacement) {
        ranges = exclusions.withPlacement.data();
        rangeCount = exclusions.withPlacementCount;
      } else if (excludeShading) {
        ranges = exclusions.cameraAndShading.data();
        rangeCount = exclusions.cameraAndShadingCount;
      }

      if (rangeCount > 0) {
        uint32_t cursor = 0;
        for (uint32_t i = 0; i < rangeCount; i++) {
          if (ranges[i].begin >= floatConstRegCount) {
            break;
          }
          hashFloatConstRange(cursor, ranges[i].begin);
          cursor = std::max(cursor, ranges[i].end);
        }
        hashFloatConstRange(cursor, floatConstRegCount);
        hashedFloatConstsWithExclusions = true;
      }
    }

    if (!hashedFloatConstsWithExclusions && floatConstRegCount > 0) {
      hash = XXH3_64bits_withSeed(
        &d3d9State().vsConsts.fConsts[0],
        size_t(floatConstRegCount) * sizeof(Vector4),
        hash);
    }

    if (cb.meta.maxConstIndexI > 0) {
      hash = XXH3_64bits_withSeed(
        &d3d9State().vsConsts.iConsts[0],
        size_t(cb.meta.maxConstIndexI) * sizeof(int) * 4,
        hash);
    }
    if (cb.meta.maxConstIndexB > 0) {
      hash = XXH3_64bits_withSeed(
        &d3d9State().vsConsts.bConsts[0],
        size_t(cb.meta.maxConstIndexB) * sizeof(uint32_t) / 32,
        hash);
    }

    if (outHashedFloatConstsWithExclusions != nullptr) {
      *outHashedFloatConstsWithExclusions = hashedFloatConstsWithExclusions;
    }

    return hash;
  }

  bool D3D9Rtx::tryReuseUe3StaticVertexCapture(XXH64_hash_t cacheKey, RasterGeometry& geoData) {
    auto it = m_ue3VertexCaptureCache.find(cacheKey);
    if (it == m_ue3VertexCaptureCache.end()) {
      return false;
    }

    Ue3VertexCaptureCacheEntry& entry = it->second;
    const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();
    if (entry.vertexCount != geoData.vertexCount ||
        !entry.positionBuffer.defined() ||
        entry.lastFrameTouched == currentFrame) {
      return false;
    }

    geoData.positionBuffer = entry.positionBuffer;
    geoData.normalBuffer = entry.normalBuffer;
    geoData.texcoordBuffer = entry.texcoordBuffer;
    geoData.color0Buffer = entry.color0Buffer;
    entry.lastFrameTouched = currentFrame;
    ++m_ue3VertexCaptureCacheFrameReuses;

    return true;
  }

  // Two-tier admission. A fresh capture is recorded in the CPU-only admission map first and
  // only promoted to the retention tier once its key has been seen on enough distinct frames,
  // because the key covers the object transform and the shader's non-camera constants: any
  // draw that animates or moves mints a new key every frame, so retaining on first sighting
  // would pin a device-local capture buffer per such draw per frame with no possible reuse.
  void D3D9Rtx::updateUe3StaticVertexCaptureCache(XXH64_hash_t cacheKey, const RasterGeometry& geoData) {
    const VkDeviceSize budgetBytes =
      VkDeviceSize(m_frameOptions.ue3StaticLocalMeshVertexCaptureCacheBudgetMiB) * 1024ull * 1024ull;
    if (budgetBytes == 0) {
      return;
    }

    // The four views alias one capture buffer, so the position slice's length is the entry's
    // whole VRAM cost.
    const VkDeviceSize byteSize = geoData.positionBuffer.length();
    if (byteSize > budgetBytes) {
      // a single capture larger than the whole budget would evict everything and then itself
      return;
    }

    const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();

    // An already-retained key reaching here was recaptured rather than reused - drawn a second
    // time in one frame, or its vertex count changed - so refresh it in place instead of
    // sending it back through admission.
    auto it = m_ue3VertexCaptureCache.find(cacheKey);
    if (it == m_ue3VertexCaptureCache.end()) {
      const uint32_t warmupFrames = std::max(1u, m_frameOptions.ue3StaticLocalMeshVertexCaptureCacheWarmupFrames);

      Ue3VertexCaptureAdmissionEntry& admission = m_ue3VertexCaptureAdmission[cacheKey];
      if (admission.vertexCount != geoData.vertexCount) {
        // same key over different geometry: restart, the previous sightings prove nothing
        admission.vertexCount = geoData.vertexCount;
        admission.sightings = 0;
      }
      if (admission.lastFrameSeen != currentFrame || admission.sightings == 0) {
        // count one sighting per frame, so a mesh drawn many times within a single frame does
        // not look like a key that has proven itself across frames
        admission.sightings = std::min(admission.sightings + 1, 0xFFFFu);
        admission.lastFrameSeen = currentFrame;
      }

      if (admission.sightings < warmupFrames) {
        return;
      }

      it = m_ue3VertexCaptureCache.emplace(cacheKey, Ue3VertexCaptureCacheEntry {}).first;
      // the admission record has served its purpose; if the budget later evicts this entry
      // the key simply serves its warmup again
      m_ue3VertexCaptureAdmission.erase(cacheKey);
    }

    Ue3VertexCaptureCacheEntry& entry = it->second;
    m_ue3VertexCaptureCacheBytes -= std::min(m_ue3VertexCaptureCacheBytes, entry.byteSize);
    entry.positionBuffer = geoData.positionBuffer;
    entry.normalBuffer = geoData.normalBuffer;
    entry.texcoordBuffer = geoData.texcoordBuffer;
    entry.color0Buffer = geoData.color0Buffer;
    entry.vertexCount = geoData.vertexCount;
    entry.lastFrameTouched = currentFrame;
    entry.byteSize = byteSize;
    m_ue3VertexCaptureCacheBytes += byteSize;
  }

  void D3D9Rtx::eraseUe3StaticVertexCaptureCacheEntry(XXH64_hash_t cacheKey) {
    auto it = m_ue3VertexCaptureCache.find(cacheKey);
    if (it == m_ue3VertexCaptureCache.end()) {
      return;
    }
    m_ue3VertexCaptureCacheBytes -= std::min(m_ue3VertexCaptureCacheBytes, it->second.byteSize);
    m_ue3VertexCaptureCache.erase(it);
  }

  void D3D9Rtx::clearUe3StaticVertexCaptureCache() {
    m_ue3VertexCaptureCache.clear();
    m_ue3VertexCaptureCacheBytes = 0;
    m_ue3VertexCaptureAdmission.clear();
  }

  void D3D9Rtx::pruneUe3StaticVertexCaptureCache() {
    const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();
    const uint32_t retentionFrames =
      std::max(1u, m_frameOptions.ue3StaticLocalMeshVertexCaptureCacheRetentionFrames);

    m_ue3VertexCaptureCache.erase_if([&](auto it) {
      if (currentFrame - it->second.lastFrameTouched <= retentionFrames) {
        return false;
      }
      m_ue3VertexCaptureCacheBytes -= std::min(m_ue3VertexCaptureCacheBytes, it->second.byteSize);
      return true;
    });

    // Admission records are pure bookkeeping, but a churning key mints one per draw per frame,
    // so they need the same expiry to stay bounded. A key still warming up is re-seen every
    // frame, so the retention window is more than enough to keep it alive.
    m_ue3VertexCaptureAdmission.erase_if([&](auto it) {
      return currentFrame - it->second.lastFrameSeen > retentionFrames;
    });

    enforceUe3StaticVertexCaptureCacheBudget();
  }

  // Backstop for the admission policy: if a game still manages to push more repeating keys
  // than expected (many placements of the same mesh each key on their own transform), the
  // cache gives up hit rate rather than VRAM.
  void D3D9Rtx::enforceUe3StaticVertexCaptureCacheBudget() {
    const VkDeviceSize budgetBytes =
      VkDeviceSize(m_frameOptions.ue3StaticLocalMeshVertexCaptureCacheBudgetMiB) * 1024ull * 1024ull;
    const size_t maxEntries = m_frameOptions.ue3StaticLocalMeshVertexCaptureCacheMaxEntries;

    auto overBudget = [&]() {
      return m_ue3VertexCaptureCacheBytes > budgetBytes ||
             (maxEntries > 0 && m_ue3VertexCaptureCache.size() > maxEntries);
    };

    if (!overBudget()) {
      return;
    }

    // A once-per-session note: from here on the cache trades hit rate for a VRAM ceiling, which
    // is worth knowing when diagnosing why captures are not being reused.
    ONCE(Logger::warn(str::format(
      "[RTX-Compatibility][UE3-Capture] static vertex capture cache reached its limit (",
      m_ue3VertexCaptureCacheBytes / (1024 * 1024), " MiB over ",
      m_ue3VertexCaptureCache.size(), " entries, budget ",
      budgetBytes / (1024 * 1024), " MiB / ", maxEntries, " entries); evicting least recently "
      "used entries from here on. Raise rtx.d3d9.ue3StaticLocalMeshVertexCaptureCacheBudgetMiB "
      "if reuse suffers.")));

    if (budgetBytes == 0) {
      clearUe3StaticVertexCaptureCache();
      return;
    }

    std::vector<std::pair<uint32_t, XXH64_hash_t>> byLastTouched;
    byLastTouched.reserve(m_ue3VertexCaptureCache.size());
    for (const auto& [key, entry] : m_ue3VertexCaptureCache) {
      byLastTouched.emplace_back(entry.lastFrameTouched, key);
    }
    std::sort(byLastTouched.begin(), byLastTouched.end());

    for (const auto& candidate : byLastTouched) {
      if (!overBudget()) {
        break;
      }
      eraseUe3StaticVertexCaptureCacheEntry(candidate.second);
      ++m_ue3VertexCaptureCacheEvictions;
    }
  }

  std::string D3D9Rtx::describeUe3VsConstantRegister(XXH64_hash_t vsBytecodeHash, uint32_t reg) const {
    auto it = m_ue3VsConstantSymbols.find(vsBytecodeHash);
    if (it != m_ue3VsConstantSymbols.end()) {
      for (const Ue3VsConstantSymbol& symbol : it->second) {
        if (reg < symbol.registerIndex || reg >= symbol.registerIndex + symbol.registerCount) {
          continue;
        }
        // Name the row for multi-register symbols, so a churning matrix says which row moved.
        if (symbol.registerCount > 1) {
          return str::format("c", reg, " (", symbol.name, "[", reg - symbol.registerIndex, "])");
        }
        return str::format("c", reg, " (", symbol.name, ")");
      }
    }
    return str::format("c", reg);
  }

  // Accumulates the three levels described on Ue3ChurnMeshEntry. Levels 1 and 2 are decided on the
  // first draw of a mesh in a new frame, since only then is the previous frame's placement set
  // complete; level 3 compares that same first draw's constants against the previous frame's.
  void D3D9Rtx::trackUe3ConstantChurn(const XXH64_hash_t iaKey, const RasterGeometry& geoData) {
    ScopedCpuProfileZone();

    const D3D9CommonShader* vertexShader =
      d3d9State().vertexShader.ptr() != nullptr ? d3d9State().vertexShader->GetCommonShader() : nullptr;
    if (vertexShader == nullptr) {
      return;
    }

    const uint32_t floatConstRegCount =
      std::min(m_parent->m_consts[DxsoProgramTypes::VertexShader].meta.maxConstIndexF,
               uint32_t(caps::MaxFloatConstantsSoftware));
    if (floatConstRegCount == 0) {
      return;
    }

    const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();
    const Vector4* floatConsts = &d3d9State().vsConsts.fConsts[0];

    // Keyed per mesh, not per placement: the transform is recorded inside the entry so that a
    // moving transform and a moving IA identity stay distinguishable.
    const XXH64_hash_t transformHash =
      XXH3_64bits(&m_activeDrawCallState.transformData.objectToWorld, sizeof(Matrix4));

    // Hash the registers the transform was extracted from, using the current draw's CTAB rather
    // than the entry's cached ranges so it is right even on the first sighting.
    XXH64_hash_t rawTransformHash = 0;
    if (m_currentUe3CtabInfo.has_value()) {
      const Ue3VsShaderCtabInfo& ctabInfo = *m_currentUe3CtabInfo;
      auto foldRange = [&](const uint32_t base, const uint32_t count) {
        if (count == 0 || base >= floatConstRegCount) {
          return;
        }
        const uint32_t end = std::min(base + count, floatConstRegCount);
        rawTransformHash = XXH3_64bits_withSeed(
          &floatConsts[base], size_t(end - base) * sizeof(Vector4), rawTransformHash);
      };
      if (ctabInfo.hasLocalToWorld) {
        foldRange(ctabInfo.localToWorldRegisterIndex, ctabInfo.localToWorldRegisterCount);
      }
      if (ctabInfo.hasWorldToLocal) {
        foldRange(ctabInfo.worldToLocalRegisterIndex, ctabInfo.worldToLocalRegisterCount);
      }
    }

    auto it = m_ue3ConstantChurn.find(iaKey);
    if (it == m_ue3ConstantChurn.end()) {
      if (m_ue3ConstantChurn.size() >= m_frameOptions.ue3VertexConstantChurnMaxTrackedDraws) {
        // An IA identity that changes every frame never revisits a key, so without eviction the
        // sample fills once with entries that can never be compared and the diagnostic goes blind.
        m_ue3ConstantChurn.erase_if([currentFrame](auto stale) {
          return stale->second.lastFrameSeen + 2 < currentFrame;
        });
        if (m_ue3ConstantChurn.size() >= m_frameOptions.ue3VertexConstantChurnMaxTrackedDraws) {
          return;
        }
      }
      it = m_ue3ConstantChurn.emplace(iaKey, Ue3ChurnMeshEntry {}).first;
      ++m_ue3ChurnMeshKeysCreated;
    }

    Ue3ChurnMeshEntry& entry = it->second;

    // Level 1 and 2 are evaluated on the first draw of this mesh in a new frame, because only then
    // is the previous frame's set of placements complete.
    const bool firstTouchThisFrame = entry.currentSetFrame != currentFrame;
    if (firstTouchThisFrame) {
      if (entry.currentSetFrame != 0) {
        std::sort(entry.transformsThisFrame.begin(), entry.transformsThisFrame.end());
        std::sort(entry.rawTransformsThisFrame.begin(), entry.rawTransformsThisFrame.end());
        const XXH64_hash_t setHash = XXH3_64bits(
          entry.transformsThisFrame.data(), entry.transformsThisFrame.size() * sizeof(XXH64_hash_t));
        const XXH64_hash_t rawSetHash = XXH3_64bits(
          entry.rawTransformsThisFrame.data(), entry.rawTransformsThisFrame.size() * sizeof(XXH64_hash_t));
        const uint32_t setSize = uint32_t(entry.transformsThisFrame.size());

        // Compare two *completed* sets, and only when they belong to consecutive frames.
        if (entry.completedSetFrame != 0 && entry.completedSetFrame + 1 == entry.currentSetFrame) {
          ++m_ue3ChurnMeshRevisited;
          const bool extractedIdentical = setHash == entry.completedSetHash;
          const bool rawIdentical = rawSetHash == entry.completedRawSetHash;

          if (extractedIdentical) {
            ++m_ue3ChurnTransformSetIdentical;
          } else {
            ++m_ue3ChurnTransformSetDiffered;
            if (setSize != entry.completedSetSize) {
              ++m_ue3ChurnTransformSetSizeChanged;
            }
          }
          if (rawIdentical) {
            ++m_ue3ChurnRawTransformSetIdentical;
            if (!extractedIdentical) {
              // Same registers in, different matrices out.
              ++m_ue3ChurnExtractionUnstable;
            }
          }
        }

        entry.completedSetHash = setHash;
        entry.completedRawSetHash = rawSetHash;
        entry.completedSetSize = setSize;
        entry.completedSetFrame = entry.currentSetFrame;
      }
      entry.transformsThisFrame.clear();
      entry.rawTransformsThisFrame.clear();
      entry.currentSetFrame = currentFrame;
    }
    entry.transformsThisFrame.push_back(transformHash);
    entry.rawTransformsThisFrame.push_back(rawTransformHash);

    // Level 3: do any constants move that are neither camera-derived nor part of the object
    // transform? Those are the only ones that could destabilise geometry identity spuriously.
    // Excluding the transform registers is what makes this independent of which placement of the
    // mesh happened to be drawn first in each frame.
    const bool canCompareConstants =
      firstTouchThisFrame &&
      entry.lastFrameSeen != 0 &&
      entry.lastFrameSeen + 1 == currentFrame &&
      entry.vsBytecodeHash == vertexShader->GetBytecodeHash() &&
      entry.floatConsts.size() == floatConstRegCount;

    if (canCompareConstants) {
      auto isExcludedRegister = [&entry](const uint32_t reg) {
        auto inRange = [reg](const uint32_t base, const uint32_t count) {
          return count > 0 && reg >= base && reg < base + count;
        };
        return inRange(entry.viewProjReg, entry.viewProjRegCount) ||
               inRange(entry.cameraPosReg, entry.cameraPosRegCount) ||
               inRange(entry.localToWorldReg, entry.localToWorldRegCount) ||
               inRange(entry.worldToLocalReg, entry.worldToLocalRegCount);
      };

      const bool viewMoved =
        std::memcmp(&entry.worldToView, &m_activeDrawCallState.transformData.worldToView, sizeof(Matrix4)) != 0;
      // Dumps only while the view is moving: a camera-derived constant is the interesting
      // hypothesis, and a still-camera example cannot distinguish one.
      const bool wantDetail = viewMoved && m_ue3ConstantChurnDetailDumps < 8;

      std::string detail;
      uint32_t changedRegisters = 0;
      for (uint32_t reg = 0; reg < floatConstRegCount; reg++) {
        if (std::memcmp(&entry.floatConsts[reg], &floatConsts[reg], sizeof(Vector4)) == 0) {
          continue;
        }
        if (isExcludedRegister(reg)) {
          continue;
        }
        ++changedRegisters;
        Ue3ChurnRegisterTally& tally = m_ue3ConstantChurnByRegister[reg];
        ++tally.count;
        tally.vsBytecodeHash = entry.vsBytecodeHash;

        if (wantDetail && changedRegisters <= 6) {
          const Vector4& was = entry.floatConsts[reg];
          const Vector4& now = floatConsts[reg];
          detail += str::format(
            detail.empty() ? "" : ", ",
            describeUe3VsConstantRegister(entry.vsBytecodeHash, reg),
            " (", was.x, ",", was.y, ",", was.z, ",", was.w, ")->(",
            now.x, ",", now.y, ",", now.z, ",", now.w, ")");
        }
      }

      ++m_ue3ConstantChurnComparisons;
      if (changedRegisters > 0) {
        ++m_ue3ChurnOtherConstantsChanged;
        if (viewMoved) {
          ++m_ue3ConstantChurnWhileViewMoved;
        } else {
          ++m_ue3ConstantChurnWhileViewStill;
        }
      }

      if (!detail.empty()) {
        ++m_ue3ConstantChurnDetailDumps;
        Logger::info(str::format(
          "[RTX-Compatibility][UE3-ConstantChurn] same mesh, next frame, ignoring camera and transform "
          "registers: vs=0x", std::hex, entry.vsBytecodeHash, std::dec,
          " vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
          " verts=", geoData.vertexCount,
          " changedRegs=", changedRegisters, ": ", detail));
      }
    }

    if (firstTouchThisFrame) {
      entry.lastFrameSeen = currentFrame;
      entry.vsBytecodeHash = vertexShader->GetBytecodeHash();
      entry.worldToView = m_activeDrawCallState.transformData.worldToView;
      entry.floatConsts.assign(floatConsts, floatConsts + floatConstRegCount);

      entry.viewProjReg = kUe3VsrViewProjMatrixRegister;
      entry.viewProjRegCount = 4;
      entry.cameraPosReg = kUe3VsrViewOriginRegister;
      entry.cameraPosRegCount = 1;
      entry.localToWorldReg = 0;
      entry.localToWorldRegCount = 0;
      entry.worldToLocalReg = 0;
      entry.worldToLocalRegCount = 0;
      if (m_currentUe3CtabInfo.has_value()) {
        const Ue3VsShaderCtabInfo& ctabInfo = *m_currentUe3CtabInfo;
        if (ctabInfo.hasViewProjectionMatrix && ctabInfo.viewProjectionMatrixRegisterCount > 0) {
          entry.viewProjReg = ctabInfo.viewProjectionMatrixRegisterIndex;
          entry.viewProjRegCount = ctabInfo.viewProjectionMatrixRegisterCount;
        }
        if (ctabInfo.hasCameraPosition && ctabInfo.cameraPositionRegisterCount > 0) {
          entry.cameraPosReg = ctabInfo.cameraPositionRegisterIndex;
          entry.cameraPosRegCount = ctabInfo.cameraPositionRegisterCount;
        }
        if (ctabInfo.hasLocalToWorld && ctabInfo.localToWorldRegisterCount > 0) {
          entry.localToWorldReg = ctabInfo.localToWorldRegisterIndex;
          entry.localToWorldRegCount = ctabInfo.localToWorldRegisterCount;
        }
        if (ctabInfo.hasWorldToLocal && ctabInfo.worldToLocalRegisterCount > 0) {
          entry.worldToLocalReg = ctabInfo.worldToLocalRegisterIndex;
          entry.worldToLocalRegCount = ctabInfo.worldToLocalRegisterCount;
        }
      }
    }
  }

  void D3D9Rtx::reportUe3ConstantChurn() {
    if (!m_frameOptions.ue3LogVertexConstantChurn) {
      return;
    }

    const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();
    constexpr uint32_t kReportIntervalFrames = 300;
    if (currentFrame - m_ue3ConstantChurnReportFrameStamp < kReportIntervalFrames) {
      return;
    }
    m_ue3ConstantChurnReportFrameStamp = currentFrame;

    const uint64_t meshKeys = m_ue3ChurnMeshKeysCreated;
    const uint64_t revisits = m_ue3ChurnMeshRevisited;

    // The interpretation guide is worth saying once; repeating it every window buries the numbers.
    ONCE(Logger::info(
      "[RTX-Compatibility][UE3-ConstantChurn] reading the levels below: level 1 is whether a mesh's "
      "input-assembler identity comes back at all (if not, nothing downstream can repeat and the draws are "
      "re-uploading their vertex/index data); level 2 is whether its placements are the same, compared as a "
      "multiset so draw order does not matter, with a changed instance count meaning ordinary culling rather "
      "than movement; level 3 is whether any other constant moved, which is the only one of the three that "
      "rtx.d3d9.ue3ExcludePlacementFromVertexShaderHash cannot address."));

    if (revisits == 0) {
      ONCE(Logger::warn(str::format(
        "[RTX-Compatibility][UE3-ConstantChurn] level 1: ", meshKeys, " mesh keys minted and not one "
        "input-assembler identity came back on the next frame, so no cache key can repeat regardless of "
        "transforms or constants.")));
      m_ue3ChurnMeshKeysCreated = 0;
      return;
    }

    auto pct = [](const uint64_t n, const uint64_t of) {
      return of > 0 ? uint32_t((n * 100ull) / of) : 0u;
    };

    const uint64_t setComparisons = m_ue3ChurnTransformSetIdentical + m_ue3ChurnTransformSetDiffered;

    Logger::info(str::format(
      "[RTX-Compatibility][UE3-ConstantChurn] level 1: ", revisits, " next-frame revisits against ",
      meshKeys, " new mesh keys, ", m_ue3ConstantChurn.size(), " sampled. level 2: of ", setComparisons,
      " comparisons the placement set was identical ", pct(m_ue3ChurnTransformSetIdentical, setComparisons),
      "%, differed ", pct(m_ue3ChurnTransformSetDiffered, setComparisons), "% (",
      m_ue3ChurnTransformSetSizeChanged, " with a changed instance count), and the raw LocalToWorld registers "
      "behind them were identical ", pct(m_ue3ChurnRawTransformSetIdentical, setComparisons),
      "%. level 3: ", pct(m_ue3ChurnOtherConstantsChanged, m_ue3ConstantChurnComparisons),
      "% of ", m_ue3ConstantChurnComparisons, " comparisons moved another constant (view had moved on ",
      m_ue3ConstantChurnWhileViewMoved, ", was still on ", m_ue3ConstantChurnWhileViewStill, ")."));

    // The conclusions are one-shot: they identify a property of the title, not of this window.
    if (m_ue3ChurnExtractionUnstable > 0) {
      ONCE(Logger::warn(str::format(
        "[RTX-Compatibility][UE3-ConstantChurn] ", pct(m_ue3ChurnExtractionUnstable, setComparisons),
        "% of comparisons had identical raw LocalToWorld registers but a different extracted objectToWorld. "
        "Same input, different output means this runtime's transform extraction is not reproducible, which is a "
        "bug here rather than a property of the game.")));
    } else if (m_ue3ChurnTransformSetDiffered > 0) {
      ONCE(Logger::info(
        "[RTX-Compatibility][UE3-ConstantChurn] every differing placement set also had differing raw "
        "LocalToWorld registers, so the game uploads a new transform each frame for these draws and the "
        "extraction of it is reproducible. Those registers are in the stable VS hash, hence in "
        "rules::FullGeometryHash, so DrawCallCache::exactMatch cannot match them across frames and a fresh "
        "BlasEntry is allocated per draw per frame. rtx.d3d9.ue3ExcludePlacementFromVertexShaderHash addresses "
        "this, at the cost described in its documentation."));
    }

    // Only worth reporting when frequent enough to matter; a handful of shadow-atlas reallocations
    // per window is not a finding.
    constexpr uint32_t kConstantChurnWarnPercent = 5;
    if (pct(m_ue3ChurnOtherConstantsChanged, m_ue3ConstantChurnComparisons) >= kConstantChurnWarnPercent) {
      ONCE(Logger::warn(
        "[RTX-Compatibility][UE3-ConstantChurn] a constant that is neither camera-derived nor part of the object "
        "transform is moving often enough to destabilise geometry identity on its own. The register tally names "
        "it; changes only while the view moves point at a camera-derived constant, changes with the view still "
        "are animation or time driven."));
    }

    if (!m_ue3ConstantChurnByRegister.empty()) {
      // Busiest first: that ordering is the answer to "which constant is doing this".
      std::vector<std::pair<uint64_t, uint32_t>> byCount;
      byCount.reserve(m_ue3ConstantChurnByRegister.size());
      for (const auto& [reg, tally] : m_ue3ConstantChurnByRegister) {
        byCount.emplace_back(tally.count, reg);
      }
      std::sort(byCount.rbegin(), byCount.rend());

      std::string top;
      const size_t shown = std::min<size_t>(byCount.size(), 12);
      for (size_t i = 0; i < shown; i++) {
        const uint32_t reg = byCount[i].second;
        top += str::format(
          top.empty() ? "" : ", ",
          describeUe3VsConstantRegister(m_ue3ConstantChurnByRegister[reg].vsBytecodeHash, reg),
          " x", byCount[i].first);
      }

      Logger::info(str::format(
        "[RTX-Compatibility][UE3-ConstantChurn] registers that moved (", byCount.size(),
        " distinct, busiest first): ", top));
    }

    m_ue3ConstantChurnByRegister.clear();
    m_ue3ChurnMeshKeysCreated = 0;
    m_ue3ChurnMeshRevisited = 0;
    m_ue3ChurnTransformSetIdentical = 0;
    m_ue3ChurnTransformSetDiffered = 0;
    m_ue3ChurnTransformSetSizeChanged = 0;
    m_ue3ChurnRawTransformSetIdentical = 0;
    m_ue3ChurnExtractionUnstable = 0;
    m_ue3ConstantChurnComparisons = 0;
    m_ue3ChurnOtherConstantsChanged = 0;
    m_ue3ConstantChurnWhileViewMoved = 0;
    m_ue3ConstantChurnWhileViewStill = 0;
  }

  void D3D9Rtx::updateUe3StaticVertexCaptureCacheState() {
    const uint32_t reuses = m_ue3VertexCaptureCacheFrameReuses;
    const uint32_t captures = m_ue3VertexCaptureCacheFrameCaptures;
    m_ue3VertexCaptureCacheFrameReuses = 0;
    m_ue3VertexCaptureCacheFrameCaptures = 0;

    // Toggling the cache off in the dev UI should not leave a stale dormancy verdict behind,
    // and nothing should stay resident while it is off.
    if (!m_frameOptions.ue3StaticLocalMeshVertexCaptureCache) {
      if (m_ue3VertexCaptureCacheDormant || !m_ue3VertexCaptureCache.empty() || !m_ue3VertexCaptureAdmission.empty()) {
        clearUe3StaticVertexCaptureCache();
        m_ue3VertexCaptureCacheDormant = false;
        m_ue3VertexCaptureCacheProbeCountdown = 0;
      }
      return;
    }

    m_ue3VertexCaptureWindowReuses += reuses;
    m_ue3VertexCaptureWindowCaptures += captures;
    ++m_ue3VertexCaptureWindowFrames;

    m_ue3VertexCaptureCacheStatReuses += reuses;
    m_ue3VertexCaptureCacheStatCaptures += captures;
    ++m_ue3VertexCaptureCacheStatFrames;

    evaluateUe3StaticVertexCaptureCacheDormancy();
    reportUe3StaticVertexCaptureCacheStats();
  }

  void D3D9Rtx::evaluateUe3StaticVertexCaptureCacheDormancy() {
    auto resetWindow = [this]() {
      m_ue3VertexCaptureWindowFrames = 0;
      m_ue3VertexCaptureWindowReuses = 0;
      m_ue3VertexCaptureWindowCaptures = 0;
    };

    const uint32_t minReusePercent = m_frameOptions.ue3StaticLocalMeshVertexCaptureCacheMinReusePercent;
    if (minReusePercent == 0) {
      // Guard disabled: never stand the cache down, and keep the window empty so re-enabling the
      // guard later judges the cache on fresh frames rather than accumulated history.
      m_ue3VertexCaptureCacheDormant = false;
      m_ue3VertexCaptureCacheProbeCountdown = 0;
      resetWindow();
      return;
    }

    if (m_ue3VertexCaptureCacheDormant) {
      // Nothing is eligible while dormant, so the window cannot measure anything. Wait out the
      // probe interval, then wake up and let the window below judge the cache afresh - a later
      // level or a different shader set may well be cacheable.
      if (m_ue3VertexCaptureCacheProbeCountdown > 0) {
        --m_ue3VertexCaptureCacheProbeCountdown;
        resetWindow();
        return;
      }
      m_ue3VertexCaptureCacheDormant = false;
      resetWindow();
      if (m_frameOptions.ue3LogStaticVertexCaptureCacheStats) {
        Logger::info("[RTX-Compatibility][UE3-Capture] static vertex capture cache waking to re-measure its reuse rate.");
      }
      return;
    }

    // Long enough to span a camera pause without being fooled by one, short enough that a
    // hopeless title is stood down within a few seconds of gameplay.
    constexpr uint32_t kReuseEvaluationFrames = 120;
    if (m_ue3VertexCaptureWindowFrames < kReuseEvaluationFrames) {
      return;
    }

    const uint64_t considered = m_ue3VertexCaptureWindowReuses + m_ue3VertexCaptureWindowCaptures;
    const uint32_t reusePercent = considered > 0
      ? uint32_t((m_ue3VertexCaptureWindowReuses * 100ull) / considered)
      : 100u;
    resetWindow();

    if (considered == 0) {
      // no eligible draws in the window (menu, loading screen): no evidence either way
      return;
    }
    if (reusePercent >= minReusePercent) {
      return;
    }

    m_ue3VertexCaptureCacheDormant = true;
    m_ue3VertexCaptureCacheProbeCountdown = m_frameOptions.ue3StaticLocalMeshVertexCaptureCacheReuseProbeFrames;
    const size_t releasedEntries = m_ue3VertexCaptureCache.size();
    const VkDeviceSize releasedBytes = m_ue3VertexCaptureCacheBytes;
    const size_t releasedKeys = m_ue3VertexCaptureAdmission.size();
    clearUe3StaticVertexCaptureCache();

    // Worth one line per session even without diagnostics on: it explains why an enabled
    // option stopped doing anything, and points at the option that turns the guard off.
    ONCE(Logger::info(str::format(
      "[RTX-Compatibility][UE3-Capture] static vertex capture cache stood down: only ", reusePercent,
      "% of eligible draws were reused (threshold ", minReusePercent, "%), so its keys are not repeating in this "
      "title - most likely a camera-dependent vertex shader constant or object transform. Released ",
      releasedBytes / (1024 * 1024), " MiB over ", releasedEntries, " entries and ", releasedKeys,
      " pending keys. Re-tested every ", m_ue3VertexCaptureCacheProbeCountdown, " frames; set "
      "rtx.d3d9.ue3StaticLocalMeshVertexCaptureCacheMinReusePercent = 0 to keep it running regardless.")));

    if (m_frameOptions.ue3LogStaticVertexCaptureCacheStats) {
      Logger::info(str::format(
        "[RTX-Compatibility][UE3-Capture] static vertex capture cache going dormant at ", reusePercent,
        "% reuse; released ", releasedBytes / (1024 * 1024), " MiB / ", releasedEntries, " entries / ",
        releasedKeys, " pending keys."));
    }
  }

  void D3D9Rtx::reportUe3StaticVertexCaptureCacheStats() {
    if (!m_frameOptions.ue3LogStaticVertexCaptureCacheStats) {
      return;
    }

    const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();

    // roughly once a second at any plausible frame rate; this is a diagnostic, not a metric
    constexpr uint32_t kStatIntervalFrames = 60;
    if (currentFrame - m_ue3VertexCaptureCacheStatFrameStamp < kStatIntervalFrames) {
      return;
    }
    m_ue3VertexCaptureCacheStatFrameStamp = currentFrame;

    if (m_ue3VertexCaptureCacheDormant) {
      Logger::info(str::format(
        "[RTX-Compatibility][UE3-Capture] static vertex capture cache: dormant, re-testing in ",
        m_ue3VertexCaptureCacheProbeCountdown, " frames."));
      m_ue3VertexCaptureCacheStatReuses = 0;
      m_ue3VertexCaptureCacheStatCaptures = 0;
      m_ue3VertexCaptureCacheStatFrames = 0;
      return;
    }

    const uint64_t considered = m_ue3VertexCaptureCacheStatReuses + m_ue3VertexCaptureCacheStatCaptures;
    const uint32_t reusePercent = considered > 0
      ? uint32_t((m_ue3VertexCaptureCacheStatReuses * 100ull) / considered)
      : 0u;

    Logger::info(str::format(
      "[RTX-Compatibility][UE3-Capture] static vertex capture cache: ",
      m_ue3VertexCaptureCache.size(), " retained entries, ",
      m_ue3VertexCaptureCacheBytes / (1024 * 1024), " MiB, ",
      m_ue3VertexCaptureAdmission.size(), " keys awaiting admission, ",
      reusePercent, "% of eligible draws reused over the last ",
      m_ue3VertexCaptureCacheStatFrames, " frames, ",
      m_ue3VertexCaptureCacheEvictions, " total evictions."));

    m_ue3VertexCaptureCacheStatReuses = 0;
    m_ue3VertexCaptureCacheStatCaptures = 0;
    m_ue3VertexCaptureCacheStatFrames = 0;
  }

  void D3D9Rtx::pruneUe3GeometryMemoCache() {
    constexpr uint32_t kMaxUntouchedFrames = 600;
    const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();
    // In-flight workers hold the entry via shared_ptr, so erasing here is always safe.
    m_ue3GeometryMemoCache.erase_if([&](auto it) {
      return currentFrame - it->second->lastFrameTouched > kMaxUntouchedFrames;
    });
  }

  template<typename T>
  void D3D9Rtx::copyIndices(const uint32_t indexCount, T*& pIndicesDst, T* pIndices, uint32_t& minIndex, uint32_t& maxIndex) {
    ScopedCpuProfileZone();

    assert(indexCount >= 3);

    // Find min/max index
    {
      ScopedCpuProfileZoneN("Find min/max");

      fast::findMinMax<T>(indexCount, pIndices, minIndex, maxIndex);
    }

    // Modify the indices if the min index is non-zero
    {
      ScopedCpuProfileZoneN("Copy indices");

      if (minIndex != 0) {
        fast::copySubtract<T>(pIndicesDst, pIndices, indexCount, (T) minIndex);
      } else {
        memcpy(pIndicesDst, pIndices, sizeof(T) * indexCount);
      }
    }
  }

  template<typename T>
  DxvkBufferSlice D3D9Rtx::processIndexBuffer(const uint32_t indexCount, const uint32_t startIndex, const IndexContext& indexCtx, uint32_t& minIndex, uint32_t& maxIndex) {
    ScopedCpuProfileZone();

    const uint32_t indexStride = sizeof(T);
    const size_t numIndexBytes = indexCount * indexStride;
    const size_t indexOffset = indexStride * startIndex;

    auto processing = [this, &indexCtx, indexCount](const size_t offset, const size_t size) -> D3D9CommonBuffer::RemixIndexBufferMemoizationData {
      D3D9CommonBuffer::RemixIndexBufferMemoizationData result;

      // Get our slice of the staging ring buffer
      result.slice = m_rtStagingData.alloc(CACHE_LINE_SIZE, size);

      // Acquire prevents the staging allocator from re-using this memory
      result.slice.buffer()->acquire(DxvkAccess::Read);

      const uint8_t* pBaseIndex = (uint8_t*) indexCtx.indexBuffer.mapPtr + offset;

      T* pIndices = (T*) pBaseIndex;
      T* pIndicesDst = (T*) result.slice.mapPtr(0);
      copyIndices<T>(indexCount, pIndicesDst, pIndices, result.min, result.max);

      return result;
    };

    if (m_frameOptions.enableIndexBufferMemoization && indexCtx.ibo != nullptr) {
      // If we have an index buffer, we can utilize memoization
      D3D9CommonBuffer::RemixIboMemoizer& memoization = indexCtx.ibo->remixMemoization;
      const auto result = memoization.memoize(indexOffset, numIndexBytes, processing);
      minIndex = result.min;
      maxIndex = result.max;
      return result.slice;
    }

    // No index buffer (so no memoization) - this could be a DrawPrimitiveUP call (where IB data is passed inline)
    const auto result = processing(indexOffset, numIndexBytes);
    minIndex = result.min;
    maxIndex = result.max;
    return result.slice;
  }

  DxvkBufferSlice allocVertexCaptureBuffer(DxvkDevice* pDevice, const VkDeviceSize size) {
    DxvkBufferCreateInfo info;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    info.access = VK_ACCESS_TRANSFER_READ_BIT;
    info.stages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    info.size = size;
    // Own category rather than AppBuffer: these are Remix-side allocations whose lifetime the
    // runtime controls (per-draw, or held by the UE3 static capture cache), so lumping them in
    // with the game's own buffers hides them from memory profiling.
    return DxvkBufferSlice(pDevice->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXVertexCapture, "Vertex Capture Buffer"));
  }

  bool D3D9Rtx::prepareVertexCapture(const int vertexIndexOffset, const Ue3CapturePositionSource positionSource) {
    ScopedCpuProfileZone();

    static_assert(sizeof CapturedVertex == 48, "The injected shader code is expecting this exact structure size to work correctly, see emitVertexCaptureWrite in dxso_compiler.cpp");

    // vertex capture requires invertible transforms (projection and affine inverses)
    // iif these are singular, inverse() will otherwise trip math validation /produce invalid data
    {
      constexpr double kDetEps = 1e-24;
      auto detOk = [&](const Matrix4& m) {
        const double det = determinant(m);
        return std::isfinite(det) && std::abs(det) > kDetEps;
      };

      const auto& t = m_activeDrawCallState.transformData;
      // objectToWorld is inverted for every source; the projection and view inverses only
      // matter to the clip-space path, so a singular projection cannot veto an exact capture.
      const bool transformsOk = detOk(t.objectToWorld)
        && (positionSource != Ue3CapturePositionSource::ClipReconstruction
            || (detOk(t.viewToProjection) && detOk(t.worldToView)));
      if (!transformsOk) {
        ONCE(Logger::warn("[RTX-Compatibility] Skipping vertex capture due to non-invertible transform(s)."));
        return false;
      }
    }

    auto BoundShaderHas = [&](const D3D9CommonShader* shader, DxsoUsage usage, bool inOut)-> bool {
      if (shader == nullptr)
        return false;

      const auto& sgn = inOut ? shader->GetIsgn() : shader->GetOsgn();
      for (uint32_t i = 0; i < sgn.elemCount; i++) {
        const auto& decl = sgn.elems[i];
        if (decl.semantic.usageIndex == 0 && decl.semantic.usage == usage)
          return true;
      }
      return false;
    };

    auto BoundShaderHasAnyUsageIndex = [&](const D3D9CommonShader* shader, DxsoUsage usage, bool inOut)-> bool {
      if (shader == nullptr)
        return false;

      const auto& sgn = inOut ? shader->GetIsgn() : shader->GetOsgn();
      for (uint32_t i = 0; i < sgn.elemCount; i++) {
        const auto& decl = sgn.elems[i];
        if (decl.semantic.usage == usage)
          return true;
      }
      return false;
    };

    auto FindVsTexcoordOutputRegister = [&](const D3D9CommonShader* shader, uint32_t usageIndex) -> uint32_t {
      if (shader == nullptr)
        return std::numeric_limits<uint32_t>::max();

      const auto& osgn = shader->GetOsgn();
      for (uint32_t i = 0; i < osgn.elemCount; i++) {
        const auto& decl = osgn.elems[i];
        if (decl.semantic.usage == DxsoUsage::Texcoord && decl.semantic.usageIndex == usageIndex)
          return decl.regNumber;
      }
      return std::numeric_limits<uint32_t>::max();
    };

    auto FindVsTexcoordOutputRegisterByRegNumber = [&](const D3D9CommonShader* shader, uint32_t regNumber) -> uint32_t {
      if (shader == nullptr)
        return std::numeric_limits<uint32_t>::max();

      const auto& osgn = shader->GetOsgn();
      for (uint32_t i = 0; i < osgn.elemCount; i++) {
        const auto& decl = osgn.elems[i];
        if (decl.semantic.usage == DxsoUsage::Texcoord && decl.regNumber == regNumber)
          return decl.regNumber;
      }
      return std::numeric_limits<uint32_t>::max();
    };

    auto FindUniqueVsTexcoordOutputRegister = [&](const D3D9CommonShader* shader) -> uint32_t {
      if (shader == nullptr)
        return std::numeric_limits<uint32_t>::max();

      const auto& osgn = shader->GetOsgn();
      uint32_t foundReg = std::numeric_limits<uint32_t>::max();
      uint32_t foundCount = 0;
      for (uint32_t i = 0; i < osgn.elemCount; i++) {
        const auto& decl = osgn.elems[i];
        if (decl.semantic.usage != DxsoUsage::Texcoord)
          continue;

        foundReg = decl.regNumber;
        foundCount++;
        if (foundCount > 1)
          return std::numeric_limits<uint32_t>::max();
      }

      return foundCount == 1
        ? foundReg
        : std::numeric_limits<uint32_t>::max();
    };

    // Get common shaders to query what data we can capture
    const D3D9CommonShader* vertexShader = d3d9State().vertexShader.ptr() != nullptr ? d3d9State().vertexShader->GetCommonShader() : nullptr;

    RasterGeometry& geoData = m_activeDrawCallState.geometryData;

    const bool hasIaTexcoord = geoData.texcoordBuffer.defined();
    const bool forceIaTexcoordForOutlier = m_forceIaTexcoordForOutlier && hasIaTexcoord;

    // Known stride for vertex capture buffers
    const uint32_t stride = sizeof(CapturedVertex);
    const size_t vertexCaptureDataSize = align(geoData.vertexCount * stride, CACHE_LINE_SIZE);

    DxvkBufferSlice slice = allocVertexCaptureBuffer(m_parent->GetDXVKDevice().ptr(), vertexCaptureDataSize);

    geoData.positionBuffer = RasterBuffer(slice, 0, stride, VK_FORMAT_R32G32B32A32_SFLOAT);
    assert(geoData.positionBuffer.offset() % 4 == 0);

    // Obey the deterministic UV resolution made in processTextures:
    // - ProvenIa: the exact IA texcoord set is bound by processVertices; do not capture UVs
    // - CaptureInterpolant: capture the proven interpolant register/components (correct by
    //   construction for procedural or otherwise unprovable VS-side UV math)
    // - LegacyTss: no provable resolution; keep the legacy capture cascade
    uint32_t capturedTexcoordOutputRegister = std::numeric_limits<uint32_t>::max();
    switch (m_uvResolutionMode) {
    case UvResolutionMode::ProvenIa:
      if (!hasIaTexcoord) {
        // proven IA set is missing from this draw's declaration (degenerate);
        // the captured interpolant is still exact
        capturedTexcoordOutputRegister = FindVsTexcoordOutputRegister(vertexShader, m_texcoordIndex);
      }
      break;
    case UvResolutionMode::CaptureInterpolant:
      capturedTexcoordOutputRegister = FindVsTexcoordOutputRegister(vertexShader, m_texcoordIndex);
      if (capturedTexcoordOutputRegister == std::numeric_limits<uint32_t>::max())
        capturedTexcoordOutputRegister = FindUniqueVsTexcoordOutputRegister(vertexShader);
      break;
    case UvResolutionMode::LegacyTss:
    default:
      capturedTexcoordOutputRegister = FindVsTexcoordOutputRegister(vertexShader, m_texcoordIndex);
      if (capturedTexcoordOutputRegister == std::numeric_limits<uint32_t>::max()) {
        if (!m_frameOptions.ue3EngineMode)
          capturedTexcoordOutputRegister = FindVsTexcoordOutputRegisterByRegNumber(vertexShader, m_texcoordIndex);
      }
      if (capturedTexcoordOutputRegister == std::numeric_limits<uint32_t>::max())
        capturedTexcoordOutputRegister = FindUniqueVsTexcoordOutputRegister(vertexShader);
      break;
    }
    if (forceIaTexcoordForOutlier)
      capturedTexcoordOutputRegister = std::numeric_limits<uint32_t>::max();

    if (useVertexCapturedTexcoords()
        && BoundShaderHasAnyUsageIndex(vertexShader, DxsoUsage::Texcoord, false)
        && capturedTexcoordOutputRegister == std::numeric_limits<uint32_t>::max()) {
      capturedTexcoordOutputRegister = FindUniqueVsTexcoordOutputRegister(vertexShader);
    }

    // By default we only capture VS output texcoords when the input vertex declaration didn't
    // already provide them. Overriding valid input texcoords with the VS output is opt-in
    // (useVertexCapturedTexcoords), since the data a VS writes to the :TEXCOORD attribute isn't
    // always actual UVs and its memory layout can't be assumed for all games.
    // CaptureInterpolant is exempt: that mode is only selected when UV analysis proved the
    // sampled UV is VS-side math, so the bound IA set is just a backstop and must not
    // suppress the capture.
    const bool captureVsTexcoords =
      BoundShaderHasAnyUsageIndex(vertexShader, DxsoUsage::Texcoord, false)
      && capturedTexcoordOutputRegister != std::numeric_limits<uint32_t>::max()
      && (m_uvResolutionMode == UvResolutionMode::CaptureInterpolant
          || useVertexCapturedTexcoords()
          || !geoData.texcoordBuffer.defined()
          || !RtxGeometryUtils::isTexcoordFormatValid(geoData.texcoordBuffer.vertexFormat()));

    if (captureVsTexcoords) {
      const uint32_t texcoordOffset = offsetof(CapturedVertex, texcoord0);
      geoData.texcoordBuffer = RasterBuffer(slice, texcoordOffset, stride, VK_FORMAT_R32G32_SFLOAT);
      assert(geoData.texcoordBuffer.offset() % 4 == 0);
    }

    // normals for vertex-capture draws
    // captured positions are post-skinning for GPU-skinned draws whichever source they come from, so when a
    // skinned shader doesn't output a normal semantic IA normals alone are bind-pose and do not
    // match captured positions, for UE3 we therefore prioritise:
    // 1 VS NORMAL output when available
    // 2 VS COLOR0 encoded skinned normal output
    // 3 bone-skinned IA normal reconstruction in vertex-capture shader using BoneMatrices CTAB metadata
    // 4 bind-pose IA fallback only when none of the above can be used
    const bool vsOutputsNormal = BoundShaderHas(vertexShader, DxsoUsage::Normal, false);
    const bool vsHasNormalInput = BoundShaderHas(vertexShader, DxsoUsage::Normal, true);
    const bool normalFromDecl = geoData.normalBuffer.defined();
    const bool isGpuSkinned = geoData.blendWeightBuffer.defined() || geoData.blendIndicesBuffer.defined();

    const bool vsOutputsColor0 = BoundShaderHas(vertexShader, DxsoUsage::Color, false);
    const bool hasBlendWeights = geoData.blendWeightBuffer.defined();
    const bool hasBlendIndices = geoData.blendIndicesBuffer.defined();
    const bool hasAnyBlendStream = hasBlendWeights || hasBlendIndices;
    const Ue3VsShaderCtabInfo* ue3CtabInfo = m_currentUe3CtabInfo.has_value() ? &(*m_currentUe3CtabInfo) : nullptr;
    const bool hasBoneMatricesInCtab = ue3CtabInfo != nullptr &&
                                       ue3CtabInfo->hasBoneMatrices &&
                                       ue3CtabInfo->boneMatricesRegisterCount >= 3;
    const bool canUseBoneSkinnedNormalCapture =
      isGpuSkinned &&
      !vsOutputsNormal &&
      !vsOutputsColor0 &&
      vsHasNormalInput &&
      normalFromDecl &&
      hasAnyBlendStream &&
      hasBoneMatricesInCtab;

    if (Logger::logLevel() <= LogLevel::Debug) {
      const uint32_t normalDiagKey =
        (vsOutputsNormal ? 1u : 0u) | (vsHasNormalInput ? 2u : 0u) |
        (normalFromDecl ? 4u : 0u) | (isGpuSkinned ? 8u : 0u) |
        (vsOutputsColor0 ? 16u : 0u) |
        (hasBlendWeights ? 32u : 0u) |
        (hasBlendIndices ? 64u : 0u) |
        (hasBoneMatricesInCtab ? 128u : 0u) |
        (canUseBoneSkinnedNormalCapture ? 256u : 0u) |
        (normalFromDecl ? (uint32_t(geoData.normalBuffer.vertexFormat()) << 8) : 0u);
      static fast_unordered_set s_loggedNormalDiag;
      if (s_loggedNormalDiag.insert(normalDiagKey).second) {
        const char* caseLabel = "Case3-keepIA";
        if (vsOutputsNormal) caseLabel = "Case1-vsOutputNormal";
        else if (isGpuSkinned && vsOutputsColor0) caseLabel = "Case2a-COLOR0skinned";
        else if (canUseBoneSkinnedNormalCapture) caseLabel = "Case2b-boneSkinCapture";
        else if (isGpuSkinned && normalFromDecl) caseLabel = "Case2c-bindPoseFallback";
        else if (isGpuSkinned) caseLabel = "Case2d-noNormals";

        Logger::debug(str::format(
          "[RTX-Compatibility] Vertex capture normal [", caseLabel, "]: vsOutputsNormal=", vsOutputsNormal,
          ", vsHasNormalInput=", vsHasNormalInput,
          ", normalFromDecl=", normalFromDecl,
          normalFromDecl ? str::format(", declFmt=", geoData.normalBuffer.vertexFormat()).c_str() : "",
          ", isGpuSkinned=", isGpuSkinned,
          ", vsOutputsColor0=", vsOutputsColor0,
          ", hasBlendWeights=", hasBlendWeights,
          ", hasBlendIndices=", hasBlendIndices,
          ", hasBoneMatricesInCtab=", hasBoneMatricesInCtab,
          ", vertexCount=", geoData.vertexCount));
      }
    }

    uint32_t vertexCaptureFlags = 0;
    switch (positionSource) {
    case Ue3CapturePositionSource::InputAssembler:
      vertexCaptureFlags |= kVertexCaptureFlag_PositionFromInput;
      break;
    case Ue3CapturePositionSource::PreProjectionRegister:
      vertexCaptureFlags |= kVertexCaptureFlag_PositionFromPreProjection;
      break;
    case Ue3CapturePositionSource::ClipReconstruction:
      break;
    }

    if (vsOutputsNormal && (m_frameOptions.useVertexCapturedNormals || m_frameOptions.ue3EngineMode)) {
      // 1: VS outputs NORMAL - use vertex-captured normals (they match the captured positions)
      const uint32_t normalOffset = offsetof(CapturedVertex, normal0);
      geoData.normalBuffer = RasterBuffer(slice, normalOffset, stride, VK_FORMAT_R32G32B32_SFLOAT);
      assert(geoData.normalBuffer.offset() % 4 == 0);
    } else if (isGpuSkinned && vsOutputsColor0) {
      // 2: GPU-skinned mesh, VS doesn't output NORMAL but has COLOR0 output
      // UE3 GpuSkinVertexFactory outputs the bone-transformed world-space tangent basis normal
      // through COLOR0 as (normal * 0.5 + 0.5), so we tell the vertex capture shader to decode COLOR0
      // as the normal source instead of using the bind-pose IA NORMAL
      vertexCaptureFlags |= kVertexCaptureFlag_NormalFromColor0;
      const uint32_t normalOffset = offsetof(CapturedVertex, normal0);
      geoData.normalBuffer = RasterBuffer(slice, normalOffset, stride, VK_FORMAT_R32G32B32_SFLOAT);
      assert(geoData.normalBuffer.offset() % 4 == 0);
      ONCE(Logger::info("[RTX-Compatibility] UE3 GPU-skinned mesh: capturing normal from VS COLOR0 output (skinned tangent basis)."));
    } else if (canUseBoneSkinnedNormalCapture) {
      // 2b: GPU-skinned mesh without VS NORMAL/COLOR0 output
      // reconstruct skinned normals in the vertex capture shader using BoneMatrices from VS constants
      vertexCaptureFlags |= kVertexCaptureFlag_NormalBoneSkinning;
      if (geoData.normalBuffer.vertexFormat() == VK_FORMAT_R8G8B8A8_UNORM)
        vertexCaptureFlags |= kVertexCaptureFlag_NormalInputEncodedUByte4;
      if (hasBlendIndices && geoData.blendIndicesBuffer.vertexFormat() == VK_FORMAT_R8G8B8A8_UNORM)
        vertexCaptureFlags |= kVertexCaptureFlag_BlendIndicesInputNormalized;
      if (hasBlendWeights && geoData.blendWeightBuffer.vertexFormat() == VK_FORMAT_R8G8B8A8_USCALED)
        vertexCaptureFlags |= kVertexCaptureFlag_BlendWeightsInputUnnormalized;

      const uint32_t normalOffset = offsetof(CapturedVertex, normal0);
      geoData.normalBuffer = RasterBuffer(slice, normalOffset, stride, VK_FORMAT_R32G32B32_SFLOAT);
      assert(geoData.normalBuffer.offset() % 4 == 0);
      ONCE(Logger::info(str::format(
        "[RTX-Compatibility] UE3 GPU-skinned mesh: reconstructing normals via BoneMatrices in vertex capture shader (boneBaseReg=c",
        ue3CtabInfo->boneMatricesRegisterIndex, ", boneRegCount=", ue3CtabInfo->boneMatricesRegisterCount,
        ", hasBlendWeights=", hasBlendWeights, ", hasBlendIndices=", hasBlendIndices, ").")));
    } else if (isGpuSkinned && normalFromDecl) {
      // 2c: GPU-skinned but no COLOR0 output and no usable bone CTAB metadata
      // keep bind-pose IA normals as smooth fallback, they're in the wrong orientation for animated poses (missing bone transform), but provide smooth per-vertex interpolation
      ONCE(Logger::info("[RTX-Compatibility] UE3 GPU-skinned mesh without COLOR0 output: using bind-pose IA normals as smooth fallback."));
    }
    // 3 : non-skinned, VS doesn't output NORMAL, just keep IA normals from processVertices they're in the same object space as the captured positions anyway..

    // Check if we should/can get colors
    if (BoundShaderHas(vertexShader, DxsoUsage::Color, false) && d3d9State().pixelShader.ptr() == nullptr) {
      const uint32_t colorOffset = offsetof(CapturedVertex, color0);
      geoData.color0Buffer = RasterBuffer(slice, colorOffset, stride, VK_FORMAT_B8G8R8A8_UNORM);
      assert(geoData.color0Buffer.offset() % 4 == 0);
    }

    auto constants = m_vsVertexCaptureData->allocSlice();

    // Upload
    auto& data = *reinterpret_cast<D3D9RtxVertexCaptureData*>(constants.mapPtr);
    // Only the clip-space path consumes these, and only it required them to be invertible
    // above, so inverting them unconditionally would trip math validation on a draw whose
    // projection is singular but whose position comes from an exact source.
    if (positionSource == Ue3CapturePositionSource::ClipReconstruction) {
      data.invProj = inverse(m_activeDrawCallState.transformData.viewToProjection);
      data.viewToWorld = inverseAffine(m_activeDrawCallState.transformData.worldToView);
    } else {
      data.invProj = Matrix4();
      data.viewToWorld = Matrix4();
    }
    data.worldToObject = inverseAffine(m_activeDrawCallState.transformData.objectToWorld);
    data.normalTransform = m_activeDrawCallState.transformData.objectToWorld;
    // note - BaseVertexIndex can be negative, so we store the raw value as uint32 so the shader's unsigned
    // subtraction (uVertexId - baseVertex) behaves correctly for two's-complement values
    data.baseVertex = (uint32_t)vertexIndexOffset;
    data.flags = vertexCaptureFlags;
    data.boneMatricesBaseReg = 0;
    data.boneCount = 0;
    data.texcoordOutputRegister = capturedTexcoordOutputRegister;
    data.texcoordCompU = m_texcoordCompU & 0x3u;
    data.texcoordCompV = m_texcoordCompV & 0x3u;
    if ((vertexCaptureFlags & kVertexCaptureFlag_NormalBoneSkinning) != 0 && ue3CtabInfo != nullptr) {
      data.boneMatricesBaseReg = ue3CtabInfo->boneMatricesRegisterIndex;
      data.boneCount = std::min(ue3CtabInfo->boneMatricesRegisterCount / 3u, 256u);
    }

    m_parent->EmitCs([cVertexDataSlice = slice,
                      cConstantBuffer = m_vsVertexCaptureData,
                      cConstants = constants](DxvkContext* ctx) {
      // Bind the new constants to buffer
      ctx->invalidateBuffer(cConstantBuffer, cConstants);

      // Invalidate rest of the members
      // customWorldToProjection is not invalidated as its use is controlled by D3D9SpecConstantId::CustomVertexTransformEnabled being enabled
      ctx->bindResourceBuffer(getVertexCaptureBufferSlot(), cVertexDataSlice);
    });

    return true;
  }

  void D3D9Rtx::processVertices(const VertexContext vertexContext[caps::MaxStreams], int vertexIndexOffset, RasterGeometry& geoData) {
    // One zone per draw, not per vertex element: with UE3's ~9 elements per declaration a
    // per-element zone emits tens of thousands of Tracy events per frame whose begin/end
    // overhead lands in the enclosing internalPrepareDraw zone and distorts captures.
    ScopedCpuProfileZoneN("Process Vertices");
    DxvkBufferSlice streamCopies[caps::MaxStreams] {};

    // FParticleInstancedMeshVertexFactory hands the mesh's TangentZ to the BINORMAL semantic
    // rather than NORMAL (see classifyUe3VertexFactory), so this is the one factory whose surface
    // normal has to be read from there.
    const bool normalFromBinormal =
      m_currentUe3VertexFactory == Ue3VertexFactoryType::ParticleInstancedMesh;

    // Process vertex buffers from CPU
    for (const auto& element : d3d9State().vertexDecl->GetElements()) {
      // Get vertex context
      const VertexContext& ctx = vertexContext[element.Stream];

      if (ctx.mappedSlice.handle == VK_NULL_HANDLE)
        continue;

      // An instance-data stream advances once per instance, so indexing it by vertex reads an
      // unrelated instance's record (and walks off the end of a short instance buffer).
      if (element.Stream < caps::MaxStreams &&
          (m_currentUe3Instancing.instanceDataStreamMask & (1u << element.Stream)) != 0)
        continue;

      const int32_t vertexOffset = ctx.offset + ctx.stride * vertexIndexOffset;
      const uint32_t numVertexBytes = ctx.stride * geoData.vertexCount;

      // Validating index data here, vertexCount and vertexIndexOffset accounts for the min/max indices
      if (m_frameOptions.validateCPUIndexData) {
        if (ctx.mappedSlice.length < vertexOffset + numVertexBytes) {
          throw DxvkError("Invalid draw call");
        }
      }

      // TODO: Simplify this by refactoring RasterGeometry to contain an array of RasterBuffer's
      RasterBuffer* targetBuffer = nullptr;
      switch (element.Usage) {
      case D3DDECLUSAGE_POSITIONT:
      case D3DDECLUSAGE_POSITION:
        if (element.UsageIndex == 0)
          targetBuffer = &geoData.positionBuffer;
        break;
      case D3DDECLUSAGE_BLENDWEIGHT:
        if (element.UsageIndex == 0)
          targetBuffer = &geoData.blendWeightBuffer;
        break;
      case D3DDECLUSAGE_BLENDINDICES:
        if (element.UsageIndex == 0)
          targetBuffer = &geoData.blendIndicesBuffer;
        break;
      case D3DDECLUSAGE_NORMAL:
        if (element.UsageIndex == 0 && !normalFromBinormal)
          targetBuffer = &geoData.normalBuffer;
        break;
      case D3DDECLUSAGE_BINORMAL:
        if (element.UsageIndex == 0 && normalFromBinormal)
          targetBuffer = &geoData.normalBuffer;
        break;
      case D3DDECLUSAGE_TEXCOORD:
        // A D3DCOLOR-typed TEXCOORD under UE3 is a vertex lightmap coefficient stream, never a
        // coordinate (see resolveIaTexcoordAvoidingNonUvElements).
        if (m_iaTexcoordIndex <= MAXD3DDECLUSAGEINDEX && element.UsageIndex == m_iaTexcoordIndex &&
            !(m_frameOptions.ue3EngineMode && element.Type == D3DDECLTYPE_D3DCOLOR))
          targetBuffer = &geoData.texcoordBuffer;
        break;
      case D3DDECLUSAGE_COLOR:
        if (element.UsageIndex == 0 &&
            !m_frameOptions.ignoreAllVertexColorBakedLighting && !m_frameOptions.ue3EngineMode &&
            !lookupHash(*m_frameOptions.ignoreBakedLightingTextures, m_activeDrawCallState.materialData.colorTextures[0].getImageHash())) {
          // only treat COLOR0 as a packed 8-bit UNORM color, UE3 can use COLOR semantics for non-color data which the rtx interleaver does not interpret as vertex color
          const VkFormat fmt = DecodeDecltype(D3DDECLTYPE(element.Type));
          if (fmt == VK_FORMAT_B8G8R8A8_UNORM || fmt == VK_FORMAT_R8G8B8A8_UNORM) {
            targetBuffer = &geoData.color0Buffer;
          }
        }
        break;
      }

      if (targetBuffer != nullptr) {
        assert(!targetBuffer->defined());

        // Only do once for each stream
        if (!streamCopies[element.Stream].defined()) {
          // Deep clonning a buffer object is not cheap (320 bytes to copy and other work). Set a min-size threshold.
          const uint32_t kMinSizeToClone = 512;

          // Check if buffer is actualy a d3d9 orphan
          const bool isOrphan = !(ctx.buffer.getSliceHandle() == ctx.mappedSlice);
          const bool canUseBuffer = ctx.canUseBuffer && m_forceGeometryCopy == false;

          if (canUseBuffer && !isOrphan) {
            // Use the buffer directly if it is not an orphan
            if (ctx.pVBO != nullptr && ctx.pVBO->NeedsUpload())
              m_parent->FlushBuffer(ctx.pVBO);

            streamCopies[element.Stream] = ctx.buffer.subSlice(vertexOffset, numVertexBytes);
          } else if (canUseBuffer && numVertexBytes > kMinSizeToClone) {
            // Create a clone for the orphaned physical slice
            auto clone = ctx.buffer.buffer()->clone();
            clone->rename(ctx.mappedSlice);
            streamCopies[element.Stream] = DxvkBufferSlice(clone, ctx.buffer.offset() + vertexOffset, numVertexBytes);
          } else {
            streamCopies[element.Stream] = m_rtStagingData.alloc(CACHE_LINE_SIZE, numVertexBytes);

            // Acquire prevents the staging allocator from re-using this memory
            streamCopies[element.Stream].buffer()->acquire(DxvkAccess::Read);

            memcpy(streamCopies[element.Stream].mapPtr(0), (uint8_t*) ctx.mappedSlice.mapPtr + vertexOffset, numVertexBytes);
          }
        }

        VkFormat fmt = DecodeDecltype(D3DDECLTYPE(element.Type));
        uint32_t elementOffset = element.Offset;

        // note: m_texcoordCompU/V select VS interpolant components (capture path only);
        // a proven IA texcoord element is always a plain (u, v) pair and is read as-is

        // UE3 packed normals use D3DDECLTYPE_UBYTE4 (not normalised) so for remix purposes we want a decoded
        // [-1, 1] normal and the interleaver supports VK_FORMAT_R8G8B8A8_UNORM for this
        if (targetBuffer == &geoData.normalBuffer && fmt == VK_FORMAT_R8G8B8A8_USCALED) {
          fmt = VK_FORMAT_R8G8B8A8_UNORM;
        }
        *targetBuffer = RasterBuffer(streamCopies[element.Stream], elementOffset, ctx.stride, fmt);
        assert(targetBuffer->offset() % 4 == 0);
      }
    }
  }

  bool D3D9Rtx::processRenderState(const DrawContext& drawContext) {
    ScopedCpuProfileZone();
    DrawCallTransforms& transformData = m_activeDrawCallState.transformData;
    m_forceIaTexcoordForOutlier = false;

    // m_activeDrawCallState is reused across draws, so these fields need an explicit per-draw reset
    m_activeDrawCallState.allowMainCameraUpdate = true;
    m_activeDrawCallState.programmableVertexShaderBytecodeHash = 0;
    m_activeDrawCallState.ue3PassDescription = describeUe3PassType(m_currentUe3PassType);
    m_activeDrawCallState.isUe3ForegroundDpg = m_ue3ForegroundDpgActive;

    const bool isUe3Mode = m_frameOptions.ue3EngineMode;
    const bool effectiveUe3Camera = m_frameOptions.ue3CameraFromShaderConstants || isUe3Mode;
    const bool effectiveUe3ObjectToWorld = m_frameOptions.ue3ObjectToWorldFromShaderConstants || isUe3Mode;
    const bool effectiveUseWorldMatricesForShaders = m_frameOptions.useWorldMatricesForShaders && !isUe3Mode;

    // When games use vertex shaders, the object to world transforms can be unreliable, and so we can ignore them.
    const bool useObjectToWorldTransform = !m_parent->UseProgrammableVS() || (m_parent->UseProgrammableVS() && m_frameOptions.useVertexCapture && effectiveUseWorldMatricesForShaders);
    transformData.objectToWorld = useObjectToWorldTransform ? d3d9State().transforms[GetTransformIndex(D3DTS_WORLD)] : Matrix4();

    transformData.worldToView = d3d9State().transforms[GetTransformIndex(D3DTS_VIEW)];
    transformData.viewToProjection = d3d9State().transforms[GetTransformIndex(D3DTS_PROJECTION)];

    const bool usesProgrammableVs = m_parent->UseProgrammableVS();
    const D3D9CommonShader* vertexShaderCommon =
      usesProgrammableVs && d3d9State().vertexShader.ptr() != nullptr
        ? d3d9State().vertexShader->GetCommonShader()
        : nullptr;

    const Ue3VsShaderCtabInfo* ue3CtabInfoPtr = nullptr;
    m_currentUe3CtabInfo.reset();
    m_currentUe3VsHashExclusions = nullptr;
    bool ue3CameraUsedTranspose = false;

    const bool needsUe3CtabInfo =
      effectiveUe3Camera ||
      effectiveUe3ObjectToWorld ||
      m_frameOptions.useVertexCapture;
    if (usesProgrammableVs && vertexShaderCommon != nullptr &&
        needsUe3CtabInfo) {
      auto parseCtabInfo = [&](const std::vector<uint8_t>& bytecode,
                               std::vector<Ue3VsConstantSymbol>* outSymbols) -> Ue3VsShaderCtabInfo {
        Ue3VsShaderCtabInfo info;
        info.initialized = true;

        try {
          if (bytecode.size() < sizeof(uint32_t) || (bytecode.size() % sizeof(uint32_t)) != 0)
            return info;

          const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
          const uint32_t headerToken = tokens[0];
          const uint32_t headerTypeMask = headerToken & 0xffff0000u;

          DxsoProgramType programType;
          if (headerTypeMask == 0xffff0000u)
            programType = DxsoProgramTypes::PixelShader;
          else if (headerTypeMask == 0xfffe0000u)
            programType = DxsoProgramTypes::VertexShader;
          else
            return info;

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
            return info;

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

          uint32_t inferredBoneMatricesRegisterIndex = 0;
          uint32_t inferredBoneMatricesRegisterCount = 0;

          if (outSymbols != nullptr) {
            outSymbols->reserve(ctab.m_constantData.size());
            for (const DxsoCtab::Constant& c : ctab.m_constantData) {
              // float constant registers only - the churn diagnostic compares fConsts
              if (c.registerSet != kD3dxRegisterSetFloat4 || c.registerCount == 0) {
                continue;
              }
              Ue3VsConstantSymbol symbol;
              symbol.registerIndex = c.registerIndex;
              symbol.registerCount = c.registerCount;
              symbol.name = c.name;
              outSymbols->push_back(std::move(symbol));
            }
          }

          for (const DxsoCtab::Constant& c : ctab.m_constantData) {
            const std::string name = lower(c.name);

            // ViewProjectionMatrix (4 registers)
            if (!info.hasViewProjectionMatrix && c.registerCount >= 4) {
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
                info.hasViewProjectionMatrix = true;
                info.viewProjectionMatrixRegisterIndex = c.registerIndex;
                info.viewProjectionMatrixRegisterCount = c.registerCount;
              }
            }

            // CameraPosition (1 register)
            if (!info.hasCameraPosition && c.registerCount >= 1) {
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
                info.hasCameraPosition = true;
                info.cameraPositionRegisterIndex = c.registerIndex;
                info.cameraPositionRegisterCount = c.registerCount;
              }
            }

            // LocalToWorld (4 registers)
            if (!info.hasLocalToWorld && c.registerCount >= 4) {
              // prefer an exact match here but also accept nested names e.g. VertexFactory.LocalToWorld
              if (contains(name, "localtoworld") ||
                  contains(name, "local_to_world") ||
                  contains(name, "objecttoworld") ||
                  contains(name, "object_to_world")) {
                if (contains(name, "previouslocaltoworld") ||
                    contains(name, "prevlocaltoworld") ||
                    contains(name, "previous_local_to_world") ||
                    contains(name, "prev_local_to_world"))
                  continue;

                info.hasLocalToWorld = true;
                info.localToWorldRegisterIndex = c.registerIndex;
                info.localToWorldRegisterCount = c.registerCount;
              }
            }

            // WorldToLocal (3 registers, typically float3x3)
            if (!info.hasWorldToLocal && c.registerCount >= 3) {
              if (contains(name, "worldtolocal") ||
                  contains(name, "world_to_local") ||
                  contains(name, "objectinverseworld") ||
                  contains(name, "object_inverse_world")) {
                info.hasWorldToLocal = true;
                info.worldToLocalRegisterIndex = c.registerIndex;
                info.worldToLocalRegisterCount = c.registerCount;
              }
            }

            // BoneMatrices (GPU skinning: commonly N bones * 3 registers in UE3)
            if (!info.hasBoneMatrices && c.registerCount >= 3) {
              const bool explicitBoneName =
                contains(name, "bonematrices") ||
                contains(name, "bone_matrices") ||
                contains(name, "bonematrix") ||
                contains(name, "skinningmatrices") ||
                contains(name, "skinmatrices") ||
                contains(name, "matrixpalette") ||
                contains(name, "bonetransforms") ||
                contains(name, "bone_transforms") ||
                (contains(name, "bone") && (c.registerCount % 3u) == 0u);

              if (explicitBoneName) {
                info.hasBoneMatrices = true;
                info.boneMatricesRegisterIndex = c.registerIndex;
                info.boneMatricesRegisterCount = c.registerCount;
              } else if ((c.registerCount % 3u) == 0u) {
                // fallback for stripped/renamed symbols, in UE3 this is typically a large contiguous
                // c-register range (3 registers per bone) usually starting after c0..c4 camera constants
                const bool isConfirmedSkinVF =
                  m_currentUe3VertexFactory == Ue3VertexFactoryType::GPUSkin ||
                  m_currentUe3VertexFactory == Ue3VertexFactoryType::GPUSkinMorph;
                const uint32_t minRegCount = isConfirmedSkinVF ? 3u : 9u;
                const uint32_t minRegIndex = isConfirmedSkinVF ? 0u : 5u;
                if (c.registerCount >= minRegCount && c.registerIndex >= minRegIndex &&
                    c.registerCount > inferredBoneMatricesRegisterCount) {
                  inferredBoneMatricesRegisterIndex = c.registerIndex;
                  inferredBoneMatricesRegisterCount = c.registerCount;
                }
              }
            }

            // additional UE3 vertexfactory hints used by shader-path UV selection
            if (!info.hasDecalTransform &&
                (contains(name, "worldtodecal") ||
                 contains(name, "world_to_decal") ||
                 contains(name, "bonetodecal") ||
                 contains(name, "bone_to_decal"))) {
              info.hasDecalTransform = true;
            }
            if (!info.hasDecalLocation &&
                (contains(name, "decallocation") ||
                 contains(name, "decal_location"))) {
              info.hasDecalLocation = true;
            }
            if (!info.hasDecalOffset &&
                (contains(name, "decaloffset") ||
                 contains(name, "decal_offset"))) {
              info.hasDecalOffset = true;
            }
            if (!info.hasTextureCoordinateScaleBias &&
                (contains(name, "texturecoordinatescalebias") ||
                 contains(name, "texture_coordinate_scale_bias"))) {
              info.hasTextureCoordinateScaleBias = true;
            }
            if (!info.hasLightMapCoordinateScaleBias &&
                (contains(name, "lightmapcoordinatescalebias") ||
                 contains(name, "light_map_coordinate_scale_bias"))) {
              info.hasLightMapCoordinateScaleBias = true;
            }
            if (!info.hasShadowCoordinateScaleBias &&
                (contains(name, "shadowcoordinatescalebias") ||
                 contains(name, "shadow_coordinate_scale_bias"))) {
              info.hasShadowCoordinateScaleBias = true;
            }
            if (!info.hasViewToLocal &&
                (contains(name, "viewtolocal") ||
                 contains(name, "view_to_local"))) {
              info.hasViewToLocal = true;
            }
            if (!info.hasWindMatrices &&
                (contains(name, "windmatrices") ||
                 contains(name, "wind_matrices") ||
                 contains(name, "windmatrix") ||
                 contains(name, "wind_matrix"))) {
              info.hasWindMatrices = true;
            }
          }

          if (!info.hasBoneMatrices && inferredBoneMatricesRegisterCount >= 3u) {
            const bool isConfirmedSkinVF =
              m_currentUe3VertexFactory == Ue3VertexFactoryType::GPUSkin ||
              m_currentUe3VertexFactory == Ue3VertexFactoryType::GPUSkinMorph;
            if (isConfirmedSkinVF || inferredBoneMatricesRegisterCount >= 9u) {
              info.hasBoneMatrices = true;
              info.boneMatricesRegisterIndex = inferredBoneMatricesRegisterIndex;
              info.boneMatricesRegisterCount = inferredBoneMatricesRegisterCount;
            }
          }
        } catch (...) {
          return info;
        }

        return info;
      };

      const auto& bytecode = vertexShaderCommon->GetBytecode();
      const XXH64_hash_t shaderHash = vertexShaderCommon->GetBytecodeHash();

      m_activeDrawCallState.programmableVertexShaderBytecodeHash = shaderHash;

      if (shaderHash != 0) {
        auto it = m_ue3VsShaderCtabCache.find(shaderHash);
        if (it == m_ue3VsShaderCtabCache.end()) {
          std::vector<Ue3VsConstantSymbol> symbols;
          const Ue3VsShaderCtabInfo parsed = parseCtabInfo(bytecode, &symbols);
          m_ue3VsHashExclusionCache.emplace(shaderHash, buildUe3VsHashExclusions(parsed, &symbols));
          m_ue3VsShaderCtabCache.emplace(shaderHash, parsed);
          if (!symbols.empty()) {
            m_ue3VsConstantSymbols.emplace(shaderHash, std::move(symbols));
          }
          it = m_ue3VsShaderCtabCache.find(shaderHash);
        }

        {
          const auto exclusionIt = m_ue3VsHashExclusionCache.find(shaderHash);
          m_currentUe3VsHashExclusions =
            exclusionIt != m_ue3VsHashExclusionCache.end() ? &exclusionIt->second : nullptr;
        }

        if (it != m_ue3VsShaderCtabCache.end()) {
          ue3CtabInfoPtr = &it->second;
          m_currentUe3CtabInfo = *ue3CtabInfoPtr;
          if (isUe3Mode &&
              m_currentUe3VertexFactory == Ue3VertexFactoryType::Local &&
              (ue3CtabInfoPtr->hasDecalTransform ||
               ue3CtabInfoPtr->hasDecalLocation ||
               ue3CtabInfoPtr->hasDecalOffset)) {
            m_currentUe3VertexFactory = Ue3VertexFactoryType::LocalDecal;
          }
        }
      }
    }

    if (usesProgrammableVs && effectiveUe3Camera) {
      uint32_t viewProjReg = kUe3VsrViewProjMatrixRegister;
      uint32_t viewOriginReg = kUe3VsrViewOriginRegister;

      if (ue3CtabInfoPtr != nullptr) {
        if (ue3CtabInfoPtr->hasViewProjectionMatrix)
          viewProjReg = ue3CtabInfoPtr->viewProjectionMatrixRegisterIndex;
        if (ue3CtabInfoPtr->hasCameraPosition)
          viewOriginReg = ue3CtabInfoPtr->cameraPositionRegisterIndex;
      }

      // cache by raw constant values to avoid repeated heavy extraction work per draw call
      struct Ue3CameraConstsKey {
        uint32_t viewProjReg;
        uint32_t viewOriginReg;
        Vector4 regs[5];
      };

      auto tryApplyFromConstants = [&](Matrix4& outWorldToView, Matrix4& outViewToProjection, bool& outUsedTranspose, float& outReconstructionError) -> bool {
        if (viewProjReg + 3 >= caps::MaxFloatConstantsSoftware || viewOriginReg >= caps::MaxFloatConstantsSoftware)
          return false;

        Ue3CameraConstsKey key {};
        key.viewProjReg = viewProjReg;
        key.viewOriginReg = viewOriginReg;
        key.regs[0] = d3d9State().vsConsts.fConsts[viewProjReg + 0];
        key.regs[1] = d3d9State().vsConsts.fConsts[viewProjReg + 1];
        key.regs[2] = d3d9State().vsConsts.fConsts[viewProjReg + 2];
        key.regs[3] = d3d9State().vsConsts.fConsts[viewProjReg + 3];
        key.regs[4] = d3d9State().vsConsts.fConsts[viewOriginReg];

        const XXH64_hash_t constantsHash = XXH3_64bits(&key, sizeof(key));

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
      };

      Matrix4 ue3WorldToView;
      Matrix4 ue3ViewToProjection;
      float ue3CameraReconstructionError = 0.0f;

      // Only draws whose CTAB explicitly names both camera constants may update the Main camera.
      // Fallback-register extractions can be light-space matrices from engine utility shaders
      // (e.g. shadow depth) that still reconstruct as a plausible camera.
      const bool ctabVerifiedCamera =
        ue3CtabInfoPtr != nullptr &&
        ue3CtabInfoPtr->hasViewProjectionMatrix &&
        ue3CtabInfoPtr->hasCameraPosition;

      // Full SceneCapture isolation for probes the viewport heuristic cannot see:
      // reflect/portal probes (e.g. Mirror's Edge scripted building window reflections)
      // render the world through a FMirrorMatrix-premultiplied view and an oblique
      // FClipProjectionMatrix near-plane clip, at viewports scaled to the parent view.
      // Geometry captured through such views is unusable - mirrored views reconstruct
      // reflected world positions, oblique projections are not decomposable - so these
      // draws are dropped outright. Restricted to CTAB-verified cameras: fallback
      // registers can hold arbitrary data that must not trigger capture classification.
      const bool ue3CaptureViewIsolation =
        (m_frameOptions.ue3SkipSceneCapturePasses || isUe3Mode) &&
        ctabVerifiedCamera &&
        isUe3WorldGeometryVertexFactory(m_currentUe3VertexFactory);

      auto classifySceneCaptureView = [&](const char* reason) {
        m_currentUe3PassType = Ue3PassType::SceneCapture;
        // Undo probe draws that armed the main-view gate before mirror/oblique detection.
        if (!m_ue3ForegroundDpgActive) {
          m_ue3SeenMainViewWorldDraw = false;
        }
        m_activeDrawCallState.allowMainCameraUpdate = false;
        m_activeDrawCallState.ue3PassDescription = describeUe3PassType(m_currentUe3PassType);
        logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, reason);
      };

      if (tryApplyFromConstants(ue3WorldToView, ue3ViewToProjection, ue3CameraUsedTranspose, ue3CameraReconstructionError)) {
        // Mirrored view detection: a reflection view premultiplies a mirror (householder)
        // matrix into the view, flipping the sign of the ViewProjection 3x3 determinant.
        // UE3's LH view (axis-swap permutation, det +1) and perspective projection keep the
        // main view's determinant positive. Checked on the raw registers because the
        // extraction reconstructs an orthonormal basis and washes the mirror out; the sign
        // is transpose-invariant so the upload convention does not matter.
        if (ue3CaptureViewIsolation) {
          const Vector4& vpRow0 = d3d9State().vsConsts.fConsts[viewProjReg + 0];
          const Vector4& vpRow1 = d3d9State().vsConsts.fConsts[viewProjReg + 1];
          const Vector4& vpRow2 = d3d9State().vsConsts.fConsts[viewProjReg + 2];
          const float vpDet3 =
            vpRow0.x * (vpRow1.y * vpRow2.z - vpRow1.z * vpRow2.y) -
            vpRow0.y * (vpRow1.x * vpRow2.z - vpRow1.z * vpRow2.x) +
            vpRow0.z * (vpRow1.x * vpRow2.y - vpRow1.y * vpRow2.x);
          if (std::isfinite(vpDet3) && vpDet3 < 0.0f) {
            ONCE(Logger::info("[RTX-Compatibility-Info] Ignored UE3 scene capture draw (mirrored view-projection, e.g. reflection probe)."));
            classifySceneCaptureView("scene capture mirrored view");
            return false;
          }
        }
        ONCE(Logger::info(str::format("[RTX-Compatibility] UE3 camera matrices extracted from shader constants (viewProjReg=c",
                                      viewProjReg, "..c", viewProjReg + 3, ", viewOriginReg=c", viewOriginReg, ").")));
        if (m_frameOptions.ue3LogCapturePrecision && Logger::logLevel() <= LogLevel::Debug) {
          ONCE(Logger::debug(str::format(
            "[RTX-Compatibility][UE3-Capture] camera matrix reconstruction error=",
            ue3CameraReconstructionError, ", usedTranspose=", ue3CameraUsedTranspose)));
        }
        transformData.worldToView = ue3WorldToView;
        transformData.viewToProjection = ue3ViewToProjection;

        if ((m_frameOptions.ue3RequireCtabCameraConstants || isUe3Mode) && !ctabVerifiedCamera) {
          m_activeDrawCallState.allowMainCameraUpdate = false;
        }

        // Non-main-view-sized world draws (below half and/or wrong aspect) must not steer Main.
        if (isUe3Mode &&
            isUe3WorldGeometryVertexFactory(m_currentUe3VertexFactory) &&
            m_activePresentParams.has_value() &&
            m_activeDrawCallState.allowMainCameraUpdate) {
          const D3DVIEWPORT9& vp = d3d9State().viewport;
          if (!ue3ViewportIsMainViewSized(
                vp.Width, vp.Height,
                m_activePresentParams->BackBufferWidth,
                m_activePresentParams->BackBufferHeight)) {
            m_activeDrawCallState.allowMainCameraUpdate = false;
          }
        }

        // Once per vertex shader, so info level stays low-volume
        {
          static fast_unordered_set s_loggedCameraSourceVsHashes;
          const XXH64_hash_t vsHash = m_activeDrawCallState.programmableVertexShaderBytecodeHash;
          if (vsHash != 0 && s_loggedCameraSourceVsHashes.insert(vsHash).second) {
            Logger::info(str::format(
              "[RTX-Compatibility][UE3] camera constants source for vsHash=0x", std::hex, vsHash, std::dec,
              ": ", ctabVerifiedCamera ? "CTAB-verified" : "fallback registers",
              " (ctabViewProj=", (ue3CtabInfoPtr != nullptr && ue3CtabInfoPtr->hasViewProjectionMatrix) ? "yes" : "no",
              ", ctabCameraPos=", (ue3CtabInfoPtr != nullptr && ue3CtabInfoPtr->hasCameraPosition) ? "yes" : "no",
              ", viewProjReg=c", viewProjReg, ", viewOriginReg=c", viewOriginReg,
              ", vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
              ", allowMainCameraUpdate=", m_activeDrawCallState.allowMainCameraUpdate ? "true" : "false", ")"));
          }
        }
      } else {
        // A CTAB-verified world-geometry draw whose declared ViewProjectionMatrix fails
        // plausibility extraction is rendering through a view Remix cannot use. In practice
        // these are SceneCapture reflect/portal probes: FClipProjectionMatrix skews the near
        // plane onto the mirror/portal plane, which the extraction rejects as shear, and
        // FMirrorMatrix reflects the view. The main view always extracts, so nothing
        // legitimate is lost - and geometry processed with the stale/identity transforms
        // this branch would otherwise fall back to reconstructs as corrupted positions.
        if (ue3CaptureViewIsolation) {
          ONCE(Logger::info("[RTX-Compatibility-Info] Ignored UE3 scene capture draw (declared camera failed extraction, e.g. reflection/portal probe oblique projection)."));
          classifySceneCaptureView("scene capture undecomposable view");
          return false;
        }

        if (Logger::logLevel() <= LogLevel::Debug &&
            viewProjReg + 3 < caps::MaxFloatConstantsSoftware && viewOriginReg < caps::MaxFloatConstantsSoftware) {
          const Vector4 c0 = d3d9State().vsConsts.fConsts[viewProjReg + 0];
          const Vector4 c1 = d3d9State().vsConsts.fConsts[viewProjReg + 1];
          const Vector4 c2 = d3d9State().vsConsts.fConsts[viewProjReg + 2];
          const Vector4 c3 = d3d9State().vsConsts.fConsts[viewProjReg + 3];
          const Vector4 cam = d3d9State().vsConsts.fConsts[viewOriginReg];

          ONCE(Logger::debug(str::format(
            "[RTX-Compatibility] UE3 camera extraction failed (viewProjReg=c", viewProjReg, "..c", viewProjReg + 3,
            ", viewOriginReg=c", viewOriginReg, "). "
            "c0={", c0.x, ", ", c0.y, ", ", c0.z, ", ", c0.w, "} "
            "c1={", c1.x, ", ", c1.y, ", ", c1.z, ", ", c1.w, "} "
            "c2={", c2.x, ", ", c2.y, ", ", c2.z, ", ", c2.w, "} "
            "c3={", c3.x, ", ", c3.y, ", ", c3.z, ", ", c3.w, "} "
            "cam={", cam.x, ", ", cam.y, ", ", cam.z, ", ", cam.w, "}")));
        } else {
          ONCE(Logger::debug(str::format(
            "[RTX-Compatibility] UE3 camera extraction failed (out of bounds registers: viewProjReg=c", viewProjReg,
            ", viewOriginReg=c", viewOriginReg, ").")));
        }
      }
    }

    if (usesProgrammableVs && effectiveUe3ObjectToWorld && ue3CtabInfoPtr != nullptr) {
      const Ue3VsShaderCtabInfo& ctabInfo = *ue3CtabInfoPtr;

      if (ctabInfo.hasLocalToWorld) {
        const uint32_t reg = ctabInfo.localToWorldRegisterIndex;
        if (reg + 3 < caps::MaxFloatConstantsSoftware) {
          const uint32_t w2lReg = ctabInfo.worldToLocalRegisterIndex;
          const bool hasWorldToLocal = ctabInfo.hasWorldToLocal && w2lReg + 2 < caps::MaxFloatConstantsSoftware;

          // Memo lookup: every input to the transpose/affinity/inverse disambiguation
          // below (register contents and the camera transpose convention tiebreaker) is
          // folded into the key, so a hit returns exactly what the computation would
          // produce. Static placements re-upload identical matrices every frame, making
          // this a per-draw matrix-inverse saving.
          XXH64_hash_t o2wKeyHash = XXH3_64bits(&d3d9State().vsConsts.fConsts[reg], 4 * sizeof(Vector4));
          if (hasWorldToLocal) {
            o2wKeyHash = XXH3_64bits_withSeed(&d3d9State().vsConsts.fConsts[w2lReg], 3 * sizeof(Vector4), o2wKeyHash);
          }
          const uint32_t o2wKeyFlags = (hasWorldToLocal ? 1u : 0u) | (ue3CameraUsedTranspose ? 2u : 0u);
          o2wKeyHash = XXH3_64bits_withSeed(&o2wKeyFlags, sizeof(o2wKeyFlags), o2wKeyHash);

          const auto o2wIt = m_ue3ObjectToWorldCache.find(o2wKeyHash);
          if (o2wIt != m_ue3ObjectToWorldCache.end()) {
            transformData.objectToWorld = o2wIt->second;
          } else {
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

            // optinally use WorldToLocal (if present) to disambiguate transpose/packing
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
                localToWorld = ue3CameraUsedTranspose ? localToWorldTransposed : localToWorldRaw;
              }
            } else if (rawAffine && transAffine) {
              localToWorld = ue3CameraUsedTranspose ? localToWorldTransposed : localToWorldRaw;
            } else if (!rawAffine && transAffine) {
              localToWorld = localToWorldTransposed;
            } else {
              localToWorld = localToWorldRaw;
            }

            if (m_ue3ObjectToWorldCache.size() >= kUe3ObjectToWorldCacheMaxEntries) {
              m_ue3ObjectToWorldCache.clear();
            }
            m_ue3ObjectToWorldCache.emplace(o2wKeyHash, localToWorld);

            transformData.objectToWorld = localToWorld;
          }

          ONCE(Logger::info("[RTX-Compatibility] UE3 LocalToWorld extracted from vertex shader constants (CTAB)"));
        }
      }
    }

    transformData.objectToView = transformData.worldToView * transformData.objectToWorld;

    // Some games pass invalid matrices which D3D9 apparently doesnt care about.
    // since we'll be doing inversions and other matrix operations, we need to 
    // sanitize those or there be nans.
    transformData.sanitize();

    if (m_flags.test(D3D9RtxFlag::DirtyClipPlanes)) {
      m_flags.clr(D3D9RtxFlag::DirtyClipPlanes);

      // Find one truly enabled clip plane because we don't support more than one
      transformData.enableClipPlane = false;
      if (d3d9State().renderStates[D3DRS_CLIPPLANEENABLE] != 0) {
        for (int i = 0; i < caps::MaxClipPlanes; ++i) {
          // Check the enable bit
          if ((d3d9State().renderStates[D3DRS_CLIPPLANEENABLE] & (1 << i)) == 0)
            continue;

          // Make sure that the plane equation is not degenerate
          const Vector4 plane = Vector4(d3d9State().clipPlanes[i].coeff);
          if (lengthSqr(plane.xyz()) > 0.f) {
            if (transformData.enableClipPlane) {
              ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Using more than 1 user clip plane is not supported.")));
              break;
            }

            transformData.enableClipPlane = true;
            transformData.clipPlane = plane;
          }
        }
      }
    }

    if (m_flags.test(D3D9RtxFlag::DirtyLights)) {
      m_flags.clr(D3D9RtxFlag::DirtyLights);

      std::vector<D3DLIGHT9> activeLightsRT;
      uint32_t lightIdx = 0;
      for (auto idx : d3d9State().enabledLightIndices) {
        if (idx == UINT32_MAX)
          continue;
        activeLightsRT.push_back(d3d9State().lights[idx].value());
      }

      m_parent->EmitCs([activeLightsRT, lightIdx](DxvkContext* ctx) {
          static_cast<RtxContext*>(ctx)->addLights(activeLightsRT.data(), activeLightsRT.size());
        });
    }

    // Stencil state is important to Remix
    m_activeDrawCallState.stencilEnabled = d3d9State().renderStates[D3DRS_STENCILENABLE];

    // translucency - UE3 draws two sided translucent meshes as backface then frontface
    // passes with the same shader/textures but inverted cull modes; detect and skip the
    // second pass to avoid duplicate geometry. The match must be strict: UE3 sorts
    // translucent prims back-to-front per camera, so consecutive draws of *different*
    // meshes or instances sharing one material are common, and a loose match falsely
    // skips them (visibility flicker that follows the camera)
    if (m_frameOptions.ue3EngineMode && usesProgrammableVs && d3d9State().vertexShader.ptr() != nullptr &&
        d3d9State().pixelShader.ptr() != nullptr) {
      const XXH64_hash_t vsHash = d3d9State().vertexShader->GetCommonShader()->GetBytecodeHash();
      const XXH64_hash_t psHash = d3d9State().pixelShader->GetCommonShader()->GetBytecodeHash();
      const XXH64_hash_t vsPsHash = vsHash ^ (psHash * 0x9E3779B97F4A7C15ull);

      XXH64_hash_t boundTextureHash = 0;
      {
        const BoundTextureSnapshot& boundTextures = ensureBoundTextureSnapshot();
        for (const uint32_t s : bit::BitMask(boundTextures.mask & 0xFu)) {
          if (boundTextures.entries[s].hasImage)
            boundTextureHash ^= boundTextures.entries[s].imageHash;
        }
      }

      // identity of the exact geometry this draw references (buffers, ranges, topology)
      struct DrawGeometryIdentity {
        const void* pIndexBuffer;
        const void* pVertexBuffer0;
        const void* pVertexDecl;
        uint32_t vb0Offset;
        uint32_t vb0Stride;
        int32_t baseVertexIndex;
        uint32_t minVertexIndex;
        uint32_t numVertices;
        uint32_t startIndex;
        uint32_t primitiveCount;
        uint32_t primitiveType;
        uint32_t indexed;
        uint32_t reserved;
      };
      static_assert(sizeof(DrawGeometryIdentity) == 3 * sizeof(void*) + 10 * sizeof(uint32_t),
                    "DrawGeometryIdentity must have no implicit padding (it is hashed by memory).");
      const DrawGeometryIdentity geometryIdentity = {
        d3d9State().indices.ptr(),
        d3d9State().vertexBuffers[0].vertexBuffer.ptr(),
        d3d9State().vertexDecl.ptr(),
        d3d9State().vertexBuffers[0].offset,
        d3d9State().vertexBuffers[0].stride,
        drawContext.BaseVertexIndex,
        drawContext.MinVertexIndex,
        drawContext.NumVertices,
        drawContext.StartIndex,
        drawContext.PrimitiveCount,
        uint32_t(drawContext.PrimitiveType),
        uint32_t(drawContext.Indexed),
        0u,
      };
      XXH64_hash_t geometryIdentityHash = XXH3_64bits(&geometryIdentity, sizeof(geometryIdentity));

      // A genuine second cull pass re-issues the same instance with identical transform
      // constants; different placements of a shared mesh never match once those are folded
      // in. Only computed for blended draws - the skip condition requires blending anyway
      if (d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] != FALSE) {
        geometryIdentityHash = mixUe3InstanceTransformConstants(geometryIdentityHash);
      }

      const DWORD cullMode = d3d9State().renderStates[D3DRS_CULLMODE];
      const bool cullInverted =
        (cullMode == D3DCULL_CW && m_prevDrawCullMode == D3DCULL_CCW) ||
        (cullMode == D3DCULL_CCW && m_prevDrawCullMode == D3DCULL_CW);
      const bool isSecondTwoSidedPass =
        vsPsHash != 0 &&
        vsPsHash == m_prevDrawVsPsHash &&
        boundTextureHash == m_prevDrawTextureHash &&
        geometryIdentityHash == m_prevDrawGeometryHash &&
        cullInverted &&
        d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] != FALSE;

      m_prevDrawVsPsHash = vsPsHash;
      m_prevDrawTextureHash = boundTextureHash;
      m_prevDrawGeometryHash = geometryIdentityHash;
      m_prevDrawCullMode = cullMode;

      if (isSecondTwoSidedPass) {
        m_ue3LastDrawDecision = "two-pass translucency dedup";
        ONCE(Logger::info("[RTX-Compatibility-Info] Skipped UE3 two-pass translucent draw (second cull-mode pass)."));
        return false;
      }
    } else {
      m_prevDrawVsPsHash = 0;
      m_prevDrawTextureHash = 0;
      m_prevDrawGeometryHash = 0;
      m_prevDrawCullMode = 0;
    }

    // Process textures
    if (m_parent->UseProgrammablePS()) {
      return processTextures<false>();
    } else {
      return processTextures<true>();
    }
  }

  D3D9Rtx::DrawCallType D3D9Rtx::makeDrawCallType(const DrawContext& drawContext) {
    ScopedCpuProfileZone();
    // Track the drawcall index so we can use it in rtx_context
    m_activeDrawCallState.drawCallID = m_drawCallID++;
    m_activeDrawCallState.isDrawingToRaytracedRenderTarget = false;
    m_activeDrawCallState.isUsingRaytracedRenderTarget = false;

    if (m_drawCallID < (uint32_t)m_frameOptions.drawCallRange.x ||
        m_drawCallID > (uint32_t)m_frameOptions.drawCallRange.y) {
      return { RtxGeometryStatus::Ignored, false };
    }

    // Draws inside an occlusion query bracket are visibility-test geometry, never scene geometry
    // to ray trace; checked first so no skip path below can starve an active query of its draws.
    // With synthesized readbacks nothing consumes a measurement, so they are ignored entirely;
    // otherwise they rasterize so the query can count their samples.
    if (m_activeOcclusionQueries > 0) {
      if (ConservativeOcclusionQueriesEnabled()) {
        m_ue3LastDrawDecision = "occlusion query test draw (ignored, result synthesized)";
        ONCE(Logger::info("[RTX-Compatibility-Info] Ignoring occlusion query test draw (conservative occlusion queries synthesize the result)."));
        return { RtxGeometryStatus::Ignored, false };
      }
      m_ue3LastDrawDecision = "occlusion query test draw (rasterized)";
      ONCE(Logger::info("[RTX-Compatibility-Info] Rasterizing occlusion query test draw without ray tracing."));
      return { RtxGeometryStatus::Rasterized, false };
    }

    // Raytraced Render Target Support
    // If the bound texture for this draw call is one that has been used as a render target then store its id
    if (m_frameOptions.raytracedRenderTargetEnable) {
      for (uint32_t i : bit::BitMask(m_parent->GetActiveRTTextures())) {
        D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[i]);
        if (!texture || texture->GetImage() == nullptr)
          continue;

        if (isTaggedRaytracedRenderTarget(texture->GetImage(), true)) {
          m_activeDrawCallState.isUsingRaytracedRenderTarget = true;
        }
      }
    }

    if (m_parent->UseProgrammableVS() && !m_frameOptions.useVertexCapture) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipping draw call with shader usage as vertex capture is not enabled."));
      return { RtxGeometryStatus::Ignored, false };
    }

    if (drawContext.PrimitiveCount == 0) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped invalid drawcall, primitive count was 0."));
      return { RtxGeometryStatus::Ignored, false };
    }

    // Only certain draw calls are worth raytracing
    if (!isPrimitiveSupported(drawContext.PrimitiveType)) {
      ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Trying to raytrace an unsupported primitive topology [", drawContext.PrimitiveType, "]. Ignoring.")));
      return { RtxGeometryStatus::Ignored, false };
    }

    if (!m_frameOptions.enableAlphaTest && m_parent->IsAlphaTestEnabled()) {
      ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Raytracing an alpha-tested draw call when alpha-tested objects disabled in RT. Ignoring.")));
      return { RtxGeometryStatus::Ignored, false };
    }

    if (!m_frameOptions.enableAlphaBlend && d3d9State().renderStates[D3DRS_ALPHABLENDENABLE]) {
      ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Raytracing an alpha-blended draw call when alpha-blended objects disabled in RT. Ignoring.")));
      return { RtxGeometryStatus::Ignored, false };
    }

    // Deferred-UI tag decision, evaluated at most once per draw: consumed by both the
    // depth-test-disabled translucency skip directly below and the deferred-overlay
    // branch further down.
    XXH64_hash_t deferredUiMatchedTextureHash = 0;
    int deferredUiTagState = -1;
    auto isDeferredUiTagged = [&]() {
      if (deferredUiTagState < 0) {
        deferredUiTagState = isDeferredUiTaggedDraw(&deferredUiMatchedTextureHash) ? 1 : 0;
      }
      return deferredUiTagState == 1;
    };
    // A matched hash of zero means the pixel shader tag matched, which the option documents as
    // explicit intent - unlike a texture tag, which a shared texture can trigger on any draw.
    // The emptiness test keeps post-process draws from building a bound-texture snapshot just
    // to discover that no pixel shader tag exists to find.
    auto isDeferredUiPixelShaderTagged = [&]() {
      return !m_frameOptions.deferredUiPixelShaders->empty() &&
             isDeferredUiTagged() &&
             deferredUiMatchedTextureHash == kEmptyHash;
    };

    // UE3 depth test disabled translucency -  NeedsDepthTestDisabled materials, fog volume composites,
    // and fullscreen overlays use alpha blend + depth test off + depth write off
    // exclude UI tagged draws since they also match this pattern but need rasterisation with RTX injection,
    // and deferred-UI tagged draws which need capture for post-injection replay
    if ((m_frameOptions.ue3SkipDepthTestDisabledTranslucency || m_frameOptions.ue3EngineMode) &&
        d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] &&
        (d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_FALSE ||
         d3d9State().renderStates[D3DRS_ZFUNC] == D3DCMP_ALWAYS) &&
        d3d9State().renderStates[D3DRS_ZWRITEENABLE] == FALSE &&
        !checkBoundTextureCategory(*m_frameOptions.uiTextures) &&
        !isDeferredUiTagged()) {
      m_ue3LastDrawDecision = "depth-test-disabled translucency skip";
      ONCE(Logger::info("[RTX-Compatibility-Info] Ignored UE3 depth-test-disabled translucent draw."));
      return { RtxGeometryStatus::Ignored, false };
    }

    if (d3d9State().renderTargets[kRenderTargetIndex] == nullptr) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped drawcall, as no color render target bound."));
      return { RtxGeometryStatus::Ignored, false };
    }

    {
      D3D9CommonTexture* rtTexture = GetCommonTexture(d3d9State().renderTargets[kRenderTargetIndex]->GetBaseTexture());
      if (rtTexture != nullptr && rtTexture->GetImage() == nullptr) {
        ONCE(Logger::info("[RTX-Compatibility-Info] Skipped drawcall, render target has no GPU image (possibly a depth-only pass)."));
        return { RtxGeometryStatus::Ignored, false };
      }
    }

    constexpr DWORD rgbWriteMask = D3DCOLORWRITEENABLE_RED | D3DCOLORWRITEENABLE_GREEN | D3DCOLORWRITEENABLE_BLUE;
    if ((d3d9State().renderStates[ColorWriteIndex(kRenderTargetIndex)] & rgbWriteMask) != rgbWriteMask) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped drawcall, colour write disabled."));
      return { RtxGeometryStatus::Ignored, false };
    }

    // UE3 depth prepass - position only vertex declarations have no texcoords/colours
    // the same geometry will be drawn again in the base pass with full material
    if ((m_frameOptions.ue3SkipDepthPrepass || m_frameOptions.ue3EngineMode) &&
        m_currentUe3VertexFactory == Ue3VertexFactoryType::PositionOnly) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped UE3 depth prepass draw (position-only vertex declaration)."));
      return { RtxGeometryStatus::Ignored, false };
    }

    // Ensure present parameters for the swapchain have been cached
    // Note: This assumes that ResetSwapChain has been called at some point before this call, typically done after creating a swapchain.
    assert(m_activePresentParams.has_value());

    m_currentUe3PassType = classifyUe3Pass(drawContext);

    // Capture TdToneMapping state before the draw is ignored below.
    if (m_currentUe3PassType == Ue3PassType::FullscreenPostProcess) {
      maybeCaptureUe3ToneMapState();
    }

    switch (m_currentUe3PassType) {
    case Ue3PassType::DepthPrepass:
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "position-only depth prepass");
      return { RtxGeometryStatus::Ignored, false };
    case Ue3PassType::ShadowDepth:
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "shadow depth render target");
      return { RtxGeometryStatus::Ignored, false };
    case Ue3PassType::Velocity:
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "native velocity helper pass");
      return { RtxGeometryStatus::Ignored, false };
    case Ue3PassType::Lighting:
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "native UE3 lighting pass");
      return { RtxGeometryStatus::Ignored, false };
    case Ue3PassType::ModulatedShadowProjection:
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "native modulated shadow projection");
      return { RtxGeometryStatus::Ignored, false };
    case Ue3PassType::SceneCapture:
      if (!m_ue3ForegroundDpgActive) {
        m_ue3SeenMainViewWorldDraw = false;
      }
      ONCE(Logger::info("[RTX-Compatibility-Info] Ignored UE3 scene capture offscreen view draw (world geometry, probe viewport)."));
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "scene capture offscreen view");
      return { RtxGeometryStatus::Ignored, false };
    case Ue3PassType::FullscreenPostProcess:
    case Ue3PassType::FogOrDistortion:
      // Ignore before the non-primary RT → Rasterized fallback; DoF gather/blur/blend
      // target FilterColor/SceneColor and would otherwise still execute. Only a deferred-UI
      // pixel shader tag outranks this: these passes sample the scene-colour target, so a
      // texture tag on it matches every one of them and would replay DoF over the frame.
      if (!isDeferredUiPixelShaderTagged()) {
        logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "native UE3 screen-space contribution pass");
        return { RtxGeometryStatus::Ignored, false };
      }
      break;
    case Ue3PassType::UiComposite:
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Rasterized, "UI composite");
      return { RtxGeometryStatus::Rasterized, true };
    case Ue3PassType::VideoCinematic:
      trackUe3MovieTextureRenderTarget("video cinematic/decode");
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Rasterized, "video/cinematic pass");
      return { RtxGeometryStatus::Rasterized, false };
    case Ue3PassType::VideoSurface:
      trackUe3MovieTextureRenderTarget("video surface/decode");
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Rasterized, "video texture surface/decode pass");
      return { RtxGeometryStatus::Rasterized, false };
    default:
      break;
    }

    // Arm the foreground-DPG gate only after a true main-view-sized world draw.
    if (m_frameOptions.ue3EngineMode &&
        !m_ue3SeenMainViewWorldDraw &&
        m_currentUe3PassType == Ue3PassType::Material &&
        isUe3WorldGeometryVertexFactory(m_currentUe3VertexFactory) &&
        m_activePresentParams.has_value()) {
      const D3DVIEWPORT9& vp = d3d9State().viewport;
      const uint32_t bbW = m_activePresentParams->BackBufferWidth;
      const uint32_t bbH = m_activePresentParams->BackBufferHeight;
      if (ue3ViewportIsMainViewSized(vp.Width, vp.Height, bbW, bbH)) {
        m_ue3SeenMainViewWorldDraw = true;
      }
    }

    // Deferred UI overlays (rtx.deferredUiTextures / rtx.d3d9.deferredUiPixelShaders, e.g. UE3
    // MaterialEffect fullscreen fades): rasterized on top of the ray-traced image WITHOUT
    // triggering RTX injection - the draw is captured and replayed after injection fires later
    // in the frame. Placed after the pass switch above, so UI composites, video passes and the
    // screen-space contribution passes can never be deferred by a texture tag (only a pixel
    // shader tag reaches here from those), and before the fullscreen-composite filter below so
    // tagging wins over that. World geometry and depth-writing draws are never deferred even
    // when tagged: shared textures (e.g. a scene-color render target sampled by translucent
    // meshes) must not pull geometry out of the ray-traced scene.
    if (!m_frameOptions.deferredUiTextures->empty() || !m_frameOptions.deferredUiPixelShaders->empty()) {
      const XXH64_hash_t& matchedTextureHash = deferredUiMatchedTextureHash;

      if (isDeferredUiTagged()) {
        const bool matchedByPixelShaderTag = isDeferredUiPixelShaderTagged();
        const bool zWriteEnabled = d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE;
        const bool isWorldGeometryVertexFactory = isUe3WorldGeometryVertexFactory(m_currentUe3VertexFactory);

        // Engine post-process/composite shaders (gamma-correction scene copy, tone mapping,
        // motion blur, distortion, fog, depth of field, bloom) legitimately sample the scene
        // render target but must never be deferred: replaying e.g. the gamma copy over the
        // ray-traced image uniformly brightens the whole screen and, being opaque and later in
        // the frame, overwrites the real overlay effects; replaying a DoF gather paints a flat
        // rectangle over it. This is the only guard for a texture tag once the draw classifies
        // as anything but a screen-space contribution pass, so it matches the same DoF/bloom
        // signatures the pass classifier uses. An explicit pixel shader tag still defers.
        bool isEnginePostProcessShader = false;
        if (!matchedByPixelShaderTag && m_parent->UseProgrammablePS() && d3d9State().pixelShader != nullptr) {
          const Ue3ShaderFeatureInfo psInfo = getUe3ShaderFeatureInfo(d3d9State().pixelShader->GetCommonShader());
          isEnginePostProcessShader = psInfo.hasGammaConstants ||
                                      psInfo.hasToneMapConstants ||
                                      psInfo.hasExposureOrToneSampler ||
                                      psInfo.hasMotionBlurConstants ||
                                      psInfo.hasVelocitySampler ||
                                      psInfo.hasDistortionSampler ||
                                      psInfo.hasFogConstants ||
                                      psInfo.hasHazeConstants ||
                                      psInfo.looksLikeDofAndBloomPostProcess();
        }

        // Fullscreen overlay tiles (UE3 MaterialEffect quads via FTileRenderer) use a
        // Local-style vertex declaration (position/tangents/color/uv) and would be caught by
        // the world-geometry guard. A tiny primitive count with depth testing disabled
        // distinguishes them from real world geometry: even small world quads (glass panes,
        // monitors) depth-test against the scene, overlay tiles never do.
        const bool depthTestDisabled = d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_FALSE ||
                                       d3d9State().renderStates[D3DRS_ZFUNC] == D3DCMP_ALWAYS;
        const bool looksLikeOverlayTile = drawContext.PrimitiveCount <= 4 && depthTestDisabled && !zWriteEnabled;

        const bool eligible = !zWriteEnabled && !isEnginePostProcessShader &&
                              (!isWorldGeometryVertexFactory || looksLikeOverlayTile);
        const char* refusalReason = isEnginePostProcessShader
                                    ? "engine post-process shader"
                                    : "world geometry or depth write";

        // One-shot diagnostics per (pixel shader, decision): prints the stable pixel shader
        // hash so tags on unstable render-target textures can be moved to
        // rtx.d3d9.deferredUiPixelShaders.
        const XXH64_hash_t psHash = (m_parent->UseProgrammablePS() && d3d9State().pixelShader != nullptr)
                                    ? d3d9State().pixelShader->GetCommonShader()->GetBytecodeHash() : 0;
        const XXH64_hash_t vsHash = (m_parent->UseProgrammableVS() && d3d9State().vertexShader != nullptr)
                                    ? d3d9State().vertexShader->GetCommonShader()->GetBytecodeHash() : 0;
        const XXH64_hash_t logKey = psHash ^ (eligible ? 0xD1B54A32D192ED03ull
                                                       : (isEnginePostProcessShader ? 0x2545F4914F6CDD1Dull
                                                                                    : 0x9E3779B97F4A7C15ull));
        if (m_deferredUiLoggedDecisions.insert(logKey).second) {
          const std::string matchedDescription = matchedTextureHash != 0
            ? str::format(" matchedTexture=0x", std::hex, matchedTextureHash, std::dec)
            : std::string(" matchedBy=pixelShaderTag");

          Logger::info(str::format(
            "[RTX-DeferredUI] ",
            eligible ? std::string("Deferring overlay draw")
                     : str::format("Tagged draw NOT deferred (", refusalReason, ")"),
            ": ps=0x", std::hex, psHash,
            " vs=0x", vsHash, std::dec,
            " vertexFactory=", describeUe3VertexFactory(m_currentUe3VertexFactory),
            " pass=", describeUe3PassType(m_currentUe3PassType),
            " prims=", drawContext.PrimitiveCount,
            " ztest=", depthTestDisabled ? 0 : 1,
            " zwrite=", zWriteEnabled ? 1 : 0,
            matchedDescription));
        }

        if (eligible) {
          m_ue3LastDrawDecision = "deferred UI overlay";
          logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Rasterized, "deferred UI overlay");
          return { RtxGeometryStatus::Rasterized, false, true };
        }

        // Screen-space contribution passes only reached this branch through a pixel shader tag;
        // a refusal here restores the pass switch's decision rather than promoting a DoF/fog
        // draw to normal classification.
        if (m_currentUe3PassType == Ue3PassType::FullscreenPostProcess ||
            m_currentUe3PassType == Ue3PassType::FogOrDistortion) {
          logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "native UE3 screen-space contribution pass");
          return { RtxGeometryStatus::Ignored, false };
        }
        // Other ineligible tagged draws fall through to normal classification - never suppressed.
      }
    }

    // Attempt to detect shadow mask draws and ignore them
    // Conditions: non-textured flood-fill draws into a small quad render target
    if (((d3d9State().textureStages[0][D3DTSS_COLOROP] == D3DTOP_SELECTARG1 && d3d9State().textureStages[0][D3DTSS_COLORARG1] != D3DTA_TEXTURE) ||
         (d3d9State().textureStages[0][D3DTSS_COLOROP] == D3DTOP_SELECTARG2 && d3d9State().textureStages[0][D3DTSS_COLORARG2] != D3DTA_TEXTURE))) {
      const auto& rtExt = d3d9State().renderTargets[kRenderTargetIndex]->GetSurfaceExtent();
      // If rt is a quad at least 4 times smaller than backbuffer and the format is invalid format, then it is likely a shadow mask
      if (rtExt.width == rtExt.height && rtExt.width < m_activePresentParams->BackBufferWidth / 4 &&
          Resources::getFormatCompatibilityCategory(d3d9State().renderTargets[kRenderTargetIndex]->GetImageView(false)->imageInfo().format) == RtxTextureFormatCompatibilityCategory::InvalidFormatCompatibilityCategory) {
        ONCE(Logger::info("[RTX-Compatibility-Info] Skipped shadow mask drawcall."));
        return { RtxGeometryStatus::Ignored, false };
      }
    }

    // UE3 shadow depth pass - draws to small square render targets that are used as shadow maps
    if ((m_frameOptions.ue3SkipShadowDepthPasses || m_frameOptions.ue3EngineMode) && m_activePresentParams.has_value()) {
      const auto& rtExt = d3d9State().renderTargets[kRenderTargetIndex]->GetSurfaceExtent();
      const uint32_t bbW = m_activePresentParams->BackBufferWidth;
      const bool isSmallSquare = rtExt.width == rtExt.height &&
                                 rtExt.width <= 2048 &&
                                 rtExt.width < bbW / 2;
      const bool hasDepthWrite = d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE;
      if (isSmallSquare && hasDepthWrite) {
        ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Skipped UE3 shadow depth pass (",
                                       rtExt.width, "x", rtExt.height, ").")));
        return { RtxGeometryStatus::Ignored, false };
      }
    }

    // Raytraced Render Target
    // If this isn't the primary render target but we have used this render target before then 
    // store the current camera matrices in case this render target is intended to be used as 
    // a texture for some geometry later
    if (m_frameOptions.raytracedRenderTargetEnable) {
      D3D9CommonTexture* texture = GetCommonTexture(d3d9State().renderTargets[kRenderTargetIndex]->GetBaseTexture());
      if (texture) {
        const Rc<DxvkImage> image = texture->GetImage();
        if (image != nullptr && isTaggedRaytracedRenderTarget(image, true)) {
          m_activeDrawCallState.isDrawingToRaytracedRenderTarget = true;
          return { RtxGeometryStatus::RayTraced, false };
        }
      }
    }

    if (!s_isDxvkResolutionEnvVarSet) {
      // NOTE: This can fail when setting DXVK_RESOLUTION_WIDTH or HEIGHT
      const bool isPrimary = isRenderTargetPrimary(*m_activePresentParams, d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture()->Desc());

      if (!isPrimary) {
        // debugging, todo remove later
        if (Logger::logLevel() <= LogLevel::Debug) {
          if (D3D9CommonTexture* rtTex = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture()) {
            if (rtTex->GetImage() != nullptr) {
              const XXH64_hash_t rtDescHash = rtTex->GetImage()->getDescriptorHash();
              if (s_loggedNonPrimaryRtDescHashes.insert(rtDescHash).second) {
                const auto* rtDesc = rtTex->Desc();
                Logger::debug(str::format(
                  "[RTX-Compatibility] Non-primary RT0 encountered: ",
                  rtDesc->Width, "x", rtDesc->Height,
                  " (backbuffer ", m_activePresentParams->BackBufferWidth, "x", m_activePresentParams->BackBufferHeight, "), ",
                  "rtDescHash=0x", std::hex, rtDescHash,
                  " resolutionAgnosticDescHash=0x", rtTex->GetImage()->getResolutionAgnosticDescriptorHash(), std::dec,
                  ". If this RT contains the main scene, add either hash to rtx.raytracedRenderTargetTextures "
                  "(the resolution-agnostic one survives resolution changes)."));
              }
            }
          }
        }

        ONCE(Logger::info("[RTX-Compatibility-Info] Found a draw call to a non-primary, non-raytraced render target. Falling back to rasterization"));
        return { RtxGeometryStatus::Rasterized, false };
      }
    }

    if (const uint32_t rtSamplerMask = m_parent->GetActiveRTTextures()) {
      const bool depthEnabled  = d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_TRUE;
      const bool zWriteEnabled = d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE;
      const bool likelyFullscreenComposite =
        !depthEnabled &&
        !zWriteEnabled &&
        drawContext.PrimitiveCount <= 4;

      if (m_frameOptions.autoRaytracedRenderTargetFromFullscreenComposite && likelyFullscreenComposite) {
        const uint32_t bbW = m_activePresentParams->BackBufferWidth;
        const uint32_t bbH = m_activePresentParams->BackBufferHeight;

        auto aspectRatioMatches = [&](uint32_t w, uint32_t h) {
          const double a = double(w) * double(bbH);
          const double b = double(h) * double(bbW);
          const double denom = std::max(a, b);
          return denom > 0.0 && (std::abs(a - b) / denom) < 0.01;
        };

        XXH64_hash_t bestHash = 0;
        XXH64_hash_t bestAgnosticHash = 0;
        uint64_t bestArea = 0;

        for (uint32_t i : bit::BitMask(rtSamplerMask)) {
          D3D9CommonTexture* tex = GetCommonTexture(d3d9State().textures[i]);
          if (!tex || tex->GetImage() == nullptr)
            continue;

          if (isTaggedRaytracedRenderTarget(tex->GetImage(), true))
            continue;

          const auto* desc = tex->Desc();
          if (!desc)
            continue;
          if (desc->Width == bbW && desc->Height == bbH)
            continue;
          if (!aspectRatioMatches(desc->Width, desc->Height))
            continue;

          const uint64_t area = uint64_t(desc->Width) * uint64_t(desc->Height);
          if (area > bestArea) {
            bestArea = area;
            // Absolute hash: this set is session-local, and the aspect-normalized hash would
            // alias the backbuffer-sized target skipped above.
            bestHash = tex->GetImage()->getDescriptorHash();
            bestAgnosticHash = tex->GetImage()->getResolutionAgnosticDescriptorHash();
          }
        }

        // Already-tagged targets were skipped as candidates above, so anything reaching here is new.
        if (bestHash != kEmptyHash && m_autoRaytracedRenderTargetDescHashes.insert(bestHash).second) {
          Logger::info(str::format(
            "[RTX-Compatibility] Auto-selected Raytraced Render Target from fullscreen composite: texDescHash=0x",
            std::hex, bestHash,
            " resolutionAgnosticDescHash=0x", bestAgnosticHash, std::dec,
            " (tag either in rtx.raytracedRenderTargetTextures to pin it)."));
        }
      }

      if (Logger::logLevel() <= LogLevel::Debug) {
        for (uint32_t i : bit::BitMask(rtSamplerMask)) {
          D3D9CommonTexture* tex = GetCommonTexture(d3d9State().textures[i]);
          if (!tex || tex->GetImage() == nullptr)
            continue;

          const XXH64_hash_t texDescHash = tex->GetImage()->getDescriptorHash();
          if (s_loggedSampledRtDescHashes.insert(texDescHash).second) {
            const auto* desc = tex->Desc();
            Logger::debug(str::format(
              "[RTX-Compatibility] Sampled render-target texture: ",
              desc->Width, "x", desc->Height,
              ", texDescHash=0x", std::hex, texDescHash,
              " resolutionAgnosticDescHash=0x", tex->GetImage()->getResolutionAgnosticDescriptorHash(), std::dec,
              " (sampler ", i, ")."));
          }
        }
      }

      // Optional: do not raytrace likely fullscreen composite passes to primary.
      if (m_frameOptions.rasterizeFullscreenCompositeToPrimary && likelyFullscreenComposite) {
        logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Rasterized, "fullscreen RT composite");
        ONCE(Logger::info("[RTX-Compatibility] Rasterizing likely fullscreen composite pass to primary RT (post-process)."));
        return { RtxGeometryStatus::Rasterized, false };
      }
    }

    // Detect stencil shadow draws and ignore them
    // Conditions: passingthrough stencil is enabled with increment or decrement z-fail action
    if (d3d9State().renderStates[D3DRS_STENCILENABLE] == TRUE &&
        d3d9State().renderStates[D3DRS_STENCILFUNC] == D3DCMP_ALWAYS &&
        (d3d9State().renderStates[D3DRS_STENCILZFAIL] == D3DSTENCILOP_DECR || d3d9State().renderStates[D3DRS_STENCILZFAIL] == D3DSTENCILOP_INCR ||
         d3d9State().renderStates[D3DRS_STENCILZFAIL] == D3DSTENCILOP_DECRSAT || d3d9State().renderStates[D3DRS_STENCILZFAIL] == D3DSTENCILOP_INCRSAT) &&
        d3d9State().renderStates[D3DRS_ZWRITEENABLE] == FALSE) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped stencil shadow drawcall."));
      return { RtxGeometryStatus::Ignored, false };
    }

    // Check UI only to the primary render target
    if (isRenderingUI()) {
      m_ue3LastDrawDecision = "UI detected (triggers RTX injection)";
      return {
        RtxGeometryStatus::Rasterized,
        true, // UI rendering detected => trigger RTX injection
      };
    }

    // TODO(REMIX-760): Support reverse engineering pre-transformed vertices
    if (d3d9State().vertexDecl != nullptr) {
      if (d3d9State().vertexDecl->TestFlag(D3D9VertexDeclFlag::HasPositionT)) {
        if (m_frameOptions.preTransformedVerticesIsUI) {
          return { RtxGeometryStatus::Rasterized, true };
        } else {
          ONCE(Logger::info("[RTX-Compatibility-Info] Skipped drawcall, using pre-transformed vertices which isn't currently supported."));
          return { RtxGeometryStatus::Rasterized, false };
        }
      }
    }

    logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::RayTraced, "default raytraced geometry");
    return { RtxGeometryStatus::RayTraced, false };
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
      // the identity exclusion option or the replacement diagnostics actually consume it.
      const bool wantDescriptorHashes =
        (m_frameOptions.ue3MicIdentityExcludedTextureDescHashes != nullptr &&
         !m_frameOptions.ue3MicIdentityExcludedTextureDescHashes->empty()) ||
        m_frameOptions.logReplacementResolution ||
        m_frameOptions.ue3LogMaterialInstanceHash;
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

  bool D3D9Rtx::isTaggedRaytracedRenderTarget(const Rc<DxvkImage>& image, bool includeAutoDetected) const {
    if (image == nullptr) {
      return false;
    }

    const XXH64_hash_t descriptorHash = image->getDescriptorHash();

    if (matchAuthoredRenderTargetTag(*m_frameOptions.raytracedRenderTargetTextures,
                                     descriptorHash,
                                     image->getResolutionAgnosticDescriptorHash()) != kEmptyHash) {
      return true;
    }

    // Auto-detection is session-local and keys on the absolute hash: the aspect-normalized hash
    // aliases every same-format target of the same aspect, including the backbuffer-sized target
    // the detector deliberately skips. The size check catches a resolution change turning a
    // previously detected size into the backbuffer's.
    return includeAutoDetected &&
           descriptorHash != kEmptyHash &&
           lookupHash(m_autoRaytracedRenderTargetDescHashes, descriptorHash) &&
           !isBackBufferSizedImage(image);
  }

  XXH64_hash_t D3D9Rtx::matchAuthoredRenderTargetTag(const fast_unordered_set& tags,
                                                     const XXH64_hash_t descriptorHash,
                                                     const XXH64_hash_t resolutionAgnosticDescriptorHash) {
    if (tags.empty()) {
      return kEmptyHash;
    }
    if (resolutionAgnosticDescriptorHash != kEmptyHash &&
        lookupHash(tags, resolutionAgnosticDescriptorHash)) {
      return resolutionAgnosticDescriptorHash;
    }
    if (descriptorHash != kEmptyHash && lookupHash(tags, descriptorHash)) {
      return descriptorHash;
    }
    return kEmptyHash;
  }

  bool D3D9Rtx::isBackBufferSizedImage(const Rc<DxvkImage>& image) const {
    if (image == nullptr || !m_activePresentParams.has_value()) {
      return false;
    }
    const VkExtent3D& extent = image->info().extent;
    return extent.width == m_activePresentParams->BackBufferWidth &&
           extent.height == m_activePresentParams->BackBufferHeight;
  }

  bool D3D9Rtx::isDeferredUiTaggedDraw(XXH64_hash_t* pMatchedTextureHash) const {
    // Pixel shader tag: stable across texture streaming and render target recreation
    if (!m_frameOptions.deferredUiPixelShaders->empty() &&
        m_parent->UseProgrammablePS() && d3d9State().pixelShader != nullptr) {
      const XXH64_hash_t psHash = d3d9State().pixelShader->GetCommonShader()->GetBytecodeHash();
      if (lookupHash(*m_frameOptions.deferredUiPixelShaders, psHash)) {
        return true;
      }
    }

    if (m_frameOptions.deferredUiTextures->empty()) {
      return false;
    }

    const uint32_t usedSamplerMask = m_parent->m_psShaderMasks.samplerMask | m_parent->m_vsShaderMasks.samplerMask;
    const BoundTextureSnapshot& boundTextures = ensureBoundTextureSnapshot();
    const uint32_t usedTextureMask = boundTextures.mask & usedSamplerMask;
    for (const uint32_t idx : bit::BitMask(usedTextureMask)) {
      const BoundTextureSnapshotEntry& entry = boundTextures.entries[idx];
      if (!entry.hasSampleView) {
        continue;
      }

      const auto reportMatch = [&](XXH64_hash_t matchedHash) {
        if (pMatchedTextureHash) {
          *pMatchedTextureHash = matchedHash;
        }
        return true;
      };

      // Non-RT overlays carry no descriptor hash, so they can only match by image hash.
      if (!entry.isRenderTarget) {
        const XXH64_hash_t texHash = entry.imageHash;
        if (texHash != kEmptyHash && lookupHash(*m_frameOptions.deferredUiTextures, texHash)) {
          return reportMatch(texHash);
        }
        continue;
      }

      const XXH64_hash_t matchedHash =
        matchAuthoredRenderTargetTag(*m_frameOptions.deferredUiTextures,
                                     entry.rtDescriptorHash,
                                     entry.rtResolutionAgnosticDescriptorHash);
      if (matchedHash != kEmptyHash) {
        return reportMatch(matchedHash);
      }
    }

    return false;
  }

  Rc<DxvkImage> D3D9Rtx::getCurrentRenderTargetImage() const {
    if (d3d9State().renderTargets[kRenderTargetIndex] == nullptr) {
      return nullptr;
    }

    D3D9CommonTexture* texInfo = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
    if (texInfo == nullptr) {
      return nullptr;
    }

    return texInfo->GetImage();
  }

  // Snapshots a draw tagged via rtx.deferredUiTextures so it can be replayed on top of the
  // ray-traced image after RTX injection. The referenced vertex/index ranges are copied to CPU
  // memory (the game may re-lock its dynamic buffers between capture and replay) and the draw
  // is later re-issued through the regular D3D9 UP draw path with the captured pipeline state.
  // Multi-stream draws (UE3 static meshes split position/tangents/UVs across streams) are
  // interleaved into a single stream-0 layout with a remapped vertex declaration.
  bool D3D9Rtx::captureDeferredUiDraw(const IndexContext& indexContext,
                                      const VertexContext vertexContext[caps::MaxStreams],
                                      const DrawContext& drawContext) {
    if (m_deferredUiDraws.size() >= kMaxDeferredUiDrawsPerFrame) {
      ONCE(Logger::warn("[RTX-DeferredUI] Too many deferred UI overlay draws in one frame; suppressing the rest. Check the rtx.deferredUiTextures tagging."));
      return false;
    }

    // Only the programmable pipeline is supported (UE3 MaterialEffect overlays are always
    // shader draws); fixed-function overlays would additionally need transform and texture
    // stage state capture.
    if (!m_parent->UseProgrammableVS() || !m_parent->UseProgrammablePS() ||
        d3d9State().vertexShader == nullptr || d3d9State().pixelShader == nullptr) {
      ONCE(Logger::warn("[RTX-DeferredUI] Fixed-function draw tagged as deferred UI overlay is not supported for replay; suppressing."));
      return false;
    }

    if (d3d9State().vertexDecl == nullptr) {
      return false;
    }

    // Instanced draws cannot be replayed through the UP path
    if ((d3d9State().streamFreq[0] & 0x7FFFFFu) > 1) {
      ONCE(Logger::warn("[RTX-DeferredUI] Instanced draw tagged as deferred UI overlay is not supported for replay; suppressing."));
      return false;
    }

    uint32_t usedStreamMask = 0;
    for (const auto& element : d3d9State().vertexDecl->GetElements()) {
      if (element.Stream == 0xFF) {
        continue; // D3DDECL_END
      }
      if (element.Stream >= caps::MaxStreams) {
        return false;
      }
      usedStreamMask |= 1u << element.Stream;
    }

    if (usedStreamMask == 0) {
      return false;
    }

    // Per-stream layout of the interleaved stream-0 vertex record used for replay
    uint32_t streamBase[caps::MaxStreams] = {};
    uint32_t combinedStride = 0;
    for (const uint32_t s : bit::BitMask(usedStreamMask)) {
      const VertexContext& v = vertexContext[s];
      if (v.stride == 0 || v.mappedSlice.mapPtr == nullptr) {
        return false;
      }
      if ((d3d9State().streamFreq[s] & D3DSTREAMSOURCE_INSTANCEDATA) != 0) {
        ONCE(Logger::warn("[RTX-DeferredUI] Instance-data stream on a draw tagged as deferred UI overlay is not supported for replay; suppressing."));
        return false;
      }
      streamBase[s] = combinedStride;
      combinedStride += v.stride;
    }

    if (combinedStride == 0 || combinedStride > 0xFFFF) {
      return false;
    }

    DeferredUiDraw draw;
    draw.primitiveType = drawContext.PrimitiveType;
    draw.primitiveCount = drawContext.PrimitiveCount;
    draw.indexed = drawContext.Indexed != FALSE;
    draw.vertexStride = combinedStride;

    int64_t firstVertex = 0;
    uint32_t vertexCount = 0;

    if (draw.indexed) {
      const uint32_t indexCount = GetVertexCount(drawContext.PrimitiveType, drawContext.PrimitiveCount);
      if (indexCount == 0 || indexCount > kMaxDeferredUiIndices) {
        ONCE(Logger::warn("[RTX-DeferredUI] Draw tagged as deferred UI overlay has too many indices for replay; suppressing."));
        return false;
      }

      if (indexContext.indexType == VK_INDEX_TYPE_NONE_KHR || indexContext.indexBuffer.mapPtr == nullptr) {
        return false;
      }

      const bool is16Bit = indexContext.indexType == VK_INDEX_TYPE_UINT16;
      const uint32_t indexStride = is16Bit ? 2 : 4;
      const size_t indexByteOffset = size_t(indexStride) * drawContext.StartIndex;
      if (indexByteOffset + size_t(indexStride) * indexCount > indexContext.indexBuffer.length) {
        return false;
      }

      const uint8_t* pIndexBase = static_cast<const uint8_t*>(indexContext.indexBuffer.mapPtr) + indexByteOffset;

      uint32_t minIndex = std::numeric_limits<uint32_t>::max();
      uint32_t maxIndex = 0;

      // Scan the used index range, then rebase the copied indices onto the copied vertex
      // window (widened to 32-bit for the replay draw)
      draw.indexData.resize(indexCount);
      const auto scanAndRebase = [&](const auto* pSrc) {
        for (uint32_t i = 0; i < indexCount; i++) {
          minIndex = std::min<uint32_t>(minIndex, pSrc[i]);
          maxIndex = std::max<uint32_t>(maxIndex, pSrc[i]);
        }
        for (uint32_t i = 0; i < indexCount; i++) {
          draw.indexData[i] = uint32_t(pSrc[i]) - minIndex;
        }
      };
      if (is16Bit) {
        scanAndRebase(reinterpret_cast<const uint16_t*>(pIndexBase));
      } else {
        scanAndRebase(reinterpret_cast<const uint32_t*>(pIndexBase));
      }

      vertexCount = maxIndex - minIndex + 1;
      firstVertex = int64_t(drawContext.BaseVertexIndex) + minIndex;
    } else {
      vertexCount = GetVertexCount(drawContext.PrimitiveType, drawContext.PrimitiveCount);
      firstVertex = drawContext.BaseVertexIndex; // StartVertex for DrawPrimitive, 0 for the UP path
    }

    if (vertexCount == 0 || firstVertex < 0) {
      return false;
    }

    const size_t vertexBytes = size_t(vertexCount) * combinedStride;
    if (vertexBytes > kMaxDeferredUiVertexBytes ||
        m_deferredUiFrameVertexBytes + vertexBytes > kMaxDeferredUiFrameVertexBytes) {
      ONCE(Logger::warn("[RTX-DeferredUI] Draw tagged as deferred UI overlay exceeds the vertex data replay budget; suppressing. Check the rtx.deferredUiTextures tagging."));
      return false;
    }

    // Validate source ranges for every referenced stream before copying anything
    for (const uint32_t s : bit::BitMask(usedStreamMask)) {
      const VertexContext& v = vertexContext[s];
      const size_t streamByteOffset = size_t(v.offset) + size_t(firstVertex) * v.stride;
      if (streamByteOffset + size_t(vertexCount) * v.stride > v.mappedSlice.length) {
        return false;
      }
    }

    draw.vertexCount = vertexCount;
    draw.vertexData.resize(vertexBytes);
    for (const uint32_t s : bit::BitMask(usedStreamMask)) {
      const VertexContext& v = vertexContext[s];
      const uint8_t* pSrc = static_cast<const uint8_t*>(v.mappedSlice.mapPtr) + v.offset + size_t(firstVertex) * v.stride;
      uint8_t* pDst = draw.vertexData.data() + streamBase[s];
      for (uint32_t i = 0; i < vertexCount; i++) {
        std::memcpy(pDst + size_t(i) * combinedStride, pSrc + size_t(i) * v.stride, v.stride);
      }
    }

    // Vertex declaration for the replay: the original when everything already lives on
    // stream 0, otherwise an internally created remap onto the interleaved stream-0 layout
    if (usedStreamMask == 1u) {
      draw.replayDecl = d3d9State().vertexDecl.ptr();
    } else {
      std::vector<D3DVERTEXELEMENT9> remappedElements;
      remappedElements.reserve(d3d9State().vertexDecl->GetElements().size() + 1);
      for (const auto& element : d3d9State().vertexDecl->GetElements()) {
        if (element.Stream == 0xFF) {
          continue;
        }
        D3DVERTEXELEMENT9 remapped = element;
        remapped.Stream = 0;
        remapped.Offset = WORD(streamBase[element.Stream] + element.Offset);
        remappedElements.push_back(remapped);
      }
      remappedElements.push_back(D3DDECL_END());

      Com<IDirect3DVertexDeclaration9> remappedDecl;
      if (FAILED(m_parent->CreateVertexDeclaration(remappedElements.data(), &remappedDecl)) || remappedDecl == nullptr) {
        ONCE(Logger::warn("[RTX-DeferredUI] Failed to create the remapped vertex declaration for a deferred UI overlay draw; suppressing."));
        return false;
      }
      draw.replayDecl = remappedDecl;
    }

    m_deferredUiFrameVertexBytes += uint32_t(vertexBytes);

    draw.vertexShader = d3d9State().vertexShader;
    draw.pixelShader = d3d9State().pixelShader;

    // Constants: only the ranges the shaders actually declare
    const auto& vsMeta = d3d9State().vertexShader->GetCommonShader()->GetMeta();
    const auto& psMeta = d3d9State().pixelShader->GetCommonShader()->GetMeta();

    const uint32_t vsFloatCount = std::min<uint32_t>(vsMeta.maxConstIndexF, caps::MaxFloatConstantsVS);
    const uint32_t psFloatCount = std::min<uint32_t>(psMeta.maxConstIndexF, caps::MaxFloatConstantsPS);
    const uint32_t vsIntCount = std::min<uint32_t>(vsMeta.maxConstIndexI, caps::MaxOtherConstants);
    const uint32_t psIntCount = std::min<uint32_t>(psMeta.maxConstIndexI, caps::MaxOtherConstants);
    const uint32_t vsBoolDwords = (std::min<uint32_t>(vsMeta.maxConstIndexB, caps::MaxOtherConstants) + 31u) / 32u;
    const uint32_t psBoolDwords = (std::min<uint32_t>(psMeta.maxConstIndexB, caps::MaxOtherConstants) + 31u) / 32u;

    draw.vsFloatConsts.assign(d3d9State().vsConsts.fConsts, d3d9State().vsConsts.fConsts + vsFloatCount);
    draw.psFloatConsts.assign(d3d9State().psConsts.fConsts, d3d9State().psConsts.fConsts + psFloatCount);
    draw.vsIntConsts.assign(d3d9State().vsConsts.iConsts, d3d9State().vsConsts.iConsts + vsIntCount);
    draw.psIntConsts.assign(d3d9State().psConsts.iConsts, d3d9State().psConsts.iConsts + psIntCount);
    draw.vsBoolConsts.assign(d3d9State().vsConsts.bConsts, d3d9State().vsConsts.bConsts + vsBoolDwords);
    draw.psBoolConsts.assign(d3d9State().psConsts.bConsts, d3d9State().psConsts.bConsts + psBoolDwords);

    // Texture bindings for every sampler the shaders use (including used-but-unbound slots so
    // the replay never samples whatever the app happens to have bound at replay time)
    const uint32_t usedSamplerMask = m_parent->m_psShaderMasks.samplerMask | m_parent->m_vsShaderMasks.samplerMask;
    for (const uint32_t idx : bit::BitMask(usedSamplerMask)) {
      if (idx >= SamplerCount) {
        continue;
      }

      DeferredUiDraw::TextureBinding binding;
      binding.slot = idx;
      binding.texture = d3d9State().textures[idx];
      binding.samplerStates = d3d9State().samplerStates[idx];

      // Track sampled render targets (scene color candidates for the refresh blit)
      if (d3d9State().textures[idx] != nullptr && (m_parent->GetActiveRTTextures() & (1u << idx)) != 0) {
        if (D3D9CommonTexture* texInfo = GetCommonTexture(d3d9State().textures[idx])) {
          binding.renderTargetImage = texInfo->GetImage();
        }
      }

      draw.textures.push_back(std::move(binding));
    }

    for (size_t i = 0; i < kDeferredUiRenderStates.size(); i++) {
      draw.renderStates[i] = d3d9State().renderStates[kDeferredUiRenderStates[i]];
    }

    draw.viewport = d3d9State().viewport;
    draw.scissorRect = d3d9State().scissorRect;

    if (d3d9State().renderTargets[kRenderTargetIndex] != nullptr) {
      const auto rtExtent = d3d9State().renderTargets[kRenderTargetIndex]->GetSurfaceExtent();
      draw.sourceRenderTargetWidth = rtExtent.width;
      draw.sourceRenderTargetHeight = rtExtent.height;
    }

    m_deferredUiDraws.push_back(std::move(draw));

    ONCE(Logger::info("[RTX-DeferredUI] Captured a deferred UI overlay draw for post-injection replay."));
    return true;
  }

  // Replays the deferred UI overlay draws captured this frame on top of the ray-traced image.
  // Called right after RTX injection is queued (mid-frame UI trigger, or the EndFrame fallback)
  // so the overlays land between the ray-traced blit and the game's UI rasterization.
  void D3D9Rtx::replayDeferredUiDraws(IDirect3DSurface9* pOverrideRenderTarget,
                                      const Rc<DxvkImage>& injectionTargetImage) {
    if (m_deferredUiDraws.empty()) {
      return;
    }

    // Take ownership up front: every early-out below must drop the captured draws rather
    // than leave them queued for a later, incorrectly ordered replay point
    std::vector<DeferredUiDraw> draws = std::move(m_deferredUiDraws);
    m_deferredUiDraws.clear();

    if (!m_frameOptions.deferredUiReplay) {
      return;
    }

    if (m_parent->ShouldRecord()) {
      // Mid state-block recording: internal Set* calls would be recorded instead of applied
      ONCE(Logger::warn("[RTX-DeferredUI] Skipping deferred UI overlay replay while a state block is being recorded."));
      return;
    }

    ScopedCpuProfileZone();

    // Refreshes the scene-color textures a replayed overlay samples with the current content
    // of the injection target, so scene-reading overlay materials (fade lerps, scope warps,
    // damage effects) composite over the ray-traced image instead of the stale rasterized
    // scene. Invoked before every replayed draw: an overlay's output on the target is picked
    // up by the next overlay's scene input, matching the game's own effect chaining.
    auto refreshSampledSceneTargets = [&](const DeferredUiDraw& draw) {
      if (!m_frameOptions.deferredUiRefreshSceneColor || injectionTargetImage == nullptr) {
        return;
      }

      for (const auto& binding : draw.textures) {
        const Rc<DxvkImage>& sceneImage = binding.renderTargetImage;
        if (sceneImage == nullptr || sceneImage == injectionTargetImage) {
          continue;
        }

        // Only refresh plausible scene-color targets (aspect ratio matching the final
        // image); small utility render targets keep their game-rendered content.
        const VkExtent3D dstExtent = sceneImage->info().extent;
        const VkExtent3D srcExtent = injectionTargetImage->info().extent;
        const double a = double(dstExtent.width) * double(srcExtent.height);
        const double b = double(dstExtent.height) * double(srcExtent.width);
        const double denom = std::max(a, b);
        if (denom <= 0.0 || (std::abs(a - b) / denom) >= 0.01) {
          continue;
        }

        m_parent->EmitCs([cSrcImage = injectionTargetImage, cDstImage = sceneImage](DxvkContext* ctx) {
          RtxContext::blitImageHelper(ctx, cSrcImage, cDstImage, VkFilter::VK_FILTER_NEAREST);
        });
      }
    };

    // ---- save every piece of application state the replay overrides ----

    uint32_t maxVsFloat = 0, maxPsFloat = 0, maxVsInt = 0, maxPsInt = 0, maxVsBool = 0, maxPsBool = 0;
    uint32_t touchedTextureSlots = 0;
    for (const auto& draw : draws) {
      maxVsFloat = std::max<uint32_t>(maxVsFloat, uint32_t(draw.vsFloatConsts.size()));
      maxPsFloat = std::max<uint32_t>(maxPsFloat, uint32_t(draw.psFloatConsts.size()));
      maxVsInt = std::max<uint32_t>(maxVsInt, uint32_t(draw.vsIntConsts.size()));
      maxPsInt = std::max<uint32_t>(maxPsInt, uint32_t(draw.psIntConsts.size()));
      maxVsBool = std::max<uint32_t>(maxVsBool, uint32_t(draw.vsBoolConsts.size()));
      maxPsBool = std::max<uint32_t>(maxPsBool, uint32_t(draw.psBoolConsts.size()));
      for (const auto& binding : draw.textures) {
        touchedTextureSlots |= 1u << binding.slot;
      }
    }

    Com<IDirect3DVertexDeclaration9> savedDecl(d3d9State().vertexDecl.ptr());
    Com<IDirect3DVertexShader9> savedVertexShader(d3d9State().vertexShader.ptr());
    Com<IDirect3DPixelShader9> savedPixelShader(d3d9State().pixelShader.ptr());
    Com<IDirect3DSurface9> savedRenderTarget(pOverrideRenderTarget != nullptr ? d3d9State().renderTargets[kRenderTargetIndex].ptr() : nullptr);
    Com<IDirect3DSurface9> savedDepthStencil(d3d9State().depthStencil.ptr());
    Com<IDirect3DVertexBuffer9> savedStream0(d3d9State().vertexBuffers[0].vertexBuffer.ptr());
    const UINT savedStream0Offset = d3d9State().vertexBuffers[0].offset;
    const UINT savedStream0Stride = d3d9State().vertexBuffers[0].stride;
    Com<IDirect3DIndexBuffer9> savedIndices(d3d9State().indices.ptr());
    const UINT savedStream0Freq = d3d9State().streamFreq[0];

    std::vector<Vector4> savedVsFloat(d3d9State().vsConsts.fConsts, d3d9State().vsConsts.fConsts + maxVsFloat);
    std::vector<Vector4> savedPsFloat(d3d9State().psConsts.fConsts, d3d9State().psConsts.fConsts + maxPsFloat);
    std::vector<Vector4i> savedVsInt(d3d9State().vsConsts.iConsts, d3d9State().vsConsts.iConsts + maxVsInt);
    std::vector<Vector4i> savedPsInt(d3d9State().psConsts.iConsts, d3d9State().psConsts.iConsts + maxPsInt);
    std::vector<uint32_t> savedVsBool(d3d9State().vsConsts.bConsts, d3d9State().vsConsts.bConsts + maxVsBool);
    std::vector<uint32_t> savedPsBool(d3d9State().psConsts.bConsts, d3d9State().psConsts.bConsts + maxPsBool);

    struct SavedTextureSlot {
      uint32_t slot;
      Com<IDirect3DBaseTexture9> texture;
      std::array<DWORD, SamplerStateCount> samplerStates;
    };
    std::vector<SavedTextureSlot> savedTextureSlots;
    for (const uint32_t idx : bit::BitMask(touchedTextureSlots)) {
      SavedTextureSlot saved;
      saved.slot = idx;
      saved.texture = d3d9State().textures[idx];
      saved.samplerStates = d3d9State().samplerStates[idx];
      savedTextureSlots.push_back(std::move(saved));
    }

    std::array<DWORD, kDeferredUiRenderStates.size()> savedRenderStates;
    for (size_t i = 0; i < kDeferredUiRenderStates.size(); i++) {
      savedRenderStates[i] = d3d9State().renderStates[kDeferredUiRenderStates[i]];
    }

    const D3DVIEWPORT9 savedViewport = d3d9State().viewport;
    const RECT savedScissor = d3d9State().scissorRect;

    // ---- replay ----

    m_replayingDeferredUiDraws = true;

    if (pOverrideRenderTarget != nullptr) {
      m_parent->SetRenderTarget(0, pOverrideRenderTarget);
    }

    // Overlays composite over the final image: depth/stencil contents at this point in the
    // frame are meaningless, and an incompatible depth surface must not clip the render area.
    m_parent->SetDepthStencilSurface(nullptr);
    m_parent->SetStreamSourceFreq(0, 1);

    uint32_t replayTargetWidth = 0, replayTargetHeight = 0;
    if (d3d9State().renderTargets[kRenderTargetIndex] != nullptr) {
      const auto rtExtent = d3d9State().renderTargets[kRenderTargetIndex]->GetSurfaceExtent();
      replayTargetWidth = rtExtent.width;
      replayTargetHeight = rtExtent.height;
    }

    for (const auto& draw : draws) {
      // Sync the overlay's scene inputs with the target's current content (ray-traced blit
      // plus any previously replayed overlays) before it draws
      refreshSampledSceneTargets(draw);

      m_parent->SetVertexDeclaration(draw.replayDecl.ptr());
      m_parent->SetVertexShader(draw.vertexShader.ptr());
      m_parent->SetPixelShader(draw.pixelShader.ptr());

      if (!draw.vsFloatConsts.empty()) {
        m_parent->SetVertexShaderConstantF(0, reinterpret_cast<const float*>(draw.vsFloatConsts.data()), UINT(draw.vsFloatConsts.size()));
      }
      if (!draw.psFloatConsts.empty()) {
        m_parent->SetPixelShaderConstantF(0, reinterpret_cast<const float*>(draw.psFloatConsts.data()), UINT(draw.psFloatConsts.size()));
      }
      if (!draw.vsIntConsts.empty()) {
        m_parent->SetVertexShaderConstantI(0, reinterpret_cast<const int*>(draw.vsIntConsts.data()), UINT(draw.vsIntConsts.size()));
      }
      if (!draw.psIntConsts.empty()) {
        m_parent->SetPixelShaderConstantI(0, reinterpret_cast<const int*>(draw.psIntConsts.data()), UINT(draw.psIntConsts.size()));
      }
      for (uint32_t i = 0; i < draw.vsBoolConsts.size(); i++) {
        m_parent->SetVertexBoolBitfield(i, ~0u, draw.vsBoolConsts[i]);
      }
      for (uint32_t i = 0; i < draw.psBoolConsts.size(); i++) {
        m_parent->SetPixelBoolBitfield(i, ~0u, draw.psBoolConsts[i]);
      }

      for (const auto& binding : draw.textures) {
        m_parent->SetStateTexture(binding.slot, binding.texture.ptr());
        for (uint32_t type = D3DSAMP_ADDRESSU; type < SamplerStateCount; type++) {
          m_parent->SetStateSamplerState(binding.slot, D3DSAMPLERSTATETYPE(type), binding.samplerStates[type]);
        }
      }

      for (size_t i = 0; i < kDeferredUiRenderStates.size(); i++) {
        m_parent->SetRenderState(kDeferredUiRenderStates[i], draw.renderStates[i]);
      }
      m_parent->SetRenderState(D3DRS_ZENABLE, D3DZB_FALSE);
      m_parent->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
      m_parent->SetRenderState(D3DRS_STENCILENABLE, FALSE);

      // Rescale the captured viewport/scissor when the capture-time render target and the
      // replay target differ in size (e.g. overlays captured on a scaled scene target)
      D3DVIEWPORT9 viewport = draw.viewport;
      RECT scissor = draw.scissorRect;
      if (replayTargetWidth != 0 && replayTargetHeight != 0 &&
          draw.sourceRenderTargetWidth != 0 && draw.sourceRenderTargetHeight != 0 &&
          (draw.sourceRenderTargetWidth != replayTargetWidth || draw.sourceRenderTargetHeight != replayTargetHeight)) {
        const double scaleX = double(replayTargetWidth) / double(draw.sourceRenderTargetWidth);
        const double scaleY = double(replayTargetHeight) / double(draw.sourceRenderTargetHeight);
        viewport.X = DWORD(viewport.X * scaleX);
        viewport.Y = DWORD(viewport.Y * scaleY);
        viewport.Width = std::max<DWORD>(1, DWORD(viewport.Width * scaleX));
        viewport.Height = std::max<DWORD>(1, DWORD(viewport.Height * scaleY));
        scissor.left = LONG(scissor.left * scaleX);
        scissor.right = LONG(scissor.right * scaleX);
        scissor.top = LONG(scissor.top * scaleY);
        scissor.bottom = LONG(scissor.bottom * scaleY);
      }
      m_parent->SetViewport(&viewport);
      m_parent->SetScissorRect(&scissor);

      if (draw.indexed) {
        m_parent->DrawIndexedPrimitiveUP(draw.primitiveType, 0, draw.vertexCount, draw.primitiveCount,
                                         draw.indexData.data(), D3DFMT_INDEX32,
                                         draw.vertexData.data(), draw.vertexStride);
      } else {
        m_parent->DrawPrimitiveUP(draw.primitiveType, draw.primitiveCount,
                                  draw.vertexData.data(), draw.vertexStride);
      }
    }

    // ---- restore the application state ----

    if (pOverrideRenderTarget != nullptr && savedRenderTarget != nullptr) {
      m_parent->SetRenderTarget(0, savedRenderTarget.ptr());
    }
    m_parent->SetDepthStencilSurface(savedDepthStencil.ptr());

    m_parent->SetVertexDeclaration(savedDecl.ptr());
    m_parent->SetVertexShader(savedVertexShader.ptr());
    m_parent->SetPixelShader(savedPixelShader.ptr());

    if (maxVsFloat != 0) {
      m_parent->SetVertexShaderConstantF(0, reinterpret_cast<const float*>(savedVsFloat.data()), maxVsFloat);
    }
    if (maxPsFloat != 0) {
      m_parent->SetPixelShaderConstantF(0, reinterpret_cast<const float*>(savedPsFloat.data()), maxPsFloat);
    }
    if (maxVsInt != 0) {
      m_parent->SetVertexShaderConstantI(0, reinterpret_cast<const int*>(savedVsInt.data()), maxVsInt);
    }
    if (maxPsInt != 0) {
      m_parent->SetPixelShaderConstantI(0, reinterpret_cast<const int*>(savedPsInt.data()), maxPsInt);
    }
    for (uint32_t i = 0; i < maxVsBool; i++) {
      m_parent->SetVertexBoolBitfield(i, ~0u, savedVsBool[i]);
    }
    for (uint32_t i = 0; i < maxPsBool; i++) {
      m_parent->SetPixelBoolBitfield(i, ~0u, savedPsBool[i]);
    }

    for (const auto& saved : savedTextureSlots) {
      m_parent->SetStateTexture(saved.slot, saved.texture.ptr());
      for (uint32_t type = D3DSAMP_ADDRESSU; type < SamplerStateCount; type++) {
        m_parent->SetStateSamplerState(saved.slot, D3DSAMPLERSTATETYPE(type), saved.samplerStates[type]);
      }
    }

    for (size_t i = 0; i < kDeferredUiRenderStates.size(); i++) {
      m_parent->SetRenderState(kDeferredUiRenderStates[i], savedRenderStates[i]);
    }

    m_parent->SetViewport(&savedViewport);
    m_parent->SetScissorRect(&savedScissor);

    m_parent->SetStreamSource(0, savedStream0.ptr(), savedStream0Offset, savedStream0Stride);
    m_parent->SetIndices(savedIndices.ptr());
    m_parent->SetStreamSourceFreq(0, savedStream0Freq);

    m_replayingDeferredUiDraws = false;

    ONCE(Logger::info(str::format("[RTX-DeferredUI] Replayed ", draws.size(), " deferred UI overlay draw(s) after RTX injection.")));
  }

  // Folds the per-instance VS transform constants (LocalToWorld, or the leading bone
  // matrix rows for skinned draws) into `seed`. Identities derived from shared
  // buffers/ranges alone cannot distinguish different placements of the same mesh.
  XXH64_hash_t D3D9Rtx::mixUe3InstanceTransformConstants(XXH64_hash_t seed) const {
    if (!m_currentUe3CtabInfo.has_value()) {
      return seed;
    }

    const Ue3VsShaderCtabInfo& ctabInfo = *m_currentUe3CtabInfo;
    auto mixRange = [&](const uint32_t reg, uint32_t count) {
      count = std::min(count, 4u);
      if (count == 0 || reg + count > caps::MaxFloatConstantsVS) {
        return;
      }
      seed = XXH3_64bits_withSeed(&d3d9State().vsConsts.fConsts[reg], count * sizeof(Vector4), seed);
    };

    if (ctabInfo.hasLocalToWorld) {
      mixRange(ctabInfo.localToWorldRegisterIndex, ctabInfo.localToWorldRegisterCount);
    }
    if (ctabInfo.hasBoneMatrices) {
      mixRange(ctabInfo.boneMatricesRegisterIndex, std::min(ctabInfo.boneMatricesRegisterCount, 3u));
    }
    return seed;
  }

  // Detects draws whose submission outcome (raytraced/rasterized/ignored) changes between
  // nearby frames. A stable scene submits every draw with the same outcome every frame; an
  // outcome flap is geometry visibly popping in/out of the raytraced scene. The log names
  // the pass classification and decision reason on both sides of the flap.
  void D3D9Rtx::trackUe3DrawStatusFlap(const DrawContext& drawContext, const PrepareDrawFlags flags) {
    struct DrawIdentity {
      const void* pIndexBuffer;
      const void* pVertexBuffer0;
      const void* pVertexDecl;
      const void* pVertexShader;
      const void* pPixelShader;
      uint32_t vb0Offset;
      int32_t baseVertexIndex;
      uint32_t startIndex;
      uint32_t primitiveCount;
      uint32_t cullMode; // separates legit two-sided back/front passes into distinct identities
      uint32_t reserved;
    };
    static_assert(sizeof(DrawIdentity) == 5 * sizeof(void*) + 6 * sizeof(uint32_t),
                  "DrawIdentity must have no implicit padding (it is hashed by memory).");
    const DrawIdentity identity = {
      d3d9State().indices.ptr(),
      d3d9State().vertexBuffers[0].vertexBuffer.ptr(),
      d3d9State().vertexDecl.ptr(),
      d3d9State().vertexShader.ptr(),
      d3d9State().pixelShader.ptr(),
      d3d9State().vertexBuffers[0].offset,
      drawContext.BaseVertexIndex,
      drawContext.StartIndex,
      drawContext.PrimitiveCount,
      d3d9State().renderStates[D3DRS_CULLMODE],
      0u,
    };
    const XXH64_hash_t key = mixUe3InstanceTransformConstants(XXH3_64bits(&identity, sizeof(identity)));

    if (m_ue3DrawStatusCache.size() > 65536) {
      m_ue3DrawStatusCache.clear();
    }

    auto& entry = m_ue3DrawStatusCache[key];
    const uint32_t frame = m_ue3FrameCounter + 1u; // +1 so lastFrame==0 means "never seen"

    // only commit-bit changes alter visibility; e.g. a vertex-capture draw alternating
    // between capture (CommitToRayTracing|preserve) and cached reuse (CommitToRayTracing)
    // is benign and would otherwise flood the log
    const bool commitChanged =
      ((entry.lastFlags ^ uint32_t(flags)) & PrepareDrawFlag::CommitToRayTracing) != 0;

    if (entry.lastFrame != 0 && frame != entry.lastFrame &&
        frame - entry.lastFrame <= 3u &&
        commitChanged &&
        entry.logCount < 8u) {
      entry.logCount++;

      XXH64_hash_t vsHash = 0;
      if (d3d9State().vertexShader.ptr() != nullptr) {
        vsHash = d3d9State().vertexShader->GetCommonShader()->GetBytecodeHash();
      }
      XXH64_hash_t psHash = 0;
      if (d3d9State().pixelShader.ptr() != nullptr) {
        psHash = d3d9State().pixelShader->GetCommonShader()->GetBytecodeHash();
      }

      Logger::warn(str::format(
        "[RTX-Compatibility][UE3-StatusFlap] draw outcome changed between frames: ",
        "flags ", entry.lastFlags, "->", uint32_t(flags),
        " pass ", describeUe3PassType(Ue3PassType(entry.lastPassType)), "->", describeUe3PassType(m_currentUe3PassType),
        " decision='", entry.lastDecision, "'->'", m_ue3LastDrawDecision, "'",
        " frameDelta=", frame - entry.lastFrame,
        " vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
        " prims=", drawContext.PrimitiveCount,
        " alphaBlend=", d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] != FALSE ? 1 : 0,
        " cull=", d3d9State().renderStates[D3DRS_CULLMODE],
        " vsHash=0x", std::hex, vsHash,
        " psHash=0x", psHash, std::dec,
        " (", entry.logCount, "/8)"));
    }

    entry.lastFlags = uint32_t(flags);
    entry.lastFrame = frame;
    entry.lastDecision = m_ue3LastDrawDecision;
    entry.lastPassType = uint8_t(m_currentUe3PassType);
  }

  // A raytraced draw whose material ends up with no albedo texture renders as an untextured
  // surface with nothing to click in the texture UI. Log the full sampler picture once per
  // pixel shader so the cause is attributable (shader samples no textures at all vs. all
  // candidates unbindable/skipped); for sampler-less shaders also dump the constant table
  // and live UniformVector values to identify the shader's role.
  void D3D9Rtx::logUe3UnboundAlbedoOnce(const D3D9CommonShader* pixelShader,
                                        const XXH64_hash_t psHash,
                                        const uint32_t usedSamplerMask,
                                        const uint32_t usedTextureMask,
                                        const PsSamplerTexcoordEntry* inferredEntry) {
    if (pixelShader == nullptr || psHash == kEmptyHash) {
      return;
    }

    static fast_unordered_set s_loggedShaders;
    if (!s_loggedShaders.insert(psHash).second) {
      return;
    }

    const auto& samplerNames = getUe3PsSamplerNames(psHash, pixelShader->GetBytecode());
    std::string samplerLog;
    for (uint32_t stage : bit::BitMask(usedSamplerMask)) {
      if (stage >= caps::MaxTexturesPS) {
        continue;
      }
      samplerLog += samplerLog.empty() ? "s" : ", s";
      samplerLog += str::format(stage);
      const auto nameIt = samplerNames.find(stage);
      samplerLog += str::format("(", nameIt != samplerNames.end() ? nameIt->second.c_str() : "?", ")");
      if (d3d9State().textures[stage] == nullptr) {
        samplerLog += "=unbound";
        continue;
      }
      D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[stage]);
      if (texture == nullptr || texture->GetImage() == nullptr) {
        samplerLog += "=noimage";
        continue;
      }
      samplerLog += str::format(
        "=hash:0x", std::hex, texture->GetImage()->getHash(),
        ",desc:0x", texture->GetImage()->getDescriptorHash(), std::dec,
        texture->IsRenderTarget() ? ",RT" : "",
        isLightmapTexture(texture->GetImage()->getHash()) ? ",lightmap" : "",
        ",type:", uint32_t(texture->GetType()),
        ",samples:", inferredEntry != nullptr ? inferredEntry->samplerSampleCount[stage] : 0);
    }

    std::string constantLog;
    if (usedSamplerMask == 0) {
      const auto& bytecode = pixelShader->GetBytecode();
      if (bytecode.size() >= sizeof(uint32_t) && (bytecode.size() % sizeof(uint32_t)) == 0) {
        const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
        if ((tokens[0] & 0xffff0000u) == 0xffff0000u) {
          DxsoProgramInfo programInfo(DxsoProgramTypes::PixelShader, tokens[0] & 0xffu, (tokens[0] >> 8) & 0xffu);
          DxsoDecodeContext decoder(programInfo);
          DxsoCodeIter iter(tokens + 1);
          while (decoder.decodeInstruction(iter)) {
            if (decoder.getCtabInfo().m_size != 0) {
              break;
            }
          }
          for (const DxsoCtab::Constant& c : decoder.getCtabInfo().m_constantData) {
            constantLog += constantLog.empty() ? "" : ", ";
            constantLog += str::format(c.name, "@", c.registerSet == kD3dxRegisterSetSampler ? "s" : "c", c.registerIndex);
          }
        }
      }
      const Ue3PsMaterialIdentityInfo& identityInfo = getOrParseUe3PsMaterialIdentityInfo(psHash, bytecode, pixelShader);
      for (const uint32_t reg : identityInfo.uniformVectorRegisters) {
        if (reg >= caps::MaxFloatConstantsPS) {
          continue;
        }
        const Vector4& v = d3d9State().psConsts.fConsts[reg];
        constantLog += str::format(" | c", reg, "=(", v.x, ",", v.y, ",", v.z, ",", v.w, ")");
      }
    }
    if (!constantLog.empty()) {
      constantLog = str::format(" ctab=[", constantLog, "]");
    }

    Logger::warn(str::format(
      "[RTX-Compatibility][UE3-NoAlbedo] Draw has no bindable albedo texture: ps=0x", std::hex, psHash, std::dec,
      " vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
      " usedSamplerMask=0x", std::hex, usedSamplerMask,
      " boundUsedMask=0x", usedTextureMask, std::dec,
      " samplers=[", samplerLog.empty() ? "none" : samplerLog, "]",
      " alphaBlend=", d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] != FALSE ? 1 : 0,
      " srcBlend=", d3d9State().renderStates[D3DRS_SRCBLEND],
      " dstBlend=", d3d9State().renderStates[D3DRS_DESTBLEND],
      constantLog));
  }

  void D3D9Rtx::logUe3ParticleDrawOnce() {
    if (!m_frameOptions.ue3LogClassification || !m_frameOptions.ue3EngineMode) {
      return;
    }

    const Ue3VertexFactoryType vf = m_currentUe3VertexFactory;
    if (vf != Ue3VertexFactoryType::Particle &&
        vf != Ue3VertexFactoryType::ParticleBeamTrail &&
        vf != Ue3VertexFactoryType::LensFlare) {
      return;
    }

    XXH64_hash_t psHash = kEmptyHash;
    if (m_parent->UseProgrammablePS() && d3d9State().pixelShader.ptr() != nullptr) {
      psHash = d3d9State().pixelShader->GetCommonShader()->GetBytecodeHash();
    }

    const XXH64_hash_t textureHash = m_activeDrawCallState.materialData.getColorTexture().getImageHash();
    const XXH64_hash_t materialHash = m_activeDrawCallState.materialData.getHash();
    const XXH64_hash_t textureSetShaderHash = m_activeDrawCallState.materialData.getTextureSetAndShaderHash();
    const bool hasAlbedo = m_activeDrawCallState.materialData.colorTextures[0].isValid();
    const bool hasConstantAlbedo = m_activeDrawCallState.materialData.hasUe3ConstantAlbedo;

    const auto& cats = m_activeDrawCallState.getCategoryFlags();
    const bool particleTagged = cats.test(InstanceCategories::Particle);
    const bool beamTagged = cats.test(InstanceCategories::Beam);
    const bool hideTagged = cats.test(InstanceCategories::Hidden);
    const bool ignoreTagged = cats.test(InstanceCategories::Ignore);

    // Include category bits so author tag toggles produce a fresh line.
    XXH64_hash_t key = XXH3_64bits_withSeed(&vf, sizeof(vf), 0);
    key = XXH3_64bits_withSeed(&psHash, sizeof(psHash), key);
    key = XXH3_64bits_withSeed(&materialHash, sizeof(materialHash), key);
    key = XXH3_64bits_withSeed(&textureHash, sizeof(textureHash), key);
    const uint32_t categoryBits =
      (particleTagged ? 1u : 0u) |
      (beamTagged ? 2u : 0u) |
      (hideTagged ? 4u : 0u) |
      (ignoreTagged ? 8u : 0u);
    key = XXH3_64bits_withSeed(&categoryBits, sizeof(categoryBits), key);

    if (!m_loggedUe3ParticleDraws.insert(key).second) {
      return;
    }

    const bool alphaBlend = d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] != FALSE;
    const DWORD srcBlend = d3d9State().renderStates[D3DRS_SRCBLEND];
    const DWORD dstBlend = d3d9State().renderStates[D3DRS_DESTBLEND];
    const bool likelyEmissiveBlend =
      alphaBlend &&
      ((srcBlend == D3DBLEND_SRCALPHA && dstBlend == D3DBLEND_ONE) ||
       (srcBlend == D3DBLEND_ONE && dstBlend == D3DBLEND_ONE) ||
       (srcBlend == D3DBLEND_SRCCOLOR && dstBlend == D3DBLEND_ONE) ||
       (srcBlend == D3DBLEND_ONE && dstBlend == D3DBLEND_SRCCOLOR));

    std::string constantAlbedoLog;
    if (hasConstantAlbedo) {
      const Vector4& c = m_activeDrawCallState.materialData.ue3ConstantAlbedo;
      constantAlbedoLog = str::format(" constantAlbedo=(", c.x, ",", c.y, ",", c.z, ",", c.w, ")");
    }

    Logger::info(str::format(
      "[RTX-Compatibility][UE3-Particle] vf=", describeUe3VertexFactory(vf),
      " ps=0x", std::hex, psHash,
      " materialHash=0x", materialHash,
      " textureHash=0x", textureHash,
      " textureSetShaderHash=0x", textureSetShaderHash, std::dec,
      " hasAlbedo=", hasAlbedo ? 1 : 0,
      " hasConstantAlbedo=", hasConstantAlbedo ? 1 : 0,
      " particleTagged=", particleTagged ? 1 : 0,
      " beamTagged=", beamTagged ? 1 : 0,
      " hideTagged=", hideTagged ? 1 : 0,
      " ignoreTagged=", ignoreTagged ? 1 : 0,
      " alphaBlend=", alphaBlend ? 1 : 0,
      " srcBlend=", srcBlend,
      " dstBlend=", dstBlend,
      " likelyEmissiveBlend=", likelyEmissiveBlend ? 1 : 0,
      constantAlbedoLog));
  }

  namespace {
    const char* ue3DeclUsageName(const BYTE usage) {
      switch (usage) {
      case D3DDECLUSAGE_POSITION:     return "POSITION";
      case D3DDECLUSAGE_BLENDWEIGHT:  return "BLENDWEIGHT";
      case D3DDECLUSAGE_BLENDINDICES: return "BLENDINDICES";
      case D3DDECLUSAGE_NORMAL:       return "NORMAL";
      case D3DDECLUSAGE_PSIZE:        return "PSIZE";
      case D3DDECLUSAGE_TEXCOORD:     return "TEXCOORD";
      case D3DDECLUSAGE_TANGENT:      return "TANGENT";
      case D3DDECLUSAGE_BINORMAL:     return "BINORMAL";
      case D3DDECLUSAGE_TESSFACTOR:   return "TESSFACTOR";
      case D3DDECLUSAGE_POSITIONT:    return "POSITIONT";
      case D3DDECLUSAGE_COLOR:        return "COLOR";
      case D3DDECLUSAGE_FOG:          return "FOG";
      case D3DDECLUSAGE_DEPTH:        return "DEPTH";
      case D3DDECLUSAGE_SAMPLE:       return "SAMPLE";
      default:                        return "?";
      }
    }

    const char* ue3DeclTypeName(const BYTE type) {
      switch (type) {
      case D3DDECLTYPE_FLOAT1:    return "FLOAT1";
      case D3DDECLTYPE_FLOAT2:    return "FLOAT2";
      case D3DDECLTYPE_FLOAT3:    return "FLOAT3";
      case D3DDECLTYPE_FLOAT4:    return "FLOAT4";
      case D3DDECLTYPE_D3DCOLOR:  return "D3DCOLOR";
      case D3DDECLTYPE_UBYTE4:    return "UBYTE4";
      case D3DDECLTYPE_SHORT2:    return "SHORT2";
      case D3DDECLTYPE_SHORT4:    return "SHORT4";
      case D3DDECLTYPE_UBYTE4N:   return "UBYTE4N";
      case D3DDECLTYPE_SHORT2N:   return "SHORT2N";
      case D3DDECLTYPE_SHORT4N:   return "SHORT4N";
      case D3DDECLTYPE_USHORT2N:  return "USHORT2N";
      case D3DDECLTYPE_USHORT4N:  return "USHORT4N";
      case D3DDECLTYPE_UDEC3:     return "UDEC3";
      case D3DDECLTYPE_DEC3N:     return "DEC3N";
      case D3DDECLTYPE_FLOAT16_2: return "FLOAT16_2";
      case D3DDECLTYPE_FLOAT16_4: return "FLOAT16_4";
      case D3DDECLTYPE_UNUSED:    return "UNUSED";
      default:                    return "?";
      }
    }

    std::string formatMatrixRows(const Matrix4& m) {
      std::string out;
      for (uint32_t row = 0; row < 4; row++) {
        out += str::format(row == 0 ? "[" : " [",
                           m[row].x, ",", m[row].y, ",", m[row].z, ",", m[row].w, "]");
      }
      return out;
    }
  }

  // Shader, material and bound-texture hashes for a draw, so a warning about one names something
  // that can be looked up in the texture picker or fed to rtx.d3d9.ue3TraceDrawTextureHashes.
  std::string D3D9Rtx::describeUe3DrawIdentity() const {
    XXH64_hash_t vsHash = kEmptyHash;
    if (m_parent->UseProgrammableVS() && d3d9State().vertexShader.ptr() != nullptr) {
      vsHash = d3d9State().vertexShader->GetCommonShader()->GetBytecodeHash();
    }
    XXH64_hash_t psHash = kEmptyHash;
    if (m_parent->UseProgrammablePS() && d3d9State().pixelShader.ptr() != nullptr) {
      psHash = d3d9State().pixelShader->GetCommonShader()->GetBytecodeHash();
    }

    std::string textures;
    for (uint32_t stage = 0; stage < LegacyMaterialData::kMaxSupportedTextures; stage++) {
      const XXH64_hash_t imageHash = m_activeDrawCallState.materialData.colorTextures[stage].getImageHash();
      if (imageHash == kEmptyHash) {
        continue;
      }
      textures += str::format(textures.empty() ? "" : " ", "s", stage, ":0x", std::hex, imageHash, std::dec);
    }

    return str::format(
      "vs=0x", std::hex, vsHash,
      " ps=0x", psHash,
      " materialHash=0x", m_activeDrawCallState.materialData.getHash(), std::dec,
      " textures=[", textures.empty() ? "none" : textures, "]");
  }

  std::string D3D9Rtx::describeUe3VertexDeclaration() const {
    if (d3d9State().vertexDecl == nullptr) {
      return "none";
    }

    std::string out;
    for (const auto& element : d3d9State().vertexDecl->GetElements()) {
      out += str::format(out.empty() ? "" : " | ",
                         "s", uint32_t(element.Stream), ":",
                         ue3DeclUsageName(element.Usage), uint32_t(element.UsageIndex),
                         " ", ue3DeclTypeName(element.Type),
                         " @", uint32_t(element.Offset));
    }
    return out.empty() ? "empty" : out;
  }

  std::string D3D9Rtx::describeUe3DrawInstancing(const VertexContext vertexContext[caps::MaxStreams]) const {
    const Ue3InstancingInfo& info = m_currentUe3Instancing;

    std::string streams;
    for (uint32_t s = 0; s < caps::MaxStreams; s++) {
      const VertexContext& ctx = vertexContext[s];
      if (ctx.mappedSlice.mapPtr == nullptr && d3d9State().streamFreq[s] == 1) {
        continue;
      }
      const bool dynamic =
        ctx.pVBO != nullptr && ctx.pVBO->Desc() != nullptr &&
        (ctx.pVBO->Desc()->Usage & D3DUSAGE_DYNAMIC) != 0;
      streams += str::format(streams.empty() ? "" : " | ",
                             "s", s,
                             " freq=0x", std::hex, uint32_t(d3d9State().streamFreq[s]), std::dec,
                             " stride=", ctx.stride,
                             " dynamic=", dynamic ? 1 : 0);
    }

    std::string transform = "none";
    if (info.hasInstanceTransform) {
      transform = str::format("stream=", info.transformStream,
                              " offset@", info.offsetByteOffset,
                              " axes@", info.axisByteOffsets[0],
                              "/", info.axisByteOffsets[1],
                              "/", info.axisByteOffsets[2]);
    }

    return str::format("instances=", info.instanceCount,
                       " instanceDataStreams=0x", std::hex, info.instanceDataStreamMask, std::dec,
                       " instanceTransform=[", transform, "]",
                       " decomposed=", uint32_t(m_ue3DecomposedInstances.size()),
                       " streams=[", streams.empty() ? "none" : streams, "]");
  }

  void D3D9Rtx::logUe3InstancedDrawOnce(const DrawContext& drawContext,
                                        const VertexContext vertexContext[caps::MaxStreams],
                                        const RasterGeometry& geoData) {
    if (!m_frameOptions.ue3LogInstancedDraws || !m_currentUe3Instancing.isInstanced()) {
      return;
    }

    XXH64_hash_t vsHash = kEmptyHash;
    if (m_parent->UseProgrammableVS() && d3d9State().vertexShader.ptr() != nullptr) {
      vsHash = d3d9State().vertexShader->GetCommonShader()->GetBytecodeHash();
    }
    XXH64_hash_t psHash = kEmptyHash;
    if (m_parent->UseProgrammablePS() && d3d9State().pixelShader.ptr() != nullptr) {
      psHash = d3d9State().pixelShader->GetCommonShader()->GetBytecodeHash();
    }

    // The instance count varies with the particle population, so it is deliberately left out of
    // the key: one line per (shader, declaration, factory) is the point.
    struct Key {
      XXH64_hash_t vsHash;
      XXH64_hash_t psHash;
      const void* pVertexDecl;
      uint32_t vertexFactory;
      uint32_t passType;
      uint32_t positionSource;
      uint32_t hasInstanceTransform;
    };
    const Key key = {
      vsHash,
      psHash,
      d3d9State().vertexDecl.ptr(),
      uint32_t(m_currentUe3VertexFactory),
      uint32_t(m_currentUe3PassType),
      uint32_t(m_activeCapturePositionSource),
      m_currentUe3Instancing.hasInstanceTransform ? 1u : 0u,
    };
    if (!m_loggedUe3InstancedDraws.insert(XXH3_64bits(&key, sizeof(key))).second) {
      return;
    }

    Logger::info(str::format(
      "[RTX-Compatibility][UE3-Instanced] vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
      " pass=", describeUe3PassType(m_currentUe3PassType),
      " posSource=", describeUe3CapturePositionSource(m_activeCapturePositionSource),
      " ", describeUe3DrawInstancing(vertexContext),
      " prims=", drawContext.PrimitiveCount,
      " verts=", geoData.vertexCount,
      " indices=", geoData.indexCount,
      " ", describeUe3DrawIdentity(),
      " decl=[", describeUe3VertexDeclaration(), "]"));
  }

  void D3D9Rtx::logUe3TracedDrawOnce(const DrawContext& drawContext,
                                     const VertexContext vertexContext[caps::MaxStreams],
                                     RasterGeometry& geoData) {
    if (m_frameOptions.ue3TraceDrawTextureHashes == nullptr ||
        m_frameOptions.ue3TraceDrawTextureHashes->empty()) {
      return;
    }

    // Any bound colour texture matching the list arms the dossier, not just the chosen albedo:
    // the point is to find a draw from a hash read off the texture picker.
    bool matched = false;
    for (const auto& texture : m_activeDrawCallState.materialData.colorTextures) {
      const XXH64_hash_t imageHash = texture.getImageHash();
      if (imageHash != kEmptyHash && lookupHash(*m_frameOptions.ue3TraceDrawTextureHashes, imageHash)) {
        matched = true;
        break;
      }
    }
    if (!matched) {
      return;
    }

    XXH64_hash_t vsHash = kEmptyHash;
    if (m_parent->UseProgrammableVS() && d3d9State().vertexShader.ptr() != nullptr) {
      vsHash = d3d9State().vertexShader->GetCommonShader()->GetBytecodeHash();
    }
    XXH64_hash_t psHash = kEmptyHash;
    if (m_parent->UseProgrammablePS() && d3d9State().pixelShader.ptr() != nullptr) {
      psHash = d3d9State().pixelShader->GetCommonShader()->GetBytecodeHash();
    }

    const XXH64_hash_t materialHash = m_activeDrawCallState.materialData.getHash();

    struct Key {
      XXH64_hash_t vsHash;
      XXH64_hash_t psHash;
      XXH64_hash_t materialHash;
      const void* pVertexDecl;
      uint32_t vertexFactory;
      uint32_t passType;
      uint32_t positionSource;
      uint32_t reserved;
    };
    const Key key = {
      vsHash,
      psHash,
      materialHash,
      d3d9State().vertexDecl.ptr(),
      uint32_t(m_currentUe3VertexFactory),
      uint32_t(m_currentUe3PassType),
      uint32_t(m_activeCapturePositionSource),
      0u,
    };
    if (!m_loggedUe3TracedDraws.insert(XXH3_64bits(&key, sizeof(key))).second) {
      return;
    }

    // Geometry hashing normally completes on a worker and is collected on the CS thread. Sync it
    // here so the dossier can report the value replacements anchor on; finalizeGeometryHashes
    // accepts an already-resolved RasterGeometry, so consuming the future is safe.
    if (geoData.futureGeometryHashes.valid()) {
      geoData.hashes = geoData.futureGeometryHashes.get();
    }
    const XXH64_hash_t assetGeometryHash = geoData.getHashForRule(RtxOptions::geometryAssetHashRule());
    const XXH64_hash_t fullGeometryHash = geoData.getHashForRule<rules::FullGeometryHash>();

    std::string textures;
    for (uint32_t stage = 0; stage < LegacyMaterialData::kMaxSupportedTextures; stage++) {
      const XXH64_hash_t imageHash = m_activeDrawCallState.materialData.colorTextures[stage].getImageHash();
      if (imageHash == kEmptyHash) {
        continue;
      }
      textures += str::format(textures.empty() ? "" : " ", "s", stage, ":0x", std::hex, imageHash, std::dec);
    }

    std::string instanceTransforms;
    {
      constexpr size_t kMaxLoggedTransforms = 4;
      const size_t count = std::min(kMaxLoggedTransforms, m_ue3DecomposedInstances.size());
      for (size_t i = 0; i < count; i++) {
        const Ue3DecomposedInstance& instance = m_ue3DecomposedInstances[i];
        instanceTransforms += str::format("\n    instance[", instance.sourceIndex, "]=",
                                          formatMatrixRows(instance.instanceToObject));
      }
      if (m_ue3DecomposedInstances.size() > count) {
        instanceTransforms += str::format("\n    ... ",
                                          uint32_t(m_ue3DecomposedInstances.size() - count),
                                          " more");
      }
    }

    Logger::info(str::format(
      "[RTX-Compatibility][UE3-DrawTrace] vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
      " pass=", describeUe3PassType(m_currentUe3PassType),
      " posSource=", describeUe3CapturePositionSource(m_activeCapturePositionSource),
      " vs=0x", std::hex, vsHash,
      " ps=0x", psHash,
      " materialHash=0x", materialHash,
      " assetGeometryHash=0x", assetGeometryHash,
      " fullGeometryHash=0x", fullGeometryHash, std::dec,
      "\n    prims=", drawContext.PrimitiveCount,
      " verts=", geoData.vertexCount,
      " indices=", geoData.indexCount,
      " topology=", uint32_t(geoData.topology),
      " cull=", uint32_t(geoData.cullMode),
      " textures=[", textures.empty() ? "none" : textures, "]",
      "\n    ", describeUe3DrawInstancing(vertexContext),
      "\n    decl=[", describeUe3VertexDeclaration(), "]",
      "\n    objectToWorld=", formatMatrixRows(m_activeDrawCallState.transformData.objectToWorld),
      instanceTransforms));
  }

  bool D3D9Rtx::isUe3RenderTargetRefusedAsAlbedo(D3D9CommonTexture* texture,
                                                 const uint32_t stage,
                                                 const PsSamplerTexcoordEntry* inferredEntry) const {
    if (texture == nullptr || !texture->IsRenderTarget() || texture->GetImage() == nullptr) {
      return false;
    }

    const Rc<DxvkImage>& image = texture->GetImage();
    if (isUe3MovieTextureDescHash(image->getDescriptorHash())) {
      return false;
    }
    if (inferredEntry != nullptr && stage < caps::MaxTexturesPS &&
        (inferredEntry->samplerSemanticFlags[stage] & kPsSamplerSemanticMovieTexture) != 0) {
      return false;
    }
    // Author intent wins: a preferred-albedo tag, or a screen whose material needs the target
    // itself bound as the albedo.
    if (lookupHash(*m_frameOptions.preferredAlbedoTextures, image->getHash())) {
      return false;
    }
    return !isTaggedRaytracedRenderTarget(image, true);
  }

  PrepareDrawFlags D3D9Rtx::internalPrepareDraw(const IndexContext& indexContext, const VertexContext vertexContext[caps::MaxStreams], const DrawContext& drawContext) {
    ScopedCpuProfileZone();

    m_ue3LastDrawDecision = "";

    // Texture bindings cannot change within a draw; rebuild the shared snapshot lazily on
    // first use per draw (UI/deferred-UI tag checks, MIC texture-set hash, diffuse key).
    m_boundTextureSnapshotValid = false;

    // Diagnostics: record every draw issued inside an occlusion query bracket, whichever path
    // routes it below, so its query's eventual result can be correlated with the drawn geometry.
    if (m_activeOcclusionQueries > 0 && m_frameOptions.ue3LogOcclusionQueries) {
      recordOcclusionQueryBracketedDraw(vertexContext, drawContext);
    }

    auto finishPrepare = [&](PrepareDrawFlags flags) {
      if (m_frameOptions.ue3LogDrawStatusFlaps && m_frameOptions.ue3EngineMode) {
        trackUe3DrawStatusFlap(drawContext, flags);
      }
      return flags;
    };

    // RTX was injected => treat everything else as rasterized,
    // unless this draw targets a raytraced render target (e.g. render-to-texture
    // in games that draw UI before 3D content).
    if (m_rtxInjectTriggered) {
      bool isRaytracedRenderTarget = false;
      if (m_frameOptions.raytracedRenderTargetEnable &&
          d3d9State().renderTargets[kRenderTargetIndex] != nullptr) {
        D3D9CommonTexture* texture = GetCommonTexture(d3d9State().renderTargets[kRenderTargetIndex]->GetBaseTexture());
        if (texture) {
          const Rc<DxvkImage> image = texture->GetImage();
          if (image != nullptr) {
            isRaytracedRenderTarget = isTaggedRaytracedRenderTarget(image, true);
          }
        }
      }
      if (!isRaytracedRenderTarget) {
        // Occlusion query test draws can land post-injection (games without a depth prepass
        // issue queries after the base pass); same conservative handling as pre-injection.
        if (ShouldApplyConservativeOcclusionQueryState()) {
          m_ue3LastDrawDecision = "occlusion query test draw post RTX injection (ignored, result synthesized)";
          return finishPrepare(PrepareDrawFlag::Ignore);
        }

        m_ue3LastDrawDecision = "post RTX injection";
        return finishPrepare(m_frameOptions.skipDrawCallsPostRTXInjection
               ? PrepareDrawFlag::Ignore
               : PrepareDrawFlag::PreserveDrawCallAndItsState);
      }
    }

    // classify UE3 vertex factory early so makeDrawCallType can use it for pass filtering
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

    // Hardware instancing state is needed before the capture source is resolved: a per-vertex
    // capture buffer cannot represent a draw that runs its vertices once per instance. Kept behind
    // ue3EngineMode with the rest of the UE3 behaviour - recovering placements from a stream relies
    // on knowing the engine's instance layout, and without that there is nothing better to do with
    // an instanced draw than what Remix already did.
    // Leaving this at its default (no instancing seen) is what makes ue3DecomposeInstancedDraws a
    // true bypass: every decision downstream keys off instanceCount, so with the option off an
    // instanced draw is handled exactly as it was before decomposition existed.
    m_currentUe3Instancing = Ue3InstancingInfo();
    m_ue3DecomposedInstances.clear();
    m_ue3DecomposedBatchKey = kEmptyHash;
    // m_activeDrawCallState is reused across draws, so a previous draw's per-instance identity must
    // not leak into an ordinary one and detach it from the transform its identity depends on.
    m_activeDrawCallState.decomposedInstanceId = kEmptyHash;
    if (m_frameOptions.ue3EngineMode && m_frameOptions.ue3DecomposeInstancedDraws &&
        d3d9State().vertexDecl != nullptr) {
      m_currentUe3Instancing = resolveUe3Instancing(
        d3d9State().vertexDecl->GetElements(), d3d9State().streamFreq, m_parent->GetInstanceCount());
    }

    const auto [status, triggerRtxInjection, deferUntilInjection] = makeDrawCallType(drawContext);

    // When raytracing is enabled we want to completely remove the ignored drawcalls from further processing as early as possible
    const PrepareDrawFlags prepareFlagsForIgnoredDraws = m_frameOptions.enableRaytracing
                                                         ? PrepareDrawFlag::Ignore
                                                         : PrepareDrawFlag::PreserveDrawCallAndItsState;

    if (status == RtxGeometryStatus::Ignored) {
      return finishPrepare(prepareFlagsForIgnoredDraws);
    }

    // Deferred UI overlay: snapshot the draw for post-injection replay and suppress it here.
    // Executing it now would rasterize into a pre-injection target that the ray-traced blit
    // overwrites; letting it trigger injection would end the ray-traced scene mid-frame.
    if (deferUntilInjection) {
      if (m_frameOptions.deferredUiReplay && captureDeferredUiDraw(indexContext, vertexContext, drawContext)) {
        m_ue3LastDrawDecision = "deferred UI overlay (captured for post-injection replay)";
      } else {
        m_ue3LastDrawDecision = m_frameOptions.deferredUiReplay
                                ? "deferred UI overlay (capture unsupported, suppressed)"
                                : "deferred UI overlay (suppressed)";
      }
      return finishPrepare(prepareFlagsForIgnoredDraws);
    }

    if (triggerRtxInjection) {
      // Bind all resources required for this drawcall to context first (i.e. render targets)
      m_parent->PrepareDraw(drawContext.PrimitiveType);

      triggerInjectRTX();

      m_rtxInjectTriggered = true;

      // Replay deferred overlays now, before this triggering UI draw executes: the required
      // order is ray-traced blit, then deferred overlays, then the game's genuine UI on top.
      // Replaying any later would put overlays above draws tagged via rtx.uiTextures.
      if (!m_deferredUiDraws.empty()) {
        replayDeferredUiDraws(nullptr, getCurrentRenderTargetImage());
      }

      return finishPrepare(PrepareDrawFlag::PreserveDrawCallAndItsState);
    }

    if (status == RtxGeometryStatus::Rasterized) {
      return finishPrepare(PrepareDrawFlag::PreserveDrawCallAndItsState);
    }

    m_forceGeometryCopy = m_frameOptions.useBuffersDirectly == false;
    m_forceGeometryCopy |= m_parent->GetOptions()->allowDiscard == false;

    // The packet we'll send to RtxContext with information about geometry
    RasterGeometry& geoData = m_activeDrawCallState.geometryData;
    geoData = {};
    geoData.cullMode = DecodeCullMode(D3DCULL(d3d9State().renderStates[D3DRS_CULLMODE]));
    geoData.frontFace = VK_FRONT_FACE_CLOCKWISE;
    geoData.topology = DecodeInputAssemblyState(drawContext.PrimitiveType).primitiveTopology;

    // This can be negative!!
    int vertexIndexOffset = drawContext.BaseVertexIndex;

    // Process index buffer
    uint32_t minIndex = 0, maxIndex = 0;
    if (indexContext.indexType != VK_INDEX_TYPE_NONE_KHR) {
      geoData.indexCount = GetVertexCount(drawContext.PrimitiveType, drawContext.PrimitiveCount);

      if (indexContext.indexType == VK_INDEX_TYPE_UINT16)
        geoData.indexBuffer = RasterBuffer(processIndexBuffer<uint16_t>(geoData.indexCount, drawContext.StartIndex, indexContext, minIndex, maxIndex), 0, 2, indexContext.indexType);
      else
        geoData.indexBuffer = RasterBuffer(processIndexBuffer<uint32_t>(geoData.indexCount, drawContext.StartIndex, indexContext, minIndex, maxIndex), 0, 4, indexContext.indexType);

      // Unlikely, but invalid
      if (maxIndex == minIndex) {
        ONCE(Logger::info("[RTX-Compatibility-Info] Skipped invalid drawcall, no triangles detected in index buffer."));
        return finishPrepare(prepareFlagsForIgnoredDraws);
      }

      geoData.vertexCount = maxIndex - minIndex + 1;
      vertexIndexOffset += minIndex;
    } else {
      geoData.vertexCount = GetVertexCount(drawContext.PrimitiveType, drawContext.PrimitiveCount);
    }

    if (geoData.vertexCount == 0) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped invalid drawcall, no vertices detected."));
      return finishPrepare(prepareFlagsForIgnoredDraws);
    }

    if (m_frameOptions.raytracedRenderTargetEnable) {
      // If this draw call has an RT texture bound
      if (m_activeDrawCallState.isUsingRaytracedRenderTarget) {
        // We validate this state below
        m_activeDrawCallState.isUsingRaytracedRenderTarget = false;
        // Try and find the has of the positions
        for (uint32_t i : bit::BitMask(m_parent->GetActiveRTTextures())) {
          D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[i]);
          if (texture == nullptr || texture->GetImage() == nullptr) {
            continue;
          }
          // Manual tags only here (auto-detection applies to drawing *to* an RT, not sampling it).
          if (isTaggedRaytracedRenderTarget(texture->GetImage(), false)) {
            m_activeDrawCallState.isUsingRaytracedRenderTarget = true;
          }
        }
      }
    }

    m_activeDrawCallState.categories = 0;
    m_activeDrawCallState.materialData = {};

    // Fetch all the legacy state (colour modes, alpha test, etc...)
    setLegacyMaterialState(m_parent, m_parent->m_alphaSwizzleRTs & (1 << kRenderTargetIndex), m_frameOptions.vertexColorIsBakedLighting, m_activeDrawCallState.materialData);

    // Fetch fog state 
    setFogState(m_parent, m_activeDrawCallState.fogState);

    // Fetch all the render state and send it to rtx context (textures, transforms, etc.)
    if (!processRenderState(drawContext)) {
      return finishPrepare(prepareFlagsForIgnoredDraws);
    }

    // Max offseted index value within a buffer slice that geoData contains
    const uint32_t maxOffsetedIndex = maxIndex - minIndex;

    // Copy all the vertices into a staging buffer.  Assign fields of the geoData structure.
    processVertices(vertexContext, vertexIndexOffset, geoData);

    // Recover the placements UE3 hid in the instance-data stream. Done before the capture source is
    // resolved so that path can prefer input-assembler positions once the placements are in hand:
    // object-space positions are only usable if there is a transform to place them with.
    m_ue3InstanceTransformReadFailure = nullptr;
    if (m_currentUe3Instancing.instanceCount > 1) {
      const char* readReason = "";
      if (readUe3InstanceTransforms(vertexContext, m_ue3DecomposedInstances, &readReason)) {
        const size_t instancesRead = m_ue3DecomposedInstances.size();

        // Both of these read the batch as the game wrote it, before any culling: the batch key's
        // centroid has to stay view-independent, and the order probe measures the game's order.
        m_ue3DecomposedBatchKey = resolveUe3InstancedBatchKey(geoData, m_ue3DecomposedInstances);
        if (m_frameOptions.ue3LogInstancedDrawStats) {
          trackUe3InstanceOrderStability(m_ue3DecomposedBatchKey, m_ue3DecomposedInstances);
        }

        uint32_t culledByDistance = 0;
        uint32_t culledByBudget = 0;
        cullAndClampUe3InstanceTransforms(m_ue3DecomposedInstances, culledByDistance, culledByBudget);

        if (culledByBudget > 0) {
          ONCE(Logger::warn(str::format(
            "[RTX-Compatibility] UE3 instanced draw expands to ", instancesRead,
            " instances, above rtx.d3d9.ue3MaxDecomposedInstances (",
            std::max(m_frameOptions.ue3MaxDecomposedInstances, 1u), "); keeping a fixed ",
            m_ue3DecomposedInstances.size(), " of them and dropping ", culledByBudget,
            ". The kept subset is deliberately the same every frame - a view-dependent one flickers. ",
            describeUe3DrawIdentity(),
            " verts=", geoData.vertexCount, " prims=", drawContext.PrimitiveCount)));
        }

        if (m_frameOptions.ue3LogInstancedDrawStats) {
          ++m_ue3InstancedStatDraws;
          m_ue3InstancedStatInstancesSeen += instancesRead;
          m_ue3InstancedStatInstancesSubmitted += m_ue3DecomposedInstances.size();
          m_ue3InstancedStatCulledDistance += culledByDistance;
          m_ue3InstancedStatCulledBudget += culledByBudget;
        }
      } else {
        // Without placements, input-assembler positions would put the mesh at the world origin, so
        // the draw is refused below rather than rendered somewhere it does not belong.
        m_ue3InstanceTransformReadFailure = readReason;
      }
    }

    // UE3 vertex shader skinned (GPUSkin) draws share one bind-pose vertex buffer and bounding box
    // across every instance of a skeletal mesh, so the BLAS cache cannot tell simultaneous instances
    // apart on geometry alone. The first bone's translation separates them, and its bone hash gives
    // the geometry refit decision a handle on the animated pose. The bone matrices are RefToLocal,
    // so that translation only becomes a world position once carried through the LocalToWorld
    // processRenderState resolved above; used raw, every instance anchors near the world origin.
    m_activeDrawCallState.m_hasSkinnedWorldAnchor = false;
    // Reset the per-draw skinning identity: only VS-skinned draws (re)assign a bone hash
    // below, and processSkinning() leaves programmable-VS skinningData untouched. Without
    // this reset the last skinned draw's bone hash leaks into every subsequent draw; as
    // the pose animates, that leaked hash churns the BLAS refit decision, the DrawCallCache
    // exact-match and the ReplacementInstance identity of every static draw each frame,
    // forcing full geometry re-uploads and the dynamic path scene-wide.
    m_activeDrawCallState.skinningData = SkinningData();
    const bool usesVertexShaderSkinning =
      m_frameOptions.ue3EngineMode &&
      m_parent->UseProgrammableVS() &&
      m_currentUe3CtabInfo.has_value() &&
      m_currentUe3CtabInfo->hasBoneMatrices;
    if (usesVertexShaderSkinning &&
        m_currentUe3CtabInfo->boneMatricesRegisterCount >= 3) {
      const D3D9ConstantSets& cb = m_parent->m_consts[DxsoProgramTypes::VertexShader];
      const uint32_t floatConstRegCount = cb.meta.maxConstIndexF;
      const uint32_t boneReg = m_currentUe3CtabInfo->boneMatricesRegisterIndex;
      const uint32_t boneRegCount =
        std::min(m_currentUe3CtabInfo->boneMatricesRegisterCount, floatConstRegCount > boneReg ? floatConstRegCount - boneReg : 0u);
      if (boneRegCount >= 3) {
        const auto& fConsts = d3d9State().vsConsts.fConsts;
        // UE3 bone matrices are float4x3 (3 float4 rows per bone) and the translation lives in
        // the .w of the first three rows of the first bone
        const Vector3 boneTranslation(
          fConsts[boneReg + 0].w,
          fConsts[boneReg + 1].w,
          fConsts[boneReg + 2].w);
        m_activeDrawCallState.m_skinnedWorldAnchor =
          (m_activeDrawCallState.transformData.objectToWorld * Vector4(boneTranslation, 1.0f)).xyz();
        m_activeDrawCallState.m_hasSkinnedWorldAnchor = true;

        // processSkinning() returns no SkinningData for programmable-VS draws, so skinningData
        // stays default (numBones == 0) and this bone hash will not be overwritten by finalise
        m_activeDrawCallState.skinningData.boneHash =
          XXH3_64bits(&fConsts[boneReg], size_t(boneRegCount) * sizeof(Vector4));
      }
    }

    // Resolve where this draw's positions come from before anything keys off it: the capture
    // cache is only valid for exact sources, and the strict gate below refuses the inexact one.
    m_activeCapturePositionSource = Ue3CapturePositionSource::ClipReconstruction;
    if (m_parent->UseProgrammableVS() && m_frameOptions.useVertexCapture) {
      const char* positionSourceReason = "";
      m_activeCapturePositionSource = resolveUe3CapturePositionSource(
        indexContext, vertexContext, geoData, &positionSourceReason);
      logUe3CapturePositionSource(m_activeCapturePositionSource, positionSourceReason);

      if (m_frameOptions.ue3RequireExactVertexCapture &&
          !isUe3ExactCapturePositionSource(m_activeCapturePositionSource)) {
        m_ue3LastDrawDecision = "no exact vertex capture position source";
        ONCE(Logger::info("[RTX-Compatibility-Info] Ignoring draw without an exact vertex capture position source (rtx.d3d9.ue3RequireExactVertexCapture)."));
        return finishPrepare(prepareFlagsForIgnoredDraws);
      }

      // An instanced draw that did not land on input-assembler positions has no correct answer
      // left: capture would have every hardware instance overwrite one another's slots, producing
      // a cloud of triangles built from mixed placements that changes every time it is captured.
      // Dropping the draw is the honest outcome.
      if (m_currentUe3Instancing.instanceCount > 1 &&
          m_activeCapturePositionSource != Ue3CapturePositionSource::InputAssembler) {
        m_ue3LastDrawDecision = "hardware-instanced draw without input-assembler positions";
        ONCE(Logger::warn(str::format(
          "[RTX-Compatibility] Ignoring hardware-instanced draw: its positions would have to come from "
          "vertex capture, which is indexed per vertex and so cannot separate instances (vf=",
          describeUe3VertexFactory(m_currentUe3VertexFactory),
          ", reason='", positionSourceReason, "') ",
          describeUe3DrawIdentity(), " ",
          describeUe3DrawInstancing(vertexContext),
          " decl=[", describeUe3VertexDeclaration(), "]")));
        return finishPrepare(prepareFlagsForIgnoredDraws);
      }
    }

    bool canUseCachedVertexCapture = false;
    XXH64_hash_t vertexCaptureCacheKey = kEmptyHash;
    {
      ScopedCpuProfileZoneN("UE3 geometry identity keys");

      // Stable VS hash (bytecode + camera-excluded constants), computed once per draw and
      // shared between the geometry hash below and the static vertex-capture cache key.
      m_activeStableVsHash = (m_parent->UseProgrammableVS() && m_frameOptions.useVertexCapture)
        ? computeUe3StableVertexShaderHash()
        : kEmptyHash;

      // Static-draw identity for the vertex-capture cache.
      canUseCachedVertexCapture =
        canUseUe3StaticVertexCaptureCache(indexContext, vertexContext, geoData);
      vertexCaptureCacheKey =
        canUseCachedVertexCapture
          ? computeUe3StaticVertexCaptureCacheKey(indexContext, vertexContext, drawContext, geoData)
          : kEmptyHash;

      // Geometry hash + bounding box memoization: draws with static IA buffers (any vertex
      // factory - skinned bind-pose data included) hash to the same IA components every
      // frame, so serve published results instead of re-hashing the full vertex/index data
      // per draw. The per-draw VertexShader component is recombined live so served hashes
      // are bit-identical to a fresh compute. First sighting schedules the normal worker
      // compute, which additionally publishes into the (heap-pinned) memo entry.
      bool servedGeometryFromMemo = false;
      std::shared_ptr<Ue3GeometryMemoEntry> geometryMemoPublishTo;
      const bool canMemoizeIaGeometry = canMemoizeUe3IaGeometryHashes(indexContext, vertexContext, geoData);
      const XXH64_hash_t iaGeometryMemoKey =
        canMemoizeIaGeometry
          ? computeUe3IaGeometryMemoKey(indexContext, vertexContext, drawContext, geoData)
          : kEmptyHash;

      // Diagnostic only. Reuses the memo key above rather than recomputing its own - that hash
      // walks every vertex stream record, which is far too much to spend twice per draw. Keyed off
      // structural eligibility rather than canUseCachedVertexCapture, because the question it
      // answers is why the cache is not hitting, so it has to keep working once the cache has
      // stood itself down.
      if (m_frameOptions.ue3LogVertexConstantChurn &&
          canMemoizeIaGeometry &&
          isUe3StaticVertexCaptureCacheEligible(indexContext, vertexContext, geoData)) {
        trackUe3ConstantChurn(iaGeometryMemoKey, geoData);
      }

      if (canMemoizeIaGeometry) {
        const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();
        const auto memoIt = m_ue3GeometryMemoCache.find(iaGeometryMemoKey);
        if (memoIt != m_ue3GeometryMemoCache.end()) {
          Ue3GeometryMemoEntry& entry = *memoIt->second;
          entry.lastFrameTouched = currentFrame;
          if (entry.hashesReady.load(std::memory_order_acquire)) {
            GeometryHashes hashes;
            for (uint32_t i = 0; i < uint32_t(HashComponents::Count); i++) {
              hashes[HashComponents(i)] = entry.componentHashes[i];
            }
            hashes[HashComponents::VertexShader] = computeLiveGeometryVertexShaderHashComponent();
            hashes.precombine();
            geoData.hashes = hashes;
            servedGeometryFromMemo = true;
            if (entry.aabbReady.load(std::memory_order_acquire)) {
              geoData.boundingBox = entry.boundingBox;
            } else {
              geoData.futureBoundingBox = computeAxisAlignedBoundingBox(geoData);
            }
          }
          // hashes not ready yet (worker still busy from an earlier frame): fall through
          // and compute normally this draw, without publishing a second time
        } else {
          geometryMemoPublishTo = std::make_shared<Ue3GeometryMemoEntry>();
          geometryMemoPublishTo->lastFrameTouched = currentFrame;
          m_ue3GeometryMemoCache.emplace(iaGeometryMemoKey, geometryMemoPublishTo);
        }
      }

      if (!servedGeometryFromMemo) {
        geoData.futureGeometryHashes = computeHash(geoData, maxOffsetedIndex, geometryMemoPublishTo);
        geoData.futureBoundingBox = computeAxisAlignedBoundingBox(geoData, geometryMemoPublishTo);

        if (geometryMemoPublishTo != nullptr && !geoData.futureGeometryHashes.valid()) {
          // hashing could not be scheduled (e.g. undefined position region): drop the
          // placeholder entry so it does not linger unfilled
          m_ue3GeometryMemoCache.erase(iaGeometryMemoKey);
        }
      }
    }

    // Process skinning data
    m_activeDrawCallState.futureSkinningData = processSkinning(geoData);

    // Note: the material hash was already updated inside processTextures; nothing
    // mutates material data after that point, so no second update is needed here.

    const bool reusedCachedVertexCapture =
      canUseCachedVertexCapture &&
      tryReuseUe3StaticVertexCapture(vertexCaptureCacheKey, geoData);
    if (m_frameOptions.ue3LogCapturePrecision &&
        Logger::logLevel() <= LogLevel::Debug &&
        canUseCachedVertexCapture) {
      static fast_unordered_set s_loggedCacheReuse;
      const XXH64_hash_t logKey = vertexCaptureCacheKey ^ (reusedCachedVertexCapture ? 0x9E3779B97F4A7C15ull : 0xD1B54A32D192ED03ull);
      if (s_loggedCacheReuse.insert(logKey).second) {
        Logger::debug(str::format(
          "[RTX-Compatibility][UE3-Capture] static vertex capture cache ",
          reusedCachedVertexCapture ? "reused" : "recapturing",
          ", key=0x", std::hex, vertexCaptureCacheKey, std::dec,
          ", vertices=", geoData.vertexCount));
      }
    }

    // Every captured member is written at data[vertexId - baseVertex], so a draw whose vertices run
    // once per hardware instance has all of its instances aliasing one another. The declaration
    // already supplied this draw's positions, normals and UVs in object space.
    const bool instancedDrawSuppressesCapture = m_currentUe3Instancing.instanceCount > 1;

    // For shader based drawcalls we also want to capture the vertex shader output
    bool needVertexCapture =
      m_parent->UseProgrammableVS() &&
      m_frameOptions.useVertexCapture &&
      !reusedCachedVertexCapture &&
      !instancedDrawSuppressesCapture;
    if (needVertexCapture) {
      needVertexCapture = prepareVertexCapture(vertexIndexOffset, m_activeCapturePositionSource);
    }
    if (canUseCachedVertexCapture && !reusedCachedVertexCapture && needVertexCapture) {
      ++m_ue3VertexCaptureCacheFrameCaptures;
      updateUe3StaticVertexCaptureCache(vertexCaptureCacheKey, geoData);
    }
    m_activeDrawCallState.usesVertexShader = m_parent->UseProgrammableVS();
    m_activeDrawCallState.usesPixelShader = m_parent->UseProgrammablePS();

    if (m_activeDrawCallState.usesVertexShader) {
      m_activeDrawCallState.programmableVertexShaderInfo = d3d9State().vertexShader->GetCommonShader()->GetInfo();
    }
    
    if (m_activeDrawCallState.usesPixelShader) {
      m_activeDrawCallState.programmablePixelShaderInfo = d3d9State().pixelShader->GetCommonShader()->GetInfo();
    }
    
    m_activeDrawCallState.cameraType = CameraType::Unknown;

    m_activeDrawCallState.minZ = std::clamp(d3d9State().viewport.MinZ, 0.0f, 1.0f);
    m_activeDrawCallState.maxZ = std::clamp(d3d9State().viewport.MaxZ, 0.0f, 1.0f);

    m_activeDrawCallState.zWriteEnable = d3d9State().renderStates[D3DRS_ZWRITEENABLE];
    m_activeDrawCallState.zEnable = d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_TRUE;
    
    // Now that the DrawCallState is complete, we can use heuristics for detection
    m_activeDrawCallState.setupCategoriesForHeuristics(m_seenCameraPositionsPrev.size(),
                                                       m_seenCameraPositions);

    if (m_frameOptions.fogIgnoreSky && m_activeDrawCallState.categories.test(InstanceCategories::Sky)) {
      m_activeDrawCallState.fogState.mode = D3DFOG_NONE;
    }

    // Ignore sky draw calls that are being drawn to a Raytraced Render Target
    // Raytraced Render Target scenes just use the same sky as the main scene, no need to duplicate them
    if (m_activeDrawCallState.isDrawingToRaytracedRenderTarget && m_activeDrawCallState.categories.test(InstanceCategories::Sky)) {
      return finishPrepare(prepareFlagsForIgnoredDraws);
    }

    assert(status == RtxGeometryStatus::RayTraced);

    logUe3InstancedDrawOnce(drawContext, vertexContext, geoData);
    logUe3TracedDrawOnce(drawContext, vertexContext, geoData);

    const bool preserveOriginalDraw = needVertexCapture;

    return finishPrepare(
      PrepareDrawFlag::CommitToRayTracing |
      (m_activeDrawCallState.testCategoryFlags(CATEGORIES_REQUIRE_DRAW_CALL_STATE) ? PrepareDrawFlag::ApplyDrawState : 0) |
      (preserveOriginalDraw ? PrepareDrawFlag::PreserveDrawCallAndItsState : 0));
  }

  void D3D9Rtx::triggerInjectRTX() {
    // Flush any pending game and RTX work
    m_parent->Flush();

    // Send command to inject RTX
    m_parent->EmitCs([cReflexFrameId = GetReflexFrameId()](DxvkContext* ctx) {
      static_cast<RtxContext*>(ctx)->injectRTX(cReflexFrameId);
    });
  }

  void D3D9Rtx::CommitGeometryToRT(const DrawContext& drawContext) {
    ScopedCpuProfileZone();
    auto drawInfo = m_parent->GenerateDrawInfo(drawContext.PrimitiveType, drawContext.PrimitiveCount, m_parent->GetInstanceCount());

    DrawParameters params;
    params.instanceCount = drawInfo.instanceCount;
    params.vertexOffset = drawContext.BaseVertexIndex;
    params.firstIndex = drawContext.StartIndex;
    // DXVK overloads the vertexCount/indexCount in DrawInfo
    if (drawContext.Indexed) {
      params.indexCount = drawInfo.vertexCount; 
    } else {
      params.vertexCount = drawInfo.vertexCount;
    }

    if (!m_ue3DecomposedInstances.empty()) {
      // UE3 only instances through programmable vertex shaders, for which processSkinning returns
      // nothing. Were skinning data pending, duplicating the draw state would hand several copies
      // the same one-shot task, and finalizeSkinningData would recompute objectToWorld from the
      // camera and discard the per-instance placement anyway - so leave such a draw undecomposed.
      if (m_activeDrawCallState.futureSkinningData.valid()) {
        ONCE(Logger::warn("[RTX-Compatibility] Not decomposing a hardware-instanced draw with pending "
                          "skinning data; the batch will be placed as a single instance."));
      } else {
        // One ray-traced instance per hardware instance, each carrying the placement read out of the
        // instance stream. The geometry is a single object-space copy of the mesh, so this reproduces
        // what UE3's non-instanced NxFluid mesh path submits: one draw per particle with
        // LocalToWorld = FMatrix(XAxis, YAxis, ZAxis, Location).
        params.instanceCount = 1;
        submitUe3DecomposedInstanceDrawCallStates(params);
        return;
      }
    }

    submitActiveDrawCallState();

    m_parent->EmitCs([params, this](DxvkContext* ctx) {
      assert(dynamic_cast<RtxContext*>(ctx));
      DrawCallState drawCallState;
      if (m_drawCallStateQueue.pop(drawCallState)) {
        static_cast<RtxContext*>(ctx)->commitGeometryToRT(params, drawCallState);
      }
    });
  }

  void D3D9Rtx::submitUe3DecomposedInstanceDrawCallStates(const DrawParameters& params) {
    ScopedCpuProfileZone();

    const bool timeSubmission = m_frameOptions.ue3LogInstancedDrawStats;
    const auto submitStart = timeSubmission ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point();

    // Every copy of the draw state shares the pending futures' task pointers, and a task result is
    // a one-shot: the first consumer disposes it and the rest would wait on a result that is never
    // set again. Resolve them here so all copies carry finished data. Steady state usually costs
    // nothing, because the geometry hash memo serves these draws without scheduling a future at all.
    // (The caller guarantees there is no pending skinning future.)
    RasterGeometry& geoData = m_activeDrawCallState.geometryData;
    if (geoData.futureGeometryHashes.valid()) {
      geoData.hashes = geoData.futureGeometryHashes.get();
    }
    if (geoData.futureBoundingBox.valid()) {
      geoData.boundingBox = geoData.futureBoundingBox.get();
    }

    const Matrix4 objectToWorld = m_activeDrawCallState.transformData.objectToWorld;
    const Matrix4 worldToView = m_activeDrawCallState.transformData.worldToView;
    const size_t instanceCount = m_ue3DecomposedInstances.size();

    const bool stableIdentity =
      m_frameOptions.ue3StableDecomposedInstanceIdentity && m_ue3DecomposedBatchKey != kEmptyHash;

    for (size_t i = 0; i < instanceCount; i++) {
      const Ue3DecomposedInstance& instance = m_ue3DecomposedInstances[i];

      DrawCallTransforms& transforms = m_activeDrawCallState.transformData;
      transforms.objectToWorld = objectToWorld * instance.instanceToObject;
      transforms.objectToView = worldToView * transforms.objectToWorld;
      transforms.sanitize();

      if (stableIdentity) {
        // Keyed on the instance's position in the game's buffer, not in this vector: culling changes
        // how many instances precede it, and renaming an instance costs it its history.
        const uint64_t sourceIndex = uint64_t(instance.sourceIndex);
        XXH64_hash_t instanceId =
          XXH3_64bits_withSeed(&sourceIndex, sizeof(sourceIndex), m_ue3DecomposedBatchKey);
        if (instanceId == kEmptyHash) {
          instanceId = 1; // kEmptyHash means "not a decomposed instance"
        }
        m_activeDrawCallState.decomposedInstanceId = instanceId;
      }

      if (i + 1 < instanceCount) {
        // push() leaves its argument untouched when the queue is full, so retrying with the same
        // copy is safe.
        DrawCallState instanceDrawCallState = m_activeDrawCallState;
        while (!m_drawCallStateQueue.push(std::move(instanceDrawCallState))) {
          Sleep(0);
        }
      } else {
        submitActiveDrawCallState();
      }

      m_parent->EmitCs([params, this](DxvkContext* ctx) {
        assert(dynamic_cast<RtxContext*>(ctx));
        DrawCallState drawCallState;
        if (m_drawCallStateQueue.pop(drawCallState)) {
          static_cast<RtxContext*>(ctx)->commitGeometryToRT(params, drawCallState);
        }
      });
    }

    if (timeSubmission) {
      m_ue3InstancedStatSubmitNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - submitStart).count());
    }
  }

  void D3D9Rtx::submitActiveDrawCallState() {
    // We must be prepared for `push` failing here, this can happen, since we're pushing to a circular buffer, which 
    //  may not have room for new entries.  In such cases, we trust that the consumer thread will make space for us, and
    //  so we may just need to wait a little bit.
    while (!m_drawCallStateQueue.push(std::move(m_activeDrawCallState))) {
      Sleep(0);
    }
  }

  Future<SkinningData> D3D9Rtx::processSkinning(const RasterGeometry& geoData) {
    ScopedCpuProfileZone();

    static const auto kEmptySkinningFuture = Future<SkinningData>();

    if (m_parent->UseProgrammableVS()) {
      return kEmptySkinningFuture;
    }

    // Some games set vertex blend without enough data to actually do the blending, handle that logic below.

    const bool hasBlendWeight = geoData.blendWeightBuffer.defined();
    const bool hasBlendIndices = d3d9State().vertexDecl != nullptr ? d3d9State().vertexDecl->TestFlag(D3D9VertexDeclFlag::HasBlendIndices) : false;
    const bool indexedVertexBlend = hasBlendIndices && d3d9State().renderStates[D3DRS_INDEXEDVERTEXBLENDENABLE];

    if (d3d9State().renderStates[D3DRS_VERTEXBLEND] == D3DVBF_DISABLE) {
      return kEmptySkinningFuture;
    }

    if (d3d9State().renderStates[D3DRS_VERTEXBLEND] != D3DVBF_0WEIGHTS) {
      if (!hasBlendWeight) {
        return kEmptySkinningFuture;
      }
    } else if (!indexedVertexBlend) {
      return kEmptySkinningFuture;
    }

    // We actually have skinning data now, process it!

    uint32_t numBonesPerVertex = 0;
    switch (d3d9State().renderStates[D3DRS_VERTEXBLEND]) {
    case D3DVBF_0WEIGHTS: numBonesPerVertex = 1; break;
    case D3DVBF_1WEIGHTS: numBonesPerVertex = 2; break;
    case D3DVBF_2WEIGHTS: numBonesPerVertex = 3; break;
    case D3DVBF_3WEIGHTS: numBonesPerVertex = 4; break;
    }

    const uint32_t vertexCount = geoData.vertexCount;

    HashQuery blendIndices;
    // Analyze the vertex data and find the min and max bone indices used in this mesh.
    // The min index is used to detect a case when vertex blend is enabled but there is just one bone used in the mesh,
    // so we can drop the skinning pass. That is processed in RtxContext::commitGeometryToRT(...)
    if (indexedVertexBlend && geoData.blendIndicesBuffer.defined()) {
      auto& buffer = geoData.blendIndicesBuffer;

      blendIndices.pBase = (uint8_t*) buffer.mapPtr(buffer.offsetFromSlice());
      blendIndices.elementSize = imageFormatInfo(buffer.vertexFormat())->elementSize;
      blendIndices.stride = buffer.stride();
      blendIndices.size = blendIndices.stride * vertexCount;
      blendIndices.ref = buffer.buffer().ptr();

      // Acquire prevents the staging allocator from re-using this memory
      blendIndices.ref->acquire(DxvkAccess::Read);
      // Make sure we hold on to this reference while the hashing is in flight
      blendIndices.ref->incRef();
    } else {
      blendIndices.ref = nullptr;
    }

    // Copy bones up to the max bone we have registered so far.
    const uint32_t maxBone = m_maxBone > 0 ? m_maxBone : 255;
    const uint32_t startBoneTransform = GetTransformIndex(D3DTS_WORLDMATRIX(0));

    const uint32_t nMat = maxBone + 1;
    const Matrix4* const boneMatrices = m_stagedBones.stageBones(
        d3d9State().transforms.data() + startBoneTransform, nMat);

    return m_pGeometryWorkers->Schedule([boneMatrices, blendIndices, numBonesPerVertex, vertexCount]()->SkinningData {
      ScopedCpuProfileZone();
      uint32_t numBones = numBonesPerVertex;

      int minBoneIndex = 0;
      if (blendIndices.ref) {
        const uint8_t* pBlendIndices = blendIndices.pBase;
        // Find out how many bone indices are specified for each vertex.
        // This is needed to find out the min bone index and ignore the padding zeroes.
        int maxBoneIndex = -1;
        if (!getMinMaxBoneIndices(pBlendIndices, blendIndices.stride, vertexCount, numBonesPerVertex, minBoneIndex, maxBoneIndex)) {
          minBoneIndex = 0;
          maxBoneIndex = 0;
        }
        numBones = maxBoneIndex + 1;

        // Release this memory back to the staging allocator
        blendIndices.ref->release(DxvkAccess::Read);
        blendIndices.ref->decRef();
      }

      // Pass bone data to RT back-end

      SkinningData skinningData;
      skinningData.pBoneMatrices.reserve(numBones);

      for (uint32_t n = 0; n < numBones; n++) {
        skinningData.pBoneMatrices.push_back(boneMatrices[n]);
      }

      skinningData.minBoneIndex = minBoneIndex;
      skinningData.numBones = numBones;
      skinningData.numBonesPerVertex = numBonesPerVertex;
      skinningData.computeHash(); // Computes the hash and stores it in the skinningData itself

      return skinningData;
    });
  }

  template<bool FixedFunction>
  bool D3D9Rtx::processTextures() {
    ScopedCpuProfileZone();
    // We don't support full legacy materials in fixed function mode yet..
    // This implementation finds the most relevant textures bound from the
    // following criteria:
    //   - Texture actually bound (and used) by stage
    //   - First N textures bound to a specific texcoord index
    //   - Prefer lowest texcoord index
    // In non-fixed function (shaders), take the first N textures.

    // Used args for a given operation.
    auto ArgsMask = [](DWORD Op) {
      switch (Op) {
      case D3DTOP_DISABLE:
        return 0b000u; // No Args
      case D3DTOP_SELECTARG1:
      case D3DTOP_PREMODULATE:
        return 0b010u; // Arg 1
      case D3DTOP_SELECTARG2:
        return 0b100u; // Arg 2
      case D3DTOP_MULTIPLYADD:
      case D3DTOP_LERP:
        return 0b111u; // Arg 0, 1, 2
      default:
        return 0b110u; // Arg 1, 2
      }
    };

    // Currently we only support 2 textures
    constexpr uint32_t NumTexcoordBins = FixedFunction ? (D3DDP_MAXTEXCOORD * LegacyMaterialData::kMaxSupportedTextures) : LegacyMaterialData::kMaxSupportedTextures;

    bool useStageTextureFactorBlending = true;
    bool useMultipleStageTextureFactorBlending = false;

    // Build a mapping of texcoord indices to stage
    const uint8_t kInvalidStage = 0xFF;
    uint8_t texcoordIndexToStage[NumTexcoordBins];
    if constexpr (FixedFunction) {
      memset(&texcoordIndexToStage[0], kInvalidStage, sizeof(texcoordIndexToStage));
      for (uint32_t stage = 0; stage < caps::TextureStageCount; stage++) {
        auto isTextureFactorBlendingEnabled = [&](const auto& tss) -> bool {
          const auto colorOp = tss[DXVK_TSS_COLOROP];
          const auto alphaOp = tss[DXVK_TSS_ALPHAOP];

          if (colorOp == D3DTOP_DISABLE && alphaOp == D3DTOP_DISABLE)
            return false;

          const auto a1c = tss[DXVK_TSS_COLORARG1] & D3DTA_SELECTMASK;
          const auto a2c = tss[DXVK_TSS_COLORARG2] & D3DTA_SELECTMASK;
          const auto a1a = tss[DXVK_TSS_ALPHAARG1] & D3DTA_SELECTMASK;
          const auto a2a = tss[DXVK_TSS_ALPHAARG2] & D3DTA_SELECTMASK;

          // If previous stage wrote to TEMP the prior result source this stage
          // should read is D3DTA_TEMP otherwise its D3DTA_CURRENT.
          DWORD prevResultSel = D3DTA_CURRENT;
          if (stage != 0) {
            const auto& prev = d3d9State().textureStages[stage - 1];
            const auto resultArg = prev[DXVK_TSS_RESULTARG] & D3DTA_SELECTMASK;
            prevResultSel = (resultArg == D3DTA_TEMP) ? D3DTA_TEMP : D3DTA_CURRENT;
          }

          auto isModulate = [](DWORD op) {
            return op == D3DTOP_MODULATE || op == D3DTOP_MODULATE2X || op == D3DTOP_MODULATE4X;
          };

          const bool colorMul =
            isModulate(colorOp) &&
            ((a1c == D3DTA_TFACTOR && a2c == prevResultSel) ||
             (a2c == D3DTA_TFACTOR && a1c == prevResultSel));

          const bool alphaMul =
            isModulate(alphaOp) &&
            ((a1a == D3DTA_TFACTOR && a2a == prevResultSel) ||
             (a2a == D3DTA_TFACTOR && a1a == prevResultSel));

          return colorMul || alphaMul;
        };

        // Support texture factor blending besides the first stage. Currently, we only support 1 additional stage tFactor blending.
        // Note: If the tFactor is disabled for current texture (useStageTextureFactorBlending) then we should ignore the multiple stage tFactor blendings.
        bool isCurrentStageTextureFactorBlendingEnabled = false;
        if (useStageTextureFactorBlending &&
            m_frameOptions.enableMultiStageTextureFactorBlending &&
            stage != 0 &&
            isTextureFactorBlendingEnabled(d3d9State().textureStages[stage])) {
          isCurrentStageTextureFactorBlendingEnabled = true;
          useMultipleStageTextureFactorBlending = true;
        }

        if (d3d9State().textures[stage] == nullptr)
          continue;

        const auto& data = d3d9State().textureStages[stage];

        // Subsequent stages do not occur if this is true.
        if (data[DXVK_TSS_COLOROP] == D3DTOP_DISABLE)
          break;

        const std::uint32_t argsMask = ArgsMask(data[DXVK_TSS_COLOROP]) | ArgsMask(data[DXVK_TSS_ALPHAOP]);
        const auto firstTexMask  = ((data[DXVK_TSS_COLORARG0] & D3DTA_SELECTMASK) == D3DTA_TEXTURE) || ((data[DXVK_TSS_ALPHAARG0] & D3DTA_SELECTMASK) == D3DTA_TEXTURE);
        const auto secondTexMask = ((data[DXVK_TSS_COLORARG1] & D3DTA_SELECTMASK) == D3DTA_TEXTURE) || ((data[DXVK_TSS_ALPHAARG1] & D3DTA_SELECTMASK) == D3DTA_TEXTURE);
        const auto thirdTexMask  = ((data[DXVK_TSS_COLORARG2] & D3DTA_SELECTMASK) == D3DTA_TEXTURE) || ((data[DXVK_TSS_ALPHAARG2] & D3DTA_SELECTMASK) == D3DTA_TEXTURE);
        const std::uint32_t texMask =
          (firstTexMask  ? 0b001 : 0) |
          (secondTexMask ? 0b010 : 0) |
          (thirdTexMask  ? 0b100 : 0);

        // Is texture used?
        if ((argsMask & texMask) == 0)
          continue;

        D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[stage]);

        // Remix can only handle 2D textures - no volumes.
        if (texture->GetType() != D3DRTYPE_TEXTURE && (!m_frameOptions.allowCubemaps || texture->GetType() != D3DRTYPE_CUBETEXTURE)) {
          continue;
        }

        const XXH64_hash_t texHash = texture->GetSampleView(true)->image()->getHash();

        // Currently we only support regular textures, skip lightmaps.
        if (isLightmapTexture(texHash)) {
          continue;
        }

        // Allow for two stage candidates per texcoord index
        const uint32_t texcoordIndex = data[DXVK_TSS_TEXCOORDINDEX] & 0b111;
        const uint32_t candidateIndex = texcoordIndex * LegacyMaterialData::kMaxSupportedTextures;
        const uint32_t subIndex = (texcoordIndexToStage[candidateIndex] == kInvalidStage) ? 0 : 1;

        // Don't override if candidate exists
        if (texcoordIndexToStage[candidateIndex + subIndex] == kInvalidStage)
          texcoordIndexToStage[candidateIndex + subIndex] = stage;

        // Check if texture factor blending is enabled for the first stage
        if (useStageTextureFactorBlending && stage == 0) {
          isCurrentStageTextureFactorBlendingEnabled = isTextureFactorBlendingEnabled(d3d9State().textureStages[stage]);
        }

        // Check if texture factor blending is enabled
        if (isCurrentStageTextureFactorBlendingEnabled &&
            lookupHash(*m_frameOptions.ignoreBakedLightingTextures, texHash)) {
          useStageTextureFactorBlending = false;
          useMultipleStageTextureFactorBlending = false;
        }
      }
    }

    // Find the ideal textures for raytracing, initialize the data to invalid (out of range) to unbind unused textures
    uint32_t firstStage = 0;
    m_activeDrawCallState.materialData.colorTextureIsSrgb = false;
    m_texcoordCompU = 0;
    m_texcoordCompV = 1;
    m_iaTexcoordIndex = 0;
    const D3D9CommonShader* inferredPs = nullptr;
    XXH64_hash_t inferredPsHash = 0;
    PsSamplerTexcoordEntry* inferredPsEntry = nullptr;
    const Ue3VertexFactoryType vfType = m_currentUe3VertexFactory;
    const bool isUe3GpuSkinVF = vfType == Ue3VertexFactoryType::GPUSkin || vfType == Ue3VertexFactoryType::GPUSkinMorph;
    const bool isUe3TerrainVF = vfType == Ue3VertexFactoryType::Terrain || vfType == Ue3VertexFactoryType::TerrainMorph;
    const bool isUe3ParticleVF =
      vfType == Ue3VertexFactoryType::Particle ||
      vfType == Ue3VertexFactoryType::ParticleBeamTrail ||
      vfType == Ue3VertexFactoryType::LensFlare;
    // Instanced mesh particles compile the same FoliageVertexFactory.usf and pack their UVs the
    // same way, so they follow foliage rather than the camera-facing particle factories.
    const bool isUe3FoliageVF =
      vfType == Ue3VertexFactoryType::Foliage ||
      vfType == Ue3VertexFactoryType::ParticleInstancedMesh;
    const bool isUe3SpeedTreeVF = vfType == Ue3VertexFactoryType::SpeedTree;
    const bool isUe3LocalDecalVF = vfType == Ue3VertexFactoryType::LocalDecal;
    const bool isUe3MorphVF = vfType == Ue3VertexFactoryType::GPUSkinMorph;

    const bool likelyGpuSkinnedMesh = isUe3GpuSkinVF || [&]() {
      if (d3d9State().vertexDecl.ptr() == nullptr)
        return false;

      for (const auto& element : d3d9State().vertexDecl->GetElements()) {
        if (element.Usage == D3DDECLUSAGE_BLENDWEIGHT ||
            element.Usage == D3DDECLUSAGE_BLENDINDICES)
          return true;
      }
      return false;
    }();
    const Ue3VsShaderCtabInfo* ue3VsHints =
      (m_parent->UseProgrammableVS() &&
       d3d9State().vertexShader.ptr() != nullptr &&
       m_currentUe3CtabInfo.has_value())
        ? &(*m_currentUe3CtabInfo)
        : nullptr;
    const bool likelyUe3DecalUvSpace =
      isUe3LocalDecalVF ||
      (ue3VsHints != nullptr &&
       (ue3VsHints->hasDecalTransform ||
        ue3VsHints->hasDecalLocation ||
        ue3VsHints->hasDecalOffset));
    const bool likelyUe3TerrainUvSpace =
      isUe3TerrainVF ||
      (ue3VsHints != nullptr &&
       (ue3VsHints->hasLightMapCoordinateScaleBias ||
        ue3VsHints->hasShadowCoordinateScaleBias));
    const bool likelyUe3BillboardUvSpace =
      isUe3ParticleVF ||
      isUe3SpeedTreeVF ||
      (ue3VsHints != nullptr &&
       (ue3VsHints->hasTextureCoordinateScaleBias ||
        ue3VsHints->hasViewToLocal ||
        ue3VsHints->hasWindMatrices));
    const bool likelyUe3FlexiblePackedUvPath =
      !likelyGpuSkinnedMesh &&
      (likelyUe3DecalUvSpace || likelyUe3TerrainUvSpace || likelyUe3BillboardUvSpace || isUe3FoliageVF);
    const bool likelyPackedUvConventions = likelyGpuSkinnedMesh || likelyUe3FlexiblePackedUvPath;
    auto resolveInferredSamplerOffset = [&](const PsSamplerTexcoordEntry* entry, const uint32_t stage, float& outU, float& outV) -> bool {
      outU = 0.0f;
      outV = 0.0f;

      if (entry == nullptr || stage >= caps::MaxTexturesPS)
        return false;

      if (entry->samplerOffsetImmediateValid[stage]) {
        outU = entry->samplerOffsetImmediateU[stage];
        outV = entry->samplerOffsetImmediateV[stage];
        return std::isfinite(outU) && std::isfinite(outV);
      }

      const int16_t offsetConstReg = entry->samplerOffsetConstReg[stage];
      if (offsetConstReg >= 0 && uint32_t(offsetConstReg) < caps::MaxFloatConstantsPS) {
        const Vector4& offsetConst = d3d9State().psConsts.fConsts[uint32_t(offsetConstReg)];
        const uint32_t compU = entry->samplerOffsetConstCompU[stage] & 0x3;
        const uint32_t compV = entry->samplerOffsetConstCompV[stage] & 0x3;
        outU = offsetConst[compU] * entry->samplerOffsetFactorU[stage];
        outV = offsetConst[compV] * entry->samplerOffsetFactorV[stage];
        return std::isfinite(outU) && std::isfinite(outV);
      }

      return false;
    };

    auto hasNonZeroInferredSamplerOffset = [&](const PsSamplerTexcoordEntry* entry, const uint32_t stage) -> bool {
      float uOffset = 0.0f;
      float vOffset = 0.0f;
      if (!resolveInferredSamplerOffset(entry, stage, uOffset, vOffset))
        return false;

      constexpr float kOffsetEps = 1e-5f;
      return std::abs(uOffset) > kOffsetEps || std::abs(vOffset) > kOffsetEps;
    };

    auto getOrInitPsSamplerTexcoordEntry = [&](const D3D9CommonShader* ps, XXH64_hash_t& outHash) -> PsSamplerTexcoordEntry* {
      if (ps == nullptr)
        return nullptr;

      outHash = ps->GetBytecodeHash();
      auto& entry = m_psSamplerTexcoordCache[outHash];
      if (!entry.initialized) {
        entry.initialized = true;
        analyzePsSamplerUvOrigins(ps, outHash, entry.samplerUvOrigin);
        entry.samplerToTexcoord.fill(-1);
        entry.samplerCoordCompValid.fill(0);
        entry.samplerCoordCompU.fill(0);
        entry.samplerCoordCompV.fill(1);
        entry.samplerSemanticFlags.fill(0);
        entry.samplerExpressionFlags.fill(0);
        entry.samplerSampleCount.fill(0);
        entry.samplerScaleConstReg.fill(-1);
        entry.samplerScaleConstCompU.fill(0);
        entry.samplerScaleConstCompV.fill(1);
        entry.samplerScaleFactorU.fill(1.0f);
        entry.samplerScaleFactorV.fill(1.0f);
        entry.samplerScaleImmediateValid.fill(0);
        entry.samplerScaleImmediateU.fill(1.0f);
        entry.samplerScaleImmediateV.fill(1.0f);
        entry.samplerOffsetConstReg.fill(-1);
        entry.samplerOffsetConstCompU.fill(0);
        entry.samplerOffsetConstCompV.fill(1);
        entry.samplerOffsetFactorU.fill(1.0f);
        entry.samplerOffsetFactorV.fill(1.0f);
        entry.samplerOffsetImmediateValid.fill(0);
        entry.samplerOffsetImmediateU.fill(0.0f);
        entry.samplerOffsetImmediateV.fill(0.0f);
        for (uint32_t s = 0; s < caps::MaxTexturesPS; s++) {
          const PsSamplerTexcoordInference inferred = inferPixelShaderTexcoordForSampler(ps, s);
          entry.samplerToTexcoord[s] = int8_t(inferred.texcoord);
          entry.samplerCoordCompValid[s] = inferred.coordCompValid ? 1u : 0u;
          entry.samplerCoordCompU[s] = inferred.coordCompU;
          entry.samplerCoordCompV[s] = inferred.coordCompV;
          entry.samplerSemanticFlags[s] = inferred.semanticFlags;
          entry.samplerExpressionFlags[s] = inferred.expressionFlags;
          entry.samplerSampleCount[s] = inferred.sampleCount;
          entry.samplerScaleConstReg[s] = int16_t(inferred.scaleConstReg);
          entry.samplerScaleConstCompU[s] = inferred.scaleConstCompU;
          entry.samplerScaleConstCompV[s] = inferred.scaleConstCompV;
          entry.samplerScaleFactorU[s] = inferred.scaleFactorU;
          entry.samplerScaleFactorV[s] = inferred.scaleFactorV;
          entry.samplerScaleImmediateValid[s] = inferred.scaleImmediateValid ? 1u : 0u;
          entry.samplerScaleImmediateU[s] = inferred.scaleImmediateU;
          entry.samplerScaleImmediateV[s] = inferred.scaleImmediateV;
          entry.samplerOffsetConstReg[s] = int16_t(inferred.offsetConstReg);
          entry.samplerOffsetConstCompU[s] = inferred.offsetConstCompU;
          entry.samplerOffsetConstCompV[s] = inferred.offsetConstCompV;
          entry.samplerOffsetFactorU[s] = inferred.offsetFactorU;
          entry.samplerOffsetFactorV[s] = inferred.offsetFactorV;
          entry.samplerOffsetImmediateValid[s] = inferred.offsetImmediateValid ? 1u : 0u;
          entry.samplerOffsetImmediateU[s] = inferred.offsetImmediateU;
          entry.samplerOffsetImmediateV[s] = inferred.offsetImmediateV;
          if ((inferred.semanticFlags & kPsSamplerSemanticLightmap) != 0)
            entry.lightmapSamplerMask |= (1u << s);
        }
      }

      return &entry;
    };

    if constexpr (!FixedFunction) {
      if ((m_frameOptions.shaderPathTexcoordIndexFromPixelShader || m_frameOptions.ue3EngineMode) && d3d9State().pixelShader.ptr() != nullptr) {
        inferredPs = d3d9State().pixelShader->GetCommonShader();
        inferredPsEntry = getOrInitPsSamplerTexcoordEntry(inferredPs, inferredPsHash);
      }
    }

    auto getRemixSampleView = [](D3D9CommonTexture* texture, const bool srgb) -> Rc<DxvkImageView> {
      if (texture == nullptr)
        return nullptr;
      // legacy Remix albedo path is 2D oriented so for cubemap albedo slots,
      // bind a face view to avoid falling back to white placeholders
      if (texture->GetType() == D3DRTYPE_CUBETEXTURE) {
        return texture->CreateView(0, 0, VK_IMAGE_USAGE_SAMPLED_BIT, srgb);
      }
      return texture->GetSampleView(srgb);
    };
    bool selectedUe3MovieTexture = false;

    if constexpr (FixedFunction) {
      uint32_t textureID = 0;
      for (uint32_t idx = 0; idx < NumTexcoordBins && textureID < LegacyMaterialData::kMaxSupportedTextures; idx++) {
        const uint8_t stage = texcoordIndexToStage[idx];
        if (stage == kInvalidStage || d3d9State().textures[stage] == nullptr)
          continue;

        D3D9CommonTexture* pTexInfo = GetCommonTexture(d3d9State().textures[stage]);
        assert(pTexInfo != nullptr);
        const XXH64_hash_t texHash = pTexInfo->GetImage()->getHash();
        const XXH64_hash_t texDescHash =
          pTexInfo->GetImage() != nullptr
            ? pTexInfo->GetImage()->getDescriptorHash()
            : kEmptyHash;

        if (texHash == kEmptyHash)
          continue;

        if (textureID == 0)
          firstStage = stage;

        D3D9SamplerKey key = m_parent->CreateSamplerKey(stage);
        XXH64_hash_t samplerHash = D3D9SamplerKeyHash{}(key);

        Rc<DxvkSampler> sampler;
        auto samplerIt = m_samplerCache.find(samplerHash);
        if (samplerIt != m_samplerCache.end()) {
          sampler = samplerIt->second;
        } else {
          const auto samplerInfo = m_parent->DecodeSamplerKey(key);
          sampler = m_parent->GetDXVKDevice()->createSampler(samplerInfo);
          m_samplerCache.insert(std::make_pair(samplerHash, sampler));
        }

        // Cache the slot we want to bind
        const bool srgb = d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1;
        Rc<DxvkImageView> sampleView = getRemixSampleView(pTexInfo, srgb);
        if (sampleView == nullptr)
          continue;
        m_activeDrawCallState.materialData.colorTextures[textureID] = TextureRef(sampleView);
        m_activeDrawCallState.materialData.samplers[textureID] = sampler;
        selectedUe3MovieTexture |= isUe3MovieTextureDescHash(texDescHash);
        if (textureID == 0)
          m_activeDrawCallState.materialData.colorTextureIsSrgb = srgb;

        auto shaderSampler = RemapStateSamplerShader(stage);
        m_activeDrawCallState.materialData.colorTextureSlot[textureID] = computeResourceSlotId(shaderSampler.first, DxsoBindingType::Image, uint32_t(shaderSampler.second));

        ++textureID;
      }
    } else {
      // for the shader path we pick the most relevant textures actually used by the pixel shader
      // we prefer sRGB textures for the first slot since normal maps/masks are usually sampled in linear space
      uint8_t chosenStages[LegacyMaterialData::kMaxSupportedTextures] = { kInvalidStage, kInvalidStage };
      int64_t chosenScore[LegacyMaterialData::kMaxSupportedTextures] = { std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::min() };
      uint8_t strictCubemapFallbackStage = kInvalidStage;
      int32_t strictCubemapFallbackScore = std::numeric_limits<int32_t>::min();

      const uint32_t usedSamplerMask = m_parent->m_psShaderMasks.samplerMask;
      // UE3 lightmaps are removed from the draw's texture set before anything reads it. A
      // score penalty is not enough: DirectionalLightmaps=True binds three coefficient
      // samplers where False binds one, so leaving them in changes the candidate pool, the
      // selection cache key and its area sum, and the per-texture material spread - every
      // one of which can flip the albedo pick between the two settings. Remix owns lighting,
      // so the coefficients carry nothing a raytraced surface wants.
      const uint32_t lightmapStageMask =
        (m_frameOptions.ue3EngineMode && inferredPsEntry != nullptr) ? inferredPsEntry->lightmapSamplerMask : 0u;
      const uint32_t usedTextureMask = m_parent->m_activeTextures & usedSamplerMask & ~lightmapStageMask;

      // Publish what was bound at those samplers so the texture paths that cannot see a pixel
      // shader - hash preservation on CPU writes, the terrain baker's stage filter, the texture
      // picker - recognise the same images. First sight per hash only; the session set is
      // append-only so the repeat check stays local.
      if (m_frameOptions.ue3AutoDetectLightmapTextures) {
        for (const uint32_t stage : bit::BitMask(lightmapStageMask & m_parent->m_activeTextures)) {
          if (stage >= SamplerCount || d3d9State().textures[stage] == nullptr)
            continue;
          D3D9CommonTexture* lightmap = GetCommonTexture(d3d9State().textures[stage]);
          if (lightmap == nullptr || lightmap->GetImage() == nullptr)
            continue;
          const XXH64_hash_t lightmapHash = lightmap->GetImage()->getHash();
          if (lightmapHash != kEmptyHash && m_ue3SeenLightmapTextures.insert(lightmapHash).second)
            registerAutoDetectedLightmapTexture(lightmapHash);
        }
      }

      // Deterministic diffuse selection: the scoring below reads live shader constant
      // values (hasNonZeroInferredSamplerOffset), which UE3 rewrites per draw for
      // panner/time/view expressions - flipping large score terms and with them the
      // chosen albedo stage between frames or with camera position. Caching the first
      // decision per (PS, texture set, sRGB, vertex factory) key pins the pick.
      XXH64_hash_t selectionCacheKey = kEmptyHash;
      bool selectionCacheUsable = false;
      bool selectionFromCache = false;
      uint64_t selectionBoundAreaSum = 0;
      bool auditingPinnedSelection = false;
      uint8_t auditedPinnedStages[2] = { kInvalidStage, kInvalidStage };
      uint8_t auditedPinnedCubemapStage = kInvalidStage;
      if ((m_frameOptions.ue3StableDiffuseSelection || m_frameOptions.ue3EngineMode) &&
          inferredPsEntry != nullptr && inferredPsHash != kEmptyHash) {
        if (!m_ue3DiffuseSelectionLoaded)
          loadUe3DiffuseSelectionCache();
        // scoring consults the user-taggable lightmap/never-albedo/preferred-albedo sets; drop cached
        // decisions when those sets change so texture tagging takes effect immediately
        const size_t lightmapSetSize = m_frameOptions.lightmapTextures->size();
        const size_t neverAlbedoSetSize = m_frameOptions.neverAlbedoTextures->size();
        const size_t preferredAlbedoSetSize = m_frameOptions.preferredAlbedoTextures->size();
        if (lightmapSetSize != m_ue3DiffuseSelectionLightmapSetSize ||
            neverAlbedoSetSize != m_ue3DiffuseSelectionNeverAlbedoSetSize ||
            preferredAlbedoSetSize != m_ue3DiffuseSelectionPreferredAlbedoSetSize) {
          m_ue3DiffuseSelectionCache.clear();
          // Tagging invalidates every stored decision, so the persisted set has to shrink with
          // the in-memory one rather than keep serving picks the tags have just overruled.
          m_ue3DiffuseSelectionDirty = true;
          // re-log re-scored selections so tag effects are visible in ue3LogAlbedoSelection output
          m_loggedAlbedoSelections.clear();
          m_ue3DiffuseSelectionLightmapSetSize = lightmapSetSize;
          m_ue3DiffuseSelectionNeverAlbedoSetSize = neverAlbedoSetSize;
          m_ue3DiffuseSelectionPreferredAlbedoSetSize = preferredAlbedoSetSize;
        }

        struct SelectionKeyTuple {
          uint32_t stage;
          uint32_t srgb;
          XXH64_hash_t texHash;
        };
        static_assert(sizeof(SelectionKeyTuple) == 16, "SelectionKeyTuple must have no implicit padding (it is hashed by memory).");
        // one tuple per bound texture plus a trailing vertex-factory context tuple,
        // whose out-of-range stage index cannot collide with a real texture tuple
        std::array<SelectionKeyTuple, SamplerCount + 1> keyTuples;
        uint32_t keyTupleCount = 0;
        const BoundTextureSnapshot& boundTextures = ensureBoundTextureSnapshot();
        for (const uint32_t stage : bit::BitMask(usedTextureMask & boundTextures.mask)) {
          const BoundTextureSnapshotEntry& entry = boundTextures.entries[stage];
          if (!entry.hasImage)
            continue;
          keyTuples[keyTupleCount++] = {
            stage,
            d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1u,
            entry.imageHash,
          };

          const auto* desc = entry.texture->Desc();
          if (desc != nullptr) {
            selectionBoundAreaSum += uint64_t(desc->Width) * uint64_t(desc->Height);
          }
        }
        // score-relevant vertex factory context (packed UV biases differ per factory)
        const uint32_t vfContext =
          uint32_t(vfType) |
          (likelyGpuSkinnedMesh ? 1u << 8 : 0u) |
          (likelyUe3FlexiblePackedUvPath ? 1u << 9 : 0u) |
          (likelyPackedUvConventions ? 1u << 10 : 0u);
        keyTuples[keyTupleCount++] = { uint32_t(SamplerCount), vfContext, kEmptyHash };
        selectionCacheKey = XXH3_64bits_withSeed(keyTuples.data(), keyTupleCount * sizeof(SelectionKeyTuple), inferredPsHash);
        selectionCacheUsable = true;

        // Streaming-stable hashes give every mip variant of a material the same key, so a
        // decision scored against streamed-down mips (smaller bound texel area) is only
        // authoritative for equal or smaller sets; a larger set re-scores and supersedes it.
        const auto cachedSelection = m_ue3DiffuseSelectionCache.find(selectionCacheKey);
        if (cachedSelection != m_ue3DiffuseSelectionCache.end()) {
          if (selectionBoundAreaSum <= cachedSelection->second.decisionAreaSum) {
            chosenStages[0] = cachedSelection->second.chosenStages[0];
            chosenStages[1] = cachedSelection->second.chosenStages[1];
            strictCubemapFallbackStage = cachedSelection->second.cubemapFallbackStage;
            selectionFromCache = true;

            // Drop cached picks that landed on a refused render target so a real material
            // sampler can re-compete. Reusing the scoring-loop predicate keeps a legitimately
            // tagged pick from being discarded and re-scored every frame.
            const auto cachedPickRefused = [&](const uint8_t stage) {
              if (stage == kInvalidStage || stage >= SamplerCount ||
                  d3d9State().textures[stage] == nullptr) {
                return false;
              }
              return isUe3RenderTargetRefusedAsAlbedo(
                GetCommonTexture(d3d9State().textures[stage]), stage, inferredPsEntry);
            };

            if (m_frameOptions.ue3EngineMode &&
                (cachedPickRefused(chosenStages[0]) || cachedPickRefused(chosenStages[1]))) {
              chosenStages[0] = kInvalidStage;
              chosenStages[1] = kInvalidStage;
              strictCubemapFallbackStage = kInvalidStage;
              selectionFromCache = false;
              m_ue3DiffuseSelectionCache.erase(cachedSelection);
              m_loggedAlbedoSelections.erase(selectionCacheKey);
            } else if (cachedSelection->second.fromDisk && m_ue3DiffuseSelectionAuditsRemaining > 0) {
              // Score this one anyway and compare. The pinned pick still wins - re-scoring at an
              // arbitrary moment is exactly the transient the pin exists to avoid - so the audit
              // only reports.
              --m_ue3DiffuseSelectionAuditsRemaining;
              auditingPinnedSelection = true;
              auditedPinnedStages[0] = chosenStages[0];
              auditedPinnedStages[1] = chosenStages[1];
              auditedPinnedCubemapStage = strictCubemapFallbackStage;
              chosenStages[0] = kInvalidStage;
              chosenStages[1] = kInvalidStage;
              strictCubemapFallbackStage = kInvalidStage;
              selectionFromCache = false;
            }
          } else {
            // superseding re-score: let ue3LogAlbedoSelection dump the authoritative decision
            m_loggedAlbedoSelections.erase(selectionCacheKey);
          }
        }
      }

      // per-stage score breakdown for rtx.d3d9.ue3LogAlbedoSelection, dumped once per selection key.
      // Audited draws are excluded: their scoring is discarded, so dumping it would report a
      // decision the draw did not use.
      const bool logAlbedoSelection =
        m_frameOptions.ue3LogAlbedoSelection &&
        selectionCacheUsable &&
        !selectionFromCache &&
        !auditingPinnedSelection &&
        m_loggedAlbedoSelections.find(selectionCacheKey) == m_loggedAlbedoSelections.end();
      std::string albedoSelectionLog;

      // Size credit in mip steps: one doubling of texel area is worth a fixed amount, so a
      // texture that has streamed one mip further gains a fixed, small advantage instead of
      // one proportional to the texel count.
      auto albedoAreaScore = [](const uint64_t area) -> int64_t {
        constexpr int64_t kScorePerMipStep = 40'000;
        constexpr int64_t kMaxMipSteps = 24;  // 4096x4096
        int64_t steps = 0;
        for (uint64_t remaining = area >> 1; remaining != 0 && steps < kMaxMipSteps; remaining >>= 1)
          ++steps;
        return steps * kScorePerMipStep;
      };

      // The unprovable-origin penalty only means something where the resolved UV transform is
      // actually consumed; the gate mirrors the one guarding that resolution below.
      const bool uvOriginPenaltyActive =
        (m_frameOptions.shaderPathTexcoordIndexFromPixelShader || m_frameOptions.ue3EngineMode) &&
        inferredPsEntry != nullptr;

      const uint32_t scoringTextureMask = selectionFromCache ? 0u : usedTextureMask;
      for (uint32_t stage : bit::BitMask(scoringTextureMask)) {
        if (stage >= SamplerCount || d3d9State().textures[stage] == nullptr)
          continue;

        D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[stage]);
        if (!texture)
          continue;

        const D3DRESOURCETYPE textureType = texture->GetType();
        const bool is2DTexture = textureType == D3DRTYPE_TEXTURE;
        const bool isCubeTexture = textureType == D3DRTYPE_CUBETEXTURE;
        if (!is2DTexture && !isCubeTexture)
          continue;

        const XXH64_hash_t texHash = texture->GetSampleView(false)->image()->getHash();
        if (isLightmapTexture(texHash))
          continue;
        const bool isNeverAlbedo = lookupHash(*m_frameOptions.neverAlbedoTextures, texHash);
        const bool isPreferredAlbedo = lookupHash(*m_frameOptions.preferredAlbedoTextures, texHash);

        // material spread: distinct pixel shaders sampling this texture. Identity albedos stay
        // at 1-2 (material instances share their parent's bytecode); shared library assets -
        // detail patterns, grunge/dirt overlays, tint ramps - appear across many unrelated shaders.
        // Scoring reads the spread loaded from disk, never the live count: a statistic that grows
        // as the level streams in would give the same material a different answer depending on
        // when it was first drawn, and the pinned selections would have to be thrown away every
        // time one texture crossed the threshold. This session's discoveries score from the next.
        uint32_t materialSpread = 0;
        if (inferredPsHash != kEmptyHash && texHash != kEmptyHash) {
          if (!m_ue3TextureSpreadLoaded)
            loadUe3TextureSpreadCache();
          Ue3TextureMaterialSpread& spread = m_ue3TextureMaterialSpread[texHash];
          bool psKnown = false;
          for (uint8_t i = 0; i < spread.count; i++) {
            if (spread.psHashes[i] == inferredPsHash) {
              psKnown = true;
              break;
            }
          }
          if (!psKnown && spread.count < spread.psHashes.size()) {
            spread.psHashes[spread.count++] = inferredPsHash;
            m_ue3TextureSpreadDirty = true;
          }
          materialSpread = spread.scoringCount;
        }

        const bool srgb = (d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1) != 0;
        const bool isRenderTarget = texture->IsRenderTarget();
        const XXH64_hash_t texDescHash =
          (texture->GetImage() != nullptr)
            ? texture->GetImage()->getDescriptorHash()
            : kEmptyHash;
        const auto* desc = texture->Desc();
        const uint64_t area = desc ? uint64_t(desc->Width) * uint64_t(desc->Height) : 0;
        uint16_t sampleCount = 0;
        bool hasInferredTexcoord = false;
        int8_t inferredTexcoordIdx = -1;
        uint8_t inferredSamplerSemanticFlags = 0;
        uint16_t inferredSamplerExpressionFlags = 0;
        bool inferredSamplerLooksEngineAuxiliary = false;
        bool inferredSamplerLooksMaterialTexture = false;
        bool inferredSamplerLooksLightmap = false;
        bool inferredSamplerLooksNonDiffuse = false;
        bool inferredSamplerLooksVideo = false;
        bool inferredSamplerLooksMovieTexture = false;
        bool inferredSamplerExprUvTransform = false;
        bool inferredSamplerExprUvOffset = false;
        bool inferredSamplerExprUvAnimated = false;
        bool inferredSamplerExprUvTimeDriven = false;
        bool inferredSamplerExprViewDependent = false;
        bool inferredSamplerExprMaskControl = false;
        bool inferredSamplerExprColorContribution = false;
        bool inferredSamplerExprBlendMath = false;
        bool inferredSamplerExprNormalDecode = false;
        bool inferredSamplerExprReachesOutputColor = false;
        bool inferredSamplerExprDiffuseAnchor = false;
        bool inferredUsesZw = false;
        bool inferredUsesWz = false;
        bool inferredUsesXy = false;
        bool inferredUsesPackedSecondary = false;
        bool hasNonZeroInferredOffset = false;
        // Whether this sampler's UV origin was proven back to a single interpolant. The
        // surface's texture transform is derived from the winning stage alone, so a stage
        // without a provable origin cannot carry one: winning the slot costs the surface
        // its whole UV transform and leaves the texture at raw interpolant scale.
        bool inferredSamplerUvOriginProvable = false;
        bool inferredSamplerReadsPrimaryUvPair = false;
        if (inferredPsEntry != nullptr && stage < caps::MaxTexturesPS) {
          const PsSamplerUvOrigin& stageUvOrigin = inferredPsEntry->samplerUvOrigin[stage];
          inferredSamplerUvOriginProvable = stageUvOrigin.originValid;
          // UE3 packs UV set 0 into an interpolant's .xy and set 1 into its .zw, so a proven
          // origin on the .xy pair names the mesh's primary channel.
          inferredSamplerReadsPrimaryUvPair =
            stageUvOrigin.originValid && stageUvOrigin.sitesAgree &&
            stageUvOrigin.compU == 0 && stageUvOrigin.compV == 1;
          sampleCount = inferredPsEntry->samplerSampleCount[stage];
          inferredTexcoordIdx = inferredPsEntry->samplerToTexcoord[stage];
          // GPUSkinMorphVF - TEXCOORD6/7 are morph delta streams, treat as noninferable UV
          if (isUe3MorphVF && inferredTexcoordIdx >= 6) {
            inferredTexcoordIdx = -1;
          }
          inferredSamplerSemanticFlags = inferredPsEntry->samplerSemanticFlags[stage];
          inferredSamplerExpressionFlags = inferredPsEntry->samplerExpressionFlags[stage];
          inferredSamplerLooksEngineAuxiliary = (inferredSamplerSemanticFlags & kPsSamplerSemanticEngineAuxiliary) != 0;
          inferredSamplerLooksMaterialTexture = (inferredSamplerSemanticFlags & kPsSamplerSemanticMaterialTexture) != 0;
          inferredSamplerLooksLightmap = (inferredSamplerSemanticFlags & kPsSamplerSemanticLightmap) != 0;
          inferredSamplerLooksNonDiffuse = (inferredSamplerSemanticFlags & kPsSamplerSemanticNonDiffuse) != 0;
          inferredSamplerLooksVideo = (inferredSamplerSemanticFlags & kPsSamplerSemanticVideo) != 0;
          inferredSamplerLooksMovieTexture = (inferredSamplerSemanticFlags & kPsSamplerSemanticMovieTexture) != 0;
          inferredSamplerExprUvTransform = (inferredSamplerExpressionFlags & kPsSamplerExprUvTransform) != 0;
          inferredSamplerExprUvOffset = (inferredSamplerExpressionFlags & kPsSamplerExprUvOffset) != 0;
          inferredSamplerExprUvAnimated = (inferredSamplerExpressionFlags & kPsSamplerExprUvAnimated) != 0;
          inferredSamplerExprUvTimeDriven = (inferredSamplerExpressionFlags & kPsSamplerExprUvTimeDriven) != 0;
          inferredSamplerExprViewDependent = (inferredSamplerExpressionFlags & kPsSamplerExprViewDependent) != 0;
          inferredSamplerExprMaskControl = (inferredSamplerExpressionFlags & kPsSamplerExprMaskControl) != 0;
          inferredSamplerExprColorContribution = (inferredSamplerExpressionFlags & kPsSamplerExprColorContribution) != 0;
          inferredSamplerExprBlendMath = (inferredSamplerExpressionFlags & kPsSamplerExprBlendMath) != 0;
          inferredSamplerExprNormalDecode = (inferredSamplerExpressionFlags & kPsSamplerExprNormalDecode) != 0;
          inferredSamplerExprReachesOutputColor = (inferredSamplerExpressionFlags & kPsSamplerExprReachesOutputColor) != 0;
          inferredSamplerExprDiffuseAnchor = (inferredSamplerExpressionFlags & kPsSamplerExprDiffuseAnchor) != 0;
          hasInferredTexcoord = inferredTexcoordIdx >= 0;
          inferredUsesZw =
            inferredPsEntry->samplerCoordCompValid[stage] != 0 &&
            inferredPsEntry->samplerCoordCompU[stage] == 2 &&
            inferredPsEntry->samplerCoordCompV[stage] == 3;
          inferredUsesWz =
            inferredPsEntry->samplerCoordCompValid[stage] != 0 &&
            inferredPsEntry->samplerCoordCompU[stage] == 3 &&
            inferredPsEntry->samplerCoordCompV[stage] == 2;
          inferredUsesXy =
            inferredPsEntry->samplerCoordCompValid[stage] != 0 &&
            inferredPsEntry->samplerCoordCompU[stage] == 0 &&
            inferredPsEntry->samplerCoordCompV[stage] == 1;
          inferredUsesPackedSecondary = inferredUsesWz || inferredUsesZw;
          hasNonZeroInferredOffset = hasNonZeroInferredSamplerOffset(inferredPsEntry, stage);
        }
        const bool isMovieTexture =
          isUe3MovieTextureDescHash(texDescHash) ||
          (isRenderTarget && inferredSamplerLooksMovieTexture);

        if (isCubeTexture && !m_frameOptions.allowCubemaps) {
          const bool looksMaterialCubemap =
            sampleCount > 0 &&
            (!isRenderTarget || isMovieTexture) &&
            !inferredSamplerLooksEngineAuxiliary &&
            !inferredSamplerLooksLightmap &&
            !inferredSamplerLooksNonDiffuse &&
            !inferredSamplerExprViewDependent &&
            !inferredSamplerExprMaskControl &&
            !isNeverAlbedo &&
            (inferredSamplerLooksMaterialTexture || inferredSamplerSemanticFlags == 0);

          if (looksMaterialCubemap) {
            int32_t cubeScore = 0;
            cubeScore += int32_t(std::min<uint16_t>(sampleCount, 16u)) * 64;
            cubeScore += hasInferredTexcoord ? 256 : 0;
            cubeScore -= hasNonZeroInferredOffset ? 96 : 0;
            cubeScore -= int32_t(stage);

            if (cubeScore > strictCubemapFallbackScore) {
              strictCubemapFallbackScore = cubeScore;
              strictCubemapFallbackStage = uint8_t(stage);
            }
          }

          continue;
        }

        // hashless textures (typically render targets) can never be bound as legacy
        // albedo; letting one win a slot starves the real diffuse and leaves the
        // surface white with no clickable texture hash
        if (texHash == kEmptyHash)
          continue;

        // Non-UE3 keeps the score penalties below instead.
        if (m_frameOptions.ue3EngineMode &&
            isUe3RenderTargetRefusedAsAlbedo(texture, stage, inferredPsEntry))
          continue;

        // effective area: a small texture tiled NxM times covers N*M times its pixel area
        // (UE3 TexCoord UTiling/VTiling folded into shader literals, or held in a scalar-parameter
        // constant resolved at decision time; ue3StableDiffuseSelection pins the resulting pick).
        // Restricted to genuinely tiny authored tiles (e.g. a 64x128 window): tiled detail/dirt/
        // tint overlays are usually 256x256+ and must not out-rank the albedo on size.
        constexpr uint64_t kTilingCreditMaxRawArea = 128ull * 128ull;
        constexpr uint64_t kTilingCreditMaxEffectiveArea = 2ull * 1024ull * 1024ull;
        uint64_t effectiveArea = area;
        if (inferredPsEntry != nullptr && stage < caps::MaxTexturesPS &&
            area > 0 && area <= kTilingCreditMaxRawArea) {
          float tilingU = 1.0f;
          float tilingV = 1.0f;
          bool tilingKnown = false;
          if (inferredPsEntry->samplerScaleImmediateValid[stage] != 0) {
            tilingU = std::abs(inferredPsEntry->samplerScaleImmediateU[stage]);
            tilingV = std::abs(inferredPsEntry->samplerScaleImmediateV[stage]);
            tilingKnown = true;
          } else if (inferredPsEntry->samplerScaleConstReg[stage] >= 0 &&
                     uint32_t(inferredPsEntry->samplerScaleConstReg[stage]) < caps::MaxFloatConstantsPS) {
            const Vector4& scaleConst =
              d3d9State().psConsts.fConsts[uint32_t(inferredPsEntry->samplerScaleConstReg[stage])];
            tilingU = std::abs(scaleConst[inferredPsEntry->samplerScaleConstCompU[stage] & 0x3u] *
                               inferredPsEntry->samplerScaleFactorU[stage]);
            tilingV = std::abs(scaleConst[inferredPsEntry->samplerScaleConstCompV[stage] & 0x3u] *
                               inferredPsEntry->samplerScaleFactorV[stage]);
            tilingKnown = std::isfinite(tilingU) && std::isfinite(tilingV);
          }
          if (tilingKnown) {
            const float tiles = std::min(std::max(tilingU * tilingV, 1.0f), 1024.0f);
            effectiveArea = std::min(uint64_t(double(area) * double(tiles)), kTilingCreditMaxEffectiveArea);
          }
        }
        // UV-math bonuses only apply where tiling resolved to an actual repeat factor: a tiled
        // small texture is an identity albedo, a plain UV transform on one is an overlay tell
        const bool hasResolvedTiling = effectiveArea > area;

        int64_t score = 0;
        score += srgb ? 1'000'000 : 0;
        score += int64_t(std::min<uint16_t>(sampleCount, 16u)) * 120'000ll;
        score += hasInferredTexcoord ? 250'000 : -150'000;
        score += hasResolvedTiling ? 80'000 : 0;
        score -= (isRenderTarget && !isMovieTexture) ? 500'000 : 0;
        score += isMovieTexture ? 4'000'000 : 0;
        score += inferredSamplerLooksMaterialTexture ? 230'000 : 0;
        score -= inferredSamplerLooksEngineAuxiliary ? 420'000 : 0;
        score -= inferredSamplerLooksLightmap ? 280'000 : 0;
        score -= inferredSamplerLooksNonDiffuse ? 220'000 : 0;
        score -= inferredSamplerExprViewDependent ? 260'000 : 0;
        score -= inferredSamplerExprMaskControl ? 320'000 : 0;
        score -= inferredSamplerLooksVideo ? 8'000'000 : 0;
        score -= isNeverAlbedo ? 6'000'000 : 0;
        score += isPreferredAlbedo ? 8'000'000 : 0;
        // bytecode-proven tangent-space normal decode (t * 2 - 1 into normalize/dot chains):
        // decisive penalty - must outweigh typical size advantages. Only applies to linear
        // (non-sRGB) samplers: UE3 imports normal maps with SRGB=0, while gamma-decoded color
        // textures can pick up the flag spuriously in lit shaders full of *2-1 remap math.
        const bool normalDecodeActive = inferredSamplerExprNormalDecode && !srgb;
        score -= normalDecodeActive ? 1'500'000 : 0;
        // only a high spread is directional: mid spreads (3-7) are just as often a legitimately
        // reused diffuse (e.g. common city wall/plaster sheets) as a shared overlay
        if (materialSpread >= 8)
          score -= 2'600'000;
        // tiny ramp/tint lookups (gradients, palettes) are material parameters, not albedo
        score -= (area > 0 && area <= 1'024) ? 350'000 : 0;
        if (m_frameOptions.ue3EngineMode) {
          // UE3 binds many scene buffers, shadow maps, exposure/color curves, and UI/video
          // surfaces alongside material samplers. Keep these out of legacy albedo slots.
          score -= ((isRenderTarget && !isMovieTexture) || inferredSamplerLooksEngineAuxiliary) ? 450'000 : 0;
          score -= (inferredSamplerLooksEngineAuxiliary && !inferredSamplerLooksMaterialTexture) ? 350'000 : 0;
          score -= (isRenderTarget && !srgb && !isMovieTexture) ? 250'000 : 0;
          score -= (!hasInferredTexcoord && inferredSamplerLooksEngineAuxiliary) ? 220'000 : 0;
        }
        const bool looksExpressionDrivenMaterial =
          !inferredSamplerLooksEngineAuxiliary &&
          !inferredSamplerLooksLightmap &&
          !inferredSamplerLooksNonDiffuse &&
          !inferredSamplerExprViewDependent &&
          !inferredSamplerExprMaskControl &&
          !isNeverAlbedo &&
          (inferredSamplerLooksMaterialTexture || inferredSamplerSemanticFlags == 0);
        // On a static mesh UV set 1 is the secondary/lightmap channel, so between two material
        // samplers the one reading the primary .xy pair is the surface's own diffuse. Proven per
        // sampler from bytecode and therefore decided on the first frame, which matters because
        // the winner also fixes the surface's texcoord set: leaving equally-sized candidates to
        // the near-tie tiebreaks below would let a second layer take the slot and drag the whole
        // surface onto its UV set. The packed-UV block below expresses the same idea from
        // statistical inference, so it cannot separate samplers whose inference came out equal;
        // this term is additive to it.
        score += (looksExpressionDrivenMaterial && inferredSamplerReadsPrimaryUvPair) ? 150'000 : 0;
        score += (looksExpressionDrivenMaterial && inferredSamplerExprUvTransform && hasResolvedTiling) ? 95'000 : 0;
        score += (looksExpressionDrivenMaterial && inferredSamplerExprUvOffset) ? 52'000 : 0;
        score += (looksExpressionDrivenMaterial && inferredSamplerExprUvAnimated) ? 115'000 : 0;
        score += (looksExpressionDrivenMaterial && inferredSamplerExprUvTimeDriven && sampleCount >= 2u) ? 140'000 : 0;
        score += (looksExpressionDrivenMaterial && inferredSamplerExprColorContribution) ? 65'000 : 0;
        score += (looksExpressionDrivenMaterial && inferredSamplerExprBlendMath && sampleCount >= 2u) ? 60'000 : 0;
        // sampled value provably reaches oC0.rgb as color - the defining trait of a diffuse/emissive
        // texture, which normal/lighting-input samplers lack (their values collapse in dot products)
        score += (looksExpressionDrivenMaterial && inferredSamplerExprReachesOutputColor) ? 200'000 : 0;
        // deterministic UE3 base-pass structure: only the diffuse expression is multiplied with the
        // lightmap sample (static geometry) or the ambient/sky lighting constants (dynamic/unlit).
        // Dominates the softer heuristics but stays below movie surfaces and user texture tags.
        // Render targets are excluded: light-environment attenuation buffers also multiply into
        // sky/ambient lighting terms but can never be a surface albedo.
        score += (looksExpressionDrivenMaterial && !normalDecodeActive && !isRenderTarget &&
                  inferredSamplerExprDiffuseAnchor) ? 2'500'000 : 0;
        // A sampler whose UV origin is unprovable must not win on size: the transform is taken
        // from the winner only, so promoting one drops the surface to raw interpolant UVs.
        // Flat, so a shader where no sampler resolves is left unaffected.
        score -= (uvOriginPenaltyActive && !inferredSamplerUvOriginProvable) ? 1'200'000 : 0;
        // normal maps get no size credit - resolution advantage must not offset the decode penalty.
        // Under UE3 the same applies to render targets that only survived the refusal above
        // through an explicit tag: their near-backbuffer area would dominate every other signal.
        const bool renderTargetSizeCreditDenied =
          m_frameOptions.ue3EngineMode && isRenderTarget && !isMovieTexture;
        // Size counts in mip steps, not texels: streaming rewrites the bound dimensions as a
        // level pages in, so raw area would let whichever candidate had paged in further win
        // outright. One doubling is worth less than any single structural signal, which orders
        // equally-classified candidates by size without letting residency overturn class.
        score += (normalDecodeActive || renderTargetSizeCreditDenied)
          ? 0
          : int64_t(albedoAreaScore(effectiveArea));
        // near-exact ties among color-chain candidates (magnitudes stay below any real signal):
        // prefer a UV transform (the tiled base material - overlays sample raw UVs), then the
        // LATER sampler (the material translator assigns Texture2D_N slots in property compile
        // order Normal -> Emissive -> Diffuse). Everywhere else prefer the earlier stage.
        const bool diffuseChainCandidate =
          looksExpressionDrivenMaterial && !normalDecodeActive && !isRenderTarget &&
          inferredSamplerExprReachesOutputColor;
        if (diffuseChainCandidate) {
          score += inferredSamplerExprUvTransform ? 200 : 0;
          score += int64_t(stage) * 4;
        } else {
          score -= int64_t(stage);
        }
        if (likelyPackedUvConventions) {
          // UE3-style shader paths (skinned and non-skinned vertex factories) frequently pack
          // secondary UVs into non-`.xy` components, so we bias slot 0 toward diffuse-like UV usage
          const bool packedPairSuspicious =
            hasNonZeroInferredOffset ||
            inferredSamplerLooksEngineAuxiliary ||
            (likelyGpuSkinnedMesh && !inferredSamplerLooksMaterialTexture);

          const bool skinnedHighConfidence =
            likelyGpuSkinnedMesh && inferredSamplerLooksMaterialTexture && sampleCount >= 3u;
          const int64_t uv0Bonus = likelyGpuSkinnedMesh ? 300'000ll : 60'000ll;
          const int64_t nonUv0Penalty = likelyGpuSkinnedMesh ? 180'000ll : 20'000ll;
          const int64_t xyBonus = likelyGpuSkinnedMesh ? 120'000ll : 35'000ll;
          const int64_t wzPenalty = likelyGpuSkinnedMesh
            ? (skinnedHighConfidence ? 160'000ll : 320'000ll)
            : (packedPairSuspicious ? 120'000ll : 8'000ll);
          const int64_t zwPenalty = likelyGpuSkinnedMesh
            ? (skinnedHighConfidence ? 125'000ll : 250'000ll)
            : (packedPairSuspicious ? 100'000ll : 8'000ll);
          const int64_t packedSecondaryPenalty = likelyGpuSkinnedMesh
            ? (skinnedHighConfidence ? 40'000ll : 80'000ll)
            : (packedPairSuspicious ? 28'000ll : 2'000ll);
          const int64_t offsetPenalty = likelyGpuSkinnedMesh
            ? 220'000ll
            : (inferredSamplerLooksEngineAuxiliary ? 180'000ll
               : (inferredSamplerLooksMaterialTexture ? 12'000ll : 45'000ll));
          const int64_t auxiliaryPenalty = likelyGpuSkinnedMesh ? 120'000ll : 70'000ll;
          const bool looksUvAnimatedMaterialTexture =
            inferredSamplerLooksMaterialTexture &&
            !inferredSamplerLooksEngineAuxiliary &&
            !inferredSamplerLooksNonDiffuse &&
            sampleCount >= 2u;
          const int64_t adjustedOffsetPenalty = looksUvAnimatedMaterialTexture
            ? std::max<int64_t>(offsetPenalty / 6ll, 2'000ll)
            : offsetPenalty;

          if (inferredTexcoordIdx == 0)
            score += uv0Bonus;
          else if (inferredTexcoordIdx > 0)
            score -= nonUv0Penalty;

          if (inferredUsesXy)
            score += xyBonus;
          if (inferredUsesWz)
            score -= wzPenalty;
          else if (inferredUsesZw)
            score -= zwPenalty;

          if (inferredUsesPackedSecondary)
            score -= packedSecondaryPenalty;

          if (hasNonZeroInferredOffset)
            score -= adjustedOffsetPenalty;

          if (inferredSamplerLooksEngineAuxiliary)
            score -= auxiliaryPenalty;

          const bool highConfidenceMaterialTexture =
            inferredSamplerLooksMaterialTexture && sampleCount >= 3u;
          if (inferredSamplerLooksMaterialTexture &&
              inferredUsesPackedSecondary &&
              !packedPairSuspicious &&
              (likelyUe3FlexiblePackedUvPath || (likelyGpuSkinnedMesh && highConfidenceMaterialTexture))) {
            score += 45'000ll;
          }
        }

        if (logAlbedoSelection) {
          std::string flagList;
          auto appendFlag = [&](const bool set, const char* name) {
            if (!set)
              return;
            if (!flagList.empty())
              flagList += "|";
            flagList += name;
          };
          appendFlag(inferredSamplerLooksMaterialTexture, "MAT");
          appendFlag(inferredSamplerLooksEngineAuxiliary, "AUX");
          appendFlag(inferredSamplerLooksLightmap, "LIGHTMAP");
          appendFlag(inferredSamplerLooksNonDiffuse, "NONDIFFUSE");
          appendFlag(inferredSamplerLooksVideo, "VIDEO");
          appendFlag(inferredSamplerLooksMovieTexture, "MOVIE");
          appendFlag(inferredSamplerExprUvTransform, "UVXFORM");
          appendFlag(inferredSamplerExprUvOffset, "UVOFS");
          appendFlag(inferredSamplerExprUvAnimated, "UVANIM");
          appendFlag(inferredSamplerExprUvTimeDriven, "UVTIME");
          appendFlag(inferredSamplerExprViewDependent, "VIEWDEP");
          appendFlag(inferredSamplerExprMaskControl, "MASKCTL");
          appendFlag(inferredSamplerExprColorContribution, "COLORCONTRIB");
          appendFlag(inferredSamplerExprBlendMath, "BLEND");
          appendFlag(normalDecodeActive, "NORMALDECODE");
          appendFlag(inferredSamplerExprNormalDecode && !normalDecodeActive, "NORMALDECODE-SRGBVETO");
          appendFlag(inferredSamplerExprReachesOutputColor, "REACHESOC0");
          appendFlag(inferredSamplerExprDiffuseAnchor, "ANCHOR");
          appendFlag(isNeverAlbedo, "TAG:NEVERALBEDO");
          appendFlag(isPreferredAlbedo, "TAG:PREFERALBEDO");
          appendFlag(isRenderTarget, "RT");
          appendFlag(uvOriginPenaltyActive && !inferredSamplerUvOriginProvable, "NOUVORIGIN");
          appendFlag(inferredSamplerReadsPrimaryUvPair, "UV0XY");

          albedoSelectionLog += str::format(
            "\n  s", stage,
            " tex=0x", std::hex, texHash, std::dec,
            " ", desc ? desc->Width : 0u, "x", desc ? desc->Height : 0u,
            " srgb=", srgb ? 1 : 0,
            " samples=", sampleCount,
            " tc=", int32_t(inferredTexcoordIdx),
            " effArea=", effectiveArea,
            " spread=", materialSpread,
            " flags=[", flagList.empty() ? "-" : flagList, "]",
            " score=", score);
        }

        // insert into top-2 (simple selection sort)
        for (uint32_t slot = 0; slot < LegacyMaterialData::kMaxSupportedTextures; slot++) {
          if (stage == chosenStages[slot])
            break;

          if (score > chosenScore[slot]) {
            for (uint32_t s = LegacyMaterialData::kMaxSupportedTextures - 1; s > slot; s--) {
              chosenStages[s] = chosenStages[s - 1];
              chosenScore[s] = chosenScore[s - 1];
            }
            chosenStages[slot] = uint8_t(stage);
            chosenScore[slot] = score;
            break;
          }
        }
      }

      if (chosenStages[0] == kInvalidStage && strictCubemapFallbackStage != kInvalidStage) {
        chosenStages[0] = strictCubemapFallbackStage;
      }

      // if no candidate scored, fall back to the lowest PS-used sampler holding a bindable
      // (hashed) texture - never raw stage 0, which may hold a stale texture the shader
      // never samples, and never a hashless texture.
      // The second pass relaxes the render-target refusal: a refused target must not displace a
      // real material texture, but when the shader samples nothing else it is the material's only
      // colour source, and dropping it leaves the surface with no albedo and no taggable hash.
      for (uint32_t pass = 0; pass < 2 && chosenStages[0] == kInvalidStage; pass++) {
        const bool allowRefusedRenderTargets = pass == 1;
        for (uint32_t stage : bit::BitMask(usedTextureMask)) {
          if (stage >= SamplerCount || d3d9State().textures[stage] == nullptr)
            continue;
          D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[stage]);
          if (texture == nullptr || texture->GetImage() == nullptr ||
              texture->GetImage()->getHash() == kEmptyHash)
            continue;
          if (!allowRefusedRenderTargets && m_frameOptions.ue3EngineMode &&
              isUe3RenderTargetRefusedAsAlbedo(texture, stage, inferredPsEntry))
            continue;
          chosenStages[0] = uint8_t(stage);
          break;
        }
      }

      // compat fix for UE3-like packed UV conventions:
      // if primary stage selection is suspicious (packed UV components, non-UV0, or atlas offset)
      // prefer a sibling sampler that references the same texture but looks more diffuse-like
      if (!selectionFromCache &&
          likelyPackedUvConventions &&
          inferredPsEntry != nullptr &&
          chosenStages[0] != kInvalidStage &&
          chosenStages[0] < caps::MaxTexturesPS &&
          d3d9State().textures[chosenStages[0]] != nullptr) {
        const uint8_t primaryStage = chosenStages[0];
        const bool primaryUsesZw = inferredPsEntry->samplerCoordCompValid[primaryStage] != 0 &&
                                   inferredPsEntry->samplerCoordCompU[primaryStage] == 2 &&
                                   inferredPsEntry->samplerCoordCompV[primaryStage] == 3;
        const bool primaryUsesWz = inferredPsEntry->samplerCoordCompValid[primaryStage] != 0 &&
                                   inferredPsEntry->samplerCoordCompU[primaryStage] == 3 &&
                                   inferredPsEntry->samplerCoordCompV[primaryStage] == 2;
        const bool primaryUsesPackedSecondary = primaryUsesWz || primaryUsesZw;
        const int8_t primaryTexcoord = inferredPsEntry->samplerToTexcoord[primaryStage];
        const bool primaryHasNonZeroOffset = hasNonZeroInferredSamplerOffset(inferredPsEntry, primaryStage);
        const bool primaryLooksEngineAuxiliary =
          (inferredPsEntry->samplerSemanticFlags[primaryStage] & kPsSamplerSemanticEngineAuxiliary) != 0;
        const bool primaryLooksMaterialTexture =
          (inferredPsEntry->samplerSemanticFlags[primaryStage] & kPsSamplerSemanticMaterialTexture) != 0;
        const bool primaryLooksViewDependent =
          (inferredPsEntry->samplerExpressionFlags[primaryStage] & kPsSamplerExprViewDependent) != 0;
        const bool primaryLooksMaskControl =
          (inferredPsEntry->samplerExpressionFlags[primaryStage] & kPsSamplerExprMaskControl) != 0;
        const bool primaryPackedPairSuspicious =
          primaryUsesPackedSecondary &&
          (primaryLooksEngineAuxiliary ||
           primaryHasNonZeroOffset ||
           (likelyGpuSkinnedMesh && !primaryLooksMaterialTexture));
        const bool primaryOffsetSuspicious =
          primaryHasNonZeroOffset &&
          (likelyGpuSkinnedMesh || primaryLooksEngineAuxiliary);
        const bool primarySuspicious =
          primaryLooksEngineAuxiliary ||
          primaryLooksViewDependent ||
          primaryLooksMaskControl ||
          primaryPackedPairSuspicious ||
          primaryOffsetSuspicious ||
          (likelyGpuSkinnedMesh && primaryTexcoord > 0);

        if (primarySuspicious) {
          D3D9CommonTexture* primaryTexture = GetCommonTexture(d3d9State().textures[primaryStage]);
          const XXH64_hash_t primaryHash =
            (primaryTexture != nullptr && primaryTexture->GetImage() != nullptr)
              ? primaryTexture->GetImage()->getHash()
              : kEmptyHash;

          auto scoreDiffuseCandidate = [&](const uint32_t stage) -> int32_t {
            if (stage >= caps::MaxTexturesPS)
              return std::numeric_limits<int32_t>::min();

            if (inferredPsEntry->samplerSampleCount[stage] == 0)
              return std::numeric_limits<int32_t>::min();

            int32_t score = 0;
            const int32_t uv0Bonus = likelyGpuSkinnedMesh ? 420 : 110;
            const int32_t nonUv0Penalty = likelyGpuSkinnedMesh ? 240 : 40;
            const int32_t xyBonus = likelyGpuSkinnedMesh ? 320 : 80;

            const int8_t tc = inferredPsEntry->samplerToTexcoord[stage];
            if (tc == 0)
              score += uv0Bonus;
            else if (tc > 0)
              score -= nonUv0Penalty;

            const uint8_t semanticFlags = inferredPsEntry->samplerSemanticFlags[stage];
            const uint16_t expressionFlags = inferredPsEntry->samplerExpressionFlags[stage];
            const bool looksMaterialTexture = (semanticFlags & kPsSamplerSemanticMaterialTexture) != 0;
            const bool looksEngineAuxiliary = (semanticFlags & kPsSamplerSemanticEngineAuxiliary) != 0;
            const bool looksNonDiffuse = (semanticFlags & kPsSamplerSemanticNonDiffuse) != 0;
            const bool looksVideo = (semanticFlags & kPsSamplerSemanticVideo) != 0;
            const bool looksExprUvTransform = (expressionFlags & kPsSamplerExprUvTransform) != 0;
            const bool looksExprUvOffset = (expressionFlags & kPsSamplerExprUvOffset) != 0;
            const bool looksExprUvAnimated = (expressionFlags & kPsSamplerExprUvAnimated) != 0;
            const bool looksExprUvTimeDriven = (expressionFlags & kPsSamplerExprUvTimeDriven) != 0;
            const bool looksExprViewDependent = (expressionFlags & kPsSamplerExprViewDependent) != 0;
            const bool looksExprMaskControl = (expressionFlags & kPsSamplerExprMaskControl) != 0;
            const bool looksExprColorContribution = (expressionFlags & kPsSamplerExprColorContribution) != 0;
            const bool looksExprBlendMath = (expressionFlags & kPsSamplerExprBlendMath) != 0;
            const bool looksExprNormalDecode = (expressionFlags & kPsSamplerExprNormalDecode) != 0;
            const bool looksExprReachesOutputColor = (expressionFlags & kPsSamplerExprReachesOutputColor) != 0;
            const bool looksExprDiffuseAnchor = (expressionFlags & kPsSamplerExprDiffuseAnchor) != 0;
            if ((semanticFlags & kPsSamplerSemanticMaterialTexture) != 0)
              score += 180;
            if ((semanticFlags & kPsSamplerSemanticEngineAuxiliary) != 0)
              score -= 360;
            if ((semanticFlags & kPsSamplerSemanticLightmap) != 0)
              score -= 260;
            if (m_frameOptions.ue3EngineMode && looksEngineAuxiliary)
              score -= looksMaterialTexture ? 160 : 320;
            if (looksNonDiffuse)
              score -= 220;
            if (looksExprViewDependent)
              score -= 240;
            if (looksExprMaskControl)
              score -= 280;
            if (looksVideo)
              score -= 8000;
            const bool candidateSrgb =
              (d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1) != 0;
            const bool candidateNormalDecode = looksExprNormalDecode && !candidateSrgb;
            if (candidateNormalDecode)
              score -= 3000;
            const bool looksExpressionDrivenMaterial =
              !looksEngineAuxiliary &&
              !looksNonDiffuse &&
              !looksExprViewDependent &&
              !looksExprMaskControl &&
              (looksMaterialTexture || semanticFlags == 0);
            if (looksExpressionDrivenMaterial && looksExprUvTransform)
              score += 95;
            if (looksExpressionDrivenMaterial && looksExprUvOffset)
              score += 55;
            if (looksExpressionDrivenMaterial && looksExprUvAnimated)
              score += 125;
            if (looksExpressionDrivenMaterial && looksExprUvTimeDriven &&
                inferredPsEntry->samplerSampleCount[stage] >= 2u)
              score += 70;
            if (looksExpressionDrivenMaterial && looksExprColorContribution)
              score += 30;
            if (looksExpressionDrivenMaterial && looksExprBlendMath &&
                inferredPsEntry->samplerSampleCount[stage] >= 2u)
              score += 45;
            if (looksExpressionDrivenMaterial && looksExprReachesOutputColor)
              score += 80;
            if (looksExpressionDrivenMaterial && !candidateNormalDecode && looksExprDiffuseAnchor)
              score += 600;

            const bool candidateHasNonZeroOffset = hasNonZeroInferredSamplerOffset(inferredPsEntry, stage);
            const bool candidatePackedPairSuspicious =
              candidateHasNonZeroOffset ||
              looksEngineAuxiliary ||
              (likelyGpuSkinnedMesh && !looksMaterialTexture);
            const int32_t wzPenalty = likelyGpuSkinnedMesh
              ? 380
              : (candidatePackedPairSuspicious ? 180 : 20);
            const int32_t zwPenalty = likelyGpuSkinnedMesh
              ? 360
              : (candidatePackedPairSuspicious ? 160 : 20);
            const int32_t offsetPenalty = likelyGpuSkinnedMesh
              ? 320
              : (looksEngineAuxiliary ? 220 : (looksMaterialTexture ? 20 : 60));
            const bool candidateLooksUvAnimatedMaterialTexture =
              looksMaterialTexture &&
              !looksEngineAuxiliary &&
              !looksNonDiffuse &&
              inferredPsEntry->samplerSampleCount[stage] >= 2u;
            const int32_t adjustedOffsetPenalty = candidateLooksUvAnimatedMaterialTexture
              ? std::max(offsetPenalty / 6, 8)
              : offsetPenalty;

            if (inferredPsEntry->samplerCoordCompValid[stage]) {
              const uint8_t compU = inferredPsEntry->samplerCoordCompU[stage] & 0x3u;
              const uint8_t compV = inferredPsEntry->samplerCoordCompV[stage] & 0x3u;
              if (compU == 0u && compV == 1u)
                score += xyBonus;
              else if (compU == 3u && compV == 2u)
                score -= wzPenalty;
              else if (compU == 2u && compV == 3u)
                score -= zwPenalty;
            }

            if (candidateHasNonZeroOffset)
              score -= adjustedOffsetPenalty;

            if (!likelyGpuSkinnedMesh &&
                likelyUe3FlexiblePackedUvPath &&
                looksMaterialTexture &&
                inferredPsEntry->samplerCoordCompValid[stage]) {
              const uint8_t compU = inferredPsEntry->samplerCoordCompU[stage] & 0x3u;
              const uint8_t compV = inferredPsEntry->samplerCoordCompV[stage] & 0x3u;
              const bool usesPackedSecondaryPair =
                (compU == 2u && compV == 3u) ||
                (compU == 3u && compV == 2u);
              if (usesPackedSecondaryPair && !candidatePackedPairSuspicious)
                score += 60;
            }

            score += int32_t(std::min<uint16_t>(inferredPsEntry->samplerSampleCount[stage], 8u)) * 16;
            score -= int32_t(stage);

            return score;
          };

          if (primaryHash != kEmptyHash) {
            int32_t bestScore = scoreDiffuseCandidate(primaryStage);
            uint8_t promotedStage = kInvalidStage;

            for (uint32_t stage : bit::BitMask(usedTextureMask)) {
              if (stage >= SamplerCount ||
                  stage == primaryStage ||
                  d3d9State().textures[stage] == nullptr ||
                  stage >= caps::MaxTexturesPS) {
                continue;
              }

              D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[stage]);
              if (texture == nullptr || texture->GetImage() == nullptr)
                continue;

              if (texture->GetImage()->getHash() != primaryHash)
                continue;

              const int32_t candidateScore = scoreDiffuseCandidate(stage);
              if (candidateScore > bestScore + 40) {
                bestScore = candidateScore;
                promotedStage = uint8_t(stage);
              }
            }

            if (promotedStage != kInvalidStage) {
              if (chosenStages[1] == promotedStage)
                std::swap(chosenStages[0], chosenStages[1]);
              else
                chosenStages[0] = promotedStage;
            }
          }
        }
      }

      // remember the final (post-fallback, post-promotion) decision for this material key,
      // overwriting any decision made against a smaller (streamed-down) texel area
      if (auditingPinnedSelection) {
        if ((chosenStages[0] != auditedPinnedStages[0] || chosenStages[1] != auditedPinnedStages[1]) &&
            !m_ue3DiffuseSelectionAuditWarned) {
          m_ue3DiffuseSelectionAuditWarned = true;
          Logger::warn(str::format(
            "[RTX-Compatibility][UE3] Albedo selection cache disagrees with current scoring "
            "(material key 0x", std::hex, selectionCacheKey, ": stored [s",
            uint32_t(auditedPinnedStages[0]), ",s", uint32_t(auditedPinnedStages[1]),
            "], scored [s", uint32_t(chosenStages[0]), ",s", uint32_t(chosenStages[1]), "])", std::dec,
            ". Stored picks are being used, so a scoring change will not take effect: bump "
            "kUe3DiffuseSelectionScoringVersion, or delete ", kUe3DiffuseSelectionCachePath, "."));
        }
        // The pin stands regardless - the audit reports, it does not re-decide.
        chosenStages[0] = auditedPinnedStages[0];
        chosenStages[1] = auditedPinnedStages[1];
        strictCubemapFallbackStage = auditedPinnedCubemapStage;
      } else if (selectionCacheUsable && !selectionFromCache) {
        Ue3DiffuseSelectionEntry cacheEntry;
        cacheEntry.chosenStages[0] = chosenStages[0];
        cacheEntry.chosenStages[1] = chosenStages[1];
        cacheEntry.cubemapFallbackStage = strictCubemapFallbackStage;
        cacheEntry.decisionAreaSum = selectionBoundAreaSum;
        m_ue3DiffuseSelectionCache[selectionCacheKey] = cacheEntry;
        m_ue3DiffuseSelectionDirty = true;
      }

      if (logAlbedoSelection) {
        m_loggedAlbedoSelections.insert(selectionCacheKey);
        auto stageName = [&](const uint8_t stage) {
          return stage == kInvalidStage ? std::string("-") : str::format("s", uint32_t(stage));
        };
        Logger::info(str::format(
          "[RTX-Compatibility][UE3-AlbedoSelection] ps=0x", std::hex, inferredPsHash,
          " key=0x", selectionCacheKey, std::dec,
          " chosen=[", stageName(chosenStages[0]), ",", stageName(chosenStages[1]), "]",
          albedoSelectionLog.empty() ? "\n  (no scoreable candidates)" : albedoSelectionLog.c_str()));
      }

      uint32_t textureID = 0;
      for (uint32_t stageIdx = 0; stageIdx < LegacyMaterialData::kMaxSupportedTextures && textureID < LegacyMaterialData::kMaxSupportedTextures; stageIdx++) {
        const uint8_t stage = chosenStages[stageIdx];
        if (stage == kInvalidStage || stage >= SamplerCount || d3d9State().textures[stage] == nullptr)
          continue;

        D3D9CommonTexture* pTexInfo = GetCommonTexture(d3d9State().textures[stage]);
        assert(pTexInfo != nullptr);
        const XXH64_hash_t texHash = pTexInfo->GetImage()->getHash();
        const XXH64_hash_t texDescHash =
          pTexInfo->GetImage() != nullptr
            ? pTexInfo->GetImage()->getDescriptorHash()
            : kEmptyHash;
        const bool allowHashlessCubemap =
          pTexInfo->GetType() == D3DRTYPE_CUBETEXTURE && stage == strictCubemapFallbackStage;

        if (texHash == kEmptyHash && !allowHashlessCubemap)
          continue;

        if (textureID == 0)
          firstStage = stage;

        D3D9SamplerKey key = m_parent->CreateSamplerKey(stage);
        XXH64_hash_t samplerHash = D3D9SamplerKeyHash{}(key);

        Rc<DxvkSampler> sampler;
        auto samplerIt = m_samplerCache.find(samplerHash);
        if (samplerIt != m_samplerCache.end()) {
          sampler = samplerIt->second;
        } else {
          const auto samplerInfo = m_parent->DecodeSamplerKey(key);
          sampler = m_parent->GetDXVKDevice()->createSampler(samplerInfo);
          m_samplerCache.insert(std::make_pair(samplerHash, sampler));
        }

        const bool srgb = d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1;
        Rc<DxvkImageView> sampleView = getRemixSampleView(pTexInfo, srgb);
        if (sampleView == nullptr)
          continue;
        m_activeDrawCallState.materialData.colorTextures[textureID] = TextureRef(sampleView);
        m_activeDrawCallState.materialData.samplers[textureID] = sampler;
        selectedUe3MovieTexture |=
          isUe3MovieTextureDescHash(texDescHash) ||
          (stage < caps::MaxTexturesPS &&
           inferredPsEntry != nullptr &&
           (inferredPsEntry->samplerSemanticFlags[stage] & kPsSamplerSemanticMovieTexture) != 0 &&
           pTexInfo->IsRenderTarget());
        if (textureID == 0)
          m_activeDrawCallState.materialData.colorTextureIsSrgb = srgb;

        auto shaderSampler = RemapStateSamplerShader(stage);
        m_activeDrawCallState.materialData.colorTextureSlot[textureID] = computeResourceSlotId(shaderSampler.first, DxsoBindingType::Image, uint32_t(shaderSampler.second));
        ++textureID;
      }

      if (textureID == 0 &&
          strictCubemapFallbackStage != kInvalidStage &&
          strictCubemapFallbackStage < SamplerCount &&
          d3d9State().textures[strictCubemapFallbackStage] != nullptr) {
        D3D9CommonTexture* pTexInfo = GetCommonTexture(d3d9State().textures[strictCubemapFallbackStage]);
        if (pTexInfo != nullptr && pTexInfo->GetImage() != nullptr) {
          firstStage = strictCubemapFallbackStage;

          D3D9SamplerKey key = m_parent->CreateSamplerKey(firstStage);
          XXH64_hash_t samplerHash = D3D9SamplerKeyHash{}(key);

          Rc<DxvkSampler> sampler;
          auto samplerIt = m_samplerCache.find(samplerHash);
          if (samplerIt != m_samplerCache.end()) {
            sampler = samplerIt->second;
          } else {
            const auto samplerInfo = m_parent->DecodeSamplerKey(key);
            sampler = m_parent->GetDXVKDevice()->createSampler(samplerInfo);
            m_samplerCache.insert(std::make_pair(samplerHash, sampler));
          }

          const bool srgb = d3d9State().samplerStates[firstStage][D3DSAMP_SRGBTEXTURE] & 0x1;
          Rc<DxvkImageView> sampleView = getRemixSampleView(pTexInfo, srgb);
          if (sampleView != nullptr) {
            m_activeDrawCallState.materialData.colorTextures[0] = TextureRef(sampleView);
            m_activeDrawCallState.materialData.samplers[0] = sampler;
            selectedUe3MovieTexture |= isUe3MovieTextureDescHash(pTexInfo->GetImage()->getDescriptorHash());
            m_activeDrawCallState.materialData.colorTextureIsSrgb = srgb;

            auto shaderSampler = RemapStateSamplerShader(firstStage);
            m_activeDrawCallState.materialData.colorTextureSlot[0] =
              computeResourceSlotId(shaderSampler.first, DxsoBindingType::Image, uint32_t(shaderSampler.second));
          }
        }
      }

      if (m_frameOptions.ue3EngineMode && !m_activeDrawCallState.materialData.colorTextures[0].isValid()) {
        logUe3UnboundAlbedoOnce(inferredPs, inferredPsHash, usedSamplerMask, usedTextureMask, inferredPsEntry);
      }
    }

    // Update the drawcall state with texture stage info
    // note: D3D9 exposes more sampler slots than fixed-function texture stages (limited to 8).
    // `setTextureStageState` reads `textureStages[stageIdx]` and `D3DTS_TEXTURE0 + stageIdx`, so clamp to a valid stage index.
    const uint32_t stageStateIdx = (firstStage < caps::TextureStageCount) ? firstStage : 0;
    if (unlikely(firstStage >= caps::TextureStageCount)) {
      ONCE(Logger::warn(str::format(
        "[RTX-Compatibility] Shader-path selected sampler stage ", firstStage,
        " but texture stage state is limited to 0..", (caps::TextureStageCount - 1),
        ". Using stage 0 for texcoord/textureTransform.")));
    }

    setTextureStageState(d3d9State(), stageStateIdx, useStageTextureFactorBlending, useMultipleStageTextureFactorBlending,
                         m_activeDrawCallState.materialData, m_activeDrawCallState.transformData);

    if constexpr (!FixedFunction) {
      if (m_frameOptions.shaderPathTexcoordIndexFromPixelShader || m_frameOptions.ue3EngineMode) {
        // shader-path draws perform UV math in shader code
        // fixed-function texture transform/texgen state can be stale and should not be reused
        m_activeDrawCallState.transformData.textureTransform = Matrix4();
        m_activeDrawCallState.transformData.texgenMode = TexGenMode::None;
      }
    }

    // Texture-less (constant-color) materials still need MIC identity for a stable
    // material hash (tagging, replacements, categories) and their color constant
    bool ue3MicIdentityAvailable = false;
    if constexpr (!FixedFunction) {
      ue3MicIdentityAvailable =
        (m_frameOptions.ue3MaterialInstanceConstantHash || m_frameOptions.ue3EngineMode) &&
        m_parent->UseProgrammablePS() &&
        d3d9State().pixelShader.ptr() != nullptr;
    }

    if (d3d9State().textures[firstStage] || ue3MicIdentityAvailable) {
      // UE3 MaterialInstanceConstant compat - deterministic child-level material identity:
      //   PS bytecode hash -> ordered material texture set -> material constants
      // this differentiates instances that share a parent but override TextureParameterValues
      // in any material sampler, VectorParameterValues/ScalarParameterValues (e.g. DiffuseColor),
      // or StaticSwitchParameterValues (different bytecode)
      // must run before setupCategoriesForTexture so category lookups use the full material hash
      if constexpr (!FixedFunction) {
        if ((m_frameOptions.ue3MaterialInstanceConstantHash || m_frameOptions.ue3EngineMode) && m_parent->UseProgrammablePS() && d3d9State().pixelShader.ptr() != nullptr) {
          const D3D9CommonShader* psCommonShader = d3d9State().pixelShader->GetCommonShader();
          const auto& bytecode = psCommonShader->GetBytecode();
          const XXH64_hash_t psHash = psCommonShader->GetBytecodeHash();
          if (psHash != 0) {
            const Ue3PsMaterialIdentityInfo& identityInfo = getOrParseUe3PsMaterialIdentityInfo(psHash, bytecode, psCommonShader);

            // UE3 recompiles the same material's base pass per lightmap policy, so a bytecode
            // hash - and everything seeded by it - varies with DirectionalLightmaps and
            // TdBicubicFiltering. Seed every UE3 material with the canonical CTAB signature
            // instead. This deliberately covers the permutations that carry no lightmap symbols
            // at all: FNoLightMapPolicy compiles the same material for movable geometry,
            // translucency and the unlit viewmode, and gating on lightmap symbols would have
            // given those draws a bytecode-seeded identity that could never match their
            // lightmapped selves.
            const bool useInvariantShaderIdentity =
              m_frameOptions.ue3EngineMode &&
              identityInfo.canonicalShaderSignature != kEmptyHash;
            const XXH64_hash_t shaderIdentitySeed =
              useInvariantShaderIdentity ? identityInfo.canonicalShaderSignature : psHash;

            m_activeDrawCallState.materialData.setPixelShaderHashForMaterialInstance(shaderIdentitySeed);

            // ordered (sampler key, image hash) set over every texture bound to a material
            // sampler (CTAB names Texture2D_* / TextureCube_*) - catches TextureParameterValues
            // overridden in any material sampler, not just the chosen primary color texture.
            // For invariant-identity shaders the sampler key is the CTAB name hash, not the
            // register: lightmap sampler counts shift register assignments between
            // permutations while the material's own sampler names stay fixed.
            const bool logMicHash = m_frameOptions.ue3LogMaterialInstanceHash;
            if (logMicHash && !identityInfo.identitySummary.empty()) {
              static fast_unordered_set s_loggedIdentitySummaries;
              if (s_loggedIdentitySummaries.insert(psHash).second) {
                Logger::info(str::format(
                  "[RTX-Ue3Identity] ps=0x", std::hex, psHash,
                  " seed=0x", identityInfo.canonicalShaderSignature, std::dec,
                  " ", identityInfo.identitySummary));
              }
            }
            std::string micTextureListLog;
            // Replacement identity drift diagnostics: record the (register, image hash, RT flag)
            // tuples that feed textureSetHash so tier drift can be attributed per sampler.
            Ue3MicIdentitySample::SamplerRecord micDiagSamplers[kMicDriftMaxTrackedSamplers];
            uint32_t micDiagSamplerCount = 0;
            const BoundTextureSnapshot& identityBoundTextures = ensureBoundTextureSnapshot();

            XXH64_hash_t textureSetHash = kEmptyHash;
            if (identityInfo.materialSamplerMask != 0) {
              XXH3_state_t* const state = getThreadLocalXxh3State();
              if (state != nullptr) {
                XXH3_64bits_reset(state);
                bool anyTextureHashed = false;
                const BoundTextureSnapshot& boundTextures = identityBoundTextures;
                auto hashMaterialSamplerTexture = [&](const void* samplerKey, const size_t samplerKeySize, const uint32_t samplerRegister, const char* samplerLogName) -> XXH64_hash_t {
                  if (samplerRegister >= SamplerCount || (boundTextures.mask & (1u << samplerRegister)) == 0)
                    return kEmptyHash;
                  const BoundTextureSnapshotEntry& entry = boundTextures.entries[samplerRegister];
                  if (!entry.hasImage)
                    return kEmptyHash;
                  const XXH64_hash_t imageHash = entry.imageHash;
                  if (imageHash == kEmptyHash)
                    return kEmptyHash; // hashless (e.g. render target bound as a material texture)
                  if (m_frameOptions.ue3MicExcludeRenderTargetsFromIdentity && entry.isRenderTarget) {
                    // RT image hashes change on every recreation (respawn/checkpoint/level
                    // load) and would re-mint the identity each time; treat as hashless.
                    if (m_frameOptions.logReplacementResolution || m_frameOptions.ue3LogMaterialInstanceHash) {
                      static fast_unordered_set s_loggedRtIdentityExclusions;
                      const XXH64_hash_t exclusionLogKey = XXH3_64bits_withSeed(&samplerRegister, sizeof(samplerRegister), psHash);
                      if (s_loggedRtIdentityExclusions.insert(exclusionLogKey).second) {
                        Logger::info(str::format(
                          "[RTX-Compatibility][UE3-MIC] Excluded render-target image 0x", std::hex, imageHash,
                          " (RT descriptor hash 0x", entry.rtDescriptorHash,
                          ") at material sampler s", std::dec, samplerRegister,
                          " of pixel shader 0x", std::hex, psHash, std::dec,
                          " from material identity (rtx.d3d9.ue3MicExcludeRenderTargetsFromIdentity)."));
                      }
                    }
                    return kEmptyHash;
                  }
                  if (!m_frameOptions.ue3MicIdentityExcludedTextureDescHashes->empty() &&
                      lookupHash(*m_frameOptions.ue3MicIdentityExcludedTextureDescHashes, entry.descriptorHash)) {
                    // User-tagged session-unstable texture (engine-composited contents give it a
                    // new image hash every session): keep it out of the identity so anchors hold.
                    if (m_frameOptions.logReplacementResolution || m_frameOptions.ue3LogMaterialInstanceHash) {
                      static fast_unordered_set s_loggedDescIdentityExclusions;
                      const XXH64_hash_t exclusionLogKey = XXH3_64bits_withSeed(&samplerRegister, sizeof(samplerRegister), psHash);
                      if (s_loggedDescIdentityExclusions.insert(exclusionLogKey).second) {
                        Logger::info(str::format(
                          "[RTX-Compatibility][UE3-MIC] Excluded texture image 0x", std::hex, imageHash,
                          " (descriptor hash 0x", entry.descriptorHash,
                          ") at material sampler s", std::dec, samplerRegister,
                          " of pixel shader 0x", std::hex, psHash, std::dec,
                          " from material identity (rtx.d3d9.ue3MicIdentityExcludedTextureDescHashes)."));
                      }
                    }
                    return kEmptyHash;
                  }
                  XXH3_64bits_update(state, samplerKey, samplerKeySize);
                  XXH3_64bits_update(state, &imageHash, sizeof(imageHash));
                  anyTextureHashed = true;
                  if (micDiagSamplerCount < kMicDriftMaxTrackedSamplers) {
                    micDiagSamplers[micDiagSamplerCount++] = Ue3MicIdentitySample::SamplerRecord {
                      uint8_t(samplerRegister), entry.isRenderTarget, imageHash, entry.descriptorHash };
                  }
                  if (logMicHash) {
                    micTextureListLog += str::format(
                      micTextureListLog.empty() ? "s" : ",s", samplerRegister,
                      samplerLogName != nullptr ? str::format("(", samplerLogName, ")") : std::string(),
                      ":0x", std::hex, imageHash, "(desc:0x", entry.descriptorHash, ")", std::dec);
                  }
                  return imageHash;
                };
                if (useInvariantShaderIdentity) {
                  // name-keyed: register assignments shift between lightmap policy permutations
                  for (const auto& [samplerName, samplerNameKey, samplerRegister] : identityInfo.materialSamplersByNameOrder) {
                    if (samplerRegister < caps::MaxTexturesPS) {
                      hashMaterialSamplerTexture(&samplerNameKey, sizeof(samplerNameKey), samplerRegister, samplerName.c_str());
                    }
                  }
                } else {
                  // (uint32_t register, image hash) pairs - the historical stream, preserved
                  // so non-lightmap material hashes stay identical to prior builds
                  for (uint32_t s = 0; s < caps::MaxTexturesPS; s++) {
                    if ((identityInfo.materialSamplerMask & (1u << s)) == 0)
                      continue;
                    hashMaterialSamplerTexture(&s, sizeof(s), s, nullptr);
                  }
                }
                if (anyTextureHashed)
                  textureSetHash = XXH3_64bits_digest(state);
              }
            }
            m_activeDrawCallState.materialData.setMaterialTextureSetHashForMaterialInstance(textureSetHash);

            const bool autoExcludeEnabled = m_frameOptions.ue3MicAutoExcludeFrameVaryingConstants;
            // Manual exclusion honours both the raw bytecode hash (existing configs) and the
            // identity seed; auto-exclusion is scoped to the (seed, texture set) group, which
            // is permutation-consistent for invariant-identity shaders yet never wider than
            // the one churning material family.
            const XXH64_hash_t micChurnGroupKey = makeUe3MicChurnGroupKey(shaderIdentitySeed, textureSetHash);
            bool constantsExcluded =
              // The tier that cannot be made lightmap-policy independent; opt-in only.
              !m_frameOptions.ue3MicConstantIdentity ||
              lookupHash(*m_frameOptions.ue3MicConstantIdentityExcludedShaders, psHash) ||
              (useInvariantShaderIdentity && lookupHash(*m_frameOptions.ue3MicConstantIdentityExcludedShaders, shaderIdentitySeed)) ||
              // Same scope as the automatic exclusion, for families it never trips on.
              lookupHash(*m_frameOptions.ue3MicConstantIdentityExcludedGroups, micChurnGroupKey) ||
              (autoExcludeEnabled && isUe3MicGroupAutoExcluded(micChurnGroupKey, m_frameOptions.ue3MicPersistAutoExcludedConstantGroups));
            // Invariant-identity shaders hash constants by uniform name and leading register:
            // lightmap policy permutations shift uniform registers and trim per-permutation
            // unreferenced elements, so the raw register-range stream is not comparable
            // across permutations. Other shaders keep the historical register-range stream
            // so their hashes stay identical to prior builds.
            XXH64_hash_t psConstsHash = kEmptyHash;
            if (!constantsExcluded) {
              psConstsHash = useInvariantShaderIdentity
                ? hashUe3MaterialConstantsByNameOrder(d3d9State().psConsts.fConsts, identityInfo.namedUniformFirstRegistersByNameOrder)
                : hashUe3MaterialConstants(d3d9State().psConsts.fConsts, identityInfo.constRanges);
            }
            // frame-varying constant registers would mint a new material identity every
            // draw; detect that here and drop constants-based identity for the group
            if (!constantsExcluded && psConstsHash != kEmptyHash && autoExcludeEnabled &&
                trackUe3MicConstantChurn(micChurnGroupKey, psHash, shaderIdentitySeed, textureSetHash, psConstsHash,
                                         m_frameOptions.ue3MicPersistAutoExcludedConstantGroups)) {
              // The exclusion changes this group's material identity mid-session; log both
              // sides so anchor mismatches around the flip are attributable.
              const XXH64_hash_t primaryTexHashForFlipLog = m_activeDrawCallState.materialData.getColorTexture().getImageHash();
              const XXH64_hash_t effectiveTextureSet = (textureSetHash != kEmptyHash) ? textureSetHash : primaryTexHashForFlipLog;
              const XXH64_hash_t identityWithoutConstants = XXH3_64bits_withSeed(&effectiveTextureSet, sizeof(effectiveTextureSet), shaderIdentitySeed);
              const XXH64_hash_t identityWithConstants = XXH3_64bits_withSeed(&psConstsHash, sizeof(psConstsHash), identityWithoutConstants);
              Logger::warn(str::format(
                "[RTX-MicDrift] Constants auto-exclusion changed material identity mid-session for group seed=0x",
                std::hex, shaderIdentitySeed, " textureSet=0x", textureSetHash, " tex=0x", primaryTexHashForFlipLog,
                ": last constants-bearing materialHash 0x", identityWithConstants,
                " -> constants-free materialHash 0x", identityWithoutConstants, std::dec,
                ". Replacements anchored on one side of this flip stop matching on the other."));
              constantsExcluded = true;
              psConstsHash = kEmptyHash;
            }
            m_activeDrawCallState.materialData.setPixelShaderConstantsHashForMaterialInstance(psConstsHash);

            // Replacement identity drift diagnostics: attribute a changed material hash
            // to the tier that moved and flag RT-poisoned identities.
            const bool replacementDiagActive =
              m_frameOptions.logReplacementResolution ||
              (m_frameOptions.replacementDebugHashes != nullptr && !m_frameOptions.replacementDebugHashes->empty());
            if (replacementDiagActive) {
              m_activeDrawCallState.materialData.updateCachedHash();
              const XXH64_hash_t materialHash = m_activeDrawCallState.materialData.getHash();
              const XXH64_hash_t textureSetShaderHash = m_activeDrawCallState.materialData.getTextureSetAndShaderHash();
              const XXH64_hash_t primaryTexHash = m_activeDrawCallState.materialData.getColorTexture().getImageHash();

              bool tracked = false;
              if (m_frameOptions.replacementDebugHashes != nullptr && !m_frameOptions.replacementDebugHashes->empty()) {
                const fast_unordered_set& dbg = *m_frameOptions.replacementDebugHashes;
                tracked = lookupHash(dbg, primaryTexHash) || lookupHash(dbg, materialHash) || lookupHash(dbg, textureSetShaderHash);
                for (uint32_t i = 0; !tracked && i < micDiagSamplerCount; i++) {
                  tracked = lookupHash(dbg, micDiagSamplers[i].imageHash);
                }
              }

              if (m_frameOptions.logReplacementResolution || tracked) {
                const XXH64_hash_t familyKey = XXH3_64bits_withSeed(&primaryTexHash, sizeof(primaryTexHash), shaderIdentitySeed);

                // Snapshot the constant registers feeding the identity so drift can name the
                // register(s) whose values moved.
                Ue3MicIdentitySample::ConstantRecord curConstants[kMicDriftMaxTrackedConstants];
                uint32_t curConstantCount = 0;
                if (!constantsExcluded && psConstsHash != kEmptyHash) {
                  if (useInvariantShaderIdentity) {
                    for (const auto& [uniformNameKey, uniformRegister] : identityInfo.namedUniformFirstRegistersByNameOrder) {
                      if (curConstantCount >= kMicDriftMaxTrackedConstants)
                        break;
                      if (uniformRegister >= caps::MaxFloatConstantsPS)
                        continue;
                      curConstants[curConstantCount++] = Ue3MicIdentitySample::ConstantRecord {
                        uint16_t(uniformRegister), d3d9State().psConsts.fConsts[uniformRegister] };
                    }
                  } else {
                    for (const auto& [rangeStart, rangeCount] : identityInfo.constRanges) {
                      for (uint32_t r = rangeStart; r < rangeStart + rangeCount; r++) {
                        if (curConstantCount >= kMicDriftMaxTrackedConstants || r >= caps::MaxFloatConstantsPS)
                          break;
                        curConstants[curConstantCount++] = Ue3MicIdentitySample::ConstantRecord {
                          uint16_t(r), d3d9State().psConsts.fConsts[r] };
                      }
                      if (curConstantCount >= kMicDriftMaxTrackedConstants)
                        break;
                    }
                  }
                }

                Ue3MicIdentitySample& prev = s_ue3MicIdentityByFamily[familyKey];
                if (prev.valid && prev.materialHash != materialHash &&
                    prev.driftLogsEmitted < (tracked ? kMicDriftMaxLogsPerTrackedFamily : kMicDriftMaxLogsPerFamily)) {
                  ++prev.driftLogsEmitted;
                  std::string detail;
                  if (prev.textureSetHash != textureSetHash) {
                    detail += str::format("\n  textureSet 0x", std::hex, prev.textureSetHash, " -> 0x", textureSetHash, std::dec, ":");
                    for (uint32_t i = 0; i < micDiagSamplerCount; i++) {
                      const Ue3MicIdentitySample::SamplerRecord& cur = micDiagSamplers[i];
                      const Ue3MicIdentitySample::SamplerRecord* old = nullptr;
                      for (uint32_t j = 0; j < prev.samplerCount; j++) {
                        if (prev.samplers[j].reg == cur.reg) {
                          old = &prev.samplers[j];
                          break;
                        }
                      }
                      if (old == nullptr) {
                        detail += str::format(" s", uint32_t(cur.reg), " added=0x", std::hex, cur.imageHash,
                                              "(desc:0x", cur.descriptorHash, ")", std::dec, cur.isRenderTarget ? "(RT)" : "");
                      } else if (old->imageHash != cur.imageHash) {
                        detail += str::format(" s", uint32_t(cur.reg), " 0x", std::hex, old->imageHash, "->0x", cur.imageHash,
                                              "(desc:0x", cur.descriptorHash, ")", std::dec, cur.isRenderTarget ? "(RT)" : "");
                      }
                    }
                    for (uint32_t j = 0; j < prev.samplerCount; j++) {
                      bool stillPresent = false;
                      for (uint32_t i = 0; i < micDiagSamplerCount; i++) {
                        if (micDiagSamplers[i].reg == prev.samplers[j].reg) {
                          stillPresent = true;
                          break;
                        }
                      }
                      if (!stillPresent) {
                        detail += str::format(" s", uint32_t(prev.samplers[j].reg), " removed=0x", std::hex, prev.samplers[j].imageHash, std::dec,
                                              prev.samplers[j].isRenderTarget ? "(RT)" : "");
                      }
                    }
                  }
                  if (prev.constantsExcluded != constantsExcluded) {
                    detail += str::format("\n  constantsExcluded ", prev.constantsExcluded ? 1 : 0, " -> ", constantsExcluded ? 1 : 0);
                  }
                  if (prev.constantsHash != psConstsHash) {
                    detail += str::format("\n  consts 0x", std::hex, prev.constantsHash, " -> 0x", psConstsHash, std::dec, ":");
                    for (uint32_t i = 0; i < curConstantCount; i++) {
                      const Ue3MicIdentitySample::ConstantRecord& cur = curConstants[i];
                      for (uint32_t j = 0; j < prev.constantCount; j++) {
                        const Ue3MicIdentitySample::ConstantRecord& old = prev.constants[j];
                        if (old.reg == cur.reg) {
                          if (old.value.x != cur.value.x || old.value.y != cur.value.y ||
                              old.value.z != cur.value.z || old.value.w != cur.value.w) {
                            detail += str::format(" c", uint32_t(cur.reg),
                                                  " (", old.value.x, ",", old.value.y, ",", old.value.z, ",", old.value.w,
                                                  ")->(", cur.value.x, ",", cur.value.y, ",", cur.value.z, ",", cur.value.w, ")");
                          }
                          break;
                        }
                      }
                    }
                  }
                  Logger::warn(str::format(
                    "[RTX-MicDrift] Material identity changed for family tex=0x", std::hex, primaryTexHash,
                    " seed=0x", shaderIdentitySeed,
                    ": materialHash 0x", prev.materialHash, " -> 0x", materialHash,
                    " (textureSet+shader tier 0x", textureSetShaderHash, ")", std::dec,
                    detail.empty() ? "\n  (no attributable tier diff captured)" : detail.c_str(),
                    "\n  Replacements anchored on the previous hash no longer match this draw."));
                }

                prev.valid = true;
                prev.materialHash = materialHash;
                prev.textureSetHash = textureSetHash;
                prev.constantsHash = psConstsHash;
                prev.constantsExcluded = constantsExcluded;
                prev.samplerCount = micDiagSamplerCount;
                for (uint32_t i = 0; i < micDiagSamplerCount; i++) {
                  prev.samplers[i] = micDiagSamplers[i];
                }
                prev.constantCount = curConstantCount;
                for (uint32_t i = 0; i < curConstantCount; i++) {
                  prev.constants[i] = curConstants[i];
                }

                // RT-poisoning sweep: a render-target-backed image hash inside the identity
                // makes it unstable across RT recreation (respawn / level load).
                for (uint32_t i = 0; i < micDiagSamplerCount; i++) {
                  if (micDiagSamplers[i].isRenderTarget) {
                    if (s_ue3MicRtPoisonWarnedFamilies.insert(familyKey).second) {
                      Logger::warn(str::format(
                        "[RTX-MicRtPoisoning] Material identity for family tex=0x", std::hex, primaryTexHash,
                        " seed=0x", shaderIdentitySeed,
                        " includes render-target image hash 0x", micDiagSamplers[i].imageHash,
                        " at material sampler s", std::dec, uint32_t(micDiagSamplers[i].reg),
                        std::hex, " (stable RT descriptor hash 0x", micDiagSamplers[i].descriptorHash,
                        "): materialHash 0x", materialHash, std::dec,
                        " will change whenever the game recreates this render target (respawn/level load),"
                        " breaking replacements anchored on it."));
                    }
                    break;
                  }
                }
              }
            }

            // Constant-color materials: the surface color lives in a UniformVector_*
            // register. Register order is compile-order, not semantic, and the lowest
            // register frequently holds a zero vector (fades, unused parameters), so scan
            // for the first value that is plausibly a color (finite, non-black, LDR-ish)
            if (identityInfo.materialSamplerMask == 0 &&
                !identityInfo.uniformVectorRegisters.empty() &&
                !m_activeDrawCallState.materialData.colorTextures[0].isValid()) {
              for (const uint32_t reg : identityInfo.uniformVectorRegisters) {
                if (reg >= caps::MaxFloatConstantsPS)
                  continue;
                const Vector4& uniformColor = d3d9State().psConsts.fConsts[reg];
                if (!std::isfinite(uniformColor.x) || !std::isfinite(uniformColor.y) ||
                    !std::isfinite(uniformColor.z) || !std::isfinite(uniformColor.w))
                  continue;
                const float maxComp = std::max({ uniformColor.x, uniformColor.y, uniformColor.z });
                const float minComp = std::min({ uniformColor.x, uniformColor.y, uniformColor.z });
                // reject blacks/negatives (not visible albedo) and HDR-scale values (intensities)
                if (minComp < 0.0f || maxComp <= 0.01f || maxComp > 8.0f)
                  continue;
                m_activeDrawCallState.materialData.ue3ConstantAlbedo = uniformColor;
                m_activeDrawCallState.materialData.hasUe3ConstantAlbedo = true;
                break;
              }
            }

            if (logMicHash) {
              m_activeDrawCallState.materialData.updateCachedHash();
              logUe3MaterialInstanceHashBreakdownOnce(
                m_activeDrawCallState.materialData.getHash(), psHash, shaderIdentitySeed, textureSetHash, psConstsHash,
                identityInfo, constantsExcluded, micTextureListLog);
            }
          }
        }
      }
      m_activeDrawCallState.materialData.updateCachedHash();

      // Texture-less materials have no presence in the texture selection UI; register
      // their material hash as an entry (white thumbnail) so they can be clicked/tagged.
      // Category and replacement lookups already accept material hashes.
      if constexpr (!FixedFunction) {
        if (ue3MicIdentityAvailable &&
            !m_activeDrawCallState.materialData.colorTextures[0].isValid()) {
          const XXH64_hash_t texturelessMaterialHash = m_activeDrawCallState.materialData.getHash();
          static fast_unordered_set s_registeredTexturelessMaterials;
          if (texturelessMaterialHash != kEmptyHash &&
              s_registeredTexturelessMaterials.insert(texturelessMaterialHash).second) {
            m_parent->EmitCs([texturelessMaterialHash](DxvkContext* ctx) {
              const Rc<DxvkImageView> whiteView =
                static_cast<RtxContext*>(ctx)->getResourceManager().getWhiteTexture(ctx);
              if (whiteView != nullptr) {
                ImGUI::AddTexture(texturelessMaterialHash, whiteView, ImGUI::kTextureFlagsDefault);
              }
            });
          }
        }
      }

      m_activeDrawCallState.setupCategoriesForTexture();
      logUe3ParticleDrawOnce();

      // Track the material hash before checking if it should be ignored
      // This ensures we track all materials sent by the game, not just the ones that are actually rendered.
      const XXH64_hash_t textureHash = m_activeDrawCallState.materialData.getColorTexture().getImageHash();
      const XXH64_hash_t materialHash = m_activeDrawCallState.materialData.getHash();
      bool usesMovieTexture = selectedUe3MovieTexture;
      for (uint32_t i = 0; i < LegacyMaterialData::kMaxSupportedTextures; i++) {
        if (m_activeDrawCallState.materialData.colorTextures[i].isValid()) {
          const DxvkImageView* imageView = m_activeDrawCallState.materialData.colorTextures[i].getImageView();
          const XXH64_hash_t descHash =
            imageView != nullptr
              ? imageView->image()->getDescriptorHash()
              : kEmptyHash;
          if (isUe3MovieTextureDescHash(descHash)) {
            usesMovieTexture = true;
            break;
          }
        }
      }
      if (usesMovieTexture) {
        m_activeDrawCallState.setCategory(InstanceCategories::WorldUI, true);
        if (Logger::logLevel() <= LogLevel::Debug) {
          static fast_unordered_set s_loggedMovieSurfaceMaterials;
          if (s_loggedMovieSurfaceMaterials.insert(materialHash).second) {
            Logger::debug(str::format(
              "[RTX-Compatibility][UE3] Marked movie texture material as WorldUI: materialHash=0x",
              std::hex, materialHash, ", textureHash=0x", textureHash, std::dec));
          }
        }
      }

      // Flag smooth normals category at the d3d9 layer
      m_activeDrawCallState.setCategory(InstanceCategories::SmoothNormals, lookupHash(*m_frameOptions.smoothNormalsTextures, textureHash) || lookupHash(*m_frameOptions.smoothNormalsTextures, materialHash));
      if (materialHash != kEmptyHash) {
        // batched into a single CS command in EndFrame (see m_pendingReplacementMaterialHashes)
        m_pendingReplacementMaterialHashes.push_back(materialHash);
      }
      
      // Check if an ignore texture is bound
      if (m_activeDrawCallState.getCategoryFlags().test(InstanceCategories::Ignore)) {
        m_ue3LastDrawDecision = "ignore texture category";
        return false;
      }

      if (m_activeDrawCallState.testCategoryFlags(InstanceCategories::Terrain)) {
        if (m_frameOptions.terrainAsDecalsEnabledIfNoBaker && !TerrainBaker::enableBaking()) {

          m_activeDrawCallState.removeCategory(InstanceCategories::Terrain);
          m_activeDrawCallState.setCategory(InstanceCategories::DecalStatic, true);

          // modulate to compensate the multilayer blending
          DxvkRtTextureOperation& texop = m_activeDrawCallState.materialData.textureColorOperation;
          if (m_frameOptions.terrainAsDecalsAllowOverModulate) {
            if (texop == DxvkRtTextureOperation::Modulate2x || texop == DxvkRtTextureOperation::Modulate4x) {
              texop = DxvkRtTextureOperation::Force_Modulate2x;
            }
          }
        }
      }

      if (!m_forceGeometryCopy && m_frameOptions.alwaysCopyDecalGeometries) {
        // Only poke decal hashes when option is enabled.
        m_forceGeometryCopy |= m_activeDrawCallState.testCategoryFlags(CATEGORIES_REQUIRE_GEOMETRY_COPY);
      }
    } else {
      // No texture / MIC identity: still refresh the material hash once so the
      // submitted draw state carries a valid hash.
      m_activeDrawCallState.materialData.updateCachedHash();
    }

    // only keep the passthrough texcoord index for selecting the vertex declaration element
    // upper bits are D3DTSS_TCI_* flags used for fixed-function texgen
    uint32_t texcoordIdx = d3d9State().textureStages[stageStateIdx][DXVK_TSS_TEXCOORDINDEX] & 0b111;
    uint32_t iaTexcoordIdx = texcoordIdx;
    m_uvResolutionMode = UvResolutionMode::LegacyTss;

    m_forceIaTexcoordForOutlier = [&]() {
      const auto& outlierSet = *m_frameOptions.vsTexcoordCaptureOutlierTextures;
      if (outlierSet.empty()) {
        return false;
      }
      for (uint32_t i = 0; i < LegacyMaterialData::kMaxSupportedTextures; i++) {
        if (lookupHash(outlierSet, m_activeDrawCallState.materialData.colorTextures[i].getImageHash()))
          return true;
      }

      for (uint32_t stage = 0; stage < SamplerCount; stage++) {
        if (d3d9State().textures[stage] == nullptr)
          continue;

        D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[stage]);
        if (texture == nullptr || texture->GetImage() == nullptr)
          continue;

        if (lookupHash(outlierSet, texture->GetImage()->getHash()))
          return true;
      }

      return false;
    }();

    if constexpr (!FixedFunction) {
      const bool forceIaTexcoordForOutlier = m_forceIaTexcoordForOutlier;

      auto getOrInitVsTexcoordTraceEntry = [&](const D3D9CommonShader* vs,
                                               const uint32_t outputReg,
                                               const uint8_t compU,
                                               const uint8_t compV) -> Ue3VsTexcoordTraceEntry* {
        if (vs == nullptr || outputReg == std::numeric_limits<uint32_t>::max())
          return nullptr;

        const XXH64_hash_t vsHash = vs->GetBytecodeHash();
        if (vsHash == 0)
          return nullptr;

        XXH64_hash_t traceKey = vsHash;
        auto mix = [&](const uint64_t v) {
          traceKey ^= v + 0x9E3779B97F4A7C15ull + (traceKey << 6) + (traceKey >> 2);
        };
        mix(outputReg);
        mix(compU);
        mix(uint64_t(compV) << 8);

        auto& entry = m_ue3VsTexcoordTraceCache[traceKey];
        if (!entry.initialized) {
          entry.initialized = true;
          const Ue3VsTexcoordTraceResult trace =
            traceVsOutputTexcoordToInputUsageIndex(vs, outputReg, compU, compV);
          entry.kind = trace.kind;
          entry.iaTexcoordIndex = trace.iaTexcoordIndex;
          entry.inputReg = trace.inputReg;
          entry.affineU = trace.affineU;
          entry.affineV = trace.affineV;
        }

        return &entry;
      };

      // resolves an affine term (imm + const*factor + const2*factor2 sum) against live
      // draw-time shader constants; an absent term resolves to its identity value
      auto resolvePsAffineTermValue = [&](const UvAffineTerm& term, const float identity, float& outValue) -> bool {
        outValue = identity;
        if (term.inexact)
          return false;
        if (!uvAffineTermPresent(term))
          return true;
        float value = term.immValid ? term.imm : 0.0f;
        if (term.constReg >= 0) {
          if (uint32_t(term.constReg) >= caps::MaxFloatConstantsPS)
            return false;
          value += d3d9State().psConsts.fConsts[uint32_t(term.constReg)][term.constComp & 0x3u] * term.factor;
        }
        if (term.constReg2 >= 0) {
          if (uint32_t(term.constReg2) >= caps::MaxFloatConstantsPS)
            return false;
          value += d3d9State().psConsts.fConsts[uint32_t(term.constReg2)][term.constComp2 & 0x3u] * term.factor2;
        }
        outValue = value;
        return std::isfinite(outValue);
      };

      auto resolveVsAffineTermValue = [&](const UvAffineTerm& term, const float identity, float& outValue) -> bool {
        outValue = identity;
        if (term.inexact)
          return false;
        if (!uvAffineTermPresent(term))
          return true;
        float value = term.immValid ? term.imm : 0.0f;
        if (term.constReg >= 0) {
          if (uint32_t(term.constReg) >= caps::MaxFloatConstantsSoftware)
            return false;
          value += d3d9State().vsConsts.fConsts[uint32_t(term.constReg)][term.constComp & 0x3u] * term.factor;
        }
        if (term.constReg2 >= 0) {
          if (uint32_t(term.constReg2) >= caps::MaxFloatConstantsSoftware)
            return false;
          value += d3d9State().vsConsts.fConsts[uint32_t(term.constReg2)][term.constComp2 & 0x3u] * term.factor2;
        }
        outValue = value;
        return std::isfinite(outValue);
      };

      if ((m_frameOptions.shaderPathTexcoordIndexFromPixelShader || m_frameOptions.ue3EngineMode) &&
          d3d9State().pixelShader.ptr() != nullptr) {
        const D3D9CommonShader* ps = inferredPs != nullptr
          ? inferredPs
          : d3d9State().pixelShader->GetCommonShader();
        XXH64_hash_t psHash = inferredPsHash;
        PsSamplerTexcoordEntry* entryPtr = inferredPsEntry;
        if (entryPtr == nullptr)
          entryPtr = getOrInitPsSamplerTexcoordEntry(ps, psHash);

        if (entryPtr != nullptr && firstStage < caps::MaxTexturesPS) {
          const auto& entry = *entryPtr;

          // A sampler whose own origin cannot be proven falls through to the fixed-function TSS
          // texcoord index below. UE3 is fully programmable and never sets that meaningfully, so
          // it is leftover device state, and a device reset restores it to the D3D9 default of
          // "stage N reads texcoord N" - the surface's UV set would change for reasons unrelated
          // to the material, presenting as the texture spontaneously rescaling.
          //
          // The shader's other material samplers shade the same surface, so where they
          // unanimously name one interpolant that agreement is proven from bytecode. Only the
          // origin is borrowed; a sibling's tiling is not this sampler's.
          PsSamplerUvOrigin borrowedUvOrigin;
          bool uvOriginBorrowed = false;
          if (!entry.samplerUvOrigin[firstStage].originValid &&
              (m_frameOptions.ue3EngineMode || m_frameOptions.shaderPathTexcoordIndexFromPixelShader)) {
            bool haveCandidate = false;
            bool candidatesConflict = false;
            uint8_t sharedSemantic = 0;
            uint8_t sharedCompU = 0;
            uint8_t sharedCompV = 1;
            for (uint32_t s = 0; s < caps::MaxTexturesPS; s++) {
              if (s == firstStage || s >= SamplerCount || d3d9State().textures[s] == nullptr)
                continue;
              const PsSamplerUvOrigin& other = entry.samplerUvOrigin[s];
              if (!other.originValid || !other.sitesAgree)
                continue;
              // Lightmap and engine buffers legitimately read their own UV set, so they must
              // not vote on the material's.
              const uint8_t semanticFlags = entry.samplerSemanticFlags[s];
              if ((semanticFlags & (kPsSamplerSemanticLightmap | kPsSamplerSemanticEngineAuxiliary)) != 0)
                continue;
              if (!haveCandidate) {
                haveCandidate = true;
                sharedSemantic = other.semanticIndex;
                sharedCompU = other.compU;
                sharedCompV = other.compV;
              } else if (other.semanticIndex != sharedSemantic ||
                         other.compU != sharedCompU ||
                         other.compV != sharedCompV) {
                candidatesConflict = true;
                break;
              }
            }
            if (haveCandidate && !candidatesConflict) {
              borrowedUvOrigin.originValid = true;
              borrowedUvOrigin.semanticIndex = sharedSemantic;
              borrowedUvOrigin.compU = sharedCompU;
              borrowedUvOrigin.compV = sharedCompV;
              uvOriginBorrowed = true;
            }
          }

          const PsSamplerUvOrigin& uvOrigin =
            uvOriginBorrowed ? borrowedUvOrigin : entry.samplerUvOrigin[firstStage];

          // rtx.d3d9.ue3LogUvAffineDetail: one-shot per-shader dump of every sampler's UV
          // origin and affine chain, with the textures bound on this draw
          if (m_frameOptions.ue3LogUvAffineDetail && ps != nullptr && psHash != 0 &&
              m_loggedUvAffineShaderDumps.insert(psHash).second) {
            const auto& samplerNames = getUe3PsSamplerNames(psHash, ps->GetBytecode());

            std::string dump;
            for (uint32_t s = 0; s < caps::MaxTexturesPS; s++) {
              const PsSamplerUvOrigin& origin = entry.samplerUvOrigin[s];
              if (origin.validSiteCount == 0 && origin.invalidSiteCount == 0)
                continue;

              XXH64_hash_t texHash = kEmptyHash;
              uint32_t texWidth = 0;
              uint32_t texHeight = 0;
              if (s < SamplerCount && d3d9State().textures[s] != nullptr) {
                D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[s]);
                if (texture != nullptr && texture->GetImage() != nullptr) {
                  texHash = texture->GetImage()->getHash();
                  texWidth = texture->Desc()->Width;
                  texHeight = texture->Desc()->Height;
                }
              }

              const auto nameIt = samplerNames.find(s);
              dump += str::format(
                "\n  s", s, "(", nameIt != samplerNames.end() ? nameIt->second.c_str() : "?", ")",
                " tex=0x", std::hex, texHash, std::dec, " ", texWidth, "x", texHeight,
                " origin=", origin.originValid ? 1 : 0,
                " interp=", uint32_t(origin.semanticIndex),
                " comps=(", uint32_t(origin.compU), ",", uint32_t(origin.compV), ")",
                " sites=", origin.validSiteCount, "/", origin.invalidSiteCount,
                " agree=", origin.sitesAgree ? 1 : 0,
                " preferHF=", origin.preferredHighestFrequencySite ? 1 : 0,
                " exact=", origin.affineExact ? 1 : 0,
                " U:[", formatUvComponentAffine(origin.affineU), "]",
                " V:[", formatUvComponentAffine(origin.affineV), "]");
            }

            Logger::info(str::format(
              "[RTX-UV-AFFINE] shader dump ps=0x", std::hex, psHash, std::dec,
              " vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
              " albedoStage=", firstStage,
              dump.empty() ? " (no traceable sample sites)" : dump.c_str()));
          }

          const D3D9CommonShader* vs =
            (m_parent->UseProgrammableVS() && d3d9State().vertexShader.ptr() != nullptr)
              ? d3d9State().vertexShader->GetCommonShader()
              : nullptr;

          const Ue3VsTexcoordTraceEntry* traceEntry = nullptr;

          if (uvOrigin.originValid) {
            texcoordIdx = uvOrigin.semanticIndex;
            m_texcoordCompU = uvOrigin.compU;
            m_texcoordCompV = uvOrigin.compV;

            uint32_t vsTexcoordOutputReg = std::numeric_limits<uint32_t>::max();
            if (vs != nullptr) {
              vsTexcoordOutputReg = findVsTexcoordOutputRegister(vs, texcoordIdx);
              if (vsTexcoordOutputReg != std::numeric_limits<uint32_t>::max())
                traceEntry = getOrInitVsTexcoordTraceEntry(vs, vsTexcoordOutputReg, m_texcoordCompU, m_texcoordCompV);
            }

            // VS-side proof: interpolant == vsAffine(IA texcoord set k)
            float vsScaleU = 1.0f;
            float vsScaleV = 1.0f;
            float vsOffsetU = 0.0f;
            float vsOffsetV = 0.0f;
            bool vsAffineFold = false;
            int32_t provenIaIndex = -1;
            if (traceEntry != nullptr) {
              if (traceEntry->kind == Ue3VsUvTraceKind::PureMove) {
                provenIaIndex = traceEntry->iaTexcoordIndex;
              } else if (traceEntry->kind == Ue3VsUvTraceKind::AffineConst) {
                // the IA set is only usable if the VS transform resolves against live constants
                if (resolveVsAffineTermValue(traceEntry->affineU.scale, 1.0f, vsScaleU) &&
                    resolveVsAffineTermValue(traceEntry->affineU.offset, 0.0f, vsOffsetU) &&
                    resolveVsAffineTermValue(traceEntry->affineV.scale, 1.0f, vsScaleV) &&
                    resolveVsAffineTermValue(traceEntry->affineV.offset, 0.0f, vsOffsetV)) {
                  provenIaIndex = traceEntry->iaTexcoordIndex;
                  vsAffineFold = true;
                }
              }
            } else if (vs == nullptr && m_texcoordCompU == 0u && m_texcoordCompV == 1u) {
              // fixed-function vertex processing: interpolant slot j is fed from the IA set
              // selected by stage j's texcoord index state
              const uint32_t ffStage = std::min(texcoordIdx, uint32_t(caps::TextureStageCount - 1u));
              provenIaIndex = int32_t(d3d9State().textureStages[ffStage][DXVK_TSS_TEXCOORDINDEX] & 0b111);
            }

            const bool canCaptureInterpolant =
              vs != nullptr &&
              m_frameOptions.useVertexCapture &&
              vsTexcoordOutputReg != std::numeric_limits<uint32_t>::max();

            if (forceIaTexcoordForOutlier) {
              // manual override via vsTexcoordCaptureOutlierTextures
              m_uvResolutionMode = UvResolutionMode::ProvenIa;
              iaTexcoordIdx = provenIaIndex >= 0 ? uint32_t(provenIaIndex) : texcoordIdx;
              m_texcoordCompU = 0;
              m_texcoordCompV = 1;
            } else if (provenIaIndex >= 0) {
              m_uvResolutionMode = UvResolutionMode::ProvenIa;
              iaTexcoordIdx = uint32_t(provenIaIndex);
            } else if (canCaptureInterpolant) {
              // the VS-side path is procedural or unprovable: capture the exact interpolant
              // from the VS output instead of guessing an IA set
              m_uvResolutionMode = UvResolutionMode::CaptureInterpolant;
              iaTexcoordIdx = (traceEntry != nullptr && traceEntry->kind != Ue3VsUvTraceKind::Invalid)
                ? traceEntry->iaTexcoordIndex  // known base set: best backstop if capture cannot run
                : 0u;
            } else {
              m_uvResolutionMode = UvResolutionMode::LegacyTss;
              iaTexcoordIdx = texcoordIdx;
            }

            // exact UV transform: sampledUv = psAffine(interpolant),
            // interpolant = vsAffine(iaUv) when the IA path is used
            float psScaleU = 1.0f;
            float psScaleV = 1.0f;
            float psOffsetU = 0.0f;
            float psOffsetV = 0.0f;
            const bool psAffineResolved =
              uvOrigin.affineExact &&
              resolvePsAffineTermValue(uvOrigin.affineU.scale, 1.0f, psScaleU) &&
              resolvePsAffineTermValue(uvOrigin.affineU.offset, 0.0f, psOffsetU) &&
              resolvePsAffineTermValue(uvOrigin.affineV.scale, 1.0f, psScaleV) &&
              resolvePsAffineTermValue(uvOrigin.affineV.offset, 0.0f, psOffsetV);
            if (!psAffineResolved) {
              // non-affine or unresolvable PS math: keep the proven base UV, apply no transform
              psScaleU = 1.0f;
              psScaleV = 1.0f;
              psOffsetU = 0.0f;
              psOffsetV = 0.0f;
            }

            float finalScaleU = psScaleU;
            float finalScaleV = psScaleV;
            float finalOffsetU = psOffsetU;
            float finalOffsetV = psOffsetV;
            if (m_uvResolutionMode == UvResolutionMode::ProvenIa && vsAffineFold) {
              // ps(vs(uv)) = (psScale * vsScale) * uv + (psScale * vsOffset + psOffset)
              finalScaleU = psScaleU * vsScaleU;
              finalScaleV = psScaleV * vsScaleV;
              finalOffsetU = psScaleU * vsOffsetU + psOffsetU;
              finalOffsetV = psScaleV * vsOffsetV + psOffsetV;
            }

            constexpr float kMinAbsScale = 1e-6f;
            const bool transformIsIdentity =
              finalScaleU == 1.0f && finalScaleV == 1.0f &&
              finalOffsetU == 0.0f && finalOffsetV == 0.0f;
            const bool transformIsUsable =
              std::isfinite(finalScaleU) && std::isfinite(finalScaleV) &&
              std::isfinite(finalOffsetU) && std::isfinite(finalOffsetV) &&
              std::abs(finalScaleU) > kMinAbsScale && std::abs(finalScaleV) > kMinAbsScale;

            if (!transformIsIdentity && transformIsUsable) {
              Matrix4& texXform = m_activeDrawCallState.transformData.textureTransform;
              texXform = Matrix4();
              texXform[0].x = finalScaleU;
              texXform[1].y = finalScaleV;
              texXform[3].x = finalOffsetU;
              texXform[3].y = finalOffsetV;
            }

            // rtx.d3d9.ue3LogUvAffineDetail: per-draw affine resolution outcome, logged once
            // per distinct resolved transform and capped per shader+stage
            if (m_frameOptions.ue3LogUvAffineDetail) {
              const bool applied = !transformIsIdentity && transformIsUsable;

              XXH64_hash_t detailKey = psHash;
              auto mixDetail = [&](const uint64_t v) {
                detailKey ^= v + 0x9E3779B97F4A7C15ull + (detailKey << 6) + (detailKey >> 2);
              };
              auto quantize = [](const float v) -> uint64_t {
                return std::isfinite(v) ? uint64_t(std::llround(double(v) * 1024.0)) : ~0ull;
              };
              mixDetail(firstStage);
              mixDetail(uint64_t(m_uvResolutionMode));
              mixDetail(quantize(finalScaleU));
              mixDetail(quantize(finalScaleV));
              mixDetail(quantize(finalOffsetU));
              mixDetail(quantize(finalOffsetV));
              mixDetail(uint64_t(psAffineResolved ? 1 : 0) | (uint64_t(applied ? 1 : 0) << 1));

              XXH64_hash_t capKey = psHash;
              capKey ^= firstStage + 0x9E3779B97F4A7C15ull + (capKey << 6) + (capKey >> 2);

              // check the cap before inserting the dedup key so frame-varying (panner)
              // transforms cannot grow the dedup set without bound once capped
              constexpr uint16_t kMaxAffineDetailLogsPerShaderStage = 32;
              uint16_t& logCount = m_uvAffineDetailLogCounts[capKey];
              if (logCount < kMaxAffineDetailLogsPerShaderStage &&
                  m_loggedUvAffineDetails.insert(detailKey).second) {
                ++logCount;
                // CTAB names + live values of the PS constant registers the affine references
                std::string ctabLog;
                if (ps != nullptr && psHash != 0) {
                  const auto& constNames = getUe3PsFloatConstantNames(psHash, ps->GetBytecode());
                  std::array<int32_t, 8> referencedRegs = {
                    uvOrigin.affineU.scale.constReg, uvOrigin.affineU.offset.constReg,
                    uvOrigin.affineV.scale.constReg, uvOrigin.affineV.offset.constReg,
                    uvOrigin.affineU.scale.constReg2, uvOrigin.affineU.offset.constReg2,
                    uvOrigin.affineV.scale.constReg2, uvOrigin.affineV.offset.constReg2 };
                  std::sort(referencedRegs.begin(), referencedRegs.end());
                  int32_t lastLogged = -1;
                  for (const int32_t reg : referencedRegs) {
                    if (reg < 0 || reg == lastLogged || uint32_t(reg) >= caps::MaxFloatConstantsPS)
                      continue;
                    lastLogged = reg;
                    const auto nameIt = constNames.find(uint32_t(reg));
                    const Vector4& value = d3d9State().psConsts.fConsts[uint32_t(reg)];
                    ctabLog += str::format(
                      ctabLog.empty() ? "" : ", ",
                      "c", reg, "=", nameIt != constNames.end() ? nameIt->second.c_str() : "?",
                      "=(", value.x, ",", value.y, ",", value.z, ",", value.w, ")");
                  }
                }

                XXH64_hash_t stageTexHash = kEmptyHash;
                uint32_t stageTexWidth = 0;
                uint32_t stageTexHeight = 0;
                if (firstStage < SamplerCount && d3d9State().textures[firstStage] != nullptr) {
                  D3D9CommonTexture* stageTexture = GetCommonTexture(d3d9State().textures[firstStage]);
                  if (stageTexture != nullptr && stageTexture->GetImage() != nullptr) {
                    stageTexHash = stageTexture->GetImage()->getHash();
                    stageTexWidth = stageTexture->Desc()->Width;
                    stageTexHeight = stageTexture->Desc()->Height;
                  }
                }

                std::string vsLog;
                if (vsAffineFold) {
                  vsLog = str::format(" vs=(", vsScaleU, ",", vsScaleV, ",", vsOffsetU, ",", vsOffsetV, ")");
                }

                Logger::info(str::format(
                  "[RTX-UV-AFFINE] ps=0x", std::hex, psHash,
                  " tex=0x", stageTexHash, std::dec, " ", stageTexWidth, "x", stageTexHeight,
                  " stage=", firstStage,
                  " vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
                  " mode=", m_uvResolutionMode == UvResolutionMode::ProvenIa
                              ? "proven-ia"
                              : m_uvResolutionMode == UvResolutionMode::CaptureInterpolant
                                  ? "capture-interpolant"
                                  : "legacy-tss",
                  " interp=", texcoordIdx,
                  " comps=(", uint32_t(m_texcoordCompU), ",", uint32_t(m_texcoordCompV), ")",
                  " sites=", uvOrigin.validSiteCount, "/", uvOrigin.invalidSiteCount,
                  " agree=", uvOrigin.sitesAgree ? 1 : 0,
                  " preferHF=", uvOrigin.preferredHighestFrequencySite ? 1 : 0,
                  " exact=", uvOrigin.affineExact ? 1 : 0,
                  " | U:[", formatUvComponentAffine(uvOrigin.affineU),
                  "] V:[", formatUvComponentAffine(uvOrigin.affineV),
                  "] | psResolved=", psAffineResolved ? 1 : 0,
                  " ps=(", psScaleU, ",", psScaleV, ",", psOffsetU, ",", psOffsetV, ")",
                  " vsFold=", vsAffineFold ? 1 : 0, vsLog,
                  " final=(", finalScaleU, ",", finalScaleV, ",", finalOffsetU, ",", finalOffsetV, ")",
                  " identity=", transformIsIdentity ? 1 : 0,
                  " usable=", transformIsUsable ? 1 : 0,
                  " applied=", applied ? 1 : 0,
                  ctabLog.empty() ? "" : str::format(" | ctab: ", ctabLog).c_str()));
              }
            }
          } else {
            // the sampled coordinate has no provable interpolant origin (screen-space,
            // reflection-driven, or untraceable): keep upstream-style TSS behavior
            m_uvResolutionMode = UvResolutionMode::LegacyTss;

            // rtx.d3d9.ue3LogUvAffineDetail: record the unprovable-origin outcome once per
            // shader+stage - the transform can never apply on this path
            if (m_frameOptions.ue3LogUvAffineDetail) {
              XXH64_hash_t noOriginKey = psHash;
              noOriginKey ^= (0xA11FE00Dull + firstStage) + 0x9E3779B97F4A7C15ull +
                             (noOriginKey << 6) + (noOriginKey >> 2);
              if (m_loggedUvAffineDetails.insert(noOriginKey).second) {
                Logger::info(str::format(
                  "[RTX-UV-AFFINE] ps=0x", std::hex, psHash,
                  " tex=0x", m_activeDrawCallState.materialData.colorTextures[0].getImageHash(), std::dec,
                  " stage=", firstStage,
                  " vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
                  " originValid=0 sites=", uvOrigin.validSiteCount, "/", uvOrigin.invalidSiteCount,
                  " - no provable interpolant origin; no texture transform derived (legacy TSS path)"));
              }
            }
          }

          if (m_frameOptions.ue3LogUvResolution || Logger::logLevel() <= LogLevel::Debug) {
            const XXH64_hash_t colorTextureHash =
              m_activeDrawCallState.materialData.colorTextures[0].getImageHash();

            XXH64_hash_t logKey = psHash;
            auto mixLog = [&](const uint64_t v) {
              logKey ^= v + 0x9E3779B97F4A7C15ull + (logKey << 6) + (logKey >> 2);
            };
            mixLog(firstStage);
            mixLog(uint64_t(m_uvResolutionMode));
            mixLog(texcoordIdx);
            mixLog(iaTexcoordIdx);
            mixLog((uint64_t(m_texcoordCompU) << 8) | uint64_t(m_texcoordCompV));
            mixLog(colorTextureHash);

            if (m_loggedUvResolutions.insert(logKey).second) {
              const char* modeName = "legacy-tss";
              if (m_uvResolutionMode == UvResolutionMode::ProvenIa) {
                modeName = "proven-ia";
              } else if (m_uvResolutionMode == UvResolutionMode::CaptureInterpolant) {
                modeName = "capture-interpolant";
              }

              const char* traceKindName = "none";
              if (traceEntry != nullptr) {
                switch (traceEntry->kind) {
                case Ue3VsUvTraceKind::PureMove: traceKindName = "pure-move"; break;
                case Ue3VsUvTraceKind::AffineConst: traceKindName = "affine-const"; break;
                case Ue3VsUvTraceKind::OriginOnly: traceKindName = "origin-only"; break;
                default: traceKindName = "invalid"; break;
                }
              }

              const char* siteDisagreementNote = "";
              if (!uvOrigin.sitesAgree) {
                siteDisagreementNote = uvOrigin.preferredHighestFrequencySite
                  ? " [sites disagree: preferred highest-frequency tiling]"
                  : " [AMBIGUOUS: sample sites disagree]";
              }

              const std::string msg = str::format(
                "[RTX-UV] ", modeName,
                ": ps=0x", std::hex, psHash,
                ", tex=0x", colorTextureHash, std::dec,
                ", stage=", firstStage,
                ", originValid=", uvOrigin.originValid,
                ", interpolantTexcoord=", texcoordIdx,
                ", comps=(", uint32_t(m_texcoordCompU), ",", uint32_t(m_texcoordCompV), ")",
                ", iaSet=", iaTexcoordIdx,
                ", vsTrace=", traceKindName,
                ", sites=", uvOrigin.validSiteCount, " valid/", uvOrigin.invalidSiteCount, " invalid",
                siteDisagreementNote,
                uvOrigin.affineExact ? "" : " [affine-inexact]",
                uvOriginBorrowed ? " [origin borrowed from sibling material samplers]" : "",
                m_forceIaTexcoordForOutlier ? " [outlier-override]" : "");
              if (m_frameOptions.ue3LogUvResolution) {
                Logger::info(msg);
              } else {
                Logger::debug(msg);
              }
            }
          }
        }
      }
    }

    m_texcoordIndex = texcoordIdx;
    m_iaTexcoordIndex = resolveIaTexcoordAvoidingNonUvElements(iaTexcoordIdx);

    return true;
  }

  // Two kinds of UE3 TEXCOORD element are not coordinates at all:
  //  - vertex lightmap policies append packed lighting coefficients as TEXCOORD5 (simple) or
  //    TEXCOORD5/6/7 (directional), typed D3DCOLOR rather than a float pair. The element count
  //    moves with the DirectionalLightmaps setting, and a D3DCOLOR-typed TEXCOORD is never a UV
  //    set in UE3, so the type alone identifies them without consulting the vertex shader.
  //  - instanced vertex factories put the per-instance basis in TEXCOORD1..4 on an instance-data
  //    stream. Those advance once per instance, so they carry no per-vertex meaning.
  uint32_t D3D9Rtx::resolveIaTexcoordAvoidingNonUvElements(const uint32_t iaTexcoordIdx) const {
    if (!m_frameOptions.ue3EngineMode || d3d9State().vertexDecl == nullptr)
      return iaTexcoordIdx;

    const uint32_t instanceDataStreamMask = m_currentUe3Instancing.instanceDataStreamMask;
    auto isNonUvElement = [instanceDataStreamMask](const D3DVERTEXELEMENT9& element) {
      if (element.Usage != D3DDECLUSAGE_TEXCOORD)
        return false;
      if (element.Type == D3DDECLTYPE_D3DCOLOR)
        return true;
      return element.Stream < caps::MaxStreams &&
             (instanceDataStreamMask & (1u << element.Stream)) != 0;
    };

    const auto& elements = d3d9State().vertexDecl->GetElements();
    bool requestedIsNonUv = false;
    for (const auto& element : elements) {
      if (element.Usage == D3DDECLUSAGE_TEXCOORD && element.UsageIndex == iaTexcoordIdx) {
        requestedIsNonUv = isNonUvElement(element);
        break;
      }
    }

    if (!requestedIsNonUv)
      return iaTexcoordIdx;

    // Fall back to the mesh's lowest real UV set rather than leaving the surface with the
    // lighting stream: an unresolvable index would drop the texcoord buffer entirely.
    uint32_t fallback = iaTexcoordIdx;
    bool found = false;
    for (const auto& element : elements) {
      if (element.Usage != D3DDECLUSAGE_TEXCOORD || isNonUvElement(element))
        continue;
      if (!found || element.UsageIndex < fallback) {
        fallback = element.UsageIndex;
        found = true;
      }
    }

    return fallback;
  }

  PrepareDrawFlags D3D9Rtx::PrepareDrawGeometryForRT(const bool indexed, const DrawContext& context) {
    // Draws issued internally by the deferred UI overlay replay bypass classification and
    // execute as plain raster draws
    if (m_replayingDeferredUiDraws) {
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    // first-frame lazy init; steady-state refreshes happen once per frame in EndFrame
    if (unlikely(!m_frameOptions.valid)) {
      refreshFrameOptionCache();
    }

    if (!m_frameOptions.enableRaytracing || !m_enableDrawCallConversion) {
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    m_parent->PrepareTextures();

    IndexContext indices;
    if (indexed) {
      D3D9CommonBuffer* ibo = GetCommonBuffer(d3d9State().indices);
      assert(ibo != nullptr);

      indices.ibo = ibo;
      indices.indexBuffer = ibo->GetMappedSlice();
      indices.indexType = DecodeIndexType(ibo->Desc()->Format);
    }

    // Copy over the vertex buffers that are actually required
    VertexContext vertices[caps::MaxStreams];
    for (uint32_t i = 0; i < caps::MaxStreams; i++) {
      const auto& dx9Vbo = d3d9State().vertexBuffers[i];
      auto* vbo = GetCommonBuffer(dx9Vbo.vertexBuffer);
      if (vbo != nullptr) {
        vertices[i].stride = dx9Vbo.stride;
        vertices[i].offset = dx9Vbo.offset;
        vertices[i].buffer = vbo->GetBufferSlice<D3D9_COMMON_BUFFER_TYPE_MAPPING>();
        vertices[i].mappedSlice = vbo->GetMappedSlice();
        vertices[i].pVBO = vbo;

        // If staging upload has been enabled on a buffer then previous buffer lock:
        //   a) triggered a pipeline stall (overlapped mapped ranges, improper flags etc)
        //   b) does not have D3DLOCK_DONOTWAIT, or was in use at Map()
        // 
        // Buffers with staged uploads may have contents valid ONLY until next Map().
        // We must NOT use such buffer directly and have to always copy the contents.
        vertices[i].canUseBuffer = vbo->DoesStagingBufferUploads() == false;
      }
    }

    return internalPrepareDraw(indices, vertices, context);
  }

  PrepareDrawFlags D3D9Rtx::PrepareDrawUPGeometryForRT(const bool indexed,
                                                       const D3D9BufferSlice& buffer,
                                                       const D3DFORMAT indexFormat,
                                                       const uint32_t indexSize,
                                                       const uint32_t indexOffset,
                                                       const uint32_t vertexSize,
                                                       const uint32_t vertexStride,
                                                       const DrawContext& drawContext) {
    // Draws issued internally by the deferred UI overlay replay bypass classification and
    // execute as plain raster draws
    if (m_replayingDeferredUiDraws) {
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    // first-frame lazy init; steady-state refreshes happen once per frame in EndFrame
    if (unlikely(!m_frameOptions.valid)) {
      refreshFrameOptionCache();
    }

    if (!m_frameOptions.enableRaytracing || !m_enableDrawCallConversion) {
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    m_parent->PrepareTextures();

    // 'buffer' - contains vertex + index data (packed in that order)

    IndexContext indices;
    if (indexed) {
      indices.indexBuffer = buffer.slice.getSliceHandle(indexOffset, indexSize);
      indices.indexType = DecodeIndexType(static_cast<D3D9Format>(indexFormat));
    }

    VertexContext vertices[caps::MaxStreams];
    vertices[0].stride = vertexStride;
    vertices[0].offset = 0;
    vertices[0].buffer = buffer.slice.subSlice(0, vertexSize);
    vertices[0].mappedSlice = buffer.slice.getSliceHandle(0, vertexSize);
    vertices[0].canUseBuffer = true;

    return internalPrepareDraw(indices, vertices, drawContext);
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

    // Inform the backend about potential presenter update
    m_parent->EmitCs([cWidth = m_activePresentParams->BackBufferWidth,
                      cHeight = m_activePresentParams->BackBufferHeight](DxvkContext* ctx) {
      static_cast<RtxContext*>(ctx)->resetScreenResolution({ cWidth, cHeight , 1 });
    });
  }

  namespace {
    constexpr char kUe3TextureSpreadCachePath[] = "rtx-remix/ue3TextureSpread.cache";
    constexpr char kUe3TextureSpreadCacheTempPath[] = "rtx-remix/ue3TextureSpread.cache.tmp";
    constexpr uint64_t kUe3TextureSpreadCacheMagic = 0x3144525053334555ull; // "UE3SPRD1"
    constexpr uint32_t kUe3TextureSpreadCacheMaxEntries = 1u << 20;
    // Both learned caches are rewritten whole, so they are saved on an interval rather than on
    // every change; the destructor flushes whatever the last interval missed.
    constexpr uint32_t kUe3CacheSaveIntervalFrames = 600;

    constexpr char kUe3DiffuseSelectionCachePath[] = "rtx-remix/ue3DiffuseSelection.cache";
    constexpr char kUe3DiffuseSelectionCacheTempPath[] = "rtx-remix/ue3DiffuseSelection.cache.tmp";
    constexpr uint64_t kUe3DiffuseSelectionCacheMagic = 0x324C455344334555ull; // "UE3DSEL2"
    constexpr uint32_t kUe3DiffuseSelectionCacheMaxEntries = 1u << 20;
    // Stored picks are only meaningful under the scoring that produced them. Bump this whenever
    // the albedo score changes, so a build with different scoring re-derives instead of serving
    // decisions its own scoring would no longer make.
    constexpr uint32_t kUe3DiffuseSelectionScoringVersion = 2;
    // How many loaded picks to re-score and check against current scoring per session. A scoring
    // change disagrees broadly, so a small sample finds one; the cost is bounded to that many draws.
    constexpr uint32_t kUe3DiffuseSelectionAuditCount = 32;
  }

  void D3D9Rtx::loadUe3DiffuseSelectionCache() {
    m_ue3DiffuseSelectionLoaded = true;

    auto refuseFutureSaves = [this](const std::string& reason) {
      m_ue3DiffuseSelectionSaveBlocked = true;
      Logger::warn(str::format(
        "[RTX-Compatibility][UE3] Albedo selection cache ", reason,
        ". Leaving the file untouched for this session; albedo picks re-derive as materials are "
        "first drawn. Delete ", kUe3DiffuseSelectionCachePath, " to start a fresh one."));
    };

    const bool fileExists = ue3CacheFileExists(kUe3DiffuseSelectionCachePath);

    std::ifstream file(kUe3DiffuseSelectionCachePath, std::ios::binary);
    if (!file.is_open()) {
      if (fileExists)
        refuseFutureSaves("exists but could not be opened");
      return;
    }

    // Scoring reads the taggable texture sets, so a stored pick is only valid under the tags that
    // produced it. Adopting the current sizes here is also what stops the tag-change check on the
    // first scoring draw from comparing against zero and clearing everything just loaded.
    const uint32_t currentLightmapSize = uint32_t(m_frameOptions.lightmapTextures->size());
    const uint32_t currentNeverAlbedoSize = uint32_t(m_frameOptions.neverAlbedoTextures->size());
    const uint32_t currentPreferredAlbedoSize = uint32_t(m_frameOptions.preferredAlbedoTextures->size());
    m_ue3DiffuseSelectionLightmapSetSize = currentLightmapSize;
    m_ue3DiffuseSelectionNeverAlbedoSetSize = currentNeverAlbedoSize;
    m_ue3DiffuseSelectionPreferredAlbedoSetSize = currentPreferredAlbedoSize;

    uint64_t magic = 0;
    uint32_t scoringVersion = 0;
    uint32_t lightmapSize = 0;
    uint32_t neverAlbedoSize = 0;
    uint32_t preferredAlbedoSize = 0;
    uint32_t entryCount = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&scoringVersion), sizeof(scoringVersion));
    file.read(reinterpret_cast<char*>(&lightmapSize), sizeof(lightmapSize));
    file.read(reinterpret_cast<char*>(&neverAlbedoSize), sizeof(neverAlbedoSize));
    file.read(reinterpret_cast<char*>(&preferredAlbedoSize), sizeof(preferredAlbedoSize));
    file.read(reinterpret_cast<char*>(&entryCount), sizeof(entryCount));
    if (!file || magic != kUe3DiffuseSelectionCacheMagic || entryCount > kUe3DiffuseSelectionCacheMaxEntries) {
      refuseFutureSaves("header is unreadable or not recognised");
      return;
    }
    if (scoringVersion != kUe3DiffuseSelectionScoringVersion) {
      // Our own file, written by different scoring: discard and rewrite rather than refuse.
      Logger::info(str::format(
        "[RTX-Compatibility][UE3] Albedo selection cache was written by scoring version ",
        scoringVersion, " (this build is ", kUe3DiffuseSelectionScoringVersion,
        "); re-deriving picks and rewriting it."));
      m_ue3DiffuseSelectionDirty = true;
      return;
    }
    if (lightmapSize != currentLightmapSize ||
        neverAlbedoSize != currentNeverAlbedoSize ||
        preferredAlbedoSize != currentPreferredAlbedoSize) {
      Logger::info(str::format(
        "[RTX-Compatibility][UE3] Albedo selection cache was written under different texture tags "
        "(lightmap/never/preferred ", lightmapSize, "/", neverAlbedoSize, "/", preferredAlbedoSize,
        ", now ", currentLightmapSize, "/", currentNeverAlbedoSize, "/", currentPreferredAlbedoSize,
        "); re-deriving picks and rewriting it."));
      m_ue3DiffuseSelectionDirty = true;
      return;
    }

    for (uint32_t i = 0; i < entryCount; i++) {
      XXH64_hash_t key = 0;
      Ue3DiffuseSelectionEntry entry;
      file.read(reinterpret_cast<char*>(&key), sizeof(key));
      file.read(reinterpret_cast<char*>(&entry.chosenStages[0]), sizeof(entry.chosenStages[0]));
      file.read(reinterpret_cast<char*>(&entry.chosenStages[1]), sizeof(entry.chosenStages[1]));
      file.read(reinterpret_cast<char*>(&entry.cubemapFallbackStage), sizeof(entry.cubemapFallbackStage));
      file.read(reinterpret_cast<char*>(&entry.decisionAreaSum), sizeof(entry.decisionAreaSum));
      if (!file) {
        refuseFutureSaves(str::format("is truncated: ", i, " of ", entryCount, " entries readable"));
        return;
      }
      entry.fromDisk = true;
      m_ue3DiffuseSelectionCache[key] = entry;
    }

    m_ue3DiffuseSelectionAuditsRemaining = kUe3DiffuseSelectionAuditCount;

    Logger::info(str::format(
      "[RTX-Compatibility][UE3] Loaded albedo selection cache: ", entryCount, " materials"));
  }

  void D3D9Rtx::saveUe3DiffuseSelectionCache() {
    if (!m_ue3DiffuseSelectionDirty || m_ue3DiffuseSelectionSaveBlocked)
      return;

    {
      std::ofstream file(kUe3DiffuseSelectionCacheTempPath, std::ios::binary | std::ios::trunc);
      if (!file.is_open())
        return;

      const uint32_t entryCount =
        uint32_t(std::min<size_t>(m_ue3DiffuseSelectionCache.size(), kUe3DiffuseSelectionCacheMaxEntries));
      // The tracked sizes rather than the live sets: these are what the stored decisions were
      // scored against, and they are readable from the destructor, where the frame options a
      // draw would have populated may never have existed.
      const uint32_t lightmapSize = uint32_t(m_ue3DiffuseSelectionLightmapSetSize);
      const uint32_t neverAlbedoSize = uint32_t(m_ue3DiffuseSelectionNeverAlbedoSetSize);
      const uint32_t preferredAlbedoSize = uint32_t(m_ue3DiffuseSelectionPreferredAlbedoSetSize);
      file.write(reinterpret_cast<const char*>(&kUe3DiffuseSelectionCacheMagic), sizeof(kUe3DiffuseSelectionCacheMagic));
      file.write(reinterpret_cast<const char*>(&kUe3DiffuseSelectionScoringVersion), sizeof(kUe3DiffuseSelectionScoringVersion));
      file.write(reinterpret_cast<const char*>(&lightmapSize), sizeof(lightmapSize));
      file.write(reinterpret_cast<const char*>(&neverAlbedoSize), sizeof(neverAlbedoSize));
      file.write(reinterpret_cast<const char*>(&preferredAlbedoSize), sizeof(preferredAlbedoSize));
      file.write(reinterpret_cast<const char*>(&entryCount), sizeof(entryCount));

      uint32_t written = 0;
      for (const auto& entry : m_ue3DiffuseSelectionCache) {
        if (written >= entryCount)
          break;
        file.write(reinterpret_cast<const char*>(&entry.first), sizeof(entry.first));
        file.write(reinterpret_cast<const char*>(&entry.second.chosenStages[0]), sizeof(entry.second.chosenStages[0]));
        file.write(reinterpret_cast<const char*>(&entry.second.chosenStages[1]), sizeof(entry.second.chosenStages[1]));
        file.write(reinterpret_cast<const char*>(&entry.second.cubemapFallbackStage), sizeof(entry.second.cubemapFallbackStage));
        file.write(reinterpret_cast<const char*>(&entry.second.decisionAreaSum), sizeof(entry.second.decisionAreaSum));
        written++;
      }

      file.close();
      if (!file)
        return;
    }

    if (ue3CommitCacheFile(kUe3DiffuseSelectionCacheTempPath, kUe3DiffuseSelectionCachePath))
      m_ue3DiffuseSelectionDirty = false;
  }

  void D3D9Rtx::loadUe3TextureSpreadCache() {
    m_ue3TextureSpreadLoaded = true;

    auto refuseFutureSaves = [this](const std::string& reason) {
      m_ue3TextureSpreadSaveBlocked = true;
      Logger::warn(str::format(
        "[RTX-Compatibility][UE3] Texture material-spread cache ", reason,
        ". Leaving the file untouched for this session so it is not overwritten with less "
        "than it holds; albedo scoring falls back to no spread data. Delete ",
        kUe3TextureSpreadCachePath, " to start a fresh one."));
    };

    const bool fileExists = ue3CacheFileExists(kUe3TextureSpreadCachePath);

    std::ifstream file(kUe3TextureSpreadCachePath, std::ios::binary);
    if (!file.is_open()) {
      // No file at all is an ordinary first run, and saving must stay available to create one.
      if (fileExists)
        refuseFutureSaves("exists but could not be opened");
      return;
    }

    uint64_t magic = 0;
    uint32_t entryCount = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&entryCount), sizeof(entryCount));
    if (!file || magic != kUe3TextureSpreadCacheMagic || entryCount > kUe3TextureSpreadCacheMaxEntries) {
      refuseFutureSaves("header is unreadable or not recognised");
      return;
    }

    for (uint32_t i = 0; i < entryCount; i++) {
      XXH64_hash_t texHash = 0;
      uint8_t count = 0;
      file.read(reinterpret_cast<char*>(&texHash), sizeof(texHash));
      file.read(reinterpret_cast<char*>(&count), sizeof(count));
      Ue3TextureMaterialSpread spread;
      if (!file || count > spread.psHashes.size()) {
        refuseFutureSaves(str::format("is truncated: ", i, " of ", entryCount, " entries readable"));
        return;
      }
      for (uint8_t p = 0; p < count; p++)
        file.read(reinterpret_cast<char*>(&spread.psHashes[p]), sizeof(XXH64_hash_t));
      if (!file) {
        refuseFutureSaves(str::format("is truncated: ", i, " of ", entryCount, " entries readable"));
        return;
      }
      spread.count = count;
      spread.scoringCount = count;
      m_ue3TextureMaterialSpread[texHash] = spread;
    }

    Logger::info(str::format(
      "[RTX-Compatibility][UE3] Loaded texture material-spread cache: ", entryCount, " textures"));
  }

  void D3D9Rtx::saveUe3TextureSpreadCache() {
    if (!m_ue3TextureSpreadDirty || m_ue3TextureSpreadSaveBlocked)
      return;

    {
      std::ofstream file(kUe3TextureSpreadCacheTempPath, std::ios::binary | std::ios::trunc);
      if (!file.is_open())
        return;

      const uint32_t entryCount =
        uint32_t(std::min<size_t>(m_ue3TextureMaterialSpread.size(), kUe3TextureSpreadCacheMaxEntries));
      file.write(reinterpret_cast<const char*>(&kUe3TextureSpreadCacheMagic), sizeof(kUe3TextureSpreadCacheMagic));
      file.write(reinterpret_cast<const char*>(&entryCount), sizeof(entryCount));

      uint32_t written = 0;
      for (const auto& entry : m_ue3TextureMaterialSpread) {
        if (written >= entryCount)
          break;
        file.write(reinterpret_cast<const char*>(&entry.first), sizeof(entry.first));
        file.write(reinterpret_cast<const char*>(&entry.second.count), sizeof(entry.second.count));
        for (uint8_t p = 0; p < entry.second.count; p++)
          file.write(reinterpret_cast<const char*>(&entry.second.psHashes[p]), sizeof(XXH64_hash_t));
        written++;
      }

      file.close();
      if (!file)
        return;
    }

    // Cleared only once the new file is in place, so a failed write is retried next interval.
    if (ue3CommitCacheFile(kUe3TextureSpreadCacheTempPath, kUe3TextureSpreadCachePath))
      m_ue3TextureSpreadDirty = false;
  }

  void D3D9Rtx::recordOcclusionQueryBracketedDraw(const VertexContext vertexContext[caps::MaxStreams],
                                                  const DrawContext& drawContext) {
    ++m_oqDiag.bracketedDraws;
    ++m_oqDiag.drawsInCurrentBracket;

    auto& record = m_oqRecords[m_oqBracketCounter & (kOcclusionQueryRecordCount - 1)];
    if (record.bracketId != m_oqBracketCounter) {
      // Bracket began before logging was enabled; initialise the slot late.
      record = {};
      record.bracketId = m_oqBracketCounter;
    }

    const auto& rs = d3d9State().renderStates;
    ++record.drawCount;
    record.primCount = uint16_t(std::min<UINT>(drawContext.PrimitiveCount, 0xFFFFu));
    record.viewportW = uint16_t(std::min<DWORD>(d3d9State().viewport.Width, 0xFFFFu));
    record.viewportH = uint16_t(std::min<DWORD>(d3d9State().viewport.Height, 0xFFFFu));
    record.conservativeActive = ConservativeOcclusionQueriesEnabled();

    // UE3 occlusion boxes are position-only world-space vertices in stream 0. Record their world
    // AABB and whether the view origin (UE3 convention: register c4, set by the view setup and
    // persisting across the query draws) sits inside it - a camera-enclosing box is fully
    // back-face culled when measured and guaranteed to report 0 samples.
    const VertexContext& vtx = vertexContext[0];
    const uint8_t* vertexData = reinterpret_cast<const uint8_t*>(vtx.mappedSlice.mapPtr);
    if (vertexData != nullptr && vtx.stride >= sizeof(float) * 3 && drawContext.NumVertices > 0) {
      constexpr uint32_t kMaxAnalyzedVertices = 64;
      const uint32_t vertexCount = std::min<uint32_t>(drawContext.NumVertices, kMaxAnalyzedVertices);

      Vector3 boxMin(FLT_MAX, FLT_MAX, FLT_MAX);
      Vector3 boxMax(-FLT_MAX, -FLT_MAX, -FLT_MAX);
      bool anyVertex = false;
      for (uint32_t i = 0; i < vertexCount; i++) {
        const size_t byteOffset = size_t(vtx.offset) + size_t(drawContext.MinVertexIndex + i) * vtx.stride;
        if (byteOffset + sizeof(float) * 3 > vtx.mappedSlice.length) {
          break;
        }
        const float* p = reinterpret_cast<const float*>(vertexData + byteOffset);
        boxMin = Vector3(std::min(boxMin.x, p[0]), std::min(boxMin.y, p[1]), std::min(boxMin.z, p[2]));
        boxMax = Vector3(std::max(boxMax.x, p[0]), std::max(boxMax.y, p[1]), std::max(boxMax.z, p[2]));
        anyVertex = true;
      }

      if (anyVertex) {
        if (record.drawCount == 1) {
          record.boxMin = boxMin;
          record.boxMax = boxMax;
        } else {
          record.boxMin = Vector3(std::min(record.boxMin.x, boxMin.x), std::min(record.boxMin.y, boxMin.y), std::min(record.boxMin.z, boxMin.z));
          record.boxMax = Vector3(std::max(record.boxMax.x, boxMax.x), std::max(record.boxMax.y, boxMax.y), std::max(record.boxMax.z, boxMax.z));
        }

        const Vector3 cameraPos = d3d9State().vsConsts.fConsts[4].xyz();
        record.cameraPos = cameraPos;
        record.cameraInsideBox =
          cameraPos.x >= record.boxMin.x && cameraPos.x <= record.boxMax.x &&
          cameraPos.y >= record.boxMin.y && cameraPos.y <= record.boxMax.y &&
          cameraPos.z >= record.boxMin.z && cameraPos.z <= record.boxMax.z;

        const Vector3 closestPoint(
          std::clamp(cameraPos.x, record.boxMin.x, record.boxMax.x),
          std::clamp(cameraPos.y, record.boxMin.y, record.boxMax.y),
          std::clamp(cameraPos.z, record.boxMin.z, record.boxMax.z));
        record.cameraToBoxDistance = length(cameraPos - closestPoint);
      }
    }

    if (m_oqDiag.stateSnapshotLogsRemaining == 0) {
      return;
    }
    --m_oqDiag.stateSnapshotLogsRemaining;

    Logger::info(str::format(
      "[UE3-OQ] bracketed draw: prims=", drawContext.PrimitiveCount,
      " indexed=", drawContext.Indexed ? 1 : 0,
      " zEnable=", rs[D3DRS_ZENABLE],
      " zFunc=", rs[D3DRS_ZFUNC],
      " zWrite=", rs[D3DRS_ZWRITEENABLE],
      " stencil=", rs[D3DRS_STENCILENABLE],
      " alphaTest=", rs[D3DRS_ALPHATESTENABLE],
      " alphaBlend=", rs[D3DRS_ALPHABLENDENABLE],
      " colorWrite=0x", std::hex, rs[ColorWriteIndex(kRenderTargetIndex)], std::dec,
      " clipPlanes=0x", std::hex, rs[D3DRS_CLIPPLANEENABLE], std::dec,
      " scissor=", rs[D3DRS_SCISSORTESTENABLE],
      " cull=", rs[D3DRS_CULLMODE],
      " viewport=", d3d9State().viewport.X, ",", d3d9State().viewport.Y,
      " ", d3d9State().viewport.Width, "x", d3d9State().viewport.Height,
      " dsBound=", d3d9State().depthStencil != nullptr ? 1 : 0,
      " programmableVS=", m_parent->UseProgrammableVS() ? 1 : 0,
      " programmablePS=", m_parent->UseProgrammablePS() ? 1 : 0,
      " cameraInside=", record.cameraInsideBox ? 1 : 0,
      " camDistToBox=", record.cameraToBoxDistance,
      " conservative=", record.conservativeActive ? 1 : 0));
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
    // Refresh the per-frame option snapshot: EndFrame's own consumers (deferred UI
    // replay) read fresh values and the next frame's draws see this frame's resolution.
    refreshFrameOptionCache();

    // Allow the next frame's TdToneMapping pass to be captured again
    m_ue3ToneMapCapturedThisFrame = false;

    // New frame: forget the UE3 foreground DPG segment
    m_ue3SeenMainViewWorldDraw = false;
    m_ue3ForegroundDpgActive = false;

    if (m_frameOptions.ue3LogOcclusionQueries) {
      flushOcclusionQueryDiagnostics();
    }

    // Flush this frame's replacement-material-hash tracking as one CS command. Must be
    // emitted before the endFrame command below: the consumers (graph components) read
    // the per-frame map during SceneManager::onFrameEnd, and the map clears there too.
    if (!m_pendingReplacementMaterialHashes.empty()) {
      const size_t flushedCount = m_pendingReplacementMaterialHashes.size();
      m_parent->EmitCs([cHashes = std::move(m_pendingReplacementMaterialHashes)](DxvkContext* ctx) {
        SceneManager& sceneManager = static_cast<RtxContext*>(ctx)->getSceneManager();
        for (const XXH64_hash_t hash : cHashes) {
          sceneManager.trackReplacementMaterialHash(hash);
        }
      });
      // the move donates the capacity to the lambda; re-establish a defined empty state
      // and pre-size for the next frame's roughly equal draw count
      m_pendingReplacementMaterialHashes.clear();
      m_pendingReplacementMaterialHashes.reserve(flushedCount);
    }

    const auto currentReflexFrameId = GetReflexFrameId();

    // persist what this session learned so the next one scores from it rather than re-converging
    if (m_ue3TextureSpreadDirty &&
        currentReflexFrameId >= m_ue3TextureSpreadLastSaveFrame + kUe3CacheSaveIntervalFrames) {
      m_ue3TextureSpreadLastSaveFrame = uint32_t(currentReflexFrameId);
      saveUe3TextureSpreadCache();
    }

    if (m_ue3DiffuseSelectionDirty &&
        currentReflexFrameId >= m_ue3DiffuseSelectionLastSaveFrame + kUe3CacheSaveIntervalFrames) {
      m_ue3DiffuseSelectionLastSaveFrame = uint32_t(currentReflexFrameId);
      saveUe3DiffuseSelectionCache();
    }


    // Flush any pending game and RTX work
    m_parent->Flush();

    // Inform backend of end-frame
    m_parent->EmitCs([currentReflexFrameId, targetImage, callInjectRtx](DxvkContext* ctx) { 
      static_cast<RtxContext*>(ctx)->endFrame(currentReflexFrameId, targetImage, callInjectRtx); 
    });

    // Replay any deferred overlays that no mid-frame injection flushed. Typically this means
    // no trigger draw fired this frame and the endFrame call above performs the fallback
    // injection onto the backbuffer; the overlays then composite on top of that blit.
    if (!m_deferredUiDraws.empty()) {
      if (callInjectRtx) {
        Com<IDirect3DSurface9> backBuffer;
        if (SUCCEEDED(m_parent->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &backBuffer)) && backBuffer != nullptr) {
          replayDeferredUiDraws(backBuffer.ptr(), targetImage);
        } else {
          m_deferredUiDraws.clear();
        }
      } else {
        // Not presenting normally (e.g. alt-tab end-of-frame events): drop leftovers
        m_deferredUiDraws.clear();
      }
    }
    m_deferredUiFrameVertexBytes = 0;

    pruneUe3StaticVertexCaptureCache();
    pruneUe3GeometryMemoCache();
    updateUe3StaticVertexCaptureCacheState();
    // Independent of the capture cache, whose state update returns early when it is disabled.
    reportUe3InstancedDrawStats();
    reportUe3ConstantChurn();

    DrawCallState::refreshCategoryLookupTable();

    // Reset for the next frame
    m_rtxInjectTriggered = false;
    m_drawCallID = 0;
    m_seenCameraPositionsPrev = std::move(m_seenCameraPositions);
    ++m_ue3FrameCounter;

    // two-pass translucency dedup state must not span frames
    m_prevDrawVsPsHash = 0;
    m_prevDrawTextureHash = 0;
    m_prevDrawGeometryHash = 0;
    m_prevDrawCullMode = 0;

    m_stagedBones.clear();
  }

  void D3D9Rtx::OnPresent(const Rc<DxvkImage>& targetImage) {
    // Inform backend of present
    m_parent->EmitCs([targetImage](DxvkContext* ctx) { static_cast<RtxContext*>(ctx)->onPresent(targetImage); });
  }
}
