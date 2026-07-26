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
#include "../dxvk/rtx_render/rtx_ngx_passthrough.h"
#include "../dxvk/rtx_render/rtx_options.h"
#include "../dxvk/rtx_render/rtx_dlfg.h"
#include "../dxvk/rtx_render/rtx_camera.h"
#include "../dxvk/imgui/dxvk_imgui.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

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

    // Canonical shader signature serialization, shared by the CTAB parser and the
    // lightmap-permutation bridge so their digests stay byte-identical. Entries carry
    // names and register classes only: register indices shift with lightmap sampler
    // counts and register counts are fxc's *used* element counts, which vary per
    // permutation - neither may leak into the permutation-invariant signature.
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
      // shader references lightmap policy symbols (LightMapTextures/LightMapScale/
      // LightMapResolution, BSplineTexture), so its bytecode identity varies with the
      // DirectionalLightmaps / TdBicubicFiltering system settings
      bool hasLightmapPermutationSymbols = false;
      bool hasCtab = false;
    };

    static Ue3PsMaterialIdentityInfo parseUe3PsMaterialIdentityFromCtab(const std::vector<uint8_t>& bytecode) {
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

      std::vector<std::pair<std::string, uint32_t>> uniformsByName;
      std::vector<std::string> samplerSignatureEntries;
      std::vector<std::string> uniformSignatureEntries;

      for (const DxsoCtab::Constant& c : ctab.m_constantData) {
        const std::string name = toLowerAscii(c.name);

        // lightmap policy symbols: LightMapTextures / LightMapScale / LightMapResolution and the
        // Mirror's Edge bicubic B-spline weights LUT. Their presence marks the shader as a
        // DirectionalLightmaps / TdBicubicFiltering permutation.
        if (name.find("lightmap") != std::string::npos || name.find("bspline") != std::string::npos) {
          info.hasLightmapPermutationSymbols = true;
        }

        if (c.registerSet == kD3dxRegisterSetSampler) {
          // strict prefix rule: only the numbered sampler names emitted by the UE3 material
          // translator count as material texture parameters; lightmaps/scene/shadow samplers
          // use other names and must stay out of the material identity
          if (c.registerCount != 0 &&
              (startsWith(name, "texture2d_") || startsWith(name, "texturecube_") || startsWith(name, "texture3d_"))) {
            const uint32_t end = std::min<uint32_t>(c.registerIndex + c.registerCount, caps::MaxTexturesPS);
            for (uint32_t s = c.registerIndex; s < end; s++) {
              info.materialSamplerMask |= (1u << s);
              // arrays get per-register names so name keys stay unambiguous (material samplers
              // are scalar in practice, this is defensive)
              const std::string samplerName =
                c.registerCount > 1u ? str::format(name, "[", s - c.registerIndex, "]") : name;
              info.materialSamplersByNameOrder.emplace_back(
                samplerName, XXH3_64bits(samplerName.data(), samplerName.size()), s);
            }
            samplerSignatureEntries.push_back(makeUe3SignatureEntry(name, c.registerSet));
          }
          continue;
        }

        const bool isUniformVector = name.find("uniformvector_") != std::string::npos;
        const bool isUniformScalar = name.find("uniformscalar_") != std::string::npos;
        if (!isUniformVector && !isUniformScalar)
          continue;
        if (c.registerSet > 2u || c.registerCount == 0)
          continue;
        if (c.registerIndex + c.registerCount > caps::MaxFloatConstantsPS)
          continue;
        if (isUniformVector) {
          info.uniformVectorRegisters.push_back(c.registerIndex);
        }
        uniformsByName.emplace_back(name, c.registerIndex);
        uniformSignatureEntries.push_back(makeUe3SignatureEntry(name, c.registerSet));
        info.constRanges.emplace_back(c.registerIndex, c.registerCount);
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

      // canonical shader signature: name-sorted material sampler declarations. Engine symbols
      // are excluded wholesale (the lightmap policy permutations reference different engine
      // constants, e.g. AmbientColorAndSkyFactor only outside SIMPLE_LIGHTING), and Uniform*
      // constants are avoided because fxc strips whichever uniforms a permutation does not
      // reference (e.g. specular-only expressions in the simple-lightmap compile) - either
      // would leak the permutation back into the identity. Constant-color materials have no
      // material samplers, so their signature falls back to the uniform declarations
      // (invariant whenever both permutations reference the same uniform set).
      info.canonicalShaderSignature = hashUe3SignatureEntries(
        !samplerSignatureEntries.empty() ? samplerSignatureEntries : uniformSignatureEntries);

      if (info.constRanges.empty())
        return info;

      std::sort(info.constRanges.begin(), info.constRanges.end());
      Ue3MaterialConstRanges merged;
      for (const auto& r : info.constRanges) {
        if (merged.empty() || r.first > merged.back().first + merged.back().second) {
          merged.push_back(r);
        } else {
          const uint32_t end = std::max(merged.back().first + merged.back().second, r.first + r.second);
          merged.back().second = end - merged.back().first;
        }
      }
      info.constRanges = std::move(merged);
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
        const std::vector<uint8_t>& bytecode) {
      auto it = s_ue3PsMaterialIdentityCache.find(psHash);
      if (it == s_ue3PsMaterialIdentityCache.end()) {
        it = s_ue3PsMaterialIdentityCache.emplace(psHash, parseUe3PsMaterialIdentityFromCtab(bytecode)).first;
      }
      return it->second;
    }

    // UE3 lightmap-permutation bridge (rtx.d3d9.ue3LightmapPermutationBridgeLookup).
    //
    // The simple-lightmap compile strips material samplers and uniforms referenced only by
    // specular/two-sided-lighting expressions, so such materials cannot share one identity
    // across DirectionalLightmaps states: the simple-state draw computes its identity over a
    // strict SUBSET of the directional-state draw's symbols. The bridge exploits the superset
    // direction: from the richer draw, recompute the identity chain for small symbol-drop
    // combinations - dropping exactly the stripped symbols reproduces the subset state's hash
    // bit-for-bit (sampler names, streamed values, and the chain formula all match by
    // construction). The replacement lookup then tries these alternates, so replacements
    // authored under DirectionalLightmaps=False match under =True with no aliasing heuristics:
    // an alternate either reconstructs a captured identity exactly or misses.
    struct Ue3PresentMaterialSampler {
      // points into the cached Ue3PsMaterialIdentityInfo entry; only consumed within the draw
      const std::string* name = nullptr;
      XXH64_hash_t nameKey = kEmptyHash;
      XXH64_hash_t imageHash = kEmptyHash;
    };

    struct Ue3PresentMaterialUniform {
      XXH64_hash_t nameKey = kEmptyHash;
      Vector4 value;
    };

    // enumeration limits: strippable symbol counts are small in practice (a spec map and a
    // couple of spec/two-sided uniforms); larger material graphs are not worth the combinatorics
    constexpr uint32_t kUe3BridgeMaxSamplers = 8;
    constexpr uint32_t kUe3BridgeMaxUniforms = 10;
    constexpr uint32_t kUe3BridgeMaxSamplerDrops = 2;
    constexpr uint32_t kUe3BridgeMaxUniformDrops = 2;
    constexpr size_t kUe3BridgeMaxVariants = 64;

    static std::shared_ptr<const std::vector<XXH64_hash_t>> buildUe3LightmapPermutationAlternateHashes(
        const XXH64_hash_t fullMaterialHash,
        const Ue3PresentMaterialSampler* samplers,
        const uint32_t samplerCount,
        const Ue3PresentMaterialUniform* uniforms,
        const uint32_t uniformCount) {
      if (samplerCount == 0 || samplerCount > kUe3BridgeMaxSamplers || uniformCount > kUe3BridgeMaxUniforms)
        return nullptr;

      XXH3_state_t* const state = getThreadLocalXxh3State();
      if (state == nullptr)
        return nullptr;

      auto computeSignature = [&](const uint32_t samplerDropMask) {
        std::vector<std::string> entries;
        entries.reserve(samplerCount);
        for (uint32_t i = 0; i < samplerCount; i++) {
          if ((samplerDropMask & (1u << i)) == 0) {
            entries.push_back(makeUe3SignatureEntry(*samplers[i].name, kD3dxRegisterSetSampler));
          }
        }
        return hashUe3SignatureEntries(entries);
      };

      auto computeTextureSet = [&](const uint32_t samplerDropMask) {
        XXH3_64bits_reset(state);
        for (uint32_t i = 0; i < samplerCount; i++) {
          if ((samplerDropMask & (1u << i)) == 0) {
            XXH3_64bits_update(state, &samplers[i].nameKey, sizeof(samplers[i].nameKey));
            XXH3_64bits_update(state, &samplers[i].imageHash, sizeof(samplers[i].imageHash));
          }
        }
        return XXH3_64bits_digest(state);
      };

      // ~0u = drop all constants (matches the subset state having constants excluded or none left)
      auto computeConstants = [&](const uint32_t uniformDropMask) -> XXH64_hash_t {
        if (uniformDropMask == ~0u || uniformCount == 0)
          return kEmptyHash;
        XXH3_64bits_reset(state);
        bool any = false;
        for (uint32_t i = 0; i < uniformCount; i++) {
          if ((uniformDropMask & (1u << i)) == 0) {
            XXH3_64bits_update(state, &uniforms[i].nameKey, sizeof(uniforms[i].nameKey));
            XXH3_64bits_update(state, &uniforms[i].value, sizeof(uniforms[i].value));
            any = true;
          }
        }
        return any ? XXH3_64bits_digest(state) : kEmptyHash;
      };

      std::vector<uint32_t> uniformDropMasks;
      uniformDropMasks.push_back(0u);
      for (uint32_t mask = 1; mask < (1u << uniformCount); mask++) {
        if (bit::popcnt(mask) <= kUe3BridgeMaxUniformDrops) {
          uniformDropMasks.push_back(mask);
        }
      }
      uniformDropMasks.push_back(~0u);

      auto result = std::make_shared<std::vector<XXH64_hash_t>>();
      result->reserve(kUe3BridgeMaxVariants);

      for (uint32_t samplerDropMask = 0; samplerDropMask < (1u << samplerCount); samplerDropMask++) {
        const uint32_t drops = bit::popcnt(samplerDropMask);
        if (drops > kUe3BridgeMaxSamplerDrops || drops >= samplerCount)
          continue;

        const XXH64_hash_t signature = computeSignature(samplerDropMask);
        const XXH64_hash_t textureSetHash = computeTextureSet(samplerDropMask);
        // replicates LegacyMaterialData::updateCachedHash's seed chain
        const XXH64_hash_t tier2 = XXH3_64bits_withSeed(&textureSetHash, sizeof(textureSetHash), signature);

        for (const uint32_t uniformDropMask : uniformDropMasks) {
          if (samplerDropMask == 0 && uniformDropMask == 0)
            continue; // identical to the draw's own identity (tier 1)

          const XXH64_hash_t constantsHash = computeConstants(uniformDropMask);
          const XXH64_hash_t variant = constantsHash != kEmptyHash
            ? XXH3_64bits_withSeed(&constantsHash, sizeof(constantsHash), tier2)
            : tier2;

          if (variant != fullMaterialHash &&
              std::find(result->begin(), result->end(), variant) == result->end()) {
            result->push_back(variant);
            if (result->size() >= kUe3BridgeMaxVariants) {
              return result;
            }
          }
        }
      }

      if (result->empty())
        return nullptr;
      return result;
    }

    // memoized per full material identity; pure function of the identity's inputs
    static fast_unordered_cache<std::shared_ptr<const std::vector<XXH64_hash_t>>> s_ue3AlternateHashCache;

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

    static XXH64_hash_t makeUe3MicChurnGroupKey(const XXH64_hash_t shaderIdentitySeed,
                                                const XXH64_hash_t textureSetHash) {
      return XXH3_64bits_withSeed(&textureSetHash, sizeof(textureSetHash), shaderIdentitySeed);
    }

    static bool isUe3MicGroupAutoExcluded(const XXH64_hash_t churnGroupKey) {
      return s_ue3MicAutoExcludedGroups.find(churnGroupKey) != s_ue3MicAutoExcludedGroups.end();
    }

    // Returns true when the group was newly auto-excluded on this call.
    static bool trackUe3MicConstantChurn(const XXH64_hash_t churnGroupKey,
                                         const XXH64_hash_t psHash,
                                         const XXH64_hash_t shaderIdentitySeed,
                                         const XXH64_hash_t textureSetHash,
                                         const XXH64_hash_t constantsHash) {
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
        Logger::warn(str::format(
          "[RTX-Compatibility][UE3-MIC] Material group (seed=0x", std::hex, shaderIdentitySeed,
          ", ps=0x", psHash,
          ", textureSet=0x", textureSetHash, std::dec,
          ") minted ", kUe3MicChurnWarnThreshold,
          "+ distinct constant hashes - its constant registers are frame-varying "
          "(Time/panner/fade/sub-UV expressions). Excluding this group's constants from "
          "material identity for this session; add the shader hash or seed to "
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
        " consts=0x", constantsHash, std::dec,
        " textures=[", textureList, "]",
        " constRanges=[", ranges, "]",
        constantsExcluded ? " constsExcluded=1" : "",
        " ctab=", identityInfo.hasCtab ? 1 : 0));

      static fast_unordered_cache<uint32_t> s_distinctHashCountPerGroup;
      static fast_unordered_set s_churnWarnedShaders;
      const XXH64_hash_t groupKey = XXH3_64bits_withSeed(&textureSetHash, sizeof(textureSetHash), shaderIdentitySeed);
      const uint32_t distinctCount = ++s_distinctHashCountPerGroup[groupKey];
      if (distinctCount == kUe3MicChurnWarnThreshold && s_churnWarnedShaders.insert(psHash).second) {
        Logger::warn(str::format(
          "[RTX-Compatibility][UE3-MIC] Pixel shader 0x", std::hex, psHash,
          " has minted ", std::dec, distinctCount, "+ distinct material hashes for a single texture set (0x",
          std::hex, textureSetHash, std::dec, ") - its constant registers are likely frame-varying ",
          "(Time/panner/fade/sub-UV expressions). Add it to ",
          "rtx.d3d9.ue3MicConstantIdentityExcludedShaders to stabilize its material identity."));
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

      if (contains("lightmap")) {
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

      auto markName = [&](const std::string& lowerName, const bool isSampler) {
        if (isSampler) {
          const uint8_t semanticFlags = classifyPixelSamplerSemanticFlags(lowerName);
          info.hasMaterialSampler |= (semanticFlags & kPsSamplerSemanticMaterialTexture) != 0;
          info.hasEngineAuxSampler |= (semanticFlags & kPsSamplerSemanticEngineAuxiliary) != 0;
          info.hasVideoSampler |= (semanticFlags & kPsSamplerSemanticVideo) != 0;

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
          info.hasExposureOrToneSampler |= containsToken(lowerName, "exposure") ||
                                           containsToken(lowerName, "colorcurves") ||
                                           containsToken(lowerName, "saturationmask") ||
                                           containsToken(lowerName, "blurredimage") ||
                                           containsToken(lowerName, "filtertexture");
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
      };

      for (const DxsoCtab::Constant& c : ctab.m_constantData) {
        const bool isSampler = c.registerSet == kD3dxRegisterSetSampler;
        const std::string lowerName = toLowerAscii(c.name);
        markName(lowerName, isSampler);

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
    case Ue3VertexFactoryType::Particle:
    case Ue3VertexFactoryType::ParticleBeamTrail:
    case Ue3VertexFactoryType::LensFlare:
      return true;
    default:
      return false;
    }
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

    // UE3 SceneCapture probes (SceneCapture2D/Reflect/Portal actors: security monitors,
    // mirrors) re-render the world from their own camera before the main view, into the same
    // shared SceneColor render target - only the viewport, sized to the probe's
    // TextureRenderTarget, tells capture draws apart from main-view draws. Their shaders
    // declare genuine ViewProjectionMatrix/CameraPosition constants, so CTAB camera
    // verification alone cannot keep them from steering the Main camera.
    if ((m_frameOptions.ue3SkipSceneCapturePasses || m_frameOptions.ue3EngineMode) &&
        isWorldGeometry &&
        m_activePresentParams.has_value()) {
      const D3DVIEWPORT9& vp = d3d9State().viewport;
      const uint32_t bbW = m_activePresentParams->BackBufferWidth;
      const uint32_t bbH = m_activePresentParams->BackBufferHeight;
      // strictly under half the backbuffer in both dimensions: capture probe targets are small
      // (typically 256-1024) while the main view renders at backbuffer size or a screen
      // percentage well above one half; exact-half viewports (splitscreen) stay untouched
      if (bbW != 0 && bbH != 0 &&
          vp.Width != 0 && vp.Height != 0 &&
          vp.Width * 2 < bbW && vp.Height * 2 < bbH) {
        return Ue3PassType::SceneCapture;
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
        m_currentUe3VertexFactory == Ue3VertexFactoryType::Foliage) {
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
      " reason=", reason));
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
    restoreNgxGameSettingsRedirects();

    // Owned bridge parent handle only; GetCurrentProcess() is a pseudo-handle.
    if (m_ngxGameProcessOwned && m_ngxGameProcess != nullptr)
      ::CloseHandle(m_ngxGameProcess);
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
    o.ue3LightmapPermutationInvariantHash = ue3LightmapPermutationInvariantHashObject().get();
    o.ue3LightmapPermutationBridgeLookup = ue3LightmapPermutationBridgeLookupObject().get();
    o.ue3LogMaterialInstanceHash = ue3LogMaterialInstanceHashObject().get();
    o.ue3SkipDepthPrepass = ue3SkipDepthPrepassObject().get();
    o.ue3SkipShadowDepthPasses = ue3SkipShadowDepthPassesObject().get();
    o.ue3SkipDepthTestDisabledTranslucency = ue3SkipDepthTestDisabledTranslucencyObject().get();
    o.ue3SkipSceneCapturePasses = ue3SkipSceneCapturePassesObject().get();
    o.conservativeOcclusionQueries = conservativeOcclusionQueriesObject().get();
    o.ue3StaticLocalMeshVertexCaptureCache = ue3StaticLocalMeshVertexCaptureCacheObject().get();
    o.ue3StaticLocalMeshVertexCaptureCacheWarmupFrames = ue3StaticLocalMeshVertexCaptureCacheWarmupFramesObject().get();
    o.ue3StaticGeometryHashMemoization = ue3StaticGeometryHashMemoizationObject().get();
    o.ue3VertexCaptureCameraCellSize = ue3VertexCaptureCameraCellSizeObject().get();
    o.ue3NativeLocalMeshVertexCapture = ue3NativeLocalMeshVertexCaptureObject().get();
    o.ue3RequireCtabCameraConstants = ue3RequireCtabCameraConstantsObject().get();
    o.ue3StableDiffuseSelection = ue3StableDiffuseSelectionObject().get();
    o.ue3MicAutoExcludeFrameVaryingConstants = ue3MicAutoExcludeFrameVaryingConstantsObject().get();
    o.ue3LogClassification = ue3LogClassificationObject().get();
    o.ue3LogUvResolution = ue3LogUvResolutionObject().get();
    o.ue3LogUvAffineDetail = ue3LogUvAffineDetailObject().get();
    o.ue3LogAlbedoSelection = ue3LogAlbedoSelectionObject().get();
    o.ue3LogCapturePrecision = ue3LogCapturePrecisionObject().get();
    o.ue3LogDrawStatusFlaps = ue3LogDrawStatusFlapsObject().get();
    o.ue3LogOcclusionQueries = ue3LogOcclusionQueriesObject().get();
    o.deferredUiReplay = deferredUiReplayObject().get();
    o.deferredUiRefreshSceneColor = deferredUiRefreshSceneColorObject().get();
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

    o.valid = true;
  }

  bool D3D9Rtx::shouldUseUe3CameraHashCell() const {
    return (m_frameOptions.ue3CameraFromShaderConstants || m_frameOptions.ue3EngineMode) &&
           m_frameOptions.ue3VertexCaptureCameraCellSize > 0.0f;
  }

  bool D3D9Rtx::computeUe3CameraHashCell(Ue3CameraHashCell& outCell) const {
    if (!shouldUseUe3CameraHashCell()) {
      return false;
    }

    const float cellSize = m_frameOptions.ue3VertexCaptureCameraCellSize;
    if (!std::isfinite(cellSize) || cellSize <= 0.0f) {
      return false;
    }

    // Memoized on the exact (worldToView, cellSize) inputs: this runs up to twice per
    // draw (static vertex-capture key + live geometry VS hash component) and the view
    // matrix repeats across most draws of a frame, so the affine inverse below would
    // otherwise be paid thousands of times per frame for one or two distinct views.
    const Matrix4& worldToView = m_activeDrawCallState.transformData.worldToView;
    if (m_ue3CameraCellMemoValid &&
        m_ue3CameraCellMemoCellSize == cellSize &&
        std::memcmp(&m_ue3CameraCellMemoWorldToView, &worldToView, sizeof(Matrix4)) == 0) {
      outCell = m_ue3CameraCellMemoCell;
      return m_ue3CameraCellMemoResult;
    }

    // compute the full result first, then publish to the memo, so the memo can never
    // hold a half-written entry regardless of how the paths below evolve
    Ue3CameraHashCell cell = {};
    bool result = false;

    const Matrix4 cameraViewToWorld = inverseAffine(worldToView);
    const Vector3 cameraPos = cameraViewToWorld[3].xyz();
    if (std::isfinite(cameraPos.x) && std::isfinite(cameraPos.y) && std::isfinite(cameraPos.z)) {
      cell = {
        int32_t(std::floor(cameraPos.x / cellSize)),
        int32_t(std::floor(cameraPos.y / cellSize)),
        int32_t(std::floor(cameraPos.z / cellSize)),
      };
      result = true;
    }

    m_ue3CameraCellMemoWorldToView = worldToView;
    m_ue3CameraCellMemoCellSize = cellSize;
    m_ue3CameraCellMemoCell = cell;
    m_ue3CameraCellMemoResult = result;
    m_ue3CameraCellMemoValid = true;

    outCell = cell;
    return result;
  }

  bool D3D9Rtx::areUe3CameraHashCellsEqual(const Ue3CameraHashCell& a, const Ue3CameraHashCell& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
  }

  void D3D9Rtx::logUe3CameraHashCellIfChanged(const Ue3CameraHashCell& cell, const char* reason) {
    if (!m_frameOptions.ue3LogCapturePrecision && Logger::logLevel() > LogLevel::Debug) {
      return;
    }

    if (m_hasLoggedUe3CameraHashCell &&
        areUe3CameraHashCellsEqual(m_lastLoggedUe3CameraHashCell, cell)) {
      return;
    }

    m_hasLoggedUe3CameraHashCell = true;
    m_lastLoggedUe3CameraHashCell = cell;
    Logger::debug(str::format(
      "[RTX-Compatibility][UE3-Capture] cameraCell=(",
      cell.x, ",", cell.y, ",", cell.z,
      "), cellSize=", m_frameOptions.ue3VertexCaptureCameraCellSize,
      ", reason=", reason));
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
    if (!m_frameOptions.ue3EngineMode) {
      return false;
    }
    if (!m_parent->UseProgrammableVS()) {
      return false;
    }
    if (!m_frameOptions.useVertexCapture) {
      return false;
    }
    if (m_currentUe3VertexFactory != Ue3VertexFactoryType::Local) {
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
    if (geoData.blendWeightBuffer.defined() || geoData.blendIndicesBuffer.defined()) {
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
    // UE3 static meshes use FPositionVertexBuffer for POSITION and FStaticMeshVertexBuffer
    // for tangent/normal/UV data. Ref-pose skeletal LocalVertexFactory draws use a unified
    // stream and are not stable static-local cache candidates.
    if (sig.positionStream == sig.tangentStream || sig.positionStream == sig.normalStream) {
      return false;
    }

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

    template<typename ElementsT, typename VertexContextT>
    XXH64_hash_t hashUe3KeyStreamRecords(const ElementsT& elements,
                                         const VertexContextT* vertexContext,
                                         XXH64_hash_t seed) {
      // decl layout itself (semantics, formats, offsets, stream assignment)
      XXH64_hash_t hash = XXH3_64bits_withSeed(elements.data(), elements.size() * sizeof(D3DVERTEXELEMENT9), seed);

      // per-element stream identity
      std::array<Ue3KeyStreamRecord, kUe3KeyStreamRecordChunk> records;
      size_t recordCount = 0;
      for (const auto& element : elements) {
        // stream bounds are pre-validated by the canUse/canMemoize eligibility gates
        assert(element.Stream < caps::MaxStreams);
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
  // identity + content generation. Deliberately excludes the stable VS-constant hash, the
  // object transform and the camera cell, so every instance of a mesh - and every animation
  // pose of a skinned mesh - shares one entry.
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
    return hashUe3KeyStreamRecords(d3d9State().vertexDecl->GetElements(), vertexContext, headerHash);
  }

  namespace {
    struct Ue3VertexCaptureKeyHeader {
      Ue3IaMemoKeyHeader iaHeader; // shared IA identity block
      uint32_t cullMode;
      uint32_t frontFace;
      uint32_t nativeLocalCapture;
      uint32_t cameraCellValid;
      uint64_t stableVsHash;
      Matrix4 objectToWorld;
      int32_t cameraCell[3];
      uint32_t explicitPad0;
    };
    static_assert(sizeof(Ue3VertexCaptureKeyHeader) ==
                    sizeof(Ue3IaMemoKeyHeader) + 4 * sizeof(uint32_t) + sizeof(uint64_t) + sizeof(Matrix4) + 4 * sizeof(int32_t),
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
    header.nativeLocalCapture = m_frameOptions.ue3NativeLocalMeshVertexCapture ? 1u : 0u;
    // computed once per draw in internalPrepareDraw and shared with computeHash
    header.stableVsHash = m_activeStableVsHash;
    header.objectToWorld = m_activeDrawCallState.transformData.objectToWorld;
    Ue3CameraHashCell cameraCell;
    if (computeUe3CameraHashCell(cameraCell)) {
      header.cameraCellValid = 1u;
      header.cameraCell[0] = cameraCell.x;
      header.cameraCell[1] = cameraCell.y;
      header.cameraCell[2] = cameraCell.z;
    }

    const XXH64_hash_t headerHash = XXH3_64bits_withSeed(&header, sizeof(header), kSeed);
    return hashUe3KeyStreamRecords(d3d9State().vertexDecl->GetElements(), vertexContext, headerHash);
  }

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

    bool hashedFloatConstsWithExclusions = false;
    if ((m_frameOptions.ue3CameraFromShaderConstants || m_frameOptions.ue3EngineMode) && floatConstRegCount > 0) {
      constexpr uint32_t kFallbackViewProjReg = 0;
      constexpr uint32_t kFallbackViewProjRegCount = 4;
      constexpr uint32_t kFallbackViewOriginReg = 4;
      constexpr uint32_t kFallbackViewOriginRegCount = 1;

      uint32_t viewProjReg = kFallbackViewProjReg;
      uint32_t viewProjRegCount = kFallbackViewProjRegCount;
      uint32_t viewOriginReg = kFallbackViewOriginReg;
      uint32_t viewOriginRegCount = kFallbackViewOriginRegCount;

      if (m_currentUe3CtabInfo.has_value()) {
        const Ue3VsShaderCtabInfo& ctabInfo = *m_currentUe3CtabInfo;
        if (ctabInfo.hasViewProjectionMatrix && ctabInfo.viewProjectionMatrixRegisterCount > 0) {
          viewProjReg = ctabInfo.viewProjectionMatrixRegisterIndex;
          viewProjRegCount = ctabInfo.viewProjectionMatrixRegisterCount;
        }
        if (ctabInfo.hasCameraPosition && ctabInfo.cameraPositionRegisterCount > 0) {
          viewOriginReg = ctabInfo.cameraPositionRegisterIndex;
          viewOriginRegCount = ctabInfo.cameraPositionRegisterCount;
        }
      }

      struct RegRange {
        uint32_t begin = 0;
        uint32_t end = 0;
      };

      std::array<RegRange, 2> ranges = {{
        { viewProjReg, viewProjReg + viewProjRegCount },
        { viewOriginReg, viewOriginReg + viewOriginRegCount },
      }};
      std::array<RegRange, 2> validRanges = {};
      uint32_t validRangeCount = 0;
      for (const RegRange& r : ranges) {
        RegRange clamped;
        clamped.begin = std::min(r.begin, floatConstRegCount);
        clamped.end = std::min(r.end, floatConstRegCount);
        if (clamped.begin < clamped.end) {
          validRanges[validRangeCount++] = clamped;
        }
      }

      if (validRangeCount > 0) {
        if (validRangeCount == 2 && validRanges[1].begin < validRanges[0].begin) {
          std::swap(validRanges[0], validRanges[1]);
        }
        if (validRangeCount == 2 && validRanges[1].begin <= validRanges[0].end) {
          validRanges[0].end = std::max(validRanges[0].end, validRanges[1].end);
          validRangeCount = 1;
        }

        uint32_t cursor = 0;
        for (uint32_t i = 0; i < validRangeCount; i++) {
          hashFloatConstRange(cursor, validRanges[i].begin);
          cursor = std::max(cursor, validRanges[i].end);
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
    const uint32_t warmupFrames = std::max(1u, m_frameOptions.ue3StaticLocalMeshVertexCaptureCacheWarmupFrames);
    if (entry.vertexCount != geoData.vertexCount ||
        !entry.positionBuffer.defined() ||
        entry.captureCount < warmupFrames ||
        entry.lastFrameTouched == currentFrame) {
      return false;
    }

    geoData.positionBuffer = entry.positionBuffer;
    geoData.normalBuffer = entry.normalBuffer;
    geoData.texcoordBuffer = entry.texcoordBuffer;
    geoData.color0Buffer = entry.color0Buffer;
    entry.lastFrameTouched = currentFrame;

    return true;
  }

  void D3D9Rtx::updateUe3StaticVertexCaptureCache(XXH64_hash_t cacheKey, const RasterGeometry& geoData) {
    Ue3VertexCaptureCacheEntry& entry = m_ue3VertexCaptureCache[cacheKey];
    entry.positionBuffer = geoData.positionBuffer;
    entry.normalBuffer = geoData.normalBuffer;
    entry.texcoordBuffer = geoData.texcoordBuffer;
    entry.color0Buffer = geoData.color0Buffer;
    entry.vertexCount = geoData.vertexCount;
    entry.captureCount = std::min(entry.captureCount + 1, 0xFFFFu);
    entry.lastFrameTouched = m_parent->GetDXVKDevice()->getCurrentFrameId();
  }

  void D3D9Rtx::pruneUe3StaticVertexCaptureCache() {
    constexpr uint32_t kMaxUntouchedFrames = 600;
    const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();
    m_ue3VertexCaptureCache.erase_if([&](auto it) {
      return currentFrame - it->second.lastFrameTouched > kMaxUntouchedFrames;
    });
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
    return DxvkBufferSlice(pDevice->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::AppBuffer, "Vertex Capture Buffer"));
  }

  bool D3D9Rtx::prepareVertexCapture(const int vertexIndexOffset, const bool capturePositionFromInput) {
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
      if (!detOk(t.viewToProjection) || !detOk(t.worldToView) || !detOk(t.objectToWorld)) {
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
    // positions are captured from VS output (post-skinning for GPU-skinned draws) then unprojected to object
    // space, when a skinned shader doesn't output a normal semantic IA normals alone are bind-pose and do not
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
    if (capturePositionFromInput) {
      vertexCaptureFlags |= kVertexCaptureFlag_PositionFromInput;
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
    // 3 : non-skinned, VS doesn't output NORMAL, just keep IA normals from processVertices they're in the same object space as the un-projected captured positions anyway..

    // Check if we should/can get colors
    if (BoundShaderHas(vertexShader, DxsoUsage::Color, false) && d3d9State().pixelShader.ptr() == nullptr) {
      const uint32_t colorOffset = offsetof(CapturedVertex, color0);
      geoData.color0Buffer = RasterBuffer(slice, colorOffset, stride, VK_FORMAT_B8G8R8A8_UNORM);
      assert(geoData.color0Buffer.offset() % 4 == 0);
    }

    auto constants = m_vsVertexCaptureData->allocSlice();

    // Upload
    auto& data = *reinterpret_cast<D3D9RtxVertexCaptureData*>(constants.mapPtr);
    data.invProj = inverse(m_activeDrawCallState.transformData.viewToProjection);
    data.viewToWorld = inverseAffine(m_activeDrawCallState.transformData.worldToView);
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

    // Process vertex buffers from CPU
    for (const auto& element : d3d9State().vertexDecl->GetElements()) {
      // Get vertex context
      const VertexContext& ctx = vertexContext[element.Stream];

      if (ctx.mappedSlice.handle == VK_NULL_HANDLE)
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
        if (element.UsageIndex == 0)
          targetBuffer = &geoData.normalBuffer;
        break;
      case D3DDECLUSAGE_TEXCOORD:
        if (m_iaTexcoordIndex <= MAXD3DDECLUSAGEINDEX && element.UsageIndex == m_iaTexcoordIndex)
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
        if (element.Usage == D3DDECLUSAGE_NORMAL && fmt == VK_FORMAT_R8G8B8A8_USCALED) {
          fmt = VK_FORMAT_R8G8B8A8_UNORM;
        }
        *targetBuffer = RasterBuffer(streamCopies[element.Stream], elementOffset, ctx.stride, fmt);
        assert(targetBuffer->offset() % 4 == 0);
      }
    }
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

  bool D3D9Rtx::processRenderState(const DrawContext& drawContext) {
    ScopedCpuProfileZone();
    DrawCallTransforms& transformData = m_activeDrawCallState.transformData;
    m_forceIaTexcoordForOutlier = false;

    // m_activeDrawCallState is reused across draws, so these fields need an explicit per-draw reset
    m_activeDrawCallState.allowMainCameraUpdate = true;
    m_activeDrawCallState.programmableVertexShaderBytecodeHash = 0;
    m_activeDrawCallState.ue3PassDescription = describeUe3PassType(m_currentUe3PassType);
    m_activeDrawCallState.ue3LightmapPermutationAlternateHashes.reset();

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
    bool ue3CameraUsedTranspose = false;

    const bool needsUe3CtabInfo =
      effectiveUe3Camera ||
      effectiveUe3ObjectToWorld ||
      m_frameOptions.useVertexCapture;
    if (usesProgrammableVs && vertexShaderCommon != nullptr &&
        needsUe3CtabInfo) {
      auto parseCtabInfo = [&](const std::vector<uint8_t>& bytecode) -> Ue3VsShaderCtabInfo {
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

          for (const DxsoCtab::Constant& c : ctab.m_constantData) {
            const std::string name = lower(c.name);

            // Any lightmap policy symbol marks the shader pair as recompiled per lightmap
            // permutation. Vertex-lightmap policies put LightMapScale in the VERTEX shader
            // only (the lightmap reaches the pixel shader through interpolators), so this is
            // the only signal for their pixel shader's material identity normalization.
            if (!info.hasLightmapSymbols && contains(name, "lightmap")) {
              info.hasLightmapSymbols = true;
            }

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
          m_ue3VsShaderCtabCache.emplace(shaderHash, parseCtabInfo(bytecode));
          it = m_ue3VsShaderCtabCache.find(shaderHash);
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
        m_activeDrawCallState.ue3PassDescription = describeUe3PassType(m_currentUe3PassType);
        logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, reason);
      };

      if (tryGetUe3CameraFromConstantsCached(viewProjReg, viewOriginReg, ue3WorldToView, ue3ViewToProjection, ue3CameraUsedTranspose, ue3CameraReconstructionError)) {
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

          transformData.objectToWorld = extractUe3ObjectToWorld(reg, hasWorldToLocal, w2lReg, ue3CameraUsedTranspose);

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

        const XXH64_hash_t texDescHash = texture->GetImage()->getDescriptorHash();
        if (lookupHash(*m_frameOptions.raytracedRenderTargetTextures, texDescHash) ||
            lookupHash(m_autoRaytracedRenderTargetDescHashes, texDescHash)) {
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
    bool deferredUiMatchedTextureIsRenderTarget = false;
    XXH64_hash_t deferredUiMatchedRtDescriptorHash = 0;
    int deferredUiTagState = -1;
    auto isDeferredUiTagged = [&]() {
      if (deferredUiTagState < 0) {
        deferredUiTagState = isDeferredUiTaggedDraw(&deferredUiMatchedTextureHash,
                                                    &deferredUiMatchedTextureIsRenderTarget,
                                                    &deferredUiMatchedRtDescriptorHash) ? 1 : 0;
      }
      return deferredUiTagState == 1;
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
      ONCE(Logger::info("[RTX-Compatibility-Info] Ignored UE3 scene capture offscreen view draw (world geometry, sub-half-backbuffer viewport)."));
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "scene capture offscreen view");
      return { RtxGeometryStatus::Ignored, false };
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

    // Deferred UI overlays (rtx.deferredUiTextures / rtx.d3d9.deferredUiPixelShaders, e.g. UE3
    // MaterialEffect fullscreen fades): rasterized on top of the ray-traced image WITHOUT
    // triggering RTX injection - the draw is captured and replayed after injection fires later
    // in the frame. Placed after the pass switch above so UI composites and video passes can
    // never be deferred, and before the fullscreen-composite/post-process filters below so
    // tagging wins over those. World geometry and depth-writing draws are never deferred even
    // when tagged: shared textures (e.g. a scene-color render target sampled by translucent
    // meshes) must not pull geometry out of the ray-traced scene.
    if (!m_frameOptions.deferredUiTextures->empty() || !m_frameOptions.deferredUiPixelShaders->empty()) {
      const XXH64_hash_t& matchedTextureHash = deferredUiMatchedTextureHash;
      const bool& matchedTextureIsRenderTarget = deferredUiMatchedTextureIsRenderTarget;
      const XXH64_hash_t& matchedRtDescriptorHash = deferredUiMatchedRtDescriptorHash;

      if (isDeferredUiTagged()) {
        const bool matchedByPixelShaderTag = matchedTextureHash == 0;
        const bool zWriteEnabled = d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE;
        const bool isWorldGeometryVertexFactory = isUe3WorldGeometryVertexFactory(m_currentUe3VertexFactory);

        // Engine post-process/composite shaders (gamma-correction scene copy, tone mapping,
        // motion blur, distortion, fog) legitimately sample the scene render target but must
        // never be deferred: replaying e.g. the gamma copy over the ray-traced image uniformly
        // brightens the whole screen and, being opaque and later in the frame, overwrites the
        // real overlay effects. Only texture tags skip them - an explicit pixel shader tag is
        // taken as user intent and still defers.
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
                                      psInfo.hasHazeConstants;
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

          if (eligible && matchedTextureIsRenderTarget) {
            Logger::info(str::format(
              "[RTX-DeferredUI] Tagged texture 0x", std::hex, matchedTextureHash, std::dec,
              " is a render target: its texture hash changes every time the game recreates it "
              "(respawn/level load). For a stable tag use the pixel shader instead: add 0x",
              std::hex, psHash, std::dec, " to rtx.d3d9.deferredUiPixelShaders",
              matchedRtDescriptorHash != 0
                ? str::format(" (the render target's stable descriptor hash 0x", std::hex, matchedRtDescriptorHash, std::dec, " also matches this category)")
                : std::string(),
              "."));
          }
        }

        if (eligible) {
          m_ue3LastDrawDecision = "deferred UI overlay";
          logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Rasterized, "deferred UI overlay");
          return { RtxGeometryStatus::Rasterized, false, true };
        }
        // Ineligible tagged draws fall through to normal classification - never suppressed.
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
        if (image != nullptr) {
          const XXH64_hash_t descHash = image->getDescriptorHash();
          if (lookupHash(*m_frameOptions.raytracedRenderTargetTextures, descHash) ||
              lookupHash(m_autoRaytracedRenderTargetDescHashes, descHash)) {
            m_activeDrawCallState.isDrawingToRaytracedRenderTarget = true;
            return { RtxGeometryStatus::RayTraced, false };
          }
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
                  "rtDescHash=0x", std::hex, rtDescHash, std::dec,
                  ". If this RT contains the main scene, add it to rtx.raytracedRenderTargetTextures."));
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
        uint64_t bestArea = 0;

        for (uint32_t i : bit::BitMask(rtSamplerMask)) {
          D3D9CommonTexture* tex = GetCommonTexture(d3d9State().textures[i]);
          if (!tex || tex->GetImage() == nullptr)
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
            bestHash = tex->GetImage()->getDescriptorHash();
          }
        }

        if (bestHash != 0 &&
            !lookupHash(*m_frameOptions.raytracedRenderTargetTextures, bestHash) &&
            m_autoRaytracedRenderTargetDescHashes.insert(bestHash).second) {
          Logger::info(str::format(
            "[RTX-Compatibility] Auto-selected Raytraced Render Target from fullscreen composite: texDescHash=0x",
            std::hex, bestHash, std::dec, "."));
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
              ", texDescHash=0x", std::hex, texDescHash, std::dec,
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

    if (m_currentUe3PassType == Ue3PassType::FullscreenPostProcess ||
        m_currentUe3PassType == Ue3PassType::FogOrDistortion) {
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "native UE3 screen-space contribution pass");
      return { RtxGeometryStatus::Ignored, false };
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

  bool D3D9Rtx::isDeferredUiTaggedDraw(XXH64_hash_t* pMatchedTextureHash,
                                       bool* pMatchedTextureIsRenderTarget,
                                       XXH64_hash_t* pMatchedRtDescriptorHash) const {
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

      const bool isRenderTarget = entry.isRenderTarget;
      const XXH64_hash_t descriptorHash = entry.rtDescriptorHash;

      const auto reportMatch = [&](XXH64_hash_t matchedHash, bool matchedIsRenderTarget) {
        if (pMatchedTextureHash) {
          *pMatchedTextureHash = matchedHash;
        }
        if (pMatchedTextureIsRenderTarget) {
          *pMatchedTextureIsRenderTarget = matchedIsRenderTarget;
        }
        if (pMatchedRtDescriptorHash) {
          *pMatchedRtDescriptorHash = descriptorHash;
        }
        return true;
      };

      const XXH64_hash_t texHash = entry.imageHash;
      if (texHash != 0 && lookupHash(*m_frameOptions.deferredUiTextures, texHash)) {
        return reportMatch(texHash, isRenderTarget);
      }

      // Render targets: also match by descriptor hash, which is derived from the target's
      // properties and thus stable across recreation (the image hash embeds a creation-order
      // counter and changes on every respawn/level load).
      if (descriptorHash != 0 && lookupHash(*m_frameOptions.deferredUiTextures, descriptorHash)) {
        return reportMatch(descriptorHash, true);
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
        lookupHash(RtxOptions::lightmapTextures(), texture->GetImage()->getHash()) ? ",lightmapTagged" : "",
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
      const Ue3PsMaterialIdentityInfo& identityInfo = getOrParseUe3PsMaterialIdentityInfo(psHash, bytecode);
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
            const XXH64_hash_t descHash = image->getDescriptorHash();
            isRaytracedRenderTarget =
              lookupHash(*m_frameOptions.raytracedRenderTargetTextures, descHash) ||
              lookupHash(m_autoRaytracedRenderTargetDescHashes, descHash);
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
          auto hash = texture->GetImage()->getDescriptorHash();
          if (lookupHash(*m_frameOptions.raytracedRenderTargetTextures, hash)) {
            // Mark this as a valid Raytraced Render Target draw call
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

    // for UE3 vertex shader skinned (GPUSkin) draws the LocalToWorld placement is baked into the
    // bone matrices where objectToWorld stays identity and the captured vertices are already in world
    // space. We should derive a per-instance worldspace anchor from the first bone's translation so the
    // BLAS cache can spatially determine simultaneous instances of a shared skeletal mesh and
    // give skinningData a real bone hash so the geometry refit decision tracks the animated pose
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
        // UE3 bone matrices are float4x3 (3 float4 rows per bone) and the world translation lives in
        // the .w of the first three rows of the first bone
        m_activeDrawCallState.m_skinnedWorldAnchor = Vector3(
          fConsts[boneReg + 0].w,
          fConsts[boneReg + 1].w,
          fConsts[boneReg + 2].w);
        m_activeDrawCallState.m_hasSkinnedWorldAnchor = true;

        // processSkinning() returns no SkinningData for programmable-VS draws, so skinningData
        // stays default (numBones == 0) and this bone hash will not be overwritten by finalise
        m_activeDrawCallState.skinningData.boneHash =
          XXH3_64bits(&fConsts[boneReg], size_t(boneRegCount) * sizeof(Vector4));
      }
    }

    bool canUseCachedVertexCapture = false;
    XXH64_hash_t vertexCaptureCacheKey = kEmptyHash;
    {
      ScopedCpuProfileZoneN("UE3 geometry identity keys");

      // Stable VS hash (bytecode + camera-excluded constants), computed once per draw and
      // shared between the geometry hash below and the static vertex-capture cache key.
      m_activeStableVsHashUsedExclusions = false;
      m_activeStableVsHash = (m_parent->UseProgrammableVS() && m_frameOptions.useVertexCapture)
        ? computeUe3StableVertexShaderHash(&m_activeStableVsHashUsedExclusions)
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

    const bool useUe3NativeLocalCapture =
      canUseUe3NativeLocalVertexCapture(indexContext, vertexContext, geoData);
    if (useUe3NativeLocalCapture) {
      ONCE(Logger::info("[RTX-Compatibility] UE3 native LocalVertexFactory capture: using IA object-space positions for conservative static local meshes."));
      if (m_frameOptions.ue3LogCapturePrecision && Logger::logLevel() <= LogLevel::Debug) {
        static fast_unordered_set s_loggedNativeLocalDraws;
        const XXH64_hash_t nativeKey = XXH3_64bits(&m_activeDrawCallState.transformData.objectToWorld, sizeof(Matrix4));
        if (s_loggedNativeLocalDraws.insert(nativeKey).second) {
          Logger::debug(str::format(
            "[RTX-Compatibility][UE3-Capture] native local mesh capture active, vertices=",
            geoData.vertexCount, ", indices=", geoData.indexCount));
        }
      }
    }

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
          "[RTX-Compatibility][UE3-Capture] static local vertex capture cache ",
          reusedCachedVertexCapture ? "reused" : "recapturing",
          ", key=0x", std::hex, vertexCaptureCacheKey, std::dec,
          ", vertices=", geoData.vertexCount));
      }
    }

    // For shader based drawcalls we also want to capture the vertex shader output
    bool needVertexCapture =
      m_parent->UseProgrammableVS() &&
      m_frameOptions.useVertexCapture &&
      !reusedCachedVertexCapture;
    if (needVertexCapture) {
      needVertexCapture = prepareVertexCapture(vertexIndexOffset, useUe3NativeLocalCapture);
    }
    if (canUseCachedVertexCapture && !reusedCachedVertexCapture && needVertexCapture) {
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

    const bool preserveOriginalDraw = needVertexCapture;

    return finishPrepare(
      PrepareDrawFlag::CommitToRayTracing |
      (m_activeDrawCallState.testCategoryFlags(CATEGORIES_REQUIRE_DRAW_CALL_STATE) ? PrepareDrawFlag::ApplyDrawState : 0) |
      (preserveOriginalDraw ? PrepareDrawFlag::PreserveDrawCallAndItsState : 0));
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

    submitActiveDrawCallState();

    m_parent->EmitCs([params, this](DxvkContext* ctx) {
      assert(dynamic_cast<RtxContext*>(ctx));
      DrawCallState drawCallState;
      if (m_drawCallStateQueue.pop(drawCallState)) {
        static_cast<RtxContext*>(ctx)->commitGeometryToRT(params, drawCallState);
      }
    });
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
        if (lookupHash(*m_frameOptions.lightmapTextures, texHash)) {
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
    const bool isUe3FoliageVF = vfType == Ue3VertexFactoryType::Foliage;
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
      const uint32_t usedTextureMask = m_parent->m_activeTextures & usedSamplerMask;

      // Deterministic diffuse selection: the scoring below reads live shader constant
      // values (hasNonZeroInferredSamplerOffset), which UE3 rewrites per draw for
      // panner/time/view expressions - flipping large score terms and with them the
      // chosen albedo stage between frames or with camera position. Caching the first
      // decision per (PS, texture set, sRGB, vertex factory) key pins the pick.
      XXH64_hash_t selectionCacheKey = kEmptyHash;
      bool selectionCacheUsable = false;
      bool selectionFromCache = false;
      uint64_t selectionBoundAreaSum = 0;
      if ((m_frameOptions.ue3StableDiffuseSelection || m_frameOptions.ue3EngineMode) &&
          inferredPsEntry != nullptr && inferredPsHash != kEmptyHash) {
        // scoring consults the user-taggable lightmap/never-albedo/preferred-albedo sets; drop cached
        // decisions when those sets change so texture tagging takes effect immediately
        const size_t lightmapSetSize = m_frameOptions.lightmapTextures->size();
        const size_t neverAlbedoSetSize = m_frameOptions.neverAlbedoTextures->size();
        const size_t preferredAlbedoSetSize = m_frameOptions.preferredAlbedoTextures->size();
        if (lightmapSetSize != m_ue3DiffuseSelectionLightmapSetSize ||
            neverAlbedoSetSize != m_ue3DiffuseSelectionNeverAlbedoSetSize ||
            preferredAlbedoSetSize != m_ue3DiffuseSelectionPreferredAlbedoSetSize) {
          m_ue3DiffuseSelectionCache.clear();
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
          } else {
            // superseding re-score: let ue3LogAlbedoSelection dump the authoritative decision
            m_loggedAlbedoSelections.erase(selectionCacheKey);
          }
        }
      }

      // per-stage score breakdown for rtx.d3d9.ue3LogAlbedoSelection, dumped once per selection key
      const bool logAlbedoSelection =
        m_frameOptions.ue3LogAlbedoSelection &&
        selectionCacheUsable &&
        !selectionFromCache &&
        m_loggedAlbedoSelections.find(selectionCacheKey) == m_loggedAlbedoSelections.end();
      std::string albedoSelectionLog;

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
        if (lookupHash(*m_frameOptions.lightmapTextures, texHash))
          continue;
        const bool isNeverAlbedo = lookupHash(*m_frameOptions.neverAlbedoTextures, texHash);
        const bool isPreferredAlbedo = lookupHash(*m_frameOptions.preferredAlbedoTextures, texHash);

        // material spread: distinct pixel shaders sampling this texture. Identity albedos stay
        // at 1-2 (material instances share their parent's bytecode); shared library assets -
        // detail patterns, grunge/dirt overlays, tint ramps - appear across many unrelated shaders
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
            // crossing the penalty threshold changes scores of already-pinned selections;
            // drop them so every material re-evaluates against the discovered spread
            if (spread.count == 8) {
              m_ue3DiffuseSelectionCache.clear();
              m_loggedAlbedoSelections.clear();
            }
          }
          materialSpread = spread.count;
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
        if (inferredPsEntry != nullptr && stage < caps::MaxTexturesPS) {
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
        // normal maps get no size credit - resolution advantage must not offset the decode penalty
        score += normalDecodeActive
          ? 0
          : int64_t(std::min<uint64_t>(effectiveArea, 16ull * 1024ull * 1024ull));
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
      // never samples, and never a hashless render target the binding loop would reject
      if (chosenStages[0] == kInvalidStage) {
        for (uint32_t stage : bit::BitMask(usedTextureMask)) {
          if (stage >= SamplerCount || d3d9State().textures[stage] == nullptr)
            continue;
          D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[stage]);
          if (texture == nullptr || texture->GetImage() == nullptr ||
              texture->GetImage()->getHash() == kEmptyHash)
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
      if (selectionCacheUsable && !selectionFromCache) {
        Ue3DiffuseSelectionEntry cacheEntry;
        cacheEntry.chosenStages[0] = chosenStages[0];
        cacheEntry.chosenStages[1] = chosenStages[1];
        cacheEntry.cubemapFallbackStage = strictCubemapFallbackStage;
        cacheEntry.decisionAreaSum = selectionBoundAreaSum;
        m_ue3DiffuseSelectionCache[selectionCacheKey] = cacheEntry;
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
            const Ue3PsMaterialIdentityInfo& identityInfo = getOrParseUe3PsMaterialIdentityInfo(psHash, bytecode);

            // Lightmap-bearing shaders are recompiled per lightmap policy permutation
            // (DirectionalLightmaps 3-coefficient vs simple 1-coefficient, Mirror's Edge
            // TdBicubicFiltering), so their bytecode hash - and everything seeded by it -
            // varies with those system settings. Seed such shaders with the canonical CTAB
            // material signature instead so material identity survives setting flips.
            // Texture-lightmap permutations declare lightmap symbols in the pixel shader;
            // vertex-lightmap permutations only in the vertex shader (the lightmap arrives
            // through interpolators), so the draw's VS CTAB flag must be considered too.
            // Shaders without lightmap symbols on either stage keep the bytecode hash and
            // their existing material hashes.
            const bool drawHasLightmapPermutationSymbols =
              identityInfo.hasLightmapPermutationSymbols ||
              (m_currentUe3CtabInfo.has_value() && m_currentUe3CtabInfo->hasLightmapSymbols);
            const bool useInvariantShaderIdentity =
              (m_frameOptions.ue3LightmapPermutationInvariantHash || m_frameOptions.ue3EngineMode) &&
              drawHasLightmapPermutationSymbols &&
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
            const bool bridgeLookupEnabled =
              useInvariantShaderIdentity &&
              (m_frameOptions.ue3LightmapPermutationBridgeLookup || m_frameOptions.ue3EngineMode);
            std::string micTextureListLog;
            std::array<Ue3PresentMaterialSampler, kUe3BridgeMaxSamplers> presentSamplers;
            uint32_t presentSamplerCount = 0;
            bool presentSamplersOverflowed = false;
            XXH64_hash_t textureSetHash = kEmptyHash;
            if (identityInfo.materialSamplerMask != 0) {
              XXH3_state_t* const state = getThreadLocalXxh3State();
              if (state != nullptr) {
                XXH3_64bits_reset(state);
                bool anyTextureHashed = false;
                const BoundTextureSnapshot& boundTextures = ensureBoundTextureSnapshot();
                auto hashMaterialSamplerTexture = [&](const void* samplerKey, const size_t samplerKeySize, const uint32_t samplerRegister, const char* samplerLogName) -> XXH64_hash_t {
                  if (samplerRegister >= SamplerCount || (boundTextures.mask & (1u << samplerRegister)) == 0)
                    return kEmptyHash;
                  const BoundTextureSnapshotEntry& entry = boundTextures.entries[samplerRegister];
                  if (!entry.hasImage)
                    return kEmptyHash;
                  const XXH64_hash_t imageHash = entry.imageHash;
                  if (imageHash == kEmptyHash)
                    return kEmptyHash; // hashless (e.g. render target bound as a material texture)
                  XXH3_64bits_update(state, samplerKey, samplerKeySize);
                  XXH3_64bits_update(state, &imageHash, sizeof(imageHash));
                  anyTextureHashed = true;
                  if (logMicHash) {
                    micTextureListLog += str::format(
                      micTextureListLog.empty() ? "s" : ",s", samplerRegister,
                      samplerLogName != nullptr ? str::format("(", samplerLogName, ")") : std::string(),
                      ":0x", std::hex, imageHash, std::dec);
                  }
                  return imageHash;
                };
                if (useInvariantShaderIdentity) {
                  // name-keyed: register assignments shift between lightmap policy permutations
                  for (const auto& [samplerName, samplerNameKey, samplerRegister] : identityInfo.materialSamplersByNameOrder) {
                    if (samplerRegister < caps::MaxTexturesPS) {
                      const XXH64_hash_t imageHash =
                        hashMaterialSamplerTexture(&samplerNameKey, sizeof(samplerNameKey), samplerRegister, samplerName.c_str());
                      if (bridgeLookupEnabled && imageHash != kEmptyHash) {
                        if (presentSamplerCount < presentSamplers.size()) {
                          presentSamplers[presentSamplerCount++] = Ue3PresentMaterialSampler { &samplerName, samplerNameKey, imageHash };
                        } else {
                          presentSamplersOverflowed = true;
                        }
                      }
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
              lookupHash(*m_frameOptions.ue3MicConstantIdentityExcludedShaders, psHash) ||
              (useInvariantShaderIdentity && lookupHash(*m_frameOptions.ue3MicConstantIdentityExcludedShaders, shaderIdentitySeed)) ||
              (autoExcludeEnabled && isUe3MicGroupAutoExcluded(micChurnGroupKey));
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
                trackUe3MicConstantChurn(micChurnGroupKey, psHash, shaderIdentitySeed, textureSetHash, psConstsHash)) {
              constantsExcluded = true;
              psConstsHash = kEmptyHash;
            }
            m_activeDrawCallState.materialData.setPixelShaderConstantsHashForMaterialInstance(psConstsHash);

            // Lightmap-permutation bridge: publish the identity hashes this draw would produce
            // under lightmap permutations that reference fewer material symbols (see
            // buildUe3LightmapPermutationAlternateHashes). Memoized per material identity.
            if (bridgeLookupEnabled && !presentSamplersOverflowed && presentSamplerCount > 0) {
              m_activeDrawCallState.materialData.updateCachedHash();
              const XXH64_hash_t fullMaterialHash = m_activeDrawCallState.materialData.getHash();
              if (fullMaterialHash != kEmptyHash) {
                auto it = s_ue3AlternateHashCache.find(fullMaterialHash);
                if (it == s_ue3AlternateHashCache.end()) {
                  std::array<Ue3PresentMaterialUniform, kUe3BridgeMaxUniforms> presentUniforms;
                  uint32_t presentUniformCount = 0;
                  bool presentUniformsOverflowed = false;
                  // Excluded constants collapse every sibling onto one identity; baking the
                  // first-seen sibling's live values into the memoized variants would make
                  // bridge results depend on draw order. Structural (constants-free) variants
                  // only for those.
                  if (!constantsExcluded) {
                    for (const auto& [uniformNameKey, uniformRegister] : identityInfo.namedUniformFirstRegistersByNameOrder) {
                      if (uniformRegister >= caps::MaxFloatConstantsPS)
                        continue;
                      if (presentUniformCount < presentUniforms.size()) {
                        presentUniforms[presentUniformCount++] =
                          Ue3PresentMaterialUniform { uniformNameKey, d3d9State().psConsts.fConsts[uniformRegister] };
                      } else {
                        presentUniformsOverflowed = true;
                        break;
                      }
                    }
                  }
                  it = s_ue3AlternateHashCache.emplace(
                    fullMaterialHash,
                    presentUniformsOverflowed
                      ? nullptr
                      : buildUe3LightmapPermutationAlternateHashes(
                          fullMaterialHash,
                          presentSamplers.data(), presentSamplerCount,
                          presentUniforms.data(), presentUniformCount)).first;
                }
                m_activeDrawCallState.ue3LightmapPermutationAlternateHashes = it->second;
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
          const PsSamplerUvOrigin& uvOrigin = entry.samplerUvOrigin[firstStage];

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
    m_iaTexcoordIndex = iaTexcoordIdx;

    return true;
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
    const bool dynamicMeshShape = vbDynamic && !ibDynamic && !ctabRegs.hasBoneMatrices &&
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
    // whole-buffer stream (offset 0) and a static index buffer. Everything else on
    // dynamic buffers is ring-pool geometry (particles, trails, canvas) whose
    // allocation offsets shift every frame - untrackable, and skipped.
    const bool dynamicMesh = dynamicMeshShape && !skinned;

    if ((vbDynamic || ibDynamic) && !dynamicMesh) {
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
        m_ngxVelocityStats.newRegistrations++;
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
    m_ngxVelocityWindow.exactMatches += m_ngxVelocityStats.exactMatches;
    m_ngxVelocityWindow.newRegistrations += m_ngxVelocityStats.newRegistrations;
    m_ngxVelocityWindow.pairedBeyondBounds += m_ngxVelocityStats.pairedBeyondBounds;
    m_ngxVelocityWindow.skippedNoCamera += m_ngxVelocityStats.skippedNoCamera;
    m_ngxVelocityWindow.skippedBudget += m_ngxVelocityStats.skippedBudget;
    m_ngxVelocityWindow.skippedZDisabled += m_ngxVelocityStats.skippedZDisabled;
    m_ngxVelocityWindow.skippedInstanceCap += m_ngxVelocityStats.skippedInstanceCap;
    m_ngxVelocityWindow.depthClears += m_ngxVelocityStats.depthClears;

    if (++m_ngxVelocityWindow.frames >= kNgxVelocityWindowFrames) {
      Logger::info(str::format(
        "[RTX NGX Passthrough][velocity] over ", m_ngxVelocityWindow.frames, " frames: ",
        m_ngxVelocityWindow.captured, " captured (", m_ngxVelocityWindow.capturedSkinned, " skinned, ",
        m_ngxVelocityWindow.capturedDynamic, " CPU-modified), ",
        m_ngxVelocityWindow.exactMatches, " paired to their own history, ",
        m_ngxVelocityWindow.newRegistrations, " registered anew, ",
        m_ngxVelocityWindow.pairedBeyondBounds, " paired beyond the motion bounds, skipped: ",
        m_ngxVelocityWindow.skippedNoCamera, " no camera / ",
        m_ngxVelocityWindow.skippedBudget, " over budget / ",
        m_ngxVelocityWindow.skippedZDisabled, " depth test off / ",
        m_ngxVelocityWindow.skippedInstanceCap, " over the instance cap; tracking ",
        m_ngxVelocityObjectCache.size(), " identities; ", m_ngxVelocityWindow.depthClears,
        " mid-scene depth clears; scene-wide transform offset (",
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
    if (!m_frameOptions.ngxPrePostProcess)
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

  void D3D9Rtx::countNgxPostInjectionSceneColorConsumer(const DrawContext& drawContext) {
    // The injection ordinal is only useful if the count it comes from covers the whole frame, and
    // the passes that decide whether this frame's injection was the last one are precisely the ones
    // that come after it. Same shape test as the pre-injection path, deliberately: a mismatch
    // between the two would bias the count and walk the injection point away from the end.
    if (!m_frameOptions.ngxPrePostProcess || m_ngxSceneColorImage == nullptr ||
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
      bool consumesSceneColor = sampledImage == m_ngxSceneColorImage.ptr();

      for (uint32_t r = 0; !consumesSceneColor && r < m_ngxSceneColorResolveCount; r++) {
        consumesSceneColor = m_ngxSceneColorResolves[r].ptr() == sampledImage;
      }

      if (consumesSceneColor) {
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
      // stretch-replacement path below) and while the debug visualization is off (the
      // debug image would be mangled by the game's post chain; the late injection point
      // writes it to the final output instead). If this trigger never fires (post
      // disabled, no scene color identified), the backbuffer trigger below is the fallback.
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

      const bool candidateShape =
        m_frameOptions.ngxPrePostProcess &&
        likelyPostProcessQuad &&
        !targetsBackbuffer && !m_ngxBackbufferDrawSeenThisFrame &&
        m_ngxSceneColorImage != nullptr && sceneViewportUsable &&
        backBufferWidth != 0 && backBufferHeight != 0 &&
        uint64_t(m_ngxSceneViewport.Width) * 100 >= uint64_t(backBufferWidth) * 97 &&
        uint64_t(m_ngxSceneViewport.Height) * 100 >= uint64_t(backBufferHeight) * 97 &&
        (renderTargetTexture == nullptr || renderTargetTexture->GetImage().ptr() != m_ngxSceneColorImage.ptr());

      if (candidateShape) {
        const uint32_t rtSamplerMask = m_parent->GetActiveRTTextures() & m_parent->m_psShaderMasks.samplerMask;

        for (const uint32_t i : bit::BitMask(rtSamplerMask)) {
          D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[i]);
          if (texture == nullptr || texture->GetImage() == nullptr) {
            continue;
          }

          const DxvkImage* sampledImage = texture->GetImage().ptr();

          Rc<DxvkImage> matchedTarget;
          if (sampledImage == m_ngxSceneColorImage.ptr()) {
            matchedTarget = m_ngxSceneColorImage;
          } else {
            for (uint32_t r = 0; r < m_ngxSceneColorResolveCount; r++) {
              if (m_ngxSceneColorResolves[r].ptr() == sampledImage) {
                matchedTarget = m_ngxSceneColorResolves[r];
                break;
              }
            }
          }

          if (matchedTarget == nullptr) {
            continue;
          }

          m_ngxPrePostCandidatesThisFrame++;
          m_ngxReadsSinceLastClear++;

          if (m_ngxFrameReadCount < kNgxFrameShapeSlots) {
            m_ngxFrameReadDraws[m_ngxFrameReadCount++] = m_drawCallID;
          }

          // Inject at the first read of the scene color that follows the frame's last depth-writing
          // draw. Anything the game draws after the injection is composited onto the upscaler's
          // finished output, un-jittered and never anti-aliased; UE3 runs post-process effects
          // inside its DPG loop, so the earlier reads are interleaved with the scene rather than
          // after it, and injecting at one of those leaves the last DPG - first person mesh and
          // held weapon included - on the wrong side of it.
          //
          // Where the scene ends is only knowable once the frame is over, so it is predicted from
          // the previous frame's shape. Counted from the most recent scene depth clear, which is
          // the only form of the count that holds still: frames differ in how many reads land
          // before the clear, so a whole-frame count inherits that variation and alternates, while
          // the stretch between the clear and the end of the scene does not. Frames with no
          // detected clear use the whole-frame count instead - menu frames, whose scene ends before
          // any read, need it anyway.
          const bool haveClearThisFrame = m_ngxFrameClearCount > 0;

          const bool atInjectionPoint = haveClearThisFrame
            ? m_ngxReadsSinceLastClear >= m_ngxReadsAfterClearInsideScene + 1
            : m_ngxPrePostCandidatesThisFrame >= m_ngxPrePostReadsInsideScene + 1;

          if (!atInjectionPoint) {
            break;
          }

          m_ngxColorTargetImage = matchedTarget;
          // When the consumed image is a resolve copy, mirror the DLSS output into the
          // scene color surface as well: later post passes may re-resolve from it (UE3
          // scene color resolves are surface -> texture copies)
          m_ngxColorMirrorImage = (matchedTarget.ptr() != m_ngxSceneColorImage.ptr()) ? m_ngxSceneColorImage : nullptr;
          m_ngxSubrect.offset = { int32_t(m_ngxSceneViewport.X), int32_t(m_ngxSceneViewport.Y) };
          m_ngxSubrect.extent = { m_ngxSceneViewport.Width, m_ngxSceneViewport.Height };
          m_ngxColorSubrectOffset = m_ngxSubrect.offset;

          triggerInjection = true;
          m_ngxFrameInjectionDraw = m_drawCallID;

          ONCE(Logger::info(str::format("[RTX NGX Passthrough] Pre-post-process injection engaged: DLSS runs on the ",
                                        (m_ngxColorMirrorImage != nullptr ? "resolved scene color" : "scene color"),
                                        " before the game's post-process chain.")));
          break;
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
          backBufferWidth != 0 && backBufferHeight != 0 &&
          uint64_t(m_ngxSceneViewport.Width) * 100 <= uint64_t(backBufferWidth) * 97 &&
          uint64_t(m_ngxSceneViewport.Height) * 100 <= uint64_t(backBufferHeight) * 97;

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
    // Draws issued internally by the deferred UI overlay replay bypass classification and
    // execute as plain raster draws
    if (m_replayingDeferredUiDraws) {
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    // first-frame lazy init; steady-state refreshes happen once per frame in EndFrame
    if (unlikely(!m_frameOptions.valid)) {
      refreshFrameOptionCache();
    }

    // NGX passthrough mode: no scene capture, everything rasterizes; takes precedence over
    // the ray traced path
    if (m_frameOptions.ngxPassthroughMode && m_enableDrawCallConversion) {
      return prepareDrawForNgxPassthrough(context);
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

    // NGX passthrough mode: no scene capture, everything rasterizes; takes precedence over
    // the ray traced path
    if (m_frameOptions.ngxPassthroughMode && m_enableDrawCallConversion) {
      return prepareDrawForNgxPassthrough(drawContext);
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

  namespace {
    constexpr char kUe3TextureSpreadCachePath[] = "rtx-remix/ue3TextureSpread.cache";
    constexpr uint64_t kUe3TextureSpreadCacheMagic = 0x3144525053334555ull; // "UE3SPRD1"
    constexpr uint32_t kUe3TextureSpreadCacheMaxEntries = 1u << 20;
    constexpr uint32_t kUe3TextureSpreadSaveIntervalFrames = 600;
  }

  void D3D9Rtx::loadUe3TextureSpreadCache() {
    m_ue3TextureSpreadLoaded = true;

    std::ifstream file(kUe3TextureSpreadCachePath, std::ios::binary);
    if (!file.is_open())
      return;

    uint64_t magic = 0;
    uint32_t entryCount = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&entryCount), sizeof(entryCount));
    if (!file || magic != kUe3TextureSpreadCacheMagic || entryCount > kUe3TextureSpreadCacheMaxEntries)
      return;

    for (uint32_t i = 0; i < entryCount; i++) {
      XXH64_hash_t texHash = 0;
      uint8_t count = 0;
      file.read(reinterpret_cast<char*>(&texHash), sizeof(texHash));
      file.read(reinterpret_cast<char*>(&count), sizeof(count));
      Ue3TextureMaterialSpread spread;
      if (!file || count > spread.psHashes.size())
        return;
      for (uint8_t p = 0; p < count; p++)
        file.read(reinterpret_cast<char*>(&spread.psHashes[p]), sizeof(XXH64_hash_t));
      if (!file)
        return;
      spread.count = count;
      m_ue3TextureMaterialSpread[texHash] = spread;
    }

    Logger::info(str::format(
      "[RTX-Compatibility][UE3] Loaded texture material-spread cache: ", entryCount, " textures"));
  }

  void D3D9Rtx::saveUe3TextureSpreadCache() {
    if (!m_ue3TextureSpreadDirty)
      return;
    m_ue3TextureSpreadDirty = false;

    std::ofstream file(kUe3TextureSpreadCachePath, std::ios::binary | std::ios::trunc);
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

    // Update the effective settings before the next frame.
    applyNgxPassthroughScreenPercentage();
    
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

    // persist newly discovered texture material-spread so the next session scores
    // deterministically from its first frame instead of re-converging
    if (m_ue3TextureSpreadDirty &&
        currentReflexFrameId >= m_ue3TextureSpreadLastSaveFrame + kUe3TextureSpreadSaveIntervalFrames) {
      m_ue3TextureSpreadLastSaveFrame = uint32_t(currentReflexFrameId);
      saveUe3TextureSpreadCache();
    }

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

    // Flush any pending game and RTX work
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

    DrawCallState::refreshCategoryLookupTable();

    // Reset for the next frame
    m_rtxInjectTriggered = false;
    m_drawCallID = 0;
    m_seenCameraPositionsPrev = std::move(m_seenCameraPositions);
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

    m_stagedBones.clear();
  }

  void D3D9Rtx::OnPresent(const Rc<DxvkImage>& targetImage) {
    // Inform backend of present
    m_parent->EmitCs([targetImage](DxvkContext* ctx) { static_cast<RtxContext*>(ctx)->onPresent(targetImage); });
  }
}
