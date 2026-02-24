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
#include "../dxvk/rtx_render/rtx_terrain_baker.h"
#include "../dxvk/rtx_render/rtx_matrix_helpers.h"
#include "../dxso/dxso_decoder.h"

#include <cctype>

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

    // compatibility fallback for certain stretched textures - todo fix this later
    constexpr std::array<XXH64_hash_t, 6> kVsTexcoordCaptureOutlierHashes = {
      0x7F3396C65628A031ull,
      0xADED98814275ACD0ull,
      0xEE12788D359E61B6ull,
      0x5E5236906134EEC6ull,
      0x57F8CD8A34E02ACBull,
      0x942CAAF9A256B683ull,
    };

    inline bool isVsTexcoordCaptureOutlierHash(const XXH64_hash_t h) {
      for (const XXH64_hash_t outlierHash : kVsTexcoordCaptureOutlierHashes) {
        if (h == outlierHash)
          return true;
      }
      return false;
    }

    fast_unordered_set s_loggedNonPrimaryRtDescHashes;
    fast_unordered_set s_loggedSampledRtDescHashes;

    static XXH64_hash_t hashDxsoBytecode(const std::vector<uint8_t>& bytecode) {
      if (bytecode.empty())
        return 0;
      return XXH3_64bits(bytecode.data(), bytecode.size());
    }

    constexpr uint16_t kD3dxRegisterSetSampler = 3u;

    constexpr uint8_t kPsSamplerSemanticEngineAuxiliary = 1u << 0;
    constexpr uint8_t kPsSamplerSemanticLightmap        = 1u << 1;
    constexpr uint8_t kPsSamplerSemanticMaterialTexture = 1u << 2;
    constexpr uint8_t kPsSamplerSemanticNonDiffuse      = 1u << 3;

    static uint8_t classifyPixelSamplerSemanticFlags(const std::string& samplerName) {
      std::string lowerName;
      lowerName.reserve(samplerName.size());
      for (const char c : samplerName)
        lowerName.push_back(char(std::tolower(static_cast<unsigned char>(c))));

      auto contains = [&](const char* token) {
        return lowerName.find(token) != std::string::npos;
      };

      uint8_t flags = 0;

      if (contains("texture2d_") || contains("texturecube_") || contains("texture3d_") ||
          contains("materialtexture") || contains("materialsampler")) {
        flags |= kPsSamplerSemanticMaterialTexture;
      }

      if (contains("scenecolor") || contains("scenedepth") || contains("lightattenuation") ||
          contains("previouslighting") || contains("exposuretexture") ||
          contains("previousexposure") || contains("scenedownsampled") ||
          contains("saturationmasktexture") || contains("randomangletexture") ||
          contains("bslinetexture") || contains("colorcurvesktexture") ||
          contains("colorcurvesmtexture") || contains("blurredimage") ||
          contains("shadowdepth") || contains("shadowvariance") ||
          contains("shadowtexture") || contains("velocitybuffer") ||
          contains("ambientocclusiontexture") || contains("aohistorytexture") ||
          contains("randomnormaltexture") || contains("filtertexture") ||
          contains("scenecolorscratchtexture") || contains("ldrtranslucencytexture") ||
          contains("accumulateddistortiontexture") || contains("accumulatedfrontfaceslineintegraltexture") ||
          contains("accumulatedbackfaceslineintegraltexture") || contains("scenecoloruitexture") ||
          contains("blurreduitexture") || contains("sceneblurtexture") ||
          contains("masktexture")) {
        flags |= kPsSamplerSemanticEngineAuxiliary;
      }

      if (contains("lightmap")) {
        flags |= kPsSamplerSemanticLightmap;
        flags |= kPsSamplerSemanticEngineAuxiliary;
      }

      if (contains("normal") || contains("specular") || contains("roughness") ||
          contains("gloss") || contains("metallic") || contains("metalness") ||
          contains("ambientocclusion") || contains("_ao") || contains("ao_") ||
          contains("heightmap") || contains("height") || contains("bump") ||
          contains("opacity") || contains("alphamask") || contains("mask")) {
        flags |= kPsSamplerSemanticNonDiffuse;
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

      while (decoder.decodeInstruction(iter)) {
        const auto& ctx = decoder.getInstructionContext();
        const DxsoOpcode op = ctx.instruction.opcode;

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

        if ((ctx.dst.id.type == DxsoRegisterType::Temp || ctx.dst.id.type == DxsoRegisterType::TempFloat16) &&
            ctx.dst.id.num < tempToTexcoord.size()) {
          int32_t derived = -1;
          bool writesTrackedTemp = false;
          PsTexcoordScaleHint derivedScaleHint;
          bool hasConstOnlyHint = false;
          PsTexcoordScaleHint constOnlyHint;

          auto assignFromSource = [&](const DxsoRegister& srcReg) {
            derived = getTexcoordFromRegister(srcReg);
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
          case DxsoOpcode::Abs:
          case DxsoOpcode::Nrm:
          case DxsoOpcode::DsX:
          case DxsoOpcode::DsY:
            writesTrackedTemp = true;
            assignFromSource(ctx.src[0]);
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
            const int32_t tc0 = getTexcoordFromRegister(ctx.src[0]);
            const int32_t tc1 = getTexcoordFromRegister(ctx.src[1]);

            if (tc0 >= 0 && tc1 < 0) {
              assignFromSource(ctx.src[0]);
              if (derived >= 0) {
                const float sign = (op == DxsoOpcode::Sub) ? -1.0f : 1.0f;
                if (!applyOffsetFromConstant(ctx.src[1], derivedScaleHint, sign)) {
                  PsTexcoordScaleHint tempHint;
                  if (getScaleHintFromRegister(ctx.src[1], tempHint)) {
                    if (tempHint.constReg >= 0 || tempHint.immediateValid) {
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
                }
              }
            } else if (tc1 >= 0 && tc0 < 0) {
              assignFromSource(ctx.src[1]);
              if (derived >= 0) {
                if (op == DxsoOpcode::Add) {
                  if (!applyOffsetFromConstant(ctx.src[0], derivedScaleHint, +1.0f)) {
                    PsTexcoordScaleHint tempHint;
                    if (getScaleHintFromRegister(ctx.src[0], tempHint)) {
                      if (tempHint.constReg >= 0 || tempHint.immediateValid) {
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
                  }
                } else {
                  // handle `const - uv` as mirrored UV with offset
                  if (derivedScaleHint.constReg < 0 && !derivedScaleHint.immediateValid) {
                    derivedScaleHint.immediateValid = true;
                    derivedScaleHint.immediateU = -1.0f;
                    derivedScaleHint.immediateV = -1.0f;
                    applyOffsetFromConstant(ctx.src[0], derivedScaleHint, +1.0f);
                  } else {
                    // avoid producing incorrect transforms when an existing scale needs negationn
                    derivedScaleHint = PsTexcoordScaleHint{};
                  }
                }
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
            const int32_t tc0 = getTexcoordFromRegister(ctx.src[0]);
            const int32_t tc1 = getTexcoordFromRegister(ctx.src[1]);
            if (tc0 >= 0 && tc1 < 0) {
              assignFromSource(ctx.src[0]);
            } else if (tc1 >= 0 && tc0 < 0) {
              assignFromSource(ctx.src[1]);
            } else {
              derived = mergeSingle({ tc0, tc1 });
            }
            break;
          }
          case DxsoOpcode::Mul: {
            writesTrackedTemp = true;
            const int32_t tc0 = getTexcoordFromRegister(ctx.src[0]);
            const int32_t tc1 = getTexcoordFromRegister(ctx.src[1]);
            const bool src0Const = isFloatConstantRegisterType(ctx.src[0].id.type);
            const bool src1Const = isFloatConstantRegisterType(ctx.src[1].id.type);

            if (tc0 >= 0 && tc1 < 0 && src1Const) {
              assignFromSource(ctx.src[0]);
              if (derived >= 0)
                loadScaleHintFromConstant(ctx.src[1], derivedScaleHint);
            } else if (tc1 >= 0 && tc0 < 0 && src0Const) {
              assignFromSource(ctx.src[1]);
              if (derived >= 0)
                loadScaleHintFromConstant(ctx.src[0], derivedScaleHint);
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
            break;
          }
          case DxsoOpcode::Mad: {
            writesTrackedTemp = true;
            const int32_t tc0 = getTexcoordFromRegister(ctx.src[0]);
            const int32_t tc1 = getTexcoordFromRegister(ctx.src[1]);
            const int32_t tc2 = getTexcoordFromRegister(ctx.src[2]);
            const bool src0Const = isFloatConstantRegisterType(ctx.src[0].id.type);
            const bool src1Const = isFloatConstantRegisterType(ctx.src[1].id.type);
            const bool src2Const = isFloatConstantRegisterType(ctx.src[2].id.type);

            if (tc0 >= 0 && tc1 < 0 && src1Const) {
              assignFromSource(ctx.src[0]);
              if (derived >= 0) {
                loadScaleHintFromConstant(ctx.src[1], derivedScaleHint);
                if (src2Const)
                  applyOffsetFromConstant(ctx.src[2], derivedScaleHint, +1.0f);
              }
            } else if (tc1 >= 0 && tc0 < 0 && src0Const) {
              assignFromSource(ctx.src[1]);
              if (derived >= 0) {
                loadScaleHintFromConstant(ctx.src[0], derivedScaleHint);
                if (src2Const)
                  applyOffsetFromConstant(ctx.src[2], derivedScaleHint, +1.0f);
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
            break;
          }
          default:
            break;
          }

          if (derived >= 0) {
            tempToTexcoord[ctx.dst.id.num] = int8_t(derived);
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

        uint32_t sampledSampler = ~0u;
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
          break;
        case DxsoOpcode::TexBem:
        case DxsoOpcode::TexBemL:
          sampledSampler = ctx.dst.id.num;
          coordRegStorage = ctx.dst;
          coordRegStorage.id.type = DxsoRegisterType::PixelTexcoord;
          coordRegStorage.id.num = ctx.dst.id.num;
          coordRegStorage.swizzle = DxsoRegSwizzle(0, 1, 2, 3);
          coordReg = &coordRegStorage;
          break;
        case DxsoOpcode::TexReg2Ar:
        case DxsoOpcode::TexReg2Gb:
        case DxsoOpcode::TexReg2Rgb:
          sampledSampler = ctx.dst.id.num;
          coordReg = &ctx.src[0];
          break;
        case DxsoOpcode::TexM3x2Tex:
        case DxsoOpcode::TexM3x3Tex:
        case DxsoOpcode::TexM3x3Spec:
        case DxsoOpcode::TexM3x3VSpec:
        case DxsoOpcode::TexDp3Tex:
          sampledSampler = ctx.dst.id.num;
          coordReg = &ctx.src[0];
          break;
        default:
          break;
        }

        if (coordReg != nullptr && sampledSampler == samplerIdx) {
          result.sampleCount = result.sampleCount < std::numeric_limits<uint16_t>::max()
            ? uint16_t(result.sampleCount + 1u)
            : std::numeric_limits<uint16_t>::max();

          const DxsoRegister* swizzleReg = coordReg;
          int32_t tc = getTexcoordFromRegister(*swizzleReg);
          uint8_t coordProvenance = getCoordProvenanceFromRegister(*coordReg);

          const std::array<DxsoRegister, 4> fallbackRegs = {
            ctx.src[0], ctx.src[1], ctx.src[2], ctx.dst
          };

          if ((coordProvenance & kCoordProvTexcoord) == 0) {
            for (const DxsoRegister& fallbackReg : fallbackRegs) {
              coordProvenance |= getCoordProvenanceFromRegister(fallbackReg);
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
            auto& agg = perTexcoordScaleHints[uint32_t(tc)];
            if (!agg.valid) {
              agg.valid = true;
              agg.hint = sampleScaleHint;
            } else if (!isSameScaleHint(agg.hint, sampleScaleHint)) {
              agg.conflict = true;
            }
          }
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

      {
        const auto& ctab = decoder.getCtabInfo();
        if (ctab.m_size != 0) {
          for (const auto& c : ctab.m_constantData) {
            if (c.registerSet != kD3dxRegisterSetSampler || c.registerCount == 0)
              continue;

            const uint64_t regBegin = c.registerIndex;
            const uint64_t regEnd = regBegin + c.registerCount;
            if (uint64_t(samplerIdx) >= regBegin && uint64_t(samplerIdx) < regEnd)
              result.semanticFlags |= classifyPixelSamplerSemanticFlags(c.name);
          }
        }
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
      bool* outUsedTranspose = nullptr) {

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
      return true;
    }
  }

  D3D9Rtx::D3D9Rtx(D3D9DeviceEx* d3d9Device, bool enableDrawCallConversion)
    : m_rtStagingData(d3d9Device->GetDXVKDevice(), "RtxStagingDataAlloc: D3D9", (VkMemoryPropertyFlagBits) (VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
    , m_parent(d3d9Device)
    , m_enableDrawCallConversion(enableDrawCallConversion)
    , m_pGeometryWorkers(enableDrawCallConversion ? std::make_unique<GeometryProcessor>(numGeometryProcessingThreads(), "geometry-processing") : nullptr) {

    // add space for 256 objects skinned with 256 bones each.
    m_stagedBones.resize(256 * 256);
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

    if (enableIndexBufferMemoization() && indexCtx.ibo != nullptr) {
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

  bool D3D9Rtx::prepareVertexCapture(const int vertexIndexOffset) {
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

    uint32_t capturedTexcoordOutputRegister = FindVsTexcoordOutputRegister(vertexShader, m_texcoordIndex);
    if (capturedTexcoordOutputRegister == std::numeric_limits<uint32_t>::max())
      capturedTexcoordOutputRegister = FindVsTexcoordOutputRegisterByRegNumber(vertexShader, m_texcoordIndex);
    if (capturedTexcoordOutputRegister == std::numeric_limits<uint32_t>::max())
      capturedTexcoordOutputRegister = FindUniqueVsTexcoordOutputRegister(vertexShader);
    if (forceIaTexcoordForOutlier)
      capturedTexcoordOutputRegister = std::numeric_limits<uint32_t>::max();

    // Shader path with vertex capture: prefer VS output TEXCOORD to preserve any VS-side UV math.
    if (BoundShaderHasAnyUsageIndex(vertexShader, DxsoUsage::Texcoord, false) &&
        capturedTexcoordOutputRegister != std::numeric_limits<uint32_t>::max()) {
      // Known offset for vertex capture buffers
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
    const bool hasBlendWeightsAndIndices = hasBlendWeights && hasBlendIndices;
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

    // logging for shading issues, todo: remove this later
    {
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

        Logger::info(str::format(
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

    if (vsOutputsNormal && useVertexCapturedNormals()) {
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
    DxvkBufferSlice streamCopies[caps::MaxStreams] {};

    // Process vertex buffers from CPU
    for (const auto& element : d3d9State().vertexDecl->GetElements()) {
      // Get vertex context
      const VertexContext& ctx = vertexContext[element.Stream];

      if (ctx.mappedSlice.handle == VK_NULL_HANDLE)
        continue;

      ScopedCpuProfileZoneN("Process Vertices");
      const int32_t vertexOffset = ctx.offset + ctx.stride * vertexIndexOffset;
      const uint32_t numVertexBytes = ctx.stride * geoData.vertexCount;

      // Validating index data here, vertexCount and vertexIndexOffset accounts for the min/max indices
      if (RtxOptions::validateCPUIndexData()) {
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
        if (m_texcoordIndex <= MAXD3DDECLUSAGEINDEX && element.UsageIndex == m_texcoordIndex)
          targetBuffer = &geoData.texcoordBuffer;
        break;
      case D3DDECLUSAGE_COLOR:
        if (element.UsageIndex == 0 &&
            !RtxOptions::ignoreAllVertexColorBakedLighting() &&
            !lookupHash(RtxOptions::ignoreBakedLightingTextures(), m_activeDrawCallState.materialData.colorTextures[0].getImageHash())) {
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

        if (element.Usage == D3DDECLUSAGE_TEXCOORD &&
            m_texcoordCompV == uint8_t(m_texcoordCompU + 1) &&
            (m_texcoordCompU != 0 || m_texcoordCompV != 1)) {
          auto describeTexcoordFormat = [](VkFormat format, uint32_t& outCompCount, uint32_t& outCompSize) {
            outCompCount = 0;
            outCompSize = 0;
            switch (format) {
            case VK_FORMAT_R16G16_SFLOAT:
              outCompCount = 2; outCompSize = 2; return true;
            case VK_FORMAT_R16G16B16A16_SFLOAT:
              outCompCount = 4; outCompSize = 2; return true;
            case VK_FORMAT_R32G32_SFLOAT:
              outCompCount = 2; outCompSize = 4; return true;
            case VK_FORMAT_R32G32B32_SFLOAT:
              outCompCount = 3; outCompSize = 4; return true;
            case VK_FORMAT_R32G32B32A32_SFLOAT:
              outCompCount = 4; outCompSize = 4; return true;
            default:
              return false;
            }
          };

          uint32_t compCount = 0;
          uint32_t compSize = 0;
          if (describeTexcoordFormat(fmt, compCount, compSize) &&
              m_texcoordCompV < compCount) {
            const uint32_t byteOffset = uint32_t(m_texcoordCompU) * compSize;
            if ((byteOffset % 4u) == 0u) {
              elementOffset += byteOffset;
              if (compSize == 2)
                fmt = VK_FORMAT_R16G16_SFLOAT;
              else if (compSize == 4)
                fmt = VK_FORMAT_R32G32_SFLOAT;
            }
          }
        }

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

  bool D3D9Rtx::processRenderState() {
    DrawCallTransforms& transformData = m_activeDrawCallState.transformData;
    m_forceIaTexcoordForOutlier = false;

    // When games use vertex shaders, the object to world transforms can be unreliable, and so we can ignore them.
    const bool useObjectToWorldTransform = !m_parent->UseProgrammableVS() || (m_parent->UseProgrammableVS() && useVertexCapture() && useWorldMatricesForShaders());
    transformData.objectToWorld = useObjectToWorldTransform ? d3d9State().transforms[GetTransformIndex(D3DTS_WORLD)] : Matrix4();

    transformData.worldToView = d3d9State().transforms[GetTransformIndex(D3DTS_VIEW)];
    transformData.viewToProjection = d3d9State().transforms[GetTransformIndex(D3DTS_PROJECTION)];

    const bool usesProgrammableVs = m_parent->UseProgrammableVS();
    const D3D9CommonShader* vertexShaderCommon =
      usesProgrammableVs && d3d9State().vertexShader.ptr() != nullptr
        ? d3d9State().vertexShader->GetCommonShader()
        : nullptr;

    // for UE3 we shouild parse shader CTAB once per unique vertex shader to locate reserved constants
    // (camera/object transforms and optional skinning metadata)
    const Ue3VsShaderCtabInfo* ue3CtabInfoPtr = nullptr;
    m_currentUe3CtabInfo.reset();
    bool ue3CameraUsedTranspose = false;
    const bool needsUe3CtabInfo =
      ue3CameraFromShaderConstants() ||
      ue3ObjectToWorldFromShaderConstants() ||
      useVertexCapture();
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
              } else if (c.registerCount >= 9u && (c.registerCount % 3u) == 0u && c.registerIndex >= 5u) {
                // fallback for stripped/renamed symbols, in UE3 this is typically a large contiguous
                // c-register range (3 registers per bone) usually starting after c0..c4 camera constants
                if (c.registerCount > inferredBoneMatricesRegisterCount) {
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

          if (!info.hasBoneMatrices && inferredBoneMatricesRegisterCount >= 9u) {
            info.hasBoneMatrices = true;
            info.boneMatricesRegisterIndex = inferredBoneMatricesRegisterIndex;
            info.boneMatricesRegisterCount = inferredBoneMatricesRegisterCount;
          }
        } catch (...) {
          return info;
        }

        return info;
      };

      const auto& bytecode = vertexShaderCommon->GetBytecode();
      const XXH64_hash_t shaderHash = (bytecode.size() > 0)
        ? XXH3_64bits(bytecode.data(), bytecode.size())
        : 0;

      if (shaderHash != 0) {
        auto it = m_ue3VsShaderCtabCache.find(shaderHash);
        if (it == m_ue3VsShaderCtabCache.end()) {
          m_ue3VsShaderCtabCache.emplace(shaderHash, parseCtabInfo(bytecode));
          it = m_ue3VsShaderCtabCache.find(shaderHash);
        }

        if (it != m_ue3VsShaderCtabCache.end()) {
          ue3CtabInfoPtr = &it->second;
          m_currentUe3CtabInfo = *ue3CtabInfoPtr;
        }
      }
    }

    // in UE3 camera matrices are often provided only via reserved shader constants (VSR_ViewProjMatrix/VSR_ViewOrigin)
    if (usesProgrammableVs && ue3CameraFromShaderConstants()) {
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

      auto tryApplyFromConstants = [&](Matrix4& outWorldToView, Matrix4& outViewToProjection, bool& outUsedTranspose) -> bool {
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

        if (m_ue3CameraConstantsCache.valid && m_ue3CameraConstantsCache.hash == constantsHash) {
          outWorldToView = m_ue3CameraConstantsCache.worldToView;
          outViewToProjection = m_ue3CameraConstantsCache.viewToProjection;
          outUsedTranspose = m_ue3CameraConstantsCache.usedTranspose;
          return true;
        }

        Matrix4 ue3WorldToView;
        Matrix4 ue3ViewToProjection;
        bool usedTranspose = false;
        if (!tryExtractUe3WorldToViewAndProjectionFromShaderConstants(d3d9State().vsConsts, viewProjReg, viewOriginReg, ue3WorldToView, ue3ViewToProjection, &usedTranspose))
          return false;

        m_ue3CameraConstantsCache.hash = constantsHash;
        m_ue3CameraConstantsCache.valid = true;
        m_ue3CameraConstantsCache.usedTranspose = usedTranspose;
        m_ue3CameraConstantsCache.worldToView = ue3WorldToView;
        m_ue3CameraConstantsCache.viewToProjection = ue3ViewToProjection;

        outWorldToView = ue3WorldToView;
        outViewToProjection = ue3ViewToProjection;
        outUsedTranspose = usedTranspose;
        return true;
      };

      Matrix4 ue3WorldToView;
      Matrix4 ue3ViewToProjection;

      if (tryApplyFromConstants(ue3WorldToView, ue3ViewToProjection, ue3CameraUsedTranspose)) {
        ONCE(Logger::info(str::format("[RTX-Compatibility] UE3 camera matrices extracted from shader constants (viewProjReg=c",
                                      viewProjReg, "..c", viewProjReg + 3, ", viewOriginReg=c", viewOriginReg, ").")));
        transformData.worldToView = ue3WorldToView;
        transformData.viewToProjection = ue3ViewToProjection;
      } else {
        // debugging, todo: remove later
        if (viewProjReg + 3 < caps::MaxFloatConstantsSoftware && viewOriginReg < caps::MaxFloatConstantsSoftware) {
          const Vector4 c0 = d3d9State().vsConsts.fConsts[viewProjReg + 0];
          const Vector4 c1 = d3d9State().vsConsts.fConsts[viewProjReg + 1];
          const Vector4 c2 = d3d9State().vsConsts.fConsts[viewProjReg + 2];
          const Vector4 c3 = d3d9State().vsConsts.fConsts[viewProjReg + 3];
          const Vector4 cam = d3d9State().vsConsts.fConsts[viewOriginReg];

          ONCE(Logger::info(str::format(
            "[RTX-Compatibility] UE3 camera extraction failed (viewProjReg=c", viewProjReg, "..c", viewProjReg + 3,
            ", viewOriginReg=c", viewOriginReg, "). "
            "c0={", c0.x, ", ", c0.y, ", ", c0.z, ", ", c0.w, "} "
            "c1={", c1.x, ", ", c1.y, ", ", c1.z, ", ", c1.w, "} "
            "c2={", c2.x, ", ", c2.y, ", ", c2.z, ", ", c2.w, "} "
            "c3={", c3.x, ", ", c3.y, ", ", c3.z, ", ", c3.w, "} "
            "cam={", cam.x, ", ", cam.y, ", ", cam.z, ", ", cam.w, "}")));
        } else {
          ONCE(Logger::info(str::format(
            "[RTX-Compatibility] UE3 camera extraction failed (out of bounds registers: viewProjReg=c", viewProjReg,
            ", viewOriginReg=c", viewOriginReg, ").")));
        }
      }
    }

    // in UE3 perobject LocalToWorld is typically passed as a shader constant not as a fixed-function world transform
    if (usesProgrammableVs && ue3ObjectToWorldFromShaderConstants() && ue3CtabInfoPtr != nullptr) {
      const Ue3VsShaderCtabInfo& ctabInfo = *ue3CtabInfoPtr;

      if (ctabInfo.hasLocalToWorld) {
        const uint32_t reg = ctabInfo.localToWorldRegisterIndex;
        if (reg + 3 < caps::MaxFloatConstantsSoftware) {
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
          bool hasWorldToLocal = false;
          Matrix4 worldToLocalRaw;
          Matrix4 worldToLocalTransposed;
          if (ctabInfo.hasWorldToLocal) {
            const uint32_t w2lReg = ctabInfo.worldToLocalRegisterIndex;
            if (w2lReg + 2 < caps::MaxFloatConstantsSoftware) {
              const Vector4 c0 = d3d9State().vsConsts.fConsts[w2lReg + 0];
              const Vector4 c1 = d3d9State().vsConsts.fConsts[w2lReg + 1];
              const Vector4 c2 = d3d9State().vsConsts.fConsts[w2lReg + 2];

              worldToLocalRaw = Matrix4();
              worldToLocalRaw[0] = Vector4(c0.x, c0.y, c0.z, 0.0f);
              worldToLocalRaw[1] = Vector4(c1.x, c1.y, c1.z, 0.0f);
              worldToLocalRaw[2] = Vector4(c2.x, c2.y, c2.z, 0.0f);
              worldToLocalRaw[3] = Vector4(0.0f, 0.0f, 0.0f, 1.0f);
              worldToLocalTransposed = transpose(worldToLocalRaw);
              hasWorldToLocal = true;
            }
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

          ONCE(Logger::info("[RTX-Compatibility] UE3 LocalToWorld extracted from vertex shader constants (CTAB)"));
          transformData.objectToWorld = localToWorld;
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

    // Process textures
    if (m_parent->UseProgrammablePS()) {
      return processTextures<false>();
    } else {
      return processTextures<true>();
    }
  }

  D3D9Rtx::DrawCallType D3D9Rtx::makeDrawCallType(const DrawContext& drawContext) {
    // Track the drawcall index so we can use it in rtx_context
    m_activeDrawCallState.drawCallID = m_drawCallID++;
    m_activeDrawCallState.isDrawingToRaytracedRenderTarget = false;
    m_activeDrawCallState.isUsingRaytracedRenderTarget = false;

    if (m_drawCallID < (uint32_t)RtxOptions::drawCallRange().x ||
        m_drawCallID > (uint32_t)RtxOptions::drawCallRange().y) {
      return { RtxGeometryStatus::Ignored, false };
    }

    // Raytraced Render Target Support
    // If the bound texture for this draw call is one that has been used as a render target then store its id
    if (RtxOptions::RaytracedRenderTarget::enable()) {
      for (uint32_t i : bit::BitMask(m_parent->GetActiveRTTextures())) {
        D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[i]);
        if (!texture || texture->GetImage() == nullptr)
          continue;

        const XXH64_hash_t texDescHash = texture->GetImage()->getDescriptorHash();
        if (lookupHash(RtxOptions::raytracedRenderTargetTextures(), texDescHash) ||
            lookupHash(m_autoRaytracedRenderTargetDescHashes, texDescHash)) {
          m_activeDrawCallState.isUsingRaytracedRenderTarget = true;
        }
      }
    }

    if (m_parent->UseProgrammableVS() && !useVertexCapture()) {
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

    if (!RtxOptions::enableAlphaTest() && m_parent->IsAlphaTestEnabled()) {
      ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Raytracing an alpha-tested draw call when alpha-tested objects disabled in RT. Ignoring.")));
      return { RtxGeometryStatus::Ignored, false };
    }

    if (!RtxOptions::enableAlphaBlend() && d3d9State().renderStates[D3DRS_ALPHABLENDENABLE]) {
      ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Raytracing an alpha-blended draw call when alpha-blended objects disabled in RT. Ignoring.")));
      return { RtxGeometryStatus::Ignored, false };
    }
    
    if (m_activeOcclusionQueries > 0) {
      ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Trying to raytrace an occlusion query. Ignoring.")));
      return { RtxGeometryStatus::Rasterized, false };
    }

    if (d3d9State().renderTargets[kRenderTargetIndex] == nullptr) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped drawcall, as no color render target bound."));
      return { RtxGeometryStatus::Ignored, false };
    }

    constexpr DWORD rgbWriteMask = D3DCOLORWRITEENABLE_RED | D3DCOLORWRITEENABLE_GREEN | D3DCOLORWRITEENABLE_BLUE;
    if ((d3d9State().renderStates[ColorWriteIndex(kRenderTargetIndex)] & rgbWriteMask) != rgbWriteMask) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped drawcall, colour write disabled."));
      return { RtxGeometryStatus::Ignored, false };
    }

    // Ensure present parameters for the swapchain have been cached
    // Note: This assumes that ResetSwapChain has been called at some point before this call, typically done after creating a swapchain.
    assert(m_activePresentParams.has_value());

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

    // Raytraced Render Target
    // If this isn't the primary render target but we have used this render target before then 
    // store the current camera matrices in case this render target is intended to be used as 
    // a texture for some geometry later
    if (RtxOptions::RaytracedRenderTarget::enable()) {
      D3D9CommonTexture* texture = GetCommonTexture(d3d9State().renderTargets[kRenderTargetIndex]->GetBaseTexture());
      if (texture) {
        const Rc<DxvkImage> image = texture->GetImage();
        if (image != nullptr) {
          const XXH64_hash_t descHash = image->getDescriptorHash();
          if (lookupHash(RtxOptions::raytracedRenderTargetTextures(), descHash) ||
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
        if (D3D9CommonTexture* rtTex = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture()) {
          if (rtTex->GetImage() != nullptr) {
            const XXH64_hash_t rtDescHash = rtTex->GetImage()->getDescriptorHash();
            if (s_loggedNonPrimaryRtDescHashes.insert(rtDescHash).second) {
              const auto* rtDesc = rtTex->Desc();
              Logger::info(str::format(
                "[RTX-Compatibility] Non-primary RT0 encountered: ",
                rtDesc->Width, "x", rtDesc->Height,
                " (backbuffer ", m_activePresentParams->BackBufferWidth, "x", m_activePresentParams->BackBufferHeight, "), ",
                "rtDescHash=0x", std::hex, rtDescHash, std::dec,
                ". If this RT contains the main scene, add it to rtx.raytracedRenderTargetTextures."));
            }
          }
        }

        ONCE(Logger::info("[RTX-Compatibility-Info] Found a draw call to a non-primary, non-raytraced render target. Falling back to rasterization"));
        return { RtxGeometryStatus::Rasterized, false };
      }
    }

    // debugging, todo remove later
    if (const uint32_t rtSamplerMask = m_parent->GetActiveRTTextures()) {
      const bool depthEnabled  = d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_TRUE;
      const bool zWriteEnabled = d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE;
      const bool likelyFullscreenComposite =
        !depthEnabled &&
        !zWriteEnabled &&
        drawContext.PrimitiveCount <= 4;

      if (autoRaytracedRenderTargetFromFullscreenComposite() && likelyFullscreenComposite) {
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
            !lookupHash(RtxOptions::raytracedRenderTargetTextures(), bestHash) &&
            m_autoRaytracedRenderTargetDescHashes.insert(bestHash).second) {
          Logger::info(str::format(
            "[RTX-Compatibility] Auto-selected Raytraced Render Target from fullscreen composite: texDescHash=0x",
            std::hex, bestHash, std::dec, "."));
        }
      }

      for (uint32_t i : bit::BitMask(rtSamplerMask)) {
        D3D9CommonTexture* tex = GetCommonTexture(d3d9State().textures[i]);
        if (!tex || tex->GetImage() == nullptr)
          continue;

        const XXH64_hash_t texDescHash = tex->GetImage()->getDescriptorHash();
        if (s_loggedSampledRtDescHashes.insert(texDescHash).second) {
          const auto* desc = tex->Desc();
          Logger::info(str::format(
            "[RTX-Compatibility] Sampled render-target texture: ",
            desc->Width, "x", desc->Height,
            ", texDescHash=0x", std::hex, texDescHash, std::dec,
            " (sampler ", i, ")."));
        }
      }

      // Optional: do not raytrace likely fullscreen composite passes to primary.
      if (rasterizeFullscreenCompositeToPrimary() && likelyFullscreenComposite) {
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
      return {
        RtxGeometryStatus::Rasterized,
        true, // UI rendering detected => trigger RTX injection
      };
    }

    // TODO(REMIX-760): Support reverse engineering pre-transformed vertices
    if (d3d9State().vertexDecl != nullptr) {
      if (d3d9State().vertexDecl->TestFlag(D3D9VertexDeclFlag::HasPositionT)) {
        ONCE(Logger::info("[RTX-Compatibility-Info] Skipped drawcall, using pre-transformed vertices which isn't currently supported."));
        return { RtxGeometryStatus::Rasterized, false };
      }
    }

    return { RtxGeometryStatus::RayTraced, false };
  }

  bool D3D9Rtx::checkBoundTextureCategory(const fast_unordered_set& textureCategory) const {
    const uint32_t usedSamplerMask = m_parent->m_psShaderMasks.samplerMask | m_parent->m_vsShaderMasks.samplerMask;
    const uint32_t usedTextureMask = m_parent->m_activeTextures & usedSamplerMask;
    for (uint32_t idx : bit::BitMask(usedTextureMask)) {
      if (!d3d9State().textures[idx]) {
        continue;
      }

      auto texture = GetCommonTexture(d3d9State().textures[idx]);

      const XXH64_hash_t texHash = texture->GetSampleView(false)->image()->getHash();
      if (textureCategory.find(texHash) != textureCategory.end()) {
        return true;
      }
    }

    return false;
  }

  bool D3D9Rtx::isRenderingUI() {
    if (!m_parent->UseProgrammableVS() && orthographicIsUI()) {
      // Here we assume drawcalls with an orthographic projection are UI calls (as this pattern is common, and we can't raytrace these objects).
      const bool isOrthographic = (d3d9State().transforms[GetTransformIndex(D3DTS_PROJECTION)][3][3] == 1.0f);
      const bool zWriteEnabled = d3d9State().renderStates[D3DRS_ZWRITEENABLE];
      if (isOrthographic && !zWriteEnabled) {
        return true;
      }
    }

    // Check if UI texture bound
    return checkBoundTextureCategory(RtxOptions::uiTextures());
  }

  PrepareDrawFlags D3D9Rtx::internalPrepareDraw(const IndexContext& indexContext, const VertexContext vertexContext[caps::MaxStreams], const DrawContext& drawContext) {
    ScopedCpuProfileZone();

    // RTX was injected => treat everything else as rasterized 
    if (m_rtxInjectTriggered) {
      return RtxOptions::skipDrawCallsPostRTXInjection()
             ? PrepareDrawFlag::Ignore
             : PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    const auto [status, triggerRtxInjection] = makeDrawCallType(drawContext);

    // When raytracing is enabled we want to completely remove the ignored drawcalls from further processing as early as possible
    const PrepareDrawFlags prepareFlagsForIgnoredDraws = RtxOptions::enableRaytracing()
                                                         ? PrepareDrawFlag::Ignore
                                                         : PrepareDrawFlag::PreserveDrawCallAndItsState;

    if (status == RtxGeometryStatus::Ignored) {
      return prepareFlagsForIgnoredDraws;
    }

    if (triggerRtxInjection) {
      // Bind all resources required for this drawcall to context first (i.e. render targets)
      m_parent->PrepareDraw(drawContext.PrimitiveType);

      triggerInjectRTX();

      m_rtxInjectTriggered = true;
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    if (status == RtxGeometryStatus::Rasterized) {
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    m_forceGeometryCopy = RtxOptions::useBuffersDirectly() == false;
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
        return prepareFlagsForIgnoredDraws;
      }

      geoData.vertexCount = maxIndex - minIndex + 1;
      vertexIndexOffset += minIndex;
    } else {
      geoData.vertexCount = GetVertexCount(drawContext.PrimitiveType, drawContext.PrimitiveCount);
    }

    if (geoData.vertexCount == 0) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped invalid drawcall, no vertices detected."));
      return prepareFlagsForIgnoredDraws;
    }

    if (RtxOptions::RaytracedRenderTarget::enable()) {
      // If this draw call has an RT texture bound
      if (m_activeDrawCallState.isUsingRaytracedRenderTarget) {
        // We validate this state below
        m_activeDrawCallState.isUsingRaytracedRenderTarget = false;
        // Try and find the has of the positions
        for (uint32_t i : bit::BitMask(m_parent->GetActiveRTTextures())) {
          D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[i]);
          auto hash = texture->GetImage()->getDescriptorHash();
          if (lookupHash(RtxOptions::raytracedRenderTargetTextures(), hash)) {
            // Mark this as a valid Raytraced Render Target draw call
            m_activeDrawCallState.isUsingRaytracedRenderTarget = true;
          }
        }
      }
    }

    m_activeDrawCallState.categories = 0;
    m_activeDrawCallState.materialData = {};

    // Fetch all the legacy state (colour modes, alpha test, etc...)
    setLegacyMaterialState(m_parent, m_parent->m_alphaSwizzleRTs & (1 << kRenderTargetIndex), m_activeDrawCallState.materialData);

    // Fetch fog state 
    setFogState(m_parent, m_activeDrawCallState.fogState);

    // Fetch all the render state and send it to rtx context (textures, transforms, etc.)
    if (!processRenderState()) {
      return prepareFlagsForIgnoredDraws;
    }

    // Max offseted index value within a buffer slice that geoData contains
    const uint32_t maxOffsetedIndex = maxIndex - minIndex;

    // Copy all the vertices into a staging buffer.  Assign fields of the geoData structure.
    processVertices(vertexContext, vertexIndexOffset, geoData);
    geoData.futureGeometryHashes = computeHash(geoData, maxOffsetedIndex);
    geoData.futureBoundingBox = computeAxisAlignedBoundingBox(geoData);
    
    // Process skinning data
    m_activeDrawCallState.futureSkinningData = processSkinning(geoData);

    // Hash material data
    m_activeDrawCallState.materialData.updateCachedHash();

    // For shader based drawcalls we also want to capture the vertex shader output
    bool needVertexCapture = m_parent->UseProgrammableVS() && useVertexCapture();
    if (needVertexCapture) {
      needVertexCapture = prepareVertexCapture(vertexIndexOffset);
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

    if (RtxOptions::fogIgnoreSky() && m_activeDrawCallState.categories.test(InstanceCategories::Sky)) {
      m_activeDrawCallState.fogState.mode = D3DFOG_NONE;
    }

    // Ignore sky draw calls that are being drawn to a Raytraced Render Target
    // Raytraced Render Target scenes just use the same sky as the main scene, no need to duplicate them
    if (m_activeDrawCallState.isDrawingToRaytracedRenderTarget && m_activeDrawCallState.categories.test(InstanceCategories::Sky)) {
      return prepareFlagsForIgnoredDraws;
    }

    assert(status == RtxGeometryStatus::RayTraced);

    const bool preserveOriginalDraw = needVertexCapture;

    return
      PrepareDrawFlag::CommitToRayTracing |
      (m_activeDrawCallState.testCategoryFlags(CATEGORIES_REQUIRE_DRAW_CALL_STATE) ? PrepareDrawFlag::ApplyDrawState : 0) |
      (preserveOriginalDraw ? PrepareDrawFlag::PreserveDrawCallAndItsState : 0);
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

    const bool hasBlendWeight = d3d9State().vertexDecl != nullptr ? d3d9State().vertexDecl->TestFlag(D3D9VertexDeclFlag::HasBlendWeight) : false;
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

    if (m_stagedBonesCount + maxBone >= m_stagedBones.size()) {
      throw DxvkError("Bones temp storage is too small.");
    }

    Matrix4* boneMatrices = m_stagedBones.data() + m_stagedBonesCount;
    memcpy(boneMatrices, d3d9State().transforms.data() + startBoneTransform, sizeof(Matrix4)*(maxBone + 1));
    m_stagedBonesCount += maxBone + 1;

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
            RtxOptions::enableMultiStageTextureFactorBlending() &&
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
        if (texture->GetType() != D3DRTYPE_TEXTURE && (!allowCubemaps() || texture->GetType() != D3DRTYPE_CUBETEXTURE)) {
          continue;
        }

        const XXH64_hash_t texHash = texture->GetSampleView(true)->image()->getHash();

        // Currently we only support regular textures, skip lightmaps.
        if (lookupHash(RtxOptions::lightmapTextures(), texHash)) {
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
            lookupHash(RtxOptions::ignoreBakedLightingTextures(), texHash)) {
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
    const D3D9CommonShader* inferredPs = nullptr;
    XXH64_hash_t inferredPsHash = 0;
    PsSamplerTexcoordEntry* inferredPsEntry = nullptr;
    const bool likelyGpuSkinnedMesh = [&]() {
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
      ue3VsHints != nullptr &&
      (ue3VsHints->hasDecalTransform ||
       ue3VsHints->hasDecalLocation ||
       ue3VsHints->hasDecalOffset);
    const bool likelyUe3TerrainUvSpace =
      ue3VsHints != nullptr &&
      (ue3VsHints->hasLightMapCoordinateScaleBias ||
       ue3VsHints->hasShadowCoordinateScaleBias);
    const bool likelyUe3BillboardUvSpace =
      ue3VsHints != nullptr &&
      (ue3VsHints->hasTextureCoordinateScaleBias ||
       ue3VsHints->hasViewToLocal ||
       ue3VsHints->hasWindMatrices);
    const bool likelyUe3FlexiblePackedUvPath =
      !likelyGpuSkinnedMesh &&
      (likelyUe3DecalUvSpace || likelyUe3TerrainUvSpace || likelyUe3BillboardUvSpace);
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

      outHash = hashDxsoBytecode(ps->GetBytecode());
      auto& entry = m_psSamplerTexcoordCache[outHash];
      if (!entry.initialized) {
        entry.initialized = true;
        entry.samplerToTexcoord.fill(-1);
        entry.samplerCoordCompValid.fill(0);
        entry.samplerCoordCompU.fill(0);
        entry.samplerCoordCompV.fill(1);
        entry.samplerSemanticFlags.fill(0);
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
      if (shaderPathTexcoordIndexFromPixelShader() && d3d9State().pixelShader.ptr() != nullptr) {
        inferredPs = d3d9State().pixelShader->GetCommonShader();
        inferredPsEntry = getOrInitPsSamplerTexcoordEntry(inferredPs, inferredPsHash);
      }
    }

    if constexpr (FixedFunction) {
      for (uint32_t idx = 0, textureID = 0; idx < NumTexcoordBins && textureID < LegacyMaterialData::kMaxSupportedTextures; idx++) {
        const uint8_t stage = texcoordIndexToStage[idx];
        if (stage == kInvalidStage || d3d9State().textures[stage] == nullptr)
          continue;

        D3D9CommonTexture* pTexInfo = GetCommonTexture(d3d9State().textures[stage]);
        assert(pTexInfo != nullptr);

        // Send the texture stage state for first texture slot (or 0th stage if no texture)
        if (textureID == 0) {
          // ColorTexture2 is optional and currently only used as RayPortal material, the material type will be checked in the submitDrawState.
          // So we don't use it to check valid drawcall or not here.
          if (pTexInfo->GetImage()->getHash() == kEmptyHash) {
            ONCE(Logger::info("[RTX-Compatibility-Info] Texture 0 without valid hash detected, skipping drawcall."));
            return false;
          }

          firstStage = stage;
        }

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
        m_activeDrawCallState.materialData.colorTextures[textureID] = TextureRef(pTexInfo->GetSampleView(srgb));
        m_activeDrawCallState.materialData.samplers[textureID] = sampler;
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

      const uint32_t usedSamplerMask = m_parent->m_psShaderMasks.samplerMask;
      const uint32_t usedTextureMask = m_parent->m_activeTextures & usedSamplerMask;

      for (uint32_t stage : bit::BitMask(usedTextureMask)) {
        if (stage >= SamplerCount || d3d9State().textures[stage] == nullptr)
          continue;

        D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[stage]);
        if (!texture)
          continue;

        if (texture->GetType() != D3DRTYPE_TEXTURE && (!allowCubemaps() || texture->GetType() != D3DRTYPE_CUBETEXTURE))
          continue;

        const XXH64_hash_t texHash = texture->GetSampleView(false)->image()->getHash();
        if (lookupHash(RtxOptions::lightmapTextures(), texHash))
          continue;

        const bool srgb = (d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1) != 0;
        const bool isRenderTarget = texture->IsRenderTarget();

        const auto* desc = texture->Desc();
        const uint64_t area = desc ? uint64_t(desc->Width) * uint64_t(desc->Height) : 0;
        uint16_t sampleCount = 0;
        bool hasInferredTexcoord = false;
        bool hasInferredUvHint = false;
        int8_t inferredTexcoordIdx = -1;
        uint8_t inferredSamplerSemanticFlags = 0;
        bool inferredSamplerLooksEngineAuxiliary = false;
        bool inferredSamplerLooksMaterialTexture = false;
        bool inferredSamplerLooksLightmap = false;
        bool inferredSamplerLooksNonDiffuse = false;
        bool inferredUsesZw = false;
        bool inferredUsesWz = false;
        bool inferredUsesXy = false;
        bool inferredUsesPackedSecondary = false;
        bool hasNonZeroInferredOffset = false;
        if (inferredPsEntry != nullptr && stage < caps::MaxTexturesPS) {
          sampleCount = inferredPsEntry->samplerSampleCount[stage];
          inferredTexcoordIdx = inferredPsEntry->samplerToTexcoord[stage];
          inferredSamplerSemanticFlags = inferredPsEntry->samplerSemanticFlags[stage];
          inferredSamplerLooksEngineAuxiliary = (inferredSamplerSemanticFlags & kPsSamplerSemanticEngineAuxiliary) != 0;
          inferredSamplerLooksMaterialTexture = (inferredSamplerSemanticFlags & kPsSamplerSemanticMaterialTexture) != 0;
          inferredSamplerLooksLightmap = (inferredSamplerSemanticFlags & kPsSamplerSemanticLightmap) != 0;
          inferredSamplerLooksNonDiffuse = (inferredSamplerSemanticFlags & kPsSamplerSemanticNonDiffuse) != 0;
          hasInferredTexcoord = inferredTexcoordIdx >= 0;
          hasInferredUvHint =
            inferredPsEntry->samplerScaleImmediateValid[stage] != 0 ||
            inferredPsEntry->samplerOffsetImmediateValid[stage] != 0 ||
            inferredPsEntry->samplerScaleConstReg[stage] >= 0 ||
            inferredPsEntry->samplerOffsetConstReg[stage] >= 0;
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

        // heuristics
        // sRGB textures are likely albedo
        // prefer frequently sampled samplers from the active pixel shader
        // favour samplers that produce an inferable texcoord (helps pick the true UV-driven stage)
        // render targets are unlikely to be albedo for world geometry
        // prefer larger textures (usually albedo/detail over tiny masks)
        // prefer lower stage index for stability as a tie breaker
        int64_t score = 0;
        score += srgb ? 1'000'000 : 0;
        score += int64_t(std::min<uint16_t>(sampleCount, 16u)) * 120'000ll;
        score += hasInferredTexcoord ? 250'000 : -150'000;
        score += hasInferredUvHint ? 80'000 : 0;
        score -= isRenderTarget ? 500'000 : 0;
        score += inferredSamplerLooksMaterialTexture ? 230'000 : 0;
        score -= inferredSamplerLooksEngineAuxiliary ? 420'000 : 0;
        score -= inferredSamplerLooksLightmap ? 280'000 : 0;
        score -= inferredSamplerLooksNonDiffuse ? 220'000 : 0;
        score += int64_t(std::min<uint64_t>(area, 16ull * 1024ull * 1024ull));
        score -= int64_t(stage);
        if (likelyPackedUvConventions) {
          // UE3-style shader paths (skinned and non-skinned vertex factories) frequently pack
          // secondary UVs into non-`.xy` components, so we bias slot 0 toward diffuse-like UV usage
          const bool packedPairSuspicious =
            hasNonZeroInferredOffset ||
            inferredSamplerLooksEngineAuxiliary ||
            (likelyGpuSkinnedMesh && !inferredSamplerLooksMaterialTexture);

          const int64_t uv0Bonus = likelyGpuSkinnedMesh ? 300'000ll : 60'000ll;
          const int64_t nonUv0Penalty = likelyGpuSkinnedMesh ? 180'000ll : 20'000ll;
          const int64_t xyBonus = likelyGpuSkinnedMesh ? 120'000ll : 35'000ll;
          const int64_t wzPenalty = likelyGpuSkinnedMesh
            ? 320'000ll
            : (packedPairSuspicious ? 120'000ll : 8'000ll);
          const int64_t zwPenalty = likelyGpuSkinnedMesh
            ? 250'000ll
            : (packedPairSuspicious ? 100'000ll : 8'000ll);
          const int64_t packedSecondaryPenalty = likelyGpuSkinnedMesh
            ? 80'000ll
            : (packedPairSuspicious ? 28'000ll : 2'000ll);
          const int64_t offsetPenalty = likelyGpuSkinnedMesh
            ? 220'000ll
            : (inferredSamplerLooksEngineAuxiliary ? 180'000ll
               : (inferredSamplerLooksMaterialTexture ? 12'000ll : 45'000ll));
          const int64_t auxiliaryPenalty = likelyGpuSkinnedMesh ? 120'000ll : 70'000ll;

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
            score -= offsetPenalty;

          if (inferredSamplerLooksEngineAuxiliary)
            score -= auxiliaryPenalty;

          if (!likelyGpuSkinnedMesh &&
              likelyUe3FlexiblePackedUvPath &&
              inferredSamplerLooksMaterialTexture &&
              inferredUsesPackedSecondary &&
              !packedPairSuspicious) {
            // decal/terrain/speedtree billboards commonly use packed UVs for their base colour path (although there's some eceptions in medge due to reasons..)
            score += 45'000ll;
          }
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

      // default to stage 0 if we couldn't find any used textures (should be rare)
      if (chosenStages[0] == kInvalidStage)
        chosenStages[0] = 0;

      // compat fix for UE3-like packed UV conventions:
      // if primary stage selection is suspicious (packed UV components, non-UV0, or atlas offset)
      // prefer a sibling sampler that references the same texture but looks more diffuse-like
      if (likelyPackedUvConventions &&
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
            const bool looksMaterialTexture = (semanticFlags & kPsSamplerSemanticMaterialTexture) != 0;
            const bool looksEngineAuxiliary = (semanticFlags & kPsSamplerSemanticEngineAuxiliary) != 0;
            const bool looksNonDiffuse = (semanticFlags & kPsSamplerSemanticNonDiffuse) != 0;
            if ((semanticFlags & kPsSamplerSemanticMaterialTexture) != 0)
              score += 180;
            if ((semanticFlags & kPsSamplerSemanticEngineAuxiliary) != 0)
              score -= 360;
            if ((semanticFlags & kPsSamplerSemanticLightmap) != 0)
              score -= 260;
            if (looksNonDiffuse)
              score -= 220;

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
              score -= offsetPenalty;

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

      firstStage = chosenStages[0];

      for (uint32_t textureID = 0; textureID < LegacyMaterialData::kMaxSupportedTextures; textureID++) {
        const uint8_t stage = chosenStages[textureID];
        if (stage == kInvalidStage || stage >= SamplerCount || d3d9State().textures[stage] == nullptr)
          continue;

        D3D9CommonTexture* pTexInfo = GetCommonTexture(d3d9State().textures[stage]);
        assert(pTexInfo != nullptr);

        if (textureID == 0) {
          if (pTexInfo->GetImage()->getHash() == kEmptyHash) {
            ONCE(Logger::info("[RTX-Compatibility-Info] Texture 0 without valid hash detected, skipping drawcall."));
            return false;
          }
        }

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
        m_activeDrawCallState.materialData.colorTextures[textureID] = TextureRef(pTexInfo->GetSampleView(srgb));
        m_activeDrawCallState.materialData.samplers[textureID] = sampler;
        if (textureID == 0)
          m_activeDrawCallState.materialData.colorTextureIsSrgb = srgb;

        auto shaderSampler = RemapStateSamplerShader(stage);
        m_activeDrawCallState.materialData.colorTextureSlot[textureID] = computeResourceSlotId(shaderSampler.first, DxsoBindingType::Image, uint32_t(shaderSampler.second));
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
      if (shaderPathTexcoordIndexFromPixelShader()) {
        // shader-path draws perform UV math in shader code
        // fixed-function texture transform/texgen state can be stale and should not be reused
        m_activeDrawCallState.transformData.textureTransform = Matrix4();
        m_activeDrawCallState.transformData.texgenMode = TexGenMode::None;
      }
    }

    if (d3d9State().textures[firstStage]) {
      m_activeDrawCallState.setupCategoriesForTexture();

      // Track the texture hash before checking if it should be ignored
      // This ensures we track all textures sent by the game, not just the ones that are actually rendered.
      const XXH64_hash_t textureHash = m_activeDrawCallState.materialData.getColorTexture().getImageHash();

      // Flag smooth normals category at the d3d9 layer
      m_activeDrawCallState.setCategory(InstanceCategories::SmoothNormals, lookupHash(RtxOptions::smoothNormalsTextures(), textureHash));
      if (textureHash != kEmptyHash) {
        m_parent->EmitCs([textureHash](DxvkContext* ctx) {
          static_cast<RtxContext*>(ctx)->getSceneManager().trackReplacementMaterialHash(textureHash);
        });
      }
      
      // Check if an ignore texture is bound
      if (m_activeDrawCallState.getCategoryFlags().test(InstanceCategories::Ignore)) {
        return false;
      }

      if (m_activeDrawCallState.testCategoryFlags(InstanceCategories::Terrain)) {
        if (RtxOptions::terrainAsDecalsEnabledIfNoBaker() && !TerrainBaker::enableBaking()) {

          m_activeDrawCallState.removeCategory(InstanceCategories::Terrain);
          m_activeDrawCallState.setCategory(InstanceCategories::DecalStatic, true);

          // modulate to compensate the multilayer blending
          DxvkRtTextureOperation& texop = m_activeDrawCallState.materialData.textureColorOperation;
          if (RtxOptions::terrainAsDecalsAllowOverModulate()) {
            if (texop == DxvkRtTextureOperation::Modulate2x || texop == DxvkRtTextureOperation::Modulate4x) {
              texop = DxvkRtTextureOperation::Force_Modulate2x;
            }
          }
        }
      }

      if (!m_forceGeometryCopy && RtxOptions::alwaysCopyDecalGeometries()) {
        // Only poke decal hashes when option is enabled.
        m_forceGeometryCopy |= m_activeDrawCallState.testCategoryFlags(CATEGORIES_REQUIRE_GEOMETRY_COPY);
      }
    }

    // only keep the passthrough texcoord index for selecting the vertex declaration element
    // upper bits are D3DTSS_TCI_* flags used for fixed-function texgen
    uint32_t texcoordIdx = d3d9State().textureStages[stageStateIdx][DXVK_TSS_TEXCOORDINDEX] & 0b111;

    m_forceIaTexcoordForOutlier = [&]() {
      for (uint32_t i = 0; i < LegacyMaterialData::kMaxSupportedTextures; i++) {
        if (isVsTexcoordCaptureOutlierHash(m_activeDrawCallState.materialData.colorTextures[i].getImageHash()))
          return true;
      }

      for (uint32_t stage = 0; stage < SamplerCount; stage++) {
        if (d3d9State().textures[stage] == nullptr)
          continue;

        D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[stage]);
        if (texture == nullptr || texture->GetImage() == nullptr)
          continue;

        if (isVsTexcoordCaptureOutlierHash(texture->GetImage()->getHash()))
          return true;
      }

      return false;
    }();

    if constexpr (!FixedFunction) {
      const bool forceIaTexcoordForOutlier = m_forceIaTexcoordForOutlier;

      // fixed-function TEXCOORDINDEX is often unreliable for shader-driven games
      // prefer inferring TEXCOORD set from pixel shader bytecode when enabled
      if (shaderPathTexcoordIndexFromPixelShader() &&
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
          const int8_t inferred = entry.samplerToTexcoord[firstStage];
          if (inferred >= 0) {
            texcoordIdx = uint32_t(inferred);
            if (entry.samplerCoordCompValid[firstStage] && !forceIaTexcoordForOutlier) {
              const uint8_t inferredCompU = entry.samplerCoordCompU[firstStage] & 0x3u;
              const uint8_t inferredCompV = entry.samplerCoordCompV[firstStage] & 0x3u;
              const bool inferredUsesPackedSecondaryPair =
                (inferredCompU == 2u && inferredCompV == 3u) ||
                (inferredCompU == 3u && inferredCompV == 2u);

              // compat fallback for skinned meshes
              // packed secondary UV pairs (`.wz` / `.zw`) are frequently non-diffuse channels
              if (likelyGpuSkinnedMesh &&
                  inferredUsesPackedSecondaryPair) {
                m_texcoordCompU = 0;
                m_texcoordCompV = 1;
              } else {
                m_texcoordCompU = inferredCompU;
                m_texcoordCompV = inferredCompV;
              }
            }

            // additional guard for skinned meshes using packed atlas UVs
            // if inference points at a non-UV0 set with a large offset, clamp to UV0 `.xy`
            if (likelyGpuSkinnedMesh && texcoordIdx > 0 && !forceIaTexcoordForOutlier) {
              float inferredOffsetU = 0.0f;
              float inferredOffsetV = 0.0f;
              const bool hasResolvedOffset = resolveInferredSamplerOffset(&entry, firstStage, inferredOffsetU, inferredOffsetV);
              constexpr float kLargeAtlasOffset = 0.2f;
              const bool hasLargeOffset = hasResolvedOffset &&
                (std::abs(inferredOffsetU) >= kLargeAtlasOffset || std::abs(inferredOffsetV) >= kLargeAtlasOffset);
              if (hasLargeOffset) {
                texcoordIdx = 0;
                m_texcoordCompU = 0;
                m_texcoordCompV = 1;
              }
            }

            // debugging, todo remove later
            const XXH64_hash_t logKey = psHash ^ (XXH64_hash_t(firstStage) * 0x9E3779B97F4A7C15ull);
            if (m_loggedPsSamplerTexcoordInference.insert(logKey).second) {
              Logger::info(str::format(
                "[RTX-Compatibility] Shader-path inferred TEXCOORD", uint32_t(inferred),
                " for sampler ", firstStage, " from pixel shader bytecode."));
            }
          }

          bool hasScale = false;
          float uScale = 1.0f;
          float vScale = 1.0f;
          bool hasOffset = false;
          float uOffset = 0.0f;
          float vOffset = 0.0f;

          if (entry.samplerScaleImmediateValid[firstStage]) {
            hasScale = true;
            uScale = entry.samplerScaleImmediateU[firstStage];
            vScale = entry.samplerScaleImmediateV[firstStage];
          } else {
            const int16_t scaleConstReg = entry.samplerScaleConstReg[firstStage];
            if (scaleConstReg >= 0 && uint32_t(scaleConstReg) < caps::MaxFloatConstantsPS) {
              hasScale = true;
              const Vector4& scaleConst = d3d9State().psConsts.fConsts[uint32_t(scaleConstReg)];
              const uint32_t compU = entry.samplerScaleConstCompU[firstStage] & 0x3;
              const uint32_t compV = entry.samplerScaleConstCompV[firstStage] & 0x3;
              uScale = scaleConst[compU] * entry.samplerScaleFactorU[firstStage];
              vScale = scaleConst[compV] * entry.samplerScaleFactorV[firstStage];
            }
          }

          if (entry.samplerOffsetImmediateValid[firstStage]) {
            hasOffset = true;
            uOffset = entry.samplerOffsetImmediateU[firstStage];
            vOffset = entry.samplerOffsetImmediateV[firstStage];
          } else {
            const int16_t offsetConstReg = entry.samplerOffsetConstReg[firstStage];
            if (offsetConstReg >= 0 && uint32_t(offsetConstReg) < caps::MaxFloatConstantsPS) {
              hasOffset = true;
              const Vector4& offsetConst = d3d9State().psConsts.fConsts[uint32_t(offsetConstReg)];
              const uint32_t compU = entry.samplerOffsetConstCompU[firstStage] & 0x3;
              const uint32_t compV = entry.samplerOffsetConstCompV[firstStage] & 0x3;
              uOffset = offsetConst[compU] * entry.samplerOffsetFactorU[firstStage];
              vOffset = offsetConst[compV] * entry.samplerOffsetFactorV[firstStage];
            }
          }

          constexpr float kMinAbsScale = 1e-6f;
          bool applyScale = hasScale &&
            std::isfinite(uScale) && std::isfinite(vScale) &&
            std::abs(uScale) > kMinAbsScale && std::abs(vScale) > kMinAbsScale;
          bool applyOffset = hasOffset &&
            std::isfinite(uOffset) && std::isfinite(vOffset);
          if (applyOffset && likelyGpuSkinnedMesh) {
            // keep tile scale but reject large atlas-like offsets on skinned diffuse paths
            constexpr float kLargeAtlasOffset = 0.2f;
            if (std::abs(uOffset) >= kLargeAtlasOffset || std::abs(vOffset) >= kLargeAtlasOffset)
              applyOffset = false;
          }

          if (applyScale || applyOffset) {
            Matrix4& texXform = m_activeDrawCallState.transformData.textureTransform;

            texXform = Matrix4();

            if (applyScale) {
              // UE3 TextureCoordinate compiles to a UV multiply by (UTiling, VTiling).
              texXform[0].x = uScale;
              texXform[1].y = vScale;
            }

            if (applyOffset) {
              // Handle common UE3 UV panning/offset patterns (e.g. texcoord * scale + offset).
              texXform[3].x = uOffset;
              texXform[3].y = vOffset;
            }
          }
        }
      }
    }

    m_texcoordIndex = texcoordIdx;

    return true;
  }

  PrepareDrawFlags D3D9Rtx::PrepareDrawGeometryForRT(const bool indexed, const DrawContext& context) {
    if (!RtxOptions::enableRaytracing() || !m_enableDrawCallConversion) {
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
    if (!RtxOptions::enableRaytracing() || !m_enableDrawCallConversion) {
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

  void D3D9Rtx::EndFrame(const Rc<DxvkImage>& targetImage, bool callInjectRtx) {
    const auto currentReflexFrameId = GetReflexFrameId();
    
    // Flush any pending game and RTX work
    m_parent->Flush();

    // Inform backend of end-frame
    m_parent->EmitCs([currentReflexFrameId, targetImage, callInjectRtx](DxvkContext* ctx) { 
      static_cast<RtxContext*>(ctx)->endFrame(currentReflexFrameId, targetImage, callInjectRtx); 
    });

    // Reset for the next frame
    m_rtxInjectTriggered = false;
    m_drawCallID = 0;
    m_seenCameraPositionsPrev = std::move(m_seenCameraPositions);

    m_stagedBonesCount = 0;
  }

  void D3D9Rtx::OnPresent(const Rc<DxvkImage>& targetImage) {
    // Inform backend of present
    m_parent->EmitCs([targetImage](DxvkContext* ctx) { static_cast<RtxContext*>(ctx)->onPresent(targetImage); });
  }
}
