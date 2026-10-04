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
#include "dxso_sampler_inference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <string>

#include "dxso_code.h"
#include "dxso_ctab.h"
#include "../util/util_vector.h"

namespace dxvk {

  uint8_t classifyPixelSamplerSemanticFlags(const std::string& samplerName) {
    const std::string lowerName = toLowerAscii(samplerName);
    auto contains = [&](const char* token) {
      return containsToken(lowerName, token);
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

    // LightMapTextures[], plus Mirror's Edge's BSplineTexture, the weight LUT that filters
    // them under TdBicubicFiltering.
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

  namespace {

    uint16_t classifyPixelSamplerExpressionFlagsFromName(const std::string& samplerName) {
      const std::string lowerName = toLowerAscii(samplerName);
      auto contains = [&](const char* token) {
        return containsToken(lowerName, token);
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

      if (contains("lerp") || contains("blend") || contains("interpolate")) {
        flags |= kPsSamplerExprBlendMath;
      }

      if (contains("albedo") || contains("diffuse") || contains("basecolor") ||
          contains("base_color") || contains("billboard") || contains("advert") ||
          contains("poster") || contains("decal") || contains("fontsample")) {
        flags |= kPsSamplerExprColorContribution;
      }

      return flags;
    }

    bool isTimeDrivenConstantName(const std::string& lowerName) {
      return containsToken(lowerName, "time") ||
             containsToken(lowerName, "gametime") ||
             containsToken(lowerName, "realtime") ||
             containsToken(lowerName, "sine") ||
             containsToken(lowerName, "cosine") ||
             containsToken(lowerName, "panner") ||
             containsToken(lowerName, "rotator") ||
             containsToken(lowerName, "rotation") ||
             containsToken(lowerName, "flipbook") ||
             containsToken(lowerName, "subuv") ||
             containsToken(lowerName, "phase") ||
             containsToken(lowerName, "oscillat") ||
             containsToken(lowerName, "wind");
    }

    bool constantRangeContainsRegister(const DxsoCtab::Constant& c, const int32_t reg) {
      if (reg < 0 || c.registerCount == 0 || c.registerSet > 2u) {
        return false;
      }
      const int64_t begin = int64_t(c.registerIndex);
      const int64_t end = begin + int64_t(c.registerCount);
      const int64_t r = int64_t(reg);
      return r >= begin && r < end;
    }

    constexpr uint32_t kTrackedTemps = 64;
    constexpr uint32_t kTrackedTexcoords = 8;

    // Recognises UE3's normal-map unpack `t * (UnpackMax - UnpackMin) + UnpackMin`, which is
    // `t * 2 - 1`, and its split forms.
    constexpr uint8_t kValueStateSignExpanded = 1u << 0;  // t * 2 - 1 applied
    constexpr uint8_t kValueStateScaledX2     = 1u << 1;  // t * 2 applied
    constexpr uint8_t kValueStateBiasedHalf   = 1u << 2;  // t - 0.5 applied

    // Diffuse-anchor provenance, see kPsSamplerExprDiffuseAnchor.
    constexpr uint8_t kAnchorLightmapValue = 1u << 0;
    constexpr uint8_t kAnchorLightingConst = 1u << 1;

    constexpr uint8_t kCoordProvTexcoord    = 1u << 0;
    constexpr uint8_t kCoordProvNonTexcoord = 1u << 1;

    constexpr uint8_t kSamplerValueRoleColor   = 1u << 0;
    constexpr uint8_t kSamplerValueRoleControl = 1u << 1;

    // Bitset over the float constant registers.
    constexpr uint32_t kConstDepWords = (kDxsoMaxPsFloatConstants + 63u) / 64u;
    using ConstDepSet = std::array<uint64_t, kConstDepWords>;

    bool isTrackedTemp(const DxsoRegisterId& id) {
      return (id.type == DxsoRegisterType::Temp || id.type == DxsoRegisterType::TempFloat16) &&
             id.num < kTrackedTemps;
    }

    bool isScalarSwizzle(const DxsoRegister& r) {
      const uint8_t c0 = r.swizzle[0] & 0x3u;
      return c0 == (r.swizzle[1] & 0x3u) &&
             c0 == (r.swizzle[2] & 0x3u) &&
             c0 == (r.swizzle[3] & 0x3u);
    }

    bool isAlphaScalarSwizzle(const DxsoRegister& r) {
      return isScalarSwizzle(r) && (r.swizzle[0] & 0x3u) == 3u;
    }

    bool maskWritesRgb(const DxsoRegMask& mask) {
      return mask.popCount() == 0 || mask[0] || mask[1] || mask[2];
    }

    bool hasScaleTerm(const PsTexcoordScaleHint& h) {
      return h.constReg >= 0 || h.immediateValid;
    }

    bool hasOffsetTerm(const PsTexcoordScaleHint& h) {
      return h.offsetConstReg >= 0 || h.offsetImmediateValid;
    }

    bool isSameScaleHint(const PsTexcoordScaleHint& a, const PsTexcoordScaleHint& b) {
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
          a.offsetImmediateValid != b.offsetImmediateValid) {
        return false;
      }
      if (a.immediateValid && (a.immediateU != b.immediateU || a.immediateV != b.immediateV)) {
        return false;
      }
      if (a.offsetImmediateValid && (a.offsetImmediateU != b.offsetImmediateU || a.offsetImmediateV != b.offsetImmediateV)) {
        return false;
      }
      return true;
    }

    // The product of two offset-free hints, when at least one side is an immediate.
    bool combineMulHints(const PsTexcoordScaleHint& a, const PsTexcoordScaleHint& b, PsTexcoordScaleHint& out) {
      out = PsTexcoordScaleHint{};

      const bool aConst = a.constReg >= 0 && !a.immediateValid && !hasOffsetTerm(a);
      const bool bConst = b.constReg >= 0 && !b.immediateValid && !hasOffsetTerm(b);
      const bool aImm = a.immediateValid && !hasOffsetTerm(a);
      const bool bImm = b.immediateValid && !hasOffsetTerm(b);

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
    }

    // The texcoord every non-negative value agrees on, or -1.
    int32_t mergeSingle(std::initializer_list<int32_t> vals) {
      int32_t v = -1;
      for (const int32_t x : vals) {
        if (x < 0) {
          continue;
        }
        if (v < 0) {
          v = x;
        } else if (v != x) {
          return -1;
        }
      }
      return v;
    }

    void setOffsetFromScaleTerm(PsTexcoordScaleHint& inOut, const PsTexcoordScaleHint& from, const float sign) {
      inOut.offsetConstReg = from.constReg;
      inOut.offsetCompU = from.compU;
      inOut.offsetCompV = from.compV;
      inOut.offsetFactorU = sign * from.scaleFactorU;
      inOut.offsetFactorV = sign * from.scaleFactorV;
      inOut.offsetImmediateValid = from.immediateValid;
      if (from.immediateValid) {
        inOut.offsetImmediateU = sign * from.immediateU;
        inOut.offsetImmediateV = sign * from.immediateV;
      }
    }

    void setOffsetFromOffsetTerm(PsTexcoordScaleHint& inOut, const PsTexcoordScaleHint& from, const float sign) {
      inOut.offsetConstReg = from.offsetConstReg;
      inOut.offsetCompU = from.offsetCompU;
      inOut.offsetCompV = from.offsetCompV;
      inOut.offsetFactorU = sign * from.offsetFactorU;
      inOut.offsetFactorV = sign * from.offsetFactorV;
      inOut.offsetImmediateValid = from.offsetImmediateValid;
      if (from.offsetImmediateValid) {
        inOut.offsetImmediateU = sign * from.offsetImmediateU;
        inOut.offsetImmediateV = sign * from.offsetImmediateV;
      }
    }

    // One pass over a pixel shader tracking, per temp register, the texcoord it derives from, its
    // UV scale and offset, and how the analysed sampler's value is used.
    class SamplerUseAnalysis {

    public:

      SamplerUseAnalysis(const DxsoShaderView& pixelShader, uint32_t samplerIdx);

      PsSamplerTexcoordInference run();

    private:

      // What a temp write derives from its sources, stored only once every phase has read them.
      struct TempWrite {
        int32_t             texcoord = -1;
        bool                tracked = false;
        PsTexcoordScaleHint scaleHint;
        uint16_t            expressionFlags = 0;
        uint8_t             samplerValueRole = 0;
        uint8_t             valueState = 0;
        uint8_t             anchorBits = 0;
        bool                hasConstOnlyHint = false;
        PsTexcoordScaleHint constOnlyHint;
      };

      // coordReg may point at coordRegStorage, so a SampleOp must not be copied.
      struct SampleOp {
        uint32_t            sampler = ~0u;
        const DxsoRegister* coordReg = nullptr;
        DxsoRegister        coordRegStorage;
        uint8_t             semanticFlags = 0;
        uint16_t            expressionFlags = 0;
      };

      struct ScaleHintVote {
        bool                valid = false;
        bool                conflict = false;
        PsTexcoordScaleHint hint;
      };

      int32_t getTexcoord(const DxsoRegister& r) const;
      uint8_t getCoordProvenance(const DxsoRegister& r) const;
      uint16_t getCoordExpressionFlags(const DxsoRegister& r) const;
      uint8_t getSamplerValueRole(const DxsoRegister& r) const;
      uint8_t getSamplerValueRoleWithSwizzle(const DxsoRegister& r) const;
      uint8_t getSamplerValueState(const DxsoRegister& r) const;
      uint8_t getAnchorBits(const DxsoRegister& r) const;
      void markConstDep(int32_t reg, ConstDepSet& inOut) const;
      void orConstDeps(const DxsoRegister& r, ConstDepSet& inOut) const;
      bool isDefConstNearRgb(const DxsoRegister& r, float target) const;

      bool getScaleHint(const DxsoRegister& r, PsTexcoordScaleHint& outHint) const;
      bool loadScaleHintFromConstant(const DxsoRegister& r, PsTexcoordScaleHint& outHint) const;
      bool tryLoadConstantLikeHint(const DxsoRegister& r, PsTexcoordScaleHint& outHint) const;
      bool applyOffsetFromConstant(const DxsoRegister& r, PsTexcoordScaleHint& inOutHint, float sign) const;
      bool applyOffset(const DxsoRegister& r, PsTexcoordScaleHint& inOutHint, float sign) const;

      void collectAnchorSources(const DxsoCtab& ctab);
      void recordDefConstant(const DxsoInstructionContext& ctx);
      void detectDiffuseAnchor(const DxsoInstructionContext& ctx);
      void propagateConstDeps(const DxsoInstructionContext& ctx);
      void trackTempWrite(const DxsoInstructionContext& ctx);
      void detectOutputColor(const DxsoInstructionContext& ctx);
      void trackSample(const DxsoInstructionContext& ctx);

      void assignFromSource(const DxsoRegister& src, TempWrite& w) const;
      void deriveUnaryCoordinate(const DxsoInstructionContext& ctx, TempWrite& w);
      void deriveAddSubCoordinate(const DxsoInstructionContext& ctx, TempWrite& w) const;
      void deriveBinaryCoordinate(const DxsoInstructionContext& ctx, TempWrite& w);
      void deriveMulCoordinate(const DxsoInstructionContext& ctx, TempWrite& w) const;
      void deriveMadCoordinate(const DxsoInstructionContext& ctx, TempWrite& w) const;
      void deriveSamplerValueRole(const DxsoInstructionContext& ctx, TempWrite& w) const;
      uint8_t deriveValueState(const DxsoInstructionContext& ctx) const;
      void writeTempTracking(const DxsoInstructionContext& ctx, const TempWrite& w);

      void decodeSampleOp(const DxsoInstructionContext& ctx, SampleOp& out) const;

      void voteTexcoord();
      void applyTexcoordVote();
      void collectCoordConstRegs();
      void deriveExpressionFlags();
      void applyCtabNameFlags();
      void applySemanticFallbacks();

      const DxsoShaderView&      m_shader;
      const DxsoProgramInfo&     m_info;
      const uint32_t             m_samplerIdx;
      DxsoDecodeContext          m_decoder;
      PsSamplerTexcoordInference m_result;

      std::array<int8_t, 2 * DxsoMaxInterfaceRegs> m_inputRegToTexcoord = {};

      std::array<int8_t, kTrackedTemps>              m_tempToTexcoord = {};
      std::array<PsTexcoordScaleHint, kTrackedTemps> m_tempToScaleHint = {};
      std::array<uint8_t, kTrackedTemps>             m_tempCoordProvenance = {};
      std::array<uint16_t, kTrackedTemps>            m_tempCoordExpressionFlags = {};
      std::array<uint8_t, kTrackedTemps>             m_tempSamplerValueRole = {};
      std::array<uint8_t, kTrackedTemps>             m_tempSamplerValueState = {};
      std::array<uint8_t, kTrackedTemps>             m_tempAnchorBits = {};
      std::array<ConstDepSet, kTrackedTemps>         m_tempConstDeps = {};

      bool                                          m_anchorSourcesCollected = false;
      uint32_t                                      m_anchorLightmapSamplerMask = 0;
      std::array<uint8_t, kDxsoMaxPsFloatConstants> m_anchorLightingConstRegs = {};

      // `def` literals are bytecode, so never draw state.
      std::array<uint8_t, kDxsoMaxPsFloatConstants> m_defFloatConstValid = {};
      std::array<Vector4, kDxsoMaxPsFloatConstants> m_defFloatConsts = {};

      std::array<uint32_t, kTrackedTexcoords> m_texcoordUseCount = {};
      // Reads of each (compU << 2) | compV component pair, per texcoord.
      std::array<std::array<uint16_t, 16>, kTrackedTexcoords> m_texcoordCoordPairUseCount = {};
      std::array<ScaleHintVote, kTrackedTexcoords> m_perTexcoordScaleHints = {};
      uint32_t    m_texcoordDerivedSampleCount = 0;
      uint32_t    m_nonTexcoordDerivedSampleCount = 0;
      uint16_t    m_sampledCoordExpressionFlags = 0;
      ConstDepSet m_sampledCoordConstDeps = {};
      bool        m_normalDecodeDetected = false;
      bool        m_reachesOutputColor = false;
      bool        m_diffuseAnchorDetected = false;

    };

    SamplerUseAnalysis::SamplerUseAnalysis(const DxsoShaderView& pixelShader, const uint32_t samplerIdx)
      : m_shader(pixelShader)
      , m_info(*pixelShader.info)
      , m_samplerIdx(samplerIdx)
      , m_decoder(*pixelShader.info) {
      m_inputRegToTexcoord.fill(-1);
      m_tempToTexcoord.fill(-1);

      const DxsoIsgn& isgn = *pixelShader.isgn;
      for (uint32_t i = 0; i < isgn.elemCount; i++) {
        const auto& e = isgn.elems[i];
        if (e.semantic.usage == DxsoUsage::Texcoord && e.regNumber < m_inputRegToTexcoord.size()) {
          m_inputRegToTexcoord[e.regNumber] = int8_t(e.semantic.usageIndex & 0b111);
        }
      }
    }

    PsSamplerTexcoordInference SamplerUseAnalysis::run() {
      DxsoCodeIter iter(m_shader.tokens + 1);
      while (m_decoder.decodeInstruction(iter)) {
        const DxsoInstructionContext& ctx = m_decoder.getInstructionContext();

        // The CTAB comment precedes every instruction.
        if (!m_anchorSourcesCollected && m_decoder.getCtabInfo().m_size != 0) {
          m_anchorSourcesCollected = true;
          collectAnchorSources(m_decoder.getCtabInfo());
        }

        recordDefConstant(ctx);
        // Before the destination is written, so a product into one of its operands sees the old value.
        detectDiffuseAnchor(ctx);
        propagateConstDeps(ctx);
        trackTempWrite(ctx);
        detectOutputColor(ctx);
        trackSample(ctx);
      }

      voteTexcoord();
      applyTexcoordVote();
      collectCoordConstRegs();
      deriveExpressionFlags();
      applyCtabNameFlags();
      applySemanticFallbacks();
      return m_result;
    }

    int32_t SamplerUseAnalysis::getTexcoord(const DxsoRegister& r) const {
      switch (r.id.type) {
      case DxsoRegisterType::Texture:
      case DxsoRegisterType::PixelTexcoord:
        if (r.id.num < m_inputRegToTexcoord.size() && m_inputRegToTexcoord[r.id.num] >= 0) {
          return m_inputRegToTexcoord[r.id.num];
        }
        // A register the signature does not declare as a texcoord reads the set of its own index.
        return int32_t(r.id.num & 0b111);
      case DxsoRegisterType::Input:
        return r.id.num < m_inputRegToTexcoord.size() ? m_inputRegToTexcoord[r.id.num] : -1;
      case DxsoRegisterType::Temp:
      case DxsoRegisterType::TempFloat16:
        return r.id.num < kTrackedTemps ? m_tempToTexcoord[r.id.num] : -1;
      default:
        return -1;
      }
    }

    uint8_t SamplerUseAnalysis::getCoordProvenance(const DxsoRegister& r) const {
      switch (r.id.type) {
      case DxsoRegisterType::Texture:
      case DxsoRegisterType::PixelTexcoord:
        return kCoordProvTexcoord;
      case DxsoRegisterType::Input:
        if (r.id.num < m_inputRegToTexcoord.size()) {
          return m_inputRegToTexcoord[r.id.num] >= 0 ? kCoordProvTexcoord : kCoordProvNonTexcoord;
        }
        return kCoordProvNonTexcoord;
      case DxsoRegisterType::Temp:
      case DxsoRegisterType::TempFloat16:
        return r.id.num < kTrackedTemps ? m_tempCoordProvenance[r.id.num] : 0u;
      default:
        return 0u;
      }
    }

    uint16_t SamplerUseAnalysis::getCoordExpressionFlags(const DxsoRegister& r) const {
      if (r.id.type == DxsoRegisterType::Temp || r.id.type == DxsoRegisterType::TempFloat16) {
        return r.id.num < kTrackedTemps ? m_tempCoordExpressionFlags[r.id.num] : uint16_t(0u);
      }
      return 0u;
    }

    uint8_t SamplerUseAnalysis::getSamplerValueRole(const DxsoRegister& r) const {
      if (r.id.type == DxsoRegisterType::Temp || r.id.type == DxsoRegisterType::TempFloat16) {
        return r.id.num < kTrackedTemps ? m_tempSamplerValueRole[r.id.num] : 0u;
      }
      return 0u;
    }

    // Reading only the alpha of a sampled value treats it as a mask.
    uint8_t SamplerUseAnalysis::getSamplerValueRoleWithSwizzle(const DxsoRegister& r) const {
      const uint8_t baseRole = getSamplerValueRole(r);
      if (!baseRole) {
        return 0u;
      }
      if (isAlphaScalarSwizzle(r)) {
        return uint8_t(baseRole | kSamplerValueRoleControl);
      }
      return baseRole;
    }

    uint8_t SamplerUseAnalysis::getSamplerValueState(const DxsoRegister& r) const {
      if (r.id.type == DxsoRegisterType::Temp || r.id.type == DxsoRegisterType::TempFloat16) {
        return r.id.num < kTrackedTemps ? m_tempSamplerValueState[r.id.num] : 0u;
      }
      return 0u;
    }

    uint8_t SamplerUseAnalysis::getAnchorBits(const DxsoRegister& r) const {
      if (r.id.type == DxsoRegisterType::Temp || r.id.type == DxsoRegisterType::TempFloat16) {
        return r.id.num < kTrackedTemps ? m_tempAnchorBits[r.id.num] : 0u;
      }
      if (isFloatConstantRegisterType(r.id.type) && !r.hasRelative) {
        const int32_t reg = getFloatConstantRegisterIndex(r);
        if (reg >= 0 && reg < int32_t(m_anchorLightingConstRegs.size()) && m_anchorLightingConstRegs[reg]) {
          return kAnchorLightingConst;
        }
      }
      return 0u;
    }

    // A `def` literal is part of the shader identity seed already and never varies between draws.
    void SamplerUseAnalysis::markConstDep(const int32_t reg, ConstDepSet& inOut) const {
      if (reg >= 0 && reg < int32_t(kDxsoMaxPsFloatConstants) && !m_defFloatConstValid[reg]) {
        inOut[uint32_t(reg) / 64u] |= 1ull << (uint32_t(reg) % 64u);
      }
    }

    void SamplerUseAnalysis::orConstDeps(const DxsoRegister& r, ConstDepSet& inOut) const {
      if (r.id.type == DxsoRegisterType::Temp || r.id.type == DxsoRegisterType::TempFloat16) {
        if (r.id.num < kTrackedTemps) {
          const ConstDepSet& deps = m_tempConstDeps[r.id.num];
          for (uint32_t w = 0; w < kConstDepWords; w++) {
            inOut[w] |= deps[w];
          }
        }
        return;
      }
      if (isFloatConstantRegisterType(r.id.type) && !r.hasRelative) {
        markConstDep(getFloatConstantRegisterIndex(r), inOut);
      }
    }

    // Whether r is a `def` literal whose swizzled rgb all equal target, as the folded normal
    // unpack constants (2, -1) are.
    bool SamplerUseAnalysis::isDefConstNearRgb(const DxsoRegister& r, const float target) const {
      if (!isFloatConstantRegisterType(r.id.type) || r.hasRelative) {
        return false;
      }

      float modifierScale = 1.0f;
      if (!decodeConstantModifierScale(r.modifier, modifierScale)) {
        return false;
      }

      const int32_t reg = getFloatConstantRegisterIndex(r);
      if (reg < 0 || reg >= int32_t(m_defFloatConstValid.size()) || !m_defFloatConstValid[reg]) {
        return false;
      }

      constexpr float kTolerance = 0.01f;
      for (uint32_t comp = 0; comp < 3; comp++) {
        const float value = modifierScale * m_defFloatConsts[reg][r.swizzle[comp] & 0x3u];
        if (std::abs(value - target) > kTolerance) {
          return false;
        }
      }
      return true;
    }

    bool SamplerUseAnalysis::getScaleHint(const DxsoRegister& r, PsTexcoordScaleHint& outHint) const {
      if ((r.id.type == DxsoRegisterType::Temp || r.id.type == DxsoRegisterType::TempFloat16) &&
          r.id.num < kTrackedTemps) {
        const PsTexcoordScaleHint& hint = m_tempToScaleHint[r.id.num];
        if (hasScaleTerm(hint) || hasOffsetTerm(hint)) {
          outHint = hint;
          return true;
        }
      }
      return false;
    }

    // Leaves outHint's offset term, and its immediate when r is not a `def` literal, untouched.
    bool SamplerUseAnalysis::loadScaleHintFromConstant(const DxsoRegister& r, PsTexcoordScaleHint& outHint) const {
      if (!isFloatConstantRegisterType(r.id.type)) {
        return false;
      }

      float modifierScale = 1.0f;
      if (!decodeConstantModifierScale(r.modifier, modifierScale)) {
        return false;
      }

      const int32_t reg = getFloatConstantRegisterIndex(r);
      if (reg < 0) {
        return false;
      }

      outHint.constReg = reg;
      outHint.compU = r.swizzle[0] & 0x3;
      outHint.compV = r.swizzle[1] & 0x3;
      outHint.scaleFactorU = modifierScale;
      outHint.scaleFactorV = modifierScale;

      if (reg < int32_t(m_defFloatConstValid.size()) && m_defFloatConstValid[reg]) {
        const Vector4& c = m_defFloatConsts[reg];
        outHint.immediateValid = true;
        outHint.immediateU = modifierScale * c[outHint.compU];
        outHint.immediateV = modifierScale * c[outHint.compV];
      }

      return true;
    }

    bool SamplerUseAnalysis::tryLoadConstantLikeHint(const DxsoRegister& r, PsTexcoordScaleHint& outHint) const {
      if (loadScaleHintFromConstant(r, outHint)) {
        return true;
      }
      return getScaleHint(r, outHint) && hasScaleTerm(outHint);
    }

    bool SamplerUseAnalysis::applyOffsetFromConstant(const DxsoRegister& r, PsTexcoordScaleHint& inOutHint, const float sign) const {
      PsTexcoordScaleHint constant;
      if (!loadScaleHintFromConstant(r, constant)) {
        return false;
      }

      setOffsetFromScaleTerm(inOutHint, constant, sign);
      if (!constant.immediateValid) {
        inOutHint.offsetImmediateU = 0.0f;
        inOutHint.offsetImmediateV = 0.0f;
      }
      return true;
    }

    // Offsets the hint by sign * r, where r is a constant or a temp with a known scale or offset.
    // False when the offset is unknown.
    bool SamplerUseAnalysis::applyOffset(const DxsoRegister& r, PsTexcoordScaleHint& inOutHint, const float sign) const {
      if (applyOffsetFromConstant(r, inOutHint, sign)) {
        return true;
      }

      PsTexcoordScaleHint tempHint;
      if (!getScaleHint(r, tempHint)) {
        return false;
      }
      if (hasScaleTerm(tempHint)) {
        setOffsetFromScaleTerm(inOutHint, tempHint, sign);
        return true;
      }
      if (hasOffsetTerm(tempHint)) {
        setOffsetFromOffsetTerm(inOutHint, tempHint, sign);
        return true;
      }
      return false;
    }

    void SamplerUseAnalysis::collectAnchorSources(const DxsoCtab& ctab) {
      for (const DxsoCtab::Constant& c : ctab.m_constantData) {
        if (c.registerCount == 0) {
          continue;
        }
        const std::string lowerName = toLowerAscii(c.name);
        if (c.registerSet == kD3dxRegisterSetSampler) {
          if (containsToken(lowerName, "lightmap")) {
            const uint32_t end = std::min<uint32_t>(c.registerIndex + c.registerCount, 32u);
            for (uint32_t reg = c.registerIndex; reg < end; reg++) {
              m_anchorLightmapSamplerMask |= 1u << reg;
            }
          }
        } else if (c.registerSet == kD3dxRegisterSetFloat4) {
          // BasePassPixelShader.usf scales unlit and dynamic diffuse by AmbientColorAndSkyFactor.rgb,
          // and sky-lit diffuse by UpperSkyColor and LowerSkyColor.
          if (containsToken(lowerName, "ambientcolorandskyfactor") ||
              containsToken(lowerName, "upperskycolor") ||
              containsToken(lowerName, "lowerskycolor")) {
            const uint32_t end = std::min<uint32_t>(c.registerIndex + c.registerCount,
                                                    uint32_t(m_anchorLightingConstRegs.size()));
            for (uint32_t reg = c.registerIndex; reg < end; reg++) {
              m_anchorLightingConstRegs[reg] = 1;
            }
          }
        }
      }
    }

    void SamplerUseAnalysis::recordDefConstant(const DxsoInstructionContext& ctx) {
      if (ctx.instruction.opcode == DxsoOpcode::Def &&
          ctx.dst.id.type == DxsoRegisterType::Const &&
          ctx.dst.id.num < m_defFloatConsts.size()) {
        m_defFloatConstValid[ctx.dst.id.num] = 1;
        m_defFloatConsts[ctx.dst.id.num] = Vector4(
          ctx.def.float32[0],
          ctx.def.float32[1],
          ctx.def.float32[2],
          ctx.def.float32[3]);
      }
    }

    // This sampler's colour multiplied by a lightmap-derived value or a UE3 lighting constant. Only
    // the product operands count; mad's src2 is additive.
    void SamplerUseAnalysis::detectDiffuseAnchor(const DxsoInstructionContext& ctx) {
      const DxsoOpcode op = ctx.instruction.opcode;
      if (m_diffuseAnchorDetected || (op != DxsoOpcode::Mul && op != DxsoOpcode::Mad)) {
        return;
      }

      const bool colorA = (getSamplerValueRole(ctx.src[0]) & kSamplerValueRoleColor) != 0;
      const bool colorB = (getSamplerValueRole(ctx.src[1]) & kSamplerValueRoleColor) != 0;
      const uint8_t anchorA = getAnchorBits(ctx.src[0]);
      const uint8_t anchorB = getAnchorBits(ctx.src[1]);
      if ((colorA && anchorB) || (colorB && anchorA)) {
        m_diffuseAnchorDetected = true;
      }
    }

    // Every temp write propagates, not only the ones coordinate tracking recognises: a rotator
    // reaches its coordinate through intermediate temps that are not coordinates themselves.
    void SamplerUseAnalysis::propagateConstDeps(const DxsoInstructionContext& ctx) {
      const DxsoOpcode op = ctx.instruction.opcode;
      if (!isTrackedTemp(ctx.dst.id) || op == DxsoOpcode::Def || op == DxsoOpcode::DefI || op == DxsoOpcode::DefB) {
        return;
      }

      ConstDepSet deps = {};
      const uint32_t srcCount = std::min<uint32_t>(getDxsoSourceOperandCount(op), uint32_t(ctx.src.size()));
      for (uint32_t s = 0; s < srcCount; s++) {
        orConstDeps(ctx.src[s], deps);
      }

      const uint32_t matrixRows = getDxsoMatrixRowCount(op);
      if (matrixRows > 1u && srcCount >= 2u &&
          isFloatConstantRegisterType(ctx.src[1].id.type) && !ctx.src[1].hasRelative) {
        const int32_t base = getFloatConstantRegisterIndex(ctx.src[1]);
        for (uint32_t row = 1; row < matrixRows; row++) {
          markConstDep(base + int32_t(row), deps);
        }
      }

      m_tempConstDeps[ctx.dst.id.num] = deps;
    }

    void SamplerUseAnalysis::trackTempWrite(const DxsoInstructionContext& ctx) {
      if (!isTrackedTemp(ctx.dst.id)) {
        return;
      }

      TempWrite w;
      w.expressionFlags = uint16_t(
        getCoordExpressionFlags(ctx.src[0]) |
        getCoordExpressionFlags(ctx.src[1]) |
        getCoordExpressionFlags(ctx.src[2]));

      switch (ctx.instruction.opcode) {
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
        deriveUnaryCoordinate(ctx, w);
        break;
      case DxsoOpcode::Add:
      case DxsoOpcode::Sub:
        deriveAddSubCoordinate(ctx, w);
        break;
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
      case DxsoOpcode::M3x2:
        deriveBinaryCoordinate(ctx, w);
        break;
      case DxsoOpcode::Mul:
        deriveMulCoordinate(ctx, w);
        break;
      case DxsoOpcode::Mad:
        deriveMadCoordinate(ctx, w);
        break;
      case DxsoOpcode::Lrp:
      case DxsoOpcode::Cmp:
      case DxsoOpcode::Dp2Add:
        w.tracked = true;
        w.texcoord = mergeSingle({
          getTexcoord(ctx.src[0]),
          getTexcoord(ctx.src[1]),
          getTexcoord(ctx.src[2]) });
        if (w.texcoord >= 0) {
          w.expressionFlags |= kPsSamplerExprBlendMath;
        }
        break;
      default:
        break;
      }

      deriveSamplerValueRole(ctx, w);
      w.valueState = deriveValueState(ctx);
      w.anchorBits = uint8_t(
        getAnchorBits(ctx.src[0]) |
        getAnchorBits(ctx.src[1]) |
        getAnchorBits(ctx.src[2]));
      writeTempTracking(ctx, w);
    }

    void SamplerUseAnalysis::assignFromSource(const DxsoRegister& src, TempWrite& w) const {
      w.texcoord = getTexcoord(src);
      w.expressionFlags |= getCoordExpressionFlags(src);
      w.samplerValueRole |= getSamplerValueRoleWithSwizzle(src);
      if (w.texcoord < 0) {
        return;
      }
      if (!getScaleHint(src, w.scaleHint)) {
        w.scaleHint = PsTexcoordScaleHint{};
      }
    }

    void SamplerUseAnalysis::deriveUnaryCoordinate(const DxsoInstructionContext& ctx, TempWrite& w) {
      const DxsoOpcode op = ctx.instruction.opcode;
      w.tracked = true;
      assignFromSource(ctx.src[0], w);

      if (op != DxsoOpcode::Mov) {
        w.expressionFlags |= kPsSamplerExprBlendMath;
      }
      if (op == DxsoOpcode::Rcp || op == DxsoOpcode::Rsq ||
          op == DxsoOpcode::Exp || op == DxsoOpcode::Log ||
          op == DxsoOpcode::Abs || op == DxsoOpcode::Nrm) {
        w.expressionFlags |= kPsSamplerExprUvTransform;
      }
      if (op == DxsoOpcode::Frc || op == DxsoOpcode::SinCos) {
        w.expressionFlags |= kPsSamplerExprUvAnimated;
      }
      if (op == DxsoOpcode::Nrm) {
        const uint8_t src0Prov = getCoordProvenance(ctx.src[0]);
        if ((src0Prov & kCoordProvNonTexcoord) != 0 && (src0Prov & kCoordProvTexcoord) == 0) {
          w.expressionFlags |= kPsSamplerExprViewDependent;
        }
        // A normalised sampled value is direction data, as a normal map's is.
        if (getSamplerValueRole(ctx.src[0])) {
          m_normalDecodeDetected = true;
        }
      }
      if (w.texcoord < 0) {
        w.hasConstOnlyHint = tryLoadConstantLikeHint(ctx.src[0], w.constOnlyHint);
      }
    }

    void SamplerUseAnalysis::deriveAddSubCoordinate(const DxsoInstructionContext& ctx, TempWrite& w) const {
      const bool isSub = ctx.instruction.opcode == DxsoOpcode::Sub;
      w.tracked = true;
      w.expressionFlags |= kPsSamplerExprBlendMath;
      const int32_t tc0 = getTexcoord(ctx.src[0]);
      const int32_t tc1 = getTexcoord(ctx.src[1]);

      if (tc0 >= 0 && tc1 < 0) {
        assignFromSource(ctx.src[0], w);
        if (w.texcoord < 0) {
          return;
        }
        w.expressionFlags |= kPsSamplerExprUvOffset;
        if (!applyOffset(ctx.src[1], w.scaleHint, isSub ? -1.0f : 1.0f)) {
          const bool rhsNonTexcoord = (getCoordProvenance(ctx.src[1]) & kCoordProvNonTexcoord) != 0;
          w.expressionFlags |= rhsNonTexcoord ? kPsSamplerExprUvAnimated : kPsSamplerExprBlendMath;
        }
      } else if (tc1 >= 0 && tc0 < 0) {
        assignFromSource(ctx.src[1], w);
        if (w.texcoord < 0) {
          return;
        }
        w.expressionFlags |= kPsSamplerExprUvOffset;
        bool hasKnownOffset = false;
        if (!isSub) {
          hasKnownOffset = applyOffset(ctx.src[0], w.scaleHint, 1.0f);
        } else if (!hasScaleTerm(w.scaleHint)) {
          // `const - uv` mirrors the coordinate around the constant.
          w.expressionFlags |= kPsSamplerExprUvTransform;
          w.scaleHint.immediateValid = true;
          w.scaleHint.immediateU = -1.0f;
          w.scaleHint.immediateV = -1.0f;
          applyOffsetFromConstant(ctx.src[0], w.scaleHint, 1.0f);
        } else {
          // Negating an existing scale is not representable.
          w.scaleHint = PsTexcoordScaleHint{};
        }
        if (!hasKnownOffset) {
          const bool lhsNonTexcoord = (getCoordProvenance(ctx.src[0]) & kCoordProvNonTexcoord) != 0;
          w.expressionFlags |= lhsNonTexcoord ? kPsSamplerExprUvAnimated : kPsSamplerExprBlendMath;
        }
      } else {
        w.texcoord = mergeSingle({ tc0, tc1 });
      }
    }

    void SamplerUseAnalysis::deriveBinaryCoordinate(const DxsoInstructionContext& ctx, TempWrite& w) {
      const DxsoOpcode op = ctx.instruction.opcode;
      w.tracked = true;
      w.expressionFlags |= kPsSamplerExprBlendMath;
      const int32_t tc0 = getTexcoord(ctx.src[0]);
      const int32_t tc1 = getTexcoord(ctx.src[1]);
      if (tc0 >= 0 && tc1 < 0) {
        assignFromSource(ctx.src[0], w);
      } else if (tc1 >= 0 && tc0 < 0) {
        assignFromSource(ctx.src[1], w);
      } else {
        w.texcoord = mergeSingle({ tc0, tc1 });
      }

      const bool isDot = op == DxsoOpcode::Dp3 || op == DxsoOpcode::Dp4;
      const bool isSelect = op == DxsoOpcode::Min || op == DxsoOpcode::Max ||
                            op == DxsoOpcode::Slt || op == DxsoOpcode::Sge;
      if (w.texcoord >= 0 && !isSelect) {
        w.expressionFlags |= kPsSamplerExprUvTransform;
      }
      if (isDot || op == DxsoOpcode::Pow || op == DxsoOpcode::Crs) {
        const uint8_t srcProv = getCoordProvenance(ctx.src[0]) | getCoordProvenance(ctx.src[1]);
        if ((srcProv & kCoordProvNonTexcoord) != 0 && (srcProv & kCoordProvTexcoord) == 0) {
          w.expressionFlags |= kPsSamplerExprViewDependent;
        }
      }
      if (isDot) {
        // A normal-map decode is a self dot product of a sampled value (UE3's normalize() compiles
        // to `dp3 r.w, n, n; rsq; mul`), or a dot product of a sign-expanded one.
        auto isSignExpandedUse = [&](const DxsoRegister& r) {
          if ((getSamplerValueState(r) & kValueStateSignExpanded) != 0) {
            return true;
          }
          // ps_1_x applies the expansion at the use site, as the _bx2 source modifier.
          return r.modifier == DxsoRegModifier::Sign || r.modifier == DxsoRegModifier::SignNeg;
        };
        const uint8_t role0 = getSamplerValueRole(ctx.src[0]);
        const uint8_t role1 = getSamplerValueRole(ctx.src[1]);
        const bool selfDot =
          ctx.src[0].id.type == ctx.src[1].id.type &&
          ctx.src[0].id.num == ctx.src[1].id.num;
        if ((selfDot && role0) ||
            (role0 && isSignExpandedUse(ctx.src[0])) ||
            (role1 && isSignExpandedUse(ctx.src[1]))) {
          m_normalDecodeDetected = true;
        }
      }
    }

    void SamplerUseAnalysis::deriveMulCoordinate(const DxsoInstructionContext& ctx, TempWrite& w) const {
      w.tracked = true;
      w.expressionFlags |= kPsSamplerExprBlendMath;
      const int32_t tc0 = getTexcoord(ctx.src[0]);
      const int32_t tc1 = getTexcoord(ctx.src[1]);
      const bool src0Const = isFloatConstantRegisterType(ctx.src[0].id.type);
      const bool src1Const = isFloatConstantRegisterType(ctx.src[1].id.type);

      if (tc0 >= 0 && tc1 < 0 && src1Const) {
        assignFromSource(ctx.src[0], w);
        if (w.texcoord >= 0) {
          w.expressionFlags |= kPsSamplerExprUvTransform;
          loadScaleHintFromConstant(ctx.src[1], w.scaleHint);
        }
      } else if (tc1 >= 0 && tc0 < 0 && src0Const) {
        assignFromSource(ctx.src[1], w);
        if (w.texcoord >= 0) {
          w.expressionFlags |= kPsSamplerExprUvTransform;
          loadScaleHintFromConstant(ctx.src[0], w.scaleHint);
        }
      } else {
        w.texcoord = mergeSingle({ tc0, tc1 });
        if (w.texcoord < 0) {
          PsTexcoordScaleHint h0, h1;
          if (tryLoadConstantLikeHint(ctx.src[0], h0) &&
              tryLoadConstantLikeHint(ctx.src[1], h1) &&
              combineMulHints(h0, h1, w.constOnlyHint)) {
            w.hasConstOnlyHint = true;
          }
        }
      }
      if (w.texcoord >= 0) {
        w.expressionFlags |= kPsSamplerExprUvTransform;
      }
    }

    void SamplerUseAnalysis::deriveMadCoordinate(const DxsoInstructionContext& ctx, TempWrite& w) const {
      w.tracked = true;
      w.expressionFlags |= kPsSamplerExprBlendMath;
      const int32_t tc0 = getTexcoord(ctx.src[0]);
      const int32_t tc1 = getTexcoord(ctx.src[1]);
      const int32_t tc2 = getTexcoord(ctx.src[2]);
      const bool src0Const = isFloatConstantRegisterType(ctx.src[0].id.type);
      const bool src1Const = isFloatConstantRegisterType(ctx.src[1].id.type);
      const bool src2Const = isFloatConstantRegisterType(ctx.src[2].id.type);

      if (tc0 >= 0 && tc1 < 0 && src1Const) {
        assignFromSource(ctx.src[0], w);
        if (w.texcoord >= 0) {
          w.expressionFlags |= kPsSamplerExprUvTransform;
          loadScaleHintFromConstant(ctx.src[1], w.scaleHint);
          if (src2Const) {
            w.expressionFlags |= kPsSamplerExprUvOffset;
            applyOffsetFromConstant(ctx.src[2], w.scaleHint, 1.0f);
          }
        }
      } else if (tc1 >= 0 && tc0 < 0 && src0Const) {
        assignFromSource(ctx.src[1], w);
        if (w.texcoord >= 0) {
          w.expressionFlags |= kPsSamplerExprUvTransform;
          loadScaleHintFromConstant(ctx.src[0], w.scaleHint);
          if (src2Const) {
            w.expressionFlags |= kPsSamplerExprUvOffset;
            applyOffsetFromConstant(ctx.src[2], w.scaleHint, 1.0f);
          }
        }
      } else if (tc2 >= 0 && tc0 < 0 && tc1 < 0) {
        assignFromSource(ctx.src[2], w);
        if (w.texcoord >= 0) {
          PsTexcoordScaleHint mulHint, h0, h1;
          if (tryLoadConstantLikeHint(ctx.src[0], h0) &&
              tryLoadConstantLikeHint(ctx.src[1], h1) &&
              combineMulHints(h0, h1, mulHint) &&
              hasScaleTerm(mulHint)) {
            w.expressionFlags |= kPsSamplerExprUvOffset;
            setOffsetFromScaleTerm(w.scaleHint, mulHint, 1.0f);
          }
        }
      } else {
        w.texcoord = mergeSingle({ tc0, tc1, tc2 });
      }
      if (w.texcoord >= 0) {
        w.expressionFlags |= kPsSamplerExprUvTransform;
      }
    }

    void SamplerUseAnalysis::deriveSamplerValueRole(const DxsoInstructionContext& ctx, TempWrite& w) const {
      const uint8_t role0 = getSamplerValueRoleWithSwizzle(ctx.src[0]);
      const uint8_t role1 = getSamplerValueRoleWithSwizzle(ctx.src[1]);
      const uint8_t role2 = getSamplerValueRoleWithSwizzle(ctx.src[2]);
      if (!role0 && !role1 && !role2) {
        return;
      }

      const bool src0Control = (role0 & kSamplerValueRoleControl) != 0;
      const bool src1Control = (role1 & kSamplerValueRoleControl) != 0;
      const bool src2Control = (role2 & kSamplerValueRoleControl) != 0;
      const bool src0Color = (role0 & kSamplerValueRoleColor) != 0;
      const bool src1Color = (role1 & kSamplerValueRoleColor) != 0;
      const bool src2Color = (role2 & kSamplerValueRoleColor) != 0;

      auto markControl = [&] {
        w.expressionFlags |= kPsSamplerExprMaskControl;
        w.samplerValueRole |= kSamplerValueRoleControl;
      };
      auto markColor = [&] {
        w.expressionFlags |= kPsSamplerExprColorContribution;
        w.samplerValueRole |= kSamplerValueRoleColor;
      };

      switch (ctx.instruction.opcode) {
      case DxsoOpcode::Lrp:
      case DxsoOpcode::Cmp:
      case DxsoOpcode::Cnd:
        if (role0 && (src0Control || isScalarSwizzle(ctx.src[0]))) {
          markControl();
        }
        if (src1Color || src2Color) {
          markColor();
        }
        if ((role1 && !src1Color) || (role2 && !src2Color)) {
          markControl();
        }
        break;
      case DxsoOpcode::Slt:
      case DxsoOpcode::Sge:
      case DxsoOpcode::TexKill:
        if (src0Control || src1Control || src2Control ||
            isAlphaScalarSwizzle(ctx.src[0]) ||
            isAlphaScalarSwizzle(ctx.src[1]) ||
            isAlphaScalarSwizzle(ctx.src[2])) {
          markControl();
        } else {
          markColor();
        }
        break;
      case DxsoOpcode::Mul:
      case DxsoOpcode::Mad:
      case DxsoOpcode::Add:
      case DxsoOpcode::Sub: {
        const bool hasControl = src0Control || src1Control || src2Control;
        const bool hasColor = src0Color || src1Color || src2Color;
        if (hasControl) {
          markControl();
        }
        if (hasColor || !hasControl) {
          markColor();
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
        // Dot products, normalises and matrix transforms turn a sampled colour into direction or
        // coefficient data, which is neither this sampler's colour nor a mask of it.
        w.samplerValueRole = 0;
        break;
      default:
        markColor();
        break;
      }
    }

    uint8_t SamplerUseAnalysis::deriveValueState(const DxsoInstructionContext& ctx) const {
      const DxsoOpcode op = ctx.instruction.opcode;
      uint8_t state = 0;
      switch (op) {
      case DxsoOpcode::Mov:
        state = getSamplerValueState(ctx.src[0]);
        break;
      case DxsoOpcode::Mul: {
        const bool role0 = getSamplerValueRole(ctx.src[0]) != 0;
        const bool role1 = getSamplerValueRole(ctx.src[1]) != 0;
        if (role0 && isDefConstNearRgb(ctx.src[1], 2.0f)) {
          state |= kValueStateScaledX2;
          if ((getSamplerValueState(ctx.src[0]) & kValueStateBiasedHalf) != 0) {
            state |= kValueStateSignExpanded;  // (t - 0.5) * 2
          }
        } else if (role1 && isDefConstNearRgb(ctx.src[0], 2.0f)) {
          state |= kValueStateScaledX2;
          if ((getSamplerValueState(ctx.src[1]) & kValueStateBiasedHalf) != 0) {
            state |= kValueStateSignExpanded;
          }
        }
        break;
      }
      case DxsoOpcode::Add:
      case DxsoOpcode::Sub: {
        const bool isSub = op == DxsoOpcode::Sub;
        const uint8_t state0 = getSamplerValueState(ctx.src[0]);
        const uint8_t state1 = getSamplerValueState(ctx.src[1]);
        const bool role0 = getSamplerValueRole(ctx.src[0]) != 0;
        const bool role1 = getSamplerValueRole(ctx.src[1]) != 0;
        // t * 2 - 1, and the commuted and mirrored -1 + t * 2 and 1 - t * 2
        if (role0 && (state0 & kValueStateScaledX2) != 0 &&
            isDefConstNearRgb(ctx.src[1], isSub ? 1.0f : -1.0f)) {
          state |= kValueStateSignExpanded;
        }
        if (role1 && (state1 & kValueStateScaledX2) != 0 &&
            isDefConstNearRgb(ctx.src[0], isSub ? 1.0f : -1.0f)) {
          state |= kValueStateSignExpanded;
        }
        // t - 0.5, the first half of (t - 0.5) * 2
        if (role0 && isDefConstNearRgb(ctx.src[1], isSub ? 0.5f : -0.5f)) {
          state |= kValueStateBiasedHalf;
        }
        if (role1 && isDefConstNearRgb(ctx.src[0], isSub ? 0.5f : -0.5f)) {
          state |= kValueStateBiasedHalf;
        }
        break;
      }
      case DxsoOpcode::Mad: {
        const bool role0 = getSamplerValueRole(ctx.src[0]) != 0;
        const bool role1 = getSamplerValueRole(ctx.src[1]) != 0;
        if (isDefConstNearRgb(ctx.src[2], -1.0f) &&
            ((role0 && isDefConstNearRgb(ctx.src[1], 2.0f)) ||
             (role1 && isDefConstNearRgb(ctx.src[0], 2.0f)))) {
          state |= kValueStateSignExpanded;  // mad(t, 2, -1)
        }
        break;
      }
      default:
        break;
      }
      return state;
    }

    void SamplerUseAnalysis::writeTempTracking(const DxsoInstructionContext& ctx, const TempWrite& w) {
      if (w.texcoord < 0 && !w.tracked) {
        return;
      }

      const uint32_t dst = ctx.dst.id.num;
      m_tempToTexcoord[dst] = int8_t(w.texcoord);
      m_tempCoordExpressionFlags[dst] = w.expressionFlags;

      // fxc packs scalar results into spare lanes of live registers (`dp3 r0.w, n, n` while r0.xyz
      // holds a tracked colour), so a write missing every rgb lane merges and keeps the unpack state.
      if (maskWritesRgb(ctx.dst.mask)) {
        m_tempSamplerValueRole[dst] = w.samplerValueRole;
        m_tempSamplerValueState[dst] = w.valueState;
        m_tempAnchorBits[dst] = w.anchorBits;
      } else {
        m_tempSamplerValueRole[dst] |= w.samplerValueRole;
        m_tempAnchorBits[dst] |= w.anchorBits;
      }

      const uint8_t provenance =
        getCoordProvenance(ctx.src[0]) |
        getCoordProvenance(ctx.src[1]) |
        getCoordProvenance(ctx.src[2]);
      if (w.texcoord >= 0) {
        m_tempCoordProvenance[dst] = provenance | kCoordProvTexcoord;
        m_tempToScaleHint[dst] = hasScaleTerm(w.scaleHint) || hasOffsetTerm(w.scaleHint)
          ? w.scaleHint
          : PsTexcoordScaleHint{};
      } else {
        m_tempCoordProvenance[dst] = provenance;
        m_tempToScaleHint[dst] = w.hasConstOnlyHint && hasScaleTerm(w.constOnlyHint)
          ? w.constOnlyHint
          : PsTexcoordScaleHint{};
      }
    }

    // Values that only feed lighting or coordinate math carry a control-only role by the time
    // they reach the output, so they do not count.
    void SamplerUseAnalysis::detectOutputColor(const DxsoInstructionContext& ctx) {
      if (m_reachesOutputColor ||
          ctx.dst.id.type != DxsoRegisterType::ColorOut ||
          ctx.dst.id.num != 0 ||
          !maskWritesRgb(ctx.dst.mask)) {
        return;
      }
      for (const DxsoRegister& srcReg : { ctx.src[0], ctx.src[1], ctx.src[2] }) {
        if ((getSamplerValueRoleWithSwizzle(srcReg) & kSamplerValueRoleColor) != 0) {
          m_reachesOutputColor = true;
          return;
        }
      }
    }

    void SamplerUseAnalysis::decodeSampleOp(const DxsoInstructionContext& ctx, SampleOp& out) const {
      // ps_1_1 to ps_1_3 sample with the texcoord of the destination's own index.
      auto useDestinationTexcoord = [&] {
        out.coordRegStorage = ctx.dst;
        out.coordRegStorage.id.type = DxsoRegisterType::PixelTexcoord;
        out.coordRegStorage.id.num = ctx.dst.id.num;
        out.coordRegStorage.swizzle = DxsoRegSwizzle(0, 1, 2, 3);
        out.coordReg = &out.coordRegStorage;
      };

      switch (ctx.instruction.opcode) {
      case DxsoOpcode::Tex:
        if (m_info.majorVersion() >= 2) {
          out.sampler = ctx.src[1].id.num;
          out.coordReg = &ctx.src[0];
        } else if (m_info.majorVersion() == 1 && m_info.minorVersion() == 4) {
          out.sampler = ctx.dst.id.num;
          out.coordReg = &ctx.src[0];
        } else {
          out.sampler = ctx.dst.id.num;
          useDestinationTexcoord();
        }
        break;
      case DxsoOpcode::TexLdd:
      case DxsoOpcode::TexLdl:
        out.sampler = ctx.src[1].id.num;
        out.coordReg = &ctx.src[0];
        out.expressionFlags |= kPsSamplerExprBlendMath;
        break;
      case DxsoOpcode::TexBem:
      case DxsoOpcode::TexBemL:
        out.sampler = ctx.dst.id.num;
        useDestinationTexcoord();
        out.semanticFlags |= kPsSamplerSemanticNonDiffuse;
        out.expressionFlags |= kPsSamplerExprUvOffset;
        break;
      case DxsoOpcode::TexReg2Ar:
      case DxsoOpcode::TexReg2Gb:
      case DxsoOpcode::TexReg2Rgb:
      case DxsoOpcode::TexM3x2Tex:
      case DxsoOpcode::TexM3x3Tex:
      case DxsoOpcode::TexDp3Tex:
        out.sampler = ctx.dst.id.num;
        out.coordReg = &ctx.src[0];
        out.expressionFlags |= kPsSamplerExprUvTransform;
        break;
      case DxsoOpcode::TexM3x3Spec:
      case DxsoOpcode::TexM3x3VSpec:
        out.sampler = ctx.dst.id.num;
        out.coordReg = &ctx.src[0];
        out.semanticFlags |= kPsSamplerSemanticNonDiffuse;
        out.expressionFlags |= kPsSamplerExprUvTransform;
        out.expressionFlags |= kPsSamplerExprViewDependent;
        break;
      case DxsoOpcode::TexM3x2Depth:
      case DxsoOpcode::TexDepth:
        out.sampler = ctx.dst.id.num;
        out.coordReg = &ctx.src[0];
        out.semanticFlags |= kPsSamplerSemanticEngineAuxiliary;
        break;
      default:
        break;
      }
    }

    void SamplerUseAnalysis::trackSample(const DxsoInstructionContext& ctx) {
      SampleOp sample;
      decodeSampleOp(ctx, sample);
      if (sample.coordReg == nullptr) {
        return;
      }

      const bool dstIsTrackedTemp = isTrackedTemp(ctx.dst.id);
      const uint32_t dst = ctx.dst.id.num;

      // A sample replaces its destination. With r# registers reused this heavily, stale tracking
      // would credit another sampler's use, such as a normal-map unpack, to this one.
      if (dstIsTrackedTemp) {
        if (sample.sampler != m_samplerIdx) {
          m_tempSamplerValueRole[dst] = 0;
          m_tempSamplerValueState[dst] = 0;
        }
        m_tempAnchorBits[dst] =
          (sample.sampler < 32u && ((m_anchorLightmapSamplerMask >> sample.sampler) & 1u) != 0)
            ? kAnchorLightmapValue
            : 0u;
      }

      if (sample.sampler != m_samplerIdx) {
        return;
      }

      m_result.sampleCount = m_result.sampleCount < std::numeric_limits<uint16_t>::max()
        ? uint16_t(m_result.sampleCount + 1u)
        : std::numeric_limits<uint16_t>::max();

      // Accumulated over every sample: a sub-UV blend reads one texture through two coordinates.
      orConstDeps(*sample.coordReg, m_sampledCoordConstDeps);

      if (dstIsTrackedTemp) {
        uint8_t sampleRole = kSamplerValueRoleColor;
        if ((sample.semanticFlags & (kPsSamplerSemanticEngineAuxiliary | kPsSamplerSemanticNonDiffuse)) != 0) {
          sampleRole |= kSamplerValueRoleControl;
        }
        m_tempSamplerValueRole[dst] = sampleRole;
        m_tempSamplerValueState[dst] = 0;
      }

      const DxsoRegister* swizzleReg = sample.coordReg;
      int32_t tc = getTexcoord(*swizzleReg);
      uint8_t coordProvenance = getCoordProvenance(*sample.coordReg);
      uint16_t coordExpressionFlags =
        uint16_t(getCoordExpressionFlags(*sample.coordReg) | sample.expressionFlags);
      if ((sample.semanticFlags & kPsSamplerSemanticNonDiffuse) != 0) {
        coordExpressionFlags |= kPsSamplerExprMaskControl;
      }
      if ((sample.semanticFlags & kPsSamplerSemanticEngineAuxiliary) == 0) {
        coordExpressionFlags |= kPsSamplerExprColorContribution;
      }
      m_result.semanticFlags |= sample.semanticFlags;

      // When the coordinate register says nothing, the other operands or the destination may.
      const std::array<DxsoRegister, 4> fallbackRegs = {
        ctx.src[0], ctx.src[1], ctx.src[2], ctx.dst
      };

      if ((coordProvenance & kCoordProvTexcoord) == 0 || !coordExpressionFlags) {
        for (const DxsoRegister& fallbackReg : fallbackRegs) {
          coordProvenance |= getCoordProvenance(fallbackReg);
          coordExpressionFlags |= getCoordExpressionFlags(fallbackReg);
          if ((coordProvenance & kCoordProvTexcoord) != 0) {
            break;
          }
        }
      }

      if (tc < 0) {
        for (const DxsoRegister& fallbackReg : fallbackRegs) {
          const int32_t fallbackTc = getTexcoord(fallbackReg);
          if (fallbackTc >= 0) {
            tc = fallbackTc;
            swizzleReg = &fallbackReg;
            break;
          }
        }
      }

      const bool coordUsesTexcoord = (coordProvenance & kCoordProvTexcoord) != 0;
      const bool coordUsesNonTexcoord = (coordProvenance & kCoordProvNonTexcoord) != 0;
      if (coordUsesTexcoord) {
        m_texcoordDerivedSampleCount++;
      } else if (coordUsesNonTexcoord) {
        m_nonTexcoordDerivedSampleCount++;
      }
      if (coordUsesNonTexcoord && !coordUsesTexcoord) {
        coordExpressionFlags |= kPsSamplerExprViewDependent;
      }

      if (tc < 0 || uint32_t(tc) >= kTrackedTexcoords) {
        m_sampledCoordExpressionFlags |= coordExpressionFlags;
        return;
      }

      m_texcoordUseCount[tc]++;
      const uint32_t compU = swizzleReg->swizzle[0] & 0x3u;
      const uint32_t compV = swizzleReg->swizzle[1] & 0x3u;
      uint16_t& pairCount = m_texcoordCoordPairUseCount[tc][(compU << 2) | compV];
      if (pairCount < std::numeric_limits<uint16_t>::max()) {
        pairCount = uint16_t(pairCount + 1u);
      }

      PsTexcoordScaleHint sampleScaleHint;
      if (getScaleHint(*sample.coordReg, sampleScaleHint)) {
        if (hasScaleTerm(sampleScaleHint)) {
          coordExpressionFlags |= kPsSamplerExprUvTransform;
        }
        if (hasOffsetTerm(sampleScaleHint)) {
          coordExpressionFlags |= kPsSamplerExprUvOffset;
        }
        ScaleHintVote& vote = m_perTexcoordScaleHints[tc];
        if (!vote.valid) {
          vote.valid = true;
          vote.hint = sampleScaleHint;
        } else if (!isSameScaleHint(vote.hint, sampleScaleHint)) {
          vote.conflict = true;
        }
      }

      m_sampledCoordExpressionFlags |= coordExpressionFlags;
    }

    // The most-sampled texcoord. A tie goes to the one tied texcoord whose samples agree on a
    // scale hint, if exactly one does.
    void SamplerUseAnalysis::voteTexcoord() {
      int32_t found = -1;
      uint32_t bestCount = 0;
      bool tie = false;
      for (uint32_t i = 0; i < kTrackedTexcoords; i++) {
        const uint32_t count = m_texcoordUseCount[i];
        if (count > bestCount) {
          bestCount = count;
          found = int32_t(i);
          tie = false;
        } else if (count > 0 && count == bestCount) {
          tie = true;
        }
      }

      if (tie) {
        found = -1;
        for (uint32_t i = 0; i < kTrackedTexcoords; i++) {
          const ScaleHintVote& vote = m_perTexcoordScaleHints[i];
          if (m_texcoordUseCount[i] != bestCount || !vote.valid || vote.conflict) {
            continue;
          }
          if (found >= 0) {
            found = -1;
            break;
          }
          found = int32_t(i);
        }
      }

      m_result.texcoord = found;
    }

    void SamplerUseAnalysis::applyTexcoordVote() {
      if (m_result.texcoord < 0) {
        return;
      }
      const uint32_t tc = uint32_t(m_result.texcoord);

      // The most-read component pair, when it beats the runner-up or has two thirds of the reads.
      uint32_t totalPairCount = 0;
      uint32_t bestPairCount = 0;
      uint32_t secondBestPairCount = 0;
      uint32_t bestPairIdx = 0;
      for (uint32_t pairIdx = 0; pairIdx < m_texcoordCoordPairUseCount[tc].size(); pairIdx++) {
        const uint32_t pairCount = m_texcoordCoordPairUseCount[tc][pairIdx];
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
          m_result.coordCompValid = true;
          m_result.coordCompU = uint8_t((bestPairIdx >> 2) & 0x3u);
          m_result.coordCompV = uint8_t(bestPairIdx & 0x3u);
        }
      }

      const ScaleHintVote& vote = m_perTexcoordScaleHints[tc];
      if (vote.valid && !vote.conflict) {
        m_result.scaleConstReg = vote.hint.constReg;
        m_result.scaleConstCompU = vote.hint.compU;
        m_result.scaleConstCompV = vote.hint.compV;
        m_result.scaleFactorU = vote.hint.scaleFactorU;
        m_result.scaleFactorV = vote.hint.scaleFactorV;
        m_result.scaleImmediateValid = vote.hint.immediateValid;
        m_result.scaleImmediateU = vote.hint.immediateU;
        m_result.scaleImmediateV = vote.hint.immediateV;
        m_result.offsetConstReg = vote.hint.offsetConstReg;
        m_result.offsetConstCompU = vote.hint.offsetCompU;
        m_result.offsetConstCompV = vote.hint.offsetCompV;
        m_result.offsetFactorU = vote.hint.offsetFactorU;
        m_result.offsetFactorV = vote.hint.offsetFactorV;
        m_result.offsetImmediateValid = vote.hint.offsetImmediateValid;
        m_result.offsetImmediateU = vote.hint.offsetImmediateU;
        m_result.offsetImmediateV = vote.hint.offsetImmediateV;
      }
    }

    void SamplerUseAnalysis::collectCoordConstRegs() {
      for (uint32_t reg = 0; reg < kDxsoMaxPsFloatConstants; reg++) {
        if ((m_sampledCoordConstDeps[reg / 64u] >> (reg % 64u)) & 1ull) {
          m_result.coordConstRegs.push_back(reg);
        }
      }
    }

    void SamplerUseAnalysis::deriveExpressionFlags() {
      PsSamplerTexcoordInference& r = m_result;
      r.expressionFlags = m_sampledCoordExpressionFlags;
      if (r.scaleConstReg >= 0 || r.scaleImmediateValid) {
        r.expressionFlags |= kPsSamplerExprUvTransform;
      }
      if (r.offsetConstReg >= 0 || r.offsetImmediateValid) {
        r.expressionFlags |= kPsSamplerExprUvOffset;
      }
      if (r.coordCompValid && (r.coordCompU != 0 || r.coordCompV != 1)) {
        r.expressionFlags |= kPsSamplerExprUvTransform;
      }
      if ((r.expressionFlags & (kPsSamplerExprUvTransform | kPsSamplerExprUvOffset)) != 0 &&
          r.sampleCount >= 2) {
        r.expressionFlags |= kPsSamplerExprUvAnimated;
      }
      if ((r.expressionFlags & kPsSamplerExprMaskControl) != 0 &&
          (r.semanticFlags & (kPsSamplerSemanticEngineAuxiliary | kPsSamplerSemanticLightmap)) == 0) {
        r.semanticFlags |= kPsSamplerSemanticNonDiffuse;
      }
      if ((r.expressionFlags & kPsSamplerExprViewDependent) != 0 &&
          (r.semanticFlags & (kPsSamplerSemanticEngineAuxiliary | kPsSamplerSemanticLightmap)) == 0) {
        r.semanticFlags |= kPsSamplerSemanticNonDiffuse;
      }
      // Not NonDiffuse: lit shaders remap colour terms by *2-1 too, so scoring applies the decode
      // only to samplers bound without SRGB, as UE3 binds its normal maps.
      if (m_normalDecodeDetected) {
        r.expressionFlags |= kPsSamplerExprNormalDecode;
      }
      // ps_1_x has no oC0; the final r0 is the output colour.
      if (!m_reachesOutputColor && m_info.majorVersion() == 1 &&
          (m_tempSamplerValueRole[0] & kSamplerValueRoleColor) != 0) {
        m_reachesOutputColor = true;
      }
      if (m_reachesOutputColor) {
        r.expressionFlags |= kPsSamplerExprReachesOutputColor;
      }
      if (m_diffuseAnchorDetected) {
        r.expressionFlags |= kPsSamplerExprDiffuseAnchor;
      }
    }

    void SamplerUseAnalysis::applyCtabNameFlags() {
      const DxsoCtab& ctab = m_decoder.getCtabInfo();
      if (ctab.m_size == 0) {
        return;
      }

      for (const DxsoCtab::Constant& c : ctab.m_constantData) {
        if (c.registerSet != kD3dxRegisterSetSampler || c.registerCount == 0) {
          continue;
        }

        const uint64_t regBegin = c.registerIndex;
        const uint64_t regEnd = regBegin + c.registerCount;
        if (uint64_t(m_samplerIdx) < regBegin || uint64_t(m_samplerIdx) >= regEnd) {
          continue;
        }

        const uint8_t semanticFlagsFromName = classifyPixelSamplerSemanticFlags(c.name);
        m_result.semanticFlags |= semanticFlagsFromName;
        m_result.expressionFlags |= classifyPixelSamplerExpressionFlagsFromName(c.name);
        // A generic Texture2D_* name does not imply colour; the dataflow decides that.
        if ((semanticFlagsFromName & (kPsSamplerSemanticEngineAuxiliary | kPsSamplerSemanticNonDiffuse)) != 0) {
          m_result.expressionFlags |= kPsSamplerExprMaskControl;
        }
      }

      if (m_result.scaleConstReg < 0 && m_result.offsetConstReg < 0) {
        return;
      }
      for (const DxsoCtab::Constant& c : ctab.m_constantData) {
        if (!isTimeDrivenConstantName(toLowerAscii(c.name))) {
          continue;
        }
        if (constantRangeContainsRegister(c, m_result.scaleConstReg) ||
            constantRangeContainsRegister(c, m_result.offsetConstReg)) {
          m_result.expressionFlags |= kPsSamplerExprUvTimeDriven;
          m_result.expressionFlags |= kPsSamplerExprUvAnimated;
          break;
        }
      }
    }

    // For shaders whose sampler names are generic or stripped.
    void SamplerUseAnalysis::applySemanticFallbacks() {
      PsSamplerTexcoordInference& r = m_result;

      // UV math on a texcoord-derived coordinate makes a material texture...
      const bool hasUvExpression =
        (r.expressionFlags & (kPsSamplerExprUvTransform | kPsSamplerExprUvOffset | kPsSamplerExprUvAnimated)) != 0;
      if (r.sampleCount > 0 &&
          r.texcoord >= 0 &&
          hasUvExpression &&
          (r.semanticFlags & (kPsSamplerSemanticEngineAuxiliary | kPsSamplerSemanticLightmap)) == 0) {
        r.semanticFlags |= kPsSamplerSemanticMaterialTexture;
      }

      // ...and sampling only from non-texcoord sources makes a screen or post-process input.
      if (r.sampleCount > 0 &&
          r.texcoord < 0 &&
          m_texcoordDerivedSampleCount == 0 &&
          m_nonTexcoordDerivedSampleCount > 0 &&
          (r.semanticFlags & kPsSamplerSemanticMaterialTexture) == 0) {
        r.semanticFlags |= kPsSamplerSemanticEngineAuxiliary;
      }
    }

  }

  PsSamplerTexcoordInference inferPixelShaderTexcoordForSampler(const DxsoShaderView& pixelShader, const uint32_t samplerIdx) {
    if (pixelShader.tokens == nullptr || pixelShader.info == nullptr || pixelShader.isgn == nullptr ||
        samplerIdx >= kDxsoMaxPsSamplers || pixelShader.info->type() != DxsoProgramTypes::PixelShader) {
      return PsSamplerTexcoordInference();
    }
    return SamplerUseAnalysis(pixelShader, samplerIdx).run();
  }

  // A material input the base pass reads only for lighting. The simple-lightmap compile strips
  // such inputs, so dropping them keeps a material's identity the same under both
  // DirectionalLightmaps settings. Fallback for bytecode the colour-term analysis cannot read
  // (see classifyDxsoColorTermSource).
  bool isUe3LightingInputSampler(const PsSamplerTexcoordInference& inferred) {
    // Albedo selection's decisive signal; a material's own base texture is never an input.
    if ((inferred.expressionFlags & kPsSamplerExprDiffuseAnchor) != 0) {
      return false;
    }
    if ((inferred.expressionFlags & kPsSamplerExprNormalDecode) != 0) {
      return true;
    }
    if ((inferred.semanticFlags & kPsSamplerSemanticNonDiffuse) != 0) {
      return true;
    }
    // Specular and mask inputs collapse into dot products and comparisons before oC0.
    if (inferred.sampleCount > 0 &&
        (inferred.expressionFlags & kPsSamplerExprReachesOutputColor) == 0) {
      return true;
    }
    return false;
  }

}
