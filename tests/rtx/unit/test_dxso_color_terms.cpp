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

// Colour-term analysis of UE3 base-pass pixel shaders. The hand-assembled cases follow the
// instruction streams fxc emits for Mirror's Edge's BasePassPixelShader.usf under each
// lightmap policy, so a material's inputs classify the same way in every permutation.
//
// Dump mode: test_dxso_color_terms <shader.dxso> [...]  prints the classification of a
// dumped pixel shader (DXVK_SHADER_DUMP_PATH), deriving the inputs from its CTAB the same
// way the runtime does.

#include <bitset>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../../src/dxso/dxso_code.h"
#include "../../../src/dxso/dxso_color_terms.h"
#include "../../../src/dxso/dxso_decoder.h"
#include "../../../src/util/log/log.h"

namespace dxvk {
  // Standalone executables own the logger singleton, as the d3d9/dxgi entry points do.
  Logger Logger::s_instance("test_dxso_color_terms.log", LogLevel::None);
}

using namespace dxvk;

namespace {

  constexpr uint32_t kPs30Header = 0xFFFF0300u;
  constexpr uint32_t kEndToken = 0x0000FFFFu;

  constexpr uint32_t kRegTemp = 0u;
  constexpr uint32_t kRegInput = 1u;
  constexpr uint32_t kRegConst = 2u;
  constexpr uint32_t kRegColorOut = 8u;
  constexpr uint32_t kRegSampler = 10u;

  constexpr uint32_t kModNeg = 1u;
  constexpr uint32_t kModAbs = 11u;

  enum WriteMask : uint32_t {
    MaskX = 0x1u, MaskY = 0x2u, MaskZ = 0x4u, MaskW = 0x8u,
    MaskXY = 0x3u, MaskZW = 0xCu, MaskXYZ = 0x7u, MaskXZW = 0xDu, MaskAll = 0xFu,
  };

  uint32_t swz(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    return x | (y << 2) | (z << 4) | (w << 6);
  }
  const uint32_t kXYZW = swz(0, 1, 2, 3);
  const uint32_t kXXXX = swz(0, 0, 0, 0);
  const uint32_t kYYYY = swz(1, 1, 1, 1);
  const uint32_t kZZZZ = swz(2, 2, 2, 2);
  const uint32_t kWWWW = swz(3, 3, 3, 3);

  uint32_t encodeRegisterType(uint32_t type) {
    return ((type & 0x7u) << 28) | ((type & 0x18u) << 8);
  }

  uint32_t dst(uint32_t type, uint32_t num, uint32_t mask) {
    return 0x80000000u | encodeRegisterType(type) | ((mask & 0xFu) << 16) | (num & 0x7FFu);
  }

  uint32_t src(uint32_t type, uint32_t num, uint32_t swizzle = kXYZW, uint32_t modifier = 0) {
    return 0x80000000u | encodeRegisterType(type) | ((modifier & 0xFu) << 24) | ((swizzle & 0xFFu) << 16) | (num & 0x7FFu);
  }

  uint32_t r(uint32_t n, uint32_t s = kXYZW, uint32_t m = 0) { return src(kRegTemp, n, s, m); }
  uint32_t v(uint32_t n, uint32_t s = kXYZW, uint32_t m = 0) { return src(kRegInput, n, s, m); }
  uint32_t c(uint32_t n, uint32_t s = kXYZW, uint32_t m = 0) { return src(kRegConst, n, s, m); }
  uint32_t smp(uint32_t n) { return src(kRegSampler, n); }
  uint32_t rd(uint32_t n, uint32_t mask = MaskAll) { return dst(kRegTemp, n, mask); }
  uint32_t oC0(uint32_t mask = MaskAll) { return dst(kRegColorOut, 0, mask); }

  uint32_t opcodeToken(DxsoOpcode opcode, uint32_t length) {
    return uint32_t(opcode) | ((length & 0xFu) << 24);
  }

  uint32_t floatBits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
  }

  class PsBuilder {
  public:
    PsBuilder() {
      m_tokens.push_back(kPs30Header);
    }

    PsBuilder& def(uint32_t constNum, float x, float y, float z, float w) {
      m_tokens.push_back(opcodeToken(DxsoOpcode::Def, 5));
      m_tokens.push_back(dst(kRegConst, constNum, MaskAll));
      m_tokens.push_back(floatBits(x));
      m_tokens.push_back(floatBits(y));
      m_tokens.push_back(floatBits(z));
      m_tokens.push_back(floatBits(w));
      return *this;
    }

    PsBuilder& dclInput(DxsoUsage usage, uint32_t usageIndex, uint32_t regNum, uint32_t mask = MaskAll) {
      m_tokens.push_back(opcodeToken(DxsoOpcode::Dcl, 2));
      m_tokens.push_back(0x80000000u | uint32_t(usage) | ((usageIndex & 0xFu) << 16));
      m_tokens.push_back(dst(kRegInput, regNum, mask));
      return *this;
    }

    PsBuilder& dclSampler(uint32_t samplerNum, DxsoTextureType type = DxsoTextureType::Texture2D) {
      m_tokens.push_back(opcodeToken(DxsoOpcode::Dcl, 2));
      m_tokens.push_back(0x80000000u | (uint32_t(type) << 27));
      m_tokens.push_back(dst(kRegSampler, samplerNum, MaskAll));
      return *this;
    }

    PsBuilder& texld(uint32_t dstToken, uint32_t coord, uint32_t samplerNum) {
      return op2(DxsoOpcode::Tex, dstToken, coord, smp(samplerNum));
    }

    PsBuilder& texkill(uint32_t regToken) {
      m_tokens.push_back(opcodeToken(DxsoOpcode::TexKill, 1));
      m_tokens.push_back(regToken);
      return *this;
    }

    PsBuilder& op1(DxsoOpcode opcode, uint32_t d, uint32_t s0) {
      m_tokens.push_back(opcodeToken(opcode, 2));
      m_tokens.push_back(d);
      m_tokens.push_back(s0);
      return *this;
    }

    PsBuilder& op2(DxsoOpcode opcode, uint32_t d, uint32_t s0, uint32_t s1) {
      m_tokens.push_back(opcodeToken(opcode, 3));
      m_tokens.push_back(d);
      m_tokens.push_back(s0);
      m_tokens.push_back(s1);
      return *this;
    }

    PsBuilder& op3(DxsoOpcode opcode, uint32_t d, uint32_t s0, uint32_t s1, uint32_t s2) {
      m_tokens.push_back(opcodeToken(opcode, 4));
      m_tokens.push_back(d);
      m_tokens.push_back(s0);
      m_tokens.push_back(s1);
      m_tokens.push_back(s2);
      return *this;
    }

    DxsoColorTermResult analyze(const DxsoColorTermInputs& inputs) const {
      std::vector<uint32_t> tokens = m_tokens;
      tokens.push_back(kEndToken);
      return analyzeDxsoColorTerms(tokens.data(), tokens.size(), inputs);
    }

  private:
    std::vector<uint32_t> m_tokens;
  };

  const char* roleName(DxsoColorTermRole role) {
    switch (role) {
    case DxsoColorTermRole::Unused: return "Unused";
    case DxsoColorTermRole::Color: return "Color";
    case DxsoColorTermRole::Opacity: return "Opacity";
    case DxsoColorTermRole::Coordinate: return "Coordinate";
    case DxsoColorTermRole::LightingOnly: return "LightingOnly";
    default: return "?";
    }
  }

  std::string describeSet(DxsoColorTermSet set) {
    std::string out = "{";
    for (uint32_t s = 0; s < kDxsoColorTermSignatureCount; s++) {
      if (((set >> s) & 1u) == 0) {
        continue;
      }
      if (out.size() > 1) {
        out += ",";
      }
      if (s == 0) {
        out += "plain";
      } else {
        if (s & DxsoColorTermFactor_Lightmap) out += "LM";
        if (s & DxsoColorTermFactor_LightConst) out += "LC";
        if (s & DxsoColorTermFactor_Interpolant) out += "IN";
        if (s & DxsoColorTermFactor_View) out += "VIEW";
        if (s & DxsoColorTermFactor_SpecularTransfer) out += "SPEC";
      }
    }
    return out + "}";
  }

  void expectSampler(const DxsoColorTermResult& result, uint32_t sampler, DxsoColorTermRole expected, const char* label) {
    const DxsoColorTermRole role = classifyDxsoColorTermSource(result.samplerTerms[sampler], result.samplerReach[sampler], result.hasLightConstTerm);
    if (role != expected) {
      std::cerr << label << ": sampler s" << sampler << " classified " << roleName(role)
                << " (terms " << describeSet(result.samplerTerms[sampler]) << ", hasLC="
                << result.hasLightConstTerm << "), expected " << roleName(expected) << std::endl;
      throw std::runtime_error(label);
    }
  }

  void expectConst(const DxsoColorTermResult& result, uint32_t reg, DxsoColorTermRole expected, const char* label) {
    const DxsoColorTermRole role = classifyDxsoColorTermSource(result.constTerms[reg], result.constReach[reg], result.hasLightConstTerm);
    if (role != expected) {
      std::cerr << label << ": constant c" << reg << " classified " << roleName(role)
                << " (terms " << describeSet(result.constTerms[reg]) << ", hasLC="
                << result.hasLightConstTerm << "), expected " << roleName(expected) << std::endl;
      throw std::runtime_error(label);
    }
  }

  void expectLiteral(const DxsoColorTermResult& result, float value, bool expectUnlit, const char* label) {
    const uint32_t bits = floatBits(value);
    for (const DxsoColorTermLiteral& lit : result.literals) {
      if (lit.bits != bits) {
        continue;
      }
      const bool unlit = dxsoColorTermSetHasUnlitTerm(lit.terms);
      if (unlit != expectUnlit) {
        std::cerr << label << ": literal " << value << " terms " << describeSet(lit.terms)
                  << ", expected unlit=" << expectUnlit << std::endl;
        throw std::runtime_error(label);
      }
      return;
    }
    std::cerr << label << ": literal " << value << " was never read" << std::endl;
    throw std::runtime_error(label);
  }

  void expectFlag(bool actual, bool expected, const char* what, const char* label) {
    if (actual != expected) {
      std::cerr << label << ": " << what << " = " << actual << ", expected " << expected << std::endl;
      throw std::runtime_error(label);
    }
  }

  class ColorTermTestApp {
  public:
    static void run() {
      std::cout << "Running DXSO colour-term analysis tests..." << std::endl;
      test_simpleVertexLightmap();
      test_directionalVertexLightmapWithSpecularTexture();
      test_simpleTextureLightmapMasked();
      test_directionalTextureLightmapWithTwoSidedMask();
      test_fresnelEmissiveCubeKeepsViewDependentColor();
      test_blendWeightTextureReachesColor();
      test_texturelessLiterals();
      test_noLightMapPolicySpecularVector();
      test_desaturationKeepsColor();
      test_coordinateConstantDependenciesArePerLane();
      test_plainReadIgnoresPackedLightmapMath();
      test_rejectsVertexShaderAndPs1x();
      std::cout << "All DXSO colour-term analysis tests passed." << std::endl;
    }

  private:
    static void expectCoordRegs(const DxsoColorTermResult& res, uint32_t sampler, std::initializer_list<uint32_t> expected, const char* label) {
      std::bitset<kDxsoColorTermMaxConstRegs> want;
      for (const uint32_t reg : expected) {
        want.set(reg);
      }
      if (res.samplerCoordConstRegs[sampler] != want) {
        std::cerr << label << ": sampler s" << sampler << " coordinate constant registers {";
        for (uint32_t reg = 0; reg < kDxsoColorTermMaxConstRegs; reg++) {
          if (res.samplerCoordConstRegs[sampler].test(reg)) std::cerr << " c" << reg;
        }
        std::cerr << " }, expected {";
        for (const uint32_t reg : expected) std::cerr << " c" << reg;
        std::cerr << " }" << std::endl;
        throw std::runtime_error(label);
      }
    }

    // The registers a coordinate is built from, per lane. fxc packs unrelated scalars into the
    // spare lanes of live registers, and the bicubic lightmap filter gives it dozens of
    // coordinate instructions to pack against, so a register-granular dependency set made a
    // material tint look like a texture transform in one lightmap-policy compile only.
    static void test_coordinateConstantDependenciesArePerLane() {
      std::cout << "  test_coordinateConstantDependenciesArePerLane" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.5f, 1.0f, 0.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1, MaskXY)
        .dclSampler(0)
        .dclSampler(1)
        .dclSampler(2)
        .dclSampler(3)
        // r0.xy = uv * LightMapResolution (c5) - 0.5, an engine transform
        .op3(DxsoOpcode::Mad, rd(0, MaskXY), v(0), c(5), c(1, kXXXX, kModNeg))
        // r0.w = tint.x * 2, the material's UniformVector_0 (c2) packed into the spare lane
        .op2(DxsoOpcode::Mul, rd(0, MaskW), c(2, kXXXX), c(1, kYYYY))
        // dependent read: weights = tex2D(BSpline, frac(r0.xy)) with the packed lane still live
        .op1(DxsoOpcode::Frc, rd(1, MaskXY), r(0))
        .texld(rd(2), r(1), 0)
        // lightmap coordinate: (floor(r0) + weights) * InvRes (c6)
        .op2(DxsoOpcode::Add, rd(3, MaskXY), r(0), r(2))
        .op2(DxsoOpcode::Mul, rd(3, MaskXY), r(3), c(6))
        .texld(rd(4), r(3), 1)
        // rotator through dp2add: r5.x = dot(uv, c7.xy) + c8.x; r5.y = dot(uv, c7.zw) + c8.y
        .op3(DxsoOpcode::Dp2Add, rd(5, MaskX), v(1), c(7), c(8, kXXXX))
        .op3(DxsoOpcode::Dp2Add, rd(5, MaskY), v(1), c(7, swz(2, 3, 2, 3)), c(8, kYYYY))
        .texld(rd(6), r(5), 2)
        // matrix transform: m3x2 reads c9 and the implicit next row c10
        .op2(DxsoOpcode::M3x2, rd(7, MaskXY), v(1, swz(0, 1, 1, 1)), c(9))
        .texld(rd(8), r(7), 3)
        // colour: diffuse * tint lane, so the tint is a colour input, not a coordinate one
        .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(6), r(0, kWWWW))
        .op1(DxsoOpcode::Mov, oC0(MaskW), c(1, kYYYY));

      DxsoColorTermInputs in;
      in.trackedSamplerMask = (1u << 2);
      in.lightmapSamplerMask = (1u << 1);
      in.trackedConstRegs.set(2);
      const DxsoColorTermResult res = ps.analyze(in);
      expectFlag(res.analyzed, true, "analyzed", "per-lane coordinate deps");
      // the packed tint lane never reaches a coordinate; the literal c1 is not a dependency
      expectCoordRegs(res, 0, { 5 }, "per-lane coordinate deps: BSpline read");
      expectCoordRegs(res, 1, { 5, 6 }, "per-lane coordinate deps: lightmap through dependent read");
      expectCoordRegs(res, 2, { 7, 8 }, "per-lane coordinate deps: dp2add rotator");
      expectCoordRegs(res, 3, { 9, 10 }, "per-lane coordinate deps: m3x2 rows");
      expectConst(res, 2, DxsoColorTermRole::Color, "per-lane coordinate deps: tint stays colour");

      // coordinate arithmetic per sampler: the BSpline read is offset and wrapped; the lightmap
      // read adds a sampled value (unknown offset) on top; the rotator offsets by constants; the
      // matrix read is arithmetic only
      expectFlag(res.samplerCoordExpr[0] == (DxsoCoordExpr_Arith | DxsoCoordExpr_Offset | DxsoCoordExpr_Wrap), true,
                 "BSpline coord expr = Arith|Offset|Wrap", "coordinate arithmetic");
      expectFlag((res.samplerCoordExpr[1] & DxsoCoordExpr_UnknownOffset) != 0, true,
                 "lightmap coord has unknown (sampled) offset", "coordinate arithmetic");
      expectFlag(res.samplerCoordExpr[2] == (DxsoCoordExpr_Arith | DxsoCoordExpr_Offset), true,
                 "rotator coord expr = Arith|Offset", "coordinate arithmetic");
      expectFlag(res.samplerCoordExpr[3] == DxsoCoordExpr_Arith, true,
                 "matrix coord expr = Arith", "coordinate arithmetic");
      expectFlag(res.samplerSampleCount[1] == 1, true, "lightmap sampled once", "coordinate arithmetic");
    }

    // The material UV and the lightmap UV share one interpolator (TEXCOORD0.xy / .zw). A plain
    // diffuse read of v0.xy must carry none of the bicubic lightmap math on v0.zw, even when the
    // compiler keeps both in one register.
    static void test_plainReadIgnoresPackedLightmapMath() {
      std::cout << "  test_plainReadIgnoresPackedLightmapMath" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.5f, 1.0f, 0.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskAll)
        .dclSampler(0)
        .dclSampler(1)
        // r0.zw = v0.zw * LightMapResolution - 0.5 (bicubic lightmap path)
        .op3(DxsoOpcode::Mad, rd(0, MaskZW), v(0), c(5), c(1, kXXXX, kModNeg))
        // r0.xy = v0.xy, the diffuse UV moved into the same register unchanged
        .op1(DxsoOpcode::Mov, rd(0, MaskXY), v(0))
        .op1(DxsoOpcode::Frc, rd(1, MaskZW), r(0))
        .texld(rd(2), r(1, swz(2, 3, 2, 3)), 1)                          // lightmap-side read of r1.zw
        .texld(rd(3), r(0), 0)                                            // diffuse reads r0.xy
        .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(3), r(2))
        .op1(DxsoOpcode::Mov, oC0(MaskW), c(1, kYYYY));

      DxsoColorTermInputs in;
      in.trackedSamplerMask = 1u << 0;
      in.lightmapSamplerMask = 1u << 1;
      const DxsoColorTermResult res = ps.analyze(in);
      expectFlag(res.analyzed, true, "analyzed", "packed lightmap math");
      expectFlag(res.samplerCoordExpr[0] == 0, true, "diffuse coord expr empty", "packed lightmap math");
      expectCoordRegs(res, 0, {}, "packed lightmap math: diffuse coordinate depends on no constant");
      expectFlag(res.samplerCoordExpr[1] == (DxsoCoordExpr_Arith | DxsoCoordExpr_Offset | DxsoCoordExpr_Wrap), true,
                 "lightmap coord expr = Arith|Offset|Wrap", "packed lightmap math");
      expectCoordRegs(res, 1, { 5 }, "packed lightmap math: lightmap coordinate depends on LightMapResolution");
    }
    // Mirror's Edge base pass, SIMPLE_VERTEX_LIGHTMAP: `pow(LightMapA, 1) * Diffuse + Emissive`
    // with the coefficient arriving in TEXCOORD2. fxc lowers the pow to max(|v1|, eps).
    static void test_simpleVertexLightmap() {
      std::cout << "  test_simpleVertexLightmap" << std::endl;
      PsBuilder ps;
      ps.def(1, 9.99999975e-005f, 1.0f, 0.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskZ | MaskW)
        .dclInput(DxsoUsage::Texcoord, 2, 1, MaskXYZ)
        .dclInput(DxsoUsage::Texcoord, 5, 2, MaskW)
        .dclSampler(0)
        .texld(rd(0), v(0, swz(3, 2, 2, 3)), 0)                         // Texture2D_1 (diffuse)
        .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(0), c(2))                // * UniformVector_1
        .op1(DxsoOpcode::Mov, rd(1, MaskY), c(1, kYYYY))
        .op2(DxsoOpcode::Add, rd(1, MaskXYZ), r(1, kYYYY), c(0, kXYZW, kModNeg)) // 1 - UniformVector_0
        .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(0), r(1))
        .op2(DxsoOpcode::Max, rd(1, MaskXYZ), v(1, kXYZW, kModAbs), c(1, kXXXX))  // pow(LM, 1)
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(1), r(0), c(0))            // LM * D + Emissive
        .op1(DxsoOpcode::Mov, oC0(MaskW), v(2, kWWWW));

      DxsoColorTermInputs in;
      in.trackedSamplerMask = 1u << 0;
      in.trackedConstRegs.set(0);
      in.trackedConstRegs.set(2);
      in.trackLiterals = true;
      const DxsoColorTermResult res = ps.analyze(in);
      expectFlag(res.analyzed, true, "analyzed", "simple vertex lightmap");
      expectFlag(res.hasLightConstTerm, false, "hasLightConstTerm", "simple vertex lightmap");
      expectSampler(res, 0, DxsoColorTermRole::Color, "simple vertex lightmap: diffuse");
      expectConst(res, 0, DxsoColorTermRole::Color, "simple vertex lightmap: emissive/diffuse factor");
      expectConst(res, 2, DxsoColorTermRole::Color, "simple vertex lightmap: tint");
      expectFlag(dxsoColorTermSetHasUnlitTerm(res.constTerms[0]), true, "emissive unlit term", "simple vertex lightmap");
      // the pow epsilon only ever gates the lightmap; the `1` only reaches colour lit
      expectLiteral(res, 9.99999975e-005f, false, "simple vertex lightmap: eps");
      expectLiteral(res, 1.0f, false, "simple vertex lightmap: one");
    }

    // The same material compiled with VERTEX_LIGHTMAP (three coefficients, TEXCOORD2/3/4):
    // normal map Texture2D_0 drives the basis transfer, Texture2D_2 is a specular colour
    // texture multiplied by pow(saturate(R.Basis), 7) * coefficient, and the diffuse also
    // reaches the output through AmbientColorAndSkyFactor.
    static void test_directionalVertexLightmapWithSpecularTexture() {
      std::cout << "  test_directionalVertexLightmapWithSpecularTexture" << std::endl;
      PsBuilder ps;
      ps.def(1, 2.0f, -1.0f, 9.99999975e-005f, 7.0f)
        .def(4, 0.816496611f, 0.577350259f, 0.0f, 0.0f)
        .def(5, -0.707106769f, -0.408248305f, 0.577350259f, 0.707106769f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskZ | MaskW)
        .dclInput(DxsoUsage::Texcoord, 2, 1, MaskXYZ)
        .dclInput(DxsoUsage::Texcoord, 3, 2, MaskXYZ)
        .dclInput(DxsoUsage::Texcoord, 4, 3, MaskXYZ)
        .dclInput(DxsoUsage::Texcoord, 5, 4, MaskW)
        .dclInput(DxsoUsage::Texcoord, 6, 5, MaskXYZ)
        .dclSampler(0).dclSampler(1).dclSampler(2)
        .op1(DxsoOpcode::Nrm, rd(0, MaskXYZ), v(5))                          // C = normalize(CameraVector)
        .texld(rd(1), v(0, swz(3, 2, 2, 3)), 0)                              // normal map
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(1), c(1, kXXXX), c(1, kYYYY)) // * 2 - 1
        .op1(DxsoOpcode::Nrm, rd(2, MaskXYZ), r(1))                          // N
        .op2(DxsoOpcode::Dp3, rd(0, MaskW), r(2), r(0))                      // dot(N, C)
        .op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(2), r(0, kWWWW))
        .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), r(1), c(1, kXXXX), r(0, kXYZW, kModNeg)) // R
        .op3(DxsoOpcode::Dp2Add, rd(1, MaskX), r(0, swz(1, 2, 2, 3)), c(4), c(4, kZZZZ))
        .op2(DxsoOpcode::Dp3, rd(1, MaskY), r(0), c(5))
        .op2(DxsoOpcode::Dp3, rd(1, MaskZ), r(0, swz(1, 2, 0, 3)), c(5, swz(1, 2, 3, 3)))
        .op2(DxsoOpcode::Max, rd(0, MaskXYZ), r(1), c(1, kZZZZ))
        .op1(DxsoOpcode::Log, rd(1, MaskX), r(0, kXXXX))
        .op1(DxsoOpcode::Log, rd(1, MaskY), r(0, kYYYY))
        .op1(DxsoOpcode::Log, rd(1, MaskZ), r(0, kZZZZ))
        .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(1), c(1, kWWWW))             // * (SpecularPower + 1)
        .op1(DxsoOpcode::Exp, rd(0, MaskX), r(0, kXXXX))                     // ST0
        .op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(0, kXXXX), v(1))             // ST0 * LightMapA
        .texld(rd(3), v(0, swz(3, 2, 2, 3)), 2)                              // specular texture
        .op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(1), r(3))                    // ... * Spec
        .op3(DxsoOpcode::Dp2Add, rd(4, MaskX), r(2, swz(1, 2, 2, 3)), c(4), c(4, kZZZZ))
        .op2(DxsoOpcode::Dp3, rd(4, MaskY), r(2), c(5))
        .op2(DxsoOpcode::Dp3, rd(4, MaskZ), r(2, swz(1, 2, 0, 3)), c(5, swz(1, 2, 3, 3)))
        .op2(DxsoOpcode::Mul, rd(2, MaskXYZ), r(4), r(4))
        .op2(DxsoOpcode::Max, rd(4, MaskXYZ), r(2), c(1, kZZZZ))             // DT
        .op2(DxsoOpcode::Mul, rd(2, MaskXYZ), r(4, kXXXX), v(1))             // DT0 * LightMapA
        .texld(rd(5), v(0, swz(3, 2, 2, 3)), 1)                              // diffuse texture
        .op2(DxsoOpcode::Mul, rd(5, MaskXYZ), r(5), c(2))                    // * UniformVector_1
        .op1(DxsoOpcode::Mov, rd(6, MaskY), c(1, kYYYY))
        .op2(DxsoOpcode::Add, rd(6, MaskXYZ), r(6, kYYYY, kModNeg), c(0, kXYZW, kModNeg)) // 1 - UniformVector_0
        .op2(DxsoOpcode::Mul, rd(5, MaskXYZ), r(5), r(6))                    // D
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(2), r(5), r(1))              // += DT0*LM0*D
        .op2(DxsoOpcode::Mul, rd(2, MaskXYZ), r(4, kYYYY), v(2))
        .op2(DxsoOpcode::Mul, rd(4, MaskXYZ), r(4, kZZZZ), v(3))
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(2), r(5), r(1))              // += DT1*LM1*D
        .op1(DxsoOpcode::Exp, rd(0, MaskX), r(0, kYYYY))
        .op1(DxsoOpcode::Exp, rd(0, MaskY), r(0, kZZZZ))
        .op2(DxsoOpcode::Mul, rd(0, MaskXZW), r(0, kXXXX), v(2, swz(0, 1, 1, 2)))
        .op3(DxsoOpcode::Mad, rd(0, MaskXZW), r(0), r(3, swz(0, 1, 1, 2)), r(1, swz(0, 1, 1, 2))) // += ST1*LM1*Spec
        .op3(DxsoOpcode::Mad, rd(0, MaskXZW), r(4, swz(0, 1, 1, 2)), r(5, swz(0, 1, 1, 2)), r(0))  // += DT2*LM2*D
        .op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(0, kYYYY), v(3))
        .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), r(1), r(3), r(0, swz(0, 2, 3, 3)))                   // += ST2*LM2*Spec
        .op2(DxsoOpcode::Add, rd(0, MaskXYZ), r(0), c(0))                                          // + Emissive
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(5), c(3), r(0))                                      // + D * AmbientColor
        .op1(DxsoOpcode::Mov, oC0(MaskW), v(4, kWWWW));

      DxsoColorTermInputs in;
      in.trackedSamplerMask = 0x7u;
      in.trackedConstRegs.set(0);
      in.trackedConstRegs.set(2);
      in.lightingConstRegs.set(3);
      in.trackLiterals = true;
      const DxsoColorTermResult res = ps.analyze(in);
      expectFlag(res.hasLightConstTerm, true, "hasLightConstTerm", "directional vertex lightmap");
      expectSampler(res, 0, DxsoColorTermRole::Unused, "directional vertex lightmap: normal map");
      expectSampler(res, 1, DxsoColorTermRole::Color, "directional vertex lightmap: diffuse");
      expectSampler(res, 2, DxsoColorTermRole::LightingOnly, "directional vertex lightmap: specular texture");
      expectConst(res, 0, DxsoColorTermRole::Color, "directional vertex lightmap: emissive");
      expectConst(res, 2, DxsoColorTermRole::Color, "directional vertex lightmap: tint");
      // the compile-specific literals never reach the colour unlit
      expectLiteral(res, 7.0f, false, "directional vertex lightmap: specular exponent");
      expectLiteral(res, 0.577350259f, false, "directional vertex lightmap: basis");
      expectLiteral(res, 2.0f, false, "directional vertex lightmap: unpack");
    }

    // SIMPLE_TEXTURE_LIGHTMAP, masked material: LightMapTextures at s1, Texture2D_0 at s0
    // feeding both the clip test (alpha) and the diffuse colour.
    static void test_simpleTextureLightmapMasked() {
      std::cout << "  test_simpleTextureLightmapMasked" << std::endl;
      PsBuilder ps;
      ps.def(1, -0.333299994f, 9.99999975e-005f, 1.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1, MaskZ | MaskW)
        .dclInput(DxsoUsage::Texcoord, 5, 2, MaskW)
        .dclSampler(0).dclSampler(1)
        .texld(rd(0), v(1, swz(3, 2, 2, 3)), 0)
        .op2(DxsoOpcode::Add, rd(1), r(0, kWWWW), c(1, kXXXX))
        .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(0), c(2))
        .texkill(rd(1))
        .op1(DxsoOpcode::Mov, rd(1, MaskZ), c(1, kZZZZ))
        .op2(DxsoOpcode::Add, rd(1, MaskXYZ), r(1, kZZZZ), c(0, kXYZW, kModNeg))
        .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(0), r(1))
        .texld(rd(1), v(0), 1)                                              // LightMapTextures
        .op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(1), c(3))                   // * LightMapScale
        .op2(DxsoOpcode::Max, rd(2, MaskXYZ), r(1, kXYZW, kModAbs), c(1, kYYYY))
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(2), r(0), c(0))
        .op1(DxsoOpcode::Mov, oC0(MaskW), v(2, kWWWW));

      DxsoColorTermInputs in;
      in.trackedSamplerMask = 1u << 0;
      in.lightmapSamplerMask = 1u << 1;
      in.trackedConstRegs.set(0);
      in.trackedConstRegs.set(2);
      const DxsoColorTermResult res = ps.analyze(in);
      expectFlag(res.hasLightConstTerm, false, "hasLightConstTerm", "simple texture lightmap");
      expectSampler(res, 0, DxsoColorTermRole::Color, "simple texture lightmap: diffuse+mask texture");
      expectConst(res, 0, DxsoColorTermRole::Color, "simple texture lightmap: emissive");
      expectConst(res, 2, DxsoColorTermRole::Color, "simple texture lightmap: tint");
    }

    // The same masked material under TEXTURE_LIGHTMAP: constant normal, three coefficient
    // textures, UniformVector_2 is the TwoSidedLightingMask parameter scaling the transfer
    // coefficients - it exists in this compile only and never reaches the ambient term.
    static void test_directionalTextureLightmapWithTwoSidedMask() {
      std::cout << "  test_directionalTextureLightmapWithTwoSidedMask" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.0f, 2.0f, -0.333299994f, 1.00000012f)
        .def(8, 1.00000012f, 1.0f, 9.99999975e-005f, 51.0f)
        .def(9, 1.75f, -1.35000002f, 0.0f, 0.0f)
        .def(10, 0.816496611f, 0.577350259f, 0.0f, 0.333333313f)
        .def(11, -0.707106769f, -0.408248305f, 0.577350259f, 0.707106769f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1, MaskZ | MaskW)
        .dclInput(DxsoUsage::Texcoord, 5, 2, MaskW)
        .dclInput(DxsoUsage::Texcoord, 6, 3, MaskXYZ)
        .dclSampler(0).dclSampler(1).dclSampler(2).dclSampler(3)
        .texld(rd(0), v(1, swz(3, 2, 2, 3)), 3)                             // Texture2D_0
        .op2(DxsoOpcode::Add, rd(1), r(0, kWWWW), c(1, kZZZZ))
        .texkill(rd(1))
        .op1(DxsoOpcode::Nrm, rd(1, MaskXYZ), v(3))                         // C
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(1, kZZZZ), c(1, swz(0, 0, 1, 3)), r(1, kXYZW, kModNeg)) // R with N=(0,0,1)
        .op3(DxsoOpcode::Dp2Add, rd(2, MaskX), r(1, swz(1, 2, 2, 3)), c(10), c(10, kZZZZ))
        .op2(DxsoOpcode::Dp3, rd(2, MaskY), r(1), c(11))
        .op2(DxsoOpcode::Dp3, rd(2, MaskZ), r(1, swz(1, 2, 0, 3)), c(11, swz(1, 2, 3, 3)))
        .op2(DxsoOpcode::Max, rd(1, MaskXYZ), r(2), c(8, kZZZZ))
        .op1(DxsoOpcode::Log, rd(2, MaskX), r(1, kXXXX))
        .op1(DxsoOpcode::Log, rd(2, MaskY), r(1, kYYYY))
        .op1(DxsoOpcode::Log, rd(2, MaskZ), r(1, kZZZZ))
        .op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(2), c(8, kWWWW))
        .op1(DxsoOpcode::Exp, rd(2, MaskX), r(1, kXXXX))
        .op1(DxsoOpcode::Exp, rd(2, MaskY), r(1, kYYYY))
        .op1(DxsoOpcode::Exp, rd(2, MaskZ), r(1, kZZZZ))                    // pow(LMR, 51)
        .op1(DxsoOpcode::Mov, rd(1, MaskXY), c(8))
        .op3(DxsoOpcode::Mad, rd(1, MaskXZW), c(6, swz(0, 1, 1, 2)), r(1, kXXXX, kModNeg), r(1, kYYYY)) // 1 - Mask
        .op2(DxsoOpcode::Mul, rd(2, MaskXYZ), r(2), r(1, swz(0, 2, 3, 3)))  // ST = pow * (1 - Mask)
        .texld(rd(3), v(0), 0)                                              // LightMapTextures[0]
        .op2(DxsoOpcode::Mul, rd(3, MaskXYZ), r(3), c(2))
        .op2(DxsoOpcode::Mul, rd(4, MaskXYZ), r(2, kXXXX), r(3))            // ST0 * LM0
        .op3(DxsoOpcode::Mad, rd(0, MaskW), r(0, kYYYY), c(9, kXXXX), c(9, kYYYY))
        .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(0), c(5))                   // Diffuse = tex * UniformVector_1
        .op2(DxsoOpcode::Add, rd(0, MaskW), r(0, kWWWW), r(0, kWWWW))       // specular colour expression
        .op2(DxsoOpcode::Mul, rd(4, MaskXYZ), r(4), r(0, kWWWW))
        .op1(DxsoOpcode::Mov, rd(2, MaskW), c(1, kWWWW))
        .op2(DxsoOpcode::Mul, rd(5, MaskXYZ), r(2, kWWWW), c(6))            // Mask
        .op3(DxsoOpcode::Mad, rd(1, MaskXZW), r(1), c(10, kWWWW), r(5, swz(0, 1, 1, 2))) // DT = 0.333 * (1 - Mask) + Mask
        .op2(DxsoOpcode::Mul, rd(3, MaskXYZ), r(3), r(1, kXXXX))            // LM0 * DT0
        .op2(DxsoOpcode::Add, rd(5, MaskXYZ), r(1, kYYYY), c(0, kXYZW, kModNeg))
        .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(0), r(5))                   // D
        .op3(DxsoOpcode::Mad, rd(3, MaskXYZ), r(3), r(0), r(4))             // LM0*DT0*D + spec
        .texld(rd(4), v(0), 1)                                              // LightMapTextures[1]
        .op2(DxsoOpcode::Mul, rd(4, MaskXYZ), r(4), c(3))
        .op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(1, kZZZZ), r(4))
        .op2(DxsoOpcode::Mul, rd(2, MaskX | MaskY | MaskW), r(2, kYYYY), r(4, swz(0, 1, 2, 2)))
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(1), r(0), r(3))
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(2, swz(0, 1, 3, 3)), r(0, kWWWW), r(1))
        .texld(rd(3), v(0), 2)                                              // LightMapTextures[2]
        .op2(DxsoOpcode::Mul, rd(2, MaskX | MaskY | MaskW), r(3, swz(0, 1, 2, 2)), c(4, swz(0, 1, 2, 2)))
        .op2(DxsoOpcode::Mul, rd(3, MaskXYZ), r(1, kWWWW), r(2, swz(0, 1, 3, 3)))
        .op2(DxsoOpcode::Mul, rd(2, MaskXYZ), r(2, kZZZZ), r(2, swz(0, 1, 3, 3)))
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(3), r(0), r(1))
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(2), r(0, kWWWW), r(1))
        .op2(DxsoOpcode::Add, rd(1, MaskXYZ), r(1), c(0))                   // + Emissive
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(0), c(7), r(1))               // + D * AmbientColor
        .op1(DxsoOpcode::Mov, oC0(MaskW), v(2, kWWWW));

      DxsoColorTermInputs in;
      in.trackedSamplerMask = 1u << 3;
      in.lightmapSamplerMask = 0x7u;
      in.trackedConstRegs.set(0);
      in.trackedConstRegs.set(5);
      in.trackedConstRegs.set(6);
      in.lightingConstRegs.set(7);
      in.trackLiterals = true;
      const DxsoColorTermResult res = ps.analyze(in);
      expectFlag(res.hasLightConstTerm, true, "hasLightConstTerm", "directional texture lightmap");
      expectSampler(res, 3, DxsoColorTermRole::Color, "directional texture lightmap: diffuse texture");
      expectConst(res, 0, DxsoColorTermRole::Color, "directional texture lightmap: emissive");
      expectConst(res, 5, DxsoColorTermRole::Color, "directional texture lightmap: tint");
      expectConst(res, 6, DxsoColorTermRole::LightingOnly, "directional texture lightmap: two-sided lighting mask");
      expectLiteral(res, 0.333333313f, false, "directional texture lightmap: basis squared");
      expectLiteral(res, 51.0f, false, "directional texture lightmap: specular exponent");
    }

    // An emissive environment reflection tinted by a Fresnel term is view-dependent but not
    // lit, and exists in every permutation: it must stay a colour input.
    static void test_fresnelEmissiveCubeKeepsViewDependentColor() {
      std::cout << "  test_fresnelEmissiveCubeKeepsViewDependentColor" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.0f, 0.0f, 1.0f, 3.0f)
        .def(2, 1.0f, 2.0f, -1.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 6, 1, MaskXYZ)
        .dclSampler(0).dclSampler(1, DxsoTextureType::TextureCube)
        .op1(DxsoOpcode::Nrm, rd(0, MaskXYZ), v(1))                         // C
        .texld(rd(1), v(0), 0)                                              // normal map
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(1), c(2, kYYYY), c(2, kZZZZ))
        .op1(DxsoOpcode::Nrm, rd(2, MaskXYZ), r(1))                         // N
        .op2(DxsoOpcode::Dp3, rd(0, MaskW), r(2), r(0))
        .op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(2), r(0, kWWWW))
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(1), c(2, kYYYY), r(0, kXYZW, kModNeg)) // R
        .texld(rd(3), r(1), 1)                                              // reflection cube
        .op2(DxsoOpcode::Dp3, rd(0, MaskX), c(1), r(0))                     // dot((0,0,1), C)
        .op2(DxsoOpcode::Max, rd(0, MaskX), r(0, kXXXX), c(1, kXXXX))
        .op2(DxsoOpcode::Add, rd(0, MaskX), c(2, kXXXX), r(0, kXXXX, kModNeg)) // 1 - dot
        .op2(DxsoOpcode::Pow, rd(0, MaskX), r(0, kXXXX), c(1, kWWWW))       // ^3
        .op2(DxsoOpcode::Add, rd(0, MaskX), c(2, kXXXX), r(0, kXXXX, kModNeg)) // Fresnel
        .op2(DxsoOpcode::Mul, rd(3, MaskXYZ), r(3), r(0, kXXXX))            // cube * Fresnel
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(3), c(3), c(0));              // * tint + emissive constant

      DxsoColorTermInputs in;
      in.trackedSamplerMask = 0x3u;
      in.trackedConstRegs.set(0);
      in.trackedConstRegs.set(3);
      in.trackLiterals = true;
      const DxsoColorTermResult res = ps.analyze(in);
      // the normal is direction data (no colour terms) but steers the cube lookup, which every
      // compile of the material has
      expectSampler(res, 0, DxsoColorTermRole::Coordinate, "fresnel: normal map");
      expectSampler(res, 1, DxsoColorTermRole::Color, "fresnel: reflection cube");
      expectConst(res, 3, DxsoColorTermRole::Color, "fresnel: tint");
      expectConst(res, 0, DxsoColorTermRole::Color, "fresnel: emissive");
      expectLiteral(res, 3.0f, false, "fresnel: exponent");
    }

    // A texture read only as a lerp blend weight between two diffuse layers still reaches the
    // colour output through those layers, in every permutation of the base pass.
    static void test_blendWeightTextureReachesColor() {
      std::cout << "  test_blendWeightTextureReachesColor" << std::endl;
      PsBuilder ps;
      ps.def(1, 2.0f, -1.0f, 9.99999975e-005f, 1.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1)
        .dclSampler(1).dclSampler(2).dclSampler(3).dclSampler(4).dclSampler(7)
        .texld(rd(1), v(1), 2)
        .texld(rd(2), v(1, swz(3, 2, 2, 3)), 3)
        .texld(rd(3), v(1), 4)                                              // blend weight
        .op3(DxsoOpcode::Lrp, rd(4, MaskXYZ), r(3, kYYYY), r(2), r(1))
        .texld(rd(1), v(1), 1)
        .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(1), r(4))
        .op1(DxsoOpcode::Mov, rd(0, MaskW), c(1, kWWWW))
        .op2(DxsoOpcode::Add, rd(1, MaskXYZ), r(0, kWWWW), c(0, kXYZW, kModNeg))
        .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(0), r(1))
        .texld(rd(1), v(0), 7)                                              // LightMapTextures
        .op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(1), c(2))
        .op2(DxsoOpcode::Max, rd(2, MaskXYZ), r(1, kXYZW, kModAbs), c(1, kZZZZ))
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(2), r(0), c(0));

      DxsoColorTermInputs in;
      in.trackedSamplerMask = (1u << 1) | (1u << 2) | (1u << 3) | (1u << 4);
      in.lightmapSamplerMask = 1u << 7;
      in.trackedConstRegs.set(0);
      const DxsoColorTermResult res = ps.analyze(in);
      for (uint32_t s = 1; s <= 4; s++) {
        expectSampler(res, s, DxsoColorTermRole::Color, "blend weight: layer textures");
      }
    }

    // A material with no textures: its emissive literal reaches the output unlit in every
    // permutation, the lightmap epsilon and the basis never do.
    static void test_texturelessLiterals() {
      std::cout << "  test_texturelessLiterals" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.25f, 0.5f, 0.75f, 9.99999975e-005f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)                                              // LightMapTextures
        .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(0), c(2))
        .op2(DxsoOpcode::Max, rd(0, MaskXYZ), r(0, kXYZW, kModAbs), c(1, kWWWW))
        .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(0), c(3))                   // * UniformVector_1 (diffuse)
        .op2(DxsoOpcode::Add, rd(0, MaskXYZ), r(0), c(1))                   // + emissive literal
        .op2(DxsoOpcode::Add, oC0(MaskXYZ), r(0), c(0));                    // + UniformVector_0

      DxsoColorTermInputs in;
      in.lightmapSamplerMask = 1u << 0;
      in.trackedConstRegs.set(0);
      in.trackedConstRegs.set(3);
      in.trackLiterals = true;
      const DxsoColorTermResult res = ps.analyze(in);
      expectConst(res, 0, DxsoColorTermRole::Color, "textureless: emissive vector");
      expectConst(res, 3, DxsoColorTermRole::Color, "textureless: diffuse vector");
      expectFlag(dxsoColorTermSetHasUnlitTerm(res.constTerms[3]), false, "diffuse vector lit only", "textureless");
      expectLiteral(res, 0.25f, true, "textureless: emissive literal r");
      expectLiteral(res, 0.75f, true, "textureless: emissive literal b");
      expectLiteral(res, 9.99999975e-005f, false, "textureless: eps");
    }

    // FNoLightMapPolicy folds the lightmap to zero: only `Emissive + Diffuse * Ambient` is
    // left, and a specular vector is gone with the transfer. What remains classifies as
    // colour, so a movable mesh shares its lightmapped self's identity.
    static void test_noLightMapPolicySpecularVector() {
      std::cout << "  test_noLightMapPolicySpecularVector" << std::endl;
      PsBuilder ps;
      ps.dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(0), c(1))
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(0), c(3), c(0));

      DxsoColorTermInputs in;
      in.trackedSamplerMask = 1u;
      in.trackedConstRegs.set(0);
      in.trackedConstRegs.set(1);
      in.lightingConstRegs.set(3);
      const DxsoColorTermResult res = ps.analyze(in);
      expectFlag(res.hasLightConstTerm, true, "hasLightConstTerm", "no lightmap policy");
      expectSampler(res, 0, DxsoColorTermRole::Color, "no lightmap policy: diffuse");
      expectConst(res, 1, DxsoColorTermRole::Color, "no lightmap policy: tint");
      expectConst(res, 0, DxsoColorTermRole::Color, "no lightmap policy: emissive");
    }

    // UE3's Desaturation expression dots the colour with luminance weights; the result is
    // still that texture's colour. A dot with a normal is not.
    static void test_desaturationKeepsColor() {
      std::cout << "  test_desaturationKeepsColor" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.3f, 0.59f, 0.11f, 0.8f)
        .def(2, 9.99999975e-005f, 1.0f, 0.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 2, 1, MaskXYZ)
        .dclSampler(0).dclSampler(1)
        .texld(rd(0), v(0), 0)                                              // diffuse
        .op2(DxsoOpcode::Dp3, rd(0, MaskW), r(0), c(1))                     // luminance
        .op3(DxsoOpcode::Lrp, rd(1, MaskXYZ), c(1, kWWWW), r(0, kWWWW), r(0)) // desaturate
        .texld(rd(2), v(0), 1)                                              // a mask-like texture
        .op2(DxsoOpcode::Dp3, rd(2, MaskX), r(2), r(1))                     // dotted with the colour
        .op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(1), r(2, kXXXX))
        .op2(DxsoOpcode::Max, rd(3, MaskXYZ), v(1, kXYZW, kModAbs), c(2, kXXXX))
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(3), r(1), c(0));

      DxsoColorTermInputs in;
      in.trackedSamplerMask = 0x3u;
      in.trackedConstRegs.set(0);
      const DxsoColorTermResult res = ps.analyze(in);
      expectSampler(res, 0, DxsoColorTermRole::Color, "desaturation: diffuse survives the luminance dot");
      expectSampler(res, 1, DxsoColorTermRole::Unused, "desaturation: vector dot with a colour is not colour");
    }

    static void test_rejectsVertexShaderAndPs1x() {
      std::cout << "  test_rejectsVertexShaderAndPs1x" << std::endl;
      DxsoColorTermInputs in;
      const std::vector<uint32_t> vs = { 0xFFFE0300u, kEndToken };
      expectFlag(analyzeDxsoColorTerms(vs.data(), vs.size(), in).analyzed, false, "vs analyzed", "reject vs");
      const std::vector<uint32_t> ps14 = { 0xFFFF0104u, kEndToken };
      expectFlag(analyzeDxsoColorTerms(ps14.data(), ps14.size(), in).analyzed, false, "ps_1_4 analyzed", "reject ps_1_x");
      expectFlag(analyzeDxsoColorTerms(nullptr, 0, in).analyzed, false, "null analyzed", "reject null");
    }
  };

  // --- dump mode -------------------------------------------------------------------------

  std::string toLower(std::string s) {
    for (char& ch : s) {
      ch = char(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
  }

  int dumpShader(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
      std::cerr << "cannot open " << path << std::endl;
      return -1;
    }
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (bytes.size() < 4 || (bytes.size() % 4) != 0) {
      std::cerr << path << ": not a token stream" << std::endl;
      return -1;
    }
    const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytes.data());
    const size_t tokenCount = bytes.size() / 4;
    if ((tokens[0] & 0xffff0000u) != 0xffff0000u) {
      std::cout << path << ": not a pixel shader" << std::endl;
      return 0;
    }

    // The runtime derives the inputs from the CTAB the same way.
    const DxsoProgramInfo programInfo(DxsoProgramTypes::PixelShader, tokens[0] & 0xffu, (tokens[0] >> 8) & 0xffu);
    DxsoDecodeContext decoder(programInfo);
    DxsoCodeIter iter(tokens + 1);
    while (decoder.decodeInstruction(iter)) {
      if (decoder.getCtabInfo().m_size != 0) {
        break;
      }
    }
    const DxsoCtab& ctab = decoder.getCtabInfo();

    DxsoColorTermInputs in;
    in.trackLiterals = true;
    struct Named { std::string name; uint32_t reg; bool sampler; };
    std::vector<Named> named;
    for (const DxsoCtab::Constant& k : ctab.m_constantData) {
      const std::string name = toLower(k.name);
      if (k.registerSet == 3) {
        for (uint32_t s = k.registerIndex; s < k.registerIndex + k.registerCount && s < 16; s++) {
          if (name.find("lightmap") != std::string::npos) {
            in.lightmapSamplerMask |= 1u << s;
          } else if (name.rfind("texture2d_", 0) == 0 || name.rfind("texturecube_", 0) == 0 || name.rfind("texture3d_", 0) == 0) {
            in.trackedSamplerMask |= 1u << s;
            named.push_back({ k.name, s, true });
          }
        }
      } else if (k.registerSet <= 2 && k.registerIndex < kDxsoColorTermMaxConstRegs) {
        if (name.find("ambientcolorandskyfactor") != std::string::npos ||
            name.find("upperskycolor") != std::string::npos ||
            name.find("lowerskycolor") != std::string::npos) {
          for (uint32_t r = k.registerIndex; r < k.registerIndex + k.registerCount && r < kDxsoColorTermMaxConstRegs; r++) {
            in.lightingConstRegs.set(r);
          }
        } else if (name.find("uniformvector_") != std::string::npos) {
          in.trackedConstRegs.set(k.registerIndex);
          named.push_back({ k.name, k.registerIndex, false });
        }
      }
    }

    const DxsoColorTermResult res = analyzeDxsoColorTerms(tokens, tokenCount, in);
    std::cout << "== " << path << "\n   analyzed=" << res.analyzed << " hasLightConstTerm=" << res.hasLightConstTerm
              << " lightmapSamplers=0x" << std::hex << in.lightmapSamplerMask << std::dec << "\n";
    for (const Named& n : named) {
      const DxsoColorTermSet set = n.sampler ? res.samplerTerms[n.reg] : res.constTerms[n.reg];
      const uint8_t reach = n.sampler ? res.samplerReach[n.reg] : res.constReach[n.reg];
      std::cout << "   " << (n.sampler ? "s" : "c") << n.reg << " " << n.name << ": "
                << roleName(classifyDxsoColorTermSource(set, reach, res.hasLightConstTerm)) << " " << describeSet(set)
                << " reach=" << uint32_t(reach) << "\n";
    }
    for (const DxsoColorTermLiteral& lit : res.literals) {
      float f;
      std::memcpy(&f, &lit.bits, sizeof(f));
      std::cout << "   literal " << f << ": " << (dxsoColorTermSetHasUnlitTerm(lit.terms) ? "kept" : "dropped")
                << " " << describeSet(lit.terms) << "\n";
    }
    // Every declared sampler's coordinate dependencies, with CTAB names for the registers -
    // the set the runtime's volatile-register harvest excludes from the constants tier.
    for (const DxsoCtab::Constant& k : ctab.m_constantData) {
      if (k.registerSet != 3) {
        continue;
      }
      for (uint32_t s = k.registerIndex; s < k.registerIndex + k.registerCount && s < kDxsoColorTermMaxSamplers; s++) {
        std::cout << "   coord s" << s << " " << k.name << " samples=" << res.samplerSampleCount[s] << " expr=[";
        if (res.samplerCoordExpr[s] & DxsoCoordExpr_Arith) std::cout << "ARITH ";
        if (res.samplerCoordExpr[s] & DxsoCoordExpr_Offset) std::cout << "OFFSET ";
        if (res.samplerCoordExpr[s] & DxsoCoordExpr_Wrap) std::cout << "WRAP ";
        if (res.samplerCoordExpr[s] & DxsoCoordExpr_UnknownOffset) std::cout << "UNKNOWNOFFSET ";
        std::cout << "] consts={";
        for (uint32_t reg = 0; reg < kDxsoColorTermMaxConstRegs; reg++) {
          if (!res.samplerCoordConstRegs[s].test(reg)) {
            continue;
          }
          std::cout << " c" << reg;
          for (const DxsoCtab::Constant& c : ctab.m_constantData) {
            if (c.registerSet <= 2 && reg >= c.registerIndex && reg < c.registerIndex + c.registerCount) {
              std::cout << "=" << c.name;
              break;
            }
          }
        }
        std::cout << " }\n";
      }
    }
    return 0;
  }

} // anonymous namespace

int main(int argc, char** argv) {
  if (argc > 1) {
    int rc = 0;
    for (int i = 1; i < argc; i++) {
      rc |= dumpShader(argv[i]);
    }
    return rc;
  }

  try {
    ColorTermTestApp::run();
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
