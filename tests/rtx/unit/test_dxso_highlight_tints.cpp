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

// Highlight tint analysis of UE3 base-pass pixel shaders. The hand-assembled cases follow the
// instruction streams fxc emits for Mirror's Edge's Runner Vision materials (LOI_Strength), and
// the shapes a tint proof must not be fooled by.
//
// Dump mode: test_dxso_highlight_tints [--summary] <shader.dxso | directory> [...]  prints the
// tint pairs of dumped pixel shaders (DXVK_SHADER_DUMP_PATH), deriving the inputs from the CTAB
// the same way the runtime does. A directory is searched for *.dxso files; --summary prints only
// the totals.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../../src/dxso/dxso_code.h"
#include "../../../src/dxso/dxso_decoder.h"
#include "../../../src/dxso/dxso_highlight_tints.h"
#include "../../../src/util/log/log.h"

namespace dxvk {
  // Standalone executables own the logger singleton, as the d3d9/dxgi entry points do.
  Logger Logger::s_instance("test_dxso_highlight_tints.log", LogLevel::None);
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
  constexpr uint32_t kRegConstBool = 14u;

  constexpr uint32_t kModNeg = 1u;
  constexpr uint32_t kModAbs = 11u;

  enum WriteMask : uint32_t {
    MaskX = 0x1u, MaskY = 0x2u, MaskZ = 0x4u, MaskW = 0x8u,
    MaskXY = 0x3u, MaskXYZ = 0x7u, MaskAll = 0xFu,
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

    PsBuilder& ifBool(uint32_t boolReg) {
      m_tokens.push_back(opcodeToken(DxsoOpcode::If, 1));
      m_tokens.push_back(src(kRegConstBool, boolReg));
      return *this;
    }

    PsBuilder& endIf() {
      m_tokens.push_back(opcodeToken(DxsoOpcode::EndIf, 0));
      return *this;
    }

    DxsoHighlightResult analyze(const DxsoHighlightInputs& inputs) const {
      std::vector<uint32_t> tokens = m_tokens;
      tokens.push_back(kEndToken);
      return analyzeDxsoHighlightTints(tokens.data(), tokens.size(), inputs);
    }

  private:
    std::vector<uint32_t> m_tokens;
  };

  DxsoHighlightInputs makeInputs(std::initializer_list<uint32_t> scalars,
                                 std::initializer_list<uint32_t> vectors,
                                 std::initializer_list<uint32_t> lightingConsts,
                                 uint32_t lightingSamplerMask) {
    DxsoHighlightInputs in;
    for (const uint32_t reg : scalars) in.scalarRegs.set(reg);
    for (const uint32_t reg : vectors) in.vectorRegs.set(reg);
    for (const uint32_t reg : lightingConsts) in.lightingConstRegs.set(reg);
    in.lightingSamplerMask = lightingSamplerMask;
    return in;
  }

  std::string describePair(const DxsoHighlightPair& pair) {
    std::string out = "c" + std::to_string(pair.scalarReg) + " ->";
    for (uint32_t lane = 0; lane < 3; lane++) {
      if (pair.colorReg[lane] < 0) {
        out += " -";
      } else {
        out += " c" + std::to_string(pair.colorReg[lane]) + "." + "xyzw"[pair.colorComponent[lane]];
      }
    }
    return out + " glow=" + std::to_string(pair.glowCoefficient);
  }

  [[noreturn]] void fail(const char* label, const std::string& why) {
    std::cerr << label << ": " << why << std::endl;
    throw std::runtime_error(label);
  }

  void expectAnalyzed(const DxsoHighlightResult& res, const char* label) {
    if (!res.analyzed) {
      fail(label, std::string("not analyzed: ") + dxsoHighlightFailureName(res.failure));
    }
  }

  void expectPairCount(const DxsoHighlightResult& res, size_t expected, const char* label) {
    if (res.pairs.size() != expected) {
      std::string found;
      for (const DxsoHighlightPair& p : res.pairs) {
        found += " [" + describePair(p) + "]";
      }
      fail(label, "found " + std::to_string(res.pairs.size()) + " pairs" + found + ", expected " + std::to_string(expected));
    }
  }

  const DxsoHighlightPair& findPair(const DxsoHighlightResult& res, uint32_t scalarReg, const char* label) {
    for (const DxsoHighlightPair& p : res.pairs) {
      if (p.scalarReg == scalarReg) {
        return p;
      }
    }
    fail(label, "no pair on c" + std::to_string(scalarReg));
  }

  // Every channel lerps towards the matching component of `colorReg`.
  void expectTint(const DxsoHighlightPair& pair, uint32_t colorReg, const char* label) {
    for (uint32_t lane = 0; lane < 3; lane++) {
      if (pair.colorReg[lane] != int32_t(colorReg) || pair.colorComponent[lane] != lane) {
        fail(label, "pair " + describePair(pair) + ", expected every channel to lerp towards c" + std::to_string(colorReg));
      }
    }
  }

  void expectGlow(const DxsoHighlightPair& pair, float expected, const char* label) {
    if (std::abs(pair.glowCoefficient - expected) > 1.0e-4f) {
      fail(label, "pair " + describePair(pair) + ", expected glow " + std::to_string(expected));
    }
  }

  void expectUnproven(const DxsoHighlightResult& res, uint32_t scalarReg, const char* label) {
    for (const uint32_t reg : res.unprovenScalarRegs) {
      if (reg == scalarReg) {
        return;
      }
    }
    fail(label, "c" + std::to_string(scalarReg) + " not reported as an unproven scalar");
  }

  class HighlightTintTestApp {
  public:
    static void run() {
      std::cout << "Running DXSO highlight tint analysis tests..." << std::endl;
      test_simpleLightmapRunnerVision();
      test_directionalSkyLitRunnerVision();
      test_maskedPartialWriteAfterLerp();
      test_lrpForm();
      test_brightnessScalarBeforeLerp();
      test_twoTintScalars();
      test_texturelessTint();
      test_unscaledGlowIsNotAFingerprint();
      test_reflectionIntensityIsNotATint();
      test_maskWeightedBlendIsNotATint();
      test_replaceLerpIsNotATint();
      test_layerBlendUnderEmissiveFactorIsNotATint();
      test_rejects();
      std::cout << "All DXSO highlight tint analysis tests passed." << std::endl;
    }

  private:
    // SIMPLE_TEXTURE_LIGHTMAP compile of the standard Runner Vision network:
    //   D' = lerp(Tex * UniformVector_1, Tex * UniformVector_1 * UniformVector_2, UniformScalar_0)
    //   out = LightMap * D' * (1 - UniformVector_0) + UniformVector_0 + 0.1 * UniformScalar_0 * D'
    static void test_simpleLightmapRunnerVision() {
      std::cout << "  test_simpleLightmapRunnerVision" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.100000001f, 9.99999975e-005f, 1.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 5, 2, MaskW)
        .dclSampler(0).dclSampler(1)
        .texld(rd(0), v(0), 1)                                               // LightMapTextures
        .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(0), c(5))                     // * LightMapScale
        .op2(DxsoOpcode::Max, rd(1, MaskXYZ), r(0, kXYZW, kModAbs), c(1, kYYYY))
        .texld(rd(0), v(1), 0)                                               // Texture2D_1
        .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(0), c(2))                     // * UniformVector_1
        .op3(DxsoOpcode::Mad, rd(2, MaskXYZ), c(3), r(0), r(0, kXYZW, kModNeg)) // x * V - x
        .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), c(4, kXXXX), r(2), r(0))        // lerp
        .op2(DxsoOpcode::Mul, rd(2, MaskXYZ), r(0), c(4, kXXXX))              // S * D'
        .op1(DxsoOpcode::Mov, rd(3, MaskX | MaskZ), c(1))
        .op3(DxsoOpcode::Mad, rd(2, MaskXYZ), r(2), r(3, kXXXX), c(0))        // 0.1 * S * D' + emissive
        .op2(DxsoOpcode::Add, rd(3, MaskXYZ), r(3, kZZZZ), c(0, kXYZW, kModNeg)) // 1 - UniformVector_0
        .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(0), r(3))
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(1), r(0), r(2))
        .op1(DxsoOpcode::Mov, oC0(MaskW), v(2, kWWWW));

      const DxsoHighlightResult res = ps.analyze(makeInputs({ 4 }, { 0, 2, 3 }, { 5 }, 1u << 1));
      const char* label = "simple lightmap Runner Vision";
      expectAnalyzed(res, label);
      expectPairCount(res, 1, label);
      const DxsoHighlightPair& pair = findPair(res, 4, label);
      expectTint(pair, 3, label);
      expectGlow(pair, 0.1f, label);
    }

    // The same network under the directional lightmap policy with sky lighting: the tinted
    // diffuse reaches the output through the lightmap, both sky colours and SkyFactor, and the
    // sky weights are derived from the normal map.
    static void test_directionalSkyLitRunnerVision() {
      std::cout << "  test_directionalSkyLitRunnerVision" << std::endl;
      PsBuilder ps;
      ps.def(1, 2.0f, -1.0f, 0.100000001f, 9.99999975e-005f)
        .def(9, 0.5f, -0.5f, 0.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 5, 2, MaskW)
        .dclInput(DxsoUsage::Texcoord, 7, 3, MaskXYZ)
        .dclSampler(0).dclSampler(1).dclSampler(2)
        .texld(rd(0), v(1), 0)                                                   // normal map
        .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), r(0), c(1, kXXXX), c(1, kYYYY))
        .op1(DxsoOpcode::Nrm, rd(1, MaskXYZ), r(0))
        .op1(DxsoOpcode::Nrm, rd(0, MaskXYZ), v(3))                              // sky vector
        .op2(DxsoOpcode::Dp3, rd(0, MaskX), r(0), r(1))
        .op3(DxsoOpcode::Mad, rd(0, MaskXY), r(0, kXXXX), c(9), c(9, kXXXX))
        .op2(DxsoOpcode::Mul, rd(0, MaskXY), r(0), r(0))                         // sky weights
        .texld(rd(1), v(1), 1)                                                   // Texture2D_1
        .op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(1), c(2))
        .op3(DxsoOpcode::Mad, rd(2, MaskXYZ), c(3), r(1), r(1, kXYZW, kModNeg))
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), c(4, kXXXX), r(2), r(1))           // D'
        .op1(DxsoOpcode::Mov, rd(2, MaskY | MaskZ), c(1))
        .op2(DxsoOpcode::Add, rd(2, MaskX | MaskY | MaskW), r(2, kYYYY, kModNeg), c(0, swz(0, 1, 2, 2), kModNeg))
        .op2(DxsoOpcode::Mul, rd(2, MaskX | MaskY | MaskW), r(1, swz(0, 1, 2, 2)), r(2)) // D
        .op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(1), c(4, kXXXX))                 // S * D'
        .op2(DxsoOpcode::Mul, rd(0, MaskY | MaskZ | MaskW), r(0, kYYYY), r(2, swz(0, 0, 1, 3)))
        .op2(DxsoOpcode::Mul, rd(3, MaskXYZ), r(0, kXXXX), r(2, swz(0, 1, 3, 3)))
        .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(0, swz(1, 2, 3, 3)), c(6))       // * LowerSkyColor
        .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), r(3), c(5), r(0))                  // + * UpperSkyColor
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(1), r(2, kZZZZ), c(0))           // glow + emissive
        .texld(rd(3), v(0), 2)                                                   // LightMapTextures
        .op2(DxsoOpcode::Mul, rd(3, MaskXYZ), r(3), c(8))
        .op2(DxsoOpcode::Max, rd(4, MaskXYZ), r(3, kXYZW, kModAbs), c(1, kWWWW))
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(4), r(2, swz(0, 1, 3, 3)), r(1))
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(0), c(7, kWWWW), r(1))             // sky * SkyFactor
        .op1(DxsoOpcode::Mov, oC0(MaskW), v(2, kWWWW));

      const DxsoHighlightResult res = ps.analyze(makeInputs({ 4 }, { 0, 2, 3 }, { 5, 6, 7, 8 }, 1u << 2));
      const char* label = "directional sky-lit Runner Vision";
      expectAnalyzed(res, label);
      expectPairCount(res, 1, label);
      const DxsoHighlightPair& pair = findPair(res, 4, label);
      expectTint(pair, 3, label);
      expectGlow(pair, 0.1f, label);
    }

    // A masked master material with the tint but no glow. After the lerp lands in r0.xyz, fxc
    // packs a sky-vector dot product into r0.w, and a register-granular proof loses the lerp to
    // that write.
    static void test_maskedPartialWriteAfterLerp() {
      std::cout << "  test_maskedPartialWriteAfterLerp" << std::endl;
      PsBuilder ps;
      ps.def(1, 2.0f, -1.0f, -0.333299994f, 9.99999975e-005f)
        .def(8, 0.5f, -0.5f, 0.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1, MaskZ | MaskW)
        .dclInput(DxsoUsage::Texcoord, 5, 2, MaskW)
        .dclInput(DxsoUsage::Texcoord, 7, 3, MaskXYZ)
        .dclSampler(0).dclSampler(1).dclSampler(2)
        .texld(rd(0), v(1, swz(3, 2, 2, 3)), 1)                                  // Texture2D_1
        .op2(DxsoOpcode::Add, rd(1), r(0, kWWWW), c(1, kZZZZ))
        .texkill(rd(1))
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), c(2), r(0), r(0, kXYZW, kModNeg))
        .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), c(3, kXXXX), r(1), r(0))           // lerp into r0.xyz
        .op1(DxsoOpcode::Mov, rd(1, MaskY), c(1, kYYYY))
        .op2(DxsoOpcode::Add, rd(1, MaskXYZ), r(1, kYYYY, kModNeg), c(0, kXYZW, kModNeg))
        .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(0), r(1))
        .texld(rd(1), v(1, swz(3, 2, 2, 3)), 0)                                  // normal map
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(1), c(1, kXXXX), c(1, kYYYY))
        .op1(DxsoOpcode::Nrm, rd(2, MaskXYZ), r(1))
        .op1(DxsoOpcode::Nrm, rd(1, MaskXYZ), v(3))
        .op2(DxsoOpcode::Dp3, rd(0, MaskW), r(1), r(2))                          // r0.w
        .op3(DxsoOpcode::Mad, rd(1, MaskXY), r(0, kWWWW), c(8), c(8, kXXXX))
        .op2(DxsoOpcode::Mul, rd(1, MaskXY), r(1), r(1))
        .op2(DxsoOpcode::Mul, rd(1, MaskY | MaskZ | MaskW), r(0, swz(0, 0, 1, 2)), r(1, kYYYY))
        .op2(DxsoOpcode::Mul, rd(2, MaskXYZ), r(0), r(1, kXXXX))
        .op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(1, swz(1, 2, 3, 3)), c(5))
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(2), c(4), r(1))
        .texld(rd(2), v(0), 2)                                                   // LightMapTextures
        .op2(DxsoOpcode::Mul, rd(2, MaskXYZ), r(2), c(7))
        .op2(DxsoOpcode::Max, rd(3, MaskXYZ), r(2, kXYZW, kModAbs), c(1, kWWWW))
        .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), r(3), r(0), c(0))
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(1), c(6, kWWWW), r(0))
        .op1(DxsoOpcode::Mov, oC0(MaskW), v(2, kWWWW));

      const DxsoHighlightResult res = ps.analyze(makeInputs({ 3 }, { 0, 2 }, { 4, 5, 6, 7 }, 1u << 2));
      const char* label = "masked partial write after lerp";
      expectAnalyzed(res, label);
      expectPairCount(res, 1, label);
      const DxsoHighlightPair& pair = findPair(res, 3, label);
      expectTint(pair, 2, label);
      expectGlow(pair, 0.0f, label);
    }

    // lerp(x, x * V, S) compiled to lrp.
    static void test_lrpForm() {
      std::cout << "  test_lrpForm" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.0f, 9.99999975e-005f, 1.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1, MaskXY)
        .dclSampler(0).dclSampler(1)
        .texld(rd(0), v(0), 0)
        .op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(0), c(3))
        .op3(DxsoOpcode::Lrp, rd(2, MaskXYZ), c(4, kXXXX), r(1), r(0))
        .texld(rd(3), v(1), 1)
        .op2(DxsoOpcode::Mul, rd(3, MaskXYZ), r(3), c(5))
        .op2(DxsoOpcode::Max, rd(3, MaskXYZ), r(3, kXYZW, kModAbs), c(1, kYYYY))
        .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(3), r(2))
        .op1(DxsoOpcode::Mov, oC0(MaskW), c(1, kZZZZ));

      const DxsoHighlightResult res = ps.analyze(makeInputs({ 4 }, { 3 }, { 5 }, 1u << 1));
      const char* label = "lrp form";
      expectAnalyzed(res, label);
      expectPairCount(res, 1, label);
      expectTint(findPair(res, 4, label), 3, label);
    }

    // A brightness scalar scales the colour before the lerp, so the strength is UniformScalar_1.
    // The brightness scalar reaches the colour output but tints nothing.
    static void test_brightnessScalarBeforeLerp() {
      std::cout << "  test_brightnessScalarBeforeLerp" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.100000001f, 9.99999975e-005f, 1.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 5, 2, MaskW)
        .dclSampler(0).dclSampler(1)
        .texld(rd(0), v(1), 0)
        .op2(DxsoOpcode::Mul, rd(4, MaskXYZ), r(0), c(6, kXXXX))                 // * UniformScalar_0
        .op3(DxsoOpcode::Mad, rd(7, MaskXYZ), c(5), r(4), r(4, kXYZW, kModNeg))
        .op3(DxsoOpcode::Mad, rd(4, MaskXYZ), c(7, kXXXX), r(7), r(4))           // lerp on UniformScalar_1
        .op1(DxsoOpcode::Mov, rd(7, MaskX | MaskZ), c(1))
        .op2(DxsoOpcode::Add, rd(6, MaskXYZ), r(7, kZZZZ), c(0, kXYZW, kModNeg))
        .op2(DxsoOpcode::Mul, rd(6, MaskXYZ), r(4), r(6))
        .op2(DxsoOpcode::Mul, rd(4, MaskXYZ), r(4), c(7, kXXXX))
        .op3(DxsoOpcode::Mad, rd(2, MaskXYZ), r(4), r(7, kXXXX), c(0))
        .texld(rd(3), v(0), 1)
        .op2(DxsoOpcode::Mul, rd(3, MaskXYZ), r(3), c(8))
        .op2(DxsoOpcode::Max, rd(3, MaskXYZ), r(3, kXYZW, kModAbs), c(1, kYYYY))
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(3), r(6), r(2))
        .op1(DxsoOpcode::Mov, oC0(MaskW), v(2, kWWWW));

      const DxsoHighlightResult res = ps.analyze(makeInputs({ 6, 7 }, { 0, 5 }, { 8 }, 1u << 1));
      const char* label = "brightness scalar before lerp";
      expectAnalyzed(res, label);
      expectPairCount(res, 1, label);
      const DxsoHighlightPair& pair = findPair(res, 7, label);
      expectTint(pair, 5, label);
      expectGlow(pair, 0.1f, label);
      expectUnproven(res, 6, label);
    }

    // Two nested tint lerps, each on its own scalar: both are pairs.
    static void test_twoTintScalars() {
      std::cout << "  test_twoTintScalars" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.0f, 9.99999975e-005f, 1.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1, MaskXY)
        .dclSampler(0).dclSampler(1)
        .texld(rd(0), v(0), 0)
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), c(3), r(0), r(0, kXYZW, kModNeg))
        .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), c(4, kXXXX), r(1), r(0))
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(0), c(5), r(0, kXYZW, kModNeg))  // other operand order
        .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), r(1), c(6, kXXXX), r(0))           // strength second
        .texld(rd(2), v(1), 1)
        .op2(DxsoOpcode::Mul, rd(2, MaskXYZ), r(2), c(7))
        .op2(DxsoOpcode::Max, rd(2, MaskXYZ), r(2, kXYZW, kModAbs), c(1, kYYYY))
        .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(2), r(0))
        .op1(DxsoOpcode::Mov, oC0(MaskW), c(1, kZZZZ));

      const DxsoHighlightResult res = ps.analyze(makeInputs({ 4, 6 }, { 3, 5 }, { 7 }, 1u << 1));
      const char* label = "two tint scalars";
      expectAnalyzed(res, label);
      expectPairCount(res, 2, label);
      expectTint(findPair(res, 4, label), 3, label);
      expectTint(findPair(res, 6, label), 5, label);
    }

    // A constant-colour material: the tinted colour is a UniformVector rather than a texture.
    static void test_texturelessTint() {
      std::cout << "  test_texturelessTint" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.100000001f, 9.99999975e-005f, 1.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .op1(DxsoOpcode::Mov, rd(0, MaskXYZ), c(2))
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), c(3), r(0), r(0, kXYZW, kModNeg))
        .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), c(4, kXXXX), r(1), r(0))
        .op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(0), c(4, kXXXX))
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(1), c(1, kXXXX), c(0))
        .texld(rd(2), v(0), 0)                                                   // LightMapTextures
        .op2(DxsoOpcode::Mul, rd(2, MaskXYZ), r(2), c(5))
        .op2(DxsoOpcode::Max, rd(2, MaskXYZ), r(2, kXYZW, kModAbs), c(1, kYYYY))
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(2), r(0), r(1))
        .op1(DxsoOpcode::Mov, oC0(MaskW), c(1, kZZZZ));

      const DxsoHighlightResult res = ps.analyze(makeInputs({ 4 }, { 0, 2, 3 }, { 5 }, 1u << 0));
      const char* label = "textureless tint";
      expectAnalyzed(res, label);
      expectPairCount(res, 1, label);
      const DxsoHighlightPair& pair = findPair(res, 4, label);
      expectTint(pair, 3, label);
      expectGlow(pair, 0.1f, label);
    }

    // Some master materials always emit 0.1 * D' rather than 0.1 * S * D'. That glow shifts from
    // the surface's own colour towards the tint as S rises instead of growing with it, so there is
    // no strength-scaled glow to reproduce and nothing that fingerprints the pair.
    static void test_unscaledGlowIsNotAFingerprint() {
      std::cout << "  test_unscaledGlowIsNotAFingerprint" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.100000001f, 9.99999975e-005f, 1.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1, MaskXY)
        .dclSampler(0).dclSampler(1)
        .texld(rd(0), v(1), 0)
        .op3(DxsoOpcode::Mad, rd(2, MaskXYZ), c(3), r(0), r(0, kXYZW, kModNeg))
        .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), c(4, kXXXX), r(2), r(0))
        .op3(DxsoOpcode::Mad, rd(2, MaskXYZ), r(0), c(1, kXXXX), c(0))           // 0.1 * D' + emissive
        .texld(rd(3), v(0), 1)
        .op2(DxsoOpcode::Mul, rd(3, MaskXYZ), r(3), c(5))
        .op2(DxsoOpcode::Max, rd(3, MaskXYZ), r(3, kXYZW, kModAbs), c(1, kYYYY))
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(3), r(0), r(2))
        .op1(DxsoOpcode::Mov, oC0(MaskW), c(1, kZZZZ));

      const DxsoHighlightResult res = ps.analyze(makeInputs({ 4 }, { 0, 3 }, { 5 }, 1u << 1));
      const char* label = "unscaled glow";
      expectAnalyzed(res, label);
      expectPairCount(res, 1, label);
      const DxsoHighlightPair& pair = findPair(res, 4, label);
      expectTint(pair, 3, label);
      expectGlow(pair, 0.0f, label);
    }

    // cube * mask + detail * S: a scalar scaling an additive layer is an intensity, not a tint.
    static void test_reflectionIntensityIsNotATint() {
      std::cout << "  test_reflectionIntensityIsNotATint" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.0f, 9.99999975e-005f, 1.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 6, 1, MaskXYZ)
        .dclSampler(0).dclSampler(1).dclSampler(2, DxsoTextureType::TextureCube)
        .texld(rd(0), v(1), 2)
        .texld(rd(1), v(0), 0)
        .texld(rd(2), v(0), 1)
        .op2(DxsoOpcode::Mul, rd(2, MaskXYZ), r(2), c(4, kXXXX))
        .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), r(0), r(1), r(2))
        .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(0), c(3))
        .op1(DxsoOpcode::Mov, oC0(MaskW), c(1, kZZZZ));

      const DxsoHighlightResult res = ps.analyze(makeInputs({ 4 }, { 3 }, {}, 0));
      const char* label = "reflection intensity";
      expectAnalyzed(res, label);
      expectPairCount(res, 0, label);
      expectUnproven(res, 4, label);
    }

    // lerp(x, x * V, mask) with a texture as the weight, scaled by a brightness scalar.
    static void test_maskWeightedBlendIsNotATint() {
      std::cout << "  test_maskWeightedBlendIsNotATint" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.0f, 9.99999975e-005f, 1.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0).dclSampler(1)
        .texld(rd(0), v(0), 0)
        .texld(rd(1), v(0), 1)
        .op3(DxsoOpcode::Mad, rd(2, MaskXYZ), c(3), r(0), r(0, kXYZW, kModNeg))
        .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), r(1, kZZZZ), r(2), r(0))
        .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(0), c(4, kXXXX))
        .op1(DxsoOpcode::Mov, oC0(MaskW), c(1, kZZZZ));

      const DxsoHighlightResult res = ps.analyze(makeInputs({ 4 }, { 3 }, {}, 0));
      const char* label = "mask-weighted blend";
      expectAnalyzed(res, label);
      expectPairCount(res, 0, label);
      expectUnproven(res, 4, label);
    }

    // lerp(x, V, S) replaces the colour rather than tinting it.
    static void test_replaceLerpIsNotATint() {
      std::cout << "  test_replaceLerpIsNotATint" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.0f, 9.99999975e-005f, 1.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1, MaskXY)
        .dclSampler(0).dclSampler(1)
        .texld(rd(0), v(0), 0)
        .op2(DxsoOpcode::Add, rd(1, MaskXYZ), c(3), r(0, kXYZW, kModNeg))
        .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), c(4, kXXXX), r(1), r(0))
        .texld(rd(2), v(1), 1)
        .op2(DxsoOpcode::Mul, rd(2, MaskXYZ), r(2), c(5))
        .op2(DxsoOpcode::Max, rd(2, MaskXYZ), r(2, kXYZW, kModAbs), c(1, kYYYY))
        .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(2), r(0))
        .op1(DxsoOpcode::Mov, oC0(MaskW), c(1, kZZZZ));

      const DxsoHighlightResult res = ps.analyze(makeInputs({ 4 }, { 3 }, { 5 }, 1u << 1));
      const char* label = "replace lerp";
      expectAnalyzed(res, label);
      expectPairCount(res, 0, label);
      expectUnproven(res, 4, label);
    }

    // lerp(A, B, S) * (1 - UniformVector_0) * AmbientColor + UniformVector_0: a layer blend. The
    // emissive factor puts -S * A and +S * A * V0 into the output, which on their own read as a
    // tint of A towards V0; the A * V0 monomial fading with no colour of its own says otherwise.
    static void test_layerBlendUnderEmissiveFactorIsNotATint() {
      std::cout << "  test_layerBlendUnderEmissiveFactorIsNotATint" << std::endl;
      PsBuilder ps;
      ps.def(1, 2.0f, -1.0f, 0.5f, 9.99999975e-005f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 5, 1, MaskW)
        .dclSampler(0).dclSampler(1)
        .texld(rd(0), v(0), 0)
        .texld(rd(1), v(0), 1)
        .op3(DxsoOpcode::Lrp, rd(2, MaskXYZ), c(2, kXXXX), r(1), r(0))
        .op1(DxsoOpcode::Mov, rd(0, MaskY), c(1, kYYYY))
        .op2(DxsoOpcode::Add, rd(0, MaskXYZ), r(0, kYYYY, kModNeg), c(0, kXYZW, kModNeg))
        .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(2), r(0))
        .op1(DxsoOpcode::Mov, rd(1, MaskXYZ), c(0))
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(0), c(3), r(1))
        .op1(DxsoOpcode::Mov, oC0(MaskW), v(1, kWWWW));

      const DxsoHighlightResult res = ps.analyze(makeInputs({ 2 }, { 0 }, { 3 }, 0));
      const char* label = "layer blend under emissive factor";
      expectAnalyzed(res, label);
      expectPairCount(res, 0, label);
      expectUnproven(res, 2, label);
    }

    static void test_rejects() {
      std::cout << "  test_rejects" << std::endl;
      const DxsoHighlightInputs in = makeInputs({ 4 }, { 3 }, {}, 0);

      const std::vector<uint32_t> vs = { 0xFFFE0300u, kEndToken };
      if (analyzeDxsoHighlightTints(vs.data(), vs.size(), in).failure != DxsoHighlightFailure::NotPixelShader) {
        fail("reject vs", "vertex shader was analyzed");
      }
      const std::vector<uint32_t> ps14 = { 0xFFFF0104u, kEndToken };
      if (analyzeDxsoHighlightTints(ps14.data(), ps14.size(), in).failure != DxsoHighlightFailure::NotPixelShader) {
        fail("reject ps_1_x", "ps_1_4 was analyzed");
      }
      if (analyzeDxsoHighlightTints(nullptr, 0, in).analyzed) {
        fail("reject null", "null bytecode was analyzed");
      }

      PsBuilder branch;
      branch.dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .ifBool(0)
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), c(3), r(0), r(0, kXYZW, kModNeg))
        .endIf()
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), c(4, kXXXX), r(1), r(0));
      const DxsoHighlightResult branchRes = branch.analyze(in);
      if (branchRes.analyzed || branchRes.failure != DxsoHighlightFailure::FlowControl) {
        fail("reject flow control", std::string("failure = ") + dxsoHighlightFailureName(branchRes.failure));
      }

      PsBuilder noColor;
      noColor.dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op1(DxsoOpcode::Mov, oC0(MaskW), r(0, kWWWW));
      const DxsoHighlightResult noColorRes = noColor.analyze(in);
      if (noColorRes.failure != DxsoHighlightFailure::NoColorOutput) {
        fail("reject no colour output", std::string("failure = ") + dxsoHighlightFailureName(noColorRes.failure));
      }
    }
  };

  // --- dump mode -------------------------------------------------------------------------

  struct DumpTotals {
    uint32_t shaders = 0;
    uint32_t pixelShaders = 0;
    uint32_t analyzed = 0;
    uint32_t withPairs = 0;
    uint32_t withGlow = 0;
    uint32_t withUnproven = 0;
    std::map<std::string, uint32_t> failures;
    double totalMs = 0.0;
    std::vector<std::pair<double, std::string>> slowest;
  };

  int dumpShader(const std::filesystem::path& path, bool summaryOnly, DumpTotals& totals) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
      std::cerr << "cannot open " << path.string() << std::endl;
      return -1;
    }
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    totals.shaders++;
    if (bytes.size() < 4 || (bytes.size() % 4) != 0) {
      std::cerr << path.string() << ": not a token stream" << std::endl;
      return -1;
    }
    const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytes.data());
    const size_t tokenCount = bytes.size() / 4;
    if ((tokens[0] & 0xffff0000u) != 0xffff0000u) {
      return 0;
    }
    totals.pixelShaders++;

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
    const DxsoHighlightInputs in = dxsoHighlightInputsFromUe3Ctab(ctab);
    const auto start = std::chrono::steady_clock::now();
    const DxsoHighlightResult res = analyzeDxsoHighlightTints(tokens, tokenCount, in);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    totals.totalMs += ms;
    totals.slowest.emplace_back(ms, path.filename().string() + " (" + dxsoHighlightFailureName(res.failure) + ")");

    constexpr uint16_t kRegisterSetFloat4 = 2u;
    auto nameOf = [&](int32_t reg) -> std::string {
      for (const DxsoCtab::Constant& k : ctab.m_constantData) {
        if (k.registerSet == kRegisterSetFloat4 && reg >= int32_t(k.registerIndex) && reg < int32_t(k.registerIndex + k.registerCount)) {
          return k.name;
        }
      }
      return "?";
    };

    if (!res.analyzed) {
      totals.failures[dxsoHighlightFailureName(res.failure)]++;
      if (!summaryOnly) {
        std::cout << "== " << path.string() << ": not analyzed (" << dxsoHighlightFailureName(res.failure) << ")\n";
      }
      return 0;
    }
    totals.analyzed++;
    totals.withPairs += res.pairs.empty() ? 0u : 1u;
    bool glow = false;
    for (const DxsoHighlightPair& p : res.pairs) {
      glow |= p.glowCoefficient > 0.0f;
    }
    totals.withGlow += glow ? 1u : 0u;
    totals.withUnproven += res.unprovenScalarRegs.empty() ? 0u : 1u;

    if (summaryOnly || (res.pairs.empty() && res.unprovenScalarRegs.empty())) {
      return 0;
    }
    std::cout << "== " << path.string() << "\n";
    for (const DxsoHighlightPair& p : res.pairs) {
      std::cout << "   tint " << nameOf(int32_t(p.scalarReg)) << " (" << describePair(p) << ") colour";
      for (uint32_t lane = 0; lane < 3; lane++) {
        std::cout << " " << (p.colorReg[lane] < 0 ? std::string("-") : nameOf(p.colorReg[lane]));
      }
      std::cout << "\n";
    }
    for (const uint32_t reg : res.unprovenScalarRegs) {
      std::cout << "   unproven " << nameOf(int32_t(reg)) << " (c" << reg << ")\n";
    }
    return 0;
  }

} // anonymous namespace

int main(int argc, char** argv) {
  if (argc > 1) {
    bool summaryOnly = false;
    DumpTotals totals;
    int rc = 0;
    for (int i = 1; i < argc; i++) {
      const std::string arg = argv[i];
      if (arg == "--summary") {
        summaryOnly = true;
        continue;
      }
      const std::filesystem::path path(arg);
      if (std::filesystem::is_directory(path)) {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
          if (entry.is_regular_file() && entry.path().extension() == ".dxso") {
            rc |= dumpShader(entry.path(), summaryOnly, totals);
          }
        }
      } else {
        rc |= dumpShader(path, summaryOnly, totals);
      }
    }
    std::cout << "shaders=" << totals.shaders << " pixelShaders=" << totals.pixelShaders
              << " analyzed=" << totals.analyzed << " withTintPairs=" << totals.withPairs
              << " withGlow=" << totals.withGlow << " withUnprovenScalars=" << totals.withUnproven << "\n";
    for (const auto& [reason, count] : totals.failures) {
      std::cout << "  not analyzed (" << reason << "): " << count << "\n";
    }
    if (totals.pixelShaders != 0) {
      std::sort(totals.slowest.begin(), totals.slowest.end(), std::greater<>());
      std::cout << "analysis time: total " << totals.totalMs << " ms, mean "
                << totals.totalMs / totals.pixelShaders << " ms, slowest:\n";
      for (size_t i = 0; i < totals.slowest.size() && i < 5; i++) {
        std::cout << "  " << totals.slowest[i].first << " ms " << totals.slowest[i].second << "\n";
      }
    }
    return rc;
  }

  try {
    HighlightTintTestApp::run();
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
