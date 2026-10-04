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

// Opacity-driven fade analysis of UE3 base-pass pixel shaders. The hand-assembled cases follow
// the output formulas of BasePassPixelShader.usf per blend mode - translucent (Color * Fog.a +
// Fog.rgb, Opacity), additive (Color * Fog.a * Opacity, 0) and modulate (Color, Opacity) - with
// the particle colour interpolant and material parameters that fade them, and the shapes the
// analysis must not be fooled by. Two are transcribed from Mirror's Edge's shader cache.
//
// Dump mode: test_dxso_material_fades [--summary] [--particle-color N] <shader.dxso | directory> [...]
// prints the fade candidates of dumped pixel shaders (DXVK_SHADER_DUMP_PATH), with the inputs
// derived from the CTAB as the runtime does. --particle-color N analyses the input declared
// TEXCOORDN as the particle colour (1 for UE3 sprites, 3 for SubUV). A directory is searched for
// *.dxso files; --summary prints only the totals.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../../src/dxso/dxso_code.h"
#include "../../../src/dxso/dxso_decoder.h"
#include "../../../src/dxso/dxso_material_fades.h"
#include "../../../src/util/log/log.h"

#include "dxso_test_assembler.h"

namespace dxvk {
  // Standalone executables own the logger singleton, as the d3d9/dxgi entry points do.
  Logger Logger::s_instance("test_dxso_material_fades.log", LogLevel::None);
}

using namespace dxvk;
using namespace dxvk::dxso_test;

namespace {

  class PsBuilder : public DxsoTestShader {
  public:
    PsBuilder() : DxsoTestShader(kPs30Header) { }

    DxsoMaterialFadeResult analyze(const DxsoHighlightInputs& inputs, int32_t particleColorInputRegister = -1) const {
      const std::vector<uint32_t> bytecode = tokens();
      return analyzeDxsoMaterialFades(bytecode.data(), bytecode.size(), inputs, particleColorInputRegister);
    }
  };

  DxsoHighlightInputs makeInputs(std::initializer_list<uint32_t> scalars,
                                 std::initializer_list<uint32_t> vectors) {
    DxsoHighlightInputs in;
    for (const uint32_t reg : scalars) in.scalarRegs.set(reg);
    for (const uint32_t reg : vectors) in.vectorRegs.set(reg);
    return in;
  }

  // The float constants a draw sets, four per register, zero unless set.
  class Constants {
  public:
    Constants& set(uint32_t reg, float x, float y = 0.0f, float z = 0.0f, float w = 0.0f) {
      m_values[reg * 4 + 0] = x;
      m_values[reg * 4 + 1] = y;
      m_values[reg * 4 + 2] = z;
      m_values[reg * 4 + 3] = w;
      return *this;
    }

    const float* data() const { return m_values.data(); }
    uint32_t count() const { return kRegisters; }

  private:
    static constexpr uint32_t kRegisters = 32;
    std::vector<float> m_values = std::vector<float>(kRegisters * 4, 0.0f);
  };

  std::string describeFade(const DxsoMaterialFade& fade) {
    size_t offsetTerms = 0, slopeTerms = 0;
    for (uint32_t lane = 0; lane < 4; lane++) {
      offsetTerms += fade.offset[lane].size();
      slopeTerms += fade.slope[lane].size();
    }
    return std::string(fade.kind == DxsoFadeModulatorKind::MaterialScalar ? "scalar" : "vector") +
      " c" + std::to_string(fade.reg) + "." + "xyzw"[fade.component & 3u] +
      (fade.laneMask == kDxsoFadeAlphaLane ? " oC0.a" : " oC0.rgb") +
      " offset " + std::to_string(offsetTerms) + " terms, slope " + std::to_string(slopeTerms) + " terms";
  }

  bool anyTerms(const DxsoFadePoly& p) { return !p.empty(); }
  template<size_t N>
  bool anyTerms(const std::array<DxsoFadePoly, N>& polys) {
    return std::any_of(polys.begin(), polys.end(), [](const DxsoFadePoly& p) { return !p.empty(); });
  }

  std::string describeParticleColor(const DxsoParticleColorUse& use) {
    std::string out;
    if (use.tintsColor) out += anyTerms(use.tintResidual) ? " tint(conditional)" : " tint";
    if (use.scalesColor) out += anyTerms(use.scalesColorResidual) ? " scalesColor(conditional)" : " scalesColor";
    if (use.scalesOpacity) out += anyTerms(use.scalesOpacityResidual) ? " scalesOpacity(conditional)" : " scalesOpacity";
    return out.empty() ? std::string(" none") : out;
  }

  [[noreturn]] void fail(const char* label, const std::string& why) {
    std::cerr << label << ": " << why << std::endl;
    throw std::runtime_error(label);
  }

  void expectAnalyzed(const DxsoMaterialFadeResult& res, const char* label) {
    if (!res.analyzed) {
      fail(label, std::string("not analyzed: ") + dxsoHighlightFailureName(res.failure));
    }
  }

  std::string describeFades(const DxsoMaterialFadeResult& res) {
    std::string found;
    for (const DxsoMaterialFade& f : res.fades) {
      found += " [" + describeFade(f) + "]";
    }
    return found.empty() ? std::string(" none") : found;
  }

  void expectFadeCount(const DxsoMaterialFadeResult& res, size_t expected, const char* label) {
    if (res.fades.size() != expected) {
      fail(label, "found " + std::to_string(res.fades.size()) + " fades" + describeFades(res) +
                  ", expected " + std::to_string(expected));
    }
  }

  std::string describeCoverage(const std::optional<float>& coverage) {
    return coverage ? std::to_string(*coverage) : std::string("no fade");
  }

  // The fade of oC0 lanes `laneMask` by c<reg>.<component>, evaluated for a draw with these
  // constants and a blend that rests at `rest`.
  void expectCoverage(const DxsoMaterialFadeResult& res, uint32_t reg, uint8_t component, uint8_t laneMask, float rest,
                      const Constants& constants, std::optional<float> expected, const char* label) {
    for (const DxsoMaterialFade& f : res.fades) {
      if (f.reg == reg && f.component == component && f.laneMask == laneMask) {
        const std::optional<float> coverage = dxsoMaterialFadeCoverage(f, rest, constants.data(), constants.count());
        const bool matches = coverage.has_value() == expected.has_value() &&
                             (!coverage || std::abs(*coverage - *expected) <= 1.0e-4f);
        if (!matches) {
          fail(label, "fade " + describeFade(f) + " covers " + describeCoverage(coverage) +
                      ", expected " + describeCoverage(expected));
        }
        return;
      }
    }
    fail(label, "missing fade on c" + std::to_string(reg) + ", found" + describeFades(res));
  }

  bool vanishes(const DxsoFadePoly& p, const Constants& constants) {
    return dxsoFadePolyVanishes(p, constants.data(), constants.count());
  }
  template<size_t N>
  bool vanishes(const std::array<DxsoFadePoly, N>& polys, const Constants& constants) {
    return std::all_of(polys.begin(), polys.end(), [&](const DxsoFadePoly& p) { return vanishes(p, constants); });
  }

  // Which particle colour uses hold for a draw with these constants.
  void expectParticleColor(const DxsoMaterialFadeResult& res, const Constants& constants,
                           bool tintsColor, bool scalesColor, bool scalesOpacity, const char* label) {
    const DxsoParticleColorUse& use = res.particleColor;
    const bool tints = use.tintsColor && vanishes(use.tintResidual, constants);
    const bool scales = use.scalesColor && vanishes(use.scalesColorResidual, constants);
    const bool opacity = use.scalesOpacity && vanishes(use.scalesOpacityResidual, constants);
    if (tints != tintsColor || scales != scalesColor || opacity != scalesOpacity) {
      auto describe = [](bool t, bool s, bool o) {
        const std::string out = std::string(t ? " tint" : "") + (s ? " scalesColor" : "") + (o ? " scalesOpacity" : "");
        return out.empty() ? std::string(" none") : out;
      };
      fail(label, "particle colour use" + describe(tints, scales, opacity) + " (proven" + describeParticleColor(use) +
                  "), expected" + describe(tintsColor, scalesColor, scalesOpacity));
    }
  }

  void expectParticleColor(const DxsoMaterialFadeResult& res, bool tintsColor, bool scalesColor, bool scalesOpacity, const char* label) {
    expectParticleColor(res, Constants(), tintsColor, scalesColor, scalesOpacity, label);
  }

  void expectUnproven(const DxsoMaterialFadeResult& res, uint32_t scalarReg, const char* label) {
    if (std::find(res.unprovenScalarRegs.begin(), res.unprovenScalarRegs.end(), scalarReg) == res.unprovenScalarRegs.end()) {
      fail(label, "c" + std::to_string(scalarReg) + " not reported as an unproven scalar");
    }
  }

  class MaterialFadeTestApp {
  public:
    static void run() {
      std::cout << "Running DXSO material fade analysis tests..." << std::endl;
      test_subUvSmokeFadesByParticleAlpha();
      test_mirrorsEdgeVentSmoke();
      test_tintWithoutParticleOpacity();
      test_additiveParticleFoldsAlphaIntoColor();
      test_particleAlphaOnlyScalesColor();
      test_particleAlphaAlongsideOtherChannels();
      test_erosionAlphaIsNotAScale();
      test_particleAlphaThroughPowIsNotAScale();
      test_swizzledParticleColorIsNotATint();
      test_particleColorNeedsItsRegister();
      test_modulateLerpFromWhite();
      test_modulateLrpFromWhite();
      test_inverseLerpRestsAtOne();
      test_mirrorsEdgeSootDecal();
      test_translucentOpacityScalar();
      test_vectorLaneOpacity();
      test_additiveBrightnessScalar();
      test_constantLaneAgreesWithRest();
      test_colorFadeNeedsEveryLane();
      test_twoScalarsScaleOneOpacity();
      test_fadeThroughRestIsNotAFade();
      test_scalarAlsoInsideMinIsNotAFade();
      test_squaredScalarIsNotAFade();
      test_runnerVisionIsNotAFade();
      test_rejects();
      std::cout << "All DXSO material fade analysis tests passed." << std::endl;
    }

  private:
    // M_FX_LevelFX_Smoke_CheapSmoke_01-style translucent SubUV smoke: the two sub-images lerped by
    // Interp_Sizer.x, times the particle colour (TEXCOORD3), with a soft-particle depth fade read
    // from the scene colour's alpha.
    //   oC0.rgb = Sub.rgb * Color.rgb * Fog.a + Fog.rgb
    //   oC0.a   = Sub.a * Color.a * saturate((SceneDepth - PixelDepth) * k)
    static void test_subUvSmokeFadesByParticleAlpha() {
      std::cout << "  test_subUvSmokeFadesByParticleAlpha" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.01f, 0.0f, 0.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 2, 2, MaskX)
        .dclInput(DxsoUsage::Texcoord, 3, 3)
        .dclInput(DxsoUsage::Texcoord, 4, 4)
        .dclInput(DxsoUsage::Texcoord, 5, 5)
        .dclSampler(0).dclSampler(1)
        .texld(rd(0), v(0), 0)                                                   // sub-image 0
        .texld(rd(1), v(1), 0)                                                   // sub-image 1
        .op3(DxsoOpcode::Lrp, rd(2), v(2, kXXXX), r(1), r(0))                    // SubUV blend
        .op2(DxsoOpcode::Mul, rd(2), r(2), v(3))                                 // * particle colour
        .op1(DxsoOpcode::Rcp, rd(3, MaskX), v(5, kWWWW))
        .op2(DxsoOpcode::Mul, rd(3, MaskXY), v(5), r(3, kXXXX))
        .texld(rd(3), r(3), 1)                                                   // SceneColorTexture
        .op2(DxsoOpcode::Add, rd(3, MaskX), r(3, kWWWW), v(5, kWWWW, kModNeg))
        .op2(DxsoOpcode::Mul, rdSat(3, MaskX), r(3), c(1, kXXXX))                // depth fade
        .op2(DxsoOpcode::Mul, rd(2, MaskW), r(2), r(3, kXXXX))
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(2), v(4, kWWWW), v(4))
        .op1(DxsoOpcode::Mov, oC0(MaskW), r(2, kWWWW));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({}, {}), 3);
      const char* label = "SubUV smoke";
      expectAnalyzed(res, label);
      expectFadeCount(res, 0, label);
      expectParticleColor(res, true, false, true, label);
    }

    // PS_FX_LevelFX_Smoke_VentSmoke_05_Quick's shader, as Mirror's Edge's cache holds it: the
    // sub-images' green channel as colour, a UniformVector_0 emissive offset, and the opacity
    // clamped to [0, 15] by a max and a min.
    static void test_mirrorsEdgeVentSmoke() {
      std::cout << "  test_mirrorsEdgeVentSmoke" << std::endl;
      PsBuilder ps;
      ps.def(2, 0.005f, 0.0f, 15.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 2, 2, MaskX)
        .dclInput(DxsoUsage::Texcoord, 3, 3)
        .dclInput(DxsoUsage::Texcoord, 4, 4)
        .dclInput(DxsoUsage::Texcoord, 5, 5)
        .dclSampler(0).dclSampler(1)
        .texld(rd(4), v(5), 0)                                                   // scene depth
        .op2(DxsoOpcode::Mul, rdSat(0, MaskX), r(4, kXXXX), c(2, kXXXX))         // depth fade
        .texld(rd(1), v(0), 1)
        .texld(rd(2), v(1), 1)
        .op3(DxsoOpcode::Lrp, rd(3), v(2, kXXXX), r(2, swz(1, 1, 1, 3)), r(1, swz(1, 1, 1, 3)))
        .op2(DxsoOpcode::Mul, rd(0, MaskY), r(3, kWWWW), v(3, kWWWW))
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(3), v(3), c(0))
        .op2(DxsoOpcode::Mul, rd(0, MaskX), r(0, kXXXX), r(0, kYYYY))
        .op2(DxsoOpcode::Max, rd(1, MaskW), r(0, kXXXX), c(2, kYYYY))
        .op2(DxsoOpcode::Min, oC0(MaskW), r(1, kWWWW), c(2, kZZZZ))
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(1), v(4, kWWWW), v(4));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({}, { 0 }), 3);
      const char* label = "Mirror's Edge vent smoke";
      expectAnalyzed(res, label);
      expectFadeCount(res, 0, label);
      // The tint holds while the emissive offset is black, which is how the vent smoke draws.
      expectParticleColor(res, Constants(), true, false, true, label);
      expectParticleColor(res, Constants().set(0, 0.2f, 0.0f, 0.0f, 0.0f), false, false, true, label);
    }

    // A material that tints by the particle colour but takes its opacity from the texture alone.
    static void test_tintWithoutParticleOpacity() {
      std::cout << "  test_tintWithoutParticleOpacity" << std::endl;
      PsBuilder ps;
      ps.dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(0), v(1))
        .op1(DxsoOpcode::Mov, oC0(MaskW), r(0, kWWWW));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({}, {}), 1);
      const char* label = "tint without particle opacity";
      expectAnalyzed(res, label);
      expectParticleColor(res, true, false, false, label);
    }

    // Additive sparks: UE3 multiplies the opacity into the colour and writes alpha 0.
    //   oC0 = (Tex.rgb * Color.rgb * Fog.a * Tex.a * Color.a, 0)
    static void test_additiveParticleFoldsAlphaIntoColor() {
      std::cout << "  test_additiveParticleFoldsAlphaIntoColor" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.0f, 0.0f, 0.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1)
        .dclInput(DxsoUsage::Texcoord, 4, 2)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op2(DxsoOpcode::Mul, rd(1), r(0), v(1))                                 // Tex * Color
        .op2(DxsoOpcode::Mul, rd(2, MaskX), r(1, kWWWW), v(2, kWWWW))            // Opacity * Fog.a
        .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(1), r(2, kXXXX))
        .op1(DxsoOpcode::Mov, oC0(MaskW), c(1, kXXXX));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({}, {}), 1);
      const char* label = "additive particle";
      expectAnalyzed(res, label);
      expectParticleColor(res, true, true, false, label);
    }

    // An additive material that fades by the particle alpha without being tinted by its colour.
    static void test_particleAlphaOnlyScalesColor() {
      std::cout << "  test_particleAlphaOnlyScalesColor" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.0f, 0.0f, 0.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1, MaskW)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(0), v(1, kWWWW))
        .op1(DxsoOpcode::Mov, oC0(MaskW), c(1, kXXXX));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({}, {}), 1);
      const char* label = "particle alpha scales colour";
      expectAnalyzed(res, label);
      expectParticleColor(res, false, true, false, label);
    }

    // The opacity is linear in the particle alpha even where the colour's red also shapes it.
    static void test_particleAlphaAlongsideOtherChannels() {
      std::cout << "  test_particleAlphaAlongsideOtherChannels" << std::endl;
      PsBuilder ps;
      ps.dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op2(DxsoOpcode::Mul, rd(1, MaskX), v(1, kXXXX), v(1, kXXXX))
        .op2(DxsoOpcode::Mul, rd(1, MaskX), r(1, kXXXX), v(1, kWWWW))
        .op2(DxsoOpcode::Mul, oC0(MaskW), r(0, kWWWW), r(1, kXXXX))
        .op1(DxsoOpcode::Mov, oC0(MaskXYZ), r(0));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({}, {}), 1);
      const char* label = "particle alpha alongside other channels";
      expectAnalyzed(res, label);
      expectParticleColor(res, false, false, true, label);
    }

    // Erosion: saturate((Tex.a - (1 - Color.a)) * 4) dissolves the texture's alpha threshold
    // rather than scaling its opacity.
    static void test_erosionAlphaIsNotAScale() {
      std::cout << "  test_erosionAlphaIsNotAScale" << std::endl;
      PsBuilder ps;
      ps.def(1, 4.0f, 1.0f, 0.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op2(DxsoOpcode::Add, rd(1, MaskX), c(1, kYYYY), v(1, kWWWW, kModNeg))   // 1 - Color.a
        .op2(DxsoOpcode::Add, rd(1, MaskX), r(0, kWWWW), r(1, kXXXX, kModNeg))
        .op2(DxsoOpcode::Mul, rdSat(1, MaskX), r(1), c(1, kXXXX))
        .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(0), v(1))
        .op1(DxsoOpcode::Mov, oC0(MaskW), r(1, kXXXX));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({}, {}), 1);
      const char* label = "erosion alpha";
      expectAnalyzed(res, label);
      expectParticleColor(res, true, false, false, label);
    }

    // The particle alpha scales the opacity linearly and also through a pow(), so the opacity is
    // not affine in it.
    static void test_particleAlphaThroughPowIsNotAScale() {
      std::cout << "  test_particleAlphaThroughPowIsNotAScale" << std::endl;
      PsBuilder ps;
      ps.def(1, 2.0f, 0.0f, 0.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op2(DxsoOpcode::Pow, rd(1, MaskX), v(1, kWWWW), c(1, kXXXX))
        .op2(DxsoOpcode::Mul, rd(1, MaskX), r(1), v(1, kWWWW))
        .op2(DxsoOpcode::Mul, oC0(MaskW), r(0, kWWWW), r(1, kXXXX))
        .op1(DxsoOpcode::Mov, oC0(MaskXYZ), r(0));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({}, {}), 1);
      const char* label = "particle alpha through pow";
      expectAnalyzed(res, label);
      expectParticleColor(res, false, false, false, label);
    }

    // Every colour channel scaled by the colour's red: not something a per-channel vertex colour
    // multiply reproduces.
    static void test_swizzledParticleColorIsNotATint() {
      std::cout << "  test_swizzledParticleColorIsNotATint" << std::endl;
      PsBuilder ps;
      ps.dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(0), v(1, kXXXX))
        .op2(DxsoOpcode::Mul, oC0(MaskW), r(0, kWWWW), v(1, kWWWW));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({}, {}), 1);
      const char* label = "swizzled particle colour";
      expectAnalyzed(res, label);
      expectParticleColor(res, false, false, true, label);
    }

    // Without a particle colour register the same input is an ordinary interpolant.
    static void test_particleColorNeedsItsRegister() {
      std::cout << "  test_particleColorNeedsItsRegister" << std::endl;
      PsBuilder ps;
      ps.dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 1, 1)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op2(DxsoOpcode::Mul, oC0(), r(0), v(1));

      const char* label = "particle colour register";
      const DxsoMaterialFadeResult without = ps.analyze(makeInputs({}, {}));
      expectAnalyzed(without, label);
      expectParticleColor(without, false, false, false, label);
      const DxsoMaterialFadeResult otherRegister = ps.analyze(makeInputs({}, {}), 0);
      expectAnalyzed(otherRegister, label);
      expectParticleColor(otherRegister, false, false, false, label);
    }

    // A modulate soot decal: Emissive = Lerp(1, Soot, S). At S = 0 it multiplies the scene by
    // white and is invisible.
    static void test_modulateLerpFromWhite() {
      std::cout << "  test_modulateLerpFromWhite" << std::endl;
      PsBuilder ps;
      ps.def(1, 1.0f, 0.0f, 0.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op2(DxsoOpcode::Add, rd(1, MaskXYZ), r(0), c(1, kXXXX, kModNeg))        // Soot - 1
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), c(3, kXXXX), r(1), c(1, kXXXX))      // S * (Soot - 1) + 1
        .op1(DxsoOpcode::Mov, oC0(MaskW), r(0, kWWWW));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({ 3 }, {}));
      const char* label = "modulate lerp from white";
      expectAnalyzed(res, label);
      expectFadeCount(res, 1, label);
      expectCoverage(res, 3, 0, kDxsoFadeColorLanes, 1.0f, Constants().set(3, 0.25f), 0.25f, label);
      expectCoverage(res, 3, 0, kDxsoFadeColorLanes, 1.0f, Constants().set(3, 0.0f), 0.0f, label);
      // An additive blend rests at black, which the lerp from white never reaches.
      expectCoverage(res, 3, 0, kDxsoFadeColorLanes, 0.0f, Constants().set(3, 0.25f), std::nullopt, label);
    }

    // The same lerp compiled to lrp.
    static void test_modulateLrpFromWhite() {
      std::cout << "  test_modulateLrpFromWhite" << std::endl;
      PsBuilder ps;
      ps.def(1, 1.0f, 0.0f, 0.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op3(DxsoOpcode::Lrp, oC0(MaskXYZ), c(3, kXXXX), r(0), c(1, kXXXX))
        .op1(DxsoOpcode::Mov, oC0(MaskW), r(0, kWWWW));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({ 3 }, {}));
      const char* label = "modulate lrp from white";
      expectAnalyzed(res, label);
      expectFadeCount(res, 1, label);
      expectCoverage(res, 3, 0, kDxsoFadeColorLanes, 1.0f, Constants().set(3, 0.6f), 0.6f, label);
    }

    // Lerp(Soot, 1, S): the decal is at rest when the parameter reaches 1.
    static void test_inverseLerpRestsAtOne() {
      std::cout << "  test_inverseLerpRestsAtOne" << std::endl;
      PsBuilder ps;
      ps.def(1, 1.0f, 0.0f, 0.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op3(DxsoOpcode::Lrp, oC0(MaskXYZ), c(3, kXXXX), c(1, kXXXX), r(0))
        .op1(DxsoOpcode::Mov, oC0(MaskW), r(0, kWWWW));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({ 3 }, {}));
      const char* label = "inverse lerp";
      expectAnalyzed(res, label);
      expectFadeCount(res, 1, label);
      expectCoverage(res, 3, 0, kDxsoFadeColorLanes, 1.0f, Constants().set(3, 0.25f), 0.75f, label);
      expectCoverage(res, 3, 0, kDxsoFadeColorLanes, 1.0f, Constants().set(3, 1.0f), 0.0f, label);
    }

    // The soot decal on The Shard, as Mirror's Edge's cache holds it. The lerp's ends are folded
    // into two UniformVectors, so the colour is 1 - Soot + V0 + S * Soot * V1: a fade only while
    // V1 is white and V0 black, which only the draw's constants say.
    static void test_mirrorsEdgeSootDecal() {
      std::cout << "  test_mirrorsEdgeSootDecal" << std::endl;
      PsBuilder ps;
      ps.def(1, -1.0f, 1.0f, 0.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op1(DxsoOpcode::Mov, rd(1, MaskX), c(1, kXXXX))
        .op1(DxsoOpcode::Mov, rd(2, MaskX), c(3, kXXXX))
        .op3(DxsoOpcode::Mad, rd(0, MaskY | MaskZ | MaskW), r(2, kXXXX), c(2, swz(0, 0, 1, 2)), r(1, kXXXX))
        .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), r(0, kXXXX), r(0, swz(1, 2, 3, 3)), c(0))
        .op2(DxsoOpcode::Add, oC0(MaskXYZ), r(0), c(1, kYYYY))
        .op1(DxsoOpcode::Mov, oC0(MaskW), c(1, kYYYY));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({ 3 }, { 0, 2 }));
      const char* label = "Mirror's Edge soot decal";
      expectAnalyzed(res, label);
      // V0 and V1's components each reach one lane, and tint it rather than fade the colour.
      expectFadeCount(res, 1, label);
      Constants draw;
      draw.set(2, 1.0f, 1.0f, 1.0f, 0.0f);
      expectCoverage(res, 3, 0, kDxsoFadeColorLanes, 1.0f, draw.set(3, 1.0f), 0.0f, label);
      expectCoverage(res, 3, 0, kDxsoFadeColorLanes, 1.0f, draw.set(3, 0.25f), 0.75f, label);
      expectCoverage(res, 3, 0, kDxsoFadeColorLanes, 1.0f, draw.set(3, 0.0f), 1.0f, label);
      Constants tinted;
      tinted.set(2, 1.0f, 0.5f, 1.0f, 0.0f).set(3, 0.25f);
      expectCoverage(res, 3, 0, kDxsoFadeColorLanes, 1.0f, tinted, std::nullopt, label);
      Constants offset;
      offset.set(0, 0.1f, 0.1f, 0.1f, 0.0f).set(2, 1.0f, 1.0f, 1.0f, 0.0f).set(3, 0.25f);
      expectCoverage(res, 3, 0, kDxsoFadeColorLanes, 1.0f, offset, std::nullopt, label);
    }

    // A translucent material whose opacity a parameter scales; its colour is fogged and stays.
    static void test_translucentOpacityScalar() {
      std::cout << "  test_translucentOpacityScalar" << std::endl;
      PsBuilder ps;
      ps.dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 4, 1)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(0), v(1, kWWWW), v(1))
        .op2(DxsoOpcode::Mul, oC0(MaskW), r(0, kWWWW), c(2, kXXXX));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({ 2 }, {}));
      const char* label = "translucent opacity scalar";
      expectAnalyzed(res, label);
      expectFadeCount(res, 1, label);
      expectCoverage(res, 2, 0, kDxsoFadeAlphaLane, 0.0f, Constants().set(2, 0.4f), 0.4f, label);
    }

    // UE3 folds parameter arithmetic into one CPU-evaluated register, so an opacity parameter
    // can arrive as one lane of a UniformVector.
    static void test_vectorLaneOpacity() {
      std::cout << "  test_vectorLaneOpacity" << std::endl;
      PsBuilder ps;
      ps.dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(0), c(4))
        .op2(DxsoOpcode::Mul, oC0(MaskW), r(0, kWWWW), c(4, kWWWW));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({}, { 4 }));
      const char* label = "vector lane opacity";
      expectAnalyzed(res, label);
      expectFadeCount(res, 1, label);
      expectCoverage(res, 4, 3, kDxsoFadeAlphaLane, 0.0f, Constants().set(4, 1.0f, 1.0f, 1.0f, 0.4f), 0.4f, label);
    }

    // An additive material whose brightness a scalar sets fades to black with it; the vector
    // colour tinting each channel separately is not a fade of the whole colour.
    static void test_additiveBrightnessScalar() {
      std::cout << "  test_additiveBrightnessScalar" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.0f, 0.0f, 0.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Texcoord, 4, 1)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(0), c(4))
        .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(0), c(2, kXXXX))
        .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(0), v(1, kWWWW))
        .op1(DxsoOpcode::Mov, oC0(MaskW), c(1, kXXXX));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({ 2 }, { 4 }));
      const char* label = "additive brightness scalar";
      expectAnalyzed(res, label);
      expectFadeCount(res, 1, label);
      Constants draw;
      draw.set(4, 1.0f, 0.5f, 0.25f, 0.0f);
      expectCoverage(res, 2, 0, kDxsoFadeColorLanes, 0.0f, draw.set(2, 0.6f), 0.6f, label);
      expectCoverage(res, 2, 0, kDxsoFadeColorLanes, 0.0f, draw.set(2, 0.0f), 0.0f, label);
      // With a black colour every lane is at rest whatever the scalar.
      expectCoverage(res, 2, 0, kDxsoFadeColorLanes, 0.0f, Constants().set(2, 0.6f), 0.0f, label);
    }

    // A lane the parameter never reaches agrees with the fade when it already holds the rest value.
    static void test_constantLaneAgreesWithRest() {
      std::cout << "  test_constantLaneAgreesWithRest" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.0f, 0.0f, 0.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op2(DxsoOpcode::Mul, oC0(MaskX), r(0), c(2, kXXXX))
        .op1(DxsoOpcode::Mov, oC0(MaskY | MaskZ | MaskW), c(1, kXXXX));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({ 2 }, {}));
      const char* label = "constant lane agrees with rest";
      expectAnalyzed(res, label);
      expectFadeCount(res, 1, label);
      expectCoverage(res, 2, 0, kDxsoFadeColorLanes, 0.0f, Constants().set(2, 0.3f), 0.3f, label);
      // Under a modulate blend the black lanes darken the scene whatever the parameter.
      expectCoverage(res, 2, 0, kDxsoFadeColorLanes, 1.0f, Constants().set(2, 0.3f), std::nullopt, label);
    }

    // A parameter that fades only the red channel leaves the others showing.
    static void test_colorFadeNeedsEveryLane() {
      std::cout << "  test_colorFadeNeedsEveryLane" << std::endl;
      PsBuilder ps;
      ps.dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op2(DxsoOpcode::Mul, oC0(MaskX), r(0), c(2, kXXXX))
        .op1(DxsoOpcode::Mov, oC0(MaskY | MaskZ | MaskW), r(0));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({ 2 }, {}));
      const char* label = "colour fade needs every lane";
      expectAnalyzed(res, label);
      expectFadeCount(res, 0, label);
      expectUnproven(res, 2, label);
    }

    // Opacity = Tex.a * S1 * S2: each scalar on its own fades the draw.
    static void test_twoScalarsScaleOneOpacity() {
      std::cout << "  test_twoScalarsScaleOneOpacity" << std::endl;
      PsBuilder ps;
      ps.dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op2(DxsoOpcode::Mul, rd(1, MaskX), r(0, kWWWW), c(2, kXXXX))
        .op2(DxsoOpcode::Mul, oC0(MaskW), r(1, kXXXX), c(3, kXXXX))
        .op1(DxsoOpcode::Mov, oC0(MaskXYZ), r(0));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({ 2, 3 }, {}));
      const char* label = "two scalars scale one opacity";
      expectAnalyzed(res, label);
      expectFadeCount(res, 2, label);
      Constants draw;
      draw.set(2, 0.5f).set(3, 0.8f);
      expectCoverage(res, 2, 0, kDxsoFadeAlphaLane, 0.0f, draw, 0.5f, label);
      expectCoverage(res, 3, 0, kDxsoFadeAlphaLane, 0.0f, draw, 0.8f, label);
      expectCoverage(res, 2, 0, kDxsoFadeAlphaLane, 0.0f, Constants(), 0.0f, label);
    }

    // Opacity = Tex.a * (2S - 1): transparent at S = 0.5, so S does not fade the draw in from rest.
    static void test_fadeThroughRestIsNotAFade() {
      std::cout << "  test_fadeThroughRestIsNotAFade" << std::endl;
      PsBuilder ps;
      ps.def(1, 2.0f, -1.0f, 0.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op3(DxsoOpcode::Mad, rd(1, MaskX), c(2, kXXXX), c(1, kXXXX), c(1, kYYYY))
        .op2(DxsoOpcode::Mul, oC0(MaskW), r(0, kWWWW), r(1, kXXXX))
        .op1(DxsoOpcode::Mov, oC0(MaskXYZ), r(0));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({ 2 }, {}));
      const char* label = "fade through rest";
      expectAnalyzed(res, label);
      expectCoverage(res, 2, 0, kDxsoFadeAlphaLane, 0.0f, Constants().set(2, 0.75f), std::nullopt, label);
    }

    // Opacity = Tex.a * S * min(S * 4, 0.5): linear in S at the top level, but S also passes
    // through the min, whose bound is no range clamp.
    static void test_scalarAlsoInsideMinIsNotAFade() {
      std::cout << "  test_scalarAlsoInsideMinIsNotAFade" << std::endl;
      PsBuilder ps;
      ps.def(1, 4.0f, 0.5f, 0.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op2(DxsoOpcode::Mul, rd(1, MaskX), c(2, kXXXX), c(1, kXXXX))
        .op2(DxsoOpcode::Min, rd(1, MaskX), r(1, kXXXX), c(1, kYYYY))
        .op2(DxsoOpcode::Mul, rd(1, MaskX), r(1, kXXXX), c(2, kXXXX))
        .op2(DxsoOpcode::Mul, oC0(MaskW), r(0, kWWWW), r(1, kXXXX))
        .op1(DxsoOpcode::Mov, oC0(MaskXYZ), r(0));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({ 2 }, {}));
      const char* label = "scalar also inside min";
      expectAnalyzed(res, label);
      expectFadeCount(res, 0, label);
      expectUnproven(res, 2, label);
    }

    static void test_squaredScalarIsNotAFade() {
      std::cout << "  test_squaredScalarIsNotAFade" << std::endl;
      PsBuilder ps;
      ps.dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op2(DxsoOpcode::Mul, rd(1, MaskX), c(2, kXXXX), c(2, kXXXX))
        .op2(DxsoOpcode::Mul, oC0(MaskW), r(0, kWWWW), r(1, kXXXX))
        .op1(DxsoOpcode::Mov, oC0(MaskXYZ), r(0));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({ 2 }, {}));
      const char* label = "squared scalar";
      expectAnalyzed(res, label);
      expectFadeCount(res, 0, label);
      expectUnproven(res, 2, label);
    }

    // lerp(X, X * V, S) plus its 0.1 * S glow: the surface never comes to rest.
    static void test_runnerVisionIsNotAFade() {
      std::cout << "  test_runnerVisionIsNotAFade" << std::endl;
      PsBuilder ps;
      ps.def(1, 0.100000001f, 1.0f, 0.0f, 0.0f)
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), c(3), r(0), r(0, kXYZW, kModNeg))
        .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), c(4, kXXXX), r(1), r(0))
        .op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(0), c(4, kXXXX))
        .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(1), c(1, kXXXX), r(0))
        .op1(DxsoOpcode::Mov, oC0(MaskW), c(1, kYYYY));

      const DxsoMaterialFadeResult res = ps.analyze(makeInputs({ 4 }, { 3 }));
      const char* label = "Runner Vision";
      expectAnalyzed(res, label);
      expectFadeCount(res, 0, label);
      expectUnproven(res, 4, label);
    }

    static void test_rejects() {
      std::cout << "  test_rejects" << std::endl;
      const DxsoHighlightInputs in = makeInputs({ 4 }, { 3 });

      const std::vector<uint32_t> vs = { 0xFFFE0300u, kEndToken };
      if (analyzeDxsoMaterialFades(vs.data(), vs.size(), in, -1).failure != DxsoHighlightFailure::NotPixelShader) {
        fail("reject vs", "vertex shader was analyzed");
      }
      const std::vector<uint32_t> ps14 = { 0xFFFF0104u, kEndToken };
      if (analyzeDxsoMaterialFades(ps14.data(), ps14.size(), in, -1).failure != DxsoHighlightFailure::NotPixelShader) {
        fail("reject ps_1_x", "ps_1_4 was analyzed");
      }
      if (analyzeDxsoMaterialFades(nullptr, 0, in, -1).analyzed) {
        fail("reject null", "null bytecode was analyzed");
      }

      PsBuilder branch;
      branch.dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .ifBool(0)
        .op2(DxsoOpcode::Mul, rd(0, MaskW), r(0), c(4, kXXXX))
        .endIf()
        .op1(DxsoOpcode::Mov, oC0(), r(0));
      const DxsoMaterialFadeResult branchRes = branch.analyze(in);
      if (branchRes.analyzed || branchRes.failure != DxsoHighlightFailure::FlowControl) {
        fail("reject flow control", std::string("failure = ") + dxsoHighlightFailureName(branchRes.failure));
      }

      PsBuilder noOutput;
      noOutput.dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .texld(rd(0), v(0), 0);
      const DxsoMaterialFadeResult noOutputRes = noOutput.analyze(in);
      if (noOutputRes.failure != DxsoHighlightFailure::NoColorOutput) {
        fail("reject no output", std::string("failure = ") + dxsoHighlightFailureName(noOutputRes.failure));
      }
    }
  };

  // --- dump mode -------------------------------------------------------------------------

  struct DumpTotals {
    uint32_t shaders = 0;
    uint32_t pixelShaders = 0;
    uint32_t analyzed = 0;
    uint32_t withParticleColorInput = 0;
    std::map<std::string, uint32_t> fadeShapes;
    std::map<std::string, uint32_t> particleUses;
    uint32_t withUnproven = 0;
    std::map<std::string, uint32_t> failures;
  };

  int dumpShader(const std::filesystem::path& path, bool summaryOnly, int32_t particleColorTexcoord, DumpTotals& totals) {
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

    // The runtime derives the inputs from the CTAB, and the particle colour register from the
    // shader's input declarations, the same way.
    const DxsoProgramInfo programInfo(DxsoProgramTypes::PixelShader, tokens[0] & 0xffu, (tokens[0] >> 8) & 0xffu);
    DxsoDecodeContext decoder(programInfo);
    DxsoCodeIter iter(tokens + 1);
    int32_t particleColorInputRegister = -1;
    while (decoder.decodeInstruction(iter)) {
      const DxsoInstructionContext& ctx = decoder.getInstructionContext();
      if (ctx.instruction.opcode == DxsoOpcode::Dcl && ctx.dst.id.type == DxsoRegisterType::Input &&
          ctx.dcl.semantic.usage == DxsoUsage::Texcoord && int32_t(ctx.dcl.semantic.usageIndex) == particleColorTexcoord) {
        particleColorInputRegister = int32_t(ctx.dst.id.num);
      }
    }
    const DxsoCtab& ctab = decoder.getCtabInfo();
    const DxsoHighlightInputs in = dxsoHighlightInputsFromUe3Ctab(ctab);
    const DxsoMaterialFadeResult res = analyzeDxsoMaterialFades(tokens, tokenCount, in, particleColorInputRegister);
    totals.withParticleColorInput += particleColorInputRegister >= 0 ? 1u : 0u;

    constexpr uint16_t kRegisterSetFloat4 = 2u;
    auto nameOf = [&](uint32_t reg) -> std::string {
      for (const DxsoCtab::Constant& k : ctab.m_constantData) {
        if (k.registerSet == kRegisterSetFloat4 && reg >= k.registerIndex && reg < k.registerIndex + k.registerCount) {
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
    for (const DxsoMaterialFade& f : res.fades) {
      totals.fadeShapes[std::string(f.kind == DxsoFadeModulatorKind::MaterialScalar ? "scalar" : "vector") +
                        (f.laneMask == kDxsoFadeAlphaLane ? " oC0.a" : " oC0.rgb")]++;
    }
    if (particleColorInputRegister >= 0) {
      totals.particleUses[describeParticleColor(res.particleColor)]++;
    }
    totals.withUnproven += res.unprovenScalarRegs.empty() ? 0u : 1u;

    const bool particleColorUsed = res.particleColor.tintsColor || res.particleColor.scalesColor || res.particleColor.scalesOpacity;
    if (summaryOnly || (res.fades.empty() && res.unprovenScalarRegs.empty() && !particleColorUsed)) {
      return 0;
    }
    std::cout << "== " << path.string() << "\n";
    for (const DxsoMaterialFade& f : res.fades) {
      std::cout << "   fade " << nameOf(f.reg) << " (" << describeFade(f) << ")\n";
    }
    if (particleColorInputRegister >= 0) {
      std::cout << "   particle colour v" << particleColorInputRegister << ":" << describeParticleColor(res.particleColor) << "\n";
    }
    for (const uint32_t reg : res.unprovenScalarRegs) {
      std::cout << "   unproven " << nameOf(reg) << " (c" << reg << ")\n";
    }
    return 0;
  }

} // anonymous namespace

int main(int argc, char** argv) {
  if (argc > 1) {
    bool summaryOnly = false;
    int32_t particleColorTexcoord = -1;
    DumpTotals totals;
    int rc = 0;
    for (int i = 1; i < argc; i++) {
      const std::string arg = argv[i];
      if (arg == "--summary") {
        summaryOnly = true;
        continue;
      }
      if (arg == "--particle-color" && i + 1 < argc) {
        particleColorTexcoord = std::stoi(argv[++i]);
        continue;
      }
      const std::filesystem::path path(arg);
      if (std::filesystem::is_directory(path)) {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
          if (entry.is_regular_file() && entry.path().extension() == ".dxso") {
            rc |= dumpShader(entry.path(), summaryOnly, particleColorTexcoord, totals);
          }
        }
      } else {
        rc |= dumpShader(path, summaryOnly, particleColorTexcoord, totals);
      }
    }
    std::cout << "shaders=" << totals.shaders << " pixelShaders=" << totals.pixelShaders
              << " analyzed=" << totals.analyzed << " withUnprovenScalars=" << totals.withUnproven << "\n";
    for (const auto& [shape, count] : totals.fadeShapes) {
      std::cout << "  fade " << shape << ": " << count << "\n";
    }
    if (particleColorTexcoord >= 0) {
      std::cout << "  with a TEXCOORD" << particleColorTexcoord << " input: " << totals.withParticleColorInput << "\n";
      for (const auto& [use, count] : totals.particleUses) {
        std::cout << "  particle colour" << use << ": " << count << "\n";
      }
    }
    for (const auto& [reason, count] : totals.failures) {
      std::cout << "  not analyzed (" << reason << "): " << count << "\n";
    }
    return rc;
  }

  try {
    MaterialFadeTestApp::run();
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
