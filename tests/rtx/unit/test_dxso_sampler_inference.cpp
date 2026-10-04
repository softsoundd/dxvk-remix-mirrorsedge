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

// Per-sampler inference that feeds albedo scoring: the texcoord set and UV hints of a sampler's
// coordinate, and how its sampled value is used. The hand-assembled cases follow the shapes
// UE3's material compiler emits.

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../../src/dxso/dxso_sampler_inference.h"
#include "../../../src/util/log/log.h"
#include "../../../src/util/util_string.h"

#include "dxso_test_assembler.h"

namespace dxvk {
  // Standalone executables own the logger singleton, as the d3d9/dxgi entry points do.
  Logger Logger::s_instance("test_dxso_sampler_inference.log", LogLevel::None);
}

using namespace dxvk;
using namespace dxvk::dxso_test;

namespace {

  std::string describeFlags(uint32_t flags, std::initializer_list<std::pair<uint32_t, const char*>> names) {
    std::string out;
    for (const auto& [bit, name] : names) {
      if ((flags & bit) != 0) {
        out += out.empty() ? "" : " ";
        out += name;
      }
    }
    return out;
  }

  std::string describeInference(const PsSamplerTexcoordInference& i) {
    std::string out = str::format("tc=", i.texcoord, " samples=", i.sampleCount);
    if (i.coordCompValid) {
      out += str::format(" comps=(", uint32_t(i.coordCompU), ",", uint32_t(i.coordCompV), ")");
    }
    if (i.scaleConstReg >= 0) {
      out += str::format(" scale=c", i.scaleConstReg, ".", "xyzw"[i.scaleConstCompU], "xyzw"[i.scaleConstCompV]);
    }
    if (i.scaleImmediateValid) {
      out += str::format(" scaleImm=(", i.scaleImmediateU, ",", i.scaleImmediateV, ")");
    }
    if (i.offsetConstReg >= 0) {
      out += str::format(" offset=c", i.offsetConstReg, ".", "xyzw"[i.offsetConstCompU], "xyzw"[i.offsetConstCompV]);
    }
    if (i.offsetImmediateValid) {
      out += str::format(" offsetImm=(", i.offsetImmediateU, ",", i.offsetImmediateV, ")");
    }
    out += " sem=[" + describeFlags(i.semanticFlags, {
      { kPsSamplerSemanticEngineAuxiliary, "AUX" },
      { kPsSamplerSemanticLightmap, "LIGHTMAP" },
      { kPsSamplerSemanticMaterialTexture, "MAT" },
      { kPsSamplerSemanticNonDiffuse, "NONDIFFUSE" },
      { kPsSamplerSemanticVideo, "VIDEO" },
      { kPsSamplerSemanticMovieTexture, "MOVIE" } }) + "]";
    out += " expr=[" + describeFlags(i.expressionFlags, {
      { kPsSamplerExprUvTransform, "UVXFORM" },
      { kPsSamplerExprUvOffset, "UVOFS" },
      { kPsSamplerExprUvAnimated, "UVANIM" },
      { kPsSamplerExprBlendMath, "BLEND" },
      { kPsSamplerExprUvTimeDriven, "TIME" },
      { kPsSamplerExprViewDependent, "VIEW" },
      { kPsSamplerExprMaskControl, "MASK" },
      { kPsSamplerExprColorContribution, "COLOR" },
      { kPsSamplerExprNormalDecode, "NORMAL" },
      { kPsSamplerExprReachesOutputColor, "REACHESOC0" },
      { kPsSamplerExprDiffuseAnchor, "ANCHOR" } }) + "]";
    if (!i.coordConstRegs.empty()) {
      out += " consts={";
      for (size_t k = 0; k < i.coordConstRegs.size(); k++) {
        out += str::format(k ? "," : "", i.coordConstRegs[k]);
      }
      out += "}";
    }
    return out;
  }

  void expectInference(const PsSamplerTexcoordInference& actual, const std::string& expected, const char* label) {
    const std::string description = describeInference(actual);
    if (description != expected) {
      std::cerr << label << ":\n  expected " << expected << "\n  actual   " << description << std::endl;
      throw std::runtime_error(label);
    }
  }

  // ps_3_0 reading TEXCOORD0 through v0.
  class PsShader : public DxsoTestShader {
  public:
    PsShader() : DxsoTestShader(kPs30Header) { }

    PsShader& base(uint32_t texcoordMask = MaskXY) {
      dclInput(DxsoUsage::Texcoord, 0, 0, texcoordMask);
      dclSampler(0);
      return *this;
    }

    PsSamplerTexcoordInference infer(uint32_t sampler = 0) {
      return inferPixelShaderTexcoordForSampler(view(), sampler);
    }
  };

  class SamplerInferenceTestApp {
  public:
    static void run() {
      std::cout << std::endl << "Begin DXSO sampler inference tests" << std::endl;
      test_plainRead();
      test_constantTiling();
      test_selfAdd();
      test_timeNamedPanner();
      test_matrixConstantDependencies();
      test_lightmapName();
      test_normalDecode();
      test_diffuseAnchor();
      test_undeclaredSamplerIsDefault();
      test_rejectsMissingInputs();
      test_lightingInputRule();
      std::cout << "All DXSO sampler inference tests passed" << std::endl;
    }

  private:
    static void test_plainRead() {
      std::cout << "  test_plainRead" << std::endl;
      PsShader ps;
      ps.ctab({ { "Texture2D_0", kD3dxRegisterSetSampler, 0 } });
      ps.base();
      ps.texld(rd(0), v(0), 0).op1(DxsoOpcode::Mov, oC0(), r(0));
      expectInference(ps.infer(), "tc=0 samples=1 comps=(0,1) sem=[MAT] expr=[COLOR REACHESOC0]", "plain read");
    }

    static void test_constantTiling() {
      std::cout << "  test_constantTiling" << std::endl;
      PsShader ps;
      ps.base();
      ps.op2(DxsoOpcode::Mul, rd(1, MaskXY), v(0), c(3))
        .texld(rd(0), r(1), 0)
        .op1(DxsoOpcode::Mov, oC0(), r(0));
      expectInference(ps.infer(),
                      "tc=0 samples=1 comps=(0,1) scale=c3.xy sem=[MAT] expr=[UVXFORM BLEND COLOR REACHESOC0] consts={3}",
                      "constant tiling");
    }

    static void test_selfAdd() {
      std::cout << "  test_selfAdd" << std::endl;
      PsShader ps;
      ps.base();
      ps.op2(DxsoOpcode::Add, rd(1, MaskXY), v(0), v(0))
        .texld(rd(0), r(1), 0)
        .op1(DxsoOpcode::Mov, oC0(), r(0));
      expectInference(ps.infer(), "tc=0 samples=1 comps=(0,1) sem=[] expr=[BLEND COLOR REACHESOC0]", "add r1, v0, v0");
    }

    // A named parameter driving the offset marks the coordinate time-driven.
    static void test_timeNamedPanner() {
      std::cout << "  test_timeNamedPanner" << std::endl;
      PsShader ps;
      ps.ctab({ { "Texture2D_0", kD3dxRegisterSetSampler, 0 }, { "PannerTime", kD3dxRegisterSetFloat4, 4 } });
      ps.base();
      ps.op2(DxsoOpcode::Add, rd(1, MaskXY), v(0), c(4))
        .texld(rd(0), r(1), 0)
        .op1(DxsoOpcode::Mov, oC0(), r(0));
      expectInference(ps.infer(),
                      "tc=0 samples=1 comps=(0,1) offset=c4.xy sem=[MAT] expr=[UVOFS UVANIM BLEND TIME COLOR REACHESOC0] consts={4}",
                      "time-named panner");
    }

    // A matrix multiply names only its first row's register.
    static void test_matrixConstantDependencies() {
      std::cout << "  test_matrixConstantDependencies" << std::endl;
      PsShader ps;
      ps.base(MaskXYZ);
      ps.op2(DxsoOpcode::M3x2, rd(1, MaskXY), v(0), c(4))
        .texld(rd(0), r(1), 0)
        .op1(DxsoOpcode::Mov, oC0(), r(0));
      expectInference(ps.infer(),
                      "tc=0 samples=1 comps=(0,1) sem=[MAT] expr=[UVXFORM BLEND COLOR REACHESOC0] consts={4,5}",
                      "m3x2");
    }

    static void test_lightmapName() {
      std::cout << "  test_lightmapName" << std::endl;
      PsShader ps;
      ps.ctab({ { "LightMapTextures", kD3dxRegisterSetSampler, 0 } });
      ps.base(MaskAll);
      ps.texld(rd(0), v(0, swz(2, 3, 2, 3)), 0).op1(DxsoOpcode::Mov, oC0(), r(0));
      expectInference(ps.infer(), "tc=0 samples=1 comps=(2,3) sem=[AUX LIGHTMAP] expr=[UVXFORM MASK COLOR REACHESOC0]", "lightmap");
    }

    // UE3's TextureSample unpack, `t * 2 - 1`, then normalize().
    static void test_normalDecode() {
      std::cout << "  test_normalDecode" << std::endl;
      PsShader ps;
      ps.def(10, 2.0f, -1.0f, 0.0f, 0.0f);
      ps.base();
      ps.texld(rd(0), v(0), 0)
        .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), r(0), c(10, kXXXX), c(10, kYYYY))
        .op2(DxsoOpcode::Dp3, rd(1, MaskX), r(0), r(0))
        .op2(DxsoOpcode::Mul, oC0(), r(1, kXXXX), c(11));
      expectInference(ps.infer(), "tc=0 samples=1 comps=(0,1) sem=[] expr=[COLOR NORMAL]", "normal decode");
    }

    // The base pass modulates only diffuse by the lightmap.
    static void test_diffuseAnchor() {
      std::cout << "  test_diffuseAnchor" << std::endl;
      PsShader ps;
      ps.ctab({ { "Texture2D_0", kD3dxRegisterSetSampler, 0 }, { "LightMapTextures", kD3dxRegisterSetSampler, 1 } });
      ps.base(MaskAll);
      ps.dclSampler(1)
        .texld(rd(0), v(0), 0)
        .texld(rd(1), v(0, swz(2, 3, 2, 3)), 1)
        .op2(DxsoOpcode::Mul, rd(0), r(0), r(1))
        .op1(DxsoOpcode::Mov, oC0(), r(0));
      expectInference(ps.infer(0), "tc=0 samples=1 comps=(0,1) sem=[MAT] expr=[COLOR REACHESOC0 ANCHOR]", "diffuse anchor");
    }

    // The runtime skips samplers neither declared nor named, which relies on this.
    static void test_undeclaredSamplerIsDefault() {
      std::cout << "  test_undeclaredSamplerIsDefault" << std::endl;
      PsShader ps;
      ps.ctab({ { "Texture2D_0", kD3dxRegisterSetSampler, 0 } });
      ps.base();
      ps.texld(rd(0), v(0), 0).op1(DxsoOpcode::Mov, oC0(), r(0));
      expectInference(ps.infer(5), describeInference(PsSamplerTexcoordInference()), "undeclared sampler");
    }

    static void test_rejectsMissingInputs() {
      std::cout << "  test_rejectsMissingInputs" << std::endl;
      PsShader ps;
      ps.base();
      ps.texld(rd(0), v(0), 0).op1(DxsoOpcode::Mov, oC0(), r(0));
      DxsoShaderView view = ps.view();
      view.isgn = nullptr;
      expectInference(inferPixelShaderTexcoordForSampler(view, 0), describeInference(PsSamplerTexcoordInference()), "no signature");
      expectInference(inferPixelShaderTexcoordForSampler(ps.view(), kDxsoMaxPsSamplers),
                      describeInference(PsSamplerTexcoordInference()), "sampler out of range");
    }

    static void test_lightingInputRule() {
      std::cout << "  test_lightingInputRule" << std::endl;
      PsSamplerTexcoordInference inferred;
      inferred.sampleCount = 1;
      inferred.expressionFlags = kPsSamplerExprReachesOutputColor;
      if (isUe3LightingInputSampler(inferred)) {
        throw std::runtime_error("colour sampler taken for a lighting input");
      }
      inferred.expressionFlags = kPsSamplerExprNormalDecode;
      if (!isUe3LightingInputSampler(inferred)) {
        throw std::runtime_error("normal map not taken for a lighting input");
      }
      inferred.expressionFlags = kPsSamplerExprNormalDecode | kPsSamplerExprDiffuseAnchor;
      if (isUe3LightingInputSampler(inferred)) {
        throw std::runtime_error("diffuse-anchored sampler taken for a lighting input");
      }
      inferred.expressionFlags = 0;
      if (!isUe3LightingInputSampler(inferred)) {
        throw std::runtime_error("sampler never reaching oC0 not taken for a lighting input");
      }
    }
  };

} // anonymous namespace

int main() {
  try {
    SamplerInferenceTestApp::run();
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
