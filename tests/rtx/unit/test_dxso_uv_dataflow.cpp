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

// UV dataflow analysis: the interpolant a pixel shader sampler's coordinate comes from and the
// affine chain applied to it, and how a vertex shader texcoord output derives from the input
// assembler. The hand-assembled cases follow the shapes UE3's material compiler emits.
//
// Dump mode: test_dxso_uv_dataflow <shader.dxso> [...]  prints the instruction-level trace of a
// dumped pixel shader (DXVK_SHADER_DUMP_PATH), or the origin of each texcoord output of a
// dumped vertex shader.

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../../src/dxso/dxso_uv_dataflow.h"
#include "../../../src/util/log/log.h"
#include "../../../src/util/util_string.h"

#include "dxso_test_assembler.h"

namespace dxvk {
  // Standalone executables own the logger singleton, as the d3d9/dxgi entry points do.
  Logger Logger::s_instance("test_dxso_uv_dataflow.log", LogLevel::None);
}

using namespace dxvk;
using namespace dxvk::dxso_test;

namespace {

  std::string describeOrigin(const PsSamplerUvOrigin& o) {
    return str::format(
      "origin=", o.originValid ? 1 : 0,
      " sem=", uint32_t(o.semanticIndex),
      " comps=(", uint32_t(o.compU), ",", uint32_t(o.compV), ")",
      " sites=", o.validSiteCount, "/", o.invalidSiteCount,
      " agree=", o.sitesAgree ? 1 : 0,
      " preferHF=", o.preferredHighestFrequencySite ? 1 : 0,
      " exact=", o.affineExact ? 1 : 0,
      " U=[", formatUvComponentAffine(o.affineU), "]",
      " V=[", formatUvComponentAffine(o.affineV), "]");
  }

  const char* traceKindName(Ue3VsUvTraceKind kind) {
    switch (kind) {
    case Ue3VsUvTraceKind::PureMove: return "PureMove";
    case Ue3VsUvTraceKind::AffineConst: return "AffineConst";
    case Ue3VsUvTraceKind::OriginOnly: return "OriginOnly";
    default: return "Invalid";
    }
  }

  std::string describeVsTrace(const Ue3VsTexcoordTraceResult& t) {
    if (t.kind == Ue3VsUvTraceKind::Invalid) {
      return "Invalid";
    }
    return str::format(traceKindName(t.kind), " ia=", uint32_t(t.iaTexcoordIndex), " v", uint32_t(t.inputReg),
                       " U=[", formatUvComponentAffine(t.affineU), "] V=[", formatUvComponentAffine(t.affineV), "]");
  }

  void expectString(const std::string& actual, const std::string& expected, const char* label) {
    if (actual != expected) {
      std::cerr << label << ":\n  expected " << expected << "\n  actual   " << actual << std::endl;
      throw std::runtime_error(label);
    }
  }

  // ps_3_0 reading TEXCOORD0 through v0, sampling s0.
  class PsShader : public DxsoTestShader {
  public:
    explicit PsShader(uint32_t texcoordMask = MaskXY) : DxsoTestShader(kPs30Header) {
      dclInput(DxsoUsage::Texcoord, 0, 0, texcoordMask);
      dclSampler(0);
    }

    std::string origin() {
      std::array<PsSamplerUvOrigin, kDxsoMaxPsSamplers> origins;
      analyzePsSamplerUvOrigins(view(), origins);
      return describeOrigin(origins[0]);
    }
  };

  // vs_3_0 passing IA texcoord set 0 (v1) to the TEXCOORD0 output o1.
  class VsShader : public DxsoTestShader {
  public:
    VsShader() : DxsoTestShader(kVs30Header) {
      dcl(DxsoUsage::Position, 0, kRegInput, 0);
      dcl(DxsoUsage::Texcoord, 0, kRegInput, 1);
      dcl(DxsoUsage::Position, 0, kRegOutput, 0);
      dcl(DxsoUsage::Texcoord, 0, kRegOutput, 1, MaskXY);
    }

    std::string trace() {
      return describeVsTrace(traceVsOutputTexcoordToInputUsageIndex(view(), 1, 0, 1));
    }
  };

  // A Rotator's centre (0.5, 0.5) as fxc folds it into a literal.
  void appendRotatorCentre(PsShader& ps) {
    ps.def(10, -0.5f, -0.5f, 0.5f, 0.5f)
      .op2(DxsoOpcode::Add, rd(1, MaskXY), v(0), c(10));
  }

  class UvDataflowTestApp {
  public:
    static void run() {
      std::cout << std::endl << "Begin DXSO UV dataflow tests" << std::endl;
      test_plainInterpolant();
      test_constantTiling();
      test_selfAdd();
      test_panner();
      test_rotatorDp2add();
      test_rotatorMulMad();
      test_highestFrequencyTilingWins();
      test_flowControlDropsDerivedCoordinates();
      test_flowControlKeepsInterpolantReads();
      test_secondaryHalfComponents();
      test_rotatorOnSecondaryHalf();
      test_mixedHalves();
      test_vsPureMove();
      test_vsAffine();
      test_vsOriginOnly();
      test_vsOutputRegisterLookup();
      std::cout << "All DXSO UV dataflow tests passed" << std::endl;
    }

  private:
    static void test_plainInterpolant() {
      std::cout << "  test_plainInterpolant" << std::endl;
      PsShader ps;
      ps.texld(rd(0), v(0), 0).op1(DxsoOpcode::Mov, oC0(), r(0));
      expectString(ps.origin(),
                   "origin=1 sem=0 comps=(0,1) sites=1/0 agree=1 preferHF=0 exact=1 U=[uv*1+0] V=[uv*1+0]",
                   "plain interpolant");
    }

    // TextureCoordinate tiling: the coordinate times a material uniform.
    static void test_constantTiling() {
      std::cout << "  test_constantTiling" << std::endl;
      PsShader ps;
      ps.op2(DxsoOpcode::Mul, rd(1, MaskXY), v(0), c(3))
        .texld(rd(0), r(1), 0)
        .op1(DxsoOpcode::Mov, oC0(), r(0));
      expectString(ps.origin(),
                   "origin=1 sem=0 comps=(0,1) sites=1/0 agree=1 preferHF=0 exact=1 U=[uv*c3.x+0] V=[uv*c3.y+0]",
                   "constant tiling");
    }

    static void test_selfAdd() {
      std::cout << "  test_selfAdd" << std::endl;
      PsShader ps;
      ps.op2(DxsoOpcode::Add, rd(1, MaskXY), v(0), v(0))
        .texld(rd(0), r(1), 0)
        .op1(DxsoOpcode::Mov, oC0(), r(0));
      expectString(ps.origin(),
                   "origin=1 sem=0 comps=(0,1) sites=1/0 agree=1 preferHF=0 exact=1 U=[uv*2+0] V=[uv*2+0]",
                   "add r1, v0, v0");
    }

    // A Panner's Time * Speed arrives as one uniform added to the coordinate.
    static void test_panner() {
      std::cout << "  test_panner" << std::endl;
      PsShader ps;
      ps.op2(DxsoOpcode::Add, rd(1, MaskXY), v(0), c(4))
        .texld(rd(0), r(1), 0)
        .op1(DxsoOpcode::Mov, oC0(), r(0));
      expectString(ps.origin(),
                   "origin=1 sem=0 comps=(0,1) sites=1/0 agree=1 preferHF=0 exact=1 U=[uv*1+c4.x] V=[uv*1+c4.y]",
                   "panner");
    }

    // A Rotator's matrix rows arrive in c4 and c5; fxc applies them with dp2add.
    static void test_rotatorDp2add() {
      std::cout << "  test_rotatorDp2add" << std::endl;
      PsShader ps;
      appendRotatorCentre(ps);
      ps.op3(DxsoOpcode::Dp2Add, rd(2, MaskX), r(1), c(4), c(10, kZZZZ))
        .op3(DxsoOpcode::Dp2Add, rd(2, MaskY), r(1), c(5), c(10, kWWWW))
        .texld(rd(0), r(2), 0)
        .op1(DxsoOpcode::Mov, oC0(), r(0));
      expectString(ps.origin(),
                   "origin=1 sem=0 comps=(0,1) sites=1/0 agree=1 preferHF=0 exact=1"
                   " U=[uv.x*c4.x+uv.y*c4.y+0.5+c4.x*-0.5+c4.y*-0.5]"
                   " V=[uv.x*c5.x+uv.y*c5.y+0.5+c5.x*-0.5+c5.y*-0.5]",
                   "rotator, dp2add form");
    }

    // The same rotation in the expanded mul/mad form fxc sometimes emits instead.
    static void test_rotatorMulMad() {
      std::cout << "  test_rotatorMulMad" << std::endl;
      PsShader ps;
      appendRotatorCentre(ps);
      ps.op2(DxsoOpcode::Mul, rd(2, MaskX), r(1, kXXXX), c(4, kXXXX))
        .op3(DxsoOpcode::Mad, rd(2, MaskX), r(1, kYYYY), c(4, kYYYY), r(2, kXXXX))
        .op2(DxsoOpcode::Add, rd(2, MaskX), r(2, kXXXX), c(10, kZZZZ))
        .op2(DxsoOpcode::Mul, rd(2, MaskY), r(1, kXXXX), c(5, kXXXX))
        .op3(DxsoOpcode::Mad, rd(2, MaskY), r(1, kYYYY), c(5, kYYYY), r(2, kYYYY))
        .op2(DxsoOpcode::Add, rd(2, MaskY), r(2, kYYYY), c(10, kWWWW))
        .texld(rd(0), r(2), 0)
        .op1(DxsoOpcode::Mov, oC0(), r(0));
      // The same rotation, with terms in the order the instructions accumulate them.
      expectString(ps.origin(),
                   "origin=1 sem=0 comps=(0,1) sites=1/0 agree=1 preferHF=0 exact=1"
                   " U=[uv.y*c4.y+uv.x*c4.x+0.5+c4.y*-0.5+c4.x*-0.5]"
                   " V=[uv.y*c5.y+uv.x*c5.x+0.5+c5.y*-0.5+c5.x*-0.5]",
                   "rotator, mul/mad form");
    }

    // UE3's distance-fade anti-tiling samples one texture at two literal tilings; the higher
    // frequency is the surface's mapping, whichever site comes first.
    static void test_highestFrequencyTilingWins() {
      std::cout << "  test_highestFrequencyTilingWins" << std::endl;
      for (const bool fineFirst : { false, true }) {
        PsShader ps;
        ps.def(10, 2.0f, 2.0f, 0.0f, 0.0f)
          .def(11, 8.0f, 8.0f, 0.0f, 0.0f)
          .op2(DxsoOpcode::Mul, rd(1, MaskXY), v(0), c(fineFirst ? 11 : 10))
          .texld(rd(2), r(1), 0)
          .op2(DxsoOpcode::Mul, rd(1, MaskXY), v(0), c(fineFirst ? 10 : 11))
          .texld(rd(3), r(1), 0)
          .op3(DxsoOpcode::Lrp, rd(0), c(12, kXXXX), r(3), r(2))
          .op1(DxsoOpcode::Mov, oC0(), r(0));
        expectString(ps.origin(),
                     "origin=1 sem=0 comps=(0,1) sites=2/0 agree=0 preferHF=1 exact=1 U=[uv*8+0] V=[uv*8+0]",
                     fineFirst ? "anti-tiling, fine site first" : "anti-tiling, coarse site first");
      }
    }

    // A temp carried across a flow-control boundary may hold either branch's value.
    static void test_flowControlDropsDerivedCoordinates() {
      std::cout << "  test_flowControlDropsDerivedCoordinates" << std::endl;
      PsShader ps;
      ps.op2(DxsoOpcode::Mul, rd(1, MaskXY), v(0), c(3))
        .ifBool(0)
        .texld(rd(0), r(1), 0)
        .endIf()
        .op1(DxsoOpcode::Mov, oC0(), r(0));
      expectString(ps.origin(),
                   "origin=0 sem=0 comps=(0,1) sites=0/1 agree=1 preferHF=0 exact=0 U=[uv*1+0] V=[uv*1+0]",
                   "derived coordinate across a branch");
    }

    static void test_flowControlKeepsInterpolantReads() {
      std::cout << "  test_flowControlKeepsInterpolantReads" << std::endl;
      PsShader ps;
      ps.ifBool(0)
        .texld(rd(0), v(0), 0)
        .endIf()
        .op1(DxsoOpcode::Mov, oC0(), r(0));
      expectString(ps.origin(),
                   "origin=1 sem=0 comps=(0,1) sites=1/0 agree=1 preferHF=0 exact=1 U=[uv*1+0] V=[uv*1+0]",
                   "interpolant read inside a branch");
    }

    // UE3 packs a second UV set into an interpolant's .zw; a plain read keeps its components.
    static void test_secondaryHalfComponents() {
      std::cout << "  test_secondaryHalfComponents" << std::endl;
      PsShader ps(MaskAll);
      ps.texld(rd(0), v(0, swz(2, 3, 2, 3)), 0).op1(DxsoOpcode::Mov, oC0(), r(0));
      expectString(ps.origin(),
                   "origin=1 sem=0 comps=(2,3) sites=1/0 agree=1 preferHF=0 exact=1 U=[uv*1+0] V=[uv*1+0]",
                   ".zw read");
    }

    // Component-mixing math loses the exact pair; a site reading only .zw is attributed (3,2),
    // the convention vertex capture and albedo scoring share.
    static void test_rotatorOnSecondaryHalf() {
      std::cout << "  test_rotatorOnSecondaryHalf" << std::endl;
      PsShader ps(MaskAll);
      ps.op3(DxsoOpcode::Dp2Add, rd(2, MaskX), v(0, swz(2, 3, 2, 3)), c(4), c(6, kXXXX))
        .op3(DxsoOpcode::Dp2Add, rd(2, MaskY), v(0, swz(2, 3, 2, 3)), c(5), c(6, kYYYY))
        .texld(rd(0), r(2), 0)
        .op1(DxsoOpcode::Mov, oC0(), r(0));
      expectString(ps.origin(),
                   "origin=1 sem=0 comps=(3,2) sites=1/0 agree=1 preferHF=0 exact=1"
                   " U=[uv.z*c4.x+uv.w*c4.y+c6.x] V=[uv.z*c5.x+uv.w*c5.y+c6.y]",
                   "rotator on .zw");
    }

    // Mixing across both halves keeps the origin, attributed to the primary pair, but not the affine.
    static void test_mixedHalves() {
      std::cout << "  test_mixedHalves" << std::endl;
      PsShader ps(MaskAll);
      ps.op2(DxsoOpcode::Add, rd(2, MaskXY), v(0), v(0, swz(2, 3, 2, 3)))
        .texld(rd(0), r(2), 0)
        .op1(DxsoOpcode::Mov, oC0(), r(0));
      expectString(ps.origin(),
                   "origin=1 sem=0 comps=(0,1) sites=1/0 agree=1 preferHF=0 exact=0"
                   " U=[uv*1(INEXACT)+0(INEXACT)] V=[uv*1(INEXACT)+0(INEXACT)]",
                   ".xy + .zw");
    }

    static void test_vsPureMove() {
      std::cout << "  test_vsPureMove" << std::endl;
      VsShader vs;
      vs.op1(DxsoOpcode::Mov, od(1, MaskXY), v(1));
      expectString(vs.trace(), "PureMove ia=0 v1 U=[uv*1+0] V=[uv*1+0]", "vs pure move");
    }

    static void test_vsAffine() {
      std::cout << "  test_vsAffine" << std::endl;
      VsShader vs;
      vs.op3(DxsoOpcode::Mad, od(1, MaskXY), v(1), c(4), c(5));
      expectString(vs.trace(), "AffineConst ia=0 v1 U=[uv*c4.x+c5.x] V=[uv*c4.y+c5.y]", "vs affine");
    }

    // A swizzled copy proves the IA set, but not as a float2 stream the surface can use.
    static void test_vsOriginOnly() {
      std::cout << "  test_vsOriginOnly" << std::endl;
      VsShader vs;
      vs.op1(DxsoOpcode::Mov, od(1, MaskXY), v(1, swz(1, 0, 2, 3)));
      expectString(vs.trace(), "OriginOnly ia=0 v1 U=[uv*1+0] V=[uv*1+0]", "vs swizzled move");
    }

    static void test_vsOutputRegisterLookup() {
      std::cout << "  test_vsOutputRegisterLookup" << std::endl;
      VsShader vs;
      const DxsoShaderView view = vs.view();
      if (findVsTexcoordOutputRegister(*view.osgn, 0) != 1 ||
          findVsTexcoordOutputRegister(*view.osgn, 3) != UINT32_MAX) {
        throw std::runtime_error("vs texcoord output lookup");
      }
    }
  };

  int dumpShader(const std::string& path) {
    DumpedShader shader;
    if (!loadDumpedShader(path, shader)) {
      std::cerr << path << ": not a shader token stream" << std::endl;
      return -1;
    }
    std::cout << "== " << path << "\n";

    if (shader.info.type() == DxsoProgramTypes::PixelShader) {
      std::array<PsSamplerUvOrigin, kDxsoMaxPsSamplers> origins;
      std::vector<std::string> trace;
      analyzePsSamplerUvOrigins(shader.view(), origins, &trace);
      for (const std::string& line : trace) {
        std::cout << line << "\n";
      }
      return 0;
    }

    for (uint32_t i = 0; i < shader.osgn.elemCount; i++) {
      const DxsoIsgnEntry& e = shader.osgn.elems[i];
      if (e.semantic.usage != DxsoUsage::Texcoord) {
        continue;
      }
      for (const auto& [compU, compV] : { std::pair<uint8_t, uint8_t>(0, 1), std::pair<uint8_t, uint8_t>(2, 3) }) {
        const Ue3VsTexcoordTraceResult t = traceVsOutputTexcoordToInputUsageIndex(shader.view(), e.regNumber, compU, compV);
        std::cout << "   TEXCOORD" << e.semantic.usageIndex << " o" << e.regNumber << "."
                  << "xyzw"[compU] << "xyzw"[compV] << ": " << describeVsTrace(t) << "\n";
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
    UvDataflowTestApp::run();
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
