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

// Exercises the vertex shader oPos transform detection that exact vertex capture depends
// on (DxsoAnalyzer / DxsoPreProjectionPositionInfo). Bytecode is assembled by hand here
// rather than compiled with fxc so the tests pin the exact instruction encodings the
// detector claims to accept, and so the rejection cases can be expressed at all.

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../../../src/dxso/dxso_analysis.h"
#include "../../../src/dxso/dxso_code.h"
#include "../../../src/dxso/dxso_decoder.h"
#include "../../../src/util/log/log.h"

#include "dxso_test_assembler.h"

namespace dxvk {
  // Standalone executables own the logger singleton, as the d3d9/dxgi entry points do.
  Logger Logger::s_instance("test_dxso_preproj_position.log", LogLevel::None);
}

using namespace dxvk;
using namespace dxvk::dxso_test;

namespace {

  class ShaderBuilder : public DxsoTestShader {
  public:
    ShaderBuilder() : DxsoTestShader(kVs30Header) { }

    // Mirrors DxsoModule::runAnalyzer, minus the module/compiler plumbing the analyzer
    // itself does not need.
    DxsoPreProjectionPositionInfo analyze() const {
      const std::vector<uint32_t> bytecode = tokens();

      DxsoAnalysisInfo analysis;
      DxsoAnalyzer analyzer(info(), analysis);

      DxsoDecodeContext decoder(info());
      DxsoCodeIter iter(bytecode.data() + 1); // skip the version header token

      while (decoder.decodeInstruction(iter)) {
        analyzer.processInstruction(decoder.getInstructionContext());
      }

      analyzer.finalize(bytecode.size());
      return analysis.preProjPosition;
    }
  };

  void expectMatch(const DxsoPreProjectionPositionInfo& info,
                   uint32_t expectedSourceReg,
                   uint32_t expectedMatrixConstBase,
                   uint32_t expectedSnapshotIdx,
                   const char* label,
                   DxsoRegisterType expectedSourceType = DxsoRegisterType::Temp) {
    if (!info.valid) {
      std::cerr << label << ": expected a match, got none (reason=" << info.failureReason
                << ", math=" << info.positionDefinitions << ")" << std::endl;
      throw std::runtime_error(label);
    }
    if (info.sourceReg.type != expectedSourceType || info.sourceReg.num != expectedSourceReg) {
      std::cerr << label << ": expected source type " << uint32_t(expectedSourceType)
                << " num " << expectedSourceReg
                << ", got type " << uint32_t(info.sourceReg.type) << " num " << info.sourceReg.num << std::endl;
      throw std::runtime_error(label);
    }
    if (info.matrixConstBase != expectedMatrixConstBase) {
      std::cerr << label << ": expected matrix base c" << expectedMatrixConstBase
                << ", got c" << info.matrixConstBase << std::endl;
      throw std::runtime_error(label);
    }
    if (info.snapshotInstructionIdx != expectedSnapshotIdx) {
      std::cerr << label << ": expected snapshot at instruction " << expectedSnapshotIdx
                << ", got " << info.snapshotInstructionIdx << std::endl;
      throw std::runtime_error(label);
    }
  }

  void expectNoMatch(const DxsoPreProjectionPositionInfo& info, const char* label) {
    if (info.valid) {
      std::cerr << label << ": expected no match, got source r" << info.sourceReg.num
                << " matrix c" << info.matrixConstBase << std::endl;
      throw std::runtime_error(label);
    }
    if (info.failureReason == nullptr || info.failureReason[0] == '\0') {
      std::cerr << label << ": rejection carried no reason for the log" << std::endl;
      throw std::runtime_error(label);
    }
  }

  // Filler that touches unrelated registers, standing in for the fog/tangent/texcoord work
  // a compiler interleaves through the position transform.
  void appendFiller(ShaderBuilder& shader, uint32_t tempNum) {
    shader.op2(DxsoOpcode::Add, dst(kRegTemp, tempNum, MaskAll),
               src(kRegTemp, tempNum, kXYZW),
               src(kRegConst, 40, kXYZW));
  }

  // mul r1, r0.x, c0 / mad r1, r0.y, c1, r1 / mad r1, r0.z, c2, r1 / mad <dst>, r0.w, c3, r1
  // This is the shape fxc emits for mul(matrix, vector) with column-major packing, which is
  // what UE3's PC vertex factories compile to. `interleave` inserts unrelated instructions
  // between the steps, and `accumTemps` walks the accumulator across separate temps.
  void appendMadChain(ShaderBuilder& shader,
                      uint32_t constBase,
                      uint32_t finalDst,
                      uint32_t sourceType = kRegTemp,
                      uint32_t sourceNum = 0,
                      bool interleave = false,
                      bool accumTemps = false) {
    const uint32_t swizzles[4] = { kXXXX, kYYYY, kZZZZ, kWWWW };

    uint32_t accumNum = 1;
    shader.op2(DxsoOpcode::Mul, dst(kRegTemp, accumNum, MaskAll),
               src(sourceType, sourceNum, swizzles[0]),
               src(kRegConst, constBase + 0, kXYZW));

    for (uint32_t step = 1; step < 4; step++) {
      if (interleave) {
        appendFiller(shader, 9);
      }

      const uint32_t accumSrc = src(kRegTemp, accumNum, kXYZW);
      if (accumTemps) {
        accumNum++;
      }
      const uint32_t stepDst = step == 3 ? finalDst : dst(kRegTemp, accumNum, MaskAll);

      shader.op3(DxsoOpcode::Mad, stepDst,
                 src(sourceType, sourceNum, swizzles[step]),
                 src(kRegConst, constBase + step, kXYZW), accumSrc);
    }
  }

  // dp4 <dst>.x, r0, c0 ... dp4 <dst>.w, r0, c3, optionally in a different component order
  // and with unrelated instructions between the steps.
  void appendDp4Group(ShaderBuilder& shader,
                      uint32_t constBase,
                      uint32_t dstType,
                      uint32_t dstNum,
                      const uint32_t order[4] = nullptr,
                      bool interleave = false) {
    const uint32_t masks[4] = { MaskX, MaskY, MaskZ, MaskW };
    const uint32_t identityOrder[4] = { 0, 1, 2, 3 };
    const uint32_t* components = order != nullptr ? order : identityOrder;

    for (uint32_t i = 0; i < 4; i++) {
      const uint32_t component = components[i];
      shader.op2(DxsoOpcode::Dp4, dst(dstType, dstNum, masks[component]),
                 src(kRegTemp, 0, kXYZW),
                 src(kRegConst, constBase + component, kXYZW));

      if (interleave && i + 1 < 4) {
        appendFiller(shader, 9);
      }
    }
  }

  class PreProjPositionTestApp {
  public:
    static void run() {
      std::cout << std::endl << "Begin DXSO pre-projection position tests" << std::endl;
      test_madChainToOutput();
      test_dp4GroupToRasterizerOut();
      test_m4x4();
      test_trailingMovThroughTemp();
      test_nonZeroMatrixBase();
      test_interleavedMadChain();
      test_interleavedDp4Group();
      test_permutedDp4Components();
      test_chainedAccumulatorTemps();
      test_inputRegisterSource();
      test_movChainThroughTwoTemps();
      test_fxcColumnMajorOutput();
      test_fxcRowMajorOutput();
      test_rejectsExtraPositionWrite();
      test_rejectsControlFlow();
      test_rejectsWrongConstantOrder();
      test_rejectsUndeclaredOutput();
      test_rejectsSourceRewrittenMidTransform();
      test_rejectsSwizzlingMovIntoPosition();
      std::cout << "All DXSO pre-projection position tests passed" << std::endl;
    }

  private:
    // vs_3_0 writes position through an o# register declared dcl_position.
    static void test_madChainToOutput() {
      std::cout << "  test_madChainToOutput" << std::endl;
      ShaderBuilder shader;
      shader.dcl(DxsoUsage::Position, 0, kRegOutput, 0);
      appendMadChain(shader, 0, dst(kRegOutput, 0, MaskAll));
      // instruction 1 is the dcl, so the mul that starts the chain is instruction 2
      expectMatch(shader.analyze(), 0, 0, 2, "mad chain to declared output");
    }

    // vs_1_1 / vs_2_0 write position through oPos.
    static void test_dp4GroupToRasterizerOut() {
      std::cout << "  test_dp4GroupToRasterizerOut" << std::endl;
      ShaderBuilder shader;
      appendDp4Group(shader, 0, kRegRasterizerOut, 0);
      expectMatch(shader.analyze(), 0, 0, 1, "dp4 group to oPos");
    }

    static void test_m4x4() {
      std::cout << "  test_m4x4" << std::endl;
      ShaderBuilder shader;
      shader.op2(DxsoOpcode::M4x4, dst(kRegRasterizerOut, 0, MaskAll),
                 src(kRegTemp, 0, kXYZW),
                 src(kRegConst, 0, kXYZW));
      expectMatch(shader.analyze(), 0, 0, 1, "m4x4 to oPos");
    }

    // The transform may land in a temp that is then moved into position.
    static void test_trailingMovThroughTemp() {
      std::cout << "  test_trailingMovThroughTemp" << std::endl;
      ShaderBuilder shader;
      appendDp4Group(shader, 0, kRegTemp, 2);
      shader.op1(DxsoOpcode::Mov, dst(kRegRasterizerOut, 0, MaskAll),
                 src(kRegTemp, 2, kXYZW));
      expectMatch(shader.analyze(), 0, 0, 1, "dp4 group then mov to oPos");
    }

    // The matrix is wherever the game put it; the CTAB match that proves it is
    // ViewProjectionMatrix happens on the D3D9 side, not here.
    static void test_nonZeroMatrixBase() {
      std::cout << "  test_nonZeroMatrixBase" << std::endl;
      ShaderBuilder shader;
      appendMadChain(shader, 12, dst(kRegRasterizerOut, 0, MaskAll));
      expectMatch(shader.analyze(), 0, 12, 1, "mad chain with matrix at c12");
    }

    // A compiler schedules the transform among the base pass's fog, tangent and texcoord
    // work, so the steps are generally not adjacent.
    static void test_interleavedMadChain() {
      std::cout << "  test_interleavedMadChain" << std::endl;
      ShaderBuilder shader;
      appendMadChain(shader, 0, dst(kRegRasterizerOut, 0, MaskAll),
                     kRegTemp, 0, /*interleave*/ true);
      expectMatch(shader.analyze(), 0, 0, 1, "interleaved mad chain");
    }

    static void test_interleavedDp4Group() {
      std::cout << "  test_interleavedDp4Group" << std::endl;
      ShaderBuilder shader;
      appendDp4Group(shader, 0, kRegRasterizerOut, 0, nullptr, /*interleave*/ true);
      expectMatch(shader.analyze(), 0, 0, 1, "interleaved dp4 group");
    }

    // Which component is computed first is the compiler's choice; only the pairing of
    // component to constant register carries meaning.
    static void test_permutedDp4Components() {
      std::cout << "  test_permutedDp4Components" << std::endl;
      ShaderBuilder shader;
      const uint32_t order[4] = { 2, 0, 3, 1 };
      appendDp4Group(shader, 0, kRegRasterizerOut, 0, order);
      expectMatch(shader.analyze(), 0, 0, 1, "permuted dp4 components");
    }

    static void test_chainedAccumulatorTemps() {
      std::cout << "  test_chainedAccumulatorTemps" << std::endl;
      ShaderBuilder shader;
      appendMadChain(shader, 0, dst(kRegRasterizerOut, 0, MaskAll),
                     kRegTemp, 0, /*interleave*/ false, /*accumTemps*/ true);
      expectMatch(shader.analyze(), 0, 0, 1, "accumulator across separate temps");
    }

    // A shader can transform an input assembler register directly, in which case that
    // register is what needs capturing.
    static void test_inputRegisterSource() {
      std::cout << "  test_inputRegisterSource" << std::endl;
      ShaderBuilder shader;
      shader.dcl(DxsoUsage::Position, 0, kRegInput, 0);
      appendMadChain(shader, 0, dst(kRegRasterizerOut, 0, MaskAll), kRegInput, 0);
      expectMatch(shader.analyze(), 0, 0, 2, "input register source", DxsoRegisterType::Input);
    }

    static void test_movChainThroughTwoTemps() {
      std::cout << "  test_movChainThroughTwoTemps" << std::endl;
      ShaderBuilder shader;
      appendDp4Group(shader, 0, kRegTemp, 2);
      shader.op1(DxsoOpcode::Mov, dst(kRegTemp, 3, MaskAll),
                 src(kRegTemp, 2, kXYZW));
      shader.op1(DxsoOpcode::Mov, dst(kRegRasterizerOut, 0, MaskAll),
                 src(kRegTemp, 3, kXYZW));
      expectMatch(shader.analyze(), 0, 0, 1, "dp4 group through two movs");
    }

    // Verbatim shape fxc emits for the UE3 base pass with default (column-major) matrix
    // packing. Note the chain starts on component y, the constant is the first operand of
    // the mads, the final step writes its result back over the source register, and position
    // reaches oPos through a mov whose value also feeds TEXCOORD5.
    static void test_fxcColumnMajorOutput() {
      std::cout << "  test_fxcColumnMajorOutput" << std::endl;
      ShaderBuilder shader;
      shader.dcl(DxsoUsage::Texcoord, 5, kRegOutput, 4);
      shader.dcl(DxsoUsage::Position, 0, kRegOutput, 7);

      const uint32_t accum = dst(kRegTemp, 1, MaskAll);
      const uint32_t accumSrc = src(kRegTemp, 1, kXYZW);

      // mul r1, r0.y, c1
      shader.op2(DxsoOpcode::Mul, accum,
                 src(kRegTemp, 0, kYYYY),
                 src(kRegConst, 1, kXYZW));
      // mad r1, c0, r0.x, r1
      shader.op3(DxsoOpcode::Mad, accum,
                 src(kRegConst, 0, kXYZW),
                 src(kRegTemp, 0, kXXXX), accumSrc);
      // mad r1, c2, r0.z, r1
      shader.op3(DxsoOpcode::Mad, accum,
                 src(kRegConst, 2, kXYZW),
                 src(kRegTemp, 0, kZZZZ), accumSrc);
      // mad r0, c3, r0.w, r1
      shader.op3(DxsoOpcode::Mad, dst(kRegTemp, 0, MaskAll),
                 src(kRegConst, 3, kXYZW),
                 src(kRegTemp, 0, kWWWW), accumSrc);
      // mov o4, r0  (PixelPosition) then mov o7, r0 (oPos)
      shader.op1(DxsoOpcode::Mov, dst(kRegOutput, 4, MaskAll),
                 src(kRegTemp, 0, kXYZW));
      shader.op1(DxsoOpcode::Mov, dst(kRegOutput, 7, MaskAll),
                 src(kRegTemp, 0, kXYZW));

      // instructions 1-2 are the dcls, so the mul starting the chain is instruction 3
      expectMatch(shader.analyze(), 0, 0, 3, "fxc column-major base pass output");
    }

    // Verbatim shape fxc emits with /Zpr (row-major packing): four dp4s with the constant
    // first, which in the real shader are split apart by unrelated instructions.
    static void test_fxcRowMajorOutput() {
      std::cout << "  test_fxcRowMajorOutput" << std::endl;
      ShaderBuilder shader;
      shader.dcl(DxsoUsage::Texcoord, 5, kRegOutput, 4);
      shader.dcl(DxsoUsage::Position, 0, kRegOutput, 7);

      const uint32_t masks[4] = { MaskX, MaskY, MaskZ, MaskW };
      for (uint32_t i = 0; i < 4; i++) {
        shader.op2(DxsoOpcode::Dp4, dst(kRegTemp, 0, masks[i]),
                   src(kRegConst, i, kXYZW),
                   src(kRegTemp, 1, kXYZW));
        if (i + 1 < 4) {
          appendFiller(shader, 9);
        }
      }
      shader.op1(DxsoOpcode::Mov, dst(kRegOutput, 4, MaskAll),
                 src(kRegTemp, 0, kXYZW));
      shader.op1(DxsoOpcode::Mov, dst(kRegOutput, 7, MaskAll),
                 src(kRegTemp, 0, kXYZW));

      expectMatch(shader.analyze(), 1, 0, 3, "fxc row-major base pass output");
    }

    // A shader that adjusts position after transforming it (depth bias, baked jitter) no
    // longer describes what was rasterized through the source register alone.
    static void test_rejectsExtraPositionWrite() {
      std::cout << "  test_rejectsExtraPositionWrite" << std::endl;
      ShaderBuilder shader;
      appendMadChain(shader, 0, dst(kRegRasterizerOut, 0, MaskAll));
      shader.op2(DxsoOpcode::Add, dst(kRegRasterizerOut, 0, MaskZ),
                 src(kRegRasterizerOut, 0, kZZZZ),
                 src(kRegConst, 8, kXYZW));
      expectNoMatch(shader.analyze(), "extra oPos write");
    }

    // Inside control flow the snapshot may not run for every vertex.
    static void test_rejectsControlFlow() {
      std::cout << "  test_rejectsControlFlow" << std::endl;
      ShaderBuilder shader;
      // if takes a single source and no destination, so it is emitted raw
      shader.raw(opcodeToken(DxsoOpcode::If, 1));
      shader.raw(src(kRegConst, 9, kXXXX));
      appendMadChain(shader, 0, dst(kRegRasterizerOut, 0, MaskAll));
      shader.raw(opcodeToken(DxsoOpcode::EndIf, 0));
      expectNoMatch(shader.analyze(), "transform inside control flow");
    }

    // Constants must be consecutive and aligned to the component they feed, or the register
    // is not the vector being transformed by that matrix.
    static void test_rejectsWrongConstantOrder() {
      std::cout << "  test_rejectsWrongConstantOrder" << std::endl;
      ShaderBuilder shader;
      const uint32_t accum = dst(kRegTemp, 1, MaskAll);
      const uint32_t accumSrc = src(kRegTemp, 1, kXYZW);
      shader.op2(DxsoOpcode::Mul, accum,
                 src(kRegTemp, 0, kXXXX),
                 src(kRegConst, 0, kXYZW));
      shader.op3(DxsoOpcode::Mad, accum,
                 src(kRegTemp, 0, kYYYY),
                 src(kRegConst, 5, kXYZW), accumSrc);
      shader.op3(DxsoOpcode::Mad, accum,
                 src(kRegTemp, 0, kZZZZ),
                 src(kRegConst, 2, kXYZW), accumSrc);
      shader.op3(DxsoOpcode::Mad, dst(kRegRasterizerOut, 0, MaskAll),
                 src(kRegTemp, 0, kWWWW),
                 src(kRegConst, 3, kXYZW), accumSrc);
      expectNoMatch(shader.analyze(), "non-consecutive matrix constants");
    }

    // An o# register only carries position when it was declared dcl_position.
    static void test_rejectsUndeclaredOutput() {
      std::cout << "  test_rejectsUndeclaredOutput" << std::endl;
      ShaderBuilder shader;
      shader.dcl(DxsoUsage::Texcoord, 0, kRegOutput, 0);
      appendMadChain(shader, 0, dst(kRegOutput, 0, MaskAll));
      expectNoMatch(shader.analyze(), "transform into a texcoord output");
    }

    // The snapshot is taken before the transform starts, so a source that changes while the
    // transform is still consuming it would be captured at the wrong value.
    static void test_rejectsSourceRewrittenMidTransform() {
      std::cout << "  test_rejectsSourceRewrittenMidTransform" << std::endl;
      ShaderBuilder shader;
      const uint32_t accum = dst(kRegTemp, 1, MaskAll);
      const uint32_t accumSrc = src(kRegTemp, 1, kXYZW);
      shader.op2(DxsoOpcode::Mul, accum,
                 src(kRegTemp, 0, kXXXX),
                 src(kRegConst, 0, kXYZW));
      // rewrite r0 halfway through, so the four steps do not all see the same position
      appendFiller(shader, 0);
      shader.op3(DxsoOpcode::Mad, accum,
                 src(kRegTemp, 0, kYYYY),
                 src(kRegConst, 1, kXYZW), accumSrc);
      shader.op3(DxsoOpcode::Mad, accum,
                 src(kRegTemp, 0, kZZZZ),
                 src(kRegConst, 2, kXYZW), accumSrc);
      shader.op3(DxsoOpcode::Mad, dst(kRegRasterizerOut, 0, MaskAll),
                 src(kRegTemp, 0, kWWWW),
                 src(kRegConst, 3, kXYZW), accumSrc);
      expectNoMatch(shader.analyze(), "source rewritten mid-transform");
    }

    // A permuting copy into position means the rasterized vertex is a shuffle of the
    // transform's result, so the source register does not describe where it landed.
    static void test_rejectsSwizzlingMovIntoPosition() {
      std::cout << "  test_rejectsSwizzlingMovIntoPosition" << std::endl;
      ShaderBuilder shader;
      appendMadChain(shader, 0, dst(kRegTemp, 2, MaskAll));
      shader.op1(DxsoOpcode::Mov, dst(kRegRasterizerOut, 0, MaskAll),
                 src(kRegTemp, 2, swz(1, 0, 2, 3)));
      expectNoMatch(shader.analyze(), "swizzling mov into position");
    }
  };

} // anonymous namespace

int main() {
  try {
    PreProjPositionTestApp::run();
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    throw;
  }

  return 0;
}
