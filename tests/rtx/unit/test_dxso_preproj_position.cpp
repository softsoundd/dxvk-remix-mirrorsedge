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

namespace dxvk {
  // Standalone executables own the logger singleton, as the d3d9/dxgi entry points do.
  Logger Logger::s_instance("test_dxso_preproj_position.log", LogLevel::None);
}

using namespace dxvk;

namespace {

  // D3D9 shader bytecode token encodings, from the vs_3_0 instruction format.
  constexpr uint32_t kVs30Header = 0xFFFE0300u;
  constexpr uint32_t kEndToken = 0x0000FFFFu;

  constexpr uint32_t kRegTypeTemp = 0u;
  constexpr uint32_t kRegTypeInput = 1u;
  constexpr uint32_t kRegTypeConst = 2u;
  constexpr uint32_t kRegTypeRasterizerOut = 4u;
  constexpr uint32_t kRegTypeOutput = 6u;

  enum Swizzle : uint32_t {
    SwizzleIdentity = 0xE4u, // x,y,z,w
    SwizzleXXXX = 0x00u,
    SwizzleYYYY = 0x55u,
    SwizzleZZZZ = 0xAAu,
    SwizzleWWWW = 0xFFu,
  };

  enum WriteMask : uint32_t {
    MaskX = 0x1u,
    MaskY = 0x2u,
    MaskZ = 0x4u,
    MaskW = 0x8u,
    MaskAll = 0xFu,
  };

  // The register type is split across bits 11..12 and 28..30.
  uint32_t encodeRegisterType(uint32_t type) {
    return ((type & 0x7u) << 28) | ((type & 0x18u) << 8);
  }

  uint32_t makeSwizzle(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    return x | (y << 2) | (z << 4) | (w << 6);
  }

  uint32_t destToken(uint32_t type, uint32_t num, uint32_t mask) {
    return 0x80000000u | encodeRegisterType(type) | ((mask & 0xFu) << 16) | (num & 0x7FFu);
  }

  uint32_t sourceToken(uint32_t type, uint32_t num, uint32_t swizzle) {
    return 0x80000000u | encodeRegisterType(type) | ((swizzle & 0xFFu) << 16) | (num & 0x7FFu);
  }

  uint32_t opcodeToken(DxsoOpcode opcode, uint32_t length) {
    return uint32_t(opcode) | ((length & 0xFu) << 24);
  }

  class ShaderBuilder {
  public:
    ShaderBuilder() {
      m_tokens.push_back(kVs30Header);
    }

    // dcl_<usage><index> <dst>
    ShaderBuilder& dcl(DxsoUsage usage, uint32_t usageIndex, uint32_t regType, uint32_t regNum) {
      m_tokens.push_back(opcodeToken(DxsoOpcode::Dcl, 2));
      m_tokens.push_back(0x80000000u | uint32_t(usage) | ((usageIndex & 0xFu) << 16));
      m_tokens.push_back(destToken(regType, regNum, MaskAll));
      return *this;
    }

    ShaderBuilder& op1(DxsoOpcode opcode, uint32_t dst, uint32_t src0) {
      m_tokens.push_back(opcodeToken(opcode, 2));
      m_tokens.push_back(dst);
      m_tokens.push_back(src0);
      return *this;
    }

    ShaderBuilder& op2(DxsoOpcode opcode, uint32_t dst, uint32_t src0, uint32_t src1) {
      m_tokens.push_back(opcodeToken(opcode, 3));
      m_tokens.push_back(dst);
      m_tokens.push_back(src0);
      m_tokens.push_back(src1);
      return *this;
    }

    ShaderBuilder& op3(DxsoOpcode opcode, uint32_t dst, uint32_t src0, uint32_t src1, uint32_t src2) {
      m_tokens.push_back(opcodeToken(opcode, 4));
      m_tokens.push_back(dst);
      m_tokens.push_back(src0);
      m_tokens.push_back(src1);
      m_tokens.push_back(src2);
      return *this;
    }

    ShaderBuilder& raw(uint32_t token) {
      m_tokens.push_back(token);
      return *this;
    }

    // Mirrors DxsoModule::runAnalyzer, minus the module/compiler plumbing the analyzer
    // itself does not need.
    DxsoPreProjectionPositionInfo analyze() const {
      std::vector<uint32_t> tokens = m_tokens;
      tokens.push_back(kEndToken);

      const DxsoProgramInfo programInfo(DxsoProgramTypes::VertexShader, 0, 3);

      DxsoAnalysisInfo info;
      DxsoAnalyzer analyzer(programInfo, info);

      DxsoDecodeContext decoder(programInfo);
      DxsoCodeIter iter(tokens.data() + 1); // skip the version header token

      while (decoder.decodeInstruction(iter)) {
        analyzer.processInstruction(decoder.getInstructionContext());
      }

      analyzer.finalize(tokens.size());
      return info.preProjPosition;
    }

  private:
    std::vector<uint32_t> m_tokens;
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
    shader.op2(DxsoOpcode::Add, destToken(kRegTypeTemp, tempNum, MaskAll),
               sourceToken(kRegTypeTemp, tempNum, SwizzleIdentity),
               sourceToken(kRegTypeConst, 40, SwizzleIdentity));
  }

  // mul r1, r0.x, c0 / mad r1, r0.y, c1, r1 / mad r1, r0.z, c2, r1 / mad <dst>, r0.w, c3, r1
  // This is the shape fxc emits for mul(matrix, vector) with column-major packing, which is
  // what UE3's PC vertex factories compile to. `interleave` inserts unrelated instructions
  // between the steps, and `accumTemps` walks the accumulator across separate temps.
  void appendMadChain(ShaderBuilder& shader,
                      uint32_t constBase,
                      uint32_t finalDst,
                      uint32_t sourceType = kRegTypeTemp,
                      uint32_t sourceNum = 0,
                      bool interleave = false,
                      bool accumTemps = false) {
    const Swizzle swizzles[4] = { SwizzleXXXX, SwizzleYYYY, SwizzleZZZZ, SwizzleWWWW };

    uint32_t accumNum = 1;
    shader.op2(DxsoOpcode::Mul, destToken(kRegTypeTemp, accumNum, MaskAll),
               sourceToken(sourceType, sourceNum, swizzles[0]),
               sourceToken(kRegTypeConst, constBase + 0, SwizzleIdentity));

    for (uint32_t step = 1; step < 4; step++) {
      if (interleave) {
        appendFiller(shader, 9);
      }

      const uint32_t accumSrc = sourceToken(kRegTypeTemp, accumNum, SwizzleIdentity);
      if (accumTemps) {
        accumNum++;
      }
      const uint32_t dst = step == 3 ? finalDst : destToken(kRegTypeTemp, accumNum, MaskAll);

      shader.op3(DxsoOpcode::Mad, dst,
                 sourceToken(sourceType, sourceNum, swizzles[step]),
                 sourceToken(kRegTypeConst, constBase + step, SwizzleIdentity), accumSrc);
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
      shader.op2(DxsoOpcode::Dp4, destToken(dstType, dstNum, masks[component]),
                 sourceToken(kRegTypeTemp, 0, SwizzleIdentity),
                 sourceToken(kRegTypeConst, constBase + component, SwizzleIdentity));

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
      shader.dcl(DxsoUsage::Position, 0, kRegTypeOutput, 0);
      appendMadChain(shader, 0, destToken(kRegTypeOutput, 0, MaskAll));
      // instruction 1 is the dcl, so the mul that starts the chain is instruction 2
      expectMatch(shader.analyze(), 0, 0, 2, "mad chain to declared output");
    }

    // vs_1_1 / vs_2_0 write position through oPos.
    static void test_dp4GroupToRasterizerOut() {
      std::cout << "  test_dp4GroupToRasterizerOut" << std::endl;
      ShaderBuilder shader;
      appendDp4Group(shader, 0, kRegTypeRasterizerOut, 0);
      expectMatch(shader.analyze(), 0, 0, 1, "dp4 group to oPos");
    }

    static void test_m4x4() {
      std::cout << "  test_m4x4" << std::endl;
      ShaderBuilder shader;
      shader.op2(DxsoOpcode::M4x4, destToken(kRegTypeRasterizerOut, 0, MaskAll),
                 sourceToken(kRegTypeTemp, 0, SwizzleIdentity),
                 sourceToken(kRegTypeConst, 0, SwizzleIdentity));
      expectMatch(shader.analyze(), 0, 0, 1, "m4x4 to oPos");
    }

    // The transform may land in a temp that is then moved into position.
    static void test_trailingMovThroughTemp() {
      std::cout << "  test_trailingMovThroughTemp" << std::endl;
      ShaderBuilder shader;
      appendDp4Group(shader, 0, kRegTypeTemp, 2);
      shader.op1(DxsoOpcode::Mov, destToken(kRegTypeRasterizerOut, 0, MaskAll),
                 sourceToken(kRegTypeTemp, 2, SwizzleIdentity));
      expectMatch(shader.analyze(), 0, 0, 1, "dp4 group then mov to oPos");
    }

    // The matrix is wherever the game put it; the CTAB match that proves it is
    // ViewProjectionMatrix happens on the D3D9 side, not here.
    static void test_nonZeroMatrixBase() {
      std::cout << "  test_nonZeroMatrixBase" << std::endl;
      ShaderBuilder shader;
      appendMadChain(shader, 12, destToken(kRegTypeRasterizerOut, 0, MaskAll));
      expectMatch(shader.analyze(), 0, 12, 1, "mad chain with matrix at c12");
    }

    // A compiler schedules the transform among the base pass's fog, tangent and texcoord
    // work, so the steps are generally not adjacent.
    static void test_interleavedMadChain() {
      std::cout << "  test_interleavedMadChain" << std::endl;
      ShaderBuilder shader;
      appendMadChain(shader, 0, destToken(kRegTypeRasterizerOut, 0, MaskAll),
                     kRegTypeTemp, 0, /*interleave*/ true);
      expectMatch(shader.analyze(), 0, 0, 1, "interleaved mad chain");
    }

    static void test_interleavedDp4Group() {
      std::cout << "  test_interleavedDp4Group" << std::endl;
      ShaderBuilder shader;
      appendDp4Group(shader, 0, kRegTypeRasterizerOut, 0, nullptr, /*interleave*/ true);
      expectMatch(shader.analyze(), 0, 0, 1, "interleaved dp4 group");
    }

    // Which component is computed first is the compiler's choice; only the pairing of
    // component to constant register carries meaning.
    static void test_permutedDp4Components() {
      std::cout << "  test_permutedDp4Components" << std::endl;
      ShaderBuilder shader;
      const uint32_t order[4] = { 2, 0, 3, 1 };
      appendDp4Group(shader, 0, kRegTypeRasterizerOut, 0, order);
      expectMatch(shader.analyze(), 0, 0, 1, "permuted dp4 components");
    }

    static void test_chainedAccumulatorTemps() {
      std::cout << "  test_chainedAccumulatorTemps" << std::endl;
      ShaderBuilder shader;
      appendMadChain(shader, 0, destToken(kRegTypeRasterizerOut, 0, MaskAll),
                     kRegTypeTemp, 0, /*interleave*/ false, /*accumTemps*/ true);
      expectMatch(shader.analyze(), 0, 0, 1, "accumulator across separate temps");
    }

    // A shader can transform an input assembler register directly, in which case that
    // register is what needs capturing.
    static void test_inputRegisterSource() {
      std::cout << "  test_inputRegisterSource" << std::endl;
      ShaderBuilder shader;
      shader.dcl(DxsoUsage::Position, 0, kRegTypeInput, 0);
      appendMadChain(shader, 0, destToken(kRegTypeRasterizerOut, 0, MaskAll), kRegTypeInput, 0);
      expectMatch(shader.analyze(), 0, 0, 2, "input register source", DxsoRegisterType::Input);
    }

    static void test_movChainThroughTwoTemps() {
      std::cout << "  test_movChainThroughTwoTemps" << std::endl;
      ShaderBuilder shader;
      appendDp4Group(shader, 0, kRegTypeTemp, 2);
      shader.op1(DxsoOpcode::Mov, destToken(kRegTypeTemp, 3, MaskAll),
                 sourceToken(kRegTypeTemp, 2, SwizzleIdentity));
      shader.op1(DxsoOpcode::Mov, destToken(kRegTypeRasterizerOut, 0, MaskAll),
                 sourceToken(kRegTypeTemp, 3, SwizzleIdentity));
      expectMatch(shader.analyze(), 0, 0, 1, "dp4 group through two movs");
    }

    // Verbatim shape fxc emits for the UE3 base pass with default (column-major) matrix
    // packing. Note the chain starts on component y, the constant is the first operand of
    // the mads, the final step writes its result back over the source register, and position
    // reaches oPos through a mov whose value also feeds TEXCOORD5.
    static void test_fxcColumnMajorOutput() {
      std::cout << "  test_fxcColumnMajorOutput" << std::endl;
      ShaderBuilder shader;
      shader.dcl(DxsoUsage::Texcoord, 5, kRegTypeOutput, 4);
      shader.dcl(DxsoUsage::Position, 0, kRegTypeOutput, 7);

      const uint32_t accum = destToken(kRegTypeTemp, 1, MaskAll);
      const uint32_t accumSrc = sourceToken(kRegTypeTemp, 1, SwizzleIdentity);

      // mul r1, r0.y, c1
      shader.op2(DxsoOpcode::Mul, accum,
                 sourceToken(kRegTypeTemp, 0, SwizzleYYYY),
                 sourceToken(kRegTypeConst, 1, SwizzleIdentity));
      // mad r1, c0, r0.x, r1
      shader.op3(DxsoOpcode::Mad, accum,
                 sourceToken(kRegTypeConst, 0, SwizzleIdentity),
                 sourceToken(kRegTypeTemp, 0, SwizzleXXXX), accumSrc);
      // mad r1, c2, r0.z, r1
      shader.op3(DxsoOpcode::Mad, accum,
                 sourceToken(kRegTypeConst, 2, SwizzleIdentity),
                 sourceToken(kRegTypeTemp, 0, SwizzleZZZZ), accumSrc);
      // mad r0, c3, r0.w, r1
      shader.op3(DxsoOpcode::Mad, destToken(kRegTypeTemp, 0, MaskAll),
                 sourceToken(kRegTypeConst, 3, SwizzleIdentity),
                 sourceToken(kRegTypeTemp, 0, SwizzleWWWW), accumSrc);
      // mov o4, r0  (PixelPosition) then mov o7, r0 (oPos)
      shader.op1(DxsoOpcode::Mov, destToken(kRegTypeOutput, 4, MaskAll),
                 sourceToken(kRegTypeTemp, 0, SwizzleIdentity));
      shader.op1(DxsoOpcode::Mov, destToken(kRegTypeOutput, 7, MaskAll),
                 sourceToken(kRegTypeTemp, 0, SwizzleIdentity));

      // instructions 1-2 are the dcls, so the mul starting the chain is instruction 3
      expectMatch(shader.analyze(), 0, 0, 3, "fxc column-major base pass output");
    }

    // Verbatim shape fxc emits with /Zpr (row-major packing): four dp4s with the constant
    // first, which in the real shader are split apart by unrelated instructions.
    static void test_fxcRowMajorOutput() {
      std::cout << "  test_fxcRowMajorOutput" << std::endl;
      ShaderBuilder shader;
      shader.dcl(DxsoUsage::Texcoord, 5, kRegTypeOutput, 4);
      shader.dcl(DxsoUsage::Position, 0, kRegTypeOutput, 7);

      const uint32_t masks[4] = { MaskX, MaskY, MaskZ, MaskW };
      for (uint32_t i = 0; i < 4; i++) {
        shader.op2(DxsoOpcode::Dp4, destToken(kRegTypeTemp, 0, masks[i]),
                   sourceToken(kRegTypeConst, i, SwizzleIdentity),
                   sourceToken(kRegTypeTemp, 1, SwizzleIdentity));
        if (i + 1 < 4) {
          appendFiller(shader, 9);
        }
      }
      shader.op1(DxsoOpcode::Mov, destToken(kRegTypeOutput, 4, MaskAll),
                 sourceToken(kRegTypeTemp, 0, SwizzleIdentity));
      shader.op1(DxsoOpcode::Mov, destToken(kRegTypeOutput, 7, MaskAll),
                 sourceToken(kRegTypeTemp, 0, SwizzleIdentity));

      expectMatch(shader.analyze(), 1, 0, 3, "fxc row-major base pass output");
    }

    // A shader that adjusts position after transforming it (depth bias, baked jitter) no
    // longer describes what was rasterized through the source register alone.
    static void test_rejectsExtraPositionWrite() {
      std::cout << "  test_rejectsExtraPositionWrite" << std::endl;
      ShaderBuilder shader;
      appendMadChain(shader, 0, destToken(kRegTypeRasterizerOut, 0, MaskAll));
      shader.op2(DxsoOpcode::Add, destToken(kRegTypeRasterizerOut, 0, MaskZ),
                 sourceToken(kRegTypeRasterizerOut, 0, SwizzleZZZZ),
                 sourceToken(kRegTypeConst, 8, SwizzleIdentity));
      expectNoMatch(shader.analyze(), "extra oPos write");
    }

    // Inside control flow the snapshot may not run for every vertex.
    static void test_rejectsControlFlow() {
      std::cout << "  test_rejectsControlFlow" << std::endl;
      ShaderBuilder shader;
      // if takes a single source and no destination, so it is emitted raw
      shader.raw(opcodeToken(DxsoOpcode::If, 1));
      shader.raw(sourceToken(kRegTypeConst, 9, SwizzleXXXX));
      appendMadChain(shader, 0, destToken(kRegTypeRasterizerOut, 0, MaskAll));
      shader.raw(opcodeToken(DxsoOpcode::EndIf, 0));
      expectNoMatch(shader.analyze(), "transform inside control flow");
    }

    // Constants must be consecutive and aligned to the component they feed, or the register
    // is not the vector being transformed by that matrix.
    static void test_rejectsWrongConstantOrder() {
      std::cout << "  test_rejectsWrongConstantOrder" << std::endl;
      ShaderBuilder shader;
      const uint32_t accum = destToken(kRegTypeTemp, 1, MaskAll);
      const uint32_t accumSrc = sourceToken(kRegTypeTemp, 1, SwizzleIdentity);
      shader.op2(DxsoOpcode::Mul, accum,
                 sourceToken(kRegTypeTemp, 0, SwizzleXXXX),
                 sourceToken(kRegTypeConst, 0, SwizzleIdentity));
      shader.op3(DxsoOpcode::Mad, accum,
                 sourceToken(kRegTypeTemp, 0, SwizzleYYYY),
                 sourceToken(kRegTypeConst, 5, SwizzleIdentity), accumSrc);
      shader.op3(DxsoOpcode::Mad, accum,
                 sourceToken(kRegTypeTemp, 0, SwizzleZZZZ),
                 sourceToken(kRegTypeConst, 2, SwizzleIdentity), accumSrc);
      shader.op3(DxsoOpcode::Mad, destToken(kRegTypeRasterizerOut, 0, MaskAll),
                 sourceToken(kRegTypeTemp, 0, SwizzleWWWW),
                 sourceToken(kRegTypeConst, 3, SwizzleIdentity), accumSrc);
      expectNoMatch(shader.analyze(), "non-consecutive matrix constants");
    }

    // An o# register only carries position when it was declared dcl_position.
    static void test_rejectsUndeclaredOutput() {
      std::cout << "  test_rejectsUndeclaredOutput" << std::endl;
      ShaderBuilder shader;
      shader.dcl(DxsoUsage::Texcoord, 0, kRegTypeOutput, 0);
      appendMadChain(shader, 0, destToken(kRegTypeOutput, 0, MaskAll));
      expectNoMatch(shader.analyze(), "transform into a texcoord output");
    }

    // The snapshot is taken before the transform starts, so a source that changes while the
    // transform is still consuming it would be captured at the wrong value.
    static void test_rejectsSourceRewrittenMidTransform() {
      std::cout << "  test_rejectsSourceRewrittenMidTransform" << std::endl;
      ShaderBuilder shader;
      const uint32_t accum = destToken(kRegTypeTemp, 1, MaskAll);
      const uint32_t accumSrc = sourceToken(kRegTypeTemp, 1, SwizzleIdentity);
      shader.op2(DxsoOpcode::Mul, accum,
                 sourceToken(kRegTypeTemp, 0, SwizzleXXXX),
                 sourceToken(kRegTypeConst, 0, SwizzleIdentity));
      // rewrite r0 halfway through, so the four steps do not all see the same position
      appendFiller(shader, 0);
      shader.op3(DxsoOpcode::Mad, accum,
                 sourceToken(kRegTypeTemp, 0, SwizzleYYYY),
                 sourceToken(kRegTypeConst, 1, SwizzleIdentity), accumSrc);
      shader.op3(DxsoOpcode::Mad, accum,
                 sourceToken(kRegTypeTemp, 0, SwizzleZZZZ),
                 sourceToken(kRegTypeConst, 2, SwizzleIdentity), accumSrc);
      shader.op3(DxsoOpcode::Mad, destToken(kRegTypeRasterizerOut, 0, MaskAll),
                 sourceToken(kRegTypeTemp, 0, SwizzleWWWW),
                 sourceToken(kRegTypeConst, 3, SwizzleIdentity), accumSrc);
      expectNoMatch(shader.analyze(), "source rewritten mid-transform");
    }

    // A permuting copy into position means the rasterized vertex is a shuffle of the
    // transform's result, so the source register does not describe where it landed.
    static void test_rejectsSwizzlingMovIntoPosition() {
      std::cout << "  test_rejectsSwizzlingMovIntoPosition" << std::endl;
      ShaderBuilder shader;
      appendMadChain(shader, 0, destToken(kRegTypeTemp, 2, MaskAll));
      shader.op1(DxsoOpcode::Mov, destToken(kRegTypeRasterizerOut, 0, MaskAll),
                 sourceToken(kRegTypeTemp, 2, makeSwizzle(1, 0, 2, 3)));
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
