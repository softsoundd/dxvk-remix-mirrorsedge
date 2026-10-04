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
#pragma once

// Hand-assembled D3D9 shader model 3 bytecode for the dxso analysis tests, so a test pins the
// exact encodings an analysis accepts and can express shapes fxc would not emit.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <string>
#include <vector>

#include "../../../src/dxso/dxso_code.h"
#include "../../../src/dxso/dxso_decoder.h"
#include "../../../src/dxso/dxso_isgn.h"
#include "../../../src/dxso/dxso_opcode_util.h"

namespace dxvk::dxso_test {

  constexpr uint32_t kVs30Header = 0xFFFE0300u;
  constexpr uint32_t kPs30Header = 0xFFFF0300u;
  constexpr uint32_t kEndToken = 0x0000FFFFu;

  // D3DSPR_* register types.
  constexpr uint32_t kRegTemp = 0u;
  constexpr uint32_t kRegInput = 1u;
  constexpr uint32_t kRegConst = 2u;
  constexpr uint32_t kRegRasterizerOut = 4u;
  constexpr uint32_t kRegOutput = 6u;
  constexpr uint32_t kRegColorOut = 8u;
  constexpr uint32_t kRegSampler = 10u;
  constexpr uint32_t kRegConstBool = 14u;

  // D3DSPSM_* source modifiers.
  constexpr uint32_t kModNeg = 1u;
  constexpr uint32_t kModAbs = 11u;

  // The destination modifier bit for _sat.
  constexpr uint32_t kDstSaturate = 1u << 20;

  enum WriteMask : uint32_t {
    MaskX = 0x1u, MaskY = 0x2u, MaskZ = 0x4u, MaskW = 0x8u,
    MaskXY = 0x3u, MaskZW = 0xCu, MaskXYZ = 0x7u, MaskXZW = 0xDu, MaskAll = 0xFu,
  };

  constexpr uint32_t swz(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    return x | (y << 2) | (z << 4) | (w << 6);
  }
  constexpr uint32_t kXYZW = swz(0, 1, 2, 3);
  constexpr uint32_t kXXXX = swz(0, 0, 0, 0);
  constexpr uint32_t kYYYY = swz(1, 1, 1, 1);
  constexpr uint32_t kZZZZ = swz(2, 2, 2, 2);
  constexpr uint32_t kWWWW = swz(3, 3, 3, 3);

  // The register type is split across bits 11..12 and 28..30.
  constexpr uint32_t encodeRegisterType(uint32_t type) {
    return ((type & 0x7u) << 28) | ((type & 0x18u) << 8);
  }

  constexpr uint32_t dst(uint32_t type, uint32_t num, uint32_t mask = MaskAll) {
    return 0x80000000u | encodeRegisterType(type) | ((mask & 0xFu) << 16) | (num & 0x7FFu);
  }

  constexpr uint32_t src(uint32_t type, uint32_t num, uint32_t swizzle = kXYZW, uint32_t modifier = 0) {
    return 0x80000000u | encodeRegisterType(type) | ((modifier & 0xFu) << 24) | ((swizzle & 0xFFu) << 16) | (num & 0x7FFu);
  }

  constexpr uint32_t r(uint32_t n, uint32_t s = kXYZW, uint32_t m = 0) { return src(kRegTemp, n, s, m); }
  constexpr uint32_t v(uint32_t n, uint32_t s = kXYZW, uint32_t m = 0) { return src(kRegInput, n, s, m); }
  constexpr uint32_t c(uint32_t n, uint32_t s = kXYZW, uint32_t m = 0) { return src(kRegConst, n, s, m); }
  constexpr uint32_t smp(uint32_t n) { return src(kRegSampler, n); }
  constexpr uint32_t rd(uint32_t n, uint32_t mask = MaskAll) { return dst(kRegTemp, n, mask); }
  constexpr uint32_t rdSat(uint32_t n, uint32_t mask = MaskAll) { return dst(kRegTemp, n, mask) | kDstSaturate; }
  constexpr uint32_t od(uint32_t n, uint32_t mask = MaskAll) { return dst(kRegOutput, n, mask); }
  constexpr uint32_t oC0(uint32_t mask = MaskAll) { return dst(kRegColorOut, 0, mask); }

  constexpr uint32_t opcodeToken(DxsoOpcode opcode, uint32_t length) {
    return uint32_t(opcode) | ((length & 0xFu) << 24);
  }

  inline uint32_t floatBits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
  }

  struct CtabConstant {
    const char* name;
    uint16_t    registerSet;
    uint16_t    registerIndex;
    uint16_t    registerCount = 1;
  };

  class DxsoTestShader {

  public:

    explicit DxsoTestShader(uint32_t versionToken)
      : m_info((versionToken >> 16) == 0xFFFEu ? DxsoProgramTypes::VertexShader : DxsoProgramTypes::PixelShader,
               versionToken & 0xFFu, (versionToken >> 8) & 0xFFu) {
      m_tokens.push_back(versionToken);
    }

    // The constant table fxc emits as the first comment of a shader.
    DxsoTestShader& ctab(std::initializer_list<CtabConstant> constants) {
      constexpr uint32_t kHeaderSize = 0x1c;
      constexpr uint32_t kConstantInfoSize = 20;
      std::vector<uint8_t> blob(kHeaderSize + kConstantInfoSize * constants.size(), 0);
      auto put32 = [&](size_t at, uint32_t value) { std::memcpy(&blob[at], &value, sizeof(value)); };
      auto put16 = [&](size_t at, uint16_t value) { std::memcpy(&blob[at], &value, sizeof(value)); };

      put32(0, kHeaderSize);
      put32(8, m_tokens[0]);
      put32(12, uint32_t(constants.size()));
      put32(16, kHeaderSize);

      size_t record = kHeaderSize;
      for (const CtabConstant& constant : constants) {
        put32(record, uint32_t(blob.size()));
        put16(record + 4, constant.registerSet);
        put16(record + 6, constant.registerIndex);
        put16(record + 8, constant.registerCount);
        const size_t nameLength = std::strlen(constant.name) + 1;
        blob.insert(blob.end(), constant.name, constant.name + nameLength);
        record += kConstantInfoSize;
      }
      blob.resize((blob.size() + 3) & ~size_t(3), 0);

      m_tokens.push_back(uint32_t(DxsoOpcode::Comment) | (uint32_t(1 + blob.size() / 4) << 16));
      m_tokens.push_back('C' | ('T' << 8) | ('A' << 16) | ('B' << 24));
      for (size_t at = 0; at < blob.size(); at += 4) {
        uint32_t token;
        std::memcpy(&token, &blob[at], sizeof(token));
        m_tokens.push_back(token);
      }
      return *this;
    }

    DxsoTestShader& def(uint32_t constNum, float x, float y, float z, float w) {
      m_tokens.push_back(opcodeToken(DxsoOpcode::Def, 5));
      m_tokens.push_back(dst(kRegConst, constNum, MaskAll));
      m_tokens.push_back(floatBits(x));
      m_tokens.push_back(floatBits(y));
      m_tokens.push_back(floatBits(z));
      m_tokens.push_back(floatBits(w));
      return *this;
    }

    // dcl_<usage><index> on an input (v#) or output (o#) register, recorded in the signature
    // the runtime would build from it.
    DxsoTestShader& dcl(DxsoUsage usage, uint32_t usageIndex, uint32_t regType, uint32_t regNum, uint32_t mask = MaskAll) {
      m_tokens.push_back(opcodeToken(DxsoOpcode::Dcl, 2));
      m_tokens.push_back(0x80000000u | uint32_t(usage) | ((usageIndex & 0xFu) << 16));
      m_tokens.push_back(dst(regType, regNum, mask));

      DxsoIsgn* signature = regType == kRegInput ? &m_isgn : regType == kRegOutput ? &m_osgn : nullptr;
      if (signature != nullptr) {
        DxsoIsgnEntry& entry = signature->elems[signature->elemCount++];
        entry.regNumber = regNum;
        entry.slot = signature->elemCount - 1;
        entry.semantic = DxsoSemantic{ usage, usageIndex };
        entry.mask = DxsoRegMask((mask & 1u) != 0, (mask & 2u) != 0, (mask & 4u) != 0, (mask & 8u) != 0);
      }
      return *this;
    }

    DxsoTestShader& dclInput(DxsoUsage usage, uint32_t usageIndex, uint32_t regNum, uint32_t mask = MaskAll) {
      return dcl(usage, usageIndex, kRegInput, regNum, mask);
    }

    DxsoTestShader& dclSampler(uint32_t samplerNum, DxsoTextureType type = DxsoTextureType::Texture2D) {
      m_tokens.push_back(opcodeToken(DxsoOpcode::Dcl, 2));
      m_tokens.push_back(0x80000000u | (uint32_t(type) << 27));
      m_tokens.push_back(dst(kRegSampler, samplerNum, MaskAll));
      return *this;
    }

    DxsoTestShader& texld(uint32_t dstToken, uint32_t coord, uint32_t samplerNum) {
      return op2(DxsoOpcode::Tex, dstToken, coord, smp(samplerNum));
    }

    DxsoTestShader& texkill(uint32_t regToken) {
      m_tokens.push_back(opcodeToken(DxsoOpcode::TexKill, 1));
      m_tokens.push_back(regToken);
      return *this;
    }

    DxsoTestShader& op1(DxsoOpcode opcode, uint32_t d, uint32_t s0) {
      m_tokens.push_back(opcodeToken(opcode, 2));
      m_tokens.push_back(d);
      m_tokens.push_back(s0);
      return *this;
    }

    DxsoTestShader& op2(DxsoOpcode opcode, uint32_t d, uint32_t s0, uint32_t s1) {
      m_tokens.push_back(opcodeToken(opcode, 3));
      m_tokens.push_back(d);
      m_tokens.push_back(s0);
      m_tokens.push_back(s1);
      return *this;
    }

    DxsoTestShader& op3(DxsoOpcode opcode, uint32_t d, uint32_t s0, uint32_t s1, uint32_t s2) {
      m_tokens.push_back(opcodeToken(opcode, 4));
      m_tokens.push_back(d);
      m_tokens.push_back(s0);
      m_tokens.push_back(s1);
      m_tokens.push_back(s2);
      return *this;
    }

    DxsoTestShader& ifBool(uint32_t boolReg) {
      m_tokens.push_back(opcodeToken(DxsoOpcode::If, 1));
      m_tokens.push_back(src(kRegConstBool, boolReg));
      return *this;
    }

    DxsoTestShader& endIf() {
      m_tokens.push_back(opcodeToken(DxsoOpcode::EndIf, 0));
      return *this;
    }

    DxsoTestShader& raw(uint32_t token) {
      m_tokens.push_back(token);
      return *this;
    }

    // The bytecode with its end token, version token first.
    std::vector<uint32_t> tokens() const {
      std::vector<uint32_t> tokens = m_tokens;
      tokens.push_back(kEndToken);
      return tokens;
    }

    // Valid until the shader is next modified.
    DxsoShaderView view() {
      m_finished = tokens();
      DxsoShaderView shaderView;
      shaderView.tokens = m_finished.data();
      shaderView.tokenCount = m_finished.size();
      shaderView.info = &m_info;
      shaderView.isgn = &m_isgn;
      shaderView.osgn = &m_osgn;
      return shaderView;
    }

    const DxsoProgramInfo& info() const {
      return m_info;
    }

  private:

    DxsoProgramInfo       m_info;
    DxsoIsgn              m_isgn;
    DxsoIsgn              m_osgn;
    std::vector<uint32_t> m_tokens;
    std::vector<uint32_t> m_finished;

  };

  // A shader dumped with DXVK_SHADER_DUMP_PATH, with the signatures its dcls declare.
  struct DumpedShader {
    std::vector<uint32_t> tokens;
    DxsoProgramInfo       info;
    DxsoIsgn              isgn;
    DxsoIsgn              osgn;

    DxsoShaderView view() const {
      DxsoShaderView shaderView;
      shaderView.tokens = tokens.data();
      shaderView.tokenCount = tokens.size();
      shaderView.info = &info;
      shaderView.isgn = &isgn;
      shaderView.osgn = &osgn;
      return shaderView;
    }
  };

  // ps_2_x texture registers are interpolated texcoords of their own index, as the runtime maps them.
  inline bool loadDumpedShader(const std::string& path, DumpedShader& out) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
      return false;
    }
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (bytes.size() < 4 || (bytes.size() % 4) != 0) {
      return false;
    }
    out.tokens.resize(bytes.size() / 4);
    std::memcpy(out.tokens.data(), bytes.data(), bytes.size());

    const uint32_t version = out.tokens[0];
    const bool vertex = (version >> 16) == 0xFFFEu;
    if (!vertex && (version >> 16) != 0xFFFFu) {
      return false;
    }
    out.info = DxsoProgramInfo(vertex ? DxsoProgramTypes::VertexShader : DxsoProgramTypes::PixelShader,
                               version & 0xFFu, (version >> 8) & 0xFFu);

    DxsoDecodeContext decoder(out.info);
    DxsoCodeIter iter(out.tokens.data() + 1);
    while (decoder.decodeInstruction(iter)) {
      const DxsoInstructionContext& ctx = decoder.getInstructionContext();
      if (ctx.instruction.opcode != DxsoOpcode::Dcl) {
        continue;
      }
      DxsoIsgn* signature = nullptr;
      DxsoSemantic semantic = ctx.dcl.semantic;
      if (ctx.dst.id.type == DxsoRegisterType::Input) {
        signature = &out.isgn;
      } else if (ctx.dst.id.type == DxsoRegisterType::Output) {
        signature = &out.osgn;
      } else if (!vertex && ctx.dst.id.type == DxsoRegisterType::PixelTexcoord) {
        signature = &out.isgn;
        semantic = DxsoSemantic{ DxsoUsage::Texcoord, ctx.dst.id.num };
      }
      if (signature == nullptr || signature->elemCount >= signature->elems.size()) {
        continue;
      }
      DxsoIsgnEntry& entry = signature->elems[signature->elemCount++];
      entry.regNumber = ctx.dst.id.num;
      entry.slot = signature->elemCount - 1;
      entry.semantic = semantic;
      entry.mask = ctx.dst.mask;
    }
    return true;
  }

}
