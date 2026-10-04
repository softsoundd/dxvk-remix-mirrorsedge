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

#include <cstddef>
#include <cstdint>
#include <string>

#include "dxso_common.h"
#include "dxso_decoder.h"
#include "dxso_isgn.h"

namespace dxvk {

  constexpr uint32_t kDxsoMaxPsSamplers = 16;
  constexpr uint32_t kDxsoMaxPsFloatConstants = 224;

  // D3DXREGISTER_SET values in a shader's constant table.
  constexpr uint16_t kD3dxRegisterSetFloat4 = 2u;
  constexpr uint16_t kD3dxRegisterSetSampler = 3u;

  // The parts of a compiled shader the bytecode analyses read.
  struct DxsoShaderView {
    // Includes the version token; null when the bytecode is not a whole number of tokens.
    const uint32_t*        tokens = nullptr;
    size_t                 tokenCount = 0;
    const DxsoProgramInfo* info = nullptr;
    const DxsoIsgn*        isgn = nullptr;
    const DxsoIsgn*        osgn = nullptr;
  };

  inline std::string toLowerAscii(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (const char c : s) {
      out.push_back(char((c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c));
    }
    return out;
  }

  inline bool containsToken(const std::string& s, const char* token) {
    return s.find(token) != std::string::npos;
  }

  bool isFloatConstantRegisterType(DxsoRegisterType type);

  int32_t getFloatConstantRegisterIndex(const DxsoRegister& r);

  // Scale a constant read carries through its source modifier: 1 or -1, false for any other modifier.
  bool decodeConstantModifierScale(DxsoRegModifier modifier, float& outScale);

  // How many of DxsoInstructionContext::src an opcode reads. Unknown opcodes report 0.
  uint32_t getDxsoSourceOperandCount(DxsoOpcode op);

  // Rows of a matrix multiply, whose constant operand names only the first row's register.
  uint32_t getDxsoMatrixRowCount(DxsoOpcode op);

  std::string formatDxsoSrcRegister(const DxsoRegister& r);

  std::string formatDxsoDstRegister(const DxsoRegister& r);

}
