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

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "dxso_opcode_util.h"

namespace dxvk {

  // A term of `value * scale + offset`: imm + c[constReg].constComp * factor
  // + c[constReg2].constComp2 * factor2, each part optional and constReg2 only alongside
  // constReg. With no part set the term is absent: 1 for a scale, 0 for an offset. A scale holds
  // one part, since a product of two draw-time constants is not representable; an offset may
  // hold all three, as UE3 adds tile offsets and literal centring biases to one coordinate.
  // inexact marks math the model cannot represent, while the origin may still be provable.
  struct UvAffineTerm {
    bool immValid = false;
    float imm = 0.0f;
    int16_t constReg = -1;
    uint8_t constComp = 0;
    float factor = 1.0f;
    int16_t constReg2 = -1;
    uint8_t constComp2 = 0;
    float factor2 = 1.0f;
    bool inexact = false;
  };

  // One coordinate component, `uv * scale + offset`, or for a UV matrix such as UE3's Rotator
  // `uv[scaleComponent] * scale + uv[crossComponent] * cross + offset` (see "Rotators and UV
  // matrices" in UE3Compatibility.md).
  struct UvComponentAffine {
    UvAffineTerm scale;   // absent => 1.0
    UvAffineTerm offset;  // absent => 0.0
    UvAffineTerm cross;   // absent => 0.0
    bool hasCross = false;
    uint8_t scaleComponent = 0;  // interpolant component `scale` multiplies when hasCross
    uint8_t crossComponent = 0;  // interpolant component `cross` multiplies when hasCross
  };

  // Whether every sample site of a sampler reads U and V from components of one TEXCOORD
  // interpolant, and the affine chain applied to them.
  struct PsSamplerUvOrigin {
    bool originValid = false;
    bool sitesAgree = true;      // every valid site agrees on origin and affine
    bool affineExact = false;    // the chain is representable for both components
    // Disagreeing static tilings resolved to the highest-frequency site, as UE3's distance-fade
    // anti-tiling needs, rather than the first in bytecode order.
    bool preferredHighestFrequencySite = false;
    uint8_t semanticIndex = 0;   // TEXCOORD usage index of the interpolant
    uint8_t compU = 0;
    uint8_t compV = 1;
    UvComponentAffine affineU;
    UvComponentAffine affineV;
    uint16_t validSiteCount = 0;
    uint16_t invalidSiteCount = 0;
  };

  // How a vertex shader's TEXCOORD output derives from the input assembler.
  enum class Ue3VsUvTraceKind : uint8_t {
    Invalid = 0,     // origin not provable: procedural UVs, mixed inputs, unsupported ops
    PureMove,        // an IA texcoord set's .xy exactly
    AffineConst,     // an IA texcoord set's .xy * scale + offset, from constants or immediates
    OriginOnly,      // origin proven, but not as an affine transform of .xy
  };

  struct Ue3VsTexcoordTraceResult {
    Ue3VsUvTraceKind kind = Ue3VsUvTraceKind::Invalid;
    uint8_t iaTexcoordIndex = 0;
    uint8_t inputReg = 0;
    UvComponentAffine affineU;
    UvComponentAffine affineV;
  };

  bool uvAffineTermPresent(const UvAffineTerm& t);

  bool uvComponentAffineExact(const UvComponentAffine& a);

  // Appends the constant registers the affine reads to regs, up to cap entries.
  void uvComponentAffineCollectConstRegs(const UvComponentAffine& a, int32_t* regs, uint32_t& count, uint32_t cap);

  std::string formatUvComponentAffine(const UvComponentAffine& affine);

  // Output register of a vertex shader's TEXCOORD<usageIndex> interpolant, or UINT32_MAX.
  uint32_t findVsTexcoordOutputRegister(const DxsoIsgn& osgn, uint32_t usageIndex);

  // trace, when given, collects an instruction-level trace (test_dxso_uv_dataflow's dump mode).
  void analyzePsSamplerUvOrigins(const DxsoShaderView& pixelShader,
                                 std::array<PsSamplerUvOrigin, kDxsoMaxPsSamplers>& outOrigins,
                                 std::vector<std::string>* trace = nullptr);

  // Which IA texcoord set feeds components compU and compV of a vertex shader output, and how.
  Ue3VsTexcoordTraceResult traceVsOutputTexcoordToInputUsageIndex(const DxsoShaderView& vertexShader,
                                                                  uint32_t outputReg,
                                                                  uint8_t compU,
                                                                  uint8_t compV);

}
