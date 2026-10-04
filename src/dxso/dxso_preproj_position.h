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

#include "dxso_common.h"
#include "dxso_decoder.h"

namespace dxvk {

  /**
   * \brief Pre-projection position transform found in a vertex shader
   *
   * The register a vertex shader multiplies by a four-register constant matrix to produce
   * oPos, which vertex capture reads instead of inverting clip space (see "Exact vertex position
   * capture" in UE3Compatibility.md). Nothing here proves which space \ref sourceReg is in: the
   * consumer must recognise \ref matrixConstBase first.
   */
  struct DxsoPreProjectionPositionInfo {
    bool           valid = false;
    DxsoRegisterId sourceReg = { DxsoRegisterType::Temp, 0 };
    uint32_t       matrixConstBase = 0;
    uint32_t       snapshotInstructionIdx = 0;

    // Why the transform was not recognised, and the instructions producing each oPos component.
    const char*    failureReason = "not analyzed";
    std::string    positionDefinitions;
  };

  class DxsoPreProjectionPositionAnalyzer {

  public:

    explicit DxsoPreProjectionPositionAnalyzer(
      const DxsoProgramInfo& programInfo);

    void recordInstruction(
      const DxsoInstructionContext& ctx);

    DxsoPreProjectionPositionInfo resolve() const;

  private:

    struct RecordedInstruction {
      DxsoInstructionContext ctx;
      uint32_t               controlFlowDepth = 0;
    };

    // Index into m_instructions of the instruction producing a register component, or -1.
    using Definition = int32_t;
    using PositionDefinitions = std::array<Definition, 4>;

    struct MatrixTransform {
      DxsoRegisterId source = { DxsoRegisterType::Temp, 0 };
      uint32_t       constBase = 0;
      Definition     firstIdx = -1;
      Definition     lastIdx = -1;
    };

    bool isPositionOutput(
      const DxsoBaseRegister& reg) const;

    // Follows movs, so a transform that lands in a temp resolves to its arithmetic.
    Definition resolveDefinition(
      const DxsoRegisterId& reg,
            uint32_t        component,
            int32_t         beforeIdx) const;

    bool findPositionDefinitions(
            PositionDefinitions& outDefs) const;

    bool matchDp4Group(
      const PositionDefinitions& defs,
            MatrixTransform&     outTransform) const;

    bool matchAccumulatorChain(
      const PositionDefinitions& defs,
            MatrixTransform&     outTransform) const;

    bool matchMatrixOp(
      const PositionDefinitions& defs,
            MatrixTransform&     outTransform) const;

    bool isSourceStableAcross(
      const MatrixTransform& transform) const;

    std::string describePositionDefinitions(
      const PositionDefinitions& defs) const;

    bool m_isVertexShader = false;

    std::vector<RecordedInstruction> m_instructions;

    uint32_t m_controlFlowDepth = 0;

    // Bitmask over o0..o31 of outputs declared dcl_position; vs_3_0 writes oPos as an o#.
    uint32_t m_declaredPositionOutputs = 0;

    // A subroutine runs one instruction index zero or many times per vertex, which breaks both
    // the dataflow walk and the index-keyed snapshot.
    bool m_usesSubroutines = false;

  };

}
