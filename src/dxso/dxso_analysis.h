#pragma once

#include "dxso_modinfo.h"
#include "dxso_decoder.h"

// NV-DXVK start: exact vertex capture
#include <array>
#include <string>
// NV-DXVK end

namespace dxvk {

  // NV-DXVK start: exact vertex capture
  /**
   * \brief Pre-projection position transform found in a vertex shader
   *
   * Records the register that a vertex shader multiplies by a four-register constant
   * matrix to produce oPos. Engines which transform vertices with a single combined
   * matrix (UE3's ViewProjectionMatrix) leave the untransformed position in that
   * register, so vertex capture can read it directly instead of inverting clip space -
   * clip space inversion loses precision proportional to the square of view depth,
   * which is what makes distant geometry shear.
   *
   * Nothing here proves which space \ref sourceReg is in. The consumer must match
   * \ref matrixConstBase against a constant register it recognises (see the UE3 CTAB
   * ViewProjectionMatrix check in D3D9Rtx) before treating it as a world position.
   */
  struct DxsoPreProjectionPositionInfo {
    bool           valid = false;
    DxsoRegisterId sourceReg = { DxsoRegisterType::Temp, 0 };
    uint32_t       matrixConstBase = 0;
    uint32_t       snapshotInstructionIdx = 0;

    // Why the transform was not recognised, plus a dump of the instructions that actually
    // produced each oPos component. A shader compiler is free to emit the same matrix
    // multiply many ways, so an unrecognised shader has to be diagnosable from a log.
    const char*    failureReason = "not analyzed";
    std::string    positionDefinitions;
  };
  // NV-DXVK end

  struct DxsoAnalysisInfo {
    uint32_t bytecodeByteLength;

    bool usesDerivatives = false;
    bool usesKill        = false;

    std::vector<DxsoInstructionContext> coissues;

    // NV-DXVK start: exact vertex capture
    DxsoPreProjectionPositionInfo preProjPosition;
    // NV-DXVK end
  };

  class DxsoAnalyzer {

  public:

    // NV-DXVK start: exact vertex capture - programInfo added
    DxsoAnalyzer(
      const DxsoProgramInfo&  programInfo,
            DxsoAnalysisInfo& analysis);
    // NV-DXVK end

    /**
     * \brief Processes a single instruction
     * \param [in] ins The instruction
     */
    void processInstruction(
      const DxsoInstructionContext& ctx);

    void finalize(size_t tokenCount);

  private:

    // NV-DXVK start: exact vertex capture
    struct RecordedInstruction {
      DxsoInstructionContext ctx;
      uint32_t               controlFlowDepth = 0;
    };

    // Index into m_instructions of the instruction producing some register component, or
    // -1 when it cannot be determined.
    using Definition = int32_t;
    using PositionDefinitions = std::array<Definition, 4>;

    // A matched transform: which register is multiplied, by which constant block, and the
    // span of instructions doing it.
    struct MatrixTransform {
      DxsoRegisterId source = { DxsoRegisterType::Temp, 0 };
      uint32_t       constBase = 0;
      Definition     firstIdx = -1;
      Definition     lastIdx = -1;
    };

    void recordInstruction(
      const DxsoInstructionContext& ctx);

    bool isPositionOutput(
      const DxsoBaseRegister& reg) const;

    // Walks back from \p beforeIdx for the instruction that produced (reg, component),
    // following movs so a transform that lands in a temp resolves to its arithmetic.
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

    void resolvePreProjectionPosition();
    // NV-DXVK end

    DxsoAnalysisInfo* m_analysis = nullptr;

    DxsoOpcode m_parentOpcode;

    // NV-DXVK start: exact vertex capture
    DxsoProgramInfo m_programInfo;

    // Recorded for vertex shaders only.
    std::vector<RecordedInstruction> m_instructions;

    uint32_t m_controlFlowDepth = 0;

    // Bitmask over o0..o31 of outputs declared dcl_position (vs_3_0 writes oPos as an o#)
    uint32_t m_declaredPositionOutputs = 0;

    // Subroutines let one instruction index run zero or many times per vertex, which makes
    // both the dataflow walk and an index-keyed snapshot meaningless.
    bool m_usesSubroutines = false;
    // NV-DXVK end

  };

}
