#pragma once

#include "dxso_modinfo.h"
#include "dxso_decoder.h"

// NV-DXVK start: pre-projection position for vertex capture
#include "dxso_preproj_position.h"
// NV-DXVK end

namespace dxvk {

  struct DxsoAnalysisInfo {
    uint32_t bytecodeByteLength;

    bool usesDerivatives = false;
    bool usesKill        = false;

    std::vector<DxsoInstructionContext> coissues;

    // NV-DXVK start: pre-projection position for vertex capture
    DxsoPreProjectionPositionInfo preProjPosition;
    // NV-DXVK end
  };

  class DxsoAnalyzer {

  public:

    // NV-DXVK start: pre-projection position for vertex capture - takes the program info
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

    DxsoAnalysisInfo* m_analysis = nullptr;

    DxsoOpcode m_parentOpcode;

    // NV-DXVK start: pre-projection position for vertex capture
    DxsoPreProjectionPositionAnalyzer m_preProjPosition;
    // NV-DXVK end

  };

}