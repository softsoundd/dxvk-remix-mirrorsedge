/*
* Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
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

#include "../dxvk_include.h"
#include "rtx_resources.h"
#include "rtx_option.h"
#include "dxvk_image.h"

namespace dxvk {

  class NGXDLSSContext;
  class DxvkContext;

  enum class DLSSProfile : uint32_t {
    UltraPerf = 0,
    MaxPerf,
    Balanced,
    MaxQuality,
    Auto,
    FullResolution,
    Invalid
  };

  class DxvkDLSS : public CommonDeviceObject, public RtxPass {
  public:
    // Note: Values must match NVSDK_NGX_DLSS_Hint_Render_Preset in nvsdk_ngx_defs.h.
    // Presets A/B/C/D were removed in the SDK, E/F are deprecated, and G/H/I/N/O are unused,
    // so only the valid presets are exposed here. Default lets DLSS pick the best preset per quality mode.
    enum class DLSSPreset : uint32_t {
      Default = 0,
      J = 10,
      K = 11,
      L = 12,
      M = 13,
    };

    explicit DxvkDLSS(DxvkDevice* device);
    ~DxvkDLSS();
    // The caller supplies the NGX enum, which stays out of this header.
    template<typename NVSDK_NGX_T>
    static NVSDK_NGX_T profileToQuality(DLSSProfile profile);

    // J, K, L, and M pass through.
    // Any other value is returned as Default.
    template<typename NgxPreset>
    static NgxPreset presetToHint(DLSSPreset preset) {
      switch (preset) {
      case DLSSPreset::J:
      case DLSSPreset::K:
      case DLSSPreset::L:
      case DLSSPreset::M:
        return static_cast<NgxPreset>(preset);
      default:
        return static_cast<NgxPreset>(DLSSPreset::Default);
      }
    }

    bool supportsDLSS() const;

    void setSetting(const uint32_t displaySize[2], const DLSSProfile profile, uint32_t outRenderSize[2]);

    void onDestroy();

    RTX_OPTION_ARGS("rtx.dlss", DLSSPreset, preset, DLSSPreset::Default,
                    "Render preset for DLSS Super Resolution. Default = 0 (DLSS picks the best preset), J = 10, K = 11, L = 12, M = 13.",
                    args.flags = RtxOptionFlags::UserSetting);

  protected:
    virtual bool isEnabled() const override;

    static DLSSProfile getAutoProfile(uint32_t displayWidth, uint32_t displayHeight);

    DLSSProfile                 mActualProfile = DLSSProfile::Invalid;

    bool                        mRecreate = true;
    uint32_t                    mInputSize[2] = {};            ///< Input size in pixels.
    uint32_t                    mDLSSOutputSize[2] = {};       ///< DLSS output size in pixels.

    std::unique_ptr < NGXDLSSContext > m_dlssContext;
  };
}  // namespace dxvk
