/*
* Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
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
#include <locale>
#include <codecvt>
#include <cassert>

#include "rtx.h"
#include "rtx_context.h"
#include "rtx_options.h"
#include "dxvk_device.h"
#include "rtx_dlss.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_ngx_wrapper.h"


namespace dxvk {
  DxvkDLSS::DxvkDLSS(DxvkDevice* device) : CommonDeviceObject(device) {
    // Trigger DLSS context creation
    if (!m_dlssContext) {
      m_dlssContext = device->getCommon()->metaNGXContext().createDLSSContext();
    };
  }

  DxvkDLSS::~DxvkDLSS() { }

  void DxvkDLSS::onDestroy() {
    if (m_dlssContext) {
      m_dlssContext->releaseNGXFeature();
    }
    m_dlssContext = nullptr;
  }

  template<>
  NVSDK_NGX_PerfQuality_Value DxvkDLSS::profileToQuality(DLSSProfile profile) {
    NVSDK_NGX_PerfQuality_Value perfQuality = NVSDK_NGX_PerfQuality_Value_Balanced;
    switch (profile)
    {
    case DLSSProfile::UltraPerf: perfQuality = NVSDK_NGX_PerfQuality_Value_UltraPerformance; break;
    case DLSSProfile::MaxPerf: perfQuality = NVSDK_NGX_PerfQuality_Value_MaxPerf; break;
    case DLSSProfile::Balanced: perfQuality = NVSDK_NGX_PerfQuality_Value_Balanced; break;
    case DLSSProfile::MaxQuality: perfQuality = NVSDK_NGX_PerfQuality_Value_MaxQuality; break;
    case DLSSProfile::FullResolution: perfQuality = NVSDK_NGX_PerfQuality_Value_DLAA; break;
    case DLSSProfile::Auto: assert(false && "DLSSProfile::Auto passed to DxvkDLSS::profileToQuality without being resolved first"); break;
    case DLSSProfile::Invalid: assert(false && "DLSSProfile::Invalid passed to DxvkDLSS::profileToQuality"); break;
    }
    return perfQuality;
  }

  bool DxvkDLSS::supportsDLSS() const {
    return m_device->getCommon()->metaNGXContext().supportsDLSS();
  }

  bool DxvkDLSS::isEnabled() const {
      return RtxOptions::isDLSSEnabled();
  }

  DLSSProfile DxvkDLSS::getAutoProfile(uint32_t displayWidth, uint32_t displayHeight) {
    // A per-GPU override or the RTX Spark default is final and skips the preset bias below.
    if (const std::optional<DLSSProfile> hardwareProfile = selectAutoDlssMode(GpuOverrides::getDlssMode(), GpuProfile::isRtxSpark())) {
      return *hardwareProfile;
    }

    DLSSProfile desiredProfile = DLSSProfile::UltraPerf;

    // Standard display resolution based DLSS config
    if (displayHeight <= 1080) {
      desiredProfile = DLSSProfile::MaxQuality;
    } else if (displayHeight < 2160) {
      desiredProfile = DLSSProfile::Balanced;
    } else if (displayHeight < 4320) {
      desiredProfile = DLSSProfile::MaxPerf;
    } else {
      // For > 4k (e.g. 8k)
      desiredProfile = DLSSProfile::UltraPerf;
    }

    if (RtxOptions::graphicsPreset() == GraphicsPreset::Medium) {
      // When using medium preset, bias DLSS more towards performance
      desiredProfile = (DLSSProfile)std::max(0, (int) desiredProfile - 1);
    } else if (RtxOptions::graphicsPreset() == GraphicsPreset::Low) {
      // When using low preset, give me all the perf I can get!!!
      desiredProfile = (DLSSProfile) std::max(0, (int) desiredProfile - 2);
    }

    // Note: Ensure the resulting desired profile has been resolved to something non-auto.
    assert(desiredProfile != DLSSProfile::Auto);

    return desiredProfile;
  }

  void DxvkDLSS::setSetting(const uint32_t displaySize[2], const DLSSProfile profile, uint32_t outRenderSize[2]) {
    ScopedCpuProfileZone();
    // Handle the "auto" case, this is the param we'll pass to determine optimal resolution setting
    DLSSProfile actualProfile = profile;
    if (actualProfile == DLSSProfile::Auto) {
      actualProfile = getAutoProfile(displaySize[0], displaySize[1]);
    }

    if (mActualProfile == actualProfile && displaySize[0] == mDLSSOutputSize[0] && displaySize[1] == mDLSSOutputSize[1]) {
      // Nothing changed that would alter DLSS resolution(s), so return the last cached optimal render size
      outRenderSize[0] = mInputSize[0];
      outRenderSize[1] = mInputSize[1];
      return;
    }
    
    mActualProfile = actualProfile;

    const NVSDK_NGX_PerfQuality_Value perfQuality = profileToQuality<NVSDK_NGX_PerfQuality_Value>(mActualProfile);
    if (!m_dlssContext) {
      m_dlssContext = m_device->getCommon()->metaNGXContext().createDLSSContext();
    }
    const auto optimalSettings = m_dlssContext->queryOptimalSettings(displaySize, perfQuality);

    mInputSize[0] = outRenderSize[0] = optimalSettings.optimalRenderSize[0];
    mInputSize[1] = outRenderSize[1] = optimalSettings.optimalRenderSize[1];

    mDLSSOutputSize[0] = displaySize[0];
    mDLSSOutputSize[1] = displaySize[1];

    // Note: Input size used for DLSS must be less than or equal to the desired output size. This is a requirement of the DLSS API currently.
    assert(mInputSize[0] <= mDLSSOutputSize[0] && mInputSize[1] <= mDLSSOutputSize[1]);
  }
}
