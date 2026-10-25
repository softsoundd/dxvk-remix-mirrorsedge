/*
* Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_options.h"

#include <mutex>
#include <nvapi.h>
#include "../imgui/imgui.h"
#include "rtx_bridge_message_channel.h"
#include "rtx_gpu_overrides.h"
#include "rtx_dlfg.h"
#include "../util/util_global_time.h"

#include "dxvk_device.h"

namespace dxvk {
  RtxOptions* RtxOptions::s_instance = nullptr;

  void RtxOptions::graphicsPresetOnChange(DxvkDevice* device) {
    // device will be nullptr during initial config loading.
    if (device == nullptr) {
      return;
    }

    // TODO[REMIX-1482]: Currently tests expect to skip applying the graphics preset, so this needs to be skipped in test runs.
    // When we fix tests to actually use the preset, this check should be removed.
    const bool applyPresets = env::getEnvVar("DXVK_TERMINATE_APP_FRAME") == "" ||
                              env::getEnvVar("DXVK_GRAPHICS_PRESET_TYPE") != "0";

    // When switching to Custom preset, migrate ALL Quality layer settings to User layer.
    // This allows users to customize settings that were previously controlled by the preset.
    // NOTE: this does not run during the initial load due to the device nullptr check above.
    if (RtxOptions::graphicsPreset() == GraphicsPreset::Custom) {
      const RtxOptionLayer* qualityLayer = RtxOptionLayer::getQualityLayer();
      const RtxOptionLayer* userLayer = RtxOptionLayer::getUserLayer();
      
      if (qualityLayer && userLayer) {
        // Migrate ALL options from Quality layer to User layer (not just those with the flag)
        const auto globalRtxOptions = RtxOptionImpl::getGlobalOptionMap();
        for (auto& [hash, optionPtr] : *globalRtxOptions) {
          optionPtr->moveLayerValue(qualityLayer, userLayer);
        }
        Logger::info("[Graphics Preset] Switched to Custom - Quality settings migrated to User layer");
      }
      // Per-GPU overrides are not quality settings, so they still refresh on this preset change.
      if (applyPresets) {
        GpuOverrides::apply();
      }
      return;  // Don't apply preset settings when Custom - Quality layer is now empty
    }

    if (applyPresets) {
      RtxOptions::updateGraphicsPresets(device);
    }
  }

  void RtxOptions::showUICursorOnChange(DxvkDevice* device) {
    if (ImGui::GetCurrentContext() != nullptr) {
      auto& io = ImGui::GetIO();
      io.MouseDrawCursor = RtxOptions::showUICursor() && RtxOptions::showUI() != UIType::None;
    }
  }

  void RtxOptions::onAdvanceTimeChanged(DxvkDevice* device) {
    GlobalTime::get().setAdvanceTime(RtxOptions::advanceTime());
  }

  void RtxOptions::blockInputToGameInUIOnChange(DxvkDevice* device) {
    const bool doBlock = RtxOptions::blockInputToGameInUI() && RtxOptions::showUI() != UIType::None;

    BridgeMessageChannel::get().send("UWM_REMIX_UIACTIVE_MSG", doBlock ? 1 : 0, 0);
  }

  void RtxOptions::updateUpscalerFromDlssPreset() {
    // Code-driven changes for DLSS preset (automatically routes to User layer when preset is Custom)
    RtxOptionLayerTarget layerTarget(RtxOptionEditTarget::Derived);

    if (RtxOptions::Automation::disableUpdateUpscaleFromDlssPreset()) {
      return;
    }

    switch (dlssPreset()) {
      // TODO[REMIX-4105] all of these are used right after being set, so this needs to be setImmediately.
      // This should be addressed by REMIX-4109 if that is done before REMIX-4105 is fully cleaned up.
      case DlssPreset::Off:
        upscalerType.setImmediately(UpscalerType::None);
        reflexMode.setImmediately(ReflexMode::None);
        break;
      case DlssPreset::On:
        // NGX passthrough selects the upscaler via rtx.upscalerType (NIS/TAA-U/XeSS/DLSS).
        // The bundled DLSS preset must not stomp a user-configured non-DLSS upscaler at launch.
        if (RtxOptions::upscalerType() != UpscalerType::DLSS) {
          dlssPreset.setImmediately(DlssPreset::Custom);
          break;
        }
        upscalerType.setImmediately(UpscalerType::DLSS);
        qualityDLSS.setImmediately(DLSSProfile::Auto);
        reflexMode.setImmediately(ReflexMode::LowLatency); // Reflex uses ON under G (not Boost)
        break;
      case DlssPreset::Custom:
        break;
    }
  }

  void RtxOptions::updateUpscalerFromNisPreset() {
    // Code-driven changes for NIS preset (automatically routes to User layer when preset is Custom)
    RtxOptionLayerTarget layerTarget(RtxOptionEditTarget::Derived);

    switch (nisPreset()) {
    case NisPreset::Performance:
      resolutionScale.setDeferred(0.5f);
      break;
    case NisPreset::Balanced:
      resolutionScale.setDeferred(0.66f);
      break;
    case NisPreset::Quality:
      resolutionScale.setDeferred(0.75f);
      break;
    case NisPreset::Fullscreen:
      resolutionScale.setDeferred(1.0f);
      break;
    }
  }

  void RtxOptions::updateUpscalerFromTaauPreset() {
    // Code-driven changes for TAAU preset (automatically routes to User layer when preset is Custom)
    RtxOptionLayerTarget layerTarget(RtxOptionEditTarget::Derived);

    switch (taauPreset()) {
    case TaauPreset::UltraPerformance:
      resolutionScale.setDeferred(0.33f);
      break;
    case TaauPreset::Performance:
      resolutionScale.setDeferred(0.5f);
      break;
    case TaauPreset::Balanced:
      resolutionScale.setDeferred(0.66f);
      break;
    case TaauPreset::Quality:
      resolutionScale.setDeferred(0.75f);
      break;
    case TaauPreset::Fullscreen:
      resolutionScale.setDeferred(1.0f);
      break;
    }
  }

  void RtxOptions::updatePresetFromUpscaler() {
    // Code-driven changes for upscaler preset (automatically routes to User layer when preset is Custom)
    RtxOptionLayerTarget layerTarget(RtxOptionEditTarget::Derived);

    if (RtxOptions::upscalerType() == UpscalerType::None &&
        reflexMode() == ReflexMode::None) {
      RtxOptions::dlssPreset.setDeferred(DlssPreset::Off);
    } else if (RtxOptions::upscalerType() == UpscalerType::DLSS &&
               reflexMode() == ReflexMode::LowLatency) {
      if ((graphicsPreset() == GraphicsPreset::Ultra || graphicsPreset() == GraphicsPreset::High) &&
          qualityDLSS() == DLSSProfile::Auto) {
        RtxOptions::dlssPreset.setDeferred(DlssPreset::On);
      } else {
        RtxOptions::dlssPreset.setDeferred(DlssPreset::Custom);
      }
    } else {
      RtxOptions::dlssPreset.setDeferred(DlssPreset::Custom);
    }

    switch (RtxOptions::upscalerType()) {
      case UpscalerType::NIS: {
        const float nisResolutionScale = resolutionScale();
        if (nisResolutionScale <= 0.5f) {
          RtxOptions::nisPreset.setDeferred(NisPreset::Performance);
        } else if (nisResolutionScale <= 0.66f) {
          RtxOptions::nisPreset.setDeferred(NisPreset::Balanced);
        } else if (nisResolutionScale <= 0.75f) {
          RtxOptions::nisPreset.setDeferred(NisPreset::Quality);
        } else {
          RtxOptions::nisPreset.setDeferred(NisPreset::Fullscreen);
        }
        break;
      }
      case UpscalerType::TAAU: {
        const float taauResolutionScale = resolutionScale();
        if (taauResolutionScale <= 0.33f) {
          RtxOptions::taauPreset.setDeferred(TaauPreset::UltraPerformance);
        } else if (taauResolutionScale <= 0.5f) {
          RtxOptions::taauPreset.setDeferred(TaauPreset::Performance);
        } else if (taauResolutionScale <= 0.66f) {
          RtxOptions::taauPreset.setDeferred(TaauPreset::Balanced);
        } else if (taauResolutionScale <= 0.75f) {
          RtxOptions::taauPreset.setDeferred(TaauPreset::Quality);
        } else {
          RtxOptions::taauPreset.setDeferred(TaauPreset::Fullscreen);
        }
        break;
      }
      default:
        break;
    }
  }

  static bool queryNvidiaArchInfo(NV_GPU_ARCH_INFO& archInfo) {
    NvAPI_Status status;
    status = NvAPI_Initialize();
    if (status != NVAPI_OK) {
      return false;
    }
    
    NvPhysicalGpuHandle nvGPUHandle[NVAPI_MAX_PHYSICAL_GPUS];
    NvU32 GpuCount;
    status = NvAPI_EnumPhysicalGPUs(nvGPUHandle, &GpuCount);
    if (status != NVAPI_OK) {
      return false;
    }
    
    assert(GpuCount > 0);

    archInfo.version = NV_GPU_ARCH_INFO_VER;
    // Note: Currently only using the first returned GPU Handle. Ideally this should use the GPU Handle Vulkan is using
    // though in the case of a mixed architecture multi-GPU system.
    status = NvAPI_GPU_GetArchInfo(nvGPUHandle[0], &archInfo);
    return status == NVAPI_OK;
  }

  NV_GPU_ARCHITECTURE_ID RtxOptions::getNvidiaArch() {
    NV_GPU_ARCH_INFO archInfo;
    if (queryNvidiaArchInfo(archInfo) == false) {
      return NV_GPU_ARCHITECTURE_TU100;
    }
    
    return archInfo.architecture_id;
  }

  void RtxOptions::updateGraphicsPresets(DxvkDevice* device) {
    // Code-driven changes for graphics preset (automatically routes to User layer when preset is Custom)
    RtxOptionLayerTarget layerTarget(RtxOptionEditTarget::Derived);

    // Refresh the per-GPU DLSS mode override along with the preset.
    GpuOverrides::apply();

    // Handle Automatic Graphics Preset (From configuration/default)

    if (RtxOptions::graphicsPreset() == GraphicsPreset::Auto) {
      const DxvkDeviceInfo& deviceInfo = device->adapter()->devicePropertiesExt();
      const VkPhysicalDeviceProperties& deviceProperties = deviceInfo.core.properties;
      const uint32_t vendorID = deviceProperties.vendorID;
      
      // Default updateGraphicsPresets value, don't want to hit this path intentionally or Low settings will be used
      assert(vendorID != 0);

      Logger::info("Automatic Graphics Preset in use (Set rtx.graphicsPreset to something other than Auto use a non-automatic preset)");

      GraphicsPreset preferredDefault = GraphicsPreset::Low;
      const std::string gpuId = GpuOverrides::formatGpuId(deviceProperties.vendorID, deviceProperties.deviceID);
      const std::optional<GraphicsPreset> overridePreset = GpuOverrides::findGraphicsPreset(gpuId);
      const bool hasMatchingOverride = overridePreset.has_value();

      if (hasMatchingOverride) {
        preferredDefault = *overridePreset;
        Logger::info(str::format("[GPU Override] GPU ", gpuId, " (", deviceProperties.deviceName, ") uses graphics preset ",
                                 GpuOverrides::graphicsPresetName(*overridePreset), "."));
      }

      // The NVAPI architecture query only runs when the discrete-GPU heuristics use it.
      const bool isRtxSpark = GpuProfile::isRtxSpark();
      const bool needsArch = !hasMatchingOverride && !isRtxSpark && vendorID == static_cast<uint32_t>(DxvkGpuVendor::Nvidia);
      const uint32_t archId = needsArch ? static_cast<uint32_t>(getNvidiaArch()) : 0;
      const AutoPresetDecision decision = selectAutoPreset(GpuProfile::getHardwareInfo(), overridePreset, archId);
      for (const std::string& line : decision.log) {
        Logger::info(line);
      }
      preferredDefault = decision.preset;

      if (decision.applyNonNvidiaDefaults) {
        // Setup some other known good defaults for other IHVs.
        RtxOptions::resolutionScale.setDeferred(0.5f);
        // Todo: Currently this code is needed to allow the the non-DLSS upscaling paths to reflect the
        // resolution scale setting otherwise the resolution scale may be automatically updated to match
        // some other preset whenever something like updateUpscalerFromTaauPreset is called. Ideally the
        // resolution scale and these presets would not be so disconnected like this (or maybe
        // updatePresetFromUpscaler just needs to be called in the right places).
        RtxOptions::nisPreset.setDeferred(NisPreset::Performance);
        RtxOptions::taauPreset.setDeferred(TaauPreset::Performance);
      }

      // graphicsPreset itself should go to User layer, not Quality layer
      // (graphicsPreset controls what goes into Quality, it's not controlled BY Quality)
      {
        RtxOptionLayerTarget userTarget(RtxOptionEditTarget::User);
        RtxOptions::graphicsPreset.setImmediately(preferredDefault);
      }
    }

    {
      std::lock_guard<std::mutex> lock(RtxOptionImpl::getUpdateMutex());
      DxvkDLFG::litePreset.disableLayerValue(RtxOptionLayer::getQualityLayer());
    }
    if (graphicsPreset() == GraphicsPreset::Medium || graphicsPreset() == GraphicsPreset::Low) {
      DxvkDLFG::litePreset.setDeferred(true);
    }
  }

  void RtxOptions::resetUpscaler() {
    // Code-driven changes for upscaler reset (automatically routes to User layer when preset is Custom)
    RtxOptionLayerTarget layerTarget(RtxOptionEditTarget::Derived);
    
    RtxOptions::upscalerType.setDeferred(UpscalerType::DLSS);
    reflexMode.setDeferred(ReflexMode::LowLatency);
  }
}
