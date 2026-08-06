/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

// rtx_fork_upscaler_ui.cpp — fork-owned UI for the FSR upscaler, the shared
// post-upscale sharpening slider, and the frame-generation backend selector.
//
// This lives here rather than in dxvk_imgui.cpp / rtx_user_menu.cpp so that
// those upstream files keep a one-line fork footprint. In particular
// ImGUI::showDLFGOptions is left byte-identical to upstream: the selector below
// decides *whether* it gets drawn, instead of the DLSS-G panel being rewritten
// to host a second backend.

#include "rtx_fork_hooks.h"
#include "rtx_fork_fsr.h"
#include "rtx_fork_fsr_framegen.h"
#include "rtx_fork_rcas.h"
#include "rtx_options.h"
#include "rtx_imgui.h"
#include "rtx_dlfg.h"
#include "rtx_ngx_wrapper.h"
#include <algorithm>
#include "../dxvk_device.h"
#include "../dxvk_context.h"
#include "../dxvk_objects.h"
#include "../util/util_string.h"

namespace dxvk {

  // Owned by dxvk_imgui.cpp; reused rather than duplicated.
  extern RemixGui::ComboWithKey<int> dlfgMfgModeCombo;

  namespace {

    RemixGui::ComboWithKey<FSRPreset> g_fsrPresetCombo {
      "FSR Preset",
      RemixGui::ComboWithKey<FSRPreset>::ComboEntries { {
          { FSRPreset::UltraPerformance, "Ultra Performance" },
          { FSRPreset::Performance,      "Performance" },
          { FSRPreset::Balanced,         "Balanced" },
          { FSRPreset::Quality,          "Quality" },
          { FSRPreset::NativeAA,         "Native Anti-Aliasing" },
      } }
    };

    // Full selector, shown when the GPU can do DLSS Frame Generation.
    RemixGui::ComboWithKey<FrameGenerationType> g_frameGenTypeCombo {
      "Frame Generation",
      RemixGui::ComboWithKey<FrameGenerationType>::ComboEntries { {
          { FrameGenerationType::None, "Off",  "Frame generation disabled" },
          { FrameGenerationType::DLSS, "DLSS", "NVIDIA DLSS Frame Generation" },
          { FrameGenerationType::FSR,  "FSR",  "AMD FSR Frame Generation" },
      } }
    };

    // Reduced selector for GPUs without DLSS Frame Generation support.
    RemixGui::ComboWithKey<FrameGenerationType> g_frameGenTypeComboNoDlss {
      "Frame Generation",
      RemixGui::ComboWithKey<FrameGenerationType>::ComboEntries { {
          { FrameGenerationType::None, "Off", "Frame generation disabled" },
          { FrameGenerationType::FSR,  "FSR", "AMD FSR Frame Generation" },
      } }
    };

    DxvkFSRFrameGen& fsrFrameGen(const Rc<DxvkContext>& ctx) {
      return ctx->getCommonObjects()->metaFSRFrameGen();
    }

  } // namespace

  namespace fork_hooks {

    bool anyFrameGenerationEnabled() {
      return DxvkDLFG::enable() || DxvkFSRFrameGen::enable();
    }

    bool anyFrameGenerationSupported(const Rc<DxvkContext>& ctx, bool isDlfgSupported) {
      return isDlfgSupported || fsrFrameGen(ctx).supportsFSRFrameGen();
    }

    void applyFrameGenerationType(DxvkDevice* device) {
      // The dropdown is the enable control: selecting a backend turns it on and
      // turns the other one off. Living in the option's onChange rather than in
      // the menu means the invariant also holds for rtx.conf, DXVK_FRAMEGEN_TYPE
      // and SetConfigVariable, not just while the settings window is open.
      const FrameGenerationType type = RtxOptions::frameGenerationType();
      const bool wantDlfg = (type == FrameGenerationType::DLSS);
      const bool wantFsrFg = (type == FrameGenerationType::FSR);

      DxvkDLFG::enable.setDeferred(wantDlfg);
      DxvkFSRFrameGen::enable.setDeferred(wantFsrFg);

      // Matches what upstream's DLFG checkbox did: turning DLSS-G on forces
      // Reflex to Low Latency.
      if (wantDlfg) {
        RtxOptions::reflexMode.setDeferred(ReflexMode::LowLatency);
      }
    }

    namespace {

      // Colours reused from the dev menu's existing status text.
      constexpr ImVec4 kStatusActive { 0.45f, 0.85f, 0.45f, 1.0f };
      constexpr ImVec4 kStatusOff    { 0.70f, 0.70f, 0.70f, 1.0f };
      constexpr ImVec4 kStatusFault  { 250 / 255.f, 176 / 255.f, 50 / 255.f, 1.0f };

      void showFrameGenerationStatus(const Rc<DxvkContext>& ctx, bool isDlfgSupported) {
        switch (RtxOptions::frameGenerationType()) {
          case FrameGenerationType::None:
            ImGui::TextColored(kStatusOff, "Status: off - rendering every frame.");
            break;

          case FrameGenerationType::DLSS:
            if (!isDlfgSupported) {
              ImGui::TextColored(kStatusFault, "Status: unavailable - DLSS Frame Generation is not supported here.");
              const auto& reason = ctx->getCommonObjects()->metaNGXContext().getDLFGNotSupportedReason();
              if (reason.size()) {
                ImGui::TextWrapped(reason.c_str());
              }
            } else if (!DxvkDLFG::enable()) {
              ImGui::TextColored(kStatusFault, "Status: selected but inactive.");
            } else {
              const uint32_t frames = std::max(DxvkDLFG::maxInterpolatedFrames(), 1u);
              ImGui::TextColored(kStatusActive,
                str::format("Status: active - DLSS Frame Generation, ", frames + 1, "x.").c_str());
            }
            break;

          case FrameGenerationType::FSR:
            if (!fsrFrameGen(ctx).supportsFSRFrameGen()) {
              ImGui::TextColored(kStatusFault, "Status: unavailable - FSR Frame Generation is not supported here.");
              if (!fsrRuntimeAvailable()) {
                ImGui::TextWrapped("amd_fidelityfx_vk.dll was not found next to d3d9.dll.");
              }
            } else if (!DxvkFSRFrameGen::enable()) {
              ImGui::TextColored(kStatusFault, "Status: selected but inactive.");
            } else {
              ImGui::TextColored(kStatusActive, "Status: active - FSR Frame Generation, 2x.");
            }
            break;
        }
      }

    } // namespace

    void showFrameGenerationOptions(const Rc<DxvkContext>& ctx, bool isDlfgSupported) {
      // Migration: a config written before the selector existed may have a
      // backend enabled without a type set. Adopt it so the dropdown reflects
      // what is actually running rather than silently reading "Off".
      if (RtxOptions::frameGenerationType() == FrameGenerationType::None) {
        if (DxvkDLFG::enable()) {
          RtxOptions::frameGenerationType.setDeferred(FrameGenerationType::DLSS);
        } else if (DxvkFSRFrameGen::enable()) {
          RtxOptions::frameGenerationType.setDeferred(FrameGenerationType::FSR);
        }
      }

      if (isDlfgSupported) {
        g_frameGenTypeCombo.getKey(&RtxOptions::frameGenerationTypeObject());
      } else {
        // Without DLSS-G support DLSS must not be selectable, but a config file
        // or an earlier session on other hardware can still have left it set.
        if (RtxOptions::frameGenerationType() == FrameGenerationType::DLSS) {
          RtxOptions::frameGenerationType.setDeferred(FrameGenerationType::None);
        }
        g_frameGenTypeComboNoDlss.getKey(&RtxOptions::frameGenerationTypeObject());
      }
      RemixGui::SetTooltipToLastWidgetOnHover(
        "Selecting a technology enables frame generation; selecting Off disables it. "
        "V-Sync is turned off automatically while frame generation is active.");

      showFrameGenerationStatus(ctx, isDlfgSupported);

      // Per-backend extras. Neither draws an enable toggle - the dropdown above
      // is the enable control.
      if (RtxOptions::frameGenerationType() == FrameGenerationType::DLSS && isDlfgSupported) {
        if (ctx->getCommonObjects()->metaNGXContext().dlfgMaxInterpolatedFrames() > 1) {
          dlfgMfgModeCombo.getKey(&DxvkDLFG::maxInterpolatedFramesObject());
        }
      }
    }

    void showFsrUpscalerSettings(const Rc<DxvkContext>& ctx) {
      g_fsrPresetCombo.getKey(&DxvkFSR::FSROptions::presetObject());

      if (DxvkFSR::FSROptions::preset() == FSRPreset::Custom) {
        RemixGui::SliderFloat("Resolution Scale", &RtxOptions::resolutionScaleObject(), 0.1f, 1.0f, "%.2f");
      }

      uint32_t inputWidth = 0;
      uint32_t inputHeight = 0;
      ctx->getCommonObjects()->metaFSR().getInputSize(inputWidth, inputHeight);
      ImGui::TextWrapped(str::format("Render Resolution: ", inputWidth, "x", inputHeight).c_str());
    }

    void showSharedSharpnessSlider() {
      // NIS has its own sharpening stage with its own control, and native
      // rendering has nothing to sharpen after; showing the slider for either
      // would be a dead widget.
      const UpscalerType upscaler = RtxOptions::upscalerType();
      if (upscaler == UpscalerType::None || upscaler == UpscalerType::NIS) {
        return;
      }

      RemixGui::SliderFloat("Sharpness", &DxvkRCAS::Options::sharpnessObject(), 0.0f, 1.0f, "%.2f");
      RemixGui::SetTooltipToLastWidgetOnHover(
        "Post-upscale RCAS sharpening. 0.0 disables the pass entirely. FSR applies this through its own "
        "built-in sharpener; the other upscalers get a standalone RCAS pass.");
    }

  } // namespace fork_hooks

} // namespace dxvk
