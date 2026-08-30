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

#include "dxvk_imgui.h"
#include "dxvk_imgui_first_use_guide.h"
#include "imgui.h"
#include "rtx_render/rtx_imgui.h"
#include "dxvk_device.h"
#include "rtx_render/rtx_context.h"
#include "rtx_render/rtx_options.h"
#include "rtx_render/rtx_dlss.h"
#include "rtx_render/rtx_dlfg.h"
#include "rtx_render/rtx_ngx_passthrough.h"
#include "rtx_render/rtx_reflex.h"
#include "rtx_render/rtx_ray_reconstruction.h"
#include "rtx_render/rtx_xess.h"
#include "rtx_render/rtx_postFx.h"
#include "rtx_render/rtx_option_layer_gui.h"
#include "../util/util_string.h"

namespace ImGui {
  // Forward declarations for TextSeparator from dxvk_imgui.cpp
  void TextSeparator(const char* text, float pre_width = 10.0f);
}

namespace dxvk {

  // Combo boxes shared with dxvk_imgui.cpp
  extern RemixGui::ComboWithKey<DLSSProfile> dlssProfileCombo;
  extern RemixGui::ComboWithKey<DxvkDLSS::DLSSPreset> dlssRenderPresetCombo;
  extern RemixGui::ComboWithKey<XeSSPreset> xessPresetCombo;

  // Combo boxes used only by the user menu
  static RemixGui::ComboWithKey<DlssPreset> dlssPresetCombo{
    "DLSS Preset",
    RemixGui::ComboWithKey<DlssPreset>::ComboEntries{ {
        {DlssPreset::Off, "Disabled"},
        {DlssPreset::On, "Enabled"},
        {DlssPreset::Custom, "Custom"},
    } }
  };

  static RemixGui::ComboWithKey<NisPreset> nisPresetCombo{
    "NIS Preset",
    RemixGui::ComboWithKey<NisPreset>::ComboEntries{ {
        {NisPreset::Performance, "Performance"},
        {NisPreset::Balanced, "Balanced"},
        {NisPreset::Quality, "Quality"},
        {NisPreset::Fullscreen, "Fullscreen"},
    } }
  };

  static RemixGui::ComboWithKey<TaauPreset> taauPresetCombo{
    "TAA-U Preset",
    RemixGui::ComboWithKey<TaauPreset>::ComboEntries{ {
        {TaauPreset::UltraPerformance, "Ultra Performance"},
        {TaauPreset::Performance, "Performance"},
        {TaauPreset::Balanced, "Balanced"},
        {TaauPreset::Quality, "Quality"},
        {TaauPreset::Fullscreen, "Fullscreen"},
    } }
  };

  // Helper functions defined in dxvk_imgui.cpp
  RemixGui::ComboWithKey<UpscalerType>& getUpscalerCombo(DxvkDLSS& dlss, DxvkRayReconstruction& rayReconstruction);

  void ImGUI::showUserMenu(const Rc<DxvkContext>& ctx) {
    // Target user.conf layer for user menu changes
    RtxOptionLayerTarget layerTarget(RtxOptionEditTarget::User);

    const ImGuiViewport* viewport = ImGui::GetMainViewport();

    ImGui::OpenPopup(m_userGraphicsWindowTitle, ImGuiPopupFlags_NoOpenOverExistingPopup);

    ImGui::SetNextWindowPos(ImVec2(viewport->Size.x * 0.5f - m_userWindowWidth * 0.5f, viewport->Size.y * 0.5f - m_userWindowHeight * 0.5f));
    ImGui::SetNextWindowSize(ImVec2(m_userWindowWidth, 0));
    ImGui::SetNextWindowSizeConstraints(ImVec2(m_userWindowWidth, 0), ImVec2(m_userWindowWidth, m_userWindowHeight));

    // Note: When changing this padding consider:
    // - Checking to ensure text including less visible instances from hover tooltips and etc do not take up more
    // lines such that empty text lines become ineffective (to prevent jittering when text changes).
    // - Updating Dummy elements as they currently are based on half the y padding for spacing consistency.
    constexpr float windowPaddingX = 74.0f;
    constexpr float windowPaddingHalfX = windowPaddingX * 0.5f;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(windowPaddingX, 10));

    // Use the same background color and alpha as other menus, PopupBg has alpha 1 because it's used for combobox popups etc. 
    ImGui::PushStyleColor(ImGuiCol_PopupBg, ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
    bool pushedPopupBg = true;

    bool basicMenuOpen = RtxOptions::showUI() == UIType::Basic;
    if (ImGui::BeginPopupModal(m_userGraphicsWindowTitle, &basicMenuOpen, ImGuiWindowFlags_AlwaysAutoResize)) {
      // Restore PopupBg
      ImGui::PopStyleColor();
      pushedPopupBg = false;

      // Always display memory stats to user.
      showMemoryStats();

      const int itemWidth = static_cast<int>(largeUiMode() ? m_largeUserWindowWidgeWidth : m_regularUserWindowWidgetWidth);
      const int subItemWidth = static_cast<int>(ImCeil(itemWidth * 0.86f));
      const int subItemIndent = (itemWidth > subItemWidth) ? (itemWidth - subItemWidth) : 0;

      const ImVec2 childSize = ImVec2(ImGui::GetContentRegionAvail().x + windowPaddingX, m_userWindowHeight * 0.63f);
      const static ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_NoCloseWithMiddleMouseButton;
      const static ImGuiTabItemFlags tab_item_flags = ImGuiTabItemFlags_NoCloseWithMiddleMouseButton;

      ImGui::PopStyleVar();

      auto beginTabChild = [&windowPaddingHalfX, &childSize, &itemWidth](const char* tabID) -> void {
        // Make child window start at the same X offset as the tab bar separator
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() - windowPaddingHalfX);

        // Make widgets within the child start at the same X offset as widgets outside of the child
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(windowPaddingHalfX, 10));
        ImGui::BeginChild(tabID, childSize, true);

        ImGui::PushItemWidth(static_cast<float>(itemWidth));
        };

      auto endTabChild = []() -> void {
        ImGui::PopItemWidth();
        ImGui::PopStyleVar();
        ImGui::EndChild();
        };


      if (ImGui::BeginTabBar("Settings Tabs", tab_bar_flags)) {
        if (ImGui::BeginTabItem("General", nullptr, tab_item_flags)) {
          beginTabChild("##tab_child_general");
          showUserGeneralSettings(ctx, subItemWidth, subItemIndent);
          endTabChild();
          ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Graphics", nullptr, tab_item_flags)) {
          beginTabChild("##tab_child_graphics");
          showUserRenderingSettings(ctx, subItemWidth, subItemIndent);
          endTabChild();
          ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Permissions", nullptr, tab_item_flags)) {
          beginTabChild("##tab_child_permissions");
          ImGuiFirstUseGuide::showPermissionsUI();
          endTabChild();
          ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
      }

      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(windowPaddingHalfX, 10));
      ImGui::Dummy(ImVec2(0.0f, 0.0f));

      // Center align - 3 buttons now
      const ImVec2 buttonSize = ImVec2((ImGui::GetWindowSize().x - windowPaddingX) / 3 - ImGui::GetStyle().ItemSpacing.x * 2 / 3, 36);

      // Make child window start at X offset of tab bar separator
      ImGui::SetCursorPosX(ImGui::GetCursorPosX() - windowPaddingHalfX);

      if (ImGui::Button("Developer Settings Menu", buttonSize)) {
        switchMenu(UIType::Advanced);
      }

      ImGui::SameLine();

      const RtxOptionLayer* userLayer = RtxOptionLayer::getUserLayer();
      const bool unsavedChanges = userLayer && userLayer->hasUnsavedChanges();
      
      // Disable button when no unsaved changes
      ImGui::BeginDisabled(!unsavedChanges);
      
      if (unsavedChanges) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.35f, 0.14f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.78f, 0.43f, 0.22f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered]);
      }

      if (ImGui::Button("Save Settings", buttonSize)) {
        const_cast<RtxOptionLayer*>(userLayer)->save();
      }

      if (unsavedChanges) {
        ImGui::PopStyleColor(3);
      }
      
      ImGui::EndDisabled();
      
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        if (unsavedChanges) {
          // Only build the expensive tooltip when hovered
          std::string tooltip = OptionLayerUI::renderToString(userLayer, "user.conf");
          tooltip += "\nSome settings may only take effect on next launch.";
          RemixGui::SetTooltipUnformatted(tooltip.c_str());
        } else {
          RemixGui::SetTooltipUnformatted("No unsaved changes in user.conf.\nSome settings may only take effect on next launch.");
        }
      }
      
      ImGui::SameLine();
      
      // Clear User Settings button
      if (ImGui::Button("Delete All User Settings", buttonSize)) {
        if (userLayer) {
          userLayer->removeFromAllOptions();
        }
      }
      RemixGui::SetTooltipToLastWidgetOnHover("Resets all user settings to their default values.");

      ImGui::EndPopup();
    }

    if (pushedPopupBg) {
      ImGui::PopStyleColor();
    }

    // Close via titlebar close button
    if (!basicMenuOpen) {
      switchMenu(UIType::None);
    }

    ImGui::PopStyleVar();
  }

  void ImGUI::showUserGeneralSettings(
    const Rc<DxvkContext>& ctx,
    const int subItemWidth,
    const int subItemIndent) {
    auto common = ctx->getCommonObjects();
    DxvkDLSS& dlss = common->metaDLSS();
    DxvkRayReconstruction& rayReconstruction = common->metaRayReconstruction();
    DxvkDLFG& dlfg = common->metaDLFG();
    const RtxReflex& reflex = m_device->getCommon()->metaReflex();

    const bool dlssSupported = dlss.supportsDLSS();
    const bool dlfgSupported = dlfg.supportsDLFG();
    const bool reflexInitialized = reflex.reflexInitialized();

    // Describe the tab

    const char* tabDescriptionText = "General performance settings. Enabling upscaling is recommended to significantly increase performance.";

    // Note: Specifically reference the DLSS preset when present.
    if (dlssSupported) {
      tabDescriptionText = "General performance settings. Enabling the DLSS preset is recommended to significantly increase performance.";
    }

    ImGui::TextWrapped(tabDescriptionText);

    ImGui::Dummy(ImVec2(0.0f, 5.0f));

    // Preset Settings

    if (dlssSupported) {
      const DlssPreset prevDlssPreset = RtxOptions::dlssPreset();

      ImGui::TextSeparator("Preset Settings");

      dlssPresetCombo.getKey(&RtxOptions::dlssPresetObject());

      // Revert back to default DLSS settings when switch from Off to Custom
      if (prevDlssPreset == DlssPreset::Off && RtxOptions::dlssPreset() == DlssPreset::Custom) {
        RtxOptions::resetUpscaler();
      }

      RtxOptions::updateUpscalerFromDlssPreset();
    }

    // Note: Disable all settings in this section beyond the preset when a non-Custom DLSS preset is in use,
    // but only when DLSS is actually supported.
    // Note: This is stored as a bool and applied in a SetDisabled per-section so that the section labels do not get disabled
    // (as this changes the color of the line and text which is undesirable).
    const bool disableNonPresetSettings = RtxOptions::dlssPreset() != DlssPreset::Custom && dlssSupported;

    // Upscaling Settings

    ImGui::Dummy(ImVec2(0.0f, 3.0f));
    ImGui::TextSeparator("Upscaling Settings");

    {
      ImGui::BeginDisabled(disableNonPresetSettings);

      // Upscaler Type

      // Note: Use a different combo box without DLSS's upscaler listed if DLSS overall is unsupported.
      auto oldUpscalerType = RtxOptions::upscalerType();

      if (dlss.supportsDLSS()) {
        getUpscalerCombo(dlss, rayReconstruction).getKey(&RtxOptions::upscalerTypeObject());
      }
      
      ImGui::PushItemWidth(static_cast<float>(subItemWidth));
      ImGui::Indent(static_cast<float>(subItemIndent));

      if (!dlss.supportsDLSS()) {
        getUpscalerCombo(dlss, rayReconstruction).getKey(&RtxOptions::upscalerTypeObject());
      }

      if (oldUpscalerType != RtxOptions::upscalerType()) {
        RtxOptions::updateLightingSetting();
      }

      // Upscaler Preset


      switch (RtxOptions::upscalerType()) {
        case UpscalerType::DLSS: {
          // In NGX passthrough mode the DLSS mode selector maps to the game's ScreenPercentage
          // (see rtx.ngxPassthrough.driveGameScreenPercentage). Keep it live even when a
          // non-Custom DLSS preset would gray the section out (End/BeginDisabled), then show
          // the passthrough status instead of the stock DLSS object's (meaningless) state.
          if (RtxNgxPassthrough::ngxPassthroughMode()) {
            ImGui::EndDisabled();
            dlssProfileCombo.getKey(&RtxOptions::qualityDLSSObject());
            common->metaNgxPassthrough().showImguiStatusLine(false);
            ImGui::BeginDisabled(disableNonPresetSettings);
            break;
          }
          // Preset is shown before the mode. Ray Reconstruction has its own preset selector
          // (rendered by showRayReconstructionEnable above), so only show the SR preset here.
          if (!RtxOptions::enableRayReconstruction()) {
            dlssRenderPresetCombo.getKey(&DxvkDLSS::presetObject());
          }

          dlssProfileCombo.getKey(&RtxOptions::qualityDLSSObject());

          // Display DLSS Upscaling Information

          const auto currentDLSSProfile = RtxOptions::enableRayReconstruction() ? rayReconstruction.getCurrentProfile() : dlss.getCurrentProfile();
          uint32_t dlssInputWidth, dlssInputHeight;

          if (RtxOptions::enableRayReconstruction()) {
            rayReconstruction.getInputSize(dlssInputWidth, dlssInputHeight);
          } else {
            dlss.getInputSize(dlssInputWidth, dlssInputHeight);
          }

          ImGui::TextWrapped(str::format("Computed DLSS Mode: ", dlssProfileToString(currentDLSSProfile), ", Render Resolution: ", dlssInputWidth, "x", dlssInputHeight).c_str());

          break;
        }
        case UpscalerType::NIS: {
          nisPresetCombo.getKey(&RtxOptions::nisPresetObject());
          RtxOptions::updateUpscalerFromNisPreset();

          // Display NIS Upscaling Information

          auto resolutionScale = RtxOptions::resolutionScale();

          ImGui::TextWrapped(str::format("NIS Resolution Scale: ", resolutionScale).c_str());

          break;
        }
        case UpscalerType::TAAU: {
          taauPresetCombo.getKey(&RtxOptions::taauPresetObject());
          RtxOptions::updateUpscalerFromTaauPreset();

          // Display TAA-U Upscaling Information

          auto resolutionScale = RtxOptions::resolutionScale();

          ImGui::TextWrapped(str::format("TAA-U Resolution Scale: ", resolutionScale).c_str());

          break;
        }
        case UpscalerType::XeSS: {
          xessPresetCombo.getKey(&DxvkXeSS::XessOptions::presetObject());

          // Show resolution slider only for Custom preset
          if (DxvkXeSS::XessOptions::preset() == XeSSPreset::Custom) {
            RemixGui::SliderFloat("Resolution Scale", &RtxOptions::resolutionScaleObject(), 0.1f, 1.0f, "%.2f");
          }

          // Display XeSS internal resolution
          uint32_t inputWidth;
          uint32_t inputHeight;
          ctx->getCommonObjects()->metaNgxPassthrough().getXeSSInputResolution(
            uint32_t(ImGui::GetIO().DisplaySize.x),
            uint32_t(ImGui::GetIO().DisplaySize.y),
            inputWidth, inputHeight);
          ImGui::TextWrapped(str::format("Render Resolution: ", inputWidth, "x", inputHeight).c_str());

          break;
        }
        case UpscalerType::None: {
          // No custom UI here.
          break;
        }
      }

      ImGui::Unindent(static_cast<float>(subItemIndent));
      ImGui::PopItemWidth();

      ImGui::EndDisabled();
    }

    // Latency Reduction Settings
    if (dlfgSupported) {
      ImGui::Dummy(ImVec2(0.0f, 3.0f));
      ImGui::TextSeparator("Frame Generation Settings");
      showDLFGOptions(ctx);
    }

    if (reflexInitialized) {
      ImGui::Dummy(ImVec2(0.0f, 3.0f));
      ImGui::TextSeparator("Latency Reduction Settings");

      {
        ImGui::BeginDisabled(disableNonPresetSettings);

        // Note: Option to toggle the stats window is set to false here as this window is currently
        // set up to display only when the "advanced" developer settings UI is active.
        showReflexOptions(ctx, false);

        ImGui::EndDisabled();
      }
    }

    ImGui::Dummy(ImVec2(0.0f, 5.0f));
  }

  void ImGUI::showUserRenderingSettings(
    const Rc<DxvkContext>& ctx,
    const int subItemWidth,
    const int subItemIndent) {
    auto common = ctx->getCommonObjects();
    DxvkPostFx& postFx = common->metaPostFx();

    ImGui::TextWrapped("Post-processing and display settings for the NGX passthrough pipeline.");

    ImGui::Dummy(ImVec2(0.0f, 5.0f));

    // Post Effect Settings

    ImGui::TextSeparator("Post Effect Settings");

    {
      RemixGui::Checkbox("Enable Post Effects", &postFx.enableObject());

      ImGui::PushItemWidth(static_cast<float>(subItemWidth));
      ImGui::Indent(static_cast<float>(subItemIndent));

      ImGui::BeginDisabled(!postFx.enable());

      RemixGui::Checkbox("Enable Motion Blur", &postFx.enableMotionBlurObject());
      if (postFx.enableMotionBlur()) {
        ImGui::Indent(static_cast<float>(subItemIndent));
        RemixGui::Combo("Motion Blur Quality", &postFx.motionBlurModeObject(), "Legacy\0Cinematic\0");
        if (postFx.motionBlurMode() == MotionBlurMode::Cinematic) {
          RemixGui::DragFloat("Shutter Angle", &postFx.motionBlurShutterAngleObject(), 1.0f, 0.0f, 360.0f, "%.0f deg", ImGuiSliderFlags_AlwaysClamp);
        }
        ImGui::Unindent(static_cast<float>(subItemIndent));
      }
      RemixGui::Checkbox("Enable Chromatic Aberration", &postFx.enableChromaticAberrationObject());
      RemixGui::Checkbox("Enable Vignette", &postFx.enableVignetteObject());

      ImGui::EndDisabled();

      ImGui::Unindent(static_cast<float>(subItemIndent));
      ImGui::PopItemWidth();
    }

    // Other Settings

    ImGui::Dummy(ImVec2(0.0f, 3.0f));
    ImGui::TextSeparator("Other Settings");

    {
      showVsyncOptions(true);
    }

    ImGui::Dummy(ImVec2(0.0f, 5.0f));
  }

} // namespace dxvk
