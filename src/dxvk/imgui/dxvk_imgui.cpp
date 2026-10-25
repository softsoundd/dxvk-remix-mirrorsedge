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

#include <cassert>
#include <tuple>
#include <string>
#include <sstream>
#include <iomanip>
#include <optional>
#include <vector>
#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <map>
#include <set>
#include <nvapi.h>
#include <NVIDIASansMd.ttf.h>
#include <NVIDIASansBd.ttf.h>
#include <RobotoMonoRg.ttf.h>

#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_dxvk.hpp"
#include "imgui_impl_win32.h"
#include "implot.h"
#include "dxvk_imgui.h"
#include "rtx_render/rtx_imgui.h"
#include "dxvk_device.h"
#include "rtx_render/rtx_utils.h"
#include "rtx_render/rtx_shader_manager.h"
#include "rtx_render/rtx_camera.h"
#include "rtx_render/rtx_context.h"
#include "rtx_render/rtx_gpu_overrides.h"
#include "rtx_render/rtx_options.h"
#include "rtx_render/rtx_nsight_capture.h"
#include "rtx_render/rtx_xess.h"
#include "rtx_render/rtx_ngx_passthrough.h"
#include "dxvk_image.h"
#include "../util/rc/util_rc_ptr.h"
#include "../util/util_math.h"
#include "../util/util_global_time.h"
#include "rtx_render/rtx_bridge_message_channel.h"
#include "dxvk_imgui_about.h"
#include "dxvk_imgui_first_use_guide.h"
#include "dxvk_imgui_splash.h"
#include "rtx_render/rtx_option_layer_gui.h"
#include "rtx_render/rtx_option_manager.h"
#include "dxvk_scoped_annotation.h"
#include "../../d3d9/d3d9_rtx.h"
#include "dxvk_memory_tracker.h"
#include "rtx_render/rtx_overlay_window.h"


extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
extern ImGuiKey ImGui_ImplWin32_VirtualKeyToImGuiKey(WPARAM wParam);

namespace ImGui {
  // Note: Implementation of text separators adapted from:
  // https://github.com/ocornut/imgui/issues/1643

  void CenteredSeparator(float width = 0) {
    ImGuiWindow* window = GetCurrentWindow();
    if (window->SkipItems)
      return;
    ImGuiContext& g = *GImGui;

    // Horizontal Separator
    float x1, x2;
    if (window->DC.CurrentColumns == NULL && (width == 0)) {
      // Span whole window
      x1 = window->DC.CursorPos.x;
      // Note: Account for padding on the Window
      x2 = window->Pos.x + window->Size.x - window->WindowPadding.x;
    } else {
      // Start at the cursor
      x1 = window->DC.CursorPos.x;
      if (width != 0) {
        x2 = x1 + width;
      } else {
        x2 = window->ClipRect.Max.x;
        // Pad right side of columns (except the last one)
        if (window->DC.CurrentColumns && (window->DC.CurrentColumns->Current < window->DC.CurrentColumns->Count - 1))
          x2 -= g.Style.ItemSpacing.x;
      }
    }
    float y1 = window->DC.CursorPos.y + int(window->DC.CurrLineSize.y / 2.0f);
    float y2 = y1 + 1.0f;

    window->DC.CursorPos.x += width; //+ g.Style.ItemSpacing.x;

    const ImRect bb(ImVec2(x1, y1), ImVec2(x2, y2));
    ItemSize(ImVec2(0.0f, 0.0f)); // NB: we don't provide our width so that it doesn't get feed back into AutoFit, we don't provide height to not alter layout.
    if (!ItemAdd(bb, NULL)) {
      return;
    }

    window->DrawList->AddLine(bb.Min, ImVec2(bb.Max.x, bb.Min.y), GetColorU32(ImGuiCol_Border));
  }

  // Create a centered separator right after the current item.
  // Eg.: 
  // ImGui::PreSeparator(10);
  // ImGui::Text("Section VI");
  // ImGui::SameLineSeparator();
  void SameLineSeparator(float width = 0) {
    ImGui::SameLine();
    CenteredSeparator(width);
  }

  // Create a centered separator which can be immediately followed by a item
  void PreSeparator(float width) {
    ImGuiWindow* window = GetCurrentWindow();
    if (window->DC.CurrLineSize.y == 0)
      window->DC.CurrLineSize.y = ImGui::GetTextLineHeight();
    CenteredSeparator(width);
    ImGui::SameLine();
  }

  // The value for width is arbitrary. But it looks nice.
  void TextSeparator(const char* text, float pre_width = 10.0f) {
    ImGui::PreSeparator(pre_width);
    ImGui::Text(text);
    ImGui::SameLineSeparator();
  }
}

namespace dxvk {
  static auto upscalerNoDLSSCombo = RemixGui::ComboWithKey<UpscalerType>(
    "Upscaler Type",
    { {
      {UpscalerType::None, "None"},
      {UpscalerType::NIS, "NIS"},
      {UpscalerType::TAAU, "TAA-U"},
      {UpscalerType::XeSS, "XeSS"},
  } });

  static auto upscalerDLSSCombo = RemixGui::ComboWithKey<UpscalerType>(
    "Upscaler Type",
    { {
      {UpscalerType::None, "None"},
      {UpscalerType::DLSS, "DLSS"},
      {UpscalerType::NIS, "NIS"},
      {UpscalerType::TAAU, "TAA-U"},
      {UpscalerType::XeSS, "XeSS"},
  } });

  RemixGui::ComboWithKey<DLSSProfile> dlssProfileCombo{
    "DLSS Mode",
    RemixGui::ComboWithKey<DLSSProfile>::ComboEntries{ {
        {DLSSProfile::UltraPerf, "Ultra Performance"},
        {DLSSProfile::MaxPerf, "Performance"},
        {DLSSProfile::Balanced, "Balanced"},
        {DLSSProfile::MaxQuality, "Quality"},
        {DLSSProfile::FullResolution, "Full Resolution"},
        {DLSSProfile::Auto, "Auto"},
    } }
  };

  // Note: named 'dlssRenderPresetCombo' to avoid colliding with the user-menu 'dlssPresetCombo'
  // (the combined On/Off/Custom preset).
  RemixGui::ComboWithKey<DxvkDLSS::DLSSPreset> dlssRenderPresetCombo{
    "DLSS SR Preset",
    RemixGui::ComboWithKey<DxvkDLSS::DLSSPreset>::ComboEntries{ {
        {DxvkDLSS::DLSSPreset::Default, "Default", "Let DLSS pick the best preset per quality mode."},
        {DxvkDLSS::DLSSPreset::J, "J", "Similar to preset K, but may show slightly less ghosting at the cost of extra flickering. Preset K is generally recommended over J."},
        {DxvkDLSS::DLSSPreset::K, "K", "Transformer-based default for DLAA/Balanced/Quality modes. Best image quality at a higher performance cost."},
        {DxvkDLSS::DLSSPreset::L, "L", "Default for Ultra Performance mode."},
        {DxvkDLSS::DLSSPreset::M, "M", "Default for Performance mode."},
    } }
  };

  RemixGui::ComboWithKey<XeSSPreset> xessPresetCombo{
    "XeSS Preset",
    RemixGui::ComboWithKey<XeSSPreset>::ComboEntries{ {
        {XeSSPreset::UltraPerf, "Ultra Performance"},
        {XeSSPreset::Performance, "Performance"},
        {XeSSPreset::Balanced, "Balanced"},
        {XeSSPreset::Quality, "Quality"},
        {XeSSPreset::UltraQuality, "Ultra Quality"},
        {XeSSPreset::UltraQualityPlus, "Ultra Quality Plus"},
        {XeSSPreset::NativeAA, "Native Anti-Aliasing"},
        {XeSSPreset::Custom, "Custom"},
    } }
  };

  RemixGui::ComboWithKey<int> dlfgMfgModeCombo {
    "DLSS Frame Generation Mode",
    RemixGui::ComboWithKey<int>::ComboEntries { {
        {1, "2x"},
        {2, "3x"},
        {3, "4x"},
        {4, "5x"},
        {5, "6x"},
    } }
  };

  RemixGui::ComboWithKey<int> dlfgPresetCombo {
    "DLSS Frame Generation Preset",
    RemixGui::ComboWithKey<int>::ComboEntries { {
        {0, "A"},
        {1, "C"},
    } }
  };

  RemixGui::ComboWithKey<ReflexMode> reflexModeCombo{
    "Reflex",
    RemixGui::ComboWithKey<ReflexMode>::ComboEntries{ {
        {ReflexMode::None, "Disabled"},
        {ReflexMode::LowLatency, "Enabled"},
        {ReflexMode::LowLatencyBoost, "Enabled + Boost"},
    } }
  };

  static auto themeCombo = RemixGui::ComboWithKey<ImGUI::Theme>(
    "Mode##theme",
    {
      {ImGUI::Theme::Toolkit,  "Default Theme"},
      {ImGUI::Theme::Legacy,   "Legacy Theme"},
      {ImGUI::Theme::Nvidia,   "NVIDIA Theme"},
  });

  // Styles 
  constexpr ImGuiSliderFlags sliderFlags = ImGuiSliderFlags_AlwaysClamp;
  constexpr ImGuiTreeNodeFlags collapsingHeaderClosedFlags = ImGuiTreeNodeFlags_CollapsingHeader;
  constexpr ImGuiTreeNodeFlags collapsingHeaderFlags = collapsingHeaderClosedFlags | ImGuiTreeNodeFlags_DefaultOpen;
  constexpr ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_AlwaysVerticalScrollbar;
  constexpr ImGuiWindowFlags popupWindowFlags = ImGuiWindowFlags_NoSavedSettings;

  RemixGui::ComboWithKey<UpscalerType>& getUpscalerCombo(DxvkDLSS& dlss) {
    if (dlss.supportsDLSS()) {
      return upscalerDLSSCombo;
    } else {
      // Drop DLSS item if unsupported.
      return upscalerNoDLSSCombo;
    }
  }

  ImGUI::ImGUI(DxvkDevice* device)
  : m_device (device)
  , m_gameHwnd   (nullptr)
  , m_about  (new ImGuiAbout)
  , m_firstUseGuide (new ImGuiFirstUseGuide)
  , m_splash  (new ImGuiSplash) {
    // Set up constant state
    m_rsState.polygonMode       = VK_POLYGON_MODE_FILL;
    m_rsState.cullMode          = VK_CULL_MODE_BACK_BIT;
    m_rsState.frontFace         = VK_FRONT_FACE_CLOCKWISE;
    m_rsState.depthClipEnable   = VK_FALSE;
    m_rsState.depthBiasEnable   = VK_FALSE;
    m_rsState.conservativeMode  = VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT;
    m_rsState.sampleCount       = VK_SAMPLE_COUNT_1_BIT;

    m_blendMode.enableBlending  = VK_TRUE;
    m_blendMode.colorSrcFactor  = VK_BLEND_FACTOR_ONE;
    m_blendMode.colorDstFactor  = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    m_blendMode.colorBlendOp    = VK_BLEND_OP_ADD;
    m_blendMode.alphaSrcFactor  = VK_BLEND_FACTOR_ONE;
    m_blendMode.alphaDstFactor  = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    m_blendMode.alphaBlendOp    = VK_BLEND_OP_ADD;
    m_blendMode.writeMask       = VK_COLOR_COMPONENT_R_BIT
                                | VK_COLOR_COMPONENT_G_BIT
                                | VK_COLOR_COMPONENT_B_BIT
                                | VK_COLOR_COMPONENT_A_BIT;
    
    // the size of the pool is oversized, but it's copied from imgui demo itself.
    VkDescriptorPoolSize pool_sizes[] =
    {
      { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
      { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
      { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
      { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
      { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
      { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
      { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
      { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
      { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    // ImGUI is currently using a single set per texture, and so we want this to be a big number 
    //  to support displaying texture lists in games that use a lot of textures.
    // See: 'ImGui_ImplDxvk::AddTexture(...)' for more details about how this system works.
    pool_info.maxSets = 10000;
    pool_info.poolSizeCount = std::size(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;

    m_device->vkd()->vkCreateDescriptorPool(m_device->handle(), &pool_info, nullptr, &m_imguiPool);

    // Initialize the core structures of ImGui and ImPlot
    m_context = ImGui::CreateContext();
    m_plotContext = ImPlot::CreateContext();

    ImGui::SetCurrentContext(m_context);
    ImPlot::SetCurrentContext(m_plotContext);

    // Setup custom style
    setupStyle();

    // Enable keyboard nav
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    if (RtxOptions::useNewGuiInputMethod()) {
      m_overlayWin = new GameOverlay("RemixGuiInputSink", this);
    }
  }

  ImGUI::~ImGUI() {
    // Joins the overlay thread, which feeds input into the ImGui context torn down below.
    m_overlayWin = nullptr;

    ImGui::SetCurrentContext(m_context);
    ImPlot::SetCurrentContext(m_plotContext);

    if(m_init) {
      ImGui_ImplWin32_Shutdown();
    }

    //add the destroy the imgui created structures
    if(m_imguiPool != VK_NULL_HANDLE)
      m_device->vkd()->vkDestroyDescriptorPool(m_device->handle(), m_imguiPool, nullptr);

    if (m_init) {
      // FontView and FontImage will be released by m_fontTextureView and m_fontTexture later
      ImGuiIO& io = ImGui::GetIO();
      ImGui_ImplDxvk::Data* bd = (ImGui_ImplDxvk::Data*) io.BackendRendererUserData;
      bd->FontView = VK_NULL_HANDLE;
      bd->FontImage = VK_NULL_HANDLE;

      ImGui_ImplDxvk::Shutdown();
      m_init = false;
    }

    // Destroy the ImGui and ImPlot context
    ImPlot::DestroyContext(m_plotContext);
    ImGui::DestroyContext(m_context);
  }
  
  void ImGUI::wndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (m_overlayWin.ptr() != nullptr) {
      m_overlayWin->gameWndProcHandler(hWnd, msg, wParam, lParam);
    } else {
      // Note this is the old method for grabbing keyboard/mouse inputs which relies on hooking
      //  the wndproc from the original game, and sending that data across the x86 -> x64 bridge.  
      //  We see compatibilities in older applications with this approach that are tricky to resolve.
      //  Favour the new approach `useNewGuiInputMethod` when possible.
      withInputLock([&] {
        ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam);
      });
    }
  }

  void ImGUI::showMemoryStats() const {
    if (RtxOptions::Automation::disableDisplayMemoryStatistics()) {
      return;
    }

    // Gather runtime vidmem stats
    VkDeviceSize vidmemSize = 0;
    VkDeviceSize vidmemUsedSize = 0;

    DxvkAdapterMemoryInfo memHeapInfo = m_device->adapter()->getMemoryHeapInfo();
    DxvkMemoryAllocator& memoryManager = m_device->getCommon()->memoryManager();
    const VkPhysicalDeviceMemoryProperties& memoryProperties = memoryManager.getMemoryProperties();

    for (uint32_t i = 0; i < memoryProperties.memoryHeapCount; i++) {
      if (memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
        vidmemSize += memHeapInfo.heaps[i].memoryBudget;
        vidmemUsedSize += memHeapInfo.heaps[i].memoryAllocated;
      }
    }

    // Calculate video memory information

    constexpr float bytesPerMebibyte = 1024.f * 1024.f;
    const VkDeviceSize vidmemFreeSize = vidmemSize - std::min(vidmemUsedSize, vidmemSize);
    const float vidmemTotalSizeMB = (float)((double) vidmemSize / bytesPerMebibyte);
    const float vidmemUsedSizeMB = (float)((double) vidmemUsedSize / bytesPerMebibyte);
    const float vidmemFreeSizeMB = (float)((double) vidmemFreeSize / bytesPerMebibyte);
    const float freeVidMemRatio = (float)std::min((double) vidmemFreeSize / (double) vidmemSize, 1.0);

    // Display video memory information

#ifdef REMIX_DEVELOPMENT
    ImGui::Text("Video Memory Usage: %.f MiB / %.f MiB (%.f MiB free)", vidmemUsedSizeMB, vidmemTotalSizeMB, vidmemFreeSizeMB);
#else
    // Note: Simplify for end users, free memory is usually not as important to list and can just be observed visually with the graph.
    ImGui::Text("Video Memory Usage: %.f MiB / %.f MiB", vidmemUsedSizeMB, vidmemTotalSizeMB);
#endif

    // Note: Map the range [0.1, 0.6] to [0, 1] and clamp outside it to bias and clamp the green->red color transition more.
    const float remappedFreeVidMemRatio = std::max(std::min(freeVidMemRatio + 0.4f, 1.0f) - 0.5f, 0.0f) * 2.0f;
    ImVec4 barColor = ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f };

    ImGui::ColorConvertHSVtoRGB(
      remappedFreeVidMemRatio * 0.32f, 0.717f, 0.704f,
      barColor.x, barColor.y, barColor.z);

    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, barColor);
    ImGui::ProgressBar(vidmemUsedSizeMB / vidmemTotalSizeMB);
    ImGui::PopStyleColor();

    ImGui::TextWrapped("RTX Remix dynamically uses available VRAM to maximize texture quality.");
    
    ImGui::Dummy(ImVec2 { 4, 0 });
  }

  void ImGUI::switchMenu(UIType type, bool force) {
    UIType oldType = RtxOptions::showUI();
    if (oldType == type && !force) {
      return;
    }
    
    if (type == UIType::None) {
      onCloseMenus();
    } else {
      onOpenMenus();
    }

    {
      // Target user layer for UI state changes (this is a user preference)
      RtxOptionLayerTarget layerTarget(RtxOptionEditTarget::User);
      RtxOptions::showUI.setDeferred(type);
    }

    if (RtxOptions::blockInputToGameInUI()) {
      BridgeMessageChannel::get().send("UWM_REMIX_UIACTIVE_MSG",
                                       type != UIType::None ? 1 : 0, 0);
    }
  }
  
  void ImGUI::processHotkeys() {
    auto& io = ImGui::GetIO();

    if (checkHotkeyState(RtxOptions::remixMenuKeyBinds())) {
      if(RtxOptions::defaultToAdvancedUI()) {
        switchMenu(RtxOptions::showUI() != UIType::None ? UIType::None : UIType::Advanced);
      } else {
        switchMenu(RtxOptions::showUI() != UIType::None ? UIType::None : UIType::Basic);
      }
    }


    // Toggle ImGUI mouse cursor. Alt-Del
    if (io.KeyAlt && ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Delete))) {
      RtxOptions::showUICursor.setDeferred(!RtxOptions::showUICursor());
    }

    // Toggle input blocking. Alt-Backspace
    if (io.KeyAlt && ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Backspace))) {
      RtxOptions::blockInputToGameInUI.setDeferred(!RtxOptions::blockInputToGameInUI());
    }
  }

  void ImGUI::update(const Rc<DxvkContext>& ctx) {
    processHotkeys();

    m_splash->update(m_largeFont);

    // On first frame, check if the first-use guide should be shown (overrides loaded showUI).
    {
      static bool s_firstUseChecked = false;
      if (!s_firstUseChecked) {
        s_firstUseChecked = true;
        if (ImGuiFirstUseGuide::shouldShow()) {
          switchMenu(UIType::FirstUseGuide);
        }
      }
    }

    const auto showUI = RtxOptions::showUI();
    if (showUI == UIType::Advanced) {
      showMainMenu(ctx);

      // Uncomment to see the ImGUI demo, good reference!  Also, need to undefine IMGUI_DISABLE_DEMO_WINDOWS (in "imconfig.h")
      //ImGui::ShowDemoWindow();
    } else if (showUI == UIType::Basic) {
      showUserMenu(ctx);
    } else if (showUI == UIType::FirstUseGuide) {
      if (m_firstUseGuide->show(m_boldFont)) {
        switchMenu(UIType::None);
      }
    }
    
    // Render any blocked edit popup warnings
    RemixGui::RenderRtxOptionBlockedEditPopup();

    // Note: Only display the latency stats window when the Advanced UI is active as the Basic UI acts as a modal which blocks other
    // windows from being interacted with.
    if (showUI == UIType::Advanced && m_reflexLatencyStatsOpen) {
      showReflexLatencyStats();
    }

    if (showUI == UIType::None) {
      ImGui::CloseCurrentPopup();
      ImGui::GetIO().MouseDrawCursor = false;
    } else {
      if (RtxOptions::showUICursor()) {
        ImGui::GetIO().MouseDrawCursor = true;
        // Force display counter into invisible state
        while (ShowCursor(FALSE) >= 0) { }
      } else {
        // Force display counter into visible state
        while (ShowCursor(TRUE) < 0) {  }
      }
    }

    showHudMessages(ctx);

#ifdef REMIX_DEVELOPMENT
    // Show visual indicator when crash hotkeys are armed (one option arms both CPU and GPU crash hotkeys)
    if (RtxOptions::enableCrashHotkey()) {
      const auto crashHotkeyStr = buildKeyBindDescriptorStringForDisplay(RtxOptions::crashHotkey());
      const auto gpuCrashHotkeyStr = buildKeyBindDescriptorStringForDisplay(RtxOptions::gpuCrashHotkey());
      const auto warningText = str::format("!! CRASH HOTKEYS ARMED (", crashHotkeyStr, " = CPU crash, ", gpuCrashHotkeyStr, " = GPU crash) !!");
      const ImVec2 textSize = ImGui::CalcTextSize(warningText.c_str());
      const ImGuiViewport* viewport = ImGui::GetMainViewport();
      const ImVec2 textPos(viewport->Size.x - textSize.x - 10.0f, 10.0f);
      ImGui::GetForegroundDrawList()->AddText(textPos, IM_COL32(255, 50, 50, 255), warningText.c_str());
    }
#endif

    ImGui::Render();
  }

  void ImGUI::showMainMenu(const Rc<DxvkContext>& ctx) {
    // Target rtx.conf layer for developer menu changes
    RtxOptionLayerTarget layerTarget(RtxOptionEditTarget::User);

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(m_windowOnRight ? viewport->Size.x - m_windowWidth : 0.f, viewport->Pos.y));
    ImGui::SetNextWindowSize(ImVec2(m_windowWidth, viewport->Size.y));

    // Remember switch state first, the switch UI when the curent window is finished.
    int switchUI = -1;
    bool advancedMenuOpen = RtxOptions::showUI() == UIType::Advanced;

    if (ImGui::Begin("RTX Remix Developer Menu", &advancedMenuOpen, windowFlags)) {
      // Begin handles window resize so this is fine. Do not set m_windowWidth after tabs so that tabs can modify the width
      m_windowWidth = ImGui::GetWindowWidth();

      if (ImGui::Button("Graphics Settings Menu", ImVec2(ImGui::GetContentRegionAvail().x * 0.5f, 0))) {
        switchUI = (int) UIType::Basic;
      }

      ImGui::SameLine();
      RemixGui::Checkbox("Default Menu", &RtxOptions::defaultToAdvancedUIObject());
      
      RemixGui::Separator();

      const static ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_Reorderable | ImGuiTabBarFlags_NoCloseWithMiddleMouseButton;
      const static ImGuiTabItemFlags tab_item_flags = ImGuiTabItemFlags_NoCloseWithMiddleMouseButton;

      // Tab Bar
      if (ImGui::BeginTabBar("Developer Tabs", tab_bar_flags)) {
        for (int n = 0; n < kTab_Count; n++) {
          auto tabItemFlags = tab_item_flags;
          if(n == m_triggerTab) {
            tabItemFlags |= ImGuiTabItemFlags_SetSelected;
            m_triggerTab = kTab_Count;
          }
          if (ImGui::BeginTabItem(tabNames[n], nullptr, tabItemFlags)) {
            const Tabs tab = (Tabs) n;
            switch (tab) {
            case kTab_Rendering:
              showRenderingSettings(ctx);
              break;
            case kTab_About:
              m_about->show(ctx);
              break;
            case kTab_Development:
              showDevelopmentSettings(ctx);
              break;
            case kTab_Count:
              assert(false && "kTab_Count hit in ImGUI::showMainMenu");
              break;
            }
            m_curTab = tab;
            ImGui::EndTabItem();
          }
        }

        if (ImGui::TabItemButton(m_windowOnRight ? "<<" : ">>")) {
          m_windowOnRight = !m_windowOnRight;
        }

        ImGui::EndTabBar();
      }
    }

    ImGui::Dummy(ImVec2(0, 2));
    RemixGui::Separator();
    ImGui::Dummy(ImVec2(0, 2));

    // Get layer pointers and check for unsaved changes
    RtxOptionLayer* rtxConfLayer = RtxOptionLayer::getRtxConfLayer();
    RtxOptionLayer* userLayer = const_cast<RtxOptionLayer*>(RtxOptionLayer::getUserLayer());
    const bool rtxHasUnsaved = rtxConfLayer && rtxConfLayer->hasUnsavedChanges();
    const bool userHasUnsaved = userLayer && userLayer->hasUnsavedChanges();
    
    // ============================================================================
    // Settings Management Section
    // ============================================================================
    if (RemixGui::CollapsingHeader("Settings Management", ImGuiTreeNodeFlags_DefaultOpen)) {

      // --- User Config Layer (higher priority, shown first) ---
      ImGui::Text("User Settings (user.conf):");
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Your personal preferences. Saved to `user.conf`.\n"
          "This includes graphics preset choices, direct graphics settings,\n"
          "and other per-user preferences like UI configuration.");
      }
      if (userHasUnsaved) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.0f, 1.0f), "(unsaved changes)");
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("Changes have been made since the user.conf file was last saved.");
        }
      }

      // Show unsaved changes in a CollapsingHeader
      if (userHasUnsaved && userLayer) {
        if (RemixGui::CollapsingHeader("View Changes##User")) {
          ImGui::Indent();
          OptionLayerUI::RenderOptions renderOpts;
          renderOpts.uniqueId = "##UserLayerList";
          OptionLayerUI::renderToImGui(userLayer, renderOpts);
          ImGui::Unindent();
        }
      }
      
      OptionLayerUI::renderLayerButtons(userLayer, "User");

      ImGui::Separator();

      // --- Migration: Move miscategorized options from user.conf to rtx.conf ---
      // Get cached count of options in user.conf that don't have UserSetting flag
      const uint32_t userMiscategorizedCount = userLayer ? userLayer->countMiscategorizedOptions() : 0;

      if (userMiscategorizedCount > 0) {
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.65f, 0.0f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.95f, 0.75f, 0.1f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.75f, 0.55f, 0.0f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
        
        std::string buttonLabel = str::format("Migrate ", userMiscategorizedCount, " Developer Setting", (userMiscategorizedCount > 1 ? "s" : ""), " to rtx.conf");
        if (ImGui::Button(buttonLabel.c_str(), ImVec2(-1, 0))) {
          const uint32_t migratedCount = userLayer->migrateMiscategorizedOptions();
          Logger::info(str::format("[RTX Option]: Migrated ", migratedCount, " developer settings from user.conf to rtx.conf"));
        }
        
        ImGui::PopStyleColor(4);

        // Tooltip needs to come after the PopStyleColor button to avoid having black text on a dark background
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("The user.conf file should only contain end user options (preferences and graphics quality settings).\n\n"
                            "This button will move all other settings from user.conf to rtx.conf.\n"
                            "It does not save the changes.");
        }
      }

      ImGui::Spacing();
      ImGui::Spacing();
      ImGui::Spacing();

      // --- RTX Config Layer ---
      ImGui::Text("Remix Config (rtx.conf):");
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("The main place where Remix configuration is stored.\n"
          "Saves to rtx.conf. This should be where mod\n"
          "developers configure game-specific settings.");
      }
      if (rtxHasUnsaved) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.0f, 1.0f), "(unsaved changes)");
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("Changes have been made since the rtx.conf file was last saved.");
        }
      }

      // Show unsaved changes in a CollapsingHeader
      if (rtxHasUnsaved && rtxConfLayer) {
        if (RemixGui::CollapsingHeader("View Changes##RtxConf")) {
          ImGui::Indent();
          OptionLayerUI::RenderOptions renderOpts;
          renderOpts.uniqueId = "##RtxConfLayerList";
          OptionLayerUI::renderToImGui(rtxConfLayer, renderOpts);
          ImGui::Unindent();
        }
      }
      
      OptionLayerUI::renderLayerButtons(rtxConfLayer, "RtxConf");

      // --- Migration: Move miscategorized options from rtx.conf to user.conf ---
      const uint32_t rtxMiscategorizedCount = rtxConfLayer ? rtxConfLayer->countMiscategorizedOptions() : 0;
      if (rtxMiscategorizedCount > 0) {
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.65f, 0.0f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.95f, 0.75f, 0.1f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.75f, 0.55f, 0.0f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
        
        std::string buttonLabel = str::format("Migrate ", rtxMiscategorizedCount, " User Setting", (rtxMiscategorizedCount > 1 ? "s" : ""), " to user.conf");
        if (ImGui::Button(buttonLabel.c_str(), ImVec2(-1, 0))) {
          const uint32_t migratedCount = rtxConfLayer->migrateMiscategorizedOptions();
          Logger::info(str::format("[RTX Option]: Migrated ", migratedCount, " user settings from rtx.conf to user.conf"));
        }
        
        ImGui::PopStyleColor(4);

        // Tooltip needs to come after the PopStyleColor button to avoid having black text on a dark background
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("The rtx.conf file contains settings that should be in user.conf (end user options that should not be in mods).\n\n"
                            "This button will move these settings from rtx.conf to user.conf.\n"
                            "It does not save the changes.");
        }
      }
    }

    ImGui::Spacing();

    // --- Bottom Buttons ---
    const float buttonWidth = ImGui::GetContentRegionAvail().x / 2 - (ImGui::GetStyle().ItemSpacing.x / 2);
    const bool anyUnsavedChanges = rtxHasUnsaved || userHasUnsaved;
    ImGui::BeginDisabled(!anyUnsavedChanges);
    if (ImGui::Button("Revert All Unsaved Changes", ImVec2(buttonWidth, 0))) {
      // Reload all layers that have unsaved changes
      if (rtxConfLayer && rtxConfLayer->hasUnsavedChanges()) {
        rtxConfLayer->reload();
      }
      if (userLayer && userLayer->hasUnsavedChanges()) {
        userLayer->reload();
      }
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
      if (anyUnsavedChanges) {
        ImGui::SetTooltip("Reload rtx.conf and user.conf from disk,\ndiscarding all unsaved changes in both layers.");
      } else {
        ImGui::SetTooltip("No unsaved changes to revert.");
      }
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    if (ImGui::Button("Hide UI", ImVec2(buttonWidth, 0))) {
      switchUI = (int) UIType::None;
    }

    RemixGui::TextCentered("[Alt + Del] Toggle cursor        [Alt + Backspace] Toggle game input");
    ImGui::End();

    // Close via titlebar close button
    if (!advancedMenuOpen) {
      switchUI = (int) UIType::None;
    }

    if (switchUI >= 0) {
      switchMenu((UIType) switchUI);
    }
  }

  struct HudMessage {
    HudMessage(const std::string& text, const std::optional<std::string>& subText) : text{ text }, subText{ subText } {}

    std::string text;
    std::optional<std::string> subText;
  };

  void ImGUI::showHudMessages(const Rc<DxvkContext>& ctx) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    auto common = ctx->getCommonObjects();
    const auto& pipelineManager = common->pipelineManager();

    std::vector<HudMessage> hudMessages;

    // Add Shader Compilation HUD messages

    uint32_t asyncShaderCompilationCount = 0;
    if (RtxOptions::Shader::enableAsyncCompilation()) {
      asyncShaderCompilationCount = pipelineManager.remixShaderCompilationCount();
    }

    if (RtxOptions::Shader::enableAsyncCompilationUI() && asyncShaderCompilationCount > 0) {
      const auto compilationText = str::format("Compiling shaders (", asyncShaderCompilationCount, " remaining)");

      hudMessages.emplace_back(std::move(compilationText), "This may take some time if shaders are not cached yet.\nRemix will not render properly until compilation is finished.");
    }

    // Draw Hud Messages

    if (!hudMessages.empty()) {
      // Reset Hud Message time if needed
      // Note: This is done to minimize any potential precision issues if the game is left running for a long time.
      // Not the best solution ever, ideally just accumulating time with delta time passed in would probably be better
      // rather than querying the OS for timestamps, but getting delta time in the ImGui system is rather annoying, so
      // this is fine for now as this code isn't performance-critical anyways.

      const auto currentTime = std::chrono::steady_clock::now();

      if (!m_hudMessageTimeReset) {
        m_hudMessageStartTime = currentTime;
        m_hudMessageTimeReset = true;
      }

      // Calculate the length of the animated dot sequence based on the current time

      const auto hudMessageDisplayDuration{ currentTime - m_hudMessageStartTime };
      const auto hudMessageDisplayMilliseconds{
        std::chrono::duration_cast<std::chrono::milliseconds>(hudMessageDisplayDuration).count()
      };
      // Note: Generates a looping set of values in the range [1, 3] based on the time and the duration of each dot.
      const auto dotSequenceLength{ (hudMessageDisplayMilliseconds / hudMessageAnimatedDotDurationMilliseconds()) % 3 + 1 };

      ImGui::SetNextWindowPos(ImVec2(0, viewport->Size.y), ImGuiCond_Always, ImVec2(0.0f, 1.0f));
      // Note: 368 pixels chosen as a minimum width for the message box width to ensure the current length of text has enough space
      // to render an animated dot sequence without causing the width of the window to change, as this is visually distracting.
      // If longer message box text fields are ever desired than the current ones, this number will have to be updated.
      // Hack: Currently ImGui does not properly respect the window size constraints when ImGuiWindowFlags_AlwaysAutoResize is set.
      // This call should be using the size constraints (368, -1), (-1, -1) as -1 indicates "don't care" (and we only care about setting
      // a minimum width), but for some reason that does not work as reported by this bug: https://github.com/ocornut/imgui/issues/2629
      ImGui::SetNextWindowSizeConstraints(ImVec2(368.0f, 0.0f), ImVec2(1000.0f, 1000.0f));
      ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.5f, 0.2f, 0.2f, 0.35f));

      const ImGuiWindowFlags hud_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;
      if (ImGui::Begin("HUD", nullptr, hud_flags)) {
        for (std::size_t i{ 0U }; i < hudMessages.size(); ++i) {
          auto&& message{ hudMessages[i] };

          // Append the animated dot sequence to the message main text

          constexpr auto dotSequence{ "..." };
          std::string animatedMessageText{ message.text };

          animatedMessageText.append(dotSequence, dotSequenceLength);

          // Add a large main text and smaller sub text for each message

          ImGui::PushFont(m_largeFont);
          ImGui::Text(animatedMessageText.c_str());
          ImGui::PopFont();

          if (message.subText) {
            ImGui::Text(message.subText->c_str());
          }

          // Add a seperator between messages

          if (i != hudMessages.size() - 1) {
            RemixGui::Separator();
          }
        }
      }

      ImGui::PopStyleColor();
      ImGui::End();
    } else {
      // Note: Indicate that the Hud Message time will need to be reset the next time it is used.
      m_hudMessageTimeReset = false;
    }
  }

  void ImGUI::showDevelopmentSettings(const Rc<DxvkContext>& ctx) {
    ImGui::PushItemWidth((largeUiMode() ? m_largeWindowWidgetWidth : m_regularWindowWidgetWidth) + 50.0f);
    if (ImGui::Button("Take Screenshot")) {
      RtxContext::triggerScreenshot();
    }

    RemixGui::SetTooltipToLastWidgetOnHover("Screenshot will be dumped to, '<exe-dir>/Screenshots'");


    RemixGui::Separator();

    if (RemixGui::CollapsingHeader("Nsight Graphics Capture", collapsingHeaderClosedFlags)) {
      ImGui::Indent();

#if DXVK_ENABLE_NSIGHT_GRAPHICS_CAPTURE
      RemixGui::Checkbox("Enable Nsight Graphics Capture On Launch", &NsightGraphicsCapture::graphicsCaptureEnabledObject());
      RemixGui::InputText("Nsight Graphics Install Path", &NsightGraphicsCapture::graphicsCaptureInstallPathObject());
      RemixGui::InputText("Capture Output Directory", &NsightGraphicsCapture::graphicsCaptureOutputDirObject());
      RemixGui::InputText("Capture Output File", &NsightGraphicsCapture::graphicsCaptureOutputFileObject());

      RemixGui::DragInt("Capture Frames", &NsightGraphicsCapture::graphicsCaptureFramesToCaptureObject(), 1.0f,
                        NsightGraphicsCapture::kMinFramesToCapture, NsightGraphicsCapture::kMaxFramesToCapture, "%d",
                        ImGuiSliderFlags_AlwaysClamp);
      RemixGui::Checkbox("Show Nsight HUD", &NsightGraphicsCapture::graphicsCaptureShowHudObject());

      ImGui::TextWrapped("Status: %s", NsightGraphicsCapture::statusText().c_str());
      ImGui::TextWrapped("Last Result: %s", NsightGraphicsCapture::lastResultText().c_str());

      ImGui::BeginDisabled(!NsightGraphicsCapture::isAvailable());
      if (ImGui::Button("Trigger Nsight Graphics Capture")) {
        NsightGraphicsCapture::requestGraphicsCapture(NsightGraphicsCapture::graphicsCaptureFramesToCapture());
      }
      ImGui::EndDisabled();
#else
      ImGui::TextWrapped("Nsight Graphics capture requires an x64 or ARM64EC build. See the build instructions in README.md.");
#endif

      ImGui::Unindent();
    }

    RemixGui::Separator();

    // NV-DXVK start: built-in GPU pass timings
    if (RemixGui::CollapsingHeader("GPU Pass Timings", collapsingHeaderClosedFlags)) {
      ImGui::Indent();
      ctx->getCommonObjects()->metaGpuPassTimer().showImguiSettings();
      ImGui::Unindent();
    }

    RemixGui::Separator();
    // NV-DXVK end
        
#ifdef REMIX_DEVELOPMENT
    { // Recompile Shaders button and its status information (Only available for Development Remix builds)
      const auto& shaderManager{ ShaderManager::getInstance() };
      const auto shaderReloadPhase{ shaderManager->getShaderReloadPhase() };
      const auto lastShaderReloadStatus{ shaderManager->getLastShaderReloadStatus() };

      // Note: Only allow the Recompile Shaders button to function if a shader recompile is not currently in progress (be
      // it one manually initiated by the user, or something automatic from the live shader edit mode).
      ImGui::BeginDisabled(shaderReloadPhase != ShaderManager::ShaderReloadPhase::Idle);

      if (ImGui::Button("Recompile Shaders")) {
        shaderManager->requestReloadShaders();
      }

      ImGui::EndDisabled();

      ImGui::SameLine(200.f);
      RemixGui::Checkbox("Live shader edit mode", &RtxOptions::Shader::useLiveEditModeObject());

      const char* shaderReloadPhaseText;
      const char* lastShaderReloadStatusText;
      ImVec4 shaderReloadPhaseTextColor;
      ImVec4 lastShaderReloadStatusTextColor;

      switch (shaderReloadPhase) {
      default: assert(false); [[fallthrough]];
      case ShaderManager::ShaderReloadPhase::Idle:
        shaderReloadPhaseText = "Idle";
        shaderReloadPhaseTextColor = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
        break;
      case ShaderManager::ShaderReloadPhase::SPIRVRecompilation:
        shaderReloadPhaseText = "Working (SPIR-V Recompilation)";
        shaderReloadPhaseTextColor = ImVec4(0.73f, 0.87f, 0.54f, 1.0f);
        break;
      case ShaderManager::ShaderReloadPhase::ShaderRecreation:
        shaderReloadPhaseText = "Working (Shader Recreation)";
        shaderReloadPhaseTextColor = ImVec4(0.73f, 0.87f, 0.54f, 1.0f);
        break;
      }

      switch (lastShaderReloadStatus) {
      default: assert(false); [[fallthrough]];
      case ShaderManager::ShaderReloadStatus::Unknown:
        lastShaderReloadStatusText = "N/A";
        lastShaderReloadStatusTextColor = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
        break;
      case ShaderManager::ShaderReloadStatus::Failure:
        lastShaderReloadStatusText = "Failure";
        lastShaderReloadStatusTextColor = ImVec4(0.83f, 0.32f, 0.32f, 1.0f);
        break;
      case ShaderManager::ShaderReloadStatus::Success:
        lastShaderReloadStatusText = "Success";
        lastShaderReloadStatusTextColor = ImVec4(0.44f, 0.81f, 0.42f, 1.0f);
        break;
      }

      ImGui::TextUnformatted("Shader Reload Phase:");
      ImGui::SameLine();
      ImGui::PushStyleColor(ImGuiCol_Text, shaderReloadPhaseTextColor);
      ImGui::TextUnformatted(shaderReloadPhaseText);
      ImGui::PopStyleColor();

      ImGui::TextUnformatted("Last Shader Reload Status:");
      ImGui::SameLine();
      ImGui::PushStyleColor(ImGuiCol_Text, lastShaderReloadStatusTextColor);
      ImGui::TextUnformatted(lastShaderReloadStatusText);
      ImGui::PopStyleColor();
    }

    ImGui::Separator();

    { // Crash Hotkey Feature - arms both CPU crash (deliberate crash) and GPU crash (dialog + Sentry) hotkeys
      const bool isArmed = RtxOptions::enableCrashHotkey();
      
      // Use warning color when armed to make it visually distinct
      if (isArmed) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
      }
      
      const bool changed = RemixGui::Checkbox("Arm Crash Hotkey", &RtxOptions::enableCrashHotkeyObject());
      
      if (isArmed) {
        ImGui::PopStyleColor();
      }
      
      const auto crashHotkeyStr = buildKeyBindDescriptorStringForDisplay(RtxOptions::crashHotkey());
      const auto gpuCrashHotkeyStr = buildKeyBindDescriptorStringForDisplay(RtxOptions::gpuCrashHotkey());
      RemixGui::SetTooltipToLastWidgetOnHover(
        str::format("When armed: ", crashHotkeyStr, " = deliberate CPU crash; ", gpuCrashHotkeyStr, " = deliberate GPU crash.\n"
        "A red warning indicator will appear on screen while armed.").c_str());
      
      // Log state changes for crash dump analysis
      if (changed) {
        const bool nowArmed = RtxOptions::enableCrashHotkey();
        if (nowArmed) {
          Logger::warn(str::format("Crash hotkeys ARMED - ", crashHotkeyStr, " = CPU crash, ", gpuCrashHotkeyStr, " = GPU crash"));
        } else {
          Logger::warn("Crash hotkeys disarmed");
        }
      }
    }
    
    RemixGui::Separator();
#endif


    showVsyncOptions(false);

    // Render GUI for memory profiler here
    GpuMemoryTracker::renderGui();



    if (RemixGui::CollapsingHeader("Developer Options", collapsingHeaderFlags)) {
      ImGui::Indent();
      RemixGui::Checkbox("Break into Debugger On Press of Key 'B'", &RtxOptions::enableBreakIntoDebuggerOnPressingBObject());
      RemixGui::Checkbox("Block Input to Game in UI", &RtxOptions::blockInputToGameInUIObject());

      ImGui::Unindent();
      RemixGui::Checkbox("Throttle presents", &RtxOptions::enablePresentThrottleObject());
      if (RtxOptions::enablePresentThrottle()) {
        ImGui::Indent();
        RemixGui::SliderInt("Present delay", &RtxOptions::presentThrottleDelayObject(), 1, 1000, "%d ms", sliderFlags);
        ImGui::Unindent();
      }
    }

    if (IMGUI_ADD_TOOLTIP(RemixGui::CollapsingHeader("Option Layers"), "View what options are present in each layer, and alter the blend strength and threshold for them.")) {
      ImGui::Indent();
      static char optionLayerFilter[256] = "";
      // Filter for option layer contents
      IMGUI_ADD_TOOLTIP(ImGui::InputText("RtxOption Display Filter", optionLayerFilter, IM_ARRAYSIZE(optionLayerFilter)), 
          "Filter options displayed in the Contents sections. Only options containing this text will be shown.");

        // Pre-compute lowercased filter once for efficiency
      std::string filterLower = optionLayerFilter;
      std::transform(filterLower.begin(), filterLower.end(), filterLower.begin(), ::tolower);

      uint32_t optionLayerCounter = 1;
      for (auto& [layerKey, optionLayerPtr] : RtxOptionManager::getLayerRegistry()) {
        RtxOptionLayer& optionLayer = *optionLayerPtr;

        const bool isDefaultLayer = layerKey == kRtxOptionLayerDefaultKey;
        const bool isUserLayer = layerKey == kRtxOptionLayerUserKey;
        const bool isQualityLayer = layerKey == kRtxOptionLayerQualityKey;
        const bool isDxvkLayer = layerKey == kRtxOptionLayerDxvkConfKey;
        const bool isRtxConfLayer = layerKey == kRtxOptionLayerRtxConfKey;
        const bool hasSaveableConfig = optionLayer.hasSaveableConfigFile() && !isDxvkLayer;
        const bool hasUnsaved = hasSaveableConfig && optionLayer.hasUnsavedChanges();

        // Skip layers with no values (empty layers), unless they have a saveable config
        // Dynamic layers with saveable configs should still be shown even when empty/disabled
        if (!optionLayer.hasValues() && !hasSaveableConfig) {
          continue;
        }
        
        // Determine display name - system layers have proper names,
        // dynamically loaded layers have file paths as names which need shortening
        std::string displayName = optionLayer.getName();
        
        // Add config file indicator for layers with associated config files
        if (isRtxConfLayer) {
          displayName += " (rtx.conf)";
        } else if (isUserLayer) {
          displayName += " (user.conf)";
        } else if (isDxvkLayer) {
          displayName += " (dxvk.conf)";
        }
        
        // For non-system layers, shorten long file paths for display
        if (displayName.length() > 40) {
          // Try to extract just the file name from the path
          const std::string modsMarker = (std::filesystem::path("rtx-remix") / "mods" / "").string();
          size_t modsPos = displayName.find(modsMarker);
          if (modsPos != std::string::npos) {
            displayName = displayName.substr(modsPos + modsMarker.length());
          } else {
            // Just take the last portion of the path
            size_t lastSep = displayName.find_last_of("/\\");
            if (lastSep != std::string::npos) {
              displayName = displayName.substr(lastSep + 1);
            }
          }
        }
        
        // Build header text and styling
        std::string unsavedIndicator = hasUnsaved ? " *" : "";
        const std::string optionLayerText = std::to_string(optionLayerCounter++) + ". " + displayName + unsavedIndicator + "###" + displayName;
        
        // Determine if layer is active (only applies to layers with blend controls)
        bool pendingEnabled = optionLayer.getPendingEnabled();
        float pendingStrength = optionLayer.getPendingBlendStrength();
        float pendingThreshold = optionLayer.getPendingBlendThreshold();
        const bool isLayerActive = !hasSaveableConfig || (pendingEnabled && pendingStrength > pendingThreshold);
        
        // Apply header styling
        bool pushedStyle = false;
        if (!isLayerActive) {
          ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
          pushedStyle = true;
        } else if (hasUnsaved) {
          ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.2f, 1.0f));
          pushedStyle = true;
        }
        
        // Build tooltip
        std::string tooltipText;
        if (isQualityLayer) {
          tooltipText = "Quality preset settings (highest priority). Empty when Graphics Preset is Custom.";
        } else if (isUserLayer) {
          tooltipText = "A User's local graphics settings.  Overrides all other layers except Quality Presets.";
        } else if (isDefaultLayer) {
          tooltipText = "Default values for each option, as defined in RtxOptions.md.";
        } else {
          tooltipText = optionLayer.getName();
        }
        if (!optionLayer.getFilePath().empty() && optionLayer.getFilePath() != optionLayer.getName()) {
          tooltipText += "\nFile: " + optionLayer.getFilePath();
        }
        if (hasUnsaved) {
          tooltipText += "\n[Has unsaved changes]";
        }
        
        bool headerOpen = IMGUI_ADD_TOOLTIP(RemixGui::CollapsingHeader(optionLayerText.c_str(), collapsingHeaderClosedFlags), tooltipText.c_str());
        
        if (pushedStyle) {
          ImGui::PopStyleColor();
        }
        
        if (headerOpen) {
          ImGui::Indent();
          
          // Priority display
          if (isQualityLayer) {
            ImGui::Text("Priority: MAX");
            if (ImGui::IsItemHovered()) {
              ImGui::SetTooltip("Highest possible priority - quality preset settings control these options when preset is not Custom.\nThis layer is empty when Graphics Preset is set to Custom.");
            }
          } else if (isUserLayer) {
            ImGui::Text("Priority: MAX - 1");
            if (ImGui::IsItemHovered()) {
              ImGui::SetTooltip("Second highest priority - user settings that override all layers except Quality Presets.\nWhen Graphics Preset is Custom, this becomes the effective highest priority layer.");
            }
          } else if (isDefaultLayer) {
            ImGui::Text("Priority: 0");
            if (ImGui::IsItemHovered()) {
              ImGui::SetTooltip("Lowest possible priority - every other layer will be applied on top of this layer.");
            }
          } else {
            ImGui::Text("Priority: %u", optionLayer.getLayerKey().priority);
            if (ImGui::IsItemHovered()) {
              ImGui::SetTooltip(
                "Layers are applied starting with the lowest priority layer, ending with the highest.\n"
                "Each layer overrides the values written before it.\n"
                "If a layer's blendWeight is not 1 and the option is a float or Vector type,\n"
                "then the values will be calculated as LERP(previousValue, layerValue, blendWeight).");
            }
          }
          
          // Enable/blend controls only for saveable config layers (Remix Config, User, dynamically loaded mods)
          if (hasSaveableConfig && !isUserLayer) {
            const std::string optionLayerEnabledText = "Enabled###Enabled_" + displayName;
            const std::string optionLayerStrengthText = " Strength###Strength_" + displayName;
            const std::string optionLayerThresholdText = " Threshold###Threshold_" + displayName;
            
            if (IMGUI_ADD_TOOLTIP(ImGui::Checkbox(optionLayerEnabledText.c_str(), &pendingEnabled), "Check to enable the option layer. Uncheck to disable it.")) {
              optionLayer.requestEnabled(pendingEnabled);
            }

            if (IMGUI_ADD_TOOLTIP(ImGui::SliderFloat(optionLayerStrengthText.c_str(), &pendingStrength, 0.0f, 1.0f),
                                  "Adjusts the blending strength of this option layer (0 = off, 1 = full effect).")) {
              optionLayer.requestBlendStrength(pendingStrength);
            }

            if (IMGUI_ADD_TOOLTIP(ImGui::SliderFloat(optionLayerThresholdText.c_str(), &pendingThreshold, 0.0f, 1.0f),
                                  "Sets the blending strength threshold for this option layer.")) {
              optionLayer.requestBlendThreshold(pendingThreshold);
            }
          }
          
          // Action buttons only for saveable config layers
          if (hasSaveableConfig) {
            OptionLayerUI::renderLayerButtons(optionLayerPtr.get(), displayName.c_str());
          }
          
          // Contents section
          const std::string optionLayerContentsText = "Contents###Contents_" + displayName;
          if (RemixGui::CollapsingHeader(optionLayerContentsText.c_str(), collapsingHeaderClosedFlags)) {
            ImGui::Indent();
            OptionLayerUI::displayContents(optionLayer, filterLower);
            ImGui::Unindent();
          }
          
          ImGui::Unindent();
        }
      }

      ImGui::Unindent();
    }

    if (RemixGui::CollapsingHeader("UI Options")) {
      ImGui::Indent();

      if (m_pendingUIOptionsScroll) {
        ImGui::SetScrollHereY(0.0f);
        m_pendingUIOptionsScroll = false;
      }

      {
        if (RemixGui::Checkbox("Compact UI", &compactGuiObject())) {
          // Scroll to UI Options on the next frame
          m_pendingUIOptionsScroll = true;
        }
      }

      RemixGui::Checkbox("Always Developer Menu", &RtxOptions::defaultToAdvancedUIObject());

      if (RemixGui::SliderFloat("Background Alpha", &backgroundAlphaObject(), 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
        adjustStyleBackgroundAlpha(backgroundAlpha());
      }

      
      if (RemixGui::Checkbox("Use Large UI", &largeUiModeObject())) {
        m_pendingUIOptionsScroll = true;
      }

      {
        constexpr float indent = 60.0f;
        ImGui::PushID("gui theme");
        ImGui::Dummy(ImVec2(0, 2));
        ImGui::Text("GUI Theme:");
        ImGui::PushItemWidth(ImGui::GetContentRegionMax().x - indent);

        if (themeCombo.getKey(&themeGuiObject())) {
          m_pendingUIOptionsScroll = true;
        }

        ImGui::PopItemWidth();
        ImGui::PopID();
      }

      ImGui::Unindent();
    }

    ImGui::PopItemWidth();
  }

  void ImGUI::adjustStyleBackgroundAlpha(const float& alpha, ImGuiStyle* dst) {
    ImGuiStyle* style = dst ? dst : &ImGui::GetStyle();
    ImVec4& currColor = style->Colors[ImGuiCol_WindowBg];
    currColor.w = alpha;
  }

  void ImGUI::updateWindowWidths() {
    // Developer menu
    m_windowWidth = largeUiMode() ? m_largeWindowWidth : m_regularWindowWidth + (compactGui() ? 0.0f : 42.0f);

    // User menu popup
    m_userWindowWidth = largeUiMode() ? m_largeUserWindowWidth : m_regularUserWindowWidth;
    m_userWindowHeight = largeUiMode() ? m_largeUserWindowHeight : m_regularUserWindowHeight;
  }

  void ImGUI::setToolkitStyle(ImGuiStyle* dst) {
    ImGuiStyle* style = dst ? dst : &ImGui::GetStyle();

    style->Alpha = 1.0f;
    style->DisabledAlpha = 0.5f;

    style->WindowPadding = ImVec2(8.0f, 10.0f);
    style->FramePadding = compactGui() ? ImVec2(4.0f, 3.0f) : ImVec2(5.0f, 4.0f);
    style->CellPadding = ImVec2(5.0f, 4.0f);
    style->ItemSpacing = compactGui() ? ImVec2(8.0f, 4.0f) : ImVec2(3.0f, 5.0f);
    style->ItemInnerSpacing = ImVec2(4.0f, 4.0f);
    style->IndentSpacing = 10.0f;
    style->ColumnsMinSpacing = 10.0f;
    style->ScrollbarSize = 15.0f;
    style->GrabMinSize = 10.0f;

    style->WindowBorderSize = 1.5f;
    style->ChildBorderSize = 1.5f;
    style->PopupBorderSize = 1.5f;
    style->FrameBorderSize = 1.5f;
    style->TabBorderSize = 0.0f;

    style->WindowRounding = 0.0f;
    style->ChildRounding = 2.0f;
    style->FrameRounding = 3.0f;
    style->PopupRounding = 2.0f;
    style->ScrollbarRounding = 2.0f;
    style->GrabRounding = 1.0f;
    style->TabRounding = 2.0f;
    style->WindowMenuButtonPosition = ImGuiDir_None;

    style->WindowMinSize = ImVec2(32, 32);
    style->TouchExtraPadding = ImVec2(0, 0);
    style->LogSliderDeadzone = 4.0f;
    style->TabMinWidthForCloseButton = 0.0f;
    style->DisplayWindowPadding = ImVec2(19, 19);
    style->DisplaySafeAreaPadding = ImVec2(3, 3);
    style->MouseCursorScale = 1.0f;

    style->Colors[ImGuiCol_WindowBg] = ImVec4(0.188f, 0.188f, 0.188f, backgroundAlpha());
    style->Colors[ImGuiCol_PopupBg] = ImVec4(0.188f, 0.188f, 0.188f, 1.00f);
    style->Colors[ImGuiCol_Text] = ImVec4(0.8f, 0.8f, 0.8f, 1.00f);
    style->Colors[ImGuiCol_TextDisabled] = ImVec4(0.44f, 0.44f, 0.44f, 1.00f);
    style->Colors[ImGuiCol_ChildBg] = ImVec4(0.16f, 0.16f, 0.16f, 0.86f);
    style->Colors[ImGuiCol_Border] = ImVec4(0.34f, 0.34f, 0.34f, 1.0f);
    style->Colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    style->Colors[ImGuiCol_FrameBg] = ImVec4(0.188f, 0.188f, 0.188f, 1.00f);
    style->Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.15f, 0.30f, 0.35f, 1.00f);
    style->Colors[ImGuiCol_FrameBgActive] = ImVec4(0.10f, 0.15f, 0.16f, 0.59f);
    style->Colors[ImGuiCol_TitleBg] = ImVec4(0.06f, 0.06f, 0.06f, 1.00f);
    style->Colors[ImGuiCol_TitleBgActive] = ImVec4(0.06f, 0.06f, 0.06f, 1.00f);
    style->Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.15f, 0.15f, 0.15f, 0.98f);
    style->Colors[ImGuiCol_MenuBarBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    style->Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.24f);
    style->Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    style->Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.31f, 0.31f, 0.31f, 0.78f);
    style->Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.78f, 0.78f, 0.78f, 0.33f);
    style->Colors[ImGuiCol_CheckMark] = ImVec4(0.53f, 0.53f, 0.53f, 1.00f);
    style->Colors[ImGuiCol_SliderGrab] = ImVec4(1.00f, 1.00f, 1.00f, 0.39f);
    style->Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.10f, 0.46f, 0.56f, 1.00f);
    style->Colors[ImGuiCol_Button] = ImVec4(0.121f, 0.129f, 0.141f, 1.00f);
    style->Colors[ImGuiCol_ButtonHovered] = ImVec4(0.27f, 0.27f, 0.27f, 1.00f);
    style->Colors[ImGuiCol_ButtonActive] = ImVec4(0.40f, 0.44f, 0.45f, 1.00f);
    style->Colors[ImGuiCol_Header] = ImVec4(0.125f, 0.125f, 0.125f, 1.00f);
    style->Colors[ImGuiCol_HeaderHovered] = ImVec4(0.17f, 0.25f, 0.27f, 0.78f);
    style->Colors[ImGuiCol_HeaderActive] = ImVec4(0.17f, 0.25f, 0.27f, 0.78f);
    style->Colors[ImGuiCol_Separator] = ImVec4(0.35f, 0.35f, 0.35f, 1.00f);
    style->Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.15f, 0.52f, 0.66f, 0.30f);
    style->Colors[ImGuiCol_SeparatorActive] = ImVec4(0.30f, 0.69f, 0.84f, 0.39f);
    style->Colors[ImGuiCol_ResizeGrip] = ImVec4(0.43f, 0.43f, 0.43f, 0.51f);
    style->Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.07f, 0.39f, 0.47f, 0.59f);
    style->Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.30f, 0.69f, 0.84f, 0.39f);
    style->Colors[ImGuiCol_Tab] = ImVec4(0.00f, 0.00f, 0.00f, 0.37f);
    style->Colors[ImGuiCol_TabHovered] = ImVec4(0.22f, 0.33f, 0.36f, 1.00f);
    style->Colors[ImGuiCol_TabActive] = ImVec4(0.11f, 0.42f, 0.51f, 1.00f);
    style->Colors[ImGuiCol_TabUnfocused] = ImVec4(0.00f, 0.00f, 0.00f, 0.16f);
    style->Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(1.00f, 1.00f, 1.00f, 0.24f);
    style->Colors[ImGuiCol_PlotLines] = ImVec4(1.00f, 1.00f, 1.00f, 0.35f);
    style->Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    style->Colors[ImGuiCol_PlotHistogram] = ImVec4(1.00f, 1.00f, 1.00f, 0.35f);
    style->Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    style->Colors[ImGuiCol_TableHeaderBg] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
    style->Colors[ImGuiCol_TableBorderStrong] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    style->Colors[ImGuiCol_TableBorderLight] = ImVec4(0.00f, 0.00f, 0.00f, 0.54f);
    style->Colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.39f);
    style->Colors[ImGuiCol_TableRowBgAlt] = ImVec4(0.11f, 0.42f, 0.51f, 0.35f);
    style->Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    style->Colors[ImGuiCol_DragDropTarget] = ImVec4(0.00f, 0.51f, 0.39f, 0.31f);
    style->Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    style->Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    style->Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.56f);
  }

  void ImGUI::setLegacyStyle(ImGuiStyle* dst) {
    ImGuiStyle* style = dst ? dst : &ImGui::GetStyle();

    // Original ImGui theme from ImGuiStyle::ImGuiStyle()
    style->Alpha = 1.0f;
    style->DisabledAlpha = 0.60f;
    style->WindowPadding = ImVec2(8, 8);
    style->WindowRounding = 0.0f;
    style->WindowBorderSize = 1.0f;
    style->WindowMinSize = ImVec2(32, 32);
    style->WindowTitleAlign = ImVec2(0.0f, 0.5f);
    style->WindowMenuButtonPosition = ImGuiDir_Left;
    style->ChildRounding = 0.0f;
    style->ChildBorderSize = 1.0f;
    style->PopupRounding = 0.0f;
    style->PopupBorderSize = 1.0f;
    style->FramePadding = compactGui() ? ImVec2(4, 3) : ImVec2(7, 5);
    style->FrameRounding = 0.0f;
    style->FrameBorderSize = 0.0f;
    style->ItemSpacing = compactGui() ? ImVec2(8, 4) : ImVec2(3, 5);
    style->ItemInnerSpacing = compactGui() ? ImVec2(4, 4) : ImVec2(3, 8);
    style->CellPadding = ImVec2(4, 2);
    style->TouchExtraPadding = ImVec2(0, 0);
    style->IndentSpacing = 21.0f;
    style->ColumnsMinSpacing = 6.0f;
    style->ScrollbarSize = 14.0f;
    style->ScrollbarRounding = 9.0f;
    style->GrabMinSize = 10.0f;
    style->GrabRounding = 0.0f;
    style->LogSliderDeadzone = 4.0f;
    style->TabRounding = 4.0f;
    style->TabBorderSize = 0.0f;
    style->TabMinWidthForCloseButton = 0.0f;
    style->ColorButtonPosition = ImGuiDir_Right;
    style->ButtonTextAlign = ImVec2(0.5f, 0.5f);
    style->SelectableTextAlign = ImVec2(0.0f, 0.0f);
    style->DisplayWindowPadding = ImVec2(19, 19);
    style->DisplaySafeAreaPadding = ImVec2(3, 3);
    style->MouseCursorScale = 1.0f;
    style->AntiAliasedLines = true;
    style->AntiAliasedLinesUseTex = true;
    style->AntiAliasedFill = true;
    style->CurveTessellationTol = 1.25f;
    style->CircleTessellationMaxError = 0.30f;
    ImGui::StyleColorsDark(style);

    // Remix changes
    style->Colors[ImGuiCol_WindowBg] = ImVec4(0.f, 0.f, 0.f, backgroundAlpha());
    style->Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.f, 0.f, 0.f, 0.4f);
    style->TabRounding = 1;
  }

  void ImGUI::setNvidiaStyle(ImGuiStyle* dst) {
    ImGuiStyle* style = dst ? dst : &ImGui::GetStyle();

    // Based on legacy theme
    setLegacyStyle(style);

    style->Colors[ImGuiCol_Text] = ImVec4(0.91f, 0.91f, 0.91f, 1.00f);
    style->Colors[ImGuiCol_TextDisabled] = ImVec4(0.44f, 0.44f, 0.44f, 1.00f);
    style->Colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.10f, 0.10f, 0.90f);
    style->Colors[ImGuiCol_ChildBg] = ImVec4(0.19f, 0.19f, 0.19f, 0.80f);
    style->Colors[ImGuiCol_PopupBg] = ImVec4(0.28f, 0.28f, 0.28f, 1.00f);
    style->Colors[ImGuiCol_Border] = ImVec4(0.31f, 0.31f, 0.31f, 0.20f);
    style->Colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.23f);
    style->Colors[ImGuiCol_FrameBg] = ImVec4(0.19f, 0.19f, 0.19f, 1.00f);
    style->Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.33f, 0.47f, 0.08f, 1.00f);
    style->Colors[ImGuiCol_FrameBgActive] = ImVec4(0.46f, 0.73f, 0.00f, 1.00f);
    style->Colors[ImGuiCol_TitleBg] = ImVec4(0.20f, 0.20f, 0.20f, 0.98f);
    style->Colors[ImGuiCol_TitleBgActive] = ImVec4(0.15f, 0.15f, 0.15f, 0.98f);
    style->Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.15f, 0.15f, 0.15f, 0.98f);
    style->Colors[ImGuiCol_MenuBarBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    style->Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.24f);
    style->Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.34f, 0.34f, 0.34f, 0.39f);
    style->Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.54f, 0.54f, 0.54f, 0.47f);
    style->Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.78f, 0.78f, 0.78f, 0.33f);
    style->Colors[ImGuiCol_CheckMark] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    style->Colors[ImGuiCol_SliderGrab] = ImVec4(1.00f, 1.00f, 1.00f, 0.39f);
    style->Colors[ImGuiCol_SliderGrabActive] = ImVec4(1.00f, 1.00f, 1.00f, 0.31f);
    style->Colors[ImGuiCol_Button] = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
    style->Colors[ImGuiCol_ButtonHovered] = ImVec4(0.33f, 0.47f, 0.08f, 1.00f);
    style->Colors[ImGuiCol_ButtonActive] = ImVec4(0.46f, 0.73f, 0.00f, 1.00f);
    style->Colors[ImGuiCol_Header] = ImVec4(0.13f, 0.13f, 0.13f, 1.00f);
    style->Colors[ImGuiCol_HeaderHovered] = ImVec4(0.33f, 0.47f, 0.08f, 1.00f);
    style->Colors[ImGuiCol_HeaderActive] = ImVec4(0.33f, 0.47f, 0.08f, 1.00f);
    style->Colors[ImGuiCol_Separator] = ImVec4(0.35f, 0.35f, 0.35f, 1.00f);
    style->Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.32f, 0.46f, 0.06f, 1.00f);
    style->Colors[ImGuiCol_SeparatorActive] = ImVec4(0.46f, 0.73f, 0.00f, 1.00f);
    style->Colors[ImGuiCol_ResizeGrip] = ImVec4(0.43f, 0.43f, 0.43f, 0.51f);
    style->Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.32f, 0.46f, 0.06f, 1.00f);
    style->Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.46f, 0.73f, 0.00f, 1.00f);
    style->Colors[ImGuiCol_Tab] = ImVec4(0.00f, 0.00f, 0.00f, 0.37f);
    style->Colors[ImGuiCol_TabHovered] = ImVec4(0.32f, 0.46f, 0.06f, 1.00f);
    style->Colors[ImGuiCol_TabActive] = ImVec4(0.46f, 0.73f, 0.00f, 1.00f);
    style->Colors[ImGuiCol_TabUnfocused] = ImVec4(0.00f, 0.00f, 0.00f, 0.16f);
    style->Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(1.00f, 1.00f, 1.00f, 0.24f);
    style->Colors[ImGuiCol_PlotLines] = ImVec4(1.00f, 1.00f, 1.00f, 0.35f);
    style->Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    style->Colors[ImGuiCol_PlotHistogram] = ImVec4(1.00f, 1.00f, 1.00f, 0.35f);
    style->Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    style->Colors[ImGuiCol_TableHeaderBg] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
    style->Colors[ImGuiCol_TableBorderStrong] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    style->Colors[ImGuiCol_TableBorderLight] = ImVec4(0.00f, 0.00f, 0.00f, 0.54f);
    style->Colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.39f);
    style->Colors[ImGuiCol_TableRowBgAlt] = ImVec4(0.46f, 0.73f, 0.00f, 1.00f);
    style->Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    style->Colors[ImGuiCol_DragDropTarget] = ImVec4(0.00f, 0.51f, 0.39f, 0.31f);
    style->Colors[ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    style->Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    style->Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    style->Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.56f);
  }

  void ImGUI::onThemeChange(DxvkDevice* device) {
    if (GImGui != nullptr) {
      ImGUI& gui = device->getCommon()->getImgui();
      gui.setupStyle();

    }
  }

  void ImGUI::onBackgroundAlphaChange(DxvkDevice* device) {
    if (GImGui != nullptr) {
      ImGUI& gui = device->getCommon()->getImgui();
      gui.adjustStyleBackgroundAlpha(backgroundAlpha());
    }
  }

  void ImGUI::setupStyle(ImGuiStyle* dst) {
    ImGui::GetIO().FontDefault = largeUiMode() ? m_largeFont : m_regularFont;
    updateWindowWidths();
   
    ImGuiStyle* style = dst ? dst : &ImGui::GetStyle();
    switch (themeGui())
    {
    default:
    case Theme::Toolkit:
      setToolkitStyle(style);
      break;

    case Theme::Legacy:
      setLegacyStyle(style);
      break;
    
    case Theme::Nvidia:
      setNvidiaStyle(style);
      break;
    }
  }

  void ImGUI::showVsyncOptions(bool enableDLFGGuard) {
    // we should never get here without a swapchain, so we must have latched the vsync value already
    assert(RtxOptions::enableVsyncState != EnableVsync::WaitingForImplicitSwapchain);
    
    if (enableDLFGGuard && DxvkDLFG::enable()) {
      ImGui::BeginDisabled();
    }

    bool vsyncEnabled = RtxOptions::enableVsyncState == EnableVsync::On;
    bool changed = RemixGui::Checkbox("Enable V-Sync", &vsyncEnabled);
    if (changed) {
      // option has been toggled manually, so we need to actually store the value in the option.
      // RtxOptions::enableVsyncState will be changed by the onChange handler at the end of the frame.
      RemixGui::CheckRtxOptionPopups(&RtxOptions::enableVsyncObject());
      RtxOptions::enableVsync.setDeferred(vsyncEnabled ? EnableVsync::On : EnableVsync::Off);
    }

    ImGui::BeginDisabled();
    ImGui::Indent();
    ImGui::TextWrapped("This setting overrides the native game's V-Sync setting.");
    ImGui::Unindent();
    ImGui::EndDisabled();
    
    if (enableDLFGGuard && DxvkDLFG::enable()) {
      ImGui::Indent();
      ImGui::TextWrapped("When Frame Generation is active, V-Sync is automatically disabled.");
      ImGui::Unindent();

      ImGui::EndDisabled();
    }
  }

  void ImGUI::showDLFGOptions(const Rc<DxvkContext>& ctx) {
    const bool supportsDLFG = ctx->getCommonObjects()->metaNGXContext().supportsDLFG() && !ctx->getCommonObjects()->metaDLFG().hasDLFGFailed();
    const uint32_t maxInterpolatedFrames = ctx->getCommonObjects()->metaNGXContext().dlfgMaxInterpolatedFrames();
    const bool supportsMultiFrame = maxInterpolatedFrames > 1;

    if (!supportsDLFG) {
      ImGui::BeginDisabled();
    }

    const bool dlfgChanged = RemixGui::Checkbox("Enable DLSS Frame Generation", &DxvkDLFG::enableObject());
    if (supportsMultiFrame) {
      dlfgMfgModeCombo.getKey(&DxvkDLFG::maxInterpolatedFramesObject());
    }

    const auto& reason = ctx->getCommonObjects()->metaNGXContext().getDLFGNotSupportedReason();
    if (reason.size()) {
      RemixGui::SetTooltipToLastWidgetOnHover(reason.c_str());
      ImGui::TextWrapped(reason.c_str());
    }

    if (!supportsDLFG) {
      ImGui::EndDisabled();
    }

    // Keep FG preset recovery available after a feature failure.
    if (RtxOptions::showUI() == UIType::Advanced && ctx->getCommonObjects()->metaNGXContext().supportsDLFG()) {
      dlfgPresetCombo.getKey(&DxvkDLFG::litePresetObject());
    }

    // Need to change Reflex in sync with DLFG, not on the next frame.
    if (dlfgChanged) {
      if (!supportsDLFG) {
        DxvkDLFG::enable.setDeferred(false);
      } else if (!DxvkDLFG::enable()){
        // DLFG was just enabled.  force Reflex to Low Latency.
        RtxOptions::reflexMode.setDeferred(ReflexMode::LowLatency);
      }
    }

  }

  void ImGUI::showReflexOptions(const Rc<DxvkContext>& ctx, bool displayStatsWindowToggle) {
    RtxReflex& reflex = m_device->getCommon()->metaReflex();

    // Note: Skip Reflex ImGUI options if Reflex is not initialized (either fully disabled or failed to be initialized).
    if (!reflex.reflexInitialized()) {
      return;
    }

    // Display Reflex mode selector

    {
      bool disableReflexUI = ctx->isDLFGEnabled();
      ImGui::BeginDisabled(disableReflexUI);
      reflexModeCombo.getKey(&RtxOptions::reflexModeObject());
      ImGui::EndDisabled();
    }

    // Add a button to toggle the Reflex latency stats Window if requested

    if (displayStatsWindowToggle) {
      if (ImGui::Button("Toggle Reflex Stats Window", ImVec2(ImGui::GetContentRegionAvail().x - GImGui->Style.FramePadding.x * 2, 0))) {
        m_reflexLatencyStatsOpen = !m_reflexLatencyStatsOpen;
      }
    }

  }

  void ImGUI::showReflexLatencyStats() {
    // Set up the latency stats Window

    ImGui::SetNextWindowSize(ImVec2(m_reflexLatencyStatsWindowWidth, m_reflexLatencyStatsWindowHeight), ImGuiCond_Once);

    if (!ImGui::Begin("Reflex Latency Stats", &m_reflexLatencyStatsOpen, popupWindowFlags)) {
      ImGui::End();

      return;
    }

    RtxReflex& reflex = m_device->getCommon()->metaReflex();
    const auto latencyStats = reflex.getLatencyStats();

    constexpr ImPlotFlags druationGraphFlags = ImPlotFlags_NoMouseText | ImPlotFlags_NoInputs | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect;
    constexpr ImPlotAxisFlags graphFrameAxisFlags = ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks | ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_Lock;
    constexpr ImPlotAxisFlags graphDurationAxisFlags = ImPlotAxisFlags_Lock;

    constexpr ImPlotFlags timingGraphFlags = ImPlotFlags_NoMouseText | ImPlotFlags_NoInputs | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoLegend;
    constexpr ImPlotAxisFlags graphTimeAxisFlags = ImPlotAxisFlags_Lock;
    constexpr ImPlotAxisFlags graphRegionAxisFlags = ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks;

    // Update Reflex stat ranges

    const auto& interpolationRate = reflexStatRangeInterpolationRate();
    const auto& paddingRatio = reflexStatRangePaddingRatio();

    const auto newCurrentGameToRenderDurationMin = std::max(latencyStats.gameToRenderDurationMin - latencyStats.gameToRenderDurationMin * paddingRatio, 0.0f);
    const auto newCurrentGameToRenderDurationMax = latencyStats.gameToRenderDurationMax + latencyStats.gameToRenderDurationMax * paddingRatio;
    const auto newCurrentCombinedDurationMin = std::max(latencyStats.combinedDurationMin - latencyStats.combinedDurationMin * paddingRatio, 0.0f);
    const auto newCurrentcombinedDurationMax = latencyStats.combinedDurationMax + latencyStats.combinedDurationMax * paddingRatio;

    // Note: Check if the various range members have been initialized yet to allow the first frame to set them rather than interpolate (since they are
    // left as undefined values right now, and even if they were set to 0 or some other value it might give slightly jarring initial behavior).
    if (m_reflexRangesInitialized) {
      // Note: Exponential-esque moving averages.
      m_currentGameToRenderDurationMin = lerp(m_currentGameToRenderDurationMin, newCurrentGameToRenderDurationMin, interpolationRate);
      m_currentGameToRenderDurationMax = lerp(m_currentGameToRenderDurationMax, newCurrentGameToRenderDurationMax, interpolationRate);
      m_currentCombinedDurationMin = lerp(m_currentCombinedDurationMin, newCurrentCombinedDurationMin, interpolationRate);
      m_currentCombinedDurationMax = lerp(m_currentCombinedDurationMax, newCurrentcombinedDurationMax, interpolationRate);
    } else {
      m_currentGameToRenderDurationMin = newCurrentGameToRenderDurationMin;
      m_currentGameToRenderDurationMax = newCurrentGameToRenderDurationMax;
      m_currentCombinedDurationMin = newCurrentCombinedDurationMin;
      m_currentCombinedDurationMax = newCurrentcombinedDurationMax;

      m_reflexRangesInitialized = true;
    }

    // Draw Total Duration Plot

    if (ImPlot::BeginPlot("Total Duration", ImVec2(-1, 200), druationGraphFlags)) {
      ImPlot::SetupAxes("Frame", "Duration (ms)", graphFrameAxisFlags, graphDurationAxisFlags);
      ImPlot::SetupAxisLimits(ImAxis_X1, static_cast<double>(latencyStats.frameIDMin), static_cast<double>(latencyStats.frameIDMax), ImGuiCond_Always);
      ImPlot::SetupAxisLimits(ImAxis_Y1, static_cast<double>(m_currentGameToRenderDurationMin), static_cast<double>(m_currentGameToRenderDurationMax), ImPlotCond_Always);

      ImPlot::PlotLine("Game to Render", latencyStats.frameID, latencyStats.gameToRenderDuration, LatencyStats::statFrames, 0, 0);

      ImPlot::EndPlot();
    }

    ImGui::Text("Game to Render Duration: %.2f ms", latencyStats.gameToRenderDuration[LatencyStats::statFrames - 1]);
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    RemixGui::SetTooltipToLastWidgetOnHover("This measures the time from the start of the simulation to the end of the GPU rendering as a total game to render latency.");

    RemixGui::Separator();

    // Draw Region Duration Plot

    if (ImPlot::BeginPlot("Region Durations", ImVec2(-1, 250), druationGraphFlags)) {
      ImPlot::SetupAxes("Frame", "Duration (ms)", graphFrameAxisFlags, graphDurationAxisFlags);
      ImPlot::SetupAxisLimits(ImAxis_X1, static_cast<double>(latencyStats.frameIDMin), static_cast<double>(latencyStats.frameIDMax), ImGuiCond_Always);
      ImPlot::SetupAxisLimits(ImAxis_Y1, static_cast<double>(m_currentCombinedDurationMin), static_cast<double>(m_currentCombinedDurationMax), ImPlotCond_Always);

      ImPlot::PlotLine("Simulation", latencyStats.frameID, latencyStats.simDuration, LatencyStats::statFrames, 0, 0);
      ImPlot::PlotLine("Render Submit", latencyStats.frameID, latencyStats.renderSubmitDuration, LatencyStats::statFrames, 0, 0);
      ImPlot::PlotLine("Present", latencyStats.frameID, latencyStats.presentDuration, LatencyStats::statFrames, 0, 0);
      ImPlot::PlotLine("Driver", latencyStats.frameID, latencyStats.driverDuration, LatencyStats::statFrames, 0, 0);
      ImPlot::PlotLine("OS Queue", latencyStats.frameID, latencyStats.osRenderQueueDuration, LatencyStats::statFrames, 0, 0);
      ImPlot::PlotLine("GPU Render", latencyStats.frameID, latencyStats.gpuRenderDuration, LatencyStats::statFrames, 0, 0);

      ImPlot::EndPlot();
    }

    ImGui::Text("Simulation Duration: %.2f ms", latencyStats.simDuration[LatencyStats::statFrames - 1]);
    ImGui::Text("Render Submit Duration: %.2f ms", latencyStats.renderSubmitDuration[LatencyStats::statFrames - 1]);
    ImGui::Text("Present Duration: %.2f ms", latencyStats.presentDuration[LatencyStats::statFrames - 1]);
    ImGui::Text("Driver Duration: %.2f ms", latencyStats.driverDuration[LatencyStats::statFrames - 1]);
    ImGui::Text("OS Queue Duration: %.2f ms", latencyStats.osRenderQueueDuration[LatencyStats::statFrames - 1]);
    ImGui::Text("GPU Render Duration: %.2f ms", latencyStats.gpuRenderDuration[LatencyStats::statFrames - 1]);

    RemixGui::Separator();

    // Draw Region Timing Plot

    constexpr float microsecondsPerMillisecond { 1000.0f };

    if (ImPlot::BeginPlot("Region Timings", ImVec2(-1, 150), timingGraphFlags)) {
      ImPlot::SetupAxes(nullptr, nullptr, graphTimeAxisFlags, graphRegionAxisFlags);
      ImPlot::SetupAxisLimits(ImAxis_X1, 0.0f, static_cast<double>(static_cast<float>(latencyStats.combinedCurrentTimeMax - latencyStats.combinedCurrentTimeMin) / microsecondsPerMillisecond), ImPlotCond_Always);

      constexpr const char* regions[]{ "Simulation", "Render Submit", "Present", "Driver", "OS Queue", "GPU Render" };
      ImPlot::SetupAxisTicks(ImAxis_Y1, 0, 5, 6, regions, false);

      // Note: Name needed to color label.
      constexpr const char* labels[]{ "", "Time", "" };
      const float timeData[]{
        // Pre-Span
        static_cast<float>(latencyStats.simCurrentStartTime - latencyStats.combinedCurrentTimeMin) / microsecondsPerMillisecond,
        static_cast<float>(latencyStats.renderSubmitCurrentStartTime - latencyStats.combinedCurrentTimeMin) / microsecondsPerMillisecond,
        static_cast<float>(latencyStats.presentCurrentStartTime - latencyStats.combinedCurrentTimeMin) / microsecondsPerMillisecond,
        static_cast<float>(latencyStats.driverCurrentStartTime - latencyStats.combinedCurrentTimeMin) / microsecondsPerMillisecond,
        static_cast<float>(latencyStats.osRenderQueueCurrentStartTime - latencyStats.combinedCurrentTimeMin) / microsecondsPerMillisecond,
        static_cast<float>(latencyStats.gpuRenderCurrentStartTime - latencyStats.combinedCurrentTimeMin) / microsecondsPerMillisecond,
        // Current Span
        latencyStats.simDuration[LatencyStats::statFrames - 1],
        latencyStats.renderSubmitDuration[LatencyStats::statFrames - 1],
        latencyStats.presentDuration[LatencyStats::statFrames - 1],
        latencyStats.driverDuration[LatencyStats::statFrames - 1],
        latencyStats.osRenderQueueDuration[LatencyStats::statFrames - 1],
        latencyStats.gpuRenderDuration[LatencyStats::statFrames - 1],
        // Post-Span
        static_cast<float>(latencyStats.combinedCurrentTimeMax - latencyStats.simCurrentEndTime) / microsecondsPerMillisecond,
        static_cast<float>(latencyStats.combinedCurrentTimeMax - latencyStats.renderSubmitCurrentEndTime) / microsecondsPerMillisecond,
        static_cast<float>(latencyStats.combinedCurrentTimeMax - latencyStats.presentCurrentEndTime) / microsecondsPerMillisecond,
        static_cast<float>(latencyStats.combinedCurrentTimeMax - latencyStats.driverCurrentEndTime) / microsecondsPerMillisecond,
        static_cast<float>(latencyStats.combinedCurrentTimeMax - latencyStats.osRenderQueueCurrentEndTime) / microsecondsPerMillisecond,
        static_cast<float>(latencyStats.combinedCurrentTimeMax - latencyStats.gpuRenderCurrentEndTime) / microsecondsPerMillisecond,
      };

      ImPlot::PlotBarGroups(labels, timeData, 3, 6, 0.75, 0, ImPlotBarGroupsFlags_Stacked | ImPlotBarGroupsFlags_Horizontal);

      ImPlot::EndPlot();
    }

    ImGui::End();
  }

  namespace {
    struct GpuOverrideChoice {
      const char* label;
      const char* value;  // nullptr: no override
    };

    constexpr GpuOverrideChoice kGraphicsPresetChoices[] = {
      { "-", nullptr }, { "Ultra", "Ultra" }, { "High", "High" }, { "Medium", "Medium" }, { "Low", "Low" },
    };
    constexpr GpuOverrideChoice kDlssModeChoices[] = {
      { "-", nullptr }, { "Ultra Performance", "UltraPerformance" }, { "Performance", "Performance" },
      { "Balanced", "Balanced" }, { "Quality", "Quality" }, { "Full Resolution", "FullResolution" },
    };
    struct GpuOverrideColumn {
      const char* header;
      const char* prefix;
      const GpuOverrideChoice* choices;
      size_t choiceCount;
    };

    const GpuOverrideColumn kGpuOverrideColumns[] = {
      { "Graphics Preset", GpuOverrides::kGraphicsPresetPrefix, kGraphicsPresetChoices, std::size(kGraphicsPresetChoices) },
      { "DLSS Mode", GpuOverrides::kDlssModePrefix, kDlssModeChoices, std::size(kDlssModeChoices) },
    };

    bool equalsIgnoreCase(const std::string& a, const char* b) {
      return a.size() == strlen(b) &&
             std::equal(a.begin(), a.end(), b, [](char x, char y) {
               return std::tolower(static_cast<unsigned char>(x)) == std::tolower(static_cast<unsigned char>(y));
             });
    }

    // Edits apply at the end of the frame; show them until the resolved value catches up or they expire.
    struct PendingGpuOverrideEdit {
      std::optional<std::string> value;
      int frame;
    };
    constexpr int kPendingGpuOverrideFrames = 10;
  }

  void ImGUI::showGpuOverrides(const Rc<DxvkContext>& ctx) {
    if (!RemixGui::CollapsingHeader("Per-GPU Overrides", collapsingHeaderClosedFlags)) {
      return;
    }

    static std::set<std::string> s_draftGpuIds;
    static std::map<std::string, PendingGpuOverrideEdit> s_pendingEdits;
    static uint64_t s_rtxConfGeneration = 0;

    const RtxOptionLayer* rtxConfLayer = RtxOptionLayer::getRtxConfLayer();
    // Edits queued before a reload or clear of rtx.conf are dropped, so stop showing them.
    const uint64_t rtxConfGeneration = rtxConfLayer ? rtxConfLayer->getGeneration() : 0;
    if (rtxConfGeneration != s_rtxConfGeneration) {
      s_pendingEdits.clear();
      s_rtxConfGeneration = rtxConfGeneration;
    }

    ImGui::Indent();

    const VkPhysicalDeviceProperties& deviceProperties = ctx->getDevice()->adapter()->devicePropertiesExt().core.properties;
    const std::string currentGpuId = GpuOverrides::formatGpuId(deviceProperties.vendorID, deviceProperties.deviceID);

    ImGui::TextWrapped(
      "Per-GPU defaults saved to the Remix Config layer (rtx.conf). Changes apply the next time the graphics preset is applied, "
      "such as at startup, when the preset changes, or when deleting user settings resets the preset to Auto.");
    ImGui::Text("Current GPU: %s (%s)", deviceProperties.deviceName, currentGpuId.c_str());

    // GPU ID -> whether each column has a value in some layer, including layers below their blend threshold.
    std::map<std::string, std::array<bool, std::size(kGpuOverrideColumns)>> rows;
    // Key -> config text of rtx.conf entries that failed to parse; they have no value but can be replaced or removed here.
    std::map<std::string, std::string> rejectedValues;
    for (size_t column = 0; column < std::size(kGpuOverrideColumns); ++column) {
      const std::string prefix = kGpuOverrideColumns[column].prefix;
      for (const DynamicOptionEntry& entry : RtxOptionManager::enumerateDynamicOptions(prefix, true)) {
        rows[entry.suffix][column] = true;
      }
      if (rtxConfLayer) {
        for (auto& [key, text] : rtxConfLayer->getRejectedEntries(prefix)) {
          rows[key.substr(prefix.size())];
          rejectedValues[key] = std::move(text);
        }
      }
    }
    for (auto it = s_draftGpuIds.begin(); it != s_draftGpuIds.end();) {
      if (rows.count(*it)) {
        it = s_draftGpuIds.erase(it);
      } else {
        rows[*it];
        ++it;
      }
    }

    ImGui::BeginDisabled(rows.count(currentGpuId) > 0);
    if (ImGui::Button("Add Current GPU")) {
      s_draftGpuIds.insert(currentGpuId);
    }
    ImGui::EndDisabled();

    const int frame = ImGui::GetFrameCount();
    auto requestEdit = [&](const std::string& key, const char* value) {
      RtxOptionLayerTarget target(RtxOptionEditTarget::User);
      const DynamicOptionResult result = value ? RtxOptionManager::setDynamicValue(key, value) : RtxOptionManager::clearDynamicValue(key);
      if (result == DynamicOptionResult::Success) {
        s_pendingEdits[key] = { value ? std::optional<std::string>(value) : std::nullopt, frame };
      }
    };

    if (rows.empty()) {
      ImGui::TextDisabled("No per-GPU overrides configured.");
    } else if (ImGui::BeginTable("GpuOverrides", static_cast<int>(std::size(kGpuOverrideColumns) + 2), ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
      ImGui::TableSetupColumn("GPU");
      for (const GpuOverrideColumn& column : kGpuOverrideColumns) {
        ImGui::TableSetupColumn(column.header);
      }
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed);
      ImGui::TableHeadersRow();

      for (const auto& [gpuId, hasValue] : rows) {
        ImGui::PushID(gpuId.c_str());
        ImGui::TableNextRow();

        ImGui::TableSetColumnIndex(0);
        if (GpuOverrides::isCanonicalGpuId(gpuId)) {
          ImGui::TextUnformatted(gpuId.c_str());
        } else {
          ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", gpuId.c_str());
          RemixGui::SetTooltipToLastWidgetOnHover("Not a GPU ID in the form VVVV_DDDD (uppercase hexadecimal); this entry never matches.");
        }
        if (gpuId == currentGpuId) {
          ImGui::SameLine();
          ImGui::TextDisabled("(current)");
        }

        for (size_t column = 0; column < std::size(kGpuOverrideColumns); ++column) {
          const GpuOverrideColumn& columnInfo = kGpuOverrideColumns[column];
          const std::string key = columnInfo.prefix + gpuId;
          std::optional<std::string> value = GpuOverrides::findActiveValue(key);
          bool inactive = hasValue[column] && !value;
          const auto rejected = rejectedValues.find(key);
          bool invalid = rejected != rejectedValues.end();

          auto pending = s_pendingEdits.find(key);
          if (pending != s_pendingEdits.end()) {
            const bool caughtUp = !invalid && pending->second.value.has_value() == value.has_value() &&
                                  (!value || equalsIgnoreCase(*value, pending->second.value->c_str()));
            if (caughtUp || frame - pending->second.frame > kPendingGpuOverrideFrames) {
              s_pendingEdits.erase(pending);
            } else {
              value = pending->second.value;
              inactive = false;
              invalid = false;
            }
          }

          int selected = value ? -1 : 0;
          for (size_t i = 1; value && i < columnInfo.choiceCount; ++i) {
            if (equalsIgnoreCase(*value, columnInfo.choices[i].value)) {
              selected = static_cast<int>(i);
            }
          }

          ImGui::TableSetColumnIndex(static_cast<int>(column + 1));
          ImGui::SetNextItemWidth(-FLT_MIN);
          const bool unrecognized = selected < 0;
          const bool flagged = invalid || inactive || unrecognized;
          if (flagged) {
            const bool disabledStyle = inactive && !invalid;
            ImGui::PushStyleColor(ImGuiCol_Text, disabledStyle ? ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled) : ImVec4(1.0f, 0.35f, 0.35f, 1.0f));
          }
          const std::string comboId = std::string("##") + columnInfo.header;
          const char* preview = invalid ? rejected->second.c_str()
                              : inactive ? "inactive"
                              : unrecognized ? value->c_str()
                              : columnInfo.choices[selected].label;
          const bool open = ImGui::BeginCombo(comboId.c_str(), preview);
          if (flagged) {
            ImGui::PopStyleColor();
            if (invalid) {
              const std::string tooltip = value
                ? str::format("Invalid value in rtx.conf; it is ignored. In effect from another layer: ", *value, ".")
                : std::string("Invalid value in rtx.conf; it is ignored and no override is in effect.");
              RemixGui::SetTooltipToLastWidgetOnHover(tooltip.c_str());
            } else {
              RemixGui::SetTooltipToLastWidgetOnHover(inactive ? "Only set in layers below their blend threshold; no override is in effect."
                                                               : "Unrecognized value; this override is ignored.");
            }
          }
          if (open) {
            for (size_t i = 0; i < columnInfo.choiceCount; ++i) {
              if (ImGui::Selectable(columnInfo.choices[i].label, static_cast<int>(i) == selected)) {
                requestEdit(key, columnInfo.choices[i].value);
              }
            }
            ImGui::EndCombo();
          }

          // Edits go to rtx.conf; a stronger layer such as user.conf keeps its value.
          if (const RtxOptionImpl* option = RtxOptionImpl::getOptionByFullName(key)) {
            if (const RtxOptionLayer* blockingLayer = option->getBlockingLayer(RtxOptionLayer::getRtxConfLayer())) {
              ImGui::TextDisabled("Set in %s", blockingLayer->getName().c_str());
            }
          }
        }

        ImGui::TableSetColumnIndex(static_cast<int>(std::size(kGpuOverrideColumns) + 1));
        if (ImGui::SmallButton("Remove")) {
          for (const GpuOverrideColumn& column : kGpuOverrideColumns) {
            requestEdit(column.prefix + gpuId, nullptr);
          }
          s_draftGpuIds.erase(gpuId);
        }

        ImGui::PopID();
      }

      ImGui::EndTable();
    }

    ImGui::Unindent();
  }

  void ImGUI::showRenderingSettings(const Rc<DxvkContext>& ctx) {
    ImGui::PushItemWidth(largeUiMode() ? m_largeWindowWidgetWidth : m_regularWindowWidgetWidth);
    auto common = ctx->getCommonObjects();

    ImGui::Text("Disclaimer: The following settings are intended for developers,\nchanging them may introduce instability.");
    RemixGui::Separator();

    // Always display memory stats to user.
    showMemoryStats();

    RemixGui::Separator();

    if (RemixGui::CollapsingHeader("General", collapsingHeaderFlags)) {
      auto& dlss = common->metaDLSS();
      ImGui::Indent();

      common->metaNgxPassthrough().showImguiSettings();

      RemixGui::Separator();

      showDLFGOptions(ctx);

      RemixGui::Separator();

      showReflexOptions(ctx, true);

      RemixGui::Separator();

      getUpscalerCombo(dlss).getKey(&RtxOptions::upscalerTypeObject());

      RtxOptions::updatePresetFromUpscaler();

      if (RtxOptions::upscalerType() == UpscalerType::DLSS && !ctx->getCommonObjects()->metaDLSS().supportsDLSS()) {
        RtxOptions::upscalerType.setDeferred(UpscalerType::TAAU);
      }

      if (RtxOptions::upscalerType() == UpscalerType::DLSS) {
        dlssRenderPresetCombo.getKey(&DxvkDLSS::presetObject());
        dlssProfileCombo.getKey(&RtxOptions::qualityDLSSObject());
      } else if (RtxOptions::upscalerType() == UpscalerType::NIS) {
        RemixGui::SliderFloat("Resolution scale", &RtxOptions::resolutionScaleObject(), 0.5f, 1.0f);
        RemixGui::SliderFloat("Sharpness", &ctx->getCommonObjects()->metaNIS().m_sharpness, 0.1f, 1.0f);
        RemixGui::Checkbox("Use FP16", &ctx->getCommonObjects()->metaNIS().m_useFp16);
      } else if (RtxOptions::upscalerType() == UpscalerType::XeSS) {
        xessPresetCombo.getKey(&DxvkXeSS::XessOptions::presetObject());

        if (DxvkXeSS::XessOptions::preset() == XeSSPreset::Custom) {
          RemixGui::SliderFloat("Resolution Scale", &RtxOptions::resolutionScaleObject(), 0.1f, 1.0f, "%.2f");
        }

        uint32_t inputWidth;
        uint32_t inputHeight;
        ctx->getCommonObjects()->metaNgxPassthrough().getXeSSInputResolution(
          uint32_t(ImGui::GetIO().DisplaySize.x),
          uint32_t(ImGui::GetIO().DisplaySize.y),
          inputWidth, inputHeight);
        ImGui::TextWrapped(str::format("Render Resolution: ", inputWidth, "x", inputHeight).c_str());
      } else if (RtxOptions::upscalerType() == UpscalerType::TAAU) {
        RemixGui::SliderFloat("Resolution scale", &RtxOptions::resolutionScaleObject(), 0.5f, 1.0f);
      }

      auto& dlssNeuralRendering = common->metaDlssNeuralRendering();
      if (dlssNeuralRendering.supportsDlssNeuralRendering()) {
        RemixGui::Separator();

        if (RemixGui::CollapsingHeader("DLSS 3D-Guided Neural Rendering", collapsingHeaderClosedFlags)) {
          ImGui::Indent();
          ImGui::PushID("DLSS 3D-Guided Neural Rendering");
          dlssNeuralRendering.showDlssNeuralRenderingImguiSettings();
          ImGui::PopID();
          ImGui::Unindent();
        }
      }

      RemixGui::Separator();

      RemixGui::Checkbox("Allow Full Screen Exclusive?", &RtxOptions::allowFSEObject());

      showGpuOverrides(ctx);

      ImGui::Unindent();
    }


    if (RemixGui::CollapsingHeader("Post-Processing", collapsingHeaderClosedFlags)) {
      ImGui::Indent();


      if (RtxOptions::upscalerType() == UpscalerType::TAAU) {
        if (RemixGui::CollapsingHeader("TAA-U", collapsingHeaderClosedFlags))
          common->metaTAA().showImguiSettings();
      }

      if (RemixGui::CollapsingHeader("Bloom", collapsingHeaderClosedFlags))
        common->metaBloom().showImguiSettings();

      if (RemixGui::CollapsingHeader("Post FX", collapsingHeaderClosedFlags))
        common->metaPostFx().showImguiSettings();

      ImGui::Unindent();
    }


    ImGui::PopItemWidth();
  }

  void ImGUI::render(const Rc<DxvkContext>& ctx, VkExtent2D surfaceSize) {
    ScopedGpuProfileZone(ctx, "ImGUI Render");

    const HWND gameHwnd = ctx->getCommonObjects()->getLastKnownWindowHandle();
    
    // We need a window to render the GUI and for input to work correctly
    if (gameHwnd == 0) {
      return;
    }

    if (m_overlayWin.ptr() != nullptr) {
      m_overlayWin->update(gameHwnd);
    }

    ImGui::SetCurrentContext(m_context);
    ImPlot::SetCurrentContext(m_plotContext);

    {
      // Pairs with withInputLock(): covers the Win32 backend data and the input queue ImGui::NewFrame() drains.
      std::lock_guard<dxvk::recursive_mutex> inputLock(m_inputMutex);

      // Sometimes games can change windows on us, so we need to check that here and tell ImGUI
      if (m_gameHwnd != gameHwnd) {
        m_gameHwnd = gameHwnd;

        if (m_init) {
          ImGui_ImplWin32_Shutdown();
        }

        ImGui_ImplWin32_Init(gameHwnd);
      }

      if (!m_init) {
        ImGui_ImplDxvk::Init(m_device);

        //execute a gpu command to upload imgui font textures
        createFontsTexture(ctx);

        m_init = true;
      }

      ImGui_ImplDxvk::NewFrame();
      ImGui_ImplWin32_NewFrame();

      ImGuiIO& io = ImGui::GetIO();
      io.DisplaySize = ImVec2((float) surfaceSize.width, (float) surfaceSize.height);

      ImGui::NewFrame();
    }

    update(ctx);

    ImGui_ImplDxvk::RenderDrawData(ImGui::GetDrawData(), ctx.ptr(), surfaceSize.width, surfaceSize.height);
  }

  void ImGUI::createFontsTexture(const Rc<DxvkContext>& ctx) {
    ImGuiIO& io = ImGui::GetIO();
    ImGui_ImplDxvk::Data* bd = (ImGui_ImplDxvk::Data*)io.BackendRendererUserData;
    
    // Range of characters we want to use the primary font
    ImVector<ImWchar> characterRange;
    {
      ImFontGlyphRangesBuilder builder;
      builder.AddText("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$ % &\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c");
      builder.BuildRanges(&characterRange);
    }

    // Range of characters we want to use the second (monospaced) font for
    ImVector<ImWchar> numericalRange;
    {
      ImFontGlyphRangesBuilder builder;
      builder.AddText("0123456789");
      builder.BuildRanges(&numericalRange);
    }

    // Build a second font, where all characters are consistent.  This will be used for non-field/title text
    ImVector<ImWchar> allRange;
    {
      ImFontGlyphRangesBuilder builder;
      builder.AddText("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$ % &\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c");
      builder.BuildRanges(&allRange);
    }

    // Normal Size Font (Default)

    ImFontConfig normalFontCfg = ImFontConfig();
    normalFontCfg.SizePixels = 16.f;
    normalFontCfg.FontDataOwnedByAtlas = false;

    const size_t nvidiaSansLength = sizeof(___NVIDIASansMd) / sizeof(___NVIDIASansMd[0]);
    const size_t nvidiaSansBdLength = sizeof(___NVIDIASansBd) / sizeof(___NVIDIASansBd[0]);
    const size_t robotoMonoLength = sizeof(___RobotoMonoRg) / sizeof(___RobotoMonoRg[0]);

    {
      // Add letters/symbols (NVIDIA-Sans)
      m_regularFont = io.Fonts->AddFontFromMemoryTTF(&___NVIDIASansMd[0], nvidiaSansLength, 0, &normalFontCfg, characterRange.Data);

      // Enable merging
      normalFontCfg.MergeMode = true;

      // Add numbers (Roboto-Mono)
      io.Fonts->AddFontFromMemoryTTF(&___RobotoMonoRg[0], robotoMonoLength, 0, &normalFontCfg, numericalRange.Data);

      normalFontCfg.MergeMode = false;
      m_boldFont = io.Fonts->AddFontFromMemoryTTF(&___NVIDIASansBd[0], nvidiaSansBdLength, 0, &normalFontCfg, allRange.Data);
    }

    // Large Size Font

    ImFontConfig largeFontCfg = ImFontConfig();
    largeFontCfg.SizePixels = 24.f;
    largeFontCfg.FontDataOwnedByAtlas = false;

    {
      // Add letters/symbols (NVIDIA-Sans)
      m_largeFont = io.Fonts->AddFontFromMemoryTTF(&___NVIDIASansBd[0], nvidiaSansLength, 0, &largeFontCfg, characterRange.Data);

      // Enable merging
      largeFontCfg.MergeMode = true;

      // Add numbers (Roboto-Mono)
      io.Fonts->AddFontFromMemoryTTF(&___RobotoMonoRg[0], robotoMonoLength, 0, &largeFontCfg, numericalRange.Data);
    }

    // Build the fonts

    io.Fonts->Build();

    // Apply the correct default font based on largeUiMode setting
    io.FontDefault = largeUiMode() ? m_largeFont : m_regularFont;


    // Allocate/upload glyph cache...

    unsigned char* pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    size_t row_pitch = (size_t)width * 4 * sizeof(char);
    size_t upload_size = height * row_pitch;

    VkResult err;

    // Create the Image:
    {
      DxvkImageCreateInfo info = {};
      info.type = VK_IMAGE_TYPE_2D;
      info.format = VK_FORMAT_R8G8B8A8_UNORM;
      info.extent.width = width;
      info.extent.height = height;
      info.extent.depth = 1;
      info.mipLevels = 1;
      info.numLayers = 1;
      info.sampleCount = VK_SAMPLE_COUNT_1_BIT;
      info.tiling = VK_IMAGE_TILING_OPTIMAL;
      info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      info.layout = VK_IMAGE_LAYOUT_GENERAL;
      info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      m_fontTexture = m_device->createImage(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::AppTexture, "imgui font texture");
      bd->FontImage = m_fontTexture->handle();
    }

    // Create the Image View:
    {
      DxvkImageViewCreateInfo info = {};
      info.type = VK_IMAGE_VIEW_TYPE_2D;
      info.format = VK_FORMAT_R8G8B8A8_UNORM;
      info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
      info.numLevels = 1;
      info.numLayers = 1;
      info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      m_fontTextureView = m_device->createImageView(m_fontTexture, info);
      bd->FontView = m_fontTextureView->handle();
    }

    ctx->updateImage(m_fontTexture,
      VkImageSubresourceLayers{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      VkOffset3D{ 0, 0, 0 },
      m_fontTexture->mipLevelExtent(0),
      pixels, row_pitch, upload_size);

    Rc<DxvkSampler> sampler = m_device->getCommon()->getResources().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    // Store our identifier
    io.Fonts->SetTexID(ImGui_ImplDxvk::AddTexture(sampler, m_fontTextureView));
  }

  bool ImGUI::checkHotkeyState(const VirtualKeys& virtKeys, const bool allowContinuousPress) {
    bool result = false;
    if(virtKeys.size() > 0) {
      auto& io = ImGui::GetIO();
      result = true;
      for(const auto& vk : virtKeys) {
        if(vk.val == VK_SHIFT) {
          result = result && io.KeyShift;
        } else if(vk.val == VK_CONTROL) {
          result = result && io.KeyCtrl;
        } else if(vk.val == VK_MENU) {
          result = result && io.KeyAlt;
        } else {
          ImGuiKey key = ImGui::GetKeyIndex(ImGui_ImplWin32_VirtualKeyToImGuiKey(vk.val));
          if (allowContinuousPress) {
            result = result && ImGui::IsKeyDown(key);
          } else {
            result = result && ImGui::IsKeyPressed(key, false);
          }
        }
      }
    }
    return result;
  }

  void ImGUI::onCloseMenus() {
    // When closing the menus, try and free up some extra memory, just in case
    //  the user has toggled a bunch of systems while in menus causing an artificial
    //  inflation.
    freeUnusedMemory();

    ::ShowCursor(m_prevCursorVisible);
    if (RtxOptions::restoreCursorPosition()) {
      ::SetPhysicalCursorPos(m_cachedGameCursorX, m_cachedGameCursorY);
    }
  }

  void ImGUI::onOpenMenus() {
    // Before opening the menus, try free some memory, the idea being the 
    //  user may want to make some changes to various settings and so they
    //  should have all available memory to do so.
    freeUnusedMemory();

    CURSORINFO info;
    GetCursorInfo(&info);
    m_prevCursorVisible = info.flags == CURSOR_SHOWING;

    // Use physical cursor position to avoid DPI scaling issues
    POINT pt;
    ::GetPhysicalCursorPos(&pt);
    m_cachedGameCursorX = pt.x;
    m_cachedGameCursorY = pt.y;
  }

  void ImGUI::freeUnusedMemory() {
    if (!m_device) {
      return;
    }

    m_device->getCommon()->memoryManager().freeUnusedChunks();
  }

}
