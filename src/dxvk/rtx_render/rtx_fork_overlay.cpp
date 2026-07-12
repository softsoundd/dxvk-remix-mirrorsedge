// src/dxvk/rtx_render/rtx_fork_overlay.cpp
//
// Fork-owned file. Contains the implementations of fork_hooks:: functions
// for the overlay subsystem, lifted from rtx_overlay_window.cpp and
// rtx_context.cpp during the 2026-04-18 fork touchpoint-pattern refactor.
//
// See docs/fork-touchpoints.md for the full fork-hooks catalogue.
//
// NOTE: overlayInputForward accesses GameOverlay::m_hwnd (private).
// dispatchScreenOverlay accesses multiple private members of RtxContext
// (m_pendingScreenOverlay, m_screenOverlay*) and uses ScreenOverlayShader.
// Both classes require the corresponding fork_hooks functions to be declared
// as friends — see rtx_overlay_window.h and rtx_context.h.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>                 // GET_X_LPARAM / GET_Y_LPARAM for mouse coord translation
#include "rtx_fork_hooks.h"

#include "rtx_overlay_window.h"       // GameOverlay
#include "imgui/imgui_impl_win32.h"   // ImGui_ImplWin32_WndProcHandler
#include "imgui/imgui.h"              // ImGui::SetCurrentContext (imguiContextPin)
#include "imgui/implot.h"             // ImPlot::SetCurrentContext (imguiContextPin)
#include "imgui/imgui_remix_exports.h" // remixapi_imgui_InvokeDrawCallback (wrapperTabDraw)

#include "rtx_context.h"              // RtxContext
#include "rtx_resources.h"            // Resources::RaytracingOutput
#include "rtx_shader_manager.h"       // ManagedShader, SHADER_SOURCE, PUSH_CONSTANTS macros
#include "rtx/pass/screen_overlay/screen_overlay.h"
#include <rtx_shaders/screen_overlay.h>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace dxvk {

  namespace {
    // ScreenOverlayShader — the compute shader that alpha-composites the
    // plugin-uploaded RGBA overlay over the final tone-mapped output image.
    // Moved here from the anonymous namespace in rtx_context.cpp so that
    // dispatchScreenOverlay (a fork_hooks function) can reference it without
    // requiring a separate TU.
    class ScreenOverlayShader : public ManagedShader {
      SHADER_SOURCE(ScreenOverlayShader, VK_SHADER_STAGE_COMPUTE_BIT, screen_overlay)

      PUSH_CONSTANTS(ScreenOverlayArgs)

      BEGIN_PARAMETER()
        RW_TEXTURE2D(SCREEN_OVERLAY_INPUT_OUTPUT)
        SAMPLER2D(SCREEN_OVERLAY_TEXTURE)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ScreenOverlayShader);
  } // anonymous namespace

namespace fork_hooks {

  // ---------------------------------------------------------------------------
  // overlayInputForward
  //
  // Forwards keyboard AND mouse WM_* messages from the legacy gameWndProc
  // path to ImGui's Win32 backend. Used when a game menu captures raw input
  // OR when plugin HUD activity pulls focus away from the overlay window,
  // either of which causes overlayWndProc to stop receiving direct messages
  // and the legacy wndProcHandler fallback to be the only delivery path.
  //
  // A previous revision of this function forwarded only keyboard messages,
  // with a comment that mouse was "intentionally not forwarded" because the
  // overlay's WM_INPUT path already synthesized scaled mouse events. That
  // assumption fails in the plugin-API case: when the plugin HUD pulls focus,
  // raw-input delivery to overlayWndProc also stops (the same root cause
  // that broke keyboard), and mouse events disappear. This function now
  // forwards both keyboard and mouse, each gated on raw-input (Path B)
  // recency: Path A only fires when Path B has gone silent, so the two
  // delivery paths never double-feed ImGui (see the in-case comments).
  //
  // Coordinate translation: mouse WM_MOUSEMOVE and WM_{L,R,M,X}BUTTON*
  // messages carry CLIENT-AREA coordinates in lParam. When the overlay is a
  // separate HWND from the game window (the common case), we translate game-
  // client coords to overlay-client coords via ClientToScreen + ScreenToClient
  // so ImGui hit-tests against the overlay's own geometry. Wheel messages
  // (WM_MOUSEWHEEL / WM_MOUSEHWHEEL) carry SCREEN coordinates per Windows
  // convention and are forwarded without translation.
  //
  // ACCESS NOTE: this function reads GameOverlay::m_hwnd (private atomic).
  // A friend declaration for this function is required in GameOverlay.
  // ---------------------------------------------------------------------------
  void overlayInputForward(
      GameOverlay& overlay, HWND gameHwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    HWND overlayHwnd = overlay.m_hwnd.load();
    HWND targetHwnd = overlayHwnd ? overlayHwnd : gameHwnd;

    switch (msg) {
    // Keyboard: gated on Path B liveness, same as mouse below.
    //
    // Path B is NOT mouse-only — the fork's ImGui Win32 backend handles
    // WM_INPUT for RIM_TYPEKEYBOARD too (imgui_impl_win32.cpp), feeding both
    // key events AND text (ToUnicodeEx -> AddInputCharacterUTF16) straight to
    // ImGui. overlayWndProc's WM_INPUT case falls through to that backend for
    // any non-mouse raw packet, so keyboard reaches ImGui via Path B exactly
    // as mouse does. RIDEV_NOLEGACY on the keyboard device only suppresses the
    // overlay's *legacy* WM_KEY* messages; it does nothing about the raw path.
    //
    // So Path A (this block, the game wnd proc fallback) and Path B both
    // deliver every keypress and character — producing doubled keys / doubled
    // text ("dual keyboard input"), the keyboard analogue of the dual-cursor
    // bug fixed in bf6e6851. The bf6e6851 note that "keyboard is Path A only"
    // was wrong: it accounted for the legacy path but not the raw one.
    //
    // The gate is identical to the mouse gate: only forward via Path A when
    // raw input has been silent for longer than kRawInputRecencyMs (~100ms).
    // Normal case: Path B fires concurrently, recency is fresh, Path A skips.
    // Plugin-HUD-pulls-focus case: Path B goes silent, recency lapses, Path A
    // picks up — preserving the legacy keyboard fallback (commit ba358314).
    case WM_KEYDOWN:
    case WM_KEYUP:
    case WM_SYSKEYDOWN:
    case WM_SYSKEYUP:
    case WM_CHAR:
    case WM_SYSCHAR:
      if (overlay.isRawInputRecent()) {
        break;
      }
      // lParam has no coordinates, pass through as-is.
      ImGui_ImplWin32_WndProcHandler(targetHwnd, msg, wParam, lParam);
      break;

    // Mouse motion + buttons + wheels: gated on Path B liveness.
    //
    // Path B (raw input via overlayWndProc::WM_INPUT) is the primary mouse
    // delivery path and uses imgui-display-space coords (scaled by the
    // overlay-client/imgui-DisplaySize ratio). Path A (this block, the game
    // wnd proc fallback) uses unscaled overlay-client coords. When both
    // fire concurrently they fight for io.MousePos and produce a visible
    // dual-cursor jitter (the imgui software cursor alternates between
    // Path A and Path B positions frame-to-frame).
    //
    // The gate: only forward via Path A when raw input has been silent for
    // longer than kRawInputRecencyMs (~100ms). Path B stamps its recency on
    // every WM_INPUT it receives (see rtx_overlay_window.cpp:WM_INPUT). In
    // the normal case Path B fires concurrently with Path A on every real
    // input event, so isRawInputRecent() is always true and Path A skips.
    // In the plugin-HUD-pulls-focus case Path B goes silent, recency lapses,
    // and Path A picks up — preserving the original behavior of commit
    // 3c8fac75 (forward mouse on legacy wnd proc fallback).
    case WM_MOUSEMOVE:
    case WM_LBUTTONDOWN: case WM_LBUTTONUP: case WM_LBUTTONDBLCLK:
    case WM_RBUTTONDOWN: case WM_RBUTTONUP: case WM_RBUTTONDBLCLK:
    case WM_MBUTTONDOWN: case WM_MBUTTONUP: case WM_MBUTTONDBLCLK:
    case WM_XBUTTONDOWN: case WM_XBUTTONUP: case WM_XBUTTONDBLCLK:
    {
      if (overlay.isRawInputRecent()) {
        break;
      }
      LPARAM translated = lParam;
      if (overlayHwnd && overlayHwnd != gameHwnd) {
        POINT pt = { GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam) };
        ClientToScreen(gameHwnd, &pt);
        ScreenToClient(overlayHwnd, &pt);
        translated = MAKELPARAM(pt.x, pt.y);
      }
      ImGui_ImplWin32_WndProcHandler(targetHwnd, msg, wParam, translated);
      break;
    }

    // Wheel: gated together with mouse motion/buttons (same Path A/B race).
    case WM_MOUSEWHEEL:
    case WM_MOUSEHWHEEL:
      if (overlay.isRawInputRecent()) {
        break;
      }
      ImGui_ImplWin32_WndProcHandler(targetHwnd, msg, wParam, lParam);
      break;

    default:
      break;
    }
  }

  // ---------------------------------------------------------------------------
  // dispatchScreenOverlay
  //
  // Alpha-composites a plugin-uploaded RGBA pixel buffer over the final
  // tone-mapped output image using the ScreenOverlayShader compute shader.
  // Called from RtxContext::dispatchScreenOverlay (which is now a one-line
  // delegate) after tone mapping and before screenshot capture.
  //
  // If no overlay has been queued this frame (m_pendingScreenOverlay is
  // empty), returns immediately with no GPU work.
  //
  // The overlay GPU image is recreated whenever dimensions or format change;
  // the staging buffer is copied in via copyBufferToImage each frame.
  //
  // ACCESS NOTE: reads and writes multiple private members of RtxContext:
  // m_pendingScreenOverlay, m_screenOverlayImage, m_screenOverlayView,
  // m_screenOverlayWidth, m_screenOverlayHeight, m_screenOverlayFormat,
  // and m_device. A friend declaration for this function is required in
  // RtxContext (see rtx_context.h).
  // ---------------------------------------------------------------------------
  void dispatchScreenOverlay(RtxContext& ctx, Resources::RaytracingOutput& rtOutput) {
    if (!ctx.m_pendingScreenOverlay.has_value()) {
      return;
    }

    ScopedGpuProfileZone(&ctx, "Screen Overlay");
    auto& overlay = *ctx.m_pendingScreenOverlay;

    // Recreate overlay image if dimensions or format changed.
    if (ctx.m_screenOverlayWidth  != overlay.width
     || ctx.m_screenOverlayHeight != overlay.height
     || ctx.m_screenOverlayFormat != overlay.format
     || !ctx.m_screenOverlayImage.ptr()) {
      DxvkImageCreateInfo imageInfo = {};
      imageInfo.type        = VK_IMAGE_TYPE_2D;
      imageInfo.format      = overlay.format;
      imageInfo.flags       = 0;
      imageInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
      imageInfo.extent      = { overlay.width, overlay.height, 1 };
      imageInfo.numLayers   = 1;
      imageInfo.mipLevels   = 1;
      imageInfo.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      imageInfo.stages      = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      imageInfo.access      = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      imageInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
      imageInfo.layout      = VK_IMAGE_LAYOUT_UNDEFINED;

      ctx.m_screenOverlayImage = ctx.m_device->createImage(
        imageInfo,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        DxvkMemoryStats::Category::RTXRenderTarget,
        "Screen overlay image");

      DxvkImageViewCreateInfo viewInfo = {};
      viewInfo.type      = VK_IMAGE_VIEW_TYPE_2D;
      viewInfo.format    = overlay.format;
      viewInfo.usage     = VK_IMAGE_USAGE_SAMPLED_BIT;
      viewInfo.aspect    = VK_IMAGE_ASPECT_COLOR_BIT;
      viewInfo.minLevel  = 0;
      viewInfo.numLevels = 1;
      viewInfo.minLayer  = 0;
      viewInfo.numLayers = 1;

      ctx.m_screenOverlayView = ctx.m_device->createImageView(ctx.m_screenOverlayImage, viewInfo);

      ctx.m_screenOverlayWidth  = overlay.width;
      ctx.m_screenOverlayHeight = overlay.height;
      ctx.m_screenOverlayFormat = overlay.format;
    }

    // Copy staging buffer to overlay image
    {
      VkImageSubresourceLayers subresource = {};
      subresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      subresource.mipLevel       = 0;
      subresource.baseArrayLayer = 0;
      subresource.layerCount     = 1;

      VkOffset3D offset = { 0, 0, 0 };
      VkExtent3D extent = { overlay.width, overlay.height, 1 };

      ctx.copyBufferToImage(ctx.m_screenOverlayImage, subresource, offset, extent,
                            overlay.stagingBuffer, 0, 0, 0);
    }

    // Dispatch overlay blend compute shader
    ctx.setPushConstantBank(DxvkPushConstantBank::RTX);

    auto& finalOutput         = rtOutput.m_finalOutput.resource(Resources::AccessType::ReadWrite);
    const VkExtent3D outputSize = finalOutput.image->info().extent;
    const VkExtent3D workgroups = util::computeBlockCount(
      outputSize, VkExtent3D { SCREEN_OVERLAY_TILE_SIZE, SCREEN_OVERLAY_TILE_SIZE, 1 });

    ScreenOverlayArgs pushArgs = {};
    pushArgs.imageSize = { outputSize.width, outputSize.height };
    pushArgs.opacity   = overlay.opacity;
    ctx.pushConstants(0, sizeof(pushArgs), &pushArgs);

    ctx.bindResourceView(SCREEN_OVERLAY_INPUT_OUTPUT, finalOutput.view, nullptr);
    ctx.bindResourceView(SCREEN_OVERLAY_TEXTURE, ctx.m_screenOverlayView, nullptr);
    ctx.bindResourceSampler(SCREEN_OVERLAY_TEXTURE,
      ctx.getResourceManager().getSampler(
        VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE));

    ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ScreenOverlayShader::getShader());
    ctx.dispatch(workgroups.width, workgroups.height, workgroups.depth);

    // Clear pending overlay after dispatch
    ctx.m_pendingScreenOverlay.reset();
  }

  // ---------------------------------------------------------------------------
  // imguiContextPin
  //
  // Pins the ImGui and ImPlot thread-local context pointers to the dev menu's
  // private contexts at wndProcHandler entry. Plugin-side rendering on other
  // threads can drift GImGui off the dev menu's context between frames;
  // without this pin, keyboard input is written into the wrong (or null)
  // context and the dev menu stops seeing keypresses (pressing Alt+X produces
  // a Windows "ding" rather than toggling the menu).
  //
  // Matches the pattern used in ImGUI's ctor, dtor, and render().
  //
  // No private-member access — the call site passes m_context and m_plotContext
  // directly. No friend declaration needed.
  // ---------------------------------------------------------------------------
  void imguiContextPin(ImGuiContext* ctx, ImPlotContext* plotCtx) {
    ImGui::SetCurrentContext(ctx);
    ImPlot::SetCurrentContext(plotCtx);
  }

  // ---------------------------------------------------------------------------
  // wrapperTabDraw
  //
  // Invokes the registered plugin ImGui draw callback for the Plugin tab in
  // the dev menu. Called from the kTab_Wrapper switch case in
  // ImGUI::showMainMenu when the Plugin tab is selected.
  //
  // No private-member access — delegates to remixapi_imgui_InvokeDrawCallback.
  // No friend declaration needed.
  // ---------------------------------------------------------------------------
  void wrapperTabDraw() {
    remixapi_imgui_InvokeDrawCallback();
  }

} // namespace fork_hooks
} // namespace dxvk
