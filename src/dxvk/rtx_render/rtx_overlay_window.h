/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

#include <atomic>
#include "../../util/rc/util_rc.h"
#include "../../util/rc/util_rc_ptr.h"
#include "rtx_common_object.h"

namespace dxvk {

// Forward decls so GameOverlay can friend the fork hook that needs access
// to private m_hwnd state. See docs/fork-touchpoints.md.
class GameOverlay;
namespace fork_hooks {
  void overlayInputForward(
    GameOverlay& overlay, HWND gameHwnd, UINT msg, WPARAM wParam, LPARAM lParam);
}

  class GameOverlay : public RcObject {
    // Fork touchpoint: the overlay input-forward hook (keyboard + mouse)
    // needs access to private m_hwnd. Tracked as an inline tweak in
    // docs/fork-touchpoints.md.
    friend void fork_hooks::overlayInputForward(
      GameOverlay&, HWND, UINT, WPARAM, LPARAM);
  public:
    GameOverlay() = delete;

    GameOverlay(const char* className, class ImGUI* pImgui);
    ~GameOverlay();

    HWND hwnd() const {  return m_hwnd; }

    void update(HWND gameHwnd);

    void gameWndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    LRESULT overlayWndProc(HWND, UINT, WPARAM, LPARAM);

    // Recency check for raw-input delivery. Used by overlayInputForward to
    // gate mouse forwarding via the game wnd proc fallback: when Path B
    // (WM_INPUT on the overlay HWND) is actively delivering, Path A must
    // skip mouse so two coord-systems don't fight for io.MousePos. When
    // raw input stops (e.g., plugin HUD pulls focus and our raw-input
    // registration goes silent), Path A picks up after the threshold lapses.
    // Threshold deliberately generous (~100ms) — Path B fires concurrently
    // with Path A on every real input event, so a tight window is fine and
    // a longer one avoids false-positive "Path B dead" on sporadic activity.
    bool isRawInputRecent() const {
      const uint64_t last = m_lastRawInputTickMs.load(std::memory_order_relaxed);
      if (last == 0) {
        return false;
      }
      return (GetTickCount64() - last) < kRawInputRecencyMs;
    }

    void setDebugDraw(bool enable, BYTE alpha = 96) {
      m_debugDraw = enable;
      m_debugAlpha = alpha;
      if (m_hwnd) {
        // Make it visible if debugging; invisible if not.
        SetLayeredWindowAttributes(m_hwnd, 0, m_debugAlpha, ULW_ALPHA);
        InvalidateRect(m_hwnd, nullptr, TRUE);
      }
    }

  private:
    void windowThreadMain();

    void show();
    void hide();

    bool isOurForeground() const;

    HWND m_gameHwnd = nullptr;

    std::atomic<HWND> m_hwnd { 0 };
    std::atomic<bool> m_running { true };
    std::thread m_thread;
    const char* m_className;

    ImGUI* m_pImgui = nullptr;
    UINT m_w = 1, m_h = 1;

    bool  m_mouseInsideOverlay = false;

    bool  m_debugDraw = false;
    BYTE  m_debugAlpha = 96;
    RECT  m_lastRect { 0,0,0,0 };

    // Raw-input liveness gate (see isRawInputRecent() above).
    static constexpr uint64_t kRawInputRecencyMs = 100;
    std::atomic<uint64_t> m_lastRawInputTickMs { 0 };
  };
}