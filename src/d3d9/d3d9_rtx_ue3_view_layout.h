/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <algorithm>
#include <cmath>
#include <cstdint>

// These helpers find where UE3's main view lies in the image the game presents.
// See "Letterboxed views and resolution changes" in documentation/UE3Compatibility.md.
// Their bbW and bbH parameters are the presented region's size, which is the backbuffer's
// unless a windowed device presents only part of it.

namespace dxvk {

  // Returns whether the viewport's aspect ratio is within 5% of the backbuffer's.
  inline bool ue3ViewportAspectMatchesBackbuffer(uint32_t vpW, uint32_t vpH, uint32_t bbW, uint32_t bbH) {
    if (vpW == 0 || vpH == 0 || bbW == 0 || bbH == 0) {
      return false;
    }
    const double a = double(vpW) * double(bbH);
    const double b = double(vpH) * double(bbW);
    const double denom = std::max(a, b);
    return denom > 0.0 && (std::abs(a - b) / denom) < 0.05;
  }

  // Returns whether the viewport spans one backbuffer axis and fits within the other.
  // A view that FViewport::CalculateViewExtents constrains to another aspect ratio has this shape, as does a split-screen half.
  // A lone view renders from the scene buffer's origin rather than centred, so only the size counts.
  inline bool ue3ViewportFillsBackbufferAxis(uint32_t vpW, uint32_t vpH, uint32_t bbW, uint32_t bbH) {
    return vpW != 0 && vpH != 0 &&
           vpW <= bbW && vpH <= bbH &&
           (vpW == bbW || vpH == bbH);
  }

  // Returns whether the viewport is half the backbuffer or more in both dimensions, with its aspect ratio.
  // That allows ScreenPercentage above 50% but keeps square SceneCapture RTs out on widescreen.
  // Returns true as well for a viewport filling one backbuffer axis.
  inline bool ue3ViewportIsMainViewSized(uint32_t vpW, uint32_t vpH, uint32_t bbW, uint32_t bbH) {
    return bbW != 0 && bbH != 0 &&
           ((vpW * 2 >= bbW && vpH * 2 >= bbH &&
             ue3ViewportAspectMatchesBackbuffer(vpW, vpH, bbW, bbH)) ||
            ue3ViewportFillsBackbufferAxis(vpW, vpH, bbW, bbH));
  }

  // Returns the offset FViewport::CalculateViewExtents centres a constrained view at, appRound(0.5f * (size - viewSize)).
  // appRound's cvtss2si rounds a half to even.
  inline int32_t ue3CentredViewOffset(uint32_t size, uint32_t viewSize) {
    const uint32_t margin = size > viewSize ? size - viewSize : 0;
    const uint32_t half = margin / 2;
    return int32_t(half + (margin & half & 1u));
  }

  // Returns whether width by height is in exact proportion to refWidth by refHeight.
  inline bool isInProportion(uint32_t width, uint32_t height, uint32_t refWidth, uint32_t refHeight) {
    return width != 0 && height != 0 && refWidth != 0 && refHeight != 0 &&
           uint64_t(width) * refHeight == uint64_t(height) * refWidth;
  }

}
