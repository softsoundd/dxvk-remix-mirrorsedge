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

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "../../../src/d3d9/d3d9_rtx_ue3_view_layout.h"

namespace dxvk {

  namespace {

    void require(bool condition, const std::string& message) {
      if (!condition) {
        throw std::runtime_error(message);
      }
    }

    void testMainViewSizedViewports() {
      require(ue3ViewportIsMainViewSized(2560, 1080, 2560, 1080), "A full-screen view must be main-view sized.");
      require(ue3ViewportIsMainViewSized(1920, 1080, 2560, 1080), "A 16:9 view pillarboxed in a 21:9 image must be main-view sized.");
      require(ue3ViewportIsMainViewSized(1920, 1080, 1920, 1200), "A 16:9 view letterboxed in a 16:10 image must be main-view sized.");
      require(ue3ViewportIsMainViewSized(1920, 810, 2560, 1080), "A view at 75% screen percentage must be main-view sized.");
      require(ue3ViewportIsMainViewSized(1280, 1080, 2560, 1080), "A split-screen half must be main-view sized.");
      require(!ue3ViewportIsMainViewSized(1024, 1024, 1920, 1080), "A square scene capture must not be main-view sized.");
      require(!ue3ViewportIsMainViewSized(1440, 900, 1920, 1080), "A smaller view of another aspect ratio must not be main-view sized.");
      require(!ue3ViewportIsMainViewSized(640, 360, 1920, 1080), "A view below half the image must not be main-view sized.");
      require(!ue3ViewportIsMainViewSized(3840, 2160, 3840, 1600), "A view larger than the image must not be main-view sized.");
      require(!ue3ViewportIsMainViewSized(1920, 1080, 0, 0), "An empty image must have no main view.");
    }

    void testViewportsFillingAnAxis() {
      require(ue3ViewportFillsBackbufferAxis(1920, 1080, 2560, 1080), "A pillarboxed view must fill the image's height.");
      require(ue3ViewportFillsBackbufferAxis(2560, 1080, 2560, 1440), "A letterboxed view must fill the image's width.");
      require(!ue3ViewportFillsBackbufferAxis(2048, 1080, 1920, 1080), "A view wider than the image must not fill it.");
      require(!ue3ViewportFillsBackbufferAxis(1024, 512, 1920, 1080), "A view smaller on both axes must not fill either.");
      require(!ue3ViewportFillsBackbufferAxis(0, 1080, 1920, 1080), "An empty view must not fill the image.");
    }

    void testCentredViewOffsets() {
      require(ue3CentredViewOffset(2560, 1920) == 320, "A 16:9 view must be centred in a 2560-wide image.");
      require(ue3CentredViewOffset(3840, 2844) == 498, "A 16:9 view must be centred in a 3840x1600 image.");
      require(ue3CentredViewOffset(1080, 1080) == 0, "A view filling the axis must have no offset.");
      require(ue3CentredViewOffset(2561, 1920) == 320, "An odd margin with an even half must round down.");
      require(ue3CentredViewOffset(2563, 1920) == 322, "An odd margin with an odd half must round up.");
      require(ue3CentredViewOffset(1000, 1200) == 0, "A view larger than the image must have no offset.");

      // cvtss2si rounds by the default MXCSR mode, round to nearest with ties to even, as nearbyint does here.
      std::fesetround(FE_TONEAREST);
      for (uint32_t margin = 0; margin <= 8192; ++margin) {
        const int32_t expected = int32_t(std::nearbyint(0.5f * float(margin)));
        require(ue3CentredViewOffset(1000 + margin, 1000) == expected,
                "The offset must match appRound(0.5f * margin) for a margin of " + std::to_string(margin) + ".");
      }
    }

    void testProportions() {
      require(isInProportion(3840, 2160, 16, 9), "A 3840x2160 image must be 16:9.");
      require(isInProportion(3440, 1935, 16, 9), "A 3440x1935 scene buffer must be 16:9.");
      require(!isInProportion(3440, 1440, 16, 9), "A 3440x1440 image must not be 16:9.");
      require(!isInProportion(1366, 768, 16, 9), "A 1366x768 image must not be exactly 16:9.");
      require(isInProportion(960, 540, 1920, 1080), "A half-resolution target must be in proportion to its scene buffer.");
      require(!isInProportion(0, 0, 16, 9), "An empty size must not be in proportion to anything.");
      require(!isInProportion(16, 9, 0, 0), "Nothing must be in proportion to an empty size.");
    }

  }

}

int main() {
  try {
    dxvk::testMainViewSizedViewports();
    dxvk::testViewportsFillingAnAxis();
    dxvk::testCentredViewOffsets();
    dxvk::testProportions();
  } catch (const std::exception& error) {
    std::cerr << "TEST FAILED: " << error.what() << std::endl;
    return -1;
  }

  std::cout << "All UE3 view layout tests passed." << std::endl;
  return 0;
}
