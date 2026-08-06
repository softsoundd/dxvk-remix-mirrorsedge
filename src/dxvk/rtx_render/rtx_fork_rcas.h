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
#pragma once

#include "rtx_resources.h"
#include "rtx_options.h"

namespace dxvk {
  class DxvkDevice;
  class DxvkSampler;

  class DxvkRCAS {
  public:
    struct Options {
      friend class DxvkRCAS;
      friend class ImGUI;

      // Note: this lives in its own `rtx.sharpening` namespace rather than under
      // `rtx.fsr`. It drives the standalone RCAS pass that runs after DLSS,
      // DLSS-RR, XeSS and TAA-U, and is also forwarded to FSR's own built-in
      // RCAS stage. An FSR-namespaced key silently owning four non-FSR
      // upscaler paths would be a trap for anyone later reworking FSR.
      RTX_OPTION("rtx.sharpening", float, sharpness, 0.0f,
                 "Post-upscale RCAS sharpening amount. 0.0 = off (no sharpening pass is dispatched), 1.0 = maximum. "
                 "Applies to DLSS, DLSS Ray Reconstruction, XeSS and TAA-U through a standalone RCAS pass, and is "
                 "forwarded to FSR's built-in sharpener when FSR is the active upscaler.");
    };

    explicit DxvkRCAS(DxvkDevice* device);
    ~DxvkRCAS();

    void dispatch(
      Rc<RtxContext> ctx,
      const Resources::Resource& inputColor,
      const Resources::Resource& outputColor,
      const Rc<DxvkSampler>& linearSampler,
      float sharpness) const;

  private:
    Rc<vk::DeviceFn> m_vkd;
  };
}
