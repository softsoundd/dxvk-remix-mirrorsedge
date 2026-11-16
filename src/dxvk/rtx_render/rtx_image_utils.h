/*
* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "dxvk_include.h"
#include "rtx_context.h"
#include "dxvk_pipemanager.h"

namespace dxvk {

  class DxvkDevice;

  /**
  * \brief Image utility shaders and related objects
  *
  * Creates the shaders, pipeline layouts, and
  * compute pipelines that are going to be used
  * for image operations.
  */
  class RtxImageUtils {
  public:
    RtxImageUtils(DxvkDevice* pDevice);

    enum class LatLongTransform {
      None,
      ZUp2OpenEXR,
    };

    /**
     * \brief Execute a compute shader to unwrap a cubemap image into a lat-long image
     */
    void cubemapToLatLong(Rc<DxvkContext>& ctx, Rc<DxvkImageView>& cube,
                          Rc<DxvkImageView>& latlong, LatLongTransform transform) const;

    /**
     * \brief Returns a 2D view's image as shaders sample it through the view
     *
     * A swizzled view, such as a D3D9 L8 or X8R8G8B8 texture's, is copied with its mips into
     * an RGBA8 image that carries the swizzle. Any other view returns its own image.
     */
    Rc<DxvkImage> copyAsSampled(Rc<DxvkContext>& ctx, const Rc<DxvkImageView>& view) const;
  };
}
