/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

// EA Importance-Sampled FAST noise — see rtx_fast_noise.h for full
// attribution. Data vendored from
// https://github.com/electronicarts/importance-sampled-FAST-noise under
// the BSD-3-Clause-variant license in data/ea-fast-noise/LICENSE.txt.

#include "rtx_fast_noise.h"
#include "rtx_fast_noise_data.h"
#include "../dxvk_context.h"

namespace dxvk {

  void RtxFastNoise::initialize(Rc<DxvkContext> ctx) {
    if (m_initialized) {
      return;
    }

    using namespace rtxFastNoise;

    // Allocate the VkImage as a 2D Array, RG8_UNORM, 32 layers.
    // TRANSFER_DST is needed because we copy data IN from a staging buffer
    // (rather than writing from a compute shader, which would need STORAGE).
    const VkExtent3D extent = { kWidth, kHeight, 1 };
    m_image = Resources::createImageResource(
      ctx,
      "Atmosphere FAST Noise",
      extent,
      VK_FORMAT_R8G8_UNORM,
      kSlices,                          // numLayers
      VK_IMAGE_TYPE_2D,
      VK_IMAGE_VIEW_TYPE_2D_ARRAY,
      0,                                // imageCreateFlags
      VK_IMAGE_USAGE_TRANSFER_DST_BIT,  // extraUsageFlags
      VkClearColorValue{},              // clearValue (unused)
      1                                 // mipLevels
    );

    // Upload the embedded byte data. DxvkContext::uploadImage handles the
    // staging-buffer dance, transfer-queue submit, layout transition, and
    // queue-ownership release to graphics internally. The source data is
    // tightly packed RG8 (no padding between rows or slices).
    VkImageSubresourceLayers subresources = {};
    subresources.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresources.mipLevel = 0;
    subresources.baseArrayLayer = 0;
    subresources.layerCount = kSlices;

    const VkDeviceSize pitchPerRow   = static_cast<VkDeviceSize>(kWidth) * kChannels;
    const VkDeviceSize pitchPerLayer = pitchPerRow * kHeight;

    ctx->uploadImage(
      m_image.image,
      subresources,
      kData,
      pitchPerRow,
      pitchPerLayer);

    m_initialized = true;
  }

}
