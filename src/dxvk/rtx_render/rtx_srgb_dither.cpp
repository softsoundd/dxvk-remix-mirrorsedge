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
#include "rtx_srgb_dither.h"
#include "rtx_context.h"
#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_render/rtx_shader_manager.h"
#include "rtx/pass/srgb_dither/srgb_dither.h"
#include "rtx/pass/tonemap/tonemapping.h"
#include "rtx_imgui.h"

#include <rtx_shaders/srgb_dither.h>

namespace dxvk {
  namespace {
    class SRGBDitherShader : public ManagedShader {
      SHADER_SOURCE(SRGBDitherShader, VK_SHADER_STAGE_COMPUTE_BIT, srgb_dither)

      PUSH_CONSTANTS(SRGBDitherArgs)

      BEGIN_PARAMETER()
        TEXTURE2DARRAY(SRGB_DITHER_BLUE_NOISE_TEXTURE_INPUT)
        RW_TEXTURE2D(SRGB_DITHER_COLOR_INPUT_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(SRGBDitherShader);
  }

  DxvkSRGBDither::DxvkSRGBDither(DxvkDevice* device)
    : CommonDeviceObject(device) { }

  void DxvkSRGBDither::showImguiSettings() {
    RemixGui::Combo("Dither Mode", &ditherModeObject(), "Disabled\0Spatial\0Spatial + Temporal\0");
  }

  uint32_t DxvkSRGBDither::getShaderDitherMode() {
    switch (ditherMode()) {
    default:
    case DitherMode::None: return ditherModeNone;
    case DitherMode::Spatial: return ditherModeSpatialOnly;
    case DitherMode::SpatialTemporal: return ditherModeSpatialTemporal;
    }
  }

  uint32_t DxvkSRGBDither::getDitherFrameIndex(const DxvkDevice* device) {
    return RtxOptions::rngSeedWithFrameIndex() ? device->getCurrentFrameId() : 0;
  }

  void DxvkSRGBDither::dispatch(
    Rc<RtxContext> ctx,
    const Resources::RaytracingOutput& rtOutput,
    bool performSRGBConversion) {

    ScopedGpuProfileZone(ctx, "sRGB + Dither");

    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);

    const Resources::Resource& inoutColorBuffer = rtOutput.m_finalOutput.resource(Resources::AccessType::ReadWrite);

    const VkExtent3D workgroups = util::computeBlockCount(inoutColorBuffer.view->imageInfo().extent, VkExtent3D{ 16, 16, 1 });

    SRGBDitherArgs pushArgs = {};
    pushArgs.performSRGBConversion = performSRGBConversion;
    pushArgs.ditherMode = getShaderDitherMode();
    pushArgs.frameIndex = getDitherFrameIndex(ctx->getDevice().ptr());

    ctx->bindResourceView(SRGB_DITHER_BLUE_NOISE_TEXTURE_INPUT, ctx->getResourceManager().getBlueNoiseTexture(ctx), nullptr);
    ctx->bindResourceView(SRGB_DITHER_COLOR_INPUT_OUTPUT, inoutColorBuffer.view, nullptr);

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, SRGBDitherShader::getShader());
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }
}
