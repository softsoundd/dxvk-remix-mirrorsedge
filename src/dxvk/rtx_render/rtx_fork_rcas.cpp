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
#include "rtx_context.h"
#include "rtx_fork_rcas.h"
#include "rtx_fork_hooks.h"
#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_render/rtx_shader_manager.h"
#include "rtx/pass/post_fx/fork_rcas.h"

#include <rtx_shaders/fork_rcas.h>

namespace dxvk {
  namespace {
    class RcasShader : public ManagedShader {
      SHADER_SOURCE(RcasShader, VK_SHADER_STAGE_COMPUTE_BIT, fork_rcas)

      PUSH_CONSTANTS(RcasArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(RCAS_INPUT)
        RW_TEXTURE2D(RCAS_OUTPUT)
        SAMPLER(RCAS_LINEAR_SAMPLER)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(RcasShader);
  }

  DxvkRCAS::DxvkRCAS(DxvkDevice* device)
    : m_vkd(device->vkd()) {
  }

  DxvkRCAS::~DxvkRCAS() {
  }

  void DxvkRCAS::dispatch(
    Rc<RtxContext> ctx,
    const Resources::Resource& inputColor,
    const Resources::Resource& outputColor,
    const Rc<DxvkSampler>& linearSampler,
    float sharpness) const {
    const VkExtent3D extent = inputColor.image->info().extent;

    RcasArgs args = {};
    args.imageSize = { extent.width, extent.height };
    args.invImageSize = { 1.0f / float(extent.width), 1.0f / float(extent.height) };
    args.sharpness = sharpness;

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, RcasShader::getShader());
    ctx->pushConstants(0, sizeof(args), &args);

    ctx->bindResourceView(RCAS_INPUT, inputColor.view, nullptr);
    ctx->bindResourceView(RCAS_OUTPUT, outputColor.view, nullptr);
    ctx->bindResourceSampler(RCAS_LINEAR_SAMPLER, linearSampler);

    const VkExtent3D workgroups = util::computeBlockCount(
      extent,
      VkExtent3D { RCAS_TILE_SIZE, RCAS_TILE_SIZE, 1 });
    ctx->dispatch(workgroups.width, workgroups.height, 1);
  }

  namespace fork_hooks {

    void dispatchRcasSharpening(RtxContext& ctx, const Resources::RaytracingOutput& rtOutput) {
      // FSR runs RCAS inside its own upscale dispatch, and NIS has its own
      // sharpening stage, so the standalone pass is only for the upscalers that
      // do not sharpen at all.
      const bool upscalerNeedsSharpening =
        ctx.m_currentUpscaler == RtxContext::InternalUpscaler::DLSS ||
        ctx.m_currentUpscaler == RtxContext::InternalUpscaler::DLSS_RR ||
        ctx.m_currentUpscaler == RtxContext::InternalUpscaler::XeSS ||
        ctx.m_currentUpscaler == RtxContext::InternalUpscaler::TAAU;

      if (!upscalerNeedsSharpening) {
        return;
      }

      const float sharpness = DxvkRCAS::Options::sharpness();
      if (sharpness <= 0.0f) {
        return;
      }

      ScopedGpuProfileZone(&ctx, "RCAS");

      ctx.getCommonObjects()->metaRCAS().dispatch(
        &ctx,
        rtOutput.m_finalOutput.resource(Resources::AccessType::Read),
        rtOutput.m_postFxIntermediateTexture,
        ctx.getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE),
        sharpness);

      ctx.copyImage(
        rtOutput.m_finalOutput.resource(Resources::AccessType::Write).image,
        { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        { 0, 0, 0 },
        rtOutput.m_postFxIntermediateTexture.image,
        { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        { 0, 0, 0 },
        rtOutput.m_finalOutputExtent);
    }

  } // namespace fork_hooks

}
