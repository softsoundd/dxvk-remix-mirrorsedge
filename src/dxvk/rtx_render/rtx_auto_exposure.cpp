/*
* Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_tone_mapping.h"
#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_render/rtx_shader_manager.h"
#include "rtx.h"
#include "rtx/pass/tonemap/tonemapping.h"

#include <rtx_shaders/auto_exposure.h>
#include <rtx_shaders/auto_exposure_histogram.h>
#include "rtx_imgui.h"
#include "rtx_render/rtx_debug_view.h"

#include "rtx/utility/debug_view_indices.h"

namespace dxvk {
  // Defined within an unnamed namespace to ensure unique definition across binary
  namespace {
    class AutoExposureHistogramShader : public ManagedShader {
      SHADER_SOURCE(AutoExposureHistogramShader, VK_SHADER_STAGE_COMPUTE_BIT, auto_exposure_histogram)

      PUSH_CONSTANTS(ToneMappingAutoExposureArgs)

      BEGIN_PARAMETER()
        RW_TEXTURE2D(AUTO_EXPOSURE_COLOR_INPUT)
        RW_TEXTURE1D(AUTO_EXPOSURE_HISTOGRAM_INPUT_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(AutoExposureHistogramShader);

    class AutoExposureShader : public ManagedShader
    {
      SHADER_SOURCE(AutoExposureShader, VK_SHADER_STAGE_COMPUTE_BIT, auto_exposure)

      PUSH_CONSTANTS(ToneMappingAutoExposureArgs)

      BEGIN_PARAMETER()
        RW_TEXTURE1D(AUTO_EXPOSURE_HISTOGRAM_INPUT_OUTPUT)
        RW_TEXTURE1D(AUTO_EXPOSURE_EXPOSURE_INPUT_OUTPUT)
        RW_TEXTURE2D(AUTO_EXPOSURE_DEBUG_VIEW_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(AutoExposureShader);
  }

  DxvkAutoExposure::DxvkAutoExposure(DxvkDevice* device)
  : CommonDeviceObject(device), m_vkd(device->vkd())  {
  }

  DxvkAutoExposure::~DxvkAutoExposure()  {  }

  void DxvkAutoExposure::showImguiSettings() {
    RemixGui::Checkbox("Perceptual Eye Adaptation", &enabledObject());
    if (enabled()) {
      ImGui::Indent();
      RemixGui::DragFloat("Light Adapt Tau (s)", &lightAdaptTauObject(), 0.005f, 0.01f, 5.f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Dark Adapt Tau (s)",  &darkAdaptTauObject(),  0.01f,  0.05f, 10.f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Target Mid-Gray (Yf)", &targetAdaptedYfObject(), 0.001f, 0.01f, 2.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Max Exposure (x)",     &maxExposureObject(),     0.1f,  1.0f, 64.0f, "%.2fx", ImGuiSliderFlags_AlwaysClamp);
      RemixGui::Separator();
      ImGui::Unindent();
    }
  }

  void DxvkAutoExposure::createResources(Rc<DxvkContext> ctx) {
    if (m_exposure.image != nullptr) {
      return;
    }

    DxvkImageCreateInfo desc;
    desc.type = VK_IMAGE_TYPE_1D;
    desc.flags = 0;
    desc.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    desc.numLayers = 1;
    desc.mipLevels = 1;
    desc.stages = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    desc.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    desc.tiling = VK_IMAGE_TILING_OPTIMAL;
    desc.layout = VK_IMAGE_LAYOUT_UNDEFINED;

    DxvkImageViewCreateInfo viewInfo;
    viewInfo.type = VK_IMAGE_VIEW_TYPE_1D;
    viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.minLevel = 0;
    viewInfo.numLevels = 1;
    viewInfo.minLayer = 0;
    viewInfo.numLayers = 1;
    viewInfo.format = desc.format;

    desc.extent = VkExtent3D{ 1, 1, 1 };

    viewInfo.format = desc.format = VK_FORMAT_R32_SFLOAT;
    viewInfo.usage = desc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    m_exposure.image = device()->createImage(desc, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXRenderTarget, "autoexposure");
    m_exposure.view = device()->createImageView(m_exposure.image, viewInfo);
    ctx->changeImageLayout(m_exposure.image, VK_IMAGE_LAYOUT_GENERAL);

    desc.extent = VkExtent3D { EXPOSURE_HISTOGRAM_SIZE, 1, 1 };
    viewInfo.format = desc.format = VK_FORMAT_R32_UINT;
    viewInfo.usage = desc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    m_exposureHistogram.image = device()->createImage(desc, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXRenderTarget, "autoexposure histogram");
    m_exposureHistogram.view = device()->createImageView(m_exposureHistogram.image, viewInfo);
    ctx->changeImageLayout(m_exposureHistogram.image, VK_IMAGE_LAYOUT_GENERAL);
  }

  void DxvkAutoExposure::dispatchAutoExposure(
    Rc<DxvkContext> ctx,
    Rc<DxvkSampler> linearSampler,
    const Resources::RaytracingOutput& rtOutput,
    const float frameTimeMilliseconds) {

    if (m_resetState || !enabled()) {
      VkClearColorValue clearColor;
      // Initial exposure scale = 1.0 so the first frame is unmodified.
      clearColor.float32[0] = clearColor.float32[1] = clearColor.float32[2] = clearColor.float32[3] = 1.0f;

      VkImageSubresourceRange subRange = {};
      subRange.layerCount = 1;
      subRange.levelCount = 1;
      subRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

      ctx->clearColorImage(m_exposure.image, clearColor, subRange);

      clearColor.uint32[0] = clearColor.uint32[1] = clearColor.uint32[2] = clearColor.uint32[3] = 0;
      ctx->clearColorImage(m_exposureHistogram.image, clearColor, subRange);
    }

    if (!enabled()) {
      return;
    }

    // Build push constants once; both passes use the same struct.
    ToneMappingAutoExposureArgs pushArgs = {};
    pushArgs.numPixels = rtOutput.m_finalOutputExtent.width * rtOutput.m_finalOutputExtent.height;
    {
      const float fallbackMs = RtxOptions::timeDeltaBetweenFrames() > 0.f ? RtxOptions::timeDeltaBetweenFrames() : 16.6f;
      const float effectiveFrameTimeMs = frameTimeMilliseconds > 0.0f ? frameTimeMilliseconds : fallbackMs;
      pushArgs.deltaTime = effectiveFrameTimeMs * 0.001f;
    }
    pushArgs.lightAdaptTau      = lightAdaptTau();
    pushArgs.darkAdaptTau       = darkAdaptTau();
    pushArgs.targetAdaptedYf    = targetAdaptedYf();
    pushArgs.maxExposure        = maxExposure();
    pushArgs.debugMode = (ctx->getCommonObjects()->metaDebugView().debugViewIdx() == DEBUG_VIEW_EXPOSURE_HISTOGRAM);

    {
      ScopedGpuProfileZone(ctx, "Histogram");
      static_cast<RtxContext*>(ctx.ptr())->setFramePassStage(RtxFramePassStage::AutoExposure_Histogram);
      ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);

      ctx->bindResourceView(AUTO_EXPOSURE_HISTOGRAM_INPUT_OUTPUT, m_exposureHistogram.view, nullptr);
      ctx->bindResourceView(AUTO_EXPOSURE_COLOR_INPUT, rtOutput.m_finalOutput.view(Resources::AccessType::Read), nullptr);

      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, AutoExposureHistogramShader::getShader());
      const VkExtent3D workgroups = util::computeBlockCount(rtOutput.m_finalOutputExtent, VkExtent3D { 16, 16, 1 });
      ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
    }

    {
      ScopedGpuProfileZone(ctx, "Exposure");
      static_cast<RtxContext*>(ctx.ptr())->setFramePassStage(RtxFramePassStage::AutoExposure_Exposure);
      DebugView& debugView = ctx->getDevice()->getCommon()->metaDebugView();

      ctx->bindResourceView(AUTO_EXPOSURE_HISTOGRAM_INPUT_OUTPUT, m_exposureHistogram.view, nullptr);
      ctx->bindResourceView(AUTO_EXPOSURE_EXPOSURE_INPUT_OUTPUT, m_exposure.view, nullptr);
      ctx->bindResourceView(AUTO_EXPOSURE_DEBUG_VIEW_OUTPUT, debugView.getDebugOutput(), nullptr);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, AutoExposureShader::getShader());
      ctx->dispatch(1, 1, 1);
    }
  }

  void DxvkAutoExposure::dispatch(
    Rc<DxvkContext> ctx,
    Rc<DxvkSampler> linearSampler,
    const Resources::RaytracingOutput& rtOutput,
    const float frameTimeMilliseconds,
    bool resetHistory) {

    ScopedGpuProfileZone(ctx, "Auto Exposure");

    m_resetState |= resetHistory;

    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);

    // TODO : set reset on significant camera changes as well
    if (m_exposureHistogram.image.ptr() == nullptr) {
      createResources(ctx);
      m_resetState = true;
    }

    dispatchAutoExposure(ctx, linearSampler, rtOutput, frameTimeMilliseconds);

    m_resetState = false;
  }
}
