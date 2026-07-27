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
#include "rtx_ngx_passthrough.h"

#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_context.h"
#include "rtx_camera.h"
#include "rtx_camera_manager.h"
#include "rtx_scene_manager.h"
#include "rtx_ngx_wrapper.h"
#include "rtx_dlfg.h"
#include "rtx_dlss.h"
#include "rtx_nis.h"
#include "rtx_taa.h"
#ifndef _M_ARM64
#include "rtx_xess.h"
#endif
#include "rtx_postFx.h"
#include "rtx_auto_exposure.h"
#include "rtx_imgui.h"
#include "rtx_initializer.h"
#include "rtx_render/rtx_shader_manager.h"
#include "rtx/pass/ngx_passthrough/ngx_passthrough_args.h"
#include "rtx/pass/post_fx/post_fx.h"

#include <rtx_shaders/ngx_passthrough_mv.h>
#include <rtx_shaders/ngx_passthrough_alpha_merge.h>
#include <rtx_shaders/ngx_passthrough_velocity_vertex.h>
#include <rtx_shaders/ngx_passthrough_velocity_skinned_vertex.h>
#include <rtx_shaders/ngx_passthrough_velocity_dynamic_vertex.h>
#include <rtx_shaders/ngx_passthrough_velocity_fragment.h>

namespace dxvk {
  void RtxNgxPassthrough::enforceRaytracingDisabledForPassthrough() {
    if (ngxPassthroughMode() && RtxOptions::enableRaytracing()) {
      RtxOptions::enableRaytracing.setImmediately(false);
    }
  }

  void RtxNgxPassthrough::ngxPassthroughModeOnChange(DxvkDevice* device) {
    enforceRaytracingDisabledForPassthrough();

    // Only the passthrough shader set is prewarmed while the mode is active, so a runtime
    // disable kicks off the full prewarm; the async compilation gate in RtxContext::injectRTX
    // holds path tracing back until it completes. device is null during initial config parsing.
    if (device != nullptr && !ngxPassthroughMode()) {
      device->getCommon()->getRtxInitializer().startPrewarmShaders();
    }
  }

  namespace {
    class NgxPassthroughMvShader : public ManagedShader {
      SHADER_SOURCE(NgxPassthroughMvShader, VK_SHADER_STAGE_COMPUTE_BIT, ngx_passthrough_mv)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        TEXTURE2D(1)
        RW_TEXTURE2D(2)
        RW_TEXTURE2D(3)
        RW_TEXTURE2D(4)
        TEXTURE2D(5)
        RW_TEXTURE2D(8)
        RW_TEXTURE2D(9)
        TEXTURE2D(10)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(NgxPassthroughMvShader);

    class NgxPassthroughAlphaMergeShader : public ManagedShader {
      SHADER_SOURCE(NgxPassthroughAlphaMergeShader, VK_SHADER_STAGE_COMPUTE_BIT, ngx_passthrough_alpha_merge)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        TEXTURE2D(1)
        TEXTURE2D(2)
        RW_TEXTURE2D(3)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(NgxPassthroughAlphaMergeShader);

    // Interface slots are location bitmasks: the vertex stage consumes the position
    // attribute (location 0) and feeds two clip-space varyings (locations 0 and 1 -> mask
    // 0b11); the fragment stage consumes both and writes one render target (location 0)
    class NgxPassthroughVelocityVertexShader : public ManagedShader {
      SHADER_SOURCE(NgxPassthroughVelocityVertexShader, VK_SHADER_STAGE_VERTEX_BIT, ngx_passthrough_velocity_vertex)

      PUSH_CONSTANTS(NgxVelocityPushConstants)

      BEGIN_PARAMETER()
      END_PARAMETER()

      INTERFACE_INPUT_SLOTS(0b1);
      INTERFACE_OUTPUT_SLOTS(0b11);
    };

    // Dynamic-mesh variant (CPU-modified meshes): current positions from the game's
    // live buffer (location 0), previous positions from the capture snapshot (location 1)
    class NgxPassthroughVelocityDynamicVertexShader : public ManagedShader {
      SHADER_SOURCE(NgxPassthroughVelocityDynamicVertexShader, VK_SHADER_STAGE_VERTEX_BIT, ngx_passthrough_velocity_dynamic_vertex)

      PUSH_CONSTANTS(NgxVelocityPushConstants)

      BEGIN_PARAMETER()
      END_PARAMETER()

      INTERFACE_INPUT_SLOTS(0b11);
      INTERFACE_OUTPUT_SLOTS(0b11);
    };

    // Skinned variant: consumes position + blend indices + blend weights + the
    // instance-rate palette slot (locations 0-3) and reads the frame's bone palette blob
    class NgxPassthroughVelocitySkinnedVertexShader : public ManagedShader {
      SHADER_SOURCE(NgxPassthroughVelocitySkinnedVertexShader, VK_SHADER_STAGE_VERTEX_BIT, ngx_passthrough_velocity_skinned_vertex)

      PUSH_CONSTANTS(NgxVelocityPushConstants)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(1)
        STRUCTURED_BUFFER(2)
      END_PARAMETER()

      INTERFACE_INPUT_SLOTS(0b1111);
      INTERFACE_OUTPUT_SLOTS(0b11);
    };

    class NgxPassthroughVelocityFragmentShader : public ManagedShader {
      SHADER_SOURCE(NgxPassthroughVelocityFragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT, ngx_passthrough_velocity_fragment)

      BEGIN_PARAMETER()
        TEXTURE2D(0)
        CONSTANT_BUFFER(1)
      END_PARAMETER()

      INTERFACE_INPUT_SLOTS(0b11);
      INTERFACE_OUTPUT_SLOTS(0b1);
    };
  }

  void RtxNgxPassthrough::prewarmShaders(DxvkPipelineManager& pipelineManager) const {
    NgxPassthroughMvShader::getShader();
    NgxPassthroughAlphaMergeShader::getShader();
  }

  RtxNgxPassthrough::RtxNgxPassthrough(DxvkDevice* device)
    : CommonDeviceObject(device) {
    DxvkBufferCreateInfo info = {};
    info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    info.access = VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    info.size = sizeof(NgxPassthroughArgs);
    m_constantsBuffer = device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "NGX passthrough constants buffer");

    DxvkBufferCreateInfo velocityInfo = {};
    velocityInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    velocityInfo.stages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    velocityInfo.access = VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    velocityInfo.size = sizeof(NgxVelocityRasterArgs);
    m_velocityRasterConstants = device->createBuffer(velocityInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "NGX passthrough velocity raster constants");

    DxvkBufferCreateInfo boneInfo = {};
    boneInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    boneInfo.stages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    boneInfo.access = VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    // Per-draw blocks: [header][previous palette][current palette]
    boneInfo.size = VkDeviceSize(kNgxVelocityMaxSkinnedDraws) *
                    (1 + 2 * VkDeviceSize(kNgxVelocityBonePaletteRegisters)) * sizeof(Vector4);
    m_boneBuffer = device->createBuffer(boneInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "NGX passthrough velocity bones");

    // Identity ramp fetched as an instance-rate vertex attribute: each skinned draw binds
    // this at slot * 4 to deliver its palette block index to the vertex shader
    DxvkBufferCreateInfo slotsInfo = {};
    slotsInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    slotsInfo.stages = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    slotsInfo.access = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    slotsInfo.size = VkDeviceSize(kNgxVelocityMaxSkinnedDraws) * sizeof(uint32_t);
    m_boneSlotsBuffer = device->createBuffer(slotsInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "NGX passthrough velocity bone slots");

    // Previous-frame position snapshots for the CPU-modified mesh draws, bound as a
    // second vertex stream (tightly packed float3, whole-buffer indexing)
    DxvkBufferCreateInfo positionsInfo = {};
    positionsInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    positionsInfo.stages = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    positionsInfo.access = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    positionsInfo.size = VkDeviceSize(kNgxVelocityMaxDynamicDraws) * kNgxVelocityMaxDynamicVertices * sizeof(Vector3);
    m_dynamicPositionsBuffer = device->createBuffer(positionsInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "NGX passthrough velocity dynamic positions");
  }

  RtxNgxPassthrough::~RtxNgxPassthrough() { }

  void RtxNgxPassthrough::onDestroy() {
    for (DlssFeature& feature : m_dlssFeatures) {
      if (feature.context) {
        feature.context->releaseNGXFeature();
      }
      feature.context = nullptr;
    }
  }

  void RtxNgxPassthrough::createResources(Rc<DxvkContext> ctx, const VkExtent2D& renderExtent, const VkExtent2D& displayExtent) {
    if (m_renderExtent.width == renderExtent.width && m_renderExtent.height == renderExtent.height &&
        m_displayExtent.width == displayExtent.width && m_displayExtent.height == displayExtent.height) {
      return;
    }

    m_renderExtent = renderExtent;
    m_displayExtent = displayExtent;

    const VkExtent3D renderExtent3D = { renderExtent.width, renderExtent.height, 1 };
    const VkExtent3D displayExtent3D = { displayExtent.width, displayExtent.height, 1 };

    m_colorInput = Resources::createImageResource(ctx, "NGX passthrough color input", renderExtent3D, VK_FORMAT_R16G16B16A16_SFLOAT);
    m_dlssOutput = Resources::createImageResource(ctx, "NGX passthrough DLSS output", displayExtent3D, VK_FORMAT_R16G16B16A16_SFLOAT);
    m_mergedOutput = Resources::createImageResource(ctx, "NGX passthrough merged output", displayExtent3D, VK_FORMAT_R16G16B16A16_SFLOAT);
    // Render-sized: the inputs the debug view visualizes are render resolution
    m_debugPresentImage = Resources::createImageResource(ctx, "NGX passthrough debug overlay", renderExtent3D, VK_FORMAT_R16G16B16A16_SFLOAT);
    m_debugPresentValid = false;

    for (uint32_t i = 0; i < m_depthQueue.size(); i++) {
      m_depthQueue[i] = Resources::createImageResource(ctx, "NGX passthrough depth", renderExtent3D, VK_FORMAT_R32_SFLOAT);
      m_motionVectorQueue[i] = Resources::createImageResource(ctx, "NGX passthrough motion vectors", renderExtent3D, VK_FORMAT_R16G16_SFLOAT);
    }

    // Remix post-processing inputs (rtx.postfx motion blur)
    m_linearViewZ = Resources::createImageResource(ctx, "NGX passthrough linear view Z", renderExtent3D, VK_FORMAT_R32_SFLOAT);
    m_surfaceFlags = Resources::createImageResource(ctx, "NGX passthrough surface flags", renderExtent3D, VK_FORMAT_R8_UINT);
    m_surfaceFlagsScratch1 = Resources::AliasedResource(ctx, renderExtent3D, VK_FORMAT_R8_UINT, "NGX passthrough surface flags scratch 1");
    m_surfaceFlagsScratch2 = Resources::AliasedResource(ctx, renderExtent3D, VK_FORMAT_R8_UINT, "NGX passthrough surface flags scratch 2");

    // Cinematic motion blur intermediates. These live at display resolution because the
    // filter runs on the upscaled colour; the tile textures are sized for the smallest
    // supported tile so changing the blur radius at runtime never reallocates.
    {
      const VkExtent3D tileExtent = {
        (displayExtent.width + POST_FX_MB_TILE_SIZE_MIN - 1) / POST_FX_MB_TILE_SIZE_MIN,
        (displayExtent.height + POST_FX_MB_TILE_SIZE_MIN - 1) / POST_FX_MB_TILE_SIZE_MIN,
        1
      };
      const VkExtent3D tileColumnExtent = { tileExtent.width, displayExtent.height, 1 };

      m_motionBlurCineVelocityDepth = Resources::createImageResource(ctx, "NGX passthrough motion blur velocity depth", displayExtent3D, VK_FORMAT_R16G16B16A16_SFLOAT);
      m_motionBlurCineCurvature = Resources::createImageResource(ctx, "NGX passthrough motion blur curvature", displayExtent3D, VK_FORMAT_R16G16_SFLOAT);
      m_motionBlurCineTileMaxX = Resources::createImageResource(ctx, "NGX passthrough motion blur tile max x", tileColumnExtent, VK_FORMAT_R16G16B16A16_SFLOAT);
      m_motionBlurCineTileMax = Resources::createImageResource(ctx, "NGX passthrough motion blur tile max", tileExtent, VK_FORMAT_R16G16B16A16_SFLOAT);
      m_motionBlurCineNeighborMax = Resources::createImageResource(ctx, "NGX passthrough motion blur neighbor max", tileExtent, VK_FORMAT_R16G16B16A16_SFLOAT);
    }

    // Per-object velocity raster target (color attachment for the raster, sampled by the
    // motion vector pass; no storage usage needed): RG = NDC delta, B = phase ownership
    // marker (world vs foreground DPG draws, rendered as two passes into this one image)
    m_objectVelocity = Resources::createImageResource(ctx, "NGX passthrough object velocity", renderExtent3D, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                      1, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 0,
                                                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);

    for (DlssFeature& feature : m_dlssFeatures) {
      feature.needsInitialize = true;
    }

    Logger::info(str::format("[RTX NGX Passthrough] Created resources: render ", renderExtent.width, "x", renderExtent.height,
                             ", display ", displayExtent.width, "x", displayExtent.height));
  }

  void RtxNgxPassthrough::captureDepthSnapshot(RtxContext* ctx, const Rc<DxvkImage>& sceneDepthImage) {
    if (sceneDepthImage == nullptr ||
        sceneDepthImage->info().sampleCount != VK_SAMPLE_COUNT_1_BIT ||
        sceneDepthImage->info().layout != VK_IMAGE_LAYOUT_GENERAL) {
      return;
    }

    const DxvkImageCreateInfo& sourceInfo = sceneDepthImage->info();

    // Lazily (re)create the snapshot image + sampled depth view to match the game's buffer
    if (m_depthSnapshotImage == nullptr ||
        m_depthSnapshotImage->info().format != sourceInfo.format ||
        m_depthSnapshotImage->info().extent.width != sourceInfo.extent.width ||
        m_depthSnapshotImage->info().extent.height != sourceInfo.extent.height) {
      DxvkImageCreateInfo imageInfo;
      imageInfo.type = VK_IMAGE_TYPE_2D;
      imageInfo.format = sourceInfo.format;
      imageInfo.flags = 0;
      imageInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
      imageInfo.extent = { sourceInfo.extent.width, sourceInfo.extent.height, 1 };
      imageInfo.numLayers = 1;
      imageInfo.mipLevels = 1;
      imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      imageInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
      imageInfo.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
      imageInfo.layout = VK_IMAGE_LAYOUT_GENERAL;

      m_depthSnapshotImage = m_device->createImage(imageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                                   DxvkMemoryStats::Category::RTXRenderTarget, "NGX passthrough depth snapshot");
      ctx->changeImageLayout(m_depthSnapshotImage, VK_IMAGE_LAYOUT_GENERAL);

      DxvkImageViewCreateInfo viewInfo;
      viewInfo.type = VK_IMAGE_VIEW_TYPE_2D;
      viewInfo.format = sourceInfo.format;
      viewInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
      viewInfo.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
      viewInfo.minLevel = 0;
      viewInfo.numLevels = 1;
      viewInfo.minLayer = 0;
      viewInfo.numLayers = 1;

      m_depthSnapshotView = m_device->createImageView(m_depthSnapshotImage, viewInfo);
    }

    const VkImageSubresourceLayers subresourceLayers = {
      imageFormatInfo(sourceInfo.format)->aspectMask, 0, 0, 1
    };

    ctx->copyImage(m_depthSnapshotImage, subresourceLayers, { 0, 0, 0 },
                   sceneDepthImage, subresourceLayers, { 0, 0, 0 },
                   m_depthSnapshotImage->info().extent);

    m_depthSnapshotPending = true;

    ONCE(Logger::info("[RTX NGX Passthrough] World depth snapshot active (game clears depth mid-scene, e.g. UE3 foreground pass)."));
  }

  void RtxNgxPassthrough::captureHudless(RtxContext* ctx, const Rc<DxvkImage>& backbufferImage) {
    if (backbufferImage == nullptr || backbufferImage->info().sampleCount != VK_SAMPLE_COUNT_1_BIT) {
      return;
    }

    const DxvkImageCreateInfo& sourceInfo = backbufferImage->info();

    // Lazily (re)create the slots to match the backbuffer (format conversion is not needed:
    // the HUD-less input should match the presented backbuffer's format and color space)
    Resources::Resource& slot = m_hudlessQueue.get();

    if (slot.image == nullptr ||
        slot.image->info().format != sourceInfo.format ||
        slot.image->info().extent.width != sourceInfo.extent.width ||
        slot.image->info().extent.height != sourceInfo.extent.height) {
      Rc<DxvkContext> dxvkCtx = ctx;

      const VkExtent3D extent = { sourceInfo.extent.width, sourceInfo.extent.height, 1 };
      for (uint32_t i = 0; i < m_hudlessQueue.size(); i++) {
        m_hudlessQueue[i] = Resources::createImageResource(dxvkCtx, "NGX passthrough hudless", extent, sourceInfo.format,
                                                           1, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 0,
                                                           /* extraUsageFlags = */ 0);
      }
    }

    const VkImageSubresourceLayers subresourceLayers = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };

    ctx->copyImage(slot.image, subresourceLayers, { 0, 0, 0 },
                   backbufferImage, subresourceLayers, { 0, 0, 0 },
                   slot.image->info().extent);

    ONCE(Logger::info("[RTX NGX Passthrough] HUD-less frame generation input active."));
  }

  uint32_t RtxNgxPassthrough::rasterizeObjectVelocities(RtxContext* ctx,
                                                        DxvkContextState& dxvkCtxState,
                                                        const std::vector<NgxVelocityDraw>& velocityDraws,
                                                        const Rc<DxvkImageView>& worldDepthView,
                                                        const Rc<DxvkImageView>& foregroundDepthView,
                                                        const VkOffset2D& subrectOffset,
                                                        const float jitter[2]) {
    m_lastVelocityDrawCount = uint32_t(velocityDraws.size());

    if (velocityDraws.empty() || worldDepthView == nullptr || m_objectVelocity.image == nullptr) {
      return 0;
    }

    ScopedGpuProfileZone(ctx, "NGX Passthrough Object Velocities");

    ONCE(Logger::info("[RTX NGX Passthrough] Object velocity raster active (dynamic objects feed true motion vectors)."));

    uint32_t worldDrawCount = 0;
    uint32_t foregroundDrawCount = 0;
    for (const NgxVelocityDraw& draw : velocityDraws) {
      (draw.foregroundPhase ? foregroundDrawCount : worldDrawCount)++;
    }

    const bool haveForeground = foregroundDrawCount > 0 && foregroundDepthView != nullptr;

    // Diagnostic freeze: the previous-frame side is composed from the current frame's
    // data, forcing zero object velocity - isolates current-side replay correctness
    // (skinning, transforms, depth test) from history pairing issues
    const bool freezePrevious = objectVelocityDebugFreeze();

    // Bone palettes for the skinned draws, uploaded as one blob of draw-sized blocks:
    // per draw [header (index scale, influence count)][previous][current], palettes of
    // kNgxVelocityBonePaletteRegisters each (padded by the capture). Each draw selects
    // its block via the instance-rate slot attribute.
    std::vector<Vector4> boneBlob;
    std::vector<uint32_t> drawBoneSlots(velocityDraws.size(), UINT32_MAX);

    {
      uint32_t skinnedDrawCount = 0;
      for (size_t i = 0; i < velocityDraws.size(); i++) {
        const NgxVelocityDraw& draw = velocityDraws[i];
        if (draw.bonesCurrent.size() != kNgxVelocityBonePaletteRegisters ||
            draw.bonesPrevious.size() != kNgxVelocityBonePaletteRegisters ||
            skinnedDrawCount >= kNgxVelocityMaxSkinnedDraws) {
          continue;
        }

        const std::vector<Vector4>& previousPalette = freezePrevious ? draw.bonesCurrent : draw.bonesPrevious;

        drawBoneSlots[i] = skinnedDrawCount;
        boneBlob.push_back(Vector4(float(draw.boneIndexScale), float(draw.skinInfluenceCount), 0.0f, 0.0f));
        boneBlob.insert(boneBlob.end(), previousPalette.begin(), previousPalette.end());
        boneBlob.insert(boneBlob.end(), draw.bonesCurrent.begin(), draw.bonesCurrent.end());
        skinnedDrawCount++;
      }

      if (skinnedDrawCount > 0) {
        ctx->updateBuffer(m_boneBuffer, 0, boneBlob.size() * sizeof(Vector4), boneBlob.data());

        // Identity ramp for the slot attribute (device-local; filled once)
        if (!m_boneSlotsInitialized) {
          m_boneSlotsInitialized = true;
          std::array<uint32_t, kNgxVelocityMaxSkinnedDraws> slotRamp;
          for (uint32_t slot = 0; slot < kNgxVelocityMaxSkinnedDraws; slot++) {
            slotRamp[slot] = slot;
          }
          ctx->updateBuffer(m_boneSlotsBuffer, 0, slotRamp.size() * sizeof(uint32_t), slotRamp.data());
        }
      }
    }

    // Previous-position snapshots for the CPU-modified mesh draws (per-draw byte
    // offsets into one upload; the freeze diagnostic binds the live buffer instead)
    std::vector<VkDeviceSize> drawPositionOffsets(velocityDraws.size(), VkDeviceSize(-1));

    {
      VkDeviceSize uploadOffset = 0;
      for (size_t i = 0; i < velocityDraws.size(); i++) {
        const NgxVelocityDraw& draw = velocityDraws[i];
        if (draw.previousPositions.empty() ||
            draw.previousPositions.size() > kNgxVelocityMaxDynamicVertices ||
            uploadOffset + draw.previousPositions.size() * sizeof(Vector3) > m_dynamicPositionsBuffer->info().size) {
          continue;
        }

        drawPositionOffsets[i] = uploadOffset;
        ctx->updateBuffer(m_dynamicPositionsBuffer, uploadOffset,
                          draw.previousPositions.size() * sizeof(Vector3), draw.previousPositions.data());
        uploadOffset += draw.previousPositions.size() * sizeof(Vector3);
      }
    }

    // Sentinel clear: the motion vector pass treats anything above -50 as a real NDC delta
    const VkClearColorValue sentinelClear = { { -100.0f, -100.0f, 0.0f, 0.0f } };
    const VkImageSubresourceRange clearRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    ctx->clearColorImage(m_objectVelocity.image, sentinelClear, clearRange);

    NgxVelocityRasterArgs rasterArgs = {};
    rasterArgs.subrectOffset = vec2(float(subrectOffset.x), float(subrectOffset.y));
    rasterArgs.depthToleranceScale = std::max(objectVelocityDepthTolerance(), 1.0f);
    rasterArgs.bonePaletteRegisterCount = kNgxVelocityBonePaletteRegisters;
    // The freeze diagnostic also exposes the raw rasterized footprint (no depth test):
    // coverage hugging the mesh silhouettes with zero motion proves the replay; coverage
    // exploding proves a geometry problem
    rasterArgs.debugSkipDepthTest = freezePrevious ? 1u : 0u;
    rasterArgs.foregroundPass = 0u;
    ctx->updateBuffer(m_velocityRasterConstants, 0, sizeof(rasterArgs), &rasterArgs);

    // Save the game's context state; the raster below rebinds most of it
    DxvkContextState stateCopy = dxvkCtxState;

    DxvkInputAssemblyState iaState;
    iaState.primitiveTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    iaState.primitiveRestart = VK_FALSE;
    iaState.patchVertexCount = 0;
    ctx->setInputAssemblyState(iaState);

    DxvkRasterizerState rsState;
    rsState.polygonMode = VK_POLYGON_MODE_FILL;
    // No culling: UE3 mover transforms may mirror, and the game's winding conventions are
    // not replicated here; occlusion is resolved by the manual depth compare instead
    rsState.cullMode = VK_CULL_MODE_NONE;
    rsState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rsState.depthClipEnable = VK_TRUE;
    rsState.depthBiasEnable = VK_FALSE;
    rsState.conservativeMode = VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT;
    rsState.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    ctx->setRasterizerState(rsState);

    DxvkMultisampleState msState;
    msState.sampleMask = 0xffffffff;
    msState.enableAlphaToCoverage = VK_FALSE;
    ctx->setMultisampleState(msState);

    VkStencilOpState stencilOp;
    stencilOp.failOp = VK_STENCIL_OP_KEEP;
    stencilOp.passOp = VK_STENCIL_OP_KEEP;
    stencilOp.depthFailOp = VK_STENCIL_OP_KEEP;
    stencilOp.compareOp = VK_COMPARE_OP_ALWAYS;
    stencilOp.compareMask = 0xFFFFFFFF;
    stencilOp.writeMask = 0xFFFFFFFF;
    stencilOp.reference = 0;

    DxvkDepthStencilState dsState;
    dsState.enableDepthTest = VK_FALSE;
    dsState.enableDepthWrite = VK_FALSE;
    dsState.enableStencilTest = VK_FALSE;
    dsState.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    dsState.stencilOpFront = stencilOp;
    dsState.stencilOpBack = stencilOp;
    ctx->setDepthStencilState(dsState);

    DxvkLogicOpState loState;
    loState.enableLogicOp = VK_FALSE;
    loState.logicOp = VK_LOGIC_OP_NO_OP;
    ctx->setLogicOpState(loState);

    DxvkBlendMode blendMode;
    blendMode.enableBlending = VK_FALSE;
    blendMode.colorSrcFactor = VK_BLEND_FACTOR_ONE;
    blendMode.colorDstFactor = VK_BLEND_FACTOR_ZERO;
    blendMode.colorBlendOp = VK_BLEND_OP_ADD;
    blendMode.alphaSrcFactor = VK_BLEND_FACTOR_ONE;
    blendMode.alphaDstFactor = VK_BLEND_FACTOR_ZERO;
    blendMode.alphaBlendOp = VK_BLEND_OP_ADD;
    blendMode.writeMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    ctx->setBlendMode(0, blendMode);

    // The clip transforms are the game's own (D3D9 y-up clip space), so the raster must
    // replicate the exact viewport DXVK gives the game's draws (D3D9DeviceEx::
    // BindViewportAndScissor): the negative-height flip that turns y-up clip space into
    // Vulkan's y-down convention, the D3D9 half-pixel correctness factor, and this frame's
    // sub-pixel jitter. Anything less and the geometry lands on different pixels than the
    // game's own rasterization of it - vertically mirrored without the flip, sub-pixel
    // shifted without cf/jitter - which both misplaces the velocities and breaks the
    // manual depth compare against the game's depth buffer.
    const float cf = 0.5f - (1.0f / 128.0f);

    VkViewport viewport;
    viewport.x = cf + jitter[0];
    viewport.y = float(m_renderExtent.height) + cf + jitter[1];
    viewport.width = float(m_renderExtent.width);
    viewport.height = -float(m_renderExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor;
    scissor.offset = { 0, 0 };
    scissor.extent = { m_renderExtent.width, m_renderExtent.height };
    ctx->setViewports(1, &viewport, &scissor);

    // Per-draw viewport depth range: the game maps some passes into a squashed range
    // (UE3's foreground DPG keeps first person meshes in front of the world) and the
    // raster's depth must match the game's buffer exactly
    float boundMinDepth = viewport.minDepth;
    float boundMaxDepth = viewport.maxDepth;
    const auto applyViewportDepthRange = [&](float minDepth, float maxDepth) {
      if (minDepth == boundMinDepth && maxDepth == boundMaxDepth) {
        return;
      }
      boundMinDepth = minDepth;
      boundMaxDepth = maxDepth;
      viewport.minDepth = minDepth;
      viewport.maxDepth = maxDepth;
      ctx->setViewports(1, &viewport, &scissor);
    };
    ctx->bindShader(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_GEOMETRY_BIT, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_FRAGMENT_BIT, NgxPassthroughVelocityFragmentShader::getShader());

    ctx->bindResourceBuffer(1, DxvkBufferSlice(m_velocityRasterConstants, 0, m_velocityRasterConstants->info().size));

    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);

    const DxvkVertexBinding vertexBinding = { 0, 0, VK_VERTEX_INPUT_RATE_VERTEX };

    DxvkRenderTargets renderTargets;
    renderTargets.color[0].view = m_objectVelocity.view;
    renderTargets.color[0].layout = VK_IMAGE_LAYOUT_GENERAL;
    ctx->bindRenderTargets(renderTargets);

    // One pass per phase into the shared target: world draws against the world depth,
    // foreground draws (UE3 foreground DPG) against the live foreground depth, told apart
    // by the phase marker each pass writes. Rigid draws first, then CPU-modified, then
    // skinned (one pipeline switch each).
    const auto drawPass = [&](bool foregroundPass, const Rc<DxvkImageView>& depthView) {
      ctx->bindResourceView(0, depthView, nullptr);

      const auto pushDrawConstants = [&](const NgxVelocityDraw& draw) {
        NgxVelocityPushConstants pushConstants;
        pushConstants.clipFromLocal = draw.clipFromLocal;
        pushConstants.prevClipFromLocal = freezePrevious ? draw.clipFromLocal : draw.prevClipFromLocal;
        ctx->pushConstants(0, sizeof(pushConstants), &pushConstants);
      };

      bool rigidShaderBound = false;
      for (const NgxVelocityDraw& draw : velocityDraws) {
        if (draw.foregroundPhase != foregroundPass || !draw.bonesCurrent.empty() || !draw.previousPositions.empty()) {
          continue;
        }

        if (!rigidShaderBound) {
          ctx->bindShader(VK_SHADER_STAGE_VERTEX_BIT, NgxPassthroughVelocityVertexShader::getShader());
          rigidShaderBound = true;
        }

        const DxvkVertexAttribute attribute = { 0, 0, draw.positionFormat, draw.positionOffset };
        ctx->setInputLayout(1, &attribute, 1, &vertexBinding);

        ctx->bindVertexBuffer(0, draw.vertexBuffer, draw.vertexStride);
        ctx->bindIndexBuffer(draw.indexBuffer, draw.indexType);
        applyViewportDepthRange(draw.viewportMinZ, draw.viewportMaxZ);
        pushDrawConstants(draw);
        ctx->drawIndexed(draw.indexCount, 1, draw.firstIndex, draw.vertexOffset, 0);
      }

      // CPU-modified meshes: current positions from the game's live buffer, previous
      // from the capture snapshot on a second stream (whole-buffer indexed, so the
      // draw's base vertex applies to both). The freeze diagnostic binds the live
      // buffer as the previous stream, forcing zero motion.
      bool dynamicShaderBound = false;
      for (size_t i = 0; i < velocityDraws.size(); i++) {
        const NgxVelocityDraw& draw = velocityDraws[i];
        if (draw.foregroundPhase != foregroundPass || draw.previousPositions.empty() ||
            drawPositionOffsets[i] == VkDeviceSize(-1)) {
          continue;
        }

        if (!dynamicShaderBound) {
          ctx->bindShader(VK_SHADER_STAGE_VERTEX_BIT, NgxPassthroughVelocityDynamicVertexShader::getShader());
          dynamicShaderBound = true;
        }

        const DxvkVertexAttribute attributes[2] = {
          { 0, 0, draw.positionFormat, draw.positionOffset },
          { 1, 1, VK_FORMAT_R32G32B32_SFLOAT, 0 },
        };
        const DxvkVertexBinding bindings[2] = {
          vertexBinding,
          { 1, 0, VK_VERTEX_INPUT_RATE_VERTEX },
        };
        ctx->setInputLayout(2, attributes, 2, bindings);

        ctx->bindVertexBuffer(0, draw.vertexBuffer, draw.vertexStride);
        if (freezePrevious) {
          ctx->bindVertexBuffer(1, DxvkBufferSlice(draw.vertexBuffer.buffer(),
                                                   draw.vertexBuffer.offset() + draw.positionOffset,
                                                   draw.vertexBuffer.length() - draw.positionOffset),
                                draw.vertexStride);
        } else {
          ctx->bindVertexBuffer(1, DxvkBufferSlice(m_dynamicPositionsBuffer, drawPositionOffsets[i],
                                                   draw.previousPositions.size() * sizeof(Vector3)),
                                sizeof(Vector3));
        }
        ctx->bindIndexBuffer(draw.indexBuffer, draw.indexType);
        applyViewportDepthRange(draw.viewportMinZ, draw.viewportMaxZ);
        pushDrawConstants(draw);
        ctx->drawIndexed(draw.indexCount, 1, draw.firstIndex, draw.vertexOffset, 0);
      }

      bool skinnedShaderBound = false;
      for (size_t i = 0; i < velocityDraws.size(); i++) {
        const NgxVelocityDraw& draw = velocityDraws[i];
        if (draw.foregroundPhase != foregroundPass || draw.bonesCurrent.empty() ||
            drawBoneSlots[i] == UINT32_MAX) {
          continue;
        }

        if (!skinnedShaderBound) {
          ctx->bindShader(VK_SHADER_STAGE_VERTEX_BIT, NgxPassthroughVelocitySkinnedVertexShader::getShader());
          ctx->bindResourceBuffer(2, DxvkBufferSlice(m_boneBuffer, 0, m_boneBuffer->info().size));
          skinnedShaderBound = true;
        }

        const DxvkVertexAttribute attributes[4] = {
          { 0, 0, draw.positionFormat, draw.positionOffset },
          { 1, 0, VK_FORMAT_R8G8B8A8_UINT, draw.blendIndicesOffset },
          { 2, 0, VK_FORMAT_R8G8B8A8_UNORM, draw.blendWeightsOffset },
          { 3, 1, VK_FORMAT_R32_UINT, 0 },
        };
        const DxvkVertexBinding bindings[2] = {
          vertexBinding,
          { 1, 0, VK_VERTEX_INPUT_RATE_INSTANCE },
        };
        ctx->setInputLayout(4, attributes, 2, bindings);

        ctx->bindVertexBuffer(0, draw.vertexBuffer, draw.vertexStride);
        // The slot attribute: a single instance at instance index 0 fetches exactly the
        // element this per-draw offset selects
        ctx->bindVertexBuffer(1, DxvkBufferSlice(m_boneSlotsBuffer, drawBoneSlots[i] * sizeof(uint32_t), sizeof(uint32_t)),
                              sizeof(uint32_t));
        ctx->bindIndexBuffer(draw.indexBuffer, draw.indexType);
        applyViewportDepthRange(draw.viewportMinZ, draw.viewportMaxZ);
        pushDrawConstants(draw);
        ctx->drawIndexed(draw.indexCount, 1, draw.firstIndex, draw.vertexOffset, 0);
      }
    };

    if (worldDrawCount > 0) {
      drawPass(false, worldDepthView);
    }
    if (haveForeground) {
      // Phase flag flip between the passes (the constant update is ordered against the
      // draws on the CS timeline). Rendering last, the foreground meshes win overlapping
      // pixels - they are the nearest content by definition.
      rasterArgs.foregroundPass = 1u;
      ctx->updateBuffer(m_velocityRasterConstants, 0, sizeof(rasterArgs), &rasterArgs);

      drawPass(true, foregroundDepthView);
    }

    dxvkCtxState = stateCopy;

    return (worldDrawCount > 0 ? 1u : 0u) | (haveForeground ? 2u : 0u);
  }

  bool RtxNgxPassthrough::generateMotionVectorsAndDepth(RtxContext* ctx,
                                                        DxvkContextState& dxvkCtxState,
                                                        DxvkBarrierSet& barriers,
                                                        const Rc<DxvkImage>& sceneDepthImage,
                                                        const VkOffset2D& subrectOffset,
                                                        const RtCamera& camera,
                                                        const std::vector<NgxVelocityDraw>& velocityDraws,
                                                        const float jitter[2]) {
    // Two depth sources, merged per pixel by the shader (min): the first-clear snapshot
    // holds the world + intermediate DPG depth (UE3 clears depth mid-scene ahead of its
    // foreground DPG; with occlusion culling enabled an extra clear precedes that one),
    // the live buffer at the injection point holds the foreground DPG (camera-locked first
    // person meshes, which Mirror's Edge movestates shuffle between SDPG_Intermediate and
    // SDPG_Foreground per move; the foreground DPG is the last depth writer of the frame).
    // Without a snapshot both sources alias the live buffer.
    const bool haveSnapshot = m_depthSnapshotPending && m_depthSnapshotImage != nullptr;
    m_depthSnapshotPending = false;

    // No scene depth at all: none was identified, or the D3D9 layer deliberately withheld
    // the frame's inputs (e.g. the scene rendered into a reduced subrect without the Super
    // Resolution interception). Fatal regardless of a pending snapshot - synthesizing
    // inputs from the snapshot alone would feed DLSS subrect content in a full-size field.
    if (sceneDepthImage == nullptr) {
      m_statusReason = "no scene depth identified";
      ONCE(Logger::warn("[RTX NGX Passthrough] No scene depth buffer identified this frame; DLSS/DLFG inputs unavailable."));
      return false;
    }

    // Validate that the live buffer can be sampled; the snapshot (always shader-readable)
    // substitutes as the sole source when it cannot
    const char* liveFailureReason = nullptr;

    if (sceneDepthImage->info().sampleCount != VK_SAMPLE_COUNT_1_BIT) {
      liveFailureReason = "multisampled depth unsupported";
      ONCE(Logger::warn("[RTX NGX Passthrough] The game's depth buffer is multisampled, which cannot be "
                        "resolved for the NGX inputs. Leave rtx.ngxPassthrough.disableGameMsaa enabled so "
                        "multisampling is refused at creation, then restart."));
    } else if (sceneDepthImage->info().layout != VK_IMAGE_LAYOUT_GENERAL) {
      // Game depth-stencil images are created with a GENERAL layout when the mode is enabled
      // at launch (see D3D9CommonTexture::CreateImage). Anything else means the image predates
      // the mode being turned on and cannot be legally sampled.
      liveFailureReason = "depth not shader-readable (enable mode at launch)";
      ONCE(Logger::warn("[RTX NGX Passthrough] Game depth buffer is not shader-readable; rtx.ngxPassthroughMode must be set at launch (rtx.conf), not toggled at runtime."));
    }

    const bool liveValid = liveFailureReason == nullptr;

    if (!haveSnapshot && !liveValid) {
      m_statusReason = liveFailureReason;
      return false;
    }

    // Lazily (re)create the sampled depth-aspect view over the game's depth image
    if (liveValid && (m_gameDepthView == nullptr || m_gameDepthViewImage != sceneDepthImage.ptr())) {
      DxvkImageViewCreateInfo viewInfo;
      viewInfo.type = VK_IMAGE_VIEW_TYPE_2D;
      viewInfo.format = sceneDepthImage->info().format;
      viewInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
      viewInfo.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
      viewInfo.minLevel = 0;
      viewInfo.numLevels = 1;
      viewInfo.minLayer = 0;
      viewInfo.numLayers = 1;

      m_gameDepthView = m_device->createImageView(sceneDepthImage, viewInfo);
      m_gameDepthViewImage = sceneDepthImage.ptr();
    }

    // World depth: the pre-clear snapshot when one was taken this frame, else the live buffer
    const Rc<DxvkImage>& worldDepthImage = haveSnapshot ? m_depthSnapshotImage : sceneDepthImage;
    const Rc<DxvkImageView>& worldDepthView = haveSnapshot ? m_depthSnapshotView : m_gameDepthView;

    // Foreground depth: the live buffer when usable, else alias world (identity merge)
    const bool separateForeground = haveSnapshot && liveValid &&
      sceneDepthImage->info().extent.width == worldDepthImage->info().extent.width &&
      sceneDepthImage->info().extent.height == worldDepthImage->info().extent.height;

    const Rc<DxvkImage>& foregroundDepthImage = separateForeground ? sceneDepthImage : worldDepthImage;
    const Rc<DxvkImageView>& foregroundDepthView = separateForeground ? m_gameDepthView : worldDepthView;

    const VkExtent3D depthExtent = worldDepthImage->info().extent;

    if (depthExtent.width < uint32_t(subrectOffset.x) + m_renderExtent.width ||
        depthExtent.height < uint32_t(subrectOffset.y) + m_renderExtent.height) {
      m_statusReason = "depth does not cover scene subrect";
      ONCE(Logger::warn(str::format("[RTX NGX Passthrough] Scene depth buffer (", depthExtent.width, "x", depthExtent.height,
                                    ") does not cover the scene subrect (", subrectOffset.x, ",", subrectOffset.y, " ",
                                    m_renderExtent.width, "x", m_renderExtent.height, "); skipping DLSS inputs.")));
      return false;
    }

    // Rasterize the dynamic objects' true motion before the motion vector pass consumes it
    // (context-managed synchronization: the raster is regular bound-state drawing)
    const uint32_t velocityTargets = rasterizeObjectVelocities(ctx, dxvkCtxState, velocityDraws,
                                                               worldDepthView, foregroundDepthView,
                                                               subrectOffset, jitter);

    // Constants: map current-frame unjittered NDC to previous-frame clip space. Computed in
    // double precision from the camera matrix cache; the view-to-previous-view leg keeps the
    // combined matrix well conditioned for large world coordinates.
    NgxPassthroughArgs args = {};

    // Where the game expresses its transforms in a space that shifts every frame, the two
    // cameras below describe different spaces, and walking from one to the other without
    // accounting for the shift reprojects every pixel as though the scene had moved by it.
    // A point standing still sits at p in this frame's space and p - offset in the previous
    // one, so that step belongs between the two legs. The offset is zero for titles uploading
    // true world space, leaving the chain exactly as it was.
    Matrix4d previousSpaceFromCurrent = Matrix4d();
    previousSpaceFromCurrent[3] = Vector4d(-double(m_sceneTransformOffset.x),
                                           -double(m_sceneTransformOffset.y),
                                           -double(m_sceneTransformOffset.z), 1.0);

    // Without a camera of its own, the matrices below still describe the last scene and would
    // reproject a movie or loading screen by motion that predates it; nothing here moved.
    const Matrix4d reprojectToPrevClip = m_sceneCameraFresh
      ? camera.getPreviousViewToProjection() *
        camera.getPreviousWorldToView() *
        previousSpaceFromCurrent *
        camera.getViewToWorld() *
        camera.getProjectionToView()
      : Matrix4d();

    // How far this matrix is from identity is how much camera motion the reprojection can
    // express. Near zero while the view is moving means the previous camera is not actually a
    // previous camera, and every static pixel gets a null motion vector no matter what the
    // velocity raster does - a different fault to the raster missing objects, and one the two
    // are easily confused for on screen.
    {
      double reprojectionFromIdentity = 0.0;
      for (uint32_t col = 0; col < 4; col++) {
        for (uint32_t row = 0; row < 4; row++) {
          reprojectionFromIdentity +=
            std::abs(reprojectToPrevClip[col][row] - (col == row ? 1.0 : 0.0));
        }
      }

      m_statReprojectionFromIdentityMax = std::max(m_statReprojectionFromIdentityMax,
                                                   float(reprojectionFromIdentity));
    }

    args.reprojectToPrevClip = Matrix4(reprojectToPrevClip);
    args.resolution = vec2(float(m_renderExtent.width), float(m_renderExtent.height));
    args.subrectOffset = vec2(float(subrectOffset.x), float(subrectOffset.y));
    args.jitter = vec2(jitter[0], jitter[1]);
    args.debugMode = uint(std::clamp(debugVisualization(), 0, int(DebugVisualization::ObjectVelocityCoverage)));

    const auto nearFarPlanes = camera.calculateNearFarPlanes();
    args.nearPlane = nearFarPlanes.first;
    args.farPlane = nearFarPlanes.second;

    // Curved sample paths for the cinematic motion blur are fitted from the camera's own
    // pose history, so they are built here where the space offset is already known, and held
    // for the post effect stage further down the frame.
    m_motionBlurNearPlane = args.nearPlane;
    m_motionBlurFarPlane = args.farPlane;
    m_motionBlurCurves = m_sceneCameraFresh
      ? buildMotionBlurCurveMatrices(camera, previousSpaceFromCurrent)
      : DxvkPostFx::MotionBlurCurveMatrices();
    args.motionBlurFirstPerson = motionBlurFirstPerson() ? 1u : 0u;
    args.objectVelocityValid = (velocityTargets & 1u) != 0 ? 1u : 0u;
    args.foregroundVelocityValid = (velocityTargets & 2u) != 0 ? 1u : 0u;

    args.outputResolution = vec2(float(m_displayExtent.width), float(m_displayExtent.height));
    args.outputTransformEnabled = m_outputTransform.enabled ? 1u : 0u;
    args.preserveOriginalAlpha = m_preserveOriginalAlpha ? 1u : 0u;
    args.outputColorScaleAndGamma = vec4(m_outputTransform.colorScale[0],
                                         m_outputTransform.colorScale[1],
                                         m_outputTransform.colorScale[2],
                                         m_outputTransform.inverseGamma);
    args.outputOverlayColor = vec4(m_outputTransform.overlayColor[0],
                                   m_outputTransform.overlayColor[1],
                                   m_outputTransform.overlayColor[2],
                                   m_outputTransform.overlayColor[3]);

    ctx->updateBuffer(m_constantsBuffer, 0, sizeof(args), &args);

    const Resources::Resource& depthOutput = m_depthQueue.get();
    const Resources::Resource& motionVectorOutput = m_motionVectorQueue.get();

    // Depth inputs to make readable by the compute pass (both aspects for the
    // layout-preserving barrier: transitions on combined depth-stencil images
    // must cover depth and stencil)
    const std::array<const Rc<DxvkImage>*, 2> depthInputs = { &worldDepthImage, &foregroundDepthImage };
    const size_t depthInputCount = separateForeground ? 2 : 1;

    for (size_t i = 0; i < depthInputCount; i++) {
      const Rc<DxvkImage>& depthInput = *depthInputs[i];

      const VkImageSubresourceRange depthInputRange = {
        imageFormatInfo(depthInput->info().format)->aspectMask, 0, 1, 0, 1
      };

      barriers.accessImage(
        depthInput,
        depthInputRange,
        depthInput->info().layout,
        depthInput->info().stages,
        depthInput->info().access,
        depthInput->info().layout,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);
    }

    const std::array<const Resources::Resource*, 5> computeOutputs = {
      &depthOutput, &motionVectorOutput, &m_dlssOutput, &m_linearViewZ, &m_surfaceFlags
    };

    for (const Resources::Resource* output : computeOutputs) {
      barriers.accessImage(
        output->image,
        output->view->imageSubresources(),
        output->image->info().layout,
        output->image->info().stages,
        output->image->info().access,
        output->image->info().layout,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT);
    }

    barriers.recordCommands(ctx->getCommandList());

    {
      ScopedGpuProfileZone(ctx, "NGX Passthrough MV Generation");

      ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
      ctx->bindResourceView(1, worldDepthView, nullptr);
      ctx->bindResourceView(2, depthOutput.view, nullptr);
      ctx->bindResourceView(3, motionVectorOutput.view, nullptr);
      // Debug visualization target (written only when the debug mode is enabled)
      ctx->bindResourceView(4, m_dlssOutput.view, nullptr);
      ctx->bindResourceView(5, foregroundDepthView, nullptr);
      ctx->bindResourceView(8, m_linearViewZ.view, nullptr);
      ctx->bindResourceView(9, m_surfaceFlags.view, nullptr);
      ctx->bindResourceView(10, m_objectVelocity.view, nullptr);

      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, NgxPassthroughMvShader::getShader());

      const VkExtent3D workgroups = util::computeBlockCount(
        VkExtent3D { m_renderExtent.width, m_renderExtent.height, 1 }, VkExtent3D { 16, 16, 1 });
      ctx->dispatch(workgroups.width, workgroups.height, 1);
    }

    // Return the depth inputs to their declared usage and make the outputs visible to
    // subsequent consumers (NGX evaluation, DLFG present thread)
    for (size_t i = 0; i < depthInputCount; i++) {
      const Rc<DxvkImage>& depthInput = *depthInputs[i];

      const VkImageSubresourceRange depthInputRange = {
        imageFormatInfo(depthInput->info().format)->aspectMask, 0, 1, 0, 1
      };

      barriers.accessImage(
        depthInput,
        depthInputRange,
        depthInput->info().layout,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        depthInput->info().layout,
        depthInput->info().stages,
        depthInput->info().access);
    }

    for (const Resources::Resource* output : computeOutputs) {
      barriers.accessImage(
        output->image,
        output->view->imageSubresources(),
        output->image->info().layout,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        output->image->info().layout,
        output->image->info().stages,
        output->image->info().access);
    }

    barriers.recordCommands(ctx->getCommandList());

    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(worldDepthImage);
    ctx->getCommandList()->trackResource<DxvkAccess::None>(worldDepthView);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(foregroundDepthImage);
    ctx->getCommandList()->trackResource<DxvkAccess::None>(foregroundDepthView);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(depthOutput.image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(motionVectorOutput.image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_linearViewZ.image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_surfaceFlags.image);

    return true;
  }

#ifndef _M_ARM64
  void RtxNgxPassthrough::syncXeSSInputResolution(uint32_t displayWidth, uint32_t displayHeight) {
    if (!RtxOptions::isXeSSEnabled()) {
      return;
    }

    if (displayWidth == 0 || displayHeight == 0) {
      displayWidth = m_displayExtent.width;
      displayHeight = m_displayExtent.height;
    }

    if (displayWidth == 0 || displayHeight == 0) {
      return;
    }

    uint32_t displaySize[2] = { displayWidth, displayHeight };
    uint32_t renderSize[2] = { 0, 0 };
    m_device->getCommon()->metaXeSS().setSetting(displaySize, DxvkXeSS::XessOptions::preset(), renderSize);
  }

  void RtxNgxPassthrough::getXeSSInputResolution(uint32_t displayWidth, uint32_t displayHeight,
                                                 uint32_t& inputWidth, uint32_t& inputHeight) {
    syncXeSSInputResolution(displayWidth, displayHeight);
    m_device->getCommon()->metaXeSS().getInputSize(inputWidth, inputHeight);
  }
#else
  void RtxNgxPassthrough::syncXeSSInputResolution(uint32_t, uint32_t) {
  }

  void RtxNgxPassthrough::getXeSSInputResolution(uint32_t, uint32_t, uint32_t& inputWidth, uint32_t& inputHeight) {
    inputWidth = 0;
    inputHeight = 0;
  }
#endif

  float RtxNgxPassthrough::screenPercentageForDisplay(uint32_t displayWidth, uint32_t displayHeight) {
#ifndef _M_ARM64
    if (RtxOptions::isXeSSEnabled() && displayWidth > 0 && displayHeight > 0) {
      syncXeSSInputResolution(displayWidth, displayHeight);
      const DxvkXeSS& xess = m_device->getCommon()->metaXeSS();
      uint32_t inputWidth = 0;
      uint32_t inputHeight = 0;
      xess.getInputSize(inputWidth, inputHeight);
      if (inputWidth > 0 && inputHeight > 0) {
        return 100.0f * float(inputWidth) / float(displayWidth);
      }
    }
#endif
    return screenPercentageForUpscaler(displayWidth, displayHeight);
  }

  float RtxNgxPassthrough::screenPercentageForUpscaler(uint32_t displayWidth, uint32_t displayHeight) {
    switch (RtxOptions::upscalerType()) {
    case UpscalerType::DLSS: {
      DLSSProfile profile = RtxOptions::qualityDLSS();
      if (profile == DLSSProfile::Auto) {
        if (displayHeight == 0 || displayHeight <= 1080) {
          profile = DLSSProfile::MaxQuality;
        } else if (displayHeight < 2160) {
          profile = DLSSProfile::Balanced;
        } else if (displayHeight < 4320) {
          profile = DLSSProfile::MaxPerf;
        } else {
          profile = DLSSProfile::UltraPerf;
        }
      }

      switch (profile) {
      case DLSSProfile::UltraPerf:      return 100.0f / 3.0f;
      case DLSSProfile::MaxPerf:        return 50.0f;
      case DLSSProfile::Balanced:       return 58.0f;
      case DLSSProfile::MaxQuality:     return 200.0f / 3.0f;
      case DLSSProfile::FullResolution: return 100.0f;
      default:                          return 100.0f;
      }
    }
    case UpscalerType::NIS:
      switch (RtxOptions::nisPreset()) {
      case NisPreset::Performance: return 50.0f;
      case NisPreset::Balanced:     return 66.0f;
      case NisPreset::Quality:      return 75.0f;
      default:                      return 100.0f;
      }
    case UpscalerType::TAAU:
      switch (RtxOptions::taauPreset()) {
      case TaauPreset::UltraPerformance: return 100.0f / 3.0f;
      case TaauPreset::Performance:      return 50.0f;
      case TaauPreset::Balanced:         return 66.0f;
      case TaauPreset::Quality:          return 75.0f;
      default:                           return 100.0f;
      }
#ifndef _M_ARM64
    case UpscalerType::XeSS:
      return DxvkXeSS::calcScreenPercentageForPreset(DxvkXeSS::XessOptions::preset());
#endif
    default:
      return 100.0f;
    }
  }

  const char* RtxNgxPassthrough::upscalerModeLabel() {
    switch (RtxOptions::upscalerType()) {
    case UpscalerType::DLSS: return "DLSS";
    case UpscalerType::NIS:  return "NIS";
    case UpscalerType::TAAU: return "TAA-U";
    case UpscalerType::XeSS: return "XeSS";
    default:                 return "None";
    }
  }

  bool RtxNgxPassthrough::needsViewportJitter(DxvkDevice* device) {
    // A plain stretch has no history to accumulate phases into, so jitter would only wobble the
    // image and leave the game's alpha a sub-pixel away from the colour the merge pairs it with
    if (bypassUpscaler()) {
      return false;
    }

    switch (RtxOptions::upscalerType()) {
    case UpscalerType::DLSS:
      return device != nullptr && device->getCommon()->metaNGXContext().supportsDLSS();
    case UpscalerType::TAAU:
    case UpscalerType::XeSS:
      return true;
    default:
      return false;
    }
  }

  // Upscale ratio above which the derived jitter phase count has to be shortened, and what to
  // shorten it to. Measured in game: at 2x the derived 32 phases are clean, at 3x the derived 72
  // are not, nor are 16 - only 8 is. Nothing between 2x and 3x has been tested because the
  // standard presets skip from one to the other, hence the midpoint.
  static constexpr float kNgxJitterSequenceClampRatio = 2.5f;
  static constexpr uint32_t kNgxClampedJitterSequenceLength = 8;

  uint32_t RtxNgxPassthrough::viewportJitterSequenceLength(DxvkDevice* device,
                                                           uint32_t renderHeight,
                                                           uint32_t displayHeight) {
#ifndef _M_ARM64
    if (RtxOptions::isXeSSEnabled() && device != nullptr &&
        DxvkXeSS::XessOptions::useRecommendedJitterSequenceLength()) {
      return device->getCommon()->metaXeSS().calcRecommendedJitterSequenceLength();
    }
#endif

    // An explicit override wins, so the loop length can actually be bisected against an artefact.
    if (jitterSequenceLength() > 0) {
      return uint32_t(jitterSequenceLength());
    }

    // 8 * ratio^2 covers the upscaled sample grid - the rule RtCamera::updateResolution applies and
    // what DLSS documents. That guidance assumes the jitter sits in the projection, where it reaches
    // every scene sample; this mode offsets the viewport instead, and the upscaler still removes the
    // offset from the whole frame, so whatever the offset missed is displaced rather than corrected.
    // The displacement is proportional to the offset and scaled into display pixels by the ratio, so
    // the sequence's deepest offsets become visible - as a shift with the period of the loop, once
    // the loop is long enough that they no longer recur inside the accumulation window. Both
    // conditions have to hold, which is why the derivation only needs shortening at high ratios.
    if (renderHeight != 0 && displayHeight > renderHeight) {
      const float ratio = float(displayHeight) / float(renderHeight);

      if (ratio > kNgxJitterSequenceClampRatio) {
        return kNgxClampedJitterSequenceLength;
      }

      return std::max(uint32_t(8.0f * ratio * ratio), 8u);
    }

    return RtxOptions::cameraJitterSequenceLength();
  }

  namespace {
    // Maps an actual render/display resolution ratio onto the NGX quality value whose
    // standard scaling factor is closest (the game dictates the render resolution through
    // its ScreenPercentage, so the quality enum just describes the ratio to DLSS).
    NVSDK_NGX_PerfQuality_Value perfQualityFromResolutionRatio(float ratio) {
      if (ratio >= 0.995f) {
        return NVSDK_NGX_PerfQuality_Value_DLAA;
      } else if (ratio >= 0.62f) {
        return NVSDK_NGX_PerfQuality_Value_MaxQuality;    // 0.667
      } else if (ratio >= 0.54f) {
        return NVSDK_NGX_PerfQuality_Value_Balanced;      // 0.58
      } else if (ratio >= 0.42f) {
        return NVSDK_NGX_PerfQuality_Value_MaxPerf;       // 0.5
      } else {
        return NVSDK_NGX_PerfQuality_Value_UltraPerformance; // 0.33
      }
    }

    NVSDK_NGX_DLSS_Hint_Render_Preset renderPresetFromOption(int optionValue) {
      switch (optionValue) {
      case 1: return NVSDK_NGX_DLSS_Hint_Render_Preset_A;
      case 2: return NVSDK_NGX_DLSS_Hint_Render_Preset_B;
      case 3: return NVSDK_NGX_DLSS_Hint_Render_Preset_C;
      case 4: return NVSDK_NGX_DLSS_Hint_Render_Preset_D;
      case 5: return NVSDK_NGX_DLSS_Hint_Render_Preset_E;
      case 6: return NVSDK_NGX_DLSS_Hint_Render_Preset_F;
      case 10: return NVSDK_NGX_DLSS_Hint_Render_Preset_J;
      default: return NVSDK_NGX_DLSS_Hint_Render_Preset_Default;
      }
    }

    const char* renderPresetToString(NVSDK_NGX_DLSS_Hint_Render_Preset preset) {
      switch (preset) {
      case NVSDK_NGX_DLSS_Hint_Render_Preset_A: return "A";
      case NVSDK_NGX_DLSS_Hint_Render_Preset_B: return "B";
      case NVSDK_NGX_DLSS_Hint_Render_Preset_C: return "C";
      case NVSDK_NGX_DLSS_Hint_Render_Preset_D: return "D";
      case NVSDK_NGX_DLSS_Hint_Render_Preset_E: return "E";
      case NVSDK_NGX_DLSS_Hint_Render_Preset_F: return "F";
      case NVSDK_NGX_DLSS_Hint_Render_Preset_J: return "J (transformer)";
      default: return "Default";
      }
    }

    // At the pre-post-process injection point the color source is the game's scene color,
    // which is a linear (pre-tonemap) target; a floating point format means HDR content
    // for the DLSS feature. The late injection point sees display-encoded UNORM output.
    bool isHDRColorFormat(VkFormat format) {
      switch (format) {
      case VK_FORMAT_R16G16B16A16_SFLOAT:
      case VK_FORMAT_R32G32B32A32_SFLOAT:
      case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
        return true;
      default:
        return false;
      }
    }

    // Blit an image 1:1 into a rect of another image (used for the DLSS output and debug
    // visualization write-back, which target a subrect of the game's scene color at the
    // pre-post-process injection point)
    // A source rect at the origin into an arbitrary destination rect. Equal extents with a
    // nearest filter deliver a result unaltered; differing extents stretch.
    void blitImageRect(DxvkContext* ctx,
                       const Rc<DxvkImage>& sourceImage, const VkExtent2D& sourceExtent,
                       const Rc<DxvkImage>& targetImage, const VkOffset2D& targetOffset,
                       const VkExtent2D& targetExtent, VkFilter filter) {
      const DxvkFormatInfo* srcFormatInfo = imageFormatInfo(sourceImage->info().format);
      const DxvkFormatInfo* dstFormatInfo = imageFormatInfo(targetImage->info().format);

      VkImageBlit blitInfo = {};
      blitInfo.srcSubresource = { srcFormatInfo->aspectMask, 0, 0, 1 };
      blitInfo.dstSubresource = { dstFormatInfo->aspectMask, 0, 0, 1 };
      blitInfo.srcOffsets[0] = { 0, 0, 0 };
      blitInfo.srcOffsets[1] = { int32_t(sourceExtent.width), int32_t(sourceExtent.height), 1 };
      blitInfo.dstOffsets[0] = { targetOffset.x, targetOffset.y, 0 };
      blitInfo.dstOffsets[1] = { targetOffset.x + int32_t(targetExtent.width),
                                 targetOffset.y + int32_t(targetExtent.height), 1 };

      const VkComponentMapping identityMap = {
        VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
      };

      ctx->blitImage(targetImage, identityMap, sourceImage, identityMap, blitInfo, filter);
    }

    void blitToTargetRect(DxvkContext* ctx,
                          const Rc<DxvkImage>& sourceImage, const VkExtent2D& sourceExtent,
                          const Rc<DxvkImage>& targetImage, const VkOffset2D& targetOffset) {
      blitImageRect(ctx, sourceImage, sourceExtent, targetImage, targetOffset, sourceExtent,
                    VK_FILTER_NEAREST);
    }
  }

  void RtxNgxPassthrough::dispatchAlphaMerge(RtxContext* ctx, DxvkBarrierSet& barriers) {
    ScopedGpuProfileZone(ctx, "NGX Passthrough Alpha Merge");

    const std::array<const Resources::Resource*, 2> mergeInputs = { &m_dlssOutput, &m_colorInput };

    for (const Resources::Resource* input : mergeInputs) {
      barriers.accessImage(
        input->image,
        input->view->imageSubresources(),
        input->image->info().layout,
        input->image->info().stages,
        input->image->info().access,
        input->image->info().layout,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);
    }

    barriers.accessImage(
      m_mergedOutput.image,
      m_mergedOutput.view->imageSubresources(),
      m_mergedOutput.image->info().layout,
      m_mergedOutput.image->info().stages,
      m_mergedOutput.image->info().access,
      m_mergedOutput.image->info().layout,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT);

    barriers.recordCommands(ctx->getCommandList());

    ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
    ctx->bindResourceView(1, m_dlssOutput.view, nullptr);
    ctx->bindResourceView(2, m_colorInput.view, nullptr);
    ctx->bindResourceView(3, m_mergedOutput.view, nullptr);

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, NgxPassthroughAlphaMergeShader::getShader());

    const VkExtent3D workgroups = util::computeBlockCount(
      VkExtent3D { m_displayExtent.width, m_displayExtent.height, 1 }, VkExtent3D { 16, 16, 1 });
    ctx->dispatch(workgroups.width, workgroups.height, 1);

    for (const Resources::Resource* input : mergeInputs) {
      barriers.accessImage(
        input->image,
        input->view->imageSubresources(),
        input->image->info().layout,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        input->image->info().layout,
        input->image->info().stages,
        input->image->info().access);
    }

    barriers.accessImage(
      m_mergedOutput.image,
      m_mergedOutput.view->imageSubresources(),
      m_mergedOutput.image->info().layout,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT,
      m_mergedOutput.image->info().layout,
      m_mergedOutput.image->info().stages,
      m_mergedOutput.image->info().access);

    barriers.recordCommands(ctx->getCommandList());

    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_dlssOutput.image);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_colorInput.image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_mergedOutput.image);
  }

  void RtxNgxPassthrough::snapshotColorInput(RtxContext* ctx,
                                             const Rc<DxvkImage>& colorSourceImage,
                                             const VkOffset2D& colorSourceOffset) {
    const DxvkFormatInfo* srcFormatInfo = imageFormatInfo(colorSourceImage->info().format);
    const DxvkFormatInfo* dstFormatInfo = imageFormatInfo(m_colorInput.image->info().format);

    VkImageBlit blitInfo = {};
    blitInfo.srcSubresource = { srcFormatInfo->aspectMask, 0, 0, 1 };
    blitInfo.dstSubresource = { dstFormatInfo->aspectMask, 0, 0, 1 };
    blitInfo.srcOffsets[0] = { colorSourceOffset.x, colorSourceOffset.y, 0 };
    blitInfo.srcOffsets[1] = { colorSourceOffset.x + int32_t(m_renderExtent.width),
                               colorSourceOffset.y + int32_t(m_renderExtent.height), 1 };
    blitInfo.dstOffsets[0] = { 0, 0, 0 };
    blitInfo.dstOffsets[1] = { int32_t(m_renderExtent.width), int32_t(m_renderExtent.height), 1 };

    const VkComponentMapping identityMap = {
      VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
      VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
    };

    ctx->blitImage(m_colorInput.image, identityMap, colorSourceImage, identityMap, blitInfo, VK_FILTER_NEAREST);
  }

  void RtxNgxPassthrough::blitDebugOverlayToPresent(DxvkContext* ctx, const Rc<DxvkImage>& targetImage) {
    if (debugVisualization() == int(DebugVisualization::Off)) {
      m_debugPresentValid = false;
      return;
    }

    // Held rather than consumed. A frame that produces no depth/MV inputs - a rejected camera, a
    // frame with no scene - captures no view, and clearing here would let the game's colour image
    // through on those frames. That reads as the upscaler dropping out when nothing of the sort
    // happened, so the last captured view stays up until a new one replaces it.
    if (!m_debugPresentValid || ctx == nullptr || targetImage == nullptr ||
        m_debugPresentImage.image == nullptr) {
      return;
    }

    const DxvkFormatInfo* srcFormatInfo = imageFormatInfo(m_debugPresentImage.image->info().format);
    const DxvkFormatInfo* dstFormatInfo = imageFormatInfo(targetImage->info().format);
    const VkExtent3D& targetExtent = targetImage->info().extent;

    VkImageBlit blitInfo = {};
    blitInfo.srcSubresource = { srcFormatInfo->aspectMask, 0, 0, 1 };
    blitInfo.dstSubresource = { dstFormatInfo->aspectMask, 0, 0, 1 };
    blitInfo.srcOffsets[0] = { 0, 0, 0 };
    blitInfo.srcOffsets[1] = { int32_t(m_renderExtent.width), int32_t(m_renderExtent.height), 1 };
    blitInfo.dstOffsets[0] = { 0, 0, 0 };
    blitInfo.dstOffsets[1] = { int32_t(targetExtent.width), int32_t(targetExtent.height), 1 };

    const VkComponentMapping identityMap = {
      VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
      VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
    };

    // Nearest keeps the visualized input pixels inspectable rather than smearing them
    ctx->blitImage(targetImage, identityMap, m_debugPresentImage.image, identityMap, blitInfo, VK_FILTER_NEAREST);
  }

  void RtxNgxPassthrough::copyColorInputToOutput(RtxContext* ctx) {
    const DxvkFormatInfo* srcFormatInfo = imageFormatInfo(m_colorInput.image->info().format);
    const DxvkFormatInfo* dstFormatInfo = imageFormatInfo(m_dlssOutput.image->info().format);

    const VkFilter filter = (m_renderExtent.width != m_displayExtent.width ||
                             m_renderExtent.height != m_displayExtent.height)
      ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;

    VkImageBlit blitInfo = {};
    blitInfo.srcSubresource = { srcFormatInfo->aspectMask, 0, 0, 1 };
    blitInfo.dstSubresource = { dstFormatInfo->aspectMask, 0, 0, 1 };
    blitInfo.srcOffsets[0] = { 0, 0, 0 };
    blitInfo.srcOffsets[1] = { int32_t(m_renderExtent.width), int32_t(m_renderExtent.height), 1 };
    blitInfo.dstOffsets[0] = { 0, 0, 0 };
    blitInfo.dstOffsets[1] = { int32_t(m_displayExtent.width), int32_t(m_displayExtent.height), 1 };

    const VkComponentMapping identityMap = {
      VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
      VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
    };

    ctx->blitImage(m_dlssOutput.image, identityMap, m_colorInput.image, identityMap, blitInfo, filter);
  }

  bool RtxNgxPassthrough::evaluatePostFxOnly(RtxContext* ctx,
                                             DxvkBarrierSet& barriers,
                                             const Rc<DxvkImage>& colorSourceImage,
                                             const VkOffset2D& colorSourceOffset,
                                             const Rc<DxvkImage>& targetImage,
                                             const VkOffset2D& targetOffset,
                                             bool preserveTargetAlpha,
                                             bool resetHistory) {
    if (!m_device->getCommon()->metaPostFx().isPostFxEnabled()) {
      return false;
    }

    snapshotColorInput(ctx, colorSourceImage, colorSourceOffset);
    copyColorInputToOutput(ctx);

    return applyPostFxAndWriteback(ctx, barriers, targetImage, targetOffset, preserveTargetAlpha, resetHistory);
  }

  bool RtxNgxPassthrough::applyPostFxAndWriteback(RtxContext* ctx,
                                                  DxvkBarrierSet& barriers,
                                                  const Rc<DxvkImage>& targetImage,
                                                  const VkOffset2D& targetOffset,
                                                  bool preserveTargetAlpha,
                                                  bool resetHistory) {
    {
      DxvkPostFx& postFx = m_device->getCommon()->metaPostFx();

      if (postFx.isPostFxEnabled()) {
        ONCE(Logger::info("[RTX NGX Passthrough] Remix post effects (rtx.postfx) active on the anti-aliased output."));

        const uvec2 renderResolution = { m_renderExtent.width, m_renderExtent.height };
        const uint32_t frameIdx = RtxOptions::rngSeedWithFrameIndex() ? m_device->getCurrentFrameId() : 0;

        Rc<DxvkSampler> nearestSampler =
          ctx->getResourceManager().getSampler(VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        Rc<DxvkSampler> linearSampler =
          ctx->getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

        DxvkPostFx::MotionBlurInputs motionBlurInputs = {};
        motionBlurInputs.inOutColor = &m_dlssOutput;
        motionBlurInputs.intermediateColor = &m_mergedOutput;
        motionBlurInputs.screenSpaceMotionVector = &m_motionVectorQueue.get();
        motionBlurInputs.surfaceFlags = &m_surfaceFlags;
        motionBlurInputs.surfaceFlagsScratch1 = &m_surfaceFlagsScratch1;
        motionBlurInputs.surfaceFlagsScratch2 = &m_surfaceFlagsScratch2;
        motionBlurInputs.linearViewZ = &m_linearViewZ;
        motionBlurInputs.cineVelocityDepth = &m_motionBlurCineVelocityDepth;
        motionBlurInputs.cineCurvature = &m_motionBlurCineCurvature;
        motionBlurInputs.cineTileMaxX = &m_motionBlurCineTileMaxX;
        motionBlurInputs.cineTileMax = &m_motionBlurCineTileMax;
        motionBlurInputs.cineNeighborMax = &m_motionBlurCineNeighborMax;
        motionBlurInputs.previousScreenSpaceMotionVector =
          m_motionVectorQueue.hasDistinctPrevious() ? &m_motionVectorQueue.getPrevious() : nullptr;
        motionBlurInputs.curves = m_motionBlurCurves;
        motionBlurInputs.nearPlane = m_motionBlurNearPlane;
        motionBlurInputs.farPlane = m_motionBlurFarPlane;

        postFx.dispatchMotionBlur(ctx, nearestSampler, linearSampler, renderResolution, frameIdx,
                                  motionBlurInputs, resetHistory);

        postFx.dispatchLensEffects(ctx, linearSampler, renderResolution, frameIdx,
                                   m_dlssOutput, m_mergedOutput);
      }
    }

    // The merge pass carries the game's alpha through and reapplies the suppressed composite's
    // transform; when neither is needed the upscaled result goes straight to the target.
    if (preserveTargetAlpha || m_outputTransform.enabled) {
      dispatchAlphaMerge(ctx, barriers);
      blitToTargetRect(ctx, m_mergedOutput.image, m_displayExtent, targetImage, targetOffset);
    } else {
      blitToTargetRect(ctx, m_dlssOutput.image, m_displayExtent, targetImage, targetOffset);
    }

    return true;
  }

  bool RtxNgxPassthrough::evaluateDlss(RtxContext* ctx,
                                       DxvkBarrierSet& barriers,
                                       const Rc<DxvkImage>& colorSourceImage,
                                       const VkOffset2D& colorSourceOffset,
                                       const Rc<DxvkImage>& targetImage,
                                       const VkOffset2D& targetOffset,
                                       bool preserveTargetAlpha,
                                       const float jitter[2],
                                       bool resetHistory) {
    NGXContext& ngxContext = m_device->getCommon()->metaNGXContext();

    if (!ngxContext.supportsDLSS()) {
      m_statusReason = "DLSS not supported on this system";
      ONCE(Logger::warn("[RTX NGX Passthrough] DLSS is not supported on this system."));
      return false;
    }

    // Snapshot the scene color subrect as the DLSS input (the DLSS output may be written
    // back over the same target, so it cannot be read in place)
    snapshotColorInput(ctx, colorSourceImage, colorSourceOffset);

    // HDR content flag follows the color source: the pre-post-process injection point feeds
    // the game's linear scene color (float target), the late one display-encoded LDR output
    const bool contentHDR = isHDRColorFormat(colorSourceImage->info().format);

    // The two injection points alternate freely - whether the pre-post trigger fires depends on
    // the frame's own post chain and on the mid-scene depth clear, both of which follow scene
    // content. They need different input sizes and different HDR flags, so one shared feature
    // meant every switch paid a device idle, a feature rebuild and a full history reset: several
    // frames of reconvergence, repeating for as long as the scene keeps flipping. Each injection
    // point keeps its own feature and its own temporal history instead.
    DlssFeature& feature = m_dlssFeatures[contentHDR ? 1 : 0];

    if (!feature.context) {
      feature.context = ngxContext.createDLSSContext();
      feature.needsInitialize = true;
    }

    if (feature.initializedRenderPreset != dlssRenderPreset() ||
        feature.initializedInput[0] != m_renderExtent.width ||
        feature.initializedInput[1] != m_renderExtent.height ||
        feature.initializedOutput[0] != m_displayExtent.width ||
        feature.initializedOutput[1] != m_displayExtent.height) {
      feature.needsInitialize = true;
    }

    if (feature.needsInitialize) {
      feature.needsInitialize = false;
      feature.initializedRenderPreset = dlssRenderPreset();
      feature.initializedInput[0] = m_renderExtent.width;
      feature.initializedInput[1] = m_renderExtent.height;
      feature.initializedOutput[0] = m_displayExtent.width;
      feature.initializedOutput[1] = m_displayExtent.height;

      // The previous feature may still be in flight
      m_device->waitForIdle();
      feature.context->releaseNGXFeature();

      uint32_t inputSize[2] = { m_renderExtent.width, m_renderExtent.height };
      uint32_t outputSize[2] = { m_displayExtent.width, m_displayExtent.height };

      const float resolutionRatio = float(inputSize[0]) / float(std::max(outputSize[0], 1u));
      const NVSDK_NGX_PerfQuality_Value perfQuality = perfQualityFromResolutionRatio(resolutionRatio);
      const NVSDK_NGX_DLSS_Hint_Render_Preset renderPreset = renderPresetFromOption(feature.initializedRenderPreset);

      // Conventional (non-inverted) depth, NGX-internal auto exposure (Remix's auto exposure
      // pass never runs in this mode; for LDR content the exposure is ignored anyway).
      feature.context->initialize(ctx, inputSize, outputSize,
                                  contentHDR,
                                  /* depthInverted = */ false,
                                  /* autoExposure = */ true,
                                  /* sharpening = */ false,
                                  perfQuality,
                                  renderPreset);

      m_dlssInitCount++;

      Logger::info(str::format("[RTX NGX Passthrough] DLSS initialized: ", inputSize[0], "x", inputSize[1],
                               " -> ", outputSize[0], "x", outputSize[1],
                               (perfQuality == NVSDK_NGX_PerfQuality_Value_DLAA ? " (DLAA)" : " (Super Resolution)"),
                               ", render preset ", renderPresetToString(renderPreset),
                               (contentHDR ? ", linear HDR input" : ", LDR input")));
    }

    if (m_dlssLastContentHDR != int(contentHDR)) {
      m_dlssLastContentHDR = int(contentHDR);
      m_statContentHdrFlips++;
    }

    // The DLSS indicator reads the exposure texture even with NGX auto exposure enabled, so
    // make sure it exists (its content is not used otherwise in this mode)
    DxvkAutoExposure& autoExposure = m_device->getCommon()->metaAutoExposure();
    autoExposure.createResources(ctx);

    const Resources::Resource& depthInput = m_depthQueue.get();
    const Resources::Resource& motionVectorInput = m_motionVectorQueue.get();

    const std::array<const Resources::Resource*, 4> inputs = {
      &m_colorInput, &motionVectorInput, &depthInput, &autoExposure.getExposureTexture()
    };

    for (const Resources::Resource* input : inputs) {
      if (input->image == nullptr) {
        continue;
      }

      barriers.accessImage(
        input->image,
        input->view->imageSubresources(),
        input->image->info().layout,
        input->image->info().stages,
        input->image->info().access,
        input->image->info().layout,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);
    }

    barriers.accessImage(
      m_dlssOutput.image,
      m_dlssOutput.view->imageSubresources(),
      m_dlssOutput.image->info().layout,
      m_dlssOutput.image->info().stages,
      m_dlssOutput.image->info().access,
      m_dlssOutput.image->info().layout,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT);

    barriers.recordCommands(ctx->getCommandList());

    NGXDLSSContext::NGXBuffers buffers = {};
    buffers.pUnresolvedColor = &m_colorInput;
    buffers.pResolvedColor = &m_dlssOutput;
    buffers.pMotionVectors = &motionVectorInput;
    buffers.pDepth = &depthInput;
    buffers.pExposure = &autoExposure.getExposureTexture();
    buffers.pBiasCurrentColorMask = nullptr;

    NGXDLSSContext::NGXSettings settings = {};
    settings.resetAccumulation = resetHistory;
    settings.antiGhost = false;
    settings.preExposure = 1.0f;
    settings.jitterOffset[0] = jitter[0];
    settings.jitterOffset[1] = jitter[1];
    settings.motionVectorScale[0] = 1.0f;
    settings.motionVectorScale[1] = 1.0f;

    bool success;

    {
      ScopedGpuProfileZone(ctx, "DLSS (NGX Passthrough)");
      success = feature.context->evaluateDLSS(ctx, buffers, settings);
    }

    barriers.accessImage(
      m_dlssOutput.image,
      m_dlssOutput.view->imageSubresources(),
      m_dlssOutput.image->info().layout,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT,
      m_dlssOutput.image->info().layout,
      m_dlssOutput.image->info().stages,
      m_dlssOutput.image->info().access);

    barriers.recordCommands(ctx->getCommandList());

    for (const Resources::Resource* input : inputs) {
      if (input->image != nullptr) {
        ctx->getCommandList()->trackResource<DxvkAccess::Read>(input->image);
      }
    }
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_dlssOutput.image);

    if (!success) {
      m_statusReason = "DLSS evaluation failed";
      ONCE(Logger::err("[RTX NGX Passthrough] DLSS evaluation failed; presenting the game's unmodified output."));
      return false;
    }

    // Remix post-processing and write-back (shared with the other passthrough upscalers).
    return applyPostFxAndWriteback(ctx, barriers, targetImage, targetOffset, preserveTargetAlpha, resetHistory);
  }

  bool RtxNgxPassthrough::evaluateNis(RtxContext* ctx,
                                      DxvkBarrierSet& barriers,
                                      const Rc<DxvkImage>& colorSourceImage,
                                      const VkOffset2D& colorSourceOffset,
                                      const Rc<DxvkImage>& targetImage,
                                      const VkOffset2D& targetOffset,
                                      bool preserveTargetAlpha,
                                      bool resetHistory) {
    snapshotColorInput(ctx, colorSourceImage, colorSourceOffset);

    {
      ScopedGpuProfileZone(ctx, "NIS (NGX Passthrough)");
      m_device->getCommon()->metaNIS().dispatch(ctx, m_colorInput, m_dlssOutput);
    }

    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_colorInput.image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_dlssOutput.image);

    return applyPostFxAndWriteback(ctx, barriers, targetImage, targetOffset, preserveTargetAlpha, resetHistory);
  }

  bool RtxNgxPassthrough::evaluateUpscalerBypass(RtxContext* ctx,
                                                 DxvkBarrierSet& barriers,
                                                 const Rc<DxvkImage>& colorSourceImage,
                                                 const VkOffset2D& colorSourceOffset,
                                                 const Rc<DxvkImage>& targetImage,
                                                 const VkOffset2D& targetOffset,
                                                 bool preserveTargetAlpha,
                                                 bool resetHistory) {
    snapshotColorInput(ctx, colorSourceImage, colorSourceOffset);

    // Stands in for the upscaler at the point it would have written, taking the same input and
    // leaving the same output for everything downstream, so the presented frame differs from the
    // upscaled one by the reconstruction alone.
    {
      ScopedGpuProfileZone(ctx, "Upscaler Bypass (NGX Passthrough)");
      blitImageRect(ctx, m_colorInput.image, m_renderExtent, m_dlssOutput.image, { 0, 0 },
                    m_displayExtent, VK_FILTER_LINEAR);
    }

    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_colorInput.image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_dlssOutput.image);

    return applyPostFxAndWriteback(ctx, barriers, targetImage, targetOffset, preserveTargetAlpha, resetHistory);
  }

  bool RtxNgxPassthrough::evaluateTaau(RtxContext* ctx,
                                       DxvkBarrierSet& barriers,
                                       const Rc<DxvkImage>& colorSourceImage,
                                       const VkOffset2D& colorSourceOffset,
                                       const Rc<DxvkImage>& targetImage,
                                       const VkOffset2D& targetOffset,
                                       bool preserveTargetAlpha,
                                       const float jitter[2],
                                       bool resetHistory) {
    DxvkTemporalAA& taa = m_device->getCommon()->metaTAA();
    if (!RtxOptions::isTAAEnabled()) {
      m_statusReason = "rtx.upscalerType is not TAA-U";
      return false;
    }

    snapshotColorInput(ctx, colorSourceImage, colorSourceOffset);

    const VkExtent3D outputExtent = { m_displayExtent.width, m_displayExtent.height, 1 };
    Rc<DxvkContext> dxvkCtx = ctx;
    taa.ensureResources(dxvkCtx, outputExtent);

    Rc<DxvkSampler> linearSampler =
      ctx->getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    const uvec2 renderResolution = { m_renderExtent.width, m_renderExtent.height };

    {
      ScopedGpuProfileZone(ctx, "TAA-U (NGX Passthrough)");
      taa.dispatch(ctx,
                   linearSampler,
                   renderResolution,
                   jitter,
                   m_colorInput,
                   m_motionVectorQueue.get(),
                   m_dlssOutput,
                   true);
    }

    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_colorInput.image);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_motionVectorQueue.get().image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_dlssOutput.image);

    return applyPostFxAndWriteback(ctx, barriers, targetImage, targetOffset, preserveTargetAlpha, resetHistory);
  }

#ifndef _M_ARM64
  bool RtxNgxPassthrough::evaluateXess(RtxContext* ctx,
                                       DxvkBarrierSet& barriers,
                                       const Rc<DxvkImage>& colorSourceImage,
                                       const VkOffset2D& colorSourceOffset,
                                       const Rc<DxvkImage>& targetImage,
                                       const VkOffset2D& targetOffset,
                                       bool preserveTargetAlpha,
                                       bool resetHistory) {
    DxvkXeSS& xess = m_device->getCommon()->metaXeSS();

    Rc<DxvkContext> dxvkCtx = ctx;
    const VkExtent3D renderExtent3D = { m_renderExtent.width, m_renderExtent.height, 1 };
    const VkExtent3D displayExtent3D = { m_displayExtent.width, m_displayExtent.height, 1 };
    xess.beginPassthroughFrame(dxvkCtx, renderExtent3D, displayExtent3D, resetHistory,
                               ctx->getSceneManager().getCamera().isCameraCut());

    if (!xess.isActive()) {
      m_statusReason = "XeSS activation failed or is not supported on this system";
      ONCE(Logger::warn("[RTX NGX Passthrough] XeSS is not active (activation failed or unsupported)."));
      return false;
    }

    snapshotColorInput(ctx, colorSourceImage, colorSourceOffset);

    uint32_t displaySize[2] = { m_displayExtent.width, m_displayExtent.height };
    uint32_t renderSize[2] = { 0, 0 };
    xess.setSetting(displaySize, DxvkXeSS::XessOptions::preset(), renderSize);

    {
      ScopedGpuProfileZone(ctx, "XeSS (NGX Passthrough)");
      xess.dispatch(ctx,
                    barriers,
                    m_colorInput,
                    m_motionVectorQueue.get(),
                    m_depthQueue.get(),
                    m_dlssOutput,
                    resetHistory);
    }

    return applyPostFxAndWriteback(ctx, barriers, targetImage, targetOffset, preserveTargetAlpha, resetHistory);
  }
#endif

  void RtxNgxPassthrough::dispatch(RtxContext* ctx,
                                   DxvkContextState& dxvkCtxState,
                                   DxvkBarrierSet& barriers,
                                   const Rc<DxvkImage>& targetImage,
                                   const Rc<DxvkImage>& mirrorTargetImage,
                                   const Rc<DxvkImage>& sceneDepthImage,
                                   const Rc<DxvkImage>& upscaleSourceImage,
                                   const VkRect2D& sourceSubrect,
                                   const VkOffset2D& colorSourceOffset,
                                   const NgxOutputTransform& outputTransform,
                                   bool prePostProcess,
                                   const std::vector<NgxVelocityDraw>& velocityDraws,
                                   const float jitter[2],
                                   bool resetHistory) {
    ScopedGpuProfileZone(ctx, "NGX Passthrough");

    m_lastDispatchFrameId = m_device->getCurrentFrameId();
    m_lastJitter[0] = jitter[0];
    m_lastJitter[1] = jitter[1];
    m_lastDispatchPrePost = prePostProcess;
    m_postFxActive = false;

    const bool dlfgSupported = m_device->getCommon()->metaNGXContext().supportsDLFG();

    // Advance the DLFG input queues once per frame so the present thread can still read the
    // previous frames' depth/motion vectors while this frame writes new ones
    if (dlfgSupported) {
      m_depthQueue.next();
      m_motionVectorQueue.next();
      m_hudlessQueue.next();
    }

    // Super Resolution path: the game rendered its scene into a reduced subrect and the
    // engine's stretch onto the target was suppressed; DLSS consumes the subrect and outputs
    // at the target resolution. DLAA path: everything happens at the target resolution.
    // Pre-post-process path: the target is the game's scene color and the subrect is the
    // scene viewport rect within it; DLAA runs inside that rect (render == display size).
    const bool superResolution = !prePostProcess && upscaleSourceImage != nullptr &&
                                 sourceSubrect.extent.width > 0 && sourceSubrect.extent.height > 0;

    const bool prePostSubrectValid = prePostProcess &&
                                     sourceSubrect.extent.width > 0 && sourceSubrect.extent.height > 0;

    const VkExtent2D targetExtent = { targetImage->info().extent.width, targetImage->info().extent.height };
    const VkExtent2D displayExtent = prePostSubrectValid ? sourceSubrect.extent : targetExtent;
    const VkExtent2D renderExtent = superResolution ? sourceSubrect.extent : displayExtent;
    const VkOffset2D subrectOffset = (superResolution || prePostSubrectValid) ? sourceSubrect.offset : VkOffset2D { 0, 0 };
    const Rc<DxvkImage>& colorSourceImage = superResolution ? upscaleSourceImage : targetImage;
    // Depth always reads at subrectOffset; colour may live elsewhere in its own image.
    const VkOffset2D colorOffset = superResolution ? colorSourceOffset : subrectOffset;

    // Latched before the constant buffer is filled in generateMotionVectorsAndDepth. The engine
    // composite only exists on the Super Resolution path; the pre-post injection leaves the
    // game's own chain to run afterwards, so nothing has been replaced there.
    m_outputTransform = superResolution ? outputTransform : NgxOutputTransform();
    m_preserveOriginalAlpha = prePostProcess;

    Rc<DxvkContext> dxvkCtx = ctx;
    createResources(dxvkCtx, renderExtent, displayExtent);

#ifndef _M_ARM64
    if (RtxOptions::isXeSSEnabled()) {
      syncXeSSInputResolution(displayExtent.width, displayExtent.height);
    }
#endif

    RtCamera& camera = ctx->getSceneManager().getCameraManager().getCamera(CameraType::Main);

    const uint32_t renderSize[2] = { m_renderExtent.width, m_renderExtent.height };
    const uint32_t displaySize[2] = { m_displayExtent.width, m_displayExtent.height };
    camera.setResolution(renderSize, displaySize);

    // The sub-pixel jitter was applied on the D3D9 side (viewport offset); the camera must
    // report exactly the value the frame was rasterized with so DLSS and DLFG agree with it
    camera.setExternalJitter(jitter);

    // Crossing into or out of that state swaps the content wholesale, with no motion to reconcile
    // the two. Only the crossing resets, so the movie still accumulates across its own frames.
    resetHistory |= m_sceneCameraFresh != m_previousSceneCameraFresh;
    m_previousSceneCameraFresh = m_sceneCameraFresh;

    const bool haveDepthMvInputs = generateMotionVectorsAndDepth(ctx, dxvkCtxState, barriers, sceneDepthImage, subrectOffset, camera,
                                                                 objectVelocities() ? velocityDraws : std::vector<NgxVelocityDraw>(),
                                                                 jitter);

    // Computed into a local and stored once at the end: the imgui panel reads m_upscalerActive
    // from the application thread while this dispatch runs on the CS thread, so a
    // clear-then-set pattern makes the displayed status flicker "inactive" mid-dispatch
    bool upscalerActive = false;

    const bool preserveTargetAlpha = prePostProcess;
    const VkOffset2D writebackOffset = prePostSubrectValid ? subrectOffset : VkOffset2D { 0, 0 };

    const bool canRunNis = RtxOptions::isNISEnabled();
    const bool canRunTaau = RtxOptions::isTAAEnabled() && haveDepthMvInputs;
    // Ray Reconstruction is path-traced only. In NGX passthrough, DLSS Super Resolution /
    // DLAA is selected purely by rtx.upscalerType (not enableRayReconstruction).
    const bool canRunDlss = RtxOptions::upscalerType() == UpscalerType::DLSS && haveDepthMvInputs;
#ifndef _M_ARM64
    const bool canRunXess = RtxOptions::isXeSSEnabled() && haveDepthMvInputs;
#else
    const bool canRunXess = false;
#endif
    const bool canRunUpscaler = canRunNis || canRunTaau || canRunDlss || canRunXess;

    // The MV pass rendered an interpretable view of the synthesized inputs into the output image
    // (motion vectors centered at neutral gray, log-scale depth). Held back for the present-time
    // overlay rather than written to the target here - see blitDebugOverlayToPresent for why
    // neither injection point can host it in place. Taken before the upscaler runs, which writes
    // its own result over the same image.
    if (haveDepthMvInputs && debugVisualization() != int(DebugVisualization::Off)) {
      VkImageCopy copyRegion = {};
      copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
      copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
      copyRegion.extent = { m_renderExtent.width, m_renderExtent.height, 1 };

      ctx->copyImage(m_debugPresentImage.image, copyRegion.dstSubresource, copyRegion.dstOffset,
                     m_dlssOutput.image, copyRegion.srcSubresource, copyRegion.srcOffset,
                     copyRegion.extent);

      m_debugPresentValid = true;
      m_statDebugVisCount++;
    }

    bool wroteOutputToTarget = false;

    // The render resolution is still driven from the selected upscaler's preset; only the
    // reconstruction is withheld, so the frame shows what that same input looks like without it.
    // Reported as an inactive upscaler, which is what it is.
    if (canRunUpscaler && bypassUpscaler()) {
      wroteOutputToTarget = evaluateUpscalerBypass(ctx, barriers, colorSourceImage, colorOffset, targetImage,
                                                   writebackOffset, preserveTargetAlpha, resetHistory);
      if (wroteOutputToTarget) {
        m_statusReason = "upscaler bypassed (plain stretch)";
      } else {
        m_statusReason = "upscaler bypass could not write its output";
        m_statEvaluateFailedCount++;
      }
    } else if (!canRunUpscaler) {
      if (RtxOptions::upscalerType() == UpscalerType::None) {
        m_statusReason = "rtx.upscalerType is None";
      } else if (RtxOptions::isRayReconstructionEnabled()) {
        m_statusReason = "Ray Reconstruction is not available in NGX passthrough mode";
      } else if (RtxOptions::isNISEnabled()) {
        m_statusReason = "NIS upscaler inactive";
      } else if (!haveDepthMvInputs) {
        m_statusReason = "depth/MV inputs unavailable";
        m_statNoInputsCount++;
      } else {
        m_statusReason = "selected upscaler unavailable";
      }
      m_statUpscalerOffCount++;

      if (evaluatePostFxOnly(ctx, barriers, colorSourceImage, colorOffset, targetImage,
                             writebackOffset, preserveTargetAlpha, resetHistory)) {
        wroteOutputToTarget = true;
        m_postFxActive = true;
        m_statusReason = "active (postfx only)";
      }
    } else if (canRunNis) {
      upscalerActive = evaluateNis(ctx, barriers, colorSourceImage, colorOffset, targetImage,
                                   writebackOffset, preserveTargetAlpha, resetHistory);
      wroteOutputToTarget = upscalerActive;
      if (upscalerActive) {
        m_statusReason = "active (NIS)";
      } else {
        m_statEvaluateFailedCount++;
      }
    } else if (canRunTaau) {
      upscalerActive = evaluateTaau(ctx, barriers, colorSourceImage, colorOffset, targetImage,
                                    writebackOffset, preserveTargetAlpha, jitter, resetHistory);
      wroteOutputToTarget = upscalerActive;
      if (upscalerActive) {
        m_statusReason = "active (TAA-U)";
      } else {
        m_statEvaluateFailedCount++;
      }
#ifndef _M_ARM64
    } else if (canRunXess) {
      upscalerActive = evaluateXess(ctx, barriers, colorSourceImage, colorOffset, targetImage,
                                    writebackOffset, preserveTargetAlpha, resetHistory);
      wroteOutputToTarget = upscalerActive;
      if (upscalerActive) {
        m_statusReason = "active (XeSS)";
      } else {
        m_statEvaluateFailedCount++;
      }
#endif
    } else if (canRunDlss) {
      upscalerActive = evaluateDlss(ctx, barriers, colorSourceImage, colorOffset, targetImage,
                                    writebackOffset, preserveTargetAlpha, jitter, resetHistory);
      wroteOutputToTarget = upscalerActive;

      if (upscalerActive) {
        m_statusReason = "active (DLSS)";
      } else {
        m_statEvaluateFailedCount++;
      }
    } else {
      m_statNoInputsCount++;
    }

    // Pre-post-process: also mirror the result into the scene color render surface when the
    // post chain consumed a resolved copy, so later surface->texture re-resolves carry the
    // anti-aliased content instead of the raw scene. The merged image carries the game's
    // alpha (scene depth on UE3 D3D9). Debug visualization never reaches this path (it is
    // late-injection only), so the merged image is always the one that was written.
    if (wroteOutputToTarget && prePostProcess && mirrorTargetImage != nullptr &&
        mirrorTargetImage.ptr() != targetImage.ptr()) {
      blitToTargetRect(ctx, m_mergedOutput.image, m_displayExtent, mirrorTargetImage,
                       prePostSubrectValid ? subrectOffset : VkOffset2D { 0, 0 });
    }

    m_upscalerActive = upscalerActive;

    // Periodic diagnostic summary: makes intermittent dispatch/upscaler dropouts visible in the
    // log with their reasons (the imgui status text alone cannot show fast alternation)
    m_statDispatchCount++;
    if (upscalerActive) {
      m_statUpscalerActiveCount++;
    }
    if (resetHistory) {
      // History resets discard DLSS's temporal accumulation; anything beyond rare camera
      // cuts here would directly explain a soft image
      m_statResetHistoryCount++;
    }
    if (prePostProcess) {
      m_statPrePostCount++;
    }

    if (m_statDispatchCount >= 600) {
      const uint32_t currentFrameId = m_device->getCurrentFrameId();

      if (m_statUpscalerActiveCount != m_statDispatchCount || m_statCameraInvalidCount != 0 || m_statResetHistoryCount != 0 ||
          m_statPrePostCount != m_statDispatchCount) {
        Logger::info(str::format("[RTX NGX Passthrough] Dispatch summary (frames ", m_statWindowStartFrameId, "-", currentFrameId, "): ",
                                 m_statDispatchCount, " dispatches, ",
                                 m_statUpscalerActiveCount, " with upscaler active, ",
                                 m_statPrePostCount, " at the pre-post-process injection point, ",
                                 m_statResetHistoryCount, " with history reset, ",
                                 m_statDebugVisCount, " in debug visualization, ",
                                 m_statEvaluateFailedCount, " with failed upscaler evaluation, ",
                                 m_statNoInputsCount, " without depth/MV inputs, ",
                                 m_statUpscalerOffCount, " with upscaler off/unavailable, ",
                                 m_statCameraInvalidCount, " frames skipped for invalid camera, ",
                                 m_statContentHdrFlips, " HDR/LDR content changes, peak camera reprojection ",
                                 m_statReprojectionFromIdentityMax, " from identity; last status: ", m_statusReason));
      }
      m_statDispatchCount = 0;
      m_statUpscalerActiveCount = 0;
      m_statNoInputsCount = 0;
      m_statUpscalerOffCount = 0;
      m_statCameraInvalidCount = 0;
      m_statEvaluateFailedCount = 0;
      m_statDebugVisCount = 0;
      m_statResetHistoryCount = 0;
      m_statPrePostCount = 0;
      m_statContentHdrFlips = 0;
      m_statReprojectionFromIdentityMax = 0.0f;
      m_statWindowStartFrameId = currentFrameId;
    }

    // The engine's stretch was suppressed expecting DLSS to fill the target; if nothing
    // else filled it (DLSS inactive and no debug view written), fall back to a plain
    // bilinear stretch so the frame is not left stale
    if (superResolution && !wroteOutputToTarget) {
      const DxvkFormatInfo* srcFormatInfo = imageFormatInfo(colorSourceImage->info().format);
      const DxvkFormatInfo* dstFormatInfo = imageFormatInfo(targetImage->info().format);

      VkImageBlit blitInfo = {};
      blitInfo.srcSubresource = { srcFormatInfo->aspectMask, 0, 0, 1 };
      blitInfo.dstSubresource = { dstFormatInfo->aspectMask, 0, 0, 1 };
      blitInfo.srcOffsets[0] = { subrectOffset.x, subrectOffset.y, 0 };
      blitInfo.srcOffsets[1] = { subrectOffset.x + int32_t(m_renderExtent.width),
                                 subrectOffset.y + int32_t(m_renderExtent.height), 1 };
      blitInfo.dstOffsets[0] = { 0, 0, 0 };
      blitInfo.dstOffsets[1] = { int32_t(m_displayExtent.width), int32_t(m_displayExtent.height), 1 };

      const VkComponentMapping identityMap = {
        VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
      };

      ctx->blitImage(targetImage, identityMap, colorSourceImage, identityMap, blitInfo, VK_FILTER_LINEAR);
    }

    // Frame generation: hand this frame's camera + depth/motion vectors to the DLFG
    // presenter. The synthesized pair carries true object motion (foreground included),
    // so the interpolator moves dynamic objects and first person meshes geometrically.
    if (haveDepthMvInputs && ctx->isDLFGEnabled()) {
      // Force vsync off if DLFG is enabled, as FG + vsync is not properly supported
      if (RtxOptions::enableVsyncState != EnableVsync::Off) {
        RtxOptions::enableVsync.setDeferred(EnableVsync::Off);
        RtxOptions::enableVsyncState = EnableVsync::Off;
      }

      const Resources::Resource& depthInput = m_depthQueue.get();
      const Resources::Resource& motionVectorInput = m_motionVectorQueue.get();

      DxvkFrameInterpolationInfo dlfgInfo = {
        m_device->getCurrentFrameId(),
        camera,
        motionVectorInput.view,
        motionVectorInput.image->info().layout,
        depthInput.view,
        depthInput.image->info().layout,
        false,
        m_device->getCommon()->metaDLFG().getInterpolatedFrameCount(),
      };

      // HUD-less input: at the late injection point the target IS the pre-UI backbuffer,
      // so this frame's slot can be captured right here; at the pre-post point the capture
      // arrives later in the frame (first UI draw / frame end), filling the slot whose
      // view is handed over now, before this frame's submit reaches the presenter.
      if (dlfgHudlessInput()) {
        if (!prePostProcess) {
          captureHudless(ctx, targetImage);
        }

        const Resources::Resource& hudlessSlot = m_hudlessQueue.get();
        if (hudlessSlot.view != nullptr &&
            hudlessSlot.image->info().extent.width == targetExtent.width &&
            hudlessSlot.image->info().extent.height == targetExtent.height) {
          dlfgInfo.hudless = hudlessSlot.view;
        }
      }

      m_device->setupFrameInterpolation(dlfgInfo);
    }
  }

  void RtxNgxPassthrough::showImguiStatusLine(bool includeInjectionPoint) {
    if (m_upscalerActive) {
      if (includeInjectionPoint) {
        const char* injectionPoint = m_lastDispatchPrePost ? "pre-post-process" : "late injection";
        ImGui::TextWrapped(str::format(m_statusReason, ": ",
                                       m_renderExtent.width, "x", m_renderExtent.height,
                                       " -> ", m_displayExtent.width, "x", m_displayExtent.height,
                                       ", ", injectionPoint).c_str());
      } else {
        ImGui::TextWrapped(str::format(m_statusReason, ": ",
                                       m_renderExtent.width, "x", m_renderExtent.height,
                                       " -> ", m_displayExtent.width, "x", m_displayExtent.height).c_str());
      }
    } else if (m_postFxActive) {
      if (includeInjectionPoint) {
        const char* injectionPoint = m_lastDispatchPrePost ? "pre-post-process" : "late injection";
        ImGui::TextWrapped(str::format(m_statusReason, ": ",
                                       m_renderExtent.width, "x", m_renderExtent.height,
                                       " -> ", m_displayExtent.width, "x", m_displayExtent.height,
                                       ", ", injectionPoint).c_str());
      } else {
        ImGui::TextWrapped(str::format(m_statusReason, ": ",
                                       m_renderExtent.width, "x", m_renderExtent.height,
                                       " -> ", m_displayExtent.width, "x", m_displayExtent.height).c_str());
      }
    } else {
      ImGui::TextWrapped(str::format(upscalerModeLabel(), " inactive: ", m_statusReason).c_str());
    }
  }

  void RtxNgxPassthrough::showImguiSettings() {
    RemixGui::Checkbox("Sub-Pixel Camera Jitter", &enableJitterObject());
    RemixGui::Checkbox("Pre-Post-Process Injection (DLSS before the game's post chain)", &prePostProcessObject());
    RemixGui::Checkbox("Bypass Upscaler (keep the reduced render, plain stretch)", &bypassUpscalerObject());

    {
      static const int kPresetOptionValues[] = { 0, 1, 2, 3, 4, 5, 6, 10 };
      static const char* kPresetLabels = "Default\0Preset A (CNN)\0Preset B (CNN)\0Preset C (CNN)\0Preset D (CNN)\0Preset E (CNN)\0Preset F (CNN)\0Preset J (Transformer)\0";

      int comboIndex = 0;
      for (int i = 0; i < int(std::size(kPresetOptionValues)); i++) {
        if (kPresetOptionValues[i] == dlssRenderPreset()) {
          comboIndex = i;
          break;
        }
      }

      if (ImGui::Combo("DLSS Model Preset", &comboIndex, kPresetLabels)) {
        dlssRenderPresetObject().setDeferred(kPresetOptionValues[comboIndex]);
      }
    }

    showImguiStatusLine();

    // Diagnostics: dispatch cadence (should track the current frame 1:1 during gameplay),
    // DLSS feature re-initializations (should stay at 1 outside resolution changes), and the
    // sub-pixel jitter the last frame was rasterized with (0,0 means jitter is not engaging)
    const uint32_t currentFrameId = m_device->getCurrentFrameId();
    ImGui::TextWrapped(str::format("Last dispatch: frame ", m_lastDispatchFrameId, " (current ", currentFrameId,
                                   "), DLSS initializations: ", m_dlssInitCount).c_str());
    ImGui::TextWrapped(str::format("Jitter: ", m_lastJitter[0], ", ", m_lastJitter[1],
                                   " | Dynamic object draws: ", m_lastVelocityDrawCount).c_str());

    // Capture health (see NgxVelocityCaptureStats): new-registration or skip spikes
    // during motion pinpoint where the capture chain loses movers
    ImGui::TextWrapped(str::format("Velocity capture: ", m_lastCaptureStats.captured, " captured (",
                                   m_lastCaptureStats.capturedSkinned, " skinned, ",
                                   m_lastCaptureStats.capturedDynamic, " cpu-mesh, ",
                                   m_lastCaptureStats.capturedForeground, " fg), ",
                                   m_lastCaptureStats.exactMatches, " static, ",
                                   m_lastCaptureStats.newRegistrations, " new, ",
                                   m_lastCaptureStats.pairedBeyondBounds, " beyond-bounds, ",
                                   m_lastCaptureStats.skippedNoCamera, " no-camera, ",
                                   m_lastCaptureStats.skippedBudget, " budget, ",
                                   m_lastCaptureStats.skippedZDisabled, " z-off, ",
                                   m_lastCaptureStats.skippedInstanceCap, " inst-cap | frame camera: ",
                                   m_lastCaptureStats.frameCameraValid ? "valid" : "missing",
                                   " | depth clears: ", m_lastCaptureStats.depthClears,
                                   " | transpose flips: ", m_lastCaptureStats.cameraTransposeFlips).c_str());

    // rtx.ngxPassthrough.motionBlurFirstPerson and rtx.ngxPassthrough.objectVelocities
    // stay config-only: both default on and only serve as kill-switches
    RemixGui::Checkbox("Object Velocity Debug Freeze (zero motion)", &objectVelocityDebugFreezeObject());
    RemixGui::Checkbox("Frame Gen HUD-less UI Input", &dlfgHudlessInputObject());

    int debugVisualizationValue = debugVisualization();
    if (ImGui::Combo("Debug Visualization", &debugVisualizationValue,
                     "Off\0Motion Vectors\0Depth\0Object Velocity Coverage\0")) {
      debugVisualizationObject().setDeferred(debugVisualizationValue);
    }

    if (ImGui::Button("Dump Post-Chain Draw Flow To Log (4 Frames)")) {
      dumpPostChainFramesObject().setDeferred(4);
    }

    if (ImGui::Button("Dump Velocity Capture To Log (2 Frames)")) {
      dumpVelocityCaptureFramesObject().setDeferred(2);
    }
  }
}
