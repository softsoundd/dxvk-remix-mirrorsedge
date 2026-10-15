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

// Runs a material's own pixel shader on the CS thread over one repeat of its layers,
// inside the draw it was requested for, and keeps the result (rtx.ue3CompositeMaterialTextures).
// See "Composite materials" in UE3Compatibility.md.

#include "rtx_ue3_composite.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_asset_replacer.h"
#include "rtx_context.h"
#include "rtx_scene_manager.h"
#include "rtx_texture_manager.h"
#include "rtx_render/rtx_shader_manager.h"
#include "rtx/pass/ue3_composite/ue3_composite_encode.h"
#include "../../d3d9/d3d9_rtx.h"
#include "../../d3d9/d3d9_spec_constants.h"

#include <rtx_shaders/ue3_composite_encode.h>

namespace dxvk {

  namespace {

    class Ue3CompositeEncodeShader : public ManagedShader {
      SHADER_SOURCE(Ue3CompositeEncodeShader, VK_SHADER_STAGE_COMPUTE_BIT, ue3_composite_encode)

      PUSH_CONSTANTS(Ue3CompositeEncodeArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(UE3_COMPOSITE_ENCODE_PASS_INPUT)
        TEXTURE2D(UE3_COMPOSITE_ENCODE_EMISSION_INPUT)
        RW_TEXTURE2D(UE3_COMPOSITE_ENCODE_ALBEDO_OUTPUT)
        RW_TEXTURE2D(UE3_COMPOSITE_ENCODE_EMISSION_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(Ue3CompositeEncodeShader);

    // A bake still in a surface's material when it is released keeps its memory until the surface
    // rebinds, so only bakes no draw has used for a while are released.
    constexpr uint32_t kUe3CompositeMinIdleFramesBeforeRelease = 120;

    // Is how long a material that already has a bake draws a new key before baking it,
    // which is about the time UE3 takes to stream in the rest of a material's mips.
    constexpr uint32_t kUe3CompositeHoldFramesBeforeRebake = 60;

    // A key waiting this long has stopped being drawn.
    constexpr uint32_t kUe3CompositeMaxWaitFrames = 600;

    uint32_t mipLevelCount(const uint32_t width, const uint32_t height) {
      return 1u + uint32_t(std::floor(std::log2(double(std::max(width, height)))));
    }

    VkDeviceSize mipChainBytes(const uint32_t width, const uint32_t height, const VkDeviceSize texelBytes) {
      VkDeviceSize bytes = 0;
      uint32_t w = width;
      uint32_t h = height;
      for (;;) {
        bytes += VkDeviceSize(w) * VkDeviceSize(h) * texelBytes;
        if (w == 1 && h == 1) {
          break;
        }
        w = std::max(w / 2, 1u);
        h = std::max(h / 2, 1u);
      }
      return bytes;
    }

  }

  bool Ue3CompositeBaker::isSettled(const XXH64_hash_t key) {
    std::lock_guard<std::mutex> lock(s_mutex);
    return s_settledKeys.count(key) != 0;
  }

  std::vector<Ue3CompositeBaker::Status> Ue3CompositeBaker::getStatus() {
    std::lock_guard<std::mutex> lock(s_mutex);
    return s_status;
  }

  VkDeviceSize Ue3CompositeBaker::getTotalBytes() {
    std::lock_guard<std::mutex> lock(s_mutex);
    return s_totalBytes;
  }

  const Ue3CompositeBaker::Entry* Ue3CompositeBaker::find(const XXH64_hash_t key, const uint32_t frame) {
    auto it = m_entries.find(key);
    if (it == m_entries.end()) {
      return nullptr;
    }
    it->second.lastUsedFrame = frame;
    return &it->second;
  }

  const Ue3CompositeBaker::Entry* Ue3CompositeBaker::findLatest(const XXH64_hash_t lineage, const Vector2& period,
                                                                 const uint32_t frame) {
    Entry* latest = nullptr;
    for (auto& [key, entry] : m_entries) {
      if (entry.lineage == lineage && entry.period == period &&
          (latest == nullptr || entry.bakedFrame > latest->bakedFrame)) {
        latest = &entry;
      }
    }
    if (latest != nullptr) {
      latest->lastUsedFrame = frame;
    }
    return latest;
  }

  const Ue3CompositeBaker::Entry* Ue3CompositeBaker::store(const XXH64_hash_t key, Entry&& entry) {
    for (auto& [otherKey, other] : m_entries) {
      if (otherKey != key && other.lineage == entry.lineage) {
        other.superseded = true;
      }
    }
    m_totalBytes += entry.bytes;
    Entry& stored = m_entries[key];
    stored = std::move(entry);
    m_firstDrawnFrames.erase(key);
    markSettled(key);
    publishStatus();
    return &stored;
  }

  bool Ue3CompositeBaker::hasHeld(const XXH64_hash_t key, const uint32_t frame) {
    const uint32_t firstDrawn = m_firstDrawnFrames.emplace(key, frame).first->second;
    return frame - firstDrawn >= kUe3CompositeHoldFramesBeforeRebake;
  }

  void Ue3CompositeBaker::markSettled(const XXH64_hash_t key) {
    std::lock_guard<std::mutex> lock(s_mutex);
    s_settledKeys.insert(key);
  }

  void Ue3CompositeBaker::markUnsettled(const XXH64_hash_t key) {
    if (m_failedKeys.count(key) != 0) {
      return;
    }
    std::lock_guard<std::mutex> lock(s_mutex);
    s_settledKeys.erase(key);
  }

  void Ue3CompositeBaker::markFailed(const XXH64_hash_t key) {
    m_failedKeys.insert(key);
    markSettled(key);
  }

  bool Ue3CompositeBaker::takeBakeSlot(const uint32_t frame) {
    if (m_bakeFrame != frame) {
      m_bakeFrame = frame;
      m_bakesThisFrame = 0;
      m_maxBakesPerFrame = D3D9Rtx::ue3CompositeMaxBakesPerFrame();
    }
    if (m_bakesThisFrame >= m_maxBakesPerFrame) {
      return false;
    }
    m_bakesThisFrame++;
    return true;
  }

  float Ue3CompositeBaker::emissiveIntensity(const uint32_t frame) {
    if (m_optionsFrame != frame) {
      m_optionsFrame = frame;
      m_emissiveIntensity = std::max(D3D9Rtx::ue3CompositeEmissiveIntensity(), 0.0f);
    }
    return m_emissiveIntensity;
  }

  const Resources::Resource& Ue3CompositeBaker::getDummyStorage(Rc<DxvkContext>& ctx) {
    if (!m_dummyStorage.isValid()) {
      m_dummyStorage = Resources::createImageResource(ctx, "UE3 composite unused output", VkExtent3D { 1, 1, 1 },
                                                      VK_FORMAT_R8G8B8A8_UNORM);
    }
    return m_dummyStorage;
  }

  namespace {

    // Creates a 1x1 texture of one texel, for the bake's passes to bind in place of a material layer.
    Rc<DxvkImageView> createConstantTexture(Rc<DxvkContext>& ctx, const VkFormat format, const void* texel,
                                            const VkDeviceSize texelBytes, const char* name) {
      DxvkImageCreateInfo desc;
      desc.type = VK_IMAGE_TYPE_2D;
      desc.format = format;
      desc.flags = 0;
      desc.sampleCount = VK_SAMPLE_COUNT_1_BIT;
      desc.extent = VkExtent3D { 1, 1, 1 };
      desc.numLayers = 1;
      desc.mipLevels = 1;
      desc.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      desc.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      desc.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
      desc.tiling = VK_IMAGE_TILING_OPTIMAL;
      desc.layout = VK_IMAGE_LAYOUT_GENERAL;
      const Rc<DxvkImage> image = ctx->getDevice()->createImage(desc, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                                                DxvkMemoryStats::Category::RTXMaterialTexture, name);
      ctx->updateImage(image, VkImageSubresourceLayers { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, VkOffset3D { 0, 0, 0 },
                       desc.extent, texel, texelBytes, texelBytes);

      DxvkImageViewCreateInfo viewInfo;
      viewInfo.type = VK_IMAGE_VIEW_TYPE_2D;
      viewInfo.format = format;
      viewInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
      viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
      viewInfo.minLevel = 0;
      viewInfo.numLevels = 1;
      viewInfo.minLayer = 0;
      viewInfo.numLayers = 1;
      return ctx->getDevice()->createImageView(image, viewInfo);
    }

  }

  const Rc<DxvkImageView>& Ue3CompositeBaker::getBlackTexture(Rc<DxvkContext>& ctx) {
    if (m_blackTexture == nullptr) {
      const uint32_t texel = 0;
      m_blackTexture = createConstantTexture(ctx, VK_FORMAT_R8G8B8A8_UNORM, &texel, sizeof(texel), "UE3 composite black");
    }
    return m_blackTexture;
  }

  const Rc<DxvkImageView>& Ue3CompositeBaker::getHalfTexture(Rc<DxvkContext>& ctx) {
    if (m_halfTexture == nullptr) {
      // 0x3800 is 0.5 as a half float. A unorm texel cannot hold 0.5 exactly.
      const std::array<uint16_t, 4> texel = { 0x3800, 0x3800, 0x3800, 0x3800 };
      m_halfTexture = createConstantTexture(ctx, VK_FORMAT_R16G16B16A16_SFLOAT, texel.data(), sizeof(texel), "UE3 composite half");
    }
    return m_halfTexture;
  }

  void Ue3CompositeBaker::release(Rc<DxvkContext>& ctx, const XXH64_hash_t key, Entry& entry, const char* reason) {
    // The texture cache holds GPU-made textures until told to drop them (REMIX-1557).
    RtxTextureManager& textureManager = ctx->getCommonObjects()->getTextureManager();
    auto releaseResource = [&](RtxMipmap::Resource& resource) {
      if (!resource.isValid()) {
        return;
      }
      TextureRef viewRef(resource.view);
      textureManager.releaseTexture(viewRef);
      for (Rc<DxvkImageView>& view : resource.views) {
        TextureRef mipRef(view);
        textureManager.releaseTexture(mipRef);
      }
      resource.reset();
    };
    releaseResource(entry.albedo);
    releaseResource(entry.emissive);
    releaseResource(entry.maskedAlbedo);
    releaseResource(entry.maskedHalfAlbedo);
    m_totalBytes -= std::min(m_totalBytes, entry.bytes);

    if (D3D9Rtx::ue3LogCompositeMaterials()) {
      Logger::info(str::format(
        "[RTX-Compatibility][UE3-Composite] Released materialHash=0x", std::hex, entry.materialHash, " key=0x", key, std::dec,
        " (", reason, "): ", double(entry.bytes) / (1024.0 * 1024.0), " MB, ",
        double(m_totalBytes) / (1024.0 * 1024.0), " MB left in composite bakes"));
    }
  }

  void Ue3CompositeBaker::onFrameEnd(Rc<DxvkContext> ctx, const uint32_t frame) {
    bool changed = false;
    if (s_clearRequested.exchange(false, std::memory_order_relaxed)) {
      for (auto& [key, entry] : m_entries) {
        release(ctx, key, entry, "re-bake requested");
      }
      m_entries.clear();
      m_failedKeys.clear();
      m_firstDrawnFrames.clear();
      {
        std::lock_guard<std::mutex> lock(s_mutex);
        s_settledKeys.clear();
      }
      changed = true;
    }

    for (auto it = m_entries.begin(); it != m_entries.end();) {
      if (it->second.superseded && frame - it->second.lastUsedFrame >= kUe3CompositeMinIdleFramesBeforeRelease) {
        const XXH64_hash_t key = it->first;
        release(ctx, key, it->second, "superseded by a newer bake of the material");
        it = m_entries.erase(it);
        markUnsettled(key);
        changed = true;
      } else {
        ++it;
      }
    }
    m_firstDrawnFrames.erase_if([frame](const auto& it) { return frame - it->second > kUe3CompositeMaxWaitFrames; });

    const VkDeviceSize budget = VkDeviceSize(D3D9Rtx::ue3CompositeVramBudgetMB()) * 1024 * 1024;
    while (m_totalBytes > budget) {
      auto oldest = m_entries.end();
      for (auto it = m_entries.begin(); it != m_entries.end(); ++it) {
        if (frame - it->second.lastUsedFrame < kUe3CompositeMinIdleFramesBeforeRelease) {
          continue;
        }
        if (oldest == m_entries.end() || it->second.lastUsedFrame < oldest->second.lastUsedFrame) {
          oldest = it;
        }
      }
      if (oldest == m_entries.end()) {
        break;
      }
      const XXH64_hash_t key = oldest->first;
      release(ctx, key, oldest->second, "over rtx.d3d9.ue3CompositeVramBudgetMB");
      m_entries.erase(oldest);
      markUnsettled(key);
      changed = true;
    }

    if (changed) {
      publishStatus();
    }
  }

  void Ue3CompositeBaker::publishStatus() {
    std::vector<Status> status;
    status.reserve(m_entries.size());
    for (const auto& [key, entry] : m_entries) {
      Status row;
      row.materialHash = entry.materialHash;
      row.pickTextureHash = entry.pickTextureHash;
      row.width = entry.width;
      row.height = entry.height;
      row.period = entry.period;
      row.hasEmission = entry.emissive.isValid();
      row.bytes = entry.bytes;
      row.lastUsedFrame = entry.lastUsedFrame;
      row.description = entry.description;
      status.push_back(std::move(row));
    }
    std::sort(status.begin(), status.end(), [](const Status& a, const Status& b) { return a.materialHash < b.materialHash; });
    std::lock_guard<std::mutex> lock(s_mutex);
    s_status = std::move(status);
    s_totalBytes = m_totalBytes;
  }

  namespace {

    // A replacement that authors its own albedo was authored over the composite's repeat, so the
    // surface only needs that repeat. The tiers are the ones SceneManager::determineMaterialData
    // resolves, in its order.
    bool replacementAuthorsAlbedo(SceneManager& sceneManager, const LegacyMaterialData& material) {
      const std::unique_ptr<AssetReplacer>& replacer = sceneManager.getAssetReplacer();
      if (replacer == nullptr) {
        return false;
      }
      const XXH64_hash_t tiers[] = {
        material.getHash(),
        material.getTextureSetAndShaderHash(),
        material.getColorTexture().getImageHash(),
      };
      for (const XXH64_hash_t hash : tiers) {
        if (hash == kEmptyHash) {
          continue;
        }
        const std::shared_ptr<MaterialData> replacement = replacer->getReplacementMaterial(hash);
        if (replacement != nullptr) {
          return replacement->getType() == MaterialDataType::Opaque &&
                 replacement->getOpaqueMaterialData().getAlbedoOpacityTexture().isValid();
        }
      }
      return false;
    }

    // Moves the surface onto the bake's UVs. The game's own Runner Vision glow texture does not tile with them,
    // so a highlight glows the surface colour instead.
    void useBakeUvs(DrawCallState& drawCallState, const Ue3CompositeBakeRequest& request) {
      drawCallState.modifyTransformData().textureTransform = request.domainTransform;
      drawCallState.modifyMaterialData().ue3HighlightGlowTexture = TextureRef();
    }

  }

  void RtxContext::bakeUe3Composite(DrawCallState& drawCallState) {
    const std::shared_ptr<const Ue3CompositeBakeRequest> request = drawCallState.ue3CompositeRequest;
    if (request == nullptr) {
      return;
    }
    ScopedCpuProfileZone();

    Ue3CompositeBaker& baker = getSceneManager().getUe3CompositeBaker();
    const uint32_t frame = m_device->getCurrentFrameId();

    if (replacementAuthorsAlbedo(getSceneManager(), drawCallState.getMaterialData())) {
      useBakeUvs(drawCallState, *request);
      baker.markSettled(request->key);
      return;
    }

    const Ue3CompositeBaker::Entry* entry = baker.find(request->key, frame);
    if (entry == nullptr && !baker.hasFailed(request->key)) {
      const Ue3CompositeBaker::Entry* previous = baker.findLatest(request->lineage, request->period, frame);
      if (!drawCallState.ue3CompositeDrawStateApplied) {
        // The draw came without its state because the key's bake was released after it settled.
        // The key's next draw brings its state for a bake again.
        baker.markUnsettled(request->key);
        entry = previous;
      } else if (previous != nullptr && !baker.hasHeld(request->key, frame)) {
        entry = previous;
      } else if (baker.takeBakeSlot(frame) && bakeUe3CompositeNow(*request)) {
        entry = baker.find(request->key, frame);
      } else {
        entry = previous;
      }
    }
    if (entry == nullptr) {
      return;
    }

    LegacyMaterialData& material = drawCallState.modifyMaterialData();
    material.ue3CompositeAlbedo = TextureRef(entry->albedo.view);
    material.ue3CompositeEmissive = entry->emissive.isValid() ? TextureRef(entry->emissive.view) : TextureRef();
    material.ue3CompositeSampler = entry->sampler;
    material.ue3CompositeIdentity = entry->identity;
    material.ue3CompositeEmissiveIntensity = baker.emissiveIntensity(frame);
    if (entry->mask.has_value() && entry->maskedAlbedo.isValid()) {
      material.ue3CompositeMaskedAlbedo = TextureRef(entry->maskedAlbedo.view);
      material.ue3CompositeMaskedHalfAlbedo =
        entry->maskedHalfAlbedo.isValid() ? TextureRef(entry->maskedHalfAlbedo.view) : TextureRef();
      material.ue3CompositeMask = entry->mask->texture;
      material.ue3CompositeMaskTransform = entry->mask->transform;
      material.ue3CompositeMaskChannel = entry->mask->channel;
      material.ue3CompositeMaskIsSrgb = entry->mask->isSrgb;
    }
    useBakeUvs(drawCallState, *request);
  }

  bool RtxContext::bakeUe3CompositeNow(const Ue3CompositeBakeRequest& request) {
    ScopedGpuProfileZone(this, "UE3 Composite Bake");
    Ue3CompositeBaker& baker = getSceneManager().getUe3CompositeBaker();
    Rc<DxvkContext> ctx = this;

    auto slotInRange = [&](const uint32_t slot) { return slot < m_rc.size(); };
    const bool slotsInRange =
      slotInRange(request.psConstantBufferSlot) &&
      std::all_of(request.whiteTextureSlots.begin(), request.whiteTextureSlots.end(), slotInRange) &&
      (!request.mask.has_value() || slotInRange(request.mask->slot));
    const DxvkBufferSlice liveConstants = slotsInRange ? m_rc[request.psConstantBufferSlot].bufferSlice : DxvkBufferSlice();
    const void* liveConstantData = liveConstants.defined() ? liveConstants.mapPtr(0) : nullptr;

    const char* failure = nullptr;
    if (request.vertexShader == nullptr || request.width == 0 || request.height == 0 || !slotsInRange) {
      failure = "the request is incomplete";
    } else if (m_state.gp.shaders.fs == nullptr) {
      failure = "no pixel shader is bound";
    } else if (m_state.gp.shaders.gs != nullptr || m_state.gp.shaders.tcs != nullptr || m_state.gp.shaders.tes != nullptr) {
      failure = "a geometry or tessellation stage is bound";
    } else if (liveConstantData == nullptr) {
      failure = "the pixel shader constants are not bound to readable memory";
    }
    if (failure != nullptr) {
      baker.markFailed(request.key);
      Logger::warn(str::format(
        "[RTX-Compatibility][UE3-Composite] materialHash=0x", std::hex, request.materialHash, std::dec,
        " cannot be baked: ", failure, ". It renders its albedo pick."));
      return false;
    }

    // Each pass reads the draw's own pixel shader constants, with its lighting registers set.
    VkDeviceSize constantsSize = liveConstants.length();
    for (const Ue3CompositeBakeRequest::ConstantOverride& constant : request.constants) {
      constantsSize = std::max<VkDeviceSize>(
        constantsSize, request.psFloatConstantsOffset + VkDeviceSize(constant.reg + 1) * sizeof(Vector4));
    }
    auto makeConstants = [&](const bool emissivePass) {
      DxvkBufferCreateInfo info = {};
      info.size = constantsSize;
      info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
      info.stages = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      info.access = VK_ACCESS_UNIFORM_READ_BIT;
      Rc<DxvkBuffer> buffer = m_device->createBuffer(
        info, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        DxvkMemoryStats::Category::RTXBuffer, "UE3 composite bake constants");
      uint8_t* data = static_cast<uint8_t*>(buffer->mapPtr(0));
      std::memset(data, 0, size_t(constantsSize));
      std::memcpy(data, liveConstantData, size_t(liveConstants.length()));
      for (const Ue3CompositeBakeRequest::ConstantOverride& constant : request.constants) {
        const VkDeviceSize offset = request.psFloatConstantsOffset + VkDeviceSize(constant.reg) * sizeof(Vector4);
        std::memcpy(data + offset, emissivePass ? &constant.emissivePass : &constant.diffusePass, sizeof(Vector4));
      }
      return DxvkBufferSlice(buffer, 0, constantsSize);
    };
    const DxvkBufferSlice diffuseConstants = makeConstants(false);
    const DxvkBufferSlice emissiveConstants = request.emissivePass ? makeConstants(true) : DxvkBufferSlice();

    // Every pass but the emission one renders into the same target, which is encoded before the next pass.
    const VkExtent3D extent = { request.width, request.height, 1 };
    const VkExtent3D emissiveExtent = request.uniformEmission ? VkExtent3D { 1, 1, 1 } : extent;
    const Resources::Resource passTarget = Resources::createImageResource(
      ctx, "UE3 composite bake", extent, VK_FORMAT_R16G16B16A16_SFLOAT, 1, VK_IMAGE_TYPE_2D,
      VK_IMAGE_VIEW_TYPE_2D, 0, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    Resources::Resource emissiveTarget;
    if (request.emissivePass) {
      emissiveTarget = Resources::createImageResource(
        ctx, "UE3 composite bake emission", emissiveExtent, VK_FORMAT_R16G16B16A16_SFLOAT, 1, VK_IMAGE_TYPE_2D,
        VK_IMAGE_VIEW_TYPE_2D, 0, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    }
    const bool masked = request.mask.has_value();
    const bool maskedHalf = masked && request.mask->degree == 2;

    // The material renders with these textures.
    const uint32_t mipLevels = mipLevelCount(request.width, request.height);
    const VkClearColorValue clear = { 0.0f, 0.0f, 0.0f, 0.0f };
    Ue3CompositeBaker::Entry entry;
    entry.albedo = RtxMipmap::createResource(ctx, "UE3 composite albedo", extent, VK_FORMAT_R8G8B8A8_UNORM, 0, clear, mipLevels);
    VkFormat emissiveFormat = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    if (request.storeEmission) {
      const VkFormatProperties properties = m_device->adapter()->formatProperties(emissiveFormat);
      if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) == 0) {
        emissiveFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
      }
      entry.emissive = RtxMipmap::createResource(ctx, "UE3 composite emission", emissiveExtent, emissiveFormat, 0, clear,
                                                 mipLevelCount(emissiveExtent.width, emissiveExtent.height));
    }
    if (masked) {
      entry.maskedAlbedo = RtxMipmap::createResource(ctx, "UE3 composite albedo with mask", extent, VK_FORMAT_R8G8B8A8_UNORM, 0,
                                                     clear, mipLevels);
      entry.mask = request.mask;
    }
    if (maskedHalf) {
      entry.maskedHalfAlbedo = RtxMipmap::createResource(ctx, "UE3 composite albedo with half mask", extent,
                                                         VK_FORMAT_R8G8B8A8_UNORM, 0, clear, mipLevels);
    }

    // The bake swaps these out of the draw's state and puts them back after its passes.
    const DxvkRenderTargets prevRenderTargets = m_state.om.renderTargets;
    const uint32_t prevViewportCount = m_state.gp.state.rs.viewportCount();
    const DxvkViewportState prevViewports = m_state.vp;
    const Rc<DxvkShader> prevVertexShader = m_state.gp.shaders.vs;
    const DxvkIaInfo prevInputAssembly = m_state.gp.state.ia;
    const DxvkOmAttachmentBlend prevBlend = m_state.gp.state.omBlend[0];
    const DxvkScInfo prevSpecConstants = getSpecConstantsInfo(VK_PIPELINE_BIND_POINT_GRAPHICS);
    std::vector<std::pair<uint32_t, Rc<DxvkImageView>>> prevTextures;
    for (const uint32_t slot : request.whiteTextureSlots) {
      prevTextures.emplace_back(slot, m_rc[slot].imageView);
    }
    if (masked) {
      prevTextures.emplace_back(request.mask->slot, m_rc[request.mask->slot].imageView);
    }

    setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::FogEnabled, false);
    setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::VertexFogMode, D3DFOG_NONE);
    setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::PixelFogMode, D3DFOG_NONE);

    DxvkInputAssemblyState inputAssembly;
    inputAssembly.primitiveTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestart = VK_FALSE;
    inputAssembly.patchVertexCount = 0;
    setInputAssemblyState(inputAssembly);

    DxvkBlendMode blend;
    blend.enableBlending = VK_FALSE;
    blend.colorSrcFactor = VK_BLEND_FACTOR_ONE;
    blend.colorDstFactor = VK_BLEND_FACTOR_ZERO;
    blend.colorBlendOp = VK_BLEND_OP_ADD;
    blend.alphaSrcFactor = VK_BLEND_FACTOR_ONE;
    blend.alphaDstFactor = VK_BLEND_FACTOR_ZERO;
    blend.alphaBlendOp = VK_BLEND_OP_ADD;
    blend.writeMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    setBlendMode(0, blend);

    const Rc<DxvkImageView> white = getResourceManager().getWhiteTexture(ctx);
    for (const uint32_t slot : request.whiteTextureSlots) {
      bindResourceView(slot, white, nullptr);
    }
    if (masked) {
      bindResourceView(request.mask->slot, baker.getBlackTexture(ctx), nullptr);
    }

    auto runPass = [&](const Resources::Resource& target, const VkExtent3D& targetExtent, const Rc<DxvkShader>& vertexShader,
                       const DxvkBufferSlice& constants) {
      // The game's pixel shader reads its render state from the D3D9 push constant bank,
      // which the encode pass and earlier RTX passes switch away from.
      setPushConstantBank(DxvkPushConstantBank::D3D9);
      bindShader(VK_SHADER_STAGE_VERTEX_BIT, vertexShader);
      bindResourceBuffer(request.psConstantBufferSlot, constants);
      const VkViewport viewport = { 0.0f, 0.0f, float(targetExtent.width), float(targetExtent.height), 0.0f, 1.0f };
      const VkRect2D scissor = { { 0, 0 }, { targetExtent.width, targetExtent.height } };
      setViewports(1, &viewport, &scissor);
      DxvkRenderTargets renderTargets;
      renderTargets.color[0].view = target.view;
      renderTargets.color[0].layout = VK_IMAGE_LAYOUT_GENERAL;
      bindRenderTargets(renderTargets);
      DxvkContext::draw(kUe3CompositeBakeVertexCount, 1, 0, 0);
    };

    // A one-level resource has no per-mip views, and its own view is mip 0.
    auto mip0 = [](const RtxMipmap::Resource& resource) {
      return resource.views.empty() ? resource.view : resource.views[0];
    };
    auto encode = [&](const RtxMipmap::Resource& albedo, const bool storeEmission) {
      Ue3CompositeEncodeArgs args = {};
      args.resolution = { request.width, request.height };
      args.flags = (request.emissivePass ? UE3_COMPOSITE_ENCODE_FLAG_EMISSION_PASS : 0u) |
                   (request.uniformEmission ? UE3_COMPOSITE_ENCODE_FLAG_UNIFORM_EMISSION : 0u) |
                   (request.coverageFromAlpha ? UE3_COMPOSITE_ENCODE_FLAG_COVERAGE_ALPHA : 0u) |
                   (storeEmission ? UE3_COMPOSITE_ENCODE_FLAG_STORE_EMISSION : 0u);
      setPushConstantBank(DxvkPushConstantBank::RTX);
      pushConstants(0, sizeof(args), &args);
      bindResourceView(UE3_COMPOSITE_ENCODE_PASS_INPUT, passTarget.view, nullptr);
      bindResourceView(UE3_COMPOSITE_ENCODE_EMISSION_INPUT, request.emissivePass ? emissiveTarget.view : passTarget.view, nullptr);
      bindResourceView(UE3_COMPOSITE_ENCODE_ALBEDO_OUTPUT, mip0(albedo), nullptr);
      bindResourceView(UE3_COMPOSITE_ENCODE_EMISSION_OUTPUT,
                       storeEmission ? mip0(entry.emissive) : baker.getDummyStorage(ctx).view, nullptr);
      bindShader(VK_SHADER_STAGE_COMPUTE_BIT, Ue3CompositeEncodeShader::getShader());
      const VkExtent3D workgroups = util::computeBlockCount(extent, VkExtent3D { 16, 16, 1 });
      dispatch(workgroups.width, workgroups.height, workgroups.depth);
    };

    if (request.emissivePass) {
      const Rc<DxvkShader>& emissiveVertexShader =
        request.emissiveVertexShader != nullptr ? request.emissiveVertexShader : request.vertexShader;
      runPass(emissiveTarget, emissiveExtent, emissiveVertexShader, emissiveConstants);
    }
    runPass(passTarget, extent, request.vertexShader, diffuseConstants);
    encode(entry.albedo, request.storeEmission);
    if (masked) {
      bindResourceView(request.mask->slot, white, nullptr);
      runPass(passTarget, extent, request.vertexShader, diffuseConstants);
      encode(entry.maskedAlbedo, false);
    }
    if (maskedHalf) {
      bindResourceView(request.mask->slot, baker.getHalfTexture(ctx), nullptr);
      runPass(passTarget, extent, request.vertexShader, diffuseConstants);
      encode(entry.maskedHalfAlbedo, false);
    }

    // Put the draw's state back.
    bindRenderTargets(prevRenderTargets);
    setViewports(prevViewportCount, prevViewports.viewports.data(), prevViewports.scissorRects.data());
    bindShader(VK_SHADER_STAGE_VERTEX_BIT, prevVertexShader);
    m_state.gp.state.ia = prevInputAssembly;
    m_state.gp.state.omBlend[0] = prevBlend;
    m_flags.set(DxvkContextFlag::GpDirtyPipelineState);
    setSpecConstantsInfo(VK_PIPELINE_BIND_POINT_GRAPHICS, prevSpecConstants);
    bindResourceBuffer(request.psConstantBufferSlot, liveConstants);
    for (const auto& [slot, view] : prevTextures) {
      bindResourceView(slot, view, nullptr);
    }

    RtxMipmap::updateMipmap(this, entry.albedo, MipmapMethod::Simple);
    if (request.storeEmission) {
      RtxMipmap::updateMipmap(this, entry.emissive, MipmapMethod::Simple);
    }
    if (masked) {
      RtxMipmap::updateMipmap(this, entry.maskedAlbedo, MipmapMethod::Simple);
    }
    if (maskedHalf) {
      RtxMipmap::updateMipmap(this, entry.maskedHalfAlbedo, MipmapMethod::Simple);
    }

    // Each bake gets hashes of its own. The render material and its surface material are cached on their
    // textures' hashes, and a bake that lands again under the same key must look new to them.
    const uint64_t generation = baker.nextGeneration();
    entry.identity = XXH3_64bits_withSeed(&generation, sizeof(generation), request.key);
    entry.albedo.image->setHash(entry.identity);
    const std::array<RtxMipmap::Resource*, 3> otherTextures = { &entry.emissive, &entry.maskedAlbedo, &entry.maskedHalfAlbedo };
    for (size_t i = 0; i < otherTextures.size(); i++) {
      if (otherTextures[i]->isValid()) {
        otherTextures[i]->image->setHash(XXH3_64bits_withSeed(&entry.identity, sizeof(entry.identity), request.key + i + 1));
      }
    }
    const uint32_t albedoTextures = 1 + (entry.maskedAlbedo.isValid() ? 1 : 0) + (entry.maskedHalfAlbedo.isValid() ? 1 : 0);
    const VkDeviceSize emissiveTexelBytes = emissiveFormat == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : 4;
    entry.sampler = getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    entry.bytes = mipChainBytes(request.width, request.height, 4) * albedoTextures +
                  (entry.emissive.isValid() ? mipChainBytes(emissiveExtent.width, emissiveExtent.height, emissiveTexelBytes) : 0);
    entry.lastUsedFrame = m_device->getCurrentFrameId();
    entry.bakedFrame = entry.lastUsedFrame;
    entry.lineage = request.lineage;
    entry.materialHash = request.materialHash;
    entry.pickTextureHash = request.pickTextureHash;
    entry.width = request.width;
    entry.height = request.height;
    entry.period = request.period;
    entry.description = request.description;

    if (request.log) {
      Logger::info(str::format(
        "[RTX-Compatibility][UE3-Composite] Baked materialHash=0x", std::hex, request.materialHash,
        " pick=0x", request.pickTextureHash, " key=0x", request.key, std::dec,
        " ", request.width, "x", request.height,
        !request.storeEmission ? "" : request.uniformEmission ? " with uniform emission" : " with emission",
        maskedHalf ? " and albedos with the mask at 1/2 and 1" : masked ? " and a second albedo with the mask at 1" : "",
        " (", double(entry.bytes) / (1024.0 * 1024.0), " MB, ",
        double(baker.getTotalBytes() + entry.bytes) / (1024.0 * 1024.0), " MB in all composite bakes)"));
    }

    baker.store(request.key, std::move(entry));
    return true;
  }

}
