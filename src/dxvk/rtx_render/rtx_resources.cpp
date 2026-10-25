/*
* Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_resources.h"
#include "dxvk_device.h"
#include "dxvk_context.h"
#include "../util/util_blueNoise_128x128x64.h"
#include <assert.h>
#include "rtx_options.h"

namespace dxvk {

  Rc<DxvkImageView> Resources::createImageView(Rc<DxvkContext>& ctx,
                                               const Rc<DxvkImage>& image,
                                               const VkFormat format,
                                               const uint32_t numLayers,
                                               const VkImageViewType imageViewType,
                                               const VkImageUsageFlags extraUsageFlags,
                                               const uint32_t mipLevels) {
    DxvkImageViewCreateInfo viewInfo;
    viewInfo.type = imageViewType;
    viewInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | extraUsageFlags;
    viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.minLevel = 0;
    viewInfo.numLevels = mipLevels;
    viewInfo.minLayer = 0;
    viewInfo.numLayers = numLayers;

    viewInfo.format = format;

    return ctx->getDevice()->createImageView(image, viewInfo);
  }

  Resources::Resource Resources::createImageResource(Rc<DxvkContext>& ctx,
                                                     const char* name,
                                                     const VkExtent3D& extent,
                                                     const VkFormat format,
                                                     const uint32_t numLayers,
                                                     const VkImageType imageType,
                                                     const VkImageViewType imageViewType,
                                                     const VkImageCreateFlags imageCreateFlags,
                                                     const VkImageUsageFlags extraUsageFlags,
                                                     const VkClearColorValue clearValue,
                                                     const uint32_t mipLevels) {
    assert (mipLevels > 0);
    // max mip levels a texture can have based on its size:
    assert (mipLevels <= static_cast<uint32_t>(std::floor(std::log2(std::max(extent.width, extent.height)))) + 1);

    DxvkImageCreateInfo desc;
    desc.type = imageType;
    desc.flags = imageCreateFlags;
    desc.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    desc.extent = extent;
    desc.numLayers = numLayers;
    desc.mipLevels = mipLevels; 
    desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | extraUsageFlags;
    desc.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    desc.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    desc.tiling = VK_IMAGE_TILING_OPTIMAL;
    desc.layout = VK_IMAGE_LAYOUT_UNDEFINED;

    desc.format = format;

    if (imageViewType == VK_IMAGE_VIEW_TYPE_CUBE) {
      desc.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    }

    Resource resource;
    resource.image = ctx->getDevice()->createImage(desc, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXRenderTarget, name);
    resource.view = createImageView(ctx, resource.image, format, numLayers, imageViewType, extraUsageFlags, mipLevels);
    ctx->changeImageLayout(resource.image, VK_IMAGE_LAYOUT_GENERAL);

    VkImageSubresourceRange subRange = {};
    subRange.layerCount = 1;
    subRange.levelCount = 1;
    subRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    // Note: Initialize to zero, or we get corruption on resolution change
    ctx->clearColorImage(resource.image, clearValue, subRange);

    return resource;
  }

  Resources::SharedResource::SharedResource(Resources::Resource&& _resource)
    : resource(std::move(_resource)) {
  }

  Resources::AliasedResource::AliasedResource(Rc<DxvkContext>& ctx,
                                              const VkExtent3D& extent,
                                              const VkFormat format,
                                              const char* name,
                                              const bool allowCompatibleFormatAliasing,
                                              const uint32_t numLayers,
                                              const VkImageType imageType,
                                              const VkImageViewType imageViewType,
                                              VkImageCreateFlags imageCreateFlags,
                                              const VkImageUsageFlags extraUsageFlags,
                                              const VkClearColorValue clearValue,
                                              const uint32_t mipLevels)
#ifdef REMIX_DEVELOPMENT
    : m_thisObjectAddress(std::make_shared<const Resources::AliasedResource*>(this))
    , m_name(name)
#endif
  {

    if (allowCompatibleFormatAliasing) {
      imageCreateFlags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    }

    m_device = ctx->getDevice().ptr();
    m_sharedResource = new SharedResource(createImageResource(ctx, name, extent, format, numLayers, imageType, imageViewType, 
                                                              imageCreateFlags, extraUsageFlags, clearValue, mipLevels));
    m_view = m_sharedResource->resource.view;
    
    // Register that we are the owner, since the createImageResource call will perform a surface clear with `clearValue`.
    registerAccess(AccessType::Write);
  }

  Resources::AliasedResource& Resources::AliasedResource::operator=(Resources::AliasedResource&& other) {
    if (this != &other) {
      m_device = other.m_device;
      const bool takeOwnershipAfterMove = other.ownsResource();
      m_sharedResource = std::move(other.m_sharedResource);
      m_view = other.m_view;
      m_writeFrameIdx = other.m_writeFrameIdx;
#ifdef REMIX_DEVELOPMENT
      m_thisObjectAddress = std::make_shared<const AliasedResource*>(this);
      m_name = other.m_name;
      if (takeOwnershipAfterMove) {
        takeOwnership();
      }
#endif
    }

    return *this;
  }

  Rc<DxvkImageView> Resources::AliasedResource::view(Resources::AccessType accessType, bool isAccessedByGPU) const {
    registerAccess(accessType, isAccessedByGPU);

    return m_view;
  }

  Rc<DxvkImage> Resources::AliasedResource::image(Resources::AccessType accessType, bool isAccessedByGPU) const {
    registerAccess(accessType, isAccessedByGPU);

    return m_sharedResource->resource.image;
  }

  const Resources::Resource& Resources::AliasedResource::resource(Resources::AccessType accessType, bool isAccessedByGPU) const {
    registerAccess(accessType, isAccessedByGPU);

    m_sharedResource->resource.view = m_view;

    return m_sharedResource->resource;
  }

  bool Resources::AliasedResource::ownsResource() const {
#ifdef REMIX_DEVELOPMENT
    return !m_sharedResource->owner.expired() && m_sharedResource->owner.lock() == m_thisObjectAddress;
#else
    return true;
#endif
  }

  void Resources::AliasedResource::reset() {
    m_sharedResource = nullptr;
    m_view = nullptr;
#ifdef REMIX_DEVELOPMENT
    m_thisObjectAddress = nullptr;
#endif
  }

  const char* Resources::AliasedResource::name() const {
#ifdef REMIX_DEVELOPMENT     
    return m_name;
#else
    return nullptr;
#endif
  }

  void Resources::AliasedResource::registerAccess(AccessType accessType, bool isAccessedByGPU) const {

    if (isAccessedByGPU) {
      switch (accessType) {
      case AccessType::Write:
      case AccessType::ReadWrite:
        m_writeFrameIdx = m_device->getCurrentFrameId();
        break;
      case AccessType::Read:
        // Do nothing
        break;
      default:
        assert(0 && "Unsupported access type");
        break;
      }
    }

#ifdef REMIX_DEVELOPMENT   
    if (isAccessedByGPU) {
      switch (accessType) {
      case AccessType::Write:
        takeOwnership();
        break;
      case AccessType::ReadWrite:
      case AccessType::Read:
        if (!ownsResource()) {
          std::string errorMessage = str::format("AliasedResource WAR hazard detected:",
                 "\nNew access type: ", accessType == AccessType::Read ? "Read" : "ReadWrite",
                 "\nNew owner: \"", name() ? name() : "name unknown", "\""
                 "\nPrevious owner: \"", m_sharedResource->owner.expired()
                                       ? "not set"
                                       : (*m_sharedResource->owner.lock().get())->name()
                                          ? (*m_sharedResource->owner.lock().get())->name()
                                          : "name unknown", "\"");
          ONCE(Logger::err(errorMessage));
          assert(0 && "[AliasedResource] WAR hazard detected");
        }
        break;
      default:
        assert(0 && "Unsupported access type");
        break;
      }
    }
#endif
  }

  void Resources::AliasedResource::takeOwnership() const {
#ifdef REMIX_DEVELOPMENT
    m_sharedResource->owner = m_thisObjectAddress;
#endif
  }

  bool Resources::AliasedResource::sharesTheSameView(const Resources::AliasedResource& other) const {
    return m_view.ptr() == other.m_view.ptr();
  }

  Resources::Resources(DxvkDevice* device)
    : CommonDeviceObject(device) { }

  void Resources::createBlueNoiseTexture(dxvk::Rc<DxvkContext> ctx) {
    DxvkImageCreateInfo desc;
    desc.type = VK_IMAGE_TYPE_2D;
    desc.format = VK_FORMAT_R8_UNORM;
    desc.flags = 0;
    desc.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    desc.extent = VkExtent3D { 128, 128, 1 };
    desc.numLayers = 64;
    desc.mipLevels = 1;
    desc.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    desc.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    desc.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    desc.tiling = VK_IMAGE_TILING_OPTIMAL;
    desc.layout = VK_IMAGE_LAYOUT_GENERAL;

    m_blueNoiseTex = m_device->createImage(desc, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXMaterialTexture, "blue noise");

    uint32_t rowPitch = desc.extent.width;
    uint32_t layerPitch = rowPitch * desc.extent.height;

    ctx->updateImage(m_blueNoiseTex,
                     VkImageSubresourceLayers { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, desc.numLayers },
                     VkOffset3D { 0, 0, 0 },
                     desc.extent,
                     (void*) &blueNoise_128x128x64[0][0], rowPitch, layerPitch);

    ctx->emitMemoryBarrier(0,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_ACCESS_SHADER_READ_BIT);

    DxvkImageViewCreateInfo viewInfo;
    viewInfo.type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    viewInfo.format = m_blueNoiseTex->info().format;
    viewInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.minLevel = 0;
    viewInfo.numLevels = 1;
    viewInfo.minLayer = 0;
    viewInfo.numLayers = 64;
    m_blueNoiseTexView = m_device->createImageView(m_blueNoiseTex, viewInfo);
  }

  Rc<DxvkSampler> dxvk::Resources::getSampler(
    const VkFilter filter,
    const VkSamplerMipmapMode mipFilter,
    const VkSamplerAddressMode addressModeU,
    const VkSamplerAddressMode addressModeV,
    const VkSamplerAddressMode addressModeW,
    const VkClearColorValue borderColor/* = VkClearColorValue()*/,
    const float mipBias/* = 0*/,
    const bool useAnisotropy/* = false*/) {
    const VkPhysicalDeviceLimits& limits = m_device->properties().core.properties.limits;
    const float maxAniso = std::min(limits.maxSamplerAnisotropy, RtxOptions::maxAnisotropySamples());

    // Fill out the rest of the sample create info
    DxvkSamplerCreateInfo samplerInfo;
    samplerInfo.magFilter = filter;
    samplerInfo.minFilter = filter;
    samplerInfo.mipmapMode = mipFilter;
    samplerInfo.mipmapLodBias = mipBias;
    samplerInfo.mipmapLodMin = 0.0f;
    samplerInfo.mipmapLodMax = VK_LOD_CLAMP_NONE;
    samplerInfo.useAnisotropy = useAnisotropy;
    samplerInfo.maxAnisotropy = maxAniso;
    samplerInfo.addressModeU = addressModeU;
    samplerInfo.addressModeV = addressModeV;
    samplerInfo.addressModeW = addressModeW;
    samplerInfo.compareToDepth = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.borderColor = borderColor;
    samplerInfo.usePixelCoord = VK_FALSE;
    
    // Build a hash key to lookup the sampler in the cache.
    XXH64_hash_t key = samplerInfo.calculateHash();

    Rc<DxvkSampler> sampler;

    auto samplerIt = m_samplerCache.find(key);
    if (samplerIt == m_samplerCache.end()) {

      // Create the sampler
      sampler = m_device->createSampler(samplerInfo);

      // Add it to the cache
      m_samplerCache.insert({ key, sampler });
    } else {
      sampler = samplerIt->second;
    }

    return sampler;
  }

  Rc<DxvkSampler> dxvk::Resources::getSampler(
    const VkFilter filter,
    const VkSamplerMipmapMode mipFilter,
    const VkSamplerAddressMode addressMode,
    const float mipBias/* = 0*/,
    const bool useAnisotropy/* = false*/) {
    return getSampler(filter, mipFilter, addressMode, addressMode, addressMode, VkClearColorValue(), mipBias, useAnisotropy);
  }

  Rc<DxvkImageView> Resources::getBlueNoiseTexture(dxvk::Rc<DxvkContext> ctx) {
    if (m_blueNoiseTex == nullptr) {
      createBlueNoiseTexture(ctx);
    }
    assert(m_blueNoiseTex != nullptr);
    assert(m_blueNoiseTexView != nullptr);

    return m_blueNoiseTexView;
  }

  void RtxPass::onFrameBegin(
    Rc<DxvkContext>& ctx,
    const FrameBeginContext& frameBeginCtx) {
    const bool prevFrameIsActive = m_isActive;
    m_isActive = isEnabled();
    if (m_isActive != prevFrameIsActive) {
      if (m_isActive) {
        if (onActivation(ctx)) {
          createTargetResource(ctx, frameBeginCtx.targetExtent);
          createDownscaledResource(ctx, frameBeginCtx.downscaledExtent);
        } else { // !onActivation()
          onDeactivation();
          m_isActive = false;
        }
      } else {
        releaseTargetResource();
        releaseDownscaledResource();
        onDeactivation();
      }
    }
  }
}  // namespace dxvk
