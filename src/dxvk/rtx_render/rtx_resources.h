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
#pragma once

#include "../dxvk_buffer.h"
#include "../dxvk_image.h"
#include "../dxvk_sampler.h"
#include "vulkan/vulkan_core.h"
#include "rtx_utils.h"
#include "rtx_common_object.h"
#include "rtx/utility/shader_types.h"

namespace dxvk 
{
  class DxvkContext;
  class DxvkDevice;

  // DLFG async compute overlap: max of 2 frames in flight
  // (set to 1 to serialize graphics and async compute queues)
  constexpr uint32_t kDLFGMaxGPUFramesInFlight = 2;

  struct Resources : public CommonDeviceObject {
    class AliasedResource;

    struct Resource {
      Rc<DxvkImage> image = nullptr;
      Rc<DxvkImageView> view = nullptr;

      void reset() {
        image = nullptr;
        view = nullptr;
      }

      bool isValid() const {
        return image.ptr() && view.ptr();
      }
    };

    class SharedResource : public RcObject {
    public:
      SharedResource(Resource&& _resource);
#ifdef REMIX_DEVELOPMENT  
      std::weak_ptr<const AliasedResource*> owner;
#endif
      Resource resource;
    };

    enum class AccessType {
      Read,
      Write,
      ReadWrite
    };

    // AliasedResource can share a SharedResource with different AliasedResource objects
    // Usage rules:
    //  - AliasedResource takes ownership of a resource on write
    //  - AliasedResource must own a resource (i.e. wrote to it) before it can read from it
    //  - Different AliasedResources can subsequently write to a resource 
    class AliasedResource {
    public:
      AliasedResource()
#ifdef REMIX_DEVELOPMENT
        : m_thisObjectAddress(std::make_shared<const AliasedResource*>(this))
#endif
      { }

      AliasedResource(Rc<DxvkContext>& ctx,
                      const VkExtent3D& extent,
                      const VkFormat format,
                      const char* name,
                      const bool allowCompatibleFormatAliasing = false,
                      const uint32_t numLayers = 1,
                      const VkImageType imageType = VK_IMAGE_TYPE_2D,
                      const VkImageViewType imageViewType = VK_IMAGE_VIEW_TYPE_2D,
                      VkImageCreateFlags imageCreateFlags = 0,
                      const VkImageUsageFlags extraUsageFlags = VK_IMAGE_USAGE_STORAGE_BIT,
                      const VkClearColorValue clearValue = { 0.0f, 0.0f, 0.0f, 0.0f },
                      const uint32_t mipLevels = 1);

      AliasedResource(AliasedResource& other) = delete;   // Prevent shallow copy
      AliasedResource& operator=(AliasedResource&& other);

      // Returns a resource view for a given access type
      // bool isAccessedByGPU - input condition that controls if the resource is actually accessed by the GPU. 
      //  This is useful for cases when the resource gets bound but has a conditional access within a shader
      //  and the conditional access governs mutually exclusivity among AliasedResources
      Rc<DxvkImageView> view(AccessType accessType, bool isAccessedByGPU = true) const;

      // Returns a resource image for a given access type
      // bool isAccessedByGPU - input condition that controls if the resource is actually accessed by the GPU. 
      //  This is useful for cases when the resource gets bound but has a conditional access within a shader
      //  and the conditional access governs mutually exclusivity among AliasedResources
      Rc<DxvkImage> image(AccessType accessType, bool isAccessedByGPU = true) const;

      // Returns a resource for a given access type
      // bool isAccessedByGPU - input condition that controls if the resource is actually accessed by the GPU. 
      //  This is useful for cases when the resource gets bound but has a conditional access within a shader
      //  and the conditional access governs mutually exclusivity among AliasedResources
      const Resource& resource(AccessType accessType, bool isAccessedByGPU = true) const;

      void reset();

      const char* name() const;

      bool ownsResource() const;
      void registerAccess(AccessType accessType, bool isAccessedByGPU = true) const;

      bool sharesTheSameView(const AliasedResource& other) const;

      uint32_t getWriteFrameIdx() const {
        return m_writeFrameIdx;
      }

      bool matchesWriteFrameIdx(uint32_t frameIdx) const {
        return ownsResource() && m_writeFrameIdx == frameIdx;
      }

      const DxvkImageCreateInfo& imageInfo() const {
        return m_sharedResource->resource.image->info();
      }

      VkImageViewType imageViewType() const {
        return m_view->type();
      }

      const bool empty() const {
        return m_sharedResource == nullptr || m_view == nullptr;
      }

    private:
      void takeOwnership() const;

      // Storing device ptr using rc ptr results in compilation error which surprisingly is hard to resolve
      // Using a native pointer instead, which is fine given the reource should be torn 
      // down before the device does
      DxvkDevice* m_device = nullptr;

      // Resource that may be shared among AliasedResource objects
      Rc<SharedResource> m_sharedResource;

      Rc<DxvkImageView> m_view = nullptr;

      // Index of a frame when the shared resource was written to via this AliasedResource
      // Initialized to a large value, but not UINT32_MAX so that it does not trigger false positive 
      // when checked against a previous frame index on frame index 0
      mutable uint32_t m_writeFrameIdx = 0x0fffffff;

#ifdef REMIX_DEVELOPMENT
      // A shared ptr to the current object's address only shared with weak_ptrs so that its destruction is forwarded the weak_ptrs
      std::shared_ptr<const AliasedResource*> m_thisObjectAddress;

      // Debug name for the aliased resource
      const char* m_name = nullptr;
#endif
    };

    // a queue of N resources used over N frames
    struct ResourceQueue : public std::array<Resource, kDLFGMaxGPUFramesInFlight> {
      Resource& get() {
        return (*this)[idx];
      }

      const Resource& get() const {
        return (*this)[idx];
      }

      Resource& getPrevious() {
        return (*this)[(idx + int(size()) - 1) % int(size())];
      }

      const Resource& getPrevious() const {
        return (*this)[(idx + int(size()) - 1) % int(size())];
      }

      // The queues are only advanced when frame generation is available, so without it every
      // slot holds the current frame and getPrevious() aliases get(). Callers that need real
      // history rather than just a valid resource have to check.
      bool hasDistinctPrevious() const {
        return advanced;
      }

      void next() {
        idx = (idx + 1) % size();
        advanced = true;
      }

    private:
      int idx = 0;
      bool advanced = false;
    };

    explicit Resources(DxvkDevice* device);

    Rc<DxvkImageView> getBlueNoiseTexture(Rc<DxvkContext> ctx);

    Rc<DxvkSampler> getSampler(const VkFilter filter, const VkSamplerMipmapMode mipFilter,
                               const VkSamplerAddressMode addressModeU,
                               const VkSamplerAddressMode addressModeV,
                               const VkSamplerAddressMode addressModeW,
                               const VkClearColorValue borderColor = VkClearColorValue(),
                               const float mipBias = 0, const bool useAnisotropy = false);
    Rc<DxvkSampler> getSampler(const VkFilter filter, const VkSamplerMipmapMode mipFilter, 
                               const VkSamplerAddressMode addrMode, 
                               const float mipBias = 0, const bool useAnisotropy = false);

    static Rc<DxvkImageView> createImageView(Rc<DxvkContext>& ctx, const Rc<DxvkImage>& image, const VkFormat format,
                                             const uint32_t numLayers, const VkImageViewType imageViewType, 
                                             const VkImageUsageFlags extraUsageFlags = VK_IMAGE_USAGE_STORAGE_BIT, const uint32_t mipLevels = 1);
    static Resource createImageResource(Rc<DxvkContext>& ctx, const char *name, const VkExtent3D& extent, const VkFormat format,
                                        const uint32_t numLayers = 1, const VkImageType imageType = VK_IMAGE_TYPE_2D,
                                        const VkImageViewType imageViewType = VK_IMAGE_VIEW_TYPE_2D,
                                        const VkImageCreateFlags imageCreateFlags = 0, const VkImageUsageFlags extraUsageFlags = VK_IMAGE_USAGE_STORAGE_BIT,
                                        const VkClearColorValue clearValue = { 0.0f, 0.0f, 0.0f, 0.0f }, const uint32_t mipLevels = 1);

  private:
    Resources(Resources const&) = delete;
    Resources& operator=(Resources const&) = delete;

    void createBlueNoiseTexture(Rc<DxvkContext> ctx);

    Rc<DxvkImage> m_blueNoiseTex;
    Rc<DxvkImageView> m_blueNoiseTexView;

    fast_unordered_cache<Rc<DxvkSampler>> m_samplerCache;
  };

  // State passed to RtxPass::onFrameBegin()
  struct FrameBeginContext {
    VkExtent3D downscaledExtent;
    VkExtent3D targetExtent;
    float frameTimeMilliseconds;
    bool resetHistory;
    bool isCameraCut;
  };

  class RtxPass {
  public:
    virtual ~RtxPass() = default;

    // m_isActive is updated from the derived class's isEnabled() in onFrameBegin(), so isActive() stays
    // consistent for the rest of the frame even if ImGUI changes the underlying option in between.
    // External callers and internal checks should use isActive() rather than isEnabled().
    bool isActive() const { return m_isActive; }

  protected:
    // Derived classes overriding this must call the base implementation first, then check isActive().
    virtual void onFrameBegin(Rc<DxvkContext>& ctx, const FrameBeginContext& frameBeginCtx);

  private:
    bool m_isActive = false;

  protected:
    // Interface to determine whether a pass is enabled
    // Note: this state updates m_isActive state at onFrameBegin() time
    virtual bool isEnabled() const = 0;

    // Called from RtxPass::onFrameBegin() on RtxPass (de)activation events.
    // onActivation returns whether it succeeded.
    // When this function should fail, the derived class implementation needs to make any 
    // runtime adjustments to ensure valid runtime state from this point onwards. 
    virtual bool onActivation(Rc<DxvkContext>& /* unused */) { return true; }
    virtual void onDeactivation() { }

    // Resource management functions, should provide implementation if a pass has custom resources
    virtual void createTargetResource(Rc<DxvkContext>& ctx, const VkExtent3D& targetExtent) { }
    virtual void releaseTargetResource() { }
    virtual void createDownscaledResource(Rc<DxvkContext>& ctx, const VkExtent3D& downscaledExtent) { }
    virtual void releaseDownscaledResource() { }
  };

} // namespace dxvk
