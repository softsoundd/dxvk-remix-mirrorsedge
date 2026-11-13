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
#include "rtx_fsr_framegen.h"

#include <algorithm>
#include <limits>

#include "rtx_fsr.h"
#include "rtx_fsr_api.h"
#include "../dxvk_adapter.h"
#include "../dxvk_device.h"
#include "../dxvk_objects.h"
#include "../dxvk_scoped_annotation.h"
#include "../util/util_once.h"

namespace dxvk {

  namespace {

    // A frame without inputs is still generated from, with a history reset, for this many frames in a row.
    // Past that, generation is switched off until inputs return, which makes FSR wait for its pending frames.
    constexpr uint32_t kMaxFramesWithoutInputs = 3;

    // The frame generation context is told at creation which format the HUD-less image has.
    // It is only recreated for a different format once that many frames in a row have asked for it.
    constexpr uint32_t kHudlessSwitchFrames = 30;

    // FSR's frame generation swapchain refuses a device with more queue families than this.
    constexpr uint32_t kMaxQueueFamilies = 16;

    constexpr uint32_t kDebugOverlayFlags =
      FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_TEAR_LINES |
      FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_RESET_INDICATORS |
      FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_PACING_LINES;

    // FSR can take a HUD-less image in another format than the swapchain's, as long as both store colour with the same precision.
    // Returns a number that formats of the same precision share.
    // Returns -1 for a format frame generation does not take at all.
    int formatPrecisionGroup(VkFormat format) {
      switch (format) {
      case VK_FORMAT_R16G16B16A16_SFLOAT:
        return 1;
      case VK_FORMAT_R8G8B8A8_UNORM:
      case VK_FORMAT_B8G8R8A8_UNORM:
        return 2;
      case VK_FORMAT_R8G8B8A8_SRGB:
      case VK_FORMAT_B8G8R8A8_SRGB:
        return 4;
      case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        return 6;
      default:
        return -1;
      }
    }

    VkQueueInfoFFXAPI queueInfo(const fsr::QueueCandidate& queue, uint32_t unusedFamily) {
      VkQueueInfoFFXAPI info = {};
      info.queue = reinterpret_cast<VkQueue>(queue.handle);
      // FSR reads the family of a queue it was not given as well, so that one has to be a family that exists.
      info.familyIndex = queue.handle != 0 ? queue.family : unusedFamily;
      info.submitFunc = nullptr;

      return info;
    }

    fsr::DepthParameters depthParameters(const DxvkFrameInterpolationInfo& inputs) {
      const Matrix4d& projection = inputs.camera.getViewToProjection();

      return fsr::depthParametersFromProjection(
        float(projection[2][2]), float(projection[3][2]), float(projection[2][3]),
        inputs.depthReversed, inputs.camera.calculateNearFarPlanes().first);
    }

    void copyVector(const Vector3& vector, float out[3]) {
      out[0] = vector.x;
      out[1] = vector.y;
      out[2] = vector.z;
    }

    void copyDirection(const Vector3& direction, float out[3]) {
      const float directionLength = length(direction);

      copyVector(directionLength > 0.0f ? direction / directionLength : direction, out);
    }

    std::string describeQueue(const fsr::QueueCandidate& queue) {
      return queue.handle != 0 ? str::format("family ", queue.family) : std::string("none");
    }

  } // namespace

  DxvkFSRFrameGen::DxvkFSRFrameGen(DxvkDevice* device)
    : CommonDeviceObject(device) { }

  DxvkFSRFrameGen::~DxvkFSRFrameGen() { }

  bool DxvkFSRFrameGen::hasRuntimeFiles() {
    return FsrApi::hasRuntimeFiles(FsrEffect::FrameGeneration);
  }

  bool DxvkFSRFrameGen::isSupported() const {
    if (!FsrApi::get().isAvailable(FsrEffect::FrameGeneration)) {
      return false;
    }

    // The device's queues do not change, so they are looked at once.
    std::call_once(m_queueSupportOnce, [this] {
      m_queuesSupported = selectQueues(VK_NULL_HANDLE).valid;
    });

    return m_queuesSupported;
  }

  const char* DxvkFSRFrameGen::unsupportedReason() const {
    if (isSupported()) {
      return "";
    }

    if (!hasRuntimeFiles()) {
      return "The AMD FSR runtime is not installed next to the Remix runtime.";
    }

    if (!FsrApi::get().isAvailable(FsrEffect::FrameGeneration)) {
      return "The AMD FSR runtime could not be loaded.";
    }

    return "The GPU does not provide the spare queues that frame generation presents from.";
  }

  bool DxvkFSRFrameGen::isActive() const {
    return enable() && !m_failed.load() && isSupported();
  }

  fsr::FrameGenQueues DxvkFSRFrameGen::selectQueues(VkSurfaceKHR surface) const {
    const Rc<vk::InstanceFn> vki = m_device->adapter()->vki();
    const VkPhysicalDevice adapter = m_device->adapter()->handle();

    uint32_t familyCount = 0;
    vki->vkGetPhysicalDeviceQueueFamilyProperties(adapter, &familyCount, nullptr);

    std::vector<VkQueueFamilyProperties> families(familyCount);
    vki->vkGetPhysicalDeviceQueueFamilyProperties(adapter, &familyCount, families.data());

    if (familyCount > kMaxQueueFamilies) {
      return fsr::FrameGenQueues();
    }

    const auto candidate = [&] (const DxvkDeviceQueue& queue) {
      fsr::QueueCandidate result;

      if (queue.queueHandle == VK_NULL_HANDLE || queue.queueFamily >= familyCount) {
        return result;
      }

      result.handle = reinterpret_cast<uintptr_t>(queue.queueHandle);
      result.family = queue.queueFamily;
      result.flags = families[queue.queueFamily].queueFlags;

      if (surface != VK_NULL_HANDLE) {
        VkBool32 supported = VK_FALSE;
        const VkResult status = vki->vkGetPhysicalDeviceSurfaceSupportKHR(adapter, queue.queueFamily, surface, &supported);

        result.canPresent = status == VK_SUCCESS && supported;
      } else {
        result.canPresent = vki->vkGetPhysicalDeviceWin32PresentationSupportKHR(adapter, queue.queueFamily);
      }

      return result;
    };

    const DxvkDeviceQueueSet& queues = m_device->queues();

    const fsr::QueueCandidate game = candidate(queues.graphics);
    const fsr::QueueCandidate fsrPresent = candidate(queues.fsrPresent);
    const fsr::QueueCandidate imageAcquire = candidate(queues.imageAcquire);
    // DLSS Frame Generation presents from this queue, and the two never run at the same time.
    const fsr::QueueCandidate present = candidate(queues.present);
    const fsr::QueueCandidate asyncCompute = candidate(queues.asyncCompute);

    // A second queue of the graphics family lets FSR compose the frames on the queue it presents from.
    // Any other present queue leaves the composition on the game's queue.
    return fsr::selectFrameGenQueues(game,
      { fsrPresent, present },
      { imageAcquire, asyncCompute, present },
      { asyncCompute, present });
  }

  void DxvkFSRFrameGen::updateSettings() {
    Settings settings;
    settings.allowAsyncWorkloads = allowAsyncWorkloads();
    settings.enableDebugChecker = enableDebugChecker();
    settings.debugOverlay = debugOverlay();
    settings.viewSpaceToMetersFactor = DxvkFSR::viewSpaceToMetersFactor();
    settings.versionOverride = frameGenerationVersion();

    std::lock_guard<std::mutex> lock(m_stateMutex);
    m_settings = settings;
  }

  DxvkFSRFrameGen::Settings DxvkFSRFrameGen::settings() const {
    std::lock_guard<std::mutex> lock(m_stateMutex);

    return m_settings;
  }

  bool DxvkFSRFrameGen::acquireProxy(const void* pOwner) {
    std::lock_guard<std::mutex> lock(m_stateMutex);

    if (m_pProxyOwner != nullptr && m_pProxyOwner != pOwner) {
      return false;
    }

    m_pProxyOwner = pOwner;

    return true;
  }

  void DxvkFSRFrameGen::releaseProxy(const void* pOwner) {
    std::lock_guard<std::mutex> lock(m_stateMutex);

    if (m_pProxyOwner == pOwner) {
      m_pProxyOwner = nullptr;
    }
  }

  void DxvkFSRFrameGen::notifyFailure(const std::string& reason) {
    if (!m_failed.exchange(true)) {
      Logger::err(str::format("FSR: Frame generation is switched off. ", reason,
                              " Switch it off and on again to retry."));
    }

    m_generating.store(false);
  }

  void DxvkFSRFrameGen::notifyContextVersion(const std::string& version) {
    std::lock_guard<std::mutex> lock(m_stateMutex);
    m_activeVersion = version;
  }

  std::string DxvkFSRFrameGen::activeVersion() const {
    std::lock_guard<std::mutex> lock(m_stateMutex);

    return m_activeVersion;
  }

  std::vector<std::string> DxvkFSRFrameGen::versionNames() const {
    std::vector<std::string> names;

    for (const FsrVersion& version : FsrApi::get().versions(FsrEffect::FrameGeneration, *m_device)) {
      names.push_back(version.name);
    }

    return names;
  }

  void DxvkFSRFrameGen::onEnableChanged(DxvkDevice* device) {
    if (device == nullptr) {
      return;
    }

    // Switching the option gives frame generation another try after a failure.
    device->getCommon()->metaFSRFrameGen().m_failed.store(false);
  }

  DxvkFSRFGPresenter::DxvkFSRFGPresenter(DxvkDevice* device,
                                         HWND window,
                                         const Rc<vk::InstanceFn>& vki,
                                         const Rc<vk::DeviceFn>& vkd,
                                         vk::PresenterDevice presenterDevice,
                                         const vk::PresenterDesc& desc)
    : vk::Presenter(window, vki, vkd, presenterDevice, desc),
      m_dxvkDevice(device) {
    // The base class has created a plain swapchain by now.
    // Its owner recreates the swapchain before the first present, which is where the proxy takes its place.
  }

  DxvkFSRFGPresenter::~DxvkFSRFGPresenter() {
    std::lock_guard<dxvk::mutex> lock(m_proxyMutex);

    // The base class destroys whatever plain swapchain is left after this.
    destroyProxy();
  }

  VkImageLayout DxvkFSRFGPresenter::presentLayout() const {
    return m_proxyActive ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  }

  void DxvkFSRFGPresenter::synchronize() {
    std::lock_guard<dxvk::mutex> lock(m_proxyMutex);

    if (!m_proxyActive || m_swapchainContext == nullptr) {
      return;
    }

    ffx::DispatchDescFrameGenerationSwapChainWaitForPresentsVK waitDesc;
    FsrApi::get().dispatch(&m_swapchainContext, &waitDesc.header);
  }

  VkResult DxvkFSRFGPresenter::acquireNextImage(vk::PresenterSync& sync, uint32_t& index, bool) {
    ScopedCpuProfileZone();

    std::lock_guard<dxvk::mutex> lock(m_proxyMutex);

    if (!m_proxyActive) {
      return vk::Presenter::acquireNextImage(sync, index, false);
    }

    sync = m_semaphores.at(m_frameIndex);

    // Each present acquires the image for the next frame, which leaves only the first frame of a new proxy to acquire here.
    if (m_acquireStatus == VK_NOT_READY) {
      m_acquireStatus = m_acquireNextImage(m_vkd->device(), m_swapchain,
        std::numeric_limits<uint64_t>::max(), sync.acquire, VK_NULL_HANDLE, &m_imageIndex);
    }

    if (m_acquireStatus != VK_SUCCESS && m_acquireStatus != VK_SUBOPTIMAL_KHR) {
      return m_acquireStatus;
    }

    index = m_imageIndex;

    return m_acquireStatus;
  }

  VkResult DxvkFSRFGPresenter::presentImage(std::atomic<VkResult>* presentStatus,
                                            const DxvkPresentInfo& presentInfo,
                                            const DxvkFrameInterpolationInfo& frameInterpolationInfo,
                                            std::uint32_t acquiredImageIndex,
                                            bool,
                                            VkSetPresentConfigNV*) {
    ScopedCpuProfileZone();

    std::lock_guard<dxvk::mutex> lock(m_proxyMutex);

    if (!m_proxyActive) {
      return vk::Presenter::presentImage(presentStatus, presentInfo, frameInterpolationInfo, acquiredImageIndex, false, nullptr);
    }

    vk::PresenterSync sync = m_semaphores.at(m_frameIndex);
    VkSemaphore presentWait = sync.present;

    // AMD's order is to configure frame generation, prepare the frame's depth and motion vectors, then present.
    // The present runs the generation itself, through frameGenerationCallback.
    updateFrameGeneration(frameInterpolationInfo, presentWait);

    VkPresentInfoKHR info;
    info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.pNext = nullptr;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &presentWait;
    info.swapchainCount = 1;
    info.pSwapchains = &m_swapchain;
    info.pImageIndices = &m_imageIndex;
    info.pResults = nullptr;

    const VkResult status = m_queuePresent(m_dxvkDevice->queues().graphics.queueHandle, &info);

    m_dxvkDevice->getCommon()->metaFSRFrameGen().notifyGenerating(m_generationEnabled.load());

    // Anything else makes the swapchain's owner recreate it, which builds a new proxy.
    if (status != VK_SUCCESS && status != VK_SUBOPTIMAL_KHR) {
      return status;
    }

    m_proxyFrameCount++;

    // Acquire the next image already, as the base presenter does, to keep the wait off the application's thread.
    m_frameIndex += 1;
    m_frameIndex %= m_semaphores.size();

    sync = m_semaphores.at(m_frameIndex);

    m_acquireStatus = m_acquireNextImage(m_vkd->device(), m_swapchain,
      std::numeric_limits<uint64_t>::max(), sync.acquire, VK_NULL_HANDLE, &m_imageIndex);

    const bool vsync = m_info.presentMode == VK_PRESENT_MODE_FIFO_KHR
                    || m_info.presentMode == VK_PRESENT_MODE_FIFO_RELAXED_KHR;

    m_fpsLimiter.delay(vsync);

    return status;
  }

  VkResult DxvkFSRFGPresenter::recreateSwapChain(const vk::PresenterDesc& desc) {
    std::lock_guard<dxvk::mutex> lock(m_proxyMutex);

    DxvkFSRFrameGen& frameGen = m_dxvkDevice->getCommon()->metaFSRFrameGen();

    // A proxy is always rebuilt from scratch.
    // The caller has idled the device, which destroying a proxy requires.
    destroyProxy();

    if (frameGen.isActive() && frameGen.acquireProxy(this)) {
      const ProxyResult result = createProxy(desc);

      if (result == ProxyResult::Created) {
        return VK_SUCCESS;
      }

      frameGen.releaseProxy(this);

      if (result == ProxyResult::Failed) {
        frameGen.notifyFailure("The frame generation swapchain could not be created.");
      }
    }

    // Present like the base presenter in the meantime.
    // A failure makes the swapchain's owner replace this presenter with a plain one at its next present.
    return vk::Presenter::recreateSwapChain(desc);
  }

  DxvkFSRFGPresenter::ProxyResult DxvkFSRFGPresenter::createProxy(const vk::PresenterDesc& desc) {
    FsrApi& api = FsrApi::get();
    DxvkFSRFrameGen& frameGen = m_dxvkDevice->getCommon()->metaFSRFrameGen();

    // FSR creates the real swapchain itself, and a surface takes one swapchain at a time.
    if (m_swapchain != VK_NULL_HANDLE) {
      vk::Presenter::destroySwapchain();
    }

    VkSurfaceCapabilitiesKHR caps;
    VkResult status = m_vki->vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_device.adapter, m_surface, &caps);

    if (status == VK_ERROR_SURFACE_LOST_KHR) {
      // Recreate the surface and try again, as the base presenter does.
      if (m_surface != VK_NULL_HANDLE) {
        destroySurface();
      }

      status = createSurface();

      if (status == VK_SUCCESS) {
        status = m_vki->vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_device.adapter, m_surface, &caps);
      }
    }

    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> modes;

    if (status != VK_SUCCESS ||
        getSupportedFormats(formats, desc) != VK_SUCCESS ||
        getSupportedPresentModes(modes, desc) != VK_SUCCESS) {
      Logger::err("FSR: Failed to query the surface for the frame generation swapchain.");
      return ProxyResult::Failed;
    }

    const VkExtent2D imageExtent = pickImageExtent(caps, desc.imageExtent);

    if (imageExtent.width == 0 || imageExtent.height == 0) {
      return ProxyResult::NoSurfaceArea;
    }

    m_queues = frameGen.selectQueues(m_surface);

    if (!m_queues.valid) {
      Logger::err("FSR: None of the device's spare queues can present to the window.");
      return ProxyResult::Failed;
    }

    m_info.format = pickFormat(formats.size(), formats.data(), desc.numFormats, desc.formats);
    m_info.presentMode = pickPresentMode(modes.size(), modes.data(), desc.numPresentModes, desc.presentModes);
    m_info.imageExtent = imageExtent;
    m_info.imageCount = fsr::proxyImageCount(desc.imageCount, caps.minImageCount, caps.maxImageCount);
    // FSR owns the real swapchain, which is the one exclusive fullscreen is acquired on.
    m_info.appOwnedFSE = false;

    VkSurfaceFullScreenExclusiveInfoEXT fullScreenInfo;
    fullScreenInfo.sType = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT;
    fullScreenInfo.pNext = nullptr;
    fullScreenInfo.fullScreenExclusive = desc.fullScreenExclusive == VK_FULL_SCREEN_EXCLUSIVE_APPLICATION_CONTROLLED_EXT
      ? VK_FULL_SCREEN_EXCLUSIVE_ALLOWED_EXT
      : desc.fullScreenExclusive;

    VkSwapchainCreateInfoKHR swapInfo;
    swapInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapInfo.pNext = m_device.features.fullScreenExclusive ? &fullScreenInfo : nullptr;
    swapInfo.flags = 0;
    swapInfo.surface = m_surface;
    // FSR creates this many images to render into, and asks for at least as many real ones.
    swapInfo.minImageCount = m_info.imageCount;
    swapInfo.imageFormat = m_info.format.format;
    swapInfo.imageColorSpace = m_info.format.colorSpace;
    swapInfo.imageExtent = m_info.imageExtent;
    swapInfo.imageArrayLayers = 1;
    swapInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    swapInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapInfo.queueFamilyIndexCount = 0;
    swapInfo.pQueueFamilyIndices = nullptr;
    swapInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    swapInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapInfo.presentMode = m_info.presentMode;
    swapInfo.clipped = VK_TRUE;
    swapInfo.oldSwapchain = VK_NULL_HANDLE;

    const DxvkDeviceQueue& gameQueue = m_dxvkDevice->queues().graphics;

    ffx::CreateContextDescFrameGenerationSwapChainVK createDesc;
    createDesc.physicalDevice = m_device.adapter;
    createDesc.device = m_vkd->device();
    // There is no swapchain to hand over, since the base presenter's was destroyed above.
    // That leaves m_swapchain null whenever FSR does not get as far as creating its own.
    createDesc.swapchain = &m_swapchain;
    createDesc.allocator = nullptr;
    createDesc.createInfo = swapInfo;
    createDesc.gameQueue.queue = gameQueue.queueHandle;
    createDesc.gameQueue.familyIndex = gameQueue.queueFamily;
    createDesc.gameQueue.submitFunc = nullptr;
    createDesc.asyncComputeQueue = queueInfo(m_queues.asyncCompute, gameQueue.queueFamily);
    createDesc.presentQueue = queueInfo(m_queues.present, gameQueue.queueFamily);
    createDesc.imageAcquireQueue = queueInfo(m_queues.imageAcquire, gameQueue.queueFamily);

    // The runtime keeps a context from using anything newer than the version it was built against, unless told.
    ffx::CreateContextDescFrameGenerationSwapChainVersionVK versionDesc;
    versionDesc.version = FFX_FRAMEGENERATION_SWAPCHAIN_VK_VERSION;

    createDesc.header.pNext = &versionDesc.header;

    ffxContext swapchainContext = nullptr;
    const ffxReturnCode_t result = api.createContext(FsrEffect::FrameGenerationSwapchain, &swapchainContext, &createDesc.header);

    if (result != FFX_API_RETURN_OK || swapchainContext == nullptr || m_swapchain == VK_NULL_HANDLE) {
      Logger::err(str::format("FSR: Failed to create the frame generation swapchain: ", FsrApi::returnCodeName(result), "."));
      m_swapchain = VK_NULL_HANDLE;
      return ProxyResult::Failed;
    }

    m_swapchainContext = swapchainContext;
    m_proxyActive = true;

    ffx::QueryDescSwapchainReplacementFunctionsVK functions;

    if (api.query(&m_swapchainContext, &functions.header) != FFX_API_RETURN_OK ||
        functions.pOutGetSwapchainImagesKHR == nullptr || functions.pOutAcquireNextImageKHR == nullptr ||
        functions.pOutQueuePresentKHR == nullptr || functions.pOutDestroySwapchainFFXAPI == nullptr) {
      Logger::err("FSR: The frame generation swapchain did not return its replacement functions.");
      destroyProxy();
      return ProxyResult::Failed;
    }

    m_acquireNextImage = functions.pOutAcquireNextImageKHR;
    m_queuePresent = functions.pOutQueuePresentKHR;
    m_destroySwapchain = functions.pOutDestroySwapchainFFXAPI;
    m_presentCount = functions.pOutGetLastPresentCountFFXAPI;
    m_proxyFrameCount = 0;

    uint32_t imageCount = 0;
    status = functions.pOutGetSwapchainImagesKHR(m_vkd->device(), m_swapchain, &imageCount, nullptr);

    std::vector<VkImage> images(imageCount);

    if (status == VK_SUCCESS) {
      status = functions.pOutGetSwapchainImagesKHR(m_vkd->device(), m_swapchain, &imageCount, images.data());
    }

    if (status != VK_SUCCESS || imageCount == 0) {
      Logger::err("FSR: The frame generation swapchain did not return its images.");
      destroyProxy();
      return ProxyResult::Failed;
    }

    m_info.imageCount = imageCount;
    m_images.resize(imageCount);
    m_semaphores.resize(imageCount);

    // The views and semaphores mirror what the base presenter creates for a swapchain of its own.
    for (uint32_t i = 0; i < imageCount && status == VK_SUCCESS; i++) {
      m_images[i].image = images[i];

      VkImageViewCreateInfo viewInfo;
      viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      viewInfo.pNext = nullptr;
      viewInfo.flags = 0;
      viewInfo.image = images[i];
      viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
      viewInfo.format = m_info.format.format;
      viewInfo.components = VkComponentMapping {
        VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
      viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

      status = m_vkd->vkCreateImageView(m_vkd->device(), &viewInfo, nullptr, &m_images[i].view);

      VkSemaphoreCreateInfo semaphoreInfo;
      semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
      semaphoreInfo.pNext = nullptr;
      semaphoreInfo.flags = 0;

      if (status == VK_SUCCESS) {
        status = m_vkd->vkCreateSemaphore(m_vkd->device(), &semaphoreInfo, nullptr, &m_semaphores[i].acquire);
      }

      if (status == VK_SUCCESS) {
        status = m_vkd->vkCreateSemaphore(m_vkd->device(), &semaphoreInfo, nullptr, &m_semaphores[i].present);
      }
    }

    if (status != VK_SUCCESS || !createPrepareCommands()) {
      Logger::err("FSR: Failed to create the resources of the frame generation swapchain.");
      destroyProxy();
      return ProxyResult::Failed;
    }

    m_imageIndex = 0;
    m_frameIndex = 0;
    m_acquireStatus = VK_NOT_READY;

    // Not every version of the swapchain answers the version query.
    const std::string swapchainVersion = api.contextVersion(&m_swapchainContext);

    Logger::info(str::format("FSR: Created the frame generation swapchain:",
      "\n  Format:        ", m_info.format.format,
      "\n  Present mode:  ", m_info.presentMode,
      "\n  Buffer size:   ", m_info.imageExtent.width, "x", m_info.imageExtent.height,
      "\n  Image count:   ", m_info.imageCount,
      "\n  Version:       ", swapchainVersion.empty() ? "not reported" : swapchainVersion,
      "\n  Present queue: ", describeQueue(m_queues.present),
      "\n  Acquire queue: ", describeQueue(m_queues.imageAcquire),
      "\n  Compute queue: ", describeQueue(m_queues.asyncCompute)));

    return ProxyResult::Created;
  }

  void DxvkFSRFGPresenter::destroyProxy() {
    if (!m_proxyActive) {
      return;
    }

    FsrApi& api = FsrApi::get();
    DxvkFSRFrameGen& frameGen = m_dxvkDevice->getCommon()->metaFSRFrameGen();

    // AMD's shutdown order is frame generation first, then the swapchain context, then the proxy itself.
    destroyFrameGenContext();
    destroyPrepareCommands();

    if (m_swapchainContext != nullptr) {
      ffx::DispatchDescFrameGenerationSwapChainWaitForPresentsVK waitDesc;
      api.dispatch(&m_swapchainContext, &waitDesc.header);

      // The two counts are equal while nothing is generated, and the first is twice the second while every frame is.
      if (m_presentCount != nullptr && m_proxyFrameCount != 0) {
        Logger::info(str::format("FSR: The frame generation swapchain presented ", m_presentCount(m_swapchain),
                                 " images for ", m_proxyFrameCount, " rendered frames."));
      }

      api.destroyContext(&m_swapchainContext);
      m_swapchainContext = nullptr;
    }

    // Destroying the swapchain context leaves the proxy alive, and this is what destroys it.
    if (m_destroySwapchain != nullptr && m_swapchain != VK_NULL_HANDLE) {
      m_destroySwapchain(m_vkd->device(), m_swapchain, nullptr, nullptr);
    }

    for (const vk::PresenterImage& image : m_images) {
      m_vkd->vkDestroyImageView(m_vkd->device(), image.view, nullptr);
    }

    for (const vk::PresenterSync& semaphores : m_semaphores) {
      m_vkd->vkDestroySemaphore(m_vkd->device(), semaphores.acquire, nullptr);
      m_vkd->vkDestroySemaphore(m_vkd->device(), semaphores.present, nullptr);
    }

    m_images.clear();
    m_semaphores.clear();

    // The handle was FSR's proxy rather than a Vulkan swapchain, so the base class must not see it again.
    m_swapchain = VK_NULL_HANDLE;
    m_proxyActive = false;

    m_acquireNextImage = nullptr;
    m_queuePresent = nullptr;
    m_destroySwapchain = nullptr;
    m_presentCount = nullptr;
    m_proxyFrameCount = 0;

    m_lastInputs.reset();
    m_framesWithoutInputs = 0;
    m_hudlessMismatchFrames = 0;
    m_contextFailed = false;

    frameGen.releaseProxy(this);
  }

  bool DxvkFSRFGPresenter::createPrepareCommands() {
    const VkDevice device = m_vkd->device();

    VkCommandPoolCreateInfo poolInfo;
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.pNext = nullptr;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = m_dxvkDevice->queues().graphics.queueFamily;

    if (m_vkd->vkCreateCommandPool(device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS) {
      return false;
    }

    for (PrepareCommands& commands : m_prepareCommands) {
      VkCommandBufferAllocateInfo allocateInfo;
      allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      allocateInfo.pNext = nullptr;
      allocateInfo.commandPool = m_commandPool;
      allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      allocateInfo.commandBufferCount = 1;

      VkFenceCreateInfo fenceInfo;
      fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
      fenceInfo.pNext = nullptr;
      fenceInfo.flags = 0;

      VkSemaphoreCreateInfo semaphoreInfo;
      semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
      semaphoreInfo.pNext = nullptr;
      semaphoreInfo.flags = 0;

      if (m_vkd->vkAllocateCommandBuffers(device, &allocateInfo, &commands.commandBuffer) != VK_SUCCESS ||
          m_vkd->vkCreateFence(device, &fenceInfo, nullptr, &commands.fence) != VK_SUCCESS ||
          m_vkd->vkCreateSemaphore(device, &semaphoreInfo, nullptr, &commands.semaphore) != VK_SUCCESS) {
        return false;
      }
    }

    m_prepareCommandsIndex = 0;

    return true;
  }

  void DxvkFSRFGPresenter::waitForPrepareCommands() {
    for (PrepareCommands& commands : m_prepareCommands) {
      if (commands.pending) {
        m_vkd->vkWaitForFences(m_vkd->device(), 1, &commands.fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
        commands.pending = false;
      }

      commands.depth = nullptr;
      commands.motionVectors = nullptr;
    }
  }

  void DxvkFSRFGPresenter::destroyPrepareCommands() {
    const VkDevice device = m_vkd->device();

    waitForPrepareCommands();

    for (PrepareCommands& commands : m_prepareCommands) {
      m_vkd->vkDestroyFence(device, commands.fence, nullptr);
      m_vkd->vkDestroySemaphore(device, commands.semaphore, nullptr);

      commands = PrepareCommands();
    }

    // This frees the command buffers along with the pool.
    m_vkd->vkDestroyCommandPool(device, m_commandPool, nullptr);
    m_commandPool = VK_NULL_HANDLE;
  }

  void DxvkFSRFGPresenter::updateFrameGeneration(const DxvkFrameInterpolationInfo& frameInterpolationInfo, VkSemaphore& presentWait) {
    DxvkFSRFrameGen& frameGen = m_dxvkDevice->getCommon()->metaFSRFrameGen();
    const DxvkFSRFrameGen::Settings settings = frameGen.settings();

    // The inputs come from the frame itself.
    // A frame without inputs takes those of the last frame that had any.
    const DxvkFrameInterpolationInfo* pInputs = nullptr;
    bool reset = false;

    if (frameGen.hasFailed()) {
      m_lastInputs.reset();
    } else if (frameInterpolationInfo.valid()) {
      m_lastInputs = frameInterpolationInfo;
      m_framesWithoutInputs = 0;
      pInputs = &m_lastInputs;
      reset = frameInterpolationInfo.resetHistory;
    } else if (m_lastInputs.valid() && m_framesWithoutInputs < kMaxFramesWithoutInputs) {
      // The depth and motion vectors on hand describe an older frame, so nothing may be reprojected with them.
      m_framesWithoutInputs++;
      pInputs = &m_lastInputs;
      reset = true;
    } else {
      m_lastInputs.reset();
    }

    // The HUD-less image is only good for the frame it was captured in.
    // FSR finds the UI by comparing it with the presented image, so it has to cover the swapchain and hold the same colours.
    Rc<DxvkImageView> hudless;

    if (pInputs != nullptr && m_framesWithoutInputs == 0 && pInputs->hudless != nullptr && !m_presentAltersColor.load()) {
      const DxvkImageCreateInfo& hudlessInfo = pInputs->hudless->imageInfo();

      const bool coversSwapchain = hudlessInfo.extent.width == m_info.imageExtent.width
                                && hudlessInfo.extent.height == m_info.imageExtent.height;
      const int precisionGroup = formatPrecisionGroup(hudlessInfo.format);

      if (coversSwapchain && precisionGroup >= 0 && precisionGroup == formatPrecisionGroup(m_info.format.format) &&
          FsrApi::resourceState(pInputs->hudlessLayout) != 0) {
        hudless = pInputs->hudless;
      }
    }

    bool generate = pInputs != nullptr && ensureFrameGenContext(*pInputs, settings, hudless);

    if (m_frameGenContext == nullptr) {
      // The swapchain presents without frame generation until a context has configured it.
      m_generationEnabled.store(false);
      return;
    }

    // A context created for a HUD-less image in a format of its own keeps its history in that format.
    // It cannot take the swapchain's image in its place, so a frame without a HUD-less image is not generated from.
    if (hudless == nullptr && m_contextConfig.hudlessFormat != VK_FORMAT_UNDEFINED) {
      generate = false;
    }

    m_frameId++;

    const uint32_t flags = settings.debugOverlay ? kDebugOverlayFlags : 0;
    // This stays the same for the life of the context, since changing it makes FSR wait for its pending frames.
    const bool asyncWorkloads = (m_contextConfig.flags & FFX_FRAMEGENERATION_ENABLE_ASYNC_WORKLOAD_SUPPORT) != 0;
    const VkImageLayout hudlessLayout = pInputs != nullptr ? pInputs->hudlessLayout : VK_IMAGE_LAYOUT_UNDEFINED;

    if (!configureFrameGeneration(generate, asyncWorkloads, hudless, hudlessLayout, flags)) {
      generate = false;
    }

    // Generation that was off has no history worth keeping.
    if (generate && !m_generationEnabled.load()) {
      reset = true;
      ONCE(Logger::info("FSR: Frame generation is running."));
    }

    m_generationEnabled.store(generate);

    if (!generate) {
      m_resetNextFrame = true;
      return;
    }

    if (m_resetNextFrame) {
      m_resetNextFrame = false;
      reset = true;
    }

    m_resetThisFrame = reset;

    if (!dispatchPrepare(*pInputs, settings, flags, reset, presentWait)) {
      // The generation then reads what an older frame prepared.
      m_resetThisFrame = true;
      m_resetNextFrame = true;
    }
  }

  bool DxvkFSRFGPresenter::ensureFrameGenContext(const DxvkFrameInterpolationInfo& inputs,
                                                 const DxvkFSRFrameGen::Settings& settings,
                                                 Rc<DxvkImageView>& hudless) {
    const VkExtent3D renderExtent = inputs.motionVectors->imageInfo().extent;
    const fsr::DepthParameters depth = depthParameters(inputs);

    ContextConfig desired;
    // The game can render at more than the display resolution, so the context is sized for the larger of the two.
    desired.maxRenderExtent.width = std::max(m_info.imageExtent.width, renderExtent.width);
    desired.maxRenderExtent.height = std::max(m_info.imageExtent.height, renderExtent.height);
    desired.versionOverride = settings.versionOverride;

    if (m_queues.asyncCompute.handle != 0 && settings.allowAsyncWorkloads) {
      desired.flags |= FFX_FRAMEGENERATION_ENABLE_ASYNC_WORKLOAD_SUPPORT;
    }

    if (depth.inverted) {
      desired.flags |= FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED;
    }

    if (depth.infinite) {
      desired.flags |= FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE;
    }

    if (settings.enableDebugChecker) {
      desired.flags |= FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING;
    }

    // A HUD-less image in the swapchain's own format needs no declaration.
    if (hudless != nullptr && hudless->imageInfo().format != m_info.format.format) {
      desired.hudlessFormat = hudless->imageInfo().format;
    }

    const bool contextFits = m_frameGenContext != nullptr
                          && m_contextConfig.flags == desired.flags
                          && m_contextConfig.versionOverride == desired.versionOverride
                          && m_contextConfig.maxRenderExtent.width >= desired.maxRenderExtent.width
                          && m_contextConfig.maxRenderExtent.height >= desired.maxRenderExtent.height;

    if (contextFits && m_contextConfig.hudlessFormat == desired.hudlessFormat) {
      m_hudlessMismatchFrames = 0;
      return true;
    }

    if (contextFits && ++m_hudlessMismatchFrames < kHudlessSwitchFrames) {
      // The context stays as it is until the HUD-less image has settled, and this frame goes without one.
      hudless = nullptr;

      return true;
    }

    m_hudlessMismatchFrames = 0;

    const bool failedBefore = m_contextFailed
                           && m_failedContextConfig.flags == desired.flags
                           && m_failedContextConfig.hudlessFormat == desired.hudlessFormat
                           && m_failedContextConfig.versionOverride == desired.versionOverride
                           && m_failedContextConfig.maxRenderExtent.width == desired.maxRenderExtent.width
                           && m_failedContextConfig.maxRenderExtent.height == desired.maxRenderExtent.height;

    if (failedBefore) {
      return false;
    }

    destroyFrameGenContext();

    if (!createFrameGenContext(desired)) {
      m_contextFailed = true;
      m_failedContextConfig = desired;

      m_dxvkDevice->getCommon()->metaFSRFrameGen().notifyFailure("The frame generation context could not be created.");

      return false;
    }

    m_contextFailed = false;

    return true;
  }

  bool DxvkFSRFGPresenter::createFrameGenContext(const ContextConfig& config) {
    FsrApi& api = FsrApi::get();
    DxvkFSRFrameGen& frameGen = m_dxvkDevice->getCommon()->metaFSRFrameGen();

    // The swapchain presents sRGB colour, so the HDR flag stays off.
    ffx::CreateContextDescFrameGeneration createDesc;
    createDesc.flags = config.flags;
    createDesc.displaySize = { m_info.imageExtent.width, m_info.imageExtent.height };
    createDesc.maxRenderSize = { config.maxRenderExtent.width, config.maxRenderExtent.height };
    createDesc.backBufferFormat = ffxApiGetSurfaceFormatVK(m_info.format.format);

    // The runtime keeps a context from using anything newer than the version it was built against, unless told.
    ffx::CreateContextDescFrameGenerationVersion versionDesc;
    versionDesc.version = FFX_FRAMEGENERATION_VERSION;

    ffx::CreateBackendVKDesc backendDesc;
    backendDesc.deviceContext = FsrApi::deviceContext(*m_dxvkDevice);

    ffx::CreateContextDescFrameGenerationHudless hudlessDesc;
    hudlessDesc.hudlessBackBufferFormat = ffxApiGetSurfaceFormatVK(config.hudlessFormat);

    ffx::CreateContextDescOverrideVersion overrideDesc;

    // Zero leaves the choice to the runtime, which picks the newest version the GPU supports.
    const uint64_t requestedVersion = api.findVersion(FsrEffect::FrameGeneration, *m_dxvkDevice, config.versionOverride);

    if (!config.versionOverride.empty() && requestedVersion == 0) {
      Logger::warn(str::format("FSR: The runtime does not offer frame generation version \"", config.versionOverride,
                               "\" on this GPU. The runtime picks the version instead."));
    }

    // When the first choice cannot be created, every version the runtime lists is tried in turn, newest first.
    std::vector<uint64_t> candidates = { requestedVersion };

    for (const FsrVersion& version : api.versions(FsrEffect::FrameGeneration, *m_dxvkDevice)) {
      if (version.id != requestedVersion) {
        candidates.push_back(version.id);
      }
    }

    ffxReturnCode_t result = FFX_API_RETURN_NO_PROVIDER;

    for (const uint64_t candidate : candidates) {
      ffxApiHeader* pLast = &backendDesc.header;

      createDesc.header.pNext = &versionDesc.header;
      versionDesc.header.pNext = &backendDesc.header;
      backendDesc.header.pNext = nullptr;

      if (config.hudlessFormat != VK_FORMAT_UNDEFINED) {
        hudlessDesc.header.pNext = nullptr;
        pLast->pNext = &hudlessDesc.header;
        pLast = &hudlessDesc.header;
      }

      if (candidate != 0) {
        overrideDesc.versionId = candidate;
        overrideDesc.header.pNext = nullptr;
        pLast->pNext = &overrideDesc.header;
      }

      m_frameGenContext = nullptr;
      result = api.createContext(FsrEffect::FrameGeneration, &m_frameGenContext, &createDesc.header);

      if (result == FFX_API_RETURN_OK && m_frameGenContext != nullptr) {
        break;
      }

      m_frameGenContext = nullptr;
    }

    if (m_frameGenContext == nullptr) {
      Logger::err(str::format("FSR: Failed to create the frame generation context: ", FsrApi::returnCodeName(result), "."));
      return false;
    }

    api.routeMessages(FsrEffect::FrameGeneration, &m_frameGenContext, (config.flags & FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING) != 0);

    const std::string version = api.contextVersion(&m_frameGenContext);
    frameGen.notifyContextVersion(version);

    m_contextConfig = config;
    // A new context counts frames from the start.
    m_frameId = 0;
    m_resetNextFrame = true;

    Logger::info(str::format("FSR: Created the frame generation context: ",
                             m_info.imageExtent.width, "x", m_info.imageExtent.height,
                             ", FSR ", version.empty() ? "of an unknown version" : version,
                             (config.flags & FFX_FRAMEGENERATION_ENABLE_ASYNC_WORKLOAD_SUPPORT) != 0 ? ", on its own compute queue" : ", on the rendering queue",
                             (config.flags & FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED) != 0 ? ", inverted depth" : "",
                             (config.flags & FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE) != 0 ? ", infinite far plane" : "",
                             config.hudlessFormat != VK_FORMAT_UNDEFINED ? ", HUD-less image in its own format" : "",
                             (config.flags & FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING) != 0 ? ", debug checker" : "", "."));

    return true;
  }

  void DxvkFSRFGPresenter::destroyFrameGenContext() {
    if (m_frameGenContext == nullptr) {
      return;
    }

    DxvkFSRFrameGen& frameGen = m_dxvkDevice->getCommon()->metaFSRFrameGen();

    // Switching generation off makes the swapchain wait for the frames that still use the context,
    // and stops it from calling back into a context that is about to go away.
    ffx::ConfigureDescFrameGeneration config;
    config.swapChain = m_swapchain;
    config.frameGenerationEnabled = false;
    config.frameID = m_frameId;

    FsrApi::get().configure(&m_frameGenContext, &config.header);

    // The preparation commands write to the context's images.
    waitForPrepareCommands();

    FsrApi::get().destroyContext(&m_frameGenContext);
    m_frameGenContext = nullptr;

    m_generationEnabled.store(false);
    m_resetNextFrame = true;

    frameGen.notifyGenerating(false);
    frameGen.notifyContextVersion(std::string());
  }

  bool DxvkFSRFGPresenter::configureFrameGeneration(bool enabled, bool allowAsyncWorkloads,
                                                    const Rc<DxvkImageView>& hudless, VkImageLayout hudlessLayout,
                                                    uint32_t flags) {
    ffx::ConfigureDescFrameGeneration config;
    config.swapChain = m_swapchain;
    config.presentCallback = nullptr;
    config.presentCallbackUserContext = nullptr;
    config.frameGenerationCallback = &frameGenerationCallback;
    config.frameGenerationCallbackUserContext = this;
    config.frameGenerationEnabled = enabled;
    config.allowAsyncWorkloads = allowAsyncWorkloads;
    config.flags = flags;
    config.onlyPresentGenerated = false;
    // The whole swapchain image holds the scene, with no bars around it.
    config.generationRect.left = 0;
    config.generationRect.top = 0;
    config.generationRect.width = int32_t(m_info.imageExtent.width);
    config.generationRect.height = int32_t(m_info.imageExtent.height);
    config.frameID = m_frameId;

    if (enabled && hudless != nullptr) {
      config.HUDLessColor = FsrApi::resource(hudless->image(), FsrApi::resourceState(hudlessLayout), false);
    }

    const ffxReturnCode_t result = FsrApi::get().configure(&m_frameGenContext, &config.header);

    if (result != FFX_API_RETURN_OK) {
      ONCE(Logger::err(str::format("FSR: Failed to configure frame generation: ", FsrApi::returnCodeName(result), ".")));
      return false;
    }

    return true;
  }

  bool DxvkFSRFGPresenter::dispatchPrepare(const DxvkFrameInterpolationInfo& inputs,
                                           const DxvkFSRFrameGen::Settings& settings,
                                           uint32_t flags,
                                           bool reset,
                                           VkSemaphore& presentWait) {
    const VkDevice device = m_vkd->device();

    const uint32_t depthState = FsrApi::resourceState(inputs.depthLayout);
    const uint32_t motionVectorState = FsrApi::resourceState(inputs.motionVectorsLayout);

    if (depthState == 0 || motionVectorState == 0) {
      ONCE(Logger::err("FSR: A frame generation input is in an image layout the FSR runtime cannot read."));
      return false;
    }

    PrepareCommands& commands = m_prepareCommands[m_prepareCommandsIndex];
    m_prepareCommandsIndex = (m_prepareCommandsIndex + 1) % uint32_t(m_prepareCommands.size());

    // These commands ran several frames ago, so this does not wait in practice.
    if (commands.pending) {
      m_vkd->vkWaitForFences(device, 1, &commands.fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
      commands.pending = false;
    }

    m_vkd->vkResetFences(device, 1, &commands.fence);
    m_vkd->vkResetCommandBuffer(commands.commandBuffer, 0);

    VkCommandBufferBeginInfo beginInfo;
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.pNext = nullptr;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    beginInfo.pInheritanceInfo = nullptr;

    if (m_vkd->vkBeginCommandBuffer(commands.commandBuffer, &beginInfo) != VK_SUCCESS) {
      return false;
    }

    const VkExtent3D renderExtent = inputs.motionVectors->imageInfo().extent;
    const fsr::DepthParameters depth = depthParameters(inputs);

    float jitter[2] = { 0.0f, 0.0f };
    inputs.camera.getJittering(jitter);

    float cameraNear = 0.0f;
    float cameraFar = 0.0f;
    fsr::cameraPlanes(depth, cameraNear, cameraFar);

    ffx::DispatchDescFrameGenerationPrepareV2 prepareDesc;
    prepareDesc.frameID = m_frameId;
    prepareDesc.flags = flags;
    prepareDesc.commandList = commands.commandBuffer;
    prepareDesc.renderSize = { renderExtent.width, renderExtent.height };
    prepareDesc.jitterOffset = { jitter[0], jitter[1] };
    // The motion vectors are already in render pixels.
    prepareDesc.motionVectorScale = { 1.0f, 1.0f };
    prepareDesc.frameTimeDelta = fsr::frameTimeMs(inputs.frameTimeMs);
    prepareDesc.reset = reset;
    prepareDesc.cameraNear = cameraNear;
    prepareDesc.cameraFar = cameraFar;
    prepareDesc.cameraFovAngleVertical = inputs.camera.getFov();
    prepareDesc.viewSpaceToMetersFactor = settings.viewSpaceToMetersFactor;
    prepareDesc.depth = FsrApi::resource(inputs.depth->image(), depthState, false);
    prepareDesc.motionVectors = FsrApi::resource(inputs.motionVectors->image(), motionVectorState, false);

    copyVector(inputs.cameraPosition, prepareDesc.cameraPosition);
    copyDirection(inputs.camera.getUp(), prepareDesc.cameraUp);
    copyDirection(inputs.camera.getRight(), prepareDesc.cameraRight);
    copyDirection(inputs.camera.getDirection(), prepareDesc.cameraForward);

    const ffxReturnCode_t result = FsrApi::get().dispatch(&m_frameGenContext, &prepareDesc.header);

    if (m_vkd->vkEndCommandBuffer(commands.commandBuffer) != VK_SUCCESS || result != FFX_API_RETURN_OK) {
      ONCE(Logger::err(str::format("FSR: Failed to prepare a frame for frame generation: ", FsrApi::returnCodeName(result), ".")));
      return false;
    }

    // The generation that the present starts must not run before these commands have.
    // FSR orders the two through the semaphore the present waits for, so the commands take over that role.
    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    VkSubmitInfo submitInfo;
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = nullptr;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &presentWait;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commands.commandBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &commands.semaphore;

    if (m_vkd->vkQueueSubmit(m_dxvkDevice->queues().graphics.queueHandle, 1, &submitInfo, commands.fence) != VK_SUCCESS) {
      return false;
    }

    commands.pending = true;
    commands.depth = inputs.depth;
    commands.motionVectors = inputs.motionVectors;

    presentWait = commands.semaphore;

    return true;
  }

  uint32_t DxvkFSRFGPresenter::frameGenerationCallback(ffxDispatchDescFrameGeneration* pParams, void* pUserContext) {
    DxvkFSRFGPresenter* pPresenter = static_cast<DxvkFSRFGPresenter*>(pUserContext);

    // The presenting thread holds m_proxyMutex here, so this must not take it.
    if (pPresenter->m_resetThisFrame) {
      pParams->reset = true;
    }

    return FsrApi::get().dispatch(&pPresenter->m_frameGenContext, &pParams->header);
  }

} // namespace dxvk
