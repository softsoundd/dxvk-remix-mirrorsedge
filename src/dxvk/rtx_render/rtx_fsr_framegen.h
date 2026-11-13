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
#pragma once

#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "../dxvk_include.h"
#include "../dxvk_queue.h"
#include "../../vulkan/vulkan_presenter.h"
#include "rtx_common_object.h"
#include "rtx_fsr_helpers.h"
#include "rtx_options.h"

// Declared by the FSR headers, which only the implementation includes.
typedef void* ffxContext;
struct ffxDispatchDescFrameGeneration;

namespace dxvk {

  class DxvkDevice;

  /**
   * \brief AMD FSR Frame Generation
   *
   * Holds the options, and the state the menus show.
   * The work itself happens in DxvkFSRFGPresenter, which the D3D9 swapchain creates while this is active.
   */
  class DxvkFSRFrameGen : public CommonDeviceObject {
  public:
    // Holds the options the presenter reads.
    struct Settings {
      bool allowAsyncWorkloads = true;
      bool enableDebugChecker = false;
      bool debugOverlay = false;
      float viewSpaceToMetersFactor = 0.01f;
      std::string versionOverride;
    };

    explicit DxvkFSRFrameGen(DxvkDevice* device);
    ~DxvkFSRFrameGen() override;

    // Returns whether the FSR runtime's frame generation is installed next to this DLL.
    // This reads the disk only, so it is safe to call before a device exists.
    static bool hasRuntimeFiles();

    // Returns whether this device can run FSR Frame Generation.
    // That requires the runtime, and device queues that nothing else submits to.
    bool isSupported() const;

    // Returns why isSupported is false.
    // Returns an empty string while it is true.
    const char* unsupportedReason() const;

    // Returns whether the swapchain should present through FSR Frame Generation.
    // A failure turns this off until the option is switched off and on again.
    bool isActive() const;

    // Returns whether frame generation was switched off by a failure.
    bool hasFailed() const {
      return m_failed.load();
    }

    // Picks the device queues the frame generation swapchain presents with.
    // surface is the surface it will present to.
    // A null surface asks whether the queue families can present at all.
    fsr::FrameGenQueues selectQueues(VkSurfaceKHR surface) const;

    // Copies the options into the snapshot settings() returns.
    // The presenter runs on the submission thread, where an option that holds a string is not safe to read.
    void updateSettings();

    Settings settings() const;

    // Only one swapchain at a time can present through frame generation, since the queues exist once.
    // These hand that role to a presenter and take it back.
    bool acquireProxy(const void* pOwner);
    void releaseProxy(const void* pOwner);

    // The presenter reports through these, from the submission thread.
    void notifyFailure(const std::string& reason);
    void notifyContextVersion(const std::string& version);
    void notifyGenerating(bool generating) {
      m_generating.store(generating);
    }

    // Returns whether the last presented frame was followed by a generated one.
    bool isGenerating() const {
      return m_generating.load();
    }

    // Returns the name of the frame generation version in use.
    // Returns an empty string while there is none.
    // Any thread may call this.
    std::string activeVersion() const;

    // Returns the names of the frame generation versions the runtime offers on this device, newest first.
    // The first call loads the runtime. Any thread may call this.
    std::vector<std::string> versionNames() const;

    static void onEnableChanged(DxvkDevice* device);

    RTX_OPTION_ARGS("rtx.fsrfg", bool, enable, false,
                    "Enables AMD FSR Frame Generation, which presents a generated frame between every two rendered ones.\n"
                    "This raises the displayed frame rate at the cost of slightly more latency.\n"
                    "DLSS Frame Generation takes precedence when both are enabled.",
                    args.environment = "RTX_FSRFG_ENABLE",
                    args.flags = RtxOptionFlags::UserSetting,
                    args.onChangeCallback = &onEnableChanged);
    RTX_OPTION("rtx.fsrfg", bool, allowAsyncWorkloads, true,
               "Lets FSR Frame Generation run on a compute queue of its own, so it overlaps with the rendering of the next frame.\n"
               "Disabling this runs it on the rendering queue instead, which uses less memory.\n"
               "Changing it recreates frame generation.");
    RTX_OPTION("rtx.fsrfg", bool, enableDebugChecker, false,
               "Makes FSR Frame Generation validate its inputs every frame, and writes what it finds to the log.\n"
               "This is a development aid, and changing it recreates frame generation.");
    RTX_OPTION("rtx.fsrfg", bool, debugOverlay, false,
               "Draws FSR Frame Generation's debug overlay onto the generated frames.\n"
               "Moving lines mark tearing and pacing, and an indicator flashes whenever the history is reset.");
    RTX_OPTION("rtx.fsrfg", std::string, frameGenerationVersion, "",
               "Names the FSR Frame Generation version to use, as the developer menu lists them, for example \"3.1.6\".\n"
               "Leave this empty to let the runtime pick the newest version the GPU supports.\n"
               "A named version also keeps the runtime from using newer frame generation supplied by the graphics driver.");

  private:
    // Set when the frame generation swapchain or context could not be created.
    std::atomic<bool> m_failed = { false };
    std::atomic<bool> m_generating = { false };

    mutable std::once_flag m_queueSupportOnce;
    mutable bool m_queuesSupported = false;

    // Guards what the submission thread and the menus' thread both touch.
    mutable std::mutex m_stateMutex;
    Settings m_settings;
    std::string m_activeVersion;
    const void* m_pProxyOwner = nullptr;
  };

  /**
   * \brief Presenter for AMD FSR Frame Generation
   *
   * Hands the window's swapchain over to FSR, which returns a proxy swapchain to render into.
   * FSR generates a frame for each one presented to the proxy, and paces both onto the real swapchain from threads of its own.
   * While there is no proxy, this presents like the base presenter.
   *
   * AMD requires every call that touches the proxy or the frame generation context to be serialized, which one mutex does here.
   * The device has to be idle wherever the proxy is created or destroyed.
   */
  class DxvkFSRFGPresenter : public vk::Presenter {
  public:
    DxvkFSRFGPresenter(DxvkDevice* device,
                       HWND window,
                       const Rc<vk::InstanceFn>& vki,
                       const Rc<vk::DeviceFn>& vkd,
                       vk::PresenterDevice presenterDevice,
                       const vk::PresenterDesc& desc);

    ~DxvkFSRFGPresenter() override;

    VkResult acquireNextImage(vk::PresenterSync& sync, uint32_t& index, bool isDlfgPresenting = false) override;

    VkResult presentImage(std::atomic<VkResult>* presentStatus,
                          const DxvkPresentInfo& presentInfo,
                          const DxvkFrameInterpolationInfo& frameInterpolationInfo,
                          std::uint32_t acquiredImageIndex = std::uint32_t(-1),
                          bool isDlfgPresenting = false,
                          VkSetPresentConfigNV* presentMetering = nullptr) override;

    // Replaces the swapchain with a new proxy.
    // Replaces it with a plain swapchain while frame generation cannot run.
    VkResult recreateSwapChain(const vk::PresenterDesc& desc) override;

    // Waits until FSR has presented every frame it was given.
    void synchronize() override;

    // FSR samples the proxy's images when they are presented.
    VkImageLayout presentLayout() const override;

    // Returns how many frames reach the display for each one the game renders, for the frame rate display.
    uint32_t getPresentFrameCount() const {
      return m_generationEnabled.load() ? 2 : 1;
    }

    // Tells the presenter that the swapchain's owner changes the colours on the way to the swapchain, as a gamma ramp does.
    // FSR finds the UI by comparing the HUD-less image with the presented one, which only works while their colours match.
    void setPresentAltersColor(bool altersColor) {
      m_presentAltersColor.store(altersColor);
    }

  private:
    enum class ProxyResult {
      Created,
      // The surface has no area, so there is nothing to present to.
      NoSurfaceArea,
      Failed,
    };

    // Describes what the frame generation context was created for.
    // A change in any of these needs a new context.
    struct ContextConfig {
      VkExtent2D maxRenderExtent = { 0, 0 };
      uint32_t flags = 0;
      // Holds the format declared for the HUD-less image.
      // Holds VK_FORMAT_UNDEFINED when that image shares the swapchain's format.
      VkFormat hudlessFormat = VK_FORMAT_UNDEFINED;
      std::string versionOverride;
    };

    struct PrepareCommands {
      VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
      VkFence fence = VK_NULL_HANDLE;
      // Signalled when the commands have run, which is what the present then waits for.
      VkSemaphore semaphore = VK_NULL_HANDLE;
      bool pending = false;

      // Keep the images the commands read alive until the commands have run.
      Rc<DxvkImageView> depth;
      Rc<DxvkImageView> motionVectors;
    };

    ProxyResult createProxy(const vk::PresenterDesc& desc);
    void destroyProxy();

    bool createPrepareCommands();
    void destroyPrepareCommands();
    void waitForPrepareCommands();

    // Configures frame generation for the frame about to be presented, and prepares its depth and motion vectors.
    // presentWait is the semaphore the present has to wait for, which the preparation replaces with its own.
    void updateFrameGeneration(const DxvkFrameInterpolationInfo& frameInterpolationInfo, VkSemaphore& presentWait);

    // Makes the frame generation context fit the frame's inputs.
    // hudless is cleared when the context cannot take the frame's HUD-less image.
    bool ensureFrameGenContext(const DxvkFrameInterpolationInfo& inputs,
                               const DxvkFSRFrameGen::Settings& settings,
                               Rc<DxvkImageView>& hudless);
    bool createFrameGenContext(const ContextConfig& config);
    void destroyFrameGenContext();
    bool configureFrameGeneration(bool enabled, bool allowAsyncWorkloads,
                                  const Rc<DxvkImageView>& hudless, VkImageLayout hudlessLayout,
                                  uint32_t flags);
    bool dispatchPrepare(const DxvkFrameInterpolationInfo& inputs, const DxvkFSRFrameGen::Settings& settings,
                         uint32_t flags, bool reset, VkSemaphore& presentWait);

    // FSR calls this from inside the present, on the thread that presents.
    static uint32_t frameGenerationCallback(ffxDispatchDescFrameGeneration* pParams, void* pUserContext);

    DxvkDevice* m_dxvkDevice;

    // Serializes every use of the proxy and the frame generation context.
    dxvk::mutex m_proxyMutex;

    // Set while the swapchain the base class holds is FSR's proxy.
    bool m_proxyActive = false;

    ffxContext m_swapchainContext = nullptr;
    ffxContext m_frameGenContext = nullptr;

    // Hold FSR's replacements for the Vulkan swapchain functions.
    PFN_vkAcquireNextImageKHR m_acquireNextImage = nullptr;
    PFN_vkQueuePresentKHR m_queuePresent = nullptr;
    void (*m_destroySwapchain)(VkDevice, VkSwapchainKHR, const VkAllocationCallbacks*, void*) = nullptr;
    // Returns how many images FSR has presented to the window through the proxy.
    uint64_t (*m_presentCount)(VkSwapchainKHR) = nullptr;

    // Counts the frames presented to the proxy, for the summary logged when it is destroyed.
    uint64_t m_proxyFrameCount = 0;

    fsr::FrameGenQueues m_queues;

    ContextConfig m_contextConfig;
    // Holds the config whose context could not be created, so that it is not retried every frame.
    ContextConfig m_failedContextConfig;
    bool m_contextFailed = false;

    // FSR resets frame generation when the frame number does not go up by exactly one.
    uint64_t m_frameId = 0;
    // Written by the submission thread, and read for the frame rate display.
    std::atomic<bool> m_generationEnabled = { false };
    // Written by the swapchain's owner, and read by the submission thread.
    std::atomic<bool> m_presentAltersColor = { false };

    // Holds the inputs of the last frame that had any.
    // A frame without inputs is prepared from these, with a history reset.
    DxvkFrameInterpolationInfo m_lastInputs;
    // Counts the frames in a row that brought no inputs of their own.
    uint32_t m_framesWithoutInputs = 0;
    // Counts the frames in a row whose HUD-less image did not fit the context.
    uint32_t m_hudlessMismatchFrames = 0;
    // Makes the next generated frame reset the history.
    bool m_resetNextFrame = true;
    // Set by the present for its callback.
    bool m_resetThisFrame = false;

    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    std::array<PrepareCommands, 4> m_prepareCommands;
    uint32_t m_prepareCommandsIndex = 0;
  };

} // namespace dxvk
