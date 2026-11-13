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
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

#include "../dxvk_include.h"
#include "../dxvk_image.h"

// The FSR headers declare the five API entry points for linking against AMD's import library.
// Nothing here links against that library, so those declarations and the ffx:: call wrappers around them stay unused.
// Every call goes through FsrApi, which resolves the entry points from AMD's loader DLL at runtime.
// These headers include the Vulkan headers, so they follow the DXVK ones, which set those up for Windows first.
#include "FidelityFX-SDK/Kits/FidelityFX/api/include/ffx_api.hpp"
#include "FidelityFX-SDK/Kits/FidelityFX/api/include/vk/ffx_api_vk.hpp"
#include "FidelityFX-SDK/Kits/FidelityFX/upscalers/include/ffx_upscale.hpp"
#include "FidelityFX-SDK/Kits/FidelityFX/framegeneration/include/ffx_framegeneration.hpp"
#include "FidelityFX-SDK/Kits/FidelityFX/framegeneration/include/vk/ffx_api_framegeneration_vk.hpp"

namespace dxvk {

  class DxvkDevice;

  enum class FsrEffect : uint32_t {
    Upscaler = 0,
    FrameGeneration,
    FrameGenerationSwapchain,

    Count
  };

  // Names one version of an effect that the runtime can create on this device.
  struct FsrVersion {
    uint64_t id = 0;
    std::string name;
  };

  // Loads AMD's FSR runtime from the directory this DLL sits in, and forwards the FSR API to it.
  // The runtime is a loader DLL plus one DLL per effect.
  // An effect's DLL loads when the effect is first used, and stays loaded until the process exits.
  class FsrApi {
  public:
    static FsrApi& get();

    // Returns whether the DLLs for the effect are on disk, without loading them.
    static bool hasRuntimeFiles(FsrEffect effect);

    // Returns whether the effect can be used.
    // That is the case while its DLLs are on disk and have not failed to load.
    bool isAvailable(FsrEffect effect);

    ffxReturnCode_t createContext(FsrEffect effect, ffxContext* pContext, ffxCreateContextDescHeader* pDesc);
    ffxReturnCode_t destroyContext(ffxContext* pContext);
    ffxReturnCode_t configure(ffxContext* pContext, const ffxConfigureDescHeader* pDesc);
    ffxReturnCode_t query(ffxContext* pContext, ffxQueryDescHeader* pDesc);
    ffxReturnCode_t dispatch(ffxContext* pContext, const ffxDispatchDescHeader* pDesc);

    // Returns the versions of the effect the runtime offers, newest first.
    // The list is read from the runtime once and stays valid until the process exits.
    const std::vector<FsrVersion>& versions(FsrEffect effect, const DxvkDevice& device);

    // Returns the id of the version with the given name.
    // Returns zero when the runtime does not offer that version.
    uint64_t findVersion(FsrEffect effect, const DxvkDevice& device, const std::string& name);

    // Returns the name of the version the runtime picked for the context.
    // Returns an empty string when the runtime does not report one.
    std::string contextVersion(ffxContext* pContext);

    // Sends the messages of the context's effect to the log.
    // Errors always arrive, and warnings join them when the effect's debug checker is on.
    void routeMessages(FsrEffect effect, ffxContext* pContext, bool includeWarnings);

    // The upscaler takes this callback at context creation, ahead of routeMessages.
    static ffxApiMessage messageCallback();

    // Describes the device to the runtime's Vulkan backend.
    static VkDeviceContext deviceContext(const DxvkDevice& device);

    // Returns the FfxApiResourceState for an image in the given layout.
    // Returns zero for a layout the runtime cannot work with.
    // The runtime moves an image to the layout each of its passes needs, and back to this state when it is done.
    static uint32_t resourceState(VkImageLayout layout);

    // Describes an image to the runtime.
    // state is the FfxApiResourceState matching the layout the image is in.
    // writable marks an image the runtime renders to.
    // format overrides the image's own format when it is not VK_FORMAT_UNDEFINED.
    static FfxApiResource resource(const Rc<DxvkImage>& image, uint32_t state, bool writable, VkFormat format = VK_FORMAT_UNDEFINED);

    static const char* returnCodeName(ffxReturnCode_t code);

  private:
    FsrApi() = default;

    // The upscaler has a DLL of its own, and frame generation shares one with its swapchain.
    enum class Module : uint32_t {
      Upscaler = 0,
      FrameGeneration,

      Count
    };

    struct ModuleState {
      bool attempted = false;
      bool loaded = false;
    };

    struct VersionList {
      bool queried = false;
      std::vector<FsrVersion> versions;
    };

    static Module moduleOf(FsrEffect effect);

    bool ensureLoaded(FsrEffect effect);
    bool loadLoader();

    std::mutex m_loadMutex;
    ModuleState m_loader;
    ModuleState m_modules[uint32_t(Module::Count)];

    // AMD documents which calls on one context need external locking, and its loader does no locking of its own.
    // Creating and destroying contexts and querying versions touch state shared by the whole runtime,
    // so those take this lock exclusively while the per-context calls share it.
    std::shared_mutex m_apiMutex;

    // The entry points are written once under an exclusive m_apiMutex, and read under at least a shared one.
    PfnFfxCreateContext m_createContext = nullptr;
    PfnFfxDestroyContext m_destroyContext = nullptr;
    PfnFfxConfigure m_configure = nullptr;
    PfnFfxQuery m_query = nullptr;
    PfnFfxDispatch m_dispatch = nullptr;

    std::mutex m_versionMutex;
    std::array<VersionList, uint32_t(FsrEffect::Count)> m_versions;
  };

} // namespace dxvk
