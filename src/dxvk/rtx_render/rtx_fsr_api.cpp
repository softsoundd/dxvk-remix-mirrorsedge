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
#include "rtx_fsr_api.h"

#include <algorithm>
#include <cstring>
#include <unordered_set>

#include "../dxvk_adapter.h"
#include "../dxvk_device.h"

namespace dxvk {

  namespace {

    constexpr char kLoaderDll[] = "amd_fidelityfx_loader_vk.dll";
    constexpr char kUpscalerDll[] = "amd_fidelityfx_upscaler_vk.dll";
    constexpr char kFrameGenerationDll[] = "amd_fidelityfx_framegeneration_vk.dll";

    // The runtime repeats most of its messages every frame, so each distinct one is logged once.
    // Logging stops after this many distinct ones, which bounds what a message that differs every frame can add.
    constexpr size_t kMaxLoggedMessages = 256;

    const char* effectDll(FsrEffect effect) {
      return effect == FsrEffect::Upscaler ? kUpscalerDll : kFrameGenerationDll;
    }

    std::wstring runtimePath(const char* pDllName) {
      return str::tows((env::getDllDirectory() + "\\" + pDllName).c_str());
    }

    bool runtimeFileExists(const char* pDllName) {
#if defined(_M_ARM64)
      // AMD only ships the runtime for x64.
      return false;
#else
      const DWORD attributes = GetFileAttributesW(runtimePath(pDllName).c_str());

      return attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
#endif
    }

    HMODULE loadRuntimeDll(const char* pDllName) {
      // The full path keeps Windows from picking up a copy of the DLL from anywhere else on its search path.
      const HMODULE module = LoadLibraryExW(runtimePath(pDllName).c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);

      if (module == nullptr) {
        Logger::warn(str::format("FSR: Failed to load ", pDllName, ", error ", GetLastError(), "."));
      }

      return module;
    }

    uint64_t createDescType(FsrEffect effect) {
      switch (effect) {
      case FsrEffect::Upscaler:                 return FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
      case FsrEffect::FrameGeneration:          return FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
      case FsrEffect::FrameGenerationSwapchain: return FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK;
      case FsrEffect::Count:                    break;
      }

      return 0;
    }

    void logRuntimeMessage(uint32_t type, const wchar_t* pMessage) {
      if (pMessage == nullptr) {
        return;
      }

      const std::string text = str::fromws(pMessage);

      bool reachedLimit = false;

      {
        static std::mutex s_mutex;
        static std::unordered_set<std::string> s_loggedMessages;

        std::lock_guard<std::mutex> lock(s_mutex);

        if (s_loggedMessages.size() >= kMaxLoggedMessages || !s_loggedMessages.insert(text).second) {
          return;
        }

        reachedLimit = s_loggedMessages.size() == kMaxLoggedMessages;
      }

      if (type == FFX_API_MESSAGE_TYPE_ERROR) {
        Logger::err(str::format("FSR: ", text));
      } else {
        Logger::warn(str::format("FSR: ", text));
      }

      if (reachedLimit) {
        Logger::warn("FSR: Further messages from the FSR runtime are not logged.");
      }
    }

    // The runtime looks up some functions under the names of the extensions that introduced them.
    // A device that gets those functions from core Vulkan does not have to enable the extensions,
    // and then only answers to the core names.
    VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL getDeviceProcAddr(VkDevice device, const char* pName) {
      PFN_vkVoidFunction function = vkGetDeviceProcAddr(device, pName);

      if (function != nullptr || pName == nullptr) {
        return function;
      }

      constexpr char kSuffix[] = "KHR";
      constexpr size_t kSuffixLength = sizeof(kSuffix) - 1;

      const size_t length = std::strlen(pName);

      if (length > kSuffixLength && std::strcmp(pName + length - kSuffixLength, kSuffix) == 0) {
        const std::string coreName(pName, length - kSuffixLength);

        function = vkGetDeviceProcAddr(device, coreName.c_str());
      }

      return function;
    }

  } // namespace

  FsrApi& FsrApi::get() {
    static FsrApi s_instance;

    return s_instance;
  }

  bool FsrApi::hasRuntimeFiles(FsrEffect effect) {
    static const bool s_hasLoader = runtimeFileExists(kLoaderDll);
    static const bool s_hasUpscaler = runtimeFileExists(kUpscalerDll);
    static const bool s_hasFrameGeneration = runtimeFileExists(kFrameGenerationDll);

    return s_hasLoader && (effect == FsrEffect::Upscaler ? s_hasUpscaler : s_hasFrameGeneration);
  }

  FsrApi::Module FsrApi::moduleOf(FsrEffect effect) {
    return effect == FsrEffect::Upscaler ? Module::Upscaler : Module::FrameGeneration;
  }

  bool FsrApi::isAvailable(FsrEffect effect) {
    if (!hasRuntimeFiles(effect)) {
      return false;
    }

    std::lock_guard<std::mutex> lock(m_loadMutex);

    const ModuleState& module = m_modules[uint32_t(moduleOf(effect))];

    const bool loaderFailed = m_loader.attempted && !m_loader.loaded;
    const bool moduleFailed = module.attempted && !module.loaded;

    return !loaderFailed && !moduleFailed;
  }

  bool FsrApi::ensureLoaded(FsrEffect effect) {
    if (!hasRuntimeFiles(effect)) {
      return false;
    }

    std::lock_guard<std::mutex> lock(m_loadMutex);

    ModuleState& module = m_modules[uint32_t(moduleOf(effect))];

    // The loader forwards each call to the DLL of the effect it belongs to, and uses a DLL that is already loaded.
    // Loading the effect's DLL first therefore settles which copy the loader ends up with.
    if (!module.attempted) {
      module.attempted = true;
      module.loaded = loadRuntimeDll(effectDll(effect)) != nullptr;
    }

    if (!module.loaded) {
      return false;
    }

    if (!m_loader.attempted) {
      m_loader.attempted = true;
      m_loader.loaded = loadLoader();
    }

    return m_loader.loaded;
  }

  bool FsrApi::loadLoader() {
    const HMODULE loader = loadRuntimeDll(kLoaderDll);

    if (loader == nullptr) {
      return false;
    }

    const PfnFfxCreateContext createContext = reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(loader, "ffxCreateContext"));
    const PfnFfxDestroyContext destroyContext = reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(loader, "ffxDestroyContext"));
    const PfnFfxConfigure configure = reinterpret_cast<PfnFfxConfigure>(GetProcAddress(loader, "ffxConfigure"));
    const PfnFfxQuery query = reinterpret_cast<PfnFfxQuery>(GetProcAddress(loader, "ffxQuery"));
    const PfnFfxDispatch dispatch = reinterpret_cast<PfnFfxDispatch>(GetProcAddress(loader, "ffxDispatch"));

    if (createContext == nullptr || destroyContext == nullptr || configure == nullptr || query == nullptr || dispatch == nullptr) {
      Logger::warn(str::format("FSR: ", kLoaderDll, " does not export the FSR API."));
      return false;
    }

    {
      std::unique_lock<std::shared_mutex> lock(m_apiMutex);

      m_createContext = createContext;
      m_destroyContext = destroyContext;
      m_configure = configure;
      m_query = query;
      m_dispatch = dispatch;
    }

    Logger::info(str::format("FSR: Loaded the AMD FSR runtime from ", env::getDllDirectory(), "."));

    return true;
  }

  ffxReturnCode_t FsrApi::createContext(FsrEffect effect, ffxContext* pContext, ffxCreateContextDescHeader* pDesc) {
    if (!ensureLoaded(effect)) {
      return FFX_API_RETURN_NO_PROVIDER;
    }

    std::unique_lock<std::shared_mutex> lock(m_apiMutex);

    return m_createContext(pContext, pDesc, nullptr);
  }

  ffxReturnCode_t FsrApi::destroyContext(ffxContext* pContext) {
    std::unique_lock<std::shared_mutex> lock(m_apiMutex);

    if (m_destroyContext == nullptr) {
      return FFX_API_RETURN_NO_PROVIDER;
    }

    return m_destroyContext(pContext, nullptr);
  }

  ffxReturnCode_t FsrApi::configure(ffxContext* pContext, const ffxConfigureDescHeader* pDesc) {
    std::shared_lock<std::shared_mutex> lock(m_apiMutex);

    if (m_configure == nullptr) {
      return FFX_API_RETURN_NO_PROVIDER;
    }

    return m_configure(pContext, pDesc);
  }

  ffxReturnCode_t FsrApi::query(ffxContext* pContext, ffxQueryDescHeader* pDesc) {
    // A query without a context reaches into the runtime's shared state, like creating a context does.
    std::shared_lock<std::shared_mutex> sharedLock(m_apiMutex, std::defer_lock);
    std::unique_lock<std::shared_mutex> exclusiveLock(m_apiMutex, std::defer_lock);

    if (pContext == nullptr) {
      exclusiveLock.lock();
    } else {
      sharedLock.lock();
    }

    if (m_query == nullptr) {
      return FFX_API_RETURN_NO_PROVIDER;
    }

    return m_query(pContext, pDesc);
  }

  ffxReturnCode_t FsrApi::dispatch(ffxContext* pContext, const ffxDispatchDescHeader* pDesc) {
    std::shared_lock<std::shared_mutex> lock(m_apiMutex);

    if (m_dispatch == nullptr) {
      return FFX_API_RETURN_NO_PROVIDER;
    }

    return m_dispatch(pContext, pDesc);
  }

  const std::vector<FsrVersion>& FsrApi::versions(FsrEffect effect, const DxvkDevice& device) {
    std::lock_guard<std::mutex> lock(m_versionMutex);

    VersionList& list = m_versions[uint32_t(effect)];

    if (list.queried) {
      return list.versions;
    }

    list.queried = true;

    if (!ensureLoaded(effect)) {
      return list.versions;
    }

    VkDeviceContext vkDeviceContext = deviceContext(device);

    ffx::QueryDescGetVersions versionQuery;
    versionQuery.createDescType = createDescType(effect);
    versionQuery.device = &vkDeviceContext;

    uint64_t versionCount = 0;
    versionQuery.outputCount = &versionCount;

    if (query(nullptr, &versionQuery.header) != FFX_API_RETURN_OK || versionCount == 0) {
      return list.versions;
    }

    std::vector<uint64_t> ids(versionCount);
    std::vector<const char*> names(versionCount);
    versionQuery.versionIds = ids.data();
    versionQuery.versionNames = names.data();

    if (query(nullptr, &versionQuery.header) != FFX_API_RETURN_OK) {
      return list.versions;
    }

    versionCount = std::min<uint64_t>(versionCount, ids.size());

    for (uint64_t i = 0; i < versionCount; i++) {
      FsrVersion version;
      version.id = ids[i];
      version.name = names[i] != nullptr ? names[i] : "";

      list.versions.push_back(version);
    }

    return list.versions;
  }

  uint64_t FsrApi::findVersion(FsrEffect effect, const DxvkDevice& device, const std::string& name) {
    if (name.empty()) {
      return 0;
    }

    for (const FsrVersion& version : versions(effect, device)) {
      if (version.name == name) {
        return version.id;
      }
    }

    return 0;
  }

  std::string FsrApi::contextVersion(ffxContext* pContext) {
    if (pContext == nullptr || *pContext == nullptr) {
      return std::string();
    }

    ffx::InitHelper<ffxQueryGetProviderVersion> versionQuery;

    if (query(pContext, &versionQuery.header) != FFX_API_RETURN_OK || versionQuery.versionName == nullptr) {
      return std::string();
    }

    return versionQuery.versionName;
  }

  void FsrApi::routeMessages(FsrEffect effect, ffxContext* pContext, bool includeWarnings) {
    if (pContext == nullptr || *pContext == nullptr) {
      return;
    }

    ffx::ConfigureDescGlobalDebug debugDesc;
    debugDesc.effectId = effect == FsrEffect::Upscaler ? FFX_API_EFFECT_ID_UPSCALE : FFX_API_EFFECT_ID_FRAMEGENERATION;
    debugDesc.fpMessage = &logRuntimeMessage;
    debugDesc.debugLevel = includeWarnings
      ? FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_WARNINGS
      : FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_ERRORS;

    configure(pContext, &debugDesc.header);
  }

  ffxApiMessage FsrApi::messageCallback() {
    return &logRuntimeMessage;
  }

  VkDeviceContext FsrApi::deviceContext(const DxvkDevice& device) {
    VkDeviceContext context = {};
    context.vkDevice = device.handle();
    context.vkPhysicalDevice = device.adapter()->handle();
    context.vkDeviceProcAddr = &getDeviceProcAddr;

    return context;
  }

  uint32_t FsrApi::resourceState(VkImageLayout layout) {
    switch (layout) {
    case VK_IMAGE_LAYOUT_GENERAL:                  return FFX_API_RESOURCE_STATE_UNORDERED_ACCESS;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: return FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ;
    default:                                       return 0;
    }
  }

  FfxApiResource FsrApi::resource(const Rc<DxvkImage>& image, uint32_t state, bool writable, VkFormat format) {
    FfxApiResource resource = {};

    if (image == nullptr) {
      return resource;
    }

    const DxvkImageCreateInfo& info = image->info();
    const VkFormat viewFormat = format != VK_FORMAT_UNDEFINED ? format : info.format;

    uint32_t usage = FFX_API_RESOURCE_USAGE_READ_ONLY;

    if (ffxApiIsDepthFormat(viewFormat)) {
      usage |= FFX_API_RESOURCE_USAGE_DEPTHTARGET;
    }

    if (ffxApiIsStencilFormat(viewFormat)) {
      usage |= FFX_API_RESOURCE_USAGE_STENCILTARGET;
    }

    // The runtime creates storage views for a resource with this flag, so only outputs carry it.
    if (writable) {
      usage |= FFX_API_RESOURCE_USAGE_UAV;
    }

    resource.resource = reinterpret_cast<void*>(image->handle());
    resource.state = state;
    resource.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    resource.description.format = ffxApiGetSurfaceFormatVK(viewFormat);
    resource.description.width = info.extent.width;
    resource.description.height = info.extent.height;
    resource.description.depth = info.numLayers;
    resource.description.mipCount = info.mipLevels;
    resource.description.flags = FFX_API_RESOURCE_FLAGS_NONE;
    resource.description.usage = usage;

    return resource;
  }

  const char* FsrApi::returnCodeName(ffxReturnCode_t code) {
    switch (code) {
    case FFX_API_RETURN_OK:                               return "FFX_API_RETURN_OK";
    case FFX_API_RETURN_ERROR:                            return "FFX_API_RETURN_ERROR";
    case FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE:           return "FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE";
    case FFX_API_RETURN_ERROR_RUNTIME_ERROR:              return "FFX_API_RETURN_ERROR_RUNTIME_ERROR";
    case FFX_API_RETURN_NO_PROVIDER:                      return "FFX_API_RETURN_NO_PROVIDER";
    case FFX_API_RETURN_ERROR_MEMORY:                     return "FFX_API_RETURN_ERROR_MEMORY";
    case FFX_API_RETURN_ERROR_PARAMETER:                  return "FFX_API_RETURN_ERROR_PARAMETER";
    case FFX_API_RETURN_PROVIDER_NO_SUPPORT_NEW_DESCTYPE: return "FFX_API_RETURN_PROVIDER_NO_SUPPORT_NEW_DESCTYPE";
    default:                                              return "unknown FSR error";
    }
  }

} // namespace dxvk
