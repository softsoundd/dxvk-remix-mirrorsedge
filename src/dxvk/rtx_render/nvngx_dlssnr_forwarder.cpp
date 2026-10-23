#include <windows.h>
#include <stdio.h>

#include <vulkan/vulkan.h>
#include <nvsdk_ngx_vk.h>

#include "nvngx_dlssnr_forwarder.h"

namespace {

char g_lastError[512] = "";

void recordError(const char* message) {
  snprintf(g_lastError, sizeof(g_lastError), "%s", message != nullptr ? message : "");
}

HMODULE loadSnippetModule() {
  HMODULE self = nullptr;
  if (!GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&loadSnippetModule),
        &self)) {
    recordError("GetModuleHandleEx failed for the DLSS-NR forwarder.");
    return nullptr;
  }

  wchar_t path[MAX_PATH];
  const DWORD length = GetModuleFileNameW(self, path, MAX_PATH);
  if (length == 0 || length >= MAX_PATH) {
    recordError("The DLSS-NR forwarder path is unavailable.");
    return nullptr;
  }

  wchar_t* slash = wcsrchr(path, L'\\');
  if (slash == nullptr) {
    recordError("The DLSS-NR forwarder path has no directory.");
    return nullptr;
  }
  if (wcscpy_s(slash + 1, MAX_PATH - static_cast<size_t>(slash + 1 - path), L"nvngx_dlssnr.dll") != 0) {
    recordError("The DLSS-NR snippet path does not fit.");
    return nullptr;
  }

  HMODULE module = LoadLibraryW(path);
  if (module == nullptr) {
    snprintf(g_lastError, sizeof(g_lastError), "LoadLibrary(nvngx_dlssnr.dll) failed (%lu).", GetLastError());
  }
  return module;
}

// This is the snippet's Init_Ext2 argument list.
// NVSDK_NGX_VULKAN_Init on the NGX core takes different arguments.
using PFN_SnippetInit = NVSDK_NGX_Result(NVSDK_CONV*)(
  unsigned long long,
  const wchar_t*,
  VkInstance,
  VkPhysicalDevice,
  VkDevice,
  PFN_vkGetInstanceProcAddr,
  PFN_vkGetDeviceProcAddr,
  NVSDK_NGX_Version,
  const NVSDK_NGX_Parameter*);

using PFN_SnippetCreate = NVSDK_NGX_Result(NVSDK_CONV*)(
  VkDevice,
  VkCommandBuffer,
  NVSDK_NGX_Feature,
  const NVSDK_NGX_Parameter*,
  NVSDK_NGX_Handle**);

using PFN_SnippetEvaluate = NVSDK_NGX_Result(NVSDK_CONV*)(
  VkCommandBuffer,
  const NVSDK_NGX_Handle*,
  const NVSDK_NGX_Parameter*,
  PFN_NVSDK_NGX_ProgressCallback);

using PFN_SnippetRelease = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle*);

struct SnippetApi {
  HMODULE module = nullptr;
  PFN_SnippetInit init = nullptr;
  PFN_SnippetCreate create = nullptr;
  PFN_SnippetEvaluate evaluate = nullptr;
  PFN_SnippetRelease release = nullptr;
  bool attempted = false;
};

SnippetApi& snippetApi() {
  static SnippetApi api;
  if (api.attempted) {
    return api;
  }
  api.attempted = true;
  api.module = loadSnippetModule();
  if (api.module == nullptr) {
    return api;
  }

  api.init = reinterpret_cast<PFN_SnippetInit>(GetProcAddress(api.module, "NVSDK_NGX_VULKAN_Init_Ext2"));
  api.create = reinterpret_cast<PFN_SnippetCreate>(GetProcAddress(api.module, "NVSDK_NGX_VULKAN_CreateFeature1"));
  api.evaluate = reinterpret_cast<PFN_SnippetEvaluate>(GetProcAddress(api.module, "NVSDK_NGX_VULKAN_EvaluateFeature"));
  api.release = reinterpret_cast<PFN_SnippetRelease>(GetProcAddress(api.module, "NVSDK_NGX_VULKAN_ReleaseFeature"));
  if (api.init == nullptr || api.create == nullptr || api.evaluate == nullptr || api.release == nullptr) {
    recordError("nvngx_dlssnr.dll is missing a Vulkan entry point.");
    api.init = nullptr;
    api.create = nullptr;
    api.evaluate = nullptr;
    api.release = nullptr;
  }
  return api;
}

} // namespace

extern "C" __declspec(dllexport) const char* __cdecl dlssnr_forwarder_last_error_message(void) {
  return g_lastError;
}

extern "C" __declspec(dllexport) int __cdecl dlssnr_vulkan_init(
    unsigned long long applicationId,
    const wchar_t* applicationDataPath,
    void* instance,
    void* physicalDevice,
    void* device,
    void* getInstanceProcAddr,
    void* getDeviceProcAddr,
    unsigned int sdkVersion,
    const void* parameters) {
  const SnippetApi& api = snippetApi();
  if (api.init == nullptr) {
    return static_cast<int>(NVSDK_NGX_Result_FAIL_FeatureNotFound);
  }

  const NVSDK_NGX_Result result = api.init(
    applicationId,
    applicationDataPath,
    reinterpret_cast<VkInstance>(instance),
    reinterpret_cast<VkPhysicalDevice>(physicalDevice),
    reinterpret_cast<VkDevice>(device),
    reinterpret_cast<PFN_vkGetInstanceProcAddr>(getInstanceProcAddr),
    reinterpret_cast<PFN_vkGetDeviceProcAddr>(getDeviceProcAddr),
    static_cast<NVSDK_NGX_Version>(sdkVersion),
    reinterpret_cast<const NVSDK_NGX_Parameter*>(parameters));
  if (NVSDK_NGX_FAILED(result)) {
    snprintf(g_lastError, sizeof(g_lastError), "NVSDK_NGX_VULKAN_Init_Ext2 returned 0x%08x", static_cast<unsigned int>(result));
  } else {
    g_lastError[0] = '\0';
  }
  return static_cast<int>(result);
}

extern "C" __declspec(dllexport) int __cdecl dlssnr_vulkan_create(
    void* device,
    void* commandBuffer,
    int featureId,
    const void* parameters,
    void** outHandle) {
  const SnippetApi& api = snippetApi();
  if (api.create == nullptr) {
    return static_cast<int>(NVSDK_NGX_Result_FAIL_FeatureNotFound);
  }

  NVSDK_NGX_Handle* handle = nullptr;
  const NVSDK_NGX_Result result = api.create(
    reinterpret_cast<VkDevice>(device),
    reinterpret_cast<VkCommandBuffer>(commandBuffer),
    static_cast<NVSDK_NGX_Feature>(featureId),
    reinterpret_cast<const NVSDK_NGX_Parameter*>(parameters),
    &handle);
  if (outHandle != nullptr) {
    *outHandle = handle;
  }
  if (NVSDK_NGX_FAILED(result)) {
    snprintf(g_lastError, sizeof(g_lastError), "NVSDK_NGX_VULKAN_CreateFeature1 returned 0x%08x", static_cast<unsigned int>(result));
  } else {
    g_lastError[0] = '\0';
  }
  return static_cast<int>(result);
}

extern "C" __declspec(dllexport) int __cdecl dlssnr_vulkan_evaluate(
    void* commandBuffer,
    const void* featureHandle,
    const void* parameters) {
  const SnippetApi& api = snippetApi();
  if (api.evaluate == nullptr) {
    return static_cast<int>(NVSDK_NGX_Result_FAIL_FeatureNotFound);
  }

  const NVSDK_NGX_Result result = api.evaluate(
    reinterpret_cast<VkCommandBuffer>(commandBuffer),
    reinterpret_cast<const NVSDK_NGX_Handle*>(featureHandle),
    reinterpret_cast<const NVSDK_NGX_Parameter*>(parameters),
    nullptr);
  if (NVSDK_NGX_FAILED(result)) {
    snprintf(g_lastError, sizeof(g_lastError), "NVSDK_NGX_VULKAN_EvaluateFeature returned 0x%08x", static_cast<unsigned int>(result));
  } else {
    g_lastError[0] = '\0';
  }
  return static_cast<int>(result);
}

extern "C" __declspec(dllexport) int __cdecl dlssnr_vulkan_release(void* featureHandle) {
  const SnippetApi& api = snippetApi();
  if (api.release == nullptr) {
    return static_cast<int>(NVSDK_NGX_Result_FAIL_FeatureNotFound);
  }

  const NVSDK_NGX_Result result = api.release(reinterpret_cast<NVSDK_NGX_Handle*>(featureHandle));
  if (NVSDK_NGX_FAILED(result)) {
    snprintf(g_lastError, sizeof(g_lastError), "NVSDK_NGX_VULKAN_ReleaseFeature returned 0x%08x", static_cast<unsigned int>(result));
  } else {
    g_lastError[0] = '\0';
  }
  return static_cast<int>(result);
}
