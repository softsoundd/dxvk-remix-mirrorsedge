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

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>

#define WIN32_NO_STATUS
#include <windows.h>
#include <ntstatus.h>
#undef WIN32_NO_STATUS
#include <Winternl.h>
#include <d3dkmthk.h>
#include <d3dkmdt.h>
#include <softpub.h>
#include <wintrust.h>
#pragma comment(lib, "wintrust")

#include "rtx_ngx_wrapper.h"
#include "rtx_matrix_helpers.h"

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_vk.h>

#include <vulkan/vulkan.h>
#include <nvsdk_ngx_helpers_dlssg.h>
#include <nvsdk_ngx_helpers_dlssg_vk.h>
#include <nvsdk_ngx_defs_dlssg.h>

#include <nvsdk_ngx_defs_dlssnr.h>
#include <nvsdk_ngx_helpers_dlssnr_vk.h>
#include "nvngx_dlssnr_forwarder.h"

#include "rtx_resources.h"
#include "rtx_semaphore.h"

#include <dxvk_device.h>

#define NGX_CALL(...) ngxCall([&] { return (__VA_ARGS__); })

namespace {
  std::atomic<int> s_dlssNeuralRenderingStatus { -1 };
  std::mutex s_ngxMutex;

  // NGX requires client-enforced thread safety across render and frame-generation threads.
  template <typename Callback>
  auto ngxCall(Callback&& callback) {
    std::lock_guard<std::mutex> lock(s_ngxMutex);
    return callback();
  }
}

namespace dxvk
{
  namespace {
    NVSDK_NGX_Result queryDlssNeuralRenderingVramAllocation(
        NVSDK_NGX_Parameter* pParameters,
        unsigned long long* pVramAllocationBytes) {
      // Invoke the stats callback directly to avoid the generic helper's D3D header dependency.
      void* pCallback = nullptr;
      NVSDK_NGX_Result result = NVSDK_NGX_Parameter_GetVoidPointer(
        pParameters, NVSDK_NGX_Parameter_DLSSNRGetStatsCallback, &pCallback);
      if (NVSDK_NGX_FAILED(result)) {
        return result;
      }

      if (pCallback == nullptr) {
        return NVSDK_NGX_Result_FAIL_OutOfDate;
      }

      const auto pStatsCallback = reinterpret_cast<PFN_NVSDK_NGX_DLSSNR_GetStatsCallback>(pCallback);
      result = pStatsCallback(pParameters);
      if (NVSDK_NGX_FAILED(result)) {
        return result;
      }

      return NVSDK_NGX_Parameter_GetULL(pParameters, NVSDK_NGX_Parameter_SizeInBytes, pVramAllocationBytes);
    }

    std::string resultToString(NVSDK_NGX_Result result) {
      char buf[1024];
      snprintf(buf, sizeof(buf), "(code: 0x%08x, info: %ls)", result, GetNGXResultAsString(result));
      buf[sizeof(buf) - 1] = '\0';
      return std::string(buf);
    }

    NVSDK_NGX_Resource_VK ViewToResourceVK(const Rc<DxvkImageView>& view, bool isUAV) {
      VkImageView imageView = view->handle();
      auto info = view->image()->info();
      VkFormat format = info.format;
      VkImage image = view->imageHandle();
      VkImageSubresourceRange subresourceRange = view->subresources();
      return NVSDK_NGX_Create_ImageView_Resource_VK(imageView, image, subresourceRange, format, info.extent.width, info.extent.height, isUAV);
    }

    NVSDK_NGX_Resource_VK TextureToResourceVK(const Resources::Resource* tex, bool isUAV/*, nvrhi::TextureSubresourceSet subresources*/) {
      if (tex == nullptr || tex->view == nullptr || tex->image == nullptr)
        return {};

      return ViewToResourceVK(tex->view, isUAV);
    }

    // A snippet with an empty certificate directory is unsigned, so its body is not hashed.
    // When a certificate is present, it is verified in full.
    // Revocation is skipped, so a signed snippet still verifies while offline.
    bool localDlssNrSnippetLacksValidSignature() {
      const std::wstring path = str::tows((env::getDllDirectory() + "\\nvngx_dlssnr.dll").c_str());
      const DWORD attributes = GetFileAttributesW(path.c_str());
      if (attributes == INVALID_FILE_ATTRIBUTES || (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
        return false;
      }

      HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
      DWORD certificateSize = 0;
      bool readCertificateSize = false;
      if (file != INVALID_HANDLE_VALUE) {
        uint8_t bytes[4096];
        DWORD read = 0;
        if (ReadFile(file, bytes, sizeof(bytes), &read, nullptr) && read >= sizeof(IMAGE_DOS_HEADER)) {
          const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(bytes);
          const size_t ntOffset = dos->e_magic == IMAGE_DOS_SIGNATURE && dos->e_lfanew > 0
            ? static_cast<size_t>(dos->e_lfanew) : 0;
          if (ntOffset != 0 && ntOffset <= read && read - ntOffset >= sizeof(IMAGE_NT_HEADERS64)) {
            const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(bytes + ntOffset);
            if (nt->Signature == IMAGE_NT_SIGNATURE && nt->OptionalHeader.Magic == IMAGE_NT_OPTIONAL_HDR64_MAGIC) {
              certificateSize = nt->OptionalHeader.NumberOfRvaAndSizes <= IMAGE_DIRECTORY_ENTRY_SECURITY
                ? 0 : nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_SECURITY].Size;
              readCertificateSize = true;
            }
          }
        }
        CloseHandle(file);
      }
      if (readCertificateSize && certificateSize == 0) {
        return true;
      }

      WINTRUST_FILE_INFO fileInfo = {};
      fileInfo.cbStruct = sizeof(fileInfo);
      fileInfo.pcwszFilePath = path.c_str();

      WINTRUST_DATA trustData = {};
      trustData.cbStruct = sizeof(trustData);
      trustData.dwUIChoice = WTD_UI_NONE;
      trustData.fdwRevocationChecks = WTD_REVOKE_NONE;
      trustData.dwUnionChoice = WTD_CHOICE_FILE;
      trustData.pFile = &fileInfo;
      trustData.dwStateAction = WTD_STATEACTION_VERIFY;

      GUID action = WINTRUST_ACTION_GENERIC_VERIFY_V2;
      const LONG status = WinVerifyTrust(static_cast<HWND>(nullptr), &action, &trustData);
      trustData.dwStateAction = WTD_STATEACTION_CLOSE;
      WinVerifyTrust(static_cast<HWND>(nullptr), &action, &trustData);
      return status != ERROR_SUCCESS;
    }
  }

  void NVSDK_CONV NVSDK_NGX_AppLogCallback(const char* message, NVSDK_NGX_Logging_Level loggingLevel, NVSDK_NGX_Feature sourceComponent) {
    static_cast<void>(loggingLevel);
    static_cast<void>(sourceComponent);

    Logger::info(str::format("DLSS Message: ", message));
  }

  bool NGXContext::initialize() {
    ScopedCpuProfileZone();

    // Early out if the NGX Context has already been initialized

    if (m_initialized) {
      return true;
    }

    s_dlssNeuralRenderingStatus.store(-1, std::memory_order_release);

    // Reset DLSS feature support flags.
    // Note: This is done here so that if initialization fails before feature checking the support will be false as expected.

    m_supportsDLSS = false;
    m_supportsDlssNeuralRendering = false;
    m_dlssNeuralRenderingSupportChecked = false;
    m_dlssNeuralRenderingDirectLoad = false;

    const std::string exePath = env::getExePath();
    const std::string exeFolder = exePath.substr(0, exePath.find_last_of("\\/"));
    const auto logFolder = str::tows(exeFolder.c_str());
    
    NVSDK_NGX_Result result = NVSDK_NGX_Result_Fail;

    VkDevice vkDevice = m_device->handle();
    auto adapter = m_device->adapter();
    VkPhysicalDevice vkPhysicalDevice = adapter->handle();
    auto instance = m_device->instance();
    VkInstance vkInstance = instance->handle();

    // Kit may load HdRemix through a symbolic link and report its module path in extended-length form. The current
    // NGX loader does not discover feature DLLs from that path form, so convert drive and UNC paths before passing it on.
    std::string featureDirectory = env::getDllDirectory();
    const bool hasExtendedPathPrefix = featureDirectory.compare(0, 4, R"(\\?\)") == 0;
    const bool isExtendedUncPath = featureDirectory.size() >= 8
      && hasExtendedPathPrefix
      && _strnicmp(featureDirectory.c_str() + 4, R"(UNC\)", 4) == 0;
    const bool isExtendedDrivePath = featureDirectory.size() >= 7
      && hasExtendedPathPrefix
      && ((featureDirectory[4] >= 'A' && featureDirectory[4] <= 'Z')
        || (featureDirectory[4] >= 'a' && featureDirectory[4] <= 'z'))
      && featureDirectory[5] == ':'
      && (featureDirectory[6] == '\\' || featureDirectory[6] == '/');
    if (isExtendedUncPath) {
      featureDirectory.replace(0, 8, R"(\\)");
    } else if (isExtendedDrivePath) {
      featureDirectory.erase(0, 4);
    } else if (hasExtendedPathPrefix) {
      Logger::warn(str::format("Unsupported extended-length NGX feature path: ", featureDirectory));
    }
    const std::wstring featureDirectoryWide = str::tows(featureDirectory.c_str());
    const wchar_t* featureSearchPath = featureDirectoryWide.c_str();

    NVSDK_NGX_FeatureCommonInfo featureCommonInfo{};
    if (!featureDirectoryWide.empty()) {
      featureCommonInfo.PathListInfo.Path = &featureSearchPath;
      featureCommonInfo.PathListInfo.Length = 1;
    }

    // Note: Enable DLSS logging for debugging in debug mode. Note this will disable all other DLSS logging sinks to ensure all logging
    // goes through the DXVK logging system.
#ifndef NDEBUG
    featureCommonInfo.LoggingInfo.LoggingCallback = &NVSDK_NGX_AppLogCallback;
    featureCommonInfo.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;
    featureCommonInfo.LoggingInfo.DisableOtherLoggingSinks = true;
#endif

    result = NGX_CALL(NVSDK_NGX_VULKAN_Init(
      RtxOptions::applicationId(), logFolder.c_str(),
      vkInstance, vkPhysicalDevice, vkDevice,
      nullptr, nullptr,
      &featureCommonInfo));

    if (NVSDK_NGX_FAILED(result)) {
      if (result == NVSDK_NGX_Result_FAIL_FeatureNotSupported || result == NVSDK_NGX_Result_FAIL_PlatformError) {
        Logger::err(str::format("NVIDIA NGX is not available on this hardware/platform: ", resultToString(result)));
      } else {
        Logger::err(str::format("Failed to initialize NGX: ", resultToString(result)));
      }

      s_dlssNeuralRenderingStatus.store(0, std::memory_order_release);
      return false;
    }

    NVSDK_NGX_Parameter* tempParams = nullptr;
    result = NGX_CALL(NVSDK_NGX_VULKAN_GetCapabilityParameters(&tempParams));
    if (NVSDK_NGX_FAILED(result)) {
      Logger::err(str::format("NVSDK_NGX_VULKAN_GetCapabilityParameters failed: ", resultToString(result)));
      s_dlssNeuralRenderingStatus.store(0, std::memory_order_release);
      return false;
    }

    m_supportsDlssNeuralRendering = checkDlssNeuralRenderingSupport(tempParams);
    m_dlssNeuralRenderingSupportChecked = true;
    
#if defined(NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver)        \
    && defined (NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor) \
    && defined (NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor)

    // If NGX Successfully initialized then it should set those flags in return
    int needsUpdatedDriver = 0;
    if (!NVSDK_NGX_FAILED(tempParams->Get(NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsUpdatedDriver)) && needsUpdatedDriver) {
      std::string message = "NVIDIA DLSS cannot be loaded due to outdated driver.";
      unsigned int majorVersion = 0;
      unsigned int minorVersion = 0;
      if (!NVSDK_NGX_FAILED(tempParams->Get(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &majorVersion)) &&
        !NVSDK_NGX_FAILED(tempParams->Get(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &minorVersion))) {
        message += "Minimum driver version required: " + std::to_string(majorVersion) + "." + std::to_string(minorVersion);
      }
      Logger::err(message);
      m_supportsDlssNeuralRendering = false;
      m_dlssNeuralRenderingDirectLoad = false;
      s_dlssNeuralRenderingStatus.store(0, std::memory_order_release);
      return false;
    }
#endif

    int dlssAvailable = 0;
    result = tempParams->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &dlssAvailable);
    if (NVSDK_NGX_FAILED(result) || !dlssAvailable) {
      int featureInitResult = NVSDK_NGX_Result_Fail;
      const NVSDK_NGX_Result featureInitResultQuery = tempParams->Get(
        NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult,
        &featureInitResult);
      Logger::err(str::format(
        "NVIDIA DLSS not available on this hardware/platform. Available: ", dlssAvailable,
        ", availability query: ", resultToString(result),
        ", feature initialization: ", resultToString(static_cast<NVSDK_NGX_Result>(featureInitResult)),
        ", feature initialization query: ", resultToString(featureInitResultQuery)));
      m_supportsDlssNeuralRendering = false;
      m_dlssNeuralRenderingDirectLoad = false;
      s_dlssNeuralRenderingStatus.store(0, std::memory_order_release);
      return false;
    }

    s_dlssNeuralRenderingStatus.store(m_supportsDlssNeuralRendering ? 1 : 0, std::memory_order_release);
    m_supportsDLSS = checkDLSSSupport(tempParams);
    checkDLFGSupport(tempParams);

    NGX_CALL(NVSDK_NGX_VULKAN_DestroyParameters(tempParams));
    m_initialized = true;
    return true;
  }

  NGXContext::NGXContext(DxvkDevice* device)
    : m_device(device) {
  }

  void NGXContext::shutdown() {
    s_dlssNeuralRenderingStatus.store(-1, std::memory_order_release);
    if (m_initialized) {
      NGX_CALL(NVSDK_NGX_VULKAN_Shutdown1(m_device->handle()));
      m_initialized = false;
    }
  }

  std::unique_ptr<NGXDLSSContext> NGXContext::createDLSSContext() {
    if (!m_initialized) {
      if (!initialize()) {
        return nullptr;
      }
    }

    if (!supportsDLSS()) {
      Logger::err("NVIDIA DLSS not supported");
      return nullptr;
    }

    return std::make_unique<NGXDLSSContext>(m_device);
  }

  std::unique_ptr<NGXDLFGContext> NGXContext::createDLFGContext() {
    if (!m_initialized) {
      if (!initialize()) {
        return nullptr;
      }
    }

    if (!supportsDLFG()) {
      Logger::err("NVIDIA DLFG not supported");
      return nullptr;
    }

    return std::make_unique<NGXDLFGContext>(m_device);
  }

  std::unique_ptr<NGXNeuralRenderingContext> NGXContext::createDlssNeuralRenderingContext() {
    if (!m_initialized) {
      if (!initialize()) {
        return nullptr;
      }
    }

    if (!supportsDlssNeuralRendering()) {
      Logger::err("NVIDIA DLSS-NR not supported");
      return nullptr;
    }

    return std::make_unique<NGXNeuralRenderingContext>(m_device);
  }

  bool NGXContext::checkDLSSSupport(NVSDK_NGX_Parameter* params) {
    NVSDK_NGX_Result result;

    int needsUpdatedDriver = 0;
    if (NVSDK_NGX_FAILED(params->Get(NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsUpdatedDriver))) {
      Logger::err("NVIDIA DLSS failed to initialize");
      return false;
    }

    if (needsUpdatedDriver) {
      std::string message = "NVIDIA DLSS cannot be loaded due to outdated driver.";
      unsigned int majorVersion = 0;
      unsigned int minorVersion = 0;
      if (!NVSDK_NGX_FAILED(params->Get(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &majorVersion)) &&
        !NVSDK_NGX_FAILED(params->Get(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &minorVersion))) {
        message += "Minimum driver version required: " + std::to_string(majorVersion) + "." + std::to_string(minorVersion);
      }

      Logger::err(message);
      return false;
    }

    int dlssAvailable = 0;
    result = params->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &dlssAvailable);
    if (NVSDK_NGX_FAILED(result) || !dlssAvailable) {
      Logger::warn(str::format("NVIDIA DLSS not available on this hardware/platform: ", resultToString(result)));
      return false;
    }

    return true;
  }

  bool NGXContext::checkDlssNeuralRenderingSupport(NVSDK_NGX_Parameter* params) {
    int needsUpdatedDriver = 0;
    NVSDK_NGX_Result result = params->Get(NVSDK_NGX_Parameter_DLSSNR_NeedsUpdatedDriver, &needsUpdatedDriver);
    if (NVSDK_NGX_FAILED(result)) {
      Logger::warn(str::format("NVIDIA DLSS-NR support query failed: ", resultToString(result)));
      return false;
    }

    if (needsUpdatedDriver) {
      std::string message = "NVIDIA DLSS-NR cannot be loaded due to an outdated driver.";
      unsigned int majorVersion = 0;
      unsigned int minorVersion = 0;
      if (!NVSDK_NGX_FAILED(params->Get(NVSDK_NGX_Parameter_DLSSNR_MinDriverVersionMajor, &majorVersion)) &&
          !NVSDK_NGX_FAILED(params->Get(NVSDK_NGX_Parameter_DLSSNR_MinDriverVersionMinor, &minorVersion))) {
        message += " Minimum driver version required: " + std::to_string(majorVersion) + "." + std::to_string(minorVersion) + ".";
      }
      Logger::warn(message);
      return false;
    }

    int dlssNeuralRenderingAvailable = 0;
    result = params->Get(NVSDK_NGX_Parameter_DLSSNR_Available, &dlssNeuralRenderingAvailable);
    if (NVSDK_NGX_FAILED(result)) {
      Logger::warn(str::format(
        "NVIDIA DLSS-NR availability query failed: ", resultToString(result)));
      return false;
    }

    if (dlssNeuralRenderingAvailable) {
      return true;
    }

    int featureInitResult = NVSDK_NGX_Result_Fail;
    const NVSDK_NGX_Result featureInitResultQuery = params->Get(
      NVSDK_NGX_Parameter_DLSSNR_FeatureInitResult,
      &featureInitResult);

    // The core will not map an unsigned snippet, so this bit stays clear.
    if (localDlssNrSnippetLacksValidSignature()) {
      m_dlssNeuralRenderingDirectLoad = true;
      Logger::warn(str::format(
        "NVIDIA DLSS-NR capability reports unavailable. Available: ", dlssNeuralRenderingAvailable,
        ", feature initialization: ", resultToString(static_cast<NVSDK_NGX_Result>(featureInitResult)),
        ", feature initialization query: ", resultToString(featureInitResultQuery),
        ". The local snippet has no valid signature, so it will be loaded directly."));
      return true;
    }

    Logger::warn(str::format(
      "NVIDIA DLSS-NR not available on this hardware/platform. Available: ", dlssNeuralRenderingAvailable,
      ", availability query: ", resultToString(result),
      ", feature initialization: ", resultToString(static_cast<NVSDK_NGX_Result>(featureInitResult)),
      ", feature initialization query: ", resultToString(featureInitResultQuery)));
    return false;
  }

  static bool checkHardwareSchedulingEnabled(DxvkDevice* device) {
    // enumerate adapters, find the right one
    D3DKMT_ENUMADAPTERS2 enumAdapters;
    enumAdapters.NumAdapters = 0;
    enumAdapters.pAdapters = nullptr;

    NTSTATUS ret;
    ret = D3DKMTEnumAdapters2(&enumAdapters);
    if (!NT_SUCCESS(ret)) {
      return false;
    }
    
    std::vector<D3DKMT_ADAPTERINFO> adapterInfo;
    adapterInfo.resize(enumAdapters.NumAdapters);
    enumAdapters.pAdapters = adapterInfo.data();

    ret = D3DKMTEnumAdapters2(&enumAdapters);
    if (!NT_SUCCESS(ret)) {
      return false;
    }

    static_assert(sizeof(LUID) == sizeof(VkPhysicalDeviceIDProperties::deviceLUID));
    LUID deviceLuid;
    memcpy(&deviceLuid, device->adapter()->devicePropertiesExt().coreDeviceId.deviceLUID, sizeof(deviceLuid));
    
    for (uint32_t i = 0; i < enumAdapters.NumAdapters; i++) {
      const auto& adapter = adapterInfo[i];
      
      if (adapter.AdapterLuid.HighPart == deviceLuid.HighPart &&
          adapter.AdapterLuid.LowPart == deviceLuid.LowPart) {
        D3DKMT_QUERYADAPTERINFO info {};
        info.hAdapter = adapter.hAdapter;
        info.Type = KMTQAITYPE_WDDM_2_7_CAPS;
        D3DKMT_WDDM_2_7_CAPS data {};
        info.pPrivateDriverData = &data;
        info.PrivateDriverDataSize = sizeof(data);
        NTSTATUS err = D3DKMTQueryAdapterInfo(&info);
        if (NT_SUCCESS(err) && data.HwSchEnabled) {
          return true;
        }
      }
    }

    return false;
  }

  void NGXContext::checkDLFGSupport(NVSDK_NGX_Parameter* params) {
    NVSDK_NGX_Result result;

    m_supportsDLFG = false;
    m_dlfgMaxInterpolatedFrames = 0;

    int dlfgAvailable = 0;
    result = params->Get(NVSDK_NGX_Parameter_FrameGeneration_Available, &dlfgAvailable);
    if (NVSDK_NGX_FAILED(result) || !dlfgAvailable) {
      Logger::info(str::format("NVIDIA DLSS Frame Generation not available on this hardware/platform: ", resultToString(result)));
      return;
    }

    int needsUpdatedDriver = 0;
    if (NVSDK_NGX_FAILED(params->Get(NVSDK_NGX_Parameter_FrameGeneration_NeedsUpdatedDriver, &needsUpdatedDriver))) {
      Logger::warn("NVIDIA DLSS Frame generation failed to initialize");
      return;
    }

    // check all the reasons to make sure we present everything to the user at once
    m_supportsDLFG = true;
    
    if (needsUpdatedDriver) {
      std::string message = "NVIDIA DLSS Frame generation cannot be loaded due to outdated driver.";
      unsigned int majorVersion = 0;
      unsigned int minorVersion = 0;
      if (!NVSDK_NGX_FAILED(params->Get(NVSDK_NGX_Parameter_FrameGeneration_MinDriverVersionMajor, &majorVersion)) &&
        !NVSDK_NGX_FAILED(params->Get(NVSDK_NGX_Parameter_FrameGeneration_MinDriverVersionMinor, &minorVersion))) {
        message += "Minimum driver version required: " + std::to_string(majorVersion) + "." + std::to_string(minorVersion);
      }

      m_dlfgNotSupportedReason = m_dlfgNotSupportedReason + message;
      m_supportsDLFG = false;
    }

    bool hardwareSchedulingEnabled = checkHardwareSchedulingEnabled(m_device);
    if (!hardwareSchedulingEnabled) {
      if (!m_dlfgNotSupportedReason.empty()) {
        m_dlfgNotSupportedReason = m_dlfgNotSupportedReason + "\n";
      }

      m_dlfgNotSupportedReason = m_dlfgNotSupportedReason + "NVIDIA DLSS Frame Generation requires GPU hardware scheduling. Please make sure you are running Windows 10 May 2020 update or later, and enable it in Settings -> System -> Display -> Graphics Settings.";
      m_supportsDLFG = false;
    }

    // check for multi-frame support
    if (NVSDK_NGX_FAILED(params->Get(NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, (int*)&m_dlfgMaxInterpolatedFrames))) {
      m_dlfgNotSupportedReason = m_dlfgNotSupportedReason + " NGX parameter query for MultiFrameCountMax failed.";
      m_supportsDLFG = false;
    }

    if (m_dlfgNotSupportedReason.size()) {
      Logger::warn(m_dlfgNotSupportedReason);
    }
  }

  NGXFeatureContext::~NGXFeatureContext() {
    if (m_parameters) {
      NGX_CALL(NVSDK_NGX_VULKAN_DestroyParameters(m_parameters));
      m_parameters = nullptr;
    }
  }

  void NGXDLFGContext::releaseNGXFeature()
  {
    ScopedCpuProfileZone();
    if (m_feature) {
      NGX_CALL(NVSDK_NGX_VULKAN_ReleaseFeature(m_feature));
      m_feature = nullptr;
    }
  }

  NGXFeatureContext::NGXFeatureContext(DxvkDevice* device): m_device(device)
  {
    NVSDK_NGX_Result result = NGX_CALL(NVSDK_NGX_VULKAN_GetCapabilityParameters(&m_parameters));
    if (NVSDK_NGX_FAILED(result)) {
      Logger::err(str::format("NVSDK_NGX_VULKAN_GetCapabilityParameters failed: ", resultToString(result)));
    }
  }

  NGXDLSSContext::NGXDLSSContext(DxvkDevice* device)
    : NGXFeatureContext(device) { }

  NGXDLSSContext::~NGXDLSSContext() {
    releaseNGXFeature();
  }

  void NGXDLSSContext::initialize(Rc<DxvkContext> renderContext,
                                  uint32_t maxRenderSize[2],
                                  uint32_t displayOutSize[2],
                                  bool isContentHDR,
                                  bool depthInverted,
                                  bool autoExposure,
                                  bool sharpening,
                                  NVSDK_NGX_DLSS_Hint_Render_Preset dlssPreset,
                                  NVSDK_NGX_PerfQuality_Value perfQuality) {
    ScopedCpuProfileZone();

    // Render preset hint (model selection) - set for every quality slot so the choice
    // follows whatever perf quality the feature is created with. Written unconditionally:
    // the parameter block outlives feature recreation, so a previously set preset must be
    // explicitly overwritten with Default (0) to actually revert to the snippet's choice.
    m_parameters->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, dlssPreset);
    m_parameters->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality, dlssPreset);
    m_parameters->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality, dlssPreset);
    m_parameters->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, dlssPreset);
    m_parameters->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance, dlssPreset);
    m_parameters->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, dlssPreset);

    const unsigned int CreationNodeMask = 1;
    const unsigned int VisibilityNodeMask = 1;

    const bool lowResolutionMotionVectors = true; // we let the Snippet do the upsampling of the motion vector
    const bool jitteredMV = false; // We don't use the jittered camera matrix to calculate motion vector
    // Next create features
    int createFlags = NVSDK_NGX_DLSS_Feature_Flags_None;
    createFlags |= lowResolutionMotionVectors ? NVSDK_NGX_DLSS_Feature_Flags_MVLowRes : 0;
    createFlags |= isContentHDR ? NVSDK_NGX_DLSS_Feature_Flags_IsHDR : 0;
    createFlags |= depthInverted ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0;
    createFlags |= jitteredMV ? NVSDK_NGX_DLSS_Feature_Flags_MVJittered : 0;
    createFlags |= autoExposure ? NVSDK_NGX_DLSS_Feature_Flags_AutoExposure : 0;
    createFlags |= sharpening ? NVSDK_NGX_DLSS_Feature_Flags_DoSharpening : 0;

    NVSDK_NGX_DLSS_Create_Params createParams = {};

    createParams.Feature.InWidth = maxRenderSize[0];
    createParams.Feature.InHeight = maxRenderSize[1];
    createParams.Feature.InTargetWidth = displayOutSize[0];
    createParams.Feature.InTargetHeight = displayOutSize[1];
    createParams.Feature.InPerfQualityValue = perfQuality;
    createParams.InFeatureCreateFlags = createFlags;

    VkCommandBuffer vkCommandBuffer = renderContext->getCommandList()->getCmdBuffer(dxvk::DxvkCmdBuffer::ExecBuffer);

    m_parameters->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, dlssPreset);
    m_parameters->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality, dlssPreset);
    m_parameters->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, dlssPreset);
    m_parameters->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance, dlssPreset);
    m_parameters->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, dlssPreset);

    // Release video memory when DLSS is disabled.
    m_parameters->Set(NVSDK_NGX_Parameter_FreeMemOnReleaseFeature, 1);

    NVSDK_NGX_Result result = NGX_CALL(NGX_VULKAN_CREATE_DLSS_EXT1(
      m_device->handle(), vkCommandBuffer, CreationNodeMask, VisibilityNodeMask, &m_featureDLSS, m_parameters, &createParams));

    if (NVSDK_NGX_FAILED(result)) {
      Logger::warn(str::format("Failed to create DLSS feature: ", resultToString(result)));
      return;
    }
  }

  void NGXDLSSContext::releaseNGXFeature() {
    if (m_featureDLSS) {
      NGX_CALL(NVSDK_NGX_VULKAN_ReleaseFeature(m_featureDLSS));
      m_featureDLSS = nullptr;
    }
  }

  NGXDLSSContext::OptimalSettings NGXDLSSContext::queryOptimalSettings(const uint32_t displaySize[2], NVSDK_NGX_PerfQuality_Value perfQuality) const
  {
    ScopedCpuProfileZone();
    OptimalSettings settings;
    // Note: Deprecated, should not be used but still must be passed into the query function.
    float dummySharpness;

    NVSDK_NGX_Result result = NGX_CALL(NGX_DLSS_GET_OPTIMAL_SETTINGS(m_parameters,
      displaySize[0], displaySize[1], perfQuality,
      &settings.optimalRenderSize[0], &settings.optimalRenderSize[1],
      &settings.maxRenderSize[0], &settings.maxRenderSize[1],
      &settings.minRenderSize[0], &settings.minRenderSize[1],
      &dummySharpness));

    if (NVSDK_NGX_FAILED(result)) {
      Logger::err(str::format("Querying optimal settings failed: ", resultToString(result)));
      return settings;
    }

    return settings;
  }

  bool NGXDLSSContext::evaluateDLSS(
    Rc<DxvkContext> renderContext,
    const NGXBuffers& buffers,
    const NGXSettings& settings) const
  {
    if (!m_featureDLSS)
      return false;
    
    ScopedCpuProfileZone();
    
    // In DLSS v2, the target is already upsampled (while in v1, the upsampling is handled in a later pass)
    uint32_t inWidth = buffers.pUnresolvedColor->image->info().extent.width;
    uint32_t inHeight = buffers.pUnresolvedColor->image->info().extent.height;
    uint32_t outWidth = buffers.pResolvedColor->image->info().extent.width;
    uint32_t outHeight = buffers.pResolvedColor->image->info().extent.height;
    assert(outWidth >= inWidth && outHeight >= inHeight);

    bool success = true;

    VkCommandBuffer vkCommandbuffer = renderContext->getCommandList()->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);

    NVSDK_NGX_Resource_VK unresolvedColorResource = TextureToResourceVK(buffers.pUnresolvedColor, false);
    NVSDK_NGX_Resource_VK resolvedColorResource = TextureToResourceVK(buffers.pResolvedColor, true);
    NVSDK_NGX_Resource_VK motionVectorsResource = TextureToResourceVK(buffers.pMotionVectors, false);
    NVSDK_NGX_Resource_VK depthResource = TextureToResourceVK(buffers.pDepth, false);
    NVSDK_NGX_Resource_VK exposureResource = TextureToResourceVK(buffers.pExposure, false);
    NVSDK_NGX_Resource_VK biasCurrentColorMaskResource = TextureToResourceVK(buffers.pBiasCurrentColorMask, false);

    NVSDK_NGX_VK_DLSS_Eval_Params evalParams = {};
    evalParams.Feature.pInColor = &unresolvedColorResource;
    evalParams.Feature.pInOutput = &resolvedColorResource;
    evalParams.pInDepth = &depthResource;
    // xxxnsubtil: the DLSS indicator reads the exposure texture even when DLSS autoexposure is on
    evalParams.pInExposureTexture = &exposureResource;
    evalParams.pInMotionVectors = &motionVectorsResource;
    evalParams.pInBiasCurrentColorMask = settings.antiGhost ? &biasCurrentColorMaskResource : nullptr;
    evalParams.InJitterOffsetX = settings.jitterOffset[0];
    evalParams.InJitterOffsetY = settings.jitterOffset[1];
    // Note: Sharpness parameter is deprecated and is not read by newer versions of DLSS, so setting it to 0 is fine here.
    evalParams.Feature.InSharpness = 0.0f;
    evalParams.InPreExposure = settings.preExposure;
    evalParams.InReset = settings.resetAccumulation ? 1 : 0;
    evalParams.InMVScaleX = settings.motionVectorScale[0];
    evalParams.InMVScaleY = settings.motionVectorScale[1];
    evalParams.InRenderSubrectDimensions = { inWidth, inHeight };

    NVSDK_NGX_Result result;
    result = NGX_CALL(NGX_VULKAN_EVALUATE_DLSS_EXT(vkCommandbuffer, m_featureDLSS, m_parameters, &evalParams));

    if (NVSDK_NGX_FAILED(result)) {
      success = false;
    }

    return success;
  }

  NGXDLFGContext::NGXDLFGContext(DxvkDevice* device)
    : NGXFeatureContext(device) { }

  NGXDLFGContext::~NGXDLFGContext() {
    releaseNGXFeature();
  }

  bool NGXDLFGContext::initialize(Rc<DxvkContext> renderContext,
                                  VkCommandBuffer commandList,
                                  uint32_t displayOutSize[2],
                                  VkFormat outputFormat,
                                  bool litePreset) {
    if (m_parameters == nullptr || commandList == VK_NULL_HANDLE) {
      Logger::err("Cannot create DLFG without NGX parameters and a command buffer.");
      return false;
    }

    NVSDK_NGX_DLSSG_Create_Params createParams = { };
    createParams.Width = displayOutSize[0];
    createParams.Height = displayOutSize[1];
    createParams.NativeBackbufferFormat = outputFormat;

    NVSDK_NGX_Result result = ngxCall([&] {
      // The bundled FG headers predate this NGX parameter.
      m_parameters->Set("DLSSG.LiteCandidateBlendingEnabled", static_cast<unsigned int>(litePreset));
      return NGX_VK_CREATE_DLSSG(commandList,
        1, // InCreationNodeMask
        1, // InVisibilityNodeMask
        &m_feature,
        m_parameters,
        &createParams);
    });

    if (NVSDK_NGX_FAILED(result) || m_feature == nullptr) {
      Logger::err(str::format("Failed to create DLFG feature: ", resultToString(result)));
      releaseNGXFeature();
      return false;
    }

    return true;
  }

  void toNGX(float (&ret)[4][4], const Matrix4& mat) {
    ret[0][0] = mat[0].x;
    ret[0][1] = mat[0].y;
    ret[0][2] = mat[0].z;
    ret[0][3] = mat[0].w;

    ret[1][0] = mat[1].x;
    ret[1][1] = mat[1].y;
    ret[1][2] = mat[1].z;
    ret[1][3] = mat[1].w;

    ret[2][0] = mat[2].x;
    ret[2][1] = mat[2].y;
    ret[2][2] = mat[2].z;
    ret[2][3] = mat[2].w;

    ret[3][0] = mat[3].x;
    ret[3][1] = mat[3].y;
    ret[3][2] = mat[3].z;
    ret[3][3] = mat[3].w;
  }

  void setNGXIdentity(float(&ret)[4][4]) {
    ret[0][0] = 1.0;
    ret[0][1] = 0.0;
    ret[0][2] = 0.0;
    ret[0][3] = 0.0;

    ret[1][0] = 0.0;
    ret[1][1] = 1.0;
    ret[1][2] = 0.0;
    ret[1][3] = 0.0;

    ret[2][0] = 0.0;
    ret[2][1] = 0.0;
    ret[2][2] = 1.0;
    ret[2][3] = 0.0;

    ret[3][0] = 0.0;
    ret[3][1] = 0.0;
    ret[3][2] = 0.0;
    ret[3][3] = 1.0;
  }

  void toNGX(float(&ret)[2], const Vector2& in) {
    ret[0] = in.x;
    ret[1] = in.y;
  }

  void toNGX(float(&ret)[3], const Vector3& in) {
    ret[0] = in.x;
    ret[1] = in.y;
    ret[2] = in.z;
  }

  NGXDLFGContext::EvaluateResult NGXDLFGContext::evaluate(Rc<DxvkContext> renderContext,
                                                          VkCommandBuffer clientCommandList,
                                                          Rc<DxvkImageView> interpolatedOutput,
                                                          Rc<DxvkImageView> compositedColorBuffer,
                                                          Rc<DxvkImageView> hudlessColorBuffer,
                                                          Rc<DxvkImageView> motionVectors,
                                                          Rc<DxvkImageView> depth,
                                                          const RtCamera& camera,
                                                          Vector2 motionVectorScale,
                                                          uint32_t interpolatedFrameIndex,
                                                          uint32_t interpolatedFrameCount,
                                                          bool resetHistory) {
    ScopedCpuProfileZone();

    if (m_feature == nullptr || m_parameters == nullptr || clientCommandList == VK_NULL_HANDLE
        || interpolatedOutput == nullptr || compositedColorBuffer == nullptr || motionVectors == nullptr || depth == nullptr) {
      return EvaluateResult::Failure;
    }

    auto ngxColorBuffer = ViewToResourceVK(compositedColorBuffer, true);
    auto ngxMVec = ViewToResourceVK(motionVectors, false);
    auto ngxDepth = ViewToResourceVK(depth, false);
    auto ngxOutput = ViewToResourceVK(interpolatedOutput, true);

    // Optional: must stay alive until the evaluate call below
    NVSDK_NGX_Resource_VK ngxHudless = {};
    if (hudlessColorBuffer != nullptr) {
      ngxHudless = ViewToResourceVK(hudlessColorBuffer, false);
    }

    NVSDK_NGX_VK_DLSSG_Eval_Params evalParams = {};
    evalParams.pBackbuffer = &ngxColorBuffer;
    evalParams.pMVecs = &ngxMVec;
    evalParams.pDepth = &ngxDepth;
    evalParams.pHudless = hudlessColorBuffer != nullptr ? &ngxHudless : nullptr;
    evalParams.pOutputInterpFrame = &ngxOutput;

    const Matrix4& viewToProjection = camera.getViewToProjection();
    const Matrix4& viewToWorld = camera.getViewToWorld();
    const Matrix4& projectionToView = camera.getProjectionToView();
    const Matrix4& prevWorldToView = camera.getPreviousWorldToView();
    const Matrix4& prevViewToProjection = camera.getPreviousViewToProjection();

    const Matrix4 clipToPrevClip = prevViewToProjection * prevWorldToView * viewToWorld * projectionToView;
    const Matrix4 prevClipToClip = inverse(clipToPrevClip);

    NVSDK_NGX_DLSSG_Opt_Eval_Params consts = {};
    toNGX(consts.cameraViewToClip, viewToProjection);
    toNGX(consts.clipToCameraView, projectionToView);
    setNGXIdentity(consts.clipToLensClip);
    toNGX(consts.clipToPrevClip, clipToPrevClip);
    toNGX(consts.prevClipToClip, prevClipToClip);

    camera.getJittering(consts.jitterOffset);
    toNGX(consts.mvecScale, motionVectorScale);
    toNGX(consts.cameraPinholeOffset, Vector2(0.0, 0.0));
    toNGX(consts.cameraPos, camera.getPosition());
    toNGX(consts.cameraUp, camera.getUp());
    toNGX(consts.cameraRight, camera.getRight());
    toNGX(consts.cameraFwd, camera.getDirection());

    float shearX, shearY;
    bool isLHS, isReverseZ;
    decomposeProjection(viewToProjection, consts.cameraAspectRatio, consts.cameraFOV, consts.cameraNear, consts.cameraFar, shearX, shearY, isLHS, isReverseZ);

    //consts.numberOfFramesToGenerate = 1;  // xxxnsubtil: this doesn't do anything, each eval call always generates one frame only
    consts.colorBuffersHDR = false;
    consts.depthInverted = false;
    consts.cameraMotionIncluded = true;
    consts.reset = resetHistory;
    consts.notRenderingGameFrames = false;
    consts.orthoProjection = false;
    consts.motionVectorsInvalidValue = 0.0; // xxxnsubtil: is this correct?
    consts.motionVectorsDilated = false;

    // The x64 SDK helper sources these values from consts, overriding direct parameter writes.
    consts.multiFrameCount = interpolatedFrameCount;
    consts.multiFrameIndex = interpolatedFrameIndex + 1;

    NVSDK_NGX_Result result;
    result = NGX_CALL(NGX_VK_EVALUATE_DLSSG(clientCommandList, m_feature, m_parameters, &evalParams, &consts));
    if (NVSDK_NGX_FAILED(result)) {
      Logger::err(str::format("NGX_VK_EVALUATE_DLSSG failed: ", resultToString(result)));
      return EvaluateResult::Failure;
    }
    
    return EvaluateResult::Success;
  }


  namespace {
    struct DlssNrForwarder {
      HMODULE module = nullptr;
      PFN_dlssnr_forwarder_last_error_message lastError = nullptr;
      PFN_dlssnr_vulkan_init init = nullptr;
      PFN_dlssnr_vulkan_create create = nullptr;
      PFN_dlssnr_vulkan_evaluate evaluate = nullptr;
      PFN_dlssnr_vulkan_release release = nullptr;
      bool snippetInitialized = false;
      bool loadAttempted = false;

      bool load() {
        if (loadAttempted) {
          return module != nullptr && init != nullptr && create != nullptr && evaluate != nullptr && release != nullptr;
        }
        loadAttempted = true;

        const std::string path = env::getDllDirectory() + "\\nvngx.dll_dlssnr.dll";
        module = LoadLibraryW(str::tows(path.c_str()).c_str());
        if (module == nullptr) {
          Logger::warn(str::format("DLSS-NR forwarder was not loaded from ", path, " (", GetLastError(), ")"));
          return false;
        }

        lastError = reinterpret_cast<PFN_dlssnr_forwarder_last_error_message>(GetProcAddress(module, "dlssnr_forwarder_last_error_message"));
        init = reinterpret_cast<PFN_dlssnr_vulkan_init>(GetProcAddress(module, "dlssnr_vulkan_init"));
        create = reinterpret_cast<PFN_dlssnr_vulkan_create>(GetProcAddress(module, "dlssnr_vulkan_create"));
        evaluate = reinterpret_cast<PFN_dlssnr_vulkan_evaluate>(GetProcAddress(module, "dlssnr_vulkan_evaluate"));
        release = reinterpret_cast<PFN_dlssnr_vulkan_release>(GetProcAddress(module, "dlssnr_vulkan_release"));
        if (init == nullptr || create == nullptr || evaluate == nullptr || release == nullptr) {
          Logger::err("DLSS-NR forwarder is missing an export");
          return false;
        }
        return true;
      }

      const char* error() const {
        return lastError != nullptr ? lastError() : "";
      }

      bool ensureSnippet(DxvkDevice* device, NVSDK_NGX_Parameter* parameters) {
        if (!load()) {
          return false;
        }
        if (snippetInitialized) {
          return true;
        }

        const std::string exePath = env::getExePath();
        const std::string exeFolder = exePath.substr(0, exePath.find_last_of("\\/"));
        const std::wstring logFolder = str::tows(exeFolder.c_str());
        const int result = init(
          RtxOptions::applicationId(),
          logFolder.c_str(),
          device->instance()->handle(),
          device->adapter()->handle(),
          device->handle(),
          nullptr,
          nullptr,
          static_cast<unsigned int>(NVSDK_NGX_Version_API),
          parameters);
        if (NVSDK_NGX_FAILED(static_cast<NVSDK_NGX_Result>(result))) {
          Logger::err(str::format(
            "DLSS-NR snippet init failed: ", resultToString(static_cast<NVSDK_NGX_Result>(result)),
            " ", error()));
          return false;
        }

        snippetInitialized = true;
        Logger::info("DLSS-NR snippet initialized directly.");
        return true;
      }
    };

    DlssNrForwarder& dlssNrForwarder() {
      static DlssNrForwarder forwarder;
      return forwarder;
    }

    // These writes match the SDK helper.
    // They are used when the snippet is called directly.
    void setDlssNrEvalParams(NVSDK_NGX_Parameter* parameters, const NVSDK_NGX_VK_DLSSNR_Eval_Params& evalParams) {
      NVSDK_NGX_Parameter_SetVoidPointer(parameters, NVSDK_NGX_Parameter_DLSSNR_Color, evalParams.pInColor);
      NVSDK_NGX_Parameter_SetVoidPointer(parameters, NVSDK_NGX_Parameter_DLSSNR_Output, evalParams.pInOutput);
      NVSDK_NGX_Parameter_SetVoidPointer(parameters, NVSDK_NGX_Parameter_DLSSNR_Backbuffer, evalParams.pInBackbuffer);
      NVSDK_NGX_Parameter_SetVoidPointer(parameters, NVSDK_NGX_Parameter_DLSSNR_MVec, evalParams.pInMVec);
      NVSDK_NGX_Parameter_SetVoidPointer(parameters, NVSDK_NGX_Parameter_DLSSNR_Depth, evalParams.pInDepth);
      NVSDK_NGX_Parameter_SetVoidPointer(parameters, NVSDK_NGX_Parameter_DLSSNR_BidirectionalDistortionField, evalParams.pInBidirectionalDistortionField);
      NVSDK_NGX_Parameter_SetVoidPointer(parameters, NVSDK_NGX_Parameter_DLSSNR_ControlMask, evalParams.pInControlMask);
      NVSDK_NGX_Parameter_SetVoidPointer(parameters, NVSDK_NGX_Parameter_DLSSNR_UI, evalParams.pInUI);
      NVSDK_NGX_Parameter_SetVoidPointer(parameters, NVSDK_NGX_Parameter_DLSSNR_UIAlpha, evalParams.pInUIAlpha);
      NVSDK_NGX_Parameter_SetF(parameters, NVSDK_NGX_Parameter_DLSSNR_LocalToneStrength, evalParams.InLocalToneStrength);
      NVSDK_NGX_Parameter_SetF(parameters, NVSDK_NGX_Parameter_DLSSNR_LocalStructureStrength, evalParams.InLocalStructureStrength);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_Style, evalParams.InStyle);
      NVSDK_NGX_Parameter_SetI(parameters, NVSDK_NGX_Parameter_DLSSNR_UseAutoMask, evalParams.InUseAutoMask);
      NVSDK_NGX_Parameter_SetI(parameters, NVSDK_NGX_Parameter_DLSSNR_MaskBlendMode, static_cast<int>(evalParams.InMaskBlendMode));
      NVSDK_NGX_Parameter_SetF(parameters, NVSDK_NGX_Parameter_DLSSNR_SkinStructureStrength, evalParams.InSkinStructureStrength);
      NVSDK_NGX_Parameter_SetI(parameters, NVSDK_NGX_Parameter_DLSSNR_Reset, evalParams.InReset);
      NVSDK_NGX_Parameter_SetI(parameters, NVSDK_NGX_Parameter_DLSSNR_DepthInverted, evalParams.InDepthInverted);
      NVSDK_NGX_Parameter_SetI(parameters, NVSDK_NGX_Parameter_DLSSNR_Enabled, evalParams.InEnabled);
      NVSDK_NGX_Parameter_SetI(parameters, NVSDK_NGX_Parameter_DLSSNR_UICorrection, evalParams.InUICorrection);
      NVSDK_NGX_Parameter_SetF(parameters, NVSDK_NGX_Parameter_DLSSNR_Intensity, evalParams.InIntensity);
      NVSDK_NGX_Parameter_SetF(parameters, NVSDK_NGX_Parameter_DLSSNR_MVecScaleX, evalParams.InMVecScaleX);
      NVSDK_NGX_Parameter_SetF(parameters, NVSDK_NGX_Parameter_DLSSNR_MVecScaleY, evalParams.InMVecScaleY);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_ColorSubrectBaseX, evalParams.InColorSubrectBase.X);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_ColorSubrectBaseY, evalParams.InColorSubrectBase.Y);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_ColorSubrectWidth, evalParams.InColorSubrectSize.Width);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_ColorSubrectHeight, evalParams.InColorSubrectSize.Height);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_OutputSubrectBaseX, evalParams.InOutputSubrectBase.X);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_OutputSubrectBaseY, evalParams.InOutputSubrectBase.Y);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_OutputSubrectWidth, evalParams.InOutputSubrectSize.Width);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_OutputSubrectHeight, evalParams.InOutputSubrectSize.Height);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_BackbufferSubrectBaseX, evalParams.InBackbufferSubrectBase.X);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_BackbufferSubrectBaseY, evalParams.InBackbufferSubrectBase.Y);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_BackbufferSubrectWidth, evalParams.InBackbufferSubrectSize.Width);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_BackbufferSubrectHeight, evalParams.InBackbufferSubrectSize.Height);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_MVecSubrectBaseX, evalParams.InMVecSubrectBase.X);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_MVecSubrectBaseY, evalParams.InMVecSubrectBase.Y);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_MVecSubrectWidth, evalParams.InMVecSubrectSize.Width);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_MVecSubrectHeight, evalParams.InMVecSubrectSize.Height);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_DepthSubrectBaseX, evalParams.InDepthSubrectBase.X);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_DepthSubrectBaseY, evalParams.InDepthSubrectBase.Y);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_DepthSubrectWidth, evalParams.InDepthSubrectSize.Width);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_DepthSubrectHeight, evalParams.InDepthSubrectSize.Height);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_BidirectionalDistortionFieldSubrectBaseX, evalParams.InBidirectionalDistortionFieldSubrectBase.X);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_BidirectionalDistortionFieldSubrectBaseY, evalParams.InBidirectionalDistortionFieldSubrectBase.Y);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_BidirectionalDistortionFieldSubrectWidth, evalParams.InBidirectionalDistortionFieldSubrectSize.Width);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_BidirectionalDistortionFieldSubrectHeight, evalParams.InBidirectionalDistortionFieldSubrectSize.Height);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_ControlMaskSubrectBaseX, evalParams.InControlMaskSubrectBase.X);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_ControlMaskSubrectBaseY, evalParams.InControlMaskSubrectBase.Y);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_ControlMaskSubrectWidth, evalParams.InControlMaskSubrectSize.Width);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_ControlMaskSubrectHeight, evalParams.InControlMaskSubrectSize.Height);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_UISubrectBaseX, evalParams.InUISubrectBase.X);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_UISubrectBaseY, evalParams.InUISubrectBase.Y);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_UISubrectWidth, evalParams.InUISubrectSize.Width);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_UISubrectHeight, evalParams.InUISubrectSize.Height);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_UIAlphaSubrectBaseX, evalParams.InUIAlphaSubrectBase.X);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_UIAlphaSubrectBaseY, evalParams.InUIAlphaSubrectBase.Y);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_UIAlphaSubrectWidth, evalParams.InUIAlphaSubrectSize.Width);
      NVSDK_NGX_Parameter_SetUI(parameters, NVSDK_NGX_Parameter_DLSSNR_UIAlphaSubrectHeight, evalParams.InUIAlphaSubrectSize.Height);
    }
  }

  // Neural Rendering
  NGXNeuralRenderingContext::NGXNeuralRenderingContext(DxvkDevice* device)
    : NGXFeatureContext(device) { }

  NGXNeuralRenderingContext::~NGXNeuralRenderingContext() {
    releaseNGXFeature();
  }

  void NGXNeuralRenderingContext::initialize(Rc<DxvkContext> renderContext, const uint32_t displaySize[2], bool queryVramAllocation) {
    if (m_neuralRenderingFeature) {
      renderContext->getDevice()->waitForIdle();
    }
    releaseNGXFeature();

    NVSDK_NGX_DLSSNR_Create_Params createParams = {};
    createParams.Width = displaySize[0];
    createParams.Height = displaySize[1];

    // Preset B selects the lightweight network independently of the appearance model.
    NVSDK_NGX_Parameter_SetI(
      m_parameters,
      NVSDK_NGX_Parameter_DLSSNR_Hint_Render_Preset,
      NVSDK_NGX_DLSSNR_Hint_Render_Preset_B);

    const VkCommandBuffer vkCommandBuffer = renderContext->getCommandList()->getCmdBuffer(dxvk::DxvkCmdBuffer::ExecBuffer);
    NVSDK_NGX_Result result = NVSDK_NGX_Result_Fail;

    // An unsigned snippet is created through nvngx.dll_dlssnr.dll.
    // A signed snippet is created through the NGX core.
    if (m_device->getCommon()->metaNGXContext().useDirectDlssNeuralRendering()) {
      DlssNrForwarder& forwarder = dlssNrForwarder();
      if (!NGX_CALL(forwarder.ensureSnippet(m_device, m_parameters))) {
        releaseNGXFeature();
        return;
      }

      NVSDK_NGX_Parameter_SetUI(m_parameters, NVSDK_NGX_Parameter_DLSSNR_Width, createParams.Width);
      NVSDK_NGX_Parameter_SetUI(m_parameters, NVSDK_NGX_Parameter_DLSSNR_Height, createParams.Height);
      void* created = nullptr;
      std::string failure;
      result = static_cast<NVSDK_NGX_Result>(ngxCall([&] {
        const int createdResult = forwarder.create(
          m_device->handle(),
          vkCommandBuffer,
          static_cast<int>(NVSDK_NGX_Feature_DLSSNR),
          m_parameters,
          &created);
        if (NVSDK_NGX_FAILED(static_cast<NVSDK_NGX_Result>(createdResult))) {
          failure = forwarder.error();
          if (created != nullptr) {
            forwarder.release(created);
            created = nullptr;
          }
        }
        return createdResult;
      }));
      if (NVSDK_NGX_FAILED(result)) {
        Logger::err(str::format(
          "Failed to create DLSS-NR feature directly: ", resultToString(result),
          " ", failure));
        releaseNGXFeature();
        return;
      }
      m_neuralRenderingFeature = static_cast<NVSDK_NGX_Handle*>(created);
      m_directSnippet = true;
    } else {
      result = NGX_CALL(NGX_VULKAN_CREATE_DLSSNR_EXT1(
        m_device->handle(), vkCommandBuffer, 1, 1, &m_neuralRenderingFeature, m_parameters, &createParams));
      if (NVSDK_NGX_FAILED(result)) {
        Logger::err(str::format("Failed to create DLSS-NR feature: ", resultToString(result)));
        releaseNGXFeature();
        return;
      }
    }

    if (queryVramAllocation) {
      unsigned long long vramAllocationBytes = 0;
      const NVSDK_NGX_Result statsResult = NGX_CALL(queryDlssNeuralRenderingVramAllocation(m_parameters, &vramAllocationBytes));
      if (NVSDK_NGX_FAILED(statsResult)) {
        Logger::warn(str::format("Failed to query DLSS-NR VRAM allocation: ", resultToString(statsResult)));
      } else {
        m_vramAllocationBytes = vramAllocationBytes;
        m_hasVramAllocation = true;
      }
    }

    m_initialized = true;
  }

  bool NGXNeuralRenderingContext::evaluateNeuralRendering(
    Rc<DxvkContext> renderContext, const NGXNeuralRenderingBuffers& buffers, const NGXNeuralRenderingSettings& settings) const {
    if (!isNeuralRenderingInitialized()) {
      return false;
    }
    ScopedCpuProfileZone();

    const uint32_t width = buffers.pInColor->image->info().extent.width;
    const uint32_t height = buffers.pInColor->image->info().extent.height;
    const uint32_t outputWidth = buffers.pOutColor->image->info().extent.width;
    const uint32_t outputHeight = buffers.pOutColor->image->info().extent.height;
    const uint32_t mvWidth = buffers.pMotionVectors->image->info().extent.width;
    const uint32_t mvHeight = buffers.pMotionVectors->image->info().extent.height;
    const uint32_t depthWidth = buffers.pDepth->image->info().extent.width;
    const uint32_t depthHeight = buffers.pDepth->image->info().extent.height;
    const uint32_t controlMaskWidth = buffers.pControlMask->image->info().extent.width;
    const uint32_t controlMaskHeight = buffers.pControlMask->image->info().extent.height;

    const VkCommandBuffer vkCommandbuffer = renderContext->getCommandList()->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);

    NVSDK_NGX_Resource_VK inColorResource = TextureToResourceVK(buffers.pInColor, false);
    NVSDK_NGX_Resource_VK outColorResource = TextureToResourceVK(buffers.pOutColor, true);
    NVSDK_NGX_Resource_VK motionVectorsResource = TextureToResourceVK(buffers.pMotionVectors, false);
    NVSDK_NGX_Resource_VK depthResource = TextureToResourceVK(buffers.pDepth, false);
    NVSDK_NGX_Resource_VK controlMaskResource = TextureToResourceVK(buffers.pControlMask, false);

    NVSDK_NGX_VK_DLSSNR_Eval_Params neuralRenderingEvalParams = {};
    neuralRenderingEvalParams.pInColor = &inColorResource;
    neuralRenderingEvalParams.pInOutput = &outColorResource;
    neuralRenderingEvalParams.pInMVec = &motionVectorsResource;
    neuralRenderingEvalParams.pInDepth = &depthResource;
    neuralRenderingEvalParams.pInControlMask = &controlMaskResource;
    neuralRenderingEvalParams.InReset = settings.resetAccumulation ? 1 : 0;
    neuralRenderingEvalParams.InDepthInverted = 0;
    neuralRenderingEvalParams.InEnabled = 1;
    neuralRenderingEvalParams.InIntensity = settings.intensity;
    neuralRenderingEvalParams.InLocalToneStrength = settings.toneStrength;
    neuralRenderingEvalParams.InLocalStructureStrength = settings.structuralStrength;
    neuralRenderingEvalParams.InMaskBlendMode = settings.useAutoMask
      ? NVSDK_NGX_DLSSNR_Mask_Blend_Mode_Min
      : NVSDK_NGX_DLSSNR_Mask_Blend_Mode_None;
    neuralRenderingEvalParams.InStyle = settings.model;
    neuralRenderingEvalParams.InUseAutoMask = settings.useAutoMask ? 1 : 0;
    neuralRenderingEvalParams.InSkinStructureStrength = settings.skinStructureStrength;
    neuralRenderingEvalParams.InMVecScaleX = settings.motionVectorScale[0];
    neuralRenderingEvalParams.InMVecScaleY = settings.motionVectorScale[1];
    neuralRenderingEvalParams.InColorSubrectBase = { 0, 0 };
    neuralRenderingEvalParams.InColorSubrectSize = { width, height };
    neuralRenderingEvalParams.InOutputSubrectBase = { 0, 0 };
    neuralRenderingEvalParams.InOutputSubrectSize = { outputWidth, outputHeight };
    neuralRenderingEvalParams.InMVecSubrectBase = { 0, 0 };
    neuralRenderingEvalParams.InMVecSubrectSize = { mvWidth, mvHeight };
    neuralRenderingEvalParams.InDepthSubrectBase = { 0, 0 };
    neuralRenderingEvalParams.InDepthSubrectSize = { depthWidth, depthHeight };
    neuralRenderingEvalParams.InControlMaskSubrectBase = { 0, 0 };
    neuralRenderingEvalParams.InControlMaskSubrectSize = { controlMaskWidth, controlMaskHeight };

    NVSDK_NGX_Parameter_SetF(m_parameters, NVSDK_NGX_Parameter_Jitter_Offset_X, settings.jitterOffset[0]);
    NVSDK_NGX_Parameter_SetF(m_parameters, NVSDK_NGX_Parameter_Jitter_Offset_Y, settings.jitterOffset[1]);

    NVSDK_NGX_Result result = NVSDK_NGX_Result_Fail;
    if (m_directSnippet) {
      setDlssNrEvalParams(m_parameters, neuralRenderingEvalParams);
      DlssNrForwarder& forwarder = dlssNrForwarder();
      std::string failure;
      result = static_cast<NVSDK_NGX_Result>(ngxCall([&] {
        const int evaluated = forwarder.evaluate(vkCommandbuffer, m_neuralRenderingFeature, m_parameters);
        if (NVSDK_NGX_FAILED(static_cast<NVSDK_NGX_Result>(evaluated))) {
          failure = forwarder.error();
        }
        return evaluated;
      }));
      if (NVSDK_NGX_FAILED(result)) {
        Logger::err(str::format(
          "DLSS-NR evaluation failed: ", resultToString(result), " ", failure));
        return false;
      }
    } else {
      result = NGX_CALL(
        NGX_VULKAN_EVALUATE_DLSSNR_EXT(vkCommandbuffer, m_neuralRenderingFeature, m_parameters, &neuralRenderingEvalParams));
      if (NVSDK_NGX_FAILED(result)) {
        Logger::err(str::format("DLSS-NR evaluation failed: ", resultToString(result)));
        return false;
      }
    }

    return true;
  }

  void NGXNeuralRenderingContext::releaseNGXFeature() {
    m_initialized = false;
    m_hasVramAllocation = false;
    m_vramAllocationBytes = 0;

    if (m_neuralRenderingFeature) {
      if (m_directSnippet) {
        NGX_CALL(dlssNrForwarder().release(m_neuralRenderingFeature));
      } else {
        NGX_CALL(NVSDK_NGX_VULKAN_ReleaseFeature(m_neuralRenderingFeature));
      }
      m_neuralRenderingFeature = nullptr;
    }
    m_directSnippet = false;
  }

} // namespace dxvk

int remixinternal_GetDlssNeuralRenderingStatus() {
  return s_dlssNeuralRenderingStatus.load(std::memory_order_acquire);
}

#undef NGX_CALL
