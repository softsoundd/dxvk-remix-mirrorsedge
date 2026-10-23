#pragma once

// The snippet rejects a caller whose module path does not contain "nvngx.dll".
// This filename is what satisfies that check.
// The DLL loads nvngx_dlssnr.dll from the same directory and forwards its Vulkan entry points.

#ifdef __cplusplus
extern "C" {
#endif

typedef const char*(__cdecl* PFN_dlssnr_forwarder_last_error_message)(void);
typedef int(__cdecl* PFN_dlssnr_vulkan_init)(
  unsigned long long applicationId,
  const wchar_t* applicationDataPath,
  void* instance,
  void* physicalDevice,
  void* device,
  void* getInstanceProcAddr,
  void* getDeviceProcAddr,
  unsigned int sdkVersion,
  const void* parameters);
typedef int(__cdecl* PFN_dlssnr_vulkan_create)(
  void* device,
  void* commandBuffer,
  int featureId,
  const void* parameters,
  void** outHandle);
typedef int(__cdecl* PFN_dlssnr_vulkan_evaluate)(
  void* commandBuffer,
  const void* featureHandle,
  const void* parameters);
typedef int(__cdecl* PFN_dlssnr_vulkan_release)(void* featureHandle);

#ifdef __cplusplus
}
#endif
