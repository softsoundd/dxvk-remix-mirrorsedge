// src/dxvk/rtx_render/rtx_fork_api_entry.cpp
//
// Fork-owned file. Contains the implementations of fork_hooks:: functions
// lifted from rtx_remix_api.cpp during the 2026-04-18 fork touchpoint-pattern
// refactor.
//
// See docs/fork-touchpoints.md for the full fork-hooks catalogue.
//
// This file is populated in four passes:
//   #7a  — infrastructure / foundation blocks:
//           * textureHashPathLookup
//           * mutateTextureHashOption
//   #7b  — API function implementations:
//           * drawScreenOverlay   — stages pixel data into s_pendingScreenOverlay
//           * presentScreenOverlayFlush — drains s_pendingScreenOverlay each frame
//           * getUiState / setUiState
//           * getSharedD3D11TextureHandle (stub)
//           * dxvkGetTextureHash
//           * createTexture / destroyTexture
//   #7c  — interception + vtable init + frame-boundary callback infrastructure:
//           * notifyBeginScene    — fires s_beginCallback on first frame submission
//           * registerCallbacks   — stores begin/end/present callbacks
//           * shutdownCallbacks   — clears all four callback-state vars
//           * presentCallbackDispatch — fires end/present callbacks + resets s_inFrame
//           * remixApiVtableInit  — populates all fork-added vtable slots
//   #7d  — VRAM telemetry + compaction hooks (this pass):
//           * requestVramCompaction  — sets SceneManager atomic flag
//           * requestTextureVramFree — sets SceneManager atomic flag
//           * getVramStats           — fills remixapi_VramStats (DXVK + driver-view)
//
// PendingScreenOverlay state (struct + s_pendingScreenOverlay) lives in the
// anonymous namespace of THIS translation unit so both the writer
// (drawScreenOverlay) and the reader (presentScreenOverlayFlush) can reference
// it directly without coupling through the fork-hooks header. The upstream copy
// of this struct and optional was removed in migration #7b.

#include "rtx_fork_hooks.h"

#include "rtx_context.h"                  // DxvkContext::getCommonObjects(), RtxContext
#include "rtx_option.h"                   // RtxOption, RtxOptionImpl, fast_unordered_set
#include "rtx_option_layer.h"             // RtxOptionLayer::getUserLayer()
#include "rtx_options.h"                  // RtxOptions::showUI(), UIType
#include "rtx_texture.h"                  // TextureRef
#include "rtx_texture_manager.h"          // RtxTextureManager::getTextureTable()

#include "../dxvk_adapter.h"              // DxvkAdapter::memoryProperties, getMemoryHeapInfo, DxvkAdapterMemoryInfo
#include "../dxvk_buffer.h"               // DxvkBuffer, DxvkBufferCreateInfo, DxvkBufferSlice
#include "../dxvk_image.h"                // DxvkImage, DxvkImageCreateInfo, DxvkImageViewCreateInfo
#include "../dxvk_memory.h"               // DxvkMemoryStats::Category
#include "../dxvk_objects.h"              // DxvkCommonObjects, getImgui()
#include "../dxvk_util.h"                 // util::computeMipLevelExtent, computeImageDataSize
#include "../imgui/dxvk_imgui.h"          // ImGUI::switchMenu, ImGUI::AddTexture
#include "rtx_scene_manager.h"            // SceneManager::requestVramCompaction, requestTextureVramFree

#include "../../d3d9/d3d9_device.h"       // D3D9DeviceEx, LockDevice, EmitCs, GetDXVKDevice
#include "../../d3d9/d3d9_texture.h"      // D3D9CommonTexture, GetCommonTexture

#include "../../util/xxHash/xxhash.h"     // XXH64_hash_t

#include <remix/remix_c.h>                // remixapi_ErrorCode, REMIXAPI_ERROR_CODE_*

// Forward declarations for the three extern-"C" fork-exported functions that
// remixApiVtableInit registers into the vtable. These have external linkage
// (REMIXAPI = __declspec(dllexport)) so they are visible across translation
// units; all anonymous-namespace functions in rtx_remix_api.cpp are NOT.
//
// NOTE: remixapi_CreateLightBatched is in the anonymous namespace (no REMIXAPI
// prefix) so it has internal linkage and cannot be named here. It is assigned
// inline in the vtable init block in rtx_remix_api.cpp.
extern "C" {
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_RegisterCallbacks(
    PFN_remixapi_BridgeCallback beginSceneCallback,
    PFN_remixapi_BridgeCallback endSceneCallback,
    PFN_remixapi_BridgeCallback presentCallback);
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_AutoInstancePersistentLights(void);
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_UpdateLightDefinition(
    remixapi_LightHandle handle,
    const remixapi_LightInfo* info);
}

#include <algorithm>                      // std::clamp
#include <atomic>
#include <cstdint>
#include <cstring>                        // memcpy
#include <optional>
#include <string>

namespace dxvk {
namespace {

  // -------------------------------------------------------------------------
  // PendingScreenOverlay — fork-canonical definition (migration #7b)
  //
  // Shared state between the API thread and the render thread for the
  // DrawScreenOverlay / Present path.  Written by drawScreenOverlay (staging a
  // pixel buffer from the API thread), drained by presentScreenOverlayFlush
  // (handing the staging buffer to the render thread via EmitCs /
  // setScreenOverlayData).
  //
  // TU-local (anonymous namespace).  The upstream copy of this struct and
  // optional in rtx_remix_api.cpp was deleted in migration #7b; this is now
  // the sole definition.
  // -------------------------------------------------------------------------
  struct PendingScreenOverlay {
    Rc<DxvkBuffer> stagingBuffer;
    uint32_t width;
    uint32_t height;
    VkFormat format;
    float opacity;
  };

  std::optional<PendingScreenOverlay> s_pendingScreenOverlay;

  // -------------------------------------------------------------------------
  // Frame-boundary callback state (migration #7c)
  //
  // Lifted from rtx_remix_api.cpp anonymous namespace. Written by
  // registerCallbacks (remixapi_RegisterCallbacks delegate) and shutdownCallbacks
  // (remixapi_Shutdown). Read by notifyBeginScene (DrawInstance /
  // DrawLightInstance call sites), presentCallbackDispatch (remixapi_Present),
  // and shutdownCallbacks (remixapi_Shutdown).
  //
  // s_inFrame is atomic because remixapi_DrawInstance and remixapi_Present can
  // race on the exchange/load under multi-threaded plugin usage.
  // -------------------------------------------------------------------------
  std::atomic<bool>              s_inFrame       { false };
  PFN_remixapi_BridgeCallback    s_beginCallback  { nullptr };
  PFN_remixapi_BridgeCallback    s_endCallback    { nullptr };
  PFN_remixapi_BridgeCallback    s_presentCallback { nullptr };

} // anonymous namespace

namespace fork_hooks {

  // ---------------------------------------------------------------------------
  // textureHashPathLookup
  //
  // Called from the preloadTexture lambda inside convert::toRtMaterialFinalized
  // (rtx_remix_api.cpp). Supports referencing API-uploaded textures by their
  // image hash via a "0x<hex>" pseudo-path, so material JSON authored by a
  // plugin can name a CreateTexture'd texture without inventing a real file
  // path on disk.
  //
  // Returns true and writes outRef when the path parses as a hex string and
  // a texture with that image hash is registered in the texture manager.
  // Returns false otherwise (including the common case of a real file path);
  // the caller falls back to the regular AssetDataManager lookup.
  //
  // ACCESS NOTE: uses only the public TextureManager::getTextureTable() and
  // TextureRef::getImageHash(). No friend declaration required.
  // ---------------------------------------------------------------------------
  bool textureHashPathLookup(
      DxvkContext& ctx,
      const std::filesystem::path& path,
      TextureRef& outRef) {
    const std::string pathStr = path.string();
    if (pathStr.size() <= 2 || pathStr[0] != '0' || (pathStr[1] != 'x' && pathStr[1] != 'X')) {
      return false;
    }
    try {
      const uint64_t hash = std::stoull(pathStr, nullptr, 16);
      if (hash == 0) {
        return false;
      }
      const auto& textureTable = ctx.getCommonObjects()->getTextureManager().getTextureTable();
      for (const auto& ref : textureTable) {
        if (ref.isValid() && ref.getImageHash() == hash) {
          outRef = ref;
          return true;
        }
      }
    } catch (...) {
      // stoull threw — fall through and return false so the caller can try
      // the normal asset-path resolver instead.
    }
    return false;
  }

  // ---------------------------------------------------------------------------
  // mutateTextureHashOption
  //
  // Implements the per-category texture-hash add/remove operation used by
  // remixapi_AddTextureHash and remixapi_RemoveTextureHash. Looks up a
  // named RtxOption<fast_unordered_set>, parses the incoming hash string as
  // hex, and mutates the set via the user config layer.
  //
  // Remove uses removeHash (not clearHash): records a negative opinion on the
  // user layer so lower-priority layers (config files, rtx.conf defaults)
  // cannot re-contribute the hash. This matches the fork's hard-delete
  // semantic — clearHash would only drop the user-layer opinion and let the
  // underlying default win.
  //
  // LOCK ORDER: the caller must hold the remix-api static mutex (s_mutex in
  // rtx_remix_api.cpp). This function then acquires the RtxOptionImpl update
  // mutex internally via addHash/removeHash. Inverting that order (taking
  // the RtxOption mutex first and then s_mutex) would deadlock against the
  // EmitCs path, which runs lambdas holding the RtxOption mutex and must
  // never re-enter s_mutex.
  //
  // ACCESS NOTE: uses only public RtxOption APIs. No friend declaration
  // required.
  // ---------------------------------------------------------------------------
  remixapi_ErrorCode mutateTextureHashOption(
      const char* textureCategory,
      const char* textureHash,
      bool add) {
    if (!textureCategory || textureCategory[0] == '\0' || !textureHash) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    RtxOptionImpl* option = RtxOptionImpl::getOptionByFullName(std::string { textureCategory });
    if (!option) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }
    if (option->getType() != OptionType::HashSet) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    XXH64_hash_t h = 0;
    try {
      h = std::stoull(textureHash, nullptr, 16);
    } catch (...) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }

    const RtxOptionLayer* layer = RtxOptionLayer::getUserLayer();
    if (!layer) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }

    // Safe because getType() == HashSet implies T == fast_unordered_set.
    auto* hashSetOption = static_cast<RtxOption<fast_unordered_set>*>(option);
    if (add) {
      hashSetOption->addHash(h, layer);
    } else {
      hashSetOption->removeHash(h, layer);
    }
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // ---------------------------------------------------------------------------
  // drawScreenOverlay (migration #7b)
  //
  // Copies pPixelData into a host-visible staging buffer and stores it in
  // s_pendingScreenOverlay. A null pPixelData or zero dims clears the pending
  // overlay.  presentScreenOverlayFlush drains this state to the render thread
  // at the next present boundary.
  //
  // Reverted [RTX-Diag] FIRST-CALL log block is intentionally absent.
  // ---------------------------------------------------------------------------
  remixapi_ErrorCode drawScreenOverlay(
      D3D9DeviceEx* remixDevice,
      const void*   pPixelData,
      uint32_t      width,
      uint32_t      height,
      remixapi_Format format,
      float         opacity) {
    // Null / zero-dim pixel data clears the overlay for this frame.
    if (!pPixelData || width == 0 || height == 0) {
      s_pendingScreenOverlay.reset();
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }

    VkFormat vkFormat = VK_FORMAT_UNDEFINED;
    switch (format) {
      case REMIXAPI_FORMAT_R8G8B8A8_UNORM: vkFormat = VK_FORMAT_R8G8B8A8_UNORM; break;
      case REMIXAPI_FORMAT_B8G8R8A8_UNORM: vkFormat = VK_FORMAT_B8G8R8A8_UNORM; break;
      default:
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    // 4 bytes per pixel for RGBA8/BGRA8.
    const uint64_t dataSize = static_cast<uint64_t>(width) * height * 4;

    DxvkBufferCreateInfo stagingInfo = {};
    stagingInfo.size   = dataSize;
    stagingInfo.usage  = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
    stagingInfo.access = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_WRITE_BIT;

    Rc<DxvkBuffer> stagingBuffer = remixDevice->GetDXVKDevice()->createBuffer(
      stagingInfo,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      DxvkMemoryStats::Category::RTXBuffer,
      "Remix API screen overlay staging");

    if (stagingBuffer == nullptr) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }

    // Copy pixel data to staging buffer.
    DxvkBufferSlice stagingSlice { stagingBuffer };
    memcpy(stagingSlice.mapPtr(0), pPixelData, dataSize);

    s_pendingScreenOverlay = PendingScreenOverlay {
      std::move(stagingBuffer),
      width, height,
      vkFormat,
      std::clamp(opacity, 0.0f, 1.0f)
    };

    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // ---------------------------------------------------------------------------
  // presentScreenOverlayFlush (migration #7b)
  //
  // Drains s_pendingScreenOverlay (if set) to the render thread via EmitCs.
  // Called from both remixapi_Present paths after the light/mesh flush EmitCs
  // and before the endScene callback dispatch.
  // ---------------------------------------------------------------------------
  void presentScreenOverlayFlush(D3D9DeviceEx* remixDevice) {
    if (!s_pendingScreenOverlay.has_value()) {
      return;
    }
    PendingScreenOverlay overlay = std::move(*s_pendingScreenOverlay);
    s_pendingScreenOverlay.reset();

    remixDevice->EmitCs([cOverlay = std::move(overlay)](DxvkContext* ctx) mutable {
      static_cast<RtxContext*>(ctx)->setScreenOverlayData(
        std::move(cOverlay.stagingBuffer),
        cOverlay.width, cOverlay.height,
        cOverlay.format, cOverlay.opacity);
    });
  }

  // ---------------------------------------------------------------------------
  // getUiState (migration #7b)
  //
  // Reads RtxOptions::showUI() and maps the internal UIType enum to
  // remixapi_UIState.  Returns REMIXAPI_UI_STATE_NONE if the device is not
  // registered.
  // ---------------------------------------------------------------------------
  remixapi_UIState getUiState(D3D9DeviceEx* remixDevice) {
    if (!remixDevice) {
      return REMIXAPI_UI_STATE_NONE;
    }
    switch (RtxOptions::showUI()) {
      case UIType::None:     return REMIXAPI_UI_STATE_NONE;
      case UIType::Basic:    return REMIXAPI_UI_STATE_BASIC;
      case UIType::Advanced: return REMIXAPI_UI_STATE_ADVANCED;
      default:               return REMIXAPI_UI_STATE_NONE;
    }
  }

  // ---------------------------------------------------------------------------
  // setUiState (migration #7b)
  //
  // Maps remixapi_UIState to UIType, acquires the device lock, and calls
  // getImgui().switchMenu(uiType).
  // ---------------------------------------------------------------------------
  remixapi_ErrorCode setUiState(D3D9DeviceEx* remixDevice, remixapi_UIState state) {
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }

    UIType uiType;
    switch (state) {
      case REMIXAPI_UI_STATE_NONE:     uiType = UIType::None;     break;
      case REMIXAPI_UI_STATE_BASIC:    uiType = UIType::Basic;    break;
      case REMIXAPI_UI_STATE_ADVANCED: uiType = UIType::Advanced; break;
      default:
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    auto dxvkDevice = remixDevice->GetDXVKDevice();
    if (!dxvkDevice.ptr() || !dxvkDevice->getCommon()) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }
    auto devLock = remixDevice->LockDevice();
    dxvkDevice->getCommon()->getImgui().switchMenu(uiType);
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // ---------------------------------------------------------------------------
  // getSharedD3D11TextureHandle (migration #7b)
  //
  // Stub. The DX11 shared-memory export backbuffer path (gmod commit 098a7471)
  // is not ported to this fork. This code path is only exercised when the host
  // app opts into external-swapchain mode via dxvk_CreateD3D9(forceNoVkSwapchain
  // == TRUE). Separate-window mode (used by Skyrim) never calls this. Returns
  // GENERAL_FAILURE so callers fall back; the vtable slot is populated so the
  // struct layout matches the header the plugin was built against.
  // ---------------------------------------------------------------------------
  remixapi_ErrorCode getSharedD3D11TextureHandle(
      D3D9DeviceEx* remixDevice,
      void**        out_sharedHandle,
      uint32_t*     out_width,
      uint32_t*     out_height) {
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!out_sharedHandle || !out_width || !out_height) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
  }

  // ---------------------------------------------------------------------------
  // dxvkGetTextureHash (migration #7b)
  //
  // Retrieves the D3D9CommonTexture from the D3D9 texture pointer, gets the
  // underlying DxvkImage, and returns image->getHash() in out_hash.
  // ---------------------------------------------------------------------------
  remixapi_ErrorCode dxvkGetTextureHash(
      IDirect3DTexture9* texture,
      uint64_t*          out_hash) {
    if (!texture || !out_hash) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    D3D9CommonTexture* commonTexture = GetCommonTexture(texture);
    if (!commonTexture) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    // Get the underlying DXVK image.
    const Rc<DxvkImage>& image = commonTexture->GetImage();
    if (image == nullptr) {
      // Texture might be in system memory (not GPU).
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }

    *out_hash = image->getHash();
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // ---------------------------------------------------------------------------
  // createTexture (migration #7b)
  //
  // Creates a DxvkImage + view + staging buffer for the supplied pixel data,
  // copies data into staging, and schedules an EmitCs lambda that transitions
  // the image, uploads all mip levels using block-aware size helpers, transitions
  // to shader-read, and registers the texture with the texture manager and
  // ImGui catalog.
  // ---------------------------------------------------------------------------
  remixapi_ErrorCode createTexture(
      D3D9DeviceEx*            remixDevice,
      const remixapi_TextureInfo* info,
      remixapi_TextureHandle*  out_handle) {
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }

    if (!out_handle || !info || info->sType != REMIXAPI_STRUCT_TYPE_TEXTURE_INFO) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    if (!info->data || info->dataSize == 0 || info->width == 0 || info->height == 0) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    // 3D textures not supported by this API yet; caller must use depth=1.
    // The image below is hardcoded to VK_IMAGE_TYPE_2D, so depth > 1 would
    // create an invalid VkImage. The header documents "Set to 1 for 2D textures";
    // treat any other value as unsupported until real 3D-texture support lands.
    if (info->depth > 1) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    auto handle = reinterpret_cast<remixapi_TextureHandle>(info->hash);
    if (!handle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    // Convert remixapi_Format to VkFormat.
    VkFormat vkFormat = VK_FORMAT_UNDEFINED;
    switch (info->format) {
      case REMIXAPI_FORMAT_R8G8B8A8_UNORM: vkFormat = VK_FORMAT_R8G8B8A8_UNORM;       break;
      case REMIXAPI_FORMAT_R8G8B8A8_SRGB:  vkFormat = VK_FORMAT_R8G8B8A8_SRGB;        break;
      case REMIXAPI_FORMAT_B8G8R8A8_UNORM: vkFormat = VK_FORMAT_B8G8R8A8_UNORM;       break;
      case REMIXAPI_FORMAT_B8G8R8A8_SRGB:  vkFormat = VK_FORMAT_B8G8R8A8_SRGB;        break;
      case REMIXAPI_FORMAT_BC1_RGB_UNORM:  vkFormat = VK_FORMAT_BC1_RGB_UNORM_BLOCK;   break;
      case REMIXAPI_FORMAT_BC1_RGB_SRGB:   vkFormat = VK_FORMAT_BC1_RGB_SRGB_BLOCK;    break;
      case REMIXAPI_FORMAT_BC3_UNORM:      vkFormat = VK_FORMAT_BC3_UNORM_BLOCK;       break;
      case REMIXAPI_FORMAT_BC3_SRGB:       vkFormat = VK_FORMAT_BC3_SRGB_BLOCK;        break;
      case REMIXAPI_FORMAT_BC5_UNORM:      vkFormat = VK_FORMAT_BC5_UNORM_BLOCK;       break;
      case REMIXAPI_FORMAT_BC7_UNORM:      vkFormat = VK_FORMAT_BC7_UNORM_BLOCK;       break;
      case REMIXAPI_FORMAT_BC7_SRGB:       vkFormat = VK_FORMAT_BC7_SRGB_BLOCK;        break;
      default:
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    // Create VkImage.
    DxvkImageCreateInfo imageInfo = {};
    imageInfo.type        = VK_IMAGE_TYPE_2D;
    imageInfo.format      = vkFormat;
    imageInfo.flags       = 0;
    imageInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.extent      = { info->width, info->height, info->depth > 0 ? info->depth : 1u };
    imageInfo.numLayers   = 1;
    imageInfo.mipLevels   = info->mipLevels > 0 ? info->mipLevels : 1u;
    imageInfo.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.stages      = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                          | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
    imageInfo.access      = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    imageInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.layout      = VK_IMAGE_LAYOUT_UNDEFINED;

    Rc<DxvkImage> image = remixDevice->GetDXVKDevice()->createImage(
      imageInfo,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXMaterialTexture,
      "Remix API uploaded texture");

    if (image == nullptr) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }

    // Create staging buffer for upload.
    DxvkBufferCreateInfo stagingInfo = {};
    stagingInfo.size   = info->dataSize;
    stagingInfo.usage  = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
    stagingInfo.access = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_WRITE_BIT;

    Rc<DxvkBuffer> stagingBuffer = remixDevice->GetDXVKDevice()->createBuffer(
      stagingInfo,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      DxvkMemoryStats::Category::RTXBuffer,
      "Remix API texture staging");

    if (stagingBuffer == nullptr) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }

    // Copy texture data to staging buffer.
    auto stagingSlice = DxvkBufferSlice { stagingBuffer };
    memcpy(stagingSlice.mapPtr(0), info->data, info->dataSize);

    // Create image view.
    DxvkImageViewCreateInfo viewInfo = {};
    viewInfo.type      = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format    = vkFormat;
    viewInfo.usage     = VK_IMAGE_USAGE_SAMPLED_BIT;
    viewInfo.aspect    = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.minLevel  = 0;
    viewInfo.numLevels = imageInfo.mipLevels;
    viewInfo.minLayer  = 0;
    viewInfo.numLayers = 1;

    Rc<DxvkImageView> imageView = remixDevice->GetDXVKDevice()->createImageView(image, viewInfo);

    if (imageView == nullptr) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }

    // Schedule upload on render thread.
    auto devLock = remixDevice->LockDevice();

    remixDevice->EmitCs([
      cHash          = info->hash,
      cImage         = image,
      cImageView     = imageView,
      cStagingBuffer = stagingBuffer,
      cBaseExtent    = imageInfo.extent,
      cMipLevels     = imageInfo.mipLevels,
      cDataSize      = info->dataSize,
      cFormat        = vkFormat
    ](DxvkContext* ctx) mutable {

      // Transition image to transfer dst.
      ctx->changeImageLayout(cImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

      // Upload each mip level. Use upstream's block-aware helpers so the
      // per-mip byte count is correct for compressed formats (BC1 = 8
      // bytes / 4x4 block, BC3/5/7 = 16 bytes / 4x4 block) as well as
      // uncompressed formats.
      VkDeviceSize offset = 0;

      for (uint32_t mip = 0; mip < cMipLevels; ++mip) {
        VkImageSubresourceLayers subresource = {};
        subresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        subresource.mipLevel       = mip;
        subresource.baseArrayLayer = 0;
        subresource.layerCount     = 1;

        const VkExtent3D   extent  = util::computeMipLevelExtent(cBaseExtent, mip);
        const VkDeviceSize mipSize = util::computeImageDataSize(cFormat, extent);

        if (offset + mipSize > cDataSize) {
          // Source data truncated for this mip; stop uploading rather than
          // read past the staging buffer.
          break;
        }

        ctx->copyBufferToImage(cImage, subresource, VkOffset3D { 0, 0, 0 }, extent,
                               cStagingBuffer, offset, 0, 0);

        offset += mipSize;

        if (offset >= cDataSize) {
          break;
        }
      }

      // Transition to shader read.
      ctx->changeImageLayout(cImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

      // Register texture with texture manager using hash.
      auto& textureManager = ctx->getCommonObjects()->getTextureManager();

      // TextureRef's getImageHash() uses the DxvkImage's hash (computed from
      // data), not the uniqueKey we provide. Set the image's hash to our
      // provided hash so the texture table lookup by image hash matches the
      // API-supplied value.
      cImage->setHash(cHash);

      auto textureRef = TextureRef(cImageView, cHash);

      // Add to texture table so materials can reference it by hash.
      uint32_t textureIndex;
      textureManager.addTexture(textureRef, 0, false, textureIndex);

      // Register with ImGui for categorization UI.
      // Flag 1 (kTextureFlagsDefault) allows assignment to texture categories.
      ctx->getCommonObjects()->getImgui().AddTexture(cHash, cImageView, 1);
    });

    *out_handle = handle;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // ---------------------------------------------------------------------------
  // destroyTexture (migration #7b)
  //
  // Schedules an EmitCs lambda that searches the texture table by hash and
  // releases the texture reference via RtxTextureManager::releaseTexture.
  // releaseTexture drops the entry from the SparseUniqueCache's internal map,
  // releasing the held Rc<DxvkImageView> / Rc<ManagedTexture> references. Once
  // no other holders remain, the GPU resources are refcount-released by DXVK.
  // ---------------------------------------------------------------------------
  remixapi_ErrorCode destroyTexture(
      D3D9DeviceEx*        remixDevice,
      remixapi_TextureHandle handle) {
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!handle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    auto devLock = remixDevice->LockDevice();

    remixDevice->EmitCs([cHash = reinterpret_cast<uint64_t>(handle)](DxvkContext* ctx) {
      auto& textureManager = ctx->getCommonObjects()->getTextureManager();

      // Find the texture entry registered for this hash and release it via
      // RtxTextureManager::releaseTexture. releaseTexture drops the entry
      // from the SparseUniqueCache's internal map, which in turn releases
      // the held Rc<DxvkImageView> / Rc<ManagedTexture> references.
      //
      // The cache's ref is NOT the only holder: createTexture also pushes
      // the image view into the ImGui texture-tagging catalog at
      // g_imguiTextureMap (see ImGUI::AddTexture above), which retains its
      // own Rc<DxvkImageView>. Without releasing that too, every plugin
      // CreateTexture permanently retains its VkImage -- a monotonic VRAM
      // leak under cell-reload workloads (plugin destroys, re-uploads,
      // destroys again; ImGui keeps every generation alive). Match the
      // AddTexture call with its corresponding ReleaseTexture here.
      ctx->getCommonObjects()->getImgui().ReleaseTexture(cHash);

      const auto& textureTable = textureManager.getTextureTable();
      for (const auto& textureRef : textureTable) {
        if (textureRef.isValid() && textureRef.getImageHash() == cHash) {
          // releaseTexture takes a non-const reference for lookup only; copy
          // so we don't alias an entry that releaseTexture will invalidate.
          TextureRef toRelease = textureRef;
          textureManager.releaseTexture(toRelease);
          break;
        }
      }
    });

    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // ---------------------------------------------------------------------------
  // notifyBeginScene (migration #7c)
  //
  // Fires s_beginCallback on the first API submission of the frame.
  // Uses an atomic exchange so only the first caller per frame triggers the
  // callback regardless of which API path (DrawInstance / DrawLightInstance)
  // reaches it first. Called while the remix-api s_mutex is NOT held (the
  // exchange itself is atomic; callback must be re-read after the exchange to
  // avoid a null race).
  // ---------------------------------------------------------------------------
  void notifyBeginScene() {
    if (!s_inFrame.exchange(true)) {
      auto cb = s_beginCallback;
      if (cb) {
        cb();
      }
    }
  }

  // ---------------------------------------------------------------------------
  // registerCallbacks (migration #7c)
  //
  // Stores the three per-frame bridge callbacks into fork-owned state.
  // Called from the one-liner remixapi_RegisterCallbacks delegate in upstream.
  // ---------------------------------------------------------------------------
  void registerCallbacks(
      PFN_remixapi_BridgeCallback beginSceneCallback,
      PFN_remixapi_BridgeCallback endSceneCallback,
      PFN_remixapi_BridgeCallback presentCallback) {
    s_beginCallback   = beginSceneCallback;
    s_endCallback     = endSceneCallback;
    s_presentCallback = presentCallback;
  }

  // ---------------------------------------------------------------------------
  // shutdownCallbacks (migration #7c)
  //
  // Resets all four frame-boundary state variables to null/false. Called from
  // remixapi_Shutdown before the device is released.
  // ---------------------------------------------------------------------------
  void shutdownCallbacks() {
    s_beginCallback   = nullptr;
    s_endCallback     = nullptr;
    s_presentCallback = nullptr;
    s_inFrame.store(false);
  }

  // ---------------------------------------------------------------------------
  // presentEndSceneDispatch (migration #7c)
  //
  // Fires the endScene callback if s_inFrame indicates a frame was started.
  // Called from remixapi_Present immediately BEFORE the native Present so the
  // endScene callback fires before the GPU flip.
  // ---------------------------------------------------------------------------
  void presentEndSceneDispatch() {
    if (s_inFrame.load()) {
      auto cb = s_endCallback;
      if (cb) {
        cb();
      }
    }
  }

  // ---------------------------------------------------------------------------
  // presentCallbackDispatch (migration #7c)
  //
  // Fires the present callback and resets s_inFrame to false. Called from
  // remixapi_Present immediately AFTER the native Present call returns
  // successfully. The present callback fires post-flip; s_inFrame is reset
  // here rather than in presentEndSceneDispatch so that any code between the
  // two hooks can still query the in-frame state if needed.
  // ---------------------------------------------------------------------------
  void presentCallbackDispatch() {
    {
      auto cb = s_presentCallback;
      if (cb) {
        cb();
      }
    }
    s_inFrame.store(false);
  }

  // ---------------------------------------------------------------------------
  // remixApiVtableInit (migration #7c)
  //
  // Populates the three fork-added extern-"C"-linked function-pointer slots
  // in the remixapi_Interface vtable. Called from remixapi_InitializeLibrary
  // after the upstream and anonymous-namespace fork slots are assigned inline.
  //
  // Only extern-"C" REMIXAPI-exported functions can be named from this TU.
  // Anonymous-namespace additions (AddTextureHash, CreateTexture, GetUIState,
  // CreateLightBatched, etc.) are assigned inline in rtx_remix_api.cpp where
  // their symbols are visible.
  //
  // The sizeof(interf) static_assert sentinel is retained inline in upstream;
  // it is not repeated here.
  // ---------------------------------------------------------------------------
  void remixApiVtableInit(remixapi_Interface& interf) {
    interf.RegisterCallbacks              = remixapi_RegisterCallbacks;
    interf.AutoInstancePersistentLights   = remixapi_AutoInstancePersistentLights;
    interf.UpdateLightDefinition          = remixapi_UpdateLightDefinition;
  }

  // ---------------------------------------------------------------------------
  // requestVramCompaction (migration #7d)
  //
  // Sets an atomic flag that the render thread consumes in
  // SceneManager::manageTextureVram on the next tick. Flag mutation is
  // lock-free; the remix-api static mutex is deliberately NOT taken by the
  // caller.
  // ---------------------------------------------------------------------------
  remixapi_ErrorCode requestVramCompaction(D3D9DeviceEx* remixDevice) {
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    remixDevice->GetDXVKDevice()->getCommon()->getSceneManager().requestVramCompaction();
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // ---------------------------------------------------------------------------
  // requestTextureVramFree (migration #7d)
  //
  // Sets an atomic flag that the render thread consumes in
  // SceneManager::manageTextureVram on the next tick. The tick calls
  // textureManager.clear() (SparseUniqueCache wipe + budget-gated streaming
  // demotion), matching the DX9 path's scene-transition behavior exposed to
  // plugins. Lock-free atomic mutation so the remix-api static mutex is not
  // taken by the caller.
  // ---------------------------------------------------------------------------
  remixapi_ErrorCode requestTextureVramFree(D3D9DeviceEx* remixDevice) {
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    remixDevice->GetDXVKDevice()->getCommon()->getSceneManager().requestTextureVramFree();
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // ---------------------------------------------------------------------------
  // getVramStats (migration #7d)
  //
  // Fills a remixapi_VramStats struct with DXVK per-category totals plus
  // driver-view heap info (matches Task Manager / nvidia-smi; gap vs
  // totalAllocatedBytes exposes non-DXVK allocations such as NGX, RT pipeline
  // state, descriptor pools, NRC, etc.) and the fork-side texture manager's
  // table size.
  //
  // forkTextureCacheCount comes from RtxTextureManager::getTextureTable()
  // (the backing SparseUniqueCache's object vector). Its length is the cache's
  // high-water mark rather than the live entry count (freed slots become
  // sentinels rather than being removed), but growth here is a proxy for
  // "more textures are getting tracked by the fork," useful for catching
  // fork-side native-texture accumulation when plugin caches are stable.
  // ---------------------------------------------------------------------------
  remixapi_ErrorCode getVramStats(
      D3D9DeviceEx*        remixDevice,
      remixapi_VramStats*  out_stats) {
    if (!out_stats) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }

    *out_stats = {};

    auto device = remixDevice->GetDXVKDevice();
    const VkPhysicalDeviceMemoryProperties memProps = device->adapter()->memoryProperties();
    for (uint32_t i = 0; i < memProps.memoryHeapCount; i++) {
      const bool isDeviceLocal = (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
      if (!isDeviceLocal) {
        continue;
      }
      const DxvkMemoryStats stats = device->getMemoryStats(i);
      out_stats->totalAllocatedBytes               += stats.totalAllocated();
      out_stats->totalUsedBytes                    += stats.totalUsed();
      out_stats->usedReplacementGeometryBytes      += stats.usedByCategory(DxvkMemoryStats::Category::RTXReplacementGeometry);
      out_stats->usedBufferBytes                   += stats.usedByCategory(DxvkMemoryStats::Category::RTXBuffer);
      out_stats->usedAccelerationStructureBytes    += stats.usedByCategory(DxvkMemoryStats::Category::RTXAccelerationStructure);
      out_stats->usedOpacityMicromapBytes          += stats.usedByCategory(DxvkMemoryStats::Category::RTXOpacityMicromap);
      out_stats->usedMaterialTextureBytes          += stats.usedByCategory(DxvkMemoryStats::Category::RTXMaterialTexture);
      out_stats->usedRenderTargetBytes             += stats.usedByCategory(DxvkMemoryStats::Category::RTXRenderTarget);
    }
    out_stats->poolRetainedBytes =
      (out_stats->totalAllocatedBytes > out_stats->totalUsedBytes)
        ? (out_stats->totalAllocatedBytes - out_stats->totalUsedBytes)
        : 0;

    const DxvkAdapterMemoryInfo memHeapInfo = device->adapter()->getMemoryHeapInfo();
    for (uint32_t i = 0; i < memHeapInfo.heapCount; i++) {
      if (memHeapInfo.heaps[i].heapFlags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
        out_stats->driverAllocatedBytes += memHeapInfo.heaps[i].memoryAllocated;
        out_stats->driverBudgetBytes    += memHeapInfo.heaps[i].memoryBudget;
      }
    }

    out_stats->forkTextureCacheCount =
      static_cast<uint32_t>(device->getCommon()->getTextureManager().getTextureTable().size());

    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

} // namespace fork_hooks
} // namespace dxvk
