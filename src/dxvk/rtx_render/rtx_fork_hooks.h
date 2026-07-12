#pragma once

// rtx_fork_hooks.h — declarations for the fork-owned hook functions that
// upstream files call into. Each hook's implementation lives in a dedicated
// rtx_fork_*.cpp file, keeping upstream files' fork footprint to one-line
// call sites only.
//
// See docs/fork-touchpoints.md for the index of every hook and which
// upstream file calls it.

// Brings in AssetReplacer, AssetReplacement, MaterialData, DrawCallState,
// and XXH64_hash_t transitively via rtx_types.h.
#include "rtx_asset_replacer.h"

// util/rc for Rc<DxvkContext> in capture hook signatures.
#include "../../util/rc/util_rc_ptr.h"

// std::filesystem::path for textureHashPathLookup's input parameter.
#include <filesystem>

// remixapi_ErrorCode for mutateTextureHashOption return type.
#include <remix/remix_c.h>

// Windows types required for the overlay hooks (HWND, UINT, WPARAM, LPARAM).
// This project is Windows-only; WIN32_LEAN_AND_MEAN keeps the include small.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

// Forward-declare lss::Export for the coord-system hook (avoids pulling the
// full USD pxr include chain into this header).
namespace lss {
  struct Export;
} // namespace lss

// remixapi_LightHandle for the light-manager hooks. Guard against redefinition
// when rtx_light_manager.h is also included in the same translation unit.
#ifndef REMIXAPI_LIGHTHANDLE_DEFINED
#define REMIXAPI_LIGHTHANDLE_DEFINED
using remixapi_LightHandle = struct remixapi_LightHandle_T*;
#endif

// d3d9_device.h provides D3D9DeviceEx; avoid pulling it into every TU.
// Only the implementation files that call device-path hooks include it directly.
namespace dxvk { class D3D9DeviceEx; }

// RaytraceArgs (for updateAtmosphereConstants) and Resources::RaytracingOutput
// (for dispatchScreenOverlay) both require the full type to appear in a function
// declaration. rtx_resources.h is the canonical header for both; it is already
// pulled transitively by most translation units that include rtx_fork_hooks.h.
#include "rtx_resources.h"
#include "rtx/pass/raytrace_args.h"

// Tonemap shader args structs (ToneMappingApplyToneMappingArgs)
// needed by fork_hooks::populateTonemapOperatorArgs.
#include "rtx/pass/tonemap/tonemapping.h"

namespace dxvk {

  // Forward declarations for types whose full definitions the hook header does
  // not need, but whose names appear in hook signatures.
  class DxvkContext;
  class DxvkDevice;
  class GameCapturer;
  class GameOverlay;
  struct LightManager;
  class RtInstance;
  class RtxContext;
  class SceneManager;
  struct LegacyMaterialData;
  struct RtLight;
  struct TextureRef;

  namespace fork_hooks {

    // Constructs the RtxAtmosphere instance during RtxContext initialization.
    // Must be called after GlobalTime::get().init() in the RtxContext constructor.
    // NOTE: requires RtxContext to declare this as a friend for access to the
    // private m_atmosphere and m_device members. See rtx_context.h.
    // Implementation in rtx_fork_atmosphere.cpp.
    void initAtmosphere(RtxContext& ctx);

    // Sets constants.skyMode, detects sky mode transitions (clearing skybox
    // buffers when switching to Numos), and when Numos
    // is active ensures m_atmosphere exists, calls initialize /
    // computeLuts, and writes atmosphereArgs into the constant block.
    // NOTE: requires RtxContext to declare this as a friend for access to private
    // members m_atmosphere, m_lastSkyMode, m_skyColorFormat, m_skyRtColorFormat,
    // and m_device. See rtx_context.h.
    // Implementation in rtx_fork_atmosphere.cpp.
    void updateAtmosphereConstants(RtxContext& ctx, RaytraceArgs& constants);

    // Ensures m_atmosphere is initialized (idempotent) and binds the three
    // atmosphere LUT textures unconditionally, because they are declared in
    // common_bindings.slangh for all shader passes.
    // NOTE: requires RtxContext to declare this as a friend for access to private
    // members m_atmosphere and m_device. See rtx_context.h.
    // Implementation in rtx_fork_atmosphere.cpp.
    void bindAtmosphereLuts(RtxContext& ctx);

    // Returns true when the caller should skip rasterized sky rendering because
    // Numos mode is active.
    // No private-member access — uses only the public RtxOptions::skyMode() API.
    // No friend declaration needed.
    // Implementation in rtx_fork_atmosphere.cpp.
    bool injectRtxAtmosphereSkySkip();

    // Checks for a USD mesh/light replacement keyed on the API mesh handle hash.
    // Returns the replacement vector if one exists, null otherwise.
    // Call site is responsible for calling determineMaterialData + drawReplacements
    // and returning early when a non-null value is returned.
    // Implementation in rtx_fork_submit.cpp.
    std::vector<AssetReplacement>* externalDrawMeshReplacement(
      AssetReplacer& replacer, XXH64_hash_t meshHash);

    // Checks for a USD material replacement and updates the material pointer in-place.
    // Implementation in rtx_fork_submit.cpp.
    void externalDrawMaterialReplacement(
      AssetReplacer& replacer, const MaterialData*& material);

    // Resolves the albedo texture hash from an API material and auto-applies
    // all texture-based instance categories (Sky, Ignore, WorldUI, etc.).
    // Writes textureHash out for use by subsequent hooks.
    // Implementation in rtx_fork_submit.cpp.
    void externalDrawTextureCategories(
      const MaterialData* material,
      DrawCallState& drawCall,
      XXH64_hash_t& textureHash);

    // Stores per-draw texture hash metadata in SceneManager::m_drawCallMeta
    // when object picking is active, mirroring the D3D9 draw path.
    // NOTE: requires SceneManager to declare fork_hooks::externalDrawObjectPicking
    // as a friend (or m_drawCallMeta to be made public) for the implementation
    // in rtx_fork_submit.cpp to compile. Flagged for Phase 4 fixup.
    // Implementation in rtx_fork_submit.cpp.
    void externalDrawObjectPicking(
      DxvkDevice& device,
      DrawCallState& drawCall,
      XXH64_hash_t textureHash,
      SceneManager& scene);

    // Forwards keyboard (WM_KEY*, WM_CHAR, WM_SYSCHAR) AND mouse
    // (WM_MOUSEMOVE, WM_{L,R,M,X}BUTTON*, WM_MOUSE{,H}WHEEL) messages to
    // ImGui_ImplWin32_WndProcHandler so ImGui's keyboard + mouse state stays
    // in sync on the legacy WndProc path. Used when a game menu captures raw
    // input OR the plugin HUD pulls focus via the Remix API — either case
    // stops overlayWndProc from receiving messages directly and the legacy
    // wndProcHandler fallback becomes the only delivery path.
    // Mouse coords in lParam are translated from gameHwnd client-space to
    // overlayHwnd client-space when the two differ, so ImGui hit-tests
    // correctly. Wheel lParam is screen-space per Windows convention and
    // forwards without translation.
    // NOTE: requires GameOverlay to declare this as a friend for access to the
    // private m_hwnd member. See rtx_overlay_window.h.
    // Implementation in rtx_fork_overlay.cpp.
    void overlayInputForward(
      GameOverlay& overlay, HWND gameHwnd, UINT msg, WPARAM wParam, LPARAM lParam);

    // Alpha-composites a plugin-uploaded RGBA pixel buffer over the final
    // tone-mapped output image using the ScreenOverlayShader compute shader.
    // Called from RtxContext::dispatchScreenOverlay (one-line delegate) after
    // tone mapping and before screenshot capture. No-ops when no overlay is
    // pending this frame.
    // NOTE: requires RtxContext to declare this as a friend for access to
    // private members m_pendingScreenOverlay, m_screenOverlay*, and m_device.
    // See rtx_context.h.
    // Implementation in rtx_fork_overlay.cpp.
    void dispatchScreenOverlay(RtxContext& ctx, Resources::RaytracingOutput& rtOutput);

    // Implements the full captureMaterial logic for both D3D9 and API-submitted
    // materials. For D3D9 materials, exports the color texture directly. For
    // API-submitted materials (colorTexture invalid), resolves the albedo texture
    // via the texture-manager table and exports it by hash. Caches the result in
    // GameCapturer::m_pCap->materials keyed by runtimeMaterialHash.
    // NOTE: requires GameCapturer to declare this as a friend for access to
    // private members m_exporter and m_pCap. See rtx_game_capturer.h.
    // Implementation in rtx_fork_capture.cpp.
    void captureMaterialApiPath(
      GameCapturer& capturer,
      const Rc<DxvkContext> ctx,
      const RtInstance& rtInstance,
      XXH64_hash_t runtimeMaterialHash,
      const LegacyMaterialData& materialData,
      bool bEnableOpacity);

    // Applies the coordinate-system transform for the USD export stage.
    // Skips the view/proj handedness inversion when the game is configured
    // as a left-handed coordinate system, since external API content is
    // already in consistent Y-up space.
    // Implementation in rtx_fork_capture.cpp.
    void captureCoordSystemSkip(lss::Export& exportPrep);

    // Drains the three deferred external-light mutation queues (erases, updates,
    // activations) and auto-instances persistent lights. Called at the top of
    // LightManager::prepareSceneData before linearization.
    // NOTE: requires LightManager to declare this as a friend for access to
    // private members. See rtx_light_manager.h.
    // Implementation in rtx_fork_light.cpp.
    void flushPendingLightMutations(LightManager& mgr);

    // Shared static-sleep update logic for both the indexed (externally-tracked)
    // and hash-map (external API) light paths. Preserves temporal denoiser data
    // by skipping the copy once isStaticCount >= numFramesToPutLightsToSleep.
    // Pass externalId = kInvalidExternallyTrackedLightId to skip id restoration
    // (hash-map path). Stamps frameLastTouched unconditionally.
    // Implementation in rtx_fork_light.cpp.
    void updateLightStaticSleep(
      RtLight* light,
      const RtLight& newLight,
      DxvkDevice* device,
      uint64_t externalId);

    // Per-frame weather preset blender update. Reads __weather.target and
    // __weather.blend_seconds from the GameStateStore and writes blended weather
    // params to the Derived layer of their underlying RTX_OPTIONs. Dormant when
    // no target is set — zero behavioural change vs upstream.
    // NOTE: requires RtxContext to declare this as a friend for access to the
    // private m_weatherBlender member. See rtx_context.h. (Wired in Task 3.)
    // Implementation in rtx_fork_weather.cpp.
    void updateWeatherBlender(class RtxContext& ctx, float deltaTimeSeconds);

    // Emplaces a new external light into m_externalLights and stamps
    // frameLastTouched. Called from the "new light" branch of addExternalLight.
    // NOTE: requires LightManager to declare this as a friend for access to
    // m_externalLights. See rtx_light_manager.h.
    // Implementation in rtx_fork_light.cpp.
    void setExternalLightEmplace(
      LightManager& mgr,
      remixapi_LightHandle handle,
      const RtLight& rtlight);

    // Queues a light handle for deferred erase in m_pendingExternalLightErases.
    // Called from LightManager::removeExternalLight.
    // NOTE: requires LightManager to declare this as a friend for access to
    // m_pendingExternalLightErases. See rtx_light_manager.h.
    // Implementation in rtx_fork_light.cpp.
    void disableExternalLightQueue(LightManager& mgr, remixapi_LightHandle handle);

    // Inserts a handle into m_persistentExternalLights if non-null.
    // Called from LightManager::registerPersistentExternalLight.
    // NOTE: requires LightManager to declare this as a friend for access to
    // m_persistentExternalLights. See rtx_light_manager.h.
    // Implementation in rtx_fork_light.cpp.
    void registerPersistentLight(LightManager& mgr, remixapi_LightHandle handle);

    // Removes a handle from m_persistentExternalLights if non-null.
    // Called from LightManager::unregisterPersistentExternalLight.
    // NOTE: requires LightManager to declare this as a friend for access to
    // m_persistentExternalLights. See rtx_light_manager.h.
    // Implementation in rtx_fork_light.cpp.
    void unregisterPersistentLight(LightManager& mgr, remixapi_LightHandle handle);

    // Copies all persistent-light handles into m_pendingExternalActiveLights.
    // Called from LightManager::queueAutoInstancePersistent.
    // NOTE: requires LightManager to declare this as a friend for access to
    // m_persistentExternalLights and m_pendingExternalActiveLights. See
    // rtx_light_manager.h.
    // Implementation in rtx_fork_light.cpp.
    void queueAutoInstancePersistent(LightManager& mgr);

    // Pins ImGui and ImPlot to the dev menu's private contexts at wndProcHandler
    // entry. Prevents context corruption when plugin activity on other threads
    // drifts GImGui off the dev menu's context between frames.
    // Call site passes m_context and m_plotContext from ImGUI directly — no friend
    // declaration needed.
    // Implementation in rtx_fork_overlay.cpp.
    void imguiContextPin(struct ImGuiContext* ctx, struct ImPlotContext* plotCtx);

    // Renders the atmosphere preset buttons and parameter tree inside the
    // "Sky Tuning" collapsing header. Owns the skyModeCombo static and branches
    // on SkyMode::Numos vs SkyboxRasterization.
    // No private-member access — uses only public RtxOptions and ImGui APIs.
    // No friend declaration needed.
    // Implementation in rtx_fork_atmosphere.cpp.
    void showAtmosphereUI();

    // Invokes the registered plugin draw callback for the Plugin tab in the dev
    // menu. Called from the kTab_Wrapper switch case in ImGUI::showMainMenu.
    // No private-member access — delegates to remixapi_imgui_InvokeDrawCallback().
    // No friend declaration needed.
    // Implementation in rtx_fork_overlay.cpp.
    void wrapperTabDraw();

    // Attempts to resolve a material texture path of the form "0x<hex>" against
    // the texture manager's hash table (populated by API-uploaded textures via
    // remixapi_CreateTexture). If the path matches the hex-hash pattern and a
    // registered texture with that image hash exists, writes the resolved
    // TextureRef into outRef and returns true. Returns false in all other
    // cases (path is not a hex string, hash not found, parse failure), in
    // which case the caller must fall back to the normal asset-path lookup.
    // No private-member access — uses public TextureManager::getTextureTable().
    // No friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    bool textureHashPathLookup(
      DxvkContext& ctx,
      const std::filesystem::path& path,
      TextureRef& outRef);

    // Looks up an RtxOption<fast_unordered_set> by full option name and adds
    // (add == true) or removes (add == false) the parsed hash in the user
    // config layer. Used by remixapi_AddTextureHash / remixapi_RemoveTextureHash
    // to mutate per-category texture hash sets at runtime.
    // NOTE: the caller is responsible for holding the remix-api static mutex
    // (s_mutex in rtx_remix_api.cpp) across this call, per the lock ordering
    // rule documented alongside s_mutex.
    // No private-member access — uses only public RtxOption / RtxOptionLayer APIs.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode mutateTextureHashOption(
      const char* textureCategory,
      const char* textureHash,
      bool add);

    // Copies pPixelData into a host-visible staging buffer and stores it in the
    // fork-owned s_pendingScreenOverlay optional. A null pPixelData or zero dims
    // clears the pending overlay. The overlay is flushed to the render thread by
    // presentScreenOverlayFlush at the next present boundary.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode drawScreenOverlay(
      D3D9DeviceEx* remixDevice,
      const void*   pPixelData,
      uint32_t      width,
      uint32_t      height,
      remixapi_Format format,
      float         opacity);

    // Drains s_pendingScreenOverlay (if set) onto the render thread via EmitCs,
    // forwarding the staging buffer to RtxContext::setScreenOverlayData. Called
    // from both remixapi_Present paths (inner-namespace and extern-C) after the
    // light/mesh flush EmitCs and before the endScene callback dispatch.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    void presentScreenOverlayFlush(D3D9DeviceEx* remixDevice);

    // Reads RtxOptions::showUI() and maps the internal UIType to remixapi_UIState.
    // Returns REMIXAPI_UI_STATE_NONE if the device is not registered.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_UIState getUiState(D3D9DeviceEx* remixDevice);

    // Acquires the device lock and calls getImgui().switchMenu(uiType) after
    // mapping the remixapi_UIState to internal UIType. Returns GENERAL_FAILURE
    // if the device or its common objects are null.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode setUiState(D3D9DeviceEx* remixDevice, remixapi_UIState state);

    // Stub: the DX11 shared-memory export backbuffer path is not ported to this
    // fork. Validates arguments then returns GENERAL_FAILURE so callers fall back;
    // the vtable slot is populated so the struct layout matches the plugin ABI.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode getSharedD3D11TextureHandle(
      D3D9DeviceEx* remixDevice,
      void**       out_sharedHandle,
      uint32_t*    out_width,
      uint32_t*    out_height);

    // Retrieves the D3D9CommonTexture from a D3D9 texture pointer, gets the
    // underlying DxvkImage, and returns image->getHash() in out_hash.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode dxvkGetTextureHash(
      IDirect3DTexture9* texture,
      uint64_t*          out_hash);

    // Creates a DxvkImage + view + staging buffer for the supplied pixel data,
    // copies data into staging, schedules an EmitCs lambda that transitions the
    // image, uploads all mip levels, transitions to shader-read, and registers
    // the texture with the texture manager and ImGui catalog.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode createTexture(
      D3D9DeviceEx*            remixDevice,
      const remixapi_TextureInfo* info,
      remixapi_TextureHandle*  out_handle);

    // Schedules an EmitCs lambda that searches the texture table by hash and
    // releases the texture reference via RtxTextureManager::releaseTexture.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode destroyTexture(
      D3D9DeviceEx*        remixDevice,
      remixapi_TextureHandle handle);

    // Fires the beginScene callback on the first submission of the frame
    // (whichever API call establishes s_inFrame first). Atomically exchanges
    // s_inFrame to true; calls s_beginCallback once per frame when it was false.
    // Called from remixapi_DrawInstance and remixapi_DrawLightInstance.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    void notifyBeginScene();

    // Stores the three per-frame bridge callbacks. Called from the
    // remixapi_RegisterCallbacks one-liner delegate in rtx_remix_api.cpp.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    void registerCallbacks(
      PFN_remixapi_BridgeCallback beginSceneCallback,
      PFN_remixapi_BridgeCallback endSceneCallback,
      PFN_remixapi_BridgeCallback presentCallback);

    // Clears all four frame-boundary state vars (s_inFrame, s_beginCallback,
    // s_endCallback, s_presentCallback) to their null/false defaults. Called
    // from remixapi_Shutdown.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    void shutdownCallbacks();

    // Fires the endScene callback if s_inFrame is set (i.e. the frame was
    // started via DrawInstance or DrawLightInstance). Called from
    // remixapi_Present immediately before the native remixDevice->Present()
    // so the endScene callback fires before GPU presentation.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    void presentEndSceneDispatch();

    // Fires the present callback and resets s_inFrame to false. Called from
    // remixapi_Present immediately after the native remixDevice->Present()
    // returns successfully. Separated from presentEndSceneDispatch because the
    // native Present call sits between the two callback fires.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    void presentCallbackDispatch();

    // Populates the three fork-added extern-C-linked vtable slots
    // (RegisterCallbacks, AutoInstancePersistentLights, UpdateLightDefinition)
    // in the remixapi_Interface vtable. Called from remixapi_InitializeLibrary
    // after the upstream and anonymous-namespace fork slots are filled inline.
    // Anonymous-namespace fork additions (AddTextureHash, CreateTexture,
    // GetUIState, CreateLightBatched, etc.) are assigned inline in upstream
    // where their symbols are visible — only externally linked symbols can be
    // named from this translation unit.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    void remixApiVtableInit(remixapi_Interface& interf);

    // Sets SceneManager's atomic VRAM-compaction request flag; the render
    // thread consumes it in manageTextureVram on the next tick. Returns
    // REMIX_DEVICE_WAS_NOT_REGISTERED if remixDevice is null. Lock-free;
    // callers need not hold the remix-api static mutex.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode requestVramCompaction(D3D9DeviceEx* remixDevice);

    // Sets SceneManager's atomic texture-VRAM-free request flag; the render
    // thread consumes it in manageTextureVram on the next tick (which then
    // calls textureManager.clear()). Returns REMIX_DEVICE_WAS_NOT_REGISTERED
    // if remixDevice is null. Lock-free; callers need not hold the remix-api
    // static mutex.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode requestTextureVramFree(D3D9DeviceEx* remixDevice);

    // Fills out_stats with DXVK per-category totals, driver-view heap info
    // (matches Task Manager / nvidia-smi — gap vs totalAllocatedBytes exposes
    // non-DXVK allocations such as NGX, RT pipeline state, descriptor pools,
    // NRC, etc.), and the fork-side texture manager's table size. Returns
    // INVALID_ARGUMENTS if out_stats is null or REMIX_DEVICE_WAS_NOT_REGISTERED
    // if remixDevice is null.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode getVramStats(
      D3D9DeviceEx*        remixDevice,
      remixapi_VramStats*  out_stats);

    // Populates the tonemap-operator-related fields of the global tonemapper's
    // shader args struct (tonemapOperator + per-operator param blocks for
    // Hable, AgX, Lottes, Psycho17). Called from
    // DxvkToneMapping::dispatchApplyToneMapping.
    // No private-member access — uses public RtxOption accessors only.
    // Implementation in rtx_fork_tonemap.cpp.
    void populateTonemapOperatorArgs(ToneMappingApplyToneMappingArgs& args);

    // Renders the Tonemapping Operator combo + per-operator parameter sliders
    // inside DxvkToneMapping::showImguiSettings. Called from the fork-hook call
    // site replacing the old "Finalize With ACES" checkbox.
    // No private-member access — uses only public RtxOption / RemixGui / ImGui APIs.
    // Implementation in rtx_fork_tonemap.cpp.
    void showTonemapOperatorUI();

    // Renders the weather preset UI inside the existing atmosphere ImGui tree.
    // Includes preset dropdown, blend duration slider, current/target/progress
    // display, "Pause Weather Blender" toggle, and per-preset slider sub-trees.
    // No private-member access. (Wired in Task 4.)
    // Implementation in rtx_fork_weather.cpp.
    void showWeatherUI();

  } // namespace fork_hooks

} // namespace dxvk
