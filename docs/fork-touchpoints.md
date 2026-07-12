# Fork touchpoints

This index lists every upstream file the fork touches. It is the authoritative
inventory of fork-vs-upstream surface area, maintained as fork edits are added
or removed.

See [`docs/CONTRIBUTING.md`](CONTRIBUTING.md) for the fork-touchpoint
discipline this index supports.

## Conventions

### Fork-owned file naming

- Fork-owned files use the `rtx_fork_*` prefix with a subsystem suffix
  (e.g. `rtx_fork_api_entry.cpp`, `rtx_fork_atmosphere.cpp`,
  `rtx_fork_overlay.cpp`, `rtx_fork_light.cpp`). Single prefix keeps the
  convention simple and `grep`-friendly.
- All fork-owned files live under `src/dxvk/rtx_render/` (or the
  subsystem-appropriate equivalent directory).
- Hook functions are declared in the `fork_hooks::` namespace
  (`src/dxvk/rtx_render/rtx_fork_hooks.h`) and implemented in their
  respective fork-owned `.cpp` files.

### Fridge-list invariant

Every edit to an upstream file must have a fridge-list entry in the
same commit. The PR-template bullet reminds contributors; a future CI
check will enforce it if discipline slips.

## Entry types

- **Hook** — upstream file contains a one-line call into fork-owned code. The
  fork logic itself lives in the fork-owned file referenced by the entry.
- **Inline tweak** — upstream file contains a small fork-introduced change
  (typically <= 20 LOC) that was not worth lifting into its own fork file.

## Upstream files touched by the fork

<!-- Entries are sorted alphabetically by upstream file path. -->

---

## public/include/remix/remix.h

**Pre-refactor fork footprint:** +101 / -9 LOC (audit 2026-04-18)

**Category:** migrate

- **Block** at `Interface` class (method declarations) — ~13 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Declares fork-added C++ wrapper methods: `CreateMeshBatched`, `GetUIState`, `SetUIState`, `AddTextureHash`, `RemoveTextureHash`, `dxvk_GetTextureHash`, `CreateLightBatched`, `UpdateLightDefinition`, `SetGameValue`.*

- **Block** at `Interface::CreateMeshBatched` (inline definition) — ~9 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Inline C++ wrapper that calls `m_CInterface.CreateMeshBatched` for the batched mesh submission API slot.*

- **Block** at `Interface::GetUIState` / `Interface::SetUIState` (inline definitions) — ~16 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Inline C++ wrappers for UI state query/set API, guarding on nullptr slot before dispatching.*

- **Block** at `Interface::AddTextureHash` / `Interface::RemoveTextureHash` (inline definitions) — ~16 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Inline C++ wrappers for texture-hash category mutation API.*

- **Block** at `Interface::dxvk_GetTextureHash` (inline definition) — ~13 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Inline C++ wrapper that retrieves the DXVK image hash for a D3D9 texture via the fork's dxvk-specific extension slot.*

- **Block** at `Interface::CreateLightBatched` / `Interface::UpdateLightDefinition` (inline definitions) — ~22 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Inline C++ wrappers for batched light creation and deferred light-definition update.*

- **Block** at `UIState` enum (file scope) — ~5 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Adds `UIState` C++ enum mirroring `remixapi_UIState` for the C++ API surface.*

- **Block** at `Interface::SetGameValue` (declaration + inline definition) — ~8 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *C++ wrapper for the `remixapi_SetGameValue` C API slot introduced in workstream 10 (plugin-injected game-state write). Wrapper guards on nullptr vtable slot before dispatching, matching the `SetConfigVariable` shape. Companion readers are graph components `GameValueReadBool` / `GameValueReadNumber`; backing store lives in `rtx_fork_game_state.h`.*

- **Block** at `remixapi_Interface` static_assert updates (file scope) — ~3 LOC (three separate assert sizes), planned target `N/A (public header)` in `N/A (public header)`.
  *Updates `sizeof(remixapi_Interface)` static_asserts in the C++ header to match each successive vtable extension (208 → 240 → 272 → 280 → 288).*

---

## public/include/remix/remix_c.h

**Pre-refactor fork footprint:** +154 / -29 LOC (audit 2026-04-18)

**Category:** migrate

- **Block** at `REMIXAPI_VERSION_MAJOR/MINOR/PATCH` (file scope) — ~3 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Sets the Remix Plus ABI version to `0.1000.0` (reserved MINOR `1000`), distinct from stock NVIDIA `0.6.x`. Because `isVersionCompatible` treats each minor as breaking while MAJOR==0, this makes the runtime reject binaries built against stock Remix `0.6.x` or older Remix Plus `0.6.x` — whose `remixapi_Interface` layout and category-bit ABI differ. Bump MINOR on any further breaking ABI change.*

- **Block** at `remixapi_StructType` enum (file scope) — ~3 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Adds `REMIXAPI_STRUCT_TYPE_TEXTURE_INFO`, `INSTANCE_INFO_PARTICLE_SYSTEM_EXT`, and `INSTANCE_INFO_GPU_INSTANCING_EXT` enumerators.*

- **Block** at `remixapi_TextureHandle` typedef (file scope) — ~1 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Declares the opaque `remixapi_TextureHandle_T*` handle type for the texture upload API.*

- **Block** at `REMIXAPI_INSTANCE_CATEGORY_BIT_*` enum (file scope) — ~16 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Bit values match upstream NVIDIA exactly (reverted 2026-06-27 from an earlier fork build that shifted `IGNORE_ALPHA_CHANNEL` to bit 8 to mirror the internal `InstanceCategories` order). The C↔internal mapping in `toRtCategories()` is by-name, so the public bit values are free to match upstream and now do. No remaining fork delta — the enum now matches upstream exactly. (The misleadingly-named `LEGACY_EMISSIVE` alias of bit 24 / `SMOOTH_NORMALS` was removed 2026-06-28: its name implied emissive behavior but it routed to `SmoothNormals`, so callers got a silent wrong-category result; removing it converts that into a compile error.)*

- **Block** at `IDirect3DTexture9` forward declaration (file scope) — ~1 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Forward-declares `IDirect3DTexture9` so the dxvk-extension function signatures compile without pulling in d3d9 headers.*

- **Block** at `remixapi_InstanceInfo.isDynamic` field (struct) — ~1 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Adds `isDynamic` bool field to `remixapi_InstanceInfo` to control temporal accumulation behavior.*

- **Block** at `remixapi_InstanceInfo.ignoreViewModel` field (struct) — ~1 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Adds `ignoreViewModel` bool field to `remixapi_LightInfo` so API-submitted lights can opt out of view-model geometry lighting.*

- **Block** at `PFN_remixapi_AddTextureHash` / `PFN_remixapi_RemoveTextureHash` typedefs (file scope) — ~8 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Declares function-pointer types for texture-hash category mutation (add/remove a texture hash from a named option set).*

- **Block** at `remixapi_Format` enum + `remixapi_TextureInfo` struct (file scope) — ~28 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Declares the texture upload type system: format enum mapping to VkFormat values, and the `remixapi_TextureInfo` struct carrying pixel data for `CreateTexture`.*

- **Block** at `PFN_remixapi_CreateTexture` / `PFN_remixapi_DestroyTexture` typedefs (file scope) — ~6 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Declares function-pointer types for the texture upload/destroy lifecycle.*

- **Block** at `PFN_remixapi_dxvk_GetTextureHash` typedef (file scope) — ~4 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Declares the dxvk-specific extension function to retrieve the GPU image hash from a D3D9 texture object.*

- **Block** at `PFN_remixapi_CreateMeshBatched` typedef (file scope) — ~7 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Declares the batched mesh-creation function-pointer type and stub comment noting slot is currently nullptr.*

- **Block** at `remixapi_UIState` enum + `remixapi_GetUIState` / `remixapi_SetUIState` declarations (file scope) — ~16 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Declares the UI state enum and the two API entry points for reading/setting the ImGui overlay visibility level.*

- **Block** at `PFN_remixapi_DrawScreenOverlay` typedef (file scope) — ~9 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Declares the function-pointer type for compositing a plugin-supplied pixel buffer over the final frame.*

- **Block** at `PFN_remixapi_BridgeCallback` + `PFN_remixapi_RegisterCallbacks` + `PFN_remixapi_AutoInstancePersistentLights` + `PFN_remixapi_UpdateLightDefinition` (file scope) — ~22 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Declares frame-boundary callback registration, the persistent-light auto-instance helper, and the deferred light-definition update function.*

- **Block** at `PFN_remixapi_CreateLightBatched` typedef (file scope) — ~4 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Declares the batched light-creation function-pointer type.*

- **Block** at `PFN_remixapi_dxvk_GetSharedD3D11TextureHandle` typedef (file scope) — ~5 LOC, planned target `N/A (public header)` in `N/A (public header)`.
  *Declares the dxvk-specific extension function for the DX11 shared-texture export path (stub in this fork; slot populated for ABI layout compatibility).*

- **Block** at `PFN_remixapi_SetGameValue` typedef (file scope) — ~14 LOC (including the contract doc block), planned target `N/A (public header)` in `N/A (public header)`.
  *Declares the function-pointer type for the plugin-injected game-state write API introduced in workstream 10. The entrypoint stores a single string/string pair under a caller-chosen key in a fork-owned thread-safe map; graph components `GameValueReadBool` / `GameValueReadNumber` read those values by name. The contract doc block above the typedef describes key/value semantics, validation, and lifetime (store survives `Shutdown` / re-init).*

- **Block** at `remixapi_Interface` vtable additions (struct fields) — ~15 LOC spread across the vtable struct, planned target `N/A (public header)` in `N/A (public header)`.
  *Appends new function-pointer slots to `remixapi_Interface`: `AddTextureHash`, `RemoveTextureHash`, `CreateTexture`, `DestroyTexture`, `dxvk_GetTextureHash`, `CreateMeshBatched`, `GetUIState`/`SetUIState`, `DrawScreenOverlay`, `RegisterCallbacks`, `AutoInstancePersistentLights`, `UpdateLightDefinition`, `CreateLightBatched`, `dxvk_GetSharedD3D11TextureHandle`, `SetGameValue`. 2026-06-27: the upstream `SetCameraMediumMaterial` slot was moved out of the middle of the struct (it had been inherited between `SetupCamera` and `DrawInstance`) to immediately after `Present`, mirroring upstream's canonical tail layout (upstream `2bac8874`); fork slots remain appended after it. Size-neutral move — the `sizeof` sentinel is unchanged. An append-at-end warning comment was restored above the struct.*

---

## src/d3d9/d3d9_device.cpp

**Pre-refactor fork footprint:** +5 / -5 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `D3D9DeviceEx::PresentEx` (~line 3748) — 4-line addition for [RTX-Diag] entry log on PresentEx.
  *Logs hwnd override and swapchain pointer at the top of `PresentEx` to correlate with the Remix API present chain during diagnostics.*

---

## src/d3d9/d3d9_rtx.cpp

**Pre-refactor fork footprint:** +11 / -11 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `D3D9Rtx::EndFrame` (~line 1216) — 5-line addition for [RTX-Diag] entry log on EndFrame.
  *Logs targetImage pointer and callInjectRtx flag at the top of EndFrame, plus a second log after the CS lambda is emitted, to trace the frame-end dispatch chain.*

---

## src/d3d9/d3d9_swapchain.cpp

**Pre-refactor fork footprint:** +14 / -4 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `D3D9SwapChainEx::Present` (~line 441) — 4-line addition for [RTX-Diag] entry log.
  *Logs `hDestWindowOverride` pointer at the top of `D3D9SwapChainEx::Present` to trace the native present path during diagnostics.*

- **Inline tweak** at `D3D9SwapChainEx::Present` (after remix API call site) — 6-line addition for `remixapi_AutoInstancePersistentLights` flush.
  *Calls `remixapi_AutoInstancePersistentLights()` once per frame on the native D3D9 present path so persistent lights submitted via the Remix C API are auto-instanced even when the caller bypasses `remixapi_Present`. The callee early-outs cheaply for native-only consumers (see its body entry below), so this call is effectively free unless the C-API light path is in use.*

---

## src/d3d9/d3d9_swapchain_external.cpp

**Pre-refactor fork footprint:** +5 / -5 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `D3D9SwapchainExternal::Present` (~line 44) — 4-line addition for [RTX-Diag] entry log.
  *Logs VkImage pointer and framebuffer extent at the top of `D3D9SwapchainExternal::Present` to trace the external swapchain present path during diagnostics.*

---

## src/dxvk/dxvk_limits.h

**Category:** index-only

- **Inline tweak** at the `MaxPushConstantSize` enum value (~line 21) — 1-LOC value change + multi-line rationale comment.
  *Bumps `MaxPushConstantSize` from 128 to 256. The fork-side tonemap apply args struct (`ToneMappingApplyToneMappingArgs` in `src/dxvk/shaders/rtx/pass/tonemap/tonemapping.h`) grew to 144 bytes across Workstream 2 commits 3–4 (Hable + AgX operator params). With the old cap, `DxvkContext::pushConstants()` silently overflowed the per-bank storage by 16 bytes on every Global / Direct tonemap dispatch, corrupting adjacent bank state and crashing the NVIDIA driver later. Every RTX-class GPU reports `maxPushConstantsSize >= 256`; pipeline layouts derive per-shader push-constant ranges from shader reflection, so the larger cap is harmless to existing shaders.*

---

## src/dxvk/imgui/dxvk_imgui.cpp

**Pre-refactor fork footprint:** +236 / -71 LOC (audit 2026-04-18)
**Post-refactor footprint:** 3 hook call sites + `#include "rtx_render/rtx_fork_hooks.h"` (migrated 2026-04-18)

**Category:** migrate

- **Block** at `ImGUI::wndProcHandler` (top of function + dispatch points) — ~30 LOC, planned target `fork_hooks::wndProcHandlerDiag` in `rtx_fork_overlay.cpp`.
  *REVERTED before migration (commit 664a9ba4). Not present in current file. No hook created.*

- **Block** at `ImGUI::processHotkeys` (top) — ~10 LOC, planned target `fork_hooks::processHotkeysDiag` in `rtx_fork_overlay.cpp`.
  *REVERTED before migration (commit 664a9ba4). Not present in current file. No hook created.*

- **Block** at `ImGUI::checkHotkeyState` (alt-chord logging branch) — ~22 LOC, planned target `fork_hooks::checkHotkeyStateDiag` in `rtx_fork_overlay.cpp`.
  *REVERTED before migration (commit 664a9ba4). Not present in current file. No hook created.*

- **Block** at `ImGUI::wndProcHandler` (context pin at entry) — ~2 LOC. **Migrated** to `fork_hooks::imguiContextPin` in `rtx_fork_overlay.cpp`.
  *Pins ImGui and ImPlot contexts at the top of `wndProcHandler` to prevent context corruption when plugin activity drifts GImGui off the dev menu's context between frames. Call site passes `m_context` and `m_plotContext` directly — no friend declaration needed.*

- **Block** at `ImGUI::showRenderingSettings` (sky mode UI section) — ~154 LOC. **Migrated** to `fork_hooks::showAtmosphereUI` in `rtx_fork_atmosphere.cpp`.
  *Adds the Sky Mode combo (Skybox Rasterization / Numos), atmosphere preset buttons (Earth, Mars, Clear Sky, Polluted/Hazy, Alien World, Desert Planet), and the full atmosphere parameter tree (sun, density sliders, advanced coefficients) under the Sky Tuning collapsing header. The `skyModeCombo` static was moved from `dxvk_imgui.cpp` into the fork-owned atmosphere file. No friend declaration needed.*

- **Block** at `ImGUI::showMainMenu` (wrapper tab handling) — ~6 LOC. **Partially migrated** to `fork_hooks::wrapperTabDraw` in `rtx_fork_overlay.cpp`.
  *The `kTab_Wrapper` guard (`remixapi_imgui_HasDrawCallback()` check + `continue`) remains as an inline tweak in the tab loop (structural control flow, not extractable). The case body (`remixapi_imgui_InvokeDrawCallback()`) is wrapped as `fork_hooks::wrapperTabDraw()`. No friend declaration needed.*

- **Hook** at tonemapper ImGui settings → `fork_hooks::showTonemapOperatorUI` / `fork_hooks::showLocalTonemapOperatorUI` in `rtx_fork_tonemap.cpp`. Also remove the standalone `RemixGui::Checkbox("Use Legacy ACES", ...)` at ~line 3888 — its RtxOption `rtx.useLegacyACES` is being deleted.
  *Operator combo + per-operator sliders replace the old ACES checkbox. "Use Legacy ACES" reachable via TonemapOperator::ACESLegacy enum value.*

- **Inline tweak** at `ImGUI::showRenderingSettings` "Tonemapping" header — removed the `Tonemapping Mode` combo (Global / Local / Direct) and the standalone "User Brightness" / "User Brightness EV Range" sliders. The header body is now a single always-visible `metaToneMapping().showImguiSettings()` call between two separators. Tuning Mode (tone curve sliders) is also removed from the panel.
  *2026-05-13 tonemap refactor: mode selector removed; operator dropdown is now the primary control. 2026-05-15: local tonemap path removed entirely, so no per-path UI gate remains.*

---

## src/dxvk/imgui/dxvk_imgui_about.cpp

**Category:** index-only

**Rationale:** Fork additions are string-literal entries in the in-game
About panel's "GitHub Contributors" list — they live inside a curly-braced
initializer list and can't be lifted into a separate TU.

- **Inline tweak** at `ImGuiAbout::Credits::Credits` constructor (GitHub Contributors string list, ~lines 91-107) — one string literal per fork contributor, sorted alphabetically per the inline comment. Per CONTRIBUTING.md, contributors add their own entry when their PR adds something visible. Tracked here per the fridge-list invariant.
  *Each entry is in the format `"FirstName 'Handle' LastName"` or handle-only (`"BrunchyChineapple"`, `"Dayton 'watbulb'"`). The list is the canonical record of community contributors visible in the About panel.*

---

## src/dxvk/imgui/dxvk_imgui.h

**Pre-refactor fork footprint:** +2 / -1 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `tabNames` array / `kTab_Wrapper` enum (in `ImGUI` class) (~line 112) — 2-line addition for wrapper tab constant and name.
  *Adds `kTab_Wrapper` enumerator and "Plugin" entry to `tabNames[]` to expose the plugin-drawn ImGui tab in the Remix dev menu.*

---

## src/dxvk/imgui/imgui_impl_win32.cpp

**Pre-refactor fork footprint:** +1 / -1 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `ImGui_ImplWin32_UpdateMouseCursor` (~line 236) — 1-line modification to `GetAsyncKeyState` call.
  *Changes key-state sampling to use `GetAsyncKeyState` (async hardware state) instead of the previous synchronous variant to fix key-state detection on the overlay's WndProc path.*

---

## src/dxvk/dxvk_limits.h

**Category:** inline-tweak

- **Inline tweak** at `MaxPushConstantSize` constant — increased from `128` to `256`.
  *`ToneMappingApplyToneMappingArgs` grew past the original 128-byte limit once per-operator parameter blocks were added (current size 176 B — see `tonemapping.h` `static_assert`). Changing to 256 keeps the constant larger than any current push-constant struct and stays within the 256-byte push-constant size supported by all target GPUs.*

---

## src/dxvk/meson.build

**Pre-refactor fork footprint:** +4 / -0 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `dxvk_src` files list (~line 187) — 2-line addition registering atmosphere sources.
  *Registers `rtx_render/rtx_atmosphere.cpp` and `rtx_render/rtx_atmosphere.h` in the DXVK build.*

- **Inline tweak** at `dxvk_src` files list (~line 400) — 2-line addition registering ImGui export sources.
  *Registers `imgui/imgui_remix_exports.cpp` and `imgui/imgui_remix_exports.h` in the DXVK build.*

- **Inline tweak** — register `src/dxvk/rtx_render/rtx_fork_tonemap.cpp` in the rtx_render source list. The fork-owned tonemap operator headers (`aces.slangh`, `adaptation_v1.slangh`, `agx.slangh`, `fork_tonemap_operators.slangh`, `gt7.slangh`, `hable.slangh`, `lottes.slangh`, `neutwo.slangh`, `psycho17.slangh`) live under `src/dxvk/shaders/rtx/pass/tonemap/` and are picked up via the shader-include glob; no explicit meson.build entry is required for those.
  *Fork-owned tonemap module.*

- **Inline tweak** at `dxvk_src` files list (rtx_render block) — 2-line addition registering weather sources.
  *Registers `'rtx_render/rtx_fork_weather.cpp'` and `'rtx_render/rtx_fork_weather.h'` in the DXVK build source list.*

---

## src/dxvk/rtx_render/graph/rtx_component_list.h

**Category:** index-only

- **Inline tweak** at `components/` include list (~line 56) — 2-line addition. Not migrated: the include manifest is the intended extension point for new components, and adding two alphabetically-placed `#include` lines is the canonical way to register fork-owned graph components.
  *Registers `components/game_value_read_bool.h` and `components/game_value_read_number.h` in the component manifest. Both are fork-owned Sense components introduced in workstream 10 (plugin-injected game-state readers); their backing store is the fork-owned `rtx_fork_game_state.h`.*

---

## src/dxvk/rtx_render/rtx_camera_manager.cpp

**Pre-refactor fork footprint:** +10 / -10 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `CameraManager::processExternalCamera` (~line 179) — 6-line addition for [RTX-Diag] entry log.
  *Logs the old and new `frameLastTouched` values when an external camera is processed, to diagnose camera-validity skip conditions.*

---

## src/dxvk/rtx_render/rtx_context.cpp

**Pre-refactor fork footprint:** +211 / -26 LOC (audit 2026-04-18)
**Post-refactor footprint:** 5 hook call sites + `#include "rtx_fork_hooks.h"` (migrated 2026-04-18)

**Note on Block 4:** The audit listed this as "physical atmosphere sky skip + diag logs" in `injectRTX`. The [RTX-Diag] portion was reverted before migration (commits ff61f77d and b0d2da33); what remains is only the 5-line early-return guard in `rasterizeSky`. Migrated as `fork_hooks::injectRtxAtmosphereSkySkip`. The audit's method name (`injectRTX`) was incorrect — the actual call site is `rasterizeSky`.

**Note on Block 6:** The `endFrame` [RTX-Diag] log block was reverted (commit b0d2da33) before this migration ran. `endFrameDiag` was not implemented; no hook call site exists in `endFrame`.

- **Hook** at `RtxContext::RtxContext` constructor (atmosphere init) → `fork_hooks::initAtmosphere` in `rtx_fork_atmosphere.cpp`
  *Constructs the `RtxAtmosphere` instance during `RtxContext` initialization.*

- **Hook** at `RtxContext::updateRaytraceArgsConstantBuffer` (sky mode + atmosphere section) → `fork_hooks::updateAtmosphereConstants` in `rtx_fork_atmosphere.cpp`
  *Sets `constants.skyMode`, detects sky mode transitions (clearing skybox buffers on switch to Numos), and calls `m_atmosphere->initialize` / `computeLuts` / `getAtmosphereArgs` to populate the atmosphere constant block.*

- **Hook** at `RtxContext::bindCommonRayTracingResources` (atmosphere LUT bindings) → `fork_hooks::bindAtmosphereLuts` in `rtx_fork_atmosphere.cpp`
  *Ensures the atmosphere object is initialized and binds the three atmosphere LUT textures (`BINDING_ATMOSPHERE_TRANSMITTANCE_LUT`, `BINDING_ATMOSPHERE_MULTISCATTERING_LUT`, `BINDING_ATMOSPHERE_SKY_VIEW_LUT`) for all passes that declare them in common_bindings.*

- **Hook** at `RtxContext::rasterizeSky` (physical atmosphere sky skip) → `fork_hooks::injectRtxAtmosphereSkySkip` in `rtx_fork_atmosphere.cpp`
  *Returns early from rasterized sky rendering when Numos mode is active. No private-member access; no friend declaration required.*

- **Hook** at `RtxContext::dispatchScreenOverlay` (method body + ScreenOverlayShader class) → `fork_hooks::dispatchScreenOverlay` in `rtx_fork_overlay.cpp`
  *`ScreenOverlayShader` lifted to `rtx_fork_overlay.cpp`; `dispatchScreenOverlay` is now a one-line delegate. The hook alpha-composites a plugin-uploaded RGBA buffer over the final tone-mapped image using the compute shader.*

- **Inline tweak** at `RtxContext::dispatchTonemapping` — removed the `TonemappingMode::Global || TonemappingMode::Direct` dispatch gate; the tonemapper now always runs (always operator-only). The `DxvkLocalToneMapping` dispatch block was removed entirely (2026-05-15).
  *2026-05-13 tonemap refactor: global tone curve removed. 2026-05-15: local tonemapper removed.*

- **Inline tweak** at `(file scope)` (weather header include) — 1-line addition near the existing `rtx_fork_*.h` includes.
  *Adds `#include "rtx_fork_weather.h"` so `WeatherBlender` and the `fork_weather` namespace are available in this translation unit.*

- **Inline tweak** at `RtxContext::RtxContext` constructor (weather blender init) — ~1 LOC.
  *Adds `m_weatherBlender = std::make_unique<fork_weather::WeatherBlender>();` so the blender is constructed alongside the atmosphere object.*

- **Hook** at `RtxContext` per-frame entry (weather blender update) — `fork_hooks::updateWeatherBlender` in `rtx_fork_weather.cpp`.
  *Calls `fork_hooks::updateWeatherBlender(*this, GlobalTime::get().deltaTime())` once per frame so the blender can read trigger keys, advance the lerp timeline, and write blended values to the Derived RTX_OPTION layer.*

---

## src/dxvk/rtx_render/rtx_context.h

**Pre-refactor fork footprint:** +26 / -0 LOC (audit 2026-04-18)
**Post-refactor fork footprint:** +26 / -0 LOC inline tweaks + 4 friend declarations added (migrated 2026-04-18)

- **Inline tweak** at `RtxContext` class (member declarations — atmosphere) — ~4 LOC.
  *Adds `m_lastSkyMode` and `m_atmosphere` member fields to `RtxContext`. These remain as class members; the fork_hooks functions access them via friend declarations.*

- **Inline tweak** at `RtxContext::setScreenOverlayData` and `dispatchScreenOverlay` declarations — ~5 LOC.
  *Declares the two overlay-path methods. `setScreenOverlayData` remains a standalone public method; `dispatchScreenOverlay` is now a one-line delegate to `fork_hooks::dispatchScreenOverlay`.*

- **Inline tweak** at `RtxContext` class (member declarations — screen overlay state) — ~11 LOC.
  *Adds `ScreenOverlayFrame` struct, `m_pendingScreenOverlay`, `m_screenOverlayImage`, `m_screenOverlayView`, `m_screenOverlayWidth`, `m_screenOverlayHeight`, and `m_screenOverlayFormat`. These remain as class members accessed via friend declarations.*

- **Inline tweak** at `RtxContext` class body (just before closing `};`) — 4-line block of `friend` declarations plus a forward-declaration block above the class.
  *Grants `fork_hooks::initAtmosphere`, `fork_hooks::updateAtmosphereConstants`, `fork_hooks::bindAtmosphereLuts`, and `fork_hooks::dispatchScreenOverlay` access to private members.*

- **Inline tweak** at `(file scope)` (weather forward declarations) — ~2 LOC above the class definition.
  *Adds `namespace fork_weather { class WeatherBlender; }` forward declaration and a `void updateWeatherBlender(class RtxContext& ctx, float deltaTimeSeconds)` forward declaration inside the `fork_hooks` namespace block, so the private member and friend declaration below can reference the type.*

- **Inline tweak** at `RtxContext` class body (private member declarations — weather) — ~1 LOC.
  *Adds `std::unique_ptr<fork_weather::WeatherBlender> m_weatherBlender;` as a private member of `RtxContext`.*

- **Inline tweak** at `RtxContext` class body (friend declarations block) — ~1 LOC addition to the existing friend block.
  *Adds `friend void fork_hooks::updateWeatherBlender(RtxContext&, float);` so the hook can access the private `m_weatherBlender` member.*

---

## src/dxvk/rtx_render/rtx_game_capturer.cpp

**Pre-refactor fork footprint:** +94 / -28 LOC (audit 2026-04-18)
**Post-refactor footprint:** 2 hook call sites + inline tweaks + `#include "rtx_fork_hooks.h"` (migrated 2026-04-18)

**Note on Block 1 (materialLookupHash selection):** The material-lookup-hash block is pervasive inline tweaks throughout `GameCapturer::newInstance` (computing `materialLookupHash`, keying `bIsNewMat`, the `captureMaterial` call, `meshes[meshHash]->matHash`, and `instance.matHash`). Lifting this into a hook would require threading too many in/out parameters. Kept as inline tweaks; tracked below.

- **Inline tweak** at `GameCapturer::newInstance` (materialLookupHash computation and usage) — ~7 LOC distributed through the function.
  *Computes `materialLookupHash = material.getHash()` and substitutes it for the raw BLAS `matHash` in the `bIsNewMat` guard, the `captureMaterial` call, `meshes[meshHash]->matHash`, and `instance.matHash`, so USD capture references align with runtime replacement lookup for API-submitted materials.*

- **Hook** at `GameCapturer::captureMaterial` (method body) → `fork_hooks::captureMaterialApiPath` in `rtx_fork_capture.cpp`
  *`GameCapturer::captureMaterial` is now a one-line delegate; all logic lives in the hook. Exports the albedo texture for both D3D9 materials (color texture valid — direct export) and API-submitted materials (fallback: resolves texture hash via the texture-manager table and exports by hash). Access to private `m_exporter` and `m_pCap` is granted via a `friend` declaration — see the `rtx_game_capturer.h` entry below.*

- **Hook** at `GameCapturer::prepExport` (coord-system transform block) → `fork_hooks::captureCoordSystemSkip` in `rtx_fork_capture.cpp`
  *Skips the view/proj handedness inversion for the global USD export transform when the game is configured as a left-handed coordinate system, since API-submitted geometry is already in consistent Y-up space.*

---

## src/dxvk/rtx_render/rtx_game_capturer.h

**Post-refactor fork footprint:** forward decl + `friend` declaration (added 2026-04-18)

**Category:** index-only

- **Inline tweak** at file scope (just before `class GameCapturer`) — 13-line forward declaration of `fork_hooks::captureMaterialApiPath` so the friend declaration inside `GameCapturer` can name the fork-owned hook.
  *Companion to the `rtx_fork_capture.cpp` hook that needs private-member access to `m_exporter` and `m_pCap`.*

- **Inline tweak** at `GameCapturer` class body (top of class, before `public:`) — 4-line `friend` declaration granting `fork_hooks::captureMaterialApiPath` access to private members.
  *Canonical pattern for hooks that must read/write private upstream state — one inline tweak per such hook, tracked here.*

---

## src/dxvk/rtx_render/rtx_light_manager.cpp

**Pre-refactor fork footprint:** +126 / -12 LOC (audit 2026-04-18)
**Post-refactor footprint:** 7 hook call sites + `#include "rtx_fork_hooks.h"` (migrated 2026-04-18)

**Category:** migrate

- **Hook** at `LightManager::prepareSceneData` (pending-mutation flush block) → `fork_hooks::flushPendingLightMutations` in `rtx_fork_light.cpp`
  *At frame start, applies queued external-light erases (clearing replacement links), applies queued updates (erase-then-emplace to handle union-type changes), registers pending active-light activations, and auto-instances all persistent lights. Access to private members granted via `friend` declaration — see `rtx_light_manager.h`.*

- **Hook** at `LightManager::updateExternallyTrackedLight` (indexed static-sleep path) → `fork_hooks::updateLightStaticSleep` in `rtx_fork_light.cpp`
  *Shared static-sleep logic: tracks `isStaticCount`, skips copy when motionless for N frames, always updates dynamic lights. Restores `externallyTrackedLightId` when externalId is valid.*

- **Hook** at `LightManager::addExternalLight` (hash-map static-sleep path) → `fork_hooks::updateLightStaticSleep` in `rtx_fork_light.cpp`
  *Second call site for the same hook (Block 2's two-copies reduction). Passes `kInvalidExternallyTrackedLightId` so the id-restore branch is skipped.*

- **Hook** at `LightManager::addExternalLight` (new-light emplace branch) → `fork_hooks::setExternalLightEmplace` in `rtx_fork_light.cpp`
  *Emplaces the new external light and stamps `frameLastTouched`. Access to `m_externalLights` via `friend` declaration.*

- **Hook** at `LightManager::removeExternalLight` (queue erase) → `fork_hooks::disableExternalLightQueue` in `rtx_fork_light.cpp`
  *Queues the handle for deferred erase instead of immediate removal. Access to `m_pendingExternalLightErases` via `friend` declaration.*

- **Hook** at `LightManager::registerPersistentExternalLight` → `fork_hooks::registerPersistentLight` in `rtx_fork_light.cpp`
  *Inserts the handle into `m_persistentExternalLights`. Access via `friend` declaration.*

- **Hook** at `LightManager::unregisterPersistentExternalLight` → `fork_hooks::unregisterPersistentLight` in `rtx_fork_light.cpp`
  *Removes the handle from `m_persistentExternalLights`. Access via `friend` declaration.*

- **Hook** at `LightManager::queueAutoInstancePersistent` → `fork_hooks::queueAutoInstancePersistent` in `rtx_fork_light.cpp`
  *Copies all persistent-light handles into `m_pendingExternalActiveLights`. Access via `friend` declaration.*

---

## src/dxvk/rtx_render/rtx_light_manager.h

**Pre-refactor fork footprint:** +10 / -0 LOC (audit 2026-04-18)
**Post-refactor footprint:** forward decls + `friend` declarations (updated 2026-04-18)

**Category:** index-only

- **Inline tweak** at `LightManager` class (public method declarations) (~line 101) — 4-line addition.
  *Declares `registerPersistentExternalLight`, `unregisterPersistentExternalLight`, and `queueAutoInstancePersistent` in the public API of `LightManager`.*

- **Inline tweak** at `LightManager` class (private member declarations) (~line 129) — 6-line addition.
  *Adds four deferred-mutation member containers: `m_pendingExternalLightErases`, `m_pendingExternalLightUpdates`, `m_pendingExternalActiveLights`, and `m_persistentExternalLights`.*

- **Inline tweak** at file scope (just before `struct LightManager`) — forward declarations of all seven `fork_hooks::` functions that need `LightManager` or `RtLight` access, plus `struct RtLight` forward decl.
  *Required so the `friend` declarations inside the class can name the fork-owned hooks.*

- **Inline tweak** at `LightManager` class body (top of class, before `public:`) — 7-line block of `friend` declarations granting the light hooks access to private members.
  *Canonical pattern for hooks that must read/write private upstream state — one `friend` line per hook, tracked here.*

---

## src/dxvk/rtx_render/rtx_lights.cpp

**Pre-refactor fork footprint:** +25 / -17 LOC (audit 2026-04-18)
**Post-refactor fork footprint:** +25 / -17 LOC inline tweaks (reclassified 2026-04-18)

**Category:** index-only

**Rationale:** All fork changes are signature modifications and single-line flag-packing additions woven into the middle of existing `writeGPUData` function bodies, plus a one-line copy in the copy constructor. There is no standalone block to lift; the `ignoreViewModel` parameter is intrinsic to each function's signature and the flag-set line (`if (ignoreViewModel) flags |= 1 << 1;`) is inseparably interleaved with the upstream flags-assembly code. A hook would require passing the entire flags word in and out, making it structurally equivalent to rewriting each function — not a meaningful extraction.

- **Inline tweak** at `RtSphereLight::writeGPUData` (ignoreViewModel parameter + bit 1 flag) — ~3 LOC.
  *Adds `bool ignoreViewModel = false` parameter; sets bit 1 of the flags word when set. Companion signature change tracked in `rtx_lights.h`.*

- **Inline tweak** at `RtRectLight::writeGPUData` (ignoreViewModel parameter + bit 1 flag) — ~3 LOC.
  *Same pattern for rect lights.*

- **Inline tweak** at `RtDiskLight::writeGPUData` (ignoreViewModel parameter + bit 1 flag) — ~3 LOC.
  *Same pattern for disk lights.*

- **Inline tweak** at `RtCylinderLight::writeGPUData` (ignoreViewModel parameter + bit 1 flag) — ~3 LOC.
  *Same pattern for cylinder lights; refactors the previously direct `writeGPUHelper` call to use a local `flags` variable so the bit can be conditionally set.*

- **Inline tweak** at `RtDistantLight::writeGPUData` (ignoreViewModel parameter + bit 1 flag) — ~4 LOC.
  *Same pattern for distant lights; same refactor of the direct helper call to a local flags variable.*

- **Inline tweak** at `RtLight::writeGPUData` (dispatch passes `this->ignoreViewModel`) — ~5 LOC (5 call-site updates).
  *Each per-type `writeGPUData` call now forwards `this->ignoreViewModel` as the third argument.*

- **Inline tweak** at `RtLight::copyFrom` (ignoreViewModel copy) — ~1 LOC.
  *Copies `ignoreViewModel` in `copyFrom`, called by the copy constructor.*

---

## src/dxvk/rtx_render/rtx_lights.h

**Pre-refactor fork footprint:** +6 / -5 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `RtSphereLight::writeGPUData` declaration (~line 133) — 1-line modification to add `ignoreViewModel` default parameter.
  *Adds `bool ignoreViewModel = false` parameter to the `writeGPUData` declaration.*

- **Inline tweak** at `RtRectLight::writeGPUData` declaration (~line 197) — 1-line modification (same pattern).
  *Same default parameter addition for `RtRectLight`.*

- **Inline tweak** at `RtDiskLight::writeGPUData` declaration (~line 275) — 1-line modification (same pattern).
  *Same default parameter addition for `RtDiskLight`.*

- **Inline tweak** at `RtCylinderLight::writeGPUData` declaration (~line 350) — 1-line modification (same pattern).
  *Same default parameter addition for `RtCylinderLight`.*

- **Inline tweak** at `RtDistantLight::writeGPUData` declaration (~line 412) — 1-line modification (same pattern).
  *Same default parameter addition for `RtDistantLight`.*

- **Inline tweak** at `RtLight` struct (~line 645) — 1-line addition of `ignoreViewModel` field.
  *Adds `bool ignoreViewModel = false` member to `RtLight` to carry the per-light view-model exclusion flag across the GPU data write.*

---

## src/dxvk/rtx_render/rtx_instance_manager.cpp

**Category:** index-only

- **Inline tweak** at `InstanceManager::onInstanceUpdated` surface-meta block (~line 988, beside `isAnimatedWater`) — REVERTED 2026-06-19, comment-only.
  *Previously set `currentInstance.surface.isDecalCategory` for the cloud-shadow zenith gate. The gate and flag were deleted when the cloud shadow moved onto the sun term in the NEE (no geometry test needed); only a "removed" comment remains.*

---

## src/dxvk/rtx_render/rtx_materials.h

**Category:** index-only

- **Inline tweak** at `RtSurface` flags block (~line 336) + `RtSurface::writeGPUData` `flags0` packing (~line 121) — REVERTED 2026-06-19, comment-only.
  *Previously added `bool isDecalCategory` packed into `flags0` bit 2 for the cloud-shadow zenith gate. Removed with the gate; `flags0` bit 2 is free again. Only "removed" comments remain.*

---

## src/dxvk/rtx_render/rtx_options.h

**Pre-refactor fork footprint:** +32 / -0 LOC (audit 2026-04-18)
**Post-refactor fork footprint:** +32 / -0 LOC inline tweaks (reclassified 2026-04-18)

**Category:** index-only

**Rationale:** All fork additions are an enum definition and `RTX_OPTION(...)` macro declarations inside the `RtxOptions` class body. `RTX_OPTION` expands to an inline static member declaration — it is structurally part of the class definition and cannot be lifted into a separate TU or wrapped in a hook. There is no function body to extract.

- **Inline tweak** at `(file scope namespace dxvk)` (SkyMode enum) — ~5 LOC.
  *Declares the `SkyMode` enum class (`SkyboxRasterization = 0`, `Numos = 1`). Required by `RtxOptions::skyMode` below and by atmosphere hook code in `rtx_fork_atmosphere.cpp` (via the `rtx_options.h` include chain).*

- **Inline tweak** at `RtxOptions` class body (skyMode RTX_OPTION) — ~2 LOC.
  *Declares `RTX_OPTION("rtx", SkyMode, skyMode, SkyMode::SkyboxRasterization, ...)` immediately after the existing sky-related options block. Consumed by `fork_hooks::updateAtmosphereConstants` in `rtx_fork_atmosphere.cpp`.*

- **Inline tweak** at `RtxOptions` class body (atmosphere RTX_OPTIONs block) — ~25 LOC for the original 17 options + ~5 LOC for night-sky + ~52 LOC for the `DECLARE_MOON_OPTIONS(N)` macro and 4 invocations + ~11 LOC for the cloud block + ~13 LOC for the cloud-enhancement block (including `cloudVerticalProfile`, `cloudCurvature`). Sun/star position fields (`sunElevation`, `sunRotation`, `starBrightness`, `starRotation`) and per-moon pose fields (`elevationN`, `rotationN`, `phaseN`) are game-drivable per-frame but persist when saved (runtime push is the last writer; cold start uses the saved value until any push lands). Cloud defaults tuned from artist iteration.
  *Declares the original 17 atmosphere tuning options under the `rtx.atmosphere` prefix (`sunDisc`, `sunSize`, `sunIntensity`, `sunElevation`, `sunRotation`, `altitude`, `airDensity`, `aerosolDensity`, `ozoneDensity`, `planetRadius`, `atmosphereThickness`, `mieAnisotropy`, `rayleighScattering`, `mieScattering`, `ozoneAbsorption`, `ozoneLayerAltitude`, `ozoneLayerWidth`, `sunIlluminance`), plus the night-sky block (`starBrightness`, `starDensity`, `starTwinkleSpeed`, `nightSkyBrightness`, `nightSkyColor`), plus a per-moon block declared via the `DECLARE_MOON_OPTIONS(N)` macro for `N` in `0..MAX_MOONS-1` (each block: `enabledN`, `angularRadiusN`, `brightnessN`, `colorN`, `surfaceStyleN`, `craterDensityN`, `surfaceContrastN`, `surfaceNoiseScaleN`, `darkSideBrightnessN`, `roughnessAmountN`, plus pose fields `elevationN`/`rotationN`/`phaseN`), plus a cloud block (`cloudEnabled`, `cloudDensity`, `cloudAltitude`, `cloudScale`, `cloudColor`, `cloudWindSpeed`, `cloudWindDirection`, `cloudShadowStrength`, `cloudAnisotropy`, plus NoSave-flagged `cloudCoverage` for game-driven weather), plus a cloud-enhancement block (`cloudViewSamples`, `cloudThickness`, `cloudDetailWeight`, `cloudShadowTint`, `cloudShadowTintStrength`, `cloudSunsetWarmth`, `cloudVariance`, `cloudVarianceScale`, `cloudVerticalProfile`, `cloudCurvature`) for volumetric ray-march tuning, color polish, vertical-shape character, and sky-dome curvature. All consumed by `RtxAtmosphere::getAtmosphereArgs()` and the atmosphere UI hook in `rtx_fork_atmosphere.cpp`.*

- **Inline tweak** at `RtxOptions` class body (cloud spatial-variation block) — +21 / -11 net LOC.
  *Adds `cloudTypeMean`, `cloudTypeSpread`, `cloudTypeNoiseScale`, `cloudCoverageMean`, `cloudCoverageSpread`, `cloudCoverageNoiseScale`, `cloudAnvilBias`, `cloudWindShearStrength` RTX_OPTIONs (Nubis-style spatial variation, spec 2026-05-06; `cloudWindShearStrength` added as a tunable knob on the existing wind-shear UV perturbation in `sampleCloudDensity`). Replaces retired `cloudCoverage`, `cloudVariance`, `cloudVarianceScale`, `cloudVerticalProfile`.*

- **Inline tweak** at `RtxOptions` class body (cloud aerial-perspective extinction) — +13 LOC.
  *Adds `cloudAerialExtinctionPerKm` RTX_OPTION (default 0.2) under the `rtx.atmosphere` prefix (2026-05-16). Distance-based atmospheric extinction applied to cloud samples: `aerialT = exp(-cloudAerialExtinctionPerKm * t)` multiplies both per-step radiance contribution AND per-step extinction so horizon-grazing rays don't accumulate through ~100 km of cloud volume into a solid white wall. Applied identically in `marchCloudSlab` (`cloud_render.comp.slang`) and the legacy analytical `evalClouds` path (`atmosphere_sky.slangh`) so the two render modes stay visually consistent. Setting the option to 0 reverts to legacy (no aerial perspective) behavior. Consumed by `RtxAtmosphere::getAtmosphereArgs()` — the new field repurposes the existing `pad_cloudWorley_0` slot in `AtmosphereArgs`, preserving 16-byte struct alignment — and the atmosphere UI hook in `rtx_fork_atmosphere.cpp`.*

- **Inline tweak** at `RtxOptions` class body (moon-lighting strength sliders + cloud-look shape) — +10 LOC (Phase 1) + +18 LOC (Phase 3 Task 1) + +28 LOC (Phase 3 Task 2).
  *Phase 1 (2026-05-07) added `moonNeeStrength` (world-side master, default 1.0) and `moonAtmosphericCouplingStrength` (sky-side, default 1.0) RTX_OPTIONs. Phase 3 Task 1 (2026-05-08) added per-path stylistic multipliers: `surfaceMoonBrightness` (default 50.0), `cloudMoonBrightness` (default 2.0), `haloMoonBrightness` (default 15.0) — empirically tuned by in-game testing on 2026-05-08 against the Fallout: New Vegas test scene at `m.brightness=1.0`. Setting all three to 1.0 reverts to architecturally-pure physical-baseline output. Phase 3 Task 2 (2026-05-08) exposed five cloud-look + halo shape constants previously hardcoded in `atmosphere_sky.slangh`: `moonCloudDiffuseGain` (0.10), `moonCloudPhaseGain` (0.30), `moonCloudAnisotropy` (0.85), `moonHaloMagnitude` (0.0015), `moonAmbientAirglow` (0.0015). Defaults preserved; exposure is for in-game tuning without shader rebuilds. All consumed across `evalAtmosphereRadiance`, `evalClouds`, `evalMoonDisk`'s halo, and `sampleAtmosphereMoonLight`.*

- **Inline tweak** at `RtxOptions` class body (Phase 2 default migration) — net 0 LOC, value/text changes only.
  *Phase 2 (2026-05-08) shifts the per-moon `brightness##N` default from 4.0 → 1.0 (physical neutral; was magic-number magnitude-cheat) and the per-moon `color##N` default from (0.85, 0.87, 0.92) → (0.12, 0.12, 0.12) (neutral lunar Bond albedo; the prior cool-blue tint was magnitude-cheating). Retires the `cloudMoonBrightness` RTX_OPTION (its job -- scaling the cloud path's magic-number magnitude -- was eliminated by the Phase 2 unified physical irradiance scaffold). See `2026-05-08-moon-physical-irradiance-design.md`.*

- **Inline tweak** at `RtxOptions` class body (cloud-look master multipliers) — +8 LOC.
  *Sky/moon ImGui simplification pass (2026-05-21) added `moonSilverLiningIntensity` and `moonHaloGlowStrength` RTX_OPTIONs (both default 1.0). Applied C++-side in `RtxAtmosphere::getAtmosphereArgs()` as multipliers on the existing cloud-look + halo + airglow knob values, so the five Phase-3-Task-2 fine knobs collapse to three ImGui sliders (Silver Lining Intensity / Sharpness / Halo Glow) while the underlying ratio constants remain `.conf`-tunable. Shaders unchanged — masters apply at args population. Default 1.0 = byte-identical to pre-2026-05-21 behavior. See `2026-05-20-sky-moon-imgui-persistence-and-reorg-design.md`.*

- **Inline tweak** at `(file scope)` (weather header include) -- 1-line addition near the existing `rtx_fork_*.h` includes.
  *Adds `#include "rtx_fork_weather.h"` so the `DECLARE_ALL_WEATHER_PRESETS()` macro is in scope before it is used inside the `RtxOptions` class body.*

- **Inline tweak** at `RtxOptions` class body (weather preset RTX_OPTION block) -- 1-line macro invocation + 14-line undef block.
  *Invokes `DECLARE_ALL_WEATHER_PRESETS()` inside the `RtxOptions` struct body to expand all 504 RTX_OPTION declarations (12 presets x 42 fields). The 14 `#undef` lines immediately following clean up the binder macros so they do not leak into downstream includes.*

- **Inline tweak** -- cloud RTX_OPTION audit cleanup (2026-05-19). Removes three dead-knob RTX_OPTIONs whose values were written into `AtmosphereArgs` but never read by any shader: `cloudScale` (pre-3D-texture era; replaced by `cloudNoiseTileKm`), `cloudDetailWeight` (legacy FBM detail-fade; Nubis Cubed sampler has no detail-vs-base split), and `cloudWindShearStrength` (legacy analytical-only wind shear; textured sampler intentionally drops it). Also flips `cloudShadowStrength` default `0.0` → `1.0` so the voxel-grid cloud-on-terrain shadow system (`cloudVoxelShadowsEnable`, `cloudShadowMarchStrength`, `cloudShadowFactorStrength`) is not silently muted at boot. Args struct slots preserved as `padDead*` placeholders to maintain 16-byte alignment until a repack pass; the 24 weather-preset entries (12 presets x 2 fields) and the matching `WeatherSnapshot` plumbing in `rtx_fork_weather.{h,cpp}` are dropped along with the RTX_OPTION declarations and the ImGui sliders.

- **Inline tweak** -- cloud system iteration #2 (2026-05-19, second pass). Coverage gate floor lowered from 0.15 to 0.0 in `sampleCloudDensityTextured` so max-Coverage truly fills the sky (was clipping the bottom 15% of the noise range). Aerial perspective split into two independent RTX_OPTIONs: `cloudAerialHazePerKm` (renamed from `cloudAerialExtinctionPerKm`; dims cloud radiance with distance — visual softness) and new `cloudAerialFadePerKm` (softens cloud extinction with distance — prevents horizon white wall). Two new layer-2 spread RTX_OPTIONs (`cloudLayer2CoverageSpread`, `cloudLayer2TypeSpread`) plus a shader-side seed-offset (`cloudLayer2NoiseSeed`, default 1000) added to `smoothNoise2D` calls for layer 2 so the two layers generate fully decorrelated horizontal weather patterns instead of stacking. Position-shift approach considered first but reverted because `computeCloudHeightFraction` uses `length()` against planet center, so any XZ shift > ~10 km pushes samples out of the slab via spherical-distance inflation. Four previously hidden knobs re-added to the menu (`cloudCoverageNoiseScale`, `cloudTypeNoiseScale`, `cloudCurvature`, `cloudPhaseG2`) since they were silently driving the look at their defaults. 16 RTX_OPTION defaults retuned from in-game FNV tuning. Args slot recycling: pad6/pad7/three padDead* slots are now all live fields (`cloudLayer2NoiseSeed`, `cloudAerialFadePerKm`, `cloudLayer2CoverageSpread`, `cloudLayer2TypeSpread`, `cloudMsScale`). Struct size unchanged.

- **Inline tweak** — remove `rtx.useLegacyACES` + `rtx.showLegacyACESOption` RtxOptions (superseded by `TonemapOperator::ACESNarkowicz` enum value).
  *Both options live at the `rtx` namespace (not `rtx.tonemap`); removed in the enum refactor.*

- **Inline tweak** — remove `TonemappingMode` enum (Global / Local / Direct) and `tonemappingMode` RTX_OPTION. The dynamic tone curve (histogram + curve passes) is removed; the apply pass dispatches the operator directly. Local tonemapping (`DxvkLocalToneMapping`, `useLocalToneMapping` RTX_OPTION, `rtx.localtonemap.*`) removed entirely on 2026-05-15. The vestigial `directOperatorMode` CB field was removed in the 2026-05-XX cleanup along with the dead histogram / tone-curve dispatch passes and the ACES enum rename (`ACES`/`ACESLegacy` → `ACESHill`/`ACESNarkowicz`).
  *2026-05-13 tonemap refactor: simplified from three-mode selector to global operator dropdown. 2026-05-15: local tonemap path removed entirely. 2026-05-XX: dead-code cleanup + snake_case shader rename.*

- **Inline tweak** — weather-preset cold-default alignment (2026-05-26). Ten cold defaults aligned to `WEATHER_PRESET_VALUES_overcast` in `rtx_fork_weather.h` (the macro the codebase comments call "current default look"): `cloudShadowStrength` 1.0 → 0.10, `cloudCoverageMean` 0.85 → 0.64, `cloudCoverageSpread` 1.0 → 0.16, `cloudTypeMean` 0.75 → 0.5, `cloudTypeSpread` 0.5 → 0.2, `cloudTypeNoiseScale` 0.001 → 0.0034, `cloudDensity` 1.65 → 1.8, `cloudThickness` 2.75 → 3.05, `aerosolDensity` 1.0 → 1.1, `sunIlluminance` (20,20,20) → (15,15,15). Fixes the regression introduced by the 2026-05-19 `cloudShadowStrength` 0→1 flip: users sitting on the dormant "(none / dormant)" weather preset saw ground geometry crushed dark by full-strength cloud-voxel shadows over 85% default coverage. The dormant blender path leaves cold RTX_OPTIONs untouched, so the cold values themselves had to move.

---

## src/dxvk/rtx_render/rtx_global_volumetrics.cpp

**Category:** index-only

- **Inline tweak** at `RtxGlobalVolumetrics::dispatch` (volumeArgs population block) — 1 LOC (2026-05-26). Populates `volumeArgs.fogSunVisibilityGain` from the matching RTX_OPTION. Adjacent to the existing `volumetricFogAnisotropy` populate; same trivial pattern. Companion to the new field in `volume_args.h` and the new option in `rtx_global_volumetrics.h`.

---

## src/dxvk/rtx_render/rtx_global_volumetrics.h

**Pre-refactor fork footprint:** N/A — value-only cold-default tweaks

**Category:** index-only

- **Inline tweak** — weather-preset cold-default alignment (2026-05-26). Three RTX_OPTION cold defaults aligned to the `WEATHER_PRESET_VALUES_overcast` block in `rtx_fork_weather.h`: `transmittanceColor` (0.999, 0.999, 0.999) → (0.995, 0.995, 0.995), `transmittanceMeasurementDistanceMeters` 200.0 → 500.0, `anisotropy` 0.0 → 0.05 (mapped from `volumetricAnisotropy` in the preset). Companion to the matching `rtx_options.h` block — same rationale: the dormant "(none / dormant)" weather preset path leaves cold RTX_OPTIONs untouched, so the cold defaults themselves had to move to match overcast.

- **Inline tweak** — new `fogSunVisibilityGain` RTX_OPTION (2026-05-26; default lowered 5.0→1.0 on 2026-06-15). `rtx.volumetrics.fogSunVisibilityGain` (default 1.0, range 0.0–50.0) replaces the historical hardcoded artistic gain (x5 with a misleading "10x" comment in the gmod-rtx port) that was previously baked into the per-cache-write expression in fork-owned `atmosphere_common.slangh`. Default is now physical (1.0 = no boost) rather than the gmod-era ~5x. Read by `volume_composite_helpers.slangh::integrateVolumetricNEE` (consumer-side fog application only — surface consumers still read the cache straight). Companions: `rtx_global_volumetrics.cpp` (CB populate), `volume_args.h` (CB field), submodule fork edit at `rtxdi-sdk/include/volumetrics/rtx/algorithm/volume_composite_helpers.slangh`.

---

## src/dxvk/rtx_render/rtx_overlay_window.cpp

**Pre-refactor fork footprint:** +57 / -35 LOC (audit 2026-04-18)
**Post-refactor footprint:** 1 hook call site + 1 `#include "rtx_fork_hooks.h"` (migrated 2026-04-18)

**Note on diag blocks:** The fridge list originally listed 3 `[RTX-Diag]` blocks (blocks 2-4). These were introduced and then immediately reverted (commit `664a9ba4` reverted `0d590fb4`) before this migration ran. They are not present in the current file; no action was taken. Only block 1 (keyboard-forward) was active and required migration.

- **Hook** at `GameOverlay::gameWndProcHandler` (after hwnd guard) → `fork_hooks::overlayInputForward` in `rtx_fork_overlay.cpp`
  *Forwards keyboard (WM_KEY\*, WM_CHAR, WM_SYSCHAR) AND mouse (WM_MOUSEMOVE, WM_{L,R,M,X}BUTTON\*, WM_MOUSE{,H}WHEEL) messages to `ImGui_ImplWin32_WndProcHandler` on the legacy WndProc path so ImGui keyboard + mouse state stays in sync when a game menu captures raw input or when the plugin HUD pulls focus via the Remix API. Mouse coords in lParam are translated from gameHwnd to overlayHwnd client-space when the two differ; wheel lParam is screen-space and forwards without translation. Access to private `m_hwnd` is granted via a `friend` declaration — see the `rtx_overlay_window.h` entry below. Previously named `overlayKeyboardForward` (keyboard-only); renamed + expanded 2026-04-19 when the plugin-API mouse-input bug was diagnosed.*

---

## src/dxvk/rtx_render/rtx_overlay_window.h

**Post-refactor fork footprint:** forward decl + `friend` declaration (added 2026-04-18)

**Category:** index-only

- **Inline tweak** at file scope (just before `class GameOverlay`) — 6-line forward declaration of `fork_hooks::overlayInputForward` so the friend declaration inside `GameOverlay` can name the fork-owned hook.
  *Companion to the `rtx_fork_overlay.cpp` hook that needs private-member access to `m_hwnd`. Renamed from `overlayKeyboardForward` on 2026-04-19 when the hook's scope expanded to cover mouse messages.*

- **Inline tweak** at `GameOverlay` class body (top of class, before `public:`) — 3-line `friend` declaration granting `fork_hooks::overlayInputForward` access to `m_hwnd`.
  *Canonical pattern for hooks that must read/write private upstream state — one inline tweak per such hook, tracked here.*

---

## src/dxvk/rtx_render/rtx_remix_api.cpp

**Pre-refactor fork footprint:** +1277 / -118 LOC (audit 2026-04-18)
**Post-refactor footprint (fully migrated — migrations #7a, #7b, #7c, #7d done):** 23 hook call sites + 1 `#include "rtx_fork_hooks.h"` + inline tweaks listed below. All extractable fork blocks have been migrated to `rtx_fork_api_entry.cpp`.

- **Inline tweak** at `(file scope)` (includes block) — ~8 LOC added. Not migrated: include lines don't get hooks — they either stay inline or the fork-owned file pulls them for its own code. Tracked here per the fridge-list invariant.
  *Adds includes for `dxvk_objects.h`, `dxvk_imgui.h`, `rtx_context.h`, `rtx_option_layer.h`, `util_hash_set_layer.h`, `xxhash.h`, `algorithm`, and `d3d9_texture.h` to support fork-added API functions, plus `rtx_fork_hooks.h` added in migration #7a and `rtx_fork_game_state.h` added in workstream 10 for the `remixapi_SetGameValue` entry point.*

- **Inline tweak** at `(file scope)` (`PendingScreenOverlay` struct + `s_pendingScreenOverlay`) — **Removed in migration #7b**. Both the struct and the optional are now defined exclusively in `rtx_fork_api_entry.cpp` (anonymous namespace). A comment marking the removal remains in the upstream file for auditability.
  *The struct held staging buffer, dimensions, format, and opacity; the optional was the hand-off point between the API thread (writer: `drawScreenOverlay`) and the render thread (reader: `presentScreenOverlayFlush`). Both now live in the fork-owned TU.*

- **Inline tweak** at `(anonymous namespace)` — `s_inFrame`, `s_beginCallback`, `s_endCallback`, `s_presentCallback` — **Removed in migration #7c**. All four vars now live in `rtx_fork_api_entry.cpp` (anonymous namespace). A comment block marking the removal remains in the upstream file for auditability.
  *Previously used inline at 6 call sites in rtx_remix_api.cpp (DrawInstance, DrawLightInstance, Shutdown, Present×3). All call sites are now one-liner hook delegates.*

- **Hook** at `convert::toRtMaterialFinalized::preloadTexture` lambda (inside the `MaterialDataType::Opaque` / `Translucent` / `Portal` texture preload path) → `fork_hooks::textureHashPathLookup` in `rtx_fork_api_entry.cpp` (migrated 2026-04-18, migration #7a).
  *Adds a "0x..." hex-path shortcut inside the upstream texture-path resolver so API-uploaded textures can be referenced by hash string in material JSON without creating a real file path. Hook returns true and writes the resolved `TextureRef` when the path parses as hex and matches a registered texture; caller returns immediately. Falls through to the normal AssetDataManager lookup otherwise.*

- **Hook** at `(anonymous namespace)` `remixapi_AddTextureHash` / `remixapi_RemoveTextureHash` → `fork_hooks::mutateTextureHashOption` in `rtx_fork_api_entry.cpp` (migrated 2026-04-18, migration #7a).
  *Looks up an `RtxOption<fast_unordered_set>` by full option name and adds or removes a hash via the user config layer. Call sites acquire `s_mutex` then delegate to the hook (which internally takes the RtxOption update mutex — lock order documented alongside `s_mutex`). The call-site signature replaced the local `TextureHashMutation` enum with a plain `bool add` parameter.*

- **Inline tweak** at `convert::toRtDrawState` (skinning hash computation) — 1 LOC, not worth a hook. Not migrated.
  *Calls `skinningData.computeHash()` on the prototype after building skinning data so the skinning hash participates in geometry deduplication.*

- **Inline tweak** at `convert::toRtDrawState` (blend-weight/index buffer stride fix) — 2 LOC across two call sites, not worth a hook. Not migrated.
  *Fixes `blendWeightBuffer` and `blendIndicesBuffer` strides to use `bonesPerVertex`-based byte widths rather than fixed-width placeholders.*

- **Inline tweak** at `remixapi_SetupCamera` (devLock RAII guard) — 1 LOC. Not extracted to a hook. The `LockDevice()` guard is scope-tied to the function body (its destructor must run at end-of-function), so a hook cannot own it without rewriting the entire function. Tracked here per the fridge-list invariant.
  *Adds `auto devLock = remixDevice->LockDevice()` so the EmitCs call that submits external camera data is race-safe.*

- **Inline tweak** at `remixapi_DrawInstance` (devLock RAII guard) — 1 LOC. Same reasoning as SetupCamera. Scope-tied RAII guard cannot be extracted without lifting the entire function. Tracked here per the fridge-list invariant.
  *Adds `auto devLock = remixDevice->LockDevice()` inside the EmitCs block that calls `commitExternalGeometryToRT`.*

- **Hook** at `remixapi_DrawInstance` (beginScene dispatch) → `fork_hooks::notifyBeginScene` in `rtx_fork_api_entry.cpp` (migrated 2026-04-18, migration #7c).
  *One-liner call. Atomically exchanges `s_inFrame` to true and fires `s_beginCallback` on the first frame submission.*

- **Hook** at `remixapi_DrawLightInstance` (beginScene dispatch) → `fork_hooks::notifyBeginScene` in `rtx_fork_api_entry.cpp` (migrated 2026-04-18, migration #7c).
  *Same hook as DrawInstance; lights-only frames also trigger the beginScene callback.*

- **Hook** at `remixapi_Shutdown` (callback + frame-state clear) → `fork_hooks::shutdownCallbacks` in `rtx_fork_api_entry.cpp` (migrated 2026-04-18, migration #7c).
  *One-liner call replacing the 4-line null/false reset. Clears `s_beginCallback`, `s_endCallback`, `s_presentCallback`, and `s_inFrame`.*

- **Hook** at `remixapi_Present` (screen overlay flush — inner namespace path) → `fork_hooks::presentScreenOverlayFlush` in `rtx_fork_api_entry.cpp` (migrated 2026-04-18, migration #7b).
  *One-liner call to the fork-owned flush hook. State (PendingScreenOverlay + s_pendingScreenOverlay) was unified in the same migration.*

- **Hook** at `remixapi_Present` (endScene callback, before native Present) → `fork_hooks::presentEndSceneDispatch` in `rtx_fork_api_entry.cpp` (migrated 2026-04-18, migration #7c).
  *Fires `s_endCallback` if `s_inFrame` is set, immediately before the native `remixDevice->Present()` call.*

- **Hook** at `remixapi_Present` (present callback + s_inFrame reset, after native Present) → `fork_hooks::presentCallbackDispatch` in `rtx_fork_api_entry.cpp` (migrated 2026-04-18, migration #7c).
  *Fires `s_presentCallback` and resets `s_inFrame` to false after a successful native Present.*

- **Hook** at `extern "C"` `remixapi_AutoInstancePersistentLights` (screen overlay flush path) → `fork_hooks::presentScreenOverlayFlush` in `rtx_fork_api_entry.cpp` (migrated 2026-04-18, migration #7b).
  *Same flush hook called on the C-export AutoInstancePersistentLights path, which also drains the pending overlay once per frame.*

- **Hook** at `remixapi_DrawScreenOverlay` (function body) → `fork_hooks::drawScreenOverlay` in `rtx_fork_api_entry.cpp` (migrated 2026-04-18, migration #7b).
  *Upstream wrapper acquires device + mutex, then delegates to the hook. The reverted [RTX-Diag] FIRST-CALL log block is absent (never present in current HEAD).*

- **Hook** at `remixapi_GetUIState` (function body) → `fork_hooks::getUiState` in `rtx_fork_api_entry.cpp` (migrated 2026-04-18, migration #7b).
  *One-liner delegate passing `tryAsDxvk()` to the hook.*

- **Hook** at `remixapi_SetUIState` (function body) → `fork_hooks::setUiState` in `rtx_fork_api_entry.cpp` (migrated 2026-04-18, migration #7b).
  *One-liner delegate.*

- **Hook** at `remixapi_dxvk_GetSharedD3D11TextureHandle` (function body) → `fork_hooks::getSharedD3D11TextureHandle` in `rtx_fork_api_entry.cpp` (migrated 2026-04-18, migration #7b).
  *Stub returning GENERAL_FAILURE. One-liner delegate.*

- **Hook** at `remixapi_dxvk_GetTextureHash` (function body) → `fork_hooks::dxvkGetTextureHash` in `rtx_fork_api_entry.cpp` (migrated 2026-04-18, migration #7b).
  *One-liner delegate.*

- **Hook** at `remixapi_CreateTexture` (function body) → `fork_hooks::createTexture` in `rtx_fork_api_entry.cpp` (migrated 2026-04-18, migration #7b).
  *Upstream wrapper acquires s_mutex, then delegates. Full GPU resource creation lives in the hook.*

- **Hook** at `remixapi_DestroyTexture` (function body) → `fork_hooks::destroyTexture` in `rtx_fork_api_entry.cpp` (migrated 2026-04-18, migration #7b).
  *Upstream wrapper acquires s_mutex, then delegates.*

- **Hook** at `remixapi_RegisterCallbacks` (function body) → `fork_hooks::registerCallbacks` in `rtx_fork_api_entry.cpp` (migrated 2026-04-18, migration #7c).
  *One-liner delegate. Body now lives in the fork-owned TU where the callback state vars live.*

- **Hook** at `remixapi_RequestVramCompaction` (function body) → `fork_hooks::requestVramCompaction` in `rtx_fork_api_entry.cpp` (migrated 2026-04-20, migration #7d).
  *One-liner delegate passing `tryAsDxvk()` to the hook. Hook does its own null check and sets SceneManager's atomic VRAM-compaction flag; render thread consumes it in manageTextureVram. Lock-free — `s_mutex` not taken.*

- **Hook** at `remixapi_RequestTextureVramFree` (function body) → `fork_hooks::requestTextureVramFree` in `rtx_fork_api_entry.cpp` (migrated 2026-04-20, migration #7d).
  *One-liner delegate. Hook sets SceneManager's atomic texture-VRAM-free flag; the render-thread tick calls `textureManager.clear()`, matching the DX9 scene-transition behavior exposed to plugins. Lock-free — `s_mutex` not taken.*

- **Hook** at `remixapi_GetVramStats` (function body) → `fork_hooks::getVramStats` in `rtx_fork_api_entry.cpp` (migrated 2026-04-20, migration #7d).
  *One-liner delegate. Hook fills `remixapi_VramStats` with per-category DXVK totals plus driver-view heap info (`driverAllocatedBytes` / `driverBudgetBytes`) and the fork-side `RtxTextureManager::getTextureTable().size()` (`forkTextureCacheCount`). Driver-view numbers match Task Manager / nvidia-smi; the gap vs `totalAllocatedBytes` exposes non-DXVK allocations (NGX, RT pipeline state, descriptor pools, NRC).*

- **Inline tweak** at `(anonymous namespace)` frame-boundary callback infrastructure — `s_pendingLightCreates`, `s_pendingLightUpdates`, `s_pendingDomeUpdates`, `s_pendingLightDestroys`, `s_pendingMeshCreates`, `s_handlesDeletedThisFrame`. Not migrated. The pending-queue state stays in upstream because it is accessed by too many anonymous-namespace functions (`flushPendingMeshes`, `remixapi_CreateMeshBatched`, `remixapi_CreateLight`, `remixapi_DestroyLight`, `remixapi_Present`, `remixapi_UpdateLightDefinition`) — moving it would require either lifting all those callers or exposing a wide accessor surface. Tracked here per the fridge-list invariant.

- **Inline tweak** at `remixapi_AutoInstancePersistentLights` / `remixapi_UpdateLightDefinition` bodies (extern-C fork-owned functions) — not extracted to hooks. These are `REMIXAPI`-exported entry points; their bodies are the fork's implementation of those API calls. The pending-queue state they access is documented as staying inline above. Tracked here per the fridge-list invariant.
  *2026-06-27: `remixapi_AutoInstancePersistentLights` now early-outs (skips the per-frame `LockDevice`+`EmitCs`) when no C-API scene work is queued and no external light has ever been registered, gated on the file-scope sticky `s_externalLightApiUsed` (set in `remixapi_CreateLight`, `remixapi_CreateLightBatched`, `remixapi_UpdateLightDefinition`). Fixes native-only consumers seeing all lights flicker from the empty per-frame dispatch on the native present path. The self-gating overlay flush still runs.*

- **Inline tweak** at bit-24 category routing (in `toRtCategories`) — ~1 LOC. Not migrated.
  *Routes `REMIXAPI_INSTANCE_CATEGORY_BIT_SMOOTH_NORMALS` (bit 24, upstream name) to `InstanceCategories::SmoothNormals`. (The misleadingly-named `LEGACY_EMISSIVE` alias of this bit was removed 2026-06-28 — it routed to `SmoothNormals` despite its name, a silent footgun.)*

- **Inline tweak** at `(anonymous namespace)` `remixapi_SetGameValue` — ~15 LOC. Not migrated (fork-owned store + direct inline body fits the surrounding `remixapi_SetConfigVariable` pattern; no anonymous-namespace state to share with other TUs).
  *Implements the plugin-injected game-state write API introduced in workstream 10. Validates args, constructs `std::string` copies of the incoming C strings, and forwards to `dxvk::fork_game_state::GameStateStore::get().set(key, value)`. Does not take `s_mutex` — the store owns its own lock, and funnelling high-frequency plugin writes through the API-wide mutex has no benefit.*

- **Block** at `extern "C"` vtable init block (fork-added anonymous-namespace slots) — ~11 LOC inline assignment block in `remixapi_InitializeLibrary`. Not fully hookable: the anonymous-namespace function pointers have internal linkage and cannot be named from another TU. Tracked here per the fridge-list invariant. The three extern-C-linked fork slots (RegisterCallbacks, AutoInstancePersistentLights, UpdateLightDefinition) are assigned via `fork_hooks::remixApiVtableInit` (migrated 2026-04-18, migration #7c).
  *Registers all fork-added API functions into the `remixapi_Interface` vtable. The inline block assigns the anonymous-namespace slots (including `SetGameValue` added in workstream 10); the hook fills the three externally-linked ones.*

- **Inline tweak** at `extern "C"` vtable size static_assert — 1 LOC. Not migrated (fridge-listed).
  *The `static_assert(sizeof(interf) == 288, ...)` sentinel is the final value in the chain (208 → 240 → 272 → 280 → 288 across five workstreams). Retained inline in `remixapi_InitializeLibrary` as a size sentinel.*

---

## src/dxvk/rtx_render/rtx_remix_specialization.inl

**Pre-refactor fork footprint:** +3 / -1 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `pnext::detail` specialization list (~line 95) — 2-line addition.
  *Adds `remixapi_CameraInfoParameterizedEXT` and `remixapi_TextureInfo` to the `pnext` type-list so `pnext::chain` can traverse these new struct types.*

- **Inline tweak** at `pnext::detail::ToEnum` specialization (~line 123) — 1-line addition.
  *Maps `remixapi_TextureInfo` to `REMIXAPI_STRUCT_TYPE_TEXTURE_INFO` in the sType enum specialization table.*

---

## src/dxvk/rtx_render/rtx_resources.cpp

**Pre-refactor fork footprint:** +18 / -0 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `Resources::getAtmosphereTransmittanceLut` (new method body) (~line 808) — 5-line addition.
  *Stub accessor returning `m_atmosphereTransmittanceLut`; LUT is populated by `RtxAtmosphere`.*

- **Inline tweak** at `Resources::getAtmosphereMultiscatteringLut` (new method body) (~line 813) — 5-line addition.
  *Stub accessor returning `m_atmosphereMultiscatteringLut`.*

- **Inline tweak** at `Resources::getAtmosphereSkyViewLut` (new method body) (~line 818) — 5-line addition.
  *Stub accessor returning `m_atmosphereSkyViewLut`.*

---

## src/dxvk/rtx_render/rtx_resources.h

**Pre-refactor fork footprint:** +7 / -0 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `Resources` class (public method declarations) (~line 393) — 3-line addition.
  *Declares `getAtmosphereTransmittanceLut`, `getAtmosphereMultiscatteringLut`, and `getAtmosphereSkyViewLut` on the `Resources` class.*

- **Inline tweak** at `Resources` class (private member fields) (~line 469) — 4-line addition.
  *Adds `m_atmosphereTransmittanceLut`, `m_atmosphereMultiscatteringLut`, and `m_atmosphereSkyViewLut` storage fields to `Resources`.*

---

## src/dxvk/rtx_render/rtx_scene_manager.cpp

**Pre-refactor footprint:** +73 / -2 LOC (migrated 2026-04-18)
**Post-refactor footprint:** 4 hook call sites + 1 `#include "rtx_fork_hooks.h"`

- **Hook** at `SceneManager::submitExternalDraw` (before submesh loop) → `fork_hooks::externalDrawMeshReplacement` in `rtx_fork_submit.cpp`
  *Checks for USD mesh/light replacements keyed on the API mesh handle hash; call site handles the early-exit + `drawReplacements` dispatch since those are private SceneManager methods.*

- **Hook** at `SceneManager::submitExternalDraw` (inside `if (material != nullptr)`, before `setHashOverride`) → `fork_hooks::externalDrawMaterialReplacement` in `rtx_fork_submit.cpp`
  *Checks for USD material replacements via `getReplacementMaterial()` and updates the `material` pointer in-place if one is found.*

- **Hook** at `SceneManager::submitExternalDraw` (inside `if (material != nullptr)`, after `setHashOverride`) → `fork_hooks::externalDrawTextureCategories` in `rtx_fork_submit.cpp`
  *Resolves albedo texture hash from the API material's opaque data and auto-applies all texture-based instance categories (Sky, Ignore, WorldUI, WorldMatte, Particle, Beam, DecalStatic, Terrain, AnimatedWater, IgnoreLights, IgnoreAntiCulling, IgnoreMotionBlur, Hidden).*

- **Hook** at `SceneManager::submitExternalDraw` (after particle setup, before `processDrawCallState`) → `fork_hooks::externalDrawObjectPicking` in `rtx_fork_submit.cpp`
  *Stores per-draw texture hash metadata in `m_drawCallMeta` when object picking is active. Access to the private `m_drawCallMeta` member is granted via a `friend` declaration — see the `rtx_scene_manager.h` entry below.*

---

## src/dxvk/rtx_render/rtx_scene_manager.h

**Post-refactor fork footprint:** forward decl + `friend` declaration (added 2026-04-18)

**Category:** index-only

- **Inline tweak** at file scope (just before `class SceneManager`) — 9-line forward declaration of `fork_hooks::externalDrawObjectPicking` so the friend declaration inside `SceneManager` can name the fork-owned hook.
  *Companion to the `rtx_fork_submit.cpp` hook that needs private-member access to `m_drawCallMeta`.*

- **Inline tweak** at `SceneManager` class body (top of class, before `public:`) — 5-line `friend` declaration granting `fork_hooks::externalDrawObjectPicking` access to `m_drawCallMeta`.
  *Canonical pattern for hooks that must read/write private upstream state — one inline tweak per such hook, tracked here.*

---

## src/dxvk/rtx_render/rtx_sky.h

**Pre-refactor fork footprint:** +6 / -0 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `tryHandleSky` (~line 145) — 6-line addition for physical atmosphere sky skip.
  *Returns `TryHandleSkyResult::SkipSubmit` early for any draw with `cameraType == CameraType::Sky` when Numos mode is active, preventing rasterized skybox geometry from being submitted.*

---

## src/dxvk/rtx_render/rtx_tone_mapping.cpp

- **Hook calls** at `DxvkToneMapping::dispatchApplyToneMapping` (args-population) and `DxvkToneMapping::showImguiSettings` (ImGui panel) → `fork_hooks::populateTonemapOperatorArgs` + `fork_hooks::showTonemapOperatorUI` in `rtx_fork_tonemap.cpp`.
  *Routes global tonemap through the fork operator dispatcher.*

---

## src/dxvk/rtx_render/rtx_tone_mapping.h

- **Inline tweak** — remove `rtx.tonemap.finalizeWithACES` RtxOption (superseded by `rtx.tonemap.tonemapOperator` in `rtx_fork_tonemap.cpp`); add `#include "rtx_fork_tonemap.h"`.
  *Adopts the fork operator enum.*

---

## src/dxvk/shaders/rtx/algorithm/geometry_resolver.slangh

**Pre-refactor fork footprint:** +171 / -1 LOC (audit 2026-04-18)

**Category:** migrate

- **Block** at `geometryResolverVertex` (miss handler — sky radiance branch) — ~30 LOC (active) + ~60 LOC (commented-out deprecated decals-on-sky block), planned target `fork_hooks::geoResolverAtmosphereMiss` in `rtx_fork_atmosphere.slangh`.
  *Adds a conditional atmosphere sky-radiance evaluation (`evalSkyRadiance`) in the geometry-resolver miss path when `cb.skyMode == 1`, selecting between dome light, physical atmosphere, and skybox rasterization. The commented-out block documents the deprecated `enableDecalsOnSky` feature. Cloud temporal smoothing (2026-05-09): the primary view ray's evalSkyRadiance call now passes `enableCloudTemporalSmoothing=true` plus the motion-vector + screen-extent args needed to reproject and EMA-blend the cloud layer against the previous frame's history at slots 206/207. PSR and indirect callers continue to pass false (their pixelCoord refers to a non-primary direction; reusing primary screen-space cloud history would smear).*

- **Block** at `geometryResolverVertex` (hit path — occluder comment block) — ~42 LOC (fully commented out), planned target `fork_hooks::geoResolverOccluder` in `rtx_fork_atmosphere.slangh`.
  *Preserves the design for the deprecated `isOccluder` surface property that would have shown sky behind occluder surfaces; kept commented for future reference.*

- **Block** at `geometryPSRResolverVertex` (PSR hit — atmosphere sky radiance) — ~9 LOC, planned target `fork_hooks::geoResolverPsrAtmosphere` in `rtx_fork_atmosphere.slangh`.
  *Adds atmosphere sky-radiance evaluation in the PSR resolver's emissive radiance accumulation path when physical atmosphere mode is active.*

- **Block** at `geometryPSRResolverVertex` (PSR hit — occluder comment block) — ~45 LOC (fully commented out), planned target `fork_hooks::geoResolverPsrOccluder` in `rtx_fork_atmosphere.slangh`.
  *Same occluder design-preservation comment block for the PSR path.*

- **Block** at `(file scope)` (atmosphere include) — ~4 LOC, planned target `fork_hooks::atmosphereInclude` in `rtx_fork_atmosphere.slangh`.
  *Adds `#include "rtx/pass/atmosphere/atmosphere_common.slangh"` at the top of the file, plus `#include "rtx/pass/atmosphere/atmosphere_sky.slangh"` gated by `#ifdef ATMOSPHERE_AVAILABLE` for the sky-radiance evaluation paths.*

---

## src/dxvk/shaders/rtx/algorithm/integrator_direct.slangh

**Pre-refactor fork footprint:** +120 / -2 LOC (audit 2026-04-18)

**Category:** migrate

- **Block** at `(file scope)` (atmosphere include) — ~1 LOC, planned target `fork_hooks::atmosphereInclude` in `rtx_fork_atmosphere.slangh`.
  *Adds `#include "rtx/pass/atmosphere/atmosphere_common.slangh"`. 2026-06-27: moved this include ahead of `rtxcr_material.slangh` so the SSS NEE code can call `sampleCloudGroundShadow_OptionB` for the SSS cloud-shadow fold (see the rtxcr_material.slangh touchpoint).*

- **Block** at `evalAtmosphereSunNEE` (full function) — ~40 LOC, planned target `fork_hooks::evalAtmosphereSunNEEDirect` in `rtx_fork_atmosphere.slangh`.
  *Implements primary-bounce sun NEE for physical atmosphere: samples sun direction + cone angle, traces multiple jittered shadow rays for soft shadows, averages visibility, evaluates BRDF split-weight, and accumulates diffuse/specular sun radiance. As of the 2026-06-19 sun-only cloud-shadow re-architecture this function no longer touches clouds at all — the cloud-on-terrain shadow folds onto the sun's radiance inside `sampleAtmosphereSunLight` (atmosphere_common.slangh), so the per-pixel `PrimaryCloudShadowFactor` write, the sealed-interior zenith up-ray gate, and the viewmodel/decal/normal-flip origin corrections were all deleted here. 2026-06-20: taught the shadow ray about thin-opaque subsurface — it now traces with `visibilityModeEnableSubsurfaceMaterials` and skips the `NdotL <= 0` early-out (both the up-front check and the per-sample jitter skip) for thin-opaque materials, so backlit translucency works under the physical-atmosphere sun (it is the only sun NEE in skyMode==1; the SSS-capable standard path only runs for RTXDI/RIS lights).*

- **Block** at `evalAtmosphereMoonNEE` (full function) — ~100 LOC, planned target `fork_hooks::evalAtmosphereMoonNEEDirect` in `rtx_fork_atmosphere.slangh`.
  *Primary-bounce moon NEE -- mirror of evalAtmosphereSunNEE for the moon. Calls `sampleAtmosphereMoonLight` with a u_pick blue-noise sample so one of the enabled, above-horizon moons is importance-picked per ray (weight = brightness × phaseGlow × elevation). Soft-shadow cone jitter via `getJitteredSunDirection` (direction-agnostic). Accumulated contribution divided by `moonSample.solidAnglePdf` (discrete pick PDF) so multi-moon importance sampling stays unbiased over many frames. Added by 2026-05-07 moon sun-parity workstream. 2026-06-20: same thin-opaque subsurface fix as the sun path (`visibilityModeEnableSubsurfaceMaterials` + thin-opaque-gated `NdotL <= 0` early-outs) so moonlit translucency matches.*

- **Block** at `integrateDirectPath` (atmosphere sun NEE call site) — ~14 LOC, planned target `fork_hooks::directPathAtmosphereSunCall` in `rtx_fork_atmosphere.slangh`.
  *Calls `evalAtmosphereSunNEE` in the direct-path integrator when `cb.skyMode == 1`. 2026-06-21 (experiment branch): the gate also requires `debugSkyBisectFlags` bit 2 clear — when the sun/moon are injected as real distant lights (`rtx.atmosphere.useDirectionalLights`), bit 2 is set and this bespoke sun+moon NEE block is skipped to avoid double-counting.*

- **Block** at `integrateDirectPath` (atmosphere moon NEE call site) — ~12 LOC, planned target `fork_hooks::directPathAtmosphereMoonCall` in `rtx_fork_atmosphere.slangh`.
  *Calls `evalAtmosphereMoonNEE` immediately after `evalAtmosphereSunNEE` in the direct-path integrator when `cb.skyMode == 1`. Sun and moon NEE are independent samples -- both can be valid at twilight, both invalid during pure daytime / pure-night-with-no-moons; each early-outs cheaply when invalid. Added by 2026-05-07 moon sun-parity workstream.*

- **Block** at `integrateDirectPath` (sky radiance miss branch) — ~8 LOC, planned target `fork_hooks::directPathAtmosphereMiss` in `rtx_fork_atmosphere.slangh`.
  *Adds `#ifdef ATMOSPHERE_AVAILABLE` branch in the miss sky-radiance evaluation to call `evalSkyRadiance` in physical atmosphere mode.*

- **Block** at `integrateDirectPath` / `sampleLightRIS` call sites (customIndex for view-model) — ~3 LOC, planned target `fork_hooks::directPathViewModelCustomIndex` in `rtx_fork_light.slangh`.
  *Synthesizes a `customIndex` carrying `CUSTOM_INDEX_IS_VIEW_MODEL` from `geometryFlags.isViewModel` and threads it through to `evalDirectLighting` call sites so view-model geometry skips `ignoreViewModel` lights.*

---

## src/dxvk/shaders/rtx/algorithm/integrator_indirect.slangh

**Pre-refactor fork footprint:** +138 / -4 LOC (audit 2026-04-18)

**Category:** migrate

- **Block** at `(file scope)` (atmosphere include) — ~4 LOC, planned target `fork_hooks::atmosphereInclude` in `rtx_fork_atmosphere.slangh`.
  *Adds `#include "rtx/pass/atmosphere/atmosphere_common.slangh"`, plus `#include "rtx/pass/atmosphere/atmosphere_sky.slangh"` gated by `#ifdef ATMOSPHERE_AVAILABLE` for the sky-radiance evaluation in the indirect-path miss handler.*

- **Block** at `evalAtmosphereSunNEESecondary` (full function) — ~100 LOC, planned target `fork_hooks::evalAtmosphereSunNEEIndirect` in `rtx_fork_atmosphere.slangh`.
  *Secondary-bounce variant of the atmosphere sun NEE function: uses half the sample count for performance, otherwise identical structure to the direct-path version.*

- **Block** at `evalAtmosphereMoonNEESecondary` (full function) — ~100 LOC, planned target `fork_hooks::evalAtmosphereMoonNEEIndirect` in `rtx_fork_atmosphere.slangh`.
  *Secondary-bounce variant of the moon NEE function -- mirror of evalAtmosphereSunNEESecondary. Same structure as the direct-path moon NEE but with the indirect shadow-mask flags, half the sample count, and additive `diffuseLight` / `specularLight` accumulation (no throughput multiplier; caller folds throughput in). Added by 2026-05-07 moon sun-parity workstream.*

- **Block** at `integratePathVertex` (atmosphere moon NEE call site) — ~16 LOC, planned target `fork_hooks::indirectPathAtmosphereMoonCall` in `rtx_fork_atmosphere.slangh`.
  *Calls `evalAtmosphereMoonNEESecondary` after the existing sun call when `cb.skyMode == 1 && isNeeEnabledOnBounce`. Accumulates the returned diffuseLight + specularLight via accumulateRadiance. Added by 2026-05-07 moon sun-parity workstream.*

- **Block** at `integratePathVertex` (atmosphere sky radiance in miss) — ~8 LOC, planned target `fork_hooks::indirectPathAtmosphereMiss` in `rtx_fork_atmosphere.slangh`.
  *Adds the `#ifdef ATMOSPHERE_AVAILABLE` sky-radiance branch in the indirect path miss handler.*

- **Block** at `integratePathVertex` (secondary bounce atmosphere sun NEE call) — ~18 LOC, planned target `fork_hooks::indirectPathAtmosphereSunCall` in `rtx_fork_atmosphere.slangh`.
  *Calls `evalAtmosphereSunNEESecondary` for secondary bounces when physical atmosphere mode is active and NEE is enabled on the bounce.*

- **Block** at `integratePathVertex` (customIndex for view-model lights) — ~4 LOC, planned target `fork_hooks::indirectPathViewModelCustomIndex` in `rtx_fork_light.slangh`.
  *Synthesizes `customIndex` from `rayInteraction.isViewModel` at both RTXDI and advanced-RIS call sites in the indirect path.*

---

## src/dxvk/shaders/rtx/algorithm/lighting.slangh

**Pre-refactor fork footprint:** +27 / -5 LOC (audit 2026-04-18)

**Category:** migrate

- **Block** at `sampleLightRTXDI` signature + ignoreViewModel filter — ~12 LOC, planned target `fork_hooks::sampleLightViewModelFilter` in `rtx_fork_light.slangh`.
  *Adds a `customIndex` parameter (default 0) to `sampleLightRTXDI` and inserts a guard that returns false when the sampled reservoir light has `ignoreViewModel` set and the caller's `customIndex` has `CUSTOM_INDEX_IS_VIEW_MODEL`.*

- **Block** at `sampleLightAdvancedRIS` signature + ignoreViewModel filter — ~10 LOC, planned target `fork_hooks::sampleLightViewModelFilter` in `rtx_fork_light.slangh`.
  *Same `customIndex` parameter and `ignoreViewModel` skip guard in the advanced-RIS sampling loop.*

- **Block** at `sampleLightRIS` dispatch (propagate customIndex) — ~3 LOC, planned target `fork_hooks::sampleLightViewModelFilter` in `rtx_fork_light.slangh`.
  *Updates the `sampleLightRIS` dispatcher to pass `customIndex` through to `sampleLightAdvancedRIS`.*

---

## src/dxvk/shaders/rtx/algorithm/rtxcr/rtxcr_material.slangh

**Pre-refactor fork footprint:** +11 / -4 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `evalSssDiffusionProfile` (~line 167) — 3-line addition for view-model customIndex.
  *Synthesizes `customIndex` from `geometryFlags.isViewModel` and threads it to the SSS diffusion-profile light-sampling calls.*

- **Inline tweak** at `evalSingleScatteringTransmission` (first call site, ~line 323) — 3-line addition for view-model customIndex.
  *Same customIndex pattern for the first single-scattering transmission light sample.*

- **Inline tweak** at `evalSingleScatteringTransmission` (second call site, ~line 423) — 3-line addition for view-model customIndex.
  *Same customIndex pattern for the second single-scattering transmission light sample.*

- **Block** at file scope (`sssApplyAtmosphereCloudShadow` helper) + 3 call sites — fork — 2026-06-27.
  *Folds the cloud-on-terrain shadow onto SSS NEE. The opaque direct/indirect integrators fold the per-pixel cloud transmittance onto the flagged atmosphere sun's `lightSample.radiance` (the `atmosphereCloudShadowed` real-light fold), but `evalSssDiffusionProfile` / `evalSingleScatteringTransmission` sample their own lights internally, so the diffusion-profile, transmission, and single-scattering terms stayed fully sunlit under heavy cloud while surrounding opaque diffuse darkened. Adds a file-local `sssApplyAtmosphereCloudShadow(inout LightSample, lightIdx, surfacePos)` mirroring the integrator fold (same `skyMode==1` / `cloudVoxelShadowsEnable` / `atmosphereCloudShadowed` / `cloudShadowFactorStrength` gating) and calls it at all three SSS light-sample sites. Depends on `sampleCloudGroundShadow_OptionB` from atmosphere_common.slangh, which is why the integrator_direct.slangh include of atmosphere_common was reordered ahead of this header.*

---

## src/dxvk/shaders/rtx/concept/light/light.h

**Pre-refactor fork footprint:** +1 / -0 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `DecodedPolymorphicLight` struct (~line 56) — 1-line addition of `ignoreViewModel` field.
  *Adds `bool ignoreViewModel` to the decoded-light struct so the GPU-side light filter can read the flag after decode.*

---

## src/dxvk/shaders/rtx/concept/light/polymorphic_light.slangh

**Pre-refactor fork footprint:** +2 / -0 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `decodePolymorphicLight` (~line 60) — 1-line addition for ignoreViewModel decode.
  *Extracts bit 1 of the flags word into `decodedPolymorphicLight.ignoreViewModel` during polymorphic-light decode.*

---

## src/dxvk/shaders/rtx/concept/surface/surface.h

**Category:** index-only

- **Inline tweak** at `Surface` flags properties (~line 306, after `isVertexColorBakedLighting`) — REVERTED 2026-06-19, comment-only.
  *Previously added the `Surface::isDecalCategory` property (`data0b.z` bit 2) read by the cloud-shadow zenith gate. Removed with the gate; bit 2 is free again. Only a "freed" comment remains.*

---

## src/dxvk/shaders/rtx/pass/common_binding_indices.h

**Pre-refactor fork footprint:** +9 / -1 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `(file scope)` (atmosphere binding index defines) (~line 49) — 3-line addition.
  *Defines `BINDING_ATMOSPHERE_TRANSMITTANCE_LUT` (200), `BINDING_ATMOSPHERE_MULTISCATTERING_LUT` (201), and `BINDING_ATMOSPHERE_SKY_VIEW_LUT` (202) at high slot numbers to avoid conflicts with pass-specific bindings.*

- **Inline tweak** at `COMMON_BINDING_DEFINITION_LIST` macro (~line 96) — 3-line addition for common-binding list.
  *Adds `TEXTURE2D` entries for the three atmosphere LUT bindings to the common-binding definition macro so they appear in all passes that include common_bindings.*

- **Inline tweak** at `(file scope)` (atmosphere binding index defines) (~line 56) and `COMMON_RAYTRACING_BINDINGS` macro (~line 103) — Stage C addition.
  *Adds `BINDING_ATMOSPHERE_CLOUD_NOISE_3D = 203` and a corresponding `TEXTURE3D` entry in the macro list for the prebaked 3D cloud noise volume (256³ R8 Perlin). No consumer yet; resource and bake pass land in subsequent Stage C tasks.*

- **Inline tweak** at `(file scope)` (atmosphere binding index defines, ~line 57) and `COMMON_RAYTRACING_BINDINGS` macro (~line 104) — Stage C Task 8a addition.
  *Adds `BINDING_ATMOSPHERE_CLOUD_NOISE_SAMPLER = 204` and a corresponding `SAMPLER` entry in the macro list. The linear/REPEAT sampler is bound alongside the cloud noise SRV in `bindAtmosphereLuts` and consumed by `sampleCloudDensityTextured` at call sites (Task 8b).*

- **Inline tweak** at `(file scope)` (atmosphere binding index defines, ~line 58) and `COMMON_RAYTRACING_BINDINGS` macro (~line 108) — FAST-noise jitter (2026-05-09).
  *Adds `BINDING_ATMOSPHERE_FAST_NOISE = 205` and a corresponding `TEXTURE2DARRAY` entry in the macro list. Resource is the EA Importance-Sampled FAST noise (128×128×32 RG8) uploaded once by `RtxFastNoise` and bound in `bindAtmosphereLuts`. Consumed by the `fastJitter()` helper in `atmosphere_common.slangh` for cloud view-march jitter (channel x) and sun-shadow tap jitter (channel y).*

- **Inline tweak** at `(file scope)` (atmosphere binding index defines) and `COMMON_RAYTRACING_BINDINGS` macro — cloud history temporal smoothing (2026-05-09, age channel added 2026-05-13).
  *Adds `BINDING_ATMOSPHERE_CLOUD_HISTORY_PREV = 206` (`TEXTURE2D`) and `BINDING_ATMOSPHERE_CLOUD_HISTORY_CURR = 207` (`RW_TEXTURE2D`). RGBA16F screen-space ping-pong owned by `RtxAtmosphere` and bound in `bindAtmosphereLuts`. Consumed by `evalSkyRadiance` in `atmosphere_sky.slangh` when called with `enableCloudTemporalSmoothing=true` (currently only the primary view ray in `geometry_resolver.slangh` miss path). Smooths per-frame FAST-noise jitter variance to give DLSS a stable signal. The age-channel companion `BINDING_ATMOSPHERE_CLOUD_HISTORY_FRAME_ID_PREV = 212` (`TEXTURE2D`) and `BINDING_ATMOSPHERE_CLOUD_HISTORY_FRAME_ID_CURR = 213` (`RW_TEXTURE2D`) carries the frame index at which each pixel was last refreshed by the sky-miss path; R16_UINT, cleared to 0xFFFF "never written" sentinel at allocation. The shader rejects history whose stored frame-id != `(frameIdx - 1) & 0xFFFFu`, which fixes the multi-frame bright-trail ghosting that the alpha-only disocclusion guard previously left exposed once the 2026-05-13 Nubis Cubed rewrite drove cloud radiance higher.*

- **Inline tweak** at `(file scope)` (atmosphere binding index defines) and `COMMON_RAYTRACING_BINDINGS` macro — cloud voxel grids (Nubis Cubed 2023, 2026-05-12).
  *Adds `BINDING_ATMOSPHERE_CLOUD_D_SUN = 210` and `BINDING_ATMOSPHERE_CLOUD_D_AMBIENT = 211`, both `TEXTURE3D`. 256x256x32 R16F camera-centered tile-wrapped voxel grids storing summed optical depth along the sun direction (D_sun) and zenith (D_ambient). Round-robin baked every 8 frames by `cloud_sun_density_grid.comp.slang` / `cloud_ambient_density_grid.comp.slang` dispatched from `RtxAtmosphere::computeLuts`; bound via `fork_hooks::bindAtmosphereLuts`. Sampled at shade time via `sampleDSun` / `sampleDAmbient` helpers in `atmosphere_common.slangh`. No consumer in this commit — the Nubis Cubed cloud-lighting rewrite (C4-C6 of the 2026-05-12 workstream) reads them.*

- **Inline tweak** at `COMMON_BINDING_DEFINITION_LIST` macro (~line 91) — 1-line addition for sampler readback buffer.
  *Adds `RW_STRUCTURED_BUFFER(BINDING_SAMPLER_READBACK_BUFFER)` to the common binding list (upstream omission fixed).*

---

## src/dxvk/shaders/rtx/pass/common_bindings.slangh

**Pre-refactor fork footprint:** +10 / -0 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `(file scope)` (atmosphere LUT texture declarations) (~line 118) — 7-line addition.
  *Declares `AtmosphereTransmittanceLut`, `AtmosphereMultiscatteringLut`, and `AtmosphereSkyViewLut` as `Texture2D` resources bound at the three atmosphere binding slots.*

- **Inline tweak** at `(file scope)` (atmosphere FAST-noise texture declaration) (~line 138) — 2-line addition (2026-05-09).
  *Declares `AtmosphereFastNoise` as a `Texture2DArray<float2>` resource bound at `BINDING_ATMOSPHERE_FAST_NOISE` (slot 205). Used by the `fastJitter()` helper in `atmosphere_common.slangh` for cloud ray-march sample-distribution jitter.*

---

## src/dxvk/shaders/rtx/pass/composite/composite.comp.slang

**Category:** migrate

- **Block** at `compositePixel` (cloud-shadow application) — REMOVED 2026-06-19, comment-only.
  *The entire screen-space cloud-shadow application in composite is gone. The **indirect** multiply was removed 2026-06-18 (issue #37, double-count + geometry-blind). The **direct** multiply (`pow(PrimaryCloudShadowFactor, cloudShadowFactorStrength)` onto post-denoise primary direct radiance) was removed 2026-06-19 when the cloud shadow was re-architected onto the SUN's radiance pre-denoise inside `sampleAtmosphereSunLight` — it now darkens only the sun (correct indoors for all surface types, no per-pixel gate). The `PrimaryCloudShadowFactor` texture binding is also removed; only a removal comment remains in the shader.*

---

## src/dxvk/shaders/rtx/pass/composite/composite_args.h

**Category:** index-only

- **Inline tweak** — cloud-shadow CB fields, now both reserved pads. *Two former cloud-shadow CB slots are now reserved `float pad1` / `float pad2` (CB layout ABI-unchanged). `pad1` held `cloudShadowFactorStrength` until 2026-06-19, when the cloud shadow moved onto the sun term in the NEE and the composite application was deleted (the knob now lives in `atmosphere_args.h::cloudShadowFactorStrength`, reusing the former `pad_artistic0` slot there). `pad2` held `cloudShadowIndirectStrength` until 2026-06-18 (issue #37 indirect multiply removal). Rename-to-pad rather than delete keeps the slots ABI-stable.*

---

## src/dxvk/shaders/rtx/pass/gbuffer/gbuffer.slang

**Pre-refactor fork footprint:** +16 / -0 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `(file scope module-level pragmas)` (ATMOSPHERE_AVAILABLE defines) (~lines 35–249) — 16-line addition spread across many pragmas.
  *Adds `//!> ATMOSPHERE_AVAILABLE` Slang module dependency annotation lines so the gbuffer module can access atmosphere functionality when the define is active.*

---

## src/dxvk/shaders/rtx/pass/gbuffer/gbuffer_miss.rmiss.slang

**Pre-refactor fork footprint:** +1 / -0 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `(file scope)` (~line 44) — 1-line addition.
  *Adds `#define ATMOSPHERE_AVAILABLE` so the gbuffer miss shader can reference atmosphere evaluation functions.*

---

## src/dxvk/shaders/rtx/pass/gbuffer/gbuffer_psr_miss.rmiss.slang

**Pre-refactor fork footprint:** +1 / -0 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `(file scope)` (~line 44) — 1-line addition.
  *Adds `#define ATMOSPHERE_AVAILABLE` so the gbuffer PSR miss shader can reference atmosphere evaluation functions.*

---

## src/dxvk/shaders/rtx/pass/integrate/integrate_direct.slang

**Category:** index-only

- **Inline tweak** at `(file scope)` (~line 36) — 1-line addition.
  *Adds `#define ATMOSPHERE_AVAILABLE` so the direct-integration pass compiles against the real `evalCloudGroundShadow` body. The macro gates a binding-free fallback intended for atmosphere LUT compute shaders that lack the cloud-noise SRV; this pass already includes `common_bindings.slangh` (which declares `AtmosphereCloudNoise3D` + sampler), so the fallback over-suppresses cloud shadow on terrain surface NEE. Without this define, `evalAtmosphereSunNEE → sampleAtmosphereSunLight → getTransmittanceToSun → evalCloudGroundShadow` short-circuits to `1.0` and terrain never darkens under clouds regardless of `cloudShadowStrength`.*

---

## src/dxvk/shaders/rtx/pass/integrate/integrate_indirect.slang

**Category:** index-only

- **Inline tweak** at `(file scope)` (~line 233) — 1-line addition.
  *Adds `#define ATMOSPHERE_AVAILABLE` so the indirect-integration pass evaluates the real `evalCloudGroundShadow` for secondary-bounce surface NEE (`evalAtmosphereSunNEESecondary`). Same rationale as `integrate_direct.slang`: the cloud-noise SRV is bound via `common_bindings.slangh` here, so the binding-free fallback is unnecessary.*

---

## src/dxvk/shaders/rtx/pass/integrate/integrate_indirect_closesthit.rchit.slang

**Category:** index-only

- **Inline tweak** at `(file scope)` (~line 241) — 1-line addition.
  *Adds `#define ATMOSPHERE_AVAILABLE` so the closest-hit variant of indirect integration evaluates the real `evalCloudGroundShadow`. Same rationale as the sibling `integrate_indirect.slang` entry — `common_bindings.slangh` provides the cloud-noise SRV.*

---

## src/dxvk/shaders/rtx/pass/integrate/integrate_indirect_miss.rmiss.slang

**Pre-refactor fork footprint:** +1 / -0 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `(file scope)` (~line 66) — 1-line addition.
  *Adds `#define ATMOSPHERE_AVAILABLE` so the indirect miss shader can evaluate atmosphere sky radiance on rays that miss all geometry.*

---

## src/dxvk/shaders/rtx/pass/raytrace_args.h

**Pre-refactor fork footprint:** +3 / -0 LOC (audit 2026-04-18)

**Category:** index-only

- **Inline tweak** at `RaytraceArgs` struct (atmosphereArgs + skyMode) (~lines 153, 363) — 2-line addition.
  *Adds `AtmosphereArgs atmosphereArgs` and `uint skyMode` fields to `RaytraceArgs` so the atmosphere parameters and sky mode flag are available in all ray-tracing passes via the constant buffer.*

- **Inline tweak** at `(file scope)` (atmosphere args include) (~line 35) — 1-line addition.
  *Adds `#include "rtx/pass/atmosphere/atmosphere_args.h"` so `AtmosphereArgs` is defined.*

---

## src/dxvk/shaders/rtx/pass/rtxdi/restir_gi_reuse_binding_indices.h

**Pre-refactor fork footprint:** +20 / -20 LOC (audit 2026-04-18)

**Category:** migrate

- **Block** at `(file scope)` (binding index renumbering) — ~20 LOC replacing 20 LOC, planned target `fork_hooks::restirGiBindingIndices` in `rtx_fork_atmosphere.slangh`.
  *Renumbers the ReSTIR GI reuse pass binding indices (WORLD_SHADING_NORMAL_INPUT through RESERVOIR_INPUT_OUTPUT) to make room for the three atmosphere LUT bindings at slots 200-202, avoiding conflicts introduced by the common-bindings expansion.*

---

## src/dxvk/shaders/rtx/pass/tonemap/aces.slangh

- **Fork-owned** — new file. ACES operator implementations: `acesHill` (Stephen Hill ACES fit) and `acesNarkowicz` (Krzysztof Narkowicz ACES approximation). Included by `fork_tonemap_operators.slangh`; dispatched as `tonemapOperatorACESHill` / `tonemapOperatorACESNarkowicz`.
  *Fork-owned ACES operator implementations.*

---

## src/dxvk/shaders/rtx/pass/tonemap/adaptation_v1.slangh

- **Fork-owned** — new file. Tiny helper namespace `adaptation::v1` exposing `ExponentialBlend` and `AdaptAsymmetric` for asymmetric exponential eye adaptation (light-tau / dark-tau). Consumed by `auto_exposure.comp.slang`. Renodx-attributed (Carlos Lopez Jr., MIT 2025).
  *Fork-owned eye-adaptation primitive.*

---

## src/dxvk/shaders/rtx/pass/tonemap/agx.slangh

- **Fork-owned** — new file (renamed from `AgX.hlsl`). AgX Minimal display rendering transform by Benjamin Wrensch (MIT 2024) — `agxMinimalToneMapping(color, saturation, look)`. Depends on `neutwo.slangh` for the max-channel pre-scale that normalizes HDR input into the curve's [0, 1] domain. Included by `fork_tonemap_operators.slangh`.
  *Fork-owned AgX operator implementation (Minimal variant).*

---

## src/dxvk/shaders/rtx/pass/tonemap/auto_exposure.comp.slang

- **Block** at `(file scope)` — full rewrite of the resolve stage to a perceptual observer model. Reads the log-Yf histogram emitted by `auto_exposure_histogram.comp.slang`, computes a count-weighted log mean of Yf as the adapted scene level, and derives the target exposure scale from a first-site cone-contrast law `exposure = targetAdaptedYf / (Y_adapt + Y_noise)` with `Y_noise = 0.0032` (Stockman & Brainard 2010 cone-system noise floor — caps the dark-scene boost without an arbitrary clamp). `targetAdaptedYf` (default 0.18, i.e. mid-gray reflectance) is a push-constant sourced from `cb.targetAdaptedYf` (RtxOption `rtx.autoExposure.targetAdaptedYf`). The raw target is then clamped to `cb.maxExposure` (RtxOption `rtx.autoExposure.maxExposure`, default 8.0) *before* the temporal blend so the smoother converges to the cap directly. Asymmetric exponential blending then runs in log-exposure space so the time-constants are invariant to absolute scene level. Spatial center-weighting (the previously-per-bin log-Yf Gaussian) was lifted into `auto_exposure_histogram.comp.slang`; bin counts arrive pre-weighted so this pass is a plain count-weighted log mean. Replaces the prior Naka-Rushton-in-resolve form (per-bin Gaussian + `1 / (L + sigma)`).
  *Auto-exposure resolve now shares an achromatic basis (Yf) with the psycho17 tonemap operator and exposes a tunable mid-gray target plus a hard exposure cap.*

---

## src/dxvk/shaders/rtx/pass/tonemap/auto_exposure_histogram.comp.slang

- **Inline tweak** at `inputToHistogramBucket` — bin Stockman-Sharpe CIE 170-2 luminosity Yf (via `renodx::tonemap::psycho::yf::from_BT709`) instead of BT.709 photometric luminance, so the resolve pass and the psycho17 observer share a single physiological achromatic measure. Also collapses NaN / negative inputs to bin 0 via `!(yf >= eps)`. Adds `#include "rtx/pass/tonemap/psycho17.slangh"` for the Yf helper. Yf is rescaled by `kYfRefWhite` (Yf-of-BT.709-white, ~1.0504) so 18% linear reflectance lands at Yf == 0.18 exactly — without it `rtx.autoExposure.targetAdaptedYf = 0.18` would silently target ~0.171 reflectance.
- **Inline tweak** at `main` — replaces the unweighted `InterlockedAdd(g_localData[bin], 1)` with spatial Gaussian center-weighted metering. Distance is normalised by the shorter screen axis so the falloff is circular in pixel space and the horizontal periphery on 16:9 / 21:9 doesn't pin adaptation on the sky; sigma = 0.25 of the short axis. Per-pixel weight is fixed-point (gauss × 256, floored to 1u) so any pixel still nudges the histogram and the resolve's `sum(weight*logYf) / sum(weight)` cancels the scale.
  *Histogram now lives in observer-model space (normalised so 18% reflectance ⇔ Yf 0.18) and meters center-weighted instead of uniformly — adaptation tracks the subject the gaze fixates on rather than the unweighted scene mean.*

---

## src/dxvk/shaders/rtx/pass/tonemap/fork_tonemap_operators.slangh

- **Fork-owned** — new file. Hosts the `applyTonemapOperator(uint op, float3 color, bool suppressBlackLevelClamp, ..., float3 adaptiveStateBT709)` dispatcher. Branches on the operator enum and forwards into the operator-specific headers (`aces.slangh`, `hable.slangh`, `agx.slangh`, `lottes.slangh`, `psycho17.slangh`, `gt7.slangh`). The trailing `adaptiveStateBT709` parameter carries the perceptual-AE observer adaptive state into psycho17's `current_adaptive_state_bt709` / `current_background_state_bt709` slots; the other operators ignore it.
  *Fork-owned shader header: operator dispatch lives here so upstream passes shrink to one-line calls.*

---

## src/dxvk/shaders/rtx/pass/tonemap/gt7.slangh

- **Fork-owned** — new file (2026-05-XX). Slang port of the Polyphony Digital "GT7 Tone Mapping" reference (MIT 2025, SIGGRAPH 2025 supplemental). SDR mode, peak hardcoded to 1.0, ICtCp UCS. Wired into the dispatcher via `tonemapOperatorGT7` (= 7).
  *Fork-owned GT7 operator implementation.*

---

## src/dxvk/shaders/rtx/pass/tonemap/hable.slangh

- **Fork-owned** — new file. Hable Filmic (Uncharted 2) tonemap operator — `hableFilmicToneMapping(color, exposureBias, A, B, C, D, E, F, W)`. Split out of `fork_tonemap_operators.slangh` so each operator has its own header. Included by `fork_tonemap_operators.slangh`.
  *Fork-owned Hable Filmic operator implementation.*

---

## src/dxvk/shaders/rtx/pass/tonemap/lottes.slangh

- **Fork-owned** — new file (renamed from `Lottes.hlsl`). Lottes 2016 tonemap operator — `lottesToneMapping(color, hdrMax, contrast, shoulder, midIn, midOut)`. Lottes shares Hable Filmic's 8 param slots in the shader args struct (the two operators are mutually exclusive); slot mapping is documented at the struct definition in `tonemapping.h`. Included by `fork_tonemap_operators.slangh`.
  *Fork-owned Lottes operator implementation.*

---

## src/dxvk/shaders/rtx/pass/tonemap/neutwo.slangh

- **Fork-owned** — new file. Slang port of the renodx "Neutwo" max-channel pre-scale helper — `neutwo_ComputeMaxChannelScale(color)` returns a channel-coherent scale that brings HDR-range input into the [0, 1] curve domain; `neutwo_Neutwo(x) = x * rsqrt(x*x + 1)` is the underlying saturation kernel. Currently consumed by `agx.slangh`. Renodx-attributed (Carlos Lopez Jr., MIT 2025).
  *Fork-owned curve-normalization helper.*

---

## src/dxvk/shaders/rtx/pass/tonemap/psycho17.slangh

- **Fork-owned** — new file. Self-contained Slang port of the renodx "Psycho Test 17" operator and its required color-pipeline dependencies (Stockman-Sharpe LMS, CIE 170-2 MacLeod-Boynton + gamut, Naka-Rushton, color grading). Dispatched as `tonemapOperatorPsycho17` (UI label: `PsychoV17_Beta`). Renodx-attributed (Carlos Lopez Jr., MIT 2025).
  *Fork-owned Psycho Test 17 operator implementation.*

---

## src/dxvk/shaders/rtx/utility/pq.slangh

- **Fork-owned** — new file (2026-05-XX). Shared SMPTE ST.2084 (PQ) constants + `PQDecode` / `PQEncode` (vec3, donut-attributed) + scalar `pq_eotfSt2084` / `pq_inverseEotfSt2084` (GT7-style, frame-buffer units). Extracted from `temporal_aa.comp.slang` so both the TAA pass and the GT7 tonemap operator can share the same math.
  *Fork-owned shared PQ math.*

---

## src/dxvk/shaders/rtx/pass/temporal_aa/temporal_aa.comp.slang

- **Inline tweak** at the include block + PQ constants/helpers (~lines 22-83) — replaced the inlined PQ constants and `PQDecode` / `PQEncode` definitions with `#include "rtx/utility/pq.slangh"` so the GT7 tonemap operator can share them. The function signatures and constant values are unchanged.
  *PQ helpers extracted to a shared header; donut attribution preserved in `utility/pq.slangh`.*

---

## src/dxvk/shaders/rtx/pass/tonemap/tonemapping.h

- **Inline tweak** at `(file scope)` (operator constants) — add `tonemapOperatorNone` / `tonemapOperatorACESHill` / `tonemapOperatorACESNarkowicz` / `tonemapOperatorHableFilmic` / `tonemapOperatorAgX` / `tonemapOperatorLottes` / `tonemapOperatorPsycho17` / `tonemapOperatorGT7` (renamed from the original `tonemapOperatorACES` / `tonemapOperatorACESLegacy` in the 2026-05-XX cleanup; `Psycho11` was renamed to `Psycho17` when the operator was replaced with a port of renodx Psycho Test 17 — see `src/dxvk/shaders/rtx/pass/tonemap/psycho17.slangh` for the MIT attribution to Carlos Lopez Jr. The UI dropdown label is `PsychoV17_Beta`. `tonemapOperatorGT7` was added 2026-05-XX for the Polyphony Digital GT7 SDR port — see `src/dxvk/shaders/rtx/pass/tonemap/gt7.slangh`).
- **Inline tweak** at `ToneMappingAutoExposureArgs` — doc comment updated to describe the new perceptual pipeline (log2-Yf histogram + geometric mean + first-site cone-contrast law + log-space asymmetric blend) instead of the previous BT.709-luminance + Gaussian + Naka-Rushton-in-resolve description. Fields: 2 of the 3 trailing `uint pad` slots were replaced with `float targetAdaptedYf` (mid-gray adaptation target, sourced from `rtx.autoExposure.targetAdaptedYf`) and `float maxExposure` (hard ceiling on the auto-exposure multiplier, sourced from `rtx.autoExposure.maxExposure`); 1 `uint pad0` remains to preserve the 20-byte struct size.
- **Inline tweak** at `ToneMappingApplyToneMappingArgs` struct — swap `finalizeWithACES`/`useLegacyACES` uints for `tonemapOperator` + per-operator param blocks (Hable, AgX, Psycho17). The vestigial `directOperatorMode` field and the legacy histogram / tone-curve bindings (`TONEMAPPING_HISTOGRAM_*`, `TONEMAPPING_TONE_CURVE_*`, `TONEMAPPING_APPLY_TONEMAPPING_TONE_CURVE_INPUT`) and the structs `ToneMappingHistogramArgs` / `ToneMappingCurveArgs` were removed in the 2026-05-XX cleanup. `static_assert(sizeof(...) == 176)` pins the current struct size: AgX block is 16 B (saturation + look + 2 pad floats) since the AgX Minimal operator only consumes those two fields; Psycho17 block is 64 B (14 floats + 2 trailing pad floats for 16-byte alignment).
  *Global tonemap shader-shared header adopts the operator enum.*

---

## src/dxvk/shaders/rtx/pass/tonemap/tonemapping_apply_tonemapping.comp.slang

- **Inline tweak** at `applyToneMapping` — replace `if (cb.finalizeWithACES) { color = ACESFilm(color, cb.useLegacyACES); }` with `color = applyTonemapOperator(cb.tonemapOperator, color, false, ..., adaptiveStateBT709);`. Add `#include "rtx/pass/tonemap/fork_tonemap_operators.slangh"`. The 2026-05-XX cleanup also stripped the dead helpers (`reinhardToneMapper`, `filmicToneMapper`, `dynamicToneMapper`, `lumaAverage`, `setSaturationAverage`) and the `InToneCurve` binding. `adaptiveStateBT709` is the observer adaptive state in post-AE-exposure BT.709 space — currently `(0.18, 0.18, 0.18)` because the perceptual auto-exposure brings the geometric-mean scene Yf to mid-gray; consumed by psycho17, ignored by other operators.
  *Global apply pass routes through the fork dispatcher for operator selection.*

- **Inline tweak** at `applyToneMapping` (dither gate) — fork — 2026-06-27 (experimental). Adds `cb.performSRGBConversion > 0 &&` to the `enableDithering` argument of the inline `ditherTo8Bit` call.
  *The fork's tonemap refactor defers sRGB conversion + dithering to the dedicated `srgb_dither` pass, calling the tonemapper with `performSRGBConversion=false`. The inline dither was left active, so it added its ±0.5/255 perturbation in LINEAR space; the later `linearToGamma` in `srgb_dither` amplifies that ~13x in the near-black sRGB toe, producing a heavy boiling grain visible only in dark regions (plus a redundant double-dither with the dedicated pass). Gating on `performSRGBConversion` disables the inline dither in the current pipeline (the gamma-space `srgb_dither` pass owns dithering) and keeps it correct should the tonemapper ever do its own sRGB encode. VALIDATED in-game 2026-06-27 — dark-region grain gone.*

---

## src/dxvk/shaders/rtx/pass/volume_args.h

**Category:** index-only

- **Inline tweak** at the `VolumeArgs` struct tail (replaces the `vec2 pad0` slot) — 1 LOC delta (2026-05-26). Adds `float fogSunVisibilityGain;` consuming 4 bytes of the existing 8-byte `pad0` slot; `pad0` shrinks to `float pad0;` to preserve 16-byte struct alignment. Consumed by the submodule fork edit at `rtxdi-sdk/include/volumetrics/rtx/algorithm/volume_composite_helpers.slangh` (named knob replaces the previous hardcoded `* 10.0f` artistic gain at the fog-render consumer site). Companions: `rtx_global_volumetrics.{h,cpp}`.

---

## src/dxvk/rtx_render/rtx_fork_atmosphere.cpp

**Category:** fork-owned (modifications by weather preset workstream)

**Note:** This is a fork-owned file. It is listed here because the weather
preset workstream added a call site inside `showAtmosphereUI()`, extending
the fork-owned ImGui surface.

- **Inline tweak** at `fork_hooks::showAtmosphereUI` (weather UI call site) — ~1 LOC.
  *Calls `fork_hooks::showWeatherUI()` between the Moons and Clouds collapsing-header tree blocks so the Weather Presets panel appears in the correct visual position in the Atmosphere dev menu.*

- **Block** (anonymous namespace) + call from `fork_hooks::updateAtmosphereConstants` (directional sun/moon lights — experiment, branch `experiment/atmosphere-directional-sun`, 2026-06-21) — ~180 LOC.
  *When `rtx.atmosphere.useDirectionalLights` is on and skyMode==Numos, injects the sun + each enabled moon as externally-tracked `RtDistantLight`s (via `LightManager::createExternallyTrackedLight` / `updateExternallyTrackedLight`) so they go through the standard NEE/RTXDI path and SSS/decals/viewmodels are handled by the unified pipeline. Radiance is the CPU port of `sampleAtmosphereSunLight`/`sampleAtmosphereMoonLight` ÷ π (distant-light irradiance convention); transmittance is the CPU port of the closed-form `getAtmosphericTransmittanceForDir` so sunset reddening is preserved. The bespoke `evalAtmosphereSunNEE`/`MoonNEE` are gated off via `debugSkyBisectFlags` bit 2. KNOWN v1 limitation: no cloud-on-terrain shadow.*

---

## src/dxvk/rtx_render/rtx_fork_hooks.h

**Category:** fork-owned (forward declaration additions)

**Note:** This is a fork-owned file. It is listed here because the weather
preset workstream added two new forward declarations to the `fork_hooks`
namespace block.

- **Inline tweak** at `fork_hooks` namespace block (forward declarations) — ~2 LOC.
  *Adds `void updateWeatherBlender(class RtxContext& ctx, float deltaTimeSeconds)` and `void showWeatherUI()` forward declarations. These allow `rtx_context.cpp` and `rtx_fork_atmosphere.cpp` to call the weather hook without including the full `rtx_fork_weather.h` header at those call sites.*

---

## submodules/rtxdi/rtxdi-sdk/include/volumetrics/rtx/algorithm/volume_integrator.slangh

**Category:** submodule (fork-controlled — `RemixProjGroup/RTXDI` branch `remix`)

**Note:** This file lives in the fork-controlled RTXDI submodule. Edits land
via PR/commit to `RemixProjGroup/RTXDI` branch `remix` and then a sibling
`dxvk-remix` commit bumps the submodule pointer (mirror of `96c56d5`). The
audit script `scripts/audit-fork-touchpoints.sh` does NOT inspect submodule
files; this entry exists for rebase-safety / human discoverability.

- **Inline tweak** at the top of the file (atmosphere helper include) — 1-line addition.
  *Adds `#include "rtx/pass/atmosphere/atmosphere_common.slangh"` so the per-froxel atmosphere-sun NEE block and the sky-ambient hemisphere integration block can call atmosphere helpers (`sampleAtmosphereSunLightVolume`, `sampleSkyAmbientForVolume`, `hgPhase`).*

- **REMOVED 2026-06-28** — the bespoke atmosphere sun NEE block (`cb.skyMode == 1`, ~35 LOC, originally CattaRappa RTXDI commit `2ff8c57`) that built an `AtmosphereVolumeSunSample` via `sampleAtmosphereSunLightVolume`, traced its own visibility ray, and added the sun to `radianceSH`. *Why removed:* since the directional-light graduation the atmosphere sun/moons are real Remix `RtDistantLight`s sampled by the volume NEE/ReSTIR loop like any light (no distant-light exclusion — `rtx_light_manager.cpp` gives every non-empty type `volumeRISSampleCount ≥ 1`), so this block **double-counted** the sun into the froxel cache (once via NEE, once here) with mismatched phase/visibility math — making volumetric fog impossible to balance (the user had to crank `fogSunVisibilityGain` to 0.01 and it still looked wrong; issue #35). This mirrors the earlier removal of the bespoke **surface** sun NEE. `atmosphereSunVolumetricRadianceScale` (which scaled only this block) is now vestigial for the sun.

- **Inline addition 2026-06-28** — `applyAtmosphereCloudShadowVolume(inout RAB_LightSample, uint lightIdx, vec3 froxelWorldPos)` helper + one call in each NEE branch (ReSTIR + non-ReSTIR fallback) of `integrateVolume`, before `evalNEE`. *Preserves cloud-on-terrain shadowing on volumetric god-rays now that the bespoke block (which applied it) is gone.* Mirrors the surface fold in `integrator_direct.slangh`: when `skyMode == 1 && cloudVoxelShadowsEnable` and the sampled light's `atmosphereCloudShadowed` flag (distant-light GPU flag bit 2) is set, multiplies the light sample's radiance by `pow(sampleCloudGroundShadow_OptionB(froxelPos, dirToLight, …), cloudShadowFactorStrength)`. Guards against `RTXDI_INVALID_LIGHT_INDEX`. Reuses only bindings the removed block already required (`lights[]`, atmosphere helpers via the file's existing `ATMOSPHERE_AVAILABLE` define).

- **Inline tweak** at the end of `integrateVolume` (sky-ambient hemisphere integration block, `cb.skyMode == 1 && cloudSkyAmbientStrength > 0`) — ~50 LOC. *(Workstream 2026-05-12.)*
  *Fixed 6-direction upper-hemisphere integration (zenith + 5 mid-elevation at 30° elevation, 72° azimuth spacing) of `sampleSkyAmbientForVolume(dir, args, AtmosphereSkyViewLut, AtmosphereCloudSkyTransmittanceLut, sampler)` weighted by HG phase against the volumetric anisotropy (0.3). Results scaled by `cloudSkyAmbientStrength`, firefly-filtered, and stored as a single SH entry with zenith as the dominant direction. Gated on `cb.skyMode == 1` and on the strength knob being > 0 so the baseline ships with zero behavior change (`cloudSkyAmbientStrength` default = 0). Consumes the sky-view LUT (slot 202), cloud-sky-transmittance LUT (slot 208), and the cloud-noise sampler (slot 204 — REPEAT, correct on azimuth, never sampled below-horizon). See `docs/superpowers/specs/2026-05-12-volumetric-sky-ambient-design.md`.*

---

## Commit C4 — Cloud render compute pass + Nubis Cubed equations (fork — 2026-05-12)

The C4 commit of the 2026-05-12 cloud-lighting workstream lands the per-
pixel screen-space cloud raymarch with the Nubis Cubed 2023 lighting
equations (paper pp. 137, 142) and a debug view (enum 876) for standalone
A/B against the existing analytical `evalClouds` rendering. No production
consumer yet — the sky-miss composite still calls analytical clouds; the
composite gate lands in C5.

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_render.comp.slang`** — new file (fork-owned).
  *Per-pixel view-direction raymarch through the cloud slab using the Nubis Cubed lighting equations. Reconstructs viewDir from CPU-pushed Y-up basis vectors (`cloudRenderForwardYUp` / `RightYUp` / `UpYUp` in `AtmosphereArgs`, pre-scaled by tan(halfFovX/Y) and aspect). Intersects the curvature-adjusted base/top cloud shells (`intersectSphere`) to get [tEntry, tExit]; marches with per-pixel FAST-noise jitter. At each density-passing sample calls `evalNubisCubedSample` for the page-137 two-HG-lobe direct term + page-142 ambient exp(-D_ambient). Writes premultiplied rgb + view-ray transmittance alpha to AtmosphereCloudRender at slot 209.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned additions.
  *Adds 8 floats of Nubis Cubed lighting params (`cloudPhaseG1/G2`, `cloudMsSunDotMax`, `cloudMsSigmaShallow/Deep`, `cloudMsSdfDepth`) + `cloudRenderFrameIdx` + pad; plus 3 × (vec3 + pad) for the cloud-render camera basis (`cloudRenderForwardYUp`, `cloudRenderRightYUp`, `cloudRenderUpYUp`). All consumed exclusively by `cloud_render.comp.slang`; the basis is pushed CPU-side from `updateAtmosphereConstants` before `computeLuts` runs so the values land in m_constantsBuffer in time.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned additions.
  *Adds five Nubis Cubed lighting helpers (~120 LOC) right after `sampleDAmbient`: `sampleDimProfile` (proxy on `cloudTypeProfile`), `sampleCloudSdf` (slab-distance + density-weighted-depth proxy, returns negative-inside meters clamped to [-cloudMsSdfDepth*4, 0]), `hgPhaseNubis` (paper-flavored HG with denom guard — distinct from the existing `hgPhase` to avoid perturbing non-Nubis callers), `NubisCubedLighting` struct, and `evalNubisCubedSample` (the paper-page-137 two-HG-lobe direct term + page-142 ambient exp(-D_ambient)). Calls `sampleDSun` / `sampleDAmbient` from C1.*

- **`src/dxvk/shaders/rtx/pass/common_binding_indices.h`** — index-only, fork.
  *Adds `BINDING_ATMOSPHERE_CLOUD_RENDER_RT = 209` and a `TEXTURE2D` entry in `COMMON_RAYTRACING_BINDINGS`. Slot 209 was reserved between 208 (cloud-sky-transmittance LUT) and 210 (cloud D_sun); fills the gap.*

- **`src/dxvk/shaders/rtx/pass/common_bindings.slangh`** — index-only, fork.
  *Declares `Texture2D<float4> AtmosphereCloudRender` at slot 209 with a 6-line comment block. Consumed by the cloud render RT debug view (enum 876) and — in C5 — by the sky-miss composite path.*

- **`src/dxvk/rtx_render/rtx_atmosphere.h`** — fork-owned additions.
  *Adds `m_cloudRenderRT` (Resources::Resource), `m_cloudRenderExtent` (VkExtent2D), `m_cloudRenderForwardYUp` / `RightYUp` / `UpYUp` (Vector3), `m_cloudRenderFrameIdx` (uint32_t); plus public methods `getCloudRenderRT()`, `ensureCloudRenderRT(ctx, downscaleExtent)`, `setCloudRenderCameraBasis(forward, right, up, frameIdx)`; plus private `dispatchCloudRender(ctx)`.*

- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned additions.
  *Adds `#include <rtx_shaders/cloud_render.h>`, the `CloudRenderShader` ManagedShader class (7-slot binding parameter list), `ensureCloudRenderRT` (resize-aware alloc of RGBA16F at downscale extent), `setCloudRenderCameraBasis` (member-state setter), `dispatchCloudRender` (rebuilds the args buffer, binds the 7 slots, dispatches 8×8 thread groups), populates the 6 Nubis Cubed lighting fields + 3-basis-vector camera fields + frameIdx into `AtmosphereArgs` in `getAtmosphereArgs()`, calls `dispatchCloudRender(ctx)` from `computeLuts` after the voxel grid bakes, and binds slot 209 (`BINDING_ATMOSPHERE_CLOUD_RENDER_RT`) in `bindResources` (which mirrors the active `bindAtmosphereLuts` site).*

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned additions.
  *Adds 6 `RTX_OPTION` declarations in the `rtx.atmosphere` cluster: `cloudPhaseG1` (default 0.8), `cloudPhaseG2` (0.3), `cloudMsSunDotMax` (0.9), `cloudMsSigmaShallow` (0.25), `cloudMsSigmaDeep` (0.05), `cloudMsSdfDepth` (128.0 meters). All surface as ImGui sliders in the Nubis Cubed Lighting collapsing block.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned additions.
  *In `updateAtmosphereConstants`: reads RtCamera basis (forward/right/up + position) + fov + aspect, applies isZUp swap, pre-scales right/up by tan(halfFovX/Y) and aspect ratio, and pushes via `setCloudRenderCameraBasis` before `computeLuts`. Also calls `ensureCloudRenderRT` with the current downscale extent. In `bindAtmosphereLuts`: adds the cloud render RT bind at slot 209. Adds new `getCloudRenderRT(ctx)` accessor for the debug view. Adds a "Nubis Cubed Lighting (fork — 2026-05-12)" ImGui collapsing header inside the Clouds tree with 6 sliders mapping to the 6 new RTX_OPTIONs.*

- **`src/dxvk/rtx_render/rtx_context.h`** — fork-touchpoint inline tweak.
  *Adds forward declaration `Resources::Resource fork_hooks::getCloudRenderRT(RtxContext&)` and a matching `friend` line inside `class RtxContext`. Mirrors the existing getCloudDSun / getCloudDAmbient pattern.*

- **`src/dxvk/shaders/rtx/utility/debug_view_indices.h`** — index-only, fork.
  *Adds `DEBUG_VIEW_CLOUD_RENDER_RT = 876` with a 4-line comment block.*

- **`src/dxvk/shaders/rtx/pass/debug_view/debug_view.comp.slang`** — fork-owned addition.
  *Adds a `[[vk::binding]]`-decorated `Texture2D<float4> DebugViewCloudRenderRT` declaration and a `case DEBUG_VIEW_CLOUD_RENDER_RT` arm in the main switch that samples the RT via Load and returns its rgb (alpha is the view-ray transmittance, not relevant to the standalone debug view).*

- **`src/dxvk/shaders/rtx/pass/debug_view/debug_view_binding_indices.h`** — index-only, fork. (Inventory substitution: not listed in the Task 4 spec, but structurally required for the debug view case to access the cloud render RT — mirrors the D_sun/D_ambient pattern at slots 35/36.)
  *Adds `DEBUG_VIEW_BINDING_CLOUD_RENDER_RT_INPUT = 37`.*

- **`src/dxvk/rtx_render/rtx_debug_view.cpp`** — fork-owned addition.
  *Adds a `TEXTURE2D(DEBUG_VIEW_BINDING_CLOUD_RENDER_RT_INPUT)` line in the debug-view shader's BEGIN_PARAMETER block, binds the cloud render RT each dispatch via `fork_hooks::getCloudRenderRT`, and adds a label + multi-line description block to the debug-view selector list ("Atmosphere: Cloud Render RT (Nubis Cubed)").*

---

## Commit C5 — Sky-miss composite of cloud RT (gated, default-off) (fork — 2026-05-12)

The C5 commit wires the Nubis Cubed cloud render RT (from C4) as the
primary-ray sky-miss cloud source, gated by a default-off RTX_OPTION
(`cloudRenderRTEnable`). With the gate off, rendering is bit-identical
to pre-C5 — analytical `evalClouds` continues to run at every site.
With the gate on, primary-ray sky-miss reads from the prerendered RT
while indirect, PSR, and reflection rays continue to use analytical
clouds (the RT is at primary-ray pixel coordinates, sampling it for a
non-primary ray direction would return the wrong cloud).

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned addition.
  *Adds `RTX_OPTION("rtx.atmosphere", bool, cloudRenderRTEnable, false, …)` in the `rtx.atmosphere` cluster directly after the C4 Nubis Cubed lighting options. Default false; flipped on in C7 after visual gate.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned addition.
  *Adds `uint cloudRenderRTEnable` plus three `uint` pads at the end of `AtmosphereArgs` for 16-byte alignment. Sits after the C4 cloud-render camera basis block; no existing field offsets change.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_sky.slangh`** — fork-owned addition.
  *In `evalSkyRadiance`: adds a trailing default-false `bool isPrimaryRay` parameter, and a primary-ray-only branch that reads `AtmosphereCloudRender.Load(int3(pixelCoord, 0))` and inverts its transmittance alpha into opacity (`vec4(rgb, 1 - cloudRT.a)`) so the downstream temporal-smoothing / mix composite operates uniformly on either source. Gate is `args.cloudRenderRTEnable != 0u && isPrimaryRay`. Falls through to analytical `evalClouds` when the gate is off OR the caller is non-primary.*

- **`src/dxvk/shaders/rtx/algorithm/geometry_resolver.slangh`** — fork-touchpoint inline tweak.
  *Primary sky-miss call site (`cb.skyMode == 1` block, formerly ending at the `historyResolution` argument) now passes `/*isPrimaryRay=*/ true` as the trailing argument to `evalSkyRadiance`. The PSR call site (line ~2553) keeps its 5-argument call shape and gets `isPrimaryRay=false` via default. ~1 LOC.*

- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned addition.
  *In `getAtmosphereArgs()` (right after the C4 camera-basis populate block): sets `args.cloudRenderRTEnable` from `RtxOptions::cloudRenderRTEnable()` and zeros the three pad slots. ~5 LOC.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned addition.
  *Adds a "Master gate (C5)" separator + `RemixGui::Checkbox("Composite cloud RT at sky-miss", …)` widget at the end of the "Nubis Cubed Lighting" ImGui collapsing header (just below the MS SDF Depth slider). Wired to `RtxOptions::cloudRenderRTEnableObject()`; tooltip explains the primary-ray-only behavior. ~8 LOC.*

---

## Commit C6 — Voxel-grid cloud-on-terrain shadows at NEE (gated) (fork — 2026-05-12)

The C6 commit wires the C3 helper `sampleCloudGroundShadow_OptionB` into the
production surface and volumetric NEE entry points via a multiplicative
ratio correction that replaces the legacy `evalCloudGroundShadow`
uniform-dimmer with the rich 3D `D_sun` voxel-grid lookup. Terrain
gains cumulus-shaped drifting shadow patches that match the cloud
positions overhead. Gated on a default-off RTX_OPTION
(`cloudVoxelShadowsEnable`).

This commit also fixes two pre-existing concerns in the C3 helper that
the diagnostic surfaced: (1) units mismatch — the helper assumed
`worldPos` was in km, but the G-buffer feeds it in engine game units; the
helper now converts via the new `worldUnitsPerKm` field. (2)
camera-relative-vs-world-absolute frame — the voxel grid is
camera-centered, so the helper now subtracts the camera position (pushed
CPU-side via the new `setCloudShadowCameraPosition` setter) before the
`cloudVoxelWorldToUVW` call. The wire-in is intentionally at the NEE
entry points (NOT at `getTransmittanceToSun`) so the sentinel-position
`getTransmittanceToSun` call from `computeGroundReflectionAnalytical`
continues to consume the legacy uniform-dimmer shadow — preserving the
cloud-shadow-map post-mortem's hard-won correctness invariant.

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned addition.
  *Adds two `RTX_OPTION` declarations in the `rtx.atmosphere` cluster directly after the C5 `cloudRenderRTEnable`: `cloudVoxelShadowsEnable` (default false) and `cloudShadowMarchStrength` (default 1.0). The strength knob is the Beer-Lambert exponent multiplier inside `sampleCloudGroundShadow_OptionB`; the C3 commit had to substitute a literal because the field didn't exist on main.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned addition.
  *Adds two 16-byte rows at the end of `AtmosphereArgs` after the C5 block: `(cloudVoxelShadowsEnable, cloudShadowMarchStrength, worldUnitsPerKm, pad_c6_0)` and `(cameraWorldPosYUpKm.xyz, pad_c6_1)`. All consumed exclusively by `sampleCloudGroundShadow_OptionB`; the camera world position is pushed CPU-side from `updateAtmosphereConstants` mirroring the existing `setCloudRenderCameraBasis` pattern.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned additions.
  *Two changes. (1) Inside `sampleCloudGroundShadow_OptionB_impl`: replaces the C3 unit-naive math with a `worldPos * (1 / worldUnitsPerKm)` game-units → km conversion, a `cloudEntryPosKm - args.cameraWorldPosYUpKm` camera-relative reframe before `cloudVoxelWorldToUVW`, and folds `args.cloudShadowMarchStrength` into the Beer-Lambert exponent (replacing the C3 literal). The legacy `cloudShadowStrength` mix at the end is preserved. (2) Adds an `#ifdef ATMOSPHERE_AVAILABLE`-gated ratio correction block inside both `sampleAtmosphereSunLight` (surface NEE, after `result.radiance` is set) and `sampleAtmosphereSunLightVolume` (volumetric NEE, after `result.radiance` is set) that — when `args.cloudVoxelShadowsEnable != 0u` — divides out the `evalCloudGroundShadow` contribution baked into the analytical path and multiplies in the `sampleCloudGroundShadow_OptionB` result. Skips the correction when the old shadow is below 0.001 to guard against divide-by-zero.*

- **`src/dxvk/shaders/rtx/pass/volumetrics/volume_integrate.comp.slang`** — fork-owned addition.
  *Adds `#define ATMOSPHERE_AVAILABLE` before the `common_bindings.slangh` include so the atmosphere helpers consumed by the volumetric pass (`sampleAtmosphereSunLightVolume → sampleCloudGroundShadow_OptionB`, plus the existing `sampleSkyAmbientForVolume`, `sampleDSun`, `fastJitter`) resolve to bound globals. Matches the `#define` already present in `integrate_direct.slang` and `integrate_indirect.slang`.*

- **`src/dxvk/shaders/rtx/pass/volumetrics/volume_restir.comp.slang`** — fork-owned addition.
  *Same `#define ATMOSPHERE_AVAILABLE` addition as `volume_integrate.comp.slang`; the four ReSTIR-stage variants (INITIAL / VISIBILITY / TEMPORAL / SPATIAL_REUSE) all need the cloud voxel-grid bindings available for the per-froxel atmosphere-sun NEE path.*

- **`src/dxvk/rtx_render/rtx_atmosphere.h`** — fork-owned additions.
  *Adds `Vector3 m_cameraWorldPosYUpKm` (default zero) member and public `setCloudShadowCameraPosition(Vector3)` setter to support the per-frame push of the camera world position from `fork_hooks::updateAtmosphereConstants` ahead of `computeLuts`. Mirrors the existing `setCloudRenderCameraBasis` plumbing.*

- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned additions.
  *Implements `setCloudShadowCameraPosition` (member-state setter). In `getAtmosphereArgs()` (right after the C5 sky-miss-composite block): populates the four new C6 fields — gate + strength from RTX_OPTIONs, `worldUnitsPerKm = 100000 * sceneScale` (canonical conversion), and `cameraWorldPosYUpKm` from the cached member.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned additions.
  *In `updateAtmosphereConstants` (immediately after `setCloudRenderCameraBasis`): reads the camera world position in game units, applies the same `toYUp` swap used for the basis vectors, converts to km via `kmPerWorldUnit = 1 / (100000 * sceneScale)`, and pushes via `setCloudShadowCameraPosition`. In the "Nubis Cubed Lighting" ImGui collapsing block (after the Master gate (C5) section): adds a "Cloud-on-terrain shadows (C6)" separator + checkbox bound to `cloudVoxelShadowsEnableObject()` + a `DragFloat` slider bound to `cloudShadowMarchStrengthObject()` with tooltips for both. ~25 LOC total.*

---

## Workstream — Cloud system slides 1+3 lift + samplePos parallax fix (fork — 2026-05-15)

Three lifts from "Real-Time Rendering of Volumetric Clouds in Red
Dead Redemption 2" (Bauer et al., SIGGRAPH 2019) sit on top of the
shipped Nubis Cubed pipeline, plus a one-line bug fix that unblocks
all three visually:

* **Slide 3 — Cloud height LUT.** 64×128 RG8 baked once at startup
  by `cloud_height_lut_baker.comp.slang`. R = per-altitude density
  envelope multiplier. G = per-altitude coverage threshold scale
  (the lever with visible silhouette teeth — lowers the coverage
  gate near cumulus tops to widen the mushroom-cap horizontally).
  Sampled by `cloud_render.comp.slang` via the new
  `cloudHeightProfileFull` helper in `atmosphere_common.slangh`;
  the procedural `cloudTypeProfile` is the fallback for the voxel
  grid bakers and the analytical evalClouds path.
* **Slide 1 — Two-layer cloud map.** The per-pixel march body is
  extracted into a `marchCloudSlab` helper with slab altitude /
  thickness / type / coverage / density-scale parameters. Layer 1
  (primary cumulus) marches first; layer 2 (cirrus deck by default)
  marches after with residual transmittance. Voxel-grid shadows,
  ground-shadow NEE, and moon shadows remain layer-1-only — cirrus
  is optically thin enough that the precompute cost isn't justified.
* **Schneider15 Worley carve.** Three new periodic-3D-Worley helpers
  in `atmosphere_common.slangh` feed the noise bake. Worley FBM is
  subtracted from the Perlin base to carve cell silhouettes (chunky
  cauliflower cumulus instead of smooth Perlin pancakes).
* **`samplePos` parallax fix.** The per-step world position was being
  computed as `viewDirYUp * t` (camera implicitly at origin), gluing
  the noise field to the camera and letting only wind translate it.
  Anchoring `samplePos` to `args.cameraWorldPosYUpKm` gives real
  parallax between near and far parts of the volume as the player
  moves through the world. The three lifts above were inert without
  this; together they restore the depth cues that make clouds read
  as 3D volumes instead of cardboard cutouts translating across the
  sky.

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_height_lut_baker.comp.slang`** — fork-only addition.
  *New one-shot bake compute pass. 8×8 thread groups over a 64×128 R8G8 RWTexture2D. R channel emits the per-type density envelope (trapezoid + Gaussian anvil bump for type > 0.6); G channel emits the per-type coverage threshold scale (1.0 = no effect, drops to ~0.30 at hf ≈ 0.80 for cumulus). Curves are tuned so type values 0 / 0.5 / 1 land close to the procedural `cloudTypeProfile` shape — keeps default-on visual parity with pre-LUT scenes.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_render.comp.slang`** — fork-owned additions.
  *Adds two binding slots (10 = `Texture2D<float2>` height LUT, 11 = `SamplerState` linear/CLAMP); defines `ATMOSPHERE_CLOUD_HEIGHT_LUT_AVAILABLE` before the common header so `cloudHeightProfileFull` resolves to LUT samples. Extracts the per-pixel raymarch into a `marchCloudSlab(slabAltKm, slabThickKm, slabTypeMean, slabCoverageMean, slabDensityScale, ctx, args, inout accumColor, inout viewTransmittance)` helper plus a `CloudShadeContext` struct bundling the sun/moon/sky precomputes. `main()` calls the helper once for layer 1 (always) and once for layer 2 (when `args.cloudLayer2Enable != 0u`). Anchors `samplePos = args.cameraWorldPosYUpKm + viewDirYUp * t` (parallax fix). The "trivially clear sky" early-out widens to `max(layer1, layer2)` coverage so a cirrus-only preset doesn't get short-circuited.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/rtx_cloud_noise_baker.comp.slang`** — fork-owned additions.
  *Adds a `worleyFbm3DPeriodic(worldPosKm * worleyFreq, worleyOctaves, 5.0f, basePeriodWorley)` tap alongside the existing Perlin base + detail FBM, with `worleyFreq` / `worleyOctaves` from `args.cloudWorleyFrequency` / `cloudWorleyOctaves`. Output is `saturate(baseDensity - worley * args.cloudWorleyCarveStrength)` instead of the old smooth Perlin sum.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned additions.
  *Adds three periodic-3D-Worley helpers (`worleyFeaturePoint3D`, `worleyNoise3DPeriodic`, `worleyFbm3DPeriodic`) used only by the bake. Adds `cloudHeightProfileFull` (vec2) + `sampleCloudHeightLUT` gated by `ATMOSPHERE_CLOUD_HEIGHT_LUT_AVAILABLE`, with a procedural fallback via the existing `cloudTypeProfile`. Extends `sampleCloudDensityTextured` and `sampleCloudDensityForShadow` with slab-parametric overloads (slab altitude / thickness / density-scale) plus thin args-default wrappers preserving the original signature; the slab-parametric versions apply the LUT G channel to the coverage-threshold step and the LUT R channel to the density envelope. Also switches the Y texcoord in both density samplers from slab-relative to isotropic `/ args.cloudNoiseTileKm` (correctness alignment with the isotropic bake).*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned additions.
  *Adds two 16-byte rows after the C6 camera block: `(cloudHeightLutEnable, cloudLayer2Enable, cloudLayer2Altitude, cloudLayer2Thickness)` and `(cloudLayer2TypeMean, cloudLayer2CoverageMean, cloudLayer2DensityScale, pad_cloudLayer2_0)`, plus one row for Worley `(cloudWorleyCarveStrength, cloudWorleyFrequency, cloudWorleyOctaves, pad_cloudWorley_0)`.*

- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned additions.
  *Adds `#include <rtx_shaders/cloud_height_lut_baker.h>`, the `CloudHeightLutBakerShader` ManagedShader class (single `RW_TEXTURE2D(0)`), and `dispatchCloudHeightLutBake` (one-shot, called from `initialize()` right after `dispatchCloudNoise3DBake`). Allocates `m_cloudHeightLut` (64×128 R8G8_UNORM) in `createLutResources`. In `dispatchCloudRender`: creates a linear/CLAMP `heightLutSampler`, binds slot 10 (LUT view) + slot 11 (sampler), and adds the resource-tracking line. In `getAtmosphereArgs`: populates the height LUT toggle + six layer-2 fields + three Worley fields from RTX_OPTIONs.*

- **`src/dxvk/rtx_render/rtx_atmosphere.h`** — fork-owned additions.
  *Adds `Resources::Resource m_cloudHeightLut` member, `getCloudHeightLut()` accessor, `dispatchCloudHeightLutBake` declaration, and `kCloudHeightLutWidth = 64` / `kCloudHeightLutHeight = 128` constants.*

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned additions.
  *Adds 10 new RTX_OPTIONs in the `rtx.atmosphere` cluster: `cloudHeightLutEnable` (default on), `cloudLayer2Enable` (default off) + 5 layer-2 tuning floats, and `cloudWorleyCarveStrength` (0.6) / `cloudWorleyFrequency` (1.0) / `cloudWorleyOctaves` (3). The Worley trio is tagged "CHANGE APPLIES ON GAME RELAUNCH" since the noise bake is one-shot at init.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned additions.
  *Adds Height LUT (1 checkbox), Layer 2 (1 checkbox + 5 sliders), and Worley carve (2 `DragFloat` + 1 `DragInt`) ImGui subsections inside the existing Clouds tree, each with tooltips describing the slide source and the relaunch requirement where applicable.*

---

## Workstream — Multiscattering: preset-faithful default + physical-blend knob (fork — 2026-05-26)

Owen's `3e37062b` (`fix(atmosphere): fix multiscattering to match reference two-term model`) on canonical replaced the numerical hemisphere integration in `computeMultiscattering` with an analytical-only fit using heavily blue-biased coefficients (`vec3(0.217, 0.347, 0.594) * 0.02`), and switched `evalAtmosphereRadiance` to sample the multiscattering LUT instead of calling the inline analytical helpers. Two consequences:

1. **Cloud-vs-sky color mismatch at sunset.** `cloud_render.comp.slang` reads the sky-view LUT at the sun direction as the warm ambient source for cumulus. The new bake pumped extra blue energy into the LUT at every elevation, so the cumulus ambient term lost its warm tint and read white against an orange sky.
2. **Preset color washed across all defaults.** With the LUT consumed by `evalAtmosphereRadiance`, the hemisphere integration's wavelength bias amplified each preset's Rayleigh into the sky (Earth too blue, Desert blue-ish, Mars desaturated). The previously-inline `getAnalyticalMultiscattering` was a tame curve fit that the presets were calibrated against.

Fork resolution: restore the numerical hemisphere integration in the LUT bake AND keep the inline analytical multiscattering as the default in `evalAtmosphereRadiance`, with a per-preset knob to blend in the LUT-based physical version when realism is wanted.

- **`src/dxvk/shaders/rtx/pass/atmosphere/multiscattering_lut.comp.slang`** — restoration to pre-`3e37062b` state.
  *Restores `computeMultiscattering` to the 64-direction × 20-march-sample hemisphere integration (Hillaire 2nd-order scattering primary term) plus ground reflection plus analytical fit. Restores `computeAnalyticalMultiscattering` coefficients to the toned-down `vec3(0.35, 0.38, 0.45) × 0.01` ("more neutral, less blue-heavy") tuning that prevents purple cast when combined with sunset orange.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned inline tweak.
  *Inside `evalAtmosphereRadiance`'s per-sample loop: replaces the post-`3e37062b` LUT-only multiscattering with a `lerp(contribAnalytical, contribLut, args.multiScatterPhysicalStrength)` blend. `contribAnalytical` calls the existing inline `getAnalyticalMultiscattering` (preset-faithful, the pre-`3e37062b` shape); `contribLut` samples the multiscattering LUT (the now-restored hemisphere integration). Default 0.0 = byte-identical to pre-`3e37062b`; 1.0 = full physical.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned addition.
  *Replaces the existing `uint pad2` slot at the end of the LUT-dims 16-byte row with `float multiScatterPhysicalStrength`. No struct layout change; existing field offsets unchanged.*

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned addition.
  *Adds `RTX_OPTION("rtx.atmosphere", float, multiScatterPhysicalStrength, 0.0f, …)` immediately after `sunIlluminance` in the `rtx.atmosphere` cluster.*

- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned addition.
  *Inside `getAtmosphereArgs()`: sets `args.multiScatterPhysicalStrength = RtxOptions::multiScatterPhysicalStrength()`. Removes the now-stale `args.pad2 = 0` write (slot is the new typed field). ~2 LOC net.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned addition.
  *Adds a `RemixGui::DragFloat("Multiscatter Physical Strength", …, 0.0f, 1.0f, "%.2f", sliderFlags)` widget at the end of the Atmosphere → Advanced ImGui tree (right after Ozone Layer Width), with a tooltip explaining the artistic-vs-physical tradeoff. ~6 LOC.*

---

## Workstream — Cloud realism: edge detail erosion, bottom darkening, vertical stretch (fork — 2026-06-10)

Three perceptual cloud upgrades, all live-tunable, all reverting to the prior look at their zero/identity values. (1) **Edge detail erosion** — a second tap of the prebaked Worley-carved noise volume at `cloudDetailScale`× the base frequency perturbs the density field *before* the coverage gate, displacing the silhouette iso-surface with 60–500 m cauliflower teeth while saturated cores stay solid. (2) **Bottom darkening** — a vertical light gradient on the Nubis Cubed multi-scatter and ambient terms; the paper's `M` term barely attenuates (`sigma_ms` ≈ 0.05–0.25) and `pow(1 - dim_profile, 0.5)` is top/bottom symmetric, so undersides previously read uniformly lit. The direct beam is exempt so silver linings survive. (3) **Vertical stretch** — lowers the Y noise-sample frequency relative to horizontal so cloud bodies become convective columns rather than round blobs (towering cumulus). New args ride in repurposed pad slots; constant-buffer layout unchanged.

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned additions.
  *In `sampleCloudDensityTextured`: step 4b applies the detail-erosion tap pre-threshold (`density += strength * 0.6 * (detailNoise - 0.6)`, erosion-biased), and the texcoord mapping divides Y frequency by `cloudVerticalStretch`. `sampleCloudDensityForShadow` applies the identical Y-stretch (the baked D_sun / D_ambient grids must describe the same shapes the view-march renders) but skips the detail tap (cheap path; un-eroded grids over-shadow edges conservatively). In `evalNubisCubedSample`: `bottomGradient = mix(1 - cloudBottomDarkening, 1, smoothstep(0, cloudBottomDarkeningHeight, heightFraction))` multiplies the multi-scatter `M` and `L_ambient` terms only.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned additions.
  *Repurposes `pad_cloudVoxel0..2` as `cloudBottomDarkening` / `cloudBottomDarkeningHeight` / `cloudDetailStrength`, `pad_nubisCubed0` as `cloudDetailScale`, and `pad_cloudLayer2_0` as `cloudVerticalStretch`. No layout change (same pattern as `cloudNoiseWarpStrength`).*

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned additions.
  *Adds 5 RTX_OPTIONs to the `rtx.atmosphere` cluster: `cloudDetailStrength` (0.6), `cloudDetailScale` (4.3), `cloudBottomDarkening` (0.55), `cloudBottomDarkeningHeight` (0.65), `cloudVerticalStretch` (1.6).*

- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned additions.
  *Inside `getAtmosphereArgs()`: populates the five new args from RtxOptions, replacing the former pad zero-writes. ~5 LOC net.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned additions.
  *Adds "Edge Detail" + "Vertical Stretch" `DragFloat`s to the Clouds → Coverage & Shape ImGui section and "Bottom Darkening" to Clouds → Lighting. The fine-tune knobs (`cloudDetailScale`, `cloudBottomDarkeningHeight`) stay user.conf-only per the 2026-05-19 menu-simplification discipline.*

---

## Workstream — Cloud march/bake perf pass + vertical-coherence rework (fork — 2026-06-10)

Performance pass on the per-frame cloud noise evaluation, each piece validated in-game via a step-by-step bisect ladder after an earlier all-at-once landing broke the cloudscape (root causes: an uncompensated warp-amplitude loss and config drift contaminating the first bisect — see the cloud-realism memory notes). Also replaces the rev-1 `cloudVerticalStretch` Y-domain stretch (stacked-puffs artifact) with a fixed-slice coherence blend, shipped EXPERIMENTAL/default-inert because its look (vertical smearing at high values) isn't approved either.

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned additions.
  *(1) Anti-tile warp refactored into offset form (`computeCloudNoiseWarpOffsetKm`) and HOISTED to once per ray in the two voxel-grid bake integrals (was 4 perlin evals per tap × 8 taps × 2.1M voxels × 2 grids per frame; the warp's ~25 km shortest wavelength makes per-tap recompute pure waste). `sampleCloudDensityForShadow` gains a `warpOffsetKm` caller-provided parameter. (2) Warp octaves switched from constant-z `perlinNoise3D` planes to new `perlinNoise2D` (4 corners, branch-free gradient pick, ~2.5× cheaper) with `kWarp2DGain = 1.4` compensating the unit-vs-sqrt(2) gradient amplitude — the uncompensated version visibly weakened the warp's load-bearing field shred. (3) The per-sample 720 cycles/km analytic perlin detail is removed (1.4 m wavelength vs ≥100 m steps = unresolvable grain), preserving its hidden −0.05 mean bias as a constant. (4) `cloudVerticalStretch` rev 2: fixed-Y-slice coherence blend with variance re-expansion anchored on the bake's ~0.4 mean, in BOTH density samplers so the self-shadow grids match.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_render.comp.slang`** — fork-owned additions.
  *Coverage/type Worley control fields hoisted out of the march loop: evaluated at slab entry + exit (4 evals/ray), lerped per step (was 2 × 9-hash evals per step × 32 steps).*

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned addition.
  *`cloudVerticalStretch` default 1.6 → 1.0 (bit-exact identity) and description rewritten to mark the feature EXPERIMENTAL pending a sky-system-level solution to towering cumulus.*

---

## Workstream — Secondary-ray cloud LUT (fork — 2026-06-10, perf)

Every indirect / PSR / reflection ray reaching sky-miss previously ran the full analytical `evalClouds` march in `evalSkyRadiance` — a hidden per-ray cost estimated to rival the visible cloud pass. Those rays now sample a 256×128 RGBA16F dome LUT (azimuth × elevation, horizon-concentrated `elevation = (π/2)·v²`) baked once per frame with the same Nubis Cubed march the visible cloud RT uses. Deliberate look change: secondary rays now see the same clouds the primaries see instead of the legacy Wrenninge analytical approximation; `rtx.atmosphere.cloudSecondaryLutEnable = False` restores the legacy per-ray march. Cloud parallax across scene-scale ray-origin offsets is negligible versus km-scale cloud distances — the same camera-anchored approximation the per-frame voxel grids already make.

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_march_common.slangh`** — NEW fork-owned file.
  *Shared cloud-march library extracted from `cloud_render.comp.slang`: `CloudShadeContext` + `buildCloudShadeContext` (the pixel-independent sun/sky/moon precompute that previously lived in main()), `sampleCloudSunOpticalDepth_local`, `marchCloudSlab` (with generic `tMinClamp` / `tMaxClamp` segment params; 0/0 reproduces the pre-extraction march bit-for-bit), and the `marchCloudLayers` layer-1+2 wrapper. Included after `atmosphere_common.slangh` under the established binding-then-include layout.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_secondary_lut.comp.slang`** — NEW fork-owned file.
  *Per-frame dome LUT bake. Bindings 0–11 in lockstep with `cloud_render.comp.slang` (slot 6 is its own RW output). Marches the full slab per dome texel at `cloudViewSamples` steps via the shared `marchCloudLayers`; clear-sky early-out mirrors `cloud_render`; frame-constant per-texel jitter keeps the LUT temporally stable.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_render.comp.slang`** — fork-owned changes.
  *March implementation moved to `cloud_march_common.slangh`; main() now builds the shade context via `buildCloudShadeContext` and calls `marchCloudLayers` with 0/0 clamps — behavior-neutral extraction, bit-identical output.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned addition.
  *`cloudDomeUvToDir` / `cloudDomeDirToUv` dome-direction mapping shared by the LUT writer and the `evalSkyRadiance` sampler (lives here because the sample side does not include the march library).*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_sky.slangh`** — fork-owned addition.
  *New middle branch in `evalSkyRadiance`'s cloud-layer selection: non-primary rays with `cloudSecondaryLutEnable` sample `AtmosphereCloudSecondaryLut` via `cloudDomeDirToUv` + the sky-view sampler (REPEAT-U wraps the azimuth seam) and reconcile alpha exactly like the existing cloud-RT branch (opacity = 1 − transmittance). Below-horizon directions clamp to the horizon row.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned addition.
  *Repurposes `pad_c5_0` as `uint cloudSecondaryLutEnable`. No layout change (same pattern as `cloudNoiseWarpStrength`).*

- **`src/dxvk/shaders/rtx/pass/common_binding_indices.h`** — upstream-touched, inline tweak (~12 LOC).
  *Adds `BINDING_ATMOSPHERE_CLOUD_SECONDARY_LUT 215` next to the other fork atmosphere bindings and appends `TEXTURE2D(BINDING_ATMOSPHERE_CLOUD_SECONDARY_LUT)` to `COMMON_RAYTRACING_BINDINGS`.*

- **`src/dxvk/shaders/rtx/pass/common_bindings.slangh`** — upstream-touched, inline tweak (~10 LOC).
  *Declares `Texture2D<float4> AtmosphereCloudSecondaryLut` at the new binding, alongside the other fork atmosphere declarations.*

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned addition.
  *Adds `RTX_OPTION("rtx.atmosphere", bool, cloudSecondaryLutEnable, true, …)` directly before the C5 `cloudRenderRTEnable`.*

- **`src/dxvk/rtx_render/rtx_atmosphere.h` / `rtx_atmosphere.cpp`** — fork-owned additions.
  *`CloudSecondaryLutShader` class, `m_cloudSecondaryLut` resource (256×128 RGBA16F) + `getCloudSecondaryLut()` accessor + `kCloudSecondaryLut*` constants, `dispatchCloudSecondaryLut` (per-frame, after the voxel-grid bakes behind the existing write→read barrier, gated on the option), `getAtmosphereArgs` populates the gate from RtxOptions, and `bindResources` binds the new slot.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned additions.
  *`fork_hooks::bindAtmosphereLuts` binds the LUT at `BINDING_ATMOSPHERE_CLOUD_SECONDARY_LUT`; adds the "Fast Cloud Reflections" checkbox to the Clouds ImGui tree.*

---

## Workstream — Half-res cloud render RT (fork — 2026-06-11, perf)

The visible cloud march ran once per DLSS-input pixel at up to 32 steps. Clouds are soft, low-frequency content, so the cloud RT is now allocated at `cloudRenderResolutionScale` (default 0.5) of the downscale extent and bilinearly upsampled at the sky-miss composite — ~4× fewer marched pixels at the default. The temporal-smoothing path runs after the upsample at full downscale resolution, so its stabilization is unchanged. Scale 1.0 lands the sample uv on texel centers of a same-size RT and matches the legacy `Load` to float precision (live A/B via the "Cloud Render Scale" slider).

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_sky.slangh`** — fork-owned change.
  *The primary-ray cloud-RT branch in `evalSkyRadiance` normalizes pixelCoord by `args.cloudRenderFullDimX/Y` and bilinearly samples the RT via the sky-view sampler (uv clamped a half-texel inside so screen content never wraps through REPEAT-U); falls back to the legacy `Load` while the published extent is still zero (first frames).*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned addition.
  *Repurposes `pad_c5_1/2` as `cloudRenderFullDimX/Y` (the downscale extent the RT is composited into). No layout change.*

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned addition.
  *Adds `RTX_OPTION("rtx.atmosphere", float, cloudRenderResolutionScale, 0.5f, …)`.*

- **`src/dxvk/rtx_render/rtx_atmosphere.h` / `rtx_atmosphere.cpp`** — fork-owned additions.
  *`ensureCloudRenderRT` records the full downscale extent in new `m_cloudRenderFullExtent` and allocates the RT at the clamped scale (a live scale change reallocates via the existing extent-mismatch path); `getAtmosphereArgs` publishes the full extent. Dispatch group math already keys off the (now scaled) `m_cloudRenderExtent`.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned addition.
  *Adds the "Cloud Render Scale" DragFloat to the Clouds ImGui tree.*

---

## Workstream — Cloud noise hex de-tiling, stage A (fork — 2026-06-11)

Root-cause fix for the prebaked noise volume's periodic repeat (the artifact the anti-tile warp was added to hide). The horizontal plane is partitioned into an equilateral-triangle lattice (golden-ratio multiple of the tile period, so lattice and texture can't resonate); each vertex carries a hash-derived random texture-space XZ offset, and samples blend their triangle's three offset taps with Heitz & Neyret 2018's variance-preserving operator (mean/variance of the field preserved → look unchanged; periodicity destroyed). Weights are pow-4 sharpened so triangle interiors (~70% of area) take a 1-tap fast path; only edge bands pay 3 trilinear taps. Ships alongside the UNCHANGED anti-tile warp — stage B walks the warp down with bake-frequency compensation after in-game validation (the warp's frequency-multiplying Jacobian is load-bearing for the look; see the cloud-realism memory notes).

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned additions.
  *New `CloudHexTile` / `computeCloudHexTile` (skewed triangle grid + per-vertex hash offsets + pow-4 weight sharpening + dominant-vertex sort), `kCloudNoiseFieldMean` (0.4 — shared anchor with the vertical-coherence re-expansion), and `sampleCloudNoiseHexBlend` (3-tap variance-preserving blend). `sampleCloudDensityTextured`'s base tap branches: hex fast path = tricubic at the dominant offset, blend zones = 3-tap trilinear blend, off = legacy tricubic. `sampleCloudDensityForShadow` mirrors the same lattice/offsets/blend trilinearly so the baked D_sun / D_ambient grids shadow the same de-tiled field. The vertical-coherence footprint slice follows the dominant vertex's offset in both samplers. The detail-erosion tap stays single-tap (its sub-3 km repeat is not the visible artifact; revisit at stage B).*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned addition.
  *Repurposes `padCloudLook0` as `float cloudHexTilingEnable`. No layout change.*

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned addition.
  *Adds `RTX_OPTION("rtx.atmosphere", bool, cloudHexTilingEnable, true, …)` before `cloudNoiseWarpStrength`.*

- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned addition.
  *`getAtmosphereArgs` populates the gate from RtxOptions (replacing the former pad zero-write).*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned addition.
  *Adds the "Seamless Cloud Field" checkbox to the Clouds ImGui tree, next to the Anti-Tile Warp slider it will eventually retire.*

---

## Workstream — Cloud noise hex de-tiling, stage B enabler (fork — 2026-06-11)

Exposes the bake's base/detail FBM frequency as a live-rebake multiplier so the anti-tile-warp walk-down can be tuned interactively: the warp's Jacobian multiplies the field's horizontal frequency 2-3x and the tuned look depends on that shred, so as `cloudNoiseWarpStrength` comes down, `cloudNoiseBaseFreqScale` goes up to fold equivalent frequency content into the bake itself. Default 1.0 = bit-identical bake; the actual walk-down is an in-game tuning session, not a code change.

- **`src/dxvk/shaders/rtx/pass/atmosphere/rtx_cloud_noise_baker.comp.slang`** — fork-owned change.
  *`baseFreq` / `detailFreq` become `0.5 / 2.0 × clamp(args.cloudNoiseBaseFreqScale, 0.25, 4)` (4x ratio preserved); the periodic-lattice snapping already handles non-default products.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned addition.
  *Repurposes `padCloudLook1` as `float cloudNoiseBaseFreqScale`. No layout change.*

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned addition.
  *Adds `RTX_OPTION("rtx.atmosphere", float, cloudNoiseBaseFreqScale, 1.0f, …)`.*

- **`src/dxvk/rtx_render/rtx_atmosphere.h` / `rtx_atmosphere.cpp`** — fork-owned additions.
  *`getAtmosphereArgs` populates the new field; `needsCloudNoiseRebake` / `cacheCloudNoiseBakeInputs` gain the new input (`m_cachedBaseFreqScale`) so dragging the slider re-bakes the volume live.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned addition.
  *Adds the "Noise Frequency" DragFloat next to the Anti-Tile Warp slider with the walk-down pairing explained in its tooltip.*

---

## Workstream — Cloud noise hex de-tiling, stage B landing (fork — 2026-06-11)

The in-game warp walk-down validated cleanly at warp **0** with the bake frequency left at its default 1.0 — with hex-tiling killing the repeat at the source, no frequency compensation turned out to be needed. The warp default is retired accordingly; the knob survives as a pure organic-distortion effect. Warp 0 also takes the early-out in `applyCloudNoiseAntiTileWarp`, refunding 4 perlin2D evaluations per march sample.

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned change.
  *`cloudNoiseWarpStrength` default 0.75 → 0.0; doc strings for it and `cloudNoiseBaseFreqScale` updated to describe the post-retirement roles (warp = optional look knob, hex-tiling owns de-tiling).*

---

## Workstream — Anti-tile warp removal (fork — 2026-06-11)

Follow-up to the stage B landing: with the warp retired at 0 and hex de-tiling user-validated as the root-cause fix, the entire warp implementation is dead code and is removed. All deleted code was fork-owned; no upstream lines return.

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned removal.
  *Deletes `perlinGradient2D` / `perlinNoise2D` (warp-only helpers), `computeCloudNoiseWarpOffsetKm` / `applyCloudNoiseAntiTileWarp`, the warp call in `sampleCloudDensityTextured`, the `warpOffsetKm` parameter from both `sampleCloudDensityForShadow` overloads, and the per-ray warp hoists in the D_sun / D_ambient bake integrals.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned change.
  *`cloudNoiseWarpStrength` reverts to padding (`padCloudC2`, its original slot) — CB layout unchanged.*

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned removal.
  *Drops the `cloudNoiseWarpStrength` RTX_OPTION; `cloudHexTilingEnable` / `cloudNoiseBaseFreqScale` doc strings scrubbed of warp pairing references.*

- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned change.
  *`getAtmosphereArgs` zeroes `padCloudC2` instead of populating the warp strength.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned removal.
  *Drops the "Anti-Tile Warp" slider from the Clouds ImGui tree; "Seamless Cloud Field" / "Noise Frequency" tooltips scrubbed of warp references.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/rtx_cloud_noise_baker.comp.slang`** — comment-only cleanup (walk-down rationale removed).

---

## Workstream — Sky perf: split sky-LUT cache keys (fork — 2026-06-11)

First change of the non-cloud sky optimization workstream. The three sky LUT bakes (transmittance / multiscatter / sky-view) shared one memcmp gate over the whole normalized `AtmosphereArgs`, with two per-frame failure modes: the game-driven sidereal `starRotation` (pushed every frame at night, feeds no LUT bake) re-baked the full cascade every frame, and a moving time-of-day sun re-baked the heavy transmittance + multiscatter pair (multiscatter alone: 32×32 texels × 64 dirs × 20 steps of transmittance taps) even though neither reads sun direction. Each bake now compares a key normalized down to the fields it actually reads. Gated by `skyLutCacheKeySplitEnable` (default on; legacy single-gate path preserved verbatim for A/B).

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned addition.
  *Adds `RTX_OPTION("rtx.atmosphere", bool, skyLutCacheKeySplitEnable, true, …)` next to the other atmosphere perf gates.*

- **`src/dxvk/rtx_render/rtx_atmosphere.h` / `rtx_atmosphere.cpp`** — fork-owned additions.
  *New `normalizeForSkyViewLutKey` (base normalize + star / Milky Way fields zeroed) and `normalizeForTransmittanceMsKey` (additionally zeroes sun direction / illuminance / disk size, Mie g, MS blend weight, and all moon fields), cached as `m_cachedSkyViewKey` / `m_cachedTransmittanceMsKey`. `computeLuts` gates the transmittance+MS pair and the sky-view bake independently (tms-dirty implies sky-view-dirty; barrier ordering unchanged); both branches keep the other path's caches coherent so the option can be toggled live without a spurious re-bake.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned addition.
  *Adds the "Minimal Sky LUT Re-bakes" checkbox to the Atmosphere → Advanced ImGui tree.*

Note: `rtx.atmosphere.cloudNoiseWarpStrength` lines in existing user confs become unknown-option no-ops after this change.

---

## Workstream — Sky perf: per-dispatch bisect toggles (fork — 2026-06-11, diagnostic)

Measurement aid for the sky perf workstream. The atmosphere pass runs several per-frame dispatches that no production option can skip — `dispatchCloudRender` in particular runs whenever its RT is valid, independent of `cloudRenderRTEnable` — so frame-time A/B via the production toggles mis-attributes their cost. Three default-ON `debugDispatch*` toggles let a live ImGui session skip each dispatch individually and read the frame-time delta. Skipping leaves consumers reading stale data (frozen clouds / shadows): diagnostic only, defaults unchanged rendering.

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned additions.
  *`debugDispatchCloudVoxelGrids` / `debugDispatchCloudRender` / `debugDispatchCloudSkyTransmittance`, all default true.*

- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned change.
  *`computeLuts` wraps the D_sun + D_ambient bakes, the cloud render dispatch, and the cloud-sky-transmittance bake in their respective toggles (barriers move inside the gates; ordering unchanged when enabled).*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned addition.
  *New "Perf Bisect (Diagnostic)" ImGui tree with the three checkboxes + pointers to the existing production toggles useful in the same session.*

---

## Workstream — Sky perf: NEE shadow-ray budget clamps (fork — 2026-06-11)

Third change of the sky perf workstream — the first to target where the milliseconds actually are. Bisect results showed all atmosphere compute dispatches together ≈ the known 2 ms cloud budget; the remaining ~2.7 ms (day) / ~3.7 ms (night) of the skyMode A/B delta lives in the integrators: with physical atmosphere on, sun NEE traces an anisotropy-driven 1–12 soft-shadow visibility rays per primary pixel (`getSunSoftShadowParams`; ~4 at typical Mie g) plus half that per indirect bounce vertex, and moon NEE adds a constant 4 (+2 indirect) at night. The denoised pipeline temporally converges one blue-noise-jittered ray per frame, so the loops are oversampled. Two clamp knobs cap the counts at the single source (`sampleAtmosphereSunLight` / `sampleAtmosphereMoonLight`), so primary and the half-rate secondary paths both inherit. Defaults 0 = legacy uncapped (bit-identical); set 1 for the perf win after visual validation.

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned change.
  *Repurposes `padMoonNee0` / `padMoonNee1` as `uint sunShadowMaxSamples` / `uint moonShadowMaxSamples`. No layout change.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned additions.
  *Clamp applied after `getSunSoftShadowParams` in `sampleAtmosphereSunLight` and after the constant-4 assignment in `sampleAtmosphereMoonLight`; 0 = uncapped.*

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned additions.
  *`RTX_OPTION("rtx.atmosphere", int, sunShadowMaxSamples, 0, …)` / `moonShadowMaxSamples`.*

- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned change.
  *`getAtmosphereArgs` populates the two fields (replacing the pad zero-writes).*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned addition.
  *"Sun Shadow Ray Cap" / "Moon Shadow Ray Cap" DragInts in the Atmosphere → Advanced ImGui tree.*

---

## Workstream — Sky perf: shader-path bisect toggles (fork — 2026-06-11, diagnostic)

Extends the per-dispatch bisect kit into the integrators: with the dispatch costs known (~1.8 ms ≈ cloud budget) and the NEE rays capped, the residual skyMode delta splits across two shader paths that host-side toggles can't reach. Two default-true diagnostic options pack into one CB gate word (`debugSkyBisectFlags`, former `padMoonNee2` slot — layout unchanged): bit 0 skips atmosphere sun+moon NEE at both integrator call sites (sun/moon lighting goes black), bit 1 makes `evalSkyRadiance` return flat grey immediately (isolates the full per-ray miss-path cost; the skipped temporal-history write self-heals on re-enable via the frame-id age check). Defaults = bits clear = production paths, bit-identical.

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned change.
  *Repurposes `padMoonNee2` as `uint debugSkyBisectFlags`. No layout change.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_sky.slangh`** — fork-owned addition.
  *Flat-grey early return at the top of `evalSkyRadiance` behind bit 1.*

- **`src/dxvk/shaders/rtx/algorithm/integrator_direct.slangh` / `integrator_indirect.slangh`** — fork-owned change (upstream files, one-line condition edits).
  *The `cb.skyMode == 1` atmosphere-NEE gates additionally require bit 0 clear.*

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned additions.
  *`debugEnableAtmosphereNee` / `debugEnableSkyMissShading`, both default true.*

- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned change.
  *`getAtmosphereArgs` packs the two options into the flag word.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned addition.
  *"Atmosphere Sun/Moon NEE" / "Full Sky Miss Shading" checkboxes in the Perf Bisect (Diagnostic) ImGui tree.*

Follow-up (same day): `debugDispatchSkyLuts` (rtx_options.h, default true) freezes the whole LUT bake cascade — the last untoggleable GPU piece of the atmosphere; with a moving time-of-day sun the sky-view LUT re-bakes every frame by design, and this measures that. `computeLuts` short-circuits the gate block when off (rtx_atmosphere.cpp); "Sky LUT Bakes" checkbox added to the Perf Bisect tree (rtx_fork_atmosphere.cpp).

---

## Workstream — Sky perf: sky-view re-bake granularity (fork — 2026-06-11)

Production fix for the per-frame sky-view re-bake the frozen-cascade bisect identified (objective frame-time improvement, no visual hit). With a continuously-animating time-of-day sun, the sky-view cache key sees a new `sunDirection` every frame. `skyViewRebakeGranularityDeg` (default 0 = legacy every-frame behavior; recommended 0.1) quantizes the sun and moon directions INSIDE the sky-view cache key, so continuous motion flips the memcmp only when a direction crosses a granularity step — the bake itself always uses exact current values, so each re-bake is exact and staleness between re-bakes is bounded by the step angle. All non-direction fields stay exact (sliders / presets re-bake immediately). Applies to the split-key path (`skyLutCacheKeySplitEnable`, on by default).

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned addition.
  *`RTX_OPTION("rtx.atmosphere", float, skyViewRebakeGranularityDeg, 0.0f, …)`.*

- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned addition.
  *`quantizeDirComponent` helper; `normalizeForSkyViewLutKey` quantizes `sunDirection` + all moon directions by the option's step (no-op at 0). The transmittance/MS key is unaffected (it zeroes those fields outright).*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned addition.
  *"Sky Re-bake Granularity" DragFloat in the Atmosphere → Advanced ImGui tree.*

---

## Workstream — Sky perf: pin validated defaults + retire workstream UI (fork — 2026-06-11)

Closing commit of the sky perf workstream. In-game validation confirmed the two production levers (sun/moon NEE shadow-ray caps at 1, sky-view re-bake granularity at 0.1°) deliver the measured wins with no visual hit, so their defaults are pinned and the workstream's ImGui surface is retired per the user's call ("this is in a good enough spot now"). All options remain conf-tunable; only the UI is removed.

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned change.
  *Defaults: `sunShadowMaxSamples` 0 → 1, `moonShadowMaxSamples` 0 → 1, `skyViewRebakeGranularityDeg` 0.0 → 0.1. Doc strings note the validation; 0 still selects legacy behavior via conf.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned removal.
  *Drops from the UI: the "Minimal Sky LUT Re-bakes" checkbox, the "Sun/Moon Shadow Ray Cap" DragInts, the "Sky Re-bake Granularity" DragFloat (Atmosphere → Advanced), and the entire "Perf Bisect (Diagnostic)" tree. The underlying options (`skyLutCacheKeySplitEnable`, the caps, the granularity, and the six `debug*` bisect toggles) stay declared and conf-tunable for future regression hunting.*

---

## Workstream — Cloud perf: voxel-grid re-bake granularity (fork — 2026-06-11)

Post-workstream follow-up at the user's request: the perf-bisect freeze of the per-frame D_sun / D_ambient bakes showed a large win with no short-term visual change — the staleness only accumulates as wind / camera / sun move. Same pattern as the sky-view fix: a cache key (`normalizeForVoxelGridKey`) quantizes the grid bake's motion inputs — wind scroll + camera position by `cloudVoxelGridRebakeGranularityKm`, sun/moon directions by the existing `skyViewRebakeGranularityDeg` — so the grids re-bake once per step of actual motion instead of every frame, staleness bounded by the step. Cloud parameter changes stay exact in the key (same-frame re-bake) and a cloud-noise re-bake force-clears the cached key. In-game validated at 0.1 km (default): ~0.7 ms saved, no visible stepping in cloud lighting or terrain shadows; 0 = legacy every-frame via conf. Note this lever lives on the cloud side — full-rate bakes were the deliberate 2026-05-19 decision against the stuttery fixed-cadence 8-frame round-robin; motion-quantized bounded-error steps passed where fixed cadence failed. Conf-only like the other workstream knobs (no UI).

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned addition.
  *`RTX_OPTION("rtx.atmosphere", float, cloudVoxelGridRebakeGranularityKm, 0.1f, …)`.*

- **`src/dxvk/rtx_render/rtx_atmosphere.h` / `rtx_atmosphere.cpp`** — fork-owned additions.
  *`normalizeForVoxelGridKey` (sky-view key base + wind/camera re-injected quantized), `m_cachedVoxelGridKey`, the dirty gate around the two grid dispatches in `computeLuts`, and the force-clear on cloud-noise re-bake.*

---

## Workstream — Per-column cloud model (column-shaping rework) (fork — 2026-06-12)

Root-cause rework of the "stacked separated layers" read that made the volumetric cloud system look like flat 2D planes. Previously EVERY vertical shaping signal — the height-LUT density envelope (which at cumulus types even baked TWO density peaks separated by a thinner neck), the coverage-threshold scale band, the anvil pow gate, the Nubis dim profile and the bottom-darkening gradient — keyed on the GLOBAL slab height fraction, so every cloud in the sky shared one vertical recipe pinned to absolute altitude (flat global base/top planes, horizontal lighting bands), while the thresholded isotropic 3D noise placed mass independently per altitude (vertically disconnected stacked puffs within a column). The rework introduces a baked 512×512 RGBA8 **cloud placement map** (R = Worley-cell cluster field, G = per-cloud top-height jitter, B = base lift; tiled at `cloudNoiseTileKm`, de-tiled by the SAME hex lattice as the 3D noise so cluster and texture travel together) and derives per-column cloud presence + local base/top from it; the sample height is re-normalized to each column's own [base, top] and ALL vertical shaping plus the Nubis lighting proxies key on that per-cloud height. The 3D noise keeps its texture/erosion role but no longer decides placement. Gated by `cloudColumnShapingEnable` (default on; legacy global-slab path preserved bit-exact for A/B).

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_placement_map_baker.comp.slang`** — NEW fork-owned file.
  *Periodic 2D Worley + value-noise FBM construction of the cluster / top-jitter / base-lift fields; integer cells-per-tile snapping from `cloudCellSizeKm` so the map tiles seamlessly.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned additions + changes.
  *New `kCloudPlacementFieldMean`, `sampleCloudPlacement` (hex-de-tiled tap; R variance-preserving blend, G/B weighted average), `CloudColumn` / `computeCloudColumn` (coverage remap of the cluster field over a `cloudColumnFeather` band; base lift; presence^`cloudColumnTopShape` × jitter top). Both density samplers hoist the hex tile, sample placement, early-out outside the column (before any 3D taps — a perf win at partial coverage), re-normalize the height fraction to the column span, and gate on per-column presence instead of weather coverage; the primary overload returns the per-cloud `shapeHeightFraction` via a new out param. Both samplers + both voxel-bake optical-depth helpers gain a `Texture2D<float4> placementMap` parameter.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_march_common.slangh`** — fork-owned change.
  *`marchCloudSlab` consumes the out-param overload and passes the per-cloud `shapeHf` to `evalNubisCubedSample`, so the dim profile / SDF proxy / bottom darkening track each cloud's own base/top instead of painting global altitude bands. The moon-shadow tap plumbs the placement map.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_height_lut_baker.comp.slang`** — fork-owned change.
  *Gains the args CB (binding 0; output moves to 1) and a second curve family for column mode: single-lobe envelope (smoothstep rise, fall reaching exactly 0 at hf=1 for cumulus) replacing the legacy trapezoid + secondary anvil bump — the two-peak profile was itself a baked-in stacked-strata structure. Legacy family preserved verbatim for the gate-off path; G channel (threshold widening near each cloud's own top) unchanged and now carries the anvil look alone in column mode.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_sky.slangh`** — fork-owned change.
  *`evalClouds` + `sampleCloudSunOpticalDepth` pass `AtmosphereCloudPlacementMap` (from common bindings) into the density sampler — the analytical fallback gets the column model automatically.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_render.comp.slang` / `cloud_secondary_lut.comp.slang`** — fork-owned additions.
  *Declare `AtmosphereCloudPlacementMap` at local binding 12 (sampled with the slot-2 cloud-noise sampler).*

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_sun_density_grid.comp.slang` / `cloud_ambient_density_grid.comp.slang`** — fork-owned additions.
  *Declare the placement map at local binding 4 and pass it through the optical-depth helpers so the baked D_sun / D_ambient grids shadow the same column shapes the view march renders.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned change.
  *Repurposes `padCloudLook2` → `cloudColumnShapingEnable`, `padCloudC2` → `cloudCellSizeKm`, `pad_cr0..2` → `cloudColumnTopVariation` / `cloudColumnTopShape` / `cloudColumnBaseVariation`, `pad_c6_0` → `cloudColumnFeather`. No CB layout change.*

- **`src/dxvk/shaders/rtx/pass/common_binding_indices.h`** — fork-owned addition (index-only).
  *Adds `BINDING_ATMOSPHERE_CLOUD_PLACEMENT_MAP = 216` + a `TEXTURE2D` entry in `COMMON_RAYTRACING_BINDINGS`.*

- **`src/dxvk/shaders/rtx/pass/common_bindings.slangh`** — fork-owned addition (index-only).
  *Declares `AtmosphereCloudPlacementMap` at the new slot for the raytrace TUs (analytical evalClouds fallback).*

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned additions.
  *`cloudColumnShapingEnable` (true), `cloudCellSizeKm` (2.0), `cloudColumnTopVariation` (0.45), `cloudColumnTopShape` (0.6), `cloudColumnBaseVariation` (0.12), `cloudColumnFeather` (0.35).*

- **`src/dxvk/rtx_render/rtx_atmosphere.h` / `rtx_atmosphere.cpp`** — fork-owned additions + changes.
  *`m_cloudPlacementMap` (512² RGBA8) + `kCloudPlacementMapSize`, `dispatchCloudPlacementMapBake` (init + live re-bake via `needsCloudPlacementRebake` on `cloudCellSizeKm` / `cloudNoiseTileKm`), height-LUT re-bake on `cloudColumnShapingEnable` flip (`m_cachedHeightLutColumnMode`; the baker now binds the args CB at 0 / output at 1), voxel-grid key force-clear when either shape input re-bakes, placement binds: slot 12 in `dispatchCloudRender` / `dispatchCloudSecondaryLut`, slot 4 in both voxel-grid bakes, `BINDING_ATMOSPHERE_CLOUD_PLACEMENT_MAP` in `bindResources`, and `getAtmosphereArgs` fills for the six repurposed fields. Shader-class parameter lists updated accordingly.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned additions.
  *`bindAtmosphereLuts` binds the placement map at the common slot; new "Cloud Columns" ImGui group (Volumetric Cloud Columns checkbox + Cloud Cell Size / Top Variation / Top Shape / Base Undulation / Edge Feather sliders); Vertical Stretch tooltip notes columns supersede it.*

---

## Workstream — Per-column underside contrast (column-shaping rev 2) (fork — 2026-06-12)

First in-game pass on the column model reported a residual "layered" read on the overcast underside. Diagnosis: the deck base's dominant light is the Nubis multi-scatter M term, which is horizontally near-uniform at production settings (sigma_ms is tiny so exp(-sigma_ms·D_sun) ≈ 1; the bottom-darkening gradient is a pure function of height) — so the underside renders as one flat-lit sheet with only fine erosion lace on top: a dark lace plane over a bright plane reads as two stacked 2D layers. Real deck undersides carry km-scale cellular relief because base brightness follows per-column water depth (thick columns occlude downwelling light, thin columns transmit it). The fix scales the bottom-darkening amount by each sample's column span (already derived by the column model), so the underside brightness is driven by the same placement field that shapes the clouds — moving with wind, matching the baked shadows.

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned changes.
  *`sampleCloudDensityTextured`'s primary overload gains an `out float columnSpan` (topFrac − baseFrac; 1.0 legacy); `evalNubisCubedSample` gains a `columnSpan` parameter and scales `cloudBottomDarkening` by `mix(1, span, cloudColumnUndersideContrast)` before building the bottom gradient. Contrast 0 or span 1 reproduce the uniform legacy darkening exactly.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_march_common.slangh`** — fork-owned change.
  *`marchCloudSlab` threads the span from the density sampler into the lighting call.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned change.
  *Repurposes `pad_c6_1` → `cloudColumnUndersideContrast`. No CB layout change.*

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned addition.
  *`RTX_OPTION("rtx.atmosphere", float, cloudColumnUndersideContrast, 0.65f, …)`.*

- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned change.
  *`getAtmosphereArgs` fills the new field (replacing the pad zero-write).*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned addition.
  *"Underside Contrast" DragFloat in the Cloud Columns ImGui group.*

(Superseded the same day by rev 3 below — the flat per-column scaling produced flat dark smudges in-game; the option and its mechanism were replaced before any release.)

---

## Workstream — Analytic downwelling light field (column-shaping rev 3) (fork — 2026-06-12)

Second in-game pass: rev 2's span-scaled darkening produced big FLAT dark smudges — user verdict "worse… changed the point at which this becomes visible, never fixing the root cause." Root cause, finally named precisely: the underside LIGHT FIELD has no 3D structure. Under a deck, T_primary ≈ 0; the dominant M term varies ≤~20-40% (sigma_ms 0.05–0.25 over D_sun ≈ 2) and that residual comes from the 8-tap voxel-grid D_sun (km-scale mush); ambient's exp(-D_ambient) saturates to ~0. With per-point illumination near-constant, the only visible pattern is opacity silhouetted against a flat backdrop — every density/shaping change just swaps the wallpaper. Rev 3 computes the downwelling light analytically from the column model: the height LUT gains a B channel holding the cumulative envelope integral from each per-cloud height to the cloud top, so `downTau = ∫envelope × columnSpan × cloudThickness × cloudDensity` is the exact macroscopic water above any sample at full resolution (no voxel grid, no taps); `exp(-cloudUndersideLightSigma × downTau)` (a diffusion-flavored sigma, far below beam extinction) lights the multi-scatter and ambient terms. Underside brightness now varies continuously with the actual water distribution — dark cores, bright thin spots, in-cell gradients. Legacy mode (column shaping off, or sigma 0) keeps the constant bottom-darkening gradient bit-for-bit.

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_height_lut_baker.comp.slang`** — fork-owned change.
  *Output RG8 → RGBA8; B = midpoint-rule ∫_hf^1 envelope(u) du of the active curve family (re-baked on mode flip as before).*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned changes.
  *New `cloudTypeProfileIntegralFromTop` (closed-form trapezoid integral, the no-LUT fallback) + `cloudHeightProfileDownIntegral` (LUT B channel / fallback). `evalNubisCubedSample`: rev 2's spanFactor block replaced by the `verticalLight` selector — column mode = analytic Beer-Lambert on downTau, legacy = the constant gradient; M multiplies `verticalLight`; ambient = `ambient_shape × verticalLight` in column mode (replacing the saturating `exp(-D_ambient)` tap — same quantity, better estimate; multiplying both would double-count).*

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_render.comp.slang` / `cloud_secondary_lut.comp.slang`** — fork-owned change.
  *`AtmosphereCloudHeightLut` declarations `Texture2D<float2>` → `Texture2D<float4>`.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned change.
  *`cloudColumnUndersideContrast` (rev 2, pad_c6_1) → `cloudUndersideLightSigma`. No CB layout change.*

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned change.
  *Drops `cloudColumnUndersideContrast`; adds `RTX_OPTION("rtx.atmosphere", float, cloudUndersideLightSigma, 0.12f, …)`.*

- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned changes.
  *Height LUT resource format R8G8 → R8G8B8A8; args fill swaps to the new option.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned change.
  *"Underside Contrast" slider replaced by "Underside Shading" (sigma). Tooltip notes it supersedes Bottom Darkening while Cloud Columns are on.*

---

## Workstream — Adaptive cloud-march sampling (fork — 2026-06-12)

Companion fix shipped with column-shaping rev 3, prompted by a community diagram suggesting the march be confined to the shell volume — it already is (marchCloudSlab intersects base + top shells), but the adjacent real defect is the FIXED step count across that volume: 32 uniform steps over a span that varies from ~4 km (zenith) to 50+ km (horizon-grazing through the curved shell) puts ~1.6 km steps against ~2 km cloud features at low elevations. The aliasing, averaged by jitter + the temporal EMA, renders as soft horizontal BANDS concentrated toward the horizon — a direct contributor to the "stacked layers" read. The march now holds a target step LENGTH: count = span / cloudViewStepKm, floored at cloudViewSamples (zenith cost unchanged) and capped at cloudViewSamplesMax; the column model's pre-tap early-outs keep added steps cheap where the slab is empty. cloudViewStepKm = 0 restores the legacy fixed count exactly.

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_march_common.slangh`** — fork-owned change.
  *`marchCloudSlab` computes `effSampleCount` from the clamped span and the step target; loop + rayFrac use it. Applies to both consumers (cloud render RT + secondary dome LUT).*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned change.
  *Repurposes `pad_cloudSunsetAmbient0` → `cloudViewStepKm` and `padStarCloud0` → `cloudViewSamplesMax`. No CB layout change.*

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned additions.
  *`cloudViewStepKm` (0.3) + `cloudViewSamplesMax` (128).*

- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned change.
  *`getAtmosphereArgs` fills both (replacing the pad zero-writes).*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned addition.
  *"March Step Size" DragFloat next to Cloud Render Scale; the cap stays conf-only.*

- **Clarity pass (same day, post-validation):** the in-game fix was user-validated with an accepted perf cost, so the controls now state it plainly: slider renamed "Cloud Sample Spacing" with a PERFORMANCE paragraph in the tooltip (cost scales up to cap/32 ≈ 4x on horizon-heavy views at defaults; how to trade it back), and the cap is exposed as a "Max Cloud Samples" DragInt (the perf governor). Option docstrings updated to match.

---

## Workstream — Night cloud-lighting knob rebase (fork — 2026-06-14)

Night clouds read too bright. The earlier suspicion (the moon-cloud directional term) was wrong: the dominant contributor is the STAR-coupling term, added uniformly and uncolored to every cloud sample (`nightLight += nightSkyColor × starBrightness × starAmbientCouplingStrength × nightFactor`), so it lifts the whole deck — and tints it blue via nightSkyColor — across the entire sky, not just near the moon. At the old default (0.03 × starBrightness 0.5 × nightSkyColor) it produced ~0.006 blue, larger and more pervasive than the moon's ~0.003 silver-lining peak. The knobs were also mis-scaled: the sub-0.01 night-radiance smallness lived inside the user-facing gains, so the only sane values were ~0.001 ("having 0.001 be the only reasonable one is painful"). Fix = rebase to O(1): the smallness moves into internal shader constants so the knobs read as a "multiple of the calibrated night level," and the star default drops to a user-tested 0.25. moonAmbientAirglow is unchanged in effect (it was never the problem). Both the RT march and the legacy analytical evalClouds get the identical edit so the day→night crossfade stays bit-matched.

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_march_common.slangh`** — fork-owned change.
  *`buildCloudShadeContext` nightLight: internal `kStarCloudCoupling = 0.008f` and `kMoonAirglowScale = 0.0015f` factored out of the star-coupling and moon-airglow terms so the user-facing gains become O(1).*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_sky.slangh`** — fork-owned change.
  *Identical `kStarCloudCoupling` / `kMoonAirglowScale` factoring in the legacy `evalClouds` nightLight block (crossfade parity with the RT path).*

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned change.
  *`starAmbientCouplingStrength` default 0.03 → 0.25; `moonAmbientAirglow` default 0.0015 → 1.0; both docstrings rewritten as "multiple of the calibrated night level."*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned change.
  *"Star Ambient Coupling" DragFloat range 0–0.1 / %.4f → 0–3 / %.2f so the O(1) default isn't clamped; tooltip updated.*

---

## Workstream — Cloud edge detail via threshold modulation (rev 4) (fork — 2026-06-14)

The additive edge-detail (rev 3) was trapped in a thin blobby rind hugging the silhouette: its containment window (`smoothstep(coverageThreshold - 0.15, coverageThreshold, density)`) keyed on the pre-detail base density and was one-sided, so detail could only displace the iso-surface from INSIDE — it could never grow billows outward. Widening the window traded the rind for a half-res smear halo (the additive positive lobe lifts a broad sub-threshold clear-sky margin into faint density on soft edges, which the half-res render + bilinear upsample smear). Rev 4 changes the mechanism: the detail tap now wobbles the COVERAGE THRESHOLD (the gate position) instead of the density field, zero-mean about kCloudNoiseFieldMean (0.4). This beats both modes at once — (a) the threshold can drop below the base value so billows grow OUTWARD past the silhouette (no rind), and (b) the gate self-localizes the effect so far-outside samples stay hard-0 (no faint margin to smear) and far-inside stay 1 (no interior holes); the modulation only bites in the gate transition band, so no explicit edge window is needed. Unlike the rejected rev-2 erosion remap it never touches the noise field — it only moves the gate, the same slide-3 anvil lever step 4a uses at low frequency. User-validated in-game ("this is perfect"). Supersedes the prior rind-vs-smear tradeoff; rev 3's window and the rejected tunable-reach knob are both gone.

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned change.
  *`sampleCloudDensityTextured` step 4b: additive `density +=` + `edgeWindow` replaced by `coverageThreshold -= cloudDetailStrength * 0.5 * (detailNoise - kCloudNoiseFieldMean)`, clamped `≥ 0`. Internal `kEdgeDetailThreshold = 0.5` (wobble→threshold scale). No new options; `cloudDetailStrength = 0` is bit-identical. Shadow sampler still skips detail.*

---

## Workstream — Artistic sunset color controls (fork — 2026-06-14)

Two artistic knobs to recover sunset warmth/saturation lost when commit `3e37062b` moved sunset reddening onto the physical Hillaire two-term LUT model: the broadband multiscatter "fill" reads pale-blue and desaturates the warm single-scatter, so the physically-correct sunset renders undersaturated. Both apply inside `evalAtmosphereRadiance` (the sky-view LUT bake integral), so the baked LUT carries them and clouds inherit the warmer ambient for free (cloud warm ambient samples the sky-view LUT). Both default to 1.0 = physical (no change). `sunsetSaturation` ramps in only near the horizon (off above ~24°, `sin 0.4`) so midday sky is untouched; `multiScatterStrength` is a global scale on the fill term. This is artistic control on top of the physical model — NOT a revert to the pre-`3e37062b` analytical air-mass reddening.

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned additions.
  *In `evalAtmosphereRadiance`'s per-sample loop: `multiScatterContrib *= args.multiScatterStrength` after the analytical/LUT blend. Before `return L`: a luma-preserving saturation boost `lerp(vec3(luma), L, satGain)` where `satGain = lerp(1, args.sunsetSaturation, 1 - smoothstep(0, 0.4, sunDir.y))`. Both no-ops at the 1.0 defaults.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned additions.
  *Repurposes the trailing `pad_cloudEdge1` slot as `multiScatterStrength` (completes the cloud-edge 16-byte row, layout unchanged), then appends a new 16-byte row `sunsetSaturation` + `pad_artistic0..2`. CB grows by one row; all prior field offsets unchanged.*

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned additions.
  *Adds `multiScatterStrength` (1.0) and `sunsetSaturation` (1.0) RTX_OPTIONs to the `rtx.atmosphere` cluster, immediately after `multiScatterPhysicalStrength`.*

- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned additions.
  *`getAtmosphereArgs()`: sets both args from RtxOptions unconditionally (after `multiScatterPhysicalStrength`), so the sky reddens with clouds disabled. `normalizeForTransmittanceMsKey()`: zeroes both (like `multiScatterPhysicalStrength`) — they feed only the sky-view bake, not the transmittance/MS LUTs, so changing them must not re-bake the heavy pair.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned additions.
  *Adds "Multiscatter Strength" (0–2) and "Sunset Saturation" (0–3) `DragFloat`s to the Atmosphere → Advanced ImGui tree, immediately after "Multiscatter Physical Strength", each with an explanatory tooltip. ~10 LOC.*

---

## Workstream — Sun-only cloud-on-terrain shadow re-architecture (fork — 2026-06-19)

Moves the cloud-on-terrain shadow from a post-denoise screen-space texture
(multiplied onto the combined direct buffer in composite) to a pre-denoise
fold-in on the SUN's radiance inside the sun NEE. This deletes the whole
geometry-blindness compensation stack (sealed-interior zenith up-ray gate,
viewmodel/decal origin correction, camera-side normal flip, dusk/dawn horizon
gate) and the screen-space plumbing it required. Several of the removed
screen-space pieces (the `PrimaryCloudShadowFactor` resource + its binding
indices/tables, debug view 878) were never previously inventoried — a prior
fridge-list gap, now closed by removal.

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned change.
  *`AtmosphereSunSample`: drops the `cloudShadowFactor` member. `sampleAtmosphereSunLight`: folds the cloud shadow onto the sun by `result.radiance *= pow(sampleCloudGroundShadow_OptionB(...), args.cloudShadowFactorStrength)` (gated on `cloudVoxelShadowsEnable`); the early issue-#37 factor computation and the long screen-space wire-in comment are gone. `sampleAtmosphereSunLightVolume`: same `pow(..., cloudShadowFactorStrength)` curve added for parity (was a plain multiply). `sampleCloudGroundShadow_OptionB_impl`: horizon gate removed (the grazing-sun OD blowup now only darkens the already-attenuated sun, never lamps).*

- **`src/dxvk/shaders/rtx/algorithm/integrator_direct.slangh`** — fork-owned change.
  *`evalAtmosphereSunNEE`: deletes the sealed-interior zenith up-ray gate, the viewmodel/decal camera-origin branches, the triangle-normal flip, and the `PrimaryCloudShadowFactor[pixelCoordinate]` write. The cloud shadow now arrives folded into `sunSample.radiance`.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned change.
  *Repurposes the trailing `pad_artistic0` slot as `cloudShadowFactorStrength` (CB layout ABI-unchanged); the knob moved here from composite_args.h.*

- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned change.
  *`getAtmosphereArgs()`: populates `args.cloudShadowFactorStrength = max(RtxOptions::cloudShadowFactorStrength(), 0)` in the cloud-shadow block.*

- **`src/dxvk/shaders/rtx/pass/composite/composite.comp.slang`** — fork-owned change.
  *Removes the post-denoise `pow(PrimaryCloudShadowFactor, cloudShadowFactorStrength)` multiply on primary direct radiance (and the texelFetch). Only an architecture comment remains.*

- **`src/dxvk/shaders/rtx/pass/composite/composite_args.h`** + **`src/dxvk/rtx_render/rtx_composite.cpp`** — fork-owned change.
  *`cloudShadowFactorStrength` CB slot renamed to reserved `pad1` (ABI-stable); the composite populate of it is removed.*

- **`src/dxvk/shaders/rtx/pass/composite/composite_bindings.slangh`** + **`composite_binding_indices.h`** + **`rtx_composite.cpp`** — index-only, fork.
  *Removes the `PrimaryCloudShadowFactor` `Texture2D` declaration, the two `TEXTURE2D(COMPOSITE_PRIMARY_CLOUD_SHADOW_FACTOR_INPUT)` parameter-list entries, and the `bindResourceView`. Binding slot 19 left reserved (number not reused).*

- **`src/dxvk/shaders/rtx/pass/integrate/integrate_direct.slangh`** + **`integrate_direct_bindings.slangh`** + **`integrate_direct_binding_indices.h`** + **`rtx_pathtracer_integrate_direct.cpp`** — index-only, fork.
  *Removes the per-frame `PrimaryCloudShadowFactor` clear, the `RWTexture2D<float> PrimaryCloudShadowFactor` declaration, the `RW_TEXTURE2D` parameter-list entry, and the `bindResourceView`. Binding slot 73 left reserved.*

- **`src/dxvk/rtx_render/rtx_resources.cpp`** + **`rtx_resources.h`** — index-only, fork.
  *Removes the `m_primaryCloudShadowFactor` `Resource` member and its `createImageResource` alloc.*

- **`src/dxvk/shaders/rtx/utility/debug_view_indices.h`** + **`debug_view.comp.slang`** + **`debug_view_binding_indices.h`** + **`rtx_debug_view.cpp`** — index-only, fork.
  *Removes debug view 878 (`DEBUG_VIEW_CLOUD_SHADOW_FACTOR_RAW`): the enum define (number burned, not reused), the case arm, the `DebugViewPrimaryCloudShadowFactor` declaration, the binding-index define (slot 38 reserved), the `TEXTURE2D` parameter-list entry, the `bindResourceView`, and the selector-list label. Views 875/877 (D_sun grid reads) stay.*

- **`src/dxvk/rtx_render/rtx_instance_manager.cpp`** + **`rtx_materials.h`** + **`src/dxvk/shaders/rtx/concept/surface/surface.h`** — index-only, fork (REVERTED).
  *Reverts the `Surface::isDecalCategory` flag (CPU set, `flags0` bit 2 pack, shader property) added 2026-06-18; bit 2 is free again. Only "removed/freed" comments remain.*

---

## Workstream — Cloud ImGui simplification (fork — 2026-06-15)

Restructured the flat ~45-control `Clouds` menu (7 non-collapsible `TextDisabled` dividers, everything always expanded) into a workflow-grouped tree of collapsible sub-nodes. No options, CB fields, or shader behavior change — pure ImGui reorganization.

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned change.
  *`Clouds` node rebuilt as: `Basic` (open by default: Coverage, Cloud Type, Density, Altitude, Depth, Color) · `Shaping ▸ {Variation, Detail & Edges, Columns}` · `Lighting` · `Wind` · `Layer 2` · `Performance` · `Horizon & Haze` (Curvature moved here from "Look"). The `Color` control is a `ColorEdit3` swatch/picker. Mode-inert controls are greyed via `ImGui::BeginDisabled`: the six Columns sliders when `cloudColumnShapingEnable` is off, `Bottom Darkening` when it is on (Underside Shading supersedes it), and the Layer 2 body when `cloudLayer2Enable` is off. The "Vertical Stretch" slider (`cloudVerticalStretch`) is removed from the menu — the option stays conf-only (experimental / superseded by Columns). All tooltips preserved.*

---

## Workstream — Remove legacy cloud paths + realistic sun-gated underside darkening (fork — 2026-06-19)

Two dead cloud code paths were removed and the column model made the only path: (1) the **legacy analytical `evalClouds` view-march** in `atmosphere_sky.slangh` (already compile-gated out behind `ATMOSPHERE_CLOUD_VIEW_MARCH`, used as a runtime A/B fallback that was never the validated default — clouds come from the `cloud_render` compute pass on primary rays and the secondary dome LUT on indirect/PSR/reflection rays), and (2) the **legacy global-slab shaping** (`cloudColumnShapingEnable == false`): the trapezoid+anvil height-profile curve family, the gate-off branches in the density/bake samplers, and the constant bottom-darkening gradient. With the legacy gradient gone, **bottom darkening was reworked** into the column model as a physically-motivated underside light field driven by cloud density *and* sun position: the analytic downwelling `exp(-cloudUndersideLightSigma × downTau)` (density / overlying-water term, unchanged shape) now scales by `cloudBottomDarkening` (overall strength) × `smoothstep(0, 0.35, sunElev)` so the darkening is strongest at high sun and **fades out toward the horizon** — the low sun rakes under the deck and the warm ambient / silver-lining terms light the bases (classic lit-orange sunset undersides). `cloudBottomDarkening` default 0.55 → 1.0 so the high-sun look matches the validated rev-3 column underside bit-for-bit; only low-sun behavior is new. The legacy ambient `exp(-D_ambient)` tap is gone (the analytic `verticalLight` replaces it), so `sampleDAmbient` is no longer called from `evalNubisCubedSample`.

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_sky.slangh`** — fork-owned change.
  *Deleted `evalClouds` (the whole `#ifdef ATMOSPHERE_CLOUD_VIEW_MARCH` block) and its only consumer `sampleCloudSunOpticalDepth`. The cloud-source `else` branch in `evalSkyRadiance` (both RT + secondary-LUT off) is now an unconditional transparent `vec4(0)` fallback, no `#ifdef`. Stale `evalClouds` comment references updated.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_height_lut_baker.comp.slang`** — fork-owned change.
  *Deleted the legacy `densityEnvelope` (trapezoid+anvil) R-curve and the `cloudColumnShapingEnable` mode select; `densityEnvelopeColumn` is now used unconditionally for both R and the B integral. `coverageThresholdScale` (G) unchanged.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned changes.
  *Dropped the `cloudColumnShapingEnable` gate in `sampleCloudDensityTextured` and the bake-mirror sampler (column derivation now unconditional). `evalNubisCubedSample`: `verticalLight` reworked from the column/legacy `if-else` to the single realistic sun-gated form (`mix(1, exp(-sigma·downTau), cloudBottomDarkening × smoothstep(0, kUndersideSunFadeElev, sunElev))`, `kUndersideSunFadeElev = 0.35`); ambient term is now `ambient_shape × verticalLight` unconditionally; `sampleDAmbient` call removed (unused).*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned change.
  *`cloudColumnShapingEnable` → `padCloudLook2` and `cloudBottomDarkeningHeight` → `pad_cloudVoxel1` (both reverted to pad placeholders). No CB layout change.*

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned change.
  *Removed `cloudColumnShapingEnable` and `cloudBottomDarkeningHeight` RTX_OPTIONs. `cloudBottomDarkening` default 0.55 → 1.0, description reworked (sun-gated strength). `cloudUndersideLightSigma` description reworked (no more legacy-gradient / column-mode wording).*

- **`src/dxvk/rtx_render/rtx_atmosphere.cpp` / `.h`** — fork-owned changes.
  *Removed the `m_cachedHeightLutColumnMode` member + the height-LUT flip-rebake in `computeLuts` (the LUT bakes one curve family once at init). Dropped the `cloudColumnShapingEnable` / `cloudBottomDarkeningHeight` args fills. Updated the bake doc comments.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned change.
  *Removed the "Volumetric Cloud Columns" checkbox and the `BeginDisabled`/`EndDisabled` gating around the Columns sliders and Bottom Darkening (all always enabled now). Reworked the Bottom Darkening + Underside Shading tooltips (strength × shape; sunset fade).*

- **`RtxOptions.md`** — hand-edited to match (removed the two options, updated `cloudBottomDarkening` / `cloudUndersideLightSigma`); regenerate in-app for canonical form.

---

## Workstream — Cloud sky-dome ambient fill (clouds reflect the sky) (fork — 2026-06-19)

First half of the sky↔cloud color-coupling work. Symptom: under a bright blue daytime sky the cloud undersides read dark / gloomy, and clouds never take on the surrounding sky's color, because the cloud ambient samples the sky-view LUT in only TWO directions — toward the sun (warm) and the anti-sun horizon (cool) — and `buildCloudShadeContext` deliberately skips the overhead/zenith sky (the old comment feared zenith would blue-tint sunset cumulus). So the large bright sky dome that lights real cloud bottoms from below/around is never sampled. Fix: add a zenith sky-view sample and a new ambient term, the "sky-dome fill," that the underside picks up WITHOUT the bottom-darkening attenuation (that skylight reaches the base from below/around, not filtered down through the overlying water). It reads the zenith color, so a bright daytime dome lifts gloomy undersides and tints them with the actual sky; at sunset the zenith sample is dim so the term self-fades and the warm top-down ambient carries the look (the validated sunset is preserved bit-for-bit at fill 0, and nearly so at the 0.5 default since zenith→0 there). The companion sky←clouds half (cloud radiance bleeding into the visible sky) is still pending.

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_march_common.slangh`** — fork-owned changes.
  *`CloudShadeContext` gains `skyRadianceZenith`; `buildCloudShadeContext` samples `sampleSkyAmbientForVolume(vec3(0,1,0), …) * dayFactor` (same cloud-occlusion + day gate as warm/cool) and assigns it; the `evalNubisCubedSample` call passes it through.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned change.
  *`evalNubisCubedSample` gains a `skyRadianceZenith` param; ambient split into `topAmbient = ambient_shape × verticalLight × skyRadiance` (legacy, attenuated) + `domeFill = ambient_shape × cloudSkyAmbientFill × skyRadianceZenith` (NOT attenuated by verticalLight); `result.ambient = topAmbient + domeFill`.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned change.
  *`pad_cloudVoxel1` (the freed cloudBottomDarkeningHeight slot) → `cloudSkyAmbientFill`. No CB layout change.*

- **`src/dxvk/rtx_render/rtx_options.h` / `rtx_atmosphere.cpp`** — fork-owned changes.
  *New `RTX_OPTION(float, cloudSkyAmbientFill, 0.5f, …)`; `getAtmosphereArgs` fills `args.cloudSkyAmbientFill`.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned change.
  *"Sky Fill" DragFloat added to the Clouds → Lighting tree after Bottom Darkening, with tooltip.*

- **`RtxOptions.md`** — hand-edited to add `cloudSkyAmbientFill`; regenerate in-app for canonical form.

---

## Workstream — Sky-reflects-clouds bleed (sky picks up cloud color) (fork — 2026-06-19)

Second half of the sky↔cloud color coupling. Symptom: the visible sky and the cloud layer are composited independently (`radiance = mix(sky, cloud.rgb, cloud.a)` in `evalSkyRadiance`), so a colored deck never tints the clear sky — leaving vivid blue gaps between orange sunset clouds (and a too-clean sky around grey overcast). Fix: before the cloud composite, add a fraction of the LOCAL cloud radiance to `radiance` as colored inscatter. The source is the secondary cloud dome LUT (`AtmosphereCloudSecondaryLut`) sampled in the view direction — it is low-res (256×128), hence inherently smooth, giving a neighborhood-averaged cloud color per direction (bright/colored next to a cloud, ~0 in open sky far from any) with no extra blur pass. Because it is added BEFORE the composite, the composite's own `(1 - cloudAlpha)` factor keeps the bleed in the visible-sky fraction and out of opaque cloud cores. Gated on the secondary LUT being baked (default on) and `cloudSkyBleedStrength > 0`. Pairs with the clouds←sky "sky-dome fill" (Sky Fill); together the two close the loop so clouds and sky share color in both directions at all times of day.

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_sky.slangh`** — fork-owned change.
  *In `evalSkyRadiance`, after the cloud-source selection and before the temporal-smoothing/composite: sample the secondary dome LUT at `cloudDomeDirToUv(viewDirYUp)` and `radiance += cloudSkyBleedStrength * bleedCloud.rgb`.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned change.
  *`padCloudLook2` (the freed cloudColumnShapingEnable slot) → `cloudSkyBleedStrength`. No CB layout change.*

- **`src/dxvk/rtx_render/rtx_options.h` / `rtx_atmosphere.cpp`** — fork-owned changes.
  *New `RTX_OPTION(float, cloudSkyBleedStrength, 0.3f, …)`; `getAtmosphereArgs` fills `args.cloudSkyBleedStrength`.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned change.
  *"Sky Cloud Bleed" DragFloat added to the Clouds → Lighting tree after "Sky Fill", with tooltip.*

- **`RtxOptions.md`** — hand-edited to add `cloudSkyBleedStrength`; regenerate in-app for canonical form.

---

## Workstream — Sky color correctness: physical multiscatter default (fork — 2026-06-19)

The clear sky read wrong vs reality at all times of day: over-saturated/wrong blue hue, no warm horizon, and a flat zenith→horizon gradient. Two Opus research passes converged on the cause: the DEFAULT multiscatter path was the analytical inline fit (`multiScatterPhysicalStrength` defaulted to 0), which is a flat, isotropic (no phase, no transmittance), strongly blue-biased fill added uniformly in every direction — it raises the floor everywhere (flattening the gradient) and desaturates the warm horizon. The `sunsetSaturation` knob had been added purely to band-aid this (a desaturation mask is itself the tell the base output was wrong). Fix: make the physical Hillaire multiscatter LUT the default and retire the band-aid. Also fixed a real dimensional bug in the analytical path (kept for A/B): its `computeGroundReflectionAnalytical` / `computeAirMultiscatteringAnalytical` baked `sunIlluminance` in, then the caller multiplied by `sunIlluminance` again — `sunIlluminance²` (≈225× at the default 15) — whereas the LUT-baker twins correctly omit it. Separately, the user's per-game config had non-physical coefficient overrides (Mie 4× low, Rayleigh ~35% low, Mie g 0.99); the code defaults are already the canonical Bruneton/Hillaire values, so those were corrected in the game's rtx.conf (not in-repo).

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned changes.
  *`multiScatterPhysicalStrength` default 0.0 → 1.0 (physical LUT is now the default multiscatter); `sunsetSaturation` default 1.0 → (briefly 0.5) → 1.0 (band-aid retired now that the base is correct). Descriptions rewritten.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned change.
  *`computeGroundReflectionAnalytical` / `computeAirMultiscatteringAnalytical` no longer bake in `sunIlluminance` (the caller applies it once, matching the physical LUT twins and `contribLut`); fixes the `sunIlluminance²` double-count on the analytical A/B path. Call-site comment added so it isn't re-introduced.*

---

## Workstream — Sky-reflects-clouds bleed: mip-blur fix (fork — 2026-06-19)

The first cut of the sky<-clouds bleed sampled the 256x128 secondary cloud dome LUT at mip 0 and added it to the sky. Result: cloud EDGES read pixelated/faceted (a single bilinear tap of a near-binary low-res signal steps across the LUT's coarse texels at silhouettes, and the coarse dome silhouette is misaligned with the sharp screen-space cloud RT), and the sky "barely changed color" (a true gap samples ~0 cloud in its exact direction — no neighborhood spread). An Opus debug pass identified both and prescribed a pre-blurred coarse-mip source. Fix: give the secondary LUT a mip chain, regenerate it (Gaussian) each frame after the bake, and sample a coarse mip (mip 4 = 16x8) for the bleed — one tap = a wide neighborhood blur, which removes the facets AND spreads cloud color smoothly into the gaps next to clouds. Re-enabled by default (`cloudSkyBleedStrength` 0 -> 0.15).

- **`src/dxvk/rtx_render/rtx_atmosphere.h` / `rtx_atmosphere.cpp`** — fork-owned changes.
  *`m_cloudSecondaryLut` becomes an `RtxMipmap::Resource` (6 mips, 256x128 -> 8x4) via `RtxMipmap::createResource`; the bake binds `.views[0]` (mip 0) for the storage write; after the dispatch a write->read barrier + `RtxMipmap::updateMipmap(..., Gaussian)` fills mips 1..5 (ctx is the RtxContext, cast from the DxvkContext param). Added `#include "rtx_mipmap.h"`.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned change.
  *The shared sky-view sampler gains `mipmapLodMax = VK_LOD_CLAMP_NONE` so explicit-LOD mip sampling works (harmless for the mip-less sky-view LUT / cloud RT).*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_sky.slangh`** — fork-owned change.
  *The bleed samples `AtmosphereCloudSecondaryLut` at `kBleedMip = 4.0` instead of 0.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned change.
  *`cloudDomeDirToUv` wraps u with `frac()` so dome sampling no longer depends on sampler address mode (the mip-gen uses a CLAMP sampler).*

- **`src/dxvk/rtx_render/rtx_options.h`** — `cloudSkyBleedStrength` default 0.0 -> 0.15 (re-enabled).

---

## Workstream — Cloud direct energy conservation + multiscatter sunset gather fix (fork — 2026-06-19)

Two physically-motivated lighting corrections, both addressing "lit things out-bright/mis-color the sky they composite against." WIP: shipped + deployed for in-game look validation, NOT yet eyeballed — values/approach may change after the look-check.

(1) **Cloud direct dual-lobe energy conservation.** The Nubis direct term summed two full-amplitude phase lobes (`L_direct = T_primary*HG1 + M*HG2`); HG1 and HG2 each integrate to 1 over the sphere, so the in-scatter phase integrated to up to (1+M) ≈ 2 — the cloud scattered up to ~2× the energy one scattering event can redistribute, worst at the sunlit edge where both lobes fire at once, so lit clouds out-brightened the physical sky LUT. Reformulated as an energy-conserving convex blend (phase integral 1), lerped from the legacy additive sum by a strength knob for in-game A/B. Brings the sun path onto the bounded-energy footing the moon path already had.

(2) **Multiscatter sunset gather fix.** The MS LUT gather in `computeMultiscattering` swept zenith 0..π at a SINGLE azimuth (`rayDir = vec3(sinZenith, cosZenith, 0)`, x-y plane), but the sun lies in the y-z plane (x=0), so every gather ray sat 90° from the sun and the integral never sampled the sun-ward sky. At sunset it gathered only the blue sky orthogonal to the sun and missed the warm reddened horizon → pale-blue MS fill (sky too blue, warm band couldn't climb, clouds inherited the cold ambient via the sky-view LUT). Fixed to a proper 2D sphere quadrature (8 zenith × 8 azimuth, same 64 samples); the sinZenith Jacobian + 2π/N normalization are kept so a uniform integrand changes <1% — corrects the integral's direction/color without shifting the tuned brightness. Knob-free. (A first attempt — a `multiScatterReddening` tint knob — was reverted as a hack once the gather bug was found.)

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned change.
  *In `evalNubisCubedSample`, the direct term `T_primary*HG1 + M*HG2` becomes `A*(1 - s*w) + B*(1 - s*(1-w))` with `A = T_primary*HG1`, `B = M*HG2`, `s = cloudEnergyConserve`, `w = cloudMsLobeWeight`. s=0 reproduces the legacy additive sum byte-for-byte.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/multiscattering_lut.comp.slang`** — fork-owned change.
  *`computeMultiscattering` gather loop: single-azimuth 64× zenith sweep → nested 8 zenith × 8 azimuth full-sphere quadrature; `rayDir` gains an azimuth term. Jacobian + 2π/N normalization unchanged (uniform-integrand magnitude preserved to <1%).*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned change.
  *`pad_artistic1` / `pad_artistic2` (trailing slots of the `cloudShadowFactorStrength` 16-byte row) → `cloudEnergyConserve` / `cloudMsLobeWeight`. No CB layout change.*

- **`src/dxvk/rtx_render/rtx_options.h` / `rtx_atmosphere.cpp`** — fork-owned changes.
  *New `RTX_OPTION(float, cloudEnergyConserve, 1.0f, …)` and `RTX_OPTION(float, cloudMsLobeWeight, 0.5f, …)` after `cloudPhaseG2`; `getAtmosphereArgs` fills both `args.*`.*

- **`RtxOptions.md`** — PENDING: `cloudEnergyConserve` / `cloudMsLobeWeight` not yet added; regenerate in-app for canonical form.

---

## Workstream — Sunset "neon cloud" fixes: cloud sun-transmittance Mie + dayFactor rolloff (fork — 2026-06-20)

Final two corrections that fixed the right-before-sunset "neon orange clouds" (flat, too bright, too saturated; worst at sun elevation below sun.y ~0.2, recovering once the sun is well below the horizon). Both validated in-game.

(1) **Mie/aerosol extinction added to the cloud + moon sun-direction transmittance.** `getAtmosphericTransmittanceForDir` (the analytical Kasten-Young model that lights clouds + moons; the sky body uses the transmittance LUT) was composing extinction as Rayleigh + ozone only — the Mie term was MISSING, while the LUT (transmittance_lut.comp.slang) correctly includes Rayleigh + Mie + ozone. At a low sun the omitted grey aerosol extinction left the cloud beam under-extinguished, so cloud direct light stayed a bright, fully-saturated red while the LUT-lit sky looked correct. Added the Mie slant optical depth (`mieScaleHeight * airMass`), mirroring the LUT's extinction composition — physically the aerosol extinction that dims a hazy setting sun, and it realigns the cloud/moon sun color with the sky.

(2) **Cloud daytime-lighting rolloff widened through the sunset approach.** `buildCloudShadeContext`'s `dayFactor = smoothstep(-0.05, 0.02, sun.y)` held the cloud's sun + sky-ambient lighting at FULL brightness until the geometric horizon, then crashed to night — so through the whole low-sun band (where airTrans + the sky-view LUT are most saturated) the clouds were lit at full strength = full-bright, full-saturated, flat orange. Widened the upper bound to 0.20 (~11.5°) so the daytime cloud lighting dims PROGRESSIVELY as the sun descends (physically: the scene darkens through sunset); the reddest light is no longer the brightest. Above 0.20 daytime is unchanged (golden hour untouched); horizon dayFactor ~0.2, reaching 0 at -0.05 where the moon/night terms take over.

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned change.
  *`getAtmosphericTransmittanceForDir`: add `- args.mieScattering * (args.mieScaleHeight * airMass)` to the transmittance exponent (Rayleigh + Mie + ozone, matching the LUT bake). Stale "mirrors lines 308-328" comment corrected.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_march_common.slangh`** — fork-owned change.
  *`buildCloudShadeContext`: `dayFactor` smoothstep upper bound 0.02 → 0.20 for a progressive sunset-approach dim of all sun-derived cloud lighting.*

---

## Workstream — Layer-2 "echo deck" rework (fork — 2026-06-21)

Replaced the old layer-2 path (a fully-independent second `marchCloudSlab` pass that inherited layer 1's ~32-step floor and tapped the layer-1-only `D_sun` grid for a decorrelated cloudscape, so every deck sample read stale residual) with a lean **echo deck**: the *same* per-column cloud-slab density model at a higher, gapped altitude, marched cheaply. Perf comes from a low, **user-adjustable** step budget (the dominant lever — deck cost is ~linear in step count) and an analytic sun-shadow proxy in place of the stale `D_sun` grid tap. Variety ("varied echo") comes from the deck's own coverage/type means + `cloudLayer2NoiseSeed`, which decorrelates its control field from layer 1 while sharing the same field machinery. The column-placement model is deliberately KEPT — it provides the per-sample early-out that refunds the placement tap, contrary to the earlier (now-corrected) analysis. The deck DOES take moonlight (same Beer-Lambert shadow + Wrenninge phase as layer 1, deck-aware self-shadow tap) so the two decks read consistently at night. 3 `cloudLayer2*` options were added (2 step-budget + 1 color) and 1 removed (`cloudLayer2CoverageSpread`); the rest are unchanged. Validated in-game 2026-06-21 (works after the CB-alignment fix below); same-day follow-ups: density/thickness/type decoupled from the main layer, the deck's coverage "tiling" fixed (coverage spread forced uniform + option removed), and an independent deck color.

**Same-day follow-ups (2026-06-21):**
- **Density/thickness/type decoupling.** The deck ran on a copy of the main `args`, so main-layer sliders (most visibly `cloudDensity`) bled into the deck. `marchCloudLayers` now overrides on `args2`: `cloudDensity` → fixed `kEchoDeckExtinction = 1.8` (= cloudDensity's default, so the deck decouples from the live slider with no look change at default), `cloudThickness` → `cloudLayer2Thickness`, `cloudTypeMean` → `cloudLayer2TypeMean` (the shared lighting — `sampleCloudSdf`, `sampleDimProfile`, underside down-integral — reads these directly rather than via the deck slab params). `marchEchoDeck` also rebuilds `ctx.moonBaseSigma` from the decoupled density. Coverage was already deck-specific. Audit conclusion: all OTHER look knobs (phase, multi-scatter, detail, bottom-darkening, edge, anvil, etc.) stay shared by "same slab" design — only color was split out (below).
- **Coverage "tiling" fix.** The single-octave Worley coverage field, amplified by the spread slider, read as a fine tiled/stippled texture at the deck's distance (~3 km cells subtend a tiny angle). A de-tile attempt (domain-warp + extra octave) made it worse (small/smudgy), so it was reverted; instead `marchCloudLayers` forces the deck's `cloudCoverageSpread` to 0 — coverage collapses to a uniform `cloudLayer2CoverageMean` sheet, no field, no tiling. `cloudLayer2CoverageSpread` was removed as an option (its arg slot kept as `pad_cloudLayer2CoverageSpread` to preserve CB layout). Type spread is still available.
- **Independent deck color.** New `cloudLayer2Color` (Vector3, defaults to cloudColor's near-white). `marchCloudLayers` overrides `args2.cloudColor`. The one look knob split from layer 1, per user request.
- **GPU-hang backstop.** `marchEchoDeck` hard-caps the step count at `kEchoStepHardCap = 256` so a future CB misread can't escalate to a device hang.

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned change.
  *`evalNubisCubedSample` gains a trailing `float dSunOverride` param: `< 0` samples the `D_sun` voxel grid (production path, byte-identical to before); `>= 0` is used verbatim and the grid tap is skipped. Only the echo-deck march passes `>= 0`.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/cloud_march_common.slangh`** — fork-owned change.
  *New `marchEchoDeck` (lean second-deck march: `cloudLayer2StepFloor`/`cloudLayer2StepMax` step budget — default 8/16 — honoring `cloudViewStepKm`; analytic `dSunProxy = (1 - shapeHf) × thickness × densityScale`; otherwise identical density model + control fields + per-sample moon block + Beer-Lambert/aerial as `marchCloudSlab`). The moon shadow uses a new slab-parametric `sampleCloudSunOpticalDepth_localSlab` so the deck self-shadows against ITS altitude band (the args-default helper keys on layer 1's slab → zero density at the deck = unshadowed). `marchCloudLayers`' layer-2 branch calls `marchEchoDeck` instead of a second `marchCloudSlab` (the old moon-coefficient zeroing is gone — the deck wants moonlight). `marchCloudSlab`'s `evalNubisCubedSample` call passes `dSunOverride = -1.0` (grid path). Both consumers (view RT + secondary dome LUT) inherit via `marchCloudLayers`.*

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned change.
  *Adds a 16-byte (vec4) tail block: `cloudLayer2StepFloor`, `cloudLayer2StepMax`, + `pad_cloudLayer2Step0/1`. No free pad slots remained, so the CB grows — but it MUST grow by a whole vec4 or `sizeof(AtmosphereArgs)` stops being 16-byte aligned and the whole-struct `updateBuffer` corrupts the cbuffer (the new tail fields read garbage → marchEchoDeck's step count blows up → GPU hang → solid black when layer 2 is on). A first draft appended bare scalars (8 bytes) and hit exactly that; the two pad words fix it. Future additions should consume the pads first.*

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned change.
  *Added `cloudLayer2StepFloor` (uint32, 8) and `cloudLayer2StepMax` (uint32, 16). Rewrote `cloudLayer2Enable` / `cloudLayer2Altitude` / `cloudLayer2Thickness` / `cloudLayer2DensityScale` help strings to describe the echo deck + inter-deck gap; corrected three stale default-value mentions (7.5 km / 0.5 km / 0.30 vs the actual 5.5 / 2.0 / 0.65). No option removed or renamed.*

- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned change.
  *Packs `args.cloudLayer2StepFloor` / `cloudLayer2StepMax` from the new RtxOptions alongside the existing `cloudLayer2*` packing.*

- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned change.
  *Adds "Layer 2 Step Floor" / "Layer 2 Max Steps" `DragInt` widgets to the Layer 2 imgui node; refreshed the Layer 2 Density tooltip off the old cirrus framing.*

- **`RtxOptions.md`** — regenerated; reflects the 3 new options (`cloudLayer2StepFloor`, `cloudLayer2StepMax`, `cloudLayer2Color`) and the removed `cloudLayer2CoverageSpread`.

  (The `atmosphere_args.h` / `rtx_options.h` / `rtx_atmosphere.cpp` / `rtx_fork_atmosphere.cpp` bullets above also cover the follow-up `cloudLayer2Color` field/option/packing/picker; the args color field is a second vec4-aligned tail block — `vec3 cloudLayer2Color` + `pad_cloudLayer2Color0`. The de-tile attempt was reverted: the deck's coverage tiling is instead handled by forcing `cloudCoverageSpread` to 0 and removing the `cloudLayer2CoverageSpread` option, see below.)

---

## Workstream — Dead-code removal + stale-doc cleanup (fork — 2026-06-21)

Audit-driven cleanup of the sky/cloud system; no behavior change. Removed two unused shader helpers — `evalSunDisk` (`atmosphere_sky.slangh`) and `uvToSkyViewLutParams` (`atmosphere_common.slangh`); removed the orphaned `rtx.atmosphere.sunDisc` option (never consumed — the sun disc renders via the sun-as-distant-light / NEE path) along with its `RemixSkyAPI.md` row and a `RemixApiChangelog.md` "Removed" entry. Demoted the three dead `cloudVoxelGrid*` dirty/offset args fields to `pad_*` reserves (CPU zero-writes dropped) — kept as pads so the CB layout is unchanged. Corrected ~30 stale comments + 3 user-facing RTX_OPTION descriptions that referenced the long-removed analytical `evalClouds` path / "no consumer in this commit" voxel grids as if current; each description was re-verified against its consumer before rewording. `RtxOptions.md` regenerated.

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned change.
  *Removed the `sunDisc` RTX_OPTION; reworded `cloudRenderRTEnable` / `cloudSecondaryLutEnable` / `cloudHeightLutEnable` descriptions to drop the removed-evalClouds references (off = cloudless; secondary rays get clouds from the dome LUT). The "original 17 atmosphere options" historical list earlier in this file still names `sunDisc` — left as history.*
- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_sky.slangh`, `atmosphere_common.slangh`** — fork-owned change.
  *Deleted `evalSunDisk` / `uvToSkyViewLutParams`; comment fixes for removed-evalClouds references.*
- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned change.
  *`cloudVoxelGridFrameOffset` / `…SunDirty` / `…AmbientDirty` → `pad_*`; comment fixes.*
- **Comment-only doc-rot fixes** in `cloud_render.comp.slang`, `cloud_secondary_lut.comp.slang`, `cloud_march_common.slangh`, `common_bindings.slangh`, `common_binding_indices.h`, `debug_view.comp.slang`, `rtx_atmosphere.cpp/.h`, `rtx_debug_view.cpp`. No functional change.
- **`RtxOptions.md` / `docs/RemixSkyAPI.md` / `docs/RemixApiChangelog.md`** — regenerated / `sunDisc` row removed / `[0.4.3] Removed` entry added.

---

## Workstream — Cloud drift overhaul: field evolution (fork — 2026-06-21)

Replaces the artificial-looking cloud "drift." The old motion had two pieces: rigid wind advection (fine) and a separate global weather-parameter `drift` (`rtx_fork_weather.cpp`) that modulated 9 cloud scalars with a sum-of-sines whose **fast layer had base period exactly 30 s** — its dominant inner sine was a clean 30 s beat, and all 9 fields shared one phase clock, so the whole sky visibly "breathed" in lockstep every ~30 s. The fix moves shape change **into the field** (differential advection, the Nubis/Decima trick) and demotes the global scalar drift to genuinely slow weather-scale change. Two new view-path levers, both reverting to legacy rigid behavior at speed 0: **morph** (a slow 3D scroll of the base noise sample position, Y-dominant so it samples a continuously different, tile-wrapping slice → formations form/dissolve in place, also breaking the 12 km wind tile-repeat) and **edge boil** (an independent faster scroll on the edge-detail tap → billows churn at the silhouette). No constant-buffer growth — four reserve pad slots are reused.

- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned change.
  *Reuses the three contiguous `pad_cloudVoxelGrid{FrameOffset,SunDirty,AmbientDirty}` slots (from the 2026-06-21 dead-code pass) as `cloudEvolutionOffsetX/Y/Z` (float; the two former `uint` slots retype to float), and the `pad_cloudLayer2CoverageSpread` slot as `cloudBoilPhase`. CB layout byte-identical.*
- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned change.
  *In `sampleCloudDensityTextured` (view path only): builds `noiseSamplePos = perturbed + cloudEvolutionOffset` and a parallel `noiseTexcoord`, used for the base 3D tap (incl. hex variants) and the step-3b footprint slice ONLY — placement map, hex lattice, column model, and height fraction stay on the real `perturbed`/`texcoord` so cluster location and altitude are unaffected. Step 4b detail tap adds `cloudBoilPhase * kBoilDir` (fixed off-axis unit dir) to its sample position. The shadow sampler (`sampleCloudDensityForShadow`) is intentionally NOT evolved in v1 (baked grids keep describing the bulk field).*
- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned change.
  *In `getAtmosphereArgs()`: accumulates the morph offset from `cloudEvolutionSpeed` / `cloudEvolutionVerticalBias` (Y-dominant + diagonal XZ remainder) and the boil phase from `cloudBoilSpeed`, both × `timeSeconds` (mirrors the existing `cloudWindOffset` pattern). Zeros the four new per-frame fields in `normalizeForSkyLutCache` so the sky-LUT memcmp gate doesn't fire every frame.*
- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned change.
  *Adds `cloudEvolutionSpeed` (0.0015), `cloudBoilSpeed` (0.004), `cloudEvolutionVerticalBias` (0.8) to the `rtx.atmosphere` cluster.*
- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned change.
  *Adds an "Evolution" subtree (Morph Speed / Morph Vertical Bias / Edge Boil Speed) to Atmosphere → Clouds, next to "Wind". Named distinctly from the weather "Cloud Drift" subtree to avoid confusion.*
- **`src/dxvk/rtx_render/rtx_fork_weather.cpp`** — fork-owned change.
  *De-pulse: `driftOffsetForField` drops the fast 30 s layer (slow-only now); removed `kDriftFastPeriodSec`. `kDriftTable` trimmed 9 → 3 (kept `cloudCoverageMean`, `cloudWindSpeed`, `cloudWindDirection` — the weather-scale fields field evolution does not reproduce; dropped the shape-ish density/thickness/type/anvil/coverage-spread entries now owned by field evolution). `static_assert` updated to 3.*
- **`RtxOptions.md`** — REGEN PENDING (run Remix with `DXVK_DOCUMENTATION_WRITE_RTX_OPTIONS_MD=1`): will add the 3 new options.

---

## Workstream — Cloud motion unification + advection integrator fix (fork — 2026-06-21)

Folds the three cloud-motion sources (wind advection, field-evolution morph/boil, and the slow weather-parameter drift) into ONE per-frame integrator, fixing a latent bug. The wind/evolution/boil offsets were computed stateless as `instantaneousSpeed * timeSeconds` inside the `const` `getAtmosphereArgs` (called ~12×/frame, hence stateless). That is only correct while wind is constant — but the weather blender writes drift-modulated `cloudWindSpeed`/`cloudWindDirection` into the Derived config layer, so each frame the *entire accumulated offset* re-scaled (speed change) or rotated about the origin (direction change), snapping the whole field. Replaced with persistent accumulators advanced once per frame by `offset += velocity * dt`, so a varying wind eases the field instead of jumping. Rates stay independent (no cross-coupling). No CB/shader change — only how the CPU fills the existing `cloudWindOffset` / `cloudEvolutionOffset*` / `cloudBoilPhase` fields. Also: merged the "Wind" + "Evolution" dev-menu subtrees into one "Cloud Motion" tree, and bumped `cloudLayer2StepMax` 16 → 32.

- **`src/dxvk/rtx_render/rtx_atmosphere.h`** — fork-owned change.
  *Adds `void advanceCloudMotion(float dt)` (public) and three persistent accumulator members `m_cloudAdvectOffset` (Vector2) / `m_cloudEvolutionOffset` (Vector3) / `m_cloudBoilPhase` (float), next to the other per-frame-pushed members.*
- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned change.
  *Implements `advanceCloudMotion` (integrates wind from drift-modulated `cloudWindSpeed`/`cloudWindDirection`, plus morph + boil, `+= vel*dt`, with a `dt<=0` pause guard). `getAtmosphereArgs` now reads the three members instead of computing `speed*timeSeconds`. `normalizeForSkyLutCache` already zeros those per-frame fields (unchanged).*
- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned change.
  *Calls `ctx.m_atmosphere->advanceCloudMotion(GlobalTime::get().deltaTime())` once per frame in `updateAtmosphereConstants` (Numos block, alongside the camera-basis push); adds `#include "../util/util_global_time.h"`. Merges the Wind + Evolution UI subtrees into one "Cloud Motion" tree with a pointer to the Weather → Cloud Drift slow modulator.*
- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned change.
  *`cloudLayer2StepMax` default 16 → 32 (per request — layer-2 echo-deck step budget to match layer 1's `cloudViewSamples = 32`).*
- **`RtxOptions.md`** — REGEN PENDING: reflects the `cloudLayer2StepMax` default change.

---

## Workstream — Rename weather "Cloud Drift" → "Weather Variation" (fork — 2026-06-21, UI clarity)

Pure dev-menu/label rename to stop the word collision with the new Atmosphere → Clouds → "Cloud Motion" tree. The weather-parameter wander (slow, preset-driven modulation of coverage + wind) is a different layer from the per-frame field motion; calling both "drift"/"motion" in adjacent panels was confusing. The `__weather.drift_*` GameStateStore API keys and the internal `m_drift*` members are UNCHANGED — only user-facing strings moved.

- **`src/dxvk/rtx_render/rtx_fork_weather.cpp`** — fork-owned change.
  *Subtree "Cloud Drift" → "Weather Variation"; slider/button labels "Drift …" → "Variation …"; tooltips note the unchanged `__weather.drift_*` keys; added a one-line pointer to Cloud Motion.*
- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned change.
  *Cloud Motion tree's cross-reference text updated to point at "Weather → Weather Variation".*
- **`docs/CloudSystem.md`, `docs/integrators/weather-presets.md`** — updated the dev-menu sub-tree name references (keys unchanged).

---

## Workstream — Remove the bespoke atmosphere sun/moon NEE (fork — 2026-06-21)

Graduates the "sun/moon as real distant lights" work from an experiment-with-fallback to the **sole** sun/moon path in Numos, deleting the now-redundant bespoke atmosphere NEE that was previously only runtime-gated off. Supersedes the earlier `evalAtmosphere*NEE*` touchpoint entries above. Runtime-neutral at default settings (the bespoke path was already gated off whenever `useDirectionalLights` was on, the default). Cloud-on-terrain shadows are unaffected — they fold per-pixel onto the real sun distant light in the integrators (gated on `atmosphereCloudShadowed` + `cloudVoxelShadowsEnable`); the volumetric fold via `sampleAtmosphereSunLightVolume` is unchanged.

- **`src/dxvk/shaders/rtx/algorithm/integrator_direct.slangh`** — fork-owned change.
  *Deleted `evalAtmosphereSunNEE` + `evalAtmosphereMoonNEE` and their gated call block; the direct path now lights the atmosphere sun/moon purely via the standard NEE on the injected distant lights.*
- **`src/dxvk/shaders/rtx/algorithm/integrator_indirect.slangh`** — fork-owned change.
  *Deleted `evalAtmosphereSunNEESecondary` + `evalAtmosphereMoonNEESecondary` and their gated secondary call block.*
- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`** — fork-owned change.
  *Deleted the now-orphaned surface samplers `sampleAtmosphereSunLight` / `sampleAtmosphereMoonLight` and their private `pickMoon` helper (only the bespoke NEE called them). The volumetric samplers (`sampleAtmosphereSunLightVolume`, …) are retained — still used by the RTXDI volume integrator.*
- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned change.
  *Removed the `useDirectionalLights` RTX_OPTION (injection is now unconditional in Numos) and the dead `debugEnableAtmosphereNee` diagnostic; reworded `directionalLightRadianceScale` (no longer references the toggle).*
- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned change.
  *`debugSkyBisectFlags` now packs only bit 1 (`debugEnableSkyMissShading`); bit 0 (atmosphere NEE) and bit 2 (directional-lights skip) retired with the bespoke NEE.*
- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned change.
  *`fhSyncAtmosphereDistantLights` now gates on `skyMode == Numos` only (dropped the `useDirectionalLights` condition); comments de-experimentalised.*
- **`docs/CloudSystem.md`** — updated the cloud-on-terrain shadow section to the real-distant-light fold; corrected `cloudShadowStrength` default (0 → **0.5**).
- **`RtxOptions.md`** — REGEN PENDING (drops `useDirectionalLights` + `debugEnableAtmosphereNee`).

---

## Workstream — Diffuse-indirect sky radiance scale (fork — 2026-06-28)

Adds the `rtx.atmosphere.skyIndirectRadianceScale` knob (default **1.0** = physical). Since the sun/moon graduated to real distant lights, the distant-light sun out-radiates the sky and indirect lighting reads dull; raising this multiplies sky radiance gathered by **diffuse indirect bounces only**. The scale is applied per-ray inside `evalSkyRadiance` *after* the sky-view LUT sample, so it never feeds any LUT bake. Eligibility is classified at the lobe-sample site (where the material type is known): only an opaque diffuse reflection/transmission gather opts in — sky seen via specular reflection, refraction, alpha-cutout (opacity transmission), translucent/glass lobes, PSR, or the primary view stays at physical brightness, so reflections keep matching the visible sky.

- **`src/dxvk/rtx_render/rtx_options.h`** — fork-owned addition.
  *Adds `RTX_OPTION_ARGS(rtx.atmosphere, float, skyIndirectRadianceScale, 1.0f, …, args.minValue = 0.0f)` after `sunsetSaturation`.*
- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`** — fork-owned change.
  *Repurposes the dead `pad_cloudAnisotropy` slot as `skyIndirectRadianceScale`; CB layout / `sizeof(AtmosphereArgs)` unchanged.*
- **`src/dxvk/rtx_render/rtx_atmosphere.cpp`** — fork-owned change.
  *`getAtmosphereArgs` populates the field (clamped >= 0). `normalizeForSkyLutCache` zeroes it in the sky-LUT memcmp key — it is applied post-LUT, so dragging the slider must not trigger a rebake.*
- **`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_sky.slangh`** — fork-owned change.
  *Adds the trailing `applySkyIndirectRadianceScale` parameter (default false) to `evalSkyRadiance` and multiplies the final radiance by `args.skyIndirectRadianceScale` only when it is set.*
- **`src/dxvk/shaders/rtx/algorithm/path_state.slangh`** — fork-owned change.
  *Adds `_skyGatherEligible` (uint8) + a `skyGatherEligible` bool property so the indirect integrator knows whether the in-flight ray came from a diffuse sky gather (`_flags` is bit-saturated, hence its own byte).*
- **`src/dxvk/shaders/rtx/algorithm/integrator_indirect.slangh`** — fork-owned change.
  *Initializes `skyGatherEligible` (false in `pathStateCreateEmpty`; `!firstSampledLobeIsSpecular` at first bounce); reclassifies it after each continuation sample using `materialType` + the opaque lobe enum; passes it as `applySkyIndirectRadianceScale` at the physical-atmosphere sky miss.*
- **`src/dxvk/rtx_render/rtx_fork_atmosphere.cpp`** — fork-owned change.
  *Adds the "Sky Indirect Scale" ImGui slider (0–20) + tooltip in the atmosphere UI.*
- **`RtxOptions.md`** — REGEN PENDING (adds `rtx.atmosphere.skyIndirectRadianceScale`).

---
