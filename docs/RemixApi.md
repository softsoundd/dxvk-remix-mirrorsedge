# Remix C API Reference

This page is the developer-facing reference for the Remix runtime's
plain-C API — every function a plugin or host application can call,
grouped by purpose. The header itself
([`public/include/remix/remix_c.h`](../public/include/remix/remix_c.h))
is the source of truth; this page exists to make it discoverable.

For SDK setup and the high-level mental model, see
[`RemixSDK.md`](RemixSDK.md). For per-subsystem string-keyed conventions
that ride on top of the API (weather, etc.), see the
[Convention namespaces](#convention-namespaces) section.

The C++ wrapper at
[`public/include/remix/remix.h`](../public/include/remix/remix.h)
is a thin RAII layer over the same surface; the conventions documented
here apply to both.

---

## Consuming the header

The recommended pattern is to **vendor `remix_c.h` directly into your
project's `extern/` tree and include it as the type definition source**.
Do not hand-type the structs, do not generate bindings — the header is
small (~1KLOC), self-contained (no dependencies beyond `<stdint.h>`
and optionally `<windows.h>`), and ABI-stable within a major version.

A working CMake setup looks like this:

```cmake
# Auto-refresh the vendored header from a sibling dxvk-remix checkout
# at configure time. Falls back to the cached snapshot if no source is
# nearby (CI, fresh clones).
set(REMIX_HEADER_DST "${CMAKE_CURRENT_SOURCE_DIR}/extern/remix/remix_c.h")
foreach(REMIX_CANDIDATE
        "${CMAKE_CURRENT_SOURCE_DIR}/dxvk-remix/public/include/remix/remix_c.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../dxvk-remix/public/include/remix/remix_c.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../dxvk-remix/public/include/remix/remix_c.h")
    if(EXISTS "${REMIX_CANDIDATE}")
        file(COPY "${REMIX_CANDIDATE}" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/extern/remix/")
        break()
    endif()
endforeach()

target_include_directories(MyPlugin PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/extern")
```

Then in source:

```c
#include "remix/remix_c.h"
```

Why this pattern dominates the alternatives:

| Property | Why it matters |
| :-- | :-- |
| **Header is the type source** | Zero hand-typed bindings → zero drift between plugin and runtime |
| **Configure-time auto-refresh** | Pulling latest API = recompile; no manual binding-regen step |
| **Cached snapshot in git** | Fresh clones / CI without dxvk-remix nearby still build |
| **Snapshot is the version pin** | The committed header IS the API contract for that build |

### Defines that affect the header

| Define | Effect |
| :-- | :-- |
| `REMIX_LIBRARY_EXPORTS` | Marks symbols as `dllexport`. Set when building the runtime; consumers leave this unset (default `dllimport`). |
| `REMIX_WINAPI_NO_INCLUDE` | Skips `<windows.h>`, declares the WinAPI types/imports the loader needs manually. |
| `REMIX_WINAPI_NO_LIBRARY_LOADER` | Skips the inline `remixapi_lib_loadRemixDllAndInitialize` helper and its imported loader symbols. Use when you load `d3d9.dll` yourself. |
| `REMIX_ALLOW_X86` | Drops the 64-bit static-assert. Required when building 32-bit consumers that talk to the runtime through the bridge. |

---

## Initialization & lifecycle

### `remixapi_InitializeLibrary`

```c
remixapi_ErrorCode remixapi_InitializeLibrary(
    const remixapi_InitializeLibraryInfo* info,
    remixapi_Interface*                   out_result);
```

The single entry point. Fills `out_result` with the function pointer
table (`remixapi_Interface`) used for every subsequent call. Pass
`REMIXAPI_VERSION_MAKE(major, minor, patch)` in `info->version`; an
incompatible runtime returns `REMIXAPI_ERROR_CODE_INCOMPATIBLE_VERSION`.

Convenience helpers `remixapi_lib_loadRemixDllAndInitialize` and
`remixapi_lib_shutdownAndUnloadRemixDll` exist in the header for
projects that want one-shot DLL load + initialize.

### `Startup` / `Shutdown`

```c
remixapi_ErrorCode Startup(const remixapi_StartupInfo* info);
remixapi_ErrorCode Shutdown(void);
```

`Startup` activates the renderer for a given HWND. Optional flags on
`remixapi_StartupInfo` control sRGB conversion, swapchain ownership
(`forceNoVkSwapchain` for OpenGL / external present paths), and editor
mode. `Shutdown` tears down all resources; safe to call repeatedly.

### `Present`

```c
remixapi_ErrorCode Present(const remixapi_PresentInfo* info);
```

Submits the current frame. `info->hwndOverride` lets a single-process
multi-window app retarget the present surface; pass `NULL` for the HWND
captured at `Startup`.

### `RegisterCallbacks`

```c
remixapi_ErrorCode RegisterCallbacks(
    PFN_remixapi_BridgeCallback beginSceneCallback,
    PFN_remixapi_BridgeCallback endSceneCallback,
    PFN_remixapi_BridgeCallback presentCallback);
```

Optional frame-boundary hooks. Mirrors the legacy bridge callback
semantics; useful for plugins that want to interleave work with the
runtime's frame timing without owning the render loop.

### `AutoInstancePersistentLights`

```c
remixapi_ErrorCode AutoInstancePersistentLights(void);
```

Re-emits a `DrawLightInstance` for every still-live light handle this
frame. Lets a plugin define lights once and let the runtime keep them
alive without per-frame draw calls.

---

## D3D9 interop

These let an application share a D3D9Ex device with the runtime, pull
intermediate render targets out as Vulkan images, or hand off a
DXGI-shared D3D11 texture for cross-API compositing.

| Function | Purpose |
| :-- | :-- |
| `dxvk_CreateD3D9` | Create the runtime's `IDirect3D9Ex`. |
| `dxvk_RegisterD3D9Device` | Bind an `IDirect3DDevice9Ex` (must come from `dxvk_CreateD3D9`). |
| `dxvk_GetExternalSwapchain` | Hand the app a `VkImage` + sync semaphores so it can present (e.g. via OpenGL interop). Requires `forceNoVkSwapchain=true` at startup. |
| `dxvk_GetVkImage` | Map a D3D9 surface to its underlying `VkImage`. |
| `dxvk_GetSharedD3D11TextureHandle` | Get an NT shared handle for the final-color render target, openable by D3D11 / D3D12. |
| `dxvk_CopyRenderingOutput` | Blit one of the runtime's intermediate outputs (final color, depth, normals, object-picking ID) into a D3D9 surface the caller owns. |
| `dxvk_SetDefaultOutput` | Set the clear value the runtime uses when an intermediate output isn't produced this frame. |
| `dxvk_GetTextureHash` | Compute the runtime's content hash for a D3D9 texture (matches what `RtxOptions` keys would see). |

---

## Camera

### `SetupCamera`

```c
remixapi_ErrorCode SetupCamera(const remixapi_CameraInfo* info);
```

Submits the active camera for the frame. `remixapi_CameraInfo` carries
view/projection matrices and a `remixapi_CameraType` (`WORLD`, `SKY`,
`VIEW_MODEL`). For applications that don't have a matrix-form camera
ready, attach `remixapi_CameraInfoParameterizedEXT` via the `pNext`
chain to specify position/forward/up/right + FOV + aspect + planes.

### `SetCameraMediumMaterial`

```c
remixapi_ErrorCode SetCameraMediumMaterial(const remixapi_CameraMediumInfo* info);
```

Sets the medium (typically a translucent material — water, fog, etc.)
the camera is currently *inside*. The runtime uses this for
view-dependent absorption and refraction.

---

## Materials

### `CreateMaterial` / `DestroyMaterial`

```c
remixapi_ErrorCode CreateMaterial(
    const remixapi_MaterialInfo*  info,
    remixapi_MaterialHandle*      out_handle);

remixapi_ErrorCode DestroyMaterial(remixapi_MaterialHandle handle);
```

Materials use a Vulkan-style `pNext` extension chain. `remixapi_MaterialInfo`
is the base; attach exactly one of the type-specific extensions:

| Extension | When to use |
| :-- | :-- |
| `remixapi_MaterialInfoOpaqueEXT` | Standard PBR opaque (albedo / roughness / metallic / normal / height / emissive). |
| `remixapi_MaterialInfoOpaqueSubsurfaceEXT` | Foliage / skin — chains *after* `OpaqueEXT`. See [`FoliageSystem.md`](FoliageSystem.md). |
| `remixapi_MaterialInfoTranslucentEXT` | Glass, water, refractive media. |
| `remixapi_MaterialInfoPortalEXT` | Ray portals. |

The runtime uses `info->hash` to dedupe and to bind replacement assets
from your USD captures; pick a stable hash that survives content
reloads.

---

## Meshes

### `CreateMesh` / `CreateMeshBatched` / `DestroyMesh`

```c
remixapi_ErrorCode CreateMesh(
    const remixapi_MeshInfo*  info,
    remixapi_MeshHandle*      out_handle);
remixapi_ErrorCode CreateMeshBatched(
    const remixapi_MeshInfo*  info,
    remixapi_MeshHandle*      out_handle);
remixapi_ErrorCode DestroyMesh(remixapi_MeshHandle handle);
```

`CreateMesh` allocates DXVK buffers and registers replacement assets
synchronously on the calling thread. `CreateMeshBatched` deep-copies
the info and defers both steps to the next render-thread flush point
(`DrawInstance`, `Present`, `AutoInstancePersistentLights`) — use it
when registering many meshes at once so the registration cost
amortizes onto the render thread.

`remixapi_MeshInfo::surfaces_values` is an array of
`remixapi_MeshInfoSurfaceTriangles`; each surface carries its own
vertex/index arrays, an optional `remixapi_MeshInfoSkinning` block
(blend weights + blend indices for hardware skinning), and a
`remixapi_MaterialHandle`.

The vertex format is `remixapi_HardcodedVertex` — position(3) +
normal(3) + texcoord(2) + color(uint32) + padding to 64 bytes. The
padding is reserved for future runtime use; leave it zeroed.

---

## Instances (per-frame draw)

### `DrawInstance`

```c
remixapi_ErrorCode DrawInstance(const remixapi_InstanceInfo* info);
```

The per-frame submission for static / dynamic geometry. `remixapi_InstanceInfo`
carries the mesh handle, world transform, double-sided flag, and a
`remixapi_InstanceCategoryFlags` bitfield used for runtime
classification (sky, decal, terrain, particle, etc.). Attach EXT chains
for additional behavior:

| Extension | Effect |
| :-- | :-- |
| `remixapi_InstanceInfoBoneTransformsEXT` | Per-bone world transforms for skinned mesh playback. |
| `remixapi_InstanceInfoBlendEXT` | D3D9-style alpha test/blend state, used when the material has `useDrawCallAlphaState=true`. |
| `remixapi_InstanceInfoObjectPickingEXT` | Tags the instance with a 32-bit ID readable via the picking API. |
| `remixapi_InstanceInfoParticleSystemEXT` | Spawns a GPU particle system bound to this instance. |
| `remixapi_InstanceInfoGpuInstancingEXT` | Submits N GPU-instanced copies in one call. |

The full list of category bits is `remixapi_InstanceCategoryBit` in the
header. Most plugins only need `WORLD_MATTE`, `SKY`, `PARTICLE`, the
decal bits, and the `IGNORE_*` modifiers.

---

## Lights

### `CreateLight` / `CreateLightBatched` / `DestroyLight`

```c
remixapi_ErrorCode CreateLight(
    const remixapi_LightInfo* info,
    remixapi_LightHandle*     out_handle);
remixapi_ErrorCode CreateLightBatched(
    const remixapi_LightInfo* info,
    remixapi_LightHandle*     out_handle);
remixapi_ErrorCode DestroyLight(remixapi_LightHandle handle);
```

Same `Batched` semantics as meshes — defers registration to the next
render-thread flush. `remixapi_LightInfo` is the base; attach one of:

| Extension | Light shape |
| :-- | :-- |
| `remixapi_LightInfoSphereEXT` | Sphere / point with optional cone shaping. |
| `remixapi_LightInfoRectEXT` | Rectangular area light. |
| `remixapi_LightInfoDiskEXT` | Disk area light. |
| `remixapi_LightInfoCylinderEXT` | Cylinder light. |
| `remixapi_LightInfoDistantEXT` | Sun / directional. |
| `remixapi_LightInfoDomeEXT` | Image-based dome (skybox). |
| `remixapi_LightInfoUSDEXT` | USD-token-backed light; if attached, sibling EXTs and `radiance` on the base are ignored. |

`remixapi_LightInfoLightShaping` (used inside the sphere/rect/disk
EXTs) gives spotlight-style cone shaping with cone angle / softness /
focus exponent.

### `DrawLightInstance`

```c
remixapi_ErrorCode DrawLightInstance(remixapi_LightHandle lightHandle);
```

Re-emits the light for the current frame. Call once per frame per
visible light — or use `AutoInstancePersistentLights` to do it in bulk.

### `UpdateLightDefinition`

```c
remixapi_ErrorCode UpdateLightDefinition(
    remixapi_LightHandle handle,
    const remixapi_LightInfo* info);
```

Queues a definition update for an existing handle, applied next frame
on the render thread. Cheaper than destroy + recreate when only the
parameters change.

---

## Textures

### `CreateTexture` / `DestroyTexture`

```c
remixapi_ErrorCode CreateTexture(
    const remixapi_TextureInfo* info,
    remixapi_TextureHandle*     out_handle);
remixapi_ErrorCode DestroyTexture(remixapi_TextureHandle handle);
```

Direct CPU → GPU texture upload with a stable runtime hash. Format
options are limited to a curated subset of Vulkan formats
(`remixapi_Format`) — RGBA / BGRA in UNORM and SRGB, plus BC1 / BC3 /
BC5 / BC7 block-compressed variants. `info->data` holds all mip levels
sequentially; `info->dataSize` is the total byte count.

### `AddTextureHash` / `RemoveTextureHash`

```c
remixapi_ErrorCode AddTextureHash(const char* textureCategory, const char* textureHash);
remixapi_ErrorCode RemoveTextureHash(const char* textureCategory, const char* textureHash);
```

Registers a texture hash into a named category — used by the asset
replacer to match captured content against runtime classifications
(sky, terrain, UI, etc.) without needing per-frame draw-time hints.

---

## Configuration — `SetConfigVariable`

```c
remixapi_ErrorCode SetConfigVariable(const char* key, const char* value);
```

The string-keyed escape hatch into the runtime's `RtxOptions` system.
Any option in [`RtxOptions.md`](../RtxOptions.md) — every `rtx.*` knob
the runtime exposes — is reachable through this single function.
Values are stringified the same way they appear in `dxvk.conf`
(numbers, booleans as `true`/`false`, vectors as `x, y, z`).

```c
iface.SetConfigVariable("rtx.bloom.enable",            "true");
iface.SetConfigVariable("rtx.bloom.luminanceThreshold", "0.4");
iface.SetConfigVariable("rtx.fogColorScale",            "0.25");
```

For the full list of available keys, see [`RtxOptions.md`](../RtxOptions.md).
For the architecture (config layers, priority, persistence), see
[`RemixConfig.md`](RemixConfig.md). For fork-side `rtx.*` namespaces
that exist specifically as plugin-tuning surfaces (e.g.
`rtx.weather.preset.*`), see [Convention namespaces](#convention-namespaces).

---

## Game-state — `SetGameValue` / `GetGameValue`

```c
remixapi_ErrorCode SetGameValue(const char* key, const char* value);
remixapi_ErrorCode GetGameValue(
    const char* key,
    char*       out_buffer,
    uint32_t    in_buffer_size,
    uint32_t*   out_actual_size);
```

A second string-keyed channel, separate from `SetConfigVariable`. The
runtime stores `(key, value)` pairs in a thread-safe map that
**fork-side subsystems and graph components read by name**. The store
survives `Shutdown` / re-init, so callers do not have to re-populate
across device resets.

`GetGameValue` is two-call sized: pass `in_buffer_size=0` first to read
the required size from `*out_actual_size`, then call again with a sized
buffer. Missing keys return `REMIXAPI_ERROR_CODE_SUCCESS` with
`*out_actual_size == 0`.

Keys are not namespace-validated by the runtime — plugins choose them.
The convention used by fork-side subsystems is a leading double
underscore: `__weather.target`, `__weather.blend_seconds`, etc. See
[Convention namespaces](#convention-namespaces) for the registry.

---

## UI

### `GetUIState` / `SetUIState`

```c
remixapi_UIState GetUIState(void);
remixapi_ErrorCode SetUIState(remixapi_UIState state);
```

Toggles the runtime's developer UI between `NONE`, `BASIC`, and
`ADVANCED`. Useful for plugins that want to bind a hotkey or expose a
"show Remix UI" toggle from the host application.

### `DrawScreenOverlay`

```c
remixapi_ErrorCode DrawScreenOverlay(
    const void*       pPixelData,
    uint32_t          width,
    uint32_t          height,
    remixapi_Format   format,
    float             opacity);
```

Composites a CPU-side pixel buffer over the final frame at the
specified opacity. Intended for plugin-drawn HUDs or debug overlays
that need to land *after* tonemapping.

---

## Object picking

### `pick_RequestObjectPicking`

```c
typedef void (*PFN_remixapi_pick_RequestObjectPickingUserCallback)(
    const uint32_t* objectPickingValues_values,
    uint32_t        objectPickingValues_count,
    void*           callbackUserData);

remixapi_ErrorCode pick_RequestObjectPicking(
    const remixapi_Rect2D*                              pixelRegion,
    PFN_remixapi_pick_RequestObjectPickingUserCallback  callback,
    void*                                               callbackUserData);
```

Asynchronous readback of the per-pixel `objectPickingValue` set on
instances via `remixapi_InstanceInfoObjectPickingEXT`. Pass an
output-resolution rect; the callback fires (from any thread) with the
unique IDs that drew into that rect this frame.

### `pick_HighlightObjects`

```c
remixapi_ErrorCode pick_HighlightObjects(
    const uint32_t* objectPickingValues_values,
    uint32_t        objectPickingValues_count,
    uint8_t colorR, uint8_t colorG, uint8_t colorB);
```

Tints the listed picking IDs with the given RGB color in the next
frame's composite. Useful for editor-style hover highlights.

---

## Memory

### `GetVramStats`

```c
remixapi_ErrorCode GetVramStats(remixapi_VramStats* out_stats);
```

Snapshot of VRAM usage broken out by category (textures, BVH,
opacity micromaps, render targets, …) plus the driver-reported total
allocation from `VK_EXT_memory_budget`. Includes
`forkTextureCacheCount`, the size of the fork-side
`RtxTextureManager`'s sparse cache — useful for diagnosing whether
texture cache growth is in plugin uploads or fork-side streaming.

### `RequestVramCompaction`

```c
remixapi_ErrorCode RequestVramCompaction(void);
```

Asks the renderer to release retained empty `VkDeviceMemory` chunks
back to the driver on the next render-thread tick. Fire at bulk
turnover events (cell transitions, fast-travel) — not every frame; the
call blocks on the next frame's `vkFreeMemory` sweep.

### `RequestTextureVramFree`

```c
remixapi_ErrorCode RequestTextureVramFree(void);
```

Forces the texture manager to demote / clear textures not currently
needed. Catches orphan cache entries whose owners died without a
matching `DestroyTexture` / `DestroyMaterial`. Same firing cadence
guidance as compaction.

---

## Convention namespaces

The two string-keyed channels above (`SetConfigVariable` and
`SetGameValue` / `GetGameValue`) host **fork-side conventions** that
expose entire subsystems without needing new typed API surface. Each
convention namespace gets its own reference page so it can grow
independently of the typed C API.

| Namespace | Channel | Reference |
| :-- | :-- | :-- |
| `__weather.*`, `__sky.*` | `SetGameValue` / `GetGameValue` | [`RemixSkyAPI.md`](RemixSkyAPI.md) |
| `rtx.weather.preset.*` | `SetConfigVariable` | [`RemixSkyAPI.md`](RemixSkyAPI.md) |

When a new fork-side subsystem starts publishing a `__<ns>.*`
GameStateStore convention or a `rtx.<ns>.*` ConfigVariable namespace
intended as a plugin-tuning surface, add it to this table and create a
matching `Remix<Ns>API.md` spoke page. See `dxvk-remix/CLAUDE.md` for
the doc-update discipline.

---

## Versioning

The runtime exports `REMIXAPI_VERSION_MAJOR`, `REMIXAPI_VERSION_MINOR`,
`REMIXAPI_VERSION_PATCH` macros in the header. Pass
`REMIXAPI_VERSION_MAKE(major, minor, patch)` to
`remixapi_InitializeLibrary` and check the return for
`REMIXAPI_ERROR_CODE_INCOMPATIBLE_VERSION`.

`remixapi_StructType` values are append-only — never reorder. Adding
new EXT structs is a minor-version event; changing the layout of an
existing struct is a major-version event.

For the dated change log, see
[`RemixApiChangelog.md`](RemixApiChangelog.md).

---

## See also

- [`RemixSDK.md`](RemixSDK.md) — SDK setup, mental model, minimal C example.
- [`RemixApiChangelog.md`](RemixApiChangelog.md) — dated API change log.
- [`RemixConfig.md`](RemixConfig.md) — `RtxOptions` config-layer architecture.
- [`RtxOptions.md`](../RtxOptions.md) — auto-generated complete `rtx.*` reference.
- [`RemixSkyAPI.md`](RemixSkyAPI.md) — Sky / atmosphere plugin surface (sun, stars, moons, clouds) plus weather preset blender.
