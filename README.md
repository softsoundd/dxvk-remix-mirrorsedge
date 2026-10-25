# dxvk-remix

[![Build Status](https://github.com/softsoundd/dxvk-remix-mirrorsedge/actions/workflows/build.yml/badge.svg?branch=mirrors-edge)](https://github.com/softsoundd/dxvk-remix-mirrorsedge/actions/workflows/build.yml)

dxvk-remix is a fork of the [DXVK](https://github.com/doitsujin/dxvk) project. On this branch, Mirror's Edge is presented as the game's own raster frame, with DLSS, frame generation, Reflex, and Remix post effects on top.

Thanks to all the contributors to DXVK for creating this foundational piece of software, on top of which we were able to build the RTX Remix Runtime.

While dxvk-remix is a fork of DXVK, please report bugs encountered with dxvk-remix to this repo rather than to the DXVK project.

dxvk-remix also contains a subproject in the `bridge` folder, which enables 32 bit games to communicate with the 64 bit dxvk-remix runtime.

## NGX passthrough branch (`mirrors-edge-ngx`)

This branch adds an NGX passthrough mode where Mirror's Edge's own rasterised rendering is presented unchanged (no path tracing, no scene capture) while DLSS Super Resolution / DLAA, DLSS Frame Generation, Reflex, and Remix's Post FX runs on top of it. Like the rest of this fork, NGX passthrough mode may work in other UE3 titles, but support is not guaranteed.

### NGX mode setup

**The game's MSAA must be off** because depth cannot be resolved while it is active - plus you want DLSS/DLAA anyway, right? By default NGX mode will attempt to force MSAA off at runtime via `rtx.ngxPassthrough.disableGameMsaa` which makes the D3D9 layer report multisampling as unsupported and create every surface single-sampled. In (rare) cases where that fails, MSAA will need to be disabled manually via the game's settings/configs.

NGX passthrough mode is enabled by default on this branch, including when no `rtx.conf` is present. Mirror's Edge also sets it in the built-in app config. A depth buffer created while the mode is off is promoted to a shader-readable layout when the mode is turned on. Multisampled depth still requires a restart after MSAA is refused.

How it works:

- Frames are intercepted before the game's postprocess chain (`rtx.ngxPassthrough.prePostProcess`). DLSS reads linear HDR scene colour, with the HDR flag and NGX auto-exposure, and writes the antialiased result back. Bloom, tonemapping, and dynamic contrast then run on the stabilised, unjittered image. Scene colour alpha stores depth. Write-back remerges that alpha with DLSS RGB. The alpha is resampled from the jittered render grid, which keeps fog and depth of field aligned.
- Injection is the first post-process pass after the scene. Stock UE3 copies scene colour into its resolve texture with a draw, and Mirror's Edge uses `StretchRect`. Both kinds of copy are learned, so post passes are recognised by the image they read. The DOF/bloom gather is the pre-post point once the frame is past its foreground depth clear, so bloom and depth of field read the upscaler output. In Mirror's Edge, TdSunHaze comes first and composites before DLSS runs. Its glow is a nonlinear function of the depth in scene-colour alpha, and that depth matches the colour only on the jittered raster. Without a gather or haze pass, injection is the first depth-of-field, bloom, tonemap or gamma read after the scene. That read is aimed from the previous frame and counted from the mid-scene depth clear. Fog and other in-scene colour reads are not that point. A final backbuffer tonemap or gamma composite counts as well. If no suitable pass is found, injection falls back to the late scene end, on post-processed LDR, before the UI.
- Two fallbacks keep an unrecognised frame shape from breaking the image. Each needs two consecutive frames of evidence and logs the draw that tripped it. Scene geometry drawn after an injection suspends the haze, gather and aimed-read points for 600 frames, and each repeat doubles that, up to 9600 frames. A post pass that stays on the reduced rect after an upscale moves sub-full presets to the late point until ScreenPercentage or the scene buffer size changes. A `UE3 profile` log line records the resolve method, injection point, sub-full mode, upscale owner and any suspension whenever they change.
- Motion vectors: depth reprojected through current/previous cameras, overridden by a depth-tested velocity raster for rigid movers, GPU-skinned meshes, and CPU-skinned dynamics (`rtx.ngxPassthrough.objectVelocities`). Used by DLSS, Frame Generation, and Remix PostFX blur. Mid-scene depth clears are snapshot-merged at injection; missed pixels fall back to camera reprojection.
- Sub-pixel Halton jitter is a fractional viewport offset on scene-target draws and depth-tested fullsize screen-space passes (shadows, distortion). `ScreenPositionScaleBias` is compensated so clip-derived UVs stay aligned. Game state is untouched; DLSS output at injection is unjittered.
- DLSS presets drive render resolution and LOD bias. Full Resolution is DLAA on linear scene colour before the post chain. Lower presets render into a ScreenPercentage subrect. DLSS upscales that HDR subrect onto the full scene-colour target at the same pre-post point. The post chain is retargeted onto the upscaled image, including a final composite drawn at the centred view rect, motion-blur clamps and the velocity buffer. The engine's final stretch becomes a 1:1 composite. When the late point upscales an image the uber pass already gamma-encoded, that encoding is removed before DLSS and applied again afterwards.
- Resolution goes through UE3's `FSystemSettings::ScreenPercentage`. Scale-factor readers are redirected to a Remix shadow. Settings and the UI still see the real value. Discovery is validated live and can be pinned with `rtx.ngxPassthrough.systemSettingsScreenPercentageRva`. If pre-post injection does not fire, a late DLSS pass can still replace the engine upscale (`rtx.ngxPassthrough.screenPercentageUpscaleOwner`). If `NeedsUpscale()` cannot be hooked, the runtime upscales the reduced subrect itself.
- Frame Generation uses the synthesised depth/MVs and camera via the standard Remix DLFG presenter on the post-UI backbuffer. First-person meshes use true object motion. A HUD-less frame at the scene-end/UI boundary separates UI without heuristics (`rtx.ngxPassthrough.dlfgHudlessInput`).
- DLSS 3D-Guided Neural Rendering (`rtx.dlssNeuralRendering.enable`) runs on the finished pre-UI backbuffer in either injection mode, so both modes give it the same image. DLSS-NR takes display-referred colour, which the game's own tonemap produces. It uses the frame's passthrough depth and motion vectors, and the HUD-less Frame Generation copy is taken after it. DLSS-NR skips any frame whose backbuffer size differs from the upscaled view, and any pre-post frame whose scene blit is not identified. Remix bloom (`rtx.bloom.enable`) is off by default, since UE3 titles bloom in their own post chain. When enabled, it runs only on linear HDR colour.
- Remix Post FX (`rtx.postfx.*`): motion blur uses synthesised MVs, linear view-Z, and static/dynamic surface flags. First-person meshes blur with the scene by default; `rtx.ngxPassthrough.motionBlurFirstPerson = False` keeps the view model crisp. Chromatic aberration and vignette share that pass.

Notes and limitations:

- Options live under `rtx.ngxPassthrough.*`. Rendering, General shows depth and motion-vector views on Debug Visualization. Dispatch, jitter, and velocity-capture figures are under Diagnostics.
- `rtx.dlss.preset` selects the DLSS model. The default is 0. Default lets the SDK pick the model for the quality mode. DLAA, Balanced, and Quality use K. Performance uses M. Ultra Performance uses L. The menu offers Default, J, K, L, and M.
- Ghosting issues with particles and textures on transparent planes (chain link fences). Will investigate solutions for this in the future.
- Separate existing issue with Mirror's Edge where sub-native render resolutions offsets the position of some effects like sun haze and first person self-shadowing. This is not Remix's doing.

## Build instructions

### Requirements:
1. Windows 10 or 11
2. [Git](https://git-scm.com/download/win)
3. [Visual Studio ](https://visualstudio.microsoft.com/vs/older-downloads/)
    - VS 2019 and 2022 are tested
    - Building for Windows on Arm requires VS 2022 with the `C++ ARM64/ARM64EC build tools` individual component
    - MSVC v14.44 or newer is preferred when building for Windows on Arm
    - Note that our build system will always use the most recent version available on the system
4. [Windows SDK](https://developer.microsoft.com/en-us/windows/downloads/sdk-archive/)
    - 10.0.19041.0 is tested
5. [Meson](https://mesonbuild.com/)
    - 1.8.2 has been tested
    - Follow [instructions](https://mesonbuild.com/SimpleStart.html#installing-meson) on how to install and reboot the PC before moving on (Meson will indicate as much)
6. [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows)
    - 1.4.313.2 or newer
    - You may need to uninstall previous SDK if you have an old version
7. [Python](https://www.python.org/downloads/)
    - 3.9 or newer
    - Ensure you are using python installed from the link above and not from the Microsoft Store
    - Python is required by developer build tooling; the packaged RTX Remix Runtime does not link against Python.
8. [DirectX Runtime](https://www.microsoft.com/en-us/download/details.aspx?id=35)
    - Latest version should work.
    - This includes d3d9x*.dll which are required to run the game
    - May already be installed if you have D3D9 games installed

#### Additional notes:
- If dependency paths change (for example, after installing a new Vulkan SDK), reconfigure the affected build from the repository root, such as `meson setup --reconfigure _CompRelease_x64`.

### Generate and build dxvk-remix Visual Studio project 
1. Clone the repository with all submodules:
	- `git clone --recursive https://github.com/NVIDIAGameWorks/dxvk-remix.git`

	If the clone was made non-recursively and the submodules are missing, clone them separately:
	- `git submodule update --init --recursive`

2. Install all the [requirements](#requirements) before proceeding further

3. Make sure PowerShell scripts are enabled
    - One-time system setup: run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned` in an elevated PowerShell prompt, then close and reopen any existing PowerShell prompts
	
4. To generate and build dxvk-remix project:
    - Right Click on `dxvk-remix\build_dxvk_all_ninja.ps1` and select "Run with Powershell"
    - If that fails or has problems, run the build manually in a way you can read the errors:
        - open a windows file explorer to the `dxvk-remix` folder
        - remove only the generated configuration that failed, such as `_CompDebug_x64/`; remove `_vs/` as well only if the generated Visual Studio solution must be recreated
        - type `cmd` in the address bar to open a command line window in that folder.
        - copy and paste `powershell -command "& .\build_dxvk_all_ninja.ps1"` into the command line, then press enter
    - Optional flags:
        - `-SkipApics` — skip downloading game test captures (requires auth token)
        - `-BuildArch <x64|arm64|arm64ec>` — target architecture, default `x64`. `arm64ec` is configured with the Visual Studio backend rather than Ninja, because Meson cannot link arm64ec targets with Ninja, so the `_Comp*_arm64ec` directories hold MSBuild projects instead of Ninja files. The generated **_vs** solution always drives the `_Comp*_x64` directories; build other architectures with `meson compile -C <build directory>`.
    - Examples:
        ```powershell
        .\build_dxvk_all_ninja.ps1
        .\build_dxvk_all_ninja.ps1 -SkipApics
        .\build_dxvk_all_ninja.ps1 -BuildArch arm64
        ```
    - This will build all 3 configurations of dxvk-remix project inside subdirectories of the build tree:
        - **_CompDebug_x64** - full debug instrumentation, runtime speed may be slow
        - **_CompDebugOptimized_x64** - partial debug instrumentation (i.e. asserts), runtime speed is generally comparable to that of release configuration
        - **_CompRelease_x64** - fastest runtime
        - Note: replace **x64** with the architecture selected via `-BuildArch`
    - This will generate a project in the **_vs** subdirectory

5. Open **_vs/dxvk-remix.sln** in Visual Studio (2019+). 
    - Do not convert the solution on load if prompted when using a newer version of Visual Studio 
    - Once generated, the project can be built via Visual Studio or via powershell scripts
    - A build will copy generated DXVK DLLs to any target project as specified in **gametargets.conf** (see its [setup section](#deploy-built-binaries-to-a-game))

### Deploy built binaries to a game 
1. First time only: copy **gametargets.example.conf** to **gametargets.conf** in the project root

2. Update paths in the **gametargets.conf** for your game. Follow example in the **gametargets.example.conf**. Make sure to remove "#" from the start of all three lines

3. Reconfigure and rebuild each configuration you use so Meson reloads **gametargets.conf**. For example:
    ```powershell
    meson setup --reconfigure _CompRelease_x64
    meson compile -C _CompRelease_x64
    ```
    The build deploys binaries to the game directories specified in **gametargets.conf**.

### Profiling Remix
Remix has support for profiling using the [Tracy](https://github.com/wolfpld/tracy) tool, specifically the [v0.8 release](https://github.com/wolfpld/tracy/releases/download/v0.8/Tracy-0.8.7z)

To enable Tracy profiling:
1. Open a command line window in a build folder (i.e. `dxvk-remix/_CompRelease_x64/`)
2. Run `meson --reconfigure -D enable_tracy=true`
3. Rebuild dxvk-remix-nv

To profile:
1. Launch tracy.exe
2. Launch the game and reach the section you wish to profile
3. When ready, hit `Connect` in Tracy to begin profiling.
4. It's best to collect at least 500 frames worth of data, so you can average out the results.

### Remix API

If there's an intent to use the Remix Renderer in projects with *available* source code, Direct3D 9 API can be utilized, since Remix's `d3d9.dll` implements the Direct3D 9 API.
Alternatively, Remix API can be used to programmatically pass the game data to the Remix Renderer, with *or* instead of Direct3D API. [Click for more info.](/documentation/RemixSDK.md)

## Project Documentation

- [Anti-Culling System](/documentation/AntiCullingSystem.md)
- [Contributing Guide](/CONTRIBUTING.md)
- [Foliage System](/documentation/FoliageSystem.md)
- [GPU Print](/documentation/GpuPrint.md)
- [Opacity Micromap](/documentation/OpacityMicromap.md)
- [Remix API](/documentation/RemixSDK.md)
- [Remix Runtime Binaries](/documentation/RemixRuntimeBinaries.md)
- [Rtx Options](/RtxOptions.md)
- [Terrain System](/documentation/TerrainSystem.md)
- [Unit Test](/documentation/UnitTest.md)
