# dxvk-remix

[![Build Status](https://github.com/NVIDIAGameWorks/dxvk-remix/actions/workflows/build.yml/badge.svg)](https://github.com/NVIDIAGameWorks/dxvk-remix/actions/workflows/build.yml)

dxvk-remix is a fork of the [DXVK](https://github.com/doitsujin/dxvk) project, which overhauls the fixed-function graphics pipeline implementation in order to remaster games with path tracing.

Thanks to all the contributors to DXVK for creating this foundational piece of software, on top of which we were able to build the RTX Remix Runtime.

While dxvk-remix is a fork of DXVK, please report bugs encountered with dxvk-remix to this repo rather than to the DXVK project.

dxvk-remix also contains a subproject in the `bridge` folder, which enables 32 bit games to communicate with the 64 bit dxvk-remix runtime.

## NGX passthrough branch (`mirrors-edge-ngx`)

This branch adds an NGX passthrough mode where Mirror's Edge's own rasterised rendering is presented unchanged (no path tracing, no scene capture) while DLSS Super Resolution / DLAA, DLSS Frame Generation, Reflex, and Remix's Post FX runs on top of it. Like the rest of this fork, NGX passthrough mode may work in other UE3 titles, but support is not guaranteed.

### NGX mode setup

**The game's MSAA must be off** because depth cannot be resolved while it is active - plus you want DLSS/DLAA anyway, right? By default NGX mode will attempt to force MSAA off at runtime via `rtx.ngxPassthrough.disableGameMsaa` which makes the D3D9 layer report multisampling as unsupported and create every surface single-sampled. In (rare) cases where that fails, MSAA will need to be disabled manually via the game's settings/configs.

Set `rtx.ngxPassthroughMode` at launch (already configured for Mirror's Edge in this branch) so depth buffers are created shader-readable.

How it works:

- Frames are intercepted before the game's postprocess chain (`rtx.ngxPassthrough.prePostProcess`). DLSS takes linear HDR scene colour in NGX HDR mode and writes the antialiased result back; bloom, tonemapping, and dynamic contrast then run on the stabilised, unjittered image. Scene colour alpha stores depth, so write-back remerges game alpha with DLSS RGB, jitter-resampled so effects like fog and depth of field stay aligned.
- Injection is at the first scene colour read after the last depth-writing draw (not the earliest postprocess quad inside UE3's DPG loop, which would leave first-person meshes unjittered and never anti-aliased). The pass is carried from the previous frame, counted from the mid-scene depth clear. If no suitable pass is found, it falls back to late scene-end (post-processed LDR, before UI).
- Motion vectors: depth reprojected through current/previous cameras, overridden by a depth-tested velocity raster for rigid movers, GPU-skinned meshes, and CPU-skinned dynamics (`rtx.ngxPassthrough.objectVelocities`). Used by DLSS, Frame Generation, and Remix PostFX blur. Mid-scene depth clears are snapshot-merged at injection; missed pixels fall back to camera reprojection.
- Sub-pixel Halton jitter is a fractional viewport offset on scene-target draws and depth-tested fullsize screen-space passes (shadows, distortion). `ScreenPositionScaleBias` is compensated so clip-derived UVs stay aligned. Game state is untouched; DLSS output at injection is unjittered.
- DLSS presets drive render resolution and LOD bias. Full Resolution is DLAA on the post-processed scene before UI; other tiers render lower and DLSS upscales as Super Resolution.
- Resolution goes through UE3's `FSystemSettings::ScreenPercentage`. Scale-factor readers are redirected to a Remix shadow; settings/UI still see the real value. Discovery is validated live and can be pinned with `rtx.ngxPassthrough.systemSettingsScreenPercentageRva`. The engine's upscale is normally replaced by DLSS (`rtx.ngxPassthrough.screenPercentageUpscaleOwner`); if `NeedsUpscale()` cannot be hooked, the runtime upscales the engine's reduced subrect itself.
- Frame Generation uses the synthesised depth/MVs and camera via the standard Remix DLFG presenter on the post-UI backbuffer. First-person meshes use true object motion. A HUD-less frame at the scene-end/UI boundary separates UI without heuristics (`rtx.ngxPassthrough.dlfgHudlessInput`).
- Remix Post FX (`rtx.postfx.*`): motion blur uses synthesised MVs, linear view-Z, and static/dynamic surface flags. First-person meshes blur with the scene by default; `rtx.ngxPassthrough.motionBlurFirstPerson = False` keeps the view model crisp. Chromatic aberration and vignette share that pass.

Notes and limitations:

- Options under `rtx.ngxPassthrough.*`; Rendering -> General shows DLSS stats and depth/MV visualisers.
- Default DLSS model is the transformer preset (`rtx.ngxPassthrough.dlssRenderPreset = 11`, preset K), which keeps more detail than NGX's default CNN presets; the developer menu can switch presets live (A-F legacy CNN, J/K transformer v1, L/M DLSS 4.5 transformer).
- Ghosting issues with particles and textures on transparent planes (chain link fences). Will investigate solutions for this in the future.
- Separate existing issue with Mirror's Edge where sub-native render resolutions offsets the position of some effects like sun haze and first person self-shadowing. This is not Remix's doing.

## Build instructions

### Requirements:
1. Windows 10 or 11
2. [Git](https://git-scm.com/download/win)
3. [Visual Studio ](https://visualstudio.microsoft.com/vs/older-downloads/)
    - VS 2019 is tested
    - VS 2022 may also work, but it is not actively tested
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
8. [DirectX Runtime](https://www.microsoft.com/en-us/download/details.aspx?id=35)
    - Latest version should work.
    - This includes d3d9x*.dll which are required to run the game
    - May already be installed if you have D3D9 games installed

#### Additional notes:
- If any dependency paths change (i.e. new Vulkan library), run `meson --reconfigure` in _Compiler64 directory via a command prompt. This may revert some custom VS project settings

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
        - remove artifacts from the previous attempt by deleting all folders that start with `_`, i.e. `_vs/` and `_Comp64Debug`
        - type `cmd` in the address bar to open a command line window in that folder.
        - copy and paste `powershell -command "& .\build_dxvk_all_ninja.ps1"` into the command line, then press enter
    - This will build all 3 configurations of dxvk-remix project inside subdirectories of the build tree: 
        - **_Comp64Debug** - full debug instrumentation, runtime speed may be slow
        - **_Comp64DebugOptimized** - partial debug instrumentation (i.e. asserts), runtime speed is generally comparable to that of release configuration
        - **_Comp64Release** - fastest runtime 
    - This will generate a project in the **_vs** subdirectory
    - Only x64 build targets are supported

5. Open **_vs/dxvk-remix.sln** in Visual Studio (2019+). 
    - Do not convert the solution on load if prompted when using a newer version of Visual Studio 
    - Once generated, the project can be built via Visual Studio or via powershell scripts
    - A build will copy generated DXVK DLLs to any target project as specified in **gametargets.conf** (see its [setup section](#deploy-built-binaries-to-a-game))

### Deploy built binaries to a game 
1. First time only: copy **gametargets.example.conf** to **gametargets.conf** in the project root

2. Update paths in the **gametargets.conf** for your game. Follow example in the **gametargets.example.conf**. Make sure to remove "#" from the start of all three lines

3. Open and, simply, re-save top-level **meson.build** file (i.e. via notepad) to update its time stamp, and rerun the build. This will trigger a full meson script run which will generate a project within the Visual Studio solution file and deploy built binaries into games' directories specified in **gametargets.conf**

### Profiling Remix
Remix has support for profiling using the [Tracy](https://github.com/wolfpld/tracy) tool, specifically the [v0.8 release](https://github.com/wolfpld/tracy/releases/download/v0.8/Tracy-0.8.7z)

To enable Tracy profiling:
1. Open a command line window in a build folder (i.e. `dxvk-remix/_Comp64Release/`)
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
- [Rtx Options](/RtxOptions.md)
- [Terrain System](/documentation/TerrainSystem.md)
- [Unit Test](/documentation/UnitTest.md)
