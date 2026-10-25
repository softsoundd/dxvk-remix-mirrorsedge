# dxvk-remix

[![Build Status](https://github.com/softsoundd/dxvk-remix-mirrorsedge/actions/workflows/build.yml/badge.svg?branch=mirrors-edge)](https://github.com/softsoundd/dxvk-remix-mirrorsedge/actions/workflows/build.yml)

dxvk-remix is a fork of the [DXVK](https://github.com/doitsujin/dxvk) project. On this branch, Mirror's Edge is presented as the game's own raster frame, with DLSS, frame generation, Reflex, and Remix post effects on top.

Thanks to all the contributors to DXVK for creating this foundational piece of software, on top of which we were able to build the RTX Remix Runtime.

While dxvk-remix is a fork of DXVK, please report bugs encountered with dxvk-remix to this repo rather than to the DXVK project.

dxvk-remix also contains a subproject in the `bridge` folder, which enables 32 bit games to communicate with the 64 bit dxvk-remix runtime.

## NGX passthrough branch (`mirrors-edge-ngx`)

Mirror's Edge renders through its own raster pipeline on this branch (no path tracing). The runtime adds DLSS Super Resolution/DLAA, DLSS Frame Generation, DLSS 3D-Guided Neural Rendering (DLSS-NR), Reflex, and Remix post effects (motion blur, chromatic aberration, vignette) on top. NIS, TAA-U, and XeSS are available as alternative upscalers. Like the rest of this fork, NGX passthrough may work in other UE3 titles, but support is not guaranteed.

### Setup

**The game's MSAA must be off** because depth cannot be resolved while it is active (plus you want DLSS/DLAA anyway, right?). The runtime forces it off by default (`rtx.ngxPassthrough.disableGameMsaa`). In the rare case that fails, disable MSAA in the game's settings or config files and restart.

Press Alt + X to open the Remix menu. Upscaling, frame generation, neural rendering, and Reflex are under User Graphics Settings > General, and post effects and V-Sync are on the Graphics tab.

How it works:

- The upscaler runs on the game's HDR scene colour before its post-processing, so bloom, tonemapping, and depth of field work on the anti-aliased image. If the runtime can't find a post-process pass, it upscales the finished frame before the UI is drawn.
- The runtime rebuilds depth and motion vectors from the game's depth buffer and camera, and adds true motion for moving objects, including skinned and CPU-animated meshes. DLSS, Frame Generation, DLSS-NR, and motion blur all use them.
- The upscaler's quality mode sets the game's render resolution through UE3's ScreenPercentage. Full Resolution with DLSS is DLAA.
- Frame Generation gets a copy of each frame without the UI, which reduces HUD warping. DLSS-NR runs on the finished frame before the UI.

Known issues:

- Ghosting issues with particles and textures on transparent planes (chain link fences). Will investigate solutions for this in the future.
- Separate existing issue with Mirror's Edge where sub-native render resolutions offset the position of some effects like sun haze and first person self-shadowing. This is not Remix's doing.

[RtxOptions.md](/RtxOptions.md) lists the passthrough options under `rtx.ngxPassthrough.*`. In the developer menu, Rendering > General has debug views for depth and motion vectors, plus a Diagnostics section.

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
        - `-BuildArch <x64|arm64|arm64ec>` — target architecture, default `x64`. `arm64ec` is configured with the Visual Studio backend rather than Ninja, because Meson cannot link arm64ec targets with Ninja, so the `_Comp*_arm64ec` directories hold MSBuild projects instead of Ninja files. The generated **_vs** solution always drives the `_Comp*_x64` directories; build other architectures with `meson compile -C <build directory>`.
    - Examples:
        ```powershell
        .\build_dxvk_all_ninja.ps1
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

## Project Documentation

- [Contributing Guide](/CONTRIBUTING.md)
- [Remix Runtime Binaries](/documentation/RemixRuntimeBinaries.md)
- [Rtx Options](/RtxOptions.md)
- [Unit Test](/documentation/UnitTest.md)
