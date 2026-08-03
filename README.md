# dxvk-remix

[![Build Status](https://github.com/NVIDIAGameWorks/dxvk-remix/actions/workflows/build.yml/badge.svg)](https://github.com/NVIDIAGameWorks/dxvk-remix/actions/workflows/build.yml)

dxvk-remix is a fork of the [DXVK](https://github.com/doitsujin/dxvk) project, which overhauls the fixed-function graphics pipeline implementation in order to remaster games with path tracing.

Thanks to all the contributors to DXVK for creating this foundational piece of software, on top of which we were able to build the RTX Remix Runtime.

While dxvk-remix is a fork of DXVK, please report bugs encountered with dxvk-remix to this repo rather than to the DXVK project.

dxvk-remix also contains a subproject in the `bridge` folder, which enables 32 bit games to communicate with the 64 bit dxvk-remix runtime.

## WIP fork containing Mirror's Edge/UE3 specific modifications

### 1) Mirror's Edge (UE3/D3D9) compatibility improvements

All UE3-specific behavior sits behind a single master `rtx.d3d9.ue3EngineMode` toggle which the Mirror's Edge game profile turns on automatically. The main differences from upstream:

- Camera and object transforms are read from UE3's reserved shader constants (CTAB parsing).
- Depth prepass, shadow depth, SceneCapture, and depth-test-disabled translucency passes are skipped so only real base-pass geometry gets ray traced.
- Texture and material identity is stable at the [MaterialInstanceConstant](https://docs.unrealengine.com/udk/Three/MaterialInstanceConstant.html) level: tags, categories and asset replacements survive texture streaming, settings changes, and restarts.
- Sampler UVs (tiling, panning, atlas tiles) are resolved, including UE3's distance fade based anti-tiling materials.
- Albedo selection is deterministic per material, with `rtx.preferredAlbedoTextures/rtx.neverAlbedoTextures` as overrides where albedo selection is missed. Textureless, constant colour materials supported too.
- Mid-frame fullscreen overlays (fades, scope/damage effects) cannot terminate the raytraced scene; they're replayed on top after RTX injection (`rtx.deferredUiTextures`).

### 2) Mirror's Edge/UE3 setup:

1. Enable the bridge's redundant state filtering. Create (or edit) `.trex\bridge.conf` and add `eliminateRedundantSetterCalls = True`.
> [!NOTE]
> UE3's D3D9 renderer doesn't filter redundant state on its own. Sampler and render state get resubmitted with nearly every texture bind, roughly 9 state calls per draw even when nothing's changed which can stack to tens of thousands per frame. Under Remix, each of those is handled twice where it gets serialised over the bridge IPC and then replayed by the runtime. This setting has the bridge client drop no-op state calls before they cross the process boundary. UE3 titles often run noticeably faster with it on.

2. Disable the game's lightmaps in its config file (`DirectionalLightmaps=False`) - this is easiest done with [Mirror's Edge Tweaks](https://github.com/softsoundd/MirrorsEdgeTweaks).
> [!IMPORTANT]
> Disabling lightmaps is strongly recommended, both for compatibility and especially when authoring assets. When lightmaps are enabled, Remix scene exports generate different material hashes that are not compatible with non-lightmapped states. As a result, assets authored with lightmaps enabled may not match correctly once lightmaps are disabled. If you want to keep lightmaps enabled for before/after comparisons, that is fully supported. As long as assets were originally authored with lightmaps disabled, enabling lightmaps later for comparison will not affect material hash matching.

3. Make a text file titled "remix" (no extension) in `<path-to-game>\Binaries` and paste the following set of commands:
```
scale set TdBicubicFiltering false
scale set TdTonemapping false
scale set MaxMultisamples 0
scale set MaxAnisotropy 0
scale set DynamicLights false
scale set DynamicShadows false
scale set AmbientOcclusion false
scale set Distortion false
scale set DropParticleDistortion true
scale set MotionBlur false
scale set DepthOfField false
scale set Bloom false
scale set LightEnvironmentShadows false
scale set LensFlares false
scale set FogVolumes false
scale set TdSunHaze false
scale set TdMotionBlur false
scale set Trilinear false
scale set UpscaleScreenPercentage false
scale set ScreenPercentage 100
scale set OnlyStreamInTextures true
toggleocclusion
ToggleDynamicContrast
viewmode unlit
show scenecapture
show dynamicshadows
show fog
```
> [!NOTE]
> The above commands ensures maximum compatibility with Remix. That being said, a lot of consideration has gone into this fork into ensuring that games with less flexibility around commands can still play somewhat nice with these graphics systems active, though game-side modding is recommended to disable them.

4. By default `MirrorsEdge.exe` whitelists only a select few launch arguments, so the above commands will not work out of the box. This can be fully unlocked with [Mirror's Edge Tweaks](https://github.com/softsoundd/MirrorsEdgeTweaks) via the launch argument patcher. Once patched, add `-exec=remix` into your game libray's launch arguments/other shortcuts, or alternatively within the launch argument field in [Mirror's Edge Tweaks](https://github.com/softsoundd/MirrorsEdgeTweaks) followed by launching via the `Launch Game w/ Args` button.
5. *(Optional)* UE3 employs frustum culling in native C++ land. This requires patching the executable to treat primitives as always visible. Doing this looks nicer compared to relying on Remix's anti-culling system, but note that performance will take a hit!
	- Use a hex editor to locate offset 008E3C6C and patch `0F 84 EE 06 00 00` to `90 90 90 90 90 90`. This has been tested against the GOG version only.

### 3) Acknowledgements
- sambow23 for their [physically based sky implementation](https://github.com/sambow23/dxvk-remix-gmod/tree/atmos).
- xoxor4d for their research into UE3->Remix support and other tidbits of info that helped guide the initial work around this.

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
