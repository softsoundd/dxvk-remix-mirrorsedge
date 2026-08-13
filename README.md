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
- Vertex positions are captured from the register the game's vertex shader multiplies by ViewProjectionMatrix, not from unprojecting its clip-space output, so accuracy does not depend on view distance. See [Exact vertex position capture](#exact-vertex-position-capture).
- Depth prepass, shadow depth, SceneCapture, depth-test-disabled translucency, and fullscreen postprocess are skipped so only real base-pass geometry gets ray traced.
- Texture and material identity is stable at the [MaterialInstanceConstant](https://docs.unrealengine.com/udk/Three/MaterialInstanceConstant.html) level: tags, categories and asset replacements survive texture streaming, settings changes, and restarts.
- Sampler UVs (tiling, panning, atlas tiles) are resolved, including UE3's distance fade based anti-tiling materials.
- Albedo selection is deterministic per material, with `rtx.preferredAlbedoTextures/rtx.neverAlbedoTextures` as overrides where albedo selection is missed. Textureless, constant colour materials supported too.
- Mid-frame fullscreen overlays (fades, scope/damage effects) cannot terminate the raytraced scene; they're replayed on top after RTX injection (`rtx.deferredUiTextures`).
- First person geometry (arms, held weapon) is detected via UE3's `SDPG_Foreground` boundary (mid-scene depth-only clear) and classified as ViewModel, overriding player-model tags (`rtx.d3d9.ue3ForegroundDpgIsViewModel`).
- A "Mirror's Edge (UE3)" tonemapping mode (`rtx.tonemappingMode = 2`) reproduces the game's native tonemapping/colour curve display transform - exposure, per-channel highlights/shadows/midtones grade, display gamma 2.0 and the per-map 16-segment colour curves - on Remix's path-traced output, with hue-preserving modernisations as toggles. Per-map curves and grade constants are captured live from the game's (skipped) tonemap pass. See [Mirror's Edge tonemapper and colour curves](#mirrors-edge-tonemapper-and-colour-curves).

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
viewmode unlit
show scenecapture
show dynamicshadows
show fog
```
> [!NOTE]
> The above commands ensures maximum compatibility with Remix. That being said, a lot of consideration has gone into this fork into ensuring that games with less flexibility around commands can still play somewhat nice with these graphics systems active, though game-side modding is recommended to disable them.

4. By default `MirrorsEdge.exe` whitelists only a select few launch arguments, so the above commands will not work out of the box. This can be fully unlocked with [Mirror's Edge Tweaks](https://github.com/softsoundd/MirrorsEdgeTweaks) via the launch argument patcher. Once patched, add `-exec=remix` into your game libray's launch arguments/other shortcuts, or alternatively within the launch argument field in [Mirror's Edge Tweaks](https://github.com/softsoundd/MirrorsEdgeTweaks) followed by launching via the `Launch Game w/ Args` button.

5. Mirror's Edge hides the third-person player model in the default first-person camera state, which means Remix cannot cast shadows or reflect the character as you'd expect in a raytraced scenario. A modded TdGame.u game file is provided which always shows the third-person model regardless of camera state - this can be downloaded from the releases section, and goes into `<path-to-game\TdGame\CookedPC>`. Make sure you have the runtime's `rtx.conf` file which has the necessary player/viewmodel hashes pre-tagged so this renders properly.

6. *(Optional)* UE3 employs frustum culling in native C++ land. This requires patching the executable to treat primitives as always visible. Doing this looks nicer compared to relying on Remix's anti-culling system, but note that performance will take a hit!
	- Use a hex editor to locate offset 008E3C6C and patch `0F 84 EE 06 00 00` to `90 90 90 90 90 90`. This has been tested against the GOG version only.

### 3) Extra fork notes/debugging

#### Exact vertex position capture

Remix injects code at the end of every compiled vertex shader that writes each vertex's position into a capture buffer. That position is either the untransformed value the shader computed before the clip-space multiply, or an unproject of clip-space `oPos` through `inverse(projection)`, `inverse(view)` and `inverse(world)`.

Unprojecting is what distorts distant meshes. UE3 uses an infinite far plane (`clip.z = viewDepth - near`, `clip.w = viewDepth`), so the inverse projection's w row evaluates `(clip.w - clip.z) / near`: two numbers the size of view depth, subtracted to produce `near`. The reconstructed position is divided by that, and the error grows roughly as `viewDepth² · 2⁻²⁴ / near`. It is negligible up close, around 2 units at 20k depth and 15 at 50k with UE3's 10-unit near plane. The error is per-vertex rounding, not a constant offset, so it appears as shear that swims as the camera moves. No matrix precision can remove it, because the cancellation is already present in the `oPos` the game itself computed.

Reading the register back has no such term. Every UE3 vertex factory (`Local`, `GpuSkin` after skinning, `ParticleSprite`, `Foliage`, `Terrain`, `LocalDecal`, `SpeedTree`) goes through `BasePassVertexShader.usf`:

```hlsl
float4 WorldPosition = VertexFactoryGetWorldPosition(Input);
Position = MulMatrix(ViewProjectionMatrix, WorldPosition);
```

`WorldPosition` stays live because fog, `CameraVector` and `PixelPosition` also use it. The DXSO analyzer recovers it from the dataflow of `oPos` rather than a fixed instruction pattern, which `fxc` does not emit stably, then snapshots that register and takes it to object space through one well-conditioned affine inverse. Error stays around 0.002 units at any distance.

A draw uses this path (`rtx.d3d9.ue3ExactVertexCapture`) only when the transform is recognised and the CTAB names the matrix `ViewProjectionMatrix`. The analyzer proves the register is multiplied by a matrix to make `oPos`; the CTAB name is what proves it is a *world* position, not factory-local space. Shaders that adjust position after the transform are rejected.

If the transform is not recognised, conservative static `LocalVertexFactory` meshes use their input-assembler object-space positions (`rtx.d3d9.ue3NativeLocalMeshVertexCapture`), which is equally exact. Anything else unprojects.

Diagnostics:

- `rtx.d3d9.ue3LogCapturePrecision` logs the resolved source per unique (vertex shader, vertex factory). On fallback it also dumps the instructions that produced each `oPos` component.
- `rtx.d3d9.ue3RequireExactVertexCapture` drops draws that would unproject, so distance-dependent distortion cannot occur, at the cost of losing whatever the exact paths miss.
- `rtx.d3d9.ue3VertexCaptureSourceOverride` forces one source for every draw. Toggling `0` (Auto) and `3` (Clip Reconstruction) on a long outdoor view is the A/B for the distortion.

The static vertex-capture cache (`rtx.d3d9.ue3StaticLocalMeshVertexCaptureCache`) reuses captured vertex data from an earlier frame for local meshes, decals and foliage that are not moving. Exact-source captures do not depend on the camera, so they stay valid across frames. It refuses reconstruction-sourced captures (those depend on where the camera was), camera-facing factories (sprites, billboards, leaf cards), morphing terrain, terrain in general (a hit would suppress the draw the terrain baker needs), and skinned draws.

The key includes the object transform and the shader constants other than camera registers, so only a draw that repeats both can hit. Skinned meshes put bone matrices in that hash and would mint a new key every frame, which is why they are refused; a moving prop does the same through its transform. A capture gets a device-local buffer only after its key has been seen on `rtx.d3d9.ue3StaticLocalMeshVertexCaptureCacheWarmupFrames` distinct frames. Until then it is a few bytes of CPU bookkeeping. `rtx.d3d9.ue3StaticLocalMeshVertexCaptureCacheBudgetMiB` caps retained bytes and evicts LRU first. `...RetentionFrames` is a staleness bound and `...MaxEntries` a backstop for many tiny captures. Expired entries are recaptured when the mesh returns to view, so a short window at a high frame rate will expire them during ordinary camera movement.

Some titles recompute a draw's transform every frame even for still geometry, so keys never repeat. Below `rtx.d3d9.ue3StaticLocalMeshVertexCaptureCacheMinReusePercent` the cache releases its buffers and key records, then re-tests every `...ReuseProbeFrames` frames. Set the threshold to `0` to disable the guard.

`rtx.d3d9.ue3LogStaticVertexCaptureCacheStats` reports entries, bytes, keys awaiting admission and reuse rate about once a second, including dormancy. Buffers appear under `RTXVertexCapture` in the memory profiler and HUD. If reuse is near zero, the keys are churning; raising the budget will not help.

#### Diagnosing a cache that never hits

`rtx.d3d9.ue3LogVertexConstantChurn` samples up to `rtx.d3d9.ue3VertexConstantChurnMaxTrackedDraws` meshes by input-assembler identity and reports what changed when a draw that should be static returns: whether the IA identity came back at all (buffer handles, draw range, or content-generation counters), whether the multiset of instance transforms matched between completed frames (same count but different placements means they moved; a count change is culling), and whether any other vertex constant moved, named by CTAB symbol. Camera and transform registers are skipped in that last check. Raw `LocalToWorld` is compared with the extracted matrices so an extraction bug is visible separately from the game. The tracker keys per mesh, not per placement, so ordinary instanced translations are not reported as churn. Changes only while the view moves point at a camera-derived constant; changes with the view still point at animation or time. `ue3ExcludePlacementFromVertexShaderHash` can address placement churn only.

#### Placement constants and the geometry hash

Stock UE3 excludes `ViewProjectionMatrix` (`c0`) and `CameraPosition` (`c4`) from the stable VS hash. Other stock camera-derived vertex constants belong to factories the cache already refuses (e.g. Mirror's Edge reuses about 98% of eligible draws).

Other UE3 titles may upload a different `LocalToWorld` every frame for every draw when the view moves. Those registers sit in `HashComponents::VertexShader` and `rules::FullGeometryHash`, so `DrawCallCache::exactMatch` never matches across frames and a new `BlasEntry` is allocated per draw per frame. Asset and replacement hashes are unaffected (`rtx.geometryAssetHashRuleString` excludes `vertexshader` by default).

`rtx.d3d9.ue3ExcludePlacementFromVertexShaderHash` leaves the transform and shading-only constants (`LightMapScale`, lightmap/shadow coordinate scale-bias) out of that hash. Bone registers stay in. It is off by default - enable it only when the churn log shows raw `LocalToWorld` registers differing on nearly every comparison.

#### Mirror's Edge tonemapper and colour curves

Select "Mirror's Edge (UE3)" under Tonemapping in the developer menu, or set:

```
rtx.tonemappingMode = 2
```

This mode applies the game's TdToneMapping display transform to Remix's path-traced HDR: exposure (Remix auto exposure plus `rtx.tonemap.ue3.exposureBias`), the per-channel `SceneShadows` / `SceneHighLights` / `SceneMidTones` grade, desaturation, display gamma (Mirror's Edge authors around 2.0, not stock UE3's 2.2), and per-map colour curves matching the PC shader. Because the output is already display-encoded, the final sRGB pass skips its own conversion. Dithering still applies.

Pathtraced radiance is unbounded where the original renderer hard-clipped at scene white, so [FaithfulLuma](https://softsoundd.github.io/posts/faithful-luma-overview/) modernisations are enabled by default to correct this.

The game's curves and grade constants are captured live: with `TdTonemapping` on, the engine blends and uploads them every frame (e.g. curves blending off when entering certain indoor areas). The fork skips that fullscreen pass and forwards the captured data to the tonemapper, giving per-map curves, volume blends, and SKU-adjusted values.

#### Material identity and replacement anchor stability

With `rtx.d3d9.ue3EngineMode`, a material's identity hash (the `mat_*` anchor that captures, texture tags, and asset replacements key off) is a chain: pixel shader identity → material texture set (image hash of every CTAB `Texture2D_*`/`TextureCube_*` sampler) → material constants (`UniformVector_*`/`UniformScalar_*`). Every tier is a pure function of the draw, so the same material instance always gets the same hash when the inputs themselves are stable. UE3 games expose three unstable input classes, so the runtime deals with each:

- Render targets bound as material samplers (scene captures, reflection buffers). An RT's image hash embeds a creation counter and changes every respawn, checkpoint, or level load. RTs are excluded from identity by default (`rtx.d3d9.ue3MicExcludeRenderTargetsFromIdentity`). Anchors that still key off RT-bearing identities need re-anchoring once.
- Frame-varying constants (time/panner/fade/sub-UV expressions). Churn auto-exclusion drops such a group's constants from identity once it has minted enough distinct hashes. That exclusion is written to `rtx-remix/ue3MicAutoExcludedGroups.cache` (`rtx.d3d9.ue3MicPersistAutoExcludedConstantGroups`), so the group's identity is deterministic from the first frame of every later session instead of flipping mid-session at an unpredictable point. You can delete the cache file in the `rtx-remix` folder to reset this.
- Session-composited textures (engine-generated textures reuploaded with different contents every session). Their content hash is session-unique, so any identity containing them cannot be anchored from a capture. Tag the texture's descriptor hash (stable across recreations; shown as `desc:0x...` in the `rtx.d3d9.ue3LogMaterialInstanceHash` breakdown and `[RTX-MicDrift]` sampler diffs) in `rtx.d3d9.ue3MicIdentityExcludedTextureDescHashes`, then re-anchor the material once. Alternatively, anchor the override at the raw texture hash (`mat_<textureHash>`): replacement lookup runs tiers material → lightmap-permutation bridge → textureSet+shader → texture, so a texture-tier anchor catches every material variant that selects that image as its albedo, while more specific anchors still win where present.

#### Replacement anchor diagnostics

When an authored enhancement does not appear (or appears intermittently), enable `rtx.logReplacementResolution = True` for one session and reproduce briefly. The log names the failing material and the drifting identity tier directly:

- `[RTX-ReplacementResolve]`: how each material/mesh resolved against the mod's anchors (which lookup tier matched, or `NO MATCH`), plus a per-mod anchor dump at load.
- `[RTX-ReplacementFlap]`: a material family that previously matched stopped matching (or vice versa) mid-session, with old/new hashes for every tier.
- `[RTX-MicDrift]`: a material family minted a new identity, attributed to the tier that moved: per-sampler image hash diffs (with `desc:0x...` and RT flags) or changed constant registers with old/new values.
- `[RTX-MicRtPoisoning]`: a material identity still embeds a render-target image hash (only possible with RT exclusion disabled).
- `[RTX-MeshAnchorDrift]`: a mesh replacement key moved, attributed to its geometry part (unstable vertex data, e.g. CPU-morphed skinned meshes) vs its material part (mesh keys are `geometryHash XOR materialHash`).

`rtx.replacementDebugHashes` tracks specific hashes in detail (matched against texture, material, textureSet+shader, geometry, and mesh-key hashes) without the full-scene log volume. Toggling enhanced assets on/off in the UI intentionally shows up as synchronised matched/`NO MATCH` flaps with unchanged hashes.

### 4) Acknowledgements
- sambow23 for their [physically based sky implementation](https://github.com/sambow23/dxvk-remix-gmod/tree/atmos).
- xoxor4d for their research into UE3 → Remix support and other tidbits of info that helped guide the initial work around this.

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
