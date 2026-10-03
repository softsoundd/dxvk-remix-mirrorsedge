# Mirror's Edge / UE3 compatibility notes

Implementation notes and debugging guidance for the UE3-specific behaviour that sits behind `rtx.d3d9.ue3EngineMode`. See the [README](../README.md) for what the fork changes and how to set Mirror's Edge up.

## Exact vertex position capture

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

Hardware-instanced draws are the one case that never captures at all, whether or not the transform is recognised, because the capture buffer is indexed per vertex and cannot separate instances - see [Hardware-instanced mesh particles and foliage](#hardware-instanced-mesh-particles-and-foliage).

Diagnostics:

- `rtx.d3d9.ue3LogCapturePrecision` logs the resolved source per unique (vertex shader, vertex factory). On fallback it also dumps the instructions that produced each `oPos` component.
- `rtx.d3d9.ue3RequireExactVertexCapture` drops draws that would unproject, so distance-dependent distortion cannot occur, at the cost of losing whatever the exact paths miss.
- `rtx.d3d9.ue3VertexCaptureSourceOverride` forces one source for every draw. Toggling `0` (Auto) and `3` (Clip Reconstruction) on a long outdoor view is the A/B for the distortion.

The static vertex-capture cache (`rtx.d3d9.ue3StaticLocalMeshVertexCaptureCache`) reuses captured vertex data from an earlier frame for local meshes, decals and foliage that are not moving. Exact-source captures do not depend on the camera, so they stay valid across frames. It refuses reconstruction-sourced captures (those depend on where the camera was), camera-facing factories (sprites, billboards, leaf cards), morphing terrain, terrain in general (a hit would suppress the draw the terrain baker needs), and skinned draws.

The key includes the object transform and the shader constants other than camera registers, so only a draw that repeats both can hit. Skinned meshes put bone matrices in that hash and would mint a new key every frame, which is why they are refused; a moving prop does the same through its transform. A capture gets a device-local buffer only after its key has been seen on `rtx.d3d9.ue3StaticLocalMeshVertexCaptureCacheWarmupFrames` distinct frames. Until then it is a few bytes of CPU bookkeeping. `rtx.d3d9.ue3StaticLocalMeshVertexCaptureCacheBudgetMiB` caps retained bytes and evicts LRU first. `...RetentionFrames` is a staleness bound and `...MaxEntries` a backstop for many tiny captures. Expired entries are recaptured when the mesh returns to view, so a short window at a high frame rate will expire them during ordinary camera movement.

Some titles recompute a draw's transform every frame even for still geometry, so keys never repeat. Below `rtx.d3d9.ue3StaticLocalMeshVertexCaptureCacheMinReusePercent` the cache releases its buffers and key records, then re-tests every `...ReuseProbeFrames` frames. Set the threshold to `0` to disable the guard.

`rtx.d3d9.ue3LogStaticVertexCaptureCacheStats` reports entries, bytes, keys awaiting admission and reuse rate about once a second, including dormancy. Buffers appear under `RTXVertexCapture` in the memory profiler and HUD. If reuse is near zero, the keys are churning; raising the budget will not help.

## Hardware-instanced mesh particles and foliage

Vertex capture cannot describe a hardware-instanced draw. The injected code writes each member at `data[gl_VertexIndex - baseVertex]`, and D3D9 instancing replays the same vertex indices once per instance, so every instance writes the same slots with no ordering between them. The surviving positions are an arbitrary mix of placements: individual triangles end up with corners from different instances, which reads on screen as an exploded cluster of stretched geometry that lands somewhere different every time it is captured.

UE3 reaches this in two places, both of which put the placement in a vertex stream rather than a shader constant:

- Foliage (`FFoliageVertexFactory`, `FoliageComponent` / `UnTerrainFoliage`).
- Mesh particles (`FParticleInstancedMeshVertexFactory`), including the PhysX/NxFluid debris emitters (`ParticleModuleTypeDataMeshNxFluid`), which is what Mirror's Edge uses for simulated trash, paper and rubble. `FDynamicMeshEmitterData::RenderNxFluidInstanced` issues one `DrawMesh` for the whole cluster with `LocalToWorld = FMatrix::Identity`, and `FD3D9DynamicRHI::SetStreamSource` turns that into `SetStreamSourceFreq(0, D3DSTREAMSOURCE_INDEXEDDATA | N)` on the mesh streams plus `D3DSTREAMSOURCE_INSTANCEDATA | 1` on the instance stream.

Both compile `FoliageVertexFactory.usf`, whose world position is

```hlsl
float4x4 GetInstanceToWorld(FVertexFactoryInput Input) { /* InstanceXAxis/YAxis/ZAxis + InstanceOffset */ }
float4 CalcWorldPosition(FVertexFactoryInput Input) { return mul(GetInstanceToWorld(Input), Input.Position); }
```

so the mesh streams still hold one plain object-space copy of the mesh, and `InstanceOffset` plus the three basis axes arrive as `TEXCOORD1..4` (`FLOAT3`) on the instance-data stream.

`rtx.d3d9.ue3DecomposeInstancedDraws` (on by default) uses exactly that. Positions come from the input assembler rather than from capture, and one ray-traced instance is submitted per hardware instance with `objectToWorld = LocalToWorld * FMatrix(XAxis, YAxis, ZAxis, Location)`, byte for byte the transform the engine's own non-instanced fallback (`RenderNxFluidNonInstanced`) would have handed to `FMeshElement::LocalToWorld`. Nothing from the instance stream reaches the geometry, so a mesh's asset hash is the base mesh's and stays stable as particles move, spawn and die.

Because each instance becomes a real `RtInstance`, replacements, categories, anti-culling and motion vectors all work per particle. Instances whose basis is degenerate are dropped, since PhysX only writes the live prefix of an emitter's instance buffer and leaves the rest holding whatever was there before. This is also why decomposition submits a real draw call state per instance rather than filling `DrawCallTransforms::instancesToObject`: the GPU point-instancer path shares one `prevObjectToWorld` across a batch, which would give moving debris wrong motion vectors.

Because an instanced factory's world position is the instance transform times `Input.Position`, the declaration's `POSITION` is object space by construction. No shader constant is involved and there is nothing to prove about the transform. Decomposed draws therefore skip the conservatism `rtx.d3d9.ue3NativeLocalMeshVertexCapture` applies to `Local` draws, where reading the input assembler is a guess about what the shader does. What they need is a complete `TEXCOORD1..4` `FLOAT3` basis on an instance-data stream, a bound position buffer, and no skinning. A draw whose placements cannot be recovered is dropped with the reason logged, rather than rendered at the world origin. With no `LocalToWorld` constant in the shader, object-space positions without their placements would land nowhere useful.

These are also the only UE3 draws that skip vertex capture entirely, which makes them the first to reach the geometry interleaver's CPU path (taken below 1024 vertices when every input buffer is host-visible; a capture buffer is device-local, so a capturing draw always interleaves on the GPU). That path reads through `GeometryBufferData`, which reports a texcoord buffer as absent unless it is float32 because packed half floats cannot be read as `float2`, and UE3 packs its UVs as `FLOAT16_2`. `RtxGeometryUtils::interleaveGeometry` forces the GPU path for any texcoord format the CPU path cannot read, via `GeometryBufferData::isCpuReadableTexcoordFormat`. Without that, it hands the interleaver a null pointer.

### Dense clusters and instance identity

A dense cluster of *moving* instances is expensive, and not because of submission cost. `rtx.d3d9.ue3LogInstancedDrawStats` reports that separately and it is a small fraction of the total.

`DrawCallTracker::computeIdentityHash` includes the object transform. That is ideal for the static geometry which dominates a scene: a still object hits the exact-identity lookup every frame, but anything moving misses it and falls through to a spatial nearest-neighbour search. That search scans an eight-cell neighbourhood sized from `rtx.uniqueObjectDistance` (cells are twice it, so 600 units by default), so a pile a few metres across sits inside a single cell and every instance in it scans the whole pile. The cost is quadratic in the batch's size, which is why it appears abruptly as a batch grows rather than scaling with it.

`rtx.d3d9.ue3StableDecomposedInstanceIdentity` (on by default) removes that. Decomposed instances arrive in a stable order in the game's instance stream, so instance N of a batch can be named directly: `DrawCallState::decomposedInstanceId` (batch identity plus index) stands in for the transform in the identity hash, the exact-identity lookup hits every frame, and the spatial search never runs. Pairing is then exact rather than a proximity guess, which also makes the instances' motion vectors correct. `rtx.d3d9.ue3LogInstancedDrawStats` reports the order stability this rests on, and `rtx.logInstanceIdentityStats` reports the hit rate and the number of spatial candidates examined.

One invariant has to be restored by hand. An exact-identity hit normally proves the transform did not change, and `SceneManager`'s preserve path relies on that: it reuses surface state, transform included, whenever the dirty flags come back clear. For these keys `ReplacementInstance::LookupKey::identityExcludesTransform` is set, and the lookup then runs the dirty-flag comparison and moves the instance's spatial entry itself. This is inert for every other caller, whose transform genuinely is unchanged on such a hit. If it were ever broken the symptom would be unmistakable: instances frozen in place while everything else moves.

Naming the batch is the subtle half, and the instance buffer cannot do it: `RenderNxFluidInstanced` calls `RHICreateVertexBuffer` for a fresh one every frame unless its two-entry pool happens to hand one back, so its handle is not an identity. Batches are matched to the previous frame's by continuity of their own centroid, which holds because a batch as a whole barely moves even while its instances do. Records are claimed once per frame so two piles of the same mesh cannot collide, and retire after 120 unseen frames so a level change cannot leave one for a new batch to latch onto.

Lowering `rtx.uniqueObjectDistance` is not a substitute. It does shrink the scan, but it is global: dropping it far enough to subdivide a pile also stops camera-attached geometry (the view model, a force-shown player model) matching during fast turns, which then loses its temporal history every frame and forces fresh BLAS and instance work.

### Bounding a scatter that is simply too large

Cost is linear in instances once the above is in place, so the instance count is the remaining lever.

- `rtx.d3d9.ue3MaxDecomposedInstances` is a hard ceiling and the bound that behaves for a dense cluster. It keeps a fixed subset: the instances the game lists first, which is spawn order, so a batch thins out roughly evenly rather than losing one side of itself.
- `rtx.d3d9.ue3DecomposedInstanceCullDistance` (0, disabled) drops instances beyond a distance from the camera. It bounds what a far-off scatter costs while still on screen, but cannot help with a cluster you are standing in, where every instance is at much the same distance.

Both must keep the *same* set frame to frame, which is why the ceiling is not "the instances nearest the camera" even though that sounds kinder. A view-dependent subset changes as you move, so instances appear and disappear. Worse, since each is named by its position in the game's buffer, a selection that also reorders the survivors renames every one and costs it its history. `Ue3DecomposedInstance::sourceIndex` carries the buffer position through culling for exactly that reason.

### The two factories are not reliably distinguishable

"Foliage" in UE3 is not plants. `UFoliageComponent` is an instanced-static-mesh scattering system: one `InstanceStaticMesh` and `Material` plus an array of per-instance `Location`/`XAxis`/`YAxis`/`ZAxis`. Level artists use it for any small repeated prop. Its declaration is very close to the mesh particle one, including the `COLOR` element aliased onto `TEXCOORD0` that foliage emits when a mesh has no shadow-map coordinate.

The reference UE3 source does distinguish them by one element: `FFoliageVertexFactory::InitRHI` walks `{VEU_Tangent, VEU_Normal}`, so it emits `NORMAL`, while `FParticleInstancedMeshVertexFactory::InitRHI` walks `{VEU_Tangent, VEU_Binormal, VEU_Normal}` and `InitInstancedResources` fills only components 0 and 1. On that source the mesh particle declaration carries `TANGENT` + `BINORMAL` and no `NORMAL`, with the mesh's `TangentZ` arriving under `BINORMAL`. `ParticleInstancedMesh` matches that layout and reads the surface normal from `BINORMAL`; the stock game cannot, because its shader still declares `TangentZ : NORMAL`, so UE3 would render such particles with no normal at all.

Mirror's Edge's shipped build emits a real `NORMAL` (`s1:NORMAL0 UBYTE4 @4`) for its instanced mesh particles, so they are declaration-identical to foliage and classify as `Foliage`. `ParticleInstancedMesh` has not been observed to match in this title. Both types take identical position and placement paths, so the type only decides which element supplies the normal.

Do not use the classified factory type to tell foliage from particles; use the material. Every instanced batch observed in Mirror's Edge is dynamic PhysX debris whose placements change every frame: trash and paper scraps that spawn above a rooftop and fall into place on level load, and pebble scatters that can be pushed around.

### Diagnostics

- `rtx.d3d9.ue3LogInstancedDraws`: one line per instanced draw identity: instance count, per-stream frequency and dynamic usage, the declaration, the classified factory and pass, the resolved capture source, and whether a per-instance transform was recovered. For a decomposed draw the source reads `InputAssembler` and `instanceTransform` names the stream and byte offsets.
- `rtx.d3d9.ue3LogInstancedDrawStats`: per-frame instance counts, what each bound dropped, the wall time spent expanding them, and the instance-order stability `ue3StableDecomposedInstanceIdentity` rests on. A mean index-paired displacement on the scale of a batch's own extent would mean the game reorders its buffer, making that option unsound.
- `rtx.logInstanceIdentityStats`: where instance lookups land and how many spatial candidates they examine. This is what shows the quadratic scan appearing or disappearing.
- `rtx.d3d9.ue3TraceDrawTextureHashes`: a full `[UE3-DrawTrace]` dossier for draws binding a listed texture, including the object-to-world transform, the first few recovered instance transforms, and the asset and full geometry hashes. The asset hash is what replacements anchor on, so this is also how to confirm a mesh's hash is identical across two sessions.

## Diagnosing a cache that never hits

`rtx.d3d9.ue3LogVertexConstantChurn` samples up to `rtx.d3d9.ue3VertexConstantChurnMaxTrackedDraws` meshes by input-assembler identity and reports what changed when a draw that should be static returns: whether the IA identity came back at all (buffer handles, draw range, or content-generation counters), whether the multiset of instance transforms matched between completed frames (same count but different placements means they moved; a count change is culling), and whether any other vertex constant moved, named by CTAB symbol. Camera and transform registers are skipped in that last check. Raw `LocalToWorld` is compared with the extracted matrices so an extraction bug is visible separately from the game. The tracker keys per mesh, not per placement, so ordinary instanced translations are not reported as churn. Changes only while the view moves point at a camera-derived constant; changes with the view still point at animation or time. `ue3ExcludePlacementFromVertexShaderHash` can address placement churn only.

## Verifying the geometry hash memo

`rtx.d3d9.ue3StaticGeometryHashMemoization` serves geometry hashes and bounding boxes from a memo keyed on the input-assembler identity, so a buffer write that reached the geometry without refreshing that identity would serve a stale hash silently. Two things guard against it:

- Each buffer's content generation (`D3D9CommonBuffer::remixContentGeneration`, part of the memo key and of the static vertex-capture cache key) comes from one process-wide counter, at creation and on every non-readonly lock, so a buffer created at a freed buffer's address with the same slice length - routine under level streaming - never repeats a key an older entry still holds.
- `rtx.d3d9.ue3GeometryMemoSelfCheckFrames = N` (default 0) hashes every memo-eligible draw in full every N frames instead of serving it; the geometry worker compares each hash component (the per-draw `VertexShader` slot excepted) and the bounding box against the memoized entry and logs `[GeometryHashMemoCheck]` with the differing component names, first 20 mismatches. The fresh result replaces the entry. Any such line names a write path that bypasses `LockBuffer`.

## Placement constants and the geometry hash

Stock UE3 excludes `ViewProjectionMatrix` (`c0`) and `CameraPosition` (`c4`) from the stable VS hash. Other stock camera-derived vertex constants belong to factories the cache already refuses (e.g. Mirror's Edge reuses about 98% of eligible draws).

Other UE3 titles may upload a different `LocalToWorld` every frame for every draw when the view moves. Those registers sit in `HashComponents::VertexShader` and `rules::FullGeometryHash`, so `DrawCallCache::exactMatch` never matches across frames and a new `BlasEntry` is allocated per draw per frame. Asset and replacement hashes are unaffected (`rtx.geometryAssetHashRuleString` excludes `vertexshader` by default).

`rtx.d3d9.ue3ExcludePlacementFromVertexShaderHash` leaves the transform out of that hash. Bone registers stay in. It is off by default - enable it only when the churn log shows raw `LocalToWorld` registers differing on nearly every comparison. The shading-only constants (`LightMapScale`, lightmap/shadow coordinate scale-bias) are excluded by `rtx.d3d9.ue3EngineMode` regardless: they never reach a vertex position, and `LightMapScale` holds a different element count per lightmap policy, which would otherwise move the geometry hash with the `DirectionalLightmaps` setting.

## Mirror's Edge tonemapper and colour curves

The Mirror's Edge game profile selects this mode by default. Elsewhere, select "Mirror's Edge (UE3)" under Tonemapping in the developer menu, or set:

```
rtx.tonemappingMode = 2
```

This mode applies the game's TdToneMapping display transform to Remix's path-traced HDR: exposure (see [Exposure](#exposure)), the per-channel `SceneHighLights` / `SceneShadows` / `SceneMidTones` grade, desaturation and overlay, display gamma, and the per-map colour curves, matching the PC shader. Because the output is already display-encoded, the final sRGB pass skips its own conversion. Dithering still applies.

### Live capture

The game's curves and grade constants are captured live: with `TdTonemapping` on, the engine blends and uploads them every frame (e.g. curves blending off when entering certain indoor areas). The fork skips that fullscreen pass and forwards the captured data to the tonemapper, giving per-map curves, volume blends, and SKU-adjusted values. The `TdToneMapExposure` pass (the 1x1 auto-exposure draw, also skipped) is read as well: its constants carry the current volume's `Scene_ExposureManual`, `Scene_ExposureLow`, `Scene_ExposureHigh` and speed uploads, which feed the exposure meter.

With `TdTonemapping` off no capture arrives; the last capture is held (`rtx.tonemap.ue3.holdStaleCapture`) or the `rtx.tonemap.ue3.manual*` constants apply.

### Exposure

Post-process settings in Mirror's Edge are authored per area: the persistent level's `WorldInfo.DefaultPostProcessSettings` cover the open rooftops, and interiors sit inside `PostProcessVolume`s with their own settings, usually more neutral (identity curves, a small `Scene_HighLights` lift) and their own exposure clamps. The shipped meter is `sqrt(E) = clamp(sqrt(0.25 / L), Scene_ExposureLow, Scene_ExposureHigh)`, so `E` is limited to `[Low^2, High^2]`, and those clamps carry the authored intent: in the Prologue the rooftops run `Low = 0.5, High = 1.5` (`E` in `[0.25, 2.25]`) while the interior volume runs `Low = 0.8` with the default `High = 1.65` (`E` in `[0.64, 2.72]`), so the white city sits exposed down at its floor and the interior may rise to a ceiling 3.4 stops above it. A generic auto exposure cannot see those clamps, and no single exposure window reproduces both areas.

`rtx.tonemap.ue3.exposureModel = MirrorsEdgeMeter` (default) runs the game's own exposure model on Remix's radiance, in a small compute pass before the display transform:

- **Meter.** Remix's linear radiance is scaled into the game's scene units by `rtx.tonemap.ue3.exposureBias` (a scene calibration in EV), then averaged over a 256x256 grid of samples with every channel clamped at 1.0 first, as the shipped fixed-point downsample chain does. Luminosity uses the shipped `0.3 / 0.59 / 0.11` weights; an exact black or NaN mean is replaced with 0.25 as shipped.
- **Key and clamps.** `sqrt(E) = clamp(sqrt(0.25 / L), Low, High)` with `Low`, `High` and `Scene_ExposureManual` from the live capture of the current volume (`rtx.tonemap.ue3.meterUseCapturedSettings`), or from `rtx.tonemap.ue3.manualExposureLow/High/Manual` (PostProcessVolume defaults 0.85 / 1.65 / 1.0) when no capture is available. The previous exposure is clamped into the current range as shipped, so entering an area with a different range snaps into it and then adapts.
- **Adaptation.** Faithful Luma's law rather than the shipped quadratic step: movement in stops at `rtx.tonemap.ue3.meterSpeedToLight` / `meterSpeedToDark` that eases in exponentially inside `meterTransitionStops` of the target, frame-rate independent. With `meterHonourLevelSpeeds` the speeds are scaled by the level's `Scene_ExposureSpeedUp/Down` relative to the engine caps (2.5 / 3.0), recovered from the captured `dt * min(speed, cap)` uploads divided by Remix's frame time; the PostProcessVolume defaults exceed the caps, so most areas run at full speed, and a zeroed speed holds the exposure as shipped.
- **Applied exposure** = `E * Scene_ExposureManual * exp2(exposureBias) * exp2(user brightness EV)`. The calibration is set once, on a reference exterior; every other area then follows its authored clamps.

`RemixAutoExposure` keeps `rtx.autoExposure.*` as the exposure source, with `exposureBias` as a plain bias.

| Option | Default | Effect |
| --- | --- | --- |
| `rtx.tonemap.ue3.exposureModel` | MirrorsEdgeMeter | MirrorsEdgeMeter = the game's meter with the captured per-volume clamps; RemixAutoExposure = `rtx.autoExposure` |
| `rtx.tonemap.ue3.exposureBias` | 0 | Scene calibration in EV (meter), or a plain bias (Remix auto exposure) |
| `rtx.tonemap.ue3.meterUseCapturedSettings` | True | Take `Low` / `High` / `Manual` from the capture; off or no capture = the manual values |
| `rtx.tonemap.ue3.manualExposureLow` / `High` / `Manual` | 0.85 / 1.65 / 1.0 | PostProcessVolume defaults |
| `rtx.tonemap.ue3.meterSpeedToLight` / `meterSpeedToDark` | 12 / 6 | Adaptation in stops per second, exposure falling / rising |
| `rtx.tonemap.ue3.meterTransitionStops` | 1.5 | Distance from the target inside which the adaptation eases in exponentially |
| `rtx.tonemap.ue3.meterHonourLevelSpeeds` | True | Scale the speeds by the level's `Scene_ExposureSpeedUp/Down` relative to the engine caps |

### Shipped behaviour

Facts about `TdToneMappingPixelShader.usf` that the transform reproduces (with `rtx.tonemap.ue3.faithfulLuma = False` it is the shipped shader verbatim):

- `saturate(Color * Exposure)` clips each channel at exposed 1.0 before the grade, so anything brighter is flat white and clipped colours shift hue (warm light toward yellow, blue sky toward cyan).
- `Common.usf` redefines `pow()` as `ClampedPow`, `pow(max(abs(x), 1e-4), y)`. The `abs` folds a base that `SceneShadows` pushed negative back to positive (the shipped shader does not NaN there), and the `1e-4` guard is the entire source of the shipped black floor: the midtone `pow` floors linear at `1e-4` and the gamma `pow` encodes it to `0.01`, i.e. `#020202`.
- Mirror's Edge maps its brightness slider midpoint to display gamma 2.0 (`SetGamma` is `GammaValue * 1.5 + 1.25`), not stock UE3's 2.2. The captured `GammaColorScaleAndInverse.w` is `1 / DisplayGamma` and is applied as is.
- The curve lookup's `15/16` is LUT addressing for the 16-texel point-sampled curve textures (segment = `floor(15x)`, texel 15 only at `x = 1`), not an output domain; nothing is divided back out after the lookup. The sampler filter the game bound (point in Mirror's Edge) is matched.
- Desaturation uses the game's `SceneScaledLuminanceWeights` verbatim; the manual constants path uses the shipped `0.3 / 0.59 / 0.11` weights.

### Faithful Luma

Path-traced radiance is unbounded where the original renderer clipped at exposed 1.0, so `rtx.tonemap.ue3.faithfulLuma` is on by default. It keeps the shipped grade, gamma and curve math and changes the shipped shader only where it loses information:

- **Range compression.** The shipped grade is applied without its clip, then the per-channel clip is replaced by a per-channel curve chosen with `rangeCompression`:
  - `Neutwo` (default) is the `x / sqrt(x^2 + 1)` family in RenoDX's display-peak / white-clip form, `f(x) = c * x / sqrt(x^2 * (c^2 - 1) + c^2)` with `c = neutwoWhiteClip`: slope 1 at black, near-identity at mid grey, compressing gradually from there and exactly white at `c` (100 by default, effectively the asymptotic curve; lower values give hot sources a true white sooner). `neutwoContrast` is RenoDRT's contrast, a power around mid grey applied to luminance before the curve with chromaticity kept. The curve spends its compression from mid grey up, which suits path-traced radiance: Remix's sky and lights run far past the 3x white the game's raster scene rarely exceeded.
  - `FaithfulLumaShoulder` is the reference shaders' curve, sized for the game's own range: identity below `softClipKnee` (0.8), slope-continuous at it, exactly 1.0 at `softClipWhite` (3.0), white above, so below the knee the image is the shipped image. All of its compression sits in the last two stops.

  Encoded 8-bit values at gamma 2.0 for the same exposed value:

  | Exposed | Shipped clip | Shoulder (0.8 / 3.0) | Neutwo (clip 100) |
  | ---: | ---: | ---: | ---: |
  | 0.18 | 108 | 108 | 107 |
  | 0.50 | 180 | 180 | 170 |
  | 1.00 | 255 | 242 | 214 |
  | 2.00 | 255 | 252 | 241 |
  | 4.00 | 255 | 255 | 251 |
  | 8.00 | 255 | 255 | 254 |

- **Hue.** A per-channel curve compresses the peak channel most and shifts hue the way the clip did, only less. The curve's value (peak channel) and saturation (floor over peak) are kept and its HSV hue is solved so the result's OKLab hue matches a target. Within one sextant this is the reference shaders' middle-channel solve; solving the HSV hue keeps it continuous where two channels cross, where a middle-channel solve clamps at the floor and jumps (banding on saturated colours). The target is the scene's hue turned by the Bezold-Brucke shift a dimmer rendering needs (`bezoldBruckePerStop`, degrees per stop the curve darkened the colour, toward yellow or blue by hue region) and, with `hueReference = ApprovedLook` (default), part of the way toward the hue the shipped clip gave the colour, weighted by the share of its luminance above display white that the clip could not show, discounted by the chroma the clip kept. `huePreservation` blends from the curve's own result (0) to the target (1). Neutrals are unchanged, and light sources still blow out to white as their floor channel climbs the curve.
- **Highlight desaturation** (`highlightDesaturation`, default 0). Over-range colours are desaturated toward their peak channel until their saturation is no higher than `min(graded, 1)` would have had, i.e. the chroma the shipped clip left them. At 1, a sunlit white with every channel over 1.0 is white as shipped rather than the colour of the light; saturated colours and anything at or below 1.0 are unaffected.
- **Black.** The shipped `1e-4` guard is dropped from the midtone and gamma `pow`, so black reaches code 0. Only encoded values below `0.01` change.

| Option | Default | Effect |
| --- | --- | --- |
| `rtx.tonemap.ue3.faithfulLuma` | True | Master toggle; off is the shipped shader verbatim |
| `rtx.tonemap.ue3.rangeCompression` | Neutwo | Neutwo = compresses from mid grey with asymptotic headroom; FaithfulLumaShoulder = identity to the knee, white at the white point |
| `rtx.tonemap.ue3.neutwoWhiteClip` | 100 | Neutwo: graded value that lands exactly on display white; 1 is the shipped hard clip |
| `rtx.tonemap.ue3.neutwoContrast` | 1.0 | Neutwo: power around mid grey on luminance before the curve |
| `rtx.tonemap.ue3.softClipKnee` | 0.8 | Shoulder: graded value where it starts; identity below |
| `rtx.tonemap.ue3.softClipWhite` | 3.0 | Shoulder: graded value that reaches display white |
| `rtx.tonemap.ue3.huePreservation` | 1.0 | 0 = the curve's per-channel hue shifts, 1 = the target hue |
| `rtx.tonemap.ue3.bezoldBruckePerStop` | 1.5 | Degrees of OKLab hue per stop of curve darkening; 0 disables |
| `rtx.tonemap.ue3.hueReference` | ApprovedLook | Scene = the scene's hue; ApprovedLook = turned toward the clip's hue by the luminance share the clip could not show |
| `rtx.tonemap.ue3.highlightDesaturation` | 0.0 | Desaturate over-range colours to the chroma the shipped clip left them |

## Deferred overlays

UE3 draws its `MaterialEffect` overlays (damage and health effects, the scope, reaction time) mid-frame, before the frame's 3D rendering has finished, so they must not trigger RTX injection. Tag the scene colour render target they sample in `rtx.deferredUiTextures`, or their pixel shaders in `rtx.d3d9.deferredUiPixelShaders`; the draws are then captured and replayed once injection fires.

Where a captured overlay replays depends on the render target the game drew it into:

- **Floating-point target** (the HDR `SceneColor`): the game ran the overlay on linear scene colour, before its display transform (`TdToneMapping`, or the `GammaCorrection` pass with `TdTonemapping` off). With `rtx.d3d9.deferredUiHdrReplay` it replays on Remix's linear HDR image, in the game's scene units, and the Mirror's Edge display transform runs after it, as the game's did.
- **8-bit target**: the game drew the overlay after its own tone mapping, so it replays on Remix's tone-mapped output.

The one-shot `[RTX-DeferredUI] Deferring overlay draw` log line reports each overlay's `target` format and `domain`.

Replaying a linear overlay on the display-encoded image runs its maths in the wrong space. At display gamma 2.0, a colour gain `k` applied in linear light shows as about `sqrt(k)` on screen, but applied to the encoded image it shows as `k`. It also lands after the range compression, which pushes compressed highlights straight back to 1.0: a tint lifts the whole frame and clips near-white surfaces.

| Option | Default | Effect |
| --- | --- | --- |
| `rtx.d3d9.deferredUiReplay` | True | Replay tagged overlays; off suppresses them while ray tracing |
| `rtx.d3d9.deferredUiHdrReplay` | True | Replay overlays drawn into floating-point targets on the linear HDR image before tone mapping; off replays every overlay on the tone-mapped output |
| `rtx.d3d9.deferredUiRefreshSceneColor` | True | Before each replayed draw, copy the image the overlay composites over into the scene colour textures it samples |

### How the replay is staged

When scene-linear overlays are pending, the injection runs in two stages with the replay between them:

1. `RtxContext::injectRTX` is given an HDR canvas: an `A16B16G16R16F` D3D9 render target the size of the injection target. It renders the frame through motion blur, scales `m_finalOutput` into scene units (`exp2(rtx.tonemap.ue3.exposureBias)`, Mirror's Edge tonemapping mode only), copies it into the canvas, and stops before tone mapping.
2. The D3D9 layer replays the scene-linear overlays onto the canvas. The scene colour refresh copies the canvas into the textures they sample, as UE3's resolve would.
3. `RtxContext::finishInjectRTX` copies the canvas back, undoes the scale, and runs tone mapping through the blit to the target.
4. Overlays drawn into 8-bit targets replay on the target, and the game's UI draws follow.

Frames without scene-linear overlays inject in a single stage. A frame that ray traces nothing (invalid camera, shaders still compiling) passes the target through the canvas unchanged, so its overlays land where they would without staging. The exposure meter runs in the second stage, so it meters the image with the overlays applied.

## UE3 lightmaps are bypassed

Remix does the lighting, so a raytraced surface has no use for UE3's baked lightmaps. Under `rtx.d3d9.ue3EngineMode` the coefficient textures are stripped from the draw before anything reads it, and material identity is independent of the lightmap policy the engine compiled: completely for the textures, and for all but a minority of materials' identities.

### Textures

A lightmap is recognised from the pixel shader's own CTAB sampler names (`LightMapTextures`, and Mirror's Edge's `BSplineTexture` bicubic weights LUT), so per-level hashes never have to be tagged by hand. Those sampler stages are subtracted from the draw's texture set up front, which keeps them out of albedo scoring, both `colorTextures` slots, the diffuse-selection cache key and its area sum, and the per-texture material spread. A score penalty would not have been enough: `DirectionalLightmaps=True` binds three coefficient samplers where `False` binds one, so leaving them in changes the candidate pool and can flip the albedo pick between the two settings. Discovered hashes also join a session-local set that the generic `rtx.lightmapTextures` call sites consult (hash preservation on CPU writes, the terrain baker's stage filter, the texture picker), so those paths behave as if you had tagged them. `rtx.d3d9.ue3AutoDetectLightmapTextures` turns the detection off; nothing is ever written to a config.

Vertex lightmaps have no samplers at all. They arrive as an extra vertex stream of packed coefficients declared as `TEXCOORD5` (simple) or `TEXCOORD5/6/7` (directional), typed `D3DCOLOR` rather than a float pair. A `D3DCOLOR`-typed `TEXCOORD` is never a UV set in UE3, so those elements can never be chosen as the surface's texcoord buffer.

Subtracting the whole sampler set leaves some draws with no albedo at all: a surface lit by nothing but a lightmap has no material texture to fall back on. Those render at `rtx.legacyMaterial.albedoConstant`, which is the intended result since Remix relights them, but they have no colour of their own for the Toolkit to show. `[RTX-Compatibility][UE3-NoAlbedo]` reports the sampler picture that explains why, and the capture exports the material carrying that same constant rather than skipping it - a skipped material leaves the mesh bound to a prim the stage never declares, which the Toolkit can neither select nor author.

Where such a material does carry a colour it is in a `UniformVector_*` register, and the value is a tint UE3 multiplied against the lightmap for brightness rather than a finished albedo. Used raw once the lightmap is gone it reads as near-black, and since a register at or below `0.01` is rejected so a later one can supply the real colour, the bottom of a ramping tint falls back to the legacy constant and the surface would step straight from there to a dark tint. `rtx.d3d9.ue3ConstantAlbedoTintGain` closes both gaps: the register's brightest channel times the gain, clamped to 1, is the weight blending the legacy constant towards the register's fully saturated hue, so a ramp fades continuously.

Textured materials are not tinted this way, because the albedo texture replaces the constant rather than modulating it. The one tint the game fades on textured surfaces, Mirror's Edge's Runner Vision highlight, is reproduced separately and per surface; see [Runner Vision](#runner-vision).

### Identity

UE3 compiles one base-pass pixel shader per lightmap policy from the same generated material HLSL. `DirectionalLightmaps=False` sets `SIMPLE_LIGHTING`, which deletes the whole `LightMapBasis` transfer block from `BasePassPixelShader.usf`; fxc then dead-strips every material symbol that only fed it: the normal map, the specular colour and the specular power. `FNoLightMapPolicy` (movable geometry, translucency, and dynamic meshes under `viewmode unlit`) folds the lightmap to zero and loses the same block. An identity built from "whatever the CTAB declares" therefore moves with a system setting.

The rule is to exclude, in *both* compiles, any material input the base pass only consumes as a lighting input. Where the directional compile declares such an input it is recognised and dropped; where fxc already stripped it there is nothing to drop, so the two sets agree without the runtime having to know which compile it is looking at. The test has to be a property of the input's own use, not of a symbol that exists in one compile only, and that is what keeps it symmetric.

The classification is a colour-term analysis of the pixel shader (`src/dxso/dxso_color_terms.cpp`). It follows every material sampler, every `UniformVector_*` register and every `def` literal component through the arithmetic to `oC0`, per register lane (fxc packs unrelated scalars into the spare lanes of live registers), and records for each additive term that reaches the colour output which classes of factor it was multiplied by on the way: a lightmap sample, a lighting constant (`AmbientColorAndSkyFactor`, the sky colours), a vector read of a `TEXCOORD` interpolant that is neither a texture coordinate nor normalised (a vertex lightmap coefficient), a value derived from normalising such an interpolant (the camera or sky vector), and the specular transfer coefficient - a view-derived vector dotted with UE3's `LightMapBasis` literals. Dot products with anything but plain literal weights, normalisation and matrix transforms end a colour term (the value has become direction or coefficient data); dot products with plain literal weights are UE3's `Desaturation` and keep it. Separately from the terms, the analysis records whether a source reaches the opacity output (`oC0.a`, `texkill`) or a texture coordinate at all.

Those terms decide the role of each input, and the rules are what make the two compiles agree:

- **Colour**: some term reaches `oC0.rgb` on a path both compiles have - unlit (emissive), or lit without passing through the specular transfer. Kept.
- **Specular**: every colour term carries the specular transfer coefficient and a lightmap, lighting constant or vertex lightmap factor. That is `LightMap * SpecularTransferCoefficients * SpecularColor`, which `SIMPLE_LIGHTING` deletes. Dropped, whether the input is a specular texture or a `UniformVector_*`. A Fresnel-weighted diffuse blend or reflection is view-dependent too but never passes through the basis transfer, and stays colour; the transfer, not view-dependence, is the discriminator.
- **Transfer-only**: every colour term is a lightmap or vertex-lightmap product and none reaches the output through a lighting constant or unlit. In the directional compile every diffuse input also carries `DiffuseColor * AmbientColor`, so an input that never does is a transfer-coefficient parameter - `TwoSidedLightingMask`. Dropped, but only in a shader that has a lighting-constant term at all: the simple compile has none, and never declares the mask either.
- **Opacity**: no colour term, but the value reaches the opacity output (`oC0.a`, `texkill`). Exists in every permutation, and is often all that separates a material from another sharing its colour inputs: a masked plastic sheet and a bare glass pane that both colour themselves from one reflection cube map hash alike without the mask, and an override authored on one lands on the other. Kept.
- **Coordinate**: no colour term, but the value reaches a texture coordinate - a bump-offset map, or a normal map perturbing a cube lookup. Policy independent as well, but nearly every reflective material has one, so keeping them would re-mint most identities to separate very few; a sampler in this role is kept only when it is all the material has. A `UniformVector_*` in this role is a texture transform and dropped like a volatile register, which also retires the rotator matrices the volatile scan could not see and whose animation made those families unanchorable.
- **Unused**: reaches no output at all as a value - a normal map consumed by the lighting transfer. Dropped.

The canonical signature is used for *every* UE3 material rather than only lightmap-bearing ones, so a material drawn on static geometry and on a movable prop shares one anchor. Samplers dropped as specular or unused also leave the albedo candidate pool, so the directional compile cannot pick a normal or specular map the simple compile never declared. Where the bytecode cannot be analysed (ps_1_x), the earlier heuristics apply: a proven normal unpack, a name that marks a non-diffuse input, or a sampled value that never reaches `oC0` as colour.

A shader left with no identity-bearing sampler - constant-colour and lightmap-only materials, and materials whose only textures are lighting inputs - is signed from what else survives every permutation: its kept `UniformVector_*` names and the `def` literal components that reach the colour output unlit. The lightmap epsilon, the basis vectors, the normal unpack constants and the specular exponent never do; an emissive constant always does. `rtx.d3d9.ue3LogMaterialInstanceHash` marks these seeds `(canonical textureless, lightmap-permutation invariant)`. Only a shader with neither keeps its bytecode hash as the seed, and with it the policy dependence. Structurally identical textureless materials share one `textureSet+shader` family tier under this seed; their constants still separate the final hashes. `rtx.d3d9.ue3TexturelessIdentityFromBytecode` restores the bytecode seed for this class wholesale, for content whose textureless anchors have not yet been re-anchored.

The `LightMapBasis` literals are used for exactly one thing: qualifying a view-derived vector's dot product as the specular transfer. An earlier implementation tainted every value derived from them and pruned whatever the taint reached; because the literals exist in one compile only, it fired on the lightmapped shaders under `DirectionalLightmaps=True` and on nothing under `False`, removed samplers the other compile keeps, and manufactured the divergence it was meant to remove. The transfer bit has the opposite shape - it only ever excludes what `SIMPLE_LIGHTING` deletes - but any broader rule seeded on a one-compile symbol still has the earlier one's.

### Constants tier

Only `UniformVector_*` parameters contribute (`rtx.d3d9.ue3MicConstantIdentity`, on by default), minus those the colour-term analysis classifies as specular, transfer-only, coordinate or unused, and minus the volatile registers. The `UniformScalar_*` class is excluded wholesale rather than analysed: `DiffusePower` exponents the lightmap under `SIMPLE_LIGHTING` and a basis-derived transfer coefficient otherwise, while `SpecularPower` is structurally identical yet present in only one compile. The vectors carry the tint UE3's colour variants are told apart by - `T_RooftopPropsClusters_DA` and its Blue/Orange/Yellow siblings are one texture set differing only in that parameter, and anything that drops it merges them.

### Coverage

Measured on the opening views of several levels with `rtx.d3d9.ue3LogMaterialInstanceHash`, matching materials between runs on their bound images, every material keeps one identity across all four combinations of `DirectionalLightmaps=True`/`False` and `TdBicubicFiltering` on/off. The `[RTX-Ue3Identity]` summary lists each shader's kept and excluded inputs.

Identity is one half of it; the *albedo pick* has to agree too, since the Remix UI, the texture-tier tags and the displayed colour all follow the texture the runtime chooses (see [Albedo selection](#albedo-selection-and-the-texture-spread-cache)). `ue3LogAlbedoSelection` is what to compare for that.

What remains outside the analysis, in decreasing order of likelihood: a `TwoSidedLightingMask` parameter on a material whose diffuse is folded to zero (no ambient term to gate the transfer-only rule); a `MATERIAL_LIGHTINGMODEL_CUSTOM` material whose custom lighting dots its inputs with the basis; a `DiffusePower` texture. None has been observed.

If you change any of this, measure it the same way: match materials between runs on their **bound images**, allowing one side to be a subset of the other, and count the materials that differ. Every identity-derived field (the texture set, the canonical seed, the material's primary texture) moves when the sampler set moves, so joining on any of them drops exactly the materials being counted and can under-report the divergence by an order of magnitude. The unit test `tests/rtx/unit/test_dxso_color_terms.cpp` covers the instruction shapes fxc emits for each policy, and invoked with a `.dxso` file from `DXVK_SHADER_DUMP_PATH` it prints that shader's classification, which is how a new divergence is diagnosed without a second game run.

Turning `rtx.d3d9.ue3MicConstantIdentity` off makes identity shader + texture set alone, at the cost of merging every instance that shares a texture set onto one anchor. Mirror's Edge tints many of its variants from one set, so this collapses a substantial fraction of them.

Any change to what feeds identity re-mints the affected material hashes, and `mat_<hash>` prims authored against the old ones stop matching. See [Re-anchoring after an identity change](#re-anchoring-after-an-identity-change).

## Runner Vision

Mirror's Edge highlights the geometry the player can use - Runner Vision, `LOI` in the game's own code - by fading a material parameter. When a highlight activates, `TdLOIAddOnObject` gives each element of the mesh a fresh `MaterialInstanceConstant` parented to the material it had, and fades its `LOI_Strength` scalar up to 1 and back down to 0. Enemy weapons about to strike use the same parameter. Nothing else about the draw changes: same shader, same textures, same vectors.

Every master material that supports it compiles the same network. The diffuse is `lerp(X, X * V, S)`, with `S` the strength and `V` a `UniformVector_*` colour, and most of them add an unlit `0.1 * S * lerp(...)` glow on top of the lit surface. UE3's material translator folds arithmetic on uniform inputs into one CPU-evaluated register, but never a `Lerp`, so the strength always reaches the shader as a raw `UniformScalar_*`. Remix samples the albedo texture itself and never evaluates that arithmetic, so without help the highlight does not exist for it; `rtx.d3d9.ue3HighlightTints` re-derives it.

### Proving the tint

`src/dxso/dxso_highlight_tints.cpp` evaluates the pixel shader symbolically, lane by lane, on the evaluator in `src/dxso/dxso_symbolic_eval.cpp` that the [fade analysis](#opacity-driven-fades) shares. Every register component holds a polynomial over the shader's inputs, with texture samples and the `UniformScalar_*` and `UniformVector_*` components as symbols. A value that involves no material parameter (lightmap filtering, normal and reflection math) is folded into a single opaque symbol, as is the result of anything non-polynomial, which keeps the expressions small. For each material scalar `S` and each output channel, the colour output splits into the part without `S`, `P0`, and the part linear in it, `Q`. `S` tints the channel towards a vector component `v` when every colour-bearing term `m` of `P0` that `S` fades out (`Q` holds `-m`) also appears in `Q` as `+m * v`, and every such term agrees on `v`. A term that fades towards nothing makes the channel a blend or a fade instead - a layer blend under UE3's `(1 - Emissive)` diffuse factor would otherwise pass as a tint towards the emissive colour - and a pair is only reported when all three channels are tinted.

Because the proof is algebraic it holds however fxc scheduled the lerp: `mad` pairs with the operands either way round, `lrp`, the colour multiplied in before or after, a brightness scalar ahead of the lerp, and a lerp whose register has unrelated math packed into its spare lane. That last case is common - fxc routinely writes a sky-vector dot product into `.w` of the register holding the tinted colour - and a register-granular proof loses the lerp there.

A strength that also scales an unlit copy of the tinted terms is the Runner Vision network by construction, and the coefficient of that copy is the glow. Where several lighting paths carry the tinted colour at different literal weights (the directional lightmap's transfer coefficients), the least attenuated one sets the colour's scale.

Enemy weapons are highlighted differently. Their materials (`M_Glock18`, `M_MP5K`, `M_Taser`) have `LOI_Strength` as their only parameter, and it never reaches the diffuse: the shader adds a multiple of `S` (2 to 10, by weapon) times one channel of a texture, a mask, to `oC0.r`, unlit, so the weapon flashes red before a strike. A strength that only adds an unlit `k * S * T` to one channel, with `T` a channel of a material texture, is proven as a glow-only pair: no tint, that channel's glow coefficient, and the texture and channel it reads. A strength that scales a texture into several channels is the material's own emission animating instead, which Remix leaves to the material.

The analysis runs once per shader, during the identity parse. Shaders that outgrow its expression budget are rim and fresnel networks with many scalar-weighted sums, never the Runner Vision lerp.

### Which tints count as a highlight

The proof identifies a tint, not what it is for. An authored "tint amount" parameter compiles to the same lerp, and a material instance can hold the strength of a Runner Vision master material at a fixed value, painting an object in the highlight colour whether Runner Vision is on or not. What sets the highlight apart is that it moves: `TdLOIAddOnObject` fades `LOI_Strength` on the material instances it creates, from 0 to 1 and back, while an authored strength stays wherever the artist left it. A tint and its glow therefore apply only once the strength has been seen moving (`rtx.d3d9.ue3HighlightTintRequireMotion`); one that holds still is the object's own colour, which Remix leaves to its material. Motion is tracked per object - material instance and placement - so an object painted in the highlight colour stays untinted when an identical one is highlighted; the material hash leaves `UniformScalar_*` constants out, so it names the instance whatever its strength. An object at a placement not seen before, as a moving one is every frame, follows its material instance instead, which is how a highlighted enemy weapon glows. A highlight first seen already at full strength tints once it starts to fade. `rtx.d3d9.ue3HighlightTintExcludedMaterials` names anything that still gets through.

### Rendering it

Per channel the tint is `lerp(1, V, S)` from the live registers, multiplied over every applying pair. The glow is its coefficient, per channel, times the strength times `rtx.d3d9.ue3HighlightGlowIntensity` (and `rtx.emissiveIntensity`), emitted as that fraction of the tinted surface colour, so it is textured like the surface. A glow-only pair instead glows the channel of its own texture that the shader reads; the draw binds that texture as the material's emissive texture, with emission left off. It is bound whatever the strength, so the material stays the same through a highlight, and a replacement material takes it when it authors no emission or emissive texture, so a replaced weapon glows like the original. The shader decodes it as the game samples it, sRGB or linear. The tint and glow travel on the draw's `RtSurface` rather than its material. A value that changes every frame of a fade would otherwise mint a new surface material per frame, and folding it into the material hash would move every replacement anchor on the surface for as long as the highlight lasts. The shader applies the tint right after the fixed-function stage ops and before linearisation, which is where the game's own lerp operates, and adds the glow to the emission; both happen in the full and the particle resolve.

Keeping it out of the material has one consequence that needs handling. The preserve path reuses a static instance's surface and material whenever its draw looks unchanged, and a highlight fading on a static mesh changes nothing it compares, so it would freeze the tint at whatever value it had at the last incidental dynamic update. That reads as a highlight that half-fades or never appears. The instance manager therefore also refreshes the tint from the draw on the preserve path; surfaces are uploaded every frame, so that is all it takes. The glow texture does change the material, so it is part of what the preserve path compares.

Living on the surface, the tint also applies over replacement materials and mesh replacements, which carry the draw's legacy state, so a replaced ledge still turns red. A capture exports the material untinted. On a textureless material the tint colour register is never taken as the surface's constant colour: it holds its value whether the highlight is on or not, and a material whose own colour is dark would otherwise render permanently red.

Not covered: translucent materials, since the tint is applied to opaque ones, and the inverted `lerp(X * V, X, S)` and replacing `lerp(X, V, S)` forms, which the shader cache shows no Runner Vision material using.

### Diagnostics

`rtx.d3d9.ue3LogHighlightTints` logs `[RTX-Compatibility][UE3-Highlight]` lines. Per pixel shader, it gives the proven pairs with their CTAB names and glow coefficients, the material scalars that reach the colour output without proving a tint, and the reason a shader could not be analysed. Per material, it reports the first time a tint applies, with the live strength, colour, tint and glow, the texture a glow-only pair glows and the hashes the exclusion list takes, and when a tint is held back because its strength has not moved. `rtx.d3d9.ue3HighlightDebugForceTint` forces a tint onto every UE3 surface, which checks the rendering half independently of detection. The unit test `tests/rtx/unit/test_dxso_highlight_tints.cpp` covers the instruction shapes above; invoked with `.dxso` files, or a directory of them, from `DXVK_SHADER_DUMP_PATH`, it prints each shader's pairs, and with `--summary` only the totals.

## Opacity-driven fades

Two kinds of fade never reach the path tracer on their own.

- **Particle colour.** A sprite emitter's colour modules (`ParticleModuleColorOverLife` and its alpha curve, `ParticleModuleColor`, ...) write each particle's colour into its vertices, and `ParticleSpriteVertexFactory.usf` hands it to the pixel shader as a `float4` interpolant: `TEXCOORD3` on SubUV sprites, whose `TEXCOORD1` and `TEXCOORD2` carry the second sub-image and the blend between them, and `TEXCOORD1` on plain sprites and beams/trails. It is not a `COLOR` element, so Remix's vertex colour stays white, and smoke the game fades in and out pops in at full strength and vanishes when it dies.
- **Material parameters.** A decal or translucent material can fade by a scalar parameter Kismet animates; the soot that spreads over the walls at the start of The Shard is a `BLEND_Modulate` decal lerping between white and the soot. UE3 never folds a `Lerp`, so the parameter reaches the shader as a raw `UniformScalar_*`, and Remix, which samples the texture itself, shows the decal at full strength throughout.

### Finding the fades

`src/dxso/dxso_material_fades.cpp` runs the lane-by-lane symbolic evaluation of the [highlight proof](#proving-the-tint), with the alpha lane tracked too and the particle colour's input register kept as its own symbol. A `max` or `min` against a literal bound outside [0, 1] is taken as the identity, as `_sat` already is: UE3 clamps every translucent opacity to [0, 15] that way, and folding the clamp would hide what it clamps.

The bytecode alone rarely settles whether a fade holds. UE3 folds constant expressions into `UniformVector_*` registers, so the soot decal's shader computes `1 - Soot + V0 + S * Soot * V1`, which fades out with `S` only while `V1` is white and `V0` black. The analysis therefore hands candidates to the draw. A **fade candidate** is a `UniformScalar_*`, or one component of a `UniformVector_*` (a fade can arrive as `(1, 1, 1, S)`), that the `oC0` lanes it reaches are affine in: each lane is `offset + M * slope`, with the other constant registers left live. A power of `M` above 1, or `M` inside a value the evaluation does not expand (`pow`, `rcp`, a clamp to a bound inside [0, 1], ...), rules it out. The colour lanes form one candidate, in which a lane `M` does not reach has to be a literal, so a per-channel tint never passes as a fade; the alpha lane forms another.

Each draw evaluates a candidate's polynomials with its constants. The candidate fades the draw when every lane deviates from the blend's rest value `K` by the same multiple of its slope, `(ratio + M) * slope`, and the lanes without a slope already sit at `K`. The draw is at rest at `M = -ratio`; a candidate at rest strictly inside (0, 1) would pass through the framebuffer's own colour and out the other side, and is no fade. Coverage is `(ratio + M) / (ratio + M_full)`, clamped, with `M_full` the end of [0, 1] furthest from rest: `M` for a fade from rest at 0, `1 - M` from rest at 1 (the soot), and 0 when every lane already sits at `K`.

The **particle colour** counts only in the shapes Remix's texture stage operations reproduce. It tints when each coloured term of a colour lane that carries the colour carries that lane's channel, once, and no other (fog terms never reach Remix's albedo and are ignored); its alpha scales the opacity when every term of `oC0.a` that carries it carries it once, whatever the other channels do; and it scales the colour, as UE3's additive blend mode folds the opacity into the colour, when every coloured term of `oC0.rgb` that carries it carries it once. The coloured terms that carry none of it are the use's residual, and the use holds for a draw only when its residual vanishes for the draw's constants; the vent smoke's tint holds only while its emissive `UniformVector_0` is black, for instance. An erosion threshold (`saturate((Tex.a - (1 - Color.a)) * k)`), an alpha passed through `pow`, or one channel reused for all three lanes is left alone.

### Rendering them

The particle colour is captured from the vertex shader output carrying that `TEXCOORD` (`D3D9RtxVertexCaptureData::colorOutputRegister`), exactly the interpolant the pixel shader reads, and becomes the draw's vertex colour. The stage operations become `Modulate(Texture, VertexColor0)` for the colour and the alpha as the draw's uses hold, and the baked-lighting normalisation of vertex colours is off, since the colour is a tint. A colour scaled by its own alpha is premultiplied in the capture, and one whose tint does not hold is captured white. A colour brighter than 1, common in fire and spark curves, is scaled down by its brightest channel to keep its hue in 8 bits. None of this needs the textures tagged in `rtx.particleTextures`, and none of it enters the material or geometry hashes.

A fade candidate is evaluated against the `K` that leaves the framebuffer unchanged under the draw's colour blend:

| Blend | Lane | Rests at |
| --- | --- | --- |
| `SrcAlpha / InvSrcAlpha` (UE3 Translucent) | `oC0.a` | 0 (1 for `InvSrcAlpha / SrcAlpha`) |
| `One / One` (UE3 Additive) | `oC0.rgb` | 0 |
| `SrcAlpha / One` | either | 0 |
| `DestColor / Zero` (UE3 Modulate) | `oC0.rgb` | 1 (0.5 for `DestColor / SrcColor`) |

The coverage of each candidate that fades the draw, counted once per modulator, multiplies into one factor that scales the surface's opacity and emission after its blend mode has derived them, in `opaqueSurfaceMaterialApplyTextureStageOps`, which the full interaction, the particle resolve and decals all go through. For Modulate this is the game's opacity exactly, `1 - lum(lerp(1, T, c)) = c * (1 - lum(T))` at coverage `c`, while the albedo keeps the decal's own colour rather than lightening towards white.

Like the highlight tint, the factor rides on the `RtSurface` rather than the material, so no hash moves and no anchor needs re-authoring; the preserve path refreshes it, since a fading static decal changes nothing else it compares; and it applies over replacement materials too, unless a replacement's own alpha state or a cutout category renders the surface opaque or cut out. For the same reason a replacement particle texture should keep the alpha of the texture it replaces: the particle alpha multiplies it, as it does in the game. Every fade applies whatever its parameter's value, so a material with a constant opacity parameter renders with it, as the game does; `rtx.d3d9.ue3MaterialFadeExcludedMaterials` opts a material out.

Opacity micromaps bake through `calcOpaqueSurfaceMaterialOpacity` directly, before the factor, so a decal baked while faded out cannot stay hidden once it fades in. Particles with animated vertex alpha bake from the texture alone: their capture buffers are device-local, so they never become CPU-built billboards, whose micromap key carries a vertex-opacity hash, and the instance-level key has none. Baking without it only ever over-estimates their opacity, and `OpacityMicromapHashSourceData::ignoresVertexOpacity` keeps those micromaps apart from ones baked with vertex opacity.

Not reproduced: colour-only lerps on Translucent draws, which change colour rather than coverage; fades by `1 - Color.a`, which no stage operation expresses; and modulators the shader also uses non-linearly, which are reported as unproven.

### Diagnostics

`rtx.d3d9.ue3LogMaterialFades` logs `[RTX-Compatibility][UE3-Fade]` lines. Per pixel shader: its candidates, its particle colour uses (`(conditional)` where a use has a residual), the material scalars the output is not affine in, and why a shader could not be analysed. A `Draw:` line once per colour texture, shader and outcome, so a fade's start, middle and end each show: the blend, the uses applied, each candidate's live value and coverage (`not at rest` where it does not fade the draw), and the hashes the exclusion list takes. A `Capture:` line names the vertex shader output a particle colour is captured from. `rtx.d3d9.ue3MaterialFadeDebugForceCoverage` forces a coverage onto every blended UE3 surface, which checks the rendering half independently of detection. The unit test `tests/rtx/unit/test_dxso_material_fades.cpp` covers the shapes above; invoked with `.dxso` files, or a directory of them, from `DXVK_SHADER_DUMP_PATH`, it prints each shader's candidates, with `--particle-color N` analysing `TEXCOORDN` as the particle colour (3 for SubUV sprites, 1 otherwise), and with `--summary` only the totals.

| Option | Default | Effect |
| --- | --- | --- |
| `rtx.d3d9.ue3ParticleVertexColor` | True | Reproduce the particle colour and alpha |
| `rtx.d3d9.ue3MaterialFades` | True | Fade draws by the material parameters they fade with |
| `rtx.d3d9.ue3MaterialFadeExcludedMaterials` | | Material, textureSet+shader or colour texture hashes never faded |
| `rtx.d3d9.ue3LogMaterialFades` | False | Log the candidates per shader and the outcome per texture |
| `rtx.d3d9.ue3MaterialFadeDebugForceCoverage` | -1 | Force this coverage onto every blended UE3 surface; negative disables |

## Albedo selection and the texture spread cache

A UE3 pixel shader binds several textures and nothing in the bytecode declares which one is the surface colour, so the runtime scores every sampler and picks a winner. One of the scoring signals is *material spread*: how many distinct pixel shaders have been seen sampling that texture. A texture used by one or two materials is that material's own albedo; a texture used by a dozen unrelated ones is a shared detail, grunge or tint sheet, and is penalised heavily so it cannot out-rank the real base map.

Spread is learned by watching draws, so it is the only scoring input that is not a pure function of the draw in front of it. It is persisted to `rtx-remix/ue3TextureSpread.cache` and **only the persisted value is scored against** - textures discovered during the current session raise the count on disk for next time but do not change any decision now. Two runs on the same cache file therefore reach the same pick for every material.

Every other scoring signal is a property of the draw itself. Size counts in mip steps rather than texel count, since the bound dimensions only say how far a texture has streamed in, and one doubling is worth less than any single structural signal. Where a sampler's UV origin is proven from bytecode, reading the primary `.xy` pair is preferred over the packed `.zw` pair, because on a static mesh set 1 is the secondary channel and the base map reads set 0. A sampler whose origin cannot be proven at all is penalised heavily, since the surface's texture transform comes from the winning stage alone. `rtx.d3d9.ue3LogAlbedoSelection` prints the per-sampler breakdown, marking these `UV0XY` and `NOUVORIGIN` alongside the `spread=` term.

The score also rewards a sampler whose coordinate is a material expression - a transform, an offset, an animated or wrapping coordinate (`UVXFORM`, `UVOFS`, `UVANIM`, `BLEND`). Those flags come from the UV inference, which tracks expressions per register, and UE3 packs the material UV and the lightmap UV into one interpolator (`TEXCOORD0.xy` and `.zw`), so whatever the shader does to the lightmap coordinate in the other lanes of a shared register was attributed to the material sampler too. That is harmless while the lightmap coordinate is read as-is; `TdBicubicFiltering` puts it through an offset and a `frc`, and in that compile alone the material samplers gained `UVOFS|UVANIM` and a different texture could win. The flags are therefore checked against the colour-term analysis, which follows only the lanes the sampler reads: a flag the lanes cannot carry is cleared, never added, and `ue3LogAlbedoSelection` marks each removal `LANECLEARED:`.

The cache file is therefore part of the material's appearance, in the same way `rtx.conf` is. Ship it with a mod, and delete it only if you intend to relearn from scratch. To regenerate one: delete the file, play through a representative spread of levels, and exit the game normally (it is flushed on shutdown as well as periodically). Repeat until a pass adds no new entries - a texture's spread only counts shaders that have actually been drawn, so a single pass through one chapter will undercount anything reused later in the game. `rtx.preferredAlbedoTextures` and `rtx.neverAlbedoTextures` remain the per-texture override for anything the scoring still gets wrong.

## Persisted albedo picks

The winning sampler is pinned per material and persisted to `rtx-remix/ue3DiffuseSelection.cache`, so every session starts from the same pick. In memory the pin survives a level reload but not a relaunch, and a decision first made while a material's textures were still streamed down can differ from the settled one. A pin is still superseded once a larger set of mips arrives, so on a cold cache a surface can briefly show a different layer while a level pages in.

The file records the scoring version and the texture-tag set sizes it was written under, and is discarded when either changes - a different build's scoring, or an edited `preferredAlbedoTextures`/`neverAlbedoTextures`/`lightmapTextures`, re-derives rather than serving picks it would no longer make. If you change the albedo score, bump `kUe3DiffuseSelectionScoringVersion`: a bounded sample of loaded picks is re-scored each session and any disagreement is reported, so forgetting is noisy rather than silent.

The pin key is per pixel shader, so each lightmap-policy compile of a material pins its own pick. The compiles agree because their scoring inputs agree, not because the pin is shared; a permutation-invariant key (the material's canonical seed and name-keyed texture set, with the pick stored by sampler name rather than stage) would make the first decision authoritative for every compile, and is the next step if a residual difference ever shows up.

## Samplers whose UV origin cannot be proven

Not every sampler's coordinate can be traced back to an interpolant - screen-space, reflection-driven and some untraceable chains resolve as `originValid=0`. Upstream falls back to the fixed-function `D3DTSS_TEXCOORDINDEX` for those, which UE3 never sets meaningfully: it is leftover device state whose D3D9 default is "stage N reads texcoord N", restored on any device reset. A material whose albedo sits on sampler 2 would start reading IA texcoord set 2 after a reset or level reload having read set 0 before, which presents as the texture spontaneously rescaling.

So under UE3 the interpolant is borrowed from the shader's other material samplers when they unanimously name one - they shade the same surface, and their agreement is proven from bytecode rather than read from device state. Lightmap and engine samplers do not vote, since they legitimately read their own set. Only the origin is borrowed; the affine chain stays unresolved, because a sibling texture's tiling is not this one's. `rtx.d3d9.ue3LogUvResolution` marks these `[origin borrowed from sibling material samplers]`, and where the siblings disagree or none resolve, the legacy path still applies.

## One texcoord set per surface

A Remix surface carries a single texcoord buffer, and the UV set it uses is resolved from the winning albedo sampler. UE3 materials routinely sample two textures from *different* IA texcoord sets - a base map plus a decal, blend or overlay layer will read one each. Only the winner's set reaches the surface, so on such a material exactly one of the two textures can be mapped correctly; the other is drawn through the wrong channel and appears at the wrong scale.

No scoring threshold resolves this - it is one UV set for two demands. `rtx.d3d9.ue3LogUvResolution` identifies it: two lines for the same pixel shader with different `iaSet=` values means that material has the conflict, and the `stage=` on each says which texture claims which set. Choose which texture should be correct with `rtx.preferredAlbedoTextures`, and expect the other layer to be mapped wrongly.

## Rotators and UV matrices

UE3's `MaterialExpressionRotator` never appears by that name at D3D9. `CenterX` / `CenterY` / `Speed` and the optional `Time` pin are evaluated on the CPU each draw (default `Time` is `GameTime`; `Speed` is radians per second, with no 2π scale) and uploaded as a 2x2 matrix in a `UniformVector_*` register pair, recognisable in the logged values as `(cos,-sin)` and `(sin,cos)`. `Center` arrives as a compile-time origin, either `def` immediates around the rotation or a translation uniform `Origin - R·Origin`. `Coordinate` is whatever UV feeds the expression - `TexCoord0` by default, or a panner or tiled chain.

The pixel shader consumes that as a `dp2add` of the UV pair against each matrix row, or as the expanded `mul`/`mad`/`add` form `fxc` sometimes emits. Either way the UV tracer folds it into the same term model tiling and panners use, adding a `cross` coefficient for the sibling interpolant component, and resolves the live constants per draw into the surface's `textureTransform`. What stays unresolved is any row the model cannot express: a product of two draw-time constants (a Panner feeding a Rotator), a nested rotator, or a `Time` pin that varies per pixel and so compiles to `sincos` in the shader. Those keep a proven UV origin but apply no transform.

`rtx.d3d9.ue3LogUvAffineDetail` reports a resolved rotator with `exact=1 psResolved=1 applied=1`, its `ps=` and `final=` values as `(a,b,c,d,tx,ty)`, and the matrix registers named in the `ctab:` list.

## Material identity and replacement anchor stability

With `rtx.d3d9.ue3EngineMode`, a material's identity hash (the `mat_*` anchor that captures, texture tags, and asset replacements key off) is a chain: pixel shader identity → material texture set (image hash of every CTAB `Texture2D_*`/`TextureCube_*` sampler) → material constants (`UniformVector_*`). Every tier is a pure function of the draw *and of the shader*, decided before the first draw is hashed and never revised, so a material's hash is reproducible from the first frame of any session with no learned state behind it.

Getting there means each of the inputs UE3 offers that is not itself reproducible has to be recognised and left out:

- **Frame-varying constants** (panners, rotators, flipbook/sub-UV frames, time-driven fades, distance blends). UE3 re-evaluates every material uniform expression on the CPU per draw and writes the result into the same registers that carry authored `VectorParameterValues`, so at the D3D9 level an animation and a tint are the same kind of value. What separates them is where the value *goes*: a texture transform reaches a sampler's coordinate, a tint reaches the output colour. `rtx.d3d9.ue3MicVolatileConstantDetection` (on by default) takes the transitive set of constant registers a sampler's coordinate depends on and drops those from the constants tier, keeping everything else. `rtx.d3d9.ue3LogMaterialInstanceHash` reports the split per shader as `volatileRegs=[...]` and `uniforms kept=[...] excludedAsVolatile=[...]`.

  It has to be the dependency set rather than only the registers the affine resolver names, which are just the ones a *resolved* transform reads. A coordinate driven through math the resolver cannot express - a Rotator row that is a product of two draw-time constants, a matrix multiply, any chain through temporaries - would otherwise keep its registers in the identity and animate it. See [Rotators and UV matrices](#rotators-and-uv-matrices) for what the resolver does express. The colour-term analysis behind [Identity](#identity) closes what that scan still misses: a vector whose value only ever reaches a texture coordinate, or no output at all, is dropped whatever shape the math takes.

  The dependency set is tracked per register *lane*, by the same colour-term analysis, and it has to be: fxc packs unrelated scalars into the spare lanes of live registers, so a register-granular set counts a material tint that happens to share a register with coordinate math as driving that coordinate, and how much coordinate math there is to share with differs per compile (the bicubic lightmap filter is dozens of instructions of it). Dependent reads count (a value fetched through a coordinate moves with whatever moved the coordinate), as do the implicit rows of a matrix instruction; `def` literals do not, being part of the bytecode.

  The tracking follows only the operands an opcode actually reads, and reports none where the layout is not certain, so it errs towards leaving a register in identity. That direction matters: an identity that still churns says so through `[RTX-MicChurn]`, whereas one built from a stale operand would merge two anchors silently.

  Nothing about the material's appearance changes: the UV path reads those same registers live and still resolves the transform per draw. Only the identity declines to include them.

  The cost is that two materials separated *only* by such a register share one anchor. Neither had a reproducible identity to anchor beforehand, so little is lost, but note that a static UV tiling parameter is no longer a discriminator: a material whose only distinguishing feature is its tiling now shares its sibling's `mat_<hash>`.

- **Render targets bound as material samplers** (scene captures, reflection buffers). An RT's image hash embeds a creation counter and changes every respawn, checkpoint, or level load. RTs are excluded from identity by default (`rtx.d3d9.ue3MicExcludeRenderTargetsFromIdentity`).

- **Session-composited textures** (engine-generated textures reuploaded with different contents every session). Their content hash is session-unique, so any identity containing them cannot be anchored from a capture. Tag the texture's descriptor hash (stable across recreations; shown as `desc:0x...` in the `rtx.d3d9.ue3LogMaterialInstanceHash` breakdown and `[RTX-MicDrift]` sampler diffs) in `rtx.d3d9.ue3MicIdentityExcludedTextureDescHashes`, then re-anchor the material once. The sampler is then identified by that descriptor hash rather than dropped from the texture set: a texture whose contents are not reproducible still has a reproducible shape, and dropping it would be self-defeating on a material whose only sampler it is, since an empty texture set falls back to the primary colour texture's image hash - the value being excluded. Note that identically shaped textures share a descriptor, so the exclusion covers all of them and materials distinguished only by which of those they bind will merge.

  Alternatively, anchor the override at the raw texture hash (`mat_<textureHash>`): replacement lookup runs tiers material → textureSet+shader → texture, so a texture-tier anchor catches every material variant that selects that image as its albedo, while more specific anchors still win where present.

- **Mipped textures at or below the streaming-stable tail size**, where the whole chain is the tail. Identity is latched while mip 0 is being flushed, so a smaller mip the game has not written yet is an allocated buffer holding recycled memory, and that becomes permanent identity - visible as an image hash that changes on every level load while the descriptor hash stays put. Use `rtx.d3d9.ue3MicIdentityExcludedTextureDescHashes` on the affected descriptor; `rtx.d3d9.ue3LogTextureHashProvenance` reports the mip 0 hash separately from the full-chain one, which is what distinguishes this from a texture whose contents genuinely differ.

  Identifying such a texture by its top mip alone looks like the obvious fix and is not one. That size is also the state every larger texture passes through while streaming in, because UE3's minimum resident mip count lands there, and the whole point of the tail scheme is that the reduced variant hashes equal to the fully resident one. Break that equality and a replacement cannot bind until streaming finishes, which presents as an asset appearing unenhanced for a moment on every level load - the regression the tail scheme was introduced to remove.

- **Cube maps assembled face by face.** A cube map's hash is latched the first time it is set up for RTX and never recomputed, so hashing whichever faces happened to have CPU data at that moment makes the value depend on upload order - identifying a material one session and nothing the next. All six faces must be present before an identity is latched; waiting costs at most a rebind, since a face without data cannot be sampled yet. For a fully resident cube map the value is unchanged, so nothing anchored on one needs re-anchoring.

  The streaming-stable mip tail deliberately does not extend to cube maps. UE3's texture streamer iterates `UTexture2D` only and a cube map's faces skip `UpdateResource`, so a cube map never appears as a series of mip-count variants the way a 2D texture does - the tail would buy no stability while re-minting the identity of every material binding one.

### What the runtime cannot recognise, and how it tells you

A frame-varying expression that reaches the *output colour* rather than a coordinate - a time-driven fade or a pulsing tint - is genuinely indistinguishable from an authored `VectorParameterValue` in the bytecode, and no rule over dataflow separates them. Such a family still mints more than one identity.

`rtx.d3d9.ue3ReportMicIdentityChurn` (on by default, and free until a family actually churns) warns once per family, naming the registers whose values moved and printing the config line that pins it:

```
[RTX-MicChurn] Material family textureSetShader=0x... (ps=0x... seed=0x... tex=0x...) has minted 16 identities from its constants tier, so a register is very likely animating:
    c12 (UniformVector_3): (1,1,1,1) -> (1,1,1,0.42)
  Every material hash it mints is unanchorable. Pin it with:
    rtx.d3d9.ue3MicConstantIdentityExcludedMaterials = 0x...
```

It reports on a count rather than on the first disagreement, because a *second* identity on one texture set is also what a colour variant looks like the first time its sibling is drawn. Measured sibling sets run to about eight while an animating register passes any bound within a second, so a threshold clear of the former keeps the report to the families that actually cannot be anchored.

Identity is never altered as a consequence of the report. An exclusion that takes effect part-way through a session is the exact failure this design exists to remove - it splits the material's anchors either side of an unpredictable moment - so the fix belongs in a config the next session starts from. `rtx.d3d9.ue3MicConstantIdentityExcludedMaterials` is keyed on `textureSetShaderHash`, which is both the second replacement lookup tier and one material family, so the same value that pins the exclusion can anchor the override. Reach for `rtx.d3d9.ue3MicConstantIdentityExcludedShaders` only when every material on a shader is frame-varying: it is keyed on the shader, and one UE3 base-pass shader commonly serves dozens of families - excluding one seed on this content stripped the constants tier from 64 of them and merged six authored anchors that differed only by tint.

### Re-anchoring after an identity change

Any change to what feeds identity re-mints the affected material hashes, and `mat_<hash>` prims authored against the old ones stop matching. There is nothing to alias them to: for a material that was animating there were many old hashes, one per frame, which is why its anchor was unreliable to begin with, and where the change is to the texture set the old value was a function of data that is no longer reproduced at all.

So re-anchoring reads the new value rather than mapping the old one. `rtx.d3d9.ue3LogMaterialInstanceHash` prints one breakdown per material, and its `textures=[...]` field identifies which material is which by the images its samplers bind - match on the albedo texture and take the `materialHash`. A fresh capture works too, and is the better option when many materials moved at once. Two outcomes need a decision rather than a rewrite: two old anchors whose materials now share one identity (typically materials that differed only by an input the analysis now drops), and one old anchor whose material now splits into several (an input it now keeps differs between them), where the override belongs on each.

## Replacement anchor diagnostics

When an authored enhancement does not appear (or appears intermittently), enable `rtx.logReplacementResolution = True` for one session and reproduce briefly. The log names the failing material and the drifting identity tier directly:

- `[RTX-ReplacementResolve]`: how each material/mesh resolved against the mod's anchors (which lookup tier matched, or `NO MATCH`), plus a per-mod anchor dump at load.
- `[RTX-ReplacementFlap]`: a material family that previously matched stopped matching (or vice versa) mid-session, with old/new hashes for every tier.
- `[RTX-MicDrift]`: a material family minted a new identity, attributed to the tier that moved: per-sampler image hash diffs (with `desc:0x...` and RT flags) or changed constant registers with old/new values. This is the full-detail form of `[RTX-MicChurn]`, which is on by default and covers the constants tier only.
- `[RTX-MicRtPoisoning]`: a material identity still embeds a render-target image hash (only possible with RT exclusion disabled).
- `[RTX-MeshAnchorDrift]`: a mesh replacement key moved, attributed to its geometry part (unstable vertex data, e.g. CPU-morphed skinned meshes) vs its material part (mesh keys are `geometryHash XOR materialHash`).

`rtx.replacementDebugHashes` tracks specific hashes in detail (matched against texture, material, textureSet+shader, geometry, and mesh-key hashes) without the full-scene log volume. Toggling enhanced assets on/off in the UI intentionally shows up as synchronised matched/`NO MATCH` flaps with unchanged hashes.

## Direct sun through wrapping-mesh windows

UE3 interiors are often a one-sided exterior static-mesh shell around BSP rooms with window openings. Rasterisation, primary rays, and GI already cull the shell's inward backfaces, so you see sky. Shadow/NEE visibility rays do not (`rtx.enableCullingInSecondaryRays` is off), so the same shell blocks the sun.

The Triangle Culling (Override Secondary Rays) option could mitigate this, but that also globally culls backfaces on every opaque mesh and weakens prop/foliage/character shadows which is not desireable. Instead, tag the wrapping shell as **Cull Backfaces in Shadows** (`rtx.cullBackfacesInShadowTextures` / `rtx.cullBackfacesInShadowGeometries`; geometry hashes when materials are shared).

Optionally enable `rtx.d3d9.ue3AutoCullEnclosingMeshShadowBackfaces` to flag one-sided opaque meshes whose object AABB contains the camera and whose world-space extents fall between `rtx.d3d9.ue3AutoCullEnclosingMeshMinExtentMeters` (2 m) and `rtx.d3d9.ue3AutoCullEnclosingMeshMaxExtentMeters` (150 m). Tagging is more precise if auto misses a hangar or flags the wrong mesh. Do not tag thin floors or whole-level BSP, or rooms below can leak sun.

## Game executable patches

Two Mirror's Edge behaviours that matter to Remix are decided inside the game process, where the runtime does not run: under the bridge it is hosted by `.trex\NvRemixBridge.exe`, and the only Remix code in `MirrorsEdge.exe` is the bridge client (`d3d9.dll`). The client therefore patches the executable's code in memory, at the runtime's request. Nothing on disk changes.

| Option | Default | Effect |
| --- | --- | --- |
| `rtx.d3d9.ue3DisableFrustumCulling` | False (True in the Mirror's Edge profile) | The renderer stops culling primitives outside the view frustum, so off-screen geometry stays in the ray traced scene |
| `rtx.d3d9.ue3ShowThirdPersonModel` | False (True in the Mirror's Edge profile) | The third-person body and weapon (`Mesh3p`) also draw in first person, as player-model geometry |

Neither is requested outside `rtx.d3d9.ue3EngineMode`, nor while ray tracing is disabled, so the rasterised image Remix shows then is the game's own. Both are under Rendering > Mirror's Edge Game Patches in the developer menu, each with its status.

### Request and status

`D3D9Rtx::EndFrame` sends the wanted patches to the bridge client as a `GamePatchBits` mask whenever it changes (see `src/util/util_game_patches.h`), over the window-message channel that also carries `UWM_REMIX_UIACTIVE_MSG`. The client handles the request on the game's window thread: on the first one it locates both patches' sites, then it applies or reverts each patch to match the mask and answers with which patches are active and which it could not find. An answer sent before the channel's handshake completes is lost, so the runtime repeats its request every 2 seconds until the first answer arrives. The client logs the sites it finds and every change it makes to `bridge32.log`, prefixed `[GamePatch]`.

The sites are found by code signature, which matches the GOG, Steam, retail and DLC executables. The EA app executable is encrypted on disk, so its signatures are unverified; the client only scans it after it has decrypted. Every change is one or two bytes within an instruction, written by a single interlocked store, so a thread executing it fetches either the old instruction or the new one and never a mix. That is what lets both patches toggle while the rendering thread is drawing.

### Frustum culling

`FSceneRenderer::InitViews` tests every primitive with `View.ViewFrustum.IntersectSphere(Bounds.Origin, Bounds.SphereRadius)` and skips one that fails, before occlusion and view relevance are considered. The patch turns that skip, a `jz rel32`, into a short jump over its own operand, so every primitive carries on as if it had passed.

Primitives are still culled by distance (`CullDistance`, cull distance volumes), and occlusion culling is left to `toggleocclusion` and `rtx.d3d9.conservativeOcclusionQueries`. While the patch is active the Remix free camera sees the whole level, not just what the game's camera can.

### Third-person model

In first person `TdPawn.SetFirstPerson` and `TdWeapon.SetFirstPerson` set `Mesh3p`'s `bOwnerNoSeeWithShadow`, a DICE addition to `PrimitiveComponent` that both `Mesh3p` archetypes also default to true. Rather than hiding the component from its owner's views, which would also drop its shadow, four of the renderer's draw loops check the flag: they skip a primitive whose component has it set when the view's `ViewActor` is among its scene proxy's `Owners`. The proxy's own copy of the flag only decides whether `Owners` is collected, and the shadow passes draw the primitive regardless.

The patch zeroes the mask of each loop's `test byte ptr [reg+0xFC], 0x10`, so the test fails and the loop draws the primitive without looking for its owner. The flags stay as the game sets them, and only whether the renderer honours them changes, so the patch takes effect on the next frame either way. The body still needs the player-model tagging described in the README to stay out of the camera's view while it casts shadows and appears in reflections.
