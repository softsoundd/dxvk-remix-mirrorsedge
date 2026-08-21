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

## Placement constants and the geometry hash

Stock UE3 excludes `ViewProjectionMatrix` (`c0`) and `CameraPosition` (`c4`) from the stable VS hash. Other stock camera-derived vertex constants belong to factories the cache already refuses (e.g. Mirror's Edge reuses about 98% of eligible draws).

Other UE3 titles may upload a different `LocalToWorld` every frame for every draw when the view moves. Those registers sit in `HashComponents::VertexShader` and `rules::FullGeometryHash`, so `DrawCallCache::exactMatch` never matches across frames and a new `BlasEntry` is allocated per draw per frame. Asset and replacement hashes are unaffected (`rtx.geometryAssetHashRuleString` excludes `vertexshader` by default).

`rtx.d3d9.ue3ExcludePlacementFromVertexShaderHash` leaves the transform out of that hash. Bone registers stay in. It is off by default - enable it only when the churn log shows raw `LocalToWorld` registers differing on nearly every comparison. The shading-only constants (`LightMapScale`, lightmap/shadow coordinate scale-bias) are excluded by `rtx.d3d9.ue3EngineMode` regardless: they never reach a vertex position, and `LightMapScale` holds a different element count per lightmap policy, which would otherwise move the geometry hash with the `DirectionalLightmaps` setting.

## Mirror's Edge tonemapper and colour curves

Select "Mirror's Edge (UE3)" under Tonemapping in the developer menu, or set:

```
rtx.tonemappingMode = 2
```

This mode applies the game's TdToneMapping display transform to Remix's path-traced HDR: exposure (Remix auto exposure plus `rtx.tonemap.ue3.exposureBias`), the per-channel `SceneShadows` / `SceneHighLights` / `SceneMidTones` grade, desaturation, display gamma (Mirror's Edge authors around 2.0, not stock UE3's 2.2), and per-map colour curves matching the PC shader. Because the output is already display-encoded, the final sRGB pass skips its own conversion. Dithering still applies.

Pathtraced radiance is unbounded where the original renderer hard-clipped at scene white, so [FaithfulLuma](https://softsoundd.github.io/posts/faithful-luma-overview/) modernisations are enabled by default to correct this.

The game's curves and grade constants are captured live: with `TdTonemapping` on, the engine blends and uploads them every frame (e.g. curves blending off when entering certain indoor areas). The fork skips that fullscreen pass and forwards the captured data to the tonemapper, giving per-map curves, volume blends, and SKU-adjusted values.

## UE3 lightmaps are bypassed

Remix does the lighting, so a raytraced surface has no use for UE3's baked lightmaps. Under `rtx.d3d9.ue3EngineMode` the coefficient textures are stripped from the draw before anything reads it, and material identity is independent of the lightmap policy the engine compiled: completely for the textures, and for all but a minority of materials' identities.

### Textures

A lightmap is recognised from the pixel shader's own CTAB sampler names (`LightMapTextures`, and Mirror's Edge's `BSplineTexture` bicubic weights LUT), so per-level hashes never have to be tagged by hand. Those sampler stages are subtracted from the draw's texture set up front, which keeps them out of albedo scoring, both `colorTextures` slots, the diffuse-selection cache key and its area sum, and the per-texture material spread. A score penalty would not have been enough: `DirectionalLightmaps=True` binds three coefficient samplers where `False` binds one, so leaving them in changes the candidate pool and can flip the albedo pick between the two settings. Discovered hashes also join a session-local set that the generic `rtx.lightmapTextures` call sites consult (hash preservation on CPU writes, the terrain baker's stage filter, the texture picker), so those paths behave as if you had tagged them. `rtx.d3d9.ue3AutoDetectLightmapTextures` turns the detection off; nothing is ever written to a config.

Vertex lightmaps have no samplers at all. They arrive as an extra vertex stream of packed coefficients declared as `TEXCOORD5` (simple) or `TEXCOORD5/6/7` (directional), typed `D3DCOLOR` rather than a float pair. A `D3DCOLOR`-typed `TEXCOORD` is never a UV set in UE3, so those elements can never be chosen as the surface's texcoord buffer.

Subtracting the whole sampler set leaves some draws with no albedo at all: a surface lit by nothing but a lightmap has no material texture to fall back on. Those render at `rtx.legacyMaterial.albedoConstant`, which is the intended result since Remix relights them, but they have no colour of their own for the Toolkit to show. `[RTX-Compatibility][UE3-NoAlbedo]` reports the sampler picture that explains why, and the capture exports the material carrying that same constant rather than skipping it - a skipped material leaves the mesh bound to a prim the stage never declares, which the Toolkit can neither select nor author.

Where such a material does carry a colour it is in a `UniformVector_*` register, and the value is a tint UE3 multiplied against the lightmap for brightness rather than a finished albedo. Used raw once the lightmap is gone it reads as near-black, and since a register at or below `0.01` is rejected so a later one can supply the real colour, the bottom of a ramping tint falls back to the legacy constant and the surface would step straight from there to a dark tint. `rtx.d3d9.ue3ConstantAlbedoTintGain` closes both gaps: the register's brightest channel times the gain, clamped to 1, is the weight blending the legacy constant towards the register's fully saturated hue, so a ramp fades continuously.

Textured materials are not tinted this way, because the albedo texture replaces the constant rather than modulating it. Mirror's Edge's Runner Vision highlighting is driven by exactly such a tint on textured surfaces, so it is not reproduced.

### Identity

UE3 compiles one base-pass pixel shader per lightmap policy from the same generated material HLSL. `DirectionalLightmaps=False` sets `SIMPLE_LIGHTING`, which deletes the whole `LightMapBasis` transfer block from `BasePassPixelShader.usf`; fxc then dead-strips every material symbol that only fed it: the normal map, the specular colour and the specular power. `FNoLightMapPolicy` (movable geometry, translucency, and dynamic meshes under `viewmode unlit`) folds the lightmap to zero and loses the same block. An identity built from "whatever the CTAB declares" therefore moves with a system setting.

The rule is to exclude, in *both* compiles, any material sampler the base pass only consumes as a lighting input: a proven tangent-space normal unpack, or a sampled value that never reaches `oC0` as colour. Where the directional compile declares such a sampler it is recognised and dropped; where fxc already stripped it there is nothing to drop, so the two sets agree without the runtime having to know which compile it is looking at. The test is a property of the sampler's own use, not of a symbol that exists in one compile only, and that is why it is symmetric. Two guards bound the damage if it misreads a shader: a sampler carrying the diffuse-anchor signal is never dropped, and a shader that classifies as nothing but lighting inputs falls back to its full declared set rather than collapsing onto a shared identity. The canonical signature is used for *every* UE3 material rather than only lightmap-bearing ones, so a material drawn on static geometry and on a movable prop shares one anchor.

A shader that declares no material samplers *at all* is the exception: it gets no canonical signature and is seeded from its bytecode hash instead. The uniform declarations are all that is left to build a signature from, and they identify no material - `UniformVector_0` alone describes a large share of the constant-colour and lightmap-only materials, and with an empty texture set to pair it with every one of them lands on the same `textureSet+shader` hash, where a single authored `mat_<hash>` covers the lot. Seeding from bytecode costs lightmap-policy invariance for these shaders alone - the same trade the all-lighting-inputs guard makes - and keeps them anchorable one material at a time.

Do not use the `LightMapBasis` literals instead, however tempting they are as a marker of the richer compile. An earlier implementation tainted every value derived from them and pruned whatever the taint reached. Because the literals exist in one compile only, it fired on the lightmapped shaders under `DirectionalLightmaps=True` and on nothing at all under `False`, so it removed samplers the other compile keeps, in some cases the material's own primary texture, and manufactured the very divergence it was meant to remove. Any rule seeded on a symbol that only one compile carries has this shape.

### Constants tier

Only `UniformVector_*` parameters contribute (`rtx.d3d9.ue3MicConstantIdentity`, on by default). The `UniformScalar_*` class is excluded wholesale rather than analysed, because that is where the two compiles genuinely disagree: `DiffusePower` exponents the lightmap under `SIMPLE_LIGHTING` and a basis-derived transfer coefficient otherwise, while `SpecularPower` is structurally identical yet present in only one compile. Nothing in the bytecode separates them, so keeping either keeps both. The vectors carry the tint UE3's colour variants are told apart by - `T_RooftopPropsClusters_DA` and its Blue/Orange/Yellow siblings are one texture set differing only in that parameter, and anything that drops it merges them - but they are not taken wholly as declared; see below.

### Coverage

Most materials keep one identity across both settings. A small minority do not: the directional compile declares an extra sampler *and uses it as colour*, which no rule over sampler dataflow separates from a second diffuse layer. Author against one setting; `False` is the better covered, since it declares the smaller sampler set and so is the state the exclusions reconstruct.

If you change any of this, measure it by matching materials between a `True` and a `False` run on their **bound images**, allowing one side to be a subset of the other, and count the materials that differ. Every identity-derived field (the texture set, the canonical seed, the material's primary texture) moves when the sampler set moves, so joining on any of them drops exactly the materials being counted and can under-report the divergence by an order of magnitude.

Turning `rtx.d3d9.ue3MicConstantIdentity` off makes identity shader + texture set alone, which is fully lightmap-independent, at the cost of merging every instance that shares a texture set onto one anchor. Mirror's Edge tints many of its variants from one set, so this collapses a substantial fraction of them.

Any change to what feeds identity re-mints the affected material hashes, and `mat_<hash>` prims authored against the old ones stop matching. See [Re-anchoring after an identity change](#re-anchoring-after-an-identity-change).

## Albedo selection and the texture spread cache

A UE3 pixel shader binds several textures and nothing in the bytecode declares which one is the surface colour, so the runtime scores every sampler and picks a winner. One of the scoring signals is *material spread*: how many distinct pixel shaders have been seen sampling that texture. A texture used by one or two materials is that material's own albedo; a texture used by a dozen unrelated ones is a shared detail, grunge or tint sheet, and is penalised heavily so it cannot out-rank the real base map.

Spread is learned by watching draws, so it is the only scoring input that is not a pure function of the draw in front of it. It is persisted to `rtx-remix/ue3TextureSpread.cache` and **only the persisted value is scored against** - textures discovered during the current session raise the count on disk for next time but do not change any decision now. Two runs on the same cache file therefore reach the same pick for every material.

Every other scoring signal is a property of the draw itself. Size counts in mip steps rather than texel count, since the bound dimensions only say how far a texture has streamed in, and one doubling is worth less than any single structural signal. Where a sampler's UV origin is proven from bytecode, reading the primary `.xy` pair is preferred over the packed `.zw` pair, because on a static mesh set 1 is the secondary channel and the base map reads set 0. A sampler whose origin cannot be proven at all is penalised heavily, since the surface's texture transform comes from the winning stage alone. `rtx.d3d9.ue3LogAlbedoSelection` prints the per-sampler breakdown, marking these `UV0XY` and `NOUVORIGIN` alongside the `spread=` term.

The cache file is therefore part of the material's appearance, in the same way `rtx.conf` is. Ship it with a mod, and delete it only if you intend to relearn from scratch. To regenerate one: delete the file, play through a representative spread of levels, and exit the game normally (it is flushed on shutdown as well as periodically). Repeat until a pass adds no new entries - a texture's spread only counts shaders that have actually been drawn, so a single pass through one chapter will undercount anything reused later in the game. `rtx.preferredAlbedoTextures` and `rtx.neverAlbedoTextures` remain the per-texture override for anything the scoring still gets wrong.

## Persisted albedo picks

The winning sampler is pinned per material and persisted to `rtx-remix/ue3DiffuseSelection.cache`, so every session starts from the same pick. In memory the pin survives a level reload but not a relaunch, and a decision first made while a material's textures were still streamed down can differ from the settled one. A pin is still superseded once a larger set of mips arrives, so on a cold cache a surface can briefly show a different layer while a level pages in.

The file records the scoring version and the texture-tag set sizes it was written under, and is discarded when either changes - a different build's scoring, or an edited `preferredAlbedoTextures`/`neverAlbedoTextures`/`lightmapTextures`, re-derives rather than serving picks it would no longer make. If you change the albedo score, bump `kUe3DiffuseSelectionScoringVersion`: a bounded sample of loaded picks is re-scored each session and any disagreement is reported, so forgetting is noisy rather than silent.

## Samplers whose UV origin cannot be proven

Not every sampler's coordinate can be traced back to an interpolant - screen-space, reflection-driven and some untraceable chains resolve as `originValid=0`. Upstream falls back to the fixed-function `D3DTSS_TEXCOORDINDEX` for those, which UE3 never sets meaningfully: it is leftover device state whose D3D9 default is "stage N reads texcoord N", restored on any device reset. A material whose albedo sits on sampler 2 would start reading IA texcoord set 2 after a reset or level reload having read set 0 before, which presents as the texture spontaneously rescaling.

So under UE3 the interpolant is borrowed from the shader's other material samplers when they unanimously name one - they shade the same surface, and their agreement is proven from bytecode rather than read from device state. Lightmap and engine samplers do not vote, since they legitimately read their own set. Only the origin is borrowed; the affine chain stays unresolved, because a sibling texture's tiling is not this one's. `rtx.d3d9.ue3LogUvResolution` marks these `[origin borrowed from sibling material samplers]`, and where the siblings disagree or none resolve, the legacy path still applies.

## One texcoord set per surface

A Remix surface carries a single texcoord buffer, and the UV set it uses is resolved from the winning albedo sampler. UE3 materials routinely sample two textures from *different* IA texcoord sets - a base map plus a decal, blend or overlay layer will read one each. Only the winner's set reaches the surface, so on such a material exactly one of the two textures can be mapped correctly; the other is drawn through the wrong channel and appears at the wrong scale.

No scoring threshold resolves this - it is one UV set for two demands. `rtx.d3d9.ue3LogUvResolution` identifies it: two lines for the same pixel shader with different `iaSet=` values means that material has the conflict, and the `stage=` on each says which texture claims which set. Choose which texture should be correct with `rtx.preferredAlbedoTextures`, and expect the other layer to be mapped wrongly.

## Material identity and replacement anchor stability

With `rtx.d3d9.ue3EngineMode`, a material's identity hash (the `mat_*` anchor that captures, texture tags, and asset replacements key off) is a chain: pixel shader identity → material texture set (image hash of every CTAB `Texture2D_*`/`TextureCube_*` sampler) → material constants (`UniformVector_*`). Every tier is a pure function of the draw *and of the shader*, decided before the first draw is hashed and never revised, so a material's hash is reproducible from the first frame of any session with no learned state behind it.

Getting there means each of the inputs UE3 offers that is not itself reproducible has to be recognised and left out:

- **Frame-varying constants** (panners, rotators, flipbook/sub-UV frames, time-driven fades, distance blends). UE3 re-evaluates every material uniform expression on the CPU per draw and writes the result into the same registers that carry authored `VectorParameterValues`, so at the D3D9 level an animation and a tint are the same kind of value. What separates them is where the value *goes*: a texture transform reaches a sampler's coordinate, a tint reaches the output colour. `rtx.d3d9.ue3MicVolatileConstantDetection` (on by default) takes the transitive set of constant registers a sampler's coordinate depends on and drops those from the constants tier, keeping everything else. `rtx.d3d9.ue3LogMaterialInstanceHash` reports the split per shader as `volatileRegs=[...]` and `uniforms kept=[...] excludedAsVolatile=[...]`.

  It has to be the dependency set rather than the affine resolver's scale and offset registers, which only exist for a transform expressible as `uv * cA + cB`. Mirror's Edge's animated materials mostly are not that shape: a Rotator arrives as a 2x2 matrix spread across a register pair, recognisable in the logged values as `(cos,-sin)` and `(sin,cos)`, which fits neither slot. Keying on the resolver's output left those materials with nothing excluded.

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

So re-anchoring reads the new value rather than mapping the old one. `rtx.d3d9.ue3LogMaterialInstanceHash` prints one breakdown per material, and its `textures=[...]` field identifies which material is which by the images its samplers bind - match on the albedo texture and take the `materialHash`. A fresh capture works too, and is the better option when many materials moved at once.

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
