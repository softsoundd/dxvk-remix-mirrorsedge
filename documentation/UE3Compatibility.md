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

Diagnostics:

- `rtx.d3d9.ue3LogCapturePrecision` logs the resolved source per unique (vertex shader, vertex factory). On fallback it also dumps the instructions that produced each `oPos` component.
- `rtx.d3d9.ue3RequireExactVertexCapture` drops draws that would unproject, so distance-dependent distortion cannot occur, at the cost of losing whatever the exact paths miss.
- `rtx.d3d9.ue3VertexCaptureSourceOverride` forces one source for every draw. Toggling `0` (Auto) and `3` (Clip Reconstruction) on a long outdoor view is the A/B for the distortion.

The static vertex-capture cache (`rtx.d3d9.ue3StaticLocalMeshVertexCaptureCache`) reuses captured vertex data from an earlier frame for local meshes, decals and foliage that are not moving. Exact-source captures do not depend on the camera, so they stay valid across frames. It refuses reconstruction-sourced captures (those depend on where the camera was), camera-facing factories (sprites, billboards, leaf cards), morphing terrain, terrain in general (a hit would suppress the draw the terrain baker needs), and skinned draws.

The key includes the object transform and the shader constants other than camera registers, so only a draw that repeats both can hit. Skinned meshes put bone matrices in that hash and would mint a new key every frame, which is why they are refused; a moving prop does the same through its transform. A capture gets a device-local buffer only after its key has been seen on `rtx.d3d9.ue3StaticLocalMeshVertexCaptureCacheWarmupFrames` distinct frames. Until then it is a few bytes of CPU bookkeeping. `rtx.d3d9.ue3StaticLocalMeshVertexCaptureCacheBudgetMiB` caps retained bytes and evicts LRU first. `...RetentionFrames` is a staleness bound and `...MaxEntries` a backstop for many tiny captures. Expired entries are recaptured when the mesh returns to view, so a short window at a high frame rate will expire them during ordinary camera movement.

Some titles recompute a draw's transform every frame even for still geometry, so keys never repeat. Below `rtx.d3d9.ue3StaticLocalMeshVertexCaptureCacheMinReusePercent` the cache releases its buffers and key records, then re-tests every `...ReuseProbeFrames` frames. Set the threshold to `0` to disable the guard.

`rtx.d3d9.ue3LogStaticVertexCaptureCacheStats` reports entries, bytes, keys awaiting admission and reuse rate about once a second, including dormancy. Buffers appear under `RTXVertexCapture` in the memory profiler and HUD. If reuse is near zero, the keys are churning; raising the budget will not help.

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

### Identity

UE3 compiles one base-pass pixel shader per lightmap policy from the same generated material HLSL. `DirectionalLightmaps=False` sets `SIMPLE_LIGHTING`, which deletes the whole `LightMapBasis` transfer block from `BasePassPixelShader.usf`; fxc then dead-strips every material symbol that only fed it: the normal map, the specular colour and the specular power. `FNoLightMapPolicy` (movable geometry, translucency, and dynamic meshes under `viewmode unlit`) folds the lightmap to zero and loses the same block. An identity built from "whatever the CTAB declares" therefore moves with a system setting.

The rule is to exclude, in *both* compiles, any material sampler the base pass only consumes as a lighting input: a proven tangent-space normal unpack, or a sampled value that never reaches `oC0` as colour. Where the directional compile declares such a sampler it is recognised and dropped; where fxc already stripped it there is nothing to drop, so the two sets agree without the runtime having to know which compile it is looking at. The test is a property of the sampler's own use, not of a symbol that exists in one compile only, and that is why it is symmetric. Two guards bound the damage if it misreads a shader: a sampler carrying the diffuse-anchor signal is never dropped, and a shader that classifies as nothing but lighting inputs falls back to its full declared set rather than collapsing onto a shared identity. The canonical signature is used for *every* UE3 material rather than only lightmap-bearing ones, so a material drawn on static geometry and on a movable prop shares one anchor.

Do not use the `LightMapBasis` literals instead, however tempting they are as a marker of the richer compile. An earlier implementation tainted every value derived from them and pruned whatever the taint reached. Because the literals exist in one compile only, it fired on the lightmapped shaders under `DirectionalLightmaps=True` and on nothing at all under `False`, so it removed samplers the other compile keeps, in some cases the material's own primary texture, and manufactured the very divergence it was meant to remove. Any rule seeded on a symbol that only one compile carries has this shape.

### Constants tier

Only `UniformVector_*` parameters contribute (`rtx.d3d9.ue3MicConstantIdentity`, on by default). The `UniformScalar_*` class is excluded wholesale rather than analysed, because that is where the two compiles genuinely disagree: `DiffusePower` exponents the lightmap under `SIMPLE_LIGHTING` and a basis-derived transfer coefficient otherwise, while `SpecularPower` is structurally identical yet present in only one compile. Nothing in the bytecode separates them, so keeping either keeps both. The vectors are taken as declared, because they carry the tint UE3's colour variants are told apart by. `T_RooftopPropsClusters_DA` and its Blue/Orange/Yellow siblings are one texture set differing only in that parameter, and anything that drops it merges them.

### Coverage

Most materials keep one identity across both settings. A small minority do not: the directional compile declares an extra sampler *and uses it as colour*, which no rule over sampler dataflow separates from a second diffuse layer. Author against one setting; `False` is the better covered, since it declares the smaller sampler set and so is the state the exclusions reconstruct.

If you change any of this, measure it by matching materials between a `True` and a `False` run on their **bound images**, allowing one side to be a subset of the other, and count the materials that differ. Every identity-derived field (the texture set, the canonical seed, the material's primary texture) moves when the sampler set moves, so joining on any of them drops exactly the materials being counted and can under-report the divergence by an order of magnitude.

Turning `rtx.d3d9.ue3MicConstantIdentity` off makes identity shader + texture set alone, which is fully lightmap-independent, at the cost of merging every instance that shares a texture set onto one anchor. Mirror's Edge tints many of its variants from one set, so this collapses a substantial fraction of them.

Any change to what feeds identity re-mints the affected material hashes, and `mat_<hash>` prims authored against the old ones stop matching. There is no automatic migration: re-anchor the affected materials, using `rtx.d3d9.ue3LogMaterialInstanceHash` to read the new hash for a material you can identify by its albedo texture.

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

With `rtx.d3d9.ue3EngineMode`, a material's identity hash (the `mat_*` anchor that captures, texture tags, and asset replacements key off) is a chain: pixel shader identity → material texture set (image hash of every CTAB `Texture2D_*`/`TextureCube_*` sampler) → material constants (`UniformVector_*`/`UniformScalar_*`). Every tier is a pure function of the draw, so the same material instance always gets the same hash when the inputs themselves are stable. UE3 games expose three unstable input classes, so the runtime deals with each:

- Render targets bound as material samplers (scene captures, reflection buffers). An RT's image hash embeds a creation counter and changes every respawn, checkpoint, or level load. RTs are excluded from identity by default (`rtx.d3d9.ue3MicExcludeRenderTargetsFromIdentity`). Anchors that still key off RT-bearing identities need re-anchoring once.
- Frame-varying constants (time/panner/fade/sub-UV expressions). Churn auto-exclusion drops such a group's constants from identity once it has minted enough distinct hashes. That exclusion is written to `rtx-remix/ue3MicAutoExcludedGroups.cache` (`rtx.d3d9.ue3MicPersistAutoExcludedConstantGroups`), so the group's identity is deterministic from the first frame of every later session instead of flipping mid-session at an unpredictable point.

  Deleting that cache is not free: detection does not always fire a second time. On Mirror's Edge, four material families minting thousands of distinct constant hashes each went undetected for a whole session after the file was removed, while two others were caught within 32 draws. Anchors authored against a settled exclusion break when this happens, so treat the file as authoring state rather than a scratch cache. Where a family churns without being caught, pin it with `rtx.d3d9.ue3MicConstantIdentityExcludedGroups`, which takes the `group=0x...` key printed by `rtx.d3d9.ue3LogMaterialInstanceHash` and covers exactly one (shader, texture set) family.

  Reach for `rtx.d3d9.ue3MicConstantIdentityExcludedShaders` only when every material on a shader is frame-varying. It is keyed on the shader, and one UE3 base-pass shader commonly serves dozens of material families - excluding one seed on this content stripped the constants tier from 64 of them and merged six authored anchors that differed only by tint.
- Session-composited textures (engine-generated textures reuploaded with different contents every session). Their content hash is session-unique, so any identity containing them cannot be anchored from a capture. Tag the texture's descriptor hash (stable across recreations; shown as `desc:0x...` in the `rtx.d3d9.ue3LogMaterialInstanceHash` breakdown and `[RTX-MicDrift]` sampler diffs) in `rtx.d3d9.ue3MicIdentityExcludedTextureDescHashes`, then re-anchor the material once. Alternatively, anchor the override at the raw texture hash (`mat_<textureHash>`): replacement lookup runs tiers material → textureSet+shader → texture, so a texture-tier anchor catches every material variant that selects that image as its albedo, while more specific anchors still win where present.

## Replacement anchor diagnostics

When an authored enhancement does not appear (or appears intermittently), enable `rtx.logReplacementResolution = True` for one session and reproduce briefly. The log names the failing material and the drifting identity tier directly:

- `[RTX-ReplacementResolve]`: how each material/mesh resolved against the mod's anchors (which lookup tier matched, or `NO MATCH`), plus a per-mod anchor dump at load.
- `[RTX-ReplacementFlap]`: a material family that previously matched stopped matching (or vice versa) mid-session, with old/new hashes for every tier.
- `[RTX-MicDrift]`: a material family minted a new identity, attributed to the tier that moved: per-sampler image hash diffs (with `desc:0x...` and RT flags) or changed constant registers with old/new values.
- `[RTX-MicRtPoisoning]`: a material identity still embeds a render-target image hash (only possible with RT exclusion disabled).
- `[RTX-MeshAnchorDrift]`: a mesh replacement key moved, attributed to its geometry part (unstable vertex data, e.g. CPU-morphed skinned meshes) vs its material part (mesh keys are `geometryHash XOR materialHash`).

`rtx.replacementDebugHashes` tracks specific hashes in detail (matched against texture, material, textureSet+shader, geometry, and mesh-key hashes) without the full-scene log volume. Toggling enhanced assets on/off in the UI intentionally shows up as synchronised matched/`NO MATCH` flaps with unchanged hashes.
