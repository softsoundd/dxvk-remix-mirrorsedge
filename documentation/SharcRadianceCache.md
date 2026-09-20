# SHARC Radiance Cache

SHARC (Spatially Hashed Radiance Cache) is a world-space cache of irradiance that lets indirect
paths terminate early. It is the fourth value of `rtx.integrateIndirectMode` (`3`), alongside
Importance Sampled, ReSTIR GI and the Neural Radiance Cache, and it uses the shader headers from
[NVIDIA-RTX/SHARC](https://github.com/NVIDIA-RTX/SHARC) vendored under
`src/dxvk/shaders/rtx/external/sharc/`.

## How it works

Each frame runs three passes, all driven from `DxvkPathtracerIntegrateIndirect::dispatchSharc`.

**Update.** A sparse pass traces one extra path per NxN tile of the render target
(`rtx.sharc.updateTileSize`, 5 by default, so roughly 4% of pixels). It runs the ordinary indirect
integrator compiled as a SHARC update variant. At every eligible opaque vertex it inserts a cell
into the hash grid, keyed on position, distance level, normal octant and ray portal space. Lighting
gathered along the path is recorded per vertex and propagated back through the chain in reverse at
path end, so each cell is written once per path rather than once per contribution.

Two behaviours extend the SDK integration guide:

- The camera-visible primary vertex is deposited as well (`rtx.sharc.updatePrimaryVertex`), valued
  as the direct pass RTXDI lighting plus the sampled continuation. Without it, a path whose first
  bounce escapes to the sky contributes nothing at all.
- A first bounce that escapes to the sky is re-sampled from a cosine lobe about the primary normal
  (`rtx.sharc.updateSkyRetries`), so the update budget lands on geometry more often outdoors. What a
  cell stores does not depend on how a ray reached it, so this changes which cells receive samples,
  not what they hold.

**Resolve.** A compute pass runs one thread per cache slot, turning accumulated samples into
readable cells and evicting entries unobserved for `rtx.sharc.staleFrames` frames. It costs the same
every frame whether or not the slots hold anything, which is why `rtx.sharc.capacityLog2` is a
performance setting and not only a memory one.

**Query.** The full-resolution indirect pass, compiled as a SHARC query variant, checks the cache at
each eligible vertex. A hit adds the cached radiance through the ordinary path throughput and ends
the path; a miss continues tracing normally.

## What a cell can represent

A cell holds one non-directional radiance value, which is the source of most of the tuning surface:

- **Eligibility.** A vertex is cached only if it is opaque, at full opacity, outside a medium, not a
  subsurface material, no brighter than `rtx.sharc.maxEmissiveLuminance`, and at or above
  `rtx.sharc.minRoughness`. The cache holds reflected light and the path adds emission separately,
  which is why emissive surfaces are excluded.
- **Lobe footprint.** A path arriving by a specular lobe is served only once that lobe has spread
  wider than the cell, tested against the length of the whole segment. This is what keeps an
  isotropic cell out of a narrow reflection, and it is tested instead of, not in addition to, a
  stricter roughness floor on the receiving surface.
- **Update roughness clamp.** `rtx.sharc.updateRoughnessClamp` roughens a material during the update
  pass only. On a glossy surface every update path arriving from a different direction otherwise
  deposits a different value and the cell mean never settles. It changes what a cell stores, never
  which surfaces qualify, so it is applied after the eligibility test and costs no coverage.
- **Deposit bounds.** A cell is a mean of its samples, so a single outlier is not averaged away,
  only divided by the sample count -- and that count falls with render resolution, because the
  update pass traces one path per tile of the *render* target. `rtx.sharc.maxDepositRatio` bounds a
  deposit against what the cell already holds, with `rtx.sharc.minDepositCeiling` as an absolute
  floor so a cell near black can still brighten. Both take effect live; they bound only values
  written from now on, and existing cells wash out within `rtx.sharc.accumulationFrames` frames.

## Grid density

`rtx.sharc.gridScale` is the SDK `sceneScale` parameter. A cell edge is the distance from the camera
rounded down to a power of two and divided by this value, so cells grow with distance and the
setting behaves as an angle rather than a length: at the default of 50 a cell spans roughly 0.6 to
1.1 degrees of arc anywhere in the scene. Carrying no world units, one value suits any game whatever
its unit convention, which is why it needs no reference to `rtx.sceneScale`.

## Ray portals

Portal space is part of the cache key. Radiance at a point depends on which portal space reached it,
because crossing a portal rewrites the ray mask that feeds the continuation ray, the NEE shadow ray
and the unordered resolve. Keying on it means a vertex reached through a portal occupies its own
cell and can never be read back by a main-space path, which is what makes insertion from portal
space safe and lets portal-only geometry be cached at all. Two bits for this are taken from the
level field of the vendored SDK headers; see `src/dxvk/shaders/rtx/external/sharc/README.md`.

## Cache invalidation

The cache is cleared on a renderer history reset, on a frame discontinuity, and when a setting that
changes the meaning of a stored estimate changes: grid density, capacity, roughness floor, emissive
limit, sample floor, update bounces, the update roughness clamp, or any of the policy flags. Moving
geometry, moving lights and weather changes rely on temporal replacement and stale eviction rather
than on a scene-change detector.

## Requirements and limits

SHARC needs `shaderInt64`, buffer int64 atomics, `shaderFloat16`, 16-bit storage and `rayQuery`. The
mode is removed from the UI on devices without them.

The estimator is a spatial and temporal approximation with finite-depth bias. Sharp reflected
lighting, material detail finer than a cell, and fast lighting changes can show cache bias or lag.

## Debug views

Six views, contiguous at 580-585, all written from the query pass. `rtx.debugKnob.x` selects which
indirect vertex a per-pixel view reports, 0 being the first indirect hit, matching the NRC bounce
views.

| Id  | View | Shows |
|-----|------|-------|
| 580 | SHARC Query Outcome | green hit, red miss, blue too close, yellow footprint too narrow, grey rejected |
| 581 | SHARC Reject Reason | the first failing eligibility term, colour-coded |
| 582 | SHARC Cached Radiance | the radiance read from the cache where a path ended on it |
| 583 | SHARC Termination Bounce | the bounce at which the cache ended the path |
| 584 | SHARC Grid Cells | the hash-coloured cell at the vertex, for reading cell size directly |
| 585 | SHARC Cell Age | R accumulated frames, G stale frames, B sample count |
