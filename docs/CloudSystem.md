# Cloud System

Remix Plus ships a fully procedural volumetric cloud layer with
physically-based lighting. The system sits on top of the Hillaire
physical atmosphere (`rtx.skyMode = 1` -- Numos) and
shares the per-frame LUT cadence with the rest of the atmosphere
pipeline. There are no cloud assets to author: the field, the
lighting, the terrain shadows, and the fog interactions are all
synthesised at runtime from a small set of `rtx.atmosphere.cloud*`
parameters.

This document describes how the cloud system is wired together in
the runtime. For weather-driven parameter blending and per-preset
recommended values, see
[Weather Presets -- Plugin Integration Guide](integrators/weather-presets.md).
For the authoritative list of individual cloud knobs and their
defaults, see [Rtx Options](../RtxOptions.md) (search for
`rtx.atmosphere.cloud`).

## Architecture

The cloud system runs a small graph of compute passes once per frame
inside `RtxAtmosphere::computeLuts` ([src/dxvk/rtx_render/rtx_atmosphere.cpp](../src/dxvk/rtx_render/rtx_atmosphere.cpp)),
plus one consumer wire-in inside the volumetric pass.

1. **Cloud noise volume** -- a prebaked 3D Perlin FBM texture is
   loaded once at startup and tile-wraps horizontally with period
   `cloudNoiseTileKm` (12 km default). All density taps in the
   system pull from this single resource.

2. **`D_sun` and `D_ambient` voxel grids** --
   [`cloud_sun_density_grid.comp.slang`](../src/dxvk/shaders/rtx/pass/atmosphere/cloud_sun_density_grid.comp.slang)
   and
   [`cloud_ambient_density_grid.comp.slang`](../src/dxvk/shaders/rtx/pass/atmosphere/cloud_ambient_density_grid.comp.slang)
   bake two world-anchored 256x256x32 R16F grids of summed optical
   depth -- one along the current sun direction, one straight up.
   Both grids rebake every frame (full rate, sequenced back-to-back
   in the command buffer with intervening write→read barriers). The
   prior round-robin every-8-frames cadence was dropped on 2026-05-19
   after the cumulus-on-terrain shadow visibility fix made the
   staggered bake read as a ~2 Hz shadow stutter at low fps; full
   rate eliminates the stutter at ~8x bake cost. Consumers read
   zero-stale grids.

3. **Cloud-sky-transmittance LUT** --
   [`cloud_sky_transmittance_lut.comp.slang`](../src/dxvk/shaders/rtx/pass/atmosphere/cloud_sky_transmittance_lut.comp.slang)
   bakes a 32x16 R16F LUT keyed by `(azimuth, elevation)` storing
   cloud transmittance along that direction. The parameterisation
   mirrors the Hillaire sky-view LUT exactly so a single UV
   computation in a consumer samples both LUTs consistently.

4. **Cloud render compute pass** --
   [`cloud_render.comp.slang`](../src/dxvk/shaders/rtx/pass/atmosphere/cloud_render.comp.slang)
   ray-marches the cloud slab per pixel at downscaled (DLSS-input)
   resolution and writes a premultiplied RGB + view-ray
   transmittance RT. Lighting uses the Nubis Cubed 2023 equations
   (paper pp. 137 / 142): a two-HG-lobe direct term with a sigma_ms
   remap on `dot(sun, viewDir)` and cloud-SDF depth, plus a
   `pow(1 - dim_profile, 0.5) * exp(-D_ambient)` page-142 ambient
   term. The ambient term is sampled from the sky-view LUT at the
   sun direction (clamped a hair above the horizon) so the ambient
   colour matches the actual atmospheric forward-scatter at sunset
   instead of the bluer zenith. The moon path is a byte-faithful
   port of the legacy analytical lighting so day-night and
   eclipse-style multi-moon scenes match the analytical baseline.

5. **Primary sky-miss composite** -- when `cloudRenderRTEnable` is
   true (default), primary-ray sky-miss samples the cloud RT
   instead of calling the analytical `evalClouds`. Indirect, PSR,
   and reflection rays continue to use analytical `evalClouds`
   because the cloud RT is at primary-ray pixel coordinates --
   sampling it for a non-primary ray direction would return the
   wrong cloud.

6. **Cloud history smoother** -- a two-buffer ping-pong with an
   R16_UINT frame-id companion buffer (see
   [`rtx_fork_atmosphere.cpp:230-258`](../src/dxvk/rtx_render/rtx_fork_atmosphere.cpp))
   accumulates the cloud RT across frames. Per-pixel age tracking
   lets the shader reject stale history at foreground-occluded
   slots, killing the bright-trail ghosting that the cloud
   accumulator would otherwise leave behind moving geometry.

## What the clouds shade

The cloud field participates in three places outside its own RT:

- **Cloud-on-terrain shadows folded onto the sun (sun-only).**
  A cloud shadow is just attenuation of the SUN along the sun direction, so it
  is applied where that is physically true: onto the sun's own radiance, *before*
  the sun is summed with other lights and *before* denoising. The sun is a real
  Remix distant light (see "directional lights" below), so the surface fold lives
  in the standard sun NEE: when `cloudVoxelShadowsEnable` is true (default) and
  the sampled light is the flagged atmosphere sun (`atmosphereCloudShadowed`), the
  direct integrator
  ([`integrator_direct.slangh`](../src/dxvk/shaders/rtx/algorithm/integrator_direct.slangh))
  samples the cloud transmittance toward the sun via
  `sampleCloudGroundShadow_OptionB` and folds it onto the light's NEE radiance as
  `radiance *= pow(cloudT, cloudShadowFactorStrength)`. The volumetric path folds
  the same factor onto the per-froxel sun radiance in
  `sampleAtmosphereSunLightVolume`. Terrain shows cumulus-shaped drifting shadow
  patches matching the cloud silhouettes overhead, and only the sun is darkened --
  indoors it is automatically a no-op (the sun's direct contribution is already
  ~0 from the roof-occluded shadow ray), and lamps / point lights are never
  touched. (Before 2026-06-21 the fold lived in a bespoke `sampleAtmosphereSunLight`
  sun NEE; that path was removed when the sun became a real distant light.)

  **Why this is correct, and what it deleted.** Because the factor rides the
  sun's contribution alone (multiplied alongside the geometric sun-shadow ray):
  indoors it is automatically a no-op for every surface type (walls, decals,
  viewmodels, particles) -- the sun's direct contribution is already ~0 indoors
  from the roof-occluded NEE shadow ray, so multiplying by the cloud factor
  changes nothing, with no per-pixel geometry test. Lamps/point lights are never
  touched. The same fold is applied to the sun distant light in the indirect
  integrator, so bounce light off cloud-shadowed ground is correct for free, with
  no double-count (the legitimate sky-bounce reduction lives in `evalSkyRadiance`). This single change deleted the entire
  geometry-blindness compensation stack: the sealed-interior zenith up-ray gate,
  the viewmodel/decal camera-origin correction (and its `Surface::isDecalCategory`
  flag across three C++ files), the camera-side triangle-normal flip, and the
  dusk/dawn horizon gate in `sampleCloudGroundShadow_OptionB_impl` (the grazing-
  sun OD blowup now only crushes the already-attenuated sun, never lamps).

  **The screen-space system is deleted.** Previously the per-pixel `newShadow`
  rode a `PrimaryCloudShadowFactor` screen-extent texture *around* the denoiser
  and composite multiplied `pow(newShadow, cloudShadowFactorStrength)` onto the
  post-denoise primary **direct** radiance (the combined buffer of ALL lights).
  That single decision was geometry- and light-blind and spawned the whole
  compensation stack. The texture, its write/clear, all bindings, the composite
  application, and debug view 878 are removed. Accepted tradeoff: the per-cumulus
  pattern now passes through NRD / DLSS-RR, which can soften the crisp shadow
  edges -- the cost of doing it in the physically correct place. (The indirect
  composite multiply, `cloudShadowIndirectStrength`, was removed earlier on
  2026-06-18 as issue #37; its CB slot is a reserved pad.)

  The volumetric path folds the same `pow(transmittance, cloudShadowFactorStrength)`
  onto the per-froxel sun radiance in `sampleAtmosphereSunLightVolume` (it always
  applied the shadow into radiance directly -- that was already architecturally
  correct; the contrast curve was added for parity).
  `computeGroundReflectionAnalytical` (multiscatter, sentinel position
  `vec3(0, 0, 0)`) keeps the analytical cloud darkening (`skipCloudShadow=false`,
  the helper default) -- it never reaches the sun-NEE fold-in, and migrating it
  to the voxel grid would poison the global ambient.

- **Volumetric god-rays.** The volume integrator's sun-NEE path
  runs through the same `sampleAtmosphereSunLightVolume` helper, so
  the same `D_sun` voxel tap modulates per-froxel direct sun radiance.
  Wherever gaps in the cloud field let sunlight reach the fog, the
  froxels light up; wherever a cumulus blocks it, the froxels go
  dark. The visible shafts of light through fog and dust come from
  exactly this. See the dedicated section below for the full call
  graph.

- **Cloud-occluded sky-ambient in volumetrics.** When
  `cloudSkyAmbientStrength > 0` (default 0, so off by default), the
  volume integrator runs a 6-direction hemisphere integration:
  zenith plus five mid-elevation directions at 72-degree azimuth
  spacing. Each direction samples the sky-view LUT * the
  cloud-sky-transmittance LUT, weights by the volumetric HG phase,
  and adds to the per-frame radiance. Overcast scenes have visibly
  darker volumetric ambient than clear-sky scenes; flipping
  `cloudSkyAmbientCloudOcclusionStrength` to 0 disables the cloud
  modulation (debug only -- visually inverted versus reality).

## User instructions

1. Enable the physical atmosphere: `rtx.skyMode = 1`. The cloud
   pipeline is gated on this; in `Rasterized` or `CombinedRasterized`
   sky modes the cloud passes are skipped entirely and only the
   prebaked noise texture stays resident.
2. The cloud RT compositor (`cloudRenderRTEnable`) and the voxel-grid
   terrain shadows (`cloudVoxelShadowsEnable`) are both default-on
   as of the 2026-05-13 ship. There is nothing to enable.
3. Volumetric sky-ambient (`cloudSkyAmbientStrength`) is default-off
   so the baseline volumetric appearance is bit-identical. Raise to
   1.0 for the physical baseline; the term contributes a sky-tinted
   ambient that brightens shadowed and overcast fog.
4. Tune cloud appearance through the dev menu under
   **Atmosphere -> Clouds** (sub-trees for coverage, type, lighting,
   multi-scatter, sun-shadow, sky-ambient, cloud motion) or by
   setting any `rtx.atmosphere.cloud*` key in `rtx.conf`. The most
   common knobs:
   - `cloudCoverageMean` -- overall cloudiness, 0 = clear, 1 = overcast.
   - `cloudDensity` -- per-cumulus opacity multiplier.
   - `cloudAltitude` / `cloudThickness` -- vertical placement.
   - `cloudShadowMarchStrength` -- pre-denoise darkness of cloud-on-terrain
     shadows (multiplier inside the `exp(-OD * density * march)` term).
   - `cloudShadowStrength` -- master fade for the same; defaults to 0.5,
     so out of the box cloud shadows on terrain are at half strength. 0
     disables them; 1.0 is the full physical baseline.
   - `cloudShadowFactorStrength` -- artistic contrast curve on the cloud
     shadow, applied as `pow(cloudTransmittance, strength)` where the shadow is
     folded onto the SUN's radiance in the NEE (it lived in composite before the
     2026-06-19 sun-only re-architecture; same knob, new home). Default 4.0 --
     the raw transmittance at strength 1 reads too faint. Raise to sharpen
     cumulus shadow contrast, lower to soften.

## Cloud motion: one integrator, three sources

Everything that moves or changes the clouds runs through a single
per-frame **cloud-motion integrator**
(`RtxAtmosphere::advanceCloudMotion`, fork — 2026-06-21). It advances
three persistent offset accumulators by `offset += velocity * dt` once
per frame; the const `getAtmosphereArgs` (called many times per frame)
just reads them into `cloudWindOffset` / `cloudEvolutionOffset*` /
`cloudBoilPhase`.

> **Why an integrator (the bug it fixed).** These offsets used to be
> computed stateless as `instantaneousSpeed * totalElapsedTime` inside
> the const accessor. That is only correct while wind is constant. The
> slow weather drift (source 3 below) varies `cloudWindSpeed` /
> `cloudWindDirection`, so every frame the *entire accumulated offset*
> re-scaled or rotated about the origin — the whole field visibly
> jumped/slid. Integrating instantaneous velocity instead means a
> changing wind smoothly *eases* the field, so the weather drift now
> composes correctly. The three sources feed the same integrator but
> keep independent rates (no cross-coupling).

The three sources:

1. **Wind advection** (`cloudWindSpeed` / `cloudWindDirection`).
   Rigid bulk translation of the whole field across the sky. The
   prebaked noise tile-wraps at `cloudNoiseTileKm` (12 km), so at the
   default 0.02 km/s the field exactly repeats every ~10 min of travel.

2. **Field evolution** (`cloudEvolutionSpeed`, `cloudEvolutionVerticalBias`,
   `cloudBoilSpeed`; dev-menu **Atmosphere -> Clouds -> Cloud Motion**).
   Added 2026-06-21. Makes the cloud field *morph in place* rather than
   only sliding: the base 3D noise is sampled at a slowly-evolving
   offset (dominated by a vertical scroll through the volume's
   decorrelated, tile-wrapping Y axis), so formations form and dissolve
   locally; an independent faster scroll on the edge-detail tap
   (`cloudBoilSpeed`) churns the cauliflower billows at the silhouette.
   Because the evolution scroll is decorrelated from wind, it also
   breaks the 10-min wind tile-repeat. The offset feeds the **view-path**
   density sampler only -- it does not touch the placement map, hex
   lattice, column model, or height fraction, so cluster locations and
   altitudes stay put while shapes change. The baked sun/ambient shadow
   grids are intentionally *not* evolved in v1 (they keep describing the
   slow bulk field; matching the boil into them is future work). Any
   speed at 0 freezes that lever and is bit-identical to the legacy
   rigid behavior. This system replaced the old global-scalar "breathing"
   (see below) as the source of cloud shape change.

3. **Weather-parameter drift** (`__weather.drift_speed` /
   `__weather.drift_intensity`; dev-menu **Weather -> Weather Variation**
   -- renamed from "Cloud Drift" to avoid colliding with Cloud Motion; in
   `rtx_fork_weather.cpp`). Slowly modulates a few *global* weather
   scalars over time so the sky's overall character changes. As of
   2026-06-21 it is **de-pulsed and trimmed**: the old fast layer had a
   base period of exactly 30 s, whose dominant sine produced a clearly
   visible whole-sky pulse every ~30 s with all fields cresting in
   lockstep -- that fast layer is removed (slow multi-minute layer only),
   and the table is cut from 9 fields to the 3 genuinely weather-scale
   ones (`cloudCoverageMean`, `cloudWindSpeed`, `cloudWindDirection`).
   The shape-ish fields it used to wobble globally (density, thickness,
   type, anvil, coverage spread) are now handled locally by field
   evolution instead.

## Debug views

The cloud system carries five dedicated debug views, all under the
dev-menu **Debug View** combo or by writing
`rtx.debugView.debugViewIdx = <N>`. Source enum lives in
[`src/dxvk/shaders/rtx/utility/debug_view_indices.h`](../src/dxvk/shaders/rtx/utility/debug_view_indices.h).

| ID  | What it shows |
|-----|----------------|
| 873 | `D_sun` voxel grid (slice through the live bake) -- verifies sun-direction optical depth is populated. |
| 874 | `D_ambient` voxel grid -- verifies zenith optical depth is populated. |
| 875 | Cloud-on-terrain production-call-shape diagnostic. Re-runs the same `sampleCloudGroundShadow_OptionB` call the NEE path uses, per pixel, in grayscale. A divergence between this view and the actual in-game shadow pattern points to a writer/reader split between the production raygen pass and this debug pass. |
| 876 | Cloud render RT -- shows the raw Nubis Cubed lighting output before composite, for A/B against analytical `evalClouds`. |
| 877 | Raw `D_sun` optical depth at the production NEE call shape (sibling of 875). Stops at the `dSunTex.SampleLevel` -- no `exp()`, no `mix(cloudShadowStrength)`. RGB encodes magnitude (R/G) + UVW.x (B); paints magenta for surface-above-slab and blue for sun-below-horizon sentinels. Use to diagnose whether shadow problems are bake (OD wrong) vs path (helper output wrong) vs read (saturate/pow killing it). |

Debug view 878 (raw `PrimaryCloudShadowFactor` texture) was **removed**
2026-06-19 with the screen-space cloud-shadow system. Use 875/877 (which read
the `D_sun` grid directly) for cloud-on-terrain shadow diagnostics; the cloud
shadow now folds onto the sun radiance in the NEE and has no screen-space
texture to inspect.

## Indirect / PSR / Reflection rays

These rays continue to use the analytical `evalClouds` path
(`atmosphere_sky.slangh`), not the Nubis Cubed RT. The cloud RT is
written at primary-ray pixel coordinates, so a reflection ray's
view direction does not align with the pixel its hit point reads from.
This is intentional v1 scoping; per-direction cloud LUT support is a
future-work item.

## Limitations

- **Cloud RT resolution.** The cloud RT is written at the downscale
  (DLSS-input) extent, not full resolution. DLSS / TAA take it from
  there. Disabling the upscaler exposes the lower-res cloud silhouette
  at native pixel granularity.
- **DLSS / TAA smear cumulus-on-terrain shadows.** Under DLSS / TAA,
  cumulus-on-terrain shadow contrast collapses substantially -- the
  upscalers reproject using terrain motion vectors that say "stationary,"
  so as cumulus drift overhead the changing shadow signal at each
  terrain pixel reads as "wrong sample" and gets averaged toward the
  prior-frame value. Native-resolution rendering shows the shadows
  cleanly. A post-DLSS shadow re-modulation path (apply `pow(newShadow, strength)`
  at upscale resolution after DLSS, instead of pre-DLSS in composite)
  would fix this; tracked as future work.
- **Cloud RT is primary-ray only.** Reflections in mirrors, water,
  and PSR-tagged surfaces show analytical clouds, not Nubis Cubed.
  This is the dominant visible discontinuity at certain angles --
  particularly large mirror-like puddles at sunset, where the
  reflected sky's cloud lighting model does not match the direct
  sky's.
- **Cloud-on-terrain shadow strength defaults to 0.5.**
  `cloudShadowStrength` is the master fade between "full sun on the
  ground" and "voxel-baked cloud shadow on the ground." It ships at
  0.5 (half strength); set 0 to disable, 1.0 for the full physical
  baseline. Tune the visible contrast separately with
  `cloudShadowFactorStrength`.
- **Sky-ambient ships off.** `cloudSkyAmbientStrength = 0` by default
  so the volumetric pass is bit-identical to the no-fork baseline.
  This is intentional rollback safety; in-game review flips it on.

## Future work

- **Sun-only direct cloud factor (DONE).** The cloud shadow folds onto the sun's
  radiance pre-denoise — on the surface via the real sun distant light in the
  direct/indirect integrators (gated on `atmosphereCloudShadowed`), and in fog via
  `sampleAtmosphereSunLightVolume` — so it darkens only the sun and is
  automatically correct indoors for every surface type. The whole
  geometry-blindness compensation stack (sealed-interior zenith gate,
  viewmodel/decal origin correction + `isDecalCategory` flag, normal flip,
  dusk/dawn horizon gate) and the screen-space `PrimaryCloudShadowFactor` system
  were deleted, as was the bespoke `sampleAtmosphereSunLight` sun NEE (2026-06-21,
  when the sun became a real distant light). The one residual is the accepted tradeoff of option (b): the
  per-cumulus pattern now passes through NRD / DLSS-RR and the denoiser can
  soften the crisp shadow edges. If that softening reads as too mushy in review,
  the follow-on is option (a) -- a dedicated denoised sun-direct buffer with its
  own NRD/DLSS-RR wiring -- which would restore the around-the-denoiser routing
  without the geometry hacks.

- **Per-direction cloud LUT.** Would let indirect / PSR / reflection
  rays see Nubis Cubed clouds, removing the primary-vs-reflection
  discontinuity.
- **Half-res reprojection.** Decima paper pp. 174-176 -- a follow-on
  perf path that would let `cloudViewSamples` rise without proportional
  cost.
- **Runtime-baked NVDF + SDF.** A C-procedural cloud field replacing
  the prebaked FBM noise volume; preserves the macro/micro decoupling
  at cumulus silhouettes that the FBM cannot.
- **Sun-direct denoiser channel (supersedes the old post-DLSS re-modulation
  item).** With the cloud shadow now folded onto the sun radiance pre-denoise,
  NRD / DLSS-RR can soften the cumulus shadow edges. A dedicated denoised
  sun-direct buffer (option (a) above) would let the high-frequency cumulus
  pattern be re-applied after upscale at full resolution, recovering the crisp
  edges the old screen-space texture preserved -- but without any of the
  geometry-blindness hacks, since it would still be a sun-only signal.

## Cross-references

- [Weather Presets -- Plugin Integration Guide](integrators/weather-presets.md) -- how plugins drive per-preset cloud parameter values, particle coordination, and per-preset drift personalities.
- [Rtx Options](../RtxOptions.md) -- search `rtx.atmosphere.cloud*` for the full per-knob reference.
- [Fork Touchpoints](fork-touchpoints.md) -- inventory of every upstream file the cloud system touches (in particular `submodules/rtxdi/.../volume_integrator.slangh` for the volumetric sun-NEE wire-in).
