# Remix Sky API

The sky / atmosphere system exposes a plugin-facing surface through the
standard Remix C API — no dedicated `remixapi_SetSky` function exists.
Control flows through two channels documented here:

- **Direct knobs (`SetConfigVariable`).** A plugin pushes individual
  `rtx.atmosphere.*` and `rtx.sky*` options per-frame (or once at
  startup). This is the lowest-level interface; it's how the FNV and
  FO4 plugins drive sun, stars, moons, and clouds today.
- **Weather preset blender (`SetGameValue` + `GetGameValue`).** A
  higher-level layer that names twelve weather archetypes (clear,
  overcast, thunderstorm, …) and transitions smoothly between them.
  The blender writes into the same `rtx.atmosphere.*` knobs the
  direct surface exposes, so the two layers compose naturally.

For the channels themselves, see
[`RemixApi.md`](RemixApi.md#configuration--setconfigvariable) and
[`RemixApi.md`](RemixApi.md#game-state--setgamevalue--getgamevalue).
For the fork-side implementation, see
[`rtx_fork_atmosphere.cpp`](../src/dxvk/rtx_render/rtx_fork_atmosphere.cpp),
[`rtx_fork_atmosphere.h`](../src/dxvk/rtx_render/rtx_fork_atmosphere.h),
[`rtx_fork_weather.cpp`](../src/dxvk/rtx_render/rtx_fork_weather.cpp),
and [`rtx_fork_weather.h`](../src/dxvk/rtx_render/rtx_fork_weather.h).

## Format conventions

Plugin-side: `SetConfigVariable(key, value)` where `value` is a string.

- `float` → `"%.4f"` (e.g. `"15.0000"`)
- `bool` → `"True"` or `"False"`
- `Vector3` → `"x, y, z"` (e.g. `"0.85, 0.87, 0.92"`)
- `uint32_t` → decimal (e.g. `"0"`, `"1"`)

`NoSave = ✓` means the option is excluded from `user.conf`
serialization on shutdown — safe to push every frame from a game
plugin without polluting the user's config file.

## Layer 1 — Direct `rtx.atmosphere.*` knobs

### Mode switch

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.skyMode` | int | `0` | Sky rendering mode. `0` = SkyboxRasterization (traditional skybox); `1` = Numos (Hillaire scattering). All `rtx.atmosphere.*` options below only take effect in mode `1`. |

### Sun pose (game-driven, push every frame)

These persist to `user.conf` (no `NoSave` flag), but a plugin driving
time-of-day should push them every frame — the last write wins.

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.atmosphere.sunElevation` | float | `15.0` | Sun angle from horizon, degrees. -90 to +90. |
| `rtx.atmosphere.sunRotation` | float | `0.0` | Sun azimuth (rotation around zenith), degrees. 0–360. |

### Sun appearance (persistent)

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.atmosphere.sunSize` | float | `0.545` | Sun disc angular diameter, degrees. |
| `rtx.atmosphere.sunIntensity` | float | `1.0` | Sun brightness multiplier. |
| `rtx.atmosphere.sunIlluminance` | Vector3 | `20, 20, 20` | Base illuminance color/intensity (multiplied by `sunIntensity`). |

### Atmosphere physical properties (persistent)

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.atmosphere.altitude` | float | `100.0` | Camera altitude above sea level, meters. |
| `rtx.atmosphere.airDensity` | float | `1.0` | Air molecule density multiplier (1 = clear sky). |
| `rtx.atmosphere.aerosolDensity` | float | `1.0` | Aerosol/dust density multiplier (1 = typical haze). |
| `rtx.atmosphere.ozoneDensity` | float | `1.0` | Ozone layer density multiplier (1 = typical). |
| `rtx.atmosphere.planetRadius` | float | `6371.0` | Planet radius, km. Affects horizon curvature in atmospheric math. |
| `rtx.atmosphere.atmosphereThickness` | float | `100.0` | Atmosphere thickness, km. |
| `rtx.atmosphere.mieAnisotropy` | float | `0.97` | Mie phase g parameter, -1 to 1. Controls haze forward-scatter. |
| `rtx.atmosphere.rayleighScattering` | Vector3 | `0.0058, 0.0135, 0.0331` | Base Rayleigh scattering coefficients, km⁻¹. |
| `rtx.atmosphere.mieScattering` | Vector3 | `0.003996, 0.003996, 0.003996` | Base Mie scattering coefficients, km⁻¹. |
| `rtx.atmosphere.ozoneAbsorption` | Vector3 | `0.00204, 0.00497, 0.000214` | Base ozone absorption coefficients, km⁻¹. |
| `rtx.atmosphere.ozoneLayerAltitude` | float | `25.0` | Ozone layer peak altitude, km. |
| `rtx.atmosphere.ozoneLayerWidth` | float | `15.0` | Ozone layer thickness, km. |

### Stars — pose & rotation (game-driven, push every frame)

These persist to `user.conf` (no `NoSave` flag), but a time-of-day plugin
should push them every frame alongside the sun pose.

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.atmosphere.starRotation` | float | `0.0` | Sidereal sky rotation angle, degrees, 0–360. The entire star pattern rotates as one rigid body around the celestial pole. |
| `rtx.atmosphere.starBrightness` | float | `0.5` | Overall star brightness multiplier. Fade to 0 at sunrise, restore at sunset. |

### Stars — celestial pole (persistent, set once at startup or via rtx.conf)

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.atmosphere.starAxisElevation` | float | `90.0` | Celestial pole elevation from horizon, degrees. `90` = pole at zenith (default, behavior matches pre-rotation). Set to the game world's latitude for a realistic star trail. |
| `rtx.atmosphere.starAxisRotation` | float | `0.0` | Celestial pole azimuth, degrees (`0` = North). Only relevant when `starAxisElevation != 90`. |

### Night sky appearance (persistent)

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.atmosphere.starDensity` | float | `0.5` | Star density linear-feel slider: 0 = no stars, 1 = maximum. Mapped internally via `pow(x,4)*0.05`. 0.5 ≈ 0.3% of cells lit; 1.0 ≈ 5%. |
| `rtx.atmosphere.starTwinkleSpeed` | float | `1.0` | Twinkle animation speed. 0 = static. |
| `rtx.atmosphere.starPsfSharpness` | float | `20.0` | PSF Gaussian exponent controlling per-star disc spread. Lower = softer, wider stars. Useful range 8–30. |
| `rtx.atmosphere.starCloudExtinctionPower` | float | `2.5` | Power exponent applied to cloud transmittance when extincting stars. Raises the effective opacity so bright pinpoints die inside thick cumulus. 1.0 = standard composite. |
| `rtx.atmosphere.starAmbientCouplingStrength` | float | `0.005` | How strongly starlight/airglow couples into cloud-march night ambient. 0 disables. |
| `rtx.atmosphere.nightSkyBrightness` | float | `0.008` | Ambient airglow and zodiacal-light brightness. |
| `rtx.atmosphere.nightSkyColor` | Vector3 | `0.15, 0.2, 0.4` | Airglow color tint. |
| `rtx.atmosphere.milkyWayEnabled` | bool | `False` | Master toggle for the galactic-band Milky Way overlay. Opt-in; off by default. |
| `rtx.atmosphere.milkyWayDensityBoost` | float | `0.3` | Extra star density inside the galactic band when Milky Way is enabled. |
| `rtx.atmosphere.milkyWayBackgroundBrightness` | float | `0.05` | Diffuse background glow of the band (unresolved stars + dust). |
| `rtx.atmosphere.milkyWayBackgroundColor` | Vector3 | `0.5, 0.55, 0.75` | Outer-edge band tint (cool blue periphery). |
| `rtx.atmosphere.milkyWayCoreColor` | Vector3 | `1.0, 0.85, 0.55` | Galactic-centre tint (warm yellow-cream). |
| `rtx.atmosphere.milkyWayDustColor` | Vector3 | `0.15, 0.08, 0.05` | Dust-lane tint (dark red-brown patches). |
| `rtx.atmosphere.milkyWayDustAmount` | float | `0.6` | Dust-lane contrast strength. 0 = smooth band, 1 = full contrast. |

### Per-moon options

4 moons available. Replace `N` with `0`, `1`, `2`, or `3` in both the
category segment AND the trailing field suffix (the suffix is part of
how the options are declared via macro expansion — `moon0.enabled0`
etc.).

#### Moon pose (game-driven, push every frame)

These persist to `user.conf` (no `NoSave` flag), but a time-of-day plugin
should push them every frame.

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.atmosphere.moonN.elevationN` | float | `45.0` | Moon angle from horizon, degrees. -90 to +90. |
| `rtx.atmosphere.moonN.rotationN` | float | `90.0` | Moon azimuth (rotation around zenith), degrees. 0–360. |
| `rtx.atmosphere.moonN.phaseN` | float | `0.5` | Lunar phase: 0 = new moon, 0.5 = full, 1.0 = new again. |

#### Moon appearance (persistent)

Core per-moon visual properties:

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.atmosphere.moonN.enabledN` | bool | `False` | Enable this moon. **Must be set `True` for the moon to render.** All moons are off by default — opt in per-game. |
| `rtx.atmosphere.moonN.angularRadiusN` | float | `3.5` | Angular diameter, degrees. |
| `rtx.atmosphere.moonN.brightnessN` | float | `1.0` | Brightness multiplier. 1.0 = physical neutral; raise for stylized scenes. |
| `rtx.atmosphere.moonN.colorN` | Vector3 | `0.12, 0.12, 0.12` | Surface albedo. Default ≈ Earth's lunar Bond albedo; raise per-channel for tinted moons. |
| `rtx.atmosphere.moonN.surfaceStyleN` | uint32 | `0` | Surface preset: `0` = Rocky, `1` = Volcanic. |
| `rtx.atmosphere.moonN.craterDensityN` | float | `1.0` | Crater density multiplier [0, 1]. |
| `rtx.atmosphere.moonN.surfaceContrastN` | float | `1.0` | Surface light/dark contrast multiplier. |
| `rtx.atmosphere.moonN.surfaceNoiseScaleN` | float | `1.0` | Surface feature size multiplier. |
| `rtx.atmosphere.moonN.darkSideBrightnessN` | float | `0.005` | Dark-side brightness as fraction of the lit side. |
| `rtx.atmosphere.moonN.roughnessAmountN` | float | `1.0` | Micro-detail surface roughness amplitude. |

World-lighting strength controls (shared across all moons):

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.atmosphere.moonNeeStrength` | float | `1.0` | Master multiplier on all direct moon lighting paths (surface NEE + clouds + volumetric). 0 = moon does not light the world. |
| `rtx.atmosphere.moonAtmosphericCouplingStrength` | float | `1.0` | Sky-side multiplier on the moon's contribution to atmospheric scattering. 0 = no atmospheric blue-dome around the moon. |
| `rtx.atmosphere.surfaceMoonBrightness` | float | `50.0` | Per-path stylistic multiplier on surface ground-moonlight NEE. Default tuned for the FNV tonemapper; 1.0 = physically pure (very dim). |
| `rtx.atmosphere.cloudMoonBrightness` | float | `2.0` | Per-path stylistic multiplier on cloud directional moonlight + ambient airglow. |
| `rtx.atmosphere.haloMoonBrightness` | float | `15.0` | Per-path stylistic multiplier on the moon disc halo Gaussian glow. |

Moon cloud-look and halo shape constants:

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.atmosphere.moonCloudDiffuseGain` | float | `0.10` | Lambert diffuse weight for off-axis cloud illumination. Lower = stronger contrast. |
| `rtx.atmosphere.moonCloudPhaseGain` | float | `1.0` | HG phase weight for peak silver-lining intensity directly in front of the moon. |
| `rtx.atmosphere.moonCloudAnisotropy` | float | `0.85` | HG anisotropy for cloud-moon forward scatter. Higher = sharper silver-lining peak. |
| `rtx.atmosphere.moonSilverLiningIntensity` | float | `1.0` | Master multiplier on the combined cloud-moon silver-lining term. |
| `rtx.atmosphere.moonHaloMagnitude` | float | `0.0015` | Disc halo Gaussian shape strength. Use `haloMoonBrightness` for tonemapper correction, this for the shape. |
| `rtx.atmosphere.moonHaloGlowStrength` | float | `1.0` | Master multiplier on the combined halo + ambient airglow contribution. |
| `rtx.atmosphere.moonAmbientAirglow` | float | `0.0015` | Ambient airglow per-moon contribution to the cloud night-light term. |

### Clouds (procedural volumetric layer)

A volumetric cloud layer rendered with the Nubis Cubed evaluator.
Density is FBM noise gated by `cloudCoverageMean`; lighting is multi-scatter
Nubis-Cubed with voxel-grid terrain shadows and a per-moon night
ambient. Clouds composite on top of stars, moons, and atmosphere, and
dim the ground via the sun-transmittance path when overcast.

#### Cloud weather (game-driven, push every frame)

The primary coverage knob is `cloudCoverageMean`. The old `cloudCoverage`
key from earlier plugin code no longer exists — update callers to push
`cloudCoverageMean` instead. These persist to `user.conf` (no `NoSave`
flag) but should be pushed every frame from a weather-driving plugin.

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.atmosphere.cloudCoverageMean` | float | `0.85` | Mean cloud coverage across the sky [0, 1]: 0 = clear, 1 = overcast. Primary game-weather knob. |
| `rtx.atmosphere.cloudCoverageSpread` | float | `1.0` | Spatial variation amplitude for coverage [0, 1]. 0 = uniform sky, 1 = full patchy range. |
| `rtx.atmosphere.cloudCoverageNoiseScale` | float | `0.0033` | Spatial frequency of the coverage noise. Smaller = larger coverage patches. |

#### Cloud appearance (persistent)

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.atmosphere.cloudEnabled` | bool | `True` | Master switch for the procedural cloud layer. |
| `rtx.atmosphere.cloudDensity` | float | `1.65` | Cloud opacity/density multiplier. Scales extinction along the march. |
| `rtx.atmosphere.cloudAltitude` | float | `1.3` | Cloud-layer base altitude, km. |
| `rtx.atmosphere.cloudColor` | Vector3 | `0.89, 0.92, 1.0` | Base cloud albedo. Tint away from white for stylized color grading. |
| `rtx.atmosphere.cloudWindSpeed` | float | `0.02` | Cloud drift speed, km/s. Wind offset accumulates from elapsed time. |
| `rtx.atmosphere.cloudWindDirection` | float | `45.0` | Wind direction, degrees. 0 = +X, 90 = +Z. |
| `rtx.atmosphere.cloudShadowStrength` | float | `1.0` | How strongly overcast clouds dim ground and atmosphere lighting [0, 1]. 1 = full voxel-grid contribution; 0 = shadows muted. |
| `rtx.atmosphere.cloudAnisotropy` | float | `0.6` | Henyey-Greenstein g for cloud forward-scatter (silver lining). |

#### Cloud volumetric & polish (advanced — persistent)

Ray-march quality:

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.atmosphere.cloudViewSamples` | uint32 | `32` | Ray-march steps through the cloud slab. Range 1–32. Higher = better quality, higher cost. |
| `rtx.atmosphere.cloudThickness` | float | `2.75` | Vertical depth of the cloud slab, km. |
| `rtx.atmosphere.cloudCurvature` | float | `0.38` | Sky-dome curvature for the cloud sphere: 0 = real-planet radius (nearly flat ceiling), 1 = tight dome. Does not affect atmospheric math. |

Shadow and color polish:

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.atmosphere.cloudShadowTint` | Vector3 | `0.55, 0.65, 0.85` | Sky-blue bounce color applied on the shadow side of clouds. |
| `rtx.atmosphere.cloudShadowTintStrength` | float | `1.0` | Strength of the shadow tint contribution [0, 1]. |
| `rtx.atmosphere.cloudSunsetWarmth` | float | `0.95` | Warm tint strength on the sunward side at low sun angles. 0 = disabled. |
| `rtx.atmosphere.cloudVoxelShadowsEnable` | bool | `True` | Use the D_sun voxel grid for cumulus-shaped cloud-on-terrain shadows at NEE entry points. Replaces the legacy uniform coverage proxy. |
| `rtx.atmosphere.cloudShadowMarchStrength` | float | `1.0` | Beer-Lambert exponent multiplier inside the voxel-grid shadow lookup. Higher = darker cumulus shadows on terrain. Only active when `cloudVoxelShadowsEnable` is on. |
| `rtx.atmosphere.cloudShadowFactorStrength` | float | `4.0` | Post-denoise pow exponent on the per-pixel cloud shadow factor at composite time. 1.0 = unchanged; higher deepens shadow contrast without re-baking the voxel grid. |

Aerial perspective (distant cloud softening):

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.atmosphere.cloudAerialHazePerKm` | float | `0.05` | Per-km haze extinction on cloud radiance. Dims distant samples toward atmospheric color (softness). Does not eliminate the horizon white-wall alone. |
| `rtx.atmosphere.cloudAerialFadePerKm` | float | `0.15` | Per-km fade extinction on cloud alpha accumulation. Prevents horizon-grazing rays from building a solid white wall. |

Volumetric sky-ambient:

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.atmosphere.cloudSkyAmbientStrength` | float | `0.0` | Strength of sky-view LUT radiance injected into the volumetric froxel pass [0, 3]. 0 = feature disabled (default-off). 1 = physical baseline. Requires `rtx.skyMode = 1`. |
| `rtx.atmosphere.cloudSkyAmbientCloudOcclusionStrength` | float | `1.0` | Strength of cloud occlusion on the sky-ambient term [0, 1]. 1 = full physical occlusion; 0 = debug only. |

Nubis Cubed lighting equations:

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.atmosphere.cloudRenderRTEnable` | bool | `True` | Use the Nubis Cubed cloud render RT for primary sky-miss rays. Indirect/PSR/reflection rays continue to use analytical evalClouds. |
| `rtx.atmosphere.cloudMultiScatterStrength` | float | `1.0` | Master multiplier on Wrenninge multi-scatter sun response. 1.0 = Hillaire/Frostbite baseline. |
| `rtx.atmosphere.cloudMultiScatterOctaves` | uint32 | `3` | Wrenninge octave count [1, 4]. 1 = single anisotropic term (no multi-scatter). |
| `rtx.atmosphere.cloudMsScale` | float | `1.0` | Multiplier on the Nubis Cubed sigma_ms remap term [0, 2]. Higher brightens cumulus bottoms. |
| `rtx.atmosphere.cloudPhaseG1` | float | `0.8` | Primary HG asymmetry — strong forward-scatter, drives silver lining at backlit edges. |
| `rtx.atmosphere.cloudPhaseG2` | float | `0.3` | Secondary HG asymmetry — broader in-scatter envelope. |
| `rtx.atmosphere.cloudMsSunDotMax` | float | `0.9` | Nubis Cubed sigma_ms remap upper bound on sun_dot. |
| `rtx.atmosphere.cloudMsSigmaShallow` | float | `0.25` | Nubis Cubed sigma_ms at cloud surface / shallow penetration. |
| `rtx.atmosphere.cloudMsSigmaDeep` | float | `0.05` | Nubis Cubed sigma_ms deep inside cloud. |
| `rtx.atmosphere.cloudMsSdfDepth` | float | `128.0` | SDF depth (meters) at which sigma_ms saturates to the deep value. |

Cloud spatial variation (Nubis-style type/coverage regions):

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.atmosphere.cloudTypeMean` | float | `0.75` | Mean cloud type [0, 1]: 0 = stratus, 0.5 = stratocumulus, 1 = cumulus. |
| `rtx.atmosphere.cloudTypeSpread` | float | `0.5` | Spatial variation amplitude for cloud type [0, 1]. |
| `rtx.atmosphere.cloudTypeNoiseScale` | float | `0.001` | Spatial frequency for type noise. Smaller = larger type-region features. |
| `rtx.atmosphere.cloudAnvilBias` | float | `0.3` | Cumulus anvil top inflation [0, 1]. 0 = flat tops, 1 = mushroom-cap spread. |

Cloud noise bake (changes apply on game relaunch):

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.atmosphere.cloudNoiseTileKm` | float | `12.0` | World-space tile period (km) for the 3D cloud noise texture. Smaller = more visible tiling. **Relaunch required.** |
| `rtx.atmosphere.cloudWorleyCarveStrength` | float | `0.6` | Worley FBM carve strength subtracted from the Perlin base. 0 = smooth blobs; 1 = aggressive cell silhouettes. **Relaunch required.** |
| `rtx.atmosphere.cloudWorleyFrequency` | float | `1.0` | Worley feature-point density, cycles per km. Smaller = larger cumulus cells. **Relaunch required.** |
| `rtx.atmosphere.cloudWorleyOctaves` | uint32 | `3` | Worley FBM octave count [1, 4]. **Relaunch required.** |

Height LUT and two-layer clouds:

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtx.atmosphere.cloudHeightLutEnable` | bool | `True` | Use a 64×128 baked height LUT for per-altitude cloud shape instead of the procedural trapezoid. Only affects the screen-space cloud render pass. |
| `rtx.atmosphere.cloudLayer2Enable` | bool | `True` | Enable a second cloud slab (cirrus deck by default) composited on top of the primary cumulus layer. |
| `rtx.atmosphere.cloudLayer2Altitude` | float | `5.5` | Base altitude (km) of the second slab. |
| `rtx.atmosphere.cloudLayer2Thickness` | float | `2.0` | Vertical depth (km) of the second slab. |
| `rtx.atmosphere.cloudLayer2TypeMean` | float | `0.6` | Mean cloud type for layer 2 [0, 1]. Low values produce stratus/cirrus shapes. |
| `rtx.atmosphere.cloudLayer2TypeSpread` | float | `1.0` | Cloud-type variation for layer 2. Independent of layer 1. |
| `rtx.atmosphere.cloudLayer2CoverageMean` | float | `0.85` | Mean coverage for layer 2 [0, 1]. |
| `rtx.atmosphere.cloudLayer2CoverageSpread` | float | `0.0` | Coverage variation for layer 2. Independent of layer 1. |
| `rtx.atmosphere.cloudLayer2DensityScale` | float | `0.65` | Per-step density multiplier for layer 2. Keep low for optically-thin cirrus. |
| `rtx.atmosphere.cloudLayer2NoiseSeed` | float | `1000.0` | Seed offset for layer 2 coverage/type noise. Any non-zero value decorrelates it from layer 1's pattern. |

## Layer 2 — Weather preset blender (`__weather.*`)

The weather system exposes a fork-side preset blender that drives the
same `rtx.atmosphere.*` knobs above through a higher-level preset
abstraction.

### The three surfaces

| Surface | Channel | Key prefix | Who writes |
| :-- | :-- | :-- | :-- |
| Trigger transitions | `SetGameValue` | `__weather.target`, `__weather.blend_seconds` | Game / plugin |
| Read status | `GetGameValue` | `__weather.current`, `__weather.previous`, `__weather.blend_progress` | Game / plugin reads; fork writes |
| Tune presets | `SetConfigVariable` | `rtx.weather.preset.<name>.<field>` | Modder / plugin (mostly via `user.conf`) |

### Surface 1 — Trigger transitions

Setting `__weather.target` to a known preset name activates the
blender. The blender lerps every weather field from the previous
preset to the new one over `__weather.blend_seconds` (default `1.0`).

```c
// Example: switch to thunderstorm over 5 seconds
iface.SetGameValue("__weather.blend_seconds", "5.0");
iface.SetGameValue("__weather.target",        "thunderstorm");
```

| Key | Type | Default | Meaning |
| :-- | :-: | :-: | :-- |
| `__weather.target` | string (preset name) | absent | Target preset to blend toward. Setting to an unknown name logs a warning and the blender goes dormant. Clear by setting to `""`. |
| `__weather.blend_seconds` | float-as-string | `"1.0"` | Blend duration. Clamped to `>= 0.001`. |
| `__weather.drift_speed` | float-as-string | `"1.0"` | Cloud-drift phase advance multiplier. Higher = faster evolution. `0` freezes drift. Smoothed inside the renderer with tau = 1.0s. |
| `__weather.drift_intensity` | float-as-string | `"1.0"` | Cloud-drift swing amplitude multiplier. `0` disables drift entirely. Smoothed inside the renderer with tau = 1.0s. |

Setting `target` mid-blend retargets cleanly: the partial blend state
becomes the new "previous" snapshot, and the timer restarts toward the
new target.

The blender is **dormant when `__weather.target` is absent or
unknown** — there is zero behavioral change for runs that do not opt
in.

### Surface 2 — Read status

Once the blender is active, it publishes its state back to the
GameStateStore each frame:

| Key | Type | Meaning |
| :-- | :-: | :-- |
| `__weather.current` | string | Currently-targeted preset name. |
| `__weather.previous` | string | Preset being blended *from*. |
| `__weather.blend_progress` | float-as-string, `0.0`–`1.0` | Fraction of the blend completed. `1.0` once the target is reached. |

Read with the standard two-call sized `GetGameValue` pattern:

```c
char buf[64];
uint32_t actual = 0;
iface.GetGameValue("__weather.blend_progress", buf, sizeof(buf), &actual);
if (actual > 0 && actual <= sizeof(buf)) {
    float progress = strtof(buf, NULL);
    // ...
}
```

### Surface 3 — Tune presets

Each preset has 42 RTX_OPTIONs declared under
`rtx.weather.preset.<name>.<field>` — generated from a single X-macro
at [`rtx_fork_weather.h`](../src/dxvk/rtx_render/rtx_fork_weather.h).

```c
// Example: make 'foggy' fog denser
iface.SetConfigVariable("rtx.weather.preset.foggy.aerosolDensity",                       "3.0");
iface.SetConfigVariable("rtx.weather.preset.foggy.transmittanceMeasurementDistanceMeters", "40.0");
```

Or, equivalently, in `user.conf` with no code at all:

```ini
rtx.weather.preset.foggy.aerosolDensity = 3.0
rtx.weather.preset.foggy.transmittanceMeasurementDistanceMeters = 40.0
```

Modder workflow — retune what each archetype looks like — is the
intended use of this surface. Per-frame tuning from a game is unusual;
prefer Surface 1 + a custom preset.

### Preset names

The 12 valid values for `__weather.target`:

| Preset | Description |
| :-- | :-- |
| `clear` | Sunny, crisp, low haze |
| `partlyCloudy` | Light scattered clouds |
| `overcast` | Default look — uniform cloud cover |
| `hazy` | Warm summer haze |
| `foggy` | Headline fog preset, low visibility |
| `drizzle` | Light rain, medium fog |
| `rainstorm` | Heavy clouds, dim sun, dense fog |
| `thunderstorm` | Heaviest, bruised tone |
| `snow` | Medium clouds, cool fog |
| `blizzard` | Whiteout, severe visibility loss |
| `sandstorm` | Yellow-orange forward-scattering fog |
| `smoggy` | Industrial dark grey-brown haze |

### Per-preset field list (42 fields)

Single source of truth: `WEATHER_PRESET_FIELD_LIST` in
[`rtx_fork_weather.h`](../src/dxvk/rtx_render/rtx_fork_weather.h).
Adding a field there propagates to all 12 presets, the
`WeatherSnapshot` struct, and the blender automatically.

The full 348-row table of `(preset × field → value)` lives in
[`RtxOptions.md`](../RtxOptions.md) under the
`rtx.weather.preset.*` namespace. For the per-field neutral defaults
and the field categories (Cloud / Atmosphere / Sky-mood /
Volumetric), see the field-list X-macro in the header.

## How the two layers compose

When both layers are active, the weather blender writes the same
`rtx.atmosphere.*` values that a direct `SetConfigVariable` push
would. Both layers ultimately drive the same RTX_OPTIONs, so the
runtime applies last-writer-wins per frame.

A plugin that wants weather presets as a baseline plus selective
overrides should: (1) set `__weather.target` to drive the bulk of the
sky, and (2) push specific `rtx.atmosphere.*` keys *after* the
GameStateStore tick each frame to override individual knobs. The
override pushes win because they happen later in the frame.

## Concrete examples

### Enable moon 0 once at startup

```cpp
static bool moon0Enabled = false;
if (!moon0Enabled) {
  if (RemixRenderer::SetConfigVariable("rtx.atmosphere.moon0.enabled0", "True")) {
    moon0Enabled = true;
  }
}
```

### Drive sun + moon 0 from in-game time

```cpp
const float hour = cachedGameHour->value;
const float kPi = 3.14159265358979323846f;

// Sun: peaks at noon, dips below horizon at night
const float sunElev = std::sin((hour - 6.0f) / 12.0f * kPi) * 90.0f;
const float sunRot  = (hour / 24.0f) * 360.0f;
PushFloat("rtx.atmosphere.sunElevation", sunElev);
PushFloat("rtx.atmosphere.sunRotation",  sunRot);

// Moon 0: opposite the sun
PushFloat("rtx.atmosphere.moon0.elevation0", -sunElev);
PushFloat("rtx.atmosphere.moon0.rotation0",  std::fmod(sunRot + 180.0f, 360.0f));
PushFloat("rtx.atmosphere.moon0.phase0",     0.5f);  // full moon for now
```

### Rotating star field synced to game time, with sunrise/sunset fade

```cpp
// Sidereal rotation: full sweep over the in-game day, in lock-step with sun.
const float starRot = std::fmod(sunRot, 360.0f);  // simplest: same rate as sun
PushFloat("rtx.atmosphere.starRotation", starRot);

// Fade stars out as the sun climbs above the horizon.
const float kStarMaxBrightness = 8.0f;  // matches default
const float fadeT = std::clamp((sunElev - 6.0f) / 6.0f, 0.0f, 1.0f);
const float starBright = kStarMaxBrightness * (1.0f - fadeT);
PushFloat("rtx.atmosphere.starBrightness", starBright);

// One-time pole config (do this on first successful weather push, then never again).
static bool poleConfigured = false;
if (!poleConfigured) {
  if (RemixRenderer::SetConfigVariable("rtx.atmosphere.starAxisElevation", "42.0") &&
      RemixRenderer::SetConfigVariable("rtx.atmosphere.starAxisRotation",  "0.0")) {
    poleConfigured = true;
  }
}
```

### Drive cloud coverage from in-game weather

```cpp
const float coverageCurrent  = weatherCloudiness(currentWeather);   // 0..1
const float coverageOutgoing = weatherCloudiness(outgoingWeather);  // 0..1
const float coverage = std::lerp(coverageOutgoing, coverageCurrent,
                                 weatherTransition);  // 0..1
PushFloat("rtx.atmosphere.cloudCoverageMean", coverage);
```

### Trigger a weather preset from C

```c
#include "remix/remix_c.h"
#include <stdio.h>
#include <stdlib.h>

void trigger_rainstorm(remixapi_Interface* iface) {
    iface->SetGameValue("__weather.blend_seconds", "3.0");
    iface->SetGameValue("__weather.target",        "rainstorm");
}

float poll_blend_progress(remixapi_Interface* iface) {
    char buf[32];
    uint32_t actual = 0;
    if (iface->GetGameValue("__weather.blend_progress", buf, sizeof(buf), &actual) != REMIXAPI_ERROR_CODE_SUCCESS)
        return 0.0f;
    if (actual == 0 || actual > sizeof(buf))
        return 0.0f;
    return strtof(buf, NULL);
}
```

### Example string conversions

```cpp
// float
char buf[32]; std::snprintf(buf, sizeof(buf), "%.4f", value);
SetConfigVariable("rtx.atmosphere.sunElevation", buf);

// bool
SetConfigVariable("rtx.atmosphere.moon0.enabled0", "True");

// Vector3
char vbuf[64];
std::snprintf(vbuf, sizeof(vbuf), "%.4f, %.4f, %.4f", c.r, c.g, c.b);
SetConfigVariable("rtx.atmosphere.moon0.color0", vbuf);

// uint32
SetConfigVariable("rtx.atmosphere.moon0.surfaceStyle0", "1");  // Volcanic
```

## Bridge support

Both `SetGameValue` / `GetGameValue` and `SetConfigVariable` are wired
through the 32-bit ↔ 64-bit bridge (see `bridge/src/client/remix_api.cpp`),
so 32-bit games using the bridge path get the same sky / weather
control as 64-bit native consumers. No special bridge call needed.

## Notes

- The fork's developer ImGui surface (Atmosphere and Weather panels in
  the Remix UI) drives the same `rtx.atmosphere.*` and `__weather.*`
  keys — there is no privileged in-engine path. Plugins and the dev UI
  share state through GameStateStore and the RtxOption registry.
- The `__` prefix is the convention for fork-side game-state keys read
  by C++ subsystems. Plugin-private keys without the prefix are fine,
  but use the prefix when collaborating with fork-side systems.
- The `WeatherSnapshot` struct (auto-generated members from the same
  X-macro) is the in-memory representation the blender uses internally.
  It is not exposed through the C API — plugins drive the system by
  string keys only.

## See also

- [`RemixApi.md`](RemixApi.md) — typed C API reference (hand-written).
- [`RemixApiSurface.md`](../RemixApiSurface.md) — auto-generated flat
  list of every C API surface element.
- [`RtxOptions.md`](../RtxOptions.md) — auto-generated table of all
  `rtx.*` keys, including the full `rtx.atmosphere.*` and
  `rtx.weather.preset.*` namespaces with their live defaults.
- [`rtx_fork_atmosphere.h`](../src/dxvk/rtx_render/rtx_fork_atmosphere.h) —
  sky / atmosphere RTX_OPTION declarations.
- [`rtx_fork_weather.h`](../src/dxvk/rtx_render/rtx_fork_weather.h) —
  weather field-list X-macro and per-preset values.
- [`rtx_fork_weather.cpp`](../src/dxvk/rtx_render/rtx_fork_weather.cpp) —
  blender implementation.
