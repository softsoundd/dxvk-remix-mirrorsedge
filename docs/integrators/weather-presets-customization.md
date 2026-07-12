# Weather Presets -- Per-Game Customization Guide

This document is for game integrators tuning the shipped weather preset values
to match their specific game's visual style.

---

## 1. Overview

Every game's "Storm" looks different. A Fallout desert game has a warm, dusty
rainstorm with low-angle sun and rust-colored fog. A northern city game has
a cold, blue-grey rainstorm with near-whiteout visibility. Both can use the
`rainstorm` preset as a starting point, with per-game overrides in `user.conf`
to push it toward the intended look.

### How customization works

All 504 weather RTX_OPTIONs (12 presets x 42 fields) accept overrides via the
standard `user.conf` mechanism. The naming convention is:

```
rtx.weather.preset.<presetName>.<presetName>_<fieldName> = <value>
```

Overrides in `user.conf` take effect on the next application launch. They sit
in the User layer of the RTX_OPTION system, which overrides the compiled-in
defaults but is overridden in turn by the Derived layer (which the blender
writes during active transitions).

---

## 2. The Naming Convention

### Preset names (12 total)

| Name | Description |
|------|-------------|
| `clear` | Sunny, crisp, low haze |
| `partlyCloudy` | Light scattered clouds |
| `overcast` | Full cloud cover, current default look |
| `hazy` | Warm summer haze |
| `foggy` | Headline fog preset |
| `drizzle` | Light rain, medium fog |
| `rainstorm` | Heavy clouds, dim sun, dense fog |
| `thunderstorm` | Heaviest clouds, bruised tone |
| `snow` | Medium clouds, cool fog |
| `blizzard` | Whiteout, severe visibility loss |
| `sandstorm` | Yellow-orange forward-scattering dust |
| `smoggy` | Industrial dark grey-brown haze |

---

## Cloud Drift Customization

Cloud drift mechanics (which parameters drift, with what relative amplitude)
live renderer-side as a fixed in-code table; they are NOT exposed as
RTX_OPTIONs in v1 and therefore cannot be customized via `user.conf`.

What IS plugin-customizable: drift speed and intensity per preset, via the
`__weather.drift_speed` and `__weather.drift_intensity` GameStateStore keys.
See [`weather-presets.md`](weather-presets.md) section 8 for the recommended
per-preset values and the integration pattern.

If a per-game tuning need emerges that requires changing which fields drift
or their relative amplitudes, those values can be promoted to RTX_OPTIONs
in a follow-on spec.

### Example user.conf entries

```ini
# Make the foggy preset significantly denser
rtx.weather.preset.foggy.foggy_transmittanceMeasurementDistanceMeters = 30.0
rtx.weather.preset.foggy.foggy_aerosolDensity = 3.5

# Warm up the rainstorm preset for a desert game
rtx.weather.preset.rainstorm.rainstorm_sunIlluminance = 6.0, 5.5, 4.0
rtx.weather.preset.rainstorm.rainstorm_transmittanceColor = 0.80, 0.72, 0.60

# Give the sandstorm a more extreme forward-scattering lobe
rtx.weather.preset.sandstorm.sandstorm_volumetricAnisotropy = 0.80
rtx.weather.preset.sandstorm.sandstorm_singleScatteringAlbedo = 0.88, 0.70, 0.45
```

---

## 3. The "Tune in Dev Menu, Dump to user.conf" Workflow

The fastest way to dial in preset values is to use the in-game dev menu and
then record the final values in `user.conf`.

### Step-by-step

1. Launch the game with Remix loaded and open the dev menu (default key: `~`).
2. Navigate to **Rendering -> Atmosphere -> Weather Presets -> Tune Preset
   Defaults**.
3. Select the preset you want to tune from the dropdown.
4. Drag the sliders for the parameters you want to change. The atmospheric
   rendering updates in real time.
5. When the values look right, note them down (or use a screenshot).
6. Add the corresponding `rtx.weather.preset.<name>.<field>` lines to your
   game's `user.conf`.

### Layer behavior

- **In-game ImGui slider writes** go to the Persistent RTX_OPTION layer, which
  persists across sessions in the runtime config file.
- **user.conf overrides** go to the User layer and take effect on next launch,
  overriding compiled-in defaults.
- **Active blender writes** go to the Derived layer each frame, which sits on
  top of both Persistent and User. During a blend transition, the blender owns
  these values. Once a transition completes, the Derived values remain until
  the next transition.

For shipping per-game defaults, `user.conf` is the correct surface. The in-game
sliders are for iterative tuning; `user.conf` is for stable per-game delivery.

### Pause toggle

When **Pause Weather Blender** is checked in the dev menu, the blender stops
writing to RTX_OPTIONs entirely. This lets you drag atmosphere sliders and
see the raw knob values without the blender overwriting them each frame.
Uncheck to resume normal blending. Use this mode when you need to tune the
underlying `rtx.atmosphere.*` globals or `rtx.volumetrics.*` knobs while a
specific weather preset is active.

---

## 4. The 42 Parameters and What They Control

Each preset defines 42 values organized into four buckets (the volumetric-fog bucket holds 23: transmittance/scattering, sun/brightness gains, the heterogeneous-fog noise field, fog reach, and fog remap). For detailed
documentation of the underlying RTX_OPTIONs see:
- `src/dxvk/rtx_render/rtx_options.h` -- `rtx.atmosphere.*` declarations
- `src/dxvk/rtx_render/rtx_global_volumetrics.h` -- `rtx.volumetrics.*`
  declarations

### Cloud bucket (19 parameters)

These drive the volumetric cloud ray-marcher and Nubis-style spatial variation.

| Field | What it controls |
|-------|-----------------|
| `cloudDensity` | Overall cloud optical thickness multiplier. Higher = thicker, darker clouds. |
| `cloudCoverageMean` | Mean of the spatial coverage distribution (0 = no clouds, 1 = full cover). |
| `cloudCoverageSpread` | Variance around the mean; wider spread = patchier coverage. |
| `cloudCoverageNoiseScale` | World-space frequency of the coverage noise field. |
| `cloudTypeMean` | Mean cloud type bias (0 = stratiform/flat, 1 = cumuliform/tall). |
| `cloudTypeSpread` | Variance around the type mean. |
| `cloudTypeNoiseScale` | World-space frequency of the type noise field. |
| `cloudAnvilBias` | Pushes type toward anvil/cumulonimbus shapes; useful for thunderstorm. |
| `cloudColor` | RGB tint of the cloud scattering albedo. |
| `cloudWindSpeed` | Animation speed of the cloud layer. |
| `cloudWindDirection` | Wind direction in degrees. |
| `cloudShadowStrength` | How strongly clouds occlude the sun (ground shadow intensity). |
| `cloudAnisotropy` | Henyey-Greenstein g factor for cloud phase function. |
| `cloudThickness` | Vertical extent of the cloud slab in km. |
| `cloudShadowTint` | RGB color tint applied to the cloud shadow. |
| `cloudShadowTintStrength` | Blend weight between neutral and tinted shadow. |
| `cloudSunsetWarmth` | Warm-tint strength applied at low sun angles. |

**Load-bearing parameters per weather category:**
- Clear / partly cloudy: `cloudCoverageMean`, `cloudDensity`, `cloudThickness`
- Storm: `cloudCoverageMean`, `cloudDensity`, `cloudShadowStrength`, `cloudAnvilBias`
- Color character: `cloudColor`, `cloudShadowTint`, `cloudSunsetWarmth`

### Atmosphere bucket (3 parameters)

| Field | What it controls |
|-------|-----------------|
| `airDensity` | Rayleigh scattering multiplier. Higher = bluer sky, more aerial perspective. |
| `aerosolDensity` | Mie scattering multiplier. Higher = hazier, sun corona wider. |
| `sunIlluminance` | RGB luminance of the solar disk. Dims the sun for storm presets. |

**Load-bearing:** `aerosolDensity` is the primary visibility knob; `sunIlluminance`
is the primary brightness knob. Together they define how dark and how hazy the
scene feels.

### Sky / moon mood bucket (3 parameters)

| Field | What it controls |
|-------|-----------------|
| `nightSkyBrightness` | Global brightness of the night sky (stars + airglow). |
| `moonNeeStrength` | World-side moon lighting multiplier. |
| `moonAtmosphericCouplingStrength` | Sky-side moon atmosphere coupling multiplier. |

These are primarily for night-time weather differentiation (foggy nights vs
clear nights vs blizzard nights).

### Volumetric bucket (4 parameters)

These control the global participating media that fills the entire scene volume.
They are the primary tool for fog, haze, and rain-diffusion effects.

| Field | What it controls |
|-------|-----------------|
| `transmittanceColor` | RGB color of the volumetric medium (controls color cast of fog). |
| `transmittanceMeasurementDistanceMeters` | Distance at which 63% of light is absorbed. **Lower = denser fog.** |
| `singleScatteringAlbedo` | RGB single-scattering albedo (how much light scatters vs absorbs). |
| `volumetricAnisotropy` | Henyey-Greenstein g for the volumetric phase function (positive = forward). |

**Load-bearing:**
- `transmittanceMeasurementDistanceMeters` is the single most important fog
  knob. 1000 = clear air; 80 = thick fog; 30 = pea-soup; 50 = whiteout snow.
- `transmittanceColor` tints the fog: neutral grey for rain, blue-grey for snow,
  yellow-brown for sandstorm/smog.
- `volumetricAnisotropy` 0.6+ creates strong forward-scattering halos around
  light sources, characteristic of sandstorm and dusty haze.

---

## 5. The 12 Shipped Archetypes

### `clear`
Sunny, crisp, minimal haze. Low cloud density and coverage. Neutral-white cloud
color. Clear atmosphere (aerosolDensity 0.7). Very long transmittance distance
(1000 m) -- essentially no visible fog. Intended as the reference "good day"
baseline.

### `partlyCloudy`
Light scattered clouds with some sun. Moderate cloud coverage (mean 0.30) with
high variance -- broken, patchy appearance. Slight volumetric haze (800 m
transmittance). Slightly dimmed sun (19 kLux vs 20 for clear).

### `overcast`
Full cloud cover, the current renderer default look. Dense coverage (mean 0.64),
high cloud density (1.8). Moderate aerosol haze. Medium transmittance (500 m).
Neutral grey-blue cloud color. Intended as the reference "cloudy but not raining"
state.

### `hazy`
Warm summer haze. Moderate cloud coverage with warm-tinted cloud color
(0.92, 0.91, 0.88). High aerosol density (1.5) with elevated sun warmth (1.10).
Warm transmittance tint (0.985, 0.97, 0.94) at 250 m. The sun still reads as
warm yellow-white through the haze. Good for dusty plains or humid summer days.

### `foggy`
The headline fog preset. Low cloud density, flat stratus cloud type. Aggressive
volumetric fog: aerosolDensity 2.0, transmittance 80 m, cool-grey transmittance
tint (0.92, 0.94, 0.96). Dim diffuse sun (10 kLux). Neutral anisotropy (no
forward scatter). Intended for valley fog, morning mist, coastal fog.

### `drizzle`
Light rain, medium fog. Heavy cloud coverage (mean 0.60) with a cool, dark grey
cloud color (0.78, 0.82, 0.88). Moderate fog (200 m transmittance). Slightly
bluish sun (11, 12, 14 kLux -- blue-heavy to suggest grey sky). Light cloud
shadow (0.20). Intended for persistent grey drizzle days.

### `rainstorm`
Heavy clouds, dim sun, dense fog. Very high cloud coverage (mean 0.80) and
density (2.5). Very dark cloud color (0.65, 0.68, 0.75). Heavy cloud shadow
(0.40). Dim, cool sun (7, 8, 10 kLux). Dense volumetric fog (100 m, blue-grey
transmittance 0.85, 0.88, 0.92). Intended for active downpours.

### `thunderstorm`
The heaviest preset. Near-total cloud coverage (mean 0.95) with maximum density
(3.5). High anvil bias (0.7) for cumulonimbus shapes. Bruised-purple cloud
color (0.50, 0.52, 0.58). Very dim, blue-shifted sun (4, 4, 6 kLux). Thick fog
(60 m transmittance, 0.75, 0.78, 0.82 transmittance color). Low sunset warmth
(0.10) -- no warm tones even at golden hour.

### `snow`
Medium cloud cover, cool bluish-white palette. High cloud coverage (mean 0.65)
with bright white-blue clouds (0.95, 0.97, 1.00). Moderate fog (250 m) with a
near-neutral cool tint (0.97, 0.98, 0.99). Blue-shifted sun (12, 13, 14 kLux).
Low sunset warmth (0.30). Slightly elevated night-sky brightness (0.012) for
the snow-glow effect on overcast nights.

### `blizzard`
Near-whiteout. Dense coverage (mean 0.95) and maximum-range cloud density (3.0).
Severe fog (50 m transmittance). Blue-white transmittance tint (0.92, 0.95, 0.98).
Heavy cloud shadow (0.50). Very dim sun (6, 7, 8 kLux). Intended for
disorienting severe conditions.

### `sandstorm`
Yellow-orange forward-scattering haze. Moderate cloud coverage (mean 0.40) with
sand-colored clouds (0.85, 0.65, 0.40). Very high aerosol density (2.5). Very
warm sun tint (10, 8, 5 kLux -- strong red-shift). High volumetric anisotropy
(0.60 -- strong forward scattering halos). Short transmittance (50 m) with an
intense rust-orange tint (0.95, 0.65, 0.35). Single-scattering albedo
(0.90, 0.75, 0.50) biased toward yellow-orange. The most strongly chromatic
preset.

### `smoggy`
Industrial dark grey-brown haze. Moderate cloud coverage (mean 0.45) with
smog-grey-brown cloud color (0.65, 0.58, 0.45). High aerosol density (1.8).
Cool-desaturated-brown transmittance (0.70, 0.65, 0.55) at 200 m. Warm-shifted
but dim sun (12, 10, 8 kLux). Moderate forward scatter (0.20). Low
scattering albedo (0.85, 0.80, 0.70) for the absorptive, dirty-air look.

---

## 6. Common Customization Recipes

### Make the foggy preset actually thick

The shipped `foggy` default (80 m transmittance) is medium fog. For pea-soup:

```ini
rtx.weather.preset.foggy.foggy_transmittanceMeasurementDistanceMeters = 25.0
rtx.weather.preset.foggy.foggy_aerosolDensity = 4.0
rtx.weather.preset.foggy.foggy_sunIlluminance = 6.0, 6.0, 6.5
```

### Custom red-orange sandstorm for a Mars-like setting

```ini
rtx.weather.preset.sandstorm.sandstorm_transmittanceColor = 0.80, 0.30, 0.15
rtx.weather.preset.sandstorm.sandstorm_cloudColor = 0.70, 0.35, 0.20
rtx.weather.preset.sandstorm.sandstorm_sunIlluminance = 4.0, 2.5, 1.5
rtx.weather.preset.sandstorm.sandstorm_singleScatteringAlbedo = 0.75, 0.40, 0.25
rtx.weather.preset.sandstorm.sandstorm_volumetricAnisotropy = 0.70
```

### Make the thunderstorm preset even darker

```ini
rtx.weather.preset.thunderstorm.thunderstorm_sunIlluminance = 2.0, 2.0, 3.5
rtx.weather.preset.thunderstorm.thunderstorm_transmittanceMeasurementDistanceMeters = 40.0
rtx.weather.preset.thunderstorm.thunderstorm_cloudDensity = 4.5
rtx.weather.preset.thunderstorm.thunderstorm_cloudShadowStrength = 0.80
```

### Snowy overcast with no fog (tundra-style flat light)

```ini
rtx.weather.preset.overcast.overcast_transmittanceMeasurementDistanceMeters = 2000.0
rtx.weather.preset.overcast.overcast_cloudColor = 0.97, 0.98, 1.00
rtx.weather.preset.overcast.overcast_sunIlluminance = 14.0, 15.0, 17.0
rtx.weather.preset.overcast.overcast_aerosolDensity = 0.8
```

---

## 7. Notes on the Pause Toggle

The **Pause Weather Blender** checkbox in the dev menu (under
**Atmosphere -> Weather Presets**) stops the blender from writing RTX_OPTION
overrides. This is specifically for the workflow of tuning the underlying
`rtx.atmosphere.*` and `rtx.volumetrics.*` knobs directly while the game is
in a specific weather state.

**When to use it:**
- You want to dial in atmosphere parameter values that will become your new
  preset defaults.
- You are investigating how the raw underlying knobs interact without the
  blender's Derived-layer writes masking your manual changes.

**Effect:**
- While paused, dragging sliders in the atmosphere panel writes to the
  Persistent layer as normal. The blender is not fighting your changes.
- While paused, no blending progress occurs. Time does not advance for the
  active transition.
- When unpaused, the blender resumes from wherever the blend progress was
  when it was paused.

The pause state is not written to GameStateStore and is not visible to plugins.
It is a dev-only tuning aid.
