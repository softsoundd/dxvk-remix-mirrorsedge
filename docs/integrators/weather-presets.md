# Weather Presets -- Plugin Integration Guide

This document is for plugin authors writing game-side weather and particle logic
that coordinates with the dxvk-remix weather preset system.

---

## 1. Overview

The weather preset system is a renderer-side lerp pipeline that blends between
12 named atmospheric archetypes (clear, partlyCloudy, overcast, hazy, foggy,
drizzle, rainstorm, thunderstorm, snow, blizzard, sandstorm, smoggy). Each
archetype defines 42 RTX_OPTION values covering clouds (13), atmosphere (3),
sky/moon mood (3), and volumetric fog (23 — transmittance/scattering, fog
gains, the full heterogeneous-fog noise field, fog reach, and fog remap).

The entire per-field set is generated from a single source of truth — the
`WEATHER_PRESET_FIELD_LIST` X-macro in
[`rtx_fork_weather.h`](../../src/dxvk/rtx_render/rtx_fork_weather.h) — which
drives the per-preset `RTX_OPTION` declarations, the blend math, and the
generated dev-menu panel together. Each field carries a *kind* (scalar, angle,
extinction, color, vec3, or step/bool) that selects both its interpolation
(e.g. fog distance lerps in extinction space; bools switch at the blend
midpoint) and its widget.

**Renderer's responsibility:**
- Owns the blending math -- reads the trigger keys once per frame, lerps
  from the previous preset snapshot toward the target preset's RTX_OPTION
  values over the requested duration, and applies the blended values to the
  Derived RTX_OPTION layer each frame.
- Publishes current blend state back into GameStateStore so plugins can
  observe progress.

**Plugin's responsibility:**
- Chooses when to trigger a weather transition and what duration to use.
- Writes the two trigger keys to GameStateStore via `remixapi_SetGameValue`.
- Owns all particle effects (rain, snow, dust, etc.). The renderer has no
  built-in particle authoring; the plugin must spawn, blend, and despawn
  particles in sync with the weather transition it initiated.

The system is dormant by default. If the plugin never writes `__weather.target`,
the blender does not run and all existing `rtx.atmosphere.*` and
`rtx.volumetrics.*` RTX_OPTION values apply unchanged.

---

## 2. Trigger Contract

The plugin writes two keys to the GameStateStore to begin a transition:

### `__weather.target`

| Property | Value |
|----------|-------|
| Type | string |
| Writer | plugin |
| Effect | begins or retargets a blend to the named preset |

Value must be one of the 12 preset names (case-sensitive):
`clear`, `partlyCloudy`, `overcast`, `hazy`, `foggy`, `drizzle`, `rainstorm`,
`thunderstorm`, `snow`, `blizzard`, `sandstorm`, `smoggy`.

An **empty string** (`""`) returns the system to dormant mode. The blender
stops writing overrides and the renderer uses whatever RTX_OPTION values are
currently set by other means (user.conf, direct SetConfigVariable calls, etc.).

If the blend is already in progress when a new target is written, the blender
retargets from the current partially-blended state -- no pop.

### `__weather.blend_seconds`

| Property | Value |
|----------|-------|
| Type | string (numeric, parsed as float) |
| Writer | plugin |
| Effect | sets the crossfade duration for the next or current transition |

Write this key before or at the same time as `__weather.target`. The blender
reads both keys on the same frame. If omitted, the previous duration carries
over (or the blender defaults to 1.0 second on first activation).

**Example values:** `"5.0"` (5-second crossfade), `"30.0"` (slow cinematic
transition), `"0.1"` (near-instant snap).

### Lifecycle

1. Plugin writes `__weather.blend_seconds = "10.0"`.
2. Plugin writes `__weather.target = "thunderstorm"`.
3. The blender picks up both keys on the next frame and begins the 10-second
   lerp from the current atmospheric state toward the thunderstorm preset.
4. After 10 seconds, the blend completes. The blender continues holding the
   thunderstorm preset values until the target is changed.
5. To return to unmanaged state, the plugin writes `__weather.target = ""`.

---

## 3. State Broadcast

The renderer publishes three read-only keys into GameStateStore each frame
while the blender is active:

### `__weather.current`

The name of the preset that is the current blend *target* (the destination the
renderer is blending toward). Empty string when the blender is dormant.

### `__weather.previous`

The name of the preset the blend started *from* (or the partially-blended
snapshot name if a retarget occurred mid-blend). Empty string before the first
activation.

### `__weather.blend_progress`

A float in `[0.0, 1.0]` encoded as a string (e.g. `"0.75"`), representing how
far the current blend has advanced. `"1.0"` means the transition is complete
and the target preset is fully applied. Useful for driving particle density
ramps in sync with the visual blend.

### Read pattern

Because GameStateStore currently does not expose a C API read path
(`remixapi_GetGameValue` is a near-term follow-on), the recommended pattern for
plugins that issue the trigger calls themselves is to track their own blend
state locally:

```cpp
// Plugin-side tracking -- no remixapi_GetGameValue needed yet.
struct WeatherState {
    std::string currentTarget;
    float       blendDurationSec = 0.0f;
    float       blendElapsedSec  = 0.0f;

    float progress() const {
        if (blendDurationSec <= 0.0f) return 1.0f;
        return std::min(blendElapsedSec / blendDurationSec, 1.0f);
    }
};
```

Once `remixapi_GetGameValue` ships, plugins can read `__weather.blend_progress`
directly from the renderer instead of tracking elapsed time locally.

---

## 4. Example Plugin Code

The following snippet shows a minimal `setWeather` function using the
`remixapi_SetGameValue` C API (declared in `public/include/remix/remix_c.h`,
line 730).

```cpp
#include "remix/remix_c.h"
#include <string>

// Cached interface pointer -- obtain during plugin init via remixapi_InitializeLibrary.
static remixapi_Interface* s_remix = nullptr;

// Trigger a weather transition.
// presetName  -- one of: clear, partlyCloudy, overcast, hazy, foggy,
//                        drizzle, rainstorm, thunderstorm, snow, blizzard,
//                        sandstorm, smoggy. Empty string to return to dormant.
// durationSec -- crossfade duration in seconds.
remixapi_ErrorCode setWeather(const std::string& presetName, float durationSec) {
    if (!s_remix || !s_remix->SetGameValue) {
        return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }

    // Write duration first so the blender reads both on the same frame.
    std::string durationStr = std::to_string(durationSec);
    remixapi_ErrorCode err = s_remix->SetGameValue("__weather.blend_seconds",
                                                   durationStr.c_str());
    if (err != REMIXAPI_ERROR_CODE_SUCCESS) return err;

    return s_remix->SetGameValue("__weather.target", presetName.c_str());
}

// Example usage -- trigger a 15-second transition to thunderstorm.
void onEnterDangerZone() {
    setWeather("thunderstorm", 15.0f);
}

// Return to unmanaged state when leaving the danger zone.
void onExitDangerZone() {
    setWeather("", 5.0f);  // duration ignored when clearing target
}
```

---

## 5. Particle Coordination

The renderer handles atmospheric rendering (clouds, fog, volumetric scattering,
sky color). Particles -- rain drops, snowflakes, dust motes, smoke -- are
**entirely the plugin's responsibility**. The renderer does not spawn or
manage any particles as part of the preset transition.

### Recommended pattern

1. Maintain a particle archetype map keyed by preset name in your plugin:

```cpp
struct ParticleArchetype {
    float rainIntensity  = 0.0f;
    float snowIntensity  = 0.0f;
    float dustIntensity  = 0.0f;
    // ... other particle channels
};

static const std::unordered_map<std::string, ParticleArchetype> kParticleMap = {
    { "clear",       { 0.0f, 0.0f, 0.0f } },
    { "drizzle",     { 0.4f, 0.0f, 0.0f } },
    { "rainstorm",   { 0.9f, 0.0f, 0.0f } },
    { "thunderstorm",{ 1.0f, 0.0f, 0.0f } },
    { "snow",        { 0.0f, 0.6f, 0.0f } },
    { "blizzard",    { 0.0f, 1.0f, 0.1f } },
    { "sandstorm",   { 0.0f, 0.0f, 1.0f } },
    // ... other presets
};
```

2. In your per-frame update, lerp particle intensities using the same duration
   the plugin issued to the blender:

```cpp
void updateParticles(float deltaTime) {
    float t = weatherState.progress();
    auto& prev = kParticleMap.at(weatherState.previousPreset);
    auto& curr = kParticleMap.at(weatherState.currentPreset);

    float rain  = std::lerp(prev.rainIntensity,  curr.rainIntensity,  t);
    float snow  = std::lerp(prev.snowIntensity,  curr.snowIntensity,  t);
    float dust  = std::lerp(prev.dustIntensity,  curr.dustIntensity,  t);

    setParticleRate("rain", rain);
    setParticleRate("snow", snow);
    setParticleRate("dust", dust);
}
```

3. Sync the particle lerp duration to the same value passed to `setWeather`.
   This ensures particle density and atmospheric appearance reach their target
   states simultaneously.

### Why particles are not just spawn rate

Particle systems involve far more than a single density scalar:
- Particle velocity, spread, and lifetime all shift between weather states
  (light drizzle has slow, wide drops; blizzard has fast horizontal streaks).
- Sound layers (rain on surfaces, thunder rumble, wind howl) blend on separate
  curves.
- Splash / accumulation VFX (puddles, snow buildup) have their own accumulation
  timelines that do not simply follow the atmospheric blend curve.
- GPU particle budget varies widely across archetypes -- the plugin must manage
  pool sizing and LOD independently.

Because of this complexity, the design leaves all particle authoring to the
plugin. The renderer provides the blend progress signal; the plugin decides what
to do with it.

---

## 6. ImGui-Driven Changes

The renderer's dev menu (under **Atmosphere -> Weather Presets**) writes the same
`__weather.target` and `__weather.blend_seconds` GameStateStore keys that a
plugin writes. This means:

- A plugin watching `__weather.target` will see ImGui-driven weather changes
  exactly as it would see plugin-driven ones.
- Particle coordination still applies -- if a designer triggers a preset change
  in the dev menu during a play session, the plugin's per-frame particle lerp
  will respond as though the plugin itself had issued the trigger.
- The "Blend Duration" slider maps directly to `__weather.blend_seconds`.

This design allows artists and level designers to test weather transitions live
in the dev menu while the full plugin stack (particles, audio, gameplay
reactions) responds naturally.

### Self-contained preset editor

The panel is a single self-contained editor (no separate "Tune Preset Defaults"
sub-tree). It is generated from the field table, so every per-preset value --
including the full volumetric-fog surface -- is editable in one place, grouped
into collapsible sections (Clouds / Atmosphere / Sky & Moon / Volumetric Fog)
with a name filter box. Editing writes the `rtx.weather.preset.<name>.<field>`
options directly. Authoring aids:

- **Editing Preset / Edit Active** -- choose which preset to edit, or jump to
  whatever the blender is currently targeting.
- **Pin Edited Preset** -- snaps the blender to the edited preset with a 0 s
  blend and freezes variation, so edits show immediately on a held image.
- **Copy Into Edited** -- copies every value from another preset into the one
  being edited (fork a new archetype from an existing one).
- **Snapshot Live -> Preset** -- captures the current live renderer values into
  the edited preset. Tune the real atmosphere/volumetrics with the blender
  dormant, then capture.
- **Export to Clipboard** -- emits the preset as `rtx.weather.preset.*` lines
  ready to paste into `user.conf` (the way to persist tuned values).

**Caveat:** Until `remixapi_GetGameValue` ships, there is no C API for plugins
to *read* the current `__weather.target` value written by the dev menu. Plugins
that need to observe external triggers should poll the renderer-published
`__weather.current` key once that API is available.

---

## 7. Plugin-Side Opt-Out

The weather preset system is fully opt-in. If your plugin never writes
`__weather.target`, the blender stays dormant and has zero effect on the
renderer.

In dormant mode, direct `remixapi_SetConfigVariable` calls (or equivalent
RTX_OPTION writes) to individual atmosphere parameters still apply as normal:

```cpp
// These work regardless of whether the weather blender is active.
s_remix->SetConfigVariable("rtx.atmosphere.cloudDensity", "2.0");
s_remix->SetConfigVariable("rtx.atmosphere.airDensity",   "1.3");
```

Note that if the blender *is* active, it writes blended values to the Derived
RTX_OPTION layer each frame, which will overwrite any values set by direct
`SetConfigVariable` calls on the same frame. To restore manual control,
write `__weather.target = ""` to put the blender back to dormant.

---

## 8. Cloud Drift

Once a weather transition completes, the renderer holds the target preset's
parameter values steady -- but real atmospheres are never frozen. The cloud
drift system continuously modulates a few weather parameters with
low-frequency noise so the sky's overall character keeps changing between
transitions.

> **Changed 2026-06-21 (de-pulse + trim).** This system used to modulate 9
> parameters with a two-layer noise whose fast layer had a 30-second base
> period -- that produced a clearly visible whole-sky "breathing" pulse
> every ~30 s. The fast layer is removed (slow multi-minute layer only) and
> the set is cut to the **3 genuinely weather-scale parameters**:
> `cloudCoverageMean`, `cloudWindSpeed`, `cloudWindDirection`. Per-preset
> *shape* change (density, thickness, type, anvil billowing) is now produced
> locally by the renderer's **field-evolution** system
> (`rtx.atmosphere.cloudEvolutionSpeed` / `cloudBoilSpeed`), not by this
> global-scalar drift. The two trigger keys below are unchanged.

The renderer ships the drift mechanism (which parameters drift, with what
relative amplitude -- fixed per the design); the plugin owns the drift
*personality* per preset by writing two GameStateStore keys.

### Drift trigger keys

| Key | Type | Default | Effect |
|---|---|---|---|
| `__weather.drift_speed`     | string (numeric) | 1.0 | Scales drift phase advance rate. Higher = faster evolution. |
| `__weather.drift_intensity` | string (numeric) | 1.0 | Scales drift swing amplitude. 0.0 = drift fully off. |

Both values are smoothed with a 1-second time constant, so plugin-side
updates ease in alongside the weather transition that triggered them -- no
visible step.

### Recommended values per preset

These values are starter recommendations. Tune per game using the dev menu's
"Weather Variation" sub-tree (formerly "Cloud Drift"), then bake the values
into your `setWeather` handler.

| Preset | drift_speed | drift_intensity | Character |
|---|---|---|---|
| `clear`        | 0.6 | 0.5 | Calm, barely-perceptible drift on a mostly-clear sky |
| `partlyCloudy` | 1.0 | 1.0 | Baseline; visible scattered-cloud evolution |
| `overcast`     | 0.7 | 0.7 | Slow, calm overcast quilt -- shifts over minutes |
| `hazy`         | 0.8 | 0.6 | Subtle haze movement |
| `foggy`        | 0.5 | 0.4 | Almost still -- fog drifts slowly |
| `drizzle`      | 1.2 | 1.1 | Active light-rain cloud cells |
| `rainstorm`    | 1.6 | 1.4 | Visible heavy-rain turbulence |
| `thunderstorm` | 2.0 | 1.6 | Fast and dramatic -- cells building and collapsing |
| `snow`         | 0.9 | 0.8 | Steady snowfall sky |
| `blizzard`     | 1.8 | 1.5 | Whiteout turbulence |
| `sandstorm`    | 1.5 | 1.6 | Gusty, large-amplitude swings |
| `smoggy`       | 0.8 | 0.7 | Slow industrial-haze drift |

### Updated `setWeather` example

Extends the example from Section 4 with drift:

```cpp
#include "remix/remix_c.h"
#include <string>
#include <unordered_map>

static remixapi_Interface* s_remix = nullptr;

// Per-preset drift recommendations (matches the table above).
static const std::unordered_map<std::string, std::pair<float, float>> kDriftRecommendations = {
    { "clear",        { 0.6f, 0.5f } },
    { "partlyCloudy", { 1.0f, 1.0f } },
    { "overcast",     { 0.7f, 0.7f } },
    { "hazy",         { 0.8f, 0.6f } },
    { "foggy",        { 0.5f, 0.4f } },
    { "drizzle",      { 1.2f, 1.1f } },
    { "rainstorm",    { 1.6f, 1.4f } },
    { "thunderstorm", { 2.0f, 1.6f } },
    { "snow",         { 0.9f, 0.8f } },
    { "blizzard",     { 1.8f, 1.5f } },
    { "sandstorm",    { 1.5f, 1.6f } },
    { "smoggy",       { 0.8f, 0.7f } },
};

remixapi_ErrorCode setWeather(const std::string& presetName, float durationSec) {
    if (!s_remix || !s_remix->SetGameValue) {
        return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }

    // Existing target + duration writes.
    std::string durationStr = std::to_string(durationSec);
    s_remix->SetGameValue("__weather.blend_seconds", durationStr.c_str());

    // Drift personality. Look up recommended values; fall back to (1.0, 1.0)
    // if the preset name is unknown (the renderer will reject the unknown
    // name anyway, but we still want the drift writes to be coherent).
    auto it = kDriftRecommendations.find(presetName);
    float driftSpeed     = (it != kDriftRecommendations.end()) ? it->second.first  : 1.0f;
    float driftIntensity = (it != kDriftRecommendations.end()) ? it->second.second : 1.0f;
    s_remix->SetGameValue("__weather.drift_speed",     std::to_string(driftSpeed).c_str());
    s_remix->SetGameValue("__weather.drift_intensity", std::to_string(driftIntensity).c_str());

    return s_remix->SetGameValue("__weather.target", presetName.c_str());
}
```

### Plugin-side opt-out

If your plugin never writes the two drift keys, the renderer uses defaults
of 1.0 / 1.0 across all presets. This produces visible-but-not-overwhelming
drift on every preset -- degraded (no per-preset personality differentiation)
but not broken. To disable drift entirely from the plugin side, write `0.0`
to `__weather.drift_intensity` and the renderer's drift modulation pass
short-circuits.

### What drifts

The renderer modulates 3 weather-scale parameters: `cloudCoverageMean`,
`cloudWindSpeed`, and `cloudWindDirection` (how cloudy the sky is overall and
how the wind gusts/shifts). The former shape parameters -- coverage spread,
cloud-type mean/spread, density, thickness, anvil bias -- are no longer drifted
here; their change is now produced locally by the field-evolution system (see
the 2026-06-21 note above). Color, optical, sky/moon, atmosphere, and
volumetric fog parameters remain intentionally NOT drifted (drift on color
looks sickly; drift on physically-calibrated parameters breaks lighting; drift
on noise-scale parameters re-tiles the entire cloud field).
