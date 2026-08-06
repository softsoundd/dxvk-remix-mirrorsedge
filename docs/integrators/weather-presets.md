# Weather Presets -- Plugin Integration Guide

This document is for plugin authors writing game-side weather and particle logic
that coordinates with the dxvk-remix weather preset system.

---

## 1. Overview

The weather preset system is a renderer-side lerp pipeline that blends between
12 named atmospheric archetypes (clear, partlyCloudy, overcast, hazy, foggy,
drizzle, rainstorm, thunderstorm, snow, blizzard, sandstorm, smoggy). Each
archetype defines 63 RTX_OPTION values covering clouds (17), atmosphere (5),
sky/moon mood (4), volumetric fog (27 — transmittance/scattering, fog gains,
the full heterogeneous-fog noise field, fog reach, and fog remap), and
precipitation (10 — rain / snow / blowing sand particles).

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
- Owns particle effects the runtime does not: sound layers, splash and
  accumulation VFX, and any bespoke channels. Precipitation itself (rain, snow,
  blowing sand) is built in since 2026-07-25 and blends with the preset — see
  section 5.

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

> **Updated 2026-07-25: precipitation is now built in.** The runtime ships a
> weather-driven precipitation particle system (rain, snow, blowing sand). It is
> part of the preset like any other parameter, so a plugin that only wants
> falling weather needs to do *nothing* beyond setting `__weather.target` --
> the drops blend in and out with the rest of the transition. The rest of this
> section still applies to everything the runtime does NOT own: sound layers,
> splash and accumulation VFX, and any bespoke particle channels.

### Built-in precipitation

Ten per-preset fields (`precipitationIntensity`, `precipitationFallSpeed`,
`precipitationWindResponse`, `precipitationTurbulence`, `precipitationDrag`,
`precipitationStreak`, `precipitationDropWidth`, `precipitationDropLength`,
`precipitationOpacity`, `precipitationColor`) blend exactly like the cloud and
fog fields and drive the live `rtx.weather.precipitation.*` options. Presets
ship authored: dry for clear/partlyCloudy/overcast/hazy/smoggy, drifting mist
for foggy, escalating rain for drizzle/rainstorm/thunderstorm, tumbling flakes
for snow/blizzard, and tinted near-horizontal grit for sandstorm. Per-preset
defaults are tabulated in `weather-presets-reference.md`.

The emitter is anchored to the main camera POSITION -- never its orientation --
and slants with the atmosphere's cloud wind, so precipitation follows the player
automatically and matches the direction the sky is moving. Nothing about
spawning depends on where the player is looking; tying the spawn volume to the
view direction makes turning drag freshly spawned particles along with it.

Keep `precipitationWindResponse` small. The slant it produces is fixed in world
space (as it should be), which means it re-projects on screen as the camera
turns -- a slant big enough to notice reads to players as "the rain changes
direction when I look around". A few degrees is plenty for anything that is not
deliberately a blizzard.

Settings that are *not* per-preset live under `rtx.weather.precipitation` and
are exposed in the dev menu under Weather Presets -> "Precipitation (global)" --
these are performance and integration choices rather than weather look:

| Option | Default | What it is for |
|---|---|---|
| `enable` | `True` | Master switch. Set `False` if the game drives its own precipitation. |
| `maxParticles` | `24000` | Particle budget and the main performance dial; buffers are allocated for this count. |
| `spawnRadiusMeters` / `spawnHeightMeters` | `20` / `14` | Size and altitude of the spawn plane above the camera. |
| `fallMarginMeters` | `8` | How far below the camera drops survive before being recycled. |
| `occludeUnderCover` | `True` | Ray-trace one occlusion query per spawned particle against the scene, ending its life where it would hit geometry. View-independent — this is what makes roofs, bridges and interiors work. |
| `enableCollision` | `False` | Additional per-step *screen-space* collision (bounce/kill against visible surfaces only). Largely redundant now that `occludeUnderCover` exists; costs a depth test per particle per frame. |
| `descUpdateIntervalMs` | `750` | How often the particle descriptor may change. Lower reacts to blends faster and costs more transient VRAM. Internally floored at one sixth of the particle lifetime so slow-falling looks (snow lives ~20 s) cannot pile up transient systems during a blend. |
| `roughness` / `metallic` | `0.35` / `0` | Drop material surface response. Edits apply on the descriptor cadence and crossfade in over one particle lifetime (a material change forks the particle-system identity the same way a descriptor change does). |

Only turn `enable` off if your game already has precipitation you want to keep;
leaving both on will double it up.

### Interiors and cover

Precipitation is stopped by real geometry. When a particle spawns, the runtime
ray traces the scene acceleration structure twice: an upward shelter probe
first (if opaque cover hangs anywhere overhead -- even far above the spawn
plane -- the drop is killed at birth, because it would have been stopped up
there), then a trace along the direction the particle is about to travel,
ending its life where it would land. Rain therefore stops under roofs,
bridges, awnings and overhangs without the game telling the runtime anything
about them.

The upward probe matters more than it looks: the spawn plane hangs only
`spawnHeightMeters` above the camera, and any cover ABOVE that plane is behind
a downward ray's origin and invisible to it. This is especially acute when
`rtx.sceneScale` under-states the game's real unit scale (authored meters
shrink relative to the world, lowering the plane beneath most roofs) -- but it
is true at any scale: a bridge 30 m up should shelter the street even though
the plane hangs at 14 m.

This is view-independent, which is the whole point: the older
`collideWithWorld` path reprojects into the previous frame G-buffer and so can
only see what is on screen -- useless for the roof above the player head, which
almost never is. Cost is at most two rays per SPAWNED particle (hundreds per
frame at default settings, not thousands), because a straight-line ray covers a
ballistic particle whole path in a single query.

Controlled by `rtx.weather.precipitation.occludeUnderCover` (default on).
Read-only diagnostics (per-frame ray/kill/hit counters, TLAS availability, and
the spawn-plane height in world units) live directly under that toggle in the
dev menu.

Limits worth knowing:
- The rays are straight lines along the spawn velocity, so they are exact for
  straight-falling rain and an approximation for heavily tumbling snow.
- Only opaque geometry blocks. Glass, alpha-blended AND alpha-tested surfaces
  (foliage cards, chain fences, gratings) do not stop precipitation.
- It uses the previous frame acceleration structure (particle simulation runs
  before this frame TLAS is built) and is skipped entirely on the first frames
  of a scene, before any TLAS exists.

That covers physical cover. It does NOT cover the case where a game considers
somewhere "indoors" that has no geometry above it, or where you simply want no
weather in an interior cell regardless. For that, the wrapper still decides: it
already owns `__weather.target`, so target a dry preset on entering an interior,
or set `rtx.weather.precipitation.intensity = 0` directly. Precipitation ramps
down over one particle lifetime, which reads as walking out of the rain rather
than a hard cut.

### Fast movement

Because only the camera's *position* anchors the spawn volume, sustained fast
horizontal movement (vehicles, flight) thins the leading edge of the volume
slightly: already-falling drops are world-anchored and get left behind, and only
the per-frame spawn smear refills the disc overhead. At on-foot speeds the
effect is negligible; at tens of m/s expect visibly lighter rain in the
direction of travel. This is the deliberate trade against view-coupled spawning
(which made turning drag the whole rain volume around the player) -- do not
"fix" it by biasing the volume along the view direction.

### Recommended pattern (plugin-owned particle channels)

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

### Why the rest is still plugin-owned

Weather is more than falling drops, and the remaining channels do not follow the
atmospheric blend curve:
- Sound layers (rain on surfaces, thunder rumble, wind howl) blend on separate
  curves.
- Splash / accumulation VFX (puddles, snow buildup) have their own accumulation
  timelines that lag the atmospheric blend by design -- the ground stays wet
  long after the rain stops.
- Game-specific effects (leaves in wind, ash, embers, spore drifts) are content,
  not weather, and the plugin owns their budget and LOD.

For those, the renderer provides the blend progress signal and the plugin
decides what to do with it.

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
