# Weather Presets -- Parameter Reference

Lookup table of every preset's shipped default for each of the 42 parameters.

> **Expanded 2026-06-28.** The volumetric-fog bucket grew from 4 to 23 fields
> (heterogeneous-fog noise field, fog remap, sun/brightness gains, fog reach).
> The authoritative per-preset defaults are the WEATHER_PRESET_FIELD_LIST /
> WEATHER_PRESET_VALUES_* macros in `rtx_fork_weather.h` and the regenerated
> `RtxOptions.md`. Tables below that predate the expansion cover the original set.
Values are extracted from `WEATHER_PRESET_VALUES_<name>` in
`src/dxvk/rtx_render/rtx_fork_weather.h`.

Override any cell in `user.conf` using the key:
`rtx.weather.preset.<presetName>.<presetName>_<fieldName> = <value>`

Vector3 values are shown as `(r, g, b)`. Float values are shown without the
trailing `f` suffix.

Auto-generation of this table from RTX_OPTION metadata is a planned future
enhancement. Until then this table is manually maintained; update it whenever
`WEATHER_PRESET_VALUES_*` macros change.

For recommended cloud-drift values per preset (set via the plugin-side
`__weather.drift_speed` and `__weather.drift_intensity` GameStateStore keys),
see the table at the end of this document.

---

## Cloud Bucket (19 parameters)

### cloudDensity

| clear | partlyCloudy | overcast | hazy | foggy | drizzle | rainstorm | thunderstorm | snow | blizzard | sandstorm | smoggy |
|-------|-------------|---------|------|-------|---------|-----------|-------------|------|----------|-----------|--------|
| 0.4 | 0.9 | 1.8 | 1.0 | 0.6 | 1.4 | 2.5 | 3.5 | 1.8 | 3.0 | 1.5 | 1.4 |

### cloudCoverageMean

| clear | partlyCloudy | overcast | hazy | foggy | drizzle | rainstorm | thunderstorm | snow | blizzard | sandstorm | smoggy |
|-------|-------------|---------|------|-------|---------|-----------|-------------|------|----------|-----------|--------|
| 0.10 | 0.30 | 0.64 | 0.40 | 0.30 | 0.60 | 0.80 | 0.95 | 0.65 | 0.95 | 0.40 | 0.45 |

### cloudCoverageSpread

| clear | partlyCloudy | overcast | hazy | foggy | drizzle | rainstorm | thunderstorm | snow | blizzard | sandstorm | smoggy |
|-------|-------------|---------|------|-------|---------|-----------|-------------|------|----------|-----------|--------|
| 0.10 | 0.20 | 0.16 | 0.20 | 0.20 | 0.20 | 0.15 | 0.10 | 0.20 | 0.10 | 0.30 | 0.20 |

### cloudCoverageNoiseScale

All presets: `0.0033`

### cloudTypeMean

| clear | partlyCloudy | overcast | hazy | foggy | drizzle | rainstorm | thunderstorm | snow | blizzard | sandstorm | smoggy |
|-------|-------------|---------|------|-------|---------|-----------|-------------|------|----------|-----------|--------|
| 0.6 | 0.5 | 0.5 | 0.4 | 0.2 | 0.3 | 0.4 | 0.7 | 0.4 | 0.5 | 0.2 | 0.3 |

### cloudTypeSpread

| clear | partlyCloudy | overcast | hazy | foggy | drizzle | rainstorm | thunderstorm | snow | blizzard | sandstorm | smoggy |
|-------|-------------|---------|------|-------|---------|-----------|-------------|------|----------|-----------|--------|
| 0.3 | 0.3 | 0.2 | 0.3 | 0.2 | 0.3 | 0.3 | 0.3 | 0.3 | 0.2 | 0.4 | 0.3 |

### cloudTypeNoiseScale

All presets: `0.0034`

### cloudAnvilBias

| clear | partlyCloudy | overcast | hazy | foggy | drizzle | rainstorm | thunderstorm | snow | blizzard | sandstorm | smoggy |
|-------|-------------|---------|------|-------|---------|-----------|-------------|------|----------|-----------|--------|
| 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.7 | 0.3 | 0.3 | 0.3 | 0.3 |

### cloudColor (Vector3)

| Preset | Value |
|--------|-------|
| clear | (0.95, 0.97, 1.00) |
| partlyCloudy | (0.92, 0.95, 1.00) |
| overcast | (0.89, 0.92, 1.00) |
| hazy | (0.92, 0.91, 0.88) |
| foggy | (0.85, 0.88, 0.92) |
| drizzle | (0.78, 0.82, 0.88) |
| rainstorm | (0.65, 0.68, 0.75) |
| thunderstorm | (0.50, 0.52, 0.58) |
| snow | (0.95, 0.97, 1.00) |
| blizzard | (0.92, 0.96, 1.00) |
| sandstorm | (0.85, 0.65, 0.40) |
| smoggy | (0.65, 0.58, 0.45) |

### cloudWindSpeed

All presets: `0.02`

### cloudWindDirection

All presets: `45.0`

### cloudShadowStrength

| clear | partlyCloudy | overcast | hazy | foggy | drizzle | rainstorm | thunderstorm | snow | blizzard | sandstorm | smoggy |
|-------|-------------|---------|------|-------|---------|-----------|-------------|------|----------|-----------|--------|
| 0.0 | 0.05 | 0.10 | 0.10 | 0.05 | 0.20 | 0.40 | 0.60 | 0.20 | 0.50 | 0.20 | 0.15 |

### cloudAnisotropy

All presets: `0.6`

### cloudThickness

| clear | partlyCloudy | overcast | hazy | foggy | drizzle | rainstorm | thunderstorm | snow | blizzard | sandstorm | smoggy |
|-------|-------------|---------|------|-------|---------|-----------|-------------|------|----------|-----------|--------|
| 2.0 | 2.5 | 3.05 | 2.5 | 2.0 | 3.0 | 4.0 | 5.0 | 3.0 | 4.5 | 2.5 | 2.5 |

### cloudShadowTint (Vector3)

All presets: `(0.55, 0.65, 0.85)`

### cloudShadowTintStrength

All presets: `1.0`

### cloudSunsetWarmth

| clear | partlyCloudy | overcast | hazy | foggy | drizzle | rainstorm | thunderstorm | snow | blizzard | sandstorm | smoggy |
|-------|-------------|---------|------|-------|---------|-----------|-------------|------|----------|-----------|--------|
| 0.95 | 0.95 | 0.95 | 1.10 | 0.50 | 0.50 | 0.20 | 0.10 | 0.30 | 0.10 | 1.30 | 0.80 |

---

## Atmosphere Bucket (3 parameters)

### airDensity

| clear | partlyCloudy | overcast | hazy | foggy | drizzle | rainstorm | thunderstorm | snow | blizzard | sandstorm | smoggy |
|-------|-------------|---------|------|-------|---------|-----------|-------------|------|----------|-----------|--------|
| 0.95 | 1.0 | 1.0 | 1.1 | 1.2 | 1.1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.1 |

### aerosolDensity

| clear | partlyCloudy | overcast | hazy | foggy | drizzle | rainstorm | thunderstorm | snow | blizzard | sandstorm | smoggy |
|-------|-------------|---------|------|-------|---------|-----------|-------------|------|----------|-----------|--------|
| 0.7 | 1.0 | 1.1 | 1.5 | 2.0 | 1.5 | 1.4 | 1.3 | 1.3 | 1.6 | 2.5 | 1.8 |

### sunIlluminance (Vector3)

| Preset | Value |
|--------|-------|
| clear | (20.0, 20.0, 20.0) |
| partlyCloudy | (19.0, 19.0, 19.0) |
| overcast | (15.0, 15.0, 15.0) |
| hazy | (17.0, 16.0, 14.0) |
| foggy | (10.0, 10.0, 10.0) |
| drizzle | (11.0, 12.0, 14.0) |
| rainstorm | (7.0, 8.0, 10.0) |
| thunderstorm | (4.0, 4.0, 6.0) |
| snow | (12.0, 13.0, 14.0) |
| blizzard | (6.0, 7.0, 8.0) |
| sandstorm | (10.0, 8.0, 5.0) |
| smoggy | (12.0, 10.0, 8.0) |

---

## Sky / Moon Mood Bucket (3 parameters)

### nightSkyBrightness

| clear | partlyCloudy | overcast | hazy | foggy | drizzle | rainstorm | thunderstorm | snow | blizzard | sandstorm | smoggy |
|-------|-------------|---------|------|-------|---------|-----------|-------------|------|----------|-----------|--------|
| 0.008 | 0.008 | 0.008 | 0.010 | 0.012 | 0.010 | 0.008 | 0.008 | 0.012 | 0.008 | 0.010 | 0.010 |

### moonNeeStrength

All presets: `1.0`

### moonAtmosphericCouplingStrength

All presets: `1.0`

---

## Volumetric Bucket (4 parameters)

### transmittanceColor (Vector3)

| Preset | Value |
|--------|-------|
| clear | (0.999, 0.999, 0.999) |
| partlyCloudy | (0.998, 0.998, 0.998) |
| overcast | (0.995, 0.995, 0.995) |
| hazy | (0.985, 0.97, 0.94) |
| foggy | (0.92, 0.94, 0.96) |
| drizzle | (0.95, 0.96, 0.97) |
| rainstorm | (0.85, 0.88, 0.92) |
| thunderstorm | (0.75, 0.78, 0.82) |
| snow | (0.97, 0.98, 0.99) |
| blizzard | (0.92, 0.95, 0.98) |
| sandstorm | (0.95, 0.65, 0.35) |
| smoggy | (0.70, 0.65, 0.55) |

### transmittanceMeasurementDistanceMeters

| clear | partlyCloudy | overcast | hazy | foggy | drizzle | rainstorm | thunderstorm | snow | blizzard | sandstorm | smoggy |
|-------|-------------|---------|------|-------|---------|-----------|-------------|------|----------|-----------|--------|
| 1000.0 | 800.0 | 500.0 | 250.0 | 80.0 | 200.0 | 100.0 | 60.0 | 250.0 | 50.0 | 50.0 | 200.0 |

### singleScatteringAlbedo (Vector3)

| Preset | Value |
|--------|-------|
| clear | (0.999, 0.999, 0.999) |
| partlyCloudy | (0.999, 0.999, 0.999) |
| overcast | (0.999, 0.999, 0.999) |
| hazy | (0.99, 0.98, 0.96) |
| foggy | (0.99, 0.99, 0.99) |
| drizzle | (0.98, 0.98, 0.99) |
| rainstorm | (0.97, 0.97, 0.98) |
| thunderstorm | (0.95, 0.95, 0.97) |
| snow | (0.99, 0.99, 0.99) |
| blizzard | (0.99, 0.99, 1.00) |
| sandstorm | (0.90, 0.75, 0.50) |
| smoggy | (0.85, 0.80, 0.70) |

### volumetricAnisotropy

| clear | partlyCloudy | overcast | hazy | foggy | drizzle | rainstorm | thunderstorm | snow | blizzard | sandstorm | smoggy |
|-------|-------------|---------|------|-------|---------|-----------|-------------|------|----------|-----------|--------|
| 0.0 | 0.05 | 0.05 | 0.30 | 0.0 | 0.10 | 0.10 | 0.0 | 0.0 | 0.0 | 0.60 | 0.20 |

---

## Recommended Cloud Drift Values

Set these via the `__weather.drift_speed` and `__weather.drift_intensity`
GameStateStore keys when transitioning to each preset. See
[`weather-presets.md`](weather-presets.md) section 8 for the full integration
guide.

| Preset | drift_speed | drift_intensity |
|---|---|---|
| `clear`        | 0.6 | 0.5 |
| `partlyCloudy` | 1.0 | 1.0 |
| `overcast`     | 0.7 | 0.7 |
| `hazy`         | 0.8 | 0.6 |
| `foggy`        | 0.5 | 0.4 |
| `drizzle`      | 1.2 | 1.1 |
| `rainstorm`    | 1.6 | 1.4 |
| `thunderstorm` | 2.0 | 1.6 |
| `snow`         | 0.9 | 0.8 |
| `blizzard`     | 1.8 | 1.5 |
| `sandstorm`    | 1.5 | 1.6 |
| `smoggy`       | 0.8 | 0.7 |

The defaults (when unset) are 1.0 / 1.0. Both values are smoothed with a
1-second time constant inside the renderer.
