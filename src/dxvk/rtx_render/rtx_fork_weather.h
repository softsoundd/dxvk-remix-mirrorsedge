#pragma once

// rtx_fork_weather.h — fork-owned weather preset declarations.
// Defines 624 RTX_OPTIONs (12 presets x 52 fields) under the
// rtx.weather.preset.<presetName> namespace.
//
// Field bucket breakdown: 16 cloud + 5 atmosphere + 4 sky/moon mood + 27 volumetric.
//
// Usage: invoke DECLARE_ALL_WEATHER_PRESETS() inside the RtxOptions struct body
// (see rtx_options.h). The macro expands all 12 preset declarations inline.
//
// The WEATHER_PRESET_FIELD_LIST(X) X-macro is preserved here for use by Task 2
// (WeatherSnapshot struct member generation). It expands to one X(type, name,
// default) call per field, and is the single source of truth for the field
// list. DECLARE_WEATHER_PRESET delegates to it via a per-preset binder, so
// adding a field requires only one edit (in WEATHER_PRESET_FIELD_LIST) and the
// 12 preset declarations regenerate automatically.

#include "rtx_option.h"
#include "../../util/util_vector.h"

// ---------------------------------------------------------------------------
// WeatherFieldKind - per-field blend/widget classifier (see field table below).
// Consumed from the descriptor table onward; an ignorable column for the
// WeatherSnapshot member generator.
// ---------------------------------------------------------------------------
namespace dxvk { namespace fork_weather {
  enum WeatherFieldKind {
    WK_Scalar,      // plain float, linear lerp
    WK_Angle,       // degrees, shortest-path angular lerp
    WK_Extinction,  // optical distance; lerp in 1/distance space
    WK_Color,       // Vector3 tint; componentwise lerp
    WK_Vec3,        // Vector3 radiometric; componentwise lerp
    WK_Step,        // non-interpolated (bool/enum); switch at blend midpoint
    // Display-transform kinds (fork - 2026-07-02, UI usability). Blend math is
    // plain linear on the STORED value; only the widget converts units, so the
    // conf/API/shader-facing value is unchanged. For these kinds the table's
    // min/max/step/fmt columns are in DISPLAY units (they feed only the widget).
    WK_SpeedKmS,    // stored km/s; widget displays m/s (x1000)
    WK_PatchPerKm,  // stored spatial frequency (1/km); widget displays patch size in km (1/x)
  };
} }

// ---------------------------------------------------------------------------
// Field table X-macro - THE single source of truth for the 52 weather fields
// (16 cloud + 5 atmosphere + 4 sky/moon mood + 27 volumetric). Every consumer
// (WeatherSnapshot members, the per-field descriptor table, the generated
// ImGui panel, and the blend/read/write loops) is driven from here, so a field
// added here propagates everywhere with no second site to keep in sync.
//
// Each row expands to:
//   X(type, name, defaultValue, kind, group, section, label, min, max, step, fmt)
//     type/name/defaultValue - C++ type, member name, NEUTRAL default
//     kind                   - WeatherFieldKind (blend math + widget type)
//     group                  - top-level panel tab ("Clouds", "Atmosphere", ...)
//     section                - subsection header within the group
//     label                  - ImGui widget label
//     min/max/step/fmt       - slider range, drag step, printf format
//
// UI metadata (group/section/label/range/fmt) is lifted verbatim from the old
// WEATHER_PRESET_SLIDERS macro so the regenerated panel matches today's ranges.
// Consumers needing only a subset (e.g. the snapshot member generator) still
// take all 11 args and ignore the rest.
// ---------------------------------------------------------------------------
#define WEATHER_PRESET_FIELD_LIST(X) \
  /* Cloud (16) */ \
  X(float,   cloudDensity,                       1.0f,                            WK_Scalar,     "Clouds",         "Look",             "Density",                    0.0f,    10.0f,   0.05f,   "%.2f") \
  X(float,   cloudCoverageMean,                  0.5f,                            WK_Scalar,     "Clouds",         "Coverage & Shape", "Coverage",                   0.0f,    1.0f,    0.01f,   "%.2f") \
  X(float,   cloudCoverageSpread,                0.2f,                            WK_Scalar,     "Clouds",         "Coverage & Shape", "Coverage Spread",            0.0f,    1.0f,    0.01f,   "%.2f") \
  X(float,   cloudCoverageNoiseScale,            0.0033f,                         WK_PatchPerKm, "Clouds",         "Coverage & Shape", "Coverage Patch Size",        100.0f,  10000.0f, 5.0f,   "%.0f km") \
  X(float,   cloudTypeMean,                      0.5f,                            WK_Scalar,     "Clouds",         "Coverage & Shape", "Cloud Type",                 0.0f,    1.0f,    0.01f,   "%.2f") \
  X(float,   cloudTypeSpread,                    0.2f,                            WK_Scalar,     "Clouds",         "Coverage & Shape", "Type Spread",                0.0f,    1.0f,    0.01f,   "%.2f") \
  X(float,   cloudTypeNoiseScale,                0.0034f,                         WK_PatchPerKm, "Clouds",         "Coverage & Shape", "Type Patch Size",            294.0f,  10000.0f, 5.0f,   "%.0f km") \
  X(Vector3, cloudColor,                         Vector3(0.89f, 0.92f, 1.0f),     WK_Color,      "Clouds",         "Look",             "Color",                      0.0f,    1.5f,    0.01f,   "%.2f") \
  X(float,   cloudWindSpeed,                     0.02f,                           WK_SpeedKmS,   "Clouds",         "Wind",             "Wind Speed",                 0.0f,    1000.0f, 0.5f,    "%.1f m/s") \
  X(float,   cloudWindDirection,                 45.0f,                           WK_Angle,      "Clouds",         "Wind",             "Wind Direction",             0.0f,    360.0f,  1.0f,    "%.1f deg") \
  X(float,   cloudShadowStrength,                1.0f,                            WK_Scalar,     "Clouds",         "Lighting",         "Ground Shadow",              0.0f,    1.0f,    0.01f,   "%.2f") \
  X(float,   cloudThickness,                     3.05f,                           WK_Scalar,     "Clouds",         "Look",             "Depth",                      0.0f,    10.0f,   0.05f,   "%.2f") \
  X(float, cloudUndersideLightSigma, 0.12f, WK_Scalar, "Clouds", "Lighting", "Underside Shading", 0.0f, 1.0f, 0.01f,  "%.2f") \
  X(float, cloudBottomDarkening,     1.0f,  WK_Scalar, "Clouds", "Lighting", "Bottom Darkening",  0.0f, 1.0f, 0.01f,  "%.2f") \
  X(float, cloudAerialFadePerKm,     0.15f, WK_Scalar, "Clouds", "Distance", "Horizon Fade",      0.0f, 1.0f, 0.005f, "%.3f") \
  X(float, cloudAerialHazePerKm,     0.05f, WK_Scalar, "Clouds", "Distance", "Distance Haze",     0.0f, 1.0f, 0.005f, "%.3f") \
  /* Atmosphere (5) */ \
  X(float,   airDensity,                         1.0f,                            WK_Scalar,     "Atmosphere",     "Atmosphere",       "Air",                        0.0f,    5.0f,    0.05f,   "%.2f") \
  X(float,   aerosolDensity,                     1.0f,                            WK_Scalar,     "Atmosphere",     "Atmosphere",       "Dust",                       0.0f,    5.0f,    0.05f,   "%.2f") \
  X(Vector3, sunIlluminance,                     Vector3(20.0f, 20.0f, 20.0f),    WK_Color,      "Atmosphere",     "Atmosphere",       "Sun Illuminance",            0.0f,    100.0f,  0.5f,    "%.1f") \
  X(Vector3, rayleighScattering,                 Vector3(5.8e-3f, 13.5e-3f, 33.1e-3f), WK_Color, "Atmosphere",     "Atmosphere",       "Air Color (Base)",           0.0f,    0.05f,   0.0005f, "%.4f") \
  X(float,   skyIndirectRadianceScale,           1.0f,                            WK_Scalar,     "Atmosphere",     "Atmosphere",       "Sky Indirect Scale",         0.0f,    20.0f,   0.01f,   "%.2f") \
  /* Sky/moon mood (4) */ \
  X(float,   nightSkyBrightness,                 0.008f,                          WK_Scalar,     "Sky & Moon",     "Sky & Moon",       "Night Sky Brightness",       0.0f,    1.0f,    0.001f,  "%.3f") \
  X(Vector3, nightSkyColor,                      Vector3(0.15f, 0.2f, 0.4f),      WK_Color,      "Sky & Moon",     "Sky & Moon",       "Night Sky Color",            0.0f,    1.0f,    0.005f,  "%.3f") \
  X(float,   moonNeeStrength,                    1.0f,                            WK_Scalar,     "Sky & Moon",     "Sky & Moon",       "NEE Strength",               0.0f,    10.0f,   0.05f,   "%.2f") \
  X(float,   moonAtmosphericCouplingStrength,    1.0f,                            WK_Scalar,     "Sky & Moon",     "Sky & Moon",       "Atmospheric Coupling",       0.0f,    10.0f,   0.05f,   "%.2f") \
  /* Volumetric (27); volumetricAnisotropy avoids clash with the old cloudAnisotropy */ \
  X(Vector3, transmittanceColor,                 Vector3(0.999f, 0.999f, 0.999f), WK_Color,      "Volumetric Fog", "Medium",           "Transmittance Color",        0.0f,    1.0f,    0.005f,  "%.3f") \
  X(float,   transmittanceMeasurementDistanceMeters, 200.0f,                      WK_Extinction, "Volumetric Fog", "Medium",           "Transmittance Measurement Distance", 1.0f, 2000.0f, 5.0f,  "%.0f") \
  X(Vector3, singleScatteringAlbedo,             Vector3(0.999f, 0.999f, 0.999f), WK_Color,      "Volumetric Fog", "Medium",           "Single Scattering Albedo",   0.0f,    1.0f,    0.005f,  "%.3f") \
  /* Volumetric appearance (fork - full set) */ \
  X(float, fogSunVisibilityGain, 1.0f, WK_Scalar, "Volumetric Fog", "Medium", "Fog Sun Visibility Gain", 0.0f, 4.0f, 0.05f, "%.2f") \
  X(float, volumetricConsumerGain, 0.008f, WK_Scalar, "Volumetric Fog", "Medium", "Fog Brightness Gain", 0.0f, 0.05f, 0.0005f, "%.4f") \
  X(bool, enableHeterogeneousFog, false, WK_Step, "Volumetric Fog", "Heterogeneous", "Enable Heterogeneous Fog", 0.0f, 1.0f, 1.0f, "%.0f") \
  X(float, noiseFieldDensityScale, 1.0f, WK_Scalar, "Volumetric Fog", "Heterogeneous", "Noise Field Density Scale", 0.0f, 5.0f, 0.05f, "%.2f") \
  X(float, noiseFieldDensityExponent, 2.0f, WK_Scalar, "Volumetric Fog", "Heterogeneous", "Noise Field Density Exponent", 0.1f, 8.0f, 0.05f, "%.2f") \
  X(float, noiseFieldInitialFrequencyPerMeter, 8.0f, WK_Scalar, "Volumetric Fog", "Heterogeneous", "Noise Field Initial Frequency", 0.1f, 64.0f, 0.1f, "%.2f") \
  X(float, noiseFieldLacunarity, 2.0f, WK_Scalar, "Volumetric Fog", "Heterogeneous", "Noise Field Lacunarity", 0.1f, 4.0f, 0.05f, "%.2f") \
  X(float, noiseFieldGain, 0.5f, WK_Scalar, "Volumetric Fog", "Heterogeneous", "Noise Field Gain", 0.0f, 1.0f, 0.01f, "%.2f") \
  X(float, noiseFieldTimeScale, 0.5f, WK_Scalar, "Volumetric Fog", "Heterogeneous", "Noise Field Time Scale", 0.0f, 4.0f, 0.05f, "%.2f") \
  X(float, noiseFieldSubStepSizeMeters, 10.0f, WK_Scalar, "Volumetric Fog", "Heterogeneous", "Noise Field Substep Size", 0.5f, 50.0f, 0.5f, "%.1f") \
  X(float, froxelMaxDistanceMeters, 20.0f, WK_Scalar, "Volumetric Fog", "Reach", "Froxel Max Distance", 1.0f, 200.0f, 1.0f, "%.0f") \
  X(bool, enableFogRemap, false, WK_Step, "Volumetric Fog", "Fog Remap", "Enable Legacy Fog Remapping", 0.0f, 1.0f, 1.0f, "%.0f") \
  X(bool, enableFogColorRemap, false, WK_Step, "Volumetric Fog", "Fog Remap", "Enable Fog Color Remapping", 0.0f, 1.0f, 1.0f, "%.0f") \
  X(bool, enableFogMaxDistanceRemap, true, WK_Step, "Volumetric Fog", "Fog Remap", "Enable Fog Max Distance Remapping", 0.0f, 1.0f, 1.0f, "%.0f") \
  X(float, fogRemapMaxDistanceMinMeters, 1.0f, WK_Scalar, "Volumetric Fog", "Fog Remap", "Legacy Max Distance Min", 0.0f, 200.0f, 0.5f, "%.1f") \
  X(float, fogRemapMaxDistanceMaxMeters, 40.0f, WK_Scalar, "Volumetric Fog", "Fog Remap", "Legacy Max Distance Max", 0.0f, 500.0f, 1.0f, "%.1f") \
  X(float, fogRemapTransmittanceMeasurementDistanceMinMeters, 20.0f, WK_Scalar, "Volumetric Fog", "Fog Remap", "Remapped Transmittance Measurement Distance Min", 1.0f, 500.0f, 1.0f, "%.1f") \
  X(float, fogRemapTransmittanceMeasurementDistanceMaxMeters, 100.0f, WK_Scalar, "Volumetric Fog", "Fog Remap", "Remapped Transmittance Measurement Distance Max", 1.0f, 2000.0f, 5.0f, "%.0f") \
  X(float, fogRemapColorMultiscatteringScale, 0.1f, WK_Scalar, "Volumetric Fog", "Fog Remap", "Color Multiscattering Scale", 0.0f, 2.0f, 0.01f, "%.2f") \
  X(bool,  enableTranslucentShadows, false, WK_Step,   "Volumetric Fog", "Medium",        "Enable Translucent Shadows", 0.0f, 1.0f,  1.0f,  "%.0f") \
  X(float, atmosphereSunFogScale,    1.0f,  WK_Scalar, "Volumetric Fog", "Medium",        "Atmosphere Sun Fog Scale", 0.0f, 50.0f, 0.05f, "%.2f") \
  X(float, depthOffset,              0.5f,  WK_Scalar, "Volumetric Fog", "Medium",        "Depth Offset",        0.0f, 1.0f,  0.01f, "%.2f") \
  X(float, noiseFieldOctaves,        2.0f,  WK_Scalar, "Volumetric Fog", "Heterogeneous", "Noise Field Number of Octaves", 1.0f, 8.0f,  1.0f,  "%.0f") \
  X(float,   volumetricAnisotropy,               0.0f,                            WK_Scalar,     "Volumetric Fog", "Medium",           "Anisotropy",                -1.0f,    1.0f,    0.01f,   "%.2f")

// ---------------------------------------------------------------------------
// Per-field RTX_OPTION generator. Takes the preset name plus the X-macro's
// (type, fieldName, defaultVal) tuple and emits one RTX_OPTION declaration in
// the rtx.weather.preset.<presetName> namespace with getter
// presetName_fieldName.
// ---------------------------------------------------------------------------
#define WEATHER_PRESET_RTX_OPTION_FOR(presetName, type, fieldName, defaultVal)                     \
  RTX_OPTION("rtx.weather.preset." #presetName, type, presetName##_##fieldName, defaultVal,        \
             "Weather preset '" #presetName "' value for " #fieldName ". Override per-game in user.conf.")

// ---------------------------------------------------------------------------
// Per-preset binder macros. WEATHER_PRESET_VALUES_<name>(X) expects X to be a
// 3-arg macro (type, name, default), but RTX_OPTION generation also needs the
// preset name. Each binder closes over a specific preset name and forwards to
// WEATHER_PRESET_RTX_OPTION_FOR.
//
// Adding a new preset requires (a) defining a new binder here, (b) defining a
// WEATHER_PRESET_VALUES_<name> macro below, and (c) adding a
// DECLARE_WEATHER_PRESET line to DECLARE_ALL_WEATHER_PRESETS below.
// ---------------------------------------------------------------------------
#define WEATHER_PRESET_BIND_clear(type, name, def)         WEATHER_PRESET_RTX_OPTION_FOR(clear,         type, name, def);
#define WEATHER_PRESET_BIND_partlyCloudy(type, name, def)  WEATHER_PRESET_RTX_OPTION_FOR(partlyCloudy,  type, name, def);
#define WEATHER_PRESET_BIND_overcast(type, name, def)      WEATHER_PRESET_RTX_OPTION_FOR(overcast,      type, name, def);
#define WEATHER_PRESET_BIND_hazy(type, name, def)          WEATHER_PRESET_RTX_OPTION_FOR(hazy,          type, name, def);
#define WEATHER_PRESET_BIND_foggy(type, name, def)         WEATHER_PRESET_RTX_OPTION_FOR(foggy,         type, name, def);
#define WEATHER_PRESET_BIND_drizzle(type, name, def)       WEATHER_PRESET_RTX_OPTION_FOR(drizzle,       type, name, def);
#define WEATHER_PRESET_BIND_rainstorm(type, name, def)     WEATHER_PRESET_RTX_OPTION_FOR(rainstorm,     type, name, def);
#define WEATHER_PRESET_BIND_thunderstorm(type, name, def)  WEATHER_PRESET_RTX_OPTION_FOR(thunderstorm,  type, name, def);
#define WEATHER_PRESET_BIND_snow(type, name, def)          WEATHER_PRESET_RTX_OPTION_FOR(snow,          type, name, def);
#define WEATHER_PRESET_BIND_blizzard(type, name, def)      WEATHER_PRESET_RTX_OPTION_FOR(blizzard,      type, name, def);
#define WEATHER_PRESET_BIND_sandstorm(type, name, def)     WEATHER_PRESET_RTX_OPTION_FOR(sandstorm,     type, name, def);
#define WEATHER_PRESET_BIND_smoggy(type, name, def)        WEATHER_PRESET_RTX_OPTION_FOR(smoggy,        type, name, def);

// ---------------------------------------------------------------------------
// Per-preset value X-macros — one per archetype, 52 fields each, in the same
// order as WEATHER_PRESET_FIELD_LIST. Fields not explicitly tuned use the
// neutral default from WEATHER_PRESET_FIELD_LIST, which is also the canonical
// field order — see that macro above rather than duplicating the list here.
// ---------------------------------------------------------------------------

// clear — sunny, crisp, low haze
#define WEATHER_PRESET_VALUES_clear(X)                                                                 \
  X(float,   cloudDensity,                              0.4f)                                          \
  X(float,   cloudCoverageMean,                         0.10f)                                         \
  X(float,   cloudCoverageSpread,                       0.10f)                                         \
  X(float,   cloudCoverageNoiseScale,                   0.0033f)                                       \
  X(float,   cloudTypeMean,                             0.6f)                                          \
  X(float,   cloudTypeSpread,                           0.3f)                                          \
  X(float,   cloudTypeNoiseScale,                       0.0034f)                                       \
  X(Vector3, cloudColor,                                Vector3(0.95f, 0.97f, 1.00f))                  \
  X(float,   cloudWindSpeed,                            0.02f)                                         \
  X(float,   cloudWindDirection,                        45.0f)                                         \
  X(float,   cloudShadowStrength,                       0.0f)                                          \
  X(float,   cloudThickness,                            2.0f)                                          \
  X(float, cloudUndersideLightSigma, 0.12f) \
  X(float, cloudBottomDarkening,     1.0f) \
  X(float, cloudAerialFadePerKm,     0.15f) \
  X(float, cloudAerialHazePerKm,     0.05f) \
  X(Vector3, rayleighScattering, Vector3(5.8e-3f, 13.5e-3f, 33.1e-3f)) \
  X(Vector3, nightSkyColor,      Vector3(0.15f, 0.2f, 0.4f)) \
  X(float, skyIndirectRadianceScale, 1.0f) \
  X(float,   airDensity,                                0.95f)                                         \
  X(float,   aerosolDensity,                            0.7f)                                          \
  X(Vector3, sunIlluminance,                            Vector3(20.0f, 20.0f, 20.0f))                  \
  X(float,   nightSkyBrightness,                        0.008f)                                        \
  X(float,   moonNeeStrength,                           1.0f)                                          \
  X(float,   moonAtmosphericCouplingStrength,           1.0f)                                          \
  X(Vector3, transmittanceColor,                        Vector3(0.999f, 0.999f, 0.999f))               \
  X(float,   transmittanceMeasurementDistanceMeters,    1000.0f)                                       \
  X(Vector3, singleScatteringAlbedo,                    Vector3(0.999f, 0.999f, 0.999f))               \
  X(float, fogSunVisibilityGain, 1.0f) \
  X(float, volumetricConsumerGain, 0.008f) \
  X(bool, enableHeterogeneousFog, false) \
  X(float, noiseFieldDensityScale, 1.0f) \
  X(float, noiseFieldDensityExponent, 2.0f) \
  X(float, noiseFieldInitialFrequencyPerMeter, 8.0f) \
  X(float, noiseFieldLacunarity, 2.0f) \
  X(float, noiseFieldGain, 0.5f) \
  X(float, noiseFieldTimeScale, 0.5f) \
  X(float, noiseFieldSubStepSizeMeters, 10.0f) \
  X(float, froxelMaxDistanceMeters, 20.0f) \
  X(bool, enableFogRemap, false) \
  X(bool, enableFogColorRemap, false) \
  X(bool, enableFogMaxDistanceRemap, true) \
  X(float, fogRemapMaxDistanceMinMeters, 1.0f) \
  X(float, fogRemapMaxDistanceMaxMeters, 40.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMinMeters, 20.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMaxMeters, 100.0f) \
  X(float, fogRemapColorMultiscatteringScale, 0.1f) \
  X(bool,  enableTranslucentShadows, false) \
  X(float, atmosphereSunFogScale,    1.0f) \
  X(float, depthOffset,              0.5f) \
  X(float, noiseFieldOctaves,        2.0f) \
  X(float,   volumetricAnisotropy,                      0.0f)

// partlyCloudy — light scattered clouds
#define WEATHER_PRESET_VALUES_partlyCloudy(X)                                                          \
  X(float,   cloudDensity,                              0.9f)                                          \
  X(float,   cloudCoverageMean,                         0.30f)                                         \
  X(float,   cloudCoverageSpread,                       0.20f)                                         \
  X(float,   cloudCoverageNoiseScale,                   0.0033f)                                       \
  X(float,   cloudTypeMean,                             0.5f)                                          \
  X(float,   cloudTypeSpread,                           0.3f)                                          \
  X(float,   cloudTypeNoiseScale,                       0.0034f)                                       \
  X(Vector3, cloudColor,                                Vector3(0.92f, 0.95f, 1.00f))                  \
  X(float,   cloudWindSpeed,                            0.02f)                                         \
  X(float,   cloudWindDirection,                        45.0f)                                         \
  X(float,   cloudShadowStrength,                       0.05f)                                         \
  X(float,   cloudThickness,                            2.5f)                                          \
  X(float, cloudUndersideLightSigma, 0.12f) \
  X(float, cloudBottomDarkening,     1.0f) \
  X(float, cloudAerialFadePerKm,     0.15f) \
  X(float, cloudAerialHazePerKm,     0.05f) \
  X(Vector3, rayleighScattering, Vector3(5.8e-3f, 13.5e-3f, 33.1e-3f)) \
  X(Vector3, nightSkyColor,      Vector3(0.15f, 0.2f, 0.4f)) \
  X(float, skyIndirectRadianceScale, 1.0f) \
  X(float,   airDensity,                                1.0f)                                          \
  X(float,   aerosolDensity,                            1.0f)                                          \
  X(Vector3, sunIlluminance,                            Vector3(19.0f, 19.0f, 19.0f))                  \
  X(float,   nightSkyBrightness,                        0.008f)                                        \
  X(float,   moonNeeStrength,                           1.0f)                                          \
  X(float,   moonAtmosphericCouplingStrength,           1.0f)                                          \
  X(Vector3, transmittanceColor,                        Vector3(0.998f, 0.998f, 0.998f))               \
  X(float,   transmittanceMeasurementDistanceMeters,    800.0f)                                        \
  X(Vector3, singleScatteringAlbedo,                    Vector3(0.999f, 0.999f, 0.999f))               \
  X(float, fogSunVisibilityGain, 1.0f) \
  X(float, volumetricConsumerGain, 0.008f) \
  X(bool, enableHeterogeneousFog, false) \
  X(float, noiseFieldDensityScale, 1.0f) \
  X(float, noiseFieldDensityExponent, 2.0f) \
  X(float, noiseFieldInitialFrequencyPerMeter, 8.0f) \
  X(float, noiseFieldLacunarity, 2.0f) \
  X(float, noiseFieldGain, 0.5f) \
  X(float, noiseFieldTimeScale, 0.5f) \
  X(float, noiseFieldSubStepSizeMeters, 10.0f) \
  X(float, froxelMaxDistanceMeters, 20.0f) \
  X(bool, enableFogRemap, false) \
  X(bool, enableFogColorRemap, false) \
  X(bool, enableFogMaxDistanceRemap, true) \
  X(float, fogRemapMaxDistanceMinMeters, 1.0f) \
  X(float, fogRemapMaxDistanceMaxMeters, 40.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMinMeters, 20.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMaxMeters, 100.0f) \
  X(float, fogRemapColorMultiscatteringScale, 0.1f) \
  X(bool,  enableTranslucentShadows, false) \
  X(float, atmosphereSunFogScale,    1.0f) \
  X(float, depthOffset,              0.5f) \
  X(float, noiseFieldOctaves,        2.0f) \
  X(float,   volumetricAnisotropy,                      0.05f)

// overcast — current default look
#define WEATHER_PRESET_VALUES_overcast(X)                                                              \
  X(float,   cloudDensity,                              1.8f)                                          \
  X(float,   cloudCoverageMean,                         0.64f)                                         \
  X(float,   cloudCoverageSpread,                       0.16f)                                         \
  X(float,   cloudCoverageNoiseScale,                   0.0033f)                                       \
  X(float,   cloudTypeMean,                             0.5f)                                          \
  X(float,   cloudTypeSpread,                           0.2f)                                          \
  X(float,   cloudTypeNoiseScale,                       0.0034f)                                       \
  X(Vector3, cloudColor,                                Vector3(0.89f, 0.92f, 1.00f))                  \
  X(float,   cloudWindSpeed,                            0.02f)                                         \
  X(float,   cloudWindDirection,                        45.0f)                                         \
  X(float,   cloudShadowStrength,                       0.10f)                                         \
  X(float,   cloudThickness,                            3.05f)                                         \
  X(float, cloudUndersideLightSigma, 0.12f) \
  X(float, cloudBottomDarkening,     1.0f) \
  X(float, cloudAerialFadePerKm,     0.15f) \
  X(float, cloudAerialHazePerKm,     0.05f) \
  X(Vector3, rayleighScattering, Vector3(5.8e-3f, 13.5e-3f, 33.1e-3f)) \
  X(Vector3, nightSkyColor,      Vector3(0.15f, 0.2f, 0.4f)) \
  X(float, skyIndirectRadianceScale, 1.0f) \
  X(float,   airDensity,                                1.0f)                                          \
  X(float,   aerosolDensity,                            1.1f)                                          \
  X(Vector3, sunIlluminance,                            Vector3(15.0f, 15.0f, 15.0f))                  \
  X(float,   nightSkyBrightness,                        0.008f)                                        \
  X(float,   moonNeeStrength,                           1.0f)                                          \
  X(float,   moonAtmosphericCouplingStrength,           1.0f)                                          \
  X(Vector3, transmittanceColor,                        Vector3(0.995f, 0.995f, 0.995f))               \
  X(float,   transmittanceMeasurementDistanceMeters,    500.0f)                                        \
  X(Vector3, singleScatteringAlbedo,                    Vector3(0.999f, 0.999f, 0.999f))               \
  X(float, fogSunVisibilityGain, 1.0f) \
  X(float, volumetricConsumerGain, 0.008f) \
  X(bool, enableHeterogeneousFog, false) \
  X(float, noiseFieldDensityScale, 1.0f) \
  X(float, noiseFieldDensityExponent, 2.0f) \
  X(float, noiseFieldInitialFrequencyPerMeter, 8.0f) \
  X(float, noiseFieldLacunarity, 2.0f) \
  X(float, noiseFieldGain, 0.5f) \
  X(float, noiseFieldTimeScale, 0.5f) \
  X(float, noiseFieldSubStepSizeMeters, 10.0f) \
  X(float, froxelMaxDistanceMeters, 20.0f) \
  X(bool, enableFogRemap, false) \
  X(bool, enableFogColorRemap, false) \
  X(bool, enableFogMaxDistanceRemap, true) \
  X(float, fogRemapMaxDistanceMinMeters, 1.0f) \
  X(float, fogRemapMaxDistanceMaxMeters, 40.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMinMeters, 20.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMaxMeters, 100.0f) \
  X(float, fogRemapColorMultiscatteringScale, 0.1f) \
  X(bool,  enableTranslucentShadows, false) \
  X(float, atmosphereSunFogScale,    1.0f) \
  X(float, depthOffset,              0.5f) \
  X(float, noiseFieldOctaves,        2.0f) \
  X(float,   volumetricAnisotropy,                      0.05f)

// hazy — warm summer haze
#define WEATHER_PRESET_VALUES_hazy(X)                                                                  \
  X(float,   cloudDensity,                              1.0f)                                          \
  X(float,   cloudCoverageMean,                         0.40f)                                         \
  X(float,   cloudCoverageSpread,                       0.20f)                                         \
  X(float,   cloudCoverageNoiseScale,                   0.0033f)                                       \
  X(float,   cloudTypeMean,                             0.4f)                                          \
  X(float,   cloudTypeSpread,                           0.3f)                                          \
  X(float,   cloudTypeNoiseScale,                       0.0034f)                                       \
  X(Vector3, cloudColor,                                Vector3(0.92f, 0.91f, 0.88f))                  \
  X(float,   cloudWindSpeed,                            0.02f)                                         \
  X(float,   cloudWindDirection,                        45.0f)                                         \
  X(float,   cloudShadowStrength,                       0.10f)                                         \
  X(float,   cloudThickness,                            2.5f)                                          \
  X(float, cloudUndersideLightSigma, 0.12f) \
  X(float, cloudBottomDarkening,     1.0f) \
  X(float, cloudAerialFadePerKm,     0.15f) \
  X(float, cloudAerialHazePerKm,     0.05f) \
  X(Vector3, rayleighScattering, Vector3(5.8e-3f, 13.5e-3f, 33.1e-3f)) \
  X(Vector3, nightSkyColor,      Vector3(0.15f, 0.2f, 0.4f)) \
  X(float, skyIndirectRadianceScale, 1.0f) \
  X(float,   airDensity,                                1.1f)                                          \
  X(float,   aerosolDensity,                            1.5f)                                          \
  X(Vector3, sunIlluminance,                            Vector3(17.0f, 16.0f, 14.0f))                  \
  X(float,   nightSkyBrightness,                        0.010f)                                        \
  X(float,   moonNeeStrength,                           1.0f)                                          \
  X(float,   moonAtmosphericCouplingStrength,           1.0f)                                          \
  X(Vector3, transmittanceColor,                        Vector3(0.985f, 0.97f, 0.94f))                 \
  X(float,   transmittanceMeasurementDistanceMeters,    250.0f)                                        \
  X(Vector3, singleScatteringAlbedo,                    Vector3(0.99f, 0.98f, 0.96f))                  \
  X(float, fogSunVisibilityGain, 1.0f) \
  X(float, volumetricConsumerGain, 0.008f) \
  X(bool, enableHeterogeneousFog, false) \
  X(float, noiseFieldDensityScale, 1.0f) \
  X(float, noiseFieldDensityExponent, 2.0f) \
  X(float, noiseFieldInitialFrequencyPerMeter, 8.0f) \
  X(float, noiseFieldLacunarity, 2.0f) \
  X(float, noiseFieldGain, 0.5f) \
  X(float, noiseFieldTimeScale, 0.5f) \
  X(float, noiseFieldSubStepSizeMeters, 10.0f) \
  X(float, froxelMaxDistanceMeters, 20.0f) \
  X(bool, enableFogRemap, false) \
  X(bool, enableFogColorRemap, false) \
  X(bool, enableFogMaxDistanceRemap, true) \
  X(float, fogRemapMaxDistanceMinMeters, 1.0f) \
  X(float, fogRemapMaxDistanceMaxMeters, 40.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMinMeters, 20.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMaxMeters, 100.0f) \
  X(float, fogRemapColorMultiscatteringScale, 0.1f) \
  X(bool,  enableTranslucentShadows, false) \
  X(float, atmosphereSunFogScale,    1.0f) \
  X(float, depthOffset,              0.5f) \
  X(float, noiseFieldOctaves,        2.0f) \
  X(float,   volumetricAnisotropy,                      0.30f)

// foggy — the headline fog preset
#define WEATHER_PRESET_VALUES_foggy(X)                                                                 \
  X(float,   cloudDensity,                              0.6f)                                          \
  X(float,   cloudCoverageMean,                         0.30f)                                         \
  X(float,   cloudCoverageSpread,                       0.20f)                                         \
  X(float,   cloudCoverageNoiseScale,                   0.0033f)                                       \
  X(float,   cloudTypeMean,                             0.2f)                                          \
  X(float,   cloudTypeSpread,                           0.2f)                                          \
  X(float,   cloudTypeNoiseScale,                       0.0034f)                                       \
  X(Vector3, cloudColor,                                Vector3(0.85f, 0.88f, 0.92f))                  \
  X(float,   cloudWindSpeed,                            0.02f)                                         \
  X(float,   cloudWindDirection,                        45.0f)                                         \
  X(float,   cloudShadowStrength,                       0.05f)                                         \
  X(float,   cloudThickness,                            2.0f)                                          \
  X(float, cloudUndersideLightSigma, 0.12f) \
  X(float, cloudBottomDarkening,     1.0f) \
  X(float, cloudAerialFadePerKm,     0.15f) \
  X(float, cloudAerialHazePerKm,     0.05f) \
  X(Vector3, rayleighScattering, Vector3(5.8e-3f, 13.5e-3f, 33.1e-3f)) \
  X(Vector3, nightSkyColor,      Vector3(0.15f, 0.2f, 0.4f)) \
  X(float, skyIndirectRadianceScale, 1.0f) \
  X(float,   airDensity,                                1.2f)                                          \
  X(float,   aerosolDensity,                            2.0f)                                          \
  X(Vector3, sunIlluminance,                            Vector3(10.0f, 10.0f, 10.0f))                  \
  X(float,   nightSkyBrightness,                        0.012f)                                        \
  X(float,   moonNeeStrength,                           1.0f)                                          \
  X(float,   moonAtmosphericCouplingStrength,           1.0f)                                          \
  X(Vector3, transmittanceColor,                        Vector3(0.92f, 0.94f, 0.96f))                  \
  X(float,   transmittanceMeasurementDistanceMeters,    80.0f)                                         \
  X(Vector3, singleScatteringAlbedo,                    Vector3(0.99f, 0.99f, 0.99f))                  \
  X(float, fogSunVisibilityGain, 1.0f) \
  X(float, volumetricConsumerGain, 0.008f) \
  X(bool, enableHeterogeneousFog, false) \
  X(float, noiseFieldDensityScale, 1.0f) \
  X(float, noiseFieldDensityExponent, 2.0f) \
  X(float, noiseFieldInitialFrequencyPerMeter, 8.0f) \
  X(float, noiseFieldLacunarity, 2.0f) \
  X(float, noiseFieldGain, 0.5f) \
  X(float, noiseFieldTimeScale, 0.5f) \
  X(float, noiseFieldSubStepSizeMeters, 10.0f) \
  X(float, froxelMaxDistanceMeters, 20.0f) \
  X(bool, enableFogRemap, false) \
  X(bool, enableFogColorRemap, false) \
  X(bool, enableFogMaxDistanceRemap, true) \
  X(float, fogRemapMaxDistanceMinMeters, 1.0f) \
  X(float, fogRemapMaxDistanceMaxMeters, 40.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMinMeters, 20.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMaxMeters, 100.0f) \
  X(float, fogRemapColorMultiscatteringScale, 0.1f) \
  X(bool,  enableTranslucentShadows, false) \
  X(float, atmosphereSunFogScale,    1.0f) \
  X(float, depthOffset,              0.5f) \
  X(float, noiseFieldOctaves,        2.0f) \
  X(float,   volumetricAnisotropy,                      0.0f)

// drizzle — light rain, medium fog
#define WEATHER_PRESET_VALUES_drizzle(X)                                                               \
  X(float,   cloudDensity,                              1.4f)                                          \
  X(float,   cloudCoverageMean,                         0.60f)                                         \
  X(float,   cloudCoverageSpread,                       0.20f)                                         \
  X(float,   cloudCoverageNoiseScale,                   0.0033f)                                       \
  X(float,   cloudTypeMean,                             0.3f)                                          \
  X(float,   cloudTypeSpread,                           0.3f)                                          \
  X(float,   cloudTypeNoiseScale,                       0.0034f)                                       \
  X(Vector3, cloudColor,                                Vector3(0.78f, 0.82f, 0.88f))                  \
  X(float,   cloudWindSpeed,                            0.02f)                                         \
  X(float,   cloudWindDirection,                        45.0f)                                         \
  X(float,   cloudShadowStrength,                       0.20f)                                         \
  X(float,   cloudThickness,                            3.0f)                                          \
  X(float, cloudUndersideLightSigma, 0.12f) \
  X(float, cloudBottomDarkening,     1.0f) \
  X(float, cloudAerialFadePerKm,     0.15f) \
  X(float, cloudAerialHazePerKm,     0.05f) \
  X(Vector3, rayleighScattering, Vector3(5.8e-3f, 13.5e-3f, 33.1e-3f)) \
  X(Vector3, nightSkyColor,      Vector3(0.15f, 0.2f, 0.4f)) \
  X(float, skyIndirectRadianceScale, 1.0f) \
  X(float,   airDensity,                                1.1f)                                          \
  X(float,   aerosolDensity,                            1.5f)                                          \
  X(Vector3, sunIlluminance,                            Vector3(11.0f, 12.0f, 14.0f))                  \
  X(float,   nightSkyBrightness,                        0.010f)                                        \
  X(float,   moonNeeStrength,                           1.0f)                                          \
  X(float,   moonAtmosphericCouplingStrength,           1.0f)                                          \
  X(Vector3, transmittanceColor,                        Vector3(0.95f, 0.96f, 0.97f))                  \
  X(float,   transmittanceMeasurementDistanceMeters,    200.0f)                                        \
  X(Vector3, singleScatteringAlbedo,                    Vector3(0.98f, 0.98f, 0.99f))                  \
  X(float, fogSunVisibilityGain, 1.0f) \
  X(float, volumetricConsumerGain, 0.008f) \
  X(bool, enableHeterogeneousFog, false) \
  X(float, noiseFieldDensityScale, 1.0f) \
  X(float, noiseFieldDensityExponent, 2.0f) \
  X(float, noiseFieldInitialFrequencyPerMeter, 8.0f) \
  X(float, noiseFieldLacunarity, 2.0f) \
  X(float, noiseFieldGain, 0.5f) \
  X(float, noiseFieldTimeScale, 0.5f) \
  X(float, noiseFieldSubStepSizeMeters, 10.0f) \
  X(float, froxelMaxDistanceMeters, 20.0f) \
  X(bool, enableFogRemap, false) \
  X(bool, enableFogColorRemap, false) \
  X(bool, enableFogMaxDistanceRemap, true) \
  X(float, fogRemapMaxDistanceMinMeters, 1.0f) \
  X(float, fogRemapMaxDistanceMaxMeters, 40.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMinMeters, 20.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMaxMeters, 100.0f) \
  X(float, fogRemapColorMultiscatteringScale, 0.1f) \
  X(bool,  enableTranslucentShadows, false) \
  X(float, atmosphereSunFogScale,    1.0f) \
  X(float, depthOffset,              0.5f) \
  X(float, noiseFieldOctaves,        2.0f) \
  X(float,   volumetricAnisotropy,                      0.10f)

// rainstorm — heavy clouds, dim sun, dense fog
#define WEATHER_PRESET_VALUES_rainstorm(X)                                                             \
  X(float,   cloudDensity,                              2.5f)                                          \
  X(float,   cloudCoverageMean,                         0.80f)                                         \
  X(float,   cloudCoverageSpread,                       0.15f)                                         \
  X(float,   cloudCoverageNoiseScale,                   0.0033f)                                       \
  X(float,   cloudTypeMean,                             0.4f)                                          \
  X(float,   cloudTypeSpread,                           0.3f)                                          \
  X(float,   cloudTypeNoiseScale,                       0.0034f)                                       \
  X(Vector3, cloudColor,                                Vector3(0.65f, 0.68f, 0.75f))                  \
  X(float,   cloudWindSpeed,                            0.02f)                                         \
  X(float,   cloudWindDirection,                        45.0f)                                         \
  X(float,   cloudShadowStrength,                       0.40f)                                         \
  X(float,   cloudThickness,                            4.0f)                                          \
  X(float, cloudUndersideLightSigma, 0.12f) \
  X(float, cloudBottomDarkening,     1.0f) \
  X(float, cloudAerialFadePerKm,     0.15f) \
  X(float, cloudAerialHazePerKm,     0.05f) \
  X(Vector3, rayleighScattering, Vector3(5.8e-3f, 13.5e-3f, 33.1e-3f)) \
  X(Vector3, nightSkyColor,      Vector3(0.15f, 0.2f, 0.4f)) \
  X(float, skyIndirectRadianceScale, 1.0f) \
  X(float,   airDensity,                                1.0f)                                          \
  X(float,   aerosolDensity,                            1.4f)                                          \
  X(Vector3, sunIlluminance,                            Vector3(7.0f, 8.0f, 10.0f))                    \
  X(float,   nightSkyBrightness,                        0.008f)                                        \
  X(float,   moonNeeStrength,                           1.0f)                                          \
  X(float,   moonAtmosphericCouplingStrength,           1.0f)                                          \
  X(Vector3, transmittanceColor,                        Vector3(0.85f, 0.88f, 0.92f))                  \
  X(float,   transmittanceMeasurementDistanceMeters,    100.0f)                                        \
  X(Vector3, singleScatteringAlbedo,                    Vector3(0.97f, 0.97f, 0.98f))                  \
  X(float, fogSunVisibilityGain, 1.0f) \
  X(float, volumetricConsumerGain, 0.008f) \
  X(bool, enableHeterogeneousFog, false) \
  X(float, noiseFieldDensityScale, 1.0f) \
  X(float, noiseFieldDensityExponent, 2.0f) \
  X(float, noiseFieldInitialFrequencyPerMeter, 8.0f) \
  X(float, noiseFieldLacunarity, 2.0f) \
  X(float, noiseFieldGain, 0.5f) \
  X(float, noiseFieldTimeScale, 0.5f) \
  X(float, noiseFieldSubStepSizeMeters, 10.0f) \
  X(float, froxelMaxDistanceMeters, 20.0f) \
  X(bool, enableFogRemap, false) \
  X(bool, enableFogColorRemap, false) \
  X(bool, enableFogMaxDistanceRemap, true) \
  X(float, fogRemapMaxDistanceMinMeters, 1.0f) \
  X(float, fogRemapMaxDistanceMaxMeters, 40.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMinMeters, 20.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMaxMeters, 100.0f) \
  X(float, fogRemapColorMultiscatteringScale, 0.1f) \
  X(bool,  enableTranslucentShadows, false) \
  X(float, atmosphereSunFogScale,    1.0f) \
  X(float, depthOffset,              0.5f) \
  X(float, noiseFieldOctaves,        2.0f) \
  X(float,   volumetricAnisotropy,                      0.10f)

// thunderstorm — heaviest, bruised tone (retuned 2026-05-09 by in-game
// tuning against the post-FAST-noise + temporal-smoother + Jensen-revert
// pipeline at cloudAltitude=1.5 km, cloudCurvature=0.38)
#define WEATHER_PRESET_VALUES_thunderstorm(X)                                                          \
  X(float,   cloudDensity,                              2.65f)                                         \
  X(float,   cloudCoverageMean,                         0.95f)                                         \
  X(float,   cloudCoverageSpread,                       0.10f)                                         \
  X(float,   cloudCoverageNoiseScale,                   0.0033f)                                       \
  X(float,   cloudTypeMean,                             0.54f)                                         \
  X(float,   cloudTypeSpread,                           0.28f)                                         \
  X(float,   cloudTypeNoiseScale,                       0.0034f)                                       \
  X(Vector3, cloudColor,                                Vector3(0.61f, 0.63f, 0.69f))                  \
  X(float,   cloudWindSpeed,                            0.02f)                                         \
  X(float,   cloudWindDirection,                        45.0f)                                         \
  X(float,   cloudShadowStrength,                       0.44f)                                         \
  X(float,   cloudThickness,                            4.13f)                                         \
  X(float, cloudUndersideLightSigma, 0.12f) \
  X(float, cloudBottomDarkening,     1.0f) \
  X(float, cloudAerialFadePerKm,     0.15f) \
  X(float, cloudAerialHazePerKm,     0.05f) \
  X(Vector3, rayleighScattering, Vector3(5.8e-3f, 13.5e-3f, 33.1e-3f)) \
  X(Vector3, nightSkyColor,      Vector3(0.15f, 0.2f, 0.4f)) \
  X(float, skyIndirectRadianceScale, 1.0f) \
  X(float,   airDensity,                                1.0f)                                          \
  X(float,   aerosolDensity,                            1.3f)                                          \
  X(Vector3, sunIlluminance,                            Vector3(4.0f, 4.0f, 6.0f))                     \
  X(float,   nightSkyBrightness,                        0.008f)                                        \
  X(float,   moonNeeStrength,                           1.0f)                                          \
  X(float,   moonAtmosphericCouplingStrength,           1.0f)                                          \
  X(Vector3, transmittanceColor,                        Vector3(0.75f, 0.78f, 0.82f))                  \
  X(float,   transmittanceMeasurementDistanceMeters,    60.0f)                                         \
  X(Vector3, singleScatteringAlbedo,                    Vector3(0.95f, 0.95f, 0.97f))                  \
  X(float, fogSunVisibilityGain, 1.0f) \
  X(float, volumetricConsumerGain, 0.008f) \
  X(bool, enableHeterogeneousFog, false) \
  X(float, noiseFieldDensityScale, 1.0f) \
  X(float, noiseFieldDensityExponent, 2.0f) \
  X(float, noiseFieldInitialFrequencyPerMeter, 8.0f) \
  X(float, noiseFieldLacunarity, 2.0f) \
  X(float, noiseFieldGain, 0.5f) \
  X(float, noiseFieldTimeScale, 0.5f) \
  X(float, noiseFieldSubStepSizeMeters, 10.0f) \
  X(float, froxelMaxDistanceMeters, 20.0f) \
  X(bool, enableFogRemap, false) \
  X(bool, enableFogColorRemap, false) \
  X(bool, enableFogMaxDistanceRemap, true) \
  X(float, fogRemapMaxDistanceMinMeters, 1.0f) \
  X(float, fogRemapMaxDistanceMaxMeters, 40.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMinMeters, 20.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMaxMeters, 100.0f) \
  X(float, fogRemapColorMultiscatteringScale, 0.1f) \
  X(bool,  enableTranslucentShadows, false) \
  X(float, atmosphereSunFogScale,    1.0f) \
  X(float, depthOffset,              0.5f) \
  X(float, noiseFieldOctaves,        2.0f) \
  X(float,   volumetricAnisotropy,                      0.0f)

// snow — medium clouds, cool fog, snow particles
#define WEATHER_PRESET_VALUES_snow(X)                                                                  \
  X(float,   cloudDensity,                              1.8f)                                          \
  X(float,   cloudCoverageMean,                         0.65f)                                         \
  X(float,   cloudCoverageSpread,                       0.20f)                                         \
  X(float,   cloudCoverageNoiseScale,                   0.0033f)                                       \
  X(float,   cloudTypeMean,                             0.4f)                                          \
  X(float,   cloudTypeSpread,                           0.3f)                                          \
  X(float,   cloudTypeNoiseScale,                       0.0034f)                                       \
  X(Vector3, cloudColor,                                Vector3(0.95f, 0.97f, 1.00f))                  \
  X(float,   cloudWindSpeed,                            0.02f)                                         \
  X(float,   cloudWindDirection,                        45.0f)                                         \
  X(float,   cloudShadowStrength,                       0.20f)                                         \
  X(float,   cloudThickness,                            3.0f)                                          \
  X(float, cloudUndersideLightSigma, 0.12f) \
  X(float, cloudBottomDarkening,     1.0f) \
  X(float, cloudAerialFadePerKm,     0.15f) \
  X(float, cloudAerialHazePerKm,     0.05f) \
  X(Vector3, rayleighScattering, Vector3(5.8e-3f, 13.5e-3f, 33.1e-3f)) \
  X(Vector3, nightSkyColor,      Vector3(0.15f, 0.2f, 0.4f)) \
  X(float, skyIndirectRadianceScale, 1.0f) \
  X(float,   airDensity,                                1.0f)                                          \
  X(float,   aerosolDensity,                            1.3f)                                          \
  X(Vector3, sunIlluminance,                            Vector3(12.0f, 13.0f, 14.0f))                  \
  X(float,   nightSkyBrightness,                        0.012f)                                        \
  X(float,   moonNeeStrength,                           1.0f)                                          \
  X(float,   moonAtmosphericCouplingStrength,           1.0f)                                          \
  X(Vector3, transmittanceColor,                        Vector3(0.97f, 0.98f, 0.99f))                  \
  X(float,   transmittanceMeasurementDistanceMeters,    250.0f)                                        \
  X(Vector3, singleScatteringAlbedo,                    Vector3(0.99f, 0.99f, 0.99f))                  \
  X(float, fogSunVisibilityGain, 1.0f) \
  X(float, volumetricConsumerGain, 0.008f) \
  X(bool, enableHeterogeneousFog, false) \
  X(float, noiseFieldDensityScale, 1.0f) \
  X(float, noiseFieldDensityExponent, 2.0f) \
  X(float, noiseFieldInitialFrequencyPerMeter, 8.0f) \
  X(float, noiseFieldLacunarity, 2.0f) \
  X(float, noiseFieldGain, 0.5f) \
  X(float, noiseFieldTimeScale, 0.5f) \
  X(float, noiseFieldSubStepSizeMeters, 10.0f) \
  X(float, froxelMaxDistanceMeters, 20.0f) \
  X(bool, enableFogRemap, false) \
  X(bool, enableFogColorRemap, false) \
  X(bool, enableFogMaxDistanceRemap, true) \
  X(float, fogRemapMaxDistanceMinMeters, 1.0f) \
  X(float, fogRemapMaxDistanceMaxMeters, 40.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMinMeters, 20.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMaxMeters, 100.0f) \
  X(float, fogRemapColorMultiscatteringScale, 0.1f) \
  X(bool,  enableTranslucentShadows, false) \
  X(float, atmosphereSunFogScale,    1.0f) \
  X(float, depthOffset,              0.5f) \
  X(float, noiseFieldOctaves,        2.0f) \
  X(float,   volumetricAnisotropy,                      0.0f)

// blizzard — whiteout, severe visibility loss
#define WEATHER_PRESET_VALUES_blizzard(X)                                                              \
  X(float,   cloudDensity,                              3.0f)                                          \
  X(float,   cloudCoverageMean,                         0.95f)                                         \
  X(float,   cloudCoverageSpread,                       0.10f)                                         \
  X(float,   cloudCoverageNoiseScale,                   0.0033f)                                       \
  X(float,   cloudTypeMean,                             0.5f)                                          \
  X(float,   cloudTypeSpread,                           0.2f)                                          \
  X(float,   cloudTypeNoiseScale,                       0.0034f)                                       \
  X(Vector3, cloudColor,                                Vector3(0.92f, 0.96f, 1.00f))                  \
  X(float,   cloudWindSpeed,                            0.02f)                                         \
  X(float,   cloudWindDirection,                        45.0f)                                         \
  X(float,   cloudShadowStrength,                       0.50f)                                         \
  X(float,   cloudThickness,                            4.5f)                                          \
  X(float, cloudUndersideLightSigma, 0.12f) \
  X(float, cloudBottomDarkening,     1.0f) \
  X(float, cloudAerialFadePerKm,     0.15f) \
  X(float, cloudAerialHazePerKm,     0.05f) \
  X(Vector3, rayleighScattering, Vector3(5.8e-3f, 13.5e-3f, 33.1e-3f)) \
  X(Vector3, nightSkyColor,      Vector3(0.15f, 0.2f, 0.4f)) \
  X(float, skyIndirectRadianceScale, 1.0f) \
  X(float,   airDensity,                                1.0f)                                          \
  X(float,   aerosolDensity,                            1.6f)                                          \
  X(Vector3, sunIlluminance,                            Vector3(6.0f, 7.0f, 8.0f))                     \
  X(float,   nightSkyBrightness,                        0.008f)                                        \
  X(float,   moonNeeStrength,                           1.0f)                                          \
  X(float,   moonAtmosphericCouplingStrength,           1.0f)                                          \
  X(Vector3, transmittanceColor,                        Vector3(0.92f, 0.95f, 0.98f))                  \
  X(float,   transmittanceMeasurementDistanceMeters,    50.0f)                                         \
  X(Vector3, singleScatteringAlbedo,                    Vector3(0.99f, 0.99f, 1.00f))                  \
  X(float, fogSunVisibilityGain, 1.0f) \
  X(float, volumetricConsumerGain, 0.008f) \
  X(bool, enableHeterogeneousFog, false) \
  X(float, noiseFieldDensityScale, 1.0f) \
  X(float, noiseFieldDensityExponent, 2.0f) \
  X(float, noiseFieldInitialFrequencyPerMeter, 8.0f) \
  X(float, noiseFieldLacunarity, 2.0f) \
  X(float, noiseFieldGain, 0.5f) \
  X(float, noiseFieldTimeScale, 0.5f) \
  X(float, noiseFieldSubStepSizeMeters, 10.0f) \
  X(float, froxelMaxDistanceMeters, 20.0f) \
  X(bool, enableFogRemap, false) \
  X(bool, enableFogColorRemap, false) \
  X(bool, enableFogMaxDistanceRemap, true) \
  X(float, fogRemapMaxDistanceMinMeters, 1.0f) \
  X(float, fogRemapMaxDistanceMaxMeters, 40.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMinMeters, 20.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMaxMeters, 100.0f) \
  X(float, fogRemapColorMultiscatteringScale, 0.1f) \
  X(bool,  enableTranslucentShadows, false) \
  X(float, atmosphereSunFogScale,    1.0f) \
  X(float, depthOffset,              0.5f) \
  X(float, noiseFieldOctaves,        2.0f) \
  X(float,   volumetricAnisotropy,                      0.0f)

// sandstorm — yellow-orange forward-scattering fog
#define WEATHER_PRESET_VALUES_sandstorm(X)                                                             \
  X(float,   cloudDensity,                              1.5f)                                          \
  X(float,   cloudCoverageMean,                         0.40f)                                         \
  X(float,   cloudCoverageSpread,                       0.30f)                                         \
  X(float,   cloudCoverageNoiseScale,                   0.0033f)                                       \
  X(float,   cloudTypeMean,                             0.2f)                                          \
  X(float,   cloudTypeSpread,                           0.4f)                                          \
  X(float,   cloudTypeNoiseScale,                       0.0034f)                                       \
  X(Vector3, cloudColor,                                Vector3(0.85f, 0.65f, 0.40f))                  \
  X(float,   cloudWindSpeed,                            0.02f)                                         \
  X(float,   cloudWindDirection,                        45.0f)                                         \
  X(float,   cloudShadowStrength,                       0.20f)                                         \
  X(float,   cloudThickness,                            2.5f)                                          \
  X(float, cloudUndersideLightSigma, 0.12f) \
  X(float, cloudBottomDarkening,     1.0f) \
  X(float, cloudAerialFadePerKm,     0.15f) \
  X(float, cloudAerialHazePerKm,     0.05f) \
  X(Vector3, rayleighScattering, Vector3(5.8e-3f, 13.5e-3f, 33.1e-3f)) \
  X(Vector3, nightSkyColor,      Vector3(0.15f, 0.2f, 0.4f)) \
  X(float, skyIndirectRadianceScale, 1.0f) \
  X(float,   airDensity,                                1.0f)                                          \
  X(float,   aerosolDensity,                            2.5f)                                          \
  X(Vector3, sunIlluminance,                            Vector3(10.0f, 8.0f, 5.0f))                    \
  X(float,   nightSkyBrightness,                        0.010f)                                        \
  X(float,   moonNeeStrength,                           1.0f)                                          \
  X(float,   moonAtmosphericCouplingStrength,           1.0f)                                          \
  X(Vector3, transmittanceColor,                        Vector3(0.95f, 0.65f, 0.35f))                  \
  X(float,   transmittanceMeasurementDistanceMeters,    50.0f)                                         \
  X(Vector3, singleScatteringAlbedo,                    Vector3(0.90f, 0.75f, 0.50f))                  \
  X(float, fogSunVisibilityGain, 1.0f) \
  X(float, volumetricConsumerGain, 0.008f) \
  X(bool, enableHeterogeneousFog, false) \
  X(float, noiseFieldDensityScale, 1.0f) \
  X(float, noiseFieldDensityExponent, 2.0f) \
  X(float, noiseFieldInitialFrequencyPerMeter, 8.0f) \
  X(float, noiseFieldLacunarity, 2.0f) \
  X(float, noiseFieldGain, 0.5f) \
  X(float, noiseFieldTimeScale, 0.5f) \
  X(float, noiseFieldSubStepSizeMeters, 10.0f) \
  X(float, froxelMaxDistanceMeters, 20.0f) \
  X(bool, enableFogRemap, false) \
  X(bool, enableFogColorRemap, false) \
  X(bool, enableFogMaxDistanceRemap, true) \
  X(float, fogRemapMaxDistanceMinMeters, 1.0f) \
  X(float, fogRemapMaxDistanceMaxMeters, 40.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMinMeters, 20.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMaxMeters, 100.0f) \
  X(float, fogRemapColorMultiscatteringScale, 0.1f) \
  X(bool,  enableTranslucentShadows, false) \
  X(float, atmosphereSunFogScale,    1.0f) \
  X(float, depthOffset,              0.5f) \
  X(float, noiseFieldOctaves,        2.0f) \
  X(float,   volumetricAnisotropy,                      0.60f)

// smoggy — industrial dark grey-brown haze
#define WEATHER_PRESET_VALUES_smoggy(X)                                                                \
  X(float,   cloudDensity,                              1.4f)                                          \
  X(float,   cloudCoverageMean,                         0.45f)                                         \
  X(float,   cloudCoverageSpread,                       0.20f)                                         \
  X(float,   cloudCoverageNoiseScale,                   0.0033f)                                       \
  X(float,   cloudTypeMean,                             0.3f)                                          \
  X(float,   cloudTypeSpread,                           0.3f)                                          \
  X(float,   cloudTypeNoiseScale,                       0.0034f)                                       \
  X(Vector3, cloudColor,                                Vector3(0.65f, 0.58f, 0.45f))                  \
  X(float,   cloudWindSpeed,                            0.02f)                                         \
  X(float,   cloudWindDirection,                        45.0f)                                         \
  X(float,   cloudShadowStrength,                       0.15f)                                         \
  X(float,   cloudThickness,                            2.5f)                                          \
  X(float, cloudUndersideLightSigma, 0.12f) \
  X(float, cloudBottomDarkening,     1.0f) \
  X(float, cloudAerialFadePerKm,     0.15f) \
  X(float, cloudAerialHazePerKm,     0.05f) \
  X(Vector3, rayleighScattering, Vector3(5.8e-3f, 13.5e-3f, 33.1e-3f)) \
  X(Vector3, nightSkyColor,      Vector3(0.15f, 0.2f, 0.4f)) \
  X(float, skyIndirectRadianceScale, 1.0f) \
  X(float,   airDensity,                                1.1f)                                          \
  X(float,   aerosolDensity,                            1.8f)                                          \
  X(Vector3, sunIlluminance,                            Vector3(12.0f, 10.0f, 8.0f))                   \
  X(float,   nightSkyBrightness,                        0.010f)                                        \
  X(float,   moonNeeStrength,                           1.0f)                                          \
  X(float,   moonAtmosphericCouplingStrength,           1.0f)                                          \
  X(Vector3, transmittanceColor,                        Vector3(0.70f, 0.65f, 0.55f))                  \
  X(float,   transmittanceMeasurementDistanceMeters,    200.0f)                                        \
  X(Vector3, singleScatteringAlbedo,                    Vector3(0.85f, 0.80f, 0.70f))                  \
  X(float, fogSunVisibilityGain, 1.0f) \
  X(float, volumetricConsumerGain, 0.008f) \
  X(bool, enableHeterogeneousFog, false) \
  X(float, noiseFieldDensityScale, 1.0f) \
  X(float, noiseFieldDensityExponent, 2.0f) \
  X(float, noiseFieldInitialFrequencyPerMeter, 8.0f) \
  X(float, noiseFieldLacunarity, 2.0f) \
  X(float, noiseFieldGain, 0.5f) \
  X(float, noiseFieldTimeScale, 0.5f) \
  X(float, noiseFieldSubStepSizeMeters, 10.0f) \
  X(float, froxelMaxDistanceMeters, 20.0f) \
  X(bool, enableFogRemap, false) \
  X(bool, enableFogColorRemap, false) \
  X(bool, enableFogMaxDistanceRemap, true) \
  X(float, fogRemapMaxDistanceMinMeters, 1.0f) \
  X(float, fogRemapMaxDistanceMaxMeters, 40.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMinMeters, 20.0f) \
  X(float, fogRemapTransmittanceMeasurementDistanceMaxMeters, 100.0f) \
  X(float, fogRemapColorMultiscatteringScale, 0.1f) \
  X(bool,  enableTranslucentShadows, false) \
  X(float, atmosphereSunFogScale,    1.0f) \
  X(float, depthOffset,              0.5f) \
  X(float, noiseFieldOctaves,        2.0f) \
  X(float,   volumetricAnisotropy,                      0.20f)

// ---------------------------------------------------------------------------
// Single-preset macro. Walks WEATHER_PRESET_VALUES_<N> via the binder for
// preset N, emitting all 52 RTX_OPTION declarations with archetype-tuned
// defaults. Must be invoked inside a class body (RTX_OPTION declares inline
// static members).
// ---------------------------------------------------------------------------
#define DECLARE_WEATHER_PRESET(N) WEATHER_PRESET_VALUES_##N(WEATHER_PRESET_BIND_##N)

// ---------------------------------------------------------------------------
// Umbrella macro. Invoke inside RtxOptions struct body to declare all 624
// RTX_OPTIONs (12 presets x 52 fields).
// ---------------------------------------------------------------------------
#define DECLARE_ALL_WEATHER_PRESETS()   \
  DECLARE_WEATHER_PRESET(clear)         \
  DECLARE_WEATHER_PRESET(partlyCloudy)  \
  DECLARE_WEATHER_PRESET(overcast)      \
  DECLARE_WEATHER_PRESET(hazy)          \
  DECLARE_WEATHER_PRESET(foggy)         \
  DECLARE_WEATHER_PRESET(drizzle)       \
  DECLARE_WEATHER_PRESET(rainstorm)     \
  DECLARE_WEATHER_PRESET(thunderstorm)  \
  DECLARE_WEATHER_PRESET(snow)          \
  DECLARE_WEATHER_PRESET(blizzard)      \
  DECLARE_WEATHER_PRESET(sandstorm)     \
  DECLARE_WEATHER_PRESET(smoggy)

// ---------------------------------------------------------------------------
// WeatherSnapshot + WeatherBlender — Task 2 additions.
// Lives in dxvk::fork_weather namespace. Included by rtx_fork_weather.cpp;
// forward-use in rtx_fork_hooks.h needs only the hook forward declarations
// (no WeatherBlender include required there).
// ---------------------------------------------------------------------------
#include <string>

namespace dxvk { namespace fork_weather {

  // -------------------------------------------------------------------------
  // WeatherSnapshot — a plain-value copy of all 52 renderer weather params.
  // Members are auto-generated from the single-source-of-truth X-macro so
  // that any field addition automatically propagates here.
  // -------------------------------------------------------------------------
  struct WeatherSnapshot {
#define WEATHER_PRESET_FIELD_AS_MEMBER_(type, name, defaultValue, kind, group, section, label, mn, mx, step, fmt) type name = defaultValue;
    WEATHER_PRESET_FIELD_LIST(WEATHER_PRESET_FIELD_AS_MEMBER_)
#undef WEATHER_PRESET_FIELD_AS_MEMBER_
  };

  // -------------------------------------------------------------------------
  // WeatherBlender — per-frame lerp pipeline.
  //
  // Reads __weather.target + __weather.blend_seconds from the GameStateStore,
  // lerps from m_previousSnapshot toward the named preset's RTX_OPTION values
  // over m_blendDurationSec seconds, and writes interpolated values into the
  // Derived layer of each underlying RTX_OPTION via setImmediately().
  //
  // Dormant when __weather.target is absent or unknown — zero upstream
  // behavioural change.
  //
  // Caller (Task 3) provides deltaTimeSeconds from the per-frame render loop.
  // ImGui surface (Task 4) implemented in showImguiSettings().
  // -------------------------------------------------------------------------
  class WeatherBlender {
  public:
    WeatherBlender();
    ~WeatherBlender();

    // Called once per frame from fork_hooks::updateWeatherBlender (Task 3).
    void update(float deltaTimeSeconds);

    // Renders the inline weather-preset panel (transition controls, status, and
    // the button that toggles the pop-out editor window).
    void showImguiSettings();

    // Renders the pop-out preset editor as a separate movable window (toggled
    // from showImguiSettings). No-op while closed. Call once per frame.
    void renderEditorWindow();

    bool isPaused() const { return m_paused; }
    void setPaused(bool paused) { m_paused = paused; }

  private:
    // Preset cache — empty string means "not yet active".
    std::string m_previousPresetName;
    std::string m_targetPresetName;

    // Blend timeline. Wall-clock accumulators are double so sub-frame precision
    // survives multi-hour sessions (a 32-bit float erodes to ~4 ms after ~9 h).
    double m_blendStartTimeSec = 0.0;
    float  m_blendDurationSec  = 1.0f;
    double m_currentTimeSec    = 0.0;

    bool m_paused = false;

    // Pop-out preset editor window visibility (toggled from showImguiSettings).
    bool m_editorWindowOpen = false;

    // "Pin & Freeze for Tuning" toggle state + the drift intensity to restore on
    // un-pin (so freezing variation for tuning is non-destructive).
    bool  m_pinnedForTuning   = false;
    float m_savedDriftIntensity = 1.0f;

    // Drift state (cloud-drift modulation; spec 2026-05-09-cloud-drift-design).
    // m_driftPhaseSeconds is monotonically advanced each frame by
    // dt * m_driftSpeedSmoothed. Smoothed values are one-pole filtered toward
    // the GameStateStore-supplied raw values with tau = 1.0s.
    double m_driftPhaseSeconds     = 0.0;
    float m_driftSpeedSmoothed     = 1.0f;
    float m_driftIntensitySmoothed = 1.0f;

    // Snapshot of renderer state at the moment the last blend began (or the
    // retarget mid-blend captured the partially-blended state).
    WeatherSnapshot m_previousSnapshot;

    // Writes interpolated snapshot values to the Derived RTX_OPTION layer.
    void applyBlendedValues(float t);

    // Returns a WeatherSnapshot populated from the current renderer RTX_OPTION
    // getters (not from any preset table). Used at first-activation to seed
    // m_previousSnapshot so the initial blend transitions smoothly from
    // whatever the renderer was already doing.
    WeatherSnapshot snapshotCurrentValues() const;

    // Writes blend progress state back to the GameStateStore.
    void publishStateToGameStateStore(float t) const;
  };

} }  // namespace dxvk::fork_weather
