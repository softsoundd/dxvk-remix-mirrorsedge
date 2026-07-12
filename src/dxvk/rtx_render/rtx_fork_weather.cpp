// src/dxvk/rtx_render/rtx_fork_weather.cpp
//
// Fork-owned file. Full implementation of WeatherBlender: the per-frame lerp
// pipeline that blends 52 weather params (cloud, atmosphere, sky/moon mood,
// volumetric fog) between named presets over a plugin-specified duration.
//
// Reads:
//   __weather.target       — name of the active target preset (string)
//   __weather.blend_seconds — blend duration override (float string, default 1.0)
//
// Writes:
//   Derived layer of each underlying RTX_OPTION via setImmediately()
//   __weather.current, __weather.previous, __weather.blend_progress (GameStateStore)
//
// Dormant when __weather.target is absent or unknown — zero upstream
// behavioural change.
//
// Task 3 wires update() into the per-frame render loop via
// fork_hooks::updateWeatherBlender(RtxContext&, float).
// Task 4 implements the full ImGui surface in showImguiSettings().
// Task 7 handles the upstream touchpoint update (rtx_fork_hooks.h comment).

#include "rtx_fork_weather.h"
#include "rtx_fork_hooks.h"
#include "rtx_context.h"
#include "rtx_fork_game_state.h"
#include "rtx_options.h"
#include "rtx_global_volumetrics.h"
#include "imgui/imgui.h"
#include "rtx_imgui.h"               // RemixGui::DragFloat, DragFloat3, SetTooltipToLastWidgetOnHover
#include "../../util/log/log.h"     // Logger::warn for unknown-preset diagnostic
#include "../../util/util_string.h" // str::format

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_set>

// ---------------------------------------------------------------------------
// Anonymous-namespace helpers
// ---------------------------------------------------------------------------
namespace dxvk { namespace fork_weather { namespace {

  // --- Active blender singleton ---
  // Set by WeatherBlender ctor, cleared by dtor. Only one RtxContext is alive
  // at a time, so at most one WeatherBlender exists during normal operation.
  WeatherBlender* g_activeBlender = nullptr;
  // Forward decl (defined later in this anonymous namespace); used by the
  // snapshot-from-live authoring helper.
  WeatherSnapshot snapshotRenderer();

  // --- Math helpers ---

  float saturate(float x) {
    return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
  }

  float lerp(float a, float b, float t) {
    return a + (b - a) * t;
  }

  Vector3 lerpV3(const Vector3& a, const Vector3& b, float t) {
    return Vector3(
      lerp(a.x, b.x, t),
      lerp(a.y, b.y, t),
      lerp(a.z, b.z, t)
    );
  }

  // Shortest-path angular interpolation (degrees).
  // 350° → 10°: delta = fmod(10-350+540, 360)-180 = fmod(200,360)-180 = 200-180 = 20°.
  float lerpAngleDeg(float a, float b, float t) {
    float delta = std::fmod((b - a + 540.0f), 360.0f) - 180.0f;
    return a + delta * t;
  }

  // Lerp optical extinction (~1/distance) rather than distance, so fog density
  // ramps perceptually even between presets (linear-in-distance is heavily
  // back-loaded). Clamped to avoid div-by-zero at the bright end.
  float lerpExtinction(float a, float b, float t) {
    const float ea = 1.0f / std::max(a, 1e-4f);
    const float eb = 1.0f / std::max(b, 1e-4f);
    return 1.0f / std::max(lerp(ea, eb, t), 1e-4f);
  }

  // Kind-aware scalar lerp: WK_Angle wraps shortest-path, WK_Extinction lerps in
  // 1/distance space, everything else is plain linear.
  float lerpField(float a, float b, float t, WeatherFieldKind kind) {
    switch (kind) {
      case WK_Angle:      return lerpAngleDeg(a, b, t);
      case WK_Extinction: return lerpExtinction(a, b, t);
      default:            return lerp(a, b, t);
    }
  }
  // Vector3 fields (WK_Color / WK_Vec3) lerp componentwise.
  Vector3 lerpField(const Vector3& a, const Vector3& b, float t, WeatherFieldKind) {
    return lerpV3(a, b, t);
  }
  // Bool fields (WK_Step): not interpolable; switch at the blend midpoint.
  bool lerpField(bool a, bool b, float t, WeatherFieldKind) {
    return (t >= 0.5f) ? b : a;
  }

  // Per-field lerp from one snapshot to another at parameter t, driven entirely
  // by WEATHER_PRESET_FIELD_LIST + the per-field kind. No hand-listed fields, so
  // a field added to the table is interpolated automatically.
  WeatherSnapshot lerpSnapshot(const WeatherSnapshot& a, const WeatherSnapshot& b, float t) {
    WeatherSnapshot out;
#define WEATHER_LERP_FIELD(type, name, def, kind, grp, sec, lbl, mn, mx, st, fmt) \
    out.name = lerpField(a.name, b.name, t, kind);
    WEATHER_PRESET_FIELD_LIST(WEATHER_LERP_FIELD)
#undef WEATHER_LERP_FIELD
    return out;
  }

  // ---------------------------------------------------------------------------
  // Drift math — sum of incommensurate sines. Cheap, deterministic, smooth.
  //
  // driftNoise1D returns approximately [-1, 1] for any phase. Three inner
  // periods (1.0, 1.527, 0.701) chosen so the sum doesn't repeat for many
  // hours of phase advance.
  //
  // De-pulsed (fork — 2026-06-21): the old two-layer model had a fast layer with
  // base period 30 s, whose dominant inner sine (weight 0.5) was exactly 30 s.
  // That produced a clearly perceptible whole-sky "breathing" beat every ~30 s —
  // all drift fields share one phase clock, so coverage/density/type all crested
  // in lockstep. The fast layer is GONE; only the slow (multi-minute) layer
  // remains, so this subsystem now reads as genuine slow weather change rather
  // than a rhythm. Local cloud SHAPE change (formation/dissolution, edge boil) is
  // now owned by the field-evolution system in the cloud taps
  // (cloudEvolutionSpeed / cloudBoilSpeed), which evolves the field spatially
  // instead of pulsing a global scalar.
  // ---------------------------------------------------------------------------

  constexpr float kDriftSlowPeriodSec = 300.0f;

  float driftNoise1D(double phaseSeconds, float periodSeconds, float fieldSeed) {
    constexpr double kTwoPi = 6.28318530718;
    const double p = phaseSeconds / static_cast<double>(periodSeconds);
    return static_cast<float>(
        0.50 * std::sin(kTwoPi * (p / 1.000) + fieldSeed * 1.000)
      + 0.30 * std::sin(kTwoPi * (p / 1.527) + fieldSeed * 1.731)
      + 0.20 * std::sin(kTwoPi * (p / 0.701) + fieldSeed * 2.331));
  }

  // Per-field slow drift offset, normalized to ~[-relativeAmp, +relativeAmp].
  // Slow-layer only (the fast 30 s layer was removed — see note above). The slow
  // layer's shortest inner period is ~210 s (3.5 min), so there is no short-cycle
  // tell. fieldIndex still seeds the phase so the few remaining fields stay
  // decorrelated from each other.
  float driftOffsetForField(int fieldIndex, double phaseSeconds, float relativeAmp) {
    constexpr float kFieldSeedStep = 0.6180f;  // golden-ratio-ish for low correlation
    const float seedSlow = static_cast<float>(fieldIndex) * kFieldSeedStep + 100.0f;
    const float nSlow = driftNoise1D(phaseSeconds, kDriftSlowPeriodSec, seedSlow);
    return nSlow * relativeAmp;
  }

  // ---------------------------------------------------------------------------
  // Drift field table — weather-SCALE fields only (fork — 2026-06-21).
  //
  // Trimmed from 9 to 3. The shape-ish fields (cloudDensity, cloudThickness,
  // cloudTypeMean/Spread, cloudCoverageSpread, cloudAnvilBias) were removed:
  // drifting them as a GLOBAL scalar is exactly the artificial "whole-sky
  // breathing" the field-evolution rework replaced — those shape changes are now
  // produced locally and incoherently by cloudEvolutionSpeed / cloudBoilSpeed in
  // the cloud taps. What remains is the genuinely weather-scale stuff the field
  // evolution does NOT reproduce: how cloudy the sky is overall (cloudCoverageMean)
  // and how the wind gusts/shifts (cloudWindSpeed / cloudWindDirection).
  //
  // Color, optical, sky/moon, atmosphere, volumetric, and noise-scale fields
  // remain excluded (drift would look sickly, break calibration, or re-tile the
  // cloud field — see spec section "Drift fields"). fieldIndex values are kept at
  // their original numbers so each field's noise seed is unchanged.
  //
  // amplitudeMode:
  //   Proportional — final delta is delta_table * intensity * field_value
  //                  (relativeAmp interpreted as fraction of midpoint)
  //   AbsoluteDeg  — final delta is delta_table * intensity, applied as
  //                  degrees with modulo-360 wrap (used for cloudWindDirection)
  //
  // clampMin / clampMax: post-modulation clamp. -kInf / +kInf disables a side.
  // ---------------------------------------------------------------------------

  enum class DriftMode { Proportional, AbsoluteDeg };

  struct DriftFieldEntry {
    const char* name;          // diagnostic only
    int         fieldIndex;    // unique per field, drives noise seed
    DriftMode   mode;
    float       relativeAmp;   // proportional: fraction; absolute: degrees
    float       clampMin;
    float       clampMax;
    float (*getter)(const WeatherSnapshot& s);
    void  (*setter)(WeatherSnapshot& s, float v);
  };

  // Per-field accessor pairs (one set per drifting field).
  #define DRIFT_FIELD_ACCESSORS(field) \
    [](const WeatherSnapshot& s) -> float { return s.field; }, \
    [](WeatherSnapshot& s, float v)      { s.field = v; }

  static const float kInf = std::numeric_limits<float>::infinity();

  static const DriftFieldEntry kDriftTable[] = {
    // name                    idx  mode                       relAmp   min     max
    { "cloudCoverageMean",      0,   DriftMode::Proportional,   0.15f,   0.0f,   1.0f,    DRIFT_FIELD_ACCESSORS(cloudCoverageMean)   },
    { "cloudWindSpeed",         6,   DriftMode::Proportional,   0.30f,   0.0f,   kInf,    DRIFT_FIELD_ACCESSORS(cloudWindSpeed)      },
    { "cloudWindDirection",     7,   DriftMode::AbsoluteDeg,   10.0f,   -kInf,  kInf,    DRIFT_FIELD_ACCESSORS(cloudWindDirection)  },
  };

  static constexpr int kDriftFieldCount = static_cast<int>(sizeof(kDriftTable) / sizeof(kDriftTable[0]));
  static_assert(kDriftFieldCount == 3, "Drift table must have exactly 3 weather-scale entries "
                "(de-pulsed 2026-06-21: shape fields moved to field evolution)");

  // ---------------------------------------------------------------------------
  // applyDriftToSnapshot — mutate interp in place by adding per-field drift
  // offsets. intensity scales the entire modulation; intensity == 0 short-
  // circuits and leaves interp untouched.
  // ---------------------------------------------------------------------------
  void applyDriftToSnapshot(WeatherSnapshot& interp, double phaseSeconds, float intensity) {
    if (intensity <= 0.0f) {
      return;
    }

    for (int i = 0; i < kDriftFieldCount; ++i) {
      const DriftFieldEntry& e = kDriftTable[i];
      const float driftRaw = driftOffsetForField(e.fieldIndex, phaseSeconds, e.relativeAmp);
      const float driftScaled = driftRaw * intensity;

      const float v = e.getter(interp);
      float vOut;
      switch (e.mode) {
        case DriftMode::Proportional:
          vOut = v + driftScaled * v;
          break;
        case DriftMode::AbsoluteDeg: {
          float w = std::fmod(v + driftScaled, 360.0f);
          if (w < 0.0f) w += 360.0f;
          vOut = w;
          break;
        }
        default:
          vOut = v;
          break;
      }

      // Clamp (no-op when both ends are +/-kInf).
      if (vOut < e.clampMin) vOut = e.clampMin;
      if (vOut > e.clampMax) vOut = e.clampMax;

      e.setter(interp, vOut);
    }
  }

  // --- GameStateStore wrappers ---

  float readFloatFromGameStateStore(const std::string& key, float defaultValue) {
    std::string raw;
    if (!fork_game_state::GameStateStore::get().tryGet(key, raw)) {
      return defaultValue;
    }
    try {
      return std::stof(raw);
    } catch (...) {
      return defaultValue;
    }
  }

  std::string readStringFromGameStateStore(const std::string& key) {
    std::string out;
    fork_game_state::GameStateStore::get().tryGet(key, out);
    return out;
  }

  void writeToGameStateStore(const std::string& key, std::string value) {
    fork_game_state::GameStateStore::get().set(key, std::move(value));
  }

  // ---------------------------------------------------------------------------
  // Preset table machinery (generated from WEATHER_PRESET_FIELD_LIST). Collapses
  // the former ~300-line readPresetValues + isKnownPresetName string cascade and
  // also drives the generated ImGui panel.
  // ---------------------------------------------------------------------------

  enum WeatherPresetIdx {
    WP_clear, WP_partlyCloudy, WP_overcast, WP_hazy, WP_foggy, WP_drizzle,
    WP_rainstorm, WP_thunderstorm, WP_snow, WP_blizzard, WP_sandstorm, WP_smoggy,
    WP_COUNT
  };

  // Type- and kind-dispatched widget helpers, matching the main panel's design
  // language: float -> DragFloat, bool -> Checkbox, Vector3 -> ColorEdit3 swatch
  // for WK_Color (click to open a picker) else DragFloat3 for radiometric vectors
  // (e.g. sun illuminance, which carries magnitude, not a 0-1 color).
  bool weatherDrag(const char* l, RtxOption<float>* o, float st, float mn, float mx, const char* fmt, ImGuiSliderFlags fl, WeatherFieldKind kind) {
    // Display-transform kinds (fork - 2026-07-02, UI usability): the option
    // keeps its canonical storage unit (conf/API/blend unchanged); only the
    // widget converts. Table min/max/step/fmt are in DISPLAY units for these.
    // Pattern mirrors RemixGui::DragFloatMB_showGB (rtx_imgui.h).
    if (kind == WK_SpeedKmS) {
      // Stored km/s, displayed m/s. 0.02 km/s reads as a draggable "20.0 m/s"
      // instead of a three-decimal crawl.
      RemixGui::RtxOptionUxWrapper wrapper(o);
      float valueMs = o->get() * 1000.0f;
      const bool changed = RemixGui::DragFloat(l, &valueMs, st, mn, mx, fmt, fl);
      if (changed) {
        RemixGui::CheckRtxOptionPopups(o);
        o->setDeferred(valueMs * 0.001f);
      }
      return changed;
    }
    if (kind == WK_PatchPerKm) {
      // Stored spatial frequency (1/km), displayed as the patch wavelength in
      // km (1/x) - the label says "Patch Size", so the number now IS a size:
      // bigger km = bigger patches. Guards keep 1/x finite for zeroed confs.
      RemixGui::RtxOptionUxWrapper wrapper(o);
      float valueKm = 1.0f / std::max(o->get(), 1e-6f);
      const bool changed = RemixGui::DragFloat(l, &valueKm, st, mn, mx, fmt, fl);
      if (changed) {
        RemixGui::CheckRtxOptionPopups(o);
        o->setDeferred(1.0f / std::max(valueKm, 1.0f));
      }
      return changed;
    }
    return RemixGui::DragFloat(l, o, st, mn, mx, fmt, fl);
  }
  bool weatherDrag(const char* l, RtxOption<Vector3>* o, float st, float mn, float mx, const char* fmt, ImGuiSliderFlags fl, WeatherFieldKind kind) {
    if (kind == WK_Color) {
      // HDR/float picker for radiometric values that exceed 1 (e.g. sun
      // illuminance); float-entry picker for sub-unit coefficient colors like the
      // sky scattering tint (~0.005-0.05, where an 8-bit 0-255 swatch is useless);
      // plain 0-1 swatch for normal colors (cloud / sky / night tints).
      ImGuiColorEditFlags cflags = 0;
      if (mx > 1.5f)      cflags = ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR;
      else if (mx < 1.0f) cflags = ImGuiColorEditFlags_Float;
      return RemixGui::ColorEdit3(l, o, cflags);
    }
    return RemixGui::DragFloat3(l, o, st, mn, mx, fmt, fl);
  }
  bool weatherDrag(const char* l, RtxOption<bool>* o, float, float, float, const char*, ImGuiSliderFlags, WeatherFieldKind) {
    return RemixGui::Checkbox(l, o);  // numeric/format args ignored for bool fields
  }

  // weatherRenderSlider_<field>(presetIdx, flags): renders this field's slider
  // bound to RtxOptions::<preset>_<field>Object(), range/format baked from the
  // field table. The 12 preset cases are written once; the field table generates
  // one such function per field.
#define WEATHER_RENDER_SLIDER_FN(type, name, def, kind, grp, sec, lbl, mn, mx, st, fmt)                               \
  void weatherRenderSlider_##name(int p, ImGuiSliderFlags fl) {                                                       \
    switch (p) {                                                                                                      \
      case WP_clear:        weatherDrag(lbl, &RtxOptions::clear_##name##Object(),        st, mn, mx, fmt, fl, kind); break; \
      case WP_partlyCloudy: weatherDrag(lbl, &RtxOptions::partlyCloudy_##name##Object(), st, mn, mx, fmt, fl, kind); break; \
      case WP_overcast:     weatherDrag(lbl, &RtxOptions::overcast_##name##Object(),     st, mn, mx, fmt, fl, kind); break; \
      case WP_hazy:         weatherDrag(lbl, &RtxOptions::hazy_##name##Object(),         st, mn, mx, fmt, fl, kind); break; \
      case WP_foggy:        weatherDrag(lbl, &RtxOptions::foggy_##name##Object(),        st, mn, mx, fmt, fl, kind); break; \
      case WP_drizzle:      weatherDrag(lbl, &RtxOptions::drizzle_##name##Object(),      st, mn, mx, fmt, fl, kind); break; \
      case WP_rainstorm:    weatherDrag(lbl, &RtxOptions::rainstorm_##name##Object(),    st, mn, mx, fmt, fl, kind); break; \
      case WP_thunderstorm: weatherDrag(lbl, &RtxOptions::thunderstorm_##name##Object(), st, mn, mx, fmt, fl, kind); break; \
      case WP_snow:         weatherDrag(lbl, &RtxOptions::snow_##name##Object(),         st, mn, mx, fmt, fl, kind); break; \
      case WP_blizzard:     weatherDrag(lbl, &RtxOptions::blizzard_##name##Object(),     st, mn, mx, fmt, fl, kind); break; \
      case WP_sandstorm:    weatherDrag(lbl, &RtxOptions::sandstorm_##name##Object(),    st, mn, mx, fmt, fl, kind); break; \
      case WP_smoggy:       weatherDrag(lbl, &RtxOptions::smoggy_##name##Object(),       st, mn, mx, fmt, fl, kind); break; \
      default: break;                                                                                                \
    }                                                                                                                \
  }
  WEATHER_PRESET_FIELD_LIST(WEATHER_RENDER_SLIDER_FN)
#undef WEATHER_RENDER_SLIDER_FN

  // weatherSetPresetField_<field>(presetIdx, snapshot): writes snapshot.<field>
  // into RtxOptions::<preset>_<field>Object() (Derived layer). Drives the
  // copy-from / snapshot-from-live authoring buttons.
#define WEATHER_SET_PRESET_FN(type, name, def, kind, grp, sec, lbl, mn, mx, st, fmt)                 \
  void weatherSetPresetField_##name(int p, const WeatherSnapshot& s) {                               \
    switch (p) {                                                                                     \
      case WP_clear:        RtxOptions::clear_##name##Object().setImmediately(s.name);        break; \
      case WP_partlyCloudy: RtxOptions::partlyCloudy_##name##Object().setImmediately(s.name); break; \
      case WP_overcast:     RtxOptions::overcast_##name##Object().setImmediately(s.name);     break; \
      case WP_hazy:         RtxOptions::hazy_##name##Object().setImmediately(s.name);         break; \
      case WP_foggy:        RtxOptions::foggy_##name##Object().setImmediately(s.name);        break; \
      case WP_drizzle:      RtxOptions::drizzle_##name##Object().setImmediately(s.name);      break; \
      case WP_rainstorm:    RtxOptions::rainstorm_##name##Object().setImmediately(s.name);    break; \
      case WP_thunderstorm: RtxOptions::thunderstorm_##name##Object().setImmediately(s.name); break; \
      case WP_snow:         RtxOptions::snow_##name##Object().setImmediately(s.name);         break; \
      case WP_blizzard:     RtxOptions::blizzard_##name##Object().setImmediately(s.name);     break; \
      case WP_sandstorm:    RtxOptions::sandstorm_##name##Object().setImmediately(s.name);    break; \
      case WP_smoggy:       RtxOptions::smoggy_##name##Object().setImmediately(s.name);       break; \
      default: break;                                                                                \
    }                                                                                                \
  }
  WEATHER_PRESET_FIELD_LIST(WEATHER_SET_PRESET_FN)
#undef WEATHER_SET_PRESET_FN

  // Per-field conf-line value formatter (type-dispatched on the snapshot member).
  std::string weatherFmtConf(float v)          { char b[48]; std::snprintf(b, sizeof(b), "%.4f", v); return b; }
  std::string weatherFmtConf(bool v)           { return v ? "True" : "False"; }
  std::string weatherFmtConf(const Vector3& v) { char b[96]; std::snprintf(b, sizeof(b), "%.4f, %.4f, %.4f", v.x, v.y, v.z); return b; }
#define WEATHER_CONF_FN(type, name, def, kind, grp, sec, lbl, mn, mx, st, fmt) \
  std::string weatherConf_##name(const WeatherSnapshot& s) { return weatherFmtConf(s.name); }
  WEATHER_PRESET_FIELD_LIST(WEATHER_CONF_FN)
#undef WEATHER_CONF_FN
  // Per-field descriptor consumed by the generated ImGui panel.
  typedef void (*WeatherSliderFn)(int presetIdx, ImGuiSliderFlags flags);
  struct WeatherFieldDesc {
    const char*      name;
    WeatherFieldKind kind;
    const char*      group;
    const char*      section;
    const char*      label;
    WeatherSliderFn  renderSlider;
    void (*setPresetField)(int presetIdx, const WeatherSnapshot& s);
    std::string (*formatValue)(const WeatherSnapshot& s);
  };
#define WEATHER_FIELD_DESC(type, name, def, kind, grp, sec, lbl, mn, mx, st, fmt) \
  { #name, kind, grp, sec, lbl, &weatherRenderSlider_##name, &weatherSetPresetField_##name, &weatherConf_##name },
  const WeatherFieldDesc kFieldDescs[] = { WEATHER_PRESET_FIELD_LIST(WEATHER_FIELD_DESC) };
#undef WEATHER_FIELD_DESC
  constexpr int kFieldCount = static_cast<int>(sizeof(kFieldDescs) / sizeof(kFieldDescs[0]));

  // Per-preset readers: fill a WeatherSnapshot from RtxOptions::<preset>_<field>().
  // One function per preset, each generated from the field table; the only
  // per-preset literal is the option-name prefix.
#define WRF(type, name, def, kind, grp, sec, lbl, mn, mx, st, fmt) s.name = RtxOptions::clear_##name();
  WeatherSnapshot readPreset_clear()        { WeatherSnapshot s; WEATHER_PRESET_FIELD_LIST(WRF) return s; }
#undef WRF
#define WRF(type, name, def, kind, grp, sec, lbl, mn, mx, st, fmt) s.name = RtxOptions::partlyCloudy_##name();
  WeatherSnapshot readPreset_partlyCloudy() { WeatherSnapshot s; WEATHER_PRESET_FIELD_LIST(WRF) return s; }
#undef WRF
#define WRF(type, name, def, kind, grp, sec, lbl, mn, mx, st, fmt) s.name = RtxOptions::overcast_##name();
  WeatherSnapshot readPreset_overcast()     { WeatherSnapshot s; WEATHER_PRESET_FIELD_LIST(WRF) return s; }
#undef WRF
#define WRF(type, name, def, kind, grp, sec, lbl, mn, mx, st, fmt) s.name = RtxOptions::hazy_##name();
  WeatherSnapshot readPreset_hazy()         { WeatherSnapshot s; WEATHER_PRESET_FIELD_LIST(WRF) return s; }
#undef WRF
#define WRF(type, name, def, kind, grp, sec, lbl, mn, mx, st, fmt) s.name = RtxOptions::foggy_##name();
  WeatherSnapshot readPreset_foggy()        { WeatherSnapshot s; WEATHER_PRESET_FIELD_LIST(WRF) return s; }
#undef WRF
#define WRF(type, name, def, kind, grp, sec, lbl, mn, mx, st, fmt) s.name = RtxOptions::drizzle_##name();
  WeatherSnapshot readPreset_drizzle()      { WeatherSnapshot s; WEATHER_PRESET_FIELD_LIST(WRF) return s; }
#undef WRF
#define WRF(type, name, def, kind, grp, sec, lbl, mn, mx, st, fmt) s.name = RtxOptions::rainstorm_##name();
  WeatherSnapshot readPreset_rainstorm()    { WeatherSnapshot s; WEATHER_PRESET_FIELD_LIST(WRF) return s; }
#undef WRF
#define WRF(type, name, def, kind, grp, sec, lbl, mn, mx, st, fmt) s.name = RtxOptions::thunderstorm_##name();
  WeatherSnapshot readPreset_thunderstorm() { WeatherSnapshot s; WEATHER_PRESET_FIELD_LIST(WRF) return s; }
#undef WRF
#define WRF(type, name, def, kind, grp, sec, lbl, mn, mx, st, fmt) s.name = RtxOptions::snow_##name();
  WeatherSnapshot readPreset_snow()         { WeatherSnapshot s; WEATHER_PRESET_FIELD_LIST(WRF) return s; }
#undef WRF
#define WRF(type, name, def, kind, grp, sec, lbl, mn, mx, st, fmt) s.name = RtxOptions::blizzard_##name();
  WeatherSnapshot readPreset_blizzard()     { WeatherSnapshot s; WEATHER_PRESET_FIELD_LIST(WRF) return s; }
#undef WRF
#define WRF(type, name, def, kind, grp, sec, lbl, mn, mx, st, fmt) s.name = RtxOptions::sandstorm_##name();
  WeatherSnapshot readPreset_sandstorm()    { WeatherSnapshot s; WEATHER_PRESET_FIELD_LIST(WRF) return s; }
#undef WRF
#define WRF(type, name, def, kind, grp, sec, lbl, mn, mx, st, fmt) s.name = RtxOptions::smoggy_##name();
  WeatherSnapshot readPreset_smoggy()       { WeatherSnapshot s; WEATHER_PRESET_FIELD_LIST(WRF) return s; }
#undef WRF

  struct WeatherPresetDesc { const char* name; int idx; WeatherSnapshot (*read)(); };
  const WeatherPresetDesc kPresetDescs[] = {
    { "clear",        WP_clear,        &readPreset_clear        },
    { "partlyCloudy", WP_partlyCloudy, &readPreset_partlyCloudy },
    { "overcast",     WP_overcast,     &readPreset_overcast     },
    { "hazy",         WP_hazy,         &readPreset_hazy         },
    { "foggy",        WP_foggy,        &readPreset_foggy        },
    { "drizzle",      WP_drizzle,      &readPreset_drizzle      },
    { "rainstorm",    WP_rainstorm,    &readPreset_rainstorm    },
    { "thunderstorm", WP_thunderstorm, &readPreset_thunderstorm },
    { "snow",         WP_snow,         &readPreset_snow         },
    { "blizzard",     WP_blizzard,     &readPreset_blizzard     },
    { "sandstorm",    WP_sandstorm,    &readPreset_sandstorm    },
    { "smoggy",       WP_smoggy,       &readPreset_smoggy       },
  };
  static_assert(sizeof(kPresetDescs) / sizeof(kPresetDescs[0]) == WP_COUNT,
                "kPresetDescs size must match WP_COUNT");

  bool isKnownPresetName(const std::string& name) {
    for (const auto& p : kPresetDescs) { if (name == p.name) return true; }
    return false;
  }
  bool readPresetValues(const std::string& name, WeatherSnapshot& out) {
    for (const auto& p : kPresetDescs) { if (name == p.name) { out = p.read(); return true; } }
    return false;  // Unknown preset name -> caller treats blender as dormant.
  }
  int presetIndexForName(const std::string& name) {
    for (const auto& p : kPresetDescs) { if (name == p.name) return p.idx; }
    return -1;
  }

  // ---------------------------------------------------------------------------
  // Authoring helpers (copy-from / snapshot-from-live / export-to-conf).
  // ---------------------------------------------------------------------------
  void copyPresetToPreset(int srcIdx, int dstIdx) {
    if (srcIdx < 0 || dstIdx < 0 || srcIdx >= WP_COUNT || dstIdx >= WP_COUNT || srcIdx == dstIdx) return;
    WeatherSnapshot s = kPresetDescs[srcIdx].read();
    for (const auto& d : kFieldDescs) { d.setPresetField(dstIdx, s); }
  }
  void snapshotLiveToPreset(int dstIdx) {
    if (dstIdx < 0 || dstIdx >= WP_COUNT) return;
    WeatherSnapshot s = snapshotRenderer();
    for (const auto& d : kFieldDescs) { d.setPresetField(dstIdx, s); }
  }
  std::string exportPresetToConf(int idx) {
    if (idx < 0 || idx >= WP_COUNT) return std::string();
    WeatherSnapshot s = kPresetDescs[idx].read();
    const char* pname = kPresetDescs[idx].name;
    std::string out;
    for (const auto& d : kFieldDescs) {
      // Full config key = category + "." + option name, and the option name is
      // itself preset-prefixed (e.g. rtx.weather.preset.foggy.foggy_cloudDensity).
      out += "rtx.weather.preset.";
      out += pname; out += "."; out += pname; out += "_"; out += d.name; out += " = ";
      out += d.formatValue(s); out += "\n";
    }
    return out;
  }
  // Case-insensitive substring filter for the panel search box.
  bool matchesFilter(const char* label, const char* filter) {
    if (!filter || !filter[0]) return true;
    std::string l(label), f(filter);
    std::transform(l.begin(), l.end(), l.begin(), [](unsigned char ch){ return (char)std::tolower(ch); });
    std::transform(f.begin(), f.end(), f.begin(), [](unsigned char ch){ return (char)std::tolower(ch); });
    return l.find(f) != std::string::npos;
  }

  // Tooltip text for each weather field, mirrored from the underlying LIVE option's
  // RTX_OPTION description (getDescription) so tooltips match the canonical docs and
  // stay in sync automatically -- no hand-copied strings. (The per-preset copies only
  // carry a generic auto-description, so we read the global option's text instead.)
  const char* weatherFieldTooltip(const char* name) {
    if (std::strcmp(name, "cloudDensity") == 0) return RtxOptions::cloudDensityObject().getDescription();
    if (std::strcmp(name, "cloudCoverageMean") == 0) return RtxOptions::cloudCoverageMeanObject().getDescription();
    if (std::strcmp(name, "cloudCoverageSpread") == 0) return RtxOptions::cloudCoverageSpreadObject().getDescription();
    if (std::strcmp(name, "cloudCoverageNoiseScale") == 0) return RtxOptions::cloudCoverageNoiseScaleObject().getDescription();
    if (std::strcmp(name, "cloudTypeMean") == 0) return RtxOptions::cloudTypeMeanObject().getDescription();
    if (std::strcmp(name, "cloudTypeSpread") == 0) return RtxOptions::cloudTypeSpreadObject().getDescription();
    if (std::strcmp(name, "cloudTypeNoiseScale") == 0) return RtxOptions::cloudTypeNoiseScaleObject().getDescription();
    if (std::strcmp(name, "cloudColor") == 0) return RtxOptions::cloudColorObject().getDescription();
    if (std::strcmp(name, "cloudWindSpeed") == 0) return RtxOptions::cloudWindSpeedObject().getDescription();
    if (std::strcmp(name, "cloudWindDirection") == 0) return RtxOptions::cloudWindDirectionObject().getDescription();
    if (std::strcmp(name, "cloudShadowStrength") == 0) return RtxOptions::cloudShadowStrengthObject().getDescription();
    if (std::strcmp(name, "cloudThickness") == 0) return RtxOptions::cloudThicknessObject().getDescription();
    if (std::strcmp(name, "cloudUndersideLightSigma") == 0) return RtxOptions::cloudUndersideLightSigmaObject().getDescription();
    if (std::strcmp(name, "cloudBottomDarkening") == 0) return RtxOptions::cloudBottomDarkeningObject().getDescription();
    if (std::strcmp(name, "cloudAerialFadePerKm") == 0) return RtxOptions::cloudAerialFadePerKmObject().getDescription();
    if (std::strcmp(name, "cloudAerialHazePerKm") == 0) return RtxOptions::cloudAerialHazePerKmObject().getDescription();
    if (std::strcmp(name, "airDensity") == 0) return RtxOptions::airDensityObject().getDescription();
    if (std::strcmp(name, "aerosolDensity") == 0) return RtxOptions::aerosolDensityObject().getDescription();
    if (std::strcmp(name, "sunIlluminance") == 0) return RtxOptions::sunIlluminanceObject().getDescription();
    if (std::strcmp(name, "rayleighScattering") == 0) return RtxOptions::rayleighScatteringObject().getDescription();
    if (std::strcmp(name, "skyIndirectRadianceScale") == 0) return RtxOptions::skyIndirectRadianceScaleObject().getDescription();
    if (std::strcmp(name, "nightSkyBrightness") == 0) return RtxOptions::nightSkyBrightnessObject().getDescription();
    if (std::strcmp(name, "nightSkyColor") == 0) return RtxOptions::nightSkyColorObject().getDescription();
    if (std::strcmp(name, "moonNeeStrength") == 0) return RtxOptions::moonNeeStrengthObject().getDescription();
    if (std::strcmp(name, "moonAtmosphericCouplingStrength") == 0) return RtxOptions::moonAtmosphericCouplingStrengthObject().getDescription();
    if (std::strcmp(name, "transmittanceColor") == 0) return RtxGlobalVolumetrics::transmittanceColorObject().getDescription();
    if (std::strcmp(name, "transmittanceMeasurementDistanceMeters") == 0) return RtxGlobalVolumetrics::transmittanceMeasurementDistanceMetersObject().getDescription();
    if (std::strcmp(name, "singleScatteringAlbedo") == 0) return RtxGlobalVolumetrics::singleScatteringAlbedoObject().getDescription();
    if (std::strcmp(name, "volumetricAnisotropy") == 0) return RtxGlobalVolumetrics::anisotropyObject().getDescription();
    if (std::strcmp(name, "fogSunVisibilityGain") == 0) return RtxGlobalVolumetrics::fogSunVisibilityGainObject().getDescription();
    if (std::strcmp(name, "volumetricConsumerGain") == 0) return RtxGlobalVolumetrics::volumetricConsumerGainObject().getDescription();
    if (std::strcmp(name, "enableHeterogeneousFog") == 0) return RtxGlobalVolumetrics::enableHeterogeneousFogObject().getDescription();
    if (std::strcmp(name, "noiseFieldDensityScale") == 0) return RtxGlobalVolumetrics::noiseFieldDensityScaleObject().getDescription();
    if (std::strcmp(name, "noiseFieldDensityExponent") == 0) return RtxGlobalVolumetrics::noiseFieldDensityExponentObject().getDescription();
    if (std::strcmp(name, "noiseFieldInitialFrequencyPerMeter") == 0) return RtxGlobalVolumetrics::noiseFieldInitialFrequencyPerMeterObject().getDescription();
    if (std::strcmp(name, "noiseFieldLacunarity") == 0) return RtxGlobalVolumetrics::noiseFieldLacunarityObject().getDescription();
    if (std::strcmp(name, "noiseFieldGain") == 0) return RtxGlobalVolumetrics::noiseFieldGainObject().getDescription();
    if (std::strcmp(name, "noiseFieldTimeScale") == 0) return RtxGlobalVolumetrics::noiseFieldTimeScaleObject().getDescription();
    if (std::strcmp(name, "noiseFieldSubStepSizeMeters") == 0) return RtxGlobalVolumetrics::noiseFieldSubStepSizeMetersObject().getDescription();
    if (std::strcmp(name, "froxelMaxDistanceMeters") == 0) return RtxGlobalVolumetrics::froxelMaxDistanceMetersObject().getDescription();
    if (std::strcmp(name, "enableFogRemap") == 0) return RtxGlobalVolumetrics::enableFogRemapObject().getDescription();
    if (std::strcmp(name, "enableFogColorRemap") == 0) return RtxGlobalVolumetrics::enableFogColorRemapObject().getDescription();
    if (std::strcmp(name, "enableFogMaxDistanceRemap") == 0) return RtxGlobalVolumetrics::enableFogMaxDistanceRemapObject().getDescription();
    if (std::strcmp(name, "fogRemapMaxDistanceMinMeters") == 0) return RtxGlobalVolumetrics::fogRemapMaxDistanceMinMetersObject().getDescription();
    if (std::strcmp(name, "fogRemapMaxDistanceMaxMeters") == 0) return RtxGlobalVolumetrics::fogRemapMaxDistanceMaxMetersObject().getDescription();
    if (std::strcmp(name, "fogRemapTransmittanceMeasurementDistanceMinMeters") == 0) return RtxGlobalVolumetrics::fogRemapTransmittanceMeasurementDistanceMinMetersObject().getDescription();
    if (std::strcmp(name, "fogRemapTransmittanceMeasurementDistanceMaxMeters") == 0) return RtxGlobalVolumetrics::fogRemapTransmittanceMeasurementDistanceMaxMetersObject().getDescription();
    if (std::strcmp(name, "fogRemapColorMultiscatteringScale") == 0) return RtxGlobalVolumetrics::fogRemapColorMultiscatteringScaleObject().getDescription();
    if (std::strcmp(name, "enableTranslucentShadows") == 0) return RtxGlobalVolumetrics::enableTranslucentShadowsObject().getDescription();
    if (std::strcmp(name, "depthOffset") == 0) return RtxGlobalVolumetrics::depthOffsetObject().getDescription();
    if (std::strcmp(name, "noiseFieldOctaves") == 0) return RtxGlobalVolumetrics::noiseFieldOctavesObject().getDescription();
    if (std::strcmp(name, "atmosphereSunFogScale") == 0) return RtxOptions::atmosphereSunVolumetricRadianceScaleObject().getDescription();
    return "";
  }
  // True if any field in this (group[, section]) matches the filter.
  bool sectionHasMatch(const char* group, const char* section, const char* filter) {
    for (int k = 0; k < kFieldCount; ++k) {
      const WeatherFieldDesc& d = kFieldDescs[k];
      if (std::strcmp(d.group, group) != 0) continue;
      if (section && std::strcmp(d.section, section) != 0) continue;
      if (matchesFilter(d.label, filter)) return true;
    }
    return false;
  }

  // ---------------------------------------------------------------------------
  // Derived "Fog Density" / "Fog Tint" controls (fork).
  //
  // The raw volumetric model is a transmittance pair (colour + measurement
  // distance) where THICKER fog means a SHORTER distance, and the fog's apparent
  // colour comes from single-scattering albedo rather than the transmittance
  // colour. Both fight intuition. These two derived widgets sit on top of the raw
  // sliders and present the familiar mental model:
  //   Fog Density  0..1   -> transmittanceMeasurementDistanceMeters, exp-mapped
  //                          so shorter distance = denser (0 = clear, 1 = whiteout).
  //   Fog Tint     colour -> singleScatteringAlbedo (the colour fog scatters).
  // The raw sliders remain below for power users; both views read/write the same
  // option objects via get()/setDeferred(), so they stay in lockstep.
  // ---------------------------------------------------------------------------
  constexpr float kFogVisMinM = 10.0f;    // maps to density 1.0 (whiteout)
  constexpr float kFogVisMaxM = 2000.0f;  // maps to density 0.0 (clear)

  float fogDistanceToDensity(float distM) {
    const float d = std::min(std::max(distM, kFogVisMinM), kFogVisMaxM);
    return saturate(std::log(kFogVisMaxM / d) / std::log(kFogVisMaxM / kFogVisMinM));
  }
  float fogDensityToDistance(float density) {
    return kFogVisMaxM * std::pow(kFogVisMinM / kFogVisMaxM, saturate(density));
  }

  // Per-preset RtxOption pointer accessors for the two underlying fog options the
  // derived widgets drive. Same option objects the raw sliders bind to.
#define WEATHER_PRESET_OBJPTR(FIELD, TYPE, FN)                                   \
  RtxOption<TYPE>* FN(int p) {                                                   \
    switch (p) {                                                                 \
      case WP_clear:        return &RtxOptions::clear_##FIELD##Object();         \
      case WP_partlyCloudy: return &RtxOptions::partlyCloudy_##FIELD##Object();  \
      case WP_overcast:     return &RtxOptions::overcast_##FIELD##Object();      \
      case WP_hazy:         return &RtxOptions::hazy_##FIELD##Object();          \
      case WP_foggy:        return &RtxOptions::foggy_##FIELD##Object();         \
      case WP_drizzle:      return &RtxOptions::drizzle_##FIELD##Object();       \
      case WP_rainstorm:    return &RtxOptions::rainstorm_##FIELD##Object();     \
      case WP_thunderstorm: return &RtxOptions::thunderstorm_##FIELD##Object();  \
      case WP_snow:         return &RtxOptions::snow_##FIELD##Object();          \
      case WP_blizzard:     return &RtxOptions::blizzard_##FIELD##Object();      \
      case WP_sandstorm:    return &RtxOptions::sandstorm_##FIELD##Object();     \
      case WP_smoggy:       return &RtxOptions::smoggy_##FIELD##Object();        \
      default:              return nullptr;                                      \
    }                                                                            \
  }
  WEATHER_PRESET_OBJPTR(transmittanceMeasurementDistanceMeters, float,   presetFogDistanceObj)
  WEATHER_PRESET_OBJPTR(singleScatteringAlbedo,                 Vector3, presetFogTintObj)
#undef WEATHER_PRESET_OBJPTR

  // Renders the derived Fog Density + Fog Tint widgets for one preset, honoring
  // the panel's name filter. Returns true if it rendered anything.
  bool renderDerivedFogControls(int presetIdx, const char* filter) {
    RtxOption<float>*   distObj = presetFogDistanceObj(presetIdx);
    RtxOption<Vector3>* tintObj = presetFogTintObj(presetIdx);
    if (!distObj || !tintObj) {
      return false;
    }

    bool rendered = false;
    if (matchesFilter("Fog Density", filter)) {
      float density = fogDistanceToDensity(distObj->get());
      if (ImGui::SliderFloat("Fog Density", &density, 0.0f, 1.0f, "%.2f")) {
        distObj->setDeferred(fogDensityToDistance(density));
      }
      RemixGui::SetTooltipToLastWidgetOnHover(
        "Intuitive thickness dial. 0 = clear, 1 = whiteout (~10 m visibility). "
        "Drives Transmittance Distance below (shorter distance = denser fog).");
      rendered = true;
    }
    if (matchesFilter("Fog Tint", filter)) {
      RemixGui::ColorEdit3("Fog Tint", tintObj);
      RemixGui::SetTooltipToLastWidgetOnHover(
        "The colour the fog scatters (single-scattering albedo). This, not "
        "Transmittance Color, is what tints the fog you actually see.");
      rendered = true;
    }
    return rendered;
  }

  // Renders the per-preset editor in the main panel's design language: nested
  // TreeNodes (group -> section), default-open, ColorEdit swatches for colors.
  // Generated from the field table, so new fields appear automatically. With a
  // filter active, matching trees auto-open and empty ones are hidden.
  void renderPresetEditor(int presetIdx, const char* filter, ImGuiSliderFlags fl) {
    const bool filtering = filter && filter[0];

    for (int gi = 0; gi < kFieldCount; ++gi) {
      const char* group = kFieldDescs[gi].group;
      bool groupSeen = false;
      for (int k = 0; k < gi; ++k) {
        if (std::strcmp(kFieldDescs[k].group, group) == 0) { groupSeen = true; break; }
      }
      if (groupSeen) continue;
      if (!sectionHasMatch(group, nullptr, filter)) continue;

      ImGui::SetNextItemOpen(true, filtering ? ImGuiCond_Always : ImGuiCond_Once);
      if (!ImGui::TreeNode(group)) continue;

      for (int si = 0; si < kFieldCount; ++si) {
        if (std::strcmp(kFieldDescs[si].group, group) != 0) continue;
        const char* section = kFieldDescs[si].section;
        bool sectionSeen = false;
        for (int k = 0; k < si; ++k) {
          if (std::strcmp(kFieldDescs[k].group, group) == 0 &&
              std::strcmp(kFieldDescs[k].section, section) == 0) { sectionSeen = true; break; }
        }
        if (sectionSeen) continue;
        if (!sectionHasMatch(group, section, filter)) continue;

        ImGui::SetNextItemOpen(true, filtering ? ImGuiCond_Always : ImGuiCond_Once);
        if (!ImGui::TreeNode(section)) continue;
        // Friendly derived controls ride at the top of Volumetric Fog -> Medium,
        // above the raw transmittance sliders they remap.
        if (std::strcmp(group, "Volumetric Fog") == 0 && std::strcmp(section, "Medium") == 0) {
          if (renderDerivedFogControls(presetIdx, filter)) {
            ImGui::Separator();
          }
        }
        for (int fi = 0; fi < kFieldCount; ++fi) {
          const WeatherFieldDesc& d = kFieldDescs[fi];
          if (std::strcmp(d.group, group) != 0 || std::strcmp(d.section, section) != 0) continue;
          if (!matchesFilter(d.label, filter)) continue;
          ImGui::PushID(fi);
          d.renderSlider(presetIdx, fl);
          const char* tip = weatherFieldTooltip(d.name);
          if (tip && tip[0]) { RemixGui::SetTooltipToLastWidgetOnHover(tip); }
          ImGui::PopID();
        }
        ImGui::TreePop();
      }
      ImGui::TreePop();
    }
  }

  // ---------------------------------------------------------------------------
  // snapshotRenderer — reads current live renderer RTX_OPTION values.
  //
  // FIELD ORDER matches WEATHER_PRESET_FIELD_LIST exactly (same 4 sites).
  // Cloud fields: RtxOptions::xxx()
  // Atmosphere fields: RtxOptions::xxx()
  // Volumetric fields: RtxGlobalVolumetrics::xxx()
  //   (anisotropy option is named 'anisotropy' in the class; the snapshot
  //    field is named 'volumetricAnisotropy' to avoid clash with cloudAnisotropy)
  // ---------------------------------------------------------------------------
  WeatherSnapshot snapshotRenderer() {
    WeatherSnapshot s;
    // Cloud (16)
    s.cloudDensity               = RtxOptions::cloudDensity();
    s.cloudCoverageMean          = RtxOptions::cloudCoverageMean();
    s.cloudCoverageSpread        = RtxOptions::cloudCoverageSpread();
    s.cloudCoverageNoiseScale    = RtxOptions::cloudCoverageNoiseScale();
    s.cloudTypeMean              = RtxOptions::cloudTypeMean();
    s.cloudTypeSpread            = RtxOptions::cloudTypeSpread();
    s.cloudTypeNoiseScale        = RtxOptions::cloudTypeNoiseScale();
    s.cloudColor                 = RtxOptions::cloudColor();
    s.cloudWindSpeed             = RtxOptions::cloudWindSpeed();
    s.cloudWindDirection         = RtxOptions::cloudWindDirection();
    s.cloudShadowStrength        = RtxOptions::cloudShadowStrength();
    s.cloudThickness             = RtxOptions::cloudThickness();
    s.cloudUndersideLightSigma = RtxOptions::cloudUndersideLightSigma();
    s.cloudBottomDarkening     = RtxOptions::cloudBottomDarkening();
    s.cloudAerialFadePerKm     = RtxOptions::cloudAerialFadePerKm();
    s.cloudAerialHazePerKm     = RtxOptions::cloudAerialHazePerKm();
    // Atmosphere (5)
    s.airDensity                 = RtxOptions::airDensity();
    s.aerosolDensity             = RtxOptions::aerosolDensity();
    s.sunIlluminance             = RtxOptions::sunIlluminance();
    s.rayleighScattering         = RtxOptions::rayleighScattering();
    s.skyIndirectRadianceScale   = RtxOptions::skyIndirectRadianceScale();
    // Sky/moon mood (4)
    s.nightSkyBrightness         = RtxOptions::nightSkyBrightness();
    s.nightSkyColor              = RtxOptions::nightSkyColor();
    s.moonNeeStrength            = RtxOptions::moonNeeStrength();
    s.moonAtmosphericCouplingStrength = RtxOptions::moonAtmosphericCouplingStrength();
    // Volumetric (27) — class is RtxGlobalVolumetrics
    s.transmittanceColor                     = RtxGlobalVolumetrics::transmittanceColor();
    s.transmittanceMeasurementDistanceMeters = RtxGlobalVolumetrics::transmittanceMeasurementDistanceMeters();
    s.singleScatteringAlbedo                 = RtxGlobalVolumetrics::singleScatteringAlbedo();
    s.volumetricAnisotropy                   = RtxGlobalVolumetrics::anisotropy();
    // Volumetric appearance (fork - full set)
    s.fogSunVisibilityGain = RtxGlobalVolumetrics::fogSunVisibilityGain();
    s.volumetricConsumerGain = RtxGlobalVolumetrics::volumetricConsumerGain();
    s.enableHeterogeneousFog = RtxGlobalVolumetrics::enableHeterogeneousFog();
    s.noiseFieldDensityScale = RtxGlobalVolumetrics::noiseFieldDensityScale();
    s.noiseFieldDensityExponent = RtxGlobalVolumetrics::noiseFieldDensityExponent();
    s.noiseFieldInitialFrequencyPerMeter = RtxGlobalVolumetrics::noiseFieldInitialFrequencyPerMeter();
    s.noiseFieldLacunarity = RtxGlobalVolumetrics::noiseFieldLacunarity();
    s.noiseFieldGain = RtxGlobalVolumetrics::noiseFieldGain();
    s.noiseFieldTimeScale = RtxGlobalVolumetrics::noiseFieldTimeScale();
    s.noiseFieldSubStepSizeMeters = RtxGlobalVolumetrics::noiseFieldSubStepSizeMeters();
    s.froxelMaxDistanceMeters = RtxGlobalVolumetrics::froxelMaxDistanceMeters();
    s.enableFogRemap = RtxGlobalVolumetrics::enableFogRemap();
    s.enableFogColorRemap = RtxGlobalVolumetrics::enableFogColorRemap();
    s.enableFogMaxDistanceRemap = RtxGlobalVolumetrics::enableFogMaxDistanceRemap();
    s.fogRemapMaxDistanceMinMeters = RtxGlobalVolumetrics::fogRemapMaxDistanceMinMeters();
    s.fogRemapMaxDistanceMaxMeters = RtxGlobalVolumetrics::fogRemapMaxDistanceMaxMeters();
    s.fogRemapTransmittanceMeasurementDistanceMinMeters = RtxGlobalVolumetrics::fogRemapTransmittanceMeasurementDistanceMinMeters();
    s.fogRemapTransmittanceMeasurementDistanceMaxMeters = RtxGlobalVolumetrics::fogRemapTransmittanceMeasurementDistanceMaxMeters();
    s.fogRemapColorMultiscatteringScale = RtxGlobalVolumetrics::fogRemapColorMultiscatteringScale();
    s.enableTranslucentShadows = RtxGlobalVolumetrics::enableTranslucentShadows();
    s.atmosphereSunFogScale    = RtxOptions::atmosphereSunVolumetricRadianceScale();
    s.depthOffset              = RtxGlobalVolumetrics::depthOffset();
    s.noiseFieldOctaves        = static_cast<float>(RtxGlobalVolumetrics::noiseFieldOctaves());
    return s;
  }

  // ---------------------------------------------------------------------------
  // writeBlendedToDerivedLayer — writes each field of interp to the Derived
  // layer of its underlying RTX_OPTION via setImmediately().
  //
  // FIELD ORDER matches WEATHER_PRESET_FIELD_LIST exactly (same 4 sites).
  // ---------------------------------------------------------------------------
  // --- Write gate (fork) ----------------------------------------------------
  // A weather param IDENTICAL across all 12 presets is not a weather
  // differentiator, so force-writing it every frame would needlessly clobber a
  // game's own config for that option. weatherVaries_<field>() reports whether a
  // field actually differs between presets; the new volumetric-appearance writes
  // are gated on it (recomputed each frame so editor tuning takes effect).
  bool weatherNeq(float a, float b) { return a != b; }
  bool weatherNeq(bool a, bool b)   { return a != b; }
  bool weatherNeq(const Vector3& a, const Vector3& b) { return a.x != b.x || a.y != b.y || a.z != b.z; }
#define WVARIES(name)                                          \
  bool weatherVaries_##name() {                                \
    const auto v0 = RtxOptions::clear_##name();                \
    return weatherNeq(RtxOptions::partlyCloudy_##name(), v0)   \
        || weatherNeq(RtxOptions::overcast_##name(),     v0)   \
        || weatherNeq(RtxOptions::hazy_##name(),         v0)   \
        || weatherNeq(RtxOptions::foggy_##name(),        v0)   \
        || weatherNeq(RtxOptions::drizzle_##name(),      v0)   \
        || weatherNeq(RtxOptions::rainstorm_##name(),    v0)   \
        || weatherNeq(RtxOptions::thunderstorm_##name(), v0)   \
        || weatherNeq(RtxOptions::snow_##name(),         v0)   \
        || weatherNeq(RtxOptions::blizzard_##name(),     v0)   \
        || weatherNeq(RtxOptions::sandstorm_##name(),    v0)   \
        || weatherNeq(RtxOptions::smoggy_##name(),       v0);  \
  }
  WVARIES(fogSunVisibilityGain)
  WVARIES(volumetricConsumerGain)
  WVARIES(enableHeterogeneousFog)
  WVARIES(noiseFieldDensityScale)
  WVARIES(noiseFieldDensityExponent)
  WVARIES(noiseFieldInitialFrequencyPerMeter)
  WVARIES(noiseFieldLacunarity)
  WVARIES(noiseFieldGain)
  WVARIES(noiseFieldTimeScale)
  WVARIES(noiseFieldSubStepSizeMeters)
  WVARIES(froxelMaxDistanceMeters)
  WVARIES(enableFogRemap)
  WVARIES(enableFogColorRemap)
  WVARIES(enableFogMaxDistanceRemap)
  WVARIES(fogRemapMaxDistanceMinMeters)
  WVARIES(fogRemapMaxDistanceMaxMeters)
  WVARIES(fogRemapTransmittanceMeasurementDistanceMinMeters)
  WVARIES(fogRemapTransmittanceMeasurementDistanceMaxMeters)
  WVARIES(fogRemapColorMultiscatteringScale)
  WVARIES(enableTranslucentShadows)
  WVARIES(atmosphereSunFogScale)
  WVARIES(depthOffset)
  WVARIES(noiseFieldOctaves)
  WVARIES(cloudUndersideLightSigma)
  WVARIES(cloudBottomDarkening)
  WVARIES(cloudAerialFadePerKm)
  WVARIES(cloudAerialHazePerKm)
  WVARIES(moonNeeStrength)
  WVARIES(moonAtmosphericCouplingStrength)
  WVARIES(rayleighScattering)
  WVARIES(nightSkyColor)
  WVARIES(skyIndirectRadianceScale)
#undef WVARIES
  void writeBlendedToDerivedLayer(const WeatherSnapshot& interp) {
    // Cloud (16)
    RtxOptions::cloudDensityObject().setImmediately(interp.cloudDensity);
    RtxOptions::cloudCoverageMeanObject().setImmediately(interp.cloudCoverageMean);
    RtxOptions::cloudCoverageSpreadObject().setImmediately(interp.cloudCoverageSpread);
    RtxOptions::cloudCoverageNoiseScaleObject().setImmediately(interp.cloudCoverageNoiseScale);
    RtxOptions::cloudTypeMeanObject().setImmediately(interp.cloudTypeMean);
    RtxOptions::cloudTypeSpreadObject().setImmediately(interp.cloudTypeSpread);
    RtxOptions::cloudTypeNoiseScaleObject().setImmediately(interp.cloudTypeNoiseScale);
    RtxOptions::cloudColorObject().setImmediately(interp.cloudColor);
    RtxOptions::cloudWindSpeedObject().setImmediately(interp.cloudWindSpeed);
    RtxOptions::cloudWindDirectionObject().setImmediately(interp.cloudWindDirection);
    RtxOptions::cloudShadowStrengthObject().setImmediately(interp.cloudShadowStrength);
    RtxOptions::cloudThicknessObject().setImmediately(interp.cloudThickness);
    if (weatherVaries_cloudUndersideLightSigma()) RtxOptions::cloudUndersideLightSigmaObject().setImmediately(interp.cloudUndersideLightSigma);
    if (weatherVaries_cloudBottomDarkening())     RtxOptions::cloudBottomDarkeningObject().setImmediately(interp.cloudBottomDarkening);
    if (weatherVaries_cloudAerialFadePerKm())     RtxOptions::cloudAerialFadePerKmObject().setImmediately(interp.cloudAerialFadePerKm);
    if (weatherVaries_cloudAerialHazePerKm())     RtxOptions::cloudAerialHazePerKmObject().setImmediately(interp.cloudAerialHazePerKm);
    // Atmosphere (5); rayleighScattering (daytime sky colour) and skyIndirectRadianceScale
    // (sky light) are neutral in every preset today, so gate them to avoid clobbering
    // the game's own sky tint / sky-fill brightness.
    RtxOptions::airDensityObject().setImmediately(interp.airDensity);
    RtxOptions::aerosolDensityObject().setImmediately(interp.aerosolDensity);
    RtxOptions::sunIlluminanceObject().setImmediately(interp.sunIlluminance);
    if (weatherVaries_rayleighScattering())       RtxOptions::rayleighScatteringObject().setImmediately(interp.rayleighScattering);
    if (weatherVaries_skyIndirectRadianceScale()) RtxOptions::skyIndirectRadianceScaleObject().setImmediately(interp.skyIndirectRadianceScale);
    // Sky/moon mood (4); nightSkyBrightness varies, so always write. nightSkyColor
    // and the two moon
    // gains are 1.0 in every preset today, so gate them like the other invariant
    // fields to avoid clobbering a game's own moon config when they don't differ.
    RtxOptions::nightSkyBrightnessObject().setImmediately(interp.nightSkyBrightness);
    if (weatherVaries_nightSkyColor())                   RtxOptions::nightSkyColorObject().setImmediately(interp.nightSkyColor);
    if (weatherVaries_moonNeeStrength())                 RtxOptions::moonNeeStrengthObject().setImmediately(interp.moonNeeStrength);
    if (weatherVaries_moonAtmosphericCouplingStrength()) RtxOptions::moonAtmosphericCouplingStrengthObject().setImmediately(interp.moonAtmosphericCouplingStrength);
    // Volumetric (27) — class is RtxGlobalVolumetrics
    RtxGlobalVolumetrics::transmittanceColorObject().setImmediately(interp.transmittanceColor);
    RtxGlobalVolumetrics::transmittanceMeasurementDistanceMetersObject().setImmediately(interp.transmittanceMeasurementDistanceMeters);
    RtxGlobalVolumetrics::singleScatteringAlbedoObject().setImmediately(interp.singleScatteringAlbedo);
    RtxGlobalVolumetrics::anisotropyObject().setImmediately(interp.volumetricAnisotropy);
    // Volumetric appearance (fork - full set)
    if (weatherVaries_fogSunVisibilityGain()) RtxGlobalVolumetrics::fogSunVisibilityGainObject().setImmediately(interp.fogSunVisibilityGain);
    if (weatherVaries_volumetricConsumerGain()) RtxGlobalVolumetrics::volumetricConsumerGainObject().setImmediately(interp.volumetricConsumerGain);
    if (weatherVaries_enableHeterogeneousFog()) RtxGlobalVolumetrics::enableHeterogeneousFogObject().setImmediately(interp.enableHeterogeneousFog);
    if (weatherVaries_noiseFieldDensityScale()) RtxGlobalVolumetrics::noiseFieldDensityScaleObject().setImmediately(interp.noiseFieldDensityScale);
    if (weatherVaries_noiseFieldDensityExponent()) RtxGlobalVolumetrics::noiseFieldDensityExponentObject().setImmediately(interp.noiseFieldDensityExponent);
    if (weatherVaries_noiseFieldInitialFrequencyPerMeter()) RtxGlobalVolumetrics::noiseFieldInitialFrequencyPerMeterObject().setImmediately(interp.noiseFieldInitialFrequencyPerMeter);
    if (weatherVaries_noiseFieldLacunarity()) RtxGlobalVolumetrics::noiseFieldLacunarityObject().setImmediately(interp.noiseFieldLacunarity);
    if (weatherVaries_noiseFieldGain()) RtxGlobalVolumetrics::noiseFieldGainObject().setImmediately(interp.noiseFieldGain);
    if (weatherVaries_noiseFieldTimeScale()) RtxGlobalVolumetrics::noiseFieldTimeScaleObject().setImmediately(interp.noiseFieldTimeScale);
    if (weatherVaries_noiseFieldSubStepSizeMeters()) RtxGlobalVolumetrics::noiseFieldSubStepSizeMetersObject().setImmediately(interp.noiseFieldSubStepSizeMeters);
    if (weatherVaries_froxelMaxDistanceMeters()) RtxGlobalVolumetrics::froxelMaxDistanceMetersObject().setImmediately(interp.froxelMaxDistanceMeters);
    if (weatherVaries_enableFogRemap()) RtxGlobalVolumetrics::enableFogRemapObject().setImmediately(interp.enableFogRemap);
    if (weatherVaries_enableFogColorRemap()) RtxGlobalVolumetrics::enableFogColorRemapObject().setImmediately(interp.enableFogColorRemap);
    if (weatherVaries_enableFogMaxDistanceRemap()) RtxGlobalVolumetrics::enableFogMaxDistanceRemapObject().setImmediately(interp.enableFogMaxDistanceRemap);
    if (weatherVaries_fogRemapMaxDistanceMinMeters()) RtxGlobalVolumetrics::fogRemapMaxDistanceMinMetersObject().setImmediately(interp.fogRemapMaxDistanceMinMeters);
    if (weatherVaries_fogRemapMaxDistanceMaxMeters()) RtxGlobalVolumetrics::fogRemapMaxDistanceMaxMetersObject().setImmediately(interp.fogRemapMaxDistanceMaxMeters);
    if (weatherVaries_fogRemapTransmittanceMeasurementDistanceMinMeters()) RtxGlobalVolumetrics::fogRemapTransmittanceMeasurementDistanceMinMetersObject().setImmediately(interp.fogRemapTransmittanceMeasurementDistanceMinMeters);
    if (weatherVaries_fogRemapTransmittanceMeasurementDistanceMaxMeters()) RtxGlobalVolumetrics::fogRemapTransmittanceMeasurementDistanceMaxMetersObject().setImmediately(interp.fogRemapTransmittanceMeasurementDistanceMaxMeters);
    if (weatherVaries_fogRemapColorMultiscatteringScale()) RtxGlobalVolumetrics::fogRemapColorMultiscatteringScaleObject().setImmediately(interp.fogRemapColorMultiscatteringScale);
    if (weatherVaries_enableTranslucentShadows()) RtxGlobalVolumetrics::enableTranslucentShadowsObject().setImmediately(interp.enableTranslucentShadows);
    if (weatherVaries_atmosphereSunFogScale())    RtxOptions::atmosphereSunVolumetricRadianceScaleObject().setImmediately(interp.atmosphereSunFogScale);
    if (weatherVaries_depthOffset())              RtxGlobalVolumetrics::depthOffsetObject().setImmediately(interp.depthOffset);
    if (weatherVaries_noiseFieldOctaves())        RtxGlobalVolumetrics::noiseFieldOctavesObject().setImmediately(static_cast<uint32_t>(interp.noiseFieldOctaves + 0.5f));
  }

} } }  // namespace dxvk::fork_weather::(anonymous)


// ---------------------------------------------------------------------------
// WeatherBlender member implementations
// ---------------------------------------------------------------------------
namespace dxvk { namespace fork_weather {

  // ---------------------------------------------------------------------------
  // WeatherBlender ctor/dtor — maintain the file-scoped active-blender pointer.
  // ---------------------------------------------------------------------------
  WeatherBlender::WeatherBlender() {
    g_activeBlender = this;
  }

  WeatherBlender::~WeatherBlender() {
    if (this == g_activeBlender) {
      g_activeBlender = nullptr;
    }
  }

  // ---------------------------------------------------------------------------
  // update — per-frame entry point.
  //
  // Lifecycle:
  //  1. Advance clock.
  //  2. If paused: return (manual ImGui edits persist).
  //  3. Read __weather.target. If absent/empty/unknown: clear state, return.
  //  4. If target changed (new or retarget):
  //     a. First activation: snapshot current renderer state.
  //     b. Mid-blend retarget: compute current t, lerp from prev toward old
  //        target, store result as new previous snapshot.
  //     c. Update m_targetPresetName, read m_blendDurationSec, reset start.
  //  5. Compute t = clamp((now - start) / dur, 0, 1).
  //  6. applyBlendedValues(t).
  //  7. publishStateToGameStateStore(t).
  // ---------------------------------------------------------------------------
  void WeatherBlender::update(float deltaTimeSeconds) {
    m_currentTimeSec += deltaTimeSeconds;

    if (m_paused) {
      return;
    }

    // Drift state advance — happens on every non-paused frame, regardless of
    // whether the blender is dormant. Smoothing reads raw values from
    // GameStateStore, low-pass-filters toward them with tau = 1.0s, then
    // advances the phase. Negative raw values are clamped to 0 at read time.
    {
      constexpr float kSmoothTau = 1.0f;
      const float alpha = (deltaTimeSeconds > 0.0f)
        ? (1.0f - std::exp(-deltaTimeSeconds / kSmoothTau))
        : 0.0f;
      const float driftSpeedRaw     = std::max(0.0f,
        readFloatFromGameStateStore("__weather.drift_speed",     1.0f));
      const float driftIntensityRaw = std::max(0.0f,
        readFloatFromGameStateStore("__weather.drift_intensity", 1.0f));
      m_driftSpeedSmoothed     += alpha * (driftSpeedRaw     - m_driftSpeedSmoothed);
      m_driftIntensitySmoothed += alpha * (driftIntensityRaw - m_driftIntensitySmoothed);
      // Belt-and-braces clamp against any pathological smoothed value.
      m_driftSpeedSmoothed     = std::min(std::max(m_driftSpeedSmoothed,     0.0f), 100.0f);
      m_driftIntensitySmoothed = std::min(std::max(m_driftIntensitySmoothed, 0.0f), 100.0f);
      m_driftPhaseSeconds += deltaTimeSeconds * m_driftSpeedSmoothed;
    }

    // Step 3: read and validate target preset.
    std::string newTarget = readStringFromGameStateStore("__weather.target");
    if (newTarget.empty()) {
      m_targetPresetName.clear();
      m_previousPresetName.clear();
      return;
    }
    if (!isKnownPresetName(newTarget)) {
      // Warn once per distinct unknown name so plugin authors who typo a
      // preset string ("rainstOrm") get a diagnostic instead of silent
      // dormancy. Subsequent SetGameValue writes with the same bad name
      // stay quiet to avoid log spam.
      static std::unordered_set<std::string> s_warned;
      if (s_warned.insert(newTarget).second) {
        Logger::warn(str::format(
          "WeatherBlender: unknown preset name '", newTarget,
          "' in __weather.target -- known names are clear, partlyCloudy, "
          "overcast, hazy, foggy, drizzle, rainstorm, thunderstorm, snow, "
          "blizzard, sandstorm, smoggy. Treating as dormant."));
      }
      m_targetPresetName.clear();
      m_previousPresetName.clear();
      return;
    }

    // Step 4: handle retarget or first activation.
    if (newTarget != m_targetPresetName) {
      if (m_targetPresetName.empty()) {
        // First activation: snapshot current live renderer state.
        m_previousSnapshot    = snapshotCurrentValues();
        m_previousPresetName  = "(initial)";
      } else {
        // Mid-blend retarget: capture the partially-blended state.
        // Lerp logic lives in lerpSnapshot (anonymous namespace).
        float currentT = saturate(static_cast<float>(
          (m_currentTimeSec - m_blendStartTimeSec) / std::max(0.001f, m_blendDurationSec)));

        WeatherSnapshot oldTargetValues;
        readPresetValues(m_targetPresetName, oldTargetValues);

        // Build retarget snapshot by lerping prev toward the old target at currentT.
        m_previousSnapshot   = lerpSnapshot(m_previousSnapshot, oldTargetValues, currentT);
        m_previousPresetName = m_targetPresetName;
      }

      m_targetPresetName   = newTarget;
      m_blendDurationSec   = std::max(0.001f, readFloatFromGameStateStore("__weather.blend_seconds", 1.0f));
      m_blendStartTimeSec  = m_currentTimeSec;
    }

    // Step 5: compute interpolation parameter.
    float t = saturate(static_cast<float>((m_currentTimeSec - m_blendStartTimeSec) / m_blendDurationSec));

    // Step 6 + 7.
    applyBlendedValues(t);
    publishStateToGameStateStore(t);
  }

  // ---------------------------------------------------------------------------
  // showImguiSettings — full ImGui weather-preset panel.
  //
  // Layout:
  //  1. Combo — 13 entries: "(none / dormant)" + 12 preset names.
  //  2. Float slider — Blend Duration (sec), 0–600.
  //  3. "Apply Preset" button — writes __weather.blend_seconds and
  //     __weather.target to GameStateStore.
  //  4. Separator.
  //  5. "Pause Weather Blender" checkbox (m_paused), with tooltip.
  //  6. Read-only state display (current / target / previous / blend progress).
  //  7. "Tune Preset Defaults" collapsing tree — per-preset slider blocks.
  // ---------------------------------------------------------------------------
  void WeatherBlender::showImguiSettings() {

    static const char* kPresetNamesUI[] = {
      "(none / dormant)",
      "clear", "partlyCloudy", "overcast", "hazy", "foggy", "drizzle",
      "rainstorm", "thunderstorm", "snow", "blizzard", "sandstorm", "smoggy"
    };
    constexpr int kPresetCountUI = static_cast<int>(IM_ARRAYSIZE(kPresetNamesUI));

    // ---- Transition controls (what the blender plays) ----
    static int s_selectedIndex = 0;
    ImGui::Combo("Target Preset", &s_selectedIndex, kPresetNamesUI, kPresetCountUI);
    static float s_blendDuration = 30.0f;
    ImGui::SliderFloat("Blend Duration (sec)", &s_blendDuration, 0.0f, 600.0f, "%.1f");
    if (ImGui::Button("Apply Preset")) {
      char durBuf[32];
      std::snprintf(durBuf, sizeof(durBuf), "%.6f", s_blendDuration);
      fork_game_state::GameStateStore::get().set("__weather.blend_seconds", durBuf);
      const char* targetName = (s_selectedIndex == 0) ? "" : kPresetNamesUI[s_selectedIndex];
      fork_game_state::GameStateStore::get().set("__weather.target", targetName);
    }

    ImGui::Checkbox("Pause Weather Blender", &m_paused);
    RemixGui::SetTooltipToLastWidgetOnHover(
      "When checked, the blender stops writing to RTX_OPTIONs. "
      "Manual edits to the underlying sliders persist undisturbed.");

    {
      float currentT = 0.0f;
      if (!m_targetPresetName.empty() && m_blendDurationSec > 0.001f) {
        currentT = saturate(static_cast<float>((m_currentTimeSec - m_blendStartTimeSec) / m_blendDurationSec));
      }
      const std::string& dominantName = (currentT > 0.5f) ? m_targetPresetName : m_previousPresetName;
      const char* currentDisplay  = m_targetPresetName.empty()   ? "(dormant)" : dominantName.c_str();
      const char* targetDisplay   = m_targetPresetName.empty()   ? "(dormant)" : m_targetPresetName.c_str();
      const char* previousDisplay = m_previousPresetName.empty() ? "(dormant)" : m_previousPresetName.c_str();
      ImGui::TextDisabled("Current: %s    Target: %s    Previous: %s    Blend: %.3f",
                          currentDisplay, targetDisplay, previousDisplay, currentT);
    }

    ImGui::Separator();

    // The full per-preset editor lives in a separate pop-out window (toggled
    // here) so this inline panel stays focused on driving weather transitions.
    if (ImGui::Button(m_editorWindowOpen ? "Close Preset Editor" : "Open Preset Editor")) {
      m_editorWindowOpen = !m_editorWindowOpen;
    }
    RemixGui::SetTooltipToLastWidgetOnHover(
      "Opens the full per-preset editor (all settings + authoring tools) in a "
      "separate, movable window.");

    // ---- Weather Variation (slow preset-scale wander; API: __weather.drift_*) ----
    ImGui::Separator();
    if (ImGui::TreeNode("Weather Variation")) {
      ImGui::TextDisabled("Slow preset-scale wander of coverage + wind. "
                          "Field motion lives in Atmosphere -> Clouds -> Cloud Motion.");
      float driftSpeed     = readFloatFromGameStateStore("__weather.drift_speed",     1.0f);
      float driftIntensity = readFloatFromGameStateStore("__weather.drift_intensity", 1.0f);

      bool changedSpeed     = ImGui::SliderFloat("Variation speed",     &driftSpeed,     0.0f, 4.0f, "%.2f");
      RemixGui::SetTooltipToLastWidgetOnHover(
        "Scales how fast the weather variation evolves. 0 = frozen. Smoothed with "
        "tau = 1.0s. (API key: __weather.drift_speed.)");

      bool changedIntensity = ImGui::SliderFloat("Variation intensity", &driftIntensity, 0.0f, 3.0f, "%.2f");
      RemixGui::SetTooltipToLastWidgetOnHover(
        "Scales how big the variation swings are around the preset midpoint. "
        "0 = fully off. (API key: __weather.drift_intensity.)");

      if (changedSpeed) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.6f", driftSpeed);
        fork_game_state::GameStateStore::get().set("__weather.drift_speed", buf);
      }
      if (changedIntensity) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.6f", driftIntensity);
        fork_game_state::GameStateStore::get().set("__weather.drift_intensity", buf);
      }

      ImGui::Text("Variation phase:     %.2f s",  m_driftPhaseSeconds);
      ImGui::Text("Speed (smoothed):    %.3f",   m_driftSpeedSmoothed);
      ImGui::Text("Intensity (smoothed):%.3f",   m_driftIntensitySmoothed);

      if (ImGui::Button("Reset to defaults")) {
        fork_game_state::GameStateStore::get().set("__weather.drift_speed",     "1.0");
        fork_game_state::GameStateStore::get().set("__weather.drift_intensity", "1.0");
      }
      ImGui::SameLine();
      if (ImGui::Button("Disable variation")) {
        fork_game_state::GameStateStore::get().set("__weather.drift_intensity", "0.0");
      }

      ImGui::TreePop();
    }
  }
  // ---------------------------------------------------------------------------
  // renderEditorWindow — the pop-out per-preset editor (separate movable window,
  // toggled from showImguiSettings). Holds the full field set grouped into
  // collapsible sections, plus the authoring tools.
  // ---------------------------------------------------------------------------
  void WeatherBlender::renderEditorWindow() {
    if (!m_editorWindowOpen) {
      return;
    }
    constexpr ImGuiSliderFlags sliderFlags = ImGuiSliderFlags_AlwaysClamp;

    ImGui::SetNextWindowSize(ImVec2(440.0f, 640.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Weather Preset Editor", &m_editorWindowOpen)) {
      ImGui::End();
      return;
    }

    static const char* kEditNames[] = {
      "clear", "partlyCloudy", "overcast", "hazy", "foggy", "drizzle",
      "rainstorm", "thunderstorm", "snow", "blizzard", "sandstorm", "smoggy"
    };
    constexpr int kEditCount = static_cast<int>(IM_ARRAYSIZE(kEditNames));
    static int s_editIndex = 0;

    ImGui::SetNextItemWidth(180.0f);
    ImGui::Combo("Editing Preset", &s_editIndex, kEditNames, kEditCount);
    ImGui::SameLine();
    if (ImGui::SmallButton("Use Active")) {
      int idx = presetIndexForName(m_targetPresetName);
      if (idx >= 0) { s_editIndex = idx; }
    }
    RemixGui::SetTooltipToLastWidgetOnHover(
      "Point the editor at whatever preset the blender is currently targeting.");

    static char s_filter[64] = "";
    ImGui::SetNextItemWidth(-1.0f);
    ImGui::InputTextWithHint("##weatherFilter", "filter settings by name...", s_filter, sizeof(s_filter));

    if (ImGui::TreeNode("Authoring tools")) {
      bool pinned = m_pinnedForTuning;
      if (ImGui::Checkbox("Pin & Freeze for Tuning", &pinned)) {
        if (pinned) {
          // Entering tuning: snap to this preset, freeze variation, remembering the
          // prior drift intensity so we can restore it on exit (non-destructive).
          m_savedDriftIntensity = readFloatFromGameStateStore("__weather.drift_intensity", 1.0f);
          fork_game_state::GameStateStore::get().set("__weather.blend_seconds", "0.0");
          fork_game_state::GameStateStore::get().set("__weather.target", kEditNames[s_editIndex]);
          fork_game_state::GameStateStore::get().set("__weather.drift_intensity", "0.0");
        } else {
          // Leaving tuning: restore the variation intensity we froze.
          char buf[32];
          std::snprintf(buf, sizeof(buf), "%.6f", m_savedDriftIntensity);
          fork_game_state::GameStateStore::get().set("__weather.drift_intensity", buf);
        }
        m_pinnedForTuning = pinned;
      }
      RemixGui::SetTooltipToLastWidgetOnHover(
        "Snap the blender to this preset (0 s blend) and freeze variation so edits show "
        "on a held image. Unchecking restores the previous variation intensity.");

      // While pinned, keep the held image on whatever preset is being edited, so
      // changing the Editing Preset combo above re-snaps the frozen view to it
      // (blend is already 0 s, so the switch is instant).
      static int s_appliedPinIndex = -1;
      if (m_pinnedForTuning) {
        if (s_editIndex != s_appliedPinIndex) {
          fork_game_state::GameStateStore::get().set("__weather.target", kEditNames[s_editIndex]);
          s_appliedPinIndex = s_editIndex;
        }
      } else {
        s_appliedPinIndex = -1;
      }

      static int s_copyFrom = 0;
      ImGui::SetNextItemWidth(160.0f);
      ImGui::Combo("##copyFrom", &s_copyFrom, kEditNames, kEditCount);
      ImGui::SameLine();
      if (ImGui::Button("Copy Into Edited")) { copyPresetToPreset(s_copyFrom, s_editIndex); }
      RemixGui::SetTooltipToLastWidgetOnHover(
        "Copy every value from the chosen preset into the one being edited.");

      if (ImGui::Button("Snapshot Live -> Preset")) { snapshotLiveToPreset(s_editIndex); }
      RemixGui::SetTooltipToLastWidgetOnHover(
        "Capture the current live renderer values into this preset. Tune the real "
        "atmosphere/volumetrics with the blender dormant, then capture.");

      if (ImGui::Button("Copy as user.conf lines")) {
        ImGui::SetClipboardText(exportPresetToConf(s_editIndex).c_str());
      }
      RemixGui::SetTooltipToLastWidgetOnHover(
        "Optional: copies this preset as rtx.weather.preset.* lines to the clipboard. "
        "The dev menu's Save Settings already persists edits to the modder config; use "
        "this only to move values into a specific game's user.conf.");

      ImGui::TreePop();
    }

    ImGui::Separator();
    renderPresetEditor(s_editIndex, s_filter, sliderFlags);

    ImGui::End();
  }
  // ---------------------------------------------------------------------------
  // snapshotCurrentValues — delegates to the free helper.
  // ---------------------------------------------------------------------------
  WeatherSnapshot WeatherBlender::snapshotCurrentValues() const {
    return snapshotRenderer();
  }

  // ---------------------------------------------------------------------------
  // applyBlendedValues — lerp prev snapshot toward target at t, write to
  // Derived layer.
  //
  // Lerp logic lives in lerpSnapshot (anonymous namespace). This member
  // reads the target preset, lerps from the previous snapshot toward it,
  // and writes the result to the Derived layer.
  // ---------------------------------------------------------------------------
  void WeatherBlender::applyBlendedValues(float t) {
    WeatherSnapshot targetValues;
    if (!readPresetValues(m_targetPresetName, targetValues)) {
      return;
    }
    WeatherSnapshot interp = lerpSnapshot(m_previousSnapshot, targetValues, t);
    applyDriftToSnapshot(interp, m_driftPhaseSeconds, m_driftIntensitySmoothed);
    writeBlendedToDerivedLayer(interp);
  }

  // ---------------------------------------------------------------------------
  // publishStateToGameStateStore — writes blend progress state.
  // ---------------------------------------------------------------------------
  void WeatherBlender::publishStateToGameStateStore(float t) const {
    // __weather.current = the destination the blender is targeting, matching the
    // documented contract (was previously the dominant-half preset, which made
    // plugins see the old preset for the first half of every transition).
    writeToGameStateStore("__weather.current", m_targetPresetName);
    writeToGameStateStore("__weather.previous", m_previousPresetName);

    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.4f", t);
    writeToGameStateStore("__weather.blend_progress", buf);
  }

} }  // namespace dxvk::fork_weather


// ---------------------------------------------------------------------------
// fork_hooks stub bodies
// ---------------------------------------------------------------------------
namespace dxvk { namespace fork_hooks {

  // Per-frame weather preset blender update. Reads __weather.target and
  // __weather.blend_seconds from the GameStateStore and writes blended weather
  // params to the Derived layer of their underlying RTX_OPTIONs. Dormant when
  // no target is set — zero behavioural change vs upstream.
  //
  // Real implementation lands in Task 3 (wires WeatherBlender into RtxContext
  // per-frame and resolves m_weatherBlender). For now, both args are unused.
  void updateWeatherBlender(class RtxContext& ctx, float deltaTimeSeconds) {
    if (ctx.m_weatherBlender) {
      ctx.m_weatherBlender->update(deltaTimeSeconds);
    }
  }

  // Renders the weather preset panel inside a TreeNode (matching the
  // surrounding atmosphere panel's tree style), delegating to the active
  // WeatherBlender's showImguiSettings(). No-op when no blender is live
  // (tests, pre-RtxContext-init).
  void showWeatherUI() {
    if (auto* b = fork_weather::g_activeBlender) {
      if (ImGui::TreeNode("Weather Presets")) {
        b->showImguiSettings();
        ImGui::TreePop();
      }
      // Pop-out editor window: drawn every frame the panel renders, so it stays
      // open regardless of whether the tree above is expanded.
      b->renderEditorWindow();
    }
  }

} }  // namespace dxvk::fork_hooks
