#pragma once

// rtx_fork_tonemap.h — fork-owned declarations for the tonemap operator
// enum and per-operator parameter plumbing. The operator logic itself
// lives in rtx_fork_tonemap.cpp and in the fork-owned shader headers
// under src/dxvk/shaders/rtx/pass/tonemap/ (aces.slangh, agx.slangh,
// lottes.slangh, psycho17.slangh, hable.slangh, fork_tonemap_operators.slangh).
//
// See docs/fork-touchpoints.md for the index of upstream files that
// call into fork_hooks::... for tonemap operator dispatch and UI.

#include <cstdint>

#include "rtx_option.h"

namespace dxvk {

  // Tonemapping operator. Shader-side constants live in
  // shaders/rtx/pass/tonemap/tonemapping.h as `tonemapOperator*` uints;
  // these two enumerations MUST stay in lockstep. The
  // populateTonemapOperatorArgs hook is the single place that casts
  // between them.
  enum class TonemapOperator : uint32_t {
    None          = 0, // Identity / passthrough.
    ACESHill      = 1, // Stephen Hill ACES fit.
    ACESNarkowicz = 2, // Krzysztof Narkowicz ACES approximation.
    HableFilmic   = 3,
    AgX           = 4,
    Lottes        = 5,
    Psycho17      = 6, // Renodx Psycho Test 17 (UI label: PsychoV17_Beta).
    GT7           = 7, // Gran Turismo 7 (Polyphony Digital / MIT). SDR, peak 1.0, ICtCp UCS.
    Neutwo        = 8, // Renodx Neutwo per-channel saturation curve (Carlos Lopez Jr. / MIT). Parameterless.
  };

  // Global-tonemapper operator selection. Defaults to None (identity); the
  // dynamic tone curve and local tonemapper were both removed in the
  // 2026-05-13 / 2026-05-15 refactors and the apply pass now dispatches
  // directly to the operator selected here.
  class RtxForkGlobalTonemap {
    RTX_OPTION_ENV("rtx.tonemap", TonemapOperator, tonemapOperator, TonemapOperator::Psycho17, "DXVK_TONEMAP_OPERATOR",
                   "Tonemapping operator applied to the post-exposure color buffer.\n"
                   "Supported values: 0 = None (saturate-only identity), 1 = Hill ACES, 2 = Narkowicz ACES, "
                   "3 = Hable Filmic, 4 = AgX, 5 = Lottes 2016, 6 = PsychoV17_Beta, "
                   "7 = Gran Turismo 7 (SDR), 8 = Neutwo (per-channel).");
  };

  // Local-tonemapper operator selection was removed; the local path now shares
  // RtxForkGlobalTonemap::tonemapOperator() to eliminate duplicated UI / config
  // knobs (rtx.localtonemap.tonemapOperator + per-operator local param sets).

  // Hable Filmic (Uncharted 2) operator parameters. Shared between the global
  // and local tonemap paths (the operator is per-selection, not per-path).
  // Defaults from gmod baad5e79 use Half-Life: Alyx values (W=4.0,
  // exposureBias=2.0) rather than the original Uncharted 2 reference (W=11.2).
  class RtxForkHableFilmic {
    RTX_OPTION("rtx.tonemap.hable", float, exposureBias,     2.00f, "Hable Filmic: pre-operator exposure multiplier.");
    RTX_OPTION("rtx.tonemap.hable", float, shoulderStrength, 0.15f, "Hable Filmic: A — shoulder strength.");
    RTX_OPTION("rtx.tonemap.hable", float, linearStrength,   0.50f, "Hable Filmic: B — linear strength.");
    RTX_OPTION("rtx.tonemap.hable", float, linearAngle,      0.10f, "Hable Filmic: C — linear angle.");
    RTX_OPTION("rtx.tonemap.hable", float, toeStrength,      0.20f, "Hable Filmic: D — toe strength.");
    RTX_OPTION("rtx.tonemap.hable", float, toeNumerator,     0.02f, "Hable Filmic: E — toe numerator.");
    RTX_OPTION("rtx.tonemap.hable", float, toeDenominator,   0.30f, "Hable Filmic: F — toe denominator.");
    RTX_OPTION("rtx.tonemap.hable", float, whitePoint,       4.00f, "Hable Filmic: W — linear-scene white point. Defaults to 4.0 (Half-Life: Alyx); Uncharted 2 reference is 11.2.");
  };

  // AgX operator parameters — uses the AgX Minimal (Benjamin Wrensch / MIT)
  // reference implementation. Look ordering: 0 = None, 1 = Golden, 2 = Punchy.
  class RtxForkAgX {
    RTX_OPTION("rtx.tonemap.agx", float, saturation, 1.0f, "AgX saturation multiplier. Range [0.0, 2.0].");
    RTX_OPTION("rtx.tonemap.agx", int,   look,       0,    "AgX look preset: 0 = None, 1 = Golden, 2 = Punchy.");
  };

  // Lottes 2016 operator parameters. Defaults from gmod cdf2c723.
  // The Lottes operator shares shader-args slots with Hable Filmic (see
  // tonemapping.h); populate hooks branch on the selected operator to
  // write the correct param set.
  class RtxForkLottes {
    RTX_OPTION("rtx.tonemap.lottes", float, hdrMax,   16.0f, "Lottes: peak HDR white value. Higher values preserve more highlight detail. Range [1.0, 64.0].");
    RTX_OPTION("rtx.tonemap.lottes", float, contrast,  1.2f, "Lottes: contrast control (also drives saturation / crosstalk). Range [1.0, 3.0].");
    RTX_OPTION("rtx.tonemap.lottes", float, shoulder,  1.0f, "Lottes: shoulder strength (highlight compression). Range [0.5, 2.0].");
    RTX_OPTION("rtx.tonemap.lottes", float, midIn,    0.18f, "Lottes: mid-grey input (scene linear). Range [0.01, 1.0].");
    RTX_OPTION("rtx.tonemap.lottes", float, midOut,   0.18f, "Lottes: mid-grey output. Range [0.01, 1.0].");
  };

  // Local-tonemapper AgX / Lottes / Psycho17 operator parameter sets were
  // removed along with rtx.localtonemap.tonemapOperator — the local tonemapper
  // now shares RtxForkAgX / RtxForkLottes / RtxForkPsycho17 with the global path.

  // Psycho Test 17 operator parameters (PsychoV17_Beta in the UI).
  // Self-contained port of renodx Psycho Test 17 — see
  // src/dxvk/shaders/rtx/pass/tonemap/psycho17.slangh for the MIT attribution
  // (Copyright (c) 2025 Carlos Lopez Jr.). Defaults track the renodx reference.
  class RtxForkPsycho17 {
    // peakValue is hardcoded to 1.0 in the populate hook (rtx_fork_tonemap.cpp) — no UI, no RTX_OPTION.
    RTX_OPTION("rtx.tonemap.psycho17", float, exposure,             1.0f,         "Psycho17: pre-operator exposure multiplier.");
    RTX_OPTION("rtx.tonemap.psycho17", float, highlights,           1.0f,         "Psycho17: highlight compression strength. Values > 1 compress highlights more aggressively.");
    RTX_OPTION("rtx.tonemap.psycho17", float, shadows,              1.0f,         "Psycho17: shadow lifting strength. Values > 1 lift shadows.");
    RTX_OPTION("rtx.tonemap.psycho17", float, contrast,             1.0f,         "Psycho17: contrast adjustment applied before tone mapping.");
    RTX_OPTION("rtx.tonemap.psycho17", float, purityScale,          1.0f,         "Psycho17: chromatic purity (saturation) scale factor.");
    RTX_OPTION("rtx.tonemap.psycho17", float, bleachingIntensity,   1.0f,         "Psycho17: Hunt-effect bleaching intensity. 0 = disabled, 1 = full bleach.");
    RTX_OPTION("rtx.tonemap.psycho17", float, clipPoint,          100.0f,         "Psycho17: accepted for parity with psycho11; unused by the psycho17 reference.");
    RTX_OPTION("rtx.tonemap.psycho17", float, hueRestore,           1.0f,         "Psycho17: source-hue restoration after adaptation (0 = none, 1 = full).");
    RTX_OPTION("rtx.tonemap.psycho17", float, adaptationContrast,   1.0f,         "Psycho17: accepted for parity with psycho11; unused by the psycho17 reference.");
    RTX_OPTION("rtx.tonemap.psycho17", int,   whiteCurveMode,       0,            "Psycho17: accepted for parity with psycho11; unused by the psycho17 reference.");
    RTX_OPTION("rtx.tonemap.psycho17", float, coneResponseExponent, 1.0f,         "Psycho17: cone response exponent for the Naka-Rushton stage.");
    RTX_OPTION("rtx.tonemap.psycho17", float, gamutCompression,     1.0f,         "Psycho17: output gamut compression strength. 0 = off, 1 = full.");
    RTX_OPTION("rtx.tonemap.psycho17", int,   gamutCompressionMode, 1,            "Psycho17: target gamut for output compression. 0 = BT.709, 1 = BT.2020.");
  };

} // namespace dxvk
