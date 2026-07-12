// rtx_fork_tonemap.cpp
//
// Fork-owned implementations of the tonemap operator hooks declared in
// rtx_fork_hooks.h. Hosts:
//   - writeOperatorParams: copies the per-operator RtxOption values
//     (Hable / AgX / Lottes / Psycho17) into the shared shader args
//     struct. When Lottes is the selected operator, its 5 params overlay
//     the 8 Hable push-constant slots — slot mapping is documented at
//     the struct definition in shaders/rtx/pass/tonemap/tonemapping.h.
//   - populateTonemapOperatorArgs: the global hook called from the
//     apply-tonemapping pass to populate the args struct.
//   - showTonemapOperatorUI: the operator-dropdown + per-operator
//     parameter panels rendered inside the global tonemap ImGui section.
//
// See docs/fork-touchpoints.md for the catalogue of fork hooks and which
// upstream files call each one.

#include "rtx_fork_hooks.h"
#include "rtx_fork_tonemap.h"
#include "rtx_imgui.h"    // RemixGui::Combo, RemixGui::DragFloat, RemixGui::Checkbox
#include "../imgui/imgui.h"

namespace dxvk {
  namespace fork_hooks {

    // Shared helper: copies the Hable / AgX / Lottes RtxOption values into
    // whatever args struct exposes the matching fields. The AgX + Lottes
    // source classes are template parameters so global and local paths
    // can share this plumbing while reading from their own option sets.
    // Hable is always read from the shared RtxForkHableFilmic (gmod also
    // uses global Hable params on the local path). When Lottes is the
    // selected operator, its 5 params overlay the 8 Hable push-constant
    // slots — see tonemapping.h for the documented slot mapping.
    template<typename AgXClass, typename LottesClass, typename Psycho17Class, typename ArgsT>
    static void writeOperatorParams(ArgsT& args, TonemapOperator op) {
      if (op == TonemapOperator::Lottes) {
        args.hableExposureBias     = LottesClass::hdrMax();
        args.hableShoulderStrength = LottesClass::contrast();
        args.hableLinearStrength   = LottesClass::shoulder();
        args.hableLinearAngle      = LottesClass::midIn();
        args.hableToeStrength      = LottesClass::midOut();
        args.hableToeNumerator     = 0.0f;
        args.hableToeDenominator   = 0.0f;
        args.hableWhitePoint       = 0.0f;
      } else {
        args.hableExposureBias     = RtxForkHableFilmic::exposureBias();
        args.hableShoulderStrength = RtxForkHableFilmic::shoulderStrength();
        args.hableLinearStrength   = RtxForkHableFilmic::linearStrength();
        args.hableLinearAngle      = RtxForkHableFilmic::linearAngle();
        args.hableToeStrength      = RtxForkHableFilmic::toeStrength();
        args.hableToeNumerator     = RtxForkHableFilmic::toeNumerator();
        args.hableToeDenominator   = RtxForkHableFilmic::toeDenominator();
        args.hableWhitePoint       = RtxForkHableFilmic::whitePoint();
      }
      args.agxSaturation = AgXClass::saturation();
      args.agxLook       = static_cast<uint32_t>(AgXClass::look());
      args.agxPad0       = 0.0f;
      args.agxPad1       = 0.0f;
      args.psycho17PeakValue            = 1.0f; // Hardcoded per design: peak luminance pinned at 1.0 (no UI / RTX_OPTION).
      args.psycho17Exposure             = Psycho17Class::exposure();
      args.psycho17Highlights           = Psycho17Class::highlights();
      args.psycho17Shadows              = Psycho17Class::shadows();
      args.psycho17Contrast             = Psycho17Class::contrast();
      args.psycho17PurityScale          = Psycho17Class::purityScale();
      args.psycho17BleachingIntensity   = Psycho17Class::bleachingIntensity();
      args.psycho17ClipPoint            = Psycho17Class::clipPoint();
      args.psycho17HueRestore           = Psycho17Class::hueRestore();
      args.psycho17AdaptationContrast   = Psycho17Class::adaptationContrast();
      args.psycho17WhiteCurveMode       = static_cast<uint32_t>(Psycho17Class::whiteCurveMode());
      args.psycho17ConeResponseExponent = Psycho17Class::coneResponseExponent();
      args.psycho17GamutCompression     = Psycho17Class::gamutCompression();
      args.psycho17GamutCompressionMode = static_cast<uint32_t>(Psycho17Class::gamutCompressionMode());
      args.psycho17Pad0                 = 0.f;
      args.psycho17Pad1                 = 0.f;
    }

    void populateTonemapOperatorArgs(ToneMappingApplyToneMappingArgs& args) {
      const TonemapOperator op = RtxForkGlobalTonemap::tonemapOperator();
      args.tonemapOperator    = static_cast<uint32_t>(op);
      writeOperatorParams<RtxForkAgX, RtxForkLottes, RtxForkPsycho17>(args, op);
    }

    // Combo items string uses ImGui's \0-separated format. User-requested
    // dropdown label for the Psycho operator is `PsychoV17_Beta`.
    static const char* k_operatorItems = "None\0Hill ACES\0Narkowicz ACES\0Hable Filmic\0AgX\0Lottes\0PsychoV17_Beta\0Gran Turismo 7\0Neutwo\0\0";

    // Shared slider rendering for per-operator parameter panels.
    static void showHableFilmicSliders() {
      ImGui::Indent();
      ImGui::Text("Hable Filmic Parameters:");
      // Presets shape the A-F+W curve only; exposureBias is a fork-added control
      // that gmod's reference presets don't cover, so it is deliberately untouched.
      if (ImGui::Button("Preset: Uncharted 2")) {
        RtxForkHableFilmic::shoulderStrengthObject().setDeferred(0.15f);
        RtxForkHableFilmic::linearStrengthObject()  .setDeferred(0.50f);
        RtxForkHableFilmic::linearAngleObject()     .setDeferred(0.10f);
        RtxForkHableFilmic::toeStrengthObject()     .setDeferred(0.20f);
        RtxForkHableFilmic::toeNumeratorObject()    .setDeferred(0.02f);
        RtxForkHableFilmic::toeDenominatorObject()  .setDeferred(0.30f);
        RtxForkHableFilmic::whitePointObject()      .setDeferred(11.2f);
      }
      ImGui::SameLine();
      if (ImGui::Button("Preset: Half-Life: Alyx")) {
        RtxForkHableFilmic::shoulderStrengthObject().setDeferred(0.319f);
        RtxForkHableFilmic::linearStrengthObject()  .setDeferred(0.5047f);
        RtxForkHableFilmic::linearAngleObject()     .setDeferred(0.1619f);
        RtxForkHableFilmic::toeStrengthObject()     .setDeferred(0.4667f);
        RtxForkHableFilmic::toeNumeratorObject()    .setDeferred(0.0f);
        RtxForkHableFilmic::toeDenominatorObject()  .setDeferred(0.7475f);
        RtxForkHableFilmic::whitePointObject()      .setDeferred(3.9996f);
      }
      RemixGui::DragFloat("Exposure Bias",     &RtxForkHableFilmic::exposureBiasObject(),     0.05f,  0.0f,  8.0f, "%.2f");
      RemixGui::DragFloat("Shoulder Strength", &RtxForkHableFilmic::shoulderStrengthObject(), 0.005f, 0.0f,  1.0f, "%.4f");
      RemixGui::DragFloat("Linear Strength",   &RtxForkHableFilmic::linearStrengthObject(),   0.005f, 0.0f,  1.0f, "%.4f");
      RemixGui::DragFloat("Linear Angle",      &RtxForkHableFilmic::linearAngleObject(),      0.005f, 0.0f,  1.0f, "%.4f");
      RemixGui::DragFloat("Toe Strength",      &RtxForkHableFilmic::toeStrengthObject(),      0.005f, 0.0f,  1.0f, "%.4f");
      RemixGui::DragFloat("Toe Numerator",     &RtxForkHableFilmic::toeNumeratorObject(),     0.001f, 0.0f,  0.5f, "%.4f");
      RemixGui::DragFloat("Toe Denominator",   &RtxForkHableFilmic::toeDenominatorObject(),   0.005f, 0.0f,  1.0f, "%.4f");
      RemixGui::DragFloat("White Point",       &RtxForkHableFilmic::whitePointObject(),       0.1f,   0.1f, 20.0f, "%.4f");
      ImGui::Unindent();
    }

    template<typename AgXClass>
    static void showAgXSlidersImpl(float minValue) {
      ImGui::Indent();
      ImGui::Text("AgX Controls:");
      ImGui::Separator();
      RemixGui::DragFloat("Saturation", &AgXClass::saturationObject(), 0.01f, 0.0f, 2.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
      ImGui::Separator();
      RemixGui::Combo(    "Look",       &AgXClass::lookObject(),        "None\0Golden\0Punchy\0\0");
      ImGui::Unindent();
    }

    static void showGlobalAgXSliders() { showAgXSlidersImpl<RtxForkAgX>(0.0f); }

    template<typename LottesClass>
    static void showLottesSlidersImpl() {
      ImGui::Indent();
      ImGui::Text("Lottes 2016 Parameters:");
      ImGui::Separator();
      RemixGui::DragFloat("HDR Max",         &LottesClass::hdrMaxObject(),   0.5f,   1.0f,  64.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Lottes Contrast", &LottesClass::contrastObject(), 0.01f,  1.0f,   3.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Shoulder",        &LottesClass::shoulderObject(), 0.01f,  0.5f,   2.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Mid In",          &LottesClass::midInObject(),    0.005f, 0.01f,  1.0f, "%.4f", ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Mid Out",         &LottesClass::midOutObject(),   0.005f, 0.01f,  1.0f, "%.4f", ImGuiSliderFlags_AlwaysClamp);
      ImGui::Unindent();
    }

    static void showGlobalLottesSliders() { showLottesSlidersImpl<RtxForkLottes>     (); }

    template<typename Psycho17Class>
    static void showPsycho17SlidersImpl() {
      ImGui::Indent();
      ImGui::Text("PsychoV17_Beta Parameters:");
      ImGui::Separator();
      RemixGui::DragFloat("Exposure",               &Psycho17Class::exposureObject(),             0.01f,  0.01f,  10.f, "%.3f",  ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Highlights",             &Psycho17Class::highlightsObject(),           0.01f,  0.0f,    5.f, "%.3f",  ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Shadows",                &Psycho17Class::shadowsObject(),              0.01f,  0.0f,    5.f, "%.3f",  ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Contrast",               &Psycho17Class::contrastObject(),             0.01f,  0.0f,    5.f, "%.3f",  ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Purity Scale",           &Psycho17Class::purityScaleObject(),          0.01f,  0.0f,    5.f, "%.3f",  ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Bleaching Intensity",    &Psycho17Class::bleachingIntensityObject(),   0.01f,  0.0f,    1.f, "%.3f",  ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Hue Restore",            &Psycho17Class::hueRestoreObject(),           0.01f,  0.0f,    1.f, "%.3f",  ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Cone Response Exponent", &Psycho17Class::coneResponseExponentObject(), 0.01f,  0.01f,  10.f, "%.3f",  ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Gamut Compression",      &Psycho17Class::gamutCompressionObject(),     0.01f,  0.0f,    1.f, "%.3f",  ImGuiSliderFlags_AlwaysClamp);
      RemixGui::Combo(    "Gamut Compression Mode", &Psycho17Class::gamutCompressionModeObject(), "BT.709\0BT.2020\0\0");
      ImGui::Unindent();
    }

    static void showGlobalPsycho17Sliders() { showPsycho17SlidersImpl<RtxForkPsycho17>     (); }

    void showTonemapOperatorUI() {
      RemixGui::Combo("Tonemapping Operator",
                      &RtxForkGlobalTonemap::tonemapOperatorObject(),
                      k_operatorItems);

      const TonemapOperator op = RtxForkGlobalTonemap::tonemapOperator();
      if (op == TonemapOperator::HableFilmic)  { showHableFilmicSliders();   }
      if (op == TonemapOperator::AgX)          { showGlobalAgXSliders();     }
      if (op == TonemapOperator::Lottes)       { showGlobalLottesSliders();  }
      if (op == TonemapOperator::Psycho17)     { showGlobalPsycho17Sliders(); }
    }

  } // namespace fork_hooks
} // namespace dxvk
