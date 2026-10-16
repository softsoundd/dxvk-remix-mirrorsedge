/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#include "rtx_clouds.h"
#include "rtx_imgui.h"

namespace dxvk {

  namespace {
    RemixGui::ComboWithKey<CloudQuality> cloudQualityCombo {
      "Quality",
      RemixGui::ComboWithKey<CloudQuality>::ComboEntries { {
          {CloudQuality::Low, "Low"},
          {CloudQuality::Medium, "Medium"},
          {CloudQuality::High, "High"},
          {CloudQuality::Ultra, "Ultra"}
      } }
    };

    RemixGui::ComboWithKey<CloudMarchRate> cloudMarchRateCombo {
      "March Rate",
      RemixGui::ComboWithKey<CloudMarchRate>::ComboEntries { {
          {CloudMarchRate::Full, "Full"},
          {CloudMarchRate::Half, "Half"},
          {CloudMarchRate::Quarter, "Quarter"}
      } }
    };

    RemixGui::ComboWithKey<CloudGenus> cloudGenusCombo {
      "Genus",
      RemixGui::ComboWithKey<CloudGenus>::ComboEntries { {
          {CloudGenus::Custom, "Custom"},
          {CloudGenus::CumulusHumilis, "Cumulus Humilis"},
          {CloudGenus::CumulusMediocris, "Cumulus Mediocris"},
          {CloudGenus::CumulusCongestus, "Cumulus Congestus"},
          {CloudGenus::Stratocumulus, "Stratocumulus"},
          {CloudGenus::Altocumulus, "Altocumulus"},
          {CloudGenus::Stratus, "Stratus"}
      } }
    };
  }

  void RtxClouds::showImguiSettings(uint32_t sliderFlagsIn, uint32_t collapsingHeaderFlags) {
    const ImGuiSliderFlags sliderFlags = ImGuiSliderFlags(sliderFlagsIn);
    if (!RemixGui::CollapsingHeader("Clouds", ImGuiTreeNodeFlags(collapsingHeaderFlags))) {
      return;
    }
    ImGui::Indent();

    RemixGui::Checkbox("Enable Clouds", &enableObject());
    RemixGui::SetTooltipToLastWidgetOnHover("A volumetric cloud layer lit by the atmosphere's sun and sky, casting shadows on the scene and the\nair, and seen in reflections.");

    if (!enable()) {
      ImGui::Unindent();
      return;
    }

    cloudQualityCombo.getKey(&qualityObject());
    RemixGui::SetTooltipToLastWidgetOnHover("Step budget and length of the march, full resolution shadow taps, scattering octaves, how often the\noptical depth grids rebake and the reflection dome's resolution.");
    cloudMarchRateCombo.getKey(&marchRateObject());
    RemixGui::SetTooltipToLastWidgetOnHover("Share of the view's rays into the layer that march each frame, half in a checkerboard or a quarter in 2x2\nblocks, while the others keep their history. Each halving halves the cost of the view's march, and its\nhistory takes twice as many frames to settle.");

    cloudGenusCombo.getKey(&genusObject());
    RemixGui::SetTooltipToLastWidgetOnHover("Shape and microphysics presets after the WMO genera. Custom exposes the layer and microphysics options.");

    {
      const GenusSettings g = getGenusSettings();
      const float baseKm = g.baseAltitudeMeters * 0.001f;
      const float topKm = baseKm + g.thicknessMeters * 0.001f;
      const float eyeKm = getEyeAltitudeKm();
      ImGui::Text("Eye %.2f km, layer %.2f - %.2f km above sea level, %+.2f - %+.2f km from the eye (%s)", eyeKm, baseKm, topKm,
        baseKm - eyeKm, topKm - eyeKm, eyeKm < baseKm ? "below" : (eyeKm > topKm ? "above" : "inside"));
      RemixGui::SetTooltipToLastWidgetOnHover("Altitudes on the atmosphere's datum: its Altitude, plus the camera's height above Ground Level when\nAltitude Follows Camera is on. If the city sits in or above the layer, set Ground Level (or raise the layer)\nfor this map.");
    }

    if (genus() == CloudGenus::Custom) {
      ImGui::Separator();
      ImGui::Text("Layer:");
      RemixGui::DragFloat("Base Altitude", &baseAltitudeMetersObject(), 10.0f, 0.0f, 8000.0f, "%.0f m", sliderFlags);
      RemixGui::SetTooltipToLastWidgetOnHover("Above sea level, on the datum of the atmosphere's Altitude.");
      RemixGui::DragFloat("Thickness", &thicknessMetersObject(), 10.0f, 50.0f, 8000.0f, "%.0f m", sliderFlags);
      RemixGui::DragFloat("Coverage", &coverageObject(), 0.005f, 0.0f, 1.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Type (Wispy - Billowy)", &cloudTypeObject(), 0.005f, 0.0f, 1.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Cloud Spacing", &cellSizeKmObject(), 0.01f, 0.5f, 8.0f, "%.2f km", sliderFlags);
      RemixGui::SetTooltipToLastWidgetOnHover("At most 32 clouds fit across a Noise Tile, so the spacing is at least the tile / 32.");
      RemixGui::DragFloat("Top Shape", &columnTopShapeObject(), 0.005f, 0.05f, 4.0f, "%.3f", sliderFlags);
      RemixGui::SetTooltipToLastWidgetOnHover("Below 1, a cloud's top rises steeply from its edge into a dome; above 1, tops stay low until the core.");
      RemixGui::DragFloat("Top Variation", &columnTopVariationObject(), 0.005f, 0.0f, 1.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Base Variation", &columnBaseVariationObject(), 0.005f, 0.0f, 0.5f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Edge Feather", &columnFeatherObject(), 0.005f, 0.02f, 1.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Top Flatten", &columnTopFlattenObject(), 0.005f, 0.05f, 1.0f, "%.3f", sliderFlags);
      RemixGui::SetTooltipToLastWidgetOnHover("Caps the tops at this fraction of the layer, as an inversion flattens stratocumulus. 1 = no cap.");

      ImGui::Text("Microphysics:");
      RemixGui::DragFloat("Liquid Water Content", &liquidWaterContentObject(), 0.005f, 0.01f, 3.0f, "%.3f g/m^3", sliderFlags);
      RemixGui::SetTooltipToLastWidgetOnHover("Liquid water 1 km above the base. Stratiform 0.2-0.4, cumulus 0.5-1, towering cumulus up to 2.");
      RemixGui::DragFloat("Droplet Concentration", &dropletConcentrationObject(), 1.0f, 10.0f, 2000.0f, "%.0f /cm^3", sliderFlags);
      RemixGui::SetTooltipToLastWidgetOnHover("More droplets for the same water make them smaller: brighter, denser clouds (the Twomey effect).\nMaritime ~100, continental 300-600, polluted urban 1000+.");
    }

    RemixGui::DragFloat("Coverage Variation", &coverageSpreadObject(), 0.005f, 0.0f, 1.0f, "%.3f", sliderFlags);
    RemixGui::DragFloat("Coverage Variation Scale", &coverageSpreadScaleKmObject(), 0.05f, 2.0f, 64.0f, "%.1f km", sliderFlags);
    RemixGui::SetTooltipToLastWidgetOnHover("Size of the patches of denser and sparser clouds.");
    RemixGui::DragFloat("Type Variation", &typeSpreadObject(), 0.005f, 0.0f, 1.0f, "%.3f", sliderFlags);
    RemixGui::DragFloat("Type Variation Scale", &typeSpreadScaleKmObject(), 0.05f, 0.5f, 32.0f, "%.1f km", sliderFlags);
    RemixGui::SetTooltipToLastWidgetOnHover("Size of the patches of wispier and more billowy clouds; about the cloud spacing gives each cloud its own\ncharacter.");
    RemixGui::DragFloat("Horizon Bias", &horizonBiasObject(), 0.005f, -1.0f, 1.0f, "%.3f", sliderFlags);
    RemixGui::SetTooltipToLastWidgetOnHover("Shrinks the clouds overhead (above 0; 1 clears them, leaving clouds only towards the horizon, like a\nskybox) or towards the horizon (below 0).");
    if (horizonBias() != 0.0f) {
      ImGui::Indent();
      RemixGui::DragFloat("Overhead Radius", &horizonBiasStartKmObject(), 0.1f, 0.0f, 100.0f, "%.1f km", sliderFlags);
      RemixGui::SetTooltipToLastWidgetOnHover("Horizontal distance from the camera within which the overhead side applies in full. Clouds 2 km above\nthe eye stand 10 degrees above the horizon 11 km away.");
      RemixGui::DragFloat("Horizon Distance", &horizonBiasEndKmObject(), 0.1f, 0.1f, 160.0f, "%.1f km", sliderFlags);
      RemixGui::SetTooltipToLastWidgetOnHover("Horizontal distance beyond which the horizon side applies in full; between the two the bias blends.");
      ImGui::Unindent();
    }

    {
      const DropletReadout readout = computeDropletReadout();
      ImGui::Text("Droplets: r_eff %.1f um, extinction %.0f /km at 1 km (mean free path %.0f m), g %.3f (truncated %.3f, f %.2f)",
        readout.effectiveRadiusUm, readout.extinctionPerKm, 1000.0f / std::max(readout.extinctionPerKm, 1e-3f), readout.asymmetry,
        readout.truncatedAsymmetry, readout.forwardPeakFraction);
      const float transportDepth = (1.0f - readout.asymmetry) * readout.verticalOpticalDepth;
      const float twoStreamAlbedo = 1.0f - 1.0f / (1.0f + std::max(diffuseTransmissionK(), 0.0f) * transportDepth);
      ImGui::Text("Full column: vertical optical depth %.0f, two-stream albedo %.2f", readout.verticalOpticalDepth, twoStreamAlbedo);
      RemixGui::SetTooltipToLastWidgetOnHover("Through the whole layer where the field's density is 1, as only cloud cores are. Observed: stratus and\nstratocumulus 5-20, fair weather cumulus 10-50, growing cumulus 100 and more. The albedo is the column's diffuse\nreflectance, 1 - 1 / (1 + k (1 - g) tau). Shadows on the ground follow exp(-(1 - f) tau) for the direct sun plus\nthe light the clouds pass on diffusely.");
    }
    RemixGui::DragFloat("Density Scale", &densityScaleObject(), 0.005f, 0.0f, 4.0f, "%.3f", sliderFlags);
    RemixGui::SetTooltipToLastWidgetOnHover("Stylisation: multiplies the physical extinction. 1 = physical.");

    ImGui::Separator();
    ImGui::Text("Motion:");
    RemixGui::Checkbox("Cloud Motion", &motionObject());
    RemixGui::SetTooltipToLastWidgetOnHover("Move the clouds: the wind carries them and they rise and shear as they travel. Off holds them where they\nare, keeping the settings below.");
    ImGui::BeginDisabled(!motion());
    RemixGui::DragFloat("Wind Speed", &windSpeedObject(), 0.1f, 0.0f, 50.0f, "%.1f m/s", sliderFlags);
    RemixGui::SetTooltipToLastWidgetOnHover("At cloud level: 5-15 m/s is typical, 25 and more a gale.");
    ImGui::EndDisabled();
    RemixGui::DragFloat("Wind Direction", &windDirectionObject(), 0.5f, 0.0f, 360.0f, "%.1f deg", sliderFlags);
    ImGui::BeginDisabled(!motion());
    RemixGui::DragFloat("Convective Rise", &evolutionRiseObject(), 0.05f, 0.0f, 10.0f, "%.2f m/s", sliderFlags);
    RemixGui::DragFloat("Top Shear", &evolutionShearObject(), 0.05f, 0.0f, 10.0f, "%.2f m/s", sliderFlags);
    ImGui::EndDisabled();

    ImGui::Separator();
    ImGui::Text("Scene and Atmosphere:");
    RemixGui::Checkbox("Cloud Shadows", &sunShadowsObject());
    RemixGui::SetTooltipToLastWidgetOnHover("The atmosphere's sun is attenuated by the layer for every surface, including through mirrors and in\nindirect light.");
    RemixGui::Checkbox("Shadows in the Air", &airShadowsObject());
    RemixGui::SetTooltipToLastWidgetOnHover("Cloud shadows in the aerial perspective and the volumetric fog: crepuscular rays beneath a broken deck.");
    RemixGui::DragFloat("Shadow Strength", &shadowStrengthObject(), 0.005f, 0.0f, 1.0f, "%.3f", sliderFlags);
    RemixGui::Checkbox("Aerial Perspective on Clouds", &skyAerialPerspectiveObject());
    RemixGui::SetTooltipToLastWidgetOnHover("See the clouds through the air in front of them, so distant clouds sink into the horizon haze and redden\nat sunset.");
    RemixGui::Checkbox("Clouds in Reflections", &reflectionDomeObject());
    RemixGui::SetTooltipToLastWidgetOnHover("Mirrors, glass and indirect sky light see the clouds, from a dome rendered from the camera.");
    RemixGui::Checkbox("Sharp Mirror and Glass Clouds", &mirrorReflectionMarchObject());
    RemixGui::SetTooltipToLastWidgetOnHover("March the clouds per pixel along mirrors' and glass's reflections into the sky, and past glass the sky is\nseen straight through, instead of the dome.");
    RemixGui::Checkbox("Sharp Near-Mirror Clouds", &glossyReflectionMarchObject());
    RemixGui::SetTooltipToLastWidgetOnHover("March the clouds per pixel along the reflections of surfaces a little too rough for PSR (polished floors,\ncurtain walls), instead of the dome, whose coarse texels flicker in them as the clouds drift.");
    if (glossyReflectionMarch()) {
      RemixGui::DragFloat("Near-Mirror Sharing Angle", &glossyShareAngleDegreesObject(), 0.1f, 0.0f, 180.0f, "%.1f deg", sliderFlags);
      RemixGui::SetTooltipToLastWidgetOnHover("A near-mirror reflection within this angle of the first in its 2x2 block of pixels takes the clouds that\none marched instead of marching its own. 0 marches every reflection.");
    }
    RemixGui::Checkbox("Path Traced Reference", &referenceObject());
    RemixGui::SetTooltipToLastWidgetOnHover("Replace the cloud layer with a progressive path traced reference to judge the march's lighting by\n(debug views Clouds Reference / Clouds Reference Error). Very expensive; the wind stops while it accumulates.");
    if (reference()) {
      ImGui::Indent();
      RemixGui::DragInt("Collisions per Frame", &referenceBouncesPerFrameObject(), 0.5f, 1, 1024, "%d", sliderFlags);
      RemixGui::SetTooltipToLastWidgetOnHover("How far each 4x4 pixel tile's path advances per frame: the reference's cost per frame, not its result.");
      RemixGui::DragInt("Exact Phase Collisions", &referenceExactBouncesObject(), 0.1f, 0, 16384, "%d", sliderFlags);
      RemixGui::SetTooltipToLastWidgetOnHover("Collisions by the droplets' exact Mie phase before a path continues in the delta-M similar medium\n(a third of the collisions, exact to second order for the diffuse light). Raise it for a strictly exact reference.");
      const ReferenceErrorReadout& error = getReferenceErrorReadout();
      if (error.tiles > 0) {
        ImGui::Text("March against reference: mean |error| %.0f%%, 95th percentile %.0f%%, bias %+.0f%% (%u opaque tiles, %.0f paths a tile)",
          error.meanAbsolute * 100.0f, error.percentile95 * 100.0f, error.bias * 100.0f, error.tiles, error.pathsPerTile);
        RemixGui::SetTooltipToLastWidgetOnHover("By luminance, over the 4x4 tiles both see as opaque cloud that have finished a few paths. The bias is the\nmarch's mean over the reference's, positive where the march is brighter; the per-tile errors include the\nreference's own noise, which falls as it accumulates.");
      } else {
        ImGui::TextDisabled("March against reference: accumulating");
      }
      ImGui::Unindent();
    }

    if (RemixGui::CollapsingHeader("Cloud Shape Detail", ImGuiTreeNodeFlags(collapsingHeaderFlags))) {
      ImGui::Indent();
      RemixGui::DragFloat("Noise Tile", &tileKmObject(), 0.05f, 4.0f, 32.0f, "%.2f km", sliderFlags);
      RemixGui::Checkbox("Hex De-Tiling", &hexTilingObject());
      RemixGui::DragFloat("Detail Scale", &detailScaleObject(), 0.05f, 2.0f, 32.0f, "%.2f", sliderFlags);
      RemixGui::DragFloat("Erosion", &erosionStrengthObject(), 0.005f, 0.0f, 2.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Sharpen", &sharpenStrengthObject(), 0.005f, 0.0f, 1.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Body Erosion (Bake)", &bodyErosionObject(), 0.005f, 0.0f, 1.5f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Shape Variety", &shapeVarietyKmObject(), 0.005f, 0.0f, 1.5f, "%.3f km", sliderFlags);
      RemixGui::SetTooltipToLastWidgetOnHover("Capped at 0.65 x the wavelength, beyond which the surface folds into detached sheets.");
      RemixGui::DragFloat("Shape Variety Wavelength", &shapeVarietyWavelengthKmObject(), 0.01f, 0.5f, 8.0f, "%.2f km", sliderFlags);
      RemixGui::DragFloat("Silhouette Wobble", &wobbleStrengthObject(), 0.005f, 0.0f, 2.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Interior Texture", &interiorTextureObject(), 0.005f, 0.0f, 1.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Edge Wisps", &edgeErosionObject(), 0.005f, 0.0f, 3.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Edge Detail", &edgeDetailObject(), 0.005f, 0.0f, 2.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Fine Detail", &fineDetailStrengthObject(), 0.005f, 0.0f, 2.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Curl Distortion", &curlStrengthMetersObject(), 0.5f, 0.0f, 500.0f, "%.0f m", sliderFlags);
      RemixGui::DragFloat("Near Detail", &nearDetailStrengthObject(), 0.005f, 0.0f, 2.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Near Detail Range", &nearDetailRangeKmObject(), 0.05f, 0.1f, 10.0f, "%.2f km", sliderFlags);
      RemixGui::DragFloat("Fly-Through Detail", &flyThroughDetailObject(), 0.005f, 0.0f, 3.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Detail LOD Bias", &detailLodBiasObject(), 0.05f, -4.0f, 2.0f, "%.2f", sliderFlags);
      RemixGui::DragFloat("Profile Depth", &profileDepthKmObject(), 0.005f, 0.05f, 4.0f, "%.3f km", sliderFlags);
      RemixGui::DragFloat("Coverage Offset", &coverageOffsetKmObject(), 0.005f, 0.0f, 2.0f, "%.3f km", sliderFlags);
      RemixGui::DragFloat("Adiabatic Exponent", &adiabaticExponentObject(), 0.005f, 0.0f, 2.0f, "%.3f", sliderFlags);
      RemixGui::SetTooltipToLastWidgetOnHover("The extinction grows as the height above the base to this power, between the profile's floor and max\n(relative to 1 km above the base).");
      RemixGui::DragFloat("Profile Floor", &adiabaticFloorObject(), 0.005f, 0.0f, 1.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Profile Max", &adiabaticMaxObject(), 0.01f, 0.25f, 5.0f, "%.2f", sliderFlags);
      ImGui::Unindent();
    }

    if (RemixGui::CollapsingHeader("Cloud Lighting", ImGuiTreeNodeFlags(collapsingHeaderFlags))) {
      ImGui::Indent();
      RemixGui::DragInt("Scattering Octaves (0 = Tier)", &multipleScatteringOctavesObject(), 0.1f, 0, 4, "%d", sliderFlags);
      RemixGui::DragFloat("Octave Extinction (a)", &msExtinctionFalloffObject(), 0.005f, 0.0f, 1.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Octave Energy (b)", &msEnergyFalloffObject(), 0.005f, 0.0f, 1.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Octave Asymmetry (c)", &msPhaseFalloffObject(), 0.005f, 0.0f, 1.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Diffusion Floor", &diffusionFloorObject(), 0.005f, 0.0f, 4.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Diffusion Anisotropy", &diffusionAnisotropyObject(), 0.005f, 0.0f, 2.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Diffuse Transmission k", &diffuseTransmissionKObject(), 0.005f, 0.0f, 4.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Sky Light", &ambientStrengthObject(), 0.005f, 0.0f, 4.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Ground Bounce", &groundBounceStrengthObject(), 0.005f, 0.0f, 4.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Ground Diffuse Share", &groundDiffuseShareObject(), 0.005f, 0.0f, 1.0f, "%.3f", sliderFlags);
      RemixGui::DragFloat("Shadow Tap Range", &shadowTapRangeMetersObject(), 1.0f, 0.0f, 2000.0f, "%.0f m", sliderFlags);
      RemixGui::ColorEdit3("Albedo Tint", &albedoTintObject());
      ImGui::Unindent();
    }

    ImGui::Unindent();
  }

} // namespace dxvk
