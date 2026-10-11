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
#include <algorithm>
#include <cmath>

#include "rtx_lens_aperture.h"
#include "rtx_eye_model.h"
#include "rtx_lens_prescriptions.h"
#include "rtx_imgui.h"

namespace dxvk {

  namespace {
    constexpr double kPiDouble = 3.14159265358979323846;

    // The full frame format.
    constexpr double kSensorWidthMm = 36.0;
    constexpr double kSensorHeightMm = 24.0;

    // Below this curvature the sides are taken as straight, as lensApertureCornerReach takes them.
    constexpr double kMinCurvature = 1e-4;

    // As lensApertureHash and lensApertureSideSpread in rtx/utility/lens_aperture.slangh.
    uint32_t apertureHash(uint32_t value) {
      value ^= value >> 16;
      value *= 0x7feb352du;
      value ^= value >> 15;
      value *= 0x846ca68bu;
      value ^= value >> 16;
      return value;
    }

    float apertureHashUnit(uint32_t value) {
      return float(apertureHash(value)) / 4294967295.0f;
    }

    void apertureSideSpread(uint32_t side, double& angle, double& distance) {
      const double scale = std::sqrt(6.0);
      angle = (double(apertureHashUnit(side * 4u + 1u)) + double(apertureHashUnit(side * 4u + 2u)) - 1.0) * scale;
      distance = (double(apertureHashUnit(side * 4u + 3u)) + double(apertureHashUnit(side * 4u + 4u)) - 1.0) * scale;
    }

    // The regular iris's corner radius in units of its inradius, as lensApertureCornerRadius.
    double getCornerRadius(uint32_t blades, double q) {
      const double halfSector = kPiDouble / double(blades);
      const double straightness = 1.0 - q;
      const double sine = straightness * std::sin(halfSector);
      return (2.0 - q) / (std::sqrt(1.0 - sine * sine) + straightness * std::cos(halfSector));
    }

    RemixGui::ComboWithKey<LensObserver> observerCombo {
      "Observer##lens",
      RemixGui::ComboWithKey<LensObserver>::ComboEntries { {
        { LensObserver::Camera, "Camera" },
        { LensObserver::Eye, "Eye" },
      } }
    };

    RemixGui::ComboWithKey<LensDesign> prescriptionCombo {
      "Lens##lens",
      RemixGui::ComboWithKey<LensDesign>::ComboEntries { {
        { LensDesign::DoubleGauss50mm, kLensPrescriptions[0].name },
        { LensDesign::WideAngle22mm, kLensPrescriptions[1].name },
        { LensDesign::UltraWide18mm, kLensPrescriptions[2].name },
      } }
    };

    RemixGui::ComboWithKey<LensCoating> coatingCombo {
      "Coating##lens",
      RemixGui::ComboWithKey<LensCoating>::ComboEntries { {
        { LensCoating::None, "None" },
        { LensCoating::SingleLayer, "Single Layer (MgF2)" },
        { LensCoating::Multilayer, "Multilayer (MgF2, ZrO2, matched oxide)" },
      } }
    };
  }

  bool LensAperture::updateLensSystem(LensSystem& lensSystem) {
    return lensSystem.update(static_cast<uint32_t>(prescription()), coating(), coatingWavelengthNm(), sensorReflectance());
  }

  float LensAperture::getCurvature(float fNumber) {
    const float circularFNumber = apertureCircularFNumber();

    if (circularFNumber <= 0.0f) {
      return 0.0f;
    }

    return std::min(circularFNumber / std::max(fNumber, 0.5f), 1.0f);
  }

  LensAperture::Shape LensAperture::getShape(float curvature) {
    Shape shape;
    const uint32_t blades = getBladeCount();
    const double q = std::clamp(double(curvature), 0.0, 1.0);

    if (blades == 0 || q >= 1.0) {
      shape.circular = true;
      shape.area = float(kPiDouble);
      return shape;
    }

    // In units of the inradius first.
    const double halfSector = kPiDouble / double(blades);
    const double cornerRadius = getCornerRadius(blades, q);
    double segmentArea = 0.0;

    if (q > kMinCurvature) {
      // Each side's disc has radius 1 / q and its centre lies (1 / q - 1) behind the origin, so its arc subtends twice
      // this angle there.
      const double arcRadius = 1.0 / q;
      const double arcHalfAngle = std::atan2(cornerRadius * std::sin(halfSector), cornerRadius * std::cos(halfSector) + arcRadius - 1.0);
      segmentArea = 0.5 * arcRadius * arcRadius * (2.0 * arcHalfAngle - std::sin(2.0 * arcHalfAngle));
    }

    const double area = double(blades) * (0.5 * cornerRadius * cornerRadius * std::sin(2.0 * halfSector) + segmentArea);
    const double normalizedArea = area / (cornerRadius * cornerRadius);

    shape.circular = false;
    shape.area = float(normalizedArea);
    shape.areaRadius = float(std::sqrt(normalizedArea / kPiDouble));
    return shape;
  }

  void LensAperture::getJitter(double stopRadius, double maxStopRadius, float& distanceJitter, float& angleJitter) {
    const double tolerance = std::max(double(apertureBladeTolerance()), 0.0);
    distanceJitter = float(tolerance * maxStopRadius / std::max(stopRadius, 1e-6));
    angleJitter = float(tolerance);
  }

  LensAperture::Sides LensAperture::getSides(float curvature, float distanceJitter, float angleJitter) {
    Sides result;
    const uint32_t blades = getBladeCount();
    const double q = std::clamp(double(curvature), 0.0, 1.0);

    if (blades == 0 || q >= 1.0) {
      return result;
    }

    const double sector = 2.0 * kPiDouble / double(blades);
    const double cornerRadius = getCornerRadius(blades, q);
    const double rotation = double(getRotationRadians());

    // Each side's normal and its distance from the centre, in units of the regular iris's inradius.
    std::vector<double> normalAngles(blades);
    std::vector<double> distances(blades);

    for (uint32_t j = 0; j < blades; j++) {
      double angleSpread;
      double distanceSpread;
      apertureSideSpread(j, angleSpread, distanceSpread);
      normalAngles[j] = rotation + (double(j) + 0.5) * sector + angleSpread * double(angleJitter);
      distances[j] = 1.0 + distanceSpread * double(distanceJitter);
    }

    // Corner j, where side j - 1 meets side j near the regular corner at rotation + j * sector.
    const bool straight = q <= kMinCurvature;
    const double arcRadius = straight ? 0.0 : 1.0 / q;
    std::vector<double> cornerX(blades);
    std::vector<double> cornerY(blades);

    for (uint32_t j = 0; j < blades; j++) {
      const uint32_t i = (j + blades - 1) % blades;
      const double nix = std::cos(normalAngles[i]);
      const double niy = std::sin(normalAngles[i]);
      const double njx = std::cos(normalAngles[j]);
      const double njy = std::sin(normalAngles[j]);

      if (straight) {
        const double determinant = nix * njy - niy * njx;
        cornerX[j] = (distances[i] * njy - distances[j] * niy) / determinant;
        cornerY[j] = (nix * distances[j] - njx * distances[i]) / determinant;
      } else {
        const double cix = (distances[i] - arcRadius) * nix;
        const double ciy = (distances[i] - arcRadius) * niy;
        const double cjx = (distances[j] - arcRadius) * njx;
        const double cjy = (distances[j] - arcRadius) * njy;
        const double gapX = cjx - cix;
        const double gapY = cjy - ciy;
        const double gapLength = std::sqrt(gapX * gapX + gapY * gapY);
        const double across = std::sqrt(std::max(arcRadius * arcRadius - 0.25 * gapLength * gapLength, 0.0));
        const double perpendicularX = -gapY / gapLength;
        const double perpendicularY = gapX / gapLength;
        const double towardX = std::cos(rotation + double(j) * sector);
        const double towardY = std::sin(rotation + double(j) * sector);
        const double sign = perpendicularX * towardX + perpendicularY * towardY >= 0.0 ? 1.0 : -1.0;
        cornerX[j] = 0.5 * (cix + cjx) + perpendicularX * across * sign;
        cornerY[j] = 0.5 * (ciy + cjy) + perpendicularY * across * sign;
      }
    }

    // Each side runs from its corner j to corner j + 1. A curved side's normals turn through the angle its corners
    // subtend at its disc's centre.
    double polygonArea = 0.0;
    double segmentArea = 0.0;
    double reach = 0.0;
    result.sides.resize(blades);

    for (uint32_t j = 0; j < blades; j++) {
      const uint32_t next = (j + 1) % blades;
      polygonArea += 0.5 * (cornerX[j] * cornerY[next] - cornerY[j] * cornerX[next]);
      reach = std::max(reach, std::sqrt(cornerX[j] * cornerX[j] + cornerY[j] * cornerY[j]));
      Side& side = result.sides[j];

      if (straight) {
        side.centerAngle = float(normalAngles[j]);
        side.halfRange = 0.0f;
        side.length = float(std::hypot(cornerX[next] - cornerX[j], cornerY[next] - cornerY[j]) / cornerRadius);
      } else {
        const double centerX = (distances[j] - arcRadius) * std::cos(normalAngles[j]);
        const double centerY = (distances[j] - arcRadius) * std::sin(normalAngles[j]);
        const double start = std::atan2(cornerY[j] - centerY, cornerX[j] - centerX);
        const double end = std::atan2(cornerY[next] - centerY, cornerX[next] - centerX);
        double span = end - start;
        span -= 2.0 * kPiDouble * std::floor(span / (2.0 * kPiDouble));
        side.centerAngle = float(start + 0.5 * span);
        side.halfRange = float(0.5 * span);
        side.length = float(span / q / cornerRadius);
        segmentArea += 0.5 * (span - std::sin(span)) / (q * q);
      }
    }

    result.area = float((polygonArea + segmentArea) / (cornerRadius * cornerRadius));
    result.cornerReach = float(reach / cornerRadius);
    return result;
  }

  void LensAperture::getStreakWave(uint32_t side, uint32_t wave, float shortestPeriodDegrees, float longestPeriodDegrees,
                                   float& frequency, float& phase) {
    const uint32_t key = side * 64u + wave * 4u;
    const double shortest = double(shortestPeriodDegrees) * kPiDouble / 180.0;
    const double period = shortest * std::pow(double(longestPeriodDegrees) / double(shortestPeriodDegrees), double(apertureHashUnit(key + 101u)));
    frequency = float(2.0 * kPiDouble / period);
    phase = float(2.0 * kPiDouble * double(apertureHashUnit(key + 102u)));
  }

  float LensAperture::getSensorHalfHeightMm(float aspectRatio) {
    const double aspect = std::max(double(aspectRatio), 1e-3);
    return float(0.5 * std::min(kSensorHeightMm, kSensorWidthMm / aspect));
  }

  float LensAperture::getImageHalfHeightMm(float aspectRatio, float focalLengthMm, float tanHalfFovY) {
    return std::min(focalLengthMm * tanHalfFovY, getSensorHalfHeightMm(aspectRatio));
  }

  void LensAperture::showImguiSettings() {
    ImGui::Text("Lens (shared by the bloom and the lens flare):");
    ImGui::Indent();
    observerCombo.getKey(&observerObject());
    RemixGui::SetTooltipToLastWidgetOnHover("Camera: this lens's diffraction, scatter and ghosts. Eye: the human eye's, with no ghosts.");

    if (!isEye()) {
      prescriptionCombo.getKey(&prescriptionObject());
      RemixGui::DragFloat("f-number##lens", &fNumberObject(), 0.05f, 1.0f, 32.0f, "f/%.1f");
      RemixGui::SetTooltipToLastWidgetOnHover("Stopping down grows the diffraction starburst and makes the iris more polygonal.");
      RemixGui::SliderInt("Iris Blades", &apertureBladesObject(), 0, 16);
      RemixGui::SetTooltipToLastWidgetOnHover("Below 3 the iris is circular. N blades diffract N spikes when N is even and 2N when it is odd.");
      RemixGui::DragFloat("Iris Circular Down To", &apertureCircularFNumberObject(), 0.05f, 0.0f, 32.0f, "f/%.1f");
      RemixGui::SetTooltipToLastWidgetOnHover("The curved blades form a circle down to this f-number, and a polygon with arcs for sides past it.\n0 for straight blades, which give a starburst at any aperture.");
      RemixGui::DragFloat("Iris Rotation", &apertureRotationObject(), 0.5f, -180.0f, 180.0f, "%.1f deg");
      RemixGui::DragFloat("Iris Blade Tolerance", &apertureBladeToleranceObject(), 0.0005f, 0.0f, 0.1f, "%.4f");
      RemixGui::SetTooltipToLastWidgetOnHover("How far each blade sits off its regular place, RMS, as a share of the widest opening.\nThe iris grows less regular as it closes. 0 for a perfect iris.");
      RemixGui::DragFloat("Iris Edge Roughness", &apertureEdgeRoughnessObject(), 0.01f, 0.0f, 2.0f, "%.2f");
      RemixGui::SetTooltipToLastWidgetOnHover("The streaks the blades' rough edges leave along the diffraction spikes, as their RMS contrast.\n0 for perfectly smooth blades.");
      coatingCombo.getKey(&coatingObject());
      RemixGui::DragFloat("Coating Wavelength##lens", &coatingWavelengthNmObject(), 1.0f, 380.0f, 780.0f, "%.0f nm");
      RemixGui::SetTooltipToLastWidgetOnHover("The coating's layers are a quarter or half wave thick here. The ghosts take the colour of the\nwavelengths furthest from it.");
      RemixGui::DragFloat("Sensor Reflectance##lens", &sensorReflectanceObject(), 0.005f, 0.0f, 1.0f, "%.3f");
      RemixGui::DragFloat("Surface Roughness##lens", &surfaceRoughnessNmObject(), 0.05f, 0.0f, 20.0f, "%.2f nm");
      RemixGui::DragFloat("Mechanical Scatter##lens", &mechanicalScatterObject(), 0.001f, 0.0f, 0.5f, "%.3f");
      RemixGui::SetTooltipToLastWidgetOnHover("Light the barrel, mount, edges, dust and camera body scatter into a wide veil.");
    } else {
      RemixGui::DragFloat("Age##eye", &EyeModel::ageObject(), 0.5f, 1.0f, 100.0f, "%.0f years");
      RemixGui::SetTooltipToLastWidgetOnHover("Older eyes have smaller pupils and scatter more light.");
      RemixGui::DragFloat("Pigmentation##eye", &EyeModel::pigmentationObject(), 0.01f, 0.0f, 1.2f, "%.2f");
      RemixGui::SetTooltipToLastWidgetOnHover("0 for very dark eyes, 0.5 brown, 1 blue-green, 1.2 very light blue.");
      RemixGui::DragFloat("Fixed Pupil##eye", &EyeModel::pupilDiameterMmObject(), 0.05f, 0.0f, 8.0f, "%.2f mm");
      RemixGui::SetTooltipToLastWidgetOnHover("0 follows the field's luminance.");
      RemixGui::DragFloat("Luminance Scale##eye", &EyeModel::luminanceScaleObject(), 10.0f, 0.0f, 1e6f, "%.0f cd/m^2");
      RemixGui::SetTooltipToLastWidgetOnHover("Candela per square metre per unit of radiance. 0 takes it from the Physical Atmosphere's sun.");
      RemixGui::DragFloat("Constriction Time##eye", &EyeModel::constrictionSecondsObject(), 0.01f, 0.0f, 10.0f, "%.2f s");
      RemixGui::DragFloat("Dilation Time##eye", &EyeModel::dilationSecondsObject(), 0.05f, 0.0f, 60.0f, "%.2f s");
      RemixGui::DragFloat("Hippus##eye", &EyeModel::hippusObject(), 0.01f, 0.0f, 4.0f, "%.2f");
      RemixGui::SetTooltipToLastWidgetOnHover("The pupil's unrest, which makes the glare pulse.");
      RemixGui::SliderInt("Eyelashes##eye", &EyeModel::lashesObject(), 0, 16);
      RemixGui::DragFloat("Particle Density##eye", &EyeModel::particleDensityObject(), 0.01f, 0.0f, 8.0f, "%.2f");
      RemixGui::SetTooltipToLastWidgetOnHover("The lens and vitreous particles that cast the ciliary corona's needles.");
      RemixGui::DragFloat("Halo Strength##eye", &EyeModel::haloStrengthObject(), 0.01f, 0.0f, 10.0f, "%.2f");
      RemixGui::SetTooltipToLastWidgetOnHover("The lenticular halo, the coloured ring the lens fibres diffract around lights.");
      RemixGui::DragFloat("Update Rate##eye", &EyeModel::updateRateHzObject(), 0.1f, 1.0f, 60.0f, "%.1f Hz");
    }

    ImGui::Unindent();
  }

}
