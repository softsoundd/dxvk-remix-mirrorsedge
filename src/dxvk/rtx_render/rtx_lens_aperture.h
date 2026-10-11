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
#pragma once

#include <cstdint>
#include <vector>

#include "rtx_option.h"
#include "rtx_lens_system.h"

namespace dxvk {

  // What sees the image. The human eye replaces the camera's lens altogether.
  enum class LensObserver : int {
    Camera = 0,
    Eye,
  };

  // The lens prescriptions of rtx_lens_prescriptions.h.
  enum class LensDesign : int {
    DoubleGauss50mm = 0,
    WideAngle22mm,
    UltraWide18mm,
  };

  // The camera's lens, its iris, prescription and coatings and the format it images onto, shared by the convolution
  // bloom and the lens flare. The iris's GPU counterpart is rtx/utility/lens_aperture.slangh.
  struct LensAperture {
    RTX_OPTION("rtx.lens", LensObserver, observer, LensObserver::Camera,
               "What sees the image: 0: Camera (this lens's diffraction, scatter and ghosts), 1: Eye (the human eye's\n"
               "diffraction and scatter, rtx.lens.eye.*, in place of the camera's lens, with no ghosts).");
    RTX_OPTION_ARGS("rtx.lens", float, fNumber, 4.0f,
                    "Aperture of the camera's lens. Stopping down grows the diffraction starburst and, past\n"
                    "rtx.lens.apertureCircularFNumber, makes the iris more polygonal. The lens flare is limited to its lens's widest\n"
                    "aperture.",
                    args.minValue = 1.0f, args.maxValue = 32.0f);
    RTX_OPTION_ARGS("rtx.lens", int, apertureBlades, 6,
                    "Number of blades of the camera's iris, 3 to 16. Below 3 the iris is circular. An iris of N blades diffracts\n"
                    "N spikes when N is even and 2N when it is odd. Shapes the convolution bloom's diffraction kernel and the\n"
                    "lens flare's ghosts.",
                    args.minValue = 0, args.maxValue = 16);
    RTX_OPTION_ARGS("rtx.lens", float, apertureCircularFNumber, 2.8f,
                    "f-number down to which the curved iris blades form a circle. Stopped down past it they leave a polygon with\n"
                    "arcs for sides, which straighten as the iris closes and bring out the starburst. 0 for straight blades.",
                    args.minValue = 0.0f, args.maxValue = 32.0f);
    RTX_OPTION("rtx.lens", float, apertureRotation, 15.0f, "Rotation of the iris in degrees.");
    RTX_OPTION("rtx.lens", LensDesign, prescription, LensDesign::UltraWide18mm,
               "Lens prescription, on a full frame sensor: 0: Double Gauss 50mm f/2 (Tronnier, US 2,673,491), 1: Wide Angle\n"
               "22mm f/2.8 (Nakamura), 2: Ultra Wide 18mm f/2.8 (Fujie, US 4,690,517), which matches a 90 degree horizontal view.\n"
               "The lens flare traces its ghosts, and the bloom spreads the light they carry as the lens's veil.");
    RTX_OPTION("rtx.lens", LensCoating, coating, LensCoating::SingleLayer,
               "Anti-reflection coating of the lens's air to glass surfaces, which sets the ghosts' brightness and colour and the\n"
               "veil they leave: 0: None (bare glass), 1: Single Layer (a quarter wave of MgF2), 2: Multilayer (quarter, half and\n"
               "quarter waves of MgF2, ZrO2 and an oxide whose index is matched to each glass).");
    RTX_OPTION_ARGS("rtx.lens", float, coatingWavelengthNm, 550.0f,
                    "Wavelength the coating's layers are a quarter (or half) wave thick at. The ghosts take the colour of the\n"
                    "wavelengths furthest from it.",
                    args.minValue = 380.0f, args.maxValue = 780.0f);
    RTX_OPTION_ARGS("rtx.lens", float, sensorReflectance, 0.02f,
                    "Reflectance of the sensor, which reflects light back into the lens and adds the ghosts between the sensor and\n"
                    "the rear elements. 0 disables them.",
                    args.minValue = 0.0f, args.maxValue = 1.0f);
    RTX_OPTION_ARGS("rtx.lens", float, surfaceRoughnessNm, 1.5f,
                    "RMS roughness of the lens's polished surfaces in nm. Each scatters (2 pi sigma dn / lambda)^2 of the light into\n"
                    "a narrow halo, more in blue.",
                    args.minValue = 0.0f, args.maxValue = 20.0f);
    RTX_OPTION_ARGS("rtx.lens", float, mechanicalScatter, 0.012f,
                    "Share of the light the barrel, mount, element edges, dust and camera body scatter into a wide veil. With the\n"
                    "coatings' inter-reflections it makes up the lens's veiling glare index, about 1.5% for a modern multicoated\n"
                    "lens, 2-6% for a single coated one and 10% or more uncoated (Kondo et al., JCII, 1981).",
                    args.minValue = 0.0f, args.maxValue = 0.5f);
    RTX_OPTION_ARGS("rtx.lens", float, apertureBladeTolerance, 0.01f,
                    "Placement tolerance of the camera's iris blades, RMS, as a share of the iris's widest opening radius. The\n"
                    "iris grows less regular as it closes, so its spikes and the ghosts' polygons vary slightly. 0 for a perfect\n"
                    "iris.",
                    args.minValue = 0.0f, args.maxValue = 0.1f);
    RTX_OPTION_ARGS("rtx.lens", float, apertureEdgeRoughness, 0.5f,
                    "RMS contrast of the streaks the iris blades' rough edges leave along the diffraction spikes. The sun's disc\n"
                    "blurs them away near the sun. 0 for perfectly smooth blades.",
                    args.minValue = 0.0f, args.maxValue = 2.0f);

    // The regular iris in units of its corner radius.
    struct Shape {
      bool circular = true;
      float area = 0.0f;
      // Radius of the circle of the same area. The f-number sets the iris's area, so this is the f-number's radius
      // and the corners lie 1 / areaRadius of it out.
      float areaRadius = 1.0f;
    };

    // Returns the blade count the shaders use, 0 for a circular iris.
    static uint32_t getBladeCount() {
      const int blades = apertureBlades();
      return blades >= 3 ? static_cast<uint32_t>(blades) : 0u;
    }

    static float getRotationRadians() {
      return apertureRotation() * (3.14159265358979323846f / 180.0f);
    }

    static bool isEye() {
      return observer() == LensObserver::Eye;
    }

    // The iris's inradius over its blades' arc radius at the given f-number: 0 for straight sides, 1 for a circle.
    static float getCurvature(float fNumber);

    static Shape getShape(float curvature);

    // A side of the iris with its blades off their regular places, in units of the regular iris's corner radius.
    struct Side {
      // The middle of the directions the side's normals turn through, and half their range, in radians.
      float centerAngle = 0.0f;
      float halfRange = 0.0f;
      float length = 0.0f;
    };

    struct Sides {
      std::vector<Side> sides;
      float area = 0.0f;
      // Distance from the centre of the farthest corner.
      float cornerReach = 1.0f;
    };

    // The RMS spreads of the blades' sides at an opening of stopRadius against the widest, maxStopRadius: their distances
    // from the centre in units of the inradius, and their normals' angles in radians. The misplacement is fixed in mm,
    // so it grows against the opening as the iris closes.
    static void getJitter(double stopRadius, double maxStopRadius, float& distanceJitter, float& angleJitter);

    // The iris's sides with its blades off their regular places, as lensApertureSignedDistance in
    // rtx/utility/lens_aperture.slangh draws it. None for a circular iris.
    static Sides getSides(float curvature, float distanceJitter, float angleJitter);

    // One of the waves that make up the streaks a blade's rough edge leaves across its spike, fixed for the blade: its
    // angular frequency in radians^-1, for a period between the given ones in degrees spread evenly on a log scale,
    // and its phase.
    static void getStreakWave(uint32_t side, uint32_t wave, float shortestPeriodDegrees, float longestPeriodDegrees,
                              float& frequency, float& phase);

    // Half the height, in mm, of the full frame (36 by 24 mm) sensor cropped to the screen's aspect ratio, as a
    // camera crops it for video.
    static float getSensorHalfHeightMm(float aspectRatio);

    // Half the height, in mm, of the image the screen shows: the full frame crop cropped further to the game's field of
    // view while the lens covers it, so that rays enter at their true angles, and otherwise the whole crop.
    static float getImageHalfHeightMm(float aspectRatio, float focalLengthMm, float tanHalfFovY);

    // Brings the shared lens system up to date with the options. Returns whether its ghosts changed.
    static bool updateLensSystem(LensSystem& lensSystem);

    static void showImguiSettings();
  };

}
