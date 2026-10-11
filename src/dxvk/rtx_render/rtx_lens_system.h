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

#include <array>
#include <cstdint>
#include <vector>

#include "../../util/util_vector.h"
#include "rtx/pass/lens_flare/lens_flare.h"

namespace dxvk {

  // Paraxial ray transfer matrix [a b; c d], mapping a ray's (height in mm, angle in radians).
  struct RayTransferMatrix {
    double a = 1.0;
    double b = 0.0;
    double c = 0.0;
    double d = 1.0;
  };

  enum class LensCoating : int {
    None = 0,
    SingleLayer,
    Multilayer,
  };

  /**
   * \brief The camera's lens prescription as an optical system.
   *
   * The prescription at the lens flare's wavelength samples, its anti-reflection coatings, and the paraxial transfer
   * of every ghost of two reflections, which the lens flare draws and the convolution bloom spreads as the lens's veil.
   */
  class LensSystem {
  public:
    // The wavelength samples, then the wavelength the lens is focused at.
    static constexpr uint32_t kFocus = LENS_FLARE_WAVELENGTHS;
    static constexpr uint32_t kWavelengthCount = LENS_FLARE_WAVELENGTHS + 1;

    // A clear aperture of radius `radius` mm that a ray from entrance height e mm at slope u meets at a e + b u.
    struct Aperture {
      double a = 1.0;
      double b = 0.0;
      double radius = 0.0;
    };

    // A ghost reflected first from surface second, then from surface first, which lies in front of it. The sensor
    // has the index getSurfaceCount().
    struct Ghost {
      uint32_t first = 0;
      uint32_t second = 0;
      std::array<RayTransferMatrix, kWavelengthCount> toStop;
      std::array<RayTransferMatrix, kWavelengthCount> toSensor;
      // From the entrance plane to just before each reflection, for the incidence of the ghost's central ray.
      std::array<RayTransferMatrix, kWavelengthCount> toFirst;
      std::array<RayTransferMatrix, kWavelengthCount> toSecond;
      // What the ghost's two extra passes through the surfaces between its reflections let through, beyond the main
      // path's one, at normal incidence.
      std::array<double, kWavelengthCount> transmission {};
      // Largest entrance height of an on-axis ray that clears every surface the ghost's path meets, the stop aside.
      double apertureLimit = 0.0;
      // Every clear aperture the ghost's path meets at the focus wavelength, the stop aside.
      std::vector<Aperture> apertures;
      // Radiance up to a common factor, reflectance / Mf11^2, on axis at the focus wavelength.
      double brightness = 0.0;
    };

    // One of a ghost's images for a source somewhere in the image: how far from the source it lands, its radius, both
    // in mm on the sensor, and the share of the ghost's light it carries.
    struct VeilImage {
      float offset = 0.0f;
      float radius = 0.0f;
      float share = 0.0f;
    };

    // The light every ghost carries relative to the image, averaged over where sources sit in it: a ghost forms only
    // for the sources whose light its path lets through, and lands away from its source.
    struct Veil {
      Vector3 energy = Vector3(0.0f);
      // Each ghost's surfaces and energy by channel, and its images, imagesPerGhost of them.
      std::vector<std::pair<uint32_t, uint32_t>> surfaces;
      std::vector<Vector3> energies;
      std::vector<VeilImage> images;
      uint32_t imagesPerGhost = 0;
    };

    LensSystem();

    // Rebuilds what the arguments change. Returns whether the ghosts changed.
    bool update(uint32_t prescription, LensCoating coating, float coatingWavelengthNm, float sensorReflectance);

    uint32_t getPrescription() const { return m_prescription; }
    uint32_t getSurfaceCount() const { return m_surfaceCount; }
    uint32_t getStopIndex() const { return m_stopIndex; }
    double getFocalLength() const { return m_focalLength; }
    double getSensorZ() const { return m_sensorZ; }
    double getMaxStopRadius() const { return m_maxStopRadius; }
    double getFrontRadius() const { return m_frontRadius; }
    // Height on the stop per height on the entrance plane of the main path at the focus wavelength.
    double getStopFromPupil() const { return m_stopFromPupil; }
    double getVertexZ(uint32_t surface) const;
    double getRadius(uint32_t surface) const;
    double getClearRadius(uint32_t surface) const;
    double getIndexBefore(uint32_t surface, uint32_t wavelength) const;
    double getIndexAfter(uint32_t surface, uint32_t wavelength) const;

    // In micrometres.
    const std::array<double, kWavelengthCount>& getWavelengths() const { return m_wavelengths; }
    const std::array<Vector3, LENS_FLARE_WAVELENGTHS>& getWavelengthWeights() const { return m_wavelengthWeights; }

    double computeReflectance(uint32_t surface, uint32_t wavelength, double incidence, bool fromFront) const;

    // The angle to a surface's normal of the ray that leaves the entrance plane at the given height and angle.
    static double incidenceAngle(const RayTransferMatrix& toSurface, double height, double angle, double radius);

    // Every ghost the stop lets through, brightest first. minSensorFromPupil bounds |Mf11| for the brightness.
    std::vector<Ghost> traceGhosts(double minSensorFromPupil) const;

    // A ghost's rays at one wavelength from the entrance plane to the sensor, in the frame of the sun's meridian, x along
    // it and y across it. A ray from entrance point (x, y) meets the sensor at (scaleX x + offset, scaleY y), in mm.
    struct GhostMap {
      double scaleX = 1.0;
      double scaleY = 1.0;
      double offset = 0.0;
    };

    // Fills each wavelength sample's map for the sun's rays at rayAngle to the axis, fitted to the middle wavelength's
    // real rays about entrance height `height` on the meridian, and offset for the others as their paraxial transfer
    // differs. halfWidth is the half width of what passes there.
    // Returns how much of the ghost's light survives the lens, 1 while those rays are well clear of being lost, easing
    // to 0 as the first of them nears it.
    double mapGhost(const Ghost& ghost, double rayAngle, double height, double halfWidth,
                    std::array<GhostMap, LENS_FLARE_WAVELENGTHS>& maps) const;

    // The veil of the ghosts with the stop at stopRadius, the radius the f-number sets, for sources spread evenly over
    // an image of the given half height in mm on the sensor and aspect ratio.
    Veil computeVeil(const std::vector<Ghost>& ghosts, double stopRadius, double imageHalfHeightMm, double aspectRatio) const;

  private:
    void buildModel(uint32_t prescription);
    bool traceGhost(uint32_t first, uint32_t second, uint32_t wavelength, Ghost& ghost) const;
    // Traces the ghost's real ray at a wavelength from entrance point (x, y) at rayAngle to the axis, in the frame of the
    // sun's meridian, to where it meets the sensor.
    // Returns how far the ray is from being lost, the least of lens_flare_trace.comp.slang's loss margins and its
    // distance inside each clear aperture, negative once lost.
    double traceRay(const Ghost& ghost, uint32_t wavelength, double x, double y, double rayAngle, double sensor[2]) const;

    std::array<double, kWavelengthCount> m_wavelengths {};
    std::array<Vector3, LENS_FLARE_WAVELENGTHS> m_wavelengthWeights {};

    uint32_t m_prescription = ~0u;
    LensCoating m_coating = LensCoating::None;
    float m_coatingWavelengthNm = 0.0f;
    float m_sensorReflectance = -1.0f;

    uint32_t m_surfaceCount = 0;
    uint32_t m_stopIndex = 0;
    std::vector<double> m_vertexZ;
    std::vector<double> m_clearRadius;
    std::array<std::vector<double>, kWavelengthCount> m_indexAfter;
    double m_focalLength = 0.0;
    double m_sensorZ = 0.0;
    double m_maxStopRadius = 0.0;
    double m_frontRadius = 0.0;
    double m_stopFromPupil = 1.0;
    // The image's height on the sensor per slope of the rays, in mm.
    double m_imageFromSlope = 1.0;
  };

}
