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
#include <complex>
#include <iterator>
#include <limits>

#include "rtx_lens_system.h"
#include "rtx_lens_prescriptions.h"
#include "rtx_lens_spectrum.h"

namespace dxvk {

  namespace {
    constexpr double kPiDouble = 3.14159265358979323846;

    // The lens is focused at infinity for this wavelength, in micrometres.
    constexpr double kFocusWavelength = 0.55;

    // The Fraunhofer d, F and C lines that n_d and the Abbe number are defined at, in micrometres.
    constexpr double kWavelengthD = 0.5876;
    constexpr double kWavelengthF = 0.4861;
    constexpr double kWavelengthC = 0.6563;

    constexpr double kMgF2Index = 1.38;
    constexpr double kZrO2Index = 2.10;

    // The least radius of a ghost's disc on the sensor, in mm, as the aberrations still blur a focused ghost.
    constexpr double kMinVeilRadiusMm = 0.01;

    // The least thickness a lens element or the spacer between two keeps at its edge, in mm, thin enough that it only
    // removes light a real lens could not pass.
    constexpr double kMinEdgeThicknessMm = 0.05;

    // Distances from the image's centre the veil averages its sources over.
    constexpr uint32_t kVeilPositions = 24;

    // Real rays start this far in front of the first vertex, clear of its surface's sag, as in lens_flare_trace.comp.slang.
    constexpr double kEntranceDistanceMm = 40.0;
    // A ray's path to the next surface gives its loss margin once shorter than this, in mm, as in the trace shader.
    constexpr double kBehindMarginMm = 0.1;
    // The loss margin over which a fitted ghost fades out as its rays near being lost.
    constexpr double kFadeMarginWidth = 0.1;
    // A real ray's distance inside a clear aperture's edge, in mm, per unit of loss margin, so that a fitted ghost fades
    // over its rays' last 0.5 mm.
    constexpr double kClipMarginMm = 5.0;

    // An aperture's paraxial image on the entrance plane: a disc whose centre moves along the source's meridian with
    // the slope of its rays.
    struct EntranceDisc {
      double centrePerSlope;
      double radius;
    };

    // Leaves out the discs that hold another for every slope up to maxSlope, which never bound what passes.
    void pruneEntranceDiscs(std::vector<EntranceDisc>& discs, double maxSlope) {
      const auto holds = [maxSlope](const EntranceDisc& outer, const EntranceDisc& inner) {
        return std::abs(inner.centrePerSlope - outer.centrePerSlope) * maxSlope + inner.radius <= outer.radius;
      };

      std::vector<EntranceDisc> kept;

      for (size_t k = 0; k < discs.size(); k++) {
        bool bounding = true;

        for (size_t j = 0; j < discs.size() && bounding; j++) {
          // Of two discs that hold each other, the first is kept.
          bounding = j == k || !holds(discs[k], discs[j]) || (holds(discs[j], discs[k]) && k < j);
        }

        if (bounding) {
          kept.push_back(discs[k]);
        }
      }

      discs.swap(kept);
    }

    // The area where every disc overlaps at the given slope, and its centroid along the meridian.
    void computeOverlap(const std::vector<EntranceDisc>& discs, double slope, double& area, double& centroid) {
      area = 0.0;
      centroid = 0.0;
      double low = -std::numeric_limits<double>::max();
      double high = std::numeric_limits<double>::max();

      for (const EntranceDisc& disc : discs) {
        low = std::max(low, disc.centrePerSlope * slope - disc.radius);
        high = std::min(high, disc.centrePerSlope * slope + disc.radius);
      }

      if (low >= high) {
        return;
      }

      // Across the meridian the overlap reaches as far as its narrowest disc does.
      constexpr uint32_t kSteps = 48;
      const double step = (high - low) / double(kSteps);
      double moment = 0.0;

      for (uint32_t i = 0; i < kSteps; i++) {
        const double x = low + (double(i) + 0.5) * step;
        double halfChord = std::numeric_limits<double>::max();

        for (const EntranceDisc& disc : discs) {
          const double d = x - disc.centrePerSlope * slope;
          halfChord = std::min(halfChord, std::sqrt(std::max(disc.radius * disc.radius - d * d, 0.0)));
        }

        area += 2.0 * halfChord * step;
        moment += 2.0 * halfChord * step * x;
      }

      centroid = moment / area;
    }

    struct CoatingLayer {
      double index;
      // Optical thickness in quarter waves of the design wavelength.
      double quarterWaves;
    };

    RayTransferMatrix operator*(const RayTransferMatrix& m, const RayTransferMatrix& n) {
      RayTransferMatrix r;
      r.a = m.a * n.a + m.b * n.c;
      r.b = m.a * n.b + m.b * n.d;
      r.c = m.c * n.a + m.d * n.c;
      r.d = m.c * n.b + m.d * n.d;
      return r;
    }

    RayTransferMatrix translation(double distance) {
      RayTransferMatrix m;
      m.b = distance;
      return m;
    }

    // Refraction from index n1 into n2 at a surface of the given radius (0 for flat), for light travelling toward
    // the image.
    RayTransferMatrix refraction(double n1, double n2, double radius) {
      RayTransferMatrix m;
      m.c = radius == 0.0 ? 0.0 : (n1 - n2) / (n2 * radius);
      m.d = n1 / n2;
      return m;
    }

    // How far a surface of the given radius (0 for flat) lies behind its vertex at a height from the axis, or not a
    // number past its hemisphere's rim.
    double surfaceSag(double radius, double height) {
      if (radius == 0.0) {
        return 0.0;
      }

      const double inside = radius * radius - height * height;
      return inside < 0.0 ? std::numeric_limits<double>::quiet_NaN() : radius - std::copysign(std::sqrt(inside), radius);
    }

    // Reflection off a surface of the given radius. Angles stay measured against the axis, so they change sign.
    RayTransferMatrix reflection(double radius) {
      RayTransferMatrix m;
      m.c = radius == 0.0 ? 0.0 : -2.0 / radius;
      m.d = -1.0;
      return m;
    }

    RayTransferMatrix inverse(const RayTransferMatrix& m) {
      const double determinant = m.a * m.d - m.b * m.c;
      RayTransferMatrix r;
      r.a = m.d / determinant;
      r.b = -m.b / determinant;
      r.c = -m.c / determinant;
      r.d = m.a / determinant;
      return r;
    }

    // Cauchy's two term fit through n_d and the F to C dispersion the Abbe number implies.
    double glassIndex(double indexD, double abbe, double wavelength) {
      if (indexD <= 1.0) {
        return 1.0;
      }

      if (abbe <= 0.0) {
        return indexD;
      }

      const double dispersion = (indexD - 1.0) / abbe;
      const double b = dispersion / (1.0 / (kWavelengthF * kWavelengthF) - 1.0 / (kWavelengthC * kWavelengthC));
      const double a = indexD - b / (kWavelengthD * kWavelengthD);

      return a + b / (wavelength * wavelength);
    }

    // Reflectance of a stack of thin films between the incident medium and the substrate, by Macleod's
    // characteristic matrices, averaged over s and p polarisation. The layers are listed from the incident side
    // unless reversed is set. Total internal reflection follows from the complex cosines.
    double stackReflectance(double incidentIndex, double substrateIndex, const CoatingLayer* pLayers, uint32_t layerCount,
                            bool reversed, double designWavelength, double wavelength, double incidence) {
      using Complex = std::complex<double>;

      const Complex invariant = incidentIndex * std::sin(incidence);
      const auto cosine = [&](double index) {
        const Complex s = invariant / index;
        return std::sqrt(Complex(1.0) - s * s);
      };

      double reflectance = 0.0;

      for (uint32_t polarisation = 0; polarisation < 2; polarisation++) {
        const auto admittance = [&](double index, Complex cosTheta) {
          return polarisation == 0 ? index * cosTheta : index / cosTheta;
        };

        Complex m11 = 1.0;
        Complex m12 = 0.0;
        Complex m21 = 0.0;
        Complex m22 = 1.0;

        for (uint32_t i = 0; i < layerCount; i++) {
          const CoatingLayer& layer = pLayers[reversed ? layerCount - 1 - i : i];
          const double thickness = layer.quarterWaves * designWavelength / (4.0 * layer.index);
          const Complex cosTheta = cosine(layer.index);
          const Complex delta = 2.0 * kPiDouble / wavelength * layer.index * thickness * cosTheta;
          const Complex eta = admittance(layer.index, cosTheta);
          const Complex i1(0.0, 1.0);

          const Complex l11 = std::cos(delta);
          const Complex l12 = i1 * std::sin(delta) / eta;
          const Complex l21 = i1 * eta * std::sin(delta);
          const Complex l22 = std::cos(delta);

          const Complex n11 = m11 * l11 + m12 * l21;
          const Complex n12 = m11 * l12 + m12 * l22;
          const Complex n21 = m21 * l11 + m22 * l21;
          const Complex n22 = m21 * l12 + m22 * l22;
          m11 = n11;
          m12 = n12;
          m21 = n21;
          m22 = n22;
        }

        const Complex etaSubstrate = admittance(substrateIndex, cosine(substrateIndex));
        const Complex etaIncident = admittance(incidentIndex, cosine(incidentIndex));
        const Complex b = m11 + m12 * etaSubstrate;
        const Complex c = m21 + m22 * etaSubstrate;
        const Complex r = (etaIncident * b - c) / (etaIncident * b + c);

        reflectance += 0.5 * std::norm(r);
      }

      return std::clamp(reflectance, 0.0, 1.0);
    }
  }

  LensSystem::LensSystem() {
    float wavelengthsNm[LENS_FLARE_WAVELENGTHS];
    computeVisibleSpectrumSamples(LENS_FLARE_WAVELENGTHS, wavelengthsNm, m_wavelengthWeights.data());

    for (uint32_t w = 0; w < LENS_FLARE_WAVELENGTHS; w++) {
      m_wavelengths[w] = double(wavelengthsNm[w]) * 1e-3;
    }

    m_wavelengths[kFocus] = kFocusWavelength;
  }

  bool LensSystem::update(uint32_t prescription, LensCoating coating, float coatingWavelengthNm, float sensorReflectance) {
    prescription = std::min(prescription, static_cast<uint32_t>(std::size(kLensPrescriptions) - 1));

    const bool changed =
      prescription != m_prescription ||
      coating != m_coating ||
      coatingWavelengthNm != m_coatingWavelengthNm ||
      sensorReflectance != m_sensorReflectance;

    if (prescription != m_prescription) {
      buildModel(prescription);
    }

    m_coating = coating;
    m_coatingWavelengthNm = coatingWavelengthNm;
    m_sensorReflectance = sensorReflectance;

    return changed;
  }

  double LensSystem::getVertexZ(uint32_t surface) const {
    return surface >= m_surfaceCount ? m_sensorZ : m_vertexZ[surface];
  }

  double LensSystem::getRadius(uint32_t surface) const {
    // The sensor, past the last surface, is flat.
    if (surface >= m_surfaceCount) {
      return 0.0;
    }

    return kLensPrescriptions[m_prescription].surfaces[surface].radius;
  }

  double LensSystem::getClearRadius(uint32_t surface) const {
    if (surface >= m_surfaceCount) {
      return 1e6;
    }

    return m_clearRadius[surface];
  }

  double LensSystem::getIndexBefore(uint32_t surface, uint32_t wavelength) const {
    return surface == 0 ? 1.0 : m_indexAfter[wavelength][surface - 1];
  }

  double LensSystem::getIndexAfter(uint32_t surface, uint32_t wavelength) const {
    return surface >= m_surfaceCount ? 1.0 : m_indexAfter[wavelength][surface];
  }

  void LensSystem::buildModel(uint32_t prescription) {
    const LensPrescription& lens = kLensPrescriptions[prescription];

    m_prescription = prescription;
    m_surfaceCount = lens.surfaceCount;
    m_vertexZ.assign(lens.surfaceCount, 0.0);

    for (auto& indices : m_indexAfter) {
      indices.clear();
    }

    double z = 0.0;

    for (uint32_t k = 0; k < lens.surfaceCount; k++) {
      const LensSurface& surface = lens.surfaces[k];
      m_vertexZ[k] = z;
      z += surface.thickness;

      if (surface.isStop) {
        m_stopIndex = k;
        m_maxStopRadius = 0.5 * surface.clearDiameter;
      }

      for (uint32_t w = 0; w < kWavelengthCount; w++) {
        m_indexAfter[w].push_back(glassIndex(surface.indexD, surface.abbe, m_wavelengths[w]));
      }
    }

    // A real lens's elements and the spacers between them keep some thickness at their edges, which a listed clear
    // diameter can leave out where deeply curved surfaces meet. Each clear radius stops where the glass or air to
    // either neighbour would thin below kMinEdgeThicknessMm, or below half its axial thickness where that is less, so
    // that no ray crosses surfaces that would overlap. The stop's opening is the iris's own.
    m_clearRadius.resize(lens.surfaceCount);

    // A surface ends at its hemisphere's rim.
    const auto rim = [&](uint32_t k) {
      return getRadius(k) == 0.0 ? std::numeric_limits<double>::max() : std::abs(getRadius(k));
    };

    for (uint32_t k = 0; k < lens.surfaceCount; k++) {
      m_clearRadius[k] = std::min(0.5 * lens.surfaces[k].clearDiameter, rim(k));
    }

    // Each pair of neighbours only meets within both of their rims. Past the nearer, the other surface goes on alone,
    // as the front of an element may reach beyond its back.
    for (uint32_t k = 0; k + 1 < lens.surfaceCount; k++) {
      const double gap = lens.surfaces[k].thickness;
      const double threshold = std::min(kMinEdgeThicknessMm, 0.5 * gap);
      const double reach = std::min({ std::max(m_clearRadius[k], m_clearRadius[k + 1]), rim(k), rim(k + 1) });
      constexpr int kSteps = 1000;

      for (int i = 1; i <= kSteps; i++) {
        const double height = reach * double(i) / double(kSteps);
        const double edge = gap + surfaceSag(getRadius(k + 1), height) - surfaceSag(getRadius(k), height);

        if (!(edge >= threshold)) {
          const double limit = reach * double(i - 1) / double(kSteps);

          for (uint32_t j = k; j <= k + 1; j++) {
            if (!lens.surfaces[j].isStop) {
              m_clearRadius[j] = std::min(m_clearRadius[j], limit);
            }
          }

          break;
        }
      }
    }

    m_frontRadius = m_clearRadius[0];

    // The lens is focused at infinity. The system matrix from the entrance plane to just behind the last surface gives
    // the focal length, and the distance behind it where a collimated beam converges.
    RayTransferMatrix system;
    RayTransferMatrix toStop;

    for (uint32_t k = 0; k < lens.surfaceCount; k++) {
      system = refraction(getIndexBefore(k, kFocus), getIndexAfter(k, kFocus), getRadius(k)) * system;

      if (k == m_stopIndex) {
        toStop = system;
      }

      if (k + 1 < lens.surfaceCount) {
        system = translation(m_vertexZ[k + 1] - m_vertexZ[k]) * system;
      }
    }

    m_focalLength = -1.0 / system.c;
    m_sensorZ = m_vertexZ[lens.surfaceCount - 1] - system.a / system.c;
    m_stopFromPupil = std::abs(toStop.a);
    m_imageFromSlope = (translation(m_sensorZ - m_vertexZ[lens.surfaceCount - 1]) * system).b;
  }

  bool LensSystem::traceGhost(uint32_t first, uint32_t second, uint32_t wavelength, Ghost& ghost) const {
    const uint32_t count = m_surfaceCount;
    const uint32_t stop = m_stopIndex;

    // The stop reflects nothing, and with a reflection on either side of it the light would cross it three times,
    // which the iris almost always blocks.
    if (first == stop || second == stop || (first < stop && second > stop)) {
      return false;
    }

    RayTransferMatrix m;
    RayTransferMatrix toStop;
    bool crossed = false;

    // A ray from entrance height h at slope u meets each surface at m.a * h + m.b * u, which its clear aperture bounds.
    const auto clip = [&](uint32_t surface, const RayTransferMatrix& toSurface) {
      if (wavelength != kFocus || surface >= count || surface == stop) {
        return;
      }

      ghost.apertures.push_back({ toSurface.a, toSurface.b, getClearRadius(surface) });

      if (std::abs(toSurface.a) >= 1e-9) {
        ghost.apertureLimit = std::min(ghost.apertureLimit, getClearRadius(surface) / std::abs(toSurface.a));
      }
    };

    if (wavelength == kFocus) {
      ghost.apertureLimit = m_frontRadius;
      ghost.apertures.clear();
    }

    // The ray heads toward the image, to the second reflection.
    for (uint32_t k = 0; k < second; k++) {
      clip(k, m);
      m = refraction(getIndexBefore(k, wavelength), getIndexAfter(k, wavelength), getRadius(k)) * m;

      if (k == stop) {
        toStop = m;
        crossed = true;
      }

      m = translation(getVertexZ(k + 1) - getVertexZ(k)) * m;
    }

    ghost.toSecond[wavelength] = m;
    clip(second, m);
    m = reflection(getRadius(second)) * m;

    // It heads back toward the object, to the first reflection.
    double z = getVertexZ(second);

    for (uint32_t k = second - 1; k > first; k--) {
      m = translation(getVertexZ(k) - z) * m;
      z = getVertexZ(k);
      clip(k, m);
      m = inverse(refraction(getIndexBefore(k, wavelength), getIndexAfter(k, wavelength), getRadius(k))) * m;
    }

    m = translation(getVertexZ(first) - z) * m;
    ghost.toFirst[wavelength] = m;
    clip(first, m);
    m = reflection(getRadius(first)) * m;

    // It heads toward the image again, to the sensor.
    z = getVertexZ(first);

    for (uint32_t k = first + 1; k < count; k++) {
      m = translation(getVertexZ(k) - z) * m;
      z = getVertexZ(k);
      clip(k, m);
      m = refraction(getIndexBefore(k, wavelength), getIndexAfter(k, wavelength), getRadius(k)) * m;

      if (k == stop) {
        toStop = m;
        crossed = true;
      }
    }

    m = translation(m_sensorZ - z) * m;

    ghost.toStop[wavelength] = toStop;
    ghost.toSensor[wavelength] = m;

    return crossed && std::abs(toStop.a) > 1e-9;
  }

  double LensSystem::computeReflectance(uint32_t surface, uint32_t wavelength, double incidence, bool fromFront) const {
    if (surface >= m_surfaceCount) {
      return m_sensorReflectance;
    }

    const double before = getIndexBefore(surface, wavelength);
    const double after = getIndexAfter(surface, wavelength);
    const double incidentIndex = fromFront ? before : after;
    const double substrateIndex = fromFront ? after : before;
    const double wavelengthNm = m_wavelengths[wavelength] * 1000.0;
    const double designWavelengthNm = m_coatingWavelengthNm;

    // Only the air to glass surfaces are coated. The cemented ones are bare.
    const bool airInterface = std::min(before, after) <= 1.0;
    CoatingLayer layers[3];
    uint32_t layerCount = 0;

    if (airInterface && m_coating == LensCoating::SingleLayer) {
      layers[layerCount++] = { kMgF2Index, 1.0 };
    } else if (airInterface && m_coating == LensCoating::Multilayer) {
      // The inner quarter wave's index cancels the reflection of the quarter wave pair at the design wavelength:
      // n_inner = n_MgF2 * sqrt(n_glass). The half wave between them is absent there, and widens the band.
      const LensPrescription& lens = kLensPrescriptions[m_prescription];
      const double glassIndexD = std::max(lens.surfaces[surface].indexD, surface > 0 ? lens.surfaces[surface - 1].indexD : 1.0);

      layers[layerCount++] = { kMgF2Index, 1.0 };
      layers[layerCount++] = { kZrO2Index, 2.0 };
      layers[layerCount++] = { kMgF2Index * std::sqrt(glassIndexD), 1.0 };
    }

    // The layers are listed from the air side, so light arriving from the glass meets them in reverse.
    const bool reversed = incidentIndex > 1.0;

    return stackReflectance(incidentIndex, substrateIndex, layers, layerCount, reversed, designWavelengthNm, wavelengthNm, incidence);
  }

  double LensSystem::incidenceAngle(const RayTransferMatrix& toSurface, double height, double angle, double radius) {
    const double h = toSurface.a * height + toSurface.b * angle;
    const double u = toSurface.c * height + toSurface.d * angle;
    const double curvature = radius == 0.0 ? 0.0 : 1.0 / radius;

    return std::min(std::abs(std::atan(u + h * curvature)), 0.5 * kPiDouble - 1e-3);
  }

  std::vector<LensSystem::Ghost> LensSystem::traceGhosts(double minSensorFromPupil) const {
    std::vector<Ghost> ghosts;

    if (m_prescription == ~0u) {
      return ghosts;
    }

    const uint32_t count = m_surfaceCount;
    const uint32_t lastSecond = m_sensorReflectance > 0.0f ? count : count - 1;

    for (uint32_t second = 1; second <= lastSecond; second++) {
      for (uint32_t first = 0; first < second; first++) {
        Ghost ghost;
        ghost.first = first;
        ghost.second = second;

        bool valid = true;

        for (uint32_t w = 0; w < kWavelengthCount; w++) {
          valid = valid && traceGhost(first, second, w, ghost);
        }

        if (!valid) {
          continue;
        }

        // Between its reflections the ghost crosses each surface three times where the main path, which the image is
        // normalised by, crosses it once.
        for (uint32_t w = 0; w < kWavelengthCount; w++) {
          double transmission = 1.0;

          for (uint32_t k = first + 1; k < second; k++) {
            const double transmitted = 1.0 - computeReflectance(k, w, 0.0, true);
            transmission *= transmitted * transmitted;
          }

          ghost.transmission[w] = transmission;
        }

        // A ghost shows by its radiance, reflectance / Mf11^2 up to a common factor. A large ghost can carry more
        // energy and still be invisible.
        const double reflectance = computeReflectance(second, kFocus, 0.0, true) * computeReflectance(first, kFocus, 0.0, false) *
                                   ghost.transmission[kFocus];
        const double sensorFromPupil = std::max(std::abs(ghost.toSensor[kFocus].a), minSensorFromPupil);
        ghost.brightness = reflectance / (sensorFromPupil * sensorFromPupil);

        ghosts.push_back(ghost);
      }
    }

    std::sort(ghosts.begin(), ghosts.end(), [](const Ghost& a, const Ghost& b) {
      return a.brightness > b.brightness;
    });

    return ghosts;
  }

  double LensSystem::traceRay(const Ghost& ghost, uint32_t wavelength, double x, double y, double rayAngle, double sensor[2]) const {
    const double angle = std::atan(rayAngle);
    double direction[3] = { std::sin(angle), 0.0, std::cos(angle) };
    double origin[3] = { x - direction[0] * kEntranceDistanceMm / direction[2], y, -kEntranceDistanceMm };
    double normal[3];
    double margin = 1.0;
    bool crossedStop = false;

    // Moves the ray to surface k's cap near its vertex, or to the sensor past the last surface, with the normal there
    // facing the object side at the vertex. The loss margin takes how near the ray comes to missing a sphere or running
    // along a plane, to meeting the surface behind it, and to the clear aperture's edge, which the trace shader's clip
    // ratio stops it at.
    const auto intersect = [&](uint32_t k) {
      const double vertexZ = getVertexZ(k);
      const double radius = getRadius(k);
      double t;

      if (radius == 0.0) {
        margin = std::min(margin, std::abs(direction[2]) - 1e-6);

        if (margin < 0.0) {
          return false;
        }

        t = (vertexZ - origin[2]) / direction[2];
      } else {
        const double toOrigin[3] = { origin[0], origin[1], origin[2] - vertexZ - radius };
        const double b = toOrigin[0] * direction[0] + toOrigin[1] * direction[1] + toOrigin[2] * direction[2];
        const double discriminant = b * b - (toOrigin[0] * toOrigin[0] + toOrigin[1] * toOrigin[1] + toOrigin[2] * toOrigin[2] - radius * radius);
        margin = std::min(margin, discriminant / (radius * radius));

        if (margin < 0.0) {
          return false;
        }

        const double t0 = -b - std::sqrt(discriminant);
        const double t1 = -b + std::sqrt(discriminant);
        t = std::abs(origin[2] + t0 * direction[2] - vertexZ) < std::abs(origin[2] + t1 * direction[2] - vertexZ) ? t0 : t1;
      }

      margin = std::min(margin, (t + 1e-3) / kBehindMarginMm);

      if (margin < 0.0) {
        return false;
      }

      for (uint32_t i = 0; i < 3; i++) {
        origin[i] += direction[i] * t;
      }

      normal[0] = radius == 0.0 ? 0.0 : origin[0] / radius;
      normal[1] = radius == 0.0 ? 0.0 : origin[1] / radius;
      normal[2] = radius == 0.0 ? -1.0 : (origin[2] - vertexZ - radius) / radius;

      if (k < m_surfaceCount && k != m_stopIndex) {
        margin = std::min(margin, (getClearRadius(k) - std::sqrt(origin[0] * origin[0] + origin[1] * origin[1])) / kClipMarginMm);
      }

      crossedStop = crossedStop || k == m_stopIndex;
      return margin >= 0.0;
    };

    // Each crossing must leave the ray heading the way its path goes, toward the image when forward is set. The loss
    // margin takes the squared cosine of a refracted ray's angle, which falls to 0 at the critical angle, and how far
    // short of turning back the ray heads.
    const auto turn = [&](bool forward) {
      margin = std::min(margin, (forward ? direction[2] : -direction[2]) - 1e-4);
      return margin >= 0.0;
    };

    const auto refract = [&](double n1, double n2, bool forward) {
      const double side = direction[0] * normal[0] + direction[1] * normal[1] + direction[2] * normal[2] > 0.0 ? -1.0 : 1.0;
      const double cosIncident = -side * (direction[0] * normal[0] + direction[1] * normal[1] + direction[2] * normal[2]);
      const double eta = n1 / n2;
      const double k = 1.0 - eta * eta * (1.0 - cosIncident * cosIncident);
      margin = std::min(margin, k);

      if (margin < 0.0) {
        return false;
      }

      for (uint32_t i = 0; i < 3; i++) {
        direction[i] = eta * direction[i] + (eta * cosIncident - std::sqrt(k)) * side * normal[i];
      }

      return turn(forward);
    };

    const auto reflect = [&](bool forward) {
      const double along = 2.0 * (direction[0] * normal[0] + direction[1] * normal[1] + direction[2] * normal[2]);

      for (uint32_t i = 0; i < 3; i++) {
        direction[i] -= along * normal[i];
      }

      return turn(forward);
    };

    // The ray heads toward the sensor to the second reflection, back to the first, and toward the sensor again.
    bool passes = true;

    for (uint32_t k = 0; k < ghost.second && passes; k++) {
      passes = intersect(k) && refract(getIndexBefore(k, wavelength), getIndexAfter(k, wavelength), true);
    }

    passes = passes && intersect(ghost.second) && reflect(false);

    for (uint32_t k = ghost.second - 1; k > ghost.first && passes; k--) {
      passes = intersect(k) && refract(getIndexAfter(k, wavelength), getIndexBefore(k, wavelength), false);
    }

    passes = passes && intersect(ghost.first) && reflect(true);

    for (uint32_t k = ghost.first + 1; k < m_surfaceCount && passes; k++) {
      passes = intersect(k) && refract(getIndexBefore(k, wavelength), getIndexAfter(k, wavelength), true);
    }

    if (!passes || !intersect(m_surfaceCount) || !crossedStop) {
      return -1.0;
    }

    sensor[0] = origin[0];
    sensor[1] = origin[1];
    return margin;
  }

  double LensSystem::mapGhost(const Ghost& ghost, double rayAngle, double height, double halfWidth,
                              std::array<GhostMap, LENS_FLARE_WAVELENGTHS>& maps) const {
    constexpr uint32_t kReference = LENS_FLARE_WAVELENGTHS / 2;
    const RayTransferMatrix& referenceToSensor = ghost.toSensor[kReference];

    // Central differences along the meridian, a quarter of the half width of what passes either side of its middle.
    // Across it, the lens's mirror symmetry about the meridian puts the central ray's sensor point on it, so one ray
    // suffices. Where those rays are lost, the maps stay paraxial about the axis.
    const double step = std::max(0.25 * halfWidth, 1e-3);
    double sensor[4][2];
    const double margin = std::min({
      traceRay(ghost, kReference, height, 0.0, rayAngle, sensor[0]),
      traceRay(ghost, kReference, height + step, 0.0, rayAngle, sensor[1]),
      traceRay(ghost, kReference, height - step, 0.0, rayAngle, sensor[2]),
      traceRay(ghost, kReference, height, step, rayAngle, sensor[3]),
    });
    GhostMap reference;
    reference.scaleX = referenceToSensor.a;
    reference.scaleY = referenceToSensor.a;
    reference.offset = referenceToSensor.b * rayAngle;

    if (margin > 0.0) {
      reference.scaleX = (sensor[1][0] - sensor[2][0]) / (2.0 * step);
      reference.scaleY = sensor[3][1] / step;
      reference.offset = sensor[0][0] - reference.scaleX * height;
    }

    for (uint32_t w = 0; w < LENS_FLARE_WAVELENGTHS; w++) {
      const double scale = ghost.toSensor[w].a - referenceToSensor.a;
      maps[w].scaleX = reference.scaleX + scale;
      maps[w].scaleY = reference.scaleY + scale;
      maps[w].offset = reference.offset + (ghost.toSensor[w].b - referenceToSensor.b) * rayAngle;
    }

    const double survival = std::clamp(margin / kFadeMarginWidth, 0.0, 1.0);
    return survival * survival * (3.0 - 2.0 * survival);
  }

  LensSystem::Veil LensSystem::computeVeil(const std::vector<Ghost>& ghosts, double stopRadius, double imageHalfHeightMm,
                                           double aspectRatio) const {
    Veil veil;

    if (m_prescription == ~0u || stopRadius <= 0.0 || imageHalfHeightMm <= 0.0) {
      return veil;
    }

    const double pupilRadius = std::min(stopRadius / m_stopFromPupil, m_frontRadius);
    const double pupilArea = kPiDouble * pupilRadius * pupilRadius;

    // Sources spread evenly over the image, by their distance from its centre, which is all a ghost's passing region
    // and offset depend on: a quarter of it on a grid, binned by radius.
    std::array<double, kVeilPositions> positionRadius {};
    std::array<double, kVeilPositions> positionShare {};
    const double halfWidth = imageHalfHeightMm * std::max(aspectRatio, 1e-3);
    const double maxRadius = std::hypot(halfWidth, imageHalfHeightMm);
    constexpr uint32_t kGridX = 64;
    constexpr uint32_t kGridY = 36;

    for (uint32_t j = 0; j < kGridY; j++) {
      for (uint32_t i = 0; i < kGridX; i++) {
        const double radius = std::hypot((double(i) + 0.5) / kGridX * halfWidth, (double(j) + 0.5) / kGridY * imageHalfHeightMm);
        const uint32_t bin = std::min(uint32_t(radius / maxRadius * kVeilPositions), kVeilPositions - 1);
        positionRadius[bin] += radius;
        positionShare[bin] += 1.0;
      }
    }

    for (uint32_t b = 0; b < kVeilPositions; b++) {
      positionRadius[b] = positionShare[b] > 0.0 ? positionRadius[b] / positionShare[b] : 0.0;
      positionShare[b] /= double(kGridX * kGridY);
    }

    veil.imagesPerGhost = kVeilPositions;
    const double maxSlope = maxRadius / m_imageFromSlope;
    std::vector<EntranceDisc> discs;

    for (const Ghost& ghost : ghosts) {
      Vector3 reflectance(0.0f);

      for (uint32_t w = 0; w < LENS_FLARE_WAVELENGTHS; w++) {
        const double r = computeReflectance(ghost.second, w, 0.0, true) * computeReflectance(ghost.first, w, 0.0, false) *
                         ghost.transmission[w];
        reflectance = reflectance + m_wavelengthWeights[w] * float(r);
      }

      // The stop's and every clear aperture's image on the entrance plane. A surface that every ray meets at the same
      // height passes the slopes up to its radius over that height's slope.
      const RayTransferMatrix& toStop = ghost.toStop[kFocus];
      double slopeLimit = std::numeric_limits<double>::max();
      discs.clear();
      discs.push_back({ -toStop.b / toStop.a, stopRadius / std::abs(toStop.a) });

      for (const Aperture& aperture : ghost.apertures) {
        if (std::abs(aperture.a) >= 1e-9) {
          discs.push_back({ -aperture.b / aperture.a, aperture.radius / std::abs(aperture.a) });
        } else if (std::abs(aperture.b) > 0.0) {
          slopeLimit = std::min(slopeLimit, aperture.radius / std::abs(aperture.b));
        }
      }

      pruneEntranceDiscs(discs, maxSlope);

      // For each source position, the light the ghost's path lets in against the main path's pupil, which can be
      // more, and where its image lands: its passing region mapped onto the sensor, against the source's image.
      const RayTransferMatrix& toSensor = ghost.toSensor[kFocus];
      const size_t first = veil.images.size();
      double total = 0.0;

      for (uint32_t b = 0; b < kVeilPositions; b++) {
        VeilImage image;
        const double slope = positionRadius[b] / m_imageFromSlope;
        double area = 0.0;
        double centroid = 0.0;

        if (positionShare[b] > 0.0 && slope <= slopeLimit) {
          computeOverlap(discs, slope, area, centroid);
        }

        if (area > 0.0) {
          const double share = area / pupilArea * positionShare[b];
          image.offset = float(std::abs(toSensor.a * centroid + toSensor.b * slope - m_imageFromSlope * slope));
          image.radius = float(std::max(std::abs(toSensor.a) * std::sqrt(area / kPiDouble), kMinVeilRadiusMm));
          image.share = float(share);
          total += share;
        }

        veil.images.push_back(image);
      }

      for (size_t i = first; i < veil.images.size(); i++) {
        veil.images[i].share = total > 0.0 ? float(veil.images[i].share / total) : 0.0f;
      }

      const Vector3 energy = reflectance * float(total);
      veil.surfaces.push_back({ ghost.first, ghost.second });
      veil.energies.push_back(energy);
      veil.energy = veil.energy + energy;
    }

    return veil;
  }

}
