/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_atmosphere.h"
#include "rtx_atmosphere_aerosol_tables.h"
#include "dxvk_device.h"
#include "dxvk_context.h"
#include "rtx_options.h"
#include "rtx_context.h"
#include "rtx_camera.h"
#include "rtx_scene_manager.h"
#include "rtx_light_manager.h"
#include "rtx_lights.h"
#include "rtx_global_volumetrics.h"
#include "rtx_render/rtx_shader_manager.h"
#include "../../util/util_color.h"
#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/atmosphere/aerial_perspective_binding_indices.h"
#include <rtx_shaders/transmittance_lut.h>
#include <rtx_shaders/multiscattering_lut.h>
#include <rtx_shaders/sky_view_lut.h>
#include <rtx_shaders/sky_view_hemisphere_mean.h>
#include <rtx_shaders/aerial_perspective_lut.h>
#include <rtx_shaders/aerial_perspective_lut_rayquery.h>
#include <rtx_shaders/aerial_perspective_tile_depth.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace dxvk {
  // Shader definitions for atmosphere LUT generation. The transmittance and multiscattering LUTs
  // depend only on the atmosphere parameters, the sky-view LUT additionally on the sun, and the
  // aerial perspective volume on the camera frustum, so they are baked in that order.
  namespace {
    class TransmittanceLutShader : public ManagedShader {
      SHADER_SOURCE(TransmittanceLutShader, VK_SHADER_STAGE_COMPUTE_BIT, transmittance_lut)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        RW_TEXTURE2D(1)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(TransmittanceLutShader);

    class MultiscatteringLutShader : public ManagedShader {
      SHADER_SOURCE(MultiscatteringLutShader, VK_SHADER_STAGE_COMPUTE_BIT, multiscattering_lut)

      PUSH_CONSTANTS(AtmosphereMultiscatteringBakeArgs)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        TEXTURE2D(1)
        TEXTURE2D(2)
        TEXTURE2D(3)
        RW_TEXTURE2D(4)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(MultiscatteringLutShader);

    class SkyViewLutShader : public ManagedShader {
      SHADER_SOURCE(SkyViewLutShader, VK_SHADER_STAGE_COMPUTE_BIT, sky_view_lut)
      
      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        TEXTURE2D(1)
        TEXTURE2D(2)
        RW_TEXTURE2D(3)
        TEXTURE2D(4)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(SkyViewLutShader);

    class SkyViewHemisphereMeanShader : public ManagedShader {
      SHADER_SOURCE(SkyViewHemisphereMeanShader, VK_SHADER_STAGE_COMPUTE_BIT, sky_view_hemisphere_mean)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        TEXTURE2D(1)
        RW_TEXTURE2D(2)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(SkyViewHemisphereMeanShader);

    class AerialPerspectiveTileDepthShader : public ManagedShader {
      SHADER_SOURCE(AerialPerspectiveTileDepthShader, VK_SHADER_STAGE_COMPUTE_BIT, aerial_perspective_tile_depth)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        TEXTURE2D(1)
        RW_TEXTURE2D(2)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(AerialPerspectiveTileDepthShader);

    class AerialPerspectiveLutShader : public ManagedShader {
      SHADER_SOURCE(AerialPerspectiveLutShader, VK_SHADER_STAGE_COMPUTE_BIT, aerial_perspective_lut)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        TEXTURE2D(1)
        TEXTURE2D(2)
        RW_TEXTURE3D(3)
        TEXTURE2D(4)
        TEXTURE2D(5)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(AerialPerspectiveLutShader);

    // Ray query variant: reads the scene through the common ray tracing bindings (constants, TLAS and
    // atmosphere LUTs included) and reprojects the previous frame's volume.
    class AerialPerspectiveRayQueryShader : public ManagedShader {
      SHADER_SOURCE(AerialPerspectiveRayQueryShader, VK_SHADER_STAGE_COMPUTE_BIT, aerial_perspective_lut_rayquery)

      BINDLESS_ENABLED()

      BEGIN_PARAMETER()
        COMMON_RAYTRACING_BINDINGS
        TEXTURE3D(AERIAL_PERSPECTIVE_BINDING_PREV_LUT_INPUT)
        TEXTURE2D(AERIAL_PERSPECTIVE_BINDING_TILE_DEPTH_INPUT)
        TEXTURE2D(AERIAL_PERSPECTIVE_BINDING_PREV_TILE_DEPTH_INPUT)
        RW_TEXTURE3D(AERIAL_PERSPECTIVE_BINDING_LUT_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(AerialPerspectiveRayQueryShader);

    constexpr float kAtmPi = 3.14159265358979323846f;

    // Koschmieder's relation for a 2% contrast threshold: V = ln(50) / sigma_t.
    constexpr float kKoschmiederConstant = 3.912f;

    // Wavelengths the RGB channels stand for (Bruneton 2017, Section 14.3), in micrometers.
    constexpr double kLambdaR = 0.680;
    constexpr double kLambdaG = 0.550;
    constexpr double kLambdaB = 0.440;

    float atmSmoothstep(float e0, float e1, float x) {
      const float denom = e1 - e0;
      float t = (denom != 0.0f) ? (x - e0) / denom : 0.0f;
      t = std::min(std::max(t, 0.0f), 1.0f);
      return t * t * (3.0f - 2.0f * t);
    }

    Vector3 atmMul(const Vector3& a, const Vector3& b) {
      return Vector3(a.x * b.x, a.y * b.y, a.z * b.z);
    }

    float atmOzoneDensity(const AtmosphereArgs& a, float altitudeKm) {
      const float halfWidth = std::max(a.ozoneLayerWidth, 1e-3f);
      return std::max(0.0f, 1.0f - std::abs(altitudeKm - a.ozoneLayerAltitude) / halfWidth);
    }

    float atmExponentialDensity(float altitudeKm, float scaleHeight) {
      return std::exp(-std::max(altitudeKm, 0.0f) / std::max(scaleHeight, 1e-3f));
    }

    // CPU counterpart of mieDensity() in atmosphere_common.slangh.
    float atmMieDensity(const AtmosphereArgs& a, float altitudeKm) {
      if (a.mieBoundaryLayerHeight <= 0.0f) {
        return atmExponentialDensity(altitudeKm, a.mieScaleHeight);
      }

      const float aboveTop = altitudeKm - a.mieBoundaryLayerHeight;
      const float tail = a.mieBoundaryLayerTailScale * atmExponentialDensity(aboveTop, a.mieScaleHeight);
      const float transitionWidth = std::max(a.mieBoundaryLayerTransition, 1e-3f);
      const float t = atmSmoothstep(-transitionWidth, transitionWidth, aboveTop);

      return 1.0f + (tail - 1.0f) * t;
    }

    // Ray-sphere roots for a unit length direction, sorted. Returns false when the ray misses.
    bool atmIntersectSphere(
      const Vector3& origin, const Vector3& direction, const Vector3& center, float radius,
      float& outNear, float& outFar) {
      const Vector3 oc = origin - center;
      const float b = 2.0f * dot(oc, direction);
      const float c = dot(oc, oc) - radius * radius;
      const float discriminant = b * b - 4.0f * c;

      if (discriminant < 0.0f) {
        return false;
      }

      const float sqrtDiscriminant = std::sqrt(discriminant);
      outNear = (-b - sqrtDiscriminant) * 0.5f;
      outFar = (-b + sqrtDiscriminant) * 0.5f;

      return true;
    }

    // CPU counterpart of evalSunShadowing(). Ray marches the optical depth from an altitude toward
    // the sun through the spherical atmosphere, the same integral the transmittance LUT bakes, so the
    // CPU-side sun light and the GPU sky agree. Returns zero once the planet occludes the sun.
    // dirYUp must be unit length.
    Vector3 atmTransmittanceYUp(const AtmosphereArgs& a, const Vector3& dirYUp, float altitudeKm = 0.0f) {
      const Vector3 planetCenter(0.0f, -a.planetRadius, 0.0f);
      const Vector3 origin(0.0f, std::max(altitudeKm, 0.0f), 0.0f);

      float tNear, tFar;

      // Any intersection ahead of the origin means the sun is below the local horizon. The radius is
      // nudged inward so a sample sitting exactly on the ground is not self shadowed.
      if (atmIntersectSphere(origin, dirYUp, planetCenter, a.planetRadius * (1.0f - 1e-5f), tNear, tFar)
          && tFar >= 0.0f) {
        return Vector3(0.0f, 0.0f, 0.0f);
      }

      // March to the top of the atmosphere along the sun direction.
      if (!atmIntersectSphere(origin, dirYUp, planetCenter, a.atmosphereRadius, tNear, tFar) || tFar <= 0.0f) {
        return Vector3(1.0f, 1.0f, 1.0f);
      }

      const float tEnd = tFar;

      // Matches the shader's power distributed steps: short near the origin where the air is densest.
      constexpr int kSteps = 40;
      constexpr float kStepExponent = 2.0f;
      Vector3 opticalDepth(0.0f, 0.0f, 0.0f);
      float segmentStart = 0.0f;

      for (int i = 0; i < kSteps; ++i) {
        const float segmentEnd = std::pow(float(i + 1) / float(kSteps), kStepExponent);
        const float dt = (segmentEnd - segmentStart) * tEnd;
        const float t = (segmentStart + (segmentEnd - segmentStart) * 0.5f) * tEnd;
        segmentStart = segmentEnd;

        if (dt <= 0.0f) {
          continue;
        }

        const Vector3 samplePos = origin + dirYUp * t;
        const float h = std::min(
          std::max(length(samplePos - planetCenter) - a.planetRadius, 0.0f), a.atmosphereThickness);

        const float densityR = atmExponentialDensity(h, a.rayleighScaleHeight);
        const float densityM = atmMieDensity(a, h);
        const float densityO3 = atmOzoneDensity(a, h);

        opticalDepth.x += (a.rayleighScattering.x * densityR
                        + (a.mieScattering.x + a.mieAbsorption.x) * densityM
                        + a.ozoneAbsorption.x * densityO3) * dt;
        opticalDepth.y += (a.rayleighScattering.y * densityR
                        + (a.mieScattering.y + a.mieAbsorption.y) * densityM
                        + a.ozoneAbsorption.y * densityO3) * dt;
        opticalDepth.z += (a.rayleighScattering.z * densityR
                        + (a.mieScattering.z + a.mieAbsorption.z) * densityM
                        + a.ozoneAbsorption.z * densityO3) * dt;
      }

      return Vector3(
        std::exp(-std::min(opticalDepth.x, 1e3f)),
        std::exp(-std::min(opticalDepth.y, 1e3f)),
        std::exp(-std::min(opticalDepth.z, 1e3f)));
    }

    // The OPAC tables follow the enum order; Custom is the one type without an entry.
    static_assert(aerosol_tables::kTypeCount == uint32_t(AtmosphereAerosolType::Custom), "aerosol tables out of step with AtmosphereAerosolType");
    static_assert(aerosol_tables::kChannelCount == 3, "aerosol tables are per RGB channel");

    // Humidity classes bracketing a relative humidity, and the blend between them.
    void aerosolHumidityClasses(float relativeHumidityPercent, uint32_t& lower, uint32_t& upper, float& blend) {
      using namespace aerosol_tables;
      const float rh = std::min(std::max(relativeHumidityPercent, kHumidityPercent[0]), kHumidityPercent[kHumidityCount - 1]);
      lower = 0;
      while (lower + 2 < kHumidityCount && rh >= kHumidityPercent[lower + 1]) {
        ++lower;
      }
      upper = lower + 1;
      blend = (rh - kHumidityPercent[lower]) / (kHumidityPercent[upper] - kHumidityPercent[lower]);
      blend = std::min(std::max(blend, 0.0f), 1.0f);
    }

    Vector3 atmLerp(const Vector3& a, const Vector3& b, float t) {
      return a + (b - a) * t;
    }

    Vector3 atmLoad3(const float* v) {
      return Vector3(v[0], v[1], v[2]);
    }

    // Spectral derivation of the RGB coefficients and the sun colour (Bruneton 2017, Section 14.3): the
    // channels are spectral samples at kLambdaR/G/B, converted to linear sRGB assuming the spectrum
    // between them follows the solar spectrum times lambda^p (p = -3 for light scattered by air, the
    // aerosol's own Angstrom law for haze, 0 for the direct sun).
    struct SpectralCalibration {
      Vector3 rayleighScattering;  // km^-1
      Vector3 ozoneAbsorption;     // km^-1, for a 300 Dobson unit column
      Vector3 skyIlluminance;      // Rayleigh sky weights, relative, linear sRGB, normalised to the disc's luminance
      Vector3 discIlluminance;     // Relative, linear sRGB, unit luminance
      float discLuminance;         // Normaliser shared by any further sky weights
    };

    // Bruneton reshapes the spectrum between the sample wavelengths by lambda^-3 for a Rayleigh sky, a
    // flattening of the lambda^-4 law; aerosol light gets the same treatment of its Angstrom law.
    constexpr double kRayleighSkyLambdaPower = -3.0;
    constexpr double kSkyLambdaPowerFlattening = 0.75;

    // ASTM E-490 solar spectral irradiance, W m^-2 nm^-1, averaged in 10 nm bins from 360 nm.
    constexpr int kSolarLambdaMin = 360;
    constexpr int kSolarLambdaStep = 10;
    constexpr int kSolarSampleCount = 48;
    constexpr double kSolarIrradiance[kSolarSampleCount] = {
      1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054, 1.6887, 1.61253,
      1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809, 1.91686, 1.8298,
      1.8685, 1.8931, 1.85149, 1.8504, 1.8341, 1.8345, 1.8147, 1.78158, 1.7533,
      1.6965, 1.68194, 1.64654, 1.6048, 1.52143, 1.55622, 1.5113, 1.474, 1.4482,
      1.41018, 1.36775, 1.34188, 1.31429, 1.28303, 1.26758, 1.2367, 1.2082,
      1.18737, 1.14683, 1.12362, 1.1058, 1.07124, 1.04992
    };

    // Ozone absorption cross sections at 233 K (Serdyuchenko et al. 2014), m^2, same binning.
    constexpr double kOzoneCrossSection[kSolarSampleCount] = {
      1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27,
      8.451e-27, 1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26,
      1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25, 4.266e-25,
      4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25,
      2.662e-25, 2.238e-25, 1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26,
      6.566e-26, 5.105e-26, 4.15e-26, 4.228e-26, 3.237e-26, 2.451e-26, 2.801e-26,
      2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-27
    };

    // 300 Dobson units spread over the 30 km wide ozone tent (integral 15 km), molecules m^-3.
    constexpr double kDobsonUnit = 2.687e20;
    constexpr double kOzoneNumberDensity = 300.0 * kDobsonUnit / 15000.0;

    // Rayleigh scattering coefficient at sea level: kRayleigh * lambda^-4 with lambda in micrometers, m^-1.
    constexpr double kRayleighCoefficient = 1.24062e-6;

    double interpolateSolarTable(const double* table, double lambdaNm) {
      const double u = (lambdaNm - kSolarLambdaMin) / kSolarLambdaStep;
      const int i0 = std::min(std::max(int(std::floor(u)), 0), kSolarSampleCount - 1);
      const int i1 = std::min(i0 + 1, kSolarSampleCount - 1);
      const double t = std::min(std::max(u - double(i0), 0.0), 1.0);
      return table[i0] * (1.0 - t) + table[i1] * t;
    }

    // Multi-lobe Gaussian fit of the CIE 1931 2 degree colour matching functions
    // (Wyman, Sloan and Shirley, JCGT 2013).
    double cieLobe(double lambdaNm, double mean, double sigmaLow, double sigmaHigh) {
      const double t = (lambdaNm - mean) / (lambdaNm < mean ? sigmaLow : sigmaHigh);
      return std::exp(-0.5 * t * t);
    }

    void cieColorMatching(double lambdaNm, double& x, double& y, double& z) {
      x = 1.056 * cieLobe(lambdaNm, 599.8, 37.9, 31.0)
        + 0.362 * cieLobe(lambdaNm, 442.0, 16.0, 26.7)
        - 0.065 * cieLobe(lambdaNm, 501.1, 20.4, 26.2);
      y = 0.821 * cieLobe(lambdaNm, 568.8, 46.9, 40.5)
        + 0.286 * cieLobe(lambdaNm, 530.9, 16.3, 31.1);
      z = 1.217 * cieLobe(lambdaNm, 437.0, 11.8, 36.0)
        + 0.681 * cieLobe(lambdaNm, 459.0, 26.0, 13.8);
    }

    // Linear sRGB response to the solar spectrum reshaped by (lambda / lambda_c)^lambdaPower for each
    // channel's reference wavelength, i.e. k_c * S(lambda_c) of Bruneton's Section 14.3.
    Vector3 integrateSolarResponse(double lambdaPower) {
      constexpr double kXyzToSrgb[9] = {
        3.2404542, -1.5371385, -0.4985314,
        -0.9692660, 1.8760108, 0.0415560,
        0.0556434, -0.2040259, 1.0572252
      };
      const double lambdaRef[3] = { kLambdaR * 1000.0, kLambdaG * 1000.0, kLambdaB * 1000.0 };

      double response[3] = { 0.0, 0.0, 0.0 };
      for (int lambda = kSolarLambdaMin; lambda < kSolarLambdaMin + kSolarLambdaStep * (kSolarSampleCount - 1); ++lambda) {
        double x, y, z;
        cieColorMatching(double(lambda), x, y, z);

        const double solar = interpolateSolarTable(kSolarIrradiance, double(lambda));
        const double bar[3] = {
          kXyzToSrgb[0] * x + kXyzToSrgb[1] * y + kXyzToSrgb[2] * z,
          kXyzToSrgb[3] * x + kXyzToSrgb[4] * y + kXyzToSrgb[5] * z,
          kXyzToSrgb[6] * x + kXyzToSrgb[7] * y + kXyzToSrgb[8] * z
        };

        for (int c = 0; c < 3; ++c) {
          response[c] += bar[c] * solar * std::pow(double(lambda) / lambdaRef[c], lambdaPower);
        }
      }

      return Vector3(float(response[0]), float(response[1]), float(response[2]));
    }

    SpectralCalibration computeSpectralCalibration() {
      SpectralCalibration calibration;

      // m^-1 to km^-1.
      calibration.rayleighScattering = Vector3(
        float(kRayleighCoefficient * std::pow(kLambdaR, -4.0) * 1000.0),
        float(kRayleighCoefficient * std::pow(kLambdaG, -4.0) * 1000.0),
        float(kRayleighCoefficient * std::pow(kLambdaB, -4.0) * 1000.0));

      calibration.ozoneAbsorption = Vector3(
        float(kOzoneNumberDensity * interpolateSolarTable(kOzoneCrossSection, kLambdaR * 1000.0) * 1000.0),
        float(kOzoneNumberDensity * interpolateSolarTable(kOzoneCrossSection, kLambdaG * 1000.0) * 1000.0),
        float(kOzoneNumberDensity * interpolateSolarTable(kOzoneCrossSection, kLambdaB * 1000.0) * 1000.0));

      const Vector3 disc = integrateSolarResponse(0.0);
      const Vector3 sky = integrateSolarResponse(kRayleighSkyLambdaPower);
      const float discLuminance = std::max(sRGBLuminance(disc), 1e-6f);

      calibration.discIlluminance = disc * (1.0f / discLuminance);
      calibration.skyIlluminance = sky * (1.0f / discLuminance);
      calibration.discLuminance = discLuminance;

      return calibration;
    }

    const SpectralCalibration& getSpectralCalibration() {
      static const SpectralCalibration calibration = computeSpectralCalibration();
      return calibration;
    }

    Vector3 atmPow(const Vector3& base, float exponent) {
      return Vector3(std::pow(base.x, exponent), std::pow(base.y, exponent), std::pow(base.z, exponent));
    }

    Vector3 atmSafeReciprocal(const Vector3& v) {
      return Vector3(
        1.0f / std::max(v.x, 1e-4f),
        1.0f / std::max(v.y, 1e-4f),
        1.0f / std::max(v.z, 1e-4f));
    }

    // Altitude quantisation step for the baked LUTs when the altitude follows the camera, in km.
    constexpr float kBakeAltitudeQuantumKm = 0.005f;

    // Manual mode takes the sun colour as authored. Physical mode keeps only its luminance and derives the
    // colour from the solar spectrum, with separate weights for the light scattered by air, the light
    // scattered by aerosol and the direct sun. Expects the ground level Mie coefficients to be set.
    void fillSunIlluminance(AtmosphereArgs& args, const SpectralCalibration& spectral, bool physicalCoefficients) {
      const Vector3 sunBase = RtxOptions::sunIlluminance();
      const float sunIntensity = RtxOptions::sunIntensity();

      if (!physicalCoefficients) {
        args.sunIlluminance = sunBase * sunIntensity;
        args.sunIlluminanceAerosol = args.sunIlluminance;
        args.sunDiscIlluminance = args.sunIlluminance;
        return;
      }

      const float sunLuminance = std::max(sRGBLuminance(sunBase), 0.0f) * sunIntensity;
      Vector3 disc = spectral.discIlluminance;
      Vector3 sky = spectral.skyIlluminance;

      // The weights describe how the spectrum runs between the three sample wavelengths. Bruneton's
      // lambda^-3 is a Rayleigh sky; light scattered by aerosol follows the aerosol's own Angstrom law,
      // read off its coefficients. Weighting haze as if it were Rayleigh reddens it by a fifth.
      const double angstrom = (args.mieScattering.x > 0.0f && args.mieScattering.z > 0.0f)
        ? -std::log(double(args.mieScattering.x) / double(args.mieScattering.z)) / std::log(kLambdaR / kLambdaB)
        : 0.0;
      Vector3 haze = integrateSolarResponse(-kSkyLambdaPowerFlattening * angstrom) * (1.0f / spectral.discLuminance);

      if (RtxOptions::spectralWhiteBalance()) {
        const Vector3 discReciprocal = atmSafeReciprocal(disc);
        sky = atmMul(sky, discReciprocal);
        haze = atmMul(haze, discReciprocal);
        disc = Vector3(1.0f, 1.0f, 1.0f);
      }

      args.sunIlluminance = sky * sunLuminance;
      args.sunIlluminanceAerosol = haze * sunLuminance;
      args.sunDiscIlluminance = disc * sunLuminance;
    }

    // Ground level aerosol coefficients and vertical profile. Expects args.rayleighScattering to be set.
    void fillAerosol(AtmosphereArgs& args) {
      if (RtxOptions::aerosolModel() != AtmosphereAerosolModel::Visibility) {
        const float aerosolDensity = RtxOptions::aerosolDensity();
        args.mieScattering = RtxOptions::mieScattering() * aerosolDensity;
        args.mieAbsorption = RtxOptions::mieAbsorption() * aerosolDensity;
        args.mieAnisotropy = RtxOptions::mieAnisotropy();

        // Pure exponential profile, the paper's setup.
        args.mieBoundaryLayerHeight = 0.0f;
        args.mieBoundaryLayerTransition = 0.0f;
        args.mieBoundaryLayerTailScale = 1.0f;
        return;
      }

      const AtmosphereAerosolType type = RtxOptions::aerosolType();
      const float relativeHumidity = RtxOptions::aerosolRelativeHumidity();
      const RtxAtmosphere::AerosolOptics optics = RtxAtmosphere::getAerosolOptics(type, relativeHumidity);

      // OPAC types bring their own typical ground level extinction at 550 nm. A visibility instead gives the
      // total through Koschmieder; the green channel stands for 550 nm, so whatever the molecules do not account
      // for is aerosol. The type's spectral extinction spreads it over the channels' wavelengths.
      float aerosolExtinction550 = optics.extinction550;
      if (!optics.tabulated || RtxOptions::visibilityOverride()) {
        const float totalExtinction550 = kKoschmiederConstant / std::max(RtxOptions::visibilityKm(), 0.5f);
        aerosolExtinction550 = std::max(totalExtinction550 - args.rayleighScattering.y, 0.0f);
      }
      const Vector3 aerosolExtinction = optics.extinctionRatio * aerosolExtinction550;

      const Vector3 ssa(
        std::min(std::max(optics.singleScatteringAlbedo.x, 0.0f), 1.0f),
        std::min(std::max(optics.singleScatteringAlbedo.y, 0.0f), 1.0f),
        std::min(std::max(optics.singleScatteringAlbedo.z, 0.0f), 1.0f));
      args.mieScattering = atmMul(aerosolExtinction, ssa);
      args.mieAbsorption = atmMul(aerosolExtinction, Vector3(1.0f, 1.0f, 1.0f) - ssa);
      // The analytic lobe, when it applies, uses the 550 nm asymmetry.
      args.mieAnisotropy = optics.asymmetry.y;

      if (optics.tabulated && RtxOptions::aerosolMiePhase()) {
        args.miePhaseTabulated = 1u;
        args.aerosolTypeId = uint32_t(type);
        args.aerosolRelativeHumidity = relativeHumidity;
      }

      args.mieBoundaryLayerHeight = RtxOptions::boundaryLayerHeightKm();
      args.mieBoundaryLayerTransition = RtxOptions::boundaryLayerTransitionKm();
      args.mieBoundaryLayerTailScale = RtxOptions::freeTroposphereAerosolFraction();
    }

    // The sun, its colour and the viewpoint only reach the sky-view LUT; the transmittance and multiple
    // scattering LUTs depend on the medium alone.
    AtmosphereArgs clearSkyViewOnlyFields(AtmosphereArgs args) {
      args.sunDirection = vec3(0.0f, 0.0f, 0.0f);
      args.sunIlluminance = vec3(0.0f, 0.0f, 0.0f);
      args.sunRayBrightness = 0.0f;
      args.sunDiscEnabled = 0;
      args.viewAltitude = 0.0f;
      args.useSkyViewLut = 0;
      args.sunAngularRadius = 0.0f;
      args.sunDiscIlluminance = vec3(0.0f, 0.0f, 0.0f);
      args.sunIlluminanceAerosol = vec3(0.0f, 0.0f, 0.0f);
      args.skyViewStepCount = 0;
      args.sunLimbDarkeningExponent = vec3(0.0f, 0.0f, 0.0f);
      return args;
    }

  }

RtxAtmosphere::RtxAtmosphere(DxvkDevice* device)
  : CommonDeviceObject(device) {
  // Create constant buffer for atmosphere parameters
  DxvkBufferCreateInfo info = {};
  info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  info.access = VK_ACCESS_UNIFORM_READ_BIT;
  info.size = sizeof(AtmosphereArgs);
  m_constantsBuffer = device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Atmosphere constants buffer");
}

RtxAtmosphere::~RtxAtmosphere() {
  dropDistantSunLight();
}

void RtxAtmosphere::initialize(Rc<DxvkContext> ctx) {
  if (m_initialized) {
    return;
  }

  createLutResources(ctx);
  m_initialized = true;
  m_lutsNeedRecompute = true;
}

AtmosphereArgs RtxAtmosphere::buildAtmosphereArgsFromOptions() {
  AtmosphereArgs args = {};

  // Convert sun angles to direction vector (in Y-up space, for LUT generation)
  constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
  float azimuthRad = RtxOptions::sunRotation() * kDegToRad; // Mapped to Rotation
  float elevationRad = RtxOptions::sunElevation() * kDegToRad;
  
  // Sun direction is always in Y-up space since the LUTs are generated in Y-up space
  args.sunDirection.x = std::cos(elevationRad) * std::sin(azimuthRad);
  args.sunDirection.y = std::sin(elevationRad);
  args.sunDirection.z = std::cos(elevationRad) * std::cos(azimuthRad);

  // Basic atmosphere parameters
  args.planetRadius = RtxOptions::planetRadius();
  args.atmosphereThickness = RtxOptions::atmosphereThickness();

  const bool physicalCoefficients = RtxOptions::coefficientMode() == AtmosphereCoefficientMode::Physical;
  const SpectralCalibration& spectral = getSpectralCalibration();

  // Molecular coefficients (Base * Density Multiplier)
  args.rayleighScattering = (physicalCoefficients ? spectral.rayleighScattering : RtxOptions::rayleighScattering()) * RtxOptions::airDensity();
  args.ozoneAbsorption = (physicalCoefficients ? spectral.ozoneAbsorption : RtxOptions::ozoneAbsorption()) * RtxOptions::ozoneDensity();
  args.ozoneLayerAltitude = RtxOptions::ozoneLayerAltitude();
  args.ozoneLayerWidth = RtxOptions::ozoneLayerWidth();

  args.rayleighScaleHeight = kRayleighScaleHeight;
  args.mieScaleHeight = kMieScaleHeight;
  fillAerosol(args);

  // After the aerosol, whose spectral weights are read off its coefficients.
  fillSunIlluminance(args, spectral, physicalCoefficients);

  args.miePhaseAlpha = RtxOptions::miePhaseAlpha();
  args.mieForwardPeakWeight = RtxOptions::mieForwardPeakWeight();
  args.mieForwardPeakG = RtxOptions::mieForwardPeakG();

  // Depolarisation factor of air rho = 0.0279 (Bodhaine et al. 1999), gamma = rho / (2 - rho).
  args.rayleighPhaseGamma = RtxOptions::rayleighDepolarization() ? (0.0279f / (2.0f - 0.0279f)) : 0.0f;

  // Hestroffer and Magnan 1998: alpha(lambda) = -0.023 + 0.292 / lambda[um], at the channel wavelengths.
  args.sunLimbDarkeningExponent = RtxOptions::sunLimbDarkening()
    ? Vector3(float(-0.023 + 0.292 / kLambdaR), float(-0.023 + 0.292 / kLambdaG), float(-0.023 + 0.292 / kLambdaB))
    : Vector3(0.0f, 0.0f, 0.0f);

  args.groundAlbedo = RtxOptions::groundAlbedo();

  // Sun Angular Radius (from Sun Size in degrees)
  // sunSize is diameter in degrees. Radius = Size / 2
  float sunSizeRad = RtxOptions::sunSize() * kDegToRad;
  args.sunAngularRadius = sunSizeRad * 0.5f;
  
  // Brightness multiplier
  args.sunRayBrightness = 1.0f; 

  args.sunDiscEnabled = RtxOptions::sunDisc() ? 1u : 0u;

  // View Altitude (converted m to km). fillAerialPerspectiveArgs() adds the camera height when enabled.
  args.viewAltitude = RtxOptions::altitude() * 0.001f;
  args.aerialPerspectiveViewAltitude = args.viewAltitude;

  args.useSkyViewLut = RtxOptions::useSkyViewLut() ? 1u : 0u;

  // LUT dimensions
  args.transmittanceLutWidth = kTransmittanceLutWidth;
  args.transmittanceLutHeight = kTransmittanceLutHeight;
  args.multiscatteringLutSize = kMultiscatteringLutSize;
  args.skyViewLutWidth = kSkyViewLutWidth;
  args.skyViewLutHeight = kSkyViewLutHeight;
  args.aerosolPhaseLutSize = kAerosolPhaseLutSize;
  args.multiscatteringStepCount = uint32_t(std::max(RtxOptions::multiscatteringSteps(), 4));
  args.skyViewStepCount = uint32_t(std::min(std::max(RtxOptions::skyViewSteps(), 16), 512));

  // Derived parameters
  args.atmosphereRadius = args.planetRadius + args.atmosphereThickness;

  // Aerial perspective. The camera basis is filled in per frame by fillAerialPerspectiveArgs().
  const float worldUnitsPerMeter = RtxOptions::getMeterToWorldUnitScale();
  const bool aerialPerspective = RtxOptions::aerialPerspective();
  args.aerialPerspectiveLutSize = aerialPerspective ? uint32_t(std::max(RtxOptions::aerialPerspectiveLutSize(), 8)) : 0u;
  args.aerialPerspectiveLutDepth = uint32_t(std::max(RtxOptions::aerialPerspectiveLutDepth(), 8));
  args.aerialPerspectiveDepthRange =
    RtxOptions::aerialPerspectiveDepthRangeMeters() * worldUnitsPerMeter;
  args.worldUnitsPerKilometer = worldUnitsPerMeter * 1000.0f;
  args.isZUp = RtxOptions::zUp() ? 1u : 0u;

  args.aerialPerspectiveShadowSteps = (aerialPerspective && RtxOptions::aerialPerspectiveShadows())
    ? uint32_t(std::max(RtxOptions::aerialPerspectiveShadowSteps(), 1))
    : 0u;
  args.aerialPerspectiveShadowMaxDistance = RtxOptions::aerialPerspectiveShadowMaxDistanceMeters() * worldUnitsPerMeter;
  args.aerialPerspectiveTemporalBlend = RtxOptions::aerialPerspectiveTemporalBlend();
  args.aerialPerspectiveAerosolScale = RtxOptions::aerialPerspectiveAerosolScale();

  // Hand off to the global volumetrics froxel grid: everything nearer than its range is already
  // integrated there, so double counting is avoided by starting the atmospheric march past it.
  const float volumetricsHandoffMeters = RtxGlobalVolumetrics::enable() ? RtxGlobalVolumetrics::froxelMaxDistanceMeters() : 0.0f;
  args.aerialPerspectiveStartDistance =
    std::max(volumetricsHandoffMeters, RtxOptions::aerialPerspectiveStartDistanceMeters()) * worldUnitsPerMeter;

  return args;
}

void RtxAtmosphere::fillAerialPerspectiveArgs(AtmosphereArgs& args, const RtCamera& camera) const {
  // Frustum half extents at unit forward distance, so the shader's ray direction always has a
  // forward component of exactly one and the slice index maps linearly to forward distance.
  const float tanHalfFovY = std::tan(camera.getFov() * 0.5f);
  const float tanHalfFovX = tanHalfFovY * camera.getAspectRatio();

  args.cameraPosition = camera.getPosition();
  args.cameraForward = camera.getDirection();
  args.cameraRight = camera.getRight() * tanHalfFovX;
  args.cameraUp = camera.getUp() * tanHalfFovY;

  // The previous basis reuses this frame's field of view: a zoom between frames is rare and only
  // costs one frame of history.
  const Matrix4d& prevViewToWorld = camera.getPreviousViewToWorld();
  const Vector3 prevRight { prevViewToWorld[0].xyz() };
  const Vector3 prevUp { prevViewToWorld[1].xyz() };
  const Vector3 prevForward = camera.isLHS() ? Vector3 { prevViewToWorld[2].xyz() } : -Vector3 { prevViewToWorld[2].xyz() };
  args.prevCameraPosition = camera.getPreviousPosition();
  args.prevCameraForward = prevForward;
  args.prevCameraRight = prevRight * tanHalfFovX;
  args.prevCameraUp = prevUp * tanHalfFovY;

  args.aerialPerspectiveHistoryValid = m_aerialPerspectiveHistoryValid ? 1u : 0u;
  args.aerialPerspectiveFrameIndex = m_aerialPerspectiveFrameIndex;

  if (RtxOptions::altitudeFollowsCamera()) {
    const float cameraHeight = RtxOptions::zUp() ? args.cameraPosition.z : args.cameraPosition.y;
    const float heightKm = (cameraHeight - RtxOptions::groundLevelWorldHeight()) / std::max(args.worldUnitsPerKilometer, 1e-6f);
    const float exactAltitude = std::max(RtxOptions::altitude() * 0.001f + heightKm, 0.0f);

    args.aerialPerspectiveViewAltitude = exactAltitude;
    // Quantised so the baked sky LUTs do not rebake on every step of a staircase.
    args.viewAltitude = std::floor(exactAltitude / kBakeAltitudeQuantumKm + 0.5f) * kBakeAltitudeQuantumKm;
  }
}

float RtxAtmosphere::computeEffectiveVisibilityKm(const AtmosphereArgs& args) {
  // Ground level extinction of the green channel, which stands for 550 nm.
  const float extinction = args.rayleighScattering.y + args.mieScattering.y + args.mieAbsorption.y;
  return kKoschmiederConstant / std::max(extinction, 1e-6f);
}

float RtxAtmosphere::computeAerosolLowSunBlend() {
  if (RtxOptions::aerosolModel() != AtmosphereAerosolModel::Visibility ||
      RtxOptions::aerosolType() != AtmosphereAerosolType::Custom ||
      !RtxOptions::aerosolLowSunBlend()) {
    return 0.0f;
  }

  // Smooth ramp from the authored optics at the start elevation to the low sun optics at the end one.
  const float start = RtxOptions::aerosolLowSunBlendStartDegrees();
  const float end = std::min(RtxOptions::aerosolLowSunBlendEndDegrees(), start - 1e-3f);
  return 1.0f - atmSmoothstep(end, start, RtxOptions::sunElevation());
}

RtxAtmosphere::AerosolOptics RtxAtmosphere::getAerosolOptics(AtmosphereAerosolType type, float relativeHumidityPercent) {
  AerosolOptics optics;

  if (type == AtmosphereAerosolType::Custom) {
    Vector3 albedo = RtxOptions::aerosolSingleScatteringAlbedo();
    float angstrom = RtxOptions::aerosolAngstromExponent();

    const float lowSunBlend = computeAerosolLowSunBlend();
    if (lowSunBlend > 0.0f) {
      albedo = atmLerp(albedo, RtxOptions::aerosolLowSunSingleScatteringAlbedo(), lowSunBlend);
      angstrom += (RtxOptions::aerosolLowSunAngstromExponent() - angstrom) * lowSunBlend;
    }

    // Angstrom's law relative to 550 nm.
    const Vector3 wavelengthRatio(float(kLambdaR / kLambdaG), 1.0f, float(kLambdaB / kLambdaG));
    optics.singleScatteringAlbedo = albedo;
    optics.extinctionRatio = atmPow(wavelengthRatio, -angstrom);
    optics.asymmetry = Vector3(RtxOptions::mieAnisotropy(), RtxOptions::mieAnisotropy(), RtxOptions::mieAnisotropy());
    return optics;
  }

  using namespace aerosol_tables;
  const uint32_t typeIndex = std::min(uint32_t(type), kTypeCount - 1);
  uint32_t lower, upper;
  float blend;
  aerosolHumidityClasses(relativeHumidityPercent, lower, upper, blend);
  const TypeOptics& a = kOptics[typeIndex][lower];
  const TypeOptics& b = kOptics[typeIndex][upper];

  optics.singleScatteringAlbedo = atmLerp(atmLoad3(a.singleScatteringAlbedo), atmLoad3(b.singleScatteringAlbedo), blend);
  optics.extinctionRatio = atmLerp(atmLoad3(a.extinctionRatio), atmLoad3(b.extinctionRatio), blend);
  optics.asymmetry = atmLerp(atmLoad3(a.asymmetry), atmLoad3(b.asymmetry), blend);
  optics.extinction550 = a.extinction550 + (b.extinction550 - a.extinction550) * blend;
  optics.tabulated = true;
  return optics;
}

bool RtxAtmosphere::needsSkyViewRecompute(const AtmosphereArgs& args) const {
  if (!m_initialized || m_lutsNeedRecompute) {
    return true;
  }

  // Only the camera independent prefix matters here; see kBakeInvariantArgsSize.
  return memcmp(&args, &m_cachedArgs, kBakeInvariantArgsSize) != 0;
}

bool RtxAtmosphere::needsMediumRecompute(const AtmosphereArgs& args) const {
  if (!m_initialized || m_lutsNeedRecompute) {
    return true;
  }

  const AtmosphereArgs medium = clearSkyViewOnlyFields(args);
  const AtmosphereArgs cachedMedium = clearSkyViewOnlyFields(m_cachedArgs);
  return memcmp(&medium, &cachedMedium, kBakeInvariantArgsSize) != 0;
}

void RtxAtmosphere::createLutResources(Rc<DxvkContext> ctx) {
  // Create transmittance LUT (stores atmospheric transmittance)
  VkExtent3D transmittanceExtent = { kTransmittanceLutWidth, kTransmittanceLutHeight, 1 };
  m_transmittanceLut = Resources::createImageResource(
    ctx,
    "Atmosphere Transmittance LUT",
    transmittanceExtent,
    VK_FORMAT_R16G16B16A16_SFLOAT,
    1, // numLayers
    VK_IMAGE_TYPE_2D,
    VK_IMAGE_VIEW_TYPE_2D,
    0, // imageCreateFlags
    VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags
    VkClearColorValue{}, // clearValue
    1 // mipLevels
  );

  // Multiple scattering atlas (layout in atmosphere_args.h), and the scratch copy its bake passes alternate with.
  VkExtent3D multiscatteringExtent = {
    kMultiscatteringLutSize * ATMOSPHERE_MS_ATLAS_TILES_X, kMultiscatteringLutSize * ATMOSPHERE_MS_ATLAS_TILES_Y, 1 };
  m_multiscatteringLut = Resources::createImageResource(
    ctx,
    "Atmosphere Multiscattering LUT",
    multiscatteringExtent,
    VK_FORMAT_R16G16B16A16_SFLOAT,
    1, // numLayers
    VK_IMAGE_TYPE_2D,
    VK_IMAGE_VIEW_TYPE_2D,
    0, // imageCreateFlags
    VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags
    VkClearColorValue{}, // clearValue
    1 // mipLevels
  );
  m_multiscatteringScratch = Resources::createImageResource(
    ctx,
    "Atmosphere Multiscattering Scratch",
    multiscatteringExtent,
    VK_FORMAT_R16G16B16A16_SFLOAT,
    1, // numLayers
    VK_IMAGE_TYPE_2D,
    VK_IMAGE_VIEW_TYPE_2D,
    0, // imageCreateFlags
    VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags
    VkClearColorValue{}, // clearValue
    1 // mipLevels
  );

  // Create sky view LUT (main view-dependent sky color LUT)
  VkExtent3D skyViewExtent = { kSkyViewLutWidth, kSkyViewLutHeight, 1 };
  m_skyViewLut = Resources::createImageResource(
    ctx,
    "Atmosphere Sky View LUT",
    skyViewExtent,
    VK_FORMAT_R16G16B16A16_SFLOAT,
    1, // numLayers
    VK_IMAGE_TYPE_2D,
    VK_IMAGE_VIEW_TYPE_2D,
    0, // imageCreateFlags
    VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags
    VkClearColorValue{}, // clearValue
    1 // mipLevels
  );

  // Isotropic hemisphere mean of the sky-view LUT.
  VkExtent3D skyHemisphereExtent = { 1, 1, 1 };
  m_skyHemisphereMean = Resources::createImageResource(
    ctx,
    "Atmosphere Sky Hemisphere Mean",
    skyHemisphereExtent,
    VK_FORMAT_R16G16B16A16_SFLOAT,
    1, // numLayers
    VK_IMAGE_TYPE_2D,
    VK_IMAGE_VIEW_TYPE_2D,
    0, // imageCreateFlags
    VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags
    VkClearColorValue{}, // clearValue
    1 // mipLevels
  );

  // Tabulated aerosol phase function, uploaded from the CPU when its type or humidity changes.
  VkExtent3D aerosolPhaseExtent = { kAerosolPhaseLutSize, 1, 1 };
  m_aerosolPhaseLut = Resources::createImageResource(
    ctx,
    "Atmosphere Aerosol Phase LUT",
    aerosolPhaseExtent,
    VK_FORMAT_R32G32B32A32_SFLOAT,
    1, // numLayers
    VK_IMAGE_TYPE_2D,
    VK_IMAGE_VIEW_TYPE_2D,
    0, // imageCreateFlags
    0, // extraUsageFlags
    VkClearColorValue{}, // clearValue
    1 // mipLevels
  );

  // The aerial perspective volumes are sized from options, so they are created on first use.
}

void RtxAtmosphere::updateAerosolPhaseLut(Rc<DxvkContext> ctx, const AtmosphereArgs& args) {
  if (args.miePhaseTabulated == 0 ||
      (args.aerosolTypeId == m_aerosolPhaseLutType && args.aerosolRelativeHumidity == m_aerosolPhaseLutHumidity)) {
    return;
  }

  using namespace aerosol_tables;
  const uint32_t typeIndex = std::min(args.aerosolTypeId, kTypeCount - 1);
  uint32_t lower, upper;
  float blend;
  aerosolHumidityClasses(args.aerosolRelativeHumidity, lower, upper, blend);

  // Texel i holds p(theta) at u = (i + 0.5) / size, theta = pi u^2, so the forward peak gets most of
  // the texels. The tables are log-linearly interpolated in angle between OPAC's samples.
  constexpr uint32_t size = kAerosolPhaseLutSize;
  std::vector<float> texels(size_t(size) * 4, 0.0f);
  for (uint32_t channel = 0; channel < kChannelCount; ++channel) {
    const float* p0 = kPhase[typeIndex][lower][channel];
    const float* p1 = kPhase[typeIndex][upper][channel];

    uint32_t segment = 0;
    for (uint32_t i = 0; i < size; ++i) {
      const float u = (float(i) + 0.5f) / float(size);
      const float thetaDegrees = 180.0f * u * u;
      while (segment + 2 < kAngleCount && thetaDegrees > kAngleDegrees[segment + 1]) {
        ++segment;
      }
      const float t = std::min(std::max(
        (thetaDegrees - kAngleDegrees[segment]) / (kAngleDegrees[segment + 1] - kAngleDegrees[segment]), 0.0f), 1.0f);
      const float lowerValue = std::exp(std::log(p0[segment]) * (1.0f - t) + std::log(p0[segment + 1]) * t);
      const float upperValue = std::exp(std::log(p1[segment]) * (1.0f - t) + std::log(p1[segment + 1]) * t);
      texels[size_t(i) * 4 + channel] = lowerValue + (upperValue - lowerValue) * blend;
    }

    // Normalise the phase function as the shader reconstructs it (linear between texel centres,
    // clamped beyond them) to a unit integral over the sphere: 2 pi Int p(theta) sin(theta) dtheta with
    // theta = pi u^2.
    double integral = 0.0;
    constexpr int kSubSamples = 8;
    const int sampleCount = int(size) * kSubSamples;
    for (int s = 0; s < sampleCount; ++s) {
      const double u = (double(s) + 0.5) / double(sampleCount);
      const double coord = std::min(std::max(u * size - 0.5, 0.0), double(size - 1));
      const uint32_t i0 = uint32_t(coord);
      const uint32_t i1 = std::min(i0 + 1, size - 1);
      const double f = coord - double(i0);
      const double p = double(texels[size_t(i0) * 4 + channel]) * (1.0 - f) + double(texels[size_t(i1) * 4 + channel]) * f;
      const double theta = kAtmPi * u * u;
      integral += p * std::sin(theta) * (2.0 * kAtmPi * u) * (1.0 / double(sampleCount));
    }
    integral *= 2.0 * kAtmPi;

    const float scale = integral > 0.0 ? float(1.0 / integral) : 1.0f;
    for (uint32_t i = 0; i < size; ++i) {
      texels[size_t(i) * 4 + channel] *= scale;
    }
  }

  const VkImageSubresourceLayers subresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
  ctx->updateImage(m_aerosolPhaseLut.image, subresource, VkOffset3D { 0, 0, 0 }, VkExtent3D { size, 1, 1 },
    texels.data(), size * 4 * sizeof(float), size * 4 * sizeof(float));

  m_aerosolPhaseLutType = args.aerosolTypeId;
  m_aerosolPhaseLutHumidity = args.aerosolRelativeHumidity;
}

void RtxAtmosphere::ensureAerialPerspectiveLuts(Rc<DxvkContext> ctx, const AtmosphereArgs& args) {
  const VkExtent3D extent = {
    std::max(args.aerialPerspectiveLutSize, 1u), std::max(args.aerialPerspectiveLutSize, 1u), std::max(args.aerialPerspectiveLutDepth, 1u) };

  if (extent.width == m_aerialPerspectiveLutExtent.width &&
      extent.height == m_aerialPerspectiveLutExtent.height &&
      extent.depth == m_aerialPerspectiveLutExtent.depth &&
      m_aerialPerspectiveLut[0].isValid() && m_aerialPerspectiveLut[1].isValid()) {
    return;
  }

  const VkExtent3D tileExtent = { extent.width, extent.height, 1 };

  // In-scatter in RGB, mean transmittance in A.
  for (uint32_t i = 0; i < 2; ++i) {
    m_aerialPerspectiveLut[i] = Resources::createImageResource(
      ctx,
      i == 0 ? "Atmosphere Aerial Perspective LUT 0" : "Atmosphere Aerial Perspective LUT 1",
      extent,
      VK_FORMAT_R16G16B16A16_SFLOAT,
      1, // numLayers
      VK_IMAGE_TYPE_3D,
      VK_IMAGE_VIEW_TYPE_3D,
      0, // imageCreateFlags
      VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags
      VkClearColorValue{}, // clearValue
      1 // mipLevels
    );

    m_aerialPerspectiveTileDepth[i] = Resources::createImageResource(
      ctx,
      i == 0 ? "Atmosphere Aerial Perspective Tile Depth 0" : "Atmosphere Aerial Perspective Tile Depth 1",
      tileExtent,
      VK_FORMAT_R32_SFLOAT,
      1, // numLayers
      VK_IMAGE_TYPE_2D,
      VK_IMAGE_VIEW_TYPE_2D,
      0, // imageCreateFlags
      VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags
      VkClearColorValue{}, // clearValue
      1 // mipLevels
    );
  }

  m_aerialPerspectiveLutExtent = extent;
  m_aerialPerspectiveHistoryValid = false;
}

void RtxAtmosphere::computeLuts(Rc<DxvkContext> ctx, const AtmosphereArgs& args) {
  // One upload serves every pass below.
  ctx->updateBuffer(m_constantsBuffer, 0, sizeof(AtmosphereArgs), &args);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);

  // Before the bakes, which sample it.
  updateAerosolPhaseLut(ctx, args);

  const bool mediumChanged = needsMediumRecompute(args);
  if (mediumChanged || needsSkyViewRecompute(args)) {
    m_cachedArgs = args;

    if (mediumChanged) {
      // Transmittance first: the multiscattering and sky-view bakes both sample it.
      dispatchTransmittanceLut(ctx);
      ctx->emitMemoryBarrier(0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

      dispatchMultiscatteringLut(ctx);
      ctx->emitMemoryBarrier(0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    }

    dispatchSkyViewLut(ctx);
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    dispatchSkyHemisphereMean(ctx);

    m_lutsNeedRecompute = false;

    // The accumulated volume was lit by a different atmosphere or sun.
    m_aerialPerspectiveHistoryValid = false;
  }

  // The aerial perspective volume is written later by dispatchAerialPerspective(); flip once per frame
  // here so composite reads the volume that frame writes.
  if (args.aerialPerspectiveLutSize > 0) {
    ensureAerialPerspectiveLuts(ctx, args);
    m_aerialPerspectiveLutIndex ^= 1u;
  }

  // LUT writes before ray tracing and composite.
  ctx->emitMemoryBarrier(0,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_ACCESS_SHADER_WRITE_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
    VK_ACCESS_SHADER_READ_BIT);
}

void RtxAtmosphere::dispatchTransmittanceLut(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Transmittance LUT");

  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_transmittanceLut.view, nullptr);

  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_transmittanceLut.image);

  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, TransmittanceLutShader::getShader());
  ctx->dispatch((kTransmittanceLutWidth + 15) / 16, (kTransmittanceLutHeight + 15) / 16, 1);
}

void RtxAtmosphere::dispatchMultiscatteringLut(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Multiscattering LUT");

  ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, MultiscatteringLutShader::getShader());
  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_transmittanceLut.view, nullptr);
  ctx->bindResourceView(2, m_aerosolPhaseLut.view, nullptr);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_transmittanceLut.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_aerosolPhaseLut.image);

  // The isotropic estimate goes to the scratch atlas, then the full order passes alternate between the two
  // atlases, each reading the previous pass, so the last one lands in m_multiscatteringLut. Dense haze needs the
  // third full order pass to converge.
  constexpr uint32_t kFullOrderPassCount = 3;
  for (uint32_t pass = 0; pass <= kFullOrderPassCount; ++pass) {
    const bool writesScratch = (pass % 2) == 0;
    const Resources::Resource& input = writesScratch ? m_multiscatteringLut : m_multiscatteringScratch;
    const Resources::Resource& output = writesScratch ? m_multiscatteringScratch : m_multiscatteringLut;

    AtmosphereMultiscatteringBakeArgs bakeArgs = {};
    bakeArgs.passIndex = pass == 0 ? ATMOSPHERE_MS_PASS_ISOTROPIC
                       : pass == 1 ? ATMOSPHERE_MS_PASS_FROM_ISOTROPIC
                       : ATMOSPHERE_MS_PASS_FROM_DIRECTIONAL;

    ctx->bindResourceView(3, input.view, nullptr);
    ctx->bindResourceView(4, output.view, nullptr);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(input.image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(output.image);

    ctx->pushConstants(0, sizeof(bakeArgs), &bakeArgs);
    // One workgroup per texel of a tile.
    ctx->dispatch(kMultiscatteringLutSize, kMultiscatteringLutSize, 1);

    if (pass < kFullOrderPassCount) {
      ctx->emitMemoryBarrier(0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }
  }
}

void RtxAtmosphere::dispatchSkyViewLut(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Sky View LUT");

  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_transmittanceLut.view, nullptr);
  ctx->bindResourceView(2, m_multiscatteringLut.view, nullptr);
  ctx->bindResourceView(3, m_skyViewLut.view, nullptr);
  ctx->bindResourceView(4, m_aerosolPhaseLut.view, nullptr);
  
  // Track resources
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_transmittanceLut.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_multiscatteringLut.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_aerosolPhaseLut.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_skyViewLut.image);
  
  // Bind shader and dispatch
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, SkyViewLutShader::getShader());
  
  // Dispatch with 16x16 thread groups
  uint32_t groupsX = (kSkyViewLutWidth + 15) / 16;
  uint32_t groupsY = (kSkyViewLutHeight + 15) / 16;
  ctx->dispatch(groupsX, groupsY, 1);
}

void RtxAtmosphere::dispatchSkyHemisphereMean(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Sky Hemisphere Mean");

  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_skyViewLut.view, nullptr);
  ctx->bindResourceView(2, m_skyHemisphereMean.view, nullptr);

  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_skyViewLut.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_skyHemisphereMean.image);

  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, SkyViewHemisphereMeanShader::getShader());
  ctx->dispatch(1, 1, 1);
}

void RtxAtmosphere::dispatchAerialPerspective(RtxContext& ctx, const Resources::RaytracingOutput& rtOutput) {
  const AtmosphereArgs& args = rtOutput.m_raytraceArgs.atmosphereArgs;
  if (!m_initialized || args.aerialPerspectiveLutSize == 0 || !m_aerialPerspectiveLut[0].isValid()) {
    return;
  }

  ScopedGpuProfileZone(&ctx, "Atmosphere Aerial Perspective");

  // The G-buffer pass wrote the primary hits the volume is bounded by.
  ctx.emitMemoryBarrier(0,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_SHADER_WRITE_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

  dispatchAerialPerspectiveTileDepth(ctx, rtOutput.m_primaryLinearViewZ.view);
  ctx.emitMemoryBarrier(0,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

  if (args.aerialPerspectiveShadowSteps == 0) {
    dispatchAerialPerspectiveLut(ctx, args);

    // The unshadowed volume is exact per frame; nothing to carry over should shadows be enabled.
    m_aerialPerspectiveHistoryValid = false;
  } else {
    ctx.bindCommonRayTracingResources(rtOutput);
    dispatchShadowedAerialPerspectiveLut(ctx, args);

    // From here on the written volume is a valid history for the next frame.
    m_aerialPerspectiveHistoryValid = args.aerialPerspectiveTemporalBlend > 0.0f;
    ++m_aerialPerspectiveFrameIndex;
  }

  // Volume writes before composite reads them.
  ctx.emitMemoryBarrier(0,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
}

void RtxAtmosphere::dispatchAerialPerspectiveTileDepth(RtxContext& ctx, const Rc<DxvkImageView>& primaryLinearViewZ) {
  ScopedGpuProfileZone(&ctx, "Atmosphere Aerial Perspective Tile Depth");

  const Resources::Resource& output = m_aerialPerspectiveTileDepth[m_aerialPerspectiveLutIndex];

  ctx.bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx.bindResourceView(1, primaryLinearViewZ, nullptr);
  ctx.bindResourceView(2, output.view, nullptr);

  ctx.getCommandList()->trackResource<DxvkAccess::Read>(primaryLinearViewZ->image());
  ctx.getCommandList()->trackResource<DxvkAccess::Write>(output.image);

  ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, AerialPerspectiveTileDepthShader::getShader());

  // One thread group per tile.
  ctx.dispatch(m_aerialPerspectiveLutExtent.width, m_aerialPerspectiveLutExtent.height, 1);
}

void RtxAtmosphere::dispatchAerialPerspectiveLut(RtxContext& ctx, const AtmosphereArgs& args) {
  ScopedGpuProfileZone(&ctx, "Atmosphere Aerial Perspective LUT");

  const Resources::Resource& output = m_aerialPerspectiveLut[m_aerialPerspectiveLutIndex];
  const Resources::Resource& tileDepth = m_aerialPerspectiveTileDepth[m_aerialPerspectiveLutIndex];

  ctx.bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx.bindResourceView(1, m_transmittanceLut.view, nullptr);
  ctx.bindResourceView(2, m_multiscatteringLut.view, nullptr);
  ctx.bindResourceView(3, output.view, nullptr);
  ctx.bindResourceView(4, tileDepth.view, nullptr);
  ctx.bindResourceView(5, m_aerosolPhaseLut.view, nullptr);

  ctx.getCommandList()->trackResource<DxvkAccess::Read>(m_transmittanceLut.image);
  ctx.getCommandList()->trackResource<DxvkAccess::Read>(m_multiscatteringLut.image);
  ctx.getCommandList()->trackResource<DxvkAccess::Read>(tileDepth.image);
  ctx.getCommandList()->trackResource<DxvkAccess::Read>(m_aerosolPhaseLut.image);
  ctx.getCommandList()->trackResource<DxvkAccess::Write>(output.image);

  ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, AerialPerspectiveLutShader::getShader());

  const uint32_t groupsXY = (args.aerialPerspectiveLutSize + 3) / 4;
  const uint32_t groupsZ = (args.aerialPerspectiveLutDepth + 3) / 4;
  ctx.dispatch(groupsXY, groupsXY, groupsZ);
}

// Expects the common ray tracing bindings (constants with this frame's atmosphere args, TLAS, atmosphere
// LUTs) to be bound.
void RtxAtmosphere::dispatchShadowedAerialPerspectiveLut(RtxContext& ctx, const AtmosphereArgs& args) {
  ScopedGpuProfileZone(&ctx, "Atmosphere Aerial Perspective Ray Query");

  const Resources::Resource& output = m_aerialPerspectiveLut[m_aerialPerspectiveLutIndex];
  const Resources::Resource& history = m_aerialPerspectiveLut[m_aerialPerspectiveLutIndex ^ 1u];
  const Resources::Resource& tileDepth = m_aerialPerspectiveTileDepth[m_aerialPerspectiveLutIndex];
  const Resources::Resource& prevTileDepth = m_aerialPerspectiveTileDepth[m_aerialPerspectiveLutIndex ^ 1u];

  ctx.bindResourceView(AERIAL_PERSPECTIVE_BINDING_PREV_LUT_INPUT, history.view, nullptr);
  ctx.bindResourceView(AERIAL_PERSPECTIVE_BINDING_TILE_DEPTH_INPUT, tileDepth.view, nullptr);
  ctx.bindResourceView(AERIAL_PERSPECTIVE_BINDING_PREV_TILE_DEPTH_INPUT, prevTileDepth.view, nullptr);
  ctx.bindResourceView(AERIAL_PERSPECTIVE_BINDING_LUT_OUTPUT, output.view, nullptr);

  ctx.getCommandList()->trackResource<DxvkAccess::Read>(history.image);
  ctx.getCommandList()->trackResource<DxvkAccess::Read>(tileDepth.image);
  ctx.getCommandList()->trackResource<DxvkAccess::Read>(prevTileDepth.image);
  ctx.getCommandList()->trackResource<DxvkAccess::Write>(output.image);

  ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, AerialPerspectiveRayQueryShader::getShader());

  const uint32_t groupsXY = (args.aerialPerspectiveLutSize + 3) / 4;
  const uint32_t groupsZ = (args.aerialPerspectiveLutDepth + 3) / 4;
  ctx.dispatch(groupsXY, groupsXY, groupsZ);
}

void RtxAtmosphere::dropDistantSunLight() {
  if (m_sunDistantLight != nullptr) {
    m_sunDistantLight->markForGarbageCollection();
    m_sunDistantLight = nullptr;
  }
}

Vector3 RtxAtmosphere::estimateVolumeAmbientRadiance(const AtmosphereArgs& args) {
  // Fallback isotropic fill from ground-reaching sun irradiance (Rayleigh-ish tint).
  // Prefer the sky-view LUT froxel path when sky ambient strength > 0.
  const Vector3 sunDirYUp(args.sunDirection.x, args.sunDirection.y, args.sunDirection.z);
  constexpr float kTwilightLo = -0.259f; // -15 deg
  constexpr float kTwilightHi = 0.15f;
  const float elevFade = atmSmoothstep(kTwilightLo, kTwilightHi, sunDirYUp.y);
  if (elevFade <= 0.0f) {
    return Vector3(0.0f, 0.0f, 0.0f);
  }

  const Vector3 T = atmTransmittanceYUp(args, sunDirYUp, args.viewAltitude);
  const Vector3 sunIll(args.sunIlluminance.x, args.sunIlluminance.y, args.sunIlluminance.z);
  const Vector3 groundIlluminance = atmMul(sunIll, T) * args.sunRayBrightness;

  // /pi: multiScatteringEstimate is radiance beside froxel SH; sunIll*T is irradiance-like.
  const Vector3 skyTint(0.65f, 0.78f, 1.0f);
  return atmMul(groundIlluminance, skyTint) * (elevFade / kAtmPi);
}

void RtxAtmosphere::estimateUnoccludedVolumeLighting(
  const AtmosphereArgs& args,
  bool isZUp,
  Vector3& outSunRadiance,
  Vector3& outSunDirectionWorld) {
  const Vector3 sunDirYUp(args.sunDirection.x, args.sunDirection.y, args.sunDirection.z);
  outSunRadiance = Vector3(0.0f, 0.0f, 0.0f);
  outSunDirectionWorld = Vector3(0.0f, 0.0f, 0.0f);
  if (sunDirYUp.y > 0.0f) {
    outSunDirectionWorld = isZUp
      ? Vector3(sunDirYUp.x, sunDirYUp.z, sunDirYUp.y)
      : sunDirYUp;
    const Vector3 T = atmTransmittanceYUp(args, sunDirYUp, args.viewAltitude);
    const Vector3 sunIll(args.sunDiscIlluminance.x, args.sunDiscIlluminance.y, args.sunDiscIlluminance.z);
    outSunRadiance = atmMul(sunIll, T) * args.sunRayBrightness;
  }
}

void RtxAtmosphere::estimateVolumeSunsetWarmTint(const AtmosphereArgs& args, Vector3& outTint, float& outBlend) {
  outTint = Vector3(1.0f, 1.0f, 1.0f);
  outBlend = 0.0f;

  const Vector3 sunDirYUp(args.sunDirection.x, args.sunDirection.y, args.sunDirection.z);
  // Ramp in from ~40° elevation so low-sun haze responds before the horizon.
  outBlend = 1.0f - atmSmoothstep(0.05f, 0.65f, sunDirYUp.y);
  if (outBlend <= 1e-4f) {
    return;
  }

  Vector3 dirForT = sunDirYUp;
  if (dirForT.y < 0.02f) {
    dirForT.y = 0.02f;
    const float len = std::sqrt(dirForT.x * dirForT.x + dirForT.y * dirForT.y + dirForT.z * dirForT.z);
    dirForT = Vector3(dirForT.x / len, dirForT.y / len, dirForT.z / len);
  }
  const Vector3 T = atmTransmittanceYUp(args, dirForT, args.viewAltitude);
  const float tLum = std::max(sRGBLuminance(T), 1e-4f);
  const float blueLoss = std::min(std::max(1.0f - (T.z / tLum), 0.0f), 1.0f);

  // Mild warm multipliers; reduce G with B so cool media stay realistic under warm T.
  outTint = Vector3(
    1.0f + 0.05f + 0.04f * blueLoss,
    1.0f - 0.08f - 0.06f * blueLoss,
    1.0f - 0.14f - 0.10f * blueLoss);
}

void RtxAtmosphere::syncDistantSunLight(RtxContext& ctx, const AtmosphereArgs& args) {
  // Sole Physical Atmosphere sun for surface NEE + Volume ReSTIR.
  LightManager& lm = ctx.getSceneManager().getLightManager();
  const bool isZUp = RtxOptions::zUp();
  constexpr float kMinHalfAngle = 0.0005f;

  const Vector3 sunDirYUp(args.sunDirection.x, args.sunDirection.y, args.sunDirection.z);
  const bool sunAboveHorizon = sunDirYUp.y > 0.0f;

  // RtDistantLight stores illuminance / pi, since distantLightSampleArea() recovers the sample
  // radiance as that divided by sin^2(halfAngle). This keeps the delivered illuminance independent
  // of the cone width, so widening the cone only softens shadows and dims the reflected disc.
  //
  // The horizon needs no artificial fade: atmTransmittanceYUp() reddens and then extinguishes the
  // sun as it descends, and returns zero once the planet occludes it. Twilight is then the sky's own
  // multiple scattering rather than direct sunlight leaking below the horizon.
  Vector3 radiance(0.0f, 0.0f, 0.0f);
  if (sunAboveHorizon) {
    const Vector3 T = atmTransmittanceYUp(args, sunDirYUp, args.viewAltitude);
    const Vector3 sunIll(args.sunDiscIlluminance.x, args.sunDiscIlluminance.y, args.sunDiscIlluminance.z);
    const Vector3 illuminance = atmMul(sunIll, T) * args.sunRayBrightness;
    radiance = illuminance * (1.0f / kAtmPi);
  }

  const Vector3 toSun = isZUp
    ? Vector3(sunDirYUp.x, sunDirYUp.z, sunDirYUp.y)
    : sunDirYUp;
  // Propagation toward the ground (= -toBody).
  const Vector3 propDir = sunAboveHorizon
    ? Vector3(-toSun.x, -toSun.y, -toSun.z)
    : Vector3(0.0f, -1.0f, 0.0f);

  // The cone half-angle doubles as the sun's apparent size in every glossy reflection, so widening it
  // to soften shadows also makes the reflected sun larger and dimmer. sunShadowSoftening keeps that
  // trade-off opt-in rather than deriving a widening from atmospheric or fog optical depth.
  constexpr float kMaxHalfAngle = 12.0f * (kAtmPi / 180.0f);
  const float softening = RtxOptions::sunShadowSoftening() * (kAtmPi / 180.0f);
  const float halfAngle = std::min(
    std::max(args.sunAngularRadius, kMinHalfAngle) + std::max(softening, 0.0f), kMaxHalfAngle);

  const Vector3 clamped(
    std::max(radiance.x, 0.0f),
    std::max(radiance.y, 0.0f),
    std::max(radiance.z, 0.0f));

  auto dl = RtDistantLight::tryCreate(propDir, halfAngle, clamped);
  if (!dl) {
    return;
  }

  RtLight rtl(*dl);
  rtl.isDynamic = true; // Keep sun direction updating each frame.

  if (m_sunDistantLight == nullptr) {
    m_sunDistantLight = lm.createExternallyTrackedLight(rtl);
  } else {
    lm.updateExternallyTrackedLight(m_sunDistantLight, rtl);
  }
}

} // namespace dxvk
