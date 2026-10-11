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
#include <random>

#include "rtx_eye_model.h"
#include "rtx_options.h"
#include "../dxvk_device.h"
#include "../dxvk_context.h"
#include "../../util/util_color.h"

namespace dxvk {

  namespace {
    constexpr double kPiDouble = 3.14159265358979323846;

    // Illuminance of the sun above the atmosphere, lux.
    constexpr double kSolarIlluminanceLux = 128000.0;

    // Stiles-Crawford coefficient for the luminance of photopic vision, mm^-2 (Applegate and Lakshminarayanan 1993).
    constexpr float kStilesCrawford = 0.05f;

    // The radius of the lens, over which its particles spread, in mm.
    constexpr float kLensRadiusMm = 4.5f;

    // A typical young eye has many small particles in its lens and fewer large ones in its vitreous (Ritschel et al.
    // 2009 take 750 in the lens), their diameters spread log-uniformly, in mm.
    constexpr uint32_t kLensParticles = 750;
    constexpr float kLensParticleMinMm = 0.008f;
    constexpr float kLensParticleMaxMm = 0.025f;
    constexpr uint32_t kVitreousParticles = 60;
    constexpr float kVitreousParticleMinMm = 0.02f;
    constexpr float kVitreousParticleMaxMm = 0.06f;

    // The lens fibres' spacing, which puts the lenticular halo's ring at 3.4 degrees at 550 nm, red outside, as
    // Simpson (1953) measured, and its width and share of the light.
    constexpr double kFibreSpacingMm = 0.0094;
    constexpr double kHaloRelativeWidth = 0.06;
    constexpr double kHaloShare = 0.002;

    constexpr uint32_t kCellsPerSide = 32;

    // Pupil diameters Watson and Yellott's formula spans, mm.
    constexpr float kMinPupilMm = 2.0f;
    constexpr float kMaxPupilMm = 8.0f;

    float hash(uint32_t value) {
      value ^= value >> 16;
      value *= 0x7feb352du;
      value ^= value >> 15;
      value *= 0x846ca68bu;
      value ^= value >> 16;
      return float(value) / 4294967295.0f;
    }

    // Smooth noise in [-1, 1] of time, with a period of about one unit.
    float valueNoise(float t, uint32_t seed) {
      const float base = std::floor(t);
      const float f = t - base;
      const float u = f * f * (3.0f - 2.0f * f);
      const uint32_t i = uint32_t(int32_t(base)) * 0x9e3779b9u + seed * 0x85ebca6bu;
      const float a = hash(i);
      const float b = hash(i + 0x9e3779b9u);
      return 2.0f * (a + (b - a) * u) - 1.0f;
    }

    // Sums three octaves of value noise, for the pupil's unrest and the particles' drift.
    float fractalNoise(float t, uint32_t seed) {
      return 0.57f * valueNoise(t, seed) + 0.29f * valueNoise(2.03f * t, seed + 1) + 0.14f * valueNoise(4.01f * t, seed + 2);
    }

    // Watson and Yellott (2012), "A unified formula for light-adapted pupil size": the diameter in mm for a luminance in
    // cd/m^2 seen over a field of fieldAreaDeg2 square degrees, by both eyes, at an age in years.
    float unifiedPupilDiameter(double luminance, double fieldAreaDeg2, double age) {
      const double flux = std::max(luminance * fieldAreaDeg2, 1e-9);
      const double power = std::pow(flux / 846.0, 0.41);
      const double standard = 7.75 - 5.75 * (power / (power + 2.0));
      const double clampedAge = std::clamp(age, 20.0, 83.0);
      const double diameter = standard + (clampedAge - 28.58) * (0.02132 - 0.009562 * standard);
      return float(std::clamp(diameter, double(kMinPupilMm), double(kMaxPupilMm)));
    }

    // CIE 135/1-6 general disability glare equation, per steradian, at theta degrees.
    double cieGlare(double theta, double ageFactor, double pigmentation) {
      return 10.0 / (theta * theta * theta) + (5.0 / (theta * theta) + 0.1 * pigmentation / theta) * ageFactor + 0.0025 * pigmentation;
    }
  }

  EyeModel::EyeModel(DxvkDevice* device)
    : m_device(device) {
  }

  EyeModel::~EyeModel() = default;

  uint32_t EyeModel::getCellsPerSide() {
    return kCellsPerSide;
  }

  float EyeModel::getStilesCrawford() {
    return kStilesCrawford;
  }

  float EyeModel::getLuminanceScale() {
    if (luminanceScale() > 0.0f) {
      return luminanceScale();
    }

    const double luminance = double(sRGBLuminance(RtxOptions::sunIlluminance())) * double(RtxOptions::sunIntensity());
    return float(kSolarIlluminanceLux / std::max(luminance, 1e-6));
  }

  float EyeModel::getCieAgeFactor() {
    const double ratio = double(age()) / 62.5;
    return float(1.0 + ratio * ratio * ratio * ratio);
  }

  float EyeModel::getCieScatteredShare() {
    // Integrated over the sphere from 0.1 to 100 degrees, in log steps, as the glare falls as the cube of the angle,
    // and eased out over the last ten as the shader does.
    constexpr uint32_t kSteps = 2048;
    const double ageFactor = getCieAgeFactor();
    const double pigment = pigmentation();
    const double logMin = std::log(0.1);
    const double logStep = (std::log(100.0) - logMin) / double(kSteps);
    double sum = 0.0;

    for (uint32_t i = 0; i < kSteps; i++) {
      const double theta = std::exp(logMin + (double(i) + 0.5) * logStep);
      const double thetaRadians = theta * kPiDouble / 180.0;
      const double dThetaRadians = theta * logStep * kPiDouble / 180.0;
      const double t = std::clamp((theta - 90.0) / 10.0, 0.0, 1.0);
      const double ease = 1.0 - t * t * (3.0 - 2.0 * t);
      sum += cieGlare(theta, ageFactor, pigment) * ease * 2.0 * kPiDouble * std::sin(thetaRadians) * dThetaRadians;
    }

    return float(sum);
  }

  float EyeModel::getHaloShare() {
    return float(kHaloShare * double(haloStrength()));
  }

  float EyeModel::getHaloAngle() {
    return float(550e-6 / kFibreSpacingMm);
  }

  float EyeModel::getHaloWidth() {
    return float(kHaloRelativeWidth * 550e-6 / kFibreSpacingMm);
  }

  void EyeModel::generateParticles() {
    const float density = particleDensity();
    std::mt19937 random(0x1ce5eedu);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    m_particles.clear();

    auto addParticles = [&](uint32_t count, float minMm, float maxMm, bool vitreous) {
      for (uint32_t i = 0; i < count; i++) {
        const float radius = kLensRadiusMm * std::sqrt(uniform(random));
        const float angle = float(2.0 * kPiDouble) * uniform(random);
        const float diameter = minMm * std::pow(maxMm / minMm, uniform(random));
        m_particles.push_back({ radius * std::cos(angle), radius * std::sin(angle), 0.5f * diameter, vitreous });
      }
    };

    addParticles(uint32_t(std::lround(kLensParticles * density)), kLensParticleMinMm, kLensParticleMaxMm, false);
    addParticles(uint32_t(std::lround(kVitreousParticles * density)), kVitreousParticleMinMm, kVitreousParticleMaxMm, true);
    m_generatedDensity = density;
  }

  void EyeModel::binParticles(float pupilRadiusMm, float time) {
    // The lens particles breathe with the lens's accommodative microfluctuations, and the vitreous's sway with the
    // eye's movements, lagging behind them.
    const float lensScale = 1.0f + 0.004f * fractalNoise(time * 1.7f, 11);
    const float vitreousAngle = 0.03f * fractalNoise(time * 0.35f, 23);
    const float vitreousShiftX = 0.05f * fractalNoise(time * 0.3f, 37);
    const float vitreousShiftY = 0.05f * fractalNoise(time * 0.3f, 41);
    const float cosAngle = std::cos(vitreousAngle);
    const float sinAngle = std::sin(vitreousAngle);

    m_pupilParticles.clear();

    for (const EyeParticle& particle : m_particles) {
      float x = particle.x;
      float y = particle.y;

      if (particle.vitreous) {
        const float rotatedX = cosAngle * x - sinAngle * y + vitreousShiftX;
        const float rotatedY = sinAngle * x + cosAngle * y + vitreousShiftY;
        x = rotatedX;
        y = rotatedY;
      } else {
        x *= lensScale;
        y *= lensScale;
      }

      const Vector4 pupilParticle(x / pupilRadiusMm, y / pupilRadiusMm, particle.radiusMm / pupilRadiusMm, 0.0f);

      if (std::sqrt(pupilParticle.x * pupilParticle.x + pupilParticle.y * pupilParticle.y) - pupilParticle.z < 1.0f) {
        m_pupilParticles.push_back(pupilParticle);
      }
    }

    // Each particle goes into every cell its bounds overlap, the cells covering the pupil's square.
    std::vector<std::vector<uint32_t>> cells(kCellsPerSide * kCellsPerSide);
    const float cellsPerUnit = 0.5f * float(kCellsPerSide);

    for (uint32_t i = 0; i < m_pupilParticles.size(); i++) {
      const Vector4& particle = m_pupilParticles[i];
      const int32_t maxCell = int32_t(kCellsPerSide) - 1;
      const int32_t x0 = std::clamp(int32_t(std::floor((particle.x - particle.z + 1.0f) * cellsPerUnit)), 0, maxCell);
      const int32_t x1 = std::clamp(int32_t(std::floor((particle.x + particle.z + 1.0f) * cellsPerUnit)), 0, maxCell);
      const int32_t y0 = std::clamp(int32_t(std::floor((particle.y - particle.z + 1.0f) * cellsPerUnit)), 0, maxCell);
      const int32_t y1 = std::clamp(int32_t(std::floor((particle.y + particle.z + 1.0f) * cellsPerUnit)), 0, maxCell);

      for (int32_t y = y0; y <= y1; y++) {
        for (int32_t x = x0; x <= x1; x++) {
          cells[y * kCellsPerSide + x].push_back(i);
        }
      }
    }

    m_cellRanges.assign(2 * kCellsPerSide * kCellsPerSide, 0);
    m_cellParticles.clear();

    for (uint32_t cell = 0; cell < cells.size(); cell++) {
      m_cellRanges[2 * cell] = uint32_t(m_cellParticles.size());
      m_cellRanges[2 * cell + 1] = uint32_t(cells[cell].size());
      m_cellParticles.insert(m_cellParticles.end(), cells[cell].begin(), cells[cell].end());
    }

    // The aperture pass needs buffers with at least one element.
    if (m_pupilParticles.empty()) {
      m_pupilParticles.push_back(Vector4(0.0f));
    }

    if (m_cellParticles.empty()) {
      m_cellParticles.push_back(0);
    }
  }

  void EyeModel::readBackFieldLuminance(DxvkContext* ctx, const Rc<DxvkImage>& image) {
    const uint32_t slot = m_readbackFrame % kReadbackRing;
    const uint32_t oldestSlot = (m_readbackFrame + 1) % kReadbackRing;

    // The slot after this frame's was written kReadbackRing - 1 frames ago, long enough for the GPU to have finished.
    if (m_readbackFrame + 1 >= kReadbackRing && m_readback[oldestSlot] != nullptr) {
      const float* value = reinterpret_cast<const float*>(m_readback[oldestSlot]->mapPtr(0));

      if (value != nullptr && std::isfinite(value[0])) {
        m_fieldLuminance = std::max(value[0], 0.0f);
      }
    }

    if (m_readback[slot] == nullptr) {
      DxvkBufferCreateInfo info {};
      info.size = sizeof(float) * 4;
      info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
      m_readback[slot] = m_device->createBuffer(info,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        DxvkMemoryStats::Category::RTXBuffer, "Eye Field Luminance Readback");
    }

    VkImageSubresourceLayers subresource {};
    subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource.layerCount = 1;
    ctx->copyImageToBuffer(m_readback[slot], 0, sizeof(float) * 4, sizeof(float) * 4, image, subresource,
                           VkOffset3D { 0, 0, 0 }, VkExtent3D { 1, 1, 1 });
    ctx->emitMemoryBarrier(0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    m_readbackFrame++;
  }

  bool EyeModel::update(float deltaSeconds, float fieldAreaDeg2) {
    const float dt = std::clamp(deltaSeconds, 0.0f, 0.25f);
    m_time += dt;

    if (m_generatedDensity != particleDensity()) {
      generateParticles();
      m_sincePublish = 1e6f;
    }

    // The pupil's light reflex closes it on the light within a fraction of a second and opens it again over seconds.
    float target = m_pupilMm;

    if (pupilDiameterMm() > 0.0f) {
      target = std::clamp(pupilDiameterMm(), kMinPupilMm, kMaxPupilMm);
      m_pupilMm = target;
    } else if (m_fieldLuminance >= 0.0f) {
      target = unifiedPupilDiameter(double(m_fieldLuminance) * getLuminanceScale(), fieldAreaDeg2, age());
      const float timeConstant = target < m_pupilMm ? constrictionSeconds() : dilationSeconds();
      m_pupilMm = timeConstant > 0.0f ? m_pupilMm + (target - m_pupilMm) * (1.0f - std::exp(-dt / timeConstant)) : target;
    }

    m_sincePublish += dt;

    if (m_sincePublish < 1.0f / updateRateHz()) {
      return false;
    }

    // Hippus, most noticeable in bright light: a slow unrest of about 0.3 Hz, wider as the pupil narrows.
    const float unrest = hippus() * 0.25f * (1.0f - m_pupilMm / 9.0f) * fractalNoise(m_time * 0.3f, 5);
    m_publishedPupilMm = std::clamp(m_pupilMm + unrest, kMinPupilMm * 0.9f, kMaxPupilMm);
    m_publishedTime = m_time;
    m_sincePublish = 0.0f;

    binParticles(0.5f * m_publishedPupilMm, m_time);
    return true;
  }

}
