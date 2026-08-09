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
#include "dxvk_device.h"
#include "dxvk_context.h"
#include "rtx_options.h"
#include "rtx_context.h"
#include "rtx_scene_manager.h"
#include "rtx_light_manager.h"
#include "rtx_lights.h"
#include "rtx_global_volumetrics.h"
#include "rtx_render/rtx_shader_manager.h"
#include "../../util/util_color.h"
#include <rtx_shaders/sky_view_lut.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace dxvk {
  // Shader definitions for atmosphere LUT generation
  // Note: the transmittance and multiscattering LUT passes are no longer dispatched -
  // sky-view LUT generation and all runtime paths use the analytical transmittance and
  // multiscattering approximations in atmosphere_common.slangh, so those two LUTs were
  // computed but never read. Their images are still created because the runtime binding
  // layout (common_bindings.slangh) declares them.
  namespace {
    class SkyViewLutShader : public ManagedShader {
      SHADER_SOURCE(SkyViewLutShader, VK_SHADER_STAGE_COMPUTE_BIT, sky_view_lut)
      
      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        TEXTURE2D(1)
        TEXTURE2D(2)
        SAMPLER(3)
        RW_TEXTURE2D(4)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(SkyViewLutShader);

    constexpr float kAtmPi = 3.14159265358979323846f;

    float atmSmoothstep(float e0, float e1, float x) {
      const float denom = e1 - e0;
      float t = (denom != 0.0f) ? (x - e0) / denom : 0.0f;
      t = std::min(std::max(t, 0.0f), 1.0f);
      return t * t * (3.0f - 2.0f * t);
    }

    Vector3 atmMul(const Vector3& a, const Vector3& b) {
      return Vector3(a.x * b.x, a.y * b.y, a.z * b.z);
    }

    // CPU port of getAtmosphericTransmittanceForDir / getTransmittanceToSunAtAltitude.
    // dirYUp must be normalized, Y-up. Ozone density at the layer altitude is 1.
    // altitudeKm scales remaining optical depth by exp(-h/H).
    Vector3 atmTransmittanceYUp(const AtmosphereArgs& a, const Vector3& dirYUp, float altitudeKm = 0.0f) {
      const float H = a.rayleighScaleHeight;
      const float zc = dirYUp.y;
      float airMass;
      if (zc > 0.01f) {
        const float zenithRad = std::acos(std::min(std::max(zc, -1.0f), 1.0f));
        const float zenithDeg = zenithRad * (180.0f / kAtmPi);
        airMass = 1.0f / (zc + 0.15f * std::pow(93.885f - zenithDeg, -1.253f));
      } else {
        airMass = 40.0f * std::exp(-zc * 10.0f);
      }
      airMass = std::min(airMass, 200.0f);
      const float h = std::max(altitudeKm, 0.0f);
      const float rayleighDensity = std::exp(-h / std::max(H, 1e-3f));
      const float mieDensity = std::exp(-h / std::max(a.mieScaleHeight, 1e-3f));
      const float rayleighOD = H * airMass * rayleighDensity;
      const float mieOD = a.mieScaleHeight * airMass * mieDensity;
      const float ozonePath = airMass * rayleighDensity;
      Vector3 t(
        std::exp(-(a.rayleighScattering.x * rayleighOD + a.mieScattering.x * mieOD + a.ozoneAbsorption.x * ozonePath * 0.15f)),
        std::exp(-(a.rayleighScattering.y * rayleighOD + a.mieScattering.y * mieOD + a.ozoneAbsorption.y * ozonePath * 0.15f)),
        std::exp(-(a.rayleighScattering.z * rayleighOD + a.mieScattering.z * mieOD + a.ozoneAbsorption.z * ozonePath * 0.15f)));
      if (zc < 0.0f) {
        const float f = std::exp(-(-zc) * 15.0f);
        t = Vector3(t.x * f, t.y * f, t.z * f);
      }
      return t;
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
  
  // Sun illuminance (Base * Intensity)
  // Allows customizing base color via options/presets, while simple UI controls intensity
  args.sunIlluminance = RtxOptions::sunIlluminance() * RtxOptions::sunIntensity();

  // Scattering coefficients (Base * Density Multiplier)
  // Allows advanced customization of scattering colors while exposing simple density sliders
  float airDensity = RtxOptions::airDensity();
  args.rayleighScattering = RtxOptions::rayleighScattering() * airDensity;
  
  float aerosolDensity = RtxOptions::aerosolDensity();
  args.mieScattering = RtxOptions::mieScattering() * aerosolDensity;
  
  args.mieAnisotropy = RtxOptions::mieAnisotropy();
  
  // Sun Angular Radius (from Sun Size in degrees)
  // sunSize is diameter in degrees. Radius = Size / 2
  float sunSizeRad = RtxOptions::sunSize() * kDegToRad;
  args.sunAngularRadius = sunSizeRad * 0.5f;
  
  // Brightness multiplier
  args.sunRayBrightness = 1.0f; 

  // Ozone absorption (Base * Density Multiplier)
  float ozoneDensity = RtxOptions::ozoneDensity();
  args.ozoneAbsorption = RtxOptions::ozoneAbsorption() * ozoneDensity;
  
  // Internal ozone params
  args.ozoneLayerAltitude = RtxOptions::ozoneLayerAltitude();
  args.ozoneLayerWidth = RtxOptions::ozoneLayerWidth();

  // View Altitude (converted m to km)
  args.viewAltitude = RtxOptions::altitude() * 0.001f;

  args.useSkyViewLut = RtxOptions::useSkyViewLut() ? 1u : 0u;

  // LUT dimensions
  args.transmittanceLutWidth = kTransmittanceLutWidth;
  args.transmittanceLutHeight = kTransmittanceLutHeight;
  args.multiscatteringLutSize = kMultiscatteringLutSize;
  args.skyViewLutWidth = kSkyViewLutWidth;
  args.skyViewLutHeight = kSkyViewLutHeight;

  // Derived parameters
  args.atmosphereRadius = args.planetRadius + args.atmosphereThickness;
  args.rayleighScaleHeight = kRayleighScaleHeight;
  args.mieScaleHeight = kMieScaleHeight;

  return args;
}

bool RtxAtmosphere::needsLutRecompute() const {
  if (!m_initialized || m_lutsNeedRecompute) {
    return true;
  }

  // Check if any parameters have changed
  AtmosphereArgs currentArgs = getAtmosphereArgs();
  
  // Compare with cached args (simple memcmp would work for POD types)
  return memcmp(&currentArgs, &m_cachedArgs, sizeof(AtmosphereArgs)) != 0;
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

  // Create multiscattering LUT (stores multiple scattering contribution)
  VkExtent3D multiscatteringExtent = { kMultiscatteringLutSize, kMultiscatteringLutSize, 1 };
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
}

void RtxAtmosphere::computeLuts(Rc<DxvkContext> ctx) {
  if (!needsLutRecompute()) {
    return;
  }

  // Update cached args
  m_cachedArgs = getAtmosphereArgs();

  dispatchSkyViewLut(ctx);
  
  // Final barrier: Ensure the LUT is written before use in ray tracing
  ctx->emitMemoryBarrier(0,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_ACCESS_SHADER_WRITE_BIT,
    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
    VK_ACCESS_SHADER_READ_BIT);

  m_lutsNeedRecompute = false;
}

void RtxAtmosphere::dispatchSkyViewLut(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Sky View LUT");
  
  // Update atmosphere args buffer
  AtmosphereArgs args = getAtmosphereArgs();
  ctx->updateBuffer(m_constantsBuffer, 0, sizeof(AtmosphereArgs), &args);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);
  
  // Bind resources
  // Note: the transmittance/multiscattering LUT inputs are still declared by the shader
  // interface but the generation code evaluates both analytically and never samples them.
  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_transmittanceLut.view, nullptr);
  ctx->bindResourceView(2, m_multiscatteringLut.view, nullptr);
  
  if (m_lutSampler == nullptr) {
    DxvkSamplerCreateInfo samplerInfo = {};
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    m_lutSampler = m_device->createSampler(samplerInfo);
  }
  ctx->bindResourceSampler(3, m_lutSampler);
  
  ctx->bindResourceView(4, m_skyViewLut.view, nullptr);
  
  // Track resources
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_transmittanceLut.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_multiscatteringLut.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_skyViewLut.image);
  
  // Bind shader and dispatch
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, SkyViewLutShader::getShader());
  
  // Dispatch with 16x16 thread groups
  uint32_t groupsX = (kSkyViewLutWidth + 15) / 16;
  uint32_t groupsY = (kSkyViewLutHeight + 15) / 16;
  ctx->dispatch(groupsX, groupsY, 1);
}

void RtxAtmosphere::bindResources(Rc<DxvkContext> ctx, VkPipelineBindPoint pipelineBindPoint) {
  // TODO: Bind atmosphere LUT resources to the pipeline
  // This will be called from RtxContext to make the LUTs available to shaders
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
  // Radiance matches sampleAtmosphereSunLight / pi (distant-light solid-angle concentration).
  LightManager& lm = ctx.getSceneManager().getLightManager();
  const bool isZUp = RtxOptions::zUp();
  constexpr float kMinHalfAngle = 0.0005f;
  constexpr float kTwilightLo = -0.259f; // -15 deg
  constexpr float kTwilightHi = 0.05f;

  const Vector3 sunDirYUp(args.sunDirection.x, args.sunDirection.y, args.sunDirection.z);
  const float elevFade = atmSmoothstep(kTwilightLo, kTwilightHi, sunDirYUp.y);

  Vector3 radiance(0.0f, 0.0f, 0.0f);
  Vector3 T(1.0f, 1.0f, 1.0f);
  if (elevFade > 0.0f) {
    const float mieModulation = 0.3f + 1.7f * args.mieAnisotropy;
    const float sunVisibility = 0.05f + 0.95f * atmSmoothstep(0.0f, 0.8f, args.mieAnisotropy);
    Vector3 dirForT = sunDirYUp;
    if (dirForT.y < 0.02f) {
      dirForT.y = 0.02f;
      const float len = std::sqrt(dirForT.x * dirForT.x + dirForT.y * dirForT.y + dirForT.z * dirForT.z);
      dirForT = Vector3(dirForT.x / len, dirForT.y / len, dirForT.z / len);
    }
    T = atmTransmittanceYUp(args, dirForT, args.viewAltitude);
    const Vector3 sunIll(args.sunIlluminance.x, args.sunIlluminance.y, args.sunIlluminance.z);
    const Vector3 sample = atmMul(sunIll, T) * (mieModulation * sunVisibility * args.sunRayBrightness * 0.5f * elevFade);
    radiance = sample * (1.0f / kAtmPi);
  }

  const Vector3 toSun = isZUp
    ? Vector3(sunDirYUp.x, sunDirYUp.z, sunDirYUp.y)
    : sunDirYUp;
  // Propagation toward the ground (= -toBody).
  const Vector3 propDir = (elevFade > 0.0f)
    ? Vector3(-toSun.x, -toSun.y, -toSun.z)
    : Vector3(0.0f, -1.0f, 0.0f);

  // Soft shadows = distant-light cone half-angle. Widen with atmosphere/fog OD.
  // Do not rescale radiance by sin²θ: GPU samples use radiance/sin²θ while ReSTIR
  // weights use raw radiance, so energy-preserving scales over-select the sun.
  const float baseHalfAngle = std::max(args.sunAngularRadius, kMinHalfAngle);
  float extraHalfAngle = 0.0f;
  if (elevFade > 0.0f) {
    const float avgT = std::max((T.x + T.y + T.z) * (1.0f / 3.0f), 1e-4f);
    const float atmOd = -std::log(avgT);
    extraHalfAngle += std::min(atmOd * 0.035f, 6.0f * (kAtmPi / 180.0f));

    if (RtxGlobalVolumetrics::enable()) {
      const Vector3 tcLin = sRGBGammaToLinear(RtxGlobalVolumetrics::transmittanceColor());
      const float tLum = std::min(std::max(sRGBLuminance(tcLin), 1e-4f), 0.999f);
      const float meas = std::max(
        RtxGlobalVolumetrics::transmittanceMeasurementDistanceMeters() * RtxOptions::getMeterToWorldUnitScale(),
        1e-3f);
      const float sigma = -std::log(tLum) / meas;
      const Vector3 alb = RtxGlobalVolumetrics::singleScatteringAlbedo();
      const float aLum = std::min(std::max(sRGBLuminance(alb), 0.0f), 1.0f);
      const float scenicMeters = std::max(
        RtxGlobalVolumetrics::froxelMaxDistanceMeters() * 3.0f,
        120.0f);
      const float scenicPath = scenicMeters * RtxOptions::getMeterToWorldUnitScale();
      const float extOd = sigma * scenicPath;
      const float absorbOd = sigma * (1.0f - aLum) * scenicPath;
      const float fogOd = std::min(extOd + 1.5f * absorbOd, 8.0f);
      // Thin media stay near the geometric disk; dense fog ramps to full widen.
      constexpr float kFogOdSoftStart = 0.35f;
      constexpr float kFogOdSoftFull = 2.0f;
      const float fogSoft = atmSmoothstep(kFogOdSoftStart, kFogOdSoftFull, fogOd);
      extraHalfAngle += fogSoft * (10.0f * (kAtmPi / 180.0f));
    }
  }
  constexpr float kMaxHalfAngle = 12.0f * (kAtmPi / 180.0f);
  const float halfAngle = std::min(baseHalfAngle + extraHalfAngle, kMaxHalfAngle);

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
