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
#include "rtx_camera.h"
#include "rtx_scene_manager.h"
#include "rtx_light_manager.h"
#include "rtx_lights.h"
#include "rtx_global_volumetrics.h"
#include "rtx_render/rtx_shader_manager.h"
#include "../../util/util_color.h"
#include <rtx_shaders/transmittance_lut.h>
#include <rtx_shaders/multiscattering_lut.h>
#include <rtx_shaders/sky_view_lut.h>
#include <rtx_shaders/sky_view_hemisphere_mean.h>
#include <rtx_shaders/aerial_perspective_lut.h>
#include <algorithm>
#include <cmath>
#include <cstring>

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

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        TEXTURE2D(1)
        RW_TEXTURE2D(2)
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

    class AerialPerspectiveLutShader : public ManagedShader {
      SHADER_SOURCE(AerialPerspectiveLutShader, VK_SHADER_STAGE_COMPUTE_BIT, aerial_perspective_lut)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        TEXTURE2D(1)
        TEXTURE2D(2)
        RW_TEXTURE3D(3)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(AerialPerspectiveLutShader);

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

    float atmOzoneDensity(const AtmosphereArgs& a, float altitudeKm) {
      const float halfWidth = std::max(a.ozoneLayerWidth, 1e-3f);
      return std::max(0.0f, 1.0f - std::abs(altitudeKm - a.ozoneLayerAltitude) / halfWidth);
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

        const float densityR = std::exp(-h / std::max(a.rayleighScaleHeight, 1e-3f));
        const float densityM = std::exp(-h / std::max(a.mieScaleHeight, 1e-3f));
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
  
  // Aerosols both scatter and absorb, so Mie extinction needs the absorption term too.
  args.mieAbsorption = RtxOptions::mieAbsorption() * aerosolDensity;

  args.mieAnisotropy = RtxOptions::mieAnisotropy();
  
  // Sun Angular Radius (from Sun Size in degrees)
  // sunSize is diameter in degrees. Radius = Size / 2
  float sunSizeRad = RtxOptions::sunSize() * kDegToRad;
  args.sunAngularRadius = sunSizeRad * 0.5f;
  
  // Brightness multiplier
  args.sunRayBrightness = 1.0f; 

  args.sunDiscEnabled = RtxOptions::sunDisc() ? 1u : 0u;

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

  // Aerial perspective. The camera basis is filled in per frame by fillAerialPerspectiveArgs().
  const float worldUnitsPerMeter = RtxOptions::getMeterToWorldUnitScale();
  args.aerialPerspectiveLutSize = RtxOptions::aerialPerspective() ? kAerialPerspectiveLutSize : 0u;
  args.aerialPerspectiveDepthRange =
    RtxOptions::aerialPerspectiveDepthRangeMeters() * worldUnitsPerMeter;
  args.worldUnitsPerKilometer = worldUnitsPerMeter * 1000.0f;
  args.isZUp = RtxOptions::zUp() ? 1u : 0u;

  // Hand off to the global volumetrics froxel grid: everything nearer than its range is already
  // integrated there, so double counting is avoided by starting the atmospheric march past it.
  args.aerialPerspectiveStartDistance = RtxGlobalVolumetrics::enable()
    ? RtxGlobalVolumetrics::froxelMaxDistanceMeters() * worldUnitsPerMeter
    : 0.0f;

  return args;
}

void RtxAtmosphere::fillAerialPerspectiveArgs(AtmosphereArgs& args, const RtCamera& camera) {
  // Frustum half extents at unit forward distance, so the shader's ray direction always has a
  // forward component of exactly one and the slice index maps linearly to forward distance.
  const float tanHalfFovY = std::tan(camera.getFov() * 0.5f);
  const float tanHalfFovX = tanHalfFovY * camera.getAspectRatio();

  args.cameraPosition = camera.getPosition();
  args.cameraForward = camera.getDirection();
  args.cameraRight = camera.getRight() * tanHalfFovX;
  args.cameraUp = camera.getUp() * tanHalfFovY;
}

bool RtxAtmosphere::needsLutRecompute(const AtmosphereArgs& args) const {
  if (!m_initialized || m_lutsNeedRecompute) {
    return true;
  }

  // Only the camera independent prefix matters here; see kBakeInvariantArgsSize.
  return memcmp(&args, &m_cachedArgs, kBakeInvariantArgsSize) != 0;
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

  // Create aerial perspective volume (in-scatter in RGB, mean transmittance in A)
  VkExtent3D aerialPerspectiveExtent = {
    kAerialPerspectiveLutSize, kAerialPerspectiveLutSize, kAerialPerspectiveLutSize };
  m_aerialPerspectiveLut = Resources::createImageResource(
    ctx,
    "Atmosphere Aerial Perspective LUT",
    aerialPerspectiveExtent,
    VK_FORMAT_R16G16B16A16_SFLOAT,
    1, // numLayers
    VK_IMAGE_TYPE_3D,
    VK_IMAGE_VIEW_TYPE_3D,
    0, // imageCreateFlags
    VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags
    VkClearColorValue{}, // clearValue
    1 // mipLevels
  );
}

void RtxAtmosphere::computeLuts(Rc<DxvkContext> ctx, const AtmosphereArgs& args) {
  // One upload serves every pass below.
  ctx->updateBuffer(m_constantsBuffer, 0, sizeof(AtmosphereArgs), &args);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);

  if (needsLutRecompute(args)) {
    m_cachedArgs = args;

    // Transmittance first: the multiscattering and sky-view bakes both sample it.
    dispatchTransmittanceLut(ctx);
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

    dispatchMultiscatteringLut(ctx);
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

    dispatchSkyViewLut(ctx);
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    dispatchSkyHemisphereMean(ctx);

    m_lutsNeedRecompute = false;
  }

  // Camera fitted, so this rebuilds every frame regardless of whether the bakes above ran. The
  // barrier covers the transmittance and multiscattering LUTs it samples.
  if (args.aerialPerspectiveLutSize > 0) {
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

    dispatchAerialPerspectiveLut(ctx);
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

  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_transmittanceLut.view, nullptr);
  ctx->bindResourceView(2, m_multiscatteringLut.view, nullptr);

  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_transmittanceLut.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_multiscatteringLut.image);

  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, MultiscatteringLutShader::getShader());
  ctx->dispatch((kMultiscatteringLutSize + 7) / 8, (kMultiscatteringLutSize + 7) / 8, 1);
}

void RtxAtmosphere::dispatchSkyViewLut(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Sky View LUT");

  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_transmittanceLut.view, nullptr);
  ctx->bindResourceView(2, m_multiscatteringLut.view, nullptr);
  ctx->bindResourceView(3, m_skyViewLut.view, nullptr);
  
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

void RtxAtmosphere::dispatchAerialPerspectiveLut(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Aerial Perspective LUT");

  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_transmittanceLut.view, nullptr);
  ctx->bindResourceView(2, m_multiscatteringLut.view, nullptr);
  ctx->bindResourceView(3, m_aerialPerspectiveLut.view, nullptr);

  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_transmittanceLut.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_multiscatteringLut.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_aerialPerspectiveLut.image);

  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, AerialPerspectiveLutShader::getShader());

  const uint32_t groups = (kAerialPerspectiveLutSize + 3) / 4;
  ctx->dispatch(groups, groups, groups);
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
    const Vector3 sunIll(args.sunIlluminance.x, args.sunIlluminance.y, args.sunIlluminance.z);
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
    const Vector3 sunIll(args.sunIlluminance.x, args.sunIlluminance.y, args.sunIlluminance.z);
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
