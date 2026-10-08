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
#include "rtx_cloud_optics_tables.h"
#include "dxvk_device.h"
#include "dxvk_context.h"
#include "rtx_context.h"
#include "rtx_camera.h"
#include "rtx_debug_view.h"
#include "rtx_options.h"
#include "rtx_render/rtx_shader_manager.h"
#include "../../util/util_global_time.h"
#include "../../util/util_string.h"
#include "../../util/log/log.h"
#include "rtx/pass/clouds/cloud_binding_indices.h"
#include <rtx_shaders/cloud_placement_bake.h>
#include <rtx_shaders/cloud_detail_noise_bake.h>
#include <rtx_shaders/cloud_volume_mip_unorm.h>
#include <rtx_shaders/cloud_volume_mip_float.h>
#include <rtx_shaders/cloud_nvdf_occupancy.h>
#include <rtx_shaders/cloud_nvdf_jfa.h>
#include <rtx_shaders/cloud_nvdf_resolve.h>
#include <rtx_shaders/cloud_lighting_grid.h>
#include <rtx_shaders/cloud_diffusion_setup.h>
#include <rtx_shaders/cloud_diffusion.h>
#include <rtx_shaders/cloud_shadow_map.h>
#include <rtx_shaders/cloud_sky_sh.h>
#include <rtx_shaders/cloud_sky_ap.h>
#include <rtx_shaders/cloud_dome.h>
#include <rtx_shaders/cloud_screen.h>
#include <rtx_shaders/cloud_glossy.h>
#include <rtx_shaders/cloud_reference.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace dxvk {

  namespace {

    // Inputs of every pass that marches the layer (cloud_bindings.slangh).
#define CLOUD_MARCH_BINDINGS                                \
    CONSTANT_BUFFER(CLOUD_BINDING_CONSTANTS)                \
    SAMPLER(CLOUD_BINDING_VOLUME_SAMPLER)                   \
    SAMPLER(CLOUD_BINDING_LUT_SAMPLER)                      \
    TEXTURE3D(CLOUD_BINDING_NVDF)                           \
    TEXTURE3D(CLOUD_BINDING_DETAIL_NOISE)                   \
    TEXTURE3D(CLOUD_BINDING_SUN_GRID)                       \
    TEXTURE3D(CLOUD_BINDING_AMBIENT_GRID)                   \
    TEXTURE3D(CLOUD_BINDING_DIFFUSION_GRID)                 \
    TEXTURE3D(CLOUD_BINDING_SUN_GRID_FAR)                   \
    TEXTURE3D(CLOUD_BINDING_AMBIENT_GRID_FAR)               \
    TEXTURE3D(CLOUD_BINDING_DIFFUSION_GRID_FAR)             \
    TEXTURE2D(CLOUD_BINDING_TRANSMITTANCE_LUT)              \
    TEXTURE2D(CLOUD_BINDING_PHASE_LUT)                      \
    TEXTURE3D(CLOUD_BINDING_SKY_AP_INSCATTER)               \
    TEXTURE3D(CLOUD_BINDING_SKY_AP_TRANSMITTANCE)           \
    TEXTURE2D(CLOUD_BINDING_SKY_SH)

    class CloudPlacementBakeShader : public ManagedShader {
      SHADER_SOURCE(CloudPlacementBakeShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_placement_bake)
      BEGIN_PARAMETER()
        CONSTANT_BUFFER(CLOUD_BINDING_CONSTANTS)
        RW_TEXTURE2D(CLOUD_BINDING_BAKE_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudPlacementBakeShader);

    class CloudDetailNoiseBakeShader : public ManagedShader {
      SHADER_SOURCE(CloudDetailNoiseBakeShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_detail_noise_bake)
      BEGIN_PARAMETER()
        RW_TEXTURE3D(CLOUD_BINDING_BAKE_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudDetailNoiseBakeShader);

    class CloudVolumeMipUnormShader : public ManagedShader {
      SHADER_SOURCE(CloudVolumeMipUnormShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_volume_mip_unorm)
      BEGIN_PARAMETER()
        TEXTURE3D(CLOUD_BINDING_BAKE_INPUT)
        SAMPLER(CLOUD_BINDING_VOLUME_SAMPLER)
        RW_TEXTURE3D(CLOUD_BINDING_BAKE_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudVolumeMipUnormShader);

    class CloudVolumeMipFloatShader : public ManagedShader {
      SHADER_SOURCE(CloudVolumeMipFloatShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_volume_mip_float)
      BEGIN_PARAMETER()
        TEXTURE3D(CLOUD_BINDING_BAKE_INPUT)
        SAMPLER(CLOUD_BINDING_VOLUME_SAMPLER)
        RW_TEXTURE3D(CLOUD_BINDING_BAKE_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudVolumeMipFloatShader);

    class CloudNvdfOccupancyShader : public ManagedShader {
      SHADER_SOURCE(CloudNvdfOccupancyShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_nvdf_occupancy)
      BEGIN_PARAMETER()
        CONSTANT_BUFFER(CLOUD_BINDING_CONSTANTS)
        TEXTURE2D(CLOUD_BINDING_PLACEMENT_MAP)
        SAMPLER(CLOUD_BINDING_VOLUME_SAMPLER)
        RW_TEXTURE3D(CLOUD_BINDING_BAKE_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudNvdfOccupancyShader);

    class CloudNvdfJfaShader : public ManagedShader {
      SHADER_SOURCE(CloudNvdfJfaShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_nvdf_jfa)
      PUSH_CONSTANTS(CloudNvdfJfaArgs)
      BEGIN_PARAMETER()
        CONSTANT_BUFFER(CLOUD_BINDING_CONSTANTS)
        TEXTURE3D(CLOUD_BINDING_BAKE_INPUT)
        TEXTURE3D(CLOUD_BINDING_BAKE_INPUT2)
        RW_TEXTURE3D(CLOUD_BINDING_BAKE_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudNvdfJfaShader);

    class CloudNvdfResolveShader : public ManagedShader {
      SHADER_SOURCE(CloudNvdfResolveShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_nvdf_resolve)
      BEGIN_PARAMETER()
        CONSTANT_BUFFER(CLOUD_BINDING_CONSTANTS)
        TEXTURE3D(CLOUD_BINDING_BAKE_INPUT)
        TEXTURE3D(CLOUD_BINDING_BAKE_INPUT2)
        RW_TEXTURE3D(CLOUD_BINDING_BAKE_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudNvdfResolveShader);

    class CloudLightingGridShader : public ManagedShader {
      SHADER_SOURCE(CloudLightingGridShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_lighting_grid)
      PUSH_CONSTANTS(CloudPassArgs)
      BEGIN_PARAMETER()
        CONSTANT_BUFFER(CLOUD_BINDING_CONSTANTS)
        SAMPLER(CLOUD_BINDING_VOLUME_SAMPLER)
        TEXTURE3D(CLOUD_BINDING_NVDF)
        TEXTURE3D(CLOUD_BINDING_DETAIL_NOISE)
        RW_TEXTURE3D(CLOUD_BINDING_BAKE_OUTPUT)
        RW_TEXTURE3D(CLOUD_BINDING_BAKE_OUTPUT2)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudLightingGridShader);

    class CloudDiffusionSetupShader : public ManagedShader {
      SHADER_SOURCE(CloudDiffusionSetupShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_diffusion_setup)
      PUSH_CONSTANTS(CloudPassArgs)
      BEGIN_PARAMETER()
        CONSTANT_BUFFER(CLOUD_BINDING_CONSTANTS)
        SAMPLER(CLOUD_BINDING_VOLUME_SAMPLER)
        TEXTURE3D(CLOUD_BINDING_BAKE_INPUT)
        RW_TEXTURE3D(CLOUD_BINDING_BAKE_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudDiffusionSetupShader);

    class CloudDiffusionShader : public ManagedShader {
      SHADER_SOURCE(CloudDiffusionShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_diffusion)
      PUSH_CONSTANTS(CloudPassArgs)
      BEGIN_PARAMETER()
        CONSTANT_BUFFER(CLOUD_BINDING_CONSTANTS)
        TEXTURE3D(CLOUD_BINDING_BAKE_INPUT)
        RW_TEXTURE3D(CLOUD_BINDING_BAKE_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudDiffusionShader);

    class CloudShadowMapShader : public ManagedShader {
      SHADER_SOURCE(CloudShadowMapShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_shadow_map)
      PUSH_CONSTANTS(CloudPassArgs)
      BEGIN_PARAMETER()
        CONSTANT_BUFFER(CLOUD_BINDING_CONSTANTS)
        SAMPLER(CLOUD_BINDING_VOLUME_SAMPLER)
        TEXTURE3D(CLOUD_BINDING_NVDF)
        TEXTURE3D(CLOUD_BINDING_DETAIL_NOISE)
        RW_TEXTURE2D(CLOUD_BINDING_BAKE_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudShadowMapShader);

    class CloudSkyShShader : public ManagedShader {
      SHADER_SOURCE(CloudSkyShShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_sky_sh)
      PUSH_CONSTANTS(CloudPassArgs)
      BEGIN_PARAMETER()
        CONSTANT_BUFFER(CLOUD_BINDING_CONSTANTS)
        SAMPLER(CLOUD_BINDING_VOLUME_SAMPLER)
        TEXTURE3D(CLOUD_BINDING_SUN_GRID)
        TEXTURE2D(CLOUD_BINDING_TRANSMITTANCE_LUT)
        TEXTURE2D(CLOUD_BINDING_MULTISCATTERING_LUT)
        TEXTURE2D(CLOUD_BINDING_AEROSOL_PHASE_LUT)
        RW_TEXTURE2D(CLOUD_BINDING_BAKE_OUTPUT)
        RW_TEXTURE2D(CLOUD_BINDING_BAKE_OUTPUT2)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudSkyShShader);

    class CloudSkyApShader : public ManagedShader {
      SHADER_SOURCE(CloudSkyApShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_sky_ap)
      PUSH_CONSTANTS(CloudPassArgs)
      BEGIN_PARAMETER()
        CONSTANT_BUFFER(CLOUD_BINDING_CONSTANTS)
        SAMPLER(CLOUD_BINDING_VOLUME_SAMPLER)
        TEXTURE3D(CLOUD_BINDING_SUN_GRID)
        TEXTURE2D(CLOUD_BINDING_SHADOW_MAP)
        TEXTURE2D(CLOUD_BINDING_TRANSMITTANCE_LUT)
        TEXTURE2D(CLOUD_BINDING_MULTISCATTERING_LUT)
        TEXTURE2D(CLOUD_BINDING_AEROSOL_PHASE_LUT)
        RW_TEXTURE3D(CLOUD_BINDING_BAKE_OUTPUT)
        RW_TEXTURE3D(CLOUD_BINDING_BAKE_OUTPUT2)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudSkyApShader);

    class CloudDomeShader : public ManagedShader {
      SHADER_SOURCE(CloudDomeShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_dome)
      PUSH_CONSTANTS(CloudPassArgs)
      BEGIN_PARAMETER()
        CLOUD_MARCH_BINDINGS
        TEXTURE2D(CLOUD_BINDING_BAKE_INPUT)
        RW_TEXTURE2D(CLOUD_BINDING_BAKE_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudDomeShader);

    class CloudScreenShader : public ManagedShader {
      SHADER_SOURCE(CloudScreenShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_screen)
      BEGIN_PARAMETER()
        CLOUD_MARCH_BINDINGS
        CONSTANT_BUFFER(CLOUD_BINDING_CAMERA)
        TEXTURE2DARRAY(CLOUD_BINDING_BLUE_NOISE)
        TEXTURE2D(CLOUD_BINDING_PRIMARY_LINEAR_VIEW_Z)
        TEXTURE2D(CLOUD_BINDING_PSR_FIRST_HIT_DISTANCE)
        TEXTURE2D(CLOUD_BINDING_PSR_REFLECTION_SEGMENT)
        TEXTURE2D(CLOUD_BINDING_PSR_REFLECTION_DIRECTION)
        TEXTURE2D(CLOUD_BINDING_SHARED_FLAGS)
        TEXTURE2D(CLOUD_BINDING_HISTORY)
        TEXTURE2D(CLOUD_BINDING_HISTORY_AGE)
        TEXTURE2D(CLOUD_BINDING_DOME_LUT)
        TEXTURE2D(CLOUD_BINDING_SHADOW_MAP)
        RW_TEXTURE2D(CLOUD_BINDING_LAYER_OUTPUT)
        RW_TEXTURE2D(CLOUD_BINDING_HISTORY_AGE_OUTPUT)
        RW_TEXTURE2D(CLOUD_BINDING_COMPOSITE_LAYER_OUTPUT)
        RW_TEXTURE2D(CLOUD_BINDING_REFLECTION_OUTPUT)
        TEXTURE2D(CLOUD_BINDING_REFLECTION_HISTORY)
        RW_TEXTURE2D(CLOUD_BINDING_DEBUG_VIEW)
        RW_TEXTURE2D(CLOUD_BINDING_MOTION_VECTOR_OUTPUT)
        RW_TEXTURE2D(CLOUD_BINDING_MOTION_VECTOR_RR_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudScreenShader);

    class CloudGlossyShader : public ManagedShader {
      SHADER_SOURCE(CloudGlossyShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_glossy)
      BEGIN_PARAMETER()
        CLOUD_MARCH_BINDINGS
        CONSTANT_BUFFER(CLOUD_BINDING_CAMERA)
        TEXTURE2DARRAY(CLOUD_BINDING_BLUE_NOISE)
        TEXTURE2D(CLOUD_BINDING_DOME_LUT)
        TEXTURE2D(CLOUD_BINDING_SKY_VIEW_LUT)
        RW_TEXTURE2D(CLOUD_BINDING_GLOSSY_RAY)
        RW_TEXTURE2D(CLOUD_BINDING_INDIRECT_RADIANCE)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudGlossyShader);

    class CloudReferenceShader : public ManagedShader {
      SHADER_SOURCE(CloudReferenceShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_reference)
      PUSH_CONSTANTS(CloudPassArgs)
      BEGIN_PARAMETER()
        CLOUD_MARCH_BINDINGS
        CONSTANT_BUFFER(CLOUD_BINDING_CAMERA)
        TEXTURE2D(CLOUD_BINDING_PRIMARY_LINEAR_VIEW_Z)
        TEXTURE2D(CLOUD_BINDING_PSR_FIRST_HIT_DISTANCE)
        TEXTURE2D(CLOUD_BINDING_SHARED_FLAGS)
        TEXTURE2D(CLOUD_BINDING_HISTORY)
        TEXTURE2D(CLOUD_BINDING_PHASE_CDF)
        RW_TEXTURE2D(CLOUD_BINDING_REFERENCE_ACCUMULATION)
        RW_TEXTURE2D(CLOUD_BINDING_REFERENCE_PATH_POSITION)
        RW_TEXTURE2D(CLOUD_BINDING_REFERENCE_PATH_DIRECTION)
        RW_TEXTURE2D(CLOUD_BINDING_REFERENCE_PATH_THROUGHPUT)
        RW_TEXTURE2D(CLOUD_BINDING_REFERENCE_PATH_RADIANCE)
        RW_STRUCTURED_BUFFER(CLOUD_BINDING_REFERENCE_STATISTICS)
        RW_TEXTURE2D(CLOUD_BINDING_LAYER_OUTPUT)
        RW_TEXTURE2D(CLOUD_BINDING_DEBUG_VIEW)
      END_PARAMETER()
    };

#undef CLOUD_MARCH_BINDINGS

    constexpr float kCloudPi = 3.14159265358979323846f;

    // The optical depth grids' window around the camera, and their far cascade's (250 m texels) for the lighting
    // of distant clouds.
    constexpr float kGridExtentKm = 12.0f;
    constexpr float kFarGridExtentKm = 96.0f;
    constexpr float kShadowMapExtentKm = 64.0f;

    // Red-black sweeps the diffusion floor's solve converges in from any start, within a thousandth on clouds
    // some tens of cells across.
    constexpr uint32_t kDiffusionSettleSweeps = 128;

    constexpr uint64_t kHashBasis = 14695981039346656037ull;

    // FNV-1a.
    uint64_t hashBytes(const void* data, size_t size, uint64_t hash) {
      const uint8_t* bytes = static_cast<const uint8_t*>(data);
      for (size_t i = 0; i < size; ++i) {
        hash = (hash ^ bytes[i]) * 1099511628211ull;
      }
      return hash;
    }

    // The atmosphere's camera independent leading fields, the ones its own LUTs rebake on.
    uint64_t hashAtmosphere(const AtmosphereArgs& atmosphere, uint64_t hash) {
      return hashBytes(&atmosphere, offsetof(AtmosphereArgs, aerialPerspectiveLutSize), hash);
    }

    // Jump flooding schedule over the 256 voxel torus, then one more unit pass (JFA+1), which catches most of
    // the far seeds plain JFA misses.
    constexpr uint32_t kNvdfJumpSchedule[] = { 128, 64, 32, 16, 8, 4, 2, 1, 1 };
    constexpr uint32_t kNvdfJumpPassCount = sizeof(kNvdfJumpSchedule) / sizeof(kNvdfJumpSchedule[0]);
    constexpr uint32_t kNvdfJumpPassesPerFrame = 2;

    // Ratio of the volume mean radius cubed to the effective radius cubed for continental droplet spectra
    // (Martin, Johnson and Spice 1994): r_eff = r_v / k^(1/3).
    constexpr float kDropletSpectralShapeK = 0.8f;

    // Droplet optics at an effective radius, interpolated in the table (green channel), and the fractional row
    // of the phase LUT.
    struct DropletOptics {
      float extinctionEfficiency;
      float asymmetry;
      float forwardPeakFraction;
      float truncatedAsymmetry;
      float row;
    };

    DropletOptics lookupDropletOptics(float effectiveRadiusUm) {
      using namespace cloud_tables;
      const float r = std::clamp(effectiveRadiusUm, kEffectiveRadiusUm[0], kEffectiveRadiusUm[kRadiusCount - 1]);
      uint32_t i = 0;
      while (i + 2 < kRadiusCount && r > kEffectiveRadiusUm[i + 1]) {
        ++i;
      }
      const float t = std::clamp((r - kEffectiveRadiusUm[i]) / (kEffectiveRadiusUm[i + 1] - kEffectiveRadiusUm[i]), 0.0f, 1.0f);
      const DropletOptics a { kOptics[i].extinctionEfficiency[1], kOptics[i].asymmetry[1], kOptics[i].forwardPeakFraction[1], kOptics[i].truncatedAsymmetry[1], 0.0f };
      const DropletOptics b { kOptics[i + 1].extinctionEfficiency[1], kOptics[i + 1].asymmetry[1], kOptics[i + 1].forwardPeakFraction[1], kOptics[i + 1].truncatedAsymmetry[1], 0.0f };
      DropletOptics optics;
      optics.extinctionEfficiency = a.extinctionEfficiency + (b.extinctionEfficiency - a.extinctionEfficiency) * t;
      optics.asymmetry = a.asymmetry + (b.asymmetry - a.asymmetry) * t;
      optics.forwardPeakFraction = a.forwardPeakFraction + (b.forwardPeakFraction - a.forwardPeakFraction) * t;
      optics.truncatedAsymmetry = a.truncatedAsymmetry + (b.truncatedAsymmetry - a.truncatedAsymmetry) * t;
      optics.row = float(i) + t;
      return optics;
    }

    // Effective radius (um) of droplets holding a liquid water content (g/m^3) in a number concentration (cm^-3).
    float effectiveRadiusUm(float liquidWaterContent, float dropletConcentration) {
      const float lwcKgPerM3 = std::max(liquidWaterContent, 1e-4f) * 1e-3f;
      const float numberPerM3 = std::max(dropletConcentration, 1.0f) * 1e6f;
      const float volumeRadiusM = std::cbrt(3.0f * lwcKgPerM3 / (4.0f * kCloudPi * 1000.0f * numberPerM3));
      return volumeRadiusM / std::cbrt(kDropletSpectralShapeK) * 1e6f;
    }

    // Extinction (km^-1) of liquid water content (g/m^3) in droplets of an effective radius (Stephens 1978):
    // beta = 3 Q LWC / (4 rho_w r_eff).
    float extinctionPerKm(float liquidWaterContent, float effectiveRadiusUm, float extinctionEfficiency) {
      const float lwcKgPerM3 = std::max(liquidWaterContent, 0.0f) * 1e-3f;
      const float radiusM = std::max(effectiveRadiusUm, 0.1f) * 1e-6f;
      return 3.0f * extinctionEfficiency * lwcKgPerM3 / (4.0f * 1000.0f * radiusM) * 1000.0f;
    }
  }

  bool RtxClouds::NvdfKey::operator==(const NvdfKey& other) const {
    return cellSizeKm == other.cellSizeKm
        && tileKm == other.tileKm
        && columnFeather == other.columnFeather
        && columnTopShape == other.columnTopShape
        && columnTopVariation == other.columnTopVariation
        && columnBaseVariation == other.columnBaseVariation
        && columnTopFlatten == other.columnTopFlatten
        && std::abs(nominalCoverage - other.nominalCoverage) <= 1e-4f
        && thicknessQuantised == other.thicknessQuantised
        && bodyErosion == other.bodyErosion;
  }

  RtxClouds::NvdfKey RtxClouds::makeNvdfKey(const CloudArgs& args) {
    NvdfKey key;
    key.cellSizeKm = args.cellSizeKm;
    key.tileKm = args.tileKm;
    key.columnFeather = args.columnFeather;
    key.columnTopShape = args.columnTopShape;
    key.columnTopVariation = args.columnTopVariation;
    key.columnBaseVariation = args.columnBaseVariation;
    key.columnTopFlatten = args.columnTopFlatten;
    key.nominalCoverage = args.nvdfNominalCoverage;
    // The voxel metric depends on the thickness; small changes only stretch the field.
    key.thicknessQuantised = std::round(args.thicknessKm / 0.25f) * 0.25f;
    key.bodyErosion = args.nvdfBodyErosion;
    return key;
  }

  uint64_t RtxClouds::computeInputsKey(const CloudArgs& args, const AtmosphereArgs& atmosphere) {
    // The frame's own fields (index, history flags, debug view, pixel footprint, motion steps), the per pixel
    // reflection paths no bake reads, the camera's and the windows' positions, which the bakes track themselves,
    // and the wind's offset, which carries the field and the bakes anchored to it alike.
    CloudArgs invariant = args;
    invariant.frameIndex = 0;
    invariant.flags &= ~(CLOUD_FLAG_HISTORY_VALID | CLOUD_FLAG_SHADOW_MAP_VALID | CLOUD_FLAG_PSR_REFLECTIONS | CLOUD_FLAG_GLOSSY_REFLECTIONS);
    invariant.debugView = 0;
    invariant.cameraPositionKm = Vector3(0.0f);
    invariant.cameraWorldHeightKm = 0.0f;
    invariant.gridOriginKm = Vector4(0.0f);
    invariant.farGridOriginKm = Vector2(0.0f);
    invariant.pixelAngle = 0.0f;
    invariant.windOffsetKm = Vector2(0.0f);
    invariant.windStepKm = Vector2(0.0f);
    invariant.riseStepKm = 0.0f;
    invariant.shearStepKm = 0.0f;
    return hashAtmosphere(atmosphere, hashBytes(&invariant, sizeof(invariant), kHashBasis));
  }

  uint64_t RtxClouds::computeSkyLightKey(const CloudArgs& args, const AtmosphereArgs& atmosphere) {
    const float layer[] = { args.baseAltitudeKm, args.thicknessKm, args.groundBounceStrength };
    return hashAtmosphere(atmosphere, hashBytes(layer, sizeof(layer), kHashBasis));
  }

  RtxClouds::TierSettings RtxClouds::getTierSettings(CloudQuality quality) {
    switch (quality) {
    case CloudQuality::Low:
      return TierSettings { 48, 0.8f, 0.06f, 1.5f, 0.02f, 0, 8, 512, 4, 0.05f, 0.2f, 2, 2 };
    case CloudQuality::Medium:
      return TierSettings { 96, 0.5f, 0.04f, 1.0f, 0.01f, 2, 4, 1024, 4, 0.05f, 0.3f, 3, 2 };
    default:
    case CloudQuality::High:
      return TierSettings { 160, 0.3f, 0.025f, 0.6f, 0.005f, 3, 2, 1024, 4, 0.08f, 0.4f, 3, 4 };
    case CloudQuality::Ultra:
      return TierSettings { 256, 0.2f, 0.015f, 0.4f, 0.002f, 4, 1, 2048, 4, 0.1f, 0.5f, 4, 4 };
    }
  }

  RtxClouds::GenusSettings RtxClouds::getGenusSettings() {
    switch (genus()) {
    case CloudGenus::CumulusHumilis:
      return GenusSettings { 1200.0f, 800.0f, 0.35f, 0.6f, 2.0f, 0.6f, 0.4f, 0.08f, 0.35f, 1.0f, 0.35f, 400.0f };
    case CloudGenus::CumulusMediocris:
      return GenusSettings { 1300.0f, 2500.0f, 0.45f, 0.75f, 3.65f, 0.4f, 0.45f, 0.12f, 0.35f, 1.0f, 0.55f, 350.0f };
    case CloudGenus::CumulusCongestus:
      return GenusSettings { 1300.0f, 4500.0f, 0.4f, 0.9f, 4.5f, 0.35f, 0.55f, 0.12f, 0.3f, 1.0f, 0.9f, 300.0f };
    case CloudGenus::Stratocumulus:
      return GenusSettings { 900.0f, 700.0f, 0.8f, 0.45f, 2.5f, 0.8f, 0.15f, 0.05f, 0.5f, 0.85f, 0.3f, 200.0f };
    case CloudGenus::Altocumulus:
      return GenusSettings { 3500.0f, 600.0f, 0.55f, 0.3f, 1.2f, 0.7f, 0.2f, 0.05f, 0.45f, 0.8f, 0.15f, 150.0f };
    case CloudGenus::Stratus:
      return GenusSettings { 600.0f, 600.0f, 0.95f, 0.25f, 3.0f, 1.0f, 0.1f, 0.03f, 0.6f, 0.85f, 0.3f, 250.0f };
    default:
    case CloudGenus::Custom:
      return GenusSettings {
        baseAltitudeMeters(), thicknessMeters(), coverage(), cloudType(), cellSizeKm(), columnTopShape(),
        columnTopVariation(), columnBaseVariation(), columnFeather(), columnTopFlatten(), liquidWaterContent(),
        dropletConcentration() };
    }
  }

  RtxClouds::DropletReadout RtxClouds::computeDropletReadout() {
    const GenusSettings g = getGenusSettings();
    DropletReadout readout;
    readout.effectiveRadiusUm = effectiveRadiusUm(g.liquidWaterContent, g.dropletConcentration);
    const DropletOptics optics = lookupDropletOptics(readout.effectiveRadiusUm);
    readout.extinctionPerKm = extinctionPerKm(g.liquidWaterContent, readout.effectiveRadiusUm, optics.extinctionEfficiency) * std::max(densityScale(), 0.0f);
    readout.asymmetry = optics.asymmetry;
    readout.forwardPeakFraction = optics.forwardPeakFraction;
    readout.truncatedAsymmetry = optics.truncatedAsymmetry;

    // The column's optical depth through the vertical profile at full density, from the base to the capped top,
    // with the profile's limits as buildArgs passes them on.
    const float columnKm = std::max(g.thicknessMeters * 0.001f, 0.05f) * std::clamp(g.columnTopFlatten, 0.05f, 1.0f);
    const float exponent = std::clamp(adiabaticExponent(), 0.0f, 2.0f);
    const float profileFloor = std::clamp(adiabaticFloor(), 0.0f, 1.0f);
    const float profileMax = std::max(adiabaticMax(), profileFloor);
    float integral = 0.0f;
    constexpr int kSteps = 64;
    for (int i = 0; i < kSteps; ++i) {
      const float heightKm = (float(i) + 0.5f) / float(kSteps) * columnKm;
      const float profile = std::clamp(std::pow(std::max(heightKm, 1e-3f), exponent), profileFloor, profileMax);
      integral += profile * columnKm / float(kSteps);
    }
    readout.verticalOpticalDepth = integral * readout.extinctionPerKm;
    return readout;
  }

  RtxClouds::RtxClouds(DxvkDevice* device)
    : CommonDeviceObject(device) {
    DxvkBufferCreateInfo info = {};
    info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    info.access = VK_ACCESS_UNIFORM_READ_BIT;
    info.size = sizeof(CloudConstants);
    m_constantsBuffer = device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Cloud constants buffer");
    m_nvdfConstantsBuffer = device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Cloud field bake constants buffer");
  }

  bool RtxClouds::isActive() {
    return enable() && RtxOptions::skyMode() == SkyMode::PhysicalAtmosphere;
  }

  Rc<DxvkSampler> RtxClouds::getVolumeSampler(Rc<DxvkContext> ctx) const {
    // Every cloud texture is read through a repeating trilinear sampler; axes that do not wrap clamp in the
    // shader (cloudClampToTexelCentres).
    return static_cast<RtxContext*>(ctx.ptr())->getResourceManager().getSampler(
      VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);
  }

  Rc<DxvkSampler> RtxClouds::getLutSampler(Rc<DxvkContext> ctx) const {
    return static_cast<RtxContext*>(ctx.ptr())->getResourceManager().getSampler(
      VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
  }

  void RtxClouds::barrier(Rc<DxvkContext> ctx) {
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  }

  void RtxClouds::initialize(Rc<DxvkContext> ctx) {
    if (m_initialized) {
      return;
    }

    createResources(ctx);
    uploadPhaseLut(ctx);

    // The fixed detail pattern. The placement map and the body field bake with their inputs on first use.
    bakeDetailNoise(ctx);
    barrier(ctx);

    m_initialized = true;
  }

  void RtxClouds::createResources(Rc<DxvkContext> ctx) {
    const auto create3D = [&](const char* name, VkExtent3D extent, VkFormat format, uint32_t mips) {
      return Resources::createImageResource(ctx, name, extent, format, 1, VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D,
        0, VK_IMAGE_USAGE_STORAGE_BIT, VkClearColorValue {}, mips);
    };
    const auto create2D = [&](const char* name, VkExtent3D extent, VkFormat format, VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT) {
      return Resources::createImageResource(ctx, name, extent, format, 1, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D,
        0, usage, VkClearColorValue {}, 1);
    };
    const auto createMipViews = [&](const Resources::Resource& resource, VkFormat format, uint32_t mips, std::vector<Rc<DxvkImageView>>& views) {
      DxvkImageViewCreateInfo viewInfo;
      viewInfo.type = VK_IMAGE_VIEW_TYPE_3D;
      viewInfo.format = format;
      viewInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
      viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
      viewInfo.minLayer = 0;
      viewInfo.numLayers = 1;
      viewInfo.numLevels = 1;
      views.clear();
      for (uint32_t level = 0; level < mips; ++level) {
        viewInfo.minLevel = level;
        views.push_back(ctx->getDevice()->createImageView(resource.image, viewInfo));
      }
    };

    const VkExtent3D nvdfExtent = { CLOUD_NVDF_SIZE_XZ, CLOUD_NVDF_SIZE_Y, CLOUD_NVDF_SIZE_XZ };
    const VkExtent3D gridExtent = { CLOUD_GRID_SIZE_XZ, CLOUD_GRID_SIZE_Y, CLOUD_GRID_SIZE_XZ };
    const VkExtent3D farGridExtent = { CLOUD_FAR_GRID_SIZE_XZ, CLOUD_GRID_SIZE_Y, CLOUD_FAR_GRID_SIZE_XZ };

    m_placementMap = create2D("Cloud Placement Map", VkExtent3D { CLOUD_PLACEMENT_MAP_SIZE, CLOUD_PLACEMENT_MAP_SIZE, 1 }, VK_FORMAT_R8G8B8A8_UNORM);
    m_detailNoise = create3D("Cloud Detail Noise", VkExtent3D { CLOUD_DETAIL_NOISE_SIZE, CLOUD_DETAIL_NOISE_SIZE, CLOUD_DETAIL_NOISE_SIZE },
      VK_FORMAT_R8G8B8A8_UNORM, CLOUD_DETAIL_NOISE_MIP_COUNT);
    createMipViews(m_detailNoise, VK_FORMAT_R8G8B8A8_UNORM, CLOUD_DETAIL_NOISE_MIP_COUNT, m_detailNoiseMipViews);

    m_nvdfSdf[0] = create3D("Cloud NVDF 0", nvdfExtent, VK_FORMAT_R16_SFLOAT, 1);
    m_nvdfSdf[1] = create3D("Cloud NVDF 1", nvdfExtent, VK_FORMAT_R16_SFLOAT, 1);

    m_sunGrid = create3D("Cloud Sun Optical Depth Grid", gridExtent, VK_FORMAT_R16G16_SFLOAT, CLOUD_SUN_GRID_MIP_COUNT);
    createMipViews(m_sunGrid, VK_FORMAT_R16G16_SFLOAT, CLOUD_SUN_GRID_MIP_COUNT, m_sunGridMipViews);
    m_ambientGrid = create3D("Cloud Vertical Optical Depth Grid", gridExtent, VK_FORMAT_R16G16_SFLOAT, 1);
    m_sunGridFar = create3D("Cloud Sun Optical Depth Grid Far", farGridExtent, VK_FORMAT_R16G16_SFLOAT, 1);
    m_ambientGridFar = create3D("Cloud Vertical Optical Depth Grid Far", farGridExtent, VK_FORMAT_R16G16_SFLOAT, 1);
    m_diffusionCells[0] = create3D("Cloud Diffusion Cells", gridExtent, VK_FORMAT_R32G32_SFLOAT, 1);
    m_diffusionCells[1] = create3D("Cloud Diffusion Cells Far", farGridExtent, VK_FORMAT_R32G32_SFLOAT, 1);
    m_diffusionFluence[0] = create3D("Cloud Diffusion Fluence", gridExtent, VK_FORMAT_R32_SFLOAT, 1);
    m_diffusionFluence[1] = create3D("Cloud Diffusion Fluence Far", farGridExtent, VK_FORMAT_R32_SFLOAT, 1);

    const VkExtent3D skyApExtent = { CLOUD_SKY_AP_LUT_WIDTH, CLOUD_SKY_AP_LUT_HEIGHT, CLOUD_SKY_AP_LUT_DEPTH };
    m_skyApInScatter = create3D("Cloud Sky Aerial Perspective In-Scatter", skyApExtent, VK_FORMAT_R16G16B16A16_SFLOAT, 1);
    m_skyApTransmittance = create3D("Cloud Sky Aerial Perspective Transmittance", skyApExtent, VK_FORMAT_R16G16B16A16_SFLOAT, 1);

    m_skySh = create2D("Cloud Sky Harmonics", VkExtent3D { CLOUD_SKY_SH_COEFFICIENTS, 2 * CLOUD_SKY_SH_ALTITUDES, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT);
    m_skyShParts = create2D("Cloud Sky Harmonics Parts", VkExtent3D { CLOUD_SKY_SH_COEFFICIENTS, 3 * CLOUD_SKY_SH_ALTITUDES, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT);
    m_phaseLut = create2D("Cloud Droplet Phase LUT", VkExtent3D { CLOUD_PHASE_LUT_SIZE, CLOUD_PHASE_LUT_RADII, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT, 0);
    m_phaseCdf = create2D("Cloud Droplet Phase Inverse CDF", VkExtent3D { cloud_tables::kInverseCdfSize, CLOUD_PHASE_LUT_RADII, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT, 0);
    m_shadowMap = create2D("Cloud Far Shadow Map", VkExtent3D { CLOUD_SHADOW_MAP_SIZE, CLOUD_SHADOW_MAP_SIZE, 1 }, VK_FORMAT_R16_SFLOAT);

    const VkDeviceSize statisticsSize = VkDeviceSize(kMaxFramesInFlight) * CLOUD_REFERENCE_STATISTICS_SIZE * sizeof(uint32_t);
    DxvkBufferCreateInfo statisticsInfo;
    statisticsInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    statisticsInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    statisticsInfo.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    statisticsInfo.size = statisticsSize;
    m_referenceStatistics = ctx->getDevice()->createBuffer(statisticsInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXBuffer, "Cloud Reference Statistics");
    DxvkBufferCreateInfo readbackInfo;
    readbackInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    readbackInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
    readbackInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    readbackInfo.size = statisticsSize;
    m_referenceStatisticsReadback = ctx->getDevice()->createBuffer(readbackInfo, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      DxvkMemoryStats::Category::RTXBuffer, "Cloud Reference Statistics Readback");
    std::memset(m_referenceStatisticsReadback->mapPtr(0), 0, size_t(statisticsSize));

    ensureDome(ctx, getTierSettings(quality()).domeWidth);
  }

  void RtxClouds::ensureDome(Rc<DxvkContext> ctx, uint32_t width) {
    if (width == m_domeWidth && m_dome[0].isValid()) {
      return;
    }
    // Transparent until the first bake, so misses see the plain sky.
    VkClearColorValue clear = {};
    clear.float32[3] = 1.0f;
    for (uint32_t i = 0; i < 2; ++i) {
      m_dome[i] = Resources::createImageResource(ctx, i == 0 ? "Cloud Dome 0" : "Cloud Dome 1",
        VkExtent3D { width, std::max(width / 2, 1u), 1 }, VK_FORMAT_R16G16B16A16_SFLOAT, 1, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D,
        0, VK_IMAGE_USAGE_STORAGE_BIT, clear, 1);
    }
    m_domeWidth = width;
    m_domeHistoryValid = false;
  }

  void RtxClouds::ensureScreenResources(Rc<DxvkContext> ctx, const VkExtent3D& extent) {
    if (extent.width == m_screenExtent.width && extent.height == m_screenExtent.height && m_layer[0].isValid()) {
      return;
    }
    const VkExtent3D size = { extent.width, extent.height, 1 };
    for (uint32_t i = 0; i < 2; ++i) {
      m_layer[i] = Resources::createImageResource(ctx, i == 0 ? "Cloud Layer 0" : "Cloud Layer 1", size, VK_FORMAT_R16G16B16A16_SFLOAT);
      m_layerAge[i] = Resources::createImageResource(ctx, i == 0 ? "Cloud Layer Age 0" : "Cloud Layer Age 1", size, VK_FORMAT_R16_SFLOAT);
    }
    m_compositeLayer = Resources::createImageResource(ctx, "Cloud Composite Layer", size, VK_FORMAT_R16G16B16A16_SFLOAT);
    for (uint32_t i = 0; i < 2; ++i) {
      m_reflection[i] = Resources::createImageResource(ctx, i == 0 ? "Cloud Reflection Layer 0" : "Cloud Reflection Layer 1", size, VK_FORMAT_R16G16B16A16_SFLOAT);
    }
    m_glossyRay = Resources::createImageResource(ctx, "Cloud Glossy Sky Rays", size, VK_FORMAT_R32G32_UINT);
    m_screenExtent = size;
    m_screenHistoryValid = false;
    // The reference's targets follow the screen's size when it next runs.
    m_referenceFrames = 0;
    releaseReferenceResources();
  }

  void RtxClouds::ensureReferenceResources(Rc<DxvkContext> ctx) {
    if (m_referenceAccumulation.isValid()) {
      return;
    }
    m_referenceAccumulation = Resources::createImageResource(ctx, "Cloud Reference Accumulation", m_screenExtent, VK_FORMAT_R32G32B32A32_SFLOAT);
    m_referenceMean = Resources::createImageResource(ctx, "Cloud Reference Mean", m_screenExtent, VK_FORMAT_R16G16B16A16_SFLOAT);
    const VkExtent3D tiles = { (m_screenExtent.width + 3) / 4, (m_screenExtent.height + 3) / 4, 1 };
    const char* pathNames[4] = { "Cloud Reference Path Position", "Cloud Reference Path Direction", "Cloud Reference Path Throughput", "Cloud Reference Path Radiance" };
    for (uint32_t i = 0; i < 4; ++i) {
      m_referencePath[i] = Resources::createImageResource(ctx, pathNames[i], tiles, VK_FORMAT_R32G32B32A32_SFLOAT);
    }
    // New targets hold nothing to continue from.
    m_referenceFrames = 0;
  }

  void RtxClouds::releaseReferenceResources() {
    m_referenceAccumulation.reset();
    m_referenceMean.reset();
    for (Resources::Resource& path : m_referencePath) {
      path.reset();
    }
  }

  void RtxClouds::releaseResources() {
    for (Resources::Resource* resource : {
           &m_placementMap, &m_detailNoise, &m_nvdfOccupancy, &m_nvdfSeeds[0], &m_nvdfSeeds[1], &m_nvdfSdf[0], &m_nvdfSdf[1],
           &m_sunGrid, &m_ambientGrid, &m_sunGridFar, &m_ambientGridFar, &m_diffusionCells[0], &m_diffusionCells[1],
           &m_diffusionFluence[0], &m_diffusionFluence[1], &m_skyApInScatter, &m_skyApTransmittance, &m_skySh, &m_skyShParts,
           &m_phaseLut, &m_phaseCdf, &m_shadowMap, &m_dome[0], &m_dome[1], &m_layer[0], &m_layer[1], &m_layerAge[0],
           &m_layerAge[1], &m_compositeLayer, &m_reflection[0], &m_reflection[1], &m_glossyRay }) {
      resource->reset();
    }
    m_detailNoiseMipViews.clear();
    m_sunGridMipViews.clear();
    m_referenceStatistics = nullptr;
    m_referenceStatisticsReadback = nullptr;
    releaseReferenceResources();

    // What described their contents.
    m_initialized = false;
    m_nvdfValid = false;
    m_nvdfBakeActive = false;
    m_nvdfJumpIndex = 0;
    m_nvdfFront = 0;
    m_publishedNvdfKey = NvdfKey {};
    m_pendingNvdfKey = NvdfKey {};
    m_gridsNeedFullBake = true;
    m_diffusionSettleSweeps[0] = m_diffusionSettleSweeps[1] = 0;
    m_farDiffusionCatchUpSweeps = 0;
    m_skyApValid = false;
    m_skyLightKey = 0;
    m_lastInputsKey = 0;
    m_lastRestartKey = 0;
    m_domeWidth = 0;
    m_domeIndex = 0;
    m_domeHistoryValid = false;
    m_screenExtent = VkExtent3D { 0, 0, 0 };
    m_layerIndex = 0;
    m_screenHistoryValid = false;
    m_referenceFrames = 0;
    m_horizonBiasAnchored = false;
  }

  void RtxClouds::uploadPhaseLut(Rc<DxvkContext> ctx) {
    using namespace cloud_tables;
    static_assert(kPhaseLutSize == CLOUD_PHASE_LUT_SIZE && kRadiusCount == CLOUD_PHASE_LUT_RADII, "Cloud phase LUT size mismatch");

    std::vector<float> texels(size_t(kPhaseLutSize) * kRadiusCount * 4, 0.0f);
    for (uint32_t row = 0; row < kRadiusCount; ++row) {
      for (uint32_t i = 0; i < kPhaseLutSize; ++i) {
        float* texel = &texels[(size_t(row) * kPhaseLutSize + i) * 4];
        for (uint32_t channel = 0; channel < kChannelCount; ++channel) {
          texel[channel] = kPhaseLut[row][channel][i];
        }
      }
    }
    const VkImageSubresourceLayers subresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    ctx->updateImage(m_phaseLut.image, subresource, VkOffset3D { 0, 0, 0 }, VkExtent3D { kPhaseLutSize, kRadiusCount, 1 },
      texels.data(), kPhaseLutSize * 4 * sizeof(float), size_t(kPhaseLutSize) * kRadiusCount * 4 * sizeof(float));

    std::vector<float> cdf(size_t(kInverseCdfSize) * kRadiusCount * 4, 0.0f);
    for (uint32_t row = 0; row < kRadiusCount; ++row) {
      for (uint32_t i = 0; i < kInverseCdfSize; ++i) {
        float* texel = &cdf[(size_t(row) * kInverseCdfSize + i) * 4];
        for (uint32_t channel = 0; channel < kChannelCount; ++channel) {
          texel[channel] = kInverseCdf[row][channel][i];
        }
      }
    }
    ctx->updateImage(m_phaseCdf.image, subresource, VkOffset3D { 0, 0, 0 }, VkExtent3D { kInverseCdfSize, kRadiusCount, 1 },
      cdf.data(), kInverseCdfSize * 4 * sizeof(float), size_t(kInverseCdfSize) * kRadiusCount * 4 * sizeof(float));
  }

  void RtxClouds::advanceMotion() {
    const float dt = std::clamp(float(GlobalTime::get().deltaTimeMs()) * 0.001f, 0.0f, 0.25f);
    if (!(dt > 0.0f)) {
      return;
    }
    const float angle = windDirection() * (kCloudPi / 180.0f);
    const float speedKm = windSpeed() * 0.001f;
    m_windOffsetKm.x += std::cos(angle) * speedKm * dt;
    m_windOffsetKm.y += std::sin(angle) * speedKm * dt;
    m_evolutionRiseKm += evolutionRise() * 0.001f * dt;
    m_evolutionShearKm += evolutionShear() * 0.001f * dt;
  }

  CloudArgs RtxClouds::buildArgs(const AtmosphereArgs& atmosphere, const RtCamera& camera, const VkExtent3D& renderExtent, uint32_t debugView) {
    const TierSettings tier = getTierSettings(quality());
    const GenusSettings g = getGenusSettings();

    CloudArgs a = {};
    a.enabled = 1;
    a.frameIndex = m_frameIndex;
    a.debugView = debugView;
    a.flags = skyAerialPerspective() ? CLOUD_FLAG_SKY_AERIAL_PERSPECTIVE : 0u;
    a.flags |= sunShadows() ? CLOUD_FLAG_SUN_SHADOWS : 0u;
    a.flags |= airShadows() ? CLOUD_FLAG_FOG_SHADOWS : 0u;
    a.flags |= reflectionDome() ? CLOUD_FLAG_DOME : 0u;
    a.flags |= mirrorReflectionMarch() ? CLOUD_FLAG_PSR_REFLECTIONS : 0u;
    a.flags |= glossyReflectionMarch() ? CLOUD_FLAG_GLOSSY_REFLECTIONS : 0u;
    a.flags |= hexTiling() ? CLOUD_FLAG_HEX_TILING : 0u;
    a.flags |= m_screenHistoryValid ? CLOUD_FLAG_HISTORY_VALID : 0u;
    a.flags |= m_shadowMapState.valid ? CLOUD_FLAG_SHADOW_MAP_VALID : 0u;
    a.baseAltitudeKm = std::max(g.baseAltitudeMeters, 0.0f) * 0.001f;
    a.thicknessKm = std::max(g.thicknessMeters * 0.001f, 0.05f);
    a.tileKm = std::max(tileKm(), 1.0f);
    a.worldUnitsPerKm = std::max(atmosphere.worldUnitsPerKilometer, 1e-6f);

    // Km in the atmosphere's Y-up frame: Z-up worlds swap their y and z.
    const Vector3 position = camera.getPosition();
    const bool zUp = atmosphere.isZUp != 0;
    a.cameraPositionKm = Vector3(position.x / a.worldUnitsPerKm, atmosphere.aerialPerspectiveViewAltitude, (zUp ? position.y : position.z) / a.worldUnitsPerKm);
    a.cameraWorldHeightKm = (zUp ? position.z : position.y) / a.worldUnitsPerKm;

    // The windows follow the camera through the field's frame, which the wind carries: as the clouds drift past,
    // the window slides over them and brings in strips.
    const float cameraFieldX = a.cameraPositionKm.x - m_windOffsetKm.x;
    const float cameraFieldZ = a.cameraPositionKm.z - m_windOffsetKm.y;
    a.gridExtentKm = kGridExtentKm;
    const float gridTexelKm = kGridExtentKm / float(CLOUD_GRID_SIZE_XZ);
    const float shadowTexelKm = kShadowMapExtentKm / float(CLOUD_SHADOW_MAP_SIZE);
    a.gridOriginKm = Vector4(
      std::round(cameraFieldX / gridTexelKm) * gridTexelKm, std::round(cameraFieldZ / gridTexelKm) * gridTexelKm,
      std::round(cameraFieldX / shadowTexelKm) * shadowTexelKm, std::round(cameraFieldZ / shadowTexelKm) * shadowTexelKm);
    a.shadowMapExtentKm = kShadowMapExtentKm;
    const float farGridTexelKm = kFarGridExtentKm / float(CLOUD_FAR_GRID_SIZE_XZ);
    a.farGridExtentKm = kFarGridExtentKm;
    a.farGridOriginKm = Vector2(
      std::round(cameraFieldX / farGridTexelKm) * farGridTexelKm, std::round(cameraFieldZ / farGridTexelKm) * farGridTexelKm);
    a.apStartDistanceKm = std::max(atmosphere.aerialPerspectiveStartDistance, 0.0f) / a.worldUnitsPerKm;
    a.pixelAngle = camera.getFov() / float(std::max(renderExtent.height, 1u));

    const float windAngle = windDirection() * (kCloudPi / 180.0f);
    a.windOffsetKm = m_windOffsetKm;
    a.windStepKm = m_windStepKm;
    a.riseStepKm = m_riseStepKm;
    a.shearStepKm = m_shearStepKm;
    a.windDirection = Vector2(std::cos(windAngle), std::sin(windAngle));
    a.evolutionRiseKm = m_evolutionRiseKm;
    a.evolutionShearKm = m_evolutionShearKm;
    a.detailBaseShearKm = detailBaseShearKm();
    a.cellSizeKm = std::max(g.cellSizeKm, 0.25f);

    a.coverage = std::clamp(g.coverage, 0.0f, 1.0f);
    a.coverageSpread = coverageSpread();
    a.coverageSpreadFrequency = 1.0f / std::max(coverageSpreadScaleKm(), 0.1f);
    a.cloudType = std::clamp(g.cloudType, 0.0f, 1.0f);
    a.typeSpread = typeSpread();
    a.typeSpreadFrequency = 1.0f / std::max(typeSpreadScaleKm(), 0.1f);
    a.columnFeather = std::max(g.columnFeather, 0.02f);
    a.columnTopShape = std::max(g.columnTopShape, 0.05f);
    a.columnTopVariation = std::clamp(g.columnTopVariation, 0.0f, 1.0f);
    a.columnBaseVariation = std::clamp(g.columnBaseVariation, 0.0f, 1.0f);
    a.columnTopFlatten = std::clamp(g.columnTopFlatten, 0.05f, 1.0f);
    a.nvdfNominalCoverage = nominalCoverage() > 0.0f ? std::clamp(nominalCoverage(), 0.0f, 1.0f) : a.coverage;

    a.nvdfCoverageOffsetKm = std::max(coverageOffsetKm(), 0.0f);
    a.nvdfProfileDepthKm = std::max(profileDepthKm(), 0.05f);
    a.nvdfBodyErosion = std::clamp(bodyErosion(), 0.0f, 1.5f);
    a.nvdfStepScale = std::clamp(stepScale(), 0.0f, 0.95f);
    a.detailScale = std::max(detailScale(), 0.1f);
    a.wobbleStrength = std::max(wobbleStrength(), 0.0f);
    a.erosionStrength = std::max(erosionStrength(), 0.0f);
    a.sharpenStrength = std::clamp(sharpenStrength(), 0.0f, 1.0f);
    a.hfDetailStrength = std::clamp(flyThroughDetail(), 0.0f, 3.0f);
    a.interiorTexture = std::clamp(interiorTexture(), 0.0f, 1.0f);
    a.edgeErosion = std::clamp(edgeErosion(), 0.0f, 3.0f);
    a.fineDetailStrength = std::clamp(fineDetailStrength(), 0.0f, 2.0f);
    a.edgeDetailStrength = std::clamp(edgeDetail(), 0.0f, 2.0f);
    a.shapeVarietyWavelengthKm = std::max(shapeVarietyWavelengthKm(), 0.05f);
    // A level set displaced by more than about wavelength / pi peak to peak folds into detached sheets.
    a.shapeVarietyKm = std::min(std::clamp(shapeVarietyKm(), 0.0f, 1.5f), 0.65f * a.shapeVarietyWavelengthKm);
    a.curlStrengthKm = std::max(curlStrengthMeters(), 0.0f) * 0.001f;
    a.nearDetailStrength = std::clamp(nearDetailStrength(), 0.0f, 2.0f);
    a.nearDetailRangeKm = std::max(nearDetailRangeKm(), 0.01f);
    a.detailLodBias = detailLodBias();

    // At full strength the horizon bias's shift reaches past the bodies' cores (half their smaller extent) and
    // what the surface's displacement pushes out.
    a.horizonBias = std::clamp(horizonBias(), -1.0f, 1.0f);
    a.horizonBiasStartKm = std::max(horizonBiasStartKm(), 0.0f);
    a.horizonBiasEndKm = std::max(horizonBiasEndKm(), a.horizonBiasStartKm + 0.1f);
    a.horizonBiasShiftKm = a.horizonBias != 0.0f
      ? 0.5f * std::min(a.thicknessKm, a.cellSizeKm) + 0.5f * (0.6f * a.wobbleStrength + a.shapeVarietyKm)
      : 0.0f;

    a.viewSamplesMax = tier.viewSamplesMax;
    a.viewStepKm = tier.viewStepKm;
    a.adaptiveStepKm = tier.adaptiveStepKm;
    a.maxMarchKm = std::max(maxMarchKm(), 1.0f);
    a.opticalDepthPerStep = tier.opticalDepthPerStep;
    a.exitTransmittance = tier.exitTransmittance;
    a.shadowTaps = tier.shadowTaps;
    a.shadowTapRangeKm = std::max(shadowTapRangeMeters(), 0.0f) * 0.001f;
    a.historyBlend = tier.historyBlend;
    a.domeBlend = tier.domeBlend;

    // Microphysics: the droplets' effective radius from the water content and number, their extinction
    // 1 km above the base (Stephens 1978), and the delta-M truncation of their phase function.
    const float radiusUm = effectiveRadiusUm(g.liquidWaterContent, g.dropletConcentration);
    const DropletOptics optics = lookupDropletOptics(radiusUm);
    a.extinctionKm = extinctionPerKm(g.liquidWaterContent, radiusUm, optics.extinctionEfficiency) * std::max(densityScale(), 0.0f);
    a.adiabaticExponent = std::clamp(adiabaticExponent(), 0.0f, 2.0f);
    a.adiabaticFloor = std::clamp(adiabaticFloor(), 0.0f, 1.0f);
    a.adiabaticMax = std::max(adiabaticMax(), a.adiabaticFloor);
    a.forwardPeakFraction = std::clamp(optics.forwardPeakFraction, 0.0f, 0.9f);
    a.asymmetry = optics.asymmetry;
    a.truncatedAsymmetry = optics.truncatedAsymmetry;
    a.phaseLutRow = optics.row;
    a.phaseLegendre = Vector2(optics.truncatedAsymmetry, optics.truncatedAsymmetry * optics.truncatedAsymmetry);
    // The scene's shadows count the forward diffraction peak as transmitted (delta-M): that light still
    // arrives from within a degree of the sun.
    a.shadowExtinctionKm = a.extinctionKm * (1.0f - a.forwardPeakFraction);
    a.shadowStrength = std::clamp(shadowStrength(), 0.0f, 1.0f);
    a.msFloorAnisotropy = std::clamp(diffusionAnisotropy(), 0.0f, 2.0f);
    a.msOctaves = multipleScatteringOctaves() > 0 ? std::min(multipleScatteringOctaves(), 4u) : tier.msOctaves;
    a.msExtinctionFalloff = std::clamp(msExtinctionFalloff(), 0.0f, 1.0f);
    a.msEnergyFalloff = std::clamp(msEnergyFalloff(), 0.0f, 1.0f);
    a.msPhaseFalloff = std::clamp(msPhaseFalloff(), 0.0f, 1.0f);
    a.msFloorWeight = std::max(diffusionFloor(), 0.0f);
    a.diffuseTransmissionK = std::max(diffuseTransmissionK(), 0.0f);
    a.ambientStrength = std::max(ambientStrength(), 0.0f);
    a.groundBounceStrength = std::max(groundBounceStrength(), 0.0f);
    a.groundDiffuseShare = std::clamp(groundDiffuseShare(), 0.0f, 1.0f);
    a.albedoTint = albedoTint();

    a.referenceMaxBounces = std::max(referenceMaxBounces(), 1u);
    a.referenceExactBounces = referenceExactBounces();
    a.referenceBouncesPerFrame = std::clamp(referenceBouncesPerFrame(), 1u, 1024u);
    return a;
  }

  CloudArgs RtxClouds::update(
    RtxContext& rtxCtx, const AtmosphereArgs& atmosphere, const RtCamera& camera, const VkExtent3D& renderExtent, uint32_t debugView)
  {
    Rc<DxvkContext> ctx = &rtxCtx;
    m_screenRanThisFrame = false;

    if (!isActive()) {
      // Nothing of the clouds stays allocated while they are off; the next frame they are on starts every bake over.
      if (m_initialized) {
        releaseResources();
      }
      m_args = CloudArgs {};
      // Passes outside this module read the constants too (the aerial perspective), so they must say off.
      if (!m_offConstantsWritten) {
        CloudConstants constants = {};
        constants.atmosphere = atmosphere;
        ctx->updateBuffer(m_constantsBuffer, 0, sizeof(CloudConstants), &constants);
        ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);
        m_offConstantsWritten = true;
      }
      return m_args;
    }
    m_offConstantsWritten = false;
    initialize(ctx);

    ScopedGpuProfileZone(ctx, "Clouds");

    ++m_frameIndex;
    // The reference converges over many frames of the same clouds.
    const Vector2 windBefore = m_windOffsetKm;
    const float riseBefore = m_evolutionRiseKm;
    const float shearBefore = m_evolutionShearKm;
    if (motion() && !reference()) {
      advanceMotion();
    }
    m_windStepKm = m_windOffsetKm - windBefore;
    m_riseStepKm = m_evolutionRiseKm - riseBefore;
    m_shearStepKm = m_evolutionShearKm - shearBefore;

    if (quality() != m_lastQuality || camera.isCameraCut()) {
      m_screenHistoryValid = false;
      m_domeHistoryValid = false;
      m_skyApValid = false;
      m_lastQuality = quality();
    }

    const TierSettings tier = getTierSettings(quality());
    ensureDome(ctx, tier.domeWidth);
    // Before the arguments, whose history flag a resize clears.
    ensureScreenResources(ctx, renderExtent);

    CloudArgs args = buildArgs(atmosphere, camera, renderExtent, debugView);

    // Where the eye sits against the layer decides most of what the clouds look like, and it follows the
    // atmosphere's altitude datum, so it is worth a line whenever it moves appreciably.
    if (std::abs(args.cameraPositionKm.y - m_loggedEyeAltitudeKm) > 0.1f) {
      m_loggedEyeAltitudeKm = args.cameraPositionKm.y;
      Logger::info(str::format("[Clouds] Eye altitude ", args.cameraPositionKm.y, " km (camera world height ", args.cameraWorldHeightKm,
        " km), layer ", args.baseAltitudeKm, " to ", args.baseAltitudeKm + args.thicknessKm, " km"));
    }
    m_eyeAltitudeKm = args.cameraPositionKm.y;

    // The body field: the first bake runs to completion; later ones spread over frames while the published
    // field keeps rendering.
    const NvdfKey key = makeNvdfKey(args);
    if (!m_nvdfValid) {
      startNvdfBake(ctx, args);
      stepNvdfBake(ctx, kNvdfJumpPassCount);
    } else if (m_nvdfBakeActive) {
      stepNvdfBake(ctx, kNvdfJumpPassesPerFrame);
    } else if (!(key == m_publishedNvdfKey)) {
      startNvdfBake(ctx, args);
    }
    // The live coverage offsets the level set against what the published field was baked at.
    args.nvdfNominalCoverage = m_publishedNominalCoverage;

    if (reference()) {
      args.flags |= CLOUD_FLAG_REFERENCE;
      if (referenceInputsChanged(camera, args, atmosphere)) {
        m_referenceFrames = 0;
      } else if (m_nvdfBakeActive) {
        if (m_referenceFrames > 64) {
          Logger::info(str::format("[Clouds] Reference restarted after ", m_referenceFrames, " frames: the body field is re-baking"));
        }
        m_referenceFrames = 0;
      }
    } else {
      m_referenceFrames = 0;
      releaseReferenceResources();
    }

    CloudConstants constants;
    constants.atmosphere = atmosphere;
    constants.cloud = args;
    ctx->updateBuffer(m_constantsBuffer, 0, sizeof(CloudConstants), &constants);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);

    // Each bake runs only where its inputs changed. The grids and the shadow map are anchored to the field the
    // wind carries, so the wind alone changes none of them and they only bake the strips their windows slide over;
    // rise and shear evolve the field, and with either on they rebake on their interleave. The sky's aerial
    // perspective and the dome see the clouds from the camera, so the wind moves what they hold.
    if (args.horizonBiasShiftKm > 0.0f) {
      // The horizon bias follows the camera while the bakes hold the field the wind carries, so they restart once the
      // camera has moved through the field far enough for the bias to move the clouds' surfaces by 20 m.
      const Vector2 cameraFieldKm(args.cameraPositionKm.x - args.windOffsetKm.x, args.cameraPositionKm.z - args.windOffsetKm.y);
      const float gradient = std::abs(args.horizonBias) * args.horizonBiasShiftKm * 1.5f / (args.horizonBiasEndKm - args.horizonBiasStartKm);
      const float toleranceKm = std::clamp(0.02f / gradient, 0.01f, 1.0f);
      const Vector2 moved = cameraFieldKm - m_horizonBiasAnchorKm;
      if (!m_horizonBiasAnchored || moved.x * moved.x + moved.y * moved.y > toleranceKm * toleranceKm) {
        m_horizonBiasAnchorKm = cameraFieldKm;
        m_horizonBiasAnchored = true;
      }
    } else {
      m_horizonBiasAnchored = false;
    }
    const auto keyOf = [&](const CloudArgs& keyArgs) {
      const uint64_t key = computeInputsKey(keyArgs, atmosphere);
      return m_horizonBiasAnchored ? hashBytes(&m_horizonBiasAnchorKm, sizeof(m_horizonBiasAnchorKm), key) : key;
    };
    const uint64_t inputsKey = keyOf(args);
    const bool inputsChanged = inputsKey != m_lastInputsKey;
    m_lastInputsKey = inputsKey;
    // The same but for the field's evolution, which changes the bakes only slowly.
    CloudArgs unevolved = args;
    unevolved.evolutionRiseKm = 0.0f;
    unevolved.evolutionShearKm = 0.0f;
    const uint64_t restartKey = keyOf(unevolved);
    const bool inputsRestarted = restartKey != m_lastRestartKey;
    m_lastRestartKey = restartKey;
    const bool windMoved = m_windStepKm.x != 0.0f || m_windStepKm.y != 0.0f;
    const vec3 cameraMove = args.cameraPositionKm - m_skyCameraPositionKm;
    const float cameraMoveKm = std::sqrt(cameraMove.x * cameraMove.x + cameraMove.y * cameraMove.y + cameraMove.z * cameraMove.z);
    const bool cameraMoved = cameraMoveKm > kSkyCameraToleranceKm ||
                             std::abs(args.cameraWorldHeightKm - m_skyCameraWorldHeightKm) > kSkyCameraToleranceKm;
    if (cameraMoved) {
      m_skyCameraPositionKm = args.cameraPositionKm;
      m_skyCameraWorldHeightKm = args.cameraWorldHeightKm;
    }
    if (m_gridsNeedFullBake) {
      m_nearGridState.valid = false;
      m_farGridState.valid = false;
      m_shadowMapState.valid = false;
      m_gridsNeedFullBake = false;
    }

    // Window origins in texels.
    const auto texels = [](float originKm, float texelKm) { return int64_t(std::llround(double(originKm) / double(texelKm))); };
    const float gridTexelKm = kGridExtentKm / float(CLOUD_GRID_SIZE_XZ);
    const float farGridTexelKm = kFarGridExtentKm / float(CLOUD_FAR_GRID_SIZE_XZ);
    const float shadowTexelKm = kShadowMapExtentKm / float(CLOUD_SHADOW_MAP_SIZE);

    bool nearWritten = false;
    bool farWritten = false;
    {
      ScopedGpuProfileZone(ctx, "Clouds Optical Depth Grids");
      {
        ScopedGpuProfileZone(ctx, "Clouds Near Grids");
        nearWritten = updateWorldBake(m_nearGridState, inputsKey, texels(args.gridOriginKm.x, gridTexelKm), texels(args.gridOriginKm.y, gridTexelKm),
          CLOUD_GRID_SIZE_XZ, std::max(tier.gridInterleave, 1u), [&](const BakeRegion& region) { bakeLightingGrids(ctx, 0, region); });
      }
      const int64_t farOriginX = texels(args.farGridOriginKm.x, farGridTexelKm);
      const int64_t farOriginZ = texels(args.farGridOriginKm.y, farGridTexelKm);
      // A full bake, or a jump that brings in over a kilometre of the window at once.
      const bool farRebaked = !m_farGridState.valid || std::abs(farOriginX - m_farGridState.interleavedOrigin) > 4 ||
                              std::abs(farOriginZ - m_farGridState.otherOrigin) > 4;
      {
        ScopedGpuProfileZone(ctx, "Clouds Far Grids");
        farWritten = updateWorldBake(m_farGridState, inputsKey, farOriginX, farOriginZ,
          CLOUD_FAR_GRID_SIZE_XZ, kFarGridInterleave, [&](const BakeRegion& region) { bakeLightingGrids(ctx, 1, region); });
      }
      if (inputsRestarted || farRebaked) {
        m_farDiffusionCatchUpSweeps = kDiffusionSettleSweeps;
      }
      barrier(ctx);
      if (nearWritten) {
        ScopedGpuProfileZone(ctx, "Clouds Sun Grid Mips");
        bakeSunGridMips(ctx);
      }
      // The diffusion floor's solve sweeps each frame its cascade's grid changes and for kDiffusionSettleSweeps
      // after, enough to converge from any start; then it holds. Between restarts the far cascade's grid changes only
      // as the field slowly evolves, which a sweep a frame keeps up with; after one it catches up at the tier's pace.
      const bool gridWritten[2] = { nearWritten, farWritten };
      if (nearWritten || farWritten || m_diffusionSettleSweeps[0] > 0 || m_diffusionSettleSweeps[1] > 0) {
        ScopedGpuProfileZone(ctx, "Clouds Diffusion");
        for (uint32_t cascade = 0; cascade < 2; ++cascade) {
          if (gridWritten[cascade]) {
            m_diffusionSettleSweeps[cascade] = kDiffusionSettleSweeps;
          }
          const bool tracking = cascade == 1 && gridWritten[cascade] && m_farDiffusionCatchUpSweeps == 0;
          const uint32_t sweeps = std::min(tracking ? 1u : std::max(tier.diffusionSweeps, 1u), m_diffusionSettleSweeps[cascade]);
          if (sweeps > 0) {
            bakeDiffusion(ctx, cascade, gridWritten[cascade], sweeps);
            m_diffusionSettleSweeps[cascade] -= sweeps;
            if (cascade == 1) {
              m_farDiffusionCatchUpSweeps -= std::min(sweeps, m_farDiffusionCatchUpSweeps);
            }
          }
        }
      }
    }
    bool shadowMapWritten = false;
    {
      ScopedGpuProfileZone(ctx, "Clouds Shadow Map");
      // Rows (world z) are the interleaved axis.
      shadowMapWritten = updateWorldBake(m_shadowMapState, inputsKey, texels(args.gridOriginKm.w, shadowTexelKm), texels(args.gridOriginKm.z, shadowTexelKm),
        CLOUD_SHADOW_MAP_SIZE, kShadowMapInterleave, [&](const BakeRegion& region) { bakeShadowMap(ctx, region); });
    }
    barrier(ctx);

    // The sky light reads the near grid (the ground under the clouds' shadow); the aerial perspective LUT also the
    // shadow map (the air under them), from the camera.
    bool skyShWritten = false;
    bool skyApWritten = false;
    {
      ScopedGpuProfileZone(ctx, "Clouds Sky Light");
      if (inputsChanged || nearWritten) {
        const uint64_t skyLightKey = computeSkyLightKey(args, atmosphere);
        bakeSkySh(ctx, skyLightKey == m_skyLightKey);
        m_skyLightKey = skyLightKey;
        skyShWritten = true;
      }
      if (inputsChanged || cameraMoved || windMoved || nearWritten || shadowMapWritten) {
        m_skyApPhasesBaked = 0;
      }
      if (!m_skyApValid) {
        bakeSkyAp(ctx, 1, 0);
        m_skyApValid = true;
        m_skyApPhasesBaked = kSkyApInterleave;
        skyApWritten = true;
      } else if (m_skyApPhasesBaked < kSkyApInterleave) {
        bakeSkyAp(ctx, kSkyApInterleave, m_frameIndex % kSkyApInterleave);
        ++m_skyApPhasesBaked;
        skyApWritten = true;
      }
    }
    barrier(ctx);

    // The dome's history converges once its inputs hold still; after that, refreshing it changes nothing.
    const bool domeInputsChanged = inputsChanged || cameraMoved || windMoved || nearWritten || farWritten || shadowMapWritten || skyShWritten || skyApWritten;
    m_domeSettledFrames = domeInputsChanged ? 0u : m_domeSettledFrames + 1u;
    if ((args.flags & CLOUD_FLAG_DOME) && (!m_domeHistoryValid || m_domeSettledFrames < kDomeSettleRefreshes * tier.domeInterleave)) {
      ScopedGpuProfileZone(ctx, "Clouds Dome");
      bakeDome(ctx, tier.domeInterleave);
    }

    // Bakes before the path tracer and the screen pass read them.
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_SHADER_READ_BIT);

    m_args = args;
    return args;
  }

  void RtxClouds::bindCloudInputs(Rc<DxvkContext> ctx, const Rc<DxvkBuffer>& constants) {
    RtxContext* rtx = static_cast<RtxContext*>(ctx.ptr());
    ctx->bindResourceBuffer(CLOUD_BINDING_CONSTANTS, DxvkBufferSlice(constants, 0, constants->info().size));
    ctx->bindResourceSampler(CLOUD_BINDING_VOLUME_SAMPLER, getVolumeSampler(ctx));
    ctx->bindResourceSampler(CLOUD_BINDING_LUT_SAMPLER, getLutSampler(ctx));
    ctx->bindResourceView(CLOUD_BINDING_NVDF, m_nvdfSdf[m_nvdfFront].view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_DETAIL_NOISE, m_detailNoise.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_SUN_GRID, m_sunGrid.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_AMBIENT_GRID, m_ambientGrid.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_DIFFUSION_GRID, m_diffusionFluence[0].view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_SUN_GRID_FAR, m_sunGridFar.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_AMBIENT_GRID_FAR, m_ambientGridFar.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_DIFFUSION_GRID_FAR, m_diffusionFluence[1].view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_TRANSMITTANCE_LUT, rtx->getAtmosphereTransmittanceLutView(), nullptr);
    ctx->bindResourceView(CLOUD_BINDING_PHASE_LUT, m_phaseLut.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_SKY_AP_INSCATTER, m_skyApInScatter.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_SKY_AP_TRANSMITTANCE, m_skyApTransmittance.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_SKY_SH, m_skySh.view, nullptr);
  }

  void RtxClouds::bakePlacement(Rc<DxvkContext> ctx) {
    ScopedGpuProfileZone(ctx, "Clouds Placement Map");
    ctx->bindResourceBuffer(CLOUD_BINDING_CONSTANTS, DxvkBufferSlice(m_nvdfConstantsBuffer, 0, m_nvdfConstantsBuffer->info().size));
    ctx->bindResourceView(CLOUD_BINDING_BAKE_OUTPUT, m_placementMap.view, nullptr);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_placementMap.image);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudPlacementBakeShader::getShader());
    const VkExtent3D groups = util::computeBlockCount(VkExtent3D { CLOUD_PLACEMENT_MAP_SIZE, CLOUD_PLACEMENT_MAP_SIZE, 1 }, VkExtent3D { 8, 8, 1 });
    ctx->dispatch(groups.width, groups.height, groups.depth);
  }

  void RtxClouds::bakeDetailNoise(Rc<DxvkContext> ctx) {
    ScopedGpuProfileZone(ctx, "Clouds Detail Noise");
    ctx->bindResourceView(CLOUD_BINDING_BAKE_OUTPUT, m_detailNoiseMipViews[0], nullptr);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_detailNoise.image);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudDetailNoiseBakeShader::getShader());
    const VkExtent3D size = { CLOUD_DETAIL_NOISE_SIZE, CLOUD_DETAIL_NOISE_SIZE, CLOUD_DETAIL_NOISE_SIZE };
    const VkExtent3D groups = util::computeBlockCount(size, VkExtent3D { 8, 8, 8 });
    ctx->dispatch(groups.width, groups.height, groups.depth);

    // The footprint filtered levels the view march reads at distance.
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudVolumeMipUnormShader::getShader());
    ctx->bindResourceSampler(CLOUD_BINDING_VOLUME_SAMPLER, getVolumeSampler(ctx));
    for (uint32_t level = 1; level < CLOUD_DETAIL_NOISE_MIP_COUNT; ++level) {
      barrier(ctx);
      ctx->bindResourceView(CLOUD_BINDING_BAKE_INPUT, m_detailNoiseMipViews[level - 1], nullptr);
      ctx->bindResourceView(CLOUD_BINDING_BAKE_OUTPUT, m_detailNoiseMipViews[level], nullptr);
      const uint32_t dim = std::max(uint32_t(CLOUD_DETAIL_NOISE_SIZE) >> level, 1u);
      const VkExtent3D mipGroups = util::computeBlockCount(VkExtent3D { dim, dim, dim }, VkExtent3D { 4, 4, 4 });
      ctx->dispatch(mipGroups.width, mipGroups.height, mipGroups.depth);
    }
  }

  void RtxClouds::startNvdfBake(Rc<DxvkContext> ctx, const CloudArgs& args) {
    ScopedGpuProfileZone(ctx, "Clouds Field Occupancy");
    // The bake keeps the inputs it started from; a change during it starts another once it is published.
    CloudConstants constants = {};
    constants.cloud = args;
    ctx->updateBuffer(m_nvdfConstantsBuffer, 0, sizeof(CloudConstants), &constants);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_nvdfConstantsBuffer);
    m_pendingNvdfKey = makeNvdfKey(args);

    // The jump flood's volumes live only while a bake runs.
    const VkExtent3D nvdfExtent = { CLOUD_NVDF_SIZE_XZ, CLOUD_NVDF_SIZE_Y, CLOUD_NVDF_SIZE_XZ };
    const auto create3D = [&](const char* name, VkFormat format) {
      return Resources::createImageResource(ctx, name, nvdfExtent, format, 1, VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D);
    };
    m_nvdfOccupancy = create3D("Cloud NVDF Occupancy", VK_FORMAT_R8_UNORM);
    m_nvdfSeeds[0] = create3D("Cloud NVDF Seeds 0", VK_FORMAT_R32_UINT);
    m_nvdfSeeds[1] = create3D("Cloud NVDF Seeds 1", VK_FORMAT_R32_UINT);

    // The placement map follows the cell size and tile, so it bakes with the field.
    bakePlacement(ctx);
    barrier(ctx);

    ctx->bindResourceBuffer(CLOUD_BINDING_CONSTANTS, DxvkBufferSlice(m_nvdfConstantsBuffer, 0, m_nvdfConstantsBuffer->info().size));
    ctx->bindResourceView(CLOUD_BINDING_PLACEMENT_MAP, m_placementMap.view, nullptr);
    ctx->bindResourceSampler(CLOUD_BINDING_VOLUME_SAMPLER, getVolumeSampler(ctx));
    ctx->bindResourceView(CLOUD_BINDING_BAKE_OUTPUT, m_nvdfOccupancy.view, nullptr);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_placementMap.image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_nvdfOccupancy.image);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudNvdfOccupancyShader::getShader());
    const VkExtent3D groups = util::computeBlockCount(VkExtent3D { CLOUD_NVDF_SIZE_XZ, CLOUD_NVDF_SIZE_Y, CLOUD_NVDF_SIZE_XZ }, VkExtent3D { 8, 4, 8 });
    ctx->dispatch(groups.width, groups.height, groups.depth);
    barrier(ctx);

    // Seeds go to buffer 0; jump pass i reads i % 2 and writes (i + 1) % 2.
    dispatchNvdfJfa(ctx, 0, 0, 1, 0);
    barrier(ctx);

    m_nvdfBakeActive = true;
    m_nvdfJumpIndex = 0;
  }

  void RtxClouds::dispatchNvdfJfa(Rc<DxvkContext> ctx, uint32_t mode, uint32_t jumpSize, uint32_t source, uint32_t destination) {
    ScopedGpuProfileZone(ctx, mode == 0 ? "Clouds Field Seeds" : "Clouds Field Jump Flood");
    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
    CloudNvdfJfaArgs pushArgs = {};
    pushArgs.mode = mode;
    pushArgs.jumpSizeVoxels = jumpSize;
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudNvdfJfaShader::getShader());
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
    ctx->bindResourceBuffer(CLOUD_BINDING_CONSTANTS, DxvkBufferSlice(m_nvdfConstantsBuffer, 0, m_nvdfConstantsBuffer->info().size));
    ctx->bindResourceView(CLOUD_BINDING_BAKE_INPUT, m_nvdfOccupancy.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_BAKE_INPUT2, m_nvdfSeeds[source].view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_BAKE_OUTPUT, m_nvdfSeeds[destination].view, nullptr);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_nvdfOccupancy.image);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_nvdfSeeds[source].image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_nvdfSeeds[destination].image);
    const VkExtent3D groups = util::computeBlockCount(VkExtent3D { CLOUD_NVDF_SIZE_XZ, CLOUD_NVDF_SIZE_Y, CLOUD_NVDF_SIZE_XZ }, VkExtent3D { 8, 4, 8 });
    ctx->dispatch(groups.width, groups.height, groups.depth);
  }

  void RtxClouds::stepNvdfBake(Rc<DxvkContext> ctx, uint32_t passBudget) {
    if (!m_nvdfBakeActive) {
      return;
    }

    for (uint32_t n = 0; n < passBudget && m_nvdfJumpIndex < kNvdfJumpPassCount; ++n) {
      dispatchNvdfJfa(ctx, 1, kNvdfJumpSchedule[m_nvdfJumpIndex], m_nvdfJumpIndex % 2, (m_nvdfJumpIndex + 1) % 2);
      barrier(ctx);
      ++m_nvdfJumpIndex;
    }

    if (m_nvdfJumpIndex < kNvdfJumpPassCount) {
      return;
    }

    // Resolve into the back buffer, then publish it.
    {
      ScopedGpuProfileZone(ctx, "Clouds Field Resolve");
      const uint32_t back = 1 - m_nvdfFront;
      ctx->bindResourceBuffer(CLOUD_BINDING_CONSTANTS, DxvkBufferSlice(m_nvdfConstantsBuffer, 0, m_nvdfConstantsBuffer->info().size));
      ctx->bindResourceView(CLOUD_BINDING_BAKE_INPUT, m_nvdfOccupancy.view, nullptr);
      ctx->bindResourceView(CLOUD_BINDING_BAKE_INPUT2, m_nvdfSeeds[kNvdfJumpPassCount % 2].view, nullptr);
      ctx->bindResourceView(CLOUD_BINDING_BAKE_OUTPUT, m_nvdfSdf[back].view, nullptr);
      ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_nvdfOccupancy.image);
      ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_nvdfSeeds[kNvdfJumpPassCount % 2].image);
      ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_nvdfSdf[back].image);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudNvdfResolveShader::getShader());
      const VkExtent3D groups = util::computeBlockCount(VkExtent3D { CLOUD_NVDF_SIZE_XZ, CLOUD_NVDF_SIZE_Y, CLOUD_NVDF_SIZE_XZ }, VkExtent3D { 8, 4, 8 });
      ctx->dispatch(groups.width, groups.height, groups.depth);
      barrier(ctx);
      m_nvdfFront = back;
    }

    m_publishedNvdfKey = m_pendingNvdfKey;
    m_publishedNominalCoverage = m_pendingNvdfKey.nominalCoverage;
    m_nvdfValid = true;
    m_nvdfBakeActive = false;
    m_nvdfJumpIndex = 0;
    m_nvdfOccupancy.reset();
    m_nvdfSeeds[0].reset();
    m_nvdfSeeds[1].reset();
    // The grids describe the old field until baked again in full.
    m_gridsNeedFullBake = true;
  }

  void RtxClouds::bakeLightingGrids(Rc<DxvkContext> ctx, uint32_t cascade, const BakeRegion& region) {
    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
    CloudPassArgs pushArgs = {};
    pushArgs.period = region.period;
    pushArgs.phase = region.phase;
    pushArgs.level = cascade;
    pushArgs.offset = region.offset;
    const Resources::Resource& sunGrid = cascade == 0 ? m_sunGrid : m_sunGridFar;
    const Resources::Resource& ambientGrid = cascade == 0 ? m_ambientGrid : m_ambientGridFar;
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudLightingGridShader::getShader());
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
    ctx->bindResourceBuffer(CLOUD_BINDING_CONSTANTS, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
    ctx->bindResourceSampler(CLOUD_BINDING_VOLUME_SAMPLER, getVolumeSampler(ctx));
    ctx->bindResourceView(CLOUD_BINDING_NVDF, m_nvdfSdf[m_nvdfFront].view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_DETAIL_NOISE, m_detailNoise.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_BAKE_OUTPUT, cascade == 0 ? m_sunGridMipViews[0] : m_sunGridFar.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_BAKE_OUTPUT2, ambientGrid.view, nullptr);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_nvdfSdf[m_nvdfFront].image);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_detailNoise.image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(sunGrid.image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(ambientGrid.image);
    const VkExtent3D groups = util::computeBlockCount(VkExtent3D { region.interleavedCount, CLOUD_GRID_SIZE_Y, region.otherCount }, VkExtent3D { 8, 8, 4 });
    ctx->dispatch(groups.width, groups.height, groups.depth);
  }

  void RtxClouds::bakeShadowMap(Rc<DxvkContext> ctx, const BakeRegion& region) {
    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
    CloudPassArgs pushArgs = {};
    pushArgs.period = region.period;
    pushArgs.phase = region.phase;
    pushArgs.offset = region.offset;
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudShadowMapShader::getShader());
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
    ctx->bindResourceBuffer(CLOUD_BINDING_CONSTANTS, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
    ctx->bindResourceSampler(CLOUD_BINDING_VOLUME_SAMPLER, getVolumeSampler(ctx));
    ctx->bindResourceView(CLOUD_BINDING_NVDF, m_nvdfSdf[m_nvdfFront].view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_DETAIL_NOISE, m_detailNoise.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_BAKE_OUTPUT, m_shadowMap.view, nullptr);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_shadowMap.image);
    // Rows are the interleaved axis.
    const VkExtent3D groups = util::computeBlockCount(VkExtent3D { region.otherCount, region.interleavedCount, 1 }, VkExtent3D { 8, 8, 1 });
    ctx->dispatch(groups.width, groups.height, groups.depth);
  }

  bool RtxClouds::updateWorldBake(
    WorldBakeState& state, uint64_t inputsKey, int64_t interleavedOrigin, int64_t otherOrigin, uint32_t size, uint32_t period,
    const std::function<void(const BakeRegion&)>& bake)
  {
    const int64_t n = int64_t(size);
    const auto wrap = [n](int64_t texel) { return uint32_t(((texel % n) + n) % n); };
    const int64_t movedInterleaved = interleavedOrigin - state.interleavedOrigin;
    const int64_t movedOther = otherOrigin - state.otherOrigin;
    if (!state.valid || std::abs(movedInterleaved) >= n || std::abs(movedOther) >= n) {
      bake(BakeRegion { 1, 0, 0, size, size });
      state = WorldBakeState { inputsKey, interleavedOrigin, otherOrigin, period, true };
      return true;
    }

    // The window spans [origin - size / 2, origin + size / 2) texels, each stored at its world index modulo the
    // size: moving it brings in a strip of world texels at its leading edge, which hold the far edge's until baked.
    bool wrote = false;
    if (movedInterleaved != 0) {
      const int64_t start = movedInterleaved > 0 ? state.interleavedOrigin + n / 2 : interleavedOrigin - n / 2;
      bake(BakeRegion { 1, wrap(start), 0, uint32_t(std::abs(movedInterleaved)), size });
      wrote = true;
    }
    if (movedOther != 0) {
      const int64_t start = movedOther > 0 ? state.otherOrigin + n / 2 : otherOrigin - n / 2;
      bake(BakeRegion { 1, 0, wrap(start), size, uint32_t(std::abs(movedOther)) });
      wrote = true;
    }
    state.interleavedOrigin = interleavedOrigin;
    state.otherOrigin = otherOrigin;

    // New inputs restart the interleave, which then runs until every texel has been baked from them.
    if (inputsKey != state.inputsKey) {
      state.inputsKey = inputsKey;
      state.phasesBaked = 0;
    }
    if (state.phasesBaked < period) {
      bake(BakeRegion { period, m_frameIndex % period, 0, (size + period - 1) / period, size });
      ++state.phasesBaked;
      wrote = true;
    }
    return wrote;
  }

  void RtxClouds::bakeDiffusion(Rc<DxvkContext> ctx, uint32_t cascade, bool cellsChanged, uint32_t sweeps) {
    const Resources::Resource& cells = m_diffusionCells[cascade];
    const Resources::Resource& fluence = m_diffusionFluence[cascade];
    const uint32_t sizeXZ = cascade == 0 ? CLOUD_GRID_SIZE_XZ : CLOUD_FAR_GRID_SIZE_XZ;
    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
    CloudPassArgs pushArgs = {};
    pushArgs.level = cascade;

    if (cellsChanged) {
      const Resources::Resource& sunGrid = cascade == 0 ? m_sunGrid : m_sunGridFar;
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudDiffusionSetupShader::getShader());
      ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
      ctx->bindResourceBuffer(CLOUD_BINDING_CONSTANTS, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
      ctx->bindResourceSampler(CLOUD_BINDING_VOLUME_SAMPLER, getVolumeSampler(ctx));
      ctx->bindResourceView(CLOUD_BINDING_BAKE_INPUT, cascade == 0 ? m_sunGridMipViews[0] : m_sunGridFar.view, nullptr);
      ctx->bindResourceView(CLOUD_BINDING_BAKE_OUTPUT, cells.view, nullptr);
      ctx->getCommandList()->trackResource<DxvkAccess::Read>(sunGrid.image);
      ctx->getCommandList()->trackResource<DxvkAccess::Write>(cells.image);
      const VkExtent3D groups = util::computeBlockCount(VkExtent3D { sizeXZ, CLOUD_GRID_SIZE_Y, sizeXZ }, VkExtent3D { 8, 4, 8 });
      ctx->dispatch(groups.width, groups.height, groups.depth);
      barrier(ctx);
    }

    // Each half sweep updates one parity of cells from the other's latest values.
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudDiffusionShader::getShader());
    ctx->bindResourceBuffer(CLOUD_BINDING_CONSTANTS, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
    ctx->bindResourceView(CLOUD_BINDING_BAKE_INPUT, cells.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_BAKE_OUTPUT, fluence.view, nullptr);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(cells.image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(fluence.image);
    const VkExtent3D halfGroups = util::computeBlockCount(VkExtent3D { sizeXZ / 2, CLOUD_GRID_SIZE_Y, sizeXZ }, VkExtent3D { 4, 4, 8 });
    for (uint32_t i = 0; i < 2 * sweeps; ++i) {
      pushArgs.phase = i & 1u;
      ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
      ctx->dispatch(halfGroups.width, halfGroups.height, halfGroups.depth);
      barrier(ctx);
    }
  }

  void RtxClouds::bakeSunGridMips(Rc<DxvkContext> ctx) {
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudVolumeMipFloatShader::getShader());
    ctx->bindResourceSampler(CLOUD_BINDING_VOLUME_SAMPLER, getVolumeSampler(ctx));
    for (uint32_t level = 1; level < CLOUD_SUN_GRID_MIP_COUNT; ++level) {
      if (level > 1) {
        barrier(ctx);
      }
      ctx->bindResourceView(CLOUD_BINDING_BAKE_INPUT, m_sunGridMipViews[level - 1], nullptr);
      ctx->bindResourceView(CLOUD_BINDING_BAKE_OUTPUT, m_sunGridMipViews[level], nullptr);
      const VkExtent3D size = {
        std::max(uint32_t(CLOUD_GRID_SIZE_XZ) >> level, 1u), std::max(uint32_t(CLOUD_GRID_SIZE_Y) >> level, 1u), std::max(uint32_t(CLOUD_GRID_SIZE_XZ) >> level, 1u) };
      const VkExtent3D groups = util::computeBlockCount(size, VkExtent3D { 4, 4, 4 });
      ctx->dispatch(groups.width, groups.height, groups.depth);
    }
  }

  void RtxClouds::bakeSkySh(Rc<DxvkContext> ctx, bool groundOnly) {
    RtxContext* rtx = static_cast<RtxContext*>(ctx.ptr());
    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
    CloudPassArgs pushArgs = {};
    pushArgs.level = groundOnly ? 1u : 0u;
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudSkyShShader::getShader());
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
    ctx->bindResourceBuffer(CLOUD_BINDING_CONSTANTS, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
    ctx->bindResourceSampler(CLOUD_BINDING_VOLUME_SAMPLER, getVolumeSampler(ctx));
    ctx->bindResourceView(CLOUD_BINDING_SUN_GRID, m_sunGrid.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_TRANSMITTANCE_LUT, rtx->getAtmosphereTransmittanceLutView(), nullptr);
    ctx->bindResourceView(CLOUD_BINDING_MULTISCATTERING_LUT, rtx->getAtmosphereMultiscatteringLutView(), nullptr);
    ctx->bindResourceView(CLOUD_BINDING_AEROSOL_PHASE_LUT, rtx->getAtmosphereAerosolPhaseLutView(), nullptr);
    ctx->bindResourceView(CLOUD_BINDING_BAKE_OUTPUT, m_skySh.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_BAKE_OUTPUT2, m_skyShParts.view, nullptr);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_skySh.image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_skyShParts.image);
    ctx->dispatch(CLOUD_SKY_SH_ALTITUDES, 1, 1);
  }

  void RtxClouds::bakeSkyAp(Rc<DxvkContext> ctx, uint32_t period, uint32_t phase) {
    RtxContext* rtx = static_cast<RtxContext*>(ctx.ptr());
    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
    CloudPassArgs pushArgs = {};
    pushArgs.period = period;
    pushArgs.phase = phase;
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudSkyApShader::getShader());
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
    ctx->bindResourceBuffer(CLOUD_BINDING_CONSTANTS, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
    ctx->bindResourceSampler(CLOUD_BINDING_VOLUME_SAMPLER, getVolumeSampler(ctx));
    ctx->bindResourceView(CLOUD_BINDING_SUN_GRID, m_sunGrid.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_SHADOW_MAP, m_shadowMap.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_TRANSMITTANCE_LUT, rtx->getAtmosphereTransmittanceLutView(), nullptr);
    ctx->bindResourceView(CLOUD_BINDING_MULTISCATTERING_LUT, rtx->getAtmosphereMultiscatteringLutView(), nullptr);
    ctx->bindResourceView(CLOUD_BINDING_AEROSOL_PHASE_LUT, rtx->getAtmosphereAerosolPhaseLutView(), nullptr);
    ctx->bindResourceView(CLOUD_BINDING_BAKE_OUTPUT, m_skyApInScatter.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_BAKE_OUTPUT2, m_skyApTransmittance.view, nullptr);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_skyApInScatter.image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_skyApTransmittance.image);
    const uint32_t columns = (CLOUD_SKY_AP_LUT_WIDTH + period - 1) / period;
    const VkExtent3D groups = util::computeBlockCount(VkExtent3D { columns, CLOUD_SKY_AP_LUT_HEIGHT, 1 }, VkExtent3D { 8, 8, 1 });
    ctx->dispatch(groups.width, groups.height, groups.depth);
  }

  void RtxClouds::bakeDome(Rc<DxvkContext> ctx, uint32_t interleave) {
    const uint32_t history = m_domeIndex;
    const uint32_t output = m_domeIndex ^ 1u;

    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
    CloudPassArgs pushArgs = {};
    pushArgs.level = m_domeHistoryValid ? 1u : 0u;
    // The shader's refresh patterns go up to one texel of each 2 x 2 block.
    pushArgs.period = std::clamp(interleave, 1u, 4u);
    pushArgs.phase = m_frameIndex % pushArgs.period;
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudDomeShader::getShader());
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
    bindCloudInputs(ctx, m_constantsBuffer);
    ctx->bindResourceView(CLOUD_BINDING_BAKE_INPUT, m_dome[history].view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_BAKE_OUTPUT, m_dome[output].view, nullptr);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_dome[history].image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_dome[output].image);
    const VkExtent3D groups = util::computeBlockCount(VkExtent3D { m_domeWidth, std::max(m_domeWidth / 2, 1u), 1 }, VkExtent3D { 8, 8, 1 });
    ctx->dispatch(groups.width, groups.height, groups.depth);

    m_domeIndex = output;
    m_domeHistoryValid = true;
  }

  void RtxClouds::dispatchScreen(RtxContext& rtxCtx, const Resources::RaytracingOutput& rtOutput) {
    if (!m_initialized || !m_args.enabled) {
      return;
    }
    Rc<DxvkContext> ctx = &rtxCtx;

    ScopedGpuProfileZone(ctx, "Clouds Screen");

    const VkExtent3D extent = m_screenExtent;

    // The G-buffer and the path tracer wrote what the march is bounded by.
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

    const uint32_t history = m_layerIndex;
    const uint32_t output = m_layerIndex ^ 1u;
    Rc<DxvkBuffer> raytraceConstants = rtxCtx.getResourceManager().getConstantsBuffer();
    DebugView& debugView = rtxCtx.getCommonObjects()->metaDebugView();

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudScreenShader::getShader());
    bindCloudInputs(ctx, m_constantsBuffer);
    ctx->bindResourceBuffer(CLOUD_BINDING_CAMERA, DxvkBufferSlice(raytraceConstants, 0, raytraceConstants->info().size));
    ctx->bindResourceView(CLOUD_BINDING_BLUE_NOISE, rtxCtx.getResourceManager().getBlueNoiseTexture(ctx), nullptr);
    ctx->bindResourceView(CLOUD_BINDING_PRIMARY_LINEAR_VIEW_Z, rtOutput.m_primaryLinearViewZ.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_PSR_FIRST_HIT_DISTANCE, rtOutput.m_secondaryHitDistance.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_PSR_REFLECTION_SEGMENT, rtOutput.m_secondaryLinearViewZ.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_PSR_REFLECTION_DIRECTION, rtOutput.m_secondaryViewDirection.view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(CLOUD_BINDING_SHARED_FLAGS, rtOutput.m_sharedFlags.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_HISTORY, m_layer[history].view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_HISTORY_AGE, m_layerAge[history].view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_DOME_LUT, m_dome[m_domeIndex].view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_SHADOW_MAP, m_shadowMap.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_LAYER_OUTPUT, m_layer[output].view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_HISTORY_AGE_OUTPUT, m_layerAge[output].view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_COMPOSITE_LAYER_OUTPUT, m_compositeLayer.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_REFLECTION_OUTPUT, m_reflection[output].view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_REFLECTION_HISTORY, m_reflection[history].view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_DEBUG_VIEW, debugView.getDebugOutput(), nullptr);
    ctx->bindResourceView(CLOUD_BINDING_MOTION_VECTOR_OUTPUT, rtOutput.m_primaryScreenSpaceMotionVector.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_MOTION_VECTOR_RR_OUTPUT, rtOutput.m_primaryScreenSpaceMotionVectorDLSSRR.view, nullptr);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_layer[history].image);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_layerAge[history].image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_layer[output].image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_layerAge[output].image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_compositeLayer.image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_reflection[output].image);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_reflection[history].image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(rtOutput.m_primaryScreenSpaceMotionVector.image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(rtOutput.m_primaryScreenSpaceMotionVectorDLSSRR.image);

    const VkExtent3D groups = util::computeBlockCount(VkExtent3D { extent.width, extent.height, 1 }, VkExtent3D { 8, 8, 1 });
    ctx->dispatch(groups.width, groups.height, groups.depth);

    // Layer writes before composite reads them.
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

    m_layerIndex = output;
    m_screenHistoryValid = true;
    m_screenRanThisFrame = true;

    m_referenceRanThisFrame = false;
    if (m_args.flags & CLOUD_FLAG_REFERENCE) {
      dispatchReference(rtxCtx, rtOutput);
    }
  }

  void RtxClouds::dispatchGlossyReflections(RtxContext& rtxCtx, const Resources::RaytracingOutput& rtOutput) {
    if (!m_initialized || !m_args.enabled || (m_args.flags & CLOUD_FLAG_GLOSSY_REFLECTIONS) == 0) {
      return;
    }
    Rc<DxvkContext> ctx = &rtxCtx;

    ScopedGpuProfileZone(ctx, "Clouds Glossy Reflections");

    // The indirect integrator wrote the rays and the radiance their clouds are replaced in.
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    Rc<DxvkBuffer> raytraceConstants = rtxCtx.getResourceManager().getConstantsBuffer();

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudGlossyShader::getShader());
    bindCloudInputs(ctx, m_constantsBuffer);
    ctx->bindResourceBuffer(CLOUD_BINDING_CAMERA, DxvkBufferSlice(raytraceConstants, 0, raytraceConstants->info().size));
    ctx->bindResourceView(CLOUD_BINDING_BLUE_NOISE, rtxCtx.getResourceManager().getBlueNoiseTexture(ctx), nullptr);
    ctx->bindResourceView(CLOUD_BINDING_DOME_LUT, m_dome[m_domeIndex].view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_SKY_VIEW_LUT, rtxCtx.getAtmosphereSkyViewLutView(), nullptr);
    ctx->bindResourceView(CLOUD_BINDING_GLOSSY_RAY, m_glossyRay.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_INDIRECT_RADIANCE, rtOutput.m_indirectRadianceHitDistance.view(Resources::AccessType::ReadWrite), nullptr);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_glossyRay.image);

    // A group per 16x16 tile (cloud_glossy.comp.slang).
    const VkExtent3D groups = util::computeBlockCount(VkExtent3D { m_screenExtent.width, m_screenExtent.height, 1 }, VkExtent3D { 16, 16, 1 });
    ctx->dispatch(groups.width, groups.height, groups.depth);

    // Before the NEE pass and demodulation read the radiance, and the next frame's integrator writes the stash.
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  }

  bool RtxClouds::referenceInputsChanged(const RtCamera& camera, const CloudArgs& args, const AtmosphereArgs& atmosphere) {
    const Vector3 position = camera.getPosition();
    const Vector3 direction = camera.getDirection();
    const Vector3 sun(atmosphere.sunDirection.x, atmosphere.sunDirection.y, atmosphere.sunDirection.z);

    // Arguments that do not change the converged image: the per frame ones (the frame index, the history and
    // shadow map flags), the debug view, the reference's own pace, and the camera, which is compared below with
    // a tolerance.
    const auto converged = [](CloudArgs a) {
      a.frameIndex = 0;
      a.flags &= ~(CLOUD_FLAG_HISTORY_VALID | CLOUD_FLAG_SHADOW_MAP_VALID);
      a.debugView = 0;
      a.referenceBouncesPerFrame = 0;
      a.cameraPositionKm = Vector3(0.0f);
      a.cameraWorldHeightKm = 0.0f;
      // Read off the game's projection every frame, it differs in its last bits; the reference does not use it.
      a.pixelAngle = 0.0f;
      return a;
    };
    const CloudArgs stable = converged(args);
    const CloudArgs previous = converged(m_referenceArgs);
    m_referenceArgs = args;

    // Against the view the accumulation started from, so a first person camera's idle sway (Mirror's Edge turns
    // the view up to a degree) keeps accumulating while a slow drift still restarts it.
    const char* reason = nullptr;
    if (m_referenceFrames == 0) {
      reason = "start";
    } else if (length(position - m_referenceCameraPosition) > 1e-3f * args.worldUnitsPerKm) {
      reason = "the camera moved";
    } else if (dot(direction, m_referenceCameraDirection) < std::cos(2.0f * kCloudPi / 180.0f)) {
      reason = "the camera turned";
    } else if (dot(sun, m_referenceSunDirection) < 0.999999f) {
      reason = "the sun moved";
    } else if (memcmp(&stable, &previous, sizeof(CloudArgs)) != 0) {
      reason = "a cloud option changed";
    }
    if (reason == nullptr) {
      return false;
    }
    if (m_referenceFrames > 64) {
      Logger::info(str::format("[Clouds] Reference restarted after ", m_referenceFrames, " frames: ", reason));
    }
    m_referenceCameraPosition = position;
    m_referenceCameraDirection = direction;
    m_referenceSunDirection = sun;
    return true;
  }

  void RtxClouds::readReferenceStatistics(uint32_t frameId) {
    // The oldest slot, which the GPU has written by now; a reset leaves the first few frames without one.
    ReferenceErrorReadout readout;
    if (m_referenceFrames > kMaxFramesInFlight) {
      const VkDeviceSize slotSize = CLOUD_REFERENCE_STATISTICS_SIZE * sizeof(uint32_t);
      const uint32_t* data = reinterpret_cast<const uint32_t*>(
        m_referenceStatisticsReadback->mapPtr(VkDeviceSize((frameId + 1) % kMaxFramesInFlight) * slotSize));
      const uint32_t tileCount = ((m_screenExtent.width + 3) / 4) * ((m_screenExtent.height + 3) / 4);
      readout.pathsPerTile = float(data[2]) / float(std::max(tileCount, 1u));
      readout.tiles = data[0];
      if (readout.tiles > 0) {
        readout.meanAbsolute = float(data[1]) / 1024.0f / float(readout.tiles);
        readout.bias = data[4] > 0 ? float(double(data[3]) / double(data[4]) - 1.0) : 0.0f;
        const uint32_t target = uint32_t(std::ceil(0.95 * double(readout.tiles)));
        uint32_t cumulative = 0;
        for (uint32_t bin = 0; bin < CLOUD_REFERENCE_ERROR_BINS; ++bin) {
          cumulative += data[CLOUD_REFERENCE_STATISTICS_HEADER + bin];
          if (cumulative >= target) {
            readout.percentile95 = float(bin + 1) * 2.0f / float(CLOUD_REFERENCE_ERROR_BINS);
            break;
          }
        }
      }
    }
    m_referenceErrorReadout = readout;

    if (readout.tiles > 0 && m_referenceFrames % 128 == 0) {
      Logger::info(str::format("[Clouds] Reference after ", m_referenceFrames, " frames (", readout.pathsPerTile, " paths a tile): ", readout.tiles,
        " opaque tiles, mean |error| ", int(readout.meanAbsolute * 100.0f + 0.5f), "%, 95th percentile ",
        int(readout.percentile95 * 100.0f + 0.5f), "%, bias ", int(std::round(readout.bias * 100.0f)), "%"));
    }
  }

  void RtxClouds::dispatchReference(RtxContext& rtxCtx, const Resources::RaytracingOutput& rtOutput) {
    Rc<DxvkContext> ctx = &rtxCtx;
    ScopedGpuProfileZone(ctx, "Clouds Reference");

    ensureReferenceResources(ctx);
    ++m_referenceFrames;
    const uint32_t frameId = ctx->getDevice()->getCurrentFrameId();
    readReferenceStatistics(frameId);
    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
    CloudPassArgs pushArgs = {};
    pushArgs.phase = m_referenceFrames - 1;
    pushArgs.level = m_referenceFrames > 1 ? 1u : 0u;

    Rc<DxvkBuffer> raytraceConstants = rtxCtx.getResourceManager().getConstantsBuffer();
    DebugView& debugView = rtxCtx.getCommonObjects()->metaDebugView();

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudReferenceShader::getShader());
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
    bindCloudInputs(ctx, m_constantsBuffer);
    ctx->bindResourceBuffer(CLOUD_BINDING_CAMERA, DxvkBufferSlice(raytraceConstants, 0, raytraceConstants->info().size));
    ctx->bindResourceView(CLOUD_BINDING_PRIMARY_LINEAR_VIEW_Z, rtOutput.m_primaryLinearViewZ.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_PSR_FIRST_HIT_DISTANCE, rtOutput.m_secondaryHitDistance.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_SHARED_FLAGS, rtOutput.m_sharedFlags.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_HISTORY, m_compositeLayer.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_PHASE_CDF, m_phaseCdf.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_REFERENCE_ACCUMULATION, m_referenceAccumulation.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_REFERENCE_PATH_POSITION, m_referencePath[0].view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_REFERENCE_PATH_DIRECTION, m_referencePath[1].view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_REFERENCE_PATH_THROUGHPUT, m_referencePath[2].view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_REFERENCE_PATH_RADIANCE, m_referencePath[3].view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_LAYER_OUTPUT, m_referenceMean.view, nullptr);
    ctx->bindResourceView(CLOUD_BINDING_DEBUG_VIEW, debugView.getDebugOutput(), nullptr);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_compositeLayer.image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_referenceAccumulation.image);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_referenceMean.image);
    for (const Resources::Resource& path : m_referencePath) {
      ctx->getCommandList()->trackResource<DxvkAccess::Write>(path.image);
    }
    const VkDeviceSize statisticsSlotSize = CLOUD_REFERENCE_STATISTICS_SIZE * sizeof(uint32_t);
    const VkDeviceSize statisticsOffset = VkDeviceSize(frameId % kMaxFramesInFlight) * statisticsSlotSize;
    ctx->clearBuffer(m_referenceStatistics, statisticsOffset, statisticsSlotSize, 0);
    ctx->bindResourceBuffer(CLOUD_BINDING_REFERENCE_STATISTICS, DxvkBufferSlice(m_referenceStatistics, statisticsOffset, statisticsSlotSize));

    // One thread per 4x4 pixel tile.
    const VkExtent3D tiles = { (m_screenExtent.width + 3) / 4, (m_screenExtent.height + 3) / 4, 1 };
    const VkExtent3D groups = util::computeBlockCount(tiles, VkExtent3D { 8, 8, 1 });
    ctx->dispatch(groups.width, groups.height, groups.depth);
    ctx->copyBuffer(m_referenceStatisticsReadback, statisticsOffset, m_referenceStatistics, statisticsOffset, statisticsSlotSize);

    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    m_referenceRanThisFrame = true;
  }

} // namespace dxvk
