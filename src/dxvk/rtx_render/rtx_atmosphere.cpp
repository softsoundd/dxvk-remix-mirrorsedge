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
#include "rtx_render/rtx_shader_manager.h"
#include "rtx/pass/common_binding_indices.h"
#include <rtx_shaders/transmittance_lut.h>
#include <rtx_shaders/multiscattering_lut.h>
#include <rtx_shaders/sky_view_lut.h>
#include <rtx_shaders/rtx_cloud_noise_baker.h>
#include <rtx_shaders/cloud_sky_transmittance_lut.h>
#include <rtx_shaders/cloud_sun_density_grid.h>
#include <rtx_shaders/cloud_ambient_density_grid.h>
#include <rtx_shaders/cloud_render.h>
#include <rtx_shaders/cloud_secondary_lut.h>
#include <rtx_shaders/cloud_height_lut_baker.h>
#include <rtx_shaders/cloud_placement_map_baker.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <chrono>

namespace dxvk {
  // Shader definitions for atmosphere LUT generation
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
        SAMPLER(2)
        RW_TEXTURE2D(3)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(MultiscatteringLutShader);

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

    // Stage C: one-shot bake of the 256-cubed R8 cloud noise volume.
    class CloudNoiseBakerShader : public ManagedShader {
      SHADER_SOURCE(CloudNoiseBakerShader, VK_SHADER_STAGE_COMPUTE_BIT, rtx_cloud_noise_baker)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        RW_TEXTURE3D(1)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudNoiseBakerShader);

    // Fork: per-frame bake of the cloud-occluded sky-ambient transmittance LUT.
    // 32x16 R16F keyed by (azimuth, elevation). Consumed by the volumetric pass.
    class CloudSkyTransmittanceLutShader : public ManagedShader {
      SHADER_SOURCE(CloudSkyTransmittanceLutShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_sky_transmittance_lut)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        RW_TEXTURE2D(1)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudSkyTransmittanceLutShader);

    // Fork (Nubis Cubed 2023, 2026-05-12): round-robin bake of the cloud
    // voxel grids. 256x256x32 R16F precomputed optical depth along the sun
    // direction (D_sun) and zenith (D_ambient). The Nubis Cubed cloud-lighting
    // path reads these at shade time via sampleDSun / sampleDAmbient.
    class CloudSunDensityGridShader : public ManagedShader {
      SHADER_SOURCE(CloudSunDensityGridShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_sun_density_grid)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        RW_TEXTURE3D(1)
        TEXTURE3D(2)
        SAMPLER(3)
        TEXTURE2D(4)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudSunDensityGridShader);

    class CloudAmbientDensityGridShader : public ManagedShader {
      SHADER_SOURCE(CloudAmbientDensityGridShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_ambient_density_grid)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        RW_TEXTURE3D(1)
        TEXTURE3D(2)
        SAMPLER(3)
        TEXTURE2D(4)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudAmbientDensityGridShader);

    // Fork (Nubis Cubed 2023, 2026-05-12, C4): per-frame screen-space cloud
    // raymarch using the Nubis Cubed lighting equations. Writes premultiplied
    // rgb + transmittance alpha to AtmosphereCloudRender at downscale extent.
    // Bindings (kept in lockstep with cloud_render.comp.slang):
    //   0: ConstantBuffer<AtmosphereArgs>
    //   1: Texture3D<float>      (AtmosphereCloudNoise3D)
    //   2: SamplerState          (linear/REPEAT)
    //   3: Texture3D<float>      (AtmosphereCloudDSun)
    //   4: Texture3D<float>      (AtmosphereCloudDAmbient)
    //   5: Texture2DArray<float2>(AtmosphereFastNoise)
    //   6: RWTexture2D<float4>   output
    //   7: Texture2D<float4>     (AtmosphereSkyViewLut)
    //   8: Texture2D<float>      (AtmosphereCloudSkyTransmittanceLut)
    //   9: SamplerState          (linear/CLAMP — sky-view LUT)
    //  10: Texture2D<float>      (AtmosphereCloudHeightLut, slide 3 lift — fork 2026-05-15)
    //  11: SamplerState          (linear/CLAMP — height LUT)
    class CloudRenderShader : public ManagedShader {
      SHADER_SOURCE(CloudRenderShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_render)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        TEXTURE3D(1)
        SAMPLER(2)
        TEXTURE3D(3)
        TEXTURE3D(4)
        TEXTURE2DARRAY(5)
        RW_TEXTURE2D(6)
        TEXTURE2D(7)
        TEXTURE2D(8)
        SAMPLER(9)
        TEXTURE2D(10)
        SAMPLER(11)
        TEXTURE2D(12)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudRenderShader);

    // Fork (2026-06-10, perf): per-frame bake of the secondary-ray cloud LUT.
    // 256x128 RGBA16F dome holding the full Nubis cloud march per direction
    // (rgb = premultiplied radiance, a = view transmittance). Consumed by
    // evalSkyRadiance's non-primary branch in place of a per-ray cloud
    // march. Bindings 0-11 kept in lockstep with
    // cloud_render.comp.slang (slot 6 is this pass's own RW output).
    class CloudSecondaryLutShader : public ManagedShader {
      SHADER_SOURCE(CloudSecondaryLutShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_secondary_lut)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        TEXTURE3D(1)
        SAMPLER(2)
        TEXTURE3D(3)
        TEXTURE3D(4)
        TEXTURE2DARRAY(5)
        RW_TEXTURE2D(6)
        TEXTURE2D(7)
        TEXTURE2D(8)
        SAMPLER(9)
        TEXTURE2D(10)
        SAMPLER(11)
        TEXTURE2D(12)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudSecondaryLutShader);

    // Fork (slide 3 lift — RDR2 SIGGRAPH 2019, 2026-05-15): one-shot bake of
    // the 64x128 R8 cloud height LUT. Indexed (typeSlice, heightFrac) -> per-
    // altitude shape modulator. Consumed by cloud_render.comp.slang via the
    // cloudHeightProfile() helper to replace the procedural cloudTypeProfile
    // trapezoid.
    class CloudHeightLutBakerShader : public ManagedShader {
      SHADER_SOURCE(CloudHeightLutBakerShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_height_lut_baker)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        RW_TEXTURE2D(1)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudHeightLutBakerShader);

    // Fork (column-shaping rework, 2026-06-11): bake of the 512x512 RGBA8
    // cloud placement map (cluster field / top jitter / base lift). At init
    // + re-baked live when cloudCellSizeKm or cloudNoiseTileKm changes.
    class CloudPlacementMapBakerShader : public ManagedShader {
      SHADER_SOURCE(CloudPlacementMapBakerShader, VK_SHADER_STAGE_COMPUTE_BIT, cloud_placement_map_baker)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(0)
        RW_TEXTURE2D(1)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(CloudPlacementMapBakerShader);
  }

RtxAtmosphere::RtxAtmosphere(DxvkDevice* device)
  : CommonDeviceObject(device) {
  // Create constant buffer for atmosphere parameters
  DxvkBufferCreateInfo info;
  info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  info.access = VK_ACCESS_UNIFORM_READ_BIT;
  info.size = sizeof(AtmosphereArgs);
  m_constantsBuffer = device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Atmosphere constants buffer");
}

RtxAtmosphere::~RtxAtmosphere() {
}

void RtxAtmosphere::initialize(Rc<DxvkContext> ctx) {
  if (m_initialized) {
    return;
  }

  createLutResources(ctx);
  dispatchCloudNoise3DBake(ctx);
  cacheCloudNoiseBakeInputs();  // seed the re-bake gate with the launch-time inputs
  dispatchCloudPlacementMapBake(ctx);
  cacheCloudPlacementBakeInputs();
  dispatchCloudHeightLutBake(ctx);
  m_initialized = true;
  m_lutsNeedRecompute = true;
}

namespace {
  // Helper: populate one MoonParams from the indexed RTX_OPTIONs for moon `i`.
  // RTX_OPTION accessors are static methods generated per-option, so we dispatch
  // by index with an inline switch. MAX_MOONS is small (4); deliberate simple
  // repetition is clearer than an indirection layer here.
  void populateMoonParams(MoonParams& m, uint32_t i) {
    constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;

    bool     enabled         = false;
    float    elevationDeg    = 0.0f;
    float    rotationDeg     = 0.0f;
    float    angularDiamDeg  = 0.0f;
    Vector3  color           = Vector3(1.0f, 1.0f, 1.0f);
    float    brightness      = 1.0f;
    uint32_t surfaceStyle    = 0u;
    float    phase           = 0.5f;
    float    craterDensity   = 1.0f;
    float    surfaceContrast = 1.0f;
    float    noiseScale      = 1.0f;
    float    darkSide        = 0.05f;
    float    roughness       = 1.0f;

    switch (i) {
    case 0:
      enabled         = RtxOptions::enabled0();         elevationDeg    = RtxOptions::elevation0();
      rotationDeg     = RtxOptions::rotation0();        angularDiamDeg  = RtxOptions::angularRadius0();
      color           = RtxOptions::color0();           brightness      = RtxOptions::brightness0();
      surfaceStyle    = RtxOptions::surfaceStyle0();    phase           = RtxOptions::phase0();
      craterDensity   = RtxOptions::craterDensity0();   surfaceContrast = RtxOptions::surfaceContrast0();
      noiseScale      = RtxOptions::surfaceNoiseScale0(); darkSide      = RtxOptions::darkSideBrightness0();
      roughness       = RtxOptions::roughnessAmount0();
      break;
    case 1:
      enabled         = RtxOptions::enabled1();         elevationDeg    = RtxOptions::elevation1();
      rotationDeg     = RtxOptions::rotation1();        angularDiamDeg  = RtxOptions::angularRadius1();
      color           = RtxOptions::color1();           brightness      = RtxOptions::brightness1();
      surfaceStyle    = RtxOptions::surfaceStyle1();    phase           = RtxOptions::phase1();
      craterDensity   = RtxOptions::craterDensity1();   surfaceContrast = RtxOptions::surfaceContrast1();
      noiseScale      = RtxOptions::surfaceNoiseScale1(); darkSide      = RtxOptions::darkSideBrightness1();
      roughness       = RtxOptions::roughnessAmount1();
      break;
    case 2:
      enabled         = RtxOptions::enabled2();         elevationDeg    = RtxOptions::elevation2();
      rotationDeg     = RtxOptions::rotation2();        angularDiamDeg  = RtxOptions::angularRadius2();
      color           = RtxOptions::color2();           brightness      = RtxOptions::brightness2();
      surfaceStyle    = RtxOptions::surfaceStyle2();    phase           = RtxOptions::phase2();
      craterDensity   = RtxOptions::craterDensity2();   surfaceContrast = RtxOptions::surfaceContrast2();
      noiseScale      = RtxOptions::surfaceNoiseScale2(); darkSide      = RtxOptions::darkSideBrightness2();
      roughness       = RtxOptions::roughnessAmount2();
      break;
    case 3:
      enabled         = RtxOptions::enabled3();         elevationDeg    = RtxOptions::elevation3();
      rotationDeg     = RtxOptions::rotation3();        angularDiamDeg  = RtxOptions::angularRadius3();
      color           = RtxOptions::color3();           brightness      = RtxOptions::brightness3();
      surfaceStyle    = RtxOptions::surfaceStyle3();    phase           = RtxOptions::phase3();
      craterDensity   = RtxOptions::craterDensity3();   surfaceContrast = RtxOptions::surfaceContrast3();
      noiseScale      = RtxOptions::surfaceNoiseScale3(); darkSide      = RtxOptions::darkSideBrightness3();
      roughness       = RtxOptions::roughnessAmount3();
      break;
    default:
      enabled = false; // out-of-range — leave defaults
      break;
    }

    const float elevRad = elevationDeg * kDegToRad;
    const float aziRad  = rotationDeg  * kDegToRad;
    m.direction.x = std::cos(elevRad) * std::sin(aziRad);
    m.direction.y = std::sin(elevRad);
    m.direction.z = std::cos(elevRad) * std::cos(aziRad);

    m.angularRadius      = (angularDiamDeg * kDegToRad) * 0.5f;
    m.color              = color;
    m.brightness         = brightness;
    m.surfaceStyle       = surfaceStyle;
    m.phase              = phase;
    m.enabled            = enabled ? 1.0f : 0.0f;
    m.craterDensity      = craterDensity;
    m.surfaceContrast    = surfaceContrast;
    m.surfaceNoiseScale  = noiseScale;
    m.darkSideBrightness = darkSide;
    m.roughnessAmount    = roughness;
  }

  // Zero the AtmosphereArgs fields that animate every frame but feed only
  // cloud / runtime-miss shaders — never the transmittance / multiscattering
  // / sky-view LUT bakes. Used to derive a sky-LUT cache key so those LUTs
  // only rebuild when their actual inputs change. Without this, timeSeconds
  // + cloudWindOffset + the per-frame frame indices and camera basis cause
  // the memcmp gate to fire every frame.
  void normalizeForSkyLutCache(AtmosphereArgs& args) {
    args.timeSeconds                 = 0.0f;
    args.cloudWindOffset             = vec2(0.0f, 0.0f);
    // Field-evolution / boil scroll (fork — 2026-06-21): per-frame animated, feeds
    // only the view-path cloud taps (not any LUT bake), so zero them in the key
    // exactly like cloudWindOffset to keep the sky-LUT memcmp gate from firing
    // every frame.
    args.cloudEvolutionOffsetX       = 0.0f;
    args.cloudEvolutionOffsetY       = 0.0f;
    args.cloudEvolutionOffsetZ       = 0.0f;
    args.cloudBoilPhase              = 0.0f;
    args.cloudRenderFrameIdx         = 0u;
    args.cloudRenderForwardYUp       = vec3(0.0f, 0.0f, 0.0f);
    args.cloudRenderRightYUp         = vec3(0.0f, 0.0f, 0.0f);
    args.cloudRenderUpYUp            = vec3(0.0f, 0.0f, 0.0f);
    args.cameraWorldPosYUpKm         = vec3(0.0f, 0.0f, 0.0f);
    // Applied post-LUT-sample per ray, never feeds a LUT bake — exclude from the key.
    args.skyIndirectRadianceScale    = 0.0f;
  }

  // Quantize one direction-vector component to the granularity step.
  // Component-wise snapping of a unit vector: a step of S radians changes a
  // component roughly every S radians of angular travel (within ~sqrt(3)x
  // depending on direction), which is exactly the precision class needed —
  // the option is a perceptual budget, not a geometric guarantee.
  float quantizeDirComponent(float v, float stepRad) {
    return std::floor(v / stepRad + 0.5f) * stepRad;
  }

  // Split cache key for the sky-view LUT bake (fork — 2026-06-11, perf).
  // Extends normalizeForSkyLutCache by zeroing the star / Milky Way fields:
  // they feed only the runtime miss shading (evalNightSky / evalStarField),
  // never any LUT bake. starRotation in particular is game-driven per frame
  // (sidereal animation — see atmosphere_args.h), which made the monolithic
  // memcmp gate fire every frame at night and re-bake the entire
  // transmittance → multiscatter → sky-view cascade for nothing.
  //
  // Sky-view re-bake granularity (fork — 2026-06-11, perf): when
  // skyViewRebakeGranularityDeg > 0, the sun and moon directions are
  // quantized INSIDE the key, so continuous time-of-day motion flips the
  // memcmp only when a direction crosses a granularity step — one re-bake
  // per ~0.1 deg of travel instead of one per frame. The LUT consumed
  // between steps is stale by at most the step angle, which the in-game
  // frozen-cascade bisect showed is imperceptible at far larger errors.
  // Every non-direction field stays exact, so slider / preset changes
  // re-bake immediately as before.
  void normalizeForSkyViewLutKey(AtmosphereArgs& args) {
    normalizeForSkyLutCache(args);

    const float granularityDeg = RtxOptions::skyViewRebakeGranularityDeg();
    if (granularityDeg > 0.0f) {
      constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
      const float stepRad = granularityDeg * kDegToRad;
      args.sunDirection.x = quantizeDirComponent(args.sunDirection.x, stepRad);
      args.sunDirection.y = quantizeDirComponent(args.sunDirection.y, stepRad);
      args.sunDirection.z = quantizeDirComponent(args.sunDirection.z, stepRad);
      for (uint32_t i = 0; i < MAX_MOONS; ++i) {
        args.moons[i].direction.x = quantizeDirComponent(args.moons[i].direction.x, stepRad);
        args.moons[i].direction.y = quantizeDirComponent(args.moons[i].direction.y, stepRad);
        args.moons[i].direction.z = quantizeDirComponent(args.moons[i].direction.z, stepRad);
      }
    }
  }

  // Cache key for the D_sun / D_ambient voxel grid bakes (fork — 2026-06-11,
  // perf). Starts from the sky-view key (per-frame fields zeroed, star /
  // Milky Way fields zeroed, sun + moon directions quantized by
  // skyViewRebakeGranularityDeg), then RE-INJECTS the two motion inputs the
  // grid bake genuinely depends on — wind scroll and camera position, which
  // the base normalization zeroes outright — quantized by the km granularity.
  // Continuous wind / camera motion then flips the memcmp once per step of
  // travel instead of every frame; each re-bake uses exact live values, so
  // grid staleness is bounded by the step. Every cloud parameter stays exact
  // in the key, so slider changes re-bake the same frame.
  // quantizeDirComponent is a plain floor-snap — domain-agnostic, used here
  // on km components.
  void normalizeForVoxelGridKey(AtmosphereArgs& args) {
    const vec2 windKm = args.cloudWindOffset;
    const vec3 camKm  = args.cameraWorldPosYUpKm;
    normalizeForSkyViewLutKey(args);

    const float stepKm = std::max(RtxOptions::cloudVoxelGridRebakeGranularityKm(), 1e-5f);
    args.cloudWindOffset.x     = quantizeDirComponent(windKm.x, stepKm);
    args.cloudWindOffset.y     = quantizeDirComponent(windKm.y, stepKm);
    args.cameraWorldPosYUpKm.x = quantizeDirComponent(camKm.x, stepKm);
    args.cameraWorldPosYUpKm.y = quantizeDirComponent(camKm.y, stepKm);
    args.cameraWorldPosYUpKm.z = quantizeDirComponent(camKm.z, stepKm);

    args.starBrightness     = 0.0f;
    args.starDensity        = 0.0f;
    args.starTwinkleSpeed   = 0.0f;
    args.nightSkyBrightness = 0.0f;
    args.nightSkyColor      = vec3(0.0f, 0.0f, 0.0f);

    args.starRotation      = 0.0f;
    args.starAxisElevation = 0.0f;
    args.starAxisRotation  = 0.0f;

    args.starPsfSharpness            = 0.0f;
    args.starCloudExtinctionPower    = 0.0f;
    args.starAmbientCouplingStrength = 0.0f;

    args.milkyWayEnabled              = 0.0f;
    args.milkyWayDensityBoost         = 0.0f;
    args.milkyWayBackgroundBrightness = 0.0f;
    args.milkyWayBackgroundColor      = vec3(0.0f, 0.0f, 0.0f);
    args.milkyWayDustAmount           = 0.0f;
    args.milkyWayCoreColor            = vec3(0.0f, 0.0f, 0.0f);
    args.milkyWayDustColor            = vec3(0.0f, 0.0f, 0.0f);
  }

  // Split cache key for the transmittance + multiscatter bakes (fork —
  // 2026-06-11, perf). Neither bake reads the sun direction / illuminance /
  // disk size (transmittance is parameterized by zenith angle; multiscatter
  // integrates an isotropic phase over the hemisphere), the analytical /
  // physical multiscatter blend weight (applied at sky-view bake time), or
  // any moon field (moon atmospheric coupling lives in evalAtmosphereRadiance,
  // i.e. the sky-view bake). Zeroing them here means a moving time-of-day sun
  // or game-driven moons re-bake ONLY the sky-view LUT — the multiscatter
  // bake alone is 32x32 texels × 64 directions × 20 steps of transmittance
  // LUT taps, by far the heaviest dispatch of the cascade.
  void normalizeForTransmittanceMsKey(AtmosphereArgs& args) {
    normalizeForSkyViewLutKey(args);

    args.sunDirection                 = vec3(0.0f, 0.0f, 0.0f);
    args.sunIlluminance               = vec3(0.0f, 0.0f, 0.0f);
    args.sunAngularRadius             = 0.0f;
    args.mieAnisotropy                = 0.0f;
    args.multiScatterPhysicalStrength = 0.0f;
    // Artistic sky-color knobs apply in evalAtmosphereRadiance (sky-view bake),
    // not the transmittance/MS LUT bakes — zero them here so changing them does
    // not needlessly re-bake the heavy transmittance + multiscatter pair.
    args.multiScatterStrength         = 0.0f;
    args.sunsetSaturation             = 0.0f;

    args.moonAtmosphericCouplingStrength = 0.0f;
    memset(&args.moons[0], 0, sizeof(args.moons));
  }
} // anonymous namespace

AtmosphereArgs RtxAtmosphere::getAtmosphereArgs() const {
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

  // Multiscattering blend: 0 = artistic (analytical inline), 1 = physical (LUT hemisphere).
  args.multiScatterPhysicalStrength = RtxOptions::multiScatterPhysicalStrength();

  // Artistic sunset color controls (fork — 2026-06-14). multiScatterStrength
  // dials back the pale-blue multiscatter fill; sunsetSaturation boosts warm
  // saturation near the horizon. Both feed the sky-view LUT (and thus clouds).
  // Defaults (1.0 / 1.0) reproduce the physical look. Set unconditionally so the
  // sky reddens even when clouds are disabled.
  args.multiScatterStrength = RtxOptions::multiScatterStrength();
  args.sunsetSaturation     = RtxOptions::sunsetSaturation();

  // Diffuse-indirect sky radiance multiplier. Applied per-ray in evalSkyRadiance
  // (post-LUT-sample), so it never feeds any LUT bake — see normalizeForSkyLutCache,
  // which zeroes it in the cache key so dragging the slider doesn't trigger a rebake.
  args.skyIndirectRadianceScale = std::max(RtxOptions::skyIndirectRadianceScale(), 0.0f);

  // View Altitude (converted m to km)
  args.viewAltitude = RtxOptions::altitude() * 0.001f;

  // LUT dimensions
  args.transmittanceLutWidth = kTransmittanceLutWidth;
  args.transmittanceLutHeight = kTransmittanceLutHeight;
  args.multiscatteringLutSize = kMultiscatteringLutSize;
  args.skyViewLutWidth = kSkyViewLutWidth;
  args.skyViewLutHeight = kSkyViewLutHeight;
  args.useSkyViewLut = RtxOptions::useSkyViewLut() ? 1u : 0u;

  // Derived parameters
  args.atmosphereRadius = args.planetRadius + args.atmosphereThickness;
  args.rayleighScaleHeight = kRayleighScaleHeight;
  args.mieScaleHeight = kMieScaleHeight;

  // ----- Night-sky shading (fork) -----
  args.starBrightness     = RtxOptions::starBrightness();
  args.starDensity        = RtxOptions::starDensity();
  args.starTwinkleSpeed   = RtxOptions::starTwinkleSpeed();
  args.nightSkyBrightness = RtxOptions::nightSkyBrightness();
  args.nightSkyColor      = RtxOptions::nightSkyColor();

  // Monotonic time origin for star-twinkle animation.
  static const auto kStartTime = std::chrono::steady_clock::now();
  args.timeSeconds = std::chrono::duration<float>(
                        std::chrono::steady_clock::now() - kStartTime).count();

  // Sidereal sky rotation. Default axis (elevation 90, rotation 0) keeps the
  // pre-rotation behavior; non-default values come from rtx.conf or game
  // plugin pushes. starRotation is game-drivable per-frame but also persists
  // when saved (last writer wins during a session; cold start uses the saved
  // value until any plugin push lands).
  args.starRotation      = RtxOptions::starRotation();
  args.starAxisElevation = RtxOptions::starAxisElevation();
  args.starAxisRotation  = RtxOptions::starAxisRotation();
  args.pad3              = 0.0f;

  args.starPsfSharpness            = RtxOptions::starPsfSharpness();
  args.starCloudExtinctionPower    = RtxOptions::starCloudExtinctionPower();
  args.starAmbientCouplingStrength = RtxOptions::starAmbientCouplingStrength();
  // Adaptive-march sample cap riding the former padStarCloud0 slot (fork —
  // 2026-06-12, adaptive march sampling); CB layout unchanged.
  args.cloudViewSamplesMax         = static_cast<float>(RtxOptions::cloudViewSamplesMax());

  args.milkyWayEnabled               = RtxOptions::milkyWayEnabled() ? 1.0f : 0.0f;
  args.milkyWayDensityBoost          = RtxOptions::milkyWayDensityBoost();
  args.milkyWayBackgroundBrightness  = RtxOptions::milkyWayBackgroundBrightness();
  args.padMilkyWay0                  = 0.0f;
  args.milkyWayBackgroundColor       = RtxOptions::milkyWayBackgroundColor();
  args.milkyWayDustAmount            = RtxOptions::milkyWayDustAmount();
  args.milkyWayCoreColor             = RtxOptions::milkyWayCoreColor();
  args.padMilkyWay1                  = 0.0f;
  args.milkyWayDustColor             = RtxOptions::milkyWayDustColor();
  args.padMilkyWay2                  = 0.0f;

  // ----- Per-moon parameters (fork) -----
  for (uint32_t i = 0; i < MAX_MOONS; ++i) {
    populateMoonParams(args.moons[i], i);
  }

  // ----- Moon NEE / atmospheric-coupling strengths (fork) -----
  args.moonNeeStrength                 = RtxOptions::moonNeeStrength();
  args.moonAtmosphericCouplingStrength = RtxOptions::moonAtmosphericCouplingStrength();
  args.surfaceMoonBrightness           = RtxOptions::surfaceMoonBrightness();
  args.cloudMoonBrightness             = RtxOptions::cloudMoonBrightness();
  args.haloMoonBrightness              = RtxOptions::haloMoonBrightness();
  // Perf-bisect shader gate (fork — 2026-06-11, diagnostic). Packed into the
  // former padMoonNee2 slot. Only bit 1 (= flat sky miss) remains; bit 0
  // (atmosphere NEE) and bit 2 (bespoke-NEE skip for directional lights) were
  // retired 2026-06-21 with the removal of the bespoke sun/moon NEE. Option
  // defaults true (= bit clear = production path). Bit 1 is read at
  // atmosphere_sky.slangh.
  args.debugSkyBisectFlags             = (RtxOptions::debugEnableSkyMissShading() ? 0u : 2u);

  // ----- Moon cloud-look + halo shape constants (fork, Phase 3 Task 2) -----
  // moonSilverLiningIntensity / moonHaloGlowStrength are master multipliers
  // applied here at args-population time so shaders see the pre-scaled value.
  // Default 1.0 yields byte-identical behavior to pre-master-multiplier builds.
  const float silverLining             = RtxOptions::moonSilverLiningIntensity();
  const float haloGlow                 = RtxOptions::moonHaloGlowStrength();
  args.moonCloudDiffuseGain            = RtxOptions::moonCloudDiffuseGain()  * silverLining;
  args.moonCloudPhaseGain              = RtxOptions::moonCloudPhaseGain()    * silverLining;
  args.moonCloudAnisotropy             = RtxOptions::moonCloudAnisotropy();
  args.moonHaloMagnitude               = RtxOptions::moonHaloMagnitude()     * haloGlow;
  args.moonAmbientAirglow              = RtxOptions::moonAmbientAirglow()    * haloGlow;
  // Hex de-tiling gate (fork — 2026-06-11, stage A). Lives in the former
  // padCloudLook0 slot so the CB layout is unchanged.
  args.cloudHexTilingEnable            = RtxOptions::cloudHexTilingEnable() ? 1.0f : 0.0f;
  // Bake frequency scale (fork — 2026-06-11, stage B). Lives in the former
  // padCloudLook1 slot so the CB layout is unchanged.
  args.cloudNoiseBaseFreqScale         = RtxOptions::cloudNoiseBaseFreqScale();
  // Sky <- clouds bleed (fork — 2026-06-19). Reuses the former
  // cloudColumnShapingEnable (padCloudLook2) slot; see atmosphere_args.h.
  args.cloudSkyBleedStrength           = RtxOptions::cloudSkyBleedStrength();

  // Cloud parameters
  {
    args.cloudColor = RtxOptions::cloudColor();
    args.cloudDensity = RtxOptions::cloudDensity();
    args.cloudAltitude = RtxOptions::cloudAltitude();
    args.cloudEnabled = RtxOptions::cloudEnabled() ? 1.0f : 0.0f;

    // Unified cloud motion (fork — 2026-06-21). Wind advection, field-evolution
    // morph, and edge boil are all integrated once per frame by advanceCloudMotion()
    // (offset += velocity * dt) into persistent members; this const accessor just
    // reads them. This replaced the former stateless `speed * timeSeconds`: that
    // form mis-scaled/rotated the entire accumulated field whenever the slow
    // weather drift varied cloudWindSpeed / cloudWindDirection (it multiplied the
    // instantaneous speed by total elapsed time instead of integrating). See
    // advanceCloudMotion().
    args.cloudWindOffset.x     = m_cloudAdvectOffset.x;
    args.cloudWindOffset.y     = m_cloudAdvectOffset.y;
    args.cloudEvolutionOffsetX = m_cloudEvolutionOffset.x;
    args.cloudEvolutionOffsetY = m_cloudEvolutionOffset.y;
    args.cloudEvolutionOffsetZ = m_cloudEvolutionOffset.z;
    args.cloudBoilPhase        = m_cloudBoilPhase;

    args.cloudShadowStrength = RtxOptions::cloudShadowStrength();
  }

  // Cloud volumetric / appearance enhancements
  {
    args.cloudThickness = RtxOptions::cloudThickness();
    args.cloudLayer2TypeSpread = RtxOptions::cloudLayer2TypeSpread();
    args.cloudViewSamples = RtxOptions::cloudViewSamples();
    args.cloudCurvature = RtxOptions::cloudCurvature();
    args.cloudTypeMean = RtxOptions::cloudTypeMean();
    args.cloudTypeSpread = RtxOptions::cloudTypeSpread();
    args.cloudTypeNoiseScale = RtxOptions::cloudTypeNoiseScale();
    args.cloudCoverageMean = RtxOptions::cloudCoverageMean();
    args.cloudCoverageSpread = RtxOptions::cloudCoverageSpread();
    args.cloudCoverageNoiseScale = RtxOptions::cloudCoverageNoiseScale();
    args.cloudAnvilBias = RtxOptions::cloudAnvilBias();
    args.cloudMsScale = RtxOptions::cloudMsScale();
    args.cloudMultiScatterOctaves = RtxOptions::cloudMultiScatterOctaves();
    args.cloudLayer2NoiseSeed = RtxOptions::cloudLayer2NoiseSeed();
    args.cloudNoiseTileKm = RtxOptions::cloudNoiseTileKm();
    // Volumetric sky-ambient illumination knobs (fork, 2026-05-12). Defaults
    // applied here are the ship-state defaults: skyAmbientStrength = 0 keeps
    // the feature off by default; cloudOcclusionStrength = 1 means full
    // physical cloud occlusion when the feature is enabled.
    args.cloudSkyAmbientStrength = RtxOptions::cloudSkyAmbientStrength();
    args.cloudSkyAmbientCloudOcclusionStrength = RtxOptions::cloudSkyAmbientCloudOcclusionStrength();
    // Cloud cluster footprint for the placement map bake (column-shaping
    // rework). Lives in the former padCloudC2 slot; CB layout unchanged.
    args.cloudCellSizeKm = RtxOptions::cloudCellSizeKm();

    // Cloud voxel grid extent (Nubis Cubed 2023, fork — 2026-05-12).
    // Horizontal: track cloudNoiseTileKm so the grid's frac-wrap stays aligned
    // with the noise period at ALL tile values — the sampleDSun / sampleDAmbient
    // math assumes extent == tile. Previously hardcoded 12 km, which only held
    // at the default tile; non-divisor tiles (7-11) desynced the voxel-grid
    // lighting from the density field. Vertical: track cloudThickness so the
    // grid spans the slab vertically. cloudThickness is already in km per
    // atmosphere_args.h:149.
    args.cloudVoxelGridExtentKm    = RtxOptions::cloudNoiseTileKm();
    args.cloudVoxelGridVerticalKm  = args.cloudThickness;
    // Bottom darkening + additive edge detail (fork — 2026-06-10). Live in the
    // former pad_cloudVoxel0..2 slots so the CB layout is unchanged.
    args.cloudBottomDarkening       = RtxOptions::cloudBottomDarkening();
    args.cloudSkyAmbientFill        = RtxOptions::cloudSkyAmbientFill();
    args.cloudDetailStrength        = RtxOptions::cloudDetailStrength();
  }

  // Nubis Cubed 2023 lighting params (fork — 2026-05-12, C4). Sourced from
  // RTX_OPTIONs so the user can tune from ImGui without rebuilding shaders.
  // The cloud_render compute pass consumes these via evalNubisCubedSample.
  {
    args.cloudPhaseG1         = RtxOptions::cloudPhaseG1();
    args.cloudPhaseG2         = RtxOptions::cloudPhaseG2();
    args.cloudEnergyConserve  = RtxOptions::cloudEnergyConserve();
    args.cloudMsLobeWeight    = RtxOptions::cloudMsLobeWeight();
    args.cloudMsSunDotMax     = RtxOptions::cloudMsSunDotMax();
    args.cloudMsSigmaShallow  = RtxOptions::cloudMsSigmaShallow();
    args.cloudMsSigmaDeep     = RtxOptions::cloudMsSigmaDeep();
    args.cloudMsSdfDepth      = RtxOptions::cloudMsSdfDepth();
    args.cloudRenderFrameIdx  = m_cloudRenderFrameIdx;
    args.cloudDetailScale     = RtxOptions::cloudDetailScale();

    args.cloudSunsetAmbientStrength    = RtxOptions::cloudSunsetAmbientStrength();
    args.cloudSunsetAmbientReachInvKm  = RtxOptions::cloudSunsetAmbientReachInvKm();
    args.cloudSunsetAmbientRampHighSun = RtxOptions::cloudSunsetAmbientRampHighSun();
    // Adaptive-march step target riding the former pad_cloudSunsetAmbient0
    // slot (fork — 2026-06-12, adaptive march sampling); CB layout unchanged.
    args.cloudViewStepKm               = RtxOptions::cloudViewStepKm();
    // Cloud-edge / halo tuning (fork — 2026-06-13). Live knobs for silhouette
    // softness and the thin-edge ambient haze fade.
    args.cloudEdgeSoftness             = RtxOptions::cloudEdgeSoftness();
    args.cloudEdgeAmbientFade          = RtxOptions::cloudEdgeAmbientFade();

    // Independent sun-only scale for volumetric fog in-scattering (issue #35).
    args.atmosphereSunVolumetricRadianceScale = RtxOptions::atmosphereSunVolumetricRadianceScale();
  }

  // Cloud render camera basis (fork — 2026-05-12, C4). Pushed from
  // updateAtmosphereConstants via setCloudRenderCameraBasis() before
  // computeLuts runs, so the values here are this-frame-fresh. The Right /
  // Up vectors are pre-scaled by tan(halfFovX/Y) and aspect ratio so the
  // shader does just a weighted sum.
  {
    args.cloudRenderForwardYUp = m_cloudRenderForwardYUp;
    args.cloudRenderRightYUp   = m_cloudRenderRightYUp;
    args.cloudRenderUpYUp      = m_cloudRenderUpYUp;
    // Column-shaping scalars riding the former pad_cr0..2 slots (fork —
    // 2026-06-11, column-shaping rework); CB layout unchanged.
    args.cloudColumnTopVariation   = RtxOptions::cloudColumnTopVariation();
    args.cloudColumnTopShape       = RtxOptions::cloudColumnTopShape();
    args.cloudColumnBaseVariation  = RtxOptions::cloudColumnBaseVariation();
  }

  // Nubis Cubed sky-miss composite gate (fork — 2026-05-12, C5).
  // Drives the primary-ray-only branch in evalSkyRadiance that composites the
  // prerendered AtmosphereCloudRender RT (when off, primary sky-miss is
  // cloudless). Default false until visual confirmation; flipped to true in C7.
  {
    args.cloudRenderRTEnable = RtxOptions::cloudRenderRTEnable() ? 1u : 0u;
    // Secondary-ray cloud LUT gate (fork — 2026-06-10, perf). Lives in the
    // former pad_c5_0 slot so the CB layout is unchanged.
    args.cloudSecondaryLutEnable = RtxOptions::cloudSecondaryLutEnable() ? 1u : 0u;
    // Downscale extent for the half-res cloud-RT composite (fork —
    // 2026-06-11). Zero until ensureCloudRenderRT has seen a real extent;
    // the shader falls back to the legacy Load path in that case.
    args.cloudRenderFullDimX = m_cloudRenderFullExtent.width;
    args.cloudRenderFullDimY = m_cloudRenderFullExtent.height;
  }

  // Voxel-grid cloud-on-terrain shadow plumbing (fork — 2026-05-12, C6).
  //   * cloudVoxelShadowsEnable / cloudShadowMarchStrength surface the C6
  //     RTX_OPTIONs to the shader.
  //   * worldUnitsPerKm derives from RtxOptions::sceneScale (cm per game
  //     unit): 1 km = 100000 cm and 1 cm = sceneScale game units, so
  //     1 km = 100000 * sceneScale game units. Matches the canonical
  //     getMeterToWorldUnitScale = 100 * sceneScale (world units per meter)
  //     convention used everywhere else in the runtime.
  //   * cameraWorldPosYUpKm is pushed by setCloudShadowCameraPosition()
  //     before computeLuts runs; default value is zero (no
  //     setCloudShadowCameraPosition call yet → camera-relative reframe
  //     reduces to "absolute frame", and the helper is gated off by default).
  {
    args.cloudVoxelShadowsEnable  = RtxOptions::cloudVoxelShadowsEnable() ? 1u : 0u;
    args.cloudShadowMarchStrength = RtxOptions::cloudShadowMarchStrength();
    // Artistic contrast curve on the cloud-on-terrain shadow (fork — 2026-06-19).
    // Folded onto the SUN's radiance as pow(cloudTransmittance, k) inside the sun
    // NEE helpers. Moved here from composite when the cloud shadow was
    // re-architected onto the sun term (the screen-space PrimaryCloudShadowFactor
    // texture it used to scale was deleted). >= 0 clamp matches the old composite
    // populate.
    args.cloudShadowFactorStrength = std::max(RtxOptions::cloudShadowFactorStrength(), 0.0f);
    const float sceneScale = std::max(RtxOptions::sceneScale(), 1e-5f);
    args.worldUnitsPerKm = 100000.0f * sceneScale;
    // Column presence feather band riding the former pad_c6_0 slot (fork —
    // 2026-06-11, column-shaping rework); CB layout unchanged.
    args.cloudColumnFeather = RtxOptions::cloudColumnFeather();
    args.cameraWorldPosYUpKm = m_cameraWorldPosYUpKm;
    // Per-column downwelling-light sigma riding the former pad_c6_1 slot
    // (fork — 2026-06-12, column-shaping rev 3); CB layout unchanged.
    args.cloudUndersideLightSigma = RtxOptions::cloudUndersideLightSigma();
  }

  // Cloud Height LUT + two-layer cloud map (slides 1 + 3 lift, fork — 2026-05-15).
  // Pulled from RTX_OPTIONs so ImGui tuning works without rebuilding shaders.
  // Default cloudLayer2Enable = false means today's single-layer Nubis Cubed
  // look is preserved bit-for-bit until the user opts in.
  {
    args.cloudHeightLutEnable     = RtxOptions::cloudHeightLutEnable() ? 1u : 0u;

    args.cloudLayer2Enable        = RtxOptions::cloudLayer2Enable() ? 1u : 0u;
    args.cloudLayer2Altitude      = RtxOptions::cloudLayer2Altitude();
    args.cloudLayer2Thickness     = RtxOptions::cloudLayer2Thickness();
    args.cloudLayer2TypeMean      = RtxOptions::cloudLayer2TypeMean();
    args.cloudLayer2CoverageMean  = RtxOptions::cloudLayer2CoverageMean();
    args.cloudLayer2DensityScale  = RtxOptions::cloudLayer2DensityScale();
    args.cloudLayer2StepFloor     = RtxOptions::cloudLayer2StepFloor();
    args.cloudLayer2StepMax       = RtxOptions::cloudLayer2StepMax();
    args.cloudLayer2Color         = RtxOptions::cloudLayer2Color();
    args.cloudVerticalStretch     = RtxOptions::cloudVerticalStretch();

    // Worley carve params — consumed only by rtx_cloud_noise_baker. Changing
    // these (or cloudNoiseTileKm) re-bakes the noise volume live via the
    // needsCloudNoiseRebake() gate in computeLuts; no relaunch required.
    args.cloudWorleyCarveStrength = RtxOptions::cloudWorleyCarveStrength();
    args.cloudWorleyFrequency     = RtxOptions::cloudWorleyFrequency();
    args.cloudWorleyOctaves       = RtxOptions::cloudWorleyOctaves();
    args.cloudAerialHazePerKm = RtxOptions::cloudAerialHazePerKm();
    args.cloudAerialFadePerKm = RtxOptions::cloudAerialFadePerKm();
  }

  return args;
}

bool RtxAtmosphere::needsLutRecompute() const {
  if (!m_initialized || m_lutsNeedRecompute) {
    return true;
  }

  // Compare a normalized snapshot against the normalized cached snapshot.
  // normalizeForSkyLutCache zeroes per-frame-animated fields (timeSeconds,
  // cloudWindOffset, cloud render frame index + camera basis, camera world
  // pos, voxel-grid dirty flags) that feed only cloud / runtime-miss
  // shaders — they don't gate sky-LUT validity. Without normalization the
  // memcmp fires every frame even when no real sky parameter changed.
  AtmosphereArgs currentArgs = getAtmosphereArgs();
  normalizeForSkyLutCache(currentArgs);
  return memcmp(&currentArgs, &m_cachedArgs, sizeof(AtmosphereArgs)) != 0;
}

bool RtxAtmosphere::needsCloudNoiseRebake() const {
  // Compares only the inputs rtx_cloud_noise_baker.comp.slang actually reads:
  // cloudNoiseTileKm (world tile period) and the cloudWorley* carve controls.
  // baseFreq / detailFreq / octave seeds are shader-side constants, so they
  // never trigger a re-bake.
  return m_cachedNoiseTileKm         != RtxOptions::cloudNoiseTileKm()
      || m_cachedWorleyFrequency     != RtxOptions::cloudWorleyFrequency()
      || m_cachedWorleyOctaves       != RtxOptions::cloudWorleyOctaves()
      || m_cachedWorleyCarveStrength != RtxOptions::cloudWorleyCarveStrength()
      || m_cachedBaseFreqScale       != RtxOptions::cloudNoiseBaseFreqScale();
}

void RtxAtmosphere::cacheCloudNoiseBakeInputs() {
  m_cachedNoiseTileKm         = RtxOptions::cloudNoiseTileKm();
  m_cachedWorleyFrequency     = RtxOptions::cloudWorleyFrequency();
  m_cachedWorleyOctaves       = RtxOptions::cloudWorleyOctaves();
  m_cachedWorleyCarveStrength = RtxOptions::cloudWorleyCarveStrength();
  m_cachedBaseFreqScale       = RtxOptions::cloudNoiseBaseFreqScale();
}

bool RtxAtmosphere::needsCloudPlacementRebake() const {
  // Compares only the inputs cloud_placement_map_baker.comp.slang reads:
  // cloudCellSizeKm (cluster footprint) and cloudNoiseTileKm (the map's tile
  // period — the cells-per-tile rounding depends on both).
  return m_cachedPlacementCellSizeKm != RtxOptions::cloudCellSizeKm()
      || m_cachedPlacementTileKm     != RtxOptions::cloudNoiseTileKm();
}

void RtxAtmosphere::cacheCloudPlacementBakeInputs() {
  m_cachedPlacementCellSizeKm = RtxOptions::cloudCellSizeKm();
  m_cachedPlacementTileKm     = RtxOptions::cloudNoiseTileKm();
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

  // Stage C: 3D R8 noise volume (256-cubed, ~16 MB). Filled once at init.
  VkExtent3D cloudNoise3DExtent = { kCloudNoise3DSize, kCloudNoise3DSize, kCloudNoise3DSize };
  m_cloudNoise3D = Resources::createImageResource(
    ctx,
    "Atmosphere Cloud Noise 3D",
    cloudNoise3DExtent,
    VK_FORMAT_R8_UNORM,
    1, // numLayers
    VK_IMAGE_TYPE_3D,
    VK_IMAGE_VIEW_TYPE_3D,
    0, // imageCreateFlags
    VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags
    VkClearColorValue{}, // clearValue
    1 // mipLevels
  );

  // Fork: cloud-occluded sky-ambient transmittance LUT (2D R16F, 32x16).
  // Baked every frame from the camera position; consumed by the volumetric
  // pass's sky-ambient hemisphere integration.
  VkExtent3D cloudSkyTransmittanceLutExtent = {
    kCloudSkyTransmittanceLutWidth, kCloudSkyTransmittanceLutHeight, 1
  };
  m_cloudSkyTransmittanceLut = Resources::createImageResource(
    ctx,
    "Atmosphere Cloud Sky Transmittance LUT",
    cloudSkyTransmittanceLutExtent,
    VK_FORMAT_R16_SFLOAT,
    1, // numLayers
    VK_IMAGE_TYPE_2D,
    VK_IMAGE_VIEW_TYPE_2D,
    0, // imageCreateFlags
    VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags
    VkClearColorValue{}, // clearValue
    1 // mipLevels
  );

  // Fork (Nubis Cubed 2023, 2026-05-12): cloud D_sun voxel grid (3D R16F,
  // 256x256x32). Camera-centered tile-wrapped precomputation of summed
  // optical depth along the sun direction. Round-robin baked every 8 frames
  // at offset 0. Consumed at shade time via sampleDSun.
  VkExtent3D cloudVoxelGridExtent = {
    kCloudVoxelGridX, kCloudVoxelGridY, kCloudVoxelGridZ
  };
  m_cloudDSun = Resources::createImageResource(
    ctx,
    "Atmosphere Cloud D_sun Voxel Grid",
    cloudVoxelGridExtent,
    VK_FORMAT_R16_SFLOAT,
    1, // numLayers
    VK_IMAGE_TYPE_3D,
    VK_IMAGE_VIEW_TYPE_3D,
    0, // imageCreateFlags
    VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags (SAMPLED implicit)
    VkClearColorValue{}, // clearValue
    1 // mipLevels
  );

  // Fork (Nubis Cubed 2023, 2026-05-12): cloud D_ambient voxel grid (3D R16F,
  // 256x256x32). Round-robin baked every 8 frames at offset 4.
  m_cloudDAmbient = Resources::createImageResource(
    ctx,
    "Atmosphere Cloud D_ambient Voxel Grid",
    cloudVoxelGridExtent,
    VK_FORMAT_R16_SFLOAT,
    1, // numLayers
    VK_IMAGE_TYPE_3D,
    VK_IMAGE_VIEW_TYPE_3D,
    0, // imageCreateFlags
    VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags
    VkClearColorValue{}, // clearValue
    1 // mipLevels
  );

  // EA Importance-Sampled FAST noise (128x128x32 RG8 Texture2DArray) used for
  // cloud ray-march jitter. One-shot upload of the embedded byte data; no-op on
  // subsequent calls.
  m_fastNoise.initialize(ctx);

  // Fork (slide 3 lift — RDR2 SIGGRAPH 2019, 2026-05-15): cloud height LUT
  // (64x128 RG8 — 16 KB VRAM). Baked once at init by dispatchCloudHeightLutBake.
  // Indexed (typeSlice, heightFrac) -> (R = density envelope, G = coverage
  // threshold scale) by atmosphere_common.slangh's cloudHeightProfileFull
  // inside cloud_render.comp.slang. The G channel is the lever with visible
  // silhouette teeth — it widens cumulus tops by lowering the coverage
  // threshold at the right altitudes.
  VkExtent3D cloudHeightLutExtent = {
    kCloudHeightLutWidth, kCloudHeightLutHeight, 1
  };
  m_cloudHeightLut = Resources::createImageResource(
    ctx,
    "Atmosphere Cloud Height LUT",
    cloudHeightLutExtent,
    VK_FORMAT_R8G8B8A8_UNORM,
    1, // numLayers
    VK_IMAGE_TYPE_2D,
    VK_IMAGE_VIEW_TYPE_2D,
    0, // imageCreateFlags
    VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags (SAMPLED implicit)
    VkClearColorValue{}, // clearValue
    1 // mipLevels
  );

  // Fork (2026-06-10, perf): secondary-ray cloud LUT (256x128 RGBA16F,
  // 256 KB). Written every frame by dispatchCloudSecondaryLut; read by
  // evalSkyRadiance's non-primary branch via
  // BINDING_ATMOSPHERE_CLOUD_SECONDARY_LUT. Note the zero clear value means
  // "no cloud but fully OPAQUE" in the (premultiplied rgb, transmittance)
  // convention — harmless because the shader gate (cloudSecondaryLutEnable)
  // and the dispatch gate are the same option, so the LUT is never sampled
  // on a frame it wasn't baked.
  // Mip chain (fork — 2026-06-19): the sky<-clouds bleed samples a COARSE mip
  // of this LUT as a wide neighborhood blur (sampling mip 0 directly showed the
  // 256x128 LUT's coarse texels as faceted cloud edges). 6 levels: 256x128 down
  // to 8x4. updateMipmap (Gaussian) fills mips 1..5 from mip 0 after each bake.
  VkExtent3D cloudSecondaryLutExtent = { kCloudSecondaryLutWidth, kCloudSecondaryLutHeight, 1 };
  m_cloudSecondaryLut = RtxMipmap::createResource(
    ctx,
    "Atmosphere Cloud Secondary LUT",
    cloudSecondaryLutExtent,
    VK_FORMAT_R16G16B16A16_SFLOAT,
    VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags (SAMPLED implicit)
    VkClearColorValue{}, // clearValue
    6 // mipLevels (256x128 -> 8x4)
  );

  // Fork (2026-06-11, column-shaping rework): cloud placement map (512x512
  // RGBA8, 1 MB). R = cluster field, G = top-height jitter, B = base lift,
  // tiled at cloudNoiseTileKm. Baked at init by dispatchCloudPlacementMapBake
  // and re-baked live when cloudCellSizeKm / cloudNoiseTileKm change. Drives
  // the per-column cloud model inside the density samplers.
  VkExtent3D cloudPlacementMapExtent = { kCloudPlacementMapSize, kCloudPlacementMapSize, 1 };
  m_cloudPlacementMap = Resources::createImageResource(
    ctx,
    "Atmosphere Cloud Placement Map",
    cloudPlacementMapExtent,
    VK_FORMAT_R8G8B8A8_UNORM,
    1, // numLayers
    VK_IMAGE_TYPE_2D,
    VK_IMAGE_VIEW_TYPE_2D,
    0, // imageCreateFlags
    VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags (SAMPLED implicit)
    VkClearColorValue{}, // clearValue
    1 // mipLevels
  );
}

void RtxAtmosphere::computeLuts(Rc<DxvkContext> ctx) {
  if (!m_initialized) {
    return;
  }

  // Re-bake the 256^3 cloud noise volume if a bake input changed at runtime
  // (e.g. dragging the ImGui cloudNoiseTileKm slider). The bake encodes the
  // tile period into the texture's periodic structure; the runtime sampler
  // divides world position by the live cloudNoiseTileKm. If they disagree the
  // cloud feature size rescales and the sky bands toward the horizon. Gated by
  // needsCloudNoiseRebake() so it fires only on an actual change, not per
  // frame. Must run before the voxel-grid bakes and cloud render below — all
  // read m_cloudNoise3D — so the write→read barrier orders the fresh volume
  // ahead of those consumers this frame.
  if (needsCloudNoiseRebake()) {
    dispatchCloudNoise3DBake(ctx);
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT);
    cacheCloudNoiseBakeInputs();
    // The voxel grids integrate the noise volume — force their re-bake this
    // frame regardless of the motion-granularity gate below.
    memset(&m_cachedVoxelGridKey, 0, sizeof(m_cachedVoxelGridKey));
  }

  // Column-shaping rework (fork — 2026-06-11): re-bake the cloud placement
  // map when its inputs change (cloudCellSizeKm / cloudNoiseTileKm). Same
  // write→read barrier + voxel-grid key clear as the noise re-bake above — the
  // D_sun / D_ambient grids integrate the column shapes, so they must refresh
  // the same frame. (The height LUT no longer re-bakes here: with the legacy
  // global-slab path removed 2026-06-19 it bakes a single curve family once at
  // init.)
  {
    bool cloudShapeInputsRebaked = false;
    if (needsCloudPlacementRebake()) {
      dispatchCloudPlacementMapBake(ctx);
      cacheCloudPlacementBakeInputs();
      cloudShapeInputsRebaked = true;
    }
    if (cloudShapeInputsRebaked) {
      ctx->emitMemoryBarrier(0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);
      memset(&m_cachedVoxelGridKey, 0, sizeof(m_cachedVoxelGridKey));
    }
  }

  // Sky LUTs (transmittance / multiscattering / sky-view) only rebake when
  // their inputs actually change. Animated fields that feed only cloud and
  // runtime-miss shaders are excluded from the cache key by
  // normalizeForSkyLutCache, so this gate stays false on frames where only
  // wind / time / camera / frame-index advanced — saving the ~0.5 ms of
  // dispatches + barriers per frame that the old memcmp burned.
  //
  // Split cache keys (fork — 2026-06-11, perf). With the split enabled, each
  // bake compares against a key normalized down to the fields it actually
  // reads: star / Milky Way animation (game-driven starRotation each frame)
  // re-bakes nothing, and sun / moon motion (time-of-day) re-bakes only the
  // sky-view LUT instead of dragging the heavy transmittance + multiscatter
  // pair along. tmsDirty implies skyViewDirty — the transmittance/MS key is
  // a strict sub-key of the sky-view key, and the sky-view bake consumes
  // both LUTs, so the explicit OR keeps the data dependency obvious.
  //
  // Perf-bisect gate (fork — 2026-06-11, diagnostic): a continuously-
  // animating time-of-day sun re-bakes the sky-view LUT every frame by
  // design; the toggle freezes the whole cascade so a live session can
  // read its per-frame cost. Sky colors stop tracking the sun while off.
  if (!RtxOptions::debugDispatchSkyLuts()) {
    // Frozen: skip all three bakes and leave caches untouched so the next
    // enabled frame re-evaluates the gates normally.
  } else if (RtxOptions::skyLutCacheKeySplitEnable()) {
    AtmosphereArgs currentArgs = getAtmosphereArgs();
    AtmosphereArgs tmsKey = currentArgs;
    normalizeForTransmittanceMsKey(tmsKey);
    AtmosphereArgs skyViewKey = currentArgs;
    normalizeForSkyViewLutKey(skyViewKey);

    const bool tmsDirty = m_lutsNeedRecompute
        || memcmp(&tmsKey, &m_cachedTransmittanceMsKey, sizeof(AtmosphereArgs)) != 0;
    const bool skyViewDirty = tmsDirty
        || memcmp(&skyViewKey, &m_cachedSkyViewKey, sizeof(AtmosphereArgs)) != 0;

    if (tmsDirty) {
      dispatchTransmittanceLut(ctx);

      // Barrier: Ensure transmittance LUT is written before reading in subsequent passes
      ctx->emitMemoryBarrier(0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);

      dispatchMultiscatteringLut(ctx);

      // Barrier: Ensure multiscattering LUT is written before reading in sky view pass
      ctx->emitMemoryBarrier(0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);

      m_cachedTransmittanceMsKey = tmsKey;
    }

    if (skyViewDirty) {
      dispatchSkyViewLut(ctx);

      // Barrier: order sky-view writes ahead of the cloud-sky-transmittance
      // bake below when the sky-view LUT actually changed this frame.
      ctx->emitMemoryBarrier(0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);

      m_cachedSkyViewKey = skyViewKey;
      // Keep the legacy monolithic key coherent so toggling the split option
      // off mid-session doesn't fire one spurious full re-bake.
      m_cachedArgs = currentArgs;
      normalizeForSkyLutCache(m_cachedArgs);
      m_lutsNeedRecompute = false;
    }
  } else if (needsLutRecompute()) {
    dispatchTransmittanceLut(ctx);

    // Barrier: Ensure transmittance LUT is written before reading in subsequent passes
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT);

    dispatchMultiscatteringLut(ctx);

    // Barrier: Ensure multiscattering LUT is written before reading in sky view pass
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT);

    dispatchSkyViewLut(ctx);

    // Barrier: order sky-view writes ahead of the cloud-sky-transmittance
    // bake below when the sky-view LUT actually changed this frame.
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT);

    // Cache the normalized snapshot for next frame's gate. The split keys are
    // refreshed too so toggling the split option on mid-session is clean.
    AtmosphereArgs currentArgs = getAtmosphereArgs();
    m_cachedArgs = currentArgs;
    normalizeForSkyLutCache(m_cachedArgs);
    m_cachedSkyViewKey = currentArgs;
    normalizeForSkyViewLutKey(m_cachedSkyViewKey);
    m_cachedTransmittanceMsKey = currentArgs;
    normalizeForTransmittanceMsKey(m_cachedTransmittanceMsKey);
    m_lutsNeedRecompute = false;
  }

  // Perf-bisect gate (fork — 2026-06-11, diagnostic): each unconditional
  // per-frame dispatch below gets a default-ON skip toggle so a live ImGui
  // session can attribute frame-time per dispatch. Skipping leaves the
  // consumer reading stale data — diagnostic only.
  if (RtxOptions::debugDispatchCloudSkyTransmittance()) {
    dispatchCloudSkyTransmittanceLut(ctx);
  }

  // Full-rate cloud voxel grid bake (Nubis Cubed 2023, fork — 2026-05-12;
  // full-rate flip 2026-05-19). The original implementation amortized each
  // grid's bake across 8 frames at staggered offsets (D_sun on frame%8==0,
  // D_ambient on frame%8==4). Once the saturate-clamp fix landed and the
  // cumulus-on-terrain shadows became visible, the 8-frame cadence read as
  // a ~2 Hz update stutter on the terrain shadow pattern at 16 fps gameplay.
  // The user asked for full-frame-rate updates — "no shortcuts here" — so
  // both grids are now dispatched every frame.
  //
  // The two bakes run sequentially in the command buffer (not in parallel),
  // separated by the existing write→read barriers, so they don't race for
  // compute units. Cost is ~8× the prior amortized bake; profile if it
  // becomes a frame-time bottleneck and revisit (a smaller grid resolution
  // or per-tile dispatch would be the first cuts to consider).
  // Voxel-grid re-bake granularity (fork — 2026-06-11, perf). At option 0
  // the grids re-bake every frame (legacy). At > 0, they re-bake only when
  // a bake input has moved past its step: wind scroll / camera travel by
  // the km granularity, sun (and moon) direction by the sky-view angular
  // granularity, any other parameter exactly. Cloud-body lighting and (when
  // enabled) terrain cumulus shadows read grids that are stale by at most
  // one step between re-bakes.
  // Force a per-frame voxel-grid re-bake whenever cloud ground shadows are on, so
  // the terrain shadow is fully up to date with zero granularity stepping (fork —
  // 2026-06-21, requested). When shadows are OFF the grid is only needed for
  // cloud-body lighting (which tolerates one step of staleness), so it falls back
  // to the km granularity gate — meaning toggling cloudVoxelShadowsEnable measures
  // the full cost of the cloud-shadow feature (per-frame grid bake + the NEE fold).
  bool voxelGridsDirty = true;
  if (RtxOptions::cloudVoxelGridRebakeGranularityKm() > 0.0f && !RtxOptions::cloudVoxelShadowsEnable()) {
    AtmosphereArgs voxelKey = getAtmosphereArgs();
    normalizeForVoxelGridKey(voxelKey);
    voxelGridsDirty = memcmp(&voxelKey, &m_cachedVoxelGridKey, sizeof(AtmosphereArgs)) != 0;
    if (voxelGridsDirty) {
      m_cachedVoxelGridKey = voxelKey;
    }
  }

  if (RtxOptions::debugDispatchCloudVoxelGrids() && voxelGridsDirty) {
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT);
    dispatchCloudSunDensityGrid(ctx);
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT);
    dispatchCloudAmbientDensityGrid(ctx);
  }

  // Cloud render compute pass (Nubis Cubed 2023, fork — 2026-05-12, C4).
  // Runs every frame after the voxel grid bakes so it reads up-to-date
  // D_sun / D_ambient. As of the full-rate flip 2026-05-19, both grids
  // are rebaked every frame above, so the render reads zero-frame-stale
  // data.
  //
  // NOTE: m_cloudRenderRT is allocated/resized externally via
  // ensureCloudRenderRT() before this dispatch fires. dispatchCloudRender
  // early-outs cleanly if the RT isn't valid yet (first frame, zero extent).
  // Secondary-ray cloud LUT bake (fork — 2026-06-10, perf). Runs after the
  // voxel-grid bakes (the march reads D_sun / D_ambient) behind the same
  // write→read barrier pattern. Gated on the same option the shader-side
  // consumer checks, so the LUT is always fresh on any frame it is sampled.
  if (RtxOptions::cloudSecondaryLutEnable() && m_cloudSecondaryLut.isValid()) {
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT);
    dispatchCloudSecondaryLut(ctx);
  }

  // NOTE (perf-bisect rationale): this dispatch runs whenever the RT is
  // valid, INDEPENDENT of cloudRenderRTEnable — turning that option off
  // makes primary sky-miss cloudless but leaves this pass running, so
  // frame-time A/B via cloudRenderRTEnable never isolates the pass cost.
  // The debug toggle is the only lever that actually skips it.
  if (RtxOptions::debugDispatchCloudRender() && m_cloudRenderRT.isValid()) {
    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT);
    dispatchCloudRender(ctx);
  }

  // Final barrier: Ensure all LUTs are written before use in ray tracing
  ctx->emitMemoryBarrier(0,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_ACCESS_SHADER_WRITE_BIT,
    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
    VK_ACCESS_SHADER_READ_BIT);
}

void RtxAtmosphere::dispatchTransmittanceLut(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Transmittance LUT");
  
  // Update atmosphere args buffer
  AtmosphereArgs args = getAtmosphereArgs();
  ctx->updateBuffer(m_constantsBuffer, 0, sizeof(AtmosphereArgs), &args);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);
  
  // Bind resources
  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_transmittanceLut.view, nullptr);
  
  // Track resources
  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_transmittanceLut.image);
  
  // Bind shader and dispatch
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, TransmittanceLutShader::getShader());
  
  // Dispatch with 16x16 thread groups
  uint32_t groupsX = (kTransmittanceLutWidth + 15) / 16;
  uint32_t groupsY = (kTransmittanceLutHeight + 15) / 16;
  ctx->dispatch(groupsX, groupsY, 1);
}

void RtxAtmosphere::dispatchMultiscatteringLut(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Multiscattering LUT");
  
  // Update atmosphere args buffer
  AtmosphereArgs args = getAtmosphereArgs();
  ctx->updateBuffer(m_constantsBuffer, 0, sizeof(AtmosphereArgs), &args);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);
  
  // Bind resources
  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_transmittanceLut.view, nullptr);

  // Create and bind a linear sampler
  DxvkSamplerCreateInfo samplerInfo = {};
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  Rc<DxvkSampler> linearSampler = m_device->createSampler(samplerInfo);
  ctx->bindResourceSampler(2, linearSampler);
  
  ctx->bindResourceView(3, m_multiscatteringLut.view, nullptr);
  
  // Track resources
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_transmittanceLut.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_multiscatteringLut.image);
  
  // Bind shader and dispatch
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, MultiscatteringLutShader::getShader());
  
  // Dispatch with 16x16 thread groups
  uint32_t groupsX = (kMultiscatteringLutSize + 15) / 16;
  uint32_t groupsY = (kMultiscatteringLutSize + 15) / 16;
  ctx->dispatch(groupsX, groupsY, 1);
}

void RtxAtmosphere::dispatchSkyViewLut(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Sky View LUT");
  
  // Update atmosphere args buffer
  AtmosphereArgs args = getAtmosphereArgs();
  ctx->updateBuffer(m_constantsBuffer, 0, sizeof(AtmosphereArgs), &args);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);
  
  // Bind resources
  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_transmittanceLut.view, nullptr);
  ctx->bindResourceView(2, m_multiscatteringLut.view, nullptr);

  // Create and bind a linear sampler
  DxvkSamplerCreateInfo samplerInfo = {};
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  Rc<DxvkSampler> linearSampler = m_device->createSampler(samplerInfo);
  ctx->bindResourceSampler(3, linearSampler);
  
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

void RtxAtmosphere::dispatchCloudSkyTransmittanceLut(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Cloud Sky Transmittance LUT");

  // Update atmosphere args buffer (the SkyView dispatch above already updates,
  // but the LUT-cascade dispatches each set their own copy to keep ordering
  // explicit and to be safe against future refactors that reorder dispatches).
  AtmosphereArgs args = getAtmosphereArgs();
  ctx->updateBuffer(m_constantsBuffer, 0, sizeof(AtmosphereArgs), &args);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);

  // Bind resources: ConstantBuffer<AtmosphereArgs> at slot 0, RWTexture2D<float> at slot 1.
  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_cloudSkyTransmittanceLut.view, nullptr);

  // Track resources
  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_cloudSkyTransmittanceLut.image);

  // Bind shader and dispatch
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudSkyTransmittanceLutShader::getShader());

  // Dispatch with 8x8 thread groups (shader declares [numthreads(8, 8, 1)]).
  uint32_t groupsX = (kCloudSkyTransmittanceLutWidth + 7) / 8;
  uint32_t groupsY = (kCloudSkyTransmittanceLutHeight + 7) / 8;
  ctx->dispatch(groupsX, groupsY, 1);
}

void RtxAtmosphere::dispatchCloudSunDensityGrid(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Cloud D_sun Bake");

  // Update atmosphere args buffer (mirrors the other dispatch sites — each
  // bake refreshes the buffer to be safe against reordering refactors).
  AtmosphereArgs args = getAtmosphereArgs();
  ctx->updateBuffer(m_constantsBuffer, 0, sizeof(AtmosphereArgs), &args);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);

  // Bind resources: ConstantBuffer<AtmosphereArgs> at 0, RWTexture3D<float>
  // at 1, Texture3D<float> cloud noise volume at 2, linear/REPEAT sampler at 3,
  // cloud placement map at 4 (column-shaping rework).
  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_cloudDSun.view, nullptr);
  ctx->bindResourceView(2, m_cloudNoise3D.view, nullptr);

  // Linear/REPEAT sampler — matches the frac()-tile-wrap convention used by
  // sampleCloudDensityForShadow's texcoord math and by the voxel grid's
  // own UVW mapping in cloudVoxelWorldToUVW.
  DxvkSamplerCreateInfo samplerInfo = {};
  samplerInfo.magFilter    = VK_FILTER_LINEAR;
  samplerInfo.minFilter    = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  Rc<DxvkSampler> cloudSampler = m_device->createSampler(samplerInfo);
  ctx->bindResourceSampler(3, cloudSampler);
  ctx->bindResourceView(4, m_cloudPlacementMap.view, nullptr);

  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_cloudNoise3D.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_cloudPlacementMap.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_cloudDSun.image);

  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudSunDensityGridShader::getShader());

  // Shader declares [numthreads(8, 8, 4)].
  const uint32_t groupsX = (kCloudVoxelGridX + 7u) / 8u;
  const uint32_t groupsY = (kCloudVoxelGridY + 7u) / 8u;
  const uint32_t groupsZ = (kCloudVoxelGridZ + 3u) / 4u;
  ctx->dispatch(groupsX, groupsY, groupsZ);
}

void RtxAtmosphere::dispatchCloudAmbientDensityGrid(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Cloud D_ambient Bake");

  AtmosphereArgs args = getAtmosphereArgs();
  ctx->updateBuffer(m_constantsBuffer, 0, sizeof(AtmosphereArgs), &args);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);

  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_cloudDAmbient.view, nullptr);
  ctx->bindResourceView(2, m_cloudNoise3D.view, nullptr);

  DxvkSamplerCreateInfo samplerInfo = {};
  samplerInfo.magFilter    = VK_FILTER_LINEAR;
  samplerInfo.minFilter    = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  Rc<DxvkSampler> cloudSampler = m_device->createSampler(samplerInfo);
  ctx->bindResourceSampler(3, cloudSampler);
  ctx->bindResourceView(4, m_cloudPlacementMap.view, nullptr);

  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_cloudNoise3D.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_cloudPlacementMap.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_cloudDAmbient.image);

  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudAmbientDensityGridShader::getShader());

  const uint32_t groupsX = (kCloudVoxelGridX + 7u) / 8u;
  const uint32_t groupsY = (kCloudVoxelGridY + 7u) / 8u;
  const uint32_t groupsZ = (kCloudVoxelGridZ + 3u) / 4u;
  ctx->dispatch(groupsX, groupsY, groupsZ);
}

void RtxAtmosphere::ensureCloudRenderRT(Rc<DxvkContext> ctx,
                                          const VkExtent2D& downscaleExtent) {
  // Bail on degenerate extents (can happen during early frames before resize
  // events have settled) — allocate on a later frame.
  if (downscaleExtent.width == 0u || downscaleExtent.height == 0u) {
    return;
  }

  // Half-res cloud RT (fork — 2026-06-11, perf). The RT is allocated at
  // cloudRenderResolutionScale of the downscale extent; the sky-miss
  // composite bilinearly upsamples using the full extent published via
  // args.cloudRenderFullDimX/Y. Scale 1.0 reproduces the legacy native-res
  // path bit-exactly (texel-center bilinear == Load). Live-tunable: a scale
  // change shows up as an extent mismatch below and reallocates.
  m_cloudRenderFullExtent = downscaleExtent;
  const float renderScale = std::min(std::max(RtxOptions::cloudRenderResolutionScale(), 0.25f), 1.0f);
  const VkExtent2D scaledExtent = {
    std::max(1u, static_cast<uint32_t>(std::lround(downscaleExtent.width  * renderScale))),
    std::max(1u, static_cast<uint32_t>(std::lround(downscaleExtent.height * renderScale))),
  };

  const bool extentsMatch = (m_cloudRenderExtent.width  == scaledExtent.width)
                         && (m_cloudRenderExtent.height == scaledExtent.height);
  if (extentsMatch && m_cloudRenderRT.isValid()) {
    return;
  }

  const VkExtent3D extent3D = { scaledExtent.width, scaledExtent.height, 1u };
  m_cloudRenderRT = Resources::createImageResource(
    ctx,
    "Atmosphere Cloud Render RT",
    extent3D,
    VK_FORMAT_R16G16B16A16_SFLOAT,
    1,                          // numLayers
    VK_IMAGE_TYPE_2D,
    VK_IMAGE_VIEW_TYPE_2D,
    0,                          // imageCreateFlags
    VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags (SAMPLED implied)
    VkClearColorValue{},        // clearValue (zero -- "no cloud, full transmittance")
    1);                         // mipLevels

  m_cloudRenderExtent = downscaleExtent;
}

void RtxAtmosphere::setCloudRenderCameraBasis(const Vector3& forwardYUp,
                                                const Vector3& rightYUp,
                                                const Vector3& upYUp,
                                                uint32_t frameIdx) {
  m_cloudRenderForwardYUp = forwardYUp;
  m_cloudRenderRightYUp   = rightYUp;
  m_cloudRenderUpYUp      = upYUp;
  m_cloudRenderFrameIdx   = frameIdx;
}

void RtxAtmosphere::setCloudShadowCameraPosition(const Vector3& cameraWorldPosYUpKm) {
  m_cameraWorldPosYUpKm = cameraWorldPosYUpKm;
}

// Unified cloud-motion integrator (fork — 2026-06-21). Called exactly once per
// frame from updateAtmosphereConstants. Integrates all three cloud-motion sources
// as offset += velocity * dt into persistent members that the const
// getAtmosphereArgs() reads. Wind velocity comes from the LIVE cloudWindSpeed /
// cloudWindDirection — which already carry the slow weather drift (written to the
// Derived config layer by the weather blender) — so the drift now composes
// smoothly: a varying wind velocity eases the field instead of re-scaling/rotating
// the whole accumulated offset the way the old `speed * timeSeconds` did. Morph
// and boil stay independent absolute rates (no cross-coupling, by design).
// Precision: the accumulators grow ~speed * sessionTime, same as the old form; the
// shader's frac() wraps them. No modulo-wrap in v1 (parity) — a future robustness
// item if very long sessions show drift in the wrap.
void RtxAtmosphere::advanceCloudMotion(float dt) {
  // Guard pause / first-frame / pathological dt. <= 0 leaves the field frozen
  // exactly where it is (no jump on resume).
  if (!(dt > 0.0f)) {
    return;
  }

  constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;

  // Wind advection — drift-modulated speed/direction, integrated.
  const float windAngle = RtxOptions::cloudWindDirection() * kDegToRad;
  const float windSpeed = RtxOptions::cloudWindSpeed();  // km/s
  m_cloudAdvectOffset.x += std::cos(windAngle) * windSpeed * dt;
  m_cloudAdvectOffset.y += std::sin(windAngle) * windSpeed * dt;

  // Field-evolution morph — Y-dominant scroll through the volume (in-place
  // morphing) with the XZ remainder split diagonally for lateral decorrelation.
  const float evoSpeed = RtxOptions::cloudEvolutionSpeed();  // km/s
  const float vBias    = std::min(std::max(RtxOptions::cloudEvolutionVerticalBias(), 0.0f), 1.0f);
  const float lateral  = (1.0f - vBias) * 0.70710678f;
  m_cloudEvolutionOffset.y += vBias   * evoSpeed * dt;
  m_cloudEvolutionOffset.x += lateral * evoSpeed * dt;
  m_cloudEvolutionOffset.z += lateral * evoSpeed * dt;

  // Edge boil — single scalar phase expanded along a fixed direction in the shader.
  m_cloudBoilPhase += RtxOptions::cloudBoilSpeed() * dt;  // km/s integrated
}

void RtxAtmosphere::dispatchCloudRender(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Cloud Render (Nubis Cubed)");

  if (!m_cloudRenderRT.isValid()) {
    return;  // ensureCloudRenderRT hasn't allocated yet (first frame with zero extent)
  }

  // Refresh the AtmosphereArgs buffer so the camera basis + Nubis Cubed
  // tuning knobs land in the GPU CB before the dispatch reads them.
  AtmosphereArgs args = getAtmosphereArgs();
  ctx->updateBuffer(m_constantsBuffer, 0, sizeof(AtmosphereArgs), &args);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);

  // Linear/REPEAT sampler for the cloud noise + voxel grid taps. REPEAT
  // matches the frac()-tile-wrap convention used everywhere else in the
  // cloud math (cloudVoxelWorldToUVW and sampleCloudDensityTextured).
  DxvkSamplerCreateInfo samplerInfo = {};
  samplerInfo.magFilter    = VK_FILTER_LINEAR;
  samplerInfo.minFilter    = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  Rc<DxvkSampler> cloudSampler = m_device->createSampler(samplerInfo);

  // Linear/CLAMP sampler for the sky-view LUT + cloud-sky-transmittance LUT.
  // CLAMP is mandatory — sky-view LUT is keyed by (azimuth, elevation) and
  // REPEAT would alias the south pole onto the north.
  DxvkSamplerCreateInfo skyViewSamplerInfo = {};
  skyViewSamplerInfo.magFilter    = VK_FILTER_LINEAR;
  skyViewSamplerInfo.minFilter    = VK_FILTER_LINEAR;
  skyViewSamplerInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  skyViewSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  skyViewSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  skyViewSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  Rc<DxvkSampler> skyViewSampler = m_device->createSampler(skyViewSamplerInfo);

  // Linear/CLAMP sampler for the cloud height LUT. CLAMP because the LUT is
  // parameterized on a bounded (typeSlice, heightFrac) domain — REPEAT would
  // alias the cumulonimbus column back into stratus territory.
  Rc<DxvkSampler> heightLutSampler = m_device->createSampler(skyViewSamplerInfo);

  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_cloudNoise3D.view, nullptr);
  ctx->bindResourceSampler(2, cloudSampler);
  ctx->bindResourceView(3, m_cloudDSun.view, nullptr);
  ctx->bindResourceView(4, m_cloudDAmbient.view, nullptr);
  ctx->bindResourceView(5, m_fastNoise.getView(), nullptr);
  ctx->bindResourceView(6, m_cloudRenderRT.view, nullptr);
  ctx->bindResourceView(7, m_skyViewLut.isValid() ? m_skyViewLut.view : nullptr, nullptr);
  ctx->bindResourceView(8, m_cloudSkyTransmittanceLut.isValid() ? m_cloudSkyTransmittanceLut.view : nullptr, nullptr);
  ctx->bindResourceSampler(9, skyViewSampler);
  ctx->bindResourceView(10, m_cloudHeightLut.isValid() ? m_cloudHeightLut.view : nullptr, nullptr);
  ctx->bindResourceSampler(11, heightLutSampler);
  ctx->bindResourceView(12, m_cloudPlacementMap.view, nullptr);

  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_cloudNoise3D.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_cloudDSun.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_cloudDAmbient.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_cloudPlacementMap.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_cloudRenderRT.image);
  if (m_skyViewLut.isValid()) {
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_skyViewLut.image);
  }
  if (m_cloudSkyTransmittanceLut.isValid()) {
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_cloudSkyTransmittanceLut.image);
  }
  if (m_cloudHeightLut.isValid()) {
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_cloudHeightLut.image);
  }

  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudRenderShader::getShader());

  // Shader declares [numthreads(8, 8, 1)].
  const uint32_t groupsX = (m_cloudRenderExtent.width  + 7u) / 8u;
  const uint32_t groupsY = (m_cloudRenderExtent.height + 7u) / 8u;
  ctx->dispatch(groupsX, groupsY, 1);
}

void RtxAtmosphere::dispatchCloudSecondaryLut(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Cloud Secondary LUT");

  if (!m_cloudSecondaryLut.isValid()) {
    return;
  }

  // Refresh the args buffer so the bake sees this frame's sun / wind /
  // camera state (mirrors the other per-frame dispatch sites).
  AtmosphereArgs args = getAtmosphereArgs();
  ctx->updateBuffer(m_constantsBuffer, 0, sizeof(AtmosphereArgs), &args);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);

  // Samplers mirror dispatchCloudRender: linear/REPEAT for the noise + voxel
  // grids, linear/CLAMP for the sky-view + height LUTs.
  DxvkSamplerCreateInfo samplerInfo = {};
  samplerInfo.magFilter    = VK_FILTER_LINEAR;
  samplerInfo.minFilter    = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  Rc<DxvkSampler> cloudSampler = m_device->createSampler(samplerInfo);

  DxvkSamplerCreateInfo skyViewSamplerInfo = {};
  skyViewSamplerInfo.magFilter    = VK_FILTER_LINEAR;
  skyViewSamplerInfo.minFilter    = VK_FILTER_LINEAR;
  skyViewSamplerInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  skyViewSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  skyViewSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  skyViewSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  Rc<DxvkSampler> skyViewSampler   = m_device->createSampler(skyViewSamplerInfo);
  Rc<DxvkSampler> heightLutSampler = m_device->createSampler(skyViewSamplerInfo);

  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_cloudNoise3D.view, nullptr);
  ctx->bindResourceSampler(2, cloudSampler);
  ctx->bindResourceView(3, m_cloudDSun.view, nullptr);
  ctx->bindResourceView(4, m_cloudDAmbient.view, nullptr);
  ctx->bindResourceView(5, m_fastNoise.getView(), nullptr);
  ctx->bindResourceView(6, m_cloudSecondaryLut.views[0], nullptr);  // mip 0 storage write
  ctx->bindResourceView(7, m_skyViewLut.isValid() ? m_skyViewLut.view : nullptr, nullptr);
  ctx->bindResourceView(8, m_cloudSkyTransmittanceLut.isValid() ? m_cloudSkyTransmittanceLut.view : nullptr, nullptr);
  ctx->bindResourceSampler(9, skyViewSampler);
  ctx->bindResourceView(10, m_cloudHeightLut.isValid() ? m_cloudHeightLut.view : nullptr, nullptr);
  ctx->bindResourceSampler(11, heightLutSampler);
  ctx->bindResourceView(12, m_cloudPlacementMap.view, nullptr);

  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_cloudNoise3D.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_cloudDSun.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_cloudDAmbient.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_cloudPlacementMap.image);
  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_cloudSecondaryLut.image);
  if (m_skyViewLut.isValid()) {
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_skyViewLut.image);
  }
  if (m_cloudSkyTransmittanceLut.isValid()) {
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_cloudSkyTransmittanceLut.image);
  }
  if (m_cloudHeightLut.isValid()) {
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_cloudHeightLut.image);
  }

  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudSecondaryLutShader::getShader());

  // Shader declares [numthreads(8, 8, 1)].
  const uint32_t groupsX = (kCloudSecondaryLutWidth  + 7u) / 8u;
  const uint32_t groupsY = (kCloudSecondaryLutHeight + 7u) / 8u;
  ctx->dispatch(groupsX, groupsY, 1);

  // Blur mip 0 down the chain so the sky<-clouds bleed can sample a coarse
  // (wide-blurred) level (fork — 2026-06-19). Barrier mip-0 write -> mip-gen
  // read first; updateMipmap needs an RtxContext (ctx is always one here —
  // computeLuts is called with the RtxContext, see rtx_fork_atmosphere.cpp).
  ctx->emitMemoryBarrier(0,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
  {
    ScopedGpuProfileZone(ctx, "Atmosphere Cloud Secondary LUT Mipmap");
    Rc<RtxContext> rtxCtx = static_cast<RtxContext*>(ctx.ptr());
    RtxMipmap::updateMipmap(rtxCtx, m_cloudSecondaryLut, MipmapMethod::Gaussian);
  }
}

void RtxAtmosphere::dispatchCloudNoise3DBake(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Cloud Noise 3D Bake");

  // Baked at atmosphere init and re-baked whenever a bake input changes (the
  // needsCloudNoiseRebake() gate in computeLuts). Runs the 3D Perlin FBM stack
  // defined in rtx_cloud_noise_baker.comp.slang and writes 256-cubed voxels of
  // R8 density. Mirrors dispatchSkyViewLut's structure but uses a 3D dispatch.

  // Update atmosphere args buffer
  AtmosphereArgs args = getAtmosphereArgs();
  ctx->updateBuffer(m_constantsBuffer, 0, sizeof(AtmosphereArgs), &args);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);

  // Bind resources: ConstantBuffer<AtmosphereArgs> at slot 0, RWTexture3D at slot 1.
  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_cloudNoise3D.view, nullptr);

  // Track resources
  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_cloudNoise3D.image);

  // Bind shader and dispatch
  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudNoiseBakerShader::getShader());

  // Dispatch: kCloudNoise3DSize / 8 = 32 groups per axis (shader uses [numthreads(8,8,8)])
  const uint32_t groupCount = kCloudNoise3DSize / 8u;
  ctx->dispatch(groupCount, groupCount, groupCount);
}

void RtxAtmosphere::dispatchCloudHeightLutBake(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Cloud Height LUT Bake");

  // Baked once at atmosphere init (see computeLuts). Procedurally fills the
  // 64x128 RGBA8 LUT with the single-lobe per-cloud height envelope (R), the
  // coverage-threshold scale (G), and the cumulative envelope integral (B)
  // consumed by the column model in cloud_render.comp.slang / atmosphere_common.
  //
  // The baker takes the args CB at slot 0 (for cloud type/shape params) and
  // writes the output RWTexture2D at slot 1. cloud_height_lut_baker.comp.slang
  // declares `[numthreads(8, 8, 1)]`, matching the dispatch dimensions below.
  AtmosphereArgs args = getAtmosphereArgs();
  ctx->updateBuffer(m_constantsBuffer, 0, sizeof(AtmosphereArgs), &args);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);

  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_cloudHeightLut.view, nullptr);
  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_cloudHeightLut.image);

  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudHeightLutBakerShader::getShader());

  const uint32_t groupsX = (kCloudHeightLutWidth  + 7u) / 8u;
  const uint32_t groupsY = (kCloudHeightLutHeight + 7u) / 8u;
  ctx->dispatch(groupsX, groupsY, 1);
}

void RtxAtmosphere::dispatchCloudPlacementMapBake(Rc<DxvkContext> ctx) {
  ScopedGpuProfileZone(ctx, "Atmosphere Cloud Placement Map Bake");

  // Baked at atmosphere init + re-baked when a bake input changes (the
  // needsCloudPlacementRebake() gate in computeLuts). Fills the 512x512
  // RGBA8 placement map with the cluster / top-jitter / base-lift fields
  // defined in cloud_placement_map_baker.comp.slang.
  AtmosphereArgs args = getAtmosphereArgs();
  ctx->updateBuffer(m_constantsBuffer, 0, sizeof(AtmosphereArgs), &args);
  ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_constantsBuffer);

  ctx->bindResourceBuffer(0, DxvkBufferSlice(m_constantsBuffer, 0, m_constantsBuffer->info().size));
  ctx->bindResourceView(1, m_cloudPlacementMap.view, nullptr);
  ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_cloudPlacementMap.image);

  ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, CloudPlacementMapBakerShader::getShader());

  // Shader declares [numthreads(8, 8, 1)].
  const uint32_t groupCount = (kCloudPlacementMapSize + 7u) / 8u;
  ctx->dispatch(groupCount, groupCount, 1);
}

void RtxAtmosphere::onFrameAdvanceForCloudHistory(uint32_t currentFrameId) {
  if (currentFrameId == m_cloudHistoryLastFrameId) {
    return;  // already advanced this frame
  }
  // Don't swap on the very first observation — leaves swap = false so the
  // initial frame writes to slot 0 and reads slot 1 (uninitialized -> zero ->
  // disocclusion fallback). Subsequent frames toggle.
  if (m_cloudHistoryLastFrameId != UINT32_MAX) {
    m_cloudHistorySwap = !m_cloudHistorySwap;
  }
  m_cloudHistoryLastFrameId = currentFrameId;
}

void RtxAtmosphere::ensureCloudHistoryResources(Rc<DxvkContext> ctx, const VkExtent3D& downscaledExtent) {
  // Bail on degenerate extents (can happen during early frames before resize
  // events have settled) — we'll allocate on a later frame.
  if (downscaledExtent.width == 0u || downscaledExtent.height == 0u) {
    return;
  }

  const bool extentsMatch = (m_cloudHistoryExtent.width == downscaledExtent.width)
                         && (m_cloudHistoryExtent.height == downscaledExtent.height);
  if (extentsMatch && m_cloudHistory[0].isValid() && m_cloudHistory[1].isValid()) {
    return;
  }

  // (Re)create both ping-pong slices at the requested screen extent.
  // RGBA16F: rgb = premultiplied cloud radiance, a = cloud alpha. STORAGE bit
  // for the RW write path; the read path uses the same view as a sampled image.
  const VkExtent3D extent = { downscaledExtent.width, downscaledExtent.height, 1u };
  for (uint32_t i = 0u; i < 2u; ++i) {
    const char* names[2] = {
      "Atmosphere Cloud History 0",
      "Atmosphere Cloud History 1",
    };
    m_cloudHistory[i] = Resources::createImageResource(
      ctx,
      names[i],
      extent,
      VK_FORMAT_R16G16B16A16_SFLOAT,
      1, // numLayers
      VK_IMAGE_TYPE_2D,
      VK_IMAGE_VIEW_TYPE_2D,
      0, // imageCreateFlags
      VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags
      VkClearColorValue{}, // clearValue (zero -- treated as "no history" by shader disocclusion guard)
      1 // mipLevels
    );
  }

  // R16_UINT companion ping-pong (fork — 2026-05-13). Holds the frame index
  // (mod 0x10000) at which each pixel of the color ping-pong was last
  // refreshed by the sky-miss path. Cleared to 0xFFFF "never written" so the
  // shader's age check rejects history at pixels that have never been
  // written by the smoother (including foreground-occluded ones whose color
  // slot retains pre-occlusion radiance). Drives the disocclusion fix for
  // the bright-trail ghosting under the 2026-05-13 Nubis Cubed work — see
  // atmosphere_sky.slangh's age-channel comment block for the mechanism.
  VkClearColorValue frameIdClearValue{};
  frameIdClearValue.uint32[0] = 0xFFFFu;
  for (uint32_t i = 0u; i < 2u; ++i) {
    const char* frameIdNames[2] = {
      "Atmosphere Cloud History Frame ID 0",
      "Atmosphere Cloud History Frame ID 1",
    };
    m_cloudHistoryFrameId[i] = Resources::createImageResource(
      ctx,
      frameIdNames[i],
      extent,
      VK_FORMAT_R16_UINT,
      1, // numLayers
      VK_IMAGE_TYPE_2D,
      VK_IMAGE_VIEW_TYPE_2D,
      0, // imageCreateFlags
      VK_IMAGE_USAGE_STORAGE_BIT, // extraUsageFlags
      frameIdClearValue,
      1 // mipLevels
    );
  }

  m_cloudHistoryExtent = extent;
}

void RtxAtmosphere::bindResources(Rc<DxvkContext> ctx, VkPipelineBindPoint pipelineBindPoint) {
  // Bind atmosphere LUT resources to the pipeline.
  // Note: The active call site for runtime binding is bindAtmosphereLuts in
  // rtx_fork_atmosphere.cpp; this method is available for direct use if needed.
  if (m_transmittanceLut.isValid()) {
    ctx->bindResourceView(BINDING_ATMOSPHERE_TRANSMITTANCE_LUT, m_transmittanceLut.view, nullptr);
  }
  if (m_multiscatteringLut.isValid()) {
    ctx->bindResourceView(BINDING_ATMOSPHERE_MULTISCATTERING_LUT, m_multiscatteringLut.view, nullptr);
  }
  if (m_skyViewLut.isValid()) {
    ctx->bindResourceView(BINDING_ATMOSPHERE_SKY_VIEW_LUT, m_skyViewLut.view, nullptr);
  }
  if (m_cloudNoise3D.isValid()) {
    ctx->bindResourceView(BINDING_ATMOSPHERE_CLOUD_NOISE_3D, m_cloudNoise3D.view, nullptr);
  }
  if (m_fastNoise.isValid()) {
    ctx->bindResourceView(BINDING_ATMOSPHERE_FAST_NOISE, m_fastNoise.getView(), nullptr);
  }
  if (m_cloudSkyTransmittanceLut.isValid()) {
    ctx->bindResourceView(BINDING_ATMOSPHERE_CLOUD_SKY_TRANSMITTANCE_LUT, m_cloudSkyTransmittanceLut.view, nullptr);
  }
  if (m_cloudDSun.isValid()) {
    ctx->bindResourceView(BINDING_ATMOSPHERE_CLOUD_D_SUN, m_cloudDSun.view, nullptr);
  }
  if (m_cloudDAmbient.isValid()) {
    ctx->bindResourceView(BINDING_ATMOSPHERE_CLOUD_D_AMBIENT, m_cloudDAmbient.view, nullptr);
  }
  if (m_cloudRenderRT.isValid()) {
    ctx->bindResourceView(BINDING_ATMOSPHERE_CLOUD_RENDER_RT, m_cloudRenderRT.view, nullptr);
  }
  if (m_cloudSecondaryLut.isValid()) {
    ctx->bindResourceView(BINDING_ATMOSPHERE_CLOUD_SECONDARY_LUT, m_cloudSecondaryLut.view, nullptr);
  }
  if (m_cloudPlacementMap.isValid()) {
    ctx->bindResourceView(BINDING_ATMOSPHERE_CLOUD_PLACEMENT_MAP, m_cloudPlacementMap.view, nullptr);
  }
  // Cloud history bindings are wired in fork_hooks::bindAtmosphereLuts (the
  // active call site) and depend on the downscaled-extent ensure step. Left
  // unbound here to keep this method's contract minimal.
}

} // namespace dxvk
