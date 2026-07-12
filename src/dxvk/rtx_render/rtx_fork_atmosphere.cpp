// src/dxvk/rtx_render/rtx_fork_atmosphere.cpp
//
// Fork-owned file. Contains the implementations of fork_hooks:: functions
// for the RtxAtmosphere subsystem (Hillaire physically-based sky), lifted
// from rtx_context.cpp during the 2026-04-18 fork touchpoint-pattern refactor.
//
// See docs/fork-touchpoints.md for the full fork-hooks catalogue.
//
// NOTE: initAtmosphere, updateAtmosphereConstants, and bindAtmosphereLuts
// access private members of RtxContext (m_atmosphere, m_lastSkyMode,
// m_skyColorFormat, m_skyRtColorFormat, m_device).  This file requires
// that RtxContext declare each hook as a friend — see rtx_context.h.
// injectRtxAtmosphereSkySkip accesses only the public RtxOptions API and
// therefore does not require a friend declaration.

#include "rtx_fork_hooks.h"
#include "rtx_context.h"
#include "rtx_atmosphere.h"
#include "rtx_scene_manager.h"       // getLightManager (directional sun/moon injection)
#include "rtx_light_manager.h"       // createExternallyTrackedLight / updateExternallyTrackedLight
#include "rtx_lights.h"              // RtDistantLight, RtLight
#include "rtx_options.h"
#include "rtx/pass/raytrace_args.h"
#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/atmosphere/atmosphere_args.h" // MAX_MOONS (showAtmosphereUI moon loop)
#include "../util/util_global_time.h" // GlobalTime::get().deltaTime (cloud-motion integrator)
#include "imgui/imgui.h"              // ImGui::Button, ImGui::Text, etc. (showAtmosphereUI)
#include "rtx_imgui.h"                // RemixGui::DragFloat, ComboWithKey (showAtmosphereUI)
#include <cstdio>                     // std::snprintf (renderMoonUI label)
#include <cmath>                      // std::tan (cloud render camera basis)
#include <algorithm>                  // std::max / std::min (renderChromaticityWidget)
#include <unordered_map>              // per-widget cached chromaticity state

namespace dxvk {
namespace fork_hooks {

  // ===========================================================================
  // Sun + moon as real Remix distant lights (fork — 2026-06-21)
  //
  // In physical-atmosphere mode the sun (and each enabled moon) is injected as
  // an externally-tracked RtDistantLight driven by the atmosphere model — the
  // sole sun/moon path in Numos. They flow through the standard NEE/RTXDI path,
  // so SSS / decals / viewmodels are handled by the unified pipeline. The
  // radiance is the CPU port of the atmosphere sun/moon sample divided by pi: a
  // distant light contributes radiance/sin^2(halfAngle) * coneSolidAngle ~=
  // pi*radiance of effective irradiance. Cloud-on-terrain shadows are folded
  // per-pixel onto the real sun in the NEE (integrator_direct.slangh). The
  // older bespoke evalAtmosphereSunNEE/MoonNEE path was removed 2026-06-21.
  // ===========================================================================
  namespace {
    constexpr float kFhPi = 3.14159265358979323846f;

    inline float fhSmoothstep(float e0, float e1, float x) {
      const float denom = e1 - e0;
      float t = (denom != 0.0f) ? (x - e0) / denom : 0.0f;
      t = std::min(std::max(t, 0.0f), 1.0f);
      return t * t * (3.0f - 2.0f * t);
    }

    inline Vector3 fhMul(const Vector3& a, const Vector3& b) {
      return Vector3(a.x * b.x, a.y * b.y, a.z * b.z);
    }

    // Port of getAtmosphericTransmittanceForDir (atmosphere_common.slangh): the
    // closed-form Kasten-Young air-mass extinction the sun/moon/cloud paths use.
    // dirYUp must be normalized, Y-up. ozoneDensity at the ozone layer altitude
    // is exactly 1.0, so the ozone path length collapses to airMass.
    Vector3 fhAtmTransmittanceYUp(const AtmosphereArgs& a, const Vector3& dirYUp) {
      const float H = a.rayleighScaleHeight;
      const float zc = dirYUp.y;  // zenith cosine
      float airMass;
      if (zc > 0.01f) {
        const float zenithRad = std::acos(std::min(std::max(zc, -1.0f), 1.0f));
        const float zenithDeg = zenithRad * (180.0f / kFhPi);
        airMass = 1.0f / (zc + 0.15f * std::pow(93.885f - zenithDeg, -1.253f));
      } else {
        airMass = 40.0f * std::exp(-zc * 10.0f);
      }
      airMass = std::min(airMass, 200.0f);
      const float rayleighOD = H * airMass;
      const float mieOD = a.mieScaleHeight * airMass;
      const float ozonePath = airMass;  // ozoneDensity(layerAltitude) == 1
      Vector3 t(
        std::exp(-(a.rayleighScattering.x * rayleighOD + a.mieScattering.x * mieOD + a.ozoneAbsorption.x * ozonePath * 0.15f)),
        std::exp(-(a.rayleighScattering.y * rayleighOD + a.mieScattering.y * mieOD + a.ozoneAbsorption.y * ozonePath * 0.15f)),
        std::exp(-(a.rayleighScattering.z * rayleighOD + a.mieScattering.z * mieOD + a.ozoneAbsorption.z * ozonePath * 0.15f)));
      if (zc < 0.0f) {
        const float f = std::exp(-(-zc) * 15.0f);  // twilight fade
        t = Vector3(t.x * f, t.y * f, t.z * f);
      }
      return t;
    }

    // Persistent externally-tracked light handles. Kept alive across frames;
    // radiance goes to 0 when a body is below the horizon / disabled (inert,
    // no create/destroy churn). Moons are created lazily on first use.
    struct AtmosphereDistantLightState {
      RtLight* sun = nullptr;
      RtLight* moons[MAX_MOONS] = {};
    };
    AtmosphereDistantLightState g_atmoLights;

    void fhDropAtmosphereLights() {
      if (g_atmoLights.sun) {
        g_atmoLights.sun->markForGarbageCollection();
        g_atmoLights.sun = nullptr;
      }
      for (uint32_t i = 0; i < MAX_MOONS; ++i) {
        if (g_atmoLights.moons[i]) {
          g_atmoLights.moons[i]->markForGarbageCollection();
          g_atmoLights.moons[i] = nullptr;
        }
      }
    }

    void fhSyncAtmosphereDistantLights(RtxContext& ctx, const AtmosphereArgs& args) {
      // Mode gate. Sun/moon distant lights are the sole atmosphere sun path in
      // Numos; drop any previously-injected lights when not in Numos.
      if (RtxOptions::skyMode() != SkyMode::Numos) {
        fhDropAtmosphereLights();
        return;
      }

      LightManager& lm = ctx.getSceneManager().getLightManager();
      const bool isZUp = RtxOptions::zUp();
      const float radScale = RtxOptions::directionalLightRadianceScale();
      constexpr float kMinHalfAngle = 0.0005f;  // avoid sin(halfAngle)==0 in distantLightSampleArea

      auto toWorld = [isZUp](const Vector3& yup) -> Vector3 {
        return isZUp ? Vector3(yup.x, yup.z, yup.y) : yup;  // Y-up -> Z-up swap
      };

      // m_direction is the propagation direction (toward the ground) = -toBody.
      auto ensureLight = [&](RtLight*& slot, const Vector3& propDir, float halfAngle, const Vector3& radiance, bool cloudShadowed) {
        const Vector3 clamped(std::max(radiance.x, 0.0f), std::max(radiance.y, 0.0f), std::max(radiance.z, 0.0f));
        auto dl = RtDistantLight::tryCreate(propDir, std::max(halfAngle, kMinHalfAngle), clamped);
        if (!dl) {
          return;
        }
        RtLight rtl(*dl);
        // Mark dynamic so updateLightStaticSleep applies *light = newLight every
        // frame. Without this the light is treated as static and put to sleep
        // after getNumFramesToPutLightsToSleep() frames — which froze the sun's
        // direction (it stopped tracking sunRotation/sunElevation).
        rtl.isDynamic = true;
        // When set, the NEE folds the per-pixel cloud-on-terrain transmittance
        // onto this light's contribution (distant-light GPU flags bit 2).
        rtl.atmosphereCloudShadowed = cloudShadowed;
        if (slot == nullptr) {
          slot = lm.createExternallyTrackedLight(rtl);
        } else {
          lm.updateExternallyTrackedLight(slot, rtl);
        }
      };

      // ---- Sun (always present in Numos; radiance 0 below horizon) ----
      {
        const Vector3 sunDirYUp(args.sunDirection.x, args.sunDirection.y, args.sunDirection.z);
        Vector3 radiance(0.0f, 0.0f, 0.0f);
        if (sunDirYUp.y > 0.0f) {
          const float mieModulation = 0.3f + 1.7f * args.mieAnisotropy;         // mix(0.3, 2.0, g)
          const float sunVisibility = 0.05f + 0.95f * fhSmoothstep(0.0f, 0.8f, args.mieAnisotropy);
          const Vector3 T = fhAtmTransmittanceYUp(args, sunDirYUp);
          const Vector3 sunIll(args.sunIlluminance.x, args.sunIlluminance.y, args.sunIlluminance.z);
          const Vector3 sample = fhMul(sunIll, T) * (mieModulation * sunVisibility * args.sunRayBrightness * 0.5f);
          radiance = sample * (radScale / kFhPi);
        }
        // Half-angle: physical sun disc radius (sunSize/2) by default, or the
        // decoupled sunShadowSoftnessDeg override (>0) so shadows can be softened
        // without enlarging the visible sun disc. Half-angle does not affect
        // brightness (contribution ~= pi * m_radiance).
        const float softnessDeg = RtxOptions::sunShadowSoftnessDeg();
        const float sunHalfAngle = (softnessDeg > 0.0f) ? (softnessDeg * (kFhPi / 180.0f))
                                                        : args.sunAngularRadius;
        const Vector3 toSun = toWorld(sunDirYUp);
        const Vector3 propDir = (sunDirYUp.y > 0.0f) ? Vector3(-toSun.x, -toSun.y, -toSun.z)
                                                     : Vector3(0.0f, -1.0f, 0.0f);
        ensureLight(g_atmoLights.sun, propDir, sunHalfAngle, radiance, /*cloudShadowed=*/true);
      }

      // ---- Moons (lazily created; mirror sampleAtmosphereMoonLight radiance) ----
      const float moonNee = args.moonNeeStrength;
      const float surfMoon = args.surfaceMoonBrightness;
      const float nightFactor = fhSmoothstep(0.02f, -0.05f, args.sunDirection.y);
      for (uint32_t i = 0; i < MAX_MOONS; ++i) {
        const MoonParams& m = args.moons[i];
        const Vector3 dirRaw(m.direction.x, m.direction.y, m.direction.z);
        const float len = std::sqrt(dirRaw.x * dirRaw.x + dirRaw.y * dirRaw.y + dirRaw.z * dirRaw.z);
        const bool lit = (m.enabled >= 0.5f) && (moonNee > 0.0f) && (nightFactor > 0.001f) && (len > 1e-4f);

        // Skip moons that have never been lit (avoid creating unused light slots).
        if (!lit && g_atmoLights.moons[i] == nullptr) {
          continue;
        }

        const Vector3 dirN = (len > 1e-4f) ? Vector3(dirRaw.x / len, dirRaw.y / len, dirRaw.z / len)
                                           : Vector3(0.0f, 1.0f, 0.0f);
        Vector3 radiance(0.0f, 0.0f, 0.0f);
        if (lit) {
          const Vector3 T = fhAtmTransmittanceYUp(args, dirN);  // ~0 below horizon (twilight fade)
          const Vector3 sunIll(args.sunIlluminance.x, args.sunIlluminance.y, args.sunIlluminance.z);
          const Vector3 color(m.color.x, m.color.y, m.color.z);
          const Vector3 sharedFactor = fhMul(fhMul(sunIll, color), T) * (m.brightness / kFhPi);
          const float phaseGlow = 0.5f - 0.5f * std::cos(m.phase * 2.0f * kFhPi);
          const float moonSolidAngleSr = 2.0f * kFhPi * (1.0f - std::cos(m.angularRadius));
          const Vector3 sample = sharedFactor * (phaseGlow * moonSolidAngleSr * moonNee * surfMoon * nightFactor);
          radiance = sample * (radScale / kFhPi);
        }
        const Vector3 toMoon = toWorld(dirN);
        const Vector3 propDir = lit ? Vector3(-toMoon.x, -toMoon.y, -toMoon.z) : Vector3(0.0f, -1.0f, 0.0f);
        // Half-angle = the moon's physical angular radius (same as the sun).
        ensureLight(g_atmoLights.moons[i], propDir, m.angularRadius, radiance, /*cloudShadowed=*/false);
      }
    }
  }  // anonymous namespace

  // ---------------------------------------------------------------------------
  // initAtmosphere
  //
  // Constructs the RtxAtmosphere object during RtxContext initialization.
  // Called from the RtxContext constructor after GlobalTime::get().init().
  //
  // ACCESS NOTE: reads m_device (private Rc<DxvkDevice>) and writes
  // m_atmosphere (private unique_ptr<RtxAtmosphere>). Friend declaration
  // required in RtxContext.
  // ---------------------------------------------------------------------------
  void initAtmosphere(RtxContext& ctx) {
    ctx.m_atmosphere = std::make_unique<RtxAtmosphere>(ctx.m_device.ptr());
  }

  // ---------------------------------------------------------------------------
  // updateAtmosphereConstants
  //
  // Sets constants.skyMode, detects sky-mode transitions (clearing rasterized
  // skybox buffers when switching to Numos), and when Numos
  // is active ensures the atmosphere object exists, calls
  // initialize/computeLuts, and writes atmosphereArgs into the constant block.
  //
  // Called from RtxContext::updateRaytraceArgsConstantBuffer immediately after
  // constants.skyBrightness is set.
  //
  // ACCESS NOTE: reads/writes m_atmosphere, m_lastSkyMode, m_skyColorFormat,
  // m_skyRtColorFormat, and m_device (all private). Friend declaration required
  // in RtxContext.
  // ---------------------------------------------------------------------------
  void updateAtmosphereConstants(RtxContext& ctx, RaytraceArgs& constants) {
    constants.skyMode = static_cast<uint32_t>(RtxOptions::skyMode());

    // Detect sky mode change and clear sky buffers when switching to Numos
    SkyMode currentSkyMode = RtxOptions::skyMode();
    if (currentSkyMode != ctx.m_lastSkyMode) {
      if (currentSkyMode == SkyMode::Numos) {
        // Clear the rasterized skybox buffers when switching to physical atmosphere
        auto skyProbe = ctx.getResourceManager().getSkyProbe(&ctx, ctx.m_skyColorFormat);
        auto skyMatte = ctx.getResourceManager().getSkyMatte(&ctx, ctx.m_skyRtColorFormat);

        VkClearValue clearValue = {};
        clearValue.color.float32[0] = 0.0f;
        clearValue.color.float32[1] = 0.0f;
        clearValue.color.float32[2] = 0.0f;
        clearValue.color.float32[3] = 0.0f;

        if (skyProbe.view != nullptr) {
          ctx.DxvkContext::clearRenderTarget(skyProbe.view, VK_IMAGE_ASPECT_COLOR_BIT, clearValue);
        }
        if (skyMatte.view != nullptr) {
          ctx.DxvkContext::clearRenderTarget(skyMatte.view, VK_IMAGE_ASPECT_COLOR_BIT, clearValue);
        }
      }
      ctx.m_lastSkyMode = currentSkyMode;
    }

    // Update atmosphere parameters
    if (RtxOptions::skyMode() == SkyMode::Numos) {
      if (!ctx.m_atmosphere) {
        ctx.m_atmosphere = std::make_unique<RtxAtmosphere>(ctx.m_device.ptr());
      }
      ctx.m_atmosphere->initialize(&ctx);

      // Unified cloud-motion integrator (fork — 2026-06-21). Advance the wind /
      // morph / boil accumulators exactly once per frame, before getAtmosphereArgs
      // (called many times per frame) reads them. dt comes from the same GlobalTime
      // clock the weather blender uses, so the drift-modulated wind it reads is
      // consistent with the parameters the blender wrote this frame.
      ctx.m_atmosphere->advanceCloudMotion(GlobalTime::get().deltaTime());

      // Cloud render compute pass setup (Nubis Cubed 2023, fork — 2026-05-12, C4).
      // Push the per-frame camera basis and ensure the screen-space RT is
      // allocated at the downscale extent BEFORE computeLuts dispatches the
      // cloud render compute. The basis vectors are in Y-up world space (cloud
      // math convention, camera at origin) and the Right/Up vectors are
      // pre-scaled by tan(halfFovX/Y) + aspect ratio so the shader does just
      // a weighted sum to reconstruct viewDir per pixel.
      {
        const RtCamera& camera = ctx.getSceneManager().getCamera();
        const Vector3 forward = camera.getDirection(/*freecam=*/true);
        const Vector3 right   = camera.getRight(/*freecam=*/true);
        const Vector3 up      = camera.getUp(/*freecam=*/true);

        const bool isZUp = RtxOptions::zUp();
        // Swap (x, y, z) -> (x, z, y) when the game is Z-up. Mirrors the
        // existing isZUp swap inside `evalSkyRadiance` in atmosphere_sky.slangh.
        auto toYUp = [isZUp](const Vector3& v) -> Vector3 {
          if (isZUp) {
            return Vector3(v.x, v.z, v.y);
          }
          return v;
        };

        const Vector3 forwardYUp = toYUp(forward);
        const Vector3 rightYUp   = toYUp(right);
        const Vector3 upYUp      = toYUp(up);

        // tan(halfFovY) and aspect. halfFov is fov/2 (RtCamera::getFov() is
        // the full vertical FOV). Pre-scale the basis vectors so the shader
        // simply does forward + ndc.x*right + ndc.y*up.
        const float fovYRad = camera.getFov();
        const float halfFovY = 0.5f * fovYRad;
        const float tanHalfFovY = std::tan(halfFovY);
        const float aspect = camera.getAspectRatio();
        const float tanHalfFovX = tanHalfFovY * aspect;

        const Vector3 rightScaled = rightYUp * tanHalfFovX;
        const Vector3 upScaled    = upYUp    * tanHalfFovY;

        const uint32_t frameIdx = static_cast<uint32_t>(ctx.m_device->getCurrentFrameId());
        ctx.m_atmosphere->setCloudRenderCameraBasis(forwardYUp, rightScaled, upScaled, frameIdx);

        // Push the camera world position (Y-up km) for the C6 voxel-grid
        // cloud-on-terrain shadow plumbing. The G-buffer worldPos that the
        // helper consumes is in engine game units; the helper converts to
        // km internally via worldUnitsPerKm. We do the matching conversion
        // here CPU-side: km = gameUnits / worldUnitsPerKm. The isZUp swap
        // mirrors the basis-vector swap above so the helper's camera-relative
        // subtraction lands in the right frame.
        {
          const Vector3 cameraPosWorldUnits = camera.getPosition(/*freecam=*/false);
          const Vector3 cameraPosWorldUnitsYUp = toYUp(cameraPosWorldUnits);
          const float sceneScaleSafe = std::max(RtxOptions::sceneScale(), 1e-5f);
          const float worldUnitsPerKm = 100000.0f * sceneScaleSafe;
          const float kmPerWorldUnit = 1.0f / worldUnitsPerKm;
          const Vector3 cameraPosYUpKm = cameraPosWorldUnitsYUp * kmPerWorldUnit;
          ctx.m_atmosphere->setCloudShadowCameraPosition(cameraPosYUpKm);
        }

        // Allocate the cloud render RT at the downscale extent (the resolution
        // the geometry resolver raygen writes to and DLSS sees as its input).
        const VkExtent3D downscaledExtent3D = ctx.getResourceManager().getDownscaleDimensions();
        const VkExtent2D downscaleExtent = { downscaledExtent3D.width, downscaledExtent3D.height };
        ctx.m_atmosphere->ensureCloudRenderRT(&ctx, downscaleExtent);
      }

      ctx.m_atmosphere->computeLuts(&ctx);
      constants.atmosphereArgs = ctx.m_atmosphere->getAtmosphereArgs();
    }

    // Inject / update (or drop) the sun + moon distant lights. Called
    // unconditionally — the helper internally gates on skyMode and drops its
    // lights when not in Numos. Uses the atmosphere args
    // just written above; when not in Numos those are stale but unread (the
    // helper early-outs before touching them). One-frame latency vs the light
    // manager's prepareSceneData linearization is acceptable (the sun moves
    // slowly); steady state the light is always present.
    fhSyncAtmosphereDistantLights(ctx, constants.atmosphereArgs);
  }

  // ---------------------------------------------------------------------------
  // bindAtmosphereLuts
  //
  // Ensures the RtxAtmosphere object exists and is initialized (it is
  // idempotent), then binds the three atmosphere LUT textures at their
  // declared shader binding slots.  Called unconditionally because the LUT
  // slots are declared in common_bindings.slangh for all passes.
  //
  // ACCESS NOTE: reads/writes m_atmosphere and m_device (both private).
  // Friend declaration required in RtxContext.
  // ---------------------------------------------------------------------------
  void bindAtmosphereLuts(RtxContext& ctx) {
    // Bind atmosphere LUTs - must always bind since they're declared in common_bindings.slangh
    // Initialize atmosphere if not already done (needed for dummy resources)
    if (!ctx.m_atmosphere) {
      ctx.m_atmosphere = std::make_unique<RtxAtmosphere>(ctx.m_device.ptr());
    }
    // Always call initialize - it's idempotent (has internal m_initialized check)
    ctx.m_atmosphere->initialize(&ctx);

    auto transmittanceLut         = ctx.m_atmosphere->getTransmittanceLut();
    auto multiscatteringLut       = ctx.m_atmosphere->getMultiscatteringLut();
    auto skyViewLut               = ctx.m_atmosphere->getSkyViewLut();
    auto cloudNoise3D             = ctx.m_atmosphere->getCloudNoise3D();  // Stage C
    auto fastNoiseView            = ctx.m_atmosphere->getFastNoiseView();  // EA importance-sampled FAST noise
    auto cloudSkyTransmittanceLut = ctx.m_atmosphere->getCloudSkyTransmittanceLut();  // Fork: per-frame cloud occlusion of sky-ambient
    auto cloudDSun                = ctx.m_atmosphere->getCloudDSun();      // Fork: Nubis Cubed sun-direction optical depth grid
    auto cloudDAmbient            = ctx.m_atmosphere->getCloudDAmbient();  // Fork: Nubis Cubed zenith optical depth grid
    auto cloudRenderRT            = ctx.m_atmosphere->getCloudRenderRT();  // Fork: Nubis Cubed screen-space cloud render (C4)
    auto cloudSecondaryLut        = ctx.m_atmosphere->getCloudSecondaryLut();  // Fork: secondary-ray cloud dome LUT (perf, 2026-06-10)
    auto cloudPlacementMap        = ctx.m_atmosphere->getCloudPlacementMap();  // Fork: per-column cloud placement map (2026-06-11)

    // Always bind the LUTs (they're declared in shaders unconditionally)
    if (transmittanceLut.isValid()) {
      ctx.bindResourceView(BINDING_ATMOSPHERE_TRANSMITTANCE_LUT, transmittanceLut.view, nullptr);
    }
    if (multiscatteringLut.isValid()) {
      ctx.bindResourceView(BINDING_ATMOSPHERE_MULTISCATTERING_LUT, multiscatteringLut.view, nullptr);
    }
    if (skyViewLut.isValid()) {
      ctx.bindResourceView(BINDING_ATMOSPHERE_SKY_VIEW_LUT, skyViewLut.view, nullptr);
    }
    if (cloudNoise3D.isValid()) {
      ctx.bindResourceView(BINDING_ATMOSPHERE_CLOUD_NOISE_3D, cloudNoise3D.view, nullptr);
    }
    if (fastNoiseView != nullptr) {
      ctx.bindResourceView(BINDING_ATMOSPHERE_FAST_NOISE, fastNoiseView, nullptr);
    }
    if (cloudSkyTransmittanceLut.isValid()) {
      ctx.bindResourceView(BINDING_ATMOSPHERE_CLOUD_SKY_TRANSMITTANCE_LUT, cloudSkyTransmittanceLut.view, nullptr);
    }
    if (cloudDSun.isValid()) {
      ctx.bindResourceView(BINDING_ATMOSPHERE_CLOUD_D_SUN, cloudDSun.view, nullptr);
    }
    if (cloudDAmbient.isValid()) {
      ctx.bindResourceView(BINDING_ATMOSPHERE_CLOUD_D_AMBIENT, cloudDAmbient.view, nullptr);
    }
    if (cloudRenderRT.isValid()) {
      ctx.bindResourceView(BINDING_ATMOSPHERE_CLOUD_RENDER_RT, cloudRenderRT.view, nullptr);
    }
    if (cloudSecondaryLut.isValid()) {
      ctx.bindResourceView(BINDING_ATMOSPHERE_CLOUD_SECONDARY_LUT, cloudSecondaryLut.view, nullptr);
    }
    if (cloudPlacementMap.isValid()) {
      ctx.bindResourceView(BINDING_ATMOSPHERE_CLOUD_PLACEMENT_MAP, cloudPlacementMap.view, nullptr);
    }

    // Cloud history (fork). Allocate at the current downscaled render extent
    // (where the geometry resolver raygen writes the per-pixel sky radiance),
    // advance the ping-pong index once per frame, then bind PREV (read) and
    // CURR (write) at their respective slots. Both slots are declared in
    // common_bindings.slangh and so must always be bound for any pass to
    // compile/dispatch — on the first frame, both slices are zero-cleared
    // and the shader's disocclusion guard treats history as invalid.
    {
      ctx.m_atmosphere->onFrameAdvanceForCloudHistory(
        static_cast<uint32_t>(ctx.m_device->getCurrentFrameId()));

      const VkExtent3D downscaledExtent = ctx.getResourceManager().getDownscaleDimensions();
      ctx.m_atmosphere->ensureCloudHistoryResources(&ctx, downscaledExtent);

      auto cloudPrev = ctx.m_atmosphere->getPreviousCloudHistory();
      auto cloudCurr = ctx.m_atmosphere->getCurrentCloudHistory();
      if (cloudPrev.isValid()) {
        ctx.bindResourceView(BINDING_ATMOSPHERE_CLOUD_HISTORY_PREV, cloudPrev.view, nullptr);
      }
      if (cloudCurr.isValid()) {
        ctx.bindResourceView(BINDING_ATMOSPHERE_CLOUD_HISTORY_CURR, cloudCurr.view, nullptr);
      }

      // R16_UINT frame-id companion (fork — 2026-05-13). Same lifecycle as the
      // color pair; carries last-refresh frame index per pixel so the shader's
      // age check can reject stale history at foreground-occluded slots.
      auto cloudFrameIdPrev = ctx.m_atmosphere->getPreviousCloudHistoryFrameId();
      auto cloudFrameIdCurr = ctx.m_atmosphere->getCurrentCloudHistoryFrameId();
      if (cloudFrameIdPrev.isValid()) {
        ctx.bindResourceView(BINDING_ATMOSPHERE_CLOUD_HISTORY_FRAME_ID_PREV, cloudFrameIdPrev.view, nullptr);
      }
      if (cloudFrameIdCurr.isValid()) {
        ctx.bindResourceView(BINDING_ATMOSPHERE_CLOUD_HISTORY_FRAME_ID_CURR, cloudFrameIdCurr.view, nullptr);
      }
    }

    // Bind a linear/REPEAT sampler for the cloud noise volume.
    // REPEAT matches the tilable wraparound logic in sampleCloudDensityTextured
    // (frac-based texcoord) so the hardware sampler and the shader math agree.
    // Created per-bind (cheap — DxvkDevice caches identical samplers).
    {
      DxvkSamplerCreateInfo samplerInfo = {};
      samplerInfo.magFilter    = VK_FILTER_LINEAR;
      samplerInfo.minFilter    = VK_FILTER_LINEAR;
      samplerInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
      samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      Rc<DxvkSampler> cloudNoiseSampler = ctx.m_device->createSampler(samplerInfo);
      ctx.bindResourceSampler(BINDING_ATMOSPHERE_CLOUD_NOISE_SAMPLER, cloudNoiseSampler);
    }

    // Sky-view LUT sampler: linear, REPEAT in azimuth (U), CLAMP in elevation
    // (V). Consumed by evalSkyRadiance to replace the per-ray ~50-step
    // atmosphere march with a single bilinear tap of AtmosphereSkyViewLut.
    // CLAMP-V avoids the pole rows mixing horizon values into zenith / nadir
    // at uv.y = 0 or 1; REPEAT-U handles the azimuth wraparound at uv.x = 0/1.
    {
      DxvkSamplerCreateInfo samplerInfo = {};
      samplerInfo.magFilter    = VK_FILTER_LINEAR;
      samplerInfo.minFilter    = VK_FILTER_LINEAR;
      samplerInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
      samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      // Allow explicit-LOD sampling of mips: the secondary cloud LUT (shared via
      // this sampler) is mipmapped and the sky<-clouds bleed samples a coarse
      // mip. Harmless for the mip-less sky-view LUT / cloud RT (only mip 0 used).
      samplerInfo.mipmapLodMax = VK_LOD_CLAMP_NONE;
      Rc<DxvkSampler> skyViewSampler = ctx.m_device->createSampler(samplerInfo);
      ctx.bindResourceSampler(BINDING_ATMOSPHERE_SKY_VIEW_SAMPLER, skyViewSampler);
    }
  }

  // ---------------------------------------------------------------------------
  // getCloudSkyTransmittanceLut
  //
  // Public accessor for the per-frame cloud-occluded sky-ambient transmittance
  // LUT. Returns an invalid Resources::Resource if the atmosphere has not been
  // initialized yet. Used by the debug view to bind the LUT into its
  // pass-local descriptor set.
  //
  // ACCESS NOTE: reads m_atmosphere (private). Friend declaration required in
  // RtxContext.
  // ---------------------------------------------------------------------------
  Resources::Resource getCloudSkyTransmittanceLut(RtxContext& ctx) {
    // Lazy-initialize the atmosphere on demand so the LUT resource is allocated
    // even when the caller (e.g. debug view dispatch) runs before any
    // ray-tracing pass has triggered bindAtmosphereLuts. createLutResources is
    // idempotent and allocates the LUT regardless of skyMode, so the returned
    // resource is always valid after initialize() returns.
    if (!ctx.m_atmosphere) {
      ctx.m_atmosphere = std::make_unique<RtxAtmosphere>(ctx.m_device.ptr());
    }
    ctx.m_atmosphere->initialize(&ctx);
    return ctx.m_atmosphere->getCloudSkyTransmittanceLut();
  }

  // ---------------------------------------------------------------------------
  // getCloudDSun / getCloudDAmbient
  //
  // Public accessors for the Nubis Cubed cloud voxel grids. D_sun stores
  // sun-direction optical depth (used by cloud-on-terrain shadow lookups);
  // D_ambient stores zenith optical depth (used for sky-ambient occlusion of
  // the cloud volume itself). Returns an invalid Resources::Resource if the
  // atmosphere has not been initialized yet. Used by the debug view to bind
  // the grids into its pass-local descriptor set so the user can visually
  // verify the bake content before any production consumer reads from it.
  //
  // ACCESS NOTE: reads m_atmosphere (private). Friend declarations required
  // in RtxContext.
  // ---------------------------------------------------------------------------
  Resources::Resource getCloudDSun(RtxContext& ctx) {
    if (!ctx.m_atmosphere) {
      ctx.m_atmosphere = std::make_unique<RtxAtmosphere>(ctx.m_device.ptr());
    }
    ctx.m_atmosphere->initialize(&ctx);
    return ctx.m_atmosphere->getCloudDSun();
  }

  Resources::Resource getCloudDAmbient(RtxContext& ctx) {
    if (!ctx.m_atmosphere) {
      ctx.m_atmosphere = std::make_unique<RtxAtmosphere>(ctx.m_device.ptr());
    }
    ctx.m_atmosphere->initialize(&ctx);
    return ctx.m_atmosphere->getCloudDAmbient();
  }

  // ---------------------------------------------------------------------------
  // getCloudRenderRT
  //
  // Public accessor for the per-frame Nubis Cubed cloud render RT (C4 of the
  // 2026-05-12 workstream). Returns an invalid Resource until the first
  // updateAtmosphereConstants pass has run ensureCloudRenderRT — the debug
  // view (enum 876) tolerates this by clearing to zero in that case.
  //
  // ACCESS NOTE: reads m_atmosphere (private). Friend declaration required
  // in RtxContext.
  // ---------------------------------------------------------------------------
  Resources::Resource getCloudRenderRT(RtxContext& ctx) {
    if (!ctx.m_atmosphere) {
      ctx.m_atmosphere = std::make_unique<RtxAtmosphere>(ctx.m_device.ptr());
    }
    ctx.m_atmosphere->initialize(&ctx);
    return ctx.m_atmosphere->getCloudRenderRT();
  }

  // ---------------------------------------------------------------------------
  // injectRtxAtmosphereSkySkip
  //
  // Returns true when the caller (RtxContext::rasterizeSky) should skip
  // rasterized sky rendering because Numos mode is active.
  //
  // No private-member access — uses only the public RtxOptions::skyMode() API.
  // No friend declaration needed.
  // ---------------------------------------------------------------------------
  bool injectRtxAtmosphereSkySkip() {
    return RtxOptions::skyMode() == SkyMode::Numos;
  }

  // ---------------------------------------------------------------------------
  // showAtmosphereUI
  //
  // Renders the sky mode selector and atmosphere preset/parameter UI inside
  // the "Sky Tuning" collapsing header (showRenderingSettings). When the sky
  // mode is SkyboxRasterization, draws only the Sky Brightness slider (upstream
  // behaviour). When Numos is selected, draws the full Hillaire
  // atmosphere preset buttons and parameter tree.
  //
  // The skyModeCombo static is owned here (moved from dxvk_imgui.cpp) so that
  // this function is self-contained and requires no parameters.
  //
  // No private-member access — uses only public RtxOptions and ImGui APIs.
  // No friend declaration needed.
  // ---------------------------------------------------------------------------

  namespace {
    // Display-transformed drag widgets (fork - 2026-07-02, UI usability).
    // The option keeps its canonical storage unit (conf/API/shader unchanged);
    // only the widget converts, so sub-decimal crawls like 0.020 km/s become
    // draggable "20.0 m/s". Pattern mirrors RemixGui::DragFloatMB_showGB
    // (rtx_imgui.h); range/step/format arguments are in DISPLAY units. The
    // weather preset editor applies the same transforms via WK_SpeedKmS /
    // WK_PatchPerKm (rtx_fork_weather.cpp) - keep them in sync.

    // Stored km/s, displayed m/s.
    bool dragSpeedKmSAsMS(const char* label, RtxOption<float>* opt,
                          float stepMs, float minMs, float maxMs,
                          ImGuiSliderFlags flags) {
      RemixGui::RtxOptionUxWrapper wrapper(opt);
      float valueMs = opt->get() * 1000.0f;
      const bool changed = RemixGui::DragFloat(label, &valueMs, stepMs, minMs, maxMs, "%.1f m/s", flags);
      if (changed) {
        RemixGui::CheckRtxOptionPopups(opt);
        opt->setDeferred(valueMs * 0.001f);
      }
      return changed;
    }

    // Stored spatial frequency (1/km), displayed as the wavelength in km (1/x)
    // - so a "Patch Size" number IS a size: bigger km = bigger patches.
    // Guards keep 1/x finite for zeroed conf values.
    bool dragFreqPerKmAsKm(const char* label, RtxOption<float>* opt,
                           float stepKm, float minKm, float maxKm,
                           ImGuiSliderFlags flags) {
      RemixGui::RtxOptionUxWrapper wrapper(opt);
      float valueKm = 1.0f / std::max(opt->get(), 1e-6f);
      const bool changed = RemixGui::DragFloat(label, &valueKm, stepKm, minKm, maxKm, "%.0f km", flags);
      if (changed) {
        RemixGui::CheckRtxOptionPopups(opt);
        opt->setDeferred(1.0f / std::max(valueKm, 1.0f));
      }
      return changed;
    }

    // Owned here so that showAtmosphereUI is self-contained. Previously this
    // static lived in dxvk_imgui.cpp at file scope and was passed implicitly
    // via the inline call site. Moved as part of the touchpoint migration.
    RemixGui::ComboWithKey<SkyMode> skyModeCombo {
      "Sky Mode",
      RemixGui::ComboWithKey<SkyMode>::ComboEntries { {
          {SkyMode::SkyboxRasterization, "Skybox Rasterization"},
          {SkyMode::Numos, "Numos"}
      } }
    };

    // Per-moon UI block. RTX_OPTION accessors are static-named per index
    // (enabled0, enabled1, ...), so we dispatch via a small macro that fans
    // the index into one set of pointers, then drive a single index-agnostic
    // ImGui body off those pointers. MAX_MOONS = 4; the macro expands four
    // times — deliberate simple repetition over a fixed cap.
    void renderMoonUI(int idx) {
      constexpr ImGuiSliderFlags sliderFlags = ImGuiSliderFlags_AlwaysClamp;

      RtxOption<bool>*     pEnabled         = nullptr;
      RtxOption<float>*    pAngularRadius   = nullptr;
      RtxOption<float>*    pBrightness      = nullptr;
      RtxOption<Vector3>*  pColor           = nullptr;
      RtxOption<uint32_t>* pSurfaceStyle    = nullptr;
      RtxOption<float>*    pCraterDensity   = nullptr;
      RtxOption<float>*    pSurfaceContrast = nullptr;
      RtxOption<float>*    pNoiseScale      = nullptr;
      RtxOption<float>*    pDarkSide        = nullptr;
      RtxOption<float>*    pRoughness       = nullptr;
      RtxOption<float>*    pElevation       = nullptr;
      RtxOption<float>*    pRotation        = nullptr;
      RtxOption<float>*    pPhase           = nullptr;

      switch (idx) {
#define MOON_PTRS(N)                                                         \
        case N:                                                              \
          pEnabled         = &RtxOptions::enabled##N##Object();              \
          pAngularRadius   = &RtxOptions::angularRadius##N##Object();        \
          pBrightness      = &RtxOptions::brightness##N##Object();           \
          pColor           = &RtxOptions::color##N##Object();                \
          pSurfaceStyle    = &RtxOptions::surfaceStyle##N##Object();         \
          pCraterDensity   = &RtxOptions::craterDensity##N##Object();        \
          pSurfaceContrast = &RtxOptions::surfaceContrast##N##Object();      \
          pNoiseScale      = &RtxOptions::surfaceNoiseScale##N##Object();    \
          pDarkSide        = &RtxOptions::darkSideBrightness##N##Object();   \
          pRoughness       = &RtxOptions::roughnessAmount##N##Object();      \
          pElevation       = &RtxOptions::elevation##N##Object();            \
          pRotation        = &RtxOptions::rotation##N##Object();             \
          pPhase           = &RtxOptions::phase##N##Object();                \
          break
        MOON_PTRS(0);
        MOON_PTRS(1);
        MOON_PTRS(2);
        MOON_PTRS(3);
#undef MOON_PTRS
      default:
        return;
      }

      char headerLabel[16];
      std::snprintf(headerLabel, sizeof(headerLabel), "Moon %d", idx);

      if (ImGui::TreeNode(headerLabel)) {
        RemixGui::Checkbox("Enabled", pEnabled);
        RemixGui::DragFloat("Angular Radius", pAngularRadius, 0.1f, 0.1f, 30.0f, "%.1f deg", sliderFlags);
        RemixGui::DragFloat("Brightness",     pBrightness,    0.1f, 0.0f, 20.0f, "%.1f",         sliderFlags);
        RemixGui::DragFloat3("Color",         pColor,         0.01f, 0.0f, 1.0f, "%.2f",         sliderFlags);

        RemixGui::DragFloat("Elevation", pElevation, 0.1f, -90.0f, 90.0f, "%.1f deg", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover("Moon elevation in degrees. Game-drivable per-frame; slider edits persist when saved unless overridden by a runtime push.");
        RemixGui::DragFloat("Rotation",  pRotation,  0.1f, 0.0f, 360.0f, "%.1f deg", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover("Moon rotation/azimuth in degrees. Same persistence rules as Elevation.");
        RemixGui::DragFloat("Phase",     pPhase,     0.005f, 0.0f, 1.0f, "%.3f",  sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover("Moon phase: 0 = new, 0.25 = first quarter, 0.5 = full, 0.75 = third quarter. Same persistence rules as Elevation.");

        if (ImGui::TreeNode("Appearance")) {
          static const char* kStyleNames[] = { "Rocky", "Volcanic" };
          int styleInt = static_cast<int>(pSurfaceStyle->get());
          if (ImGui::Combo("Surface Style", &styleInt, kStyleNames, IM_ARRAYSIZE(kStyleNames))) {
            pSurfaceStyle->setImmediately(static_cast<uint32_t>(styleInt));
          }
          RemixGui::SetTooltipToLastWidgetOnHover("Procedural surface preset. Knobs below tune the chosen style.");

          RemixGui::DragFloat("Crater Density", pCraterDensity, 0.01f, 0.0f, 2.0f, "%.2f", sliderFlags);

          // #8: Detail knob replaces Surface Contrast + Surface Noise Scale.
          // Detail is transient ImGui state — reconstructed from current Contrast on each
          // frame. NoiseScale is overwritten by the curve when Detail changes; off-curve
          // .conf values are preserved on the Contrast side only.
          //
          // Curve (two-segment linear hitting three anchors exactly):
          //   Detail = 0.0 -> Contrast=0.5, NoiseScale=2.0  (smooth, coarse)
          //   Detail = 1.0 -> Contrast=1.0, NoiseScale=1.0  (default)
          //   Detail = 2.0 -> Contrast=1.5, NoiseScale=0.5  (punchy, fine)
          float detail = (pSurfaceContrast->get() - 0.5f) / 0.5f;
          detail = std::max(0.0f, std::min(2.0f, detail));
          if (ImGui::DragFloat("Detail", &detail, 0.01f, 0.0f, 2.0f, "%.2f", sliderFlags)) {
            float newContrast, newNoiseScale;
            if (detail <= 1.0f) {
              newContrast   = 0.5f + 0.5f * detail;          // 0.5 -> 1.0
              newNoiseScale = 2.0f - 1.0f * detail;          // 2.0 -> 1.0
            } else {
              newContrast   = 1.0f + 0.5f * (detail - 1.0f); // 1.0 -> 1.5
              newNoiseScale = 1.0f - 0.5f * (detail - 1.0f); // 1.0 -> 0.5
            }
            pSurfaceContrast->setImmediately(newContrast);
            pNoiseScale->setImmediately(newNoiseScale);
          }
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Combined surface detail: smooth/coarse <- 0.0 ... 1.0 (default) ... 2.0 -> punchy/fine. "
              "Drives Surface Contrast and Surface Noise Scale via a two-segment linear curve. "
              "Power users can .conf-tune surfaceContrast / surfaceNoiseScale individually for off-curve combinations.");

          RemixGui::DragFloat("Dark Side Brightness", pDarkSide,  0.005f, 0.0f, 1.0f, "%.3f", sliderFlags);
          RemixGui::DragFloat("Roughness",            pRoughness, 0.01f,  0.0f, 3.0f, "%.2f", sliderFlags);
          ImGui::TreePop();
        }

        ImGui::TreePop();
      }
    }

    void renderSunUI() {
      constexpr ImGuiSliderFlags sliderFlags = ImGuiSliderFlags_AlwaysClamp;

      if (ImGui::TreeNode("Sun")) {
        RemixGui::DragFloat("Sun Size", &RtxOptions::sunSizeObject(), 0.01f, 0.0f, 10.0f, "%.3f deg", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover(
            "Sun angular diameter in degrees (Earth's sun is ~0.545 deg). Sets the "
            "visible sun disc, and (unless Shadow Softness below overrides it) the "
            "sun light's half-angle = Sun Size / 2, which drives shadow softness.");

        RemixGui::DragFloat("Shadow Softness", &RtxOptions::sunShadowSoftnessDegObject(), 0.01f, 0.0f, 10.0f, "%.3f deg", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover(
            "Decoupled sun shadow softness (the distant light's angular half-angle, "
            "degrees). 0 = physical: track Sun Size / 2. When > 0 it overrides the "
            "half-angle WITHOUT changing the visible sun disc - larger = softer "
            "penumbra, for soft shadows under a small, crisp sun.");

        RemixGui::DragFloat("Sun Intensity", &RtxOptions::sunIntensityObject(), 0.01f, 0.0f, 100.0f, "%.2f", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover("Strength of Sun");

        RemixGui::DragFloat("Sun Elevation", &RtxOptions::sunElevationObject(), 0.01f, -90.0f, 90.0f, "%.2f deg", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover("Sun angle from horizon");

        RemixGui::DragFloat("Sun Rotation", &RtxOptions::sunRotationObject(), 0.01f, 0.0f, 360.0f, "%.1f deg", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover("Rotation of sun around zenith");

        ImGui::TreePop();
      }
    }

    // Render an RtxOption<Vector3> as a ColorEdit3 chromaticity picker plus a
    // magnitude scalar. opt holds chromaticity * magnitude; the picker shows
    // chromaticity (normalized to max channel == 1 in steady state) and the
    // DragFloat shows magnitude == max(opt).
    //
    // Designed for atmospheric-coefficient triplets (Base Rayleigh / Base Mie /
    // Base Ozone / Base Sun Illuminance) where the Vector3's per-channel ratio
    // IS the visible "color" and the overall magnitude is the user-tunable
    // strength.
    //
    // We cache chromaticity and magnitude per widget across frames because the
    // picker popup manipulates RGB in place and re-deriving them every frame
    // from opt = chromaticity * magnitude makes the SV cursor spring back to
    // V=1 mid-drag (and collapse to (1,1,1) entirely when the user crosses
    // the S=0 axis, taking the popup's "Original" ref swatch with it). Sync
    // from opt only on external mutation (preset load, .conf reload), and
    // re-normalize chromaticity to max=1 once the picker popup closes so the
    // magnitude slider keeps reading max(opt) in steady state.
    void renderChromaticityWidget(const char* colorLabel,
                                  const char* magLabel,
                                  RtxOption<Vector3>* opt,
                                  float magSpeed,
                                  float magMax,
                                  const char* magFormat,
                                  const char* colorTooltip,
                                  const char* magTooltip) {
      constexpr ImGuiSliderFlags sliderFlags = ImGuiSliderFlags_AlwaysClamp;

      struct State {
        Vector3 chromaticity { 1.0f, 1.0f, 1.0f };
        float magnitude = 0.0f;
        Vector3 lastWrittenOpt { 0.0f, 0.0f, 0.0f };
        bool initialized = false;
      };
      static std::unordered_map<const char*, State> states;
      State& st = states[colorLabel];

      const Vector3 v = opt->get();
      const bool externallyChanged = !st.initialized
          || std::abs(v.x - st.lastWrittenOpt.x) > 1e-9f
          || std::abs(v.y - st.lastWrittenOpt.y) > 1e-9f
          || std::abs(v.z - st.lastWrittenOpt.z) > 1e-9f;
      if (externallyChanged) {
        st.magnitude = std::max({v.x, v.y, v.z});
        st.chromaticity = (st.magnitude > 1e-9f)
                        ? Vector3(v.x / st.magnitude, v.y / st.magnitude, v.z / st.magnitude)
                        : Vector3(1.0f, 1.0f, 1.0f);
        st.lastWrittenOpt = v;
        st.initialized = true;
      }

      const bool colorChanged = ImGui::ColorEdit3(colorLabel, &st.chromaticity.x, ImGuiColorEditFlags_NoAlpha);
      if (colorTooltip) RemixGui::SetTooltipToLastWidgetOnHover(colorTooltip);

      const bool magChanged = ImGui::DragFloat(magLabel, &st.magnitude, magSpeed, 0.0f, magMax, magFormat, sliderFlags);
      const bool magActive = ImGui::IsItemActive();
      if (magTooltip) RemixGui::SetTooltipToLastWidgetOnHover(magTooltip);

      if (colorChanged || magChanged) {
        // If the user picks a color while magnitude is zero, color * 0 = (0,0,0)
        // erases the chromaticity entirely. Nudge magnitude to magSpeed so the
        // pick is recoverable.
        if (colorChanged && st.magnitude <= 1e-9f) {
          st.magnitude = magSpeed;
        }
        st.chromaticity.x = std::max(0.0f, std::min(1.0f, st.chromaticity.x));
        st.chromaticity.y = std::max(0.0f, std::min(1.0f, st.chromaticity.y));
        st.chromaticity.z = std::max(0.0f, std::min(1.0f, st.chromaticity.z));
        const Vector3 newOpt(st.chromaticity.x * st.magnitude,
                             st.chromaticity.y * st.magnitude,
                             st.chromaticity.z * st.magnitude);
        opt->setImmediately(newOpt);
        st.lastWrittenOpt = newOpt;
      }

      // Detect ColorEdit3's internal popup state. ColorEdit3 calls
      // PushID(label) then OpenPopup("picker"); mirror the PushID so the
      // hash matches.
      ImGui::PushID(colorLabel);
      const bool pickerOpen = ImGui::IsPopupOpen("picker");
      ImGui::PopID();
      if (!pickerOpen && !magActive) {
        const float maxCh = std::max({st.chromaticity.x, st.chromaticity.y, st.chromaticity.z});
        if (maxCh > 1e-9f && maxCh < 1.0f - 1e-6f) {
          const float invMax = 1.0f / maxCh;
          st.chromaticity = Vector3(st.chromaticity.x * invMax,
                                     st.chromaticity.y * invMax,
                                     st.chromaticity.z * invMax);
          st.magnitude *= maxCh;
          // chromaticity * magnitude is preserved, so opt and lastWrittenOpt
          // stay correct without a writeback.
        }
      }
    }

    void renderStarsUI() {
      constexpr ImGuiSliderFlags sliderFlags = ImGuiSliderFlags_AlwaysClamp;
      if (ImGui::TreeNode("Stars")) {
        RemixGui::DragFloat("Star Brightness", &RtxOptions::starBrightnessObject(),
                            0.1f, 0.0f, 50.0f, "%.1f", sliderFlags);
        RemixGui::DragFloat("Star Density", &RtxOptions::starDensityObject(),
                            0.01f, 0.0f, 1.0f, "%.2f", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover("Threshold: 0 = all stars visible, 1 = no stars.");
        RemixGui::DragFloat("Star Twinkle Speed", &RtxOptions::starTwinkleSpeedObject(),
                            0.1f, 0.0f, 10.0f, "%.1f", sliderFlags);
        ImGui::TreePop();
      }
    }

    void renderMilkyWayUI() {
      constexpr ImGuiSliderFlags sliderFlags = ImGuiSliderFlags_AlwaysClamp;
      if (ImGui::TreeNode("Milky Way")) {
        RemixGui::Checkbox("Enabled##milkyway", &RtxOptions::milkyWayEnabledObject());
        RemixGui::SetTooltipToLastWidgetOnHover(
            "Master toggle for galactic-band effects: in-band density boost, band-specific "
            "star colors, and the diffuse background glow. When off, stars distribute uniformly.");
        RemixGui::DragFloat("Density Boost", &RtxOptions::milkyWayDensityBoostObject(),
                            0.005f, 0.0f, 0.3f, "%.3f", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover(
            "Extra star density inside the galactic band. Higher = more (dim) band stars.");
        RemixGui::DragFloat("Glow Brightness", &RtxOptions::milkyWayBackgroundBrightnessObject(),
                            0.01f, 0.0f, 2.0f, "%.3f", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover(
            "Diffuse band-glow brightness (the soft dust haze across the Milky Way). 0 disables the glow.");
        RemixGui::ColorEdit3("Outer Color", &RtxOptions::milkyWayBackgroundColorObject());
        RemixGui::SetTooltipToLastWidgetOnHover(
            "Cool outer-edge tint of the band (where young stars dominate). Default cool blue.");
        RemixGui::ColorEdit3("Core Color", &RtxOptions::milkyWayCoreColorObject(),
                             ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
        RemixGui::SetTooltipToLastWidgetOnHover(
            "Warm bright-core tint at the galactic center. Default warm cream/yellow. "
            "HDR — values above 1.0 push beyond LDR gamut for a brighter core.");
        // #4: Dust Color slider is intentionally dropped from ImGui.
        // RtxOption rtx.atmosphere.milkyWayDustColor remains .conf-tunable.
        RemixGui::DragFloat("Dust Amount", &RtxOptions::milkyWayDustAmountObject(),
                            0.01f, 0.0f, 1.0f, "%.2f", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover(
            "How strongly dust patches darken the glow. 0 = no dust, 1 = full dust contrast.");
        ImGui::TreePop();
      }
    }

    void renderStarAppearanceUI() {
      constexpr ImGuiSliderFlags sliderFlags = ImGuiSliderFlags_AlwaysClamp;
      if (ImGui::TreeNode("Star Appearance")) {
        RemixGui::DragFloat("Star PSF Sharpness", &RtxOptions::starPsfSharpnessObject(),
                            0.5f, 1.0f, 500.0f, "%.1f", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover(
            "Gaussian PSF exponent. Lower = bigger softer stars, higher = sharper pinpoints.");
        RemixGui::DragFloat("Star Cloud Extinction Power", &RtxOptions::starCloudExtinctionPowerObject(),
                            0.1f, 1.0f, 6.0f, "%.2f", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover(
            "Exponent on cloud view-transmittance when extincting stars. Higher = stars die through clouds faster.");
        RemixGui::DragFloat("Star Ambient Coupling", &RtxOptions::starAmbientCouplingStrengthObject(),
                            0.02f, 0.0f, 3.0f, "%.2f", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover(
            "Star/airglow coupling into cloud-march nightLight, as a multiple of the calibrated "
            "night level (1.0 = calibrated, ~2 doubles it). 0 = disabled.");
        ImGui::TreePop();
      }
    }

    void renderMoonGlobalLightingUI() {
      constexpr ImGuiSliderFlags sliderFlags = ImGuiSliderFlags_AlwaysClamp;
      if (ImGui::TreeNode("Global Lighting")) {
        RemixGui::DragFloat("Atmospheric Coupling", &RtxOptions::moonAtmosphericCouplingStrengthObject(),
                            0.05f, 0.0f, 5.0f, "%.2f", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover(
            "Multiplier on the moon's contribution to atmospheric scattering. "
            "0 = no blue-dome around the moon; 1 = default; >1 = exaggerated.");

        RemixGui::DragFloat("NEE Strength", &RtxOptions::moonNeeStrengthObject(),
                            0.05f, 0.0f, 5.0f, "%.2f", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover(
            "World-side master multiplier on direct moon lighting (surface NEE + cloud + future volumetric).");

        RemixGui::DragFloat("Surface Brightness", &RtxOptions::surfaceMoonBrightnessObject(),
                            1.0f, 0.0f, 200.0f, "%.1f", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover("Per-path multiplier on surface NEE (ground moonlight).");

        RemixGui::DragFloat("Cloud Brightness", &RtxOptions::cloudMoonBrightnessObject(),
                            0.1f, 0.0f, 50.0f, "%.2f", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover(
            "Per-path multiplier on cloud-moon lighting (silver-lining + ambient airglow).");

        RemixGui::DragFloat("Halo Brightness", &RtxOptions::haloMoonBrightnessObject(),
                            0.5f, 0.0f, 100.0f, "%.1f", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover("Per-path multiplier on the disk halo Gaussian glow.");
        ImGui::TreePop();
      }
    }

    void renderMoonCloudLookUI() {
      constexpr ImGuiSliderFlags sliderFlags = ImGuiSliderFlags_AlwaysClamp;
      if (ImGui::TreeNode("Cloud-Look & Halo Shape")) {
        RemixGui::DragFloat("Silver Lining Intensity", &RtxOptions::moonSilverLiningIntensityObject(),
                            0.05f, 0.0f, 5.0f, "%.2f", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover(
            "Brightness of the cloud glow right in front of the moon. Master multiplier "
            "on silver-lining contribution (Lambert diffuse + HG phase). 0 = no silver lining. "
            "1 = default. Power users can .conf-tune moonCloudDiffuseGain / moonCloudPhaseGain for ratio.");

        RemixGui::DragFloat("Silver Lining Sharpness", &RtxOptions::moonCloudAnisotropyObject(),
                            0.01f, -1.0f, 1.0f, "%.2f", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover(
            "Tightness of the silver-lining glow peak. Higher = sharper pinpoint; lower = softer falloff. "
            "Henyey-Greenstein g for cloud-moon forward scatter. Default 0.85.");

        RemixGui::DragFloat("Halo Glow", &RtxOptions::moonHaloGlowStrengthObject(),
                            0.05f, 0.0f, 5.0f, "%.2f", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover(
            "Brightness of the disk halo + ambient airglow around the moon. Master multiplier. "
            "0 = no halo / airglow. 1 = default. Power users can .conf-tune moonHaloMagnitude / "
            "moonAmbientAirglow for ratio.");
        ImGui::TreePop();
      }
    }
  } // anonymous namespace

  void showAtmosphereUI() {
    constexpr ImGuiSliderFlags sliderFlags = ImGuiSliderFlags_AlwaysClamp;

    // Sky mode selection
    skyModeCombo.getKey(&RtxOptions::skyModeObject());
    RemixGui::SetTooltipToLastWidgetOnHover("Skybox Rasterization: Traditional skybox rendering\nNumos: Hillaire atmospheric scattering");

    if (RtxOptions::skyMode() == SkyMode::SkyboxRasterization) {
      RemixGui::DragFloat("Sky Brightness", &RtxOptions::skyBrightnessObject(), 0.01f, 0.01f, FLT_MAX, "%.3f", sliderFlags);
    } else {
      RemixGui::Checkbox("Use Sky View LUT", &RtxOptions::useSkyViewLutObject());
      RemixGui::SetTooltipToLastWidgetOnHover(
        "Sample the precomputed sky-view LUT at sky-miss instead of per-ray evalSkyRadiance. "
        "Improves performance with minimal visual difference.");

      // Atmosphere Presets
      ImGui::Separator();
      ImGui::Text("Atmosphere Presets:");

      if (ImGui::Button("Earth (Default)", ImVec2(120, 0))) {
        // Earth-like atmosphere based on Hillaire paper
        RtxOptions::sunIlluminanceObject().setImmediately(Vector3(20.0f, 20.0f, 20.0f));
        RtxOptions::planetRadiusObject().setImmediately(6371.0f);  // Earth's actual radius
        RtxOptions::atmosphereThicknessObject().setImmediately(100.0f);
        RtxOptions::rayleighScatteringObject().setImmediately(Vector3(5.8e-3f, 13.5e-3f, 33.1e-3f));
        RtxOptions::mieScatteringObject().setImmediately(Vector3(3.996e-3f, 3.996e-3f, 3.996e-3f));
        RtxOptions::mieAnisotropyObject().setImmediately(0.8f);
        RtxOptions::ozoneAbsorptionObject().setImmediately(Vector3(2.04e-3f, 4.97e-3f, 2.14e-4f));
        RtxOptions::ozoneLayerAltitudeObject().setImmediately(25.0f);
        RtxOptions::ozoneLayerWidthObject().setImmediately(15.0f);
      }
      RemixGui::SetTooltipToLastWidgetOnHover("Physically accurate Earth atmosphere parameters from Hillaire paper");

      ImGui::SameLine();
      if (ImGui::Button("Mars", ImVec2(120, 0))) {
        // Mars atmosphere (thin, dusty, red-shifted)
        RtxOptions::sunIlluminanceObject().setImmediately(Vector3(15.0f, 12.0f, 10.0f));  // Weaker, reddish sun
        RtxOptions::planetRadiusObject().setImmediately(3389.5f);  // Mars radius
        RtxOptions::atmosphereThicknessObject().setImmediately(50.0f);  // Thinner atmosphere
        RtxOptions::rayleighScatteringObject().setImmediately(Vector3(8.0e-3f, 10.0e-3f, 12.0e-3f));  // Red bias
        RtxOptions::mieScatteringObject().setImmediately(Vector3(8.0e-3f, 8.0e-3f, 8.0e-3f));  // More dust
        RtxOptions::mieAnisotropyObject().setImmediately(0.7f);
        RtxOptions::ozoneAbsorptionObject().setImmediately(Vector3(0.0f, 0.0f, 0.0f));  // No ozone
        RtxOptions::ozoneLayerAltitudeObject().setImmediately(0.0f);
        RtxOptions::ozoneLayerWidthObject().setImmediately(1.0f);
      }
      RemixGui::SetTooltipToLastWidgetOnHover("Mars-like atmosphere: thin, dusty, yellowish sky with blue sunsets");

      ImGui::SameLine();
      if (ImGui::Button("Clear Sky", ImVec2(120, 0))) {
        // Very clear, minimal scattering (high altitude/clean air)
        RtxOptions::sunIlluminanceObject().setImmediately(Vector3(25.0f, 25.0f, 25.0f));
        RtxOptions::planetRadiusObject().setImmediately(6371.0f);
        RtxOptions::atmosphereThicknessObject().setImmediately(80.0f);
        RtxOptions::rayleighScatteringObject().setImmediately(Vector3(4.0e-3f, 9.0e-3f, 22.0e-3f));  // Reduced
        RtxOptions::mieScatteringObject().setImmediately(Vector3(1.0e-3f, 1.0e-3f, 1.0e-3f));  // Minimal dust
        RtxOptions::mieAnisotropyObject().setImmediately(0.9f);  // Sharp sun
        RtxOptions::ozoneAbsorptionObject().setImmediately(Vector3(2.04e-3f, 4.97e-3f, 2.14e-4f));
        RtxOptions::ozoneLayerAltitudeObject().setImmediately(25.0f);
        RtxOptions::ozoneLayerWidthObject().setImmediately(15.0f);
      }
      RemixGui::SetTooltipToLastWidgetOnHover("Crystal clear atmosphere with minimal haze");

      if (ImGui::Button("Polluted/Hazy", ImVec2(120, 0))) {
        // Heavy pollution/haze (smoggy city)
        RtxOptions::sunIlluminanceObject().setImmediately(Vector3(18.0f, 18.0f, 18.0f));
        RtxOptions::planetRadiusObject().setImmediately(6371.0f);
        RtxOptions::atmosphereThicknessObject().setImmediately(100.0f);
        RtxOptions::rayleighScatteringObject().setImmediately(Vector3(5.8e-3f, 13.5e-3f, 33.1e-3f));
        RtxOptions::mieScatteringObject().setImmediately(Vector3(12.0e-3f, 12.0e-3f, 12.0e-3f));  // Heavy aerosols
        RtxOptions::mieAnisotropyObject().setImmediately(0.65f);  // More diffuse sun
        RtxOptions::ozoneAbsorptionObject().setImmediately(Vector3(2.04e-3f, 4.97e-3f, 2.14e-4f));
        RtxOptions::ozoneLayerAltitudeObject().setImmediately(25.0f);
        RtxOptions::ozoneLayerWidthObject().setImmediately(15.0f);
      }
      RemixGui::SetTooltipToLastWidgetOnHover("Heavy atmospheric haze with strong light scattering");

      ImGui::SameLine();
      if (ImGui::Button("Alien World", ImVec2(120, 0))) {
        // Exotic alien atmosphere (greenish tint)
        RtxOptions::sunIlluminanceObject().setImmediately(Vector3(15.0f, 22.0f, 18.0f));  // Green bias
        RtxOptions::planetRadiusObject().setImmediately(5000.0f);
        RtxOptions::atmosphereThicknessObject().setImmediately(120.0f);
        RtxOptions::rayleighScatteringObject().setImmediately(Vector3(4.0e-3f, 18.0e-3f, 10.0e-3f));  // Green peak
        RtxOptions::mieScatteringObject().setImmediately(Vector3(5.0e-3f, 5.0e-3f, 5.0e-3f));
        RtxOptions::mieAnisotropyObject().setImmediately(0.75f);
        RtxOptions::ozoneAbsorptionObject().setImmediately(Vector3(1.0e-3f, 0.5e-3f, 3.0e-3f));  // Exotic absorption
        RtxOptions::ozoneLayerAltitudeObject().setImmediately(30.0f);
        RtxOptions::ozoneLayerWidthObject().setImmediately(20.0f);
      }
      RemixGui::SetTooltipToLastWidgetOnHover("Fictional alien atmosphere with green-tinted scattering");

      ImGui::SameLine();
      if (ImGui::Button("Desert Planet", ImVec2(120, 0))) {
        // Arid desert world (Dune-like)
        RtxOptions::sunIlluminanceObject().setImmediately(Vector3(28.0f, 24.0f, 18.0f));  // Warm sun
        RtxOptions::planetRadiusObject().setImmediately(6000.0f);
        RtxOptions::atmosphereThicknessObject().setImmediately(90.0f);
        RtxOptions::rayleighScatteringObject().setImmediately(Vector3(7.0e-3f, 11.0e-3f, 18.0e-3f));
        RtxOptions::mieScatteringObject().setImmediately(Vector3(15.0e-3f, 12.0e-3f, 8.0e-3f));  // Sandy dust
        RtxOptions::mieAnisotropyObject().setImmediately(0.6f);  // Diffuse from dust
        RtxOptions::ozoneAbsorptionObject().setImmediately(Vector3(0.5e-3f, 1.0e-3f, 0.1e-3f));
        RtxOptions::ozoneLayerAltitudeObject().setImmediately(20.0f);
        RtxOptions::ozoneLayerWidthObject().setImmediately(10.0f);
      }
      RemixGui::SetTooltipToLastWidgetOnHover("Hot, arid world with sandy atmospheric dust");

      ImGui::Separator();

      // ----- Weather Presets panel (fork, placed right under atmosphere presets) -----
      fork_hooks::showWeatherUI();

      ImGui::Separator();

      // Sun (lifted out of former "Atmosphere Parameters" tree)
      renderSunUI();

      // Numos controls (renamed; Sun fields moved to renderSunUI above)
      if (ImGui::TreeNode("Atmosphere")) {

        RemixGui::DragFloat("Altitude", &RtxOptions::altitudeObject(), 1.0f, 0.0f, 100000.0f, "%.0f m", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover("Height from sea level");

        RemixGui::DragFloat("Air", &RtxOptions::airDensityObject(), 0.01f, 0.0f, 100.0f, "%.2f", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover("Density of air molecules");

        RemixGui::DragFloat("Dust", &RtxOptions::aerosolDensityObject(), 0.01f, 0.0f, 100.0f, "%.2f", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover("Density of aerosols/dust");

        RemixGui::DragFloat("Ozone", &RtxOptions::ozoneDensityObject(), 0.01f, 0.0f, 100.0f, "%.2f", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover("Density of ozone layer");

        if (ImGui::TreeNode("Advanced")) {
          RemixGui::DragFloat("Planet Radius", &RtxOptions::planetRadiusObject(), 10.0f, 1000.0f, 10000.0f, "%.0f km", sliderFlags);
          RemixGui::DragFloat("Atmosphere Thickness", &RtxOptions::atmosphereThicknessObject(), 1.0f, 10.0f, 500.0f, "%.0f km", sliderFlags);
          RemixGui::DragFloat("Mie Anisotropy", &RtxOptions::mieAnisotropyObject(), 0.01f, -1.0f, 1.0f, "%.2f", sliderFlags);

          renderChromaticityWidget(
              "Sun Color (Base)", "Sun Illuminance",
              &RtxOptions::sunIlluminanceObject(),
              0.1f, 100.0f, "%.1f",
              "Sun spectral color (Hillaire base illuminance, chromaticity).",
              "Sun base illuminance magnitude (overall sun-power level).");

          renderChromaticityWidget(
              "Air Color (Base)", "Air Scattering Strength",
              &RtxOptions::rayleighScatteringObject(),
              0.0005f, 0.1f, "%.4f /km",
              "Air molecule scattering chromaticity (Rayleigh per-channel scattering coefficients). "
              "Larger blue = cooler sky.",
              "Air scattering magnitude. Higher = more atmospheric scattering overall.");

          renderChromaticityWidget(
              "Dust Color (Base)", "Dust Scattering Strength",
              &RtxOptions::mieScatteringObject(),
              0.0005f, 0.05f, "%.4f /km",
              "Aerosol / dust scattering chromaticity (Mie per-channel coefficients).",
              "Dust scattering magnitude. Higher = hazier atmosphere.");

          renderChromaticityWidget(
              "Ozone Tint (Base)", "Ozone Absorption Strength",
              &RtxOptions::ozoneAbsorptionObject(),
              0.0001f, 0.05f, "%.5f /km",
              "Ozone absorption chromaticity (per-channel coefficients). "
              "Affects twilight color and high-altitude tint.",
              "Ozone absorption magnitude.");
          RemixGui::DragFloat("Ozone Layer Altitude", &RtxOptions::ozoneLayerAltitudeObject(), 0.5f, 0.0f, 50.0f, "%.1f km", sliderFlags);
          RemixGui::DragFloat("Ozone Layer Width", &RtxOptions::ozoneLayerWidthObject(), 0.5f, 1.0f, 30.0f, "%.1f km", sliderFlags);

          RemixGui::DragFloat("Multiscatter Physical Strength", &RtxOptions::multiScatterPhysicalStrengthObject(), 0.01f, 0.0f, 1.0f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "0 = artistic multiscattering (analytical inline fit; preset color stays faithful, easy to style). "
              "1 = physical multiscattering (Hillaire-style LUT hemisphere integration; wavelength-amplifies each preset's "
              "Rayleigh bias for realistic saturation but harder to art-direct). Intermediate values blend.");

          // Artistic sunset color controls (fork — 2026-06-14). Recover the
          // sunset warmth/saturation lost when reddening moved onto the physical
          // two-term LUT model; both feed the sky-view LUT so clouds inherit them.
          RemixGui::DragFloat("Multiscatter Strength", &RtxOptions::multiScatterStrengthObject(), 0.01f, 0.0f, 2.0f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Global scale on the multiscattering 'fill' term. The physical model adds a broadband (pale-blue) "
              "multiscatter term that desaturates warm sunset color. Lower (e.g. 0.3-0.6) to let warm single-scatter "
              "dominate for a punchier sunset; 1.0 = physical. Feeds the sky-view LUT, so clouds inherit it.");

          RemixGui::DragFloat("Sunset Saturation", &RtxOptions::sunsetSaturationObject(), 0.01f, 0.0f, 3.0f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Saturation boost on sky radiance, ramped in only as the sun nears the horizon (midday sky untouched). "
              ">1 amplifies the warm horizon hues the physical model renders accurately but undersaturated; 1.0 = no change. "
              "Feeds the sky-view LUT, so clouds inherit the warmer ambient.");

          RemixGui::DragFloat("Sky Indirect Scale", &RtxOptions::skyIndirectRadianceScaleObject(), 0.01f, 0.0f, 20.0f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Multiplier for sky radiance gathered by diffuse indirect bounces only. 1.0 = physical. "
              "Raise it to brighten diffuse sky fill (the distant-light sun out-radiates the sky, so indirect "
              "lighting reads dull). Sky seen via reflection, refraction, alpha-cutout, or the primary view stays "
              "at physical brightness, so reflections keep matching the visible sky.");

          // Sky perf workstream knobs (fork — 2026-06-11) are conf-only by
          // design: skyLutCacheKeySplitEnable, skyViewRebakeGranularityDeg and
          // the debug* bisect toggles all default to their validated production
          // values and stay out of the UI (user decision after the in-game
          // validation pass — "this is in a good enough spot now").

          // (cloudVoxelGridRebakeGranularityKm is conf-only like the other
          // workstream knobs above; validated at its 0.1 default.)

          ImGui::TreePop();
        }

        ImGui::TreePop();
      }

      // The Perf Bisect (Diagnostic) tree (fork — 2026-06-11) was removed
      // from the UI after the sky perf workstream closed; the six
      // rtx.atmosphere.debug* skip toggles it drove remain conf-tunable
      // (all default ON = normal rendering) for future regression hunting.
      // See docs/fork-touchpoints.md, sky perf workstream entries.

      // ----- Night Sky tree (fork, restructured) -----
      if (ImGui::TreeNode("Night Sky")) {
        RemixGui::DragFloat("Night Sky Brightness", &RtxOptions::nightSkyBrightnessObject(),
                            0.001f, 0.0f, 0.1f, "%.4f", sliderFlags);
        RemixGui::SetTooltipToLastWidgetOnHover("Airglow / ambient night-sky brightness.");
        RemixGui::ColorEdit3("Night Sky Color", &RtxOptions::nightSkyColorObject());
        RemixGui::SetTooltipToLastWidgetOnHover(
            "Tint of the ambient night-sky / airglow contribution. Magnitude is set by Night Sky Brightness above.");

        renderStarsUI();
        renderMilkyWayUI();
        renderStarAppearanceUI();

        ImGui::TreePop();
      }

      // ----- Moons tree (fork, restructured) -----
      if (ImGui::TreeNode("Moons")) {
        renderMoonGlobalLightingUI();
        renderMoonCloudLookUI();

        for (int i = 0; i < static_cast<int>(MAX_MOONS); ++i) {
          renderMoonUI(i);
        }
        ImGui::TreePop();
      }

      // ----- Clouds tree (fork) -----
      // Simplified menu surface 2026-05-19. 14 user-facing sliders + 1 checkbox
      // + 1 color picker, down from ~38 controls. The hidden RTX_OPTIONs
      // (curvature, view samples, layer-2, Worley, sigma_ms detail knobs,
      // analytical-secondary-ray color polish, etc.) are still alive in code
      // and accessible via user.conf for power tuning. See the 2026-05-19
      // cleanup commit + cloud-settings-audit memory.
      if (ImGui::TreeNode("Clouds")) {
        RemixGui::Checkbox("Enable Clouds", &RtxOptions::cloudEnabledObject());

        // Conditional-disable gates (fork — 2026-06-15, cloud UI rework). Controls
        // that the shader only consumes in a given mode are greyed (not hidden) so
        // they stay discoverable but can't be dragged when inert.
        const bool layer2On  = RtxOptions::cloudLayer2Enable();

        ImGui::SetNextItemOpen(true, ImGuiCond_Once);
        if (ImGui::TreeNode("Basic")) {
          RemixGui::DragFloat("Coverage", &RtxOptions::cloudCoverageMeanObject(),
                              0.01f, 0.0f, 1.0f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "How much of the sky has clouds. 0 = clear, 1 = overcast.");
          RemixGui::DragFloat("Cloud Type", &RtxOptions::cloudTypeMeanObject(),
                              0.01f, 0.0f, 1.0f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Cloud shape from stratus to cumulus. 0 = flat stratus, "
              "0.5 = stratocumulus, 1 = tall cumulus.");
          RemixGui::DragFloat("Density", &RtxOptions::cloudDensityObject(),
                              0.05f, 0.0f, 4.0f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Cloud opacity. Higher = thicker / darker clouds.");
          RemixGui::DragFloat("Altitude", &RtxOptions::cloudAltitudeObject(),
                              0.1f, 0.5f, 12.0f, "%.1f km", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Cloud layer altitude (km above the ground).");
          RemixGui::DragFloat("Depth", &RtxOptions::cloudThicknessObject(),
                              0.05f, 0.1f, 5.0f, "%.2f km", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Vertical depth of the cloud layer in km.");
          RemixGui::ColorEdit3("Color", &RtxOptions::cloudColorObject());
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Base cloud albedo (RGB). Click the swatch for a color picker.");
          ImGui::TreePop();
        }

        if (ImGui::TreeNode("Shaping")) {
          if (ImGui::TreeNode("Variation")) {
            RemixGui::DragFloat("Coverage Spread", &RtxOptions::cloudCoverageSpreadObject(),
                                0.01f, 0.0f, 1.0f, "%.2f", sliderFlags);
            RemixGui::SetTooltipToLastWidgetOnHover(
                "Spatial variation around the Coverage mean. 0 = uniform across "
                "the sky, 1 = mixed clear / cloudy patches.");
            dragFreqPerKmAsKm("Coverage Patch Size", &RtxOptions::cloudCoverageNoiseScaleObject(),
                              5.0f, 100.0f, 10000.0f, sliderFlags);
            RemixGui::SetTooltipToLastWidgetOnHover(
                "Size of the coverage-variation patches (km). Larger = broad "
                "weather regions; smaller = finer clear/cloudy patchwork. "
                "Default ~300 km. (Stored as spatial frequency 1/km in the "
                "conf.)");
            RemixGui::DragFloat("Type Spread", &RtxOptions::cloudTypeSpreadObject(),
                                0.01f, 0.0f, 1.0f, "%.2f", sliderFlags);
            RemixGui::SetTooltipToLastWidgetOnHover(
                "Spatial variation around the Cloud Type mean. 0 = uniform type "
                "everywhere, 1 = full stratus-to-cumulus range across the sky.");
            dragFreqPerKmAsKm("Type Patch Size", &RtxOptions::cloudTypeNoiseScaleObject(),
                              5.0f, 294.0f, 10000.0f, sliderFlags);
            RemixGui::SetTooltipToLastWidgetOnHover(
                "Size of the patches of one cloud type (km). Larger = whole "
                "regions share a type; smaller = finer stratus/cumulus mix. "
                "Floored at ~294 km because faster variation puts visible 2D "
                "cell structure at sub-cumulus scales. Independent of Coverage "
                "Patch Size. (Stored as spatial frequency 1/km in the conf.)");
            RemixGui::DragFloat("Anvil Spread", &RtxOptions::cloudAnvilBiasObject(),
                                0.01f, 0.0f, 1.0f, "%.2f", sliderFlags);
            RemixGui::SetTooltipToLastWidgetOnHover(
                "Cumulus top inflation. 0 = flat tops, 1 = mushroom-cap anvils. "
                "Most visible on tall cumulus / thunderstorm scenes.");
            ImGui::TreePop();
          }

          if (ImGui::TreeNode("Detail & Edges")) {
            RemixGui::DragFloat("Texture Scale", &RtxOptions::cloudNoiseTileKmObject(),
                                1.0f, 6.0f, 24.0f, "%.0f km", sliderFlags);
            RemixGui::SetTooltipToLastWidgetOnHover(
                "World-space tile size for the 3D cloud noise. Smaller = visible "
                "repetition; larger = lower-frequency detail. Re-bakes the noise "
                "volume live on change.");
            RemixGui::Checkbox("Seamless Cloud Field", &RtxOptions::cloudHexTilingEnableObject());
            RemixGui::SetTooltipToLastWidgetOnHover(
                "Randomizes the cloud noise tiling on a triangle lattice so the "
                "texture repeat can never show, while preserving the cloud look. "
                "Uncheck for the legacy periodic field. Applies live.");
            RemixGui::DragFloat("Noise Frequency", &RtxOptions::cloudNoiseBaseFreqScaleObject(),
                                0.01f, 0.25f, 4.0f, "%.2f", sliderFlags);
            RemixGui::SetTooltipToLastWidgetOnHover(
                "Multiplier on the baked cloud noise frequency. 1.0 = current "
                "look. Raise for smaller/busier cloud features, lower for "
                "larger ones. Re-bakes the noise volume live as you drag "
                "(brief hitch per change).");
            RemixGui::DragFloat("Edge Detail", &RtxOptions::cloudDetailStrengthObject(),
                                0.01f, 0.0f, 1.0f, "%.2f", sliderFlags);
            RemixGui::SetTooltipToLastWidgetOnHover(
                "High-frequency detail concentrated at cloud EDGES - grows "
                "wispy cauliflower billows OUTWARD from silhouettes while dense cores stay "
                "solid. 0 = smooth edges (legacy look). Detail frequency is "
                "tunable via rtx.atmosphere.cloudDetailScale in user.conf.");
            RemixGui::DragFloat("Edge Softness", &RtxOptions::cloudEdgeSoftnessObject(),
                                0.005f, 0.02f, 0.4f, "%.3f", sliderFlags);
            RemixGui::SetTooltipToLastWidgetOnHover(
                "Width of the coverage-gate transition band - how soft the "
                "cloud silhouette is. Lower = crisper edges, tighter silhouette; "
                "higher = softer edges but a broader faint skirt that can read as a "
                "halo. Affects the view only; self-shadowing is held at the legacy "
                "softness so this won't shift cloud lighting.");
            RemixGui::DragFloat("Edge Haze Fade", &RtxOptions::cloudEdgeAmbientFadeObject(),
                                0.005f, 0.0f, 0.5f, "%.3f", sliderFlags);
            RemixGui::SetTooltipToLastWidgetOnHover(
                "Fades the (horizon-tinted) ambient on the thinnest edge samples so "
                "the soft skirt around clouds doesn't read as dirty grey-brown haze "
                "- the faintest edges fall toward transparent instead. "
                "Higher = scrub more of the haze tint (can dim thin wisps); 0 = off. "
                "Backlit edges keep their glow (only ambient is faded).");
            ImGui::TreePop();
          }

          if (ImGui::TreeNode("Columns")) {
            // Per-cloud column model (always on since 2026-06-19; the legacy
            // global-slab shaping was removed). Every cloud gets its own base,
            // tower height and complete vertical shape from the placement map.
            RemixGui::DragFloat("Cloud Cell Size", &RtxOptions::cloudCellSizeKmObject(),
                                0.05f, 0.5f, 6.0f, "%.2f km", sliderFlags);
            RemixGui::SetTooltipToLastWidgetOnHover(
                "Average footprint of a cloud cluster in km. Smaller = many "
                "small clouds; larger = fewer, broader cloud banks. Re-bakes "
                "the placement map live on change.");
            RemixGui::DragFloat("Top Variation", &RtxOptions::cloudColumnTopVariationObject(),
                                0.01f, 0.0f, 1.0f, "%.2f", sliderFlags);
            RemixGui::SetTooltipToLastWidgetOnHover(
                "How much cloud-top heights vary from cloud to cloud. 0 = all "
                "tops at the same altitude (flat deck); higher = a varied "
                "skyline of taller and shorter clouds. Applies live.");
            RemixGui::DragFloat("Top Shape", &RtxOptions::cloudColumnTopShapeObject(),
                                0.01f, 0.1f, 2.0f, "%.2f", sliderFlags);
            RemixGui::SetTooltipToLastWidgetOnHover(
                "How cloud height follows cloud thickness toward the cluster "
                "core. Low = even the thin edges tower (blocky); high = only "
                "the dense cores rise (domed, feathered edges). Default 0.6.");
            RemixGui::DragFloat("Base Undulation", &RtxOptions::cloudColumnBaseVariationObject(),
                                0.01f, 0.0f, 0.4f, "%.2f", sliderFlags);
            RemixGui::SetTooltipToLastWidgetOnHover(
                "How much local cloud bases drift above the layer altitude, as "
                "a fraction of the layer depth. 0 = machined-flat ceiling; "
                "higher = gently undulating cloud bases. Applies live.");
            RemixGui::DragFloat("Edge Feather", &RtxOptions::cloudColumnFeatherObject(),
                                0.01f, 0.05f, 1.0f, "%.2f", sliderFlags);
            RemixGui::SetTooltipToLastWidgetOnHover(
                "Width of the transition band at cloud-cluster edges. Narrow = "
                "crisp, solid-cored clouds with sharp gaps; wide = soft, wispy "
                "transitions between cloud and sky. Applies live.");
            RemixGui::DragFloat("Underside Shading", &RtxOptions::cloudUndersideLightSigmaObject(),
                                0.005f, 0.0f, 0.5f, "%.3f", sliderFlags);
            RemixGui::SetTooltipToLastWidgetOnHover(
                "How quickly the light filtering down through each cloud dies "
                "out. The underside brightness then varies continuously with "
                "the water above every point — dark cores, bright thin spots, "
                "smooth gradients — instead of one flat-lit sheet. Higher = "
                "darker, more dramatic undersides; 0 = underside darkening off "
                "(flat-lit base). Sets the SHAPE of the darkening; its overall "
                "strength and sunset fade are set by Bottom Darkening "
                "(Lighting). Applies live.");
            ImGui::TreePop();
          }
          ImGui::TreePop();
        }

        if (ImGui::TreeNode("Lighting")) {
          RemixGui::DragFloat("Forward Scatter", &RtxOptions::cloudPhaseG1Object(),
                              0.01f, 0.0f, 0.99f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Strength of the silver-lining glow when looking toward the sun. "
              "Higher = sharper rim of bright light around backlit clouds.");
          RemixGui::DragFloat("Glow Spread", &RtxOptions::cloudPhaseG2Object(),
                              0.01f, 0.0f, 0.99f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Width of the softer secondary glow lobe around the silver "
              "lining. Higher = tighter / brighter halo; lower = broader / "
              "softer in-scatter envelope. Default 0.3.");
          RemixGui::DragFloat("Multi-Scatter", &RtxOptions::cloudMsScaleObject(),
                              0.05f, 0.0f, 2.0f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Strength of the milky brightness on the underside of cumulus "
              "clouds. 1.0 = Nubis Cubed paper baseline; higher = brighter "
              "cumulus bottoms, lower = flatter lighting.");
          RemixGui::DragFloat("Ground Shadow", &RtxOptions::cloudShadowStrengthObject(),
                              0.01f, 0.0f, 1.0f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "How strongly clouds cast shadows on terrain. 0 = no cloud "
              "shadows, 1 = full voxel-grid cumulus-shaped shadow patches.");
          RemixGui::DragFloat("Bottom Darkening", &RtxOptions::cloudBottomDarkeningObject(),
                              0.01f, 0.0f, 1.0f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Overall strength of the cloud-underside darkening. Scales the "
              "analytic per-column light field (whose shape is set by Underside "
              "Shading) on the multi-scatter and ambient terms; the direct sun "
              "beam (silver lining) is unaffected. Strongest with the sun "
              "overhead and fades out toward the horizon, where the low sun "
              "lights the bases directly (sunset glow). 0 = uniformly lit "
              "(paper baseline).");
          RemixGui::DragFloat("Sky Fill", &RtxOptions::cloudSkyAmbientFillObject(),
                              0.01f, 0.0f, 1.0f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "How strongly cloud undersides pick up the open sky around them. "
              "Adds the overhead sky color as fill light that bypasses Bottom "
              "Darkening (skylight reaches the base from below/around, not "
              "through the cloud), so a bright daytime sky lifts gloomy "
              "undersides and tints them with the real sky color. Fades on its "
              "own at sunset. Higher = brighter, more sky-colored bases; 0 = "
              "undersides ignore the open sky.");
          RemixGui::DragFloat("Sky Cloud Bleed", &RtxOptions::cloudSkyBleedStrengthObject(),
                              0.01f, 0.0f, 1.0f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "How strongly the clouds tint the sky around them (the reverse of "
              "Sky Fill). The sky picks up cloud-colored light next to clouds, "
              "so an orange sunset deck warms the blue gaps and a grey overcast "
              "greys the surrounding sky, instead of clouds and sky looking like "
              "two separate layers. Fades to nothing in open sky far from any "
              "cloud. Higher = more cloud color in the sky; 0 = off. Needs the "
              "secondary cloud LUT (Performance).");
          ImGui::TreePop();
        }

        // Cloud Motion (fork — 2026-06-21, unification). One subtree for every
        // way the cloud field moves/changes: bulk wind advection, in-place field
        // morphing, and edge boil. All three are integrated by a single per-frame
        // accumulator (RtxAtmosphere::advanceCloudMotion), so the slow weather
        // "Weather Variation" (Weather panel) that varies wind speed/direction
        // composes smoothly here rather than snapping the field. Rates are
        // independent (no cross-coupling). Any speed at 0 freezes that part.
        if (ImGui::TreeNode("Cloud Motion")) {
          dragSpeedKmSAsMS("Wind Speed", &RtxOptions::cloudWindSpeedObject(),
                           0.5f, 0.0f, 1000.0f, sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "How fast the whole cloud field drifts across the sky (m/s). "
              "Real decks drift ~5-30 m/s. (Stored as km/s in the conf.)");
          RemixGui::DragFloat("Wind Direction", &RtxOptions::cloudWindDirectionObject(),
                              1.0f, 0.0f, 360.0f, "%.1f deg", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Compass direction the wind blows toward in degrees. "
              "0 = +X, 90 = +Z.");

          ImGui::Separator();

          dragSpeedKmSAsMS("Morph Speed", &RtxOptions::cloudEvolutionSpeedObject(),
                           0.1f, 0.0f, 50.0f, sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "How fast cloud formations form and dissolve in place (m/s). "
              "Scrolls the base 3D noise through the volume, decorrelated from "
              "wind. 0 = field frozen (legacy rigid drift). (Stored as km/s "
              "in the conf.)");
          RemixGui::DragFloat("Morph Vertical Bias", &RtxOptions::cloudEvolutionVerticalBiasObject(),
                              0.02f, 0.0f, 1.0f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Share of the morph scroll along the volume's vertical axis [0..1]. "
              "Higher = more in-place churn; lower = more lateral sliding.");
          dragSpeedKmSAsMS("Edge Boil Speed", &RtxOptions::cloudBoilSpeedObject(),
                           0.1f, 0.0f, 50.0f, sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "How fast cloud edge billows churn (m/s), independent of the base "
              "shape. Only active when edge detail strength > 0. 0 = edges "
              "frozen. (Stored as km/s in the conf.)");

          ImGui::TextDisabled("Slow weather-scale wind/coverage wander: Weather "
                              "-> Weather Variation");
          ImGui::TreePop();
        }

        if (ImGui::TreeNode("Layer 2")) {
          RemixGui::Checkbox("Enable Layer 2",
                             &RtxOptions::cloudLayer2EnableObject());
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Adds a second high-altitude cloud deck on top of the main "
              "layer. Off by default. Voxel-grid terrain shadows still come "
              "from layer 1 only.");
          ImGui::BeginDisabled(!layer2On);
          RemixGui::DragFloat("Layer 2 Altitude", &RtxOptions::cloudLayer2AltitudeObject(),
                              0.1f, 0.5f, 20.0f, "%.1f km", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Layer-2 altitude in km. Default 7.5 km targets the cirrus band.");
          RemixGui::DragFloat("Layer 2 Depth", &RtxOptions::cloudLayer2ThicknessObject(),
                              0.05f, 0.05f, 3.0f, "%.2f km", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Vertical depth of the layer-2 slab. Cirrus is thin - default 0.5 km.");
          RemixGui::DragFloat("Layer 2 Coverage", &RtxOptions::cloudLayer2CoverageMeanObject(),
                              0.01f, 0.0f, 1.0f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "How much of the sky has layer-2 clouds. Defaults sparser than "
              "layer 1 so cirrus reads as patches, not overcast.");
          RemixGui::DragFloat("Layer 2 Cloud Type", &RtxOptions::cloudLayer2TypeMeanObject(),
                              0.01f, 0.0f, 1.0f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Cloud type for layer 2. Low values (~0.05) read as stratiform "
              "wisps - appropriate for cirrus.");
          RemixGui::DragFloat("Layer 2 Type Spread", &RtxOptions::cloudLayer2TypeSpreadObject(),
                              0.01f, 0.0f, 1.0f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Spatial variation around the Layer 2 Cloud Type mean. "
              "Independent of Layer 1's Type Spread.");
          RemixGui::DragFloat("Layer 2 Density", &RtxOptions::cloudLayer2DensityScaleObject(),
                              0.01f, 0.0f, 2.0f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Per-step density multiplier for layer 2 only. Lower values keep "
              "the echo deck from competing with the main cumulus deck.");
          RemixGui::DragInt("Layer 2 Step Floor", &RtxOptions::cloudLayer2StepFloorObject(),
                            1.0f, 2, 64, "%d", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Minimum march steps through the echo deck (hit on near-zenith "
              "sightlines). The deck is marched more cheaply than layer 1 "
              "(which floors at 32); raise for a smoother deck at higher cost.");
          RemixGui::DragInt("Layer 2 Max Steps", &RtxOptions::cloudLayer2StepMaxObject(),
                            1.0f, 2, 128, "%d", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Hard cap on echo-deck samples per ray - the deck's performance "
              "governor. Between the floor and this cap the count follows Cloud "
              "Sample Spacing (cloudViewStepKm).");
          RemixGui::ColorEdit3("Layer 2 Color", &RtxOptions::cloudLayer2ColorObject());
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Base color (albedo) of the echo deck, independent of the main "
              "cloud Color. Defaults to the same near-white; tint it to "
              "differentiate the upper deck. All other look knobs stay shared "
              "with layer 1.");
          ImGui::EndDisabled();
          ImGui::TreePop();
        }

        if (ImGui::TreeNode("Performance")) {
          RemixGui::Checkbox("Fast Cloud Reflections", &RtxOptions::cloudSecondaryLutEnableObject());
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Reflections and indirect light sample a small per-frame cloud "
              "lookup table instead of re-marching the cloud volume per ray. "
              "Large performance win on cloudy skies; reflected clouds also "
              "match the main sky exactly. Uncheck to restore the legacy "
              "per-ray cloud march for comparison.");
          RemixGui::DragFloat("Cloud Render Scale", &RtxOptions::cloudRenderResolutionScaleObject(),
                              0.05f, 0.25f, 1.0f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Resolution of the cloud render relative to the internal render "
              "resolution. 0.5 = quarter the pixels (~4x cheaper clouds, "
              "slightly softer); 1.0 = native (legacy). Applies live.");
          RemixGui::DragFloat("Cloud Sample Spacing", &RtxOptions::cloudViewStepKmObject(),
                              0.01f, 0.0f, 1.0f, "%.2f km", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Distance between cloud samples along each view ray, in km. "
              "This is the fix for the horizontal banding near the horizon: "
              "sightlines there cross 50+ km of cloud layer, and the old "
              "fixed 32-sample march spaced samples too far apart to resolve "
              "the clouds.\n\nPERFORMANCE: cost scales with how many samples "
              "a ray needs -- overhead sightlines are unchanged, but "
              "horizon-heavy views can take up to Max Cloud Samples / 32 "
              "times the cloud cost (2x at the defaults). Raise the spacing "
              "or lower Max Cloud Samples to claw the cost back, or set 0 "
              "to restore the legacy fixed march (banding returns). "
              "Cloud Render Scale above also directly offsets this cost. "
              "Applies live.");
          RemixGui::DragInt("Max Cloud Samples", &RtxOptions::cloudViewSamplesMaxObject(),
                            1.0f, 32, 256, "%d", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Hard cap on cloud samples per ray -- the performance governor "
              "for Cloud Sample Spacing. 64 resolves the default spacing "
              "out to ~6 km of cloud span; lower values cost less but let "
              "a little banding back in at the far horizon. 32 = legacy "
              "cost ceiling. Applies live.");
          ImGui::TreePop();
        }

        if (ImGui::TreeNode("Horizon & Haze")) {
          RemixGui::DragFloat("Curvature", &RtxOptions::cloudCurvatureObject(),
                              0.01f, 0.0f, 1.0f, "%.2f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "Sky-dome curvature. 0 = real-planet radius (nearly flat ceiling, "
              "horizon-grazing clouds stretch far); 1 = tight dome (clouds curve "
              "visibly down to the horizon). Atmosphere math unaffected.");
          RemixGui::DragFloat("Distance Haze", &RtxOptions::cloudAerialHazePerKmObject(),
                              0.005f, 0.0f, 0.5f, "%.3f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "How much distant cloud samples dim toward atmospheric color "
              "(per-km haze extinction on cloud radiance). Higher = softer, "
              "more washed-out distant clouds; 0 = no haze (clouds stay bright "
              "all the way to horizon). Does NOT prevent the horizon white "
              "wall - that's the Horizon Fade slider below. Default 0.05.");
          RemixGui::DragFloat("Horizon Fade", &RtxOptions::cloudAerialFadePerKmObject(),
                              0.005f, 0.0f, 0.5f, "%.3f", sliderFlags);
          RemixGui::SetTooltipToLastWidgetOnHover(
              "How quickly distant cloud samples stop piling up extinction "
              "(per-km fade rate on alpha accumulation). Higher = sky shows "
              "through earlier at the horizon; 0 = no fade (clouds can pile "
              "into a solid white wall on horizon-grazing rays through thick "
              "overcast). Does NOT affect cloud appearance close to camera. "
              "Default 0.15.");
          ImGui::TreePop();
        }

        ImGui::TreePop();
      }
    }
  }

} // namespace fork_hooks
} // namespace dxvk
