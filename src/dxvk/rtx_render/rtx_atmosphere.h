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
#pragma once

#include "rtx_resources.h"
#include "rtx_common_object.h"
#include "rtx/pass/atmosphere/atmosphere_args.h"

namespace dxvk {

class DxvkContext;
class DxvkDevice;
class RtxContext;
struct RtLight;

/**
 * \brief Hillaire Physically-Based Atmospheric Scattering
 *
 * Manages LUT resources and compute dispatch for Hillaire atmosphere, and injects
 * the sun as an RtDistantLight shared by surface NEE and Volume ReSTIR.
 */
class RtxAtmosphere : public CommonDeviceObject {
public:
  explicit RtxAtmosphere(DxvkDevice* device);
  ~RtxAtmosphere();

  /**
   * \brief Initialize atmosphere resources
   */
  void initialize(Rc<DxvkContext> ctx);

  /**
   * \brief Compute atmospheric LUTs if needed
   *
   * The transmittance and multiscattering LUTs depend on the medium alone and the sky-view LUT also on the
   * sun and the viewpoint, so each is rebaked when its inputs change. The sky-view hemisphere mean is derived
   * from the sky-view LUT at the same time. The aerial perspective volume is rebuilt every frame by
   * dispatchAerialPerspective() instead, once the primary hits it is bounded by exist.
   */
  void computeLuts(Rc<DxvkContext> ctx, const AtmosphereArgs& args);

  /**
   * \brief Build this frame's aerial perspective volume, unshadowed or ray traced per the options.
   *
   * Reads the primary linear view Z of rtOutput, so it must run after the G-buffer pass and before
   * composite. computeLuts() must have run this frame.
   */
  void dispatchAerialPerspective(RtxContext& ctx, const Resources::RaytracingOutput& rtOutput);

  /**
   * \brief Get transmittance LUT resource
   */
  Resources::Resource getTransmittanceLut() const { return m_transmittanceLut; }

  /**
   * \brief Get multiscattering LUT resource
   */
  Resources::Resource getMultiscatteringLut() const { return m_multiscatteringLut; }

  /**
   * \brief Get sky view LUT resource
   */
  Resources::Resource getSkyViewLut() const { return m_skyViewLut; }

  /**
   * \brief 1x1 isotropic hemisphere mean of the sky-view LUT.
   */
  Resources::Resource getSkyHemisphereMean() const { return m_skyHemisphereMean; }

  /**
   * \brief Get this frame's aerial perspective LUT resource
   */
  Resources::Resource getAerialPerspectiveLut() const { return m_aerialPerspectiveLut[m_aerialPerspectiveLutIndex]; }

  /**
   * \brief Tabulated aerosol phase function, 1D over sqrt(theta / pi), RGB per channel wavelength.
   * Filled when the tabulated phase is in use; miePhase() only samples it then.
   */
  Resources::Resource getAerosolPhaseLut() const { return m_aerosolPhaseLut; }

  /**
   * \brief Optics of an aerosol type of the Visibility model, per RGB channel wavelength.
   * OPAC types come from the tables at the given relative humidity; the Custom type from its options.
   */
  struct AerosolOptics {
    Vector3 singleScatteringAlbedo;
    Vector3 extinctionRatio;  // Extinction relative to 550 nm
    Vector3 asymmetry;        // Phase asymmetry g
    float extinction550 = 0.0f;  // km^-1 at the database's own number density, 0 for the Custom type
    bool tabulated = false;      // Phase function and spectral data come from the tables
  };
  static AerosolOptics getAerosolOptics(AtmosphereAerosolType type, float relativeHumidityPercent);

  /**
   * \brief Build atmosphere parameters from current RtxOptions (no GPU state).
   *
   * The aerial perspective camera fields are left zeroed; fillAerialPerspectiveArgs supplies them
   * once a camera is available.
   */
  static AtmosphereArgs buildAtmosphereArgsFromOptions();

  /**
   * \brief Fill in the per-frame camera fields: the frustum basis of the aerial perspective volume,
   * the camera following altitude and the previous frame's basis for the volume's history.
   */
  void fillAerialPerspectiveArgs(AtmosphereArgs& args, const class RtCamera& camera) const;

  /**
   * \brief Get current atmosphere parameters
   */
  AtmosphereArgs getAtmosphereArgs() const {
    return buildAtmosphereArgsFromOptions();
  }

  /**
   * \brief Koschmieder visibility (km) implied by the ground level extinction of the given parameters.
   */
  static float computeEffectiveVisibilityKm(const AtmosphereArgs& args);

  /**
   * \brief Weight (0 at high sun, 1 at low sun) with which the custom aerosol takes on its low sun optics.
   * Zero unless the Visibility model's Custom type is active with aerosolLowSunBlend enabled.
   */
  static float computeAerosolLowSunBlend();

  /**
   * \brief Whether the previous ray traced aerial perspective volume may be reprojected this frame.
   * computeLuts() can clear this after fillAerialPerspectiveArgs() ran, so callers re-read it.
   */
  bool isAerialPerspectiveHistoryValid() const { return m_aerialPerspectiveHistoryValid; }

  /**
   * \brief Isotropic sky ambient estimate for volumetric multi-scatter fill.
   * Scale is applied by the caller. Prefer sky-view LUT froxel inject when available.
   */
  static Vector3 estimateVolumeAmbientRadiance(const AtmosphereArgs& args);

  /**
   * \brief Unoccluded ground-reaching sun illuminance and world-space direction.
   * Composite applies medium transmittance, firefly filtering, and SH HG.
   */
  static void estimateUnoccludedVolumeLighting(
    const AtmosphereArgs& args,
    bool isZUp,
    Vector3& outSunRadiance,
    Vector3& outSunDirectionWorld);

  /**
   * \brief Low-sun warm tint for froxel composite (σ_s desaturation + mild warm bias).
   *
   * outBlend is 0 at high sun and rises toward the horizon. Keeps artistic medium colour
   * at high sun while shifting in-scatter toward expected low-sun haze behaviour.
   */
  static void estimateVolumeSunsetWarmTint(const AtmosphereArgs& args, Vector3& outTint, float& outBlend);

  /**
   * \brief Sync the atmosphere sun into the scene light pool as an RtDistantLight.
   * Sole sun path for surface NEE and Volume ReSTIR while Physical Atmosphere is active.
   */
  void syncDistantSunLight(RtxContext& ctx, const AtmosphereArgs& args);

  /** \brief Remove the injected sun distant light. */
  void dropDistantSunLight();

private:
  bool needsSkyViewRecompute(const AtmosphereArgs& args) const;
  bool needsMediumRecompute(const AtmosphereArgs& args) const;
  void createLutResources(Rc<DxvkContext> ctx);
  void ensureAerialPerspectiveLuts(Rc<DxvkContext> ctx, const AtmosphereArgs& args);
  void updateAerosolPhaseLut(Rc<DxvkContext> ctx, const AtmosphereArgs& args);
  void dispatchTransmittanceLut(Rc<DxvkContext> ctx);
  void dispatchMultiscatteringLut(Rc<DxvkContext> ctx);
  void dispatchSkyViewLut(Rc<DxvkContext> ctx);
  void dispatchSkyHemisphereMean(Rc<DxvkContext> ctx);
  void dispatchAerialPerspectiveTileDepth(RtxContext& ctx, const Rc<DxvkImageView>& primaryLinearViewZ);
  void dispatchAerialPerspectiveLut(RtxContext& ctx, const AtmosphereArgs& args);
  void dispatchShadowedAerialPerspectiveLut(RtxContext& ctx, const AtmosphereArgs& args);

  // LUT dimensions
  static constexpr uint32_t kTransmittanceLutWidth = 512;
  static constexpr uint32_t kTransmittanceLutHeight = 128;
  static constexpr uint32_t kMultiscatteringLutSize = 32;  // Per tile of the multiple scattering atlas
  static constexpr uint32_t kSkyViewLutWidth = 512;
  static constexpr uint32_t kSkyViewLutHeight = 256;
  // Over sqrt(theta / pi): 0.07 degree texels at the forward peak, 1.4 degrees at back-scatter.
  static constexpr uint32_t kAerosolPhaseLutSize = 512;

  // Scale heights for exponential density profiles (in km)
  static constexpr float kRayleighScaleHeight = 8.0f;
  static constexpr float kMieScaleHeight = 1.2f;

  // Only the leading, camera independent portion of AtmosphereArgs invalidates the baked LUTs. The
  // aerial perspective fields that follow change every frame and have their own dispatch.
  static constexpr size_t kBakeInvariantArgsSize = offsetof(AtmosphereArgs, aerialPerspectiveLutSize);

  Resources::Resource m_transmittanceLut;
  Resources::Resource m_multiscatteringLut;
  Resources::Resource m_multiscatteringScratch;
  Resources::Resource m_skyViewLut;
  Resources::Resource m_skyHemisphereMean;
  // Two of each so the ray traced variant can reproject the previous frame while writing the current one.
  Resources::Resource m_aerialPerspectiveLut[2];
  // Farthest primary hit under each screen tile of the volume, which bounds the tile's march.
  Resources::Resource m_aerialPerspectiveTileDepth[2];
  uint32_t m_aerialPerspectiveLutIndex = 0;
  VkExtent3D m_aerialPerspectiveLutExtent = { 0, 0, 0 };
  bool m_aerialPerspectiveHistoryValid = false;
  uint32_t m_aerialPerspectiveFrameIndex = 0;

  Resources::Resource m_aerosolPhaseLut;
  // Type and humidity the phase LUT currently holds; re-uploaded when they change.
  uint32_t m_aerosolPhaseLutType = ~0u;
  float m_aerosolPhaseLutHumidity = -1.0f;

  Rc<DxvkBuffer> m_constantsBuffer;

  AtmosphereArgs m_cachedArgs;
  bool m_initialized = false;
  bool m_lutsNeedRecompute = true;

  RtLight* m_sunDistantLight = nullptr; // LightManager-owned; pointer only.
};

} // namespace dxvk
