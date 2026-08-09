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
   * Checks if parameters have changed and recomputes LUTs if necessary.
   */
  void computeLuts(Rc<DxvkContext> ctx);

  /**
   * \brief Bind atmosphere resources to pipeline
   */
  void bindResources(Rc<DxvkContext> ctx, VkPipelineBindPoint pipelineBindPoint);

  /**
   * \brief Check if LUTs need recomputation
   */
  bool needsLutRecompute() const;

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
   * \brief Build atmosphere parameters from current RtxOptions (no GPU state).
   */
  static AtmosphereArgs buildAtmosphereArgsFromOptions();

  /**
   * \brief Get current atmosphere parameters
   */
  AtmosphereArgs getAtmosphereArgs() const {
    return buildAtmosphereArgsFromOptions();
  }

  /**
   * \brief Isotropic sky ambient estimate for volumetric multi-scatter fill.
   * Scale is applied by the caller. Prefer sky-view LUT froxel inject when available.
   */
  static Vector3 estimateVolumeAmbientRadiance(const AtmosphereArgs& args);

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
  void createLutResources(Rc<DxvkContext> ctx);
  void dispatchSkyViewLut(Rc<DxvkContext> ctx);

  // LUT dimensions
  static constexpr uint32_t kTransmittanceLutWidth = 512;   // Increased from 256 for better precision
  static constexpr uint32_t kTransmittanceLutHeight = 128;  // Increased from 64 for better precision
  static constexpr uint32_t kMultiscatteringLutSize = 32;
  static constexpr uint32_t kSkyViewLutWidth = 512;   // Increased from 192 to eliminate aliasing artifacts
  static constexpr uint32_t kSkyViewLutHeight = 256;  // Increased from 108 to eliminate aliasing artifacts

  // Scale heights for exponential density profiles (in km)
  static constexpr float kRayleighScaleHeight = 8.0f;
  static constexpr float kMieScaleHeight = 1.2f;

  Resources::Resource m_transmittanceLut;
  Resources::Resource m_multiscatteringLut;
  Resources::Resource m_skyViewLut;
  
  Rc<DxvkBuffer> m_constantsBuffer;
  Rc<DxvkSampler> m_lutSampler;

  AtmosphereArgs m_cachedArgs;
  bool m_initialized = false;
  bool m_lutsNeedRecompute = true;

  RtLight* m_sunDistantLight = nullptr; // LightManager-owned; pointer only.
};

} // namespace dxvk
