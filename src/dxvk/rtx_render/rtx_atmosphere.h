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
#include "rtx_mipmap.h"
#include "rtx_common_object.h"
#include "rtx_fast_noise.h"
#include "rtx/pass/atmosphere/atmosphere_args.h"

#include <atomic>

namespace dxvk {

class DxvkContext;
class DxvkDevice;

/**
 * \brief Hillaire Physically-Based Atmospheric Scattering
 * 
 * Manages lookup table (LUT) resources and compute shader dispatch
 * for atmospheric scattering based on Sebastien Hillaire's method.
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
   * \brief Get the cloud-occluded sky-ambient transmittance LUT (fork).
   *
   * 2D R16F texture keyed by (azimuth, elevation). Baked per frame from camera
   * position. Consumed by the volumetric pass's sky-ambient hemisphere
   * integration to attenuate sky-view radiance per direction.
   */
  Resources::Resource getCloudSkyTransmittanceLut() const { return m_cloudSkyTransmittanceLut; }

  /**
   * \brief Get the cloud D_sun voxel grid (Nubis Cubed 2023, fork — 2026-05-12).
   *
   * 256x256x32 R16F camera-centered tile-wrapped voxel grid storing summed
   * optical depth along the sun direction. Baked every 8 frames at offset 0
   * by cloud_sun_density_grid.comp.slang. Consumed at shade time via
   * sampleDSun() by the Nubis Cubed cloud-lighting path.
   */
  const Resources::Resource& getCloudDSun() const { return m_cloudDSun; }

  /**
   * \brief Get the cloud D_ambient voxel grid (Nubis Cubed 2023, fork — 2026-05-12).
   *
   * 256x256x32 R16F camera-centered tile-wrapped voxel grid storing summed
   * optical depth toward zenith. Baked every 8 frames at offset 4 by
   * cloud_ambient_density_grid.comp.slang. Consumed at shade time via
   * sampleDAmbient() for the Nubis Cubed page-142 ambient attenuation term.
   */
  const Resources::Resource& getCloudDAmbient() const { return m_cloudDAmbient; }

  /**
   * \brief Get the published cloud NVDF SDF (fork — Nubis3 conversion Phase A).
   *
   * 256x64x256 R16F tile-periodic signed distance field of the cloud BODY
   * (placement map + column model at the baked nominal coverage), in raw km,
   * negative inside. Double-buffered; this returns the FRONT buffer — the last
   * fully-published bake. Consumed by the enum 879 debug slice view (Phase A)
   * and, from Phase B, by the Nubis3 density sampler + sphere-traced stepping.
   */
  const Resources::Resource& getCloudNvdfSdf() const { return m_cloudNvdfSdf[m_cloudNvdfSdfFront]; }

  /**
   * \brief Get the cloud render RT (Nubis Cubed 2023, fork — 2026-05-12, C4).
   *
   * Screen-space RGBA16F at the downscale render extent containing per-pixel
   * cloud color (premultiplied, rgb) and view-ray cloud transmittance (alpha).
   * Produced once per frame by cloud_render.comp.slang via dispatchCloudRender
   * (called from computeLuts). Visualized standalone via the enum 876 debug
   * view; will feed the sky-miss composite (C5 of the 2026-05-12 workstream).
   */
  const Resources::Resource& getCloudRenderRT() const { return m_cloudRenderRT; }

  /**
   * \brief Get the secondary-ray cloud LUT (fork — 2026-06-10, perf).
   *
   * 256x128 RGBA16F dome keyed (azimuth, elevation = (pi/2)*v^2) holding the
   * full Nubis cloud march per direction: rgb = premultiplied cloud radiance,
   * a = view transmittance. Baked once per frame by
   * cloud_secondary_lut.comp.slang; consumed by evalSkyRadiance's non-primary
   * branch (indirect / PSR / reflection sky-miss) to supply clouds without a
   * per-ray cloud march.
   */
  const Resources::Resource& getCloudSecondaryLut() const { return m_cloudSecondaryLut; }

  /**
   * \brief Ensure the cloud render RT exists at the requested downscale extent.
   *
   * Recreates the RT on resize. Cheap when the extent is unchanged. Called
   * each frame from RtxAtmosphere::computeLuts before dispatchCloudRender.
   */
  void ensureCloudRenderRT(Rc<DxvkContext> ctx, const VkExtent2D& downscaleExtent);

  /**
   * \brief Push the per-frame camera basis vectors that cloud_render.comp.slang
   *        consumes for view-ray reconstruction.
   *
   * Called from `fork_hooks::updateAtmosphereConstants` before `computeLuts`
   * runs, so the values land in m_constantsBuffer in time for the cloud-
   * render dispatch. All in Y-up world space; the Right/Up vectors are
   * pre-scaled by tan(halfFovX/Y) and aspect ratio so the shader only needs
   * to do a weighted sum.
   */
  void setCloudRenderCameraBasis(const Vector3& forwardYUp,
                                  const Vector3& rightYUp,
                                  const Vector3& upYUp,
                                  uint32_t frameIdx);

  /**
   * \brief Push the per-frame camera world position (Y-up km) used by
   *        sampleCloudGroundShadow_OptionB to express the surface worldPos
   *        camera-relative for the camera-centered D_sun voxel grid lookup.
   *
   * Called from `fork_hooks::updateAtmosphereConstants` BEFORE `computeLuts`
   * runs so the value lands in m_constantsBuffer alongside the other C6
   * voxel-grid plumbing. The position is world-absolute, in km, in the same
   * Y-up frame the cloud math uses elsewhere — the caller does the
   * game-units → km conversion and the isZUp swap, mirroring the existing
   * setCloudRenderCameraBasis() pattern.
   */
  void setCloudShadowCameraPosition(const Vector3& cameraWorldPosYUpKm);

  /**
   * \brief Get the EA Importance-Sampled FAST noise view for descriptor binding
   *
   * Returns nullptr if the FAST noise has not been initialized.
   */
  Rc<DxvkImageView> getFastNoiseView() const { return m_fastNoise.isValid() ? m_fastNoise.getView() : Rc<DxvkImageView>(); }

  /**
   * \brief Ensure the cloud history ping-pong textures exist at the requested
   *        screen-space dimensions, recreating them on resize.
   *
   * Call once per frame (cheap when extent is unchanged). Allocated on the
   * downscaled render extent — the resolution at which the geometry resolver
   * raygen writes pixels (and the resolution DLSS sees as its input).
   */
  void ensureCloudHistoryResources(Rc<DxvkContext> ctx, const VkExtent3D& downscaledExtent);

  /**
   * \brief Swap the cloud history ping-pong index when a new frame starts.
   *
   * Idempotent across multiple calls within the same frame — uses the device
   * frame counter to detect frame transitions. Safe to call from each
   * raygen-bind site without double-swapping.
   */
  void onFrameAdvanceForCloudHistory(uint32_t currentFrameId);

  /**
   * \brief Get the current frame's cloud history (write target this frame).
   *
   * Returns an invalid resource if ensureCloudHistoryResources hasn't yet run.
   */
  const Resources::Resource& getCurrentCloudHistory() const { return m_cloudHistory[m_cloudHistorySwap ? 1u : 0u]; }

  /**
   * \brief Get the previous frame's cloud history (read source this frame).
   */
  const Resources::Resource& getPreviousCloudHistory() const { return m_cloudHistory[m_cloudHistorySwap ? 0u : 1u]; }

  /**
   * \brief Get the current frame's cloud-history frame-id buffer (write target this frame).
   *
   * R16_UINT companion to getCurrentCloudHistory; carries the frame index at
   * which each pixel was last written by the sky-miss path. Read at lookup
   * time by evalSkyRadiance to reject stale history at foreground-occluded
   * pixels.
   */
  const Resources::Resource& getCurrentCloudHistoryFrameId() const { return m_cloudHistoryFrameId[m_cloudHistorySwap ? 1u : 0u]; }

  /**
   * \brief Get the previous frame's cloud-history frame-id buffer (read source this frame).
   */
  const Resources::Resource& getPreviousCloudHistoryFrameId() const { return m_cloudHistoryFrameId[m_cloudHistorySwap ? 0u : 1u]; }

  /**
   * \brief Get current atmosphere parameters
   */
  AtmosphereArgs getAtmosphereArgs() const;

  /**
   * \brief Advance the unified cloud-motion accumulators by one frame
   *        (fork — 2026-06-21, cloud-motion unification).
   *
   * Integrates wind advection + field-evolution morph + edge boil as
   * `offset += velocity * dt` from the live (drift-modulated) RtxOptions, into
   * persistent members read by the const getAtmosphereArgs(). MUST be called
   * exactly once per frame (from updateAtmosphereConstants, alongside the other
   * per-frame setters); getAtmosphereArgs is called many times per frame and so
   * cannot integrate. Replaces the former stateless `speed * timeSeconds`, which
   * mis-scaled/rotated the whole field whenever the weather drift varied wind.
   */
  void advanceCloudMotion(float dt);

  /**
   * \brief Advance the lightning strike scheduler (fork — 2026-07-14)
   *
   * Decays the flash flicker envelope, fires restrike pulses, and schedules /
   * places new strikes around the camera. MUST be called exactly once per
   * frame (from updateAtmosphereConstants), after the camera-position push so
   * placement uses this frame's camera. getAtmosphereArgs publishes the
   * resulting envelope + strike position into the lightning CB fields.
   */
  void advanceLightning(float dt);

  /**
   * \brief Queue a lightning strike for the next advanceLightning tick
   *
   * ImGui "Test Strike" hook. Static so the panel code doesn't need the
   * atmosphere instance; consumed (and cleared) once per frame. Ignored while
   * lightningEnable is off.
   */
  static void requestLightningStrike();

private:
  void createLutResources(Rc<DxvkContext> ctx);
  void dispatchTransmittanceLut(Rc<DxvkContext> ctx);
  void dispatchMultiscatteringLut(Rc<DxvkContext> ctx);
  void dispatchSkyViewLut(Rc<DxvkContext> ctx);
  // Cloud placement map bake (fork — 2026-06-11, column-shaping rework).
  // At init + on bake-input change (cloudCellSizeKm / cloudNoiseTileKm).
  void dispatchCloudPlacementMapBake(Rc<DxvkContext> ctx);
  bool needsCloudPlacementRebake() const;
  void cacheCloudPlacementBakeInputs();
  void dispatchCloudSkyTransmittanceLut(Rc<DxvkContext> ctx);  // Fork: per-frame
  // Cloud NVDF SDF bake chain (fork — Nubis3 conversion Phase A):
  // occupancy voxelize -> JFA seed init -> 9 jump passes -> signed resolve
  // into the back SDF buffer, then publish-swap. Full synchronous chain at
  // init (runCloudNvdfBakeFull); at runtime stepCloudNvdfBake advances the
  // state machine a couple of JFA passes per frame so weather-drift re-bakes
  // never spike a frame. Dirty gate mirrors the placement-map pattern.
  void dispatchCloudNvdfOccupancy(Rc<DxvkContext> ctx);
  void dispatchCloudNvdfJfaPass(Rc<DxvkContext> ctx, uint32_t mode, uint32_t jumpSizeVoxels,
                                uint32_t srcIdx, uint32_t dstIdx);
  void dispatchCloudNvdfResolve(Rc<DxvkContext> ctx, uint32_t seedsIdx);
  void runCloudNvdfBakeFull(Rc<DxvkContext> ctx);
  void stepCloudNvdfBake(Rc<DxvkContext> ctx);
  bool needsCloudNvdfRebake() const;
  void cacheCloudNvdfBakeInputs();
  // Nubis3 wispy/billowy detail volume bake (fork — Nubis3 conversion
  // Phase B). Fixed pattern: baked once at init, no live inputs.
  void dispatchCloudDetailNoiseBake(Rc<DxvkContext> ctx);
  // Cloud voxel grid bakes (Nubis Cubed 2023, fork — 2026-05-12). Round-robin
  // every 8 frames. Driven from computeLuts based on the device frame ID.
  void dispatchCloudSunDensityGrid(Rc<DxvkContext> ctx);
  void dispatchCloudAmbientDensityGrid(Rc<DxvkContext> ctx);
  // Cloud render compute pass (Nubis Cubed 2023, fork — 2026-05-12, C4).
  // Runs each frame from computeLuts after the voxel grid bakes; produces
  // m_cloudRenderRT (screen-space premultiplied cloud rgb + transmittance a).
  void dispatchCloudRender(Rc<DxvkContext> ctx);
  // Secondary-ray cloud LUT bake (fork — 2026-06-10, perf). Runs each frame
  // from computeLuts after the voxel grid bakes (it reads D_sun / D_ambient
  // like the visible march). Gated on cloudSecondaryLutEnable.
  void dispatchCloudSecondaryLut(Rc<DxvkContext> ctx);

  // LUT dimensions
  static constexpr uint32_t kTransmittanceLutWidth = 512;   // Increased from 256 for better precision
  static constexpr uint32_t kTransmittanceLutHeight = 128;  // Increased from 64 for better precision
  static constexpr uint32_t kMultiscatteringLutSize = 32;
  static constexpr uint32_t kSkyViewLutWidth = 512;   // Increased from 192 to eliminate aliasing artifacts
  static constexpr uint32_t kSkyViewLutHeight = 256;  // Increased from 108 to eliminate aliasing artifacts
  // Cloud-occluded sky-ambient transmittance LUT (fork). Small 2D R16F texture
  // keyed by (azimuth, elevation). 32x16 chosen because cumulus features at the
  // bake scale are low-frequency relative to a 360x90 sweep — 32 azimuthal
  // steps approximate cumulus cell width perception; 16 elevation steps cover
  // [-pi/2, pi/2] with finer-than-necessary near-horizon detail.
  // Keep in lockstep with kLutWidth/kLutHeight in cloud_sky_transmittance_lut.comp.slang.
  static constexpr uint32_t kCloudSkyTransmittanceLutWidth = 32;
  static constexpr uint32_t kCloudSkyTransmittanceLutHeight = 16;
  // Cloud voxel grids (Nubis Cubed 2023, fork — 2026-05-12). 256x256x32 R16F,
  // ~4 MB each, ~8 MB combined VRAM. The XY resolution matches the cumulus
  // detail expectation across a 12 km camera-centered tile-wrapped grid (~47 m
  // per voxel horizontally); Z = 32 spans the cloud slab vertically (~30 m per
  // voxel for a 1 km slab). Round-robin baked every 8 frames; each bake costs
  // an 8x8x4 dispatch covering 256x256x32 voxels (~0.1-0.2 ms target).
  // Keep in lockstep with kGridX/Y/Z constants in
  // cloud_sun_density_grid.comp.slang / cloud_ambient_density_grid.comp.slang.
  //
  // AXIS FIX (fork — 2026-07-16, found in the GT7 cross-audit): the UVW
  // mapping routes uvw.y = VERTICAL and uvw.z = world Z
  // (cloudVoxelUVWToWorld), but the allocation had Y=256 / Z=32 — so the
  // vertical axis burned 256 texels on a ~3 km slab (~12 m) while world-Z
  // shadow texels were 375 m wide. Those fat Z-texels were the blocky
  // "squares" cast into the deck (worst at sunset, both density models).
  // Design intent (Nubis: 256x256x32 with 32 VERTICAL) restored: world
  // X/Z get 256 (47 m at the 12 km tile), vertical gets 32 (~95 m).
  static constexpr uint32_t kCloudVoxelGridX = 256;
  static constexpr uint32_t kCloudVoxelGridY = 32;
  static constexpr uint32_t kCloudVoxelGridZ = 256;

  // Cloud NVDF (fork — Nubis3 conversion Phase A). 256x64x256 tile-periodic
  // body grid: occupancy R8 (~4 MB), two R32_UINT JFA seed ping-pongs
  // (~17 MB each), two R16F SDF buffers (~8 MB each, front/back publish
  // pair) — ~54 MB total. Axis convention: texture y = VERTICAL (explicit —
  // see cloud_nvdf.h; the D_sun grids' axis comments disagree with their own
  // mapping, do not pattern-match them). ~47 m voxels at the 12 km / 3 km
  // defaults. Keep in lockstep with CLOUD_NVDF_SIZE_XZ / CLOUD_NVDF_SIZE_Y
  // in cloud_nvdf.h.
  static constexpr uint32_t kCloudNvdfSizeXZ = 256;
  static constexpr uint32_t kCloudNvdfSizeY  = 64;
  // JFA jump schedule: standard halving from half the wrapped XZ extent,
  // plus one extra 1-refinement pass (JFA+1).
  static constexpr uint32_t kCloudNvdfJumpSchedule[] = { 128, 64, 32, 16, 8, 4, 2, 1, 1 };
  static constexpr uint32_t kCloudNvdfJumpPassCount =
      sizeof(kCloudNvdfJumpSchedule) / sizeof(kCloudNvdfJumpSchedule[0]);
  // Runtime re-bake budget: JFA passes advanced per frame while a re-bake is
  // in flight (~0.2-0.5 ms each; the full chain completes in ~5 frames).
  static constexpr uint32_t kCloudNvdfJumpPassesPerFrame = 2;

  // Nubis3 detail volume (fork — Nubis3 conversion Phase B). 128^3 RGBA8
  // (~8 MB): R/G = low/high-frequency wispy, B/A = low/high-frequency
  // billowy. Baked once at init. Keep in lockstep with kDetailVolumeSize in
  // cloud_detail_noise_baker.comp.slang.
  static constexpr uint32_t kCloudDetailNoise3DSize = 128;

  // Secondary-ray cloud LUT (fork — 2026-06-10, perf). 256 azimuth x 128
  // elevation RGBA16F = 256 KB VRAM. Elevation rows concentrate near the
  // horizon (elevation = (pi/2)*v^2 — see cloudDomeUvToDir). Sized for
  // secondary-ray fidelity: indirect bounces and reflections are roughness-
  // filtered downstream, so ~1.4 degree azimuth texels are below the
  // perceptual floor there.
  static constexpr uint32_t kCloudSecondaryLutWidth  = 256;
  static constexpr uint32_t kCloudSecondaryLutHeight = 128;

  // Cloud placement map (fork — 2026-06-11, column-shaping rework). 512x512
  // RGBA8 = 1 MB VRAM, tiled at cloudNoiseTileKm (~23 m/texel at the 12 km
  // default — ample for ~2 km cloud clusters). Keep in lockstep with
  // kPlacementMapSize in cloud_placement_map_baker.comp.slang.
  static constexpr uint32_t kCloudPlacementMapSize = 512;

  // Scale heights for exponential density profiles (in km)
  static constexpr float kRayleighScaleHeight = 8.0f;
  static constexpr float kMieScaleHeight = 1.2f;

  Resources::Resource m_transmittanceLut;
  Resources::Resource m_multiscatteringLut;
  Resources::Resource m_skyViewLut;
  Resources::Resource m_cloudSkyTransmittanceLut;  // Fork: per-frame cloud occlusion of sky-ambient hemisphere
  // Cloud voxel grids (Nubis Cubed 2023, fork — 2026-05-12). Round-robin baked
  // every 8 frames by dispatchCloudSunDensityGrid / dispatchCloudAmbientDensityGrid.
  Resources::Resource m_cloudDSun;
  Resources::Resource m_cloudDAmbient;
  // Cloud NVDF bake chain resources (fork — Nubis3 conversion Phase A).
  // Occupancy + JFA ping-pong are bake scratch; the SDF pair is the
  // published product (front = last complete bake, back = in-flight bake;
  // stepCloudNvdfBake swaps after the resolve pass so consumers never read
  // a half-baked field).
  Resources::Resource m_cloudNvdfOccupancy;
  Resources::Resource m_cloudNvdfJfa[2];
  Resources::Resource m_cloudNvdfSdf[2];
  uint32_t            m_cloudNvdfSdfFront = 0;
  // Nubis3 wispy/billowy detail volume (fork — Nubis3 conversion Phase B).
  Resources::Resource m_cloudDetailNoise3D;
  // Cloud render RT (Nubis Cubed 2023, fork — 2026-05-12, C4). Screen-space
  // RGBA16F at downscale extent; produced each frame by dispatchCloudRender.
  // m_cloudRenderExtent tracks the current allocation so resize triggers a
  // realloc inside ensureCloudRenderRT.
  Resources::Resource m_cloudRenderRT;
  VkExtent2D          m_cloudRenderExtent = { 0u, 0u };
  // Full downscale (DLSS-input) extent the cloud RT is composited into —
  // the RT itself may be allocated smaller (cloudRenderResolutionScale,
  // fork — 2026-06-11). Published to shaders via
  // args.cloudRenderFullDimX/Y for the bilinear upsample at sky-miss.
  VkExtent2D          m_cloudRenderFullExtent = { 0u, 0u };

  // Secondary-ray cloud LUT (fork — 2026-06-10, perf). 256x128 RGBA16F,
  // baked every frame by dispatchCloudSecondaryLut.
  RtxMipmap::Resource m_cloudSecondaryLut;

  // Cloud placement map (fork — 2026-06-11, column-shaping rework). 512x512
  // RGBA8, baked at init + on input change by dispatchCloudPlacementMapBake.
  Resources::Resource m_cloudPlacementMap;

  // Per-frame camera basis for cloud_render.comp.slang. Pushed via
  // setCloudRenderCameraBasis() from updateAtmosphereConstants before
  // computeLuts runs; read by getAtmosphereArgs() into m_constantsBuffer.
  Vector3  m_cloudRenderForwardYUp { 0.0f, 0.0f, 1.0f };
  Vector3  m_cloudRenderRightYUp   { 1.0f, 0.0f, 0.0f };
  Vector3  m_cloudRenderUpYUp      { 0.0f, 1.0f, 0.0f };
  uint32_t m_cloudRenderFrameIdx   { 0u };
  RtxFastNoise m_fastNoise;            // EA Importance-Sampled FAST noise (cloud ray-march jitter)

  // Per-frame camera world position in Y-up km, for the C6 voxel-grid
  // cloud-on-terrain shadow plumbing. Pushed via setCloudShadowCameraPosition
  // from updateAtmosphereConstants; read by getAtmosphereArgs() into
  // m_constantsBuffer.cameraWorldPosYUpKm. Default (0,0,0) is safe: the
  // helper is gated off by default so the field is unused unless the user
  // flips cloudVoxelShadowsEnable, by which point the setter will have run
  // at least one frame.
  Vector3  m_cameraWorldPosYUpKm   { 0.0f, 0.0f, 0.0f };

  // Unified cloud-motion accumulators (fork — 2026-06-21). Integrated once per
  // frame by advanceCloudMotion() (offset += velocity * dt) and read by the const
  // getAtmosphereArgs() into cloudWindOffset / cloudEvolutionOffset* / cloudBoilPhase.
  // Persistent integration (vs the old stateless speed*timeSeconds) is what lets
  // the slow weather drift vary wind speed/direction without snapping the field.
  Vector2  m_cloudAdvectOffset     { 0.0f, 0.0f };  // wind translation (km)
  Vector3  m_cloudEvolutionOffset  { 0.0f, 0.0f, 0.0f };  // morph scroll (km)
  float    m_cloudBoilPhase        { 0.0f };  // edge-boil scroll phase (km)

  // Lightning strike scheduler state (fork — 2026-07-14). Advanced once per
  // frame by advanceLightning(); published by getAtmosphereArgs() into the
  // lightning CB fields. See the scheduler comment in rtx_atmosphere.cpp for
  // the envelope / restrike / inter-arrival model.
  Vector3  m_lightningStrikePosKm       { 0.0f, 0.0f, 0.0f };
  float    m_lightningEnvelope          { 0.0f };   // raw flicker envelope [0..~1.2]
  float    m_lightningHistoryFade       { 0.0f };   // ghost-suppression window (outlasts the envelope)
  int      m_lightningPulsesLeft        { 0 };      // restrike pulses of the active flash
  float    m_lightningTimeToPulse       { 0.0f };   // seconds to the next restrike pulse
  uint32_t m_lightningRngState          { 0x9E3779B9u };  // xorshift32 state
  static std::atomic<bool> s_lightningStrikeRequested;    // ImGui Test Strike latch

  // Cloud history ping-pong (fork). Screen-space RGBA16F (premultiplied
  // radiance, alpha) used by the temporal-smoothing path inside
  // evalSkyRadiance. Allocated lazily at downscaled render extent; recreated
  // on resize. m_cloudHistorySwap toggles each frame; getCurrentCloudHistory
  // is the WRITE target (this frame's accumulator), getPreviousCloudHistory
  // is the READ source (last frame's accumulator).
  Resources::Resource m_cloudHistory[2];
  // R16_UINT companion ping-pong (fork — 2026-05-13) carrying the frame index
  // at which each pixel of m_cloudHistory was last refreshed by the sky-miss
  // path. Read by evalSkyRadiance's age-check disocclusion guard to reject
  // stale history at pixels that were foreground-occluded last frame (their
  // m_cloudHistory slot retains pre-occlusion radiance because nothing
  // refreshes it). Without this, the alpha-only guard misidentifies stale
  // bright values as valid history and produces ~30-frame ghost trails.
  // Same extent/lifecycle as m_cloudHistory; cleared to 0xFFFF "never written"
  // sentinel at allocation.
  Resources::Resource m_cloudHistoryFrameId[2];
  VkExtent3D          m_cloudHistoryExtent = { 0u, 0u, 0u };
  bool                m_cloudHistorySwap = false;
  // Frame ID at which the swap last advanced. UINT32_MAX is a sentinel meaning
  // "never advanced yet" (first frame keeps swap = false; subsequent frames
  // toggle).
  uint32_t            m_cloudHistoryLastFrameId = UINT32_MAX;

  Rc<DxvkBuffer> m_constantsBuffer;

  AtmosphereArgs m_cachedArgs;
  // Split LUT cache keys (fork — 2026-06-11, perf). Each sky LUT bake gets a
  // cache key normalized down to the fields it actually reads, so a per-frame
  // starRotation push no longer re-bakes anything and a moving sun re-bakes
  // only the sky-view LUT. Used when skyLutCacheKeySplitEnable is on;
  // m_cachedArgs above remains the legacy monolithic key for the off path.
  AtmosphereArgs m_cachedSkyViewKey = {};
  AtmosphereArgs m_cachedTransmittanceMsKey = {};
  // Voxel-grid re-bake key (fork — 2026-06-11, perf). Wind / camera motion
  // quantized by cloudVoxelGridRebakeGranularityKm; see
  // normalizeForVoxelGridKey. Zero-init forces a first-frame bake; the
  // cloud-noise re-bake path also zeroes it to force same-frame refresh.
  AtmosphereArgs m_cachedVoxelGridKey = {};
  // Cloud noise 3D re-bake gate (fork). The 256^3 noise volume bakes its
  // Cloud placement map re-bake gate (fork — 2026-06-11, column-shaping
  // rework). Same pattern as the noise gate above: snapshot the last-baked
  // inputs, re-bake only on actual change.
  float    m_cachedPlacementCellSizeKm = 0.0f;
  float    m_cachedPlacementTileKm     = 0.0f;
  // Cloud NVDF re-bake gate + state machine (fork — Nubis3 Phase A). Same
  // snapshot pattern as the placement gate above. Deliberately NOT keys:
  // live weather coverage (sample-time level-set offset — but see the
  // quantized nominal below, which re-centers the bake when live coverage
  // drifts > half a quantization step from it), wind / evolution / boil
  // (sample-time translations of the erosion field), type / density / sun
  // (never shape the body). cloudThickness quantizes to 0.25 km so the slow
  // weather-drift blend can't rebake-storm.
  struct CloudNvdfBakeKey {
    float cellSizeKm       = 0.0f;
    float tileKm           = 0.0f;
    float columnFeather    = 0.0f;
    float columnTopShape   = 0.0f;
    float columnTopVar     = 0.0f;
    float columnBaseVar    = 0.0f;
    float nominalCoverage  = 0.0f;
    float thicknessQ       = 0.0f;  // cloudThickness quantized to 0.25 km
    float bodyErosion      = 0.0f;  // nvdfBodyErosionStrength (bake-time carve)
  };
  CloudNvdfBakeKey m_cachedNvdfKey = {};
  // In-flight runtime re-bake: index into kCloudNvdfJumpSchedule of the next
  // jump pass to run. Inactive when m_nvdfBakeActive is false. The full-chain
  // init path never touches these.
  bool     m_nvdfBakeActive = false;
  uint32_t m_nvdfJumpIdx    = 0;
  bool m_initialized = false;
  bool m_lutsNeedRecompute = true;
};

} // namespace dxvk
