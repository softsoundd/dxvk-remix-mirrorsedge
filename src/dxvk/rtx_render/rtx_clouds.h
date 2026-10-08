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
#pragma once

#include "rtx_resources.h"
#include "rtx_common_object.h"
#include "rtx_option.h"
#include "rtx/pass/clouds/cloud_args.h"
#include "rtx/pass/atmosphere/atmosphere_args.h"
#include <functional>

namespace dxvk {

class DxvkContext;
class DxvkDevice;
class RtxContext;
class RtCamera;

// Cost against fidelity of the cloud march, its lighting and its bakes (rtx.clouds.quality).
enum class CloudQuality : int {
  Low = 0,
  Medium,
  High,
  Ultra,
};

// rtx.clouds.genus: shape and microphysics presets. Custom uses the individual options.
enum class CloudGenus : int {
  Custom = 0,
  CumulusHumilis,    // Fair weather cumulus: small, flat based, well separated
  CumulusMediocris,  // Moderate cumulus with domed tops
  CumulusCongestus,  // Towering cumulus
  Stratocumulus,     // A broken low deck under an inversion: flat tops, lumpy bases
  Altocumulus,       // A thin mid level layer of small cells
  Stratus,           // A low, nearly unbroken grey sheet with a flat base
};

/**
 * \brief Volumetric cloud layer on the Physical Atmosphere.
 *
 * A Nubis Cubed style field (a body signed distance field under a hex de-tiled detail erosion) of water
 * droplet clouds, lit from their Mie optics by the atmosphere's sun and sky and seen through its aerial
 * perspective.
 *
 * Per frame, after the atmosphere's LUTs, whichever bakes their inputs or the camera's movement call for: the
 * body field, the sun and vertical optical depth grids around the camera, the far field shadow map, the sky's
 * harmonics at cloud altitude, the sky aerial perspective LUT and the reflection dome. After the path tracer,
 * before composite: the screen march. The path tracer reads the sun grid (shadows on the scene), the dome (sky
 * misses) and the far field shadow map through the common ray tracing bindings.
 */
class RtxClouds : public CommonDeviceObject {
public:
  explicit RtxClouds(DxvkDevice* device);

  /**
   * \brief Whether clouds render this frame: enabled under the Physical Atmosphere.
   */
  static bool isActive();

  /**
   * \brief This frame's cloud parameters, and every bake that is due. Must follow RtxAtmosphere::computeLuts.
   * Returns arguments with enabled = 0 when the clouds are off.
   */
  CloudArgs update(
    RtxContext& ctx, const AtmosphereArgs& atmosphere, const RtCamera& camera, const VkExtent3D& renderExtent, uint32_t debugView);

  /**
   * \brief The cloud layer along every camera ray, bounded by the scene. Needs the G-buffer and the ray tracing
   * constants of this frame; composite reads its output.
   */
  void dispatchScreen(RtxContext& ctx, const Resources::RaytracingOutput& rtOutput);

  /**
   * \brief The clouds of near-mirror reflections into the sky, marched along the rays the indirect integrator
   * stashed and put in place of the dome's in its radiance. Between the indirect integrator and the NEE pass.
   */
  void dispatchGlossyReflections(RtxContext& ctx, const Resources::RaytracingOutput& rtOutput);

  // This frame's CloudConstants (cloud then atmosphere arguments).
  Rc<DxvkBuffer> getConstantsBuffer() const { return m_constantsBuffer; }
  Rc<DxvkImageView> getSunGridView() const { return m_sunGrid.view; }
  Rc<DxvkImageView> getDomeView() const { return m_dome[m_domeIndex].view; }
  Rc<DxvkImageView> getShadowMapView() const { return m_shadowMap.view; }
  // This frame's layer for composite (the reference's mean in reference mode), or nullptr when the screen march
  // did not run.
  Rc<DxvkImageView> getLayerView() const {
    if (!m_screenRanThisFrame) {
      return nullptr;
    }
    return m_referenceRanThisFrame ? m_referenceMean.view : m_compositeLayer.view;
  }
  // This frame's layer along mirrors' and glass's first reflections into the sky, likewise.
  Rc<DxvkImageView> getReflectionView() const { return m_screenRanThisFrame ? m_reflection[m_layerIndex].view : nullptr; }
  Rc<DxvkImageView> getGlossyRayView() const { return m_glossyRay.view; }

  // Optical properties of the droplets at the current microphysics, for the UI's readout.
  struct DropletReadout {
    float effectiveRadiusUm;
    float extinctionPerKm;     // At the reference height, 1 km above the base
    float asymmetry;
    float forwardPeakFraction;
    float truncatedAsymmetry;
    float verticalOpticalDepth;  // From the base to the capped top at the field's full density
  };
  static DropletReadout computeDropletReadout();

  // The eye's altitude on the atmosphere's datum, as of the last update, for the UI.
  static float getEyeAltitudeKm() { return m_eyeAltitudeKm; }

  // The march against the reference over the opaque 4x4 tiles, as of a few frames ago, for the UI.
  struct ReferenceErrorReadout {
    uint32_t tiles = 0;
    float pathsPerTile = 0.0f;
    float meanAbsolute = 0.0f;  // Mean |march / reference - 1|
    float percentile95 = 0.0f;  // 95th percentile of the same
    float bias = 0.0f;          // The march's mean over the reference's, less 1: positive where the march is brighter
  };
  static const ReferenceErrorReadout& getReferenceErrorReadout() { return m_referenceErrorReadout; }

  static void showImguiSettings(uint32_t sliderFlags, uint32_t collapsingHeaderFlags);

  // Options. Shape options marked (genus) are overridden by a genus other than Custom.
  RTX_OPTION_ARGS("rtx.clouds", bool, enable, true,
    "Render a volumetric cloud layer. Requires the Physical Atmosphere sky mode.",
    args.flags = RtxOptionFlags::UserSetting);
  RTX_OPTION_ARGS("rtx.clouds", CloudQuality, quality, CloudQuality::High,
    "Cost of the cloud march, its lighting and its bakes: Low (0), Medium (1), High (2), Ultra (3).",
    args.minValue = CloudQuality::Low, args.maxValue = CloudQuality::Ultra, args.flags = RtxOptionFlags::UserSetting);
  RTX_OPTION_FLAG("rtx.clouds", bool, reference, false, RtxOptionFlags::NoSave,
    "Replace the cloud layer with a progressive path traced reference of the same field and lighting, for judging the\n"
    "march's lighting model (debug views Clouds Reference and Clouds Reference Error). Very expensive; the wind stops while\n"
    "it accumulates. This will not save.");
  RTX_OPTION("rtx.clouds", uint32_t, referenceMaxBounces, 16384, "Collisions after which a reference path ends, a safety net: paths otherwise run until they leave the layer.");
  RTX_OPTION("rtx.clouds", uint32_t, referenceExactBounces, 4,
    "Collisions a reference path scatters by the droplets' exact Mie phase before it continues in the delta-M similar\n"
    "medium, which needs a third of the collisions. Raise it for a strictly exact, slower reference.");
  RTX_OPTION("rtx.clouds", uint32_t, referenceBouncesPerFrame, 16, "Collisions each 4x4 pixel tile's reference path advances by per frame: the reference's cost per frame.");
  RTX_OPTION_ARGS("rtx.clouds", CloudGenus, genus, CloudGenus::CumulusMediocris,
    "Shape and microphysics preset: Custom (0), Cumulus Humilis (1), Cumulus Mediocris (2), Cumulus Congestus (3),\n"
    "Stratocumulus (4), Altocumulus (5), Stratus (6). Presets override the options marked (genus).",
    args.minValue = CloudGenus::Custom, args.maxValue = CloudGenus::Stratus);

  // Layer (genus)
  RTX_OPTION("rtx.clouds", float, baseAltitudeMeters, 1300.0f, "(genus) Altitude of the cloud base above sea level, on the same datum as rtx.atmosphere.altitude.");
  RTX_OPTION("rtx.clouds", float, thicknessMeters, 2500.0f, "(genus) Depth of the cloud layer.");
  RTX_OPTION("rtx.clouds", float, coverage, 0.45f, "(genus) Fraction of the sky the layer covers, 0 to 1.");
  RTX_OPTION("rtx.clouds", float, coverageSpread, 0.0f, "Amplitude of the large scale variation of the coverage.");
  RTX_OPTION("rtx.clouds", float, coverageSpreadScaleKm, 16.0f, "Size of the patches over which the coverage varies: denser and sparser clusters of clouds.");
  RTX_OPTION("rtx.clouds", float, horizonBias, 0.0f,
    "Shifts the cloud cover between overhead and the horizon by shrinking the clouds on one side: above 0 those overhead\n"
    "(1 clears them, leaving clouds only towards the horizon, like a skybox), below 0 those towards the horizon (-1 leaves\n"
    "them only overhead).");
  RTX_OPTION("rtx.clouds", float, horizonBiasStartKm, 5.0f, "Horizontal distance from the camera within which the horizon bias's overhead side applies in full.");
  RTX_OPTION("rtx.clouds", float, horizonBiasEndKm, 30.0f, "Horizontal distance from the camera beyond which its horizon side applies in full; between the two it blends.");
  RTX_OPTION("rtx.clouds", float, cloudType, 0.75f, "(genus) Erosion character: 0 wispy, 1 billowy.");
  RTX_OPTION("rtx.clouds", float, typeSpread, 0.54f, "Amplitude of the large scale variation of the type.");
  RTX_OPTION("rtx.clouds", float, typeSpreadScaleKm, 3.0f, "Size of the patches over which the type varies; about the cloud spacing gives each cloud its own character.");
  RTX_OPTION("rtx.clouds", float, cellSizeKm, 3.65f, "(genus) Typical spacing of individual clouds.");
  RTX_OPTION("rtx.clouds", float, columnTopShape, 0.4f, "(genus) Exponent from a column's presence to its top: below 1, edges rise steeply into domed tops.");
  RTX_OPTION("rtx.clouds", float, columnTopVariation, 0.45f, "(genus) Random spread of the tops of equally present clouds.");
  RTX_OPTION("rtx.clouds", float, columnBaseVariation, 0.12f, "(genus) Undulation of the cloud base, as a fraction of the layer.");
  RTX_OPTION("rtx.clouds", float, columnFeather, 0.35f, "(genus) Width of the band over which a cloud's edge feathers in.");
  RTX_OPTION("rtx.clouds", float, columnTopFlatten, 1.0f, "(genus) Caps the tops at this fraction of the layer above each base, as an inversion does. 1 = no cap.");

  // Microphysics
  RTX_OPTION("rtx.clouds", float, liquidWaterContent, 0.55f, "(genus) Liquid water content 1 km above the base, g/m^3. Typical: 0.2 to 0.4 stratiform, 0.5 to 1 cumulus.");
  RTX_OPTION("rtx.clouds", float, dropletConcentration, 350.0f, "(genus) Droplet number concentration, per cm^3. About 100 maritime, 300 to 600 continental and urban.");
  RTX_OPTION("rtx.clouds", float, adiabaticExponent, 0.667f, "Extinction grows as height above the base to this power, 2/3 for an adiabatic parcel (Brenguier et al. 2000).");
  RTX_OPTION("rtx.clouds", float, adiabaticFloor, 0.15f, "Least extinction of the vertical profile relative to its value 1 km above the base.");
  RTX_OPTION("rtx.clouds", float, adiabaticMax, 2.0f, "Greatest extinction of the vertical profile relative to its value 1 km above the base.");
  RTX_OPTION("rtx.clouds", float, densityScale, 1.0f, "Stylisation: multiplies the physical extinction. 1 = physical.");

  // Field
  RTX_OPTION("rtx.clouds", float, tileKm, 12.0f, "Horizontal period of the body field and its detail, hidden by the hex lattice.");
  RTX_OPTION("rtx.clouds", bool, hexTiling, true, "De-tile the body field with a hex lattice of randomly transformed tiles (Heitz and Neyret 2018).");
  RTX_OPTION("rtx.clouds", float, detailScale, 12.0f, "Detail volume repeats per tile.");
  RTX_OPTION("rtx.clouds", float, nominalCoverage, 0.0f, "Coverage the body field is baked at; 0 follows the live coverage (re-baked as it changes).");
  RTX_OPTION("rtx.clouds", float, coverageOffsetKm, 0.2f, "Level set shift per unit of coverage away from the baked nominal.");
  RTX_OPTION("rtx.clouds", float, profileDepthKm, 0.7f, "Depth below the surface over which the density ramps to full.");
  RTX_OPTION("rtx.clouds", float, bodyErosion, 1.5f, "Height varying carve of the baked bodies, for overhangs and lumps.");
  RTX_OPTION("rtx.clouds", float, stepScale, 0.95f, "Safety factor of the empty space skipping on the body distance field; 0 disables it.");
  RTX_OPTION("rtx.clouds", float, erosionStrength, 0.58f, "Value erosion of the body by the detail volume.");
  RTX_OPTION("rtx.clouds", float, sharpenStrength, 1.0f, "Lifts low densities by an exponent below 1, defining wisps and edges.");
  RTX_OPTION("rtx.clouds", float, flyThroughDetail, 0.62f, "High frequency detail within 600 m of the camera.");
  RTX_OPTION("rtx.clouds", float, wobbleStrength, 0.0f, "Zero mean displacement of the silhouette by the detail volume.");
  RTX_OPTION("rtx.clouds", float, interiorTexture, 0.0f, "Density variation through the body, not only at its skin.");
  RTX_OPTION("rtx.clouds", float, edgeErosion, 0.0f, "Wispy strands cut through the outer shell.");
  RTX_OPTION("rtx.clouds", float, fineDetailStrength, 0.0f, "A finer erosion band, faded out between 3 and 9 km.");
  RTX_OPTION("rtx.clouds", float, edgeDetail, 1.0f, "Fine detail along wispy clouds' edges, cutting strands into the rim and drawing others out past it; none on billowy erosion, faded out between 10 and 16 km.");
  RTX_OPTION("rtx.clouds", float, shapeVarietyKm, 1.35f, "Amplitude of the mid frequency lobes that break bodies into clusters, at most 0.65 x their wavelength (beyond that the surface folds).");
  RTX_OPTION("rtx.clouds", float, shapeVarietyWavelengthKm, 2.1f, "Wavelength of those lobes.");
  RTX_OPTION("rtx.clouds", float, curlStrengthMeters, 60.0f, "Domain warp of the erosion field (Schneider and Vos 2015's curl distortion), strongest at the base.");
  RTX_OPTION("rtx.clouds", float, nearDetailStrength, 0.5f, "An extra fine erosion octave close to the camera, where the detail volume's texels would show.");
  RTX_OPTION("rtx.clouds", float, nearDetailRangeKm, 3.0f, "Distance over which that octave fades out.");
  RTX_OPTION("rtx.clouds", float, detailLodBias, -1.0f, "Mip bias of the detail volume's pixel footprint filtering; lower keeps more detail at distance.");

  // Motion
  RTX_OPTION("rtx.clouds", bool, motion, true, "Move the clouds: the wind carries them and they rise and shear as they travel. Off holds them where they are.");
  RTX_OPTION("rtx.clouds", float, windSpeed, 10.0f, "Wind speed at cloud level, m/s: 5 to 15 is typical of the lower troposphere. The field moves with it.");
  RTX_OPTION("rtx.clouds", float, windDirection, 45.0f, "Direction the wind blows toward, degrees (0 = +X, 90 = +Z in the atmosphere's frame).");
  RTX_OPTION("rtx.clouds", float, evolutionRise, 2.0f, "Convective rise of the detail field, m/s: clouds boil.");
  RTX_OPTION("rtx.clouds", float, evolutionShear, 2.0f, "Downwind shear of the detail field's tops, m/s.");
  RTX_OPTION("rtx.clouds", float, detailBaseShearKm, 0.2f, "Static downwind lean of the detail at the base.");

  // Lighting. The octave and diffusion defaults are fitted to Monte Carlo transport through water cloud slabs.
  RTX_OPTION("rtx.clouds", uint32_t, multipleScatteringOctaves, 0, "Octaves of the higher scattering orders, 1 to 4; 0 = the quality tier's.");
  RTX_OPTION("rtx.clouds", float, msExtinctionFalloff, 0.3f, "a: extinction scale per octave (Wrenninge et al. 2013).");
  RTX_OPTION("rtx.clouds", float, msEnergyFalloff, 0.4f, "b: contribution per octave.");
  RTX_OPTION("rtx.clouds", float, msPhaseFalloff, 0.5f, "c: asymmetry scale per octave.");
  RTX_OPTION("rtx.clouds", float, diffusionFloor, 1.0f, "Weight of the diffusion floor under the octaves, the scattered sunlight diffusing through the clouds, which carries the light deep inside thick cloud.");
  RTX_OPTION("rtx.clouds", float, diffusionAnisotropy, 1.0f, "Weight of the diffusion field's flux term (Eddington's 3 mu): faces the diffused light leaves by glow brighter than those it runs along. 1 = Eddington.");
  RTX_OPTION("rtx.clouds", float, diffuseTransmissionK, 0.75f, "k of the diffuse transmittance 1 / (1 + k (1 - g) tau) of sky light into the cloud; 3/4 is Eddington's.");
  RTX_OPTION("rtx.clouds", float, ambientStrength, 1.0f, "Scale of the sky and ground light on the clouds. 1 = physical.");
  RTX_OPTION("rtx.clouds", float, groundBounceStrength, 1.0f, "Scale of the ground's light on the cloud bases. 1 = physical.");
  RTX_OPTION("rtx.clouds", float, groundDiffuseShare, 0.35f, "Share of the shadowed sunlight the ground still receives diffusely through the clouds.");
  RTX_OPTION("rtx.clouds", Vector3, albedoTint, Vector3(1.0f, 1.0f, 1.0f), "Stylisation: tints the scattered light. 1 = physical (water barely absorbs in the visible).");
  RTX_OPTION("rtx.clouds", float, shadowTapRangeMeters, 250.0f, "Range of the full resolution taps toward the sun ahead of the optical depth grid.");

  // Coupling with the scene and the atmosphere
  RTX_OPTION("rtx.clouds", bool, sunShadows, true, "Cloud shadows on the scene: the atmosphere's sun is attenuated by the layer.");
  RTX_OPTION("rtx.clouds", float, shadowStrength, 1.0f, "Blend of the cloud shadows on the scene and the air. 1 = physical.");
  RTX_OPTION("rtx.clouds", bool, airShadows, true, "Cloud shadows in the air: the aerial perspective and fog beneath a broken deck.");
  RTX_OPTION("rtx.clouds", bool, skyAerialPerspective, true, "Composite the clouds through the air in front of them (the sky aerial perspective LUT).");
  RTX_OPTION("rtx.clouds", bool, reflectionDome, true, "Clouds in sky reflections and indirect sky light, from a dome rendered from the camera.");
  RTX_OPTION("rtx.clouds", bool, mirrorReflectionMarch, true, "March the clouds per pixel along mirrors' and glass's first reflections into the sky, and past glass the sky is seen straight through, instead of reading the dome.");
  RTX_OPTION("rtx.clouds", bool, glossyReflectionMarch, true, "March the clouds per pixel along near-mirrors' reflections into the sky (surfaces a little too rough for PSR, such as polished floors and curtain walls), instead of reading the dome. Costs a march for each such pixel.");
  RTX_OPTION("rtx.clouds", float, maxMarchKm, 160.0f, "Longest distance the march follows a ray through the layer.");
private:
  struct TierSettings {
    uint32_t viewSamplesMax;
    float viewStepKm;
    float adaptiveStepKm;
    float opticalDepthPerStep;
    float exitTransmittance;
    uint32_t shadowTaps;
    uint32_t gridInterleave;
    uint32_t domeWidth;
    uint32_t domeInterleave;
    float historyBlend;
    float domeBlend;  // Per refresh of a texel
    uint32_t msOctaves;
    uint32_t diffusionSweeps;  // Of the diffusion floor's solve, each frame its grid changes
  };
  static TierSettings getTierSettings(CloudQuality quality);

  struct GenusSettings {
    float baseAltitudeMeters;
    float thicknessMeters;
    float coverage;
    float cloudType;
    float cellSizeKm;
    float columnTopShape;
    float columnTopVariation;
    float columnBaseVariation;
    float columnFeather;
    float columnTopFlatten;
    float liquidWaterContent;
    float dropletConcentration;
  };
  static GenusSettings getGenusSettings();

  // Inputs of the body field's bake; a change starts a re-bake.
  struct NvdfKey {
    float cellSizeKm = -1.0f;
    float tileKm = -1.0f;
    float columnFeather = -1.0f;
    float columnTopShape = -1.0f;
    float columnTopVariation = -1.0f;
    float columnBaseVariation = -1.0f;
    float columnTopFlatten = -1.0f;
    float nominalCoverage = -1.0f;
    float thicknessQuantised = -1.0f;
    float bodyErosion = -1.0f;
    bool operator==(const NvdfKey& other) const;
  };
  static NvdfKey makeNvdfKey(const CloudArgs& args);

  CloudArgs buildArgs(const AtmosphereArgs& atmosphere, const RtCamera& camera, const VkExtent3D& renderExtent, uint32_t debugView);
  void advanceMotion();

  // The resources the bakes, the passes and the common bindings read, and the static noise and tables, on the
  // first frame the clouds are on after being off.
  void initialize(Rc<DxvkContext> ctx);
  void createResources(Rc<DxvkContext> ctx);
  void ensureDome(Rc<DxvkContext> ctx, uint32_t width);
  void ensureScreenResources(Rc<DxvkContext> ctx, const VkExtent3D& extent);
  void ensureReferenceResources(Rc<DxvkContext> ctx);
  void releaseReferenceResources();
  // Everything initialize, the dome and the screen targets allocate, and the state describing it.
  void releaseResources();
  void uploadPhaseLut(Rc<DxvkContext> ctx);

  void barrier(Rc<DxvkContext> ctx);
  void bindCloudInputs(Rc<DxvkContext> ctx, const Rc<DxvkBuffer>& constants);
  void bakePlacement(Rc<DxvkContext> ctx);
  void bakeDetailNoise(Rc<DxvkContext> ctx);
  void startNvdfBake(Rc<DxvkContext> ctx, const CloudArgs& args);
  void stepNvdfBake(Rc<DxvkContext> ctx, uint32_t passBudget);
  void dispatchNvdfJfa(Rc<DxvkContext> ctx, uint32_t mode, uint32_t jumpSize, uint32_t source, uint32_t destination);
  // A bake's texels: an interleave phase, a strip a moving window brings in (period 1, from phase along the
  // interleaved axis and offset along the other), or all of them.
  struct BakeRegion {
    uint32_t period;
    uint32_t phase;
    uint32_t offset;
    uint32_t interleavedCount;
    uint32_t otherCount;
  };
  // What a world anchored bake (a grid cascade, the shadow map) holds: the key of the inputs its interleave last
  // restarted from, its window's origin in texels, and the phases baked since.
  struct WorldBakeState {
    uint64_t inputsKey = 0;
    int64_t interleavedOrigin = 0;
    int64_t otherOrigin = 0;
    uint32_t phasesBaked = 0;
    bool valid = false;
  };
  // Bakes what of a world anchored target is stale: the strips its window moved over, and the interleave until
  // every texel has been baked from the current inputs. Returns whether it wrote anything.
  bool updateWorldBake(
    WorldBakeState& state, uint64_t inputsKey, int64_t interleavedOrigin, int64_t otherOrigin, uint32_t size, uint32_t period,
    const std::function<void(const BakeRegion&)>& bake);
  // Of the arguments the bakes read, less what changes every frame without changing what they hold.
  static uint64_t computeInputsKey(const CloudArgs& args, const AtmosphereArgs& atmosphere);
  // Of what the sky light's sky and its ground under unit light read: neither the field nor its motion.
  static uint64_t computeSkyLightKey(const CloudArgs& args, const AtmosphereArgs& atmosphere);
  void bakeLightingGrids(Rc<DxvkContext> ctx, uint32_t cascade, const BakeRegion& region);
  void bakeShadowMap(Rc<DxvkContext> ctx, const BakeRegion& region);
  void bakeSunGridMips(Rc<DxvkContext> ctx);
  // The diffusion floor's solve over a cascade: its cells from the sun grid when that changed, then red-black sweeps.
  void bakeDiffusion(Rc<DxvkContext> ctx, uint32_t cascade, bool cellsChanged, uint32_t sweeps);
  // The sky light's harmonics in full, or with groundOnly only re-weighted by the ground's light under the clouds.
  void bakeSkySh(Rc<DxvkContext> ctx, bool groundOnly);
  void bakeSkyAp(Rc<DxvkContext> ctx, uint32_t period, uint32_t phase);
  void bakeDome(Rc<DxvkContext> ctx, uint32_t interleave);
  void dispatchReference(RtxContext& ctx, const Resources::RaytracingOutput& rtOutput);
  void readReferenceStatistics(uint32_t frameId);
  bool referenceInputsChanged(const RtCamera& camera, const CloudArgs& args, const AtmosphereArgs& atmosphere);

  Rc<DxvkSampler> getVolumeSampler(Rc<DxvkContext> ctx) const;
  Rc<DxvkSampler> getLutSampler(Rc<DxvkContext> ctx) const;

  Rc<DxvkBuffer> m_constantsBuffer;      // This frame's CloudConstants
  Rc<DxvkBuffer> m_nvdfConstantsBuffer;  // CloudConstants the running body field bake started from

  Resources::Resource m_placementMap;
  Resources::Resource m_detailNoise;
  std::vector<Rc<DxvkImageView>> m_detailNoiseMipViews;
  // The body field's jump flood, allocated only while a bake runs, and the published field with its back buffer.
  Resources::Resource m_nvdfOccupancy;
  Resources::Resource m_nvdfSeeds[2];
  Resources::Resource m_nvdfSdf[2];
  uint32_t m_nvdfFront = 0;
  Resources::Resource m_sunGrid;
  std::vector<Rc<DxvkImageView>> m_sunGridMipViews;
  Resources::Resource m_ambientGrid;
  // The far cascade of the two, over a wider window at more columns, for distant clouds.
  Resources::Resource m_sunGridFar;
  Resources::Resource m_ambientGridFar;
  // Per cascade, near then far: the diffusion floor's cells (Q, density) and fluence, and the sweeps left before
  // its solve has settled on the cascade's last bake.
  Resources::Resource m_diffusionCells[2];
  Resources::Resource m_diffusionFluence[2];
  uint32_t m_diffusionSettleSweeps[2] = {};
  // Sweeps the far cascade's solve still runs at the tier's pace after a restart, before it tracks a sweep a frame.
  uint32_t m_farDiffusionCatchUpSweeps = 0;
  Resources::Resource m_skyApInScatter;
  Resources::Resource m_skyApTransmittance;
  Resources::Resource m_skySh;
  // The sky's harmonics and the ground's under unit light, which m_skySh combines.
  Resources::Resource m_skyShParts;
  Resources::Resource m_phaseLut;
  Resources::Resource m_shadowMap;
  Resources::Resource m_dome[2];
  uint32_t m_domeIndex = 0;
  uint32_t m_domeWidth = 0;
  bool m_domeHistoryValid = false;
  // The filtered layer (each other's history, which marks the pixels the scene hides the clouds behind), the frames
  // each texel's history holds, and the layer for composite.
  Resources::Resource m_layer[2];
  Resources::Resource m_layerAge[2];
  Resources::Resource m_compositeLayer;
  // The layer along mirrors' and glass's reflections, alternating like the layer.
  Resources::Resource m_reflection[2];
  // The indirect integrator's near-mirror rays into the sky, zeroed by the glossy pass as it consumes them.
  Resources::Resource m_glossyRay;

  // Reference mode: the paths' sums, their mean for composite and each 4x4 tile's path in flight (position,
  // direction, throughput, radiance), allocated only while the reference runs; the droplets' inverse phase CDF;
  // and the frames accumulated since the view or the clouds last changed.
  Resources::Resource m_referenceAccumulation;
  Resources::Resource m_referenceMean;
  Resources::Resource m_referencePath[4];
  Resources::Resource m_phaseCdf;
  // The error statistics, a slot per frame in flight, and their host readable copy.
  Rc<DxvkBuffer> m_referenceStatistics;
  Rc<DxvkBuffer> m_referenceStatisticsReadback;
  uint32_t m_referenceFrames = 0;
  bool m_referenceRanThisFrame = false;
  Vector3 m_referenceCameraPosition = Vector3(0.0f);
  Vector3 m_referenceCameraDirection = Vector3(0.0f);
  CloudArgs m_referenceArgs = {};
  Vector3 m_referenceSunDirection = Vector3(0.0f);
  uint32_t m_layerIndex = 0;
  VkExtent3D m_screenExtent = { 0, 0, 0 };
  bool m_screenHistoryValid = false;
  bool m_screenRanThisFrame = false;

  bool m_initialized = false;

  // Body field bake state: the published field's key, and the amortised re-bake in flight.
  NvdfKey m_publishedNvdfKey;
  NvdfKey m_pendingNvdfKey;
  bool m_nvdfBakeActive = false;
  bool m_nvdfValid = false;
  uint32_t m_nvdfJumpIndex = 0;
  float m_publishedNominalCoverage = 0.0f;
  bool m_gridsNeedFullBake = true;

  // The far field shadow map bakes a quarter of its rows a frame.
  static constexpr uint32_t kShadowMapInterleave = 4;
  // The far grid cascade bakes a sixteenth of its columns a frame: distant clouds change slowly on screen.
  static constexpr uint32_t kFarGridInterleave = 16;
  // The sky aerial perspective LUT refreshes a quarter of its columns a frame once it has been baked in full.
  static constexpr uint32_t kSkyApInterleave = 4;
  bool m_skyApValid = false;
  // Refreshes of every dome texel after its inputs last changed, by which its history has converged.
  static constexpr uint32_t kDomeSettleRefreshes = 16;
  // Camera movement the sky aerial perspective LUT and the dome ignore: against clouds half a kilometre away it is a
  // tenth of a dome texel, and misses read the dome from anywhere in the scene anyway.
  static constexpr float kSkyCameraToleranceKm = 0.0005f;

  // What the bakes hold, so unchanged inputs (a still field: no wind, rise or shear) bake only what the camera's
  // movement brings into view.
  WorldBakeState m_nearGridState;
  WorldBakeState m_farGridState;
  WorldBakeState m_shadowMapState;
  uint64_t m_lastInputsKey = 0;
  // The inputs key without the field's evolution.
  uint64_t m_lastRestartKey = 0;
  uint64_t m_skyLightKey = 0;
  // The camera's place in the field the bakes last restarted from for the horizon bias, which follows the camera.
  Vector2 m_horizonBiasAnchorKm = Vector2(0.0f, 0.0f);
  bool m_horizonBiasAnchored = false;
  // Whether the constants say off, as passes outside the module read them while the clouds are off.
  bool m_offConstantsWritten = false;
  // Camera the sky aerial perspective LUT and the dome last restarted their refreshes for.
  vec3 m_skyCameraPositionKm = vec3(0.0f, 0.0f, 0.0f);
  float m_skyCameraWorldHeightKm = 0.0f;
  uint32_t m_skyApPhasesBaked = 0;
  uint32_t m_domeSettledFrames = 0;

  // Integrated motion, km, and each part's step over the last frame.
  Vector2 m_windOffsetKm = Vector2(0.0f, 0.0f);
  Vector2 m_windStepKm = Vector2(0.0f, 0.0f);
  float m_evolutionRiseKm = 0.0f;
  float m_evolutionShearKm = 0.0f;
  float m_riseStepKm = 0.0f;
  float m_shearStepKm = 0.0f;

  uint32_t m_frameIndex = 0;
  float m_loggedEyeAltitudeKm = -1e9f;
  CloudArgs m_args = {};
  CloudQuality m_lastQuality = CloudQuality::High;

  inline static float m_eyeAltitudeKm = 0.0f;
  inline static ReferenceErrorReadout m_referenceErrorReadout = {};
};

} // namespace dxvk
