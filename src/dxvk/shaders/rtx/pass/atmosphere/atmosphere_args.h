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

#include "rtx/utility/shader_types.h"

// Maximum number of independent moons the atmosphere system can render.
// Bumping requires a corresponding bump in the shader unrolling budget —
// currently 4 fits comfortably in the constant buffer and unrolls cleanly.
#define MAX_MOONS 4u

// Per-moon parameters. Hand-padded to 16-byte alignment.
struct MoonParams {
  // Pose (game-driven via NoSave RTX_OPTIONs)
  vec3 direction;          // Normalized direction in Y-up space
  float angularRadius;     // Half-angle in radians

  vec3 color;              // Base albedo
  float brightness;        // Overall radiance multiplier

  uint surfaceStyle;       // 0 = Rocky, 1 = Volcanic
  float phase;             // [0,1]: 0=new, 0.5=full
  float enabled;           // 1.0 = render, 0.0 = skip
  float craterDensity;     // [0,1] multiplier on crater contribution

  float surfaceContrast;   // Multiplier on surface light/dark variation
  float surfaceNoiseScale; // Multiplier on UV scale fed into surface noise
  float darkSideBrightness;// Fraction of lit radiance applied on dark side
  float roughnessAmount;   // Multiplier on micro-detail amplitude
};

// Atmosphere parameters for Hillaire physically-based atmospheric scattering
struct AtmosphereArgs {
  vec3 sunDirection;
  float planetRadius;  // in km

  vec3 sunIlluminance;
  float atmosphereThickness;  // in km

  vec3 rayleighScattering;
  float mieAnisotropy;  // Henyey-Greenstein phase function g parameter [-1, 1]

  vec3 mieScattering;
  float sunRayBrightness;  // Multiplier for direct sun ray brightness

  // Ozone absorption (important for realistic sunset colors per Hillaire paper Section 3.4)
  vec3 ozoneAbsorption;  // Absorption coefficients (km^-1)
  float ozoneLayerAltitude;  // Peak altitude of ozone layer (km)

  uint transmittanceLutWidth;
  uint transmittanceLutHeight;
  uint multiscatteringLutSize;
  uint skyViewLutWidth;

  uint skyViewLutHeight;
  float ozoneLayerWidth;  // Width of ozone layer (km)
  float padRetired10;     // retired: viewAltitude (camera altitude offset, km) — never read by any pass.
  float multiScatterPhysicalStrength;  // 0 = pure analytical (artistic, preset-faithful), 1 = pure LUT-based hemisphere integration (physical)

  // Derived parameters (computed on CPU)
  float atmosphereRadius;  // planetRadius + atmosphereThickness
  float rayleighScaleHeight;  // exponential density falloff for Rayleigh (km)
  float mieScaleHeight;  // exponential density falloff for Mie (km)
  float sunAngularRadius; // Sun angular radius in radians

  // ----- Night-sky additions (fork) -----
  float starBrightness;     // Overall star brightness multiplier
  float starDensity;        // Density threshold (0=all stars, 1=no stars)
  float starTwinkleSpeed;   // Animation rate
  float nightSkyBrightness; // Airglow / ambient night-sky brightness

  vec3 nightSkyColor;       // Base color of night-sky airglow
  float timeSeconds;        // Elapsed time for star twinkle animation

  // Sidereal sky rotation (axis-angle representation).
  // Default elevation=90 / rotation=0 puts the celestial pole at zenith,
  // and starRotation=0 leaves the star sample direction unchanged — preserving
  // original at-the-pole behavior. Games push starRotation per frame; the axis
  // fields are persistent and set once at startup or via rtx.conf.
  float starRotation;       // Sidereal angle, degrees [0, 360]
  float starAxisElevation;  // Celestial pole elevation from horizon, degrees
  float starAxisRotation;   // Celestial pole azimuth, degrees
  // Nubis3 up-rez result sharpen (fork — Nubis3 conversion; Phase B consumer).
  // Exponent-blend strength of the page-123 pow() sharpen on the eroded
  // density. Rides the former pad3 slot; CB layout unchanged. 0 until Phase B.
  float nubis3SharpenStrength;

  // ----- Star anti-aliasing + cloud interaction (fork) -----
  //
  // starPsfSharpness: exponent in the per-star Gaussian PSF inside evalStarField.
  // The hash-grid cube-face mapping (gridScale = 400 cells per face = ~13.5
  // arcmin/cell) was originally evaluated with a hardcoded exp(-dist² * 800) —
  // half-width ~0.029 cells = ~0.08 pixels at 1080p/90° FOV, well sub-pixel. As
  // the camera rotated, star centers crossed pixel boundaries discontinuously,
  // producing severe per-frame flicker even with DLSS disabled. Lowering the
  // exponent widens the PSF: k=20 gives a ~1-pixel-FWHM star at 1080p, k=8 at
  // 720p. Sized to anti-alias at typical FNV render resolutions (~720p–1440p
  // including DLSS internal resolutions). Lower = bigger softer stars;
  // higher = sharper pinpoints (with more flicker).
  //
  // Stars are also very bright HDR point sources (peak ~= starBrightness,
  // default 8). The standard alpha-composite (stars * (1-cloudOpacity)) attenuates
  // them by cloud view-transmittance T = 1-opacity, but cumulus cores rarely
  // reach T<0.05 and stars at T=0.05 still read as bright pinpoints over a dim
  // moonlit cloud (~0.1). Two corrections, mirroring the moon's "integrated"
  // look:
  //   - starCloudExtinctionPower raises the cloud_T extinction applied to stars
  //     to a power > 1, so star contribution dies as T^k rather than T. At k=2.5
  //     and T=0.05, stars get 0.05^2.5 = 0.00056 -- well below cloud body level.
  //     Clear sky (T=1) is unaffected (1^k = 1).
  //   - starAmbientCouplingStrength couples sky airglow brightness into the
  //     cloud-march nightLight term, analogous to moon-zenith fill. Brightens
  //     cloud bodies under starry skies so they visually compete with the
  //     bright HDR stars and don't read as "floating dots on a dim cloud."
  float starPsfSharpness;               // PSF exponent for evalStarField (default 20.0; was hardcoded 800)
  float starCloudExtinctionPower;       // Power exponent on cloud view-T when extincting stars (default 2.5)
  float starAmbientCouplingStrength;    // Star/airglow coupling into cloud nightLight (default 0.01)
  float cloudViewSamplesMax;            // Cap on the adaptive cloud-march step count (fork —
                                        // 2026-06-12, adaptive march sampling). Reuses the former
                                        // padStarCloud0 slot; CB layout unchanged.

  // ----- Milky Way controls (fork) -----
  // The galactic band is two independent visual layers: (1) increased star
  // density inside the band region (drives the "thick with stars" look) and
  // (2) a diffuse background glow (the real Milky Way's dust haze). Each is
  // independently tunable; both gate on milkyWayEnabled.
  //
  // The prior implementation hardcoded the density boost at 0.15 (galacticDensity
  // * 0.15 reduces the local star threshold), which combined with the brightness
  // remap quirk -- (starMag - args.starDensity) producing large negative values
  // for band-only stars, then squared into massive brightness -- made band stars
  // explosively bright. The refactored evalStarField uses localThreshold as the
  // brightness-remap floor instead of args.starDensity, so band stars sit in a
  // physically-sensible brightness range and the user controls "how many" and
  // "how bright" via these knobs without the explosion.
  float milkyWayEnabled;                // 1.0 = render galactic-band effects, 0.0 = uniform star field
  float milkyWayDensityBoost;           // Threshold reduction inside the band (default 0.05; was hardcoded 0.15)
  float milkyWayBackgroundBrightness;   // Diffuse band-glow brightness multiplier (default 0.3)
  // Nubis3 sphere-trace step scale (fork — Nubis3 conversion Phase C).
  // Safety factor on the SDF empty-space skip in marchCloudSlab; 0 disables
  // the skip (uniform stepping). Rides the former padMilkyWay0 slot; CB
  // layout unchanged.
  float nvdfStepScale;

  // Diffuse glow color zones. Real Milky Way photos show three distinct
  // colors: a warm yellow-cream galactic center, dark red-brown dust lanes
  // weaving through the band, and cool blue outer halo from young stars
  // along the spiral arms. The glow blends between them based on direction
  // and per-pixel multi-octave noise so the result has structure rather than
  // a single flat tint × noise (the prior "blocky" look).
  vec3 milkyWayBackgroundColor;         // OUTER edge tint (default cool blue 0.5/0.55/0.75)
  float milkyWayDustAmount;             // How strongly dust lanes darken the glow (default 0.6)

  vec3 milkyWayCoreColor;               // BRIGHT core tint (default warm cream 1.0/0.85/0.55)
  // Nubis3 body erosion (fork — Nubis3 conversion, anti-blobby pass). Strength
  // of the 3D FBM carve applied to the placement waterline in the NVDF
  // occupancy bake — bakes concavity (overhangs / notches) into the otherwise
  // convex column bodies. Consumed at BAKE time only (NVDF rebake dirty key).
  // Rides the former padMilkyWay1 slot; CB layout unchanged.
  float nvdfBodyErosionStrength;

  vec3 milkyWayDustColor;               // DARK dust-lane tint (default red-brown 0.15/0.08/0.05)
  // Nubis3 near-camera HF detail (fork — Nubis Cubed p.125 `inHFDetails`).
  // Mix strength of the twice-folded high-frequency noise into the erosion
  // composite near the camera (fly-through crispness). 0 = off; 1 = the
  // paper's 10% max mix. Rides the former padMilkyWay2 slot; CB layout
  // unchanged.
  float nubis3HFDetailStrength;

  // ----- Per-moon parameters (fork) -----
  MoonParams moons[MAX_MOONS];

  // ----- Moon NEE / atmospheric-coupling strength sliders (fork) -----
  float moonNeeStrength;                  // World-side master multiplier (surface NEE + cloud + future volumetric)
  float moonAtmosphericCouplingStrength;  // Sky-side multiplier (atmospheric scattering blue-dome)
  float surfaceMoonBrightness;            // Per-path stylistic multiplier on surface NEE only (Phase 3, 2026-05-08)
  float cloudMoonBrightness;             // Per-path stylistic multiplier on cloud-moon directional + ambient airglow (Phase 3)

  float haloMoonBrightness;               // Per-path stylistic multiplier on disk halo Gaussian glow (Phase 3)
  // Nubis3 conversion (fork). nvdfNominalCoverage = the coverage the NVDF
  // body SDF was baked at (quantized live weather coverage in auto mode, or
  // a pinned value); consumed by cloud_nvdf_occupancy at bake time and by
  // the sampler's coverage level-set offset.
  // useSkyViewLut reclaims the former padRetired0 / nubis3ModelEnable slot
  // (Mirror's Edge perf opt): sample sky-view LUT at sky-miss instead of
  // per-ray evalSkyRadiance when non-zero. CB layout unchanged.
  uint  useSkyViewLut;
  float nvdfNominalCoverage;
  // Perf-bisect shader toggle (fork — 2026-06-11, diagnostic). Rides the former
  // padMoonNee2 slot; only bit 1 remains in use:
  //   bit 1: flat sky miss — evalSkyRadiance returns a constant grey immediately,
  //          isolating the full per-ray miss-path cost
  // (bit 0, "skip atmosphere NEE", was retired 2026-06-21 with the bespoke NEE.)
  // Bit clear in production; ImGui "Perf Bisect (Diagnostic)" tree drives it.
  uint  debugSkyBisectFlags;

  // ----- Moon cloud-look + halo shape constants (fork, Phase 3 Task 2) -----
  // Tunable shape parameters for cloud-moon silver-lining contrast and halo glow.
  // Defaults preserve current calibrated values; exposed via RTX_OPTION + ImGui
  // for in-game tuning of cloud-moon look without rebuilding shaders.
  float moonCloudDiffuseGain;             // Cloud-moon Lambert diffuse weight (silver-lining off-axis darkening)
  float moonCloudPhaseGain;               // Cloud-moon HG phase weight (silver-lining peak boost)
  float moonCloudAnisotropy;              // HG g for cloud-moon forward scatter (silver-lining sharpness)
  float moonHaloMagnitude;                // Disk halo Gaussian strength (was kHaloMagnitude in atmosphere_sky.slangh)

  float moonAmbientAirglow;               // Ambient airglow per-moon strength (was 0.0015 literal in nightLight)
  float cloudHexTilingEnable;             // Hex de-tiling of the cloud noise volume (fork — 2026-06-11,
                                          // stage A): 1 = stochastic triangle-lattice randomization of the
                                          // tiled volume (periodicity destroyed at the source), 0 = legacy
                                          // single periodic tap. Reuses the former padCloudLook0 slot;
                                          // CB layout unchanged.
  float nubis3FineDetailStrength;         // Fine-frequency band strength [0..2] (fork —
                                          // detail round follow-up 2026-07-16): third
                                          // incommensurate detail tap (~177..33 m) folded
                                          // into erosion + interior within ~8 km. 0 = off.
                                          // First reuse of the retired legacy pads (was
                                          // padRetired1 / cloudNoiseBaseFreqScale).
  float cloudSkyBleedStrength;            // [0,1+] strength of cloud-color inscatter bled into the
                                          // visible sky (sky reflects clouds; sampled from the smooth
                                          // secondary dome LUT). 0 = off. Reuses the former
                                          // cloudColumnShapingEnable slot (was padCloudLook2); CB
                                          // layout unchanged.

  // ----- Cloud parameters (fork: procedural FBM cloud layer at fixed altitude) -----
  vec3 cloudColor;          // Cloud base color (typically white)
  float cloudDensity;       // Overall opacity/density multiplier

  float cloudAltitude;      // Altitude of cloud layer (km)
  float cloudBoilPhase;     // Accumulated edge-boil scroll phase (km) — drives the detail tap's
                            // offset along a fixed internal direction so cloud edges churn
                            // independently of the base shape (fork — 2026-06-21, field-evolution
                            // rework). 0 = frozen. Reuses the former pad_cloudLayer2CoverageSpread
                            // slot; CB layout unchanged.
  float cloudEnabled;       // 1.0 if clouds should be rendered, 0.0 otherwise
  float cloudShadowStrength;// How strongly clouds dim ground/atmosphere lighting [0..1]

  vec2 cloudWindOffset;     // Accumulated wind-driven UV offset (km)
  float skyIndirectRadianceScale; // Diffuse-indirect sky radiance multiplier (>=0). Reuses the
                                  // former pad_cloudAnisotropy slot; CB layout unchanged.
  float cloudCurvature;     // 0 = Earth-scale dome, 1 = tight dome

  // ----- Cloud volumetric / appearance enhancements (fork) -----
  // Cloud detail-shading pass (fork — 2026-07-14). Four scalars reuse the
  // former pad_cloudShadowTint (vec3) + pad_cloudShadowTintStrength row, so
  // the CB layout is unchanged.
  float cloudMicroAoStrength;       // [0..1] billow-scale shading modulation from the
                                    // edge-detail field (bright knuckles / dark crevices
                                    // near the cloud surface). 0 = off (legacy smooth shading).
  float cloudPowderStrength;        // [0..1] Schneider powder darkening of low-density
                                    // sun-facing samples (crevice / edge darkening when the
                                    // sun is behind the viewer). 0 = off.
  float nubis3ShapeVarietyKm;       // Mid-band SHAPE displacement amplitude in km
                                    // (fork — 2026-07-17): dedicated level-set
                                    // push/pull by the 2.4 km mid tap (±0.5·this),
                                    // deep enough to lobe/split bodies — varied
                                    // clusters instead of round singles. 0 = off.
                                    // (Second reuse of the retired legacy pads.)
  float cloudDetailBaseShearKm;     // Horizontal displacement of the detail tap at the cloud
                                    // base (km), fading to 0 at the top — sheared/streaked
                                    // base wisps. 0 = no shear (legacy).

  float cloudThickness;        // Cloud-slab vertical depth, km
  float cloudLayer2TypeSpread; // [0,1] cloud-type variation for layer 2 (independent of layer 1)
  float lightningHistoryFade;  // [0..1] lightning ghost-suppression signal (fork — 2026-07-14):
                               // 1 while a flash is live, decaying over ~0.25 s after it ends.
                               // evalSkyRadiance collapses the cloud temporal-history weight by
                               // this factor so the flash never embeds into the ~1 s EMA (the
                               // reprojected "old frame" ghost on camera move). Reuses the former
                               // pad_cloudSunsetWarmth slot; CB layout unchanged. Zeroed in
                               // normalizeForSkyLutCache (per-frame animated, never feeds a bake).
  uint cloudViewSamples;       // Ray-march steps through cloud slab

  // ----- Spatial variation fields (Nubis-style weather) -----
  float cloudTypeMean;             // [0,1] mean cloud type. 0=stratus, 0.5=stratocumulus, 1=cumulus.
  float cloudTypeSpread;           // [0,1] amplitude of type variation around mean.
  float cloudTypeNoiseScale;       // Region size frequency for type noise.
  float cloudCoverageMean;         // [0,1] mean coverage across the sky.

  float cloudCoverageSpread;       // [0,1] amplitude of coverage variation around mean.
  float cloudCoverageNoiseScale;   // Region size frequency for coverage noise (independent of type).
  float padRetired12;              // retired 2026-07-30: nubis3SunNearFieldKm (near-field live
                                   // sun-occlusion range). The live path was removed once the
                                   // de-jittered D_sun bake reproduced the look; see
                                   // docs/fork-touchpoints.md.
  float cloudMsScale;              // Multi-scatter sigma_ms master multiplier (1.0 = paper baseline)

  float cloudAmbientShadowStrength; // [0..1] D_sun-keyed attenuation of the cloud AMBIENT term
                                    // (fork — 2026-07-14, dramatic-shading pass): sun-shadowed
                                    // bulk loses its sky-ambient fill (exp falloff on the same
                                    // optical depth that drives the direct lobes), so shaded
                                    // cores plunge dark while lit faces keep full ambient.
                                    // 0 = off (legacy flat ambient floor). Reuses the former
                                    // pad_cloudMultiScatterStrength slot; CB layout unchanged.
  uint  cloudMultiScatterOctaves;  // Number of Wrenninge octaves to sum (clamped 1..4 in shader).
  float cloudLayer2NoiseSeed;      // Seed offset added to layer 2's 2D coverage/type smoothNoise2D
                                   // calls so layer 2 generates a fully decorrelated noise pattern
                                   // at the same XZ (no geometry interference).
  float cloudAerialFadePerKm;      // Aerial-perspective FADE on cloud extinction (1/km). Softens
                                   // alpha-accumulation at distance so horizon-grazing rays don't
                                   // pile up into a solid white wall.

  // ----- Stage C: 3D noise texture (fork) -----
  float cloudNoiseTileKm;   // World-space tile period for the prebaked 3D noise.
                            // Texture is tilable; this controls how many km of
                            // unique cloud structure before the pattern repeats.
                            // Default 12.0 (~47 m/voxel at 256 resolution).

  // ----- Volumetric sky-ambient illumination (fork — 2026-05-12) -----
  // Multipliers consumed by sampleSkyAmbientForVolume and the hemisphere
  // integration injected into the rtxdi volumetric pass at
  // volume_integrator.slangh:302. Defaults below preserve baseline behavior
  // (skyAmbientStrength = 0 means the feature is off by default).
  float cloudSkyAmbientStrength;                 // Overall multiplier on the sky-ambient term [0..3]. 0 = feature off.
  float cloudSkyAmbientCloudOcclusionStrength;   // Strength of cloud occlusion of sky ambient [0..1]. 1 = physical.
  float cloudCellSizeKm;         // Average cloud-cluster footprint size (km) for the placement
                                 // map bake (fork — 2026-06-11, column-shaping rework). Re-bakes
                                 // the placement map live on change. Reuses the former padCloudC2
                                 // slot (ex-cloudNoiseWarpStrength); CB layout unchanged.

  // ----- Cloud voxel grid (Nubis Cubed 2023, fork — 2026-05-12) -----
  // 256x256x32 R16F precomputed grids storing summed optical depth along the
  // sun direction (D_sun) and zenith (D_ambient), camera-centered with
  // horizontal tile-wrap. Baked round-robin every 8 frames by
  // cloud_sun_density_grid.comp.slang / cloud_ambient_density_grid.comp.slang.
  // Consumed at shade time by the Nubis Cubed cloud-lighting path via
  // sampleDSun / sampleDAmbient.
  float cloudVoxelGridExtentKm;     // Horizontal extent of camera-centered grid (default 12.0 km)
  float cloudVoxelGridVerticalKm;   // Vertical extent — populated CPU-side from cloudThickness
  // The three below were informational round-robin/dirty scalars with no shader
  // consumer; demoted to reserve pads 2026-06-21, then reused the same day as the
  // base field-evolution scroll offset (fork — field-evolution rework). This is a
  // slow 3D offset added to the base 3D noise sample position only (NOT to
  // heightFraction / hex placement / column model), so the cloud field morphs in
  // place — clouds form and dissolve — while altitude and placement stay put. The
  // Y component scrolls through the volume's decorrelated, tile-wrapping vertical
  // axis (the dominant morph term); X/Z add lateral decorrelation. 0 = frozen.
  // Three separate scalars reuse the three contiguous voxel-grid pad slots, so the
  // CB layout is byte-identical.
  float cloudEvolutionOffsetX;
  float cloudEvolutionOffsetY;
  float cloudEvolutionOffsetZ;
  // The three fields below reuse the former pad_cloudVoxel0..2 slots so the
  // constant-buffer layout is unchanged.
  float cloudBottomDarkening;       // [0,1] strength of the sun-gated underside darkening (multi-scatter + ambient)
  float cloudSkyAmbientFill;        // [0,1] strength of the sky-dome underside fill (clouds reflect the
                                    // open sky from below/around, bypassing bottom-darkening; bright by
                                    // day, fades at sunset). Reuses the former cloudBottomDarkeningHeight
                                    // slot (was pad_cloudVoxel1); CB layout unchanged.
  float cloudDetailStrength;        // Silhouette wobble amplitude: how strongly the detail field
                                    // displaces the cloud silhouette (0 = off). Scaled by
                                    // kWobbleKmPerDetailStrength in cloud_nubis3_common.slangh.

  // ----- Nubis Cubed 2023 lighting params (fork — 2026-05-12, C4) -----
  // Consumed by cloud_render.comp.slang via evalNubisCubedSampleCore.
  float cloudPhaseG1;              // Primary HG asymmetry (silver-lining peak)
  float cloudPhaseG2;              // Secondary HG asymmetry (broader envelope)
  float cloudMsSunDotMax;          // sigma_ms remap upper bound on sun_dot (page-137 magic constant)
  float cloudMsSigmaShallow;       // sigma_ms at cloud surface / shallow penetration

  float cloudMsSigmaDeep;          // sigma_ms deep inside cloud (saturated)
  float cloudMsSdfDepth;           // SDF depth in meters at which sigma_ms saturates to deep
  uint  cloudRenderFrameIdx;       // Frame counter for fastJitter() in cloud_render.comp.slang
  float cloudDetailScale;          // Edge-detail noise frequency, as a multiple of the base
                                   // cloudNoiseTileKm frequency. Reuses the former pad_nubisCubed0
                                   // slot, so the constant-buffer layout is unchanged.

  // ----- Cloud render camera basis (fork — 2026-05-12, C4) -----
  // Pre-computed Y-up basis vectors (camera at origin). Per-pixel view direction
  // is reconstructed in cloud_render.comp.slang as:
  //   viewDirYUp = normalize(forward + ndc.x * rightScaled + ndc.y * upScaled)
  // The `Right` and `Up` vectors are pre-multiplied by tan(halfFovX/Y) so the
  // shader doesn't need fov/aspect knowledge. All in Y-up world (cloud math
  // convention — camera at origin).
  // The three scalars riding the camera-basis vec3 padding below belong to
  // the column-shaping rework (fork — 2026-06-11); they reuse the former
  // pad_cr0..2 slots so the CB layout is unchanged.
  vec3  cloudRenderForwardYUp;
  float cloudColumnTopVariation;   // [0,1] per-cloud tower-height jitter amount (0 = uniform tops)

  vec3  cloudRenderRightYUp;       // Pre-scaled by tan(halfFovX) * aspectRatio
  float cloudColumnTopShape;       // Exponent mapping column presence -> top height (lower = taller edges)

  vec3  cloudRenderUpYUp;          // Pre-scaled by tan(halfFovY)
  float cloudColumnBaseVariation;  // [0,~0.4] max local cloud-base lift as a fraction of the slab

  // ----- Nubis Cubed sky-miss composite gate (fork — 2026-05-12, C5) -----
  // When 1, the primary-ray branch in evalSkyRadiance reads the prerendered
  // AtmosphereCloudRender RT to composite clouds; when 0, primary-ray sky-miss
  // is cloudless. PSR, indirect, and reflection rays use the secondary dome LUT
  // (see cloudSecondaryLutEnable) regardless of this gate — the cloud RT is at
  // primary-ray pixel coords, sampling it for a different ray direction at the
  // same pixel would return the wrong cloud.
  uint  cloudRenderRTEnable;       // 0 or 1
  // Secondary-ray cloud LUT gate (fork — 2026-06-10, perf). When 1, the
  // non-primary branch in evalSkyRadiance samples the per-frame
  // AtmosphereCloudSecondaryLut dome; when 0, secondary sky-miss rays are
  // cloudless. Reuses the former pad_c5_0 slot, so the CB layout is unchanged.
  uint  cloudSecondaryLutEnable;   // 0 or 1
  // Downscale (DLSS-input) render extent, i.e. the coordinate space of the
  // pixelCoord evalSkyRadiance receives (fork — 2026-06-11, half-res cloud
  // RT). The cloud RT may be allocated SMALLER than this
  // (cloudRenderResolutionScale); the primary-ray composite divides
  // pixelCoord by these dims to get normalized screen uv and bilinearly
  // samples the RT — exact texel-center fetch when the RT is full-size.
  // 0 means "not yet known" (first frames) and selects the legacy Load
  // path. Reuse the former pad_c5_1/2 slots; CB layout unchanged.
  uint  cloudRenderFullDimX;
  uint  cloudRenderFullDimY;

  // ----- Voxel-grid cloud-on-terrain shadows at NEE (fork — 2026-05-12, C6) -----
  // Plumbing for sampleCloudGroundShadow_OptionB, called from the surface and
  // volumetric NEE entry points via a ratio correction that replaces the
  // legacy evalCloudGroundShadow uniform dimmer with the 3D D_sun grid lookup.
  //   * cloudVoxelShadowsEnable — master gate (default 0 / off).
  //   * cloudShadowMarchStrength — multiplier on the Beer-Lambert exponent in
  //     transmittance = exp(-D_sun * cloudDensity * cloudShadowMarchStrength).
  //     1.0 = physical baseline.
  //   * worldUnitsPerKm — game-units per kilometer, derived CPU-side from
  //     RtxOptions::sceneScale (which is cm per game unit). 1 km = 100000 cm
  //     and 1 cm = sceneScale game units, so 1 km = 100000 * sceneScale game
  //     units. Used by sampleCloudGroundShadow_OptionB to convert
  //     G-buffer worldPos (game units) into km for the slab + voxel-grid math.
  //   * cameraWorldPosYUpKm — camera world position in Y-up km, used to
  //     express the surface worldPos as camera-relative for cloudVoxelWorldToUVW
  //     (the voxel grid is camera-centered with horizontal tile-wrap).
  uint  cloudVoxelShadowsEnable;   // 0 or 1
  float cloudShadowMarchStrength;  // Beer-Lambert exponent multiplier
  float worldUnitsPerKm;           // game units per km
  float cloudColumnFeather;        // Coverage-remap feather band width for column presence
                                   // (fork — 2026-06-11, column-shaping rework). Reuses the
                                   // former pad_c6_0 slot; CB layout unchanged.

  vec3  cameraWorldPosYUpKm;       // Camera position in Y-up km, world-absolute
  float cloudUndersideLightSigma;  // Extinction sigma (per km·density of overlying water) for
                                   // the analytic per-column downwelling light field that lights
                                   // cloud undersides in column mode (fork — 2026-06-12,
                                   // column-shaping rev 3; replaced rev 2's flat span scaling).
                                   // 0 = legacy constant bottom-darkening gradient. Reuses the
                                   // former pad_c6_1 slot; CB layout unchanged.

  // ----- (former Cloud Height LUT block — retired with the legacy model) -----
  uint  padRetired4;               // retired: legacy height-LUT enable.

  // ----- Two-layer cloud map (slide 1 lift — RDR2 SIGGRAPH 2019) -----
  // Adds an independent second cloud slab at its own altitude band, sampled
  // by cloud_render.comp.slang only (voxel grid bakers + ground-shadow march
  // all stay layer-1 only for v1 — cirrus is too thin to need precomputed
  // terrain shadows, and the simpler
  // scoping caps the change surface). When enabled, cloud_render marches
  // the lower slab first (front-to-back) and composes layer 2 on top of
  // the residual transmittance.
  uint  cloudLayer2Enable;         // 0 = single-layer (default), 1 = render layer 2 also
  float cloudLayer2Altitude;       // Altitude of layer 2 slab base (km)
  float cloudLayer2Thickness;      // Layer 2 slab vertical depth (km)

  float cloudLayer2TypeMean;       // [0,1] mean cloud type for layer 2 (defaults to a cirrus-shaped 0.0)
  float cloudLayer2CoverageMean;   // [0,1] mean coverage for layer 2 (defaults sparse)
  float cloudLayer2DensityScale;   // Per-step density multiplier for layer 2 (cirrus is optically thin)
  float padRetired5;               // retired: legacy vertical noise stretch.

  // ----- (former Worley carve params — retired with the legacy 256^3 bake) -----
  // Contribution-weighted lighting LOD threshold (fork — 2026-07-30, perf).
  // A march sample whose contribution weight (viewTransmittance x aerial haze x
  // its own opacity) falls below this drops the near-field live sun refinement
  // and the moon shadow march, falling back to the D_sun grid and unshadowed
  // moonlight respectively. 0 disables the LOD. Rides the former padRetired6
  // slot — CB layout unchanged (16-byte-row discipline preserved).
  float cloudLightingLodThreshold;
  float padRetired7;
  uint  padRetired8;
  float cloudAerialHazePerKm;      // Aerial-perspective HAZE on cloud radiance (1/km). Dims distant
                                   // cloud samples toward atmospheric color. Visual softness control.

  // ----- Sunset ambient warm/cool blend (fork — 2026-05-21) -----
  // At low sun elevations, lerp the per-sample ambient sky color between the
  // sun-direction sky LUT sample (warm) and the anti-sun-horizon sky LUT
  // sample (cool), driven by the D_sun voxel grid so shadowed cloud interiors
  // read cool while sun-lit edges stay warm. Ramped off above rampHighSun so
  // midday clouds are unaffected. Consumed by evalNubisCubedSampleCore.
  float cloudSunsetAmbientStrength;   // Master multiplier on cool blend (0 = feature off, 1 = baseline)
  float cloudSunsetAmbientReachInvKm; // D_sun reach in 1/km — higher = clouds turn cool faster with shadow depth
  float cloudSunsetAmbientRampHighSun;// sin(sun elevation) at which the effect smooth-fades to zero
  float cloudViewStepKm;              // Target cloud-march step LENGTH in km (fork — 2026-06-12,
                                      // adaptive march sampling). The per-ray step count becomes
                                      // span / step, floored at cloudViewSamples and capped at
                                      // cloudViewSamplesMax, so horizon-grazing rays (slab spans
                                      // of 50+ km) stop undersampling 2 km cloud features with
                                      // 1.6 km steps — the aliasing read as soft horizontal
                                      // banding. 0 = legacy fixed count. Reuses the former
                                      // pad_cloudSunsetAmbient0 slot; CB layout unchanged.

  // ----- Cloud-edge / halo tuning (fork — 2026-06-13). Exposed live in ImGui. -----
  float padRetired9;                  // retired: legacy view coverage-gate softness.
  float cloudEdgeAmbientFade;         // Density at which a thin sample's (horizon-tinted) ambient
                                      // reaches full strength [0..~0.5]. Below it the ambient fades
                                      // toward 0 so the soft skirt doesn't read as grey-brown haze.
                                      // 0 = off (ambient at full strength on all samples).
  // retired: atmosphereSunVolumetricRadianceScale (fork — issue #35) scaled the
  // sun term where it was added to the froxel SH in volume_integrator.slangh.
  // That injection was removed on 2026-06-28 (it double-counted the sun, which
  // is already sampled by the volume NEE loop), leaving this with no consumer.
  float padRetired11;

  // ----- Artistic sunset color controls (fork — 2026-06-14) -----
  // Counteract the desaturation introduced when sunset reddening moved onto the
  // physical Hillaire two-term LUT model (commit 3e37062b): the multiscatter
  // fill reads pale-blue and washes the warm single-scatter. Both apply inside
  // evalAtmosphereRadiance, so the sky-view LUT carries them and clouds inherit
  // the warmer ambient for free. Defaults (1.0) reproduce the physical look.
  float multiScatterStrength; // Global scale on the multiscatter fill term. <1 = less
                              // pale-blue wash so warm single-scatter dominates. 1 = physical.
                              // Reuses the former pad_cloudEdge1 slot; CB layout unchanged.

  float sunsetSaturation;     // Saturation boost on sky radiance, ramped in near the horizon
                              // (midday untouched). >1 = punchier warm sunset. 1 = no change.
  // Artistic contrast curve on the cloud-on-terrain shadow (fork — 2026-06-19).
  // Applied as pow(cloudTransmittance, cloudShadowFactorStrength) where the
  // factor is folded onto the SUN's radiance in sampleAtmosphereSunLight /
  // sampleAtmosphereSunLightVolume. This is the same perception-side knob that
  // previously lived in composite_args.h (it used to scale the screen-space
  // PrimaryCloudShadowFactor texture); it moved here when the cloud shadow was
  // re-architected onto the sun term and the screen-space texture was deleted.
  // 1.0 = raw physical transmittance, >1 deepens cumulus shadows, <1 fades.
  // Mirrors RtxOptions::cloudShadowFactorStrength(). Reuses the former
  // pad_artistic0 slot; CB layout unchanged.
  float cloudShadowFactorStrength;
  // ----- Cloud direct-lighting energy conservation (fork — 2026-06-19) -----
  // Reformulates the direct dual-lobe from the legacy additive sum
  // (T_primary*HG1 + M*HG2, phase integral up to ~2) into an energy-conserving
  // convex blend (phase integral 1) — the fix for lit clouds out-brightening
  // the physical sky LUT. Consumed by evalNubisCubedSampleCore. Both reuse the
  // former pad_artistic1/2 slots; CB layout unchanged.
  float cloudEnergyConserve;  // [0,1] 0 = legacy additive look (A/B), 1 = energy-conserving convex blend
  float cloudMsLobeWeight;    // [0,1] convex weight: forward single-scatter lobe (1-w) vs multi-scatter body fill (w)

  // ----- Layer-2 echo-deck step budget (fork — 2026-06-21) -----
  // The echo deck is marched far more cheaply than layer 1; these are its own
  // floor/cap on the adaptive (cloudViewStepKm-driven) step count, decoupled
  // from layer 1's cloudViewSamples / cloudViewSamplesMax. Consumed by
  // marchEchoDeck in cloud_march_common.slangh.
  //
  // IMPORTANT (CB alignment): no free pad slots remained, so this grows the
  // constant buffer. It MUST grow by a whole 16-byte (vec4) block or
  // sizeof(AtmosphereArgs) stops being 16-byte aligned and the updateBuffer of
  // the whole struct corrupts the cbuffer (the two real fields below then read
  // garbage, and marchEchoDeck's step count blows up into a GPU hang — solid
  // black whenever layer 2 is enabled). Hence the two explicit pad words: 2
  // real + 2 pad = one vec4 row. Future additions should consume these pads
  // first (reuse-the-pad-slot discipline) before growing the struct again.
  uint cloudLayer2StepFloor;     // Min march steps through the echo deck (near-zenith floor)
  uint cloudLayer2StepMax;       // Hard cap on echo-deck steps per ray (perf governor)
  // Nubis3 profile controls (fork — Nubis3 conversion; Phase B consumers).
  // Ride the former pad_cloudLayer2Step0/1 reserve slots; CB layout unchanged.
  //   nvdfProfileDepthKm — depth into the body over which the dimensional
  //     profile ramps 0 -> 1 (profile = remap of the offset SDF).
  //   nvdfCoverageOffsetKm — km of level-set shift per unit of coverage
  //     delta from the baked nominal (coverage grows/shrinks bodies live).
  float nvdfProfileDepthKm;
  float nvdfCoverageOffsetKm;

  // Layer-2 echo-deck color (fork — 2026-06-21). Independent albedo for the
  // deck (the one look knob split out from layer 1). vec3 + 1 pad word = one
  // vec4 row, so the CB stays 16-byte aligned (see the step block above —
  // appending a bare vec3 would straddle the row boundary and corrupt the CB).
  vec3  cloudLayer2Color;        // Deck base color; defaults to cloudColor's near-white
  // Nubis3 value-erosion strength (fork — Nubis3 conversion; Phase B
  // consumer). Scales the wispy/billowy noise composite before the
  // ValueErosion remap of the dimensional profile. Rides the former
  // pad_cloudLayer2Color0 reserve slot; CB layout unchanged. 0 until Phase B.
  float nubis3ErosionStrength;

  // Lightning (fork — 2026-07-14). Two whole vec4 rows (CB alignment rule
  // above). The CPU-side strike scheduler (RtxAtmosphere::advanceLightning)
  // drives a flickering flash envelope; the view-path cloud march adds an
  // emissive glow around the strike position (compile-gated to the screen
  // cloud pass only — the secondary-ray cloud LUT must never bake a transient
  // flash in), and fhSyncAtmosphereDistantLights injects a transient sphere
  // light at the same position so the scene flashes too. Both rows are zeroed
  // in normalizeForSkyLutCache so a flash never invalidates the sky-LUT
  // cache key.
  vec3  lightningStrikePosKm;     // Strike position, world-anchored Y-up km (same
                                  // frame as cameraWorldPosYUpKm / samplePos).
  float lightningFlashIntensity;  // Envelope × lightningFlashIntensity option,
                                  // premultiplied. 0 = no active flash (the cloud
                                  // march skips the term entirely).

  vec3  lightningColor;           // Flash tint (linear RGB), shared by the in-cloud
                                  // glow and the scene sphere light.
  float lightningEnvelope;        // RAW flicker envelope [0..~1.2], before any
                                  // intensity scaling — the scene-light sync scales
                                  // this by lightningSceneLightIntensity so the two
                                  // consumers calibrate independently.

  // ----- Cloud temporal-smoothing weight + reserve pads (fork — 2026-07-16,
  // anti-blobby crispness pass). NEW 16-byte block appended at the struct
  // tail (all former reserve pads are consumed); grow this struct ONLY in
  // full vec4 rows (see the CB-alignment discipline note at the top).
  float cloudHistoryWeight;  // EMA history weight of the cloud temporal
                             // smoother in evalSkyRadiance [0..0.98]. Was the
                             // hardcoded 0.92; lower = crisper/faster response
                             // but more visible per-frame jitter for DLSS to
                             // chew on. Zeroed in normalizeForSkyLutCache
                             // (composite-only — never feeds a bake).
  float nubis3InteriorTexture;  // [0..1] strength of the interior density
                                // modulation by the raw detail channels
                                // (Nubis3 Density-Scale-NVDF / iw3xo
                                // self-gate stand-in — see the sampler's
                                // step 8). 0 = flat saturated interiors
                                // (pre-2026-07-16 behavior). Rides the
                                // former padReserve0 slot.
  float nubis3EdgeErosion;      // [0..3] edge wisp cut: additional erosion
                                // shaped by the WISPY channel, concentrated
                                // in the outer shell (x(1-profile)^2) — deep
                                // strand-shaped cuts that shred silhouettes
                                // into trailing wisps while billowy cores
                                // keep rounded edges. 0 = uniform erosion
                                // only. Rides the former padReserve1 slot.
  float nubis3AdaptiveStepKm;   // √-adaptive hybrid march step FLOOR in km
                                // (fork — detail round 2026-07-16, Nubis
                                // p.172/174). Nubis3 view march only: step =
                                // max(SDF · nvdfStepScale, clamp(
                                // cloudViewStepKm · sqrt(t/12 km), THIS,
                                // 4·cloudViewStepKm)). 0 = off (fixed-length
                                // lattice stepping). Rides the former
                                // padReserve2 slot — THE LAST reserve pad;
                                // any further growth needs a new full
                                // 16-byte row (see the CB-alignment
                                // discipline note at the top).
};
