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

#include "rtx/utility/shader_types.h"
#include "rtx/pass/atmosphere/atmosphere_args.h"

// Volumetric cloud layer on the Physical Atmosphere (rtx_clouds.cpp). The field, its bakes and the march
// work in kilometres in the atmosphere's Y-up frame: x and z are absolute world positions (the field is
// anchored to the world, so it shows parallax as the camera moves), y is altitude above sea level measured
// on the atmosphere's planet. Altitudes are relative to the same datum as rtx.atmosphere.altitude.

// Cloud body signed distance field (NVDF). One noise tile horizontally, the slab vertically; texture y is
// vertical.
#define CLOUD_NVDF_SIZE_XZ 256
#define CLOUD_NVDF_SIZE_Y 64

// Detail erosion volume, periodic in its own domain.
#define CLOUD_DETAIL_NOISE_SIZE 128
#define CLOUD_DETAIL_NOISE_MIP_COUNT 8

// 2D placement map of the column model, periodic at one noise tile.
#define CLOUD_PLACEMENT_MAP_SIZE 512

// Sun and vertical optical depth grids around the camera; texture y is vertical (the slab). The far cascade has more
// columns over its wider window.
#define CLOUD_GRID_SIZE_XZ 256
#define CLOUD_FAR_GRID_SIZE_XZ 384
#define CLOUD_GRID_SIZE_Y 32
// Box filtered levels of the sun grid, read by the higher scattering orders (optical depth through blurred
// density).
#define CLOUD_SUN_GRID_MIP_COUNT 4

// Aerial perspective between the camera and any distance along any direction: the sky-view LUT's
// parameterisation with logarithmically spaced distance slices.
#define CLOUD_SKY_AP_LUT_WIDTH 128
#define CLOUD_SKY_AP_LUT_HEIGHT 64
#define CLOUD_SKY_AP_LUT_DEPTH 48
#define CLOUD_SKY_AP_MIN_DISTANCE_KM 0.05f
#define CLOUD_SKY_AP_MAX_DISTANCE_KM 320.0f

// L2 spherical harmonics of the sky radiance at altitudes spanning the cloud layer, upper and lower hemisphere
// projected separately: CLOUD_SKY_SH_ALTITUDES rows of 2 x 9 texels (RGB in xyz).
#define CLOUD_SKY_SH_ALTITUDES 3
#define CLOUD_SKY_SH_COEFFICIENTS 9

// Droplet phase function: texels over u = sqrt(theta / pi), one row per effective radius.
#define CLOUD_PHASE_LUT_SIZE 512
#define CLOUD_PHASE_LUT_RADII 8

// Far field cloud shadow map: optical depth through the layer along the sun, indexed by where the sun ray
// enters the layer's base.
#define CLOUD_SHADOW_MAP_SIZE 512

// Bits of CloudArgs.flags.
#define CLOUD_FLAG_SKY_AERIAL_PERSPECTIVE (1u << 0)
#define CLOUD_FLAG_SUN_SHADOWS            (1u << 1)
#define CLOUD_FLAG_FOG_SHADOWS            (1u << 2)
#define CLOUD_FLAG_SHADOW_MAP_VALID       (1u << 3)
#define CLOUD_FLAG_DOME                   (1u << 4)
#define CLOUD_FLAG_PSR_REFLECTIONS        (1u << 5)
#define CLOUD_FLAG_HEX_TILING             (1u << 6)
#define CLOUD_FLAG_REFERENCE              (1u << 7)
#define CLOUD_FLAG_HISTORY_VALID          (1u << 8)
#define CLOUD_FLAG_GLOSSY_REFLECTIONS     (1u << 9)

struct CloudArgs {
  uint enabled;  // Non-zero when the layer renders this frame
  uint frameIndex;
  uint flags;    // CLOUD_FLAG_*
  uint debugView;

  // Slab, in km. baseAltitudeKm is above sea level.
  float baseAltitudeKm;
  float thicknessKm;
  float tileKm;      // Horizontal period of the body field and its detail
  float worldUnitsPerKm;

  // x and z: world position in km; y: the eye's altitude above sea level in km.
  vec3 cameraPositionKm;
  float cameraWorldHeightKm;  // The camera's coordinate along the world's up axis, km

  // Snapped origins of the optical depth grids' near window (x, y) and of the far field shadow map (z, w).
  vec4 gridOriginKm;

  vec2 farGridOriginKm;  // The optical depth grids' far cascade
  float gridExtentKm;
  float farGridExtentKm;

  float shadowMapExtentKm;
  float apStartDistanceKm;  // Where the sky aerial perspective starts, past the froxel fog's range
  float pixelAngle;         // Angle one render pixel subtends, radians
  float detailLodBias;      // Mip bias of the detail volume's footprint filtering

  vec2 windOffsetKm;
  vec2 windDirection;  // Unit vector in x / z

  float evolutionRiseKm;   // Integrated convective rise of the detail field
  float evolutionShearKm;  // Integrated downwind shear of the detail field's tops
  float detailBaseShearKm;
  float cellSizeKm;

  float coverage;
  float coverageSpread;
  float coverageSpreadFrequency;  // Cells per km of the field that varies the coverage
  float cloudType;                // Erosion character: 0 wispy, 1 billowy

  float typeSpread;
  float typeSpreadFrequency;
  float columnFeather;
  float columnTopShape;

  float columnTopVariation;
  float columnBaseVariation;
  float columnTopFlatten;  // Caps column tops below the slab top for stratiform layers, 1 = none
  float nvdfNominalCoverage;

  float nvdfCoverageOffsetKm;
  float nvdfProfileDepthKm;
  float nvdfBodyErosion;
  float nvdfStepScale;

  float detailScale;
  float wobbleStrength;
  float erosionStrength;
  float sharpenStrength;

  float hfDetailStrength;
  float interiorTexture;
  float edgeErosion;
  float fineDetailStrength;

  float shapeVarietyKm;
  float shapeVarietyWavelengthKm;
  float curlStrengthKm;      // Sample time domain warp of the erosion field, strongest at the base
  float nearDetailStrength;  // Extra high frequency erosion octave close to the camera

  float nearDetailRangeKm;
  float adaptiveStepKm;
  float viewStepKm;
  float maxMarchKm;

  uint viewSamplesMax;
  uint shadowTaps;            // Full resolution taps toward the sun ahead of the grid
  float shadowTapRangeKm;
  float opticalDepthPerStep;  // Step length limit inside cloud, in optical depths (0 = off)

  float exitTransmittance;
  float historyBlend;  // Least weight of a new screen frame against the reprojected history, 1 = none
  float domeBlend;     // Weight of a dome texel's refresh against its history
  uint msOctaves;

  // Extinction (km^-1) of field density 1 at the reference height (1 km above the base), before the vertical
  // profile. Field density is the cloud's local fraction of the profile's liquid water.
  float extinctionKm;
  float adiabaticExponent;  // Extinction ~ height^exponent above the base (2/3 for an adiabatic parcel)
  float adiabaticFloor;
  float adiabaticMax;

  float forwardPeakFraction;  // delta-M truncation fraction f of the droplet phase function
  float asymmetry;            // Full asymmetry g
  float truncatedAsymmetry;   // Asymmetry of the truncated phase, (g - f) / (1 - f)
  float phaseLutRow;          // Fractional row of the droplet phase LUT for this layer's effective radius

  vec2 phaseLegendre;        // Legendre moments chi1, chi2 of the truncated phase, for the sky light's convolution
  float shadowExtinctionKm;  // Extinction of density 1 for the cloud shadows on the scene and the air
  float shadowStrength;      // 1 = physical cloud shadows on the scene and the air

  // Wrenninge et al. 2013's octave series of the higher scattering orders.
  float msExtinctionFalloff;  // a: extinction scale per order
  float msEnergyFalloff;      // b: contribution per order
  float msPhaseFalloff;       // c: asymmetry scale per order
  float msFloorWeight;        // Weight of the diffusion floor

  float msFloorAnisotropy;     // Weight of the diffusion field's flux term, 1 = Eddington's
  float diffuseTransmissionK;  // k of the two-stream diffuse transmittance 1 / (1 + k (1 - g) tau)
  float ambientStrength;
  float groundDiffuseShare;    // Share of shadowed sunlight the ground still receives through the cloud, diffusely

  vec3 albedoTint;  // Artistic tint of the cloud's scattering, 1 = physical
  float groundBounceStrength;

  uint referenceMaxBounces;
  uint referenceBouncesPerFrame;  // Collisions each reference tile's path advances by per frame
  uint referenceExactBounces;     // Collisions by the exact phase before the reference's paths go delta-M similar
  uint pad0;

  vec2 windStepKm;    // The field's displacement since the last frame, x / z
  float riseStepKm;   // The detail's convective rise since the last frame
  float shearStepKm;  // The detail's downwind shear at the top since the last frame

  // Overhead against horizon bias: shrinks the clouds within horizonBiasStartKm of the camera (bias > 0) or beyond
  // horizonBiasEndKm (bias < 0), blending between; the level set shift at full strength, 0 for none.
  float horizonBias;
  float horizonBiasStartKm;
  float horizonBiasEndKm;
  float horizonBiasShiftKm;
};

// Constant buffer of the cloud passes: this frame's cloud and atmosphere parameters together. The cloud
// parameters lead, so passes with their own AtmosphereArgs block bind just that leading range as a
// ConstantBuffer<CloudArgs> (one struct cannot be both a block and a block's member).
struct CloudConstants {
  CloudArgs cloud;
  AtmosphereArgs atmosphere;
};

// Push constants of the jump flooding pass.
struct CloudNvdfJfaArgs {
  uint jumpSizeVoxels;
  uint mode;  // 0 = seed init, 1 = jump pass
  uint pad0;
  uint pad1;
};

// The reference's error statistics over opaque tiles (uints): the tile count, the summed |error| in 1/1024ths,
// the paths every tile has finished, the summed mean luminance of the march and of the reference in 1/256ths
// (whose ratio, unlike a mean of ratios, the reference's noise does not bias), then a histogram of |error| over
// [0, 2) in CLOUD_REFERENCE_ERROR_BINS bins.
#define CLOUD_REFERENCE_ERROR_BINS 64
#define CLOUD_REFERENCE_STATISTICS_HEADER 5
#define CLOUD_REFERENCE_STATISTICS_SIZE (CLOUD_REFERENCE_STATISTICS_HEADER + CLOUD_REFERENCE_ERROR_BINS)

// Push constants of the bakes, the dome and the reference.
struct CloudPassArgs {
  uint period;  // Interleave period of the bake, 1 = full
  uint phase;   // Of the interleave; with period 1, the first texel along the interleaved axis
  // The grid bakes' cascade (0 near, 1 far); for the dome and the reference, non-zero with history; for the sky
  // light, non-zero to re-weight only its ground
  uint level;
  uint offset;  // First texel along the other axis: the bakes of the strips a moving window brings in wrap round
};

#ifdef __cplusplus
static_assert((sizeof(CloudArgs) & 15) == 0);
#endif
