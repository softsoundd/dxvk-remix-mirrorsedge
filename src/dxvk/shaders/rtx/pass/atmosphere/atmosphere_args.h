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

// The multiple scattering texture (multiscattering_lut.comp.slang) is an atlas of tiles of multiscatteringLutSize^2
// texels over (sun zenith cosine, altitude). Tile row 0 holds the isotropic estimate of the bake's first pass (only
// its scratch copy keeps one), then the second moments of the radiance field that the R, G and B Rayleigh responses
// need; each later row holds one relative azimuth of the aerosol response, one tile per view zenith.
#define ATMOSPHERE_MS_VIEW_ZENITH_COUNT 16
#define ATMOSPHERE_MS_RELATIVE_AZIMUTH_COUNT 8
#define ATMOSPHERE_MS_ATLAS_TILES_X ATMOSPHERE_MS_VIEW_ZENITH_COUNT
#define ATMOSPHERE_MS_ATLAS_TILES_Y (ATMOSPHERE_MS_RELATIVE_AZIMUTH_COUNT + 1)

// Passes of the multiple scattering bake, see multiscattering_lut.comp.slang.
#define ATMOSPHERE_MS_PASS_ISOTROPIC 0
#define ATMOSPHERE_MS_PASS_FROM_ISOTROPIC 1
#define ATMOSPHERE_MS_PASS_FROM_DIRECTIONAL 2

struct AtmosphereMultiscatteringBakeArgs {
  uint passIndex;
};

// Atmosphere parameters for Hillaire physically-based atmospheric scattering.
struct AtmosphereArgs {
  vec3 sunDirection;
  float planetRadius;  // in km
  
  // Illuminance driving the light scattered by air molecules. In the Physical coefficient mode this
  // carries the Rayleigh sky's spectral weights, which differ slightly from the direct sun's
  // (sunDiscIlluminance) and from the aerosol's (sunIlluminanceAerosol).
  vec3 sunIlluminance;
  float atmosphereThickness;  // in km
  
  vec3 rayleighScattering;
  float mieAnisotropy;  // Aerosol phase asymmetry g [-1, 1]
  
  vec3 mieScattering;  // Ground level scattering coefficients (km^-1)
  float sunRayBrightness;  // Multiplier for direct sun ray brightness

  // Aerosols absorb as well as scatter (paper Table 1), so Mie extinction is scattering + absorption
  vec3 mieAbsorption;  // Ground level absorption coefficients (km^-1)
  uint sunDiscEnabled; // Draw the sun disc into the environment on camera / mirror rays

  // Ozone absorption (important for realistic sunset colors per Hillaire paper Section 3.4)
  vec3 ozoneAbsorption;  // Absorption coefficients (km^-1)
  float ozoneLayerAltitude;  // Peak altitude of the ozone tent profile (km)
  
  uint transmittanceLutWidth;
  uint transmittanceLutHeight;
  uint multiscatteringLutSize;
  uint skyViewLutWidth;
  
  uint skyViewLutHeight;
  float ozoneLayerWidth;  // Half-width of the ozone tent profile (km)
  float viewAltitude;     // Altitude of the baked LUTs' viewpoint (km), quantised when following the camera
  uint pad3;
  
  // Derived parameters (computed on CPU)
  float atmosphereRadius;  // planetRadius + atmosphereThickness
  float rayleighScaleHeight;  // exponential density falloff for Rayleigh (km)
  float mieScaleHeight;  // exponential density falloff of the free troposphere aerosol tail (km)
  float sunAngularRadius; // Sun angular radius in radians

  // Illuminance of the sun disc and the distant sun light (direct-sun spectral weights).
  vec3 sunDiscIlluminance;
  float groundAlbedo;  // Diffuse albedo of the virtual planet ground (paper Section 4)

  // Illuminance driving the light scattered by aerosol, whose spectral weights follow the aerosol's own
  // wavelength dependence rather than the Rayleigh sky's. Equal to sunIlluminance in the Manual mode.
  vec3 sunIlluminanceAerosol;
  uint skyViewStepCount;  // Ray march steps of the sky-view LUT bake

  // Hestroffer-Magnan limb darkening exponents per channel, I(mu) = mu^alpha. 0 = uniform disc.
  vec3 sunLimbDarkeningExponent;
  // Rayleigh depolarisation term gamma = rho / (2 - rho) of Chandrasekhar's phase function. 0 = classic.
  float rayleighPhaseGamma;

  // Boundary (mixing) layer aerosol profile: unit density up to the layer top, then the exponential
  // tail scaled by mieBoundaryLayerTailScale. 0 height = pure exponential profile (paper default).
  float mieBoundaryLayerHeight;      // km
  float mieBoundaryLayerTransition;  // km, half-width of the smooth layer top
  float mieBoundaryLayerTailScale;   // Free troposphere / mixing layer concentration ratio
  // Draine phase shape: 0 = Henyey-Greenstein, 1 = Cornette-Shanks.
  float miePhaseAlpha;

  float mieForwardPeakWeight;  // Blend weight of the narrow HG forward lobe (sun aureole), 0 disables
  float mieForwardPeakG;       // Asymmetry of that forward lobe
  uint pad0;
  uint multiscatteringStepCount;  // Ray march steps per direction of the multiple scattering bake

  // Tabulated aerosol phase function (OPAC Mie / T-matrix data), sampled by miePhase() instead of the
  // analytic lobes when enabled. The type and humidity identify the LUT's contents for the bake check.
  uint miePhaseTabulated;
  uint aerosolPhaseLutSize;       // Texel count of the phase LUT, parameterised by sqrt(theta / pi)
  uint aerosolTypeId;
  float aerosolRelativeHumidity;  // Percent

  // Aerial perspective froxel volume (camera frustum fitted, rebuilt every frame).
  // Note: RtxAtmosphere::kBakeInvariantArgsSize assumes every field from here on is camera dependent,
  // so anything that should invalidate the baked LUTs must be declared above this point.
  uint aerialPerspectiveLutSize;   // Width / height of the froxel volume, 0 when disabled
  float aerialPerspectiveDepthRange;  // Depth covered by the volume, in world units
  // Where the march starts, in world units: the global volumetrics range, which already integrates the
  // air nearer than it, or the artistic start distance option, whichever is larger.
  float aerialPerspectiveStartDistance;
  float worldUnitsPerKilometer;

  // Camera basis for the aerial perspective volume, in world units. cameraRight and cameraUp are
  // pre-scaled by the frustum half extents at unit forward distance.
  vec3 cameraPosition;
  uint isZUp;  // Non-zero when the game's world is Z-up rather than the atmosphere's internal Y-up

  vec3 cameraForward;
  float aerialPerspectiveViewAltitude;  // Exact camera altitude (km) for the per-frame volume

  vec3 cameraRight;
  uint aerialPerspectiveLutDepth;  // Slice count of the froxel volume

  vec3 cameraUp;
  uint aerialPerspectiveShadowSteps;  // Samples per froxel ray in the ray-traced volume, each tracing a sun and a sky ray; 0 = unshadowed

  // Previous frame's camera basis, for reprojecting the ray-traced volume's history.
  vec3 prevCameraPosition;
  float aerialPerspectiveShadowMaxDistance;  // Sun and sky visibility ray length, in world units

  vec3 prevCameraForward;
  float aerialPerspectiveTemporalBlend;  // History weight of the ray-traced volume, 0 = no accumulation

  vec3 prevCameraRight;
  uint aerialPerspectiveHistoryValid;  // Non-zero when the previous volume may be reprojected

  vec3 prevCameraUp;
  uint aerialPerspectiveFrameIndex;  // Drives the per-frame sample jitter of the ray-traced volume

  // Stylisation: multiplies the aerosol coefficients inside the aerial perspective marches only, so haze
  // on geometry can be thickened without touching the sky, the sun or the baked LUTs. 1 = physical.
  float aerialPerspectiveAerosolScale;
  // Linear view Z the G-buffer writes for a primary miss, which the tile depth pass treats as unbounded.
  float aerialPerspectiveMissLinearViewZ;
  float pad1;
  float pad2;
};

#ifdef __cplusplus
static_assert((sizeof(AtmosphereArgs) & 15) == 0);
#endif
