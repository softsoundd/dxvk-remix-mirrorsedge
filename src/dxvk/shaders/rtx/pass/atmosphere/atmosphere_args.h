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

  // Aerosols absorb as well as scatter (paper Table 1), so Mie extinction is scattering + absorption
  vec3 mieAbsorption;  // Absorption coefficients (km^-1)
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
  float viewAltitude;     // Camera altitude offset (km)
  uint useSkyViewLut;     // Sample the precomputed sky-view LUT at runtime instead of ray marching per miss ray
  
  // Derived parameters (computed on CPU)
  float atmosphereRadius;  // planetRadius + atmosphereThickness
  float rayleighScaleHeight;  // exponential density falloff for Rayleigh (km)
  float mieScaleHeight;  // exponential density falloff for Mie (km)
  float sunAngularRadius; // Sun angular radius in radians

  // Aerial perspective froxel volume (camera frustum fitted, rebuilt every frame).
  // Note: RtxAtmosphere::kBakeInvariantArgsSize assumes every field from here on is camera dependent,
  // so anything that should invalidate the baked LUTs must be declared above this point.
  uint aerialPerspectiveLutSize;   // Width / height / depth of the froxel volume, 0 when disabled
  float aerialPerspectiveDepthRange;  // Depth covered by the volume, in world units
  // In-scatter closer than this is already integrated by the global volumetrics froxel grid, so the
  // aerial perspective march starts here to avoid double counting. 0 when volumetrics are disabled.
  float aerialPerspectiveStartDistance;
  float worldUnitsPerKilometer;

  // Camera basis for the aerial perspective volume, in world units. cameraRight and cameraUp are
  // pre-scaled by the frustum half extents at unit forward distance.
  vec3 cameraPosition;
  uint isZUp;  // Non-zero when the game's world is Z-up rather than the atmosphere's internal Y-up

  vec3 cameraForward;
  float pad0;

  vec3 cameraRight;
  float pad1;

  vec3 cameraUp;
  float pad2;
};
