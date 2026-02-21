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

// Multiscattering LUT computation for Hillaire atmospheric scattering
// Approximates multiple scattering contributions

#include "rtx/pass/atmosphere/atmosphere_common.slangh"

layout(binding = 0)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

layout(binding = 1)
Texture2D<float4> transmittanceLutInput;

layout(binding = 2)
SamplerState transmittanceSampler;

layout(binding = 3, rgba16f)
RWTexture2D<float4> multiscatteringLutOutput;

// Ground albedo for reflected light (Blender uses 0.3)
static const vec3 GROUND_ALBEDO = vec3(0.3f, 0.3f, 0.3f);
static const float PHASE_ISOTROPIC = 1.0f / (4.0f * kPi);

// Compute ground reflection contribution (from Blender's sky_multiple_scattering.cpp)
// This is crucial for realistic sunset colors - light bouncing off the ground
vec3 computeGroundReflection(float altitude, float sunZenithCosAngle, AtmosphereArgs args) {
  // Distance from sample point to planet center
  float d = args.planetRadius + altitude;
  
  // Solid angle subtended by the planet from a point at d distance from center
  // omega = 2π(1 - sqrt(1 - (R/d)²))
  float ratio = args.planetRadius / d;
  float omega = 2.0f * kPi * (1.0f - sqrt(max(0.0f, 1.0f - ratio * ratio)));
  
  // Get transmittance from ground to sun (sun zenith at ground level)
  vec3 T_to_ground = sampleTransmittanceLut(transmittanceLutInput, transmittanceSampler,
                                             0.0f, sunZenithCosAngle, args);
  
  // Get transmittance ratio: T(ground,sample) = T(ground,sun) / T(sample,sun)
  vec3 T_ground_to_sun = sampleTransmittanceLut(transmittanceLutInput, transmittanceSampler,
                                                 0.0f, 1.0f, args);  // Zenith at ground
  vec3 T_sample_to_sun = sampleTransmittanceLut(transmittanceLutInput, transmittanceSampler,
                                                 altitude, max(sunZenithCosAngle, 0.01f), args);
  vec3 T_ground_to_sample = T_ground_to_sun / max(T_sample_to_sun, vec3(1e-6f));
  
  // 2nd order scattering from the ground (Lambertian reflection)
  // L_ground = (1/4π) * omega * (albedo/π) * T_to_ground * T_ground_to_sample * cos_theta
  float cosTheta = max(sunZenithCosAngle, 0.0f);
  vec3 L_ground = PHASE_ISOTROPIC * omega * (GROUND_ALBEDO / kPi) * T_to_ground * T_ground_to_sample * cosTheta;
  
  return L_ground;
}

// Analytical fit for higher-order atmospheric scattering (from Blender)
// This approximation is wavelength-dependent for accurate color
vec3 computeAnalyticalMultiscattering(float sunZenithCosAngle) {
  // Blender's fitted coefficients - but toned down to avoid excessive purple at sunset
  // Original Blender values were too strong for our RGB approximation
  // Using more neutral colors to prevent purple tint when combined with orange sunset
  vec3 L_ms_color = vec3(0.35f, 0.38f, 0.45f);  // More neutral, less blue-heavy
  
  // Analytical fit: color * (1 / (1 + 5 * exp(-17.92 * cos_theta)))
  // This models how multiple scattering increases as sun gets higher
  // Reduced overall strength to 0.01 (from 0.02) to prevent purple cast
  float fitTerm = 1.0f / (1.0f + 5.0f * exp(-17.92f * sunZenithCosAngle));
  
  return 0.01f * L_ms_color * fitTerm;
}

// Compute multiscattering contribution at given altitude and sun zenith angle
vec3 computeMultiscattering(float altitude, float sunZenithCosAngle, AtmosphereArgs args) {
  vec3 planetCenter = vec3(0.0f, -args.planetRadius, 0.0f);
  vec3 rayOrigin = vec3(0.0f, altitude, 0.0f);
  
  // Integrate over hemisphere for volumetric scattering
  const int numDirSamples = 64;
  const int numDistanceSamples = 20;
  
  vec3 multiScatter = vec3(0.0f);
  
  for (int i = 0; i < numDirSamples; ++i) {
    float zenithAngle = kPi * (float(i) + 0.5f) / float(numDirSamples);
    float cosZenith = cos(zenithAngle);
    float sinZenith = sin(zenithAngle);
    
    vec3 rayDir = vec3(sinZenith, cosZenith, 0.0f);
    
    // Intersect with atmosphere
    float t0, t1;
    if (!intersectSphere(rayOrigin, rayDir, planetCenter, args.atmosphereRadius, t0, t1)) {
      continue;
    }
    
    t0 = max(0.0f, t0);
    
    // Check planet intersection
    float tPlanet0, tPlanet1;
    bool hitsPlanet = intersectSphere(rayOrigin, rayDir, planetCenter, args.planetRadius, tPlanet0, tPlanet1);
    if (hitsPlanet && tPlanet0 > 0.0f) {
      t1 = tPlanet0;
    }
    
    float dt = (t1 - t0) / float(numDistanceSamples);
    
    // March along ray
    for (int j = 0; j < numDistanceSamples; ++j) {
      float t = t0 + (float(j) + 0.5f) * dt;
      vec3 samplePos = rayOrigin + rayDir * t;
      float sampleAltitude = length(samplePos - planetCenter) - args.planetRadius;
      
      float densityR = rayleighDensity(sampleAltitude, args);
      float densityM = mieDensity(sampleAltitude, args);
      float densityO3 = ozoneDensity(sampleAltitude, args);
      
      vec3 scatteringR = args.rayleighScattering * densityR;
      vec3 scatteringM = args.mieScattering * densityM;
      vec3 scattering = scatteringR + scatteringM;
      
      // Note: Ozone only absorbs, doesn't scatter, so it's not added to scattering
      // but it does affect the sun transmittance sampled from the LUT
      
      // Sample transmittance to sun
      vec3 toSun = vec3(0.0f, sunZenithCosAngle, sqrt(max(0.0f, 1.0f - sunZenithCosAngle * sunZenithCosAngle)));
      float sunZenithCos = dot(normalize(samplePos - planetCenter), toSun);
      vec3 sunTransmittance = sampleTransmittanceLut(transmittanceLutInput, transmittanceSampler, 
                                                      sampleAltitude, sunZenithCos, args);
      
      multiScatter += scattering * sunTransmittance * dt * sinZenith;
    }
  }
  
  // Normalize by hemisphere solid angle
  multiScatter *= 2.0f * kPi / float(numDirSamples);
  
  // Add ground reflection contribution (key for sunset colors!)
  vec3 groundReflection = computeGroundReflection(altitude, sunZenithCosAngle, args);
  
  // Add analytical fit for higher-order scattering (wavelength-dependent)
  vec3 analyticalMS = computeAnalyticalMultiscattering(sunZenithCosAngle);
  
  return multiScatter + groundReflection + analyticalMS;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 threadId : SV_DispatchThreadID) {
  AtmosphereArgs args = atmosphereArgs;
  
  // Check bounds
  if (threadId.x >= args.multiscatteringLutSize || threadId.y >= args.multiscatteringLutSize) {
    return;
  }
  
  // Convert pixel to parameters
  // X axis: sun zenith angle cos
  // Y axis: altitude
  vec2 uv = (vec2(threadId) + 0.5f) / float(args.multiscatteringLutSize);
  
  float sunZenithCosAngle = uv.x * 2.0f - 1.0f;  // [-1, 1]
  float altitude = uv.y * args.atmosphereThickness;  // [0, atmosphereThickness]
  
  // Compute multiscattering
  vec3 multiScatter = computeMultiscattering(altitude, sunZenithCosAngle, args);
  
  // Write result
  multiscatteringLutOutput[threadId] = float4(multiScatter, 1.0f);
}
