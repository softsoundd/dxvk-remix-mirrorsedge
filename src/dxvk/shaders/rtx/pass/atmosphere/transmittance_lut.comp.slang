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

// Transmittance LUT computation for Hillaire atmospheric scattering
// Precomputes transmittance from any point in atmosphere looking in any direction

#include "rtx/pass/atmosphere/atmosphere_common.slangh"

layout(binding = 0)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

layout(binding = 1, rgba16f)
RWTexture2D<float4> transmittanceLutOutput;

// Compute transmittance along a ray through the atmosphere
vec3 computeTransmittance(float altitude, float viewZenithCosAngle, AtmosphereArgs args) {
  // Safety clamps
  altitude = clamp(altitude, 0.0f, args.atmosphereThickness);
  viewZenithCosAngle = clamp(viewZenithCosAngle, -1.0f, 1.0f);
  
  vec3 planetCenter = vec3(0.0f, -args.planetRadius, 0.0f);
  float r = altitude + args.planetRadius;
  vec3 rayOrigin = vec3(0.0f, altitude, 0.0f);
  
  // Construct ray direction from cosine of zenith angle
  float sinZenith = sqrt(max(0.0f, 1.0f - viewZenithCosAngle * viewZenithCosAngle));
  vec3 rayDir = vec3(sinZenith, viewZenithCosAngle, 0.0f);
  rayDir = normalize(rayDir);
  
  // Intersect with top of atmosphere
  float t0, t1;
  if (!intersectSphere(rayOrigin, rayDir, planetCenter, args.atmosphereRadius, t0, t1)) {
    return vec3(1.0f);  // No atmosphere intersection
  }
  
  // Check if ray hits planet
  float tPlanet0, tPlanet1;
  bool hitsPlanet = intersectSphere(rayOrigin, rayDir, planetCenter, args.planetRadius, tPlanet0, tPlanet1);
  if (hitsPlanet && tPlanet0 > 0.0f) {
    return vec3(0.0f);  // Ray hits planet, full absorption
  }
  
  // March along ray and accumulate optical depth
  const int numSamples = 40;
  float tStart = max(0.0f, t0);
  float tEnd = t1;
  
  // Safety check
  if (tEnd <= tStart) {
    return vec3(1.0f);
  }
  
  float dt = (tEnd - tStart) / float(numSamples);
  
  vec3 opticalDepth = vec3(0.0f);
  
  for (int i = 0; i < numSamples; ++i) {
    float sampleT = tStart + (float(i) + 0.5f) * dt;
    vec3 samplePos = rayOrigin + rayDir * sampleT;
    float sampleAltitude = length(samplePos - planetCenter) - args.planetRadius;
    
    // Clamp to valid range
    sampleAltitude = clamp(sampleAltitude, 0.0f, args.atmosphereThickness);
    
    float densityR = rayleighDensity(sampleAltitude, args);
    float densityM = mieDensity(sampleAltitude, args);
    float densityO3 = ozoneDensity(sampleAltitude, args);
    
    vec3 extinctionR = args.rayleighScattering * densityR;
    vec3 extinctionM = args.mieScattering * densityM;
    vec3 extinctionO3 = args.ozoneAbsorption * densityO3;  // Ozone only absorbs
    vec3 extinction = extinctionR + extinctionM + extinctionO3;
    
    opticalDepth += extinction * dt;
  }
  
  // Transmittance = exp(-optical depth)
  // Clamp optical depth to prevent numerical issues
  opticalDepth = min(opticalDepth, vec3(1e3f));
  vec3 transmittance = exp(-opticalDepth);
  
  // Ensure valid output
  return clamp(transmittance, vec3(0.0f), vec3(1.0f));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 threadId : SV_DispatchThreadID) {
  AtmosphereArgs args = atmosphereArgs;
  
  // Check bounds
  if (threadId.x >= args.transmittanceLutWidth || threadId.y >= args.transmittanceLutHeight) {
    return;
  }
  
  // Convert pixel coordinates to UV
  vec2 uv = (vec2(threadId) + 0.5f) / vec2(args.transmittanceLutWidth, args.transmittanceLutHeight);
  
  // Convert UV to atmosphere parameters
  float viewZenithCosAngle, altitude;
  uvToTransmittanceLutParams(uv, args, viewZenithCosAngle, altitude);
  
  // Compute transmittance
  vec3 transmittance = computeTransmittance(altitude, viewZenithCosAngle, args);
  
  // Write result
  transmittanceLutOutput[threadId] = float4(transmittance, 1.0f);
}
