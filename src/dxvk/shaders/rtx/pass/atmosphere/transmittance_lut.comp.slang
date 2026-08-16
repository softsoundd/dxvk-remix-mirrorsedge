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

// Transmittance from (altitude, viewZenithCosAngle) to the atmosphere's top boundary.
//
// Note: the planet is deliberately ignored here. Rays that would pass through the ground still get
// an entry, integrated with the density clamped at ground level, so bilinear filtering near the
// horizon does not bleed hard zeros into valid neighbours. Occlusion by the planet is a separate
// visibility term (evalPlanetShadow, the Vis factor of Equation 4).
vec3 computeTransmittance(float altitude, float viewZenithCosAngle, AtmosphereArgs args) {
  altitude = clamp(altitude, 0.0f, args.atmosphereThickness);
  viewZenithCosAngle = clamp(viewZenithCosAngle, -1.0f, 1.0f);

  const vec3 planetCenter = getPlanetCenter(args);
  const vec3 rayOrigin = vec3(0.0f, altitude, 0.0f);

  // Construct ray direction from cosine of zenith angle
  const float sinZenith = sqrt(max(0.0f, 1.0f - viewZenithCosAngle * viewZenithCosAngle));
  const vec3 rayDir = normalize(vec3(sinZenith, viewZenithCosAngle, 0.0f));

  // Intersect with top of atmosphere
  float t0, t1;
  if (!intersectSphere(rayOrigin, rayDir, planetCenter, args.atmosphereRadius, t0, t1)) {
    return vec3(1.0f);  // No atmosphere intersection
  }

  const float tStart = max(0.0f, t0);
  const float tEnd = t1;
  if (tEnd <= tStart) {
    return vec3(1.0f);
  }

  // Segments are shorter near the ray origin where the medium is densest.
  const int numSamples = 40;
  const float span = tEnd - tStart;
  const float invSamples = 1.0f / float(numSamples);

  vec3 opticalDepth = vec3(0.0f);
  float segmentStart = 0.0f;

  for (int i = 0; i < numSamples; ++i) {
    const float segmentEnd = atmosphereStepFraction(float(i + 1) * invSamples);
    const float dt = (segmentEnd - segmentStart) * span;
    const float sampleT = tStart + (segmentStart + (segmentEnd - segmentStart) * 0.5f) * span;
    segmentStart = segmentEnd;

    if (dt <= 0.0f) {
      continue;
    }

    const vec3 samplePos = rayOrigin + rayDir * sampleT;
    const float sampleAltitude = clamp(
      length(samplePos - planetCenter) - args.planetRadius, 0.0f, args.atmosphereThickness);

    opticalDepth += sampleAtmosphereMedium(sampleAltitude, args).extinction * dt;
  }

  // Clamp optical depth to prevent numerical issues
  opticalDepth = min(opticalDepth, vec3(1e3f));

  return clamp(exp(-opticalDepth), vec3(0.0f), vec3(1.0f));
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
