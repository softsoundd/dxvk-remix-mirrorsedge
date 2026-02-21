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

// Sky View LUT computation for Hillaire atmospheric scattering
// Precomputes sky appearance for different view directions from camera position

#include "rtx/pass/atmosphere/atmosphere_common.slangh"

layout(binding = 0)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

layout(binding = 1)
Texture2D<float4> transmittanceLutInput;

layout(binding = 2)
Texture2D<float4> multiscatteringLutInput;

layout(binding = 3)
SamplerState lutSampler;

layout(binding = 4, rgba16f)
RWTexture2D<float4> skyViewLutOutput;

// Compute sky color for a given view direction
vec3 computeSkyView(vec2 uv, AtmosphereArgs args) {
  // Assume camera at ground level
  float cameraAltitude = 0.0f;
  vec3 cameraPos = vec3(0.0f, cameraAltitude, 0.0f);
  
  // Convert UV to spherical coordinates
  // uv.x: azimuth [0, 2π]
  // uv.y: elevation [-π/2, π/2] mapped from [0, 1]
  float azimuth = uv.x * 2.0f * kPi;
  float elevation = (uv.y - 0.5f) * kPi;
  
  // Construct view direction
  float cosElevation = cos(elevation);
  float sinElevation = sin(elevation);
  vec3 viewDir = vec3(
    cosElevation * cos(azimuth),
    sinElevation,
    cosElevation * sin(azimuth)
  );
  viewDir = normalize(viewDir);
  
  // Compute atmosphere radiance along this ray
  vec3 sunDir = args.sunDirection;
  vec3 radiance = evalAtmosphereRadiance(
    cameraPos,
    viewDir,
    sunDir,
    args,
    32  // numSamples
  );
  
  return radiance;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 threadId : SV_DispatchThreadID) {
  AtmosphereArgs args = atmosphereArgs;
  
  // Check bounds
  if (threadId.x >= args.skyViewLutWidth || threadId.y >= args.skyViewLutHeight) {
    return;
  }
  
  // Convert pixel coordinates to UV
  vec2 uv = (vec2(threadId) + 0.5f) / vec2(args.skyViewLutWidth, args.skyViewLutHeight);
  
  // Compute sky appearance for this direction
  vec3 skyColor = computeSkyView(uv, args);
  
  // Write result
  skyViewLutOutput[threadId] = float4(skyColor, 1.0f);
}
