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

layout(binding = 3, rgba16f)
RWTexture2D<float4> skyViewLutOutput;

// Compute sky color for a given view direction
vec3 computeSkyView(vec2 uv, AtmosphereArgs args) {
  // Camera on the local vertical at the configured altitude, matching evalSkyRadiance().
  const vec3 cameraPos = vec3(0.0f, max(args.viewAltitude, 0.0f), 0.0f);

  // uv.x maps linearly to azimuth; uv.y uses the paper's Section 5.3 latitude compression, which
  // concentrates texels near the horizon where the sky has its highest visual frequency.
  const float azimuth = uv.x * 2.0f * kPi;
  const float elevation = skyViewLutVToElevation(uv.y);

  const float cosElevation = cos(elevation);
  const float sinElevation = sin(elevation);
  const vec3 viewDir = normalize(vec3(
    cosElevation * cos(azimuth),
    sinElevation,
    cosElevation * sin(azimuth)
  ));

  return evalAtmosphereRadiance(
    cameraPos,
    viewDir,
    args.sunDirection,
    args,
    transmittanceLutInput,
    multiscatteringLutInput,
    32  // numSamples
  );
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
