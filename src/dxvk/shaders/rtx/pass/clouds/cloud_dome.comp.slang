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

// The cloud layer over the whole sky as seen from the camera, in the sky-view LUT's parameterisation: what the
// path tracer's misses (indirect rays, reflections), the volumetrics' sky light and the surfaces' sky
// visibility read. Premultiplied radiance in rgb, transmittance in a, so a miss composites C + T sky exactly
// like the screen pass. Marched at lower quality than the screen and blended with its history, which the wind
// moves slowly enough to keep: each frame refreshes one texel of every 2x2 block (a period of 4) or of every
// checkerboard pair (2), and the rest carry their history over.

#include "rtx/pass/clouds/cloud_march.slangh"

layout(binding = CLOUD_BINDING_BAKE_INPUT)
Texture2D<float4> domeHistory;

layout(binding = CLOUD_BINDING_BAKE_OUTPUT, rgba16f)
RWTexture2D<float4> domeOutput;

layout(push_constant)
ConstantBuffer<CloudPassArgs> passArgs;

uint domeRefreshPhase(uint2 texel, uint period) {
  return period >= 4u ? ((texel.x & 1u) | ((texel.y & 1u) << 1u)) % period : (texel.x + texel.y) % period;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
  uint width, height;
  domeOutput.GetDimensions(width, height);
  if (any(threadId.xy >= uint2(width, height))) {
    return;
  }
  if (passArgs.level != 0u && domeRefreshPhase(threadId.xy, max(passArgs.period, 1u)) != passArgs.phase) {
    domeOutput[threadId.xy] = domeHistory[threadId.xy];
    return;
  }

  const CloudArgs cloud = CloudCb.cloud;
  const AtmosphereArgs atmosphere = CloudCb.atmosphere;

  const vec2 uv = (vec2(threadId.xy) + 0.5f) / vec2(width, height);
  const vec3 direction = cloudSkyUvToDirection(uv);

  CloudMarchSettings settings;
  settings.maxSamples = max(cloud.viewSamplesMax / 2u, 32u);
  settings.stepKm = max(cloud.viewStepKm * 2.0f, 0.2f);
  settings.minStepKm = max(cloud.adaptiveStepKm * 4.0f, 0.1f);
  settings.maxDistanceKm = 1e9f;
  settings.footprintPerKm = 2.0f * kPi / float(width);
  settings.fineShadowTaps = min(cloud.shadowTaps, 1u);
  settings.cameraDetail = false;
  settings.fringeStepScale = 1.0f;

  const float jitter = frac(cloudHash31(vec3(vec2(threadId.xy), float(cloud.frameIndex & 1023u))) + 0.5f);
  const CloudMarchResult result = cloudMarch(direction, settings, jitter, cloud, atmosphere);

  vec4 value = vec4(result.radiance, result.transmittance);
  if (passArgs.level) {
    value = lerp(domeHistory[threadId.xy], value, saturate(cloud.domeBlend));
  }
  domeOutput[threadId.xy] = value;
}
