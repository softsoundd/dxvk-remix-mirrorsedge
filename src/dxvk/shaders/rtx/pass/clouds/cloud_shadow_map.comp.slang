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

// Far field cloud shadow map: the optical depth along the sun through the whole layer, indexed by where the sun
// ray enters the layer's base. It carries the clouds' shadows on the city and the air beyond the near optical
// depth grid's window: distant city blocks under a deck, crepuscular rays far down the view. Toroidal in the
// field's frame like the grids, rows interleaved over frames while the field evolves; the strips a moving window
// brings in bake whole (period 1, from phase in rows and offset in columns).

#include "rtx/pass/clouds/cloud_binding_indices.h"
#include "rtx/pass/clouds/cloud_field.slangh"

layout(binding = CLOUD_BINDING_CONSTANTS)
ConstantBuffer<CloudConstants> constants;

layout(binding = CLOUD_BINDING_VOLUME_SAMPLER)
SamplerState volumeSampler;

layout(binding = CLOUD_BINDING_NVDF)
Texture3D<float> nvdf;

layout(binding = CLOUD_BINDING_DETAIL_NOISE)
Texture3D<float4> detailNoise;

layout(binding = CLOUD_BINDING_BAKE_OUTPUT, r16f)
RWTexture2D<float> shadowMapOutput;

layout(push_constant)
ConstantBuffer<CloudPassArgs> passArgs;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
  const CloudArgs cloud = constants.cloud;
  const AtmosphereArgs atmosphere = constants.atmosphere;

  const uint period = max(passArgs.period, 1u);
  const uint2 texel = uint2(dispatchId.x + passArgs.offset, cloudInterleavedTexel(dispatchId.y, period, passArgs.phase)) % uint2(CLOUD_SHADOW_MAP_SIZE);

  const float extentKm = max(cloud.shadowMapExtentKm, 1e-3f);
  const vec2 uv = (vec2(texel) + 0.5f) / float(CLOUD_SHADOW_MAP_SIZE);
  const vec2 fieldXz = vec2(
    cloudWindowTexelToFieldKm(uv.x, cloud.gridOriginKm.z, extentKm),
    cloudWindowTexelToFieldKm(uv.y, cloud.gridOriginKm.w, extentKm));
  const vec3 baseKm = vec3(fieldXz.x + cloud.windOffsetKm.x, cloud.baseAltitudeKm + 1e-3f, fieldXz.y + cloud.windOffsetKm.y);

  const vec3 sunDir = normalize(atmosphere.sunDirection);
  if (sunDir.y <= 1e-3f) {
    shadowMapOutput[texel] = kCloudOccludedOpticalDepth;
    return;
  }

  const float tExit = cloudSlabExitDistance(baseKm.y, sunDir.y, cloud, atmosphere.planetRadius);
  const float span = min(max(tExit, 0.0f), 40.0f);
  const int taps = 20;
  const float dt = span / float(taps);
  // Anchored to the field, so the map does not shimmer between bakes.
  const float jitter = cloudHash31(vec3(fieldXz.x, baseKm.y, fieldXz.y) * 97.1f);

  float opticalDepth = 0.0f;
  for (int i = 0; i < taps; ++i) {
    const float t = (float(i) + jitter) * dt;
    const vec3 p = vec3(baseKm.x + sunDir.x * t, cloudAltitudeAlong(baseKm.y, sunDir.y, t, atmosphere.planetRadius), baseKm.z + sunDir.z * t);
    const vec2 controls = cloudControls(p.xz, cloud);
    opticalDepth += cloudSampleExtinctionDensity(p, controls.x, controls.y, true, cloud, nvdf, detailNoise, volumeSampler) * dt;
  }

  shadowMapOutput[texel] = min(opticalDepth, kCloudOccludedOpticalDepth);
}
