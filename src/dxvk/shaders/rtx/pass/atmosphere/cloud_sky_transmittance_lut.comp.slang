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

// Cloud-occluded sky-ambient transmittance LUT bake (fork).
//
// For each (azimuth, elevation) cell, marches the cloud slab from camera
// position along that direction and stores cloud transmittance in [0, 1].
// Output is sampled by the volumetric pass's sky-ambient hemisphere
// integration to attenuate sky-view-LUT radiance per direction by cloud
// coverage along that direction.
//
// LUT parameterization mirrors sky_view_lut.comp.slang exactly so a single
// UV computation in the consumer samples both LUTs consistently:
//   uv.x: azimuth   in [0, 2pi]
//   uv.y: elevation in [-pi/2, pi/2] mapped from [0, 1]
//
// Energy model is the single-coverage Beer-Lambert approximation used by
// evalCloudGroundShadow (atmosphere_common.slangh:653) for the sun direction.
// Reusing the same model keeps sun-shadowed and sky-ambient-shadowed
// transmittance consistent: the directional cumulus footprint visible in
// this LUT matches what the sun path sees.

#include "rtx/pass/atmosphere/atmosphere_common.slangh"

layout(binding = 0)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

layout(binding = 1, r16f)
RWTexture2D<float> cloudSkyTransmittanceLutOutput;

// LUT dimensions (kept in lockstep with RtxAtmosphere::kCloudSkyTransmittanceLutWidth/Height)
static const uint kLutWidth = 32u;
static const uint kLutHeight = 16u;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 threadId : SV_DispatchThreadID) {
  if (threadId.x >= kLutWidth || threadId.y >= kLutHeight) {
    return;
  }

  AtmosphereArgs args = atmosphereArgs;

  // Pixel -> UV -> (azimuth, elevation). Matches sky_view_lut.comp.slang:52-53.
  vec2 uv = (vec2(threadId) + 0.5f) / vec2(kLutWidth, kLutHeight);
  float azimuth = uv.x * 2.0f * kPi;
  float elevation = (uv.y - 0.5f) * kPi;

  vec3 dirYUp = vec3(
    cos(elevation) * cos(azimuth),
    sin(elevation),
    cos(elevation) * sin(azimuth));
  dirYUp = normalize(dirYUp);

  // Below-horizon directions: no cloud occlusion (consumer clamps elevation >= 0
  // anyway). 1.0 stored as a safety floor so a stray below-horizon sample
  // doesn't darken the sky-ambient term.
  if (dirYUp.y <= 0.001f) {
    cloudSkyTransmittanceLutOutput[threadId] = 1.0f;
    return;
  }

  // Cloud system disabled or trivially clear: no occlusion.
  if (args.cloudEnabled < 0.5f || args.cloudCoverageMean < 0.01f) {
    cloudSkyTransmittanceLutOutput[threadId] = 1.0f;
    return;
  }

  // Intersect the cloud slab. intersectCloudLayer's dir.y > 0.001 gate above
  // means this returns true for all sample directions we keep.
  float t;
  vec3 hitPos;
  if (!intersectCloudLayer(dirYUp, args, t, hitPos)) {
    cloudSkyTransmittanceLutOutput[threadId] = 1.0f;
    return;
  }

  // Single-coverage Beer-Lambert. Structurally mirrors evalCloudGroundShadow at
  // atmosphere_common.slangh:682-686 -- coverage tap at hitXZ gives each
  // direction its own coverageLocal so cumulus footprints stay visible in the
  // LUT -- but DROPS the `* 4.0f` multiplier the sun-shadow path uses. That
  // multiplier was tuned for `evalCloudGroundShadow`'s use where the result
  // is then wrapped in `mix(1, transmittance, cloudShadowStrength)` and the
  // default `cloudShadowStrength = 0` masks the strong extinction. The sky-
  // ambient path has no such mix, so the 4x crushes ambient to invisibility
  // at the default coverage (0.64) and density (1.8). `exp(-cov * density)`
  // at those defaults = ~0.32 -- a visible ~3x darkening, not extinction.
  vec2 hitXZ = vec2(hitPos.x, hitPos.z);
  float coverageNoise = cloudControlField2D(hitXZ * args.cloudCoverageNoiseScale, 250.0f);
  float coverageLocal = saturate(
    args.cloudCoverageMean + (coverageNoise - 0.5f) * args.cloudCoverageSpread);

  float transmittance = exp(-coverageLocal * args.cloudDensity);

  cloudSkyTransmittanceLutOutput[threadId] = transmittance;
}
