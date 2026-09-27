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

// Aerial perspective volume (paper Section 5.4).
//
// A froxel volume fitted to the camera frustum holding, per slice, the atmospheric in-scattered
// luminance toward the camera in RGB and the mean transmittance in A. Applying it to a shaded pixel
// is then a single multiply-add, which is what gives distant geometry its haze and desaturation.
//
// Unlike the paper this runs alongside Remix's global volumetrics froxel grid, which already
// integrates a homogeneous medium out to its own range. The march therefore starts at
// aerialPerspectiveStartDistance so the two systems hand off instead of stacking.

#include "rtx/pass/atmosphere/atmosphere_common.slangh"

layout(binding = 0)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

layout(binding = 1)
Texture2D<float4> transmittanceLutInput;

layout(binding = 2)
Texture2D<float4> multiscatteringLutInput;

layout(binding = 3, rgba16f)
RWTexture3D<float4> aerialPerspectiveLutOutput;

layout(binding = 4)
Texture2D<float> aerialPerspectiveTileDepthInput;

layout(binding = 5)
Texture2D<float4> aerosolPhaseLutInput;

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 threadId : SV_DispatchThreadID) {
  const AtmosphereArgs args = aerialPerspectiveMediumArgs(atmosphereArgs);

  const uint3 size = uint3(args.aerialPerspectiveLutSize, args.aerialPerspectiveLutSize, args.aerialPerspectiveLutDepth);
  if (any(threadId >= size)) {
    return;
  }

  const vec3 uvw = (vec3(threadId) + 0.5f) / vec3(size);

  const vec3 rayVectorWorld = aerialPerspectiveUvwToWorldPosition(uvw, args) - args.cameraPosition;
  const float rayLengthWorld = length(rayVectorWorld);
  if (rayLengthWorld <= 0.0f) {
    aerialPerspectiveLutOutput[threadId] = float4(0.0f, 0.0f, 0.0f, 1.0f);
    return;
  }

  const vec3 rayDir = aerialPerspectiveWorldToAtmosphereDirection(rayVectorWorld / rayLengthWorld, args);

  const float kmPerWorldUnit = 1.0f / max(args.worldUnitsPerKilometer, 1e-6f);
  const vec3 rayOrigin = aerialPerspectiveRayOrigin(args);

  // Only the portion beyond the global volumetrics range and short of the tile's farthest surface
  // belongs to this froxel.
  const float maxRayLengthWorld = aerialPerspectiveMaxRayLength(
    aerialPerspectiveTileDepthInput.Load(int3(threadId.xy, 0)), uvw.xy, args);
  const float startDistanceKm = args.aerialPerspectiveStartDistance * kmPerWorldUnit;
  const float endDistanceKm = min(rayLengthWorld, maxRayLengthWorld) * kmPerWorldUnit;

  vec3 transmittance = vec3(1.0f);
  vec3 inScatter = vec3(0.0f, 0.0f, 0.0f);

  if (endDistanceKm > startDistanceKm) {
    // Fewer steps than the sky, since each froxel covers a short segment and neighbouring slices
    // are filtered together on lookup.
    inScatter = evalAtmosphereRadianceSegment(
      rayOrigin, rayDir, args.sunDirection, args,
      transmittanceLutInput, multiscatteringLutInput, aerosolPhaseLutInput,
      12, endDistanceKm, startDistanceKm, transmittance);
  }

  // A stores the mean of the wavelength dependent transmittance, per paper Section 5.4.
  const float meanTransmittance = (transmittance.x + transmittance.y + transmittance.z) * (1.0f / 3.0f);

  aerialPerspectiveLutOutput[threadId] = float4(inScatter, saturate(meanTransmittance));
}
