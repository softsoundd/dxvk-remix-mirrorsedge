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

// The fast lens flare adds the sun's ghosts to the linear HDR image, after Lee and Eisemann 2013. At each of a
// ghost's wavelengths, every pixel maps back to the point on the entrance plane its light came from, by the ghost's
// rays as fitted to a few real ones, and on to the stop paraxially. It is lit if that ray passes both the iris and
// every clear aperture on the ghost's path that cuts it. Each tile first culls the ghosts whose bounds miss it.

#include "rtx/pass/lens_flare/lens_flare.h"
#include "rtx/utility/lens_aperture.slangh"

layout(binding = LENS_FLARE_GHOSTS_INPUT)
StructuredBuffer<LensFlareGhost> Ghosts;

layout(binding = LENS_FLARE_SUN_VISIBILITY_INPUT)
Texture2D<float4> InSunVisibility;

layout(binding = LENS_FLARE_COLOR_INPUT_OUTPUT)
RWTexture2D<float4> InOutColor;

layout(push_constant)
ConstantBuffer<LensFlareArgs> cb;

static const uint kTileSize = LENS_FLARE_TILE_SIZE;

groupshared uint g_tileGhostCount;
groupshared uint g_tileGhosts[LENS_FLARE_MAX_GHOSTS];

vec2 pixelToNdcAspect(vec2 pixel)
{
  const vec2 ndc = vec2(pixel.x * cb.invImageSize.x * 2.0f - 1.0f, 1.0f - pixel.y * cb.invImageSize.y * 2.0f);

  return vec2(ndc.x * cb.aspectRatio, ndc.y);
}

// One wavelength of a ghost at sensor position p, in mm in the frame of the sun's meridian. The ghost lies inside the
// iris's image and the discs of the entrance plane its path's clear apertures pass, all centred on the meridian. Its
// edges blend over edgeMm on the sensor, mapped back to both. irisRotation is the iris's in the meridian's frame.
float ghostCoverage(vec2 p, LensFlareGhostSample ghost, float edgeMm, float irisRotation, uint discCount, vec2 discs[LENS_FLARE_PASS_DISCS])
{
  const vec2 e = vec2(p.x - ghost.sensorOffset, p.y) * ghost.entranceFromSensor;
  const vec2 a = e * ghost.stopFromEntrance + vec2(ghost.stopOffset, 0.0f);
  const vec2 irisPoint = a / cb.stopRadius;

  const float edgeOnEntrance = max(edgeMm * ghost.entrancePerSensor, 1e-6f);
  float pass = 1.0f;

  for (uint k = 0; k < discCount; k++)
  {
    pass = min(pass, saturate(0.5f - (length(vec2(e.x - discs[k].x, e.y)) - discs[k].y) / edgeOnEntrance));
  }

  // The discs are far cheaper than the iris.
  if (pass <= 0.0f)
  {
    return 0.0f;
  }

  const float irisEdge = max(edgeOnEntrance * abs(ghost.stopFromEntrance) / cb.stopRadius, 1e-4f);
  const float iris = lensApertureCoverage(irisPoint, cb.apertureBlades, cb.apertureCurvature, irisRotation, cb.apertureDistanceJitter,
                                          cb.apertureAngleJitter, irisEdge);

  return iris * pass;
}

bool ghostOverlapsRect(uint ghostIndex, vec2 rectMin, vec2 rectMax)
{
  const vec2 center = Ghosts[ghostIndex].boundCenter;
  const vec2 closest = clamp(center, rectMin, rectMax);

  return length(closest - center) <= Ghosts[ghostIndex].boundRadius;
}

[shader("compute")]
[numthreads(kTileSize, kTileSize, 1)]
void main(uint2 pixel : SV_DispatchThreadID, uint2 groupId : SV_GroupID, uint threadIndex : SV_GroupIndex)
{
  if (threadIndex == 0)
  {
    g_tileGhostCount = 0;
  }

  GroupMemoryBarrierWithGroupSync();

  if (threadIndex < cb.ghostCount)
  {
    // Rows run down while NDC runs up, so the tile's corners swap in y.
    const vec2 cornerA = pixelToNdcAspect(vec2(groupId * kTileSize));
    const vec2 cornerB = pixelToNdcAspect(vec2((groupId + 1) * kTileSize));

    if (ghostOverlapsRect(threadIndex, min(cornerA, cornerB), max(cornerA, cornerB)))
    {
      uint slot;
      InterlockedAdd(g_tileGhostCount, 1, slot);
      g_tileGhosts[slot] = threadIndex;
    }
  }

  GroupMemoryBarrierWithGroupSync();

  if (any(pixel >= cb.imageSize))
  {
    return;
  }

  const vec2 ndcAspect = pixelToNdcAspect(vec2(pixel) + 0.5f);
  const vec2 sensorPosition = -ndcAspect * cb.sensorScale;
  const float pixelMm = 2.0f * cb.invImageSize.y * cb.sensorScale;

  // The ghosts' maps lie in the frame of the sun's meridian, the direction its rays tilt in.
  const float rayAngle = length(cb.sunRayAngle);
  const vec2 meridian = rayAngle > 0.0f ? cb.sunRayAngle / rayAngle : vec2(1.0f, 0.0f);
  const vec2 meridianPosition = vec2(dot(sensorPosition, meridian), meridian.x * sensorPosition.y - meridian.y * sensorPosition.x);
  const float irisRotation = cb.apertureRotation - atan2(meridian.y, meridian.x);

  vec3 flare = vec3(0.0f);

  for (uint i = 0; i < g_tileGhostCount; i++)
  {
    const uint ghostIndex = g_tileGhosts[i];
    const uint sampleCount = Ghosts[ghostIndex].sampleCount;
    const float edgeMm = max(pixelMm, Ghosts[ghostIndex].edgeSpread);
    const uint discCount = Ghosts[ghostIndex].passDiscCount;
    const vec2 discs[LENS_FLARE_PASS_DISCS] = Ghosts[ghostIndex].passDiscs;

    for (uint k = 0; k < sampleCount; k++)
    {
      const LensFlareGhostSample ghost = Ghosts[ghostIndex].samples[k];

      flare += ghost.radiance * ghostCoverage(meridianPosition, ghost, edgeMm, irisRotation, discCount, discs);
    }
  }

  flare *= cb.sourceIlluminance * InSunVisibility.Load(int3(0, 0, 0)).rgb;

  vec4 color = InOutColor[pixel];

  if (cb.debugView == LENS_FLARE_DEBUG_VIEW_FLARE_ONLY)
  {
    color.rgb = flare;
  }
  else
  {
    color.rgb += flare;
  }

  if (cb.debugView == LENS_FLARE_DEBUG_VIEW_GHOST_BOUNDS)
  {
    const float lineWidth = 3.0f * cb.invImageSize.y;

    for (uint i = 0; i < cb.ghostCount; i++)
    {
      const float distance = length(ndcAspect - Ghosts[i].boundCenter);

      if (abs(distance - Ghosts[i].boundRadius) < lineWidth)
      {
        color.rgb = vec3(0.0f, 1.0f, 0.0f);
      }
    }
  }

  InOutColor[pixel] = color;
}
