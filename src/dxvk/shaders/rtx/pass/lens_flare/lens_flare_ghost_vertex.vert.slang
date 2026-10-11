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

// Draws each traced ghost's grid as triangles, two per cell, pulling their corners from the trace. Each triangle is
// drawn over a cover of every pixel it touches, for the fragment shader to weigh by the share of the pixel it covers:
// the triangle with its edges moved out by the pixel's half extent across them, or for a sliver, whose moved corners
// would reach far, the box along its longest edge grown the same. Each cover is two triangles, the first alone for the
// triangle, and carries the attributes on the triangle's plane. Triangles with a ray the lens lost are dropped, as the
// grid reaches just past the rays that pass, unless the extension passes gave that ray a place beside the traced rays.
// So are those the fragment shader would leave dark throughout.

#include "rtx/pass/lens_flare/lens_flare.h"
#include "rtx/utility/lens_aperture.slangh"

layout(binding = LENS_FLARE_RASTER_VERTICES_INPUT)
StructuredBuffer<LensFlareTracedVertex> Vertices;

layout(binding = LENS_FLARE_RASTER_INSTANCES_INPUT)
StructuredBuffer<LensFlareTraceInstance> Instances;

layout(binding = LENS_FLARE_RASTER_SUN_VISIBILITY_INPUT)
Texture2D<float4> InSunVisibility;

layout(push_constant)
ConstantBuffer<LensFlareRasterArgs> cb;

struct VsOut
{
  float4 position : SV_Position;
  float2 stop : STOP;
  float clipRatio : CLIP;
  float3 radiance : RADIANCE;
  // How far the image moves in pixels from the sun's centre to its limb, along the sun's meridian (xy) and across it
  // (zw), and the stop coordinate's change, for a fixed entrance point.
  float4 motion : MOTION;
  float4 stopFromSun : STOP_SUN;
  // The clip ratio's change, then how much the stop coordinate and the clip ratio change to the next ray.
  float4 clipFromSun : CLIP_SUN;
  // A carrier's light of the whole spectrum past the edge band, and the share of the light each wavelength draws
  // itself.
  float4 interior : INTERIOR;
  // The loss margin, its change across the sun's disc along and across the meridian, and how much it changes to the
  // next ray.
  float4 loss : LOSS;
  // The triangle's corners in pixels.
  nointerpolation float4 corners01 : CORNERS01;
  nointerpolation float2 corner2 : CORNER2;
  // Whether every point of the triangle passes the iris, the clear apertures and the lost rays' boundary across the
  // sun's disc and the pixel.
  nointerpolation uint fullyLit : FULLY_LIT;
};

static const uint2 kCorners[6] = { uint2(0, 0), uint2(1, 0), uint2(1, 1), uint2(0, 0), uint2(1, 1), uint2(0, 1) };

LensFlareTracedVertex gridVertex(uint instance, int2 cell)
{
  return Vertices[instance * cb.gridSize * cb.gridSize + uint(cell.y) * cb.gridSize + uint(cell.x)];
}

float cross2(vec2 a, vec2 b)
{
  return a.x * b.y - a.y * b.x;
}

// The sensor's image is inverted on the screen.
vec2 toPixels(vec2 sensor)
{
  const vec2 ndcAspect = -sensor / cb.sensorScale;
  const vec2 ndc = vec2(ndcAspect.x / cb.aspectRatio, -ndcAspect.y);
  return (ndc * 0.5f + 0.5f) * cb.imageSize;
}

// Half the pixel's extent across a line of unit normal n.
float pixelReach(vec2 n)
{
  return 0.5f * (abs(n.x) + abs(n.y));
}

// Bounds over the whole triangle on the fragment shader's tests. The iris lies inside the circle through its farthest
// corner, and the stop coordinate's length changes no faster than the coordinate. Inside the iris its distance is
// convex, and the clip ratio and the loss margin are linear. Each reach is bounded by its largest at the corners with
// the triangle's gradients, which the fragment shader only ever caps.
struct TriangleBounds
{
  // Whether iris >= irisReach || clip >= clipReach || -margin >= lossReach throughout the triangle.
  bool dark;
  // Whether iris <= -irisReach && clip <= -clipReach && margin >= lossReach throughout the triangle.
  bool lit;
};

TriangleBounds boundTriangle(LensFlareTracedVertex corners[3], vec2 p[3], float twiceArea)
{
  const vec2 e1 = p[1] - p[0];
  const vec2 e2 = p[2] - p[0];
  const vec2 stop1 = corners[1].stop - corners[0].stop;
  const vec2 stop2 = corners[2].stop - corners[0].stop;
  const float clip1 = corners[1].clipRatio - corners[0].clipRatio;
  const float clip2 = corners[2].clipRatio - corners[0].clipRatio;
  const float margin1 = corners[1].lossMargin - corners[0].lossMargin;
  const float margin2 = corners[2].lossMargin - corners[0].lossMargin;
  const vec2 stopDx = (stop1 * e2.y - stop2 * e1.y) / twiceArea;
  const vec2 stopDy = (stop2 * e1.x - stop1 * e2.x) / twiceArea;
  const float clipDx = (clip1 * e2.y - clip2 * e1.y) / twiceArea;
  const float clipDy = (clip2 * e1.x - clip1 * e2.x) / twiceArea;
  const float marginDx = (margin1 * e2.y - margin2 * e1.y) / twiceArea;
  const float marginDy = (margin2 * e1.x - margin1 * e2.x) / twiceArea;

  float minStop = 1e30f;
  float minClip = 1e30f;
  float minMargin = 1e30f;
  float maxIris = -1e30f;
  float maxClip = -1e30f;
  float maxMargin = -1e30f;
  float longestEdge = 0.0f;
  float irisReach = 0.0f;
  float clipReach = 0.0f;
  float lossReach = 0.0f;

  [unroll]
  for (uint i = 0; i < 3; i++)
  {
    const LensFlareTracedVertex corner = corners[i];
    // The sensor's x runs against the screen's.
    const vec4 motion = vec4(-corner.sensorFromSun.x, corner.sensorFromSun.y, -corner.sensorFromSun.z, corner.sensorFromSun.w) / cb.pixelMm;
    const float stopAlong = length(corner.stopFromSun.xy) + length(stopDx * motion.x + stopDy * motion.y);
    const float stopAcross = length(corner.stopFromSun.zw) + length(stopDx * motion.z + stopDy * motion.w);
    const float clipAlong = abs(corner.clipFromSun.x) + abs(clipDx * motion.x + clipDy * motion.y);
    const float clipAcross = abs(corner.clipFromSun.y) + abs(clipDx * motion.z + clipDy * motion.w);
    const float lossAlong = abs(corner.lossFromSun.x) + abs(marginDx * motion.x + marginDy * motion.y);
    const float lossAcross = abs(corner.lossFromSun.y) + abs(marginDx * motion.z + marginDy * motion.w);

    minStop = min(minStop, length(corner.stop));
    minClip = min(minClip, corner.clipRatio);
    minMargin = min(minMargin, corner.lossMargin);
    maxIris = max(maxIris, lensApertureSignedDistance(corner.stop, cb.apertureBlades, cb.apertureCurvature, cb.apertureRotation,
                                                      cb.apertureDistanceJitter, cb.apertureAngleJitter));
    maxClip = max(maxClip, corner.clipRatio);
    maxMargin = max(maxMargin, corner.lossMargin);
    longestEdge = max(longestEdge, length(corners[(i + 1) % 3].stop - corner.stop));
    irisReach = max(irisReach, 1.5f * (stopAlong + stopAcross + length(stopDx) + length(stopDy)));
    clipReach = max(clipReach, clipAlong + clipAcross + abs(clipDx) + abs(clipDy));
    lossReach = max(lossReach, lossAlong + lossAcross + abs(marginDx) + abs(marginDy));
  }

  TriangleBounds bounds;
  bounds.dark = minStop - longestEdge - cb.apertureCornerReach >= irisReach || minClip - 1.0f >= clipReach || -maxMargin >= lossReach;
  bounds.lit = maxIris <= -irisReach && maxClip - 1.0f <= -clipReach && minMargin >= lossReach;
  return bounds;
}

[shader("vertex")]
VsOut main(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID)
{
  const uint instanceIndex = cb.firstInstance + instanceId;
  const uint quadsPerSide = cb.gridSize - 1;
  const uint triangle = vertexId / 6;
  const uint coverVertex = vertexId % 6;
  const uint quad = triangle / 2;
  const uint first = (triangle % 2) * 3;
  const int2 quadCell = int2(quad % quadsPerSide, quad / quadsPerSide);

  VsOut output;
  output.position = float4(-2.0f, -2.0f, 0.0f, 1.0f);
  output.stop = vec2(0.0f);
  output.clipRatio = 2.0f;
  output.radiance = vec3(0.0f);
  output.motion = vec4(0.0f);
  output.stopFromSun = vec4(0.0f);
  output.clipFromSun = vec4(0.0f);
  output.interior = vec4(0.0f, 0.0f, 0.0f, 1.0f);
  output.loss = vec4(1.0f, 0.0f, 0.0f, 0.0f);
  output.corners01 = vec4(0.0f);
  output.corner2 = vec2(0.0f);
  output.fullyLit = 0u;

  LensFlareTracedVertex corners[3];
  vec2 p[3];

  [unroll]
  for (uint i = 0; i < 3; i++)
  {
    corners[i] = gridVertex(instanceIndex, quadCell + int2(kCorners[first + i]));
    p[i] = toPixels(corners[i].sensor);
  }

  const float twiceArea = cross2(p[1] - p[0], p[2] - p[0]);

  if (!corners[0].valid || !corners[1].valid || !corners[2].valid || !(abs(twiceArea) > 1e-8f))
  {
    return output;
  }

  // The triangle with its edges moved out: each corner where its two edges' moved lines meet.
  vec2 normals[3];
  float offsets[3];

  [unroll]
  for (uint i = 0; i < 3; i++)
  {
    const vec2 edge = p[(i + 1) % 3] - p[i];
    vec2 normal = normalize(vec2(edge.y, -edge.x));
    normal = dot(normal, p[(i + 2) % 3] - p[i]) > 0.0f ? -normal : normal;
    normals[i] = normal;
    offsets[i] = dot(normal, p[i]) + pixelReach(normal);
  }

  vec2 grown[3];
  bool grownValid = true;

  [unroll]
  for (uint i = 0; i < 3; i++)
  {
    const vec2 a = normals[(i + 2) % 3];
    const vec2 b = normals[i];
    const float ca = offsets[(i + 2) % 3];
    const float cbOffset = offsets[i];
    const float determinant = cross2(a, b);
    grownValid = grownValid && abs(determinant) > 1e-3f;
    grown[i] = vec2(ca * b.y - cbOffset * a.y, a.x * cbOffset - b.x * ca) / determinant;
  }

  const float grownArea = 0.5f * abs(cross2(grown[1] - grown[0], grown[2] - grown[0]));

  // The alternative cover is the box along the longest edge, and the smaller of the two is drawn.
  const vec2 edges[3] = { p[1] - p[0], p[2] - p[1], p[0] - p[2] };
  uint longest = 0;

  [unroll]
  for (uint i = 1; i < 3; i++)
  {
    longest = dot(edges[i], edges[i]) > dot(edges[longest], edges[longest]) ? i : longest;
  }

  const vec2 u = normalize(edges[longest]);
  const vec2 v = vec2(-u.y, u.x);
  const vec2 anchor = p[longest];
  vec2 low = vec2(1e30f);
  vec2 high = vec2(-1e30f);

  [unroll]
  for (uint i = 0; i < 3; i++)
  {
    const vec2 local = vec2(dot(p[i] - anchor, u), dot(p[i] - anchor, v));
    low = min(low, local);
    high = max(high, local);
  }

  const vec2 reach = vec2(pixelReach(u), pixelReach(v));
  low -= reach;
  high += reach;
  const float boxArea = (high.x - low.x) * (high.y - low.y);
  const bool grownCover = grownValid && grownArea <= boxArea;

  // The grown triangle takes only the cover's first triangle, so the second's corners skip the triangle's setup and
  // leave it empty.
  if (grownCover && coverVertex >= 3)
  {
    return output;
  }

  const LensFlareTraceInstance instance = Instances[instanceIndex];
  const TriangleBounds bounds = boundTriangle(corners, p, twiceArea);
  const bool banded = instance.kind != LENS_FLARE_INSTANCE_WHOLE;

  // Each wavelength hands its light to the carrier by the carrier's hand-off, found at the wavelength's own ray from
  // the carrier's ray at the same grid point, so that every wavelength hands over at the same pixel. Past the edge
  // band the carrier draws every wavelength's light, each with its own irradiance at that grid point.
  float ownShare[3] = { 1.0f, 1.0f, 1.0f };
  vec3 interiorIrradiance[3] = { vec3(0.0f), vec3(0.0f), vec3(0.0f) };

  if (banded)
  {
    const uint carrier = instance.ghostFirst + LENS_FLARE_CARRIER_OFFSET;

    [unroll]
    for (uint i = 0; i < 3; i++)
    {
      const int2 cell = quadCell + int2(kCorners[first + i]);
      const LensFlareTracedVertex carrierCorner = gridVertex(carrier, cell);
      ownShare[i] = saturate(1.0f - carrierCorner.handOff - dot(carrierCorner.handOffGradient, corners[i].sensor - carrierCorner.sensor));

      if (instance.kind == LENS_FLARE_INSTANCE_CARRIER)
      {
        for (uint w = 0; w < LENS_FLARE_WAVELENGTHS; w++)
        {
          interiorIrradiance[i] += Instances[instance.ghostFirst + w].weight * gridVertex(instance.ghostFirst + w, cell).irradiance;
        }
      }
    }
  }

  const bool interior = banded && ownShare[0] <= 0.0f && ownShare[1] <= 0.0f && ownShare[2] <= 0.0f;

  if (bounds.dark || (interior && instance.kind == LENS_FLARE_INSTANCE_BANDED))
  {
    return output;
  }

  vec2 cover;

  if (grownCover)
  {
    cover = grown[coverVertex];
  }
  else
  {
    static const uint kBoxCorners[6] = { 0, 1, 2, 0, 2, 3 };
    const uint corner = kBoxCorners[coverVertex];
    const vec2 local = vec2(corner == 1 || corner == 2 ? high.x : low.x, corner >= 2 ? high.y : low.y);
    cover = anchor + u * local.x + v * local.y;
  }

  // Carries the attributes on the triangle's plane out to the cover's corner.
  const vec3 w = vec3(cross2(p[1] - cover, p[2] - cover), cross2(p[2] - cover, p[0] - cover), cross2(p[0] - cover, p[1] - cover)) / twiceArea;
  const vec3 source = cb.sourceRadiance * InSunVisibility.Load(int3(0, 0, 0)).rgb;
  const vec4 sensorFromSun = w.x * corners[0].sensorFromSun + w.y * corners[1].sensorFromSun + w.z * corners[2].sensorFromSun;

  output.position = float4(cover / cb.imageSize * 2.0f - 1.0f, 0.0f, 1.0f);
  output.stop = w.x * corners[0].stop + w.y * corners[1].stop + w.z * corners[2].stop;
  output.clipRatio = w.x * corners[0].clipRatio + w.y * corners[1].clipRatio + w.z * corners[2].clipRatio;
  output.radiance = instance.weight * source * (w.x * corners[0].irradiance + w.y * corners[1].irradiance + w.z * corners[2].irradiance);
  output.interior = vec4(source * (w.x * interiorIrradiance[0] + w.y * interiorIrradiance[1] + w.z * interiorIrradiance[2]),
                         w.x * ownShare[0] + w.y * ownShare[1] + w.z * ownShare[2]);
  // The sensor's x runs against the screen's.
  output.motion = vec4(-sensorFromSun.x, sensorFromSun.y, -sensorFromSun.z, sensorFromSun.w) / cb.pixelMm;
  output.stopFromSun = w.x * corners[0].stopFromSun + w.y * corners[1].stopFromSun + w.z * corners[2].stopFromSun;
  output.clipFromSun = vec4(w.x * corners[0].clipFromSun + w.y * corners[1].clipFromSun + w.z * corners[2].clipFromSun,
                            w.x * corners[0].stopPerCell + w.y * corners[1].stopPerCell + w.z * corners[2].stopPerCell,
                            w.x * corners[0].clipPerCell + w.y * corners[1].clipPerCell + w.z * corners[2].clipPerCell);
  output.loss = vec4(w.x * corners[0].lossMargin + w.y * corners[1].lossMargin + w.z * corners[2].lossMargin,
                     w.x * corners[0].lossFromSun + w.y * corners[1].lossFromSun + w.z * corners[2].lossFromSun,
                     w.x * corners[0].lossPerCell + w.y * corners[1].lossPerCell + w.z * corners[2].lossPerCell);
  output.corners01 = vec4(p[0], p[1]);
  output.corner2 = p[2];
  output.fullyLit = bounds.lit ? 1u : 0u;
  return output;
}
