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

// Shades a traced ghost's triangles. Each pixel takes the triangle's light over the share of the pixel it covers,
// with the attributes at the middle of that share, lit by the share of the sun's disc and of the pixel through which
// the ray passes the iris and every clear aperture, and stays on the traced side of the lost rays' boundary.

#include "rtx/pass/lens_flare/lens_flare.h"
#include "rtx/utility/math.slangh"
#include "rtx/utility/lens_aperture.slangh"

layout(push_constant)
ConstantBuffer<LensFlareRasterArgs> cb;

float cross2(vec2 a, vec2 b)
{
  return a.x * b.y - a.y * b.x;
}

// The share of the unit pixel at the origin the triangle covers, and the middle of that share.
float trianglePixelShare(vec2 p0, vec2 p1, vec2 p2, out vec2 centroid)
{
  centroid = vec2(0.5f);
  const float orientation = cross2(p1 - p0, p2 - p0) > 0.0f ? 1.0f : -1.0f;
  const vec2 corners[3] = { p0, p1, p2 };
  bool inside = true;

  [unroll]
  for (uint i = 0; i < 3; i++)
  {
    // The edge's side at the pixel's corners, positive inside.
    const vec2 a = corners[i];
    const vec2 e = (corners[(i + 1) % 3] - a) * orientation;
    const float f00 = cross2(e, -a);
    const float f10 = f00 - e.y;
    const float f01 = f00 + e.x;
    const float f11 = f10 + e.x;

    if (max(max(f00, f10), max(f01, f11)) <= 0.0f)
    {
      return 0.0f;
    }

    inside = inside && min(min(f00, f10), min(f01, f11)) >= 0.0f;
  }

  if (inside)
  {
    return 1.0f;
  }

  // Clips the triangle to the pixel's four sides, normal . p <= offset.
  static const vec2 kSideNormals[4] = { vec2(1.0f, 0.0f), vec2(-1.0f, 0.0f), vec2(0.0f, 1.0f), vec2(0.0f, -1.0f) };
  static const float kSideOffsets[4] = { 1.0f, 0.0f, 1.0f, 0.0f };
  vec2 polygon[8];
  vec2 clipped[8];
  polygon[0] = p0;
  polygon[1] = p1;
  polygon[2] = p2;
  uint count = 3;

  for (uint side = 0; side < 4; side++)
  {
    const vec2 normal = kSideNormals[side];
    const float offset = kSideOffsets[side];
    uint clippedCount = 0;

    for (uint k = 0; k < count; k++)
    {
      const vec2 p = polygon[k];
      const vec2 q = polygon[k + 1 == count ? 0 : k + 1];
      const float dp = dot(normal, p) - offset;
      const float dq = dot(normal, q) - offset;

      if (dp <= 0.0f)
      {
        clipped[clippedCount++] = p;
      }

      if ((dp <= 0.0f) != (dq <= 0.0f))
      {
        clipped[clippedCount++] = p + (q - p) * (dp / (dp - dq));
      }
    }

    count = clippedCount;

    for (uint k = 0; k < count; k++)
    {
      polygon[k] = clipped[k];
    }

    if (count == 0)
    {
      return 0.0f;
    }
  }

  float area = 0.0f;
  vec2 moment = vec2(0.0f);

  for (uint k = 0; k < count; k++)
  {
    const vec2 p = polygon[k];
    const vec2 q = polygon[k + 1 == count ? 0 : k + 1];
    const float c = cross2(p, q);
    area += c;
    moment += (p + q) * c;
  }

  if (!(abs(area) > 1e-12f))
  {
    return 0.0f;
  }

  centroid = moment / (3.0f * area);
  return 0.5f * abs(area);
}

// The share of the unit disc where u.n <= d.
float capFraction(float d)
{
  if (d >= 1.0f)
  {
    return 1.0f;
  }

  if (d <= -1.0f)
  {
    return 0.0f;
  }

  return 1.0f - (acos(d) - d * sqrt(1.0f - d * d)) / pi;
}

// Signed area of the unit disc's intersection with the triangle (0, a, b).
float discTriangleArea(vec2 a, vec2 b)
{
  const float a2 = dot(a, a);
  const float b2 = dot(b, b);

  if (a2 <= 1.0f && b2 <= 1.0f)
  {
    return 0.5f * cross2(a, b);
  }

  // Finds where the segment from a to b meets the circle.
  const vec2 d = b - a;
  const float qa = dot(d, d);
  const float qb = dot(a, d);
  const float discriminant = qb * qb - qa * (a2 - 1.0f);
  const float root = sqrt(max(discriminant, 0.0f));

  if (a2 <= 1.0f)
  {
    const vec2 p = a + d * ((-qb + root) / qa);
    return 0.5f * cross2(a, p) + 0.5f * atan2(cross2(p, b), dot(p, b));
  }

  if (b2 <= 1.0f)
  {
    const vec2 p = a + d * ((-qb - root) / qa);
    return 0.5f * atan2(cross2(a, p), dot(a, p)) + 0.5f * cross2(p, b);
  }

  const float t1 = (-qb - root) / qa;
  const float t2 = (-qb + root) / qa;

  if (qa > 0.0f && discriminant > 0.0f && t1 > 0.0f && t2 < 1.0f)
  {
    const vec2 p1 = a + d * t1;
    const vec2 p2 = a + d * t2;
    return 0.5f * atan2(cross2(a, p1), dot(a, p1)) + 0.5f * cross2(p1, p2) + 0.5f * atan2(cross2(p2, b), dot(p2, b));
  }

  return 0.5f * atan2(cross2(a, b), dot(a, b));
}

// The share of the unit disc where u.x <= d1 and u.n2 <= d2: the disc's intersection with a square around it cut by
// both lines.
float twoChordFraction(float d1, float d2, vec2 n2)
{
  vec2 polygon[8];
  vec2 clipped[8];
  polygon[0] = vec2(-2.0f, -2.0f);
  polygon[1] = vec2(2.0f, -2.0f);
  polygon[2] = vec2(2.0f, 2.0f);
  polygon[3] = vec2(-2.0f, 2.0f);
  uint count = 4;

  for (uint h = 0; h < 2; h++)
  {
    const vec2 normal = h == 0 ? vec2(1.0f, 0.0f) : n2;
    const float offset = h == 0 ? d1 : d2;
    uint clippedCount = 0;

    for (uint k = 0; k < count; k++)
    {
      const vec2 p = polygon[k];
      const vec2 q = polygon[k + 1 == count ? 0 : k + 1];
      const float dp = dot(normal, p) - offset;
      const float dq = dot(normal, q) - offset;

      if (dp <= 0.0f)
      {
        clipped[clippedCount++] = p;
      }

      if ((dp <= 0.0f) != (dq <= 0.0f))
      {
        clipped[clippedCount++] = p + (q - p) * (dp / (dp - dq));
      }
    }

    count = clippedCount;

    for (uint k = 0; k < count; k++)
    {
      polygon[k] = clipped[k];
    }
  }

  float area = 0.0f;

  for (uint k = 0; k < count; k++)
  {
    area += discTriangleArea(polygon[k], polygon[k + 1 == count ? 0 : k + 1]);
  }

  return saturate(area / pi);
}

// The share of the sun's disc and the pixel through which a ray passes two edges, a + g.u + p.x <= 0 for u over the
// unit disc and x over the pixel. The pixel's box joins the disc as a disc of the same second moments, which keeps
// how the two edges move together.
float discCoverage(float a1, vec2 g1, vec2 p1, float a2, vec2 g2, vec2 p2)
{
  const float g11 = dot(g1, g1) + dot(p1, p1) / 3.0f;
  const float g22 = dot(g2, g2) + dot(p2, p2) / 3.0f;
  const float g12 = dot(g1, g2) + dot(p1, p2) / 3.0f;
  const float s1 = sqrt(max(g11, 1e-14f));
  const float s2 = sqrt(max(g22, 1e-14f));
  const float d1 = -a1 / s1;
  const float d2 = -a2 / s2;

  if (d1 <= -1.0f || d2 <= -1.0f)
  {
    return 0.0f;
  }

  if (d1 >= 1.0f)
  {
    return capFraction(d2);
  }

  if (d2 >= 1.0f)
  {
    return capFraction(d1);
  }

  const float c = clamp(g12 / (s1 * s2), -1.0f, 1.0f);
  return twoChordFraction(d1, d2, vec2(c, sqrt(max(1.0f - c * c, 0.0f))));
}

float irisDistance(vec2 stop)
{
  return lensApertureSignedDistance(stop, cb.apertureBlades, cb.apertureCurvature, cb.apertureRotation, cb.apertureDistanceJitter,
                                    cb.apertureAngleJitter);
}

[shader("fragment")]
float4 main(float4 screenPosition : SV_Position, float2 stop : STOP, float clipRatio : CLIP, float3 radiance : RADIANCE,
            float4 motion : MOTION, float4 stopFromSun : STOP_SUN, float4 clipFromSun : CLIP_SUN, float4 interior : INTERIOR,
            float4 loss : LOSS, nointerpolation float4 corners01 : CORNERS01, nointerpolation float2 corner2 : CORNER2,
            nointerpolation uint fullyLit : FULLY_LIT) : SV_Target
{
  // The attributes lie on the triangle's plane, so their screen derivatives are its gradients.
  const vec3 radianceDx = ddx(radiance);
  const vec3 radianceDy = ddy(radiance);
  const vec4 interiorDx = ddx(interior);
  const vec4 interiorDy = ddy(interior);

  const vec2 pixel = floor(screenPosition.xy);
  vec2 centroid;
  const float share = trianglePixelShare(corners01.xy - pixel, corners01.zw - pixel, corner2 - pixel, centroid);

  // Every attribute is taken at the middle of the share, inside the triangle, where a sliver's plane does not
  // extrapolate it.
  const vec2 offset = pixel + centroid - screenPosition.xy;
  const vec3 radianceAt = max(radiance + radianceDx * offset.x + radianceDy * offset.y, vec3(0.0f));
  const vec4 interiorAt = interior + interiorDx * offset.x + interiorDy * offset.y;
  const float ownShare = saturate(interiorAt.w);
  const vec3 light = radianceAt * ownShare + max(interiorAt.rgb, vec3(0.0f)) * (1.0f - ownShare);

  // The flag is the same over the whole triangle, which keeps the derivatives below in uniform control flow.
  if (fullyLit)
  {
    return float4(light * share, 0.0f);
  }

  const vec2 stopDx0 = ddx(stop);
  const vec2 stopDy0 = ddy(stop);
  const float clipDx0 = ddx(clipRatio);
  const float clipDy0 = ddy(clipRatio);
  const vec4 motionDx = ddx(motion);
  const vec4 motionDy = ddy(motion);
  const vec4 stopFromSunDx = ddx(stopFromSun);
  const vec4 stopFromSunDy = ddy(stopFromSun);
  const vec4 clipFromSunDx = ddx(clipFromSun);
  const vec4 clipFromSunDy = ddy(clipFromSun);
  const vec4 lossDx = ddx(loss);
  const vec4 lossDy = ddy(loss);

  if (share <= 0.0f)
  {
    return float4(0.0f);
  }

  const vec2 stopAt = stop + stopDx0 * offset.x + stopDy0 * offset.y;
  const float clipAt = clipRatio + clipDx0 * offset.x + clipDy0 * offset.y;
  const float iris = irisDistance(stopAt);
  const float clip = clipAt - 1.0f;
  const vec4 motionAt = motion + motionDx * offset.x + motionDy * offset.y;
  const vec4 stopFromSunAt = stopFromSun + stopFromSunDx * offset.x + stopFromSunDy * offset.y;
  const vec4 clipFromSunAt = clipFromSun + clipFromSunDx * offset.x + clipFromSunDy * offset.y;
  const vec4 lossAt = loss + lossDx * offset.x + lossDy * offset.y;
  const float margin = lossAt.x;

  // A pixel gathers rays from at most a cell or so of the grid: where the grid folds over, the screen gradients grow
  // without bound though the rays stay close.
  vec2 stopDx = stopDx0;
  vec2 stopDy = stopDy0;
  float clipDx = clipDx0;
  float clipDy = clipDy0;
  float marginDx = lossDx.x;
  float marginDy = lossDy.x;
  const float stopPerPixel = max(length(stopDx), length(stopDy));
  const float clipPerPixel = max(abs(clipDx), abs(clipDy));
  const float marginPerPixel = max(abs(marginDx), abs(marginDy));

  if (stopPerPixel > clipFromSunAt.z)
  {
    const float k = clipFromSunAt.z / stopPerPixel;
    stopDx *= k;
    stopDy *= k;
  }

  if (clipPerPixel > clipFromSunAt.w)
  {
    const float k = clipFromSunAt.w / clipPerPixel;
    clipDx *= k;
    clipDy *= k;
  }

  if (marginPerPixel > lossAt.w)
  {
    const float k = lossAt.w / marginPerPixel;
    marginDx *= k;
    marginDy *= k;
  }

  // Across the sun's disc at this pixel, each changes by its change at a fixed entrance point less what the image's
  // own motion brings.
  const vec2 stopAlong = stopFromSunAt.xy - (stopDx * motionAt.x + stopDy * motionAt.y);
  const vec2 stopAcross = stopFromSunAt.zw - (stopDx * motionAt.z + stopDy * motionAt.w);
  const float clipAlong = clipFromSunAt.x - (clipDx * motionAt.x + clipDy * motionAt.y);
  const float clipAcross = clipFromSunAt.y - (clipDx * motionAt.z + clipDy * motionAt.w);
  const float lossAlong = lossAt.y - (marginDx * motionAt.x + marginDy * motionAt.y);
  const float lossAcross = lossAt.z - (marginDx * motionAt.z + marginDy * motionAt.w);

  // The most each can change across the disc and the pixel, the iris's distance changing no faster than the stop
  // coordinate, with room for its arcs.
  const float irisReach = 1.5f * (length(stopAlong) + length(stopAcross) + length(stopDx) + length(stopDy));
  const float clipReach = abs(clipAlong) + abs(clipAcross) + abs(clipDx) + abs(clipDy);
  const float lossReach = abs(lossAlong) + abs(lossAcross) + abs(marginDx) + abs(marginDy);

  if (iris >= irisReach || clip >= clipReach || -margin >= lossReach)
  {
    return float4(0.0f);
  }

  const bool apertureEdge = iris > -irisReach || clip > -clipReach;
  const bool lossEdge = margin < lossReach;
  float coverage = 1.0f;

  if (apertureEdge || lossEdge)
  {
    const vec2 irisSun = vec2(irisDistance(stopAt + stopAlong) - iris, irisDistance(stopAt + stopAcross) - iris);
    const vec2 irisPixel = vec2(irisDistance(stopAt + stopDx) - iris, irisDistance(stopAt + stopDy) - iris);

    if (apertureEdge)
    {
      coverage = discCoverage(iris, irisSun, irisPixel, clip, vec2(clipAlong, clipAcross), vec2(clipDx, clipDy));
    }

    // The lost rays' boundary is an edge of its own. Each field is close to linear across a cell, but where the two
    // edges meet the larger of them is not, and the triangles would step along its kink.
    if (lossEdge)
    {
      coverage = min(coverage, discCoverage(iris, irisSun, irisPixel, -margin, -vec2(lossAlong, lossAcross), -vec2(marginDx, marginDy)));
    }
  }

  return float4(light * (coverage * share), 0.0f);
}
