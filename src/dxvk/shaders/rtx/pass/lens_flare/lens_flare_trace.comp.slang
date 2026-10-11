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

//!variant lens_flare_search.comp TRACE_PASS=0
//!variant lens_flare_trace.comp TRACE_PASS=1
//!variant lens_flare_irradiance.comp TRACE_PASS=2
//!variant lens_flare_extend.comp TRACE_PASS=3
//!variant lens_flare_extend_outer.comp TRACE_PASS=4
//!end-variants

// Traces the sun's ghosts through the lens's exact surfaces, after Hullin et al. 2011, "Physically-Based Real-Time
// Lens Flare Rendering", in the frame of the sun's meridian. The search pass fits each ghost's fine grid around the
// rays that pass, the trace pass traces the grid from the sun's centre and its limb, and the irradiance pass gives
// each ray the light of the grid's triangles around it. The extension pass places the lost rays beside traced ones,
// so that the raster finds the lost rays' boundary between the grid's points, where their loss margin crosses 0, as it
// does a clear aperture's edge. The outer extension pass places the lost rays beyond those, so that the boundary's
// light fades out, blurred by the sun's disc, before the drawn triangles end. The wavelengths of a ghost drawn across
// the spectrum share one grid, whose carrier hands each entrance point's light over between them.

#include "rtx/pass/lens_flare/lens_flare.h"
#include "rtx/utility/math.slangh"
#include "rtx/utility/lens_aperture.slangh"

layout(binding = LENS_FLARE_TRACE_SURFACES_INPUT)
StructuredBuffer<LensFlareSurface> Surfaces;

layout(binding = LENS_FLARE_TRACE_INSTANCES_INPUT)
StructuredBuffer<LensFlareTraceInstance> Instances;

layout(binding = LENS_FLARE_TRACE_DOMAINS_INPUT_OUTPUT)
RWStructuredBuffer<LensFlareTraceDomain> Domains;

layout(binding = LENS_FLARE_TRACE_VERTICES_INPUT_OUTPUT)
RWStructuredBuffer<LensFlareTracedVertex> Vertices;

layout(push_constant)
ConstantBuffer<LensFlareTraceArgs> cb;

static const float kMgF2Index = 1.38f;
static const float kZrO2Index = 2.10f;
// The rays start this far in front of the first vertex, clear of its surface's sag.
static const float kEntranceDistanceMm = 40.0f;
// The fine grid reaches this many output pixels past what passes, as its image on the sensor, for the edges to blend.
static const float kMarginPixels = 3.0f;
// A ray's path to the next surface gives its loss margin once shorter than this, in mm, which only happens where
// surfaces nearly meet.
static const float kBehindMarginMm = 0.1f;

float indexAfter(uint surface, uint wavelength)
{
  if (surface >= cb.surfaceCount)
  {
    return 1.0f;
  }

  return Surfaces[surface].indexAfter[wavelength / 4][wavelength % 4];
}

float indexBefore(uint surface, uint wavelength)
{
  return surface == 0 ? 1.0f : indexAfter(surface - 1, wavelength);
}

float bareReflectance(float n1, float n2, float cosIncident)
{
  const float sinTransmitted = n1 / n2 * sqrt(max(1.0f - cosIncident * cosIncident, 0.0f));

  if (sinTransmitted >= 1.0f)
  {
    return 1.0f;
  }

  const float cosTransmitted = sqrt(1.0f - sinTransmitted * sinTransmitted);
  const float s = (n1 * cosIncident - n2 * cosTransmitted) / (n1 * cosIncident + n2 * cosTransmitted);
  const float p = (n1 * cosTransmitted - n2 * cosIncident) / (n1 * cosTransmitted + n2 * cosIncident);

  return 0.5f * (s * s + p * p);
}

// Reflectance of a stack of thin films between an incident medium and a substrate, by Macleod's characteristic
// matrices, averaged over s and p polarisation. Without absorption or evanescent layers the product of the layers'
// matrices keeps the form [a, ib; ic, d] with real a, b, c and d. The layers are listed from the incident side.
float filmReflectance(float n0, float ns, float layerIndex[3], float layerQuarterWaves[3], uint layerCount, float wavelengthNm, float cosIncident)
{
  const float invariant = n0 * sqrt(max(1.0f - cosIncident * cosIncident, 0.0f));
  const float sinSubstrate = invariant / ns;

  if (sinSubstrate >= 1.0f)
  {
    return 1.0f;
  }

  const float cosSubstrate = sqrt(1.0f - sinSubstrate * sinSubstrate);
  float reflectance = 0.0f;

  [unroll]
  for (uint polarisation = 0; polarisation < 2; polarisation++)
  {
    float a = 1.0f;
    float b = 0.0f;
    float c = 0.0f;
    float d = 1.0f;

    for (uint i = 0; i < layerCount; i++)
    {
      const float n = layerIndex[i];
      const float sinLayer = invariant / n;

      if (sinLayer >= 1.0f)
      {
        return bareReflectance(n0, ns, cosIncident);
      }

      const float cosLayer = sqrt(1.0f - sinLayer * sinLayer);
      const float eta = polarisation == 0 ? n * cosLayer : n / cosLayer;
      // A layer of q quarter waves at the design wavelength: delta = (pi / 2) q (design / lambda) cos.
      const float delta = 0.5f * pi * layerQuarterWaves[i] * (cb.coatingWavelengthNm / wavelengthNm) * cosLayer;
      const float cosDelta = cos(delta);
      const float sinDelta = sin(delta);

      const float na = a * cosDelta - b * eta * sinDelta;
      const float nb = a * sinDelta / eta + b * cosDelta;
      const float nc = c * cosDelta + d * eta * sinDelta;
      const float nd = d * cosDelta - c * sinDelta / eta;
      a = na;
      b = nb;
      c = nc;
      d = nd;
    }

    const float etaSubstrate = polarisation == 0 ? ns * cosSubstrate : ns / cosSubstrate;
    const float etaIncident = polarisation == 0 ? n0 * cosIncident : n0 / cosIncident;
    const float numeratorRe = etaIncident * a - d * etaSubstrate;
    const float numeratorIm = etaIncident * b * etaSubstrate - c;
    const float denominatorRe = etaIncident * a + d * etaSubstrate;
    const float denominatorIm = etaIncident * b * etaSubstrate + c;

    reflectance += 0.5f * (numeratorRe * numeratorRe + numeratorIm * numeratorIm) /
                          (denominatorRe * denominatorRe + denominatorIm * denominatorIm);
  }

  return saturate(reflectance);
}

// Reflectance of a surface at a wavelength, for light arriving from its front (toward the sensor) or its back.
float surfaceReflectance(uint surface, uint wavelength, float cosIncident, bool fromFront)
{
  if (surface >= cb.surfaceCount)
  {
    return cb.sensorReflectance;
  }

  const float before = indexBefore(surface, wavelength);
  const float after = indexAfter(surface, wavelength);
  const float incidentIndex = fromFront ? before : after;
  const float substrateIndex = fromFront ? after : before;
  const float wavelengthNm = cb.wavelengthsNm[wavelength / 4][wavelength % 4];
  const float glassIndex = Surfaces[surface].coatedGlassIndex;

  if (glassIndex <= 0.0f || cb.coating == LENS_FLARE_COATING_NONE)
  {
    return bareReflectance(incidentIndex, substrateIndex, cosIncident);
  }

  // Listed from the air side, and reversed for light arriving from the glass.
  float layerIndex[3];
  float layerQuarterWaves[3];
  uint layerCount;

  if (cb.coating == LENS_FLARE_COATING_SINGLE_LAYER)
  {
    layerIndex[0] = kMgF2Index;
    layerQuarterWaves[0] = 1.0f;
    layerCount = 1;
  }
  else
  {
    layerIndex[0] = kMgF2Index;
    layerQuarterWaves[0] = 1.0f;
    layerIndex[1] = kZrO2Index;
    layerQuarterWaves[1] = 2.0f;
    layerIndex[2] = kMgF2Index * sqrt(glassIndex);
    layerQuarterWaves[2] = 1.0f;
    layerCount = 3;
  }

  if (incidentIndex > 1.0f)
  {
    for (uint i = 0; i < layerCount / 2; i++)
    {
      const float index = layerIndex[i];
      const float quarterWaves = layerQuarterWaves[i];
      layerIndex[i] = layerIndex[layerCount - 1 - i];
      layerQuarterWaves[i] = layerQuarterWaves[layerCount - 1 - i];
      layerIndex[layerCount - 1 - i] = index;
      layerQuarterWaves[layerCount - 1 - i] = quarterWaves;
    }
  }

  return filmReflectance(incidentIndex, substrateIndex, layerIndex, layerQuarterWaves, layerCount, wavelengthNm, cosIncident);
}

// Moves the ray to surface's intersection, the one on its cap near the vertex, with the normal there facing the
// object side at the vertex. The sensor, past the last surface, is flat. The loss margin takes how near the ray comes
// to missing a sphere, its discriminant over the radius squared, or to running along a plane, and how near it comes to
// meeting the surface behind it, which only paths of less than kBehindMarginMm reach.
bool intersectSurface(uint surface, inout vec3 origin, vec3 direction, out vec3 normal, inout float margin)
{
  const bool sensor = surface >= cb.surfaceCount;
  const float vertexZ = sensor ? cb.sensorZ : Surfaces[surface].z;
  const float radius = sensor ? 0.0f : Surfaces[surface].radius;
  normal = vec3(0.0f, 0.0f, -1.0f);

  if (radius == 0.0f)
  {
    margin = min(margin, abs(direction.z) - 1e-6f);

    if (abs(direction.z) < 1e-6f)
    {
      return false;
    }

    const float t = (vertexZ - origin.z) / direction.z;
    margin = min(margin, (t + 1e-3f) / kBehindMarginMm);

    if (t < -1e-3f)
    {
      return false;
    }

    origin += direction * t;
    return true;
  }

  const vec3 center = vec3(0.0f, 0.0f, vertexZ + radius);
  const vec3 toOrigin = origin - center;
  const float b = dot(toOrigin, direction);
  const float c = dot(toOrigin, toOrigin) - radius * radius;
  const float discriminant = b * b - c;
  margin = min(margin, discriminant / (radius * radius));

  if (discriminant < 0.0f)
  {
    return false;
  }

  const float root = sqrt(discriminant);
  const float t0 = -b - root;
  const float t1 = -b + root;
  // Takes the cap near the vertex, rather than the far side of the sphere.
  const float z0 = origin.z + t0 * direction.z;
  const float z1 = origin.z + t1 * direction.z;
  const float t = abs(z0 - vertexZ) < abs(z1 - vertexZ) ? t0 : t1;
  margin = min(margin, (t + 1e-3f) / kBehindMarginMm);

  if (t < -1e-3f)
  {
    return false;
  }

  origin += direction * t;
  normal = (origin - center) / radius;
  return true;
}

// The loss margin takes the squared cosine of the refracted ray's angle, which falls to 0 at the critical angle.
bool refractRay(inout vec3 direction, vec3 normal, float n1, float n2, inout float margin)
{
  const vec3 facing = dot(direction, normal) > 0.0f ? -normal : normal;
  const float eta = n1 / n2;
  const float cosIncident = -dot(facing, direction);
  const float k = 1.0f - eta * eta * (1.0f - cosIncident * cosIncident);
  margin = min(margin, k);

  if (k < 0.0f)
  {
    return false;
  }

  direction = normalize(eta * direction + (eta * cosIncident - sqrt(k)) * facing);
  return true;
}

struct TraceState
{
  vec3 origin;
  vec3 direction;
  float clipRatio;
  vec2 stop;
  bool valid;
  float margin;
};

void meetSurface(uint surface, inout TraceState state)
{
  if (surface == cb.stopIndex)
  {
    state.stop = state.origin.xy / cb.irisCornerRadius;
  }
  else if (surface < cb.surfaceCount)
  {
    state.clipRatio = max(state.clipRatio, length(state.origin.xy) / Surfaces[surface].clearRadius);
  }
}

// Crosses surface toward the sensor, or toward the object when backward is set.
void crossSurface(uint surface, uint wavelength, bool backward, inout TraceState state)
{
  vec3 normal;

  if (!state.valid)
  {
    return;
  }

  if (!intersectSurface(surface, state.origin, state.direction, normal, state.margin))
  {
    state.valid = false;
    return;
  }

  meetSurface(surface, state);

  const float before = indexBefore(surface, wavelength);
  const float after = indexAfter(surface, wavelength);

  // Totally reflected, or bent back the way it came, the ray is lost. The loss margin takes how far short of turning
  // back it heads.
  if (!refractRay(state.direction, normal, backward ? after : before, backward ? before : after, state.margin))
  {
    state.valid = false;
    return;
  }

  state.margin = min(state.margin, (backward ? -state.direction.z : state.direction.z) - 1e-4f);
  state.valid = state.margin >= 0.0f;
}

// Reflects from surface, returning its reflectance at the ray's incidence, or 1 without withReflectance.
float reflectFromSurface(uint surface, uint wavelength, bool fromFront, bool withReflectance, inout TraceState state)
{
  vec3 normal;

  if (!state.valid)
  {
    return 0.0f;
  }

  if (!intersectSurface(surface, state.origin, state.direction, normal, state.margin))
  {
    state.valid = false;
    return 0.0f;
  }

  meetSurface(surface, state);

  const float cosIncident = min(abs(dot(state.direction, normal)), 1.0f);
  state.direction = reflect(state.direction, normal);

  // Reflected from the front it heads back toward the object, and from behind toward the sensor.
  state.margin = min(state.margin, (fromFront ? -state.direction.z : state.direction.z) - 1e-4f);

  if (state.margin < 0.0f)
  {
    state.valid = false;
    return 0.0f;
  }

  if (!withReflectance)
  {
    return 1.0f;
  }

  return surfaceReflectance(surface, wavelength, cosIncident, fromFront);
}

// One ray of a ghost in a direction from a point of the entrance plane, in mm. Rays that only find where the light
// goes leave out withReflectance, which skips the coatings.
LensFlareTracedVertex traceRay(LensFlareTraceInstance instance, vec2 entrance, vec3 direction, bool withReflectance)
{
  const uint wavelength = instance.wavelength;

  TraceState state;
  state.direction = direction;
  state.origin = vec3(entrance, 0.0f) - state.direction * (kEntranceDistanceMm / state.direction.z);
  state.clipRatio = 0.0f;
  state.stop = vec2(0.0f);
  state.valid = true;
  state.margin = 1.0f;

  // The ray heads toward the sensor to the second reflection, back to the first, and toward the sensor again.
  for (uint k = 0; k < instance.second; k++)
  {
    crossSurface(k, wavelength, false, state);
  }

  float reflectance = reflectFromSurface(instance.second, wavelength, true, withReflectance, state);

  for (uint k = instance.second - 1; k > instance.first; k--)
  {
    crossSurface(k, wavelength, true, state);
  }

  reflectance *= reflectFromSurface(instance.first, wavelength, false, withReflectance, state);

  for (uint k = instance.first + 1; k < cb.surfaceCount; k++)
  {
    crossSurface(k, wavelength, false, state);
  }

  vec3 normal;
  state.valid = state.valid && intersectSurface(cb.surfaceCount, state.origin, state.direction, normal, state.margin);
  const bool reached = all(isfinite(state.origin)) && all(abs(state.origin.xy) < vec2(1000.0f));
  const float margin = (state.valid && !reached) || !isfinite(state.margin) ? LENS_FLARE_NO_MARGIN : state.margin;

  LensFlareTracedVertex result;
  result.sensor = state.origin.xy;
  result.stop = state.stop;
  result.clipRatio = state.clipRatio;
  result.reflectance = reflectance;
  result.valid = state.valid && reached ? LENS_FLARE_VERTEX_TRACED : LENS_FLARE_VERTEX_LOST;
  result.irradiance = 0.0f;
  result.sensorFromSun = vec4(0.0f);
  result.stopFromSun = vec4(0.0f);
  result.clipFromSun = vec2(0.0f);
  result.stopPerCell = 0.0f;
  result.clipPerCell = 0.0f;
  result.handOff = -1.0f;
  result.handOffGradient = vec2(0.0f);
  result.lossMargin = margin;
  result.lossFromSun = vec2(0.0f);
  result.lossPerCell = 0.0f;
  result.pad0 = 0;
  return result;
}

// A loss margin for differences across the disc and the grid, where a loss with no margin of its own counts as one of
// -1.
float boundedMargin(float margin)
{
  return max(margin, -1.0f);
}

// The rays' direction for a sun at an angle to the axis, in the meridian's frame.
vec3 meridionalDirection(float angle)
{
  return vec3(sin(angle), 0.0f, cos(angle));
}

// How far inside the circle through the iris's farthest corner, cornerReach, and every clear aperture a ray passes, and
// how far from being lost, negative outside. For a ray the lens lost, the margin it was lost by, which carries on
// smoothly across the boundary.
float passMargin(LensFlareTracedVertex traced, float cornerReach)
{
  return traced.valid ? min(min(1.0f - traced.clipRatio, traced.lossMargin), cornerReach - length(traced.stop))
                      : min(boundedMargin(traced.lossMargin), 0.0f);
}

// Rotates v from the meridian's frame into the lens's.
vec2 fromMeridian(vec2 v)
{
  return vec2(cb.meridian.x * v.x - cb.meridian.y * v.y, cb.meridian.y * v.x + cb.meridian.x * v.y);
}

// Bits that order as the floats they come from, for the atomic minimum and maximum.
uint orderedBits(float value)
{
  const uint bits = asuint(value);
  return (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
}

float orderedFloat(uint bits)
{
  return asfloat((bits & 0x80000000u) != 0 ? bits & 0x7fffffffu : ~bits);
}

LensFlareTracedVertex gridVertex(uint base, int2 cell)
{
  return Vertices[base + uint(cell.y) * cb.gridSize + uint(cell.x)];
}

#if TRACE_PASS == 0
// For the sun's centre and the ends of its diameter in the meridian: each ray's margin, then the passing span's ends
// along the meridian and its reach across it, as ordered bits.
groupshared float sharedMargins[3 * LENS_FLARE_SEARCH_SAMPLES];
groupshared float sharedSensor[LENS_FLARE_SEARCH_SAMPLES];
groupshared uint sharedLow[3];
groupshared uint sharedHigh[3];
groupshared uint sharedReach[3];
// The farthest any passing ray lies from the axis.
groupshared uint sharedRho;
// The largest distance on the sensor per distance on the entrance plane between neighbouring passing rays, along the
// meridian and across it.
groupshared uint sharedMagnification[2];

[shader("compute")]
[numthreads(LENS_FLARE_SEARCH_SAMPLES, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
  const uint instanceIndex = groupId.y;
  const uint index = groupThreadId.x;

  if (instanceIndex >= cb.instanceCount)
  {
    return;
  }

  const LensFlareTraceInstance instance = Instances[instanceIndex];
  const float sunRadius = cb.sunRadius * cb.edgeSoftness;
  const float halfSize = instance.searchHalfSize;
  const float step = 2.0f * halfSize / float(LENS_FLARE_SEARCH_SAMPLES - 1);
  const float cornerReach = lensApertureCornerReach(cb.apertureBlades, cb.apertureCurvature, cb.apertureRotation, cb.apertureDistanceJitter,
                                                    cb.apertureAngleJitter);

  if (index < 3)
  {
    sharedLow[index] = 0xffffffffu;
    sharedHigh[index] = 0u;
    sharedReach[index] = orderedBits(0.0f);
  }

  if (index < 2)
  {
    sharedMagnification[index] = orderedBits(0.0f);
  }

  if (index == 0)
  {
    sharedRho = orderedBits(0.0f);
  }

  // First the rays along the meridian, from the sun's centre and the ends of its diameter in the meridian.
  const float x = instance.searchCenter - halfSize + step * float(index);

  [loop]
  for (uint k = 0; k < 3; k++)
  {
    const LensFlareTracedVertex traced = traceRay(instance, vec2(x, 0.0f), meridionalDirection(cb.sunAngle + (float(k) - 1.0f) * sunRadius), false);
    sharedMargins[k * LENS_FLARE_SEARCH_SAMPLES + index] = passMargin(traced, cornerReach);

    if (k == 1)
    {
      sharedSensor[index] = traced.sensor.x;
    }
  }

  GroupMemoryBarrierWithGroupSync();

  // The passing rays, and where the margin crosses 0 between neighbours, so that the span's ends move smoothly.
  for (uint k = 0; k < 3; k++)
  {
    const float margin = sharedMargins[k * LENS_FLARE_SEARCH_SAMPLES + index];

    if (margin > 0.0f)
    {
      InterlockedMin(sharedLow[k], orderedBits(x));
      InterlockedMax(sharedHigh[k], orderedBits(x));
      InterlockedMax(sharedRho, orderedBits(abs(x)));
    }

    if (index + 1 < LENS_FLARE_SEARCH_SAMPLES)
    {
      const float next = sharedMargins[k * LENS_FLARE_SEARCH_SAMPLES + index + 1];

      if ((margin > 0.0f) != (next > 0.0f))
      {
        const float crossing = x + step * margin / (margin - next);
        InterlockedMin(sharedLow[k], orderedBits(crossing));
        InterlockedMax(sharedHigh[k], orderedBits(crossing));
        InterlockedMax(sharedRho, orderedBits(abs(crossing)));
      }

      if (k == 1 && margin > 0.0f && next > 0.0f)
      {
        InterlockedMax(sharedMagnification[0], orderedBits(abs(sharedSensor[index + 1] - sharedSensor[index]) / step));
      }
    }
  }

  GroupMemoryBarrierWithGroupSync();

  // Across the meridian, on lines spread over each span. By the symmetry, one side suffices.
  const uint line = index / LENS_FLARE_SEARCH_LINE_SAMPLES;
  const uint sample = index % LENS_FLARE_SEARCH_LINE_SAMPLES;
  const float stepAcross = halfSize / float(LENS_FLARE_SEARCH_LINE_SAMPLES - 1);
  const float y = stepAcross * float(sample);
  float lineX[3] = { 0.0f, 0.0f, 0.0f };

  [loop]
  for (uint k = 0; k < 3; k++)
  {
    float margin = -1.0f;

    if (sharedLow[k] <= sharedHigh[k])
    {
      const float low = orderedFloat(sharedLow[k]);
      const float high = orderedFloat(sharedHigh[k]);
      lineX[k] = low + (float(line) + 0.5f) / float(LENS_FLARE_SEARCH_LINES) * (high - low);
      const LensFlareTracedVertex traced = traceRay(instance, vec2(lineX[k], y), meridionalDirection(cb.sunAngle + (float(k) - 1.0f) * sunRadius), false);
      margin = passMargin(traced, cornerReach);

      if (k == 1)
      {
        sharedSensor[index] = traced.sensor.y;
      }
    }

    sharedMargins[k * LENS_FLARE_SEARCH_SAMPLES + index] = margin;
  }

  GroupMemoryBarrierWithGroupSync();

  for (uint k = 0; k < 3; k++)
  {
    const float margin = sharedMargins[k * LENS_FLARE_SEARCH_SAMPLES + index];

    if (margin > 0.0f)
    {
      InterlockedMax(sharedReach[k], orderedBits(y));
      InterlockedMax(sharedRho, orderedBits(length(vec2(lineX[k], y))));
    }

    if (sample + 1 < LENS_FLARE_SEARCH_LINE_SAMPLES)
    {
      const float next = sharedMargins[k * LENS_FLARE_SEARCH_SAMPLES + index + 1];

      if ((margin > 0.0f) != (next > 0.0f))
      {
        const float crossing = y + stepAcross * margin / (margin - next);
        InterlockedMax(sharedReach[k], orderedBits(crossing));
        InterlockedMax(sharedRho, orderedBits(length(vec2(lineX[k], crossing))));
      }

      if (k == 1 && margin > 0.0f && next > 0.0f)
      {
        InterlockedMax(sharedMagnification[1], orderedBits(abs(sharedSensor[index + 1] - sharedSensor[index]) / stepAcross));
      }
    }
  }

  GroupMemoryBarrierWithGroupSync();

  if (index != 0)
  {
    return;
  }

  float low = 1e30f;
  float high = -1e30f;
  float reach = 0.0f;
  bool passes = false;

  for (uint k = 0; k < 3; k++)
  {
    if (sharedLow[k] <= sharedHigh[k])
    {
      low = min(low, orderedFloat(sharedLow[k]));
      high = max(high, orderedFloat(sharedHigh[k]));
      reach = max(reach, orderedFloat(sharedReach[k]));
      passes = true;
    }
  }

  LensFlareTraceDomain domain;
  domain.origin = vec2(0.0f);
  domain.spacing = vec2(0.0f);

  if (passes)
  {
    // Across the sun's disc its meridian turns by up to its radius over sin(angle), turning what passes about the
    // axis. However far it turns, it stays within its farthest reach from the axis, which binds once the disc nears
    // the axis.
    const float turn = min(sunRadius / max(sin(cb.sunAngle), 1e-6f), 1.0f);
    const float radius = max(abs(low), abs(high));
    const float rho = orderedFloat(sharedRho);
    const float turnedReach = min(reach + radius * turn, rho);
    low = max(low - reach * turn, -rho);
    high = min(high + reach * turn, rho);
    reach = turnedReach;

    // The grid reaches a few pixels further on the sensor, for the edges to blend, though not past the span's own size.
    const float width = max(high - low, step);
    const float height = max(2.0f * reach, stepAcross);
    const float magnificationAlong = orderedFloat(sharedMagnification[0]);
    const float magnificationAcross = orderedFloat(sharedMagnification[1]);
    const float marginAlong = min(magnificationAlong > 0.0f ? kMarginPixels * cb.pixelMm / magnificationAlong : 2.0f * step, 0.5f * width);
    const float marginAcross = min(magnificationAcross > 0.0f ? kMarginPixels * cb.pixelMm / magnificationAcross : 2.0f * stepAcross, 0.5f * height);

    domain.origin = vec2(low - marginAlong, -(reach + marginAcross));
    domain.spacing = vec2(high - low + 2.0f * marginAlong, 2.0f * (reach + marginAcross)) / float(cb.gridSize - 1);
  }

  Domains[instanceIndex] = domain;
}
#endif

#if TRACE_PASS != 0
// The triangles of the grid around a ray, as the corners' offsets from it: both of its own cell's, the first of the
// cell to its left, both of the cell below and to the left, and the second of the cell below. The raster splits each
// cell into (0, 0), (1, 0), (1, 1) and (0, 0), (1, 1), (0, 1).
static const int2 kStarTriangles[18] = {
  int2(0, 0), int2(1, 0), int2(1, 1),
  int2(0, 0), int2(1, 1), int2(0, 1),
  int2(-1, 0), int2(0, 0), int2(0, 1),
  int2(-1, -1), int2(0, -1), int2(0, 0),
  int2(-1, -1), int2(0, 0), int2(-1, 0),
  int2(0, -1), int2(1, 0), int2(0, 0),
};

// The domain an instance's grid spans: its own, or for a split ghost the union of its wavelengths', so that each
// wavelength's ray at a grid point leaves the same point of the entrance plane.
LensFlareTraceDomain sharedDomain(LensFlareTraceInstance instance, uint instanceIndex)
{
  if (instance.kind == LENS_FLARE_INSTANCE_WHOLE)
  {
    return Domains[instanceIndex];
  }

  const float last = float(cb.gridSize - 1);
  vec2 low = vec2(1e30f);
  vec2 high = vec2(-1e30f);

  for (uint w = 0; w < LENS_FLARE_WAVELENGTHS; w++)
  {
    const LensFlareTraceDomain domain = Domains[instance.ghostFirst + w];

    if (domain.spacing.x > 0.0f)
    {
      low = min(low, domain.origin);
      high = max(high, domain.origin + domain.spacing * last);
    }
  }

  LensFlareTraceDomain shared;
  shared.origin = high.x >= low.x ? low : vec2(0.0f);
  shared.spacing = high.x >= low.x ? (high - low) / last : vec2(0.0f);
  return shared;
}

#if TRACE_PASS == 2
// The change of the sensor point, the stop coordinate, the clip ratio and the loss margin per grid step along an axis,
// from the neighbours the lens did not lose.
// Returns false when it lost both.
bool gridStep(uint base, int2 cell, int2 axis, LensFlareTracedVertex traced, out vec2 sensorStep, out vec2 stopStep, out float clipStep,
              out float marginStep)
{
  LensFlareTracedVertex ends[2] = { traced, traced };
  float steps = 0.0f;

  [unroll]
  for (uint side = 0; side < 2; side++)
  {
    const int2 neighbour = side == 0 ? cell - axis : cell + axis;

    if (all(neighbour >= int2(0)) && all(neighbour < int2(cb.gridSize)))
    {
      const LensFlareTracedVertex other = gridVertex(base, neighbour);

      if (other.valid)
      {
        ends[side] = other;
        steps += 1.0f;
      }
    }
  }

  sensorStep = (ends[1].sensor - ends[0].sensor) / max(steps, 1.0f);
  stopStep = (ends[1].stop - ends[0].stop) / max(steps, 1.0f);
  clipStep = (ends[1].clipRatio - ends[0].clipRatio) / max(steps, 1.0f);
  marginStep = (ends[1].lossMargin - ends[0].lossMargin) / max(steps, 1.0f);
  return steps > 0.0f;
}

// How far a carrier's ray lies through the hand-off, from how far it lands inside the iris's and the clear apertures'
// edges and the lost rays' boundary in pixels, and how that grows per mm on the sensor, through the grid's Jacobian at
// the ray. The band and the hand-off span at least a few of the grid's cells there.
// Leaves the ray in the band where the grid gives no Jacobian.
void handOff(LensFlareTraceInstance instance, uint base, int2 cell, LensFlareTracedVertex traced, out float position, out vec2 gradient)
{
  position = -1.0f;
  gradient = vec2(0.0f);
  vec2 sensorX;
  vec2 sensorY;
  vec2 stopX;
  vec2 stopY;
  float clipX;
  float clipY;
  float marginX;
  float marginY;

  if (!gridStep(base, cell, int2(1, 0), traced, sensorX, stopX, clipX, marginX) ||
      !gridStep(base, cell, int2(0, 1), traced, sensorY, stopY, clipY, marginY))
  {
    return;
  }

  const float determinant = sensorX.x * sensorY.y - sensorY.x * sensorX.y;

  if (!(abs(determinant) > 1e-12f))
  {
    return;
  }

  // The iris's distance grows along the stop coordinate's radius. Each gradient per grid step becomes one per mm on the
  // sensor through the inverse of the sensor point's.
  const vec2 radial = traced.stop / max(length(traced.stop), 1e-6f);
  const vec2 iris = vec2(dot(radial, stopX), dot(radial, stopY));
  const vec2 clip = vec2(clipX, clipY);
  const vec2 irisGradient = vec2(iris.x * sensorY.y - iris.y * sensorX.y, iris.y * sensorX.x - iris.x * sensorY.x) / determinant;
  const vec2 clipGradient = vec2(clip.x * sensorY.y - clip.y * sensorX.y, clip.y * sensorX.x - clip.x * sensorY.x) / determinant;
  const float irisRate = max(length(irisGradient), 1e-12f);
  const float clipRate = max(length(clipGradient), 1e-12f);
  const float irisDistance = -lensApertureSignedDistance(traced.stop, cb.apertureBlades, cb.apertureCurvature, cb.apertureRotation,
                                                         cb.apertureDistanceJitter, cb.apertureAngleJitter) / (irisRate * cb.pixelMm);
  const float clipDistance = (1.0f - traced.clipRatio) / (clipRate * cb.pixelMm);
  const vec2 margin = vec2(marginX, marginY);
  const vec2 marginGradient = vec2(margin.x * sensorY.y - margin.y * sensorX.y, margin.y * sensorX.x - margin.x * sensorY.x) / determinant;
  const float marginRate = max(length(marginGradient), 1e-12f);
  const float lossDistance = traced.lossMargin / (marginRate * cb.pixelMm);

  // Each distance shrinks toward its edge at one pixel per pixel.
  float distance = min(irisDistance, clipDistance);
  vec2 distanceGradient = (irisDistance < clipDistance ? -irisGradient / irisRate : -clipGradient / clipRate) / cb.pixelMm;

  if (lossDistance < distance)
  {
    distance = lossDistance;
    distanceGradient = marginGradient / (marginRate * cb.pixelMm);
  }

  const float cellPixels = sqrt(abs(determinant)) / cb.pixelMm;
  const float inner = max(instance.bandInner, LENS_FLARE_BAND_CELLS * cellPixels);
  const float width = max(instance.bandWidth, LENS_FLARE_HAND_OFF_CELLS * cellPixels);
  position = (distance - inner) / width;
  gradient = distanceGradient / width;
}
#endif

#if TRACE_PASS >= 3
static const int2 kExtendDirections[8] = {
  int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1), int2(1, 1), int2(-1, -1), int2(1, -1), int2(-1, 1),
};
// A lost ray's margin below this comes from a loss with no margin of its own.
static const float kMinExtendedMargin = 0.5f * LENS_FLARE_NO_MARGIN;
// How far an unreliable place pushes the loss margin down, which brings the boundary back to the traced rays.
static const float kUnreliableMargin = 8.0f;
// The places' disagreement, and one place's reach, in the edges' blurs, up to which a place is relied on.
static const float kExtendedSpreadBlurs = 3.0f;
static const float kExtendedReachBlurs = 6.0f;

// The rays a lost ray's place is extrapolated from: the traced ones, and for the outer layer also those the first
// layer placed. Those this pass places itself are left out, as other threads may not have written them yet.
bool extendsFrom(uint valid)
{
#if TRACE_PASS == 3
  return valid == LENS_FLARE_VERTEX_TRACED;
#else
  return valid == LENS_FLARE_VERTEX_TRACED || valid == LENS_FLARE_VERTEX_EXTENDED;
#endif
}
#endif

[shader("compute")]
[numthreads(LENS_FLARE_TRACE_GROUP_SIZE, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
  const uint instanceIndex = threadId.y;
  const uint index = threadId.x;
  const uint gridSize = cb.gridSize;

  if (instanceIndex >= cb.instanceCount || index >= gridSize * gridSize)
  {
    return;
  }

  const LensFlareTraceInstance instance = Instances[instanceIndex];
  const LensFlareTraceDomain domain = sharedDomain(instance, instanceIndex);
  const uint base = instanceIndex * gridSize * gridSize;

#if TRACE_PASS == 1
  LensFlareTracedVertex result;
  result.sensor = vec2(0.0f);
  result.stop = vec2(0.0f);
  result.clipRatio = 2.0f;
  result.reflectance = 0.0f;
  result.valid = LENS_FLARE_VERTEX_LOST;
  result.irradiance = 0.0f;
  result.sensorFromSun = vec4(0.0f);
  result.stopFromSun = vec4(0.0f);
  result.clipFromSun = vec2(0.0f);
  result.stopPerCell = 0.0f;
  result.clipPerCell = 0.0f;
  result.handOff = -1.0f;
  result.handOffGradient = vec2(0.0f);
  result.lossMargin = LENS_FLARE_NO_MARGIN;
  result.lossFromSun = vec2(0.0f);
  result.lossPerCell = 0.0f;
  result.pad0 = 0;

  if (domain.spacing.x > 0.0f)
  {
    // Traces from the sun's centre, then from its limb along the meridian and across it. Only the centre's light is
    // drawn.
    const vec2 entrance = domain.origin + vec2(float(index % gridSize), float(index / gridSize)) * domain.spacing;
    const vec3 direction = meridionalDirection(cb.sunAngle);
    const vec3 directions[3] = {
      direction,
      meridionalDirection(cb.sunAngle + cb.sunRadius),
      vec3(direction.x * cos(cb.sunRadius), sin(cb.sunRadius), direction.z * cos(cb.sunRadius)),
    };
    LensFlareTracedVertex rays[3];

    [loop]
    for (uint k = 0; k < 3; k++)
    {
      rays[k] = traceRay(instance, entrance, directions[k], k == 0);
    }

    result = rays[0];
    const LensFlareTracedVertex along = rays[1];
    const LensFlareTracedVertex across = rays[2];

    if (result.valid && along.valid)
    {
      result.sensorFromSun.xy = fromMeridian(along.sensor - result.sensor) * cb.edgeSoftness;
      result.stopFromSun.xy = fromMeridian(along.stop - result.stop) * cb.edgeSoftness;
      result.clipFromSun.x = (along.clipRatio - result.clipRatio) * cb.edgeSoftness;
    }

    if (result.valid && across.valid)
    {
      result.sensorFromSun.zw = fromMeridian(across.sensor - result.sensor) * cb.edgeSoftness;
      result.stopFromSun.zw = fromMeridian(across.stop - result.stop) * cb.edgeSoftness;
      result.clipFromSun.y = (across.clipRatio - result.clipRatio) * cb.edgeSoftness;
    }

    // The loss margin carries on smoothly through the lost rays' boundary, so its change across the disc is known for
    // every ray.
    const float margin = boundedMargin(result.lossMargin);
    result.lossFromSun = vec2(boundedMargin(along.lossMargin) - margin, boundedMargin(across.lossMargin) - margin) * cb.edgeSoftness;

    result.sensor = fromMeridian(result.sensor);
    result.stop = fromMeridian(result.stop);
  }

  Vertices[base + index] = result;
#elif TRACE_PASS == 2
  const LensFlareTracedVertex traced = Vertices[base + index];

  if (!traced.valid)
  {
    return;
  }

  // The ray's irradiance is the light of the triangles around it over their area on the sensor, their areas taken
  // whole so that triangles that fold over do not cancel. Interpolated across each triangle, it then carries exactly
  // the light of the triangles.
  const int2 cell = int2(index % gridSize, index / gridSize);
  const float halfCell = 0.5f * domain.spacing.x * domain.spacing.y;
  float energy = 0.0f;
  float area = 0.0f;

  for (uint t = 0; t < 6; t++)
  {
    LensFlareTracedVertex corners[3];
    bool valid = true;

    [unroll]
    for (uint i = 0; i < 3; i++)
    {
      const int2 corner = cell + kStarTriangles[3 * t + i];
      corners[i] = gridVertex(base, clamp(corner, int2(0), int2(gridSize - 1)));
      valid = valid && all(corner >= int2(0)) && all(corner < int2(gridSize)) && bool(corners[i].valid);
    }

    if (!valid)
    {
      continue;
    }

    const vec2 a = corners[1].sensor - corners[0].sensor;
    const vec2 b = corners[2].sensor - corners[0].sensor;
    energy += (corners[0].reflectance + corners[1].reflectance + corners[2].reflectance) / 3.0f * halfCell;
    area += 0.5f * abs(a.x * b.y - a.y * b.x);
  }

  // Finds the most the stop coordinate, the clip ratio and the loss margin change to the ray's neighbours. The margin
  // changes as smoothly toward a lost neighbour.
  static const int2 kNeighbours[4] = { int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1) };
  float stopPerCell = 0.0f;
  float clipPerCell = 0.0f;
  float lossPerCell = 0.0f;

  [unroll]
  for (uint n = 0; n < 4; n++)
  {
    const int2 neighbour = cell + kNeighbours[n];

    if (any(neighbour < int2(0)) || any(neighbour >= int2(gridSize)))
    {
      continue;
    }

    const LensFlareTracedVertex other = gridVertex(base, neighbour);
    lossPerCell = max(lossPerCell, abs(boundedMargin(other.lossMargin) - boundedMargin(traced.lossMargin)));

    if (other.valid)
    {
      stopPerCell = max(stopPerCell, length(other.stop - traced.stop));
      clipPerCell = max(clipPerCell, abs(other.clipRatio - traced.clipRatio));
    }
  }

  Vertices[base + index].irradiance = area > 0.0f
    ? instance.transmission * energy / max(area, 1e-6f * cb.pixelMm * cb.pixelMm)
    : 0.0f;
  Vertices[base + index].stopPerCell = stopPerCell;
  Vertices[base + index].clipPerCell = clipPerCell;
  Vertices[base + index].lossPerCell = lossPerCell;

  if (instance.kind == LENS_FLARE_INSTANCE_CARRIER)
  {
    float position;
    vec2 gradient;
    handOff(instance, base, cell, traced, position, gradient);
    Vertices[base + index].handOff = position;
    Vertices[base + index].handOffGradient = gradient;
  }
#else
  // A lost ray beside traced ones takes the place their trend gives it, from each direction along which two traced rays
  // lie in a row, so that the triangles across the boundary are drawn. Each direction gives a place on the line through
  // its two traced rays and, where a third lies beyond them, on the parabola through all three. The set that agrees
  // better is taken: the parabolas follow a map that curves toward the boundary, the lines one that folds. The ray
  // carries on its neighbours' clip ratio and takes their light and change across the disc, and keeps its own loss
  // margin, whose edge then falls between it and them. A loss without a margin of its own has no boundary to place.
  const LensFlareTracedVertex lost = Vertices[base + index];

  if (lost.valid != LENS_FLARE_VERTEX_LOST || lost.lossMargin <= kMinExtendedMargin)
  {
    return;
  }

  const int2 cell = int2(index % gridSize, index / gridSize);
  vec2 lineSensor[8];
  vec2 lineStop[8];
  vec2 curveSensor[8];
  vec2 curveStop[8];
  vec4 sensorFromSun = vec4(0.0f);
  vec4 stopFromSun = vec4(0.0f);
  vec2 clipFromSun = vec2(0.0f);
  float clipRatio = 0.0f;
  float irradiance = 0.0f;
  float stopPerCell = 0.0f;
  float clipPerCell = 0.0f;
  float lossPerCell = 0.0f;
  float marginTrend = 0.0f;
  float lowestMargin = 1e30f;
  float highestMargin = -1e30f;
  float step = 0.0f;
  uint count = 0;

  [unroll]
  for (uint d = 0; d < 8; d++)
  {
    const int2 far = cell + 2 * kExtendDirections[d];

    if (any(far < int2(0)) || any(far >= int2(gridSize)))
    {
      continue;
    }

    const LensFlareTracedVertex near = gridVertex(base, cell + kExtendDirections[d]);
    const LensFlareTracedVertex beyond = gridVertex(base, far);

    if (!extendsFrom(near.valid) || !extendsFrom(beyond.valid))
    {
      continue;
    }

    lineSensor[count] = 2.0f * near.sensor - beyond.sensor;
    lineStop[count] = 2.0f * near.stop - beyond.stop;
    curveSensor[count] = lineSensor[count];
    curveStop[count] = lineStop[count];
    const int2 farthest = cell + 3 * kExtendDirections[d];

    if (all(farthest >= int2(0)) && all(farthest < int2(gridSize)))
    {
      const LensFlareTracedVertex third = gridVertex(base, farthest);

      if (extendsFrom(third.valid))
      {
        curveSensor[count] = 3.0f * (near.sensor - beyond.sensor) + third.sensor;
        curveStop[count] = 3.0f * (near.stop - beyond.stop) + third.stop;
      }
    }

    clipRatio += 2.0f * near.clipRatio - beyond.clipRatio;
    clipFromSun += near.clipFromSun;
    sensorFromSun += near.sensorFromSun;
    stopFromSun += near.stopFromSun;
    irradiance += near.irradiance;
    stopPerCell = max(stopPerCell, near.stopPerCell);
    clipPerCell = max(clipPerCell, near.clipPerCell);
    lossPerCell = max(lossPerCell, near.lossPerCell);
    marginTrend += 2.0f * near.lossMargin - beyond.lossMargin;
    lowestMargin = min(lowestMargin, near.lossMargin);
    highestMargin = max(highestMargin, near.lossMargin);
    step = max(step, length(near.sensor - beyond.sensor));
    count++;
  }

  if (count == 0)
  {
    return;
  }

  const float inverseCount = 1.0f / float(count);

  // A lost ray's margin only counts the interactions before its loss. Where an earlier loss cut its path short, it can
  // lie above the trend the traced rays' margin falls by, and the lower of the two is taken.
  const float ownMargin = min(lost.lossMargin, marginTrend * inverseCount);
  const float boundedOwn = boundedMargin(ownMargin);
  lossPerCell = max(lossPerCell, max(abs(boundedOwn - lowestMargin), abs(boundedOwn - highestMargin)));
  vec2 lineMean = vec2(0.0f);
  vec2 curveMean = vec2(0.0f);

  for (uint e = 0; e < count; e++)
  {
    lineMean += lineSensor[e] * inverseCount;
    curveMean += curveSensor[e] * inverseCount;
  }

  float lineSpread = 0.0f;
  float curveSpread = 0.0f;

  for (uint e = 0; e < count; e++)
  {
    lineSpread = max(lineSpread, length(lineSensor[e] - lineMean));
    curveSpread = max(curveSpread, length(curveSensor[e] - curveMean));
  }

  const bool curve = curveSpread < lineSpread;
  vec2 stop = vec2(0.0f);

  for (uint e = 0; e < count; e++)
  {
    stop += (curve ? curveStop[e] : lineStop[e]) * inverseCount;
  }

  // Where the estimates disagree by more than a few of the edges' blurs, or one alone reaches further than several,
  // the grid does not resolve the boundary there, and its edge eases back toward the traced rays, so that it does not
  // pop.
  const float blur = max(instance.edgeBlur, 2.0f * cb.pixelMm);
  const float spread = curve ? curveSpread : lineSpread;
  const float reliability = saturate(2.0f - (count >= 2 ? spread / (kExtendedSpreadBlurs * blur) : step / (kExtendedReachBlurs * blur)));

  if (reliability <= 0.0f)
  {
    return;
  }

  LensFlareTracedVertex result = lost;
  result.sensor = curve ? curveMean : lineMean;
  result.stop = stop;
  result.clipRatio = clipRatio * inverseCount;
  result.sensorFromSun = sensorFromSun * inverseCount;
  result.stopFromSun = stopFromSun * inverseCount;
  result.clipFromSun = clipFromSun * inverseCount;
  result.irradiance = irradiance * inverseCount;
  result.stopPerCell = stopPerCell;
  result.clipPerCell = clipPerCell;
  result.lossMargin = ownMargin - (1.0f - reliability) * kUnreliableMargin;
  result.lossPerCell = max(lossPerCell, (1.0f - reliability) * kUnreliableMargin);
  // It stays in the edge band, drawing its own light.
  result.handOff = -1.0f;
  result.handOffGradient = vec2(0.0f);
  result.valid = TRACE_PASS == 3 ? LENS_FLARE_VERTEX_EXTENDED : LENS_FLARE_VERTEX_EXTENDED_OUTER;
  Vertices[base + index] = result;
#endif
}
#endif
