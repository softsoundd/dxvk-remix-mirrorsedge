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

// Ground truth for the cloud layer: a progressive path tracer of the same field under the same sun, sky and
// ground, against which the march's lighting model is judged (debug views Clouds Reference and Reference Error).
//
// Free flights use delta tracking (Woodcock) against the field's maximum extinction, jumping the body distance
// field's provably empty radius (and the air above and below the slab), so the kilometres of air between
// clouds cost a handful of steps. The sun is sampled at every collision by ratio tracking toward it, traced with
// a probability that follows the sun grid's estimate of its transmittance (the estimate sets only the
// probability, so the result stays unbiased). Paths leaving through the top see the sky's spherical harmonics
// there, the same sky the march lights the clouds with. Paths leaving through the base reach a Lambertian
// ground of the atmosphere's albedo, which takes the sun through the clouds and sends the path on upward.
//
// The first referenceExactBounces collisions scatter by the droplets' exact Mie phase (disc convolved toward the
// sun) under the full extinction, new directions drawn by one-sample MIS over the three channels' phases (a
// channel picked at random, its inverse CDF sampled, every channel weighted by its phase against their mean).
// Later collisions continue in the delta-M similar medium, extinction (1 - f) sigma and phase HG(g'), which
// keeps the high orders' diffusion exact to second order in the phase's moments for a third of the collisions;
// a path through a thick cumulus still takes thousands.
//
// Paths persist across frames. One thread owns a 4x4 pixel tile and advances its path by at most
// referenceBouncesPerFrame collisions a frame, so no dispatch runs long however deep the cloud, and a path runs
// until it leaves the layer or referenceMaxBounces. A finished path adds to its pixel's sums and the tile's next
// pixel (in Bayer order) starts one.

#include "rtx/pass/raytrace_args.h"
#include "rtx/pass/clouds/cloud_march.slangh"
#include "rtx/utility/geometry_flags.slangh"
#include "rtx/utility/debug_view_indices.h"

layout(binding = CLOUD_BINDING_CAMERA)
ConstantBuffer<RaytraceArgs> cb;

layout(r32f, binding = CLOUD_BINDING_PRIMARY_LINEAR_VIEW_Z)
Texture2D<float> PrimaryLinearViewZ;

layout(r32f, binding = CLOUD_BINDING_PSR_FIRST_HIT_DISTANCE)
Texture2D<float> PsrFirstHitDistance;

layout(r16ui, binding = CLOUD_BINDING_SHARED_FLAGS)
Texture2D<uint16_t> SharedFlags;

// The march's layer this frame, which pixels without reference samples fall back to.
layout(binding = CLOUD_BINDING_HISTORY)
Texture2D<float4> CloudMarchLayer;

// Scattering angle at CDF values of each channel's phase (r, g, b).
layout(binding = CLOUD_BINDING_PHASE_CDF)
Texture2D<float4> CloudPhaseInverseCdf;

// Per pixel: the finished paths' summed radiance and transmittance.
layout(binding = CLOUD_BINDING_REFERENCE_ACCUMULATION, rgba32f)
RWTexture2D<float4> ReferenceAccumulation;

// Per tile, the path in flight: its position (w the index of the collision there, negative for no path), its
// direction (w the paths the tile has finished), its throughput (w the largest channel it started with, for
// the roulette) and the radiance it has gathered.
layout(binding = CLOUD_BINDING_REFERENCE_PATH_POSITION, rgba32f)
RWTexture2D<float4> ReferencePathPosition;

layout(binding = CLOUD_BINDING_REFERENCE_PATH_DIRECTION, rgba32f)
RWTexture2D<float4> ReferencePathDirection;

layout(binding = CLOUD_BINDING_REFERENCE_PATH_THROUGHPUT, rgba32f)
RWTexture2D<float4> ReferencePathThroughput;

layout(binding = CLOUD_BINDING_REFERENCE_PATH_RADIANCE, rgba32f)
RWTexture2D<float4> ReferencePathRadiance;

// This frame's error statistics (CLOUD_REFERENCE_STATISTICS_SIZE uints), cleared before the pass.
layout(binding = CLOUD_BINDING_REFERENCE_STATISTICS)
RWStructuredBuffer<uint> ReferenceStatistics;

layout(binding = CLOUD_BINDING_LAYER_OUTPUT, rgba16f)
RWTexture2D<float4> ReferenceLayerOutput;

layout(binding = CLOUD_BINDING_DEBUG_VIEW)
RWTexture2D<float4> CloudDebugView;

// phase: frames since the reset, for the random streams. level: 0 starts the sums over.
layout(push_constant)
ConstantBuffer<CloudPassArgs> passArgs;

static const uint kRussianRouletteStart = 8u;

// The virtual ground's altitude, kept clear of the planet's surface like the atmosphere's march origins.
static const float kGroundAltitudeKm = 0.01f;

// The order a tile visits its pixels in, and each pixel's place in that order (a 4x4 Bayer matrix), so the
// first paths after a reset spread over the tile.
static const uint kTileOrder[16] = { 0u, 10u, 2u, 8u, 5u, 15u, 7u, 13u, 1u, 11u, 3u, 9u, 4u, 14u, 6u, 12u };
static const uint kTileRank[16] = { 0u, 8u, 2u, 10u, 12u, 4u, 14u, 6u, 3u, 11u, 1u, 9u, 15u, 7u, 13u, 5u };

struct ReferenceRng {
  uint state;
};

uint referenceHash(uint value) {
  const uint state = value * 747796405u + 2891336453u;
  const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

float nextRandom(inout ReferenceRng rng) {
  rng.state = rng.state * 747796405u + 2891336453u;
  uint word = ((rng.state >> ((rng.state >> 28u) + 4u)) ^ rng.state) * 277803737u;
  word = (word >> 22u) ^ word;
  return float(word >> 8u) * (1.0f / 16777216.0f);
}

struct ReferencePath {
  vec3 positionKm;          // Cloud frame (horizontal km, altitude)
  float collision;          // Index of the collision at positionKm; negative with no path in flight
  vec3 direction;
  vec3 throughput;
  float rouletteReference;  // The throughput's largest channel at the start
  vec3 radiance;            // Gathered so far, the air in front of the first collision included
};

// Extinction at a cloud frame position, scaled by mediumScale (1, or 1 - f in the similar medium). emptyKm is the
// radius around it the field guarantees empty.
float pointExtinction(vec3 positionKm, float mediumScale, CloudArgs cloud, out float emptyKm) {
  // Outside the slab the nearest point of it is straight up or down, the shells being concentric.
  const float topKm = cloud.baseAltitudeKm + cloud.thicknessKm;
  if (positionKm.y <= cloud.baseAltitudeKm || positionKm.y >= topKm) {
    emptyKm = positionKm.y <= cloud.baseAltitudeKm ? cloud.baseAltitudeKm - positionKm.y : positionKm.y - topKm;
    return 0.0f;
  }
  emptyKm = 0.0f;
  const float heightFraction = cloudHeightFraction(positionKm.y, cloud);
  const vec2 controls = cloudControls(positionKm.xz, cloud);
  const float cameraDistanceKm = length(positionKm - cloud.cameraPositionKm);
  const CloudDensitySample s = cloudSampleDensity(
    positionKm, heightFraction, controls.x, controls.y, cameraDistanceKm, 0.0f, false, cloud, CloudNvdf, CloudDetailNoise, CloudVolumeSampler);
  emptyKm = s.emptyKm;
  return mediumScale * cloud.extinctionKm * s.density * cloudVerticalProfile(heightFraction, cloud);
}

float referenceMajorant(float mediumScale, CloudArgs cloud) {
  return max(mediumScale * cloud.extinctionKm * max(cloud.adiabaticMax, 1.0f), 1e-3f);
}

// Path vertices advance along a direction with the altitude following the planet's curvature like the march's.
vec3 advance(vec3 positionKm, vec3 direction, float t, float planetRadiusKm) {
  return vec3(positionKm.x + direction.x * t, cloudAltitudeAlong(positionKm.y, direction.y, t, planetRadiusKm), positionKm.z + direction.z * t);
}

// Delta tracking from a point along a direction, up to tMax: the collision distance, or -1 for none. Each
// tentative collision's sample also says whether the next step may jump empty space instead.
float freeFlight(vec3 positionKm, vec3 direction, float tMax, float mediumScale, CloudArgs cloud, float planetRadiusKm, inout ReferenceRng rng) {
  const float majorant = referenceMajorant(mediumScale, cloud);
  float emptyKm;
  float extinction = pointExtinction(positionKm, mediumScale, cloud, emptyKm);
  float t = 0.0f;
  for (uint i = 0u; i < 4096u; ++i) {
    const bool jump = extinction <= 0.0f && emptyKm > 1e-4f;
    t += jump ? emptyKm * 0.95f : -log(max(nextRandom(rng), 1e-7f)) / majorant;
    if (t >= tMax) {
      return -1.0f;
    }
    extinction = pointExtinction(advance(positionKm, direction, t, planetRadiusKm), mediumScale, cloud, emptyKm);
    if (!jump && nextRandom(rng) * majorant < extinction) {
      return t;
    }
  }
  return -1.0f;
}

// Ratio tracking toward a direction to the slab's exit: an unbiased transmittance estimate.
float ratioTracking(vec3 positionKm, vec3 direction, float mediumScale, CloudArgs cloud, float planetRadiusKm, inout ReferenceRng rng) {
  const float tExit = cloudSlabExitDistance(positionKm.y, direction.y, cloud, planetRadiusKm);
  if (tExit <= 0.0f) {
    return 0.0f;
  }
  const float majorant = referenceMajorant(mediumScale, cloud);
  float transmittance = 1.0f;
  float t = 0.0f;
  bool jump = false;
  float emptyKm = 0.0f;
  for (uint i = 0u; i < 4096u; ++i) {
    t += jump ? emptyKm * 0.95f : -log(max(nextRandom(rng), 1e-7f)) / majorant;
    if (t >= tExit) {
      break;
    }
    const float extinction = pointExtinction(advance(positionKm, direction, t, planetRadiusKm), mediumScale, cloud, emptyKm);
    if (!jump) {
      transmittance *= 1.0f - min(extinction / majorant, 1.0f);
      // Russian roulette once the estimate is negligible, keeping it unbiased.
      if (transmittance < 1e-3f) {
        if (nextRandom(rng) > 0.1f) {
          return 0.0f;
        }
        transmittance *= 10.0f;
      }
    }
    jump = extinction <= 0.0f && emptyKm > 1e-4f;
  }
  return transmittance;
}

// The sun grid's estimate of the transmittance toward the sun in the similar medium (an upper bound on the
// exact medium's), or 1 outside the grid's window.
float sunTransmittanceEstimate(vec3 positionKm, CloudArgs cloud) {
  if (!cloudGridContains(positionKm.xz, cloud)) {
    return 1.0f;
  }
  const float opticalDepth = CloudSunGrid.SampleLevel(CloudVolumeSampler, cloudGridUvw(positionKm, cloud), 0).x;
  return exp(-cloud.extinctionKm * (1.0f - cloud.forwardPeakFraction) * opticalDepth);
}

vec3 rotateAbout(vec3 axis, float cosTheta, float phi) {
  const vec3 helper = abs(axis.y) < 0.999f ? vec3(0.0f, 1.0f, 0.0f) : vec3(1.0f, 0.0f, 0.0f);
  const vec3 tangent = normalize(cross(helper, axis));
  const vec3 bitangent = cross(axis, tangent);
  const float sinTheta = sqrt(max(1.0f - cosTheta * cosTheta, 0.0f));
  return normalize(axis * cosTheta + tangent * (sinTheta * cos(phi)) + bitangent * (sinTheta * sin(phi)));
}

// Scattering cosine drawn from one channel's droplet phase (the row nearest this layer's effective radius).
float samplePhaseCosine(uint channel, CloudArgs cloud, inout ReferenceRng rng) {
  const float row = clamp(round(cloud.phaseLutRow), 0.0f, float(CLOUD_PHASE_LUT_RADII - 1));
  uint width, height;
  CloudPhaseInverseCdf.GetDimensions(width, height);
  const float x = nextRandom(rng) * float(width) - 0.5f;
  const int i0 = clamp(int(floor(x)), 0, int(width) - 1);
  const int i1 = min(i0 + 1, int(width) - 1);
  const float frac = saturate(x - float(i0));
  const float theta = lerp(CloudPhaseInverseCdf.Load(int3(i0, int(row), 0))[channel], CloudPhaseInverseCdf.Load(int3(i1, int(row), 0))[channel], frac);
  return cos(theta);
}

float sampleHgCosine(float g, float u) {
  if (abs(g) < 1e-3f) {
    return 1.0f - 2.0f * u;
  }
  const float s = (1.0f - g * g) / (1.0f - g + 2.0f * g * u);
  return clamp((1.0f + g * g - s * s) / (2.0f * g), -1.0f, 1.0f);
}

// What a path leaving the layer without meeting the ground sees: the sky's harmonics (both hemispheres) at the
// top for paths going up, at the base for paths going down.
vec3 environmentRadiance(vec3 direction) {
  float basis[CLOUD_SKY_SH_COEFFICIENTS];
  cloudShBasis(direction, basis);
  const int altitude = direction.y > 0.0f ? CLOUD_SKY_SH_ALTITUDES - 1 : 0;
  vec3 radiance = vec3(0.0f);
  [unroll]
  for (int m = 0; m < CLOUD_SKY_SH_COEFFICIENTS; ++m) {
    radiance += (cloudLoadSkySh(m, altitude, 0) + cloudLoadSkySh(m, altitude, 1)) * basis[m];
  }
  return max(radiance, vec3(0.0f));
}

// The camera ray of a new path at a pixel, to its first collision. False when it gets through the layer.
bool startPath(uint2 pixel, CloudArgs cloud, AtmosphereArgs atmosphere, inout ReferencePath path, inout ReferenceRng rng) {
  const vec2 jitter = vec2(nextRandom(rng), nextRandom(rng)) - 0.5f;
  const vec2 screenUv = (vec2(pixel) + 0.5f + jitter) / vec2(cb.camera.resolution);
  const CameraDirections directions = cameraScreenUVToDirection(cb.camera, screenUv, false);
  const vec3 dirYUp = normalize(cloudWorldToYUp(directions.worldDirection, atmosphere.isZUp != 0));

  // How far the scene lets the ray go (the first surface a mirror or glass reflects from, for those), on past glass
  // the sky is seen through, as in the screen pass.
  const GeometryFlags geometryFlags = geometryFlagsReadFromGBuffer(ivec2(pixel), SharedFlags);
  const float linearViewZ = PrimaryLinearViewZ[pixel];
  const bool performedPsr = geometryFlags.performedAnyPSR() && !geometryFlags.secondarySurfaceMask;
  const bool primaryMiss = linearViewZ == cb.primaryDirectMissLinearViewZ && !geometryFlags.performedAnyPSR();
  const bool transmittedSky = (cloud.flags & CLOUD_FLAG_PSR_REFLECTIONS) != 0 && !cb.domeLightArgs.active
    && geometryFlags.performPSTR && linearViewZ == cb.primaryDirectMissLinearViewZ;
  const float kmPerWorldUnit = 1.0f / max(cloud.worldUnitsPerKm, 1e-6f);
  float sceneDistanceKm = 1e9f;
  if (performedPsr && !transmittedSky) {
    const float firstHit = PsrFirstHitDistance[pixel];
    if (firstHit > 0.0f) {
      sceneDistanceKm = firstHit * kmPerWorldUnit;
    }
  } else if (!primaryMiss && !transmittedSky) {
    sceneDistanceKm = length(cameraReconstructViewPosition(cb.camera, directions.viewDirection, linearViewZ)) * kmPerWorldUnit;
  }

  // The first collision, over both slab crossings.
  const float planetRadiusKm = atmosphere.planetRadius;
  const float mediumScale = cloud.referenceExactBounces > 0u ? 1.0f : 1.0f - cloud.forwardPeakFraction;
  float collisionT = -1.0f;
  for (int piece = 0; piece < 2 && collisionT < 0.0f; ++piece) {
    float tEntry, tExit;
    if (!cloudSlabSpan(dirYUp, cloud, planetRadiusKm, piece, tEntry, tExit)) {
      continue;
    }
    tExit = min(tExit, min(sceneDistanceKm, cloud.maxMarchKm));
    if (tExit <= tEntry) {
      continue;
    }
    const vec3 entryKm = vec3(cloud.cameraPositionKm.x + dirYUp.x * tEntry, cloudRayAltitudeKm(dirYUp, tEntry, cloud, planetRadiusKm),
      cloud.cameraPositionKm.z + dirYUp.z * tEntry);
    const float t = freeFlight(entryKm, dirYUp, tExit - tEntry, mediumScale, cloud, planetRadiusKm, rng);
    if (t >= 0.0f) {
      collisionT = tEntry + t;
    }
  }
  if (collisionT < 0.0f) {
    return false;
  }

  // The air in front of the collision, as the march composites its samples: its transmittance carried in the
  // throughput, its in-scattering the path's first radiance.
  vec2 apUv = cloudDirectionToSkyUv(dirYUp);
  apUv.y = cloudClampToTexelCentres(apUv.y, float(CLOUD_SKY_AP_LUT_HEIGHT));
  const CloudAirSegment air = cloudSampleSkyAp(apUv, collisionT, cloud);

  path.positionKm = vec3(cloud.cameraPositionKm.x + dirYUp.x * collisionT, cloudRayAltitudeKm(dirYUp, collisionT, cloud, planetRadiusKm),
    cloud.cameraPositionKm.z + dirYUp.z * collisionT);
  path.collision = 0.0f;
  path.direction = dirYUp;
  path.throughput = cloud.albedoTint * air.transmittance;
  path.rouletteReference = max(path.throughput.r, max(path.throughput.g, path.throughput.b));
  path.radiance = air.inScatter;
  return true;
}

// One event of a path, a collision in the cloud or a bounce off the ground: the sun toward it, then a new
// direction and the flight to the next event. False once the path has ended (left for the sky, its environment
// added, or lost the roulette).
bool scatterPath(CloudArgs cloud, AtmosphereArgs atmosphere, inout ReferencePath path, inout ReferenceRng rng) {
  const float planetRadiusKm = atmosphere.planetRadius;
  const float similarity = 1.0f - cloud.forwardPeakFraction;
  const uint collision = uint(path.collision);
  const bool exact = collision < cloud.referenceExactBounces;
  const vec3 sunDir = normalize(atmosphere.sunDirection);
  const vec3 sunIrradiance = cloudSunIrradiance(vec3(0.0f, path.positionKm.y, 0.0f), atmosphere);

  if (path.positionKm.y < 0.5f * cloud.baseAltitudeKm) {
    // The ground: Lambertian, lit by the sun through the clouds, the path going on upward cosine weighted.
    const float albedo = saturate(atmosphere.groundAlbedo);
    if (sunDir.y > 0.0f && any(sunIrradiance > vec3(0.0f))) {
      const float transmittance = ratioTracking(path.positionKm, sunDir, exact ? 1.0f : similarity, cloud, planetRadiusKm, rng);
      path.radiance += path.throughput * sunIrradiance * (albedo * (1.0f / kPi) * sunDir.y * transmittance);
    }
    path.throughput *= albedo;
    const float u = nextRandom(rng);
    const float phi = 2.0f * kPi * nextRandom(rng);
    const float sinTheta = sqrt(u);
    path.direction = vec3(sinTheta * cos(phi), sqrt(max(1.0f - u, 0.0f)), sinTheta * sin(phi));
  } else {
    if (any(sunIrradiance > vec3(0.0f))) {
      const float traceProbability = clamp(sunTransmittanceEstimate(path.positionKm, cloud) * 64.0f, 1.0f / 32.0f, 1.0f);
      if (nextRandom(rng) < traceProbability) {
        const float transmittance = ratioTracking(path.positionKm, sunDir, exact ? 1.0f : similarity, cloud, planetRadiusKm, rng);
        if (transmittance > 0.0f) {
          const float cosSun = dot(path.direction, sunDir);
          const vec3 phase = exact ? cloudMiePhase(cosSun, cloud) : vec3(cloudHgPhase(cosSun, cloud.truncatedAsymmetry));
          path.radiance += path.throughput * sunIrradiance * phase * (transmittance / traceProbability);
        }
      }
    }

    float cosTheta;
    if (exact) {
      const uint channel = min(uint(nextRandom(rng) * 3.0f), 2u);
      cosTheta = samplePhaseCosine(channel, cloud, rng);
      const vec3 phase = cloudMiePhase(cosTheta, cloud);
      path.throughput *= cloud.albedoTint * phase / max((phase.r + phase.g + phase.b) * (1.0f / 3.0f), 1e-6f);
    } else {
      cosTheta = sampleHgCosine(cloud.truncatedAsymmetry, nextRandom(rng));
      path.throughput *= cloud.albedoTint;
    }
    path.direction = rotateAbout(path.direction, cosTheta, 2.0f * kPi * nextRandom(rng));
  }

  // The flight after an event is in the medium of the event it leads to.
  const uint next = collision + 1u;
  const float nextScale = next < cloud.referenceExactBounces ? 1.0f : similarity;
  const float tExit = cloudSlabExitDistance(path.positionKm.y, path.direction.y, cloud, planetRadiusKm);
  const float t = tExit > 0.0f ? freeFlight(path.positionKm, path.direction, tExit, nextScale, cloud, planetRadiusKm, rng) : -1.0f;
  if (t < 0.0f) {
    // Out through the base, the ground is next; the next event bounces off it.
    const float tGround = path.direction.y < 0.0f ? cloudShellDistance(path.positionKm.y, path.direction.y, kGroundAltitudeKm, planetRadiusKm) : -1.0f;
    if (tGround > 0.0f) {
      path.positionKm = advance(path.positionKm, path.direction, tGround, planetRadiusKm);
      path.positionKm.y = kGroundAltitudeKm;
    } else {
      path.radiance += path.throughput * environmentRadiance(path.direction);
      return false;
    }
  } else {
    path.positionKm = advance(path.positionKm, path.direction, t, planetRadiusKm);
  }
  path.collision = float(next);

  if (next >= max(cloud.referenceMaxBounces, 1u)) {
    return false;
  }
  // Only a throughput below its start (absorption, the other channels' phase weights) plays the roulette:
  // a white cloud's paths run until they leave.
  if (next >= kRussianRouletteStart) {
    const float largest = max(path.throughput.r, max(path.throughput.g, path.throughput.b));
    const float survival = clamp(largest / max(path.rouletteReference, 1e-6f), 0.05f, 1.0f);
    if (survival < 1.0f) {
      if (nextRandom(rng) > survival) {
        return false;
      }
      path.throughput /= survival;
    }
  }
  return true;
}

void addToPixel(uint2 pixel, vec4 value) {
  // A rare non-finite path counts as one that got through untouched, keeping the count exact.
  ReferenceAccumulation[pixel] = ReferenceAccumulation[pixel] + (all(isfinite(value)) ? value : vec4(0.0f, 0.0f, 0.0f, 1.0f));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 tile : SV_DispatchThreadID) {
  const uint2 origin = tile * 4u;
  if (any(origin >= cb.camera.resolution)) {
    return;
  }

  const CloudArgs cloud = CloudCb.cloud;
  const AtmosphereArgs atmosphere = CloudCb.atmosphere;
  const bool restart = !passArgs.level;

  ReferencePath path;
  const vec4 position = restart ? vec4(0.0f, 0.0f, 0.0f, -1.0f) : ReferencePathPosition[tile];
  const vec4 direction = restart ? vec4(0.0f) : ReferencePathDirection[tile];
  const vec4 throughput = restart ? vec4(0.0f) : ReferencePathThroughput[tile];
  path.positionKm = position.xyz;
  path.collision = position.w;
  path.direction = direction.xyz;
  path.throughput = throughput.rgb;
  path.rouletteReference = throughput.w;
  path.radiance = restart ? vec3(0.0f) : ReferencePathRadiance[tile].rgb;
  uint finished = uint(direction.w);

  if (restart) {
    for (uint j = 0u; j < 16u; ++j) {
      const uint2 pixel = origin + uint2(j & 3u, j >> 2u);
      if (all(pixel < cb.camera.resolution)) {
        ReferenceAccumulation[pixel] = vec4(0.0f);
      }
    }
  }

  ReferenceRng rng;
  rng.state = referenceHash(tile.x + tile.y * 4099u + referenceHash(passArgs.phase)) | 1u;

  // Starting a path costs a step of the budget like a collision does.
  const uint budget = max(cloud.referenceBouncesPerFrame, 1u);
  for (uint step = 0u; step < budget; ++step) {
    const uint index = kTileOrder[finished & 15u];
    const uint2 pixel = origin + uint2(index & 3u, index >> 2u);
    if (any(pixel >= cb.camera.resolution)) {
      ++finished;
      continue;
    }
    if (path.collision < 0.0f) {
      if (!startPath(pixel, cloud, atmosphere, path, rng)) {
        addToPixel(pixel, vec4(0.0f, 0.0f, 0.0f, 1.0f));
        ++finished;
      }
      continue;
    }
    if (!scatterPath(cloud, atmosphere, path, rng)) {
      addToPixel(pixel, vec4(path.radiance, 0.0f));
      path.collision = -1.0f;
      ++finished;
    }
  }

  ReferencePathPosition[tile] = vec4(path.positionKm, path.collision);
  ReferencePathDirection[tile] = vec4(path.direction, float(finished));
  ReferencePathThroughput[tile] = vec4(path.throughput, path.rouletteReference);
  ReferencePathRadiance[tile] = vec4(path.radiance, 0.0f);

  // Every pixel of the tile: the mean of its finished paths, or the march's layer until it has one. The error
  // pools the tile's paths against the march over the same pixels, weighted alike, since a pixel's few paths
  // alone are far noisier than the difference being judged.
  vec4 pooledReference = vec4(0.0f);
  vec4 pooledMarch = vec4(0.0f);
  float pooledCount = 0.0f;
  for (uint j = 0u; j < 16u; ++j) {
    const uint2 pixel = origin + uint2(j & 3u, j >> 2u);
    if (any(pixel >= cb.camera.resolution)) {
      continue;
    }
    const uint rank = kTileRank[j];
    const float count = finished > rank ? float((finished - 1u - rank) / 16u + 1u) : 0.0f;
    const vec4 march = CloudMarchLayer[pixel];
    const vec4 sum = ReferenceAccumulation[pixel];
    const vec4 mean = count > 0.0f ? sum / count : march;
    ReferenceLayerOutput[pixel] = mean;
    pooledReference += sum;
    pooledMarch += march * count;
    pooledCount += count;

    if (cloud.debugView == DEBUG_VIEW_CLOUD_REFERENCE) {
      CloudDebugView[pixel] = vec4(mean.rgb, 1.0f);
    }
  }

  // Tiles both see as opaque cloud (a mean transmittance below 5%) with a few paths count toward the
  // statistics: the lighting's error, which a camera's sway smearing the reference's silhouettes would bury.
  const float referenceLuminance = calcBt709Luminance(pooledReference.rgb);
  const bool cloudy = pooledCount >= 4.0f && referenceLuminance > 1e-4f &&
    pooledReference.a < 0.05f * pooledCount && pooledMarch.a < 0.05f * pooledCount;
  const float error = referenceLuminance > 1e-4f && pooledCount > 0.0f ? calcBt709Luminance(pooledMarch.rgb) / referenceLuminance - 1.0f : 0.0f;
  InterlockedAdd(ReferenceStatistics[2], finished);
  if (cloudy) {
    const float magnitude = abs(error);
    const uint bin = min(uint(magnitude * float(CLOUD_REFERENCE_ERROR_BINS) * 0.5f), uint(CLOUD_REFERENCE_ERROR_BINS - 1));
    InterlockedAdd(ReferenceStatistics[0], 1u);
    InterlockedAdd(ReferenceStatistics[1], uint(min(magnitude, 64.0f) * 1024.0f));
    InterlockedAdd(ReferenceStatistics[3], uint(min(calcBt709Luminance(pooledMarch.rgb) / pooledCount, 1024.0f) * 256.0f));
    InterlockedAdd(ReferenceStatistics[4], uint(min(referenceLuminance / pooledCount, 1024.0f) * 256.0f));
    InterlockedAdd(ReferenceStatistics[CLOUD_REFERENCE_STATISTICS_HEADER + bin], 1u);
  }

  if (cloud.debugView == DEBUG_VIEW_CLOUD_REFERENCE_ERROR) {
    // Red where the march is brighter than the reference, blue where darker.
    for (uint j = 0u; j < 16u; ++j) {
      const uint2 pixel = origin + uint2(j & 3u, j >> 2u);
      if (all(pixel < cb.camera.resolution)) {
        CloudDebugView[pixel] = vec4(max(error, 0.0f), 0.0f, max(-error, 0.0f), 1.0f);
      }
    }
  }
}
