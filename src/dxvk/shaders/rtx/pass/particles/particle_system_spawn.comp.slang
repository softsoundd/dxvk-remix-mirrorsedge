/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx/pass/particles/particle_system_bindings.slangh"

#include "rtx/pass/instance_definitions.h"   // OBJECT_MASK_OPAQUE (spawn occlusion trace)

#include "rtx/utility/common.slangh"
#include "rtx/utility/noise.slangh"
#include "rtx/utility/procedural_noise.slangh"
#include "rtx/utility/brdf.slangh"
#include "rtx/utility/packing.slangh"
#include "rtx/utility/gbuffer_helpers.slangh"
#include "rtx/utility/debug_view_helpers.slangh"
#include "rtx/concept/camera/camera.slangh"
    
layout(binding = PARTICLE_SYSTEM_BINDING_SPAWN_CONTEXT_PARTICLE_MAPPING_INPUT)
StructuredBuffer<uint16_t> SpawnContextParticleMapping;

layout(binding = PARTICLE_SYSTEM_BINDING_SPAWN_CONTEXTS_INPUT)
StructuredBuffer<GpuSpawnContext> SpawnContexts;

// Outputs

layout(binding = PARTICLE_SYSTEM_BINDING_PARTICLES_BUFFER_INPUT_OUTPUT)
RWStructuredBuffer<GpuParticle> Particles;

// Spawn-occlusion diagnostics (fork - 2026-07-25). Four counters, cleared by
// the CPU every frame and read back on a delay (see RtxParticleSystemManager::
// simulate): [0] = particles that ran the occlusion trace, [1] = particles
// killed at birth because EVERY relocation attempt found cover overhead,
// [2] = particles whose lifetime was shortened by the downward landing trace,
// [3] = particles rescued by shelter relocation (first candidate was covered,
// a retry found open sky). Surfaced read-only in the Precipitation dev panel
// so "occlusion doesn't work" stops being a guessing game: rays > 0 with both
// hit counters at 0 while standing under a roof means the roof is not
// reachable in the TLAS under OBJECT_MASK_OPAQUE (not submitted, culled, or
// non-opaque) - a completely different bug class than the trace not running at
// all (rays == 0). Relocations > 0 while sheltered with rain visible outside
// means the relocation fix is doing its job.
layout(binding = PARTICLE_SYSTEM_BINDING_SPAWN_TRACE_STATS_OUTPUT)
RWStructuredBuffer<uint> SpawnTraceStats;


f16vec3 sampleDirectionInCone(f16vec3 axis, float cosConeHalfAngle, float2 rndUV)
{
  // Sample a direction in a cone about +Z, and get its PDF (we ignore PDF here)
  float solidAnglePdf;
  f16vec3 localDir = calcUniformConeDirectionSample(vec2(rndUV), cosConeHalfAngle, solidAnglePdf);

  f16vec3 tangent, bitangent;
  calcOrthonormalBasis(axis, tangent, bitangent);

  // Rotate the local-Z sample into the world-axis frame
  return localDir.x * tangent + localDir.y * bitangent + localDir.z * axis;
}

// Everything random about one spawn candidate: a triangle + barycentric point
// on the emitter mesh, the smear along the emitter's frame-to-frame motion, and
// the emission-cone direction. Factored out of respawn() (fork - 2026-07-25) so
// the shelter relocation loop there can draw a fresh, fully independent
// candidate per attempt.
struct SpawnSample
{
  float3 spawnPosition;
  float3 initialVelocity;
  float4 uvMinMax;
  float4 color;
  float  randSeed;
};

SpawnSample sampleSpawnPoint(const GpuSpawnContext spawnCtx, const GpuParticleSystem particleSystem, uint seedBase)
{
  // Get random triangle

  float3 positions[3];
  float3 prevPositions[3];
  float4 color = (spawnCtx.spawnMeshColorsIdx != BINDING_INDEX_INVALID) ? 0.0f : 1.0f;
  float4 uvMinMax = float4(0..xx, 1..xx);

  uint randomSeed = seedBase;

  float3 random01 = float3(randomFloat(uint2(randomSeed++, particleCb.frameIdx)), randomFloat(uint2(randomSeed++, particleCb.frameIdx)), randomFloat(uint2(randomSeed++, particleCb.frameIdx)));
  float3 randomBary = saturate(random01 / dot(random01, 1..xxx));
  
  uint triIdx = randomFloat(uint2(randomSeed++, particleCb.frameIdx)) * spawnCtx.numTriangles;
  
  for (uint i=0 ; i<3 ; i++)
  {
    uint indexIdx = triIdx * 3 + i;
    uint index = spawnCtx.indices32bit ? BUFFER_ARRAY(indices32, (uint)spawnCtx.spawnMeshIndexIdx, indexIdx) : BUFFER_ARRAY(indices, (uint)spawnCtx.spawnMeshIndexIdx, indexIdx);
    
    const uint baseSrcPositionOffset = (spawnCtx.spawnMeshPositionsOffset + index * spawnCtx.spawnMeshPositionsStride) / 4;
    positions[i] = float3(BUFFER_ARRAY(geometries, (uint)spawnCtx.spawnMeshPositionsIdx, baseSrcPositionOffset + 0),
                          BUFFER_ARRAY(geometries, (uint)spawnCtx.spawnMeshPositionsIdx, baseSrcPositionOffset + 1),
                          BUFFER_ARRAY(geometries, (uint)spawnCtx.spawnMeshPositionsIdx, baseSrcPositionOffset + 2));
    
    if (spawnCtx.spawnMeshPrevPositionsIdx != BINDING_INDEX_INVALID)
    {
      prevPositions[i].x = BUFFER_ARRAY(geometries, (uint)spawnCtx.spawnMeshPrevPositionsIdx, baseSrcPositionOffset + 0);
      prevPositions[i].y = BUFFER_ARRAY(geometries, (uint)spawnCtx.spawnMeshPrevPositionsIdx, baseSrcPositionOffset + 1);
      prevPositions[i].z = BUFFER_ARRAY(geometries, (uint)spawnCtx.spawnMeshPrevPositionsIdx, baseSrcPositionOffset + 2);
    }
    else
    {
      prevPositions[i] = positions[i];
    }
    
    if(spawnCtx.spawnMeshColorsIdx != BINDING_INDEX_INVALID)
    {
      const uint baseSrcColorOffset = (spawnCtx.spawnMeshColorsOffset + index * spawnCtx.spawnMeshColorsStride) / 4;
      const uint colorBits = floatBitsToUint(BUFFER_ARRAY(geometries, (uint) spawnCtx.spawnMeshColorsIdx, baseSrcColorOffset));
      // Format is B8G8R8A8_UNORM
      color += unorm4x8ToFloat4x32(colorBits).zyxw * randomBary[i];
    }
    
    // Only fetch emitter UVs when a valid texcoord buffer is bound. For a mesh without UVs,
    // spawnMeshTexcoordsIdx == BINDING_INDEX_INVALID (0xFFFF); reading geometries[0xFFFF] is an
    // out-of-range bindless descriptor access and faults the GPU (device loss). Mirrors the
    // existing guards on the color and prev-position reads above.
    float2 texcoord = float2(0.0f, 0.0f);
    if (spawnCtx.spawnMeshTexcoordsIdx != BINDING_INDEX_INVALID)
    {
      const uint baseSrcTexcoordOffset = (spawnCtx.spawnMeshTexcoordsOffset + index * spawnCtx.spawnMeshTexcoordsStride) / 4;
      texcoord = float2(BUFFER_ARRAY(geometries, (uint)spawnCtx.spawnMeshTexcoordsIdx, baseSrcTexcoordOffset + 0),
                        BUFFER_ARRAY(geometries, (uint)spawnCtx.spawnMeshTexcoordsIdx, baseSrcTexcoordOffset + 1));
    }
    
    if (particleSystem.desc.useSpawnTexcoords && spawnCtx.spawnMeshTexcoordsIdx != BINDING_INDEX_INVALID)
    {
      if (i == 0)
      {
        uvMinMax.xy = texcoord;
        uvMinMax.zw = texcoord;
      }
      else
      {
        uvMinMax.xy = min(uvMinMax.xy, texcoord);
        uvMinMax.zw = max(uvMinMax.zw, texcoord);
      }
    }
  }
  
  // Sample random point on triangle
  const float randSeed = randomFloat(uint2(randomSeed++, particleCb.frameIdx));

  float3 prevWorldPosition = mul(spawnCtx.spawnPrevObjectToWorld, float4(interpolateHitAttribute(prevPositions, randomBary), 1.f)).xyz;
  float3 worldPosition = mul(spawnCtx.spawnObjectToWorld, float4(interpolateHitAttribute(positions, randomBary), 1.f));
  
  float3 worldSpawnVelocity = (worldPosition - prevWorldPosition) * particleCb.invDeltaTimeSecs;
  
  float3 localNormal = cross(normalize(positions[1] - positions[0]), normalize(positions[2] - positions[0]));
  float3 worldNormal = mul(spawnCtx.spawnObjectToWorld, float4(localNormal, 0.f));
  
  float cosCone = cos(radians(particleSystem.desc.initialVelocityConeAngleDegrees));
  f16vec3 dir = sampleDirectionInCone(worldNormal, cosCone, float2(randomFloat(uint2(randomSeed++, particleCb.frameIdx)), randomFloat(uint2(randomSeed++, particleCb.frameIdx))));
  
  SpawnSample result;
  result.initialVelocity = worldSpawnVelocity * particleSystem.desc.initialVelocityFromMotion + dir * particleSystem.desc.initialVelocityFromNormal * particleCb.sceneScale;
  result.spawnPosition = lerp(worldPosition, prevWorldPosition, randSeed);
  result.uvMinMax = uvMinMax;
  result.color = color;
  result.randSeed = randSeed;
  return result;
}

// How many spawn candidates the shelter probe may try before giving up on a
// particle (fork - 2026-07-25). 1 would be the old kill-at-birth behaviour;
// 8 lets a spawn volume that is 95% covered still land ~1/3 of its budget in
// the open 5% (1 - 0.95^8). Worst case cost is 8 short upward rays per spawned
// particle, and only in fully-enclosed spaces where that upper bound is hit.
static const uint kShelterRelocationAttempts = 8;

// Generate a new particle in-place with random properties
GpuParticle respawn(uint particleIdx, uint spawnCtxIdx, const GpuParticleSystem particleSystem)
{
  const GpuSpawnContext spawnCtx = SpawnContexts[spawnCtxIdx];

  SpawnSample candidate = sampleSpawnPoint(spawnCtx, particleSystem, particleIdx);

  // Occlusion traces (fork - 2026-07-25). Ask the actual scene geometry
  // whether this particle should exist at all, and if so, where it should die.
  //
  // The alternative already in the codebase (enableCollisionDetection) samples
  // the previous frame's G-buffer, which is screen-space: it sees nothing
  // off-screen, behind the camera, or outside the view frustum. That is exactly
  // the geometry that matters for sheltering - the roof over the player's head
  // is usually not on screen. Tracing the TLAS answers it properly, so rain
  // stops under roofs, bridges and overhangs whether or not they are in view.
  //
  //  1. SHELTER PROBE with RELOCATION - upward, against the fall direction,
  //     from each candidate spawn position. The spawn plane hangs
  //     spawnHeightMeters * meterToWorldUnitScale above the CAMERA, and
  //     whenever the sheltering geometry sits ABOVE that plane the drop is
  //     born already inside the sheltered space: a downward ray can never see
  //     the roof, because the roof is behind the ray's origin. Probing UP from
  //     the spawn position is correct at ANY scale: if opaque geometry hangs
  //     anywhere overhead, this drop would have been stopped up there, so it
  //     must not exist down here.
  //
  //     A sheltered candidate is NOT killed outright. The first shipped
  //     version did that, and it produced the second reported bug ("step
  //     under a roof and ALL rain stops, even the rain outside the door"):
  //     the whole spawn volume is a camera-centred disc, and with the
  //     integration's understated scene scale (FO4 runs rtx.sceneScale = 0.1
  //     => 10 units per authored meter, against a world that is really ~70
  //     units/m) that disc is only a ~3 REAL meter bubble around the player's
  //     head - so any porch, canopy or awning covers ALL of it, every drop
  //     died at birth, and nothing was left to fall ANYWHERE, including in
  //     the open air visible through the doorway. Instead, a sheltered
  //     candidate REROLLS: draw a fresh random point on the emitter (up to
  //     kShelterRelocationAttempts in all) and keep the first one with open
  //     sky. The particle budget migrates to wherever the sky is actually
  //     visible - exactly the rain you can see from inside a shelter - at any
  //     scene scale, with no dependence on how big the covering geometry is.
  //     Only when every attempt is covered (a genuinely enclosed interior) is
  //     the drop killed at birth: timeToLive = 0 means born sleeping, the
  //     geometry pass skips sleeping particles (never rendered for even a
  //     frame), and evolve retires it through the normal conservative-counter
  //     path - the exact lifecycle upstream's Kill collision mode already
  //     uses.
  //
  //     DENSITY CAVEAT, accepted deliberately: relocation conserves the spawn
  //     rate, so when most of the disc is covered the surviving fraction
  //     concentrates in the uncovered part, which can read as up-to-Kx denser
  //     rain framed in a doorway. That errs on the visible side, which is the
  //     point of the fix; the alternative (kill proportionally) is exactly
  //     the starvation this replaces.
  //
  //  2. LANDING TRACE - along the accepted candidate's spawn velocity for the
  //     whole ballistic path, shortening timeToLive so the drop dies at the
  //     surface it would land on. For rain (no drag, no turbulence) the path
  //     IS a straight line, so this is exact; heavy turbulence/drag curves it
  //     and makes this an approximation, which is an acceptable trade for the
  //     cost. Skipped when the shelter probe killed the particle.
  //
  // All rays are opaque-only (OBJECT_MASK_OPAQUE + RAY_FLAG_CULL_NON_OPAQUE):
  // alpha-blended and translucent surfaces (glass, decals, particles) and
  // alpha-TESTED cards (foliage, chain fences, gratings) do not stop
  // precipitation, and skipping them keeps traversal single-pass with no
  // any-hit. This is a verified-safe combination in this integration: fully
  // opaque draws get VK_GEOMETRY_OPAQUE_BIT_KHR + OBJECT_MASK_OPAQUE
  // (rtx_instance_manager.cpp, determine-geometry-flags block), so solid world
  // geometry is never culled by RAY_FLAG_CULL_NON_OPAQUE; and because every
  // in-mask candidate is opaque (and the Opaque TLAS carries no in-mask AABB
  // primitives), a single Proceed() completes the whole traversal - the
  // "while (Proceed()) commit candidates" idiom is only needed when non-opaque
  // candidates can occur.
  //
  // topLevelAS is LAST frame's structure (particle simulation runs before this
  // frame's TLAS build) and may not exist at all early in a scene - hence the
  // sceneTlasValid gate.
  const bool traceOcclusion = particleSystem.desc.traceSpawnOcclusion && particleCb.sceneTlasValid != 0;
  bool sheltered = false;

  if (traceOcclusion)
  {
    InterlockedAdd(SpawnTraceStats[0], 1);

    // Shelter probe reach: 1 km of AUTHORED distance (100000 cm scaled by
    // sceneScale, the same units-per-centimeter factor every other spatial
    // particle parameter uses). Deliberately far: with a mis-stated scene
    // scale the authored meter shrinks relative to the real world (FO4: 7x),
    // and the probe must still clear real-world bridge decks. Sky-bound rays
    // leave the TLAS bounds almost immediately, so the long TMax costs
    // nothing in the common uncovered case.
    const float shelterProbeDistance = 100000.0f * particleCb.sceneScale;

    bool relocated = false;
    for (uint attempt = 0; attempt < kShelterRelocationAttempts; ++attempt)
    {
      if (attempt > 0)
      {
        // Fresh candidate. The seed base strides by maxNumParticles so a
        // retry can never replay the seed window of another particle's first
        // attempt in the same frame, which would spawn two drops at the
        // identical position.
        candidate = sampleSpawnPoint(spawnCtx, particleSystem,
                                  particleIdx + attempt * particleSystem.desc.maxNumParticles);
      }

      const float speed = length(candidate.initialVelocity);
      if (!(speed > 1e-6f))
      {
        // Motionless particle: no fall direction to probe against. Accept it
        // untraced, matching the pre-relocation behaviour.
        sheltered = false;
        break;
      }

      RayDesc shelterRay;
      shelterRay.Origin = candidate.spawnPosition;
      shelterRay.Direction = -(candidate.initialVelocity / speed);
      shelterRay.TMin = 0.0f;
      shelterRay.TMax = shelterProbeDistance;

      RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_NON_OPAQUE> shelterQuery;
      shelterQuery.TraceRayInline(topLevelAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_NON_OPAQUE, OBJECT_MASK_OPAQUE, shelterRay);
      shelterQuery.Proceed();

      sheltered = shelterQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
      if (!sheltered)
      {
        relocated = attempt > 0;
        break;
      }
    }

    if (relocated)
    {
      InterlockedAdd(SpawnTraceStats[3], 1);
    }
  }

  // Spawn the particle from the accepted candidate (or from the last sheltered
  // candidate, which is about to be killed and never rendered).
  GpuParticle newParticle;
  newParticle.reset(particleSystem, candidate.spawnPosition, candidate.initialVelocity, candidate.uvMinMax, candidate.color, candidate.randSeed);

  if (traceOcclusion)
  {
    if (sheltered)
    {
      // Every candidate had cover overhead: a genuinely enclosed space. Born
      // sleeping - see the block comment.
      newParticle.timeToLive = 0.0f;
      InterlockedAdd(SpawnTraceStats[1], 1);
    }
    else
    {
      const float speed = length(candidate.initialVelocity);
      if (speed > 1e-6f)
      {
        const float maxDistance = speed * float(newParticle.timeToLive);

        RayDesc ray;
        ray.Origin = candidate.spawnPosition;
        ray.Direction = candidate.initialVelocity / speed;
        ray.TMin = 0.0f;
        ray.TMax = maxDistance;

        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_NON_OPAQUE> rayQuery;
        rayQuery.TraceRayInline(topLevelAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_NON_OPAQUE, OBJECT_MASK_OPAQUE, ray);
        rayQuery.Proceed();

        if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
          // Die at the surface. Nudged slightly short so the final frame of
          // the particle is just above the geometry rather than intersecting
          // it.
          const float hitDistance = rayQuery.CommittedRayT();
          newParticle.timeToLive = min(newParticle.timeToLive, max(hitDistance * 0.98f, 0.0f) / speed);
          InterlockedAdd(SpawnTraceStats[2], 1);
        }
      }
    }
  }

  return newParticle;
}


[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint2 threadIdx : SV_DispatchThreadID)
{
  const GpuParticleSystem particleSystem = particleCb.particleSystem;
  
  if(threadIdx.x >= particleSystem.spawnParticleCount)
  {
    return;
  }
  
  const uint particleIdx = (threadIdx.x + particleSystem.spawnParticleOffset) % particleSystem.desc.maxNumParticles;
  const uint spawnCtxIdx = SpawnContextParticleMapping[threadIdx.x];
  const bool shouldRespawn = Particles[particleIdx].isDead() || Particles[particleIdx].isSleeping();
  if (shouldRespawn)
  {
    Particles[particleIdx] = respawn(particleIdx, spawnCtxIdx, particleSystem);
  }
}
