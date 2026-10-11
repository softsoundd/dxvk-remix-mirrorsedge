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

// How much of the sun's disc the camera sees, traced through the scene rather than read from the image. Each thread
// traces one ray to its stratum of the disc, a 16 by 16 grid of the unit square mapped onto the disc by Shirley and
// Chiu's concentric map, which keeps the strata's areas equal. The rays' jitter within their strata is fixed, so a
// partly covered sun measures the same from frame to frame. The rays are weighted by the disc's limb darkening.

#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/sun_probe/sun_probe.h"

#include "rtx/utility/common.slangh"
#include "rtx/utility/color.slangh"
#include "rtx/utility/noise.slangh"
#include "rtx/concept/ray/ray.slangh"
#include "rtx/algorithm/visibility.slangh"
#include "rtx/pass/atmosphere/atmosphere_common.slangh"
#include "rtx/pass/clouds/cloud_common.slangh"

layout(push_constant)
ConstantBuffer<SunVisibilityArgs> visibilityArgs;

layout(rgba32f, binding = SUN_VISIBILITY_OUTPUT)
RWTexture2D<float4> OutVisibility;

groupshared vec3 g_visible[SUN_VISIBILITY_RAYS];
groupshared vec3 g_weight[SUN_VISIBILITY_RAYS];

vec2 concentricDisc(vec2 u)
{
  const vec2 p = 2.0f * u - 1.0f;

  if (all(p == vec2(0.0f)))
  {
    return vec2(0.0f);
  }

  float radius;
  float angle;

  if (abs(p.x) > abs(p.y))
  {
    radius = p.x;
    angle = 0.25f * pi * (p.y / p.x);
  }
  else
  {
    radius = p.y;
    angle = 0.5f * pi - 0.25f * pi * (p.x / p.y);
  }

  return radius * vec2(cos(angle), sin(angle));
}

[shader("compute")]
[numthreads(SUN_VISIBILITY_RAYS, 1, 1)]
void main(uint threadIndex : SV_GroupIndex)
{
  const vec2 jitter = vec2(randomFloat(uint2(threadIndex, 1u)), randomFloat(uint2(threadIndex, 2u)));
  const vec2 stratum = vec2(float(threadIndex % 16u), float(threadIndex / 16u));
  const vec2 disc = concentricDisc((stratum + jitter) / 16.0f);

  const vec3 direction = normalize(visibilityArgs.sunDirection + disc.x * visibilityArgs.discRight + disc.y * visibilityArgs.discUp);

  Ray ray = rayCreateEmpty();
  ray.origin = visibilityArgs.origin + direction * visibilityArgs.tMin;
  ray.direction = direction;
  ray.tMax = visibilityArgs.tMax;

  // What the camera sees, as its primary rays do: the player's arms, but not the third-person body the game keeps for
  // shadows and reflections, which lags the camera through fast turns, unless it is shown to the camera.
  uint8_t rayMask = OBJECT_MASK_OPAQUE | OBJECT_MASK_ALPHA_BLEND | OBJECT_MASK_TRANSLUCENT | OBJECT_MASK_VIEWMODEL | OBJECT_MASK_PLAYER_MODEL_VIRTUAL;

  if (cb.enablePlayerModelInPrimarySpace)
  {
    rayMask |= OBJECT_MASK_PLAYER_MODEL;
  }

  const VisibilityResult visibility = traceVisibilityRay<visibilityModeDisableCulling | visibilityModeEnableTranslucentMaterials>(
    ray, rayMask, invalidRayPortalIndex, PORTAL_SPACE_NONE, true);

  const float rho = min(length(disc), 1.0f);
  const float mu = sqrt(max(1.0f - rho * rho, 1e-6f));
  const vec3 weight = pow(vec3(mu), visibilityArgs.limbDarkeningExponent);

  g_visible[threadIndex] = weight * (visibility.hasOpaqueHit ? vec3(0.0f) : saturate(visibility.attenuation));
  g_weight[threadIndex] = weight;
  GroupMemoryBarrierWithGroupSync();

  for (uint stride = SUN_VISIBILITY_RAYS / 2; stride > 0; stride /= 2)
  {
    if (threadIndex < stride)
    {
      g_visible[threadIndex] += g_visible[threadIndex + stride];
      g_weight[threadIndex] += g_weight[threadIndex + stride];
    }

    GroupMemoryBarrierWithGroupSync();
  }

  if (threadIndex != 0)
  {
    return;
  }

  vec3 result = g_visible[0] / max(g_weight[0], vec3(1e-6f));

  if ((visibilityArgs.flags & SUN_VISIBILITY_FLAG_CLOUDS) != 0)
  {
    result *= cloudSunTransmittance(CloudSunGrid, CloudShadowMap, CloudSampler, visibilityArgs.origin, cb.cloudArgs, cb.atmosphereArgs);
  }

  if ((visibilityArgs.flags & SUN_VISIBILITY_FLAG_FOG) != 0 && cb.volumeArgs.enable)
  {
    result *= exp(-cb.volumeArgs.attenuationCoefficient * cb.volumeArgs.froxelMaxDistance);
  }

  OutVisibility[uint2(0, 0)] = float4(saturate(result), 1.0f);
}
