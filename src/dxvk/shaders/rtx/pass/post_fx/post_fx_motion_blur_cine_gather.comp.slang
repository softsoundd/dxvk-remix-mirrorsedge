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

// Cinematic motion blur, pass 4 of 4: the reconstruction filter.
//
// Guertin, McGuire & Nowrouzezahrai, HPG 2014, Appendix A. Samples are split between the
// tile's dominant velocity and a centre direction so that overlapping and opposed motion
// both resolve, weighted by how well each sample's own velocity aligns with the direction
// being walked, and placed along a curved path when the camera is sweeping an arc.

#include "rtx/pass/post_fx/post_fx.h"
#include "rtx/pass/post_fx/post_fx_motion_blur_cine_common.slangh"

[[vk::binding(POST_FX_MB_CINE_GATHER_VELOCITY_DEPTH_INPUT)]]
Texture2D<float4> VelocityDepth;

[[vk::binding(POST_FX_MB_CINE_GATHER_CURVATURE_INPUT)]]
Texture2D<float2> Curvature;

[[vk::binding(POST_FX_MB_CINE_GATHER_TILE_MAX_INPUT)]]
Texture2D<float4> TileMax;

[[vk::binding(POST_FX_MB_CINE_GATHER_NEIGHBOR_MAX_INPUT)]]
Texture2D<float4> NeighborMax;

[[vk::binding(POST_FX_MB_CINE_GATHER_COLOR_INPUT)]]
Texture2D<float4> InColorTexture;

[[vk::binding(POST_FX_MB_CINE_GATHER_OUTPUT)]]
RWTexture2D<float4> OutColorTexture;

[[vk::binding(POST_FX_MB_CINE_GATHER_LINEAR_SAMPLER)]]
SamplerState LinearSampler;

layout(push_constant)
ConstantBuffer<PostFxMotionBlurCineGatherArgs> cb;

// Paper section 4.2. Blur direction changes abruptly at tile boundaries, and interpolating
// velocities across them is ill defined - two anti-parallel trails do not average to
// anything meaningful, and blurring along the mean produces wavey artefacts worse than the
// seam. Jittering the lookup instead trades that banding for noise.
//
// Displacing the coordinate keeps this continuous: the chance of landing in the adjacent
// tile rises smoothly toward the border, reaching exactly half at it. Picking the axis at
// random rather than from geometry matters - keying it off which border is nearer partitions
// every tile into four triangles along its diagonals, plainly visible in the output. One
// axis only, so the lookup never reaches a diagonal tile, where two unrelated trails meeting
// at a corner would combine into a direction neither object is moving in.
int2 jitteredTileCoord(const uint2 pixelPos, const float axisRandom, const float offsetRandom)
{
  float2 tileSpace = (float2(pixelPos) + 0.5f) / float(cb.tileSize);

  const float offset = (offsetRandom - 0.5f) * cb.tileBlendSlope;

  if (axisRandom < 0.5f)
  {
    tileSpace.x += offset;
  }
  else
  {
    tileSpace.y += offset;
  }

  return int2(floor(tileSpace));
}

// Distribute samples between the two directions. Spread rather than partitioned, so the
// jittered sequence stays stratified along both.
bool useCenterDirection(const uint sampleIndex, const uint sampleCount, const float variance)
{
  if (cb.directionSplit == POST_FX_MB_DIRECTION_SPLIT_DOMINANT)
  {
    return false;
  }

  if (cb.directionSplit == POST_FX_MB_DIRECTION_SPLIT_EVEN)
  {
    return (sampleIndex & 1u) == 1u;
  }

  // Eq. 2 driven split: allocate variance * N samples to the centre direction, distributed
  // evenly through the sequence with a Bresenham style step.
  const uint centerCount = uint(variance * float(sampleCount) + 0.5f);
  return ((sampleIndex + 1u) * centerCount) / sampleCount != (sampleIndex * centerCount) / sampleCount;
}

[shader("compute")]
[numthreads(POST_FX_TILE_SIZE, POST_FX_TILE_SIZE, 1)]
void main(in uint2 pixelPos : SV_DispatchThreadID)
{
  if (any(pixelPos >= cb.imageSize))
  {
    return;
  }

  const float4 centerColor = InColorTexture[pixelPos];
  const MotionBlurSample center = motionBlurUnpackVelocityDepth(VelocityDepth[pixelPos]);

  // Three decorrelated noise streams. Interleaved gradient noise is cheap enough to evaluate
  // per use, but sampling it at nearly the same position would leave the tile jitter and the
  // sample jitter correlated, which shows up as structure rather than noise.
  const float axisRandom = motionBlurInterleavedGradientNoise(float2(pixelPos) + float2(17.0f, 43.0f), cb.frameIdx);
  const float tileRandom = motionBlurInterleavedGradientNoise(float2(pixelPos) + float2(101.0f, 7.0f), cb.frameIdx);

  const int2 maxTile = int2(cb.tileCount) - 1;
  const int2 baseTile = clamp(int2(pixelPos) / int(cb.tileSize), int2(0, 0), maxTile);
  const int2 tileCoord = clamp(jitteredTileCoord(pixelPos, axisRandom, tileRandom), int2(0, 0), maxTile);

  const float4 neighborMax = NeighborMax[tileCoord];
  const float2 dominantVelocity = neighborMax.xy;
  const float dominantSpeed = neighborMax.z;
  const float variance = neighborMax.w;

  // Blur amount is measured from the finished gather, so it falls through rather than
  // short circuiting here like the input visualisations do.
  if (cb.debugView != POST_FX_MB_DEBUG_OFF && cb.debugView != POST_FX_MB_DEBUG_BLUR_AMOUNT)
  {
    // Read through the unjittered tile. The jitter is a sampling decision, not tile content,
    // and visualising it just draws its own noise over whatever is being inspected.
    const float4 tileMaxUnjittered = TileMax[baseTile];
    const float4 neighborMaxUnjittered = NeighborMax[baseTile];
    const float velocityScale = 1.0f / max(float(cb.tileSize), 1.0f);

    float3 debugColor = centerColor.rgb;
    switch (cb.debugView)
    {
      case POST_FX_MB_DEBUG_VELOCITY:
        debugColor = float3(abs(center.velocity) * velocityScale, center.validity * 0.25f);
        break;
      case POST_FX_MB_DEBUG_TILE_MAX:
        debugColor = float3(abs(tileMaxUnjittered.xy) * velocityScale, 0.0f);
        break;
      case POST_FX_MB_DEBUG_NEIGHBOR_MAX:
        debugColor = float3(abs(neighborMaxUnjittered.xy) * velocityScale, 0.0f);
        break;
      case POST_FX_MB_DEBUG_VARIANCE:
        debugColor = float3(neighborMaxUnjittered.w, 1.0f - neighborMaxUnjittered.w, 0.0f);
        break;
    }
    OutColorTexture[pixelPos] = float4(debugColor, centerColor.a);
    return;
  }

  // Nothing in this tile's neighbourhood moves far enough to reach here, or this surface is
  // excluded from motion blur entirely.
  if (dominantSpeed < max(cb.minVelocityPixels, POST_FX_MB_MIN_SPEED) || center.validity <= 0.0f)
  {
    OutColorTexture[pixelPos] = cb.debugView == POST_FX_MB_DEBUG_BLUR_AMOUNT
      ? float4(0.0f, 1.0f, 0.0f, centerColor.a)
      : centerColor;
    return;
  }

  const float centerSpeedRaw = length(center.velocity);
  const float2 centerCurvature = Curvature[pixelPos];

  // Paper Eq. 1. Blend from the pixel's own velocity toward the direction perpendicular to
  // the dominant one as the pixel slows down: once a pixel is barely moving, its velocity
  // carries no useful direction, and the perpendicular is the best guess at a secondary
  // motion the dominant velocity would otherwise mask entirely.
  const float2 dominantDirection = motionBlurSafeNormalize(dominantVelocity, float2(1.0f, 0.0f));
  float2 perpendicularDirection = motionBlurPerpendicular(dominantDirection);
  if (dot(perpendicularDirection, center.velocity) < 0.0f)
  {
    perpendicularDirection = -perpendicularDirection;
  }

  const float centerDirectionBlend = saturate((centerSpeedRaw - POST_FX_MB_MIN_SPEED) / max(cb.gammaThreshold, 1e-4f));
  const float2 centerDirection = motionBlurSafeNormalize(
    lerp(perpendicularDirection, motionBlurSafeNormalize(center.velocity, perpendicularDirection), centerDirectionBlend),
    perpendicularDirection);

  // Both directions span the same extent, so sample density stays uniform over the blur
  // regardless of which one a tap was placed on, and the two share a curvature scale.
  const float2 directionDominant = dominantVelocity;
  const float2 directionCenter = centerDirection * dominantSpeed;

  const float2 walkCurvature = cb.enableCurvedPaths != 0
    ? motionBlurCurvatureForDirection(centerCurvature, dominantSpeed, centerSpeedRaw)
    : float2(0.0f, 0.0f);

  const float arcDominant = motionBlurPathHalfLength(directionDominant, walkCurvature);
  const float arcCenter = motionBlurPathHalfLength(directionCenter, walkCurvature);
  const float centerSpeed = motionBlurPathHalfLength(center.velocity, cb.enableCurvedPaths != 0 ? centerCurvature : float2(0.0f, 0.0f));

  const uint sampleCount = max(min(cb.sampleCount, uint(POST_FX_MB_MAX_SAMPLE_COUNT)), 1u);

  // Paper section 4.3. Normalising by N/k keeps the centre tap's influence constant as the
  // sample count changes, which is what stops thin features from ghosting or dissolving at
  // low sample counts.
  float totalWeight = float(sampleCount) / (max(cb.centerWeightBias, 1e-3f) * max(centerSpeed, POST_FX_MB_MIN_SPEED));
  const float centerTapWeight = totalWeight;
  float3 accumulatedColor = centerColor.rgb * totalWeight;

  // Paper section 4.5. Per pixel jitter, scaled down as the sample count rises so that the
  // noise it trades for banding shrinks when there are enough samples not to band.
  const float perPixelNoise = motionBlurInterleavedGradientNoise(float2(pixelPos), cb.frameIdx);
  const float jitter = (perPixelNoise * 2.0f - 1.0f) * 0.95f * cb.jitterScale / float(sampleCount);

  const float2 centerPixel = float2(pixelPos) + 0.5f;

  for (uint i = 0u; i < sampleCount; ++i)
  {
    const float t = lerp(-1.0f, 1.0f, (float(i) + jitter + 1.0f) / (float(sampleCount) + 1.0f));

    const bool centerDirectionSample = useCenterDirection(i, sampleCount, variance);
    const float2 direction = centerDirectionSample ? directionCenter : directionDominant;
    const float directionArc = centerDirectionSample ? arcCenter : arcDominant;

    const float2 samplePosition = motionBlurEvaluatePath(centerPixel, direction, walkCurvature, t);
    const int2 samplePixel = clamp(int2(floor(samplePosition)), int2(0, 0), int2(cb.imageSize) - 1);

    // Velocity, depth and validity are point sampled: interpolating them across a silhouette
    // would blend a foreground surface's motion with the background's. Colour is filtered,
    // which is what keeps the trail smooth rather than stepped at low sample counts.
    const MotionBlurSample tap = motionBlurUnpackVelocityDepth(VelocityDepth[samplePixel]);
    const float3 tapColor = InColorTexture.SampleLevel(LinearSampler, samplePosition * cb.invImageSize, 0.0f).rgb;

    // Distance travelled along the arc rather than the straight line to the sample, so a
    // curved trail's falloff stays consistent with its actual length.
    const float distanceAlongPath = abs(t) * directionArc;

    // Linear view Z grows with distance, the opposite of the convention the paper's
    // pseudocode assumes, so the sample depth leads for the foreground term.
    const float foreground = motionBlurZCompare(tap.linearZ, center.linearZ);
    const float background = motionBlurZCompare(center.linearZ, tap.linearZ);

    const float2 normalizedDirection = motionBlurSafeNormalize(direction, dominantDirection);
    const float centerAlignment = abs(dot(centerDirection, normalizedDirection));
    const float sampleAlignment = abs(dot(motionBlurSafeNormalize(tap.velocity, normalizedDirection), normalizedDirection));

    // The sample's own curvature is a second order effect on its speed, so its straight line
    // magnitude is used here rather than paying for another arc length per tap.
    const float tapSpeed = length(tap.velocity);

    const float weight = motionBlurSampleWeight(
      distanceAlongPath, centerSpeed, tapSpeed,
      foreground, background, centerAlignment, sampleAlignment) * tap.validity;

    accumulatedColor += tapColor * weight;
    totalWeight += weight;
  }

  if (cb.debugView == POST_FX_MB_DEBUG_BLUR_AMOUNT)
  {
    // Green where the pixel kept its own colour, red where the filter replaced it with
    // colour gathered from elsewhere. Anything strongly red well away from a moving object
    // is the filter reaching further than it should.
    const float borrowed = saturate(1.0f - centerTapWeight / max(totalWeight, 1e-6f));
    OutColorTexture[pixelPos] = float4(borrowed, 1.0f - borrowed, 0.0f, centerColor.a);
    return;
  }

  // This normalisation is the silhouette reconstruction: background revealed by a moving
  // object arrives through the b * cone(T, centerSpeed) term already, in proportion to how
  // much of the shutter it was visible for. Estimating a background separately and blending
  // it on top double counts the same contribution.
  OutColorTexture[pixelPos] = float4(accumulatedColor / max(totalWeight, 1e-6f), centerColor.a);
}
