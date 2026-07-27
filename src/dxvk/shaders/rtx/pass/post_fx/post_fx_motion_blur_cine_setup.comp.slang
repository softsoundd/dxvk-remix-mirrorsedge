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

// Cinematic motion blur, pass 1 of 4.
//
// Resolves the G-buffer inputs (which sit at render resolution) onto the colour image
// (which sits at display resolution after upscaling), applies the shutter, the dynamic and
// frame generation deductions and the radius clamp once, and fits the curved sample path.
// Everything downstream then works in a single space with a single fetch per tap.

#include "rtx/pass/post_fx/post_fx.h"
#include "rtx/pass/post_fx/post_fx_motion_blur_geometry_flags.slangh"
#include "rtx/pass/post_fx/post_fx_motion_blur_cine_common.slangh"

[[vk::binding(POST_FX_MB_CINE_SETUP_MOTION_VECTOR_INPUT)]]
Texture2D<float2> ScreenSpaceMotionVector;

[[vk::binding(POST_FX_MB_CINE_SETUP_SURFACE_FLAGS_INPUT)]]
Texture2D<uint> PrimarySurfaceFlags;

[[vk::binding(POST_FX_MB_CINE_SETUP_LINEAR_VIEW_Z_INPUT)]]
Texture2D<float> PrimaryLinearViewZ;

[[vk::binding(POST_FX_MB_CINE_SETUP_PREV_MOTION_VECTOR_INPUT)]]
Texture2D<float2> PreviousScreenSpaceMotionVector;

[[vk::binding(POST_FX_MB_CINE_SETUP_VELOCITY_DEPTH_OUTPUT)]]
RWTexture2D<float4> VelocityDepthOutput;

layout(rg16f, binding = POST_FX_MB_CINE_SETUP_CURVATURE_OUTPUT)
RWTexture2D<float2> CurvatureOutput;

layout(binding = POST_FX_MB_CINE_SETUP_CONSTANTS)
ConstantBuffer<PostFxMotionBlurCineSetupArgs> cb;

int2 toGBufferPixel(const uint2 displayPixel)
{
  const float2 gbufferPos = (float2(displayPixel) + 0.5f) * cb.inputOverOutputViewSize - 0.5f;
  const int2 maxPixel = int2(cb.gbufferResolution) - 1;
  return clamp(int2(round(gbufferPos)), int2(0, 0), maxPixel);
}

// Project this pixel through one of the half frame camera matrices and return the resulting
// display space pixel position. Sub pixel jitter is deliberately ignored: it is bounded by a
// pixel while the curvature it feeds is a second order correction measured over tens of
// pixels, so folding it in would cost a constant buffer field for no visible difference.
bool projectThroughCurveMatrix(
  const float2 currentNdc,
  const float  ndcDepth,
  const mat4   curveMatrix,
  out float2   resultPixel)
{
  resultPixel = float2(0.0f, 0.0f);

  const float4 clipPosition = mul(curveMatrix, float4(currentNdc, ndcDepth, 1.0f));
  if (abs(clipPosition.w) < 1e-6f)
  {
    return false;
  }

  const float2 resultNdc = clipPosition.xy / clipPosition.w;
  resultPixel = (resultNdc * float2(0.5f, -0.5f) + 0.5f) * float2(cb.imageSize);

  return true;
}

[shader("compute")]
[numthreads(POST_FX_TILE_SIZE, POST_FX_TILE_SIZE, 1)]
void main(in uint2 pixelPos : SV_DispatchThreadID)
{
  if (any(pixelPos >= cb.imageSize))
  {
    return;
  }

  const int2 gbufferPixel = toGBufferPixel(pixelPos);

  const MotionBlurSurfaceFlags surfaceFlags =
    motionBlurSurfaceFlagsReadFromGBuffer(PrimarySurfaceFlags[gbufferPixel]);

  const bool isValid =
    (cb.enableEmissive != 0 || !surfaceFlags.isEmissive) &&
    !surfaceFlags.isViewModel &&
    !surfaceFlags.isMaskOut;

  const float linearZ = max(PrimaryLinearViewZ[gbufferPixel], 1e-4f);

  // Motion vectors are stored in G-buffer pixels and point from the current position to the
  // previous one, so the forward displacement over a frame is its negation. Rescale to
  // display pixels, since the colour image this blur runs on is post upscale.
  const float2 gbufferToDisplay = 1.0f / max(cb.inputOverOutputViewSize, float2(1e-6f, 1e-6f));
  const float2 motionVectorDisplay = ScreenSpaceMotionVector[gbufferPixel] * gbufferToDisplay;

  const float deduction = surfaceFlags.isStatic ? 1.0f : cb.dynamicDeduction;
  const float shutterScale = cb.shutterFraction * deduction * cb.dlfgDeduction;

  // Half extent of the trail: half a frame of displacement, scaled by the shutter fraction.
  float2 velocity = -0.5f * motionVectorDisplay * shutterScale;
  float2 curvature = float2(0.0f, 0.0f);

  if (cb.enableCurvedPaths != 0)
  {
    const float2 centerPixel = float2(pixelPos) + 0.5f;

    // The path is evaluated as c + v*t - k*t^2 with t in [-1,1], so folding the shutter into
    // the parameter means the quadratic term takes the square of the scale.
    const float curvatureScale = shutterScale * shutterScale;

    // Camera derived arc. Analytically exact, and the dominant source of curvature in a
    // first person game: even at constant angular velocity a turn sweeps static geometry
    // along a curve, because the perspective divide is non-linear.
    bool haveCameraCurvature = false;
    float2 cameraPredictedMinusOne = float2(0.0f, 0.0f);

    if (cb.cameraValid != 0)
    {
      const float2 uv = centerPixel * cb.invImageSize;
      const float2 currentNdc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);

      // Linear view Z back to a projected depth, so the curve matrices can consume the same
      // NDC triple the motion vector pass does.
      const float depthRange = max(cb.farPlane - cb.nearPlane, 1e-4f);
      const float projectionA = cb.farPlane / depthRange;
      const float projectionB = -cb.nearPlane * cb.farPlane / depthRange;
      const float ndcDepth = saturate(projectionA + projectionB / linearZ);

      float2 previousHalfPixel;
      float2 nextHalfPixel;

      if (projectThroughCurveMatrix(currentNdc, ndcDepth, cb.curveToPrevHalfClip, previousHalfPixel) &&
          projectThroughCurveMatrix(currentNdc, ndcDepth, cb.curveToNextHalfClip, nextHalfPixel))
      {
        // k = c - midpoint of the chord: zero for straight line motion, growing with the
        // sagitta of the arc this pixel is swept along.
        curvature = (centerPixel - (previousHalfPixel + nextHalfPixel) * 0.5f) * curvatureScale;
        haveCameraCurvature = true;

        // Where the camera alone would have put this pixel a whole frame ago.
        cameraPredictedMinusOne = centerPixel + 2.0f * (previousHalfPixel - centerPixel);
      }
    }

    // Per object curvature. The camera arc only describes geometry standing still in the
    // world; a swinging limb, a rotating fan or a moving platform bends along its own path.
    // Sampling last frame's motion vector where this surface point was gives a genuine third
    // position p(-2), and the same quadratic fit recovers that bend.
    //
    // Note this cannot key off the static surface flag: the NGX passthrough motion vector
    // pass marks every pixel static because object motion is already folded into the vector
    // field. Comparing the actual displacement against the camera's prediction identifies
    // moving geometry in both pipelines.
    if (cb.enableObjectCurvature != 0)
    {
      const float2 actualMinusOne = centerPixel + motionVectorDisplay;
      const bool objectIsMoving = !haveCameraCurvature ||
                                  length(actualMinusOne - cameraPredictedMinusOne) > 0.5f;

      const int2 previousPixel = int2(round(float2(gbufferPixel) + ScreenSpaceMotionVector[gbufferPixel]));

      if (objectIsMoving &&
          all(previousPixel >= int2(0, 0)) &&
          all(previousPixel < int2(cb.gbufferResolution)))
      {
        const float2 previousMotionDisplay = PreviousScreenSpaceMotionVector[previousPixel] * gbufferToDisplay;

        // Reprojecting a frame back lands on unrelated geometry wherever something was
        // occluded, and silhouettes are where that happens. Reading an acceleration off two
        // different surfaces bends the sample path along a direction nothing actually moved
        // in, which is the same failure mode a bad optical flow field produces and it shows
        // up exactly on outlines. Real geometry is temporally coherent, so requiring the two
        // frames to broadly agree on the motion rejects the mismatches.
        const float motionDisagreement = length(previousMotionDisplay - motionVectorDisplay);
        const float motionMagnitude = max(length(motionVectorDisplay), 1.0f);

        if (motionDisagreement < motionMagnitude * 0.5f)
        {
          const float2 positionMinusTwo = actualMinusOne + previousMotionDisplay;

          // Lagrange through t = -2, -1, 0 evaluated at -0.5 and +0.5.
          const float2 halfBack = -0.125f * positionMinusTwo + 0.75f * actualMinusOne + 0.375f * centerPixel;
          const float2 halfForward = 0.375f * positionMinusTwo - 1.25f * actualMinusOne + 1.875f * centerPixel;
          const float2 objectCurvature = centerPixel - (halfBack + halfForward) * 0.5f;

          // An implied acceleration larger than the motion itself is never physical over a
          // single frame at these magnitudes.
          if (length(objectCurvature) < length(motionVectorDisplay) * 0.5f)
          {
            curvature = objectCurvature * curvatureScale;
          }
        }
      }
    }
  }

  if (!isValid)
  {
    // Excluded surfaces neither blur nor smear: a non zero velocity here would still feed
    // TileMax and drag the view model's motion across the world behind it.
    velocity = float2(0.0f, 0.0f);
    curvature = float2(0.0f, 0.0f);
  }
  else
  {
    // Clamp the whole trail, curvature included, to the maximum radius the tile size can
    // service. Scaling both terms together preserves the shape of the arc.
    const float trailExtent = length(velocity) + length(curvature);
    if (trailExtent > cb.maxBlurRadiusPixels && trailExtent > 1e-6f)
    {
      const float clampScale = cb.maxBlurRadiusPixels / trailExtent;
      velocity *= clampScale;
      curvature *= clampScale;
    }

    // Below the threshold the pixel is considered motionless. It keeps its validity so it
    // can still receive smear from moving neighbours and contribute colour to them, which
    // is the entire point of the tile dilation.
    if (length(velocity) < cb.minVelocityPixels)
    {
      velocity = float2(0.0f, 0.0f);
      curvature = float2(0.0f, 0.0f);
    }
  }

  VelocityDepthOutput[pixelPos] = motionBlurPackVelocityDepth(velocity, linearZ, isValid ? 1.0f : 0.0f);
  CurvatureOutput[pixelPos] = curvature;
}
