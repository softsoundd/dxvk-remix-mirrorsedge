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

// NGX passthrough mode: synthesizes DLSS inputs from the game's own rasterized output.
// Merges the game's hardware depth from two sources into an R32F texture and derives
// screen space motion vectors: camera reprojection through the previous frame's camera,
// overridden with true object motion wherever the velocity raster captured a dynamic
// object (rigid movers, GPU-skinned characters, CPU-modified meshes).
//
// Two depth sources: UE3 clears the depth buffer mid-scene ahead of its foreground DPG
// (camera-locked first person meshes), so SceneDepth is the first-clear snapshot (world +
// intermediate DPGs) and ForegroundDepth the live buffer at the injection point (foreground
// DPG only, far-plane elsewhere). The merged depth takes the nearer value; pixels the
// foreground wins are camera-locked content, which falls back to camera reprojection at its
// own depth when no foreground-phase object velocity covers it. Note the depth comparison
// alone cannot classify pixels where both sources hold the same surface (the game may
// render the first person meshes in an intermediate DPG at the same depth as their
// foreground copy) - which is why velocity consumption trusts the raster's phase marker,
// not this classification. When no mid-scene clear occurred, both bindings alias the same
// buffer and the merge is an identity. One output pair serves DLSS Super Resolution and
// Frame Generation alike.

#include "rtx/pass/ngx_passthrough/ngx_passthrough_args.h"
#include "rtx/pass/post_fx/post_fx_motion_blur_geometry_flags.slangh"

layout(binding = 0)
ConstantBuffer<NgxPassthroughArgs> cb;

layout(binding = 1)
Texture2D<float> SceneDepth;

layout(binding = 2, r32f)
RWTexture2D<float> OutDepth;

layout(binding = 3, rg16f)
RWTexture2D<float2> OutMotionVector;

layout(binding = 4, rgba16f)
RWTexture2D<float4> DebugOutput;

layout(binding = 5)
Texture2D<float> ForegroundDepth;

// Remix post-processing (motion blur) inputs: linear view-space Z and the motion blur
// surface flags mask. The camera-locked foreground maps onto the view model flag, which
// excludes it from blur exactly like the path tracer's first person exclusion.
layout(binding = 8, r32f)
RWTexture2D<float> OutLinearViewZ;

layout(binding = 9, r8ui)
RWTexture2D<uint> OutSurfaceFlags;

// Rasterized per-object NDC motion deltas for dynamic objects (movers, doors, elevators,
// characters, first person meshes), sentinel-cleared. RG = NDC delta, B = phase ownership
// marker (0 = world/intermediate DPG draw tested against the world depth, 1 = foreground
// DPG draw tested against the live foreground depth; the foreground pass renders last and
// wins overlapping pixels). World pixels consume world-owned velocity, camera-locked
// pixels foreground-owned velocity - cross-phase samples are ignored so an occluded mover
// behind the first person meshes cannot leak motion onto them, and vice versa.
layout(binding = 10)
Texture2D<float4> ObjectVelocity;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 pixelCoordinate : SV_DispatchThreadID) {
  if (pixelCoordinate.x >= uint(cb.resolution.x) || pixelCoordinate.y >= uint(cb.resolution.y)) {
    return;
  }

  const uint2 sourceCoordinate = pixelCoordinate + uint2(cb.subrectOffset);

  const float sceneDepth = SceneDepth[sourceCoordinate];
  const float foregroundDepth = ForegroundDepth[sourceCoordinate];

  const float depth = min(sceneDepth, foregroundDepth);
  const bool isCameraLocked = foregroundDepth < sceneDepth;

  OutDepth[pixelCoordinate] = depth;

  // Remix post-processing input: linear view-space Z from the hardware depth (standard
  // D3D projection; depth 1.0 lands exactly on the far plane). The surface flags are
  // written below, after the velocity overrides resolve the first person coverage.
  OutLinearViewZ[pixelCoordinate] =
    (cb.nearPlane * cb.farPlane) / max(cb.farPlane - depth * (cb.farPlane - cb.nearPlane), 1e-6f);

  // Unjittered NDC for the pixel center (y up), matching cameraPixelCoordinateToNDC. The depth
  // here was rasterized through a viewport displaced by cb.jitter, so the surface this pixel
  // holds sits at the unjittered position (pixel center - jitter); pairing jittered depth with
  // the unjittered matrices below would skew every motion vector by the frame's jitter.
  const float2 uv = (float2(pixelCoordinate) + 0.5f - cb.jitter) / cb.resolution;
  const float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);

  // Reproject through the previous frame camera. Far plane depth (sky) still reprojects
  // correctly under camera rotation, which dominates its apparent motion. Same convention
  // as the path traced G-buffer resolver: current -> previous delta, NDC flipped into
  // y-down screen space and scaled to absolute pixels. Camera-locked foreground pixels
  // reproject at their own depth - the same treatment the meshes receive when the game
  // renders them pre-clear in an intermediate DPG, so they respond to camera motion
  // identically in every movestate.
  float2 motionVectorPixels = float2(0.0f, 0.0f);

  const float reprojectionDepth = isCameraLocked ? depth : sceneDepth;
  const float4 prevClip = mul(cb.reprojectToPrevClip, float4(ndc.x, ndc.y, reprojectionDepth, 1.0f));

  if (abs(prevClip.w) > 1e-6f) {
    const float2 prevNdc = prevClip.xy / prevClip.w;
    motionVectorPixels = (prevNdc - ndc) * float2(0.5f, -0.5f) * cb.resolution;
  }

  // Per-object velocity override: where a dynamic object was rasterized this frame, its
  // true motion (camera + object + skinning combined) replaces the camera reprojection.
  // Foreground-marked velocity (blue channel) is consumed unconditionally: the raster
  // depth-tested it against the live foreground depth, which proves the first person
  // mesh owns the pixel - the depth-merge classification cannot (the foreground DPG may
  // render at the same depth as the meshes' own intermediate-DPG copy in the snapshot,
  // e.g. full-range dual renders with occlusion culling enabled). World-marked velocity
  // keeps the camera-locked exclusion: a mover occluded behind the first person meshes
  // must not leak motion onto them.
  bool objectOverride = false;
  bool foregroundOverride = false;

  if ((cb.objectVelocityValid != 0 || cb.foregroundVelocityValid != 0)) {
    const float4 objectSample = ObjectVelocity[pixelCoordinate];

    if (objectSample.x > -50.0f) {
      const bool sampleIsForeground = objectSample.z > 0.5f;

      if (sampleIsForeground) {
        motionVectorPixels = objectSample.xy * float2(0.5f, -0.5f) * cb.resolution;
        foregroundOverride = true;
      } else if (!isCameraLocked) {
        motionVectorPixels = objectSample.xy * float2(0.5f, -0.5f) * cb.resolution;
        objectOverride = true;
      }
    }
  }

  OutMotionVector[pixelCoordinate] = motionVectorPixels;

  // Remix motion blur surface flags. First person pixels map onto the view model flag
  // (excluded from blur like the path tracer's first person exclusion) unless
  // cb.motionBlurFirstPerson opts them into world-style blur (vanilla UE3 camera motion
  // blur smears the meshes the same way). Everything counts as static geometry (the
  // velocity override already carries true object motion). Foreground-marked velocity
  // coverage classifies the meshes robustly where the depth comparison cannot (equal-
  // depth dual renders); camera-locked classification covers the uncaptured remainder.
  const bool firstPersonPixel = foregroundOverride || isCameraLocked;

  MotionBlurSurfaceFlags surfaceFlags;
  surfaceFlags.isViewModel = cb.motionBlurFirstPerson != 0 ? false : firstPersonPixel;
  surfaceFlags.isStatic = true;
  surfaceFlags.isEmissive = false;
  surfaceFlags.isMaskOut = false;
  motionBlurSurfaceFlagsWriteToGBuffer(surfaceFlags, int2(pixelCoordinate), OutSurfaceFlags);

  // Interpretable debug visualizations (the raw outputs saturate a plain color view):
  // motion vectors are centered at neutral gray and scaled so +-16 pixels spans the range,
  // hardware depth (hyperbolic, ~0.99+ past a few meters) is remapped on a log scale.
  // The blue channel encodes the pixel's motion source independently of magnitude:
  // 0 = per-object velocity override (dynamic object coverage), 0.25 = foreground object
  // velocity (first person meshes), 0.5 = world camera reprojection, 1 = camera-locked
  // foreground without object velocity.
  const float motionSourceBlue = objectOverride ? 0.0f
                               : (foregroundOverride ? 0.25f : (isCameraLocked ? 1.0f : 0.5f));

  if (cb.debugMode == 1) {
    DebugOutput[pixelCoordinate] = float4(
      saturate(motionVectorPixels.x / 32.0f + 0.5f),
      saturate(motionVectorPixels.y / 32.0f + 0.5f),
      motionSourceBlue,
      1.0f);
  } else if (cb.debugMode == 2) {
    const float logFar = -log2(max(1.0f - depth, exp2(-24.0f))) / 24.0f;
    DebugOutput[pixelCoordinate] = float4(1.0f - logFar, 1.0f - logFar, 1.0f - logFar, 1.0f);
  } else if (cb.debugMode == 3) {
    // Object velocity coverage: the raw velocity raster output. Green = world-phase
    // coverage, red = foreground-phase coverage (first person meshes), dark = no
    // rasterized velocity; the blue channel carries the camera-locked classification
    // for spatial reference.
    const float4 rawVelocity = (cb.objectVelocityValid != 0 || cb.foregroundVelocityValid != 0)
      ? ObjectVelocity[pixelCoordinate] : float4(-100.0f, -100.0f, 0.0f, 0.0f);
    const bool written = rawVelocity.x > -50.0f;
    const bool writtenForeground = written && rawVelocity.z > 0.5f;
    DebugOutput[pixelCoordinate] = float4(
      writtenForeground ? max(saturate(length(rawVelocity.xy) * 16.0f), 0.4f) : 0.0f,
      (written && !writtenForeground) ? 1.0f : 0.0f,
      isCameraLocked ? 0.5f : 0.25f,
      1.0f);
  }
}
