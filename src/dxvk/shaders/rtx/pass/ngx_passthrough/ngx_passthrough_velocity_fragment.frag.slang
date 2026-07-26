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

// NGX passthrough object velocity pass, fragment stage: writes the NDC motion delta for
// visible fragments. There is no depth attachment; visibility is resolved manually against
// the game's depth for the draw's phase (world snapshot or live foreground buffer). The
// raster replicates the game's viewport exactly (flip, half-pixel correction, jitter), so
// the depth tolerance only absorbs float differences between the game's two-step vertex
// transform and the precomposed clip matrix used here.

#include "rtx/pass/ngx_passthrough/ngx_passthrough_args.h"

layout(binding = 0)
Texture2D<float> WorldDepth;

layout(binding = 1)
ConstantBuffer<NgxVelocityRasterArgs> cb;

[shader("fragment")]
float4 main(float4 svPosition : SV_Position,
            float4 currClip : TEXCOORD0,
            float4 prevClip : TEXCOORD1) : SV_Target0 {
  // The raster covers the scene subrect at origin; the game depth holds it at the subrect
  // offset (Super Resolution renders the scene into a subrect of a full-size buffer)
  const float sceneDepth = WorldDepth[int2(svPosition.xy) + int2(cb.subrectOffset)];

  // Phase ownership marker (blue channel): both scene phases raster into one target, and
  // the motion vector pass consumes by phase - foreground-marked velocity is proof the
  // first person mesh owns the pixel, world-marked velocity is excluded from camera-locked
  // pixels (an occluded mover matching the world depth BEHIND the first person meshes
  // must not leak its motion onto them)
  const float phaseMarker = cb.foregroundPass != 0 ? 1.0f : 0.0f;

  // Two-sided visibility: a pixel owns this velocity only when the rasterized depth
  // matches the surface the game shows there. Behind = occluded; in front is equally
  // wrong - a mesh whose depth lives in the other scene phase (first person meshes
  // shuffled between DPGs) or a fragment over an alpha-tested hole would otherwise
  // splat its motion onto whatever is actually visible at the pixel.
  //
  // The tolerance is relative: standard hyperbolic depth has (1 - z) proportional to
  // 1/viewZ, so |dz| <= t * (1 - z) accepts a constant fraction of the view distance.
  // A fixed epsilon is simultaneously too tight near the camera and too loose at range
  // (depth compression lets objects many meters behind a distant wall "match" it,
  // seeping their motion through walls). The raster replays the game's packed clip
  // transform verbatim, so the residual error is float rounding - 0.1% of view distance
  // absorbs it with a wide margin while keeping occlusion strict.
  const float depthTolerance = max(1e-3f * (1.0f - sceneDepth), 1e-6f) * cb.depthToleranceScale;

  if (cb.debugSkipDepthTest == 0 && abs(svPosition.z - sceneDepth) > depthTolerance) {
    discard;
  }

  // A previous position behind the camera has no meaningful screen projection: report zero
  // motion rather than a wrapped-around delta
  if (prevClip.w <= 1e-6f) {
    return float4(0.0f, 0.0f, phaseMarker, 1.0f);
  }

  // Per-pixel projective divide of the perspective-correct interpolated clip positions;
  // previous - current, the camera reprojection convention of the motion vector pass.
  // Clamped to a few screens: fragments approaching the previous-frame camera plane
  // (w -> 0) would otherwise blow past the 16-bit float target's range
  const float2 ndcDelta = clamp(prevClip.xy / prevClip.w - currClip.xy / currClip.w, -8.0f, 8.0f);
  return float4(ndcDelta.x, ndcDelta.y, phaseMarker, 1.0f);
}
