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
#pragma once

#include "rtx/utility/shader_types.h"

// Constants for the NGX passthrough motion vector / depth generation pass.
// The pass reprojects the game's hardware depth buffer through the current and previous
// frame cameras to synthesize the screen space motion vectors DLSS / DLFG consume.
struct NgxPassthroughArgs {
  // Maps unjittered current-frame NDC (x, y in [-1,1], z = hardware depth in [0,1], w = 1)
  // to previous-frame clip space:
  //   prevViewToProjection * prevWorldToView * viewToWorld * projectionToView
  mat4 reprojectToPrevClip;

  vec2 resolution;         // output resolution in pixels
  vec2 subrectOffset;      // top-left corner of the source subrect in the game depth buffer, in pixels

  uint debugMode;          // 0: off, 1: motion vectors, 2: depth, 3: object velocity coverage, 4: scene color (injection input)
  // Near/far planes for hardware depth -> linear view-space Z (Remix motion blur input)
  float nearPlane;
  float farPlane;
  // 1: the camera-locked foreground participates in motion blur (no view model flag);
  // 0: excluded like the path tracer's view model
  uint motionBlurFirstPerson;

  // 1: the object velocity texture holds this frame's rasterized world-phase NDC deltas
  // (sentinel-cleared); when both flags are 0 the texture must not be sampled
  uint objectVelocityValid;
  // Same for the foreground-phase draws sharing the texture (marked via the blue-channel
  // phase marker; consumed by camera-locked pixels only)
  uint foregroundVelocityValid;

  // UE3's FinishRenderViewTarget composite is replaced by the Super Resolution upscale, so its
  // transform is reapplied to the result instead: rgb = ColorScale, a = InverseGamma.
  uint outputTransformEnabled;
  // 1: take alpha from the game's original colour (the pre-post injection writes back over
  // scene colour, whose alpha carries UE3's depth); 0: the target alpha is free.
  uint preserveOriginalAlpha;

  // Display extent. `resolution` above is the render extent, which Super Resolution makes
  // smaller than the merged output.
  vec2 outputResolution;
  vec2 outputPad;

  vec4 outputColorScaleAndGamma;
  vec4 outputOverlayColor;

  // The sub-pixel viewport offset this frame was rasterized with, in render pixels, y down -
  // exactly what was added to the game's viewport. The depth buffer is displaced by it, so
  // reconstructing a position from a pixel coordinate has to take it back out.
  vec2 jitter;
  vec2 jitterPad;
};

// Push constants for the object velocity raster pass: clip transforms composed from the
// reconstructed camera and the disambiguated LocalToWorld, consumed with mul() exactly
// like the motion vector pass consumes its reprojection matrix.
struct NgxVelocityPushConstants {
  mat4 clipFromLocal;
  mat4 prevClipFromLocal;
};

// Constant buffer for the object velocity raster (the push constant block above is at the
// 128 byte limit): locates the scene subrect within the depth buffer for the manual depth
// compare and sizes the skinned palette blocks (per-draw layout [header][previous]
// [current], bonePaletteRegisterCount registers per palette; the header carries the
// draw's index scale and influence count).
struct NgxVelocityRasterArgs {
  vec2 subrectOffset;
  uint bonePaletteRegisterCount;
  // Diagnostic (rtx.ngxPassthrough.objectVelocityDebugFreeze): visibility is not resolved
  // against the game depth, exposing the raw rasterized footprint of the velocity draws
  uint debugSkipDepthTest;
  // 1 during the foreground-phase pass: fragments mark the velocity texture's blue
  // channel so the motion vector pass can tell foreground-owned pixels from world-owned
  // ones (both phases share one target; re-uploaded between the passes)
  uint foregroundPass;
  // Diagnostic (rtx.ngxPassthrough.objectVelocityDepthTolerance): widens the depth match so an
  // object whose velocity is being discarded can be bisected for how far off its depth actually is
  float depthToleranceScale;
  uint pad0;
  uint pad1;
};
