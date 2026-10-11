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
#include "rtx/pass/common_binding_indices.h"

// The image probe, kept as a debug source: it measures the disc in the rendered image.

#define SUN_PROBE_COLOR_INPUT       0
#define SUN_PROBE_VISIBILITY_OUTPUT 20

// Half the threads sample the disc, the other half the annulus around it.
#define SUN_PROBE_GROUP_SIZE 256

struct SunProbeArgs {
  vec2 sunPixel;
  float discRadiusPixels;
  float annulusInnerRadiusPixels;

  // Unclamped radiance at the centre of the disc, per channel.
  vec3 centerRadiance;
  float maxDiscRadiance;

  vec3 limbDarkeningExponent;
  uint pad0;

  uvec2 imageSize;
  uvec2 pad1;
};

// The ray traced visibility, through the common ray tracing bindings: one workgroup, a ray per thread.

#define SUN_VISIBILITY_OUTPUT 40

#if SUN_VISIBILITY_OUTPUT <= COMMON_MAX_BINDING
#error "Increase the base index of the sun visibility bindings to avoid overlap with common bindings!"
#endif

// One ray per stratum of a 16 by 16 grid over the disc.
#define SUN_VISIBILITY_RAYS 256

#define SUN_VISIBILITY_FLAG_CLOUDS (1 << 0)
#define SUN_VISIBILITY_FLAG_FOG    (1 << 1)

struct SunVisibilityArgs {
  // In world units, the camera's position and the sun's direction, with two directions across the disc whose
  // lengths are the tangent of its angular radius.
  vec3 origin;
  float tMin;

  vec3 sunDirection;
  float tMax;

  vec3 discRight;
  uint flags;

  vec3 discUp;
  uint pad0;

  vec3 limbDarkeningExponent;
  uint pad1;
};
