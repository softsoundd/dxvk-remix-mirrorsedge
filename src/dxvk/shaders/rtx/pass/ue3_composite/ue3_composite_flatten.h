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

// Inputs

#define UE3_COMPOSITE_FLATTEN_SOURCE_INPUT       0
#define UE3_COMPOSITE_FLATTEN_MASKED_INPUT       1
#define UE3_COMPOSITE_FLATTEN_MASKED_HALF_INPUT  2
#define UE3_COMPOSITE_FLATTEN_MASK_INPUT         3

// Outputs

#define UE3_COMPOSITE_FLATTEN_OUTPUT 10

// Is set when the source is the albedo baked with the mask at 0, which the mask blends towards the masked inputs.
// Without it, the source is only repeated.
#define UE3_COMPOSITE_FLATTEN_FLAG_MASK      (1u << 0)
// Is set when the colour is quadratic in the mask, so the blend also interpolates through the half mask input.
#define UE3_COMPOSITE_FLATTEN_FLAG_QUADRATIC (1u << 1)
// Is set when the game samples the mask as sRGB.
#define UE3_COMPOSITE_FLATTEN_FLAG_MASK_SRGB (1u << 2)

struct Ue3CompositeFlattenArgs
{
  uvec2 resolution;
  // Is how many of the bake's repeats the output covers along each axis.
  vec2 repeat;
  // Maps the bake's UV to the mask's as (scale U, scale V, offset U, offset V).
  vec4 maskTransform;
  uint flags;
  uint maskChannel;
};
