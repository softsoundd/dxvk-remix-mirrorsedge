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

#define UE3_COMPOSITE_ENCODE_PASS_INPUT     0
#define UE3_COMPOSITE_ENCODE_EMISSION_INPUT 1

// Outputs

#define UE3_COMPOSITE_ENCODE_ALBEDO_OUTPUT   10
#define UE3_COMPOSITE_ENCODE_EMISSION_OUTPUT 11

// The emission pass is bound, and its emission is taken out of the albedo.
#define UE3_COMPOSITE_ENCODE_FLAG_EMISSION_PASS  (1u << 0)
// The pass's alpha marks the texels the material did not clip.
#define UE3_COMPOSITE_ENCODE_FLAG_COVERAGE_ALPHA (1u << 1)
// The emission is written out as well.
#define UE3_COMPOSITE_ENCODE_FLAG_STORE_EMISSION (1u << 2)
// The emission pass and output are one texel, which holds the emission everywhere.
#define UE3_COMPOSITE_ENCODE_FLAG_UNIFORM_EMISSION (1u << 3)

struct Ue3CompositeEncodeArgs
{
  uvec2 resolution;
  uint flags;
  uint padding;
};
