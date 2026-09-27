/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
#ifndef AERIAL_PERSPECTIVE_BINDING_INDICES_H
#define AERIAL_PERSPECTIVE_BINDING_INDICES_H

#include "rtx/pass/common_binding_indices.h"

// Ray traced aerial perspective volume pass. Uses the common ray tracing bindings for the scene, the
// constants and the atmosphere LUTs, plus the volumes and tile bounds below.

// Inputs

#define AERIAL_PERSPECTIVE_BINDING_PREV_LUT_INPUT        40
#define AERIAL_PERSPECTIVE_BINDING_TILE_DEPTH_INPUT      41
#define AERIAL_PERSPECTIVE_BINDING_PREV_TILE_DEPTH_INPUT 42

// Outputs

#define AERIAL_PERSPECTIVE_BINDING_LUT_OUTPUT            43

#define AERIAL_PERSPECTIVE_MIN_BINDING              AERIAL_PERSPECTIVE_BINDING_PREV_LUT_INPUT
#define AERIAL_PERSPECTIVE_MAX_BINDING              AERIAL_PERSPECTIVE_BINDING_LUT_OUTPUT

#if AERIAL_PERSPECTIVE_MIN_BINDING <= COMMON_MAX_BINDING
#error "Increase the base index of Aerial Perspective bindings to avoid overlap with common bindings!"
#endif

#endif
