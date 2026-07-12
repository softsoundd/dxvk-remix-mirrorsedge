/*
* Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
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

// Bindings

// Inputs

#define DEBUG_VIEW_BINDING_CONSTANTS_INPUT                                                 1

#define DEBUG_VIEW_BINDING_DENOISED_PRIMARY_DIRECT_DIFFUSE_RADIANCE_HIT_T_INPUT            3
#define DEBUG_VIEW_BINDING_DENOISED_PRIMARY_DIRECT_SPECULAR_RADIANCE_HIT_T_INPUT           4
#define DEBUG_VIEW_BINDING_DENOISED_SECONDARY_COMBINED_DIFFUSE_RADIANCE_HIT_T_INPUT        5
#define DEBUG_VIEW_BINDING_DENOISED_SECONDARY_COMBINED_SPECULAR_RADIANCE_HIT_T_INPUT       6
#define DEBUG_VIEW_BINDING_SHARED_FLAGS_INPUT                                              7
#define DEBUG_VIEW_BINDING_PRIMARY_LINEAR_VIEW_Z_INPUT                                     8
#define DEBUG_VIEW_BINDING_PRIMARY_VIRTUAL_WORLD_SHADING_NORMAL_PERCEPTUAL_ROUGHNESS_INPUT 9

#define DEBUG_VIEW_BINDING_PRIMARY_SCREEN_SPACE_MOTION_VECTOR_INPUT                        11
#define DEBUG_VIEW_BINDING_RTXDI_CONFIDENCE_INPUT                                          12
#define DEBUG_VIEW_BINDING_RENDER_OUTPUT_INPUT                                             13

#define DEBUG_VIEW_BINDING_INSTRUMENTATION_INPUT                                           15
#define DEBUG_VIEW_BINDING_TERRAIN_INPUT                                                   17

// Slot 34 is upstream's DEBUG_VIEW_BINDING_SHARED_TERMINATOR_FIX_INPUT. The fork
// cloud-debug bindings below were moved out of 34-37 to 39-42 on the 2026-06-21
// upstream sync to avoid colliding with it; they live above the reserved slot 38
// as a contiguous fork-owned block.

// Fork: per-frame cloud-occluded sky-ambient transmittance LUT (32x16 R16F)
#define DEBUG_VIEW_BINDING_CLOUD_SKY_TRANSMITTANCE_LUT_INPUT                                39

// Fork: Nubis Cubed cloud voxel grids (D_sun = sun-direction optical depth,
// D_ambient = zenith optical depth). Sampled by DEBUG_VIEW_CLOUD_D_SUN /
// DEBUG_VIEW_CLOUD_D_AMBIENT debug views.
#define DEBUG_VIEW_BINDING_CLOUD_D_SUN_INPUT                                                40
#define DEBUG_VIEW_BINDING_CLOUD_D_AMBIENT_INPUT                                            41

// Fork: Nubis Cubed screen-space cloud render RT (2026-05-12, C4). Sampled
// by DEBUG_VIEW_CLOUD_RENDER_RT (enum 876).
#define DEBUG_VIEW_BINDING_CLOUD_RENDER_RT_INPUT                                            42

// Slot 38 was DEBUG_VIEW_BINDING_PRIMARY_CLOUD_SHADOW_FACTOR_INPUT (fork
// screen-space cloud-shadow texture, debug view 878). Removed 2026-06-19 with
// the screen-space cloud-shadow system; number left reserved (no descriptor
// bound).

#define DEBUG_VIEW_BINDING_VOLUME_RESERVOIRS_INPUT                                         19
#define DEBUG_VIEW_BINDING_VOLUME_AGE_INPUT                                                20
#define DEBUG_VIEW_BINDING_VOLUME_RADIANCE_Y_INPUT                                         21
#define DEBUG_VIEW_BINDING_VOLUME_RADIANCE_COCG_INPUT                                      22
#define DEBUG_VIEW_BINDING_VALUE_NOISE_SAMPLER                                             23
#define DEBUG_VIEW_BINDING_BLUE_NOISE_TEXTURE                                              24

#define DEBUG_VIEW_BINDING_DEBUG_VIEW_INPUT                                                28

#define DEBUG_VIEW_BINDING_NRD_VALIDATION_LAYER_INPUT                                      30
#define DEBUG_VIEW_BINDING_COMPOSITE_INPUT                                                 31

#define DEBUG_VIEW_BINDING_ALTERNATE_DISOCCLUSION_THRESHOLD_INPUT                          32

#define DEBUG_VIEW_BINDING_PREV_WORLD_POSITION_INPUT                                       33
#define DEBUG_VIEW_BINDING_SHARED_TERMINATOR_FIX_INPUT                                     34

// Inputs / Outputs

#define DEBUG_VIEW_BINDING_ACCUMULATED_DEBUG_VIEW_INPUT_OUTPUT                             60 

// Outputs

#define DEBUG_VIEW_BINDING_STATISTICS_BUFFER_OUTPUT                                        90

// Samplers

#define DEBUG_VIEW_BINDING_NEAREST_SAMPLER                                                 300
#define DEBUG_VIEW_BINDING_LINEAR_SAMPLER                                                  301
