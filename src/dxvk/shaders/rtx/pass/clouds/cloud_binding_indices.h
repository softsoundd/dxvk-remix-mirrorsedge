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

// Bindings of the cloud compute passes. Every pass uses its own subset of this one layout, so the field and
// lighting headers can refer to the resources by name wherever they are compiled.
#define CLOUD_BINDING_CONSTANTS                  0
#define CLOUD_BINDING_CAMERA                     1
#define CLOUD_BINDING_VOLUME_SAMPLER             2   // Trilinear, repeats; non-periodic axes clamp in the shader
#define CLOUD_BINDING_LUT_SAMPLER                3   // Linear, clamps to the edge
#define CLOUD_BINDING_NVDF                       4
#define CLOUD_BINDING_DETAIL_NOISE               5
#define CLOUD_BINDING_SUN_GRID                   6
#define CLOUD_BINDING_AMBIENT_GRID               7
#define CLOUD_BINDING_DIFFUSION_GRID             8
#define CLOUD_BINDING_TRANSMITTANCE_LUT          9
#define CLOUD_BINDING_MULTISCATTERING_LUT        10
#define CLOUD_BINDING_AEROSOL_PHASE_LUT          11
#define CLOUD_BINDING_PHASE_LUT                  12
#define CLOUD_BINDING_SKY_AP_INSCATTER           13
#define CLOUD_BINDING_SKY_AP_TRANSMITTANCE       14
#define CLOUD_BINDING_SKY_SH                     15
#define CLOUD_BINDING_BLUE_NOISE                 16
#define CLOUD_BINDING_PRIMARY_LINEAR_VIEW_Z      17
#define CLOUD_BINDING_PSR_FIRST_HIT_DISTANCE     18
#define CLOUD_BINDING_PSR_REFLECTION_SEGMENT     19
#define CLOUD_BINDING_PSR_REFLECTION_DIRECTION   20
#define CLOUD_BINDING_SHARED_FLAGS               21
#define CLOUD_BINDING_LAYER_OUTPUT               22
#define CLOUD_BINDING_MOTION_VECTOR_OUTPUT       23  // The upscalers' screen space motion vectors, for the clouds' pixels
#define CLOUD_BINDING_REFLECTION_OUTPUT          24
#define CLOUD_BINDING_DEBUG_VIEW                 25
#define CLOUD_BINDING_SHADOW_MAP                 26
#define CLOUD_BINDING_HISTORY                    27
#define CLOUD_BINDING_DOME_LUT                   28
#define CLOUD_BINDING_REFERENCE_ACCUMULATION     29
#define CLOUD_BINDING_PHASE_CDF                  30
#define CLOUD_BINDING_PLACEMENT_MAP              31
#define CLOUD_BINDING_REFERENCE_STATISTICS       32
#define CLOUD_BINDING_REFERENCE_PATH_POSITION    33
#define CLOUD_BINDING_REFERENCE_PATH_DIRECTION   34
#define CLOUD_BINDING_REFERENCE_PATH_THROUGHPUT  35
#define CLOUD_BINDING_REFERENCE_PATH_RADIANCE    36

// The optical depth grids' far cascade, read by the marching passes.
#define CLOUD_BINDING_SUN_GRID_FAR               37
#define CLOUD_BINDING_AMBIENT_GRID_FAR           38
#define CLOUD_BINDING_DIFFUSION_GRID_FAR         39

// Outputs and scratch of the bakes, which never share a pass with the march's outputs above. The dome binds
// these alongside the march's inputs, so no index may repeat one of the above.
#define CLOUD_BINDING_BAKE_OUTPUT                40
#define CLOUD_BINDING_BAKE_INPUT                 41
#define CLOUD_BINDING_BAKE_INPUT2                42
#define CLOUD_BINDING_BAKE_OUTPUT2               43

// The screen pass's layer for composite: the filtered layer without its history's marks.
#define CLOUD_BINDING_COMPOSITE_LAYER_OUTPUT     44
#define CLOUD_BINDING_MOTION_VECTOR_RR_OUTPUT    45  // DLSS Ray Reconstruction's own

// Frames the screen pass's layer history holds, read and written beside the history.
#define CLOUD_BINDING_HISTORY_AGE                46
#define CLOUD_BINDING_HISTORY_AGE_OUTPUT         47

// Near-mirror reflections into the sky (cloud_glossy.comp.slang): the indirect integrator's rays, the sky it
// composited their clouds over, and its radiance they are replaced in.
#define CLOUD_BINDING_GLOSSY_RAY                 48
#define CLOUD_BINDING_SKY_VIEW_LUT               49
#define CLOUD_BINDING_INDIRECT_RADIANCE          50

// The screen pass's layer along mirrors' and glass's reflections a frame ago.
#define CLOUD_BINDING_REFLECTION_HISTORY         51

// Sparse rendering's map from a pixel to the compacted slot that holds its secondary GBuffer and indirect radiance.
#define CLOUD_BINDING_COMPACTED_PIXEL_INDICES    52
#define CLOUD_BINDING_TILE_ACTIVE_COUNTS         53
