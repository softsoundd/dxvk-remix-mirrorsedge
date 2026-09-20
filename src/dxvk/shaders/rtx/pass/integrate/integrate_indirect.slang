/*
* Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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

//!variant integrate_indirect_raygen_neeCache.rgen
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_raygen_ser_neeCache.rgen
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_rayquery_neeCache.comp
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_rayquery_raygen_neeCache.rgen
//!>       RAY_PIPELINE
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_rayquery.comp
//!>       NEE_CACHE_ENABLE=0

//!variant integrate_indirect_rayquery_raygen.rgen
//!>       RAY_PIPELINE
//!>       NEE_CACHE_ENABLE=0

//!variant integrate_indirect_raygen.rgen
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       NEE_CACHE_ENABLE=0

//!variant integrate_indirect_raygen_ser.rgen
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING
//!>       NEE_CACHE_ENABLE=0


/* NRC */

//!variant integrate_indirect_raygen_nrc_neeCache.rgen
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       ENABLE_NRC=1
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_raygen_ser_nrc_neeCache.rgen
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING
//!>       ENABLE_NRC=1
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_rayquery_nrc_neeCache.comp
//!>       ENABLE_NRC=1
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_rayquery_raygen_nrc_neeCache.rgen
//!>       RAY_PIPELINE
//!>       ENABLE_NRC=1
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_rayquery_nrc.comp
//!>       ENABLE_NRC=1
//!>       NEE_CACHE_ENABLE=0

//!variant integrate_indirect_rayquery_raygen_nrc.rgen
//!>       RAY_PIPELINE
//!>       ENABLE_NRC=1
//!>       NEE_CACHE_ENABLE=0

//!variant integrate_indirect_raygen_nrc.rgen
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       ENABLE_NRC=1
//!>       NEE_CACHE_ENABLE=0

//!variant integrate_indirect_raygen_ser_nrc.rgen
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING
//!>       ENABLE_NRC=1
//!>       NEE_CACHE_ENABLE=0


// WBOIT


//!variant integrate_indirect_raygen_neeCache_wboit.rgen
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_raygen_ser_neeCache_wboit.rgen
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_rayquery_neeCache_wboit.comp
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_rayquery_raygen_neeCache_wboit.rgen
//!>       RAY_PIPELINE
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_rayquery_wboit.comp
//!>       NEE_CACHE_ENABLE=0
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_rayquery_raygen_wboit.rgen
//!>       RAY_PIPELINE
//!>       NEE_CACHE_ENABLE=0
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_raygen_wboit.rgen
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       NEE_CACHE_ENABLE=0
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_raygen_ser_wboit.rgen
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING
//!>       NEE_CACHE_ENABLE=0
//!>       UNORDERED_RESOLVE_WBOIT

/* NRC */

//!variant integrate_indirect_raygen_nrc_neeCache_wboit.rgen
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       ENABLE_NRC=1
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_raygen_ser_nrc_neeCache_wboit.rgen
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING
//!>       ENABLE_NRC=1
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_rayquery_nrc_neeCache_wboit.comp
//!>       ENABLE_NRC=1
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_rayquery_raygen_nrc_neeCache_wboit.rgen
//!>       RAY_PIPELINE
//!>       ENABLE_NRC=1
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_rayquery_nrc_wboit.comp
//!>       ENABLE_NRC=1
//!>       NEE_CACHE_ENABLE=0
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_rayquery_raygen_nrc_wboit.rgen
//!>       RAY_PIPELINE
//!>       ENABLE_NRC=1
//!>       NEE_CACHE_ENABLE=0
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_raygen_nrc_wboit.rgen
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       ENABLE_NRC=1
//!>       NEE_CACHE_ENABLE=0
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_raygen_ser_nrc_wboit.rgen
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING
//!>       ENABLE_NRC=1
//!>       NEE_CACHE_ENABLE=0
//!>       UNORDERED_RESOLVE_WBOIT


/* SHARC */
// Dedicated variants keep SHARC descriptors and payload state out of every other shader.
// The update stage traces inline RayQuery from a ray generation shader; the query stage uses
// the same TraceRay hit and miss shaders as the ordinary indirect pass.

//!variant integrate_indirect_sharc_update4.rgen
//!>       RAY_PIPELINE
//!>       ENABLE_SHARC=1
//!>       SHARC_UPDATE=1
//!>       SHARC_PROPAGATION_DEPTH=4
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_sharc_update4_wboit.rgen
//!>       RAY_PIPELINE
//!>       ENABLE_SHARC=1
//!>       SHARC_UPDATE=1
//!>       SHARC_PROPAGATION_DEPTH=4
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_sharc_update8.rgen
//!>       RAY_PIPELINE
//!>       ENABLE_SHARC=1
//!>       SHARC_UPDATE=1
//!>       SHARC_PROPAGATION_DEPTH=8
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_sharc_update8_wboit.rgen
//!>       RAY_PIPELINE
//!>       ENABLE_SHARC=1
//!>       SHARC_UPDATE=1
//!>       SHARC_PROPAGATION_DEPTH=8
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_sharc_query.rgen
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       ENABLE_SHARC=1
//!>       SHARC_QUERY=1
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_sharc_query_wboit.rgen
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       ENABLE_SHARC=1
//!>       SHARC_QUERY=1
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_sharc_query_ser.rgen
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING
//!>       ENABLE_SHARC=1
//!>       SHARC_QUERY=1
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_sharc_query_ser_wboit.rgen
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING
//!>       ENABLE_SHARC=1
//!>       SHARC_QUERY=1
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!end-variants

#define INTEGRATE_INDIRECT
#define RAB_HAS_PREVIOUS_GBUFFER 1
#define OPAQUE_MATERIAL_USE_POM 1
#if ENABLE_SHARC
// The cache update stage depends on secondary NEE sample stealing: an update path that gathers
// only what a brute-force secondary bounce finds deposits a far noisier value into a cell, and a
// cell is a mean, so that noise does not average out per pixel the way a path tracer's does.
#define RAB_HAS_RTXDI_RESERVOIRS 1
#endif

// Note: Opacity lighting approximation used in the indirect integration pass due to particle lighting being somewhat
// important in reflections off shiny surfaces (as without this it can make areas with many particles look weird in reflections).
#define RESOLVE_OPACITY_LIGHTING_APPROXIMATION 1
#define RESOLVER_USE_VOLUMETRIC_ATTENUATION 1

#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/integrate/integrate_indirect_binding_indices.h"
#include "rtx/pass/integrate/integrate_indirect_bindings.slangh"
// Note: Needed for declaration of integration ray payload structure
#include "rtx/algorithm/integrator.slangh"

#include "rtx/pass/integrate/integrate_indirect.slangh"

#ifdef RAY_PIPELINE
[shader("raygeneration")]
void main()
{
  integrate_indirect_pass(DispatchRaysIndex().xy);
}
#else
[shader("compute")]
[numthreads(16, 8, 1)]
void main(uint2 dispatchThreadId : SV_DispatchThreadID)
{
  integrate_indirect_pass(dispatchThreadId);
}
#endif
