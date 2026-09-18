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


// SHARC variants
//!variant integrate_indirect_sharc_update.comp
//!>       ENABLE_SHARC=1
//!>       SHARC_UPDATE=1
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_sharc_update_deferred.comp
//!>       ENABLE_SHARC=1
//!>       SHARC_UPDATE=1
//!>       SHARC_DEFERRED_UPDATE=1
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_sharc_update_deferred4.comp
//!>       ENABLE_SHARC=1
//!>       SHARC_UPDATE=1
//!>       SHARC_DEFERRED_UPDATE=1
//!>       SHARC_PROPAGATION_DEPTH=4
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_sharc_update_raygen.rgen
//!>       RAY_PIPELINE
//!>       ENABLE_SHARC=1
//!>       SHARC_UPDATE=1
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_sharc_update_deferred_raygen.rgen
//!>       RAY_PIPELINE
//!>       ENABLE_SHARC=1
//!>       SHARC_UPDATE=1
//!>       SHARC_DEFERRED_UPDATE=1
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_sharc_update_deferred4_raygen.rgen
//!>       RAY_PIPELINE
//!>       ENABLE_SHARC=1
//!>       SHARC_UPDATE=1
//!>       SHARC_DEFERRED_UPDATE=1
//!>       SHARC_PROPAGATION_DEPTH=4
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_sharc_query.comp
//!>       ENABLE_SHARC=1
//!>       SHARC_QUERY=1
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_sharc_query_stats.comp
//!>       ENABLE_SHARC=1
//!>       SHARC_QUERY=1
//!>       SHARC_QUERY_STATS=1
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_sharc_query_raygen.rgen
//!>       RAY_PIPELINE
//!>       ENABLE_SHARC=1
//!>       SHARC_QUERY=1
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_sharc_query_raygen_stats.rgen
//!>       RAY_PIPELINE
//!>       ENABLE_SHARC=1
//!>       SHARC_QUERY=1
//!>       SHARC_QUERY_STATS=1
//!>       NEE_CACHE_ENABLE=1

//!variant integrate_indirect_sharc_update_wboit.comp
//!>       ENABLE_SHARC=1
//!>       SHARC_UPDATE=1
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_sharc_update_deferred_wboit.comp
//!>       ENABLE_SHARC=1
//!>       SHARC_UPDATE=1
//!>       SHARC_DEFERRED_UPDATE=1
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_sharc_update_deferred4_wboit.comp
//!>       ENABLE_SHARC=1
//!>       SHARC_UPDATE=1
//!>       SHARC_DEFERRED_UPDATE=1
//!>       SHARC_PROPAGATION_DEPTH=4
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_sharc_update_raygen_wboit.rgen
//!>       RAY_PIPELINE
//!>       ENABLE_SHARC=1
//!>       SHARC_UPDATE=1
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_sharc_update_deferred_raygen_wboit.rgen
//!>       RAY_PIPELINE
//!>       ENABLE_SHARC=1
//!>       SHARC_UPDATE=1
//!>       SHARC_DEFERRED_UPDATE=1
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_sharc_update_deferred4_raygen_wboit.rgen
//!>       RAY_PIPELINE
//!>       ENABLE_SHARC=1
//!>       SHARC_UPDATE=1
//!>       SHARC_DEFERRED_UPDATE=1
//!>       SHARC_PROPAGATION_DEPTH=4
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_sharc_query_wboit.comp
//!>       ENABLE_SHARC=1
//!>       SHARC_QUERY=1
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_sharc_query_stats_wboit.comp
//!>       ENABLE_SHARC=1
//!>       SHARC_QUERY=1
//!>       SHARC_QUERY_STATS=1
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_sharc_query_raygen_wboit.rgen
//!>       RAY_PIPELINE
//!>       ENABLE_SHARC=1
//!>       SHARC_QUERY=1
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_sharc_query_raygen_stats_wboit.rgen
//!>       RAY_PIPELINE
//!>       ENABLE_SHARC=1
//!>       SHARC_QUERY=1
//!>       SHARC_QUERY_STATS=1
//!>       NEE_CACHE_ENABLE=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_sharc_query_trace.rgen
//!>       ENABLE_SHARC=1
//!>       SHARC_QUERY=1
//!>       NEE_CACHE_ENABLE=1
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE

//!variant integrate_indirect_sharc_query_trace_wboit.rgen
//!>       ENABLE_SHARC=1
//!>       SHARC_QUERY=1
//!>       NEE_CACHE_ENABLE=1
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_sharc_query_trace_stats.rgen
//!>       ENABLE_SHARC=1
//!>       SHARC_QUERY=1
//!>       NEE_CACHE_ENABLE=1
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       SHARC_QUERY_STATS=1

//!variant integrate_indirect_sharc_query_trace_stats_wboit.rgen
//!>       ENABLE_SHARC=1
//!>       SHARC_QUERY=1
//!>       NEE_CACHE_ENABLE=1
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       SHARC_QUERY_STATS=1
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_sharc_query_trace_ser.rgen
//!>       ENABLE_SHARC=1
//!>       SHARC_QUERY=1
//!>       NEE_CACHE_ENABLE=1
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING

//!variant integrate_indirect_sharc_query_trace_ser_wboit.rgen
//!>       ENABLE_SHARC=1
//!>       SHARC_QUERY=1
//!>       NEE_CACHE_ENABLE=1
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING
//!>       UNORDERED_RESOLVE_WBOIT

//!variant integrate_indirect_sharc_query_trace_ser_stats.rgen
//!>       ENABLE_SHARC=1
//!>       SHARC_QUERY=1
//!>       NEE_CACHE_ENABLE=1
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING
//!>       SHARC_QUERY_STATS=1

//!variant integrate_indirect_sharc_query_trace_ser_stats_wboit.rgen
//!>       ENABLE_SHARC=1
//!>       SHARC_QUERY=1
//!>       NEE_CACHE_ENABLE=1
//!>       RT_TRACE_RAY
//!>       RAY_PIPELINE
//!>       RT_SHADER_EXECUTION_REORDERING
//!>       SHARC_QUERY_STATS=1
//!>       UNORDERED_RESOLVE_WBOIT

//!end-variants

#ifndef ENABLE_SHARC
#define ENABLE_SHARC 0
#endif
#ifndef SHARC_QUERY
#define SHARC_QUERY 0
#endif

#define INTEGRATE_INDIRECT
#define RAB_HAS_PREVIOUS_GBUFFER 1
#define OPAQUE_MATERIAL_USE_POM 1
// The compute and RayQuery ray-generation backends run integratePathVertex inline, so they
// need the same reservoir access as the TraceRay closest-hit stage. Without these defines
// RAB_LoadReservoir / RAB_LoadGIReservoir return empty reservoirs: an RTXDI steal at a
// camera-visible secondary vertex then fails and skips the NEE cache / RIS fallback (no
// shadow ray, no direct light at that vertex), and a ReSTIR GI steal in StealSample mode
// adds zero radiance and still terminates the path on rough surfaces. SHARC stages keep
// the inline policy of never touching the reservoirs; see integrator_indirect.slangh.
#if !ENABLE_SHARC
#define RAB_HAS_RTXDI_RESERVOIRS 1
#define RAB_HAS_RESTIR_GI_RESERVOIRS 1
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
#if SHARC_QUERY
[numthreads(8, 8, 1)]
#else
[numthreads(16, 8, 1)]
#endif
void main(uint2 dispatchThreadId : SV_DispatchThreadID)
{
  integrate_indirect_pass(dispatchThreadId);
}
#endif