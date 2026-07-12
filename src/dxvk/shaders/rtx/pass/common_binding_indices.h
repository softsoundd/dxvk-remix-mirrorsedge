/*
* Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

// These are set indices - not bindings
#define BINDING_SET_BINDLESS_RAW_BUFFER          1
#define BINDING_SET_BINDLESS_TEXTURE2D           2
#define BINDING_SET_BINDLESS_SAMPLER             3


#define BINDING_ACCELERATION_STRUCTURE           0
#define BINDING_ACCELERATION_STRUCTURE_PREVIOUS  1
#define BINDING_ACCELERATION_STRUCTURE_UNORDERED 2
#define BINDING_ACCELERATION_STRUCTURE_SSS       3
#define BINDING_SURFACE_DATA_BUFFER              4
#define BINDING_SURFACE_MAPPING_BUFFER           5
#define BINDING_SURFACE_MATERIAL_DATA_BUFFER     6
#define BINDING_SURFACE_MATERIAL_EXT_DATA_BUFFER 7
#define BINDING_VOLUME_MATERIAL_DATA_BUFFER      8
#define BINDING_LIGHT_DATA_BUFFER                9
#define BINDING_PREVIOUS_LIGHT_DATA_BUFFER       10
#define BINDING_LIGHT_MAPPING                    11
#define BINDING_BILLBOARDS_BUFFER                12
#define BINDING_BLUE_NOISE_TEXTURE               13
#define BINDING_BINDLESS_INDICES_BUFFER          14
#define BINDING_CONSTANTS                        15
#define BINDING_DEBUG_VIEW_TEXTURE               16
#define BINDING_GPU_PRINT_BUFFER                 17
#define BINDING_VALUE_NOISE_SAMPLER              18
#define BINDING_SAMPLER_READBACK_BUFFER          19

// Atmosphere LUTs use high binding slots to avoid conflicts with pass-specific bindings
#define BINDING_ATMOSPHERE_TRANSMITTANCE_LUT     200
#define BINDING_ATMOSPHERE_MULTISCATTERING_LUT   201
#define BINDING_ATMOSPHERE_SKY_VIEW_LUT          202
#define BINDING_ATMOSPHERE_CLOUD_NOISE_3D        203
#define BINDING_ATMOSPHERE_CLOUD_NOISE_SAMPLER   204
#define BINDING_ATMOSPHERE_FAST_NOISE            205
// Cloud history textures (fork): screen-space ping-pong for temporal smoothing
// of the per-frame FAST-noise jitter on the cloud ray-march. The PREV slot is
// the read view of last frame's accumulated cloud (rgb = premultiplied
// radiance, a = alpha). The CURR slot is the RW write target this frame.
// Allocated full-screen at downscale dimensions; ping-pong handled in
// RtxAtmosphere. Consumed only when evalSkyRadiance is called from the
// primary view ray (geometry_resolver miss path); PSR/indirect callers see
// the raw cloud value.
#define BINDING_ATMOSPHERE_CLOUD_HISTORY_PREV    206
#define BINDING_ATMOSPHERE_CLOUD_HISTORY_CURR    207
// Cloud history frame-id ping-pong (fork). R16_UINT screen-space pair carrying
// the frame index at which each pixel of the cloud-history color buffer was
// last written by the sky-miss path. Read at lookup time and compared against
// `(frameIdx - 1) & 0xFFFF` to reject stale history at pixels that were
// foreground-occluded last frame (their color slot retains pre-occlusion
// values because the sky-miss path didn't run there to refresh them). Without
// this, the temporal smoother's existing alpha-only disocclusion guard mis-
// identifies stale-but-nonzero history as valid and produces ~30-frame bright
// ghost trails when foreground geometry moves through bright sky / emissives.
// Clear value 0xFFFF is a "never written" sentinel; the only frame-id
// collision is once every ~18 minutes at 60fps when frameIdx wraps to
// 0xFFFF — a single rejected blend at affected pixels, imperceptible.
#define BINDING_ATMOSPHERE_CLOUD_HISTORY_FRAME_ID_PREV 212
#define BINDING_ATMOSPHERE_CLOUD_HISTORY_FRAME_ID_CURR 213
// Cloud-occluded sky-ambient transmittance LUT (fork). 2D (azimuth, elevation)
// R16F texture baked per frame from camera position: for each direction, marches
// the cloud slab and stores the directional cloud transmittance in [0, 1].
// Consumed by the volumetric pass's sky-ambient hemisphere integration to
// attenuate sky-view-LUT radiance per direction by cloud coverage along that
// direction. See docs/superpowers/specs/2026-05-12-volumetric-sky-ambient-design.md.
#define BINDING_ATMOSPHERE_CLOUD_SKY_TRANSMITTANCE_LUT 208

// Cloud render RT (Nubis Cubed 2023, fork — 2026-05-12). RGBA16F screen-space
// RT at downscale resolution containing per-pixel cloud color (premultiplied)
// in rgb and cloud transmittance in alpha. Produced by cloud_render.comp.slang
// once per frame from RtxAtmosphere::computeLuts; visualized standalone via
// DEBUG_VIEW_CLOUD_RENDER_RT (876) before sky-miss composite lands in C5.
#define BINDING_ATMOSPHERE_CLOUD_RENDER_RT 209

// Cloud voxel grids (Nubis Cubed 2023, fork — 2026-05-12). 256x256x32 R16F
// precomputed grids storing summed optical depth along the sun direction
// (D_sun) and zenith (D_ambient) at each voxel of a camera-centered tile-
// wrapped grid. Round-robin baked every 8 frames by
// cloud_sun_density_grid.comp.slang / cloud_ambient_density_grid.comp.slang.
// Consumed at shade time by the Nubis Cubed cloud-lighting path via
// sampleDSun / sampleDAmbient.
#define BINDING_ATMOSPHERE_CLOUD_D_SUN 210
#define BINDING_ATMOSPHERE_CLOUD_D_AMBIENT 211

// Sky-view LUT sampler (fork). Linear, REPEAT in azimuth (U) and CLAMP in
// elevation (V) so the wraparound at uv.x=0/1 is seamless and the poles
// don't smear horizon values across zenith/nadir. Consumed by the sky-miss
// path in evalSkyRadiance to replace the ~50-step live atmosphere march
// with one bilinear LUT tap; the LUT bake already integrates the exact
// same math evalAtmosphereRadiance does (cameraPos = origin, 32 samples,
// same multiscattering), so the swap is a pure cache hit with no behavioral
// change. Cloud-render uses its own pass-local sampler (binding 122) for
// the same texture.
#define BINDING_ATMOSPHERE_SKY_VIEW_SAMPLER 214

// Secondary-ray cloud LUT (fork — 2026-06-10, perf). 256x128 RGBA16F dome
// keyed (azimuth, elevation = (pi/2)*v^2) holding the full Nubis cloud march
// per direction: rgb = premultiplied cloud radiance, a = view transmittance
// (same convention as BINDING_ATMOSPHERE_CLOUD_RENDER_RT). Baked once per
// frame by cloud_secondary_lut.comp.slang; consumed by evalSkyRadiance's
// NON-primary branch (indirect / PSR / reflection sky-miss). Sampled with the
// sky-view sampler (REPEAT-U handles the azimuth seam).
#define BINDING_ATMOSPHERE_CLOUD_SECONDARY_LUT 215

// Cloud placement map (fork — 2026-06-11, column-shaping rework). 512x512
// RGBA8 tiled at cloudNoiseTileKm: R = cluster field (where clouds are, at
// cloud scale), G = per-cloud top-height jitter, B = base lift. Baked by
// cloud_placement_map_baker.comp.slang (live re-bake on input change).
// Drives the per-column cloud model in the density samplers (and their
// shadow-march taps). Sampled with the linear/REPEAT cloud noise sampler.
#define BINDING_ATMOSPHERE_CLOUD_PLACEMENT_MAP 216

// Fork atmosphere/cloud bindings occupy a contiguous range ABOVE COMMON_MAX_BINDING
// (which only covers the base common bindings). Expose the range so passes that
// also bind their own resources (e.g. sparse rendering) can assert no overlap.
#define BINDING_ATMOSPHERE_MIN                   BINDING_ATMOSPHERE_TRANSMITTANCE_LUT
#define BINDING_ATMOSPHERE_MAX                   BINDING_ATMOSPHERE_CLOUD_PLACEMENT_MAP

#define COMMON_MAX_BINDING                       BINDING_SAMPLER_READBACK_BUFFER
#define COMMON_NUM_BINDINGS                      (COMMON_MAX_BINDING + 1)

// Note: Used to represent a non-existent buffer
#define BINDING_INDEX_INVALID uint16_t(0xFFFF)

// Sentinel for an invalid surface index.  Equals the 21-bit maximum (SURFACE_INDEX_MAX_VALUE
// from instance_definitions.h) so that it fits inside the packed RayInteraction._surfaceAndFlags
// field.  The surfaceMapping buffer returns int32_t(-1) for unmapped surfaces; the 21-bit
// property setter truncates 0xFFFFFFFF to 0x1FFFFF automatically.
// This reserves the highest representable surface index as "invalid", reducing the usable
// range by one (max usable index = SURFACE_INDEX_MAX_VALUE - 1 = 2,097,150).
#define SURFACE_INDEX_INVALID 0x001FFFFFu

#define SAMPLER_FEEDBACK_INVALID           uint16_t(0xFFFF)
#define SAMPLER_FEEDBACK_MAX_TEXTURE_COUNT uint16_t(0xFFFF)

// Note: Light array may only be up to a size of 2^16-1, allowing the last index to be
// used for an invalid index similar to the max binding index for materials.
#define LIGHT_INDEX_INVALID (0xFFFF)

#ifdef __cplusplus

#define COMMON_RAYTRACING_BINDINGS \
  ACCELERATION_STRUCTURE(BINDING_ACCELERATION_STRUCTURE)            \
  ACCELERATION_STRUCTURE(BINDING_ACCELERATION_STRUCTURE_UNORDERED)  \
  ACCELERATION_STRUCTURE(BINDING_ACCELERATION_STRUCTURE_PREVIOUS)   \
  ACCELERATION_STRUCTURE(BINDING_ACCELERATION_STRUCTURE_SSS)        \
  STRUCTURED_BUFFER(BINDING_SURFACE_DATA_BUFFER)                    \
  STRUCTURED_BUFFER(BINDING_SURFACE_MAPPING_BUFFER)                 \
  STRUCTURED_BUFFER(BINDING_SURFACE_MATERIAL_DATA_BUFFER)           \
  STRUCTURED_BUFFER(BINDING_SURFACE_MATERIAL_EXT_DATA_BUFFER)       \
  STRUCTURED_BUFFER(BINDING_VOLUME_MATERIAL_DATA_BUFFER)            \
  STRUCTURED_BUFFER(BINDING_LIGHT_DATA_BUFFER)                      \
  STRUCTURED_BUFFER(BINDING_PREVIOUS_LIGHT_DATA_BUFFER)             \
  STRUCTURED_BUFFER(BINDING_LIGHT_MAPPING)                          \
  STRUCTURED_BUFFER(BINDING_BILLBOARDS_BUFFER)                      \
  TEXTURE2DARRAY(BINDING_BLUE_NOISE_TEXTURE)                        \
  CONSTANT_BUFFER(BINDING_CONSTANTS)                                \
  RW_TEXTURE2D(BINDING_DEBUG_VIEW_TEXTURE)                          \
  RW_STRUCTURED_BUFFER(BINDING_GPU_PRINT_BUFFER)                    \
  SAMPLER3D(BINDING_VALUE_NOISE_SAMPLER)                            \
  RW_STRUCTURED_BUFFER(BINDING_SAMPLER_READBACK_BUFFER)             \
  TEXTURE2D(BINDING_ATMOSPHERE_TRANSMITTANCE_LUT)                   \
  TEXTURE2D(BINDING_ATMOSPHERE_MULTISCATTERING_LUT)                 \
  TEXTURE2D(BINDING_ATMOSPHERE_SKY_VIEW_LUT)                        \
  TEXTURE3D(BINDING_ATMOSPHERE_CLOUD_NOISE_3D)                      \
  SAMPLER(BINDING_ATMOSPHERE_CLOUD_NOISE_SAMPLER)                   \
  TEXTURE2DARRAY(BINDING_ATMOSPHERE_FAST_NOISE)                     \
  TEXTURE2D(BINDING_ATMOSPHERE_CLOUD_HISTORY_PREV)                  \
  RW_TEXTURE2D(BINDING_ATMOSPHERE_CLOUD_HISTORY_CURR)                \
  TEXTURE2D(BINDING_ATMOSPHERE_CLOUD_HISTORY_FRAME_ID_PREV)          \
  RW_TEXTURE2D(BINDING_ATMOSPHERE_CLOUD_HISTORY_FRAME_ID_CURR)       \
  TEXTURE2D(BINDING_ATMOSPHERE_CLOUD_SKY_TRANSMITTANCE_LUT)          \
  TEXTURE2D(BINDING_ATMOSPHERE_CLOUD_RENDER_RT)                     \
  TEXTURE3D(BINDING_ATMOSPHERE_CLOUD_D_SUN)                         \
  TEXTURE3D(BINDING_ATMOSPHERE_CLOUD_D_AMBIENT)                     \
  SAMPLER(BINDING_ATMOSPHERE_SKY_VIEW_SAMPLER)                      \
  TEXTURE2D(BINDING_ATMOSPHERE_CLOUD_SECONDARY_LUT)                 \
  TEXTURE2D(BINDING_ATMOSPHERE_CLOUD_PLACEMENT_MAP)

#endif
