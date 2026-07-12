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

// Cloud Render Compute Pass — Nubis Cubed 2023 (fork — 2026-05-12, C4).
//
// Per-pixel view-direction raymarch through the cloud slab using the Nubis
// Cubed 2023 lighting equations (paper pp. 137, 142). Writes premultiplied
// RGB + view-ray cloud transmittance alpha to AtmosphereCloudRender at the
// downscale render extent.
//
// The view-march replaced the old analytical sky-pass cloud rendering — the
// Wrenninge octave sum + isotropic deep-scatter floor (removed 2026-06-19) —
// with the page-137 two-HG-lobe direct term (sigma_ms remap on sun_dot + sdf
// depth) and the page-142 ambient pow(1 - dim_profile, 0.5) * exp(-D_ambient).
//
// Composited into primary sky-miss when cloudRenderRTEnable is on (gated by
// RTX_OPTION). A debug view (enum 876) shows the raw RT for inspection.
//
// Coordinate frame: ray-vs-slab intersection runs in camera-at-origin Y-up
// space (the established cloud-math convention — see `intersectCloudLayer`'s
// "Ray origin is at the camera (0,0,0) in Y-up space" comment), so the
// per-step `t` distances are measured from the camera. Per-sample positions
// passed to the density / lighting helpers, however, are WORLD-ANCHORED:
// `samplePos = args.cameraWorldPosYUpKm + viewDirYUp * t`. This anchors the
// noise / coverage / type fields and the D_sun / D_ambient voxel-grid
// lookups to the player's world position (km-scale Y-up coords pushed by
// setCloudShadowCameraPosition()) so the cloud volume translates with the
// world rather than rigidly with the camera — the parallax depth cue that
// distinguishes a 3D volume from cardboard cutouts.
//
// Per-pixel view directions are reconstructed CPU-side into the
// cloudRenderForwardYUp / RightYUp / UpYUp basis vectors in AtmosphereArgs;
// the shader builds viewDir per pixel from those basis vectors. Sun
// direction comes from args.sunDirection (already Y-up).

// Note: define ATMOSPHERE_AVAILABLE so the sampleDSun / sampleDAmbient
// helpers in atmosphere_common.slangh resolve to texture lookups (rather
// than the binding-free fallback stubs used by the LUT bakers). The
// global names the helpers reach for (AtmosphereCloudDSun, AtmosphereCloudDAmbient,
// AtmosphereCloudNoise3D, AtmosphereCloudNoise3DSampler, AtmosphereFastNoise)
// are declared at our local slots BELOW so they're in scope before the
// include's helper bodies are parsed.
#define ATMOSPHERE_AVAILABLE

// Likewise enable the LUT-backed cloudHeightProfile() in atmosphere_common.slangh
// for this TU. The AtmosphereCloudHeightLut + AtmosphereCloudHeightLutSampler
// globals the helper reaches for are declared at slots 10 / 11 below.
// Voxel grid bakers, the cloud-noise baker, and the height LUT baker itself
// do NOT define this — they get the procedural cloudTypeProfile fallback via
// atmosphere_common.slangh's #else branch.
#define ATMOSPHERE_CLOUD_HEIGHT_LUT_AVAILABLE

// Bring in shader_types (for vec3/uint/etc.) and AtmosphereArgs ahead of the
// global texture declarations so AtmosphereArgs is defined before slot 0's CB.
#include "rtx/utility/shader_types.h"
#include "rtx/pass/atmosphere/atmosphere_args.h"

layout(binding = 0)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

// Cloud 3D Perlin FBM noise volume — same resource that sampleCloudDensityTextured
// consumes elsewhere. Linear/REPEAT sampler matches the frac()-tile-wrap
// convention. NOTE: the binding *name* `AtmosphereCloudNoise3D` matches the
// global declared in common_bindings.slangh; we re-declare it here at our own
// binding slot since this compute pass owns its descriptor set independently
// from the COMMON_RAYTRACING_BINDINGS macro.
layout(binding = 1)
Texture3D<float> AtmosphereCloudNoise3D;

layout(binding = 2)
SamplerState AtmosphereCloudNoise3DSampler;

// Cloud voxel grids (D_sun and D_ambient), sampled via the same linear/REPEAT
// sampler. Names match the common_bindings.slangh globals so the helpers in
// atmosphere_common.slangh (sampleDSun / sampleDAmbient) resolve them.
layout(binding = 3)
Texture3D<float> AtmosphereCloudDSun;

layout(binding = 4)
Texture3D<float> AtmosphereCloudDAmbient;

// EA Importance-Sampled FAST noise (128x128x32 RG8 Texture2DArray) — same
// resource consumed by fastJitter() in atmosphere_common.slangh.
layout(binding = 5)
Texture2DArray<float2> AtmosphereFastNoise;

layout(binding = 6, rgba16f)
RWTexture2D<float4> cloudRenderOutput;

// Sky-view LUT + cloud-sky-transmittance LUT for the page-142 ambient
// term. sampleSkyAmbientForVolume in atmosphere_common.slangh consumes
// these as explicit parameters (so the helper compiles in any TU).
// Linear/CLAMP sampler — sky-view LUT is keyed by (azimuth, elevation);
// REPEAT would alias the south pole onto the north.
layout(binding = 7)
Texture2D<float4> AtmosphereSkyViewLut;

layout(binding = 8)
Texture2D<float> AtmosphereCloudSkyTransmittanceLut;

layout(binding = 9)
SamplerState AtmosphereSkyViewSampler;

// Cloud height LUT (slide 3 lift — RDR2 SIGGRAPH 2019). 64x128 RG8 baked
// once at startup, keyed by (typeSlice, heightFrac).
//   R = per-altitude density envelope multiplier (gentle altitude-shape effect)
//   G = per-altitude coverage threshold scale (widens silhouette horizontally —
//       this is the lever with visible bite, encoding the anvil-cap effect)
// Consumed by atmosphere_common.slangh's cloudHeightProfileFull() when
// ATMOSPHERE_CLOUD_HEIGHT_LUT_AVAILABLE is defined. Linear/CLAMP sampler —
// the LUT is parameterized on a bounded domain, REPEAT would alias the
// cumulonimbus anvil column back into stratus territory.
// RGBA8 since column-shaping rev 3 (B = cumulative envelope integral for
// the analytic downwelling light field; see cloud_height_lut_baker).
layout(binding = 10)
Texture2D<float4> AtmosphereCloudHeightLut;

layout(binding = 11)
SamplerState AtmosphereCloudHeightLutSampler;

// Cloud placement map (fork — 2026-06-11, column-shaping rework). 512x512
// RGBA8 baked by cloud_placement_map_baker.comp.slang, tiled at
// cloudNoiseTileKm: R = cluster field, G = top-height jitter, B = base
// lift. Drives the per-column cloud model in sampleCloudDensityTextured.
// Sampled with the cloud-noise sampler (linear/REPEAT, slot 2).
layout(binding = 12)
Texture2D<float4> AtmosphereCloudPlacementMap;

// Include the helper library AFTER the texture globals are declared so the
// ATMOSPHERE_AVAILABLE-gated helper bodies (sampleDSun / sampleDAmbient /
// fastJitter / cloudHeightProfile / etc.) resolve to our local declarations.
#include "rtx/pass/atmosphere/atmosphere_common.slangh"

// Shared slab-march + shade-context library (fork — 2026-06-10). Hosts
// marchCloudSlab / marchCloudLayers / buildCloudShadeContext /
// sampleCloudSunOpticalDepth_local — everything this pass shares with the
// secondary-ray cloud LUT bake (cloud_secondary_lut.comp.slang). Must
// follow atmosphere_common.slangh. The extraction is behavior-neutral:
// this pass calls marchCloudLayers with 0/0 distance clamps, which is
// bit-identical to the pre-extraction inline march.
#include "rtx/pass/atmosphere/cloud_march_common.slangh"

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
  uint2 pixelCoord = dispatchId.xy;
  uint2 outputDim;
  cloudRenderOutput.GetDimensions(outputDim.x, outputDim.y);
  if (any(pixelCoord >= outputDim)) return;

  AtmosphereArgs args = atmosphereArgs;

  // Trivially-clear sky: no clouds means full transmittance, zero radiance.
  // Effective coverage is layer 1's mean OR layer 2's mean (whichever is
  // enabled), so a "cirrus-only" preset (layer 1 coverage near zero, layer 2
  // enabled with its own coverage) still passes the early-out.
  float effectiveCoverageMean = args.cloudCoverageMean;
  if (args.cloudLayer2Enable != 0u) {
    effectiveCoverageMean = max(effectiveCoverageMean, args.cloudLayer2CoverageMean);
  }
  if (args.cloudEnabled < 0.5f || effectiveCoverageMean < 0.01f || args.cloudViewSamples == 0u) {
    cloudRenderOutput[pixelCoord] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    return;
  }

  // ===== Reconstruct per-pixel view ray in Y-up world space =====
  // NDC in [-1, 1] with top-left pixel mapping to (-1, +1) Vulkan convention.
  vec2 uv = (vec2(pixelCoord) + 0.5f) / vec2(outputDim);
  vec2 ndc = uv * 2.0f - 1.0f;
  ndc.y = -ndc.y;  // y-down pixels -> y-up NDC

  // Basis vectors are pre-multiplied by tan(halfFovX/Y) and aspect ratio
  // CPU-side, so the math is just a weighted sum.
  vec3 viewDirYUp = normalize(args.cloudRenderForwardYUp
                              + ndc.x * args.cloudRenderRightYUp
                              + ndc.y * args.cloudRenderUpYUp);

  // Below-horizon view direction: no clouds above this pixel.
  if (viewDirYUp.y <= 0.001f) {
    cloudRenderOutput[pixelCoord] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    return;
  }

  // ===== Per-pixel FAST-noise jitter on the fractional step offset =====
  // Channel x is the view-march; we don't take more than one tap so channel
  // y is unused.
  // Slab intersection + stepLen now live inside marchCloudSlab so both layer
  // marches can reuse them with their own slab params.
  float jitter = fastJitter(pixelCoord, args.cloudRenderFrameIdx).x;

  // ===== Static-per-frame shading context (cloud_march_common.slangh) =====
  // Sun/sky radiances, the full per-moon precompute, combined-moon shadow
  // direction, and the Wrenninge octave count — all pixel-independent, all
  // extracted verbatim into buildCloudShadeContext.
  CloudShadeContext ctx = buildCloudShadeContext(args);

  // ===== View-ray raymarch, both layers (cloud_march_common.slangh) =====
  // evalNubisCubedSample inside marchCloudSlab uses sigma = args.cloudDensity
  // (no doubling — Nubis Cubed equations take raw cloudDensity as extinction).
  // marchCloudLayers sequences layer 1 (cumulus) then the optional layer-2
  // cirrus deck front-to-back, zeroing the moon coefficients for layer 2.
  // The 0/0 distance clamps mean "march the full slab" — bit-identical to
  // the pre-extraction inline march.
  vec3 accumColor = vec3(0.0f);
  float viewTransmittance = 1.0f;

  marchCloudLayers(viewDirYUp, jitter,
                   /*tMinClamp=*/0.0f,
                   /*tMaxClamp=*/0.0f,
                   args.cloudViewSamples, ctx, args,
                   accumColor, viewTransmittance);

  // Output is premultiplied: rgb = accumulated cloud radiance,
  // a = view-ray cloud transmittance.
  cloudRenderOutput[pixelCoord] = vec4(accumColor, viewTransmittance);
}
