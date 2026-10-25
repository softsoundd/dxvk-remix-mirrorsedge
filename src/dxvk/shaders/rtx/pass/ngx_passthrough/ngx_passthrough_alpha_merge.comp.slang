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

// Combines the DLSS colour output with the game's original alpha before write-back.
// UE3's D3D9 path stores scene depth in scene-colour alpha.
// Common.usf CalcSceneDepth calls this "depth stored in alpha" when depth textures are unsupported.
// Shadow projection, fog, and depth of field read that alpha back.
// DLSS output alpha is undefined.
// Write-back carries the game's alpha through.
//
// cb.resolution is the render extent, where OriginalColor's alpha was rasterized.
// cb.outputResolution is the display extent.
// Pre-post Super Resolution upscales the scene subrect onto the full scene-colour target.
// The two extents differ in that case.

#include "rtx/pass/ngx_passthrough/ngx_passthrough_args.h"

layout(binding = 0)
ConstantBuffer<NgxPassthroughArgs> cb;

layout(binding = 1)
Texture2D<float4> DlssColor;

layout(binding = 2)
Texture2D<float4> OriginalColor;

layout(binding = 3, rgba16f)
RWTexture2D<float4> MergedOutput;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 pixelCoordinate : SV_DispatchThreadID) {
  if (pixelCoordinate.x >= uint(cb.outputResolution.x) ||
      pixelCoordinate.y >= uint(cb.outputResolution.y)) {
    return;
  }

  float3 color = DlssColor[pixelCoordinate].rgb;

  // Inverse of the encode branch: pow(encoded, 1 / inverseGamma) / colorScale.
  // Overlay is left in the encoded colour.
  if (cb.linearizeInput != 0) {
    const float exponent = max(cb.outputColorScaleAndGamma.a, 1e-4);
    const float3 scale = max(cb.outputColorScaleAndGamma.rgb, float3(1e-4));
    color = pow(max(color, float3(0.0)), 1.0 / exponent) / scale;
  } else if (cb.outputTransformEnabled != 0) {
    // Reapplies the gamma of the composite Super Resolution replaced.
    // That draw is the last colour step before the backbuffer.
    color = lerp(color * cb.outputColorScaleAndGamma.rgb,
                 cb.outputOverlayColor.rgb,
                 cb.outputOverlayColor.a);
    color = pow(saturate(color), cb.outputColorScaleAndGamma.a);
  }

  // OriginalColor is at render resolution.
  // DLAA output matches it one to one.
  // Pre-post Super Resolution maps this display pixel back into that grid.
  float alpha;

  if (cb.preserveOriginalAlpha != 0) {
    // That alpha is scene depth, rasterized through the jittered viewport.
    // The colour merged with it has already been resolved unjittered.
    // A load at the output pixel would read depth from a texel that slides with the jitter.
    // The sample position follows the jitter.
    // When the extents differ, the sample position is scaled from the display pixel into the render grid.
    // The post chain reads this depth for fog and depth of field.
    const float2 renderExtent = max(cb.resolution, float2(1.0f, 1.0f));
    const float2 displayExtent = max(cb.outputResolution, float2(1.0f, 1.0f));
    const float2 maxCoordinate = renderExtent - 1.0f;
    const float2 samplePosition = clamp((float2(pixelCoordinate) + 0.5f) * (renderExtent / displayExtent) - 0.5f + cb.jitter,
                                        float2(0.0f, 0.0f), maxCoordinate);

    const int2 corner00 = int2(floor(samplePosition));
    const int2 corner11 = int2(min(float2(corner00) + 1.0f, maxCoordinate));
    const float2 weight = samplePosition - float2(corner00);

    const float alpha00 = OriginalColor[uint2(corner00)].a;
    const float alpha10 = OriginalColor[uint2(corner11.x, corner00.y)].a;
    const float alpha01 = OriginalColor[uint2(corner00.x, corner11.y)].a;
    const float alpha11 = OriginalColor[uint2(corner11)].a;

    alpha = lerp(lerp(alpha00, alpha10, weight.x),
                 lerp(alpha01, alpha11, weight.x),
                 weight.y);
  } else {
    alpha = DlssColor[pixelCoordinate].a;
  }

  MergedOutput[pixelCoordinate] = float4(color, alpha);
}
