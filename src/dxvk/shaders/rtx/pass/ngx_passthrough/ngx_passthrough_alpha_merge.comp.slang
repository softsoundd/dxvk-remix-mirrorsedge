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

// NGX passthrough mode, pre-post-process injection: combines the DLSS color output with
// the game's original alpha channel before the result is written back over the scene
// color. UE3's D3D9 path stores the scene depth in the scene color alpha (see UE3
// Common.usf CalcSceneDepth: "depth stored in alpha" when depth textures are
// unsupported); shadow projection, fog, and depth of field read it back, and DLSS output
// alpha is undefined, so the write-back must carry the game's own alpha through.
//
// Note: only used when render extent == display extent (the pre-post injection is
// DLAA-only), so cb.resolution matches the output size and both inputs are 1:1.

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

  // The Super Resolution path replaces UE3's FinishRenderViewTarget composite, so that draw's
  // own transform is reapplied here: it is the last thing between the post chain and the
  // backbuffer, and skipping it leaves the frame un-gamma-corrected.
  if (cb.outputTransformEnabled != 0) {
    color = lerp(color * cb.outputColorScaleAndGamma.rgb,
                 cb.outputOverlayColor.rgb,
                 cb.outputOverlayColor.a);
    color = pow(saturate(color), cb.outputColorScaleAndGamma.a);
  }

  // OriginalColor is render-resolution, so it is only 1:1 with the output on the pre-post
  // injection path - which is exactly where the game's alpha needs carrying through.
  float alpha;

  if (cb.preserveOriginalAlpha != 0) {
    // That alpha is scene depth, rasterized through the jittered viewport, while the colour it is
    // being merged with has been resolved un-jittered. Buffer position q therefore holds the depth
    // of un-jittered position q - jitter, so a straight load would pair each pixel's colour with
    // depth from a slightly different place - one that moves every frame as the jitter does.
    // Reconstructing at the pixel centre pins it, which matters because the game's post chain runs
    // after this injection point and reads this depth for fog and depth of field.
    const float2 maxCoordinate = cb.resolution - 1.0f;
    const float2 samplePosition = clamp(float2(pixelCoordinate) + cb.jitter, float2(0.0f, 0.0f), maxCoordinate);

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
