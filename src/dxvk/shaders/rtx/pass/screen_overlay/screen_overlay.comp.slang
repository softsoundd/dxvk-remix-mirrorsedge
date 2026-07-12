/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
#include "screen_overlay.h"

layout(binding = SCREEN_OVERLAY_INPUT_OUTPUT)
RWTexture2D<float4> InOutImage;

layout(binding = SCREEN_OVERLAY_TEXTURE)
Sampler2D<float4> OverlayTexture;

layout(push_constant)
ConstantBuffer<ScreenOverlayArgs> cb;

[shader("compute")]
[numthreads(SCREEN_OVERLAY_TILE_SIZE, SCREEN_OVERLAY_TILE_SIZE, 1)]
void main(uint2 threadId : SV_DispatchThreadID) {
  if (any(threadId >= cb.imageSize))
    return;

  // Map pixel position to normalized UV for overlay sampling
  float2 uv = (float2(threadId) + 0.5f) / float2(cb.imageSize);

  float4 overlay = OverlayTexture.SampleLevel(uv, 0);
  float4 target = InOutImage[threadId];

  // Standard alpha blend: overlay over target
  float alpha = overlay.a * cb.opacity;
  float3 blended = overlay.rgb * alpha + target.rgb * (1.0f - alpha);

  InOutImage[threadId] = float4(blended, 1.0f);
}
