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

// NGX passthrough debug view: the linear colour sampled at the upscaler injection
// point (the same input DLSS/NIS/TAA-U/XeSS would consume). At the pre-post-process
// injection this is the game's HDR scene colour before bloom/tonemap; at the late
// injection point it is the post-processed output instead. Display-mapped for SDR
// inspection without crushing low values.

#include "rtx/pass/ngx_passthrough/ngx_passthrough_args.h"

layout(binding = 0)
ConstantBuffer<NgxPassthroughArgs> cb;

layout(binding = 1)
Texture2D<float4> SceneColor;

layout(binding = 2, rgba16f)
RWTexture2D<float4> DebugOutput;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 pixelCoordinate : SV_DispatchThreadID) {
  if (pixelCoordinate.x >= uint(cb.resolution.x) ||
      pixelCoordinate.y >= uint(cb.resolution.y)) {
    return;
  }

  float3 color = max(SceneColor[pixelCoordinate].rgb, 0.0f);

  // Fixed exposure + peak-normalised Reinhard: preserves relative HDR ratios while
  // mapping into a range the swapchain blit can show without clipping everything.
  const float exposure = 4.0f;
  color *= exposure;
  const float peak = max(max(color.r, color.g), color.b);
  color /= (1.0f + peak);

  DebugOutput[pixelCoordinate] = float4(color, 1.0f);
}
