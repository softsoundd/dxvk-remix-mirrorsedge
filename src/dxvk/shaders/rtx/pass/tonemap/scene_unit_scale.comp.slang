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

// Scales linear HDR colour in place, e.g. Remix radiance into the game's scene units and back.

#include "rtx/pass/tonemap/scene_unit_scale.h"

[[vk::binding(SCENE_UNIT_SCALE_COLOR)]]
RWTexture2D<float4> InOutColorBuffer;

layout(push_constant)
ConstantBuffer<SceneUnitScaleArgs> cb;

// The image is R16G16B16A16_SFLOAT; keep scaled values finite
static const float kMaxHalf = 65504.0f;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  uint2 dims;
  InOutColorBuffer.GetDimensions(dims.x, dims.y);
  if (any(threadId >= dims))
  {
    return;
  }

  const float4 color = InOutColorBuffer[threadId];
  InOutColorBuffer[threadId] = float4(clamp(color.rgb * cb.scale, -kMaxHalf, kMaxHalf), color.a);
}
