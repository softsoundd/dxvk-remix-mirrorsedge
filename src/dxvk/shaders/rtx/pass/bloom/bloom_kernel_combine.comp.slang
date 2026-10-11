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

// Completes the kernel: its centre texel takes the light the near and coarse parts leave, so the whole has unit
// energy. When the parts already hold more than all of it they are scaled down instead. The centre tap is also written
// out, for the convolution to take out.

#include "rtx/pass/bloom/bloom.h"

layout(binding = BLOOM_KERNEL_COMBINE_TOTALS_INPUT)
Texture2D<float4> InTotals;

layout(rgba32f, binding = BLOOM_KERNEL_COMBINE_KERNEL_INPUT_OUTPUT)
RWTexture2D<float4> InOutKernel;

layout(rgba32f, binding = BLOOM_KERNEL_COMBINE_COARSE_KERNEL_INPUT_OUTPUT)
RWTexture2D<float4> InOutCoarseKernel;

layout(rgba32f, binding = BLOOM_KERNEL_COMBINE_CENTER_TAP_OUTPUT)
RWTexture2D<float4> OutCenterTap;

layout(push_constant)
ConstantBuffer<BloomKernelCombineArgs> cb;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 texel : SV_DispatchThreadID)
{
  const bool center = all(texel == uint2(0, 0));

  if (cb.identity)
  {
    if (all(texel < cb.bufferSize))
    {
      InOutKernel[texel] = center ? float4(1.0f, 1.0f, 1.0f, 0.0f) : float4(0.0f);
    }

    if (cb.coarseEnabled && all(texel < cb.coarseBufferSize))
    {
      InOutCoarseKernel[texel] = float4(0.0f);
    }

    if (center)
    {
      OutCenterTap[uint2(0, 0)] = float4(1.0f, 1.0f, 1.0f, 0.0f);
    }

    return;
  }

  const vec3 spread = InTotals.Load(int3(0, 0, 0)).rgb + (cb.coarseEnabled ? InTotals.Load(int3(1, 0, 0)).rgb : vec3(0.0f));
  const vec3 scale = vec3(
    spread.x > 1.0f ? 1.0f / spread.x : 1.0f,
    spread.y > 1.0f ? 1.0f / spread.y : 1.0f,
    spread.z > 1.0f ? 1.0f / spread.z : 1.0f);
  const vec3 centerTap = max(vec3(1.0f) - spread, vec3(0.0f));

  if (all(texel < cb.bufferSize))
  {
    const vec3 value = InOutKernel[texel].rgb * scale;

    InOutKernel[texel] = float4(center ? centerTap : value, 0.0f);
  }

  if (cb.coarseEnabled && all(texel < cb.coarseBufferSize))
  {
    InOutCoarseKernel[texel] = float4(InOutCoarseKernel[texel].rgb * scale, 0.0f);
  }

  if (center)
  {
    OutCenterTap[uint2(0, 0)] = float4(centerTap, 0.0f);
  }
}
