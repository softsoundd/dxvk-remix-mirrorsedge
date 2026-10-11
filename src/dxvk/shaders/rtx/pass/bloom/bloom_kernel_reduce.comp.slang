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

//!variant bloom_kernel_reduce_rows.comp REDUCE_ROWS=1
//!variant bloom_kernel_reduce_total.comp REDUCE_ROWS=0
//!end-variants

// Sums the kernel's two parts, the near and the coarse buffer's, per channel: one workgroup per row of a part into
// the row sums (x is the row, y the part), then one workgroup over each part's rows into the totals (x is the part).
// For the totals, bufferSize holds each part's row count.

#include "rtx/pass/bloom/bloom.h"

layout(push_constant)
ConstantBuffer<BloomKernelReduceArgs> cb;

#if REDUCE_ROWS

layout(binding = BLOOM_KERNEL_REDUCE_INPUT)
Texture2D<float4> InKernel;

layout(rgba32f, binding = BLOOM_KERNEL_REDUCE_ROW_SUMS_OUTPUT)
RWTexture2D<float4> OutRowSums;

#else

layout(binding = BLOOM_KERNEL_REDUCE_ROW_SUMS_INPUT)
Texture2D<float4> InRowSums;

layout(rgba32f, binding = BLOOM_KERNEL_REDUCE_TOTALS_OUTPUT)
RWTexture2D<float4> OutTotals;

#endif

static const uint kGroupSize = BLOOM_KERNEL_REDUCE_GROUP_SIZE;

groupshared vec3 g_sums[kGroupSize];

vec3 reduceGroup(uint threadIndex, vec3 value)
{
  g_sums[threadIndex] = value;
  GroupMemoryBarrierWithGroupSync();

  for (uint stride = kGroupSize / 2; stride > 0; stride /= 2)
  {
    if (threadIndex < stride)
    {
      g_sums[threadIndex] += g_sums[threadIndex + stride];
    }

    GroupMemoryBarrierWithGroupSync();
  }

  const vec3 sum = g_sums[0];
  GroupMemoryBarrierWithGroupSync();
  return sum;
}

[shader("compute")]
[numthreads(kGroupSize, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint threadIndex : SV_GroupIndex)
{
#if REDUCE_ROWS
  const uint row = groupId.x;
  vec3 sum = vec3(0.0f);

  for (uint x = threadIndex; x < cb.bufferSize.x; x += kGroupSize)
  {
    sum += InKernel.Load(int3(x, row, 0)).rgb;
  }

  sum = reduceGroup(threadIndex, sum);

  if (threadIndex == 0)
  {
    OutRowSums[uint2(row, cb.part)] = float4(sum, 0.0f);
  }
#else
  [unroll]
  for (uint part = 0; part < 2; part++)
  {
    const uint rows = part == 0 ? cb.bufferSize.x : cb.bufferSize.y;
    vec3 sum = vec3(0.0f);

    for (uint row = threadIndex; row < rows; row += kGroupSize)
    {
      sum += InRowSums.Load(int3(row, part, 0)).rgb;
    }

    sum = reduceGroup(threadIndex, sum);

    if (threadIndex == 0)
    {
      OutTotals[uint2(part, 0)] = float4(sum, 0.0f);
    }
  }
#endif
}
