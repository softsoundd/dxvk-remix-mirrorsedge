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

//!variant bloom_fft_128.comp FFT_SIZE=128
//!variant bloom_fft_256.comp FFT_SIZE=256
//!variant bloom_fft_512.comp FFT_SIZE=512
//!variant bloom_fft_1024.comp FFT_SIZE=1024
//!variant bloom_fft_2048.comp FFT_SIZE=2048
//!variant bloom_fft_4096.comp FFT_SIZE=4096
//!end-variants

// One orthonormal 1D FFT pass over the rows (or columns) of a two channel complex image from firstLine on, in place.
// One workgroup transforms one line, its two complex channels in turn through shared memory: it reads the whole line
// before its first barrier and writes it after its last, and no other workgroup touches that line.

#include "rtx/pass/bloom/bloom.h"
#include "rtx/utility/fft.slangh"

layout(rgba32f, binding = BLOOM_FFT_SPECTRUM_INPUT_OUTPUT)
RWTexture2D<float4> InOutSpectrum;

layout(push_constant)
ConstantBuffer<BloomFftArgs> cb;

uint2 lineTexel(uint line, uint element)
{
  return bool(cb.alongY) ? uint2(line, element) : uint2(element, line);
}

[shader("compute")]
[numthreads(kFftThreads, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint threadIndex : SV_GroupIndex)
{
  const bool forward = bool(cb.forward);
  const uint line = cb.firstLine + groupId.x;

  float2 rg[kFftElements];
  float2 ba[kFftElements];

  [unroll]
  for (uint e = 0; e < kFftElements; e++)
  {
    const uint element = threadIndex + e * kFftThreads;
    const float4 value = InOutSpectrum[lineTexel(line, element)];

    g_fftLine[element] = value.xy;
    ba[e] = value.zw;
  }

  fftLine(threadIndex, forward);

  [unroll]
  for (uint e = 0; e < kFftElements; e++)
  {
    rg[e] = g_fftLine[threadIndex + e * kFftThreads];
  }

  // Every thread has its transformed red and green before blue and alpha overwrite them.
  GroupMemoryBarrierWithGroupSync();

  [unroll]
  for (uint e = 0; e < kFftElements; e++)
  {
    g_fftLine[threadIndex + e * kFftThreads] = ba[e];
  }

  fftLine(threadIndex, forward);

  const float scale = rsqrt(float(FFT_SIZE));

  [unroll]
  for (uint e = 0; e < kFftElements; e++)
  {
    const uint element = threadIndex + e * kFftThreads;

    InOutSpectrum[lineTexel(line, element)] = float4(rg[e], g_fftLine[element]) * scale;
  }
}
