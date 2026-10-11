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

// Multiplies the image spectrum by the kernel's. The spectrum at k and its mirror at N - k are needed together, so
// each thread of the lower half of the rows handles both, writing every frequency exactly once. With orthonormal
// transforms the product of two spectra is sqrt(N) short of the convolution's. The kernel keeps its own energy: the
// near and coarse parts each hold their share of the light, which together make one. A delta's spectrum times
// sqrt(N) is 1, so taking the centre tap out of the kernel takes that share of the image's own spectrum out of the
// product.

#include "rtx/pass/bloom/bloom.h"
#include "rtx/utility/real_fft_packing.slangh"

layout(binding = BLOOM_FFT_CONVOLVE_SPECTRUM_INPUT)
Texture2D<float4> InSpectrum;

layout(binding = BLOOM_FFT_CONVOLVE_KERNEL_INPUT)
Texture2D<float4> InKernelSpectrum;

layout(binding = BLOOM_FFT_CONVOLVE_CENTER_TAP_INPUT)
Texture2D<float4> InCenterTap;

layout(rgba32f, binding = BLOOM_FFT_CONVOLVE_OUTPUT)
RWTexture2D<float4> OutSpectrum;

layout(push_constant)
ConstantBuffer<BloomFftConvolveArgs> cb;

[shader("compute")]
[numthreads(32, 4, 1)]
void main(uint2 p : SV_DispatchThreadID)
{
  const uint2 size = cb.bufferSize;

  if (p.x >= size.x || p.y > size.y / 2)
  {
    return;
  }

  const uint2 ip = uint2((size.x - p.x) % size.x, (size.y - p.y) % size.y);

  const float4 zk = InSpectrum.Load(int3(p, 0));
  const float4 zNk = InSpectrum.Load(int3(ip, 0));
  const float4 mk = InKernelSpectrum.Load(int3(p, 0));
  const float4 mNk = InKernelSpectrum.Load(int3(ip, 0));

  // The alpha channel carries no image, so its kernel is left at zero.
  const float scale = sqrt(float(size.x) * float(size.y));
  const float4 normFactor = float4(scale, scale, scale, 0.0f);

  const float4 centerTap = cb.subtractCenter ? float4(InCenterTap.Load(int3(0, 0, 0)).xyz, 0.0f) : float4(0.0f);

  OutSpectrum[p] = convolveRealPairs(zk, zNk, mk, mNk, normFactor, centerTap);

  if (ip.y > size.y / 2)
  {
    OutSpectrum[ip] = convolveRealPairs(zNk, zk, mNk, mk, normFactor, centerTap);
  }
}
