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

// The aperture's power spectrum as a share of its total, for the kernel build to interpolate. The power is taken
// before any interpolation, as interpolating the complex spectrum would interfere neighbouring coefficients. The
// raster's texels average the aperture over their footprint, which dims the spectrum by sinc^2 along each axis, so
// that is divided out here.

#include "rtx/pass/bloom/bloom.h"
#include "rtx/utility/math.slangh"

layout(binding = BLOOM_KERNEL_POWER_INPUT)
Texture2D<float4> InApertureSpectrum;

layout(r32f, binding = BLOOM_KERNEL_POWER_OUTPUT)
RWTexture2D<float> OutPower;

layout(push_constant)
ConstantBuffer<BloomKernelPowerArgs> cb;

float boxFootprint(float frequency)
{
  const float x = pi * frequency;
  return abs(x) < 1e-4f ? 1.0f : sin(x) / x;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 texel : SV_DispatchThreadID)
{
  if (any(texel >= uint2(cb.size)))
  {
    return;
  }

  const int size = int(cb.size);
  const int2 t = int2(texel);
  const vec2 k = vec2(float(t.x < size / 2 ? t.x : t.x - size), float(t.y < size / 2 ? t.y : t.y - size));
  const vec2 coefficient = InApertureSpectrum.Load(int3(texel, 0)).xy;
  const vec2 footprint = vec2(boxFootprint(k.x / float(size)), boxFootprint(k.y / float(size)));

  OutPower[texel] = dot(coefficient, coefficient) / (cb.apertureEnergy * footprint.x * footprint.x * footprint.y * footprint.y);
}
