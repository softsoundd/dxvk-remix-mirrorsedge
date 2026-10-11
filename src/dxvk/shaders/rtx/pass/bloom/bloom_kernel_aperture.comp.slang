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

// Rasterises the observer's aperture as the real part of a complex image, for its far field diffraction pattern. The
// camera's iris takes up 1 / BLOOM_KERNEL_APERTURE_OVERSAMPLING of the buffer's width, half, the least that keeps the
// pattern, its autocorrelation's transform, from aliasing. The eye's pupil fills it: its diffraction core is always
// far smaller than a kernel texel, and the particles' corona needs the reach. Rows run down while the aperture is
// defined with y up, which flips it, and the kernel build samples the pattern with offsets that run down too, which
// flips it back.
//
// The eye's aperture is its pupil, apodised by the Stiles-Crawford effect and crossed by eyelashes and by the lens's
// and vitreous's particles projected into it (after Ritschel et al. 2009 and van den Berg et al. 2005).

#include "rtx/pass/bloom/bloom.h"
#include "rtx/utility/lens_aperture.slangh"

layout(binding = BLOOM_KERNEL_APERTURE_PARTICLES_INPUT)
StructuredBuffer<float4> Particles;

layout(binding = BLOOM_KERNEL_APERTURE_CELLS_INPUT)
StructuredBuffer<uint2> Cells;

layout(binding = BLOOM_KERNEL_APERTURE_CELL_PARTICLES_INPUT)
StructuredBuffer<uint> CellParticles;

layout(rgba32f, binding = BLOOM_KERNEL_APERTURE_OUTPUT)
RWTexture2D<float4> OutAperture;

layout(push_constant)
ConstantBuffer<BloomKernelApertureArgs> cb;

static const uint kSupersampling = 4;

// Transmission of the eye's pupil at p, in units of the pupil's radius, y up.
float eyeTransmission(vec2 p)
{
  const float radius = length(p);

  if (radius > 1.0f)
  {
    return 0.0f;
  }

  // By the Stiles-Crawford effect the cones accept light from the pupil's edge less, eta(r) = 10^(-rho r^2), an
  // amplitude of its square root.
  const float radiusMm = radius * cb.pupilRadiusMm;
  float transmission = pow(10.0f, -0.5f * cb.stilesCrawford * radiusMm * radiusMm);

  // Lashes hang from the upper lid across the top of the pupil, each a slightly slanted opaque strand.
  for (uint i = 0; i < cb.lashCount; i++)
  {
    const float x = lerp(-0.9f, 0.9f, (float(i) + lensApertureHashUnit(i * 3 + 1)) / float(cb.lashCount));
    const float slant = (lensApertureHashUnit(i * 3 + 2) - 0.5f) * 0.6f + 0.05f * sin(cb.lashPhase + float(i));
    const float tip = lerp(0.2f, 0.7f, lensApertureHashUnit(i * 3 + 3));

    if (p.y > tip)
    {
      const float strand = x + slant * (1.0f - p.y);

      if (abs(p.x - strand) < 0.5f * cb.lashWidth)
      {
        return 0.0f;
      }
    }
  }

  // Tests the particles binned in the cell over p.
  const uvec2 cell = uvec2(clamp((p * 0.5f + 0.5f) * float(cb.cellsPerSide), vec2(0.0f), vec2(float(cb.cellsPerSide) - 1.0f)));
  const uint2 range = Cells[cell.y * cb.cellsPerSide + cell.x];

  for (uint i = 0; i < range.y; i++)
  {
    const float4 particle = Particles[CellParticles[range.x + i]];
    const vec2 d = p - particle.xy;

    if (dot(d, d) < particle.z * particle.z)
    {
      return 0.0f;
    }
  }

  return transmission;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 texel : SV_DispatchThreadID)
{
  if (any(texel >= uint2(cb.size)))
  {
    return;
  }

  const float center = 0.5f * float(cb.size);
  float amplitude = 0.0f;

  // Supersampled, so that the stair-stepped edges of the aperture do not streak its pattern.
  [unroll]
  for (uint j = 0; j < kSupersampling; j++)
  {
    [unroll]
    for (uint i = 0; i < kSupersampling; i++)
    {
      const vec2 position = vec2(texel) + (vec2(float(i), float(j)) + 0.5f) / float(kSupersampling);
      const vec2 p = vec2(position.x - center, center - position.y) / cb.radiusTexels;

      amplitude += cb.observer == BLOOM_OBSERVER_EYE
        ? eyeTransmission(p)
        : lensApertureCoverage(p, cb.blades, cb.curvature, cb.rotation, cb.distanceJitter, cb.angleJitter, 1e-4f);
    }
  }

  OutAperture[texel] = float4(amplitude / float(kSupersampling * kSupersampling), 0.0f, 0.0f, 0.0f);
}
