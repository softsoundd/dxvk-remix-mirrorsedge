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

//!variant cloud_volume_mip_unorm.comp
//!>       CLOUD_VOLUME_MIP_FLOAT=0

//!variant cloud_volume_mip_float.comp
//!>       CLOUD_VOLUME_MIP_FLOAT=1

//!end-variants

// One mip of a periodic volume from the level above, a 2x2x2 box: a linear tap at the destination texel's centre
// lands between eight source texels, and the repeating sampler wraps the edges. Unorm (the detail volume, RGBA)
// and float (the sun grid, RG) variants differ only in the output's format.

#include "rtx/pass/clouds/cloud_binding_indices.h"
#include "rtx/utility/shader_types.h"

layout(binding = CLOUD_BINDING_BAKE_INPUT)
Texture3D<float4> source;

layout(binding = CLOUD_BINDING_VOLUME_SAMPLER)
SamplerState boxSampler;

#if CLOUD_VOLUME_MIP_FLOAT
layout(binding = CLOUD_BINDING_BAKE_OUTPUT, rg16f)
RWTexture3D<float2> destination;
#else
layout(binding = CLOUD_BINDING_BAKE_OUTPUT, rgba8)
RWTexture3D<float4> destination;
#endif

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 threadId : SV_DispatchThreadID) {
  uint3 size;
  destination.GetDimensions(size.x, size.y, size.z);
  if (any(threadId >= size)) {
    return;
  }

  const vec3 uvw = (vec3(threadId) + 0.5f) / vec3(size);
#if CLOUD_VOLUME_MIP_FLOAT
  destination[threadId] = source.SampleLevel(boxSampler, uvw, 0).xy;
#else
  destination[threadId] = source.SampleLevel(boxSampler, uvw, 0);
#endif
}
