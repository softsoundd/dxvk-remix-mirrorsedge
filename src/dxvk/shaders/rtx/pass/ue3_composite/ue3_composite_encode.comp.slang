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

// Turns a pass of a UE3 composite material bake into a texture the material renders with. A pass holds
// Emissive + Diffuse in linear scene units and the emission pass the emission alone, so the material's own
// colour is their difference. Both are stored gamma encoded, which is how the surface material decodes
// legacy albedo and every emissive texture. The passes of a lifted mask differ only in the mask, which
// does not reach the emission, so they all take out the same one.

#include "rtx/pass/ue3_composite/ue3_composite_encode.h"
#include "rtx/utility/color.slangh"

layout(binding = UE3_COMPOSITE_ENCODE_PASS_INPUT)
Texture2D<vec4> Pass;

layout(binding = UE3_COMPOSITE_ENCODE_EMISSION_INPUT)
Texture2D<vec4> EmissionPass;

layout(binding = UE3_COMPOSITE_ENCODE_ALBEDO_OUTPUT)
RWTexture2D<vec4> AlbedoOutput;

layout(binding = UE3_COMPOSITE_ENCODE_EMISSION_OUTPUT)
RWTexture2D<vec4> EmissionOutput;

layout(push_constant)
ConstantBuffer<Ue3CompositeEncodeArgs> cb;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uvec2 threadId : SV_DispatchThreadID)
{
  if (any(threadId >= cb.resolution))
  {
    return;
  }

  const vec4 full = Pass[threadId];
  const bool uniformEmission = (cb.flags & UE3_COMPOSITE_ENCODE_FLAG_UNIFORM_EMISSION) != 0;
  const uvec2 emissionTexel = uniformEmission ? uvec2(0, 0) : threadId;

  vec3 emission = vec3(0.0f, 0.0f, 0.0f);
  if ((cb.flags & UE3_COMPOSITE_ENCODE_FLAG_EMISSION_PASS) != 0)
  {
    emission = max(EmissionPass[emissionTexel].rgb, vec3(0.0f, 0.0f, 0.0f));
  }

  float coverage = 1.0f;
  if ((cb.flags & UE3_COMPOSITE_ENCODE_FLAG_COVERAGE_ALPHA) != 0)
  {
    coverage = full.a >= 0.5f ? 1.0f : 0.0f;
  }

  AlbedoOutput[threadId] = vec4(linearToGamma(saturate(full.rgb - emission)), coverage);
  if ((cb.flags & UE3_COMPOSITE_ENCODE_FLAG_STORE_EMISSION) != 0 && (!uniformEmission || all(threadId == uvec2(0, 0))))
  {
    EmissionOutput[emissionTexel] = vec4(linearToGamma(emission), 1.0f);
  }
}
