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

// Lays a UE3 composite bake out over several of its repeats for a capture, whose material has a single albedo and no
// mask layer. The mask is blended in at its own tiling, as the surface blends it per hit, so the capture shows it.
// Kept emission has no mask and is only repeated, to line up with the albedo. See "Masks" in UE3Compatibility.md.

#include "rtx/pass/ue3_composite/ue3_composite_flatten.h"
#include "rtx/concept/surface_material/mask_layer_blend.slangh"

layout(binding = UE3_COMPOSITE_FLATTEN_SOURCE_INPUT)
Sampler2D Source;

layout(binding = UE3_COMPOSITE_FLATTEN_MASKED_INPUT)
Sampler2D MaskedAlbedo;

layout(binding = UE3_COMPOSITE_FLATTEN_MASKED_HALF_INPUT)
Sampler2D MaskedHalfAlbedo;

layout(binding = UE3_COMPOSITE_FLATTEN_MASK_INPUT)
Sampler2D Mask;

layout(binding = UE3_COMPOSITE_FLATTEN_OUTPUT)
RWTexture2D<vec4> Output;

layout(push_constant)
ConstantBuffer<Ue3CompositeFlattenArgs> cb;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uvec2 threadId : SV_DispatchThreadID)
{
  if (any(threadId >= cb.resolution))
  {
    return;
  }

  // An output texel spans this much of the bake's UV, so the reads filter as the surface's do at the output's density.
  const vec2 texelSize = cb.repeat / vec2(cb.resolution);
  const vec2 bakeUv = (vec2(threadId) + 0.5f) * texelSize;
  const vec2 gradientX = vec2(texelSize.x, 0.0f);
  const vec2 gradientY = vec2(0.0f, texelSize.y);

  const vec4 source = Source.SampleGrad(bakeUv, gradientX, gradientY);
  if ((cb.flags & UE3_COMPOSITE_FLATTEN_FLAG_MASK) == 0)
  {
    Output[threadId] = source;
    return;
  }

  const vec2 maskScale = cb.maskTransform.xy;
  const vec4 maskSample = Mask.SampleGrad(bakeUv * maskScale + cb.maskTransform.zw, gradientX * maskScale, gradientY * maskScale);
  const float mask = maskLayerDecode(maskSample, cb.maskChannel, (cb.flags & UE3_COMPOSITE_FLATTEN_FLAG_MASK_SRGB) != 0);

  const bool quadratic = (cb.flags & UE3_COMPOSITE_FLATTEN_FLAG_QUADRATIC) != 0;
  const vec3 maskedAlbedo = MaskedAlbedo.SampleGrad(bakeUv, gradientX, gradientY).rgb;
  const vec3 maskedHalfAlbedo = quadratic
    ? MaskedHalfAlbedo.SampleGrad(bakeUv, gradientX, gradientY).rgb
    : vec3(0.0f);

  // The blend leaves the alpha alone, as the surface's does.
  Output[threadId] = vec4(maskLayerBlendAlbedo(source.rgb, maskedAlbedo, maskedHalfAlbedo, quadratic, mask), source.a);
}
