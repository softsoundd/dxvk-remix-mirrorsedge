/*
* Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
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

// Isotropic hemisphere mean of the sky-view LUT.

#include "rtx/pass/atmosphere/atmosphere_common.slangh"
#include "rtx/utility/sampling.slangh"

layout(binding = 0)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

layout(binding = 1)
Texture2D skyViewLutInput;

layout(rgba16f, binding = 2)
RWTexture2D<float4> hemisphereMeanOutput;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
  const uint kSkyLutSamples = 8;
  float3 skySum = float3(0.0f, 0.0f, 0.0f);
  [unroll]
  for (uint i = 0; i < kSkyLutSamples; ++i) {
    const float u1 = (float(i) + 0.5f) / float(kSkyLutSamples);
    const float u2 = frac(float(i) * silverRatio);
    float unusedPdf;
    const float3 hemiZ = calcUniformHemisphereDirectionSample(float2(u1, u2), unusedPdf);
    const float3 dirYUp = float3(hemiZ.x, hemiZ.z, hemiZ.y);
    skySum += sampleSkyAmbientForVolume(dirYUp, atmosphereArgs, skyViewLutInput);
  }

  hemisphereMeanOutput[uint2(0, 0)] = float4(skySum / float(kSkyLutSamples), 1.0f);
}
