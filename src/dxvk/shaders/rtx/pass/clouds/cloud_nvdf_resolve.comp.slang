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

// Turns the final jump flood seeds into the body distance field: km to the nearest boundary seed, negative
// inside. Raw km in fp16, whose step stays below the voxel size up to the field's largest distances. Centre to
// centre distances put the zero crossing half a voxel out; the profile remap and the conservative stepping
// absorb that. Writes the back buffer, which is swapped in once complete.

#include "rtx/pass/clouds/cloud_binding_indices.h"
#include "rtx/pass/clouds/cloud_nvdf_common.slangh"

layout(binding = CLOUD_BINDING_CONSTANTS)
ConstantBuffer<CloudConstants> constants;

layout(binding = CLOUD_BINDING_BAKE_INPUT)
Texture3D<float> occupancyInput;

layout(binding = CLOUD_BINDING_BAKE_INPUT2)
Texture3D<uint> seedsInput;

layout(binding = CLOUD_BINDING_BAKE_OUTPUT, r16f)
RWTexture3D<float> sdfOutput;

[shader("compute")]
[numthreads(8, 4, 8)]
void main(uint3 threadId : SV_DispatchThreadID) {
  if (any(threadId >= uint3(CLOUD_NVDF_SIZE_XZ, CLOUD_NVDF_SIZE_Y, CLOUD_NVDF_SIZE_XZ))) {
    return;
  }

  const CloudArgs cloud = constants.cloud;
  const uint seed = seedsInput.Load(int4(int3(threadId), 0));

  float sdfKm = kCloudNvdfEmptyKm;
  if (seed != kCloudNvdfSeedInvalid) {
    const float distance = cloudNvdfVoxelDistanceKm(threadId, cloudNvdfUnpackSeed(seed), cloud.tileKm, cloud.thicknessKm);
    const bool inside = occupancyInput.Load(int4(int3(threadId), 0)) > 0.5f;
    sdfKm = inside ? -distance : distance;
  }

  sdfOutput[threadId] = sdfKm;
}
