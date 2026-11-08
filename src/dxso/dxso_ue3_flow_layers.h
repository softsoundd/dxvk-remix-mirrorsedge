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
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "dxso_opcode_util.h"
#include "dxso_uv_dataflow.h"

namespace dxvk {

  /**
   * \brief The static half of a UE3 flow layer
   *
   * A flow layer renders the normal detail a material moves over its surface: normal maps it pans, or a panned texture
   * that blends between two normal maps. This is what the bytecode alone decides. Whether a read actually moves is only
   * known from the draws. See "Flow layers" in documentation/UE3Compatibility.md.
   */

  // Caps how many panned samples one layer combines.
  constexpr uint32_t kDxsoUe3FlowMaxSamples = 2;
  // Marks a normal lane no texel channel feeds, which the runtime reconstructs from the others.
  constexpr uint8_t kDxsoUe3FlowNoComponent = 0xFFu;

  enum class DxsoUe3FlowRefusal : uint8_t {
    None = 0,
    // The shader is not a ps_2_0+ pixel shader with a constant table.
    NotAnalyzed,
    // No read of a material texture outside the colour has a coordinate offset by a constant register.
    NoPannedSampler,
    // The symbolic evaluation hit flow control, relative addressing or its term budget.
    EvaluationFailed,
    // No panned texture reaches a normalised vector.
    NotNormalized,
    // A panned texture reaches the normal other than as a vector times one weight: squared, through a
    // normalisation of its own, or with its channels mixed across lanes.
    NotLinear,
    // The panned textures are weighted by different masks.
    NotProportional,
    // The panned textures are weighted by more than one texture channel.
    MaskNotSingleChannel,
    // The panned textures carry no positive weight where their mask is at its peak.
    NoWeight,
    // The mask's texture coordinate is not one the runtime can resolve.
    MaskUnresolved,
    // A later normalisation reaches the panned textures through math the evaluation had to fold, which may blend
    // them in by a weight the analysis cannot see.
    FoldedBlend,
  };

  const char* describeDxsoUe3FlowRefusal(DxsoUe3FlowRefusal refusal);

  // Describes one read of a panned texture. A material can read one normal map several times, panned differently, and
  // each read is a sample of its own.
  struct DxsoUe3FlowSample {
    uint8_t sampler = 0;
    // Is the read's instruction index.
    uint32_t site = 0;
    // Holds the read's own texture coordinate.
    PsSamplerUvOrigin uv;
    // Holds the texel channel each normal lane x, y and z reads, or kDxsoUe3FlowNoComponent.
    std::array<uint8_t, 3> components = { 0, 1, 2 };
    // Is the sample's weight in the normal where the mask is at its peak, in units of a decoded (2t - 1) normal.
    // The game sometimes adds a normal map undecoded, which only biases its sum, and the weight leaves the bias out.
    float weight = 0.f;
    std::string name;
  };

  // The mask is saturate(bias + scale * channel) of the sampler's texel.
  struct DxsoUe3FlowMask {
    bool present = false;
    uint8_t sampler = 0;
    // Holds the read's own texture coordinate.
    PsSamplerUvOrigin uv;
    uint8_t channel = 0;
    float scale = 1.f;
    float bias = 0.f;
    std::string name;
  };

  // Describes the panned read that blends a layer between its two samples, as a tarp's wind does. The normal is
  // normalize(lerp(samples[0], samples[1], scale * channel)), with the weight multiplied by the mask when there is one.
  struct DxsoUe3FlowBlend {
    bool present = false;
    uint8_t sampler = 0;
    // Is the read's instruction index.
    uint32_t site = 0;
    // Holds the read's own texture coordinate.
    PsSamplerUvOrigin uv;
    uint8_t channel = 0;
    float scale = 1.f;
    std::string name;
  };

  struct DxsoUe3FlowShaderInfo {
    DxsoUe3FlowRefusal refusal = DxsoUe3FlowRefusal::NotAnalyzed;
    // Names the sampler behind a refusal, for the log.
    std::string refusalDetail;
    // Lists the panned samples, heaviest first, or with a blend, the normal maps it blends from and to.
    std::vector<DxsoUe3FlowSample> samples;
    DxsoUe3FlowMask mask;
    // Is present when the layer blends between two static normal maps rather than adding panned ones.
    DxsoUe3FlowBlend blend;
    // Holds the samplers of the samples and the blend.
    uint32_t flowSamplerMask = 0;
    // Lists the constant registers the coordinates of the samples and the blend are offset by: the panners, and any
    // static tile offset.
    std::vector<uint32_t> offsetRegisters;
    // Lists the offset registers no other read's coordinate reads, so a composite bake that binds the layer's samplers
    // flat does not depend on them.
    std::vector<uint32_t> exclusiveOffsetRegisters;
    // Is set when other textures weigh the samples as well, as a debris layer drawn over the canals does.
    // The layer leaves them out.
    bool ignoresOtherFactors = false;
  };

  DxsoUe3FlowShaderInfo analyzeDxsoUe3FlowLayers(const DxsoShaderView& pixelShader);

}
