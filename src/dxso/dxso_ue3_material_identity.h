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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "dxso_highlight_tints.h"
#include "dxso_opcode_util.h"
#include "../dxvk/rtx_render/rtx_constants.h"
#include "../util/xxHash/xxhash.h"

namespace dxvk {

  // The CTAB names material texture parameters Texture2D_* / TextureCube_* and material constants
  // UniformVector_* / UniformScalar_*. Frame-varying uniform expressions share those registers;
  // see "Material identity and replacement anchor stability" in UE3Compatibility.md.

  // Merged (start, count) register ranges of the UniformVector_* constants kept in identity.
  using Ue3MaterialConstRanges = std::vector<std::pair<uint32_t, uint32_t>>;

  struct Ue3PsMaterialIdentityInfo {
    Ue3MaterialConstRanges constRanges;
    // Texture2D_* / TextureCube_* / Texture3D_* samplers carrying the material's identity: its
    // colour and opacity inputs, or its coordinate inputs when they are all it has.
    uint32_t materialSamplerMask = 0;
    // Material samplers read only as lighting inputs; out of identity and of albedo selection.
    uint32_t lightingInputSamplerMask = 0;
    // Ascending. One of these holds the colour of a material without textures.
    std::vector<uint32_t> uniformVectorRegisters;
    // (name, name key, sampler register) per material sampler, ordered by CTAB name. Unlike
    // registers, names are the same in every lightmap-policy compile of a material.
    std::vector<std::tuple<std::string, XXH64_hash_t, uint32_t>> materialSamplersByNameOrder;
    // (name key, first register) per kept uniform, ordered by name.
    std::vector<std::pair<XXH64_hash_t, uint32_t>> namedUniformFirstRegistersByNameOrder;
    // Uniform registers left out of identity, ascending. Diagnostics only.
    std::vector<uint32_t> volatileUniformRegisters;
    // Hash of the material sampler names, which every lightmap-policy compile of a material shares.
    // Without material samplers, it hashes the kept uniform names and the literals reaching the
    // colour unlit (see "Identity" in UE3Compatibility.md).
    XXH64_hash_t canonicalShaderSignature = kEmptyHash;
    bool texturelessSignature = false;
    bool hasCtab = false;

    // The kept and excluded inputs, set when anything was excluded, for
    // rtx.d3d9.ue3LogMaterialInstanceHash.
    std::string identitySummary;

    // Material scalars the shader proves tint its colour as lerp(X, X * V, S): Mirror's Edge's
    // Runner Vision highlight (rtx.d3d9.ue3HighlightTints). nameKey hashes the strength's and
    // colour's CTAB names, which are fixed per material while their registers move between
    // lightmap policy compiles; it keys the at-rest evidence every compile shares.
    struct HighlightPair {
      DxsoHighlightPair tint;
      XXH64_hash_t nameKey = kEmptyHash;
      std::string strengthName;
      std::string colorName;
    };
    std::vector<HighlightPair> highlightPairs;
    // The colour registers of those pairs: what the surface is tinted towards, never its colour.
    std::vector<uint32_t> highlightColorRegisters;
    // For rtx.d3d9.ue3LogHighlightTints.
    DxsoHighlightFailure highlightFailure = DxsoHighlightFailure::None;
    std::vector<uint32_t> highlightUnprovenScalars;
  };

  Ue3PsMaterialIdentityInfo parseUe3PsMaterialIdentityFromCtab(const DxsoShaderView& pixelShader,
                                                               bool detectVolatileConstants);

}
