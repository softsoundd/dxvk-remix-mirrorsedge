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
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dxvk {

  /**
   * \brief Colour-term analysis of a pixel shader
   *
   * Follows every tracked source (a sampler's samples, a constant register, a `def`
   * literal component) through the arithmetic to the colour output and records, per
   * additive term that reaches it, which classes of lighting factor the term was
   * multiplied by on the way. See documentation/UE3Compatibility.md, "UE3 lightmaps are
   * bypassed" / "Identity", for how the classification is used.
   */

  // Multiplicative factor classes a colour term can carry.
  enum DxsoColorTermFactor : uint8_t {
    // multiplied by a value sampled from a designated lightmap sampler
    DxsoColorTermFactor_Lightmap         = 1u << 0,
    // multiplied by a designated lighting constant (ambient / sky colour)
    DxsoColorTermFactor_LightConst       = 1u << 1,
    // multiplied by a vector read of a TEXCOORD interpolant that is neither a texture
    // coordinate nor normalised (vertex lightmap coefficients)
    DxsoColorTermFactor_Interpolant      = 1u << 2,
    // multiplied by a value derived from normalising such an interpolant (camera / sky vector)
    DxsoColorTermFactor_View             = 1u << 3,
    // multiplied by a view-derived vector's dot product with UE3's LightMapBasis literals:
    // the directional lightmap specular transfer coefficient
    DxsoColorTermFactor_SpecularTransfer = 1u << 4,
  };

  constexpr uint32_t kDxsoColorTermSignatureCount = 32;

  // Bit `s` set: at least one term with factor signature `s` (a DxsoColorTermFactor mask)
  // reaches the colour output.
  using DxsoColorTermSet = uint32_t;

  // Which outputs a source's value reaches at all, colour terms aside.
  enum DxsoColorTermReach : uint8_t {
    DxsoColorTermReach_Color      = 1u << 0,   // oC0.rgb
    DxsoColorTermReach_Opacity    = 1u << 1,   // oC0.a or texkill
    DxsoColorTermReach_Coordinate = 1u << 2,   // a texture coordinate
  };

  constexpr uint32_t kDxsoColorTermMaxSamplers  = 16;
  constexpr uint32_t kDxsoColorTermMaxConstRegs = 256;

  struct DxsoColorTermInputs {
    // samplers whose samples are a lightmap factor
    uint32_t                                 lightmapSamplerMask = 0;
    // constant registers that are a lighting-constant factor (ambient / sky colour)
    std::bitset<kDxsoColorTermMaxConstRegs>  lightingConstRegs;
    // the sources whose colour terms and reach are reported
    uint32_t                                 trackedSamplerMask = 0;
    std::bitset<kDxsoColorTermMaxConstRegs>  trackedConstRegs;
    bool                                     trackLiterals = false;
  };

  struct DxsoColorTermLiteral {
    uint32_t         bits;   // float32 bit pattern of the literal component
    DxsoColorTermSet terms;
  };

  // What a sampler's texture coordinate went through on its way from the interpolant, per lane.
  enum DxsoCoordExpr : uint8_t {
    // any arithmetic at all (the coordinate is not a plain interpolant read)
    DxsoCoordExpr_Arith         = 1u << 0,
    // an additive term that is not itself coordinate data (add/sub/mad/dp2add against a
    // constant, literal or derived value)
    DxsoCoordExpr_Offset        = 1u << 1,
    // frc or sincos: a wrapping or periodic coordinate
    DxsoCoordExpr_Wrap          = 1u << 2,
    // an additive term that is neither a literal, a constant register nor an interpolant - a
    // sampled or otherwise derived value, which the UV resolver cannot express
    DxsoCoordExpr_UnknownOffset = 1u << 3,
  };

  struct DxsoColorTermResult {
    // False when the bytecode is not a ps_2_0+ pixel shader or could not be decoded.
    bool analyzed = false;
    // Some value reaching the colour output was multiplied by a lighting constant. UE3's
    // non-simple base pass always adds `DiffuseColor * AmbientColor`; the simple-lightmap
    // compile never references the constant.
    bool hasLightConstTerm = false;
    std::array<DxsoColorTermSet, kDxsoColorTermMaxSamplers>  samplerTerms = {};
    std::array<DxsoColorTermSet, kDxsoColorTermMaxConstRegs> constTerms = {};
    std::array<uint8_t, kDxsoColorTermMaxSamplers>           samplerReach = {};
    std::array<uint8_t, kDxsoColorTermMaxConstRegs>          constReach = {};
    // Distinct literal component values read by the shader, in first-read order.
    std::vector<DxsoColorTermLiteral> literals;
    // Per sampler, tracked or not, over the coordinate lanes it reads across all its samples:
    // every non-literal constant register the coordinate is built from (transitively, through
    // temporaries, dot products, matrix rows and dependent reads), the DxsoCoordExpr bits, and
    // the sample count. Lane-precise, so math the compiler packed into the other lanes of the
    // coordinate's register is not attributed to it.
    std::array<std::bitset<kDxsoColorTermMaxConstRegs>, kDxsoColorTermMaxSamplers> samplerCoordConstRegs = {};
    std::array<uint8_t, kDxsoColorTermMaxSamplers>  samplerCoordExpr = {};
    std::array<uint16_t, kDxsoColorTermMaxSamplers> samplerSampleCount = {};
  };

  DxsoColorTermResult analyzeDxsoColorTerms(
    const uint32_t*             tokens,
    size_t                      tokenCount,
    const DxsoColorTermInputs&  inputs);

  // Every term carries the specular transfer coefficient and a lightmap, lighting constant
  // or vertex lightmap: UE3's `LightMap * SpecularTransferCoefficients * SpecularColor`,
  // which only the directional-lightmap compile emits. A Fresnel-weighted diffuse blend is
  // view-dependent too but never passes through the basis transfer, and stays colour.
  bool isDxsoColorTermSetSpecularOnly(DxsoColorTermSet set);

  // Every term is a lightmap or vertex-lightmap product and none reaches the output via a
  // lighting constant or unlit: a transfer-coefficient input (TwoSidedLightingMask). Only
  // meaningful in a shader with a lighting-constant term, where every diffuse input also
  // carries `DiffuseColor * AmbientColor`.
  bool isDxsoColorTermSetTransferOnly(DxsoColorTermSet set);

  // Some term reaches the output without any lighting factor at all.
  bool dxsoColorTermSetHasUnlitTerm(DxsoColorTermSet set);

  enum class DxsoColorTermRole : uint8_t {
    // reaches no output at all as a value (normal maps consumed by the lighting transfer)
    Unused = 0,
    // reaches the colour output on a path every lightmap-policy compile has
    Color,
    // reaches only the opacity output (alpha, clip) - policy independent, but not colour
    Opacity,
    // reaches only a texture coordinate - a UV transform, policy independent
    Coordinate,
    // reaches the colour output only through lighting math the simple-lightmap compile strips
    LightingOnly,
  };

  // Colour terms decide Color versus LightingOnly. Without a policy-stable colour path the
  // opacity / coordinate reach names the role, so a lighting input that also drives a
  // coordinate classifies the same way in the compile that has no lighting path at all.
  DxsoColorTermRole classifyDxsoColorTermSource(DxsoColorTermSet set, uint8_t reach, bool shaderHasLightConstTerm);

}
