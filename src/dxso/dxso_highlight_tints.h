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
   * \brief Material highlight tint analysis of a pixel shader
   *
   * Proves which material scalar parameters tint the colour output as lerp(X, X * V, S): the
   * shape Mirror's Edge's Runner Vision highlight (LOI_Strength) compiles to in the world's master
   * materials. Its weapon materials instead only add an unlit glow, which is proven as a glow-only
   * pair. The shader is evaluated symbolically, lane by lane, so the proof holds
   * however fxc scheduled the lerp - mad pairs in either operand order, lrp, or the lerp spread
   * over registers whose other lanes carry unrelated math. See documentation/UE3Compatibility.md,
   * "Runner Vision".
   */

  constexpr uint32_t kDxsoHighlightMaxConstRegs = 256;

  struct DxsoHighlightInputs {
    // material scalar parameters (UE3 UniformScalar_*): the candidate strengths
    std::bitset<kDxsoHighlightMaxConstRegs> scalarRegs;
    // material vector parameters (UE3 UniformVector_*): the candidate tint colours
    std::bitset<kDxsoHighlightMaxConstRegs> vectorRegs;
    // constants that are a lighting factor (ambient and sky colour, lightmap scale)
    std::bitset<kDxsoHighlightMaxConstRegs> lightingConstRegs;
    // samplers whose samples are a lighting factor (lightmaps and their filtering LUT)
    uint32_t lightingSamplerMask = 0;
  };

  enum class DxsoHighlightFailure : uint8_t {
    None = 0,
    // not ps_2_0+ bytecode
    NotPixelShader,
    // branches, loops or calls, which the straight-line evaluation does not model
    FlowControl,
    // a constant read through an address register
    RelativeAddressing,
    // an expression outgrew the term or symbol budget
    TooComplex,
    // oC0.rgb is never written
    NoColorOutput,
  };

  struct DxsoHighlightPair {
    // The strength: a material scalar register, read from its x component.
    uint32_t scalarReg = 0;
    // Per output channel (oC0.r, g, b), the vector register and component the channel is lerped
    // towards. A tint pair is only reported when every channel is tinted; a glow-only pair has none.
    std::array<int32_t, 3> colorReg = { -1, -1, -1 };
    std::array<uint8_t, 3> colorComponent = { 0, 0, 0 };
    // Per output channel, the coefficient of an unlit copy of the colour scaled by the strength. The
    // Runner Vision network adds 0.1 * S * tinted diffuse on top of the lit surface; no other use
    // of a tint lerp has that shape, so it also identifies a tint pair as a highlight.
    std::array<float, 3> glowCoefficient = { 0.0f, 0.0f, 0.0f };
    // No tint: the strength only adds an unlit k * S * texture sample to one channel, a flash of
    // colour, as Mirror's Edge's weapons glow red before a strike (8 * LOI_Strength * a mask's red).
    bool glowOnly = false;
    // A glow-only pair's texture, and the channel of it (x to w) the glow reads.
    int32_t glowSampler = -1;
    uint8_t glowComponent = 0;
  };

  struct DxsoHighlightResult {
    // False when the bytecode could not be evaluated; failure says why.
    bool analyzed = false;
    DxsoHighlightFailure failure = DxsoHighlightFailure::None;
    std::vector<DxsoHighlightPair> pairs;
    // Material scalars that reach the colour output without proving a tint, for diagnostics.
    std::vector<uint32_t> unprovenScalarRegs;
  };

  DxsoHighlightResult analyzeDxsoHighlightTints(
    const uint32_t*             tokens,
    size_t                      tokenCount,
    const DxsoHighlightInputs&  inputs);

  class DxsoCtab;

  // The inputs as UE3 names them in a pixel shader's constant table: UniformScalar_* and
  // UniformVector_* are the material's parameters, AmbientColorAndSkyFactor, Upper/LowerSkyColor
  // and LightMapScale its lighting constants, LightMapTextures and Mirror's Edge's BSplineTexture
  // filtering LUT its lighting samplers.
  DxsoHighlightInputs dxsoHighlightInputsFromUe3Ctab(const DxsoCtab& ctab);

  const char* dxsoHighlightFailureName(DxsoHighlightFailure failure);

}
