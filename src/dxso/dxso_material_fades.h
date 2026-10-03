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
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "dxso_highlight_tints.h"

namespace dxvk {

  /**
   * \brief Opacity-driven fade analysis of a pixel shader
   *
   * Finds the material parameters the colour output is affine in, and how UE3's particle colour
   * interpolant reaches it. What the bytecode alone cannot settle - other constants the output
   * depends on, such as a tint that has to be white for a lerp to come to rest - is left in the
   * result for each draw to evaluate. See documentation/UE3Compatibility.md, "Opacity-driven fades".
   */

  enum class DxsoFadeModulatorKind : uint8_t {
    // the x component of a UniformScalar_* register
    MaterialScalar,
    // one component of a UniformVector_* register
    MaterialVector,
  };

  constexpr uint8_t kDxsoFadeColorLanes = 0x7u;
  constexpr uint8_t kDxsoFadeAlphaLane = 0x8u;

  // One term of a lane: a coefficient times float constant components, which the draw supplies,
  // times everything else - samples, interpolants, folded values - which only the monomial key
  // tells apart from the lane's other terms (0 for none).
  struct DxsoFadeTerm {
    static constexpr uint32_t kMaxParameters = 16;

    double   coef = 0.0;
    uint64_t monomial = 0;
    // reg * 4 + component
    std::array<uint16_t, kMaxParameters> parameters = {};
    uint8_t  parameterCount = 0;
  };
  // Sorted by monomial key, so terms that differ only in their constants are adjacent.
  using DxsoFadePoly = std::vector<DxsoFadeTerm>;

  // A material parameter M every lane in laneMask is affine in, each lane as offset + M * slope.
  // Whether that fades the draw depends on the constants the polynomials leave live, so
  // dxsoMaterialFadeCoverage decides it per draw.
  struct DxsoMaterialFade {
    DxsoFadeModulatorKind kind = DxsoFadeModulatorKind::MaterialScalar;
    uint32_t reg = 0;
    uint8_t  component = 0;
    // kDxsoFadeColorLanes or kDxsoFadeAlphaLane
    uint8_t  laneMask = 0;
    std::array<DxsoFadePoly, 4> offset;
    std::array<DxsoFadePoly, 4> slope;
  };

  // How the particle colour interpolant reaches the colour output, in the shapes a vertex colour
  // reproduces through Remix's texture stage operations. A use is reported where every coloured term
  // carrying the channel has that shape; the coloured terms carrying none of it are its residual, and
  // the use holds for a draw only when the residual vanishes for its constants.
  struct DxsoParticleColorUse {
    // the colour's own channel scales each of oC0.r, g and b
    bool tintsColor = false;
    std::array<DxsoFadePoly, 3> tintResidual;
    // the colour's alpha scales oC0.rgb, as UE3's additive blend mode folds the opacity into colour
    bool scalesColor = false;
    std::array<DxsoFadePoly, 3> scalesColorResidual;
    // the colour's alpha scales oC0.a
    bool scalesOpacity = false;
    DxsoFadePoly scalesOpacityResidual;
  };

  struct DxsoMaterialFadeResult {
    // False when the bytecode could not be evaluated; failure says why.
    bool analyzed = false;
    DxsoHighlightFailure failure = DxsoHighlightFailure::None;
    // A modulator's candidates are adjacent, alpha first.
    std::vector<DxsoMaterialFade> fades;
    DxsoParticleColorUse particleColor;
    // Material scalars that reach oC0 without the output being affine in them, for diagnostics.
    std::vector<uint32_t> unprovenScalarRegs;
  };

  // particleColorInputRegister: the ps_3_0 input register (v#) carrying UE3's particle colour, or -1.
  DxsoMaterialFadeResult analyzeDxsoMaterialFades(
    const uint32_t*             tokens,
    size_t                      tokenCount,
    const DxsoHighlightInputs&  inputs,
    int32_t                     particleColorInputRegister);

  // constants holds four floats per float constant register, as the draw sets them.

  // How far the fade's lanes have faded in from restValue, the value the draw's blend leaves the
  // framebuffer unchanged by: 0 at rest, 1 at whichever end of M's [0, 1] range is furthest from it.
  // Nothing unless every lane deviates from restValue as the same multiple of its slope, without
  // crossing restValue inside the range; 0 when every lane already sits at restValue.
  std::optional<float> dxsoMaterialFadeCoverage(
    const DxsoMaterialFade& fade,
    float                   restValue,
    const float*            constants,
    uint32_t                registerCount);

  // Whether every term of the polynomial cancels for these constants.
  bool dxsoFadePolyVanishes(
    const DxsoFadePoly&     poly,
    const float*            constants,
    uint32_t                registerCount);

}
