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
#include <vector>

#include "dxso_opcode_util.h"

namespace dxvk {

  constexpr uint8_t kPsSamplerSemanticEngineAuxiliary = 1u << 0;
  constexpr uint8_t kPsSamplerSemanticLightmap        = 1u << 1;
  constexpr uint8_t kPsSamplerSemanticMaterialTexture = 1u << 2;
  constexpr uint8_t kPsSamplerSemanticNonDiffuse      = 1u << 3;
  constexpr uint8_t kPsSamplerSemanticVideo           = 1u << 4;
  constexpr uint8_t kPsSamplerSemanticMovieTexture    = 1u << 5;

  // Hints inferred from the dataflow around a sampler's coordinate and sampled value.
  constexpr uint16_t kPsSamplerExprUvTransform = 1u << 0;
  constexpr uint16_t kPsSamplerExprUvOffset    = 1u << 1;
  constexpr uint16_t kPsSamplerExprUvAnimated  = 1u << 2;
  constexpr uint16_t kPsSamplerExprBlendMath   = 1u << 3;
  constexpr uint16_t kPsSamplerExprUvTimeDriven = 1u << 4;
  constexpr uint16_t kPsSamplerExprViewDependent = 1u << 5;
  constexpr uint16_t kPsSamplerExprMaskControl = 1u << 6;
  constexpr uint16_t kPsSamplerExprColorContribution = 1u << 7;
  // Decoded as a tangent-space normal (`t * 2 - 1`, nrm or a self dot product), as UE3 does for
  // every Normal-input texture.
  constexpr uint16_t kPsSamplerExprNormalDecode = 1u << 8;
  // Reaches oC0.rgb as a colour term rather than only feeding coordinate or lighting math.
  constexpr uint16_t kPsSamplerExprReachesOutputColor = 1u << 9;
  // Multiplied, directly or transitively, by a lightmap sample or by UE3's ambient and sky colour
  // constants. The UE3 base pass modulates only the material's diffuse this way.
  constexpr uint16_t kPsSamplerExprDiffuseAnchor = 1u << 10;

  struct PsSamplerTexcoordInference {
    int32_t texcoord = -1;
    bool coordCompValid = false;
    uint8_t coordCompU = 0;
    uint8_t coordCompV = 1;
    uint16_t sampleCount = 0;
    uint8_t semanticFlags = 0;
    uint16_t expressionFlags = 0;
    int32_t scaleConstReg = -1;
    uint8_t scaleConstCompU = 0;
    uint8_t scaleConstCompV = 1;
    float scaleFactorU = 1.0f;
    float scaleFactorV = 1.0f;
    bool scaleImmediateValid = false;
    float scaleImmediateU = 1.0f;
    float scaleImmediateV = 1.0f;
    int32_t offsetConstReg = -1;
    uint8_t offsetConstCompU = 0;
    uint8_t offsetConstCompV = 1;
    float offsetFactorU = 1.0f;
    float offsetFactorV = 1.0f;
    bool offsetImmediateValid = false;
    float offsetImmediateU = 0.0f;
    float offsetImmediateV = 0.0f;
    // Every float constant register the coordinate depends on transitively, ascending, excluding
    // `def` literals. Unlike scaleConstReg and offsetConstReg it covers any expression shape.
    std::vector<uint32_t> coordConstRegs;
  };

  struct PsTexcoordScaleHint {
    int32_t constReg = -1;
    uint8_t compU = 0;
    uint8_t compV = 1;
    float scaleFactorU = 1.0f;
    float scaleFactorV = 1.0f;
    bool immediateValid = false;
    float immediateU = 1.0f;
    float immediateV = 1.0f;
    int32_t offsetConstReg = -1;
    uint8_t offsetCompU = 0;
    uint8_t offsetCompV = 1;
    float offsetFactorU = 1.0f;
    float offsetFactorV = 1.0f;
    bool offsetImmediateValid = false;
    float offsetImmediateU = 0.0f;
    float offsetImmediateV = 0.0f;
  };

  uint8_t classifyPixelSamplerSemanticFlags(const std::string& samplerName);

  // The TEXCOORD set a pixel shader samples a sampler with, the UV tiling and offset applied to it,
  // and how the sampled value is used.
  PsSamplerTexcoordInference inferPixelShaderTexcoordForSampler(const DxsoShaderView& pixelShader, uint32_t samplerIdx);

  bool isUe3LightingInputSampler(const PsSamplerTexcoordInference& inferred);

}
