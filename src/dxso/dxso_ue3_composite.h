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

namespace dxvk {

  /**
   * \brief The static half of a UE3 composite material bake
   *
   * A composite bake (rtx.ue3CompositeMaterialTextures) re-runs a material's own base pass
   * pixel shader over one repeat of its texture layers and renders the result instead of a
   * single picked texture. This is what the bytecode alone decides: which material textures
   * make up the colour, and which engine constants select the lighting the shader was
   * compiled for, so the bake can leave only the material's own colour. See "Composite
   * materials" in documentation/UE3Compatibility.md.
   */

  // Selects how a bake isolates the material colour from the base pass lighting.
  enum class DxsoUe3CompositeLighting : uint8_t {
    // Diffuse arrives as DiffuseColor * AmbientColorAndSkyFactor.rgb, as in the unlit view, on movable
    // geometry and in the directional lightmap compiles. A unit ambient with the lightmap scaled to zero
    // leaves Emissive + Diffuse.
    AmbientConstant,
    // Is the simple texture lightmap compile, whose diffuse only arrives through pow(LightMap, DiffusePower).
    // A white lightmap at unit scale leaves Emissive + Diffuse.
    SimpleLightmap,
    // Is the simple vertex lightmap compile, whose diffuse only arrives multiplied by one lighting interpolant.
    // A white interpolant leaves Emissive + Diffuse.
    SimpleVertexLightmap,
    // Is an unlit material, whose whole output is its colour.
    Unlit,
  };

  enum class DxsoUe3CompositeRefusal : uint8_t {
    None = 0,
    // The shader is not a ps_2_0+ pixel shader with a constant table.
    NotAnalyzed,
    // No material texture reaches the colour output.
    NoColorLayer,
    // An engine texture, such as scene colour, scene depth or light attenuation, reaches an output.
    EngineSampler,
    // A volume texture reaches an output.
    VolumeLayer,
    // The shader is the directional vertex lightmap compile. Its diffuse arrives through several lighting
    // interpolants weighted by the normal, which no single value turns into a unit factor.
    VertexLightmap,
    // The colour reads a vertex colour, which has no single value over a texture repeat.
    VertexColor,
    // The shader clip()s, but its alpha output is not an interpolant the bake can set to mark coverage.
    MaskedWithoutCoverage,
  };

  const char* describeDxsoUe3CompositeRefusal(DxsoUe3CompositeRefusal refusal);

  const char* describeDxsoUe3CompositeLighting(DxsoUe3CompositeLighting lighting);

  // Describes a material sampler that reaches the colour or opacity output, or the coordinate of a sampler
  // that does, such as a bump offset height map or a normal map steering a reflection lookup.
  // Each one's coordinate has to repeat over the bake for the bake to tile.
  struct DxsoUe3CompositeLayer {
    uint8_t sampler = 0;
    bool isCube = false;

    // These say whether the layer reaches the colour output unlit, and whether it reaches it through the lighting.
    bool emissive = false;
    bool diffuse = false;

    // Is set when the layer reaches oC0.a or texkill.
    bool opacity = false;
    std::string name;

    bool contributesColor() const { return emissive || diffuse; }
  };

  // Describes a material sampler that only feeds the lighting, such as a normal map. The bake does not render it,
  // but its repeat covers the layer's tiling where it can, so a normal map authored over the bake lines up with it.
  struct DxsoUe3CompositeLightingLayer {
    uint8_t sampler = 0;
    std::string name;
  };

  // Describes a colour layer the colour output is a polynomial of degree 1 or 2 in, as oC0.rgb = P + m * S (+ m^2 * Q).
  // Here m is one channel of the layer's only read, and none of P, S and Q reads its texture.
  // See "Masks" in UE3Compatibility.md.
  struct DxsoUe3CompositeMaskLayer {
    uint8_t sampler = 0;
    uint8_t channel = 0;
    uint8_t degree = 1;
  };

  struct DxsoUe3CompositeShaderInfo {
    DxsoUe3CompositeRefusal refusal = DxsoUe3CompositeRefusal::NotAnalyzed;
    // Names the engine sampler or volume layer behind a refusal, for the log.
    std::string refusalDetail;
    DxsoUe3CompositeLighting lighting = DxsoUe3CompositeLighting::Unlit;
    std::vector<DxsoUe3CompositeLayer> layers;
    std::vector<DxsoUe3CompositeLightingLayer> lightingLayers;
    // Lists the colour layers that are masks and reach neither the alpha, a coordinate nor, in a lit compile, the emission.
    std::vector<DxsoUe3CompositeMaskLayer> maskLayers;
    // Is set when some term reaches the colour output unlit, so a second pass with the lighting zeroed
    // separates the emission from the diffuse.
    bool hasEmissive = false;
    // Is set when some colour term depends on the camera or reflection vector, which the bake sees face-on.
    bool viewDependent = false;
    bool hasTexkill = false;
    // Is set when COLOR0/COLOR1 carry the tangent basis rather than a vertex colour (WORLD_COORDS).
    bool worldCoords = false;
    // oC0.w is a plain read of this input register's w lane, UE3's PixelPosition.w. The bake sets it to 1
    // so the alpha of a clip()ped material marks its coverage.
    int32_t alphaInputRegister = -1;
    int32_t ambientColorAndSkyFactorRegister = -1;
    int32_t upperSkyColorRegister = -1;
    int32_t lowerSkyColorRegister = -1;
    int32_t lightMapScaleRegister = -1;
    uint32_t lightMapScaleCount = 0;
    // Holds the LightMapTextures[] samplers, which are bound white for the simple lightmap compile's diffuse pass.
    uint32_t lightmapTextureSamplerMask = 0;
    // Holds the simple vertex lightmap compile's lighting interpolant, which is white in the diffuse pass and black
    // in the emission pass. It has the bit layout of DxsoColorTermResult::interpolantFactorInputs.
    uint32_t vertexLightmapInputMask = 0;
    // Holds, per TEXCOORD index, the lanes the colour reads by arithmetic rather than only as a texture coordinate.
    std::array<uint8_t, 8> colorTexcoordLanes = {};
    // Lists the UniformVector_* and UniformScalar_* registers that reach an output, whose live values make a bake current.
    std::vector<uint32_t> materialConstantRegisters;
  };

  DxsoUe3CompositeShaderInfo analyzeDxsoUe3CompositeShader(const DxsoShaderView& pixelShader);

  // Caps how many repeats of its coarsest layer a bake may cover. A long repeat costs detail rather than
  // memory, since the bake's resolution is capped.
  constexpr uint32_t kUe3CompositeMaxPeriodMultiple = 32;

  // Returns the smallest period P = k / minScale, k = 1..maxMultiple, at which every scale * P is a whole
  // number, so every layer repeats a whole number of times over [0, P].
  // Returns 0 when there is none.
  // Scales are the absolute UV scales of the layers along one axis. `tolerance` is a fixed fraction of one
  // repeat where the bake would wrap, so a longer period cannot pass off a near miss.
  float solveUe3CompositePeriod(const float* scales, uint32_t count, uint32_t maxMultiple = kUe3CompositeMaxPeriodMultiple,
                                float tolerance = 1e-3f);

  // Returns the axis resolution of a bake, the largest number of texels any layer spans over one period,
  // rounded up to a power of two and clamped to [minResolution, maxResolution].
  uint32_t solveUe3CompositeResolution(const float* texelsPerPeriod, uint32_t count, uint32_t minResolution, uint32_t maxResolution);

}
