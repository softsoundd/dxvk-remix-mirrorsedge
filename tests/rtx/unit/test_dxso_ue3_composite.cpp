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

// Tests the static half of a UE3 composite material bake (rtx.ue3CompositeMaterialTextures), which finds the
// material textures that make up a pixel shader's colour and the lighting constants the bake neutralises,
// and the repeat and resolution solvers the per-draw domain uses. The hand-assembled ceiling case is the
// instruction stream fxc emitted for the Escape_p office ceiling under viewmode unlit.
//
// Run as test_dxso_ue3_composite <shader.dxso> [...], it prints the composite analysis of pixel shaders
// dumped through DXVK_SHADER_DUMP_PATH, with each layer's UV origin and tiling, the mask layers,
// and the repeat the immediate tilings solve to, with and without the masks.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../../src/dxso/dxso_ue3_composite.h"
#include "../../../src/dxso/dxso_uv_dataflow.h"
#include "../../../src/util/log/log.h"
#include "../../../src/util/util_string.h"

#include "dxso_test_assembler.h"

namespace dxvk {
  // Standalone executables own the logger singleton, as the d3d9/dxgi entry points do.
  Logger Logger::s_instance("test_dxso_ue3_composite.log", LogLevel::None);
}

using namespace dxvk;
using namespace dxvk::dxso_test;

namespace {

  constexpr uint16_t kFloat4 = kD3dxRegisterSetFloat4;
  constexpr uint16_t kSampler = kD3dxRegisterSetSampler;
  constexpr uint32_t MaskXW = 0x9u;
  constexpr uint32_t MaskYZW = 0xEu;

  void check(bool condition, const std::string& what) {
    if (!condition) {
      throw std::runtime_error(what);
    }
  }

  const DxsoUe3CompositeLayer* findLayer(const DxsoUe3CompositeShaderInfo& info, uint32_t sampler) {
    for (const DxsoUe3CompositeLayer& layer : info.layers) {
      if (layer.sampler == sampler) {
        return &layer;
      }
    }
    return nullptr;
  }

  // Builds the Escape_p ceiling, a 1x mask choosing between vents (Texture2D_6 * Texture2D_7) and panels
  // (UniformVector_2 * Texture2D_7, with a 32x Texture2D_8 where the 4x Texture2D_4 says so), lit by
  // AmbientColorAndSkyFactor, plus lights emitting 200 * UniformVector_1 * Texture2D_4 * Texture2D_5.
  DxsoTestShader buildCeiling() {
    DxsoTestShader ps(kPs30Header);
    ps.ctab({
        { "UniformVector_0", kFloat4, 0 },
        { "UniformVector_1", kFloat4, 2 },
        { "UniformVector_2", kFloat4, 3 },
        { "AmbientColorAndSkyFactor", kFloat4, 4 },
        { "Texture2D_2", kSampler, 0 },
        { "Texture2D_4", kSampler, 1 },
        { "Texture2D_5", kSampler, 2 },
        { "Texture2D_6", kSampler, 3 },
        { "Texture2D_7", kSampler, 4 },
        { "Texture2D_8", kSampler, 5 },
      })
      .def(1, 200.0f, 32.0f, 4.0f, 1.0f)
      .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
      .dclInput(DxsoUsage::Texcoord, 5, 1, MaskW);
    for (uint32_t s = 0; s < 6; s++) {
      ps.dclSampler(s);
    }
    ps.op2(DxsoOpcode::Mul, rd(0, MaskXY), c(1, kYYYY), v(0))
      .texld(rd(1), r(0), 5)
      .texld(rd(0), r(0), 2)
      .op2(DxsoOpcode::Mul, rd(2, MaskXY), c(1, kZZZZ), v(0))
      .texld(rd(3), r(2), 3)
      .texld(rd(4), v(0), 0)
      .op3(DxsoOpcode::Lrp, rd(5, MaskXYZ), r(4, kXXXX), r(3), c(3))
      .texld(rd(3), r(2), 4)
      .texld(rd(2), r(2), 1)
      .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(5), r(3, kXYZW, kModNeg), r(1))
      .op2(DxsoOpcode::Mul, rd(4, MaskYZW), r(5, swz(0, 0, 1, 2)), r(3, swz(0, 0, 1, 2)))
      .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(2, kXXXX), r(1), r(4, swz(1, 2, 3, 3)))
      .op3(DxsoOpcode::Mad, rd(3, MaskXYZ), r(5), r(3), r(1, kXYZW, kModNeg))
      .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(4, kXXXX), r(3), r(1))
      .op1(DxsoOpcode::Mov, rd(3, MaskXW), c(1))
      .op2(DxsoOpcode::Add, rd(3, MaskYZW), r(3, kWWWW), c(0, swz(0, 0, 1, 2), kModNeg))
      .op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(1), r(3, swz(1, 2, 3, 3)))
      .op2(DxsoOpcode::Mul, rd(3, MaskXYZ), r(3, kXXXX), c(2))
      .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(0), r(3))
      .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(2), r(0))
      .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), r(4, kXXXX), r(0, kXYZW, kModNeg), r(0))
      .op2(DxsoOpcode::Add, rd(0, MaskXYZ), r(0), c(0))
      .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(1), c(4), r(0))
      .op1(DxsoOpcode::Mov, oC0(MaskW), v(1, kWWWW));
    return ps;
  }

  void testCeiling() {
    DxsoTestShader ps = buildCeiling();
    const DxsoUe3CompositeShaderInfo info = analyzeDxsoUe3CompositeShader(ps.view());
    check(info.refusal == DxsoUe3CompositeRefusal::None,
          str::format("ceiling refused: ", describeDxsoUe3CompositeRefusal(info.refusal), " ", info.refusalDetail));
    check(info.lighting == DxsoUe3CompositeLighting::AmbientConstant, "ceiling lighting is the ambient constant");
    check(info.hasEmissive, "ceiling emits (the lights)");
    check(!info.hasTexkill, "ceiling does not clip");
    check(!info.worldCoords, "ceiling is not WORLD_COORDS");
    check(info.alphaInputRegister == 1, "ceiling alpha is PixelPosition.w (v1)");
    check(info.ambientColorAndSkyFactorRegister == 4, "AmbientColorAndSkyFactor is c4");
    check(info.layers.size() == 6, str::format("ceiling has 6 layers, found ", info.layers.size()));
    for (uint32_t s = 0; s < 6; s++) {
      const DxsoUe3CompositeLayer* layer = findLayer(info, s);
      check(layer != nullptr && layer->contributesColor(), str::format("ceiling s", s, " contributes colour"));
    }
    check(findLayer(info, 2)->emissive && !findLayer(info, 2)->diffuse, "Texture2D_5 only emits");
    for (uint32_t s : { 3u, 4u, 5u }) {
      check(findLayer(info, s)->diffuse && !findLayer(info, s)->emissive, str::format("ceiling s", s, " is diffuse only"));
    }
    check(findLayer(info, 0)->diffuse && findLayer(info, 0)->emissive, "the mask feeds both diffuse and emission");
    check(info.materialConstantRegisters == std::vector<uint32_t>({ 0, 2, 3 }), "material constants are c0, c2, c3");

    // The layers tile 1, 4, 4, 32, 4 and 32 times. One repeat of the coarsest covers them all,
    // and the 32x 128px layers need 4096 texels across it.
    const float scales[] = { 1.0f, 4.0f, 32.0f, 4.0f, 4.0f, 32.0f };
    check(solveUe3CompositePeriod(scales, 6) == 1.0f, "ceiling repeats every 1 UV");
    const float texels[] = { 128.0f, 1024.0f, 4096.0f, 1024.0f, 2048.0f, 4096.0f };
    check(solveUe3CompositeResolution(texels, 6, 64, 4096) == 4096, "ceiling bakes at 4096");
    check(solveUe3CompositeResolution(texels, 6, 64, 2048) == 2048, "the resolution cap holds");
  }

  void testSimpleLightmap() {
    // The DirectionalLightmaps=False compile outputs Diffuse * pow(LightMap, DiffusePower) with no ambient term.
    DxsoTestShader ps(kPs30Header);
    ps.ctab({
        { "LightMapScale", kFloat4, 1 },
        { "Texture2D_0", kSampler, 0 },
        { "LightMapTextures", kSampler, 1 },
      })
      .dclInput(DxsoUsage::Texcoord, 0, 0, MaskAll)
      .dclSampler(0)
      .dclSampler(1)
      .texld(rd(0), v(0), 0)
      .texld(rd(1), v(0, swz(2, 3, 2, 3)), 1)
      .op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(1), c(1))
      .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(0), r(1))
      .op1(DxsoOpcode::Mov, oC0(MaskW), v(0, kWWWW));
    const DxsoUe3CompositeShaderInfo info = analyzeDxsoUe3CompositeShader(ps.view());
    check(info.refusal == DxsoUe3CompositeRefusal::None, "simple lightmap compile bakes");
    check(info.lighting == DxsoUe3CompositeLighting::SimpleLightmap, "simple lightmap compile uses the white lightmap");
    check(info.lightmapTextureSamplerMask == (1u << 1), "the lightmap is s1");
    check(info.lightMapScaleRegister == 1 && info.lightMapScaleCount == 1, "LightMapScale is c1");
    check(!info.hasEmissive, "nothing reaches the colour unlit");
    check(info.layers.size() == 1 && info.layers[0].sampler == 0, "only the material texture is a layer");
  }

  void testSimpleVertexLightmap() {
    // Scraper_p's buildings draw with the simple vertex lightmap compile, which outputs Emissive + max(|LightMapA|, 1e-4) * Diffuse.
    // The lighting is in TEXCOORD2 and the emissive layer is on the mesh's other UV channel.
    DxsoTestShader ps(kPs30Header);
    ps.ctab({
        { "UniformVector_0", kFloat4, 0 },
        { "Texture2D_0", kSampler, 0 },
        { "Texture2D_1", kSampler, 1 },
      })
      .def(1, 0.0001f, 0.0f, 0.0f, 0.0f)
      .dclInput(DxsoUsage::Texcoord, 0, 0, MaskAll)
      .dclInput(DxsoUsage::Texcoord, 2, 1, MaskXYZ)
      .dclInput(DxsoUsage::Texcoord, 5, 2, MaskW)
      .dclSampler(0)
      .dclSampler(1)
      .texld(rd(0), v(0, swz(3, 2, 2, 3)), 0)
      .texld(rd(1), v(0), 1)
      .op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(1), c(0))
      .op2(DxsoOpcode::Max, rd(2, MaskXYZ), v(1, kXYZW, kModAbs), c(1, kXXXX))
      .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(2), r(0), r(1))
      .op1(DxsoOpcode::Mov, oC0(MaskW), v(2, kWWWW));
    const DxsoUe3CompositeShaderInfo info = analyzeDxsoUe3CompositeShader(ps.view());
    check(info.refusal == DxsoUe3CompositeRefusal::None,
          str::format("simple vertex lightmap refused: ", describeDxsoUe3CompositeRefusal(info.refusal)));
    check(info.lighting == DxsoUe3CompositeLighting::SimpleVertexLightmap, "the lighting is the vertex lightmap interpolant");
    check(info.vertexLightmapInputMask == (1u << 1), "the vertex lightmap is v1 (TEXCOORD2)");
    check(info.hasEmissive, "the second layer emits");
    check(findLayer(info, 0) != nullptr && findLayer(info, 0)->diffuse && !findLayer(info, 0)->emissive, "s0 is diffuse");
    check(findLayer(info, 1) != nullptr && findLayer(info, 1)->emissive && !findLayer(info, 1)->diffuse, "s1 only emits");

    // With two lighting interpolants, as the directional vertex lightmap has, no one value is a unit factor.
    DxsoTestShader directional(kPs30Header);
    directional.ctab({ { "Texture2D_0", kSampler, 0 } })
      .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
      .dclInput(DxsoUsage::Texcoord, 2, 1, MaskXYZ)
      .dclInput(DxsoUsage::Texcoord, 3, 2, MaskXYZ)
      .dclSampler(0)
      .texld(rd(0), v(0), 0)
      .op2(DxsoOpcode::Add, rd(1, MaskXYZ), v(1), v(2))
      .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(1), r(0));
    check(analyzeDxsoUe3CompositeShader(directional.view()).refusal == DxsoUe3CompositeRefusal::VertexLightmap,
          "several lighting interpolants are refused");
  }

  void testUnlit() {
    DxsoTestShader ps(kPs30Header);
    ps.ctab({ { "Texture2D_0", kSampler, 0 } })
      .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
      .dclSampler(0)
      .texld(rd(0), v(0), 0)
      .op1(DxsoOpcode::Mov, oC0(MaskAll), r(0));
    const DxsoUe3CompositeShaderInfo info = analyzeDxsoUe3CompositeShader(ps.view());
    check(info.refusal == DxsoUe3CompositeRefusal::None, "unlit material bakes");
    check(info.lighting == DxsoUe3CompositeLighting::Unlit, "an unlit material's output is its colour");
  }

  void testMasked() {
    // A clip()ped shader that passes its alpha through from PixelPosition.w marks its coverage in the alpha.
    DxsoTestShader ps(kPs30Header);
    ps.ctab({ { "AmbientColorAndSkyFactor", kFloat4, 0 }, { "Texture2D_0", kSampler, 0 } })
      .def(1, 0.5f, 0.0f, 0.0f, 0.0f)
      .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
      .dclInput(DxsoUsage::Texcoord, 5, 1, MaskW)
      .dclSampler(0)
      .texld(rd(0), v(0), 0)
      .op2(DxsoOpcode::Add, rd(1), r(0, kWWWW), c(1, kXXXX, kModNeg))
      .texkill(rd(1))
      .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(0), c(0))
      .op1(DxsoOpcode::Mov, oC0(MaskW), v(1, kWWWW));
    DxsoUe3CompositeShaderInfo info = analyzeDxsoUe3CompositeShader(ps.view());
    check(info.refusal == DxsoUe3CompositeRefusal::None, "masked material with a passthrough alpha bakes");
    check(info.hasTexkill && info.alphaInputRegister == 1, "coverage comes from v1.w");

    // When the shader computes its alpha, nothing the bake can set marks coverage.
    DxsoTestShader computed(kPs30Header);
    computed.ctab({ { "AmbientColorAndSkyFactor", kFloat4, 0 }, { "Texture2D_0", kSampler, 0 } })
      .def(1, 0.5f, 0.0f, 0.0f, 0.0f)
      .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
      .dclSampler(0)
      .texld(rd(0), v(0), 0)
      .op2(DxsoOpcode::Add, rd(1), r(0, kWWWW), c(1, kXXXX, kModNeg))
      .texkill(rd(1))
      .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(0), c(0))
      .op1(DxsoOpcode::Mov, oC0(MaskW), r(0, kWWWW));
    info = analyzeDxsoUe3CompositeShader(computed.view());
    check(info.refusal == DxsoUe3CompositeRefusal::MaskedWithoutCoverage, "masked material without a passthrough alpha is refused");
  }

  enum class MaskUse { Tint, AlsoAlpha, Squared, Cubed, Folded };

  // Builds Boat_p's deck without its reflection, which is Texture2D_1 at 6x times Texture2D_0, tinted by
  // lerp(1, UniformVector_1, Texture2D_2.x) at 0.85x, plus an untinted 0.4 * UniformVector_0, all lit
  // by AmbientColorAndSkyFactor.
  DxsoTestShader buildDeck(const MaskUse use) {
    DxsoTestShader ps(kPs30Header);
    ps.ctab({
        { "UniformVector_0", kFloat4, 0 },
        { "UniformVector_1", kFloat4, 2 },
        { "AmbientColorAndSkyFactor", kFloat4, 3 },
        { "Texture2D_0", kSampler, 0 },
        { "Texture2D_1", kSampler, 1 },
        { "Texture2D_2", kSampler, 2 },
      })
      .def(1, 6.0f, 0.85f, 0.4f, 1.0f)
      .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
      .dclInput(DxsoUsage::Texcoord, 5, 1, MaskW)
      .dclSampler(0)
      .dclSampler(1)
      .dclSampler(2)
      .op2(DxsoOpcode::Mul, rd(0, MaskXY), c(1, kXXXX), v(0))
      .texld(rd(1), r(0), 1)
      .texld(rd(2), v(0), 0)
      .op2(DxsoOpcode::Mul, rd(0, MaskXY), c(1, kYYYY), v(0))
      .texld(rd(3), r(0), 2);
    if (use == MaskUse::Squared) {
      ps.op2(DxsoOpcode::Mul, rd(3, MaskX), r(3, kXXXX), r(3, kXXXX));
    } else if (use == MaskUse::Cubed) {
      ps.op2(DxsoOpcode::Mul, rd(4, MaskX), r(3, kXXXX), r(3, kXXXX))
        .op2(DxsoOpcode::Mul, rd(3, MaskX), r(4, kXXXX), r(3, kXXXX));
    } else if (use == MaskUse::Folded) {
      ps.op1(DxsoOpcode::Rsq, rd(3, MaskX), r(3, kXXXX));
    }
    ps.op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(1), r(2))
      .op3(DxsoOpcode::Mad, rd(2, MaskXYZ), c(2), r(1), r(1, kXYZW, kModNeg))
      .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(3, kXXXX), r(2), r(1))
      .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), c(0), c(1, kZZZZ), r(1))
      .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(1), c(3))
      .op1(DxsoOpcode::Mov, oC0(MaskW), use == MaskUse::AlsoAlpha ? r(3, kXXXX) : v(1, kWWWW));
    return ps;
  }

  void testMaskLayers() {
    const DxsoUe3CompositeShaderInfo tint = analyzeDxsoUe3CompositeShader(buildDeck(MaskUse::Tint).view());
    check(tint.refusal == DxsoUe3CompositeRefusal::None, "the deck bakes");
    check(tint.layers.size() == 3, str::format("the deck has 3 layers, found ", tint.layers.size()));
    check(tint.maskLayers.size() == 1 && tint.maskLayers[0].sampler == 2 && tint.maskLayers[0].channel == 0 &&
          tint.maskLayers[0].degree == 1,
          "Texture2D_2.x is the deck's only mask, an affine one: the textures read through three channels are not masks");

    check(analyzeDxsoUe3CompositeShader(buildDeck(MaskUse::AlsoAlpha).view()).maskLayers.empty(),
          "a mask that also reaches the alpha is not lifted");
    const DxsoUe3CompositeShaderInfo squared = analyzeDxsoUe3CompositeShader(buildDeck(MaskUse::Squared).view());
    check(squared.maskLayers.size() == 1 && squared.maskLayers[0].degree == 2, "a squared mask is a quadratic one");
    check(analyzeDxsoUe3CompositeShader(buildDeck(MaskUse::Cubed).view()).maskLayers.empty(),
          "a cubed mask is not lifted");
    check(analyzeDxsoUe3CompositeShader(buildDeck(MaskUse::Folded).view()).maskLayers.empty(),
          "a mask through rsq is not a polynomial");

    // Escape_p's RPA carpet computes lerp(UniformVector_1 * Texture2D_2, UniformVector_2 * m * Texture2D_2, m)
    // with m = Texture2D_1.x at uv * UniformScalar_1 - UniformScalar_2. The mask weights the lerp and scales
    // what it lerps towards, and the tiling is computed in the register the coordinate is then written to.
    DxsoTestShader carpet(kPs30Header);
    carpet.ctab({
        { "UniformVector_0", kFloat4, 0 },
        { "UniformVector_1", kFloat4, 2 },
        { "UniformVector_2", kFloat4, 3 },
        { "UniformScalar_1", kFloat4, 4 },
        { "UniformScalar_2", kFloat4, 5 },
        { "AmbientColorAndSkyFactor", kFloat4, 6 },
        { "Texture2D_1", kSampler, 0 },
        { "Texture2D_2", kSampler, 1 },
      })
      .def(1, 1.0f, 0.0f, 0.0f, 0.0f)
      .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
      .dclInput(DxsoUsage::Texcoord, 5, 1, MaskW)
      .dclSampler(0)
      .dclSampler(1)
      .op1(DxsoOpcode::Mov, rd(0, MaskX), c(4, kXXXX))
      .op3(DxsoOpcode::Mad, rd(0, MaskXY), r(0, kXXXX), v(0), c(5, kXXXX, kModNeg))
      .texld(rd(0), r(0), 0)
      .texld(rd(1), v(0), 1)
      .op2(DxsoOpcode::Mul, rd(0, MaskYZW), r(0, kXXXX), r(1, swz(0, 0, 1, 2)))
      .op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(1), c(2))
      .op3(DxsoOpcode::Mad, rd(0, MaskYZW), c(3, swz(0, 0, 1, 2)), r(0), r(1, swz(0, 0, 1, 2), kModNeg))
      .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), r(0, kXXXX), r(0, swz(1, 2, 3, 3)), r(1))
      .op1(DxsoOpcode::Mov, rd(1, MaskXYZ), c(0))
      .op2(DxsoOpcode::Add, rd(2, MaskXYZ), r(1, kXYZW, kModNeg), c(1, kXXXX))
      .op2(DxsoOpcode::Mul, rd(0, MaskXYZ), r(0), r(2))
      .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(0), c(6), r(1))
      .op1(DxsoOpcode::Mov, oC0(MaskW), v(1, kWWWW));
    const DxsoUe3CompositeShaderInfo carpetInfo = analyzeDxsoUe3CompositeShader(carpet.view());
    check(carpetInfo.refusal == DxsoUe3CompositeRefusal::None, "the carpet bakes");
    check(carpetInfo.maskLayers.size() == 1 && carpetInfo.maskLayers[0].sampler == 0 &&
          carpetInfo.maskLayers[0].channel == 0 && carpetInfo.maskLayers[0].degree == 2,
          "the carpet's Texture2D_1.x is a quadratic mask");

    // The same mask is not lifted when it emits, because the bake at full mask only replaces the lit pass.
    DxsoTestShader emits(kPs30Header);
    emits.ctab({
        { "AmbientColorAndSkyFactor", kFloat4, 0 },
        { "Texture2D_0", kSampler, 0 },
        { "Texture2D_1", kSampler, 1 },
      })
      .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
      .dclSampler(0)
      .dclSampler(1)
      .texld(rd(0), v(0), 0)
      .texld(rd(1), v(0), 1)
      .op2(DxsoOpcode::Mul, rd(2, MaskXYZ), r(0), c(0))
      .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(0), r(1, kXXXX), r(2));
    const DxsoUe3CompositeShaderInfo emitting = analyzeDxsoUe3CompositeShader(emits.view());
    check(emitting.refusal == DxsoUe3CompositeRefusal::None && emitting.hasEmissive, "the emitting mask material bakes");
    check(emitting.maskLayers.empty(), "an emitting mask is not lifted out of a lit compile");
  }

  void testLightingLayers() {
    // RPA_Lower's white paint in its directional lightmap compile weights the lightmap by a normal map at 1x and
    // tints its paint at 2x. The normal map only feeds the lighting, so the bake leaves it out of what it renders.
    DxsoTestShader ps(kPs30Header);
    ps.ctab({
        { "UniformVector_1", kFloat4, 2 },
        { "AmbientColorAndSkyFactor", kFloat4, 3 },
        { "Texture2D_0", kSampler, 0 },
        { "Texture2D_1", kSampler, 1 },
      })
      .def(1, 2.0f, -1.0f, 0.0f, 0.0f)
      .def(5, -0.707106769f, -0.408248305f, 0.577350259f, 0.0f)
      .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
      .dclInput(DxsoUsage::Texcoord, 2, 1, MaskXYZ)
      .dclSampler(0)
      .dclSampler(1)
      .texld(rd(0), v(0), 0)
      .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), r(0), c(1, kXXXX), c(1, kYYYY))
      .op1(DxsoOpcode::Nrm, rd(1, MaskXYZ), r(0))
      .op2(DxsoOpcode::Dp3, rdSat(2, MaskX), r(1), c(5))
      .op2(DxsoOpcode::Mul, rd(2, MaskX), r(2, kXXXX), r(2, kXXXX))
      .op2(DxsoOpcode::Add, rd(3, MaskXY), v(0), v(0))
      .texld(rd(4), r(3), 1)
      .op2(DxsoOpcode::Mul, rd(4, MaskXYZ), r(4), c(2))
      .op2(DxsoOpcode::Mul, rd(5, MaskXYZ), r(2, kXXXX), v(1))
      .op2(DxsoOpcode::Mul, rd(5, MaskXYZ), r(5), r(4))
      .op3(DxsoOpcode::Mad, oC0(MaskXYZ), r(4), c(3), r(5))
      .op1(DxsoOpcode::Mov, oC0(MaskW), c(1, kZZZZ));
    const DxsoUe3CompositeShaderInfo info = analyzeDxsoUe3CompositeShader(ps.view());
    check(info.refusal == DxsoUe3CompositeRefusal::None, "the normal-mapped paint bakes");
    check(info.lighting == DxsoUe3CompositeLighting::AmbientConstant, "its ambient term carries the paint");
    check(info.layers.size() == 1 && info.layers[0].sampler == 1 && info.layers[0].diffuse, "only the paint is a baked layer");
    check(info.lightingLayers.size() == 1 && info.lightingLayers[0].sampler == 0 && info.lightingLayers[0].name == "Texture2D_0",
          "the normal map is a lighting layer");

    // The paint at 2x repeats every 0.5 UV, and covering the normal map at 1x makes that 1.
    const float paint[] = { 2.0f };
    const float paintAndNormalMap[] = { 2.0f, 1.0f };
    check(solveUe3CompositePeriod(paint, 1) == 0.5f && solveUe3CompositePeriod(paintAndNormalMap, 2) == 1.0f,
          "covering the normal map doubles the repeat");
  }

  void testTexcoordInColor() {
    // Escape_p's Avenue overpass tints its floor tile yellow or blue by which half of TEXCOORD2 it is on, and a
    // mask on TEXCOORD2 fades the tint out between them. Only TEXCOORD2.y feeds the colour other than as a
    // coordinate, and the normalised camera vector is a direction rather than a coordinate.
    DxsoTestShader ps(kPs30Header);
    ps.ctab({
        { "UniformVector_1", kFloat4, 2 },
        { "UniformVector_2", kFloat4, 3 },
        { "AmbientColorAndSkyFactor", kFloat4, 4 },
        { "Texture2D_1", kSampler, 0 },
        { "Texture2D_2", kSampler, 1 },
      })
      .def(1, 256.0f, -128.0f, 0.5f, 1.0f)
      .dclInput(DxsoUsage::Texcoord, 1, 1, MaskZW)
      .dclInput(DxsoUsage::Texcoord, 2, 2, MaskXY)
      .dclInput(DxsoUsage::Texcoord, 6, 3, MaskXYZ)
      .dclSampler(0)
      .dclSampler(1)
      .texld(rd(0), v(1, swz(3, 2, 2, 3)), 0)
      .op3(DxsoOpcode::Mad, rdSat(1, MaskW), v(2, kYYYY), c(1, kXXXX), c(1, kYYYY))
      .op1(DxsoOpcode::Mov, rd(2, MaskXYZ), c(2))
      .op2(DxsoOpcode::Add, rd(2, MaskXYZ), r(2, kXYZW, kModNeg), c(3))
      .op3(DxsoOpcode::Mad, rd(2, MaskXYZ), r(1, kWWWW), r(2), c(2))
      .op2(DxsoOpcode::Mul, rd(3, MaskXYZ), r(0), r(2))
      .op3(DxsoOpcode::Mad, rd(4, MaskXYZ), r(2), r(0, kXYZW, kModNeg), r(0))
      .texld(rd(5), v(2), 1)
      .op3(DxsoOpcode::Mad, rd(4, MaskXYZ), r(5, kXXXX), r(4), r(3))
      .op1(DxsoOpcode::Nrm, rd(6, MaskXYZ), v(3))
      .op2(DxsoOpcode::Mul, rd(4, MaskXYZ), r(4), r(6, kZZZZ))
      .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(4), c(4))
      .op1(DxsoOpcode::Mov, oC0(MaskW), c(1, kWWWW));
    const DxsoUe3CompositeShaderInfo info = analyzeDxsoUe3CompositeShader(ps.view());
    check(info.refusal == DxsoUe3CompositeRefusal::None, "the overpass floor is analysed");
    check(info.colorTexcoordLanes[2] == MaskY, "the colour reads TEXCOORD2.y by arithmetic");
    check(info.colorTexcoordLanes[1] == 0, "the tile's coordinate reaches the colour only through its sample");
    check(info.colorTexcoordLanes[6] == 0, "the normalised camera vector is not a coordinate the colour reads");
  }

  void testRefusals() {
    {
      // A vertex colour blend has no value over a texture repeat.
      DxsoTestShader ps(kPs30Header);
      ps.ctab({ { "AmbientColorAndSkyFactor", kFloat4, 0 }, { "Texture2D_0", kSampler, 0 } })
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclInput(DxsoUsage::Color, 0, 1, MaskAll)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op2(DxsoOpcode::Mul, rd(0), r(0), v(1))
        .op2(DxsoOpcode::Mul, oC0(MaskXYZ), r(0), c(0));
      check(analyzeDxsoUe3CompositeShader(ps.view()).refusal == DxsoUe3CompositeRefusal::VertexColor, "vertex colour is refused");
    }
    {
      // The scene colour behind the surface has no value offscreen.
      DxsoTestShader ps(kPs30Header);
      ps.ctab({ { "Texture2D_0", kSampler, 0 }, { "SceneColorTexture", kSampler, 1 } })
        .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .dclSampler(1)
        .texld(rd(0), v(0), 0)
        .texld(rd(1), v(0), 1)
        .op2(DxsoOpcode::Mul, oC0(MaskAll), r(0), r(1));
      const DxsoUe3CompositeShaderInfo info = analyzeDxsoUe3CompositeShader(ps.view());
      check(info.refusal == DxsoUe3CompositeRefusal::EngineSampler, "an engine sampler is refused");
    }
    {
      // A shader without a constant table is not a material the analysis can name.
      DxsoTestShader ps(kPs30Header);
      ps.dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op1(DxsoOpcode::Mov, oC0(MaskAll), r(0));
      check(analyzeDxsoUe3CompositeShader(ps.view()).refusal == DxsoUe3CompositeRefusal::NotAnalyzed, "no CTAB is not analysed");
    }
  }

  void testPeriod() {
    auto period = [](std::initializer_list<float> scales) {
      const std::vector<float> v(scales);
      return solveUe3CompositePeriod(v.data(), uint32_t(v.size()));
    };
    check(period({ 1.0f }) == 1.0f, "a single layer repeats with itself");
    check(period({ 0.5f, 4.0f }) == 2.0f, "the coarsest layer's repeat");
    check(period({ 1.0f, 1.5f }) == 2.0f, "two repeats of the coarsest when a ratio is a half");
    check(std::abs(period({ 1.0f, 0.3f }) - 10.0f) < 1e-3f, "three repeats of a 0.3 layer");
    check(std::abs(period({ 3.0f, 3.0f * 1.0000333f }) - 1.0f / 3.0f) < 1e-4f, "float noise within the tolerance");
    check(period({ 1.0f, 1.37f }) == 0.0f, "no common repeat within 32 of the coarsest");
    // Boat_p's stripe mask tiles 0.85 times beside layers at 1 and 6, so 17 of its repeats are 20 of theirs.
    check(std::abs(period({ 1.0f, 6.0f, 1.0f, 0.85f }) - 20.0f) < 1e-3f, "a 0.85 tiling repeats with whole ones every 20");
    check(solveUe3CompositePeriod(std::vector<float>({ 1.0f, 0.85f }).data(), 2, 8) == 0.0f, "but not within 8");
    check(period({ 1.0f, 0.0f }) == 0.0f, "a zero tiling does not repeat");
    check(solveUe3CompositePeriod(nullptr, 0) == 0.0f, "no layers, no repeat");
  }

  void testResolution() {
    auto resolution = [](std::initializer_list<float> texels, uint32_t maxResolution) {
      const std::vector<float> v(texels);
      return solveUe3CompositeResolution(v.data(), uint32_t(v.size()), 64, maxResolution);
    };
    check(resolution({ 100.0f }, 4096) == 128, "rounded up to a power of two");
    check(resolution({ 10.0f }, 4096) == 64, "at least the minimum");
    check(resolution({ 5000.0f }, 4096) == 4096, "at most the maximum");
    check(resolution({ 512.0f, 2048.0f }, 4096) == 2048, "the finest layer decides");
  }

  int dumpShader(const char* path) {
    DumpedShader shader;
    if (!loadDumpedShader(path, shader)) {
      std::cout << path << ": could not be read" << std::endl;
      return 1;
    }
    const DxsoShaderView view = shader.view();
    const DxsoUe3CompositeShaderInfo info = analyzeDxsoUe3CompositeShader(view);
    std::array<PsSamplerUvOrigin, kDxsoMaxPsSamplers> origins;
    analyzePsSamplerUvOrigins(view, origins);

    std::cout << "== " << path << "\n   refusal=" << describeDxsoUe3CompositeRefusal(info.refusal);
    if (!info.refusalDetail.empty()) {
      std::cout << " (" << info.refusalDetail << ")";
    }
    std::cout << "\n   lighting=" << describeDxsoUe3CompositeLighting(info.lighting)
              << " emissive=" << info.hasEmissive << " viewDependent=" << info.viewDependent
              << " texkill=" << info.hasTexkill << " worldCoords=" << info.worldCoords
              << " alphaInput=" << info.alphaInputRegister
              << " ambient=c" << info.ambientColorAndSkyFactorRegister
              << " lightMapScale=c" << info.lightMapScaleRegister << "+" << info.lightMapScaleCount
              << " lightmapSamplers=0x" << std::hex << info.lightmapTextureSamplerMask
              << " vertexLightmapInputs=0x" << info.vertexLightmapInputMask << std::dec << "\n";
    for (uint32_t t = 0; t < info.colorTexcoordLanes.size(); t++) {
      if (info.colorTexcoordLanes[t]) {
        std::cout << "   colour computes with TEXCOORD" << t << ".";
        for (uint32_t lane = 0; lane < 4; lane++) {
          if ((info.colorTexcoordLanes[t] >> lane) & 1u) {
            std::cout << "xyzw"[lane];
          }
        }
        std::cout << "\n";
      }
    }
    for (const DxsoUe3CompositeMaskLayer& mask : info.maskLayers) {
      std::cout << "   mask s" << uint32_t(mask.sampler) << "." << "xyzw"[mask.channel & 3u]
                << (mask.degree == 2 ? " (quadratic)" : "") << "\n";
    }

    // Layers whose tilings are immediates solve to a repeat here. Constant tilings need the draw.
    std::vector<float> scalesU;
    std::vector<float> scalesV;
    std::vector<float> unmaskedScalesU;
    std::vector<float> unmaskedScalesV;
    bool allImmediate = true;
    for (const DxsoUe3CompositeLayer& layer : info.layers) {
      const PsSamplerUvOrigin& origin = origins[layer.sampler];
      std::cout << "   s" << uint32_t(layer.sampler) << " " << layer.name << (layer.isCube ? " cube" : "")
                << " emissive=" << layer.emissive << " diffuse=" << layer.diffuse << " opacity=" << layer.opacity
                << " origin=" << origin.originValid << " sem=" << uint32_t(origin.semanticIndex)
                << " comps=(" << uint32_t(origin.compU) << "," << uint32_t(origin.compV) << ")"
                << " exact=" << origin.affineExact
                << " U=[" << formatUvComponentAffine(origin.affineU) << "] V=[" << formatUvComponentAffine(origin.affineV) << "]\n";
      if (layer.isCube) {
        continue;
      }
      const UvAffineTerm& u = origin.affineU.scale;
      const UvAffineTerm& v = origin.affineV.scale;
      const bool immediate = origin.affineExact && !origin.affineU.hasCross && !origin.affineV.hasCross &&
                             u.constReg < 0 && v.constReg < 0;
      allImmediate &= immediate;
      scalesU.push_back(u.immValid ? u.imm : 1.0f);
      scalesV.push_back(v.immValid ? v.imm : 1.0f);
      const bool isMask = std::any_of(info.maskLayers.begin(), info.maskLayers.end(),
                                      [&](const DxsoUe3CompositeMaskLayer& mask) { return mask.sampler == layer.sampler; });
      if (!isMask) {
        unmaskedScalesU.push_back(scalesU.back());
        unmaskedScalesV.push_back(scalesV.back());
      }
    }
    std::vector<float> coveringScalesU = scalesU;
    std::vector<float> coveringScalesV = scalesV;
    for (const DxsoUe3CompositeLightingLayer& layer : info.lightingLayers) {
      const PsSamplerUvOrigin& origin = origins[layer.sampler];
      std::cout << "   lighting s" << uint32_t(layer.sampler) << " " << layer.name
                << " origin=" << origin.originValid << " sem=" << uint32_t(origin.semanticIndex)
                << " comps=(" << uint32_t(origin.compU) << "," << uint32_t(origin.compV) << ")"
                << " exact=" << origin.affineExact
                << " U=[" << formatUvComponentAffine(origin.affineU) << "] V=[" << formatUvComponentAffine(origin.affineV) << "]\n";
      if (origin.affineExact && !origin.affineU.hasCross && !origin.affineV.hasCross &&
          origin.affineU.scale.constReg < 0 && origin.affineV.scale.constReg < 0) {
        coveringScalesU.push_back(origin.affineU.scale.immValid ? origin.affineU.scale.imm : 1.0f);
        coveringScalesV.push_back(origin.affineV.scale.immValid ? origin.affineV.scale.imm : 1.0f);
      }
    }
    if (allImmediate && !scalesU.empty()) {
      std::cout << "   repeat=(" << solveUe3CompositePeriod(scalesU.data(), uint32_t(scalesU.size())) << ","
                << solveUe3CompositePeriod(scalesV.data(), uint32_t(scalesV.size())) << ")";
      if (!unmaskedScalesU.empty() && unmaskedScalesU.size() < scalesU.size()) {
        std::cout << " without masks=(" << solveUe3CompositePeriod(unmaskedScalesU.data(), uint32_t(unmaskedScalesU.size())) << ","
                  << solveUe3CompositePeriod(unmaskedScalesV.data(), uint32_t(unmaskedScalesV.size())) << ")";
      }
      if (coveringScalesU.size() > scalesU.size()) {
        std::cout << " covering lighting layers=(" << solveUe3CompositePeriod(coveringScalesU.data(), uint32_t(coveringScalesU.size())) << ","
                  << solveUe3CompositePeriod(coveringScalesV.data(), uint32_t(coveringScalesV.size())) << ")";
      }
      std::cout << "\n";
    } else if (!scalesU.empty()) {
      std::cout << "   repeat depends on the draw's constants\n";
    }
    return 0;
  }

} // anonymous namespace

int main(int argc, char** argv) {
  if (argc > 1) {
    int rc = 0;
    for (int i = 1; i < argc; i++) {
      rc |= dumpShader(argv[i]);
    }
    return rc;
  }

  try {
    testCeiling();
    testSimpleLightmap();
    testSimpleVertexLightmap();
    testUnlit();
    testMasked();
    testMaskLayers();
    testLightingLayers();
    testTexcoordInColor();
    testRefusals();
    testPeriod();
    testResolution();
  }
  catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << std::endl;
    return -1;
  }

  std::cout << "test_dxso_ue3_composite: all tests passed" << std::endl;
  return 0;
}
