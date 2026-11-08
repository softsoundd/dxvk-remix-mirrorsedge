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

// Tests the static half of a UE3 flow layer (rtx.d3d9.ue3FlowLayers), which finds the reads that move a material's
// normal detail, their weights in the material normal and the texture channel that confines them.
// The hand-assembled canal is the network Stormdrain_p's M_StormDrainCanals_01 compiles to: two ripple maps panned at
// different tilings, one decoded and one added as it was sampled, lerped into the concrete's normal by a strip mask
// before the normal bends a reflection lookup. The hand-assembled tarp blends its normal map towards a wrinkled one by
// a panned texture.
//
// Run as test_dxso_ue3_flow_layers <shader.dxso> [...], it prints the flow layer analysis of pixel shaders dumped
// through DXVK_SHADER_DUMP_PATH, with each read's UV origin and tiling.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../../src/dxso/dxso_ue3_flow_layers.h"
#include "../../../src/dxso/dxso_uv_dataflow.h"
#include "../../../src/util/log/log.h"
#include "../../../src/util/util_string.h"

#include "dxso_test_assembler.h"

namespace dxvk {
  // Standalone executables own the logger singleton, as the d3d9/dxgi entry points do.
  Logger Logger::s_instance("test_dxso_ue3_flow_layers.log", LogLevel::None);
}

using namespace dxvk;
using namespace dxvk::dxso_test;

namespace {

  constexpr uint16_t kFloat4 = kD3dxRegisterSetFloat4;
  constexpr uint16_t kSampler = kD3dxRegisterSetSampler;

  void check(bool condition, const std::string& what) {
    if (!condition) {
      throw std::runtime_error(what);
    }
  }

  bool approximately(float a, float b) {
    return std::abs(a - b) <= 1e-4f;
  }

  enum class Water {
    // The normal is lerp(concrete, lerp(decode(A), B, 0.5), mask), as M_StormDrainCanals_01 compiles.
    Lerp,
    // The normal is lerp(concrete, decode(A) + B, mask), as M_StormDrainCanals_01b compiles.
    Add,
    // The water shows where the mask is 0.
    InvertedMask,
    // The water is added to the concrete everywhere.
    NoMask,
    // A debris layer lerps the concrete back over the water by a second mask.
    Debris,
    // A is a two-channel normal map read from .wy, with a constant z.
    TwoChannel,
    // The water is normalised on its own before the mask blends it in.
    NormalizedFirst,
    // A is squared on the way.
    Squared,
    // B is weighted by the diffuse's green rather than by the mask.
    OtherMask,
    // The ripple coordinates have no constant offset.
    StaticOffsets,
    // The normal is never normalised.
    NoNormalize,
    // The diffuse pans with ripple B's panner.
    SharedPanner,
    // Ripple B is a second read of ripple A's texture, as Stormdrain_p's spillway pans one normal map twice.
    SharedSampler,
  };

  // Texture2D_0 is the concrete's normal map, Texture2D_1 ripple A at 4x panned by UniformVector_1, Texture2D_2 ripple B
  // at 1x panned by UniformVector_2, Texture2D_3 the strip mask, Texture2D_4 the diffuse, Texture2D_5 a debris mask and
  // TextureCube_0 the reflection. TEXCOORD6 is the tangent-space camera vector.
  DxsoTestShader buildCanal(const Water water) {
    DxsoTestShader ps(kPs30Header);
    ps.ctab({
        { "UniformVector_1", kFloat4, 8 },
        { "UniformVector_2", kFloat4, 9 },
        { "Texture2D_0", kSampler, 0 },
        { "Texture2D_1", kSampler, 1 },
        { "Texture2D_2", kSampler, 2 },
        { "Texture2D_3", kSampler, 3 },
        { "Texture2D_4", kSampler, 4 },
        { "Texture2D_5", kSampler, 5 },
        { "TextureCube_0", kSampler, 6 },
      })
      .def(1, 2.0f, -1.0f, 0.5f, 4.0f)
      .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
      .dclInput(DxsoUsage::Texcoord, 6, 1, MaskXYZ);
    for (uint32_t s = 0; s < 6; s++) {
      ps.dclSampler(s);
    }
    ps.dclSampler(6, DxsoTextureType::TextureCube);

    ps.texld(rd(0), v(0), 0)
      .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), r(0), c(1, kXXXX), c(1, kYYYY));
    if (water == Water::StaticOffsets) {
      ps.op2(DxsoOpcode::Mul, rd(1, MaskXY), v(0), c(1, kWWWW));
    } else {
      ps.op3(DxsoOpcode::Mad, rd(1, MaskXY), v(0), c(1, kWWWW), c(8));
    }
    ps.texld(rd(1), r(1), 1);
    if (water == Water::Squared) {
      ps.op2(DxsoOpcode::Mul, rd(1, MaskXYZ), r(1), r(1));
    }
    if (water == Water::TwoChannel) {
      ps.op3(DxsoOpcode::Mad, rd(1, MaskXY), r(1, swz(3, 1, 2, 2)), c(1, kXXXX), c(1, kYYYY))
        .op1(DxsoOpcode::Mov, rd(1, MaskZ), c(1, kZZZZ));
    } else {
      ps.op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(1), c(1, kXXXX), c(1, kYYYY));
    }
    if (water == Water::StaticOffsets) {
      ps.op1(DxsoOpcode::Mov, rd(2, MaskXY), v(0));
    } else {
      ps.op2(DxsoOpcode::Add, rd(2, MaskXY), v(0), c(9));
    }
    ps.texld(rd(2), r(2), water == Water::SharedSampler ? 1 : 2)
      .texld(rd(3), v(0), 3);

    if (water == Water::OtherMask) {
      ps.op3(DxsoOpcode::Lrp, rd(4, MaskXYZ), r(3, kXXXX), r(1), r(0))
        .texld(rd(8), v(0), 4)
        .op3(DxsoOpcode::Mad, rd(4, MaskXYZ), r(2), r(8, kYYYY), r(4));
    } else {
      if (water == Water::Add) {
        ps.op2(DxsoOpcode::Add, rd(1, MaskXYZ), r(1), r(2));
      } else {
        ps.op3(DxsoOpcode::Lrp, rd(1, MaskXYZ), c(1, kZZZZ), r(2), r(1));
      }
      if (water == Water::NormalizedFirst) {
        ps.op1(DxsoOpcode::Nrm, rd(1, MaskXYZ), r(1));
      }
      if (water == Water::NoMask) {
        ps.op2(DxsoOpcode::Add, rd(4, MaskXYZ), r(0), r(1));
      } else if (water == Water::InvertedMask) {
        ps.op3(DxsoOpcode::Lrp, rd(4, MaskXYZ), r(3, kXXXX), r(0), r(1));
      } else {
        ps.op3(DxsoOpcode::Lrp, rd(4, MaskXYZ), r(3, kXXXX), r(1), r(0));
      }
    }
    if (water == Water::Debris) {
      ps.texld(rd(5), v(0), 5)
        .op3(DxsoOpcode::Lrp, rd(4, MaskXYZ), r(5, kXXXX), r(0), r(4));
    }

    if (water == Water::NoNormalize) {
      ps.op1(DxsoOpcode::Mov, rd(5, MaskXYZ), r(4));
    } else {
      ps.op1(DxsoOpcode::Nrm, rd(5, MaskXYZ), r(4));
    }
    ps.op1(DxsoOpcode::Nrm, rd(6, MaskXYZ), v(1))
      .op2(DxsoOpcode::Dp3, rd(0, MaskW), r(5), r(6))
      .op2(DxsoOpcode::Mul, rd(5, MaskXYZ), r(5), r(0, kWWWW))
      .op3(DxsoOpcode::Mad, rd(5, MaskXYZ), r(5), c(1, kXXXX), r(6, kXYZW, kModNeg))
      .texld(rd(7), r(5), 6);
    if (water == Water::SharedPanner) {
      ps.op2(DxsoOpcode::Add, rd(9, MaskXY), v(0), c(9))
        .texld(rd(8), r(9), 4);
    } else {
      ps.texld(rd(8), v(0), 4);
    }
    ps.op2(DxsoOpcode::Add, oC0(MaskXYZ), r(7), r(8))
      .op1(DxsoOpcode::Mov, oC0(MaskW), c(1, kZZZZ));
    return ps;
  }

  DxsoUe3FlowShaderInfo analyze(const Water water) {
    DxsoTestShader ps = buildCanal(water);
    return analyzeDxsoUe3FlowLayers(ps.view());
  }

  // Texture2D_0 is the tarp's own normal map, Texture2D_1 the wrinkles at 4x, Texture2D_2 the wind, panned by
  // UniformVector_1, and Texture2D_3 the diffuse, whose green also weighs the wind, as Mirror's Edge's construction
  // tarps compile.
  DxsoTestShader buildTarp(const bool diffuseWeighsWind) {
    DxsoTestShader ps(kPs30Header);
    ps.ctab({
        { "UniformVector_1", kFloat4, 8 },
        { "Texture2D_0", kSampler, 0 },
        { "Texture2D_1", kSampler, 1 },
        { "Texture2D_2", kSampler, 2 },
        { "Texture2D_3", kSampler, 3 },
        { "TextureCube_0", kSampler, 4 },
      })
      .def(1, 2.0f, -1.0f, 0.8f, 4.0f)
      .dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
      .dclInput(DxsoUsage::Texcoord, 6, 1, MaskXYZ);
    for (uint32_t s = 0; s < 4; s++) {
      ps.dclSampler(s);
    }
    ps.dclSampler(4, DxsoTextureType::TextureCube);

    ps.texld(rd(0), v(0), 0)
      .op3(DxsoOpcode::Mad, rd(0, MaskXYZ), r(0), c(1, kXXXX), c(1, kYYYY))
      .op2(DxsoOpcode::Mul, rd(1, MaskXY), v(0), c(1, kWWWW))
      .texld(rd(1), r(1), 1)
      .op3(DxsoOpcode::Mad, rd(1, MaskXYZ), r(1), c(1, kXXXX), c(1, kYYYY))
      .op3(DxsoOpcode::Mad, rd(2, MaskXY), v(0), c(1, kXXXX), c(8))
      .texld(rd(2), r(2), 2)
      .texld(rd(3), v(0), 3);
    if (diffuseWeighsWind) {
      ps.op2(DxsoOpcode::Mul, rd(4, MaskW), r(2, kYYYY), r(3, kYYYY))
        .op2(DxsoOpcode::Mul, rd(4, MaskW), r(4, kWWWW), c(1, kZZZZ));
    } else {
      ps.op2(DxsoOpcode::Mul, rd(4, MaskW), r(2, kYYYY), c(1, kZZZZ));
    }
    ps.op3(DxsoOpcode::Lrp, rd(5, MaskXYZ), r(4, kWWWW), r(1), r(0))
      .op1(DxsoOpcode::Nrm, rd(5, MaskXYZ), r(5))
      .op1(DxsoOpcode::Nrm, rd(6, MaskXYZ), v(1))
      .op2(DxsoOpcode::Dp3, rd(0, MaskW), r(5), r(6))
      .op2(DxsoOpcode::Mul, rd(5, MaskXYZ), r(5), r(0, kWWWW))
      .op3(DxsoOpcode::Mad, rd(5, MaskXYZ), r(5), c(1, kXXXX), r(6, kXYZW, kModNeg))
      .texld(rd(7), r(5), 4)
      .op2(DxsoOpcode::Add, oC0(MaskXYZ), r(7), r(3))
      .op1(DxsoOpcode::Mov, oC0(MaskW), c(1, kZZZZ));
    return ps;
  }

  void checkAccepted(const DxsoUe3FlowShaderInfo& info, const char* what) {
    check(info.refusal == DxsoUe3FlowRefusal::None,
          str::format(what, " refused: ", describeDxsoUe3FlowRefusal(info.refusal), " ", info.refusalDetail));
    check(info.samples.size() == 2, str::format(what, " pans 2 samples, found ", info.samples.size()));
    check(info.samples[0].sampler == 1 && info.samples[1].sampler == 2, str::format(what, ": ripple A is the heavier"));
    check(info.flowSamplerMask == ((1u << 1) | (1u << 2)), str::format(what, ": the flow samplers are s1 and s2"));
    check(info.offsetRegisters == std::vector<uint32_t>({ 8, 9 }), str::format(what, ": the panners are c8 and c9"));
  }

  void testCanal() {
    const DxsoUe3FlowShaderInfo info = analyze(Water::Lerp);
    checkAccepted(info, "the canal");
    // A is 0.5 * (2A - 1) and B adds 0.5 * B, which is 0.25 of a decoded normal.
    check(approximately(info.samples[0].weight, 0.5f) && approximately(info.samples[1].weight, 0.25f),
          str::format("the canal weighs its ripples 0.5 and 0.25, found ", info.samples[0].weight, " and ", info.samples[1].weight));
    check(info.samples[0].components == std::array<uint8_t, 3>({ 0, 1, 2 }), "ripple A's xyz are the normal's xyz");
    check(info.samples[1].components == std::array<uint8_t, 3>({ 0, 1, 2 }), "ripple B's xyz are the normal's xyz");
    check(info.mask.present && info.mask.sampler == 3 && info.mask.channel == 0, "the strip mask is Texture2D_3.x");
    check(approximately(info.mask.scale, 1.0f) && approximately(info.mask.bias, 0.0f), "the strip mask is used as it is");
    check(!info.ignoresOtherFactors, "nothing else weighs the canal's water");

    const DxsoUe3FlowShaderInfo added = analyze(Water::Add);
    checkAccepted(added, "the canal that adds its ripples");
    check(approximately(added.samples[0].weight, 1.0f) && approximately(added.samples[1].weight, 0.5f),
          "added ripples keep their 2:1 weighting at twice the scale");
  }

  void testMasks() {
    const DxsoUe3FlowShaderInfo inverted = analyze(Water::InvertedMask);
    checkAccepted(inverted, "an inverted mask");
    check(inverted.mask.present && inverted.mask.sampler == 3 && approximately(inverted.mask.scale, -1.0f) &&
            approximately(inverted.mask.bias, 1.0f),
          "an inverted mask is 1 - mask");

    const DxsoUe3FlowShaderInfo unmasked = analyze(Water::NoMask);
    checkAccepted(unmasked, "unmasked water");
    check(!unmasked.mask.present, "water added everywhere has no mask");

    const DxsoUe3FlowShaderInfo debris = analyze(Water::Debris);
    checkAccepted(debris, "water under debris");
    check(debris.mask.present && debris.mask.sampler == 3 && approximately(debris.mask.scale, 1.0f),
          "the strip is still the mask under a debris layer");
    check(debris.ignoresOtherFactors, "the debris mask is reported as left out");

    const DxsoUe3FlowShaderInfo normalized = analyze(Water::NormalizedFirst);
    checkAccepted(normalized, "water normalised before it is masked");
    check(normalized.mask.present && normalized.mask.sampler == 3,
          "the mask comes from the normalisation that blends the normalised water in");
    check(approximately(normalized.samples[1].weight / normalized.samples[0].weight, 0.5f), "normalised water keeps the ripple weighting");
  }

  void testTwoChannel() {
    const DxsoUe3FlowShaderInfo info = analyze(Water::TwoChannel);
    checkAccepted(info, "a two-channel ripple map");
    check(info.samples[0].components == std::array<uint8_t, 3>({ 3, 1, kDxsoUe3FlowNoComponent }),
          "ripple A's normal is .wy with a reconstructed z");
  }

  void testSharedSampler() {
    const DxsoUe3FlowShaderInfo info = analyze(Water::SharedSampler);
    check(info.refusal == DxsoUe3FlowRefusal::None,
          str::format("two reads of one ripple map refused: ", describeDxsoUe3FlowRefusal(info.refusal), " ", info.refusalDetail));
    check(info.samples.size() == 2, str::format("two reads of one ripple map make 2 samples, found ", info.samples.size()));
    check(info.samples[0].sampler == 1 && info.samples[1].sampler == 1, "both reads are of Texture2D_1");
    check(info.samples[0].site != info.samples[1].site, "the reads are told apart");
    check(approximately(info.samples[0].weight, 0.5f) && approximately(info.samples[1].weight, 0.25f),
          str::format("the reads keep the canal's weights, found ", info.samples[0].weight, " and ", info.samples[1].weight));
    check(info.samples[0].uv.affineU.offset.constReg == 8 && info.samples[1].uv.affineU.offset.constReg == 9,
          "each read keeps its own panner");
    check(approximately(info.samples[0].uv.affineU.scale.imm, 4.0f) && !info.samples[1].uv.affineU.scale.immValid,
          "each read keeps its own tiling");
    check(info.flowSamplerMask == (1u << 1), "the flow sampler is s1 alone");
    check(info.offsetRegisters == std::vector<uint32_t>({ 8, 9 }) && info.exclusiveOffsetRegisters == info.offsetRegisters,
          "the panners are c8 and c9, and only the ripples read them");
  }

  void testBlend() {
    for (const bool diffuseWeighsWind : { true, false }) {
      DxsoTestShader ps = buildTarp(diffuseWeighsWind);
      const DxsoUe3FlowShaderInfo info = analyzeDxsoUe3FlowLayers(ps.view());
      const char* what = diffuseWeighsWind ? "the tarp" : "the tarp without its diffuse weight";
      check(info.refusal == DxsoUe3FlowRefusal::None,
            str::format(what, " refused: ", describeDxsoUe3FlowRefusal(info.refusal), " ", info.refusalDetail));
      check(info.blend.present && info.blend.sampler == 2 && info.blend.channel == 1,
            str::format(what, ": the wind blends by Texture2D_2.y"));
      check(approximately(info.blend.scale, 0.8f), str::format(what, ": the wind blends by 0.8 of the texel, found ", info.blend.scale));
      check(info.blend.uv.affineU.offset.constReg == 8, str::format(what, ": the wind pans with c8"));
      check(info.samples.size() == 2 && info.samples[0].sampler == 0 && info.samples[1].sampler == 1,
            str::format(what, ": the wind blends Texture2D_0 towards Texture2D_1"));
      check(info.samples[0].components == std::array<uint8_t, 3>({ 0, 1, 2 }) &&
            info.samples[1].components == std::array<uint8_t, 3>({ 0, 1, 2 }),
            str::format(what, ": both normal maps are read as xyz"));
      check(info.mask.present == diffuseWeighsWind && (!diffuseWeighsWind || (info.mask.sampler == 3 && info.mask.channel == 1)),
            str::format(what, ": the diffuse's green is the mask when it weighs the wind"));
      check(info.flowSamplerMask == 0x7u, str::format(what, ": the layer's samplers are s0 to s2"));
      check(info.offsetRegisters == std::vector<uint32_t>({ 8 }) && info.exclusiveOffsetRegisters == info.offsetRegisters,
            str::format(what, ": the panner is c8, and only the wind reads it"));
    }
  }

  void testSharedPanner() {
    const DxsoUe3FlowShaderInfo canal = analyze(Water::Lerp);
    check(canal.exclusiveOffsetRegisters == std::vector<uint32_t>({ 8, 9 }), "only the canal's ripples pan with c8 and c9");

    const DxsoUe3FlowShaderInfo shared = analyze(Water::SharedPanner);
    checkAccepted(shared, "ripples sharing a panner with the diffuse");
    check(shared.exclusiveOffsetRegisters == std::vector<uint32_t>({ 8 }),
          "c9 also pans the diffuse, so a composite bake still depends on it");
  }

  void testRefusals() {
    auto refusal = [](const Water water) { return analyze(water).refusal; };
    check(refusal(Water::Squared) == DxsoUe3FlowRefusal::NotLinear, "a squared ripple is not linear");
    check(refusal(Water::OtherMask) == DxsoUe3FlowRefusal::NotProportional, "ripples with different masks are refused");
    check(refusal(Water::StaticOffsets) == DxsoUe3FlowRefusal::NoPannedSampler, "ripples without a constant offset cannot pan");
    check(refusal(Water::NoNormalize) == DxsoUe3FlowRefusal::NotNormalized, "a normal never normalised is not found");

    DxsoTestShader empty(kPs30Header);
    empty.op1(DxsoOpcode::Mov, oC0(MaskAll), c(0));
    check(analyzeDxsoUe3FlowLayers(empty.view()).refusal == DxsoUe3FlowRefusal::NotAnalyzed,
          "a shader without a constant table is not analysed");
  }

  int dumpShader(const char* path) {
    DumpedShader shader;
    if (!loadDumpedShader(path, shader)) {
      std::cout << path << ": could not be read" << std::endl;
      return 1;
    }
    const DxsoShaderView view = shader.view();
    const DxsoUe3FlowShaderInfo info = analyzeDxsoUe3FlowLayers(view);

    std::cout << "== " << path << "\n   refusal=" << describeDxsoUe3FlowRefusal(info.refusal);
    if (!info.refusalDetail.empty()) {
      std::cout << " (" << info.refusalDetail << ")";
    }
    std::cout << "\n";
    for (const DxsoUe3FlowSample& sample : info.samples) {
      const PsSamplerUvOrigin& origin = sample.uv;
      std::cout << "   sample s" << uint32_t(sample.sampler) << " " << sample.name << " at #" << sample.site
                << " weight=" << sample.weight << " normal=";
      for (const uint8_t component : sample.components) {
        std::cout << (component == kDxsoUe3FlowNoComponent ? '_' : "xyzw"[component & 3u]);
      }
      std::cout << " sem=" << uint32_t(origin.semanticIndex) << " comps=(" << uint32_t(origin.compU) << ","
                << uint32_t(origin.compV) << ") U=[" << formatUvComponentAffine(origin.affineU) << "] V=["
                << formatUvComponentAffine(origin.affineV) << "]\n";
    }
    if (info.blend.present) {
      const PsSamplerUvOrigin& origin = info.blend.uv;
      std::cout << "   blend s" << uint32_t(info.blend.sampler) << "." << "xyzw"[info.blend.channel & 3u] << " " << info.blend.name
                << " at #" << info.blend.site << " = " << info.blend.scale << " * texel sem="
                << uint32_t(origin.semanticIndex) << " comps=(" << uint32_t(origin.compU) << "," << uint32_t(origin.compV)
                << ") U=[" << formatUvComponentAffine(origin.affineU) << "] V=[" << formatUvComponentAffine(origin.affineV) << "]\n";
    }
    if (info.mask.present) {
      const PsSamplerUvOrigin& origin = info.mask.uv;
      std::cout << "   mask s" << uint32_t(info.mask.sampler) << "." << "xyzw"[info.mask.channel & 3u] << " " << info.mask.name
                << " = " << info.mask.bias << " + " << info.mask.scale << " * texel"
                << " sem=" << uint32_t(origin.semanticIndex) << " U=[" << formatUvComponentAffine(origin.affineU)
                << "] V=[" << formatUvComponentAffine(origin.affineV) << "]\n";
    } else if (info.refusal == DxsoUe3FlowRefusal::None) {
      std::cout << "   no mask\n";
    }
    if (!info.offsetRegisters.empty()) {
      std::cout << "   offsets";
      for (const uint32_t reg : info.offsetRegisters) {
        const bool exclusive = std::find(info.exclusiveOffsetRegisters.begin(), info.exclusiveOffsetRegisters.end(), reg) !=
                               info.exclusiveOffsetRegisters.end();
        std::cout << " c" << reg << (exclusive ? "" : " (shared)");
      }
      std::cout << "\n";
    }
    if (info.ignoresOtherFactors) {
      std::cout << "   other textures weigh the samples too, and are left out\n";
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
    testCanal();
    testMasks();
    testTwoChannel();
    testSharedSampler();
    testBlend();
    testSharedPanner();
    testRefusals();
  }
  catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << std::endl;
    return -1;
  }

  std::cout << "test_dxso_ue3_flow_layers: all tests passed" << std::endl;
  return 0;
}
