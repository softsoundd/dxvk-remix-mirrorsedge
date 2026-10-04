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

// The shader half of UE3 material identity: which samplers and uniforms of a base-pass pixel
// shader identify its material, and the signature shared by the lightmap-policy compiles of one
// material (see "Material identity and replacement anchor stability" in UE3Compatibility.md).

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../../src/dxso/dxso_ue3_material_identity.h"
#include "../../../src/util/log/log.h"
#include "../../../src/util/util_string.h"

#include "dxso_test_assembler.h"

namespace dxvk {
  // Standalone executables own the logger singleton, as the d3d9/dxgi entry points do.
  Logger Logger::s_instance("test_dxso_ue3_material_identity.log", LogLevel::None);
}

using namespace dxvk;
using namespace dxvk::dxso_test;

namespace {

  std::string describeIdentity(const Ue3PsMaterialIdentityInfo& info) {
    std::string out = str::format(
      "ctab=", info.hasCtab ? 1 : 0,
      " samplers=0x", std::hex, info.materialSamplerMask,
      " lightingInputs=0x", info.lightingInputSamplerMask, std::dec,
      " textureless=", info.texturelessSignature ? 1 : 0,
      " signed=", info.canonicalShaderSignature != kEmptyHash ? 1 : 0,
      " names=[");
    for (const auto& [name, key, reg] : info.materialSamplersByNameOrder) {
      out += str::format(out.back() == '[' ? "" : ",", name, "@s", reg);
    }
    out += "] consts=[";
    for (const auto& [start, count] : info.constRanges) {
      out += str::format(out.back() == '[' ? "" : ",", "c", start, "+", count);
    }
    out += "] dropped=[";
    for (const uint32_t reg : info.volatileUniformRegisters) {
      out += str::format(out.back() == '[' ? "" : ",", "c", reg);
    }
    return out + "]";
  }

  void expectIdentity(const Ue3PsMaterialIdentityInfo& actual, const std::string& expected, const char* label) {
    const std::string description = describeIdentity(actual);
    if (description != expected) {
      std::cerr << label << ":\n  expected " << expected << "\n  actual   " << description << std::endl;
      throw std::runtime_error(label);
    }
  }

  class PsShader : public DxsoTestShader {
  public:
    PsShader() : DxsoTestShader(kPs30Header) { }

    Ue3PsMaterialIdentityInfo identity(bool detectVolatileConstants = true) {
      return parseUe3PsMaterialIdentityFromCtab(view(), detectVolatileConstants);
    }
  };

  // A lightmapped base pass: the material texture modulated by a two-coefficient lightmap, plus
  // an engine scene-colour sampler.
  void appendLightmappedBasePass(PsShader& ps, uint32_t materialSampler, uint32_t lightmapSampler) {
    ps.dclInput(DxsoUsage::Texcoord, 0, 0, MaskAll)
      .dclSampler(materialSampler)
      .dclSampler(lightmapSampler)
      .dclSampler(lightmapSampler + 1)
      .dclSampler(3)
      .texld(rd(0), v(0), materialSampler)
      .texld(rd(1), v(0, swz(2, 3, 2, 3)), lightmapSampler)
      .texld(rd(2), v(0, swz(2, 3, 2, 3)), lightmapSampler + 1)
      .op2(DxsoOpcode::Add, rd(1), r(1), r(2))
      .op2(DxsoOpcode::Mul, rd(0), r(0), r(1))
      .texld(rd(3), v(0), 3)
      .op2(DxsoOpcode::Add, rd(0), r(0), r(3))
      .op1(DxsoOpcode::Mov, oC0(), r(0));
  }

  class MaterialIdentityTestApp {
  public:
    static void run() {
      std::cout << std::endl << "Begin DXSO UE3 material identity tests" << std::endl;
      test_textureKeptLightmapDropped();
      test_signatureIgnoresRegisterAssignment();
      test_volatilePannerConstantExcluded();
      test_texturelessSignature();
      test_noCtab();
      std::cout << "All DXSO UE3 material identity tests passed" << std::endl;
    }

  private:
    // Only Texture2D_* names are material parameters; lightmaps and engine samplers stay out.
    static void test_textureKeptLightmapDropped() {
      std::cout << "  test_textureKeptLightmapDropped" << std::endl;
      PsShader ps;
      ps.ctab({ { "Texture2D_0", kD3dxRegisterSetSampler, 0 },
                { "LightMapTextures", kD3dxRegisterSetSampler, 1, 2 },
                { "SceneColorTexture", kD3dxRegisterSetSampler, 3 } });
      appendLightmappedBasePass(ps, 0, 1);
      expectIdentity(ps.identity(),
                     "ctab=1 samplers=0x1 lightingInputs=0x0 textureless=0 signed=1 names=[texture2d_0@s0] consts=[] dropped=[]",
                     "material texture kept, lightmap dropped");
    }

    // The lightmap-policy compiles of one material assign registers differently.
    static void test_signatureIgnoresRegisterAssignment() {
      std::cout << "  test_signatureIgnoresRegisterAssignment" << std::endl;
      PsShader a;
      a.ctab({ { "Texture2D_0", kD3dxRegisterSetSampler, 0 },
               { "LightMapTextures", kD3dxRegisterSetSampler, 1, 2 },
               { "SceneColorTexture", kD3dxRegisterSetSampler, 3 } });
      appendLightmappedBasePass(a, 0, 1);

      PsShader b;
      b.ctab({ { "LightMapTextures", kD3dxRegisterSetSampler, 0, 2 },
               { "Texture2D_0", kD3dxRegisterSetSampler, 2 },
               { "SceneColorTexture", kD3dxRegisterSetSampler, 3 } });
      appendLightmappedBasePass(b, 2, 0);

      const Ue3PsMaterialIdentityInfo identityA = a.identity();
      const Ue3PsMaterialIdentityInfo identityB = b.identity();
      if (identityA.canonicalShaderSignature == kEmptyHash ||
          identityA.canonicalShaderSignature != identityB.canonicalShaderSignature) {
        throw std::runtime_error("signature depends on register assignment");
      }
    }

    // A uniform driving the coordinate animates; the tint reaching the colour is the identity.
    static void test_volatilePannerConstantExcluded() {
      std::cout << "  test_volatilePannerConstantExcluded" << std::endl;
      PsShader ps;
      ps.ctab({ { "Texture2D_0", kD3dxRegisterSetSampler, 0 },
                { "UniformVector_0", kD3dxRegisterSetFloat4, 0 },
                { "UniformVector_1", kD3dxRegisterSetFloat4, 1 } });
      ps.dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .op2(DxsoOpcode::Add, rd(1, MaskXY), v(0), c(0))
        .texld(rd(0), r(1), 0)
        .op2(DxsoOpcode::Mul, rd(0), r(0), c(1))
        .op1(DxsoOpcode::Mov, oC0(), r(0));

      const Ue3PsMaterialIdentityInfo info = ps.identity(true);
      expectIdentity(info,
                     "ctab=1 samplers=0x1 lightingInputs=0x0 textureless=0 signed=1 names=[texture2d_0@s0] consts=[c1+1] dropped=[c0]",
                     "panner offset dropped, tint kept");
      if (info.identitySummary.find("excludedAsVolatile=[uniformvector_0]") == std::string::npos) {
        std::cerr << info.identitySummary << std::endl;
        throw std::runtime_error("panner offset not reported as volatile");
      }
    }

    // A material without textures is signed from its colour uniforms' names, not their registers.
    static void test_texturelessSignature() {
      std::cout << "  test_texturelessSignature" << std::endl;
      auto texturelessShader = [](PsShader& ps, const char* name, uint16_t reg) {
        ps.ctab({ { name, kD3dxRegisterSetFloat4, reg } });
        ps.op1(DxsoOpcode::Mov, oC0(), c(reg));
      };

      PsShader a;
      texturelessShader(a, "UniformVector_0", 0);
      const Ue3PsMaterialIdentityInfo identityA = a.identity();
      expectIdentity(identityA,
                     "ctab=1 samplers=0x0 lightingInputs=0x0 textureless=1 signed=1 names=[] consts=[c0+1] dropped=[]",
                     "textureless");

      PsShader moved;
      texturelessShader(moved, "UniformVector_0", 3);
      PsShader renamed;
      texturelessShader(renamed, "UniformVector_1", 0);
      if (moved.identity().canonicalShaderSignature != identityA.canonicalShaderSignature) {
        throw std::runtime_error("textureless signature depends on the register");
      }
      if (renamed.identity().canonicalShaderSignature == identityA.canonicalShaderSignature) {
        throw std::runtime_error("textureless signature ignores the uniform name");
      }
    }

    static void test_noCtab() {
      std::cout << "  test_noCtab" << std::endl;
      PsShader ps;
      ps.dclInput(DxsoUsage::Texcoord, 0, 0, MaskXY)
        .dclSampler(0)
        .texld(rd(0), v(0), 0)
        .op1(DxsoOpcode::Mov, oC0(), r(0));
      expectIdentity(ps.identity(),
                     "ctab=0 samplers=0x0 lightingInputs=0x0 textureless=0 signed=0 names=[] consts=[] dropped=[]",
                     "no ctab");
    }
  };

} // anonymous namespace

int main() {
  try {
    MaterialIdentityTestApp::run();
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
