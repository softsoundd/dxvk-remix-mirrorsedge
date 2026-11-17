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

// Tests a replacement emitting a UE3 composite material's bake (emissive_from_composite_bake):
// the switch read from USD, the bake texture the merge puts in the replacement's emissive texture,
// and the mask layer bits that blend a lifted mask into it. See "Emission from the bake" in UE3Compatibility.md.

#include <array>
#include <cstdint>
#include <cstring>

#include "../../test_utils.h"
#include "../../../src/dxvk/rtx_render/rtx_materials.h"

#include "../../../src/lssusd/usd_include_begin.h"
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/stage.h>
#include "../../../src/lssusd/usd_include_end.h"

namespace dxvk {

  Logger Logger::s_instance("test_ue3_composite_emission.log");

  namespace {

    class TestAsset final : public AssetData {
    public:
      explicit TestAsset(const XXH64_hash_t hash) { m_hash = hash; }
      const void* data(int, int) override { return nullptr; }
      void placement(int, int, int, uint64_t& offset, size_t& size) const override { offset = 0; size = 0; }
      void evictCache(int, int) override { }
      void releaseSource() override { }
    };

    void require(bool condition, const char* pMessage) {
      if (!condition) {
        throw DxvkError(pMessage);
      }
    }

    // Returns a replacement texture, which needs no device.
    TextureRef makeTexture(const XXH64_hash_t hash) {
      Rc<ManagedTexture> pTexture = new ManagedTexture;
      pTexture->m_uniqueKey = hash;
      pTexture->m_assetData = new TestAsset(hash);
      return TextureRef(pTexture);
    }

    void testDeserialization() {
      const auto stage = pxr::UsdStage::CreateInMemory();
      const auto shader = stage->DefinePrim(pxr::SdfPath("/Shader"), pxr::TfToken("Shader"));
      auto load = [&]() {
        return OpaqueMaterialData::deserialize([](const pxr::UsdPrim&, const pxr::TfToken&) { return TextureRef {}; }, shader);
      };
      require(!load().getEmissiveFromCompositeBake(), "Emission from the bake must be off by default.");
      shader.CreateAttribute(pxr::TfToken("inputs:emissive_from_composite_bake"), pxr::SdfValueTypeNames->Bool).Set(true);
      require(load().getEmissiveFromCompositeBake(), "emissive_from_composite_bake must reach the material from USD.");
    }

    void testMerge() {
      const TextureRef authored = makeTexture(0x1111);
      const TextureRef albedo = makeTexture(0x2222);
      const TextureRef keptEmission = makeTexture(0x3333);

      // Returns the unique key of the emissive texture a replacement renders with over the game's material.
      auto mergedEmission = [&](const bool enableEmission, const bool fromBake, const LegacyMaterialData& legacy) {
        OpaqueMaterialData replacement;
        replacement.setEnableEmission(enableEmission);
        replacement.setEmissiveFromCompositeBake(fromBake);
        replacement.setEmissiveColorTexture(authored);
        MaterialData material(replacement);
        material.mergeLegacyMaterial(legacy);
        return material.getOpaqueMaterialData().getEmissiveColorTexture().getUniqueKey();
      };

      LegacyMaterialData unlit;
      unlit.ue3CompositeAlbedo = albedo;
      unlit.ue3CompositeAlbedoIsEmission = true;
      require(mergedEmission(true, true, unlit) == albedo.getUniqueKey(),
              "A replacement that emits an unlit material's bake must emit its albedo.");
      require(mergedEmission(true, false, unlit) == authored.getUniqueKey(),
              "A replacement that does not emit the bake must keep its own emissive texture.");
      require(mergedEmission(false, true, unlit) == authored.getUniqueKey(),
              "A replacement with its emission off must keep its own emissive texture.");

      LegacyMaterialData lit;
      lit.ue3CompositeAlbedo = albedo;
      lit.ue3CompositeEmissive = keptEmission;
      require(mergedEmission(true, true, lit) == keptEmission.getUniqueKey(),
              "A replacement that emits a lit material's bake must emit the emission the bake keeps.");

      LegacyMaterialData litWithoutEmission;
      litWithoutEmission.ue3CompositeAlbedo = albedo;
      require(mergedEmission(true, true, litWithoutEmission) == authored.getUniqueKey(),
              "A bake that keeps no emission must leave the replacement's own emissive texture.");
      require(mergedEmission(true, true, LegacyMaterialData {}) == authored.getUniqueKey(),
              "A material without a bake must leave the replacement's own emissive texture.");
    }

    void testMaskLayerPacking() {
      // Is the byte offset of MaskLayerMaterial::maskedTextures in surface_material.h.
      constexpr size_t kMaskedTexturesOffset = 26;

      auto makeMaskLayer = [](const uint32_t maskedTextures) {
        return RtMaskLayerMaterial(1, kSurfaceMaterialInvalidTextureIndex, 2, Vector4(0.1f, 0.1f, 0.0f, 0.0f), 0, false,
                                   Vector2(10.0f, 10.0f), MASK_LAYER_MATERIAL_BAKE_ALBEDO, maskedTextures);
      };
      const RtMaskLayerMaterial emissive = makeMaskLayer(MASK_LAYER_MATERIAL_BAKE_EMISSIVE);
      std::array<unsigned char, kSurfaceMaterialGPUSize> packed {};
      size_t offset = 0;
      emissive.writeGPUData(packed.data(), offset);
      uint16_t maskedTextures = 0;
      std::memcpy(&maskedTextures, packed.data() + kMaskedTexturesOffset, sizeof(maskedTextures));
      require(maskedTextures == MASK_LAYER_MATERIAL_BAKE_EMISSIVE,
              "maskedTextures must be packed where MaskLayerMaterial reads it.");
      require(!(emissive == makeMaskLayer(MASK_LAYER_MATERIAL_BAKE_ALBEDO)),
              "Mask layers that blend into different textures must not share a cache entry.");
    }

  }

}

int main() {
  try {
    dxvk::testDeserialization();
    dxvk::testMerge();
    dxvk::testMaskLayerPacking();
  } catch (const dxvk::DxvkError& error) {
    std::cerr << "TEST FAILED: " << error.message() << std::endl;
    return -1;
  } catch (const std::exception& error) {
    std::cerr << "TEST FAILED with exception: " << error.what() << std::endl;
    return -1;
  }

  std::cout << "All UE3 composite emission tests passed." << std::endl;
  return 0;
}
