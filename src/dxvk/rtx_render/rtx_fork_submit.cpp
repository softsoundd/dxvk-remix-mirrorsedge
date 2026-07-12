// src/dxvk/rtx_render/rtx_fork_submit.cpp
//
// Fork-owned file. Contains the implementations of fork_hooks:: functions
// for the SceneManager::submitExternalDraw path, lifted from
// rtx_scene_manager.cpp during the 2026-04-18 fork touchpoint-pattern refactor.
//
// See docs/fork-touchpoints.md for the full fork-hooks catalogue.
//
// NOTE: externalDrawObjectPicking accesses SceneManager::m_drawCallMeta, which
// is a private member. This file requires that SceneManager declare
// fork_hooks::externalDrawObjectPicking as a friend, OR that DrawCallMetaInfo
// and m_drawCallMeta be made accessible via a public accessor. Flagged for
// Phase 4 build-validation fixup.

#include "rtx_fork_hooks.h"

#include "rtx_asset_replacer.h"   // AssetReplacer, AssetReplacement
#include "rtx_options.h"          // RtxOptions::*, fast_unordered_set, InstanceCategories
#include "rtx_scene_manager.h"    // SceneManager, DrawCallMetaInfo

#include "dxvk_device.h"          // DxvkDevice::getCommon()->getResources()

namespace dxvk {
namespace fork_hooks {

  // ---------------------------------------------------------------------------
  // externalDrawMeshReplacement
  //
  // Checks for a USD mesh/light replacement keyed on the API mesh handle hash,
  // same as the D3D9 draw-call path. Returns the replacement vector pointer if
  // one is found (caller must call determineMaterialData + drawReplacements and
  // then return), or null if no replacement exists.
  // ---------------------------------------------------------------------------
  std::vector<AssetReplacement>* externalDrawMeshReplacement(
      AssetReplacer& replacer, XXH64_hash_t meshHash) {
    // Check for mesh/light replacements keyed on the API mesh handle, same as
    // the D3D9 draw-call path. This lets .usd replacements target API-submitted
    // meshes (e.g. hash the remix API mesh handle and author a replacement).
    return replacer.getReplacementsForMesh(meshHash);
  }

  // ---------------------------------------------------------------------------
  // externalDrawMaterialReplacement
  //
  // Checks for a USD material replacement via getReplacementMaterial() and
  // updates the caller's material pointer in-place if one is found.
  // ---------------------------------------------------------------------------
  void externalDrawMaterialReplacement(
      AssetReplacer& replacer, const MaterialData*& material) {
    // Check for material replacement (matches the D3D9 draw path behavior).
    MaterialData* pReplacementMaterial = replacer.getReplacementMaterial(material->getHash());
    if (pReplacementMaterial != nullptr) {
      material = pReplacementMaterial;
    }
  }

  // ---------------------------------------------------------------------------
  // externalDrawTextureCategories
  //
  // Auto-applies all texture-based instance categories for API-submitted draws.
  // Resolves the albedo texture hash from the material's opaque data and writes
  // it to outTextureHash (so subsequent hooks — e.g. object-picking — can use
  // it). Then looks the hash up against every RtxOption category set.
  // ---------------------------------------------------------------------------
  void externalDrawTextureCategories(
      const MaterialData* material,
      DrawCallState& drawCall,
      XXH64_hash_t& outTextureHash) {
    // Auto-apply texture categories for API-submitted content (matches D3D9 behavior).
    // For API materials, the albedo texture hash is what D3D9's setupCategoriesForTexture()
    // pattern normally keys off, so look it up directly from the material's opaque data.
    if (material->getType() == MaterialDataType::Opaque) {
      const auto& opaqueMat = material->getOpaqueMaterialData();
      if (opaqueMat.getAlbedoOpacityTexture().isValid()) {
        outTextureHash = opaqueMat.getAlbedoOpacityTexture().getImageHash();
      }
    }

    if (outTextureHash != 0 && outTextureHash != kEmptyHash) {
      auto applyCategory = [&](const fast_unordered_set& hashSet, InstanceCategories cat) {
        if (hashSet.find(outTextureHash) != hashSet.end()) {
          drawCall.setCategory(cat, true);
        }
      };

      applyCategory(RtxOptions::skyBoxTextures(), InstanceCategories::Sky);
      applyCategory(RtxOptions::ignoreTextures(), InstanceCategories::Ignore);
      applyCategory(RtxOptions::worldSpaceUiTextures(), InstanceCategories::WorldUI);
      applyCategory(RtxOptions::worldSpaceUiBackgroundTextures(), InstanceCategories::WorldMatte);
      applyCategory(RtxOptions::particleTextures(), InstanceCategories::Particle);
      applyCategory(RtxOptions::beamTextures(), InstanceCategories::Beam);
      applyCategory(RtxOptions::decalTextures(), InstanceCategories::DecalStatic);
      applyCategory(RtxOptions::terrainTextures(), InstanceCategories::Terrain);
      applyCategory(RtxOptions::animatedWaterTextures(), InstanceCategories::AnimatedWater);
      applyCategory(RtxOptions::ignoreLights(), InstanceCategories::IgnoreLights);
      applyCategory(RtxOptions::antiCullingTextures(), InstanceCategories::IgnoreAntiCulling);
      applyCategory(RtxOptions::motionBlurMaskOutTextures(), InstanceCategories::IgnoreMotionBlur);
      applyCategory(RtxOptions::hideInstanceTextures(), InstanceCategories::Hidden);
    }
  }

  // ---------------------------------------------------------------------------
  // externalDrawObjectPicking
  //
  // Stores per-draw texture hash metadata in SceneManager::m_drawCallMeta when
  // object picking is active, mirroring the D3D9 draw path which populates
  // m_drawCallMeta in processDrawCallState. API draws supply their own
  // drawCallID via remixapi_InstanceInfoObjectPickingEXT, so we store it here.
  //
  // ACCESS NOTE: this function uses SceneManager::m_drawCallMeta (private) and
  // SceneManager::DrawCallMetaInfo (private nested type). A friend declaration
  // for this function is required in SceneManager, or the member / type must be
  // made accessible via a public helper. See file-level comment above.
  // ---------------------------------------------------------------------------
  void externalDrawObjectPicking(
      DxvkDevice& device,
      DrawCallState& drawCall,
      XXH64_hash_t textureHash,
      SceneManager& scene) {
    // Store texture hash metadata for object picking (mirrors the D3D9 draw
    // path which populates m_drawCallMeta in processDrawCallState). API draws
    // supply their own drawCallID via remixapi_InstanceInfoObjectPickingEXT,
    // so we hash it in directly here.
    const bool objectPickingActive = device.getCommon()->getResources().getRaytracingOutput()
      .m_primaryObjectPicking.isValid();
    if (objectPickingActive && drawCall.drawCallID != 0 &&
        textureHash != 0 && textureHash != kEmptyHash) {
      auto meta = SceneManager::DrawCallMetaInfo {};
      meta.legacyTextureHash = textureHash;

      std::lock_guard lock { scene.m_drawCallMeta.mutex };
      auto [iter, isNew] = scene.m_drawCallMeta.infos[scene.m_drawCallMeta.ticker].emplace(drawCall.drawCallID, meta);
      ONCE_IF_FALSE(isNew, Logger::warn(
        "Found multiple API draw calls with the same \'objectPickingValue\'. "
        "Some objects might not be available through object picking"));
    }
  }

} // namespace fork_hooks
} // namespace dxvk
