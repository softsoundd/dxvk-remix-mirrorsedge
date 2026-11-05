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

#include <cstdint>
#include <iterator>

#include "../../test_utils.h"
#include "../../../src/dxvk/rtx_render/rtx_player_model_mesh_classifier.h"

namespace dxvk {

  Logger Logger::s_instance("test_player_model_mesh_classifier.log");

  namespace {

    using Change = PlayerModelMeshClassifier::Change;

    void require(bool condition, const char* message) {
      if (!condition) {
        throw DxvkError(message);
      }
    }

    // The classifier only compares mesh addresses, so plain values stand in for index buffers.
    const void* const pPlayerMesh = reinterpret_cast<const void*>(uintptr_t(0x1000));
    const void* const pOtherMesh = reinterpret_cast<const void*>(uintptr_t(0x2000));

    // The player's mesh has four sections that are always tagged, and an eye.
    constexpr XXH64_hash_t kPlayerSections[] = { 0x101, 0x102, 0x103, 0x104 };
    constexpr XXH64_hash_t kPlayerMaterials[] = { 0x201, 0x202, 0x203, 0x204 };
    constexpr XXH64_hash_t kPlayerEye = 0x105;
    constexpr XXH64_hash_t kPlayerEyeMaterial = 0x205;

    // The other character's mesh has four untagged sections, and an eye that can match the player's.
    constexpr XXH64_hash_t kOtherSections[] = { 0x301, 0x302, 0x303, 0x304 };
    constexpr XXH64_hash_t kOtherMaterials[] = { 0x401, 0x402, 0x403, 0x404 };
    constexpr XXH64_hash_t kOtherEye = 0x305;
    constexpr XXH64_hash_t kOtherEyeMaterial = 0x405;

    // Draws the player's mesh, and returns the change its eye needs.
    Change drawPlayerMesh(PlayerModelMeshClassifier& classifier, bool isEyeTagged) {
      for (size_t i = 0; i < std::size(kPlayerSections); ++i) {
        require(
          classifier.classifySection(pPlayerMesh, kPlayerSections[i], kPlayerMaterials[i], true) == Change::None,
          "A tagged section of the player's mesh must keep its tag.");
      }
      return classifier.classifySection(pPlayerMesh, kPlayerEye, kPlayerEyeMaterial, isEyeTagged);
    }

    // Draws the other mesh with its untagged sections ahead of its tagged eye, and returns the change the eye needs.
    Change drawOtherMesh(PlayerModelMeshClassifier& classifier, XXH64_hash_t eye, XXH64_hash_t eyeMaterial) {
      for (size_t i = 0; i < std::size(kOtherSections); ++i) {
        require(
          classifier.classifySection(pOtherMesh, kOtherSections[i], kOtherMaterials[i], false) == Change::None,
          "An untagged section of a mostly untagged mesh must stay world geometry.");
      }
      return classifier.classifySection(pOtherMesh, eye, eyeMaterial, true);
    }

    // Draws the player's mesh for the two scenes a new mesh needs before its decision applies.
    void settlePlayerMesh(PlayerModelMeshClassifier& classifier, bool isEyeTagged) {
      for (int scene = 0; scene < 2; ++scene) {
        require(drawPlayerMesh(classifier, isEyeTagged) == Change::None, "A decision must only apply from the scene after it.");
        classifier.endScene();
      }
    }

    void testUntaggedSectionJoinsThePlayersMesh() {
      PlayerModelMeshClassifier classifier;
      settlePlayerMesh(classifier, false);
      require(
        drawPlayerMesh(classifier, false) == Change::AddPlayerModel,
        "The untagged eye of a mostly tagged mesh must be player model.");
    }

    void testSharedGeometryTagOnlyAppliesToThePlayersMesh() {
      PlayerModelMeshClassifier classifier;
      settlePlayerMesh(classifier, true);
      for (int scene = 0; scene < 3; ++scene) {
        require(drawPlayerMesh(classifier, true) == Change::None, "The player's tagged eye must keep its tag.");
        require(
          drawOtherMesh(classifier, kPlayerEye, kOtherEyeMaterial) == Change::RemovePlayerModel,
          "Another mesh's section tagged by the player's eye geometry must stay world geometry.");
        classifier.endScene();
      }
    }

    void testSharedMaterialTagOnlyAppliesToThePlayersMesh() {
      PlayerModelMeshClassifier classifier;
      settlePlayerMesh(classifier, true);
      drawPlayerMesh(classifier, true);
      require(
        drawOtherMesh(classifier, kOtherEye, kPlayerEyeMaterial) == Change::RemovePlayerModel,
        "Another mesh's section tagged by the player's eye material must stay world geometry.");
    }

    void testUnsharedTagOnAnotherMeshApplies() {
      PlayerModelMeshClassifier classifier;
      settlePlayerMesh(classifier, false);
      for (int scene = 0; scene < 3; ++scene) {
        drawPlayerMesh(classifier, false);
        require(
          drawOtherMesh(classifier, kOtherEye, kOtherEyeMaterial) == Change::None,
          "A tag that matches no section of the player's mesh must apply.");
        classifier.endScene();
      }
    }

    void testHalfTaggedMeshIsNotThePlayers() {
      PlayerModelMeshClassifier classifier;
      for (int scene = 0; scene < 3; ++scene) {
        require(
          classifier.classifySection(pOtherMesh, kOtherSections[0], kOtherMaterials[0], true) == Change::None,
          "A tagged section must keep its tag while no mesh is the player's.");
        require(
          classifier.classifySection(pOtherMesh, kOtherSections[1], kOtherMaterials[1], false) == Change::None,
          "A mesh with only half of its sections tagged must not be the player's.");
        classifier.endScene();
      }
    }

    void testFirstSceneIsNotDecided() {
      // The mesh's first scene only counts its tagged eye, which comes after its four untagged sections.
      // Deciding from it would make the mesh the player's, and the untagged sections would fail drawOtherMesh.
      PlayerModelMeshClassifier classifier;
      for (int scene = 0; scene < 4; ++scene) {
        require(
          drawOtherMesh(classifier, kOtherEye, kOtherEyeMaterial) == Change::None,
          "A tag must apply while no mesh is the player's.");
        classifier.endScene();
      }
    }

    void testDecisionOutlastsScenesWithoutTheMesh() {
      PlayerModelMeshClassifier classifier;
      settlePlayerMesh(classifier, false);
      for (int scene = 0; scene < 100; ++scene) {
        require(
          drawOtherMesh(classifier, kPlayerEye, kOtherEyeMaterial) == Change::RemovePlayerModel,
          "The player's mesh must keep its decision through scenes that do not draw it.");
        classifier.endScene();
      }
      require(
        drawPlayerMesh(classifier, false) == Change::AddPlayerModel,
        "The player's mesh must still be the player's when it is drawn again.");
    }

    void testUnseenMeshIsForgotten() {
      PlayerModelMeshClassifier classifier;
      settlePlayerMesh(classifier, false);
      for (uint32_t scene = 0; scene <= PlayerModelMeshClassifier::kMaxUnseenScenes; ++scene) {
        classifier.endScene();
      }
      require(drawPlayerMesh(classifier, false) == Change::None, "A mesh unseen for too long must be decided again.");
      require(
        drawOtherMesh(classifier, kPlayerEye, kOtherEyeMaterial) == Change::None,
        "A forgotten player's mesh must no longer claim its tags.");
    }

    void testClearForgetsDecisions() {
      PlayerModelMeshClassifier classifier;
      settlePlayerMesh(classifier, false);
      classifier.clear();
      require(drawPlayerMesh(classifier, false) == Change::None, "Clearing must forget every decision.");
    }

  }

}

int main() {
  try {
    dxvk::testUntaggedSectionJoinsThePlayersMesh();
    dxvk::testSharedGeometryTagOnlyAppliesToThePlayersMesh();
    dxvk::testSharedMaterialTagOnlyAppliesToThePlayersMesh();
    dxvk::testUnsharedTagOnAnotherMeshApplies();
    dxvk::testHalfTaggedMeshIsNotThePlayers();
    dxvk::testFirstSceneIsNotDecided();
    dxvk::testDecisionOutlastsScenesWithoutTheMesh();
    dxvk::testUnseenMeshIsForgotten();
    dxvk::testClearForgetsDecisions();
  } catch (const dxvk::DxvkError& error) {
    std::cerr << "TEST FAILED: " << error.message() << std::endl;
    return -1;
  } catch (const std::exception& error) {
    std::cerr << "TEST FAILED with exception: " << error.what() << std::endl;
    return -1;
  }

  std::cout << "All player-model mesh classifier tests passed." << std::endl;
  return 0;
}
