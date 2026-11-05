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
#include "rtx_player_model_mesh_classifier.h"

#include "../../util/log/log.h"
#include "../../util/util_string.h"

namespace dxvk {

  PlayerModelMeshClassifier::Change PlayerModelMeshClassifier::classifySection(const void* pMesh,
                                                                               const XXH64_hash_t topologyHash,
                                                                               const XXH64_hash_t materialHash,
                                                                               const bool isTagged) {
    // Only meshes with a tagged section are tracked, so most draws stop here.
    auto meshIt = m_taggedMeshes.find(pMesh);
    if (meshIt == m_taggedMeshes.end()) {
      if (!isTagged) {
        return Change::None;
      }
      meshIt = m_taggedMeshes.try_emplace(pMesh).first;
      meshIt->second.firstScene = m_scene;
    }

    TaggedMesh& mesh = meshIt->second;
    mesh.lastScene = m_scene;

    Section& section = mesh.sections[topologyHash];
    if (section.lastScene != m_scene) {
      section.lastScene = m_scene;
      section.isTagged = false;
    }
    section.isTagged |= isTagged;
    section.materialHash = materialHash;

    Change change = Change::None;
    if (mesh.isPlayerMesh) {
      if (!isTagged) {
        change = Change::AddPlayerModel;
      }
    } else if (isTagged &&
               (lookupHash(m_playerMeshSectionHashes, topologyHash) || lookupHash(m_playerMeshSectionHashes, materialHash))) {
      change = Change::RemovePlayerModel;
    }

    if (change != Change::None && m_loggedSections.insert(topologyHash).second) {
      const char* outcome = change == Change::AddPlayerModel
        ? " is player model through its mesh, most of whose sections are tagged."
        : " shares its player-model tag with a section of the player's mesh, but is on another mesh, so it stays world geometry.";
      Logger::info(str::format(
        "[RTX-PlayerModel] Section 0x", std::hex, topologyHash, " (material 0x", materialHash, ")", std::dec, outcome));
    }
    return change;
  }

  void PlayerModelMeshClassifier::endScene() {
    const uint32_t scene = m_scene;
    m_playerMeshSectionHashes.clear();
    for (auto meshIt = m_taggedMeshes.begin(); meshIt != m_taggedMeshes.end();) {
      TaggedMesh& mesh = meshIt->second;
      if (scene - mesh.lastScene > kMaxUnseenScenes) {
        meshIt = m_taggedMeshes.erase(meshIt);
        continue;
      }

      // A mesh keeps its decision through scenes that do not draw it. Its first scene can miss sections drawn
      // before its first tagged one, so it is decided from its second.
      if (mesh.lastScene == scene && mesh.firstScene != scene) {
        mesh.sections.erase_if([scene](const auto& sectionIt) {
          return scene - sectionIt->second.lastScene > kMaxUnseenScenes;
        });

        uint32_t sectionCount = 0;
        uint32_t taggedCount = 0;
        for (const auto& sectionEntry : mesh.sections) {
          const Section& section = sectionEntry.second;
          if (section.lastScene == scene) {
            ++sectionCount;
            if (section.isTagged) {
              ++taggedCount;
            }
          }
        }

        const bool isPlayerMesh = taggedCount * 2 > sectionCount;
        if (isPlayerMesh && !mesh.isPlayerMesh) {
          Logger::info(str::format("[RTX-PlayerModel] A mesh with ", taggedCount, " of its ", sectionCount,
                                   " sections tagged is the player's."));
        }
        mesh.isPlayerMesh = isPlayerMesh;
      }

      if (mesh.isPlayerMesh) {
        for (const auto& sectionEntry : mesh.sections) {
          const Section& section = sectionEntry.second;
          if (section.lastScene == mesh.lastScene) {
            m_playerMeshSectionHashes.insert(sectionEntry.first);
            if (section.materialHash != kEmptyHash) {
              m_playerMeshSectionHashes.insert(section.materialHash);
            }
          }
        }
      }
      ++meshIt;
    }
    ++m_scene;
  }

  void PlayerModelMeshClassifier::clear() {
    m_taggedMeshes.clear();
    m_playerMeshSectionHashes.clear();
  }

}
