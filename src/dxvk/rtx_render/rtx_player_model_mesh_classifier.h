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
#include <unordered_map>

#include "../../util/util_fast_cache.h"
#include "rtx_constants.h"

namespace dxvk {

  // Settles player-model tags per mesh, a mesh being the sections drawn from one static index buffer.
  // See "First person and the player model" in documentation/UE3Compatibility.md.
  class PlayerModelMeshClassifier {
  public:
    // Meshes and sections unseen for this many scenes are forgotten, as their index buffers may be gone.
    static constexpr uint32_t kMaxUnseenScenes = 3600;

    // Lists the changes a section's player-model category can need.
    enum class Change {
      None,
      // The section is untagged, and its mesh is the player's.
      AddPlayerModel,
      // The section is tagged on another mesh, and shares its geometry or material with the player's mesh.
      RemovePlayerModel,
    };

    // Records a draw of a mesh section, and returns the change its player-model category needs.
    // The mesh address is only compared, never dereferenced.
    Change classifySection(const void* pMesh, XXH64_hash_t topologyHash, XXH64_hash_t materialHash, bool isTagged);

    // Decides from the scene just drawn which meshes are the player's, for the draws of the next scene.
    void endScene();

    // Forgets every mesh and its decision.
    void clear();

  private:
    struct Section {
      XXH64_hash_t materialHash = kEmptyHash;
      uint32_t lastScene = 0;
      bool isTagged = false;
    };

    struct TaggedMesh {
      fast_unordered_cache<Section> sections;
      uint32_t firstScene = 0;
      uint32_t lastScene = 0;
      bool isPlayerMesh = false;
    };

    // Holds every mesh that has drawn a tagged section, with its sections keyed by topology hash.
    std::unordered_map<const void*, TaggedMesh> m_taggedMeshes;
    // Holds the topology and material hashes of the sections the player's meshes drew when last seen.
    fast_unordered_set m_playerMeshSectionHashes;
    // Holds the topology hashes of the sections whose change has been logged. It outlives clear().
    fast_unordered_set m_loggedSections;
    // Numbers the scene that draws are recorded into, which endScene advances.
    uint32_t m_scene = 1;
  };

}
