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

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "rtx_mipmap.h"
#include "rtx_texture.h"
#include "../dxvk_sampler.h"
#include "../dxvk_shader.h"
#include "../../util/util_fast_cache.h"
#include "../../util/util_matrix.h"
#include "../../util/util_vector.h"

namespace dxvk {

  // The bake's vertex shader draws two quads of opposite winding, so whichever face the game's
  // cull mode keeps covers the target.
  constexpr uint32_t kUe3CompositeBakeVertexCount = 12;

  // Holds everything a composite bake (rtx.ue3CompositeMaterialTextures) needs that the CS thread cannot
  // read from the draw state bound for it. It is built once per key on the application thread and shared
  // by every draw of that key. See "Composite materials" in UE3Compatibility.md.
  struct Ue3CompositeBakeRequest {
    // Hashes every input the bake reads, so a different key is a different bake.
    XXH64_hash_t key = kEmptyHash;
    // Hashes the material, pixel shader and UV set. A newer bake of the same lineage, such as one with
    // streamed-in mips, replaces an older one, while the material's other lightmap compiles bake alongside.
    XXH64_hash_t lineage = kEmptyHash;
    XXH64_hash_t materialHash = kEmptyHash;
    XXH64_hash_t textureSetShaderHash = kEmptyHash;
    XXH64_hash_t pickTextureHash = kEmptyHash;
    uint32_t width = 0;
    uint32_t height = 0;
    // Spans one repeat of every layer, in the interpolant UV the pixel shader reads.
    Vector2 period = Vector2(1.0f, 1.0f);
    // Is the surface's texture transform while it renders the bake, which maps mesh UV to bake UV.
    Matrix4 domainTransform;
    // Is set when a pass renders the emission alone, which is taken out of the Emissive + Diffuse the main pass renders.
    bool emissivePass = false;
    // Is set when the emission is also kept as the emissive texture, rather than only taken out of the albedo
    // (Ue3CompositeDomain::storeEmission).
    bool storeEmission = false;
    // Is set when no texture layer emits, so the emission is the same over the whole bake and one texel of it
    // is rendered and kept.
    bool uniformEmission = false;
    // Is set when the main pass's alpha marks the texels the shader did not clip.
    bool coverageFromAlpha = false;

    struct ConstantOverride {
      uint32_t reg = 0;
      Vector4 diffusePass;
      Vector4 emissivePass;
    };
    // Lists the pixel shader float registers set for each pass.
    std::vector<ConstantOverride> constants;
    uint32_t psConstantBufferSlot = 0;
    // Is the byte offset of the float registers in the bound constant buffer.
    uint32_t psFloatConstantsOffset = 0;
    // Lists the resource slots bound to white for the bake, which hold the simple lightmap compile's lightmap.
    std::vector<uint32_t> whiteTextureSlots;

    // Describes a mask the repeat leaves out (DxsoUe3CompositeMaskLayer). Its resource slot is bound black for
    // the passes and white for a third, which bakes the albedo the surface blends towards by the mask. A
    // quadratic mask is bound to 1/2 for a fourth, which the surface interpolates through.
    struct Mask {
      uint32_t slot = 0;
      // Holds the game's texture as a unorm view. `isSrgb` says whether the game samples it decoded.
      TextureRef texture;
      bool isSrgb = false;
      uint8_t channel = 0;
      uint8_t degree = 1;
      // Maps bake UV to the mask's UV as (scale U, scale V, offset U, offset V).
      Vector4 transform = Vector4(1.0f, 1.0f, 0.0f, 0.0f);
    };
    std::optional<Mask> mask;
    Rc<DxvkShader> vertexShader;
    // Replaces vertexShader for the emission pass when its interpolants differ, as the simple vertex lightmap
    // compile's black lighting does.
    Rc<DxvkShader> emissiveVertexShader;
    bool log = false;
    // Lists the layers with their tilings, for the log and the dev menu.
    std::string description;
  };

  class Ue3CompositeBaker {
  public:
    struct Entry {
      RtxMipmap::Resource albedo;
      RtxMipmap::Resource emissive;
      // These hold, for a mask left out of the repeat, the albedo with the mask at 1 (and at 1/2 for a
      // quadratic mask) and how the surface reads the mask.
      RtxMipmap::Resource maskedAlbedo;
      RtxMipmap::Resource maskedHalfAlbedo;
      std::optional<Ue3CompositeBakeRequest::Mask> mask;
      Rc<DxvkSampler> sampler;
      // Hashes the key and the bake's generation. The surface's material identity carries it, so the
      // preserve path rebinds when a key bakes again.
      XXH64_hash_t identity = kEmptyHash;
      VkDeviceSize bytes = 0;
      uint32_t lastUsedFrame = 0;
      uint32_t bakedFrame = 0;
      // Is set when a newer bake of the same lineage exists. This one is released once no draw uses it.
      bool superseded = false;
      XXH64_hash_t lineage = kEmptyHash;
      XXH64_hash_t materialHash = kEmptyHash;
      XXH64_hash_t pickTextureHash = kEmptyHash;
      uint32_t width = 0;
      uint32_t height = 0;
      Vector2 period = Vector2(1.0f, 1.0f);
      std::string description;
    };

    struct Status {
      XXH64_hash_t materialHash = kEmptyHash;
      XXH64_hash_t pickTextureHash = kEmptyHash;
      uint32_t width = 0;
      uint32_t height = 0;
      Vector2 period = Vector2(1.0f, 1.0f);
      bool hasEmission = false;
      VkDeviceSize bytes = 0;
      uint32_t lastUsedFrame = 0;
      std::string description;
    };

    Ue3CompositeBaker() = default;
    ~Ue3CompositeBaker() = default;

    // Settled keys no longer need their draw's state applied, because they are baked, failed to bake,
    // or a replacement only needs their UV repeat. The application thread reads them.
    static bool isSettled(XXH64_hash_t key);

    // Releases every bake at the end of the frame (dev menu).
    static void requestClear() {
      s_clearRequested.store(true, std::memory_order_relaxed);
    }

    static std::vector<Status> getStatus();
    static VkDeviceSize getTotalBytes();

    const Entry* find(XXH64_hash_t key, uint32_t frame);
    // Returns the lineage's newest bake over the same repeat, which a draw renders while its own key waits.
    const Entry* findLatest(XXH64_hash_t lineage, const Vector2& period, uint32_t frame);
    const Entry* store(XXH64_hash_t key, Entry&& entry);
    // Returns whether a key not baked yet has been drawn long enough to bake. A material's layers mint a
    // key for every mip they stream in, and baking each one would bake several times in a second.
    bool hasHeld(XXH64_hash_t key, uint32_t frame);
    void markSettled(XXH64_hash_t key);
    void markUnsettled(XXH64_hash_t key);

    // A key whose draw state cannot be baked from is not retried until the bakes are cleared.
    void markFailed(XXH64_hash_t key);
    bool hasFailed(XXH64_hash_t key) const {
      return m_failedKeys.count(key) != 0;
    }

    // Returns false once this frame's bakes are spent (rtx.d3d9.ue3CompositeMaxBakesPerFrame).
    bool takeBakeSlot(uint32_t frame);

    // Returns rtx.d3d9.ue3CompositeEmissiveIntensity, read once a frame.
    float emissiveIntensity(uint32_t frame);

    uint64_t nextGeneration() {
      return ++m_generation;
    }

    // Returns a 1x1 storage image for the encode pass's unused output.
    const Resources::Resource& getDummyStorage(Rc<DxvkContext>& ctx);
    // Returns a 1x1 texture of zeros, which a mask is bound to for the passes that leave it out.
    const Rc<DxvkImageView>& getBlackTexture(Rc<DxvkContext>& ctx);
    // Returns a 1x1 float texture of exactly 1/2, for a quadratic mask's middle pass.
    const Rc<DxvkImageView>& getHalfTexture(Rc<DxvkContext>& ctx);

    void onFrameEnd(Rc<DxvkContext> ctx, uint32_t frame);

  private:
    void release(Rc<DxvkContext>& ctx, XXH64_hash_t key, Entry& entry, const char* reason);
    void publishStatus();

    fast_unordered_cache<Entry> m_entries;
    fast_unordered_set m_failedKeys;
    // Holds the frame each key still waiting to bake was first drawn.
    fast_unordered_cache<uint32_t> m_firstDrawnFrames;
    VkDeviceSize m_totalBytes = 0;
    uint32_t m_bakeFrame = ~0u;
    uint32_t m_bakesThisFrame = 0;
    uint32_t m_maxBakesPerFrame = 0;
    uint32_t m_optionsFrame = ~0u;
    float m_emissiveIntensity = 1.0f;
    uint64_t m_generation = 0;
    Resources::Resource m_dummyStorage;
    Rc<DxvkImageView> m_blackTexture;
    Rc<DxvkImageView> m_halfTexture;

    inline static std::mutex s_mutex;
    inline static fast_unordered_set s_settledKeys;
    inline static std::vector<Status> s_status;
    inline static VkDeviceSize s_totalBytes = 0;
    inline static std::atomic<bool> s_clearRequested { false };
  };

}
