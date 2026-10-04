#include "d3d9_common_texture.h"
#include "d3d9_rtx.h"

#include <numeric>
#include <unordered_set>

namespace dxvk {

  constexpr uint32_t kUe3TailMaxDimension = 64;

  // A hash of the mips at or below kUe3TailMaxDimension, which every streamed variant of a texture
  // shares (see "Streaming-stable texture hashes" in UE3Compatibility.md). `source` supplies the CPU mip buffers,
  // as in SetupForRtxFrom. kEmptyHash when ineligible.
  XXH64_hash_t D3D9CommonTexture::ComputeUe3StreamingStableHash(const D3D9CommonTexture* source) const {
    if (!D3D9Rtx::ue3StreamingStableTextureHashing() || !D3D9Rtx::ue3EngineMode()) {
      return kEmptyHash;
    }
    if (m_desc.MipLevels <= 1 || IsRenderTarget()) {
      return kEmptyHash;
    }

    // A texture at or below the tail size is all tail, and must stay so: it is also the state every larger
    // texture streams in through (see "Material identity and replacement anchor stability" in UE3Compatibility.md).
    XXH64_hash_t tailHash = kEmptyHash;
    uint32_t tailMipCount = 0;
    // chain from the smallest mip upward so the value is independent of how many
    // larger mips this particular variant has
    for (int32_t mip = int32_t(m_desc.MipLevels) - 1; mip >= 0; mip--) {
      const uint32_t mipWidth = std::max(1u, m_desc.Width >> mip);
      const uint32_t mipHeight = std::max(1u, m_desc.Height >> mip);
      if (std::max(mipWidth, mipHeight) > kUe3TailMaxDimension) {
        break;
      }

      const auto& mipBuffer = source->m_buffers[mip];
      if (mipBuffer.ptr() == nullptr) {
        return kEmptyHash;
      }

      tailHash = XXH3_64bits_withSeed(mipBuffer->mapPtr(0), mipBuffer->info().size, tailHash);
      tailMipCount++;
    }

    if (tailMipCount == 0) {
      return kEmptyHash;
    }

    struct TailIdentitySeed {
      uint32_t format;
      uint32_t aspectW;
      uint32_t aspectH;
    };
    const uint32_t aspectGcd = std::max(1u, std::gcd(m_desc.Width, m_desc.Height));
    const TailIdentitySeed seed = {
      uint32_t(m_desc.Format),
      m_desc.Width / aspectGcd,
      m_desc.Height / aspectGcd,
    };
    return XXH3_64bits_withSeed(&seed, sizeof(seed), tailHash);
  }

  // Keyed on (descriptor, image) rather than descriptor alone: identically shaped textures share a
  // descriptor, so a texture whose content hash moves between runs would otherwise be hidden
  // behind the first value seen. See ue3LogTextureHashProvenance.
  void D3D9CommonTexture::LogUe3TextureHashProvenance(
      const D3D9CommonTexture* source, const XXH64_hash_t imageHash, const bool is2DTexture) const {
    if (!D3D9Rtx::ue3LogTextureHashProvenance() || source == nullptr) {
      return;
    }

    const XXH64_hash_t descriptorHash = m_desc.CalculateHash();

    static dxvk::mutex s_mutex;
    static std::unordered_set<XXH64_hash_t> s_logged;
    {
      const XXH64_hash_t logKey = XXH3_64bits_withSeed(&imageHash, sizeof(imageHash), descriptorHash);
      std::lock_guard<dxvk::mutex> lock(s_mutex);
      if (!s_logged.insert(logKey).second) {
        return;
      }
    }

    const bool tailEligible = D3D9Rtx::ue3StreamingStableTextureHashing() && D3D9Rtx::ue3EngineMode() &&
                              is2DTexture && m_desc.MipLevels > 1 && !IsRenderTarget();

    // needsUpload is set for every subresource at creation, so it does not distinguish "written by
    // the game" from "allocated and untouched" - it is reported as state, not as evidence that a
    // buffer holds content.
    std::string tailState;
    uint32_t tailMipCount = 0;
    for (int32_t mip = tailEligible ? int32_t(m_desc.MipLevels) - 1 : -1; mip >= 0; mip--) {
      const uint32_t mipWidth = std::max(1u, m_desc.Width >> mip);
      const uint32_t mipHeight = std::max(1u, m_desc.Height >> mip);
      if (std::max(mipWidth, mipHeight) > kUe3TailMaxDimension) {
        break;
      }
      tailMipCount++;
      tailState += str::format(
        tailState.empty() ? "" : ",", "mip", mip,
        source->m_buffers[mip].ptr() == nullptr ? "=noBuffer" : (source->NeedsUpload(mip) ? "=needsUpload" : "=uploaded"));
    }

    // Mip 0 on its own is what makes this diagnostic decisive: repeating across level loads while
    // the image hash does not means the picture is stable and only the smaller mips moved.
    const auto& topBuffer = source->m_buffers[0];
    const std::string mip0 = topBuffer.ptr() != nullptr
      ? str::format("0x", std::hex, XXH3_64bits(topBuffer->mapPtr(0), topBuffer->info().size), std::dec,
                    "(", topBuffer->info().size, "B)")
      : std::string("unavailable");

    Logger::info(str::format(
      "[RTX-Compatibility][UE3-TexHash] desc=0x", std::hex, descriptorHash,
      " image=0x", imageHash, std::dec,
      " mip0=", mip0,
      " ", m_desc.Width, "x", m_desc.Height,
      " mips=", m_desc.MipLevels,
      " fmt=", uint32_t(m_desc.Format),
      " usage=0x", std::hex, m_desc.Usage, std::dec,
      " pool=", uint32_t(m_desc.Pool),
      " rt=", IsRenderTarget() ? 1 : 0,
      " path=", tailEligible ? "tail" : "topMip",
      " tailMips=", tailMipCount,
      " [", tailState.empty() ? "-" : tailState, "]"));
  }

}
