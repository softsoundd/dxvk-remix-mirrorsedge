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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../util/xxHash/xxhash.h"

// These helpers hash the mip tail of a texture with 8-bit channels as UE3 derives a reduced variant's tail from its
// top tail mip. See "Streaming-stable texture hashes" in documentation/UE3Compatibility.md.

namespace dxvk {

  // Halves a mip of tightly packed texels with the 2x2 box filter the game mips a reduced variant with, which rounds
  // down. A side of 1 repeats its texels.
  inline std::vector<uint8_t> halveUe3ByteChannelMip(const std::vector<uint8_t>& texels, const uint32_t width,
                                                     const uint32_t height, const uint32_t texelBytes) {
    const uint32_t halfWidth = std::max(1u, width / 2);
    const uint32_t halfHeight = std::max(1u, height / 2);
    std::vector<uint8_t> half(size_t(halfWidth) * halfHeight * texelBytes);
    for (uint32_t y = 0; y < halfHeight; y++) {
      const size_t row0 = size_t(std::min(2 * y, height - 1)) * width;
      const size_t row1 = size_t(std::min(2 * y + 1, height - 1)) * width;
      for (uint32_t x = 0; x < halfWidth; x++) {
        const size_t column0 = std::min(2 * x, width - 1);
        const size_t column1 = std::min(2 * x + 1, width - 1);
        for (uint32_t channel = 0; channel < texelBytes; channel++) {
          const uint32_t sum = uint32_t(texels[(row0 + column0) * texelBytes + channel]) +
                               uint32_t(texels[(row0 + column1) * texelBytes + channel]) +
                               uint32_t(texels[(row1 + column0) * texelBytes + channel]) +
                               uint32_t(texels[(row1 + column1) * texelBytes + channel]);
          half[(size_t(y) * halfWidth + x) * texelBytes + channel] = uint8_t(sum / 4);
        }
      }
    }
    return half;
  }

  // Returns the hash of a tail of `levels` mips chained onto `seed` from the smallest mip upward: the top tail mip and
  // the mips the game derives from it, each in the layout of a mip buffer, whose rows are aligned to 4 bytes.
  // topMip holds the top tail mip's rows, `pitch` bytes apart.
  inline XXH64_hash_t hashUe3DerivedTail(const uint8_t* topMip, const size_t pitch, const uint32_t width,
                                         const uint32_t height, const uint32_t texelBytes, const uint32_t levels,
                                         const XXH64_hash_t seed) {
    if (levels == 0) {
      return seed;
    }

    auto levelWidth = [width](const uint32_t level) { return std::max(1u, width >> level); };
    auto levelHeight = [height](const uint32_t level) { return std::max(1u, height >> level); };

    std::vector<std::vector<uint8_t>> tail(levels);
    const size_t topRowBytes = size_t(width) * texelBytes;
    tail[0].resize(topRowBytes * height);
    for (uint32_t row = 0; row < height; row++) {
      std::memcpy(tail[0].data() + row * topRowBytes, topMip + row * pitch, topRowBytes);
    }
    for (uint32_t level = 1; level < levels; level++) {
      tail[level] = halveUe3ByteChannelMip(tail[level - 1], levelWidth(level - 1), levelHeight(level - 1), texelBytes);
    }

    XXH64_hash_t hash = seed;
    std::vector<uint8_t> buffer;
    for (uint32_t level = levels; level-- > 0;) {
      const size_t rowBytes = size_t(levelWidth(level)) * texelBytes;
      const size_t bufferPitch = (rowBytes + 3) & ~size_t(3);
      buffer.assign(bufferPitch * levelHeight(level), 0);
      for (uint32_t row = 0; row < levelHeight(level); row++) {
        std::memcpy(buffer.data() + row * bufferPitch, tail[level].data() + row * rowBytes, rowBytes);
      }
      hash = XXH3_64bits_withSeed(buffer.data(), buffer.size(), hash);
    }
    return hash;
  }

}
