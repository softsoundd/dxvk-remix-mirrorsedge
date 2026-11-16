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
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../../src/d3d9/d3d9_rtx_ue3_texture_tail.h"

namespace dxvk {

  namespace {

    void require(bool condition, const std::string& message) {
      if (!condition) {
        throw std::runtime_error(message);
      }
    }

    // Chains the hashes of mip buffers from the smallest upward, as the tail of a texture of any other format is hashed.
    XXH64_hash_t hashMipBuffers(const std::vector<std::vector<uint8_t>>& buffers) {
      XXH64_hash_t hash = 0;
      for (size_t mip = buffers.size(); mip-- > 0;) {
        hash = XXH3_64bits_withSeed(buffers[mip].data(), buffers[mip].size(), hash);
      }
      return hash;
    }

    void testHalving() {
      require(halveUe3ByteChannelMip({ 1, 2, 3, 4 }, 2, 2, 1) == std::vector<uint8_t> { 2 },
              "A 2x2 mip must halve to the mean of its texels.");
      require(halveUe3ByteChannelMip({ 255, 255, 255, 254 }, 2, 2, 1) == std::vector<uint8_t> { 254 },
              "The mean must round down, as the game's filter does, not to the nearest value.");
      require(halveUe3ByteChannelMip({ 10, 13 }, 2, 1, 1) == std::vector<uint8_t> { 11 },
              "A side of 1 must repeat its texels.");
      // The four texels' channels sum to 24, 64, 104 and 144.
      require(halveUe3ByteChannelMip({ 0, 10, 20, 30, 4, 14, 24, 34, 8, 18, 28, 38, 12, 22, 32, 42 }, 2, 2, 4) ==
                std::vector<uint8_t> { 6, 16, 26, 36 },
              "The channels of an A8R8G8B8 texel must halve independently.");
    }

    void testDerivedTailHashesAsMipBuffers() {
      // A 4x4 L8 top tail mip, whose 2x2 mip the filter derives as 25, 45, 105, 125 and whose 1x1 mip as 75.
      const std::vector<uint8_t> top = { 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150 };
      // The buffers pad each row to 4 bytes with zeros.
      const std::vector<std::vector<uint8_t>> buffers = {
        top,
        { 25, 45, 0, 0, 105, 125, 0, 0 },
        { 75, 0, 0, 0 },
      };
      require(hashUe3DerivedTail(top.data(), 4, 4, 4, 1, 3, 0) == hashMipBuffers(buffers),
              "A cooked tail that the game's filter reproduces must hash as its mip buffers do.");
    }

    void testDerivedTailReadsOnlyTheTopMip() {
      // A 3x3 L8 mip's rows are 4 bytes apart in its buffer, and the byte after each row is not part of the texture.
      const std::vector<uint8_t> padded = { 1, 2, 3, 0xAA, 4, 5, 6, 0xBB, 7, 8, 9, 0xCC };
      const std::vector<uint8_t> zeroed = { 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0 };
      require(hashUe3DerivedTail(padded.data(), 4, 3, 3, 1, 2, 0) == hashUe3DerivedTail(zeroed.data(), 4, 3, 3, 1, 2, 0),
              "The bytes between a top mip's rows must not reach the hash.");

      const std::vector<uint8_t> top(64 * 64, 128);
      require(hashUe3DerivedTail(top.data(), 64, 64, 64, 1, 7, 0) != hashUe3DerivedTail(top.data(), 64, 64, 64, 1, 6, 0),
              "Every mip of the tail must reach the hash.");
      require(hashUe3DerivedTail(top.data(), 64, 64, 64, 1, 0, 42) == 42, "A tail without mips must leave the seed.");
    }

  }

}

int main() {
  try {
    dxvk::testHalving();
    dxvk::testDerivedTailHashesAsMipBuffers();
    dxvk::testDerivedTailReadsOnlyTheTopMip();
  } catch (const std::exception& error) {
    std::cerr << "TEST FAILED: " << error.what() << std::endl;
    return -1;
  }

  std::cout << "All UE3 texture tail tests passed." << std::endl;
  return 0;
}
