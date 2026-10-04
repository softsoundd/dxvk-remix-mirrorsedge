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

// Messages between the runtime and the 32-bit bridge client for the name of the map the game has loaded.
// See documentation/UE3Compatibility.md, "Per-map settings".

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace dxvk {
  // Runtime -> bridge client. wParam: hash of the map name the runtime holds, 0 for none.
  static constexpr const char* kUe3MapQueryMsgName = "UWM_REMIX_UE3_MAP_QUERY";
  // Bridge client -> runtime, answering every query. wParam: hash of the current map name, 0 for none.
  // lParam: see packUe3MapAnswer. When the hash differs from the query's, the name follows as chunks.
  static constexpr const char* kUe3MapAnswerMsgName = "UWM_REMIX_UE3_MAP_ANSWER";
  // Bridge client -> runtime. wParam and lParam: see packUe3MapChunk.
  static constexpr const char* kUe3MapChunkMsgName = "UWM_REMIX_UE3_MAP_CHUNK";

  enum class Ue3MapStatus : uint32_t {
    Ok = 0,
    // The engine has not finished loading a map yet.
    NoMap = 1,
    // The GNames or GObjects signature does not match this executable.
    SignatureNotFound = 2,
    // No TdGameEngine object exists yet.
    EngineNotFound = 3,
    // The engine's LastURL does not hold the protocol it should, so this executable's layout is not the SDK's.
    LayoutMismatch = 4,
  };

  static constexpr size_t kUe3MapNameMaxLength = 63;
  static constexpr size_t kUe3MapChunkSize = 8;

  inline uint32_t packUe3MapAnswer(const size_t length, const Ue3MapStatus status) {
    return static_cast<uint32_t>(length & 0xFFFFu) | (static_cast<uint32_t>(status) << 16);
  }

  inline size_t unpackUe3MapAnswerLength(const uint32_t answer) {
    return answer & 0xFFFFu;
  }

  inline Ue3MapStatus unpackUe3MapAnswerStatus(const uint32_t answer) {
    return static_cast<Ue3MapStatus>(answer >> 16);
  }

  // FNV-1a, never 0, which stands for no map.
  inline uint32_t hashUe3MapName(const std::string_view name) {
    uint32_t hash = 2166136261u;
    for (const char c : name) {
      hash ^= static_cast<uint8_t>(c);
      hash *= 16777619u;
    }
    return hash == 0 ? 1 : hash;
  }

  // A chunk message's wParam and lParam: the name's kUe3MapChunkSize characters from offset, four to a parameter
  // with the first in the lowest byte, zero padded past the end of the name.
  inline std::pair<uint32_t, uint32_t> packUe3MapChunk(const std::string_view name, const size_t offset) {
    uint32_t words[2] = {};
    for (size_t i = 0; i < kUe3MapChunkSize && offset + i < name.size(); ++i) {
      words[i / 4] |= static_cast<uint32_t>(static_cast<uint8_t>(name[offset + i])) << (8 * (i % 4));
    }
    return { words[0], words[1] };
  }

  inline void appendUe3MapChunk(std::string& name, const uint32_t first, const uint32_t second) {
    for (size_t i = 0; i < kUe3MapChunkSize; ++i) {
      name.push_back(static_cast<char>(((i < 4 ? first : second) >> (8 * (i % 4))) & 0xFFu));
    }
  }
}
