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

#include <cstddef>
#include <cstdint>
#include <vector>

// Code signature search in the game executable.
namespace code_scan {
  struct CodeSpan {
    const uint8_t* begin;
    size_t size;
  };

  struct BytePattern {
    std::vector<uint8_t> bytes;
    std::vector<uint8_t> mask;
  };

  constexpr size_t kMaxMatches = 8;

  // Hex bytes separated by spaces, with ?? matching any byte.
  BytePattern parsePattern(const char* text);

  // Raw arguments only: __try cannot share a frame with objects that need unwinding.
  // Returns the match count, storing at most maxMatches of them, or SIZE_MAX if the span faulted.
  size_t scanSpan(const uint8_t* pBegin, size_t size, const uint8_t* pBytes, const uint8_t* pMask,
                  size_t length, const uint8_t** pMatches, size_t maxMatches);

  // The executable's committed, readable code. Adjacent regions are merged so a pattern straddling a protection
  // change another mod made is still found.
  std::vector<CodeSpan> collectExecutableCode();

  const CodeSpan* findSpan(const std::vector<CodeSpan>& spans, const uint8_t* pAddress);

  std::vector<const uint8_t*> findInCode(const std::vector<CodeSpan>& spans, const BytePattern& pattern,
                                         size_t maxMatches = kMaxMatches);
}
