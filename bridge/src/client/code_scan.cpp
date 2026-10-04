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
#include "code_scan.h"

#include "log/log.h"

#include <windows.h>

#include <algorithm>

using namespace bridge_util;

namespace code_scan {
  namespace {
    uint8_t parseHexDigit(const char c) {
      return static_cast<uint8_t>(c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10);
    }

    bool isReadableCode(const DWORD protect) {
      return (protect & PAGE_GUARD) == 0 &&
             (protect & (PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)) != 0;
    }
  }

  BytePattern parsePattern(const char* text) {
    BytePattern pattern;
    for (const char* p = text; *p != '\0'; ) {
      if (*p == ' ') {
        ++p;
        continue;
      }
      if (p[0] == '?') {
        pattern.bytes.push_back(0);
        pattern.mask.push_back(0);
      } else {
        pattern.bytes.push_back(static_cast<uint8_t>((parseHexDigit(p[0]) << 4) | parseHexDigit(p[1])));
        pattern.mask.push_back(0xFF);
      }
      p += 2;
    }
    return pattern;
  }

  size_t scanSpan(const uint8_t* pBegin, const size_t size, const uint8_t* pBytes, const uint8_t* pMask,
                  const size_t length, const uint8_t** pMatches, const size_t maxMatches) {
    size_t count = 0;
    __try {
      for (size_t offset = 0; offset + length <= size; ++offset) {
        size_t i = 0;
        while (i < length && (pBegin[offset + i] & pMask[i]) == pBytes[i]) {
          ++i;
        }
        if (i == length) {
          if (count < maxMatches) {
            pMatches[count] = pBegin + offset;
          }
          ++count;
        }
      }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
      return SIZE_MAX;
    }
    return count;
  }

  std::vector<CodeSpan> collectExecutableCode() {
    std::vector<CodeSpan> spans;
    const auto pBase = reinterpret_cast<const uint8_t*>(GetModuleHandleW(nullptr));
    const uint8_t* pCursor = pBase;
    MEMORY_BASIC_INFORMATION info;
    while (VirtualQuery(pCursor, &info, sizeof(info)) == sizeof(info) && info.AllocationBase == pBase) {
      const auto pRegion = static_cast<const uint8_t*>(info.BaseAddress);
      if (info.State == MEM_COMMIT && isReadableCode(info.Protect)) {
        if (!spans.empty() && spans.back().begin + spans.back().size == pRegion) {
          spans.back().size += info.RegionSize;
        } else {
          spans.push_back({ pRegion, info.RegionSize });
        }
      }
      pCursor = pRegion + info.RegionSize;
    }
    return spans;
  }

  const CodeSpan* findSpan(const std::vector<CodeSpan>& spans, const uint8_t* pAddress) {
    for (const CodeSpan& span : spans) {
      if (pAddress >= span.begin && pAddress < span.begin + span.size) {
        return &span;
      }
    }
    return nullptr;
  }

  std::vector<const uint8_t*> findInCode(const std::vector<CodeSpan>& spans, const BytePattern& pattern,
                                         const size_t maxMatches) {
    std::vector<const uint8_t*> found;
    std::vector<const uint8_t*> matches(maxMatches);
    for (const CodeSpan& span : spans) {
      const size_t count = scanSpan(span.begin, span.size, pattern.bytes.data(), pattern.mask.data(),
                                    pattern.bytes.size(), matches.data(), maxMatches);
      if (count == SIZE_MAX) {
        Logger::warn(format_string("[CodeScan] Skipped unreadable code at 0x%p.", static_cast<const void*>(span.begin)));
        continue;
      }
      for (size_t i = 0; i < std::min(count, maxMatches) && found.size() < maxMatches; ++i) {
        found.push_back(matches[i]);
      }
    }
    return found;
  }
}
