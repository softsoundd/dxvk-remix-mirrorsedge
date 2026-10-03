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
#include "game_patches.h"

#include "log/log.h"

#include "../../../src/util/util_game_patches.h"

#include <windows.h>
#include <intrin.h>

#include <algorithm>
#include <cstring>
#include <vector>

using namespace bridge_util;

namespace {
  struct CodeSpan {
    const uint8_t* begin;
    size_t size;
  };

  struct BytePattern {
    std::vector<uint8_t> bytes;
    std::vector<uint8_t> mask;
  };

  // Bytes swapped between their original and patched values by one interlocked store, so a thread
  // executing the instruction they belong to never fetches a torn one.
  struct PatchWrite {
    uint8_t* address = nullptr;
    size_t size = 0;
    uint8_t original[2] = {};
    uint8_t patched[2] = {};
  };

  struct GamePatch {
    const char* name;
    uint32_t bit;
    std::vector<PatchWrite> writes;
    bool applied = false;
  };

  // FSceneRenderer::InitViews: View.ViewFrustum.IntersectSphere(Bounds.Origin, Bounds.SphereRadius),
  // then the jz that skips the primitive when it is outside the frustum.
  constexpr const char* kFrustumTestSignature =
    "F3 0F 10 41 28 83 C1 10 51 8D 8F E0 01 00 00 C7 44 24 ?? 00 00 00 00 0F C6 C0 00 "
    "E8 ?? ?? ?? ?? 85 C0 ?? ?? ?? ?? ?? ??";
  constexpr size_t kFrustumJumpOffset = 34;
  constexpr uint8_t kJumpIfZero[] = { 0x0F, 0x84 };
  // jmp short over the jz's rel32, onto the next instruction
  constexpr uint8_t kJumpOverRel32[] = { 0xEB, 0x04 };

  // The draw loops' test of Component->bOwnerNoSeeWithShadow. With a zero mask the test fails, and the
  // loop draws the primitive without looking for the viewer among its proxy's Owners.
  constexpr const char* kOwnerNoSeeWithShadowTestSignature = "F6 ?? FC 00 00 00 10";
  constexpr size_t kOwnerNoSeeWithShadowTestLength = 7;
  constexpr size_t kOwnerNoSeeWithShadowMaskOffset = 6;
  constexpr uint8_t kOwnerNoSeeWithShadowMask[] = { 0x10 };
  constexpr uint8_t kNoMask[] = { 0x00 };
  // add eax, 0x74, reaching the proxy's Owners array
  constexpr uint8_t kOwnersAccess[] = { 0x83, 0xC0, 0x74 };
  constexpr size_t kOwnersAccessWindow = 0x20;

  constexpr size_t kMaxMatches = 8;
  constexpr size_t kMaxOwnerNoSeeWithShadowTests = 32;

  GamePatch g_frustumCulling { "Disable frustum culling", dxvk::kGamePatchDisableFrustumCulling };
  GamePatch g_thirdPersonModel { "Show third-person model", dxvk::kGamePatchShowThirdPersonModel };
  GamePatch* const g_patches[] = { &g_frustumCulling, &g_thirdPersonModel };

  MessageChannelClient* g_pMsgChannel = nullptr;
  bool g_patchesLocated = false;

  uint8_t parseHexDigit(const char c) {
    return static_cast<uint8_t>(c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10);
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

  // Raw arguments only: __try cannot share a frame with objects that need unwinding.
  // Returns the match count, storing at most maxMatches of them, or SIZE_MAX if the span faulted.
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

  bool isReadableCode(const DWORD protect) {
    return (protect & PAGE_GUARD) == 0 &&
           (protect & (PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)) != 0;
  }

  // Adjacent regions are merged so a pattern straddling a protection change another mod made is still found.
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
                                         const size_t maxMatches = kMaxMatches) {
    std::vector<const uint8_t*> found;
    std::vector<const uint8_t*> matches(maxMatches);
    for (const CodeSpan& span : spans) {
      const size_t count = scanSpan(span.begin, span.size, pattern.bytes.data(), pattern.mask.data(),
                                    pattern.bytes.size(), matches.data(), maxMatches);
      if (count == SIZE_MAX) {
        Logger::warn(format_string("[GamePatch] Skipped unreadable code at 0x%p.", static_cast<const void*>(span.begin)));
        continue;
      }
      for (size_t i = 0; i < std::min(count, maxMatches) && found.size() < maxMatches; ++i) {
        found.push_back(matches[i]);
      }
    }
    return found;
  }

  void locateFrustumCulling(const std::vector<CodeSpan>& spans, GamePatch& patch) {
    const std::vector<const uint8_t*> matches = findInCode(spans, parsePattern(kFrustumTestSignature));
    if (matches.size() != 1) {
      Logger::warn(format_string("[GamePatch] %s: view frustum test not found (%zu matches).", patch.name, matches.size()));
      return;
    }

    uint8_t* const pJump = const_cast<uint8_t*>(matches[0]) + kFrustumJumpOffset;
    if (memcmp(pJump, kJumpIfZero, sizeof(kJumpIfZero)) != 0 || reinterpret_cast<uintptr_t>(pJump) % 2 != 0) {
      Logger::warn(format_string("[GamePatch] %s: unexpected instruction at 0x%p.", patch.name, static_cast<const void*>(pJump)));
      return;
    }

    PatchWrite write;
    write.address = pJump;
    write.size = sizeof(kJumpIfZero);
    memcpy(write.original, kJumpIfZero, sizeof(kJumpIfZero));
    memcpy(write.patched, kJumpOverRel32, sizeof(kJumpOverRel32));
    patch.writes.push_back(write);
    Logger::info(format_string("[GamePatch] %s: view frustum test at 0x%p.", patch.name, static_cast<const void*>(pJump)));
  }

  // The test must follow the mov r32, [r32+0x0C] that loads the scene info's Component into its base
  // register, and precede the loop over the proxy's Owners.
  bool isOwnerNoSeeWithShadowTest(const std::vector<CodeSpan>& spans, const uint8_t* pTest) {
    const uint8_t testModRm = pTest[1];
    if ((testModRm & 0xF8) != 0x80 || (testModRm & 7) == 4) {
      return false;
    }

    const CodeSpan* pSpan = findSpan(spans, pTest);
    if (pTest - pSpan->begin < 3) {
      return false;
    }
    const uint8_t* pLoad = pTest - 3;
    const uint8_t loadModRm = pLoad[1];
    if (pLoad[0] != 0x8B || (loadModRm & 0xC0) != 0x40 || (loadModRm & 7) == 4 || pLoad[2] != 0x0C ||
        ((loadModRm >> 3) & 7) != (testModRm & 7)) {
      return false;
    }

    const uint8_t* pAfter = pTest + kOwnerNoSeeWithShadowTestLength;
    const uint8_t* pEnd = std::min(pAfter + kOwnersAccessWindow, pSpan->begin + pSpan->size);
    return std::search(pAfter, pEnd, kOwnersAccess, kOwnersAccess + sizeof(kOwnersAccess)) != pEnd;
  }

  void locateThirdPersonModel(const std::vector<CodeSpan>& spans, GamePatch& patch) {
    const std::vector<const uint8_t*> matches =
      findInCode(spans, parsePattern(kOwnerNoSeeWithShadowTestSignature), kMaxOwnerNoSeeWithShadowTests);
    for (const uint8_t* pTest : matches) {
      if (!isOwnerNoSeeWithShadowTest(spans, pTest)) {
        continue;
      }
      PatchWrite write;
      write.address = const_cast<uint8_t*>(pTest) + kOwnerNoSeeWithShadowMaskOffset;
      write.size = sizeof(kOwnerNoSeeWithShadowMask);
      memcpy(write.original, kOwnerNoSeeWithShadowMask, sizeof(kOwnerNoSeeWithShadowMask));
      memcpy(write.patched, kNoMask, sizeof(kNoMask));
      patch.writes.push_back(write);
    }

    if (patch.writes.empty()) {
      Logger::warn(format_string("[GamePatch] %s: owner-see tests not found in the draw loops (%zu candidates).",
                                 patch.name, matches.size()));
      return;
    }
    Logger::info(format_string("[GamePatch] %s: %zu owner-see tests in the draw loops, the first at 0x%p.",
                               patch.name, patch.writes.size(), static_cast<const void*>(patch.writes[0].address)));
  }

  bool storeBytes(uint8_t* pAddress, const uint8_t* pBytes, const size_t size) {
    DWORD oldProtect = 0;
    if (!VirtualProtect(pAddress, size, PAGE_EXECUTE_READWRITE, &oldProtect)) {
      return false;
    }

    if (size == sizeof(short)) {
      short value;
      memcpy(&value, pBytes, sizeof(value));
      _InterlockedExchange16(reinterpret_cast<short volatile*>(pAddress), value);
    } else {
      _InterlockedExchange8(reinterpret_cast<char volatile*>(pAddress), static_cast<char>(pBytes[0]));
    }

    DWORD unusedProtect = 0;
    VirtualProtect(pAddress, size, oldProtect, &unusedProtect);
    FlushInstructionCache(GetCurrentProcess(), pAddress, size);
    return true;
  }

  bool setPatch(const GamePatch& patch, const bool enable) {
    for (const PatchWrite& write : patch.writes) {
      if (memcmp(write.address, enable ? write.original : write.patched, write.size) != 0) {
        Logger::warn(format_string("[GamePatch] %s: bytes at 0x%p were changed by something else, leaving them alone.",
                                   patch.name, static_cast<const void*>(write.address)));
        return false;
      }
    }

    for (size_t i = 0; i < patch.writes.size(); ++i) {
      const PatchWrite& write = patch.writes[i];
      if (!storeBytes(write.address, enable ? write.patched : write.original, write.size)) {
        Logger::err(format_string("[GamePatch] %s: VirtualProtect failed (%lu).", patch.name, GetLastError()));
        while (i-- > 0) {
          const PatchWrite& done = patch.writes[i];
          storeBytes(done.address, enable ? done.original : done.patched, done.size);
        }
        return false;
      }
    }
    return true;
  }

  void sendStatus() {
    uint32_t active = 0;
    uint32_t notFound = 0;
    for (const GamePatch* pPatch : g_patches) {
      if (pPatch->applied) {
        active |= pPatch->bit;
      } else if (pPatch->writes.empty()) {
        notFound |= pPatch->bit;
      }
    }
    // Fails until the channel handshake completes; the runtime repeats its request until answered.
    g_pMsgChannel->send(dxvk::kGamePatchStatusMsgName, active, notFound);
  }

  bool onRequest(const uint32_t requested) {
    if (!g_patchesLocated) {
      g_patchesLocated = true;
      const std::vector<CodeSpan> spans = collectExecutableCode();
      locateFrustumCulling(spans, g_frustumCulling);
      locateThirdPersonModel(spans, g_thirdPersonModel);
    }

    for (GamePatch* pPatch : g_patches) {
      const bool enable = (requested & pPatch->bit) != 0;
      if (pPatch->writes.empty() || enable == pPatch->applied) {
        continue;
      }
      if (setPatch(*pPatch, enable)) {
        pPatch->applied = enable;
        Logger::info(format_string("[GamePatch] %s: %s.", pPatch->name, enable ? "applied" : "reverted"));
      }
    }

    sendStatus();
    return true;
  }
}

void GamePatches::init(MessageChannelClient& msgChannel) {
  g_pMsgChannel = &msgChannel;
  msgChannel.registerHandler(dxvk::kGamePatchRequestMsgName, [](uint32_t wParam, uint32_t) {
    return onRequest(wParam);
  });
}
