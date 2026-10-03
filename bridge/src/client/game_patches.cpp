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
#include <initializer_list>
#include <iterator>
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

  // A branch found by a signature that must match exactly once, swapped between original and patched.
  struct BranchSite {
    GamePatch* patch;
    const char* description;
    const char* signature;
    size_t offset;
    const uint8_t* original;
    const uint8_t* patched;
    size_t size;
  };

  // The jz that sends a primitive failing FSceneRenderer::InitViews' frustum test to the cull path. All turns it
  // into a short jump over its own rel32, so every primitive carries on as visible. Limited points the rel32 at a
  // stub that lets through only the primitives within the request's limits.
  enum class FrustumBypassMode { Stock, All, Limited };

  struct FrustumBypass {
    uint8_t* jump = nullptr;
    int32_t cullDisplacement = 0;
    const uint8_t* stub = nullptr;
    int32_t stubDisplacement = 0;
    // What the jz's rel32 holds now: cullDisplacement or stubDisplacement.
    int32_t displacement = 0;
    FrustumBypassMode mode = FrustumBypassMode::Stock;
  };

  // FSceneRenderer::InitViews: View.ViewFrustum.IntersectSphere(Bounds.Origin, Bounds.SphereRadius),
  // then the jz that skips the primitive when it is outside the frustum.
  constexpr const char* kFrustumTestSignature =
    "F3 0F 10 41 28 83 C1 10 51 8D 8F E0 01 00 00 C7 44 24 ?? 00 00 00 00 0F C6 C0 00 "
    "E8 ?? ?? ?? ?? 85 C0 ?? ?? ?? ?? ?? ??";
  constexpr size_t kFrustumJumpOffset = 34;
  // The signature's movss xmm0, [ecx+SphereRadius], from the primitive's FPrimitiveSceneInfoCompact
  constexpr size_t kSphereRadiusOffsetInSignature = 4;
  constexpr uint8_t kJumpIfZero[] = { 0x0F, 0x84 };
  constexpr size_t kJumpIfZeroLength = 6;
  // jmp short over the jz's rel32, onto the next instruction
  constexpr uint8_t kJumpOverRel32[] = { 0xEB, 0x04 };

  // InitViews keeps DistanceSquared on the stack across the frustum test: it is spilled just before
  // (comiss xmm0, xmm2; movss [esp+XX], xmm0; ja) and reloaded after for the static meshes' draw distances.
  // The visible path starts by reloading the primitive's FPrimitiveSceneInfoCompact (mov esi, [esp+YY]).
  constexpr const char* kDistanceSquaredSpillSignature = "0F 2F C2 F3 0F 11 44 24 ?? 0F 87";
  constexpr size_t kDistanceSquaredSpillSlotOffset = 8;
  constexpr size_t kDistanceSquaredSpillWindow = 0x100;
  constexpr uint8_t kDistanceSquaredReload[] = { 0xF3, 0x0F, 0x10, 0x44, 0x24 };
  constexpr size_t kDistanceSquaredReloadWindow = 0x140;
  constexpr uint8_t kPrimitiveReload[] = { 0x8B, 0x74, 0x24 };
  constexpr size_t kFrustumBypassStubSize = 64;
  constexpr long kInfinityBits = 0x7F800000;

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

  // FSceneRenderer::InitViews, both variants: if (GIsTiledScreenshot || GIgnoreAllOcclusionQueries || bIsHitTesting),
  // set View.bDisableQuerySubmissions and View.bIgnoreExistingQueries.
  constexpr const char* kOcclusionQueryOptOutSignature =
    "39 ?? ?? ?? ?? ?? 75 0C 39 ?? ?? ?? ?? ?? 75 04 3B ?? 74 ?? ?? 01 00 00 00 89 ?? 78 06 00 00 89 ?? 74 06 00 00";
  // Its cmp dword ptr [GIgnoreAllOcclusionQueries], r32
  constexpr size_t kOcclusionFlagTestOffset = 8;

  // RenderViewFamily_RenderThread: if (ShowFlags & SHOW_SceneCaptureUpdates) RenderSceneCaptures(), then Render().
  constexpr const char* kSceneCaptureTestSignature = "81 E2 00 01 00 00 33 C0 0B D0 74 07 8B CE E8 ?? ?? ?? ?? 8B CE E8";
  constexpr size_t kSceneCaptureJumpOffset = 10;
  // The end of FSceneRenderer::InitViews: if ((ShowFlags & SHOW_DynamicShadows) && bAllowDynamicShadows)
  // InitDynamicShadows(), then InitFogConstants(). The jump is the bAllowDynamicShadows test's.
  constexpr const char* kShadowSetupTestSignature =
    "8B 46 18 83 E0 20 33 C9 0B C1 74 0F 39 0D ?? ?? ?? ?? 74 07 8B CE E8 ?? ?? ?? ?? 8B CE E8";
  constexpr size_t kShadowSetupJumpOffset = 18;
  // FSceneRenderer::Render: the same two tests before RenderModulatedShadows.
  constexpr const char* kModulatedShadowTestSignature =
    "83 E1 20 0B CB 74 ?? 39 1D ?? ?? ?? ?? 74 ?? 39 1D ?? ?? ?? ?? 74 05 83 FF 01 75 ?? 57 8B CD E8";
  constexpr size_t kModulatedShadowJumpOffset = 13;
  // FSceneRenderer::Render: if (ShowFlags & SHOW_Lighting) { RenderLights, RenderModulatedShadows and the
  // lighting-only post process effects }.
  constexpr const char* kLightingTestSignature = "8B 45 18 25 00 10 00 00 0B C3 0F 84 ?? ?? ?? ?? 8B 44 24 ?? 50 57 8B CD E8";
  constexpr size_t kLightingJumpOffset = 10;
  // FSceneRenderer::Render: if (bWorldDpg && bAllowMotionBlur) RenderVelocities().
  constexpr const char* kVelocityTestSignature = "3B F3 74 2E 39 1D ?? ?? ?? ?? 74 26 57 8B CD E8";
  constexpr size_t kVelocityJumpOffset = 10;

  constexpr uint8_t kJumpIfZeroShort[] = { 0x74 };
  constexpr uint8_t kJumpIfNotZeroShort[] = { 0x75 };
  constexpr uint8_t kJumpShort[] = { 0xEB };
  // nop, then a jmp reusing the jz's rel32: it ends where the jz did, so it reaches the same target
  constexpr uint8_t kNopThenJump[] = { 0x90, 0xE9 };

  constexpr size_t kMaxMatches = 8;
  constexpr size_t kMaxOwnerNoSeeWithShadowTests = 32;

  constexpr const char* kFrustumCullingName = "Disable frustum culling";
  FrustumBypass g_frustumBypass;
  // Read by the limit stub as the bit patterns of non-negative floats, which order like unsigned integers.
  volatile long g_frustumBypassMinRadiusBits = kInfinityBits;
  volatile long g_frustumBypassMaxDistanceSquaredBits = 0;

  GamePatch g_thirdPersonModel { "Show third-person model", dxvk::kGamePatchShowThirdPersonModel };
  GamePatch g_occlusionQueries { "Disable occlusion queries", dxvk::kGamePatchDisableOcclusionQueries };
  GamePatch g_sceneCaptures { "Disable scene captures", dxvk::kGamePatchDisableSceneCaptures };
  GamePatch g_dynamicShadows { "Disable dynamic shadows", dxvk::kGamePatchDisableDynamicShadows };
  GamePatch g_dynamicLighting { "Disable dynamic lighting", dxvk::kGamePatchDisableDynamicLighting };
  GamePatch g_velocityPass { "Disable velocity pass", dxvk::kGamePatchDisableVelocityPass };
  GamePatch* const g_patches[] = {
    &g_thirdPersonModel, &g_occlusionQueries, &g_sceneCaptures, &g_dynamicShadows, &g_dynamicLighting, &g_velocityPass
  };

  const BranchSite g_branchSites[] = {
    { &g_sceneCaptures, "scene capture test", kSceneCaptureTestSignature, kSceneCaptureJumpOffset,
      kJumpIfZeroShort, kJumpShort, sizeof(kJumpShort) },
    { &g_dynamicShadows, "shadow setup test", kShadowSetupTestSignature, kShadowSetupJumpOffset,
      kJumpIfZeroShort, kJumpShort, sizeof(kJumpShort) },
    { &g_dynamicShadows, "modulated shadow test", kModulatedShadowTestSignature, kModulatedShadowJumpOffset,
      kJumpIfZeroShort, kJumpShort, sizeof(kJumpShort) },
    { &g_dynamicLighting, "lighting test", kLightingTestSignature, kLightingJumpOffset,
      kJumpIfZero, kNopThenJump, sizeof(kNopThenJump) },
    { &g_velocityPass, "velocity pass test", kVelocityTestSignature, kVelocityJumpOffset,
      kJumpIfZeroShort, kJumpShort, sizeof(kJumpShort) },
  };

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

  void addWrite(GamePatch& patch, uint8_t* pAddress, const uint8_t* original, const uint8_t* patched, const size_t size) {
    PatchWrite write;
    write.address = pAddress;
    write.size = size;
    memcpy(write.original, original, size);
    memcpy(write.patched, patched, size);
    patch.writes.push_back(write);
  }

  // Up to two bytes go in one interlocked store. Longer writes are not atomic, so they are only made to bytes no
  // thread can reach.
  bool storeBytes(uint8_t* pAddress, const void* pBytes, const size_t size) {
    DWORD oldProtect = 0;
    if (!VirtualProtect(pAddress, size, PAGE_EXECUTE_READWRITE, &oldProtect)) {
      return false;
    }

    if (size == sizeof(short)) {
      short value;
      memcpy(&value, pBytes, sizeof(value));
      _InterlockedExchange16(reinterpret_cast<short volatile*>(pAddress), value);
    } else if (size == sizeof(char)) {
      _InterlockedExchange8(reinterpret_cast<char volatile*>(pAddress), *static_cast<const char*>(pBytes));
    } else {
      memcpy(pAddress, pBytes, size);
    }

    DWORD unusedProtect = 0;
    VirtualProtect(pAddress, size, oldProtect, &unusedProtect);
    FlushInstructionCache(GetCurrentProcess(), pAddress, size);
    return true;
  }

  bool findFrustumBypassSlots(const CodeSpan& span, const uint8_t* pSignature, const uint8_t* pFallThrough,
                              uint8_t& distanceSquaredSlot, uint8_t& primitiveSlot) {
    const uint8_t* const pSpanEnd = span.begin + span.size;
    const size_t spillWindow = std::min(kDistanceSquaredSpillWindow, static_cast<size_t>(pSignature - span.begin));
    const BytePattern spill = parsePattern(kDistanceSquaredSpillSignature);
    const uint8_t* spills[2] = {};
    if (scanSpan(pSignature - spillWindow, spillWindow, spill.bytes.data(), spill.mask.data(), spill.bytes.size(),
                 spills, 2) != 1) {
      return false;
    }
    distanceSquaredSlot = spills[0][kDistanceSquaredSpillSlotOffset];

    if (static_cast<size_t>(pSpanEnd - pFallThrough) <= sizeof(kPrimitiveReload) ||
        memcmp(pFallThrough, kPrimitiveReload, sizeof(kPrimitiveReload)) != 0) {
      return false;
    }
    primitiveSlot = pFallThrough[sizeof(kPrimitiveReload)];

    BytePattern reload;
    reload.bytes.assign(std::begin(kDistanceSquaredReload), std::end(kDistanceSquaredReload));
    reload.bytes.push_back(distanceSquaredSlot);
    reload.mask.assign(reload.bytes.size(), 0xFF);
    const size_t reloadWindow = std::min(kDistanceSquaredReloadWindow, static_cast<size_t>(pSpanEnd - pFallThrough));
    const uint8_t* reloads[1] = {};
    const size_t reloadCount = scanSpan(pFallThrough, reloadWindow, reload.bytes.data(), reload.mask.data(),
                                        reload.bytes.size(), reloads, 1);
    return reloadCount != 0 && reloadCount != SIZE_MAX;
  }

  // Where the frustum test's jz goes for a primitive outside the view frustum: on to the visible path if the
  // primitive is large or near enough, otherwise to the cull path. Only eax and the flags change, which both paths
  // overwrite before reading; InitViews keeps values in SSE registers across the test. Never freed, as a thread can
  // still be running it after the jz stops pointing at it.
  const uint8_t* buildFrustumBypassStub(const uint8_t* pFallThrough, const uint8_t* pCull, const uint8_t primitiveSlot,
                                        const uint8_t sphereRadiusOffset, const uint8_t distanceSquaredSlot) {
    static_assert(sizeof(void*) == 4, "The limit stub is 32-bit code that addresses its thresholds absolutely.");
    uint8_t* const pStub =
      static_cast<uint8_t*>(VirtualAlloc(nullptr, kFrustumBypassStubSize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (pStub == nullptr) {
      return nullptr;
    }

    uint8_t* p = pStub;
    const auto emit = [&p](const std::initializer_list<uint8_t> bytes) {
      for (const uint8_t byte : bytes) {
        *p++ = byte;
      }
    };
    const auto emitAddress = [&p](const volatile long* pValue) {
      const uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pValue));
      memcpy(p, &address, sizeof(address));
      p += sizeof(address);
    };
    const auto emitJump = [&p](const uint8_t* pTarget) {
      *p++ = 0xE9;
      const auto displacement =
        static_cast<int32_t>(reinterpret_cast<uintptr_t>(pTarget) - reinterpret_cast<uintptr_t>(p + sizeof(int32_t)));
      memcpy(p, &displacement, sizeof(displacement));
      p += sizeof(displacement);
    };

    emit({ 0x8B, 0x44, 0x24, primitiveSlot });          // mov eax, [esp+primitiveSlot]
    emit({ 0x8B, 0x40, sphereRadiusOffset });           // mov eax, [eax+SphereRadius]
    emit({ 0x3B, 0x05 });                               // cmp eax, [g_frustumBypassMinRadiusBits]
    emitAddress(&g_frustumBypassMinRadiusBits);
    emit({ 0x73, 0x00 });                               // jae keep
    uint8_t* const pKeepIfLarge = p - 1;
    emit({ 0x8B, 0x44, 0x24, distanceSquaredSlot });    // mov eax, [esp+distanceSquaredSlot]
    emit({ 0x3B, 0x05 });                               // cmp eax, [g_frustumBypassMaxDistanceSquaredBits]
    emitAddress(&g_frustumBypassMaxDistanceSquaredBits);
    emit({ 0x76, 0x00 });                               // jbe keep
    uint8_t* const pKeepIfNear = p - 1;
    emitJump(pCull);
    *pKeepIfLarge = static_cast<uint8_t>(p - (pKeepIfLarge + 1));
    *pKeepIfNear = static_cast<uint8_t>(p - (pKeepIfNear + 1));
    emitJump(pFallThrough);                             // keep:

    DWORD oldProtect = 0;
    if (!VirtualProtect(pStub, kFrustumBypassStubSize, PAGE_EXECUTE_READ, &oldProtect)) {
      VirtualFree(pStub, 0, MEM_RELEASE);
      return nullptr;
    }
    FlushInstructionCache(GetCurrentProcess(), pStub, kFrustumBypassStubSize);
    return pStub;
  }

  void locateFrustumCulling(const std::vector<CodeSpan>& spans, FrustumBypass& bypass) {
    const std::vector<const uint8_t*> matches = findInCode(spans, parsePattern(kFrustumTestSignature));
    if (matches.size() != 1) {
      Logger::warn(format_string("[GamePatch] %s: view frustum test not found (%zu matches).", kFrustumCullingName, matches.size()));
      return;
    }

    uint8_t* const pJump = const_cast<uint8_t*>(matches[0]) + kFrustumJumpOffset;
    if (memcmp(pJump, kJumpIfZero, sizeof(kJumpIfZero)) != 0 || reinterpret_cast<uintptr_t>(pJump) % 2 != 0) {
      Logger::warn(format_string("[GamePatch] %s: unexpected instruction at 0x%p.", kFrustumCullingName, static_cast<const void*>(pJump)));
      return;
    }
    bypass.jump = pJump;
    memcpy(&bypass.cullDisplacement, pJump + sizeof(kJumpIfZero), sizeof(bypass.cullDisplacement));
    bypass.displacement = bypass.cullDisplacement;
    Logger::info(format_string("[GamePatch] %s: view frustum test at 0x%p.", kFrustumCullingName, static_cast<const void*>(pJump)));

    const uint8_t* const pFallThrough = pJump + kJumpIfZeroLength;
    uint8_t distanceSquaredSlot = 0;
    uint8_t primitiveSlot = 0;
    if (!findFrustumBypassSlots(*findSpan(spans, matches[0]), matches[0], pFallThrough, distanceSquaredSlot, primitiveSlot)) {
      Logger::warn(format_string("[GamePatch] %s: InitViews' stack slots not found, so the bypass cannot be limited.", kFrustumCullingName));
      return;
    }

    bypass.stub = buildFrustumBypassStub(pFallThrough, pFallThrough + bypass.cullDisplacement, primitiveSlot,
                                         matches[0][kSphereRadiusOffsetInSignature], distanceSquaredSlot);
    if (bypass.stub == nullptr) {
      Logger::err(format_string("[GamePatch] %s: creating the limit stub failed (%lu).", kFrustumCullingName, GetLastError()));
      return;
    }
    bypass.stubDisplacement =
      static_cast<int32_t>(reinterpret_cast<uintptr_t>(bypass.stub) - reinterpret_cast<uintptr_t>(pFallThrough));
    Logger::info(format_string("[GamePatch] %s: limit stub at 0x%p, reading DistanceSquared at [esp+0x%02X] and the primitive at [esp+0x%02X].",
                               kFrustumCullingName, static_cast<const void*>(bypass.stub),
                               static_cast<unsigned>(distanceSquaredSlot), static_cast<unsigned>(primitiveSlot)));
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
      if (isOwnerNoSeeWithShadowTest(spans, pTest)) {
        addWrite(patch, const_cast<uint8_t*>(pTest) + kOwnerNoSeeWithShadowMaskOffset, kOwnerNoSeeWithShadowMask, kNoMask,
                 sizeof(kNoMask));
      }
    }

    if (patch.writes.empty()) {
      Logger::warn(format_string("[GamePatch] %s: owner-see tests not found in the draw loops (%zu candidates).",
                                 patch.name, matches.size()));
      return;
    }
    Logger::info(format_string("[GamePatch] %s: %zu owner-see tests in the draw loops, the first at 0x%p.",
                               patch.name, patch.writes.size(), static_cast<const void*>(patch.writes[0].address)));
  }

  // cmp dword ptr [address], r32
  bool isAbsoluteCompare(const uint8_t* pInstruction) {
    return pInstruction[0] == 0x39 && (pInstruction[1] & 0xC7) == 0x05;
  }

  // GIgnoreAllOcclusionQueries is read from InitViews' own tests of it and confirmed as the flag the TOGGLEOCCLUSION
  // console command flips; each branch taken when it is set is made unconditional.
  void locateOcclusionQueries(const std::vector<CodeSpan>& spans, GamePatch& patch) {
    const std::vector<const uint8_t*> optOuts = findInCode(spans, parsePattern(kOcclusionQueryOptOutSignature));
    uint32_t flag = 0;
    for (const uint8_t* pOptOut : optOuts) {
      const uint8_t* const pTest = pOptOut + kOcclusionFlagTestOffset;
      uint32_t address;
      memcpy(&address, pTest + 2, sizeof(address));
      if (!isAbsoluteCompare(pTest) || (flag != 0 && address != flag)) {
        Logger::warn(format_string("[GamePatch] %s: InitViews' occlusion query tests disagree.", patch.name));
        return;
      }
      flag = address;
    }
    if (flag == 0) {
      Logger::warn(format_string("[GamePatch] %s: InitViews' occlusion query test not found.", patch.name));
      return;
    }

    uint8_t flagBytes[sizeof(flag)];
    memcpy(flagBytes, &flag, sizeof(flag));

    // xor eax, eax; cmp [flag], eax; sete al; mov [flag], eax
    BytePattern toggle;
    toggle.bytes = { 0x33, 0xC0, 0x39, 0x05, flagBytes[0], flagBytes[1], flagBytes[2], flagBytes[3],
                     0x0F, 0x94, 0xC0, 0xA3, flagBytes[0], flagBytes[1], flagBytes[2], flagBytes[3] };
    toggle.mask.assign(toggle.bytes.size(), 0xFF);
    if (findInCode(spans, toggle).size() != 1) {
      Logger::warn(format_string("[GamePatch] %s: 0x%08X is not the flag TOGGLEOCCLUSION toggles.", patch.name, flag));
      return;
    }

    // cmp dword ptr [flag], r32; jne short
    BytePattern test;
    test.bytes = { 0x39, 0x05, flagBytes[0], flagBytes[1], flagBytes[2], flagBytes[3], kJumpIfNotZeroShort[0] };
    test.mask = { 0xFF, 0xC7, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
    for (const uint8_t* pTest : findInCode(spans, test)) {
      addWrite(patch, const_cast<uint8_t*>(pTest) + test.bytes.size() - 1, kJumpIfNotZeroShort, kJumpShort, sizeof(kJumpShort));
    }

    if (patch.writes.empty()) {
      Logger::warn(format_string("[GamePatch] %s: no branch on GIgnoreAllOcclusionQueries (0x%08X) found.", patch.name, flag));
      return;
    }
    Logger::info(format_string("[GamePatch] %s: %zu branches on GIgnoreAllOcclusionQueries (0x%08X), the first at 0x%p.",
                               patch.name, patch.writes.size(), flag, static_cast<const void*>(patch.writes[0].address)));
  }

  bool locateBranch(const std::vector<CodeSpan>& spans, const BranchSite& site) {
    GamePatch& patch = *site.patch;
    const std::vector<const uint8_t*> matches = findInCode(spans, parsePattern(site.signature));
    if (matches.size() != 1) {
      Logger::warn(format_string("[GamePatch] %s: %s not found (%zu matches).", patch.name, site.description, matches.size()));
      return false;
    }

    uint8_t* const pBranch = const_cast<uint8_t*>(matches[0]) + site.offset;
    if (memcmp(pBranch, site.original, site.size) != 0 || reinterpret_cast<uintptr_t>(pBranch) % site.size != 0) {
      Logger::warn(format_string("[GamePatch] %s: unexpected instruction at 0x%p.", patch.name, static_cast<const void*>(pBranch)));
      return false;
    }
    addWrite(patch, pBranch, site.original, site.patched, site.size);
    Logger::info(format_string("[GamePatch] %s: %s at 0x%p.", patch.name, site.description, static_cast<const void*>(pBranch)));
    return true;
  }

  // A patch is only usable with every one of its sites.
  void locateBranches(const std::vector<CodeSpan>& spans) {
    std::vector<GamePatch*> incomplete;
    for (const BranchSite& site : g_branchSites) {
      if (!locateBranch(spans, site)) {
        incomplete.push_back(site.patch);
      }
    }
    for (GamePatch* pPatch : incomplete) {
      pPatch->writes.clear();
    }
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

  bool isFrustumJumpIntact(const FrustumBypass& bypass) {
    uint8_t expected[kJumpIfZeroLength];
    memcpy(expected, bypass.mode == FrustumBypassMode::All ? kJumpOverRel32 : kJumpIfZero, sizeof(kJumpIfZero));
    memcpy(expected + sizeof(kJumpIfZero), &bypass.displacement, sizeof(bypass.displacement));
    return memcmp(bypass.jump, expected, sizeof(expected)) == 0;
  }

  // Every change goes through All: the rel32 is only rewritten while the short jump keeps threads off it.
  bool setFrustumBypassMode(FrustumBypass& bypass, const FrustumBypassMode mode) {
    if (!isFrustumJumpIntact(bypass)) {
      Logger::warn(format_string("[GamePatch] %s: bytes at 0x%p were changed by something else, leaving them alone.",
                                 kFrustumCullingName, static_cast<const void*>(bypass.jump)));
      return false;
    }
    const auto store = [](uint8_t* pAddress, const void* pBytes, const size_t size) {
      if (storeBytes(pAddress, pBytes, size)) {
        return true;
      }
      Logger::err(format_string("[GamePatch] %s: VirtualProtect failed (%lu).", kFrustumCullingName, GetLastError()));
      return false;
    };

    if (bypass.mode != FrustumBypassMode::All) {
      if (!store(bypass.jump, kJumpOverRel32, sizeof(kJumpOverRel32))) {
        return false;
      }
      bypass.mode = FrustumBypassMode::All;
    }
    if (mode == FrustumBypassMode::All) {
      return true;
    }

    const int32_t displacement = mode == FrustumBypassMode::Limited ? bypass.stubDisplacement : bypass.cullDisplacement;
    if (displacement != bypass.displacement) {
      // Interrupting every processor before the rel32 changes ensures none is still fetching the jz the short jump
      // replaced, and after, that the new rel32 is visible before the jz is restored.
      FlushProcessWriteBuffers();
      if (!store(bypass.jump + sizeof(kJumpIfZero), &displacement, sizeof(displacement))) {
        return false;
      }
      bypass.displacement = displacement;
      FlushProcessWriteBuffers();
    }
    if (!store(bypass.jump, kJumpIfZero, sizeof(kJumpIfZero))) {
      return false;
    }
    bypass.mode = mode;
    return true;
  }

  long floatBits(const float value) {
    long bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
  }

  void applyFrustumBypass(const uint32_t requested, const uint32_t limits) {
    FrustumBypass& bypass = g_frustumBypass;
    if (bypass.jump == nullptr) {
      return;
    }

    FrustumBypassMode mode = FrustumBypassMode::Stock;
    if ((requested & dxvk::kGamePatchDisableFrustumCulling) != 0) {
      const bool limited = (requested & dxvk::kGamePatchLimitFrustumBypass) != 0 && bypass.stub != nullptr;
      mode = limited ? FrustumBypassMode::Limited : FrustumBypassMode::All;
    }

    const float maxDistance = dxvk::unpackFrustumBypassMaxDistance(limits);
    const float minRadius = dxvk::unpackFrustumBypassMinRadius(limits);
    if (mode == FrustumBypassMode::Limited) {
      _InterlockedExchange(&g_frustumBypassMaxDistanceSquaredBits, floatBits(maxDistance * maxDistance));
      _InterlockedExchange(&g_frustumBypassMinRadiusBits, minRadius > 0.0f ? floatBits(minRadius) : kInfinityBits);
    }

    if (mode == bypass.mode || !setFrustumBypassMode(bypass, mode)) {
      return;
    }
    if (mode == FrustumBypassMode::Stock) {
      Logger::info(format_string("[GamePatch] %s: reverted.", kFrustumCullingName));
    } else if (mode == FrustumBypassMode::All) {
      Logger::info(format_string("[GamePatch] %s: applied.", kFrustumCullingName));
    } else if (minRadius > 0.0f) {
      Logger::info(format_string("[GamePatch] %s: applied within %.0f units, or from a bounding radius of %.0f.",
                                 kFrustumCullingName, maxDistance, minRadius));
    } else {
      Logger::info(format_string("[GamePatch] %s: applied within %.0f units.", kFrustumCullingName, maxDistance));
    }
  }

  void sendStatus() {
    uint32_t active = 0;
    uint32_t notFound = 0;
    const FrustumBypass& bypass = g_frustumBypass;
    if (bypass.jump == nullptr) {
      notFound |= dxvk::kGamePatchDisableFrustumCulling | dxvk::kGamePatchLimitFrustumBypass;
    } else {
      if (bypass.mode != FrustumBypassMode::Stock) {
        active |= dxvk::kGamePatchDisableFrustumCulling;
      }
      if (bypass.mode == FrustumBypassMode::Limited) {
        active |= dxvk::kGamePatchLimitFrustumBypass;
      }
      if (bypass.stub == nullptr) {
        notFound |= dxvk::kGamePatchLimitFrustumBypass;
      }
    }

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

  bool onRequest(const uint32_t requested, const uint32_t limits) {
    if (!g_patchesLocated) {
      g_patchesLocated = true;
      const std::vector<CodeSpan> spans = collectExecutableCode();
      locateFrustumCulling(spans, g_frustumBypass);
      locateThirdPersonModel(spans, g_thirdPersonModel);
      locateOcclusionQueries(spans, g_occlusionQueries);
      locateBranches(spans);
    }

    applyFrustumBypass(requested, limits);
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
  msgChannel.registerHandler(dxvk::kGamePatchRequestMsgName, [](uint32_t wParam, uint32_t lParam) {
    return onRequest(wParam, lParam);
  });
}
