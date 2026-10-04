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
#include "game_map.h"

#include "code_scan.h"
#include "log/log.h"

#include "../../../src/util/util_game_map.h"

#include <windows.h>

#include <cstring>
#include <cwchar>
#include <optional>
#include <string>
#include <vector>

using namespace bridge_util;
using namespace code_scan;

namespace {
  static_assert(sizeof(void*) == 4, "The engine layouts below are the 32-bit executable's.");

  // mov ecx, [GNames]; mov eax, [esp+disp32]; mov eax, [ecx+eax*4]
  constexpr const char* kNamesSignature = "8B 0D ?? ?? ?? ?? 8B 84 24 ?? ?? ?? ?? 8B 04 81";
  // mov edx, [GObjects]; mov ecx, [edx+esi*4]; lea eax, [esp+0x30]
  constexpr const char* kObjectsSignature = "8B 15 ?? ?? ?? ?? 8B 0C B2 8D 44 24 30";
  // The disp32 of each signature's first mov: the address of the array.
  constexpr size_t kArrayAddressOffset = 2;

  // TArray, and FString as one of characters whose Count includes the terminator.
  struct RawArray {
    uintptr_t data;
    int32_t count;
    int32_t max;
  };

  // FNameEntry: Index, Flags (a QWORD), HashNext, then the name's characters.
  constexpr uintptr_t kNameEntryTextOffset = 0x10;
  // UObject. ObjectFlags is a QWORD whose low half holds RF_ClassDefaultObject; Name is an FName, Index first.
  constexpr uintptr_t kObjectFlagsOffset = 0x08;
  constexpr uintptr_t kObjectNameOffset = 0x2C;
  constexpr uintptr_t kObjectClassOffset = 0x34;
  constexpr uint32_t kClassDefaultObjectFlag = 0x200;
  // UGameEngine::LastURL, an FURL: Protocol, Host (FStrings), Port, Map (an FString).
  constexpr uintptr_t kLastUrlOffset = 0x3B0;
  constexpr uintptr_t kUrlProtocolOffset = 0x00;
  constexpr uintptr_t kUrlMapOffset = 0x1C;

  constexpr const wchar_t* kEngineClassName = L"TdGameEngine";
  constexpr const wchar_t* kUrlProtocol = L"unreal";
  constexpr int32_t kProtocolCapacity = 16;
  constexpr int32_t kMapCapacity = 260;
  constexpr DWORD kEngineSearchIntervalMs = 2000;

  struct MapReading {
    dxvk::Ue3MapStatus status;
    std::string name;
  };

  MessageChannelClient* g_pMsgChannel = nullptr;
  bool g_arraysLocated = false;
  uintptr_t g_names = 0;
  uintptr_t g_objects = 0;
  int32_t g_engineClassName = -1;
  uintptr_t g_engine = 0;
  bool g_engineSearched = false;
  DWORD g_engineSearchTime = 0;
  std::optional<MapReading> g_loggedReading;

  // findNameIndex, isInstance, findInstance and readString take raw arguments only: __try cannot share a frame with
  // objects that need unwinding.

  // The index of the first name in the table equal to pName, or -1.
  int32_t findNameIndex(const uintptr_t names, const wchar_t* pName) {
    __try {
      const RawArray& table = *reinterpret_cast<const RawArray*>(names);
      const auto* pEntries = reinterpret_cast<const uintptr_t*>(table.data);
      for (int32_t i = 0; i < table.count; ++i) {
        if (pEntries[i] != 0 && wcscmp(reinterpret_cast<const wchar_t*>(pEntries[i] + kNameEntryTextOffset), pName) == 0) {
          return i;
        }
      }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
    return -1;
  }

  bool isInstance(const uintptr_t object, const int32_t className) {
    __try {
      const uintptr_t objectClass = *reinterpret_cast<const uintptr_t*>(object + kObjectClassOffset);
      return objectClass != 0 &&
             *reinterpret_cast<const int32_t*>(objectClass + kObjectNameOffset) == className &&
             (*reinterpret_cast<const uint32_t*>(object + kObjectFlagsOffset) & kClassDefaultObjectFlag) == 0;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
      return false;
    }
  }

  // The first object in the table of the class named className, other than the class's default object, or 0.
  uintptr_t findInstance(const uintptr_t objects, const int32_t className) {
    __try {
      const RawArray& table = *reinterpret_cast<const RawArray*>(objects);
      const auto* pObjects = reinterpret_cast<const uintptr_t*>(table.data);
      for (int32_t i = 0; i < table.count; ++i) {
        if (pObjects[i] != 0 && isInstance(pObjects[i], className)) {
          return pObjects[i];
        }
      }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
    return 0;
  }

  // Copies up to capacity characters of the FString at address, stopping at its terminator. Returns how many, or
  // -1 if it cannot be read.
  int32_t readString(const uintptr_t address, wchar_t* pText, const int32_t capacity) {
    __try {
      const RawArray& text = *reinterpret_cast<const RawArray*>(address);
      if (text.count < 0 || text.count > text.max) {
        return -1;
      }
      if (text.count == 0 || text.data == 0) {
        return 0;
      }
      const auto* pData = reinterpret_cast<const wchar_t*>(text.data);
      int32_t length = 0;
      while (length < capacity && length < text.count && pData[length] != L'\0') {
        pText[length] = pData[length];
        ++length;
      }
      return length;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
      return -1;
    }
  }

  uintptr_t locateArray(const std::vector<CodeSpan>& spans, const char* name, const char* signature) {
    const std::vector<const uint8_t*> matches = findInCode(spans, parsePattern(signature));
    if (matches.size() != 1) {
      Logger::warn(format_string("[GameMap] %s not found (%zu matches).", name, matches.size()));
      return 0;
    }
    uint32_t address;
    memcpy(&address, matches[0] + kArrayAddressOffset, sizeof(address));
    Logger::info(format_string("[GameMap] %s at 0x%08X, read at 0x%p.", name, address, static_cast<const void*>(matches[0])));
    return address;
  }

  // A URL's map is a package name or the path of the package's file, whose base name is the package's. The result
  // names the map's settings file: lowercase, as UE3 compares names regardless of case, with anything a package
  // name cannot contain replaced by '_'.
  std::string normalizeMapName(const wchar_t* pText, const size_t length) {
    size_t begin = 0;
    size_t end = length;
    for (size_t i = 0; i < length; ++i) {
      if (pText[i] == L'\\' || pText[i] == L'/') {
        begin = i + 1;
        end = length;
      } else if (pText[i] == L'.' && end == length) {
        end = i;
      }
    }

    std::string name;
    for (size_t i = begin; i < end && name.size() < dxvk::kUe3MapNameMaxLength; ++i) {
      const wchar_t c = pText[i];
      if ((c >= L'a' && c <= L'z') || (c >= L'0' && c <= L'9') || c == L'_' || c == L'-') {
        name.push_back(static_cast<char>(c));
      } else if (c >= L'A' && c <= L'Z') {
        name.push_back(static_cast<char>(c - L'A' + L'a'));
      } else {
        name.push_back('_');
      }
    }
    return name;
  }

  bool locateEngine() {
    if (g_engine != 0 && isInstance(g_engine, g_engineClassName)) {
      return true;
    }
    g_engine = 0;

    const DWORD now = GetTickCount();
    if (g_engineSearched && now - g_engineSearchTime < kEngineSearchIntervalMs) {
      return false;
    }
    g_engineSearched = true;
    g_engineSearchTime = now;

    if (g_engineClassName < 0) {
      g_engineClassName = findNameIndex(g_names, kEngineClassName);
    }
    if (g_engineClassName >= 0) {
      g_engine = findInstance(g_objects, g_engineClassName);
    }
    if (g_engine != 0) {
      Logger::info(format_string("[GameMap] TdGameEngine at 0x%08X.", static_cast<uint32_t>(g_engine)));
    }
    return g_engine != 0;
  }

  MapReading readMap() {
    if (!g_arraysLocated) {
      g_arraysLocated = true;
      const std::vector<CodeSpan> spans = collectExecutableCode();
      g_names = locateArray(spans, "GNames", kNamesSignature);
      g_objects = locateArray(spans, "GObjects", kObjectsSignature);
    }
    if (g_names == 0 || g_objects == 0) {
      return { dxvk::Ue3MapStatus::SignatureNotFound, {} };
    }
    if (!locateEngine()) {
      return { dxvk::Ue3MapStatus::EngineNotFound, {} };
    }

    wchar_t protocol[kProtocolCapacity];
    wchar_t map[kMapCapacity];
    const uintptr_t lastUrl = g_engine + kLastUrlOffset;
    const int32_t protocolLength = readString(lastUrl + kUrlProtocolOffset, protocol, kProtocolCapacity);
    const int32_t mapLength = readString(lastUrl + kUrlMapOffset, map, kMapCapacity);
    if (protocolLength < 0 || mapLength < 0) {
      return { dxvk::Ue3MapStatus::LayoutMismatch, {} };
    }
    // LastURL can be empty until the first map has loaded.
    if (protocolLength == 0 && mapLength == 0) {
      return { dxvk::Ue3MapStatus::NoMap, {} };
    }
    if (protocolLength != static_cast<int32_t>(wcslen(kUrlProtocol)) ||
        _wcsnicmp(protocol, kUrlProtocol, static_cast<size_t>(protocolLength)) != 0) {
      return { dxvk::Ue3MapStatus::LayoutMismatch, {} };
    }

    MapReading reading = { dxvk::Ue3MapStatus::Ok, normalizeMapName(map, static_cast<size_t>(mapLength)) };
    if (reading.name.empty()) {
      reading.status = dxvk::Ue3MapStatus::NoMap;
    }
    return reading;
  }

  void logReading(const MapReading& reading) {
    if (g_loggedReading && reading.status == g_loggedReading->status && reading.name == g_loggedReading->name) {
      return;
    }
    g_loggedReading = reading;

    switch (reading.status) {
    case dxvk::Ue3MapStatus::Ok:
      Logger::info(format_string("[GameMap] Map: %s.", reading.name.c_str()));
      break;
    case dxvk::Ue3MapStatus::NoMap:
      Logger::info("[GameMap] No map has loaded yet.");
      break;
    case dxvk::Ue3MapStatus::SignatureNotFound:
      Logger::warn("[GameMap] Map detection is unavailable: GNames or GObjects was not found.");
      break;
    case dxvk::Ue3MapStatus::EngineNotFound:
      Logger::info("[GameMap] No TdGameEngine object yet, retrying.");
      break;
    case dxvk::Ue3MapStatus::LayoutMismatch:
      Logger::warn(format_string("[GameMap] Map detection is unavailable: the engine at 0x%08X has no readable LastURL "
                                 "with the unreal protocol.", static_cast<uint32_t>(g_engine)));
      break;
    }
  }

  bool onQuery(const uint32_t runtimeHash) {
    const MapReading reading = readMap();
    logReading(reading);

    const uint32_t hash = reading.status == dxvk::Ue3MapStatus::Ok ? dxvk::hashUe3MapName(reading.name) : 0;
    // Fails until the channel handshake completes; the runtime repeats its query until answered.
    if (!g_pMsgChannel->send(dxvk::kUe3MapAnswerMsgName, hash, dxvk::packUe3MapAnswer(reading.name.size(), reading.status))) {
      return true;
    }
    if (hash != 0 && hash != runtimeHash) {
      for (size_t offset = 0; offset < reading.name.size(); offset += dxvk::kUe3MapChunkSize) {
        const auto [first, second] = dxvk::packUe3MapChunk(reading.name, offset);
        g_pMsgChannel->send(dxvk::kUe3MapChunkMsgName, first, second);
      }
    }
    return true;
  }
}

void GameMap::init(MessageChannelClient& msgChannel) {
  g_pMsgChannel = &msgChannel;
  msgChannel.registerHandler(dxvk::kUe3MapQueryMsgName, [](uint32_t wParam, uint32_t) {
    return onQuery(wParam);
  });
}
