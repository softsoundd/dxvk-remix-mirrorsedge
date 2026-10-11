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
#include "rtx_ue3_map_settings.h"

#include "rtx_bridge_message_channel.h"
#include "rtx_imgui.h"
#include "rtx_option_layer.h"
#include "rtx_option_layer_gui.h"
#include "rtx_option_manager.h"
#include "../../d3d9/d3d9_rtx.h"
#include "../util/log/log.h"
#include "../util/util_game_map.h"
#include "../util/util_string.h"

#include <algorithm>
#include <chrono>
#include <filesystem>

namespace dxvk {

  namespace {
    // Above rtx.conf and below Remix Logic's default, so a map's file overrides rtx.conf and a Logic layer can
    // still override the map's file.
    constexpr uint32_t kUe3MapLayerPriority = 1000;
    constexpr auto kQueryInterval = std::chrono::milliseconds(250);
    constexpr auto kQueryRetryInterval = std::chrono::seconds(2);

    // The CS thread's queries and the bridge client's answers, which arrive on the message channel's thread.
    struct BridgeMap {
      std::mutex mutex;
      bool answered = false;
      Ue3MapStatus status = Ue3MapStatus::NoMap;
      std::string name;
      uint32_t hash = 0;
      // The name of a new map, as its chunks arrive
      std::string pending;
      size_t pendingLength = 0;
      uint32_t pendingHash = 0;
      bool receiving = false;
      bool queryInFlight = false;
      std::chrono::steady_clock::time_point queryTime;
      std::chrono::steady_clock::time_point answerTime;
    };

    BridgeMap& bridgeMap() {
      static BridgeMap s_bridgeMap;
      return s_bridgeMap;
    }

    bool onAnswer(const uint32_t hash, const uint32_t answer) {
      BridgeMap& bridge = bridgeMap();
      std::lock_guard<std::mutex> lock(bridge.mutex);
      bridge.answered = true;
      bridge.queryInFlight = false;
      bridge.answerTime = std::chrono::steady_clock::now();
      bridge.status = unpackUe3MapAnswerStatus(answer);
      bridge.receiving = false;
      if (hash == 0) {
        bridge.name.clear();
        bridge.hash = 0;
      } else if (hash != bridge.hash) {
        bridge.pending.clear();
        bridge.pendingLength = std::min(unpackUe3MapAnswerLength(answer), kUe3MapNameMaxLength);
        bridge.pendingHash = hash;
        bridge.receiving = bridge.pendingLength != 0;
      }
      return true;
    }

    bool onChunk(const uint32_t first, const uint32_t second) {
      BridgeMap& bridge = bridgeMap();
      std::lock_guard<std::mutex> lock(bridge.mutex);
      if (!bridge.receiving) {
        return true;
      }
      appendUe3MapChunk(bridge.pending, first, second);
      if (bridge.pending.size() < bridge.pendingLength) {
        return true;
      }

      bridge.pending.resize(bridge.pendingLength);
      bridge.receiving = false;
      if (hashUe3MapName(bridge.pending) == bridge.pendingHash) {
        bridge.name = bridge.pending;
        bridge.hash = bridge.pendingHash;
      } else {
        // The held name stays, so the next query asks for the new one again.
        Logger::warn(str::format("[UE3 Map] Discarded a map name that does not match its hash: ", bridge.pending));
      }
      return true;
    }

    // Sends a query when one is due, and returns the current map's name or an empty string.
    std::string pollBridge() {
      BridgeMap& bridge = bridgeMap();
      bool queryDue;
      uint32_t heldHash;
      std::string map;
      {
        std::lock_guard<std::mutex> lock(bridge.mutex);
        const auto now = std::chrono::steady_clock::now();
        queryDue = bridge.queryInFlight ? now - bridge.queryTime >= kQueryRetryInterval
                                        : now - bridge.answerTime >= kQueryInterval;
        if (queryDue) {
          bridge.queryInFlight = true;
          bridge.queryTime = now;
        }
        heldHash = bridge.hash;
        if (bridge.status == Ue3MapStatus::Ok) {
          map = bridge.name;
        }
      }
      // Not sent under the mutex: the channel holds its own while it runs the answer handlers, which take this one.
      // A query lost before the channel's handshake completes is repeated after the retry interval.
      if (queryDue) {
        BridgeMessageChannel::get().send(kUe3MapQueryMsgName, heldHash, 0);
      }
      return map;
    }

    std::string mapFilePath(const std::string& directory, const std::string& map) {
      const std::string fileName = map + ".conf";
      return (directory.empty() ? std::filesystem::path(fileName) : std::filesystem::path(directory) / fileName).generic_string();
    }

    // Requests are resolved here rather than by RtxOptionManager::applyPendingValues, so a layer's file values are
    // in place before its unsaved values go over them. Expects the update mutex to be held.
    void setLayerEnabled(RtxOptionLayer& layer, const bool enabled) {
      layer.requestEnabled(enabled);
      layer.resolvePendingRequests();
      layer.applyPendingChanges();
    }

    // Leaves the layer holding exactly the values it had when its map unloaded, so values removed from it since
    // its file was read stay removed and hash sets are not merged with the file's. Expects the update mutex to be
    // held.
    void restoreLayerValues(const RtxOptionLayer& layer, const Config& values) {
      const auto globalRtxOptions = RtxOptionImpl::getGlobalOptionMap();
      for (const auto& [hash, pOption] : *globalRtxOptions) {
        // RtxOptionManager::writeOptions leaves these out of the values, so the file's stand.
        if ((pOption->getFlags() & static_cast<uint32_t>(RtxOptionFlags::NoSave)) != 0) {
          continue;
        }
        pOption->disableLayerValue(&layer);
        pOption->readOption(values, &layer);
      }
      layer.onLayerValueChanged();
    }

    // A map file must not switch map settings or UE3 mode off, which would unload the map file on the next frame
    // and so switch them on again.
    void removeSwitchesFromLayer(const RtxOptionLayer& layer) {
      RtxOptionImpl* const switches[] = {
        &D3D9Rtx::ue3EngineModeObject(),
        &Ue3MapSettings::ue3MapSettingsObject(),
        &Ue3MapSettings::ue3MapSettingsDirectoryObject(),
      };
      for (RtxOptionImpl* pOption : switches) {
        if (pOption->hasValueInLayer(&layer)) {
          Logger::warn(str::format("[UE3 Map] Ignoring ", pOption->getFullName(), " in ", layer.getFilePath(), "."));
          pOption->disableLayerValue(&layer);
          layer.onLayerValueChanged();
        }
      }
    }
  }

  Ue3MapSettings::Ue3MapSettings(DxvkDevice* device)
    : CommonDeviceObject(device) {
    BridgeMessageChannel::get().registerHandler(kUe3MapAnswerMsgName, onAnswer);
    BridgeMessageChannel::get().registerHandler(kUe3MapChunkMsgName, onChunk);
  }

  Ue3MapSettings::~Ue3MapSettings() {
    releaseLayers();
  }

  void Ue3MapSettings::onFrameEnd() {
    const bool active = D3D9Rtx::ue3EngineMode() && ue3MapSettings();
    const std::string map = active ? pollBridge() : std::string();

    std::lock_guard<std::mutex> lock(m_mutex);
    if (!active && m_map.empty()) {
      return;
    }
    const std::string& directory = ue3MapSettingsDirectory();
    const bool editing = m_editRequested.load() && !map.empty();
    const bool directoryChanged = directory != m_directory;
    if (map == m_map && !directoryChanged && (!editing || m_layer != nullptr)) {
      m_editing = editing;
      return;
    }

    std::lock_guard<std::mutex> updateLock(RtxOptionImpl::getUpdateMutex());
    leaveMap();
    if (directoryChanged) {
      // Unsaved values belong to the files of the previous folder.
      if (!m_unsavedValues.empty()) {
        Logger::warn(str::format("[UE3 Map] Discarding the unsaved changes to ", m_unsavedValues.size(), " map file(s) in ", m_directory, "."));
      }
      releaseLayers();
      m_unsavedValues.clear();
      m_directory = directory;
    }
    if (map != m_map) {
      m_map = map;
      if (m_map.empty()) {
        Logger::info(active ? "[UE3 Map] No map is loaded." : "[UE3 Map] Per-map settings are off.");
      }
    }
    enterMap(editing);
    m_editing = editing && m_layer != nullptr;
  }

  void Ue3MapSettings::leaveMap() {
    if (m_layer == nullptr) {
      return;
    }
    if (m_layer->hasUnsavedChanges()) {
      Config unsavedValues;
      RtxOptionManager::writeOptions(unsavedValues, m_layer, false);
      m_unsavedValues[m_map] = std::move(unsavedValues);
      Logger::info(str::format("[UE3 Map] Keeping the unsaved changes to ", m_layer->getFilePath(), " until ", m_map, " loads again."));
    } else {
      m_unsavedValues.erase(m_map);
    }
    setLayerEnabled(*m_layer, false);
    m_layer = nullptr;
  }

  void Ue3MapSettings::enterMap(const bool editing) {
    m_fileExists = false;
    if (m_map.empty()) {
      return;
    }

    const std::string path = mapFilePath(m_directory, m_map);
    std::error_code error;
    m_fileExists = std::filesystem::exists(path, error);
    const auto unsavedValues = m_unsavedValues.find(m_map);
    const bool hasUnsavedValues = unsavedValues != m_unsavedValues.end();

    auto layerIt = m_layers.find(m_map);
    if (layerIt == m_layers.end()) {
      if (!m_fileExists && !editing && !hasUnsavedValues) {
        Logger::info(str::format("[UE3 Map] Map ", m_map, ": no settings file at ", path, "."));
        return;
      }
      if (!m_fileExists) {
        // Saving the new layer writes the file without creating its folder.
        const std::filesystem::path folder = std::filesystem::path(path).parent_path();
        if (!folder.empty()) {
          std::filesystem::create_directories(folder, error);
        }
      }
      RtxOptionLayer* const pLayer = RtxOptionManager::acquireLayer(path, { kUe3MapLayerPriority, path });
      if (pLayer == nullptr) {
        Logger::err(str::format("[UE3 Map] Map ", m_map, ": failed to create the option layer for ", path, "."));
        return;
      }
      layerIt = m_layers.emplace(m_map, pLayer).first;
    } else {
      // Picks up edits made to the file while another map was loaded.
      layerIt->second->setConfig(Config::getOptionLayerConfig(path));
      layerIt->second->onLayerValueChanged();
      setLayerEnabled(*layerIt->second, true);
    }

    m_layer = layerIt->second;
    if (hasUnsavedValues) {
      restoreLayerValues(*m_layer, unsavedValues->second);
      m_unsavedValues.erase(unsavedValues);
    }
    removeSwitchesFromLayer(*m_layer);

    Logger::info(str::format("[UE3 Map] Map ", m_map, ": applying ", path,
                             m_fileExists ? "" : " (not saved yet)", hasUnsavedValues ? " with its unsaved changes." : "."));
  }

  void Ue3MapSettings::releaseLayers() {
    for (const auto& [map, pLayer] : m_layers) {
      RtxOptionManager::releaseLayer(pLayer);
    }
    m_layers.clear();
    m_layer = nullptr;
  }

  const RtxOptionLayer* Ue3MapSettings::getEditLayer() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_editing ? m_layer : nullptr;
  }

  std::string Ue3MapSettings::getActiveFilePath() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_layer != nullptr ? m_layer->getFilePath() : std::string();
  }

  void Ue3MapSettings::showImguiSettings() {
    // These switch the map's file on and off, so they never go into it.
    RtxOptionLayerTarget rtxConfTarget(RtxOptionEditTarget::User, nullptr);

    RemixGui::Checkbox("Apply Per-Map Settings", &ue3MapSettingsObject());
    RemixGui::SetTooltipToLastWidgetOnHover(
      "Applies the current map's settings file, named after the map in the folder below, over rtx.conf while the\n"
      "map is loaded. The bridge client reads the map's name inside the game.");
    // Committed on Enter: every change of folder reloads the map files and drops their unsaved changes.
    RemixGui::InputText("Settings Folder", &ue3MapSettingsDirectoryObject(), ImGuiInputTextFlags_EnterReturnsTrue);
    RemixGui::SetTooltipToLastWidgetOnHover("Folder of the per-map settings files, relative to the game's working directory.");

    bool answered;
    Ue3MapStatus status;
    std::string bridgeName;
    {
      BridgeMap& bridge = bridgeMap();
      std::lock_guard<std::mutex> lock(bridge.mutex);
      answered = bridge.answered;
      status = bridge.status;
      bridgeName = bridge.name;
    }

    if (!ue3MapSettings()) {
      ImGui::Text("Map: detection is off");
      return;
    }
    if (!answered) {
      ImGui::TextWrapped("Map: no response from the bridge client yet");
      return;
    }
    switch (status) {
    case Ue3MapStatus::Ok:
      ImGui::Text("Map: %s", bridgeName.c_str());
      break;
    case Ue3MapStatus::NoMap:
      ImGui::Text("Map: none has loaded yet");
      break;
    case Ue3MapStatus::SignatureNotFound:
      ImGui::TextWrapped("Map: unavailable, GNames or GObjects was not found in this executable (see bridge32.log)");
      break;
    case Ue3MapStatus::EngineNotFound:
      ImGui::Text("Map: waiting for the game's engine object");
      break;
    case Ue3MapStatus::LayoutMismatch:
      ImGui::TextWrapped("Map: unavailable, this executable's engine layout is not the supported one (see bridge32.log)");
      break;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_map.empty()) {
      return;
    }

    const std::string path = mapFilePath(m_directory, m_map);
    ImGui::TextWrapped("Settings file: %s", path.c_str());
    if (m_layer == nullptr) {
      ImGui::TextWrapped("Not found, so rtx.conf applies on this map. Editing this map's settings creates it.");
    } else if (!m_fileExists) {
      ImGui::Text("Not saved yet");
    }

    bool editRequested = m_editRequested.load();
    if (ImGui::Checkbox("Edit This Map's Settings", &editRequested)) {
      m_editRequested = editRequested;
    }
    RemixGui::SetTooltipToLastWidgetOnHover(
      "Sends the developer menu's changes to this map's settings file instead of rtx.conf, so they apply on this map\n"
      "only: the Sky Tuning sun elevation, rotation and illuminance, for example. User settings (graphics\n"
      "preferences) still go to user.conf. Stays on across map loads, editing each map's own file.");

    if (m_layer == nullptr) {
      return;
    }
    if (m_layer->hasUnsavedChanges()) {
      ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.0f, 1.0f), "(unsaved changes)");
    }
    if (OptionLayerUI::renderLayerButtons(m_layer, "Ue3Map")) {
      std::error_code error;
      m_fileExists = std::filesystem::exists(path, error);
    }
    if (RemixGui::CollapsingHeader("View Settings##Ue3Map")) {
      ImGui::Indent();
      OptionLayerUI::RenderOptions renderOptions;
      renderOptions.showUnchanged = true;
      renderOptions.uniqueId = "##Ue3MapLayerList";
      OptionLayerUI::renderToImGui(m_layer, renderOptions);
      ImGui::Unindent();
    }
    if (!m_unsavedValues.empty()) {
      ImGui::TextWrapped("Unsaved changes to %zu other map file(s) are kept until their map loads again or the game exits.",
                         m_unsavedValues.size());
    }
  }
}
