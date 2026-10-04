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

#include "rtx_common_object.h"
#include "rtx_option.h"
#include "../util/config/config.h"

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>

namespace dxvk {

  class RtxOptionLayer;

  // Mirror's Edge: a settings file per map. The bridge client reads the name of the map the game has loaded, and
  // that map's file is applied as an option layer above rtx.conf while it is loaded.
  // See documentation/UE3Compatibility.md, "Per-map settings".
  class Ue3MapSettings : public CommonDeviceObject {
  public:
    RTX_OPTION("rtx.d3d9", bool, ue3MapSettings, true,
               "Mirror's Edge: apply a settings file per map. The bridge client reads the name of the map the game has "
               "loaded, and while that map is loaded <rtx.d3d9.ue3MapSettingsDirectory>/<map>.conf is applied over "
               "rtx.conf, so settings such as the Physical Atmosphere sun can differ per map. Only active in "
               "rtx.d3d9.ue3EngineMode; see documentation/UE3Compatibility.md, \"Per-map settings\".");
    RTX_OPTION("rtx.d3d9", std::string, ue3MapSettingsDirectory, "rtx-remix/maps",
               "Mirror's Edge: folder of the per-map settings files, relative to the game's working directory. Each is "
               "named after the map's package in lowercase, e.g. edge_p.conf for the Prologue or tdmainmenu.conf for "
               "the main menu.");

    explicit Ue3MapSettings(DxvkDevice* device);
    ~Ue3MapSettings();

    // CS thread, at the end of every frame before the option layers are applied.
    void onFrameEnd();

    // The current map's layer while its settings are being edited, otherwise null. Developer menu edits of options
    // other than user settings go to it instead of rtx.conf.
    const RtxOptionLayer* getEditLayer() const;

    // The file of the current map's layer, or empty while no map file applies.
    std::string getActiveFilePath() const;

    void showImguiSettings();

  private:
    void leaveMap();
    void enterMap(bool editing);
    void releaseLayers();

    std::atomic<bool> m_editRequested { false };

    // Guards everything below; the CS thread switches maps while the ImGui thread shows them.
    mutable std::mutex m_mutex;
    std::string m_map;
    std::string m_directory;
    RtxOptionLayer* m_layer = nullptr;
    bool m_fileExists = false;
    bool m_editing = false;
    // Every layer acquired this session by map. A layer is only disabled when its map unloads, so a pointer the
    // ImGui thread holds stays valid.
    std::unordered_map<std::string, RtxOptionLayer*> m_layers;
    // A disabled layer loses its values, so unsaved changes wait here until their map is loaded again.
    std::unordered_map<std::string, Config> m_unsavedValues;
  };

}
