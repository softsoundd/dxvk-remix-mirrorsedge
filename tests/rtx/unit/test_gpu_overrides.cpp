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
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

#include "../../../src/dxvk/rtx_render/rtx_gpu_overrides.h"
#include "../../../src/dxvk/rtx_render/rtx_option.h"
#include "../../../src/dxvk/rtx_render/rtx_option_layer.h"
#include "../../../src/dxvk/rtx_render/rtx_option_manager.h"
#include "../../../src/util/config/config.h"

#include "../../test_utils.h"

namespace dxvk {
  // Logger needed by some shared code used in this Unit Test.
  Logger Logger::s_instance("test_gpu_overrides.log");

namespace gpu_overrides_test {

  #define TEST_ASSERT(condition, message) \
    do { \
      if (!(condition)) { \
        std::ostringstream oss; \
        oss << "FAILED: " << __FUNCTION__ << " line " << __LINE__ << ": " << message; \
        throw DxvkError(oss.str()); \
      } \
    } while(0)

  const std::string kPreset = GpuOverrides::kGraphicsPresetPrefix;
  const std::string kDlss = GpuOverrides::kDlssModePrefix;
  // Bool namespace registered by this test for the typed-value cases the string namespaces cannot exercise.
  const std::string kFlag = "rtx.gpuOverrideTest.flag.";

  RtxOptionLayer* acquireConfigLayer(const Config& config, uint32_t priority, const char* name, float blendStrength = 1.0f) {
    return RtxOptionManager::acquireLayer("", RtxOptionLayerKey { priority, name }, blendStrength, 0.1f, false, &config);
  }

  void resolve() {
    RtxOptionManager::applyPendingValues(nullptr, false);
  }

  namespace fs = std::filesystem;

  fs::path confPath(const char* name) {
    return fs::temp_directory_path() / (std::string("gpu_overrides_") + name + ".conf");
  }

  void writeFile(const fs::path& path, const std::string& text) {
    std::ofstream file(path, std::ios::trunc);
    file << text;
  }

  // Returns the value text for key in a saved config file, or "<missing>".
  std::string savedValue(const fs::path& path, const std::string& key) {
    std::ifstream file(path);
    std::string line;
    const std::string linePrefix = key + " = ";
    while (std::getline(file, line)) {
      if (line.compare(0, linePrefix.size(), linePrefix) == 0) {
        return line.substr(linePrefix.size());
      }
    }
    return "<missing>";
  }

  std::string presetName(const std::optional<GraphicsPreset>& preset) {
    return preset ? GpuOverrides::graphicsPresetName(*preset) : "<none>";
  }

  std::string dlssName(const std::optional<DLSSProfile>& profile) {
    return profile ? GpuOverrides::dlssModeName(*profile) : "<none>";
  }

  void test_gpuIdFormat() {
    TEST_ASSERT(GpuOverrides::formatGpuId(0x10DE, 0x2684) == "10DE_2684", "NVIDIA ID");
    TEST_ASSERT(GpuOverrides::formatGpuId(0x8086, 0x56A0) == "8086_56A0", "Intel ID");
    TEST_ASSERT(GpuOverrides::formatGpuId(0x10005, 0x1) == "10005_0001", "five-digit vendor ID");

    TEST_ASSERT(GpuOverrides::isCanonicalGpuId("10DE_2684"), "canonical");
    TEST_ASSERT(GpuOverrides::isCanonicalGpuId("10005_0001"), "canonical five-digit vendor");
    for (const char* id : { "10de_2684", "10DE_684", "10DE2684", "_2684", "10DE_", "10DE_2684_1", "10DE_XYZ1" }) {
      TEST_ASSERT(!GpuOverrides::isCanonicalGpuId(id), "non-canonical " << id);
    }
  }

  void test_valueNames() {
    TEST_ASSERT(presetName(GpuOverrides::parseGraphicsPreset("ultra")) == "Ultra", "lowercase preset");
    TEST_ASSERT(presetName(GpuOverrides::parseGraphicsPreset("HIGH")) == "High", "uppercase preset");
    TEST_ASSERT(presetName(GpuOverrides::parseGraphicsPreset("medium")) == "Medium", "medium preset");
    TEST_ASSERT(presetName(GpuOverrides::parseGraphicsPreset("Low")) == "Low", "low preset");
    for (const char* name : { "Auto", "Custom", "", "Hihg" }) {
      TEST_ASSERT(!GpuOverrides::parseGraphicsPreset(name), "rejected preset " << name);
    }

    for (const char* name : { "UltraPerformance", "Performance", "Balanced", "Quality", "FullResolution" }) {
      TEST_ASSERT(dlssName(GpuOverrides::parseDlssMode(name)) == name, "DLSS mode " << name);
    }
    TEST_ASSERT(dlssName(GpuOverrides::parseDlssMode("quality")) == "Quality", "lowercase DLSS mode");
    for (const char* name : { "Auto", "Ultra Performance", "", "Invalid" }) {
      TEST_ASSERT(!GpuOverrides::parseDlssMode(name), "rejected DLSS mode " << name);
    }

  }

  void test_lookups() {
    Config weak;
    weak.setOption(kPreset + "10DE_2684", std::string("high"));
    weak.setOption(kDlss + "10DE_2684", std::string("Balanced"));
    weak.setOption(kPreset + "1002_744C", std::string("Hihg"));
    weak.setOption(kPreset + "10de_1111", std::string("Ultra"));
    RtxOptionLayer* weakLayer = acquireConfigLayer(weak, 1000, "GpuOverridesWeak");
    resolve();

    TEST_ASSERT(presetName(GpuOverrides::findGraphicsPreset("10DE_2684")) == "High", "preset lookup");
    TEST_ASSERT(dlssName(GpuOverrides::findDlssMode("10DE_2684")) == "Balanced", "DLSS lookup");
    TEST_ASSERT(!GpuOverrides::findGraphicsPreset("1002_744C"), "unrecognized value ignored");
    TEST_ASSERT(!GpuOverrides::findGraphicsPreset("10DE_1111"), "non-canonical key does not match");
    TEST_ASSERT(!GpuOverrides::findDlssMode("1002_744C"), "missing DLSS override");
    TEST_ASSERT(!GpuOverrides::findGraphicsPreset(""), "empty GPU ID");
    GpuOverrides::logInvalidEntries();

    Config strong;
    strong.setOption(kPreset + "10DE_2684", std::string("Low"));
    RtxOptionLayer* strongLayer = acquireConfigLayer(strong, 2000, "GpuOverridesStrong");
    resolve();
    TEST_ASSERT(presetName(GpuOverrides::findGraphicsPreset("10DE_2684")) == "Low", "stronger layer wins");

    RtxOptionManager::releaseLayer(strongLayer);
    RtxOptionManager::releaseLayer(weakLayer);
    resolve();
    TEST_ASSERT(!GpuOverrides::findGraphicsPreset("10DE_2684"), "no override once layers are released");
  }

  void test_dlssModeCache() {
    GpuOverrides::setCurrentGpu(0x10DE, 0x2684);
    TEST_ASSERT(GpuOverrides::getCurrentGpuId() == "10DE_2684", "current GPU");
    GpuOverrides::updateDlssMode();
    TEST_ASSERT(!GpuOverrides::getDlssMode(), "no DLSS override without a key");

    Config config;
    config.setOption(kDlss + "10DE_2684", std::string("Performance"));
    RtxOptionLayer* layer = acquireConfigLayer(config, 1000, "GpuOverridesDlssMode");
    resolve();
    TEST_ASSERT(!GpuOverrides::getDlssMode(), "cached value holds until the next update");
    GpuOverrides::updateDlssMode();
    TEST_ASSERT(dlssName(GpuOverrides::getDlssMode()) == "Performance", "update picks up the override");

    RtxOptionManager::releaseLayer(layer);
    resolve();
    TEST_ASSERT(dlssName(GpuOverrides::getDlssMode()) == "Performance", "removal waits for the next update");
    GpuOverrides::updateDlssMode();
    TEST_ASSERT(!GpuOverrides::getDlssMode(), "update clears a removed override");
  }

  void test_inactiveOverrides() {
    GpuOverrides::setCurrentGpu(0x10DE, 0x2684);
    const std::string dlssKey = kDlss + "10DE_2684";

    // Below its blend threshold (0.1) the layer contributes nothing, so nothing counts as an override.
    Config overrides;
    overrides.setOption(kPreset + "10DE_2684", std::string("High"));
    overrides.setOption(dlssKey, std::string("Balanced"));
    RtxOptionLayer* layer = acquireConfigLayer(overrides, 1000, "GpuOverridesInactive", 0.0f);
    resolve();
    TEST_ASSERT(!GpuOverrides::findActiveValue(dlssKey), "inactive DLSS entry");
    TEST_ASSERT(!GpuOverrides::findGraphicsPreset("10DE_2684"), "no preset override");
    GpuOverrides::apply();
    TEST_ASSERT(!GpuOverrides::getDlssMode(), "inactive DLSS entry not cached");

    layer->requestBlendStrength(0.1f);
    resolve();
    TEST_ASSERT(dlssName(GpuOverrides::findDlssMode("10DE_2684")) == "Balanced", "entry at threshold is active");
    TEST_ASSERT(presetName(GpuOverrides::findGraphicsPreset("10DE_2684")) == "High", "preset at threshold");
    GpuOverrides::apply();
    TEST_ASSERT(dlssName(GpuOverrides::getDlssMode()) == "Balanced", "apply refreshes the DLSS cache");

    layer->requestBlendStrength(0.05f);
    resolve();
    GpuOverrides::apply();
    TEST_ASSERT(!GpuOverrides::getDlssMode(), "faded DLSS entry cleared");

    // A weaker active layer is used when the stronger one is inactive.
    Config weak;
    weak.setOption(dlssKey, std::string("Quality"));
    RtxOptionLayer* weakLayer = acquireConfigLayer(weak, 950, "GpuOverridesWeakDlssMode");
    resolve();
    TEST_ASSERT(dlssName(GpuOverrides::findDlssMode("10DE_2684")) == "Quality", "weaker active layer used");
    GpuOverrides::apply();
    TEST_ASSERT(dlssName(GpuOverrides::getDlssMode()) == "Quality", "weaker override applied");

    RtxOptionManager::releaseLayer(weakLayer);
    RtxOptionManager::releaseLayer(layer);
    resolve();
    GpuOverrides::apply();
    TEST_ASSERT(!GpuOverrides::getDlssMode(), "no override after release");
  }

  void test_redundancyCleanupKeepsDefaultValuedEntry() {
    const std::string flagKey = kFlag + "10DE_2684";
    const fs::path path = confPath("flag_cleanup");
    // The entry matches the namespace default, so only its presence distinguishes it.
    writeFile(path, flagKey + " = True\n");
    RtxOptionLayer* layer = RtxOptionManager::acquireLayer(path.string(), RtxOptionLayerKey { 1100, "GpuOverridesCleanup" }, 1.0f, 0.1f, false, nullptr);
    resolve();
    TEST_ASSERT(GpuOverrides::findActiveValue(flagKey) == std::optional<std::string>("True"), "entry active");

    RtxOptionManager::removeRedundantLayerValues(layer);
    TEST_ASSERT(layer->save(), "save");
    TEST_ASSERT(savedValue(path, flagKey) == "True", "cleanup keeps the entry");
    layer->reload();
    resolve();
    TEST_ASSERT(GpuOverrides::findActiveValue(flagKey) == std::optional<std::string>("True"), "entry found after reload");

    RtxOptionManager::releaseLayer(layer);
    resolve();
    TEST_ASSERT(!GpuOverrides::findActiveValue(flagKey), "no entry after release");
    fs::remove(path);
  }

  void test_rejectedEntries() {
    GpuOverrides::setCurrentGpu(0x10DE, 0x2684);
    const std::string otherKey = kFlag + "1002_744C";
    const fs::path path = confPath("rejected");
    writeFile(path, otherKey + " = Flase\n");
    Config weak;
    weak.setOption(otherKey, std::string("True"));
    RtxOptionLayer* weakLayer = acquireConfigLayer(weak, 1150, "GpuOverridesRejectedWeak");
    RtxOptionLayer* layer = RtxOptionManager::acquireLayer(path.string(), RtxOptionLayerKey { 1200, "GpuOverridesRejected" }, 1.0f, 0.1f, false, nullptr);
    resolve();

    // The editor lists rejected entries for any GPU next to the effective value from other layers.
    auto entries = layer->getRejectedEntries(kFlag);
    TEST_ASSERT(entries.size() == 1 && entries[0].first == otherKey && entries[0].second == "Flase", "rejected entry with its text");
    TEST_ASSERT(GpuOverrides::findActiveValue(otherKey) == std::optional<std::string>("True"), "weaker valid value in effect");
    RtxOptionManager::releaseLayer(weakLayer);
    resolve();
    TEST_ASSERT(!GpuOverrides::findActiveValue(otherKey), "rejected entry alone has no value");
    TEST_ASSERT(layer->getRejectedEntries(kFlag).size() == 1, "rejected-only entry still listed");

    TEST_ASSERT(RtxOptionManager::setDynamicValue(otherKey, "False", layer) == DynamicOptionResult::Success, "replace");
    resolve();
    TEST_ASSERT(layer->getRejectedEntries(kFlag).empty(), "replacement clears the rejection");
    TEST_ASSERT(GpuOverrides::findActiveValue(otherKey) == std::optional<std::string>("False"), "replacement value");
    TEST_ASSERT(layer->save(), "save replacement");
    TEST_ASSERT(savedValue(path, otherKey) == "False", "replacement saved");

    writeFile(path, otherKey + " = Flase\n");
    layer->reload();
    resolve();
    TEST_ASSERT(layer->getRejectedEntries(kFlag).size() == 1, "rejected again after reload");
    TEST_ASSERT(RtxOptionManager::clearDynamicValue(otherKey, layer) == DynamicOptionResult::Success, "remove");
    resolve();
    TEST_ASSERT(layer->getRejectedEntries(kFlag).empty(), "removal clears the rejection");
    TEST_ASSERT(layer->save(), "save removal");
    TEST_ASSERT(savedValue(path, otherKey) == "<missing>", "removed entry not saved");
    layer->reload();
    resolve();
    TEST_ASSERT(layer->getRejectedEntries(kFlag).empty() && !GpuOverrides::findActiveValue(otherKey), "removal survives reload");

    RtxOptionManager::releaseLayer(layer);
    resolve();
    fs::remove(path);
  }

  void runAllTests() {
    std::cout << "Running GPU override tests" << std::endl;

    GpuOverrides::registerNamespaces();
    DynamicOptionNamespace flagNamespace;
    flagNamespace.prefix = kFlag;
    flagNamespace.type = OptionType::Bool;
    flagNamespace.defaultValue = "True";
    flagNamespace.description = "Test-only per-GPU flag.";
    TEST_ASSERT(RtxOptionManager::registerDynamicNamespace(flagNamespace), "flag namespace registered");
    RtxOptionLayer::initializeSystemLayers();
    RtxOptionImpl::setInitialized(true);
    RtxOptionManager::markOptionsWithCallbacksDirty();
    RtxOptionManager::applyPendingValues(nullptr, true);

    test_gpuIdFormat();
    test_valueNames();
    test_lookups();
    test_dlssModeCache();
    test_inactiveOverrides();
    test_redundancyCleanupKeepsDefaultValuedEntry();
    test_rejectedEntries();

    std::cout << "All GPU override tests PASSED" << std::endl;
  }

}  // namespace gpu_overrides_test
}  // namespace dxvk

int main() {
  try {
    dxvk::gpu_overrides_test::runAllTests();
  } catch (const dxvk::DxvkError& error) {
    std::cerr << "TEST FAILED: " << error.message() << std::endl;
    return -1;
  } catch (const std::exception& e) {
    std::cerr << "TEST FAILED with exception: " << e.what() << std::endl;
    return -1;
  }
  return 0;
}
