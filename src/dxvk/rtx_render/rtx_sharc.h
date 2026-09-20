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

#include "rtx_option.h"
#include "rtx_resources.h"
#include "rtx/pass/sharc/sharc_args.h"

namespace dxvk {
  class RtxContext;
  class DxvkDevice;

  // Owns the SHARC world-space radiance cache: a hash grid of irradiance keyed on position,
  // distance level, normal octant and ray portal space. A sparse update pass fills it and the
  // full-resolution indirect pass reads it, terminating paths that hit a converged cell.
  class RtxSharc : public CommonDeviceObject, public RtxPass {
  public:
    enum class QualityPreset : uint32_t {
      Medium = 0,
      High,
      Ultra,

      Count
    };

    explicit RtxSharc(DxvkDevice* device);

    static bool checkIsSupported(const DxvkDevice* device);

    void setRaytraceArgs(RtxContext& ctx, RaytraceArgs& args);
    void bindResources(RtxContext& ctx) const;
    void dispatchResolve(RtxContext& ctx, const Resources::RaytracingOutput& rtOutput);
    void showImguiSettings();
    void setQualityPreset(QualityPreset preset);

    // Selects between the compact and full propagation depth update shaders.
    uint32_t getUpdateBounces() const { return m_args.updateBounces; }
    uint32_t getUpdateTileSize() const { return m_args.updateTileSize; }

    // Applies the tier by writing the individual knobs below, as NRC does.
    static void onQualityPresetChanged(DxvkDevice* device);

    RTX_OPTION_ARGS("rtx.sharc", QualityPreset, qualityPreset, QualityPreset::High,
                    "Quality Preset: Medium (0), High (1), Ultra (2).\n"
                    "Adjusts how much of the frame ray budget the cache update consumes, how deep its update\n"
                    "paths run, and how many cells it can hold. Lower presets trace fewer and shorter update\n"
                    "paths, which is cheaper but leaves cells that are only reached by a bounce under-sampled.",
                    args.environment = "RTX_SHARC_QUALITY_PRESET",
                    args.onChangeCallback = &onQualityPresetChanged,
                    args.flags = RtxOptionFlags::UserSetting);
    inline static QualityPreset s_prevQualityPreset = QualityPreset::Count;

    RTX_OPTION("rtx.sharc", int, capacityLog2, 22,
               "Cache capacity exponent, 18 to 22. Each slot costs 40 bytes, so 20 uses 40 MiB, 21 uses 80 MiB "
               "and 22 uses 160 MiB. The resolve pass runs one thread per slot every frame, so a larger capacity "
               "costs resolve time whether or not the slots hold anything.");
    RTX_OPTION("rtx.sharc", int, updateTileSize, 5,
               "One cache update path per NxN tile of the render target, 1 to 16. The update pass traces one path "
               "per tile, so its cost falls as 1/N squared; this is the main cost lever. Larger tiles under-sample "
               "the cells that are only ever reached by a bounce.");
    RTX_OPTION("rtx.sharc", int, updateBounces, 3,
               "Maximum cache update bounces, 1 to 8. With rtx.sharc.updatePrimaryVertex enabled the primary takes "
               "a propagation slot, so 3 or fewer selects the compact four-slot update shader.");
    RTX_OPTION("rtx.sharc", int, accumulationFrames, 8, "Temporal cache accumulation, 1 to 64 frames.");
    RTX_OPTION("rtx.sharc", int, staleFrames, 32, "Evict entries that go unobserved for this many frames, 8 to 128.");
    RTX_OPTION("rtx.sharc", float, gridScale, 50.0f,
               "Hash grid density, 1 to 1000. A cell edge is the vertex distance from the camera rounded down to a "
               "power of two and divided by this value, so cells grow with distance and the setting behaves as an "
               "angle rather than a length: at 50 a cell spans roughly 0.6 to 1.1 degrees of arc anywhere in the "
               "scene. Carrying no world units, one value suits any game whatever its unit convention. Larger "
               "values give finer cells and quadratically more of them.");
    RTX_OPTION("rtx.sharc", float, minRoughness, 0.05f,
               "Minimum isotropic roughness (GGX alpha) for a cached surface, 0.05 to 1. A cell holds one "
               "non-directional radiance value, so a surface below this reflects more sharply than the cell can "
               "represent. The width of the lobe that reached the surface is tested separately, at query time.");
    RTX_OPTION("rtx.sharc", float, maxEmissiveLuminance, 0.1f,
               "Cache surfaces whose emissive luminance is at or below this. The cache holds reflected light and "
               "the path adds emission separately, so emissive surfaces are excluded; at 0 any emission at all "
               "disqualifies a surface, which rejects every faint emissive map.");
    RTX_OPTION("rtx.sharc", int, minSampleCount, 2,
               "Ignore cells holding this many accumulated samples or fewer, 0 to 32. At 0 a single sample answers "
               "a query, so geometry coming into view can read one bright path as converged radiance and glow.");
    RTX_OPTION("rtx.sharc", float, maxDepositRatio, 20.0f,
               "Ceiling on a single deposit into a cell, as a multiple of what the cell already holds; 0 disables "
               "it. A cell is a mean of its samples, so an outlier is not averaged away, only divided by the sample "
               "count. An absolute cap has to sit below the dimmest cell worth keeping, so scaling the limit by the "
               "converged value of the cell leaves a bright cell its headroom while a dark one refuses spikes.");
    RTX_OPTION("rtx.sharc", float, minDepositCeiling, 2.0f,
               "Absolute luminance floor under the rtx.sharc.maxDepositRatio ceiling. Without it a cell near black "
               "pins its own ceiling near zero and can never brighten when the lighting changes.");
    RTX_OPTION("rtx.sharc", float, updateRoughnessClamp, 0.25f,
               "Roughen materials to at least this isotropic roughness (GGX alpha) while the cache is updated, 0 to "
               "disable. A cell cannot represent a narrow specular highlight, so on a glossy surface every update "
               "path arriving from a different direction deposits a different value and the cell mean never "
               "settles. This changes what a cell stores, not which surfaces qualify.");
    RTX_OPTION("rtx.sharc", bool, allowSpecularPaths, true,
               "Allow cache insertion and reuse at rough opaque surfaces reached by a non-diffuse lobe. The lobe "
               "footprint must still have spread wider than the cell. Changing this resets the cache.");
    RTX_OPTION("rtx.sharc", bool, updatePrimaryVertex, true,
               "Let update paths also deposit their camera-visible primary vertex, valued as the direct pass RTXDI "
               "lighting plus the sampled continuation, so a path whose first bounce exits to the sky still feeds "
               "one cell. Changing this resets the cache.");
    RTX_OPTION("rtx.sharc", int, updateSkyRetries, 1,
               "When the first bounce of an update path exits to the sky, re-sample that bounce from a cosine lobe "
               "about the primary normal up to this many times, 0 to 4, so the update budget lands on geometry more "
               "often outdoors. Changing this resets the cache.");

  private:
    friend class ImGUI;

    bool isEnabled() const override;
    void onFrameBegin(Rc<DxvkContext>& ctx, const FrameBeginContext& frameBeginCtx) override;
    bool onActivation(Rc<DxvkContext>& ctx) override;
    void onDeactivation() override;

    bool allocateBuffers(uint32_t capacity);

    Rc<DxvkBuffer> m_hash;
    Rc<DxvkBuffer> m_accumulation;
    Rc<DxvkBuffer> m_resolved;

    SharcArgs m_args = {};
    bool m_resetRequested = true;
    bool m_allocationFailed = false;
    uint32_t m_compatibilityFlags = 0;
    uint32_t m_cacheAge = 0;
    uint32_t m_lastFrame = ~0u;
    bool m_clearedThisFrame = true;
    const char* m_status = "Inactive";
  };
}
