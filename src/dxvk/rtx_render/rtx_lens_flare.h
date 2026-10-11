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

#include <array>
#include <vector>

#include "rtx_resources.h"
#include "rtx_common_object.h"
#include "rtx_option.h"
#include "rtx_lens_system.h"
#include "rtx/pass/lens_flare/lens_flare.h"

namespace dxvk {

  class RtxContext;
  class RtxSunProbe;
  struct DxvkContextState;

  enum class LensFlareQuality : int {
    Fast = 0,
    RayTraced,
  };

  enum class LensFlareDebugView : int {
    None = 0,
    FlareOnly,
    GhostBounds,
  };

  /**
   * \brief Lens flare of the Physical Atmosphere sun.
   *
   * Draws the brightest ghosts of the sun's light reflected twice inside the camera's lens, rtx.lens.*, either
   * paraxially, each placed by a few real rays, or ray traced through the lens's real surfaces, dimmed by the sun's
   * visibility.
   */
  class RtxLensFlare : public CommonDeviceObject {
  public:
    explicit RtxLensFlare(DxvkDevice* device);

    static bool isEnabled();

    // Adds the flare to rtOutput's final output. Runs after motion blur and before tone mapping. The ray traced ghosts'
    // raster pass changes the context's state, which it restores from state.
    void dispatch(RtxContext& ctx, DxvkContextState& state, const Resources::RaytracingOutput& rtOutput, const RtxSunProbe& sunProbe);

    // The ghosts the flare draws itself, whose light the sun's veil leaves out.
    const std::vector<LensSystem::Ghost>& getDrawnGhosts() const {
      static const std::vector<LensSystem::Ghost> kNone;
      return isEnabled() ? m_ghosts : kNone;
    }

    void showImguiSettings();

    RTX_OPTION("rtx.lensFlare", bool, enable, false, "Draws the lens flare of the Physical Atmosphere sun.");
    RTX_OPTION_ARGS("rtx.lensFlare", float, intensity, 1.0f,
                    "Scale on the ghosts' physically derived brightness. 1 = physical.",
                    args.minValue = 0.0f);
    RTX_OPTION("rtx.lensFlare", LensFlareQuality, quality, LensFlareQuality::Fast,
               "How the ghosts are drawn: 0: Fast (paraxial optics after Lee and Eisemann 2013, each ghost placed by a few real\n"
               "rays), 1: Ray Traced (grids of real rays through the lens's surfaces, Hullin et al. 2011, with their aberrations,\n"
               "caustics, rims and vignetting).");
    RTX_OPTION_ARGS("rtx.lensFlare", int, maxGhosts, 32,
                    "Number of ghosts considered, those of every pair of reflecting surfaces that are brightest on axis. Ghosts\n"
                    "too faint to see are skipped. The rest of the lens's ghosts make up its veil in the convolution bloom.",
                    args.minValue = 1, args.maxValue = LENS_FLARE_MAX_GHOSTS);
    RTX_OPTION_ARGS("rtx.lensFlare", int, traceResolution, 48,
                    "Cells per side of each ray traced ghost's grid of rays, which spans the rays that pass the lens. More cells\n"
                    "resolve the ghosts' caustic rims and vignetted edges finer.",
                    args.minValue = 8, args.maxValue = 128);
    RTX_OPTION_ARGS("rtx.lensFlare", float, edgeSoftness, 1.0f,
                    "Scale on the blur of the ghosts' edges by the sun's disc, every point of which casts the ghost. 1 = physical,\n"
                    "0 leaves only antialiasing.",
                    args.minValue = 0.0f, args.maxValue = 10.0f);
    RTX_OPTION("rtx.lensFlare", bool, reducedResolution, true,
               "Draws the ray traced ghosts whose sharpest edges the sun's disc blurs over several pixels at half or quarter\n"
               "resolution, which the blur hides, and adds them to the output bilinearly.");
    RTX_OPTION("rtx.lensFlare", LensFlareDebugView, debugView, LensFlareDebugView::None,
               "Lens flare debug view: 0: None, 1: Flare Only, 2: Ghost Bounds (the fast path's culling circles over the image).");

  private:
    static constexpr uint32_t kFocus = LensSystem::kFocus;
    static constexpr uint32_t kWavelengthCount = LensSystem::kWavelengthCount;
    // The resolutions the traced ghosts are drawn at: full, half and quarter.
    static constexpr uint32_t kRasterLevels = 3;

    void selectGhosts(double minSensorFromPupil);
    void dispatchFast(RtxContext& ctx, const Resources::Resource& color, const LensFlareArgs& args,
                      const std::vector<LensFlareGhost>& ghosts, const RtxSunProbe& sunProbe);
    // Each resolution's instances lie in turn, those of level l from levelStarts[l] up to levelStarts[l + 1].
    void dispatchRayTraced(RtxContext& ctx, DxvkContextState& state, const Resources::Resource& color,
                           const LensFlareTraceArgs& traceArgs, const LensFlareRasterArgs& rasterArgs,
                           const std::vector<LensFlareTraceInstance>& instances,
                           const std::array<uint32_t, kRasterLevels + 1>& levelStarts, const RtxSunProbe& sunProbe);
    void updateSurfaceBuffer(RtxContext& ctx);
    void updateReducedTargets(RtxContext& ctx, const VkExtent3D& extent);

    LensSystem m_lensSystem;
    std::vector<LensSystem::Ghost> m_ghosts;
    // Whether each selected ghost was drawn across the spectrum last frame.
    std::vector<uint8_t> m_spectralGhosts;
    // What the ghost selection depends on.
    uint32_t m_selectionVersion = ~0u;
    uint32_t m_lensVersion = 0;
    float m_selectionMinSensorFromPupil = -1.0f;
    int m_selectionMaxGhosts = -1;
    uint32_t m_candidateCount = 0;
    uint32_t m_drawnGhostCount = 0;
    uint32_t m_drawnInstanceCount = 0;
    std::array<uint32_t, kRasterLevels> m_levelGhostCounts {};
    double m_stopRadius = 0.0;
    double m_entrancePupilRadius = 0.0;

    Rc<DxvkBuffer> m_ghostBuffer;
    Rc<DxvkBuffer> m_surfaceBuffer;
    uint32_t m_surfaceVersion = ~0u;
    Rc<DxvkBuffer> m_instanceBuffer;
    Rc<DxvkBuffer> m_domainBuffer;
    Rc<DxvkBuffer> m_vertexBuffer;
    // The half and quarter resolution layers.
    std::array<Resources::Resource, kRasterLevels - 1> m_reducedTargets;
  };

}
