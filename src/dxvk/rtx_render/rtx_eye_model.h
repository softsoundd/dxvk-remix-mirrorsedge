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
#include <cstdint>
#include <vector>

#include "dxvk_include.h"
#include "rtx_option.h"
#include "../../util/util_vector.h"

namespace dxvk {

  class DxvkBuffer;
  class DxvkContext;
  class DxvkDevice;
  class DxvkImage;

  /**
   * \brief The human eye as the observer.
   *
   * Its pupil, which follows the field's light, the particles that cast its ciliary corona, and its scatter, for the
   * convolution bloom's kernel. Angles are the game's own, from its field of view.
   */
  class EyeModel {
  public:
    RTX_OPTION_ARGS("rtx.lens.eye", float, age, 25.0f,
                    "Age of the observer in years. Older eyes have smaller pupils (from 20 to 83 years, Watson and Yellott 2012)\n"
                    "and scatter more light (CIE 135/1-6).",
                    args.minValue = 1.0f, args.maxValue = 100.0f);
    RTX_OPTION_ARGS("rtx.lens.eye", float, pigmentation, 0.5f,
                    "Ocular pigmentation factor of the CIE glare: 0 for very dark eyes, 0.5 brown, 1 blue-green and 1.2 very light\n"
                    "blue. Lighter eyes let more light through the eye wall.",
                    args.minValue = 0.0f, args.maxValue = 1.2f);
    RTX_OPTION_ARGS("rtx.lens.eye", float, pupilDiameterMm, 0.0f,
                    "Fixed pupil diameter in mm, 2 to 8. 0 follows the field's luminance.",
                    args.minValue = 0.0f, args.maxValue = 8.0f);
    RTX_OPTION_ARGS("rtx.lens.eye", float, luminanceScale, 0.0f,
                    "Candela per square metre per unit of radiance, which the pupil's response needs. 0 takes it from the Physical\n"
                    "Atmosphere's sun, whose illuminance above the atmosphere is 128 klux.",
                    args.minValue = 0.0f);
    RTX_OPTION_ARGS("rtx.lens.eye", float, constrictionSeconds, 0.4f,
                    "Time constant of the pupil closing as the field brightens.",
                    args.minValue = 0.0f);
    RTX_OPTION_ARGS("rtx.lens.eye", float, dilationSeconds, 4.0f,
                    "Time constant of the pupil opening as the field darkens, much slower than its closing.",
                    args.minValue = 0.0f);
    RTX_OPTION_ARGS("rtx.lens.eye", float, hippus, 1.0f,
                    "Strength of the pupil's unrest, its hippus, which makes the glare pulse, more so in bright light. 0 holds it\n"
                    "steady.",
                    args.minValue = 0.0f, args.maxValue = 4.0f);
    RTX_OPTION_ARGS("rtx.lens.eye", int, lashes, 0,
                    "Number of eyelashes across the top of the pupil, as when squinting into a light, which streak the glare.",
                    args.minValue = 0, args.maxValue = 16);
    RTX_OPTION_ARGS("rtx.lens.eye", float, particleDensity, 1.0f,
                    "Density of the lens and vitreous particles that cast the ciliary corona's needles, relative to a typical eye.",
                    args.minValue = 0.0f, args.maxValue = 8.0f);
    RTX_OPTION_ARGS("rtx.lens.eye", float, haloStrength, 1.0f,
                    "Strength of the lenticular halo, the coloured ring the lens fibres diffract around lights. 1 = typical.",
                    args.minValue = 0.0f, args.maxValue = 10.0f);
    RTX_OPTION_ARGS("rtx.lens.eye", float, updateRateHz, 15.0f,
                    "How often the eye's point spread function is rebuilt to follow its pupil and particles.",
                    args.minValue = 1.0f, args.maxValue = 60.0f);

    explicit EyeModel(DxvkDevice* device);
    ~EyeModel();

    // Advances the pupil and particles. fieldAreaDeg2 is the solid angle of the view in square degrees. Returns whether
    // a new state is ready for the point spread function.
    bool update(float deltaSeconds, float fieldAreaDeg2);

    // Copies the field luminance texel the GPU wrote this frame toward the CPU, which reads it a few frames later.
    void readBackFieldLuminance(DxvkContext* ctx, const Rc<DxvkImage>& image);

    float getPupilDiameterMm() const { return m_publishedPupilMm; }
    float getLashPhase() const { return m_publishedTime; }

    // The particles inside the published pupil, in pupil radii as (x, y, radius, 0), binned in getCellsPerSide()
    // squared cells over the pupil's square.
    const std::vector<Vector4>& getParticles() const { return m_pupilParticles; }
    const std::vector<uint32_t>& getCellRanges() const { return m_cellRanges; }
    const std::vector<uint32_t>& getCellParticles() const { return m_cellParticles; }
    static uint32_t getCellsPerSide();

    // Candela per square metre per unit of radiance.
    static float getLuminanceScale();

    // CIE 135/1-6: the factor its age terms take, and the share of the light it scatters between 0.1 and 90 degrees.
    static float getCieAgeFactor();
    static float getCieScatteredShare();

    // The lenticular halo's share of the light, and its ring's angle and width at 550 nm, in radians.
    static float getHaloShare();
    static float getHaloAngle();
    static float getHaloWidth();

    // Stiles-Crawford coefficient, mm^-2.
    static float getStilesCrawford();

  private:
    struct EyeParticle {
      // In mm from the eye's axis.
      float x;
      float y;
      float radiusMm;
      bool vitreous;
    };

    void generateParticles();
    void binParticles(float pupilRadiusMm, float time);

    DxvkDevice* m_device;

    std::vector<EyeParticle> m_particles;
    float m_generatedDensity = -1.0f;

    std::vector<Vector4> m_pupilParticles;
    std::vector<uint32_t> m_cellRanges;
    std::vector<uint32_t> m_cellParticles;

    static constexpr uint32_t kReadbackRing = 8;
    std::array<Rc<DxvkBuffer>, kReadbackRing> m_readback;
    uint32_t m_readbackFrame = 0;
    float m_fieldLuminance = -1.0f;

    float m_pupilMm = 4.0f;
    float m_time = 0.0f;
    float m_sincePublish = 1e6f;
    float m_publishedPupilMm = 4.0f;
    float m_publishedTime = 0.0f;
  };

}
