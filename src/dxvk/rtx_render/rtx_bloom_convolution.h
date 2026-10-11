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

#include <memory>
#include <vector>

#include "dxvk_include.h"
#include "rtx_resources.h"
#include "rtx_option.h"
#include "rtx_lens_system.h"

// The kernel's and the sun glare's constants, shared with the shaders in rtx/pass/bloom/bloom.h.
struct BloomKernelArgs;
struct BloomSunGlareArgs;

namespace dxvk {

  class DxvkDevice;
  class EyeModel;
  class RtxContext;
  class RtxSunProbe;

  // rtx.bloom.convolutionQuality: the near FFT buffer's size. The coarse buffer is a quarter of it each way.
  enum class BloomConvolutionQuality : int {
    Low = 0,  // 1024x512
    Medium,   // 2048x1024
    High,     // 4096x2048
  };

  enum class BloomConvolutionDebugView : int {
    None = 0,
    BloomOnly,
    Kernel,
    IdentityKernel,
  };

  /**
   * \brief Convolution bloom: the observer's point spread function.
   *
   * Convolves the image with the point spread function of the observer, rtx.lens.observer, through the FFT on a near
   * and a coarse buffer, then adds the glare of the sun's light that its rendered disc's radiance clamp holds back.
   */
  class BloomConvolution {
  public:
    explicit BloomConvolution(DxvkDevice* device);
    ~BloomConvolution();

    // Convolves inOutColorBuffer, linear HDR at output resolution, in place, and adds the sun's glare. pSunProbe may
    // be null.
    void dispatch(Rc<RtxContext> ctx, const Resources::Resource& inOutColorBuffer, const RtxSunProbe* pSunProbe);

    void releaseResources();

    void showImguiSettings();

    // Whether the convolution changes the image.
    static bool hasKernel();

    // Whether the sun's glare needs the sun's visibility.
    static bool wantsSunVisibility();

    RTX_OPTION_ARGS("rtx.bloom", BloomConvolutionQuality, convolutionQuality, BloomConvolutionQuality::Medium,
                    "Size of the convolution bloom's near FFT buffer: 0: Low (1024x512), 1: Medium (2048x1024), 2: High\n"
                    "(4096x2048). Larger buffers resolve finer kernel detail near bright sources. A coarse buffer a quarter of\n"
                    "its size each way carries the kernel's far reaches.",
                    args.minValue = BloomConvolutionQuality::Low, args.maxValue = BloomConvolutionQuality::High);
    RTX_OPTION_ARGS("rtx.bloom", float, convolutionIntensity, 1.0f,
                    "Strength of the convolution bloom, 0 to 1. At 1 all of the image's light passes through the observer's point\n"
                    "spread function, as through a real lens or eye, and lower values blend back toward the image without it. The\n"
                    "kernel has unit energy, so the image keeps its overall brightness.",
                    args.minValue = 0.0f, args.maxValue = 1.0f);
    RTX_OPTION_ARGS("rtx.bloom", float, convolutionPadding, 0.3f,
                    "Fraction of the near FFT buffer kept black around the image, 0 to 0.5, so that its glow does not wrap around\n"
                    "to the opposite edge. More padding widens the near kernel at a lower resolution.",
                    args.minValue = 0.0f, args.maxValue = 0.5f);
    RTX_OPTION("rtx.bloom", bool, convolutionCoarse, true,
               "Carries the kernel's far reaches, beyond the near buffer's window, on a coarse FFT buffer at half padding, so\n"
               "that every bright source spreads its spikes and scatter across the whole frame.");
    RTX_OPTION_ARGS("rtx.bloom", float, maxInputRadiance, 65504.0f,
                    "Radiance the convolution bloom clamps its input to.",
                    args.minValue = 0.0f);
    RTX_OPTION("rtx.bloom", bool, sunGlare, true,
               "Adds the glare of the Physical Atmosphere sun's light that its rendered disc's radiance clamp holds back, at\n"
               "full resolution and with no limit on its reach, on or off the screen, through the sun's ray traced\n"
               "visibility.");
    RTX_OPTION_ARGS("rtx.bloom", float, kernelScale, 1.0f,
                    "Size of the diffraction pattern. 1 = physical, the true size for the camera's f-number and sensor or the\n"
                    "eye's pupil. Larger exaggerates the starburst or corona.",
                    args.minValue = 0.01f, args.maxValue = 8.0f);
    RTX_OPTION_ARGS("rtx.bloom", float, spikeBoost, 1.0f,
                    "Weight of the diffraction pattern outside its central core, the spikes, rings and corona, against the core.\n"
                    "1 = physical.",
                    args.minValue = 0.0f, args.maxValue = 1000.0f);
    RTX_OPTION_ARGS("rtx.bloom", float, spectralDispersion, 1.0f,
                    "How much each wavelength's diffraction pattern scales with the wavelength, which tints the spikes' ends. 1 =\n"
                    "physical, 0 = one pattern for every colour.",
                    args.minValue = 0.0f, args.maxValue = 2.0f);
    RTX_OPTION_ARGS("rtx.bloom", float, scatterScale, 1.0f,
                    "Scale on the observer's scatter: the camera lens's inter-reflection veil, roughness halo and mechanical\n"
                    "scatter, or the eye's disability glare and lenticular halo. 1 = physical.",
                    args.minValue = 0.0f, args.maxValue = 10.0f);
    RTX_OPTION("rtx.bloom", BloomConvolutionDebugView, debugView, BloomConvolutionDebugView::None,
               "Convolution bloom debug view: 0: None, 1: Bloom Only (the light the kernel spreads away from each pixel), 2:\n"
               "Kernel (its near part's weights on a log scale, centred on the screen), 3: Identity Kernel (the image through\n"
               "the FFT buffer with a delta kernel, which must line up with the scene exactly).");

  private:
    // Where the output image sits in an FFT buffer.
    struct BufferFit {
      uint32_t width = 0;
      uint32_t height = 0;
      float texelsPerPixel = 0.0f;
      Vector2 bufferOffset = Vector2(0.0f);
      float screenHeightTexels = 0.0f;
      float aspectRatio = 1.0f;
      // Radius beyond which the kernel's glow would wrap around into the image.
      float windowRadiusTexels = 0.0f;
    };

    // Everything the kernel depends on. Compared bytewise, so it holds no padding.
    struct KernelKey {
      uint32_t width = 0;
      uint32_t height = 0;
      uint32_t coarseWidth = 0;
      uint32_t coarseHeight = 0;
      uint32_t observer = 0;
      uint32_t blades = 0;
      uint32_t identity = 0;
      uint32_t eyeVersion = 0;
      uint32_t lensVersion = 0;
      uint32_t pad = 0;
      float values[20] = {};

      bool operator==(const KernelKey& other) const;
    };

    // The lens's veil and the light its scatter takes, per channel.
    struct LensScatter {
      Vector3 veil = Vector3(0.0f);
      Vector3 roughness = Vector3(0.0f);
      float mechanical = 0.0f;
    };

    static VkExtent2D getBufferExtent(BloomConvolutionQuality quality);
    static BufferFit computeFit(const VkExtent3D& imageExtent, const VkExtent2D& bufferExtent, float padding);
    // The buffer's columns within reach of the image's resampling filters. The image's transform is zero in the others
    // going forward, and the composite reads none of them back.
    static void getImageColumns(const BufferFit& fit, uint32_t& firstColumn, uint32_t& columnCount);
    KernelKey makeKernelKey(const BufferFit& fit, const BufferFit& coarseFit, bool identity) const;

    void createResources(Rc<DxvkContext> ctx, const VkExtent2D& bufferExtent);
    void ensureApertureResources(Rc<DxvkContext> ctx, uint32_t size);
    void ensureSunPsfResources(Rc<DxvkContext> ctx);

    // The camera's lens system, its ghosts and their veil for the current settings and the image's extent at
    // m_kernelTanHalfFovY. Returns whether they changed.
    bool updateLens(float aspectRatio);
    // The stop's radius the f-number opens it to, up to its clear aperture, and the entrance pupil's, in mm.
    void getStop(double& stopRadius, double& pupilRadius) const;
    LensScatter computeLensScatter() const;
    // The veil's density on the sensor against the distance from its source, for the kernel's table.
    void updateVeilTable();

    void fillKernelArgs(BloomKernelArgs& args, const BufferFit& fit, const BufferFit& coarseFit, float tanHalfFovY) const;
    void uploadEyeParticles(Rc<RtxContext>& ctx);

    // Transforms lineCount of target's rows, or columns along Y, from firstLine on.
    void dispatchFft(Rc<RtxContext>& ctx, const Resources::Resource& target, bool alongY, bool forward,
                     uint32_t firstLine = 0, uint32_t lineCount = ~0u);
    // Builds the near and coarse kernels and their centre tap, transformed when transform is set, into the given
    // targets, and the sun's filtered point spread function when buildSunPsf is set.
    void buildKernel(Rc<RtxContext>& ctx, const BufferFit& fit, const BufferFit& coarseFit, float tanHalfFovY,
                     bool identity, bool coarse, const Resources::Resource& nearTarget,
                     const Resources::Resource& coarseTarget, const Resources::Resource& centerTap, bool transform,
                     bool buildSunPsf);
    // The sun's point spread function from the aperture spectrum the last kernel build left, then filtered by its
    // disc, which also sets how far out the glare pass takes the diffraction from it.
    void buildSunDiffraction(Rc<RtxContext>& ctx, const BloomKernelArgs& args);
    void filterSunPsf(Rc<RtxContext>& ctx, const BloomKernelArgs& args, const BufferFit& fit);
    // Refilters the sun's texture for its disc, rebuilding its diffraction only when the disc changes its scale.
    void refreshSunPsf(Rc<RtxContext>& ctx, const BufferFit& fit, const BufferFit& coarseFit, float tanHalfFovY);
    void dispatchSetup(Rc<RtxContext>& ctx, const Resources::Resource& source, const VkExtent2D& sourceSize,
                       const Resources::Resource& target, const BufferFit& fit, float texelsPerSourcePixel,
                       const Vector2& offset);
    void dispatchConvolve(Rc<RtxContext>& ctx, const Resources::Resource& spectrum, const Resources::Resource& kernel,
                          const Resources::Resource& output, const VkExtent2D& extent, bool subtractCenter);
    // Adds the sun's glare as well when pSunGlareProbe is set, from the arguments in m_sunGlareConstants.
    void dispatchComposite(Rc<RtxContext>& ctx, const Resources::Resource& color, const BufferFit& fit,
                           const BufferFit& coarseFit, bool coarse, const Resources::Resource& bloom, uint32_t debugView,
                           float intensity, bool subtractCenter, const RtxSunProbe* pSunGlareProbe = nullptr);
    void dispatchFieldLuminance(Rc<RtxContext>& ctx, const BufferFit& coarseFit, const RtxSunProbe* pSunProbe,
                                float sunLuminance);
    // Fills args with the sun's glare over an image of imageExtent. Returns false when there is no glare to add.
    bool computeSunGlareArgs(Rc<RtxContext>& ctx, const VkExtent3D& imageExtent, const BufferFit& fit,
                             const RtxSunProbe& sunProbe, BloomSunGlareArgs& args);

    DxvkDevice* m_device;

    VkExtent2D m_bufferExtent = { 0, 0 };
    VkExtent2D m_coarseExtent = { 0, 0 };
    // Each buffer's image spectrum, then its convolved image's, and its kernel's spectrum.
    Resources::Resource m_spectrum;
    Resources::Resource m_bloom;
    Resources::Resource m_kernel;
    Resources::Resource m_coarseSpectrum;
    Resources::Resource m_coarseBloom;
    Resources::Resource m_coarseKernel;
    // The aperture and its spectrum, and its normalised power spectrum.
    uint32_t m_apertureSize = 0;
    Resources::Resource m_aperture;
    Resources::Resource m_aperturePower;
    Resources::Resource m_rowSums;
    Resources::Resource m_totals;
    // The cached kernel's centre tap, which the convolution takes out and each pixel keeps.
    Resources::Resource m_centerTap;
    Resources::Resource m_dummy;
    // The sun's point spread function, then filtered by its disc.
    Resources::Resource m_sunPsf;
    Resources::Resource m_sunPsfFiltered;
    Resources::Resource m_fieldLuminance;
    Rc<DxvkBuffer> m_kernelConstants;
    Rc<DxvkBuffer> m_sunGlareConstants;
    Rc<DxvkBuffer> m_particleBuffer;
    Rc<DxvkBuffer> m_cellBuffer;
    Rc<DxvkBuffer> m_cellParticleBuffer;

    KernelKey m_kernelKey;
    bool m_kernelValid = false;
    // The tangent of half the vertical field of view that the kernel and the veil follow, snapped to a grid.
    float m_kernelTanHalfFovY = 0.0f;
    bool m_sunPsfValid = false;
    // What the sun's filtered PSF texture was built for: the disc's radius on the axis and its limb darkening, and the
    // scale its diffraction was built at, 0 when stale.
    float m_sunPsfDiscRadiusPixels = -1.0f;
    Vector3 m_sunLimbDarkening = Vector3(0.0f);
    float m_sunPsfPixelsPerTexel = 0.0f;
    float m_sunPsfRadiusPixels = 0.0f;

    LensSystem m_lensSystem;
    std::vector<LensSystem::Ghost> m_ghosts;
    LensSystem::Veil m_veil;
    std::vector<float> m_veilTable;
    float m_veilStopRadius = -1.0f;
    float m_veilImageHalfHeightMm = -1.0f;
    float m_veilAspectRatio = -1.0f;
    uint32_t m_lensVersion = 0;

    std::unique_ptr<EyeModel> m_eye;
    uint32_t m_eyeVersion = 0;
  };

}
