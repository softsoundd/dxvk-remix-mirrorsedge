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
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>

#include "rtx_bloom_convolution.h"
#include "rtx_context.h"
#include "rtx_eye_model.h"
#include "rtx_imgui.h"
#include "rtx_lens_aperture.h"
#include "rtx_lens_flare.h"
#include "rtx_lens_spectrum.h"
#include "rtx_sun_probe.h"
#include "rtx_utils.h"
#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_render/rtx_shader_manager.h"
#include "rtx/pass/bloom/bloom.h"
#include "../util/util_global_time.h"
#include "../../util/util_color.h"

#include <rtx_shaders/bloom_fft_128.h>
#include <rtx_shaders/bloom_fft_256.h>
#include <rtx_shaders/bloom_fft_512.h>
#include <rtx_shaders/bloom_fft_1024.h>
#include <rtx_shaders/bloom_fft_2048.h>
#include <rtx_shaders/bloom_fft_4096.h>
#include <rtx_shaders/bloom_fft_setup.h>
#include <rtx_shaders/bloom_fft_convolve.h>
#include <rtx_shaders/bloom_fft_composite.h>
#include <rtx_shaders/bloom_field_luminance.h>
#include <rtx_shaders/bloom_kernel_aperture.h>
#include <rtx_shaders/bloom_kernel_power.h>
#include <rtx_shaders/bloom_kernel_build_near.h>
#include <rtx_shaders/bloom_kernel_build_coarse.h>
#include <rtx_shaders/bloom_kernel_build_sun.h>
#include <rtx_shaders/bloom_kernel_reduce_rows.h>
#include <rtx_shaders/bloom_kernel_reduce_total.h>
#include <rtx_shaders/bloom_kernel_combine.h>
#include <rtx_shaders/bloom_sun_psf_filter.h>
#include <rtx_shaders/bloom_sun_glare.h>

namespace dxvk {

  // Defined within an unnamed namespace to ensure unique definition across binary
  namespace {
#define BLOOM_FFT_SHADER(className, shaderName)                                  \
    class className : public ManagedShader {                                     \
      SHADER_SOURCE(className, VK_SHADER_STAGE_COMPUTE_BIT, shaderName)          \
      PUSH_CONSTANTS(BloomFftArgs)                                               \
      BEGIN_PARAMETER()                                                          \
        RW_TEXTURE2D(BLOOM_FFT_SPECTRUM_INPUT_OUTPUT)                            \
      END_PARAMETER()                                                            \
    };                                                                           \
    PREWARM_SHADER_PIPELINE(className);

    BLOOM_FFT_SHADER(BloomFft128Shader, bloom_fft_128)
    BLOOM_FFT_SHADER(BloomFft256Shader, bloom_fft_256)
    BLOOM_FFT_SHADER(BloomFft512Shader, bloom_fft_512)
    BLOOM_FFT_SHADER(BloomFft1024Shader, bloom_fft_1024)
    BLOOM_FFT_SHADER(BloomFft2048Shader, bloom_fft_2048)
    BLOOM_FFT_SHADER(BloomFft4096Shader, bloom_fft_4096)

#undef BLOOM_FFT_SHADER

    class BloomFftSetupShader : public ManagedShader {
      SHADER_SOURCE(BloomFftSetupShader, VK_SHADER_STAGE_COMPUTE_BIT, bloom_fft_setup)

      PUSH_CONSTANTS(BloomFftSetupArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(BLOOM_FFT_SETUP_COLOR_INPUT)
        RW_TEXTURE2D(BLOOM_FFT_SETUP_SPECTRUM_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(BloomFftSetupShader);

    class BloomFftConvolveShader : public ManagedShader {
      SHADER_SOURCE(BloomFftConvolveShader, VK_SHADER_STAGE_COMPUTE_BIT, bloom_fft_convolve)

      PUSH_CONSTANTS(BloomFftConvolveArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(BLOOM_FFT_CONVOLVE_SPECTRUM_INPUT)
        TEXTURE2D(BLOOM_FFT_CONVOLVE_KERNEL_INPUT)
        TEXTURE2D(BLOOM_FFT_CONVOLVE_CENTER_TAP_INPUT)
        RW_TEXTURE2D(BLOOM_FFT_CONVOLVE_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(BloomFftConvolveShader);

    class BloomFftCompositeShader : public ManagedShader {
      SHADER_SOURCE(BloomFftCompositeShader, VK_SHADER_STAGE_COMPUTE_BIT, bloom_fft_composite)

      PUSH_CONSTANTS(BloomFftCompositeArgs)

      BEGIN_PARAMETER()
        SAMPLER2D(BLOOM_FFT_COMPOSITE_BLOOM_INPUT)
        TEXTURE2D(BLOOM_FFT_COMPOSITE_CENTER_TAP_INPUT)
        SAMPLER2D(BLOOM_FFT_COMPOSITE_COARSE_BLOOM_INPUT)
        RW_TEXTURE2D(BLOOM_FFT_COMPOSITE_COLOR_INPUT_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(BloomFftCompositeShader);

    class BloomFieldLuminanceShader : public ManagedShader {
      SHADER_SOURCE(BloomFieldLuminanceShader, VK_SHADER_STAGE_COMPUTE_BIT, bloom_field_luminance)

      PUSH_CONSTANTS(BloomFieldLuminanceArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(BLOOM_FIELD_LUMINANCE_INPUT)
        TEXTURE2D(BLOOM_FIELD_LUMINANCE_SUN_VISIBILITY_INPUT)
        RW_TEXTURE2D(BLOOM_FIELD_LUMINANCE_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(BloomFieldLuminanceShader);

    class BloomKernelApertureShader : public ManagedShader {
      SHADER_SOURCE(BloomKernelApertureShader, VK_SHADER_STAGE_COMPUTE_BIT, bloom_kernel_aperture)

      PUSH_CONSTANTS(BloomKernelApertureArgs)

      BEGIN_PARAMETER()
        STRUCTURED_BUFFER(BLOOM_KERNEL_APERTURE_PARTICLES_INPUT)
        STRUCTURED_BUFFER(BLOOM_KERNEL_APERTURE_CELLS_INPUT)
        STRUCTURED_BUFFER(BLOOM_KERNEL_APERTURE_CELL_PARTICLES_INPUT)
        RW_TEXTURE2D(BLOOM_KERNEL_APERTURE_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(BloomKernelApertureShader);

    class BloomKernelPowerShader : public ManagedShader {
      SHADER_SOURCE(BloomKernelPowerShader, VK_SHADER_STAGE_COMPUTE_BIT, bloom_kernel_power)

      PUSH_CONSTANTS(BloomKernelPowerArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(BLOOM_KERNEL_POWER_INPUT)
        RW_TEXTURE2D(BLOOM_KERNEL_POWER_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(BloomKernelPowerShader);

#define BLOOM_KERNEL_BUILD_SHADER(className, shaderName)                         \
    class className : public ManagedShader {                                     \
      SHADER_SOURCE(className, VK_SHADER_STAGE_COMPUTE_BIT, shaderName)          \
      BEGIN_PARAMETER()                                                          \
        CONSTANT_BUFFER(BLOOM_KERNEL_BUILD_CONSTANTS_INPUT)                      \
        SAMPLER2D(BLOOM_KERNEL_BUILD_APERTURE_POWER_INPUT)                       \
        RW_TEXTURE2D(BLOOM_KERNEL_BUILD_OUTPUT)                                  \
      END_PARAMETER()                                                            \
    };                                                                           \
    PREWARM_SHADER_PIPELINE(className);

    BLOOM_KERNEL_BUILD_SHADER(BloomKernelBuildNearShader, bloom_kernel_build_near)
    BLOOM_KERNEL_BUILD_SHADER(BloomKernelBuildCoarseShader, bloom_kernel_build_coarse)
    BLOOM_KERNEL_BUILD_SHADER(BloomKernelBuildSunShader, bloom_kernel_build_sun)

#undef BLOOM_KERNEL_BUILD_SHADER

    class BloomKernelReduceRowsShader : public ManagedShader {
      SHADER_SOURCE(BloomKernelReduceRowsShader, VK_SHADER_STAGE_COMPUTE_BIT, bloom_kernel_reduce_rows)

      PUSH_CONSTANTS(BloomKernelReduceArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(BLOOM_KERNEL_REDUCE_INPUT)
        RW_TEXTURE2D(BLOOM_KERNEL_REDUCE_ROW_SUMS_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(BloomKernelReduceRowsShader);

    class BloomKernelReduceTotalShader : public ManagedShader {
      SHADER_SOURCE(BloomKernelReduceTotalShader, VK_SHADER_STAGE_COMPUTE_BIT, bloom_kernel_reduce_total)

      PUSH_CONSTANTS(BloomKernelReduceArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(BLOOM_KERNEL_REDUCE_ROW_SUMS_INPUT)
        RW_TEXTURE2D(BLOOM_KERNEL_REDUCE_TOTALS_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(BloomKernelReduceTotalShader);

    class BloomKernelCombineShader : public ManagedShader {
      SHADER_SOURCE(BloomKernelCombineShader, VK_SHADER_STAGE_COMPUTE_BIT, bloom_kernel_combine)

      PUSH_CONSTANTS(BloomKernelCombineArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(BLOOM_KERNEL_COMBINE_TOTALS_INPUT)
        RW_TEXTURE2D(BLOOM_KERNEL_COMBINE_KERNEL_INPUT_OUTPUT)
        RW_TEXTURE2D(BLOOM_KERNEL_COMBINE_COARSE_KERNEL_INPUT_OUTPUT)
        RW_TEXTURE2D(BLOOM_KERNEL_COMBINE_CENTER_TAP_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(BloomKernelCombineShader);

    class BloomSunPsfFilterShader : public ManagedShader {
      SHADER_SOURCE(BloomSunPsfFilterShader, VK_SHADER_STAGE_COMPUTE_BIT, bloom_sun_psf_filter)

      PUSH_CONSTANTS(BloomSunPsfFilterArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(BLOOM_SUN_PSF_FILTER_INPUT)
        RW_TEXTURE2D(BLOOM_SUN_PSF_FILTER_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(BloomSunPsfFilterShader);

    class BloomSunGlareShader : public ManagedShader {
      SHADER_SOURCE(BloomSunGlareShader, VK_SHADER_STAGE_COMPUTE_BIT, bloom_sun_glare)

      PUSH_CONSTANTS(BloomSunGlareArgs)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(BLOOM_SUN_GLARE_CONSTANTS_INPUT)
        SAMPLER2D(BLOOM_SUN_GLARE_PSF_INPUT)
        TEXTURE2D(BLOOM_SUN_GLARE_SUN_VISIBILITY_INPUT)
        RW_TEXTURE2D(BLOOM_SUN_GLARE_COLOR_INPUT_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(BloomSunGlareShader);

    constexpr double kPiDouble = 3.14159265358979323846;

    // The camera's iris spans half of its spectrum's width, its corners kCameraApertureSize / 4 texels out. The eye's
    // pupil spans all of a larger one.
    constexpr uint32_t kCameraApertureSize = 1024;
    constexpr float kCameraApertureRadiusTexels = 0.5f * float(kCameraApertureSize) / float(BLOOM_KERNEL_APERTURE_OVERSAMPLING);
    constexpr uint32_t kEyeApertureSize = 2048;
    constexpr float kEyeApertureRadiusTexels = 0.5f * float(kEyeApertureSize);

    // Where the near field hands over to the far field, as fractions of the spectrum's Nyquist radius: well inside
    // the camera's, where both agree, and far out for the eye, whose particles' corona lives there.
    constexpr float kCameraFarFieldStart = 0.2f;
    constexpr float kCameraFarFieldEnd = 0.45f;
    constexpr float kEyeFarFieldStart = 0.6f;
    constexpr float kEyeFarFieldEnd = 0.9f;

    constexpr float kReferenceWavelengthNm = 550.0f;

    // The diffraction pattern's core ends at its second dark ring, 1.12 wavelengths over the corner radius out.
    constexpr float kCoreRadiusSigmas = 1.12f;

    // The coarse buffer keeps half of itself black, so its window spans the frame.
    constexpr float kCoarsePadding = 0.5f;

    // ABg lobes of the camera's scatter, shoulder in radians and slope. Polished glass scatters about as theta^-2 past a
    // shoulder its longest correlation lengths set (Harvey 1976; Peterson, "Analyzing scattered light", SPIE 1992),
    // while dust, edges and the barrel spread their light far more broadly.
    constexpr float kRoughnessShoulder = 0.003f;
    constexpr float kRoughnessSlope = 2.0f;
    constexpr float kMechanicalShoulder = 0.03f;
    constexpr float kMechanicalSlope = 1.5f;

    // The veil table's radii on the sensor, in mm, and the share of each ghost's radius its edge blurs over either
    // way.
    constexpr float kVeilMinRadiusMm = 0.02f;
    constexpr float kVeilMaxRadiusMm = 200.0f;
    constexpr double kVeilEdgeSoftness = 0.15;

    // Eyelashes, opaque strands about a tenth of a millimetre across where they cross the pupil's light.
    constexpr float kLashWidthMm = 0.1f;

    // The sun's point spread function texture, and the texels its edge keeps clear for the disc.
    constexpr uint32_t kSunPsfSize = 512;
    constexpr float kSunPsfMarginTexels = 16.0f;
    // Disc radii the filter reaches, in texels.
    constexpr float kSunPsfMaxDiscTexels = 40.0f;
    constexpr float kMaxSunGlareRadiance = 60000.0f;

    // The lenses cover the full frame's diagonal, past which their barrels vignette the rays over this many radians.
    constexpr double kImageCircleRadiusMm = 21.633;
    constexpr double kVignettingRange = 0.26;

    Rc<DxvkShader> getFftShader(uint32_t lineLength) {
      switch (lineLength) {
      case 128: return BloomFft128Shader::getShader();
      case 256: return BloomFft256Shader::getShader();
      case 512: return BloomFft512Shader::getShader();
      case 1024: return BloomFft1024Shader::getShader();
      case 2048: return BloomFft2048Shader::getShader();
      case 4096: return BloomFft4096Shader::getShader();
      default:
        assert(!"Unsupported convolution bloom FFT size.");
        return BloomFft2048Shader::getShader();
      }
    }

    // Spectral samples of the pattern, which a white aperture leaves untinted overall. Each scales it by the reference
    // wavelength over its own, with the dispersion scaling the difference.
    void computeSpectralSamples(float dispersion, BloomKernelSpectralSample (&samples)[BLOOM_KERNEL_SPECTRAL_SAMPLES]) {
      float wavelengths[BLOOM_KERNEL_SPECTRAL_SAMPLES];
      Vector3 weights[BLOOM_KERNEL_SPECTRAL_SAMPLES];
      computeVisibleSpectrumSamples(BLOOM_KERNEL_SPECTRAL_SAMPLES, wavelengths, weights);

      for (uint32_t i = 0; i < BLOOM_KERNEL_SPECTRAL_SAMPLES; i++) {
        const float effectiveLambda = kReferenceWavelengthNm + dispersion * (wavelengths[i] - kReferenceWavelengthNm);
        samples[i].weight = weights[i];
        samples[i].scale = kReferenceWavelengthNm / std::max(effectiveLambda, 1.0f);
      }
    }

    // The integral of an ABg lobe 1 / (B^g + theta^g) over the hemisphere, 2 pi sin(theta) dtheta out to 90 degrees,
    // all of which the image plane takes in.
    float abgNorm(float shoulder, float slope) {
      constexpr uint32_t kSteps = 2048;
      const double logMin = std::log(1e-6);
      const double logStep = (std::log(0.5 * kPiDouble) - logMin) / double(kSteps);
      const double shoulderPower = std::pow(double(shoulder), double(slope));
      double sum = 0.0;

      for (uint32_t i = 0; i < kSteps; i++) {
        const double theta = std::exp(logMin + (double(i) + 0.5) * logStep);
        sum += 2.0 * kPiDouble * std::sin(theta) * theta * logStep / (shoulderPower + std::pow(theta, double(slope)));
      }

      return float(sum);
    }

    // The field of view as the kernel follows it, rounded so that the game's zooms do not rebuild it every frame.
    float roundTanHalfFovY(float tanHalfFovY) {
      return std::round(tanHalfFovY * 200.0f) / 200.0f;
    }

    // How far from its source the far field takes over for the reddest wavelength, in output pixels.
    float getNearFieldPixels(const BloomKernelArgs& args, float texelsPerPixel) {
      float smallestScale = 1.0f;

      for (const BloomKernelSpectralSample& sample : args.spectralSamples) {
        smallestScale = std::min(smallestScale, sample.scale);
      }

      const float nyquistTexelsPerSigma = 0.5f * float(args.apertureSize) / args.spectrumTexelsPerSigma;
      return args.farFieldEnd * nyquistTexelsPerSigma * args.sigmaTexels / smallestScale / texelsPerPixel;
    }

    void setComponent(vec4& v, uint32_t index, float value) {
      switch (index) {
      case 0: v.x = value; break;
      case 1: v.y = value; break;
      case 2: v.z = value; break;
      default: v.w = value; break;
      }
    }

    Vector3 clampShare(const Vector3& v) {
      return Vector3(std::clamp(v.x, 0.0f, 1.0f), std::clamp(v.y, 0.0f, 1.0f), std::clamp(v.z, 0.0f, 1.0f));
    }

    // The density at r of a unit energy disc of radius rho whose centre lies d from the origin, averaged around the
    // origin: the share of the circle of radius r inside the disc, over the disc's area.
    double ringDiscDensity(double r, double d, double rho) {
      const double discArea = kPiDouble * rho * rho;

      if (r + d <= rho) {
        return 1.0 / discArea;
      }

      if (r <= d - rho || r >= d + rho) {
        return 0.0;
      }

      const double cosine = std::clamp((r * r + d * d - rho * rho) / (2.0 * r * d), -1.0, 1.0);
      return std::acos(cosine) / kPiDouble / discArea;
    }

    void uploadChunked(Rc<RtxContext>& ctx, const Rc<DxvkBuffer>& buffer, const void* data, size_t size) {
      // vkCmdUpdateBuffer takes at most 64 KB at a time.
      constexpr size_t kChunk = 65536;
      const uint8_t* bytes = static_cast<const uint8_t*>(data);

      for (size_t offset = 0; offset < size; offset += kChunk) {
        ctx->updateBuffer(buffer, offset, std::min(kChunk, size - offset), bytes + offset);
      }
    }

    RemixGui::ComboWithKey<BloomConvolutionQuality> qualityCombo {
      "Quality##convolutionBloom",
      RemixGui::ComboWithKey<BloomConvolutionQuality>::ComboEntries { {
        { BloomConvolutionQuality::Low, "Low (1024x512)" },
        { BloomConvolutionQuality::Medium, "Medium (2048x1024)" },
        { BloomConvolutionQuality::High, "High (4096x2048)" },
      } }
    };

    RemixGui::ComboWithKey<BloomConvolutionDebugView> debugViewCombo {
      "Debug View##convolutionBloom",
      RemixGui::ComboWithKey<BloomConvolutionDebugView>::ComboEntries { {
        { BloomConvolutionDebugView::None, "None" },
        { BloomConvolutionDebugView::BloomOnly, "Bloom Only" },
        { BloomConvolutionDebugView::Kernel, "Kernel" },
        { BloomConvolutionDebugView::IdentityKernel, "Identity Kernel" },
      } }
    };
  }

  bool BloomConvolution::KernelKey::operator==(const KernelKey& other) const {
    return std::memcmp(this, &other, sizeof(KernelKey)) == 0;
  }

  BloomConvolution::BloomConvolution(DxvkDevice* device)
    : m_device(device) {
  }

  BloomConvolution::~BloomConvolution() = default;

  bool BloomConvolution::hasKernel() {
    return debugView() != BloomConvolutionDebugView::None || convolutionIntensity() > 0.0f;
  }

  bool BloomConvolution::wantsSunVisibility() {
    return sunGlare() && convolutionIntensity() > 0.0f && debugView() == BloomConvolutionDebugView::None;
  }

  VkExtent2D BloomConvolution::getBufferExtent(BloomConvolutionQuality quality) {
    switch (quality) {
    case BloomConvolutionQuality::Low: return { 1024, 512 };
    case BloomConvolutionQuality::High: return { 4096, 2048 };
    default:
    case BloomConvolutionQuality::Medium: return { 2048, 1024 };
    }
  }

  BloomConvolution::BufferFit BloomConvolution::computeFit(const VkExtent3D& imageExtent, const VkExtent2D& bufferExtent, float padding) {
    BufferFit fit;
    fit.width = bufferExtent.width;
    fit.height = bufferExtent.height;

    const float imageWidth = float(std::max(imageExtent.width, 1u));
    const float imageHeight = float(std::max(imageExtent.height, 1u));
    const float bufferWidth = float(bufferExtent.width);
    const float bufferHeight = float(bufferExtent.height);

    fit.texelsPerPixel = (1.0f - std::clamp(padding, 0.0f, 0.5f)) * std::min(bufferWidth / imageWidth, bufferHeight / imageHeight);
    fit.bufferOffset = Vector2(0.5f * (bufferWidth - imageWidth * fit.texelsPerPixel),
                               0.5f * (bufferHeight - imageHeight * fit.texelsPerPixel));
    fit.screenHeightTexels = imageHeight * fit.texelsPerPixel;
    fit.aspectRatio = imageWidth / imageHeight;

    // Glow reaching further than the black border on both sides together wraps around into the image.
    const float borderX = bufferWidth - imageWidth * fit.texelsPerPixel;
    const float borderY = bufferHeight - imageHeight * fit.texelsPerPixel;
    fit.windowRadiusTexels = std::max(std::min(borderX, borderY), 2.0f);

    return fit;
  }

  void BloomConvolution::getImageColumns(const BufferFit& fit, uint32_t& firstColumn, uint32_t& columnCount) {
    // Covers the coarse setup's tent and the composite's B-spline either side.
    constexpr float kMarginTexels = 4.0f;
    const float left = std::floor(fit.bufferOffset.x - kMarginTexels);
    const float right = std::ceil(fit.bufferOffset.x + fit.screenHeightTexels * fit.aspectRatio + kMarginTexels);

    firstColumn = uint32_t(std::clamp(left, 0.0f, float(fit.width)));
    columnCount = uint32_t(std::clamp(right, float(firstColumn), float(fit.width))) - firstColumn;
  }

  BloomConvolution::KernelKey BloomConvolution::makeKernelKey(const BufferFit& fit, const BufferFit& coarseFit, float tanHalfFovY, bool identity) const {
    const bool eye = LensAperture::isEye();

    KernelKey key;
    key.width = fit.width;
    key.height = fit.height;
    key.coarseWidth = convolutionCoarse() ? coarseFit.width : 0u;
    key.coarseHeight = convolutionCoarse() ? coarseFit.height : 0u;
    key.observer = eye ? 1u : 0u;
    key.blades = eye ? 0u : LensAperture::getBladeCount();
    key.identity = identity ? 1u : 0u;
    key.eyeVersion = eye ? m_eyeVersion : 0u;
    key.lensVersion = eye ? 0u : m_lensVersion;

    const float values[] = {
      fit.screenHeightTexels,
      fit.windowRadiusTexels,
      fit.aspectRatio,
      coarseFit.windowRadiusTexels,
      coarseFit.texelsPerPixel,
      kernelScale(),
      spikeBoost(),
      spectralDispersion(),
      scatterScale(),
      // The eye's kernel is angular, so it follows the field of view. The camera's follows only the image's extent on the
      // sensor, which the lens version carries and which stops changing once the view is wider than the lens covers.
      eye ? roundTanHalfFovY(tanHalfFovY) : 0.0f,
      eye ? 0.0f : LensAperture::fNumber(),
      eye ? 0.0f : LensAperture::apertureCircularFNumber(),
      eye ? 0.0f : LensAperture::apertureRotation(),
      eye ? 0.0f : LensAperture::surfaceRoughnessNm(),
      eye ? 0.0f : LensAperture::mechanicalScatter(),
      eye ? 0.0f : LensAperture::apertureBladeTolerance(),
      eye ? 0.0f : LensAperture::apertureEdgeRoughness(),
      eye ? EyeModel::age() : 0.0f,
      eye ? EyeModel::pigmentation() : 0.0f,
      eye ? EyeModel::haloStrength() : 0.0f,
    };
    static_assert(std::size(values) == sizeof(KernelKey::values) / sizeof(float));
    std::copy(std::begin(values), std::end(values), std::begin(key.values));

    return key;
  }

  void BloomConvolution::createResources(Rc<DxvkContext> ctx, const VkExtent2D& bufferExtent) {
    releaseResources();

    const VkExtent2D coarseExtent = { bufferExtent.width / 4, bufferExtent.height / 4 };
    const VkExtent3D extent = { bufferExtent.width, bufferExtent.height, 1 };
    const VkExtent3D coarse = { coarseExtent.width, coarseExtent.height, 1 };

    m_spectrum = Resources::createImageResource(ctx, "convolution bloom spectrum", extent, VK_FORMAT_R32G32B32A32_SFLOAT);
    m_bloom = Resources::createImageResource(ctx, "convolution bloom result", extent, VK_FORMAT_R32G32B32A32_SFLOAT);
    m_kernel = Resources::createImageResource(ctx, "convolution bloom kernel spectrum", extent, VK_FORMAT_R32G32B32A32_SFLOAT);
    m_coarseSpectrum = Resources::createImageResource(ctx, "convolution bloom coarse spectrum", coarse, VK_FORMAT_R32G32B32A32_SFLOAT);
    m_coarseBloom = Resources::createImageResource(ctx, "convolution bloom coarse result", coarse, VK_FORMAT_R32G32B32A32_SFLOAT);
    m_coarseKernel = Resources::createImageResource(ctx, "convolution bloom coarse kernel spectrum", coarse, VK_FORMAT_R32G32B32A32_SFLOAT);
    m_rowSums = Resources::createImageResource(ctx, "convolution bloom kernel row sums", { bufferExtent.height, 2, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT);
    m_totals = Resources::createImageResource(ctx, "convolution bloom kernel totals", { 2, 1, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT);
    m_centerTap = Resources::createImageResource(ctx, "convolution bloom kernel centre tap", { 1, 1, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT);
    m_dummy = Resources::createImageResource(ctx, "convolution bloom dummy", { 1, 1, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT);
    m_fieldLuminance = Resources::createImageResource(ctx, "convolution bloom field luminance", { 1, 1, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT);

    DxvkBufferCreateInfo info = {};
    info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    info.access = VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    info.size = sizeof(BloomKernelArgs);
    m_kernelConstants = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Convolution bloom kernel constants");

    m_bufferExtent = bufferExtent;
    m_coarseExtent = coarseExtent;
    m_kernelValid = false;
    m_sunPsfValid = false;
  }

  void BloomConvolution::ensureApertureResources(Rc<DxvkContext> ctx, uint32_t size) {
    if (m_apertureSize == size && m_aperture.isValid()) {
      return;
    }

    m_aperture = Resources::createImageResource(ctx, "convolution bloom aperture spectrum", { size, size, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT);
    m_aperturePower = Resources::createImageResource(ctx, "convolution bloom aperture power", { size, size, 1 }, VK_FORMAT_R32_SFLOAT);
    m_apertureSize = size;
  }

  void BloomConvolution::ensureSunPsfResources(Rc<DxvkContext> ctx) {
    if (m_sunPsf.isValid()) {
      return;
    }

    const VkExtent3D extent = { kSunPsfSize, kSunPsfSize, 1 };
    m_sunPsf = Resources::createImageResource(ctx, "convolution bloom sun psf", extent, VK_FORMAT_R32G32B32A32_SFLOAT);
    m_sunPsfFiltered = Resources::createImageResource(ctx, "convolution bloom sun psf filtered", extent, VK_FORMAT_R32G32B32A32_SFLOAT);
  }

  void BloomConvolution::releaseResources() {
    m_spectrum.reset();
    m_bloom.reset();
    m_kernel.reset();
    m_coarseSpectrum.reset();
    m_coarseBloom.reset();
    m_coarseKernel.reset();
    m_aperture.reset();
    m_aperturePower.reset();
    m_apertureSize = 0;
    m_rowSums.reset();
    m_totals.reset();
    m_centerTap.reset();
    m_dummy.reset();
    m_sunPsf.reset();
    m_sunPsfFiltered.reset();
    m_fieldLuminance.reset();
    m_kernelConstants = nullptr;
    m_particleBuffer = nullptr;
    m_cellBuffer = nullptr;
    m_cellParticleBuffer = nullptr;
    m_bufferExtent = { 0, 0 };
    m_coarseExtent = { 0, 0 };
    m_kernelValid = false;
    m_sunPsfValid = false;
    m_sunPsfPixelsPerTexel = 0.0f;
  }

  void BloomConvolution::getStop(double& stopRadius, double& pupilRadius) const {
    const double focalLength = m_lensSystem.getFocalLength();
    const double requestedPupilRadius = focalLength / (2.0 * std::max(double(LensAperture::fNumber()), 0.5));

    stopRadius = std::min(requestedPupilRadius * m_lensSystem.getStopFromPupil(), m_lensSystem.getMaxStopRadius());
    pupilRadius = std::min(stopRadius / m_lensSystem.getStopFromPupil(), m_lensSystem.getFrontRadius());
  }

  bool BloomConvolution::updateLens(float tanHalfFovY, float aspectRatio) {
    bool changed = LensAperture::updateLensSystem(m_lensSystem);

    if (changed) {
      // The veil takes every ghost, so the bound on |Mf11| only affects their order.
      m_ghosts = m_lensSystem.traceGhosts(1e-3);
    }

    double stopRadius;
    double pupilRadius;
    getStop(stopRadius, pupilRadius);

    // The veil averages its sources over the image's extent on the sensor.
    const float imageHalfHeightMm =
      LensAperture::getImageHalfHeightMm(aspectRatio, float(m_lensSystem.getFocalLength()), roundTanHalfFovY(tanHalfFovY));

    if (changed || m_veilStopRadius != float(stopRadius) || m_veilImageHalfHeightMm != imageHalfHeightMm || m_veilAspectRatio != aspectRatio) {
      m_veil = m_lensSystem.computeVeil(m_ghosts, stopRadius, imageHalfHeightMm, aspectRatio);
      m_veilStopRadius = float(stopRadius);
      m_veilImageHalfHeightMm = imageHalfHeightMm;
      m_veilAspectRatio = aspectRatio;
      updateVeilTable();
      changed = true;
    }

    return changed;
  }

  BloomConvolution::LensScatter BloomConvolution::computeLensScatter() const {
    LensScatter scatter;
    scatter.veil = m_veil.energy;

    // Each polished surface scatters (2 pi sigma dn / lambda)^2 of the light passing it, by its total integrated scatter
    // (Bennett and Porteus 1961), summed over the surfaces at each wavelength sample.
    const double roughnessMm = double(LensAperture::surfaceRoughnessNm()) * 1e-6;
    const auto& wavelengths = m_lensSystem.getWavelengths();
    const auto& weights = m_lensSystem.getWavelengthWeights();

    for (uint32_t w = 0; w < LENS_FLARE_WAVELENGTHS; w++) {
      const double wavelengthMm = wavelengths[w] * 1e-3;
      double total = 0.0;

      for (uint32_t k = 0; k < m_lensSystem.getSurfaceCount(); k++) {
        if (k == m_lensSystem.getStopIndex()) {
          continue;
        }

        const double step = std::abs(m_lensSystem.getIndexAfter(k, w) - m_lensSystem.getIndexBefore(k, w));
        const double phase = 2.0 * kPiDouble * roughnessMm * step / wavelengthMm;
        total += phase * phase;
      }

      scatter.roughness = scatter.roughness + weights[w] * float(total);
    }

    scatter.mechanical = LensAperture::mechanicalScatter();
    return scatter;
  }

  void BloomConvolution::updateVeilTable() {
    const double logStep = std::log(double(kVeilMaxRadiusMm / kVeilMinRadiusMm)) / double(BLOOM_VEIL_TABLE_SIZE - 1);
    m_veilTable.assign(BLOOM_VEIL_TABLE_SIZE, 0.0f);
    double totalEnergy = 0.0;

    for (const Vector3& energy : m_veil.energies) {
      totalEnergy += sRGBLuminance(energy);
    }

    if (totalEnergy > 0.0) {
      // A unit energy mix of every ghost's images, each a disc whose edge the lens's aberrations and the iris's shape
      // soften, at its distance from the source and averaged around it, as a kernel the same for every source must
      // be. The last entry stays zero so the table ramps out.
      const uint32_t imagesPerGhost = m_veil.imagesPerGhost;

      for (uint32_t i = 0; i + 1 < BLOOM_VEIL_TABLE_SIZE; i++) {
        const double radius = kVeilMinRadiusMm * std::exp(logStep * double(i));
        double density = 0.0;

        for (size_t g = 0; g < m_veil.energies.size(); g++) {
          const double energy = sRGBLuminance(m_veil.energies[g]);

          if (energy <= 0.0) {
            continue;
          }

          for (uint32_t k = 0; k < imagesPerGhost; k++) {
            const LensSystem::VeilImage& image = m_veil.images[g * imagesPerGhost + k];

            if (image.share <= 0.0f) {
              continue;
            }

            const double rho = image.radius;
            const double d = image.offset;
            const double disc =
              0.25 * ringDiscDensity(radius, d, rho * (1.0 - kVeilEdgeSoftness)) +
              0.5 * ringDiscDensity(radius, d, rho) +
              0.25 * ringDiscDensity(radius, d, rho * (1.0 + kVeilEdgeSoftness));

            density += energy * double(image.share) * disc;
          }
        }

        m_veilTable[i] = float(density / totalEnergy);
      }
    }
  }

  void BloomConvolution::fillKernelArgs(BloomKernelArgs& args, const BufferFit& fit, const BufferFit& coarseFit,
                                        float tanHalfFovY) const {
    const bool eye = LensAperture::isEye();
    const bool coarse = convolutionCoarse();
    const float scale = scatterScale();

    args.nearBufferSize = { fit.width, fit.height };
    args.coarseBufferSize = { coarseFit.width, coarseFit.height };
    args.observer = eye ? BLOOM_OBSERVER_EYE : BLOOM_OBSERVER_CAMERA;
    args.nearWindowRadius = fit.windowRadiusTexels;
    args.nearTexelsPerCoarseTexel = fit.texelsPerPixel / coarseFit.texelsPerPixel;
    args.coarseWindowRadius = coarse ? coarseFit.windowRadiusTexels * args.nearTexelsPerCoarseTexel : fit.windowRadiusTexels;
    args.nearTexelsPerPixel = fit.texelsPerPixel;
    args.spikeBoost = spikeBoost();
    computeSpectralSamples(spectralDispersion(), args.spectralSamples);

    if (eye) {
      // The pupil's diffraction scales as the wavelength over its radius, in the game's own angles.
      const float pupilRadiusMm = 0.5f * m_eye->getPupilDiameterMm();
      const float radiansPerTexel = 2.0f * tanHalfFovY / std::max(fit.screenHeightTexels, 1.0f);
      const float wavelengthMm = kReferenceWavelengthNm * 1e-6f;
      const float stilesCrawford = EyeModel::getStilesCrawford();
      const double apodisation = double(stilesCrawford) * pupilRadiusMm * pupilRadiusMm;

      args.apertureSize = kEyeApertureSize;
      args.spectrumTexelsPerSigma = 2.0f;
      args.farFieldStart = kEyeFarFieldStart;
      args.farFieldEnd = kEyeFarFieldEnd;
      args.sigmaTexels = wavelengthMm / pupilRadiusMm / radiansPerTexel * kernelScale();
      args.apertureBlades = 0;
      args.apertureStreakContrast = 0.0f;
      args.apertureArea = float(kPiDouble);
      // The apodised pupil's power is the integral of 10^(-rho r^2) over its disc.
      const double apodisedArea = apodisation > 1e-6
        ? kPiDouble * (1.0 - std::pow(10.0, -apodisation)) / (apodisation * std::log(10.0))
        : kPiDouble;
      args.apertureEnergy = float(apodisedArea * double(kEyeApertureRadiusTexels) * double(kEyeApertureRadiusTexels));

      const float cieShare = EyeModel::getCieScatteredShare();
      args.physicalPerTexel = radiansPerTexel;
      args.focalLengthMm = 0.0f;
      args.scatterShare0 = clampShare(Vector3(cieShare * scale));
      args.scatterShare1 = clampShare(Vector3(EyeModel::getHaloShare() * scale));
      args.scatterShare2 = Vector3(0.0f);
      args.cieAgeFactor = EyeModel::getCieAgeFactor();
      args.ciePigmentation = EyeModel::pigmentation();
      args.cieNorm = std::max(cieShare, 1e-6f);
      args.haloAngle = EyeModel::getHaloAngle();
      args.haloWidth = EyeModel::getHaloWidth();
    } else {
      // The stop opens to the f-number up to its clear aperture, as the lens flare's.
      const double focalLength = m_lensSystem.getFocalLength();
      double stopRadius;
      double pupilRadius;
      getStop(stopRadius, pupilRadius);
      const float fNumber = float(focalLength / (2.0 * pupilRadius));
      const float curvature = LensAperture::getCurvature(fNumber);
      const LensAperture::Shape shape = LensAperture::getShape(curvature);
      float distanceJitter;
      float angleJitter;
      LensAperture::getJitter(stopRadius, m_lensSystem.getMaxStopRadius(), distanceJitter, angleJitter);
      const LensAperture::Sides sides = LensAperture::getSides(curvature, distanceJitter, angleJitter);
      const float sensorHalfHeightMm = LensAperture::getImageHalfHeightMm(fit.aspectRatio, float(focalLength), tanHalfFovY);
      const float wavelengthMm = kReferenceWavelengthNm * 1e-6f;

      args.apertureSize = kCameraApertureSize;
      args.spectrumTexelsPerSigma = 2.0f * float(BLOOM_KERNEL_APERTURE_OVERSAMPLING);
      args.farFieldStart = kCameraFarFieldStart;
      args.farFieldEnd = kCameraFarFieldEnd;
      // The pattern's angular scale is the wavelength over the iris's corner radius, f / (2 N) over its area radius. A
      // texel subtends its share of the sensor's height over f, so f drops out:
      // sigma = lambda N areaRadius / (sensor height per texel).
      args.sigmaTexels = wavelengthMm * fNumber * shape.areaRadius * fit.screenHeightTexels / sensorHalfHeightMm * kernelScale();
      args.apertureBlades = static_cast<uint32_t>(std::min<size_t>(sides.sides.size(), BLOOM_APERTURE_MAX_SIDES));
      args.apertureStreakContrast = LensAperture::apertureEdgeRoughness();

      for (uint32_t i = 0; i < args.apertureBlades; i++) {
        const LensAperture::Side& side = sides.sides[i];
        args.apertureSides[i] = Vector4(side.centerAngle, side.halfRange, side.length, 0.0f);

        for (uint32_t w = 0; w < BLOOM_STREAK_WAVES / 2; w++) {
          vec4& waves = args.apertureStreakWaves[i * (BLOOM_STREAK_WAVES / 2) + w];
          LensAperture::getStreakWave(i, 2 * w, BLOOM_STREAK_SHORTEST_PERIOD_DEGREES, BLOOM_STREAK_LONGEST_PERIOD_DEGREES, waves.x, waves.y);
          LensAperture::getStreakWave(i, 2 * w + 1, BLOOM_STREAK_SHORTEST_PERIOD_DEGREES, BLOOM_STREAK_LONGEST_PERIOD_DEGREES, waves.z, waves.w);
        }
      }

      args.apertureArea = args.apertureBlades > 0 ? sides.area : shape.area;
      args.apertureEnergy = args.apertureArea * kCameraApertureRadiusTexels * kCameraApertureRadiusTexels;

      const LensScatter scatter = computeLensScatter();
      args.physicalPerTexel = 2.0f * sensorHalfHeightMm / std::max(fit.screenHeightTexels, 1.0f);
      args.focalLengthMm = float(focalLength);
      args.scatterShare0 = clampShare(scatter.veil * scale);
      args.scatterShare1 = clampShare(scatter.roughness * scale);
      args.scatterShare2 = clampShare(Vector3(scatter.mechanical * scale));
      static const float kRoughnessNorm = abgNorm(kRoughnessShoulder, kRoughnessSlope);
      static const float kMechanicalNorm = abgNorm(kMechanicalShoulder, kMechanicalSlope);
      args.roughnessShoulder = kRoughnessShoulder;
      args.roughnessSlope = kRoughnessSlope;
      args.roughnessNorm = kRoughnessNorm;
      args.mechanicalShoulder = kMechanicalShoulder;
      args.mechanicalSlope = kMechanicalSlope;
      args.mechanicalNorm = kMechanicalNorm;
      args.veilMinRadiusMm = kVeilMinRadiusMm;
      args.veilLogStep = std::log(kVeilMaxRadiusMm / kVeilMinRadiusMm) / float(BLOOM_VEIL_TABLE_SIZE - 1);

      for (uint32_t i = 0; i < BLOOM_VEIL_TABLE_SIZE && i < m_veilTable.size(); i++) {
        setComponent(args.veilTable[i / 4], i % 4, m_veilTable[i]);
      }
    }

    args.coreRadiusTexels = std::max(kCoreRadiusSigmas * args.sigmaTexels, 1.0f);

    // The sun's texture reaches past where the far field takes over, and past the disc.
    const float nearFieldPixels = getNearFieldPixels(args, fit.texelsPerPixel);
    const float discRadiusPixels = std::max(m_sunPsfDiscRadiusPixels, 0.0f);
    const float reachTexels = 0.5f * float(kSunPsfSize) - kSunPsfMarginTexels;

    args.sunPsfSize = kSunPsfSize;
    args.sunPsfPixelsPerTexel = std::max({ 1.0f, (nearFieldPixels + discRadiusPixels) / reachTexels, discRadiusPixels / kSunPsfMaxDiscTexels });
  }

  void BloomConvolution::uploadEyeParticles(Rc<RtxContext>& ctx) {
    static const Vector4 kNoParticle(0.0f);
    static const uint32_t kNoCell[2] = { 0, 0 };
    static const uint32_t kNoIndex = 0;

    const bool eye = LensAperture::isEye() && m_eye != nullptr;
    const void* particles = eye ? static_cast<const void*>(m_eye->getParticles().data()) : &kNoParticle;
    const size_t particleBytes = eye ? m_eye->getParticles().size() * sizeof(Vector4) : sizeof(Vector4);
    const void* cells = eye ? static_cast<const void*>(m_eye->getCellRanges().data()) : kNoCell;
    const size_t cellBytes = eye ? m_eye->getCellRanges().size() * sizeof(uint32_t) : sizeof(kNoCell);
    const void* indices = eye ? static_cast<const void*>(m_eye->getCellParticles().data()) : &kNoIndex;
    const size_t indexBytes = eye ? m_eye->getCellParticles().size() * sizeof(uint32_t) : sizeof(uint32_t);

    const auto ensure = [&](Rc<DxvkBuffer>& buffer, size_t size, const char* name) {
      if (buffer != nullptr && buffer->info().size >= size) {
        return;
      }

      DxvkBufferCreateInfo info = {};
      info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
      info.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      // The buffer grows in steps, so the particles' count drifting does not reallocate it every rebuild.
      info.size = std::max<VkDeviceSize>(align(size * 3 / 2, 256), 256);
      buffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, name);
    };

    ensure(m_particleBuffer, particleBytes, "Eye particles");
    ensure(m_cellBuffer, cellBytes, "Eye particle cells");
    ensure(m_cellParticleBuffer, indexBytes, "Eye particle cell indices");

    uploadChunked(ctx, m_particleBuffer, particles, particleBytes);
    uploadChunked(ctx, m_cellBuffer, cells, cellBytes);
    uploadChunked(ctx, m_cellParticleBuffer, indices, indexBytes);
  }

  void BloomConvolution::dispatchFft(Rc<RtxContext>& ctx, const Resources::Resource& target, bool alongY, bool forward,
                                     uint32_t firstLine, uint32_t lineCount) {
    const VkExtent3D extent = target.image->info().extent;
    const uint32_t lineLength = alongY ? extent.height : extent.width;
    const uint32_t lines = alongY ? extent.width : extent.height;

    if (firstLine >= lines) {
      return;
    }

    BloomFftArgs args = {};
    args.alongY = alongY ? 1u : 0u;
    args.forward = forward ? 1u : 0u;
    args.firstLine = firstLine;
    ctx->pushConstants(0, sizeof(args), &args);

    ctx->bindResourceView(BLOOM_FFT_SPECTRUM_INPUT_OUTPUT, target.view, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, getFftShader(lineLength));
    ctx->dispatch(std::min(lineCount, lines - firstLine), 1, 1);
  }

  void BloomConvolution::buildKernel(Rc<RtxContext>& ctx, const BufferFit& fit, const BufferFit& coarseFit, float tanHalfFovY,
                                     bool identity, bool coarse, const Resources::Resource& nearTarget,
                                     const Resources::Resource& coarseTarget, const Resources::Resource& centerTap, bool transform,
                                     bool buildSunPsf) {
    ScopedGpuProfileZone(ctx, "Convolution Bloom Kernel");

    const VkExtent3D nearWorkgroups = util::computeBlockCount(VkExtent3D { fit.width, fit.height, 1 }, VkExtent3D { 16, 16, 1 });
    const VkExtent3D coarseWorkgroups = util::computeBlockCount(VkExtent3D { coarseFit.width, coarseFit.height, 1 }, VkExtent3D { 16, 16, 1 });
    const bool eye = LensAperture::isEye();
    Rc<DxvkContext> baseCtx = ctx;

    if (!identity) {
      const uint32_t apertureSize = eye ? kEyeApertureSize : kCameraApertureSize;
      ensureApertureResources(baseCtx, apertureSize);
      uploadEyeParticles(ctx);

      // Rasterises the observer's aperture and transforms it into its spectrum.
      BloomKernelApertureArgs apertureArgs = {};
      apertureArgs.size = apertureSize;
      apertureArgs.observer = eye ? BLOOM_OBSERVER_EYE : BLOOM_OBSERVER_CAMERA;

      if (eye) {
        const float pupilRadiusMm = 0.5f * m_eye->getPupilDiameterMm();
        apertureArgs.radiusTexels = kEyeApertureRadiusTexels;
        apertureArgs.pupilRadiusMm = pupilRadiusMm;
        apertureArgs.stilesCrawford = EyeModel::getStilesCrawford();
        apertureArgs.lashCount = uint32_t(std::max(EyeModel::lashes(), 0));
        apertureArgs.lashWidth = kLashWidthMm / std::max(pupilRadiusMm, 0.5f);
        apertureArgs.cellsPerSide = EyeModel::getCellsPerSide();
        apertureArgs.lashPhase = m_eye->getLashPhase();
      } else {
        double stopRadius;
        double pupilRadius;
        getStop(stopRadius, pupilRadius);
        const float curvature = LensAperture::getCurvature(float(m_lensSystem.getFocalLength() / (2.0 * pupilRadius)));
        const LensAperture::Shape shape = LensAperture::getShape(curvature);

        apertureArgs.radiusTexels = kCameraApertureRadiusTexels;
        apertureArgs.blades = shape.circular ? 0u : LensAperture::getBladeCount();
        apertureArgs.curvature = curvature;
        apertureArgs.rotation = LensAperture::getRotationRadians();
        apertureArgs.cellsPerSide = 1;
        LensAperture::getJitter(stopRadius, m_lensSystem.getMaxStopRadius(), apertureArgs.distanceJitter, apertureArgs.angleJitter);
      }

      ctx->pushConstants(0, sizeof(apertureArgs), &apertureArgs);
      ctx->bindResourceBuffer(BLOOM_KERNEL_APERTURE_PARTICLES_INPUT, DxvkBufferSlice(m_particleBuffer));
      ctx->bindResourceBuffer(BLOOM_KERNEL_APERTURE_CELLS_INPUT, DxvkBufferSlice(m_cellBuffer));
      ctx->bindResourceBuffer(BLOOM_KERNEL_APERTURE_CELL_PARTICLES_INPUT, DxvkBufferSlice(m_cellParticleBuffer));
      ctx->bindResourceView(BLOOM_KERNEL_APERTURE_OUTPUT, m_aperture.view, nullptr);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, BloomKernelApertureShader::getShader());

      const VkExtent3D apertureWorkgroups = util::computeBlockCount(VkExtent3D { apertureSize, apertureSize, 1 }, VkExtent3D { 16, 16, 1 });
      ctx->dispatch(apertureWorkgroups.width, apertureWorkgroups.height, apertureWorkgroups.depth);

      dispatchFft(ctx, m_aperture, false, true);
      dispatchFft(ctx, m_aperture, true, true);

      BloomKernelArgs args = {};
      fillKernelArgs(args, fit, coarseFit, tanHalfFovY);

      BloomKernelPowerArgs powerArgs = {};
      powerArgs.size = apertureSize;
      powerArgs.apertureEnergy = args.apertureEnergy;
      ctx->pushConstants(0, sizeof(powerArgs), &powerArgs);
      ctx->bindResourceView(BLOOM_KERNEL_POWER_INPUT, m_aperture.view, nullptr);
      ctx->bindResourceView(BLOOM_KERNEL_POWER_OUTPUT, m_aperturePower.view, nullptr);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, BloomKernelPowerShader::getShader());
      ctx->dispatch(apertureWorkgroups.width, apertureWorkgroups.height, apertureWorkgroups.depth);

      ctx->updateBuffer(m_kernelConstants, 0, sizeof(args), &args);

      const Rc<DxvkSampler> wrapSampler = ctx->getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT);
      const auto bindBuild = [&](const Resources::Resource& output) {
        ctx->bindResourceBuffer(BLOOM_KERNEL_BUILD_CONSTANTS_INPUT, DxvkBufferSlice(m_kernelConstants, 0, m_kernelConstants->info().size));
        ctx->bindResourceView(BLOOM_KERNEL_BUILD_APERTURE_POWER_INPUT, m_aperturePower.view, nullptr);
        ctx->bindResourceSampler(BLOOM_KERNEL_BUILD_APERTURE_POWER_INPUT, wrapSampler);
        ctx->bindResourceView(BLOOM_KERNEL_BUILD_OUTPUT, output.view, nullptr);
      };

      bindBuild(nearTarget);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, BloomKernelBuildNearShader::getShader());
      ctx->dispatch(nearWorkgroups.width, nearWorkgroups.height, nearWorkgroups.depth);

      if (coarse) {
        bindBuild(coarseTarget);
        ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, BloomKernelBuildCoarseShader::getShader());
        ctx->dispatch(coarseWorkgroups.width, coarseWorkgroups.height, coarseWorkgroups.depth);
      }

      if (buildSunPsf) {
        buildSunDiffraction(ctx, args);
        filterSunPsf(ctx, args, fit);
      } else {
        m_sunPsfPixelsPerTexel = 0.0f;
      }

      // Sums each of the two parts.
      BloomKernelReduceArgs reduceArgs = {};
      reduceArgs.bufferSize = { fit.width, fit.height };
      reduceArgs.part = 0;
      ctx->pushConstants(0, sizeof(reduceArgs), &reduceArgs);
      ctx->bindResourceView(BLOOM_KERNEL_REDUCE_INPUT, nearTarget.view, nullptr);
      ctx->bindResourceView(BLOOM_KERNEL_REDUCE_ROW_SUMS_OUTPUT, m_rowSums.view, nullptr);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, BloomKernelReduceRowsShader::getShader());
      ctx->dispatch(fit.height, 1, 1);

      if (coarse) {
        reduceArgs.bufferSize = { coarseFit.width, coarseFit.height };
        reduceArgs.part = 1;
        ctx->pushConstants(0, sizeof(reduceArgs), &reduceArgs);
        ctx->bindResourceView(BLOOM_KERNEL_REDUCE_INPUT, coarseTarget.view, nullptr);
        ctx->bindResourceView(BLOOM_KERNEL_REDUCE_ROW_SUMS_OUTPUT, m_rowSums.view, nullptr);
        ctx->dispatch(coarseFit.height, 1, 1);
      }

      reduceArgs.bufferSize = { fit.height, coarse ? coarseFit.height : 0u };
      reduceArgs.part = 0;
      ctx->pushConstants(0, sizeof(reduceArgs), &reduceArgs);
      ctx->bindResourceView(BLOOM_KERNEL_REDUCE_ROW_SUMS_INPUT, m_rowSums.view, nullptr);
      ctx->bindResourceView(BLOOM_KERNEL_REDUCE_TOTALS_OUTPUT, m_totals.view, nullptr);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, BloomKernelReduceTotalShader::getShader());
      ctx->dispatch(1, 1, 1);
    }

    BloomKernelCombineArgs combineArgs = {};
    combineArgs.bufferSize = { fit.width, fit.height };
    combineArgs.coarseBufferSize = { coarseFit.width, coarseFit.height };
    combineArgs.identity = identity ? 1u : 0u;
    combineArgs.coarseEnabled = coarse ? 1u : 0u;
    ctx->pushConstants(0, sizeof(combineArgs), &combineArgs);

    ctx->bindResourceView(BLOOM_KERNEL_COMBINE_TOTALS_INPUT, m_totals.view, nullptr);
    ctx->bindResourceView(BLOOM_KERNEL_COMBINE_KERNEL_INPUT_OUTPUT, nearTarget.view, nullptr);
    ctx->bindResourceView(BLOOM_KERNEL_COMBINE_COARSE_KERNEL_INPUT_OUTPUT, coarseTarget.view, nullptr);
    ctx->bindResourceView(BLOOM_KERNEL_COMBINE_CENTER_TAP_OUTPUT, centerTap.view, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, BloomKernelCombineShader::getShader());
    ctx->dispatch(nearWorkgroups.width, nearWorkgroups.height, nearWorkgroups.depth);

    if (transform) {
      dispatchFft(ctx, nearTarget, false, true);
      dispatchFft(ctx, nearTarget, true, true);

      if (coarse) {
        dispatchFft(ctx, coarseTarget, false, true);
        dispatchFft(ctx, coarseTarget, true, true);
      }
    }
  }

  void BloomConvolution::buildSunDiffraction(Rc<RtxContext>& ctx, const BloomKernelArgs& args) {
    Rc<DxvkContext> baseCtx = ctx;
    ensureSunPsfResources(baseCtx);

    const Rc<DxvkSampler> wrapSampler = ctx->getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    ctx->bindResourceBuffer(BLOOM_KERNEL_BUILD_CONSTANTS_INPUT, DxvkBufferSlice(m_kernelConstants, 0, m_kernelConstants->info().size));
    ctx->bindResourceView(BLOOM_KERNEL_BUILD_APERTURE_POWER_INPUT, m_aperturePower.view, nullptr);
    ctx->bindResourceSampler(BLOOM_KERNEL_BUILD_APERTURE_POWER_INPUT, wrapSampler);
    ctx->bindResourceView(BLOOM_KERNEL_BUILD_OUTPUT, m_sunPsf.view, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, BloomKernelBuildSunShader::getShader());

    const VkExtent3D workgroups = util::computeBlockCount(VkExtent3D { kSunPsfSize, kSunPsfSize, 1 }, VkExtent3D { 16, 16, 1 });
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);

    m_sunPsfPixelsPerTexel = args.sunPsfPixelsPerTexel;
  }

  void BloomConvolution::filterSunPsf(Rc<RtxContext>& ctx, const BloomKernelArgs& args, const BufferFit& fit) {
    BloomSunPsfFilterArgs filterArgs = {};
    filterArgs.size = kSunPsfSize;
    filterArgs.discRadiusTexels = std::max(m_sunPsfDiscRadiusPixels, 0.0f) / args.sunPsfPixelsPerTexel;
    filterArgs.limbDarkeningExponent = m_sunLimbDarkening;
    ctx->pushConstants(0, sizeof(filterArgs), &filterArgs);
    ctx->bindResourceView(BLOOM_SUN_PSF_FILTER_INPUT, m_sunPsf.view, nullptr);
    ctx->bindResourceView(BLOOM_SUN_PSF_FILTER_OUTPUT, m_sunPsfFiltered.view, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, BloomSunPsfFilterShader::getShader());

    const VkExtent3D workgroups = util::computeBlockCount(VkExtent3D { kSunPsfSize, kSunPsfSize, 1 }, VkExtent3D { 16, 16, 1 });
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);

    // The glare pass reads the texture out to where the far field takes over, within the texture's reach around the disc.
    const float nearFieldPixels = getNearFieldPixels(args, fit.texelsPerPixel);
    const float reachPixels = (0.5f * float(kSunPsfSize) - kSunPsfMarginTexels) * args.sunPsfPixelsPerTexel - m_sunPsfDiscRadiusPixels;
    m_sunPsfRadiusPixels = std::min(std::max(nearFieldPixels, 3.0f * m_sunPsfDiscRadiusPixels), reachPixels);
  }

  void BloomConvolution::refreshSunPsf(Rc<RtxContext>& ctx, const BufferFit& fit, const BufferFit& coarseFit, float tanHalfFovY) {
    ScopedGpuProfileZone(ctx, "Convolution Bloom Sun PSF");

    BloomKernelArgs args = {};
    fillKernelArgs(args, fit, coarseFit, tanHalfFovY);

    // The texture's scale follows the disc only once the disc grows large.
    if (args.sunPsfPixelsPerTexel != m_sunPsfPixelsPerTexel) {
      ctx->updateBuffer(m_kernelConstants, 0, sizeof(args), &args);
      buildSunDiffraction(ctx, args);
    }

    filterSunPsf(ctx, args, fit);
  }

  void BloomConvolution::dispatchSetup(Rc<RtxContext>& ctx, const Resources::Resource& source, const VkExtent2D& sourceSize,
                                       const Resources::Resource& target, const BufferFit& fit, float texelsPerSourcePixel,
                                       const Vector2& offset) {
    BloomFftSetupArgs args = {};
    args.bufferSize = { fit.width, fit.height };
    args.imageSize = { sourceSize.width, sourceSize.height };
    args.texelsPerPixel = texelsPerSourcePixel;
    args.maxInputRadiance = maxInputRadiance();
    args.bufferOffset = offset;
    ctx->pushConstants(0, sizeof(args), &args);

    ctx->bindResourceView(BLOOM_FFT_SETUP_COLOR_INPUT, source.view, nullptr);
    ctx->bindResourceView(BLOOM_FFT_SETUP_SPECTRUM_OUTPUT, target.view, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, BloomFftSetupShader::getShader());

    const VkExtent3D workgroups = util::computeBlockCount(VkExtent3D { fit.width, fit.height, 1 }, VkExtent3D { 16, 16, 1 });
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void BloomConvolution::dispatchConvolve(Rc<RtxContext>& ctx, const Resources::Resource& spectrum, const Resources::Resource& kernel,
                                          const Resources::Resource& output, const VkExtent2D& extent, bool subtractCenter) {
    BloomFftConvolveArgs args = {};
    args.bufferSize = { extent.width, extent.height };
    args.subtractCenter = subtractCenter ? 1u : 0u;
    ctx->pushConstants(0, sizeof(args), &args);

    ctx->bindResourceView(BLOOM_FFT_CONVOLVE_SPECTRUM_INPUT, spectrum.view, nullptr);
    ctx->bindResourceView(BLOOM_FFT_CONVOLVE_KERNEL_INPUT, kernel.view, nullptr);
    ctx->bindResourceView(BLOOM_FFT_CONVOLVE_CENTER_TAP_INPUT, m_centerTap.view, nullptr);
    ctx->bindResourceView(BLOOM_FFT_CONVOLVE_OUTPUT, output.view, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, BloomFftConvolveShader::getShader());

    // Rows 0 to half the height, inclusive. Each thread also writes its mirror row.
    const VkExtent3D workgroups = util::computeBlockCount(VkExtent3D { extent.width, extent.height / 2 + 1, 1 }, VkExtent3D { 32, 4, 1 });
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void BloomConvolution::dispatchComposite(Rc<RtxContext>& ctx, const Resources::Resource& color, const BufferFit& fit,
                                           const BufferFit& coarseFit, bool coarse, const Resources::Resource& bloom,
                                           uint32_t debugView, float intensity, bool subtractCenter) {
    const VkExtent3D imageExtent = color.image->info().extent;

    BloomFftCompositeArgs args = {};
    args.imageSize = { imageExtent.width, imageExtent.height };
    args.bufferSize = { fit.width, fit.height };
    args.bufferOffset = fit.bufferOffset;
    args.texelsPerPixel = fit.texelsPerPixel;
    args.intensity = intensity;
    args.coarseBufferSize = { coarseFit.width, coarseFit.height };
    args.coarseBufferOffset = coarseFit.bufferOffset;
    args.coarseTexelsPerPixel = coarseFit.texelsPerPixel;
    args.coarseEnabled = coarse ? 1u : 0u;
    args.debugView = debugView;
    args.subtractCenter = subtractCenter ? 1u : 0u;
    ctx->pushConstants(0, sizeof(args), &args);

    const Rc<DxvkSampler> linearSampler = ctx->getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    ctx->bindResourceView(BLOOM_FFT_COMPOSITE_BLOOM_INPUT, bloom.view, nullptr);
    ctx->bindResourceSampler(BLOOM_FFT_COMPOSITE_BLOOM_INPUT, linearSampler);
    ctx->bindResourceView(BLOOM_FFT_COMPOSITE_CENTER_TAP_INPUT, m_centerTap.view, nullptr);
    ctx->bindResourceView(BLOOM_FFT_COMPOSITE_COARSE_BLOOM_INPUT, m_coarseBloom.view, nullptr);
    ctx->bindResourceSampler(BLOOM_FFT_COMPOSITE_COARSE_BLOOM_INPUT, linearSampler);
    ctx->bindResourceView(BLOOM_FFT_COMPOSITE_COLOR_INPUT_OUTPUT, color.view, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, BloomFftCompositeShader::getShader());

    const VkExtent3D workgroups = util::computeBlockCount(imageExtent, VkExtent3D { 16, 16, 1 });
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void BloomConvolution::dispatchFieldLuminance(Rc<RtxContext>& ctx, const BufferFit& coarseFit, const RtxSunProbe* pSunProbe,
                                                float sunLuminance) {
    const Resources::Resource& source = convolutionCoarse() ? m_coarseSpectrum : m_spectrum;
    const VkExtent3D sourceExtent = source.image->info().extent;
    const Rc<DxvkImageView> visibility = pSunProbe != nullptr && pSunProbe->getVisibilityView() != nullptr
      ? pSunProbe->getVisibilityView()
      : m_dummy.view;

    BloomFieldLuminanceArgs args = {};
    args.bufferSize = { sourceExtent.width, sourceExtent.height };
    args.imageMin = coarseFit.bufferOffset;
    args.imageMax = coarseFit.bufferOffset + Vector2(coarseFit.screenHeightTexels * coarseFit.aspectRatio, coarseFit.screenHeightTexels);
    args.sunEnabled = sunLuminance > 0.0f ? 1u : 0u;
    args.sunLuminance = sunLuminance;
    ctx->pushConstants(0, sizeof(args), &args);

    ctx->bindResourceView(BLOOM_FIELD_LUMINANCE_INPUT, source.view, nullptr);
    ctx->bindResourceView(BLOOM_FIELD_LUMINANCE_SUN_VISIBILITY_INPUT, visibility, nullptr);
    ctx->bindResourceView(BLOOM_FIELD_LUMINANCE_OUTPUT, m_fieldLuminance.view, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, BloomFieldLuminanceShader::getShader());
    ctx->dispatch(1, 1, 1);

    m_eye->readBackFieldLuminance(ctx.ptr(), m_fieldLuminance.image);
  }

  void BloomConvolution::dispatchSunGlare(Rc<RtxContext>& ctx, const Resources::Resource& color, const BufferFit& fit,
                                          const RtxSunProbe& sunProbe) {
    ScopedGpuProfileZone(ctx, "Sun Glare");

    const RtxSunProbe::State& sun = sunProbe.getState();
    const VkExtent3D imageExtent = color.image->info().extent;
    const Vector3 restored = sunProbe.getRestoredIlluminance();

    if (sRGBLuminance(restored) <= 0.0f) {
      return;
    }

    const bool eye = LensAperture::isEye();
    const float tanHalfFovY = sun.tanHalfFovY;
    const float pixelAngle = 2.0f * tanHalfFovY / float(std::max(imageExtent.height, 1u));
    const double offAxis = std::acos(std::clamp(double(sun.cosOffAxis), -1.0, 1.0));

    // lensTanScale maps the game's tangents to the observer's, as the camera's lens fills the sensor with a view wider
    // than it covers by shrinking its angles. acceptance is the share of the sun's light the observer takes in. The
    // camera's lens passes rays out to the edge of its image circle, past which its barrel vignettes them, and the eye's
    // pupil, seen from the sun, is foreshortened by the angle off its line of sight, as the CIE's glare illuminance is
    // measured.
    float lensTanScale = 1.0f;
    float acceptance = std::max(sun.cosOffAxis, 0.0f);

    if (!eye) {
      const double focalLength = m_lensSystem.getFocalLength();
      lensTanScale = LensAperture::getImageHalfHeightMm(sun.aspectRatio, float(focalLength), tanHalfFovY) /
                     std::max(float(focalLength) * tanHalfFovY, 1e-6f);

      const double lensOffAxis = std::atan(double(lensTanScale) * std::tan(std::min(offAxis, 0.5 * kPiDouble - 1e-4)));
      const double coverage = std::atan(kImageCircleRadiusMm / std::max(focalLength, 1e-3));
      const double fade = std::clamp((lensOffAxis - coverage) / kVignettingRange, 0.0, 1.0);
      acceptance = float(1.0 - fade * fade * (3.0 - 2.0 * fade));
    }

    if (acceptance <= 0.0f) {
      return;
    }

    // The flare draws its brightest ghosts itself, so the sun's veil leaves their light out.
    Vector3 veilScale(1.0f);

    if (!eye) {
      Vector3 drawn(0.0f);

      for (const LensSystem::Ghost& ghost : ctx->getCommonObjects()->metaLensFlare().getDrawnGhosts()) {
        for (size_t g = 0; g < m_veil.surfaces.size(); g++) {
          if (m_veil.surfaces[g].first == ghost.first && m_veil.surfaces[g].second == ghost.second) {
            drawn = drawn + m_veil.energies[g];
            break;
          }
        }
      }

      const Vector3 veil = m_veil.energy;
      veilScale = Vector3(
        veil.x > 0.0f ? std::clamp(1.0f - drawn.x / veil.x, 0.0f, 1.0f) : 1.0f,
        veil.y > 0.0f ? std::clamp(1.0f - drawn.y / veil.y, 0.0f, 1.0f) : 1.0f,
        veil.z > 0.0f ? std::clamp(1.0f - drawn.z / veil.z, 0.0f, 1.0f) : 1.0f);
    }

    // The game's tangents of the sun's direction are its aspect corrected NDC times the view's.
    BloomSunGlareArgs args = {};
    args.imageSize = { imageExtent.width, imageExtent.height };
    args.sunPixel = sun.pixel;
    args.sunTan = sun.ndcAspect * (tanHalfFovY * lensTanScale);
    args.tanHalfFov = Vector2(tanHalfFovY * sun.aspectRatio, tanHalfFovY) * lensTanScale;
    args.radiancePerShare = restored * (acceptance * convolutionIntensity() / (pixelAngle * pixelAngle));
    // A disc of radius R whose radiance falls as mu^alpha to its limb spreads its light across a line with a variance
    // of R^2 / (alpha + 4).
    const float discRadiusTexels = sun.discRadiusPixels * sun.cosOffAxis * fit.texelsPerPixel;
    const float limbDarkening = std::max(sRGBLuminance(sun.limbDarkeningExponent), 0.0f);
    args.discVarianceTexels = discRadiusTexels * discRadiusTexels / (limbDarkening + 4.0f);
    args.nearTexelsPerPixel = fit.texelsPerPixel;
    args.anglePerNearTexel = 2.0f * tanHalfFovY * lensTanScale / std::max(fit.screenHeightTexels, 1.0f);
    args.psfRadiusPixels = m_sunPsfRadiusPixels;
    args.maxRadiance = kMaxSunGlareRadiance;
    args.veilScale = veilScale;
    args.psfEnabled = m_sunPsfValid ? 1u : 0u;
    ctx->pushConstants(0, sizeof(args), &args);

    ctx->bindResourceBuffer(BLOOM_SUN_GLARE_CONSTANTS_INPUT, DxvkBufferSlice(m_kernelConstants, 0, m_kernelConstants->info().size));
    ctx->bindResourceView(BLOOM_SUN_GLARE_PSF_INPUT, m_sunPsfValid ? m_sunPsfFiltered.view : m_dummy.view, nullptr);
    ctx->bindResourceSampler(BLOOM_SUN_GLARE_PSF_INPUT,
      ctx->getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE));
    ctx->bindResourceView(BLOOM_SUN_GLARE_SUN_VISIBILITY_INPUT, sunProbe.getVisibilityView(), nullptr);
    ctx->bindResourceView(BLOOM_SUN_GLARE_COLOR_INPUT_OUTPUT, color.view, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, BloomSunGlareShader::getShader());

    const VkExtent3D workgroups = util::computeBlockCount(imageExtent, VkExtent3D { 16, 16, 1 });
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void BloomConvolution::dispatch(Rc<RtxContext> ctx, const Resources::Resource& inOutColorBuffer, const RtxSunProbe* pSunProbe) {
    ScopedGpuProfileZone(ctx, "Convolution Bloom");

    const VkExtent2D bufferExtent = getBufferExtent(convolutionQuality());

    if (m_bufferExtent.width != bufferExtent.width || m_bufferExtent.height != bufferExtent.height || !m_spectrum.isValid()) {
      Rc<DxvkContext> baseCtx = ctx;
      createResources(baseCtx, bufferExtent);
    }

    const VkExtent3D imageExtent = inOutColorBuffer.image->info().extent;
    const BufferFit fit = computeFit(imageExtent, bufferExtent, convolutionPadding());
    const BufferFit coarseFit = computeFit(imageExtent, m_coarseExtent, kCoarsePadding);
    const bool coarse = convolutionCoarse();
    const RtCamera& camera = ctx->getSceneManager().getCamera();
    const float tanHalfFovY = std::tan(0.5f * camera.getFov());
    const float tanHalfFovX = tanHalfFovY * fit.aspectRatio;
    const BloomConvolutionDebugView view = debugView();
    const bool identity = view == BloomConvolutionDebugView::IdentityKernel;
    const bool eye = LensAperture::isEye();
    const float deltaSeconds = GlobalTime::get().deltaTimeMs() * 0.001f;

    const RtxSunProbe::State* pSun = pSunProbe != nullptr ? &pSunProbe->getState() : nullptr;
    const bool sunGlareActive = wantsSunVisibility() && !identity && pSun != nullptr && pSun->active && pSun->measured &&
                                pSunProbe->getVisibilityView() != nullptr;

    if (eye) {
      if (m_eye == nullptr) {
        m_eye = std::make_unique<EyeModel>(m_device);
      }

      const float fieldAreaDeg2 = float(4.0 * std::atan(tanHalfFovX) * std::atan(tanHalfFovY) * (180.0 / kPiDouble) * (180.0 / kPiDouble));

      if (m_eye->update(deltaSeconds, fieldAreaDeg2)) {
        m_eyeVersion++;
      }
    } else if (updateLens(tanHalfFovY, fit.aspectRatio)) {
      m_lensVersion++;
    }

    // The sun's disc as the filtered PSF texture holds it, on the axis.
    if (sunGlareActive) {
      const float discRadius = pSun->discRadiusPixels * pSun->cosOffAxis;
      const Vector3 limbDarkening = pSun->limbDarkeningExponent;

      if (std::abs(discRadius - m_sunPsfDiscRadiusPixels) > 0.02f * std::max(discRadius, 0.5f) || limbDarkening != m_sunLimbDarkening) {
        m_sunPsfDiscRadiusPixels = discRadius;
        m_sunLimbDarkening = limbDarkening;
        m_sunPsfValid = false;
      }
    }

    if (view == BloomConvolutionDebugView::Kernel) {
      // The spatial kernel is rebuilt into the result buffers every frame for display, which leaves the cached kernel
      // spectra and centre tap as they are.
      buildKernel(ctx, fit, coarseFit, tanHalfFovY, false, coarse, m_bloom, m_coarseBloom, m_dummy, false, false);
      dispatchComposite(ctx, inOutColorBuffer, fit, coarseFit, false, m_bloom, BLOOM_FFT_DEBUG_VIEW_KERNEL, 0.0f, false);
      return;
    }

    const KernelKey key = makeKernelKey(fit, coarseFit, tanHalfFovY, identity);

    if (!m_kernelValid || !(key == m_kernelKey)) {
      buildKernel(ctx, fit, coarseFit, tanHalfFovY, identity, coarse, m_kernel, m_coarseKernel, m_centerTap, true, sunGlareActive);
      m_kernelKey = key;
      m_kernelValid = true;
      m_sunPsfValid = sunGlareActive;
    } else if (sunGlareActive && !m_sunPsfValid) {
      refreshSunPsf(ctx, fit, coarseFit, tanHalfFovY);
      m_sunPsfValid = true;
    }

    // Fills the near buffer from the image, and the coarse one from the near buffer before its transform.
    dispatchSetup(ctx, inOutColorBuffer, { imageExtent.width, imageExtent.height }, m_spectrum, fit, fit.texelsPerPixel, fit.bufferOffset);

    if (coarse) {
      const float coarseTexelsPerNearTexel = coarseFit.texelsPerPixel / fit.texelsPerPixel;
      dispatchSetup(ctx, m_spectrum, m_bufferExtent, m_coarseSpectrum, coarseFit, coarseTexelsPerNearTexel,
                    coarseFit.bufferOffset - fit.bufferOffset * coarseTexelsPerNearTexel);
    }

    if (eye && !identity) {
      // The restored sun's illuminance over the field's solid angle, while it is in view.
      float sunLuminance = 0.0f;

      if (sunGlareActive && pSun->onScreen) {
        const double fieldSolidAngle = 4.0 * std::asin(tanHalfFovX * tanHalfFovY /
          std::sqrt((1.0 + tanHalfFovX * tanHalfFovX) * (1.0 + tanHalfFovY * tanHalfFovY)));
        sunLuminance = float(sRGBLuminance(pSunProbe->getRestoredIlluminance()) / std::max(fieldSolidAngle, 1e-6));
      }

      dispatchFieldLuminance(ctx, coarse ? coarseFit : fit, pSunProbe, sunLuminance);
    }

    uint32_t firstColumn;
    uint32_t columnCount;
    uint32_t coarseFirstColumn;
    uint32_t coarseColumnCount;
    getImageColumns(fit, firstColumn, columnCount);
    getImageColumns(coarseFit, coarseFirstColumn, coarseColumnCount);

    // The columns go first forward and last back, so the transforms skip those outside the image.
    {
      ScopedGpuProfileZone(ctx, "Convolution Bloom FFT");
      dispatchFft(ctx, m_spectrum, true, true, firstColumn, columnCount);
      dispatchFft(ctx, m_spectrum, false, true);

      if (coarse) {
        dispatchFft(ctx, m_coarseSpectrum, true, true, coarseFirstColumn, coarseColumnCount);
        dispatchFft(ctx, m_coarseSpectrum, false, true);
      }
    }

    // The identity view shows the image's whole round trip through the buffer, so it keeps the centre tap in.
    const bool subtractCenter = !identity;
    dispatchConvolve(ctx, m_spectrum, m_kernel, m_bloom, m_bufferExtent, subtractCenter);

    if (coarse) {
      dispatchConvolve(ctx, m_coarseSpectrum, m_coarseKernel, m_coarseBloom, m_coarseExtent, false);
    }

    {
      ScopedGpuProfileZone(ctx, "Convolution Bloom Inverse FFT");
      dispatchFft(ctx, m_bloom, false, false);
      dispatchFft(ctx, m_bloom, true, false, firstColumn, columnCount);

      if (coarse) {
        dispatchFft(ctx, m_coarseBloom, false, false);
        dispatchFft(ctx, m_coarseBloom, true, false, coarseFirstColumn, coarseColumnCount);
      }
    }

    const bool bloomOnly = identity || view == BloomConvolutionDebugView::BloomOnly;
    dispatchComposite(ctx, inOutColorBuffer, fit, coarseFit, coarse && !identity, m_bloom,
                      bloomOnly ? BLOOM_FFT_DEBUG_VIEW_BLOOM_ONLY : BLOOM_FFT_DEBUG_VIEW_NONE,
                      convolutionIntensity(), subtractCenter);

    if (sunGlareActive) {
      dispatchSunGlare(ctx, inOutColorBuffer, fit, *pSunProbe);
    }
  }

  void BloomConvolution::showImguiSettings() {
    qualityCombo.getKey(&convolutionQualityObject());
    RemixGui::SetTooltipToLastWidgetOnHover("Size of the near FFT buffer. A coarse buffer a quarter of its size carries the far reaches.");
    RemixGui::DragFloat("Intensity##convolutionBloom", &convolutionIntensityObject(), 0.002f, 0.0f, 1.0f, "%.3f");
    RemixGui::SetTooltipToLastWidgetOnHover("1 passes all of the image's light through the observer's point spread function, as through a real lens\nor eye. Lower values blend back toward the image without it. The image keeps its overall brightness.");
    RemixGui::DragFloat("Padding##convolutionBloom", &convolutionPaddingObject(), 0.005f, 0.0f, 0.5f, "%.3f");
    RemixGui::SetTooltipToLastWidgetOnHover("Black border around the image in the near FFT buffer. More padding widens the near kernel at a lower\nresolution.");
    RemixGui::Checkbox("Coarse Scale##convolutionBloom", &convolutionCoarseObject());
    RemixGui::SetTooltipToLastWidgetOnHover("Carries the kernel's far reaches on a coarse buffer, so every bright source spreads light across the frame.");
    RemixGui::Checkbox("Sun Glare##convolutionBloom", &sunGlareObject());
    RemixGui::SetTooltipToLastWidgetOnHover("Adds the glare of the sun's light that its rendered disc's radiance clamp holds back, at full\nresolution, on or off the screen, through its ray traced visibility.");

    ImGui::Separator();
    RemixGui::DragFloat("Diffraction Scale##convolutionBloom", &kernelScaleObject(), 0.01f, 0.01f, 8.0f, "%.2f");
    RemixGui::SetTooltipToLastWidgetOnHover("1 = the true size of the diffraction pattern. Larger exaggerates the starburst or corona.");
    RemixGui::DragFloat("Spike Boost##convolutionBloom", &spikeBoostObject(), 0.1f, 0.0f, 1000.0f, "%.1f");
    RemixGui::SetTooltipToLastWidgetOnHover("Weight of the pattern outside its core, the spikes, rings and corona. 1 = physical.");
    RemixGui::DragFloat("Spectral Dispersion##convolutionBloom", &spectralDispersionObject(), 0.01f, 0.0f, 2.0f, "%.2f");
    RemixGui::DragFloat("Scatter Scale##convolutionBloom", &scatterScaleObject(), 0.01f, 0.0f, 10.0f, "%.2f");
    RemixGui::SetTooltipToLastWidgetOnHover("Scale on the lens's veil, roughness halo and mechanical scatter, or the eye's disability glare and\nlenticular halo. 1 = physical.");

    if (LensAperture::isEye() && m_eye != nullptr) {
      ImGui::Text("Pupil %.2f mm", m_eye->getPupilDiameterMm());
    } else if (m_lensSystem.getPrescription() != ~0u) {
      const LensScatter scatter = computeLensScatter();
      ImGui::Text("Veiling glare: %.2f%% veil, %.2f%% roughness, %.2f%% mechanical", sRGBLuminance(scatter.veil) * 100.0f,
                  sRGBLuminance(scatter.roughness) * 100.0f, scatter.mechanical * 100.0f);
    }

    ImGui::Separator();
    LensAperture::showImguiSettings();

    ImGui::Separator();
    debugViewCombo.getKey(&debugViewObject());
  }

}
