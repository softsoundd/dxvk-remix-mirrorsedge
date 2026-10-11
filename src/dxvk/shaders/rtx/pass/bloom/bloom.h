/*
* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BLOOM_H
#define BLOOM_H

#include "rtx/utility/shader_types.h"

#define BLOOM_DOWNSAMPLE_INPUT  0
#define BLOOM_DOWNSAMPLE_OUTPUT 1

#define BLOOM_UPSAMPLE_INPUT    0
#define BLOOM_UPSAMPLE_OUTPUT   1

#define BLOOM_COMPOSITE_COLOR_INPUT_OUTPUT 0
#define BLOOM_COMPOSITE_BLOOM              1

// Convolution bloom. Each pass groups its bindings as inputs from 0, input-outputs from 10 and outputs from 20.

#define BLOOM_FFT_SPECTRUM_INPUT_OUTPUT 10

#define BLOOM_FFT_SETUP_COLOR_INPUT     0
#define BLOOM_FFT_SETUP_SPECTRUM_OUTPUT 20

#define BLOOM_FFT_CONVOLVE_SPECTRUM_INPUT   0
#define BLOOM_FFT_CONVOLVE_KERNEL_INPUT     1
#define BLOOM_FFT_CONVOLVE_CENTER_TAP_INPUT 2
#define BLOOM_FFT_CONVOLVE_OUTPUT           20

#define BLOOM_FFT_COMPOSITE_BLOOM_INPUT        0
#define BLOOM_FFT_COMPOSITE_CENTER_TAP_INPUT   1
#define BLOOM_FFT_COMPOSITE_COARSE_BLOOM_INPUT 2
#define BLOOM_FFT_COMPOSITE_COLOR_INPUT_OUTPUT 10

#define BLOOM_KERNEL_APERTURE_PARTICLES_INPUT      0
#define BLOOM_KERNEL_APERTURE_CELLS_INPUT          1
#define BLOOM_KERNEL_APERTURE_CELL_PARTICLES_INPUT 2
#define BLOOM_KERNEL_APERTURE_OUTPUT               20

#define BLOOM_KERNEL_POWER_INPUT  0
#define BLOOM_KERNEL_POWER_OUTPUT 20

#define BLOOM_KERNEL_BUILD_CONSTANTS_INPUT       0
#define BLOOM_KERNEL_BUILD_APERTURE_POWER_INPUT  1
#define BLOOM_KERNEL_BUILD_OUTPUT                20

#define BLOOM_KERNEL_REDUCE_INPUT           0
#define BLOOM_KERNEL_REDUCE_ROW_SUMS_INPUT  1
#define BLOOM_KERNEL_REDUCE_ROW_SUMS_OUTPUT 20
#define BLOOM_KERNEL_REDUCE_TOTALS_OUTPUT   21

#define BLOOM_KERNEL_COMBINE_TOTALS_INPUT              0
#define BLOOM_KERNEL_COMBINE_KERNEL_INPUT_OUTPUT       10
#define BLOOM_KERNEL_COMBINE_COARSE_KERNEL_INPUT_OUTPUT 11
#define BLOOM_KERNEL_COMBINE_CENTER_TAP_OUTPUT         20

#define BLOOM_FIELD_LUMINANCE_INPUT                0
#define BLOOM_FIELD_LUMINANCE_SUN_VISIBILITY_INPUT 1
#define BLOOM_FIELD_LUMINANCE_OUTPUT               20

#define BLOOM_SUN_PSF_FILTER_INPUT  0
#define BLOOM_SUN_PSF_FILTER_OUTPUT 20

#define BLOOM_SUN_GLARE_CONSTANTS_INPUT      0
#define BLOOM_SUN_GLARE_PSF_INPUT            1
#define BLOOM_SUN_GLARE_SUN_VISIBILITY_INPUT 2
#define BLOOM_SUN_GLARE_COLOR_INPUT_OUTPUT   10

#define BLOOM_KERNEL_REDUCE_GROUP_SIZE 256

// The aperture spans 1 / BLOOM_KERNEL_APERTURE_OVERSAMPLING of its spectrum's width, which keeps its far field pattern
// from aliasing.
#define BLOOM_KERNEL_APERTURE_OVERSAMPLING 2

#define BLOOM_OBSERVER_CAMERA 0
#define BLOOM_OBSERVER_EYE    1

#define BLOOM_FFT_DEBUG_VIEW_NONE       0
#define BLOOM_FFT_DEBUG_VIEW_BLOOM_ONLY 1
#define BLOOM_FFT_DEBUG_VIEW_KERNEL     2

#define BLOOM_KERNEL_SPECTRAL_SAMPLES 16
// Radial profile of the camera lens's inter-reflection veil, on log spaced radii.
#define BLOOM_VEIL_TABLE_SIZE 64
// The most blades the camera's iris has (rtx.lens.apertureBlades).
#define BLOOM_APERTURE_MAX_SIDES 16
// The waves across each blade edge's spike that make up its streaks, and their periods' range in degrees.
#define BLOOM_STREAK_WAVES 8
#define BLOOM_STREAK_SHORTEST_PERIOD_DEGREES 0.4f
#define BLOOM_STREAK_LONGEST_PERIOD_DEGREES 3.0f

// Push constants

struct BloomDownsampleArgs {
  float2 inputSizeInverse;
  uint2  downsampledOutputSize;
  float2 downsampledOutputSizeInverse;
  float  threshold;
};

struct BloomUpsampleArgs {
  float2 inputSizeInverse;
  uint2  upsampledOutputSize;
  float2 upsampledOutputSizeInverse;
};

struct BloomCompositeArgs {
  uint2  imageSize;
  float2 imageSizeInverse;
  float  intensity;
};

struct BloomFftArgs {
  uint alongY;
  uint forward;
  uint firstLine;
  uint pad0;
};

// Output pixel coordinates (top left origin, pixel centres at + 0.5) map to continuous buffer texel coordinates as
// texel = pixel * texelsPerPixel + bufferOffset.
struct BloomFftSetupArgs {
  uvec2 bufferSize;
  uvec2 imageSize;

  float texelsPerPixel;
  float maxInputRadiance;
  vec2 bufferOffset;
};

// With subtractCenter set, the kernel's centre tap is taken out of the convolution and each pixel keeps that share of
// its own light at full resolution, so the low resolution buffer carries only the light the kernel spreads.
struct BloomFftConvolveArgs {
  uvec2 bufferSize;
  uint subtractCenter;
  uint pad0;
};

struct BloomFftCompositeArgs {
  uvec2 imageSize;
  uvec2 bufferSize;

  vec2 bufferOffset;
  float texelsPerPixel;
  float intensity;

  uvec2 coarseBufferSize;
  vec2 coarseBufferOffset;

  float coarseTexelsPerPixel;
  uint coarseEnabled;
  uint debugView;
  uint subtractCenter;
};

// The aperture rasterised for its far field pattern, in units of its corner radius (the camera's iris) or its radius
// (the eye's pupil). The eye's pupil carries the Stiles-Crawford apodisation, its lashes and the particles of the lens
// and vitreous, which sit binned in a grid of cellsPerSide cells over the aperture's square.
struct BloomKernelApertureArgs {
  uint size;
  uint blades;
  float radiusTexels;
  float curvature;

  float rotation;
  uint observer;
  float pupilRadiusMm;
  // The Stiles-Crawford coefficient, by which the pupil's amplitude falls as 10^(-stilesCrawford r^2 / 2), r in mm.
  float stilesCrawford;

  uint lashCount;
  float lashWidth;
  uint cellsPerSide;
  float lashPhase;

  // The camera's iris blades' spreads off their regular places (see lens_aperture.slangh).
  float distanceJitter;
  float angleJitter;
  uint pad0;
  uint pad1;
};

struct BloomKernelPowerArgs {
  uint size;
  // The aperture spectrum's total power, which normalises it.
  float apertureEnergy;
  uint pad0;
  uint pad1;
};

struct BloomKernelReduceArgs {
  uvec2 bufferSize;
  uint part;
  uint pad0;
};

struct BloomKernelCombineArgs {
  uvec2 bufferSize;
  uvec2 coarseBufferSize;

  uint identity;
  uint coarseEnabled;
  uint pad0;
  uint pad1;
};

// The mean luminance of the field over the coarse buffer's image, with the restored sun's illuminance spread over the
// field's solid angle, for the eye's pupil.
struct BloomFieldLuminanceArgs {
  uvec2 bufferSize;
  vec2 imageMin;

  vec2 imageMax;
  uint sunEnabled;
  float sunLuminance;
};

// Filters the sun's point spread function, on a centred square texture, by the sun's disc: its limb darkened radiance
// over discRadiusTexels.
struct BloomSunPsfFilterArgs {
  uint size;
  float discRadiusTexels;
  uint pad0;
  uint pad1;

  vec3 limbDarkeningExponent;
  uint pad2;
};

// The restored sun's glare at full resolution. Each pixel takes the point spread function at its true angle from the
// sun, in the observer's own angles, so the pattern stays right however far the sun is from the axis or the screen.
// Great circles through the sun project to straight lines through its image, so the pattern's directions are the
// screen's.
struct BloomSunGlareArgs {
  uvec2 imageSize;
  // The sun's image, off the screen as well.
  vec2 sunPixel;

  // The tangents of the sun's direction and of the view's half angles, in the observer's angles, y up.
  vec2 sunTan;
  vec2 tanHalfFov;

  // The illuminance the clamped disc lacks, before its visibility, times the share of it the observer takes in, over
  // an on axis pixel's solid angle: the radiance a pixel's share of the light gives.
  vec3 radiancePerShare;
  // The variance in near texels squared that the limb darkened disc spreads the spikes' light over across them.
  float discVarianceTexels;

  // Near buffer texels per on axis output pixel, and radians of the observer's angle per near texel.
  float nearTexelsPerPixel;
  float anglePerNearTexel;
  // The filtered PSF texture's reach in on axis pixels, past which the analytic far field takes over.
  float psfRadiusPixels;
  float maxRadiance;

  // Scales the camera's inter-reflection veil for the sun, whose brightest ghosts the lens flare draws itself.
  vec3 veilScale;
  uint psfEnabled;
};

// Constant buffer

struct BloomKernelSpectralSample {
  // Linear Rec.709 response of the sample's wavelength. The weights of each channel sum to one.
  vec3 weight;
  // Reference wavelength over the sample's: the factor the diffraction pattern is resampled by.
  float scale;
};

// The kernel's parts, as each texel's share of the light. The near pass builds the kernel inside the near buffer's
// window, the coarse pass what lies beyond it on texels nearTexelsPerCoarseTexel near texels wide, and the sun glare
// pass the same far part at full resolution. Radii are in near buffer texels.
struct BloomKernelArgs {
  uvec2 nearBufferSize;
  uvec2 coarseBufferSize;

  uint apertureSize;
  uint observer;
  float nearWindowRadius;
  float coarseWindowRadius;

  float nearTexelsPerCoarseTexel;
  // The reference wavelength over the aperture's corner radius, as an angle in near texels: the pattern's scale.
  float sigmaTexels;
  float spikeBoost;
  // Radius past which the spike boost applies.
  float coreRadiusTexels;

  // Shares of the light in the scatter parts.
  // The camera's are the inter-reflection veil, surface roughness, and mechanical scatter.
  // The eye's are the CIE glare and the lenticular halo, with no third part.
  vec3 scatterShare0;
  // Millimetres on the camera's sensor per near texel.
  // Radians per near texel for the eye.
  float physicalPerTexel;

  vec3 scatterShare1;
  // The camera lens's focal length in mm, which turns sensor mm into scatter angles.
  float focalLengthMm;

  vec3 scatterShare2;
  // Near buffer texels per on axis output pixel, for the sun's point spread function.
  float nearTexelsPerPixel;

  // ABg scatter lobes, p(theta) = 1 / (norm (shoulder^slope + theta^slope)) per steradian, for the camera's
  // surface roughness and mechanical scatter.
  float roughnessShoulder;
  float roughnessSlope;
  float roughnessNorm;
  float mechanicalShoulder;

  float mechanicalSlope;
  float mechanicalNorm;
  // The veil table's first radius in mm and the natural log of the ratio between consecutive radii.
  float veilMinRadiusMm;
  float veilLogStep;

  // The age factor and pigmentation of the eye's CIE disability glare, and the integral that normalises it.
  float cieAgeFactor;
  float ciePigmentation;
  float cieNorm;
  // The ring angle of the eye's lenticular halo at the reference wavelength, and its width, in radians.
  float haloAngle;

  float haloWidth;
  // Aperture spectrum texels per sigma kernel texels, which is 2 * oversampling for an aperture spanning
  // 1 / oversampling of the spectrum's width and 2 for one spanning all of it.
  float spectrumTexelsPerSigma;
  // The near field comes from the aperture's spectrum and the far field from its edges, crossfaded over this range of
  // fractions of the spectrum's Nyquist radius.
  float farFieldStart;
  float farFieldEnd;

  // The aperture's side count for its far field, 0 for a circle. apertureSides holds the sides, in units of its corner
  // radius.
  uint apertureBlades;
  // The RMS contrast of the streaks the blades' rough edges leave across their spikes.
  float apertureStreakContrast;
  uint pad0;
  uint pad1;

  float apertureArea;
  // The aperture spectrum's total power, which normalises it.
  float apertureEnergy;
  // The size of the sun's point spread function's centred square texture, and its output pixels per texel.
  uint sunPsfSize;
  float sunPsfPixelsPerTexel;

  // The camera's veil density in mm^-2, a unit energy mix of the ghosts' discs, at the table's radii.
  vec4 veilTable[BLOOM_VEIL_TABLE_SIZE / 4];

  // The iris's sides with its blades off their regular places (see LensAperture::Side): the middle of the directions
  // each side's normals turn through, half their range, and its length.
  vec4 apertureSides[BLOOM_APERTURE_MAX_SIDES];
  // Each side's streak waves, two to an entry: angular frequency in radians^-1 and phase.
  vec4 apertureStreakWaves[BLOOM_APERTURE_MAX_SIDES * BLOOM_STREAK_WAVES / 2];

  BloomKernelSpectralSample spectralSamples[BLOOM_KERNEL_SPECTRAL_SAMPLES];
};

#endif  // BLOOM_H
