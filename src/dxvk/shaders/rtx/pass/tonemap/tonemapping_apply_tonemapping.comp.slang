/*
* Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx/pass/tonemap/tonemapping.slangh"
#include "rtx/pass/tonemap/tonemapping_noise.slangh"
#include "rtx/pass/tonemap/fork_tonemap_operators.slangh"
#include "rtx/utility/math.slangh"
#include "rtx/utility/color.slangh"

[[vk::binding(TONEMAPPING_APPLY_BLUE_NOISE_TEXTURE_INPUT)]]
Texture2DArray BlueNoise;

[[vk::binding(TONEMAPPING_APPLY_TONEMAPPING_COLOR_INPUT)]]
RWTexture2D<float4> InColorBuffer;

[[vk::binding(TONEMAPPING_APPLY_TONEMAPPING_EXPOSURE_INPUT)]]
RWTexture1D<float> InExposure;

[[vk::binding(TONEMAPPING_APPLY_TONEMAPPING_COLOR_OUTPUT)]]
RWTexture2D<float4> OutColorBuffer;

layout(push_constant)
ConstantBuffer<ToneMappingApplyToneMappingArgs> cb;

// Apply post-tonemap color grading: tint, contrast, saturation.
float3 colorGrading(float3 inputColor)
{
  // Tint.
  inputColor = inputColor * cb.colorBalance;

  // Contrast around 18% mid grey.
  inputColor = saturate((cb.contrast * (inputColor - 0.18f)) + 0.18f);

  // Saturation around BT.709 luminance.
  inputColor = mix(float3(calcBt709Luminance(inputColor)), inputColor, cb.saturation);

  return inputColor;
}

float3 applyToneMapping(float3 color, uint2 pixelPosition)
{
  color *= getExposure(InExposure, cb.enableAutoExposure, cb.exposureFactor);

  if (cb.toneMappingEnabled != 0)
  {
    // Observer adaptive state passed to psycho17; all other operators
    // ignore this parameter. Fixed at the renodx reference mid-gray
    // (0.18, 0.18, 0.18) unconditionally — AE state is not wired in.
    const float3 adaptiveStateBT709 = float3(0.18f, 0.18f, 0.18f);

    color = applyTonemapOperator(cb.tonemapOperator, color, /*suppressBlackLevelClamp=*/false,
                                 cb.hableExposureBias,
                                 cb.hableShoulderStrength,
                                 cb.hableLinearStrength,
                                 cb.hableLinearAngle,
                                 cb.hableToeStrength,
                                 cb.hableToeNumerator,
                                 cb.hableToeDenominator,
                                 cb.hableWhitePoint,
                                 cb.agxSaturation,
                                 cb.agxLook,
                                 cb.psycho17PeakValue,
                                 cb.psycho17Exposure,
                                 cb.psycho17Highlights,
                                 cb.psycho17Shadows,
                                 cb.psycho17Contrast,
                                 cb.psycho17PurityScale,
                                 cb.psycho17BleachingIntensity,
                                 cb.psycho17ClipPoint,
                                 cb.psycho17HueRestore,
                                 cb.psycho17AdaptationContrast,
                                 cb.psycho17WhiteCurveMode,
                                 cb.psycho17ConeResponseExponent,
                                 cb.psycho17GamutCompression,
                                 cb.psycho17GamutCompressionMode,
                                 adaptiveStateBT709);
  }

  if (cb.colorGradingEnabled > 0)
  {
    color = colorGrading(color);
  }

  if (cb.performSRGBConversion > 0)
  {
    color = linearToGamma(color);
  }

  // Apply dithering to the output to break up banding artifacts. First-order
  // only; ideally we would dither in linear space using sRGB range info, but
  // the resulting error is small.
  // fork — 2026-06-27 (experimental): only dither when this pass actually
  // encoded to gamma. The fork's tonemap refactor defers sRGB conversion to the
  // dedicated srgb_dither pass (performSRGBConversion=false here), so dithering
  // here would add the ±0.5/255 perturbation in LINEAR space — which the later
  // linearToGamma in srgb_dither amplifies ~13x in the near-black toe, producing
  // a heavy boiling grain only in dark regions (plus a redundant double-dither).
  // Gating on performSRGBConversion ties the dither to the space it is valid in.
  return ditherTo8Bit(
    color,
    pixelPosition,
    cb.frameIndex,
    cb.performSRGBConversion > 0 && cb.ditherMode != ditherModeNone,
    cb.ditherMode == ditherModeSpatialTemporal
  );
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  float3 inputColor = imageLoad(InColorBuffer, threadId).xyz;

  float3 outputColor = applyToneMapping(inputColor, threadId);

  // Clamp outputColor to avoid problems with NIS.
  outputColor = saturate(outputColor);

  OutColorBuffer[threadId] = vec4(outputColor, 1.0f);
}
