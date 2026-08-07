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

// Mirror's Edge (UE3 / TdToneMapping) display transform.
//
// Reproduces the game's tone map pipeline on Remix's linear HDR output:
//   exposure -> highlights/shadows grade -> (optional hue-preserving shoulder) ->
//   midtone pow grade -> desaturation + overlay -> display gamma encode ->
//   16-segment per-channel colour curves -> (optional white correction).
//
// The hue-preserving shoulder, grade-preserve blend and white-neutrality
// stages come from the FaithfulLuma shader set; with all of them disabled the
// transform matches the shipped TdToneMappingPixelShader.usf behavior
// (per-channel grade with a hard clip at scene white).
//
// IMPORTANT: the output of this pass is already display-encoded (gamma +
// curves), exactly what the game wrote to its backbuffer. The final sRGB
// conversion in the srgb_dither pass must be skipped when this pass runs.

#include "rtx/pass/tonemap/tonemapping_ue3.h"
#include "rtx/pass/tonemap/tonemapping.slangh"
#include "rtx/utility/math.slangh"
#include "rtx/utility/color.slangh"

[[vk::binding(TONEMAPPING_UE3_COLOR_INPUT)]]
RWTexture2D<float4> InColorBuffer;

[[vk::binding(TONEMAPPING_UE3_EXPOSURE_INPUT)]]
RWTexture1D<float> InExposure;

[[vk::binding(TONEMAPPING_UE3_CURVE_K_INPUT)]]
Sampler1D<float4> InCurveK;

[[vk::binding(TONEMAPPING_UE3_CURVE_M_INPUT)]]
Sampler1D<float4> InCurveM;

layout(binding = TONEMAPPING_UE3_CONSTANTS_INPUT)
ConstantBuffer<ToneMappingUe3Args> cb;

[[vk::binding(TONEMAPPING_UE3_COLOR_OUTPUT)]]
RWTexture2D<float4> OutColorBuffer;

static const float3 kLumaWeightsRec709 = float3(0.2126f, 0.7152f, 0.0722f);
static const float kCurveDomainWhite = (UE3_TONEMAP_NUM_CURVE_SEGMENTS - 1.0f) / UE3_TONEMAP_NUM_CURVE_SEGMENTS; // 15/16

// FaithfulLuma: neutralize bright, near-neutral display pixels toward their own
// luminance so clipped sunlit whites lose baked warmth without a fixed blue offset.
float3 applyWhiteNeutralityCorrection(float3 color)
{
  const float luma = dot(color, kLumaWeightsRec709);
  const float displayEnergy = (color.r + color.g + color.b) / 3.0f;
  const float peak = max(max(color.r, color.g), color.b);
  const float lowChannel = min(min(color.r, color.g), color.b);
  const float relativeChroma = (peak - lowChannel) / max(peak, 0.001f);
  float lumaMask = saturate((luma - cb.whiteLumaStart) / cb.whiteLumaRange);
  float neutralityMask = 1.0f - saturate((relativeChroma - cb.whiteChromaStart) / cb.whiteChromaRange);
  const float whiteMask = lumaMask * lumaMask * neutralityMask * neutralityMask;
  const float3 correctedColor = mix(color, float3(luma, luma, luma), whiteMask * cb.whiteNeutralityStrength);
  const float correctedEnergy = (correctedColor.r + correctedColor.g + correctedColor.b) / 3.0f;
  const float energyProtection = min(1.0f, displayEnergy / max(correctedEnergy, 0.001f));

  return saturate(correctedColor * mix(1.0f, energyProtection, whiteMask));
}

// Maps a display value to the curve LUT sample coordinate, honoring the
// filter mode the game bound. Mirror's Edge point-filters its curve LUTs
// (exact piecewise segments: segment = floor(15x), texel 15 only at x = 1);
// point mode snaps to the center of the texel D3D9 would select.
float curveSampleCoord(float x)
{
  const float u = saturate(x * kCurveDomainWhite);

  if (cb.curvePointSampling != 0)
  {
    const float texel = min(floor(u * UE3_TONEMAP_NUM_CURVE_SEGMENTS), UE3_TONEMAP_NUM_CURVE_SEGMENTS - 1.0f);
    return (texel + 0.5f) / UE3_TONEMAP_NUM_CURVE_SEGMENTS;
  }

  return u;
}

// Evaluates the game's baked 16-segment curves exactly like the PC
// TdToneMappingPixelShader: 16-texel LUT sampled with the 15/16 domain
// correction, then y = x * Ms + Bs per channel.
float3 applyMsBsColorCurves(float3 color)
{
  const float4 rCurve = InCurveK.SampleLevel(curveSampleCoord(color.r), 0);
  const float4 gCurve = InCurveK.SampleLevel(curveSampleCoord(color.g), 0);
  const float4 bCurve = InCurveM.SampleLevel(curveSampleCoord(color.b), 0);

  float3 result;
  result.r = color.r * rCurve.r + rCurve.g; // KTexture.r = Ms.r, KTexture.g = Bs.r
  result.g = color.g * gCurve.b + gCurve.a; // KTexture.b = Ms.g, KTexture.a = Bs.g
  result.b = color.b * bCurve.r + bCurve.g; // MTexture.r = Ms.b, MTexture.g = Bs.b

  return result;
}

float3 applyUe3ToneMapping(float3 linearColor)
{
  const float3 sceneShadows = cb.sceneShadowsAndDesaturation.rgb;
  const float sceneDesaturation = cb.sceneShadowsAndDesaturation.a; // stored as (1 - desaturation), applied verbatim
  const float gammaInverse = cb.gammaColorScaleAndInverse.a;

  const float exposure = getExposure(InExposure, cb.enableAutoExposure, cb.exposureFactor);
  const float3 hdrColor = linearColor * exposure;

  float3 graded;

  if (cb.huePreservingShoulder != 0)
  {
    // FaithfulLuma: grade in linear HDR, tone map the luminance with an extended
    // Reinhard shoulder and reconstruct RGB around it so hue stays stable.
    const float3 gradedHDR = max(0.0f, hdrColor * cb.sceneInverseHighLights.rgb - sceneShadows);
    const float lumaHDR = dot(gradedHDR, kLumaWeightsRec709);
    const float lumaTM = lumaHDR * (1.0f + (lumaHDR / (cb.linearWhite * cb.linearWhite))) / (1.0f + lumaHDR);
    const float3 reconstructed = gradedHDR * (lumaTM / max(lumaHDR, 0.0001f));

    float3 tonemapped = reconstructed;

    if (cb.highlightDesaturation != 0)
    {
      // Only genuinely hot sources desaturate toward display white.
      float highlightChromaWeight = saturate((lumaHDR - (cb.linearWhite * 0.5f)) / (cb.linearWhite * 1.5f));
      highlightChromaWeight = highlightChromaWeight * highlightChromaWeight;
      tonemapped = mix(reconstructed, float3(lumaTM, lumaTM, lumaTM), highlightChromaWeight * cb.highlightDesaturationStrength);
    }

    const float3 safeTonemapped = max(0.0001f, tonemapped);

    if (cb.gradePreserveBlend != 0)
    {
      // Keep the shipped per-channel midtone grade in shadows/lower mids, fading
      // to a neutral luminance grade in highlights (hue-stable).
      const float neutralMidTone = dot(cb.sceneMidTones.rgb, kLumaWeightsRec709);
      const float3 neutralGraded = pow(safeTonemapped, float3(neutralMidTone, neutralMidTone, neutralMidTone));
      const float3 perChannelGraded = pow(safeTonemapped, cb.sceneMidTones.rgb);
      const float gradePreserveWeight = 1.0f - saturate((lumaTM - cb.gradePreservePivot) * cb.gradePreserveSlope);
      graded = mix(neutralGraded, perChannelGraded, gradePreserveWeight);
    }
    else
    {
      graded = pow(safeTonemapped, cb.sceneMidTones.rgb);
    }
  }
  else
  {
    // Verbatim shipped TdToneMappingPixelShader: hard clip at scene white, then
    // the per-channel grade pow. The shipped shader can feed a negative base into
    // pow (NaN on SM3, flushed to black by the 8-bit output); clamp to ~0 instead
    // which matches the visible result.
    graded = pow(max(0.0001f, saturate(hdrColor) * cb.sceneInverseHighLights.rgb - sceneShadows), cb.sceneMidTones.rgb);
  }

  // Desaturation + overlay (verbatim shipped math)
  const float scaledLuminance = dot(graded, cb.sceneScaledLuminanceWeights.rgb);
  const float3 materialResult = cb.gammaOverlayColor.rgb + graded * sceneDesaturation + scaledLuminance;

  // Display gamma encode (Mirror's Edge authored around DisplayGamma = 2.0)
  float3 outColor = pow(saturate(materialResult * cb.gammaColorScaleAndInverse.rgb), gammaInverse);

  // Per-map colour curves in display space
  if (cb.applyColorCurves != 0)
  {
    outColor = applyMsBsColorCurves(outColor);
  }

  outColor = saturate(outColor);

  if (cb.whiteNeutrality != 0)
  {
    outColor = applyWhiteNeutralityCorrection(outColor);
  }

  return outColor;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  if (any(threadId >= uint2(imageSize(InColorBuffer))))
  {
    return;
  }

  const float3 inputColor = imageLoad(InColorBuffer, threadId).xyz;

  const float3 outputColor = applyUe3ToneMapping(inputColor);

  OutColorBuffer[threadId] = float4(saturate(outputColor), 1.0f);
}
