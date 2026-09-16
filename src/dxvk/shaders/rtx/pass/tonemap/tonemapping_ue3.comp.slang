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
//   exposure -> highlights/shadows/midtones grade -> desaturation + overlay ->
//   display gamma encode -> 16-segment per-channel colour curves.
//
// With cb.faithfulLuma == 0 the transform is the shipped TdToneMappingPixelShader.usf
// verbatim: saturate(Color * Exposure) clips each channel at exposed 1.0 and every pow()
// is Common.usf's ClampedPow, pow(max(abs(x), 1e-4), y), whose guard is what puts the
// shipped black at #020202 (1e-4 encodes to 0.01 at gamma 2.0).
//
// With cb.faithfulLuma != 0 the shipped shader is changed only where it loses information:
//   - the clip becomes a per-channel range compression curve: Neutwo (RenoDX's peak /
//     white-clip form of x / sqrt(x^2 + 1)) or the Faithful Luma shoulder (identity below a
//     knee, extended Reinhard to a white point);
//   - the curve's value and saturation are kept and its hue is solved so the OKLab hue is the
//     scene's, turned by the small Bezold-Bruecke shift a dimmer rendering needs and,
//     optionally, part of the way toward the hue the shipped clip gave the colour;
//   - optionally over-range colours are desaturated to the chroma the shipped clip left them;
//   - the 1e-4 pow() guard is dropped so black reaches 0.
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
// 15/16 is LUT addressing for the 16-texel point-sampled curve textures (segment = floor(15x),
// texel 15 only at x = 1), not an output domain: nothing is divided back out after the lookup.
static const float kCurveLutAddressScale = (UE3_TONEMAP_NUM_CURVE_SEGMENTS - 1.0f) / UE3_TONEMAP_NUM_CURVE_SEGMENTS;
// Common.usf redefines pow() as ClampedPow: pow(max(abs(x), 1e-4), y).
static const float kClampedPowGuard = 1e-4f;
static const float kRadToDeg = 180.0f / pi;
static const float kDegToRad = pi / 180.0f;

// ---------------------------------------------------------------------------
// pow() variants

// The shipped ClampedPow on a non-negative base. Floors linear at 1e-4, which the gamma encode
// carries to 0.01: the shipped #020202 black.
float3 clampedPow(float3 x, float3 e)
{
  return pow(max(x, kClampedPowGuard), e);
}

// Faithful Luma: the guard sits far below 8-bit resolution, so black reaches code 0.
float3 gradePow(float3 x, float3 e)
{
  return (cb.faithfulLuma != 0) ? pow(max(x, 1e-10f), e) : clampedPow(x, e);
}

// ---------------------------------------------------------------------------
// Range compression curves and hue restoration

static const float kMidGrey = 0.18f;

// Identity below the knee, then an extended-Reinhard shoulder that is C1 at the knee and hits 1.0 at white.
float3 shoulder(float3 x, float knee, float white)
{
  const float range = 1.0f - knee;
  const float whiteU = (white - knee) / range;
  const float3 u = max(x - knee, 0.0f) / range;
  const float3 r = u * (1.0f + u / (whiteU * whiteU)) / (1.0f + u);
  return min(min(x, knee) + range * r, 1.0f);
}

// Neutwo (RenoDX, Carlos Lopez, MIT): the Naka-Rushton / Reinhard-style curve x / sqrt(x^2 + 1)
// with display peak 1 and a white clip c, the input that lands exactly on the peak:
//   f(x) = c * x / sqrt(x^2 * (c^2 - 1) + c^2)
// Slope 1 at 0, near-identity at mid grey, asymptotic to x / sqrt(x^2 + 1) as c grows; c = 1 is
// the hard clip.
float3 neutwo(float3 x, float clip)
{
  const float cc = clip * clip;
  const float3 xx = x * x;
  return min(clip * x * rsqrt(xx * (cc - 1.0f) + cc), 1.0f);
}

// Power around mid grey applied to luminance (colour scaled by the ratio, so chromaticity is kept).
float3 applyContrast(float3 c, float contrast)
{
  const float y = dot(c, kLumaWeightsRec709);
  if (y <= 1e-6f)
  {
    return c;
  }
  const float yContrast = kMidGrey * pow(y / kMidGrey, contrast);
  return c * (yContrast / y);
}

// OKLab (a, b) of a linear BT.709 colour; hue is atan2(b, a). Scale invariant, so over-range input is fine.
float2 oklabChroma(float3 c)
{
  const float3 lms = float3(
    dot(c, float3(0.4122214708f, 0.5363325363f, 0.0514459929f)),
    dot(c, float3(0.2119034982f, 0.6806995451f, 0.1073969566f)),
    dot(c, float3(0.0883024619f, 0.2817188376f, 0.6299787005f)));
  const float3 lms3 = pow(max(lms, 0.0f), float3(1.0f / 3.0f));
  return float2(
    dot(lms3, float3(1.9779984951f, -2.4285922050f, 0.4505937099f)),
    dot(lms3, float3(0.0259040371f, 0.7827717662f, -0.8086757660f)));
}

// Direction of the Bezold-Bruecke shift for an OKLab hue in degrees: +1 turns counter-clockwise
// (red toward yellow, green toward cyan), -1 clockwise. Zero at the invariant hues (yellow 110,
// green 165, blue 264, purplish red 355) and largest midway between them.
float bezoldBruckeDirection(float hueDeg)
{
  const float d = frac((hueDeg - 355.0f) / 360.0f) * 360.0f; // degrees past the purplish-red invariant
  float s = sin(pi * d / 115.0f);                                     // toward yellow
  s = (d >= 115.0f) ? -sin(pi * (d - 115.0f) / 55.0f) : s;            // toward yellow
  s = (d >= 170.0f) ? sin(pi * (d - 170.0f) / 99.0f) : s;             // toward blue
  s = (d >= 269.0f) ? -sin(pi * (d - 269.0f) / 91.0f) : s;            // toward blue
  return s;
}

// Signed misalignment of a colour's OKLab hue from a target chroma direction; zero when they match.
float hueMismatch(float3 c, float2 target)
{
  const float2 ab = oklabChroma(c);
  return ab.x * target.y - ab.y * target.x;
}

// HSV hue in degrees. Continuous across sextants: at a tie both branches give the same value.
float rgbToHsvHue(float3 c)
{
  const float peak = max(c.r, max(c.g, c.b));
  const float low = min(c.r, min(c.g, c.b));
  const float d = peak - low;
  if (d <= 1e-7f)
  {
    return 0.0f;
  }
  float h;
  if (c.r >= c.g && c.r >= c.b)
  {
    h = frac(((c.g - c.b) / d) / 6.0f) * 6.0f; // (g - b) / d mod 6
  }
  else if (c.g >= c.b)
  {
    h = (c.b - c.r) / d + 2.0f;
  }
  else
  {
    h = (c.r - c.g) / d + 4.0f;
  }
  return 60.0f * h;
}

// Colour with the given HSV hue (degrees, any range), saturation and value. Every channel lies in
// [v * (1 - s), v], so the result is always in gamut.
float3 hsvToRgb(float hueDeg, float s, float v)
{
  const float h = frac(hueDeg / 360.0f) * 6.0f;
  const float c = v * s;
  const float x = c * (1.0f - abs(fmod(h, 2.0f) - 1.0f));
  const float m = v - c;
  float3 rgb;
  if (h < 1.0f)      rgb = float3(c, x, 0.0f);
  else if (h < 2.0f) rgb = float3(x, c, 0.0f);
  else if (h < 3.0f) rgb = float3(0.0f, c, x);
  else if (h < 4.0f) rgb = float3(0.0f, x, c);
  else if (h < 5.0f) rgb = float3(x, 0.0f, c);
  else               rgb = float3(c, 0.0f, x);
  return rgb + m;
}

// The curve's value (peak channel) and saturation (floor over peak) are kept, and its HSV hue is
// solved so the result's OKLab hue is the target. Within one sextant this is the reference
// shaders' middle-channel solve; solving the HSV hue instead keeps it continuous where two
// channels cross, where moving only the middle channel clamps at the floor and then jumps
// (banding on saturated colours, whose two lower channels are often nearly tied). OKLab hue is
// monotonic in HSV hue at fixed saturation and value, so a few secant steps suffice.
float3 restoreHue(float3 graded, float3 perChannel)
{
  const float peakOut = max(perChannel.r, max(perChannel.g, perChannel.b));
  const float lowOut = min(perChannel.r, min(perChannel.g, perChannel.b));

  const float2 chromaIn = oklabChroma(graded);
  const float chromaInLen = length(chromaIn);

  // Neutral input has no hue to restore; a neutral result cannot carry one.
  if ((peakOut - lowOut) < 1e-5f || chromaInLen < 1e-6f)
  {
    return perChannel;
  }

  const float hueIn = atan2(chromaIn.y, chromaIn.x) * kRadToDeg;
  const float lumaIn = max(dot(graded, kLumaWeightsRec709), 1e-4f);
  const float stops = max(log2(lumaIn / max(dot(perChannel, kLumaWeightsRec709), 1e-4f)), 0.0f);
  float turnDeg = cb.bezoldBruckePerStop * stops * bezoldBruckeDirection(hueIn);

  if (cb.hueReference == UE3_TONEMAP_HUE_REFERENCE_APPROVED_LOOK)
  {
    // Turn toward the hue the shipped clip gave the colour by the share of its luminance that
    // lay above display white, discounted by the chroma the clip kept.
    const float3 clipped = min(graded, 1.0f);
    const float2 chromaClip = oklabChroma(clipped);
    const float chromaClipLen = length(chromaClip);
    if (chromaClipLen > 1e-6f)
    {
      const float hueClip = atan2(chromaClip.y, chromaClip.x) * kRadToDeg;
      const float share = saturate((lumaIn - dot(clipped, kLumaWeightsRec709)) / lumaIn);
      const float chromaKept = saturate(chromaClipLen / max(chromaInLen, 1e-4f));
      float toClip = hueClip - hueIn;
      toClip -= 360.0f * round(toClip / 360.0f); // shortest way round
      turnDeg += share * chromaKept * toClip;
    }
  }

  const float turn = turnDeg * kDegToRad;
  const float2 target = float2(chromaIn.x * cos(turn) - chromaIn.y * sin(turn),
                               chromaIn.x * sin(turn) + chromaIn.y * cos(turn));

  const float v = peakOut;
  const float s = 1.0f - lowOut / peakOut;

  // Seeds: the curve's own hue and the input's hue, unwrapped next to it. At an exact tie the
  // two agree, and the secant needs a second distinct point.
  const float h1 = rgbToHsvHue(perChannel);
  float h0 = rgbToHsvHue(graded);
  h0 -= 360.0f * round((h0 - h1) / 360.0f);
  if (abs(h0 - h1) < 0.25f)
  {
    h0 = h1 + 1.0f;
  }

  float ha = h0;
  float fa = hueMismatch(hsvToRgb(ha, s, v), target);
  float hb = h1;
  float fb = hueMismatch(hsvToRgb(hb, s, v), target);

  [unroll]
  for (uint i = 0; i < 3; ++i)
  {
    const float d = fb - fa;
    float hn = (abs(d) > 1e-9f) ? hb - fb * (hb - ha) / d : hb;
    // Never further than the neighbouring secondaries; the corrections are a few degrees.
    hn = clamp(hn, h1 - 120.0f, h1 + 120.0f);
    const float fn = hueMismatch(hsvToRgb(hn, s, v), target);
    ha = hb;
    fa = fb;
    hb = hn;
    fb = fn;
  }

  return hsvToRgb(hb, s, v);
}

// Faithful Luma range limiting of the graded (unclipped) colour: the selected per-channel curve,
// then the hue solve on its result.
float3 applyRangeCompression(float3 gradedIn)
{
  float3 graded = gradedIn;
  float3 perChannel;

  if (cb.rangeCompression == UE3_TONEMAP_RANGE_COMPRESSION_FAITHFUL_LUMA_SHOULDER)
  {
    const float knee = clamp(cb.softClipKnee, 0.01f, 0.99f);
    const float white = max(cb.softClipWhite, 1.01f);
    perChannel = shoulder(graded, knee, white);
  }
  else
  {
    // Contrast is applied to the curve's input, and to the white clip so the same scene value
    // still lands on display white (as RenoDRT does).
    float clip = max(cb.neutwoWhiteClip, 1.0f);
    if (cb.neutwoContrast != 1.0f)
    {
      graded = applyContrast(graded, cb.neutwoContrast);
      clip = max(kMidGrey * pow(clip / kMidGrey, cb.neutwoContrast), 1.0f);
    }
    perChannel = neutwo(graded, clip);
  }

  float3 color = lerp(perChannel, restoreHue(graded, perChannel), cb.huePreservation);

  // Chroma the shipped clip would have left: saturation of min(graded, 1). Desaturate toward the
  // peak channel (hue-preserving, keeps the curve's brightness) until ours is no higher.
  // The blend above keeps the curve's peak and floor, so they still describe color here.
  if (cb.highlightDesaturation > 0.0f)
  {
    const float peakIn = max(graded.r, max(graded.g, graded.b));
    const float lowIn = min(graded.r, min(graded.g, graded.b));
    const float peakOut = max(perChannel.r, max(perChannel.g, perChannel.b));
    const float lowOut = min(perChannel.r, min(perChannel.g, perChannel.b));
    const float clipSaturation = 1.0f - min(lowIn, 1.0f) / max(min(peakIn, 1.0f), 1e-4f);
    const float outSaturation = 1.0f - lowOut / max(peakOut, 1e-4f);
    const float keepChroma = min(clipSaturation / max(outSaturation, 1e-4f), 1.0f);
    color = lerp(float3(peakOut), color, lerp(1.0f, keepChroma, cb.highlightDesaturation));
  }

  return color;
}

// ---------------------------------------------------------------------------
// Shipped tail: colour curves

// Maps a display value to the curve LUT sample coordinate, honoring the
// filter mode the game bound. Mirror's Edge point-filters its curve LUTs
// (exact piecewise segments: segment = floor(15x), texel 15 only at x = 1);
// point mode snaps to the center of the texel D3D9 would select.
float curveSampleCoord(float x)
{
  const float u = saturate(x * kCurveLutAddressScale);

  if (cb.curvePointSampling != 0)
  {
    const float texel = min(floor(u * UE3_TONEMAP_NUM_CURVE_SEGMENTS), UE3_TONEMAP_NUM_CURVE_SEGMENTS - 1.0f);
    return (texel + 0.5f) / UE3_TONEMAP_NUM_CURVE_SEGMENTS;
  }

  return u;
}

// Evaluates the game's baked 16-segment curves exactly like the PC
// TdToneMappingPixelShader: 16-texel LUT addressed with the 15/16 scale,
// then y = x * Ms + Bs per channel.
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

// ---------------------------------------------------------------------------

float3 applyUe3ToneMapping(float3 linearColor)
{
  const float3 sceneShadows = cb.sceneShadowsAndDesaturation.rgb;
  const float sceneDesaturation = cb.sceneShadowsAndDesaturation.a; // stored as (1 - desaturation), applied verbatim
  const float gammaInverse = cb.gammaColorScaleAndInverse.a;

  const float exposure = getExposure(InExposure, cb.enableAutoExposure, cb.exposureFactor);
  const float3 hdrColor = max(linearColor, 0.0f) * exposure;

  float3 color;

  if (cb.faithfulLuma != 0)
  {
    // The shipped grade, minus the clip: the range compression curve takes over the range limiting.
    const float3 graded = gradePow(max(hdrColor * cb.sceneInverseHighLights.rgb - sceneShadows, 0.0f), cb.sceneMidTones.rgb);

    color = applyRangeCompression(graded);
  }
  else
  {
    // Verbatim shipped TdToneMappingPixelShader: saturate(Color * Exposure) clips each channel
    // at exposed 1.0, then ClampedPow(x * SceneInverseHighLights - SceneShadows, SceneMidTones),
    // whose abs() folds a negative base back to positive and whose 1e-4 guard is the black floor.
    color = clampedPow(abs(saturate(hdrColor) * cb.sceneInverseHighLights.rgb - sceneShadows), cb.sceneMidTones.rgb);
  }

  // Desaturation + overlay (verbatim shipped math)
  const float scaledLuminance = dot(color, cb.sceneScaledLuminanceWeights.rgb);
  const float3 materialResult = cb.gammaOverlayColor.rgb + color * sceneDesaturation + scaledLuminance;

  // Display gamma encode (Mirror's Edge authored around DisplayGamma = 2.0)
  float3 outColor = gradePow(saturate(materialResult * cb.gammaColorScaleAndInverse.rgb), float3(gammaInverse));

  // Per-map colour curves in display space
  if (cb.applyColorCurves != 0)
  {
    outColor = applyMsBsColorCurves(outColor);
  }

  return saturate(outColor);
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
