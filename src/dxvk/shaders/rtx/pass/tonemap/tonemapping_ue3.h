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
#ifndef TONEMAPPING_UE3_H
#define TONEMAPPING_UE3_H

#include "rtx/utility/shader_types.h"

#define TONEMAPPING_UE3_COLOR_INPUT      0
#define TONEMAPPING_UE3_EXPOSURE_INPUT   1
#define TONEMAPPING_UE3_CURVE_K_INPUT    2
#define TONEMAPPING_UE3_CURVE_M_INPUT    3
#define TONEMAPPING_UE3_CONSTANTS_INPUT  4
#define TONEMAPPING_UE3_COLOR_OUTPUT     5

// Mirror's Edge bakes its per-map colour curves into 16 piecewise-linear
// segments (FCurveInfo.Ms[16]/Bs[16]) uploaded as two 16x1 LUT textures:
// ColorCurvesK texel = [Ms.r, Bs.r, Ms.g, Bs.g], ColorCurvesM texel = [Ms.b, Bs.b, -, -].
#define UE3_TONEMAP_NUM_CURVE_SEGMENTS   16

// Which hue the Faithful Luma shoulder aims an over-range colour at.
#define UE3_TONEMAP_HUE_REFERENCE_SCENE          0  // the scene's hue
#define UE3_TONEMAP_HUE_REFERENCE_APPROVED_LOOK  1  // turned toward the shipped clip's hue by the share of luminance the clip could not show

// Which curve limits the graded colour to the display range under Faithful Luma.
#define UE3_TONEMAP_RANGE_COMPRESSION_NEUTWO                  0  // x / sqrt(x^2 + 1) family, compresses from mid grey, asymptotic headroom
#define UE3_TONEMAP_RANGE_COMPRESSION_FAITHFUL_LUMA_SHOULDER  1  // identity below the knee, extended Reinhard to the white point

// Note: layout is kept to full 16-byte rows (vec4 or 4 scalars) so the C++ struct
// matches the shader constant buffer layout without packing surprises.
struct ToneMappingUe3Args {
  vec4 sceneShadowsAndDesaturation;   // rgb: SceneShadows, a: (1 - SceneDesaturation)
  vec4 sceneInverseHighLights;        // rgb: 1 / SceneHighLights
  vec4 sceneMidTones;                 // rgb: per-channel midtone grade exponents
  vec4 sceneScaledLuminanceWeights;   // rgb: LuminanceWeights * SceneDesaturation
  vec4 gammaColorScaleAndInverse;     // rgb: GammaColorScale, a: 1 / DisplayGamma
  vec4 gammaOverlayColor;             // rgb: engine overlay colour (fades)

  uint enableAutoExposure;
  float exposureFactor;               // exp2(exposure bias + user brightness EV), multiplied into the auto exposure
  uint applyColorCurves;              // 0 when no captured curves are available (identity)
  uint curvePointSampling;            // 1 when the game bound the curve LUTs with point filtering

  // Faithful Luma: the shipped grade without its per-channel clip, a range compression curve in
  // its place, the curve's hue shift replaced by an OKLab hue solve, and no pow() guard so black
  // reaches 0.
  uint faithfulLuma;                  // 0 = verbatim shipped TdToneMapping (hard clip at exposed 1.0, #020202 black floor)
  uint rangeCompression;              // UE3_TONEMAP_RANGE_COMPRESSION_*
  float softClipKnee;                 // shoulder: graded value where it starts (identity below)
  float softClipWhite;                // shoulder: graded value that reaches 1.0

  float huePreservation;              // 0 = the curve's per-channel hue shifts, 1 = the target hue
  float bezoldBruckePerStop;          // degrees of OKLab hue the target turns per stop the curve darkened the colour
  uint hueReference;                  // UE3_TONEMAP_HUE_REFERENCE_*
  float highlightDesaturation;        // 0..1: desaturate over-range colours to the chroma the shipped clip left them

  float neutwoWhiteClip;              // Neutwo: graded value that lands exactly on display white
  float neutwoContrast;               // Neutwo: power around mid grey (0.18) applied to luminance before the curve; 1 = none
  uint pad0;
  uint pad1;
};

#endif  // TONEMAPPING_UE3_H
