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
#pragma once

// mat4 and the float2 / uint2 aliases the argument structs below are built from. Previously
// picked up transitively, which only held while every including translation unit happened to
// pull it in first.
#include "rtx/utility/shader_types.h"

#define POST_FX_MOTION_BLUR_PREFILTER_PRIMARY_SURFACE_FLAGS_INPUT           0
#define POST_FX_MOTION_BLUR_PREFILTER_PRIMARY_SURFACE_FLAGS_FILTERED_OUTPUT 1

#define POST_FX_MOTION_BLUR_PRIMARY_SCREEN_SPACE_MOTION_INPUT 0
#define POST_FX_MOTION_BLUR_PRIMARY_SURFACE_FLAGS_INPUT       1
#define POST_FX_MOTION_BLUR_PRIMARY_LINEAR_VIEW_Z_INPUT       2
#define POST_FX_MOTION_BLUR_BLUE_NOISE_TEXTURE_INPUT          3
#define POST_FX_MOTION_BLUR_INPUT                             4
#define POST_FX_MOTION_BLUR_OUTPUT                            5
#define POST_FX_MOTION_BLUR_NEAREST_SAMPLER                   6
#define POST_FX_MOTION_BLUR_LINEAR_SAMPLER                    7

// Cinematic motion blur (Guertin/McGuire/Nowrouzezahrai, HPG 2014). Four passes:
// setup -> TileMax (separable) -> NeighborMax -> gather.
#define POST_FX_MB_CINE_SETUP_MOTION_VECTOR_INPUT      0
#define POST_FX_MB_CINE_SETUP_SURFACE_FLAGS_INPUT      1
#define POST_FX_MB_CINE_SETUP_LINEAR_VIEW_Z_INPUT      2
#define POST_FX_MB_CINE_SETUP_PREV_MOTION_VECTOR_INPUT 3
#define POST_FX_MB_CINE_SETUP_VELOCITY_DEPTH_OUTPUT    4
#define POST_FX_MB_CINE_SETUP_CURVATURE_OUTPUT         5
#define POST_FX_MB_CINE_SETUP_CONSTANTS                6

#define POST_FX_MB_CINE_TILEMAX_INPUT  0
#define POST_FX_MB_CINE_TILEMAX_OUTPUT 1

#define POST_FX_MB_CINE_NEIGHBORMAX_INPUT  0
#define POST_FX_MB_CINE_NEIGHBORMAX_OUTPUT 1

#define POST_FX_MB_CINE_GATHER_VELOCITY_DEPTH_INPUT 0
#define POST_FX_MB_CINE_GATHER_CURVATURE_INPUT      1
#define POST_FX_MB_CINE_GATHER_TILE_MAX_INPUT       2
#define POST_FX_MB_CINE_GATHER_NEIGHBOR_MAX_INPUT   3
#define POST_FX_MB_CINE_GATHER_COLOR_INPUT          4
#define POST_FX_MB_CINE_GATHER_OUTPUT               5
#define POST_FX_MB_CINE_GATHER_LINEAR_SAMPLER       6

// Tile textures are allocated for the smallest supported tile so that changing the blur
// radius at runtime never reallocates; larger tiles dispatch into a sub-rect.
#define POST_FX_MB_TILE_SIZE_MIN    16
#define POST_FX_MB_MAX_SAMPLE_COUNT 64

// Sample distribution between the dominant and centre directions (paper section 4.1)
#define POST_FX_MB_DIRECTION_SPLIT_DOMINANT 0
#define POST_FX_MB_DIRECTION_SPLIT_EVEN     1
#define POST_FX_MB_DIRECTION_SPLIT_VARIANCE 2

#define POST_FX_MB_DEBUG_OFF          0
#define POST_FX_MB_DEBUG_VELOCITY     1
#define POST_FX_MB_DEBUG_TILE_MAX     2
#define POST_FX_MB_DEBUG_NEIGHBOR_MAX 3
#define POST_FX_MB_DEBUG_VARIANCE     4
// Fraction of the final colour that came from somewhere other than the pixel itself. Runs
// the full gather rather than short circuiting, so it localises where the filter is actually
// moving colour and by how much.
#define POST_FX_MB_DEBUG_BLUR_AMOUNT  5

#define POST_FX_INPUT  0
#define POST_FX_OUTPUT 1

#define POST_FX_HIGHLIGHT_INPUT                       0
#define POST_FX_HIGHLIGHT_OBJECT_PICKING_INPUT        1
#define POST_FX_HIGHLIGHT_PRIMARY_CONE_RADIUS_INPUT   2
#define POST_FX_HIGHLIGHT_OUTPUT                      3
#define POST_FX_HIGHLIGHT_VALUES                      4

#define POST_FX_TILE_SIZE 8

struct PostFxArgs {
  // Display image information
  uint2  imageSize;
  float2 invImageSize;

  // Camera Resolution
  float2 invMainCameraResolution;
  float2 inputOverOutputViewSize;

  // Post Fx Attributes
  // Motion Blur
  uint   motionBlurSampleCount;
  float  blurDiameterFraction;
  bool   enableMotionBlurNoiseSample;
  float  motionBlurMinimumVelocityThresholdInPixel;

  // Chromatic Aberration
  float2 chromaticAberrationScale;
  float  chromaticCenterAttenuationAmount;
  float  exposureFraction;
  
  // Vignette
  float  vignetteIntensity;
  float  vignetteRadius;
  float  vignetteSoftness;
  uint   frameIdx;

  float  motionBlurDynamicDeduction;
  bool   enableMotionBlurEmissive;
  float  jitterStrength;
  float  motionBlurDlfgDeduction;
};

struct PostFxMotionBlurPrefilterArgs {
  uint2 imageSize;
  int2  pixelStep;
};

// Cinematic motion blur setup pass. Passed as a uniform buffer rather than push constants:
// the two curve matrices alone consume the entire 128 byte push constant budget.
// Note the `uint` booleans - there is no bool typedef on the C++ side of shader_types.h,
// so a C++ bool (1 byte) would not match Slang's 4 byte bool without incidental padding.
struct PostFxMotionBlurCineSetupArgs {
  // Map current frame unjittered NDC (xy in [-1,1], z = hardware-equivalent depth, w = 1)
  // to clip space at half a frame back / forward. Used to fit the sample path curvature.
  mat4 curveToPrevHalfClip;
  mat4 curveToNextHalfClip;

  uint2  imageSize;                // display resolution, this pass's output extent
  float2 invImageSize;

  float2 inputOverOutputViewSize;  // gbuffer resolution / display resolution
  float2 gbufferResolution;

  float  shutterFraction;          // shutterAngle / 360
  float  maxBlurRadiusPixels;
  float  dynamicDeduction;
  float  dlfgDeduction;

  float  nearPlane;
  float  farPlane;
  float  minVelocityPixels;
  uint   enableEmissive;

  uint   enableCurvedPaths;
  uint   cameraValid;              // curve matrices are usable this frame
  uint   enableObjectCurvature;    // previous frame motion vectors are bound and valid
  uint   pad0;
};

// Shared by both separable TileMax passes and by NeighborMax (which leaves pixelStep and
// tileSize unused and sets srcSize == dstSize == the tile count).
struct PostFxMotionBlurCineTileArgs {
  uint2 srcSize;
  uint2 dstSize;
  int2  pixelStep;
  uint  tileSize;
  uint  pad0;
};

struct PostFxMotionBlurCineGatherArgs {
  uint2  imageSize;
  float2 invImageSize;

  uint2  tileCount;
  uint   tileSize;
  uint   sampleCount;              // N

  float  centerWeightBias;         // k, paper section 4.3
  float  jitterScale;              // phi, paper section 4.5
  float  tileBlendSlope;           // tau, paper section 4.2
  float  gammaThreshold;           // gamma, paper Eq. 1

  uint   frameIdx;
  uint   directionSplit;
  uint   enableCurvedPaths;
  float  minVelocityPixels;

  uint   debugView;
  uint   pad0;
  uint   pad1;
  uint   pad2;
};

#define POST_FX_HIGHLIGHTING_MAX_VALUES_POW 14
#define POST_FX_HIGHLIGHTING_MAX_VALUES     (1 << POST_FX_HIGHLIGHTING_MAX_VALUES_POW)
#define POST_FX_HIGHLIGHTING_INVALID_VALUE  0xFFFFFFFF

struct PostFxHighlightingArgs
{
  // Display image information
  uint2 imageSize;
  // If need to highlight an object under this pixel
  int2  pixel;
  // Highlighting params
  uint  desaturateNonHighlighted;
  float timeSinceStartMS;
  uint  highlightColorPacked;
  uint  valuesToHighlightCountPow;
};
