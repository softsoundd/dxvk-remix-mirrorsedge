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
#ifndef TONEMAPPING_UE3_EXPOSURE_H
#define TONEMAPPING_UE3_EXPOSURE_H

#include "rtx/utility/shader_types.h"

#define TONEMAPPING_UE3_EXPOSURE_COLOR_INPUT  0
#define TONEMAPPING_UE3_EXPOSURE_OUTPUT       1

// The meter samples the image on a fixed grid: one workgroup of 16x16 threads, each thread
// covering a 16x16 block of grid points, 65536 samples in total.
#define UE3_EXPOSURE_METER_THREADS_PER_AXIS   16
#define UE3_EXPOSURE_METER_SAMPLES_PER_THREAD 16
#define UE3_EXPOSURE_METER_GRID               (UE3_EXPOSURE_METER_THREADS_PER_AXIS * UE3_EXPOSURE_METER_SAMPLES_PER_THREAD)

// Mirror's Edge (TdToneMapExposure) exposure model run on Remix's radiance. The state texel
// holds E as the game's shader does (before Scene_ExposureManual), and persists across frames.
struct ToneMappingUe3ExposureArgs {
  float sceneScale;          // Remix linear radiance -> the game's scene units (exp2 of the calibration EV)
  float deltaTimeSeconds;    // frame time for the adaptation
  float exposureLow;         // Scene_ExposureLow: clamp on sqrt(E), as uploaded
  float exposureHigh;        // Scene_ExposureHigh: clamp on sqrt(E), as uploaded

  float speedToLight;        // stops per second while the exposure falls (already scaled by the level's speed factor)
  float speedToDark;         // stops per second while it rises
  float transitionStops;     // distance from the target inside which the movement eases in exponentially
  uint resetHistory;         // 1 = start on the target (camera cut / history reset)
};

#endif  // TONEMAPPING_UE3_EXPOSURE_H
