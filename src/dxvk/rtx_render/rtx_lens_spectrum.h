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

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "../../util/util_vector.h"

namespace dxvk {

  // Wyman, Sloan and Shirley 2013, "Simple Analytic Approximations to the CIE XYZ Color Matching Functions":
  // the multi-lobe fit of the CIE 1931 2 degree observer.
  inline Vector3 cieXyzFit(float lambdaNm) {
    const auto lobe = [lambdaNm](float mean, float inverseSigmaLow, float inverseSigmaHigh) {
      const float t = (lambdaNm - mean) * (lambdaNm < mean ? inverseSigmaLow : inverseSigmaHigh);
      return std::exp(-0.5f * t * t);
    };

    return Vector3(
      0.362f * lobe(442.0f, 0.0624f, 0.0374f) + 1.056f * lobe(599.8f, 0.0264f, 0.0323f) - 0.065f * lobe(501.1f, 0.0490f, 0.0382f),
      0.821f * lobe(568.8f, 0.0213f, 0.0247f) + 0.286f * lobe(530.9f, 0.0613f, 0.0322f),
      1.217f * lobe(437.0f, 0.0845f, 0.0278f) + 0.681f * lobe(459.0f, 0.0385f, 0.0725f));
  }

  // Wavelengths spread evenly over 400 to 700 nm, for the lens effects that sum a few of them into Rec.709. Each
  // sample's response is clamped to the gamut and each channel's weights sum to one, so a flat spectrum stays white.
  inline void computeVisibleSpectrumSamples(uint32_t count, float* pWavelengthsNm, Vector3* pWeights) {
    Vector3 sums(0.0f);

    for (uint32_t i = 0; i < count; i++) {
      const float lambda = 400.0f + (float(i) + 0.5f) * 300.0f / float(count);
      const Vector3 xyz = cieXyzFit(lambda);

      pWavelengthsNm[i] = lambda;
      pWeights[i] = Vector3(
        std::max(3.2406f * xyz.x - 1.5372f * xyz.y - 0.4986f * xyz.z, 0.0f),
        std::max(-0.9689f * xyz.x + 1.8758f * xyz.y + 0.0415f * xyz.z, 0.0f),
        std::max(0.0557f * xyz.x - 0.2040f * xyz.y + 1.0570f * xyz.z, 0.0f));
      sums = sums + pWeights[i];
    }

    for (uint32_t i = 0; i < count; i++) {
      pWeights[i] = Vector3(
        pWeights[i].x / std::max(sums.x, 1e-6f),
        pWeights[i].y / std::max(sums.y, 1e-6f),
        pWeights[i].z / std::max(sums.z, 1e-6f));
    }
  }

}
