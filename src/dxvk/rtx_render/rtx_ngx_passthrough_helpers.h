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
#include <limits>

#include "../../util/util_vector.h"

// Holds the NGX passthrough's skinned velocity and camera cut decisions that need neither a device
// nor game state, so the unit tests can cover them.

namespace dxvk {

  namespace ngxPassthrough {

    // UE3 uploads each bone as three rows of a 3x4 matrix, and each row carries one axis of the bone's translation in w.
    constexpr uint32_t kRegistersPerBone = 3;

    // Returns how many registers from the start of a bone palette the draw's own upload wrote.
    // A register counts while its write stamp equals the draw's serial.
    // The count is rounded down to whole bones, and stays within both the declared palette and the stamp array.
    inline uint32_t countUploadedBoneRegisters(const uint32_t* pWriteStamps, uint32_t stampCount,
                                               uint32_t paletteRegister, uint32_t declaredRegisters,
                                               uint32_t drawSerial) {
      uint32_t uploadedRegisters = 0;
      while (uploadedRegisters < declaredRegisters && paletteRegister + uploadedRegisters < stampCount &&
             pWriteStamps[paletteRegister + uploadedRegisters] == drawSerial) {
        uploadedRegisters++;
      }
      return uploadedRegisters - uploadedRegisters % kRegistersPerBone;
    }

    // Returns true when a bone palette differs from a cached one within the compared registers.
    // A translation counts as unchanged when it matches either as it is or shifted by expectedTranslationShift.
    inline bool bonePaletteChanged(const Vector4* pBones, const Vector4* pCachedBones, uint32_t comparedRegisters,
                                   const Vector3& expectedTranslationShift, float translationTolerance) {
      // The rotation rows hold unit-scale values, so they are held to a much tighter tolerance than translations
      constexpr float kRotationTolerance = 1e-5f;

      for (uint32_t reg = 0; reg < comparedRegisters; reg++) {
        const Vector4 delta = pBones[reg] - pCachedBones[reg];
        if (std::abs(delta.x) > kRotationTolerance || std::abs(delta.y) > kRotationTolerance ||
            std::abs(delta.z) > kRotationTolerance) {
          return true;
        }

        const float axisShift = expectedTranslationShift[reg % kRegistersPerBone];
        if (std::abs(delta.w) > translationTolerance && std::abs(delta.w - axisShift) > translationTolerance) {
          return true;
        }
      }
      return false;
    }

    // Returns the mean distance between the bone translations of two palettes within the compared registers.
    // Characters that share a mesh and a placement differ only in this.
    // Returns infinity when the compared registers hold no whole bone.
    inline float bonePoseDelta(const Vector4* pBones, const Vector4* pCachedBones, uint32_t comparedRegisters) {
      const uint32_t comparedBones = comparedRegisters / kRegistersPerBone;
      if (comparedBones == 0) {
        return std::numeric_limits<float>::infinity();
      }

      float distanceSum = 0.0f;
      for (uint32_t bone = 0; bone < comparedBones; bone++) {
        const uint32_t reg = bone * kRegistersPerBone;
        distanceSum += length(Vector3(pBones[reg + 0].w - pCachedBones[reg + 0].w,
                                      pBones[reg + 1].w - pCachedBones[reg + 1].w,
                                      pBones[reg + 2].w - pCachedBones[reg + 2].w));
      }
      return distanceSum / float(comparedBones);
    }

    // Returns true when the view direction turned by more than maxAngleDegrees between two frames.
    // Returns false when maxAngleDegrees is 0 or less, or when either direction has no length.
    inline bool viewTurnedBeyond(const Vector3& forward, const Vector3& previousForward, float maxAngleDegrees) {
      if (maxAngleDegrees <= 0.0f) {
        return false;
      }

      const float lengthProduct = length(forward) * length(previousForward);
      if (lengthProduct <= 1e-6f) {
        return false;
      }

      // No turn exceeds 180 degrees, and past it the cosine would fold back onto a smaller angle
      constexpr float kRadiansPerDegree = 3.14159265358979f / 180.0f;
      const float maxAngleRadians = std::min(maxAngleDegrees, 180.0f) * kRadiansPerDegree;
      return dot(forward, previousForward) < std::cos(maxAngleRadians) * lengthProduct;
    }

  }

}
