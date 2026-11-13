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
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../../test_utils.h"
#include "../../../src/util/util_matrix.h"
#include "../../../src/dxvk/rtx_render/rtx_matrix_helpers.h"

namespace dxvk {
  // The debug logging in rtx_matrix_helpers.h needs a Logger instance.
  Logger Logger::s_instance("test_depth_convention.log");
}

using namespace dxvk;

namespace {

  constexpr double kPi = 3.14159265358979323846;

  void expect(bool condition, const std::string& message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  // The tolerance is relative to the expected value, and absolute below 1.
  void expectNear(double actual, double expected, double tolerance, const std::string& label) {
    if (std::abs(actual - expected) > tolerance * std::max(std::abs(expected), 1.0)) {
      std::ostringstream message;
      message << label << ": expected " << expected << ", got " << actual;
      throw std::runtime_error(message.str());
    }
  }

  // Returns UE3's player projection, FPerspectiveMatrix(FOV, Width, Height, MinZ),
  // in the column-vector form the camera extraction produces.
  // Its far plane is infinite, and depth approaches 1 - Z_PRECISION with distance.
  Matrix4 ue3Projection(double fovXDegrees, float aspectRatio, float nearPlane) {
    constexpr float kZPrecision = 0.001f;
    const float scaleX = float(1.0 / std::tan(fovXDegrees * kPi / 360.0));

    Matrix4 projection;
    projection[0] = Vector4(scaleX, 0.0f, 0.0f, 0.0f);
    projection[1] = Vector4(0.0f, scaleX * aspectRatio, 0.0f, 0.0f);
    projection[2] = Vector4(0.0f, 0.0f, 1.0f - kZPrecision, 1.0f);
    projection[3] = Vector4(0.0f, 0.0f, -nearPlane * (1.0f - kZPrecision), 0.0f);
    return projection;
  }

  // Returns the projection with clip z = w - z, as a reversed depth patch makes it, which maps depth d to 1 - d.
  Matrix4 reverseDepth(const Matrix4& projection) {
    Matrix4 reversedFromStandardClip;
    reversedFromStandardClip[2] = Vector4(0.0f, 0.0f, -1.0f, 0.0f);
    reversedFromStandardClip[3] = Vector4(0.0f, 0.0f, 1.0f, 1.0f);
    return reversedFromStandardClip * projection;
  }

  struct ProjectionCase {
    const char* name;
    double fovXDegrees;
    float aspectRatio;
    float nearPlane;
  };

  const ProjectionCase kProjectionCases[] = {
    { "90 degrees 16:9", 90.0, 16.0f / 9.0f, 10.0f },
    { "110 degrees 21:9", 110.0, 21.0f / 9.0f, 10.0f },
    { "70 degrees 4:3 with a closer near plane", 70.0, 4.0f / 3.0f, 3.0f },
  };

  // Reversed depth detection reads isReverseZ, and linear view Z reuses the near and far planes.
  // The reversed projection therefore has to decompose like the standard one apart from that flag.
  void testReversedProjectionDecomposesLikeStandard() {
    std::cout << "  testReversedProjectionDecomposesLikeStandard" << std::endl;

    for (const ProjectionCase& projectionCase : kProjectionCases) {
      const std::string name = projectionCase.name;
      const Matrix4 standard = ue3Projection(projectionCase.fovXDegrees, projectionCase.aspectRatio, projectionCase.nearPlane);
      const Matrix4 reversed = reverseDepth(standard);

      DecomposeProjectionParams standardParams {};
      DecomposeProjectionParams reversedParams {};
      decomposeProjection(standard, standardParams);
      decomposeProjection(reversed, reversedParams);

      expect(!standardParams.isReverseZ, name + ": the standard projection is reported as reversed");
      expect(reversedParams.isReverseZ, name + ": the reversed projection is not detected");
      expect(standardParams.isLHS && reversedParams.isLHS, name + ": the projection is not reported as left-handed");

      expectNear(standardParams.nearPlane, projectionCase.nearPlane, 1e-4, name + ": standard near plane");
      expectNear(reversedParams.nearPlane, standardParams.nearPlane, 1e-4, name + ": reversed near plane");
      expectNear(reversedParams.farPlane, standardParams.farPlane, 1e-4, name + ": reversed far plane");
      expectNear(reversedParams.fov, standardParams.fov, 1e-5, name + ": reversed vertical FOV");
      expectNear(reversedParams.aspectRatio, standardParams.aspectRatio, 1e-5, name + ": reversed aspect ratio");

      // The D3D9 camera extraction rejects projections that fail these checks.
      expect(reversedParams.nearPlane > 0.0f && reversedParams.farPlane > reversedParams.nearPlane &&
             reversedParams.fov >= 0.001f && std::abs(reversedParams.shearX) <= 0.01f,
             name + ": the reversed projection fails the camera extraction's checks");
    }
  }

  // The motion vector pass unprojects raw depth through the camera's own projection.
  // The reversed projection therefore has to recover the same view positions from reversed depth.
  void testReversedDepthUnprojectsToTheSamePosition() {
    std::cout << "  testReversedDepthUnprojectsToTheSamePosition" << std::endl;

    const Matrix4 standardFloat = ue3Projection(90.0, 16.0f / 9.0f, 10.0f);
    const Matrix4d standard { standardFloat };
    const Matrix4d reversed { reverseDepth(standardFloat) };
    const Matrix4d reversedInverse = inverse(reversed);

    const Vector4d viewPositions[] = {
      Vector4d(0.0, 0.0, 10.5, 1.0),
      Vector4d(3.0, -2.0, 100.0, 1.0),
      Vector4d(-150.0, 80.0, 1000.0, 1.0),
      Vector4d(2000.0, 500.0, 50000.0, 1.0),
    };

    for (const Vector4d& viewPosition : viewPositions) {
      const Vector4d standardClip = standard * viewPosition;
      const Vector4d reversedClip = reversed * viewPosition;
      const double standardDepth = standardClip.z / standardClip.w;
      const double reversedDepth = reversedClip.z / reversedClip.w;

      expectNear(reversedDepth, 1.0 - standardDepth, 1e-9, "reversed depth");
      expect(reversedDepth > 0.0 && reversedDepth < 1.0, "reversed depth outside the depth range");

      const Vector4d ndc(reversedClip.x / reversedClip.w, reversedClip.y / reversedClip.w, reversedDepth, 1.0);
      const Vector4d unprojected = reversedInverse * ndc;

      expectNear(unprojected.x / unprojected.w, viewPosition.x, 1e-6, "unprojected view x");
      expectNear(unprojected.y / unprojected.w, viewPosition.y, 1e-6, "unprojected view y");
      expectNear(unprojected.z / unprojected.w, viewPosition.z, 1e-6, "unprojected view z");
    }
  }

}

int main() {
  try {
    std::cout << std::endl << "Begin depth convention tests" << std::endl;
    testReversedProjectionDecomposesLikeStandard();
    testReversedDepthUnprojectsToTheSamePosition();
    std::cout << "All depth convention tests passed" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
