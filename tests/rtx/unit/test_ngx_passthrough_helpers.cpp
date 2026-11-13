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
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../test_utils.h"
#include "../../../src/dxvk/rtx_render/rtx_ngx_passthrough_helpers.h"

using namespace dxvk;

namespace {

  void expect(bool condition, const std::string& message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  void expectNear(double actual, double expected, double tolerance, const std::string& label) {
    if (std::abs(actual - expected) > tolerance) {
      std::ostringstream message;
      message << label << ": expected " << expected << ", got " << actual;
      throw std::runtime_error(message.str());
    }
  }

  // UE3's GPU skin shader declares 75 bones of three registers each.
  constexpr uint32_t kDeclaredRegisters = 225;
  // Matches the tolerance the velocity capture holds a bone translation to.
  constexpr float kTranslationTolerance = 0.05f;
  const Vector3 kNoShift(0.0f, 0.0f, 0.0f);

  // Returns a declared-size palette of identity rotations with the given bone translations.
  // The registers past those bones hold leftoverValue, as if an earlier draw had uploaded them.
  std::vector<Vector4> makePalette(const std::vector<Vector3>& translations, float leftoverValue = 0.0f) {
    std::vector<Vector4> palette(kDeclaredRegisters, Vector4(leftoverValue));
    for (size_t bone = 0; bone < translations.size(); bone++) {
      palette[bone * 3 + 0] = Vector4(1.0f, 0.0f, 0.0f, translations[bone].x);
      palette[bone * 3 + 1] = Vector4(0.0f, 1.0f, 0.0f, translations[bone].y);
      palette[bone * 3 + 2] = Vector4(0.0f, 0.0f, 1.0f, translations[bone].z);
    }
    return palette;
  }

  // Returns the pose with every bone moved by the same amount.
  std::vector<Vector3> movePose(std::vector<Vector3> pose, const Vector3& motion) {
    for (Vector3& translation : pose) {
      translation = translation + motion;
    }
    return pose;
  }

  // Returns a horizontal view direction at the given yaw, scaled to the given length.
  Vector3 viewDirection(double yawDegrees, float length = 1.0f) {
    const double yaw = yawDegrees * 3.14159265358979323846 / 180.0;
    return Vector3(float(std::cos(yaw)) * length, 0.0f, float(std::sin(yaw)) * length);
  }

  // UE3 uploads only the bones a mesh section uses. The run of registers stamped with the draw's serial is that
  // section's palette, and every register past it holds what an earlier draw left.
  void testUploadedBoneRegisters() {
    std::cout << "  testUploadedBoneRegisters" << std::endl;

    constexpr uint32_t kStampCount = 256;
    constexpr uint32_t kPaletteRegister = 30;
    constexpr uint32_t kDrawSerial = 7;

    const auto stampRun = [&](uint32_t firstRegister, uint32_t registerCount, uint32_t serial) {
      std::vector<uint32_t> stamps(kStampCount, 0u);
      for (uint32_t reg = firstRegister; reg < std::min(firstRegister + registerCount, kStampCount); reg++) {
        stamps[reg] = serial;
      }
      return stamps;
    };
    const auto countUploaded = [&](const std::vector<uint32_t>& stamps, uint32_t paletteRegister) {
      return ngxPassthrough::countUploadedBoneRegisters(stamps.data(), uint32_t(stamps.size()), paletteRegister,
                                                        kDeclaredRegisters, kDrawSerial);
    };

    // A 20-bone mesh section uploads 60 registers.
    expect(countUploaded(stampRun(kPaletteRegister, 60, kDrawSerial), kPaletteRegister) == 60, "a 20-bone upload");
    // A run that ends partway through a bone keeps only the whole bones.
    expect(countUploaded(stampRun(kPaletteRegister, 61, kDrawSerial), kPaletteRegister) == 60, "a partial last bone");
    // Registers stamped for the previous draw were not uploaded for this one.
    expect(countUploaded(stampRun(kPaletteRegister, 60, kDrawSerial - 1), kPaletteRegister) == 0, "the previous draw's upload");
    // Registers that were never written count for nothing.
    expect(countUploaded(std::vector<uint32_t>(kStampCount, 0u), kPaletteRegister) == 0, "no upload");
    // An upload that starts past the palette's first register is not this palette's.
    expect(countUploaded(stampRun(kPaletteRegister + 3, 60, kDrawSerial), kPaletteRegister) == 0, "an upload that misses the first bone");
    // An upload that continues past the declared palette stops at its end.
    expect(countUploaded(stampRun(0, kStampCount, kDrawSerial), kPaletteRegister) == kDeclaredRegisters, "an upload past the declared palette");
    // A palette declared past the last stamp stops there, which here leaves two whole bones.
    expect(countUploaded(stampRun(250, 6, kDrawSerial), 250) == 6, "a palette at the end of the stamps");
  }

  // A palette is compared only within the registers its own upload wrote, because the rest holds whatever the
  // previous skinned draw uploaded there.
  void testPaletteChange() {
    std::cout << "  testPaletteChange" << std::endl;

    const std::vector<Vector3> pose = { Vector3(0.0f, 0.0f, 90.0f), Vector3(10.0f, 0.0f, 140.0f),
                                        Vector3(-10.0f, 0.0f, 140.0f), Vector3(0.0f, 5.0f, 160.0f) };
    const uint32_t comparedRegisters = uint32_t(pose.size()) * ngxPassthrough::kRegistersPerBone;
    const std::vector<Vector4> palette = makePalette(pose, 1.0f);

    const auto changedFrom = [&](const std::vector<Vector4>& cachedBones, uint32_t registers, const Vector3& shift) {
      return ngxPassthrough::bonePaletteChanged(palette.data(), cachedBones.data(), registers, shift, kTranslationTolerance);
    };

    // A repeat draw of the section uploads the same palette again.
    expect(!changedFrom(palette, comparedRegisters, kNoShift), "an identical palette");

    // Another character drawn in between leaves different registers past the uploaded bones.
    const std::vector<Vector4> otherLeftovers = makePalette(pose, 7.0f);
    expect(!changedFrom(otherLeftovers, comparedRegisters, kNoShift), "different leftovers past the uploaded bones");
    expect(changedFrom(otherLeftovers, kDeclaredRegisters, kNoShift), "leftovers compared as part of the palette");

    // A rotation row that moved by a fraction of a degree.
    std::vector<Vector4> rotated = palette;
    rotated[3].y += 1e-3f;
    expect(changedFrom(rotated, comparedRegisters, kNoShift), "a rotated bone");

    // A translation that moved by a unit, and one that moved within the tolerance.
    std::vector<Vector3> movedPose = pose;
    movedPose[2].z += 1.0f;
    expect(changedFrom(makePalette(movedPose, 1.0f), comparedRegisters, kNoShift), "a moved bone");
    movedPose = pose;
    movedPose[2].z += 0.01f;
    expect(!changedFrom(makePalette(movedPose, 1.0f), comparedRegisters, kNoShift), "a bone moved within the tolerance");

    // The cached palette is the current one before the scene-wide transform offset moved it.
    // A component-space palette that never took the offset is unchanged as well.
    // Moving by anything else is motion.
    const Vector3 sceneShift(120.0f, -40.0f, 8.0f);
    expect(!changedFrom(makePalette(movePose(pose, kNoShift - sceneShift), 1.0f), comparedRegisters, sceneShift),
           "a palette that took the scene offset");
    expect(!changedFrom(palette, comparedRegisters, sceneShift), "a component-space palette beside a scene offset");
    expect(changedFrom(makePalette(movePose(pose, kNoShift - sceneShift * 2.0f), 1.0f), comparedRegisters, sceneShift),
           "a palette that moved by twice the scene offset");
  }

  // Pairing continues the instance whose pose is nearest. Two guards that a cutscene places at one spot share the
  // mesh and the LocalToWorld, so their palettes are all that tells their histories apart.
  void testPoseDelta() {
    std::cout << "  testPoseDelta" << std::endl;

    const std::vector<Vector3> standing = { Vector3(0.0f, 0.0f, 90.0f), Vector3(20.0f, 0.0f, 140.0f),
                                            Vector3(-20.0f, 0.0f, 140.0f), Vector3(0.0f, 0.0f, 165.0f) };
    const uint32_t comparedRegisters = uint32_t(standing.size()) * ngxPassthrough::kRegistersPerBone;
    const std::vector<Vector4> standingPalette = makePalette(standing);

    const auto poseDelta = [&](const std::vector<Vector4>& bones, const std::vector<Vector4>& cachedBones) {
      return ngxPassthrough::bonePoseDelta(bones.data(), cachedBones.data(), comparedRegisters);
    };

    // The same pose lies at no distance.
    expectNear(poseDelta(standingPalette, standingPalette), 0.0, 1e-6, "an identical pose");

    // One of four bones moved by (3, 4, 0), which is 5 units, so the mean over the bones is 1.25.
    std::vector<Vector3> oneBoneMoved = standing;
    oneBoneMoved[1] = oneBoneMoved[1] + Vector3(3.0f, 4.0f, 0.0f);
    expectNear(poseDelta(makePalette(oneBoneMoved), standingPalette), 1.25, 1e-5, "one moved bone");

    // Fewer compared registers than one bone leave nothing to measure.
    expect(std::isinf(ngxPassthrough::bonePoseDelta(standingPalette.data(), standingPalette.data(), 2)),
           "a comparison without a whole bone");

    // Guard A stands while guard B crouches 60 units away in mesh space, and each moves slightly between frames.
    const std::vector<Vector3> crouching = { Vector3(60.0f, 0.0f, 50.0f), Vector3(75.0f, 10.0f, 80.0f),
                                             Vector3(45.0f, 10.0f, 80.0f), Vector3(60.0f, 5.0f, 105.0f) };
    const std::vector<Vector4> guardAPrevious = standingPalette;
    const std::vector<Vector4> guardACurrent = makePalette(movePose(standing, Vector3(1.5f, 0.0f, 0.0f)));
    const std::vector<Vector4> guardBPrevious = makePalette(crouching);
    const std::vector<Vector4> guardBCurrent = makePalette(movePose(crouching, Vector3(0.0f, -1.5f, 0.0f)));

    expectNear(poseDelta(guardACurrent, guardAPrevious), 1.5, 1e-5, "guard A's own motion");
    expect(poseDelta(guardACurrent, guardAPrevious) < poseDelta(guardACurrent, guardBPrevious),
           "guard A continues guard B's history");
    expect(poseDelta(guardBCurrent, guardBPrevious) < poseDelta(guardBCurrent, guardAPrevious),
           "guard B continues guard A's history");

    // Guard B is drawn after guard A has claimed their shared placement this frame.
    // Its palette differs from the one guard A uploaded, so it is not a repeat draw of guard A.
    expect(ngxPassthrough::bonePaletteChanged(guardBCurrent.data(), guardACurrent.data(), comparedRegisters, kNoShift,
                                              kTranslationTolerance),
           "guard B read as a repeat draw of guard A");
  }

  // A cutscene cut that turns the camera resets history, while turning within the threshold does not.
  void testViewTurn() {
    std::cout << "  testViewTurn" << std::endl;

    const Vector3 forward = viewDirection(0.0);

    expect(!ngxPassthrough::viewTurnedBeyond(forward, forward, 45.0f), "a camera that did not turn");
    expect(!ngxPassthrough::viewTurnedBeyond(viewDirection(30.0), forward, 45.0f), "a 30 degree turn");
    expect(!ngxPassthrough::viewTurnedBeyond(viewDirection(44.5), forward, 45.0f), "a turn just inside the threshold");
    expect(ngxPassthrough::viewTurnedBeyond(viewDirection(45.5), forward, 45.0f), "a turn just past the threshold");
    expect(ngxPassthrough::viewTurnedBeyond(viewDirection(150.0), forward, 45.0f), "a reverse shot");
    expect(ngxPassthrough::viewTurnedBeyond(viewDirection(180.0), forward, 45.0f), "a 180 degree turn");

    // The view matrix axes need not have unit length.
    expect(ngxPassthrough::viewTurnedBeyond(viewDirection(60.0, 250.0f), viewDirection(0.0, 0.5f), 45.0f),
           "a large turn between scaled directions");
    expect(!ngxPassthrough::viewTurnedBeyond(viewDirection(30.0, 250.0f), viewDirection(0.0, 0.5f), 45.0f),
           "a small turn between scaled directions");

    // 0 or less disables the test, and no turn can exceed a threshold past 180 degrees.
    expect(!ngxPassthrough::viewTurnedBeyond(viewDirection(180.0), forward, 0.0f), "a disabled test");
    expect(!ngxPassthrough::viewTurnedBeyond(viewDirection(180.0), forward, -10.0f), "a negative threshold");
    expect(!ngxPassthrough::viewTurnedBeyond(viewDirection(120.0), forward, 270.0f), "a threshold past 180 degrees");

    // A direction without length has no angle to compare.
    expect(!ngxPassthrough::viewTurnedBeyond(Vector3(0.0f, 0.0f, 0.0f), forward, 45.0f), "a direction without length");
  }

}

int main() {
  try {
    std::cout << std::endl << "Begin NGX passthrough helper tests" << std::endl;
    testUploadedBoneRegisters();
    testPaletteChange();
    testPoseDelta();
    testViewTurn();
    std::cout << "All NGX passthrough helper tests passed" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
