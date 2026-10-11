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

#include <cstdint>
#include <iterator>

namespace dxvk {

  // One spherical interface of a lens, listed from the object side. The medium behind it fills the gap to the next.
  struct LensSurface {
    // Radius of curvature in mm, positive when its centre lies toward the image. 0 is flat.
    double radius;
    // Axial distance to the next surface in mm. The last surface's distance to the sensor follows from focusing.
    double thickness;
    // Refractive index at the helium d line (587.6 nm) and Abbe number of the medium behind the surface. 1 and 0
    // for air.
    double indexD;
    double abbe;
    double clearDiameter;
    bool isStop;
  };

  struct LensPrescription {
    const char* name;
    const LensSurface* surfaces;
    uint32_t surfaceCount;
  };

  // Tronnier's f/2 double Gauss, US 2,673,491, scaled to 50 mm as in pbrt's dgauss.50mm.dat (from Smith, Modern Lens
  // Design, p. 312), with the patent's indices and Abbe numbers.
  inline constexpr LensSurface kDoubleGauss50mm[] = {
    {  29.475, 3.76,  1.67125, 47.1, 25.2, false },
    {  84.83,  0.12,  1.0,      0.0, 25.2, false },
    {  19.275, 4.025, 1.67125, 47.1, 23.0, false },
    {  40.77,  3.275, 1.69842, 30.1, 23.0, false },
    {  12.75,  5.705, 1.0,      0.0, 18.0, false },
    {   0.0,   4.5,   1.0,      0.0, 17.1, true  },
    { -14.495, 1.18,  1.60266, 38.4, 17.0, false },
    {  40.77,  6.065, 1.65953, 57.0, 20.0, false },
    { -20.385, 0.19,  1.0,      0.0, 20.0, false },
    { 437.065, 3.22,  1.71740, 48.1, 20.0, false },
    { -39.73,  0.0,   1.0,      0.0, 20.0, false },
  };

  // Nakamura's f/2.8 retrofocus wide angle, scaled to 22 mm as in pbrt's wide.22mm.dat (from Smith, Modern Lens
  // Design, p. 360). The source gives indices only, so the Abbe numbers are those of catalogue glasses of matching index.
  inline constexpr LensSurface kWideAngle22mm[] = {
    {   35.98738, 1.21638, 1.54,  59.7, 23.716, false },
    {   11.69718, 9.9957,  1.0,    0.0, 17.996, false },
    {   13.08714, 5.12622, 1.772, 49.6, 12.364, false },
    {  -22.63294, 1.76924, 1.617, 44.0,  9.812, false },
    {   71.05802, 0.8184,  1.0,    0.0,  9.152, false },
    {    0.0,     2.27766, 1.0,    0.0,  8.756, true  },
    {   -9.58584, 2.43254, 1.617, 44.0,  8.184, false },
    {  -11.28864, 0.11506, 1.0,    0.0,  9.152, false },
    { -166.7765,  3.09606, 1.713, 53.8, 10.648, false },
    {   -7.5911,  1.32682, 1.805, 25.4, 11.44,  false },
    {  -16.7662,  3.98068, 1.0,    0.0, 12.276, false },
    {   -7.70286, 1.21638, 1.617, 44.0, 13.42,  false },
    {  -11.97328, 0.0,     1.0,    0.0, 17.996, false },
  };

  // Fujie's f/2.8 retrofocus ultra wide angle for Nippon Kogaku, US 4,690,517, second embodiment (2w = 96 degrees),
  // scaled from f = 100 to 18 mm so that a full frame sensor cropped to 16:9 sees a 90 degree horizontal view. The
  // patent places the stop between L6 and L7 without its spacing, so it sits halfway across that gap. Patents give no
  // clear diameters: these come from real rays at full aperture on axis and through 85% of the stop out to 48 degrees
  // off axis, which lets the corners vignette as such lenses do.
  inline constexpr LensSurface kUltraWide18mm[] = {
    {   29.1189, 3.7060, 1.59319, 67.9, 38.537, false },
    {   78.3536, 0.0882, 1.0,      0.0, 40.077, false },
    {   23.6481, 0.7942, 1.79504, 28.6, 28.576, false },
    {    9.2652, 3.0001, 1.0,      0.0, 18.520, false },
    {   17.4428, 0.7942, 1.71700, 48.1, 20.112, false },
    {    9.2259, 3.0885, 1.0,      0.0, 16.695, false },
    {   88.2392, 2.5589, 1.74077, 27.6, 16.752, false },
    {  -21.1774, 1.3235, 1.74810, 52.3, 16.698, false },
    {    9.5298, 4.5002, 1.61293, 37.0, 15.998, false },
    { -114.7110, 0.1764, 1.0,      0.0, 16.015, false },
    {   27.1776, 3.1767, 1.64831, 33.8, 16.257, false },
    { -150.0066, 1.1912, 1.0,      0.0, 15.978, false },
    { -101.2650, 3.1767, 1.62041, 60.3, 15.612, false },
    {  -14.4712, 1.3235, 1.0,      0.0, 15.422, false },
    {    0.0,    1.3235, 1.0,      0.0, 11.513, true  },
    {  -19.6773, 3.7060, 1.79668, 45.5, 11.851, false },
    {  -15.0007, 0.9706, 1.78472, 25.8, 12.481, false },
    {   41.3841, 0.8824, 1.0,      0.0, 14.501, false },
    {  -82.0625, 2.0295, 1.60311, 60.7, 14.412, false },
    {  -14.8645, 0.0882, 1.0,      0.0, 14.914, false },
    {  116.4757, 2.4707, 1.62041, 60.4, 17.861, false },
    {  -28.1708, 0.0,    1.0,      0.0, 18.283, false },
  };

  // Indexed by LensDesign.
  inline constexpr LensPrescription kLensPrescriptions[] = {
    { "Double Gauss 50mm f/2 (Tronnier)", kDoubleGauss50mm, static_cast<uint32_t>(std::size(kDoubleGauss50mm)) },
    { "Wide Angle 22mm f/2.8 (Nakamura)", kWideAngle22mm, static_cast<uint32_t>(std::size(kWideAngle22mm)) },
    { "Ultra Wide 18mm f/2.8 (Fujie)", kUltraWide18mm, static_cast<uint32_t>(std::size(kUltraWide18mm)) },
  };

}
