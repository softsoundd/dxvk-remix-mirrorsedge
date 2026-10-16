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
#include <cstring>
#include <iostream>
#include <string>

#include "d3d9_sent_state.h"

using namespace std;

namespace {
  void check(const bool condition, const char* what) {
    if (!condition) {
      throw string(what);
    }
  }

  void testScalarStates() {
    SentState s;
    check(s.renderState(D3DRS_ZENABLE, 1), "A first render state is sent");
    check(!s.renderState(D3DRS_ZENABLE, 1), "A repeated render state is dropped");
    check(s.renderState(D3DRS_ZENABLE, 0), "A changed render state is sent");
    check(s.renderState(D3DRS_POINTSIZE, 5) && s.renderState(D3DRS_POINTSIZE, 5), "Point size is always sent");
    check(s.renderState(D3DRS_ADAPTIVETESS_Y, 7) && s.renderState(D3DRS_ADAPTIVETESS_Y, 7),
          "Adaptive tessellation states are always sent");

    check(s.samplerState(D3DVERTEXTEXTURESAMPLER0, D3DSAMP_MINFILTER, 2),
          "A first vertex texture sampler state is sent");
    check(!s.samplerState(D3DVERTEXTEXTURESAMPLER0, D3DSAMP_MINFILTER, 2),
          "A repeated vertex texture sampler state is dropped");
    check(s.samplerState(D3DVERTEXTEXTURESAMPLER0 + 4, D3DSAMP_MINFILTER, 2) &&
          s.samplerState(D3DVERTEXTEXTURESAMPLER0 + 4, D3DSAMP_MINFILTER, 2),
          "A sampler outside the tracked ones is always sent");

    check(s.texture(3, 41), "A first texture is sent");
    check(!s.texture(3, 41), "A repeated texture is dropped");
    check(s.texture(3, 0), "Unbinding a texture is sent");

    D3DMATRIX matrix = {};
    matrix._11 = 1.0f;
    check(s.transform(D3DTS_WORLDMATRIX(255), matrix), "A first world matrix is sent");
    check(!s.transform(D3DTS_WORLDMATRIX(255), matrix), "A repeated world matrix is dropped");
    matrix._44 = 2.0f;
    check(s.transform(D3DTS_WORLDMATRIX(255), matrix), "A changed world matrix is sent");
  }

  void testInvalidation() {
    SentState s;
    const D3DVIEWPORT9 viewport = { 0, 0, 640, 480, 0.0f, 1.0f };
    const RECT scissor = { 0, 0, 640, 480 };
    s.viewport(viewport);
    s.scissor(scissor);
    s.renderState(D3DRS_CULLMODE, D3DCULL_CW);

    s.invalidateViewportAndScissor();
    check(s.viewport(viewport), "The viewport is sent again after SetRenderTarget(0)");
    check(s.scissor(scissor), "The scissor rect is sent again after SetRenderTarget(0)");
    check(!s.renderState(D3DRS_CULLMODE, D3DCULL_CW), "Other state is still known after SetRenderTarget(0)");

    s.invalidateAll();
    check(s.renderState(D3DRS_CULLMODE, D3DCULL_CW), "Everything is sent again after a reset");
    check(!s.renderState(D3DRS_CULLMODE, D3DCULL_CW), "State is tracked again after a reset");
  }

  void testFvfAndDeclaration() {
    SentState s;
    check(s.vertexDeclaration(5), "A first declaration is sent");
    check(!s.vertexDeclaration(5), "A repeated declaration is dropped");
    check(s.fvf(D3DFVF_XYZ), "An FVF is sent after a declaration");
    check(s.vertexDeclaration(5), "The declaration is sent again after an FVF replaced it");
    check(s.fvf(D3DFVF_XYZ), "The FVF is sent again after a declaration replaced it");
    check(!s.fvf(D3DFVF_XYZ), "A repeated FVF is dropped");
  }

  void testStreams() {
    SentState s;
    check(s.streamSource(0, 7, 0, 32), "A first stream source is sent");
    check(!s.streamSource(0, 7, 0, 32), "A repeated stream source is dropped");
    check(s.streamSource(0, 7, 16, 32), "A new stream offset is sent");
    check(s.streamSource(caps::MaxStreams, 7, 0, 32) && s.streamSource(caps::MaxStreams, 7, 0, 32),
          "A stream outside the tracked ones is always sent");
    check(s.streamSourceFreq(1, D3DSTREAMSOURCE_INSTANCEDATA | 1), "A first stream frequency is sent");
    check(!s.streamSourceFreq(1, D3DSTREAMSOURCE_INSTANCEDATA | 1), "A repeated stream frequency is dropped");

    s.streamSource(1, 8, 0, 16);
    s.indices(9);
    s.invalidateUserPointerDraw(false);
    check(s.streamSource(0, 7, 16, 32), "Stream 0 is sent again after a DrawPrimitiveUP");
    check(!s.streamSource(1, 8, 0, 16), "Other streams are still known after a DrawPrimitiveUP");
    check(!s.indices(9), "The indices are still known after a DrawPrimitiveUP");
    s.invalidateUserPointerDraw(true);
    check(s.streamSource(0, 7, 16, 32), "Stream 0 is sent again after a DrawIndexedPrimitiveUP");
    check(s.indices(9), "The indices are sent again after a DrawIndexedPrimitiveUP");
  }

  void testConstants() {
    SentState s;
    float a[4 * 4];
    for (uint32_t i = 0; i < 16; ++i) {
      a[i] = float(i);
    }

    uint32_t start = 10;
    uint32_t count = 4;
    const float* data = a;
    check(s.vsConstantsF(start, data, count) && start == 10 && count == 4 && data == a,
          "First constants are sent whole");

    start = 10;
    count = 4;
    data = a;
    check(!s.vsConstantsF(start, data, count), "Repeated constants are dropped");

    float b[4 * 4];
    std::memcpy(b, a, sizeof(a));
    b[4] += 1.0f;
    b[8] += 1.0f;
    start = 10;
    count = 4;
    data = b;
    check(s.vsConstantsF(start, data, count) && start == 11 && count == 2 && data == b + 4,
          "Changed constants are narrowed to the registers that differ");

    b[0] += 1.0f;
    b[12] += 1.0f;
    start = 10;
    count = 4;
    data = b;
    check(s.vsConstantsF(start, data, count) && start == 10 && count == 4 && data == b,
          "The narrowed range spans from the first to the last register that differs");

    start = 12;
    count = 2;
    data = b + 8;
    check(!s.vsConstantsF(start, data, count), "A sub-range of what was sent is dropped");

    const float c[4 * 2] = {};
    start = caps::MaxFloatConstantsPS - 1;
    count = 2;
    const float* outside = c;
    check(s.psConstantsF(start, outside, count) && start == caps::MaxFloatConstantsPS - 1 && count == 2,
          "A range past the end of the bank is sent whole");
    start = caps::MaxFloatConstantsPS - 1;
    count = 2;
    outside = c;
    check(s.psConstantsF(start, outside, count), "A range past the end of the bank is not recorded");

    const BOOL flags[3] = { TRUE, FALSE, TRUE };
    uint32_t boolStart = 0;
    uint32_t boolCount = 3;
    const BOOL* boolData = flags;
    check(s.psConstantsB(boolStart, boolData, boolCount), "First bool constants are sent");
    boolStart = 0;
    boolCount = 3;
    boolData = flags;
    check(!s.psConstantsB(boolStart, boolData, boolCount), "Repeated bool constants are dropped");

    s.invalidateAll();
    start = 10;
    count = 4;
    data = b;
    check(s.vsConstantsF(start, data, count) && start == 10 && count == 4, "Constants are sent whole after a reset");
  }
}

int main() {
  try {
    testScalarStates();
    testInvalidation();
    testFvfAndDeclaration();
    testStreams();
    testConstants();
  }
  catch (const string& errorMessage) {
    cerr << "SentState test failed: " << errorMessage << endl;
    return -1;
  }
  cout << "SentState tests passed" << endl;
  return 0;
}
