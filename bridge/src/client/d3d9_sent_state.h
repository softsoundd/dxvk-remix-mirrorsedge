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

#include <array>
#include <cstdint>
#include <cstring>

#include <d3d9.h>

#include "util_common.h"

// Records what the server's device holds, so that client.dropRedundantState can drop a setter writing a value
// the server already has.
//
// This is not the game-visible state (BaseDirect3DDevice9Ex_LSS::m_state). It holds only the values this client sent
// outside state recording, and forgets them whenever the server's state can change without a setter from here:
// - StateBlock Apply, Reset and ResetEx forget everything.
// - SetRenderTarget(0) forgets the viewport and scissor rect.
// - SetFVF and SetVertexDeclaration forget each other, because one replaces the other on the server.
// - DrawPrimitiveUP forgets stream 0, and DrawIndexedPrimitiveUP the indices too.
//   The runtime unbinds them unless it skips the draw.
//
// Objects are identified by the client's object ids, which are never reused, so an equal id is the same object.
// Every method returns true when the call must be sent, and then records the value as sent.
class SentState {
  template<typename T>
  struct Entry {
    T value;
    uint32_t gen = 0;
  };

  template<typename T>
  bool update(Entry<T>& entry, const T& value) {
    if (entry.gen == m_gen && std::memcmp(&entry.value, &value, sizeof(T)) == 0) {
      return false;
    }
    entry.value = value;
    entry.gen = m_gen;
    return true;
  }

  template<typename T, uint32_t Elements, uint32_t Registers>
  struct ConstantBank {
    struct Reg {
      T v[Elements];
      uint32_t gen = 0;
    };
    std::array<Reg, Registers> regs;
  };

  // Records which registers of the range differ from what was sent, and narrows the range to span them.
  // Returns false when none differs.
  // Returns true without recording for a range outside the bank.
  template<typename T, uint32_t Elements, uint32_t Registers>
  bool updateConstants(ConstantBank<T, Elements, Registers>& bank, uint32_t& start, const T*& data, uint32_t& count) {
    if (count == 0 || start >= Registers || count > Registers - start) {
      return true;
    }
    uint32_t first = count;
    uint32_t last = 0;
    for (uint32_t i = 0; i < count; ++i) {
      auto& reg = bank.regs[start + i];
      const T* src = data + size_t(i) * Elements;
      if (reg.gen != m_gen || std::memcmp(reg.v, src, sizeof(T) * Elements) != 0) {
        std::memcpy(reg.v, src, sizeof(T) * Elements);
        reg.gen = m_gen;
        if (first == count) {
          first = i;
        }
        last = i;
      }
    }
    if (first == count) {
      return false;
    }
    data += size_t(first) * Elements;
    start += first;
    count = last - first + 1;
    return true;
  }

public:
  static constexpr uint32_t kRenderStates = 256;
  static constexpr uint32_t kTextureStages = 8;
  // Texture stage state types run up to D3DTSS_CONSTANT (32).
  static constexpr uint32_t kTextureStageStateTypes = 33;
  // There are 16 pixel samplers, D3DDMAPSAMPLER and 4 vertex texture samplers.
  static constexpr uint32_t kSamplers = 21;
  // Sampler state types run up to D3DSAMP_DMAPOFFSET (13).
  static constexpr uint32_t kSamplerStateTypes = 14;
  // Transform state types run up to D3DTS_WORLDMATRIX(255).
  static constexpr uint32_t kTransforms = 512;
  static constexpr uint32_t kStreams = caps::MaxStreams;

  // Forgets everything. An old entry could only look current again after 2^32 calls.
  void invalidateAll() {
    if (++m_gen == 0) {
      m_gen = 1;
    }
  }

  void invalidateViewportAndScissor() {
    m_viewport.gen = 0;
    m_scissor.gen = 0;
  }

  void invalidateUserPointerDraw(const bool indexed) {
    m_streams[0].gen = 0;
    if (indexed) {
      m_indices.gen = 0;
    }
  }

  bool renderState(uint32_t state, DWORD value) {
    // Driver hack channels such as RESZ, ATOC and NVDB ride on these states, so they are always sent.
    if (state >= kRenderStates || state == D3DRS_POINTSIZE ||
        (state >= D3DRS_ADAPTIVETESS_X && state <= D3DRS_ENABLEADAPTIVETESSELLATION)) {
      return true;
    }
    return update(m_renderStates[state], value);
  }

  bool textureStageState(uint32_t stage, uint32_t type, DWORD value) {
    if (stage >= kTextureStages || type >= kTextureStageStateTypes) {
      return true;
    }
    return update(m_textureStageStates[stage][type], value);
  }

  bool samplerState(uint32_t sampler, uint32_t type, DWORD value) {
    const uint32_t idx = samplerIndex(sampler);
    if (idx >= kSamplers || type >= kSamplerStateTypes) {
      return true;
    }
    return update(m_samplerStates[idx][type], value);
  }

  bool texture(uint32_t sampler, uint32_t objectId) {
    const uint32_t idx = samplerIndex(sampler);
    if (idx >= kSamplers) {
      return true;
    }
    return update(m_textures[idx], objectId);
  }

  bool transform(uint32_t state, const D3DMATRIX& matrix) {
    if (state >= kTransforms) {
      return true;
    }
    return update(m_transforms[state], matrix);
  }

  bool viewport(const D3DVIEWPORT9& value) {
    return update(m_viewport, value);
  }

  bool scissor(const RECT& value) {
    return update(m_scissor, value);
  }

  bool material(const D3DMATERIAL9& value) {
    return update(m_material, value);
  }

  bool streamSource(uint32_t stream, uint32_t objectId, uint32_t offset, uint32_t stride) {
    if (stream >= kStreams) {
      return true;
    }
    const StreamBinding binding { objectId, offset, stride };
    return update(m_streams[stream], binding);
  }

  bool streamSourceFreq(uint32_t stream, uint32_t divider) {
    if (stream >= kStreams) {
      return true;
    }
    return update(m_streamFreqs[stream], divider);
  }

  bool indices(uint32_t objectId) {
    return update(m_indices, objectId);
  }

  bool vertexDeclaration(uint32_t objectId) {
    m_fvf.gen = 0;
    return update(m_vertexDecl, objectId);
  }

  bool fvf(DWORD value) {
    m_vertexDecl.gen = 0;
    return update(m_fvf, value);
  }

  bool vertexShader(uint32_t objectId) {
    return update(m_vertexShader, objectId);
  }

  bool pixelShader(uint32_t objectId) {
    return update(m_pixelShader, objectId);
  }

  bool vsConstantsF(uint32_t& start, const float*& data, uint32_t& count) {
    return updateConstants(m_vsF, start, data, count);
  }

  bool vsConstantsI(uint32_t& start, const int*& data, uint32_t& count) {
    return updateConstants(m_vsI, start, data, count);
  }

  bool vsConstantsB(uint32_t& start, const BOOL*& data, uint32_t& count) {
    return updateConstants(m_vsB, start, data, count);
  }

  bool psConstantsF(uint32_t& start, const float*& data, uint32_t& count) {
    return updateConstants(m_psF, start, data, count);
  }

  bool psConstantsI(uint32_t& start, const int*& data, uint32_t& count) {
    return updateConstants(m_psI, start, data, count);
  }

  bool psConstantsB(uint32_t& start, const BOOL*& data, uint32_t& count) {
    return updateConstants(m_psB, start, data, count);
  }

private:
  struct StreamBinding {
    uint32_t objectId;
    uint32_t offset;
    uint32_t stride;
  };

  // D3D9 numbers the pixel samplers 0 to 15, D3DDMAPSAMPLER 256, and the vertex texture samplers 257 to 260.
  static uint32_t samplerIndex(uint32_t sampler) {
    if (sampler < 16) {
      return sampler;
    }
    if (sampler >= D3DDMAPSAMPLER && sampler <= D3DVERTEXTEXTURESAMPLER3) {
      return 16 + (sampler - D3DDMAPSAMPLER);
    }
    return kSamplers;
  }

  // Entries start at generation 0, so nothing counts as sent until it has been.
  uint32_t m_gen = 1;

  std::array<Entry<DWORD>, kRenderStates> m_renderStates;
  std::array<std::array<Entry<DWORD>, kTextureStageStateTypes>, kTextureStages> m_textureStageStates;
  std::array<std::array<Entry<DWORD>, kSamplerStateTypes>, kSamplers> m_samplerStates;
  std::array<Entry<uint32_t>, kSamplers> m_textures;
  std::array<Entry<D3DMATRIX>, kTransforms> m_transforms;
  Entry<D3DVIEWPORT9> m_viewport;
  Entry<RECT> m_scissor;
  Entry<D3DMATERIAL9> m_material;
  std::array<Entry<StreamBinding>, kStreams> m_streams;
  std::array<Entry<uint32_t>, kStreams> m_streamFreqs;
  Entry<uint32_t> m_indices;
  Entry<uint32_t> m_vertexDecl;
  Entry<DWORD> m_fvf;
  Entry<uint32_t> m_vertexShader;
  Entry<uint32_t> m_pixelShader;
  ConstantBank<float, 4, caps::MaxFloatConstantsSoftware> m_vsF;
  ConstantBank<int, 4, caps::MaxOtherConstantsSoftware> m_vsI;
  ConstantBank<BOOL, 1, caps::MaxOtherConstantsSoftware> m_vsB;
  ConstantBank<float, 4, caps::MaxFloatConstantsPS> m_psF;
  ConstantBank<int, 4, caps::MaxOtherConstants> m_psI;
  ConstantBank<BOOL, 1, caps::MaxOtherConstants> m_psB;
};
