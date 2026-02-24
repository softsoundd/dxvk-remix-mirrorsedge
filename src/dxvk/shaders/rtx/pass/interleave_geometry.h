/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "../utility/packing_helpers.h"

// This function can be executed on the CPU or GPU!!
#ifdef __cplusplus
#define asfloat(x) *reinterpret_cast<const float*>(&x)
#define asuint(x) *reinterpret_cast<const uint32_t*>(&x)
#define WriteBuffer(T) T*
#define ReadBuffer(T) const T*
#else
#define WriteBuffer(T) RWStructuredBuffer<T>
#define ReadBuffer(T) StructuredBuffer<T>
#endif

namespace interleaver {

  enum SupportedVkFormats : uint32_t {
    VK_FORMAT_R8G8B8A8_UNORM = 37,
    VK_FORMAT_A2B10G10R10_SNORM_PACK32 = 65,

    // Passthrough format mapping
    VK_FORMAT_B8G8R8A8_UNORM = 44,
    VK_FORMAT_R16G16_SFLOAT = 83,
    VK_FORMAT_R32G32_SFLOAT = 103,
    VK_FORMAT_R32G32B32_SFLOAT = 106,
    VK_FORMAT_R32G32B32A32_SFLOAT = 109,
    VK_FORMAT_R16G16B16A16_SFLOAT = 97,
  };

#ifdef __cplusplus
  // HLSL provides a f16tof32 intrinsic,  provide a compatible implementation for CPU execution
  static inline float f16tof32(uint32_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = (h & 0x03FFu);

    uint32_t out;

    if (exp == 0) {
      // zero / subnormal
      if (mant == 0) {
        out = sign;
      } else {
        // normalise the subnormal number
        exp = 1;
        while ((mant & 0x0400u) == 0) {
          mant <<= 1;
          exp--;
        }
        mant &= 0x03FFu;

        const uint32_t exp32 = exp + (127u - 15u);
        out = sign | (exp32 << 23) | (mant << 13);
      }
    } else if (exp == 31) {
      // inf / NaN
      out = sign | 0x7F800000u | (mant << 13);
    } else {
      // normalised
      const uint32_t exp32 = exp + (127u - 15u);
      out = sign | (exp32 << 23) | (mant << 13);
    }

    return asfloat(out);
  }
#endif

  static inline int signExtend10(uint x) {
    // keep lower 10 bits and sign extend to 32-bit integer
    return (int(x << 22)) >> 22;
  }

  static inline float snorm10ToF32_packed(uint x) {
    const int v = signExtend10(x);
    float f = float(v) / 511.0f;
    // clamp to [-1, 1] (since -512 / 511 < -1)
    f = (f < -1.0f) ? -1.0f : ((f > 1.0f) ? 1.0f : f);
    return f;
  }

  bool formatConversionFloatSupported(uint32_t format) {
    switch (format) {
    case SupportedVkFormats::VK_FORMAT_R16G16_SFLOAT:
    case SupportedVkFormats::VK_FORMAT_R16G16B16A16_SFLOAT:
    case SupportedVkFormats::VK_FORMAT_R32G32_SFLOAT:
    case SupportedVkFormats::VK_FORMAT_R32G32B32_SFLOAT:
    case SupportedVkFormats::VK_FORMAT_R32G32B32A32_SFLOAT:
    case SupportedVkFormats::VK_FORMAT_R8G8B8A8_UNORM:
    case SupportedVkFormats::VK_FORMAT_A2B10G10R10_SNORM_PACK32:
      return true;
    default:
      return false;
    }
  }

  bool formatConversionUintSupported(uint32_t format) {
    switch (format) {
    case SupportedVkFormats::VK_FORMAT_B8G8R8A8_UNORM:
      return true;
    default:
      return false;
    }
  }

  float3 convert(uint32_t format, ReadBuffer(float) input, uint32_t index) {
    switch (format) {
    case SupportedVkFormats::VK_FORMAT_R16G16_SFLOAT:
    {
      uint data = asuint(input[index]);
      float x = f16tof32(data & 0xffff);
      float y = f16tof32(data >> 16);
      return float3(x, y, 0);
    }
    case SupportedVkFormats::VK_FORMAT_R16G16B16A16_SFLOAT:
    {
      uint data0 = asuint(input[index + 0]);
      uint data1 = asuint(input[index + 1]);
      float x = f16tof32(data0 & 0xffff);
      float y = f16tof32(data0 >> 16);
      float z = f16tof32(data1 & 0xffff);
      return float3(x, y, z);
    }
    case SupportedVkFormats::VK_FORMAT_R32G32_SFLOAT:
      return float3(input[index + 0], input[index + 1], 0);
    case SupportedVkFormats::VK_FORMAT_R32G32B32_SFLOAT:
    case SupportedVkFormats::VK_FORMAT_R32G32B32A32_SFLOAT:
      return float3(input[index + 0], input[index + 1], input[index + 2]);
    case SupportedVkFormats::VK_FORMAT_R8G8B8A8_UNORM:
    {
      uint data = asuint(input[index]);
      float b = unorm8ToF32(uint8_t((data >> 16) & 0xFF));
      float g = unorm8ToF32(uint8_t((data >> 8) & 0xFF));
      float r = unorm8ToF32(uint8_t((data >> 0) & 0xFF));
      return float3(r, g, b) * 2.f - 1.f;
    }
    case SupportedVkFormats::VK_FORMAT_A2B10G10R10_SNORM_PACK32:
    {
      uint data = asuint(input[index]);
      float b = snorm10ToF32_packed(data >> 20);
      float g = snorm10ToF32_packed(data >> 10);
      float r = snorm10ToF32_packed(data >> 0);
      return float3(r, g, b);
    }
    }
    return float3(1, 1, 1);
  }

  uint3 convert(uint32_t format, ReadBuffer(uint32_t) input, uint32_t index) {
    switch (format) {
    case SupportedVkFormats::VK_FORMAT_B8G8R8A8_UNORM:
      // Passthrough format we support in other places
      return uint3(input[index], 0, 0);
    }
    return uint3(1,1,1);
  }

  void interleave(const uint32_t idx, WriteBuffer(float) dst, ReadBuffer(float) srcPosition, ReadBuffer(float) srcNormal, ReadBuffer(float) srcTexcoord, ReadBuffer(uint32_t) srcColor0, const InterleaveGeometryArgs cb) {
    const uint32_t srcVertexIndex = idx + cb.minVertexIndex;

    uint32_t writeOffset = 0;

    float3 position = convert(cb.positionFormat, srcPosition, srcVertexIndex * cb.positionStride + cb.positionOffset);
    dst[idx * cb.outputStride + writeOffset++] = position.x;
    dst[idx * cb.outputStride + writeOffset++] = position.y;
    dst[idx * cb.outputStride + writeOffset++] = position.z;

    if (cb.hasNormals) {
      float3 normals = convert(cb.normalFormat, srcNormal, srcVertexIndex * cb.normalStride + cb.normalOffset);
      dst[idx * cb.outputStride + writeOffset++] = normals.x;
      dst[idx * cb.outputStride + writeOffset++] = normals.y;
      dst[idx * cb.outputStride + writeOffset++] = normals.z;
    } else if (cb.forceNormals) {
      // Reserve normal space with zeros; will be filled by smooth normals pass
      dst[idx * cb.outputStride + writeOffset++] = 0.0f;
      dst[idx * cb.outputStride + writeOffset++] = 0.0f;
      dst[idx * cb.outputStride + writeOffset++] = 0.0f;
    }

    if (cb.hasTexcoord) {
      float3 texcoords = convert(cb.texcoordFormat, srcTexcoord, srcVertexIndex * cb.texcoordStride + cb.texcoordOffset);
      dst[idx * cb.outputStride + writeOffset++] = texcoords.x;
      dst[idx * cb.outputStride + writeOffset++] = texcoords.y;
    }

    if (cb.hasColor0) {
      uint3 color0 = convert(cb.color0Format, srcColor0, srcVertexIndex * cb.color0Stride + cb.color0Offset);
      dst[idx * cb.outputStride + writeOffset++] = asfloat(color0.x);
    }
  }
}

#ifdef __cplusplus
#undef WriteBuffer
#undef ReadBuffer

#undef asfloat
#undef asuint
#endif