#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include "d3d9_device.h"
#include "d3d9_rtx.h"
#include "d3d9_rtx_utils.h"
#include "d3d9_state.h"
#include "../dxvk/dxvk_buffer.h"
#include "../dxvk/rtx_render/rtx_hashing.h"
#include "../util/util_fastops.h"

namespace dxvk {
  // Geometry indices should never be signed.  Using this to handle the non-indexed case for templates.
  typedef int NoIndices;

  namespace VertexRegions {
    enum Type : uint32_t {
      Position = 0,
      Texcoord,
      Count
    };
  }

  // NOTE: Intentionally leaving the legacy hashes out of here, because they are special (REMIX-656)
  const std::map<HashComponents, VertexRegions::Type> componentToRegionMap = {
    { HashComponents::VertexPosition,   VertexRegions::Position },
    { HashComponents::VertexTexcoord,   VertexRegions::Texcoord },
  };

  bool getVertexRegion(const RasterBuffer& buffer, const size_t vertexCount, HashQuery& outResult) {
    ScopedCpuProfileZone();

    if (!buffer.defined())
      return false;

    outResult.pBase = (uint8_t*) buffer.mapPtr(buffer.offsetFromSlice());
    outResult.elementSize = imageFormatInfo(buffer.vertexFormat())->elementSize;
    outResult.stride = buffer.stride();
    outResult.size = outResult.stride * vertexCount;
    // Make sure we hold on to this reference while the hashing is in flight
    outResult.ref = buffer.buffer().ptr();
    assert(outResult.ref);
    return true;
  }

  // Sorts and deduplicates a set of integers, storing the result in a vector
  template<typename T>
  void deduplicateSortIndices(const void* pIndexData, const size_t indexCount, const uint32_t maxIndexValue, std::vector<T>& uniqueIndicesOut) {
    // TODO (REMIX-657): Implement optimized variant of this function
    // We know there will be at most, this many unique indices
    const uint32_t indexRange = maxIndexValue + 1;

    // Initialize all to 0
    uniqueIndicesOut.resize(indexRange, (T)0);

    // Use memory as a bin table for index data
    for (uint32_t i = 0; i < indexCount; i++) {
      const T& index = ((T*) pIndexData)[i];
      assert(index <= maxIndexValue);
      uniqueIndicesOut[index] = 1;
    }

    // Repopulate the bins with contiguous index values
    uint32_t uniqueIndexCount = 0;
    for (uint32_t i = 0; i < indexRange; i++) {
      if (uniqueIndicesOut[i])
        uniqueIndicesOut[uniqueIndexCount++] = i;
    }

    // Remove any unused entries
    uniqueIndicesOut.resize(uniqueIndexCount);
  }

  template<typename T>
  void hashGeometryData(const size_t indexCount, const uint32_t maxIndexValue, const void* pIndexData,
                        DxvkBuffer* indexBufferRef, const HashQuery vertexRegions[VertexRegions::Count], GeometryHashes& hashesOut) {
    ScopedCpuProfileZone();

    const HashRule& globalHashRule = RtxOptions::geometryHashGenerationRule();

    // TODO (REMIX-658): Improve this by reducing allocation overhead of vector
    std::vector<T> uniqueIndices(0);
    if constexpr (!std::is_same<T, NoIndices>::value) {
      assert((indexCount > 0 && indexBufferRef));
      deduplicateSortIndices(pIndexData, indexCount, maxIndexValue, uniqueIndices);

      if (globalHashRule.test(HashComponents::Indices)) {
        hashesOut[HashComponents::Indices] = hashContiguousMemory(pIndexData, indexCount * sizeof(T));
      }

      // TODO (REMIX-656): Remove this once we can transition content to new hash
      if (globalHashRule.test(HashComponents::LegacyIndices)) {
        hashesOut[HashComponents::LegacyIndices] = hashIndicesLegacy<T>(pIndexData, indexCount);
      }

      // Release this memory back to the staging allocator
      indexBufferRef->release(DxvkAccess::Read);
      indexBufferRef->decRef();
    }

    // Do vertex based rules
    for (uint32_t i = 0; i < (uint32_t) HashComponents::Count; i++) {
      const HashComponents& component = (HashComponents) i;

      if (globalHashRule.test(component) && componentToRegionMap.count(component) > 0) {
        const VertexRegions::Type region = componentToRegionMap.at(component);
        hashesOut[component] = hashVertexRegionIndexed(vertexRegions[(uint32_t)region], uniqueIndices);
      }
    }

    // TODO (REMIX-656): Remove this once we can transition content to new hash
    if (globalHashRule.test(HashComponents::LegacyPositions0) || globalHashRule.test(HashComponents::LegacyPositions1)) {
      hashRegionLegacy(vertexRegions[VertexRegions::Position], hashesOut[HashComponents::LegacyPositions0], hashesOut[HashComponents::LegacyPositions1]);
    }

    // Release this memory back to the staging allocator
    for (uint32_t i = 0; i < VertexRegions::Count; i++) {
      const HashQuery& region = vertexRegions[i];
      if (region.size == 0)
        continue;

      if (region.ref) {
        region.ref->release(DxvkAccess::Read);
        region.ref->decRef();
      }
    }
  }

  Future<GeometryHashes> D3D9Rtx::computeHash(const RasterGeometry& geoData, const uint32_t maxIndexValue) {
    ScopedCpuProfileZone();

    const uint32_t indexCount = geoData.indexCount;
    const uint32_t vertexCount = geoData.vertexCount;

    HashQuery vertexRegions[VertexRegions::Count];
    memset(&vertexRegions[0], 0, sizeof(vertexRegions));

    if (!getVertexRegion(geoData.positionBuffer, vertexCount, vertexRegions[VertexRegions::Position]))
      return Future<GeometryHashes>(); //invalid

    // Acquire prevents the staging allocator from re-using this memory
    vertexRegions[VertexRegions::Position].ref->acquire(DxvkAccess::Read);
    vertexRegions[VertexRegions::Position].ref->incRef();

    if (getVertexRegion(geoData.texcoordBuffer, vertexCount, vertexRegions[VertexRegions::Texcoord])) {
      vertexRegions[VertexRegions::Texcoord].ref->acquire(DxvkAccess::Read);
      vertexRegions[VertexRegions::Texcoord].ref->incRef();
    }

    // Make sure we hold a ref to the index buffer while hashing.
    const Rc<DxvkBuffer> indexBufferRef = geoData.indexBuffer.buffer();
    if (indexBufferRef.ptr()) {
      indexBufferRef->acquire(DxvkAccess::Read);
      indexBufferRef->incRef();
    }
    const void* pIndexData = geoData.indexBuffer.defined() ? geoData.indexBuffer.mapPtr(0) : nullptr;
    const size_t indexStride = geoData.indexBuffer.stride();
    const size_t indexDataSize = indexCount * indexStride;

    // Assume the GPU changed the data via shaders, include the constant buffer data in hash
    XXH64_hash_t vertexShaderHash = kEmptyHash;
    if (m_parent->UseProgrammableVS() && useVertexCapture()) {
      if (RtxOptions::geometryHashGenerationRule().test(HashComponents::GeometryDescriptor)) {
        const D3D9ConstantSets& cb = m_parent->m_consts[DxsoProgramTypes::VertexShader];
        auto& shaderByteCode = d3d9State().vertexShader->GetCommonShader()->GetBytecode();
        vertexShaderHash = XXH3_64bits(shaderByteCode.data(), shaderByteCode.size());

        const uint32_t floatConstRegCount = cb.meta.maxConstIndexF;
        const uint8_t* const floatConstBase = reinterpret_cast<const uint8_t*>(&d3d9State().vsConsts.fConsts[0]);

        auto hashFloatConstRange = [&](uint32_t beginReg, uint32_t endReg) {
          beginReg = std::min(beginReg, floatConstRegCount);
          endReg = std::min(endReg, floatConstRegCount);
          if (beginReg >= endReg)
            return;

          const size_t offsetBytes = size_t(beginReg) * sizeof(Vector4);
          const size_t sizeBytes = size_t(endReg - beginReg) * sizeof(Vector4);
          vertexShaderHash = XXH3_64bits_withSeed(floatConstBase + offsetBytes, sizeBytes, vertexShaderHash);
        };

        bool hashedFloatConstsWithExclusions = false;
        if (ue3CameraFromShaderConstants() && floatConstRegCount > 0) {
          // UE3 compat/perf to avoid camera-motion-only hash churn by excluding known
          // camera constants (ViewProjection + CameraPosition) from the VS constant hash
          constexpr uint32_t kFallbackViewProjReg = 0;
          constexpr uint32_t kFallbackViewProjRegCount = 4;
          constexpr uint32_t kFallbackViewOriginReg = 4;
          constexpr uint32_t kFallbackViewOriginRegCount = 1;

          uint32_t viewProjReg = kFallbackViewProjReg;
          uint32_t viewProjRegCount = kFallbackViewProjRegCount;
          uint32_t viewOriginReg = kFallbackViewOriginReg;
          uint32_t viewOriginRegCount = kFallbackViewOriginRegCount;

          if (m_currentUe3CtabInfo.has_value()) {
            const Ue3VsShaderCtabInfo& ctabInfo = *m_currentUe3CtabInfo;
            if (ctabInfo.hasViewProjectionMatrix && ctabInfo.viewProjectionMatrixRegisterCount > 0) {
              viewProjReg = ctabInfo.viewProjectionMatrixRegisterIndex;
              viewProjRegCount = ctabInfo.viewProjectionMatrixRegisterCount;
            }
            if (ctabInfo.hasCameraPosition && ctabInfo.cameraPositionRegisterCount > 0) {
              viewOriginReg = ctabInfo.cameraPositionRegisterIndex;
              viewOriginRegCount = ctabInfo.cameraPositionRegisterCount;
            }
          }

          struct RegRange {
            uint32_t begin = 0;
            uint32_t end = 0;
          };

          std::array<RegRange, 2> ranges = {{
            { viewProjReg, viewProjReg + viewProjRegCount },
            { viewOriginReg, viewOriginReg + viewOriginRegCount },
          }};

          std::array<RegRange, 2> validRanges = {};
          uint32_t validRangeCount = 0;
          for (const RegRange& r : ranges) {
            RegRange clamped;
            clamped.begin = std::min(r.begin, floatConstRegCount);
            clamped.end = std::min(r.end, floatConstRegCount);
            if (clamped.begin < clamped.end)
              validRanges[validRangeCount++] = clamped;
          }

          if (validRangeCount > 0) {
            if (validRangeCount == 2 && validRanges[1].begin < validRanges[0].begin)
              std::swap(validRanges[0], validRanges[1]);

            if (validRangeCount == 2 && validRanges[1].begin <= validRanges[0].end) {
              validRanges[0].end = std::max(validRanges[0].end, validRanges[1].end);
              validRangeCount = 1;
            }

            uint32_t cursor = 0;
            for (uint32_t i = 0; i < validRangeCount; i++) {
              hashFloatConstRange(cursor, validRanges[i].begin);
              cursor = std::max(cursor, validRanges[i].end);
            }
            hashFloatConstRange(cursor, floatConstRegCount);
            hashedFloatConstsWithExclusions = true;
          }
        }

        if (!hashedFloatConstsWithExclusions && floatConstRegCount > 0) {
          vertexShaderHash = XXH3_64bits_withSeed(
            &d3d9State().vsConsts.fConsts[0],
            size_t(floatConstRegCount) * sizeof(Vector4),
            vertexShaderHash);
        }

        if (cb.meta.maxConstIndexI > 0) {
          vertexShaderHash = XXH3_64bits_withSeed(
            &d3d9State().vsConsts.iConsts[0],
            size_t(cb.meta.maxConstIndexI) * sizeof(int) * 4,
            vertexShaderHash);
        }
        if (cb.meta.maxConstIndexB > 0) {
          vertexShaderHash = XXH3_64bits_withSeed(
            &d3d9State().vsConsts.bConsts[0],
            size_t(cb.meta.maxConstIndexB) * sizeof(uint32_t) / 32,
            vertexShaderHash);
        }

        if (hashedFloatConstsWithExclusions) {
          // refresh geometry as the camera travels by folding a coarse camera anchor into the hash
          // doing this to avoid the distortion that grows with distance from the location where RT was enabled
          // todo: revisit this, it still doesn't solve scene capture distortion
          constexpr float kCameraHashCellSize = 2000.0f;
          const Matrix4 cameraViewToWorld = inverseAffine(m_activeDrawCallState.transformData.worldToView);
          const Vector3 cameraPos = cameraViewToWorld[3].xyz();
          struct CameraHashCell {
            int32_t x;
            int32_t y;
            int32_t z;
          } cameraCell = {
            int32_t(std::floor(cameraPos.x / kCameraHashCellSize)),
            int32_t(std::floor(cameraPos.y / kCameraHashCellSize)),
            int32_t(std::floor(cameraPos.z / kCameraHashCellSize)),
          };
          vertexShaderHash = XXH3_64bits_withSeed(
            &cameraCell,
            sizeof(cameraCell),
            vertexShaderHash);
        }

        if (m_forceIaTexcoordForOutlier) {
          // compat cache key - outlier draws force IA texcoords in vertex capture
          // include this mode bit in the VS hash so cache entries built with VS TEXCOORD output
          // are not reused when outlier fallback wants IA TEXCOORDs and vice versa
          constexpr uint64_t kOutlierIaTexcoordHashMode = 0x4A5D9C5E6F10B2D3ull;
          vertexShaderHash = XXH3_64bits_withSeed(
            &kOutlierIaTexcoordHashMode,
            sizeof(kOutlierIaTexcoordHashMode),
            vertexShaderHash);
        }
      }
    }

    // Calculate this based on the RasterGeometry input data
    XXH64_hash_t geometryDescriptorHash = kEmptyHash;
    if (RtxOptions::geometryHashGenerationRule().test(HashComponents::GeometryDescriptor)) {
      geometryDescriptorHash = hashGeometryDescriptor(geoData.indexCount, 
                                                      geoData.vertexCount, 
                                                      geoData.indexBuffer.indexType(), 
                                                      geoData.topology);
    }

    // Calculate this based on the RasterGeometry input data
    XXH64_hash_t vertexLayoutHash = kEmptyHash;
    if (RtxOptions::geometryHashGenerationRule().test(HashComponents::VertexLayout)) {
      vertexLayoutHash = hashVertexLayout(geoData);
    }

    return m_pGeometryWorkers->Schedule([vertexRegions, indexBufferRef = indexBufferRef.ptr(),
                                 pIndexData, indexStride, indexDataSize, indexCount,
                                 maxIndexValue, vertexShaderHash, geometryDescriptorHash,
                                 vertexLayoutHash]() -> GeometryHashes {
      ScopedCpuProfileZone();

      GeometryHashes hashes;

      // Finalize the descriptor hash
      hashes[HashComponents::GeometryDescriptor] = geometryDescriptorHash;
      hashes[HashComponents::VertexLayout] = vertexLayoutHash;
      hashes[HashComponents::VertexShader] = vertexShaderHash;

      // Index hash
      switch (indexStride) {
      case 2:
        hashGeometryData<uint16_t>(indexCount, maxIndexValue, pIndexData, indexBufferRef, vertexRegions, hashes);
        break;
      case 4:
        hashGeometryData<uint32_t>(indexCount, maxIndexValue, pIndexData, indexBufferRef, vertexRegions, hashes);
        break;
      default:
        hashGeometryData<NoIndices>(indexCount, maxIndexValue, pIndexData, indexBufferRef, vertexRegions, hashes);
        break;
      }

      assert(hashes[HashComponents::VertexPosition] != kEmptyHash);

      hashes.precombine();

      return hashes;
    });
  }

  Future<AxisAlignedBoundingBox> D3D9Rtx::computeAxisAlignedBoundingBox(const RasterGeometry& geoData) {
    ScopedCpuProfileZone();

    if (!RtxOptions::needsMeshBoundingBox()) {
      return Future<AxisAlignedBoundingBox>();
    }

    const void* pVertexData = geoData.positionBuffer.mapPtr((size_t)geoData.positionBuffer.offsetFromSlice());
    const uint32_t vertexCount = geoData.vertexCount;
    const size_t vertexStride = geoData.positionBuffer.stride();

    if (pVertexData == nullptr) {
      return Future<AxisAlignedBoundingBox>();
    }

    auto vertexBuffer = geoData.positionBuffer.buffer().ptr();
    vertexBuffer->incRef();

    return m_pGeometryWorkers->Schedule([pVertexData, vertexCount, vertexStride, vertexBuffer]()->AxisAlignedBoundingBox {
      ScopedCpuProfileZone();

      __m128 minPos = _mm_set_ps1(FLT_MAX);
      __m128 maxPos = _mm_set_ps1(-FLT_MAX);

      const uint8_t* pVertex = static_cast<const uint8_t*>(pVertexData);
      for (uint32_t vertexIdx = 0; vertexIdx < vertexCount; ++vertexIdx) {
        const Vector3* const pVertexPos = reinterpret_cast<const Vector3* const>(pVertex);
        __m128 vertexPos = _mm_set_ps(0.0f, pVertexPos->z, pVertexPos->y, pVertexPos->x);
        minPos = _mm_min_ps(minPos, vertexPos);
        maxPos = _mm_max_ps(maxPos, vertexPos);

        pVertex += vertexStride;
      }

      AxisAlignedBoundingBox boundingBox{
        Vector3{ minPos.m128_f32[0], minPos.m128_f32[1], minPos.m128_f32[2] },
        Vector3{ maxPos.m128_f32[0], maxPos.m128_f32[1], maxPos.m128_f32[2] }
      };

      vertexBuffer->decRef();

      return boundingBox;
    });
  }
}
