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

  XXH64_hash_t D3D9Rtx::computeLiveGeometryVertexShaderHashComponent() {
    XXH64_hash_t vertexShaderHash = kEmptyHash;

    if (m_parent->UseProgrammableVS() && m_frameOptions.useVertexCapture) {
      if (RtxOptions::geometryHashGenerationRule().test(HashComponents::GeometryDescriptor)) {
        vertexShaderHash = m_activeStableVsHash;

        if (m_activeStableVsHashUsedExclusions) {
          // refresh geometry as the camera travels by folding a coarse camera anchor into the hash
          // doing this to avoid the distortion that grows with distance from the location where RT was enabled
          // todo: revisit this, it still doesn't solve scene capture distortion
          Ue3CameraHashCell cameraCell;
          if (computeUe3CameraHashCell(cameraCell)) {
            logUe3CameraHashCellIfChanged(cameraCell, "geometry hash");
            vertexShaderHash = XXH3_64bits_withSeed(
              &cameraCell,
              sizeof(cameraCell),
              vertexShaderHash);
          }
          if (m_frameOptions.ue3LogCapturePrecision && Logger::logLevel() <= LogLevel::Debug) {
            ONCE(Logger::debug(str::format(
              "[RTX-Compatibility][UE3-Capture] VS camera constants excluded from geometry hash, cameraCellEnabled=",
              shouldUseUe3CameraHashCell())));
          }
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

    return vertexShaderHash;
  }

  Future<GeometryHashes> D3D9Rtx::computeHash(const RasterGeometry& geoData, const uint32_t maxIndexValue,
                                              const std::shared_ptr<Ue3GeometryMemoEntry>& publishTo) {
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

    // Assume the GPU changed the data via shaders, include the constant buffer data in hash.
    // The bytecode + constant hashing (with UE3 camera-register exclusions) is computed once
    // per draw in internalPrepareDraw (m_activeStableVsHash) and shared with the static
    // vertex-capture cache key; only the geometry-hash-specific folds happen here. Shared
    // with the geometry memo hit path so served hashes recombine to identical values.
    const XXH64_hash_t vertexShaderHash = computeLiveGeometryVertexShaderHashComponent();

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
                                 vertexLayoutHash, publishTo]() -> GeometryHashes {
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

      // Publish into the static-geometry memo entry so later frames can reuse the
      // result without recomputing (entry storage is heap-pinned via shared_ptr).
      // The VertexShader component is per-draw (stable VS-constant hash, camera cell)
      // and is recombined live by the memo consumer, so it is not stored.
      if (publishTo != nullptr) {
        for (uint32_t i = 0; i < uint32_t(HashComponents::Count); i++) {
          publishTo->componentHashes[i] = hashes[HashComponents(i)];
        }
        publishTo->componentHashes[uint32_t(HashComponents::VertexShader)] = kEmptyHash;
        publishTo->hashesReady.store(true, std::memory_order_release);
      }

      return hashes;
    });
  }

  Future<AxisAlignedBoundingBox> D3D9Rtx::computeAxisAlignedBoundingBox(const RasterGeometry& geoData,
                                                                        const std::shared_ptr<Ue3GeometryMemoEntry>& publishTo) {
    ScopedCpuProfileZone();

    if (!m_frameOptions.needsMeshBoundingBox) {
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

    return m_pGeometryWorkers->Schedule([pVertexData, vertexCount, vertexStride, vertexBuffer, publishTo]()->AxisAlignedBoundingBox {
      ScopedCpuProfileZone();

#if defined(_M_ARM64) || defined(_M_ARM64EC)
      float32x4_t minPos = vdupq_n_f32(FLT_MAX);
      float32x4_t maxPos = vdupq_n_f32(-FLT_MAX);

      const uint8_t* pVertex = static_cast<const uint8_t*>(pVertexData);
      for (uint32_t vertexIdx = 0; vertexIdx < vertexCount; ++vertexIdx) {
        const Vector3* const pVertexPos = reinterpret_cast<const Vector3* const>(pVertex);
        float32x4_t vertexPos;
        vertexPos.n128_f32[0] = pVertexPos->x;
        vertexPos.n128_f32[1] = pVertexPos->y;
        vertexPos.n128_f32[2] = pVertexPos->z;

        minPos = vminq_f32(minPos, vertexPos);
        maxPos = vmaxq_f32(maxPos, vertexPos);

        pVertex += vertexStride;
      }

      AxisAlignedBoundingBox boundingBox {
        Vector3{ vgetq_lane_f32(minPos, 0), vgetq_lane_f32(minPos, 1), vgetq_lane_f32(minPos, 2) },
        Vector3{ vgetq_lane_f32(maxPos, 0), vgetq_lane_f32(maxPos, 1), vgetq_lane_f32(maxPos, 2) }
      };
#else
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
#endif

      vertexBuffer->decRef();

      // Publish into the static-geometry memo entry for cross-frame reuse
      if (publishTo != nullptr) {
        publishTo->boundingBox = boundingBox;
        publishTo->aabbReady.store(true, std::memory_order_release);
      }

      return boundingBox;
    });
  }
}
