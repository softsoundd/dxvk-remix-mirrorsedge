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
#include "d3d9_rtx.h"
#include "d3d9_rtx_ue3_helpers.h"

#include "d3d9_include.h"
#include "d3d9_state.h"
#include "d3d9_util.h"
#include "d3d9_buffer.h"
#include "d3d9_device.h"
#include "d3d9_initializer.h"
#include "../util/util_fastops.h"
#include "../util/util_game_patches.h"
#include "../util/util_math.h"
#include "d3d9_rtx_utils.h"
#include "d3d9_texture.h"
#include "../dxso/dxso_color_terms.h"
#include "../dxso/dxso_highlight_tints.h"
#include "../dxso/dxso_material_fades.h"
#include "../dxso/dxso_sampler_inference.h"
#include "../dxso/dxso_ue3_material_identity.h"
#include "../dxso/dxso_uv_dataflow.h"
#include "../dxso/dxso_tables.h"
#include "../dxvk/rtx_render/rtx_bridge_message_channel.h"
#include "../dxvk/rtx_render/rtx_terrain_baker.h"
#include "../dxvk/rtx_render/rtx_ue3_tone_mapping.h"
#include "../dxvk/rtx_render/rtx_gpu_pass_timer.h"
#include "../dxvk/imgui/dxvk_imgui.h"
#include <algorithm>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <shared_mutex>
#include <sstream>
#include <system_error>

namespace dxvk {

  // Unlike the capture cache this needs only immutable IA content, so any vertex factory qualifies: bone
  // constants live in the VertexShader component, which is recombined live.
  bool D3D9Rtx::canMemoizeUe3IaGeometryHashes(const IndexContext& indexContext,
                                              const VertexContext vertexContext[caps::MaxStreams],
                                              const RasterGeometry& geoData) const {
    if (!m_frameOptions.ue3StaticGeometryHashMemoization) {
      return false;
    }
    if (!m_frameOptions.ue3EngineMode) {
      return false;
    }
    // staging copies get a fresh physical slice every draw, so their keys never repeat and
    // memo entries would be dead weight
    if (m_forceGeometryCopy) {
      return false;
    }
    if (d3d9State().vertexDecl == nullptr) {
      return false;
    }
    if (!geoData.positionBuffer.defined()) {
      return false;
    }

    if (indexContext.indexType != VK_INDEX_TYPE_NONE_KHR && !isStaticD3D9Buffer(indexContext.ibo)) {
      return false;
    }

    for (const auto& element : d3d9State().vertexDecl->GetElements()) {
      if (element.Stream >= caps::MaxStreams) {
        return false;
      }

      // Instance-data streams are dynamic by nature and contribute nothing to the geometry the
      // memo describes; computeUe3IaGeometryMemoKey leaves them out of the key for the same reason.
      if ((m_currentUe3Instancing.instanceDataStreamMask & (1u << element.Stream)) != 0) {
        continue;
      }

      const VertexContext& ctx = vertexContext[element.Stream];
      if (ctx.mappedSlice.handle == VK_NULL_HANDLE || !isStaticD3D9Buffer(ctx.pVBO)) {
        return false;
      }
    }

    return true;
  }

  // Leaves out the stable VS hash and the object transform, so every placement and pose of a mesh shares
  // one entry.
  XXH64_hash_t D3D9Rtx::computeUe3IaGeometryMemoKey(const IndexContext& indexContext,
                                                    const VertexContext vertexContext[caps::MaxStreams],
                                                    const DrawContext& drawContext,
                                                    const RasterGeometry& geoData) const {
    constexpr uint64_t kSeed = 0x7BD5C66E91A3D4F1ull;

    const Ue3IaMemoKeyHeader header = makeUe3IaMemoKeyHeader(
        indexContext, drawContext, geoData,
        uint32_t(m_texcoordIndex), uint32_t(m_iaTexcoordIndex),
        uint32_t(m_texcoordCompU), uint32_t(m_texcoordCompV),
        uint32_t(m_uvResolutionMode), m_forceIaTexcoordForOutlier);

    const XXH64_hash_t headerHash = XXH3_64bits_withSeed(&header, sizeof(header), kSeed);
    return hashUe3KeyStreamRecords(d3d9State().vertexDecl->GetElements(), vertexContext, headerHash,
                                   m_currentUe3Instancing.instanceDataStreamMask);
  }

  // Resolved once per vertex shader. Camera registers are always excluded; the second variant also drops
  // the object transform and the shading-only constants.
  Ue3VsHashExclusions D3D9Rtx::buildUe3VsHashExclusions(
      const Ue3VsShaderCtabInfo& ctabInfo,
      const std::vector<Ue3VsConstantSymbol>* symbols) {
    using Range = Ue3VsHashExclusions::Range;
    std::array<Range, Ue3VsHashExclusions::kMaxRanges> raw = {};
    uint32_t rawCount = 0;
    auto add = [&](const uint32_t begin, const uint32_t count) {
      if (count == 0 || rawCount >= Ue3VsHashExclusions::kMaxRanges) {
        return;
      }
      raw[rawCount++] = Range { begin, begin + count };
    };

    // Sorts, merges overlaps, and writes the result out. Merging means the per-draw path can walk
    // the ranges once in order and hash the gaps between them.
    auto finalize = [](std::array<Range, Ue3VsHashExclusions::kMaxRanges>& ranges,
                       const uint32_t count,
                       std::array<Range, Ue3VsHashExclusions::kMaxRanges>& out,
                       uint32_t& outCount) {
      outCount = 0;
      if (count == 0) {
        return;
      }
      std::sort(ranges.begin(), ranges.begin() + count,
                [](const Range& a, const Range& b) { return a.begin < b.begin; });
      out[0] = ranges[0];
      outCount = 1;
      for (uint32_t i = 1; i < count; i++) {
        if (ranges[i].begin <= out[outCount - 1].end) {
          out[outCount - 1].end = std::max(out[outCount - 1].end, ranges[i].end);
        } else {
          out[outCount++] = ranges[i];
        }
      }
    };

    // Where the CTAB names them, those locations replace the reserved defaults rather than adding
    // to them. Excluding both would drop c0..c4 from the hash for a shader that keeps real
    // per-draw state there, letting draws that differ collide on the same hash.
    uint32_t viewProjReg = kUe3VsrViewProjMatrixRegister;
    uint32_t viewProjRegCount = 4;
    uint32_t viewOriginReg = kUe3VsrViewOriginRegister;
    uint32_t viewOriginRegCount = 1;
    if (ctabInfo.hasViewProjectionMatrix && ctabInfo.viewProjectionMatrixRegisterCount > 0) {
      viewProjReg = ctabInfo.viewProjectionMatrixRegisterIndex;
      viewProjRegCount = ctabInfo.viewProjectionMatrixRegisterCount;
    }
    if (ctabInfo.hasCameraPosition && ctabInfo.cameraPositionRegisterCount > 0) {
      viewOriginReg = ctabInfo.cameraPositionRegisterIndex;
      viewOriginRegCount = ctabInfo.cameraPositionRegisterCount;
    }
    add(viewProjReg, viewProjRegCount);
    add(viewOriginReg, viewOriginRegCount);

    Ue3VsHashExclusions result;
    std::array<Range, Ue3VsHashExclusions::kMaxRanges> cameraRaw = raw;
    const uint32_t cameraRawCount = rawCount;
    finalize(cameraRaw, cameraRawCount, result.cameraOnly, result.cameraOnlyCount);

    if (symbols != nullptr) {
      auto contains = [](const std::string& haystack, const char* needle) {
        return haystack.find(needle) != std::string::npos;
      };
      for (const Ue3VsConstantSymbol& symbol : *symbols) {
        const std::string name = toLowerAscii(symbol.name);
        // Shading-only: these reach interpolators, never vertex positions.
        if (contains(name, "lightmapscale") ||
            contains(name, "light_map_scale") ||
            contains(name, "lightmapcoordinatescalebias") ||
            contains(name, "light_map_coordinate_scale_bias") ||
            contains(name, "shadowcoordinatescalebias") ||
            contains(name, "shadow_coordinate_scale_bias")) {
          add(symbol.registerIndex, symbol.registerCount);
        }
      }
    }

    std::array<Range, Ue3VsHashExclusions::kMaxRanges> shadingRaw = raw;
    const uint32_t shadingRawCount = rawCount;
    finalize(shadingRaw, shadingRawCount, result.cameraAndShading, result.cameraAndShadingCount);

    if (ctabInfo.hasLocalToWorld) {
      add(ctabInfo.localToWorldRegisterIndex, ctabInfo.localToWorldRegisterCount);
    }
    if (ctabInfo.hasWorldToLocal) {
      add(ctabInfo.worldToLocalRegisterIndex, ctabInfo.worldToLocalRegisterCount);
    }
    finalize(raw, rawCount, result.withPlacement, result.withPlacementCount);

    return result;
  }

  XXH64_hash_t D3D9Rtx::computeUe3StableVertexShaderHash(bool* outHashedFloatConstsWithExclusions) const {
    if (outHashedFloatConstsWithExclusions != nullptr) {
      *outHashedFloatConstsWithExclusions = false;
    }

    if (d3d9State().vertexShader.ptr() == nullptr) {
      return kEmptyHash;
    }

    const D3D9ConstantSets& cb = m_parent->m_consts[DxsoProgramTypes::VertexShader];
    XXH64_hash_t hash = d3d9State().vertexShader->GetCommonShader()->GetBytecodeHash();

    const uint32_t floatConstRegCount = cb.meta.maxConstIndexF;
    const uint8_t* const floatConstBase = reinterpret_cast<const uint8_t*>(&d3d9State().vsConsts.fConsts[0]);

    auto hashFloatConstRange = [&](uint32_t beginReg, uint32_t endReg) {
      beginReg = std::min(beginReg, floatConstRegCount);
      endReg = std::min(endReg, floatConstRegCount);
      if (beginReg >= endReg) {
        return;
      }

      const size_t offsetBytes = size_t(beginReg) * sizeof(Vector4);
      const size_t sizeBytes = size_t(endReg - beginReg) * sizeof(Vector4);
      hash = XXH3_64bits_withSeed(floatConstBase + offsetBytes, sizeBytes, hash);
    };

    // The exclusion ranges are a pure function of the shader's CTAB, so they are resolved once per
    // shader and borrowed here. This path runs for every draw; it must not build or sort anything.
    bool hashedFloatConstsWithExclusions = false;
    if (m_frameOptions.ue3EngineMode &&
        floatConstRegCount > 0) {
      // A shader with no resolved CTAB still must not hash the reserved camera registers, or its
      // identity would move with the view.
      static const Ue3VsHashExclusions s_reservedOnly = buildUe3VsHashExclusions(Ue3VsShaderCtabInfo {}, nullptr);
      const Ue3VsHashExclusions& exclusions =
        m_currentUe3VsHashExclusions != nullptr ? *m_currentUe3VsHashExclusions : s_reservedOnly;
      const bool excludePlacement = m_frameOptions.ue3ExcludePlacementFromVertexShaderHash;
      // Dropping the shading-only constants is free and keeps the hash off the lightmap policy,
      // so UE3 mode takes it unconditionally; the placement registers stay opt-in for their cost.
      const bool excludeShading = m_frameOptions.ue3EngineMode;
      const Ue3VsHashExclusions::Range* ranges = exclusions.cameraOnly.data();
      uint32_t rangeCount = exclusions.cameraOnlyCount;
      if (excludePlacement) {
        ranges = exclusions.withPlacement.data();
        rangeCount = exclusions.withPlacementCount;
      } else if (excludeShading) {
        ranges = exclusions.cameraAndShading.data();
        rangeCount = exclusions.cameraAndShadingCount;
      }

      if (rangeCount > 0) {
        uint32_t cursor = 0;
        for (uint32_t i = 0; i < rangeCount; i++) {
          if (ranges[i].begin >= floatConstRegCount) {
            break;
          }
          hashFloatConstRange(cursor, ranges[i].begin);
          cursor = std::max(cursor, ranges[i].end);
        }
        hashFloatConstRange(cursor, floatConstRegCount);
        hashedFloatConstsWithExclusions = true;
      }
    }

    if (!hashedFloatConstsWithExclusions && floatConstRegCount > 0) {
      hash = XXH3_64bits_withSeed(
        &d3d9State().vsConsts.fConsts[0],
        size_t(floatConstRegCount) * sizeof(Vector4),
        hash);
    }

    if (cb.meta.maxConstIndexI > 0) {
      hash = XXH3_64bits_withSeed(
        &d3d9State().vsConsts.iConsts[0],
        size_t(cb.meta.maxConstIndexI) * sizeof(int) * 4,
        hash);
    }
    if (cb.meta.maxConstIndexB > 0) {
      hash = XXH3_64bits_withSeed(
        &d3d9State().vsConsts.bConsts[0],
        size_t(cb.meta.maxConstIndexB) * sizeof(uint32_t) / 32,
        hash);
    }

    if (outHashedFloatConstsWithExclusions != nullptr) {
      *outHashedFloatConstsWithExclusions = hashedFloatConstsWithExclusions;
    }

    return hash;
  }

  Ue3GeometryMemo::Lookup Ue3GeometryMemo::lookup(const XXH64_hash_t key, const uint32_t currentFrame, const bool selfCheck) {
    Lookup result;
    const auto it = m_entries.find(key);
    if (it == m_entries.end()) {
      result.publishTo = std::make_shared<Ue3GeometryMemoEntry>();
      result.publishTo->lastFrameTouched = currentFrame;
      m_entries.emplace(key, result.publishTo);
      return result;
    }

    Ue3GeometryMemoEntry& entry = *it->second;
    entry.lastFrameTouched = currentFrame;
    if (!entry.hashesReady.load(std::memory_order_acquire)) {
      // The worker from an earlier frame is still busy: hash in full, without publishing a second time.
      return result;
    }
    if (selfCheck) {
      // The fresh result goes into a new entry that replaces this one in the map; the old one is
      // only read from now on.
      result.verifyAgainst = it->second;
      result.publishTo = std::make_shared<Ue3GeometryMemoEntry>();
      result.publishTo->lastFrameTouched = currentFrame;
      it->second = result.publishTo;
      return result;
    }
    result.ready = &entry;
    return result;
  }

  void Ue3GeometryMemo::prune(const uint32_t currentFrame) {
    constexpr uint32_t kMaxUntouchedFrames = 600;
    // In-flight workers hold the entry via shared_ptr, so erasing here is always safe.
    m_entries.erase_if([&](auto it) {
      return currentFrame - it->second->lastFrameTouched > kMaxUntouchedFrames;
    });
  }

  // Skinned instances of a mesh share its bind-pose buffers, so the first bone's translation tells them
  // apart. The bones are RefToLocal: the translation must go through LocalToWorld, or every instance
  // anchors near the world origin.
  void D3D9Rtx::updateUe3SkinnedDrawIdentity() {
    m_activeDrawCallState.m_hasSkinnedWorldAnchor = false;
    // Only VS-skinned draws assign a bone hash, so reset it: a leaked one churns every static draw's BLAS
    // refit, draw call cache match and replacement identity as the pose animates.
    m_activeDrawCallState.skinningData = SkinningData();
    const bool usesVertexShaderSkinning =
      m_frameOptions.ue3EngineMode &&
      m_parent->UseProgrammableVS() &&
      m_currentUe3CtabInfo.has_value() &&
      m_currentUe3CtabInfo->hasBoneMatrices;
    if (usesVertexShaderSkinning &&
        m_currentUe3CtabInfo->boneMatricesRegisterCount >= 3) {
      const D3D9ConstantSets& cb = m_parent->m_consts[DxsoProgramTypes::VertexShader];
      const uint32_t floatConstRegCount = cb.meta.maxConstIndexF;
      const uint32_t boneReg = m_currentUe3CtabInfo->boneMatricesRegisterIndex;
      const uint32_t boneRegCount =
        std::min(m_currentUe3CtabInfo->boneMatricesRegisterCount, floatConstRegCount > boneReg ? floatConstRegCount - boneReg : 0u);
      if (boneRegCount >= 3) {
        const auto& fConsts = d3d9State().vsConsts.fConsts;
        // UE3 bone matrices are float4x3 (3 float4 rows per bone) and the translation lives in
        // the .w of the first three rows of the first bone
        const Vector3 boneTranslation(
          fConsts[boneReg + 0].w,
          fConsts[boneReg + 1].w,
          fConsts[boneReg + 2].w);
        m_activeDrawCallState.m_skinnedWorldAnchor =
          (m_activeDrawCallState.transformData.objectToWorld * Vector4(boneTranslation, 1.0f)).xyz();
        m_activeDrawCallState.m_hasSkinnedWorldAnchor = true;

        // processSkinning() returns no SkinningData for programmable-VS draws, so skinningData
        // stays default (numBones == 0) and this bone hash will not be overwritten by finalise
        m_activeDrawCallState.skinningData.boneHash =
          XXH3_64bits(&fConsts[boneReg], size_t(boneRegCount) * sizeof(Vector4));
      }
    }
  }

  // Identifies a CPU-skinned mesh's double-buffered vertices by its index buffer, which stays the same.
  // See "First person and the player model" in UE3Compatibility.md.
  void D3D9Rtx::updateUe3DynamicMeshIdentity(const IndexContext& indexContext,
                                             const VertexContext vertexContext[caps::MaxStreams],
                                             RasterGeometry& geoData) {
    if (!m_frameOptions.ue3EngineMode || indexContext.ibo == nullptr) {
      return;
    }

    const D3D9VertexElements& elements = d3d9State().vertexDecl->GetElements();
    const auto position = std::find_if(elements.begin(), elements.end(), [](const D3DVERTEXELEMENT9& element) {
      return element.Usage == D3DDECLUSAGE_POSITION && element.UsageIndex == 0;
    });
    if (position == elements.end() || position->Stream >= caps::MaxStreams) {
      return;
    }

    // A ring pool's draws read slices of shared dynamic buffers instead, which identify no mesh.
    const VertexContext& positions = vertexContext[position->Stream];
    if (positions.pVBO == nullptr || (positions.pVBO->Desc()->Usage & D3DUSAGE_DYNAMIC) == 0 || positions.offset != 0) {
      return;
    }
    const D3D9_BUFFER_DESC& indexDesc = *indexContext.ibo->Desc();
    const uint64_t indexBytes = uint64_t(geoData.indexCount) * (indexContext.indexType == VK_INDEX_TYPE_UINT32 ? 4u : 2u);
    if ((indexDesc.Usage & D3DUSAGE_DYNAMIC) != 0 && indexBytes * 4 < indexDesc.Size) {
      return;
    }

    geoData.sourceVertexBufferAddress = indexContext.ibo->GetBuffer<D3D9_COMMON_BUFFER_TYPE_MAPPING>().ptr();
  }

  // Serves static IA draws' geometry hashes from the memo. A first sighting hashes on a worker, which
  // publishes into the entry.
  D3D9Rtx::Ue3GeometryMemoLookup D3D9Rtx::lookupUe3GeometryMemo(const IndexContext& indexContext,
                                                                const VertexContext vertexContext[caps::MaxStreams],
                                                                const DrawContext& drawContext,
                                                                RasterGeometry& geoData) {
    Ue3GeometryMemoLookup memo;
    
    const bool canMemoizeIaGeometry = canMemoizeUe3IaGeometryHashes(indexContext, vertexContext, geoData);
    memo.key =
      canMemoizeIaGeometry
        ? computeUe3IaGeometryMemoKey(indexContext, vertexContext, drawContext, geoData)
        : kEmptyHash;

    // Keyed off structural eligibility rather than the cache's, so it still explains a dormant cache.
    if (m_frameOptions.ue3LogVertexConstantChurn &&
        canMemoizeIaGeometry &&
        isUe3StaticVertexCaptureCacheEligible(indexContext, vertexContext, geoData)) {
      trackUe3ConstantChurn(memo.key, geoData);
    }

    if (canMemoizeIaGeometry) {
      // rtx.d3d9.ue3GeometryMemoSelfCheckFrames: hash in full and have the worker compare against the published entry.
      const uint32_t selfCheckFrames = m_frameOptions.ue3GeometryMemoSelfCheckFrames;
      const bool selfCheck = selfCheckFrames != 0 && (m_ue3FrameCounter % selfCheckFrames) == 0;
      Ue3GeometryMemo::Lookup found =
        m_ue3GeometryMemo.lookup(memo.key, m_parent->GetDXVKDevice()->getCurrentFrameId(), selfCheck);
      memo.publishTo = std::move(found.publishTo);
      memo.verifyAgainst = std::move(found.verifyAgainst);
      if (found.ready != nullptr) {
        const Ue3GeometryMemoEntry& entry = *found.ready;
        GeometryHashes hashes;
        for (uint32_t i = 0; i < uint32_t(HashComponents::Count); i++) {
          hashes[HashComponents(i)] = entry.componentHashes[i];
        }
        hashes[HashComponents::VertexShader] = computeGeometryVertexShaderHash();
        hashes.precombine();
        geoData.hashes = hashes;
        memo.served = true;
        if (entry.aabbReady.load(std::memory_order_acquire)) {
          geoData.boundingBox = entry.boundingBox;
        } else {
          geoData.futureBoundingBox = computeAxisAlignedBoundingBox(geoData);
        }
      }
    }

    return memo;
  }

}
