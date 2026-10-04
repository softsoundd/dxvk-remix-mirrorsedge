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

  // UE3 instances a mesh by leaving its placement out of the shader constants entirely: the mesh
  // streams carry D3DSTREAMSOURCE_INDEXEDDATA | instanceCount, one further stream is tagged
  // D3DSTREAMSOURCE_INSTANCEDATA, and the vertex factory reads InstanceOffset (TEXCOORD1) plus the
  // three basis axes (TEXCOORD2..4) out of it. See FParticleInstancedMeshVertexFactory::InitRHI and
  // FFoliageVertexFactory::InitRHI in the UE3 engine, and GetInstanceToWorld in
  // FoliageVertexFactory.usf which both compile down to that layout.
  Ue3InstancingInfo D3D9Rtx::resolveUe3Instancing(
      const D3D9VertexElements& elements,
      const std::array<UINT, caps::MaxStreams>& streamFreq,
      const uint32_t instanceCount) {
    Ue3InstancingInfo info;

    // D3D9 only honours a stream-0 instance count when some stream the declaration reads is
    // tagged as instance data; GenerateDrawInfo gates the replayed draw the same way, so the
    // scene has to agree or it would decompose a draw the hardware runs once.
    uint32_t usedStreamMask = 0;
    for (const auto& element : elements) {
      if (element.Stream < caps::MaxStreams) {
        usedStreamMask |= 1u << element.Stream;
      }
    }

    for (const uint32_t s : bit::BitMask(usedStreamMask)) {
      if ((streamFreq[s] & D3DSTREAMSOURCE_INSTANCEDATA) != 0) {
        info.instanceDataStreamMask |= 1u << s;
      }
    }

    if (info.instanceDataStreamMask == 0) {
      return info;
    }
    info.instanceCount = std::max(instanceCount, 1u);

    // A complete basis is all four elements, FLOAT3, on one and the same instance-data stream.
    // Anything short of that is some other instancing scheme and must not be decomposed.
    const D3DVERTEXELEMENT9* basis[4] = {};
    for (const auto& element : elements) {
      if (element.Usage != D3DDECLUSAGE_TEXCOORD ||
          element.UsageIndex < 1 || element.UsageIndex > 4 ||
          element.Type != D3DDECLTYPE_FLOAT3 ||
          element.Stream >= caps::MaxStreams ||
          (info.instanceDataStreamMask & (1u << element.Stream)) == 0) {
        continue;
      }
      basis[element.UsageIndex - 1] = &element;
    }

    if (basis[0] == nullptr || basis[1] == nullptr || basis[2] == nullptr || basis[3] == nullptr) {
      return info;
    }
    if (basis[1]->Stream != basis[0]->Stream ||
        basis[2]->Stream != basis[0]->Stream ||
        basis[3]->Stream != basis[0]->Stream) {
      return info;
    }

    info.hasInstanceTransform = true;
    info.transformStream = basis[0]->Stream;
    info.offsetByteOffset = basis[0]->Offset;
    info.axisByteOffsets[0] = basis[1]->Offset;
    info.axisByteOffsets[1] = basis[2]->Offset;
    info.axisByteOffsets[2] = basis[3]->Offset;
    return info;
  }

  bool D3D9Rtx::readUe3InstanceTransforms(const VertexContext vertexContext[caps::MaxStreams],
                                          std::vector<Ue3DecomposedInstance>& instances,
                                          const char** outReason) const {
    instances.clear();

    auto fail = [&](const char* reason) {
      if (outReason != nullptr) {
        *outReason = reason;
      }
      return false;
    };
    if (outReason != nullptr) {
      *outReason = "";
    }

    const Ue3InstancingInfo& info = m_currentUe3Instancing;
    if (!info.hasInstanceTransform || info.instanceCount == 0) {
      return fail("no per-instance transform basis in the declaration");
    }

    const VertexContext& ctx = vertexContext[info.transformStream];
    if (ctx.stride == 0) {
      return fail("instance stream has no stride");
    }
    if (ctx.mappedSlice.mapPtr == nullptr) {
      return fail("instance stream is not host-visible");
    }

    // The four FLOAT3 reads have to stay inside the record, and the whole array inside the slice.
    uint32_t maxByteOffset = info.offsetByteOffset;
    for (const uint32_t axisOffset : info.axisByteOffsets) {
      maxByteOffset = std::max(maxByteOffset, axisOffset);
    }
    if (maxByteOffset + sizeof(float) * 3 > ctx.stride) {
      return fail("instance basis does not fit the instance stream's stride");
    }

    const size_t firstByte = size_t(ctx.offset);
    const size_t requiredBytes = size_t(ctx.stride) * info.instanceCount;
    if (firstByte + requiredBytes > ctx.mappedSlice.length) {
      return fail("instance stream is shorter than the draw's instance count");
    }

    const uint8_t* records = static_cast<const uint8_t*>(ctx.mappedSlice.mapPtr) + firstByte;
    auto readVector3 = [](const uint8_t* record, const uint32_t byteOffset) {
      float v[3];
      std::memcpy(v, record + byteOffset, sizeof(v));
      return Vector3(v[0], v[1], v[2]);
    };

    instances.reserve(info.instanceCount);
    for (uint32_t i = 0; i < info.instanceCount; i++) {
      const uint8_t* record = records + size_t(ctx.stride) * i;
      const Vector3 offset = readVector3(record, info.offsetByteOffset);
      const Vector3 xAxis = readVector3(record, info.axisByteOffsets[0]);
      const Vector3 yAxis = readVector3(record, info.axisByteOffsets[1]);
      const Vector3 zAxis = readVector3(record, info.axisByteOffsets[2]);

      // PhysX only writes the live prefix of an NxFluid emitter's instance buffer, so trailing
      // records hold whatever the previous frame left - or nothing at all on a fresh allocation.
      // A zero or non-finite basis is not a placement.
      const bool finite =
        std::isfinite(offset.x) && std::isfinite(offset.y) && std::isfinite(offset.z) &&
        std::isfinite(xAxis.x) && std::isfinite(xAxis.y) && std::isfinite(xAxis.z) &&
        std::isfinite(yAxis.x) && std::isfinite(yAxis.y) && std::isfinite(yAxis.z) &&
        std::isfinite(zAxis.x) && std::isfinite(zAxis.y) && std::isfinite(zAxis.z);
      if (!finite) {
        continue;
      }
      if (std::abs(dot(xAxis, cross(yAxis, zAxis))) <= 1e-12f) {
        continue;
      }

      // Row-vector layout to match Remix's Matrix4 (basis in rows 0..2, translation in row 3),
      // identical to the FMatrix(XAxis, YAxis, ZAxis, Location) the engine's non-instanced
      // NxFluid mesh path hands to FMeshElement::LocalToWorld.
      Ue3DecomposedInstance instance;
      instance.instanceToObject[0] = Vector4(xAxis.x, xAxis.y, xAxis.z, 0.0f);
      instance.instanceToObject[1] = Vector4(yAxis.x, yAxis.y, yAxis.z, 0.0f);
      instance.instanceToObject[2] = Vector4(zAxis.x, zAxis.y, zAxis.z, 0.0f);
      instance.instanceToObject[3] = Vector4(offset.x, offset.y, offset.z, 1.0f);
      // The buffer position, not the position in this vector: a degenerate record earlier in the
      // buffer must not shift the identity of everything after it.
      instance.sourceIndex = i;
      instances.push_back(instance);
    }

    if (instances.empty()) {
      return fail("every instance's basis was degenerate");
    }
    return true;
  }

  // Whatever these bounds drop, the kept set has to be the same set next frame: an instance that
  // comes and goes as the camera moves flickers, and a selection that reorders the survivors also
  // renames them (see Ue3DecomposedInstance::sourceIndex). Hence the count clamp keeps the lowest
  // source indices rather than the nearest to the camera. Distance culling is view-dependent by
  // definition and can pop at its boundary, which is why it is opt-in.
  void D3D9Rtx::cullAndClampUe3InstanceTransforms(std::vector<Ue3DecomposedInstance>& instances,
                                                  uint32_t& outCulledByDistance,
                                                  uint32_t& outCulledByBudget) const {
    outCulledByDistance = 0;
    outCulledByBudget = 0;

    const uint32_t budget = std::max(m_frameOptions.ue3MaxDecomposedInstances, 1u);
    const float cullDistance = m_frameOptions.ue3DecomposedInstanceCullDistance;
    if (cullDistance <= 0.0f && instances.size() <= budget) {
      return;
    }

    if (cullDistance > 0.0f) {
      const DrawCallTransforms& transformData = m_activeDrawCallState.transformData;

      // worldToView is affine, so its inverse's translation row is the view origin in world space.
      // A camera that cannot be inverted leaves distances meaningless; skip to the count clamp.
      const double det = determinant(transformData.worldToView);
      if (std::isfinite(det) && std::abs(det) > 1e-24) {
        const Matrix4 viewToWorld = inverseAffine(transformData.worldToView);
        const Vector3 cameraPosition(viewToWorld[3].x, viewToWorld[3].y, viewToWorld[3].z);
        if (std::isfinite(cameraPosition.x) && std::isfinite(cameraPosition.y) &&
            std::isfinite(cameraPosition.z)) {
          const float cullDistanceSq = cullDistance * cullDistance;
          const size_t before = instances.size();
          // Order-preserving, so the surviving instances keep their relative buffer order.
          instances.erase(
            std::remove_if(instances.begin(), instances.end(),
                           [&](const Ue3DecomposedInstance& instance) {
                             // Carried through the draw's object transform, which is identity for
                             // UE3's instanced factories but composed anyway to stay general.
                             const Vector4 world = transformData.objectToWorld *
                               Vector4(instance.instanceToObject[3].x,
                                       instance.instanceToObject[3].y,
                                       instance.instanceToObject[3].z, 1.0f);
                             const Vector3 delta = Vector3(world.x, world.y, world.z) - cameraPosition;
                             return dot(delta, delta) > cullDistanceSq;
                           }),
            instances.end());
          outCulledByDistance = uint32_t(before - instances.size());
        }
      }
    }

    if (instances.size() > budget) {
      outCulledByBudget = uint32_t(instances.size() - budget);
      instances.resize(budget);
    }
  }

  // Names the batch an instanced draw belongs to, without involving any transform.
  //
  // The mesh streams say which mesh is being instanced but not which component is instancing it, and
  // two piles of the same debris share them. The instance buffer would distinguish those but cannot
  // serve as an identity - RenderNxFluidInstanced creates a fresh one per frame unless its pool hands
  // one back - so batches are matched to the previous frame's by continuity of their own centroid,
  // which holds because a batch as a whole barely moves even while its instances do.
  XXH64_hash_t D3D9Rtx::resolveUe3InstancedBatchKey(const RasterGeometry& geoData,
                                                    const std::vector<Ue3DecomposedInstance>& instances) {
    if (instances.empty()) {
      return kEmptyHash;
    }

    struct MeshKey {
      const void* pVertexDecl;
      const void* pVertexBuffer0;
      const void* pIndexBuffer;
      uint32_t vertexCount;
      uint32_t indexCount;
    };
    const MeshKey meshKey = {
      d3d9State().vertexDecl.ptr(),
      d3d9State().vertexBuffers[0].vertexBuffer.ptr(),
      d3d9State().indices.ptr(),
      geoData.vertexCount,
      geoData.indexCount,
    };
    const XXH64_hash_t meshHash = XXH3_64bits(&meshKey, sizeof(meshKey));

    Vector3 centroid(0.f, 0.f, 0.f);
    for (const Ue3DecomposedInstance& instance : instances) {
      centroid += Vector3(instance.instanceToObject[3].x,
                          instance.instanceToObject[3].y,
                          instance.instanceToObject[3].z);
    }
    centroid *= 1.0f / float(instances.size());

    const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();
    std::vector<Ue3InstancedBatchRecord>& records = m_ue3InstancedBatches[meshHash];

    // Retire batches that have not been drawn for a while so a level change cannot leave a stale
    // record that a new batch of the same mesh would latch onto.
    constexpr uint32_t kRetireAfterFrames = 120;
    records.erase(
      std::remove_if(records.begin(), records.end(),
                     [&](const Ue3InstancedBatchRecord& record) {
                       return currentFrame - record.lastFrame > kRetireAfterFrames;
                     }),
      records.end());

    // A whole batch travelling this far in one frame is a different batch, not the same one moved.
    constexpr float kMaxCentroidDriftSqr = 1000.f * 1000.f;
    Ue3InstancedBatchRecord* nearest = nullptr;
    float nearestDistSqr = kMaxCentroidDriftSqr;
    for (Ue3InstancedBatchRecord& record : records) {
      if (record.claimedFrame == currentFrame) {
        continue; // another draw already matched this batch this frame
      }
      const Vector3 delta = record.centroid - centroid;
      const float distSqr = dot(delta, delta);
      if (distSqr < nearestDistSqr) {
        nearestDistSqr = distSqr;
        nearest = &record;
      }
    }

    if (nearest == nullptr) {
      Ue3InstancedBatchRecord record;
      record.id = XXH3_64bits_withSeed(&m_ue3NextInstancedBatchId, sizeof(m_ue3NextInstancedBatchId), meshHash);
      ++m_ue3NextInstancedBatchId;
      records.push_back(record);
      nearest = &records.back();
    }

    nearest->centroid = centroid;
    nearest->lastFrame = currentFrame;
    nearest->claimedFrame = currentFrame;
    return nearest->id;
  }

  void D3D9Rtx::trackUe3InstanceOrderStability(const XXH64_hash_t batchKey,
                                               const std::vector<Ue3DecomposedInstance>& instances) {
    const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();

    // One entry per batch identity, each holding a translation per instance. Batch identities are
    // minted afresh across a level change, so drop the lot rather than grow without bound.
    constexpr size_t kMaxProbes = 64;
    if (m_ue3InstanceOrderProbes.size() > kMaxProbes) {
      m_ue3InstanceOrderProbes.clear();
    }

    Ue3InstanceOrderProbe& probe = m_ue3InstanceOrderProbes[batchKey];
    const bool consecutiveFrames = probe.lastFrame != 0 && probe.lastFrame + 1 == currentFrame;

    if (consecutiveFrames) {
      if (probe.translations.size() != instances.size()) {
        // A length change shifts every index past the first added or removed instance, so this
        // batch cannot be index-paired at all this frame.
        ++m_ue3InstancedStatOrderSizeChanges;
      } else {
        ++m_ue3InstancedStatOrderComparableBatches;
        const float stableThreshold = RtxOptions::uniqueObjectDistance();
        for (size_t i = 0; i < instances.size(); i++) {
          const Matrix4& m = instances[i].instanceToObject;
          const Vector3 current(m[3].x, m[3].y, m[3].z);
          const Vector3 delta = current - probe.translations[i];
          const float displacement = std::sqrt(dot(delta, delta));
          ++m_ue3InstancedStatOrderPairs;
          m_ue3InstancedStatOrderDisplacementSum += double(displacement);
          m_ue3InstancedStatOrderDisplacementMax =
            std::max(m_ue3InstancedStatOrderDisplacementMax, displacement);
          if (displacement <= stableThreshold) {
            ++m_ue3InstancedStatOrderStablePairs;
          }
        }
      }
    }

    probe.lastFrame = currentFrame;
    probe.translations.resize(instances.size());
    for (size_t i = 0; i < instances.size(); i++) {
      const Matrix4& m = instances[i].instanceToObject;
      probe.translations[i] = Vector3(m[3].x, m[3].y, m[3].z);
    }
  }

  // An instanced mesh factory's world position is GetInstanceToWorld(Input) * Input.Position
  // (FoliageVertexFactory.usf, shared by FFoliageVertexFactory and
  // FParticleInstancedMeshVertexFactory), so the declaration's POSITION is object space by
  // construction - there is no shader constant involved and nothing to prove about the transform.
  // That makes the conservatism canUseUe3NativeLocalVertexCapture applies to Local draws, where
  // reading the input assembler is a guess about what the shader does, unnecessary here.
  bool D3D9Rtx::canUseUe3InstancedMeshVertexPositions(const RasterGeometry& geoData,
                                                     const char** outReason) const {
    auto fail = [&](const char* reason) {
      if (outReason != nullptr) {
        *outReason = reason;
      }
      return false;
    };

    if (!m_frameOptions.ue3EngineMode) {
      return fail("UE3 engine mode disabled");
    }
    if (!m_currentUe3Instancing.hasInstanceTransform) {
      return fail("no per-instance transform basis in the declaration");
    }
    if (m_currentUe3VertexFactory != Ue3VertexFactoryType::Foliage &&
        m_currentUe3VertexFactory != Ue3VertexFactoryType::ParticleInstancedMesh) {
      return fail("not a recognised instanced mesh vertex factory");
    }
    // Object-space positions are only meaningful alongside the transforms that place them.
    if (m_ue3InstanceTransformReadFailure != nullptr) {
      return fail(m_ue3InstanceTransformReadFailure);
    }
    if (!geoData.positionBuffer.defined()) {
      return fail("no input-assembler position buffer");
    }
    // Skinning would move the vertices after the input assembler; instanced factories never skin.
    if (geoData.blendWeightBuffer.defined() || geoData.blendIndicesBuffer.defined()) {
      return fail("draw carries skinning data");
    }

    if (outReason != nullptr) {
      *outReason = "";
    }
    return true;
  }

  // Folds the per-instance VS transform constants (LocalToWorld, or the leading bone
  // matrix rows for skinned draws) into `seed`. Identities derived from shared
  // buffers/ranges alone cannot distinguish different placements of the same mesh.
  XXH64_hash_t D3D9Rtx::mixUe3InstanceTransformConstants(XXH64_hash_t seed) const {
    if (!m_currentUe3CtabInfo.has_value()) {
      return seed;
    }

    const Ue3VsShaderCtabInfo& ctabInfo = *m_currentUe3CtabInfo;
    auto mixRange = [&](const uint32_t reg, uint32_t count) {
      count = std::min(count, 4u);
      if (count == 0 || reg + count > caps::MaxFloatConstantsVS) {
        return;
      }
      seed = XXH3_64bits_withSeed(&d3d9State().vsConsts.fConsts[reg], count * sizeof(Vector4), seed);
    };

    if (ctabInfo.hasLocalToWorld) {
      mixRange(ctabInfo.localToWorldRegisterIndex, ctabInfo.localToWorldRegisterCount);
    }
    if (ctabInfo.hasBoneMatrices) {
      mixRange(ctabInfo.boneMatricesRegisterIndex, std::min(ctabInfo.boneMatricesRegisterCount, 3u));
    }
    return seed;
  }

  void D3D9Rtx::submitUe3DecomposedInstanceDrawCallStates(const DrawParameters& params) {
    ScopedCpuProfileZone();

    const bool timeSubmission = m_frameOptions.ue3LogInstancedDrawStats;
    const auto submitStart = timeSubmission ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point();

    // Every copy of the draw state shares the pending futures' task pointers, and a task result is
    // a one-shot: the first consumer disposes it and the rest would wait on a result that is never
    // set again. Resolve them here so all copies carry finished data. Steady state usually costs
    // nothing, because the geometry hash memo serves these draws without scheduling a future at all.
    // (The caller guarantees there is no pending skinning future.)
    RasterGeometry& geoData = m_activeDrawCallState.geometryData;
    if (geoData.futureGeometryHashes.valid()) {
      geoData.hashes = geoData.futureGeometryHashes.get();
    }
    if (geoData.futureBoundingBox.valid()) {
      geoData.boundingBox = geoData.futureBoundingBox.get();
    }

    const Matrix4 objectToWorld = m_activeDrawCallState.transformData.objectToWorld;
    const Matrix4 worldToView = m_activeDrawCallState.transformData.worldToView;
    const size_t instanceCount = m_ue3DecomposedInstances.size();

    const bool stableIdentity =
      m_frameOptions.ue3StableDecomposedInstanceIdentity && m_ue3DecomposedBatchKey != kEmptyHash;

    for (size_t i = 0; i < instanceCount; i++) {
      const Ue3DecomposedInstance& instance = m_ue3DecomposedInstances[i];

      DrawCallTransforms& transforms = m_activeDrawCallState.transformData;
      transforms.objectToWorld = objectToWorld * instance.instanceToObject;
      transforms.objectToView = worldToView * transforms.objectToWorld;
      transforms.sanitize();

      if (stableIdentity) {
        // Keyed on the instance's position in the game's buffer, not in this vector: culling changes
        // how many instances precede it, and renaming an instance costs it its history.
        const uint64_t sourceIndex = uint64_t(instance.sourceIndex);
        XXH64_hash_t instanceId =
          XXH3_64bits_withSeed(&sourceIndex, sizeof(sourceIndex), m_ue3DecomposedBatchKey);
        if (instanceId == kEmptyHash) {
          instanceId = 1; // kEmptyHash means "not a decomposed instance"
        }
        m_activeDrawCallState.decomposedInstanceId = instanceId;
      }

      if (i + 1 < instanceCount) {
        // push() leaves its argument untouched when the queue is full, so retrying with the same
        // copy is safe.
        DrawCallState instanceDrawCallState = m_activeDrawCallState;
        while (!m_drawCallStateQueue.push(std::move(instanceDrawCallState))) {
          Sleep(0);
        }
      } else {
        submitActiveDrawCallState();
      }

      m_parent->EmitCs([params, this](DxvkContext* ctx) {
        assert(dynamic_cast<RtxContext*>(ctx));
        DrawCallState drawCallState;
        if (m_drawCallStateQueue.pop(drawCallState)) {
          static_cast<RtxContext*>(ctx)->commitGeometryToRT(params, drawCallState);
        }
      });
    }

    if (timeSubmission) {
      m_ue3InstancedStatSubmitNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - submitStart).count());
    }
  }

}
