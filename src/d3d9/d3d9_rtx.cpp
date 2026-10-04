#pragma once

#include "d3d9_include.h"
#include "d3d9_state.h"

#include "d3d9_util.h"
#include "d3d9_buffer.h"

#include "d3d9_rtx.h"
#include "d3d9_rtx_ue3_helpers.h"
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
// NV-DXVK start: draw disposition statistics
#include "../dxvk/rtx_render/rtx_gpu_pass_timer.h"
// NV-DXVK end
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
  static const bool s_isDxvkResolutionEnvVarSet = (env::getEnvVar("DXVK_RESOLUTION_WIDTH") != "") || (env::getEnvVar("DXVK_RESOLUTION_HEIGHT") != "");
  
  #define CATEGORIES_REQUIRE_DRAW_CALL_STATE  InstanceCategories::Sky, InstanceCategories::Terrain
  #define CATEGORIES_REQUIRE_GEOMETRY_COPY    InstanceCategories::Terrain, InstanceCategories::WorldUI

  D3D9Rtx::D3D9Rtx(D3D9DeviceEx* d3d9Device, bool enableDrawCallConversion)
    : m_rtStagingData(d3d9Device->GetDXVKDevice(), "RtxStagingDataAlloc: D3D9", (VkMemoryPropertyFlagBits) (VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
    , m_parent(d3d9Device)
    , m_enableDrawCallConversion(enableDrawCallConversion)
    , m_pGeometryWorkers(enableDrawCallConversion ? std::make_unique<GeometryProcessor>(numGeometryProcessingThreads(), "geometry-processing") : nullptr) {
    BridgeMessageChannel::get().registerHandler(kGamePatchStatusMsgName, [](uint32_t active, uint32_t notFound) {
      s_ue3GamePatchStatus.store(kUe3GamePatchAnswered | (uint64_t(notFound & 0xFFFFu) << 16) | (active & 0xFFFFu),
                                 std::memory_order_relaxed);
      return true;
    });
  }

  D3D9Rtx::~D3D9Rtx() {
    // EndFrame saves on an interval, so the tail of a session would otherwise be lost.
    m_ue3TextureSpreadCache.save();
    m_ue3DiffuseSelectionCache.save();
  }

  void D3D9Rtx::SkinningMatrixPool::clear() {
    m_blockIndex = 0;
    m_nextIndexInBlock = 0;
  }

  const Matrix4* D3D9Rtx::SkinningMatrixPool::stageBones(const Matrix4* source, size_t matrixCount) {
    assert(matrixCount <= kMatricesPerBlock);
    if (matrixCount == 0) {
      return nullptr;
    }
    for (;;) {
      if (m_blocks.empty()) {
        m_blocks.push_back(std::make_unique<Block>());
        m_blockIndex = 0;
        m_nextIndexInBlock = 0;
      } else if (m_nextIndexInBlock + matrixCount > kMatricesPerBlock) {
        ++m_blockIndex;
        m_nextIndexInBlock = 0;
        if (m_blockIndex == m_blocks.size()) {
          m_blocks.push_back(std::make_unique<Block>());
        }
        continue;
      }
      break;
    }
    Matrix4* const dest = m_blocks[m_blockIndex]->m_matrices.data() + m_nextIndexInBlock;
    memcpy(dest, source, matrixCount * sizeof(Matrix4));
    m_nextIndexInBlock += matrixCount;
    return dest;
  }

  void D3D9Rtx::Initialize() {
    m_vsVertexCaptureData = m_parent->CreateConstantBuffer(false,
                                        sizeof(D3D9RtxVertexCaptureData),
                                        DxsoProgramType::VertexShader,
                                        DxsoConstantBuffers::VSVertexCaptureData);

    // Get constant buffer bindings from D3D9
    m_parent->EmitCs([vertexCaptureCB = m_vsVertexCaptureData](DxvkContext* ctx) {
      const uint32_t vsFixedFunctionConstants = computeResourceSlotId(DxsoProgramType::VertexShader, DxsoBindingType::ConstantBuffer, DxsoConstantBuffers::VSFixedFunction);
      const uint32_t psSharedStateConstants = computeResourceSlotId(DxsoProgramType::PixelShader, DxsoBindingType::ConstantBuffer, DxsoConstantBuffers::PSShared);
      static_cast<RtxContext*>(ctx)->setConstantBuffers(vsFixedFunctionConstants, psSharedStateConstants, vertexCaptureCB);
    });
  }

  void D3D9Rtx::refreshFrameOptionCache() {
    FrameOptionCache& o = m_frameOptions;

    // Reads use the xObject().get() accessor form (the same locked read as x()) so
    // the plain x() spelling never appears in this function: any plain option
    // accessor found in per-draw code is then, by construction, an un-snapshotted
    // per-draw locked read that should be moved into this cache.
    o.orthographicIsUI = orthographicIsUIObject().get();
    o.preTransformedVerticesIsUI = preTransformedVerticesIsUIObject().get();
    o.allowCubemaps = allowCubemapsObject().get();
    o.useVertexCapture = useVertexCaptureObject().get();
    o.useVertexCapturedNormals = useVertexCapturedNormalsObject().get();
    o.useWorldMatricesForShaders = useWorldMatricesForShadersObject().get();
    o.ue3EngineMode = ue3EngineModeObject().get();
    o.autoRaytracedRenderTargetFromFullscreenComposite = autoRaytracedRenderTargetFromFullscreenCompositeObject().get();
    o.rasterizeFullscreenCompositeToPrimary = rasterizeFullscreenCompositeToPrimaryObject().get();
    o.ue3MicConstantIdentity = ue3MicConstantIdentityObject().get();
    o.ue3MicExcludeRenderTargetsFromIdentity = ue3MicExcludeRenderTargetsFromIdentityObject().get();
    o.ue3MicVolatileConstantDetection = ue3MicVolatileConstantDetectionObject().get();
    o.ue3ReportMicIdentityChurn = ue3ReportMicIdentityChurnObject().get();
    o.ue3LogMaterialInstanceHash = ue3LogMaterialInstanceHashObject().get();
    o.ue3ForegroundDpgIsViewModel = ue3ForegroundDpgIsViewModelObject().get();
    o.conservativeOcclusionQueries = conservativeOcclusionQueriesObject().get();
    o.eventQueryCsCompletion = eventQueryCsCompletionObject().get();
    o.sequenceTrackedLockWaits = sequenceTrackedLockWaitsObject().get();
    o.discardCaptureOnlyDrawFragments = discardCaptureOnlyDrawFragmentsObject().get();
    o.skipRenderTargetCopies = skipRenderTargetCopiesObject().get();
    Ue3StaticVertexCaptureCache::Settings& captureCache = o.ue3StaticVertexCaptureCache;
    captureCache.enabled = ue3StaticLocalMeshVertexCaptureCacheObject().get();
    captureCache.warmupFrames = ue3StaticLocalMeshVertexCaptureCacheWarmupFramesObject().get();
    captureCache.budgetMiB = ue3StaticLocalMeshVertexCaptureCacheBudgetMiBObject().get();
    captureCache.maxEntries = ue3StaticLocalMeshVertexCaptureCacheMaxEntriesObject().get();
    captureCache.retentionFrames = ue3StaticLocalMeshVertexCaptureCacheRetentionFramesObject().get();
    captureCache.minReusePercent = ue3StaticLocalMeshVertexCaptureCacheMinReusePercentObject().get();
    captureCache.reuseProbeFrames = ue3StaticLocalMeshVertexCaptureCacheReuseProbeFramesObject().get();
    captureCache.logStats = ue3LogStaticVertexCaptureCacheStatsObject().get();
    o.ue3ExcludePlacementFromVertexShaderHash = ue3ExcludePlacementFromVertexShaderHashObject().get();
    o.ue3LogVertexConstantChurn = ue3LogVertexConstantChurnObject().get();
    o.ue3VertexConstantChurnMaxTrackedDraws = ue3VertexConstantChurnMaxTrackedDrawsObject().get();
    o.ue3StaticGeometryHashMemoization = ue3StaticGeometryHashMemoizationObject().get();
    o.ue3GeometryMemoSelfCheckFrames = ue3GeometryMemoSelfCheckFramesObject().get();
    o.ue3ExactVertexCapture = ue3ExactVertexCaptureObject().get();
    o.ue3RequireExactVertexCapture = ue3RequireExactVertexCaptureObject().get();
    o.ue3VertexCaptureSourceOverride = ue3VertexCaptureSourceOverrideObject().get();
    o.ue3NativeLocalMeshVertexCapture = ue3NativeLocalMeshVertexCaptureObject().get();
    o.ue3DecomposeInstancedDraws = ue3DecomposeInstancedDrawsObject().get();
    o.ue3MaxDecomposedInstances = ue3MaxDecomposedInstancesObject().get();
    o.ue3DecomposedInstanceCullDistance = ue3DecomposedInstanceCullDistanceObject().get();
    o.ue3StableDecomposedInstanceIdentity = ue3StableDecomposedInstanceIdentityObject().get();
    o.ue3LogInstancedDrawStats = ue3LogInstancedDrawStatsObject().get();
    o.ue3AutoDetectLightmapTextures = ue3AutoDetectLightmapTexturesObject().get() && o.ue3EngineMode;
    o.ue3ConstantAlbedoTintGain = ue3ConstantAlbedoTintGainObject().get();
    o.ue3HighlightTints = ue3HighlightTintsObject().get() && o.ue3EngineMode;
    o.ue3HighlightTintRequireMotion = ue3HighlightTintRequireMotionObject().get();
    o.ue3HighlightGlowIntensity = ue3HighlightGlowIntensityObject().get();
    o.ue3LogHighlightTints = ue3LogHighlightTintsObject().get();
    o.ue3HighlightDebugForceTint = ue3HighlightDebugForceTintObject().get();
    o.ue3ParticleVertexColor = ue3ParticleVertexColorObject().get() && o.ue3EngineMode;
    o.ue3MaterialFades = ue3MaterialFadesObject().get() && o.ue3EngineMode;
    o.ue3LogMaterialFades = ue3LogMaterialFadesObject().get();
    o.ue3MaterialFadeDebugForceCoverage = ue3MaterialFadeDebugForceCoverageObject().get();
    o.ue3LogUvResolution = ue3LogUvResolutionObject().get();
    o.ue3LogUvAffineDetail = ue3LogUvAffineDetailObject().get();
    o.ue3LogAlbedoSelection = ue3LogAlbedoSelectionObject().get();
    o.ue3LogCapturePrecision = ue3LogCapturePrecisionObject().get();
    o.ue3LogInstancedDraws = ue3LogInstancedDrawsObject().get();
    o.deferredUiReplay = deferredUiReplayObject().get();
    o.deferredUiRefreshSceneColor = deferredUiRefreshSceneColorObject().get();
    o.deferredUiHdrReplay = deferredUiHdrReplayObject().get();
    o.enableIndexBufferMemoization = enableIndexBufferMemoizationObject().get();
    o.poolVertexCaptureBuffers = poolVertexCaptureBuffersObject().get();

    o.enableRaytracing = RtxOptions::enableRaytracingObject().get();
    o.enableAlphaTest = RtxOptions::enableAlphaTestObject().get();
    o.enableAlphaBlend = RtxOptions::enableAlphaBlendObject().get();
    o.raytracedRenderTargetEnable = RtxOptions::RaytracedRenderTarget::enableObject().get();
    o.skipDrawCallsPostRTXInjection = RtxOptions::skipDrawCallsPostRTXInjectionObject().get();
    o.useBuffersDirectly = RtxOptions::useBuffersDirectlyObject().get();
    o.fogIgnoreSky = RtxOptions::fogIgnoreSkyObject().get();
    o.needsMeshBoundingBox = RtxOptions::needsMeshBoundingBox(); // derived helper, not an RtxOption
    o.validateCPUIndexData = RtxOptions::validateCPUIndexDataObject().get();
    o.alwaysCopyDecalGeometries = RtxOptions::alwaysCopyDecalGeometriesObject().get();
    o.terrainAsDecalsEnabledIfNoBaker = RtxOptions::terrainAsDecalsEnabledIfNoBakerObject().get();
    o.terrainAsDecalsAllowOverModulate = RtxOptions::terrainAsDecalsAllowOverModulateObject().get();
    o.enableMultiStageTextureFactorBlending = RtxOptions::enableMultiStageTextureFactorBlendingObject().get();
    o.ignoreAllVertexColorBakedLighting = RtxOptions::ignoreAllVertexColorBakedLightingObject().get();
    o.vertexColorIsBakedLighting = RtxOptions::vertexColorIsBakedLightingObject().get();
    o.logReplacementResolution = RtxOptions::logReplacementResolutionObject().get();
    o.drawCallRange = RtxOptions::drawCallRangeObject().get();

    refreshFrameOptionSets();
    const FrameOptionSets& s = m_frameOptionSets;
    o.uiTextures = &s.uiTextures;
    o.deferredUiTextures = &s.deferredUiTextures;
    o.deferredUiPixelShaders = &s.deferredUiPixelShaders;
    o.lightmapTextures = &s.lightmapTextures;
    o.neverAlbedoTextures = &s.neverAlbedoTextures;
    o.preferredAlbedoTextures = &s.preferredAlbedoTextures;
    o.smoothNormalsTextures = &s.smoothNormalsTextures;
    o.ignoreBakedLightingTextures = &s.ignoreBakedLightingTextures;
    o.raytracedRenderTargetTextures = &s.raytracedRenderTargetTextures;
    o.vsTexcoordCaptureOutlierTextures = &s.vsTexcoordCaptureOutlierTextures;
    o.ue3MicConstantIdentityExcludedShaders = &s.ue3MicConstantIdentityExcludedShaders;
    o.ue3MicConstantIdentityExcludedMaterials = &s.ue3MicConstantIdentityExcludedMaterials;
    o.ue3MicIdentityExcludedTextureDescHashes = &s.ue3MicIdentityExcludedTextureDescHashes;
    o.ue3HighlightTintExcludedMaterials = &s.ue3HighlightTintExcludedMaterials;
    o.ue3MaterialFadeExcludedMaterials = &s.ue3MaterialFadeExcludedMaterials;
    o.ue3TraceDrawTextureHashes = &s.ue3TraceDrawTextureHashes;
    o.replacementDebugHashes = &s.replacementDebugHashes;

    o.valid = true;
  }

  void D3D9Rtx::refreshFrameOptionSets() {
    const uint64_t generation = g_rtxOptionResolveGeneration.load(std::memory_order_acquire);
    if (generation == m_frameOptionSets.generation) {
      return;
    }

    // The accessors take the option mutex themselves, so gather the source addresses before holding it.
    FrameOptionSets& s = m_frameOptionSets;
    const std::pair<fast_unordered_set*, const fast_unordered_set*> sets[] = {
      { &s.uiTextures, &RtxOptions::uiTexturesObject().get() },
      { &s.deferredUiTextures, &RtxOptions::deferredUiTexturesObject().get() },
      { &s.deferredUiPixelShaders, &deferredUiPixelShadersObject().get() },
      { &s.lightmapTextures, &RtxOptions::lightmapTexturesObject().get() },
      { &s.neverAlbedoTextures, &RtxOptions::neverAlbedoTexturesObject().get() },
      { &s.preferredAlbedoTextures, &RtxOptions::preferredAlbedoTexturesObject().get() },
      { &s.smoothNormalsTextures, &RtxOptions::smoothNormalsTexturesObject().get() },
      { &s.ignoreBakedLightingTextures, &RtxOptions::ignoreBakedLightingTexturesObject().get() },
      { &s.raytracedRenderTargetTextures, &RtxOptions::raytracedRenderTargetTexturesObject().get() },
      { &s.vsTexcoordCaptureOutlierTextures, &vsTexcoordCaptureOutlierTexturesObject().get() },
      { &s.ue3MicConstantIdentityExcludedShaders, &ue3MicConstantIdentityExcludedShadersObject().get() },
      { &s.ue3MicConstantIdentityExcludedMaterials, &ue3MicConstantIdentityExcludedMaterialsObject().get() },
      { &s.ue3MicIdentityExcludedTextureDescHashes, &ue3MicIdentityExcludedTextureDescHashesObject().get() },
      { &s.ue3HighlightTintExcludedMaterials, &ue3HighlightTintExcludedMaterialsObject().get() },
      { &s.ue3MaterialFadeExcludedMaterials, &ue3MaterialFadeExcludedMaterialsObject().get() },
      { &s.ue3TraceDrawTextureHashes, &ue3TraceDrawTextureHashesObject().get() },
      { &s.replacementDebugHashes, &RtxOptions::replacementDebugHashesObject().get() },
    };

    // The CS thread assigns whole sets under this mutex when it resolves pending option changes.
    {
      std::lock_guard<std::mutex> lock(RtxOptionImpl::getUpdateMutex());
      for (const auto& [dst, src] : sets) {
        *dst = *src;
      }
    }
    s.albedoTagDigests.lightmap = digestTextureTags(s.lightmapTextures);
    s.albedoTagDigests.neverAlbedo = digestTextureTags(s.neverAlbedoTextures);
    s.albedoTagDigests.preferredAlbedo = digestTextureTags(s.preferredAlbedoTextures);
    s.generation = generation;
  }

  // NV-DXVK end

  template<typename T>
  void D3D9Rtx::copyIndices(const uint32_t indexCount, T*& pIndicesDst, T* pIndices, uint32_t& minIndex, uint32_t& maxIndex) {
    ScopedCpuProfileZone();

    assert(indexCount >= 3);

    // Find min/max index
    {
      ScopedCpuProfileZoneN("Find min/max");

      fast::findMinMax<T>(indexCount, pIndices, minIndex, maxIndex);
    }

    // Modify the indices if the min index is non-zero
    {
      ScopedCpuProfileZoneN("Copy indices");

      if (minIndex != 0) {
        fast::copySubtract<T>(pIndicesDst, pIndices, indexCount, (T) minIndex);
      } else {
        memcpy(pIndicesDst, pIndices, sizeof(T) * indexCount);
      }
    }
  }

  template<typename T>
  DxvkBufferSlice D3D9Rtx::processIndexBuffer(const uint32_t indexCount, const uint32_t startIndex, const IndexContext& indexCtx, uint32_t& minIndex, uint32_t& maxIndex) {
    ScopedCpuProfileZone();

    const uint32_t indexStride = sizeof(T);
    const size_t numIndexBytes = indexCount * indexStride;
    const size_t indexOffset = indexStride * startIndex;

    auto processing = [this, &indexCtx, indexCount](const size_t offset, const size_t size) -> D3D9CommonBuffer::RemixIndexBufferMemoizationData {
      D3D9CommonBuffer::RemixIndexBufferMemoizationData result;

      // Get our slice of the staging ring buffer
      result.slice = m_rtStagingData.alloc(CACHE_LINE_SIZE, size);

      // Acquire prevents the staging allocator from re-using this memory
      result.slice.buffer()->acquire(DxvkAccess::Read);

      const uint8_t* pBaseIndex = (uint8_t*) indexCtx.indexBuffer.mapPtr + offset;

      T* pIndices = (T*) pBaseIndex;
      T* pIndicesDst = (T*) result.slice.mapPtr(0);
      copyIndices<T>(indexCount, pIndicesDst, pIndices, result.min, result.max);

      return result;
    };

    if (m_frameOptions.enableIndexBufferMemoization && indexCtx.ibo != nullptr) {
      // If we have an index buffer, we can utilize memoization
      D3D9CommonBuffer::RemixIboMemoizer& memoization = indexCtx.ibo->remixMemoization;
      const auto result = memoization.memoize(indexOffset, numIndexBytes, processing);
      minIndex = result.min;
      maxIndex = result.max;
      return result.slice;
    }

    // No index buffer (so no memoization) - this could be a DrawPrimitiveUP call (where IB data is passed inline)
    const auto result = processing(indexOffset, numIndexBytes);
    minIndex = result.min;
    maxIndex = result.max;
    return result.slice;
  }

  static Rc<DxvkBuffer> createVertexCaptureBuffer(DxvkDevice* pDevice, const VkDeviceSize size) {
    DxvkBufferCreateInfo info;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    info.access = VK_ACCESS_TRANSFER_READ_BIT;
    info.stages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    info.size = size;
    // Own category rather than AppBuffer: these are Remix-side allocations whose lifetime the
    // runtime controls (per-draw, pooled, or held by the UE3 static capture cache), so lumping
    // them in with the game's own buffers hides them from memory profiling.
    return pDevice->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXVertexCapture, "Vertex Capture Buffer");
  }

  DxvkBufferSlice D3D9Rtx::allocVertexCaptureBuffer(const VkDeviceSize size, const bool allowPooledBuffer) {
    DxvkDevice* pDevice = m_parent->GetDXVKDevice().ptr();

    if (!m_frameOptions.poolVertexCaptureBuffers) {
      if (!m_captureBufferPool.empty()) {
        m_captureBufferPool.clear();
      }
      return DxvkBufferSlice(createVertexCaptureBuffer(pDevice, size));
    }

    // A capture the static vertex-capture cache retains lives as long as its entry: pooling gains
    // nothing and the power-of-two rounding would make the cache's byte budget undercount VRAM.
    if (!allowPooledBuffer) {
      return DxvkBufferSlice(createVertexCaptureBuffer(pDevice, size));
    }

    // Reuse only when the pool holds the last reference (no draw-call state, BlasEntry input, CS
    // chunk, command list or context binding refers to it) and no command list still uses it.
    VkDeviceSize sizeClass = kMinCaptureBufferClass;
    while (sizeClass < size) {
      sizeClass <<= 1;
    }
    CaptureBufferBucket& bucket = m_captureBufferPool[sizeClass];
    const size_t count = bucket.buffers.size();
    const size_t probes = std::min<size_t>(count, kCaptureBufferProbes);
    for (size_t i = 0; i < probes; ++i) {
      const size_t idx = (bucket.cursor + i) % count;
      PooledCaptureBuffer& entry = bucket.buffers[idx];
      entry.buffer->incRef();
      const uint32_t refs = entry.buffer->decRef();
      if (refs == 1 && !entry.buffer->isInUse()) { // isInUse(Read) checks readers and writers
        bucket.cursor = idx + 1;
        entry.lastUsedFrame = m_ue3FrameCounter;
        return DxvkBufferSlice(entry.buffer, 0, size);
      }
    }

    Rc<DxvkBuffer> buffer = createVertexCaptureBuffer(pDevice, sizeClass);
    if (count < kMaxPooledCaptureBuffersPerClass) {
      bucket.buffers.push_back({ buffer, m_ue3FrameCounter });
    }
    return DxvkBufferSlice(buffer, 0, size);
  }

  void D3D9Rtx::trimVertexCaptureBufferPool() {
    // Drop the pool's reference to buffers not handed out for a while; one still referenced
    // elsewhere simply stops being pooled.
    if (m_captureBufferPool.empty() || (m_ue3FrameCounter % 64) != 0) {
      return;
    }
    for (auto it = m_captureBufferPool.begin(); it != m_captureBufferPool.end();) {
      std::vector<PooledCaptureBuffer>& buffers = it->second.buffers;
      buffers.erase(std::remove_if(buffers.begin(), buffers.end(), [this](const PooledCaptureBuffer& e) {
        return m_ue3FrameCounter - e.lastUsedFrame > kCaptureBufferMaxIdleFrames;
      }), buffers.end());
      it->second.cursor = 0;
      it = buffers.empty() ? m_captureBufferPool.erase(it) : std::next(it);
    }
  }

  bool D3D9Rtx::prepareVertexCapture(const int vertexIndexOffset, const Ue3CapturePositionSource positionSource, const bool allowPooledBuffer) {
    ScopedCpuProfileZone();

    static_assert(sizeof CapturedVertex == 48, "The injected shader code is expecting this exact structure size to work correctly, see emitVertexCaptureWrite in dxso_compiler.cpp");

    // vertex capture requires invertible transforms (projection and affine inverses)
    // iif these are singular, inverse() will otherwise trip math validation /produce invalid data
    {
      constexpr double kDetEps = 1e-24;
      auto detOk = [&](const Matrix4& m) {
        const double det = determinant(m);
        return std::isfinite(det) && std::abs(det) > kDetEps;
      };

      const auto& t = m_activeDrawCallState.transformData;
      // objectToWorld is inverted for every source; the projection and view inverses only
      // matter to the clip-space path, so a singular projection cannot veto an exact capture.
      const bool transformsOk = detOk(t.objectToWorld)
        && (positionSource != Ue3CapturePositionSource::ClipReconstruction
            || (detOk(t.viewToProjection) && detOk(t.worldToView)));
      if (!transformsOk) {
        ONCE(Logger::warn("[RTX-Compatibility] Skipping vertex capture due to non-invertible transform(s)."));
        return false;
      }
    }

    // Get common shaders to query what data we can capture
    const D3D9CommonShader* vertexShader = d3d9State().vertexShader.ptr() != nullptr ? d3d9State().vertexShader->GetCommonShader() : nullptr;

    RasterGeometry& geoData = m_activeDrawCallState.geometryData;

    // Known stride for vertex capture buffers
    const uint32_t stride = sizeof(CapturedVertex);
    const size_t vertexCaptureDataSize = align(geoData.vertexCount * stride, CACHE_LINE_SIZE);

    DxvkBufferSlice slice = allocVertexCaptureBuffer(vertexCaptureDataSize, allowPooledBuffer);

    geoData.positionBuffer = RasterBuffer(slice, 0, stride, VK_FORMAT_R32G32B32A32_SFLOAT);
    assert(geoData.positionBuffer.offset() % 4 == 0);

    const VertexCapturePlan plan = planUe3VertexCapture(vertexShader, positionSource);

    if (plan.captureTexcoords) {
      const uint32_t texcoordOffset = offsetof(CapturedVertex, texcoord0);
      geoData.texcoordBuffer = RasterBuffer(slice, texcoordOffset, stride, VK_FORMAT_R32G32_SFLOAT);
      assert(geoData.texcoordBuffer.offset() % 4 == 0);
    }

    // Without a captured normal the IA normals stay: they are in the captured positions' space.
    if (plan.captureNormals) {
      const uint32_t normalOffset = offsetof(CapturedVertex, normal0);
      geoData.normalBuffer = RasterBuffer(slice, normalOffset, stride, VK_FORMAT_R32G32B32_SFLOAT);
      assert(geoData.normalBuffer.offset() % 4 == 0);
    }

    if (plan.captureColor) {
      const uint32_t colorOffset = offsetof(CapturedVertex, color0);
      geoData.color0Buffer = RasterBuffer(slice, colorOffset, stride, VK_FORMAT_B8G8R8A8_UNORM);
      assert(geoData.color0Buffer.offset() % 4 == 0);
    }

    auto constants = m_vsVertexCaptureData->allocSlice();

    // Upload
    auto& data = *reinterpret_cast<D3D9RtxVertexCaptureData*>(constants.mapPtr);
    // Only the clip-space path consumes these, and only it required them to be invertible
    // above, so inverting them unconditionally would trip math validation on a draw whose
    // projection is singular but whose position comes from an exact source.
    if (positionSource == Ue3CapturePositionSource::ClipReconstruction) {
      data.invProj = inverse(m_activeDrawCallState.transformData.viewToProjection);
      data.viewToWorld = inverseAffine(m_activeDrawCallState.transformData.worldToView);
    } else {
      data.invProj = Matrix4();
      data.viewToWorld = Matrix4();
    }
    data.worldToObject = inverseAffine(m_activeDrawCallState.transformData.objectToWorld);
    data.normalTransform = m_activeDrawCallState.transformData.objectToWorld;
    // note - BaseVertexIndex can be negative, so we store the raw value as uint32 so the shader's unsigned
    // subtraction (uVertexId - baseVertex) behaves correctly for two's-complement values
    data.baseVertex = (uint32_t)vertexIndexOffset;
    data.flags = plan.flags;
    data.boneMatricesBaseReg = plan.boneMatricesBaseReg;
    data.boneCount = plan.boneCount;
    data.texcoordOutputRegister = plan.texcoordOutputRegister;
    data.texcoordCompU = m_texcoordCompU & 0x3u;
    data.texcoordCompV = m_texcoordCompV & 0x3u;
    data.colorOutputRegister = plan.colorOutputRegister;

    m_parent->EmitCs([cVertexDataSlice = slice,
                      cConstantBuffer = m_vsVertexCaptureData,
                      cConstants = constants](DxvkContext* ctx) {
      // Bind the new constants to buffer
      ctx->invalidateBuffer(cConstantBuffer, cConstants);

      // Invalidate rest of the members
      // customWorldToProjection is not invalidated as its use is controlled by D3D9SpecConstantId::CustomVertexTransformEnabled being enabled
      ctx->bindResourceBuffer(getVertexCaptureBufferSlot(), cVertexDataSlice);
    });

    return true;
  }

  void D3D9Rtx::processVertices(const VertexContext vertexContext[caps::MaxStreams], int vertexIndexOffset, RasterGeometry& geoData) {
    // One zone per draw, not per vertex element: with UE3's ~9 elements per declaration a
    // per-element zone emits tens of thousands of Tracy events per frame whose begin/end
    // overhead lands in the enclosing internalPrepareDraw zone and distorts captures.
    ScopedCpuProfileZoneN("Process Vertices");
    DxvkBufferSlice streamCopies[caps::MaxStreams] {};

    // FParticleInstancedMeshVertexFactory hands the mesh's TangentZ to the BINORMAL semantic
    // rather than NORMAL (see classifyUe3VertexFactory), so this is the one factory whose surface
    // normal has to be read from there.
    const bool normalFromBinormal =
      m_currentUe3VertexFactory == Ue3VertexFactoryType::ParticleInstancedMesh;

    // Process vertex buffers from CPU
    for (const auto& element : d3d9State().vertexDecl->GetElements()) {
      // Get vertex context
      const VertexContext& ctx = vertexContext[element.Stream];

      if (ctx.mappedSlice.handle == VK_NULL_HANDLE)
        continue;

      // An instance-data stream advances once per instance, so indexing it by vertex reads an
      // unrelated instance's record (and walks off the end of a short instance buffer).
      if (element.Stream < caps::MaxStreams &&
          (m_currentUe3Instancing.instanceDataStreamMask & (1u << element.Stream)) != 0)
        continue;

      const int32_t vertexOffset = ctx.offset + ctx.stride * vertexIndexOffset;
      const uint32_t numVertexBytes = ctx.stride * geoData.vertexCount;

      // Validating index data here, vertexCount and vertexIndexOffset accounts for the min/max indices
      if (m_frameOptions.validateCPUIndexData) {
        if (ctx.mappedSlice.length < vertexOffset + numVertexBytes) {
          throw DxvkError("Invalid draw call");
        }
      }

      // TODO: Simplify this by refactoring RasterGeometry to contain an array of RasterBuffer's
      RasterBuffer* targetBuffer = nullptr;
      switch (element.Usage) {
      case D3DDECLUSAGE_POSITIONT:
      case D3DDECLUSAGE_POSITION:
        if (element.UsageIndex == 0)
          targetBuffer = &geoData.positionBuffer;
        break;
      case D3DDECLUSAGE_BLENDWEIGHT:
        if (element.UsageIndex == 0)
          targetBuffer = &geoData.blendWeightBuffer;
        break;
      case D3DDECLUSAGE_BLENDINDICES:
        if (element.UsageIndex == 0)
          targetBuffer = &geoData.blendIndicesBuffer;
        break;
      case D3DDECLUSAGE_NORMAL:
        if (element.UsageIndex == 0 && !normalFromBinormal)
          targetBuffer = &geoData.normalBuffer;
        break;
      case D3DDECLUSAGE_BINORMAL:
        if (element.UsageIndex == 0 && normalFromBinormal)
          targetBuffer = &geoData.normalBuffer;
        break;
      case D3DDECLUSAGE_TEXCOORD:
        // A D3DCOLOR-typed TEXCOORD under UE3 is a vertex lightmap coefficient stream, never a
        // coordinate (see resolveIaTexcoordIndex).
        if (m_iaTexcoordIndex <= MAXD3DDECLUSAGEINDEX && element.UsageIndex == m_iaTexcoordIndex &&
            !(m_frameOptions.ue3EngineMode && element.Type == D3DDECLTYPE_D3DCOLOR))
          targetBuffer = &geoData.texcoordBuffer;
        break;
      case D3DDECLUSAGE_COLOR:
        if (element.UsageIndex == 0 &&
            !m_frameOptions.ignoreAllVertexColorBakedLighting && !m_frameOptions.ue3EngineMode &&
            !lookupHash(*m_frameOptions.ignoreBakedLightingTextures, m_activeDrawCallState.materialData.colorTextures[0].getImageHash())) {
          // only treat COLOR0 as a packed 8-bit UNORM color, UE3 can use COLOR semantics for non-color data which the rtx interleaver does not interpret as vertex color
          const VkFormat fmt = DecodeDecltype(D3DDECLTYPE(element.Type));
          if (fmt == VK_FORMAT_B8G8R8A8_UNORM || fmt == VK_FORMAT_R8G8B8A8_UNORM) {
            targetBuffer = &geoData.color0Buffer;
          }
        }
        break;
      }

      if (targetBuffer != nullptr) {
        assert(!targetBuffer->defined());

        // Only do once for each stream
        if (!streamCopies[element.Stream].defined()) {
          // Deep clonning a buffer object is not cheap (320 bytes to copy and other work). Set a min-size threshold.
          const uint32_t kMinSizeToClone = 512;

          // Check if buffer is actualy a d3d9 orphan
          const bool isOrphan = !(ctx.buffer.getSliceHandle() == ctx.mappedSlice);
          const bool canUseBuffer = ctx.canUseBuffer && m_forceGeometryCopy == false;

          if (canUseBuffer && !isOrphan) {
            // Use the buffer directly if it is not an orphan
            if (ctx.pVBO != nullptr && ctx.pVBO->NeedsUpload())
              m_parent->FlushBuffer(ctx.pVBO);

            streamCopies[element.Stream] = ctx.buffer.subSlice(vertexOffset, numVertexBytes);
          } else if (canUseBuffer && numVertexBytes > kMinSizeToClone) {
            // Create a clone for the orphaned physical slice
            auto clone = ctx.buffer.buffer()->clone();
            clone->rename(ctx.mappedSlice);
            streamCopies[element.Stream] = DxvkBufferSlice(clone, ctx.buffer.offset() + vertexOffset, numVertexBytes);
          } else {
            streamCopies[element.Stream] = m_rtStagingData.alloc(CACHE_LINE_SIZE, numVertexBytes);

            // Acquire prevents the staging allocator from re-using this memory
            streamCopies[element.Stream].buffer()->acquire(DxvkAccess::Read);

            memcpy(streamCopies[element.Stream].mapPtr(0), (uint8_t*) ctx.mappedSlice.mapPtr + vertexOffset, numVertexBytes);
          }
        }

        VkFormat fmt = DecodeDecltype(D3DDECLTYPE(element.Type));
        uint32_t elementOffset = element.Offset;

        // note: m_texcoordCompU/V select VS interpolant components (capture path only);
        // a proven IA texcoord element is always a plain (u, v) pair and is read as-is

        // UE3 packed normals use D3DDECLTYPE_UBYTE4 (not normalised) so for remix purposes we want a decoded
        // [-1, 1] normal and the interleaver supports VK_FORMAT_R8G8B8A8_UNORM for this
        if (targetBuffer == &geoData.normalBuffer && fmt == VK_FORMAT_R8G8B8A8_USCALED) {
          fmt = VK_FORMAT_R8G8B8A8_UNORM;
        }
        *targetBuffer = RasterBuffer(streamCopies[element.Stream], elementOffset, ctx.stride, fmt);
        assert(targetBuffer->offset() % 4 == 0);
      }
    }
  }

  bool D3D9Rtx::processRenderState(const DrawContext& drawContext) {
    ScopedCpuProfileZone();
    DrawCallTransforms& transformData = m_activeDrawCallState.transformData;
    m_forceIaTexcoordForOutlier = false;

    // m_activeDrawCallState is reused across draws, so these fields need an explicit per-draw reset
    m_activeDrawCallState.allowMainCameraUpdate = true;
    m_activeDrawCallState.programmableVertexShaderBytecodeHash = 0;
    m_activeDrawCallState.ue3PassDescription = describeUe3PassType(m_currentUe3PassType);
    m_activeDrawCallState.isUe3ForegroundDpg = m_ue3ForegroundDpgActive;

    const bool isUe3Mode = m_frameOptions.ue3EngineMode;
    const bool effectiveUseWorldMatricesForShaders = m_frameOptions.useWorldMatricesForShaders && !isUe3Mode;

    // When games use vertex shaders, the object to world transforms can be unreliable, and so we can ignore them.
    const bool useObjectToWorldTransform = !m_parent->UseProgrammableVS() || (m_parent->UseProgrammableVS() && m_frameOptions.useVertexCapture && effectiveUseWorldMatricesForShaders);
    transformData.objectToWorld = useObjectToWorldTransform ? d3d9State().transforms[GetTransformIndex(D3DTS_WORLD)] : Matrix4();

    transformData.worldToView = d3d9State().transforms[GetTransformIndex(D3DTS_VIEW)];
    transformData.viewToProjection = d3d9State().transforms[GetTransformIndex(D3DTS_PROJECTION)];

    if (!applyUe3ShaderConstantTransforms(drawContext, transformData)) {
      return false;
    }

    transformData.objectToView = transformData.worldToView * transformData.objectToWorld;

    // Some games pass invalid matrices which D3D9 apparently doesnt care about.
    // since we'll be doing inversions and other matrix operations, we need to 
    // sanitize those or there be nans.
    transformData.sanitize();

    if (m_flags.test(D3D9RtxFlag::DirtyClipPlanes)) {
      m_flags.clr(D3D9RtxFlag::DirtyClipPlanes);

      // Find one truly enabled clip plane because we don't support more than one
      transformData.enableClipPlane = false;
      if (d3d9State().renderStates[D3DRS_CLIPPLANEENABLE] != 0) {
        for (int i = 0; i < caps::MaxClipPlanes; ++i) {
          // Check the enable bit
          if ((d3d9State().renderStates[D3DRS_CLIPPLANEENABLE] & (1 << i)) == 0)
            continue;

          // Make sure that the plane equation is not degenerate
          const Vector4 plane = Vector4(d3d9State().clipPlanes[i].coeff);
          if (lengthSqr(plane.xyz()) > 0.f) {
            if (transformData.enableClipPlane) {
              ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Using more than 1 user clip plane is not supported.")));
              break;
            }

            transformData.enableClipPlane = true;
            transformData.clipPlane = plane;
          }
        }
      }
    }

    if (m_flags.test(D3D9RtxFlag::DirtyLights)) {
      m_flags.clr(D3D9RtxFlag::DirtyLights);

      std::vector<D3DLIGHT9> activeLightsRT;
      uint32_t lightIdx = 0;
      for (auto idx : d3d9State().enabledLightIndices) {
        if (idx == UINT32_MAX)
          continue;
        activeLightsRT.push_back(d3d9State().lights[idx].value());
      }

      m_parent->EmitCs([activeLightsRT, lightIdx](DxvkContext* ctx) {
          static_cast<RtxContext*>(ctx)->addLights(activeLightsRT.data(), activeLightsRT.size());
        });
    }

    // Stencil state is important to Remix
    m_activeDrawCallState.stencilEnabled = d3d9State().renderStates[D3DRS_STENCILENABLE];

    if (isUe3SecondTwoSidedTranslucentPass(drawContext)) {
      return false;
    }

    // Process textures
    if (m_parent->UseProgrammablePS()) {
      return processTextures<false>();
    } else {
      return processTextures<true>();
    }
  }

  D3D9Rtx::DrawCallType D3D9Rtx::makeDrawCallType(const DrawContext& drawContext) {
    ScopedCpuProfileZone();
    // Track the drawcall index so we can use it in rtx_context
    m_activeDrawCallState.drawCallID = m_drawCallID++;
    m_activeDrawCallState.isDrawingToRaytracedRenderTarget = false;
    m_activeDrawCallState.isUsingRaytracedRenderTarget = false;

    if (m_drawCallID < (uint32_t)m_frameOptions.drawCallRange.x ||
        m_drawCallID > (uint32_t)m_frameOptions.drawCallRange.y) {
      return { RtxGeometryStatus::Ignored, false };
    }

    // Draws inside an occlusion query bracket are visibility-test geometry, never scene geometry
    // to ray trace; checked first so no skip path below can starve an active query of its draws.
    // With synthesized readbacks nothing consumes a measurement, so they are ignored entirely;
    // otherwise they rasterize so the query can count their samples.
    if (m_activeOcclusionQueries > 0) {
      if (ConservativeOcclusionQueriesEnabled()) {
        ONCE(Logger::info("[RTX-Compatibility-Info] Ignoring occlusion query test draw (conservative occlusion queries synthesize the result)."));
        return { RtxGeometryStatus::Ignored, false };
      }
      ONCE(Logger::info("[RTX-Compatibility-Info] Rasterizing occlusion query test draw without ray tracing."));
      return { RtxGeometryStatus::Rasterized, false };
    }

    // Raytraced Render Target Support
    // If the bound texture for this draw call is one that has been used as a render target then store its id
    if (m_frameOptions.raytracedRenderTargetEnable) {
      for (uint32_t i : bit::BitMask(m_parent->GetActiveRTTextures())) {
        D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[i]);
        if (!texture || texture->GetImage() == nullptr)
          continue;

        if (isTaggedRaytracedRenderTarget(texture->GetImage(), true)) {
          m_activeDrawCallState.isUsingRaytracedRenderTarget = true;
        }
      }
    }

    if (m_parent->UseProgrammableVS() && !m_frameOptions.useVertexCapture) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipping draw call with shader usage as vertex capture is not enabled."));
      return { RtxGeometryStatus::Ignored, false };
    }

    if (drawContext.PrimitiveCount == 0) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped invalid drawcall, primitive count was 0."));
      return { RtxGeometryStatus::Ignored, false };
    }

    // Only certain draw calls are worth raytracing
    if (!isPrimitiveSupported(drawContext.PrimitiveType)) {
      ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Trying to raytrace an unsupported primitive topology [", drawContext.PrimitiveType, "]. Ignoring.")));
      return { RtxGeometryStatus::Ignored, false };
    }

    if (!m_frameOptions.enableAlphaTest && m_parent->IsAlphaTestEnabled()) {
      ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Raytracing an alpha-tested draw call when alpha-tested objects disabled in RT. Ignoring.")));
      return { RtxGeometryStatus::Ignored, false };
    }

    if (!m_frameOptions.enableAlphaBlend && d3d9State().renderStates[D3DRS_ALPHABLENDENABLE]) {
      ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Raytracing an alpha-blended draw call when alpha-blended objects disabled in RT. Ignoring.")));
      return { RtxGeometryStatus::Ignored, false };
    }

    DeferredUiTagQuery deferredUiTag(*this);
    if (isUe3DepthTestDisabledTranslucency(deferredUiTag)) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Ignored UE3 depth-test-disabled translucent draw."));
      return { RtxGeometryStatus::Ignored, false };
    }

    if (d3d9State().renderTargets[kRenderTargetIndex] == nullptr) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped drawcall, as no color render target bound."));
      return { RtxGeometryStatus::Ignored, false };
    }

    {
      D3D9CommonTexture* rtTexture = GetCommonTexture(d3d9State().renderTargets[kRenderTargetIndex]->GetBaseTexture());
      if (rtTexture != nullptr && rtTexture->GetImage() == nullptr) {
        ONCE(Logger::info("[RTX-Compatibility-Info] Skipped drawcall, render target has no GPU image (possibly a depth-only pass)."));
        return { RtxGeometryStatus::Ignored, false };
      }
    }

    constexpr DWORD rgbWriteMask = D3DCOLORWRITEENABLE_RED | D3DCOLORWRITEENABLE_GREEN | D3DCOLORWRITEENABLE_BLUE;
    if ((d3d9State().renderStates[ColorWriteIndex(kRenderTargetIndex)] & rgbWriteMask) != rgbWriteMask) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped drawcall, colour write disabled."));
      return { RtxGeometryStatus::Ignored, false };
    }

    // Ensure present parameters for the swapchain have been cached
    // Note: This assumes that ResetSwapChain has been called at some point before this call, typically done after creating a swapchain.
    assert(m_activePresentParams.has_value());

    if (const std::optional<DrawCallType> ue3Pass = classifyUe3DrawPass(drawContext, deferredUiTag)) {
      return *ue3Pass;
    }

    if (const std::optional<DrawCallType> deferredUi = decideDeferredUiDraw(drawContext, deferredUiTag)) {
      return *deferredUi;
    }

    // Attempt to detect shadow mask draws and ignore them
    // Conditions: non-textured flood-fill draws into a small quad render target
    if (((d3d9State().textureStages[0][D3DTSS_COLOROP] == D3DTOP_SELECTARG1 && d3d9State().textureStages[0][D3DTSS_COLORARG1] != D3DTA_TEXTURE) ||
         (d3d9State().textureStages[0][D3DTSS_COLOROP] == D3DTOP_SELECTARG2 && d3d9State().textureStages[0][D3DTSS_COLORARG2] != D3DTA_TEXTURE))) {
      const auto& rtExt = d3d9State().renderTargets[kRenderTargetIndex]->GetSurfaceExtent();
      // If rt is a quad at least 4 times smaller than backbuffer and the format is invalid format, then it is likely a shadow mask
      if (rtExt.width == rtExt.height && rtExt.width < m_activePresentParams->BackBufferWidth / 4 &&
          Resources::getFormatCompatibilityCategory(d3d9State().renderTargets[kRenderTargetIndex]->GetImageView(false)->imageInfo().format) == RtxTextureFormatCompatibilityCategory::InvalidFormatCompatibilityCategory) {
        ONCE(Logger::info("[RTX-Compatibility-Info] Skipped shadow mask drawcall."));
        return { RtxGeometryStatus::Ignored, false };
      }
    }

    if (isUe3ShadowDepthPass()) {
      return { RtxGeometryStatus::Ignored, false };
    }

    // Raytraced Render Target
    // If this isn't the primary render target but we have used this render target before then 
    // store the current camera matrices in case this render target is intended to be used as 
    // a texture for some geometry later
    if (m_frameOptions.raytracedRenderTargetEnable) {
      D3D9CommonTexture* texture = GetCommonTexture(d3d9State().renderTargets[kRenderTargetIndex]->GetBaseTexture());
      if (texture) {
        const Rc<DxvkImage> image = texture->GetImage();
        if (image != nullptr && isTaggedRaytracedRenderTarget(image, true)) {
          m_activeDrawCallState.isDrawingToRaytracedRenderTarget = true;
          return { RtxGeometryStatus::RayTraced, false };
        }
      }
    }

    if (!s_isDxvkResolutionEnvVarSet) {
      // NOTE: This can fail when setting DXVK_RESOLUTION_WIDTH or HEIGHT
      const bool isPrimary = isRenderTargetPrimary(*m_activePresentParams, d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture()->Desc());

      if (!isPrimary) {
        logNonPrimaryRenderTargetOnce();

        ONCE(Logger::info("[RTX-Compatibility-Info] Found a draw call to a non-primary, non-raytraced render target. Falling back to rasterization"));
        return { RtxGeometryStatus::Rasterized, false };
      }
    }

    if (const std::optional<DrawCallType> rtSampling = classifyRenderTargetSamplingDraw(drawContext)) {
      return *rtSampling;
    }

    // Detect stencil shadow draws and ignore them
    // Conditions: passingthrough stencil is enabled with increment or decrement z-fail action
    if (d3d9State().renderStates[D3DRS_STENCILENABLE] == TRUE &&
        d3d9State().renderStates[D3DRS_STENCILFUNC] == D3DCMP_ALWAYS &&
        (d3d9State().renderStates[D3DRS_STENCILZFAIL] == D3DSTENCILOP_DECR || d3d9State().renderStates[D3DRS_STENCILZFAIL] == D3DSTENCILOP_INCR ||
         d3d9State().renderStates[D3DRS_STENCILZFAIL] == D3DSTENCILOP_DECRSAT || d3d9State().renderStates[D3DRS_STENCILZFAIL] == D3DSTENCILOP_INCRSAT) &&
        d3d9State().renderStates[D3DRS_ZWRITEENABLE] == FALSE) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped stencil shadow drawcall."));
      return { RtxGeometryStatus::Ignored, false };
    }

    // Check UI only to the primary render target
    if (isRenderingUI()) {
      return {
        RtxGeometryStatus::Rasterized,
        true, // UI rendering detected => trigger RTX injection
      };
    }

    // TODO(REMIX-760): Support reverse engineering pre-transformed vertices
    if (d3d9State().vertexDecl != nullptr) {
      if (d3d9State().vertexDecl->TestFlag(D3D9VertexDeclFlag::HasPositionT)) {
        if (m_frameOptions.preTransformedVerticesIsUI) {
          return { RtxGeometryStatus::Rasterized, true };
        } else {
          ONCE(Logger::info("[RTX-Compatibility-Info] Skipped drawcall, using pre-transformed vertices which isn't currently supported."));
          return { RtxGeometryStatus::Rasterized, false };
        }
      }
    }

    logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::RayTraced, "default raytraced geometry");
    return { RtxGeometryStatus::RayTraced, false };
  }

  // Draws sampling a render target: picks a raytraced render target from fullscreen composites
  // and optionally rasterizes those composites.
  std::optional<D3D9Rtx::DrawCallType> D3D9Rtx::classifyRenderTargetSamplingDraw(const DrawContext& drawContext) {
    if (const uint32_t rtSamplerMask = m_parent->GetActiveRTTextures()) {
      const bool depthEnabled  = d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_TRUE;
      const bool zWriteEnabled = d3d9State().renderStates[D3DRS_ZWRITEENABLE];
      const bool likelyFullscreenComposite =
        !depthEnabled &&
        !zWriteEnabled &&
        drawContext.PrimitiveCount <= 4;

      if (m_frameOptions.autoRaytracedRenderTargetFromFullscreenComposite && likelyFullscreenComposite) {
        const uint32_t bbW = m_activePresentParams->BackBufferWidth;
        const uint32_t bbH = m_activePresentParams->BackBufferHeight;

        auto aspectRatioMatches = [&](uint32_t w, uint32_t h) {
          const double a = double(w) * double(bbH);
          const double b = double(h) * double(bbW);
          const double denom = std::max(a, b);
          return denom > 0.0 && (std::abs(a - b) / denom) < 0.01;
        };

        XXH64_hash_t bestHash = 0;
        XXH64_hash_t bestAgnosticHash = 0;
        uint64_t bestArea = 0;

        for (uint32_t i : bit::BitMask(rtSamplerMask)) {
          D3D9CommonTexture* tex = GetCommonTexture(d3d9State().textures[i]);
          if (!tex || tex->GetImage() == nullptr)
            continue;

          if (isTaggedRaytracedRenderTarget(tex->GetImage(), true))
            continue;

          const auto* desc = tex->Desc();
          if (!desc)
            continue;
          if (desc->Width == bbW && desc->Height == bbH)
            continue;
          if (!aspectRatioMatches(desc->Width, desc->Height))
            continue;

          const uint64_t area = uint64_t(desc->Width) * uint64_t(desc->Height);
          if (area > bestArea) {
            bestArea = area;
            // Absolute hash: this set is session-local, and the aspect-normalized hash would
            // alias the backbuffer-sized target skipped above.
            bestHash = tex->GetImage()->getDescriptorHash();
            bestAgnosticHash = tex->GetImage()->getResolutionAgnosticDescriptorHash();
          }
        }

        // Already-tagged targets were skipped as candidates above, so anything reaching here is new.
        if (bestHash != kEmptyHash && m_autoRaytracedRenderTargetDescHashes.insert(bestHash).second) {
          Logger::info(str::format(
            "[RTX-Compatibility] Auto-selected Raytraced Render Target from fullscreen composite: texDescHash=0x",
            std::hex, bestHash,
            " resolutionAgnosticDescHash=0x", bestAgnosticHash, std::dec,
            " (tag either in rtx.raytracedRenderTargetTextures to pin it)."));
        }
      }

      if (Logger::logLevel() <= LogLevel::Debug) {
        for (uint32_t i : bit::BitMask(rtSamplerMask)) {
          D3D9CommonTexture* tex = GetCommonTexture(d3d9State().textures[i]);
          if (!tex || tex->GetImage() == nullptr)
            continue;

          const XXH64_hash_t texDescHash = tex->GetImage()->getDescriptorHash();
          if (s_loggedSampledRtDescHashes.insert(texDescHash).second) {
            const auto* desc = tex->Desc();
            Logger::debug(str::format(
              "[RTX-Compatibility] Sampled render-target texture: ",
              desc->Width, "x", desc->Height,
              ", texDescHash=0x", std::hex, texDescHash,
              " resolutionAgnosticDescHash=0x", tex->GetImage()->getResolutionAgnosticDescriptorHash(), std::dec,
              " (sampler ", i, ")."));
          }
        }
      }

      // Optional: do not raytrace likely fullscreen composite passes to primary.
      if (m_frameOptions.rasterizeFullscreenCompositeToPrimary && likelyFullscreenComposite) {
        logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Rasterized, "fullscreen RT composite");
        ONCE(Logger::info("[RTX-Compatibility] Rasterizing likely fullscreen composite pass to primary RT (post-process)."));
        return DrawCallType { RtxGeometryStatus::Rasterized, false };
      }
    }

    return std::nullopt;
  }

  const D3D9Rtx::BoundTextureSnapshot& D3D9Rtx::ensureBoundTextureSnapshot() const {
    if (m_boundTextureSnapshotValid) {
      return m_boundTextureSnapshot;
    }

    m_boundTextureSnapshot.mask = 0;

    const uint32_t boundMask = m_parent->m_activeTextures & ((1u << SamplerCount) - 1u);
    for (const uint32_t idx : bit::BitMask(boundMask)) {
      if (d3d9State().textures[idx] == nullptr) {
        continue;
      }

      D3D9CommonTexture* const texture = GetCommonTexture(d3d9State().textures[idx]);
      if (texture == nullptr) {
        continue;
      }

      BoundTextureSnapshotEntry& entry = m_boundTextureSnapshot.entries[idx];
      entry.texture = texture;
      entry.hasSampleView = texture->GetSampleView(false) != nullptr;
      DxvkImage* const image = texture->GetImage().ptr();
      entry.hasImage = image != nullptr;
      entry.imageHash = entry.hasImage ? image->getHash() : kEmptyHash;
      entry.isRenderTarget = texture->IsRenderTarget();
      entry.rtDescriptorHash = (entry.isRenderTarget && entry.hasImage) ? image->getDescriptorHash() : 0;
      entry.rtResolutionAgnosticDescriptorHash =
        (entry.isRenderTarget && entry.hasImage) ? image->getResolutionAgnosticDescriptorHash() : 0;
      // Non-RT images carry no descriptor hash on the DxvkImage; compute one only when
      // the identity exclusion option or the replacement diagnostics actually consume it.
      const bool wantDescriptorHashes =
        (m_frameOptions.ue3MicIdentityExcludedTextureDescHashes != nullptr &&
         !m_frameOptions.ue3MicIdentityExcludedTextureDescHashes->empty()) ||
        m_frameOptions.logReplacementResolution ||
        m_frameOptions.ue3LogMaterialInstanceHash;
      if (entry.rtDescriptorHash != 0) {
        entry.descriptorHash = entry.rtDescriptorHash;
      } else if (wantDescriptorHashes && entry.hasImage && texture->Desc() != nullptr) {
        entry.descriptorHash = texture->Desc()->CalculateHash();
      } else {
        entry.descriptorHash = 0;
      }

      m_boundTextureSnapshot.mask |= (1u << idx);
    }

    m_boundTextureSnapshotValid = true;
    return m_boundTextureSnapshot;
  }

  bool D3D9Rtx::checkBoundTextureCategory(const fast_unordered_set& textureCategory) const {
    const uint32_t usedSamplerMask = m_parent->m_psShaderMasks.samplerMask | m_parent->m_vsShaderMasks.samplerMask;
    const BoundTextureSnapshot& boundTextures = ensureBoundTextureSnapshot();
    const uint32_t usedTextureMask = boundTextures.mask & usedSamplerMask;
    for (const uint32_t idx : bit::BitMask(usedTextureMask)) {
      const XXH64_hash_t texHash = boundTextures.entries[idx].imageHash;
      if (textureCategory.find(texHash) != textureCategory.end()) {
        return true;
      }
    }

    return false;
  }

  bool D3D9Rtx::isRenderingUI() {
    if (!m_parent->UseProgrammableVS() && m_frameOptions.orthographicIsUI) {
      // Here we assume drawcalls with an orthographic projection are UI calls (as this pattern is common, and we can't raytrace these objects).
      const bool isOrthographic = (d3d9State().transforms[GetTransformIndex(D3DTS_PROJECTION)][3][3] == 1.0f);
      const bool zWriteEnabled = d3d9State().renderStates[D3DRS_ZWRITEENABLE];
      if (isOrthographic && !zWriteEnabled) {
        return true;
      }
    }

    // Check if UI texture bound
    return checkBoundTextureCategory(*m_frameOptions.uiTextures);
  }

  bool D3D9Rtx::isTaggedRaytracedRenderTarget(const Rc<DxvkImage>& image, bool includeAutoDetected) const {
    if (image == nullptr) {
      return false;
    }

    const XXH64_hash_t descriptorHash = image->getDescriptorHash();

    if (matchAuthoredRenderTargetTag(*m_frameOptions.raytracedRenderTargetTextures,
                                     descriptorHash,
                                     image->getResolutionAgnosticDescriptorHash()) != kEmptyHash) {
      return true;
    }

    // Auto-detection is session-local and keys on the absolute hash: the aspect-normalized hash
    // aliases every same-format target of the same aspect, including the backbuffer-sized target
    // the detector deliberately skips. The size check catches a resolution change turning a
    // previously detected size into the backbuffer's.
    return includeAutoDetected &&
           descriptorHash != kEmptyHash &&
           lookupHash(m_autoRaytracedRenderTargetDescHashes, descriptorHash) &&
           !isBackBufferSizedImage(image);
  }

  XXH64_hash_t D3D9Rtx::matchAuthoredRenderTargetTag(const fast_unordered_set& tags,
                                                     const XXH64_hash_t descriptorHash,
                                                     const XXH64_hash_t resolutionAgnosticDescriptorHash) {
    if (tags.empty()) {
      return kEmptyHash;
    }
    if (resolutionAgnosticDescriptorHash != kEmptyHash &&
        lookupHash(tags, resolutionAgnosticDescriptorHash)) {
      return resolutionAgnosticDescriptorHash;
    }
    if (descriptorHash != kEmptyHash && lookupHash(tags, descriptorHash)) {
      return descriptorHash;
    }
    return kEmptyHash;
  }

  bool D3D9Rtx::isBackBufferSizedImage(const Rc<DxvkImage>& image) const {
    if (image == nullptr || !m_activePresentParams.has_value()) {
      return false;
    }
    const VkExtent3D& extent = image->info().extent;
    return extent.width == m_activePresentParams->BackBufferWidth &&
           extent.height == m_activePresentParams->BackBufferHeight;
  }

  Rc<DxvkImage> D3D9Rtx::getCurrentRenderTargetImage() const {
    if (d3d9State().renderTargets[kRenderTargetIndex] == nullptr) {
      return nullptr;
    }

    D3D9CommonTexture* texInfo = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
    if (texInfo == nullptr) {
      return nullptr;
    }

    return texInfo->GetImage();
  }

  D3D9Format D3D9Rtx::getCurrentRenderTargetFormat() const {
    if (d3d9State().renderTargets[kRenderTargetIndex] == nullptr) {
      return D3D9Format::Unknown;
    }

    const D3D9CommonTexture* texInfo = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
    return texInfo != nullptr ? texInfo->Desc()->Format : D3D9Format::Unknown;
  }

  PrepareDrawFlags D3D9Rtx::internalPrepareDraw(const IndexContext& indexContext, const VertexContext vertexContext[caps::MaxStreams], const DrawContext& drawContext) {
    ScopedCpuProfileZone();

    // Texture bindings cannot change within a draw; rebuild the shared snapshot lazily on
    // first use per draw (UI/deferred-UI tag checks, MIC texture-set hash, diffuse key).
    m_boundTextureSnapshotValid = false;

    // Per-phase CPU attribution of this function (pass timer only)
    RtxGpuPassTimer::CpuPhaseTimer phaseTimer(RtxGpuPassTimer::isEnabled() ? &m_parent->GetDXVKDevice()->getCommon()->metaGpuPassTimer() : nullptr);

    auto finishPrepare = [&](PrepareDrawFlags flags) {
      // NV-DXVK start: draw disposition statistics (with the pass timer enabled) - how many draws
      // still execute on the GPU as rasterised draws, and how many primitives they carry.
      if (RtxGpuPassTimer::isEnabled()) {
        DrawDispositionStats& s = m_drawDispositionStats;
        ++s.draws;
        const bool rayTraced = (flags & PrepareDrawFlag::CommitToRayTracing) != 0;
        const bool original = (flags & PrepareDrawFlag::OriginalDrawCall) != 0;
        if (rayTraced) {
          ++s.rayTraced;
        }
        if (original) {
          ++s.rasterized;
          s.rasterizedPrims += drawContext.PrimitiveCount;
          if (rayTraced) {
            ++s.rasterizedForCapture;
            s.rasterizedForCapturePrims += drawContext.PrimitiveCount;
          } else if (m_rtxInjectTriggered) {
            ++s.rasterizedPostInjection;
          }
        }
        if (!rayTraced && !original) {
          ++s.ignored;
        }
      }
      // NV-DXVK end
      return flags;
    };

    // RTX was injected => treat everything else as rasterized,
    // unless this draw targets a raytraced render target (e.g. render-to-texture
    // in games that draw UI before 3D content).
    if (m_rtxInjectTriggered) {
      bool isRaytracedRenderTarget = false;
      if (m_frameOptions.raytracedRenderTargetEnable &&
          d3d9State().renderTargets[kRenderTargetIndex] != nullptr) {
        D3D9CommonTexture* texture = GetCommonTexture(d3d9State().renderTargets[kRenderTargetIndex]->GetBaseTexture());
        if (texture) {
          const Rc<DxvkImage> image = texture->GetImage();
          if (image != nullptr) {
            isRaytracedRenderTarget = isTaggedRaytracedRenderTarget(image, true);
          }
        }
      }
      if (!isRaytracedRenderTarget) {
        // Occlusion query test draws can land post-injection (games without a depth prepass
        // issue queries after the base pass); same conservative handling as pre-injection.
        if (ShouldApplyConservativeOcclusionQueryState()) {
          return finishPrepare(PrepareDrawFlag::Ignore);
        }

        return finishPrepare(m_frameOptions.skipDrawCallsPostRTXInjection
               ? PrepareDrawFlag::Ignore
               : PrepareDrawFlag::PreserveDrawCallAndItsState);
      }
    }

    // Classified before makeDrawCallType, which filters passes by vertex factory and instancing.
    classifyUe3DrawVertexFactory();
    resolveUe3DrawInstancing();

    const auto [status, triggerRtxInjection, deferUntilInjection] = makeDrawCallType(drawContext);
    phaseTimer.lap(RtxGpuPassTimer::CpuCounter::AppPrepClassify);

    // When raytracing is enabled we want to completely remove the ignored drawcalls from further processing as early as possible
    const PrepareDrawFlags prepareFlagsForIgnoredDraws = m_frameOptions.enableRaytracing
                                                         ? PrepareDrawFlag::Ignore
                                                         : PrepareDrawFlag::PreserveDrawCallAndItsState;

    if (status == RtxGeometryStatus::Ignored) {
      return finishPrepare(prepareFlagsForIgnoredDraws);
    }

    // Deferred UI overlay: snapshot the draw for post-injection replay and suppress it here.
    // Executing it now would rasterize into a pre-injection target that the ray-traced blit
    // overwrites; letting it trigger injection would end the ray-traced scene mid-frame.
    if (deferUntilInjection) {
      if (m_frameOptions.deferredUiReplay) {
        captureDeferredUiDraw(indexContext, vertexContext, drawContext);
      }
      return finishPrepare(prepareFlagsForIgnoredDraws);
    }

    if (triggerRtxInjection) {
      // Bind all resources required for this drawcall to context first (i.e. render targets)
      m_parent->PrepareDraw(drawContext.PrimitiveType);

      m_rtxInjectTriggered = true;

      // Deferred overlays replay now, before this triggering UI draw executes: the required
      // order is ray-traced image, then deferred overlays, then the game's genuine UI on top.
      // Replaying any later would put overlays above draws tagged via rtx.uiTextures.
      injectRtxWithOverlays(getCurrentRenderTargetImage(), nullptr);

      return finishPrepare(PrepareDrawFlag::PreserveDrawCallAndItsState);
    }

    if (status == RtxGeometryStatus::Rasterized) {
      return finishPrepare(PrepareDrawFlag::PreserveDrawCallAndItsState);
    }

    m_forceGeometryCopy = m_frameOptions.useBuffersDirectly == false;
    m_forceGeometryCopy |= m_parent->GetOptions()->allowDiscard == false;

    // The packet we'll send to RtxContext with information about geometry
    RasterGeometry& geoData = m_activeDrawCallState.geometryData;
    geoData = {};
    geoData.cullMode = DecodeCullMode(D3DCULL(d3d9State().renderStates[D3DRS_CULLMODE]));
    geoData.frontFace = VK_FRONT_FACE_CLOCKWISE;
    geoData.topology = DecodeInputAssemblyState(drawContext.PrimitiveType).primitiveTopology;

    // This can be negative!!
    int vertexIndexOffset = drawContext.BaseVertexIndex;

    // Process index buffer
    uint32_t minIndex = 0, maxIndex = 0;
    if (indexContext.indexType != VK_INDEX_TYPE_NONE_KHR) {
      geoData.indexCount = GetVertexCount(drawContext.PrimitiveType, drawContext.PrimitiveCount);

      if (indexContext.indexType == VK_INDEX_TYPE_UINT16)
        geoData.indexBuffer = RasterBuffer(processIndexBuffer<uint16_t>(geoData.indexCount, drawContext.StartIndex, indexContext, minIndex, maxIndex), 0, 2, indexContext.indexType);
      else
        geoData.indexBuffer = RasterBuffer(processIndexBuffer<uint32_t>(geoData.indexCount, drawContext.StartIndex, indexContext, minIndex, maxIndex), 0, 4, indexContext.indexType);

      // Unlikely, but invalid
      if (maxIndex == minIndex) {
        ONCE(Logger::info("[RTX-Compatibility-Info] Skipped invalid drawcall, no triangles detected in index buffer."));
        return finishPrepare(prepareFlagsForIgnoredDraws);
      }

      geoData.vertexCount = maxIndex - minIndex + 1;
      vertexIndexOffset += minIndex;
    } else {
      geoData.vertexCount = GetVertexCount(drawContext.PrimitiveType, drawContext.PrimitiveCount);
    }

    if (geoData.vertexCount == 0) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped invalid drawcall, no vertices detected."));
      return finishPrepare(prepareFlagsForIgnoredDraws);
    }
    phaseTimer.lap(RtxGpuPassTimer::CpuCounter::AppPrepIndices);

    if (m_frameOptions.raytracedRenderTargetEnable) {
      // If this draw call has an RT texture bound
      if (m_activeDrawCallState.isUsingRaytracedRenderTarget) {
        // We validate this state below
        m_activeDrawCallState.isUsingRaytracedRenderTarget = false;
        // Try and find the has of the positions
        for (uint32_t i : bit::BitMask(m_parent->GetActiveRTTextures())) {
          D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[i]);
          if (texture == nullptr || texture->GetImage() == nullptr) {
            continue;
          }
          // Manual tags only here (auto-detection applies to drawing *to* an RT, not sampling it).
          if (isTaggedRaytracedRenderTarget(texture->GetImage(), false)) {
            m_activeDrawCallState.isUsingRaytracedRenderTarget = true;
          }
        }
      }
    }

    m_activeDrawCallState.categories = 0;
    m_activeDrawCallState.materialData = {};
    m_ue3ParticleColorTexcoordIndex = UINT32_MAX;
    m_ue3ParticleColorCaptureFlags = 0;

    // Fetch all the legacy state (colour modes, alpha test, etc...)
    setLegacyMaterialState(m_parent, m_parent->m_alphaSwizzleRTs & (1 << kRenderTargetIndex), m_frameOptions.vertexColorIsBakedLighting, m_activeDrawCallState.materialData);

    // Fetch fog state 
    setFogState(m_parent, m_activeDrawCallState.fogState);

    // Fetch all the render state and send it to rtx context (textures, transforms, etc.)
    if (!processRenderState(drawContext)) {
      return finishPrepare(prepareFlagsForIgnoredDraws);
    }
    phaseTimer.lap(RtxGpuPassTimer::CpuCounter::AppPrepRenderState);

    // Max offseted index value within a buffer slice that geoData contains
    const uint32_t maxOffsetedIndex = maxIndex - minIndex;

    // Copy all the vertices into a staging buffer.  Assign fields of the geoData structure.
    processVertices(vertexContext, vertexIndexOffset, geoData);

    readUe3DrawInstances(vertexContext, drawContext, geoData);
    updateUe3SkinnedDrawIdentity();

    if (!resolveUe3DrawCaptureSource(indexContext, vertexContext, geoData)) {
      return finishPrepare(prepareFlagsForIgnoredDraws);
    }
    phaseTimer.lap(RtxGpuPassTimer::CpuCounter::AppPrepVertices);

    Ue3StaticVertexCaptureKey staticCapture;
    {
      ScopedCpuProfileZoneN("UE3 geometry identity keys");
      staticCapture = computeUe3DrawStaticVertexCaptureKey(indexContext, vertexContext, drawContext, geoData);
      const Ue3GeometryMemoLookup memo = lookupUe3GeometryMemo(indexContext, vertexContext, drawContext, geoData);

      if (!memo.served) {
        geoData.futureGeometryHashes = computeHash(geoData, maxOffsetedIndex, memo.publishTo, memo.verifyAgainst);
        geoData.futureBoundingBox = computeAxisAlignedBoundingBox(geoData, memo.publishTo, memo.verifyAgainst);

        if (memo.publishTo != nullptr && !geoData.futureGeometryHashes.valid()) {
          // hashing could not be scheduled (e.g. undefined position region): drop the
          // placeholder entry so it does not linger unfilled
          m_ue3GeometryMemo.erase(memo.key);
        }
      }
    }
    phaseTimer.lap(RtxGpuPassTimer::CpuCounter::AppPrepIdentity);

    // Process skinning data
    m_activeDrawCallState.futureSkinningData = processSkinning(geoData);

    const bool reusedCachedVertexCapture = tryReuseUe3DrawStaticVertexCapture(staticCapture, geoData);

    // Every captured member is written at data[vertexId - baseVertex], so a draw whose vertices run
    // once per hardware instance has all of its instances aliasing one another. The declaration
    // already supplied this draw's positions, normals and UVs in object space.
    const bool instancedDrawSuppressesCapture = m_currentUe3Instancing.instanceCount > 1;

    // For shader based drawcalls we also want to capture the vertex shader output
    bool needVertexCapture =
      m_parent->UseProgrammableVS() &&
      m_frameOptions.useVertexCapture &&
      !reusedCachedVertexCapture &&
      !instancedDrawSuppressesCapture;
    if (needVertexCapture) {
      needVertexCapture = prepareVertexCapture(vertexIndexOffset, m_activeCapturePositionSource,
                                               /* allowPooledBuffer */ !staticCapture.canUseCache);
    }
    if (needVertexCapture) {
      recordUe3StaticVertexCapture(staticCapture, reusedCachedVertexCapture, geoData);
    }
    phaseTimer.lap(RtxGpuPassTimer::CpuCounter::AppPrepCapture);

    m_activeDrawCallState.usesVertexShader = m_parent->UseProgrammableVS();
    m_activeDrawCallState.usesPixelShader = m_parent->UseProgrammablePS();

    if (m_activeDrawCallState.usesVertexShader) {
      m_activeDrawCallState.programmableVertexShaderInfo = d3d9State().vertexShader->GetCommonShader()->GetInfo();
    }
    
    if (m_activeDrawCallState.usesPixelShader) {
      m_activeDrawCallState.programmablePixelShaderInfo = d3d9State().pixelShader->GetCommonShader()->GetInfo();
    }
    
    m_activeDrawCallState.cameraType = CameraType::Unknown;

    m_activeDrawCallState.minZ = std::clamp(d3d9State().viewport.MinZ, 0.0f, 1.0f);
    m_activeDrawCallState.maxZ = std::clamp(d3d9State().viewport.MaxZ, 0.0f, 1.0f);

    m_activeDrawCallState.zWriteEnable = d3d9State().renderStates[D3DRS_ZWRITEENABLE];
    m_activeDrawCallState.zEnable = d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_TRUE;
    
    // Now that the DrawCallState is complete, we can use heuristics for detection
    m_activeDrawCallState.setupCategoriesForHeuristics(m_seenCameraPositionsPrev.size(),
                                                       m_seenCameraPositions);

    if (m_frameOptions.fogIgnoreSky && m_activeDrawCallState.categories.test(InstanceCategories::Sky)) {
      m_activeDrawCallState.fogState.mode = D3DFOG_NONE;
    }

    // Ignore sky draw calls that are being drawn to a Raytraced Render Target
    // Raytraced Render Target scenes just use the same sky as the main scene, no need to duplicate them
    if (m_activeDrawCallState.isDrawingToRaytracedRenderTarget && m_activeDrawCallState.categories.test(InstanceCategories::Sky)) {
      return finishPrepare(prepareFlagsForIgnoredDraws);
    }

    assert(status == RtxGeometryStatus::RayTraced);

    logUe3InstancedDrawOnce(drawContext, vertexContext, geoData);
    logUe3TracedDrawOnce(drawContext, vertexContext, geoData);

    const bool preserveOriginalDraw = needVertexCapture;

    return finishPrepare(
      PrepareDrawFlag::CommitToRayTracing |
      (m_activeDrawCallState.testCategoryFlags(CATEGORIES_REQUIRE_DRAW_CALL_STATE) ? PrepareDrawFlag::ApplyDrawState : 0) |
      (preserveOriginalDraw ? PrepareDrawFlag::PreserveDrawCallAndItsState : 0));
  }

  void D3D9Rtx::triggerInjectRTX(const Rc<DxvkImage>& targetImage, const Rc<DxvkImage>& hdrCanvas) {
    // Flush any pending game and RTX work
    m_parent->Flush();

    // Send command to inject RTX
    m_parent->EmitCs([cReflexFrameId = GetReflexFrameId(), cTargetImage = targetImage, cHdrCanvas = hdrCanvas](DxvkContext* ctx) {
      static_cast<RtxContext*>(ctx)->injectRTX(cReflexFrameId, cTargetImage, cHdrCanvas);
    });
  }

  void D3D9Rtx::CommitGeometryToRT(const DrawContext& drawContext) {
    ScopedCpuProfileZone();
    auto drawInfo = m_parent->GenerateDrawInfo(drawContext.PrimitiveType, drawContext.PrimitiveCount, m_parent->GetInstanceCount());

    DrawParameters params;
    params.instanceCount = drawInfo.instanceCount;
    params.vertexOffset = drawContext.BaseVertexIndex;
    params.firstIndex = drawContext.StartIndex;
    // DXVK overloads the vertexCount/indexCount in DrawInfo
    if (drawContext.Indexed) {
      params.indexCount = drawInfo.vertexCount; 
    } else {
      params.vertexCount = drawInfo.vertexCount;
    }

    if (!m_ue3DecomposedInstances.empty()) {
      // UE3 only instances through programmable vertex shaders, for which processSkinning returns
      // nothing. Were skinning data pending, duplicating the draw state would hand several copies
      // the same one-shot task, and finalizeSkinningData would recompute objectToWorld from the
      // camera and discard the per-instance placement anyway - so leave such a draw undecomposed.
      if (m_activeDrawCallState.futureSkinningData.valid()) {
        ONCE(Logger::warn("[RTX-Compatibility] Not decomposing a hardware-instanced draw with pending "
                          "skinning data; the batch will be placed as a single instance."));
      } else {
        // One ray-traced instance per hardware instance, each carrying the placement read out of the
        // instance stream. The geometry is a single object-space copy of the mesh, so this reproduces
        // what UE3's non-instanced NxFluid mesh path submits: one draw per particle with
        // LocalToWorld = FMatrix(XAxis, YAxis, ZAxis, Location).
        params.instanceCount = 1;
        submitUe3DecomposedInstances(params);
        return;
      }
    }

    submitActiveDrawCallState();

    m_parent->EmitCs([params, this](DxvkContext* ctx) {
      assert(dynamic_cast<RtxContext*>(ctx));
      DrawCallState drawCallState;
      if (m_drawCallStateQueue.pop(drawCallState)) {
        static_cast<RtxContext*>(ctx)->commitGeometryToRT(params, drawCallState);
      }
    });
  }

  void D3D9Rtx::submitActiveDrawCallState() {
    // The next draw carries on from m_activeDrawCallState, so the queue gets a copy of it.
    // We must be prepared for `push` failing here, this can happen, since we're pushing to a circular buffer, which 
    //  may not have room for new entries.  In such cases, we trust that the consumer thread will make space for us, and
    //  so we may just need to wait a little bit.
    DrawCallState drawCallState = m_activeDrawCallState;
    while (!m_drawCallStateQueue.push(std::move(drawCallState))) {
      Sleep(0);
    }
  }

  Future<SkinningData> D3D9Rtx::processSkinning(const RasterGeometry& geoData) {
    ScopedCpuProfileZone();

    static const auto kEmptySkinningFuture = Future<SkinningData>();

    if (m_parent->UseProgrammableVS()) {
      return kEmptySkinningFuture;
    }

    // Some games set vertex blend without enough data to actually do the blending, handle that logic below.

    const bool hasBlendWeight = geoData.blendWeightBuffer.defined();
    const bool hasBlendIndices = d3d9State().vertexDecl != nullptr ? d3d9State().vertexDecl->TestFlag(D3D9VertexDeclFlag::HasBlendIndices) : false;
    const bool indexedVertexBlend = hasBlendIndices && d3d9State().renderStates[D3DRS_INDEXEDVERTEXBLENDENABLE];

    if (d3d9State().renderStates[D3DRS_VERTEXBLEND] == D3DVBF_DISABLE) {
      return kEmptySkinningFuture;
    }

    if (d3d9State().renderStates[D3DRS_VERTEXBLEND] != D3DVBF_0WEIGHTS) {
      if (!hasBlendWeight) {
        return kEmptySkinningFuture;
      }
    } else if (!indexedVertexBlend) {
      return kEmptySkinningFuture;
    }

    // We actually have skinning data now, process it!

    uint32_t numBonesPerVertex = 0;
    switch (d3d9State().renderStates[D3DRS_VERTEXBLEND]) {
    case D3DVBF_0WEIGHTS: numBonesPerVertex = 1; break;
    case D3DVBF_1WEIGHTS: numBonesPerVertex = 2; break;
    case D3DVBF_2WEIGHTS: numBonesPerVertex = 3; break;
    case D3DVBF_3WEIGHTS: numBonesPerVertex = 4; break;
    }

    const uint32_t vertexCount = geoData.vertexCount;

    HashQuery blendIndices;
    // Analyze the vertex data and find the min and max bone indices used in this mesh.
    // The min index is used to detect a case when vertex blend is enabled but there is just one bone used in the mesh,
    // so we can drop the skinning pass. That is processed in RtxContext::commitGeometryToRT(...)
    if (indexedVertexBlend && geoData.blendIndicesBuffer.defined()) {
      auto& buffer = geoData.blendIndicesBuffer;

      blendIndices.pBase = (uint8_t*) buffer.mapPtr(buffer.offsetFromSlice());
      blendIndices.elementSize = imageFormatInfo(buffer.vertexFormat())->elementSize;
      blendIndices.stride = buffer.stride();
      blendIndices.size = blendIndices.stride * vertexCount;
      blendIndices.ref = buffer.buffer().ptr();

      // Acquire prevents the staging allocator from re-using this memory
      blendIndices.ref->acquire(DxvkAccess::Read);
      // Make sure we hold on to this reference while the hashing is in flight
      blendIndices.ref->incRef();
    } else {
      blendIndices.ref = nullptr;
    }

    // Copy bones up to the max bone we have registered so far.
    const uint32_t maxBone = m_maxBone > 0 ? m_maxBone : 255;
    const uint32_t startBoneTransform = GetTransformIndex(D3DTS_WORLDMATRIX(0));

    const uint32_t nMat = maxBone + 1;
    const Matrix4* const boneMatrices = m_stagedBones.stageBones(
        d3d9State().transforms.data() + startBoneTransform, nMat);

    return m_pGeometryWorkers->Schedule([boneMatrices, blendIndices, numBonesPerVertex, vertexCount]()->SkinningData {
      ScopedCpuProfileZone();
      uint32_t numBones = numBonesPerVertex;

      int minBoneIndex = 0;
      if (blendIndices.ref) {
        const uint8_t* pBlendIndices = blendIndices.pBase;
        // Find out how many bone indices are specified for each vertex.
        // This is needed to find out the min bone index and ignore the padding zeroes.
        int maxBoneIndex = -1;
        if (!getMinMaxBoneIndices(pBlendIndices, blendIndices.stride, vertexCount, numBonesPerVertex, minBoneIndex, maxBoneIndex)) {
          minBoneIndex = 0;
          maxBoneIndex = 0;
        }
        numBones = maxBoneIndex + 1;

        // Release this memory back to the staging allocator
        blendIndices.ref->release(DxvkAccess::Read);
        blendIndices.ref->decRef();
      }

      // Pass bone data to RT back-end

      SkinningData skinningData;
      skinningData.pBoneMatrices.reserve(numBones);

      for (uint32_t n = 0; n < numBones; n++) {
        skinningData.pBoneMatrices.push_back(boneMatrices[n]);
      }

      skinningData.minBoneIndex = minBoneIndex;
      skinningData.numBones = numBones;
      skinningData.numBonesPerVertex = numBonesPerVertex;
      skinningData.computeHash(); // Computes the hash and stores it in the skinningData itself

      return skinningData;
    });
  }

  // The legacy albedo path is 2D, so a cube map binds a face view rather than a white placeholder.
  Rc<DxvkImageView> D3D9Rtx::getRemixSampleView(D3D9CommonTexture* texture, const bool srgb) {
    if (texture == nullptr)
      return nullptr;
    if (texture->GetType() == D3DRTYPE_CUBETEXTURE) {
      return texture->GetCubeFaceView(srgb);
    }
    return texture->GetSampleView(srgb);
  }

  template<bool FixedFunction>
  bool D3D9Rtx::processTextures() {
    ScopedCpuProfileZone();
    // We don't support full legacy materials in fixed function mode yet..
    // This implementation finds the most relevant textures bound from the
    // following criteria:
    //   - Texture actually bound (and used) by stage
    //   - First N textures bound to a specific texcoord index
    //   - Prefer lowest texcoord index
    // In non-fixed function (shaders), take the first N textures.

    // Used args for a given operation.
    auto ArgsMask = [](DWORD Op) {
      switch (Op) {
      case D3DTOP_DISABLE:
        return 0b000u; // No Args
      case D3DTOP_SELECTARG1:
      case D3DTOP_PREMODULATE:
        return 0b010u; // Arg 1
      case D3DTOP_SELECTARG2:
        return 0b100u; // Arg 2
      case D3DTOP_MULTIPLYADD:
      case D3DTOP_LERP:
        return 0b111u; // Arg 0, 1, 2
      default:
        return 0b110u; // Arg 1, 2
      }
    };

    // Currently we only support 2 textures
    constexpr uint32_t NumTexcoordBins = FixedFunction ? (D3DDP_MAXTEXCOORD * LegacyMaterialData::kMaxSupportedTextures) : LegacyMaterialData::kMaxSupportedTextures;

    bool useStageTextureFactorBlending = true;
    bool useMultipleStageTextureFactorBlending = false;

    // Build a mapping of texcoord indices to stage
    const uint8_t kInvalidStage = 0xFF;
    uint8_t texcoordIndexToStage[NumTexcoordBins];
    if constexpr (FixedFunction) {
      memset(&texcoordIndexToStage[0], kInvalidStage, sizeof(texcoordIndexToStage));
      for (uint32_t stage = 0; stage < caps::TextureStageCount; stage++) {
        auto isTextureFactorBlendingEnabled = [&](const auto& tss) -> bool {
          const auto colorOp = tss[DXVK_TSS_COLOROP];
          const auto alphaOp = tss[DXVK_TSS_ALPHAOP];

          if (colorOp == D3DTOP_DISABLE && alphaOp == D3DTOP_DISABLE)
            return false;

          const auto a1c = tss[DXVK_TSS_COLORARG1] & D3DTA_SELECTMASK;
          const auto a2c = tss[DXVK_TSS_COLORARG2] & D3DTA_SELECTMASK;
          const auto a1a = tss[DXVK_TSS_ALPHAARG1] & D3DTA_SELECTMASK;
          const auto a2a = tss[DXVK_TSS_ALPHAARG2] & D3DTA_SELECTMASK;

          // If previous stage wrote to TEMP the prior result source this stage
          // should read is D3DTA_TEMP otherwise its D3DTA_CURRENT.
          DWORD prevResultSel = D3DTA_CURRENT;
          if (stage != 0) {
            const auto& prev = d3d9State().textureStages[stage - 1];
            const auto resultArg = prev[DXVK_TSS_RESULTARG] & D3DTA_SELECTMASK;
            prevResultSel = (resultArg == D3DTA_TEMP) ? D3DTA_TEMP : D3DTA_CURRENT;
          }

          auto isModulate = [](DWORD op) {
            return op == D3DTOP_MODULATE || op == D3DTOP_MODULATE2X || op == D3DTOP_MODULATE4X;
          };

          const bool colorMul =
            isModulate(colorOp) &&
            ((a1c == D3DTA_TFACTOR && a2c == prevResultSel) ||
             (a2c == D3DTA_TFACTOR && a1c == prevResultSel));

          const bool alphaMul =
            isModulate(alphaOp) &&
            ((a1a == D3DTA_TFACTOR && a2a == prevResultSel) ||
             (a2a == D3DTA_TFACTOR && a1a == prevResultSel));

          return colorMul || alphaMul;
        };

        // Support texture factor blending besides the first stage. Currently, we only support 1 additional stage tFactor blending.
        // Note: If the tFactor is disabled for current texture (useStageTextureFactorBlending) then we should ignore the multiple stage tFactor blendings.
        bool isCurrentStageTextureFactorBlendingEnabled = false;
        if (useStageTextureFactorBlending &&
            m_frameOptions.enableMultiStageTextureFactorBlending &&
            stage != 0 &&
            isTextureFactorBlendingEnabled(d3d9State().textureStages[stage])) {
          isCurrentStageTextureFactorBlendingEnabled = true;
          useMultipleStageTextureFactorBlending = true;
        }

        if (d3d9State().textures[stage] == nullptr)
          continue;

        const auto& data = d3d9State().textureStages[stage];

        // Subsequent stages do not occur if this is true.
        if (data[DXVK_TSS_COLOROP] == D3DTOP_DISABLE)
          break;

        const std::uint32_t argsMask = ArgsMask(data[DXVK_TSS_COLOROP]) | ArgsMask(data[DXVK_TSS_ALPHAOP]);
        const auto firstTexMask  = ((data[DXVK_TSS_COLORARG0] & D3DTA_SELECTMASK) == D3DTA_TEXTURE) || ((data[DXVK_TSS_ALPHAARG0] & D3DTA_SELECTMASK) == D3DTA_TEXTURE);
        const auto secondTexMask = ((data[DXVK_TSS_COLORARG1] & D3DTA_SELECTMASK) == D3DTA_TEXTURE) || ((data[DXVK_TSS_ALPHAARG1] & D3DTA_SELECTMASK) == D3DTA_TEXTURE);
        const auto thirdTexMask  = ((data[DXVK_TSS_COLORARG2] & D3DTA_SELECTMASK) == D3DTA_TEXTURE) || ((data[DXVK_TSS_ALPHAARG2] & D3DTA_SELECTMASK) == D3DTA_TEXTURE);
        const std::uint32_t texMask =
          (firstTexMask  ? 0b001 : 0) |
          (secondTexMask ? 0b010 : 0) |
          (thirdTexMask  ? 0b100 : 0);

        // Is texture used?
        if ((argsMask & texMask) == 0)
          continue;

        D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[stage]);

        // Remix can only handle 2D textures - no volumes.
        if (texture->GetType() != D3DRTYPE_TEXTURE && (!m_frameOptions.allowCubemaps || texture->GetType() != D3DRTYPE_CUBETEXTURE)) {
          continue;
        }

        const XXH64_hash_t texHash = texture->GetSampleView(true)->image()->getHash();

        // Currently we only support regular textures, skip lightmaps.
        if (isLightmapTexture(texHash)) {
          continue;
        }

        // Allow for two stage candidates per texcoord index
        const uint32_t texcoordIndex = data[DXVK_TSS_TEXCOORDINDEX] & 0b111;
        const uint32_t candidateIndex = texcoordIndex * LegacyMaterialData::kMaxSupportedTextures;
        const uint32_t subIndex = (texcoordIndexToStage[candidateIndex] == kInvalidStage) ? 0 : 1;

        // Don't override if candidate exists
        if (texcoordIndexToStage[candidateIndex + subIndex] == kInvalidStage)
          texcoordIndexToStage[candidateIndex + subIndex] = stage;

        // Check if texture factor blending is enabled for the first stage
        if (useStageTextureFactorBlending && stage == 0) {
          isCurrentStageTextureFactorBlendingEnabled = isTextureFactorBlendingEnabled(d3d9State().textureStages[stage]);
        }

        // Check if texture factor blending is enabled
        if (isCurrentStageTextureFactorBlendingEnabled &&
            lookupHash(*m_frameOptions.ignoreBakedLightingTextures, texHash)) {
          useStageTextureFactorBlending = false;
          useMultipleStageTextureFactorBlending = false;
        }
      }
    }

    // Find the ideal textures for raytracing, initialize the data to invalid (out of range) to unbind unused textures
    uint32_t firstStage = 0;
    m_activeDrawCallState.materialData.colorTextureIsSrgb = false;
    m_texcoordCompU = 0;
    m_texcoordCompV = 1;
    m_iaTexcoordIndex = 0;
    Ue3TextureState ue3 = beginUe3TextureState(!FixedFunction);

    if constexpr (FixedFunction) {
      uint32_t textureID = 0;
      for (uint32_t idx = 0; idx < NumTexcoordBins && textureID < LegacyMaterialData::kMaxSupportedTextures; idx++) {
        const uint8_t stage = texcoordIndexToStage[idx];
        if (stage == kInvalidStage || d3d9State().textures[stage] == nullptr) {
          continue;
        }

        D3D9CommonTexture* pTexInfo = GetCommonTexture(d3d9State().textures[stage]);
        assert(pTexInfo != nullptr);
        const XXH64_hash_t texHash = pTexInfo->GetImage()->getHash();
        const XXH64_hash_t texDescHash =
          pTexInfo->GetImage() != nullptr
            ? pTexInfo->GetImage()->getDescriptorHash()
            : kEmptyHash;

        if (texHash == kEmptyHash) {
          continue;
        }

        if (textureID == 0) {
          firstStage = stage;
        }

        D3D9SamplerKey key = m_parent->CreateSamplerKey(stage);
        XXH64_hash_t samplerHash = D3D9SamplerKeyHash{}(key);

        Rc<DxvkSampler> sampler;
        auto samplerIt = m_samplerCache.find(samplerHash);
        if (samplerIt != m_samplerCache.end()) {
          sampler = samplerIt->second;
        } else {
          const auto samplerInfo = m_parent->DecodeSamplerKey(key);
          sampler = m_parent->GetDXVKDevice()->createSampler(samplerInfo);
          m_samplerCache.insert(std::make_pair(samplerHash, sampler));
        }

        // Cache the slot we want to bind
        const bool srgb = d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1;
        Rc<DxvkImageView> sampleView = getRemixSampleView(pTexInfo, srgb);
        if (sampleView == nullptr) {
          continue;
        }
        m_activeDrawCallState.materialData.colorTextures[textureID] = TextureRef(sampleView);
        m_activeDrawCallState.materialData.samplers[textureID] = sampler;
        ue3.selectedUe3MovieTexture |= isUe3MovieTextureDescHash(texDescHash);
        if (textureID == 0) {
          m_activeDrawCallState.materialData.colorTextureIsSrgb = srgb;
        }

        auto shaderSampler = RemapStateSamplerShader(stage);
        m_activeDrawCallState.materialData.colorTextureSlot[textureID] = computeResourceSlotId(shaderSampler.first, DxsoBindingType::Image, uint32_t(shaderSampler.second));

        ++textureID;
      }
    } else {
      selectUe3BoundTextures(ue3, firstStage);
    }

    // Update the drawcall state with texture stage info
    // note: D3D9 exposes more sampler slots than fixed-function texture stages (limited to 8).
    // `setTextureStageState` reads `textureStages[stageIdx]` and `D3DTS_TEXTURE0 + stageIdx`, so clamp to a valid stage index.
    const uint32_t stageStateIdx = (firstStage < caps::TextureStageCount) ? firstStage : 0;
    if (unlikely(firstStage >= caps::TextureStageCount)) {
      ONCE(Logger::warn(str::format(
        "[RTX-Compatibility] Shader-path selected sampler stage ", firstStage,
        " but texture stage state is limited to 0..", (caps::TextureStageCount - 1),
        ". Using stage 0 for texcoord/textureTransform.")));
    }

    setTextureStageState(d3d9State(), stageStateIdx, useStageTextureFactorBlending, useMultipleStageTextureFactorBlending,
                         m_activeDrawCallState.materialData, m_activeDrawCallState.transformData);

    if constexpr (!FixedFunction) {
      if (m_frameOptions.ue3EngineMode) {
        // shader-path draws perform UV math in shader code
        // fixed-function texture transform/texgen state can be stale and should not be reused
        m_activeDrawCallState.transformData.textureTransform = Matrix4();
        m_activeDrawCallState.transformData.texgenMode = TexGenMode::None;
      }
    }

    // Texture-less (constant-color) materials still need MIC identity for a stable
    // material hash (tagging, replacements, categories) and their color constant
    bool ue3MicIdentityAvailable = false;
    if constexpr (!FixedFunction) {
      ue3MicIdentityAvailable =
        m_frameOptions.ue3EngineMode &&
        m_parent->UseProgrammablePS() &&
        d3d9State().pixelShader.ptr() != nullptr;
    }

    if (d3d9State().textures[firstStage] || ue3MicIdentityAvailable) {
      if constexpr (!FixedFunction) {
        computeUe3MaterialIdentity(ue3, firstStage);
      }
      m_activeDrawCallState.materialData.updateCachedHash();

      if constexpr (!FixedFunction) {
        if (ue3MicIdentityAvailable) {
          registerUe3TexturelessMaterial();
        }
      }

      m_activeDrawCallState.setupCategoriesForTexture();

      // Track the material hash before checking if it should be ignored
      // This ensures we track all materials sent by the game, not just the ones that are actually rendered.
      const XXH64_hash_t textureHash = m_activeDrawCallState.materialData.getColorTexture().getImageHash();
      const XXH64_hash_t materialHash = m_activeDrawCallState.materialData.getHash();
      markUe3MovieTextureMaterial(ue3, materialHash, textureHash);

      // Flag smooth normals category at the d3d9 layer
      m_activeDrawCallState.setCategory(InstanceCategories::SmoothNormals, lookupHash(*m_frameOptions.smoothNormalsTextures, textureHash) || lookupHash(*m_frameOptions.smoothNormalsTextures, materialHash));
      if (materialHash != kEmptyHash && SceneManager::s_hashUsageTrackingWanted.load(std::memory_order_relaxed)) {
        // batched into a single CS command in EndFrame (see m_pendingReplacementMaterialHashes)
        m_pendingReplacementMaterialHashes.push_back(materialHash);
      }
      
      // Check if an ignore texture is bound
      if (m_activeDrawCallState.getCategoryFlags().test(InstanceCategories::Ignore)) {
        return false;
      }

      if (m_activeDrawCallState.testCategoryFlags(InstanceCategories::Terrain)) {
        if (m_frameOptions.terrainAsDecalsEnabledIfNoBaker && !TerrainBaker::enableBaking()) {

          m_activeDrawCallState.removeCategory(InstanceCategories::Terrain);
          m_activeDrawCallState.setCategory(InstanceCategories::DecalStatic, true);

          // modulate to compensate the multilayer blending
          DxvkRtTextureOperation& texop = m_activeDrawCallState.materialData.textureColorOperation;
          if (m_frameOptions.terrainAsDecalsAllowOverModulate) {
            if (texop == DxvkRtTextureOperation::Modulate2x || texop == DxvkRtTextureOperation::Modulate4x) {
              texop = DxvkRtTextureOperation::Force_Modulate2x;
            }
          }
        }
      }

      if (!m_forceGeometryCopy && m_frameOptions.alwaysCopyDecalGeometries) {
        // Only poke decal hashes when option is enabled.
        m_forceGeometryCopy |= m_activeDrawCallState.testCategoryFlags(CATEGORIES_REQUIRE_GEOMETRY_COPY);
      }
    } else {
      // No texture / MIC identity: still refresh the material hash once so the
      // submitted draw state carries a valid hash.
      m_activeDrawCallState.materialData.updateCachedHash();
    }

    // only keep the passthrough texcoord index for selecting the vertex declaration element
    // upper bits are D3DTSS_TCI_* flags used for fixed-function texgen
    uint32_t texcoordIdx = d3d9State().textureStages[stageStateIdx][DXVK_TSS_TEXCOORDINDEX] & 0b111;
    uint32_t iaTexcoordIdx = texcoordIdx;
    m_uvResolutionMode = UvResolutionMode::LegacyTss;

    m_forceIaTexcoordForOutlier = [&]() {
      const auto& outlierSet = *m_frameOptions.vsTexcoordCaptureOutlierTextures;
      if (outlierSet.empty()) {
        return false;
      }
      for (uint32_t i = 0; i < LegacyMaterialData::kMaxSupportedTextures; i++) {
        if (lookupHash(outlierSet, m_activeDrawCallState.materialData.colorTextures[i].getImageHash())) {
          return true;
        }
      }

      for (uint32_t stage = 0; stage < SamplerCount; stage++) {
        if (d3d9State().textures[stage] == nullptr) {
          continue;
        }

        D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[stage]);
        if (texture == nullptr || texture->GetImage() == nullptr) {
          continue;
        }

        if (lookupHash(outlierSet, texture->GetImage()->getHash())) {
          return true;
        }
      }

      return false;
    }();

    if constexpr (!FixedFunction) {
      resolveUe3Texcoords(ue3, firstStage, stageStateIdx, texcoordIdx, iaTexcoordIdx);
    }

    m_texcoordIndex = texcoordIdx;
    m_iaTexcoordIndex = resolveIaTexcoordIndex(iaTexcoordIdx);

    return true;
  }

  // UE3 TEXCOORD elements that are not UV sets: vertex lightmap coefficients (D3DCOLOR-typed, TEXCOORD5..7)
  // and the per-instance basis on an instance-data stream (TEXCOORD1..4).
  uint32_t D3D9Rtx::resolveIaTexcoordIndex(const uint32_t iaTexcoordIdx) const {
    if (!m_frameOptions.ue3EngineMode || d3d9State().vertexDecl == nullptr) {
      return iaTexcoordIdx;
    }

    const uint32_t instanceDataStreamMask = m_currentUe3Instancing.instanceDataStreamMask;
    auto isNonUvElement = [instanceDataStreamMask](const D3DVERTEXELEMENT9& element) {
      if (element.Usage != D3DDECLUSAGE_TEXCOORD) {
        return false;
      }
      if (element.Type == D3DDECLTYPE_D3DCOLOR) {
        return true;
      }
      return element.Stream < caps::MaxStreams &&
             (instanceDataStreamMask & (1u << element.Stream)) != 0;
    };

    const auto& elements = d3d9State().vertexDecl->GetElements();
    bool requestedIsNonUv = false;
    for (const auto& element : elements) {
      if (element.Usage == D3DDECLUSAGE_TEXCOORD && element.UsageIndex == iaTexcoordIdx) {
        requestedIsNonUv = isNonUvElement(element);
        break;
      }
    }

    if (!requestedIsNonUv) {
      return iaTexcoordIdx;
    }

    // Fall back to the mesh's lowest real UV set rather than leaving the surface with the
    // lighting stream: an unresolvable index would drop the texcoord buffer entirely.
    uint32_t fallback = iaTexcoordIdx;
    bool found = false;
    for (const auto& element : elements) {
      if (element.Usage != D3DDECLUSAGE_TEXCOORD || isNonUvElement(element)) {
        continue;
      }
      if (!found || element.UsageIndex < fallback) {
        fallback = element.UsageIndex;
        found = true;
      }
    }

    return fallback;
  }

  bool D3D9Rtx::ignoreOcclusionTestDrawEarly() {
    if (!ShouldApplyConservativeOcclusionQueryState()) {
      return false;
    }

    // What makeDrawCallType and finishPrepare would have recorded for this ignored draw.
    ++m_drawCallID;
    if (RtxGpuPassTimer::isEnabled()) {
      ++m_drawDispositionStats.draws;
      ++m_drawDispositionStats.ignored;
    }
    return true;
  }

  PrepareDrawFlags D3D9Rtx::PrepareDrawGeometryForRT(const bool indexed, const DrawContext& context) {
    // Draws issued internally by the deferred UI overlay replay bypass classification and
    // execute as plain raster draws
    if (m_replayingDeferredUiDraws) {
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    // first-frame lazy init; steady-state refreshes happen once per frame in EndFrame
    if (unlikely(!m_frameOptions.valid)) {
      refreshFrameOptionCache();
    }

    if (!m_frameOptions.enableRaytracing || !m_enableDrawCallConversion || m_sceneCaptureSuspended) {
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    if (ignoreOcclusionTestDrawEarly()) {
      return PrepareDrawFlag::Ignore;
    }

    m_parent->PrepareTextures();

    IndexContext indices;
    if (indexed) {
      D3D9CommonBuffer* ibo = GetCommonBuffer(d3d9State().indices);
      assert(ibo != nullptr);

      indices.ibo = ibo;
      indices.indexBuffer = ibo->GetMappedSlice();
      indices.indexType = DecodeIndexType(ibo->Desc()->Format);
    }

    // Copy over the vertex buffers that are actually required
    VertexContext vertices[caps::MaxStreams];
    for (uint32_t i = 0; i < caps::MaxStreams; i++) {
      const auto& dx9Vbo = d3d9State().vertexBuffers[i];
      auto* vbo = GetCommonBuffer(dx9Vbo.vertexBuffer);
      if (vbo != nullptr) {
        vertices[i].stride = dx9Vbo.stride;
        vertices[i].offset = dx9Vbo.offset;
        vertices[i].buffer = vbo->GetBufferSlice<D3D9_COMMON_BUFFER_TYPE_MAPPING>();
        vertices[i].mappedSlice = vbo->GetMappedSlice();
        vertices[i].pVBO = vbo;

        // If staging upload has been enabled on a buffer then previous buffer lock:
        //   a) triggered a pipeline stall (overlapped mapped ranges, improper flags etc)
        //   b) does not have D3DLOCK_DONOTWAIT, or was in use at Map()
        // 
        // Buffers with staged uploads may have contents valid ONLY until next Map().
        // We must NOT use such buffer directly and have to always copy the contents.
        vertices[i].canUseBuffer = vbo->DoesStagingBufferUploads() == false;
      }
    }

    return internalPrepareDraw(indices, vertices, context);
  }

  PrepareDrawFlags D3D9Rtx::PrepareDrawUPGeometryForRT(const bool indexed,
                                                       const D3D9BufferSlice& buffer,
                                                       const D3DFORMAT indexFormat,
                                                       const uint32_t indexSize,
                                                       const uint32_t indexOffset,
                                                       const uint32_t vertexSize,
                                                       const uint32_t vertexStride,
                                                       const DrawContext& drawContext) {
    // Draws issued internally by the deferred UI overlay replay bypass classification and
    // execute as plain raster draws
    if (m_replayingDeferredUiDraws) {
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    // first-frame lazy init; steady-state refreshes happen once per frame in EndFrame
    if (unlikely(!m_frameOptions.valid)) {
      refreshFrameOptionCache();
    }

    if (!m_frameOptions.enableRaytracing || !m_enableDrawCallConversion || m_sceneCaptureSuspended) {
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    if (ignoreOcclusionTestDrawEarly()) {
      return PrepareDrawFlag::Ignore;
    }

    m_parent->PrepareTextures();

    // 'buffer' - contains vertex + index data (packed in that order)

    IndexContext indices;
    if (indexed) {
      indices.indexBuffer = buffer.slice.getSliceHandle(indexOffset, indexSize);
      indices.indexType = DecodeIndexType(static_cast<D3D9Format>(indexFormat));
    }

    VertexContext vertices[caps::MaxStreams];
    vertices[0].stride = vertexStride;
    vertices[0].offset = 0;
    vertices[0].buffer = buffer.slice.subSlice(0, vertexSize);
    vertices[0].mappedSlice = buffer.slice.getSliceHandle(0, vertexSize);
    vertices[0].canUseBuffer = true;

    return internalPrepareDraw(indices, vertices, drawContext);
  }

  void D3D9Rtx::ResetSwapChain(const D3DPRESENT_PARAMETERS& presentationParameters) {
    // Early out if the cached present parameters are not out of date

    if (m_activePresentParams.has_value()) {
      if (
        m_activePresentParams->BackBufferWidth == presentationParameters.BackBufferWidth &&
        m_activePresentParams->BackBufferHeight == presentationParameters.BackBufferHeight &&
        m_activePresentParams->BackBufferFormat == presentationParameters.BackBufferFormat &&
        m_activePresentParams->BackBufferCount == presentationParameters.BackBufferCount &&
        m_activePresentParams->MultiSampleType == presentationParameters.MultiSampleType &&
        m_activePresentParams->MultiSampleQuality == presentationParameters.MultiSampleQuality &&
        m_activePresentParams->SwapEffect == presentationParameters.SwapEffect &&
        m_activePresentParams->hDeviceWindow == presentationParameters.hDeviceWindow &&
        m_activePresentParams->Windowed == presentationParameters.Windowed &&
        m_activePresentParams->EnableAutoDepthStencil == presentationParameters.EnableAutoDepthStencil &&
        m_activePresentParams->AutoDepthStencilFormat == presentationParameters.AutoDepthStencilFormat &&
        m_activePresentParams->Flags == presentationParameters.Flags &&
        m_activePresentParams->FullScreen_RefreshRateInHz == presentationParameters.FullScreen_RefreshRateInHz &&
        m_activePresentParams->PresentationInterval == presentationParameters.PresentationInterval
      ) {
        return;
      }
    }

    // Cache the present parameters
    m_activePresentParams = presentationParameters;

    // Recreated at the new size on the next staged injection
    m_deferredUiHdrCanvas = nullptr;

    // Inform the backend about potential presenter update
    m_parent->EmitCs([cWidth = m_activePresentParams->BackBufferWidth,
                      cHeight = m_activePresentParams->BackBufferHeight](DxvkContext* ctx) {
      static_cast<RtxContext*>(ctx)->resetScreenResolution({ cWidth, cHeight , 1 });
    });
  }

  void D3D9Rtx::EndFrame(const Rc<DxvkImage>& targetImage, bool callInjectRtx) {
    // Refresh the per-frame option snapshot: EndFrame's own consumers (deferred UI
    // replay) read fresh values and the next frame's draws see this frame's resolution.
    refreshFrameOptionCache();

    updateUe3GamePatchRequest();

    // NV-DXVK start: draw disposition statistics
    reportDrawDispositionStats();
    // NV-DXVK end

    // Allow the next frame's TdToneMapping pass to be captured again
    m_ue3ToneMapCapturedThisFrame = false;

    // New frame: forget the UE3 foreground DPG segment
    m_ue3SeenMainViewWorldDraw = false;
    m_ue3ForegroundDpgActive = false;

    // Flush this frame's replacement-material-hash tracking as one CS command. Must be
    // emitted before the endFrame command below: the consumers (graph components) read
    // the per-frame map during SceneManager::onFrameEnd, and the map clears there too.
    if (!m_pendingReplacementMaterialHashes.empty()) {
      const size_t flushedCount = m_pendingReplacementMaterialHashes.size();
      m_parent->EmitCs([cHashes = std::move(m_pendingReplacementMaterialHashes)](DxvkContext* ctx) {
        SceneManager& sceneManager = static_cast<RtxContext*>(ctx)->getSceneManager();
        for (const XXH64_hash_t hash : cHashes) {
          sceneManager.trackReplacementMaterialHash(hash);
        }
      });
      // the move donates the capacity to the lambda; re-establish a defined empty state
      // and pre-size for the next frame's roughly equal draw count
      m_pendingReplacementMaterialHashes.clear();
      m_pendingReplacementMaterialHashes.reserve(flushedCount);
    }

    const auto currentReflexFrameId = GetReflexFrameId();

    // persist what this session learned so the next one scores from it rather than re-converging
    m_ue3TextureSpreadCache.saveIfDue(currentReflexFrameId);
    m_ue3DiffuseSelectionCache.saveIfDue(currentReflexFrameId);

    // Deferred overlays still pending mean no trigger draw fired this frame: inject here rather
    // than in endFrame's fallback, so they replay onto this frame's image. Frames not presenting
    // normally (e.g. alt-tab end-of-frame events) drop them.
    bool injected = false;
    if (!m_deferredUiDraws.empty() && callInjectRtx) {
      Com<IDirect3DSurface9> backBuffer;
      m_parent->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &backBuffer);
      if (backBuffer != nullptr) {
        injectRtxWithOverlays(targetImage, backBuffer.ptr());
        injected = true;
      }
    }
    m_deferredUiDraws.clear();
    m_deferredUiFrameVertexBytes = 0;

    // Flush any pending game and RTX work
    m_parent->Flush();

    // Inform backend of end-frame
    m_parent->EmitCs([currentReflexFrameId, targetImage, cCallInjectRtx = callInjectRtx && !injected](DxvkContext* ctx) { 
      static_cast<RtxContext*>(ctx)->endFrame(currentReflexFrameId, targetImage, cCallInjectRtx); 
    });

    const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();
    m_ue3StaticVertexCaptureCache.endFrame(currentFrame, m_frameOptions.ue3StaticVertexCaptureCache);
    m_ue3GeometryMemo.prune(currentFrame);
    reportUe3InstancedDrawStats();
    reportUe3ConstantChurn();

    DrawCallState::refreshCategoryLookupTable();

    // The per-draw profile zones in this file are only meaningful against the draw count, which
    // a UE3 title moves by an order of magnitude depending on its occlusion and frustum culling.
    ProfilerPlotValue("D3D9 Draw Calls", int64_t(m_drawCallID));

    // Reset for the next frame
    m_rtxInjectTriggered = false;
    m_drawCallID = 0;
    m_seenCameraPositionsPrev = std::move(m_seenCameraPositions);
    ++m_ue3FrameCounter;
    trimVertexCaptureBufferPool();

    // two-pass translucency dedup state must not span frames
    m_prevDrawVsPsHash = 0;
    m_prevDrawTextureHash = 0;
    m_prevDrawGeometryHash = 0;
    m_prevDrawCullMode = 0;

    m_stagedBones.clear();
  }

  void D3D9Rtx::OnPresent(const Rc<DxvkImage>& targetImage) {
    // Inform backend of present
    m_parent->EmitCs([targetImage](DxvkContext* ctx) { static_cast<RtxContext*>(ctx)->onPresent(targetImage); });
  }
}
