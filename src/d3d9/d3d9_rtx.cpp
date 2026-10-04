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
    saveUe3TextureSpreadCache();
    saveUe3DiffuseSelectionCache();
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

  const Direct3DState9& D3D9Rtx::d3d9State() const {
    return *m_parent->GetRawState();
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
    o.ue3StaticLocalMeshVertexCaptureCache = ue3StaticLocalMeshVertexCaptureCacheObject().get();
    o.ue3StaticLocalMeshVertexCaptureCacheWarmupFrames = ue3StaticLocalMeshVertexCaptureCacheWarmupFramesObject().get();
    o.ue3StaticLocalMeshVertexCaptureCacheBudgetMiB = ue3StaticLocalMeshVertexCaptureCacheBudgetMiBObject().get();
    o.ue3StaticLocalMeshVertexCaptureCacheMaxEntries = ue3StaticLocalMeshVertexCaptureCacheMaxEntriesObject().get();
    o.ue3StaticLocalMeshVertexCaptureCacheRetentionFrames = ue3StaticLocalMeshVertexCaptureCacheRetentionFramesObject().get();
    o.ue3StaticLocalMeshVertexCaptureCacheMinReusePercent = ue3StaticLocalMeshVertexCaptureCacheMinReusePercentObject().get();
    o.ue3StaticLocalMeshVertexCaptureCacheReuseProbeFrames = ue3StaticLocalMeshVertexCaptureCacheReuseProbeFramesObject().get();
    o.ue3LogStaticVertexCaptureCacheStats = ue3LogStaticVertexCaptureCacheStatsObject().get();
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
    s.lightmapTextureDigest = digestTextureTags(s.lightmapTextures);
    s.neverAlbedoTextureDigest = digestTextureTags(s.neverAlbedoTextures);
    s.preferredAlbedoTextureDigest = digestTextureTags(s.preferredAlbedoTextures);
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

    auto BoundShaderHas = [&](const D3D9CommonShader* shader, DxsoUsage usage, bool inOut)-> bool {
      if (shader == nullptr)
        return false;

      const auto& sgn = inOut ? shader->GetIsgn() : shader->GetOsgn();
      for (uint32_t i = 0; i < sgn.elemCount; i++) {
        const auto& decl = sgn.elems[i];
        if (decl.semantic.usageIndex == 0 && decl.semantic.usage == usage)
          return true;
      }
      return false;
    };

    auto BoundShaderHasAnyUsageIndex = [&](const D3D9CommonShader* shader, DxsoUsage usage, bool inOut)-> bool {
      if (shader == nullptr)
        return false;

      const auto& sgn = inOut ? shader->GetIsgn() : shader->GetOsgn();
      for (uint32_t i = 0; i < sgn.elemCount; i++) {
        const auto& decl = sgn.elems[i];
        if (decl.semantic.usage == usage)
          return true;
      }
      return false;
    };

    auto FindVsTexcoordOutputRegister = [&](const D3D9CommonShader* shader, uint32_t usageIndex) -> uint32_t {
      if (shader == nullptr)
        return std::numeric_limits<uint32_t>::max();

      const auto& osgn = shader->GetOsgn();
      for (uint32_t i = 0; i < osgn.elemCount; i++) {
        const auto& decl = osgn.elems[i];
        if (decl.semantic.usage == DxsoUsage::Texcoord && decl.semantic.usageIndex == usageIndex)
          return decl.regNumber;
      }
      return std::numeric_limits<uint32_t>::max();
    };

    auto FindVsTexcoordOutputRegisterByRegNumber = [&](const D3D9CommonShader* shader, uint32_t regNumber) -> uint32_t {
      if (shader == nullptr)
        return std::numeric_limits<uint32_t>::max();

      const auto& osgn = shader->GetOsgn();
      for (uint32_t i = 0; i < osgn.elemCount; i++) {
        const auto& decl = osgn.elems[i];
        if (decl.semantic.usage == DxsoUsage::Texcoord && decl.regNumber == regNumber)
          return decl.regNumber;
      }
      return std::numeric_limits<uint32_t>::max();
    };

    auto FindUniqueVsTexcoordOutputRegister = [&](const D3D9CommonShader* shader) -> uint32_t {
      if (shader == nullptr)
        return std::numeric_limits<uint32_t>::max();

      const auto& osgn = shader->GetOsgn();
      uint32_t foundReg = std::numeric_limits<uint32_t>::max();
      uint32_t foundCount = 0;
      for (uint32_t i = 0; i < osgn.elemCount; i++) {
        const auto& decl = osgn.elems[i];
        if (decl.semantic.usage != DxsoUsage::Texcoord)
          continue;

        foundReg = decl.regNumber;
        foundCount++;
        if (foundCount > 1)
          return std::numeric_limits<uint32_t>::max();
      }

      return foundCount == 1
        ? foundReg
        : std::numeric_limits<uint32_t>::max();
    };

    // Get common shaders to query what data we can capture
    const D3D9CommonShader* vertexShader = d3d9State().vertexShader.ptr() != nullptr ? d3d9State().vertexShader->GetCommonShader() : nullptr;

    RasterGeometry& geoData = m_activeDrawCallState.geometryData;

    const bool hasIaTexcoord = geoData.texcoordBuffer.defined();
    const bool forceIaTexcoordForOutlier = m_forceIaTexcoordForOutlier && hasIaTexcoord;

    // Known stride for vertex capture buffers
    const uint32_t stride = sizeof(CapturedVertex);
    const size_t vertexCaptureDataSize = align(geoData.vertexCount * stride, CACHE_LINE_SIZE);

    DxvkBufferSlice slice = allocVertexCaptureBuffer(vertexCaptureDataSize, allowPooledBuffer);

    geoData.positionBuffer = RasterBuffer(slice, 0, stride, VK_FORMAT_R32G32B32A32_SFLOAT);
    assert(geoData.positionBuffer.offset() % 4 == 0);

    // Obey the deterministic UV resolution made in processTextures:
    // - ProvenIa: the exact IA texcoord set is bound by processVertices; do not capture UVs
    // - CaptureInterpolant: capture the proven interpolant register/components (correct by
    //   construction for procedural or otherwise unprovable VS-side UV math)
    // - LegacyTss: no provable resolution; keep the legacy capture cascade
    uint32_t capturedTexcoordOutputRegister = std::numeric_limits<uint32_t>::max();
    switch (m_uvResolutionMode) {
    case UvResolutionMode::ProvenIa:
      if (!hasIaTexcoord) {
        // proven IA set is missing from this draw's declaration (degenerate);
        // the captured interpolant is still exact
        capturedTexcoordOutputRegister = FindVsTexcoordOutputRegister(vertexShader, m_texcoordIndex);
      }
      break;
    case UvResolutionMode::CaptureInterpolant:
      capturedTexcoordOutputRegister = FindVsTexcoordOutputRegister(vertexShader, m_texcoordIndex);
      if (capturedTexcoordOutputRegister == std::numeric_limits<uint32_t>::max())
        capturedTexcoordOutputRegister = FindUniqueVsTexcoordOutputRegister(vertexShader);
      break;
    case UvResolutionMode::LegacyTss:
    default:
      capturedTexcoordOutputRegister = FindVsTexcoordOutputRegister(vertexShader, m_texcoordIndex);
      if (capturedTexcoordOutputRegister == std::numeric_limits<uint32_t>::max()) {
        if (!m_frameOptions.ue3EngineMode)
          capturedTexcoordOutputRegister = FindVsTexcoordOutputRegisterByRegNumber(vertexShader, m_texcoordIndex);
      }
      if (capturedTexcoordOutputRegister == std::numeric_limits<uint32_t>::max())
        capturedTexcoordOutputRegister = FindUniqueVsTexcoordOutputRegister(vertexShader);
      break;
    }
    if (forceIaTexcoordForOutlier)
      capturedTexcoordOutputRegister = std::numeric_limits<uint32_t>::max();

    if (useVertexCapturedTexcoords()
        && BoundShaderHasAnyUsageIndex(vertexShader, DxsoUsage::Texcoord, false)
        && capturedTexcoordOutputRegister == std::numeric_limits<uint32_t>::max()) {
      capturedTexcoordOutputRegister = FindUniqueVsTexcoordOutputRegister(vertexShader);
    }

    // By default we only capture VS output texcoords when the input vertex declaration didn't
    // already provide them. Overriding valid input texcoords with the VS output is opt-in
    // (useVertexCapturedTexcoords), since the data a VS writes to the :TEXCOORD attribute isn't
    // always actual UVs and its memory layout can't be assumed for all games.
    // CaptureInterpolant is exempt: that mode is only selected when UV analysis proved the
    // sampled UV is VS-side math, so the bound IA set is just a backstop and must not
    // suppress the capture.
    const bool captureVsTexcoords =
      BoundShaderHasAnyUsageIndex(vertexShader, DxsoUsage::Texcoord, false)
      && capturedTexcoordOutputRegister != std::numeric_limits<uint32_t>::max()
      && (m_uvResolutionMode == UvResolutionMode::CaptureInterpolant
          || useVertexCapturedTexcoords()
          || !geoData.texcoordBuffer.defined()
          || !RtxGeometryUtils::isTexcoordFormatValid(geoData.texcoordBuffer.vertexFormat()));

    if (captureVsTexcoords) {
      const uint32_t texcoordOffset = offsetof(CapturedVertex, texcoord0);
      geoData.texcoordBuffer = RasterBuffer(slice, texcoordOffset, stride, VK_FORMAT_R32G32_SFLOAT);
      assert(geoData.texcoordBuffer.offset() % 4 == 0);
    }

    // normals for vertex-capture draws
    // captured positions are post-skinning for GPU-skinned draws whichever source they come from, so when a
    // skinned shader doesn't output a normal semantic IA normals alone are bind-pose and do not
    // match captured positions, for UE3 we therefore prioritise:
    // 1 VS NORMAL output when available
    // 2 VS COLOR0 encoded skinned normal output
    // 3 bone-skinned IA normal reconstruction in vertex-capture shader using BoneMatrices CTAB metadata
    // 4 bind-pose IA fallback only when none of the above can be used
    const bool vsOutputsNormal = BoundShaderHas(vertexShader, DxsoUsage::Normal, false);
    const bool vsHasNormalInput = BoundShaderHas(vertexShader, DxsoUsage::Normal, true);
    const bool normalFromDecl = geoData.normalBuffer.defined();
    const bool isGpuSkinned = geoData.blendWeightBuffer.defined() || geoData.blendIndicesBuffer.defined();

    const bool vsOutputsColor0 = BoundShaderHas(vertexShader, DxsoUsage::Color, false);
    const bool hasBlendWeights = geoData.blendWeightBuffer.defined();
    const bool hasBlendIndices = geoData.blendIndicesBuffer.defined();
    const bool hasAnyBlendStream = hasBlendWeights || hasBlendIndices;
    const Ue3VsShaderCtabInfo* ue3CtabInfo = m_currentUe3CtabInfo.has_value() ? &(*m_currentUe3CtabInfo) : nullptr;
    const bool hasBoneMatricesInCtab = ue3CtabInfo != nullptr &&
                                       ue3CtabInfo->hasBoneMatrices &&
                                       ue3CtabInfo->boneMatricesRegisterCount >= 3;
    const bool canUseBoneSkinnedNormalCapture =
      isGpuSkinned &&
      !vsOutputsNormal &&
      !vsOutputsColor0 &&
      vsHasNormalInput &&
      normalFromDecl &&
      hasAnyBlendStream &&
      hasBoneMatricesInCtab;

    if (Logger::logLevel() <= LogLevel::Debug) {
      const uint32_t normalDiagKey =
        (vsOutputsNormal ? 1u : 0u) | (vsHasNormalInput ? 2u : 0u) |
        (normalFromDecl ? 4u : 0u) | (isGpuSkinned ? 8u : 0u) |
        (vsOutputsColor0 ? 16u : 0u) |
        (hasBlendWeights ? 32u : 0u) |
        (hasBlendIndices ? 64u : 0u) |
        (hasBoneMatricesInCtab ? 128u : 0u) |
        (canUseBoneSkinnedNormalCapture ? 256u : 0u) |
        (normalFromDecl ? (uint32_t(geoData.normalBuffer.vertexFormat()) << 8) : 0u);
      static fast_unordered_set s_loggedNormalDiag;
      if (s_loggedNormalDiag.insert(normalDiagKey).second) {
        const char* caseLabel = "Case3-keepIA";
        if (vsOutputsNormal) caseLabel = "Case1-vsOutputNormal";
        else if (isGpuSkinned && vsOutputsColor0) caseLabel = "Case2a-COLOR0skinned";
        else if (canUseBoneSkinnedNormalCapture) caseLabel = "Case2b-boneSkinCapture";
        else if (isGpuSkinned && normalFromDecl) caseLabel = "Case2c-bindPoseFallback";
        else if (isGpuSkinned) caseLabel = "Case2d-noNormals";

        Logger::debug(str::format(
          "[RTX-Compatibility] Vertex capture normal [", caseLabel, "]: vsOutputsNormal=", vsOutputsNormal,
          ", vsHasNormalInput=", vsHasNormalInput,
          ", normalFromDecl=", normalFromDecl,
          normalFromDecl ? str::format(", declFmt=", geoData.normalBuffer.vertexFormat()).c_str() : "",
          ", isGpuSkinned=", isGpuSkinned,
          ", vsOutputsColor0=", vsOutputsColor0,
          ", hasBlendWeights=", hasBlendWeights,
          ", hasBlendIndices=", hasBlendIndices,
          ", hasBoneMatricesInCtab=", hasBoneMatricesInCtab,
          ", vertexCount=", geoData.vertexCount));
      }
    }

    uint32_t vertexCaptureFlags = 0;
    switch (positionSource) {
    case Ue3CapturePositionSource::InputAssembler:
      vertexCaptureFlags |= kVertexCaptureFlag_PositionFromInput;
      break;
    case Ue3CapturePositionSource::PreProjectionRegister:
      vertexCaptureFlags |= kVertexCaptureFlag_PositionFromPreProjection;
      break;
    case Ue3CapturePositionSource::ClipReconstruction:
      break;
    }

    if (vsOutputsNormal && (m_frameOptions.useVertexCapturedNormals || m_frameOptions.ue3EngineMode)) {
      // 1: VS outputs NORMAL - use vertex-captured normals (they match the captured positions)
      const uint32_t normalOffset = offsetof(CapturedVertex, normal0);
      geoData.normalBuffer = RasterBuffer(slice, normalOffset, stride, VK_FORMAT_R32G32B32_SFLOAT);
      assert(geoData.normalBuffer.offset() % 4 == 0);
    } else if (isGpuSkinned && vsOutputsColor0) {
      // 2: GPU-skinned mesh, VS doesn't output NORMAL but has COLOR0 output
      // UE3 GpuSkinVertexFactory outputs the bone-transformed world-space tangent basis normal
      // through COLOR0 as (normal * 0.5 + 0.5), so we tell the vertex capture shader to decode COLOR0
      // as the normal source instead of using the bind-pose IA NORMAL
      vertexCaptureFlags |= kVertexCaptureFlag_NormalFromColor0;
      const uint32_t normalOffset = offsetof(CapturedVertex, normal0);
      geoData.normalBuffer = RasterBuffer(slice, normalOffset, stride, VK_FORMAT_R32G32B32_SFLOAT);
      assert(geoData.normalBuffer.offset() % 4 == 0);
      ONCE(Logger::info("[RTX-Compatibility] UE3 GPU-skinned mesh: capturing normal from VS COLOR0 output (skinned tangent basis)."));
    } else if (canUseBoneSkinnedNormalCapture) {
      // 2b: GPU-skinned mesh without VS NORMAL/COLOR0 output
      // reconstruct skinned normals in the vertex capture shader using BoneMatrices from VS constants
      vertexCaptureFlags |= kVertexCaptureFlag_NormalBoneSkinning;
      if (geoData.normalBuffer.vertexFormat() == VK_FORMAT_R8G8B8A8_UNORM)
        vertexCaptureFlags |= kVertexCaptureFlag_NormalInputEncodedUByte4;
      if (hasBlendIndices && geoData.blendIndicesBuffer.vertexFormat() == VK_FORMAT_R8G8B8A8_UNORM)
        vertexCaptureFlags |= kVertexCaptureFlag_BlendIndicesInputNormalized;
      if (hasBlendWeights && geoData.blendWeightBuffer.vertexFormat() == VK_FORMAT_R8G8B8A8_USCALED)
        vertexCaptureFlags |= kVertexCaptureFlag_BlendWeightsInputUnnormalized;

      const uint32_t normalOffset = offsetof(CapturedVertex, normal0);
      geoData.normalBuffer = RasterBuffer(slice, normalOffset, stride, VK_FORMAT_R32G32B32_SFLOAT);
      assert(geoData.normalBuffer.offset() % 4 == 0);
      ONCE(Logger::info(str::format(
        "[RTX-Compatibility] UE3 GPU-skinned mesh: reconstructing normals via BoneMatrices in vertex capture shader (boneBaseReg=c",
        ue3CtabInfo->boneMatricesRegisterIndex, ", boneRegCount=", ue3CtabInfo->boneMatricesRegisterCount,
        ", hasBlendWeights=", hasBlendWeights, ", hasBlendIndices=", hasBlendIndices, ").")));
    } else if (isGpuSkinned && normalFromDecl) {
      // 2c: GPU-skinned but no COLOR0 output and no usable bone CTAB metadata
      // keep bind-pose IA normals as smooth fallback, they're in the wrong orientation for animated poses (missing bone transform), but provide smooth per-vertex interpolation
      ONCE(Logger::info("[RTX-Compatibility] UE3 GPU-skinned mesh without COLOR0 output: using bind-pose IA normals as smooth fallback."));
    }
    // 3 : non-skinned, VS doesn't output NORMAL, just keep IA normals from processVertices they're in the same object space as the captured positions anyway..

    // Check if we should/can get colors. UE3 passes its particle colour through the vertex shader
    // as a TEXCOORD interpolant (ParticleSpriteVertexFactory.usf), which takes the place of COLOR0.
    const uint32_t capturedColorOutputRegister = m_ue3ParticleColorTexcoordIndex != UINT32_MAX
      ? FindVsTexcoordOutputRegister(vertexShader, m_ue3ParticleColorTexcoordIndex)
      : std::numeric_limits<uint32_t>::max();
    if (capturedColorOutputRegister != std::numeric_limits<uint32_t>::max() ||
        (BoundShaderHas(vertexShader, DxsoUsage::Color, false) && d3d9State().pixelShader.ptr() == nullptr)) {
      const uint32_t colorOffset = offsetof(CapturedVertex, color0);
      geoData.color0Buffer = RasterBuffer(slice, colorOffset, stride, VK_FORMAT_B8G8R8A8_UNORM);
      assert(geoData.color0Buffer.offset() % 4 == 0);
    }
    if (capturedColorOutputRegister != std::numeric_limits<uint32_t>::max()) {
      vertexCaptureFlags |= m_ue3ParticleColorCaptureFlags;
    }
    if (m_ue3ParticleColorTexcoordIndex != UINT32_MAX && m_frameOptions.ue3LogMaterialFades && vertexShader != nullptr) {
      static fast_unordered_set s_loggedColorCaptures;
      const XXH64_hash_t vsHash = vertexShader->GetBytecodeHash();
      if (s_loggedColorCaptures.insert(XXH3_64bits_withSeed(&m_ue3ParticleColorTexcoordIndex, sizeof(uint32_t), vsHash)).second) {
        Logger::info(str::format(
          "[RTX-Compatibility][UE3-Fade] Capture: vs=0x", std::hex, vsHash, std::dec, " particle colour TEXCOORD",
          m_ue3ParticleColorTexcoordIndex,
          capturedColorOutputRegister != std::numeric_limits<uint32_t>::max()
            ? str::format(" from o", capturedColorOutputRegister)
            : std::string(" is not a vertex shader output, so the particle colour stays white")));
      }
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
    data.flags = vertexCaptureFlags;
    data.boneMatricesBaseReg = 0;
    data.boneCount = 0;
    data.texcoordOutputRegister = capturedTexcoordOutputRegister;
    data.texcoordCompU = m_texcoordCompU & 0x3u;
    data.texcoordCompV = m_texcoordCompV & 0x3u;
    data.colorOutputRegister = capturedColorOutputRegister;
    if ((vertexCaptureFlags & kVertexCaptureFlag_NormalBoneSkinning) != 0 && ue3CtabInfo != nullptr) {
      data.boneMatricesBaseReg = ue3CtabInfo->boneMatricesRegisterIndex;
      data.boneCount = std::min(ue3CtabInfo->boneMatricesRegisterCount / 3u, 256u);
    }

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
        // coordinate (see resolveIaTexcoordAvoidingNonUvElements).
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

    const bool usesProgrammableVs = m_parent->UseProgrammableVS();
    const D3D9CommonShader* vertexShaderCommon =
      usesProgrammableVs && d3d9State().vertexShader.ptr() != nullptr
        ? d3d9State().vertexShader->GetCommonShader()
        : nullptr;

    const Ue3VsShaderCtabInfo* ue3CtabInfoPtr = nullptr;
    m_currentUe3CtabInfo.reset();
    m_currentUe3VsHashExclusions = nullptr;
    bool ue3CameraUsedTranspose = false;

    const bool needsUe3CtabInfo =
      isUe3Mode ||
      m_frameOptions.useVertexCapture;
    if (usesProgrammableVs && vertexShaderCommon != nullptr &&
        needsUe3CtabInfo) {
      auto parseCtabInfo = [&](const std::vector<uint8_t>& bytecode,
                               std::vector<Ue3VsConstantSymbol>* outSymbols) -> Ue3VsShaderCtabInfo {
        Ue3VsShaderCtabInfo info;
        info.initialized = true;

        try {
          if (bytecode.size() < sizeof(uint32_t) || (bytecode.size() % sizeof(uint32_t)) != 0)
            return info;

          const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
          const uint32_t headerToken = tokens[0];
          const uint32_t headerTypeMask = headerToken & 0xffff0000u;

          DxsoProgramType programType;
          if (headerTypeMask == 0xffff0000u)
            programType = DxsoProgramTypes::PixelShader;
          else if (headerTypeMask == 0xfffe0000u)
            programType = DxsoProgramTypes::VertexShader;
          else
            return info;

          const uint32_t majorVersion = (headerToken >> 8) & 0xffu;
          const uint32_t minorVersion = headerToken & 0xffu;
          DxsoProgramInfo programInfo { programType, minorVersion, majorVersion };

          DxsoDecodeContext decoder(programInfo);
          DxsoCodeIter iter(tokens + 1);

          while (decoder.decodeInstruction(iter)) {
            if (decoder.getCtabInfo().m_size != 0)
              break;
          }

          const DxsoCtab& ctab = decoder.getCtabInfo();
          if (ctab.m_size == 0 || ctab.m_constantData.empty())
            return info;

          auto lower = [](const std::string& s) {
            std::string out;
            out.reserve(s.size());
            for (const char c : s)
              out.push_back(char(std::tolower(static_cast<unsigned char>(c))));
            return out;
          };

          auto contains = [](const std::string& s, const char* needle) {
            return s.find(needle) != std::string::npos;
          };

          uint32_t inferredBoneMatricesRegisterIndex = 0;
          uint32_t inferredBoneMatricesRegisterCount = 0;

          if (outSymbols != nullptr) {
            outSymbols->reserve(ctab.m_constantData.size());
            for (const DxsoCtab::Constant& c : ctab.m_constantData) {
              // float constant registers only - the churn diagnostic compares fConsts
              if (c.registerSet != kD3dxRegisterSetFloat4 || c.registerCount == 0) {
                continue;
              }
              Ue3VsConstantSymbol symbol;
              symbol.registerIndex = c.registerIndex;
              symbol.registerCount = c.registerCount;
              symbol.name = c.name;
              outSymbols->push_back(std::move(symbol));
            }
          }

          for (const DxsoCtab::Constant& c : ctab.m_constantData) {
            const std::string name = lower(c.name);

            // ViewProjectionMatrix (4 registers)
            if (!info.hasViewProjectionMatrix && c.registerCount >= 4) {
              const bool looksLikeViewProj =
                contains(name, "viewprojectionmatrix") ||
                contains(name, "viewprojmatrix") ||
                contains(name, "view_projection_matrix") ||
                contains(name, "view_proj_matrix");
              const bool isPreviousViewProj =
                contains(name, "prevviewprojectionmatrix") ||
                contains(name, "prevviewprojmatrix") ||
                contains(name, "previousviewprojectionmatrix") ||
                contains(name, "previousviewprojmatrix") ||
                contains(name, "prev_view_projection_matrix") ||
                contains(name, "prev_view_proj_matrix");
              if (looksLikeViewProj && !isPreviousViewProj) {
                info.hasViewProjectionMatrix = true;
                info.viewProjectionMatrixRegisterIndex = c.registerIndex;
                info.viewProjectionMatrixRegisterCount = c.registerCount;
              }
            }

            // CameraPosition (1 register)
            if (!info.hasCameraPosition && c.registerCount >= 1) {
              const bool looksLikeCameraPosition =
                contains(name, "cameraposition") ||
                contains(name, "vieworigin") ||
                contains(name, "cameraworldpos") ||
                contains(name, "cameraworldposition") ||
                contains(name, "camerapos") ||
                contains(name, "eyeposition");
              const bool isPreviousCameraPosition =
                contains(name, "prevcameraposition") ||
                contains(name, "prevvieworigin") ||
                contains(name, "previouscameraposition") ||
                contains(name, "previousvieworigin") ||
                contains(name, "prevcameraworldposition") ||
                contains(name, "previouseyeposition");
              if (looksLikeCameraPosition && !isPreviousCameraPosition) {
                info.hasCameraPosition = true;
                info.cameraPositionRegisterIndex = c.registerIndex;
                info.cameraPositionRegisterCount = c.registerCount;
              }
            }

            // LocalToWorld (4 registers)
            if (!info.hasLocalToWorld && c.registerCount >= 4) {
              // prefer an exact match here but also accept nested names e.g. VertexFactory.LocalToWorld
              if (contains(name, "localtoworld") ||
                  contains(name, "local_to_world") ||
                  contains(name, "objecttoworld") ||
                  contains(name, "object_to_world")) {
                if (contains(name, "previouslocaltoworld") ||
                    contains(name, "prevlocaltoworld") ||
                    contains(name, "previous_local_to_world") ||
                    contains(name, "prev_local_to_world"))
                  continue;

                info.hasLocalToWorld = true;
                info.localToWorldRegisterIndex = c.registerIndex;
                info.localToWorldRegisterCount = c.registerCount;
              }
            }

            // WorldToLocal (3 registers, typically float3x3)
            if (!info.hasWorldToLocal && c.registerCount >= 3) {
              if (contains(name, "worldtolocal") ||
                  contains(name, "world_to_local") ||
                  contains(name, "objectinverseworld") ||
                  contains(name, "object_inverse_world")) {
                info.hasWorldToLocal = true;
                info.worldToLocalRegisterIndex = c.registerIndex;
                info.worldToLocalRegisterCount = c.registerCount;
              }
            }

            // BoneMatrices (GPU skinning: commonly N bones * 3 registers in UE3)
            if (!info.hasBoneMatrices && c.registerCount >= 3) {
              const bool explicitBoneName =
                contains(name, "bonematrices") ||
                contains(name, "bone_matrices") ||
                contains(name, "bonematrix") ||
                contains(name, "skinningmatrices") ||
                contains(name, "skinmatrices") ||
                contains(name, "matrixpalette") ||
                contains(name, "bonetransforms") ||
                contains(name, "bone_transforms") ||
                (contains(name, "bone") && (c.registerCount % 3u) == 0u);

              if (explicitBoneName) {
                info.hasBoneMatrices = true;
                info.boneMatricesRegisterIndex = c.registerIndex;
                info.boneMatricesRegisterCount = c.registerCount;
              } else if ((c.registerCount % 3u) == 0u) {
                // fallback for stripped/renamed symbols, in UE3 this is typically a large contiguous
                // c-register range (3 registers per bone) usually starting after c0..c4 camera constants
                const bool isConfirmedSkinVF =
                  m_currentUe3VertexFactory == Ue3VertexFactoryType::GPUSkin ||
                  m_currentUe3VertexFactory == Ue3VertexFactoryType::GPUSkinMorph;
                const uint32_t minRegCount = isConfirmedSkinVF ? 3u : 9u;
                const uint32_t minRegIndex = isConfirmedSkinVF ? 0u : 5u;
                if (c.registerCount >= minRegCount && c.registerIndex >= minRegIndex &&
                    c.registerCount > inferredBoneMatricesRegisterCount) {
                  inferredBoneMatricesRegisterIndex = c.registerIndex;
                  inferredBoneMatricesRegisterCount = c.registerCount;
                }
              }
            }

            // additional UE3 vertexfactory hints used by shader-path UV selection
            if (!info.hasDecalTransform &&
                (contains(name, "worldtodecal") ||
                 contains(name, "world_to_decal") ||
                 contains(name, "bonetodecal") ||
                 contains(name, "bone_to_decal"))) {
              info.hasDecalTransform = true;
            }
            if (!info.hasDecalLocation &&
                (contains(name, "decallocation") ||
                 contains(name, "decal_location"))) {
              info.hasDecalLocation = true;
            }
            if (!info.hasDecalOffset &&
                (contains(name, "decaloffset") ||
                 contains(name, "decal_offset"))) {
              info.hasDecalOffset = true;
            }
            if (!info.hasTextureCoordinateScaleBias &&
                (contains(name, "texturecoordinatescalebias") ||
                 contains(name, "texture_coordinate_scale_bias"))) {
              info.hasTextureCoordinateScaleBias = true;
            }
            if (!info.hasLightMapCoordinateScaleBias &&
                (contains(name, "lightmapcoordinatescalebias") ||
                 contains(name, "light_map_coordinate_scale_bias"))) {
              info.hasLightMapCoordinateScaleBias = true;
            }
            if (!info.hasShadowCoordinateScaleBias &&
                (contains(name, "shadowcoordinatescalebias") ||
                 contains(name, "shadow_coordinate_scale_bias"))) {
              info.hasShadowCoordinateScaleBias = true;
            }
            if (!info.hasViewToLocal &&
                (contains(name, "viewtolocal") ||
                 contains(name, "view_to_local"))) {
              info.hasViewToLocal = true;
            }
            if (!info.hasWindMatrices &&
                (contains(name, "windmatrices") ||
                 contains(name, "wind_matrices") ||
                 contains(name, "windmatrix") ||
                 contains(name, "wind_matrix"))) {
              info.hasWindMatrices = true;
            }
          }

          if (!info.hasBoneMatrices && inferredBoneMatricesRegisterCount >= 3u) {
            const bool isConfirmedSkinVF =
              m_currentUe3VertexFactory == Ue3VertexFactoryType::GPUSkin ||
              m_currentUe3VertexFactory == Ue3VertexFactoryType::GPUSkinMorph;
            if (isConfirmedSkinVF || inferredBoneMatricesRegisterCount >= 9u) {
              info.hasBoneMatrices = true;
              info.boneMatricesRegisterIndex = inferredBoneMatricesRegisterIndex;
              info.boneMatricesRegisterCount = inferredBoneMatricesRegisterCount;
            }
          }
        } catch (...) {
          return info;
        }

        return info;
      };

      const auto& bytecode = vertexShaderCommon->GetBytecode();
      const XXH64_hash_t shaderHash = vertexShaderCommon->GetBytecodeHash();

      m_activeDrawCallState.programmableVertexShaderBytecodeHash = shaderHash;

      if (shaderHash != 0) {
        auto it = m_ue3VsShaderCtabCache.find(shaderHash);
        if (it == m_ue3VsShaderCtabCache.end()) {
          std::vector<Ue3VsConstantSymbol> symbols;
          const Ue3VsShaderCtabInfo parsed = parseCtabInfo(bytecode, &symbols);
          m_ue3VsHashExclusionCache.emplace(shaderHash, buildUe3VsHashExclusions(parsed, &symbols));
          m_ue3VsShaderCtabCache.emplace(shaderHash, parsed);
          if (!symbols.empty()) {
            m_ue3VsConstantSymbols.emplace(shaderHash, std::move(symbols));
          }
          it = m_ue3VsShaderCtabCache.find(shaderHash);
        }

        {
          const auto exclusionIt = m_ue3VsHashExclusionCache.find(shaderHash);
          m_currentUe3VsHashExclusions =
            exclusionIt != m_ue3VsHashExclusionCache.end() ? &exclusionIt->second : nullptr;
        }

        if (it != m_ue3VsShaderCtabCache.end()) {
          ue3CtabInfoPtr = &it->second;
          m_currentUe3CtabInfo = *ue3CtabInfoPtr;
          if (isUe3Mode &&
              m_currentUe3VertexFactory == Ue3VertexFactoryType::Local &&
              (ue3CtabInfoPtr->hasDecalTransform ||
               ue3CtabInfoPtr->hasDecalLocation ||
               ue3CtabInfoPtr->hasDecalOffset)) {
            m_currentUe3VertexFactory = Ue3VertexFactoryType::LocalDecal;
          }
        }
      }
    }

    if (usesProgrammableVs && isUe3Mode) {
      uint32_t viewProjReg = kUe3VsrViewProjMatrixRegister;
      uint32_t viewOriginReg = kUe3VsrViewOriginRegister;

      if (ue3CtabInfoPtr != nullptr) {
        if (ue3CtabInfoPtr->hasViewProjectionMatrix)
          viewProjReg = ue3CtabInfoPtr->viewProjectionMatrixRegisterIndex;
        if (ue3CtabInfoPtr->hasCameraPosition)
          viewOriginReg = ue3CtabInfoPtr->cameraPositionRegisterIndex;
      }

      // cache by raw constant values to avoid repeated heavy extraction work per draw call
      struct Ue3CameraConstsKey {
        uint32_t viewProjReg;
        uint32_t viewOriginReg;
        Vector4 regs[5];
      };

      auto tryApplyFromConstants = [&](Matrix4& outWorldToView, Matrix4& outViewToProjection, bool& outUsedTranspose, float& outReconstructionError) -> bool {
        if (viewProjReg + 3 >= caps::MaxFloatConstantsSoftware || viewOriginReg >= caps::MaxFloatConstantsSoftware)
          return false;

        Ue3CameraConstsKey key {};
        key.viewProjReg = viewProjReg;
        key.viewOriginReg = viewOriginReg;
        key.regs[0] = d3d9State().vsConsts.fConsts[viewProjReg + 0];
        key.regs[1] = d3d9State().vsConsts.fConsts[viewProjReg + 1];
        key.regs[2] = d3d9State().vsConsts.fConsts[viewProjReg + 2];
        key.regs[3] = d3d9State().vsConsts.fConsts[viewProjReg + 3];
        key.regs[4] = d3d9State().vsConsts.fConsts[viewOriginReg];

        const XXH64_hash_t constantsHash = XXH3_64bits(&key, sizeof(key));

        for (const Ue3CameraConstantsCache& slot : m_ue3CameraConstantsCache) {
          if (slot.valid && slot.hash == constantsHash) {
            if (slot.extractionFailed) {
              return false;
            }
            outWorldToView = slot.worldToView;
            outViewToProjection = slot.viewToProjection;
            outUsedTranspose = slot.usedTranspose;
            outReconstructionError = slot.reconstructionError;
            return true;
          }
        }

        Matrix4 ue3WorldToView;
        Matrix4 ue3ViewToProjection;
        bool usedTranspose = false;
        float reconstructionError = 0.0f;
        const bool extracted = tryExtractUe3WorldToViewAndProjectionFromShaderConstants(
            d3d9State().vsConsts, viewProjReg, viewOriginReg, ue3WorldToView, ue3ViewToProjection, &usedTranspose, &reconstructionError);

        Ue3CameraConstantsCache& slot = m_ue3CameraConstantsCache[m_ue3CameraConstantsCacheNextSlot];
        m_ue3CameraConstantsCacheNextSlot = (m_ue3CameraConstantsCacheNextSlot + 1u) % kUe3CameraConstantsCacheSlots;
        slot.hash = constantsHash;
        slot.valid = true;
        slot.extractionFailed = !extracted;
        slot.usedTranspose = usedTranspose;
        slot.worldToView = ue3WorldToView;
        slot.viewToProjection = ue3ViewToProjection;
        slot.reconstructionError = reconstructionError;

        if (!extracted) {
          return false;
        }

        outWorldToView = ue3WorldToView;
        outViewToProjection = ue3ViewToProjection;
        outUsedTranspose = usedTranspose;
        outReconstructionError = reconstructionError;
        return true;
      };

      Matrix4 ue3WorldToView;
      Matrix4 ue3ViewToProjection;
      float ue3CameraReconstructionError = 0.0f;

      // Only draws whose CTAB explicitly names both camera constants may update the Main camera.
      // Fallback-register extractions can be light-space matrices from engine utility shaders
      // (e.g. shadow depth) that still reconstruct as a plausible camera.
      const bool ctabVerifiedCamera =
        ue3CtabInfoPtr != nullptr &&
        ue3CtabInfoPtr->hasViewProjectionMatrix &&
        ue3CtabInfoPtr->hasCameraPosition;

      // Full SceneCapture isolation for probes the viewport heuristic cannot see:
      // reflect/portal probes (e.g. Mirror's Edge scripted building window reflections)
      // render the world through a FMirrorMatrix-premultiplied view and an oblique
      // FClipProjectionMatrix near-plane clip, at viewports scaled to the parent view.
      // Geometry captured through such views is unusable - mirrored views reconstruct
      // reflected world positions, oblique projections are not decomposable - so these
      // draws are dropped outright. Restricted to CTAB-verified cameras: fallback
      // registers can hold arbitrary data that must not trigger capture classification.
      const bool ue3CaptureViewIsolation =
        isUe3Mode &&
        ctabVerifiedCamera &&
        isUe3WorldGeometryVertexFactory(m_currentUe3VertexFactory);

      auto classifySceneCaptureView = [&](const char* reason) {
        m_currentUe3PassType = Ue3PassType::SceneCapture;
        // Undo probe draws that armed the main-view gate before mirror/oblique detection.
        if (!m_ue3ForegroundDpgActive) {
          m_ue3SeenMainViewWorldDraw = false;
        }
        m_activeDrawCallState.allowMainCameraUpdate = false;
        m_activeDrawCallState.ue3PassDescription = describeUe3PassType(m_currentUe3PassType);
        logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, reason);
      };

      if (tryApplyFromConstants(ue3WorldToView, ue3ViewToProjection, ue3CameraUsedTranspose, ue3CameraReconstructionError)) {
        // Mirrored view detection: a reflection view premultiplies a mirror (householder)
        // matrix into the view, flipping the sign of the ViewProjection 3x3 determinant.
        // UE3's LH view (axis-swap permutation, det +1) and perspective projection keep the
        // main view's determinant positive. Checked on the raw registers because the
        // extraction reconstructs an orthonormal basis and washes the mirror out; the sign
        // is transpose-invariant so the upload convention does not matter.
        if (ue3CaptureViewIsolation) {
          const Vector4& vpRow0 = d3d9State().vsConsts.fConsts[viewProjReg + 0];
          const Vector4& vpRow1 = d3d9State().vsConsts.fConsts[viewProjReg + 1];
          const Vector4& vpRow2 = d3d9State().vsConsts.fConsts[viewProjReg + 2];
          const float vpDet3 =
            vpRow0.x * (vpRow1.y * vpRow2.z - vpRow1.z * vpRow2.y) -
            vpRow0.y * (vpRow1.x * vpRow2.z - vpRow1.z * vpRow2.x) +
            vpRow0.z * (vpRow1.x * vpRow2.y - vpRow1.y * vpRow2.x);
          if (std::isfinite(vpDet3) && vpDet3 < 0.0f) {
            ONCE(Logger::info("[RTX-Compatibility-Info] Ignored UE3 scene capture draw (mirrored view-projection, e.g. reflection probe)."));
            classifySceneCaptureView("scene capture mirrored view");
            return false;
          }
        }
        ONCE(Logger::info(str::format("[RTX-Compatibility] UE3 camera matrices extracted from shader constants (viewProjReg=c",
                                      viewProjReg, "..c", viewProjReg + 3, ", viewOriginReg=c", viewOriginReg, ").")));
        if (m_frameOptions.ue3LogCapturePrecision && Logger::logLevel() <= LogLevel::Debug) {
          ONCE(Logger::debug(str::format(
            "[RTX-Compatibility][UE3-Capture] camera matrix reconstruction error=",
            ue3CameraReconstructionError, ", usedTranspose=", ue3CameraUsedTranspose)));
        }
        transformData.worldToView = ue3WorldToView;
        transformData.viewToProjection = ue3ViewToProjection;

        if (isUe3Mode && !ctabVerifiedCamera) {
          m_activeDrawCallState.allowMainCameraUpdate = false;
        }

        // Non-main-view-sized world draws (below half and/or wrong aspect) must not steer Main.
        if (isUe3Mode &&
            isUe3WorldGeometryVertexFactory(m_currentUe3VertexFactory) &&
            m_activePresentParams.has_value() &&
            m_activeDrawCallState.allowMainCameraUpdate) {
          const D3DVIEWPORT9& vp = d3d9State().viewport;
          if (!ue3ViewportIsMainViewSized(
                vp.Width, vp.Height,
                m_activePresentParams->BackBufferWidth,
                m_activePresentParams->BackBufferHeight)) {
            m_activeDrawCallState.allowMainCameraUpdate = false;
          }
        }

        // Once per vertex shader, so info level stays low-volume
        {
          static fast_unordered_set s_loggedCameraSourceVsHashes;
          const XXH64_hash_t vsHash = m_activeDrawCallState.programmableVertexShaderBytecodeHash;
          if (vsHash != 0 && s_loggedCameraSourceVsHashes.insert(vsHash).second) {
            Logger::info(str::format(
              "[RTX-Compatibility][UE3] camera constants source for vsHash=0x", std::hex, vsHash, std::dec,
              ": ", ctabVerifiedCamera ? "CTAB-verified" : "fallback registers",
              " (ctabViewProj=", (ue3CtabInfoPtr != nullptr && ue3CtabInfoPtr->hasViewProjectionMatrix) ? "yes" : "no",
              ", ctabCameraPos=", (ue3CtabInfoPtr != nullptr && ue3CtabInfoPtr->hasCameraPosition) ? "yes" : "no",
              ", viewProjReg=c", viewProjReg, ", viewOriginReg=c", viewOriginReg,
              ", vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
              ", allowMainCameraUpdate=", m_activeDrawCallState.allowMainCameraUpdate ? "true" : "false", ")"));
          }
        }
      } else {
        // A CTAB-verified world-geometry draw whose declared ViewProjectionMatrix fails
        // plausibility extraction is rendering through a view Remix cannot use. In practice
        // these are SceneCapture reflect/portal probes: FClipProjectionMatrix skews the near
        // plane onto the mirror/portal plane, which the extraction rejects as shear, and
        // FMirrorMatrix reflects the view. The main view always extracts, so nothing
        // legitimate is lost - and geometry processed with the stale/identity transforms
        // this branch would otherwise fall back to reconstructs as corrupted positions.
        if (ue3CaptureViewIsolation) {
          ONCE(Logger::info("[RTX-Compatibility-Info] Ignored UE3 scene capture draw (declared camera failed extraction, e.g. reflection/portal probe oblique projection)."));
          classifySceneCaptureView("scene capture undecomposable view");
          return false;
        }

        if (Logger::logLevel() <= LogLevel::Debug &&
            viewProjReg + 3 < caps::MaxFloatConstantsSoftware && viewOriginReg < caps::MaxFloatConstantsSoftware) {
          const Vector4 c0 = d3d9State().vsConsts.fConsts[viewProjReg + 0];
          const Vector4 c1 = d3d9State().vsConsts.fConsts[viewProjReg + 1];
          const Vector4 c2 = d3d9State().vsConsts.fConsts[viewProjReg + 2];
          const Vector4 c3 = d3d9State().vsConsts.fConsts[viewProjReg + 3];
          const Vector4 cam = d3d9State().vsConsts.fConsts[viewOriginReg];

          ONCE(Logger::debug(str::format(
            "[RTX-Compatibility] UE3 camera extraction failed (viewProjReg=c", viewProjReg, "..c", viewProjReg + 3,
            ", viewOriginReg=c", viewOriginReg, "). "
            "c0={", c0.x, ", ", c0.y, ", ", c0.z, ", ", c0.w, "} "
            "c1={", c1.x, ", ", c1.y, ", ", c1.z, ", ", c1.w, "} "
            "c2={", c2.x, ", ", c2.y, ", ", c2.z, ", ", c2.w, "} "
            "c3={", c3.x, ", ", c3.y, ", ", c3.z, ", ", c3.w, "} "
            "cam={", cam.x, ", ", cam.y, ", ", cam.z, ", ", cam.w, "}")));
        } else {
          ONCE(Logger::debug(str::format(
            "[RTX-Compatibility] UE3 camera extraction failed (out of bounds registers: viewProjReg=c", viewProjReg,
            ", viewOriginReg=c", viewOriginReg, ").")));
        }
      }
    }

    if (usesProgrammableVs && isUe3Mode && ue3CtabInfoPtr != nullptr) {
      const Ue3VsShaderCtabInfo& ctabInfo = *ue3CtabInfoPtr;

      if (ctabInfo.hasLocalToWorld) {
        const uint32_t reg = ctabInfo.localToWorldRegisterIndex;
        if (reg + 3 < caps::MaxFloatConstantsSoftware) {
          const uint32_t w2lReg = ctabInfo.worldToLocalRegisterIndex;
          const bool hasWorldToLocal = ctabInfo.hasWorldToLocal && w2lReg + 2 < caps::MaxFloatConstantsSoftware;

          // Memo lookup: every input to the transpose/affinity/inverse disambiguation
          // below (register contents and the camera transpose convention tiebreaker) is
          // folded into the key, so a hit returns exactly what the computation would
          // produce. Static placements re-upload identical matrices every frame, making
          // this a per-draw matrix-inverse saving.
          XXH64_hash_t o2wKeyHash = XXH3_64bits(&d3d9State().vsConsts.fConsts[reg], 4 * sizeof(Vector4));
          if (hasWorldToLocal) {
            o2wKeyHash = XXH3_64bits_withSeed(&d3d9State().vsConsts.fConsts[w2lReg], 3 * sizeof(Vector4), o2wKeyHash);
          }
          const uint32_t o2wKeyFlags = (hasWorldToLocal ? 1u : 0u) | (ue3CameraUsedTranspose ? 2u : 0u);
          o2wKeyHash = XXH3_64bits_withSeed(&o2wKeyFlags, sizeof(o2wKeyFlags), o2wKeyHash);

          const auto o2wIt = m_ue3ObjectToWorldCache.find(o2wKeyHash);
          if (o2wIt != m_ue3ObjectToWorldCache.end()) {
            transformData.objectToWorld = o2wIt->second;
          } else {
            const Matrix4 localToWorldRaw = [&] {
              Matrix4 m;
              m[0] = d3d9State().vsConsts.fConsts[reg + 0];
              m[1] = d3d9State().vsConsts.fConsts[reg + 1];
              m[2] = d3d9State().vsConsts.fConsts[reg + 2];
              m[3] = d3d9State().vsConsts.fConsts[reg + 3];
              return m;
            }();

            const Matrix4 localToWorldTransposed = transpose(localToWorldRaw);

            auto isAffineColumnVector = [](const Matrix4& m) {
              constexpr float kEps = 1e-3f;
              return std::abs(m[0].w) < kEps &&
                     std::abs(m[1].w) < kEps &&
                     std::abs(m[2].w) < kEps &&
                     std::abs(m[3].w - 1.0f) < kEps;
            };

            const bool rawAffine = isAffineColumnVector(localToWorldRaw);
            const bool transAffine = isAffineColumnVector(localToWorldTransposed);

            // optinally use WorldToLocal (if present) to disambiguate transpose/packing
            Matrix4 worldToLocalRaw;
            Matrix4 worldToLocalTransposed;
            if (hasWorldToLocal) {
              const Vector4 c0 = d3d9State().vsConsts.fConsts[w2lReg + 0];
              const Vector4 c1 = d3d9State().vsConsts.fConsts[w2lReg + 1];
              const Vector4 c2 = d3d9State().vsConsts.fConsts[w2lReg + 2];

              worldToLocalRaw = Matrix4();
              worldToLocalRaw[0] = Vector4(c0.x, c0.y, c0.z, 0.0f);
              worldToLocalRaw[1] = Vector4(c1.x, c1.y, c1.z, 0.0f);
              worldToLocalRaw[2] = Vector4(c2.x, c2.y, c2.z, 0.0f);
              worldToLocalRaw[3] = Vector4(0.0f, 0.0f, 0.0f, 1.0f);
              worldToLocalTransposed = transpose(worldToLocalRaw);
            }

            auto l1Error3x3 = [](const Matrix4& a, const Matrix4& b) {
              float err = 0.0f;
              for (uint32_t c = 0; c < 3; c++) {
                for (uint32_t r = 0; r < 3; r++) {
                  err += std::abs(a[c][r] - b[c][r]);
                }
              }
              return err;
            };

            Matrix4 localToWorld = localToWorldRaw;
            if (hasWorldToLocal && rawAffine && transAffine) {
              // both candidates look affine, so we choose the one whose inverse best matches the provided WorldToLocal basis
              const Matrix4 invRaw = inverseAffine(localToWorldRaw);
              const Matrix4 invTrans = inverseAffine(localToWorldTransposed);

              float bestErr = std::numeric_limits<float>::infinity();
              bool bestIsTransposed = false;

              const float errRaw0 = l1Error3x3(invRaw, worldToLocalRaw);
              const float errRaw1 = l1Error3x3(invRaw, worldToLocalTransposed);
              const float errTrans0 = l1Error3x3(invTrans, worldToLocalRaw);
              const float errTrans1 = l1Error3x3(invTrans, worldToLocalTransposed);

              bestErr = errRaw0;
              bestIsTransposed = false;
              if (errRaw1 < bestErr) { bestErr = errRaw1; bestIsTransposed = false; }
              if (errTrans0 < bestErr) { bestErr = errTrans0; bestIsTransposed = true; }
              if (errTrans1 < bestErr) { bestErr = errTrans1; bestIsTransposed = true; }

              constexpr float kMaxWorldToLocalMatchError = 0.25f;
              if (std::isfinite(bestErr) && bestErr <= kMaxWorldToLocalMatchError) {
                localToWorld = bestIsTransposed ? localToWorldTransposed : localToWorldRaw;
              } else {
                localToWorld = ue3CameraUsedTranspose ? localToWorldTransposed : localToWorldRaw;
              }
            } else if (rawAffine && transAffine) {
              localToWorld = ue3CameraUsedTranspose ? localToWorldTransposed : localToWorldRaw;
            } else if (!rawAffine && transAffine) {
              localToWorld = localToWorldTransposed;
            } else {
              localToWorld = localToWorldRaw;
            }

            if (m_ue3ObjectToWorldCache.size() >= kUe3ObjectToWorldCacheMaxEntries) {
              m_ue3ObjectToWorldCache.clear();
            }
            m_ue3ObjectToWorldCache.emplace(o2wKeyHash, localToWorld);

            transformData.objectToWorld = localToWorld;
          }

          ONCE(Logger::info("[RTX-Compatibility] UE3 LocalToWorld extracted from vertex shader constants (CTAB)"));
        }
      }
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

    // translucency - UE3 draws two sided translucent meshes as backface then frontface
    // passes with the same shader/textures but inverted cull modes; detect and skip the
    // second pass to avoid duplicate geometry. The match must be strict: UE3 sorts
    // translucent prims back-to-front per camera, so consecutive draws of *different*
    // meshes or instances sharing one material are common, and a loose match falsely
    // skips them (visibility flicker that follows the camera)
    if (m_frameOptions.ue3EngineMode && usesProgrammableVs && d3d9State().vertexShader.ptr() != nullptr &&
        d3d9State().pixelShader.ptr() != nullptr) {
      const XXH64_hash_t vsHash = d3d9State().vertexShader->GetCommonShader()->GetBytecodeHash();
      const XXH64_hash_t psHash = d3d9State().pixelShader->GetCommonShader()->GetBytecodeHash();
      const XXH64_hash_t vsPsHash = vsHash ^ (psHash * 0x9E3779B97F4A7C15ull);

      XXH64_hash_t boundTextureHash = 0;
      {
        const BoundTextureSnapshot& boundTextures = ensureBoundTextureSnapshot();
        for (const uint32_t s : bit::BitMask(boundTextures.mask & 0xFu)) {
          if (boundTextures.entries[s].hasImage)
            boundTextureHash ^= boundTextures.entries[s].imageHash;
        }
      }

      // identity of the exact geometry this draw references (buffers, ranges, topology)
      struct DrawGeometryIdentity {
        const void* pIndexBuffer;
        const void* pVertexBuffer0;
        const void* pVertexDecl;
        uint32_t vb0Offset;
        uint32_t vb0Stride;
        int32_t baseVertexIndex;
        uint32_t minVertexIndex;
        uint32_t numVertices;
        uint32_t startIndex;
        uint32_t primitiveCount;
        uint32_t primitiveType;
        uint32_t indexed;
        uint32_t reserved;
      };
      static_assert(sizeof(DrawGeometryIdentity) == 3 * sizeof(void*) + 10 * sizeof(uint32_t),
                    "DrawGeometryIdentity must have no implicit padding (it is hashed by memory).");
      const DrawGeometryIdentity geometryIdentity = {
        d3d9State().indices.ptr(),
        d3d9State().vertexBuffers[0].vertexBuffer.ptr(),
        d3d9State().vertexDecl.ptr(),
        d3d9State().vertexBuffers[0].offset,
        d3d9State().vertexBuffers[0].stride,
        drawContext.BaseVertexIndex,
        drawContext.MinVertexIndex,
        drawContext.NumVertices,
        drawContext.StartIndex,
        drawContext.PrimitiveCount,
        uint32_t(drawContext.PrimitiveType),
        uint32_t(drawContext.Indexed),
        0u,
      };
      XXH64_hash_t geometryIdentityHash = XXH3_64bits(&geometryIdentity, sizeof(geometryIdentity));

      // A genuine second cull pass re-issues the same instance with identical transform
      // constants; different placements of a shared mesh never match once those are folded
      // in. Only computed for blended draws - the skip condition requires blending anyway
      if (d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] != FALSE) {
        geometryIdentityHash = mixUe3InstanceTransformConstants(geometryIdentityHash);
      }

      const DWORD cullMode = d3d9State().renderStates[D3DRS_CULLMODE];
      const bool cullInverted =
        (cullMode == D3DCULL_CW && m_prevDrawCullMode == D3DCULL_CCW) ||
        (cullMode == D3DCULL_CCW && m_prevDrawCullMode == D3DCULL_CW);
      const bool isSecondTwoSidedPass =
        vsPsHash != 0 &&
        vsPsHash == m_prevDrawVsPsHash &&
        boundTextureHash == m_prevDrawTextureHash &&
        geometryIdentityHash == m_prevDrawGeometryHash &&
        cullInverted &&
        d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] != FALSE;

      m_prevDrawVsPsHash = vsPsHash;
      m_prevDrawTextureHash = boundTextureHash;
      m_prevDrawGeometryHash = geometryIdentityHash;
      m_prevDrawCullMode = cullMode;

      if (isSecondTwoSidedPass) {
        ONCE(Logger::info("[RTX-Compatibility-Info] Skipped UE3 two-pass translucent draw (second cull-mode pass)."));
        return false;
      }
    } else {
      m_prevDrawVsPsHash = 0;
      m_prevDrawTextureHash = 0;
      m_prevDrawGeometryHash = 0;
      m_prevDrawCullMode = 0;
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

    // Deferred-UI tag decision, evaluated at most once per draw: consumed by both the
    // depth-test-disabled translucency skip directly below and the deferred-overlay
    // branch further down.
    XXH64_hash_t deferredUiMatchedTextureHash = 0;
    int deferredUiTagState = -1;
    auto isDeferredUiTagged = [&]() {
      if (deferredUiTagState < 0) {
        deferredUiTagState = isDeferredUiTaggedDraw(&deferredUiMatchedTextureHash) ? 1 : 0;
      }
      return deferredUiTagState == 1;
    };
    // A matched hash of zero means the pixel shader tag matched, which the option documents as
    // explicit intent - unlike a texture tag, which a shared texture can trigger on any draw.
    // The emptiness test keeps post-process draws from building a bound-texture snapshot just
    // to discover that no pixel shader tag exists to find.
    auto isDeferredUiPixelShaderTagged = [&]() {
      return !m_frameOptions.deferredUiPixelShaders->empty() &&
             isDeferredUiTagged() &&
             deferredUiMatchedTextureHash == kEmptyHash;
    };

    // UE3 depth test disabled translucency -  NeedsDepthTestDisabled materials, fog volume composites,
    // and fullscreen overlays use alpha blend + depth test off + depth write off
    // exclude UI tagged draws since they also match this pattern but need rasterisation with RTX injection,
    // and deferred-UI tagged draws which need capture for post-injection replay
    if (m_frameOptions.ue3EngineMode &&
        d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] &&
        (d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_FALSE ||
         d3d9State().renderStates[D3DRS_ZFUNC] == D3DCMP_ALWAYS) &&
        d3d9State().renderStates[D3DRS_ZWRITEENABLE] == FALSE &&
        !checkBoundTextureCategory(*m_frameOptions.uiTextures) &&
        !isDeferredUiTagged()) {
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

    // UE3 depth prepass - position only vertex declarations have no texcoords/colours
    // the same geometry will be drawn again in the base pass with full material
    if (m_frameOptions.ue3EngineMode &&
        m_currentUe3VertexFactory == Ue3VertexFactoryType::PositionOnly) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped UE3 depth prepass draw (position-only vertex declaration)."));
      return { RtxGeometryStatus::Ignored, false };
    }

    // Ensure present parameters for the swapchain have been cached
    // Note: This assumes that ResetSwapChain has been called at some point before this call, typically done after creating a swapchain.
    assert(m_activePresentParams.has_value());

    m_currentUe3PassType = classifyUe3Pass(drawContext);

    // Capture TdToneMapping state before the draw is ignored below.
    if (m_currentUe3PassType == Ue3PassType::FullscreenPostProcess) {
      maybeCaptureUe3ToneMapState();
    }

    switch (m_currentUe3PassType) {
    case Ue3PassType::DepthPrepass:
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "position-only depth prepass");
      return { RtxGeometryStatus::Ignored, false };
    case Ue3PassType::ShadowDepth:
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "shadow depth render target");
      return { RtxGeometryStatus::Ignored, false };
    case Ue3PassType::Velocity:
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "native velocity helper pass");
      return { RtxGeometryStatus::Ignored, false };
    case Ue3PassType::Lighting:
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "native UE3 lighting pass");
      return { RtxGeometryStatus::Ignored, false };
    case Ue3PassType::ModulatedShadowProjection:
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "native modulated shadow projection");
      return { RtxGeometryStatus::Ignored, false };
    case Ue3PassType::SceneCapture:
      if (!m_ue3ForegroundDpgActive) {
        m_ue3SeenMainViewWorldDraw = false;
      }
      ONCE(Logger::info("[RTX-Compatibility-Info] Ignored UE3 scene capture offscreen view draw (world geometry, probe viewport)."));
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "scene capture offscreen view");
      return { RtxGeometryStatus::Ignored, false };
    case Ue3PassType::FullscreenPostProcess:
    case Ue3PassType::FogOrDistortion:
      // Ignore before the non-primary RT → Rasterized fallback; DoF gather/blur/blend
      // target FilterColor/SceneColor and would otherwise still execute. Only a deferred-UI
      // pixel shader tag outranks this: these passes sample the scene-colour target, so a
      // texture tag on it matches every one of them and would replay DoF over the frame.
      if (!isDeferredUiPixelShaderTagged()) {
        logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "native UE3 screen-space contribution pass");
        return { RtxGeometryStatus::Ignored, false };
      }
      break;
    case Ue3PassType::UiComposite:
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Rasterized, "UI composite");
      return { RtxGeometryStatus::Rasterized, true };
    case Ue3PassType::VideoCinematic:
      trackUe3MovieTextureRenderTarget("video cinematic/decode");
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Rasterized, "video/cinematic pass");
      return { RtxGeometryStatus::Rasterized, false };
    case Ue3PassType::VideoSurface:
      trackUe3MovieTextureRenderTarget("video surface/decode");
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Rasterized, "video texture surface/decode pass");
      return { RtxGeometryStatus::Rasterized, false };
    default:
      break;
    }

    // Arm the foreground-DPG gate only after a true main-view-sized world draw.
    if (m_frameOptions.ue3EngineMode &&
        !m_ue3SeenMainViewWorldDraw &&
        m_currentUe3PassType == Ue3PassType::Material &&
        isUe3WorldGeometryVertexFactory(m_currentUe3VertexFactory) &&
        m_activePresentParams.has_value()) {
      const D3DVIEWPORT9& vp = d3d9State().viewport;
      const uint32_t bbW = m_activePresentParams->BackBufferWidth;
      const uint32_t bbH = m_activePresentParams->BackBufferHeight;
      if (ue3ViewportIsMainViewSized(vp.Width, vp.Height, bbW, bbH)) {
        m_ue3SeenMainViewWorldDraw = true;
      }
    }

    // Deferred UI overlays (rtx.deferredUiTextures / rtx.d3d9.deferredUiPixelShaders, e.g. UE3
    // MaterialEffect fullscreen fades): rasterized on top of the ray-traced image WITHOUT
    // triggering RTX injection - the draw is captured and replayed after injection fires later
    // in the frame. Placed after the pass switch above, so UI composites, video passes and the
    // screen-space contribution passes can never be deferred by a texture tag (only a pixel
    // shader tag reaches here from those), and before the fullscreen-composite filter below so
    // tagging wins over that. World geometry and depth-writing draws are never deferred even
    // when tagged: shared textures (e.g. a scene-color render target sampled by translucent
    // meshes) must not pull geometry out of the ray-traced scene.
    if (!m_frameOptions.deferredUiTextures->empty() || !m_frameOptions.deferredUiPixelShaders->empty()) {
      const XXH64_hash_t& matchedTextureHash = deferredUiMatchedTextureHash;

      if (isDeferredUiTagged()) {
        const bool matchedByPixelShaderTag = isDeferredUiPixelShaderTagged();
        const bool zWriteEnabled = d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE;
        const bool isWorldGeometryVertexFactory = isUe3WorldGeometryVertexFactory(m_currentUe3VertexFactory);

        // Engine post-process/composite shaders (gamma-correction scene copy, tone mapping,
        // motion blur, distortion, fog, depth of field, bloom) legitimately sample the scene
        // render target but must never be deferred: replaying e.g. the gamma copy over the
        // ray-traced image uniformly brightens the whole screen and, being opaque and later in
        // the frame, overwrites the real overlay effects; replaying a DoF gather paints a flat
        // rectangle over it. This is the only guard for a texture tag once the draw classifies
        // as anything but a screen-space contribution pass, so it matches the same DoF/bloom
        // signatures the pass classifier uses. An explicit pixel shader tag still defers.
        bool isEnginePostProcessShader = false;
        if (!matchedByPixelShaderTag && m_parent->UseProgrammablePS() && d3d9State().pixelShader != nullptr) {
          const Ue3ShaderFeatureInfo psInfo = getUe3ShaderFeatureInfo(d3d9State().pixelShader->GetCommonShader());
          isEnginePostProcessShader = psInfo.hasGammaConstants ||
                                      psInfo.hasToneMapConstants ||
                                      psInfo.hasExposureOrToneSampler ||
                                      psInfo.hasMotionBlurConstants ||
                                      psInfo.hasVelocitySampler ||
                                      psInfo.hasDistortionSampler ||
                                      psInfo.hasFogConstants ||
                                      psInfo.hasHazeConstants ||
                                      psInfo.looksLikeDofAndBloomPostProcess();
        }

        // Fullscreen overlay tiles (UE3 MaterialEffect quads via FTileRenderer) use a
        // Local-style vertex declaration (position/tangents/color/uv) and would be caught by
        // the world-geometry guard. A tiny primitive count with depth testing disabled
        // distinguishes them from real world geometry: even small world quads (glass panes,
        // monitors) depth-test against the scene, overlay tiles never do.
        const bool depthTestDisabled = d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_FALSE ||
                                       d3d9State().renderStates[D3DRS_ZFUNC] == D3DCMP_ALWAYS;
        const bool looksLikeOverlayTile = drawContext.PrimitiveCount <= 4 && depthTestDisabled && !zWriteEnabled;

        const bool eligible = !zWriteEnabled && !isEnginePostProcessShader &&
                              (!isWorldGeometryVertexFactory || looksLikeOverlayTile);
        const char* refusalReason = isEnginePostProcessShader
                                    ? "engine post-process shader"
                                    : "world geometry or depth write";

        // One-shot diagnostics per (pixel shader, decision): prints the stable pixel shader
        // hash so tags on unstable render-target textures can be moved to
        // rtx.d3d9.deferredUiPixelShaders.
        const XXH64_hash_t psHash = (m_parent->UseProgrammablePS() && d3d9State().pixelShader != nullptr)
                                    ? d3d9State().pixelShader->GetCommonShader()->GetBytecodeHash() : 0;
        const XXH64_hash_t vsHash = (m_parent->UseProgrammableVS() && d3d9State().vertexShader != nullptr)
                                    ? d3d9State().vertexShader->GetCommonShader()->GetBytecodeHash() : 0;
        const XXH64_hash_t logKey = psHash ^ (eligible ? 0xD1B54A32D192ED03ull
                                                       : (isEnginePostProcessShader ? 0x2545F4914F6CDD1Dull
                                                                                    : 0x9E3779B97F4A7C15ull));
        if (m_deferredUiLoggedDecisions.insert(logKey).second) {
          const std::string matchedDescription = matchedTextureHash != 0
            ? str::format(" matchedTexture=0x", std::hex, matchedTextureHash, std::dec)
            : std::string(" matchedBy=pixelShaderTag");

          Logger::info(str::format(
            "[RTX-DeferredUI] ",
            eligible ? std::string("Deferring overlay draw")
                     : str::format("Tagged draw NOT deferred (", refusalReason, ")"),
            ": ps=0x", std::hex, psHash,
            " vs=0x", vsHash, std::dec,
            " vertexFactory=", describeUe3VertexFactory(m_currentUe3VertexFactory),
            " pass=", describeUe3PassType(m_currentUe3PassType),
            " prims=", drawContext.PrimitiveCount,
            " ztest=", depthTestDisabled ? 0 : 1,
            " zwrite=", zWriteEnabled ? 1 : 0,
            " target=", getCurrentRenderTargetFormat(),
            " domain=", isSceneLinearRenderTarget() ? "sceneLinear" : "display",
            matchedDescription));
        }

        if (eligible) {
          logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Rasterized, "deferred UI overlay");
          return { RtxGeometryStatus::Rasterized, false, true };
        }

        // Screen-space contribution passes only reached this branch through a pixel shader tag;
        // a refusal here restores the pass switch's decision rather than promoting a DoF/fog
        // draw to normal classification.
        if (m_currentUe3PassType == Ue3PassType::FullscreenPostProcess ||
            m_currentUe3PassType == Ue3PassType::FogOrDistortion) {
          logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "native UE3 screen-space contribution pass");
          return { RtxGeometryStatus::Ignored, false };
        }
        // Other ineligible tagged draws fall through to normal classification - never suppressed.
      }
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

    // UE3 shadow depth pass - draws to small square render targets that are used as shadow maps
    if (m_frameOptions.ue3EngineMode && m_activePresentParams.has_value()) {
      const auto& rtExt = d3d9State().renderTargets[kRenderTargetIndex]->GetSurfaceExtent();
      const uint32_t bbW = m_activePresentParams->BackBufferWidth;
      const bool isSmallSquare = rtExt.width == rtExt.height &&
                                 rtExt.width <= 2048 &&
                                 rtExt.width < bbW / 2;
      const bool hasDepthWrite = d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE;
      if (isSmallSquare && hasDepthWrite) {
        ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Skipped UE3 shadow depth pass (",
                                       rtExt.width, "x", rtExt.height, ").")));
        return { RtxGeometryStatus::Ignored, false };
      }
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
        // debugging, todo remove later
        if (Logger::logLevel() <= LogLevel::Debug) {
          if (D3D9CommonTexture* rtTex = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture()) {
            if (rtTex->GetImage() != nullptr) {
              const XXH64_hash_t rtDescHash = rtTex->GetImage()->getDescriptorHash();
              if (s_loggedNonPrimaryRtDescHashes.insert(rtDescHash).second) {
                const auto* rtDesc = rtTex->Desc();
                Logger::debug(str::format(
                  "[RTX-Compatibility] Non-primary RT0 encountered: ",
                  rtDesc->Width, "x", rtDesc->Height,
                  " (backbuffer ", m_activePresentParams->BackBufferWidth, "x", m_activePresentParams->BackBufferHeight, "), ",
                  "rtDescHash=0x", std::hex, rtDescHash,
                  " resolutionAgnosticDescHash=0x", rtTex->GetImage()->getResolutionAgnosticDescriptorHash(), std::dec,
                  ". If this RT contains the main scene, add either hash to rtx.raytracedRenderTargetTextures "
                  "(the resolution-agnostic one survives resolution changes)."));
              }
            }
          }
        }

        ONCE(Logger::info("[RTX-Compatibility-Info] Found a draw call to a non-primary, non-raytraced render target. Falling back to rasterization"));
        return { RtxGeometryStatus::Rasterized, false };
      }
    }

    if (const uint32_t rtSamplerMask = m_parent->GetActiveRTTextures()) {
      const bool depthEnabled  = d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_TRUE;
      const bool zWriteEnabled = d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE;
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
        return { RtxGeometryStatus::Rasterized, false };
      }
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

    // classify UE3 vertex factory early so makeDrawCallType can use it for pass filtering
    m_currentUe3VertexFactory = Ue3VertexFactoryType::Unknown;
    m_currentUe3PassType = Ue3PassType::Unknown;
    if (m_frameOptions.ue3EngineMode && d3d9State().vertexDecl != nullptr) {
      const auto& elements = d3d9State().vertexDecl->GetElements();
      XXH64_hash_t declKey = XXH3_64bits(elements.data(), elements.size() * sizeof(D3DVERTEXELEMENT9));
      auto it = m_ue3VertexFactoryCache.find(declKey);
      if (it != m_ue3VertexFactoryCache.end()) {
        m_currentUe3VertexFactory = it->second;
      } else {
        m_currentUe3VertexFactory = classifyUe3VertexFactory(elements);
        m_ue3VertexFactoryCache.emplace(declKey, m_currentUe3VertexFactory);
      }
    }

    // Hardware instancing state is needed before the capture source is resolved: a per-vertex
    // capture buffer cannot represent a draw that runs its vertices once per instance. Kept behind
    // ue3EngineMode with the rest of the UE3 behaviour - recovering placements from a stream relies
    // on knowing the engine's instance layout, and without that there is nothing better to do with
    // an instanced draw than what Remix already did.
    // Leaving this at its default (no instancing seen) is what makes ue3DecomposeInstancedDraws a
    // true bypass: every decision downstream keys off instanceCount, so with the option off an
    // instanced draw is handled exactly as it was before decomposition existed.
    m_currentUe3Instancing = Ue3InstancingInfo();
    m_ue3DecomposedInstances.clear();
    m_ue3DecomposedBatchKey = kEmptyHash;
    // m_activeDrawCallState is reused across draws, so a previous draw's per-instance identity must
    // not leak into an ordinary one and detach it from the transform its identity depends on.
    m_activeDrawCallState.decomposedInstanceId = kEmptyHash;
    if (m_frameOptions.ue3EngineMode && m_frameOptions.ue3DecomposeInstancedDraws &&
        d3d9State().vertexDecl != nullptr) {
      m_currentUe3Instancing = resolveUe3Instancing(
        d3d9State().vertexDecl->GetElements(), d3d9State().streamFreq, m_parent->GetInstanceCount());
    }

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

    // Recover the placements UE3 hid in the instance-data stream. Done before the capture source is
    // resolved so that path can prefer input-assembler positions once the placements are in hand:
    // object-space positions are only usable if there is a transform to place them with.
    m_ue3InstanceTransformReadFailure = nullptr;
    if (m_currentUe3Instancing.instanceCount > 1) {
      const char* readReason = "";
      if (readUe3InstanceTransforms(vertexContext, m_ue3DecomposedInstances, &readReason)) {
        const size_t instancesRead = m_ue3DecomposedInstances.size();

        // Both of these read the batch as the game wrote it, before any culling: the batch key's
        // centroid has to stay view-independent, and the order probe measures the game's order.
        m_ue3DecomposedBatchKey = resolveUe3InstancedBatchKey(geoData, m_ue3DecomposedInstances);
        if (m_frameOptions.ue3LogInstancedDrawStats) {
          trackUe3InstanceOrderStability(m_ue3DecomposedBatchKey, m_ue3DecomposedInstances);
        }

        uint32_t culledByDistance = 0;
        uint32_t culledByBudget = 0;
        cullAndClampUe3InstanceTransforms(m_ue3DecomposedInstances, culledByDistance, culledByBudget);

        if (culledByBudget > 0) {
          ONCE(Logger::warn(str::format(
            "[RTX-Compatibility] UE3 instanced draw expands to ", instancesRead,
            " instances, above rtx.d3d9.ue3MaxDecomposedInstances (",
            std::max(m_frameOptions.ue3MaxDecomposedInstances, 1u), "); keeping a fixed ",
            m_ue3DecomposedInstances.size(), " of them and dropping ", culledByBudget,
            ". The kept subset is deliberately the same every frame - a view-dependent one flickers. ",
            describeUe3DrawIdentity(),
            " verts=", geoData.vertexCount, " prims=", drawContext.PrimitiveCount)));
        }

        if (m_frameOptions.ue3LogInstancedDrawStats) {
          ++m_ue3InstancedStatDraws;
          m_ue3InstancedStatInstancesSeen += instancesRead;
          m_ue3InstancedStatInstancesSubmitted += m_ue3DecomposedInstances.size();
          m_ue3InstancedStatCulledDistance += culledByDistance;
          m_ue3InstancedStatCulledBudget += culledByBudget;
        }
      } else {
        // Without placements, input-assembler positions would put the mesh at the world origin, so
        // the draw is refused below rather than rendered somewhere it does not belong.
        m_ue3InstanceTransformReadFailure = readReason;
      }
    }

    // UE3 vertex shader skinned (GPUSkin) draws share one bind-pose vertex buffer and bounding box
    // across every instance of a skeletal mesh, so the BLAS cache cannot tell simultaneous instances
    // apart on geometry alone. The first bone's translation separates them, and its bone hash gives
    // the geometry refit decision a handle on the animated pose. The bone matrices are RefToLocal,
    // so that translation only becomes a world position once carried through the LocalToWorld
    // processRenderState resolved above; used raw, every instance anchors near the world origin.
    m_activeDrawCallState.m_hasSkinnedWorldAnchor = false;
    // Reset the per-draw skinning identity: only VS-skinned draws (re)assign a bone hash
    // below, and processSkinning() leaves programmable-VS skinningData untouched. Without
    // this reset the last skinned draw's bone hash leaks into every subsequent draw; as
    // the pose animates, that leaked hash churns the BLAS refit decision, the DrawCallCache
    // exact-match and the ReplacementInstance identity of every static draw each frame,
    // forcing full geometry re-uploads and the dynamic path scene-wide.
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

    // Resolve where this draw's positions come from before anything keys off it: the capture
    // cache is only valid for exact sources, and the strict gate below refuses the inexact one.
    m_activeCapturePositionSource = Ue3CapturePositionSource::ClipReconstruction;
    if (m_parent->UseProgrammableVS() && m_frameOptions.useVertexCapture) {
      const char* positionSourceReason = "";
      m_activeCapturePositionSource = resolveUe3CapturePositionSource(
        indexContext, vertexContext, geoData, &positionSourceReason);
      logUe3CapturePositionSource(m_activeCapturePositionSource, positionSourceReason);

      if (m_frameOptions.ue3RequireExactVertexCapture &&
          !isUe3ExactCapturePositionSource(m_activeCapturePositionSource)) {
        ONCE(Logger::info("[RTX-Compatibility-Info] Ignoring draw without an exact vertex capture position source (rtx.d3d9.ue3RequireExactVertexCapture)."));
        return finishPrepare(prepareFlagsForIgnoredDraws);
      }

      // An instanced draw that did not land on input-assembler positions has no correct answer
      // left: capture would have every hardware instance overwrite one another's slots, producing
      // a cloud of triangles built from mixed placements that changes every time it is captured.
      // Dropping the draw is the honest outcome.
      if (m_currentUe3Instancing.instanceCount > 1 &&
          m_activeCapturePositionSource != Ue3CapturePositionSource::InputAssembler) {
        ONCE(Logger::warn(str::format(
          "[RTX-Compatibility] Ignoring hardware-instanced draw: its positions would have to come from "
          "vertex capture, which is indexed per vertex and so cannot separate instances (vf=",
          describeUe3VertexFactory(m_currentUe3VertexFactory),
          ", reason='", positionSourceReason, "') ",
          describeUe3DrawIdentity(), " ",
          describeUe3DrawInstancing(vertexContext),
          " decl=[", describeUe3VertexDeclaration(), "]")));
        return finishPrepare(prepareFlagsForIgnoredDraws);
      }
    }
    phaseTimer.lap(RtxGpuPassTimer::CpuCounter::AppPrepVertices);

    bool canUseCachedVertexCapture = false;
    XXH64_hash_t vertexCaptureCacheKey = kEmptyHash;
    {
      ScopedCpuProfileZoneN("UE3 geometry identity keys");

      // Stable VS hash (bytecode + camera-excluded constants), computed once per draw and
      // shared between the geometry hash below and the static vertex-capture cache key.
      m_activeStableVsHash = (m_parent->UseProgrammableVS() && m_frameOptions.useVertexCapture)
        ? computeUe3StableVertexShaderHash()
        : kEmptyHash;

      // Static-draw identity for the vertex-capture cache.
      canUseCachedVertexCapture =
        canUseUe3StaticVertexCaptureCache(indexContext, vertexContext, geoData);
      vertexCaptureCacheKey =
        canUseCachedVertexCapture
          ? computeUe3StaticVertexCaptureCacheKey(indexContext, vertexContext, drawContext, geoData)
          : kEmptyHash;

      // Geometry hash + bounding box memoization: draws with static IA buffers (any vertex
      // factory - skinned bind-pose data included) hash to the same IA components every
      // frame, so serve published results instead of re-hashing the full vertex/index data
      // per draw. The per-draw VertexShader component is recombined live so served hashes
      // are bit-identical to a fresh compute. First sighting schedules the normal worker
      // compute, which additionally publishes into the (heap-pinned) memo entry.
      bool servedGeometryFromMemo = false;
      std::shared_ptr<Ue3GeometryMemoEntry> geometryMemoPublishTo;
      std::shared_ptr<const Ue3GeometryMemoEntry> geometryMemoVerifyAgainst;
      const bool canMemoizeIaGeometry = canMemoizeUe3IaGeometryHashes(indexContext, vertexContext, geoData);
      const XXH64_hash_t iaGeometryMemoKey =
        canMemoizeIaGeometry
          ? computeUe3IaGeometryMemoKey(indexContext, vertexContext, drawContext, geoData)
          : kEmptyHash;

      // Diagnostic only. Reuses the memo key above rather than recomputing its own - that hash
      // walks every vertex stream record, which is far too much to spend twice per draw. Keyed off
      // structural eligibility rather than canUseCachedVertexCapture, because the question it
      // answers is why the cache is not hitting, so it has to keep working once the cache has
      // stood itself down.
      if (m_frameOptions.ue3LogVertexConstantChurn &&
          canMemoizeIaGeometry &&
          isUe3StaticVertexCaptureCacheEligible(indexContext, vertexContext, geoData)) {
        trackUe3ConstantChurn(iaGeometryMemoKey, geoData);
      }

      if (canMemoizeIaGeometry) {
        const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();
        const auto memoIt = m_ue3GeometryMemoCache.find(iaGeometryMemoKey);
        if (memoIt != m_ue3GeometryMemoCache.end()) {
          Ue3GeometryMemoEntry& entry = *memoIt->second;
          entry.lastFrameTouched = currentFrame;
          if (entry.hashesReady.load(std::memory_order_acquire)) {
            const uint32_t selfCheckFrames = m_frameOptions.ue3GeometryMemoSelfCheckFrames;
            const bool selfCheck = selfCheckFrames != 0 && (m_ue3FrameCounter % selfCheckFrames) == 0;
            if (selfCheck) {
              // rtx.d3d9.ue3GeometryMemoSelfCheckFrames: hash in full and have the worker compare
              // against the published entry. The fresh result goes into a new entry that replaces
              // this one in the map; the old one is only read from now on.
              geometryMemoVerifyAgainst = memoIt->second;
              geometryMemoPublishTo = std::make_shared<Ue3GeometryMemoEntry>();
              geometryMemoPublishTo->lastFrameTouched = currentFrame;
              memoIt->second = geometryMemoPublishTo;
            } else {
              GeometryHashes hashes;
              for (uint32_t i = 0; i < uint32_t(HashComponents::Count); i++) {
                hashes[HashComponents(i)] = entry.componentHashes[i];
              }
              hashes[HashComponents::VertexShader] = computeLiveGeometryVertexShaderHashComponent();
              hashes.precombine();
              geoData.hashes = hashes;
              servedGeometryFromMemo = true;
              if (entry.aabbReady.load(std::memory_order_acquire)) {
                geoData.boundingBox = entry.boundingBox;
              } else {
                geoData.futureBoundingBox = computeAxisAlignedBoundingBox(geoData);
              }
            }
          }
          // hashes not ready yet (worker still busy from an earlier frame): fall through
          // and compute normally this draw, without publishing a second time
        } else {
          geometryMemoPublishTo = std::make_shared<Ue3GeometryMemoEntry>();
          geometryMemoPublishTo->lastFrameTouched = currentFrame;
          m_ue3GeometryMemoCache.emplace(iaGeometryMemoKey, geometryMemoPublishTo);
        }
      }

      if (!servedGeometryFromMemo) {
        geoData.futureGeometryHashes = computeHash(geoData, maxOffsetedIndex, geometryMemoPublishTo, geometryMemoVerifyAgainst);
        geoData.futureBoundingBox = computeAxisAlignedBoundingBox(geoData, geometryMemoPublishTo, geometryMemoVerifyAgainst);

        if (geometryMemoPublishTo != nullptr && !geoData.futureGeometryHashes.valid()) {
          // hashing could not be scheduled (e.g. undefined position region): drop the
          // placeholder entry so it does not linger unfilled
          m_ue3GeometryMemoCache.erase(iaGeometryMemoKey);
        }
      }
    }
    phaseTimer.lap(RtxGpuPassTimer::CpuCounter::AppPrepIdentity);

    // Process skinning data
    m_activeDrawCallState.futureSkinningData = processSkinning(geoData);

    // Note: the material hash was already updated inside processTextures; nothing
    // mutates material data after that point, so no second update is needed here.

    const bool reusedCachedVertexCapture =
      canUseCachedVertexCapture &&
      tryReuseUe3StaticVertexCapture(vertexCaptureCacheKey, geoData);
    if (m_frameOptions.ue3LogCapturePrecision &&
        Logger::logLevel() <= LogLevel::Debug &&
        canUseCachedVertexCapture) {
      static fast_unordered_set s_loggedCacheReuse;
      const XXH64_hash_t logKey = vertexCaptureCacheKey ^ (reusedCachedVertexCapture ? 0x9E3779B97F4A7C15ull : 0xD1B54A32D192ED03ull);
      if (s_loggedCacheReuse.insert(logKey).second) {
        Logger::debug(str::format(
          "[RTX-Compatibility][UE3-Capture] static vertex capture cache ",
          reusedCachedVertexCapture ? "reused" : "recapturing",
          ", key=0x", std::hex, vertexCaptureCacheKey, std::dec,
          ", vertices=", geoData.vertexCount));
      }
    }

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
                                               /* allowPooledBuffer */ !canUseCachedVertexCapture);
    }
    if (canUseCachedVertexCapture && !reusedCachedVertexCapture && needVertexCapture) {
      ++m_ue3VertexCaptureCacheFrameCaptures;
      updateUe3StaticVertexCaptureCache(vertexCaptureCacheKey, geoData);
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
        submitUe3DecomposedInstanceDrawCallStates(params);
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
    const D3D9CommonShader* inferredPs = nullptr;
    XXH64_hash_t inferredPsHash = 0;
    PsSamplerTexcoordEntry* inferredPsEntry = nullptr;
    const Ue3VertexFactoryType vfType = m_currentUe3VertexFactory;
    const bool isUe3GpuSkinVF = vfType == Ue3VertexFactoryType::GPUSkin || vfType == Ue3VertexFactoryType::GPUSkinMorph;
    const bool isUe3TerrainVF = vfType == Ue3VertexFactoryType::Terrain || vfType == Ue3VertexFactoryType::TerrainMorph;
    const bool isUe3ParticleVF =
      vfType == Ue3VertexFactoryType::Particle ||
      vfType == Ue3VertexFactoryType::ParticleBeamTrail ||
      vfType == Ue3VertexFactoryType::LensFlare;
    // Instanced mesh particles compile the same FoliageVertexFactory.usf and pack their UVs the
    // same way, so they follow foliage rather than the camera-facing particle factories.
    const bool isUe3FoliageVF =
      vfType == Ue3VertexFactoryType::Foliage ||
      vfType == Ue3VertexFactoryType::ParticleInstancedMesh;
    const bool isUe3SpeedTreeVF = vfType == Ue3VertexFactoryType::SpeedTree;
    const bool isUe3LocalDecalVF = vfType == Ue3VertexFactoryType::LocalDecal;
    const bool isUe3MorphVF = vfType == Ue3VertexFactoryType::GPUSkinMorph;

    const bool likelyGpuSkinnedMesh = isUe3GpuSkinVF || [&]() {
      if (d3d9State().vertexDecl.ptr() == nullptr)
        return false;

      for (const auto& element : d3d9State().vertexDecl->GetElements()) {
        if (element.Usage == D3DDECLUSAGE_BLENDWEIGHT ||
            element.Usage == D3DDECLUSAGE_BLENDINDICES)
          return true;
      }
      return false;
    }();
    const Ue3VsShaderCtabInfo* ue3VsHints =
      (m_parent->UseProgrammableVS() &&
       d3d9State().vertexShader.ptr() != nullptr &&
       m_currentUe3CtabInfo.has_value())
        ? &(*m_currentUe3CtabInfo)
        : nullptr;
    const bool likelyUe3DecalUvSpace =
      isUe3LocalDecalVF ||
      (ue3VsHints != nullptr &&
       (ue3VsHints->hasDecalTransform ||
        ue3VsHints->hasDecalLocation ||
        ue3VsHints->hasDecalOffset));
    const bool likelyUe3TerrainUvSpace =
      isUe3TerrainVF ||
      (ue3VsHints != nullptr &&
       (ue3VsHints->hasLightMapCoordinateScaleBias ||
        ue3VsHints->hasShadowCoordinateScaleBias));
    const bool likelyUe3BillboardUvSpace =
      isUe3ParticleVF ||
      isUe3SpeedTreeVF ||
      (ue3VsHints != nullptr &&
       (ue3VsHints->hasTextureCoordinateScaleBias ||
        ue3VsHints->hasViewToLocal ||
        ue3VsHints->hasWindMatrices));
    const bool likelyUe3FlexiblePackedUvPath =
      !likelyGpuSkinnedMesh &&
      (likelyUe3DecalUvSpace || likelyUe3TerrainUvSpace || likelyUe3BillboardUvSpace || isUe3FoliageVF);
    const bool likelyPackedUvConventions = likelyGpuSkinnedMesh || likelyUe3FlexiblePackedUvPath;
    auto resolveInferredSamplerOffset = [&](const PsSamplerTexcoordEntry* entry, const uint32_t stage, float& outU, float& outV) -> bool {
      outU = 0.0f;
      outV = 0.0f;

      if (entry == nullptr || stage >= caps::MaxTexturesPS)
        return false;

      if (entry->samplers[stage].offsetImmediateValid) {
        outU = entry->samplers[stage].offsetImmediateU;
        outV = entry->samplers[stage].offsetImmediateV;
        return std::isfinite(outU) && std::isfinite(outV);
      }

      const int16_t offsetConstReg = entry->samplers[stage].offsetConstReg;
      if (offsetConstReg >= 0 && uint32_t(offsetConstReg) < caps::MaxFloatConstantsPS) {
        const Vector4& offsetConst = d3d9State().psConsts.fConsts[uint32_t(offsetConstReg)];
        const uint32_t compU = entry->samplers[stage].offsetConstCompU & 0x3;
        const uint32_t compV = entry->samplers[stage].offsetConstCompV & 0x3;
        outU = offsetConst[compU] * entry->samplers[stage].offsetFactorU;
        outV = offsetConst[compV] * entry->samplers[stage].offsetFactorV;
        return std::isfinite(outU) && std::isfinite(outV);
      }

      return false;
    };

    auto hasNonZeroInferredSamplerOffset = [&](const PsSamplerTexcoordEntry* entry, const uint32_t stage) -> bool {
      float uOffset = 0.0f;
      float vOffset = 0.0f;
      if (!resolveInferredSamplerOffset(entry, stage, uOffset, vOffset))
        return false;

      constexpr float kOffsetEps = 1e-5f;
      return std::abs(uOffset) > kOffsetEps || std::abs(vOffset) > kOffsetEps;
    };

    auto getOrInitPsSamplerTexcoordEntry = [&](const D3D9CommonShader* ps, XXH64_hash_t& outHash) -> PsSamplerTexcoordEntry* {
      if (ps == nullptr)
        return nullptr;

      outHash = ps->GetBytecodeHash();
      auto& entry = m_psSamplerTexcoordCache[outHash];
      if (!entry.initialized) {
        entry.initialized = true;
        const DxsoShaderView psView = makeDxsoShaderView(ps->GetBytecode(), ps);
        analyzePsSamplerUvOrigins(psView, entry.samplerUvOrigin);
        // Any other sampler infers to the defaults.
        uint32_t inferredSamplerMask = ps->GetShaderMask().samplerMask;
        for (const auto& [samplerRegister, name] : getUe3PsSamplerNames(outHash, ps->GetBytecode())) {
          inferredSamplerMask |= 1u << samplerRegister;
        }
        for (uint32_t s = 0; s < caps::MaxTexturesPS; s++) {
          if ((inferredSamplerMask & (1u << s)) == 0) {
            continue;
          }
          entry.samplers[s] = inferPixelShaderTexcoordForSampler(psView, s);
          if ((entry.samplers[s].semanticFlags & kPsSamplerSemanticLightmap) != 0)
            entry.lightmapSamplerMask |= (1u << s);
        }

        // The inference tracks coordinate expressions per register, so a sampler's coordinate
        // inherits whatever else fxc packed into its register's spare lanes. In UE3's base pass
        // that is the lightmap coordinate, which shares TEXCOORD0 with the material UV and
        // under TdBicubicFiltering goes through an offset and a `frc`; material samplers in
        // that compile alone were flagged UVOFS|UVANIM, and the score followed. The colour-term
        // analysis follows the coordinate lanes alone, and a flag it shows to be impossible on
        // those lanes is cleared. Clears only, so a flag derived from the sampler's own read
        // stays.
        const std::vector<uint8_t>& bytecode = ps->GetBytecode();
        const DxsoColorTermResult coordFacts = analyzeDxsoColorTerms(
          reinterpret_cast<const uint32_t*>(bytecode.data()), bytecode.size() / sizeof(uint32_t), DxsoColorTermInputs {});
        if (coordFacts.analyzed) {
          for (uint32_t s = 0; s < caps::MaxTexturesPS && s < kDxsoColorTermMaxSamplers; s++) {
            if (entry.samplers[s].sampleCount == 0 || coordFacts.samplerSampleCount[s] == 0)
              continue;
            const uint8_t expr = coordFacts.samplerCoordExpr[s];
            uint16_t& flags = entry.samplers[s].expressionFlags;
            const uint16_t before = flags;
            if ((expr & DxsoCoordExpr_Arith) == 0) {
              // a plain interpolant read: the only transform it can carry is a non-.xy swizzle
              flags &= ~uint16_t(kPsSamplerExprUvOffset | kPsSamplerExprUvAnimated | kPsSamplerExprBlendMath);
              const bool swizzled =
                entry.samplers[s].coordCompValid != 0 &&
                (entry.samplers[s].coordCompU != 0 || entry.samplers[s].coordCompV != 1);
              if (!swizzled)
                flags &= ~uint16_t(kPsSamplerExprUvTransform);
            }
            if ((expr & DxsoCoordExpr_Offset) == 0)
              flags &= ~uint16_t(kPsSamplerExprUvOffset);
            const bool animatedPossible =
              (expr & (DxsoCoordExpr_Wrap | DxsoCoordExpr_UnknownOffset)) != 0 ||
              entry.samplers[s].sampleCount >= 2u ||
              (flags & kPsSamplerExprUvTimeDriven) != 0;
            if (!animatedPossible)
              flags &= ~uint16_t(kPsSamplerExprUvAnimated);
            entry.samplerExpressionFlagsCleared[s] = uint16_t(before & ~flags);
          }
        }
      }

      return &entry;
    };

    if constexpr (!FixedFunction) {
      if (m_frameOptions.ue3EngineMode && d3d9State().pixelShader.ptr() != nullptr) {
        // Measures the bytecode analysis a shader's first sighting pays; the result is cached,
        // so the zone is near-empty on every later draw.
        ScopedCpuProfileZoneN("UE3 PS sampler analysis");
        inferredPs = d3d9State().pixelShader->GetCommonShader();
        inferredPsEntry = getOrInitPsSamplerTexcoordEntry(inferredPs, inferredPsHash);
      }
    }

    auto getRemixSampleView = [](D3D9CommonTexture* texture, const bool srgb) -> Rc<DxvkImageView> {
      if (texture == nullptr)
        return nullptr;
      // legacy Remix albedo path is 2D oriented so for cubemap albedo slots,
      // bind a face view to avoid falling back to white placeholders
      if (texture->GetType() == D3DRTYPE_CUBETEXTURE) {
        return texture->GetCubeFaceView(srgb);
      }
      return texture->GetSampleView(srgb);
    };
    bool selectedUe3MovieTexture = false;

    if constexpr (FixedFunction) {
      uint32_t textureID = 0;
      for (uint32_t idx = 0; idx < NumTexcoordBins && textureID < LegacyMaterialData::kMaxSupportedTextures; idx++) {
        const uint8_t stage = texcoordIndexToStage[idx];
        if (stage == kInvalidStage || d3d9State().textures[stage] == nullptr)
          continue;

        D3D9CommonTexture* pTexInfo = GetCommonTexture(d3d9State().textures[stage]);
        assert(pTexInfo != nullptr);
        const XXH64_hash_t texHash = pTexInfo->GetImage()->getHash();
        const XXH64_hash_t texDescHash =
          pTexInfo->GetImage() != nullptr
            ? pTexInfo->GetImage()->getDescriptorHash()
            : kEmptyHash;

        if (texHash == kEmptyHash)
          continue;

        if (textureID == 0)
          firstStage = stage;

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
        if (sampleView == nullptr)
          continue;
        m_activeDrawCallState.materialData.colorTextures[textureID] = TextureRef(sampleView);
        m_activeDrawCallState.materialData.samplers[textureID] = sampler;
        selectedUe3MovieTexture |= isUe3MovieTextureDescHash(texDescHash);
        if (textureID == 0)
          m_activeDrawCallState.materialData.colorTextureIsSrgb = srgb;

        auto shaderSampler = RemapStateSamplerShader(stage);
        m_activeDrawCallState.materialData.colorTextureSlot[textureID] = computeResourceSlotId(shaderSampler.first, DxsoBindingType::Image, uint32_t(shaderSampler.second));

        ++textureID;
      }
    } else {
      // for the shader path we pick the most relevant textures actually used by the pixel shader
      // we prefer sRGB textures for the first slot since normal maps/masks are usually sampled in linear space
      uint8_t chosenStages[LegacyMaterialData::kMaxSupportedTextures] = { kInvalidStage, kInvalidStage };
      int64_t chosenScore[LegacyMaterialData::kMaxSupportedTextures] = { std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::min() };
      uint8_t strictCubemapFallbackStage = kInvalidStage;
      int32_t strictCubemapFallbackScore = std::numeric_limits<int32_t>::min();

      const uint32_t usedSamplerMask = m_parent->m_psShaderMasks.samplerMask;
      // UE3 lightmaps are removed from the draw's texture set before anything reads it. A
      // score penalty is not enough: DirectionalLightmaps=True binds three coefficient
      // samplers where False binds one, so leaving them in changes the candidate pool, the
      // selection cache key and its area sum, and the per-texture material spread - every
      // one of which can flip the albedo pick between the two settings. Remix owns lighting,
      // so the coefficients carry nothing a raytraced surface wants.
      const uint32_t lightmapStageMask =
        (m_frameOptions.ue3EngineMode && inferredPsEntry != nullptr) ? inferredPsEntry->lightmapSamplerMask : 0u;
      // Material samplers the base pass consumes only as lighting inputs (normal maps, specular
      // colour, transfer masks) leave the pool with them: the simple-lightmap compile never
      // declares them, so a directional compile that could pick one would render the same
      // material differently. Opacity masks and coordinate drivers stay eligible - a cutout's
      // mask is its albedo's alpha.
      uint32_t lightingInputStageMask = 0;
      if (m_frameOptions.ue3EngineMode && inferredPs != nullptr && inferredPsHash != kEmptyHash) {
        lightingInputStageMask = getOrParseUe3PsMaterialIdentityInfo(
          inferredPsHash, inferredPs->GetBytecode(), inferredPs,
          m_frameOptions.ue3MicVolatileConstantDetection).lightingInputSamplerMask;
      }
      const uint32_t usedTextureMask =
        m_parent->m_activeTextures & usedSamplerMask & ~lightmapStageMask & ~lightingInputStageMask;

      // Publish what was bound at those samplers so the texture paths that cannot see a pixel
      // shader - hash preservation on CPU writes, the terrain baker's stage filter, the texture
      // picker - recognise the same images. First sight per hash only; the session set is
      // append-only so the repeat check stays local.
      if (m_frameOptions.ue3AutoDetectLightmapTextures) {
        for (const uint32_t stage : bit::BitMask(lightmapStageMask & m_parent->m_activeTextures)) {
          if (stage >= SamplerCount || d3d9State().textures[stage] == nullptr)
            continue;
          D3D9CommonTexture* lightmap = GetCommonTexture(d3d9State().textures[stage]);
          if (lightmap == nullptr || lightmap->GetImage() == nullptr)
            continue;
          const XXH64_hash_t lightmapHash = lightmap->GetImage()->getHash();
          if (lightmapHash != kEmptyHash && m_ue3SeenLightmapTextures.insert(lightmapHash).second)
            registerAutoDetectedLightmapTexture(lightmapHash);
        }
      }

      // Deterministic diffuse selection: the scoring below reads live shader constant
      // values (hasNonZeroInferredSamplerOffset), which UE3 rewrites per draw for
      // panner/time/view expressions - flipping large score terms and with them the
      // chosen albedo stage between frames or with camera position. Caching the first
      // decision per (PS, texture set, sRGB, vertex factory) key pins the pick.
      XXH64_hash_t selectionCacheKey = kEmptyHash;
      bool selectionCacheUsable = false;
      bool selectionFromCache = false;
      uint64_t selectionBoundAreaSum = 0;
      bool auditingPinnedSelection = false;
      uint8_t auditedPinnedStages[2] = { kInvalidStage, kInvalidStage };
      uint8_t auditedPinnedCubemapStage = kInvalidStage;
      if (m_frameOptions.ue3EngineMode &&
          inferredPsEntry != nullptr && inferredPsHash != kEmptyHash) {
        ScopedCpuProfileZoneN("UE3 diffuse selection lookup");
        if (!m_ue3DiffuseSelectionLoaded)
          loadUe3DiffuseSelectionCache();
        const FrameOptionSets& tagSets = m_frameOptionSets;
        if (tagSets.lightmapTextureDigest != m_ue3DiffuseSelectionLightmapTagDigest ||
            tagSets.neverAlbedoTextureDigest != m_ue3DiffuseSelectionNeverAlbedoTagDigest ||
            tagSets.preferredAlbedoTextureDigest != m_ue3DiffuseSelectionPreferredAlbedoTagDigest) {
          m_ue3DiffuseSelectionCache.clear();
          // Tagging invalidates every stored decision, so the persisted set has to shrink with
          // the in-memory one rather than keep serving picks the tags have just overruled.
          m_ue3DiffuseSelectionDirty = true;
          // re-log re-scored selections so tag effects are visible in ue3LogAlbedoSelection output
          m_loggedAlbedoSelections.clear();
          m_ue3DiffuseSelectionLightmapTagDigest = tagSets.lightmapTextureDigest;
          m_ue3DiffuseSelectionNeverAlbedoTagDigest = tagSets.neverAlbedoTextureDigest;
          m_ue3DiffuseSelectionPreferredAlbedoTagDigest = tagSets.preferredAlbedoTextureDigest;
        }

        struct SelectionKeyTuple {
          uint32_t stage;
          uint32_t srgb;
          XXH64_hash_t texHash;
        };
        static_assert(sizeof(SelectionKeyTuple) == 16, "SelectionKeyTuple must have no implicit padding (it is hashed by memory).");
        // one tuple per bound texture plus a trailing vertex-factory context tuple,
        // whose out-of-range stage index cannot collide with a real texture tuple
        std::array<SelectionKeyTuple, SamplerCount + 1> keyTuples;
        uint32_t keyTupleCount = 0;
        const BoundTextureSnapshot& boundTextures = ensureBoundTextureSnapshot();
        for (const uint32_t stage : bit::BitMask(usedTextureMask & boundTextures.mask)) {
          const BoundTextureSnapshotEntry& entry = boundTextures.entries[stage];
          if (!entry.hasImage)
            continue;
          keyTuples[keyTupleCount++] = {
            stage,
            d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1u,
            entry.imageHash,
          };

          const auto* desc = entry.texture->Desc();
          if (desc != nullptr) {
            selectionBoundAreaSum += uint64_t(desc->Width) * uint64_t(desc->Height);
          }
        }
        // score-relevant vertex factory context (packed UV biases differ per factory)
        const uint32_t vfContext =
          uint32_t(vfType) |
          (likelyGpuSkinnedMesh ? 1u << 8 : 0u) |
          (likelyUe3FlexiblePackedUvPath ? 1u << 9 : 0u) |
          (likelyPackedUvConventions ? 1u << 10 : 0u);
        keyTuples[keyTupleCount++] = { uint32_t(SamplerCount), vfContext, kEmptyHash };
        selectionCacheKey = XXH3_64bits_withSeed(keyTuples.data(), keyTupleCount * sizeof(SelectionKeyTuple), inferredPsHash);
        selectionCacheUsable = true;

        // Streaming-stable hashes give every mip variant of a material the same key, so a
        // decision scored against streamed-down mips (smaller bound texel area) is only
        // authoritative for equal or smaller sets; a larger set re-scores and supersedes it.
        const auto cachedSelection = m_ue3DiffuseSelectionCache.find(selectionCacheKey);
        if (cachedSelection != m_ue3DiffuseSelectionCache.end()) {
          if (selectionBoundAreaSum <= cachedSelection->second.decisionAreaSum) {
            chosenStages[0] = cachedSelection->second.chosenStages[0];
            chosenStages[1] = cachedSelection->second.chosenStages[1];
            strictCubemapFallbackStage = cachedSelection->second.cubemapFallbackStage;
            selectionFromCache = true;

            // Drop cached picks that landed on a refused render target so a real material
            // sampler can re-compete. Reusing the scoring-loop predicate keeps a legitimately
            // tagged pick from being discarded and re-scored every frame.
            const auto cachedPickRefused = [&](const uint8_t stage) {
              if (stage == kInvalidStage || stage >= SamplerCount ||
                  d3d9State().textures[stage] == nullptr) {
                return false;
              }
              return isUe3RenderTargetRefusedAsAlbedo(
                GetCommonTexture(d3d9State().textures[stage]), stage, inferredPsEntry);
            };

            if (m_frameOptions.ue3EngineMode &&
                (cachedPickRefused(chosenStages[0]) || cachedPickRefused(chosenStages[1]))) {
              chosenStages[0] = kInvalidStage;
              chosenStages[1] = kInvalidStage;
              strictCubemapFallbackStage = kInvalidStage;
              selectionFromCache = false;
              m_ue3DiffuseSelectionCache.erase(cachedSelection);
              m_loggedAlbedoSelections.erase(selectionCacheKey);
            } else if (cachedSelection->second.fromDisk && m_ue3DiffuseSelectionAuditsRemaining > 0) {
              // Score this one anyway and compare. The pinned pick still wins - re-scoring at an
              // arbitrary moment is exactly the transient the pin exists to avoid - so the audit
              // only reports.
              --m_ue3DiffuseSelectionAuditsRemaining;
              auditingPinnedSelection = true;
              auditedPinnedStages[0] = chosenStages[0];
              auditedPinnedStages[1] = chosenStages[1];
              auditedPinnedCubemapStage = strictCubemapFallbackStage;
              chosenStages[0] = kInvalidStage;
              chosenStages[1] = kInvalidStage;
              strictCubemapFallbackStage = kInvalidStage;
              selectionFromCache = false;
            }
          } else {
            // superseding re-score: let ue3LogAlbedoSelection dump the authoritative decision
            m_loggedAlbedoSelections.erase(selectionCacheKey);
          }
        }
      }

      // per-stage score breakdown for rtx.d3d9.ue3LogAlbedoSelection, dumped once per selection key.
      // Audited draws are excluded: their scoring is discarded, so dumping it would report a
      // decision the draw did not use.
      const bool logAlbedoSelection =
        m_frameOptions.ue3LogAlbedoSelection &&
        selectionCacheUsable &&
        !selectionFromCache &&
        !auditingPinnedSelection &&
        m_loggedAlbedoSelections.find(selectionCacheKey) == m_loggedAlbedoSelections.end();
      std::string albedoSelectionLog;

      // Size credit in mip steps: one doubling of texel area is worth a fixed amount, so a
      // texture that has streamed one mip further gains a fixed, small advantage instead of
      // one proportional to the texel count.
      auto albedoAreaScore = [](const uint64_t area) -> int64_t {
        constexpr int64_t kScorePerMipStep = 40'000;
        constexpr int64_t kMaxMipSteps = 24;  // 4096x4096
        int64_t steps = 0;
        for (uint64_t remaining = area >> 1; remaining != 0 && steps < kMaxMipSteps; remaining >>= 1)
          ++steps;
        return steps * kScorePerMipStep;
      };

      // The unprovable-origin penalty only means something where the resolved UV transform is
      // actually consumed; the gate mirrors the one guarding that resolution below.
      const bool uvOriginPenaltyActive =
        m_frameOptions.ue3EngineMode &&
        inferredPsEntry != nullptr;

      const uint32_t scoringTextureMask = selectionFromCache ? 0u : usedTextureMask;
      for (uint32_t stage : bit::BitMask(scoringTextureMask)) {
        if (stage >= SamplerCount || d3d9State().textures[stage] == nullptr)
          continue;

        D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[stage]);
        if (!texture)
          continue;

        const D3DRESOURCETYPE textureType = texture->GetType();
        const bool is2DTexture = textureType == D3DRTYPE_TEXTURE;
        const bool isCubeTexture = textureType == D3DRTYPE_CUBETEXTURE;
        if (!is2DTexture && !isCubeTexture)
          continue;

        const XXH64_hash_t texHash = texture->GetSampleView(false)->image()->getHash();
        if (isLightmapTexture(texHash))
          continue;
        const bool isNeverAlbedo = lookupHash(*m_frameOptions.neverAlbedoTextures, texHash);
        const bool isPreferredAlbedo = lookupHash(*m_frameOptions.preferredAlbedoTextures, texHash);

        // material spread: distinct pixel shaders sampling this texture. Identity albedos stay
        // at 1-2 (material instances share their parent's bytecode); shared library assets -
        // detail patterns, grunge/dirt overlays, tint ramps - appear across many unrelated shaders.
        // Scoring reads the spread loaded from disk, never the live count: a statistic that grows
        // as the level streams in would give the same material a different answer depending on
        // when it was first drawn, and the pinned selections would have to be thrown away every
        // time one texture crossed the threshold. This session's discoveries score from the next.
        uint32_t materialSpread = 0;
        if (inferredPsHash != kEmptyHash && texHash != kEmptyHash) {
          if (!m_ue3TextureSpreadLoaded)
            loadUe3TextureSpreadCache();
          Ue3TextureMaterialSpread& spread = m_ue3TextureMaterialSpread[texHash];
          bool psKnown = false;
          for (uint8_t i = 0; i < spread.count; i++) {
            if (spread.psHashes[i] == inferredPsHash) {
              psKnown = true;
              break;
            }
          }
          if (!psKnown && spread.count < spread.psHashes.size()) {
            spread.psHashes[spread.count++] = inferredPsHash;
            m_ue3TextureSpreadDirty = true;
          }
          materialSpread = spread.scoringCount;
        }

        const bool srgb = (d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1) != 0;
        const bool isRenderTarget = texture->IsRenderTarget();
        const XXH64_hash_t texDescHash =
          (texture->GetImage() != nullptr)
            ? texture->GetImage()->getDescriptorHash()
            : kEmptyHash;
        const auto* desc = texture->Desc();
        const uint64_t area = desc ? uint64_t(desc->Width) * uint64_t(desc->Height) : 0;
        uint16_t sampleCount = 0;
        bool hasInferredTexcoord = false;
        int8_t inferredTexcoordIdx = -1;
        uint8_t inferredSamplerSemanticFlags = 0;
        uint16_t inferredSamplerExpressionFlags = 0;
        bool inferredSamplerLooksEngineAuxiliary = false;
        bool inferredSamplerLooksMaterialTexture = false;
        bool inferredSamplerLooksLightmap = false;
        bool inferredSamplerLooksNonDiffuse = false;
        bool inferredSamplerLooksVideo = false;
        bool inferredSamplerLooksMovieTexture = false;
        bool inferredSamplerExprUvTransform = false;
        bool inferredSamplerExprUvOffset = false;
        bool inferredSamplerExprUvAnimated = false;
        bool inferredSamplerExprUvTimeDriven = false;
        bool inferredSamplerExprViewDependent = false;
        bool inferredSamplerExprMaskControl = false;
        bool inferredSamplerExprColorContribution = false;
        bool inferredSamplerExprBlendMath = false;
        bool inferredSamplerExprNormalDecode = false;
        bool inferredSamplerExprReachesOutputColor = false;
        bool inferredSamplerExprDiffuseAnchor = false;
        bool inferredUsesZw = false;
        bool inferredUsesWz = false;
        bool inferredUsesXy = false;
        bool inferredUsesPackedSecondary = false;
        bool hasNonZeroInferredOffset = false;
        // Whether this sampler's UV origin was proven back to a single interpolant. The
        // surface's texture transform is derived from the winning stage alone, so a stage
        // without a provable origin cannot carry one: winning the slot costs the surface
        // its whole UV transform and leaves the texture at raw interpolant scale.
        bool inferredSamplerUvOriginProvable = false;
        bool inferredSamplerReadsPrimaryUvPair = false;
        if (inferredPsEntry != nullptr && stage < caps::MaxTexturesPS) {
          const PsSamplerUvOrigin& stageUvOrigin = inferredPsEntry->samplerUvOrigin[stage];
          inferredSamplerUvOriginProvable = stageUvOrigin.originValid;
          // UE3 packs UV set 0 into an interpolant's .xy and set 1 into its .zw, so a proven
          // origin on the .xy pair names the mesh's primary channel.
          inferredSamplerReadsPrimaryUvPair =
            stageUvOrigin.originValid && stageUvOrigin.sitesAgree &&
            stageUvOrigin.compU == 0 && stageUvOrigin.compV == 1;
          sampleCount = inferredPsEntry->samplers[stage].sampleCount;
          inferredTexcoordIdx = inferredPsEntry->samplers[stage].texcoord;
          // GPUSkinMorphVF - TEXCOORD6/7 are morph delta streams, treat as noninferable UV
          if (isUe3MorphVF && inferredTexcoordIdx >= 6) {
            inferredTexcoordIdx = -1;
          }
          inferredSamplerSemanticFlags = inferredPsEntry->samplers[stage].semanticFlags;
          inferredSamplerExpressionFlags = inferredPsEntry->samplers[stage].expressionFlags;
          inferredSamplerLooksEngineAuxiliary = (inferredSamplerSemanticFlags & kPsSamplerSemanticEngineAuxiliary) != 0;
          inferredSamplerLooksMaterialTexture = (inferredSamplerSemanticFlags & kPsSamplerSemanticMaterialTexture) != 0;
          inferredSamplerLooksLightmap = (inferredSamplerSemanticFlags & kPsSamplerSemanticLightmap) != 0;
          inferredSamplerLooksNonDiffuse = (inferredSamplerSemanticFlags & kPsSamplerSemanticNonDiffuse) != 0;
          inferredSamplerLooksVideo = (inferredSamplerSemanticFlags & kPsSamplerSemanticVideo) != 0;
          inferredSamplerLooksMovieTexture = (inferredSamplerSemanticFlags & kPsSamplerSemanticMovieTexture) != 0;
          inferredSamplerExprUvTransform = (inferredSamplerExpressionFlags & kPsSamplerExprUvTransform) != 0;
          inferredSamplerExprUvOffset = (inferredSamplerExpressionFlags & kPsSamplerExprUvOffset) != 0;
          inferredSamplerExprUvAnimated = (inferredSamplerExpressionFlags & kPsSamplerExprUvAnimated) != 0;
          inferredSamplerExprUvTimeDriven = (inferredSamplerExpressionFlags & kPsSamplerExprUvTimeDriven) != 0;
          inferredSamplerExprViewDependent = (inferredSamplerExpressionFlags & kPsSamplerExprViewDependent) != 0;
          inferredSamplerExprMaskControl = (inferredSamplerExpressionFlags & kPsSamplerExprMaskControl) != 0;
          inferredSamplerExprColorContribution = (inferredSamplerExpressionFlags & kPsSamplerExprColorContribution) != 0;
          inferredSamplerExprBlendMath = (inferredSamplerExpressionFlags & kPsSamplerExprBlendMath) != 0;
          inferredSamplerExprNormalDecode = (inferredSamplerExpressionFlags & kPsSamplerExprNormalDecode) != 0;
          inferredSamplerExprReachesOutputColor = (inferredSamplerExpressionFlags & kPsSamplerExprReachesOutputColor) != 0;
          inferredSamplerExprDiffuseAnchor = (inferredSamplerExpressionFlags & kPsSamplerExprDiffuseAnchor) != 0;
          hasInferredTexcoord = inferredTexcoordIdx >= 0;
          inferredUsesZw =
            inferredPsEntry->samplers[stage].coordCompValid != 0 &&
            inferredPsEntry->samplers[stage].coordCompU == 2 &&
            inferredPsEntry->samplers[stage].coordCompV == 3;
          inferredUsesWz =
            inferredPsEntry->samplers[stage].coordCompValid != 0 &&
            inferredPsEntry->samplers[stage].coordCompU == 3 &&
            inferredPsEntry->samplers[stage].coordCompV == 2;
          inferredUsesXy =
            inferredPsEntry->samplers[stage].coordCompValid != 0 &&
            inferredPsEntry->samplers[stage].coordCompU == 0 &&
            inferredPsEntry->samplers[stage].coordCompV == 1;
          inferredUsesPackedSecondary = inferredUsesWz || inferredUsesZw;
          hasNonZeroInferredOffset = hasNonZeroInferredSamplerOffset(inferredPsEntry, stage);
        }
        const bool isMovieTexture =
          isUe3MovieTextureDescHash(texDescHash) ||
          (isRenderTarget && inferredSamplerLooksMovieTexture);

        if (isCubeTexture && !m_frameOptions.allowCubemaps) {
          const bool looksMaterialCubemap =
            sampleCount > 0 &&
            (!isRenderTarget || isMovieTexture) &&
            !inferredSamplerLooksEngineAuxiliary &&
            !inferredSamplerLooksLightmap &&
            !inferredSamplerLooksNonDiffuse &&
            !inferredSamplerExprViewDependent &&
            !inferredSamplerExprMaskControl &&
            !isNeverAlbedo &&
            (inferredSamplerLooksMaterialTexture || inferredSamplerSemanticFlags == 0);

          if (looksMaterialCubemap) {
            int32_t cubeScore = 0;
            cubeScore += int32_t(std::min<uint16_t>(sampleCount, 16u)) * 64;
            cubeScore += hasInferredTexcoord ? 256 : 0;
            cubeScore -= hasNonZeroInferredOffset ? 96 : 0;
            cubeScore -= int32_t(stage);

            if (cubeScore > strictCubemapFallbackScore) {
              strictCubemapFallbackScore = cubeScore;
              strictCubemapFallbackStage = uint8_t(stage);
            }
          }

          continue;
        }

        // hashless textures (typically render targets) can never be bound as legacy
        // albedo; letting one win a slot starves the real diffuse and leaves the
        // surface white with no clickable texture hash
        if (texHash == kEmptyHash)
          continue;

        // Non-UE3 keeps the score penalties below instead.
        if (m_frameOptions.ue3EngineMode &&
            isUe3RenderTargetRefusedAsAlbedo(texture, stage, inferredPsEntry))
          continue;

        // effective area: a small texture tiled NxM times covers N*M times its pixel area
        // (UE3 TexCoord UTiling/VTiling folded into shader literals, or held in a scalar-parameter
        // constant resolved at decision time; the selection cache pins the resulting pick).
        // Restricted to genuinely tiny authored tiles (e.g. a 64x128 window): tiled detail/dirt/
        // tint overlays are usually 256x256+ and must not out-rank the albedo on size.
        constexpr uint64_t kTilingCreditMaxRawArea = 128ull * 128ull;
        constexpr uint64_t kTilingCreditMaxEffectiveArea = 2ull * 1024ull * 1024ull;
        uint64_t effectiveArea = area;
        if (inferredPsEntry != nullptr && stage < caps::MaxTexturesPS &&
            area > 0 && area <= kTilingCreditMaxRawArea) {
          float tilingU = 1.0f;
          float tilingV = 1.0f;
          bool tilingKnown = false;
          if (inferredPsEntry->samplers[stage].scaleImmediateValid != 0) {
            tilingU = std::abs(inferredPsEntry->samplers[stage].scaleImmediateU);
            tilingV = std::abs(inferredPsEntry->samplers[stage].scaleImmediateV);
            tilingKnown = true;
          } else if (inferredPsEntry->samplers[stage].scaleConstReg >= 0 &&
                     uint32_t(inferredPsEntry->samplers[stage].scaleConstReg) < caps::MaxFloatConstantsPS) {
            const Vector4& scaleConst =
              d3d9State().psConsts.fConsts[uint32_t(inferredPsEntry->samplers[stage].scaleConstReg)];
            tilingU = std::abs(scaleConst[inferredPsEntry->samplers[stage].scaleConstCompU & 0x3u] *
                               inferredPsEntry->samplers[stage].scaleFactorU);
            tilingV = std::abs(scaleConst[inferredPsEntry->samplers[stage].scaleConstCompV & 0x3u] *
                               inferredPsEntry->samplers[stage].scaleFactorV);
            tilingKnown = std::isfinite(tilingU) && std::isfinite(tilingV);
          }
          if (tilingKnown) {
            const float tiles = std::min(std::max(tilingU * tilingV, 1.0f), 1024.0f);
            effectiveArea = std::min(uint64_t(double(area) * double(tiles)), kTilingCreditMaxEffectiveArea);
          }
        }
        // UV-math bonuses only apply where tiling resolved to an actual repeat factor: a tiled
        // small texture is an identity albedo, a plain UV transform on one is an overlay tell
        const bool hasResolvedTiling = effectiveArea > area;

        int64_t score = 0;
        score += srgb ? 1'000'000 : 0;
        score += int64_t(std::min<uint16_t>(sampleCount, 16u)) * 120'000ll;
        score += hasInferredTexcoord ? 250'000 : -150'000;
        score += hasResolvedTiling ? 80'000 : 0;
        score -= (isRenderTarget && !isMovieTexture) ? 500'000 : 0;
        score += isMovieTexture ? 4'000'000 : 0;
        score += inferredSamplerLooksMaterialTexture ? 230'000 : 0;
        score -= inferredSamplerLooksEngineAuxiliary ? 420'000 : 0;
        score -= inferredSamplerLooksLightmap ? 280'000 : 0;
        score -= inferredSamplerLooksNonDiffuse ? 220'000 : 0;
        score -= inferredSamplerExprViewDependent ? 260'000 : 0;
        score -= inferredSamplerExprMaskControl ? 320'000 : 0;
        score -= inferredSamplerLooksVideo ? 8'000'000 : 0;
        score -= isNeverAlbedo ? 6'000'000 : 0;
        score += isPreferredAlbedo ? 8'000'000 : 0;
        // bytecode-proven tangent-space normal decode (t * 2 - 1 into normalize/dot chains):
        // decisive penalty - must outweigh typical size advantages. Only applies to linear
        // (non-sRGB) samplers: UE3 imports normal maps with SRGB=0, while gamma-decoded color
        // textures can pick up the flag spuriously in lit shaders full of *2-1 remap math.
        const bool normalDecodeActive = inferredSamplerExprNormalDecode && !srgb;
        score -= normalDecodeActive ? 1'500'000 : 0;
        // only a high spread is directional: mid spreads (3-7) are just as often a legitimately
        // reused diffuse (e.g. common city wall/plaster sheets) as a shared overlay
        if (materialSpread >= 8)
          score -= 2'600'000;
        // tiny ramp/tint lookups (gradients, palettes) are material parameters, not albedo
        score -= (area > 0 && area <= 1'024) ? 350'000 : 0;
        if (m_frameOptions.ue3EngineMode) {
          // UE3 binds many scene buffers, shadow maps, exposure/color curves, and UI/video
          // surfaces alongside material samplers. Keep these out of legacy albedo slots.
          score -= ((isRenderTarget && !isMovieTexture) || inferredSamplerLooksEngineAuxiliary) ? 450'000 : 0;
          score -= (inferredSamplerLooksEngineAuxiliary && !inferredSamplerLooksMaterialTexture) ? 350'000 : 0;
          score -= (isRenderTarget && !srgb && !isMovieTexture) ? 250'000 : 0;
          score -= (!hasInferredTexcoord && inferredSamplerLooksEngineAuxiliary) ? 220'000 : 0;
        }
        const bool looksExpressionDrivenMaterial =
          !inferredSamplerLooksEngineAuxiliary &&
          !inferredSamplerLooksLightmap &&
          !inferredSamplerLooksNonDiffuse &&
          !inferredSamplerExprViewDependent &&
          !inferredSamplerExprMaskControl &&
          !isNeverAlbedo &&
          (inferredSamplerLooksMaterialTexture || inferredSamplerSemanticFlags == 0);
        // On a static mesh UV set 1 is the secondary/lightmap channel, so between two material
        // samplers the one reading the primary .xy pair is the surface's own diffuse. Proven per
        // sampler from bytecode and therefore decided on the first frame, which matters because
        // the winner also fixes the surface's texcoord set: leaving equally-sized candidates to
        // the near-tie tiebreaks below would let a second layer take the slot and drag the whole
        // surface onto its UV set. The packed-UV block below expresses the same idea from
        // statistical inference, so it cannot separate samplers whose inference came out equal;
        // this term is additive to it.
        score += (looksExpressionDrivenMaterial && inferredSamplerReadsPrimaryUvPair) ? 150'000 : 0;
        score += (looksExpressionDrivenMaterial && inferredSamplerExprUvTransform && hasResolvedTiling) ? 95'000 : 0;
        score += (looksExpressionDrivenMaterial && inferredSamplerExprUvOffset) ? 52'000 : 0;
        score += (looksExpressionDrivenMaterial && inferredSamplerExprUvAnimated) ? 115'000 : 0;
        score += (looksExpressionDrivenMaterial && inferredSamplerExprUvTimeDriven && sampleCount >= 2u) ? 140'000 : 0;
        score += (looksExpressionDrivenMaterial && inferredSamplerExprColorContribution) ? 65'000 : 0;
        score += (looksExpressionDrivenMaterial && inferredSamplerExprBlendMath && sampleCount >= 2u) ? 60'000 : 0;
        // sampled value provably reaches oC0.rgb as color - the defining trait of a diffuse/emissive
        // texture, which normal/lighting-input samplers lack (their values collapse in dot products)
        score += (looksExpressionDrivenMaterial && inferredSamplerExprReachesOutputColor) ? 200'000 : 0;
        // deterministic UE3 base-pass structure: only the diffuse expression is multiplied with the
        // lightmap sample (static geometry) or the ambient/sky lighting constants (dynamic/unlit).
        // Dominates the softer heuristics but stays below movie surfaces and user texture tags.
        // Render targets are excluded: light-environment attenuation buffers also multiply into
        // sky/ambient lighting terms but can never be a surface albedo.
        score += (looksExpressionDrivenMaterial && !normalDecodeActive && !isRenderTarget &&
                  inferredSamplerExprDiffuseAnchor) ? 2'500'000 : 0;
        // A sampler whose UV origin is unprovable must not win on size: the transform is taken
        // from the winner only, so promoting one drops the surface to raw interpolant UVs.
        // Flat, so a shader where no sampler resolves is left unaffected.
        score -= (uvOriginPenaltyActive && !inferredSamplerUvOriginProvable) ? 1'200'000 : 0;
        // normal maps get no size credit - resolution advantage must not offset the decode penalty.
        // Under UE3 the same applies to render targets that only survived the refusal above
        // through an explicit tag: their near-backbuffer area would dominate every other signal.
        const bool renderTargetSizeCreditDenied =
          m_frameOptions.ue3EngineMode && isRenderTarget && !isMovieTexture;
        // Size counts in mip steps, not texels: streaming rewrites the bound dimensions as a
        // level pages in, so raw area would let whichever candidate had paged in further win
        // outright. One doubling is worth less than any single structural signal, which orders
        // equally-classified candidates by size without letting residency overturn class.
        score += (normalDecodeActive || renderTargetSizeCreditDenied)
          ? 0
          : int64_t(albedoAreaScore(effectiveArea));
        // near-exact ties among color-chain candidates (magnitudes stay below any real signal):
        // prefer a UV transform (the tiled base material - overlays sample raw UVs), then the
        // LATER sampler (the material translator assigns Texture2D_N slots in property compile
        // order Normal -> Emissive -> Diffuse). Everywhere else prefer the earlier stage.
        const bool diffuseChainCandidate =
          looksExpressionDrivenMaterial && !normalDecodeActive && !isRenderTarget &&
          inferredSamplerExprReachesOutputColor;
        if (diffuseChainCandidate) {
          score += inferredSamplerExprUvTransform ? 200 : 0;
          score += int64_t(stage) * 4;
        } else {
          score -= int64_t(stage);
        }
        if (likelyPackedUvConventions) {
          // UE3-style shader paths (skinned and non-skinned vertex factories) frequently pack
          // secondary UVs into non-`.xy` components, so we bias slot 0 toward diffuse-like UV usage
          const bool packedPairSuspicious =
            hasNonZeroInferredOffset ||
            inferredSamplerLooksEngineAuxiliary ||
            (likelyGpuSkinnedMesh && !inferredSamplerLooksMaterialTexture);

          const bool skinnedHighConfidence =
            likelyGpuSkinnedMesh && inferredSamplerLooksMaterialTexture && sampleCount >= 3u;
          const int64_t uv0Bonus = likelyGpuSkinnedMesh ? 300'000ll : 60'000ll;
          const int64_t nonUv0Penalty = likelyGpuSkinnedMesh ? 180'000ll : 20'000ll;
          const int64_t xyBonus = likelyGpuSkinnedMesh ? 120'000ll : 35'000ll;
          const int64_t wzPenalty = likelyGpuSkinnedMesh
            ? (skinnedHighConfidence ? 160'000ll : 320'000ll)
            : (packedPairSuspicious ? 120'000ll : 8'000ll);
          const int64_t zwPenalty = likelyGpuSkinnedMesh
            ? (skinnedHighConfidence ? 125'000ll : 250'000ll)
            : (packedPairSuspicious ? 100'000ll : 8'000ll);
          const int64_t packedSecondaryPenalty = likelyGpuSkinnedMesh
            ? (skinnedHighConfidence ? 40'000ll : 80'000ll)
            : (packedPairSuspicious ? 28'000ll : 2'000ll);
          const int64_t offsetPenalty = likelyGpuSkinnedMesh
            ? 220'000ll
            : (inferredSamplerLooksEngineAuxiliary ? 180'000ll
               : (inferredSamplerLooksMaterialTexture ? 12'000ll : 45'000ll));
          const int64_t auxiliaryPenalty = likelyGpuSkinnedMesh ? 120'000ll : 70'000ll;
          const bool looksUvAnimatedMaterialTexture =
            inferredSamplerLooksMaterialTexture &&
            !inferredSamplerLooksEngineAuxiliary &&
            !inferredSamplerLooksNonDiffuse &&
            sampleCount >= 2u;
          const int64_t adjustedOffsetPenalty = looksUvAnimatedMaterialTexture
            ? std::max<int64_t>(offsetPenalty / 6ll, 2'000ll)
            : offsetPenalty;

          if (inferredTexcoordIdx == 0)
            score += uv0Bonus;
          else if (inferredTexcoordIdx > 0)
            score -= nonUv0Penalty;

          if (inferredUsesXy)
            score += xyBonus;
          if (inferredUsesWz)
            score -= wzPenalty;
          else if (inferredUsesZw)
            score -= zwPenalty;

          if (inferredUsesPackedSecondary)
            score -= packedSecondaryPenalty;

          if (hasNonZeroInferredOffset)
            score -= adjustedOffsetPenalty;

          if (inferredSamplerLooksEngineAuxiliary)
            score -= auxiliaryPenalty;

          const bool highConfidenceMaterialTexture =
            inferredSamplerLooksMaterialTexture && sampleCount >= 3u;
          if (inferredSamplerLooksMaterialTexture &&
              inferredUsesPackedSecondary &&
              !packedPairSuspicious &&
              (likelyUe3FlexiblePackedUvPath || (likelyGpuSkinnedMesh && highConfidenceMaterialTexture))) {
            score += 45'000ll;
          }
        }

        if (logAlbedoSelection) {
          std::string flagList;
          auto appendFlag = [&](const bool set, const char* name) {
            if (!set)
              return;
            if (!flagList.empty())
              flagList += "|";
            flagList += name;
          };
          appendFlag(inferredSamplerLooksMaterialTexture, "MAT");
          appendFlag(inferredSamplerLooksEngineAuxiliary, "AUX");
          appendFlag(inferredSamplerLooksLightmap, "LIGHTMAP");
          appendFlag(inferredSamplerLooksNonDiffuse, "NONDIFFUSE");
          appendFlag(inferredSamplerLooksVideo, "VIDEO");
          appendFlag(inferredSamplerLooksMovieTexture, "MOVIE");
          appendFlag(inferredSamplerExprUvTransform, "UVXFORM");
          appendFlag(inferredSamplerExprUvOffset, "UVOFS");
          appendFlag(inferredSamplerExprUvAnimated, "UVANIM");
          appendFlag(inferredSamplerExprUvTimeDriven, "UVTIME");
          appendFlag(inferredSamplerExprViewDependent, "VIEWDEP");
          appendFlag(inferredSamplerExprMaskControl, "MASKCTL");
          appendFlag(inferredSamplerExprColorContribution, "COLORCONTRIB");
          appendFlag(inferredSamplerExprBlendMath, "BLEND");
          appendFlag(normalDecodeActive, "NORMALDECODE");
          appendFlag(inferredSamplerExprNormalDecode && !normalDecodeActive, "NORMALDECODE-SRGBVETO");
          appendFlag(inferredSamplerExprReachesOutputColor, "REACHESOC0");
          appendFlag(inferredSamplerExprDiffuseAnchor, "ANCHOR");
          appendFlag(isNeverAlbedo, "TAG:NEVERALBEDO");
          appendFlag(isPreferredAlbedo, "TAG:PREFERALBEDO");
          appendFlag(isRenderTarget, "RT");
          appendFlag(uvOriginPenaltyActive && !inferredSamplerUvOriginProvable, "NOUVORIGIN");
          appendFlag(inferredSamplerReadsPrimaryUvPair, "UV0XY");
          // what the register-granular inference had derived that the coordinate lanes rule out
          if (inferredPsEntry != nullptr && inferredPsEntry->samplerExpressionFlagsCleared[stage] != 0) {
            const uint16_t cleared = inferredPsEntry->samplerExpressionFlagsCleared[stage];
            std::string clearedList;
            if (cleared & kPsSamplerExprUvTransform) clearedList += "UVXFORM ";
            if (cleared & kPsSamplerExprUvOffset)    clearedList += "UVOFS ";
            if (cleared & kPsSamplerExprUvAnimated)  clearedList += "UVANIM ";
            if (cleared & kPsSamplerExprBlendMath)   clearedList += "BLEND ";
            appendFlag(true, str::format("LANECLEARED:", clearedList).c_str());
          }

          albedoSelectionLog += str::format(
            "\n  s", stage,
            " tex=0x", std::hex, texHash, std::dec,
            " ", desc ? desc->Width : 0u, "x", desc ? desc->Height : 0u,
            " srgb=", srgb ? 1 : 0,
            " samples=", sampleCount,
            " tc=", int32_t(inferredTexcoordIdx),
            " effArea=", effectiveArea,
            " spread=", materialSpread,
            " flags=[", flagList.empty() ? "-" : flagList, "]",
            " score=", score);
        }

        // insert into top-2 (simple selection sort)
        for (uint32_t slot = 0; slot < LegacyMaterialData::kMaxSupportedTextures; slot++) {
          if (stage == chosenStages[slot])
            break;

          if (score > chosenScore[slot]) {
            for (uint32_t s = LegacyMaterialData::kMaxSupportedTextures - 1; s > slot; s--) {
              chosenStages[s] = chosenStages[s - 1];
              chosenScore[s] = chosenScore[s - 1];
            }
            chosenStages[slot] = uint8_t(stage);
            chosenScore[slot] = score;
            break;
          }
        }
      }

      if (chosenStages[0] == kInvalidStage && strictCubemapFallbackStage != kInvalidStage) {
        chosenStages[0] = strictCubemapFallbackStage;
      }

      // if no candidate scored, fall back to the lowest PS-used sampler holding a bindable
      // (hashed) texture - never raw stage 0, which may hold a stale texture the shader
      // never samples, and never a hashless texture.
      // The second pass relaxes the render-target refusal: a refused target must not displace a
      // real material texture, but when the shader samples nothing else it is the material's only
      // colour source, and dropping it leaves the surface with no albedo and no taggable hash.
      for (uint32_t pass = 0; pass < 2 && chosenStages[0] == kInvalidStage; pass++) {
        const bool allowRefusedRenderTargets = pass == 1;
        for (uint32_t stage : bit::BitMask(usedTextureMask)) {
          if (stage >= SamplerCount || d3d9State().textures[stage] == nullptr)
            continue;
          D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[stage]);
          if (texture == nullptr || texture->GetImage() == nullptr ||
              texture->GetImage()->getHash() == kEmptyHash)
            continue;
          if (!allowRefusedRenderTargets && m_frameOptions.ue3EngineMode &&
              isUe3RenderTargetRefusedAsAlbedo(texture, stage, inferredPsEntry))
            continue;
          chosenStages[0] = uint8_t(stage);
          break;
        }
      }

      // compat fix for UE3-like packed UV conventions:
      // if primary stage selection is suspicious (packed UV components, non-UV0, or atlas offset)
      // prefer a sibling sampler that references the same texture but looks more diffuse-like
      if (!selectionFromCache &&
          likelyPackedUvConventions &&
          inferredPsEntry != nullptr &&
          chosenStages[0] != kInvalidStage &&
          chosenStages[0] < caps::MaxTexturesPS &&
          d3d9State().textures[chosenStages[0]] != nullptr) {
        const uint8_t primaryStage = chosenStages[0];
        const bool primaryUsesZw = inferredPsEntry->samplers[primaryStage].coordCompValid != 0 &&
                                   inferredPsEntry->samplers[primaryStage].coordCompU == 2 &&
                                   inferredPsEntry->samplers[primaryStage].coordCompV == 3;
        const bool primaryUsesWz = inferredPsEntry->samplers[primaryStage].coordCompValid != 0 &&
                                   inferredPsEntry->samplers[primaryStage].coordCompU == 3 &&
                                   inferredPsEntry->samplers[primaryStage].coordCompV == 2;
        const bool primaryUsesPackedSecondary = primaryUsesWz || primaryUsesZw;
        const int8_t primaryTexcoord = inferredPsEntry->samplers[primaryStage].texcoord;
        const bool primaryHasNonZeroOffset = hasNonZeroInferredSamplerOffset(inferredPsEntry, primaryStage);
        const bool primaryLooksEngineAuxiliary =
          (inferredPsEntry->samplers[primaryStage].semanticFlags & kPsSamplerSemanticEngineAuxiliary) != 0;
        const bool primaryLooksMaterialTexture =
          (inferredPsEntry->samplers[primaryStage].semanticFlags & kPsSamplerSemanticMaterialTexture) != 0;
        const bool primaryLooksViewDependent =
          (inferredPsEntry->samplers[primaryStage].expressionFlags & kPsSamplerExprViewDependent) != 0;
        const bool primaryLooksMaskControl =
          (inferredPsEntry->samplers[primaryStage].expressionFlags & kPsSamplerExprMaskControl) != 0;
        const bool primaryPackedPairSuspicious =
          primaryUsesPackedSecondary &&
          (primaryLooksEngineAuxiliary ||
           primaryHasNonZeroOffset ||
           (likelyGpuSkinnedMesh && !primaryLooksMaterialTexture));
        const bool primaryOffsetSuspicious =
          primaryHasNonZeroOffset &&
          (likelyGpuSkinnedMesh || primaryLooksEngineAuxiliary);
        const bool primarySuspicious =
          primaryLooksEngineAuxiliary ||
          primaryLooksViewDependent ||
          primaryLooksMaskControl ||
          primaryPackedPairSuspicious ||
          primaryOffsetSuspicious ||
          (likelyGpuSkinnedMesh && primaryTexcoord > 0);

        if (primarySuspicious) {
          D3D9CommonTexture* primaryTexture = GetCommonTexture(d3d9State().textures[primaryStage]);
          const XXH64_hash_t primaryHash =
            (primaryTexture != nullptr && primaryTexture->GetImage() != nullptr)
              ? primaryTexture->GetImage()->getHash()
              : kEmptyHash;

          auto scoreDiffuseCandidate = [&](const uint32_t stage) -> int32_t {
            if (stage >= caps::MaxTexturesPS)
              return std::numeric_limits<int32_t>::min();

            if (inferredPsEntry->samplers[stage].sampleCount == 0)
              return std::numeric_limits<int32_t>::min();

            int32_t score = 0;
            const int32_t uv0Bonus = likelyGpuSkinnedMesh ? 420 : 110;
            const int32_t nonUv0Penalty = likelyGpuSkinnedMesh ? 240 : 40;
            const int32_t xyBonus = likelyGpuSkinnedMesh ? 320 : 80;

            const int8_t tc = inferredPsEntry->samplers[stage].texcoord;
            if (tc == 0)
              score += uv0Bonus;
            else if (tc > 0)
              score -= nonUv0Penalty;

            const uint8_t semanticFlags = inferredPsEntry->samplers[stage].semanticFlags;
            const uint16_t expressionFlags = inferredPsEntry->samplers[stage].expressionFlags;
            const bool looksMaterialTexture = (semanticFlags & kPsSamplerSemanticMaterialTexture) != 0;
            const bool looksEngineAuxiliary = (semanticFlags & kPsSamplerSemanticEngineAuxiliary) != 0;
            const bool looksNonDiffuse = (semanticFlags & kPsSamplerSemanticNonDiffuse) != 0;
            const bool looksVideo = (semanticFlags & kPsSamplerSemanticVideo) != 0;
            const bool looksExprUvTransform = (expressionFlags & kPsSamplerExprUvTransform) != 0;
            const bool looksExprUvOffset = (expressionFlags & kPsSamplerExprUvOffset) != 0;
            const bool looksExprUvAnimated = (expressionFlags & kPsSamplerExprUvAnimated) != 0;
            const bool looksExprUvTimeDriven = (expressionFlags & kPsSamplerExprUvTimeDriven) != 0;
            const bool looksExprViewDependent = (expressionFlags & kPsSamplerExprViewDependent) != 0;
            const bool looksExprMaskControl = (expressionFlags & kPsSamplerExprMaskControl) != 0;
            const bool looksExprColorContribution = (expressionFlags & kPsSamplerExprColorContribution) != 0;
            const bool looksExprBlendMath = (expressionFlags & kPsSamplerExprBlendMath) != 0;
            const bool looksExprNormalDecode = (expressionFlags & kPsSamplerExprNormalDecode) != 0;
            const bool looksExprReachesOutputColor = (expressionFlags & kPsSamplerExprReachesOutputColor) != 0;
            const bool looksExprDiffuseAnchor = (expressionFlags & kPsSamplerExprDiffuseAnchor) != 0;
            if ((semanticFlags & kPsSamplerSemanticMaterialTexture) != 0)
              score += 180;
            if ((semanticFlags & kPsSamplerSemanticEngineAuxiliary) != 0)
              score -= 360;
            if ((semanticFlags & kPsSamplerSemanticLightmap) != 0)
              score -= 260;
            if (m_frameOptions.ue3EngineMode && looksEngineAuxiliary)
              score -= looksMaterialTexture ? 160 : 320;
            if (looksNonDiffuse)
              score -= 220;
            if (looksExprViewDependent)
              score -= 240;
            if (looksExprMaskControl)
              score -= 280;
            if (looksVideo)
              score -= 8000;
            const bool candidateSrgb =
              (d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1) != 0;
            const bool candidateNormalDecode = looksExprNormalDecode && !candidateSrgb;
            if (candidateNormalDecode)
              score -= 3000;
            const bool looksExpressionDrivenMaterial =
              !looksEngineAuxiliary &&
              !looksNonDiffuse &&
              !looksExprViewDependent &&
              !looksExprMaskControl &&
              (looksMaterialTexture || semanticFlags == 0);
            if (looksExpressionDrivenMaterial && looksExprUvTransform)
              score += 95;
            if (looksExpressionDrivenMaterial && looksExprUvOffset)
              score += 55;
            if (looksExpressionDrivenMaterial && looksExprUvAnimated)
              score += 125;
            if (looksExpressionDrivenMaterial && looksExprUvTimeDriven &&
                inferredPsEntry->samplers[stage].sampleCount >= 2u)
              score += 70;
            if (looksExpressionDrivenMaterial && looksExprColorContribution)
              score += 30;
            if (looksExpressionDrivenMaterial && looksExprBlendMath &&
                inferredPsEntry->samplers[stage].sampleCount >= 2u)
              score += 45;
            if (looksExpressionDrivenMaterial && looksExprReachesOutputColor)
              score += 80;
            if (looksExpressionDrivenMaterial && !candidateNormalDecode && looksExprDiffuseAnchor)
              score += 600;

            const bool candidateHasNonZeroOffset = hasNonZeroInferredSamplerOffset(inferredPsEntry, stage);
            const bool candidatePackedPairSuspicious =
              candidateHasNonZeroOffset ||
              looksEngineAuxiliary ||
              (likelyGpuSkinnedMesh && !looksMaterialTexture);
            const int32_t wzPenalty = likelyGpuSkinnedMesh
              ? 380
              : (candidatePackedPairSuspicious ? 180 : 20);
            const int32_t zwPenalty = likelyGpuSkinnedMesh
              ? 360
              : (candidatePackedPairSuspicious ? 160 : 20);
            const int32_t offsetPenalty = likelyGpuSkinnedMesh
              ? 320
              : (looksEngineAuxiliary ? 220 : (looksMaterialTexture ? 20 : 60));
            const bool candidateLooksUvAnimatedMaterialTexture =
              looksMaterialTexture &&
              !looksEngineAuxiliary &&
              !looksNonDiffuse &&
              inferredPsEntry->samplers[stage].sampleCount >= 2u;
            const int32_t adjustedOffsetPenalty = candidateLooksUvAnimatedMaterialTexture
              ? std::max(offsetPenalty / 6, 8)
              : offsetPenalty;

            if (inferredPsEntry->samplers[stage].coordCompValid) {
              const uint8_t compU = inferredPsEntry->samplers[stage].coordCompU & 0x3u;
              const uint8_t compV = inferredPsEntry->samplers[stage].coordCompV & 0x3u;
              if (compU == 0u && compV == 1u)
                score += xyBonus;
              else if (compU == 3u && compV == 2u)
                score -= wzPenalty;
              else if (compU == 2u && compV == 3u)
                score -= zwPenalty;
            }

            if (candidateHasNonZeroOffset)
              score -= adjustedOffsetPenalty;

            if (!likelyGpuSkinnedMesh &&
                likelyUe3FlexiblePackedUvPath &&
                looksMaterialTexture &&
                inferredPsEntry->samplers[stage].coordCompValid) {
              const uint8_t compU = inferredPsEntry->samplers[stage].coordCompU & 0x3u;
              const uint8_t compV = inferredPsEntry->samplers[stage].coordCompV & 0x3u;
              const bool usesPackedSecondaryPair =
                (compU == 2u && compV == 3u) ||
                (compU == 3u && compV == 2u);
              if (usesPackedSecondaryPair && !candidatePackedPairSuspicious)
                score += 60;
            }

            score += int32_t(std::min<uint16_t>(inferredPsEntry->samplers[stage].sampleCount, 8u)) * 16;
            score -= int32_t(stage);

            return score;
          };

          if (primaryHash != kEmptyHash) {
            int32_t bestScore = scoreDiffuseCandidate(primaryStage);
            uint8_t promotedStage = kInvalidStage;

            for (uint32_t stage : bit::BitMask(usedTextureMask)) {
              if (stage >= SamplerCount ||
                  stage == primaryStage ||
                  d3d9State().textures[stage] == nullptr ||
                  stage >= caps::MaxTexturesPS) {
                continue;
              }

              D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[stage]);
              if (texture == nullptr || texture->GetImage() == nullptr)
                continue;

              if (texture->GetImage()->getHash() != primaryHash)
                continue;

              const int32_t candidateScore = scoreDiffuseCandidate(stage);
              if (candidateScore > bestScore + 40) {
                bestScore = candidateScore;
                promotedStage = uint8_t(stage);
              }
            }

            if (promotedStage != kInvalidStage) {
              if (chosenStages[1] == promotedStage)
                std::swap(chosenStages[0], chosenStages[1]);
              else
                chosenStages[0] = promotedStage;
            }
          }
        }
      }

      // remember the final (post-fallback, post-promotion) decision for this material key,
      // overwriting any decision made against a smaller (streamed-down) texel area
      if (auditingPinnedSelection) {
        if ((chosenStages[0] != auditedPinnedStages[0] || chosenStages[1] != auditedPinnedStages[1]) &&
            !m_ue3DiffuseSelectionAuditWarned) {
          m_ue3DiffuseSelectionAuditWarned = true;
          Logger::warn(str::format(
            "[RTX-Compatibility][UE3] Albedo selection cache disagrees with current scoring "
            "(material key 0x", std::hex, selectionCacheKey, ": stored [s",
            uint32_t(auditedPinnedStages[0]), ",s", uint32_t(auditedPinnedStages[1]),
            "], scored [s", uint32_t(chosenStages[0]), ",s", uint32_t(chosenStages[1]), "])", std::dec,
            ". Stored picks are being used, so a scoring change will not take effect: bump "
            "kUe3DiffuseSelectionScoringVersion, or delete ", kUe3DiffuseSelectionCachePath, "."));
        }
        // The pin stands regardless - the audit reports, it does not re-decide.
        chosenStages[0] = auditedPinnedStages[0];
        chosenStages[1] = auditedPinnedStages[1];
        strictCubemapFallbackStage = auditedPinnedCubemapStage;
      } else if (selectionCacheUsable && !selectionFromCache) {
        Ue3DiffuseSelectionEntry cacheEntry;
        cacheEntry.chosenStages[0] = chosenStages[0];
        cacheEntry.chosenStages[1] = chosenStages[1];
        cacheEntry.cubemapFallbackStage = strictCubemapFallbackStage;
        cacheEntry.decisionAreaSum = selectionBoundAreaSum;
        m_ue3DiffuseSelectionCache[selectionCacheKey] = cacheEntry;
        m_ue3DiffuseSelectionDirty = true;
      }

      if (logAlbedoSelection) {
        m_loggedAlbedoSelections.insert(selectionCacheKey);
        auto stageName = [&](const uint8_t stage) {
          return stage == kInvalidStage ? std::string("-") : str::format("s", uint32_t(stage));
        };
        Logger::info(str::format(
          "[RTX-Compatibility][UE3-AlbedoSelection] ps=0x", std::hex, inferredPsHash,
          " key=0x", selectionCacheKey, std::dec,
          " chosen=[", stageName(chosenStages[0]), ",", stageName(chosenStages[1]), "]",
          albedoSelectionLog.empty() ? "\n  (no scoreable candidates)" : albedoSelectionLog.c_str()));
      }

      uint32_t textureID = 0;
      for (uint32_t stageIdx = 0; stageIdx < LegacyMaterialData::kMaxSupportedTextures && textureID < LegacyMaterialData::kMaxSupportedTextures; stageIdx++) {
        const uint8_t stage = chosenStages[stageIdx];
        if (stage == kInvalidStage || stage >= SamplerCount || d3d9State().textures[stage] == nullptr)
          continue;

        D3D9CommonTexture* pTexInfo = GetCommonTexture(d3d9State().textures[stage]);
        assert(pTexInfo != nullptr);
        const XXH64_hash_t texHash = pTexInfo->GetImage()->getHash();
        const XXH64_hash_t texDescHash =
          pTexInfo->GetImage() != nullptr
            ? pTexInfo->GetImage()->getDescriptorHash()
            : kEmptyHash;
        const bool allowHashlessCubemap =
          pTexInfo->GetType() == D3DRTYPE_CUBETEXTURE && stage == strictCubemapFallbackStage;

        if (texHash == kEmptyHash && !allowHashlessCubemap)
          continue;

        if (textureID == 0)
          firstStage = stage;

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

        const bool srgb = d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1;
        Rc<DxvkImageView> sampleView = getRemixSampleView(pTexInfo, srgb);
        if (sampleView == nullptr)
          continue;
        m_activeDrawCallState.materialData.colorTextures[textureID] = TextureRef(sampleView);
        m_activeDrawCallState.materialData.samplers[textureID] = sampler;
        selectedUe3MovieTexture |=
          isUe3MovieTextureDescHash(texDescHash) ||
          (stage < caps::MaxTexturesPS &&
           inferredPsEntry != nullptr &&
           (inferredPsEntry->samplers[stage].semanticFlags & kPsSamplerSemanticMovieTexture) != 0 &&
           pTexInfo->IsRenderTarget());
        if (textureID == 0)
          m_activeDrawCallState.materialData.colorTextureIsSrgb = srgb;

        auto shaderSampler = RemapStateSamplerShader(stage);
        m_activeDrawCallState.materialData.colorTextureSlot[textureID] = computeResourceSlotId(shaderSampler.first, DxsoBindingType::Image, uint32_t(shaderSampler.second));
        ++textureID;
      }

      if (textureID == 0 &&
          strictCubemapFallbackStage != kInvalidStage &&
          strictCubemapFallbackStage < SamplerCount &&
          d3d9State().textures[strictCubemapFallbackStage] != nullptr) {
        D3D9CommonTexture* pTexInfo = GetCommonTexture(d3d9State().textures[strictCubemapFallbackStage]);
        if (pTexInfo != nullptr && pTexInfo->GetImage() != nullptr) {
          firstStage = strictCubemapFallbackStage;

          D3D9SamplerKey key = m_parent->CreateSamplerKey(firstStage);
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

          const bool srgb = d3d9State().samplerStates[firstStage][D3DSAMP_SRGBTEXTURE] & 0x1;
          Rc<DxvkImageView> sampleView = getRemixSampleView(pTexInfo, srgb);
          if (sampleView != nullptr) {
            m_activeDrawCallState.materialData.colorTextures[0] = TextureRef(sampleView);
            m_activeDrawCallState.materialData.samplers[0] = sampler;
            selectedUe3MovieTexture |= isUe3MovieTextureDescHash(pTexInfo->GetImage()->getDescriptorHash());
            m_activeDrawCallState.materialData.colorTextureIsSrgb = srgb;

            auto shaderSampler = RemapStateSamplerShader(firstStage);
            m_activeDrawCallState.materialData.colorTextureSlot[0] =
              computeResourceSlotId(shaderSampler.first, DxsoBindingType::Image, uint32_t(shaderSampler.second));
          }
        }
      }

      if (m_frameOptions.ue3EngineMode && !m_activeDrawCallState.materialData.colorTextures[0].isValid()) {
        logUe3UnboundAlbedoOnce(inferredPs, inferredPsHash, usedSamplerMask, usedTextureMask, inferredPsEntry);
      }
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
      // UE3 MaterialInstanceConstant compat - deterministic child-level material identity:
      //   PS bytecode hash -> ordered material texture set -> material constants
      // this differentiates instances that share a parent but override TextureParameterValues
      // in any material sampler, VectorParameterValues/ScalarParameterValues (e.g. DiffuseColor),
      // or StaticSwitchParameterValues (different bytecode)
      // must run before setupCategoriesForTexture so category lookups use the full material hash
      if constexpr (!FixedFunction) {
        if (m_frameOptions.ue3EngineMode && m_parent->UseProgrammablePS() && d3d9State().pixelShader.ptr() != nullptr) {
          ScopedCpuProfileZoneN("UE3 material identity");
          const D3D9CommonShader* psCommonShader = d3d9State().pixelShader->GetCommonShader();
          const auto& bytecode = psCommonShader->GetBytecode();
          const XXH64_hash_t psHash = psCommonShader->GetBytecodeHash();
          if (psHash != 0) {
            const Ue3PsMaterialIdentityInfo& identityInfo = getOrParseUe3PsMaterialIdentityInfo(
              psHash, bytecode, psCommonShader, m_frameOptions.ue3MicVolatileConstantDetection);

            // UE3 recompiles the same material's base pass per lightmap policy, so a bytecode
            // hash - and everything seeded by it - varies with DirectionalLightmaps and
            // TdBicubicFiltering. Seed every UE3 material with the canonical CTAB signature
            // instead. This deliberately covers the permutations that carry no lightmap symbols
            // at all: FNoLightMapPolicy compiles the same material for movable geometry,
            // translucency and the unlit viewmode, and gating on lightmap symbols would have
            // given those draws a bytecode-seeded identity that could never match their
            // lightmapped selves.
            const bool useInvariantShaderIdentity =
              m_frameOptions.ue3EngineMode &&
              identityInfo.canonicalShaderSignature != kEmptyHash;
            const XXH64_hash_t shaderIdentitySeed =
              useInvariantShaderIdentity ? identityInfo.canonicalShaderSignature : psHash;

            m_activeDrawCallState.materialData.setPixelShaderHashForMaterialInstance(shaderIdentitySeed);

            // ordered (sampler key, image hash) set over every texture bound to a material
            // sampler (CTAB names Texture2D_* / TextureCube_*) - catches TextureParameterValues
            // overridden in any material sampler, not just the chosen primary color texture.
            // For invariant-identity shaders the sampler key is the CTAB name hash, not the
            // register: lightmap sampler counts shift register assignments between
            // permutations while the material's own sampler names stay fixed.
            const bool logMicHash = m_frameOptions.ue3LogMaterialInstanceHash;
            if (logMicHash && !identityInfo.identitySummary.empty()) {
              static fast_unordered_set s_loggedIdentitySummaries;
              if (s_loggedIdentitySummaries.insert(psHash).second) {
                Logger::info(str::format(
                  "[RTX-Ue3Identity] ps=0x", std::hex, psHash,
                  " seed=0x", identityInfo.canonicalShaderSignature, std::dec,
                  " ", identityInfo.identitySummary));
              }
            }
            std::string micTextureListLog;
            // Replacement identity drift diagnostics: record the (register, image hash, RT flag)
            // tuples that feed textureSetHash so tier drift can be attributed per sampler.
            Ue3MicIdentitySample::SamplerRecord micDiagSamplers[kMicDriftMaxTrackedSamplers];
            uint32_t micDiagSamplerCount = 0;
            const BoundTextureSnapshot& identityBoundTextures = ensureBoundTextureSnapshot();

            XXH64_hash_t textureSetHash = kEmptyHash;
            if (identityInfo.materialSamplerMask != 0) {
              XXH3_state_t* const state = getThreadLocalXxh3State();
              if (state != nullptr) {
                XXH3_64bits_reset(state);
                bool anyTextureHashed = false;
                const BoundTextureSnapshot& boundTextures = identityBoundTextures;
                auto hashMaterialSamplerTexture = [&](const void* samplerKey, const size_t samplerKeySize, const uint32_t samplerRegister, const char* samplerLogName) -> XXH64_hash_t {
                  if (samplerRegister >= SamplerCount || (boundTextures.mask & (1u << samplerRegister)) == 0)
                    return kEmptyHash;
                  const BoundTextureSnapshotEntry& entry = boundTextures.entries[samplerRegister];
                  if (!entry.hasImage)
                    return kEmptyHash;
                  const XXH64_hash_t imageHash = entry.imageHash;
                  if (imageHash == kEmptyHash)
                    return kEmptyHash; // hashless (e.g. render target bound as a material texture)
                  if (m_frameOptions.ue3MicExcludeRenderTargetsFromIdentity && entry.isRenderTarget) {
                    // RT image hashes change on every recreation (respawn/checkpoint/level
                    // load) and would re-mint the identity each time; treat as hashless.
                    if (m_frameOptions.logReplacementResolution || m_frameOptions.ue3LogMaterialInstanceHash) {
                      static fast_unordered_set s_loggedRtIdentityExclusions;
                      const XXH64_hash_t exclusionLogKey = XXH3_64bits_withSeed(&samplerRegister, sizeof(samplerRegister), psHash);
                      if (s_loggedRtIdentityExclusions.insert(exclusionLogKey).second) {
                        Logger::info(str::format(
                          "[RTX-Compatibility][UE3-MIC] Excluded render-target image 0x", std::hex, imageHash,
                          " (RT descriptor hash 0x", entry.rtDescriptorHash,
                          ") at material sampler s", std::dec, samplerRegister,
                          " of pixel shader 0x", std::hex, psHash, std::dec,
                          " from material identity (rtx.d3d9.ue3MicExcludeRenderTargetsFromIdentity)."));
                      }
                    }
                    return kEmptyHash;
                  }
                  if (!m_frameOptions.ue3MicIdentityExcludedTextureDescHashes->empty() &&
                      lookupHash(*m_frameOptions.ue3MicIdentityExcludedTextureDescHashes, entry.descriptorHash)) {
                    // A texture whose contents are not reproducible (engine-composited, or
                    // re-uploaded with a different animation frame every level load) still has a
                    // reproducible *shape*, so the sampler is identified by its descriptor hash
                    // rather than dropped. Dropping it would be self-defeating on a material
                    // whose only sampler this is: an empty texture set falls back to the primary
                    // colour texture's image hash, which is the very value being excluded.
                    if (m_frameOptions.logReplacementResolution || m_frameOptions.ue3LogMaterialInstanceHash) {
                      static fast_unordered_set s_loggedDescIdentityExclusions;
                      const XXH64_hash_t exclusionLogKey = XXH3_64bits_withSeed(&samplerRegister, sizeof(samplerRegister), psHash);
                      if (s_loggedDescIdentityExclusions.insert(exclusionLogKey).second) {
                        Logger::info(str::format(
                          "[RTX-Compatibility][UE3-MIC] Identifying material sampler s", std::dec, samplerRegister,
                          " of pixel shader 0x", std::hex, psHash,
                          " by its descriptor hash 0x", entry.descriptorHash,
                          " instead of its image hash 0x", imageHash, std::dec,
                          " (rtx.d3d9.ue3MicIdentityExcludedTextureDescHashes)."));
                      }
                    }
                    XXH3_64bits_update(state, samplerKey, samplerKeySize);
                    XXH3_64bits_update(state, &entry.descriptorHash, sizeof(entry.descriptorHash));
                    anyTextureHashed = true;
                    if (micDiagSamplerCount < kMicDriftMaxTrackedSamplers) {
                      micDiagSamplers[micDiagSamplerCount++] = Ue3MicIdentitySample::SamplerRecord {
                        uint8_t(samplerRegister), entry.isRenderTarget, entry.descriptorHash, entry.descriptorHash };
                    }
                    if (logMicHash) {
                      micTextureListLog += str::format(
                        micTextureListLog.empty() ? "s" : ",s", samplerRegister,
                        samplerLogName != nullptr ? str::format("(", samplerLogName, ")") : std::string(),
                        ":desc0x", std::hex, entry.descriptorHash, std::dec);
                    }
                    return kEmptyHash;
                  }
                  XXH3_64bits_update(state, samplerKey, samplerKeySize);
                  XXH3_64bits_update(state, &imageHash, sizeof(imageHash));
                  anyTextureHashed = true;
                  if (micDiagSamplerCount < kMicDriftMaxTrackedSamplers) {
                    micDiagSamplers[micDiagSamplerCount++] = Ue3MicIdentitySample::SamplerRecord {
                      uint8_t(samplerRegister), entry.isRenderTarget, imageHash, entry.descriptorHash };
                  }
                  if (logMicHash) {
                    micTextureListLog += str::format(
                      micTextureListLog.empty() ? "s" : ",s", samplerRegister,
                      samplerLogName != nullptr ? str::format("(", samplerLogName, ")") : std::string(),
                      ":0x", std::hex, imageHash, "(desc:0x", entry.descriptorHash, ")", std::dec);
                  }
                  return imageHash;
                };
                if (useInvariantShaderIdentity) {
                  // name-keyed: register assignments shift between lightmap policy permutations
                  for (const auto& [samplerName, samplerNameKey, samplerRegister] : identityInfo.materialSamplersByNameOrder) {
                    if (samplerRegister < caps::MaxTexturesPS) {
                      hashMaterialSamplerTexture(&samplerNameKey, sizeof(samplerNameKey), samplerRegister, samplerName.c_str());
                    }
                  }
                } else {
                  // (uint32_t register, image hash) pairs - the historical stream, preserved
                  // so non-lightmap material hashes stay identical to prior builds
                  for (uint32_t s = 0; s < caps::MaxTexturesPS; s++) {
                    if ((identityInfo.materialSamplerMask & (1u << s)) == 0)
                      continue;
                    hashMaterialSamplerTexture(&s, sizeof(s), s, nullptr);
                  }
                }
                if (anyTextureHashed)
                  textureSetHash = XXH3_64bits_digest(state);
              }
            }
            m_activeDrawCallState.materialData.setMaterialTextureSetHashForMaterialInstance(textureSetHash);
            // A canonical textureless signature already names the material; whatever albedo
            // scoring bound for display must not leak into its identity.
            const bool textureSetIsComplete = useInvariantShaderIdentity && identityInfo.materialSamplerMask == 0;
            m_activeDrawCallState.materialData.setMaterialTextureSetIsComplete(textureSetIsComplete);

            // Constants tier. Frame-varying registers are already gone by this point: the parse
            // left them out of identityInfo (rtx.d3d9.ue3MicVolatileConstantDetection), so what
            // remains is hashed unconditionally and the result cannot change mid-session.
            //
            // Both manual escape hatches are keyed on values that outlive a session. The shader
            // one honours the raw bytecode hash (for existing configs) as well as the identity
            // seed. The per-material one is keyed on textureSetShaderHash, the same value the
            // second replacement lookup tier uses, so an exclusion and an anchor are named alike.
            const XXH64_hash_t identityTextureSetHash =
              (textureSetHash != kEmptyHash || textureSetIsComplete)
                ? textureSetHash
                : m_activeDrawCallState.materialData.getColorTexture().getImageHash();
            const XXH64_hash_t textureSetShaderHash =
              XXH3_64bits_withSeed(&identityTextureSetHash, sizeof(identityTextureSetHash), shaderIdentitySeed);
            const bool constantsExcluded =
              // The tier that cannot be made lightmap-policy independent; opt-in only.
              !m_frameOptions.ue3MicConstantIdentity ||
              lookupHash(*m_frameOptions.ue3MicConstantIdentityExcludedShaders, psHash) ||
              (useInvariantShaderIdentity && lookupHash(*m_frameOptions.ue3MicConstantIdentityExcludedShaders, shaderIdentitySeed)) ||
              lookupHash(*m_frameOptions.ue3MicConstantIdentityExcludedMaterials, textureSetShaderHash);
            // Invariant-identity shaders hash constants by uniform name and leading register:
            // lightmap policy permutations shift uniform registers and trim per-permutation
            // unreferenced elements, so the raw register-range stream is not comparable
            // across permutations. Other shaders keep the historical register-range stream
            // so their hashes stay identical to prior builds.
            XXH64_hash_t psConstsHash = kEmptyHash;
            if (!constantsExcluded) {
              psConstsHash = useInvariantShaderIdentity
                ? hashUe3MaterialConstantsByNameOrder(d3d9State().psConsts.fConsts, identityInfo.namedUniformFirstRegistersByNameOrder)
                : hashUe3MaterialConstants(d3d9State().psConsts.fConsts, identityInfo.constRanges);
            }
            m_activeDrawCallState.materialData.setPixelShaderConstantsHashForMaterialInstance(psConstsHash);

            // Report - never act on - a family that still mints more than one identity, so a
            // frame-varying register the bytecode could not see announces itself instead of
            // quietly breaking the replacements anchored on it.
            if (m_frameOptions.ue3ReportMicIdentityChurn && !constantsExcluded && psConstsHash != kEmptyHash) {
              reportUe3MicIdentityChurnOnce(
                textureSetShaderHash, psHash, shaderIdentitySeed,
                m_activeDrawCallState.materialData.getColorTexture().getImageHash(),
                psConstsHash, d3d9State().psConsts.fConsts, identityInfo, useInvariantShaderIdentity, bytecode);
            }

            // Replacement identity drift diagnostics: attribute a changed material hash
            // to the tier that moved and flag RT-poisoned identities.
            const bool replacementDiagActive =
              m_frameOptions.logReplacementResolution ||
              (m_frameOptions.replacementDebugHashes != nullptr && !m_frameOptions.replacementDebugHashes->empty());
            if (replacementDiagActive) {
              m_activeDrawCallState.materialData.updateCachedHash();
              const XXH64_hash_t materialHash = m_activeDrawCallState.materialData.getHash();
              const XXH64_hash_t primaryTexHash = m_activeDrawCallState.materialData.getColorTexture().getImageHash();

              bool tracked = false;
              if (m_frameOptions.replacementDebugHashes != nullptr && !m_frameOptions.replacementDebugHashes->empty()) {
                const fast_unordered_set& dbg = *m_frameOptions.replacementDebugHashes;
                tracked = lookupHash(dbg, primaryTexHash) || lookupHash(dbg, materialHash) || lookupHash(dbg, textureSetShaderHash);
                for (uint32_t i = 0; !tracked && i < micDiagSamplerCount; i++) {
                  tracked = lookupHash(dbg, micDiagSamplers[i].imageHash);
                }
              }

              if (m_frameOptions.logReplacementResolution || tracked) {
                const XXH64_hash_t familyKey = XXH3_64bits_withSeed(&primaryTexHash, sizeof(primaryTexHash), shaderIdentitySeed);

                // Snapshot the constant registers feeding the identity so drift can name the
                // register(s) whose values moved.
                Ue3MicIdentitySample::ConstantRecord curConstants[kMicDriftMaxTrackedConstants];
                uint32_t curConstantCount = 0;
                if (!constantsExcluded && psConstsHash != kEmptyHash) {
                  curConstantCount = snapshotUe3MicIdentityConstants(
                    d3d9State().psConsts.fConsts, identityInfo, useInvariantShaderIdentity,
                    curConstants, kMicDriftMaxTrackedConstants);
                }

                Ue3MicIdentitySample& prev = s_ue3MicIdentityByFamily[familyKey];
                if (prev.valid && prev.materialHash != materialHash &&
                    prev.driftLogsEmitted < (tracked ? kMicDriftMaxLogsPerTrackedFamily : kMicDriftMaxLogsPerFamily)) {
                  ++prev.driftLogsEmitted;
                  std::string detail;
                  if (prev.textureSetHash != textureSetHash) {
                    detail += str::format("\n  textureSet 0x", std::hex, prev.textureSetHash, " -> 0x", textureSetHash, std::dec, ":");
                    for (uint32_t i = 0; i < micDiagSamplerCount; i++) {
                      const Ue3MicIdentitySample::SamplerRecord& cur = micDiagSamplers[i];
                      const Ue3MicIdentitySample::SamplerRecord* old = nullptr;
                      for (uint32_t j = 0; j < prev.samplerCount; j++) {
                        if (prev.samplers[j].reg == cur.reg) {
                          old = &prev.samplers[j];
                          break;
                        }
                      }
                      if (old == nullptr) {
                        detail += str::format(" s", uint32_t(cur.reg), " added=0x", std::hex, cur.imageHash,
                                              "(desc:0x", cur.descriptorHash, ")", std::dec, cur.isRenderTarget ? "(RT)" : "");
                      } else if (old->imageHash != cur.imageHash) {
                        detail += str::format(" s", uint32_t(cur.reg), " 0x", std::hex, old->imageHash, "->0x", cur.imageHash,
                                              "(desc:0x", cur.descriptorHash, ")", std::dec, cur.isRenderTarget ? "(RT)" : "");
                      }
                    }
                    for (uint32_t j = 0; j < prev.samplerCount; j++) {
                      bool stillPresent = false;
                      for (uint32_t i = 0; i < micDiagSamplerCount; i++) {
                        if (micDiagSamplers[i].reg == prev.samplers[j].reg) {
                          stillPresent = true;
                          break;
                        }
                      }
                      if (!stillPresent) {
                        detail += str::format(" s", uint32_t(prev.samplers[j].reg), " removed=0x", std::hex, prev.samplers[j].imageHash, std::dec,
                                              prev.samplers[j].isRenderTarget ? "(RT)" : "");
                      }
                    }
                  }
                  if (prev.constantsExcluded != constantsExcluded) {
                    detail += str::format("\n  constantsExcluded ", prev.constantsExcluded ? 1 : 0, " -> ", constantsExcluded ? 1 : 0);
                  }
                  if (prev.constantsHash != psConstsHash) {
                    detail += str::format("\n  consts 0x", std::hex, prev.constantsHash, " -> 0x", psConstsHash, std::dec, ":");
                    for (uint32_t i = 0; i < curConstantCount; i++) {
                      const Ue3MicIdentitySample::ConstantRecord& cur = curConstants[i];
                      for (uint32_t j = 0; j < prev.constantCount; j++) {
                        const Ue3MicIdentitySample::ConstantRecord& old = prev.constants[j];
                        if (old.reg == cur.reg) {
                          if (old.value.x != cur.value.x || old.value.y != cur.value.y ||
                              old.value.z != cur.value.z || old.value.w != cur.value.w) {
                            detail += str::format(" c", uint32_t(cur.reg),
                                                  " (", old.value.x, ",", old.value.y, ",", old.value.z, ",", old.value.w,
                                                  ")->(", cur.value.x, ",", cur.value.y, ",", cur.value.z, ",", cur.value.w, ")");
                          }
                          break;
                        }
                      }
                    }
                  }
                  Logger::warn(str::format(
                    "[RTX-MicDrift] Material identity changed for family tex=0x", std::hex, primaryTexHash,
                    " seed=0x", shaderIdentitySeed,
                    ": materialHash 0x", prev.materialHash, " -> 0x", materialHash,
                    " (textureSet+shader tier 0x", textureSetShaderHash, ")", std::dec,
                    detail.empty() ? "\n  (no attributable tier diff captured)" : detail.c_str(),
                    "\n  Replacements anchored on the previous hash no longer match this draw."));
                }

                prev.valid = true;
                prev.materialHash = materialHash;
                prev.textureSetHash = textureSetHash;
                prev.constantsHash = psConstsHash;
                prev.constantsExcluded = constantsExcluded;
                prev.samplerCount = micDiagSamplerCount;
                for (uint32_t i = 0; i < micDiagSamplerCount; i++) {
                  prev.samplers[i] = micDiagSamplers[i];
                }
                prev.constantCount = curConstantCount;
                for (uint32_t i = 0; i < curConstantCount; i++) {
                  prev.constants[i] = curConstants[i];
                }

                // RT-poisoning sweep: a render-target-backed image hash inside the identity
                // makes it unstable across RT recreation (respawn / level load).
                for (uint32_t i = 0; i < micDiagSamplerCount; i++) {
                  if (micDiagSamplers[i].isRenderTarget) {
                    if (s_ue3MicRtPoisonWarnedFamilies.insert(familyKey).second) {
                      Logger::warn(str::format(
                        "[RTX-MicRtPoisoning] Material identity for family tex=0x", std::hex, primaryTexHash,
                        " seed=0x", shaderIdentitySeed,
                        " includes render-target image hash 0x", micDiagSamplers[i].imageHash,
                        " at material sampler s", std::dec, uint32_t(micDiagSamplers[i].reg),
                        std::hex, " (stable RT descriptor hash 0x", micDiagSamplers[i].descriptorHash,
                        "): materialHash 0x", materialHash, std::dec,
                        " will change whenever the game recreates this render target (respawn/level load),"
                        " breaking replacements anchored on it."));
                    }
                    break;
                  }
                }
              }
            }

            // Constant-color materials: the surface color lives in a UniformVector_*
            // register. Scan the identity's kept vectors in CTAB-name order - the same list in
            // every lightmap-policy compile, whereas register order shifts per permutation and a
            // specular-only vector declared by one compile could otherwise win - and take the
            // first value that is plausibly a color (finite, non-black, LDR-ish). The lowest
            // name frequently holds a zero vector (fades, unused parameters).
            if (identityInfo.materialSamplerMask == 0 &&
                !identityInfo.uniformVectorRegisters.empty() &&
                !m_activeDrawCallState.materialData.colorTextures[0].isValid()) {
              const float tintGain = m_frameOptions.ue3ConstantAlbedoTintGain;
              auto tryConstantAlbedo = [&](const uint32_t reg) {
                if (reg >= caps::MaxFloatConstantsPS)
                  return false;
                // A highlight colour holds its value whether the highlight is on or not, so taking
                // it would leave the surface permanently tinted.
                if (std::find(identityInfo.highlightColorRegisters.begin(), identityInfo.highlightColorRegisters.end(), reg) !=
                    identityInfo.highlightColorRegisters.end())
                  return false;
                const Vector4& uniformColor = d3d9State().psConsts.fConsts[reg];
                if (!std::isfinite(uniformColor.x) || !std::isfinite(uniformColor.y) ||
                    !std::isfinite(uniformColor.z) || !std::isfinite(uniformColor.w))
                  return false;
                const float maxComp = std::max({ uniformColor.x, uniformColor.y, uniformColor.z });
                const float minComp = std::min({ uniformColor.x, uniformColor.y, uniformColor.z });
                // reject blacks/negatives (not visible albedo) and HDR-scale values (intensities).
                // Rejecting black also lets a register holding the material's switched-off colour
                // fall through to whichever one holds its real tint.
                if (minComp < 0.0f || maxComp <= 0.01f || maxComp > 8.0f)
                  return false;

                Vector4 albedo = uniformColor;
                if (tintGain > 0.0f) {
                  // The register holds a tint UE3 multiplied against baked lighting for brightness,
                  // so raw it is near-black once the lightmap is gone. Blend the legacy constant
                  // towards the fully saturated hue by the register's own strength, which keeps a
                  // ramping tint continuous instead of stepping away from the unlit surface.
                  const Vector3 base = LegacyMaterialDefaults::albedoConstant();
                  const float weight = std::min(maxComp * tintGain, 1.0f);
                  albedo.x = base.x + (uniformColor.x / maxComp - base.x) * weight;
                  albedo.y = base.y + (uniformColor.y / maxComp - base.y) * weight;
                  albedo.z = base.z + (uniformColor.z / maxComp - base.z) * weight;
                }
                m_activeDrawCallState.materialData.ue3ConstantAlbedo = albedo;
                m_activeDrawCallState.materialData.hasUe3ConstantAlbedo = true;
                return true;
              };
              if (!identityInfo.namedUniformFirstRegistersByNameOrder.empty()) {
                for (const auto& [uniformNameKey, uniformRegister] : identityInfo.namedUniformFirstRegistersByNameOrder) {
                  if (tryConstantAlbedo(uniformRegister))
                    break;
                }
              } else {
                for (const uint32_t reg : identityInfo.uniformVectorRegisters) {
                  if (tryConstantAlbedo(reg))
                    break;
                }
              }
            }

            // Runner Vision: the game fades a strength parameter up on the surfaces it highlights
            // and tints them through constants Remix never evaluates. Carried per surface rather
            // than in the material, so the fade reaches the renderer frame by frame without
            // re-minting the material or moving its identity.
            if (m_frameOptions.ue3HighlightTints) {
              const bool logHighlight = m_frameOptions.ue3LogHighlightTints;
              if (logHighlight)
                logUe3HighlightPairsOnce(psHash, shaderIdentitySeed, bytecode, identityInfo);

              LegacyMaterialData& materialData = m_activeDrawCallState.materialData;
              const auto& pairs = identityInfo.highlightPairs;
              const Vector4* fConsts = d3d9State().psConsts.fConsts;
              const float glowIntensity = std::max(m_frameOptions.ue3HighlightGlowIntensity, 0.0f);

              // A glow-only pair glows one of the material's textures. It is bound whatever the
              // strength, so the material stays the same through a highlight.
              const auto glowPair = std::find_if(pairs.begin(), pairs.end(), [](const auto& pair) { return pair.tint.glowOnly; });
              if (glowPair != pairs.end() && glowIntensity > 0.0f) {
                const uint32_t stage = uint32_t(glowPair->tint.glowSampler);
                D3D9CommonTexture* const glowTexture =
                  stage < caps::MaxTexturesPS ? GetCommonTexture(d3d9State().textures[stage]) : nullptr;
                if (glowTexture != nullptr && glowTexture->GetImage() != nullptr &&
                    glowTexture->GetImage()->getHash() != kEmptyHash) {
                  if (const Rc<DxvkImageView> view = getRemixSampleView(glowTexture, false); view != nullptr) {
                    materialData.ue3HighlightGlowTexture = TextureRef(view);
                    materialData.ue3HighlightGlowTextureIsSrgb = d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1;
                    materialData.ue3HighlightGlowTextureChannel = glowPair->tint.glowComponent;
                  }
                }
              }

              // Nearly every draw holds its strengths at 0, which leaves the surface as it is.
              const bool active = std::any_of(pairs.begin(), pairs.end(), [&](const auto& pair) {
                return pair.tint.scalarReg < caps::MaxFloatConstantsPS && fConsts[pair.tint.scalarReg].x > 0.0f;
              });
              if (active) {
                materialData.updateCachedHash();
                const fast_unordered_set& excluded = *m_frameOptions.ue3HighlightTintExcludedMaterials;
                const XXH64_hash_t colorTextureHash = materialData.getColorTexture().getImageHash();
                const bool isExcluded = !excluded.empty() &&
                  (lookupHash(excluded, materialData.getHash()) || lookupHash(excluded, textureSetShaderHash) ||
                   lookupHash(excluded, colorTextureHash));
                if (!isExcluded) {
                  Ue3HighlightTintDraw draw;
                  draw.fConsts = fConsts;
                  draw.materialHash = materialData.getHash();
                  draw.requireMotion = m_frameOptions.ue3HighlightTintRequireMotion;
                  if (draw.requireMotion) {
                    const Matrix4& objectToWorld = m_activeDrawCallState.transformData.objectToWorld;
                    draw.objectHash = XXH3_64bits_withSeed(&objectToWorld, sizeof(objectToWorld), draw.materialHash);
                  }
                  draw.glowIntensity = glowIntensity;
                  draw.log = logHighlight;

                  static fast_unordered_set s_loggedHighlightMaterials;
                  const bool logApplied = logHighlight && !lookupHash(s_loggedHighlightMaterials, materialData.getHash());
                  std::string appliedLog;
                  const bool applied = evaluateUe3HighlightTints(identityInfo, draw, materialData.ue3HighlightTint,
                                                                 materialData.ue3HighlightGlow, logApplied ? &appliedLog : nullptr);
                  if (applied && logApplied) {
                    s_loggedHighlightMaterials.insert(materialData.getHash());
                    const Vector3& tint = materialData.ue3HighlightTint;
                    const Vector3& glow = materialData.ue3HighlightGlow;
                    const TextureRef& glowTexture = materialData.ue3HighlightGlowTexture;
                    Logger::info(str::format(
                      "[RTX-Compatibility][UE3-Highlight] Tint applied: materialHash=0x", std::hex, materialData.getHash(),
                      " textureSetShader=0x", textureSetShaderHash, " texture=0x", colorTextureHash, " ps=0x", psHash,
                      glowTexture.isValid() ? str::format(" glowTexture=0x", std::hex, glowTexture.getImageHash()) : std::string(),
                      std::dec, " tint=(", tint.x, ",", tint.y, ",", tint.z, ") glow=(", glow.x, ",", glow.y, ",", glow.z, ") ",
                      appliedLog));
                  }
                }
              }
            }

            const Vector3& forcedHighlightTint = m_frameOptions.ue3HighlightDebugForceTint;
            if (forcedHighlightTint.x != 1.0f || forcedHighlightTint.y != 1.0f || forcedHighlightTint.z != 1.0f)
              m_activeDrawCallState.materialData.ue3HighlightTint = forcedHighlightTint;

            // Opacity-driven fades: UE3 hands its particle colour to the shader as an interpolant Remix
            // never treats as a vertex colour, and fades draws through material constants it never
            // evaluates. Both come from the shader analysis and are carried per draw, like the highlight.
            if (m_frameOptions.ue3ParticleVertexColor || m_frameOptions.ue3MaterialFades)
              applyUe3MaterialFades(psHash, bytecode, psCommonShader, textureSetShaderHash);

            const float forcedCoverage = m_frameOptions.ue3MaterialFadeDebugForceCoverage;
            if (forcedCoverage >= 0.0f && m_activeDrawCallState.materialData.blendMode.enableBlending)
              m_activeDrawCallState.materialData.ue3FadeCoverage = std::min(forcedCoverage, 1.0f);

            if (logMicHash) {
              m_activeDrawCallState.materialData.updateCachedHash();
              logUe3MaterialInstanceHashBreakdownOnce(
                m_activeDrawCallState.materialData.getHash(), psHash, shaderIdentitySeed, textureSetHash,
                textureSetShaderHash, psConstsHash, identityInfo, constantsExcluded, micTextureListLog);
            }

          }
        }
      }
      m_activeDrawCallState.materialData.updateCachedHash();

      // Texture-less materials have no presence in the texture selection UI; register
      // their material hash as an entry (white thumbnail) so they can be clicked/tagged.
      // Category and replacement lookups already accept material hashes.
      if constexpr (!FixedFunction) {
        if (ue3MicIdentityAvailable &&
            !m_activeDrawCallState.materialData.colorTextures[0].isValid()) {
          const XXH64_hash_t texturelessMaterialHash = m_activeDrawCallState.materialData.getHash();
          static fast_unordered_set s_registeredTexturelessMaterials;
          if (texturelessMaterialHash != kEmptyHash &&
              s_registeredTexturelessMaterials.insert(texturelessMaterialHash).second) {
            m_parent->EmitCs([texturelessMaterialHash](DxvkContext* ctx) {
              const Rc<DxvkImageView> whiteView =
                static_cast<RtxContext*>(ctx)->getResourceManager().getWhiteTexture(ctx);
              if (whiteView != nullptr) {
                ImGUI::AddTexture(texturelessMaterialHash, whiteView, ImGUI::kTextureFlagsDefault);
              }
            });
          }
        }
      }

      m_activeDrawCallState.setupCategoriesForTexture();

      // Track the material hash before checking if it should be ignored
      // This ensures we track all materials sent by the game, not just the ones that are actually rendered.
      const XXH64_hash_t textureHash = m_activeDrawCallState.materialData.getColorTexture().getImageHash();
      const XXH64_hash_t materialHash = m_activeDrawCallState.materialData.getHash();
      bool usesMovieTexture = selectedUe3MovieTexture;
      for (uint32_t i = 0; i < LegacyMaterialData::kMaxSupportedTextures; i++) {
        if (m_activeDrawCallState.materialData.colorTextures[i].isValid()) {
          const DxvkImageView* imageView = m_activeDrawCallState.materialData.colorTextures[i].getImageView();
          const XXH64_hash_t descHash =
            imageView != nullptr
              ? imageView->image()->getDescriptorHash()
              : kEmptyHash;
          if (isUe3MovieTextureDescHash(descHash)) {
            usesMovieTexture = true;
            break;
          }
        }
      }
      if (usesMovieTexture) {
        m_activeDrawCallState.setCategory(InstanceCategories::WorldUI, true);
        if (Logger::logLevel() <= LogLevel::Debug) {
          static fast_unordered_set s_loggedMovieSurfaceMaterials;
          if (s_loggedMovieSurfaceMaterials.insert(materialHash).second) {
            Logger::debug(str::format(
              "[RTX-Compatibility][UE3] Marked movie texture material as WorldUI: materialHash=0x",
              std::hex, materialHash, ", textureHash=0x", textureHash, std::dec));
          }
        }
      }

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
        if (lookupHash(outlierSet, m_activeDrawCallState.materialData.colorTextures[i].getImageHash()))
          return true;
      }

      for (uint32_t stage = 0; stage < SamplerCount; stage++) {
        if (d3d9State().textures[stage] == nullptr)
          continue;

        D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[stage]);
        if (texture == nullptr || texture->GetImage() == nullptr)
          continue;

        if (lookupHash(outlierSet, texture->GetImage()->getHash()))
          return true;
      }

      return false;
    }();

    if constexpr (!FixedFunction) {
      const bool forceIaTexcoordForOutlier = m_forceIaTexcoordForOutlier;

      auto getOrInitVsTexcoordTraceEntry = [&](const D3D9CommonShader* vs,
                                               const uint32_t outputReg,
                                               const uint8_t compU,
                                               const uint8_t compV) -> Ue3VsTexcoordTraceEntry* {
        if (vs == nullptr || outputReg == std::numeric_limits<uint32_t>::max())
          return nullptr;

        const XXH64_hash_t vsHash = vs->GetBytecodeHash();
        if (vsHash == 0)
          return nullptr;

        XXH64_hash_t traceKey = vsHash;
        auto mix = [&](const uint64_t v) {
          traceKey ^= v + 0x9E3779B97F4A7C15ull + (traceKey << 6) + (traceKey >> 2);
        };
        mix(outputReg);
        mix(compU);
        mix(uint64_t(compV) << 8);

        auto& entry = m_ue3VsTexcoordTraceCache[traceKey];
        if (!entry.initialized) {
          entry.initialized = true;
          const Ue3VsTexcoordTraceResult trace =
            traceVsOutputTexcoordToInputUsageIndex(makeDxsoShaderView(vs->GetBytecode(), vs), outputReg, compU, compV);
          entry.kind = trace.kind;
          entry.iaTexcoordIndex = trace.iaTexcoordIndex;
          entry.inputReg = trace.inputReg;
          entry.affineU = trace.affineU;
          entry.affineV = trace.affineV;
        }

        return &entry;
      };

      // resolves an affine term (imm + const*factor + const2*factor2 sum) against live
      // draw-time shader constants; an absent term resolves to its identity value
      auto resolvePsAffineTermValue = [&](const UvAffineTerm& term, const float identity, float& outValue) -> bool {
        outValue = identity;
        if (term.inexact)
          return false;
        if (!uvAffineTermPresent(term))
          return true;
        float value = term.immValid ? term.imm : 0.0f;
        if (term.constReg >= 0) {
          if (uint32_t(term.constReg) >= caps::MaxFloatConstantsPS)
            return false;
          value += d3d9State().psConsts.fConsts[uint32_t(term.constReg)][term.constComp & 0x3u] * term.factor;
        }
        if (term.constReg2 >= 0) {
          if (uint32_t(term.constReg2) >= caps::MaxFloatConstantsPS)
            return false;
          value += d3d9State().psConsts.fConsts[uint32_t(term.constReg2)][term.constComp2 & 0x3u] * term.factor2;
        }
        outValue = value;
        return std::isfinite(outValue);
      };

      auto resolveVsAffineTermValue = [&](const UvAffineTerm& term, const float identity, float& outValue) -> bool {
        outValue = identity;
        if (term.inexact)
          return false;
        if (!uvAffineTermPresent(term))
          return true;
        float value = term.immValid ? term.imm : 0.0f;
        if (term.constReg >= 0) {
          if (uint32_t(term.constReg) >= caps::MaxFloatConstantsSoftware)
            return false;
          value += d3d9State().vsConsts.fConsts[uint32_t(term.constReg)][term.constComp & 0x3u] * term.factor;
        }
        if (term.constReg2 >= 0) {
          if (uint32_t(term.constReg2) >= caps::MaxFloatConstantsSoftware)
            return false;
          value += d3d9State().vsConsts.fConsts[uint32_t(term.constReg2)][term.constComp2 & 0x3u] * term.factor2;
        }
        outValue = value;
        return std::isfinite(outValue);
      };

      if (m_frameOptions.ue3EngineMode &&
          d3d9State().pixelShader.ptr() != nullptr) {
        ScopedCpuProfileZoneN("UE3 UV resolution");
        const D3D9CommonShader* ps = inferredPs != nullptr
          ? inferredPs
          : d3d9State().pixelShader->GetCommonShader();
        XXH64_hash_t psHash = inferredPsHash;
        PsSamplerTexcoordEntry* entryPtr = inferredPsEntry;
        if (entryPtr == nullptr)
          entryPtr = getOrInitPsSamplerTexcoordEntry(ps, psHash);

        if (entryPtr != nullptr && firstStage < caps::MaxTexturesPS) {
          const auto& entry = *entryPtr;

          // A sampler whose own origin cannot be proven falls through to the fixed-function TSS
          // texcoord index below. UE3 is fully programmable and never sets that meaningfully, so
          // it is leftover device state, and a device reset restores it to the D3D9 default of
          // "stage N reads texcoord N" - the surface's UV set would change for reasons unrelated
          // to the material, presenting as the texture spontaneously rescaling.
          //
          // The shader's other material samplers shade the same surface, so where they
          // unanimously name one interpolant that agreement is proven from bytecode. Only the
          // origin is borrowed; a sibling's tiling is not this sampler's.
          PsSamplerUvOrigin borrowedUvOrigin;
          bool uvOriginBorrowed = false;
          if (!entry.samplerUvOrigin[firstStage].originValid &&
              m_frameOptions.ue3EngineMode) {
            bool haveCandidate = false;
            bool candidatesConflict = false;
            uint8_t sharedSemantic = 0;
            uint8_t sharedCompU = 0;
            uint8_t sharedCompV = 1;
            for (uint32_t s = 0; s < caps::MaxTexturesPS; s++) {
              if (s == firstStage || s >= SamplerCount || d3d9State().textures[s] == nullptr)
                continue;
              const PsSamplerUvOrigin& other = entry.samplerUvOrigin[s];
              if (!other.originValid || !other.sitesAgree)
                continue;
              // Lightmap and engine buffers legitimately read their own UV set, so they must
              // not vote on the material's.
              const uint8_t semanticFlags = entry.samplers[s].semanticFlags;
              if ((semanticFlags & (kPsSamplerSemanticLightmap | kPsSamplerSemanticEngineAuxiliary)) != 0)
                continue;
              if (!haveCandidate) {
                haveCandidate = true;
                sharedSemantic = other.semanticIndex;
                sharedCompU = other.compU;
                sharedCompV = other.compV;
              } else if (other.semanticIndex != sharedSemantic ||
                         other.compU != sharedCompU ||
                         other.compV != sharedCompV) {
                candidatesConflict = true;
                break;
              }
            }
            if (haveCandidate && !candidatesConflict) {
              borrowedUvOrigin.originValid = true;
              borrowedUvOrigin.semanticIndex = sharedSemantic;
              borrowedUvOrigin.compU = sharedCompU;
              borrowedUvOrigin.compV = sharedCompV;
              uvOriginBorrowed = true;
            }
          }

          const PsSamplerUvOrigin& uvOrigin =
            uvOriginBorrowed ? borrowedUvOrigin : entry.samplerUvOrigin[firstStage];

          // rtx.d3d9.ue3LogUvAffineDetail: one-shot per-shader dump of every sampler's UV
          // origin and affine chain, with the textures bound on this draw
          if (m_frameOptions.ue3LogUvAffineDetail && ps != nullptr && psHash != 0 &&
              m_loggedUvAffineShaderDumps.insert(psHash).second) {
            const auto& samplerNames = getUe3PsSamplerNames(psHash, ps->GetBytecode());

            std::string dump;
            for (uint32_t s = 0; s < caps::MaxTexturesPS; s++) {
              const PsSamplerUvOrigin& origin = entry.samplerUvOrigin[s];
              if (origin.validSiteCount == 0 && origin.invalidSiteCount == 0)
                continue;

              XXH64_hash_t texHash = kEmptyHash;
              uint32_t texWidth = 0;
              uint32_t texHeight = 0;
              if (s < SamplerCount && d3d9State().textures[s] != nullptr) {
                D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[s]);
                if (texture != nullptr && texture->GetImage() != nullptr) {
                  texHash = texture->GetImage()->getHash();
                  texWidth = texture->Desc()->Width;
                  texHeight = texture->Desc()->Height;
                }
              }

              const auto nameIt = samplerNames.find(s);
              dump += str::format(
                "\n  s", s, "(", nameIt != samplerNames.end() ? nameIt->second.c_str() : "?", ")",
                " tex=0x", std::hex, texHash, std::dec, " ", texWidth, "x", texHeight,
                " origin=", origin.originValid ? 1 : 0,
                " interp=", uint32_t(origin.semanticIndex),
                " comps=(", uint32_t(origin.compU), ",", uint32_t(origin.compV), ")",
                " sites=", origin.validSiteCount, "/", origin.invalidSiteCount,
                " agree=", origin.sitesAgree ? 1 : 0,
                " preferHF=", origin.preferredHighestFrequencySite ? 1 : 0,
                " exact=", origin.affineExact ? 1 : 0,
                " U:[", formatUvComponentAffine(origin.affineU), "]",
                " V:[", formatUvComponentAffine(origin.affineV), "]");
            }

            Logger::info(str::format(
              "[RTX-UV-AFFINE] shader dump ps=0x", std::hex, psHash, std::dec,
              " vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
              " albedoStage=", firstStage,
              dump.empty() ? " (no traceable sample sites)" : dump.c_str()));
          }

          const D3D9CommonShader* vs =
            (m_parent->UseProgrammableVS() && d3d9State().vertexShader.ptr() != nullptr)
              ? d3d9State().vertexShader->GetCommonShader()
              : nullptr;

          const Ue3VsTexcoordTraceEntry* traceEntry = nullptr;

          if (uvOrigin.originValid) {
            texcoordIdx = uvOrigin.semanticIndex;
            m_texcoordCompU = uvOrigin.compU;
            m_texcoordCompV = uvOrigin.compV;

            uint32_t vsTexcoordOutputReg = std::numeric_limits<uint32_t>::max();
            if (vs != nullptr) {
              vsTexcoordOutputReg = findVsTexcoordOutputRegister(vs->GetOsgn(), texcoordIdx);
              if (vsTexcoordOutputReg != std::numeric_limits<uint32_t>::max())
                traceEntry = getOrInitVsTexcoordTraceEntry(vs, vsTexcoordOutputReg, m_texcoordCompU, m_texcoordCompV);
            }

            // VS-side proof: interpolant == vsAffine(IA texcoord set k)
            float vsScaleU = 1.0f;
            float vsScaleV = 1.0f;
            float vsOffsetU = 0.0f;
            float vsOffsetV = 0.0f;
            bool vsAffineFold = false;
            int32_t provenIaIndex = -1;
            if (traceEntry != nullptr) {
              if (traceEntry->kind == Ue3VsUvTraceKind::PureMove) {
                provenIaIndex = traceEntry->iaTexcoordIndex;
              } else if (traceEntry->kind == Ue3VsUvTraceKind::AffineConst) {
                // the IA set is only usable if the VS transform resolves against live constants
                if (resolveVsAffineTermValue(traceEntry->affineU.scale, 1.0f, vsScaleU) &&
                    resolveVsAffineTermValue(traceEntry->affineU.offset, 0.0f, vsOffsetU) &&
                    resolveVsAffineTermValue(traceEntry->affineV.scale, 1.0f, vsScaleV) &&
                    resolveVsAffineTermValue(traceEntry->affineV.offset, 0.0f, vsOffsetV)) {
                  provenIaIndex = traceEntry->iaTexcoordIndex;
                  vsAffineFold = true;
                }
              }
            } else if (vs == nullptr && m_texcoordCompU == 0u && m_texcoordCompV == 1u) {
              // fixed-function vertex processing: interpolant slot j is fed from the IA set
              // selected by stage j's texcoord index state
              const uint32_t ffStage = std::min(texcoordIdx, uint32_t(caps::TextureStageCount - 1u));
              provenIaIndex = int32_t(d3d9State().textureStages[ffStage][DXVK_TSS_TEXCOORDINDEX] & 0b111);
            }

            const bool canCaptureInterpolant =
              vs != nullptr &&
              m_frameOptions.useVertexCapture &&
              vsTexcoordOutputReg != std::numeric_limits<uint32_t>::max();

            if (forceIaTexcoordForOutlier) {
              // manual override via vsTexcoordCaptureOutlierTextures
              m_uvResolutionMode = UvResolutionMode::ProvenIa;
              iaTexcoordIdx = provenIaIndex >= 0 ? uint32_t(provenIaIndex) : texcoordIdx;
              m_texcoordCompU = 0;
              m_texcoordCompV = 1;
            } else if (provenIaIndex >= 0) {
              m_uvResolutionMode = UvResolutionMode::ProvenIa;
              iaTexcoordIdx = uint32_t(provenIaIndex);
            } else if (canCaptureInterpolant) {
              // the VS-side path is procedural or unprovable: capture the exact interpolant
              // from the VS output instead of guessing an IA set
              m_uvResolutionMode = UvResolutionMode::CaptureInterpolant;
              iaTexcoordIdx = (traceEntry != nullptr && traceEntry->kind != Ue3VsUvTraceKind::Invalid)
                ? traceEntry->iaTexcoordIndex  // known base set: best backstop if capture cannot run
                : 0u;
            } else {
              m_uvResolutionMode = UvResolutionMode::LegacyTss;
              iaTexcoordIdx = texcoordIdx;
            }

            // exact UV transform: sampledUv = psAffine(interpolant),
            // interpolant = vsAffine(iaUv) when the IA path is used. Resolves to
            // U' = aU + bV + tx, V' = cU + dV + ty, where an axis-aligned transform
            // (tiling, panner) leaves the cross terms b and c at zero.
            float psA = 1.0f;
            float psB = 0.0f;
            float psC = 0.0f;
            float psD = 1.0f;
            float psTx = 0.0f;
            float psTy = 0.0f;
            const bool hasCross =
              uvOrigin.affineU.hasCross || uvOrigin.affineV.hasCross;
            bool psAffineResolved = false;

            if (!hasCross) {
              float psScaleU = 1.0f;
              float psScaleV = 1.0f;
              float psOffsetU = 0.0f;
              float psOffsetV = 0.0f;
              psAffineResolved =
                uvOrigin.affineExact &&
                resolvePsAffineTermValue(uvOrigin.affineU.scale, 1.0f, psScaleU) &&
                resolvePsAffineTermValue(uvOrigin.affineU.offset, 0.0f, psOffsetU) &&
                resolvePsAffineTermValue(uvOrigin.affineV.scale, 1.0f, psScaleV) &&
                resolvePsAffineTermValue(uvOrigin.affineV.offset, 0.0f, psOffsetV);
              if (!psAffineResolved) {
                psScaleU = 1.0f;
                psScaleV = 1.0f;
                psOffsetU = 0.0f;
                psOffsetV = 0.0f;
              }
              psA = psScaleU;
              psD = psScaleV;
              psTx = psOffsetU;
              psTy = psOffsetV;
            } else {
              auto resolveCrossRow = [&](const UvComponentAffine& aff,
                                         const bool scaleAppliesToU,
                                         float& outCoeffU, float& outCoeffV, float& outTrans) -> bool {
                if (!uvComponentAffineExact(aff))
                  return false;
                if (!aff.hasCross) {
                  float scale = 1.0f;
                  float offset = 0.0f;
                  if (!resolvePsAffineTermValue(aff.scale, 1.0f, scale) ||
                      !resolvePsAffineTermValue(aff.offset, 0.0f, offset))
                    return false;
                  outCoeffU = scaleAppliesToU ? scale : 0.0f;
                  outCoeffV = scaleAppliesToU ? 0.0f : scale;
                  outTrans = offset;
                  return true;
                }
                float scale = 1.0f;
                float cross = 0.0f;
                float offset = 0.0f;
                if (!resolvePsAffineTermValue(aff.scale, 1.0f, scale) ||
                    !resolvePsAffineTermValue(aff.cross, 0.0f, cross) ||
                    !resolvePsAffineTermValue(aff.offset, 0.0f, offset))
                  return false;
                outCoeffU = 0.0f;
                outCoeffV = 0.0f;
                auto accumulate = [&](const uint8_t comp, const float coeff) -> bool {
                  if (comp == m_texcoordCompU) {
                    outCoeffU += coeff;
                    return true;
                  }
                  if (comp == m_texcoordCompV) {
                    outCoeffV += coeff;
                    return true;
                  }
                  return false;
                };
                if (!accumulate(aff.scaleComponent, scale) ||
                    !accumulate(aff.crossComponent, cross))
                  return false;
                outTrans = offset;
                return std::isfinite(outCoeffU) && std::isfinite(outCoeffV) && std::isfinite(outTrans);
              };

              psAffineResolved =
                uvOrigin.affineExact &&
                resolveCrossRow(uvOrigin.affineU, true, psA, psB, psTx) &&
                resolveCrossRow(uvOrigin.affineV, false, psC, psD, psTy);
              if (!psAffineResolved) {
                psA = 1.0f;
                psB = 0.0f;
                psC = 0.0f;
                psD = 1.0f;
                psTx = 0.0f;
                psTy = 0.0f;
              }
            }

            float finalA = psA;
            float finalB = psB;
            float finalC = psC;
            float finalD = psD;
            float finalTx = psTx;
            float finalTy = psTy;
            if (m_uvResolutionMode == UvResolutionMode::ProvenIa && vsAffineFold) {
              finalA = psA * vsScaleU;
              finalB = psB * vsScaleV;
              finalC = psC * vsScaleU;
              finalD = psD * vsScaleV;
              finalTx = psA * vsOffsetU + psB * vsOffsetV + psTx;
              finalTy = psC * vsOffsetU + psD * vsOffsetV + psTy;
            }

            constexpr float kMinAbsScale = 1e-6f;
            constexpr float kMinAbsDeterminant = 1e-12f;  // a product of two scales, so squared
            const bool transformIsIdentity =
              finalA == 1.0f && finalB == 0.0f &&
              finalC == 0.0f && finalD == 1.0f &&
              finalTx == 0.0f && finalTy == 0.0f;
            // A rotation puts zeroes on the diagonal every quarter turn, so a mixed transform
            // is judged degenerate by its determinant rather than by its diagonal terms.
            const bool transformIsUsable =
              std::isfinite(finalA) && std::isfinite(finalB) &&
              std::isfinite(finalC) && std::isfinite(finalD) &&
              std::isfinite(finalTx) && std::isfinite(finalTy) &&
              (hasCross
                 ? std::abs(finalA * finalD - finalB * finalC) > kMinAbsDeterminant
                 : (std::abs(finalA) > kMinAbsScale && std::abs(finalD) > kMinAbsScale));

            if (!transformIsIdentity && transformIsUsable) {
              Matrix4& texXform = m_activeDrawCallState.transformData.textureTransform;
              texXform = Matrix4();
              texXform[0].x = finalA;
              texXform[1].x = finalB;
              texXform[3].x = finalTx;
              texXform[0].y = finalC;
              texXform[1].y = finalD;
              texXform[3].y = finalTy;
            }

            // rtx.d3d9.ue3LogUvAffineDetail: per-draw affine resolution outcome, logged once
            // per distinct resolved transform and capped per shader+stage
            if (m_frameOptions.ue3LogUvAffineDetail) {
              const bool applied = !transformIsIdentity && transformIsUsable;

              XXH64_hash_t detailKey = psHash;
              auto mixDetail = [&](const uint64_t v) {
                detailKey ^= v + 0x9E3779B97F4A7C15ull + (detailKey << 6) + (detailKey >> 2);
              };
              auto quantize = [](const float v) -> uint64_t {
                return std::isfinite(v) ? uint64_t(std::llround(double(v) * 1024.0)) : ~0ull;
              };
              mixDetail(firstStage);
              mixDetail(uint64_t(m_uvResolutionMode));
              mixDetail(quantize(finalA));
              mixDetail(quantize(finalB));
              mixDetail(quantize(finalC));
              mixDetail(quantize(finalD));
              mixDetail(quantize(finalTx));
              mixDetail(quantize(finalTy));
              mixDetail(uint64_t(psAffineResolved ? 1 : 0) | (uint64_t(applied ? 1 : 0) << 1));

              XXH64_hash_t capKey = psHash;
              capKey ^= firstStage + 0x9E3779B97F4A7C15ull + (capKey << 6) + (capKey >> 2);

              // check the cap before inserting the dedup key so frame-varying (panner)
              // transforms cannot grow the dedup set without bound once capped
              constexpr uint16_t kMaxAffineDetailLogsPerShaderStage = 32;
              uint16_t& logCount = m_uvAffineDetailLogCounts[capKey];
              if (logCount < kMaxAffineDetailLogsPerShaderStage &&
                  m_loggedUvAffineDetails.insert(detailKey).second) {
                ++logCount;
                // CTAB names + live values of the PS constant registers the affine references
                std::string ctabLog;
                if (ps != nullptr && psHash != 0) {
                  const auto& constNames = getUe3PsFloatConstantNames(psHash, ps->GetBytecode());
                  std::array<int32_t, 32> referencedRegs = {};
                  uint32_t referencedCount = 0;
                  uvComponentAffineCollectConstRegs(uvOrigin.affineU, referencedRegs.data(), referencedCount, uint32_t(referencedRegs.size()));
                  uvComponentAffineCollectConstRegs(uvOrigin.affineV, referencedRegs.data(), referencedCount, uint32_t(referencedRegs.size()));
                  std::sort(referencedRegs.begin(), referencedRegs.begin() + referencedCount);
                  int32_t lastLogged = -1;
                  for (uint32_t i = 0; i < referencedCount; i++) {
                    const int32_t reg = referencedRegs[i];
                    if (reg < 0 || reg == lastLogged || uint32_t(reg) >= caps::MaxFloatConstantsPS)
                      continue;
                    lastLogged = reg;
                    const auto nameIt = constNames.find(uint32_t(reg));
                    const Vector4& value = d3d9State().psConsts.fConsts[uint32_t(reg)];
                    ctabLog += str::format(
                      ctabLog.empty() ? "" : ", ",
                      "c", reg, "=", nameIt != constNames.end() ? nameIt->second.c_str() : "?",
                      "=(", value.x, ",", value.y, ",", value.z, ",", value.w, ")");
                  }
                }

                XXH64_hash_t stageTexHash = kEmptyHash;
                uint32_t stageTexWidth = 0;
                uint32_t stageTexHeight = 0;
                if (firstStage < SamplerCount && d3d9State().textures[firstStage] != nullptr) {
                  D3D9CommonTexture* stageTexture = GetCommonTexture(d3d9State().textures[firstStage]);
                  if (stageTexture != nullptr && stageTexture->GetImage() != nullptr) {
                    stageTexHash = stageTexture->GetImage()->getHash();
                    stageTexWidth = stageTexture->Desc()->Width;
                    stageTexHeight = stageTexture->Desc()->Height;
                  }
                }

                std::string vsLog;
                if (vsAffineFold) {
                  vsLog = str::format(" vs=(", vsScaleU, ",", vsScaleV, ",", vsOffsetU, ",", vsOffsetV, ")");
                }

                Logger::info(str::format(
                  "[RTX-UV-AFFINE] ps=0x", std::hex, psHash,
                  " tex=0x", stageTexHash, std::dec, " ", stageTexWidth, "x", stageTexHeight,
                  " stage=", firstStage,
                  " vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
                  " mode=", m_uvResolutionMode == UvResolutionMode::ProvenIa
                              ? "proven-ia"
                              : m_uvResolutionMode == UvResolutionMode::CaptureInterpolant
                                  ? "capture-interpolant"
                                  : "legacy-tss",
                  " interp=", texcoordIdx,
                  " comps=(", uint32_t(m_texcoordCompU), ",", uint32_t(m_texcoordCompV), ")",
                  " sites=", uvOrigin.validSiteCount, "/", uvOrigin.invalidSiteCount,
                  " agree=", uvOrigin.sitesAgree ? 1 : 0,
                  " preferHF=", uvOrigin.preferredHighestFrequencySite ? 1 : 0,
                  " exact=", uvOrigin.affineExact ? 1 : 0,
                  " | U:[", formatUvComponentAffine(uvOrigin.affineU),
                  "] V:[", formatUvComponentAffine(uvOrigin.affineV),
                  "] | psResolved=", psAffineResolved ? 1 : 0,
                  " ps=(", psA, ",", psB, ",", psC, ",", psD, ",", psTx, ",", psTy, ")",
                  " vsFold=", vsAffineFold ? 1 : 0, vsLog,
                  " final=(", finalA, ",", finalB, ",", finalC, ",", finalD, ",", finalTx, ",", finalTy, ")",
                  " identity=", transformIsIdentity ? 1 : 0,
                  " usable=", transformIsUsable ? 1 : 0,
                  " applied=", applied ? 1 : 0,
                  ctabLog.empty() ? "" : str::format(" | ctab: ", ctabLog).c_str()));
              }
            }
          } else {
            // the sampled coordinate has no provable interpolant origin (screen-space,
            // reflection-driven, or untraceable): keep upstream-style TSS behavior
            m_uvResolutionMode = UvResolutionMode::LegacyTss;

            // rtx.d3d9.ue3LogUvAffineDetail: record the unprovable-origin outcome once per
            // shader+stage - the transform can never apply on this path
            if (m_frameOptions.ue3LogUvAffineDetail) {
              XXH64_hash_t noOriginKey = psHash;
              noOriginKey ^= (0xA11FE00Dull + firstStage) + 0x9E3779B97F4A7C15ull +
                             (noOriginKey << 6) + (noOriginKey >> 2);
              if (m_loggedUvAffineDetails.insert(noOriginKey).second) {
                Logger::info(str::format(
                  "[RTX-UV-AFFINE] ps=0x", std::hex, psHash,
                  " tex=0x", m_activeDrawCallState.materialData.colorTextures[0].getImageHash(), std::dec,
                  " stage=", firstStage,
                  " vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
                  " originValid=0 sites=", uvOrigin.validSiteCount, "/", uvOrigin.invalidSiteCount,
                  " - no provable interpolant origin; no texture transform derived (legacy TSS path)"));
              }
            }
          }

          if (m_frameOptions.ue3LogUvResolution || Logger::logLevel() <= LogLevel::Debug) {
            const XXH64_hash_t colorTextureHash =
              m_activeDrawCallState.materialData.colorTextures[0].getImageHash();

            XXH64_hash_t logKey = psHash;
            auto mixLog = [&](const uint64_t v) {
              logKey ^= v + 0x9E3779B97F4A7C15ull + (logKey << 6) + (logKey >> 2);
            };
            mixLog(firstStage);
            mixLog(uint64_t(m_uvResolutionMode));
            mixLog(texcoordIdx);
            mixLog(iaTexcoordIdx);
            mixLog((uint64_t(m_texcoordCompU) << 8) | uint64_t(m_texcoordCompV));
            mixLog(colorTextureHash);

            if (m_loggedUvResolutions.insert(logKey).second) {
              const char* modeName = "legacy-tss";
              if (m_uvResolutionMode == UvResolutionMode::ProvenIa) {
                modeName = "proven-ia";
              } else if (m_uvResolutionMode == UvResolutionMode::CaptureInterpolant) {
                modeName = "capture-interpolant";
              }

              const char* traceKindName = "none";
              if (traceEntry != nullptr) {
                switch (traceEntry->kind) {
                case Ue3VsUvTraceKind::PureMove: traceKindName = "pure-move"; break;
                case Ue3VsUvTraceKind::AffineConst: traceKindName = "affine-const"; break;
                case Ue3VsUvTraceKind::OriginOnly: traceKindName = "origin-only"; break;
                default: traceKindName = "invalid"; break;
                }
              }

              const char* siteDisagreementNote = "";
              if (!uvOrigin.sitesAgree) {
                siteDisagreementNote = uvOrigin.preferredHighestFrequencySite
                  ? " [sites disagree: preferred highest-frequency tiling]"
                  : " [AMBIGUOUS: sample sites disagree]";
              }

              const std::string msg = str::format(
                "[RTX-UV] ", modeName,
                ": ps=0x", std::hex, psHash,
                ", tex=0x", colorTextureHash, std::dec,
                ", stage=", firstStage,
                ", originValid=", uvOrigin.originValid,
                ", interpolantTexcoord=", texcoordIdx,
                ", comps=(", uint32_t(m_texcoordCompU), ",", uint32_t(m_texcoordCompV), ")",
                ", iaSet=", iaTexcoordIdx,
                ", vsTrace=", traceKindName,
                ", sites=", uvOrigin.validSiteCount, " valid/", uvOrigin.invalidSiteCount, " invalid",
                siteDisagreementNote,
                uvOrigin.affineExact ? "" : " [affine-inexact]",
                uvOriginBorrowed ? " [origin borrowed from sibling material samplers]" : "",
                m_forceIaTexcoordForOutlier ? " [outlier-override]" : "");
              if (m_frameOptions.ue3LogUvResolution) {
                Logger::info(msg);
              } else {
                Logger::debug(msg);
              }
            }
          }
        }
      }
    }

    m_texcoordIndex = texcoordIdx;
    m_iaTexcoordIndex = resolveIaTexcoordAvoidingNonUvElements(iaTexcoordIdx);

    return true;
  }

  // Two kinds of UE3 TEXCOORD element are not coordinates at all:
  //  - vertex lightmap policies append packed lighting coefficients as TEXCOORD5 (simple) or
  //    TEXCOORD5/6/7 (directional), typed D3DCOLOR rather than a float pair. The element count
  //    moves with the DirectionalLightmaps setting, and a D3DCOLOR-typed TEXCOORD is never a UV
  //    set in UE3, so the type alone identifies them without consulting the vertex shader.
  //  - instanced vertex factories put the per-instance basis in TEXCOORD1..4 on an instance-data
  //    stream. Those advance once per instance, so they carry no per-vertex meaning.
  uint32_t D3D9Rtx::resolveIaTexcoordAvoidingNonUvElements(const uint32_t iaTexcoordIdx) const {
    if (!m_frameOptions.ue3EngineMode || d3d9State().vertexDecl == nullptr)
      return iaTexcoordIdx;

    const uint32_t instanceDataStreamMask = m_currentUe3Instancing.instanceDataStreamMask;
    auto isNonUvElement = [instanceDataStreamMask](const D3DVERTEXELEMENT9& element) {
      if (element.Usage != D3DDECLUSAGE_TEXCOORD)
        return false;
      if (element.Type == D3DDECLTYPE_D3DCOLOR)
        return true;
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

    if (!requestedIsNonUv)
      return iaTexcoordIdx;

    // Fall back to the mesh's lowest real UV set rather than leaving the surface with the
    // lighting stream: an unresolvable index would drop the texcoord buffer entirely.
    uint32_t fallback = iaTexcoordIdx;
    bool found = false;
    for (const auto& element : elements) {
      if (element.Usage != D3DDECLUSAGE_TEXCOORD || isNonUvElement(element))
        continue;
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
    if (m_ue3TextureSpreadDirty &&
        currentReflexFrameId >= m_ue3TextureSpreadLastSaveFrame + kUe3CacheSaveIntervalFrames) {
      m_ue3TextureSpreadLastSaveFrame = uint32_t(currentReflexFrameId);
      saveUe3TextureSpreadCache();
    }

    if (m_ue3DiffuseSelectionDirty &&
        currentReflexFrameId >= m_ue3DiffuseSelectionLastSaveFrame + kUe3CacheSaveIntervalFrames) {
      m_ue3DiffuseSelectionLastSaveFrame = uint32_t(currentReflexFrameId);
      saveUe3DiffuseSelectionCache();
    }


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

    pruneUe3StaticVertexCaptureCache();
    pruneUe3GeometryMemoCache();
    updateUe3StaticVertexCaptureCacheState();
    // Independent of the capture cache, whose state update returns early when it is disabled.
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
