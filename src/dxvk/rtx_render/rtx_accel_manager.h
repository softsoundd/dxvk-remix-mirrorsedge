/*
* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include <mutex>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "../util/rc/util_rc_ptr.h"
#include "rtx_types.h"
#include "rtx_common_object.h"
#include "rtx_gpu_crash_recorder.h"
#include "rtx_staging.h"
#include "rtx_point_instancer_system.h"
#include "../util/util_vector.h"
#include "../util/util_matrix.h"
#include "../util/util_struct_hash.h"

namespace dxvk 
{
class DxvkContext;
class DxvkDevice;
class ResourceCache;
class CameraManager;
class OpacityMicromapManager;

// AccelManager is responsible for maintaining the acceleration structures (BLAS and TLAS)
class AccelManager : public CommonDeviceObject {
  class BlasBucket {
  public:
    std::vector<VkAccelerationStructureGeometryKHR> geometries {};
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges {};
    std::vector<RtInstance*> originalInstances {};
    std::vector<uint32_t> primitiveCounts {};
    std::vector<uint32_t> instanceBillboardIndices {};  // Billboard index within an instance's billboard array
    std::vector<uint32_t> indexOffsets {};              // Index offsets within geometry
    uint8_t instanceMask = 0;
    uint32_t instanceShaderBindingTableRecordOffset = 0;
    uint32_t customIndexFlags = 0;
    VkGeometryInstanceFlagsKHR instanceFlags = 0;
    bool usesUnorderedApproximations = false;
    uint32_t reorderedSurfacesOffset = UINT32_MAX;
    bool hasOmmInstances = false;
    bool hasSssInstances = false;
    // NV-DXVK start: churn-aware bucketing
    // Buckets holding instances that move or animate every frame are kept apart from the static
    // ones so that the per-frame rebuild/refit and CPU re-routing only touch the churning set.
    bool churning = false;
    // NV-DXVK end

    // The PooledBlas assigned to this bucket by createBlasBuffersAndInstances.
    // Stored here so the per-bucket cache can capture it after buildBlases.
    PooledBlas* assignedBlas = nullptr;
    
    // Tries to add a geometry instance to the bucket. The addition is successful if either:
    //   a) the bucket is empty,
    //   b) the instance has the same mask etc. as all other instances in the bucket.
    bool tryAddInstance(RtInstance* instance, bool instanceChurning);
  };

  // NV-DXVK start: churn-aware bucketing
  // An instance churns when its transform or its geometry has been changing on back-to-back
  // frames (with a short hold once it stops). Such instances dirty whichever merged bucket they
  // sit in every frame, so they are bucketed separately from the static geometry.
  static bool isInstanceChurning(const RtInstance* instance, uint32_t currentFrame);
  // NV-DXVK end

  // Key for O(1) bucket lookup in the merged-BLAS path.
  // Two instances can share a merged BLAS bucket iff they have identical keys.
  struct BlasBucketKey {
    uint32_t instanceShaderBindingTableRecordOffset = 0;
    uint32_t customIndexFlags = 0;
    VkGeometryInstanceFlagsKHR instanceFlags = 0;
    uint8_t instanceMask = 0;
    bool usesUnorderedApproximations = false;
    bool isSubsurface = false;
    bool churning = false;  // NV-DXVK: see BlasBucket::churning

    bool operator==(const BlasBucketKey& other) const {
      return instanceMask == other.instanceMask &&
             instanceShaderBindingTableRecordOffset == other.instanceShaderBindingTableRecordOffset &&
             customIndexFlags == other.customIndexFlags &&
             instanceFlags == other.instanceFlags &&
             usesUnorderedApproximations == other.usesUnorderedApproximations &&
             isSubsurface == other.isSubsurface &&
             churning == other.churning;
    }
  };

  struct BlasBucketKeyHash {
    size_t operator()(const BlasBucketKey& k) const {
      return static_cast<size_t>(hashStructByMemory<BlasBucketKey,
          &BlasBucketKey::instanceShaderBindingTableRecordOffset,
          &BlasBucketKey::customIndexFlags,
          &BlasBucketKey::instanceFlags,
          &BlasBucketKey::instanceMask,
          &BlasBucketKey::usesUnorderedApproximations,
          &BlasBucketKey::isSubsurface,
          &BlasBucketKey::churning>(k));
    }
  };

  struct UniqueBlasInstances {
    BlasEntry* blasEntry = nullptr;
    std::vector<RtInstance*> instances;
  };

public:
  AccelManager(AccelManager const&) = delete;
  AccelManager& operator=(AccelManager const&) = delete;

  explicit AccelManager(DxvkDevice* device);

  // Returns a GPU buffer containing the surface data for active instances
  const Rc<DxvkBuffer> getSurfaceBuffer() const { return m_surfaceBuffer; }

  const Rc<DxvkBuffer> getSurfaceMappingBuffer() const { return m_surfaceMappingBuffer; }

  const Rc<DxvkBuffer> getCurrentFramePrimitiveIDPrefixSumBuffer() const {
    return m_primitiveIDPrefixSumBuffer;
  }

  const Rc<DxvkBuffer> getLastFramePrimitiveIDPrefixSumBuffer() const {
    return m_primitiveIDPrefixSumBufferLastFrame;
  }

  const Rc<DxvkBuffer> getBillboardsBuffer() const { return m_billboardsBuffer; }

  // Clear all instances currently tracked by manager
  void clear();

  // Clean up instances which are deemed as no longer required
  void garbageCollection();

  // Prepares instance buffers for rendering by the GPU
  void prepareSceneData(Rc<DxvkContext> ctx, class DxvkBarrierSet& execBarriers, InstanceManager& instanceManager);

  // Uploads instances' surface data to the GPU
  void uploadSurfaceData(Rc<DxvkContext> ctx);

  // Merges the RtInstance's into a set of BLAS. Some of the BLAS will contain multiple geometries/instances,
  // and some other BLAS will be dedicated to instances with static geometries.
  void mergeInstancesIntoBlas(Rc<DxvkContext> ctx, class DxvkBarrierSet& execBarriers,
                              const std::vector<TextureRef>& textures, const CameraManager& cameraManager, 
                              InstanceManager& instanceManager, OpacityMicromapManager* opacityMicromapManager);

  // Dispatches GPU compute culling for all PointInstancer batches recorded during
  // mergeInstancesIntoBlas. Must be called after prepareSceneData (placeholders uploaded)
  // and before buildTlas.
  void dispatchPointInstancerCulling(Rc<DxvkContext> ctx, const CameraManager& cameraManager,
                                     const Rc<DxvkBuffer>& surfaceMaterialBuffer);

  void buildTlas(Rc<DxvkContext> ctx);

  void dumpCrashState(const char* reason) const { m_gpuCrashRecorder.dump(reason); }

  // Returns the number of live BLAS objects
  static uint32_t getBlasCount();

  uint32_t getSurfaceCount() const { return m_reorderedSurfaces.size(); }
  const std::vector<RtInstance*>& getOrderedInstances() const { return m_reorderedSurfaces; }

  bool hasShadowBackfaceSkipInstances() const { return m_hasShadowBackfaceSkipInstances; }

  // Returns true if the last mergeInstancesIntoBlas call took the fast-skip
  // path (scene generation unchanged).  When true, m_reorderedSurfaces and
  // all BLAS/surface data are identical to the previous frame, so callers
  // can skip redundant GPU uploads (e.g. surface materials).
  bool wasSceneUnchangedThisFrame() const { return m_sceneUnchangedThisFrame; }

  void removeInstanceFromBucketCache(RtInstance* instance);
  void invalidateOpacityMicromapBindings() { m_ommBindPending = true; }

private:
  struct SurfaceInfo {
    uint32_t surfaceMaterialIndex;
    Vector3 worldPosition;
  };

  // Persistent containers to reduce frame to frame reallocations in ::buildParticleSurfaceMapping()
  struct {
    std::vector<AccelManager::SurfaceInfo> surfaceInfoLists[2];   // Two containers for subsequent frames, ping-pong framed to frame
    uint32_t currIndex = 0;
    uint32_t prevIndex = 1;
  } buildParticleSurfaceMappingFuncState;

  // Persistent containers to reduce frame to frame reallocations in ::uploadSurfaceData()
  struct {
    std::vector<unsigned char> surfacesGPUData;
    std::vector<uint32_t> surfaceIndexMapping;
    uint32_t previousFrameSurfaceCount = 0; // Tracks last frame's surface count for mapping coverage
  } uploadSurfaceDataFuncState;

  void buildBlases(Rc<DxvkContext> ctx, DxvkBarrierSet& execBarriers,
                   const CameraManager& cameraManager, OpacityMicromapManager* opacityMicromapManager, const InstanceManager& instanceManager,
                   const std::vector<TextureRef>& textures, const std::vector<RtInstance*>& instances,
                   const std::vector<std::unique_ptr<BlasBucket>>& blasBuckets, 
                   std::vector<VkAccelerationStructureBuildGeometryInfoKHR>& blasToBuild,
                   std::vector<VkAccelerationStructureBuildRangeInfoKHR*>& blasRangesToBuild,
                   const std::vector<VkTransformMatrixKHR>& instanceTransforms,
                   size_t& currentScratchOffset);
  
  void addBlas(RtInstance* instance, BlasEntry* blasEntry, const Matrix4* instanceToObject);
  void addPointInstancerBlas(RtInstance* rtInstance, BlasEntry* blasEntry);

  void createBlasBuffersAndInstances(Rc<DxvkContext> ctx, 
                                     const std::vector<std::unique_ptr<BlasBucket>>& blasBuckets,
                                     std::vector<VkAccelerationStructureBuildGeometryInfoKHR>& blasToBuild,
                                     std::vector<VkAccelerationStructureBuildRangeInfoKHR*>& blasRangesToBuild,
                                     size_t& currentScratchOffset);
  template<Tlas::Type type>
  void internalBuildTlas(Rc<DxvkContext> ctx, size_t& totalScratchSize);

  void buildParticleSurfaceMapping(std::vector<uint32_t>& surfaceIndexMapping);

  bool validateUpdateMode(const VkAccelerationStructureBuildGeometryInfoKHR& oldInfo, const VkAccelerationStructureBuildGeometryInfoKHR& newInfo);

  std::vector<RtInstance*> m_reorderedSurfaces;
  std::vector<uint32_t> m_reorderedSurfacesFirstIndexOffset;
  std::vector<uint32_t> m_reorderedSurfacesPrimitiveIDPrefixSum;              // Exclusive prefix sum for this frame's surface primitive count array
  std::vector<uint32_t> m_reorderedSurfacesPrimitiveIDPrefixSumLastFrame;     // Exclusive prefix sum for last frame's surface primitive count array
  std::vector<VkAccelerationStructureInstanceKHR> m_mergedInstances[Tlas::Count];
  std::vector<Rc<PooledBlas>> m_blasPool;

  // GPU-driven PointInstancer culling batches, recorded per frame in mergeInstancesIntoBlas
  std::vector<PointInstancerBatch> m_pointInstancerBatches;

  // Number of VkAccelerationStructureInstanceKHR slots reserved for PointInstancer
  // instances in each TLAS type.  These slots are NOT stored in m_mergedInstances —
  // the GPU culling shader fills them directly in the instance buffer.
  uint32_t m_pointInstancerSlotsPerType[Tlas::Count] = {};

  // --- Incremental BLAS build caching ---
  // Scene generation from InstanceManager when the BLAS was last built.
  uint64_t m_lastProcessedGeneration = UINT64_MAX;
  bool m_sceneUnchangedThisFrame = false;
  bool m_hasShadowBackfaceSkipInstances = false;
  // Set when newly built OMMs need to be bound to BLASes.  Forces all buckets
  // dirty on the next incremental rebuild so tryBindOpacityMicromap runs.
  bool m_ommBindPending = false;

  // Number of non-billboard entries in m_mergedInstances per TLAS type.
  // prepareSceneData() truncates to this baseline before appending fresh billboard instances.
  uint32_t m_mergedInstancesBaselineCount[Tlas::Count] = {};

  // Tracks all dynamic BLAS references from the last full build so they can be
  // touched during the cached (skip) path to prevent GC from collecting them.
  std::vector<Rc<PooledBlas>> m_activeDynamicBlases;

  // --- Dynamics-only rebuild cached state ---
  // Per-bucket cache from the last build.  Each entry corresponds to one merged
  // BLAS bucket and stores all the data needed to restore its portion of
  // m_reorderedSurfaces and m_mergedInstances without re-running the bucket pipeline.
  struct CachedBucketState {
    // The instances that contributed geometry to this bucket (for dirty checking)
    std::vector<RtInstance*> instances;
    std::vector<uint64_t> instanceCacheIdentities;
    // Surface data for m_reorderedSurfaces
    std::vector<RtInstance*> surfaces;
    std::vector<uint32_t> indexOffsets;
    // The PooledBlas assigned to this bucket (kept alive via Rc)
    Rc<PooledBlas> assignedBlas;
    // TLAS instance template (surface offset must be adjusted each frame)
    VkAccelerationStructureInstanceKHR tlasInstance {};
    // Which TLAS type(s) this bucket was emitted to
    bool isUnordered = false;
    bool hasSssInstances = false;
    bool churning = false;  // NV-DXVK: see BlasBucket::churning
    // Set when one of this bucket's instances was destroyed: the bucket must be rebuilt from the
    // live instance list next frame and its cached instance pointers must not be dereferenced.
    bool forceDirty = false;
  };
  std::vector<CachedBucketState> m_cachedBuckets;

  // NV-DXVK start: churn-aware bucketing
  static BlasBucketKey cachedBucketKey(const CachedBucketState& bucket);

  // Bucket dirty diagnostics (rtx.logDynamicGeometryStats): how many merged buckets were
  // re-routed per frame, why, and which instances triggered it.
  enum class BucketDirtyReason : uint32_t {
    InstanceRemoved = 0,
    IdentityChanged,
    InstanceDirty,
    GeometryUpdated,
    KeyChanged,
    Consolidation,
    Count
  };
  struct BucketDirtyOffender {
    uint32_t prims = 0;
    uint32_t hits = 0;
    uint32_t transformHits = 0;   // hits where the instance's transform changed that frame
    uint32_t bucketInstances = 0; // size of the largest bucket this instance dirtied
    bool churnBucket = false;
  };
  struct BucketStats {
    uint32_t firstFrame = kInvalidFrameIndex;
    uint32_t frames = 0;
    uint64_t bucketsScanned = 0;
    uint64_t bucketsDirty = 0;
    uint64_t staticBucketsDirty = 0;
    uint64_t instancesInDirtyBuckets = 0;
    uint32_t reasons[static_cast<size_t>(BucketDirtyReason::Count)] = {};
    std::unordered_map<XXH64_hash_t, BucketDirtyOffender> offenders;
  };
  BucketStats m_bucketStats;
  void noteBucketDirty(const CachedBucketState& bucket, BucketDirtyReason reason, const RtInstance* inst, uint32_t currentFrame);
  void reportBucketStats(uint32_t currentFrame);
  // NV-DXVK end

  // Maps a merged instance pointer to its bucket index in m_cachedBuckets.
  // Allows O(1) "is this instance in a clean bucket?" check in the main loop.
  std::unordered_map<RtInstance*, uint32_t> m_instanceBucketIndex;

  // Set of BlasEntry* that went to the dynamic path on the last full rebuild.
  // Used for quick O(1) per-instance classification on the dynamics-only path.
  std::unordered_set<BlasEntry*> m_cachedDynamicBlasEntries;

  // Scratch containers for collecting dynamic BLAS groups during mergeInstancesIntoBlas().
  // The map is lookup-only; the vector preserves first-seen emission order.
  std::vector<UniqueBlasInstances> m_uniqueDynamicBlas;
  uint32_t m_uniqueDynamicBlasCount = 0;
  std::unordered_map<BlasEntry*, uint32_t> m_uniqueDynamicBlasIndex;

  void resetUniqueDynamicBlasGroups();

  Rc<DxvkBuffer> m_vkInstanceBuffer; // Note: Holds Vulkan AS Instances, not RtInstances
  Rc<DxvkBuffer> m_surfaceBuffer;
  Rc<DxvkBuffer> m_surfaceMappingBuffer;
  Rc<DxvkBuffer> m_transformBuffer;
  Rc<DxvkBuffer> m_primitiveIDPrefixSumBuffer;
  Rc<DxvkBuffer> m_primitiveIDPrefixSumBufferLastFrame;

  int getCurrentFramePrimitiveIDPrefixSumBufferID() const;

  Rc<PooledBlas> m_intersectionBlas;
  Rc<DxvkBuffer> m_aabbBuffer;
  Rc<DxvkBuffer> m_billboardsBuffer;
  void createAndBuildIntersectionBlas(Rc<DxvkContext> ctx, class DxvkBarrierSet& execBarriers);
  
  Rc<DxvkBuffer> getScratchMemory(const size_t requiredScratchAllocSize);
  Rc<PooledBlas> createPooledBlas(size_t bufferSize, const char* name) const;

  // The recorder currently lives here because it only captures acceleration
  // structure state. Move it higher if crash recording expands beyond AS data.
  RtxGpuCrashRecorder m_gpuCrashRecorder;
  VkDeviceSize m_scratchAlignment;
  Rc<DxvkBuffer> m_scratchBuffer;
};

}  // namespace dxvk

