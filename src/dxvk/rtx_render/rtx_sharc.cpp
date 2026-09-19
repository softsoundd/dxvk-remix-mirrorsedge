// Copyright (c) 2026, NVIDIA CORPORATION. SPDX-License-Identifier: MIT
#include <algorithm>
#include <cmath>
#include <cstring>
#include "rtx_sharc.h"
#include "dxvk_device.h"
#include "rtx_context.h"
#include "rtx_options.h"
#include "rtx_shader_manager.h"
#include "rtx_imgui.h"
#include "dxvk_scoped_annotation.h"
#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/sharc/sharc_binding_indices.h"
#include <rtx_shaders/sharc_resolve.h>

namespace dxvk {
  namespace {
    class SharcResolveShader : public ManagedShader {
      SHADER_SOURCE(SharcResolveShader, VK_SHADER_STAGE_COMPUTE_BIT, sharc_resolve)
      BEGIN_PARAMETER()
        CONSTANT_BUFFER(BINDING_CONSTANTS)
        { SHARC_BINDING_HASH, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT },
        { SHARC_BINDING_ACCUMULATION, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT },
        { SHARC_BINDING_RESOLVED, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT },
      END_PARAMETER()
    };
  }

  RtxSharc::RtxSharc(DxvkDevice* device) : CommonDeviceObject(device) { }

  bool RtxSharc::isSupported(const DxvkDevice& device) {
    const auto& f = device.features();
    return f.core.features.shaderInt64 && f.vulkan12Features.shaderBufferInt64Atomics
      && f.vulkan12Features.shaderFloat16 && f.vulkan11Features.storageBuffer16BitAccess
      && f.vulkan11Features.uniformAndStorageBuffer16BitAccess && f.khrRayQueryFeatures.rayQuery;
  }

  void RtxSharc::prepareFrame(RtxContext& ctx, RaytraceArgs& args, bool resetHistory) {
    args.sharcArgs = {};
    const bool selected = RtxOptions::integrateIndirectMode() == IntegrateIndirectMode::Sharc;
    const bool wasActive = m_active;
    m_active = false;
    if (!selected) {
      m_hash = nullptr;
      m_accumulation = nullptr;
      m_resolved = nullptr;
      m_allocationFailed = false;
      m_status = "Inactive";
      return;
    }
    ++m_selectedFrames;
    logFallbackStatsIfDue();
    if (!isSupported(*m_device)) {
      ++m_otherFallbackFrames;
      m_status = "Unsupported device features; using importance-sampled paths";
      return;
    }
    // The render-target camera changes the indirect path semantics and its
    // secondary rays are not yet represented by the SHARC estimator.  Keep a
    // conservative fallback until the cache can tag those paths explicitly.
    if (args.enableRaytracedRenderTarget) {
      ++m_rttFallbackFrames;
      m_status = "Raytraced render target active; using importance-sampled paths";
      return;
    }
    const bool wboit = RtxOptions::wboitEnabled();
    // Only an actually active portal can put a path into portal space. Configuring portal
    // texture hashes does not, and testing them here disabled SHARC for a whole session in
    // any game that defines portals at all, including rooms with none in sight.
    const bool rayPortals = args.numActiveRayPortals > 0;
    const bool opacityMicromap = RtxOptions::getEnableOpacityMicromap();
    if (rayPortals && !allowRayPortals()) {
      ++m_otherFallbackFrames;
      m_status = "Ray portals block SHARC: enable Allow SHARC with ray portals below to test; tracing normally";
      return;
    }
    const uint32_t skyRetries = uint32_t(std::clamp(updateSkyRetries(), 0, 4));
    // Discard cached estimates when the tested feature combination changes.
    const uint32_t compatibilityFlags = (wboit ? 1u : 0u) | (rayPortals ? 2u : 0u)
      | (updatePrimaryVertex() ? 8192u : 0u) | (skyRetries << 14)
      | (opacityMicromap ? 4u : 0u)
      | (allowRayPortals() ? 16u : 0u)
      | (deferredUpdates() ? 64u : 0u) | (queryRayGeneration() ? 128u : 0u)
      | (updateRayGeneration() ? 256u : 0u) | (allowSpecularPaths() ? 512u : 0u)
      | (queryTraceRay() ? 1024u : 0u) | (footprintGate() ? 4096u : 0u)
      | (queryTraceRay() && RtxOptions::isShaderExecutionReorderingInPathtracerIntegrateIndirectEnabled() ? 2048u : 0u);
    if (m_allocationFailed && !m_resetRequested) {
      ++m_otherFallbackFrames;
      return;
    }

    const uint32_t frame = m_device->getCurrentFrameId();
    const uint32_t capacity = 1u << std::clamp(capacityLog2(), 18, 22);
    const float scale = std::isfinite(gridScale()) ? std::clamp(gridScale(), 1.0f, 1000.0f) : 50.0f;
    const float roughness = std::isfinite(minRoughness()) ? std::clamp(minRoughness(), 0.05f, 1.0f) : 0.8f;
    // Never looser than the diffuse floor: a specular arrival is the case that needs the
    // stricter test, so a lower value here would invert the whole point of the split.
    const float roughnessSpecular = std::max(roughness,
      std::isfinite(minRoughnessSpecular()) ? std::clamp(minRoughnessSpecular(), 0.05f, 1.0f) : 0.5f);
    const float roughnessClamp = std::isfinite(updateRoughnessClamp())
      ? std::clamp(updateRoughnessClamp(), 0.0f, 1.0f) : 0.0f;
    const uint32_t sampleFloor = uint32_t(std::clamp(minSampleCount(), 0, 32));
    const float emissiveLimit = std::isfinite(maxEmissiveLuminance()) ? std::max(maxEmissiveLuminance(), 0.0f) : 0.0f;
    const uint32_t updateBounceLimit = std::clamp(updateBounces(), 1, 8);
    bool clear = resetHistory || m_resetRequested || compatibilityFlags != m_compatibilityFlags || !wasActive || m_lastFrame + 1 != frame
      || m_args.gridScale != scale || m_args.minRoughness != roughness
      || m_args.maxEmissiveLuminance != emissiveLimit
      || m_args.minRoughnessSpecular != roughnessSpecular
      || m_args.updateRoughnessClamp != roughnessClamp
      || m_args.minSampleCount != sampleFloor
      || m_args.updateBounces != updateBounceLimit;
    if (m_hash == nullptr || m_args.capacity != capacity) {
      try {
        DxvkBufferCreateInfo info = {};
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_TRANSFER_BIT;
        info.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        info.size = VkDeviceSize(capacity) * 8;
        auto hash = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "SHARC hash");
        info.size = VkDeviceSize(capacity) * 16;
        auto accumulation = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "SHARC accumulation");
        auto resolved = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "SHARC resolved");
        m_hash = hash;
        m_accumulation = accumulation;
        m_resolved = resolved;
        clear = true;
      } catch (const DxvkError& e) {
        Logger::err(str::format("SHARC allocation failed: ", e.message()));
        m_allocationFailed = true;
        m_resetRequested = false;
        m_status = "Cache allocation failed; tracing normally. Reset to retry.";
        return;
      }
    }
    if (clear) {
      m_haveGpuTimes = false;
      ctx.clearBuffer(m_hash, 0, m_hash->info().size, 0);
      ctx.clearBuffer(m_accumulation, 0, m_accumulation->info().size, 0);
      ctx.clearBuffer(m_resolved, 0, m_resolved->info().size, 0);
    }
    m_cacheAge = clear ? 0 : m_cacheAge + 1;
    m_args.cameraPositionPrev = m_args.cameraPosition;
    const auto& camera = ctx.getSceneManager().getCamera();
    const auto position = camera.getPosition();
    m_args.cameraPosition = vec3(position.x, position.y, position.z);
    if (clear) {
      m_args.cameraPositionPrev = m_args.cameraPosition;
    }
    m_args.capacity = capacity;
    m_args.gridScale = scale;
    m_args.minRoughness = roughness;
    m_args.maxEmissiveLuminance = emissiveLimit;
    m_args.minRoughnessSpecular = roughnessSpecular;
    m_args.updateRoughnessClamp = roughnessClamp;
    m_args.minSampleCount = sampleFloor;
    m_args.accumulationFrames = std::clamp(accumulationFrames(), 1, 64);
    m_args.staleFrames = std::clamp(staleFrames(), 8, 128);
    m_args.updateTileSize = std::clamp(updateTileSize(), 1, 16);
    m_args.updateBounces = updateBounceLimit;
    m_args.radianceScale = 1000.0f;
    // Deliberately absent from the clear condition above. The threshold only bounds values
    // deposited from now on; cells already holding an unclamped outlier wash it out within
    // accumulationFrames frames on their own, so clearing 1M-4M cells to make the change take
    // effect a tenth of a second sooner would cost far more than it bought. It is therefore one
    // of the few options that can be dragged live, which is what a threshold found by eye needs.
    m_args.maxDepositLuminance = (deferredUpdates() && std::isfinite(maxDepositLuminance()))
      ? std::max(maxDepositLuminance(), 0.0f) : 0.0f;
    // Deliberately absent from the clear condition above, for the same reason as the cap it
    // replaces: a cell already holding an unclamped outlier washes it out on its own.
    m_args.maxDepositRatio = (deferredUpdates() && std::isfinite(maxDepositRatio()))
      ? std::max(maxDepositRatio(), 0.0f) : 0.0f;
    m_args.minDepositCeiling = std::isfinite(minDepositCeiling()) ? std::max(minDepositCeiling(), 0.0f) : 0.0f;
    m_args.enabled = 1u | (updatePrimaryVertex() ? SHARC_UPDATE_FLAG_PRIMARY_VERTEX : 0u)
      | (skyRetries << SHARC_UPDATE_SKY_RETRY_SHIFT);
    m_args.allowSpecularPaths = allowSpecularPaths() ? 1u : 0u;
    m_args.footprintGate = footprintGate() ? 1u : 0u;
    args.sharcArgs = m_args;
    m_compatibilityFlags = compatibilityFlags;
    m_lastFrame = frame;
    m_resetRequested = false;
    m_allocationFailed = false;
    m_active = true;
    m_status = rayPortals
      ? "SHARC active: experimental ray portal override in use"
      : "Experimental diffuse cache; finite update paths";
  }

  void RtxSharc::bindResources(RtxContext& ctx) const {
    ctx.bindResourceBuffer(SHARC_BINDING_HASH, DxvkBufferSlice(m_hash, 0, m_hash->info().size));
    ctx.bindResourceBuffer(SHARC_BINDING_ACCUMULATION, DxvkBufferSlice(m_accumulation, 0, m_accumulation->info().size));
    ctx.bindResourceBuffer(SHARC_BINDING_RESOLVED, DxvkBufferSlice(m_resolved, 0, m_resolved->info().size));
  }

  void RtxSharc::dispatchResolve(RtxContext& ctx, const Resources::RaytracingOutput& output) {
    ScopedGpuProfileZone(&ctx, "SHARC Resolve");
    ctx.bindCommonRayTracingResources(output);
    bindResources(ctx);
    ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, SharcResolveShader::getShader());
    ctx.dispatch((m_args.capacity + 255) / 256, 1, 1);
  }

  void RtxSharc::recordTimestamp(RtxContext& ctx, TimingPoint point) {
    if (point == TimingPoint::Begin) {
      m_timingSlot = -1;
      if (!measureGpuTime() || !m_active
          || !m_device->adapter()->deviceProperties().limits.timestampComputeAndGraphics) {
        m_haveGpuTimes = false;
        return;
      }
      const uint32_t slot = m_device->getCurrentFrameId() % m_frameTimings.size();
      auto& frame = m_frameTimings[slot];
      if (frame.pending) {
        std::array<DxvkQueryData, 4> data;
        for (uint32_t i = 0; i < data.size(); ++i) {
          const auto status = frame.queries[i]->getData(data[i]);
          if (status == DxvkGpuQueryStatus::Pending) {
            return; // Skip measurement instead of waiting for the GPU.
          }
          if (status != DxvkGpuQueryStatus::Available) {
            frame.pending = false;
            m_haveGpuTimes = false;
            return;
          }
        }
        const double scale = m_device->adapter()->deviceProperties().limits.timestampPeriod * 1e-6;
        for (uint32_t i = 0; i < m_gpuTimes.size(); ++i) {
          const float ms = float((data[i + 1].timestamp.time - data[i].timestamp.time) * scale);
          m_gpuTimes[i] = m_haveGpuTimes ? m_gpuTimes[i] * 0.9f + ms * 0.1f : ms;
        }
        m_haveGpuTimes = true;
        frame.pending = false;
      }
      for (auto& query : frame.queries) {
        if (query == nullptr) {
          query = m_device->createGpuQuery(VK_QUERY_TYPE_TIMESTAMP, 0, 0);
        }
      }
      m_timingSlot = int(slot);
    }
    if (m_timingSlot >= 0) {
      auto& frame = m_frameTimings[m_timingSlot];
      ctx.writeTimestamp(frame.queries[static_cast<uint32_t>(point)]);
      if (point == TimingPoint::QueryEnd) {
        frame.pending = true;
        m_timingSlot = -1;
      }
    }
  }

  void RtxSharc::beginQueryStats(RtxContext& ctx) {
    m_statsSlot = -1;
    if (!measureGpuTime() || !collectQueryStats() || !m_active) {
      m_haveQueryStats = false;
      return;
    }
    if (m_statsGpu == nullptr) {
      DxvkBufferCreateInfo info = {};
      info.size = kStatsStride * m_statsReady.size();
      info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_TRANSFER_BIT;
      info.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      m_statsGpu = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "SHARC Query Stats GPU");
      info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
      m_statsReadback = m_device->createBuffer(info, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, DxvkMemoryStats::Category::RTXBuffer, "SHARC Query Stats Readback");
    }
    const uint32_t slot = m_device->getCurrentFrameId() % m_statsReady.size();
    const VkDeviceSize offset = slot * kStatsStride;
    if (m_statsReady[slot] != nullptr) {
      if (m_statsReady[slot]->test() != DxvkGpuEventStatus::Signaled) {
        return;
      }
      std::memcpy(m_queryStats.data(), m_statsReadback->mapPtr(offset), sizeof(m_queryStats));
      m_haveQueryStats = true;
      for (size_t i = 0; i < m_queryStats.size(); ++i) {
        m_queryStatsAccum[i] += m_queryStats[i];
      }
      if (++m_queryStatsAccumFrames >= kStatsWindowFrames) {
        m_queryStatsWindow = m_queryStatsAccum;
        m_queryStatsAccum.fill(0);
        m_queryStatsAccumFrames = 0;
        m_haveQueryStatsWindow = true;
      }
    } else {
      m_statsReady[slot] = m_device->createGpuEvent();
    }
    ctx.clearBuffer(m_statsGpu, offset, kStatsStride, 0);
    ctx.bindResourceBuffer(SHARC_BINDING_QUERY_STATS, DxvkBufferSlice(m_statsGpu, offset, kStatsStride));
    m_statsSlot = int(slot);
  }

  void RtxSharc::endQueryStats(RtxContext& ctx) {
    if (m_statsSlot < 0) {
      return;
    }
    const VkDeviceSize offset = m_statsSlot * kStatsStride;
    ctx.copyBuffer(m_statsReadback, offset, m_statsGpu, offset, sizeof(m_queryStats));
    ctx.emitMemoryBarrier(0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    ctx.signalGpuEvent(m_statsReady[m_statsSlot]);
    m_statsSlot = -1;
  }

  void RtxSharc::logFallbackStatsIfDue() {
    if (!logFallbackStats() || (m_selectedFrames % kFallbackLogInterval) != 0) {
      return;
    }
    const float denominator = float(m_selectedFrames);
    Logger::info(str::format(
      "SHARC fallback stats: selected for ", m_selectedFrames, " frames; ",
      m_rttFallbackFrames, " (", (100.0f * float(m_rttFallbackFrames) / denominator),
      "%) fell back on a raytraced render target; ",
      m_otherFallbackFrames, " (", (100.0f * float(m_otherFallbackFrames) / denominator),
      "%) on other conditions."));
  }

  void RtxSharc::showImguiSettings() {
    ImGui::TextWrapped("%s", m_status);
    if (m_selectedFrames > 0 && (m_rttFallbackFrames + m_otherFallbackFrames) > 0) {
      const float denominator = float(m_selectedFrames);
      ImGui::Text("Fell back on %u of %u frames: %.1f%% render target, %.1f%% other",
        m_rttFallbackFrames + m_otherFallbackFrames, m_selectedFrames,
        100.0f * float(m_rttFallbackFrames) / denominator,
        100.0f * float(m_otherFallbackFrames) / denominator);
    }
    RemixGui::Checkbox("Log SHARC fallback statistics", &logFallbackStatsObject());
    RemixGui::Checkbox("Allow SHARC with ray portals", &allowRayPortalsObject());
    ImGui::TextWrapped("Portal space is part of the cache key, so a vertex reached through a portal occupies its own cell and portal-only geometry is cached instead of left to brute-force paths; confirmed in a Portal RTX session with portals open. The split cells raise occupancy against the fixed capacity. WBOIT and opacity micromaps need no override; SHARC has dedicated variants for the first and sets the micromap pipeline flag for the second.");
    RemixGui::Checkbox("Batch SHARC cache writes", &deferredUpdatesObject());
    RemixGui::Checkbox("TraceRay SHARC query", &queryTraceRayObject());
    if (!queryTraceRay()) {
      RemixGui::Checkbox("Ray-generation SHARC query", &queryRayGenerationObject());
    }
    RemixGui::Checkbox("Ray-generation SHARC updates", &updateRayGenerationObject());
    RemixGui::Checkbox("Cache the primary vertex of update paths", &updatePrimaryVertexObject());
    RemixGui::DragInt("Sky-miss retries per update path", &updateSkyRetriesObject(), 1.0f, 0, 4);
    RemixGui::Checkbox("Reuse rough surfaces in specular paths", &allowSpecularPathsObject());
    if (allowSpecularPaths()) {
      ImGui::TextWrapped("Experimental: wider cache reuse can soften reflected lighting. Roughness and distance limits still apply.");
      RemixGui::Checkbox("Footprint gate for specular paths", &footprintGateObject());
      ImGui::TextWrapped(footprintGate()
        ? "Specular arrivals are gated by the footprint of the lobe that launched the segment; the specular roughness floor is not used."
        : "Specular arrivals are gated by the roughness of the surface they hit, using the specular roughness floor below.");
    }
    RemixGui::Checkbox("Measure SHARC GPU time", &measureGpuTimeObject());
    if (measureGpuTime() && m_haveGpuTimes) {
      ImGui::Text("GPU ms: update %.2f | resolve %.2f | query %.2f", m_gpuTimes[0], m_gpuTimes[1], m_gpuTimes[2]);
    }
    if (measureGpuTime()) {
      RemixGui::Checkbox("Include cache reuse statistics", &collectQueryStatsObject());
      ImGui::Text("Cache age: %u frames", m_cacheAge);
      ImGui::Text("Query backend: %s", (m_compatibilityFlags & 1024u)
        ? ((m_compatibilityFlags & 2048u) ? "TraceRay + SER" : "TraceRay")
        : ((m_compatibilityFlags & 128u) ? "RayQuery (ray generation)" : "RayQuery (compute)"));
      ImGui::Text("Particle transparency: %s", (m_compatibilityFlags & 1u) ? "WBOIT" : "sorted bins");
      if (collectQueryStats() && m_haveQueryStatsWindow) {
        const auto& s = m_queryStatsWindow;
        const float paths = float(std::max(uint64_t(1), s[0]));
        const float surfaces = float(std::max(uint64_t(1), s[2]));
        ImGui::Text("Cache terminates %.1f%% of paths | %.2f segments/path", 100.0f * s[7] / paths, s[1] / paths);
        ImGui::Text("Lookup hit rate %.1f%% | eligible surfaces %.1f%%", 100.0f * s[7] / float(std::max(uint64_t(1), s[6] + s[7])), 100.0f * s[4] / surfaces);
        const float misses = float(std::max(uint64_t(1), s[23] + s[24]));
        ImGui::Text("  Of misses: no cell %.1f%% | below sample floor %.1f%%",
          100.0f * s[23] / misses, 100.0f * s[24] / misses);
        ImGui::Text("Surface rejects: roughness %.1f%% | incoming non-diffuse %.1f%% | other %.1f%%",
          100.0f * s[3] / surfaces, 100.0f * s[8] / surfaces, 100.0f * s[9] / surfaces);
        ImGui::Text("  Other: non-opaque %.1f%% | medium %.1f%% | opacity < 1 %.1f%% | subsurface %.1f%% | emissive %.1f%%",
          100.0f * s[14] / surfaces, 100.0f * s[15] / surfaces, 100.0f * s[16] / surfaces,
          100.0f * s[17] / surfaces, 100.0f * s[18] / surfaces);
        const float tooClose = float(std::max(uint64_t(1), s[5]));
        ImGui::Text("Too close: %.1f%% of eligible surfaces | of those: last leg only %.1f%% | post-portal %.1f%% | first bounce %.1f%%",
          100.0f * s[5] / float(std::max(uint64_t(1), s[4])), 100.0f * s[19] / tooClose, 100.0f * s[20] / tooClose, 100.0f * s[21] / tooClose);
        ImGui::Text("Footprint too narrow: %.1f%% of eligible surfaces (specular arrivals, rtx.sharc.footprintGate)",
          100.0f * s[22] / float(std::max(uint64_t(1), s[4])));
        ImGui::Text("Samples: %llu paths over %u frames", (unsigned long long)s[0], kStatsWindowFrames);
        ImGui::Text("Path ends: sky %.1f%% | bounce limit %.1f%% | zero weight %.1f%% | roulette %.1f%%",
          100.0f * s[10] / paths, 100.0f * s[11] / paths, 100.0f * s[12] / paths, 100.0f * s[13] / paths);
        ImGui::Text("Path limits: %u..%u | roulette %s", RtxOptions::pathMinBounces(), RtxOptions::pathMaxBounces(),
          RtxOptions::enableRussianRoulette() ? "on" : "off");
      }
    }
    // Three presets along one axis: how much work the update pass does per frame.  Five
    // options actually trade along it -- the tile size and bounce count that set the update
    // budget, the sky retries that spend it outdoors, the primary deposit that spends it on
    // camera-visible surfaces, and the capacity the resolve pass pays for every frame.
    // Everything else a preset writes is a correctness or coverage control that buys
    // artefacts rather than speed when it is loosened, so all three write the same value for
    // it; docs/SHARC-presets-2026-09-16.md says why each one does or does not move and
    // docs/SHARC-adaptive-2026-09-16.md says why the primary deposit joined the list.
    // The diagnostic options (Measure SHARC GPU time, Include cache reuse statistics, Log
    // SHARC fallback statistics) and the backend A/B toggles are never part of a preset: they
    // cost about a millisecond and say nothing about quality.
    if (ImGui::BeginCombo("SHARC preset", "Choose to apply...")) {
      auto applyShared = [] {
        allowSpecularPathsObject().setDeferred(true);
        footprintGateObject().setDeferred(true);
        accumulationFramesObject().setDeferred(8);
        staleFramesObject().setDeferred(32);
        gridScaleObject().setDeferred(50.0f);
        minSampleCountObject().setDeferred(2);
        minRoughnessObject().setDeferred(0.05f);
        minRoughnessSpecularObject().setDeferred(0.7f);
        maxEmissiveLuminanceObject().setDeferred(0.1f);
        updateRoughnessClampObject().setDeferred(0.25f);
        // Deposit bounds are shared by every preset because they are not a budget: they decide what a
        // cell is allowed to hold, not how much work is spent filling it. The relative ceiling replaced
        // an absolute one that had to be set below the dimmest cell worth keeping and so darkened the
        // scene at any value low enough to catch an outlier; the absolute cap stays available at 0.
        maxDepositRatioObject().setDeferred(20.0f);
        minDepositCeilingObject().setDeferred(2.0f);
        maxDepositLuminanceObject().setDeferred(0.0f);
      };
      // The primary deposit is the one preset value backed by a frame-time measurement rather
      // than an estimate: the user read about 0.1 ms for it in game, which is the same order as
      // the whole cache's measured net benefit.  Quality and Balanced keep it because the user
      // also confirmed the quality gain; Performance is the preset whose entire purpose is to
      // spend less, and this is the largest measured item it can decline.
      auto applyUpdateBudget = [](int tileSize, int bounces, int capacity, int skyRetries, bool primaryVertex) {
        updateTileSizeObject().setDeferred(tileSize);
        updateBouncesObject().setDeferred(bounces);
        capacityLog2Object().setDeferred(capacity);
        updateSkyRetriesObject().setDeferred(skyRetries);
        updatePrimaryVertexObject().setDeferred(primaryVertex);
      };
      if (ImGui::Selectable("Quality")) {
        applyShared();
        applyUpdateBudget(3, 8, 22, 2, true);
        m_resetRequested = true;
      }
      RemixGui::SetTooltipToLastWidgetOnHover(
        "About 2.8 times the update paths of Balanced and twice the sky retries, so the cells Balanced leaves sparse -- "
        "hidden faces, surfaces off screen, distant relief -- are fed better, and update paths run to the full eight "
        "bounces so cells hold more of the multi-bounce tail. That depth is the real cost rather than the tile size: "
        "it more than doubles the traced segments per path and drops back to the eight-slot update shader, which "
        "Balanced avoids. Unmeasured.");
      if (ImGui::Selectable("Balanced (default)")) {
        applyShared();
        applyUpdateBudget(5, 3, 22, 1, true);
        m_resetRequested = true;
      }
      RemixGui::SetTooltipToLastWidgetOnHover(
        "The shipped default, tuned in Half-Life 2 RTX until cache boiling sat level with NRC. Tile 5 is NVIDIA's own "
        "recommended update downscale; the earlier tile 8 starved the cells that are reached only by a bounce, which "
        "is what boiled. Three bounces rather than four is most of what pays for that: with the primary vertex "
        "deposited it fits the compact four-slot update shader, where four bounces forced the eight-slot one and its "
        "register cost. The primary deposit itself measured about 0.1 ms in game and Balanced keeps it, because the "
        "quality it buys is confirmed and the time it costs is under a percent of a frame. Setting nothing in "
        "rtx.conf gives you this.");
      if (ImGui::Selectable("Performance")) {
        applyShared();
        applyUpdateBudget(12, 3, 20, 0, false);
        m_resetRequested = true;
      }
      RemixGui::SetTooltipToLastWidgetOnHover(
        "About 5.8 times fewer update paths than Balanced, no sky retries, a quarter of the resolve threads, and no "
        "primary-vertex deposit, which is the one item here with a measured price: about 0.1 ms. Bounce depth already "
        "matches Balanced, so the compact four-slot update shader is not something this preset buys. It gives up the "
        "sparsest cells first: hidden faces, freshly revealed geometry and outdoor relief. Expect tenths of a "
        "millisecond, not a transformation -- the whole cache measured about 0.1 ms net, so this trims the cost "
        "side and the benefit side together. Under open sky, without the primary deposit most update paths exit "
        "to the sky and store nothing, so the cache thins to near nothing out there; if that is where you play and "
        "you want the time back, rtx.integrateIndirectMode = 0 is the honest setting rather than this preset. "
        "Unmeasured except the 0.1 ms.");
      ImGui::EndCombo();
    }
    RemixGui::DragInt("Capacity exponent", &capacityLog2Object(), 1.0f, 18, 22);
    RemixGui::DragInt("Update tile size", &updateTileSizeObject(), 1.0f, 1, 16);
    RemixGui::DragInt("Update bounces", &updateBouncesObject(), 1.0f, 1, 8);
    RemixGui::DragInt("Accumulation frames", &accumulationFramesObject(), 1.0f, 1, 64);
    RemixGui::DragInt("Stale frames", &staleFramesObject(), 1.0f, 8, 128);
    RemixGui::DragFloat("Grid density (SHARC scene scale)", &gridScaleObject(), 1.0f, 1.0f, 1000.0f);
    RemixGui::DragFloat("Minimum roughness (squared)", &minRoughnessObject(), 0.01f, 0.05f, 1.0f);
    RemixGui::DragFloat("Update roughness clamp (squared)", &updateRoughnessClampObject(), 0.01f, 0.0f, 1.0f);
    RemixGui::SetTooltipToLastWidgetOnHover(
      "0 disables it. Roughens materials to at least this value while the cache is being updated, before the "
      "continuation is sampled and before NEE is evaluated. A cell holds one non-directional radiance value and "
      "cannot stand in for a narrow highlight, so on a glossy surface every update path arriving from a different "
      "direction deposits a different value and the cell's mean never settles -- which is what boils in reflections, "
      "and why neither a longer accumulation window nor a smaller tile fixes it. This makes a cell store what its "
      "surface would reflect if it were rough, which an isotropic cell can represent. Costs no coverage: no lookup "
      "refused, no surface rejected, no cell lost. Cached reflections soften in exchange. Same scale as Minimum "
      "roughness above, but a different job -- that one decides what may be cached, this one never affects "
      "eligibility. Clears the cache when changed. Unmeasured.");
    RemixGui::DragFloat("Max emissive luminance", &maxEmissiveLuminanceObject(), 0.001f, 0.0f, 1.0f);
    RemixGui::DragInt("Minimum cell samples", &minSampleCountObject(), 1.0f, 0, 32);
    if (deferredUpdates()) {
      RemixGui::DragFloat("Max deposit ratio", &maxDepositRatioObject(), 0.5f, 0.0f, 200.0f);
      RemixGui::SetTooltipToLastWidgetOnHover(
        "0 disables it. Caps a single deposit at this multiple of what the cell already holds. An absolute cap has "
        "to be set below the dimmest cell worth keeping, so any value low enough to catch fireflies also caps every "
        "cell's mean and darkens the whole scene -- which is why Max deposit luminance is so hard to set. This asks "
        "the question that matters instead: is this deposit wildly unlike what this cell has already converged on? "
        "A bright cell keeps its headroom, a dark one still refuses spikes. Cells below Minimum cell samples are "
        "exempt, having no mean to measure against yet. Washes out within Accumulation frames, so it can be dragged "
        "live. Unmeasured.");
      RemixGui::DragFloat("Min deposit ceiling", &minDepositCeilingObject(), 0.1f, 0.0f, 100.0f);
      RemixGui::SetTooltipToLastWidgetOnHover(
        "Floor under the ratio ceiling above, in absolute luminance. A cell sitting near black would otherwise pin "
        "its own ceiling near zero and never brighten when the lighting changes, since every deposit that would "
        "have raised it gets clamped away first. Raise it if lights turning on are slow to show up in indirect "
        "lighting; lower it if dark areas still sparkle.");
      RemixGui::DragFloat("Max deposit luminance", &maxDepositLuminanceObject(), 0.5f, 0.0f, 1000.0f);
      RemixGui::SetTooltipToLastWidgetOnHover(
        "0 disables it. Caps the luminance of a single value an update path writes into a cell. A cell is a mean, so "
        "one outlier is not averaged away, only divided by the cell's sample count -- and that count falls with the "
        "render resolution, because the update pass traces one path per tile of the render target. That is why "
        "fireflies on reflective surfaces get worse the lower the DLSS preset, and why lowering the tile size cures "
        "them: both move the same divisor. This bounds the outlier instead, and unlike every other remedy it costs no "
        "coverage -- no lookup is refused, no surface rejected, no cell lost. The cost is bias: a cell whose true "
        "radiance exceeds the threshold is stored dark. Pick the value from debug view 591 (Update Deposit), whose red "
        "channel is the brightest single deposit a path made, measured before the clamp. Not view 583: that is a "
        "cell mean, a smaller number, and setting this from it would clamp far too hard. View 591 draws one pixel "
        "per update tile, so most of the frame is black by design. Takes effect within "
        "Accumulation frames without clearing the cache, so it can be dragged live. Unmeasured.");
    }
    if (allowSpecularPaths() && !footprintGate()) {
      RemixGui::DragFloat("Minimum roughness, specular paths", &minRoughnessSpecularObject(), 0.01f, 0.05f, 1.0f);
    }
    if (ImGui::Button("Reset SHARC")) {
      m_resetRequested = true;
    }
  }
}
