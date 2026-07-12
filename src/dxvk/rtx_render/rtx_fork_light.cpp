// src/dxvk/rtx_render/rtx_fork_light.cpp
//
// Fork-owned file. Contains the implementations of fork_hooks:: functions
// for the LightManager external-light and persistent-light paths, lifted from
// rtx_light_manager.cpp during the 2026-04-18 fork touchpoint-pattern refactor.
//
// See docs/fork-touchpoints.md for the full fork-hooks catalogue.
//
// NOTE: Several hooks below access private members of LightManager
// (m_externalLights, m_externalDomeLights, m_externalActiveLightList,
// m_externalActiveDomeLight, m_pendingExternalLightErases,
// m_pendingExternalLightUpdates, m_pendingExternalActiveLights,
// m_persistentExternalLights). This file requires that LightManager declare
// the relevant hooks as friends — see rtx_light_manager.h.

#include "rtx_fork_hooks.h"

#include "rtx_light_manager.h"   // LightManager (full definition for friend access)
#include "rtx_lights.h"          // RtLight, kInvalidExternallyTrackedLightId
#include "rtx_options.h"         // RtxOptions::getNumFramesToPutLightsToSleep
#include "rtx_types.h"           // ReplacementInstance, PrimInstance

namespace dxvk {
namespace fork_hooks {

  // ---------------------------------------------------------------------------
  // flushPendingLightMutations
  //
  // Called at the top of LightManager::prepareSceneData (before linearization)
  // to drain the three deferred-mutation queues in the correct order:
  //   1. Erase: clears replacement links then removes from all external maps.
  //   2. Update/create: erase-then-emplace (handles union-type changes safely),
  //      stamps frameLastTouched.
  //   3. Activate: moves handles from the pending-active queue into the active
  //      set / active dome slot.
  //   4. Auto-instance: re-activates all persistent lights (idempotent per frame).
  //
  // ACCESS NOTE: reads/writes m_pendingExternalLightErases,
  // m_externalLights, m_externalDomeLights, m_externalActiveLightList,
  // m_externalActiveDomeLight, m_pendingExternalLightUpdates,
  // m_pendingExternalActiveLights, m_persistentExternalLights — all private.
  // Friend declarations in rtx_light_manager.h are required.
  // ---------------------------------------------------------------------------
  void flushPendingLightMutations(LightManager& mgr) {
    // Erase first
    for (auto h : mgr.m_pendingExternalLightErases) {
      auto it = mgr.m_externalLights.find(h);
      if (it != mgr.m_externalLights.end()) {
        // Ensure any replacement ownership is cleared before destruction
        it->second.getPrimInstanceOwner().setReplacementInstance(
          nullptr, ReplacementInstance::kInvalidReplacementIndex,
          &it->second, PrimInstance::Type::Light);
        mgr.m_externalLights.erase(it);
      }
      mgr.m_externalDomeLights.erase(h);
      mgr.m_externalActiveLightList.erase(h);
      if (mgr.m_externalActiveDomeLight == h) {
        mgr.m_externalActiveDomeLight = nullptr;
      }
    }
    mgr.m_pendingExternalLightErases.clear();

    // Apply updates/creates; erase then emplace to safely handle union types
    for (auto& upd : mgr.m_pendingExternalLightUpdates) {
      auto it = mgr.m_externalLights.find(upd.first);
      if (it != mgr.m_externalLights.end()) {
        // Clear replacement links on the old light before replacing
        it->second.getPrimInstanceOwner().setReplacementInstance(
          nullptr, ReplacementInstance::kInvalidReplacementIndex,
          &it->second, PrimInstance::Type::Light);
        mgr.m_externalLights.erase(it);
      }
      auto [it_new, success] = mgr.m_externalLights.emplace(upd.first, upd.second);
      it_new->second.setFrameLastTouched(mgr.device()->getCurrentFrameId());
    }
    mgr.m_pendingExternalLightUpdates.clear();

    // Apply active instances registration
    for (auto h : mgr.m_pendingExternalActiveLights) {
      if (mgr.m_externalLights.find(h) != mgr.m_externalLights.end()) {
        mgr.m_externalActiveLightList.insert(h);
      } else if (mgr.m_externalDomeLights.find(h) != mgr.m_externalDomeLights.end()
                 && mgr.m_externalActiveDomeLight == nullptr) {
        mgr.m_externalActiveDomeLight = h;
      }
    }
    mgr.m_pendingExternalActiveLights.clear();

    // Auto-instance any persistent external lights (idempotent within frame)
    for (auto h : mgr.m_persistentExternalLights) {
      if (mgr.m_externalLights.find(h) != mgr.m_externalLights.end()) {
        mgr.m_externalActiveLightList.insert(h);
      } else if (mgr.m_externalDomeLights.find(h) != mgr.m_externalDomeLights.end()
                 && mgr.m_externalActiveDomeLight == nullptr) {
        mgr.m_externalActiveDomeLight = h;
      }
    }

    // Note: Do not auto-instance all external lights here; activation is driven by API per-frame
  }

  // ---------------------------------------------------------------------------
  // updateLightStaticSleep
  //
  // Shared static-sleep logic used by both updateExternallyTrackedLight (indexed
  // path) and addExternalLight (hash-map path). The two call sites in
  // rtx_light_manager.cpp were nearly identical; this single hook eliminates
  // the duplication.
  //
  // Preserves temporal denoiser data for static lights by skipping the copy
  // when isStaticCount >= numFramesToPutLightsToSleep. Always updates dynamic
  // lights. Restores bufferIdx in both paths; restores externallyTrackedLightId
  // only when externalId != kInvalidExternallyTrackedLightId (indexed path).
  // Stamps frameLastTouched unconditionally.
  //
  // Parameters:
  //   light    — pointer to the existing RtLight to update in-place.
  //   newLight — the incoming light data.
  //   device   — used to stamp getCurrentFrameId() on frameLastTouched.
  //   externalId — pass kInvalidExternallyTrackedLightId to skip id restore
  //               (hash-map / addExternalLight path).
  // ---------------------------------------------------------------------------
  void updateLightStaticSleep(
      RtLight* light,
      const RtLight& newLight,
      DxvkDevice* device,
      uint64_t externalId) {

    const uint16_t bufferIdx = light->getBufferIdx();

    if (!newLight.isDynamic && !LightManager::suppressLightKeeping()) {
      const uint32_t isStaticCount = light->isStaticCount;

      // If this light hasn't moved for N frames, put it to sleep to preserve
      // temporal data
      if (isStaticCount < RtxOptions::getNumFramesToPutLightsToSleep()) {
        *light = newLight;
        light->setBufferIdx(bufferIdx);
        if (externalId != kInvalidExternallyTrackedLightId) {
          light->setExternallyTrackedLightId(externalId);
        }
        light->isStaticCount = isStaticCount + 1;  // Preserve and increment counter
      } else {
        // Light is asleep — don't update, just increment counter
        light->isStaticCount = isStaticCount + 1;
      }
    } else {
      // Dynamic lights always update
      *light = newLight;
      light->setBufferIdx(bufferIdx);
      if (externalId != kInvalidExternallyTrackedLightId) {
        light->setExternallyTrackedLightId(externalId);
      }
    }

    light->setFrameLastTouched(device->getCurrentFrameId());
  }

  // ---------------------------------------------------------------------------
  // setExternalLightEmplace
  //
  // Called from addExternalLight when the handle is not already in
  // m_externalLights (the "new light" branch). Emplaces the light and stamps
  // frameLastTouched.
  //
  // ACCESS NOTE: writes m_externalLights (private). Friend declaration in
  // rtx_light_manager.h is required.
  // ---------------------------------------------------------------------------
  void setExternalLightEmplace(
      LightManager& mgr,
      remixapi_LightHandle handle,
      const RtLight& rtlight) {

    auto [it, inserted] = mgr.m_externalLights.emplace(handle, rtlight);
    if (inserted) {
      it->second.setFrameLastTouched(mgr.device()->getCurrentFrameId());
    }
  }

  // ---------------------------------------------------------------------------
  // disableExternalLightQueue
  //
  // Called from removeExternalLight. Queues the handle for deferred erase at
  // frame start (in flushPendingLightMutations) instead of erasing immediately,
  // preventing mid-frame iterator invalidation.
  //
  // ACCESS NOTE: writes m_pendingExternalLightErases (private). Friend
  // declaration in rtx_light_manager.h is required.
  // ---------------------------------------------------------------------------
  void disableExternalLightQueue(LightManager& mgr, remixapi_LightHandle handle) {
    mgr.m_pendingExternalLightErases.push_back(handle);
  }

  // ---------------------------------------------------------------------------
  // registerPersistentLight
  //
  // Called from LightManager::registerPersistentExternalLight. Inserts the
  // handle into m_persistentExternalLights if non-null.
  //
  // ACCESS NOTE: writes m_persistentExternalLights (private). Friend
  // declaration in rtx_light_manager.h is required.
  // ---------------------------------------------------------------------------
  void registerPersistentLight(LightManager& mgr, remixapi_LightHandle handle) {
    if (handle) {
      mgr.m_persistentExternalLights.insert(handle);
    }
  }

  // ---------------------------------------------------------------------------
  // unregisterPersistentLight
  //
  // Called from LightManager::unregisterPersistentExternalLight. Removes the
  // handle from m_persistentExternalLights if non-null.
  //
  // ACCESS NOTE: writes m_persistentExternalLights (private). Friend
  // declaration in rtx_light_manager.h is required.
  // ---------------------------------------------------------------------------
  void unregisterPersistentLight(LightManager& mgr, remixapi_LightHandle handle) {
    if (handle) {
      mgr.m_persistentExternalLights.erase(handle);
    }
  }

  // ---------------------------------------------------------------------------
  // queueAutoInstancePersistent
  //
  // Called from LightManager::queueAutoInstancePersistent. Copies all handles
  // in m_persistentExternalLights into m_pendingExternalActiveLights so they
  // are activated at the next frame-start flush.
  //
  // ACCESS NOTE: reads m_persistentExternalLights, writes
  // m_pendingExternalActiveLights (both private). Friend declaration in
  // rtx_light_manager.h is required.
  // ---------------------------------------------------------------------------
  void queueAutoInstancePersistent(LightManager& mgr) {
    for (auto h : mgr.m_persistentExternalLights) {
      mgr.m_pendingExternalActiveLights.insert(h);
    }
  }

} // namespace fork_hooks
} // namespace dxvk
