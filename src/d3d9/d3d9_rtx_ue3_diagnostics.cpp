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

  namespace {
    const char* ue3DeclUsageName(const BYTE usage) {
      switch (usage) {
      case D3DDECLUSAGE_POSITION:     return "POSITION";
      case D3DDECLUSAGE_BLENDWEIGHT:  return "BLENDWEIGHT";
      case D3DDECLUSAGE_BLENDINDICES: return "BLENDINDICES";
      case D3DDECLUSAGE_NORMAL:       return "NORMAL";
      case D3DDECLUSAGE_PSIZE:        return "PSIZE";
      case D3DDECLUSAGE_TEXCOORD:     return "TEXCOORD";
      case D3DDECLUSAGE_TANGENT:      return "TANGENT";
      case D3DDECLUSAGE_BINORMAL:     return "BINORMAL";
      case D3DDECLUSAGE_TESSFACTOR:   return "TESSFACTOR";
      case D3DDECLUSAGE_POSITIONT:    return "POSITIONT";
      case D3DDECLUSAGE_COLOR:        return "COLOR";
      case D3DDECLUSAGE_FOG:          return "FOG";
      case D3DDECLUSAGE_DEPTH:        return "DEPTH";
      case D3DDECLUSAGE_SAMPLE:       return "SAMPLE";
      default:                        return "?";
      }
    }

    const char* ue3DeclTypeName(const BYTE type) {
      switch (type) {
      case D3DDECLTYPE_FLOAT1:    return "FLOAT1";
      case D3DDECLTYPE_FLOAT2:    return "FLOAT2";
      case D3DDECLTYPE_FLOAT3:    return "FLOAT3";
      case D3DDECLTYPE_FLOAT4:    return "FLOAT4";
      case D3DDECLTYPE_D3DCOLOR:  return "D3DCOLOR";
      case D3DDECLTYPE_UBYTE4:    return "UBYTE4";
      case D3DDECLTYPE_SHORT2:    return "SHORT2";
      case D3DDECLTYPE_SHORT4:    return "SHORT4";
      case D3DDECLTYPE_UBYTE4N:   return "UBYTE4N";
      case D3DDECLTYPE_SHORT2N:   return "SHORT2N";
      case D3DDECLTYPE_SHORT4N:   return "SHORT4N";
      case D3DDECLTYPE_USHORT2N:  return "USHORT2N";
      case D3DDECLTYPE_USHORT4N:  return "USHORT4N";
      case D3DDECLTYPE_UDEC3:     return "UDEC3";
      case D3DDECLTYPE_DEC3N:     return "DEC3N";
      case D3DDECLTYPE_FLOAT16_2: return "FLOAT16_2";
      case D3DDECLTYPE_FLOAT16_4: return "FLOAT16_4";
      case D3DDECLTYPE_UNUSED:    return "UNUSED";
      default:                    return "?";
      }
    }

    std::string formatMatrixRows(const Matrix4& m) {
      std::string out;
      for (uint32_t row = 0; row < 4; row++) {
        out += str::format(row == 0 ? "[" : " [",
                           m[row].x, ",", m[row].y, ",", m[row].z, ",", m[row].w, "]");
      }
      return out;
    }
  }

  void D3D9Rtx::reportUe3InstancedDrawStats() {
    if (!m_frameOptions.ue3LogInstancedDrawStats) {
      return;
    }

    ++m_ue3InstancedStatFrames;

    const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();

    // roughly once a second at any plausible frame rate; this is a diagnostic, not a metric
    constexpr uint32_t kStatIntervalFrames = 60;
    if (currentFrame - m_ue3InstancedStatFrameStamp < kStatIntervalFrames) {
      return;
    }
    m_ue3InstancedStatFrameStamp = currentFrame;

    const uint32_t frames = std::max(m_ue3InstancedStatFrames, 1u);
    const double perFrame = 1.0 / double(frames);

    Logger::info(str::format(
      "[RTX-Compatibility][UE3-Instanced] per frame over ", frames, " frames: ",
      double(m_ue3InstancedStatDraws) * perFrame, " instanced draws, ",
      double(m_ue3InstancedStatInstancesSeen) * perFrame, " hardware instances, ",
      double(m_ue3InstancedStatInstancesSubmitted) * perFrame, " submitted, ",
      double(m_ue3InstancedStatCulledDistance) * perFrame, " distance-culled, ",
      double(m_ue3InstancedStatCulledBudget) * perFrame, " budget-culled; ",
      double(m_ue3InstancedStatSubmitNs) * perFrame / 1.0e6, " ms/frame expanding them on the "
      "submitting thread (the rest of an instance's cost lands on the consumer thread and the GPU)."));

    // ue3StableDecomposedInstanceIdentity rests on the game keeping its instance order, so report how
    // far index-paired instances actually moved rather than taking that on trust.
    if (m_ue3InstancedStatOrderPairs > 0 || m_ue3InstancedStatOrderSizeChanges > 0) {
      const double meanDisplacement = m_ue3InstancedStatOrderPairs > 0
        ? m_ue3InstancedStatOrderDisplacementSum / double(m_ue3InstancedStatOrderPairs)
        : 0.0;
      const uint32_t stablePercent = m_ue3InstancedStatOrderPairs > 0
        ? uint32_t((m_ue3InstancedStatOrderStablePairs * 100ull) / m_ue3InstancedStatOrderPairs)
        : 0u;

      Logger::info(str::format(
        "[RTX-Compatibility][UE3-Instanced] instance order: ",
        m_ue3InstancedStatOrderComparableBatches, " batch comparisons over ", frames, " frames (",
        m_ue3InstancedStatOrderSizeChanges, " skipped on an instance count change), ",
        m_ue3InstancedStatOrderPairs, " index-paired instances, mean displacement ",
        meanDisplacement, " units, max ", m_ue3InstancedStatOrderDisplacementMax, ", ",
        stablePercent, "% within rtx.uniqueObjectDistance. A mean on the scale of the batch's own "
        "extent means the game reorders its instance buffer and index pairing is meaningless."));
    }

    m_ue3InstancedStatFrames = 0;
    m_ue3InstancedStatDraws = 0;
    m_ue3InstancedStatInstancesSeen = 0;
    m_ue3InstancedStatInstancesSubmitted = 0;
    m_ue3InstancedStatCulledDistance = 0;
    m_ue3InstancedStatCulledBudget = 0;
    m_ue3InstancedStatSubmitNs = 0;
    m_ue3InstancedStatOrderPairs = 0;
    m_ue3InstancedStatOrderStablePairs = 0;
    m_ue3InstancedStatOrderSizeChanges = 0;
    m_ue3InstancedStatOrderComparableBatches = 0;
    m_ue3InstancedStatOrderDisplacementSum = 0.0;
    m_ue3InstancedStatOrderDisplacementMax = 0.f;
  }

  const char* D3D9Rtx::describeUe3VertexFactory(const Ue3VertexFactoryType type) {
    switch (type) {
    case Ue3VertexFactoryType::Unknown: return "Unknown";
    case Ue3VertexFactoryType::Local: return "Local";
    case Ue3VertexFactoryType::GPUSkin: return "GPUSkin";
    case Ue3VertexFactoryType::GPUSkinMorph: return "GPUSkinMorph";
    case Ue3VertexFactoryType::Terrain: return "Terrain";
    case Ue3VertexFactoryType::TerrainMorph: return "TerrainMorph";
    case Ue3VertexFactoryType::Particle: return "Particle";
    case Ue3VertexFactoryType::ParticleBeamTrail: return "ParticleBeamTrail";
    case Ue3VertexFactoryType::SpeedTree: return "SpeedTree";
    case Ue3VertexFactoryType::Foliage: return "Foliage";
    case Ue3VertexFactoryType::ParticleInstancedMesh: return "ParticleInstancedMesh";
    case Ue3VertexFactoryType::LocalDecal: return "LocalDecal";
    case Ue3VertexFactoryType::LensFlare: return "LensFlare";
    case Ue3VertexFactoryType::PositionOnly: return "PositionOnly";
    }
    return "Unknown";
  }

  const char* D3D9Rtx::describeUe3PassType(const Ue3PassType type) {
    switch (type) {
    case Ue3PassType::Unknown: return "Unknown";
    case Ue3PassType::Material: return "Material";
    case Ue3PassType::DepthPrepass: return "DepthPrepass";
    case Ue3PassType::ShadowDepth: return "ShadowDepth";
    case Ue3PassType::Velocity: return "Velocity";
    case Ue3PassType::Lighting: return "Lighting";
    case Ue3PassType::ModulatedShadowProjection: return "ModulatedShadowProjection";
    case Ue3PassType::FullscreenPostProcess: return "FullscreenPostProcess";
    case Ue3PassType::UiComposite: return "UiComposite";
    case Ue3PassType::FogOrDistortion: return "FogOrDistortion";
    case Ue3PassType::VideoCinematic: return "VideoCinematic";
    case Ue3PassType::VideoSurface: return "VideoSurface";
    case Ue3PassType::SceneCapture: return "SceneCapture";
    }
    return "Unknown";
  }

  const char* D3D9Rtx::describeGeometryStatus(const RtxGeometryStatus status) {
    switch (status) {
    case RtxGeometryStatus::Ignored: return "Ignored";
    case RtxGeometryStatus::Rasterized: return "Rasterized";
    case RtxGeometryStatus::RayTraced: return "RayTraced";
    }
    return "Unknown";
  }

  void D3D9Rtx::logUe3Classification(const DrawContext& drawContext,
                                     const Ue3PassType passType,
                                     const RtxGeometryStatus status,
                                     const char* reason) {
    // The line below is a debug-level message: when the logger would drop it, return before
    // formatting it. This runs for every ray-traced draw, and formatting alone costs a few
    // microseconds each, which at thousands of draws per frame is several milliseconds.
    if (Logger::logLevel() > LogLevel::Debug) {
      return;
    }

    XXH64_hash_t vsHash = 0;
    if (m_parent->UseProgrammableVS() && d3d9State().vertexShader.ptr() != nullptr) {
      vsHash = d3d9State().vertexShader->GetCommonShader()->GetBytecodeHash();
    }
    XXH64_hash_t psHash = 0;
    if (m_parent->UseProgrammablePS() && d3d9State().pixelShader.ptr() != nullptr) {
      psHash = d3d9State().pixelShader->GetCommonShader()->GetBytecodeHash();
    }

    uint32_t rtWidth = 0;
    uint32_t rtHeight = 0;
    if (d3d9State().renderTargets[kRenderTargetIndex] != nullptr) {
      const auto& rtExt = d3d9State().renderTargets[kRenderTargetIndex]->GetSurfaceExtent();
      rtWidth = rtExt.width;
      rtHeight = rtExt.height;
    }

    Logger::debug(str::format(
      "[RTX-Compatibility][UE3] draw=", m_activeDrawCallState.drawCallID,
      " vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
      " pass=", describeUe3PassType(passType),
      " status=", describeGeometryStatus(status),
      " prims=", drawContext.PrimitiveCount,
      " vsHash=0x", std::hex, vsHash,
      " psHash=0x", psHash, std::dec,
      " rt=", rtWidth, "x", rtHeight,
      " fg=", m_ue3ForegroundDpgActive ? 1 : 0,
      " reason=", reason));
  }

  const char* D3D9Rtx::describeUe3CapturePositionSource(const Ue3CapturePositionSource source) {
    switch (source) {
    case Ue3CapturePositionSource::ClipReconstruction: return "ClipReconstruction";
    case Ue3CapturePositionSource::InputAssembler: return "InputAssembler";
    case Ue3CapturePositionSource::PreProjectionRegister: return "PreProjectionRegister";
    }
    return "Unknown";
  }

  std::string D3D9Rtx::describeUe3VsConstantRegister(XXH64_hash_t vsBytecodeHash, uint32_t reg) const {
    auto it = m_ue3VsConstantSymbols.find(vsBytecodeHash);
    if (it != m_ue3VsConstantSymbols.end()) {
      for (const Ue3VsConstantSymbol& symbol : it->second) {
        if (reg < symbol.registerIndex || reg >= symbol.registerIndex + symbol.registerCount) {
          continue;
        }
        // Name the row for multi-register symbols, so a churning matrix says which row moved.
        if (symbol.registerCount > 1) {
          return str::format("c", reg, " (", symbol.name, "[", reg - symbol.registerIndex, "])");
        }
        return str::format("c", reg, " (", symbol.name, ")");
      }
    }
    return str::format("c", reg);
  }

  // Accumulates the three levels described on Ue3ChurnMeshEntry. Levels 1 and 2 are decided on the
  // first draw of a mesh in a new frame, since only then is the previous frame's placement set
  // complete; level 3 compares that same first draw's constants against the previous frame's.
  void D3D9Rtx::trackUe3ConstantChurn(const XXH64_hash_t iaKey, const RasterGeometry& geoData) {
    ScopedCpuProfileZone();

    const D3D9CommonShader* vertexShader =
      d3d9State().vertexShader.ptr() != nullptr ? d3d9State().vertexShader->GetCommonShader() : nullptr;
    if (vertexShader == nullptr) {
      return;
    }

    const uint32_t floatConstRegCount =
      std::min(m_parent->m_consts[DxsoProgramTypes::VertexShader].meta.maxConstIndexF,
               uint32_t(caps::MaxFloatConstantsSoftware));
    if (floatConstRegCount == 0) {
      return;
    }

    const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();
    const Vector4* floatConsts = &d3d9State().vsConsts.fConsts[0];

    // Keyed per mesh, not per placement: the transform is recorded inside the entry so that a
    // moving transform and a moving IA identity stay distinguishable.
    const XXH64_hash_t transformHash =
      XXH3_64bits(&m_activeDrawCallState.transformData.objectToWorld, sizeof(Matrix4));

    // Hash the registers the transform was extracted from, using the current draw's CTAB rather
    // than the entry's cached ranges so it is right even on the first sighting.
    XXH64_hash_t rawTransformHash = 0;
    if (m_currentUe3CtabInfo.has_value()) {
      const Ue3VsShaderCtabInfo& ctabInfo = *m_currentUe3CtabInfo;
      auto foldRange = [&](const uint32_t base, const uint32_t count) {
        if (count == 0 || base >= floatConstRegCount) {
          return;
        }
        const uint32_t end = std::min(base + count, floatConstRegCount);
        rawTransformHash = XXH3_64bits_withSeed(
          &floatConsts[base], size_t(end - base) * sizeof(Vector4), rawTransformHash);
      };
      if (ctabInfo.hasLocalToWorld) {
        foldRange(ctabInfo.localToWorldRegisterIndex, ctabInfo.localToWorldRegisterCount);
      }
      if (ctabInfo.hasWorldToLocal) {
        foldRange(ctabInfo.worldToLocalRegisterIndex, ctabInfo.worldToLocalRegisterCount);
      }
    }

    auto it = m_ue3ConstantChurn.find(iaKey);
    if (it == m_ue3ConstantChurn.end()) {
      if (m_ue3ConstantChurn.size() >= m_frameOptions.ue3VertexConstantChurnMaxTrackedDraws) {
        // An IA identity that changes every frame never revisits a key, so without eviction the
        // sample fills once with entries that can never be compared and the diagnostic goes blind.
        m_ue3ConstantChurn.erase_if([currentFrame](auto stale) {
          return stale->second.lastFrameSeen + 2 < currentFrame;
        });
        if (m_ue3ConstantChurn.size() >= m_frameOptions.ue3VertexConstantChurnMaxTrackedDraws) {
          return;
        }
      }
      it = m_ue3ConstantChurn.emplace(iaKey, Ue3ChurnMeshEntry {}).first;
      ++m_ue3ChurnMeshKeysCreated;
    }

    Ue3ChurnMeshEntry& entry = it->second;

    // Level 1 and 2 are evaluated on the first draw of this mesh in a new frame, because only then
    // is the previous frame's set of placements complete.
    const bool firstTouchThisFrame = entry.currentSetFrame != currentFrame;
    if (firstTouchThisFrame) {
      if (entry.currentSetFrame != 0) {
        std::sort(entry.transformsThisFrame.begin(), entry.transformsThisFrame.end());
        std::sort(entry.rawTransformsThisFrame.begin(), entry.rawTransformsThisFrame.end());
        const XXH64_hash_t setHash = XXH3_64bits(
          entry.transformsThisFrame.data(), entry.transformsThisFrame.size() * sizeof(XXH64_hash_t));
        const XXH64_hash_t rawSetHash = XXH3_64bits(
          entry.rawTransformsThisFrame.data(), entry.rawTransformsThisFrame.size() * sizeof(XXH64_hash_t));
        const uint32_t setSize = uint32_t(entry.transformsThisFrame.size());

        // Compare two *completed* sets, and only when they belong to consecutive frames.
        if (entry.completedSetFrame != 0 && entry.completedSetFrame + 1 == entry.currentSetFrame) {
          ++m_ue3ChurnMeshRevisited;
          const bool extractedIdentical = setHash == entry.completedSetHash;
          const bool rawIdentical = rawSetHash == entry.completedRawSetHash;

          if (extractedIdentical) {
            ++m_ue3ChurnTransformSetIdentical;
          } else {
            ++m_ue3ChurnTransformSetDiffered;
            if (setSize != entry.completedSetSize) {
              ++m_ue3ChurnTransformSetSizeChanged;
            }
          }
          if (rawIdentical) {
            ++m_ue3ChurnRawTransformSetIdentical;
            if (!extractedIdentical) {
              // Same registers in, different matrices out.
              ++m_ue3ChurnExtractionUnstable;
            }
          }
        }

        entry.completedSetHash = setHash;
        entry.completedRawSetHash = rawSetHash;
        entry.completedSetSize = setSize;
        entry.completedSetFrame = entry.currentSetFrame;
      }
      entry.transformsThisFrame.clear();
      entry.rawTransformsThisFrame.clear();
      entry.currentSetFrame = currentFrame;
    }
    entry.transformsThisFrame.push_back(transformHash);
    entry.rawTransformsThisFrame.push_back(rawTransformHash);

    // Level 3: do any constants move that are neither camera-derived nor part of the object
    // transform? Those are the only ones that could destabilise geometry identity spuriously.
    // Excluding the transform registers is what makes this independent of which placement of the
    // mesh happened to be drawn first in each frame.
    const bool canCompareConstants =
      firstTouchThisFrame &&
      entry.lastFrameSeen != 0 &&
      entry.lastFrameSeen + 1 == currentFrame &&
      entry.vsBytecodeHash == vertexShader->GetBytecodeHash() &&
      entry.floatConsts.size() == floatConstRegCount;

    if (canCompareConstants) {
      auto isExcludedRegister = [&entry](const uint32_t reg) {
        auto inRange = [reg](const uint32_t base, const uint32_t count) {
          return count > 0 && reg >= base && reg < base + count;
        };
        return inRange(entry.viewProjReg, entry.viewProjRegCount) ||
               inRange(entry.cameraPosReg, entry.cameraPosRegCount) ||
               inRange(entry.localToWorldReg, entry.localToWorldRegCount) ||
               inRange(entry.worldToLocalReg, entry.worldToLocalRegCount);
      };

      const bool viewMoved =
        std::memcmp(&entry.worldToView, &m_activeDrawCallState.transformData.worldToView, sizeof(Matrix4)) != 0;
      // Dumps only while the view is moving: a camera-derived constant is the interesting
      // hypothesis, and a still-camera example cannot distinguish one.
      const bool wantDetail = viewMoved && m_ue3ConstantChurnDetailDumps < 8;

      std::string detail;
      uint32_t changedRegisters = 0;
      for (uint32_t reg = 0; reg < floatConstRegCount; reg++) {
        if (std::memcmp(&entry.floatConsts[reg], &floatConsts[reg], sizeof(Vector4)) == 0) {
          continue;
        }
        if (isExcludedRegister(reg)) {
          continue;
        }
        ++changedRegisters;
        Ue3ChurnRegisterTally& tally = m_ue3ConstantChurnByRegister[reg];
        ++tally.count;
        tally.vsBytecodeHash = entry.vsBytecodeHash;

        if (wantDetail && changedRegisters <= 6) {
          const Vector4& was = entry.floatConsts[reg];
          const Vector4& now = floatConsts[reg];
          detail += str::format(
            detail.empty() ? "" : ", ",
            describeUe3VsConstantRegister(entry.vsBytecodeHash, reg),
            " (", was.x, ",", was.y, ",", was.z, ",", was.w, ")->(",
            now.x, ",", now.y, ",", now.z, ",", now.w, ")");
        }
      }

      ++m_ue3ConstantChurnComparisons;
      if (changedRegisters > 0) {
        ++m_ue3ChurnOtherConstantsChanged;
        if (viewMoved) {
          ++m_ue3ConstantChurnWhileViewMoved;
        } else {
          ++m_ue3ConstantChurnWhileViewStill;
        }
      }

      if (!detail.empty()) {
        ++m_ue3ConstantChurnDetailDumps;
        Logger::info(str::format(
          "[RTX-Compatibility][UE3-ConstantChurn] same mesh, next frame, ignoring camera and transform "
          "registers: vs=0x", std::hex, entry.vsBytecodeHash, std::dec,
          " vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
          " verts=", geoData.vertexCount,
          " changedRegs=", changedRegisters, ": ", detail));
      }
    }

    if (firstTouchThisFrame) {
      entry.lastFrameSeen = currentFrame;
      entry.vsBytecodeHash = vertexShader->GetBytecodeHash();
      entry.worldToView = m_activeDrawCallState.transformData.worldToView;
      entry.floatConsts.assign(floatConsts, floatConsts + floatConstRegCount);

      entry.viewProjReg = kUe3VsrViewProjMatrixRegister;
      entry.viewProjRegCount = 4;
      entry.cameraPosReg = kUe3VsrViewOriginRegister;
      entry.cameraPosRegCount = 1;
      entry.localToWorldReg = 0;
      entry.localToWorldRegCount = 0;
      entry.worldToLocalReg = 0;
      entry.worldToLocalRegCount = 0;
      if (m_currentUe3CtabInfo.has_value()) {
        const Ue3VsShaderCtabInfo& ctabInfo = *m_currentUe3CtabInfo;
        if (ctabInfo.hasViewProjectionMatrix && ctabInfo.viewProjectionMatrixRegisterCount > 0) {
          entry.viewProjReg = ctabInfo.viewProjectionMatrixRegisterIndex;
          entry.viewProjRegCount = ctabInfo.viewProjectionMatrixRegisterCount;
        }
        if (ctabInfo.hasCameraPosition && ctabInfo.cameraPositionRegisterCount > 0) {
          entry.cameraPosReg = ctabInfo.cameraPositionRegisterIndex;
          entry.cameraPosRegCount = ctabInfo.cameraPositionRegisterCount;
        }
        if (ctabInfo.hasLocalToWorld && ctabInfo.localToWorldRegisterCount > 0) {
          entry.localToWorldReg = ctabInfo.localToWorldRegisterIndex;
          entry.localToWorldRegCount = ctabInfo.localToWorldRegisterCount;
        }
        if (ctabInfo.hasWorldToLocal && ctabInfo.worldToLocalRegisterCount > 0) {
          entry.worldToLocalReg = ctabInfo.worldToLocalRegisterIndex;
          entry.worldToLocalRegCount = ctabInfo.worldToLocalRegisterCount;
        }
      }
    }
  }

  void D3D9Rtx::reportUe3ConstantChurn() {
    if (!m_frameOptions.ue3LogVertexConstantChurn) {
      return;
    }

    const uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();
    constexpr uint32_t kReportIntervalFrames = 300;
    if (currentFrame - m_ue3ConstantChurnReportFrameStamp < kReportIntervalFrames) {
      return;
    }
    m_ue3ConstantChurnReportFrameStamp = currentFrame;

    const uint64_t meshKeys = m_ue3ChurnMeshKeysCreated;
    const uint64_t revisits = m_ue3ChurnMeshRevisited;

    // The interpretation guide is worth saying once; repeating it every window buries the numbers.
    ONCE(Logger::info(
      "[RTX-Compatibility][UE3-ConstantChurn] reading the levels below: level 1 is whether a mesh's "
      "input-assembler identity comes back at all (if not, nothing downstream can repeat and the draws are "
      "re-uploading their vertex/index data); level 2 is whether its placements are the same, compared as a "
      "multiset so draw order does not matter, with a changed instance count meaning ordinary culling rather "
      "than movement; level 3 is whether any other constant moved, which is the only one of the three that "
      "rtx.d3d9.ue3ExcludePlacementFromVertexShaderHash cannot address."));

    if (revisits == 0) {
      ONCE(Logger::warn(str::format(
        "[RTX-Compatibility][UE3-ConstantChurn] level 1: ", meshKeys, " mesh keys minted and not one "
        "input-assembler identity came back on the next frame, so no cache key can repeat regardless of "
        "transforms or constants.")));
      m_ue3ChurnMeshKeysCreated = 0;
      return;
    }

    auto pct = [](const uint64_t n, const uint64_t of) {
      return of > 0 ? uint32_t((n * 100ull) / of) : 0u;
    };

    const uint64_t setComparisons = m_ue3ChurnTransformSetIdentical + m_ue3ChurnTransformSetDiffered;

    Logger::info(str::format(
      "[RTX-Compatibility][UE3-ConstantChurn] level 1: ", revisits, " next-frame revisits against ",
      meshKeys, " new mesh keys, ", m_ue3ConstantChurn.size(), " sampled. level 2: of ", setComparisons,
      " comparisons the placement set was identical ", pct(m_ue3ChurnTransformSetIdentical, setComparisons),
      "%, differed ", pct(m_ue3ChurnTransformSetDiffered, setComparisons), "% (",
      m_ue3ChurnTransformSetSizeChanged, " with a changed instance count), and the raw LocalToWorld registers "
      "behind them were identical ", pct(m_ue3ChurnRawTransformSetIdentical, setComparisons),
      "%. level 3: ", pct(m_ue3ChurnOtherConstantsChanged, m_ue3ConstantChurnComparisons),
      "% of ", m_ue3ConstantChurnComparisons, " comparisons moved another constant (view had moved on ",
      m_ue3ConstantChurnWhileViewMoved, ", was still on ", m_ue3ConstantChurnWhileViewStill, ")."));

    // The conclusions are one-shot: they identify a property of the title, not of this window.
    if (m_ue3ChurnExtractionUnstable > 0) {
      ONCE(Logger::warn(str::format(
        "[RTX-Compatibility][UE3-ConstantChurn] ", pct(m_ue3ChurnExtractionUnstable, setComparisons),
        "% of comparisons had identical raw LocalToWorld registers but a different extracted objectToWorld. "
        "Same input, different output means this runtime's transform extraction is not reproducible, which is a "
        "bug here rather than a property of the game.")));
    } else if (m_ue3ChurnTransformSetDiffered > 0) {
      ONCE(Logger::info(
        "[RTX-Compatibility][UE3-ConstantChurn] every differing placement set also had differing raw "
        "LocalToWorld registers, so the game uploads a new transform each frame for these draws and the "
        "extraction of it is reproducible. Those registers are in the stable VS hash, hence in "
        "rules::FullGeometryHash, so DrawCallCache::exactMatch cannot match them across frames and a fresh "
        "BlasEntry is allocated per draw per frame. rtx.d3d9.ue3ExcludePlacementFromVertexShaderHash addresses "
        "this, at the cost described in its documentation."));
    }

    // Only worth reporting when frequent enough to matter; a handful of shadow-atlas reallocations
    // per window is not a finding.
    constexpr uint32_t kConstantChurnWarnPercent = 5;
    if (pct(m_ue3ChurnOtherConstantsChanged, m_ue3ConstantChurnComparisons) >= kConstantChurnWarnPercent) {
      ONCE(Logger::warn(
        "[RTX-Compatibility][UE3-ConstantChurn] a constant that is neither camera-derived nor part of the object "
        "transform is moving often enough to destabilise geometry identity on its own. The register tally names "
        "it; changes only while the view moves point at a camera-derived constant, changes with the view still "
        "are animation or time driven."));
    }

    if (!m_ue3ConstantChurnByRegister.empty()) {
      // Busiest first: that ordering is the answer to "which constant is doing this".
      std::vector<std::pair<uint64_t, uint32_t>> byCount;
      byCount.reserve(m_ue3ConstantChurnByRegister.size());
      for (const auto& [reg, tally] : m_ue3ConstantChurnByRegister) {
        byCount.emplace_back(tally.count, reg);
      }
      std::sort(byCount.rbegin(), byCount.rend());

      std::string top;
      const size_t shown = std::min<size_t>(byCount.size(), 12);
      for (size_t i = 0; i < shown; i++) {
        const uint32_t reg = byCount[i].second;
        top += str::format(
          top.empty() ? "" : ", ",
          describeUe3VsConstantRegister(m_ue3ConstantChurnByRegister[reg].vsBytecodeHash, reg),
          " x", byCount[i].first);
      }

      Logger::info(str::format(
        "[RTX-Compatibility][UE3-ConstantChurn] registers that moved (", byCount.size(),
        " distinct, busiest first): ", top));
    }

    m_ue3ConstantChurnByRegister.clear();
    m_ue3ChurnMeshKeysCreated = 0;
    m_ue3ChurnMeshRevisited = 0;
    m_ue3ChurnTransformSetIdentical = 0;
    m_ue3ChurnTransformSetDiffered = 0;
    m_ue3ChurnTransformSetSizeChanged = 0;
    m_ue3ChurnRawTransformSetIdentical = 0;
    m_ue3ChurnExtractionUnstable = 0;
    m_ue3ConstantChurnComparisons = 0;
    m_ue3ChurnOtherConstantsChanged = 0;
    m_ue3ConstantChurnWhileViewMoved = 0;
    m_ue3ConstantChurnWhileViewStill = 0;
  }

  // NV-DXVK start: draw disposition statistics
  void D3D9Rtx::reportDrawDispositionStats() {
    DrawDispositionStats& s = m_drawDispositionStats;
    if (!RtxGpuPassTimer::isEnabled()) {
      s = DrawDispositionStats {};
      return;
    }

    ++s.frames;
    constexpr uint32_t kReportIntervalFrames = 300;
    if (s.frames < kReportIntervalFrames) {
      return;
    }

    const double inv = 1.0 / static_cast<double>(s.frames);
    Logger::info(str::format(
      "[RTX-DrawStats] per frame over ", s.frames, " frames: draws=", static_cast<double>(s.draws) * inv,
      " rayTraced=", static_cast<double>(s.rayTraced) * inv,
      " rasterized(originalDraw)=", static_cast<double>(s.rasterized) * inv, " (", static_cast<double>(s.rasterizedPrims) * inv, " prims)",
      " ofWhichForVertexCapture=", static_cast<double>(s.rasterizedForCapture) * inv, " (", static_cast<double>(s.rasterizedForCapturePrims) * inv, " prims)",
      " postInjection=", static_cast<double>(s.rasterizedPostInjection) * inv,
      " ignored=", static_cast<double>(s.ignored) * inv,
      " renderTargetCopiesSkipped=", static_cast<double>(s.renderTargetCopiesSkipped) * inv));
    s = DrawDispositionStats {};
  }

  // Shader, material and bound-texture hashes for a draw, so a warning about one names something
  // that can be looked up in the texture picker or fed to rtx.d3d9.ue3TraceDrawTextureHashes.
  std::string D3D9Rtx::describeUe3DrawIdentity() const {
    XXH64_hash_t vsHash = kEmptyHash;
    if (m_parent->UseProgrammableVS() && d3d9State().vertexShader.ptr() != nullptr) {
      vsHash = d3d9State().vertexShader->GetCommonShader()->GetBytecodeHash();
    }
    XXH64_hash_t psHash = kEmptyHash;
    if (m_parent->UseProgrammablePS() && d3d9State().pixelShader.ptr() != nullptr) {
      psHash = d3d9State().pixelShader->GetCommonShader()->GetBytecodeHash();
    }

    std::string textures;
    for (uint32_t stage = 0; stage < LegacyMaterialData::kMaxSupportedTextures; stage++) {
      const XXH64_hash_t imageHash = m_activeDrawCallState.materialData.colorTextures[stage].getImageHash();
      if (imageHash == kEmptyHash) {
        continue;
      }
      textures += str::format(textures.empty() ? "" : " ", "s", stage, ":0x", std::hex, imageHash, std::dec);
    }

    return str::format(
      "vs=0x", std::hex, vsHash,
      " ps=0x", psHash,
      " materialHash=0x", m_activeDrawCallState.materialData.getHash(), std::dec,
      " textures=[", textures.empty() ? "none" : textures, "]");
  }

  std::string D3D9Rtx::describeUe3VertexDeclaration() const {
    if (d3d9State().vertexDecl == nullptr) {
      return "none";
    }

    std::string out;
    for (const auto& element : d3d9State().vertexDecl->GetElements()) {
      out += str::format(out.empty() ? "" : " | ",
                         "s", uint32_t(element.Stream), ":",
                         ue3DeclUsageName(element.Usage), uint32_t(element.UsageIndex),
                         " ", ue3DeclTypeName(element.Type),
                         " @", uint32_t(element.Offset));
    }
    return out.empty() ? "empty" : out;
  }

  std::string D3D9Rtx::describeUe3DrawInstancing(const VertexContext vertexContext[caps::MaxStreams]) const {
    const Ue3InstancingInfo& info = m_currentUe3Instancing;

    std::string streams;
    for (uint32_t s = 0; s < caps::MaxStreams; s++) {
      const VertexContext& ctx = vertexContext[s];
      if (ctx.mappedSlice.mapPtr == nullptr && d3d9State().streamFreq[s] == 1) {
        continue;
      }
      const bool dynamic =
        ctx.pVBO != nullptr && ctx.pVBO->Desc() != nullptr &&
        (ctx.pVBO->Desc()->Usage & D3DUSAGE_DYNAMIC) != 0;
      streams += str::format(streams.empty() ? "" : " | ",
                             "s", s,
                             " freq=0x", std::hex, uint32_t(d3d9State().streamFreq[s]), std::dec,
                             " stride=", ctx.stride,
                             " dynamic=", dynamic ? 1 : 0);
    }

    std::string transform = "none";
    if (info.hasInstanceTransform) {
      transform = str::format("stream=", info.transformStream,
                              " offset@", info.offsetByteOffset,
                              " axes@", info.axisByteOffsets[0],
                              "/", info.axisByteOffsets[1],
                              "/", info.axisByteOffsets[2]);
    }

    return str::format("instances=", info.instanceCount,
                       " instanceDataStreams=0x", std::hex, info.instanceDataStreamMask, std::dec,
                       " instanceTransform=[", transform, "]",
                       " decomposed=", uint32_t(m_ue3DecomposedInstances.size()),
                       " streams=[", streams.empty() ? "none" : streams, "]");
  }

  void D3D9Rtx::logUe3InstancedDrawOnce(const DrawContext& drawContext,
                                        const VertexContext vertexContext[caps::MaxStreams],
                                        const RasterGeometry& geoData) {
    if (!m_frameOptions.ue3LogInstancedDraws || !m_currentUe3Instancing.isInstanced()) {
      return;
    }

    XXH64_hash_t vsHash = kEmptyHash;
    if (m_parent->UseProgrammableVS() && d3d9State().vertexShader.ptr() != nullptr) {
      vsHash = d3d9State().vertexShader->GetCommonShader()->GetBytecodeHash();
    }
    XXH64_hash_t psHash = kEmptyHash;
    if (m_parent->UseProgrammablePS() && d3d9State().pixelShader.ptr() != nullptr) {
      psHash = d3d9State().pixelShader->GetCommonShader()->GetBytecodeHash();
    }

    // The instance count varies with the particle population, so it is deliberately left out of
    // the key: one line per (shader, declaration, factory) is the point.
    struct Key {
      XXH64_hash_t vsHash;
      XXH64_hash_t psHash;
      const void* pVertexDecl;
      uint32_t vertexFactory;
      uint32_t passType;
      uint32_t positionSource;
      uint32_t hasInstanceTransform;
    };
    const Key key = {
      vsHash,
      psHash,
      d3d9State().vertexDecl.ptr(),
      uint32_t(m_currentUe3VertexFactory),
      uint32_t(m_currentUe3PassType),
      uint32_t(m_activeCapturePositionSource),
      m_currentUe3Instancing.hasInstanceTransform ? 1u : 0u,
    };
    if (!m_loggedUe3InstancedDraws.insert(XXH3_64bits(&key, sizeof(key))).second) {
      return;
    }

    Logger::info(str::format(
      "[RTX-Compatibility][UE3-Instanced] vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
      " pass=", describeUe3PassType(m_currentUe3PassType),
      " posSource=", describeUe3CapturePositionSource(m_activeCapturePositionSource),
      " ", describeUe3DrawInstancing(vertexContext),
      " prims=", drawContext.PrimitiveCount,
      " verts=", geoData.vertexCount,
      " indices=", geoData.indexCount,
      " ", describeUe3DrawIdentity(),
      " decl=[", describeUe3VertexDeclaration(), "]"));
  }

  void D3D9Rtx::logUe3TracedDrawOnce(const DrawContext& drawContext,
                                     const VertexContext vertexContext[caps::MaxStreams],
                                     RasterGeometry& geoData) {
    if (m_frameOptions.ue3TraceDrawTextureHashes == nullptr ||
        m_frameOptions.ue3TraceDrawTextureHashes->empty()) {
      return;
    }

    // Any bound colour texture matching the list arms the dossier, not just the chosen albedo:
    // the point is to find a draw from a hash read off the texture picker.
    bool matched = false;
    for (const auto& texture : m_activeDrawCallState.materialData.colorTextures) {
      const XXH64_hash_t imageHash = texture.getImageHash();
      if (imageHash != kEmptyHash && lookupHash(*m_frameOptions.ue3TraceDrawTextureHashes, imageHash)) {
        matched = true;
        break;
      }
    }
    if (!matched) {
      return;
    }

    XXH64_hash_t vsHash = kEmptyHash;
    if (m_parent->UseProgrammableVS() && d3d9State().vertexShader.ptr() != nullptr) {
      vsHash = d3d9State().vertexShader->GetCommonShader()->GetBytecodeHash();
    }
    XXH64_hash_t psHash = kEmptyHash;
    if (m_parent->UseProgrammablePS() && d3d9State().pixelShader.ptr() != nullptr) {
      psHash = d3d9State().pixelShader->GetCommonShader()->GetBytecodeHash();
    }

    const XXH64_hash_t materialHash = m_activeDrawCallState.materialData.getHash();

    struct Key {
      XXH64_hash_t vsHash;
      XXH64_hash_t psHash;
      XXH64_hash_t materialHash;
      const void* pVertexDecl;
      uint32_t vertexFactory;
      uint32_t passType;
      uint32_t positionSource;
      uint32_t reserved;
    };
    const Key key = {
      vsHash,
      psHash,
      materialHash,
      d3d9State().vertexDecl.ptr(),
      uint32_t(m_currentUe3VertexFactory),
      uint32_t(m_currentUe3PassType),
      uint32_t(m_activeCapturePositionSource),
      0u,
    };
    if (!m_loggedUe3TracedDraws.insert(XXH3_64bits(&key, sizeof(key))).second) {
      return;
    }

    // Geometry hashing normally completes on a worker and is collected on the CS thread. Sync it
    // here so the dossier can report the value replacements anchor on; finalizeGeometryHashes
    // accepts an already-resolved RasterGeometry, so consuming the future is safe.
    if (geoData.futureGeometryHashes.valid()) {
      geoData.hashes = geoData.futureGeometryHashes.get();
    }
    const XXH64_hash_t assetGeometryHash = geoData.getHashForRule(RtxOptions::geometryAssetHashRule());
    const XXH64_hash_t fullGeometryHash = geoData.getHashForRule<rules::FullGeometryHash>();

    std::string textures;
    for (uint32_t stage = 0; stage < LegacyMaterialData::kMaxSupportedTextures; stage++) {
      const XXH64_hash_t imageHash = m_activeDrawCallState.materialData.colorTextures[stage].getImageHash();
      if (imageHash == kEmptyHash) {
        continue;
      }
      textures += str::format(textures.empty() ? "" : " ", "s", stage, ":0x", std::hex, imageHash, std::dec);
    }

    std::string instanceTransforms;
    {
      constexpr size_t kMaxLoggedTransforms = 4;
      const size_t count = std::min(kMaxLoggedTransforms, m_ue3DecomposedInstances.size());
      for (size_t i = 0; i < count; i++) {
        const Ue3DecomposedInstance& instance = m_ue3DecomposedInstances[i];
        instanceTransforms += str::format("\n    instance[", instance.sourceIndex, "]=",
                                          formatMatrixRows(instance.instanceToObject));
      }
      if (m_ue3DecomposedInstances.size() > count) {
        instanceTransforms += str::format("\n    ... ",
                                          uint32_t(m_ue3DecomposedInstances.size() - count),
                                          " more");
      }
    }

    Logger::info(str::format(
      "[RTX-Compatibility][UE3-DrawTrace] vf=", describeUe3VertexFactory(m_currentUe3VertexFactory),
      " pass=", describeUe3PassType(m_currentUe3PassType),
      " posSource=", describeUe3CapturePositionSource(m_activeCapturePositionSource),
      " vs=0x", std::hex, vsHash,
      " ps=0x", psHash,
      " materialHash=0x", materialHash,
      " assetGeometryHash=0x", assetGeometryHash,
      " fullGeometryHash=0x", fullGeometryHash, std::dec,
      "\n    prims=", drawContext.PrimitiveCount,
      " verts=", geoData.vertexCount,
      " indices=", geoData.indexCount,
      " topology=", uint32_t(geoData.topology),
      " cull=", uint32_t(geoData.cullMode),
      " textures=[", textures.empty() ? "none" : textures, "]",
      "\n    ", describeUe3DrawInstancing(vertexContext),
      "\n    decl=[", describeUe3VertexDeclaration(), "]",
      "\n    objectToWorld=", formatMatrixRows(m_activeDrawCallState.transformData.objectToWorld),
      instanceTransforms));
  }

  void D3D9Rtx::logNonPrimaryRenderTargetOnce() const {
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
  }

}
