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

  // The per-draw facts the UE3 texture hooks share: the pixel shader's sampler analysis, the
  // vertex factory, and the UV packing conventions it implies.
  Ue3TextureState D3D9Rtx::beginUe3TextureState(const bool programmablePs) {
    Ue3TextureState state;
    state.inferredPs = nullptr;
    state.inferredPsHash = 0;
    state.inferredPsEntry = nullptr;
    state.vfType = m_currentUe3VertexFactory;
    state.isUe3GpuSkinVF = state.vfType == Ue3VertexFactoryType::GPUSkin || state.vfType == Ue3VertexFactoryType::GPUSkinMorph;
    state.isUe3TerrainVF = state.vfType == Ue3VertexFactoryType::Terrain || state.vfType == Ue3VertexFactoryType::TerrainMorph;
    state.isUe3ParticleVF =
      state.vfType == Ue3VertexFactoryType::Particle ||
      state.vfType == Ue3VertexFactoryType::ParticleBeamTrail ||
      state.vfType == Ue3VertexFactoryType::LensFlare;
    // Instanced mesh particles compile the same FoliageVertexFactory.usf and pack their UVs the
    // same way, so they follow foliage rather than the camera-facing particle factories.
    state.isUe3FoliageVF =
      state.vfType == Ue3VertexFactoryType::Foliage ||
      state.vfType == Ue3VertexFactoryType::ParticleInstancedMesh;
    state.isUe3SpeedTreeVF = state.vfType == Ue3VertexFactoryType::SpeedTree;
    state.isUe3LocalDecalVF = state.vfType == Ue3VertexFactoryType::LocalDecal;
    state.isUe3MorphVF = state.vfType == Ue3VertexFactoryType::GPUSkinMorph;

    state.likelyGpuSkinnedMesh = state.isUe3GpuSkinVF || [&]() {
      if (d3d9State().vertexDecl.ptr() == nullptr) {
        return false;
      }

      for (const auto& element : d3d9State().vertexDecl->GetElements()) {
        if (element.Usage == D3DDECLUSAGE_BLENDWEIGHT ||
            element.Usage == D3DDECLUSAGE_BLENDINDICES) {
          return true;
        }
      }
      return false;
    }();
    state.ue3VsHints =
      (m_parent->UseProgrammableVS() &&
       d3d9State().vertexShader.ptr() != nullptr &&
       m_currentUe3CtabInfo.has_value())
        ? &(*m_currentUe3CtabInfo)
        : nullptr;
    state.likelyUe3DecalUvSpace =
      state.isUe3LocalDecalVF ||
      (state.ue3VsHints != nullptr &&
       (state.ue3VsHints->hasDecalTransform ||
        state.ue3VsHints->hasDecalLocation ||
        state.ue3VsHints->hasDecalOffset));
    state.likelyUe3TerrainUvSpace =
      state.isUe3TerrainVF ||
      (state.ue3VsHints != nullptr &&
       (state.ue3VsHints->hasLightMapCoordinateScaleBias ||
        state.ue3VsHints->hasShadowCoordinateScaleBias));
    state.likelyUe3BillboardUvSpace =
      state.isUe3ParticleVF ||
      state.isUe3SpeedTreeVF ||
      (state.ue3VsHints != nullptr &&
       (state.ue3VsHints->hasTextureCoordinateScaleBias ||
        state.ue3VsHints->hasViewToLocal ||
        state.ue3VsHints->hasWindMatrices));
    state.likelyUe3FlexiblePackedUvPath =
      !state.likelyGpuSkinnedMesh &&
      (state.likelyUe3DecalUvSpace || state.likelyUe3TerrainUvSpace || state.likelyUe3BillboardUvSpace || state.isUe3FoliageVF);
    state.likelyPackedUvConventions = state.likelyGpuSkinnedMesh || state.likelyUe3FlexiblePackedUvPath;
    if (programmablePs) {
      if (m_frameOptions.ue3EngineMode && d3d9State().pixelShader.ptr() != nullptr) {
        // Measures the bytecode analysis a shader's first sighting pays; the result is cached,
        // so the zone is near-empty on every later draw.
        ScopedCpuProfileZoneN("UE3 PS sampler analysis");
        state.inferredPs = d3d9State().pixelShader->GetCommonShader();
        state.inferredPsEntry = getOrInitPsSamplerTexcoordEntry(state.inferredPs, state.inferredPsHash);
      }
    }
    return state;
  }

  bool D3D9Rtx::resolveInferredSamplerOffset(const PsSamplerTexcoordEntry* entry, const uint32_t stage, float& outU, float& outV) const {
    outU = 0.0f;
    outV = 0.0f;

    if (entry == nullptr || stage >= caps::MaxTexturesPS) {
      return false;
    }

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
  }

  bool D3D9Rtx::hasNonZeroInferredSamplerOffset(const PsSamplerTexcoordEntry* entry, const uint32_t stage) const {
    float uOffset = 0.0f;
    float vOffset = 0.0f;
    if (!resolveInferredSamplerOffset(entry, stage, uOffset, vOffset)) {
      return false;
    }

    constexpr float kOffsetEps = 1e-5f;
    return std::abs(uOffset) > kOffsetEps || std::abs(vOffset) > kOffsetEps;
  }

  // First sighting of a pixel shader analyses its samplers; the result is cached by bytecode hash.
  PsSamplerTexcoordEntry* D3D9Rtx::getOrInitPsSamplerTexcoordEntry(const D3D9CommonShader* ps, XXH64_hash_t& outHash) {
    if (ps == nullptr) {
      return nullptr;
    }

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
        if ((entry.samplers[s].semanticFlags & kPsSamplerSemanticLightmap) != 0) {
          entry.lightmapSamplerMask |= (1u << s);
        }
      }

      // Expression flags are tracked per register, so a sampler inherits what fxc packed into its register's
      // spare lanes (the lightmap UV under TdBicubicFiltering). Flags the colour-term analysis shows to be
      // impossible on the coordinate lanes are cleared, never added.
      const std::vector<uint8_t>& bytecode = ps->GetBytecode();
      const DxsoColorTermResult coordFacts = analyzeDxsoColorTerms(
        reinterpret_cast<const uint32_t*>(bytecode.data()), bytecode.size() / sizeof(uint32_t), DxsoColorTermInputs {});
      if (coordFacts.analyzed) {
        for (uint32_t s = 0; s < caps::MaxTexturesPS && s < kDxsoColorTermMaxSamplers; s++) {
          if (entry.samplers[s].sampleCount == 0 || coordFacts.samplerSampleCount[s] == 0) {
            continue;
          }
          const uint8_t expr = coordFacts.samplerCoordExpr[s];
          uint16_t& flags = entry.samplers[s].expressionFlags;
          const uint16_t before = flags;
          if ((expr & DxsoCoordExpr_Arith) == 0) {
            // a plain interpolant read: the only transform it can carry is a non-.xy swizzle
            flags &= ~uint16_t(kPsSamplerExprUvOffset | kPsSamplerExprUvAnimated | kPsSamplerExprBlendMath);
            const bool swizzled =
              entry.samplers[s].coordCompValid &&
              (entry.samplers[s].coordCompU != 0 || entry.samplers[s].coordCompV != 1);
            if (!swizzled) {
              flags &= ~uint16_t(kPsSamplerExprUvTransform);
            }
          }
          if ((expr & DxsoCoordExpr_Offset) == 0) {
            flags &= ~uint16_t(kPsSamplerExprUvOffset);
          }
          const bool animatedPossible =
            (expr & (DxsoCoordExpr_Wrap | DxsoCoordExpr_UnknownOffset)) != 0 ||
            entry.samplers[s].sampleCount >= 2u ||
            (flags & kPsSamplerExprUvTimeDriven) != 0;
          if (!animatedPossible) {
            flags &= ~uint16_t(kPsSamplerExprUvAnimated);
          }
          entry.samplerExpressionFlagsCleared[s] = uint16_t(before & ~flags);
        }
      }
    }

    return &entry;
  }

  // UE3's deterministic UV resolution: the texcoord set and components the surface reads, proven
  // from the pixel and vertex shader bytecode where possible.
  void D3D9Rtx::resolveUe3Texcoords(Ue3TextureState& ue3, const uint32_t firstStage, const uint32_t stageStateIdx,
                                    uint32_t& texcoordIdx, uint32_t& iaTexcoordIdx) {
    const D3D9CommonShader*& inferredPs = ue3.inferredPs;
    XXH64_hash_t& inferredPsHash = ue3.inferredPsHash;
    PsSamplerTexcoordEntry*& inferredPsEntry = ue3.inferredPsEntry;
    const bool forceIaTexcoordForOutlier = m_forceIaTexcoordForOutlier;

    auto getOrInitVsTexcoordTraceEntry = [&](const D3D9CommonShader* vs,
                                             const uint32_t outputReg,
                                             const uint8_t compU,
                                             const uint8_t compV) -> Ue3VsTexcoordTraceEntry* {
      if (vs == nullptr || outputReg == std::numeric_limits<uint32_t>::max()) {
        return nullptr;
      }

      const XXH64_hash_t vsHash = vs->GetBytecodeHash();
      if (vsHash == 0) {
        return nullptr;
      }

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
      if (term.inexact) {
        return false;
      }
      if (!uvAffineTermPresent(term)) {
        return true;
      }
      float value = term.immValid ? term.imm : 0.0f;
      if (term.constReg >= 0) {
        if (uint32_t(term.constReg) >= caps::MaxFloatConstantsPS) {
          return false;
        }
        value += d3d9State().psConsts.fConsts[uint32_t(term.constReg)][term.constComp & 0x3u] * term.factor;
      }
      if (term.constReg2 >= 0) {
        if (uint32_t(term.constReg2) >= caps::MaxFloatConstantsPS) {
          return false;
        }
        value += d3d9State().psConsts.fConsts[uint32_t(term.constReg2)][term.constComp2 & 0x3u] * term.factor2;
      }
      outValue = value;
      return std::isfinite(outValue);
    };

    auto resolveVsAffineTermValue = [&](const UvAffineTerm& term, const float identity, float& outValue) -> bool {
      outValue = identity;
      if (term.inexact) {
        return false;
      }
      if (!uvAffineTermPresent(term)) {
        return true;
      }
      float value = term.immValid ? term.imm : 0.0f;
      if (term.constReg >= 0) {
        if (uint32_t(term.constReg) >= caps::MaxFloatConstantsSoftware) {
          return false;
        }
        value += d3d9State().vsConsts.fConsts[uint32_t(term.constReg)][term.constComp & 0x3u] * term.factor;
      }
      if (term.constReg2 >= 0) {
        if (uint32_t(term.constReg2) >= caps::MaxFloatConstantsSoftware) {
          return false;
        }
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
      if (entryPtr == nullptr) {
        entryPtr = getOrInitPsSamplerTexcoordEntry(ps, psHash);
      }

      if (entryPtr != nullptr && firstStage < caps::MaxTexturesPS) {
        const auto& entry = *entryPtr;

        // An unproven origin would fall through to the TSS texcoord index, which UE3 never sets. Borrow the
        // interpolant the other material samplers agree on instead, origin only (see "Samplers whose UV origin
        // cannot be proven" in UE3Compatibility.md).
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
            if (s == firstStage || s >= SamplerCount || d3d9State().textures[s] == nullptr) {
              continue;
            }
            const PsSamplerUvOrigin& other = entry.samplerUvOrigin[s];
            if (!other.originValid || !other.sitesAgree) {
              continue;
            }
            // Lightmap and engine buffers legitimately read their own UV set, so they must
            // not vote on the material's.
            const uint8_t semanticFlags = entry.samplers[s].semanticFlags;
            if ((semanticFlags & (kPsSamplerSemanticLightmap | kPsSamplerSemanticEngineAuxiliary)) != 0) {
              continue;
            }
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
            if (origin.validSiteCount == 0 && origin.invalidSiteCount == 0) {
              continue;
            }

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
            if (vsTexcoordOutputReg != std::numeric_limits<uint32_t>::max()) {
              traceEntry = getOrInitVsTexcoordTraceEntry(vs, vsTexcoordOutputReg, m_texcoordCompU, m_texcoordCompV);
            }
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
              if (!uvComponentAffineExact(aff)) {
                return false;
              }
              if (!aff.hasCross) {
                float scale = 1.0f;
                float offset = 0.0f;
                if (!resolvePsAffineTermValue(aff.scale, 1.0f, scale) ||
                    !resolvePsAffineTermValue(aff.offset, 0.0f, offset)) {
                  return false;
                }
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
                  !resolvePsAffineTermValue(aff.offset, 0.0f, offset)) {
                return false;
              }
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
                  !accumulate(aff.crossComponent, cross)) {
                return false;
              }
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
                  if (reg < 0 || reg == lastLogged || uint32_t(reg) >= caps::MaxFloatConstantsPS) {
                    continue;
                  }
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

}
