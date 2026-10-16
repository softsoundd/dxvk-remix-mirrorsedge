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

// Decides on the application thread which draws render from a bake of their own pixel shader
// (rtx.ue3CompositeMaterialTextures), the repeat that bake covers, and the request the CS thread bakes from.
// See "Composite materials" in UE3Compatibility.md.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>

#include "d3d9_rtx.h"
#include "d3d9_rtx_ue3_helpers.h"
#include "d3d9_constant_set.h"
#include "d3d9_device.h"
#include "d3d9_sampler.h"
#include "d3d9_texture.h"
#include "d3d9_util.h"
#include "../dxso/dxso_ue3_composite.h"
#include "../dxso/dxso_util.h"
#include "../dxvk/dxvk_scoped_annotation.h"
#include "../dxvk/rtx_render/rtx_ue3_composite.h"
#include "../spirv/spirv_module.h"
#include "../util/util_string.h"

namespace dxvk {

  namespace {

    // A material drawn with more distinct values of its constants than this is taken to be animating one of them.
    constexpr uint32_t kUe3CompositeMaxConstantSetsPerMaterial = 8;

    constexpr uint32_t kUe3CompositeMinResolution = 64;

    // The request map is rebuilt on demand, so it is simply dropped when it grows past this.
    constexpr size_t kUe3CompositeMaxRequests = 4096;

    // Holds the corner ids (x | y << 1) of the bake's two quads.
    constexpr uint32_t kCornerIds[kUe3CompositeBakeVertexCount] = { 0, 1, 2, 2, 1, 3, 0, 2, 1, 2, 3, 1 };

    constexpr uint32_t packCornerIds() {
      uint32_t packed = 0;
      for (uint32_t i = 0; i < kUe3CompositeBakeVertexCount; i++) {
        packed |= kCornerIds[i] << (2 * i);
      }
      return packed;
    }

    // Selects what one lane of a bake interpolant carries.
    enum BakeLaneSource : uint8_t {
      BakeLane_Value = 0,
      // Carry the UV ramp over [0, period].
      BakeLane_RampU,
      BakeLane_RampV,
      // Carry a ramp so steep that a layer read through it samples its smallest mip, which is its average.
      BakeLane_AveragedU,
      BakeLane_AveragedV,
    };

    // Steepens the ramp past any texture's mip chain at any bake resolution.
    constexpr float kUe3CompositeAveragedRampScale = 65536.0f;

    // Describes one output of the bake's vertex shader, which is a constant with the bake's UV ramp in the lanes
    // the material layers read their coordinate from.
    struct BakeInterpolant {
      uint32_t slot = 0;
      Vector4 value = Vector4(0.0f, 0.0f, 0.0f, 0.0f);
      std::array<uint8_t, 4> lanes = { BakeLane_Value, BakeLane_Value, BakeLane_Value, BakeLane_Value };
    };
    static_assert(sizeof(BakeInterpolant) == 24, "BakeInterpolant is hashed by memory.");

    Rc<DxvkShader> buildBakeVertexShader(const std::vector<BakeInterpolant>& interpolants, const Vector2& period) {
      SpirvModule module(spvVersion(1, 3));
      module.setMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450);
      module.enableCapability(spv::CapabilityShader);

      const uint32_t floatType = module.defFloatType(32);
      const uint32_t uintType = module.defIntType(32, 0);
      const uint32_t intType = module.defIntType(32, 1);
      const uint32_t vec4Type = module.defVectorType(floatType, 4);
      const uint32_t voidType = module.defVoidType();
      const uint32_t entryPointId = module.allocateId();
      std::vector<uint32_t> interfaces;

      const uint32_t vertexIndexPtr =
        module.newVar(module.defPointerType(intType, spv::StorageClassInput), spv::StorageClassInput);
      module.decorateBuiltIn(vertexIndexPtr, spv::BuiltInVertexIndex);
      module.setDebugName(vertexIndexPtr, "vertex_index");
      interfaces.push_back(vertexIndexPtr);

      const uint32_t positionPtr =
        module.newVar(module.defPointerType(vec4Type, spv::StorageClassOutput), spv::StorageClassOutput);
      module.decorateBuiltIn(positionPtr, spv::BuiltInPosition);
      module.setDebugName(positionPtr, "position");
      interfaces.push_back(positionPtr);

      DxvkInterfaceSlots interfaceSlots;
      std::vector<uint32_t> outputPtrs;
      for (const BakeInterpolant& interpolant : interpolants) {
        const uint32_t ptr = module.newVar(module.defPointerType(vec4Type, spv::StorageClassOutput), spv::StorageClassOutput);
        module.decorateLocation(ptr, interpolant.slot);
        module.setDebugName(ptr, str::format("out_", interpolant.slot).c_str());
        interfaces.push_back(ptr);
        outputPtrs.push_back(ptr);
        interfaceSlots.outputSlots |= 1u << interpolant.slot;
      }

      module.functionBegin(voidType, entryPointId, module.defFunctionType(voidType, 0, nullptr), spv::FunctionControlMaskNone);
      module.setDebugName(entryPointId, "main");
      module.opLabel(module.allocateId());

      // corner = (packed >> 2 * vertexIndex) & 3, x = corner & 1, y = corner >> 1
      const uint32_t vertexIndex = module.opBitcast(uintType, module.opLoad(intType, vertexIndexPtr));
      const uint32_t shift = module.opIMul(uintType, vertexIndex, module.constu32(2));
      const uint32_t corner = module.opBitwiseAnd(uintType,
        module.opShiftRightLogical(uintType, module.constu32(packCornerIds()), shift), module.constu32(3));
      const uint32_t cornerX = module.opConvertUtoF(floatType, module.opBitwiseAnd(uintType, corner, module.constu32(1)));
      const uint32_t cornerY = module.opConvertUtoF(floatType, module.opShiftRightLogical(uintType, corner, module.constu32(1)));

      // The bake's row 0 is v = 0, as UE3 and Remix both address textures.
      const std::array<uint32_t, 4> position = {
        module.opFSub(floatType, module.opFMul(floatType, cornerX, module.constf32(2.0f)), module.constf32(1.0f)),
        module.opFSub(floatType, module.opFMul(floatType, cornerY, module.constf32(2.0f)), module.constf32(1.0f)),
        module.constf32(0.0f),
        module.constf32(1.0f),
      };
      module.opStore(positionPtr, module.opCompositeConstruct(vec4Type, uint32_t(position.size()), position.data()));

      const uint32_t u = module.opFMul(floatType, cornerX, module.constf32(period.x));
      const uint32_t v = module.opFMul(floatType, cornerY, module.constf32(period.y));
      const uint32_t averagedU = module.opFMul(floatType, u, module.constf32(kUe3CompositeAveragedRampScale));
      const uint32_t averagedV = module.opFMul(floatType, v, module.constf32(kUe3CompositeAveragedRampScale));
      for (size_t i = 0; i < interpolants.size(); i++) {
        const BakeInterpolant& interpolant = interpolants[i];
        std::array<uint32_t, 4> lanes;
        for (uint32_t lane = 0; lane < 4; lane++) {
          switch (interpolant.lanes[lane]) {
          case BakeLane_RampU: lanes[lane] = u; break;
          case BakeLane_RampV: lanes[lane] = v; break;
          case BakeLane_AveragedU: lanes[lane] = averagedU; break;
          case BakeLane_AveragedV: lanes[lane] = averagedV; break;
          default: lanes[lane] = module.constf32(interpolant.value[lane]); break;
          }
        }
        module.opStore(outputPtrs[i], module.opCompositeConstruct(vec4Type, uint32_t(lanes.size()), lanes.data()));
      }

      module.opReturn();
      module.functionEnd();
      module.addEntryPoint(entryPointId, spv::ExecutionModelVertex, "main", uint32_t(interfaces.size()), interfaces.data());

      const DxvkShaderOptions shaderOptions = { };
      return new DxvkShader(VK_SHADER_STAGE_VERTEX_BIT, 0, nullptr, interfaceSlots, module.compile(),
                            shaderOptions, DxvkShaderConstData());
    }

    // Returns whether the bake knows how to stand in for the vertex factory's base pass interpolants.
    bool isUe3CompositeVertexFactory(const Ue3VertexFactoryType type) {
      switch (type) {
      case Ue3VertexFactoryType::Local:
      case Ue3VertexFactoryType::GPUSkin:
      case Ue3VertexFactoryType::GPUSkinMorph:
      case Ue3VertexFactoryType::Foliage:
      case Ue3VertexFactoryType::ParticleInstancedMesh:
        return true;
      default:
        return false;
      }
    }

    struct TilingText {
      float u;
      float v;
    };

    std::ostream& operator<<(std::ostream& stream, const TilingText& tiling) {
      stream << "x" << tiling.u;
      if (tiling.u != tiling.v) {
        stream << "," << tiling.v;
      }
      return stream;
    }

    // `inputMask` has the bit layout of DxsoColorTermResult::interpolantFactorInputs, where a ps_2_x t#
    // register's number is its TEXCOORD index.
    bool readsVertexLightmap(const uint32_t inputMask, const DxsoIsgnEntry& element) {
      if (element.regNumber < 16 && ((inputMask >> element.regNumber) & 1u) != 0) {
        return true;
      }
      return element.semantic.usage == DxsoUsage::Texcoord && element.semantic.usageIndex < 16 &&
             ((inputMask >> (16 + element.semantic.usageIndex)) & 1u) != 0;
    }

    const char* describeBlendFactor(const DWORD factor) {
      switch (factor) {
      case D3DBLEND_ZERO: return "ZERO";
      case D3DBLEND_ONE: return "ONE";
      case D3DBLEND_SRCCOLOR: return "SRCCOLOR";
      case D3DBLEND_INVSRCCOLOR: return "INVSRCCOLOR";
      case D3DBLEND_SRCALPHA: return "SRCALPHA";
      case D3DBLEND_INVSRCALPHA: return "INVSRCALPHA";
      case D3DBLEND_DESTALPHA: return "DESTALPHA";
      case D3DBLEND_INVDESTALPHA: return "INVDESTALPHA";
      case D3DBLEND_DESTCOLOR: return "DESTCOLOR";
      case D3DBLEND_INVDESTCOLOR: return "INVDESTCOLOR";
      case D3DBLEND_SRCALPHASAT: return "SRCALPHASAT";
      case D3DBLEND_BOTHSRCALPHA: return "BOTHSRCALPHA";
      case D3DBLEND_BOTHINVSRCALPHA: return "BOTHINVSRCALPHA";
      case D3DBLEND_BLENDFACTOR: return "BLENDFACTOR";
      case D3DBLEND_INVBLENDFACTOR: return "INVBLENDFACTOR";
      default: return "?";
      }
    }

    bool blendFactorReadsAlpha(const DWORD factor) {
      switch (factor) {
      case D3DBLEND_SRCALPHA:
      case D3DBLEND_INVSRCALPHA:
      case D3DBLEND_DESTALPHA:
      case D3DBLEND_INVDESTALPHA:
      case D3DBLEND_SRCALPHASAT:
      case D3DBLEND_BOTHSRCALPHA:
      case D3DBLEND_BOTHINVSRCALPHA:
        return true;
      default:
        return false;
      }
    }

    std::string formatBakeSize(const Ue3CompositeDomain& domain) {
      std::string size = str::format(domain.width, "x", domain.height);
      if (domain.width < domain.neededWidth || domain.height < domain.neededHeight) {
        size += str::format(" (capped from ", domain.neededWidth, "x", domain.neededHeight, ")");
      }
      return size;
    }

  }

  const DxsoUe3CompositeShaderInfo& D3D9Rtx::getOrAnalyzeUe3CompositeShader(const XXH64_hash_t psHash, const D3D9CommonShader* pixelShader) {
    auto it = m_ue3CompositeShaders.find(psHash);
    if (it == m_ue3CompositeShaders.end()) {
      it = m_ue3CompositeShaders.emplace(
        psHash, analyzeDxsoUe3CompositeShader(makeDxsoShaderView(pixelShader->GetBytecode(), pixelShader))).first;
    }
    return it->second;
  }

  const char* D3D9Rtx::solveUe3CompositeDomain(const Ue3TextureState& ue3, const DxsoUe3CompositeShaderInfo& info,
                                               const uint32_t texcoordIdx, const std::vector<Vector2>& materialLightingScales,
                                               const bool describe, Ue3CompositeDomain& domain, std::string& refusal) const {
    auto refuse = [&](const char* reason, const auto&... details) -> const char* {
      if (describe) {
        if constexpr (sizeof...(details) == 0) {
          refusal = reason;
        } else {
          refusal = str::format(details...);
        }
      }
      return reason;
    };

    if (info.refusal != DxsoUe3CompositeRefusal::None) {
      const char* reason = describeDxsoUe3CompositeRefusal(info.refusal);
      return info.refusalDetail.empty() ? refuse(reason) : refuse(reason, reason, " (", info.refusalDetail, ")");
    }
    // Remix derives a blend by colour alone, such as a modulated or additive one, from the surface colour,
    // which the bake supplies. A blend by alpha would need the shader's alpha, which the bake keeps only as coverage.
    const bool blended = d3d9State().renderStates[D3DRS_ALPHABLENDENABLE];
    const DWORD srcBlend = d3d9State().renderStates[D3DRS_SRCBLEND];
    const DWORD dstBlend = d3d9State().renderStates[D3DRS_DESTBLEND];
    if (blended && (blendFactorReadsAlpha(srcBlend) || blendFactorReadsAlpha(dstBlend))) {
      return refuse("alpha blend", "the draw blends by alpha (", describeBlendFactor(srcBlend), ", ",
                    describeBlendFactor(dstBlend), "), and the bake keeps its alpha only as coverage");
    }
    // Decals are not among them: a decal category only changes how the surface layers over the one
    // beneath, and a decal the game blends by alpha is refused above.
    if (m_activeDrawCallState.testCategoryFlags(InstanceCategories::Sky, InstanceCategories::Particle, InstanceCategories::Beam,
                                                InstanceCategories::Terrain, InstanceCategories::WorldUI,
                                                InstanceCategories::WorldMatte, InstanceCategories::AnimatedWater)) {
      return refuse("the draw is tagged as sky, particle, beam, terrain, world UI or water");
    }
    if (!isUe3CompositeVertexFactory(ue3.vfType)) {
      return refuse("vertex factory", "the ", describeUe3VertexFactory(ue3.vfType),
                    " vertex factory's interpolants are not ones the bake stands in for");
    }
    if (m_uvResolutionMode == UvResolutionMode::LegacyTss) {
      return refuse("the surface's UV set is not proven from the shaders");
    }

    // Records each layer's bound texture and sampler state, which together decide whether a bake is current.
    struct TextureRecord {
      uint32_t stage;
      uint32_t width;
      uint32_t height;
      uint32_t mipLevels;
      XXH64_hash_t imageHash;
      XXH64_hash_t samplerHash;
      uint32_t srgb;
      uint32_t padding;
    };
    static_assert(sizeof(TextureRecord) == 40, "TextureRecord is hashed by memory.");
    std::vector<TextureRecord> textureRecords;

    const PsSamplerTexcoordEntry& entry = *ue3.inferredPsEntry;
    std::vector<float> scalesU;
    std::vector<float> scalesV;
    std::vector<float> texelsU;
    std::vector<float> texelsV;
    std::vector<Vector2> colorScales;
    bool emissionLeftOut = false;
    std::string description;
    auto describeLayer = [&](const auto&... parts) {
      if (describe) {
        description += str::format(description.empty() ? "" : ", ", parts...);
      }
    };

    // Holds back the first mask layer on the surface's UV set until it is known whether leaving it out shortens the repeat.
    struct MaskCandidate {
      uint8_t stage;
      uint8_t channel;
      uint8_t degree;
      float scaleU;
      float scaleV;
      float offsetU;
      float offsetV;
      float texelsU;
      float texelsV;
      size_t descriptionEnd;
    };
    std::optional<MaskCandidate> maskCandidate;

    for (const DxsoUe3CompositeLayer& layer : info.layers) {
      const uint32_t stage = layer.sampler;
      D3D9CommonTexture* texture =
        (stage < SamplerCount && d3d9State().textures[stage] != nullptr) ? GetCommonTexture(d3d9State().textures[stage]) : nullptr;
      if (texture == nullptr || texture->GetImage() == nullptr) {
        return refuse("unbound layer", "layer s", stage, " (", layer.name, ") has no texture bound");
      }
      const D3D9_COMMON_TEXTURE_DESC* desc = texture->Desc();

      TextureRecord record = {};
      record.stage = stage;
      record.width = desc->Width;
      record.height = desc->Height;
      record.mipLevels = desc->MipLevels;
      record.imageHash = texture->GetImage()->getHash();
      record.samplerHash = D3D9SamplerKeyHash {}(m_parent->CreateSamplerKey(stage));
      record.srgb = d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1u;
      textureRecords.push_back(record);

      const char* roles =
        layer.contributesColor()
          ? (layer.emissive && layer.diffuse ? "diffuse+emissive" : layer.emissive ? "emissive" : "diffuse")
          : layer.opacity ? "opacity" : "coordinate";

      if (layer.isCube) {
        // A cube map is looked up by a view or reflection vector, which is constant over the bake.
        if (entry.samplerUvOrigin[stage].originValid) {
          return refuse("cube layer coordinate", "cube layer s", stage, " (", layer.name, ") reads a texture coordinate");
        }
        describeLayer("s", stage, " ", layer.name, " cube ", roles, " (face-on)");
        continue;
      }

      const PsSamplerUvOrigin& origin = entry.samplerUvOrigin[stage];
      const bool readsSurfaceUvSet =
        origin.originValid && origin.sitesAgree && origin.semanticIndex == texcoordIdx &&
        origin.compU == m_texcoordCompU && origin.compV == m_texcoordCompV;
      const bool sharesSurfaceLanes =
        origin.semanticIndex == texcoordIdx &&
        (origin.compU == m_texcoordCompU || origin.compU == m_texcoordCompV ||
         origin.compV == m_texcoordCompU || origin.compV == m_texcoordCompV);
      const bool emits = layer.emissive && info.lighting != DxsoUe3CompositeLighting::Unlit;
      if (!readsSurfaceUvSet && origin.originValid && origin.sitesAgree && !sharesSurfaceLanes &&
          layer.contributesColor() && !layer.opacity) {
        // A layer on the mesh's other UV channel has no one place in a repeat of this one, so the bake
        // samples it at its average, through a ramp too steep for its mips. Its emission is left out
        // rather than glowing evenly. It is the same in both passes, so it cancels out of the albedo.
        // The ramp only averages a texture, so colour math on the channel's coordinate itself is refused.
        const uint8_t averagedLanes = uint8_t((1u << (origin.compU & 3u)) | (1u << (origin.compV & 3u)));
        if (origin.semanticIndex < info.colorTexcoordLanes.size() &&
            (info.colorTexcoordLanes[origin.semanticIndex] & averagedLanes)) {
          return refuse("averaged UV set in the colour", "the colour computes with TEXCOORD", uint32_t(origin.semanticIndex),
                        " itself as well as through layer s", stage, " (", layer.name,
                        "), and the bake can only average that UV set");
        }
        domain.averagedLanes.push_back({ origin.semanticIndex, origin.compU, origin.compV });
        emissionLeftOut |= emits;
        describeLayer("s", stage, " ", layer.name, " ", desc->Width, "x", desc->Height, " ", roles, " on another UV set (",
                      layer.diffuse ? (emits ? "averaged, emission left out" : "averaged") : "emission left out", ")");
        continue;
      }
      if (!readsSurfaceUvSet && !origin.originValid && emits && !layer.diffuse && !layer.opacity) {
        // An emitting layer read from a coordinate the bake cannot place is the same in both passes, so it
        // cancels out of the albedo, and its emission is left out.
        emissionLeftOut = true;
        describeLayer("s", stage, " ", layer.name, " ", desc->Width, "x", desc->Height, " ", roles,
                      " without a provable UV origin (emission left out)");
        continue;
      }
      if (!origin.originValid || !origin.sitesAgree) {
        return refuse("layer UV origin", "layer s", stage, " (", layer.name, ") has no single provable UV origin");
      }
      if (origin.semanticIndex != texcoordIdx || origin.compU != m_texcoordCompU || origin.compV != m_texcoordCompV) {
        return refuse("layer UV set", "layer s", stage, " (", layer.name, ") reads TEXCOORD", uint32_t(origin.semanticIndex),
                      " (", uint32_t(origin.compU), ",", uint32_t(origin.compV), ") but the surface reads TEXCOORD",
                      texcoordIdx, " (", uint32_t(m_texcoordCompU), ",", uint32_t(m_texcoordCompV), ")");
      }
      if (!origin.affineExact || origin.affineU.hasCross || origin.affineV.hasCross) {
        return refuse("layer tiling", "layer s", stage, " (", layer.name,
                      ") is not an exact axis-aligned tiling of the surface's UV set");
      }
      float scaleU = 1.0f;
      float scaleV = 1.0f;
      float offsetU = 0.0f;
      float offsetV = 0.0f;
      if (!resolveUe3PsAffineTerm(origin.affineU.scale, 1.0f, scaleU) ||
          !resolveUe3PsAffineTerm(origin.affineU.offset, 0.0f, offsetU) ||
          !resolveUe3PsAffineTerm(origin.affineV.scale, 1.0f, scaleV) ||
          !resolveUe3PsAffineTerm(origin.affineV.offset, 0.0f, offsetV)) {
        return refuse("layer tiling constants", "layer s", stage, " (", layer.name,
                      ")'s tiling does not resolve against the draw's constants");
      }

      // The surface reads a lifted mask through a wrapping sampler, so only a mask the game wraps is lifted.
      // A blended draw keeps its masks in the bake, since its opacity follows the albedo and opacity
      // micromaps read only the baked one.
      const auto mask = std::find_if(info.maskLayers.begin(), info.maskLayers.end(),
                                     [&](const DxsoUe3CompositeMaskLayer& m) { return m.sampler == stage; });
      const bool maskWraps = d3d9State().samplerStates[stage][D3DSAMP_ADDRESSU] == D3DTADDRESS_WRAP &&
                             d3d9State().samplerStates[stage][D3DSAMP_ADDRESSV] == D3DTADDRESS_WRAP;
      if (mask != info.maskLayers.end() && maskWraps && !blended && !maskCandidate.has_value() &&
          getRemixSampleView(texture, false) != nullptr) {
        colorScales.push_back(Vector2(scaleU, scaleV));
        describeLayer("s", stage, " ", layer.name, " ", desc->Width, "x", desc->Height, " ", TilingText { scaleU, scaleV }, " ", roles);
        maskCandidate = MaskCandidate { uint8_t(stage), mask->channel, mask->degree, scaleU, scaleV, offsetU, offsetV,
                                        std::abs(scaleU) * float(desc->Width), std::abs(scaleV) * float(desc->Height),
                                        description.size() };
        continue;
      }

      scalesU.push_back(std::abs(scaleU));
      scalesV.push_back(std::abs(scaleV));
      // Every layer's detail shows in the bake. A coordinate layer's shows through what it steers, as a normal
      // map's tiles show through the reflection it bends.
      texelsU.push_back(std::abs(scaleU) * float(desc->Width));
      texelsV.push_back(std::abs(scaleV) * float(desc->Height));
      if (layer.contributesColor()) {
        colorScales.push_back(Vector2(scaleU, scaleV));
      }
      describeLayer("s", stage, " ", layer.name, " ", desc->Width, "x", desc->Height, " ", TilingText { scaleU, scaleV }, " ", roles);
    }

    // A lighting-only layer never refuses a bake. One that is not a plain tiling of the surface's UV set is left out.
    for (const DxsoUe3CompositeLightingLayer& layer : info.lightingLayers) {
      const uint32_t stage = layer.sampler;
      D3D9CommonTexture* texture = stage < SamplerCount && d3d9State().textures[stage] != nullptr
        ? GetCommonTexture(d3d9State().textures[stage])
        : nullptr;
      if (texture == nullptr) {
        continue;
      }
      const PsSamplerUvOrigin& origin = entry.samplerUvOrigin[stage];
      float scaleU = 1.0f;
      float scaleV = 1.0f;
      const bool tilesSurfaceUvSet =
        origin.originValid && origin.sitesAgree && origin.semanticIndex == texcoordIdx &&
        origin.compU == m_texcoordCompU && origin.compV == m_texcoordCompV &&
        origin.affineExact && !origin.affineU.hasCross && !origin.affineV.hasCross &&
        resolveUe3PsAffineTerm(origin.affineU.scale, 1.0f, scaleU) && resolveUe3PsAffineTerm(origin.affineV.scale, 1.0f, scaleV);
      if (!tilesSurfaceUvSet) {
        describeLayer("s", stage, " ", layer.name, " lighting (not a tiling of the surface's UV set)");
        continue;
      }
      domain.lightingScales.push_back(Vector2(std::abs(scaleU), std::abs(scaleV)));
      describeLayer("s", stage, " ", layer.name, " ", texture->Desc()->Width, "x", texture->Desc()->Height, " ",
                    TilingText { scaleU, scaleV }, " lighting");
    }

    float periodU = scalesU.empty() ? 0.0f : solveUe3CompositePeriod(scalesU.data(), uint32_t(scalesU.size()));
    float periodV = scalesV.empty() ? 0.0f : solveUe3CompositePeriod(scalesV.data(), uint32_t(scalesV.size()));
    bool liftMask = false;
    if (maskCandidate.has_value()) {
      std::vector<float> withMaskU = scalesU;
      std::vector<float> withMaskV = scalesV;
      withMaskU.push_back(std::abs(maskCandidate->scaleU));
      withMaskV.push_back(std::abs(maskCandidate->scaleV));
      const float fullU = solveUe3CompositePeriod(withMaskU.data(), uint32_t(withMaskU.size()));
      const float fullV = solveUe3CompositePeriod(withMaskV.data(), uint32_t(withMaskV.size()));
      liftMask = periodU > 0.0f && periodV > 0.0f && (fullU <= 0.0f || fullV <= 0.0f || periodU < fullU || periodV < fullV);
      if (liftMask) {
        if (describe) {
          description.insert(maskCandidate->descriptionEnd,
                             str::format(maskCandidate->degree == 2 ? " (quadratic mask ." : " (mask .",
                                         "xyzw"[maskCandidate->channel & 3u], ", blended at render time)"));
        }
      } else {
        scalesU = std::move(withMaskU);
        scalesV = std::move(withMaskV);
        texelsU.push_back(maskCandidate->texelsU);
        texelsV.push_back(maskCandidate->texelsV);
        periodU = fullU;
        periodV = fullV;
      }
    }

    if (scalesU.empty()) {
      return refuse("no layer reads the surface's UV set");
    }
    if (periodU <= 0.0f || periodV <= 0.0f) {
      return refuse("no common repeat", "the layer tilings do not repeat together within ", kUe3CompositeMaxPeriodMultiple,
                    " repeats of the coarsest [", description, "]");
    }

    const uint32_t maxResolution = std::max(m_frameOptions.ue3CompositeMaxResolution, kUe3CompositeMinResolution);
    // Returns the resolution the layers need along one axis to cover `period`, before the cap.
    auto neededResolution = [](std::vector<float> texelsPerUv, const float period) {
      constexpr uint32_t kUncapped = 1u << 15;
      for (float& texels : texelsPerUv) {
        texels *= period;
      }
      return solveUe3CompositeResolution(texelsPerUv.data(), uint32_t(texelsPerUv.size()), kUe3CompositeMinResolution, kUncapped);
    };

    // The repeat also covers the material's lighting-only layers where they share one with the others and the bake
    // still fits under the resolution cap, so a normal map authored over the bake lines up with the game's.
    // Covering them never costs the baked layers any detail.
    std::vector<Vector2> lightingScales = materialLightingScales;
    for (const Vector2& scale : domain.lightingScales) {
      if (std::find(lightingScales.begin(), lightingScales.end(), scale) == lightingScales.end()) {
        lightingScales.push_back(scale);
      }
    }
    if (!lightingScales.empty()) {
      std::vector<float> coveringU = scalesU;
      std::vector<float> coveringV = scalesV;
      for (const Vector2& scale : lightingScales) {
        coveringU.push_back(scale.x);
        coveringV.push_back(scale.y);
      }
      const float coveredU = solveUe3CompositePeriod(coveringU.data(), uint32_t(coveringU.size()));
      const float coveredV = solveUe3CompositePeriod(coveringV.data(), uint32_t(coveringV.size()));
      const bool fits = coveredU > 0.0f && coveredV > 0.0f &&
                        neededResolution(texelsU, coveredU) <= maxResolution &&
                        neededResolution(texelsV, coveredV) <= maxResolution;
      if (fits) {
        periodU = coveredU;
        periodV = coveredV;
      } else if (describe) {
        description += ", lighting layers left out of the repeat";
      }
    }

    if (liftMask) {
      domain.mask = Ue3CompositeDomain::LiftedMask {
        maskCandidate->stage, maskCandidate->channel, maskCandidate->degree,
        Vector4(maskCandidate->scaleU * periodU, maskCandidate->scaleV * periodV, maskCandidate->offsetU, maskCandidate->offsetV) };
    }

    domain.neededWidth = neededResolution(std::move(texelsU), periodU);
    domain.neededHeight = neededResolution(std::move(texelsV), periodV);
    domain.width = std::min(domain.neededWidth, maxResolution);
    domain.height = std::min(domain.neededHeight, maxResolution);
    domain.period = Vector2(periodU, periodV);

    // The surface's texcoords are the IA set, through the vertex shader's affine when it has one. The bake
    // covers [0, period] of the interpolant the pixel shader reads.
    float scaleTu = 1.0f / periodU;
    float scaleTv = 1.0f / periodV;
    float offsetTu = 0.0f;
    float offsetTv = 0.0f;
    if (m_uvResolutionMode == UvResolutionMode::ProvenIa && m_ue3UvFacts.vsAffineFold) {
      scaleTu = m_ue3UvFacts.vsScaleU / periodU;
      scaleTv = m_ue3UvFacts.vsScaleV / periodV;
      offsetTu = m_ue3UvFacts.vsOffsetU / periodU;
      offsetTv = m_ue3UvFacts.vsOffsetV / periodV;
    }
    domain.transform = Matrix4();
    domain.transform[0].x = scaleTu;
    domain.transform[1].y = scaleTv;
    domain.transform[3].x = offsetTu;
    domain.transform[3].y = offsetTv;

    domain.layered = std::any_of(colorScales.begin(), colorScales.end(), [&](const Vector2& s) {
      return std::abs(s.x - colorScales.front().x) > 1e-4f || std::abs(s.y - colorScales.front().y) > 1e-4f;
    });
    domain.storeEmission = m_frameOptions.ue3CompositeEmission && info.hasEmissive &&
                           info.lighting != DxsoUe3CompositeLighting::Unlit && !emissionLeftOut;
    domain.texturesHash = textureRecords.empty()
      ? kEmptyHash
      : XXH3_64bits(textureRecords.data(), textureRecords.size() * sizeof(TextureRecord));
    domain.description = std::move(description);
    return nullptr;
  }

  Rc<DxvkShader> D3D9Rtx::getOrBuildUe3CompositeVertexShader(const D3D9CommonShader* pixelShader, const DxsoUe3CompositeShaderInfo& info,
                                                              const Ue3CompositeDomain& domain, const uint32_t texcoordIdx,
                                                              const float vertexLightmap) {
    // The bake's vertex shader writes every input the pixel shader declares, at the linker slot it reads it from.
    // UE3's base pass interpolants are stood in for as a surface seen face-on. TEXCOORD5 is PixelPosition,
    // TEXCOORD6 CameraVector_FogA and TEXCOORD7 SkyVector, all in tangent space (BasePassPixelShader.usf).
    // COLOR0/COLOR1 are the vertex colour or, under WORLD_COORDS, the [0,1]-encoded tangent basis.
    const DxsoIsgn& isgn = pixelShader->GetIsgn();
    std::vector<BakeInterpolant> interpolants;
    for (uint32_t i = 0; i < isgn.elemCount; i++) {
      const DxsoIsgnEntry& element = isgn.elems[i];
      BakeInterpolant interpolant;
      interpolant.slot = element.slot;
      if (element.semantic.usage == DxsoUsage::Texcoord) {
        if (element.semantic.usageIndex == 6) {
          interpolant.value = Vector4(0.0f, 0.0f, 1.0f, 1.0f);
        } else if (element.semantic.usageIndex == 7) {
          interpolant.value = Vector4(0.0f, 0.0f, 1.0f, 0.0f);
        }
        for (const Ue3CompositeDomain::AveragedLanes& averaged : domain.averagedLanes) {
          if (averaged.semanticIndex == element.semantic.usageIndex) {
            interpolant.lanes[averaged.compU & 3u] = BakeLane_AveragedU;
            interpolant.lanes[averaged.compV & 3u] = BakeLane_AveragedV;
          }
        }
        if (element.semantic.usageIndex == texcoordIdx) {
          interpolant.lanes[m_texcoordCompU & 3u] = BakeLane_RampU;
          interpolant.lanes[m_texcoordCompV & 3u] = BakeLane_RampV;
        }
      } else if (element.semantic.usage == DxsoUsage::Color) {
        interpolant.value = !info.worldCoords ? Vector4(1.0f, 1.0f, 1.0f, 1.0f)
                          : element.semantic.usageIndex == 0 ? Vector4(0.5f, 0.5f, 1.0f, 1.0f)
                          : Vector4(1.0f, 0.5f, 0.5f, 1.0f);
      }
      if (readsVertexLightmap(info.vertexLightmapInputMask, element)) {
        interpolant.value = Vector4(vertexLightmap, vertexLightmap, vertexLightmap, vertexLightmap);
      }
      // The alpha output reads this register's w, which is 1 wherever the shader did not clip.
      if (int32_t(element.regNumber) == info.alphaInputRegister) {
        interpolant.value.w = 1.0f;
      }
      if (interpolant.slot < 32) {
        interpolants.push_back(interpolant);
      }
    }

    XXH64_hash_t key = XXH3_64bits(&domain.period, sizeof(domain.period));
    if (!interpolants.empty()) {
      key = XXH3_64bits_withSeed(interpolants.data(), interpolants.size() * sizeof(BakeInterpolant), key);
    }
    Rc<DxvkShader>& shader = m_ue3CompositeVertexShaders[key];
    if (shader == nullptr) {
      shader = buildBakeVertexShader(interpolants, domain.period);
    }
    return shader;
  }

  std::shared_ptr<const Ue3CompositeBakeRequest> D3D9Rtx::buildUe3CompositeRequest(const Ue3TextureState& ue3,
                                                                                   const DxsoUe3CompositeShaderInfo& info,
                                                                                   const Ue3CompositeDomain& domain,
                                                                                   const uint32_t texcoordIdx,
                                                                                   const XXH64_hash_t key) {
    const LegacyMaterialData& material = m_activeDrawCallState.materialData;
    auto request = std::make_shared<Ue3CompositeBakeRequest>();
    request->key = key;
    const uint64_t lineage[] = {
      material.getHash(), ue3.inferredPsHash, texcoordIdx, (uint64_t(m_texcoordCompU) << 8) | uint64_t(m_texcoordCompV),
    };
    request->lineage = XXH3_64bits(lineage, sizeof(lineage));
    request->materialHash = material.getHash();
    request->textureSetShaderHash = material.getTextureSetAndShaderHash();
    request->pickTextureHash = material.getColorTexture().getImageHash();
    request->width = domain.width;
    request->height = domain.height;
    request->period = domain.period;
    request->domainTransform = domain.transform;
    request->emissivePass = info.hasEmissive && info.lighting != DxsoUe3CompositeLighting::Unlit;
    request->storeEmission = domain.storeEmission;
    request->uniformEmission = request->emissivePass &&
      std::none_of(info.layers.begin(), info.layers.end(), [](const DxsoUe3CompositeLayer& layer) { return layer.emissive; });
    request->coverageFromAlpha = info.hasTexkill;

    auto setRegisters = [&](const int32_t reg, const uint32_t count, const Vector4& diffusePass, const Vector4& emissivePass) {
      if (reg < 0) {
        return;
      }
      for (uint32_t r = uint32_t(reg); r < uint32_t(reg) + count && r < caps::MaxFloatConstantsPS; r++) {
        request->constants.push_back({ r, diffusePass, emissivePass });
      }
    };
    const Vector4 zero(0.0f, 0.0f, 0.0f, 0.0f);
    switch (info.lighting) {
    case DxsoUe3CompositeLighting::AmbientConstant:
      // With the sky light off and the lightmap scaled to nothing, this compile outputs Emissive + DiffuseColor * AmbientColor.
      setRegisters(info.ambientColorAndSkyFactorRegister, 1, Vector4(1.0f, 1.0f, 1.0f, 0.0f), zero);
      setRegisters(info.lightMapScaleRegister, info.lightMapScaleCount, zero, zero);
      break;
    case DxsoUe3CompositeLighting::SimpleLightmap:
      // This compile outputs Emissive + pow(1, DiffusePower) * DiffuseColorNormalized through a white lightmap.
      setRegisters(info.ambientColorAndSkyFactorRegister, 1, zero, zero);
      setRegisters(info.lightMapScaleRegister, info.lightMapScaleCount, Vector4(1.0f, 1.0f, 1.0f, 1.0f), zero);
      for (uint32_t s = 0; s < 32; s++) {
        if ((info.lightmapTextureSamplerMask >> s) & 1u) {
          const auto shaderSampler = RemapStateSamplerShader(s);
          request->whiteTextureSlots.push_back(
            computeResourceSlotId(shaderSampler.first, DxsoBindingType::Image, uint32_t(shaderSampler.second)));
        }
      }
      break;
    case DxsoUe3CompositeLighting::SimpleVertexLightmap:
      // This compile outputs Emissive + Diffuse through the lighting interpolant, which the bake's vertex shaders set.
      setRegisters(info.ambientColorAndSkyFactorRegister, 1, zero, zero);
      setRegisters(info.lightMapScaleRegister, info.lightMapScaleCount, zero, zero);
      break;
    case DxsoUe3CompositeLighting::Unlit:
      break;
    }
    setRegisters(info.upperSkyColorRegister, 1, zero, zero);
    setRegisters(info.lowerSkyColorRegister, 1, zero, zero);

    if (domain.mask.has_value()) {
      const uint32_t stage = domain.mask->stage;
      Ue3CompositeBakeRequest::Mask mask;
      const auto shaderSampler = RemapStateSamplerShader(stage);
      mask.slot = computeResourceSlotId(shaderSampler.first, DxsoBindingType::Image, uint32_t(shaderSampler.second));
      mask.texture = TextureRef(getRemixSampleView(GetCommonTexture(d3d9State().textures[stage]), false));
      mask.isSrgb = (d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1u) != 0;
      mask.channel = domain.mask->channel;
      mask.degree = domain.mask->degree;
      mask.transform = domain.mask->transform;
      request->mask = std::move(mask);
    }

    request->psConstantBufferSlot =
      computeResourceSlotId(DxsoProgramType::PixelShader, DxsoBindingType::ConstantBuffer, DxsoConstantBuffers::PSConstantBuffer);
    request->psFloatConstantsOffset = uint32_t(offsetof(D3D9ShaderConstantsPS, fConsts));
    request->vertexShader = getOrBuildUe3CompositeVertexShader(ue3.inferredPs, info, domain, texcoordIdx, 1.0f);
    if (info.lighting == DxsoUe3CompositeLighting::SimpleVertexLightmap && request->emissivePass) {
      request->emissiveVertexShader = getOrBuildUe3CompositeVertexShader(ue3.inferredPs, info, domain, texcoordIdx, 0.0f);
    }
    request->log = m_frameOptions.ue3LogCompositeMaterials;
    request->description = domain.description;
    return request;
  }

  bool D3D9Rtx::isUe3CompositeTagged(const fast_unordered_set& tags, const Ue3TextureState& ue3) {
    const LegacyMaterialData& material = m_activeDrawCallState.materialData;
    const XXH64_hash_t textureSetShaderHash = material.getTextureSetAndShaderHash();
    const XXH64_hash_t pickHash = material.getColorTexture().getImageHash();
    if (lookupHash(tags, material.getHash()) ||
        (textureSetShaderHash != kEmptyHash && lookupHash(tags, textureSetShaderHash)) ||
        (pickHash != kEmptyHash && lookupHash(tags, pickHash))) {
      return true;
    }

    // Any of its layers counts too, because the bake renders all of them and a mask or emissive layer can
    // win the pick. The shader is only analysed once a tagged texture is bound, to tell whether it is a layer.
    auto isStageTagged = [&](const uint32_t stage) {
      D3D9CommonTexture* texture =
        d3d9State().textures[stage] != nullptr ? GetCommonTexture(d3d9State().textures[stage]) : nullptr;
      return texture != nullptr && texture->GetImage() != nullptr && lookupHash(tags, texture->GetImage()->getHash());
    };
    bool anyStageTagged = false;
    for (uint32_t stage = 0; stage < caps::MaxTexturesPS && !anyStageTagged; stage++) {
      anyStageTagged = isStageTagged(stage);
    }
    if (!anyStageTagged) {
      return false;
    }
    const DxsoUe3CompositeShaderInfo& info = getOrAnalyzeUe3CompositeShader(ue3.inferredPsHash, ue3.inferredPs);
    return std::any_of(info.layers.begin(), info.layers.end(), [&](const DxsoUe3CompositeLayer& layer) {
      return layer.sampler < caps::MaxTexturesPS && isStageTagged(layer.sampler);
    });
  }

  uint32_t D3D9Rtx::selectUe3CompositeUvStage(const Ue3TextureState& ue3, const uint32_t pickStage) {
    m_ue3CompositeTagged = false;
    const fast_unordered_set& tags = *m_frameOptions.ue3CompositeMaterialTextures;
    if (!m_frameOptions.ue3EngineMode || tags.empty() || ue3.inferredPs == nullptr || ue3.inferredPsEntry == nullptr ||
        m_activeDrawCallState.materialData.getHash() == kEmptyHash) {
      return pickStage;
    }
    m_ue3CompositeTagged = isUe3CompositeTagged(tags, ue3);
    if (!m_ue3CompositeTagged || pickStage >= caps::MaxTexturesPS) {
      return pickStage;
    }
    const DxsoUe3CompositeShaderInfo& info = getOrAnalyzeUe3CompositeShader(ue3.inferredPsHash, ue3.inferredPs);
    if (info.refusal != DxsoUe3CompositeRefusal::None) {
      return pickStage;
    }

    // The bake covers one repeat of the UV set its diffuse layers read (its colour layers', when none is
    // lit), so that is the set the surface has to read, whichever layer won the pick. A diffuse pick
    // keeps its own set, so where diffuse layers read several, the bake keeps the pick's detail and
    // averages the others.
    const PsSamplerTexcoordEntry& entry = *ue3.inferredPsEntry;
    const PsSamplerUvOrigin& pickOrigin = entry.samplerUvOrigin[pickStage];
    const bool pickIsDiffuse = std::any_of(info.layers.begin(), info.layers.end(), [&](const DxsoUe3CompositeLayer& layer) {
      return layer.sampler == pickStage && layer.diffuse && !layer.isCube;
    });
    if (pickIsDiffuse && pickOrigin.originValid && pickOrigin.sitesAgree) {
      return pickStage;
    }
    for (const bool diffuseOnly : { true, false }) {
      for (const DxsoUe3CompositeLayer& layer : info.layers) {
        if (layer.isCube || layer.sampler >= caps::MaxTexturesPS ||
            !(diffuseOnly ? layer.diffuse : layer.contributesColor())) {
          continue;
        }
        const PsSamplerUvOrigin& origin = entry.samplerUvOrigin[layer.sampler];
        if (!origin.originValid || !origin.sitesAgree) {
          continue;
        }
        const bool pickReadsSameSet = pickOrigin.originValid && pickOrigin.semanticIndex == origin.semanticIndex &&
                                      pickOrigin.compU == origin.compU && pickOrigin.compV == origin.compV;
        return pickReadsSameSet ? pickStage : uint32_t(layer.sampler);
      }
    }
    return pickStage;
  }

  bool D3D9Rtx::resolveUe3Composite(const Ue3TextureState& ue3, const uint32_t texcoordIdx) {
    const bool tagged = m_ue3CompositeTagged;
    const bool logging = m_frameOptions.ue3LogCompositeMaterials;
    if (!m_frameOptions.ue3EngineMode || (!tagged && !logging) ||
        ue3.inferredPs == nullptr || ue3.inferredPsEntry == nullptr) {
      return false;
    }

    const LegacyMaterialData& material = m_activeDrawCallState.materialData;
    const XXH64_hash_t materialHash = material.getHash();
    if (materialHash == kEmptyHash) {
      return false;
    }
    const XXH64_hash_t textureSetShaderHash = material.getTextureSetAndShaderHash();
    const XXH64_hash_t pickHash = material.getColorTexture().getImageHash();

    Ue3CompositeMaterialState& state = m_ue3CompositeMaterials[materialHash];
    if (!tagged) {
      if (state.candidateChecked) {
        return false;
      }
      state.candidateChecked = true;
    } else if (state.animated) {
      return false;
    }

    ScopedCpuProfileZoneN("UE3 composite material");
    const DxsoUe3CompositeShaderInfo& info = getOrAnalyzeUe3CompositeShader(ue3.inferredPsHash, ue3.inferredPs);
    Ue3CompositeDomain domain;
    std::string refusal;

    if (!tagged) {
      if (solveUe3CompositeDomain(ue3, info, texcoordIdx, state.lightingScales, true, domain, refusal) == nullptr && domain.layered) {
        Logger::info(str::format(
          "[RTX-Compatibility][UE3-Composite] Candidate materialHash=0x", std::hex, materialHash,
          " textureSetShader=0x", textureSetShaderHash, " pick=0x", pickHash, std::dec,
          " is built from layers at different tilings [", domain.description, "], repeating every (",
          domain.period.x, ",", domain.period.y, ") for a ", formatBakeSize(domain),
          " bake: list one of its hashes in rtx.ue3CompositeMaterialTextures to render the whole composite."));
      }
      return false;
    }

    // Only the log and a new request read the descriptions, so the draws of a known key solve without them.
    if (const char* reason = solveUe3CompositeDomain(ue3, info, texcoordIdx, state.lightingScales, false, domain, refusal)) {
      const XXH64_hash_t refusalKey = XXH3_64bits(reason, std::strlen(reason));
      if (state.loggedRefusal != refusalKey) {
        state.loggedRefusal = refusalKey;
        Ue3CompositeDomain described;
        solveUe3CompositeDomain(ue3, info, texcoordIdx, state.lightingScales, true, described, refusal);
        Logger::warn(str::format(
          "[RTX-Compatibility][UE3-Composite] materialHash=0x", std::hex, materialHash, " pick=0x", pickHash, std::dec,
          " is tagged in rtx.ue3CompositeMaterialTextures but cannot be baked: ", refusal,
          ". It renders its albedo pick."));
      }
      return false;
    }

    // A compile that samples a lighting layer the material has not drawn with before moves every compile's repeat,
    // which is not a conflict between two materials sharing one hash.
    bool newLightingScale = false;
    for (const Vector2& scale : domain.lightingScales) {
      if (std::find(state.lightingScales.begin(), state.lightingScales.end(), scale) == state.lightingScales.end()) {
        state.lightingScales.push_back(scale);
        newLightingScale = true;
      }
    }
    if (newLightingScale) {
      state.hasDomain = false;
    }

    // A bake is only current for the material constant values it ran with, so they are part of its key.
    XXH3_state_t* const hashState = getThreadLocalXxh3State();
    XXH64_hash_t constantsHash = kEmptyHash;
    if (hashState != nullptr && !info.materialConstantRegisters.empty()) {
      XXH3_64bits_reset(hashState);
      for (const uint32_t reg : info.materialConstantRegisters) {
        if (reg < caps::MaxFloatConstantsPS) {
          XXH3_64bits_update(hashState, &d3d9State().psConsts.fConsts[reg], sizeof(Vector4));
        }
      }
      constantsHash = XXH3_64bits_digest(hashState);
    }

    struct KeyRecord {
      uint32_t lighting;
      uint32_t width;
      uint32_t height;
      float periodU;
      float periodV;
      uint32_t texcoordIdx;
      uint32_t texcoordComps;
      uint32_t storeEmission;
      XXH64_hash_t materialHash;
      XXH64_hash_t psHash;
      XXH64_hash_t texturesHash;
      XXH64_hash_t constantsHash;
    };
    static_assert(sizeof(KeyRecord) == 64, "KeyRecord is hashed by memory.");
    KeyRecord record = {};
    record.lighting = uint32_t(info.lighting);
    record.width = domain.width;
    record.height = domain.height;
    record.periodU = domain.period.x;
    record.periodV = domain.period.y;
    record.texcoordIdx = texcoordIdx;
    record.texcoordComps = (uint32_t(m_texcoordCompU) << 8) | uint32_t(m_texcoordCompV);
    record.storeEmission = domain.storeEmission;
    record.materialHash = materialHash;
    record.psHash = ue3.inferredPsHash;
    record.texturesHash = domain.texturesHash;
    record.constantsHash = constantsHash;
    const XXH64_hash_t key = XXH3_64bits(&record, sizeof(record));

    if (state.constantSets.insert(constantsHash).second && state.constantSets.size() > kUe3CompositeMaxConstantSetsPerMaterial) {
      state.animated = true;
      Logger::warn(str::format(
        "[RTX-Compatibility][UE3-Composite] materialHash=0x", std::hex, materialHash, " pick=0x", pickHash, std::dec,
        " has been drawn with ", state.constantSets.size(), " different values of its material constants, so one of "
        "them is animating (a panner or a time-driven tint). It renders its albedo pick for the rest of the session."));
      return false;
    }

    if (!state.hasDomain) {
      state.hasDomain = true;
      state.period = domain.period;
      state.domainTransform = domain.transform;
    } else if (!state.domainConflictWarned &&
               (state.period.x != domain.period.x || state.period.y != domain.period.y ||
                state.domainTransform != domain.transform)) {
      state.domainConflictWarned = true;
      Logger::warn(str::format(
        "[RTX-Compatibility][UE3-Composite] materialHash=0x", std::hex, materialHash, std::dec,
        " is drawn with two UV repeats, (", state.period.x, ",", state.period.y, ") and (", domain.period.x, ",",
        domain.period.y, "): a replacement authored over one of them is misaligned on the other. Material identity "
        "leaves tiling parameters out, so these are materials that differ only by their tiling."));
    }

    if (m_ue3CompositeRequests.size() > kUe3CompositeMaxRequests) {
      m_ue3CompositeRequests.clear();
    }
    std::shared_ptr<const Ue3CompositeBakeRequest>& request = m_ue3CompositeRequests[key];
    if (request == nullptr) {
      Ue3CompositeDomain described;
      solveUe3CompositeDomain(ue3, info, texcoordIdx, state.lightingScales, true, described, refusal);
      request = buildUe3CompositeRequest(ue3, info, described, texcoordIdx, key);
      if (logging) {
        const double textures = 1.0 + (request->mask.has_value() ? double(request->mask->degree) : 0.0) +
                                (request->storeEmission && !request->uniformEmission ? 1.0 : 0.0);
        const double megabytes = double(domain.width) * double(domain.height) * 4.0 * (4.0 / 3.0) * textures / (1024.0 * 1024.0);
        std::string constants;
        for (const uint32_t reg : info.materialConstantRegisters) {
          if (reg < caps::MaxFloatConstantsPS) {
            const Vector4& value = d3d9State().psConsts.fConsts[reg];
            constants += str::format(constants.empty() ? "" : " ", "c", reg, "=(", value.x, ",", value.y, ",", value.z, ",", value.w, ")");
          }
        }
        Logger::info(str::format(
          "[RTX-Compatibility][UE3-Composite] materialHash=0x", std::hex, materialHash,
          " textureSetShader=0x", textureSetShaderHash, " pick=0x", pickHash, " key=0x", key, std::dec,
          " lighting=", describeDxsoUe3CompositeLighting(info.lighting),
          " emission=", !request->emissivePass ? "none" : request->uniformEmission ? "uniform" : "textured",
          request->emissivePass && !request->storeEmission ? " (taken out of the albedo only)" : "",
          " coverage=", request->coverageFromAlpha ? 1 : 0,
          d3d9State().renderStates[D3DRS_ALPHABLENDENABLE]
            ? str::format(" blend=(", describeBlendFactor(d3d9State().renderStates[D3DRS_SRCBLEND]), ", ",
                          describeBlendFactor(d3d9State().renderStates[D3DRS_DESTBLEND]), ")")
            : std::string(),
          info.viewDependent ? " view-dependent terms baked face-on" : "",
          " layers=[", described.description, "] constants=[", constants, "] repeat=(", domain.period.x, ",", domain.period.y, ")",
          " bake=", formatBakeSize(domain), " (~", megabytes, " MB)",
          " uvTransform=(", domain.transform[0].x, ",", domain.transform[1].y, ",",
          domain.transform[3].x, ",", domain.transform[3].y, ")"));
      }
    }

    m_activeDrawCallState.ue3CompositeRequest = request;
    m_activeDrawCallState.ue3CompositeDrawStateApplied = !Ue3CompositeBaker::isSettled(key);
    return true;
  }

}
