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
    // IEEE 754 half -> float (no denormal flush; curve data never relies on
    // half denormals, but decode them correctly anyway)
    float ue3HalfToFloat(const uint16_t half) {
      const uint32_t sign = (half & 0x8000u) << 16;
      uint32_t exponent = (half & 0x7C00u) >> 10;
      uint32_t mantissa = half & 0x03FFu;

      if (exponent == 0) {
        if (mantissa == 0) {
          const uint32_t bits = sign;
          float result;
          std::memcpy(&result, &bits, sizeof(result));
          return result;
        }
        // subnormal half: normalize
        while ((mantissa & 0x0400u) == 0) {
          mantissa <<= 1;
          exponent--;
        }
        exponent++;
        mantissa &= ~0x0400u;
      } else if (exponent == 0x1F) {
        const uint32_t bits = sign | 0x7F800000u | (mantissa << 13);
        float result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
      }

      const uint32_t bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
      float result;
      std::memcpy(&result, &bits, sizeof(result));
      return result;
    }

    // Session registry of UE3 lightmap textures recognised from pixel shader CTAB sampler
    // names. Append-only, and read from texture paths that run on other threads than the
    // draw-call setup that discovers them.
    std::shared_mutex g_autoLightmapTexturesMutex;

    fast_unordered_set g_autoLightmapTextures;

    // Fast negative for the common case of a title that never discovers any.
    std::atomic<bool> g_autoLightmapTexturesPopulated { false };

    constexpr auto kUe3GamePatchRequestRetryInterval = std::chrono::seconds(2);
  }

  VDeclSignature buildVDeclSignature(const D3D9VertexElements& elements) {
    VDeclSignature sig;
    sig.totalElements = uint8_t(std::min<size_t>(elements.size(), 255u));
    for (const auto& e : elements) {
      switch (e.Usage) {
      case D3DDECLUSAGE_POSITION:
      case D3DDECLUSAGE_POSITIONT:
        sig.hasPosition = true;
        sig.positionType = e.Type;
        sig.positionStream = e.Stream;
        break;
      case D3DDECLUSAGE_TANGENT:
        sig.hasTangent = true;
        sig.tangentType = e.Type;
        sig.tangentStream = e.Stream;
        break;
      case D3DDECLUSAGE_NORMAL:
        sig.hasNormal = true;
        sig.normalType = e.Type;
        sig.normalStream = e.Stream;
        break;
      case D3DDECLUSAGE_BINORMAL:
        sig.hasBinormal = true;
        sig.binormalType = e.Type;
        break;
      case D3DDECLUSAGE_BLENDWEIGHT:
        sig.hasBlendWeight = true;
        sig.blendWeightType = e.Type;
        break;
      case D3DDECLUSAGE_BLENDINDICES:
        sig.hasBlendIndices = true;
        break;
      case D3DDECLUSAGE_COLOR:
        if (e.UsageIndex == 0) sig.hasColor0 = true;
        if (e.UsageIndex == 1) sig.hasColor1 = true;
        break;
      case D3DDECLUSAGE_TEXCOORD:
        if (e.UsageIndex < 8) {
          sig.texcoordCount++;
          if (e.UsageIndex > sig.maxTexcoordIndex) {
            sig.maxTexcoordIndex = e.UsageIndex;
          }
          sig.texcoordTypes[e.UsageIndex] = e.Type;
        }
        if (e.UsageIndex == 6) {
          sig.hasTexcoord6 = true;
          sig.texcoord6Type = e.Type;
        }
        if (e.UsageIndex == 7) {
          sig.hasTexcoord7 = true;
          sig.texcoord7Type = e.Type;
        }
        break;
      default:
        break;
      }
    }
    return sig;
  }

  Ue3VertexFactoryType D3D9Rtx::classifyUe3VertexFactory(const D3D9VertexElements& elements) {
    if (elements.empty()) {
      return Ue3VertexFactoryType::Unknown;
    }

    const VDeclSignature sig = buildVDeclSignature(elements);

    if (!sig.hasPosition) {
      return Ue3VertexFactoryType::Unknown;
    }

    // terrain = POSITION(UBYTE4) + BLENDWEIGHT(FLOAT1) + TANGENT(SHORT2)
    // packed UBYTE4 position is unique to terrain
    if (sig.positionType == D3DDECLTYPE_UBYTE4 &&
        sig.hasBlendWeight && sig.blendWeightType == D3DDECLTYPE_FLOAT1 &&
        sig.hasTangent && sig.tangentType == D3DDECLTYPE_SHORT2) {
      if (sig.hasNormal) {
        return Ue3VertexFactoryType::TerrainMorph;
      }
      return Ue3VertexFactoryType::Terrain;
    }

    // particle = POSITION(FLOAT3) + NORMAL(FLOAT3) + TANGENT(FLOAT3) + TEXCOORD0(FLOAT2) + BLENDWEIGHT(FLOAT1) + TEXCOORD1(FLOAT4)
    // NORMAL is FLOAT3 (not UBYTE4), TANGENT is FLOAT3, plus BLENDWEIGHT(FLOAT1)
    if ((sig.positionType == D3DDECLTYPE_FLOAT3 || sig.positionType == D3DDECLTYPE_FLOAT4) &&
        !sig.hasNormal &&
        sig.hasTangent && sig.tangentType == D3DDECLTYPE_FLOAT4 &&
        sig.hasBlendWeight && sig.blendWeightType == D3DDECLTYPE_FLOAT1 &&
        sig.texcoordCount >= 4 &&
        !sig.hasBlendIndices) {
      return Ue3VertexFactoryType::LensFlare;
    }

    if ((sig.positionType == D3DDECLTYPE_FLOAT3 || sig.positionType == D3DDECLTYPE_FLOAT4) &&
        sig.hasNormal && sig.normalType == D3DDECLTYPE_FLOAT4 &&
        sig.hasTangent && sig.tangentType == D3DDECLTYPE_FLOAT3 &&
        sig.hasBlendWeight && sig.blendWeightType == D3DDECLTYPE_FLOAT1 &&
        sig.texcoordCount >= 2 &&
        !sig.hasBlendIndices) {
      return Ue3VertexFactoryType::ParticleBeamTrail;
    }

    if (sig.positionType == D3DDECLTYPE_FLOAT3 &&
        sig.hasNormal && sig.normalType == D3DDECLTYPE_FLOAT3 &&
        sig.hasTangent && sig.tangentType == D3DDECLTYPE_FLOAT3 &&
        sig.hasBlendWeight && sig.blendWeightType == D3DDECLTYPE_FLOAT1 &&
        !sig.hasBlendIndices) {
      return Ue3VertexFactoryType::Particle;
    }

    // SpeedTree variants carry an explicit binormal and wind info through BLENDINDICES,
    // but they do not have skinning blend weights.
    if (sig.positionType == D3DDECLTYPE_FLOAT3 &&
        sig.hasBinormal &&
        sig.hasBlendIndices &&
        !sig.hasBlendWeight &&
        sig.hasTangent &&
        sig.hasNormal) {
      return Ue3VertexFactoryType::SpeedTree;
    }

    // position only (depth prepass) = only POSITION, no tangent/normal/texcoord/color
    if (sig.positionType == D3DDECLTYPE_FLOAT3 &&
        !sig.hasTangent && !sig.hasNormal && sig.texcoordCount == 0 &&
        !sig.hasColor0 && !sig.hasColor1 &&
        !sig.hasBlendWeight && !sig.hasBlendIndices) {
      return Ue3VertexFactoryType::PositionOnly;
    }

    // GPUSkin variants - must have BLENDINDICES + BLENDWEIGHT (bone data)
    // normal/tangent are UBYTE4 (PackedNormal)
    if (sig.positionType == D3DDECLTYPE_FLOAT3 &&
        sig.hasBlendIndices &&
        sig.hasBlendWeight &&
        sig.hasTangent && sig.tangentType == D3DDECLTYPE_UBYTE4 &&
        sig.hasNormal && sig.normalType == D3DDECLTYPE_UBYTE4) {
      // morph variant adds TEXCOORD6(FLOAT3) for delta position + TEXCOORD7(UBYTE4) for delta normal
      if (sig.hasTexcoord6 && sig.texcoord6Type == D3DDECLTYPE_FLOAT3 &&
          sig.hasTexcoord7) {
        return Ue3VertexFactoryType::GPUSkinMorph;
      }
      return Ue3VertexFactoryType::GPUSkin;
    }

    // Instanced mesh particles have no NORMAL: the mesh's TangentZ arrives under BINORMAL (see "The two
    // factories are not reliably distinguishable" in UE3Compatibility.md).
    if (sig.positionType == D3DDECLTYPE_FLOAT3 &&
        sig.hasTangent && sig.tangentType == D3DDECLTYPE_UBYTE4 &&
        sig.hasBinormal && sig.binormalType == D3DDECLTYPE_UBYTE4 &&
        !sig.hasNormal &&
        !sig.hasBlendIndices && !sig.hasBlendWeight &&
        sig.texcoordCount >= 5 &&
        sig.texcoordTypes[1] == D3DDECLTYPE_FLOAT3 &&
        sig.texcoordTypes[2] == D3DDECLTYPE_FLOAT3 &&
        sig.texcoordTypes[3] == D3DDECLTYPE_FLOAT3 &&
        sig.texcoordTypes[4] == D3DDECLTYPE_FLOAT3) {
      return Ue3VertexFactoryType::ParticleInstancedMesh;
    }

    // local (static mesh) = POSITION(FLOAT3) + TANGENT(UBYTE4) + NORMAL(UBYTE4) + TEXCOORDs, no BLENDINDICES
    // foliage extends the local layout with instancing axes in TEXCOORD1..4.
    if (sig.positionType == D3DDECLTYPE_FLOAT3 &&
        sig.hasTangent &&
        sig.hasNormal &&
        !sig.hasBlendIndices &&
        sig.texcoordCount >= 5 &&
        sig.texcoordTypes[1] == D3DDECLTYPE_FLOAT3 &&
        sig.texcoordTypes[2] == D3DDECLTYPE_FLOAT3 &&
        sig.texcoordTypes[3] == D3DDECLTYPE_FLOAT3 &&
        sig.texcoordTypes[4] == D3DDECLTYPE_FLOAT3) {
      return Ue3VertexFactoryType::Foliage;
    }

    if (sig.positionType == D3DDECLTYPE_FLOAT3 &&
        sig.hasTangent && sig.tangentType == D3DDECLTYPE_UBYTE4 &&
        sig.hasNormal && sig.normalType == D3DDECLTYPE_UBYTE4 &&
        !sig.hasBlendIndices) {
      return Ue3VertexFactoryType::Local;
    }

    return Ue3VertexFactoryType::Unknown;
  }

  Ue3ShaderFeatureInfo D3D9Rtx::getUe3ShaderFeatureInfo(const D3D9CommonShader* shader) {
    Ue3ShaderFeatureInfo empty;
    empty.initialized = true;

    if (shader == nullptr) {
      return empty;
    }

    const auto& bytecode = shader->GetBytecode();
    const XXH64_hash_t shaderHash = shader->GetBytecodeHash();
    if (shaderHash == 0) {
      return empty;
    }

    auto it = m_ue3ShaderFeatureCache.find(shaderHash);
    if (it != m_ue3ShaderFeatureCache.end()) {
      return it->second;
    }

    Ue3ShaderFeatureInfo info;
    info.initialized = true;

    try {
      if (bytecode.size() < sizeof(uint32_t) || (bytecode.size() % sizeof(uint32_t)) != 0) {
        m_ue3ShaderFeatureCache.emplace(shaderHash, info);
        return info;
      }

      const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
      DxsoDecodeContext decoder(shader->GetInfo());
      DxsoCodeIter iter(tokens + 1);
      while (decoder.decodeInstruction(iter)) {
        if (decoder.getCtabInfo().m_size != 0) {
          break;
        }
      }

      const DxsoCtab& ctab = decoder.getCtabInfo();
      if (ctab.m_size == 0 || ctab.m_constantData.empty()) {
        m_ue3ShaderFeatureCache.emplace(shaderHash, info);
        return info;
      }

      auto markName = [&](const std::string& lowerName, const bool isSampler, const bool isFloat4, const uint32_t registerIndex) {
        if (isSampler) {
          const uint8_t semanticFlags = classifyPixelSamplerSemanticFlags(lowerName);
          info.hasMaterialSampler |= (semanticFlags & kPsSamplerSemanticMaterialTexture) != 0;
          info.hasEngineAuxSampler |= (semanticFlags & kPsSamplerSemanticEngineAuxiliary) != 0;
          info.hasVideoSampler |= (semanticFlags & kPsSamplerSemanticVideo) != 0;

          // TdToneMapping capture: record the exact sampler indices of the
          // baked colour curve LUT textures.
          if (registerIndex < 0xFF) {
            if (lowerName == "colorcurvesktexture") {
              info.colorCurvesKSamplerIndex = uint8_t(registerIndex);
            } else if (lowerName == "colorcurvesmtexture") {
              info.colorCurvesMSamplerIndex = uint8_t(registerIndex);
            }
          }

          info.hasSceneColorSampler |= containsToken(lowerName, "scenecolor");
          info.hasSceneDepthSampler |= containsToken(lowerName, "scenedepth") ||
                                       containsToken(lowerName, "destdepth") ||
                                       containsToken(lowerName, "pixeldepth");
          info.hasLightAttenuationSampler |= containsToken(lowerName, "lightattenuation");
          info.hasShadowSampler |= containsToken(lowerName, "shadowdepth") ||
                                   containsToken(lowerName, "shadowtexture") ||
                                   containsToken(lowerName, "shadowvariance");
          info.hasVelocitySampler |= containsToken(lowerName, "velocitybuffer") ||
                                     containsToken(lowerName, "velocitytexture");
          info.hasBlurredImageSampler |= containsToken(lowerName, "blurredimage");
          info.hasFilterTextureSampler |= containsToken(lowerName, "filtertexture");
          info.hasExposureOrToneSampler |= containsToken(lowerName, "exposure") ||
                                           containsToken(lowerName, "colorcurves") ||
                                           containsToken(lowerName, "saturationmask") ||
                                           info.hasBlurredImageSampler ||
                                           info.hasFilterTextureSampler;
          info.hasUiSampler |= containsToken(lowerName, "scenecoloruitexture") ||
                               containsToken(lowerName, "blurredui") ||
                               containsToken(lowerName, "uitexture") ||
                               containsToken(lowerName, "uibuffer");
          info.hasDistortionSampler |= containsToken(lowerName, "distortion") ||
                                       containsToken(lowerName, "lineintegral");
        }

        info.hasBinkConstants |= containsToken(lowerName, "ycrcb") ||
                                 containsToken(lowerName, "yuv") ||
                                 containsToken(lowerName, "bink") ||
                                 lowerName == "tor" ||
                                 lowerName == "tog" ||
                                 lowerName == "tob";
        info.hasPrevViewProjection |= containsToken(lowerName, "prevviewprojectionmatrix") ||
                                      containsToken(lowerName, "previousviewprojectionmatrix") ||
                                      containsToken(lowerName, "prev_view_projection_matrix");
        info.hasVelocityConstants |= containsToken(lowerName, "velocityscaleoffset") ||
                                     containsToken(lowerName, "individualvelocityscale") ||
                                     containsToken(lowerName, "stretchtimescale");
        info.hasMotionBlurConstants |= containsToken(lowerName, "motionpacked") ||
                                       containsToken(lowerName, "staticvelocityparameters") ||
                                       containsToken(lowerName, "motionblur");
        info.hasDynamicLightingConstants |= lowerName == "lightcolor" ||
                                            containsToken(lowerName, "lightcolorandfalloffexponent") ||
                                            containsToken(lowerName, "lightpositionandinvradius") ||
                                            containsToken(lowerName, "lightdirection") ||
                                            containsToken(lowerName, "spotdirection") ||
                                            containsToken(lowerName, "tangentlightvector") ||
                                            containsToken(lowerName, "worldlightvector");
        info.hasLightFunctionConstants |= containsToken(lowerName, "screentolight") ||
                                          containsToken(lowerName, "screen_to_light");
        info.hasSphericalHarmonicLightingConstants |= containsToken(lowerName, "worldincidentlighting") ||
                                                     containsToken(lowerName, "shbasiscubetextures") ||
                                                     containsToken(lowerName, "shbasis");
        info.hasScreenToShadowMatrix |= containsToken(lowerName, "screentoshadowmatrix") ||
                                        containsToken(lowerName, "screen_to_shadow");
        info.hasShadowModulateConstants |= containsToken(lowerName, "shadowmodulatecolor") ||
                                           containsToken(lowerName, "shadowattenuation") ||
                                           containsToken(lowerName, "invmaxsubjectdepth");
        info.hasToneMapConstants |= containsToken(lowerName, "sceneshadowsanddesaturation") ||
                                    containsToken(lowerName, "sceneinversehighlights") ||
                                    containsToken(lowerName, "scenemidtones") ||
                                    containsToken(lowerName, "scenescaledluminanceweights") ||
                                    containsToken(lowerName, "exposuresettings");
        info.hasGammaConstants |= containsToken(lowerName, "gammacolorscaleandinverse") ||
                                  containsToken(lowerName, "gammaoverlaycolor") ||
                                  containsToken(lowerName, "inversegamma") ||
                                  containsToken(lowerName, "colorscale");
        info.hasFogConstants |= containsToken(lowerName, "fog") ||
                                containsToken(lowerName, "heightfog") ||
                                containsToken(lowerName, "lineintegral");
        info.hasHazeConstants |= containsToken(lowerName, "haze") ||
                                 containsToken(lowerName, "sunvector");
        info.hasUiCompositeConstants |= lowerName == "fade" ||
                                        containsToken(lowerName, "scenecolorui") ||
                                        containsToken(lowerName, "bluramount");
        info.hasDofPackedParameters |= containsToken(lowerName, "packedparameters");
        info.hasDofMinMaxBlurClamp |= containsToken(lowerName, "minmaxblurclamp");
        info.hasFilterSampleWeights |= containsToken(lowerName, "sampleweights");

        // TdToneMapping capture: record the exact float register indices of
        // the grade constants so the (skipped) tonemap pass's pixel shader
        // constants can be read at draw time.
        if (isFloat4 && registerIndex <= 0x7FFF) {
          if (lowerName == "sceneshadowsanddesaturation") {
            info.toneMapSceneShadowsReg = int16_t(registerIndex);
          } else if (lowerName == "sceneinversehighlights") {
            info.toneMapInverseHighLightsReg = int16_t(registerIndex);
          } else if (lowerName == "scenemidtones") {
            info.toneMapMidTonesReg = int16_t(registerIndex);
          } else if (lowerName == "scenescaledluminanceweights") {
            info.toneMapScaledLumaWeightsReg = int16_t(registerIndex);
          } else if (lowerName == "gammacolorscaleandinverse") {
            info.toneMapGammaColorScaleReg = int16_t(registerIndex);
          } else if (lowerName == "gammaoverlaycolor") {
            info.toneMapGammaOverlayReg = int16_t(registerIndex);
          } else if (lowerName == "exposuresettings") {
            info.exposureSettingsReg = int16_t(registerIndex);
          } else if (lowerName == "maxdeltadown") {
            info.maxDeltaDownReg = int16_t(registerIndex);
          }
        }
      };

      for (const DxsoCtab::Constant& c : ctab.m_constantData) {
        const bool isSampler = c.registerSet == kD3dxRegisterSetSampler;
        const bool isFloat4 = c.registerSet == kD3dxRegisterSetFloat4;
        markName(toLowerAscii(c.name), isSampler, isFloat4, c.registerIndex);
      }
    } catch (...) {
      // Cached as-is so a malformed CTAB is decoded (and reported) once per shader.
      Logger::warn(str::format("[RTX-Compatibility][UE3] Could not read the CTAB of shader 0x", std::hex, shaderHash,
                               std::dec, "; its draws are classified without UE3 shader features."));
    }

    m_ue3ShaderFeatureCache.emplace(shaderHash, info);
    return info;
  }

  bool D3D9Rtx::isUe3WorldGeometryVertexFactory(const Ue3VertexFactoryType type) {
    switch (type) {
    case Ue3VertexFactoryType::Local:
    case Ue3VertexFactoryType::LocalDecal:
    case Ue3VertexFactoryType::GPUSkin:
    case Ue3VertexFactoryType::GPUSkinMorph:
    case Ue3VertexFactoryType::Terrain:
    case Ue3VertexFactoryType::TerrainMorph:
    case Ue3VertexFactoryType::SpeedTree:
    case Ue3VertexFactoryType::Foliage:
    case Ue3VertexFactoryType::ParticleInstancedMesh:
    case Ue3VertexFactoryType::Particle:
    case Ue3VertexFactoryType::ParticleBeamTrail:
    case Ue3VertexFactoryType::LensFlare:
      return true;
    default:
      return false;
    }
  }

  bool D3D9Rtx::ue3ViewportAspectMatchesBackbuffer(
      const uint32_t vpW, const uint32_t vpH, const uint32_t bbW, const uint32_t bbH) {
    if (vpW == 0 || vpH == 0 || bbW == 0 || bbH == 0) {
      return false;
    }
    const double a = double(vpW) * double(bbH);
    const double b = double(vpH) * double(bbW);
    const double denom = std::max(a, b);
    return denom > 0.0 && (std::abs(a - b) / denom) < 0.05;
  }

  bool D3D9Rtx::ue3ViewportIsMainViewSized(
      const uint32_t vpW, const uint32_t vpH, const uint32_t bbW, const uint32_t bbH) {
    return bbW != 0 && bbH != 0 &&
           vpW * 2 >= bbW && vpH * 2 >= bbH &&
           ue3ViewportAspectMatchesBackbuffer(vpW, vpH, bbW, bbH);
  }

  Ue3PassType D3D9Rtx::classifyUe3Pass(const DrawContext& drawContext) {
    if (!m_frameOptions.ue3EngineMode) {
      return Ue3PassType::Unknown;
    }

    const bool depthEnabled = d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_TRUE;
    const bool zWriteEnabled = d3d9State().renderStates[D3DRS_ZWRITEENABLE];
    const bool samplesRenderTarget = m_parent->GetActiveRTTextures() != 0;
    // UE3 binds textures lazily, leaving render targets in slots the current shader never
    // reads; classification keyed on raw bound-RT state varies with draw order (which UE3
    // re-sorts per camera for translucency) and flickers. Only count render targets bound
    // to samplers the active pixel shader actually uses.
    const bool psSamplesRenderTarget =
      (m_parent->GetActiveRTTextures() & m_parent->m_psShaderMasks.samplerMask) != 0;
    const bool likelyFullscreen = !depthEnabled && !zWriteEnabled && drawContext.PrimitiveCount <= 4;

    const bool isWorldGeometry = isUe3WorldGeometryVertexFactory(m_currentUe3VertexFactory);

    // Cheap early-outs first: depth prepass and shadow depth draws are the most common
    // skipped passes and need no shader feature information.
    if (m_currentUe3VertexFactory == Ue3VertexFactoryType::PositionOnly) {
      return Ue3PassType::DepthPrepass;
    }

    if (m_activePresentParams.has_value() &&
        d3d9State().renderTargets[kRenderTargetIndex] != nullptr) {
      const auto& rtExt = d3d9State().renderTargets[kRenderTargetIndex]->GetSurfaceExtent();
      const uint32_t bbW = m_activePresentParams->BackBufferWidth;
      if (rtExt.width == rtExt.height &&
          rtExt.width <= 2048 &&
          rtExt.width < bbW / 2 &&
          zWriteEnabled) {
        return Ue3PassType::ShadowDepth;
      }
    }

    const D3D9CommonShader* vertexShaderCommon =
      m_parent->UseProgrammableVS() && d3d9State().vertexShader.ptr() != nullptr
        ? d3d9State().vertexShader->GetCommonShader()
        : nullptr;
    const D3D9CommonShader* pixelShaderCommon =
      m_parent->UseProgrammablePS() && d3d9State().pixelShader.ptr() != nullptr
        ? d3d9State().pixelShader->GetCommonShader()
        : nullptr;

    const Ue3ShaderFeatureInfo vsInfo = getUe3ShaderFeatureInfo(vertexShaderCommon);
    const Ue3ShaderFeatureInfo psInfo = getUe3ShaderFeatureInfo(pixelShaderCommon);

    if (psInfo.hasBinkConstants) {
      if (likelyFullscreen && !isWorldGeometry) {
        return Ue3PassType::VideoCinematic;
      }
      return Ue3PassType::VideoSurface;
    }

    if (psInfo.hasUiSampler || psInfo.hasUiCompositeConstants) {
      return Ue3PassType::UiComposite;
    }

    // UE3 SceneCapture probes re-render the world before the main view; viewport size/aspect
    // (and later mirrored/undecomposable camera checks) keep them from stealing Main.
    if (m_frameOptions.ue3EngineMode &&
        isWorldGeometry &&
        m_activePresentParams.has_value()) {
      const D3DVIEWPORT9& vp = d3d9State().viewport;
      const uint32_t bbW = m_activePresentParams->BackBufferWidth;
      const uint32_t bbH = m_activePresentParams->BackBufferHeight;
      if (bbW != 0 && bbH != 0 && vp.Width != 0 && vp.Height != 0) {
        if (vp.Width * 2 < bbW && vp.Height * 2 < bbH) {
          return Ue3PassType::SceneCapture;
        }
        // Exact-half splitscreen must not be treated as a mismatched-aspect probe.
        const bool exactHalfSplit =
          (vp.Width * 2 == bbW && vp.Height == bbH) ||
          (vp.Height * 2 == bbH && vp.Width == bbW);
        if (!exactHalfSplit &&
            (vp.Width < bbW || vp.Height < bbH) &&
            !ue3ViewportAspectMatchesBackbuffer(vp.Width, vp.Height, bbW, bbH)) {
          return Ue3PassType::SceneCapture;
        }
      }
    }

    if (psInfo.hasScreenToShadowMatrix ||
        psInfo.hasShadowModulateConstants ||
        (psInfo.hasShadowSampler && psInfo.hasSceneDepthSampler)) {
      return Ue3PassType::ModulatedShadowProjection;
    }

    if ((vsInfo.hasPrevViewProjection || psInfo.hasVelocityConstants) &&
        !samplesRenderTarget) {
      return Ue3PassType::Velocity;
    }

    const bool hasDynamicLightPassSignals =
      (psInfo.hasDynamicLightingConstants || vsInfo.hasDynamicLightingConstants) &&
      (psInfo.hasLightAttenuationSampler ||
       psInfo.hasShadowSampler ||
       vsInfo.hasDynamicLightingConstants);
    // UE3 folds light-environment lighting into the BASE pass (SH WorldIncidentLighting /
    // TangentLightVector constants), so a draw with material samplers is the primary
    // textured draw, not a per-light pass - unless it uses additive light-pass blending
    const bool isAdditiveLightBlend =
      d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] &&
      d3d9State().renderStates[D3DRS_SRCBLEND] == D3DBLEND_ONE &&
      d3d9State().renderStates[D3DRS_DESTBLEND] == D3DBLEND_ONE;
    const bool looksLikeLitBasePassMaterial = psInfo.hasMaterialSampler && !isAdditiveLightBlend;
    if (!looksLikeLitBasePassMaterial &&
        (hasDynamicLightPassSignals ||
         psInfo.hasLightFunctionConstants ||
         psInfo.hasSphericalHarmonicLightingConstants ||
         (psInfo.hasLightAttenuationSampler && psInfo.hasShadowSampler))) {
      return Ue3PassType::Lighting;
    }

    if ((psInfo.hasVelocitySampler || psInfo.hasMotionBlurConstants) &&
        (samplesRenderTarget || likelyFullscreen)) {
      return Ue3PassType::FullscreenPostProcess;
    }

    // Explicit DOFAndBloom/Uber signals (also covered by the catch-all below for typical
    // ME shaders); kept so FilterColor blur is classified even if z-write state is atypical.
    if (!isWorldGeometry &&
        (likelyFullscreen || psSamplesRenderTarget) &&
        psInfo.looksLikeDofAndBloomPostProcess()) {
      return Ue3PassType::FullscreenPostProcess;
    }

    // The two screen-space catch-alls below must never swallow world geometry:
    // translucent world meshes render with depth writes off and legitimately declare
    // scene color/depth samplers (DepthBiasedAlpha, DestColor/refraction) or fog inputs
    if (!isWorldGeometry &&
        (likelyFullscreen || psSamplesRenderTarget || !zWriteEnabled) &&
        (psInfo.hasFogConstants || psInfo.hasHazeConstants || psInfo.hasDistortionSampler)) {
      return Ue3PassType::FogOrDistortion;
    }

    if (!isWorldGeometry &&
        !zWriteEnabled &&
        (likelyFullscreen || psSamplesRenderTarget) &&
        (psInfo.hasSceneColorSampler ||
         psInfo.hasSceneDepthSampler ||
         psInfo.hasLightAttenuationSampler ||
         psInfo.hasExposureOrToneSampler ||
         psInfo.hasToneMapConstants ||
         psInfo.hasGammaConstants ||
         psInfo.hasEngineAuxSampler)) {
      return Ue3PassType::FullscreenPostProcess;
    }

    if (psInfo.hasMaterialSampler ||
        m_currentUe3VertexFactory == Ue3VertexFactoryType::Local ||
        m_currentUe3VertexFactory == Ue3VertexFactoryType::LocalDecal ||
        m_currentUe3VertexFactory == Ue3VertexFactoryType::GPUSkin ||
        m_currentUe3VertexFactory == Ue3VertexFactoryType::GPUSkinMorph ||
        m_currentUe3VertexFactory == Ue3VertexFactoryType::Terrain ||
        m_currentUe3VertexFactory == Ue3VertexFactoryType::TerrainMorph ||
        m_currentUe3VertexFactory == Ue3VertexFactoryType::SpeedTree ||
        m_currentUe3VertexFactory == Ue3VertexFactoryType::Foliage ||
        m_currentUe3VertexFactory == Ue3VertexFactoryType::ParticleInstancedMesh) {
      return Ue3PassType::Material;
    }

    return Ue3PassType::Unknown;
  }

  void D3D9Rtx::onUe3CurveTextureUpload(const D3D9CommonTexture* dstTexture, D3D9CommonTexture* srcTexture, const uint32_t srcSubresource,
                                        const uint32_t srcTexelOffsetX, const uint32_t dstTexelOffsetX,
                                        const uint32_t texelWidth, const uint32_t texelHeight) {
    if (!m_frameOptions.ue3EngineMode || dstTexture == nullptr || srcTexture == nullptr) {
      return;
    }

    // The curve LUTs are 16x1 float RGBA textures (one texel per curve segment)
    const auto* desc = dstTexture->Desc();
    if (desc->Width != kUe3CurveTexelCount || desc->Height != 1) {
      return;
    }

    const D3D9Format format = desc->Format;
    if (format != D3D9Format::A32B32G32R32F && format != D3D9Format::A16B16G16R16F) {
      // Near-miss diagnostic: a 16x1 destination in an unexpected format would
      // mean the game's curve textures need another decode path.
      ONCE(Logger::info(str::format("[RTX-UE3-Tonemap] Ignoring 16x1 texture upload in unsupported format ",
                                    uint32_t(format), " (expected A32B32G32R32F/A16B16G16R16F).")));
      return;
    }

    if (texelHeight != 1 || texelWidth == 0 || dstTexelOffsetX >= kUe3CurveTexelCount) {
      return;
    }

    const void* srcData = srcTexture->GetMappedSlice(srcSubresource).mapPtr;
    if (srcData == nullptr) {
      return;
    }

    // Bounded cache: keys are only ever compared against currently-bound
    // textures (never dereferenced), so stale entries are harmless; still keep
    // the map tiny since only a handful of curve textures ever exist.
    if (m_ue3CurveTexelCache.size() > 16 && m_ue3CurveTexelCache.find(dstTexture) == m_ue3CurveTexelCache.end()) {
      m_ue3CurveTexelCache.clear();
    }
    Ue3CurveTexels& payload = m_ue3CurveTexelCache[dstTexture];

    const uint32_t count = std::min(texelWidth, kUe3CurveTexelCount - dstTexelOffsetX);

    if (format == D3D9Format::A32B32G32R32F) {
      const float* texels = reinterpret_cast<const float*>(srcData) + srcTexelOffsetX * 4;
      for (uint32_t i = 0; i < count; i++) {
        payload.texels[dstTexelOffsetX + i] = Vector4(texels[i * 4 + 0], texels[i * 4 + 1], texels[i * 4 + 2], texels[i * 4 + 3]);
      }
    } else {
      const uint16_t* texels = reinterpret_cast<const uint16_t*>(srcData) + srcTexelOffsetX * 4;
      for (uint32_t i = 0; i < count; i++) {
        payload.texels[dstTexelOffsetX + i] = Vector4(ue3HalfToFloat(texels[i * 4 + 0]),
                                                      ue3HalfToFloat(texels[i * 4 + 1]),
                                                      ue3HalfToFloat(texels[i * 4 + 2]),
                                                      ue3HalfToFloat(texels[i * 4 + 3]));
      }
    }

    ONCE(Logger::info(str::format("[RTX-UE3-Tonemap] Snooping curve LUT texture uploads (",
                                  format == D3D9Format::A32B32G32R32F ? "fp32" : "fp16",
                                  ", rect x=", dstTexelOffsetX, " w=", count, ").")));
  }

  void D3D9Rtx::maybeCaptureUe3ToneMapState() {
    if (!m_frameOptions.ue3EngineMode) {
      return;
    }

    // Only spend effort when the Mirror's Edge tonemapper consumes the capture
    if (RtxOptions::tonemappingMode() != TonemappingMode::MirrorsEdge) {
      return;
    }

    if (!m_parent->UseProgrammablePS() || d3d9State().pixelShader == nullptr) {
      return;
    }

    const Ue3ShaderFeatureInfo psInfo = getUe3ShaderFeatureInfo(d3d9State().pixelShader->GetCommonShader());

    // The TdToneMapExposure draw (ExposureSettings + MaxDeltaDown, no curve samplers) precedes
    // the tonemap draw, so it is recognised ahead of the once-per-frame tonemap gate. Its
    // constants are the current PostProcessVolume's exposure clamps and speeds, joined into the
    // tonemap capture below for the Mirror's Edge exposure meter.
    if (psInfo.exposureSettingsReg >= 0 && psInfo.exposureSettingsReg < int16_t(caps::MaxFloatConstantsPS) &&
        psInfo.colorCurvesKSamplerIndex >= caps::MaxTexturesPS) {
      m_ue3ExposureSettings = d3d9State().psConsts.fConsts[psInfo.exposureSettingsReg];
      if (psInfo.maxDeltaDownReg >= 0 && psInfo.maxDeltaDownReg < int16_t(caps::MaxFloatConstantsPS)) {
        m_ue3MaxDeltaDown = d3d9State().psConsts.fConsts[psInfo.maxDeltaDownReg].x;
      }
      if (!m_ue3HasExposureSettings) {
        Logger::info(str::format("[RTX-UE3-Tonemap] Captured TdToneMapExposure settings (Manual=", m_ue3ExposureSettings.x,
                                 " Low=", m_ue3ExposureSettings.z, " High=", m_ue3ExposureSettings.w, ")."));
      }
      m_ue3HasExposureSettings = true;
      return;
    }

    if (m_ue3ToneMapCapturedThisFrame) {
      return;
    }

    // Only the TdToneMapping pass declares ColorCurvesK/M; require those sampler
    // indices so UberPostProcessBlend (grade+gamma only) cannot lock out capture.
    if (!psInfo.hasToneMapConstants || !psInfo.hasGammaConstants ||
        psInfo.toneMapSceneShadowsReg < 0 || psInfo.toneMapGammaColorScaleReg < 0 ||
        psInfo.toneMapMidTonesReg < 0 ||
        psInfo.colorCurvesKSamplerIndex >= caps::MaxTexturesPS ||
        psInfo.colorCurvesMSamplerIndex >= caps::MaxTexturesPS) {
      return;
    }

    Ue3ToneMapCapture capture;

    const auto readConstant = [&](const int16_t reg, Vector4& target) {
      if (reg >= 0 && reg < int16_t(caps::MaxFloatConstantsPS)) {
        target = d3d9State().psConsts.fConsts[reg];
      }
    };
    readConstant(psInfo.toneMapSceneShadowsReg, capture.sceneShadowsAndDesaturation);
    readConstant(psInfo.toneMapInverseHighLightsReg, capture.sceneInverseHighLights);
    readConstant(psInfo.toneMapMidTonesReg, capture.sceneMidTones);
    readConstant(psInfo.toneMapScaledLumaWeightsReg, capture.sceneScaledLuminanceWeights);
    readConstant(psInfo.toneMapGammaColorScaleReg, capture.gammaColorScaleAndInverse);
    readConstant(psInfo.toneMapGammaOverlayReg, capture.gammaOverlayColor);
    capture.hasConstants = true;

    capture.hasExposureSettings = m_ue3HasExposureSettings;
    capture.exposureSettings = m_ue3ExposureSettings;
    capture.maxDeltaDown = m_ue3MaxDeltaDown;

    const auto resolveCurveTexels = [&](const uint8_t samplerIndex, std::array<Vector4, kUe3CurveTexelCount>& target) {
      if (samplerIndex >= caps::MaxTexturesPS) {
        return false;
      }
      IDirect3DBaseTexture9* texture = d3d9State().textures[samplerIndex];
      if (texture == nullptr) {
        return false;
      }
      const auto it = m_ue3CurveTexelCache.find(GetCommonTexture(texture));
      if (it == m_ue3CurveTexelCache.end()) {
        return false;
      }
      target = it->second.texels;
      return true;
    };

    capture.hasCurves = resolveCurveTexels(psInfo.colorCurvesKSamplerIndex, capture.curveK) &&
                        resolveCurveTexels(psInfo.colorCurvesMSamplerIndex, capture.curveM);

    // Capture the sampler filter bound for the curve LUTs so the tonemapper's
    // lookup matches the game exactly (point snaps to one segment texel,
    // bilinear blends adjacent ones).
    if (psInfo.colorCurvesKSamplerIndex < caps::MaxTexturesPS) {
      const DWORD magFilter = d3d9State().samplerStates[psInfo.colorCurvesKSamplerIndex][D3DSAMP_MAGFILTER];
      capture.curvePointFiltering = (magFilter == D3DTEXF_POINT || magFilter == D3DTEXF_NONE);
      if (capture.curvePointFiltering) {
        ONCE(Logger::info("[RTX-UE3-Tonemap] Game samples its curve LUTs with point filtering; matching in the tonemapper."));
      }
    }

    m_ue3ToneMapCapturedThisFrame = true;

    ONCE(Logger::info(str::format("[RTX-UE3-Tonemap] Capturing TdToneMapping pass state (curve textures resolved: ",
                                  capture.hasCurves ? "yes" : "no", ")")));

    if (!capture.hasCurves) {
      // Pinpoint why curve resolution failed: missing CTAB sampler indices,
      // nothing bound at the slots, or no snooped upload for the bound texture.
      const auto describeCurveSampler = [&](const uint8_t samplerIndex) -> std::string {
        if (samplerIndex >= caps::MaxTexturesPS) {
          return "sampler not in CTAB";
        }
        IDirect3DBaseTexture9* texture = d3d9State().textures[samplerIndex];
        if (texture == nullptr) {
          return str::format("slot ", uint32_t(samplerIndex), ": no texture bound");
        }
        const auto* common = GetCommonTexture(texture);
        const auto* desc = common->Desc();
        const bool snooped = m_ue3CurveTexelCache.find(common) != m_ue3CurveTexelCache.end();
        return str::format("slot ", uint32_t(samplerIndex), ": ", desc->Width, "x", desc->Height,
                           " fmt=", uint32_t(desc->Format), snooped ? " [snooped]" : " [no snooped upload]");
      };
      ONCE(Logger::warn(str::format("[RTX-UE3-Tonemap] Curve textures unresolved at tonemap pass. K={",
                                    describeCurveSampler(psInfo.colorCurvesKSamplerIndex), "} M={",
                                    describeCurveSampler(psInfo.colorCurvesMSamplerIndex), "}")));
    }

    m_parent->EmitCs([capture](DxvkContext* ctx) {
      static_cast<RtxContext*>(ctx)->setUe3ToneMapCapture(capture);
    });
  }

  bool D3D9Rtx::isAutoDetectedLightmapTexture(const XXH64_hash_t textureHash) {
    if (textureHash == kEmptyHash || !g_autoLightmapTexturesPopulated.load(std::memory_order_acquire)) {
      return false;
    }

    std::shared_lock lock(g_autoLightmapTexturesMutex);
    return g_autoLightmapTextures.find(textureHash) != g_autoLightmapTextures.end();
  }

  void D3D9Rtx::registerAutoDetectedLightmapTexture(const XXH64_hash_t textureHash) {
    if (textureHash == kEmptyHash) {
      return;
    }

    bool inserted = false;
    {
      std::unique_lock lock(g_autoLightmapTexturesMutex);
      inserted = g_autoLightmapTextures.insert(textureHash).second;
      if (inserted) {
        g_autoLightmapTexturesPopulated.store(true, std::memory_order_release);
      }
    }

    if (inserted) {
      ImGUI::AddAutoTaggedLightmapTexture(textureHash);
    }
  }

  void D3D9Rtx::OnClear(DWORD flags) {
    if (!m_frameOptions.ue3EngineMode || !m_frameOptions.ue3ForegroundDpgIsViewModel) {
      return;
    }

    // UE3 renders SDPG_Foreground (first-person arms, held weapon, muzzle flash) after the
    // world DPG behind a depth-only clear so foreground meshes never depth-clash with the
    // world. A z-only clear on a main-view-sized viewport after this frame's world draws
    // marks every subsequent draw as foreground until EndFrame.
    if ((flags & D3DCLEAR_TARGET) != 0 || (flags & D3DCLEAR_ZBUFFER) == 0) {
      return;
    }
    if (m_ue3ForegroundDpgActive || !m_ue3SeenMainViewWorldDraw) {
      return;
    }
    if (d3d9State().depthStencil == nullptr || !m_activePresentParams.has_value()) {
      return;
    }

    // Shadow / SceneCapture depth clears use sub-main-view viewports; ignore those.
    const D3DVIEWPORT9& vp = d3d9State().viewport;
    const uint32_t bbW = m_activePresentParams->BackBufferWidth;
    const uint32_t bbH = m_activePresentParams->BackBufferHeight;
    if (!ue3ViewportIsMainViewSized(vp.Width, vp.Height, bbW, bbH)) {
      return;
    }

    m_ue3ForegroundDpgActive = true;
    ONCE(Logger::info("[RTX-Compatibility-Info] UE3 foreground DPG boundary detected (mid-scene depth-only clear); subsequent draws classify as ViewModel."));
    Logger::debug(str::format("[RTX-Compatibility][UE3] Foreground DPG boundary after draw=", m_activeDrawCallState.drawCallID));
  }

  bool D3D9Rtx::trackUe3MovieTextureRenderTarget(const char* reason) {
    if (d3d9State().renderTargets[kRenderTargetIndex] == nullptr ||
        !m_activePresentParams.has_value()) {
      return false;
    }

    D3D9CommonTexture* rtTexture = GetCommonTexture(d3d9State().renderTargets[kRenderTargetIndex]->GetBaseTexture());
    if (rtTexture == nullptr || rtTexture->GetImage() == nullptr || rtTexture->Desc() == nullptr) {
      return false;
    }

    if (isRenderTargetPrimary(*m_activePresentParams, rtTexture->Desc())) {
      return false;
    }

    const XXH64_hash_t descHash = rtTexture->GetImage()->getDescriptorHash();
    if (descHash == kEmptyHash) {
      return false;
    }

    const bool inserted = m_ue3MovieTextureDescHashes.insert(descHash).second;
    if (inserted && Logger::logLevel() <= LogLevel::Debug) {
      const auto* desc = rtTexture->Desc();
      Logger::debug(str::format(
        "[RTX-Compatibility][UE3] Tracked movie texture render target: ",
        desc->Width, "x", desc->Height,
        ", descHash=0x", std::hex, descHash, std::dec,
        ", reason=", reason));
    }
    return true;
  }

  bool D3D9Rtx::isUe3MovieTextureDescHash(const XXH64_hash_t descHash) const {
    return descHash != kEmptyHash &&
           m_ue3MovieTextureDescHashes.find(descHash) != m_ue3MovieTextureDescHashes.end();
  }

  void D3D9Rtx::updateUe3GamePatchRequest() {
    uint32_t request = 0;
    uint32_t limits = 0;
    if (m_frameOptions.ue3EngineMode && m_frameOptions.enableRaytracing) {
      const std::pair<bool, GamePatchBits> patches[] = {
        { ue3DisableFrustumCulling(), kGamePatchDisableFrustumCulling },
        { ue3ShowThirdPersonModel(), kGamePatchShowThirdPersonModel },
        { ue3DisableOcclusionQueries(), kGamePatchDisableOcclusionQueries },
        { ue3DisableSceneCaptures(), kGamePatchDisableSceneCaptures },
        { ue3DisableDynamicShadows(), kGamePatchDisableDynamicShadows },
        { ue3DisableDynamicLighting(), kGamePatchDisableDynamicLighting },
        { ue3DisableVelocityPass(), kGamePatchDisableVelocityPass },
      };
      for (const auto& [enabled, bit] : patches) {
        if (enabled) {
          request |= bit;
        }
      }

      const float meterToWorldUnits = RtxOptions::getMeterToWorldUnitScale();
      const float maxDistance = ue3FrustumBypassMaxDistanceMeters() * meterToWorldUnits;
      if ((request & kGamePatchDisableFrustumCulling) != 0 && maxDistance > 0.f) {
        request |= kGamePatchLimitFrustumBypass;
        limits = packFrustumBypassLimits(maxDistance, ue3FrustumBypassMinRadiusMeters() * meterToWorldUnits);
      }
    }

    // Repeated until the bridge client first answers, as an answer sent before the message
    // channel handshake completes is lost.
    const auto now = std::chrono::steady_clock::now();
    const bool unanswered = (s_ue3GamePatchStatus.load(std::memory_order_relaxed) & kUe3GamePatchAnswered) == 0;
    const bool retry = request != 0 && unanswered && now - m_ue3GamePatchRequestTime >= kUe3GamePatchRequestRetryInterval;
    if ((request != m_ue3GamePatchRequest || limits != m_ue3GamePatchLimits || retry) &&
        BridgeMessageChannel::get().send(kGamePatchRequestMsgName, request, limits)) {
      m_ue3GamePatchRequest = request;
      m_ue3GamePatchLimits = limits;
      m_ue3GamePatchRequestTime = now;
    }
  }

  // NeedsDepthTestDisabled materials, fog volume composites and fullscreen overlays: alpha blend with
  // depth test and depth write off. UI and deferred-UI tagged draws match too, but rasterize.
  bool D3D9Rtx::isUe3DepthTestDisabledTranslucency(DeferredUiTagQuery& deferredUiTag) const {
    return m_frameOptions.ue3EngineMode &&
           d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] &&
           (d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_FALSE ||
            d3d9State().renderStates[D3DRS_ZFUNC] == D3DCMP_ALWAYS) &&
           !d3d9State().renderStates[D3DRS_ZWRITEENABLE] &&
           !checkBoundTextureCategory(*m_frameOptions.uiTextures) &&
           !deferredUiTag.isTagged();
  }

  // The UE3 passes Remix ignores or rasterizes, and the main-view gate for the foreground DPG.
  std::optional<D3D9Rtx::DrawCallType> D3D9Rtx::classifyUe3DrawPass(const DrawContext& drawContext,
                                                                    DeferredUiTagQuery& deferredUiTag) {
    // UE3 depth prepass - position only vertex declarations have no texcoords/colours
    // the same geometry will be drawn again in the base pass with full material
    if (m_frameOptions.ue3EngineMode &&
        m_currentUe3VertexFactory == Ue3VertexFactoryType::PositionOnly) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped UE3 depth prepass draw (position-only vertex declaration)."));
      return DrawCallType { RtxGeometryStatus::Ignored, false };
    }

    m_currentUe3PassType = classifyUe3Pass(drawContext);

    // Capture TdToneMapping state before the draw is ignored below.
    if (m_currentUe3PassType == Ue3PassType::FullscreenPostProcess) {
      maybeCaptureUe3ToneMapState();
    }

    switch (m_currentUe3PassType) {
    case Ue3PassType::DepthPrepass:
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "position-only depth prepass");
      return DrawCallType { RtxGeometryStatus::Ignored, false };
    case Ue3PassType::ShadowDepth:
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "shadow depth render target");
      return DrawCallType { RtxGeometryStatus::Ignored, false };
    case Ue3PassType::Velocity:
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "native velocity helper pass");
      return DrawCallType { RtxGeometryStatus::Ignored, false };
    case Ue3PassType::Lighting:
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "native UE3 lighting pass");
      return DrawCallType { RtxGeometryStatus::Ignored, false };
    case Ue3PassType::ModulatedShadowProjection:
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "native modulated shadow projection");
      return DrawCallType { RtxGeometryStatus::Ignored, false };
    case Ue3PassType::SceneCapture:
      if (!m_ue3ForegroundDpgActive) {
        m_ue3SeenMainViewWorldDraw = false;
      }
      ONCE(Logger::info("[RTX-Compatibility-Info] Ignored UE3 scene capture offscreen view draw (world geometry, probe viewport)."));
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "scene capture offscreen view");
      return DrawCallType { RtxGeometryStatus::Ignored, false };
    case Ue3PassType::FullscreenPostProcess:
    case Ue3PassType::FogOrDistortion:
      // Ignore before the non-primary RT → Rasterized fallback; DoF gather/blur/blend
      // target FilterColor/SceneColor and would otherwise still execute. Only a deferred-UI
      // pixel shader tag outranks this: these passes sample the scene-colour target, so a
      // texture tag on it matches every one of them and would replay DoF over the frame.
      if (!deferredUiTag.isPixelShaderTagged()) {
        logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "native UE3 screen-space contribution pass");
        return DrawCallType { RtxGeometryStatus::Ignored, false };
      }
      break;
    case Ue3PassType::UiComposite:
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Rasterized, "UI composite");
      return DrawCallType { RtxGeometryStatus::Rasterized, true };
    case Ue3PassType::VideoCinematic:
      trackUe3MovieTextureRenderTarget("video cinematic/decode");
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Rasterized, "video/cinematic pass");
      return DrawCallType { RtxGeometryStatus::Rasterized, false };
    case Ue3PassType::VideoSurface:
      trackUe3MovieTextureRenderTarget("video surface/decode");
      logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Rasterized, "video texture surface/decode pass");
      return DrawCallType { RtxGeometryStatus::Rasterized, false };
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

    return std::nullopt;
  }

  // UE3 shadow depth pass - draws to small square render targets that are used as shadow maps
  bool D3D9Rtx::isUe3ShadowDepthPass() const {
    if (m_frameOptions.ue3EngineMode && m_activePresentParams.has_value()) {
      const auto& rtExt = d3d9State().renderTargets[kRenderTargetIndex]->GetSurfaceExtent();
      const uint32_t bbW = m_activePresentParams->BackBufferWidth;
      const bool isSmallSquare = rtExt.width == rtExt.height &&
                                 rtExt.width <= 2048 &&
                                 rtExt.width < bbW / 2;
      const bool hasDepthWrite = d3d9State().renderStates[D3DRS_ZWRITEENABLE];
      if (isSmallSquare && hasDepthWrite) {
        ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Skipped UE3 shadow depth pass (",
                                       rtExt.width, "x", rtExt.height, ").")));
        return true;
      }
    }
    return false;
  }

  void D3D9Rtx::classifyUe3DrawVertexFactory() {
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
  }

  // A material showing a movie render target is world UI (monitors, billboards).
  void D3D9Rtx::markUe3MovieTextureMaterial(const Ue3TextureState& ue3, const XXH64_hash_t materialHash,
                                            const XXH64_hash_t textureHash) {
    bool usesMovieTexture = ue3.selectedUe3MovieTexture;
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
  }

}
