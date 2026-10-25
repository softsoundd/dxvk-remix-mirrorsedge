#pragma once

#include "d3d9_include.h"
#include "d3d9_state.h"

#include "d3d9_util.h"
#include "d3d9_buffer.h"

#include "d3d9_rtx.h"
#include "d3d9_device.h"

#include "../util/util_fastops.h"
#include "../util/util_math.h"
#include "d3d9_texture.h"
#include "../dxso/dxso_tables.h"
#include "../dxvk/rtx_render/rtx_context.h"
#include "../dxvk/rtx_render/rtx_ngx_passthrough.h"
#include "../dxvk/rtx_render/rtx_options.h"
#include "../dxvk/rtx_render/rtx_dlfg.h"
#include "../dxvk/rtx_render/rtx_dlss_neural_rendering.h"
#include "../dxvk/rtx_render/rtx_camera.h"
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

// Toolhelp snapshots for cross-process game-settings driving.
#include <tlhelp32.h>

namespace dxvk {
  static const bool s_isDxvkResolutionEnvVarSet = (env::getEnvVar("DXVK_RESOLUTION_WIDTH") != "") || (env::getEnvVar("DXVK_RESOLUTION_HEIGHT") != "");
  
  // We only look at RT 0 currently.
  const uint32_t kRenderTargetIndex = 0;

  namespace {
    struct VDeclSignature {
      bool hasPosition = false;
      uint8_t positionType = 0;
      uint8_t positionStream = 0xFF;
      bool hasTangent = false;
      uint8_t tangentType = 0;
      uint8_t tangentStream = 0xFF;
      bool hasNormal = false;
      uint8_t normalType = 0;
      uint8_t normalStream = 0xFF;
      bool hasBinormal = false;
      uint8_t binormalType = 0;
      bool hasBlendWeight = false;
      uint8_t blendWeightType = 0;
      bool hasBlendIndices = false;
      bool hasColor0 = false;
      bool hasColor1 = false;
      uint8_t texcoordCount = 0;
      uint8_t maxTexcoordIndex = 0;
      uint8_t texcoordTypes[8] = {};
      bool hasTexcoord6 = false;
      uint8_t texcoord6Type = 0;
      bool hasTexcoord7 = false;
      uint8_t texcoord7Type = 0;
      uint8_t totalElements = 0;
    };

    static VDeclSignature buildVDeclSignature(const D3D9VertexElements& elements) {
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

    static std::string toLowerAscii(const std::string& s) {
      std::string out;
      out.reserve(s.size());
      for (const char c : s)
        out.push_back(char(std::tolower(static_cast<unsigned char>(c))));
      return out;
    }

    static bool containsToken(const std::string& s, const char* token) {
      return s.find(token) != std::string::npos;
    }

    constexpr uint16_t kD3dxRegisterSetFloat4 = 2u;
    constexpr uint16_t kD3dxRegisterSetSampler = 3u;

    constexpr uint8_t kPsSamplerSemanticEngineAuxiliary = 1u << 0;
    constexpr uint8_t kPsSamplerSemanticMaterialTexture = 1u << 1;

    static uint8_t classifyPixelSamplerSemanticFlags(const std::string& samplerName) {
      std::string lowerName;
      lowerName.reserve(samplerName.size());
      for (const char c : samplerName)
        lowerName.push_back(char(std::tolower(static_cast<unsigned char>(c))));

      auto contains = [&](const char* token) {
        return lowerName.find(token) != std::string::npos;
      };

      uint8_t flags = 0;
      const bool isBinkPlaneName =
        contains("ycrcb") || contains("yuv") || contains("bink");

      if (!isBinkPlaneName &&
          (contains("texture2d_") || contains("texturecube_") || contains("texture3d_") ||
          contains("materialtexture") || contains("materialsampler") ||
          contains("textureparameter") || contains("texturesample") ||
          contains("texturesampleparameter2d") || contains("texturesampleparametercube") ||
          contains("texturesampleparametermovie") || contains("texturesampleparametersubuv") ||
          contains("fontsampleparameter") || contains("movie") ||
          contains("albedo") || contains("diffuse") || contains("basecolor") ||
          contains("base_color") || contains("billboard") || contains("advert") ||
          contains("poster") || contains("decal") || contains("fontsample") ||
          contains("subuv") || contains("flipbook") || contains("particlesubuv") ||
          contains("meshsubuv") || contains("cubemap"))) {
        flags |= kPsSamplerSemanticMaterialTexture;
      }

      if (contains("scenecolor") || contains("scenedepth") || contains("lightattenuation") ||
          contains("previouslighting") || contains("exposuretexture") ||
          contains("previousexposure") || contains("scenedownsampled") ||
          contains("saturationmasktexture") || contains("randomangletexture") ||
          contains("bsplinetexture") || contains("colorcurvesktexture") ||
          contains("colorcurvesmtexture") || contains("blurredimage") ||
          contains("shadowdepth") || contains("shadowvariance") ||
          contains("shadowtexture") || contains("velocitybuffer") ||
          contains("ambientocclusiontexture") || contains("aohistorytexture") ||
          contains("randomnormaltexture") || contains("filtertexture") ||
          contains("scenecolorscratchtexture") || contains("ldrtranslucencytexture") ||
          contains("accumulateddistortiontexture") || contains("accumulatedfrontfaceslineintegraltexture") ||
          contains("accumulatedbackfaceslineintegraltexture") || contains("scenecoloruitexture") ||
          contains("uibuffer") || contains("uitexture") ||
          contains("blurreduitexture") || contains("sceneblurtexture") ||
          contains("colorcurves") || contains("exposure") ||
          contains("ycrcb") || contains("yuv") || contains("bink") ||
          contains("destdepth") || contains("destcolor") ||
          contains("pixeldepth") || contains("scenetexture") ||
          contains("depthbias") || contains("depthbiased") ||
          contains("lensflare") || contains("lens_flare") ||
          contains("screenposition") || contains("screen_position") ||
          contains("masktexture")) {
        flags |= kPsSamplerSemanticEngineAuxiliary;
      }

      // UE3 lightmap machinery: the LightMapTextures[] coefficient array declared by every
      // texture lightmap permutation, plus Mirror's Edge's BSplineTexture weight LUT, which
      // exists only to filter those coefficients (TdBicubicFiltering).
      if (contains("lightmap") || contains("bspline")) {
        flags |= kPsSamplerSemanticEngineAuxiliary;
      }

      return flags;
    }

    bool tryExtractUe3WorldToViewAndProjectionFromShaderConstants(
      const D3D9ShaderConstantsVSSoftware& vsConsts,
      const uint32_t viewProjRegisterBase,
      const uint32_t viewOriginRegister,
      Matrix4& outWorldToView,
      Matrix4& outViewToProjection,
      bool* outUsedTranspose = nullptr,
      float* outReconstructionError = nullptr) {

      // UE3 uploads ViewProjectionMatrix via SetVertexShaderConstantF as 4 consecutive float4 registers
      // These values are the raw shader constants and in UE3/HLSL this matrix is typically treated as column-major
      // We derive a stable (world to view, view to projection) pair by
      // 1 treating the provided matrix as a world to projection transform
      // 2 unprojecting a few NDC points to recover view orientation (using camera position)
      // 3 deriving a pure projection matrix and validating it via projection decomposition

      if (viewProjRegisterBase + 3 >= caps::MaxFloatConstantsSoftware)
        return false;
      if (viewOriginRegister >= caps::MaxFloatConstantsSoftware)
        return false;

      Matrix4 worldToProjection;
      worldToProjection[0] = vsConsts.fConsts[viewProjRegisterBase + 0];
      worldToProjection[1] = vsConsts.fConsts[viewProjRegisterBase + 1];
      worldToProjection[2] = vsConsts.fConsts[viewProjRegisterBase + 2];
      worldToProjection[3] = vsConsts.fConsts[viewProjRegisterBase + 3];

      const Vector3 camPos = vsConsts.fConsts[viewOriginRegister].xyz();

      // Quick reject: all-zero matrices show up during some initialization paths
      constexpr float kEps = 1e-6f;
      if (lengthSqr(worldToProjection[0].xyz()) < kEps &&
          lengthSqr(worldToProjection[1].xyz()) < kEps &&
          lengthSqr(worldToProjection[2].xyz()) < kEps &&
          lengthSqr(worldToProjection[3].xyz()) < kEps) {
        return false;
      }

      auto validateProjection = [](const Matrix4& viewToProjection, DecomposeProjectionParams& outParams) {
        decomposeProjection(viewToProjection, outParams);

        const bool finite =
          std::isfinite(outParams.fov) &&
          std::isfinite(outParams.aspectRatio) &&
          std::isfinite(outParams.nearPlane) &&
          std::isfinite(outParams.farPlane) &&
          std::isfinite(outParams.shearX) &&
          std::isfinite(outParams.shearY);
        if (!finite)
          return false;

        if (outParams.fov < 0.001f)
          return false;
        if (std::abs(outParams.shearX) > 0.01f)
          return false;
        if (outParams.nearPlane <= 0.0f)
          return false;
        if (outParams.farPlane <= outParams.nearPlane)
          return false;

        return true;
      };

      auto matrixL1Error = [](const Matrix4& a, const Matrix4& b) {
        float err = 0.0f;
        for (uint32_t c = 0; c < 4; c++) {
          for (uint32_t r = 0; r < 4; r++) {
            err += std::abs(a[c][r] - b[c][r]);
          }
        }
        return err;
      };

      auto tryBuild = [&](const Matrix4& candidateWorldToProjection, Matrix4& outWorldToViewLocal, Matrix4& outViewToProjectionLocal) -> bool {
        // avoid attempting to invert singular matrices
        {
          constexpr double kDetEps = 1e-24;
          const double det = determinant(candidateWorldToProjection);
          if (!std::isfinite(det) || std::abs(det) <= kDetEps)
            return false;
        }

        // invert world to projection to unproject a few NDC points
        Matrix4 invWorldToProjection;
        invWorldToProjection = inverse(candidateWorldToProjection);

        auto unprojectNdc = [&](float ndcX, float ndcY, float ndcZ, Vector3& outWorldPos) -> bool {
          const Vector4 clip(ndcX, ndcY, ndcZ, 1.0f);
          const Vector4 worldH = invWorldToProjection * clip;
          if (!std::isfinite(worldH.w) || std::abs(worldH.w) < kEps)
            return false;
          const float invW = 1.0f / worldH.w;
          outWorldPos = worldH.xyz() * invW;
          return std::isfinite(outWorldPos.x) && std::isfinite(outWorldPos.y) && std::isfinite(outWorldPos.z);
        };

        // reference - D3D NDC: x/y in [-1, 1], z in [0, 1]
        constexpr float ndcZ = 0.5f;
        Vector3 worldCenter, worldUp, worldRight;
        if (!unprojectNdc(0.0f, 0.0f, ndcZ, worldCenter))
          return false;
        if (!unprojectNdc(0.0f, 1.0f, ndcZ, worldUp))
          return false;
        if (!unprojectNdc(1.0f, 0.0f, ndcZ, worldRight))
          return false;

        Vector3 forward = worldCenter - camPos;
        if (lengthSqr(forward) < kEps)
          return false;
        forward = normalize(forward);

        Vector3 upHint = worldUp - worldCenter;
        if (lengthSqr(upHint) < kEps)
          upHint = Vector3(0.0f, 1.0f, 0.0f);
        else
          upHint = normalize(upHint);

        // construct an orthonormal basis, we use the unprojected "up" direction as a hint to fix roll
        Vector3 right = cross(upHint, forward);
        if (lengthSqr(right) < kEps)
          return false;
        right = normalize(right);
        Vector3 up = normalize(cross(forward, right));

        auto buildViewToWorld = [&](const Vector3& r, const Vector3& u, const Vector3& f) {
          Matrix4 viewToWorld;
          viewToWorld[0] = Vector4(r.x, r.y, r.z, 0.0f);
          viewToWorld[1] = Vector4(u.x, u.y, u.z, 0.0f);
          viewToWorld[2] = Vector4(f.x, f.y, f.z, 0.0f);
          viewToWorld[3] = Vector4(camPos.x, camPos.y, camPos.z, 1.0f);
          return viewToWorld;
        };

        // 1st attempt - assume +Z is forward in view space but if the unprojected point ends up behind the camera, flip the forward axis and rebuild
        Matrix4 viewToWorld = buildViewToWorld(right, up, forward);
        Matrix4 worldToView = inverseAffine(viewToWorld);
        const Vector4 centerInView = worldToView * Vector4(worldCenter.x, worldCenter.y, worldCenter.z, 1.0f);
        if (std::isfinite(centerInView.z) && centerInView.z < 0.0f) {
          forward = -forward;
          right = cross(upHint, forward);
          if (lengthSqr(right) < kEps)
            return false;
          right = normalize(right);
          up = normalize(cross(forward, right));
          viewToWorld = buildViewToWorld(right, up, forward);
          worldToView = inverseAffine(viewToWorld);
        }

        // candidate projections - handle possible multiplication order conventions by picking the one that
        // yields a valid projection on decomposition and best reconstructs the original matrix
        Matrix4 bestViewToProjection;
        float bestErr = std::numeric_limits<float>::infinity();
        bool found = false;

        {
          Matrix4 viewToProjection = candidateWorldToProjection * viewToWorld;
          DecomposeProjectionParams projParams {};
          if (validateProjection(viewToProjection, projParams)) {
            const Matrix4 recon = viewToProjection * worldToView;
            const float err = matrixL1Error(recon, candidateWorldToProjection);
            if (err < bestErr) {
              bestErr = err;
              bestViewToProjection = viewToProjection;
              found = true;
            }
          }
        }

        {
          Matrix4 viewToProjection = viewToWorld * candidateWorldToProjection;
          DecomposeProjectionParams projParams {};
          if (validateProjection(viewToProjection, projParams)) {
            const Matrix4 recon = worldToView * viewToProjection;
            const float err = matrixL1Error(recon, candidateWorldToProjection);
            if (err < bestErr) {
              bestErr = err;
              bestViewToProjection = viewToProjection;
              found = true;
            }
          }
        }

        if (!found)
          return false;

        outWorldToViewLocal = worldToView;
        outViewToProjectionLocal = bestViewToProjection;
        return true;
      };

      // try both the raw matrix and its transpose, prefer the one that best reconstructs the original world to projection
      struct CandidateResult {
        bool ok = false;
        float error = std::numeric_limits<float>::infinity();
        Matrix4 worldToView;
        Matrix4 viewToProjection;
      };

      auto evalCandidate = [&](const Matrix4& candidateWorldToProjection) -> CandidateResult {
        CandidateResult r;
        Matrix4 w2v;
        Matrix4 v2p;
        if (!tryBuild(candidateWorldToProjection, w2v, v2p))
          return r;
        r.ok = true;
        r.worldToView = w2v;
        r.viewToProjection = v2p;
        r.error = matrixL1Error(v2p * w2v, candidateWorldToProjection);
        return r;
      };

      const CandidateResult raw = evalCandidate(worldToProjection);
      const CandidateResult transposed = evalCandidate(transpose(worldToProjection));

      const CandidateResult* best = nullptr;
      if (raw.ok && transposed.ok) {
        best = (raw.error <= transposed.error) ? &raw : &transposed;
      } else if (raw.ok) {
        best = &raw;
      } else if (transposed.ok) {
        best = &transposed;
      } else {
        return false;
      }

      outWorldToView = best->worldToView;
      outViewToProjection = best->viewToProjection;
      if (outUsedTranspose != nullptr)
        *outUsedTranspose = (best == &transposed);
      if (outReconstructionError != nullptr)
        *outReconstructionError = best->error;
      return true;
    }
  }

  D3D9Rtx::Ue3VertexFactoryType D3D9Rtx::classifyUe3VertexFactory(const D3D9VertexElements& elements) {
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

    // Instanced mesh particles share FoliageVertexFactory.usf and the instancing axes in
    // TEXCOORD1..4, but their declaration has no NORMAL: FParticleInstancedMeshVertexFactory
    // walks {VEU_Tangent, VEU_Binormal, VEU_Normal} while only filling components 0 and 1, so
    // the mesh's TangentZ arrives under the BINORMAL semantic. (The stock game's shader still
    // declares it as NORMAL and therefore never reads it - so UE3 renders these unlit.)
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

  D3D9Rtx::Ue3ShaderFeatureInfo D3D9Rtx::getUe3ShaderFeatureInfo(const D3D9CommonShader* shader) {
    Ue3ShaderFeatureInfo empty;
    empty.initialized = true;

    if (shader == nullptr)
      return empty;

    const auto& bytecode = shader->GetBytecode();
    const XXH64_hash_t shaderHash = shader->GetBytecodeHash();
    if (shaderHash == 0)
      return empty;

    auto it = m_ue3ShaderFeatureCache.find(shaderHash);
    if (it != m_ue3ShaderFeatureCache.end())
      return it->second;

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
        if (decoder.getCtabInfo().m_size != 0)
          break;
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

          info.hasSceneColorSampler |= containsToken(lowerName, "scenecolor");
          info.hasSceneDepthSampler |= containsToken(lowerName, "scenedepth") ||
                                       containsToken(lowerName, "destdepth") ||
                                       containsToken(lowerName, "pixeldepth");
          info.hasLightAttenuationSampler |= containsToken(lowerName, "lightattenuation");
          info.hasShadowSampler |= containsToken(lowerName, "shadowdepth") ||
                                   containsToken(lowerName, "shadowtexture") ||
                                   containsToken(lowerName, "shadowvariance");
          const bool velocitySampler = containsToken(lowerName, "velocitybuffer") ||
                                       containsToken(lowerName, "velocitytexture");
          info.hasVelocitySampler |= velocitySampler;
          if (velocitySampler && registerIndex < 0xFF)
            info.velocitySamplerIndex = uint8_t(registerIndex);
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
        info.hasBloomScale |= containsToken(lowerName, "bloomscale");

        if (isFloat4 && registerIndex <= 0x7FFF && lowerName == "gammacolorscaleandinverse") {
          info.toneMapGammaColorScaleReg = int16_t(registerIndex);
        }
      };

      for (const DxsoCtab::Constant& c : ctab.m_constantData) {
        const bool isSampler = c.registerSet == kD3dxRegisterSetSampler;
        const bool isFloat4 = c.registerSet == kD3dxRegisterSetFloat4;
        const std::string lowerName = toLowerAscii(c.name);
        markName(lowerName, isSampler, isFloat4, c.registerIndex);

        // Exact names, so UberPostProcessBlend's GammaColorScaleAndInverse/GammaOverlayColor
        // (a different shader, whose output the composite still has to gamma correct) cannot
        // be mistaken for the final composite's own constants.
        if (!isSampler) {
          if (lowerName == "inversegamma")
            info.gammaInverseReg = c.registerIndex;
          else if (lowerName == "colorscale")
            info.gammaColorScaleReg = c.registerIndex;
          else if (lowerName == "overlaycolor")
            info.gammaOverlayColorReg = c.registerIndex;
        }
      }
    } catch (...) {
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

  // UE3 instances a mesh by leaving its placement out of the shader constants entirely: the mesh
  // streams carry D3DSTREAMSOURCE_INDEXEDDATA | instanceCount, one further stream is tagged
  // D3DSTREAMSOURCE_INSTANCEDATA, and the vertex factory reads InstanceOffset (TEXCOORD1) plus the
  // three basis axes (TEXCOORD2..4) out of it. See FParticleInstancedMeshVertexFactory::InitRHI and
  // FFoliageVertexFactory::InitRHI in the UE3 engine, and GetInstanceToWorld in
  // FoliageVertexFactory.usf which both compile down to that layout.


  // Whatever these bounds drop, the kept set has to be the same set next frame: an instance that
  // comes and goes as the camera moves flickers, and a selection that reorders the survivors also
  // renames them (see Ue3DecomposedInstance::sourceIndex). Hence the count clamp keeps the lowest
  // source indices rather than the nearest to the camera. Distance culling is view-dependent by
  // definition and can pop at its boundary, which is why it is opt-in.

  // Names the batch an instanced draw belongs to, without involving any transform.
  //
  // The mesh streams say which mesh is being instanced but not which component is instancing it, and
  // two piles of the same debris share them. The instance buffer would distinguish those but cannot
  // serve as an identity - RenderNxFluidInstanced creates a fresh one per frame unless its pool hands
  // one back - so batches are matched to the previous frame's by continuity of their own centroid,
  // which holds because a batch as a whole barely moves even while its instances do.



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

  D3D9Rtx::Ue3PassType D3D9Rtx::classifyUe3Pass(const DrawContext& drawContext) {
    if (!m_frameOptions.ue3EngineMode)
      return Ue3PassType::Unknown;

    const bool depthEnabled = d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_TRUE;
    const bool zWriteEnabled = d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE;
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
    if (m_currentUe3VertexFactory == Ue3VertexFactoryType::PositionOnly)
      return Ue3PassType::DepthPrepass;

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

    if (psInfo.hasUiSampler || psInfo.hasUiCompositeConstants)
      return Ue3PassType::UiComposite;

    // UE3 SceneCapture probes re-render the world before the main view; viewport size/aspect
    // (and later mirrored/undecomposable camera checks) keep them from stealing Main.
    if ((m_frameOptions.ue3SkipSceneCapturePasses || m_frameOptions.ue3EngineMode) &&
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
      d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] != FALSE &&
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


  D3D9Rtx::D3D9Rtx(D3D9DeviceEx* d3d9Device)
    : m_parent(d3d9Device) {
  }

  D3D9Rtx::~D3D9Rtx() {
    restoreNgxGameSettingsRedirects();

    // Owned bridge parent handle only; GetCurrentProcess() is a pseudo-handle.
    if (m_ngxGameProcessOwned && m_ngxGameProcess != nullptr)
      ::CloseHandle(m_ngxGameProcess);
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
    o.ue3EngineMode = ue3EngineModeObject().get();
    o.ue3SkipSceneCapturePasses = ue3SkipSceneCapturePassesObject().get();
    o.sequenceTrackedLockWaits = sequenceTrackedLockWaitsObject().get();
    o.ngxPassthroughJitter = RtxNgxPassthrough::enableJitter();
    o.ngxPrePostProcess = RtxNgxPassthrough::prePostProcess();
    o.ngxDlfgHudless = RtxNgxPassthrough::dlfgHudlessInput() && DxvkDLFG::enable() &&
                       m_parent->GetDXVKDevice()->getCommon()->metaNGXContext().supportsDLFG();
    o.ngxDlfgHudlessPreview = RtxNgxPassthrough::dlfgHudlessPreview();
    o.ngxNeuralRendering = DlssNeuralRendering::enable() &&
                           m_parent->GetDXVKDevice()->getCommon()->metaDlssNeuralRendering().supportsDlssNeuralRendering();
    o.ngxObjectVelocities = RtxNgxPassthrough::objectVelocities();
    o.ngxDebugVisualization = RtxNgxPassthrough::debugVisualization();

    // On-demand post-chain dump (one-shot: consumed here, option resets itself)
    if (RtxNgxPassthrough::dumpPostChainFrames() > 0 && m_ngxPostChainDumpFramesLeft == 0) {
      m_ngxPostChainDumpFramesLeft = uint32_t(RtxNgxPassthrough::dumpPostChainFrames());
      RtxNgxPassthrough::dumpPostChainFramesObject().setDeferred(0);
      Logger::info(str::format("[RTX NGX Passthrough][dump] On-demand dump armed for ", m_ngxPostChainDumpFramesLeft, " frames."));
    }

    if (RtxNgxPassthrough::dumpVelocityCaptureFrames() > 0 && m_ngxVelocityDumpFramesLeft == 0) {
      m_ngxVelocityDumpFramesLeft = uint32_t(RtxNgxPassthrough::dumpVelocityCaptureFrames());
      RtxNgxPassthrough::dumpVelocityCaptureFramesObject().setDeferred(0);
      Logger::info(str::format("[RTX NGX Passthrough][velocity dump] Armed for ",
                               m_ngxVelocityDumpFramesLeft, " frames."));
    }
    o.uiTextures = &RtxOptions::uiTexturesObject().get();

    o.valid = true;
  }







  bool D3D9Rtx::tryGetUe3CameraFromConstantsCached(uint32_t viewProjReg,
                                                   uint32_t viewOriginReg,
                                                   Matrix4& outWorldToView,
                                                   Matrix4& outViewToProjection,
                                                   bool& outUsedTranspose,
                                                   float& outReconstructionError,
                                                   XXH64_hash_t* outConstantsHash) {
    if (viewProjReg + 3 >= caps::MaxFloatConstantsSoftware || viewOriginReg >= caps::MaxFloatConstantsSoftware)
      return false;

    // cache by raw constant values to avoid repeated heavy extraction work per draw call
    struct Ue3CameraConstsKey {
      uint32_t viewProjReg;
      uint32_t viewOriginReg;
      Vector4 regs[5];
    };

    Ue3CameraConstsKey key {};
    key.viewProjReg = viewProjReg;
    key.viewOriginReg = viewOriginReg;
    key.regs[0] = d3d9State().vsConsts.fConsts[viewProjReg + 0];
    key.regs[1] = d3d9State().vsConsts.fConsts[viewProjReg + 1];
    key.regs[2] = d3d9State().vsConsts.fConsts[viewProjReg + 2];
    key.regs[3] = d3d9State().vsConsts.fConsts[viewProjReg + 3];
    key.regs[4] = d3d9State().vsConsts.fConsts[viewOriginReg];

    const XXH64_hash_t constantsHash = XXH3_64bits(&key, sizeof(key));

    if (outConstantsHash != nullptr) {
      *outConstantsHash = constantsHash;
    }

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
  }

  Matrix4 D3D9Rtx::extractUe3ObjectToWorld(uint32_t reg, bool hasWorldToLocal, uint32_t w2lReg, bool cameraUsedTranspose) {
    // Memo lookup: every input to the transpose/affinity/inverse disambiguation below
    // (register contents and the camera transpose convention tiebreaker) is folded into
    // the key, so a hit returns exactly what the computation would produce. Static
    // placements re-upload identical matrices every frame, making this a per-draw
    // matrix-inverse saving.
    XXH64_hash_t o2wKeyHash = XXH3_64bits(&d3d9State().vsConsts.fConsts[reg], 4 * sizeof(Vector4));
    if (hasWorldToLocal) {
      o2wKeyHash = XXH3_64bits_withSeed(&d3d9State().vsConsts.fConsts[w2lReg], 3 * sizeof(Vector4), o2wKeyHash);
    }
    const uint32_t o2wKeyFlags = (hasWorldToLocal ? 1u : 0u) | (cameraUsedTranspose ? 2u : 0u);
    o2wKeyHash = XXH3_64bits_withSeed(&o2wKeyFlags, sizeof(o2wKeyFlags), o2wKeyHash);

    const auto o2wIt = m_ue3ObjectToWorldCache.find(o2wKeyHash);
    if (o2wIt != m_ue3ObjectToWorldCache.end()) {
      return o2wIt->second;
    }

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

    // optionally use WorldToLocal (if present) to disambiguate transpose/packing
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
        localToWorld = cameraUsedTranspose ? localToWorldTransposed : localToWorldRaw;
      }
    } else if (rawAffine && transAffine) {
      localToWorld = cameraUsedTranspose ? localToWorldTransposed : localToWorldRaw;
    } else if (!rawAffine && transAffine) {
      localToWorld = localToWorldTransposed;
    } else {
      localToWorld = localToWorldRaw;
    }

    if (m_ue3ObjectToWorldCache.size() >= kUe3ObjectToWorldCacheMaxEntries) {
      m_ue3ObjectToWorldCache.clear();
    }
    m_ue3ObjectToWorldCache.emplace(o2wKeyHash, localToWorld);

    return localToWorld;
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

      DxvkImage* const image = texture->GetImage().ptr();
      m_boundTextureSnapshot.imageHashes[idx] = image != nullptr ? image->getHash() : kEmptyHash;
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
      const XXH64_hash_t texHash = boundTextures.imageHashes[idx];
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



  void D3D9Rtx::triggerInjectRTX() {
    // Flush any pending game and RTX work
    m_parent->Flush();

    // Send command to inject RTX. Pass the backbuffer when known: the CS thread's bound render
    // target is often the scene color surface at the injection trigger, and falling back to
    // that would upscale the wrong image while the presented backbuffer stays native.
    m_parent->EmitCs([cReflexFrameId = GetReflexFrameId(), cTargetImage = m_ngxFrameBackbufferImage](DxvkContext* ctx) {
      if (cTargetImage != nullptr) {
        static_cast<RtxContext*>(ctx)->injectRTX(cReflexFrameId, cTargetImage);
      } else {
        static_cast<RtxContext*>(ctx)->injectRTX(cReflexFrameId);
      }
    });
  }

  D3D9Rtx::NgxCameraCtabRegs D3D9Rtx::scanNgxCameraCtabRegs(const std::vector<uint8_t>& bytecode) const {
    // Compact variant of the UE3 CTAB parse used by the ray traced path: only the two camera
    // symbols matter here, and only shaders declaring BOTH may steer the main camera
    // (fallback-register extraction can pick up light-space matrices from utility shaders).
    NgxCameraCtabRegs result;

    try {
      if (bytecode.size() < sizeof(uint32_t) || (bytecode.size() % sizeof(uint32_t)) != 0)
        return result;

      const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
      const uint32_t headerToken = tokens[0];
      const uint32_t headerTypeMask = headerToken & 0xffff0000u;

      DxsoProgramType programType;
      if (headerTypeMask == 0xffff0000u)
        programType = DxsoProgramTypes::PixelShader;
      else if (headerTypeMask == 0xfffe0000u)
        programType = DxsoProgramTypes::VertexShader;
      else
        return result;

      const uint32_t majorVersion = (headerToken >> 8) & 0xffu;
      const uint32_t minorVersion = headerToken & 0xffu;
      DxsoProgramInfo programInfo { programType, minorVersion, majorVersion };

      DxsoDecodeContext decoder(programInfo);
      DxsoCodeIter iter(tokens + 1);

      // Full instruction walk (the CTAB comment arrives early; the rest determines the
      // GPU skin replay parameters): a mul/mad reading the BLENDINDICES input means the
      // shader scales raw bone indices by the 3-registers-per-bone stride itself; without
      // one the vertex data is pre-scaled and feeds the address register directly. The
      // BLENDWEIGHT swizzles reveal how many influences the shader variant blends.
      uint32_t blendIndicesInputRegister = UINT32_MAX;
      uint32_t blendWeightsInputRegister = UINT32_MAX;
      bool blendIndicesScaledInShader = false;
      bool blendIndicesRead = false;
      uint32_t weightComponentsRead = 0;  // bitmask of BLENDWEIGHT components consumed

      // The decode context reuses its source array across instructions, so only the
      // operands the opcode actually consumes may be inspected
      const auto sourceOperandCount = [](DxsoOpcode op) -> uint32_t {
        switch (op) {
          case DxsoOpcode::Mov:
          case DxsoOpcode::Rcp:
          case DxsoOpcode::Rsq:
          case DxsoOpcode::Exp:
          case DxsoOpcode::Log:
          case DxsoOpcode::Frc:
            return 1;
          case DxsoOpcode::Mad:
          case DxsoOpcode::Lrp:
            return 3;
          default:
            return 2;
        }
      };

      while (decoder.decodeInstruction(iter)) {
        const DxsoInstructionContext& instructionCtx = decoder.getInstructionContext();
        const DxsoOpcode opcode = instructionCtx.instruction.opcode;

        if (opcode == DxsoOpcode::Dcl &&
            instructionCtx.dst.id.type == DxsoRegisterType::Input &&
            instructionCtx.dcl.semantic.usageIndex == 0) {
          if (instructionCtx.dcl.semantic.usage == DxsoUsage::BlendIndices) {
            blendIndicesInputRegister = instructionCtx.dst.id.num;
          } else if (instructionCtx.dcl.semantic.usage == DxsoUsage::BlendWeight) {
            blendWeightsInputRegister = instructionCtx.dst.id.num;
          }
        }

        if ((blendIndicesInputRegister != UINT32_MAX || blendWeightsInputRegister != UINT32_MAX) &&
            opcode != DxsoOpcode::Dcl && opcode != DxsoOpcode::Def && opcode != DxsoOpcode::Comment) {
          const uint32_t sourceCount = std::min(sourceOperandCount(opcode), uint32_t(instructionCtx.src.size()));
          for (uint32_t s = 0; s < sourceCount; s++) {
            const DxsoRegister& source = instructionCtx.src[s];
            if (source.id.type != DxsoRegisterType::Input) {
              continue;
            }

            if (source.id.num == blendIndicesInputRegister) {
              blendIndicesRead = true;
              if (opcode == DxsoOpcode::Mul || opcode == DxsoOpcode::Mad) {
                blendIndicesScaledInShader = true;
              }
            }

            if (source.id.num == blendWeightsInputRegister) {
              for (uint32_t component = 0; component < 4; component++) {
                weightComponentsRead |= 1u << source.swizzle[component];
              }
            }
          }
        }
      }

      result.boneIndicesPreScaled = blendIndicesInputRegister != UINT32_MAX && !blendIndicesScaledInShader;

      // Influence count: distinct weight components consumed; indices without weights =
      // the rigid-skin variant (single bone, implicit weight 1)
      if (blendIndicesRead) {
        uint32_t weightCount = 0;
        for (uint32_t component = 0; component < 4; component++) {
          weightCount += (weightComponentsRead >> component) & 1u;
        }
        result.skinInfluenceCount = std::max(weightCount, 1u);
      }

      const DxsoCtab& ctab = decoder.getCtabInfo();
      if (ctab.m_size == 0 || ctab.m_constantData.empty())
        return result;

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

      bool hasViewProjectionMatrix = false;
      bool hasCameraPosition = false;
      uint32_t viewProjRegister = 0;
      uint32_t viewOriginRegister = 0;

      for (const DxsoCtab::Constant& c : ctab.m_constantData) {
        const std::string name = lower(c.name);

        if (!hasViewProjectionMatrix && c.registerCount >= 4) {
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
            hasViewProjectionMatrix = true;
            viewProjRegister = c.registerIndex;
          }
        }

        if (!hasCameraPosition && c.registerCount >= 1) {
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
            hasCameraPosition = true;
            viewOriginRegister = c.registerIndex;
          }
        }

        // Object velocity capture inputs: rigid LocalToWorld (4 registers), the optional
        // WorldToLocal basis (transpose disambiguation), and the skinning marker that
        // excludes a draw from rigid capture
        if (!result.hasLocalToWorld && c.registerCount >= 4 &&
            (contains(name, "localtoworld") || contains(name, "local_to_world")) &&
            !contains(name, "prev")) {
          result.hasLocalToWorld = true;
          result.localToWorldRegister = c.registerIndex;
        }

        if (!result.hasWorldToLocal && c.registerCount >= 3 &&
            (contains(name, "worldtolocal") || contains(name, "world_to_local"))) {
          result.hasWorldToLocal = true;
          result.worldToLocalRegister = c.registerIndex;
        }

        if (!result.hasBoneMatrices && c.registerCount >= 3 && contains(name, "bone")) {
          result.hasBoneMatrices = true;
          result.boneMatricesRegister = c.registerIndex;
          result.boneMatricesRegisterCount = c.registerCount;
        }
      }

      result.hasViewProjection = hasViewProjectionMatrix;
      if (hasViewProjectionMatrix) {
        result.viewProjRegister = viewProjRegister;
      }

      if (hasViewProjectionMatrix && hasCameraPosition) {
        result.ctabVerified = true;
        result.viewOriginRegister = viewOriginRegister;
      }
    } catch (...) {
      return result;
    }

    return result;
  }

  void D3D9Rtx::rememberNgxReducedSceneDraw(uint32_t primitiveCount, XXH64_hash_t vertexShaderHash) {
    for (const NgxReducedSceneDraw& seen : m_ngxReducedSceneDraws) {
      if (seen.primitiveCount == primitiveCount && seen.vertexShaderHash == vertexShaderHash)
        return;
    }

    m_ngxReducedSceneDraws.push_back({ primitiveCount, vertexShaderHash });
  }

  bool D3D9Rtx::ngxFullSurfaceDrawReplaysReducedScene(uint32_t primitiveCount, XXH64_hash_t vertexShaderHash) const {
    if (!m_ngxReducedSceneViewportThisFrame || m_ngxSceneColorImage == nullptr)
      return false;

    if (d3d9State().renderTargets[kRenderTargetIndex] == nullptr)
      return false;

    D3D9CommonTexture* renderTargetTexture = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
    if (renderTargetTexture == nullptr || renderTargetTexture->GetImage() == nullptr)
      return false;

    if (renderTargetTexture->GetImage().ptr() != m_ngxSceneColorImage.ptr())
      return false;

    for (const NgxReducedSceneDraw& seen : m_ngxReducedSceneDraws) {
      if (seen.primitiveCount == primitiveCount && seen.vertexShaderHash == vertexShaderHash)
        return true;
    }

    return false;
  }

  void D3D9Rtx::tryNgxPassthroughCameraCapture(uint32_t primitiveCount) {
    const D3D9CommonShader* vertexShaderCommon = d3d9State().vertexShader->GetCommonShader();
    if (vertexShaderCommon == nullptr)
      return;

    const XXH64_hash_t shaderHash = vertexShaderCommon->GetBytecodeHash();
    if (shaderHash == 0)
      return;

    auto it = m_ngxCameraCtabCache.find(shaderHash);
    if (it == m_ngxCameraCtabCache.end()) {
      it = m_ngxCameraCtabCache.emplace(shaderHash, scanNgxCameraCtabRegs(vertexShaderCommon->GetBytecode())).first;
    }

    const NgxCameraCtabRegs& ctabRegs = it->second;
    if (!ctabRegs.ctabVerified)
      return;

    const uint32_t viewProjReg = ctabRegs.viewProjRegister;
    const uint32_t viewOriginReg = ctabRegs.viewOriginRegister;

    if (viewProjReg + 3 >= caps::MaxFloatConstantsSoftware || viewOriginReg >= caps::MaxFloatConstantsSoftware)
      return;

    // Mirrored view rejection (SceneCapture reflection/portal probes premultiply a mirror
    // matrix into the view, flipping the 3x3 determinant sign; the main view is always
    // positive). Checked on the raw registers - the sign is transpose-invariant.
    {
      const Vector4& vpRow0 = d3d9State().vsConsts.fConsts[viewProjReg + 0];
      const Vector4& vpRow1 = d3d9State().vsConsts.fConsts[viewProjReg + 1];
      const Vector4& vpRow2 = d3d9State().vsConsts.fConsts[viewProjReg + 2];
      const float vpDet3 =
        vpRow0.x * (vpRow1.y * vpRow2.z - vpRow1.z * vpRow2.y) -
        vpRow0.y * (vpRow1.x * vpRow2.z - vpRow1.z * vpRow2.x) +
        vpRow0.z * (vpRow1.x * vpRow2.y - vpRow1.y * vpRow2.x);
      if (!std::isfinite(vpDet3) || vpDet3 < 0.0f)
        return;
    }

    // Only the main scene view may steer the main camera and scene targets.
    // It renders at backbuffer * ScreenPercentage / 100, or the full backbuffer at 100.
    // Shadow cascades, scene captures, and velocity passes use other sizes.
    // A low ScreenPercentage view is as small as those passes, so the match uses the known percentage.
    if (m_activePresentParams.has_value()) {
      const D3DVIEWPORT9& vp = d3d9State().viewport;
      const uint32_t bbW = m_activePresentParams->BackBufferWidth;
      const uint32_t bbH = m_activePresentParams->BackBufferHeight;

      if (bbW != 0 && bbH != 0) {
        const float sp = m_ngxGameScreenPercentage;
        if (sp > 0.0f && sp <= 100.0f) {
          const bool viewportNearFullSize =
            int32_t(vp.Width) >= int32_t(bbW) - kNgxDrivenViewportTolerance &&
            int32_t(vp.Height) >= int32_t(bbH) - kNgxDrivenViewportTolerance;
          const bool viewportMatchesExpected =
            ngxViewportMatchesDrivenScreenPercentage(vp) ||
            (sp >= 99.5f && viewportNearFullSize);
          if (!viewportMatchesExpected) {
            // The ScreenPercentage shadow can move a frame before the game consumes it.
            // A full-backbuffer view is accepted for that frame so the camera stays valid.
            // After the viewport shrinks, only the reduced rect passes.
            const bool expectingReducedResolution = sp < 99.5f;
            if (!(expectingReducedResolution && viewportNearFullSize))
              return;

            // Rebinding scene colour leaves the viewport at the whole target.
            // A repeat of an accepted draw leaves the recorded rect in place.
            // Any other full-surface view replaces the rect.
            if (ngxFullSurfaceDrawReplaysReducedScene(primitiveCount, shaderHash)) {
              ONCE(Logger::info("[RTX NGX Passthrough] Kept the ScreenPercentage scene rect. A later full-target draw was not the view."));
              return;
            }
          }
        } else if (vp.Width * 2 < bbW || vp.Height * 2 < bbH) {
          // ScreenPercentage not resolved yet (first frame): fall back to the near-backbuffer
          // heuristic, safe for >= 50% and self-correcting once it resolves.
          return;
        }
      }
    }

    Matrix4 worldToView;
    Matrix4 viewToProjection;
    bool usedTranspose = false;
    float reconstructionError = 0.0f;
    XXH64_hash_t constantsHash = 0;

    if (!tryGetUe3CameraFromConstantsCached(viewProjReg, viewOriginReg, worldToView, viewToProjection,
                                            usedTranspose, reconstructionError, &constantsHash))
      return;

    // Record the scene color/depth targets from depth-writing scene draws: the depth image
    // feeds the motion vector pass, the color image anchors the viewport jitter scope, the
    // resolve tracking, and the pre-post-process trigger
    if (d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE &&
        d3d9State().renderTargets[kRenderTargetIndex] != nullptr &&
        d3d9State().depthStencil != nullptr) {
      D3D9CommonTexture* renderTargetTexture = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
      D3D9CommonTexture* depthStencilTexture = d3d9State().depthStencil->GetCommonTexture();

      if (renderTargetTexture != nullptr && depthStencilTexture != nullptr &&
          renderTargetTexture->GetImage() != nullptr && depthStencilTexture->GetImage() != nullptr) {
        // The frame's accepted camera (validity gate for the velocity capture, transpose
        // flag for LocalToWorld disambiguation). Only depth-writing draws with bound
        // color+depth targets donate it: the velocity raster depth-tests against the depth
        // these very draws produce, so their constants are by construction the camera that
        // buffer was rasterized with. Draws outside the scene (utility passes early in the
        // frame) may carry stale view constants from the previous frame.
        if (!m_ngxFrameCameraValid) {
          m_ngxFrameCameraValid = true;
          m_ngxFrameCameraUsedTranspose = usedTranspose;
        }

        const bool sceneTargetsChanged = m_ngxSceneColorImage != renderTargetTexture->GetImage();

        m_ngxSceneColorImage = renderTargetTexture->GetImage();
        m_ngxSceneDepthImage = depthStencilTexture->GetImage();
        m_ngxSceneTargetsLastSeenFrame = m_ue3FrameCounter;

        // The viewport of the scene draws is the subrect the game renders into; when the
        // game runs with a reduced ScreenPercentage this is smaller than the backbuffer
        m_ngxSceneViewport = d3d9State().viewport;
        m_ngxSceneViewportValid = true;
        m_ngxSceneViewportLastValidFrame = m_ue3FrameCounter;
        if (m_ngxGameScreenPercentage < 99.5f &&
            ngxViewportMatchesDrivenScreenPercentage(m_ngxSceneViewport)) {
          m_ngxReducedSceneViewportThisFrame = true;
          rememberNgxReducedSceneDraw(primitiveCount, shaderHash);
        }

        if (sceneTargetsChanged) {
          ONCE(Logger::info(str::format("[RTX NGX Passthrough] Scene color/depth targets identified. color=0x",
                                        std::hex, uintptr_t(m_ngxSceneColorImage.ptr()),
                                        " depth=0x", uintptr_t(m_ngxSceneDepthImage.ptr()), std::dec,
                                        " extent=", m_ngxSceneColorImage->info().extent.width, "x",
                                        m_ngxSceneColorImage->info().extent.height,
                                        " format=", int(m_ngxSceneColorImage->info().format))));
          // Automatic dump on target changes (level transitions): validates the pre-post
          // injection point against the game's compositing without user action. Budgeted, because
          // a title that alternates between two scene color surfaces changes target every frame
          // and would re-arm this forever - Mirror's Edge does exactly that.
          if (m_frameOptions.ngxPrePostProcess && m_ngxAutoDumpArmsRemaining > 0) {
            m_ngxAutoDumpArmsRemaining--;
            m_ngxPostChainDumpFramesLeft = 2;
          }
          // Rebind the viewport so this draw already gets the sub-pixel jitter
          m_parent->m_flags.set(D3D9DeviceFlag::DirtyViewportScissor);
        }
      }
    }

    // Feed the main camera once per unique constants; RtCamera keeps the first update per frame
    if (constantsHash == m_ngxLastCameraConstantsHash)
      return;
    m_ngxLastCameraConstantsHash = constantsHash;

    ONCE(Logger::info(str::format("[RTX NGX Passthrough] UE3 camera captured from shader constants (viewProjReg=c",
                                  viewProjReg, "..c", viewProjReg + 3, ", viewOriginReg=c", viewOriginReg, ").")));

    m_ngxFrameWorldToView = worldToView;
    m_ngxFrameViewToProjection = viewToProjection;
    m_ngxFrameCameraMatricesValid = true;

    m_parent->EmitCs([cWorldToView = worldToView, cViewToProjection = viewToProjection](DxvkContext* ctx) {
      static_cast<RtxContext*>(ctx)->getSceneManager().getCameraManager().processExternalCamera(
        CameraType::Main, cWorldToView, cViewToProjection);
    });
  }

  void D3D9Rtx::tryCaptureNgxVelocityDraw(const DrawContext& drawContext) {
    // Dynamic object draws for the velocity raster: indexed triangle lists into the scene
    // color with depth writes and a known camera, in both scene phases (world/intermediate
    // and, after the mid-scene depth clear, the foreground DPG). Rigid draws are captured
    // when their LocalToWorld moved believably since the previous frame; skinned draws
    // (UE3 GPU skin) additionally when their bone palette animated (see the emission
    // gates below). Sightings are matched against per-identity instance history.
    constexpr size_t kMaxVelocityDrawsPerFrame = 384;

    if (!m_frameOptions.ngxObjectVelocities) {
      return;
    }

    if (!drawContext.Indexed || drawContext.PrimitiveType != D3DPT_TRIANGLELIST || drawContext.PrimitiveCount == 0) {
      return;
    }

    if (m_ngxSceneColorImage == nullptr || d3d9State().renderTargets[kRenderTargetIndex] == nullptr) {
      return;
    }

    D3D9CommonTexture* renderTargetTexture = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
    if (renderTargetTexture == nullptr || renderTargetTexture->GetImage().ptr() != m_ngxSceneColorImage.ptr()) {
      return;
    }

    // A draw into the scene color through a viewport that cannot hold the scene is not the visible
    // scene. Some titles run proxy passes over the same geometry into that target through a
    // degenerate viewport, and those satisfy every other gate here. Capturing them costs twice
    // over: the sighting claims the identity's history before the real draw arrives, so the real
    // one finds nothing from the previous frame to pair with and loses its velocity, and whatever
    // velocity the proxy does emit carries that pass's transforms into the raster and lands on
    // unrelated pixels.
    if (m_activePresentParams.has_value()) {
      const uint32_t backBufferWidth = m_activePresentParams->BackBufferWidth;
      const uint32_t backBufferHeight = m_activePresentParams->BackBufferHeight;
      const D3DVIEWPORT9& vp = d3d9State().viewport;

      // A quarter of the backbuffer in each axis: below the most aggressive ScreenPercentage the
      // scene is ever driven to, and far above a proxy or probe pass.
      if (uint64_t(vp.Width) * 4 < backBufferWidth || uint64_t(vp.Height) * 4 < backBufferHeight) {
        ONCE(Logger::info(str::format(
          "[RTX NGX Passthrough] Ignoring scene color draws through a ", vp.Width, "x", vp.Height,
          " viewport for object velocity; the scene cannot be rendered through it.")));
        return;
      }
    }

    // Depth-tested draws only; depth WRITES are re-checked after classification (the
    // CPU-modified meshes' shaded pass composites without z-writes during motion blur,
    // with their depth coming from a prepass - the raster's two-sided depth test anchors
    // visibility either way). Scene draws with the CPU-modified-mesh buffer signature
    // (large dedicated dynamic VB) are counted when rejected here: a nonzero counter
    // means such meshes render z-test-disabled in some game state.
    if (d3d9State().renderStates[D3DRS_ZENABLE] != D3DZB_TRUE) {
      D3D9CommonBuffer* zGateVertexBuffer = GetCommonBuffer(d3d9State().vertexBuffers[0].vertexBuffer);
      D3D9CommonBuffer* zGateIndexBuffer = GetCommonBuffer(d3d9State().indices);
      if (zGateVertexBuffer != nullptr && zGateIndexBuffer != nullptr &&
          (zGateVertexBuffer->Desc()->Usage & D3DUSAGE_DYNAMIC) != 0 &&
          (zGateIndexBuffer->Desc()->Usage & D3DUSAGE_DYNAMIC) == 0 &&
          d3d9State().vertexBuffers[0].offset == 0 &&
          zGateVertexBuffer->Desc()->Size > 100000) {
        m_ngxVelocityStats.skippedZDisabled++;
      }
      return;
    }
    const bool zWriteEnabled = d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE;

    // Draws after the game's mid-scene depth clear are the UE3 foreground DPG (first
    // person meshes and their attachments); they depth-test against the live foreground
    // depth and carry the foreground phase marker in the velocity raster
    const bool foregroundPhase = m_ngxDepthSnapshotTakenThisFrame;

    // The remaining gates run after draw qualification so their skip counters describe
    // scene draws that would otherwise have been considered
    if (m_ngxVelocityDraws.size() >= kMaxVelocityDrawsPerFrame) {
      m_ngxVelocityStats.skippedBudget++;
      return;
    }

    if (!m_ngxFrameCameraValid || !m_ngxPrevCameraValid) {
      m_ngxVelocityStats.skippedNoCamera++;
      return;
    }

    if (!m_parent->UseProgrammableVS() || d3d9State().vertexShader.ptr() == nullptr || d3d9State().vertexDecl == nullptr) {
      return;
    }

    const D3D9CommonShader* vertexShaderCommon = d3d9State().vertexShader->GetCommonShader();
    if (vertexShaderCommon == nullptr) {
      return;
    }

    const XXH64_hash_t shaderHash = vertexShaderCommon->GetBytecodeHash();
    if (shaderHash == 0) {
      return;
    }

    auto ctabIt = m_ngxCameraCtabCache.find(shaderHash);
    if (ctabIt == m_ngxCameraCtabCache.end()) {
      ctabIt = m_ngxCameraCtabCache.emplace(shaderHash, scanNgxCameraCtabRegs(vertexShaderCommon->GetBytecode())).first;
    }

    const NgxCameraCtabRegs& ctabRegs = ctabIt->second;

    // Buffers and the CPU-modified-mesh shape, established early: the gate relaxations
    // below depend on them
    D3D9CommonBuffer* vertexBufferCommon = GetCommonBuffer(d3d9State().vertexBuffers[0].vertexBuffer);
    D3D9CommonBuffer* indexBufferCommon = GetCommonBuffer(d3d9State().indices);
    if (vertexBufferCommon == nullptr || indexBufferCommon == nullptr) {
      return;
    }

    const bool vbDynamic = (vertexBufferCommon->Desc()->Usage & D3DUSAGE_DYNAMIC) != 0;
    const bool ibDynamic = (indexBufferCommon->Desc()->Usage & D3DUSAGE_DYNAMIC) != 0;
    // UE3 gives a CPU-modified mesh a dedicated dynamic vertex buffer, and a dedicated dynamic
    // index buffer as well wherever cloth tearing is enabled, since torn triangles need new
    // indices. Requiring a static index buffer therefore rejects exactly the meshes whose motion
    // this path exists to carry. A dedicated buffer is told from a shared ring pool by the draw
    // covering a large part of it: a ring-pool draw takes a small slice of a big buffer, and it is
    // the shifting allocations within one that make that geometry untrackable to begin with.
    const uint32_t indexStride =
      static_cast<D3D9Format>(indexBufferCommon->Desc()->Format) == D3D9Format::INDEX32 ? 4u : 2u;
    const uint64_t drawIndexBytes = uint64_t(drawContext.PrimitiveCount) * 3u * indexStride;
    const bool indexBufferDedicated =
      !ibDynamic || drawIndexBytes * 4 >= uint64_t(indexBufferCommon->Desc()->Size);

    const bool dynamicMeshShape = vbDynamic && indexBufferDedicated &&
                                  !ctabRegs.hasBoneMatrices &&
                                  d3d9State().vertexBuffers[0].offset == 0;

    if (!ctabRegs.hasLocalToWorld) {
      return;
    }

    // The full camera verification (ViewProjection + CameraPosition) guards camera
    // steering; the velocity capture itself only needs the per-draw ViewProjection and
    // LocalToWorld. CPU-modified-mesh-shaped draws are accepted on that weaker contract
    // (their motion-blur shader variants may lack the camera position symbol).
    if (!ctabRegs.ctabVerified && !(dynamicMeshShape && ctabRegs.hasViewProjection)) {
      return;
    }

    if (ctabRegs.viewProjRegister + 3 >= caps::MaxFloatConstantsSoftware) {
      return;
    }

    // Skinned draws (UE3 GPU skin): motion is bone palette + rigid transform combined.
    // The velocity raster replays the skinning with both frames' palettes.
    const bool skinned = ctabRegs.hasBoneMatrices;

    if (skinned) {
      if (ctabRegs.boneMatricesRegisterCount == 0 ||
          ctabRegs.boneMatricesRegisterCount > kNgxVelocityBonePaletteRegisters ||
          ctabRegs.boneMatricesRegister + ctabRegs.boneMatricesRegisterCount > caps::MaxFloatConstantsSoftware) {
        // The palette cap follows stock UE3's 75 bones per chunk, but a licensee that needed a
        // richer rig may have raised it, and a mesh past the cap is dropped here while simpler
        // ones around it are captured - one character without velocity, everything else with.
        m_ngxVelocityStats.skippedBonePalette++;

        ONCE(Logger::info(str::format(
          "[RTX NGX Passthrough] Velocity capture rejecting a skinned draw over the bone palette "
          "cap: ", ctabRegs.boneMatricesRegisterCount, " registers (",
          ctabRegs.boneMatricesRegisterCount / 3, " bones) at register ",
          ctabRegs.boneMatricesRegister, ", cap ", kNgxVelocityBonePaletteRegisters, " (",
          kNgxVelocityBonePaletteRegisters / 3, " bones).")));
        return;
      }
      if (m_ngxVelocitySkinnedDraws >= kNgxVelocityMaxSkinnedDraws) {
        m_ngxVelocityStats.skippedBudget++;
        return;
      }
    }

    const uint32_t l2wReg = ctabRegs.localToWorldRegister;
    if (l2wReg + 3 >= caps::MaxFloatConstantsSoftware) {
      return;
    }

    const bool hasWorldToLocal = ctabRegs.hasWorldToLocal &&
                                 ctabRegs.worldToLocalRegister + 2 < caps::MaxFloatConstantsSoftware;

    // The packed ViewProjection at THIS draw (oriented): scene phases render with their
    // own projections (the foreground DPG uses a first-person FOV), so every velocity
    // draw composes with its draw-time matrix - the game's exact vertex transform.
    Matrix4 drawWorldToProjection;
    drawWorldToProjection[0] = d3d9State().vsConsts.fConsts[ctabRegs.viewProjRegister + 0];
    drawWorldToProjection[1] = d3d9State().vsConsts.fConsts[ctabRegs.viewProjRegister + 1];
    drawWorldToProjection[2] = d3d9State().vsConsts.fConsts[ctabRegs.viewProjRegister + 2];
    drawWorldToProjection[3] = d3d9State().vsConsts.fConsts[ctabRegs.viewProjRegister + 3];
    if (m_ngxFrameCameraUsedTranspose) {
      drawWorldToProjection = transpose(drawWorldToProjection);
    }

    // On-demand per-draw record of everything the gates below decide on, so a draw whose velocity
    // never reaches the screen can be told from one whose does. Armed by the user while the object
    // in question is on screen, because a one-shot log fires during startup instead and describes
    // menu frames.
    const bool dumpThisDraw = [&] {
      if (likely(m_ngxVelocityDumpFramesLeft == 0)) {
        return false;
      }
      if (skinned) {
        return m_ngxVelocityDumpSkinnedThisFrame++ < kNgxVelocityDumpMaxSkinnedPerFrame;
      }
      if (vbDynamic || ibDynamic) {
        return m_ngxVelocityDumpDynamicThisFrame++ < kNgxVelocityDumpMaxDynamicPerFrame;
      }
      return m_ngxVelocityDumpRigidThisFrame++ < kNgxVelocityDumpMaxRigidPerFrame;
    }();

    if (unlikely(dumpThisDraw)) {
      const D3DVIEWPORT9& vp = d3d9State().viewport;
      Logger::info(str::format(
        "[RTX NGX Passthrough][velocity dump] draw=", m_drawCallID,
        " prims=", drawContext.PrimitiveCount,
        " skinned=", skinned ? 1 : 0,
        " boneRegs=", skinned ? ctabRegs.boneMatricesRegisterCount : 0,
        " vbDynamic=", vbDynamic ? 1 : 0, " ibDynamic=", ibDynamic ? 1 : 0,
        " vbBytes=", vertexBufferCommon->Desc()->Size,
        " streamOffset=", d3d9State().vertexBuffers[0].offset,
        " zwrite=", d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE ? 1 : 0,
        " zfunc=", int(d3d9State().renderStates[D3DRS_ZFUNC]),
        " vp=", vp.X, ",", vp.Y, " ", vp.Width, "x", vp.Height,
        " z=", vp.MinZ, "..", vp.MaxZ,
        " sceneVp=", m_ngxSceneViewport.X, ",", m_ngxSceneViewport.Y, " ",
        m_ngxSceneViewport.Width, "x", m_ngxSceneViewport.Height));
    }

    // Vertex layout on stream 0: position always; skinned draws additionally need the
    // UE3 GPU skin blend attributes (UBYTE4 indices, UBYTE4N weights)
    const auto& vertexElements = d3d9State().vertexDecl->GetElements();
    uint32_t positionOffset = 0;
    VkFormat positionFormat = VK_FORMAT_UNDEFINED;
    uint32_t blendIndicesOffset = 0;
    uint32_t blendWeightsOffset = 0;
    bool hasBlendIndices = false;
    bool hasBlendWeights = false;

    for (const D3DVERTEXELEMENT9& element : vertexElements) {
      if (element.Stream != 0 || element.UsageIndex != 0) {
        continue;
      }

      if (element.Usage == D3DDECLUSAGE_POSITION) {
        if (element.Type == D3DDECLTYPE_FLOAT3) {
          positionFormat = VK_FORMAT_R32G32B32_SFLOAT;
        } else if (element.Type == D3DDECLTYPE_FLOAT4) {
          positionFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
        }
        positionOffset = element.Offset;
      } else if (element.Usage == D3DDECLUSAGE_BLENDINDICES && element.Type == D3DDECLTYPE_UBYTE4) {
        hasBlendIndices = true;
        blendIndicesOffset = element.Offset;
      } else if (element.Usage == D3DDECLUSAGE_BLENDWEIGHT && element.Type == D3DDECLTYPE_UBYTE4N) {
        hasBlendWeights = true;
        blendWeightsOffset = element.Offset;
      }
    }

    if (positionFormat == VK_FORMAT_UNDEFINED) {
      return;
    }

    if (skinned && (!hasBlendIndices || !hasBlendWeights)) {
      ONCE(Logger::info("[RTX NGX Passthrough] Skinned draw with an unsupported blend attribute layout; not captured."));
      return;
    }

    // Dynamic-buffer draws split into two kinds. CPU-modified meshes (UE3 CPU-skins
    // morph/cloth-augmented skeletal meshes into DEDICATED dynamic buffers, e.g. the
    // first person arms) carry their motion in the vertex positions and are captured
    // with a position snapshot below; they are recognizable by a bone-less shader, a
    // whole-buffer stream (offset 0) and a buffer the draw largely fills. Everything
    // else on dynamic buffers is ring-pool geometry (particles, trails, canvas) whose
    // allocation offsets shift every frame - untrackable, and skipped.
    const bool dynamicMesh = dynamicMeshShape && !skinned;

    if ((vbDynamic || ibDynamic) && !dynamicMesh) {
      // Ring-pool geometry: a dynamic buffer shared between draws whose allocations move every
      // frame, so nothing here can be matched against a previous sighting. Counted because a mesh
      // that carries its motion in its vertex positions is silently left without velocity if it
      // ever lands in this shape rather than being recognised above.
      m_ngxVelocityStats.skippedDynamicBuffer++;

      ONCE(Logger::info(str::format(
        "[RTX NGX Passthrough] Velocity capture rejecting a dynamic-buffer draw: vb dynamic=",
        vbDynamic ? 1 : 0, ", ib dynamic=", ibDynamic ? 1 : 0, ", bone palette=", skinned ? 1 : 0,
        ", stream offset=", d3d9State().vertexBuffers[0].offset,
        ", vb bytes=", vertexBufferCommon->Desc()->Size,
        ". Motion carried in vertex positions is invisible to the capture in this shape.")));
      return;
    }

    // Depth-writing draws only, except CPU-modified meshes (see the z gate above)
    if (!zWriteEnabled && !dynamicMesh) {
      return;
    }

    // Snapshot the CPU-modified mesh's current positions from the dynamic buffer's CPU
    // mapping (tightly packed, whole buffer: the index/base-vertex ranges then apply to
    // the snapshot exactly as to the live stream)
    std::vector<Vector3> currentPositions;

    if (dynamicMesh) {
      if (m_ngxVelocityDynamicDraws >= kNgxVelocityMaxDynamicDraws) {
        m_ngxVelocityStats.skippedBudget++;
        return;
      }

      const uint32_t stride = d3d9State().vertexBuffers[0].stride;
      const uint32_t vertexCount = stride != 0 ? uint32_t(vertexBufferCommon->Desc()->Size / stride) : 0;
      const DxvkBufferSliceHandle mappedSlice = vertexBufferCommon->GetMappedSlice();

      if (vertexCount == 0 || vertexCount > kNgxVelocityMaxDynamicVertices || mappedSlice.mapPtr == nullptr) {
        return;
      }

      currentPositions.resize(vertexCount);
      const uint8_t* positionBytes = reinterpret_cast<const uint8_t*>(mappedSlice.mapPtr) + positionOffset;
      for (uint32_t vertex = 0; vertex < vertexCount; vertex++) {
        std::memcpy(&currentPositions[vertex], positionBytes + size_t(vertex) * stride, sizeof(Vector3));
      }
    }

    // Draw identity: geometry references plus draw parameters. Stable for UE3 static
    // meshes; buffer reallocation across level loads simply re-registers the object.
    // CPU-modified meshes swap between vertex buffer objects when re-skinned (double
    // buffering), so their identity anchors on the buffer SIZE instead of its address -
    // the static index buffer and the draw parameters still separate mesh sections.
    // Hashed from an explicitly packed array: hashing a struct with pointer members would
    // include compiler tail padding, which aggregate initialization leaves uninitialized,
    // making identities jitter with incidental stack contents and breaking the pairing.
    const uint64_t identityData[5] = {
      dynamicMesh ? uint64_t(vertexBufferCommon->Desc()->Size)
                  : uint64_t(reinterpret_cast<uintptr_t>(vertexBufferCommon)),
      uint64_t(reinterpret_cast<uintptr_t>(indexBufferCommon)),
      (uint64_t(drawContext.StartIndex) << 32) | uint64_t(uint32_t(drawContext.PrimitiveCount)),
      (uint64_t(uint32_t(drawContext.BaseVertexIndex)) << 32) | uint64_t(d3d9State().vertexBuffers[0].stride),
      uint64_t(positionOffset),
    };
    const XXH64_hash_t identity = XXH3_64bits(identityData, sizeof(identityData));

    const Vector4* boneRegisters = skinned ? &d3d9State().vsConsts.fConsts[ctabRegs.boneMatricesRegister] : nullptr;
    const uint32_t boneRegisterCount = skinned ? ctabRegs.boneMatricesRegisterCount : 0;

    NgxVelocityObjectState& objectState = m_ngxVelocityObjectCache[identity];

    // Shared draw construction for the emission sites below. previousBones is the
    // instance's cached palette; when unusable (first skinned sighting of the identity),
    // the current palette stands in for both sides - transform-only motion that frame.
    const auto appendVelocityDraw = [&](const Matrix4& objectToWorldCurrent,
                                        const Matrix4& worldToProjectionPrevious,
                                        const Matrix4& objectToWorldPrevious,
                                        const std::vector<Vector4>& previousBones,
                                        const std::vector<Vector3>& previousPositions) {
      NgxVelocityDraw velocityDraw;
      velocityDraw.vertexBuffer = vertexBufferCommon->GetBufferSlice<D3D9_COMMON_BUFFER_TYPE_REAL>(d3d9State().vertexBuffers[0].offset);
      velocityDraw.vertexStride = d3d9State().vertexBuffers[0].stride;
      velocityDraw.positionOffset = positionOffset;
      velocityDraw.positionFormat = positionFormat;
      velocityDraw.indexBuffer = indexBufferCommon->GetBufferSlice<D3D9_COMMON_BUFFER_TYPE_REAL>();
      velocityDraw.indexType = DecodeIndexType(static_cast<D3D9Format>(indexBufferCommon->Desc()->Format));
      velocityDraw.indexCount = drawContext.PrimitiveCount * 3;
      velocityDraw.firstIndex = drawContext.StartIndex;
      velocityDraw.vertexOffset = drawContext.BaseVertexIndex;

      velocityDraw.clipFromLocal = drawWorldToProjection * objectToWorldCurrent;
      velocityDraw.prevClipFromLocal = worldToProjectionPrevious * objectToWorldPrevious;
      velocityDraw.foregroundPhase = foregroundPhase;
      velocityDraw.viewportMinZ = d3d9State().viewport.MinZ;
      velocityDraw.viewportMaxZ = d3d9State().viewport.MaxZ;

      if (foregroundPhase && (d3d9State().viewport.MinZ != 0.0f || d3d9State().viewport.MaxZ != 1.0f)) {
        ONCE(Logger::info(str::format("[RTX NGX Passthrough] Foreground DPG renders with a squashed viewport depth range: [",
                                      d3d9State().viewport.MinZ, ", ", d3d9State().viewport.MaxZ, "].")));
      }

      if (skinned) {
        velocityDraw.blendIndicesOffset = blendIndicesOffset;
        velocityDraw.blendWeightsOffset = blendWeightsOffset;
        velocityDraw.boneIndexScale = ctabRegs.boneIndicesPreScaled ? 1u : 3u;
        velocityDraw.skinInfluenceCount = std::clamp(ctabRegs.skinInfluenceCount, 1u, 4u);

        ONCE(Logger::info(str::format("[RTX NGX Passthrough] Skinned velocity capture active (bone index addressing: ",
                                      ctabRegs.boneIndicesPreScaled ? "pre-scaled" : "shader-scaled",
                                      ", influences: ", velocityDraw.skinInfluenceCount, ").")));

        velocityDraw.bonesPrevious.resize(kNgxVelocityBonePaletteRegisters, Vector4(0.0f));
        velocityDraw.bonesCurrent.resize(kNgxVelocityBonePaletteRegisters, Vector4(0.0f));

        const Vector4* previousPalette =
          previousBones.size() == boneRegisterCount ? previousBones.data() : boneRegisters;
        std::memcpy(velocityDraw.bonesPrevious.data(), previousPalette, boneRegisterCount * sizeof(Vector4));
        std::memcpy(velocityDraw.bonesCurrent.data(), boneRegisters, boneRegisterCount * sizeof(Vector4));

        m_ngxVelocitySkinnedDraws++;
        m_ngxVelocityStats.capturedSkinned++;
      }

      if (dynamicMesh) {
        velocityDraw.previousPositions =
          previousPositions.size() == currentPositions.size() ? previousPositions : currentPositions;
        m_ngxVelocityDynamicDraws++;
        m_ngxVelocityStats.capturedDynamic++;
      }

      if (foregroundPhase) {
        m_ngxVelocityStats.capturedForeground++;
      }

      m_ngxVelocityDraws.push_back(std::move(velocityDraw));
      m_ngxVelocityStats.captured++;
    };

    // Tolerances around the scene-wide transform offset, all in world units and all sized
    // against the same thing: how exactly a translation that did not move can be expected to
    // reproduce that offset. The cluster width is the floor, set by float precision on a
    // difference of two level-scale positions. A bone's translation is compared at the same
    // width, being the same kind of quantity. A placement is held to a wider bound because the
    // offset it is judged against was itself only measured to within the cluster width, and
    // because a residual has to stay far below the spacing between two copies of an asset to
    // keep them apart.
    constexpr float kOffsetClusterTolerance = 0.05f;
    constexpr float kBoneTranslationTolerance = kOffsetClusterTolerance;
    constexpr float kSamePlacementTolerance = 0.5f;

    // Position delta against a cached snapshot (CPU-modified meshes animate their
    // vertex data with a typically static LocalToWorld)
    const auto dynamicPositionsChangedFrom = [&](const std::vector<Vector3>& cachedPositions) {
      if (!dynamicMesh) {
        return false;
      }
      if (cachedPositions.size() != currentPositions.size()) {
        return true;
      }
      return std::memcmp(cachedPositions.data(), currentPositions.data(),
                         cachedPositions.size() * sizeof(Vector3)) != 0;
    };

    // Bone palette delta against a cached palette, net of a translation the whole palette is
    // expected to have taken. The palette holds bone-to-world matrices, so where that space
    // shifts every frame every bone's translation shifts with it, and a skinned mesh that never
    // moved reports its entire palette as animated - which emits velocity unconditionally,
    // ahead of every motion gate. Destructible and instanced scenery is commonly drawn through
    // skinned vertex factories, so that reads as level geometry moving under its own power.
    // Only the translation is affected, which the palette carries in .w, one axis per row of
    // each bone's three; the rotation rows are compared as they are.
    const auto bonesChangedFrom = [&](const std::vector<Vector4>& cachedBones,
                                      const Vector3& expectedTranslationShift) {
      if (!skinned || cachedBones.size() != boneRegisterCount) {
        return false;
      }
      for (uint32_t reg = 0; reg < boneRegisterCount; reg++) {
        const Vector4 delta = boneRegisters[reg] - cachedBones[reg];
        if (std::abs(delta.x) > 1e-5f || std::abs(delta.y) > 1e-5f || std::abs(delta.z) > 1e-5f) {
          return true;
        }

        const uint32_t axis = reg % 3;
        const float axisShift = axis == 0 ? expectedTranslationShift.x
                              : axis == 1 ? expectedTranslationShift.y
                                          : expectedTranslationShift.z;
        // Looser than the rotation rows: a bone's translation is a world position, and the
        // shift being subtracted was itself measured to within a fraction of a unit
        if (std::abs(delta.w - axisShift) > kBoneTranslationTolerance) {
          return true;
        }
      }
      return false;
    };

    // The furthest anything is credited with travelling in one frame, applied both to an
    // object pairing with its own past and to the scene's own coordinate space shifting.
    // Generous: a frame hitch multiplies every per-frame delta, and a delta pushed past the
    // bound costs two frames of velocity.
    constexpr float kMaxFrameTranslation = 250.0f;  // world units per frame

    const Matrix4 sightingObjectToWorld =
      extractUe3ObjectToWorld(l2wReg, hasWorldToLocal, ctabRegs.worldToLocalRegister, m_ngxFrameCameraUsedTranspose);

    // Every placement that held still moved by the same global offset, so the offset is whatever
    // the most sightings agree on. Movers disagree with each other as much as with the static
    // scene, which is what keeps them out of the winning slot.
    auto voteForGlobalTransformOffset = [&](const Vector3& delta) {
      // One vote per identity per frame. Copies of a single instanced asset sit on a shared
      // grid, so when the visible set shifts they pair with their neighbours and agree, in
      // bulk, on a delta that is one grid step rather than the scene's: counted per sighting
      // that bloc outvotes the real offset outright. Counted per identity it is worth one
      // voice, and the offset is instead carried by how many unrelated assets report it.
      if (objectState.lastOffsetVoteFrame == m_ue3FrameCounter) {
        return;
      }

      // No camera crosses a level in a frame. A delta this large is a pairing across two
      // placements, and adopting it would put the same error into every reprojected pixel.
      if (lengthSqr(delta) > kMaxFrameTranslation * kMaxFrameTranslation) {
        return;
      }

      objectState.lastOffsetVoteFrame = m_ue3FrameCounter;

      NgxTranslationDeltaVote* slot = nullptr;
      NgxTranslationDeltaVote* freeSlot = nullptr;
      for (NgxTranslationDeltaVote& vote : m_ngxTranslationDeltaVotes) {
        if (vote.votes == 0) {
          freeSlot = freeSlot != nullptr ? freeSlot : &vote;
          continue;
        }
        // Grouped loosely enough to survive float precision at level-scale coordinates, where
        // a difference of two positions in the tens of thousands already carries thousandths
        // of error: too tight and the far reaches of a level fragment one true offset across
        // several slots, none of them carrying enough votes to be believed.
        if (lengthSqr(vote.delta - delta) <= kOffsetClusterTolerance * kOffsetClusterTolerance) {
          slot = &vote;
          break;
        }
      }

      if (slot != nullptr) {
        slot->votes++;
      } else if (freeSlot != nullptr) {
        freeSlot->delta = delta;
        freeSlot->votes = 1;
        slot = freeSlot;
      } else {
        // Every slot is held by a candidate this one disagrees with; nothing to record it in
        return;
      }

      // Follow the leader from its opening vote rather than waiting for a quorum: until the
      // frame has adopted an offset of its own the placement test is judging against the
      // previous frame's, which differs by however much the camera accelerated, and a static
      // placement that fails that test emits a velocity it should not. An opening vote may
      // come from a mover, so it only stands until two sightings agree on something else.
      if (slot->votes > m_ngxAdoptedOffsetVotes) {
        m_ngxGlobalTransformOffset = slot->delta;
        m_ngxAdoptedOffsetVotes = slot->votes;
      }
    };

    // Same placement: the basis is untouched and the translation moved by exactly what the rest
    // of the scene moved by. A repeat draw within the frame has not moved at all; one carried
    // over from the previous frame moved by the global offset, which is zero wherever
    // LocalToWorld is true world space and the comparison is then plain equality.
    auto isSamePlacement = [&](const NgxVelocityObjectInstance& instance) {
      for (uint32_t col = 0; col < 3; col++) {
        const Vector4 delta = sightingObjectToWorld[col] - instance.objectToWorld[col];
        if (std::abs(delta.x) > 1e-4f || std::abs(delta.y) > 1e-4f || std::abs(delta.z) > 1e-4f) {
          return false;
        }
      }

      Vector3 expectedMove(0.0f, 0.0f, 0.0f);
      if (instance.lastSeenFrame == m_ue3FrameCounter - 1) {
        expectedMove = m_ngxGlobalTransformOffset;
      } else if (instance.lastSeenFrame != m_ue3FrameCounter) {
        // Older history spans an unknown number of global offsets, so it cannot be judged here;
        // the near match below pairs it if the motion is believable
        return false;
      }

      // A placement that held still reproduces the offset to float precision, so this only has
      // to absorb rounding at level-scale coordinates - staying far below the slowest motion
      // worth a velocity, and far below the spacing between two copies of an instanced asset
      const Vector3 residual = (sightingObjectToWorld[3].xyz() - instance.objectToWorld[3].xyz()) - expectedMove;
      return lengthSqr(residual) <= kSamePlacementTolerance * kSamePlacementTolerance;
    };

    // Static placement re-uploading the same matrix every frame, or a repeat draw of an instance
    // already handled this frame (depth prepass + base pass, multi-pass lighting). Static
    // placements get no velocity draw - the camera reprojection is exact for them. Skinned
    // instances with a changed palette are animation in place: same placement, no swap
    // ambiguity, always emitted.
    for (NgxVelocityObjectInstance& instance : objectState.instances) {
      if (isSamePlacement(instance)) {
        // Static placement, but the content may animate in place: skinned palettes or
        // CPU-modified positions changing under an unchanged LocalToWorld - no swap
        // ambiguity, always emitted
        // A repeat draw within the frame has taken no shift; one carried over from the previous
        // frame has taken the scene's, exactly as isSamePlacement judged its transform
        const Vector3 expectedBoneShift = instance.lastSeenFrame == m_ue3FrameCounter
                                            ? Vector3(0.0f, 0.0f, 0.0f)
                                            : m_ngxGlobalTransformOffset;
        const bool contentAnimated = (skinned && bonesChangedFrom(instance.bones, expectedBoneShift)) ||
                                     (dynamicMesh && dynamicPositionsChangedFrom(instance.dynamicPositions));

        if (contentAnimated) {
          appendVelocityDraw(sightingObjectToWorld, instance.worldToProjection, instance.objectToWorld,
                             instance.bones, instance.dynamicPositions);
          instance.lastEmitFrame = m_ue3FrameCounter;
          instance.lastEmitDrawIndex = uint32_t(m_ngxVelocityDraws.size() - 1);
          instance.lastEmitPrevObjectToWorld = instance.objectToWorld;
          instance.lastEmitPrevWorldToProjection = instance.worldToProjection;
        }

        if (skinned) {
          instance.bones.assign(boneRegisters, boneRegisters + boneRegisterCount);
        }
        if (dynamicMesh) {
          // Last use of the snapshot (appendVelocityDraw above copied what it needed)
          instance.dynamicPositions = std::move(currentPositions);
        }

        // Same-frame repeat draw in the other scene phase: re-emit this frame's draw for
        // this phase so both velocity targets receive the object, with both clip
        // transforms recomposed for this phase's projection (the foreground DPG renders
        // with its own first-person projection; each target depth-tests against its own
        // phase's depth, discarding any copy that does not belong)
        if (instance.lastEmitFrame == m_ue3FrameCounter &&
            instance.lastPhaseDuplicateFrame != m_ue3FrameCounter &&
            instance.lastEmitDrawIndex < m_ngxVelocityDraws.size() &&
            m_ngxVelocityDraws.size() < kMaxVelocityDrawsPerFrame) {
          const NgxVelocityDraw& emittedDraw = m_ngxVelocityDraws[instance.lastEmitDrawIndex];

          if (emittedDraw.foregroundPhase != foregroundPhase &&
              (emittedDraw.bonesCurrent.empty() || m_ngxVelocitySkinnedDraws < kNgxVelocityMaxSkinnedDraws) &&
              (emittedDraw.previousPositions.empty() || m_ngxVelocityDynamicDraws < kNgxVelocityMaxDynamicDraws)) {
            NgxVelocityDraw duplicateDraw = emittedDraw;
            duplicateDraw.foregroundPhase = foregroundPhase;
            duplicateDraw.clipFromLocal = drawWorldToProjection * instance.objectToWorld;
            duplicateDraw.prevClipFromLocal = instance.lastEmitPrevWorldToProjection * instance.lastEmitPrevObjectToWorld;
            duplicateDraw.viewportMinZ = d3d9State().viewport.MinZ;
            duplicateDraw.viewportMaxZ = d3d9State().viewport.MaxZ;

            if (!duplicateDraw.bonesCurrent.empty()) {
              m_ngxVelocitySkinnedDraws++;
              m_ngxVelocityStats.capturedSkinned++;
            }
            if (!duplicateDraw.previousPositions.empty()) {
              m_ngxVelocityDynamicDraws++;
              m_ngxVelocityStats.capturedDynamic++;
            }
            if (duplicateDraw.foregroundPhase) {
              m_ngxVelocityStats.capturedForeground++;
            }

            m_ngxVelocityDraws.push_back(std::move(duplicateDraw));
            m_ngxVelocityStats.captured++;
            instance.lastPhaseDuplicateFrame = m_ue3FrameCounter;
          }
        }

        // The placement is the same one, but where LocalToWorld carries a global offset its
        // matrix is not: it has to be carried forward, or the stored transform ages by a frame
        // every frame and the offset the next sighting has to reproduce compounds out of reach
        if (instance.lastSeenFrame != m_ue3FrameCounter) {
          voteForGlobalTransformOffset(sightingObjectToWorld[3].xyz() - instance.objectToWorld[3].xyz());
        }
        instance.objectToWorld = sightingObjectToWorld;
        instance.lastSeenFrame = m_ue3FrameCounter;
        instance.worldToProjection = drawWorldToProjection;

        // Confirmed-mover status decays after a couple seconds at rest: a stale latch
        // is an instant-emit backdoor for visibility swaps pairing against this
        // instance, while a real mover restarting re-confirms through the gentle-onset
        // path with zero frames lost (motion from rest is always gentle at first)
        constexpr uint32_t kWasMovingDecayFrames = 120;
        if (instance.wasMoving && m_ue3FrameCounter - instance.lastEmitFrame > kWasMovingDecayFrames) {
          instance.wasMoving = false;
        }

        m_ngxVelocityStats.exactMatches++;
        return;
      }
    }

    // Disambiguated transform in the same convention the camera reconstruction uses;
    // composed exactly like the motion vector pass composes its reprojection chain
    const Matrix4& objectToWorld = sightingObjectToWorld;

    // Near match against instances seen exactly one frame ago: the same object having
    // moved. Static placements are consumed by the exact match above, so the bounds only
    // arbitrate between simultaneously moving identical movers - the nearest-score pairs
    // each with its own history even when both fit the bounds (double door leaves).
    constexpr float kMaxFrameRotScaleL1 = 3.0f;     // L1 delta over the 3x3 rotation/scale
    const uint32_t previousFrame = m_ue3FrameCounter - 1;

    NgxVelocityObjectInstance* matchedInstance = nullptr;
    float matchedScore = 0.0f;
    float matchedTranslationDelta = 0.0f;
    float matchedRotScaleDelta = 0.0f;

    // Nearest last-frame candidate regardless of bounds: consumed by the single-candidate
    // acceptance below and by the pairing-miss diagnostics
    NgxVelocityObjectInstance* nearestCandidate = nullptr;
    float nearestScore = 0.0f;
    float nearestTranslationDelta = 0.0f;
    float nearestRotScaleDelta = 0.0f;
    uint32_t lastFrameCandidates = 0;

    // Placements sighted within the last couple frames: the cache also holds stale
    // instances from previously visited areas (pruned lazily), which must not count
    // against the identity-stability assessment below
    uint32_t activeInstances = 0;

    // Frames since the freshest sighting of this identity, for the pairing-miss diagnostics.
    // With no last-frame candidate the question is whether the history is one frame too old -
    // the capture skipping frames - or absent entirely, which is a different fault.
    uint32_t freshestSightingAge = UINT32_MAX;

    for (NgxVelocityObjectInstance& instance : objectState.instances) {
      freshestSightingAge = std::min(freshestSightingAge, m_ue3FrameCounter - instance.lastSeenFrame);

      if (instance.lastSeenFrame + 2 >= m_ue3FrameCounter) {
        activeInstances++;
      }

      if (instance.lastSeenFrame != previousFrame) {
        continue;
      }
      lastFrameCandidates++;

      // Motion is what is left after the offset the scene as a whole moved by, so a placement
      // that held still reads as motionless here however far the space its transform is
      // expressed in travelled. Measured raw, every static placement that misses the
      // same-placement test above - through an evicted slot, an ambiguity between copies, or
      // simply never having been seen before - instead reads as moving by the camera's own
      // travel. That is gentle enough to pass the onset gate below, and emitting once sets
      // wasMoving, which keeps it emitting from then on.
      const Vector3 rawTranslationDelta = objectToWorld[3].xyz() - instance.objectToWorld[3].xyz();
      const float translationDelta = length(rawTranslationDelta - m_ngxGlobalTransformOffset);

      float rotScaleDelta = 0.0f;
      for (uint32_t col = 0; col < 3; col++) {
        const Vector4 delta = objectToWorld[col] - instance.objectToWorld[col];
        rotScaleDelta += std::abs(delta.x) + std::abs(delta.y) + std::abs(delta.z);
      }

      // The offset itself is measured raw, and from here rather than only from the matches
      // above, so an estimate that has gone stale can still be re-measured: were it fed only
      // by sightings the estimate itself admitted, a wrong estimate would reject every
      // sighting and never be corrected
      if (rotScaleDelta <= 1e-4f) {
        voteForGlobalTransformOffset(rawTranslationDelta);
      }

      const float score = translationDelta / kMaxFrameTranslation + rotScaleDelta / kMaxFrameRotScaleL1;

      if (nearestCandidate == nullptr || score < nearestScore) {
        nearestCandidate = &instance;
        nearestScore = score;
        nearestTranslationDelta = translationDelta;
        nearestRotScaleDelta = rotScaleDelta;
      }

      if (translationDelta > kMaxFrameTranslation || rotScaleDelta > kMaxFrameRotScaleL1) {
        continue;
      }

      if (matchedInstance == nullptr || score < matchedScore) {
        matchedInstance = &instance;
        matchedScore = score;
        matchedTranslationDelta = translationDelta;
        matchedRotScaleDelta = rotScaleDelta;
      }
    }

    // The bounds exist to disambiguate between multiple placements sharing an identity;
    // with exactly one unclaimed last-frame instance the pairing is unambiguous by
    // elimination, so faster-than-bounds motion (trains cover hundreds of units per
    // frame; frame hitches multiply every delta) may pair too.
    bool pairedBeyondBounds = false;
    if (matchedInstance == nullptr && lastFrameCandidates == 1) {
      matchedInstance = nearestCandidate;
      matchedTranslationDelta = nearestTranslationDelta;
      matchedRotScaleDelta = nearestRotScaleDelta;
      pairedBeyondBounds = matchedInstance != nullptr;
      if (pairedBeyondBounds) {
        m_ngxVelocityStats.pairedBeyondBounds++;
      }
    }

    if (matchedInstance != nullptr) {
      // Net of the scene-wide offset, matching matchedTranslationDelta and the stored history
      // this is compared against on the next sighting
      const Vector3 moveDelta =
        (objectToWorld[3].xyz() - matchedInstance->objectToWorld[3].xyz()) - m_ngxGlobalTransformOffset;

      // Emission gate: a pairing only produces velocity when the motion is believable.
      //  - Negligible deltas (one-time transform settles) are claimed silently: sub-pixel
      //    motion is served equally well by camera reprojection.
      //  - Confirmed movers (wasMoving) emit on sight.
      //  - Unconfirmed instances emit immediately only for gentle motion, plausible for
      //    an object accelerating from rest. Larger first deltas - what a visibility swap
      //    between two placements of the same asset looks like - must repeat consistently
      //    for one frame first: a real fast mover sustains its per-frame delta, a swap
      //    does not repeat.
      //  - Pairings taken by elimination take that confirmation path unconditionally.
      constexpr float kNegligibleTranslation = 0.05f;
      constexpr float kNegligibleRotScale = 1e-3f;
      constexpr float kOnsetTranslationTrust = 8.0f;   // ~480 units/s at 60 fps
      constexpr float kOnsetRotScaleTrust = 0.35f;     // ~3 degrees/frame

      // Animating content (skinned palettes / CPU-modified positions changing) rules out a
      // swap between two static copies of an asset, whose vertex data never animates, so a
      // pairing that stayed within the bounds skips the anti-swap gates below (first person
      // meshes pair through camera-attached transform deltas that routinely exceed them).
      // It cannot rule out a swap between two animated copies - see pairedBeyondBounds.
      // Paired against a last-frame instance by construction, so the scene's shift applies
      const bool contentAnimated = bonesChangedFrom(matchedInstance->bones, m_ngxGlobalTransformOffset) ||
                                   dynamicPositionsChangedFrom(matchedInstance->dynamicPositions);

      const bool negligibleMotion = !contentAnimated &&
                                    matchedTranslationDelta <= kNegligibleTranslation &&
                                    matchedRotScaleDelta <= kNegligibleRotScale;

      bool emitVelocity = false;

      // A pairing taken by elimination sits beyond the per-frame bounds, so it is as much the
      // shape of one placement leaving view as another arrives as it is of real motion, and
      // nothing about the sighting itself tells them apart. Only the delta repeating does -
      // a fast mover sustains it, a visibility swap does not - so these take the confirmation
      // path below regardless of what else vouches for them. Animating content does not: two
      // copies of one skeletal asset both animate, and pairing across them emits the distance
      // between two different characters as a single frame of motion. Neither does wasMoving,
      // which describes the instance's own past and not this sighting's claim to it.
      if (contentAnimated && !pairedBeyondBounds) {
        emitVelocity = true;
      } else if (!negligibleMotion) {
        if (matchedInstance->wasMoving && !pairedBeyondBounds) {
          emitVelocity = true;
        } else if (!pairedBeyondBounds &&
                   matchedTranslationDelta <= kOnsetTranslationTrust &&
                   matchedRotScaleDelta <= kOnsetRotScaleTrust) {
          emitVelocity = true;
        } else {
          // Consistency confirmation is only trusted for identities with few placements
          // currently in view: real movers have one or two, while grids of instanced
          // meshes produce repeating pop-in deltas under steady camera movement that
          // pass any repetition test (those may only confirm through gentle onset, which
          // pop-in distances can never satisfy). A single active placement is
          // unambiguous and skips the churn-recency requirement - its own first
          // registration would otherwise defer its confirmation.
          const bool identityStable = activeInstances <= 4 &&
                                      (activeInstances <= 1 ||
                                       m_ue3FrameCounter > objectState.lastNewRegistrationFrame + 2);

          const bool hadMotion = lengthSqr(matchedInstance->lastMoveDelta) > 0.0f ||
                                 matchedInstance->lastRotScaleDelta > 0.0f;
          const bool translationConsistent =
            length(moveDelta - matchedInstance->lastMoveDelta) <= std::max(0.25f * matchedTranslationDelta, 2.0f);
          const bool rotScaleConsistent =
            std::abs(matchedRotScaleDelta - matchedInstance->lastRotScaleDelta) <= std::max(0.25f * matchedRotScaleDelta, 0.05f);

          emitVelocity = identityStable && hadMotion && translationConsistent && rotScaleConsistent;
        }
      }

      if (emitVelocity) {
        appendVelocityDraw(objectToWorld, matchedInstance->worldToProjection, matchedInstance->objectToWorld,
                           matchedInstance->bones, matchedInstance->dynamicPositions);
        matchedInstance->lastEmitPrevObjectToWorld = matchedInstance->objectToWorld;
        matchedInstance->lastEmitPrevWorldToProjection = matchedInstance->worldToProjection;
      } else {
        // Claimed without velocity (negligible motion or deferred onset confirmation)
        m_ngxVelocityStats.claimedWithoutVelocity++;
      }

      // Claim the instance: repeat draws this frame match the updated transform, other
      // instances cannot pair with it anymore. The movement history feeds the emission
      // gate above on the next sighting.
      matchedInstance->objectToWorld = objectToWorld;
      matchedInstance->worldToProjection = drawWorldToProjection;
      matchedInstance->lastSeenFrame = m_ue3FrameCounter;
      matchedInstance->lastMoveDelta = moveDelta;
      matchedInstance->lastRotScaleDelta = matchedRotScaleDelta;
      if (skinned) {
        matchedInstance->bones.assign(boneRegisters, boneRegisters + boneRegisterCount);
      } else {
        matchedInstance->bones.clear();
      }
      if (dynamicMesh) {
        // Last use of the snapshot (appendVelocityDraw above copied what it needed)
        matchedInstance->dynamicPositions = std::move(currentPositions);
      } else {
        matchedInstance->dynamicPositions.clear();
      }
      if (emitVelocity) {
        matchedInstance->wasMoving = true;
        matchedInstance->lastEmitFrame = m_ue3FrameCounter;
        matchedInstance->lastEmitDrawIndex = uint32_t(m_ngxVelocityDraws.size() - 1);
      }
      return;
    }

    // Self-triggering pairing-miss dump: a moving object failing to pair with its own
    // one-frame-ago history is the exact failure mode behind velocity dropouts, and the
    // log line carries everything needed to tell apart the possible causes (no last-frame
    // sighting at all vs a candidate rejected by the bounds, and by how much).
    {
      constexpr uint32_t kPairingLogMaxLinesPerFrame = 6;
      constexpr uint32_t kPairingLogCooldownFrames = 120;

      // One burst of lines per cooldown window: the burst frame is latched and its
      // remaining lines stay allowed, further frames wait out the cooldown
      const bool inActiveBurst = m_ngxVelocityPairingLogFrame == m_ue3FrameCounter;

      if (inActiveBurst || m_ue3FrameCounter >= m_ngxVelocityPairingLogNextAllowedFrame) {
        if (!inActiveBurst) {
          m_ngxVelocityPairingLogFrame = m_ue3FrameCounter;
          m_ngxVelocityPairingLogLines = 0;
          m_ngxVelocityPairingLogNextAllowedFrame = m_ue3FrameCounter + kPairingLogCooldownFrames;
        }

        if (m_ngxVelocityPairingLogLines < kPairingLogMaxLinesPerFrame) {
          m_ngxVelocityPairingLogLines++;

          const Vector3 translation = objectToWorld[3].xyz();
          std::string line = str::format(
            "[RTX NGX Passthrough][pairing miss] frame=", m_ue3FrameCounter,
            " identity=0x", std::hex, identity, std::dec,
            " instances=", objectState.instances.size(),
            " lastFrameCandidates=", lastFrameCandidates,
            " freshestSightingAge=", (freshestSightingAge == UINT32_MAX
                                        ? std::string("never") : std::to_string(freshestSightingAge)),
            " o2wPos=(", translation.x, ",", translation.y, ",", translation.z, ")",
            " camTranspose=", int(m_ngxFrameCameraUsedTranspose));

          if (nearestCandidate != nullptr) {
            const Vector3 nearestTranslation = nearestCandidate->objectToWorld[3].xyz();
            line += str::format(
              " nearest: dPos=", nearestTranslationDelta,
              " dRotScale=", nearestRotScaleDelta,
              " pos=(", nearestTranslation.x, ",", nearestTranslation.y, ",", nearestTranslation.z, ")");
          } else {
            line += " nearest: none seen last frame";
          }

          Logger::info(line);
        }
      }
    }

    // No usable history: a new placement of this mesh (or a sighting after a visibility
    // gap, where a one-frame velocity cannot be derived). Register it without velocity;
    // replace the stalest slot when the identity is heavily instanced.
    // Modular level geometry puts far more placements of one asset on screen than a small cap
    // can hold: hundreds of copies of a wall or railing section, all sharing this identity and
    // separated only by their transform.
    constexpr size_t kMaxInstancesPerIdentity = 256;

    NgxVelocityObjectInstance* targetInstance = nullptr;
    if (objectState.instances.size() >= kMaxInstancesPerIdentity) {
      for (NgxVelocityObjectInstance& instance : objectState.instances) {
        // Never recycle a slot the last two frames are still using. Over the cap, the stalest
        // slot is one this frame just claimed, so recycling it blindly makes an identity spend
        // the frame overwriting the history its remaining placements are about to pair against:
        // none of them can then match their own placement, and those that pair at all pair with
        // a neighbouring copy and emit the distance between two of them as motion. Leaving the
        // surplus untracked instead costs nothing, since static geometry is served exactly by
        // camera reprojection.
        if (instance.lastSeenFrame + 1 >= m_ue3FrameCounter) {
          continue;
        }
        if (targetInstance == nullptr || instance.lastSeenFrame < targetInstance->lastSeenFrame) {
          targetInstance = &instance;
        }
      }

      if (targetInstance == nullptr) {
        m_ngxVelocityStats.skippedInstanceCap++;
        return;
      }
    } else {
      targetInstance = &objectState.instances.emplace_back();
    }

    // Fresh occupancy: recycled slots must not inherit the previous occupant's movement
    // history (a stale confirmed-mover latch would instant-emit for the next pairing)
    targetInstance->objectToWorld = objectToWorld;
    targetInstance->worldToProjection = drawWorldToProjection;
    targetInstance->lastSeenFrame = m_ue3FrameCounter;
    targetInstance->wasMoving = false;
    targetInstance->lastMoveDelta = Vector3(0.0f, 0.0f, 0.0f);
    targetInstance->lastRotScaleDelta = 0.0f;
    targetInstance->lastEmitFrame = 0;
    targetInstance->lastEmitDrawIndex = 0;
    targetInstance->lastPhaseDuplicateFrame = 0;
    if (skinned) {
      targetInstance->bones.assign(boneRegisters, boneRegisters + boneRegisterCount);
    } else {
      targetInstance->bones.clear();
    }
    if (dynamicMesh) {
      targetInstance->dynamicPositions = std::move(currentPositions);
    } else {
      targetInstance->dynamicPositions.clear();
    }

    objectState.lastNewRegistrationFrame = m_ue3FrameCounter;
    m_ngxVelocityStats.newRegistrations++;
    if (skinned) {
      m_ngxVelocityStats.newRegistrationsSkinned++;
    }

    // Why the sighting had nothing to pair with. A one-frame velocity dropout is exactly this
    // outcome, so the population behind it has to be readable as a breakdown - the per-sighting
    // miss lines are rate limited to a burst every couple of seconds and cannot characterise it.
    if (lastFrameCandidates == 0) {
      m_ngxVelocityStats.missNoLastFrameSighting++;
    } else if (nearestTranslationDelta > kMaxFrameTranslation) {
      m_ngxVelocityStats.missBeyondTranslation++;
    } else {
      m_ngxVelocityStats.missBeyondRotation++;
    }
  }

  namespace {
    // UE3 declares FSystemSettingsData's sub-structs in a fixed order (Engine/Inc/SystemSettings.h),
    // so however the engine version sizes the earlier ones (FExposedTextureLODSettings grows with
    // TEXTUREGROUP_MAX), the tail is always:
    //   FLOAT ScreenPercentage; UBOOL bUpscaleScreenPercentage;
    //   INT ResX; INT ResY; UBOOL bFullscreen; INT MaxMultiSamples;
    // ResX/ResY/bFullscreen are known exactly from the D3D9 present parameters, which is what the
    // whole layout is discovered from.
    constexpr uint32_t kUe3UpscaleFromScreenPercentage = 0x04;
    constexpr uint32_t kUe3ResXFromScreenPercentage = 0x08;
    constexpr uint32_t kUe3ResYFromScreenPercentage = 0x0c;
    constexpr uint32_t kUe3FullscreenFromScreenPercentage = 0x10;
    constexpr uint32_t kUe3MaxMultisamplesFromScreenPercentage = 0x14;
    constexpr uint32_t kUe3SettingsDataTailSize = 0x18;

    // A record opens with FSystemSettingsDataWorldDetail: INT DetailMode then twenty UBOOLs.
    // Recognising that shape is how the offset of ScreenPercentage within the record - and so
    // the address of GSystemSettings itself - is recovered without knowing the engine version.
    constexpr int32_t kUe3DetailModeMax = 3;
    constexpr uint32_t kUe3HeadBooleanRun = 12;
    constexpr uint32_t kUe3MinSettingsDataSize = 0x40;
    constexpr uint32_t kUe3MaxSettingsDataSize = 0x2000;

    // Relaxed pass: how far back from the resolution anchor ScreenPercentage may sit when a
    // build does not use the stock tail adjacency.
    constexpr uint32_t kUe3MaxRelaxedScreenPercentageBacktrack = 0x40;

    // FSystemSettings embeds Defaults[FSL_LevelCount]; finding those repeats at a constant
    // stride both confirms the hit and recovers sizeof(FSystemSettingsData).
    constexpr uint32_t kUe3DefaultsSearchWindow = 0x8000;
    constexpr uint32_t kUe3MinDefaultsRepeats = 2;

    constexpr int32_t kUe3MaxMultisamplesPlausibleLimit = 64;
    constexpr float kUe3MaxPlausibleScreenPercentage = 400.0f;

    // A wrong candidate must never be trusted on shape alone, so the redirect set is proven by
    // driving this value and watching the game's own scene viewport respond. Above the 50%
    // floor the camera gate falls back to while ScreenPercentage is still unknown.
    constexpr float kNgxScreenPercentageProbeValue = 75.0f;
    constexpr uint32_t kNgxScreenPercentageProbeFrames = 240;
    constexpr int32_t kNgxScreenPercentageProbeTolerance = 16;

    // How long an engine-owned upscale may stay missing before the runtime takes over.
    constexpr uint32_t kNgxMissingEngineUpscaleFrameLimit = 30;

    // Bound retries for transient process/module read failures.
    constexpr uint32_t kNgxSettingsScanMaxAttempts = 30;
    constexpr uint64_t kNgxSettingsScanRetryIntervalMs = 1000;

    // Cap accepted operand sites so a pathological match set can never be patched wholesale.
    constexpr size_t kNgxMaxOperandSites = 64;
    constexpr size_t kNgxMaxRawOperandSites = 256;

    // Every renderer-facing use of ScreenPercentage in UE3 converts it to a scale factor:
    // ScaleScreenCoords compares it against 100.0f before dividing by it, UnScaleScreenCoords
    // multiplies by the reciprocal, NeedsUpscale compares against 100.0f. Reads that merely
    // copy the field elsewhere - settings mirrors, script exposure - have no such constant
    // beside them, and leaving those alone is what keeps the player's saved configuration out
    // of this. The surrounding cluster then picks up same-function reads the compiler shared.
    constexpr size_t kNgxPercentScaleConstantWindow = 24;
    constexpr uintptr_t kNgxSettingsCodeClusterRadius = 0x800;

    // Renderer-only values; GSystemSettings remains authoritative for game configuration.
    struct NgxGameSettingsShadow {
      float screenPercentage;
      int32_t upscaleScreenPercentage;
    };
    static_assert(sizeof(NgxGameSettingsShadow) == 8);

    constexpr uintptr_t kNgxShadowScreenPercentageOffset =
      offsetof(NgxGameSettingsShadow, screenPercentage);
    constexpr uintptr_t kNgxShadowUpscaleScreenPercentageOffset =
      offsetof(NgxGameSettingsShadow, upscaleScreenPercentage);

    struct NgxOperandSite {
      uintptr_t address = 0;
      uint32_t original = 0;
    };

    // Which field a site reads, and how it addresses it. The relative pair is what
    // FSystemSettings::NeedsUpscale() uses, so it decides whether the engine can still perform
    // its own upscale.
    enum class NgxOperandCategory : uint8_t {
      AbsoluteScreenPercentage,
      AbsoluteUpscale,
      RelativeScreenPercentage,
      RelativeUpscale,
    };

    struct NgxOperandRedirect {
      uintptr_t address = 0;
      uint32_t original = 0;
      uint32_t redirected = 0;
      DWORD originalProtection = 0;
      bool originalProtectionKnown = false;
      bool forceRestore = false;
      NgxOperandCategory category = NgxOperandCategory::AbsoluteScreenPercentage;
    };

    struct NgxLocatedGameSettings {
      uintptr_t screenPercentageAddress = 0;
      uintptr_t systemSettingsAddress = 0;
      float currentScreenPercentage = 100.0f;
      int32_t currentUpscaleScreenPercentage = 1;
      int32_t currentMaxMultisamples = 0;
      // Offset of ScreenPercentage within FSystemSettingsData; 0 when the relative readers
      // could not be resolved, in which case only the absolute readers are redirected.
      uint32_t screenPercentageStructOffset = 0;
      uint32_t settingsDataStride = 0;
      uint32_t defaultsRepeats = 0;
      bool strictTailLayout = false;
      bool iniConfirmed = false;
      // Which stage rejected the module, so a failure is readable straight from the log.
      const char* missReason = "no reason recorded";
      std::vector<NgxOperandSite> directScreenPercentageSites;
      std::vector<NgxOperandSite> directUpscaleScreenPercentageSites;
      std::vector<NgxOperandSite> relativeScreenPercentageSites;
      std::vector<NgxOperandSite> relativeUpscaleScreenPercentageSites;
    };

    template <typename T>
    bool ngxReadProcessExact(HANDLE process, uintptr_t address, T& value) {
      SIZE_T bytesRead = 0;
      return ::ReadProcessMemory(process, reinterpret_cast<LPCVOID>(address),
                                 &value, sizeof(value), &bytesRead) &&
             bytesRead == sizeof(value);
    }

    template <typename T>
    bool ngxWriteProcessExact(HANDLE process, uintptr_t address, const T& value) {
      SIZE_T bytesWritten = 0;
      return ::WriteProcessMemory(process, reinterpret_cast<LPVOID>(address),
                                  &value, sizeof(value), &bytesWritten) &&
             bytesWritten == sizeof(value);
    }

    template <typename T>
    bool ngxWriteProcessExactRetry(HANDLE process, uintptr_t address,
                                   const T& value) {
      for (uint32_t attempt = 0; attempt < 3; attempt++) {
        if (ngxWriteProcessExact(process, address, value))
          return true;
      }
      return false;
    }

    bool ngxUpdateScreenPercentageShadow(HANDLE process, uintptr_t shadowAddress,
                                         float screenPercentage, int32_t upscale,
                                         bool* outRollbackVerified = nullptr) {
      if (outRollbackVerified != nullptr)
        *outRollbackVerified = true;
      float oldScreenPercentage = 0.0f;
      int32_t oldUpscale = 0;
      if (!ngxReadProcessExact(
            process, shadowAddress + kNgxShadowScreenPercentageOffset,
            oldScreenPercentage) ||
          !ngxReadProcessExact(
            process, shadowAddress + kNgxShadowUpscaleScreenPercentageOffset,
            oldUpscale))
        return false;

      const auto writeScreenPercentage = [&](float value) {
        return ngxWriteProcessExactRetry(
          process, shadowAddress + kNgxShadowScreenPercentageOffset, value);
      };
      const auto writeUpscale = [&](int32_t value) {
        return ngxWriteProcessExactRetry(
          process, shadowAddress + kNgxShadowUpscaleScreenPercentageOffset,
          value);
      };
      const auto pairEquals = [&](float expectedScreenPercentage,
                                  int32_t expectedUpscale) {
        float verifiedScreenPercentage = 0.0f;
        int32_t verifiedUpscale = 0;
        return
          ngxReadProcessExact(
            process, shadowAddress + kNgxShadowScreenPercentageOffset,
            verifiedScreenPercentage) &&
          ngxReadProcessExact(
            process, shadowAddress + kNgxShadowUpscaleScreenPercentageOffset,
            verifiedUpscale) &&
          std::memcmp(&verifiedScreenPercentage, &expectedScreenPercentage,
                      sizeof(expectedScreenPercentage)) == 0 &&
          verifiedUpscale == expectedUpscale;
      };
      const auto writePairSafely = [&](float targetScreenPercentage,
                                       int32_t targetUpscale) {
        bool firstRestored = false;
        bool secondRestored = false;
        if (targetUpscale != 0) {
          firstRestored = writeUpscale(targetUpscale);
          if (firstRestored)
            secondRestored = writeScreenPercentage(targetScreenPercentage);
        } else {
          firstRestored = writeScreenPercentage(targetScreenPercentage);
          if (firstRestored)
            secondRestored = writeUpscale(targetUpscale);
        }
        return firstRestored && secondRestored;
      };
      const auto rollbackOldPair = [&]() {
        return writePairSafely(oldScreenPercentage, oldUpscale) &&
               pairEquals(oldScreenPercentage, oldUpscale);
      };

      if (!writePairSafely(screenPercentage, upscale)) {
        const bool rollbackVerified = rollbackOldPair();
        if (outRollbackVerified != nullptr)
          *outRollbackVerified = rollbackVerified;
        return false;
      }

      const bool verified = pairEquals(screenPercentage, upscale);
      if (!verified) {
        const bool rollbackVerified = rollbackOldPair();
        if (outRollbackVerified != nullptr)
          *outRollbackVerified = rollbackVerified;
      }
      return verified;
    }

    bool ngxRestoreCodeProtection(HANDLE process, uintptr_t address,
                                  DWORD protection) {
      for (uint32_t attempt = 0; attempt < 3; attempt++) {
        DWORD ignoredProtection = 0;
        if (::VirtualProtectEx(
              process, reinterpret_cast<LPVOID>(address), sizeof(uint32_t),
              protection, &ignoredProtection))
          return true;
      }
      return false;
    }

    bool ngxCodeProtectionMatches(HANDLE process, uintptr_t address,
                                  DWORD protection) {
      MEMORY_BASIC_INFORMATION memoryInfo = {};
      return ::VirtualQueryEx(
               process, reinterpret_cast<LPCVOID>(address),
               &memoryInfo, sizeof(memoryInfo)) == sizeof(memoryInfo) &&
             memoryInfo.Protect == protection;
    }

    // Restore the exact pre-write Protect (WRITECOPY pages cannot always regain it).
    bool ngxWriteCodeOperand(HANDLE process, NgxOperandRedirect& redirect) {
      if (!redirect.originalProtectionKnown) {
        MEMORY_BASIC_INFORMATION memoryInfo = {};
        if (::VirtualQueryEx(
              process, reinterpret_cast<LPCVOID>(redirect.address),
              &memoryInfo, sizeof(memoryInfo)) != sizeof(memoryInfo))
          return false;
        redirect.originalProtection = memoryInfo.Protect;
        redirect.originalProtectionKnown = true;
      }

      DWORD oldProtection = 0;
      if (!::VirtualProtectEx(
            process, reinterpret_cast<LPVOID>(redirect.address),
            sizeof(redirect.redirected), PAGE_EXECUTE_READWRITE,
            &oldProtection))
        return false;

      if (oldProtection != redirect.originalProtection) {
        redirect.originalProtection = oldProtection;
        ngxRestoreCodeProtection(process, redirect.address, oldProtection);
        return false;
      }

      const bool wrote =
        ngxWriteProcessExact(process, redirect.address, redirect.redirected);
      const bool flushed = wrote &&
        ::FlushInstructionCache(
          process, reinterpret_cast<LPCVOID>(redirect.address),
          sizeof(redirect.redirected));
      const bool restoredProtection =
        ngxRestoreCodeProtection(
          process, redirect.address, redirect.originalProtection) &&
        ngxCodeProtectionMatches(
          process, redirect.address, redirect.originalProtection);
      return wrote && flushed && restoredProtection;
    }

    bool ngxRestoreCodeOperand(HANDLE process,
                               const NgxOperandRedirect& redirect) {
      if (!redirect.originalProtectionKnown)
        return false;

      DWORD ignoredProtection = 0;
      if (!::VirtualProtectEx(
            process, reinterpret_cast<LPVOID>(redirect.address),
            sizeof(uint32_t), PAGE_EXECUTE_READWRITE, &ignoredProtection))
        return false;

      const bool wrote =
        ngxWriteProcessExactRetry(
          process, redirect.address, redirect.original);
      const bool flushed = wrote &&
        ::FlushInstructionCache(
          process, reinterpret_cast<LPCVOID>(redirect.address),
          sizeof(uint32_t));
      const bool protectionRestored =
        ngxRestoreCodeProtection(
          process, redirect.address, redirect.originalProtection) &&
        ngxCodeProtectionMatches(
          process, redirect.address, redirect.originalProtection);
      uint32_t verified = 0;
      return wrote && flushed && protectionRestored &&
             ngxReadProcessExact(process, redirect.address, verified) &&
             verified == redirect.original;
    }

    // Quiesce target threads while unaligned x86 operands are changed.
    class NgxScopedThreadSuspension {
    public:
      explicit NgxScopedThreadSuspension(DWORD processId) {
        const DWORD currentProcessId = ::GetCurrentProcessId();
        const DWORD currentThreadId = ::GetCurrentThreadId();

        // Repeat snapshots to close the thread-creation race. Fixed storage avoids allocating
        // from a heap that a suspended single-process thread may own.
        uint32_t stablePasses = 0;
        for (uint32_t pass = 0; pass < 8; pass++) {
          HANDLE snapshot =
            ::CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
          if (snapshot == INVALID_HANDLE_VALUE)
            return;

          THREADENTRY32 entry = {};
          entry.dwSize = sizeof(entry);
          if (!::Thread32First(snapshot, &entry)) {
            ::CloseHandle(snapshot);
            return;
          }

          bool discoveredThread = false;
          bool transientThreadExit = false;
          DWORD enumerationError = ERROR_SUCCESS;
          for (;;) {
            if (entry.th32OwnerProcessID == processId &&
                !(processId == currentProcessId &&
                  entry.th32ThreadID == currentThreadId) &&
                !isTracked(entry.th32ThreadID)) {
              discoveredThread = true;
              if (m_threadCount >= m_threads.size()) {
                ::CloseHandle(snapshot);
                return;
              }

              HANDLE thread =
                ::OpenThread(THREAD_SUSPEND_RESUME, FALSE,
                             entry.th32ThreadID);
              if (thread == nullptr) {
                if (::GetLastError() == ERROR_INVALID_PARAMETER) {
                  transientThreadExit = true;
                } else {
                  ::CloseHandle(snapshot);
                  return;
                }
              } else if (::SuspendThread(thread) == DWORD(-1)) {
                transientThreadExit = true;
                ::CloseHandle(thread);
              } else {
                m_threadIds[m_threadCount] = entry.th32ThreadID;
                m_threads[m_threadCount] = thread;
                m_threadCount++;
              }
            }

            ::SetLastError(ERROR_SUCCESS);
            if (!::Thread32Next(snapshot, &entry)) {
              enumerationError = ::GetLastError();
              break;
            }
          }

          ::CloseHandle(snapshot);
          if (enumerationError != ERROR_NO_MORE_FILES)
            return;

          if (!discoveredThread && !transientThreadExit) {
            if (++stablePasses >= 2) {
              m_complete = true;
              return;
            }
          } else {
            stablePasses = 0;
          }
        }
      }

      ~NgxScopedThreadSuspension() {
        for (size_t i = m_threadCount; i > 0; i--)
          ::ResumeThread(m_threads[i - 1]);
        for (size_t i = 0; i < m_threadCount; i++)
          ::CloseHandle(m_threads[i]);
      }

      bool complete() const {
        return m_complete;
      }

    private:
      bool isTracked(DWORD threadId) const {
        for (size_t i = 0; i < m_threadCount; i++) {
          if (m_threadIds[i] == threadId)
            return true;
        }
        return false;
      }

      bool m_complete = false;
      size_t m_threadCount = 0;
      std::array<HANDLE, 256> m_threads = {};
      std::array<DWORD, 256> m_threadIds = {};
    };

    struct NgxOperandApplyResult {
      size_t applied = 0;
      size_t skipped = 0;
      size_t unrestorable = 0;
      size_t appliedByCategory[4] = {};
    };

    // Per-site tolerance: a site that cannot be written, or that cannot regain its original page
    // protection afterwards (WRITECOPY pages do not always come back), is rolled back and left
    // stock instead of failing the whole set. Whether the sites that did land are sufficient is
    // settled by the behavioural probe rather than guessed at here.
    NgxOperandApplyResult ngxApplyOperandRedirects(
        HANDLE process,
        std::vector<NgxOperandRedirect>& redirects,
        std::vector<NgxOperandRedirect>& outActive) {
      NgxOperandApplyResult result;
      outActive.clear();
      // Caller must reserve before threads are suspended (no heap growth while suspended).
      if (outActive.capacity() < redirects.size()) {
        result.skipped = redirects.size();
        return result;
      }

      for (NgxOperandRedirect& redirect : redirects) {
        uint32_t current = 0;
        MEMORY_BASIC_INFORMATION memoryInfo = {};
        if (!ngxReadProcessExact(process, redirect.address, current) ||
            current != redirect.original ||
            ::VirtualQueryEx(
              process, reinterpret_cast<LPCVOID>(redirect.address),
              &memoryInfo, sizeof(memoryInfo)) != sizeof(memoryInfo)) {
          result.skipped++;
          continue;
        }
        redirect.originalProtection = memoryInfo.Protect;
        redirect.originalProtectionKnown = true;

        uint32_t verified = 0;
        if (ngxWriteCodeOperand(process, redirect) &&
            ngxReadProcessExact(process, redirect.address, verified) &&
            verified == redirect.redirected) {
          result.applied++;
          result.appliedByCategory[size_t(redirect.category)]++;
          outActive.push_back(redirect);
          continue;
        }

        // A failed write may still have landed before protection/cache restoration failed.
        if (!ngxRestoreCodeOperand(process, redirect)) {
          redirect.forceRestore = true;
          result.unrestorable++;
          outActive.push_back(redirect);
        } else {
          result.skipped++;
        }
      }

      return result;
    }

    // Bridge host: parent PID is the game process.
    DWORD ngxGetParentPid() {
      const DWORD selfPid = ::GetCurrentProcessId();
      HANDLE snapshot = ::CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
      if (snapshot == INVALID_HANDLE_VALUE)
        return 0;

      PROCESSENTRY32W entry = {};
      entry.dwSize = sizeof(entry);
      DWORD parentPid = 0;
      if (::Process32FirstW(snapshot, &entry)) {
        do {
          if (entry.th32ProcessID == selfPid) {
            parentPid = entry.th32ParentProcessID;
            break;
          }
        } while (::Process32NextW(snapshot, &entry));
      }

      ::CloseHandle(snapshot);
      return parentPid;
    }

    // SNAPMODULE32 so the 64-bit bridge host can see the 32-bit game module.
    bool ngxGetMainModule(DWORD pid, uintptr_t& outBase, uint32_t& outSize,
                          std::wstring& outPath) {
      HANDLE snapshot = ::CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid);
      if (snapshot == INVALID_HANDLE_VALUE)
        return false;

      MODULEENTRY32W module = {};
      module.dwSize = sizeof(module);
      bool ok = false;
      if (::Module32FirstW(snapshot, &module)) {
        outBase = reinterpret_cast<uintptr_t>(module.modBaseAddr);
        outSize = module.modBaseSize;
        outPath = module.szExePath;
        ok = true;
      }

      ::CloseHandle(snapshot);
      return ok;
    }

    enum class NgxLocateResult {
      Found,
      Miss,
      Incomplete,
    };

    // ---------------------------------------------------------------------------------------
    // Optional independent validator: the game's own [SystemSettings] ini section.
    // UE3's property-name table is fixed across engine versions, so agreement between the ini
    // and a candidate record is a known-plaintext confirmation of both base and layout. It only
    // ever raises confidence - a user ini that lags the live values must not reject a candidate.
    // ---------------------------------------------------------------------------------------
    struct NgxIniSystemSettings {
      bool found = false;
      bool hasScreenPercentage = false;
      bool hasUpscale = false;
      bool hasResolution = false;
      bool hasFullscreen = false;
      bool hasMaxMultisamples = false;
      float screenPercentage = 0.0f;
      int32_t upscale = 0;
      int32_t resX = 0;
      int32_t resY = 0;
      int32_t fullscreen = 0;
      int32_t maxMultisamples = 0;
    };

    std::string ngxLowerAscii(const std::string& value) {
      std::string out = value;
      for (char& c : out)
        c = char(std::tolower(uint8_t(c)));
      return out;
    }

    std::string ngxTrimAscii(const std::string& value) {
      size_t first = value.find_first_not_of(" \t\r\n");
      if (first == std::string::npos)
        return std::string();
      size_t last = value.find_last_not_of(" \t\r\n");
      return value.substr(first, last - first + 1);
    }

    bool ngxParseIniBool(const std::string& value, int32_t& out) {
      const std::string lowered = ngxLowerAscii(value);
      if (lowered == "true" || lowered == "1") {
        out = 1;
        return true;
      }
      if (lowered == "false" || lowered == "0") {
        out = 0;
        return true;
      }
      return false;
    }

    bool ngxReadFileText(const std::wstring& path, std::string& out) {
      HANDLE file = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                  nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
      if (file == INVALID_HANDLE_VALUE)
        return false;

      LARGE_INTEGER size = {};
      constexpr LONGLONG kMaxIniBytes = 1 << 20;
      if (!::GetFileSizeEx(file, &size) || size.QuadPart <= 0 || size.QuadPart > kMaxIniBytes) {
        ::CloseHandle(file);
        return false;
      }

      std::vector<uint8_t> raw(size_t(size.QuadPart));
      DWORD bytesRead = 0;
      const bool ok = ::ReadFile(file, raw.data(), DWORD(raw.size()), &bytesRead, nullptr) &&
                      bytesRead == raw.size();
      ::CloseHandle(file);
      if (!ok)
        return false;

      // UE3 writes user configs as UTF-16LE; the shipped defaults are plain ASCII.
      if (raw.size() >= 2 && raw[0] == 0xFF && raw[1] == 0xFE) {
        out.clear();
        out.reserve(raw.size() / 2);
        for (size_t i = 2; i + 1 < raw.size(); i += 2) {
          const uint16_t unit = uint16_t(raw[i]) | uint16_t(uint16_t(raw[i + 1]) << 8);
          out.push_back(unit < 0x80 ? char(unit) : '?');
        }
      } else {
        out.assign(reinterpret_cast<const char*>(raw.data()), raw.size());
      }

      return true;
    }

    bool ngxParseIniSystemSettings(const std::string& text, NgxIniSystemSettings& out) {
      std::istringstream stream(text);
      std::string line;
      bool inSection = false;
      NgxIniSystemSettings parsed;

      while (std::getline(stream, line)) {
        const std::string trimmed = ngxTrimAscii(line);
        if (trimmed.empty() || trimmed[0] == ';')
          continue;

        if (trimmed[0] == '[') {
          if (inSection)
            break;
          inSection = ngxLowerAscii(trimmed) == "[systemsettings]";
          continue;
        }

        if (!inSection)
          continue;

        const size_t separator = trimmed.find('=');
        if (separator == std::string::npos)
          continue;

        const std::string key = ngxLowerAscii(ngxTrimAscii(trimmed.substr(0, separator)));
        const std::string value = ngxTrimAscii(trimmed.substr(separator + 1));
        if (value.empty())
          continue;

        try {
          if (key == "screenpercentage") {
            parsed.screenPercentage = std::stof(value);
            parsed.hasScreenPercentage = std::isfinite(parsed.screenPercentage);
          } else if (key == "upscalescreenpercentage") {
            parsed.hasUpscale = ngxParseIniBool(value, parsed.upscale);
          } else if (key == "fullscreen") {
            parsed.hasFullscreen = ngxParseIniBool(value, parsed.fullscreen);
          } else if (key == "resx") {
            parsed.resX = std::stoi(value);
          } else if (key == "resy") {
            parsed.resY = std::stoi(value);
          } else if (key == "maxmultisamples") {
            parsed.maxMultisamples = std::stoi(value);
            parsed.hasMaxMultisamples = true;
          }
        } catch (const std::exception&) {
          // A malformed entry only costs us that one cross-check.
        }
      }

      parsed.hasResolution = parsed.resX > 0 && parsed.resY > 0;
      parsed.found = parsed.hasScreenPercentage || parsed.hasResolution ||
                     parsed.hasMaxMultisamples;
      if (!parsed.found)
        return false;

      out = parsed;
      return true;
    }

    void ngxCollectEngineInis(const std::wstring& directory,
                              std::vector<std::wstring>& out) {
      constexpr size_t kMaxIniCandidates = 16;
      if (out.size() >= kMaxIniCandidates)
        return;

      WIN32_FIND_DATAW find = {};
      HANDLE search = ::FindFirstFileW((directory + L"\\Config\\*Engine.ini").c_str(), &find);
      if (search == INVALID_HANDLE_VALUE)
        return;

      do {
        if (find.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
          continue;
        out.push_back(directory + L"\\Config\\" + find.cFileName);
      } while (out.size() < kMaxIniCandidates && ::FindNextFileW(search, &find));

      ::FindClose(search);
    }

    // UE3 keeps its configs at <root>\<X>Game\Config\<Y>Engine.ini, with the live user copy
    // often under Documents. Only immediate subdirectories are enumerated, so this stays cheap.
    void ngxCollectEngineIniRoots(const std::wstring& root, uint32_t remainingDepth,
                                  std::vector<std::wstring>& out) {
      constexpr size_t kMaxIniCandidates = 16;
      ngxCollectEngineInis(root, out);
      if (remainingDepth == 0 || out.size() >= kMaxIniCandidates)
        return;

      WIN32_FIND_DATAW find = {};
      HANDLE search = ::FindFirstFileW((root + L"\\*").c_str(), &find);
      if (search == INVALID_HANDLE_VALUE)
        return;

      constexpr uint32_t kMaxSubdirectories = 64;
      uint32_t visited = 0;
      do {
        if (!(find.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
          continue;
        const std::wstring name = find.cFileName;
        if (name == L"." || name == L"..")
          continue;
        ngxCollectEngineIniRoots(root + L"\\" + name, remainingDepth - 1, out);
      } while (++visited < kMaxSubdirectories && out.size() < kMaxIniCandidates &&
               ::FindNextFileW(search, &find));

      ::FindClose(search);
    }

    NgxIniSystemSettings ngxReadIniSystemSettings(const std::wstring& exePath) {
      NgxIniSystemSettings result;
      if (exePath.empty())
        return result;

      std::vector<std::wstring> candidates;

      // The install tree: the exe lives under <root>\Binaries[\Win32], so walk up a few levels.
      std::wstring directory = exePath;
      for (uint32_t level = 0; level < 4; level++) {
        const size_t separator = directory.find_last_of(L"\\/");
        if (separator == std::wstring::npos || separator == 0)
          break;
        directory = directory.substr(0, separator);
        ngxCollectEngineIniRoots(directory, 1, candidates);
      }

      wchar_t userProfile[MAX_PATH] = {};
      if (::GetEnvironmentVariableW(L"USERPROFILE", userProfile, MAX_PATH) != 0) {
        ngxCollectEngineIniRoots(std::wstring(userProfile) + L"\\Documents\\My Games", 2,
                                 candidates);
      }

      for (const std::wstring& candidate : candidates) {
        std::string text;
        NgxIniSystemSettings parsed;
        if (ngxReadFileText(candidate, text) && ngxParseIniSystemSettings(text, parsed)) {
          // Prefer the copy that carries a resolution: that is the one the game writes back to.
          if (!result.found || (parsed.hasResolution && !result.hasResolution))
            result = parsed;
        }
      }

      return result;
    }

    // ---------------------------------------------------------------------------------------
    // Data-side recognition of an FSystemSettingsData record
    // ---------------------------------------------------------------------------------------

    bool ngxIsPlausibleScreenPercentage(float value) {
      return std::isfinite(value) && value > 0.0f &&
             value <= kUe3MaxPlausibleScreenPercentage;
    }

    // UE3 reads MaxMultisamples straight out of the ini as a plain INT and never constrains it,
    // so real installs carry values the D3D9 sample counts do not (Mirror's Edge ships 10).
    bool ngxIsPlausibleMaxMultisamples(int32_t value) {
      return value >= 0 && value <= kUe3MaxMultisamplesPlausibleLimit;
    }

    bool ngxIsPlausibleResolution(int32_t x, int32_t y) {
      return (x == 0 && y == 0) ||
             (x >= 320 && x <= 16384 && y >= 240 && y <= 16384);
    }

    int32_t ngxReadInt32(const uint8_t* data) {
      int32_t value = 0;
      std::memcpy(&value, data, sizeof(value));
      return value;
    }

    float ngxReadFloat(const uint8_t* data) {
      float value = 0.0f;
      std::memcpy(&value, data, sizeof(value));
      return value;
    }

    // FSystemSettingsDataWorldDetail opens the record: INT DetailMode then a long run of UBOOLs.
    bool ngxLooksLikeSettingsRecordHead(const uint8_t* data, size_t available) {
      if (available < sizeof(int32_t) * (1 + kUe3HeadBooleanRun))
        return false;

      const int32_t detailMode = ngxReadInt32(data);
      if (detailMode < 0 || detailMode > kUe3DetailModeMax)
        return false;

      for (uint32_t i = 1; i <= kUe3HeadBooleanRun; i++) {
        const int32_t value = ngxReadInt32(data + i * sizeof(int32_t));
        if (value != 0 && value != 1)
          return false;
      }

      return true;
    }

    // The record tail, with the resolution left loose so the Defaults[] copies (populated from
    // the compat ini rather than the live mode) match as well as the live record does.
    bool ngxLooksLikeSettingsRecordTail(const uint8_t* tail, size_t available) {
      if (available < kUe3SettingsDataTailSize)
        return false;

      const float screenPercentage = ngxReadFloat(tail);
      const int32_t upscale = ngxReadInt32(tail + kUe3UpscaleFromScreenPercentage);
      const int32_t resX = ngxReadInt32(tail + kUe3ResXFromScreenPercentage);
      const int32_t resY = ngxReadInt32(tail + kUe3ResYFromScreenPercentage);
      const int32_t fullscreen = ngxReadInt32(tail + kUe3FullscreenFromScreenPercentage);
      const int32_t maxMultisamples = ngxReadInt32(tail + kUe3MaxMultisamplesFromScreenPercentage);

      return ngxIsPlausibleScreenPercentage(screenPercentage) &&
             (upscale == 0 || upscale == 1) &&
             (fullscreen == 0 || fullscreen == 1) &&
             ngxIsPlausibleMaxMultisamples(maxMultisamples) &&
             ngxIsPlausibleResolution(resX, resY);
    }

    // ---------------------------------------------------------------------------------------
    // x86 operand classification
    //
    // Sites are accepted on instruction form rather than on surrounding byte context: only the
    // encodings that *read* one of these fields are patched, so the engine's own writes (and the
    // config-facing consumers, which all reach the fields through `this`) keep the real values.
    // ---------------------------------------------------------------------------------------

    bool ngxIsAbsoluteModrm(uint8_t modrm) {
      return (modrm & 0xC7) == 0x05;
    }

    bool ngxIsBaseDisp32Modrm(uint8_t modrm) {
      return (modrm & 0xC0) == 0x80 && (modrm & 0x07) != 0x04;
    }

    uint8_t ngxModrmReg(uint8_t modrm) {
      return uint8_t((modrm >> 3) & 0x07);
    }

    uint8_t ngxModrmRm(uint8_t modrm) {
      return uint8_t(modrm & 0x07);
    }

    bool ngxIsAbsoluteFloatReadSite(const uint8_t* bytes, size_t operandOffset) {
      if (operandOffset >= 4) {
        const uint8_t modrm = bytes[operandOffset - 1];
        // movss/addss/mulss/subss/divss xmm, m32. 0x11 (movss m32, xmm) is a store.
        if (bytes[operandOffset - 4] == 0xF3 && bytes[operandOffset - 3] == 0x0F &&
            ngxIsAbsoluteModrm(modrm)) {
          const uint8_t op = bytes[operandOffset - 2];
          if (op == 0x10 || op == 0x58 || op == 0x59 || op == 0x5C || op == 0x5E)
            return true;
        }
      }

      if (operandOffset >= 3) {
        const uint8_t modrm = bytes[operandOffset - 1];
        const uint8_t op = bytes[operandOffset - 2];
        // ucomiss/comiss xmm, m32; a 0x66 prefix would make these the f64 variants.
        if (bytes[operandOffset - 3] == 0x0F && (op == 0x2E || op == 0x2F) &&
            ngxIsAbsoluteModrm(modrm) &&
            (operandOffset < 4 || bytes[operandOffset - 4] != 0x66))
          return true;
      }

      if (operandOffset >= 2) {
        const uint8_t modrm = bytes[operandOffset - 1];
        const uint8_t op = bytes[operandOffset - 2];
        if (ngxIsAbsoluteModrm(modrm)) {
          // x87 arithmetic/compare against m32; D9 /0 is fld, D9 /3 (fstp) is a store.
          if (op == 0xD8)
            return true;
          if (op == 0xD9 && ngxModrmReg(modrm) == 0)
            return true;
        }
      }

      return false;
    }

    bool ngxIsAbsoluteIntReadSite(const uint8_t* bytes, size_t operandOffset) {
      if (operandOffset >= 1 && bytes[operandOffset - 1] == 0xA1)
        return true;  // mov eax, [disp32]

      if (operandOffset >= 2) {
        const uint8_t modrm = bytes[operandOffset - 1];
        const uint8_t op = bytes[operandOffset - 2];
        if (!ngxIsAbsoluteModrm(modrm))
          return false;

        // mov r32, m32 / cmp / test / xor / or, all reading the memory operand.
        if (op == 0x8B || op == 0x3B || op == 0x39 || op == 0x85 || op == 0x33 || op == 0x0B)
          return true;
        // Group 1 immediate forms: only /7 (cmp) leaves the memory operand untouched.
        if ((op == 0x83 || op == 0x81) && ngxModrmReg(modrm) == 7)
          return true;
        // test m32, imm32.
        if (op == 0xF7 && ngxModrmReg(modrm) == 0)
          return true;
      }

      return false;
    }

    // Distinguishes a renderer-facing read from a read that just copies the field somewhere.
    bool ngxReadsPercentScaleConstantNearby(HANDLE proc, const uint8_t* bytes,
                                            size_t operandOffset, size_t length) {
      if (length < sizeof(uint32_t))
        return false;

      const size_t begin = operandOffset > kNgxPercentScaleConstantWindow
        ? operandOffset - kNgxPercentScaleConstantWindow
        : 0;
      const size_t end = std::min(length - sizeof(uint32_t),
                                  operandOffset + kNgxPercentScaleConstantWindow);

      for (size_t offset = begin; offset < end; offset++) {
        if (offset == operandOffset || !ngxIsAbsoluteFloatReadSite(bytes, offset))
          continue;

        uint32_t constantAddress = 0;
        std::memcpy(&constantAddress, bytes + offset, sizeof(constantAddress));

        float constantValue = 0.0f;
        if (ngxReadProcessExact(proc, uintptr_t(constantAddress), constantValue) &&
            (constantValue == 100.0f || constantValue == 0.01f))
          return true;
      }

      return false;
    }

    struct NgxRelativeOperand {
      size_t operandOffset = 0;
      uint32_t displacement = 0;
      uint8_t baseRegister = 0;
      bool isFloat = false;
      bool valid = false;
    };

    // Decodes only the [reg+disp32] read forms UE3 emits for these two fields; anything else
    // simply does not match, which is what keeps the paired-window search precise.
    NgxRelativeOperand ngxDecodeRelativeRead(const uint8_t* bytes, size_t offset,
                                             size_t available) {
      NgxRelativeOperand out;

      const auto emit = [&](size_t operandOffset, bool isFloat, uint8_t modrm) {
        if (operandOffset + sizeof(uint32_t) > available)
          return;
        std::memcpy(&out.displacement, bytes + operandOffset, sizeof(out.displacement));
        out.operandOffset = operandOffset;
        out.baseRegister = ngxModrmRm(modrm);
        out.isFloat = isFloat;
        out.valid = true;
      };

      if (offset + 3 <= available && bytes[offset] == 0x0F &&
          (bytes[offset + 1] == 0x2E || bytes[offset + 1] == 0x2F) &&
          ngxIsBaseDisp32Modrm(bytes[offset + 2]) &&
          (offset == 0 || bytes[offset - 1] != 0x66)) {
        emit(offset + 3, true, bytes[offset + 2]);
      } else if (offset + 4 <= available && bytes[offset] == 0xF3 &&
                 bytes[offset + 1] == 0x0F && bytes[offset + 2] == 0x10 &&
                 ngxIsBaseDisp32Modrm(bytes[offset + 3])) {
        emit(offset + 4, true, bytes[offset + 3]);
      } else if (offset + 2 <= available &&
                 (bytes[offset] == 0x8B || bytes[offset] == 0x3B ||
                  bytes[offset] == 0x39 || bytes[offset] == 0x85) &&
                 ngxIsBaseDisp32Modrm(bytes[offset + 1])) {
        emit(offset + 2, false, bytes[offset + 1]);
      } else if (offset + 2 <= available && bytes[offset] == 0x83 &&
                 ngxIsBaseDisp32Modrm(bytes[offset + 1]) &&
                 ngxModrmReg(bytes[offset + 1]) == 7) {
        emit(offset + 2, false, bytes[offset + 1]);
      }

      return out;
    }

    struct NgxGameSettingsScanInputs {
      uint32_t backBufferWidth = 0;
      uint32_t backBufferHeight = 0;
      bool windowed = false;
      // Escape hatch: when set, discovery is skipped in favour of this module-relative address
      // (still subject to the shape checks and the behavioural probe).
      uint32_t forcedScreenPercentageRva = 0;
      NgxIniSystemSettings ini;
    };

    // The filesystem search is worth doing once per process, not once per scan retry.
    const NgxIniSystemSettings& ngxCachedIniSystemSettings(const std::wstring& exePath) {
      static const NgxIniSystemSettings cached = ngxReadIniSystemSettings(exePath);
      return cached;
    }

    // Finds and validates the renderer-facing settings operands in a 32-bit game module.
    // Candidates the behavioural probe has already disproven are skipped, so a failed probe can
    // simply rescan for the next best match.
    NgxLocateResult ngxLocateGameSettings(HANDLE proc, uintptr_t base, uint32_t moduleSize,
                                          const NgxGameSettingsScanInputs& inputs,
                                          const std::vector<uintptr_t>& rejectedCandidates,
                                          NgxLocatedGameSettings& outSettings) {
      outSettings = {};

      if (inputs.backBufferWidth == 0 || inputs.backBufferHeight == 0)
        return NgxLocateResult::Incomplete;

      uint8_t headers[0x1000];
      SIZE_T bytesRead = 0;
      if (!::ReadProcessMemory(proc, reinterpret_cast<LPCVOID>(base), headers, sizeof(headers), &bytesRead) ||
          bytesRead < sizeof(IMAGE_DOS_HEADER))
        return NgxLocateResult::Incomplete;

      outSettings.missReason = "not a 32-bit PE module";

      const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(headers);
      if (dos->e_magic != IMAGE_DOS_SIGNATURE)
        return NgxLocateResult::Miss;

      const uint32_t ntOffset = uint32_t(dos->e_lfanew);
      if (ntOffset + sizeof(IMAGE_NT_HEADERS32) > sizeof(headers))
        return NgxLocateResult::Miss;

      const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS32*>(headers + ntOffset);
      if (nt->Signature != IMAGE_NT_SIGNATURE ||
          nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR32_MAGIC)
        return NgxLocateResult::Miss;

      outSettings.missReason = "malformed PE section table";

      // IMAGE_NT_HEADERS32 = DWORD Signature; IMAGE_FILE_HEADER; IMAGE_OPTIONAL_HEADER32.
      // The section table follows the optional header (whose size is declared, so this is
      // robust to layout differences).
      const uint32_t sectionTableOffset =
        ntOffset + uint32_t(sizeof(uint32_t) + sizeof(IMAGE_FILE_HEADER)) + nt->FileHeader.SizeOfOptionalHeader;
      const uint32_t sectionCount = nt->FileHeader.NumberOfSections;
      if (sectionCount == 0 || sectionCount > 96)
        return NgxLocateResult::Miss;

      struct SectionRange {
        uintptr_t address = 0;
        uint32_t size = 0;
        bool executable = false;
      };
      std::vector<SectionRange> sectionRanges;

      for (uint32_t i = 0; i < sectionCount; i++) {
        const uint32_t entryOffset = sectionTableOffset + i * uint32_t(sizeof(IMAGE_SECTION_HEADER));
        if (entryOffset + sizeof(IMAGE_SECTION_HEADER) > sizeof(headers))
          return NgxLocateResult::Incomplete;

        const auto* section = reinterpret_cast<const IMAGE_SECTION_HEADER*>(headers + entryOffset);
        const DWORD characteristics = section->Characteristics;
        const bool executable = (characteristics & IMAGE_SCN_MEM_EXECUTE) != 0;
        // GSystemSettings has a constructor, so it lands in an initialised or zero-filled
        // writable data section; both are committed and readable at this point.
        const bool writableData = !executable &&
                                  (characteristics & IMAGE_SCN_MEM_WRITE) != 0 &&
                                  (characteristics & IMAGE_SCN_MEM_READ) != 0;
        if (!executable && !writableData)
          continue;

        uint32_t size = section->Misc.VirtualSize != 0 ? section->Misc.VirtualSize : section->SizeOfRawData;
        if (moduleSize != 0) {
          if (section->VirtualAddress >= moduleSize)
            return NgxLocateResult::Incomplete;
          size = std::min(size,
                          moduleSize - uint32_t(section->VirtualAddress));
        }
        if (size < kUe3SettingsDataTailSize)
          continue;

        sectionRanges.push_back({ base + section->VirtualAddress, size, executable });
      }

      struct SectionCopy {
        uintptr_t address = 0;
        std::vector<uint8_t> bytes;
      };

      const auto readSections = [&](bool executable, std::vector<SectionCopy>& out) {
        for (const SectionRange& range : sectionRanges) {
          if (range.executable != executable)
            continue;

          SectionCopy copy;
          copy.address = range.address;
          copy.bytes.resize(range.size);
          SIZE_T sectionRead = 0;
          if (!::ReadProcessMemory(proc, reinterpret_cast<LPCVOID>(range.address),
                                   copy.bytes.data(), range.size, &sectionRead) ||
              sectionRead < range.size)
            return false;
          out.push_back(std::move(copy));
        }
        return !out.empty();
      };

      // The code sections are two orders of magnitude larger than the data ones, so they are
      // only read once the data side has something worth looking for. Scans repeat until the
      // engine has populated GSystemSettings, and paying for the code copy every retry would
      // be a visible hitch each second.
      std::vector<SectionCopy> dataSections;
      if (!readSections(false, dataSections))
        return NgxLocateResult::Incomplete;

      // ---- Resolution-anchored candidate search -------------------------------------------
      struct Candidate {
        const SectionCopy* section = nullptr;
        size_t offset = 0;              // ScreenPercentage, within the section
        bool strictTail = false;
        bool fullscreenAgrees = false;
        int32_t score = 0;
        float screenPercentage = 100.0f;
        int32_t upscale = 1;
        int32_t maxMultisamples = 0;
        uint32_t defaultsStride = 0;
        uint32_t defaultsRepeats = 0;
      };
      // Relaxed hits are only consulted when nothing matched the stock tail layout, so an
      // unusual build cannot bury the obvious answer under near-misses.
      std::vector<Candidate> candidates;
      std::vector<Candidate> relaxedCandidates;
      constexpr size_t kMaxCandidates = 64;

      const int32_t expectedResX = int32_t(inputs.backBufferWidth);
      const int32_t expectedResY = int32_t(inputs.backBufferHeight);
      const int32_t expectedFullscreen = inputs.windowed ? 0 : 1;

      const auto isRejected = [&](uintptr_t address) {
        return std::find(rejectedCandidates.begin(), rejectedCandidates.end(), address) !=
               rejectedCandidates.end();
      };

      for (const SectionCopy& section : dataSections) {
        const uint8_t* bytes = section.bytes.data();
        const size_t length = section.bytes.size();

        if (inputs.forcedScreenPercentageRva != 0) {
          const uintptr_t forcedAddress = base + inputs.forcedScreenPercentageRva;
          if (forcedAddress < section.address ||
              forcedAddress + kUe3UpscaleFromScreenPercentage + sizeof(int32_t) >
                section.address + length)
            continue;

          const size_t offset = size_t(forcedAddress - section.address);
          // A pinned address the probe already disproved is not worth retrying.
          if (isRejected(forcedAddress))
            continue;

          Candidate candidate;
          candidate.section = &section;
          candidate.offset = offset;
          candidate.strictTail = ngxLooksLikeSettingsRecordTail(bytes + offset, length - offset);
          candidate.screenPercentage = ngxReadFloat(bytes + offset);
          candidate.upscale = ngxReadInt32(bytes + offset + kUe3UpscaleFromScreenPercentage);
          if (candidate.strictTail) {
            candidate.maxMultisamples =
              ngxReadInt32(bytes + offset + kUe3MaxMultisamplesFromScreenPercentage);
          }
          candidates.push_back(candidate);
          continue;
        }

        for (size_t resOffset = 0; resOffset + 2 * sizeof(int32_t) <= length;
             resOffset += sizeof(int32_t)) {
          if (ngxReadInt32(bytes + resOffset) != expectedResX ||
              ngxReadInt32(bytes + resOffset + sizeof(int32_t)) != expectedResY)
            continue;

          // Strict stock layout first: ScreenPercentage sits two dwords ahead of ResX, with
          // bFullscreen and MaxMultiSamples behind ResY.
          if (resOffset >= kUe3ResXFromScreenPercentage) {
            const size_t offset = resOffset - kUe3ResXFromScreenPercentage;
            if (ngxLooksLikeSettingsRecordTail(bytes + offset, length - offset) &&
                !isRejected(section.address + offset)) {
              Candidate candidate;
              candidate.section = &section;
              candidate.offset = offset;
              candidate.strictTail = true;
              candidate.screenPercentage = ngxReadFloat(bytes + offset);
              candidate.upscale = ngxReadInt32(bytes + offset + kUe3UpscaleFromScreenPercentage);
              candidate.maxMultisamples =
                ngxReadInt32(bytes + offset + kUe3MaxMultisamplesFromScreenPercentage);
              // Borderless windowed makes the engine's own idea of fullscreen disagree with
              // D3D9's, so agreement raises confidence rather than gating the candidate.
              candidate.fullscreenAgrees =
                ngxReadInt32(bytes + offset + kUe3FullscreenFromScreenPercentage) ==
                expectedFullscreen;
              if (candidates.size() < kMaxCandidates)
                candidates.push_back(candidate);
              continue;
            }
          }

          // Relaxed pass for builds that do not use the stock tail adjacency: only the
          // ScreenPercentage/bUpscaleScreenPercentage pair itself is required, and the probe
          // carries the burden of proof.
          for (uint32_t back = kUe3ResXFromScreenPercentage + sizeof(int32_t);
               back <= kUe3ResXFromScreenPercentage + kUe3MaxRelaxedScreenPercentageBacktrack &&
               relaxedCandidates.size() < kMaxCandidates;
               back += sizeof(int32_t)) {
            if (resOffset < back)
              break;
            const size_t offset = resOffset - back;
            const int32_t upscale = ngxReadInt32(bytes + offset + kUe3UpscaleFromScreenPercentage);
            if (!ngxIsPlausibleScreenPercentage(ngxReadFloat(bytes + offset)) ||
                (upscale != 0 && upscale != 1) ||
                isRejected(section.address + offset))
              continue;

            Candidate candidate;
            candidate.section = &section;
            candidate.offset = offset;
            candidate.screenPercentage = ngxReadFloat(bytes + offset);
            candidate.upscale = upscale;
            relaxedCandidates.push_back(candidate);
          }
        }
      }

      if (candidates.empty())
        candidates = std::move(relaxedCandidates);

      if (candidates.empty()) {
        outSettings.missReason =
          "no FSystemSettings record in writable data matched the live resolution";
        return NgxLocateResult::Miss;
      }

      std::vector<SectionCopy> executableSections;
      if (!readSections(true, executableSections))
        return NgxLocateResult::Incomplete;

      // ---- Scoring: structural repeats and the optional ini cross-check --------------------
      for (Candidate& candidate : candidates) {
        const uint8_t* bytes = candidate.section->bytes.data();
        const size_t length = candidate.section->bytes.size();

        candidate.score = (candidate.strictTail ? 100 : 0) +
                          (candidate.fullscreenAgrees ? 20 : 0);

        // FSystemSettings embeds Defaults[FSL_LevelCount] behind the live record, so the tail
        // shape repeats at a constant stride equal to sizeof(FSystemSettingsData).
        uint32_t bestStride = 0;
        uint32_t bestRepeats = 0;
        if (candidate.strictTail) {
          // Bounded: the pairwise stride search below is quadratic in the number of hits.
          constexpr size_t kMaxTailHits = 64;
          std::vector<size_t> tailHits;
          const size_t searchEnd =
            std::min(length, candidate.offset + kUe3DefaultsSearchWindow);
          for (size_t offset = candidate.offset + sizeof(int32_t);
               offset + kUe3SettingsDataTailSize <= searchEnd && tailHits.size() < kMaxTailHits;
               offset += sizeof(int32_t)) {
            if (ngxLooksLikeSettingsRecordTail(bytes + offset, length - offset))
              tailHits.push_back(offset);
          }

          for (size_t i = 0; i < tailHits.size(); i++) {
            for (size_t j = i + 1; j < tailHits.size(); j++) {
              const size_t stride = tailHits[j] - tailHits[i];
              if (stride < kUe3MinSettingsDataSize || stride > kUe3MaxSettingsDataSize)
                continue;

              uint32_t repeats = 1;
              for (size_t next = tailHits[j] + stride;
                   next + kUe3SettingsDataTailSize <= searchEnd; next += stride) {
                if (!ngxLooksLikeSettingsRecordTail(bytes + next, length - next))
                  break;
                repeats++;
              }

              if (repeats > bestRepeats) {
                bestRepeats = repeats;
                bestStride = uint32_t(stride);
              }
            }
          }
        }

        if (bestRepeats >= kUe3MinDefaultsRepeats) {
          candidate.defaultsStride = bestStride;
          candidate.defaultsRepeats = bestRepeats;
          candidate.score += 20 * int32_t(std::min<uint32_t>(bestRepeats, 5u));
        }

        const NgxIniSystemSettings& ini = inputs.ini;
        if (ini.found) {
          if (ini.hasScreenPercentage &&
              std::fabs(ini.screenPercentage - candidate.screenPercentage) < 0.01f)
            candidate.score += 50;
          if (ini.hasUpscale && ini.upscale == candidate.upscale)
            candidate.score += 15;
          if (candidate.strictTail && ini.hasMaxMultisamples &&
              ini.maxMultisamples == candidate.maxMultisamples)
            candidate.score += 25;
        }
      }

      std::stable_sort(candidates.begin(), candidates.end(),
                       [](const Candidate& a, const Candidate& b) {
                         return a.score > b.score;
                       });

      const Candidate& chosen = candidates.front();
      const uint8_t* dataBytes = chosen.section->bytes.data();
      const size_t dataLength = chosen.section->bytes.size();
      const uintptr_t screenPercentageAddress = chosen.section->address + chosen.offset;

      outSettings.screenPercentageAddress = screenPercentageAddress;
      outSettings.currentScreenPercentage = chosen.screenPercentage;
      outSettings.currentUpscaleScreenPercentage = chosen.upscale;
      outSettings.currentMaxMultisamples = chosen.maxMultisamples;
      outSettings.settingsDataStride = chosen.defaultsStride;
      outSettings.defaultsRepeats = chosen.defaultsRepeats;
      outSettings.strictTailLayout = chosen.strictTail;
      outSettings.iniConfirmed =
        inputs.ini.found && inputs.ini.hasScreenPercentage &&
        std::fabs(inputs.ini.screenPercentage - chosen.screenPercentage) < 0.01f;

      // ---- Code scan ----------------------------------------------------------------------
      const uint32_t screenPercentageOperand = uint32_t(screenPercentageAddress);
      const uint32_t upscaleOperand =
        uint32_t(screenPercentageAddress + kUe3UpscaleFromScreenPercentage);

      struct AbsoluteSite {
        uintptr_t address = 0;
        uint32_t original = 0;
        bool upscale = false;
      };
      std::vector<AbsoluteSite> absoluteSites;
      std::vector<uintptr_t> seedAddresses;

      struct RelativeWindow {
        uint32_t structOffset = 0;
        NgxOperandSite floatSite;
        NgxOperandSite intSite;
      };
      std::vector<RelativeWindow> relativeWindows;
      constexpr size_t kRelativePairWindow = 48;

      // Two float reads in one function can pair with the same integer read, so sites are
      // deduplicated by address rather than trusting one entry per match.
      const auto appendUniqueSite = [](std::vector<NgxOperandSite>& sites,
                                       const NgxOperandSite& site) {
        for (const NgxOperandSite& existing : sites) {
          if (existing.address == site.address)
            return;
        }
        sites.push_back(site);
      };

      for (const SectionCopy& section : executableSections) {
        const uint8_t* bytes = section.bytes.data();
        const size_t length = section.bytes.size();

        for (size_t offset = 0; offset + sizeof(uint32_t) <= length; offset++) {
          uint32_t operand = 0;
          std::memcpy(&operand, bytes + offset, sizeof(operand));

          if (operand == screenPercentageOperand &&
              ngxIsAbsoluteFloatReadSite(bytes, offset)) {
            if (ngxReadsPercentScaleConstantNearby(proc, bytes, offset, length))
              seedAddresses.push_back(section.address + offset);
            if (absoluteSites.size() < kNgxMaxRawOperandSites)
              absoluteSites.push_back({ section.address + offset, operand, false });
          } else if (operand == upscaleOperand && ngxIsAbsoluteIntReadSite(bytes, offset)) {
            if (absoluteSites.size() < kNgxMaxRawOperandSites)
              absoluteSites.push_back({ section.address + offset, operand, true });
          }
        }
      }

      // Without a scale-factor conversion there is no renderer-facing reader to redirect, and
      // patching the copies alone would corrupt whatever they feed.
      if (seedAddresses.empty()) {
        outSettings.missReason =
          "the record was found but no code converts ScreenPercentage into a scale factor";
        return NgxLocateResult::Miss;
      }

      const auto nearSeed = [&](uintptr_t address) {
        for (const uintptr_t seed : seedAddresses) {
          const uintptr_t distance = address > seed ? address - seed : seed - address;
          if (distance <= kNgxSettingsCodeClusterRadius)
            return true;
        }
        return false;
      };

      for (const AbsoluteSite& site : absoluteSites) {
        if (!nearSeed(site.address))
          continue;
        appendUniqueSite(site.upscale ? outSettings.directUpscaleScreenPercentageSites
                                      : outSettings.directScreenPercentageSites,
                         { site.address, site.original });
      }

      // Without ScaleScreenCoords' absolute reader nothing can shrink the view.
      if (outSettings.directScreenPercentageSites.empty() ||
          outSettings.directScreenPercentageSites.size() > kNgxMaxOperandSites ||
          outSettings.directUpscaleScreenPercentageSites.size() > kNgxMaxOperandSites) {
        outSettings.missReason = "no usable absolute ScreenPercentage readers";
        return NgxLocateResult::Miss;
      }

      // FSystemSettings::NeedsUpscale() reads both fields through `this`, and
      // UnScaleScreenCoords provably inlines it right beside an absolute ScreenPercentage read.
      // So the engine's own displacement is simply the one that appears next to a seed - which
      // is what identifies GSystemSettings without having to recognise the *head* of the
      // record. The head is the part that genuinely varies: the FSystemSettingsData sub-struct
      // split (DetailMode followed by a run of UBOOLs) only exists in later UE3 versions, and
      // Mirror's Edge predates it.
      for (const SectionCopy& section : executableSections) {
        const uint8_t* bytes = section.bytes.data();
        const size_t length = section.bytes.size();

        for (const uintptr_t seed : seedAddresses) {
          if (seed < section.address || seed - section.address >= length)
            continue;

          const size_t seedOffset = size_t(seed - section.address);
          const size_t begin = seedOffset > kNgxSettingsCodeClusterRadius
            ? seedOffset - kNgxSettingsCodeClusterRadius
            : 0;
          const size_t end = std::min(length, seedOffset + kNgxSettingsCodeClusterRadius);

          for (size_t offset = begin;
               offset < end && relativeWindows.size() < kNgxMaxOperandSites; offset++) {
            const NgxRelativeOperand decoded = ngxDecodeRelativeRead(bytes, offset, length);
            if (!decoded.valid || !decoded.isFloat ||
                decoded.displacement < kUe3MinSettingsDataSize ||
                decoded.displacement > kUe3MaxSettingsDataSize ||
                decoded.displacement > chosen.offset)
              continue;

            const uintptr_t floatSiteAddress = section.address + decoded.operandOffset;
            bool alreadyKnown = false;
            for (const RelativeWindow& window : relativeWindows)
              alreadyKnown |= window.floatSite.address == floatSiteAddress;
            if (alreadyKnown)
              continue;

            const size_t pairBegin = offset > kRelativePairWindow ? offset - kRelativePairWindow : 0;
            const size_t pairEnd = std::min(length, offset + kRelativePairWindow);
            for (size_t pairOffset = pairBegin; pairOffset < pairEnd; pairOffset++) {
              const NgxRelativeOperand paired = ngxDecodeRelativeRead(bytes, pairOffset, length);
              if (!paired.valid || paired.isFloat ||
                  paired.baseRegister != decoded.baseRegister ||
                  paired.displacement != decoded.displacement + kUe3UpscaleFromScreenPercentage)
                continue;

              relativeWindows.push_back({
                decoded.displacement,
                { floatSiteAddress, decoded.displacement },
                { section.address + paired.operandOffset, paired.displacement },
              });
              break;
            }
          }
        }
      }

      // The displacement backed by the most windows wins; the Defaults[] stride and the record
      // head (when this engine version has one) are independent derivations that break ties.
      const uint32_t strideDerivedOffset =
        chosen.defaultsStride > kUe3SettingsDataTailSize
          ? chosen.defaultsStride - kUe3SettingsDataTailSize
          : 0;
      uint32_t bestStructOffset = 0;
      size_t bestWindowCount = 0;
      int32_t bestTieBreak = -1;
      for (const RelativeWindow& candidateWindow : relativeWindows) {
        const uint32_t structOffset = candidateWindow.structOffset;
        size_t count = 0;
        for (const RelativeWindow& window : relativeWindows)
          count += window.structOffset == structOffset ? 1 : 0;

        const size_t headOffset = chosen.offset - structOffset;
        const int32_t tieBreak =
          (structOffset == strideDerivedOffset ? 2 : 0) +
          (ngxLooksLikeSettingsRecordHead(dataBytes + headOffset, dataLength - headOffset) ? 1 : 0);

        if (count > bestWindowCount ||
            (count == bestWindowCount && tieBreak > bestTieBreak)) {
          bestWindowCount = count;
          bestTieBreak = tieBreak;
          bestStructOffset = structOffset;
        }
      }

      if (bestWindowCount != 0 && bestStructOffset <= chosen.offset) {
        outSettings.screenPercentageStructOffset = bestStructOffset;
        outSettings.systemSettingsAddress = screenPercentageAddress - bestStructOffset;
        for (const RelativeWindow& window : relativeWindows) {
          if (window.structOffset != bestStructOffset)
            continue;
          appendUniqueSite(outSettings.relativeScreenPercentageSites, window.floatSite);
          appendUniqueSite(outSettings.relativeUpscaleScreenPercentageSites, window.intSite);
        }
      }

      return NgxLocateResult::Found;
    }

  }

  void D3D9Rtx::bootstrapNgxPassthroughUpscaler(uint32_t displayWidth, uint32_t displayHeight) {
    if (m_ngxPassthroughBootstrapped) {
      return;
    }

    if (!m_frameOptions.valid) {
      refreshFrameOptionCache();
    }

    // Preset sync already runs in RtxInitializer; here we only prime XeSS input resolution
    // (what the Remix UI does via getXeSSInputResolution) once the display size is known.
    if (displayWidth > 0 && displayHeight > 0) {
      m_parent->GetDXVKDevice()->getCommon()->metaNgxPassthrough()
        .syncXeSSInputResolution(displayWidth, displayHeight);
    }

    m_ngxPassthroughBootstrapped = true;
  }

  bool D3D9Rtx::ngxRuntimeOwnsUpscale() const {
    return m_ngxGameSettingsRedirectsValid && m_ngxRuntimeOwnedUpscale;
  }

  void D3D9Rtx::mirrorNgxLiveScreenPercentage() {
    float liveScreenPercentage = 0.0f;
    int32_t liveUpscale = 0;
    bool rollbackVerified = true;
    const bool mirrored =
      ngxReadProcessExact(m_ngxGameProcess, m_ngxScreenPercentageRemoteAddr,
                          liveScreenPercentage) &&
      ngxReadProcessExact(m_ngxGameProcess,
                          m_ngxScreenPercentageRemoteAddr + sizeof(float),
                          liveUpscale) &&
      ngxUpdateScreenPercentageShadow(
        m_ngxGameProcess, m_ngxGameSettingsShadowRemoteAddr,
        liveScreenPercentage, liveUpscale, &rollbackVerified);

    if (!mirrored) {
      ONCE(Logger::warn("[RTX NGX Passthrough] Could not mirror the game's live "
                        "ScreenPercentage/UpscaleScreenPercentage into the renderer shadow; "
                        "retaining the last effective values."));
      if (!rollbackVerified) {
        ONCE(Logger::err("[RTX NGX Passthrough] ScreenPercentage shadow rollback could "
                         "not be verified after the failed update."));
      }
    } else {
      if (m_ngxScreenPercentageDriven) {
        Logger::info(str::format(
          "[RTX NGX Passthrough] ScreenPercentage override released; renderer follows the "
          "game's live value ", liveScreenPercentage, " (UpscaleScreenPercentage ",
          liveUpscale, ")."));
      }
      m_ngxScreenPercentageDriven = false;
      m_ngxScreenPercentageLastLogged = 0.0f;
    }

    m_ngxGameScreenPercentage =
      (mirrored && std::isfinite(liveScreenPercentage) && liveScreenPercentage > 0.0f)
        ? liveScreenPercentage
        : 0.0f;
  }

  bool D3D9Rtx::writeNgxScreenPercentageShadow(float screenPercentage) {
    // The engine stretch needs NeedsUpscale() to see the reduced percentage.
    // A runtime-owned upscale leaves that flag off and composites the reduced rect as-is.
    const int32_t upscale = m_ngxRuntimeOwnedUpscale ? 0 : 1;

    bool rollbackVerified = true;
    if (ngxUpdateScreenPercentageShadow(
          m_ngxGameProcess, m_ngxGameSettingsShadowRemoteAddr,
          screenPercentage, upscale, &rollbackVerified))
      return true;

    ONCE(Logger::warn("[RTX NGX Passthrough] Could not transactionally update the "
                      "renderer-isolated ScreenPercentage shadow; it will be retried."));
    if (!rollbackVerified) {
      ONCE(Logger::err("[RTX NGX Passthrough] ScreenPercentage shadow rollback could "
                       "not be verified after the failed update."));
    }
    return false;
  }

  void D3D9Rtx::updateNgxSettingsProbe() {
    if (m_ngxSettingsProbeState != NgxSettingsProbeState::Pending)
      return;

    const uint32_t backBufferWidth =
      m_activePresentParams.has_value() ? m_activePresentParams->BackBufferWidth : 0;
    const uint32_t backBufferHeight =
      m_activePresentParams.has_value() ? m_activePresentParams->BackBufferHeight : 0;

    if (backBufferWidth != 0 && backBufferHeight != 0 && m_ngxSceneViewportValid) {
      const int32_t expectedWidth =
        int32_t(float(backBufferWidth) * m_ngxSettingsProbeValue / 100.0f);
      const int32_t expectedHeight =
        int32_t(float(backBufferHeight) * m_ngxSettingsProbeValue / 100.0f);
      const int32_t widthError = int32_t(m_ngxSceneViewport.Width) - expectedWidth;
      const int32_t heightError = int32_t(m_ngxSceneViewport.Height) - expectedHeight;

      if (widthError >= -kNgxScreenPercentageProbeTolerance &&
          widthError <= kNgxScreenPercentageProbeTolerance &&
          heightError >= -kNgxScreenPercentageProbeTolerance &&
          heightError <= kNgxScreenPercentageProbeTolerance) {
        m_ngxSettingsProbeState = NgxSettingsProbeState::Confirmed;
        Logger::info(str::format(
          "[RTX NGX Passthrough] Game settings redirects confirmed: driving ",
          m_ngxSettingsProbeValue, "% produced a ", m_ngxSceneViewport.Width, "x",
          m_ngxSceneViewport.Height, " scene viewport. Upscale owner: ",
          (m_ngxRuntimeOwnedUpscale ? "runtime." : "engine.")));
        return;
      }
    }

    if (m_ue3FrameCounter - m_ngxSettingsProbeStartFrame < kNgxScreenPercentageProbeFrames)
      return;

    Logger::warn(str::format(
      "[RTX NGX Passthrough] The candidate ScreenPercentage at 0x", std::hex,
      m_ngxScreenPercentageRemoteAddr, std::dec, " did not move the game's scene viewport within ",
      kNgxScreenPercentageProbeFrames, " frames; rejecting it."));

    m_ngxSettingsProbeState = NgxSettingsProbeState::Failed;
    m_ngxRejectedSettingsCandidates.push_back(m_ngxScreenPercentageRemoteAddr);
    restoreNgxGameSettingsRedirects();

    // Only a clean rollback leaves the process in a state where another candidate can be tried.
    constexpr size_t kMaxRejectedCandidates = 4;
    if (m_ngxGameSettingsShadowRemoteAddr == 0 &&
        m_ngxRejectedSettingsCandidates.size() < kMaxRejectedCandidates) {
      if (m_ngxGameProcessOwned && m_ngxGameProcess != nullptr)
        ::CloseHandle(m_ngxGameProcess);
      m_ngxGameProcess = nullptr;
      m_ngxGameProcessOwned = false;
      m_ngxGameProcessId = 0;
      m_ngxScreenPercentageRemoteAddr = 0;
      m_ngxGameScreenPercentage = 0.0f;
      m_ngxScreenPercentageScanDone = false;
      m_ngxSettingsProbeState = NgxSettingsProbeState::Idle;
    } else {
      m_ngxScreenPercentageScanDone = true;
    }
  }

  void D3D9Rtx::applyNgxPassthroughScreenPercentage() {
    const bool driving = RtxNgxPassthrough::driveGameScreenPercentage();

    // The camera gate needs ScreenPercentage even when automatic driving is disabled.
    if (m_ngxScreenPercentageRemoteAddr == 0 && !m_ngxScreenPercentageScanDone) {
      const uint64_t nowMs = ::GetTickCount64();
      if (nowMs >= m_ngxScreenPercentageNextScanMs) {
        m_ngxScreenPercentageNextScanMs = nowMs + kNgxSettingsScanRetryIntervalMs;
        locateNgxPassthroughGameSettings();
      }
    }

    if (m_ngxScreenPercentageRemoteAddr == 0 ||
        m_ngxGameSettingsShadowRemoteAddr == 0 ||
        m_ngxGameProcess == nullptr) {
      m_ngxGameScreenPercentage = 0.0f;
      return;
    }

    // A retained partial transaction is mirrored but never driven.
    if (!m_ngxGameSettingsRedirectsValid || !driving) {
      // A probe interrupted by driving being switched off would otherwise run out its frame
      // budget while nothing is being driven, and reject a candidate it never tested.
      if (m_ngxSettingsProbeState == NgxSettingsProbeState::Pending)
        m_ngxSettingsProbeState = NgxSettingsProbeState::Idle;

      mirrorNgxLiveScreenPercentage();
      return;
    }

    uint32_t displayHeight = 0;
    uint32_t displayWidth = 0;
    if (m_activePresentParams.has_value()) {
      displayHeight = m_activePresentParams->BackBufferHeight;
      displayWidth = m_activePresentParams->BackBufferWidth;
    }

    bootstrapNgxPassthroughUpscaler(displayWidth, displayHeight);

    float screenPercentage = m_parent->GetDXVKDevice()->getCommon()->metaNgxPassthrough()
      .screenPercentageForDisplay(displayWidth, displayHeight);

    if (!std::isfinite(screenPercentage) || screenPercentage <= 0.0f) {
      ONCE(Logger::warn("[RTX NGX Passthrough] Upscaler returned an invalid "
                        "ScreenPercentage; retaining the previous effective value."));
      return;
    }

    // Nothing is trusted until the game's own scene viewport has been seen responding to a value
    // we drove, so the probe has to drive something the game must visibly react to. There is
    // nothing to prove while the request matches what the game is already using - and probing
    // then would cost a visible resolution blip for no gain, so keep mirroring until the
    // selected preset actually asks for something different.
    if (m_ngxSettingsProbeState == NgxSettingsProbeState::Idle) {
      if (m_ngxGameScreenPercentage > 0.0f &&
          std::fabs(screenPercentage - m_ngxGameScreenPercentage) < 0.5f) {
        mirrorNgxLiveScreenPercentage();
        return;
      }

      m_ngxSettingsProbeState = NgxSettingsProbeState::Pending;
      m_ngxSettingsProbeStartFrame = m_ue3FrameCounter;
      // Probing at the target itself avoids a blip whenever the selected preset is reduced.
      m_ngxSettingsProbeValue = screenPercentage < 90.0f
        ? screenPercentage
        : kNgxScreenPercentageProbeValue;
      Logger::info(str::format(
        "[RTX NGX Passthrough] Verifying the game-settings redirects by driving ",
        m_ngxSettingsProbeValue, "%."));
    }

    if (m_ngxSettingsProbeState == NgxSettingsProbeState::Pending) {
      updateNgxSettingsProbe();
      if (m_ngxSettingsProbeState == NgxSettingsProbeState::Pending) {
        if (writeNgxScreenPercentageShadow(m_ngxSettingsProbeValue))
          m_ngxGameScreenPercentage = m_ngxSettingsProbeValue;
        return;
      }
    }

    if (m_ngxSettingsProbeState != NgxSettingsProbeState::Confirmed)
      return;

    if (!writeNgxScreenPercentageShadow(screenPercentage))
      return;

    m_ngxScreenPercentageDriven = true;
    m_ngxGameScreenPercentage = screenPercentage;

    if (screenPercentage != m_ngxScreenPercentageLastLogged) {
      m_ngxScreenPercentageLastLogged = screenPercentage;
      Logger::info(str::format("[RTX NGX Passthrough] ", RtxNgxPassthrough::upscalerModeLabel(),
                               " mode -> effective ScreenPercentage ", screenPercentage,
                               (screenPercentage >= 99.5f ? " (native)." : " (Super Resolution).")));
    }
  }

  void D3D9Rtx::locateNgxPassthroughGameSettings() {
    // The live resolution is the anchor the whole layout is discovered from, so there is
    // nothing to look for until the device's present parameters are known.
    if (!m_activePresentParams.has_value())
      return;

    NgxGameSettingsScanInputs scanInputs;
    scanInputs.backBufferWidth = m_activePresentParams->BackBufferWidth;
    scanInputs.backBufferHeight = m_activePresentParams->BackBufferHeight;
    scanInputs.windowed = m_activePresentParams->Windowed != FALSE;
    scanInputs.forcedScreenPercentageRva =
      uint32_t(std::max(0, RtxNgxPassthrough::systemSettingsScreenPercentageRva()));
    if (scanInputs.backBufferWidth == 0 || scanInputs.backBufferHeight == 0)
      return;

    // Current process + bridge parent; Incomplete results retry within the budget.
    struct Candidate { HANDLE handle; DWORD pid; bool ownsHandle; };
    std::vector<Candidate> candidates;
    candidates.push_back({ ::GetCurrentProcess(), ::GetCurrentProcessId(), false });

    bool anyIncomplete = false;

    const DWORD parentPid = ngxGetParentPid();
    if (parentPid != 0) {
      HANDLE parent = ::OpenProcess(PROCESS_VM_READ | PROCESS_VM_WRITE | PROCESS_VM_OPERATION | PROCESS_QUERY_INFORMATION,
                                    FALSE, parentPid);
      if (parent != nullptr) {
        candidates.push_back({ parent, parentPid, true });
      } else {
        anyIncomplete = true;
      }
    } else {
      anyIncomplete = true;
    }

    for (const Candidate& candidate : candidates) {
      uintptr_t moduleBase = 0;
      uint32_t moduleSize = 0;
      std::wstring modulePath;
      if (!ngxGetMainModule(candidate.pid, moduleBase, moduleSize, modulePath)) {
        anyIncomplete = true;
        continue;
      }

      scanInputs.ini = ngxCachedIniSystemSettings(modulePath);

      NgxLocatedGameSettings located;
      const NgxLocateResult result =
        ngxLocateGameSettings(candidate.handle, moduleBase, moduleSize, scanInputs,
                              m_ngxRejectedSettingsCandidates, located);
      if (result == NgxLocateResult::Incomplete)
        anyIncomplete = true;
      if (result != NgxLocateResult::Found) {
        if (result == NgxLocateResult::Miss && candidate.ownsHandle) {
          ONCE(Logger::info(str::format(
            "[RTX NGX Passthrough] Game-settings scan of the parent game process found nothing: ",
            located.missReason, ". Retrying while the engine finishes starting up.")));
        }
        continue;
      }

      void* shadowMemory =
        ::VirtualAllocEx(candidate.handle, nullptr, sizeof(NgxGameSettingsShadow),
                         MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
      const uintptr_t shadowAddress = reinterpret_cast<uintptr_t>(shadowMemory);
      if (shadowMemory == nullptr ||
          shadowAddress > uintptr_t(UINT32_MAX) - sizeof(NgxGameSettingsShadow)) {
        if (shadowMemory != nullptr)
          ::VirtualFreeEx(candidate.handle, shadowMemory, 0, MEM_RELEASE);
        anyIncomplete = true;
        continue;
      }

      NgxGameSettingsShadow shadow = {
        located.currentScreenPercentage,
        located.currentUpscaleScreenPercentage,
      };

      if (!ngxWriteProcessExact(candidate.handle, shadowAddress, shadow)) {
        ::VirtualFreeEx(candidate.handle, shadowMemory, 0, MEM_RELEASE);
        anyIncomplete = true;
        continue;
      }

      const uint32_t shadowScreenPercentage =
        uint32_t(shadowAddress + kNgxShadowScreenPercentageOffset);
      const uint32_t shadowUpscaleScreenPercentage =
        uint32_t(shadowAddress + kNgxShadowUpscaleScreenPercentageOffset);
      // Relative sites keep their [reg+disp32] form. Only the displacement moves.
      // They still depend on the base register holding GSystemSettings.
      const uint32_t relativeShadowScreenPercentage =
        uint32_t((shadowAddress + kNgxShadowScreenPercentageOffset) -
                 located.systemSettingsAddress);
      const uint32_t relativeShadowUpscaleScreenPercentage =
        uint32_t((shadowAddress + kNgxShadowUpscaleScreenPercentageOffset) -
                 located.systemSettingsAddress);

      std::vector<NgxOperandRedirect> redirects;
      redirects.reserve(
        located.directScreenPercentageSites.size() +
        located.directUpscaleScreenPercentageSites.size() +
        located.relativeScreenPercentageSites.size() +
        located.relativeUpscaleScreenPercentageSites.size());

      for (const NgxOperandSite& site : located.directScreenPercentageSites)
        redirects.push_back({ site.address, site.original, shadowScreenPercentage,
                              0, false, false,
                              NgxOperandCategory::AbsoluteScreenPercentage });
      for (const NgxOperandSite& site : located.directUpscaleScreenPercentageSites)
        redirects.push_back({ site.address, site.original, shadowUpscaleScreenPercentage,
                              0, false, false,
                              NgxOperandCategory::AbsoluteUpscale });
      // NeedsUpscale() and UnScaleScreenCoords read these fields through GSystemSettings.
      // Without them the reduced view stays in its rect and the frame is left with a black border.
      if (located.systemSettingsAddress != 0) {
        for (const NgxOperandSite& site : located.relativeScreenPercentageSites)
          redirects.push_back({ site.address, site.original, relativeShadowScreenPercentage,
                                0, false, false,
                                NgxOperandCategory::RelativeScreenPercentage });
        for (const NgxOperandSite& site : located.relativeUpscaleScreenPercentageSites)
          redirects.push_back({ site.address, site.original, relativeShadowUpscaleScreenPercentage,
                                0, false, false,
                                NgxOperandCategory::RelativeUpscale });
      }

      std::vector<NgxOperandRedirect> activeRedirects;
      activeRedirects.reserve(redirects.size());
      NgxOperandApplyResult applyResult;
      {
        NgxScopedThreadSuspension suspended(candidate.pid);
        if (suspended.complete())
          applyResult = ngxApplyOperandRedirects(candidate.handle, redirects, activeRedirects);
        else
          applyResult.skipped = redirects.size();
      }

      // Without ScaleScreenCoords' reader nothing can shrink the view, so a set that lost it is
      // worthless; roll the rest back rather than let the probe spend frames disproving it.
      const bool redirectsApplied =
        applyResult.appliedByCategory[size_t(NgxOperandCategory::AbsoluteScreenPercentage)] != 0;
      if (!redirectsApplied) {
        for (NgxOperandRedirect& redirect : activeRedirects) {
          if (!redirect.forceRestore && !ngxRestoreCodeOperand(candidate.handle, redirect))
            redirect.forceRestore = true;
        }
        activeRedirects.erase(
          std::remove_if(activeRedirects.begin(), activeRedirects.end(),
                         [](const NgxOperandRedirect& redirect) {
                           return !redirect.forceRestore;
                         }),
          activeRedirects.end());
      }

      if (!redirectsApplied) {
        if (activeRedirects.empty()) {
          ::VirtualFreeEx(candidate.handle, shadowMemory, 0, MEM_RELEASE);
          anyIncomplete = true;
          continue;
        }

        // Keep memory referenced by an operand that rollback could not restore.
        m_ngxGameProcess = candidate.handle;
        m_ngxGameProcessOwned = candidate.ownsHandle;
        m_ngxGameProcessId = candidate.pid;
        m_ngxScreenPercentageRemoteAddr =
          located.screenPercentageAddress;
        m_ngxGameSettingsShadowRemoteAddr = shadowAddress;
        for (const NgxOperandRedirect& redirect : activeRedirects) {
          m_ngxGameSettingsCodePatches.push_back({
            redirect.address, redirect.original, redirect.redirected,
            redirect.originalProtection,
            redirect.originalProtectionKnown, true
          });
          uint32_t currentOperand = 0;
          const bool currentReadable =
            ngxReadProcessExact(candidate.handle, redirect.address,
                                currentOperand);
          MEMORY_BASIC_INFORMATION memoryInfo = {};
          const bool protectionReadable =
            ::VirtualQueryEx(
              candidate.handle,
              reinterpret_cast<LPCVOID>(redirect.address),
              &memoryInfo, sizeof(memoryInfo)) == sizeof(memoryInfo);
          Logger::err(str::format(
            "[RTX NGX Passthrough] Outstanding operand at 0x", std::hex,
            redirect.address, ": original=0x", redirect.original,
            " redirected=0x", redirect.redirected,
            " current=",
            (currentReadable
              ? str::format("0x", std::hex, currentOperand)
              : std::string("<unreadable>")),
            " originalProtect=0x", redirect.originalProtection,
            " currentProtect=",
            (protectionReadable
              ? str::format("0x", std::hex, memoryInfo.Protect)
              : std::string("<unreadable>")),
            std::dec, "."));
        }
        m_ngxScreenPercentageScanDone = true;
        Logger::err("[RTX NGX Passthrough] Game-settings operand redirection failed and "
                    "could not be fully rolled back. Effective values remain mirrored from "
                    "the game, but automatic ScreenPercentage driving is disabled.");
        break;
      }

      m_ngxGameProcess = candidate.handle;
      m_ngxGameProcessOwned = candidate.ownsHandle;
      m_ngxGameProcessId = candidate.pid;
      m_ngxScreenPercentageRemoteAddr =
        located.screenPercentageAddress;
      m_ngxGameSettingsShadowRemoteAddr = shadowAddress;
      m_ngxGameSettingsRedirectsValid = true;
      m_ngxGameScreenPercentage = located.currentScreenPercentage;
      m_ngxGameSettingsCodePatches.reserve(activeRedirects.size());
      for (const NgxOperandRedirect& redirect : activeRedirects) {
        m_ngxGameSettingsCodePatches.push_back({
          redirect.address, redirect.original, redirect.redirected,
          redirect.originalProtection,
          redirect.originalProtectionKnown, redirect.forceRestore
        });
      }
      m_ngxScreenPercentageScanDone = true;

      // FSystemSettings::NeedsUpscale() reads both fields through `this`.
      // The engine stretch runs only when that pair is redirected.
      m_ngxEngineUpscaleAvailable =
        applyResult.appliedByCategory[size_t(NgxOperandCategory::RelativeScreenPercentage)] != 0 &&
        applyResult.appliedByCategory[size_t(NgxOperandCategory::RelativeUpscale)] != 0;

      const int upscaleOwner = RtxNgxPassthrough::screenPercentageUpscaleOwner();
      m_ngxRuntimeOwnedUpscale = upscaleOwner == 2 || !m_ngxEngineUpscaleAvailable;
      if (upscaleOwner == 1 && !m_ngxEngineUpscaleAvailable) {
        Logger::warn(
          "[RTX NGX Passthrough] screenPercentageUpscaleOwner requests the engine, but its "
          "NeedsUpscale() readers could not be redirected; the runtime will upscale instead.");
      }

      m_ngxSettingsProbeState = NgxSettingsProbeState::Idle;

      Logger::info(str::format(
        "[RTX NGX Passthrough] Game settings redirects installed in ",
        (candidate.ownsHandle
          ? "the parent game process (RTX Remix bridge)"
          : "the current process"),
        ": ScreenPercentage=0x", std::hex, located.screenPercentageAddress,
        ", GSystemSettings=0x", located.systemSettingsAddress,
        ", shadow=0x", shadowAddress, std::dec,
        ", live=", located.currentScreenPercentage, "/",
        located.currentUpscaleScreenPercentage, "."));
      Logger::info(str::format(
        "[RTX NGX Passthrough] Layout: ",
        (located.strictTailLayout ? "stock tail" : "relaxed tail"),
        ", ScreenPercentage offset 0x", std::hex, located.screenPercentageStructOffset,
        std::dec, ", FSystemSettingsData stride ", located.settingsDataStride,
        " (", located.defaultsRepeats, " Defaults repeats), ini cross-check ",
        (located.iniConfirmed ? "agrees" : "unavailable"),
        ". Operands: ", applyResult.applied, " patched, ", applyResult.skipped,
        " left stock, ", applyResult.unrestorable, " unrestorable. Upscale owner: ",
        (m_ngxRuntimeOwnedUpscale ? "runtime." : "engine.")));
      break;
    }

    for (const Candidate& candidate : candidates) {
      if (candidate.ownsHandle && candidate.handle != m_ngxGameProcess)
        ::CloseHandle(candidate.handle);
    }

    // Unlike a code-signature scan, this one reads engine state: GSystemSettings is only
    // populated once FSystemSettings::Initialize has run and the resolution has been applied,
    // so an early miss says nothing about whether the record exists. Keep retrying on the
    // shared budget instead of giving up on the first look.
    if (m_ngxScreenPercentageRemoteAddr == 0 &&
        m_ngxGameSettingsShadowRemoteAddr == 0 &&
        ++m_ngxScreenPercentageScanAttempts >= kNgxSettingsScanMaxAttempts) {
      m_ngxScreenPercentageScanDone = true;
      Logger::warn(str::format(
        "[RTX NGX Passthrough] No usable FSystemSettings record was found in the game's main "
        "module after ", kNgxSettingsScanMaxAttempts,
        (anyIncomplete ? " attempts (the process or module was not fully readable); "
                       : " attempts; "),
        "the DLSS mode selector will not drive render resolution. Set "
        "rtx.ngxPassthrough.systemSettingsScreenPercentageRva to pin the address, or use the "
        "in-game 'scale set ScreenPercentage <value>' console command."));
    }
  }

  void D3D9Rtx::restoreNgxGameSettingsRedirects() {
    if (m_ngxGameProcess == nullptr ||
        m_ngxGameSettingsShadowRemoteAddr == 0)
      return;

    // Mirror live values first so any unrestorable redirect stays coherent.
    bool screenShadowMirrored = false;
    if (m_ngxScreenPercentageRemoteAddr != 0) {
      float liveScreenPercentage = 0.0f;
      int32_t liveUpscale = 0;
      if (ngxReadProcessExact(m_ngxGameProcess,
                              m_ngxScreenPercentageRemoteAddr,
                              liveScreenPercentage) &&
          ngxReadProcessExact(m_ngxGameProcess,
                              m_ngxScreenPercentageRemoteAddr + sizeof(float),
                              liveUpscale)) {
        bool rollbackVerified = true;
        screenShadowMirrored =
          ngxUpdateScreenPercentageShadow(
            m_ngxGameProcess, m_ngxGameSettingsShadowRemoteAddr,
            liveScreenPercentage, liveUpscale,
            &rollbackVerified) &&
          rollbackVerified;
      }
    }

    bool allRedirectsReleased = true;
    bool suspensionComplete = false;
    bool shadowReleased = false;
    std::vector<uintptr_t> externallyChangedOperands;
    externallyChangedOperands.reserve(m_ngxGameSettingsCodePatches.size());
    {
      NgxScopedThreadSuspension suspended(m_ngxGameProcessId);
      suspensionComplete = suspended.complete();
      if (suspensionComplete) {
        for (auto it = m_ngxGameSettingsCodePatches.rbegin();
             it != m_ngxGameSettingsCodePatches.rend(); ++it) {
          NgxOperandRedirect redirect = {
            it->operandAddress,
            it->originalOperand,
            it->redirectedOperand,
            it->originalProtection,
            it->originalProtectionKnown,
            it->forceRestore,
          };
          uint32_t current = 0;
          if (!ngxReadProcessExact(m_ngxGameProcess, it->operandAddress,
                                   current)) {
            allRedirectsReleased = false;
            continue;
          }

          if (it->forceRestore ||
              current == it->redirectedOperand) {
            if (!ngxRestoreCodeOperand(m_ngxGameProcess, redirect))
              allRedirectsReleased = false;
          } else if (current != it->originalOperand) {
            // External owner; do not clobber. Safe to free our shadow for this site.
            externallyChangedOperands.push_back(it->operandAddress);
          }
        }

        // Confirm no live operand still references the shadow before free.
        for (const NgxGameSettingsCodePatch& patch :
             m_ngxGameSettingsCodePatches) {
          uint32_t current = 0;
          if (!ngxReadProcessExact(m_ngxGameProcess, patch.operandAddress,
                                   current) ||
              current == patch.redirectedOperand ||
              (patch.forceRestore && current != patch.originalOperand))
            allRedirectsReleased = false;
        }

        // Free while still suspended so resumed threads cannot race the release.
        if (allRedirectsReleased) {
          shadowReleased =
            ::VirtualFreeEx(
              m_ngxGameProcess,
              reinterpret_cast<LPVOID>(
                m_ngxGameSettingsShadowRemoteAddr),
              0, MEM_RELEASE) != FALSE;
        }
      }
    }

    if (!suspensionComplete) {
      Logger::warn(
        "[RTX NGX Passthrough] Could not quiesce the game threads to restore settings "
        "reader operands; retaining the renderer shadow allocation.");
      if (!screenShadowMirrored) {
        Logger::err(
          "[RTX NGX Passthrough] The retained renderer shadow could not be fully returned "
          "to the game's live values before teardown.");
      }
      return;
    }

    for (const uintptr_t address : externallyChangedOperands) {
      Logger::warn(str::format(
        "[RTX NGX Passthrough] Settings reader operand at 0x", std::hex,
        address, std::dec,
        " changed by another component; leaving its value intact."));
    }

    if (allRedirectsReleased) {
      if (!shadowReleased) {
        Logger::warn(
          "[RTX NGX Passthrough] Settings reader operands were restored, but the renderer "
          "shadow allocation could not be released.");
      }
      m_ngxGameSettingsCodePatches.clear();
      m_ngxGameSettingsShadowRemoteAddr = 0;
    } else {
      Logger::warn(
        "[RTX NGX Passthrough] One or more settings reader operands could not be restored; "
        "retaining the renderer shadow allocation so no game code points to freed memory.");
      if (!screenShadowMirrored) {
        Logger::err(
          "[RTX NGX Passthrough] The retained renderer shadow could not be fully returned "
          "to the game's live values.");
      }
    }

    m_ngxScreenPercentageDriven = false;
    m_ngxGameSettingsRedirectsValid = false;
    m_ngxEngineUpscaleAvailable = false;
    m_ngxRuntimeOwnedUpscale = false;
    m_ngxMissingEngineUpscaleFrames = 0;
  }

  // Reads the constants of the draw about to be suppressed. UE3's GammaCorrectionPixelShader is
  // `pow(saturate(lerp(scene * ColorScale, OverlayColor.rgb, OverlayColor.a)), InverseGamma)`;
  // when the post chain already gamma corrected into LDR scene colour the engine sets
  // InverseGamma to 1 and this reduces to a copy, which is why Mirror's Edge never needed it.
  void D3D9Rtx::captureNgxOutputTransform() {
    m_ngxOutputTransform = NgxOutputTransform();

    if (!m_parent->UseProgrammablePS() || d3d9State().pixelShader.ptr() == nullptr)
      return;

    const D3D9CommonShader* pixelShaderCommon = d3d9State().pixelShader->GetCommonShader();
    const Ue3ShaderFeatureInfo psInfo = getUe3ShaderFeatureInfo(pixelShaderCommon);

    if (psInfo.gammaInverseReg == Ue3ShaderFeatureInfo::kNgxNoRegister) {
      // Not UE3's GammaCorrectionPixelShader. Report what the suppressed draw actually is:
      // if it carries the Uber post-process constants then the engine composited its whole
      // post chain here, and replacing that draw costs far more than a gamma curve.
      ONCE(Logger::warn(str::format(
        "[RTX NGX Passthrough] The suppressed composite is not UE3's GammaCorrectionPixelShader "
        "(ps=0x", std::hex, pixelShaderCommon->GetBytecodeHash(), std::dec,
        ", gammaConstants=", psInfo.hasGammaConstants ? 1 : 0,
        ", toneMapConstants=", psInfo.hasToneMapConstants ? 1 : 0,
        ", exposureOrToneSampler=", psInfo.hasExposureOrToneSampler ? 1 : 0,
        ", sceneColorSampler=", psInfo.hasSceneColorSampler ? 1 : 0,
        ", colorScaleReg=", psInfo.gammaColorScaleReg != Ue3ShaderFeatureInfo::kNgxNoRegister ? 1 : 0,
        ", overlayColorReg=", psInfo.gammaOverlayColorReg != Ue3ShaderFeatureInfo::kNgxNoRegister ? 1 : 0,
        "); its colour transform cannot be carried over to the upscaled result.")));
      return;
    }

    const auto& constants = d3d9State().psConsts.fConsts;
    const auto readRegister = [&](uint32_t reg) -> const Vector4* {
      return reg < std::size(constants) ? &constants[reg] : nullptr;
    };

    const Vector4* inverseGamma = readRegister(psInfo.gammaInverseReg);
    if (inverseGamma == nullptr || !std::isfinite(inverseGamma->x) ||
        inverseGamma->x <= 0.0f || inverseGamma->x > 4.0f)
      return;

    m_ngxOutputTransform.inverseGamma = inverseGamma->x;

    if (const Vector4* colorScale = readRegister(psInfo.gammaColorScaleReg)) {
      m_ngxOutputTransform.colorScale[0] = colorScale->x;
      m_ngxOutputTransform.colorScale[1] = colorScale->y;
      m_ngxOutputTransform.colorScale[2] = colorScale->z;
    }

    if (const Vector4* overlayColor = readRegister(psInfo.gammaOverlayColorReg)) {
      m_ngxOutputTransform.overlayColor[0] = overlayColor->x;
      m_ngxOutputTransform.overlayColor[1] = overlayColor->y;
      m_ngxOutputTransform.overlayColor[2] = overlayColor->z;
      m_ngxOutputTransform.overlayColor[3] = overlayColor->w;
    }

    const bool isIdentity =
      m_ngxOutputTransform.inverseGamma == 1.0f &&
      m_ngxOutputTransform.colorScale[0] == 1.0f &&
      m_ngxOutputTransform.colorScale[1] == 1.0f &&
      m_ngxOutputTransform.colorScale[2] == 1.0f &&
      m_ngxOutputTransform.overlayColor[3] == 0.0f;
    m_ngxOutputTransform.enabled = !isIdentity;

    ONCE(Logger::info(str::format(
      "[RTX NGX Passthrough] Suppressed composite transform: InverseGamma ",
      m_ngxOutputTransform.inverseGamma, ", ColorScale (",
      m_ngxOutputTransform.colorScale[0], ", ", m_ngxOutputTransform.colorScale[1], ", ",
      m_ngxOutputTransform.colorScale[2], "), OverlayColor alpha ",
      m_ngxOutputTransform.overlayColor[3], " -> ",
      (m_ngxOutputTransform.enabled ? "reapplied to the upscaled result."
                                    : "identity, nothing to reapply."))));
  }

  void D3D9Rtx::emitNgxPassthroughFrameData() {
    if (m_ngxFrameDataEmitted)
      return;
    m_ngxFrameDataEmitted = true;

    Rc<DxvkImage> sceneDepth = m_ngxSceneDepthImage;

    // Runtime-owned upscale: the engine composited its reduced view rect into the backbuffer and
    // performed no upscale of its own, so DLSS reads that subrect and writes the full frame. The
    // depth subrect stays where the scene rasterized (scene color space), which is why the two
    // offsets are tracked separately.
    if (m_ngxUpscaleSourceImage == nullptr && m_ngxRuntimeUpscaleRectValid &&
        m_ngxColorTargetImage == nullptr && m_ngxFrameBackbufferImage != nullptr &&
        m_ngxSceneViewportValid) {
      m_ngxUpscaleSourceImage = m_ngxFrameBackbufferImage;
      m_ngxSubrect.offset = { int32_t(m_ngxSceneViewport.X), int32_t(m_ngxSceneViewport.Y) };
      m_ngxSubrect.extent = { m_ngxSceneViewport.Width, m_ngxSceneViewport.Height };
      m_ngxColorSubrectOffset = { int32_t(m_ngxRuntimeUpscaleRect.X),
                                  int32_t(m_ngxRuntimeUpscaleRect.Y) };
    }

    // A reduced subrect with no Super Resolution interception means the depth rect
    // does not match the full-resolution colour this injection sees.
    // The DLSS inputs are skipped in that case.
    if (m_ngxUpscaleSourceImage == nullptr && m_ngxColorTargetImage == nullptr &&
        m_ngxSceneViewportValid && m_activePresentParams.has_value()) {
      const uint32_t backBufferWidth = m_activePresentParams->BackBufferWidth;
      const uint32_t backBufferHeight = m_activePresentParams->BackBufferHeight;

      if (ngxSceneViewportIsSubrect(m_ngxSceneViewport, backBufferWidth, backBufferHeight)) {
        ONCE(Logger::warn("[RTX NGX Passthrough] Scene rendered into a reduced subrect without Super Resolution interception; DLSS is skipped."));
        sceneDepth = nullptr;

        // Neither the engine's upscale stretch nor its subrect composite was found. If the
        // engine was supposed to be doing the upscale, hand ownership to the runtime rather
        // than keep presenting a reduced image; the next frame renders through that path.
        if (!m_ngxRuntimeOwnedUpscale && m_ngxGameSettingsRedirectsValid &&
            RtxNgxPassthrough::screenPercentageUpscaleOwner() != 1 &&
            ++m_ngxMissingEngineUpscaleFrames >= kNgxMissingEngineUpscaleFrameLimit) {
          m_ngxRuntimeOwnedUpscale = true;
          m_ngxMissingEngineUpscaleFrames = 0;
          Logger::warn("[RTX NGX Passthrough] The engine's own upscale never appeared; taking "
                       "ownership of the Super Resolution upscale in the runtime instead.");
        }
      }
    } else {
      m_ngxMissingEngineUpscaleFrames = 0;
    }

    m_ngxVelocityStats.frameCameraValid = m_ngxFrameCameraValid;
    m_ngxVelocityStats.depthClears = m_ngxDepthClearsThisFrame;
    m_ngxVelocityStats.cameraTransposeFlips = m_ngxCameraTransposeFlips;

    // Accumulated across the window and logged as a rate. The per-sighting pairing-miss lines are
    // rate limited to a burst every couple of seconds, which shows what a miss looks like but not
    // how often one happens - and whether the capture is healthy is entirely a question of the
    // ratio between sightings that paired and sightings that had to register anew.
    m_ngxVelocityWindow.captured += m_ngxVelocityStats.captured;
    m_ngxVelocityWindow.capturedSkinned += m_ngxVelocityStats.capturedSkinned;
    m_ngxVelocityWindow.capturedDynamic += m_ngxVelocityStats.capturedDynamic;
    m_ngxVelocityWindow.capturedForeground += m_ngxVelocityStats.capturedForeground;
    m_ngxVelocityWindow.exactMatches += m_ngxVelocityStats.exactMatches;
    m_ngxVelocityWindow.newRegistrations += m_ngxVelocityStats.newRegistrations;
    m_ngxVelocityWindow.newRegistrationsSkinned += m_ngxVelocityStats.newRegistrationsSkinned;
    m_ngxVelocityWindow.missNoLastFrameSighting += m_ngxVelocityStats.missNoLastFrameSighting;
    m_ngxVelocityWindow.missBeyondTranslation += m_ngxVelocityStats.missBeyondTranslation;
    m_ngxVelocityWindow.missBeyondRotation += m_ngxVelocityStats.missBeyondRotation;
    m_ngxVelocityWindow.claimedWithoutVelocity += m_ngxVelocityStats.claimedWithoutVelocity;
    m_ngxVelocityWindow.pairedBeyondBounds += m_ngxVelocityStats.pairedBeyondBounds;
    m_ngxVelocityWindow.skippedNoCamera += m_ngxVelocityStats.skippedNoCamera;
    m_ngxVelocityWindow.skippedBudget += m_ngxVelocityStats.skippedBudget;
    m_ngxVelocityWindow.skippedZDisabled += m_ngxVelocityStats.skippedZDisabled;
    m_ngxVelocityWindow.skippedInstanceCap += m_ngxVelocityStats.skippedInstanceCap;
    m_ngxVelocityWindow.skippedDynamicBuffer += m_ngxVelocityStats.skippedDynamicBuffer;
    m_ngxVelocityWindow.skippedBonePalette += m_ngxVelocityStats.skippedBonePalette;
    m_ngxVelocityWindow.depthClears += m_ngxVelocityStats.depthClears;
    if (m_ngxVelocityStats.depthClears > 1) {
      m_ngxVelocityWindow.framesWithOrphanedDepthPhase++;
    }

    if (++m_ngxVelocityWindow.frames >= kNgxVelocityWindowFrames) {
      Logger::info(str::format(
        "[RTX NGX Passthrough][velocity] over ", m_ngxVelocityWindow.frames, " frames: ",
        m_ngxVelocityWindow.captured, " captured (", m_ngxVelocityWindow.capturedSkinned, " skinned, ",
        m_ngxVelocityWindow.capturedDynamic, " CPU-modified, ",
        m_ngxVelocityWindow.capturedForeground, " foreground-phase), ",
        m_ngxVelocityWindow.exactMatches, " paired to their own history, ",
        m_ngxVelocityWindow.newRegistrations, " registered anew (",
        m_ngxVelocityWindow.newRegistrationsSkinned, " of them skinned; ",
        m_ngxVelocityWindow.missNoLastFrameSighting, " unseen last frame / ",
        m_ngxVelocityWindow.missBeyondTranslation, " beyond translation / ",
        m_ngxVelocityWindow.missBeyondRotation, " beyond rotation), ",
        m_ngxVelocityWindow.claimedWithoutVelocity, " claimed without velocity, ",
        m_ngxVelocityWindow.pairedBeyondBounds, " paired beyond the motion bounds, skipped: ",
        m_ngxVelocityWindow.skippedNoCamera, " no camera / ",
        m_ngxVelocityWindow.skippedBudget, " over budget / ",
        m_ngxVelocityWindow.skippedZDisabled, " depth test off / ",
        m_ngxVelocityWindow.skippedInstanceCap, " over the instance cap / ",
        m_ngxVelocityWindow.skippedDynamicBuffer, " on unrecognised dynamic buffers / ",
        m_ngxVelocityWindow.skippedBonePalette, " over the bone palette cap; tracking ",
        m_ngxVelocityObjectCache.size(), " identities; ", m_ngxVelocityWindow.depthClears,
        " mid-scene depth clears over ", m_ngxVelocityWindow.framesWithOrphanedDepthPhase,
        " frames cleared more than once; scene-wide transform offset (",
        m_ngxGlobalTransformOffset.x, ",", m_ngxGlobalTransformOffset.y, ",",
        m_ngxGlobalTransformOffset.z, ")."));

      m_ngxVelocityWindow = NgxVelocityWindowTotals();
    }

    m_parent->EmitCs([cSceneDepth = sceneDepth,
                      cColorTarget = m_ngxColorTargetImage,
                      cColorMirror = m_ngxColorMirrorImage,
                      cUpscaleSource = m_ngxUpscaleSourceImage,
                      cSubrect = m_ngxSubrect,
                      cColorSubrectOffset = m_ngxColorSubrectOffset,
                      cOutputTransform = m_ngxOutputTransform,
                      cVelocityDraws = std::move(m_ngxVelocityDraws),
                      cVelocityStats = m_ngxVelocityStats,
                      cSceneTransformOffset = m_ngxGlobalTransformOffset,
                      cJitterX = m_ngxFrameJitter[0],
                      cJitterY = m_ngxFrameJitter[1],
                      cCameraMatricesValid = m_ngxFrameCameraMatricesValid,
                      cWorldToView = m_ngxFrameWorldToView,
                      cViewToProjection = m_ngxFrameViewToProjection](DxvkContext* ctx) mutable {
      static_cast<RtxContext*>(ctx)->setNgxPassthroughFrameData(cSceneDepth, cColorTarget, cColorMirror, cUpscaleSource, cSubrect,
                                                                cColorSubrectOffset, cOutputTransform,
                                                                std::move(cVelocityDraws), cVelocityStats, cSceneTransformOffset,
                                                                cJitterX, cJitterY,
                                                                cCameraMatricesValid, cWorldToView, cViewToProjection);
    });

    m_ngxVelocityDraws.clear();
  }

  const char* D3D9Rtx::classifyNgxImageForDump(const DxvkImage* image) const {
    if (image == nullptr)
      return "null";
    if (m_ngxSceneColorImage != nullptr && image == m_ngxSceneColorImage.ptr())
      return "sceneColor";
    if (m_ngxSceneDepthImage != nullptr && image == m_ngxSceneDepthImage.ptr())
      return "sceneDepth";
    if (m_ngxFrameBackbufferImage != nullptr && image == m_ngxFrameBackbufferImage.ptr())
      return "backbuffer";
    for (uint32_t i = 0; i < m_ngxSceneColorResolveCount; i++) {
      if (m_ngxSceneColorResolves[i].ptr() == image)
        return "sceneColorResolve";
    }
    return "other";
  }

  // Line cap shared by the per-draw and StretchRect dump paths (heavy frames would
  // otherwise flood the log)
  static constexpr uint32_t kNgxPostChainDumpMaxLinesPerFrame = 160;

  // Frames a donated scene view rect stays usable for the injection decision
  static constexpr uint32_t kNgxSceneViewportStaleFrameLimit = 8;

  // Frames between the bounded frame-shape log lines
  static constexpr uint32_t kNgxFrameShapeLogInterval = 900;

  // A frame reaching the late injection point while the pre-post one is enabled has relocated
  // the injection, which costs the upscaler its history. Attribute it to the requirement that
  // failed; rate limited because what matters is which reason dominates, not each instance.
  void D3D9Rtx::reportNgxPrePostMiss() {
    if (!ngxPrePostInjectionAllowed())
      return;

    // A frame that never identified a scene color has no scene to inject into - a menu, a loading
    // screen, the frames either side of a level transition. Nothing relocated there and no history
    // is lost, so counting it here would misattribute a normal transition as a fault.
    if (m_ngxSceneColorImage == nullptr) {
      return;
    }

    m_ngxPrePostMissCount++;

    if (m_ngxPrePostMissCount != 1 &&
        m_ngxPrePostMissCount - m_ngxPrePostMissLastReportedCount < 60)
      return;

    m_ngxPrePostMissLastReportedCount = m_ngxPrePostMissCount;

    const char* reason =
      !m_ngxDiagSceneColorReady ? "no scene color/view rect was established" :
      !m_ngxDiagSawPostQuad     ? "no pass matched the post-process quad shape" :
      m_ngxPrePostCandidatesThisFrame == 0
                                ? "a qualifying pass ran but sampled neither the scene color nor a resolve of it" :
                                  "the frame's scene color reads ran out before the one the injection was aimed at";

    Logger::info(str::format("[RTX NGX Passthrough] Injection relocated to the late point on ",
                             m_ngxPrePostMissCount, " frames so far; most recently because ", reason,
                             ". DLSS keeps a separate history per injection point, so relocating "
                             "costs the moving content its accumulation."));
  }

  const char* D3D9Rtx::classifyNgxPassForDump(uint32_t rtSamplerMask, const DxvkImage* renderTargetImage) {
    Ue3ShaderFeatureInfo info;
    if (!tryGetBoundPixelFeatures(info))
      return "other";

    if (info.hasHazeConstants)
      return "haze";
    if (ngxBoundPixelIsDofGather())
      return "gather";
    if (info.hasFilterTextureSampler && info.hasFilterSampleWeights)
      return "blur";
    if (info.hasBlurredImageSampler)
      return info.hasToneMapConstants ? "uber" : "DOF blend";
    if (info.hasToneMapConstants || info.hasExposureOrToneSampler)
      return "tonemap";
    if (info.gammaInverseReg != Ue3ShaderFeatureInfo::kNgxNoRegister)
      return "gamma";
    if (info.hasVelocitySampler || info.hasMotionBlurConstants)
      return "motion blur";

    const bool targetIsBackbuffer =
      m_ngxFrameBackbufferImage != nullptr && renderTargetImage == m_ngxFrameBackbufferImage.ptr();
    if (!targetIsBackbuffer && !ngxShaderHasPostProcessNames(info) && m_ngxSceneColorImage != nullptr &&
        rtSamplerMask != 0 && (rtSamplerMask & (rtSamplerMask - 1)) == 0) {
      for (const uint32_t i : bit::BitMask(rtSamplerMask)) {
        D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[i]);
        if (texture != nullptr && texture->GetImage().ptr() == m_ngxSceneColorImage.ptr())
          return "resolve";
      }
    }

    return "other";
  }

  void D3D9Rtx::dumpNgxPostChainDraw(const DrawContext& drawContext) {
    if (m_ngxPostChainDumpLinesThisFrame >= kNgxPostChainDumpMaxLinesPerFrame)
      return;

    D3D9CommonTexture* renderTargetTexture = d3d9State().renderTargets[kRenderTargetIndex] != nullptr
      ? d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture() : nullptr;
    const DxvkImage* renderTargetImage = (renderTargetTexture != nullptr && renderTargetTexture->GetImage() != nullptr)
      ? renderTargetTexture->GetImage().ptr() : nullptr;

    const uint32_t rtSamplerMask = m_parent->GetActiveRTTextures() & m_parent->m_psShaderMasks.samplerMask;

    // Plain scene geometry (into the scene color, no render-target textures sampled) would
    // flood the log; the interesting flow is everything else: post passes, composites, UI
    const bool rtIsSceneColor = m_ngxSceneColorImage != nullptr && renderTargetImage == m_ngxSceneColorImage.ptr();
    if (rtIsSceneColor && rtSamplerMask == 0)
      return;

    m_ngxPostChainDumpLinesThisFrame++;

    std::string line = str::format(
      "[RTX NGX Passthrough][dump] draw=", m_drawCallID,
      (m_rtxInjectTriggered ? " (post-inject)" : ""),
      " rt=", classifyNgxImageForDump(renderTargetImage), "(0x", std::hex, uintptr_t(renderTargetImage), std::dec);

    if (renderTargetImage != nullptr) {
      line += str::format(",", renderTargetImage->info().extent.width, "x", renderTargetImage->info().extent.height,
                          ",fmt=", int(renderTargetImage->info().format));
    }

    const D3DVIEWPORT9& vp = d3d9State().viewport;
    line += str::format(") vp=", vp.Width, "x", vp.Height,
                        " prims=", drawContext.PrimitiveCount,
                        " z=", int(d3d9State().renderStates[D3DRS_ZENABLE]),
                        " zw=", int(d3d9State().renderStates[D3DRS_ZWRITEENABLE]),
                        " st=", int(d3d9State().renderStates[D3DRS_STENCILENABLE]),
                        " ab=", int(d3d9State().renderStates[D3DRS_ALPHABLENDENABLE]),
                        " kind=", classifyNgxPassForDump(rtSamplerMask, renderTargetImage));

    for (const uint32_t i : bit::BitMask(rtSamplerMask)) {
      D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[i]);
      const DxvkImage* sampledImage = (texture != nullptr && texture->GetImage() != nullptr) ? texture->GetImage().ptr() : nullptr;

      line += str::format(" s", i, "=", classifyNgxImageForDump(sampledImage), "(0x", std::hex, uintptr_t(sampledImage), std::dec);
      if (sampledImage != nullptr) {
        line += str::format(",", sampledImage->info().extent.width, "x", sampledImage->info().extent.height);
      }
      line += ")";
    }

    Logger::info(line);
  }

  void D3D9Rtx::NotifyStretchRect(const Rc<DxvkImage>& sourceImage, const Rc<DxvkImage>& destImage) {
    if (m_passthroughSuspended)
      return;

    if (m_ngxPostChainDumpFramesLeft > 0 && m_ngxPostChainDumpLinesThisFrame < kNgxPostChainDumpMaxLinesPerFrame) {
      m_ngxPostChainDumpLinesThisFrame++;
      Logger::info(str::format(
        "[RTX NGX Passthrough][dump] StretchRect",
        (m_rtxInjectTriggered ? " (post-inject)" : ""),
        " src=", classifyNgxImageForDump(sourceImage.ptr()), "(0x", std::hex, uintptr_t(sourceImage.ptr()), std::dec,
        (sourceImage != nullptr ? str::format(",", sourceImage->info().extent.width, "x", sourceImage->info().extent.height) : ""),
        ") dst=", classifyNgxImageForDump(destImage.ptr()), "(0x", std::hex, uintptr_t(destImage.ptr()), std::dec,
        (destImage != nullptr ? str::format(",", destImage->info().extent.width, "x", destImage->info().extent.height) : ""), ")"));
    }

    if (m_rtxInjectTriggered || sourceImage == nullptr || destImage == nullptr)
      return;

    // Only full-size copies are resolves.
    // Reduced-size copies are downsamples.
    if (destImage->info().extent.width != sourceImage->info().extent.width ||
        destImage->info().extent.height != sourceImage->info().extent.height)
      return;

    if (ngxImageIsUberEncoded(sourceImage.ptr()))
      rememberNgxUberEncodedImage(destImage);

    if (!m_frameOptions.ngxPrePostProcess || m_ngxSceneColorImage == nullptr ||
        sourceImage.ptr() != m_ngxSceneColorImage.ptr())
      return;

    m_ngxResolvesFromStretchRect = true;
    rememberNgxSceneColorResolve(destImage);
  }

  void D3D9Rtx::rememberNgxSceneColorResolve(const Rc<DxvkImage>& image) {
    for (uint32_t i = 0; i < m_ngxSceneColorResolveCount; i++) {
      if (m_ngxSceneColorResolves[i].ptr() == image.ptr())
        return;
    }

    if (m_ngxSceneColorResolveCount < m_ngxSceneColorResolves.size()) {
      m_ngxSceneColorResolves[m_ngxSceneColorResolveCount++] = image;
    }
  }

  // Stock UE3 D3D9 CopyToResolveTarget draws a surface into its resolve texture.
  // The draw is two opaque triangles with depth off, sampling only that surface.
  // Its pixel shader is a plain copy with none of the post-process names.
  // A copy of an image the uber pass encoded is marked as encoded too.
  void D3D9Rtx::learnNgxDrawResolve(const DrawContext& drawContext) {
    const bool learnSceneColor = m_frameOptions.ngxPrePostProcess && m_ngxSceneColorImage != nullptr;
    if ((!learnSceneColor && m_ngxUberEncodedImageCount == 0) ||
        drawContext.PrimitiveCount == 0 || drawContext.PrimitiveCount > 4)
      return;

    if (!ngxDepthTestDisabled() ||
        d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE ||
        d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] != FALSE ||
        d3d9State().renderTargets[kRenderTargetIndex] == nullptr)
      return;

    D3D9CommonTexture* target = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
    if (target == nullptr || target->GetImage() == nullptr)
      return;

    const Rc<DxvkImage>& targetImage = target->GetImage();
    if ((m_ngxSceneColorImage != nullptr && targetImage.ptr() == m_ngxSceneColorImage.ptr()) ||
        (m_ngxFrameBackbufferImage != nullptr && targetImage.ptr() == m_ngxFrameBackbufferImage.ptr()))
      return;

    uint32_t sampledTargets = 0;
    const DxvkImage* source = nullptr;
    const uint32_t rtSamplerMask = m_parent->GetActiveRTTextures() & m_parent->m_psShaderMasks.samplerMask;
    for (const uint32_t i : bit::BitMask(rtSamplerMask)) {
      D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[i]);
      if (texture == nullptr || texture->GetImage() == nullptr)
        continue;

      sampledTargets++;
      source = texture->GetImage().ptr();
    }

    if (sampledTargets != 1 || source == nullptr || source == targetImage.ptr())
      return;

    const DxvkImageCreateInfo& sourceInfo = source->info();
    const DxvkImageCreateInfo& targetInfo = targetImage->info();
    if (targetInfo.format != sourceInfo.format ||
        targetInfo.extent.width != sourceInfo.extent.width ||
        targetInfo.extent.height != sourceInfo.extent.height)
      return;

    Ue3ShaderFeatureInfo info;
    if (tryGetBoundPixelFeatures(info) && ngxShaderHasPostProcessNames(info))
      return;

    if (learnSceneColor && source == m_ngxSceneColorImage.ptr()) {
      m_ngxResolvesFromDraws = true;
      rememberNgxSceneColorResolve(targetImage);
      ONCE(Logger::info("[RTX NGX Passthrough] Scene colour resolve learned from a draw (stock UE3 D3D9 CopyToResolveTarget)."));
    } else if (ngxImageIsUberEncoded(source)) {
      rememberNgxUberEncodedImage(targetImage);
    }
  }

  bool D3D9Rtx::ngxImageIsUberEncoded(const DxvkImage* image) const {
    for (uint32_t i = 0; i < m_ngxUberEncodedImageCount; i++) {
      if (m_ngxUberEncodedImages[i].ptr() == image)
        return true;
    }
    return false;
  }

  void D3D9Rtx::rememberNgxUberEncodedImage(const Rc<DxvkImage>& image) {
    if (!ngxImageIsUberEncoded(image.ptr()) && m_ngxUberEncodedImageCount < m_ngxUberEncodedImages.size())
      m_ngxUberEncodedImages[m_ngxUberEncodedImageCount++] = image;
  }

  bool D3D9Rtx::readNgxBoundUberGamma(Vector4& gammaColorScaleAndInverse, bool requireBlurredImage) {
    Ue3ShaderFeatureInfo info;
    if (!tryGetBoundPixelFeatures(info) || info.toneMapGammaColorScaleReg < 0 ||
        (requireBlurredImage && !info.hasBlurredImageSampler))
      return false;

    const auto& constants = d3d9State().psConsts.fConsts;
    const uint32_t gammaReg = uint32_t(info.toneMapGammaColorScaleReg);
    if (gammaReg >= std::size(constants))
      return false;

    gammaColorScaleAndInverse = constants[gammaReg];
    return true;
  }

  NgxOutputTransform D3D9Rtx::ngxLinearizingTransform(const Vector4& gammaColorScaleAndInverse) {
    const Vector4& gamma = gammaColorScaleAndInverse;
    const bool finiteScale =
      std::isfinite(gamma.x) && std::isfinite(gamma.y) && std::isfinite(gamma.z) &&
      gamma.x > 0.05f && gamma.y > 0.05f && gamma.z > 0.05f;
    const bool encoded = std::isfinite(gamma.w) && gamma.w > 0.05f && gamma.w < 0.95f;

    NgxOutputTransform transform;
    if (finiteScale && encoded) {
      transform.colorScale[0] = gamma.x;
      transform.colorScale[1] = gamma.y;
      transform.colorScale[2] = gamma.z;
      transform.inverseGamma = gamma.w;
      transform.enabled = true;
      transform.linearizeInput = true;
    }
    return transform;
  }

  // An uber pass that writes an offscreen target leaves it display-encoded with GammaColorScaleAndInverse.
  void D3D9Rtx::noteNgxUberEncode(const DrawContext& drawContext) {
    if (drawContext.PrimitiveCount == 0 || drawContext.PrimitiveCount > 4 || !ngxDepthTestDisabled() ||
        d3d9State().renderTargets[kRenderTargetIndex] == nullptr)
      return;

    D3D9CommonTexture* target = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
    if (target == nullptr || target->GetImage() == nullptr ||
        (m_ngxFrameBackbufferImage != nullptr && target->GetImage().ptr() == m_ngxFrameBackbufferImage.ptr()))
      return;

    Vector4 gamma;
    if (!readNgxBoundUberGamma(gamma, true) || !ngxLinearizingTransform(gamma).enabled)
      return;

    for (uint32_t i = 0; i < m_ngxUberEncodedImageCount; i++)
      m_ngxUberEncodedImages[i] = nullptr;
    m_ngxUberEncodedImageCount = 0;
    m_ngxUberGammaColorScaleAndInverse = gamma;
    rememberNgxUberEncodedImage(target->GetImage());
  }

  void D3D9Rtx::NotifyClear(DWORD clearFlags) {
    if (m_passthroughSuspended || m_rtxInjectTriggered)
      return;

    if ((clearFlags & D3DCLEAR_ZBUFFER) == 0)
      return;

    // Only depth clears AFTER world geometry was drawn matter (frame-start clears and
    // SceneCapture probe clears precede any accepted scene camera draw)
    if (!m_ngxSceneViewportValid || m_ngxSceneDepthImage == nullptr)
      return;

    if (d3d9State().depthStencil == nullptr)
      return;

    D3D9CommonTexture* depthStencilTexture = d3d9State().depthStencil->GetCommonTexture();
    if (depthStencilTexture == nullptr || depthStencilTexture->GetImage().ptr() != m_ngxSceneDepthImage.ptr())
      return;

    // The world depth is about to be destroyed (UE3 clears depth ahead of its foreground
    // DPG; with occlusion culling enabled an extra clear precedes that one); snapshot it
    // at the FIRST clear for this frame's motion vector generation. Emitted before the
    // clear itself is recorded, so the copy is ordered ahead of it on the CS timeline.
    // Later clears only advance the counter (the world depth is already gone; the
    // foreground DPG following the last clear stays live in the depth buffer through the
    // injection point).
    m_ngxDepthClearsThisFrame++;
    m_ngxMidSceneClearSeen = true;
    m_ngxLastMidSceneClearFrame = m_ue3FrameCounter;

    if (m_ngxFrameClearCount < kNgxFrameShapeSlots) {
      m_ngxFrameClearDraws[m_ngxFrameClearCount++] = m_drawCallID;
    }

    // The injection counts reads from the clear, so a later clear restarts the count.
    m_ngxReadsSinceLastClear = 0;

    if (m_ngxDepthClearsThisFrame > 1)
      return;

    m_ngxDepthSnapshotTakenThisFrame = true;

    m_parent->EmitCs([cSceneDepth = m_ngxSceneDepthImage](DxvkContext* ctx) {
      static_cast<RtxContext*>(ctx)->snapshotNgxPassthroughDepth(cSceneDepth);
    });
  }

  bool D3D9Rtx::GetNgxPassthroughViewportJitter(float* pJitterX, float* pJitterY) const {
    if (m_passthroughSuspended || !m_ngxFrameJitterValid || m_rtxInjectTriggered)
      return false;

    if (m_ngxFrameJitter[0] == 0.0f && m_ngxFrameJitter[1] == 0.0f)
      return false;

    if (m_ngxSceneColorImage == nullptr || d3d9State().renderTargets[kRenderTargetIndex] == nullptr)
      return false;

    D3D9CommonTexture* renderTargetTexture = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();

    bool jitterThisDraw = renderTargetTexture != nullptr &&
                          renderTargetTexture->GetImage().ptr() == m_ngxSceneColorImage.ptr();

    // Screen space passes that render into other full-size targets while testing against
    // the scene depth-stencil (dynamic shadow projections and their stencil marking into
    // the light attenuation buffer, distortion accumulation) must shift with the scene:
    // their output is consumed at scene-aligned coordinates and their stencil/depth tests
    // run against the jittered scene depth. Reduced-size targets (bloom, AO) and the UI/
    // post chain (no scene depth bound) stay untouched.
    if (!jitterThisDraw &&
        m_ngxSceneDepthImage != nullptr &&
        renderTargetTexture != nullptr && renderTargetTexture->GetImage() != nullptr &&
        d3d9State().depthStencil != nullptr) {
      D3D9CommonTexture* depthStencilTexture = d3d9State().depthStencil->GetCommonTexture();

      if (depthStencilTexture != nullptr && depthStencilTexture->GetImage() != nullptr &&
          depthStencilTexture->GetImage().ptr() == m_ngxSceneDepthImage.ptr()) {
        const VkExtent3D& rtExtent = renderTargetTexture->GetImage()->info().extent;
        const VkExtent3D& sceneExtent = m_ngxSceneColorImage->info().extent;

        jitterThisDraw = rtExtent.width == sceneExtent.width && rtExtent.height == sceneExtent.height;
      }
    }

    if (!jitterThisDraw)
      return false;

    *pJitterX = m_ngxFrameJitter[0];
    *pJitterY = m_ngxFrameJitter[1];
    return true;
  }

  float D3D9Rtx::GetNgxPassthroughSamplerLodBias() const {
    if (m_passthroughSuspended || m_rtxInjectTriggered)
      return 0.0f;

    if (!m_ngxSceneViewportValid || m_ngxSceneColorImage == nullptr || !m_activePresentParams.has_value())
      return 0.0f;

    // Scene-color draws only: the same reduced-resolution content DLSS upscales. UI and
    // post passes (different targets, or post-injection) stay unbiased.
    if (d3d9State().renderTargets[kRenderTargetIndex] == nullptr)
      return 0.0f;

    D3D9CommonTexture* renderTargetTexture = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
    if (renderTargetTexture == nullptr || renderTargetTexture->GetImage().ptr() != m_ngxSceneColorImage.ptr())
      return 0.0f;

    // Bias only for a ScreenPercentage subrect.
    // One short axis is letterboxing, and full resolution needs no bias.
    const uint32_t backBufferWidth = m_activePresentParams->BackBufferWidth;
    const uint32_t backBufferHeight = m_activePresentParams->BackBufferHeight;

    if (!ngxSceneViewportIsSubrect(m_ngxSceneViewport, backBufferWidth, backBufferHeight))
      return 0.0f;

    // The standard DLSS integration bias: mip selection matches the upscaled output's
    // texel density instead of the reduced render resolution
    const float bias = std::log2(float(m_ngxSceneViewport.Width) / float(backBufferWidth));
    return std::clamp(bias, -4.0f, 0.0f);
  }

  D3D9Rtx::NgxSpsbCtabReg D3D9Rtx::scanNgxSpsbCtabReg(const std::vector<uint8_t>& bytecode) const {
    NgxSpsbCtabReg result;

    try {
      if (bytecode.size() < sizeof(uint32_t) || (bytecode.size() % sizeof(uint32_t)) != 0)
        return result;

      const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
      const uint32_t headerToken = tokens[0];
      const uint32_t headerTypeMask = headerToken & 0xffff0000u;

      DxsoProgramType programType;
      if (headerTypeMask == 0xffff0000u)
        programType = DxsoProgramTypes::PixelShader;
      else if (headerTypeMask == 0xfffe0000u)
        programType = DxsoProgramTypes::VertexShader;
      else
        return result;

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
        return result;

      for (const DxsoCtab::Constant& c : ctab.m_constantData) {
        if (c.registerCount < 1)
          continue;

        std::string name;
        name.reserve(c.name.size());
        for (const char ch : c.name)
          name.push_back(char(std::tolower(static_cast<unsigned char>(ch))));

        if (!result.present && name.find("screenpositionscalebias") != std::string::npos) {
          result.present = true;
          result.reg = c.registerIndex;
        } else if (!result.clampPresent && name.find("rendertargetclampparameter") != std::string::npos) {
          result.clampPresent = true;
          result.clampReg = c.registerIndex;
        }
      }
    } catch (...) {
      return result;
    }

    return result;
  }

  D3D9Rtx::NgxSpsbCtabReg D3D9Rtx::lookupNgxSpsbCtabReg(const D3D9CommonShader* shader) {
    if (shader == nullptr)
      return NgxSpsbCtabReg();

    const XXH64_hash_t shaderHash = shader->GetBytecodeHash();
    if (shaderHash == 0)
      return NgxSpsbCtabReg();

    auto it = m_ngxSpsbCtabCache.find(shaderHash);
    if (it == m_ngxSpsbCtabCache.end()) {
      m_ngxSpsbCtabCache.emplace(shaderHash, scanNgxSpsbCtabReg(shader->GetBytecode()));
      it = m_ngxSpsbCtabCache.find(shaderHash);
    }
    return it->second;
  }

  void D3D9Rtx::PatchNgxScreenPositionScaleBias(DxsoProgramType stage, void* floatConstants, uint32_t floatConstantCount) const {
    if (m_passthroughSuspended || !m_ngxSpsbPatchActive)
      return;

    const NgxSpsbCtabReg& patch = stage == DxsoProgramTypes::VertexShader ? m_ngxSpsbPatchVs : m_ngxSpsbPatchPs;
    float* constants = reinterpret_cast<float*>(floatConstants);

    // UV.x = clip.x * SPSB.x + SPSB.w; UV.y = clip.y * SPSB.y + SPSB.z (UE3 samples with
    // ".xy * SPSB.xy + SPSB.wz").
    if (m_ngxSpsbReplaceFull) {
      // Clip ±1 maps across the whole target.
      // The scale the game uploaded still addresses the subrect the scene was rasterized into.
      if (patch.present && patch.reg < floatConstantCount) {
        const Vector4 full = ngxFullBufferScaleBias();
        float* constant = constants + size_t(patch.reg) * 4;
        constant[0] = full.x;
        constant[1] = full.y;
        constant[2] = full.z;
        constant[3] = full.w;
      }

      // UE3 sets the clamp to the view rect in buffer UVs, shifted back by half a pixel.
      // The view now spans the whole buffer.
      if (patch.clampPresent && patch.clampReg < floatConstantCount && m_ngxSceneColorImage != nullptr) {
        const VkExtent3D& extent = m_ngxSceneColorImage->info().extent;
        if (extent.width != 0 && extent.height != 0) {
          float* constant = constants + size_t(patch.clampReg) * 4;
          constant[0] = -0.5f / float(extent.width);
          constant[1] = -0.5f / float(extent.height);
          constant[2] = (float(extent.width) - 0.5f) / float(extent.width);
          constant[3] = (float(extent.height) - 0.5f) / float(extent.height);
        }
      }
      return;
    }

    if (!patch.present || patch.reg >= floatConstantCount)
      return;

    float* constant = constants + size_t(patch.reg) * 4;
    constant[3] += m_ngxSpsbPatchAdd[0];
    constant[2] += m_ngxSpsbPatchAdd[1];
  }

  int32_t D3D9Rtx::ngxPatchRegisterKey(const NgxSpsbCtabReg& patch, bool replaceFull) {
    const int32_t scaleBias = patch.present ? int32_t(patch.reg) + 1 : 0;
    const int32_t clamp = (replaceFull && patch.clampPresent) ? int32_t(patch.clampReg) + 1 : 0;
    return (scaleBias | (clamp << 16)) - 1;
  }

  Vector4 D3D9Rtx::ngxFullBufferScaleBias() const {
    // GPixelCenterOffset is 0.5 in every UE3 D3D9 renderer.
    constexpr float kPixelCenterOffset = 0.5f;

    Vector4 full(0.5f, -0.5f, 0.5f, 0.5f);
    if (m_ngxSceneColorImage != nullptr) {
      const VkExtent3D& extent = m_ngxSceneColorImage->info().extent;
      if (extent.width != 0 && extent.height != 0) {
        full.z += kPixelCenterOffset / float(extent.height);
        full.w += kPixelCenterOffset / float(extent.width);
      }
    }
    return full;
  }

  void D3D9Rtx::armNgxFullBufferScreenPositionScaleBias() {
    m_ngxSpsbPatchVs = lookupNgxSpsbCtabReg(m_parent->UseProgrammableVS() && d3d9State().vertexShader.ptr() != nullptr
                                              ? d3d9State().vertexShader->GetCommonShader() : nullptr);
    m_ngxSpsbPatchPs = lookupNgxSpsbCtabReg(d3d9State().pixelShader.ptr() != nullptr
                                              ? d3d9State().pixelShader->GetCommonShader() : nullptr);
    m_ngxSpsbReplaceFull = true;
    m_ngxSpsbPatchActive =
      m_ngxSpsbPatchVs.present || m_ngxSpsbPatchPs.present ||
      m_ngxSpsbPatchVs.clampPresent || m_ngxSpsbPatchPs.clampPresent;

    if (m_ngxSpsbPatchPs.clampPresent) {
      ONCE(Logger::info("[RTX NGX Passthrough] Motion blur clamp now covers the upscaled scene colour."));
    }

    // The constant buffer is shared by every shader on the stage.
    // A register patched for one shader is re-uploaded when the next shader patches a different register.
    // It is also re-uploaded when the next shader has no patch.
    const int32_t vsReg = ngxPatchRegisterKey(m_ngxSpsbPatchVs, true);
    const int32_t psReg = ngxPatchRegisterKey(m_ngxSpsbPatchPs, true);

    if (vsReg != m_ngxSpsbLastVsReg) {
      m_parent->m_consts[DxsoProgramTypes::VertexShader].dirty = true;
      m_ngxSpsbLastVsReg = vsReg;
    }
    if (psReg != m_ngxSpsbLastPsReg) {
      m_parent->m_consts[DxsoProgramTypes::PixelShader].dirty = true;
      m_ngxSpsbLastPsReg = psReg;
    }
  }

  void D3D9Rtx::noteNgxPostUpscaleViewport() {
    D3DVIEWPORT9 viewport = d3d9State().viewport;
    const bool expand = GetNgxPassthroughExpandedViewport(&viewport);
    const uint32_t width = expand ? viewport.Width : 0;
    const uint32_t height = expand ? viewport.Height : 0;
    if (expand == m_ngxViewportExpandApplied &&
        width == m_ngxExpandedViewportWidth &&
        height == m_ngxExpandedViewportHeight) {
      return;
    }

    m_ngxViewportExpandApplied = expand;
    m_ngxExpandedViewportWidth = width;
    m_ngxExpandedViewportHeight = height;
    m_parent->m_flags.set(D3D9DeviceFlag::DirtyViewportScissor);
  }

  bool D3D9Rtx::ngxDepthTestDisabled() const {
    return d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_FALSE ||
           d3d9State().renderStates[D3DRS_ZFUNC] == D3DCMP_ALWAYS;
  }

  bool D3D9Rtx::GetNgxPassthroughExpandedViewport(D3DVIEWPORT9* viewport) const {
    if (!m_ngxPrePostUpscaled || viewport == nullptr || m_ngxSceneViewport.Width == 0 || m_ngxSceneViewport.Height == 0)
      return false;

    // Depth-tested draws test against the scene depth, which exists only in the screen-percentage rect.
    // A full-target viewport would rasterize them across the whole image.
    if (!ngxDepthTestDisabled())
      return false;

    if (d3d9State().renderTargets[kRenderTargetIndex] == nullptr)
      return false;

    D3D9CommonTexture* renderTarget = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
    if (renderTarget == nullptr || renderTarget->GetImage() == nullptr)
      return false;

    const auto withinPixels = [](uint32_t a, uint32_t b) {
      return a + kNgxViewportMatchPixels >= b && b + kNgxViewportMatchPixels >= a;
    };
    const D3DVIEWPORT9& scene = m_ngxSceneViewport;
    const bool sizeMatchesScene =
      withinPixels(viewport->Width, scene.Width) && withinPixels(viewport->Height, scene.Height);
    const bool atSceneRect =
      sizeMatchesScene && withinPixels(viewport->X, scene.X) && withinPixels(viewport->Y, scene.Y);

    // A final pass on the backbuffer draws at the centred view rect from ScaleScreenCoords.
    bool atCentredRect = false;
    if (sizeMatchesScene && m_activePresentParams.has_value() && m_ngxFrameBackbufferImage != nullptr &&
        renderTarget->GetImage().ptr() == m_ngxFrameBackbufferImage.ptr()) {
      const uint32_t backBufferWidth = m_activePresentParams->BackBufferWidth;
      const uint32_t backBufferHeight = m_activePresentParams->BackBufferHeight;
      atCentredRect =
        backBufferWidth > scene.Width && backBufferHeight > scene.Height &&
        withinPixels(viewport->X, (backBufferWidth - scene.Width) / 2) &&
        withinPixels(viewport->Y, (backBufferHeight - scene.Height) / 2);
    }

    if (!atSceneRect && !atCentredRect)
      return false;

    const VkExtent3D& extent = renderTarget->GetImage()->info().extent;

    // The viewport already covers this target.
    if (extent.width <= viewport->Width + 1 && extent.height <= viewport->Height + 1)
      return false;

    viewport->X = 0;
    viewport->Y = 0;
    viewport->Width = extent.width;
    viewport->Height = extent.height;
    return true;
  }

  // A title that stops clearing depth mid-scene is not held to the clear after this many frames.
  static constexpr uint32_t kNgxMidSceneClearMemoryFrames = 120;

  bool D3D9Rtx::ngxPastForegroundClear() const {
    if (m_ngxDepthClearsThisFrame > 0)
      return true;

    return !m_ngxMidSceneClearSeen ||
           m_ue3FrameCounter - m_ngxLastMidSceneClearFrame > kNgxMidSceneClearMemoryFrames;
  }

  bool D3D9Rtx::ngxBoundPixelIsUpscaleBoundary() {
    Ue3ShaderFeatureInfo info;
    if (!tryGetBoundPixelFeatures(info))
      return false;

    return info.hasHazeConstants ||
           info.looksLikeDofAndBloomPostProcess() ||
           info.hasToneMapConstants ||
           info.hasGammaConstants ||
           info.hasExposureOrToneSampler;
  }

  void D3D9Rtx::stretchNgxVelocityBuffer() {
    if (m_ngxVelocityStretchedThisFrame || m_ngxSceneColorImage == nullptr)
      return;

    Ue3ShaderFeatureInfo info;
    if (!tryGetBoundPixelFeatures(info) || info.velocitySamplerIndex >= caps::MaxTexturesPS)
      return;

    D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[info.velocitySamplerIndex]);
    if (texture == nullptr || texture->GetImage() == nullptr)
      return;

    m_ngxVelocityStretchedThisFrame = true;

    // The velocity pass renders into the same subrect as the scene, so only a scene-sized buffer can be scaled.
    const Rc<DxvkImage>& image = texture->GetImage();
    const VkExtent3D& extent = image->info().extent;
    const VkExtent3D& sceneExtent = m_ngxSceneColorImage->info().extent;
    if (extent.width != sceneExtent.width || extent.height != sceneExtent.height) {
      ONCE(Logger::info(str::format(
        "[RTX NGX Passthrough] The ", extent.width, "x", extent.height,
        " velocity buffer is not scene-sized. Motion blur reads it as rendered.")));
      return;
    }

    if (stretchNgxSubrectToFull(image)) {
      ONCE(Logger::info(str::format(
        "[RTX NGX Passthrough] Motion blur velocity scaled from ", m_ngxSceneViewport.Width, "x",
        m_ngxSceneViewport.Height, " onto ", extent.width, "x", extent.height, ".")));
    }
  }

  bool D3D9Rtx::stretchNgxSubrectToFull(const Rc<DxvkImage>& image) {
    if (image == nullptr || m_ngxSceneViewport.Width == 0 || m_ngxSceneViewport.Height == 0)
      return false;

    const VkExtent3D& fullExtent = image->info().extent;
    if (fullExtent.width == 0 || fullExtent.height == 0)
      return false;

    if (m_ngxSceneViewport.X >= fullExtent.width || m_ngxSceneViewport.Y >= fullExtent.height)
      return false;

    const uint32_t subWidth = std::min(m_ngxSceneViewport.Width, fullExtent.width - m_ngxSceneViewport.X);
    const uint32_t subHeight = std::min(m_ngxSceneViewport.Height, fullExtent.height - m_ngxSceneViewport.Y);
    if (subWidth + 1 >= fullExtent.width && subHeight + 1 >= fullExtent.height)
      return true;

    const DxvkImageCreateInfo& sourceInfo = image->info();
    const VkImageUsageFlags transferUsage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if ((sourceInfo.usage & transferUsage) != transferUsage)
      return false;

    if (m_ngxSubrectStretchCopy == nullptr ||
        m_ngxSubrectStretchCopy->info().format != sourceInfo.format ||
        m_ngxSubrectStretchCopy->info().extent.width != subWidth ||
        m_ngxSubrectStretchCopy->info().extent.height != subHeight) {
      DxvkImageCreateInfo imageInfo;
      imageInfo.type = VK_IMAGE_TYPE_2D;
      imageInfo.format = sourceInfo.format;
      imageInfo.flags = 0;
      imageInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
      imageInfo.extent = { subWidth, subHeight, 1 };
      imageInfo.numLayers = 1;
      imageInfo.mipLevels = 1;
      imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      imageInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
      imageInfo.access = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
      imageInfo.layout = VK_IMAGE_LAYOUT_GENERAL;

      m_ngxSubrectStretchCopy = m_parent->GetDXVKDevice()->createImage(
        imageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        DxvkMemoryStats::Category::RTXRenderTarget, "NGX subrect stretch");
    }

    const Rc<DxvkImage> subrectCopy = m_ngxSubrectStretchCopy;
    const VkOffset3D subOffset = {
      int32_t(m_ngxSceneViewport.X), int32_t(m_ngxSceneViewport.Y), 0
    };
    const VkExtent3D subExtent = { subWidth, subHeight, 1 };
    const DxvkFormatInfo* formatInfo = imageFormatInfo(sourceInfo.format);
    if (formatInfo == nullptr)
      return false;

    m_parent->EmitCs([
      cImage = image,
      cSubrectCopy = subrectCopy,
      cSubOffset = subOffset,
      cSubExtent = subExtent,
      cFullExtent = fullExtent,
      cAspect = formatInfo->aspectMask
    ](DxvkContext* ctx) {
      ctx->changeImageLayout(cSubrectCopy, VK_IMAGE_LAYOUT_GENERAL);

      const VkImageSubresourceLayers layers = { cAspect, 0, 0, 1 };
      ctx->copyImage(cSubrectCopy, layers, VkOffset3D { 0, 0, 0 },
                     cImage, layers, cSubOffset, cSubExtent);

      VkImageBlit blitInfo = {};
      blitInfo.srcSubresource = layers;
      blitInfo.dstSubresource = layers;
      blitInfo.srcOffsets[0] = { 0, 0, 0 };
      blitInfo.srcOffsets[1] = { int32_t(cSubExtent.width), int32_t(cSubExtent.height), 1 };
      blitInfo.dstOffsets[0] = { 0, 0, 0 };
      blitInfo.dstOffsets[1] = { int32_t(cFullExtent.width), int32_t(cFullExtent.height), 1 };

      const VkComponentMapping identity = {
        VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY
      };
      ctx->blitImage(cImage, identity, cSubrectCopy, identity, blitInfo, VK_FILTER_LINEAR);
    });

    return true;
  }

  namespace {

  // Returns the float-constant register whose name contains the token.
  // Writes the array length when count is not null.
  // Returns UINT32_MAX when the shader has no match.
  uint32_t findShaderFloatConstant(const D3D9CommonShader* shader, const char* token, uint32_t* count) {
    if (count != nullptr)
      *count = 0;

    if (shader == nullptr || token == nullptr)
      return UINT32_MAX;

    try {
      const auto& bytecode = shader->GetBytecode();
      if (bytecode.size() < sizeof(uint32_t) || (bytecode.size() % sizeof(uint32_t)) != 0)
        return UINT32_MAX;

      const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
      const uint32_t headerToken = tokens[0];
      const uint32_t headerTypeMask = headerToken & 0xffff0000u;

      DxsoProgramType programType;
      if (headerTypeMask == 0xffff0000u)
        programType = DxsoProgramTypes::PixelShader;
      else if (headerTypeMask == 0xfffe0000u)
        programType = DxsoProgramTypes::VertexShader;
      else
        return UINT32_MAX;

      DxsoProgramInfo programInfo {
        programType, headerToken & 0xffu, (headerToken >> 8) & 0xffu
      };
      DxsoDecodeContext decoder(programInfo);
      DxsoCodeIter iter(tokens + 1);
      while (decoder.decodeInstruction(iter)) {
        if (decoder.getCtabInfo().m_size != 0)
          break;
      }

      const DxsoCtab& ctab = decoder.getCtabInfo();
      if (ctab.m_size == 0 || ctab.m_constantData.empty())
        return UINT32_MAX;

      for (const DxsoCtab::Constant& constant : ctab.m_constantData) {
        std::string name;
        name.reserve(constant.name.size());
        for (const char ch : constant.name)
          name.push_back(char(std::tolower(static_cast<unsigned char>(ch))));
        if (name.find(token) == std::string::npos)
          continue;

        if (count != nullptr)
          *count = constant.registerCount;
        return constant.registerIndex;
      }
    } catch (...) {
      return UINT32_MAX;
    }

    return UINT32_MAX;
  }

  }

  bool D3D9Rtx::ngxBoundDeclIsFilterVertex() const {
    if (d3d9State().vertexDecl == nullptr)
      return false;

    bool positionAtStart = false;
    bool uvAfterPosition = false;
    for (const D3DVERTEXELEMENT9& element : d3d9State().vertexDecl->GetElements()) {
      if (element.Stream != 0 || element.UsageIndex != 0)
        continue;

      if (element.Usage == D3DDECLUSAGE_POSITION)
        positionAtStart = element.Offset == 0 && element.Type == D3DDECLTYPE_FLOAT4;
      else if (element.Usage == D3DDECLUSAGE_TEXCOORD)
        uvAfterPosition = element.Offset == 16 && element.Type == D3DDECLTYPE_FLOAT2;
    }
    return positionAtStart && uvAfterPosition;
  }

  void D3D9Rtx::RetargetNgxUpscaledPostQuad(void* vertices, uint32_t vertexCount, uint32_t stride) {
    if (!m_ngxPrePostUpscaled || vertices == nullptr || vertexCount != 4 || stride < 24 ||
        m_ngxSceneColorImage == nullptr || m_ngxSceneViewport.Width == 0 || m_ngxSceneViewport.Height == 0 ||
        !ngxBoundDeclIsFilterVertex()) {
      return;
    }

    struct Vertex {
      float x, y, z, w;
      float u, v;
    };

    const VkExtent3D& extent = m_ngxSceneColorImage->info().extent;
    if (extent.width == 0 || extent.height == 0)
      return;

    const float u0 = float(m_ngxSceneViewport.X) / float(extent.width);
    const float u1 = float(m_ngxSceneViewport.X + m_ngxSceneViewport.Width) / float(extent.width);
    const float v0 = float(m_ngxSceneViewport.Y) / float(extent.height);
    const float v1 = float(m_ngxSceneViewport.Y + m_ngxSceneViewport.Height) / float(extent.height);
    const float clipLeft = -1.0f + 2.0f * u0;
    const float clipRight = -1.0f + 2.0f * u1;
    const float clipTop = 1.0f - 2.0f * v0;
    const float clipBottom = 1.0f - 2.0f * v1;

    bool targetsBackbuffer = false;
    uint32_t targetWidth = 0;
    uint32_t targetHeight = 0;
    if (d3d9State().renderTargets[kRenderTargetIndex] != nullptr) {
      D3D9CommonTexture* renderTarget = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
      if (renderTarget != nullptr && renderTarget->GetImage() != nullptr) {
        targetWidth = renderTarget->GetImage()->info().extent.width;
        targetHeight = renderTarget->GetImage()->info().extent.height;
        targetsBackbuffer = m_ngxFrameBackbufferImage != nullptr &&
                            renderTarget->GetImage().ptr() == m_ngxFrameBackbufferImage.ptr();
      }
    }

    // ScaleScreenCoords centres the same view on the backbuffer.
    bool centredKnown = false;
    float centredLeft = 0.0f, centredRight = 0.0f, centredTop = 0.0f, centredBottom = 0.0f;
    if (targetsBackbuffer && m_activePresentParams.has_value()) {
      const uint32_t backBufferWidth = m_activePresentParams->BackBufferWidth;
      const uint32_t backBufferHeight = m_activePresentParams->BackBufferHeight;
      if (backBufferWidth > m_ngxSceneViewport.Width && backBufferHeight > m_ngxSceneViewport.Height) {
        const uint32_t centredX = (backBufferWidth - m_ngxSceneViewport.Width) / 2;
        const uint32_t centredY = (backBufferHeight - m_ngxSceneViewport.Height) / 2;
        centredLeft = -1.0f + 2.0f * float(centredX) / float(backBufferWidth);
        centredRight = -1.0f + 2.0f * float(centredX + m_ngxSceneViewport.Width) / float(backBufferWidth);
        centredTop = 1.0f - 2.0f * float(centredY) / float(backBufferHeight);
        centredBottom = 1.0f - 2.0f * float(centredY + m_ngxSceneViewport.Height) / float(backBufferHeight);
        centredKnown = true;
      }
    }

    Vertex* quad[4];
    float minX = 0.0f, maxX = 0.0f, minY = 0.0f, maxY = 0.0f;
    float minU = 0.0f, maxU = 0.0f, minV = 0.0f, maxV = 0.0f;

    for (uint32_t i = 0; i < 4; i++) {
      quad[i] = reinterpret_cast<Vertex*>(reinterpret_cast<uint8_t*>(vertices) + size_t(i) * stride);

      // Filter quads are pre-transformed, with w near 1.
      // World draws fail this and are left alone.
      if (!std::isfinite(quad[i]->w) || quad[i]->w < 0.5f || quad[i]->w > 2.0f)
        return;

      if (i == 0) {
        minX = maxX = quad[i]->x;
        minY = maxY = quad[i]->y;
        minU = maxU = quad[i]->u;
        minV = maxV = quad[i]->v;
      } else {
        minX = std::min(minX, quad[i]->x);
        maxX = std::max(maxX, quad[i]->x);
        minY = std::min(minY, quad[i]->y);
        maxY = std::max(maxY, quad[i]->y);
        minU = std::min(minU, quad[i]->u);
        maxU = std::max(maxU, quad[i]->u);
        minV = std::min(minV, quad[i]->v);
        maxV = std::max(maxV, quad[i]->v);
      }
    }

    const auto close = [](float a, float b) { return std::abs(a - b) < 0.08f; };
    const bool posMatchesSubrect =
      close(minX, std::min(clipLeft, clipRight)) && close(maxX, std::max(clipLeft, clipRight)) &&
      close(minY, std::min(clipTop, clipBottom)) && close(maxY, std::max(clipTop, clipBottom));
    const bool posFullscreen = close(minX, -1.0f) && close(maxX, 1.0f) && close(minY, -1.0f) && close(maxY, 1.0f);
    const bool uvMatchesSubrect = close(minU, u0) && close(maxU, u1) && close(minV, v0) && close(maxV, v1);
    const bool posMatchesCentred =
      centredKnown && close(minX, centredLeft) && close(maxX, centredRight) &&
      close(minY, centredBottom) && close(maxY, centredTop);

    const auto remap = [](float value, float lo, float hi, float newLo, float newHi) {
      const float span = hi - lo;
      if (std::abs(span) < 1e-5f)
        return value;
      return newLo + (value - lo) / span * (newHi - newLo);
    };

    // Expands the quad from the given clip rect onto the whole target.
    // Each edge keeps its sub-pixel offset from that rect, which UE3 sets to half a pixel on D3D9.
    // An offset beyond a pixel means the rect was misjudged, and the D3D9 half pixel is used instead.
    const auto expandPositions = [&](float rectMinX, float rectMaxX, float rectMinY, float rectMaxY) {
      const float pixelX = targetWidth != 0 ? 2.0f / float(targetWidth) : 0.0f;
      const float pixelY = targetHeight != 0 ? 2.0f / float(targetHeight) : 0.0f;
      const auto edgeOffset = [](float offset, float pixel, float halfPixel) {
        return std::abs(offset) <= pixel ? offset : halfPixel;
      };

      const float newMinX = -1.0f + edgeOffset(minX - rectMinX, pixelX, -0.5f * pixelX);
      const float newMaxX = 1.0f + edgeOffset(maxX - rectMaxX, pixelX, -0.5f * pixelX);
      const float newMinY = -1.0f + edgeOffset(minY - rectMinY, pixelY, 0.5f * pixelY);
      const float newMaxY = 1.0f + edgeOffset(maxY - rectMaxY, pixelY, 0.5f * pixelY);
      for (Vertex* vertex : quad) {
        vertex->x = remap(vertex->x, minX, maxX, newMinX, newMaxX);
        vertex->y = remap(vertex->y, minY, maxY, newMinY, newMaxY);
      }
    };

    // SampleOffsets are added to InUV.xyyx.
    // .x and .w are U.
    // .y and .z are V.
    // Opening the UVs without scaling those offsets leaves the kernel on the pre-upscale texels.
    const auto scaleGatherSampleOffsets = [&](float scaleU, float scaleV) {
      if (!m_parent->UseProgrammableVS() || d3d9State().vertexShader.ptr() == nullptr)
        return;

      const D3D9CommonShader* vertexShader = d3d9State().vertexShader->GetCommonShader();
      uint32_t offsetCount = 0;
      const uint32_t offsetReg = findShaderFloatConstant(vertexShader, "sampleoffsets", &offsetCount);
      if (offsetReg >= caps::MaxFloatConstantsVS || offsetCount == 0)
        return;

      Direct3DState9* state = m_parent->GetRawState();
      const uint32_t count = std::min(offsetCount, caps::MaxFloatConstantsVS - offsetReg);
      for (uint32_t i = 0; i < count; i++) {
        Vector4& offset = state->vsConsts.fConsts[offsetReg + i];
        offset.x *= scaleU;
        offset.y *= scaleV;
        offset.z *= scaleV;
        offset.w *= scaleU;
      }
      m_parent->m_consts[DxsoProgramTypes::VertexShader].dirty = true;
    };

    // The gather reads scene colour through the quad UVs.
    // After the upscale those UVs cover the full image.
    const auto openGatherSceneUvs = [&]() {
      if (!uvMatchesSubrect) {
        ONCE(Logger::info(str::format(
          "[RTX NGX Passthrough] DOF/bloom gather UVs were not the scene subrect (uv ",
          minU, "..", maxU, ", ", minV, "..", maxV,
          ", scene ", u0, "..", u1, ", ", v0, "..", v1, ").")));
        noteNgxReducedRectDraw("A DOF/bloom gather read outside the scene subrect");
        return;
      }

      for (Vertex* vertex : quad) {
        vertex->u = remap(vertex->u, minU, maxU, 0.0f, 1.0f);
        vertex->v = remap(vertex->v, minV, maxV, 0.0f, 1.0f);
      }

      const float spanU = maxU - minU;
      const float spanV = maxV - minV;
      if (spanU <= 1e-4f || spanV <= 1e-4f)
        return;

      // A scale of 8 matches a 12.5 percent view, below any DLSS preset.
      scaleGatherSampleOffsets(std::min(1.0f / spanU, 8.0f), std::min(1.0f / spanV, 8.0f));
    };

    // Filter-buffer positions stay on the texels the gather wrote.
    // Only the gather's scene UVs open onto the upscaled image.
    if (ngxBoundImageIsDofFilterBuffer() || ngxBoundPixelIsDofGather()) {
      if (ngxBoundPixelIsDofGather())
        openGatherSceneUvs();
      return;
    }

    const bool dofBlend = ngxBoundPixelIsDofBlend();
    Ue3ShaderFeatureInfo blendInfo;
    const bool uberBlend = dofBlend && tryGetBoundPixelFeatures(blendInfo) && blendInfo.hasToneMapConstants;

    // The scene viewport is the top-left of scene colour.
    // When NeedsUpscale() reads false, the final composite draws the same view at the centred backbuffer rect.
    // An uber blend that is reduced on both axes is taken as that composite even off-centre.
    const float spanX = maxX - minX;
    const float spanY = maxY - minY;
    const bool reducedOnBothAxes = spanX < 1.85f && spanY < 1.85f;
    const bool centredComposite = !posFullscreen && !posMatchesSubrect &&
      (posMatchesCentred || (uberBlend && reducedOnBothAxes));

    if (posMatchesSubrect) {
      expandPositions(std::min(clipLeft, clipRight), std::max(clipLeft, clipRight),
                      std::min(clipTop, clipBottom), std::max(clipTop, clipBottom));
    } else if (posMatchesCentred) {
      expandPositions(centredLeft, centredRight, centredBottom, centredTop);
    } else if (centredComposite) {
      expandPositions(minX, maxX, minY, maxY);
    }

    // The blend UV addresses the gathered filter texels.
    // Those texels coincide with the scene subrect.
    // Stretching them reads the rest of the filter buffer, which the gather leaves empty.
    // Scene colour comes from the clip position.
    // Growing the quad is enough.
    if (!dofBlend && uvMatchesSubrect && (posMatchesSubrect || posFullscreen || centredComposite)) {
      for (Vertex* vertex : quad) {
        vertex->u = remap(vertex->u, minU, maxU, 0.0f, 1.0f);
        vertex->v = remap(vertex->v, minV, maxV, 0.0f, 1.0f);
      }
    }

    // The gamma copy and the uber blend have to cover the render target.
    // A viewport left at the view rect keeps that copy inside the reduced area.
    // The uber blend derives scene UVs from clip position.
    // Its scale bias has to match a full-screen quad.
    const bool coverTarget =
      (!dofBlend && (posMatchesSubrect || centredComposite || (posFullscreen && uvMatchesSubrect))) ||
      (uberBlend && (posMatchesSubrect || posFullscreen || centredComposite));
    if (coverTarget && uberBlend &&
        m_parent->UseProgrammableVS() && d3d9State().vertexShader.ptr() != nullptr) {
      const D3D9CommonShader* vertexShader = d3d9State().vertexShader->GetCommonShader();
      const uint32_t sceneUvReg = findShaderFloatConstant(vertexShader, "scenecoordinatescalebias", nullptr);

      if (sceneUvReg < caps::MaxFloatConstantsVS) {
        Direct3DState9* state = m_parent->GetRawState();
        state->vsConsts.fConsts[sceneUvReg] = ngxFullBufferScaleBias();
        m_parent->m_consts[DxsoProgramTypes::VertexShader].dirty = true;
        ONCE(Logger::info("[RTX NGX Passthrough] Uber post scene UVs now cover the upscaled scene color."));
      }
    }

    if (coverTarget &&
        d3d9State().renderTargets[kRenderTargetIndex] != nullptr) {
      D3D9CommonTexture* renderTarget = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
      if (renderTarget != nullptr && renderTarget->GetImage() != nullptr) {
        const VkExtent3D& targetExtent = renderTarget->GetImage()->info().extent;
        Direct3DState9* state = m_parent->GetRawState();
        state->renderStates[D3DRS_SCISSORTESTENABLE] = FALSE;
        m_parent->m_flags.set(D3D9DeviceFlag::DirtyViewportScissor);
        if (state->viewport.Width + 1 < targetExtent.width ||
            state->viewport.Height + 1 < targetExtent.height) {
          state->viewport.X = 0;
          state->viewport.Y = 0;
          state->viewport.Width = targetExtent.width;
          state->viewport.Height = targetExtent.height;
        }
      }
    }

    if (dofBlend && !uberBlend && !posMatchesSubrect && !posFullscreen && !centredComposite) {
      ONCE(Logger::info(str::format(
        "[RTX NGX Passthrough] DOF/bloom blend quad stayed on a reduced rect after the upscale (clip ",
        minX, "..", maxX, ", ", minY, "..", maxY,
        ", scene ", std::min(clipLeft, clipRight), "..", std::max(clipLeft, clipRight),
        ", ", std::min(clipTop, clipBottom), "..", std::max(clipTop, clipBottom), ").")));
      noteNgxReducedRectDraw("A DOF/bloom blend quad stayed on a reduced rect");
    }
  }

  // Returns true for the opaque fullscreen blit that lands the scene on the backbuffer.
  // The source is the scene-color resolve, or any other full-frame target.
  bool D3D9Rtx::ngxDrawIsBackbufferSceneComposite(const DrawContext& drawContext) const {
    if (drawContext.PrimitiveCount == 0 || drawContext.PrimitiveCount > 4 ||
        !m_activePresentParams.has_value() ||
        d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] != FALSE) {
      return false;
    }

    const bool depthTestDisabled = d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_FALSE ||
                                   d3d9State().renderStates[D3DRS_ZFUNC] == D3DCMP_ALWAYS;
    if (!depthTestDisabled || d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE) {
      return false;
    }

    const uint32_t backBufferWidth = m_activePresentParams->BackBufferWidth;
    const uint32_t backBufferHeight = m_activePresentParams->BackBufferHeight;

    // After an in-place upscale, a composite at the scene or centred view rect rasterizes over the whole backbuffer.
    D3DVIEWPORT9 vp = d3d9State().viewport;
    GetNgxPassthroughExpandedViewport(&vp);
    if (backBufferWidth == 0 || backBufferHeight == 0 ||
        vp.Width + 1 < backBufferWidth || vp.Height + 1 < backBufferHeight) {
      return false;
    }

    const uint32_t rtSamplerMask = m_parent->GetActiveRTTextures() & m_parent->m_psShaderMasks.samplerMask;
    for (const uint32_t i : bit::BitMask(rtSamplerMask)) {
      D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[i]);
      if (texture == nullptr || texture->GetImage() == nullptr) {
        continue;
      }

      const DxvkImage* sampledImage = texture->GetImage().ptr();
      if (m_ngxFrameBackbufferImage != nullptr && sampledImage == m_ngxFrameBackbufferImage.ptr()) {
        continue;
      }

      if (matchNgxSceneColorSample(sampledImage) != nullptr) {
        return true;
      }

      const VkExtent3D& extent = sampledImage->info().extent;
      if (extent.width + 1 >= backBufferWidth && extent.height + 1 >= backBufferHeight) {
        return true;
      }
    }

    return false;
  }

  bool D3D9Rtx::ngxDrawFollowsSceneComposite(const DrawContext& drawContext) const {
    if (!m_ngxHudlessSceneCompositeSeen || m_ngxFrameBackbufferImage == nullptr ||
        d3d9State().renderTargets[kRenderTargetIndex] == nullptr)
      return false;

    D3D9CommonTexture* target = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
    return target != nullptr && target->GetImage().ptr() == m_ngxFrameBackbufferImage.ptr() &&
           !ngxDrawIsBackbufferSceneComposite(drawContext);
  }

  // Logs a site change immediately.
  // The same site repeats at most once every 300 frames.
  void D3D9Rtx::noteNgxHudlessCapture(NgxHudlessCaptureSite site) {
    constexpr uint32_t kNgxHudlessLogInterval = 300;
    const bool siteChanged = !m_ngxHudlessCaptureLogValid || site != m_ngxHudlessLoggedSite;
    if (!siteChanged && m_ue3FrameCounter - m_ngxHudlessLastLogFrame < kNgxHudlessLogInterval) {
      return;
    }

    m_ngxHudlessCaptureLogValid = true;
    m_ngxHudlessLoggedSite = site;
    m_ngxHudlessLastLogFrame = m_ue3FrameCounter;

    if (site == NgxHudlessCaptureSite::FrameEnd) {
      Logger::info("[RTX NGX Passthrough] HUD-less capture at frame end. No draw followed the scene composite.");
      return;
    }

    const char* where = site == NgxHudlessCaptureSite::LateInjection
      ? "at late injection"
      : "after the scene composite";
    Logger::info(str::format(
      "[RTX NGX Passthrough] HUD-less capture before draw ", m_drawCallID, ", ", where, "."));
  }

  // Records DLSS-NR and the HUD-less copy on the first draw after the scene blit.
  void D3D9Rtx::maybeFinishNgxScene(const DrawContext& drawContext) {
    // Late injection runs DLSS-NR and takes the HUD-less copy in its own dispatch.
    const bool needsNeuralRendering =
      m_frameOptions.ngxNeuralRendering && !m_ngxNeuralRenderingAppliedThisFrame && m_ngxColorTargetImage != nullptr;
    const bool needsHudless = m_frameOptions.ngxDlfgHudless && !m_ngxHudlessCapturedThisFrame;
    if ((!needsNeuralRendering && !needsHudless) || m_ngxFrameBackbufferImage == nullptr) {
      return;
    }

    if (drawContext.PrimitiveCount == 0 || d3d9State().renderTargets[kRenderTargetIndex] == nullptr) {
      return;
    }

    D3D9CommonTexture* renderTargetTexture = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
    if (renderTargetTexture == nullptr || renderTargetTexture->GetImage().ptr() != m_ngxFrameBackbufferImage.ptr()) {
      return;
    }

    // This draw is the scene blit, so the backbuffer does not hold it yet.
    if (ngxDrawIsBackbufferSceneComposite(drawContext)) {
      m_ngxHudlessSceneCompositeSeen = true;
      return;
    }

    if (!m_ngxHudlessSceneCompositeSeen) {
      return;
    }

    if (needsNeuralRendering) {
      m_ngxNeuralRenderingAppliedThisFrame = true;
      m_parent->EmitCs([cBackbuffer = m_ngxFrameBackbufferImage](DxvkContext* ctx) {
        static_cast<RtxContext*>(ctx)->applyNgxPassthroughNeuralRendering(cBackbuffer);
      });
    }

    if (needsHudless) {
      m_ngxHudlessCapturedThisFrame = true;
      noteNgxHudlessCapture(NgxHudlessCaptureSite::AfterSceneComposite);

      m_parent->EmitCs([cBackbuffer = m_ngxFrameBackbufferImage](DxvkContext* ctx) {
        static_cast<RtxContext*>(ctx)->captureNgxPassthroughHudless(cBackbuffer);
      });
    }
  }

  // Positions the next frame's injection from the shape this one turned out to have, and logs that
  // shape a bounded number of times so a placement can be checked in a title whose frame shape has
  // not been seen before. Called once per frame, before the shape is cleared.
  void D3D9Rtx::updateNgxPrePostInjectionAim() {
    if (m_ngxFrameShapeLogsRemaining > 0 && m_ngxFrameReadCount > 0 &&
        (m_ue3FrameCounter % kNgxFrameShapeLogInterval) == 0) {
      m_ngxFrameShapeLogsRemaining--;

      std::string clears;
      for (uint32_t i = 0; i < m_ngxFrameClearCount; i++) {
        clears += (i != 0 ? ", " : "") + std::to_string(m_ngxFrameClearDraws[i]);
      }

      std::string reads;
      for (uint32_t i = 0; i < m_ngxFrameReadCount; i++) {
        reads += (i != 0 ? ", " : "") + std::to_string(m_ngxFrameReadDraws[i]);
      }

      // A correct placement has injectedAtDraw equal to the first read past lastDepthWritingDraw
      Logger::info(str::format(
        "[RTX NGX Passthrough][frame shape] frame=", m_ue3FrameCounter,
        " sceneDepthClearsAtDraw={", clears, "}",
        " lastDepthWritingDraw=", m_ngxFrameLastGeometryDraw,
        " sceneColorReadsAtDraw={", reads, "}",
        " injectedAtDraw=", m_ngxFrameInjectionDraw,
        " firstBackbufferDraw=", m_ngxFrameFirstBackbufferDraw));
    }

    if (m_ngxFrameLastGeometryDraw != 0 && m_ngxFrameReadCount != 0) {
      uint32_t readsInsideScene = 0;
      while (readsInsideScene < m_ngxFrameReadCount &&
             m_ngxFrameReadDraws[readsInsideScene] < m_ngxFrameLastGeometryDraw) {
        readsInsideScene++;
      }

      // Two consecutive frames must agree before a count is raised.
      // Frames can alternate between two and three scene-colour reads.
      // Chasing that alternation either injects inside the longer frame or misses the shorter one.
      // A lower count is adopted immediately.
      // A post-process volume that removes a pass would otherwise leave the aim past the end of the chain.
      // Every frame in that volume then goes late.
      // A higher count still waits for the second frame.
      // One long frame cannot inject before the foreground depth priority group.
      const bool lowerInside = readsInsideScene < m_ngxPrePostReadsInsideScene;
      if (readsInsideScene != m_ngxPrePostReadsInsideScene &&
          (lowerInside || readsInsideScene == m_ngxPrevReadsInsideScene)) {
        m_ngxPrePostReadsInsideScene = readsInsideScene;
      }
      m_ngxPrevReadsInsideScene = readsInsideScene;

      if (m_ngxFrameClearCount > 0) {
        const uint32_t lastClearDraw = m_ngxFrameClearDraws[m_ngxFrameClearCount - 1];
        uint32_t readsAfterClearInsideScene = 0;

        for (uint32_t i = 0; i < m_ngxFrameReadCount; i++) {
          if (m_ngxFrameReadDraws[i] > lastClearDraw &&
              m_ngxFrameReadDraws[i] < m_ngxFrameLastGeometryDraw) {
            readsAfterClearInsideScene++;
          }
        }

        const bool lowerAfterClear = readsAfterClearInsideScene < m_ngxReadsAfterClearInsideScene;
        if (readsAfterClearInsideScene != m_ngxReadsAfterClearInsideScene &&
            (lowerAfterClear || readsAfterClearInsideScene == m_ngxPrevReadsAfterClearInsideScene)) {
          Logger::info(str::format(
            "[RTX NGX Passthrough] Pre-post-process injection aimed at scene color read ",
            readsAfterClearInsideScene + 1, " after the scene depth clear: ",
            readsAfterClearInsideScene, " of them fall inside the scene."));

          m_ngxReadsAfterClearInsideScene = readsAfterClearInsideScene;
        }
        m_ngxPrevReadsAfterClearInsideScene = readsAfterClearInsideScene;
      }
    }

    m_ngxReadsSinceLastClear = 0;
    m_ngxFrameClearCount = 0;
    m_ngxFrameReadCount = 0;
    m_ngxFrameLastGeometryDraw = 0;
    m_ngxFrameInjectionDraw = 0;
    m_ngxFrameFirstBackbufferDraw = 0;
    m_ngxPrePostCandidatesThisFrame = 0;
  }

  bool D3D9Rtx::ngxPrePostInjectionAllowed() const {
    if (!m_frameOptions.ngxPrePostProcess)
      return false;

    const int debugVis = m_frameOptions.ngxDebugVisualization;
    if (debugVis != 0 && debugVis != int(dxvk::RtxNgxPassthrough::DebugVisualization::SceneColor))
      return false;

    if (m_ngxSubrectPrePostSuspended && m_ngxSceneColorImage != nullptr) {
      const VkExtent3D& extent = m_ngxSceneColorImage->info().extent;
      if (ngxSceneViewportIsSubrect(m_ngxSceneViewport, extent.width, extent.height))
        return false;
    }

    return true;
  }

  bool D3D9Rtx::ngxEarlyInjectionSuspended() const {
    return m_ue3FrameCounter < m_ngxEarlyInjectionSuspendedUntil;
  }

  std::string D3D9Rtx::describeNgxDraw(const DrawContext& drawContext) const {
    uint32_t targetWidth = 0;
    uint32_t targetHeight = 0;
    if (d3d9State().renderTargets[kRenderTargetIndex] != nullptr) {
      D3D9CommonTexture* target = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
      if (target != nullptr && target->GetImage() != nullptr) {
        targetWidth = target->GetImage()->info().extent.width;
        targetHeight = target->GetImage()->info().extent.height;
      }
    }

    const D3D9CommonShader* vertexShader = m_parent->UseProgrammableVS() && d3d9State().vertexShader.ptr() != nullptr
      ? d3d9State().vertexShader->GetCommonShader() : nullptr;
    const D3D9CommonShader* pixelShader = m_parent->UseProgrammablePS() && d3d9State().pixelShader.ptr() != nullptr
      ? d3d9State().pixelShader->GetCommonShader() : nullptr;

    const D3DVIEWPORT9& viewport = d3d9State().viewport;
    return str::format(
      "draw ", m_drawCallID, ", target ", targetWidth, "x", targetHeight,
      ", viewport ", viewport.Width, "x", viewport.Height, " at (", viewport.X, ", ", viewport.Y, "), ",
      drawContext.PrimitiveCount, " primitives, vs 0x", std::hex,
      (vertexShader != nullptr ? vertexShader->GetBytecodeHash() : 0),
      ", ps 0x", (pixelShader != nullptr ? pixelShader->GetBytecodeHash() : 0), std::dec);
  }

  void D3D9Rtx::checkNgxPostInjectionDraw(const DrawContext& drawContext) {
    if (m_ngxSceneColorImage == nullptr || d3d9State().renderTargets[kRenderTargetIndex] == nullptr)
      return;

    D3D9CommonTexture* target = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
    if (target == nullptr || target->GetImage() == nullptr)
      return;

    // A depth-tested mesh drawn into scene colour is scene geometry.
    if (!m_ngxSceneDrawAfterInjection && !m_ngxBackbufferDrawSeenThisFrame &&
        drawContext.PrimitiveCount > 4 && !ngxDepthTestDisabled() &&
        target->GetImage().ptr() == m_ngxSceneColorImage.ptr() &&
        d3d9State().depthStencil != nullptr && m_ngxSceneDepthImage != nullptr) {
      D3D9CommonTexture* depth = d3d9State().depthStencil->GetCommonTexture();
      if (depth != nullptr && depth->GetImage().ptr() == m_ngxSceneDepthImage.ptr()) {
        m_ngxSceneDrawAfterInjection = true;
        m_ngxSceneDrawEvidence = describeNgxDraw(drawContext);
      }
    }

    // A depth-off pass through a scene-sized viewport covers only the reduced rect unless that viewport expands.
    if (m_ngxPrePostUpscaled && !m_ngxReducedRectDrawThisFrame && ngxDepthTestDisabled()) {
      D3DVIEWPORT9 viewport = d3d9State().viewport;
      const VkExtent3D& extent = target->GetImage()->info().extent;
      const auto withinPixels = [](uint32_t a, uint32_t b) {
        return a + kNgxViewportMatchPixels >= b && b + kNgxViewportMatchPixels >= a;
      };
      const bool sceneSized =
        withinPixels(viewport.Width, m_ngxSceneViewport.Width) &&
        withinPixels(viewport.Height, m_ngxSceneViewport.Height);
      if (sceneSized && extent.width > viewport.Width + 1 && extent.height > viewport.Height + 1 &&
          !GetNgxPassthroughExpandedViewport(&viewport)) {
        m_ngxReducedRectDrawThisFrame = true;
        m_ngxReducedRectEvidence = describeNgxDraw(drawContext);
      }
    }
  }

  void D3D9Rtx::noteNgxReducedRectDraw(const char* reason) {
    if (m_ngxReducedRectDrawThisFrame)
      return;

    m_ngxReducedRectDrawThisFrame = true;
    m_ngxReducedRectEvidence = str::format(reason, " at draw ", m_drawCallID);
  }

  void D3D9Rtx::logNgxUe3Profile() {
    if (m_ngxProfileLogsRemaining == 0)
      return;

    const char* resolves = m_ngxResolvesFromStretchRect
      ? (m_ngxResolvesFromDraws ? "StretchRect and draw" : "StretchRect")
      : (m_ngxResolvesFromDraws ? "draw" : "none");

    const char* prePost = "late";
    switch (m_ngxFrameInjectionSite) {
    case NgxInjectionSite::Haze: prePost = "haze"; break;
    case NgxInjectionSite::Gather: prePost = "gather"; break;
    case NgxInjectionSite::AimedRead: prePost = "aimed read"; break;
    case NgxInjectionSite::FinalComposite: prePost = "final composite"; break;
    case NgxInjectionSite::Stretch: prePost = "stretch"; break;
    default: break;
    }

    bool sceneIsSubrect = false;
    if (m_ngxSceneViewportValid && m_activePresentParams.has_value()) {
      sceneIsSubrect = ngxSceneViewportIsSubrect(m_ngxSceneViewport,
                                                 m_activePresentParams->BackBufferWidth,
                                                 m_activePresentParams->BackBufferHeight);
    }
    const char* subFull = !sceneIsSubrect ? "off" : (m_ngxFrameUpscaledInPlace ? "in-place" : "late");

    const char* owner = !m_ngxGameSettingsRedirectsValid ? "unknown"
                      : (m_ngxRuntimeOwnedUpscale ? "runtime" : "engine");

    const bool earlySuspended = ngxEarlyInjectionSuspended();
    const char* suspended = earlySuspended
      ? (m_ngxSubrectPrePostSuspended ? "early and sub-full" : "early")
      : (m_ngxSubrectPrePostSuspended ? "sub-full" : "none");

    std::string profile = str::format(
      "resolves=", resolves, ", pre-post=", prePost, ", sub-full=", subFull,
      ", owner=", owner, ", suspended=", suspended);

    if (profile == m_ngxPendingProfile) {
      m_ngxPendingProfileFrames++;
    } else {
      m_ngxPendingProfile = std::move(profile);
      m_ngxPendingProfileFrames = 1;
    }

    if (m_ngxPendingProfileFrames == 2 && m_ngxPendingProfile != m_ngxLoggedProfile) {
      m_ngxProfileLogsRemaining--;
      m_ngxLoggedProfile = m_ngxPendingProfile;
      Logger::info(str::format("[RTX NGX Passthrough] UE3 profile: ", m_ngxLoggedProfile, ".",
                               m_ngxProfileLogsRemaining == 0 ? " Later profile changes are not logged." : ""));
    }
  }

  void D3D9Rtx::updateNgxInjectionFallbacks() {
    // A fallback needs two frames in a row, so one odd frame cannot move the injection.
    constexpr uint32_t kNgxFallbackStrikeFrames = 2;

    // Each repeat doubles the early suspension, up to 16 times the base length.
    // A title that always trips it settles on the later points.
    constexpr uint32_t kNgxEarlySuspensionFrames = 600;
    constexpr uint32_t kNgxEarlySuspensionMaxDoublings = 4;

    const bool earlySite =
      m_ngxFrameInjectionSite == NgxInjectionSite::Haze ||
      m_ngxFrameInjectionSite == NgxInjectionSite::Gather ||
      m_ngxFrameInjectionSite == NgxInjectionSite::AimedRead;

    if (earlySite) {
      m_ngxSceneAfterInjectionStrikes = m_ngxSceneDrawAfterInjection ? m_ngxSceneAfterInjectionStrikes + 1 : 0;
      if (m_ngxSceneAfterInjectionStrikes >= kNgxFallbackStrikeFrames) {
        m_ngxSceneAfterInjectionStrikes = 0;
        const uint32_t suspensionFrames = kNgxEarlySuspensionFrames
          << std::min(m_ngxEarlyInjectionSuspensions, kNgxEarlySuspensionMaxDoublings);
        m_ngxEarlyInjectionSuspensions++;
        m_ngxEarlyInjectionSuspendedUntil = m_ue3FrameCounter + suspensionFrames;
        Logger::warn(str::format(
          "[RTX NGX Passthrough] Scene geometry followed the pre-post injection on consecutive frames (",
          m_ngxSceneDrawEvidence, "). The haze, gather and aimed-read points are suspended for ",
          suspensionFrames, " frames."));
      }
    }

    if (m_ngxFrameUpscaledInPlace) {
      m_ngxReducedRectStrikes = m_ngxReducedRectDrawThisFrame ? m_ngxReducedRectStrikes + 1 : 0;
      if (m_ngxReducedRectStrikes >= kNgxFallbackStrikeFrames && !m_ngxSubrectPrePostSuspended) {
        m_ngxReducedRectStrikes = 0;
        m_ngxSubrectPrePostSuspended = true;
        m_ngxSuspendedScreenPercentage = m_ngxGameScreenPercentage;
        m_ngxSuspendedSceneExtent = m_ngxSceneColorImage != nullptr
          ? m_ngxSceneColorImage->info().extent : VkExtent3D { 0, 0, 0 };
        Logger::warn(str::format(
          "[RTX NGX Passthrough] A post pass kept the reduced view rect after the upscale on consecutive frames (",
          m_ngxReducedRectEvidence, "). Sub-full pre-post injection is suspended. DLSS upscales at the late point."));
      }
    }

    if (m_ngxSubrectPrePostSuspended) {
      const VkExtent3D sceneExtent = m_ngxSceneColorImage != nullptr
        ? m_ngxSceneColorImage->info().extent : VkExtent3D { 0, 0, 0 };
      if (m_ngxGameScreenPercentage != m_ngxSuspendedScreenPercentage ||
          sceneExtent.width != m_ngxSuspendedSceneExtent.width ||
          sceneExtent.height != m_ngxSuspendedSceneExtent.height) {
        m_ngxSubrectPrePostSuspended = false;
        Logger::info("[RTX NGX Passthrough] Sub-full pre-post injection resumes. ScreenPercentage or the scene extent changed.");
      }
    }

    m_ngxSceneDrawAfterInjection = false;
    m_ngxReducedRectDrawThisFrame = false;
    m_ngxFrameUpscaledInPlace = false;
  }

  Rc<DxvkImage> D3D9Rtx::matchNgxSceneColorSample(const DxvkImage* sampledImage) const {
    if (sampledImage == nullptr || m_ngxSceneColorImage == nullptr)
      return nullptr;

    if (sampledImage == m_ngxSceneColorImage.ptr())
      return m_ngxSceneColorImage;

    for (uint32_t r = 0; r < m_ngxSceneColorResolveCount; r++) {
      if (m_ngxSceneColorResolves[r].ptr() == sampledImage)
        return m_ngxSceneColorResolves[r];
    }

    return nullptr;
  }

  Rc<DxvkImage> D3D9Rtx::findNgxSceneColorSource() const {
    const DxvkImage* renderTargetImage = nullptr;
    if (d3d9State().renderTargets[kRenderTargetIndex] != nullptr) {
      D3D9CommonTexture* renderTarget = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
      if (renderTarget != nullptr)
        renderTargetImage = renderTarget->GetImage().ptr();
    }

    Rc<DxvkImage> best;
    bool bestIsFloat = false;
    uint64_t bestArea = 0;

    const uint32_t rtSamplerMask = m_parent->GetActiveRTTextures() & m_parent->m_psShaderMasks.samplerMask;
    for (const uint32_t i : bit::BitMask(rtSamplerMask)) {
      D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[i]);
      if (texture == nullptr || texture->GetImage() == nullptr)
        continue;

      const Rc<DxvkImage>& image = texture->GetImage();
      const Rc<DxvkImage> known = matchNgxSceneColorSample(image.ptr());
      if (known != nullptr)
        return known;

      if (image.ptr() == renderTargetImage ||
          (m_ngxFrameBackbufferImage != nullptr && image.ptr() == m_ngxFrameBackbufferImage.ptr()) ||
          m_ngxSceneViewport.Width == 0 || m_ngxSceneViewport.Height == 0)
        continue;

      const VkExtent3D& extent = image->info().extent;
      if (extent.width < m_ngxSceneViewport.X + m_ngxSceneViewport.Width ||
          extent.height < m_ngxSceneViewport.Y + m_ngxSceneViewport.Height)
        continue;

      const bool isFloat = RtxNgxPassthrough::isHdrColorFormat(image->info().format);
      const uint64_t area = uint64_t(extent.width) * extent.height;
      if (best == nullptr || (isFloat && !bestIsFloat) || (isFloat == bestIsFloat && area > bestArea)) {
        best = image;
        bestIsFloat = isFloat;
        bestArea = area;
      }
    }

    if (best != nullptr) {
      ONCE(Logger::info(str::format(
        "[RTX NGX Passthrough] Scene colour is read through a ", best->info().extent.width, "x",
        best->info().extent.height, " image that was never learned as a resolve. It is used as the scene-colour source.")));
    }
    return best;
  }

  void D3D9Rtx::recordNgxPrePostSceneColorRead() {
    m_ngxPrePostCandidatesThisFrame++;
    m_ngxReadsSinceLastClear++;

    if (m_ngxFrameReadCount < kNgxFrameShapeSlots)
      m_ngxFrameReadDraws[m_ngxFrameReadCount++] = m_drawCallID;
  }

  void D3D9Rtx::engageNgxPrePostInjection(const Rc<DxvkImage>& matchedTarget) {
    m_ngxColorTargetImage = matchedTarget;
    // When the consumed image is a resolve copy, mirror the DLSS output into the scene color
    // surface as well: later post passes may re-resolve from it (UE3 scene color resolves are
    // surface -> texture copies)
    m_ngxColorMirrorImage = (matchedTarget.ptr() != m_ngxSceneColorImage.ptr()) ? m_ngxSceneColorImage : nullptr;
    m_ngxSubrect.offset = { int32_t(m_ngxSceneViewport.X), int32_t(m_ngxSceneViewport.Y) };
    m_ngxSubrect.extent = { m_ngxSceneViewport.Width, m_ngxSceneViewport.Height };
    m_ngxColorSubrectOffset = m_ngxSubrect.offset;
    m_ngxFrameInjectionDraw = m_drawCallID;
  }

  Rc<DxvkImage> D3D9Rtx::findNgxPrePostSceneColorFromSamplers(bool applyOrdinalGate, bool recordRead) {
    const uint32_t rtSamplerMask = m_parent->GetActiveRTTextures() & m_parent->m_psShaderMasks.samplerMask;

    for (const uint32_t i : bit::BitMask(rtSamplerMask)) {
      D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[i]);
      if (texture == nullptr || texture->GetImage() == nullptr)
        continue;

      const Rc<DxvkImage> matchedTarget = matchNgxSceneColorSample(texture->GetImage().ptr());
      if (matchedTarget == nullptr)
        continue;

      if (recordRead)
        recordNgxPrePostSceneColorRead();

      if (applyOrdinalGate) {
        const bool haveClearThisFrame = m_ngxFrameClearCount > 0;

        const bool atInjectionPoint = haveClearThisFrame
          ? m_ngxReadsSinceLastClear >= m_ngxReadsAfterClearInsideScene + 1
          : m_ngxPrePostCandidatesThisFrame >= m_ngxPrePostReadsInsideScene + 1;

        if (!atInjectionPoint)
          break;
      }

      return matchedTarget;
    }

    return nullptr;
  }

  bool D3D9Rtx::ngxSceneViewportIsSubrect(const D3DVIEWPORT9& sceneViewport,
                                          uint32_t backBufferWidth, uint32_t backBufferHeight) {
    return RtxNgxPassthrough::extentIsReducedSubrect(
      sceneViewport.Width, sceneViewport.Height, backBufferWidth, backBufferHeight);
  }

  bool D3D9Rtx::ngxViewportMatchesDrivenScreenPercentage(const D3DVIEWPORT9& viewport) const {
    if (!m_activePresentParams.has_value())
      return false;

    const float screenPercentage = m_ngxGameScreenPercentage;
    if (!(screenPercentage > 0.0f && screenPercentage <= 100.0f))
      return false;

    const uint32_t backBufferWidth = m_activePresentParams->BackBufferWidth;
    const uint32_t backBufferHeight = m_activePresentParams->BackBufferHeight;
    if (backBufferWidth == 0 || backBufferHeight == 0)
      return false;

    const int32_t expectedWidth = int32_t(float(backBufferWidth) * screenPercentage / 100.0f);
    const int32_t expectedHeight = int32_t(float(backBufferHeight) * screenPercentage / 100.0f);
    return int32_t(viewport.Width) >= expectedWidth - kNgxDrivenViewportTolerance &&
           int32_t(viewport.Width) <= expectedWidth + kNgxDrivenViewportTolerance &&
           int32_t(viewport.Height) >= expectedHeight - kNgxDrivenViewportTolerance &&
           int32_t(viewport.Height) <= expectedHeight + kNgxDrivenViewportTolerance;
  }

  bool D3D9Rtx::ngxShaderIsFinishRenderViewTargetGamma(const Ue3ShaderFeatureInfo& psInfo) {
    return psInfo.gammaInverseReg != Ue3ShaderFeatureInfo::kNgxNoRegister &&
           psInfo.hasSceneColorSampler &&
           !psInfo.hasToneMapConstants &&
           !psInfo.hasExposureOrToneSampler;
  }

  bool D3D9Rtx::ngxShaderIsFinalBackbufferComposite(const Ue3ShaderFeatureInfo& psInfo) {
    return ngxShaderIsFinishRenderViewTargetGamma(psInfo) ||
           psInfo.hasToneMapConstants ||
           psInfo.hasExposureOrToneSampler ||
           psInfo.looksLikeDofAndBloomPostProcess();
  }

  bool D3D9Rtx::ngxShaderHasPostProcessNames(const Ue3ShaderFeatureInfo& psInfo) {
    return psInfo.hasToneMapConstants ||
           psInfo.hasGammaConstants ||
           psInfo.hasExposureOrToneSampler ||
           psInfo.hasBlurredImageSampler ||
           psInfo.hasFilterTextureSampler ||
           psInfo.hasFilterSampleWeights ||
           psInfo.hasDofPackedParameters ||
           psInfo.hasDofMinMaxBlurClamp ||
           psInfo.hasBloomScale ||
           psInfo.hasHazeConstants ||
           psInfo.hasDistortionSampler ||
           psInfo.hasFogConstants ||
           psInfo.hasVelocitySampler ||
           psInfo.hasMotionBlurConstants ||
           psInfo.hasSceneDepthSampler ||
           psInfo.hasLightAttenuationSampler ||
           psInfo.hasShadowSampler ||
           psInfo.hasUiSampler ||
           psInfo.hasUiCompositeConstants;
  }

  void D3D9Rtx::refreshNgxUe3VertexFactory() {
    m_currentUe3VertexFactory = Ue3VertexFactoryType::Unknown;
    if (!m_frameOptions.ue3EngineMode || d3d9State().vertexDecl == nullptr)
      return;

    const auto& elements = d3d9State().vertexDecl->GetElements();
    const XXH64_hash_t declKey = XXH3_64bits(elements.data(), elements.size() * sizeof(D3DVERTEXELEMENT9));
    const auto it = m_ue3VertexFactoryCache.find(declKey);
    if (it != m_ue3VertexFactoryCache.end()) {
      m_currentUe3VertexFactory = it->second;
    } else {
      m_currentUe3VertexFactory = classifyUe3VertexFactory(elements);
      m_ue3VertexFactoryCache.emplace(declKey, m_currentUe3VertexFactory);
    }
  }

  bool D3D9Rtx::ngxDrawIsUi(const DrawContext& drawContext) {
    if (drawContext.PrimitiveCount == 0)
      return false;

    m_boundTextureSnapshotValid = false;
    refreshNgxUe3VertexFactory();

    return classifyUe3Pass(drawContext) == Ue3PassType::UiComposite ||
           isRenderingUI() ||
           (m_frameOptions.preTransformedVerticesIsUI &&
            d3d9State().vertexDecl != nullptr &&
            d3d9State().vertexDecl->TestFlag(D3D9VertexDeclFlag::HasPositionT));
  }

  bool D3D9Rtx::ngxImageIsDofFilterBuffer(const DxvkImage* image) const {
    if (image == nullptr || m_ngxSceneColorImage == nullptr || image == m_ngxSceneColorImage.ptr())
      return false;

    const VkExtent3D& filter = image->info().extent;
    const VkExtent3D& scene = m_ngxSceneColorImage->info().extent;
    if (filter.width < 8 || filter.height < 8 || scene.width < 16 || scene.height < 16)
      return false;

    // The two-pixel border keeps same-ratio downsample targets without one out of this match.
    for (const uint32_t factor : { 2u, 4u, 8u }) {
      const auto matches = [factor](uint32_t filterDim, uint32_t sceneDim) {
        const uint32_t base = sceneDim / factor;
        return filterDim >= base + 1u && filterDim <= base + 3u;
      };
      if (matches(filter.width, scene.width) && matches(filter.height, scene.height))
        return true;
    }
    return false;
  }

  bool D3D9Rtx::ngxBoundImageIsDofFilterBuffer() const {
    if (d3d9State().renderTargets[kRenderTargetIndex] == nullptr)
      return false;

    D3D9CommonTexture* renderTarget = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
    if (renderTarget == nullptr || renderTarget->GetImage() == nullptr)
      return false;

    return ngxImageIsDofFilterBuffer(renderTarget->GetImage().ptr());
  }

  bool D3D9Rtx::tryGetBoundPixelFeatures(Ue3ShaderFeatureInfo& info) {
    if (!m_parent->UseProgrammablePS() || d3d9State().pixelShader.ptr() == nullptr)
      return false;

    const D3D9CommonShader* shader = d3d9State().pixelShader->GetCommonShader();
    if (shader == nullptr)
      return false;

    info = getUe3ShaderFeatureInfo(shader);
    return true;
  }

  bool D3D9Rtx::ngxBoundPixelIsDofGather() {
    Ue3ShaderFeatureInfo info;
    if (!tryGetBoundPixelFeatures(info))
      return false;

    return info.hasSceneColorSampler &&
           !info.hasBlurredImageSampler &&
           !info.hasFilterSampleWeights &&
           !info.hasToneMapConstants &&
           (info.hasBloomScale || (info.hasDofPackedParameters && info.hasDofMinMaxBlurClamp));
  }

  bool D3D9Rtx::ngxBoundPixelIsDofBlend() {
    Ue3ShaderFeatureInfo info;
    if (!tryGetBoundPixelFeatures(info))
      return false;

    return info.hasBlurredImageSampler &&
           (info.hasDofPackedParameters || info.hasSceneColorSampler);
  }

  bool D3D9Rtx::ngxOffscreenPrePostShape(const DrawContext& drawContext) {
    m_currentUe3PassType = Ue3PassType::Unknown;

    if (!m_activePresentParams.has_value())
      return false;

    const uint32_t backBufferWidth = m_activePresentParams->BackBufferWidth;
    const uint32_t backBufferHeight = m_activePresentParams->BackBufferHeight;
    const D3DVIEWPORT9& vp = d3d9State().viewport;
    const bool depthTestDisabled = d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_FALSE ||
                                   d3d9State().renderStates[D3DRS_ZFUNC] == D3DCMP_ALWAYS;

    if (drawContext.PrimitiveCount > 4 || !depthTestDisabled ||
        d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE ||
        d3d9State().renderStates[D3DRS_STENCILENABLE] != FALSE ||
        backBufferWidth == 0 || backBufferHeight == 0) {
      return false;
    }

    const bool coversBackbuffer = vp.Width + 1 >= backBufferWidth && vp.Height + 1 >= backBufferHeight;

    // ScreenPercentage and post-process volumes cover the scene viewport.
    const bool sceneViewportKnown =
      m_ngxSceneViewportValid ||
      (m_ngxSceneViewportLastValidFrame != 0 &&
       m_ue3FrameCounter - m_ngxSceneViewportLastValidFrame <= kNgxSceneViewportStaleFrameLimit);
    const bool coversScene =
      sceneViewportKnown &&
      m_ngxSceneViewport.Width != 0 && m_ngxSceneViewport.Height != 0 &&
      vp.Width + 1 >= m_ngxSceneViewport.Width &&
      vp.Height + 1 >= m_ngxSceneViewport.Height &&
      vp.Width <= m_ngxSceneViewport.Width + 8 &&
      vp.Height <= m_ngxSceneViewport.Height + 8;

    // The DOF gather viewport is the filter buffer.
    // It stays out of the scene-colour read count.
    // Including it aims the next frame at a pass that runs before bloom.
    if (!coversBackbuffer && !coversScene)
      return false;

    const bool noBlend = d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] == FALSE;

    // An unblended backbuffer quad qualifies on its shape alone.
    if (noBlend && coversBackbuffer)
      return true;

    refreshNgxUe3VertexFactory();
    m_currentUe3PassType = classifyUe3Pass(drawContext);
    return m_currentUe3PassType == Ue3PassType::FullscreenPostProcess ||
           m_currentUe3PassType == Ue3PassType::FogOrDistortion;
  }

  void D3D9Rtx::countNgxPostInjectionSceneColorConsumer(const DrawContext& drawContext) {
    // The injection ordinal is only useful if the count it comes from covers the whole frame, and
    // the passes that decide whether this frame's injection was the last one are precisely the ones
    // that come after it. Same shape test as the pre-injection path, deliberately: a mismatch
    // between the two would bias the count and walk the injection point away from the end.
    if (!ngxPrePostInjectionAllowed() || m_ngxSceneColorImage == nullptr ||
        m_ngxBackbufferDrawSeenThisFrame ||
        !m_activePresentParams.has_value() || d3d9State().renderTargets[kRenderTargetIndex] == nullptr) {
      return;
    }

    // Same shape as the pre-injection offscreen candidate. A mismatch makes the frame's read
    // total depend on where the injection landed, and since that total aims the next frame, the
    // aim then feeds back into itself and never settles.
    if (!ngxOffscreenPrePostShape(drawContext))
      return;

    D3D9CommonTexture* renderTargetTexture = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
    if (renderTargetTexture != nullptr &&
        (renderTargetTexture->GetImage().ptr() == m_ngxSceneColorImage.ptr() ||
         (m_ngxFrameBackbufferImage != nullptr &&
          renderTargetTexture->GetImage().ptr() == m_ngxFrameBackbufferImage.ptr()))) {
      return;
    }

    const uint32_t rtSamplerMask = m_parent->GetActiveRTTextures() & m_parent->m_psShaderMasks.samplerMask;

    for (const uint32_t i : bit::BitMask(rtSamplerMask)) {
      D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[i]);
      if (texture == nullptr || texture->GetImage() == nullptr) {
        continue;
      }

      const DxvkImage* sampledImage = texture->GetImage().ptr();
      if (matchNgxSceneColorSample(sampledImage) != nullptr) {
        m_ngxPrePostCandidatesThisFrame++;

        if (m_ngxFrameReadCount < kNgxFrameShapeSlots) {
          m_ngxFrameReadDraws[m_ngxFrameReadCount++] = m_drawCallID;
        }
        break;
      }
    }
  }

  PrepareDrawFlags D3D9Rtx::prepareDrawForNgxPassthrough(const DrawContext& drawContext) {
    ScopedCpuProfileZone();

    // Per-frame draw index for the diagnostics (nothing else advances it in this mode)
    m_drawCallID++;

    if (unlikely(m_ngxPostChainDumpFramesLeft > 0)) {
      dumpNgxPostChainDraw(drawContext);
    }

    // Super Resolution texture LOD bias transitions: the bias is folded into the sampler
    // keys at bind time (see GetNgxPassthroughSamplerLodBias), but samplers are only
    // (re)created when their stage is dirtied - so whenever this draw's effective bias
    // differs from the last draw's (scene <-> UI target switches, the injection trigger),
    // every stage must re-bind or draws would sample with the previous scope's bias.
    {
      const float samplerLodBias = GetNgxPassthroughSamplerLodBias();
      if (samplerLodBias != m_ngxAppliedSamplerLodBias) {
        m_ngxAppliedSamplerLodBias = samplerLodBias;
        m_parent->m_dirtySamplerStates = (1u << uint32_t(d3d9State().samplerStates.size())) - 1u;
      }
    }

    // The same hazard applies to the viewport jitter. Whether a draw is jittered depends on its
    // render target and depth-stencil (see GetNgxPassthroughViewportJitter), but the Vulkan
    // viewport is only rebuilt when D3D9 marks the viewport dirty, which binding a different
    // target need not do - so a draw can inherit the previous draw's jitter state and land on a
    // different sub-pixel grid to the rest of the frame. Force the rebind whenever it flips.
    {
      float jitterX = 0.0f;
      float jitterY = 0.0f;
      const bool jitterThisDraw = GetNgxPassthroughViewportJitter(&jitterX, &jitterY);

      if (jitterThisDraw != m_ngxViewportJitterApplied ||
          (jitterThisDraw && (jitterX != m_ngxAppliedViewportJitter[0] ||
                              jitterY != m_ngxAppliedViewportJitter[1]))) {
        m_ngxViewportJitterApplied = jitterThisDraw;
        m_ngxAppliedViewportJitter[0] = jitterX;
        m_ngxAppliedViewportJitter[1] = jitterY;
        m_parent->m_flags.set(D3D9DeviceFlag::DirtyViewportScissor);
      }
    }

    // Where the scene ends. Deliberately independent of camera reconstruction: the last depth
    // priority group is where the first person mesh and held weapon are, they are skinned, and
    // skinned draws are exactly the ones whose constants the camera extraction rejects - reading
    // this from that path would make the frame look as if its last group never drew anything.
    if (drawContext.PrimitiveCount > 4 &&
        d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE &&
        d3d9State().depthStencil != nullptr) {
      m_ngxFrameLastGeometryDraw = m_drawCallID;
    }

    // Once the game has bound the backbuffer it is finished with the scene and its post chain, and
    // what follows is the UI being composited. Tracked across the whole frame, including past the
    // injection, because the shape that positions the next frame spans the whole frame.
    if (!m_ngxBackbufferDrawSeenThisFrame && m_ngxFrameBackbufferImage != nullptr &&
        d3d9State().renderTargets[kRenderTargetIndex] != nullptr) {
      D3D9CommonTexture* drawTarget = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();

      if (drawTarget != nullptr && drawTarget->GetImage().ptr() == m_ngxFrameBackbufferImage.ptr()) {
        m_ngxBackbufferDrawSeenThisFrame = true;
        m_ngxFrameFirstBackbufferDraw = m_drawCallID;
      }
    }

    // Records the scene blit on every backbuffer draw, including the injection trigger.
    // maybeFinishNgxScene does not run until the following draw.
    if (!m_ngxHudlessSceneCompositeSeen &&
        m_ngxFrameBackbufferImage != nullptr &&
        d3d9State().renderTargets[kRenderTargetIndex] != nullptr) {
      D3D9CommonTexture* hudlessTarget = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();
      if (hudlessTarget != nullptr &&
          hudlessTarget->GetImage().ptr() == m_ngxFrameBackbufferImage.ptr() &&
          ngxDrawIsBackbufferSceneComposite(drawContext)) {
        m_ngxHudlessSceneCompositeSeen = true;
      }
    }

    // Every draw executes as plain rasterization in this mode. The remaining per-draw work:
    // UE3 camera extraction, scene target identification, and the thin injection trigger.
    if (m_rtxInjectTriggered) {
      if (m_ngxPrePostUpscaled && (ngxDrawIsUi(drawContext) || ngxDrawFollowsSceneComposite(drawContext))) {
        m_ngxPrePostUpscaled = false;
        m_ngxSpsbReplaceFull = false;
        m_ngxSpsbPatchActive = false;
        m_ngxSpsbLastVsReg = -1;
        m_ngxSpsbLastPsReg = -1;
        m_parent->m_consts[DxsoProgramTypes::VertexShader].dirty = true;
        m_parent->m_consts[DxsoProgramTypes::PixelShader].dirty = true;
      } else if (m_ngxPrePostUpscaled && ngxDepthTestDisabled()) {
        armNgxFullBufferScreenPositionScaleBias();
        stretchNgxVelocityBuffer();
      } else if (m_ngxPrePostUpscaled) {
        // A depth-tested mesh keeps the bias the game uploaded.
        // That bias addresses the screen-percentage viewport the mesh is projected into.
        // A full-buffer bias scales a distortion offset against the wrong rect.
        if (m_ngxSpsbReplaceFull || m_ngxSpsbPatchActive) {
          m_parent->m_consts[DxsoProgramTypes::VertexShader].dirty = true;
          m_parent->m_consts[DxsoProgramTypes::PixelShader].dirty = true;
        }
        m_ngxSpsbReplaceFull = false;
        m_ngxSpsbPatchActive = false;
        m_ngxSpsbLastVsReg = -1;
        m_ngxSpsbLastPsReg = -1;
      }

      checkNgxPostInjectionDraw(drawContext);
      noteNgxPostUpscaleViewport();
      countNgxPostInjectionSceneColorConsumer(drawContext);
      maybeFinishNgxScene(drawContext);
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    m_boundTextureSnapshotValid = false;

    // Decide this frame's sub-pixel jitter before the first draw that may consume it
    if (!m_ngxFrameJitterValid) {
      if (unlikely(!m_frameOptions.valid)) {
        refreshFrameOptionCache();
      }

      // Drive ScreenPercentage before scene draws so the game can apply the reduced render
      // resolution this frame (EndFrame alone is one frame late for the first scene pass).
      applyNgxPassthroughScreenPercentage();

      m_ngxFrameJitterValid = true;
      m_ngxFrameJitter[0] = 0.0f;
      m_ngxFrameJitter[1] = 0.0f;

      // Temporal upscaler selected and usable on this system
      const bool temporalUpscalerUsable = RtxNgxPassthrough::needsViewportJitter(m_parent->GetDXVKDevice().ptr());

      if (m_frameOptions.ngxPassthroughJitter && temporalUpscalerUsable) {
        // The render height the game is about to draw at, so the phase count matches the
        // upscale ratio DLSS will see this frame.
        uint32_t renderHeight = 0;
        uint32_t displayHeight = 0;
        if (m_activePresentParams.has_value()) {
          displayHeight = m_activePresentParams->BackBufferHeight;
          if (m_ngxGameScreenPercentage > 0.0f && m_ngxGameScreenPercentage <= 100.0f) {
            renderHeight = uint32_t(float(displayHeight) * m_ngxGameScreenPercentage / 100.0f);
          } else {
            renderHeight = displayHeight;
          }
        }

        const uint32_t jitterSequenceLength = RtxNgxPassthrough::viewportJitterSequenceLength(
          m_parent->GetDXVKDevice().ptr(), renderHeight, displayHeight);

        // An artefact with the period of this loop is the signature of a phase that recurs too
        // rarely to accumulate, so the length has to be readable rather than re-derived by hand
        if (jitterSequenceLength != m_ngxLoggedJitterSequenceLength) {
          m_ngxLoggedJitterSequenceLength = jitterSequenceLength;
          Logger::info(str::format("[RTX NGX Passthrough] Viewport jitter loop: ", jitterSequenceLength,
                                   " phases at ", renderHeight, "p render / ", displayHeight, "p display."));
        }
        const Vector2 jitter = calculateHaltonJitter(m_parent->GetDXVKDevice()->getCurrentFrameId(),
                                                     jitterSequenceLength);
        m_ngxFrameJitter[0] = jitter.x;
        m_ngxFrameJitter[1] = jitter.y;
      }

      // The viewport carries the previous frame's jitter offset until rebound
      m_parent->m_flags.set(D3D9DeviceFlag::DirtyViewportScissor);

      // The GPU constant buffers may carry the previous frame's ScreenPositionScaleBias
      // patch (a different jitter): force one fresh upload per stage this frame
      m_parent->m_consts[DxsoProgramTypes::VertexShader].dirty = true;
      m_parent->m_consts[DxsoProgramTypes::PixelShader].dirty = true;
      m_ngxSpsbLastVsReg = -1;
      m_ngxSpsbLastPsReg = -1;

      // Resolve this frame's backbuffer image (backbuffers rotate every present) for the
      // scene-end trigger: description-based primary checks also match offscreen buffers
      // allocated at backbuffer size (e.g. ME's TdUI compositing targets), which would make
      // the injection run against - and DLSS write into - the wrong image on some frames.
      m_ngxFrameBackbufferImage = nullptr;

      Com<IDirect3DSurface9> backBuffer;
      if (SUCCEEDED(m_parent->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &backBuffer)) && backBuffer != nullptr) {
        if (D3D9CommonTexture* backBufferTexture = static_cast<D3D9Surface*>(backBuffer.ptr())->GetCommonTexture()) {
          m_ngxFrameBackbufferImage = backBufferTexture->GetImage();
        }
      }
    }

    // ScreenPositionScaleBias jitter compensation for this draw's shaders: screen space
    // lookups (shadow projections reading scene depth from the scene color alpha,
    // translucency, distortion, fog) compute UVs from clip-space varyings, which do not
    // follow the viewport jitter; the bias constant is shifted by the jitter at constant
    // upload so those lookups sample the jittered content aligned. Without this, passes
    // whose result depends sharply on the sampled depth (dynamic shadow projections with
    // tight bias) oscillate with the per-frame jitter sign.
    //
    // Decided before any early-out below: the patch must always describe the shaders bound
    // for whatever the device uploads next, including degenerate draws that skip the rest
    // of the per-draw work.
    m_ngxSpsbPatchActive = false;
    m_ngxSpsbReplaceFull = false;

    if ((m_ngxFrameJitter[0] != 0.0f || m_ngxFrameJitter[1] != 0.0f) && m_ngxSceneColorImage != nullptr) {
      m_ngxSpsbPatchVs = lookupNgxSpsbCtabReg(m_parent->UseProgrammableVS() && d3d9State().vertexShader.ptr() != nullptr
                                                ? d3d9State().vertexShader->GetCommonShader() : nullptr);
      m_ngxSpsbPatchPs = lookupNgxSpsbCtabReg(d3d9State().pixelShader.ptr() != nullptr
                                                ? d3d9State().pixelShader->GetCommonShader() : nullptr);

      if (m_ngxSpsbPatchVs.present || m_ngxSpsbPatchPs.present) {
        const VkExtent3D& sceneExtent = m_ngxSceneColorImage->info().extent;

        if (sceneExtent.width != 0 && sceneExtent.height != 0) {
          m_ngxSpsbPatchAdd[0] = m_ngxFrameJitter[0] / float(sceneExtent.width);
          m_ngxSpsbPatchAdd[1] = m_ngxFrameJitter[1] / float(sceneExtent.height);
          m_ngxSpsbPatchActive = true;
        }
      }
    }

    // Force a fresh constant upload when the patch register moved (shader switch without a
    // constant change): the buffered copy carries the previous shader's patch layout
    {
      const int32_t vsReg = (m_ngxSpsbPatchActive && m_ngxSpsbPatchVs.present) ? int32_t(m_ngxSpsbPatchVs.reg) : -1;
      const int32_t psReg = (m_ngxSpsbPatchActive && m_ngxSpsbPatchPs.present) ? int32_t(m_ngxSpsbPatchPs.reg) : -1;

      if (vsReg != m_ngxSpsbLastVsReg) {
        m_parent->m_consts[DxsoProgramTypes::VertexShader].dirty = true;
        m_ngxSpsbLastVsReg = vsReg;
      }
      if (psReg != m_ngxSpsbLastPsReg) {
        m_parent->m_consts[DxsoProgramTypes::PixelShader].dirty = true;
        m_ngxSpsbLastPsReg = psReg;
      }
    }

    if (drawContext.PrimitiveCount == 0 || d3d9State().renderTargets[kRenderTargetIndex] == nullptr) {
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    // UE3 vertex factory classification (cached by vertex declaration) feeds classifyUe3Pass
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

    // Camera extraction from UE3 reserved shader constants (CTAB-verified draws only)
    if (m_parent->UseProgrammableVS() && d3d9State().vertexShader.ptr() != nullptr) {
      tryNgxPassthroughCameraCapture(drawContext.PrimitiveCount);
    }

    // Dynamic object capture for the velocity raster pass
    tryCaptureNgxVelocityDraw(drawContext);

    learnNgxDrawResolve(drawContext);
    noteNgxUberEncode(drawContext);

    // Scene end detection. Pre-post injection runs DLSS on linear scene colour before the post chain.
    // A lower preset upscales its subrect at that same point.
    // If no pre-post candidate fires, late injection is the first UI draw on the primary target.
    // Late injection can also be the engine's final stretch.
    bool triggerInjection = false;
    bool suppressDraw = false;

    if (m_activePresentParams.has_value()) {
      const uint32_t backBufferWidth = m_activePresentParams->BackBufferWidth;
      const uint32_t backBufferHeight = m_activePresentParams->BackBufferHeight;

      D3D9CommonTexture* renderTargetTexture = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture();

      // First composite that samples scene colour and writes a different target.
      // That starts the post chain, after the world and foreground depth-priority groups.
      // The sampled image may be the scene-colour surface or the resolve UE3 creates for sampling.
      // TdSunHaze is completed after its own draw.
      // Every other candidate injects before the draw.
      // That draw then samples the upscaler output.
      // A reduced scene rect is upscaled here at any DLSS preset.
      // The late stretch below runs only when no candidate fires.
      const bool likelyPostProcessQuad = ngxOffscreenPrePostShape(drawContext);

      // The view rect comes from the frame's scene draws, and a frame whose geometry all failed
      // camera reconstruction donates none. Losing the injection over a rectangle that had not
      // changed sends the frame to the late point, so a recently donated rect stays usable.
      const bool sceneViewportUsable =
        m_ngxSceneViewportValid ||
        (m_ngxSceneViewportLastValidFrame != 0 &&
         m_ue3FrameCounter - m_ngxSceneViewportLastValidFrame <= kNgxSceneViewportStaleFrameLimit);

      // Which requirements the frame met, so one that ends up at the late point can say why.
      // Fog and other scene-colour quads are still inside the scene, ahead of the distortion apply.
      // Upscaling on them leaves the offset field and its stencil in the screen-percentage rect.
      const bool upscaleBoundary = likelyPostProcessQuad && ngxBoundPixelIsUpscaleBoundary();

      m_ngxDiagSawPostQuad |= upscaleBoundary;
      m_ngxDiagSceneColorReady |= (m_ngxSceneColorImage != nullptr && sceneViewportUsable);

      // The DOF/bloom gather is the first pass of stock UE3's DOF, bloom and uber effects.
      // It writes the filter buffer and stays out of the read ordinal.
      // A gather before the foreground clear belongs to an effect in the world group.
      const bool gatherIsUncounted =
        !triggerInjection &&
        drawContext.PrimitiveCount > 0 &&
        drawContext.PrimitiveCount <= 4 &&
        sceneViewportUsable &&
        m_ngxSceneViewport.Width >= 8 &&
        m_ngxSceneViewport.Height >= 8 &&
        ngxPastForegroundClear() &&
        ngxBoundPixelIsDofGather();

      // A pass with the backbuffer bound is the final composite or the UI being drawn over it, and
      // the pre-post point is mid-chain by definition. Letting those count walks the injection past
      // the end of the post chain onto a pass with UI in it, which the upscaler then resolves and
      // softens; that territory belongs to the late injection point.
      const bool targetsBackbuffer =
        renderTargetTexture != nullptr && m_ngxFrameBackbufferImage != nullptr &&
        renderTargetTexture->GetImage().ptr() == m_ngxFrameBackbufferImage.ptr();

      const bool candidateShape =
        !triggerInjection &&
        ngxPrePostInjectionAllowed() &&
        !ngxEarlyInjectionSuspended() &&
        (upscaleBoundary || gatherIsUncounted) &&
        !targetsBackbuffer && !m_ngxBackbufferDrawSeenThisFrame &&
        (renderTargetTexture == nullptr || renderTargetTexture->GetImage().ptr() != m_ngxSceneColorImage.ptr());

      if (candidateShape) {
        // The gather is not recorded, so the ordinal stays with the reads that follow bloom.
        const Rc<DxvkImage> matchedTarget = gatherIsUncounted
          ? findNgxSceneColorSource()
          : findNgxPrePostSceneColorFromSamplers(true, true);

        if (matchedTarget != nullptr) {
          // TdDirHaze's glow is a nonlinear function of the depth in scene-colour alpha.
          // That depth matches the colour only on the jittered raster.
          // The draw composites first.
          // DLSS then runs on the image it wrote, still before bloom and tonemap.
          const bool sunHaze =
            m_parent->UseProgrammablePS() && d3d9State().pixelShader.ptr() != nullptr &&
            getUe3ShaderFeatureInfo(d3d9State().pixelShader->GetCommonShader()).hasHazeConstants;

          if (sunHaze && renderTargetTexture != nullptr && renderTargetTexture->GetImage() != nullptr) {
            m_ngxDeferPrePostInjection = true;
            m_ngxDeferredPrePostTarget = renderTargetTexture->GetImage();
            m_ngxDeferredPrimitiveType = drawContext.PrimitiveType;
            ONCE(Logger::info("[RTX NGX Passthrough] TdSunHaze composites on the jittered scene. DLSS runs on its output, before the rest of the post chain."));
          } else {
            if (gatherIsUncounted) {
              ONCE(Logger::info("[RTX NGX Passthrough] The DOF/bloom gather is the pre-post point. Bloom and depth of field read the upscaler output."));
            }

            engageNgxPrePostInjection(matchedTarget);
            m_ngxFrameInjectionSite = gatherIsUncounted ? NgxInjectionSite::Gather : NgxInjectionSite::AimedRead;
            triggerInjection = true;

            ONCE(Logger::info(str::format("[RTX NGX Passthrough] Pre-post-process injection engaged: DLSS runs on the ",
                                          (m_ngxColorMirrorImage != nullptr ? "resolved scene color" : "scene color"),
                                          " before the game's post-process chain.")));
          }
        }
      }

      // The trigger requires the actual backbuffer image, not just a backbuffer-sized
      // description: offscreen composition buffers (e.g. ME's TdUI targets) share the
      // description and would otherwise receive the injection on some frames, making DLSS
      // alternate between the real backbuffer and an offscreen image
      bool isPrimary;

      if (m_ngxFrameBackbufferImage != nullptr) {
        isPrimary = renderTargetTexture != nullptr &&
                    renderTargetTexture->GetImage().ptr() == m_ngxFrameBackbufferImage.ptr();
      } else {
        isPrimary = s_isDxvkResolutionEnvVarSet ||
          (m_activePresentParams->BackBufferWidth == renderTargetTexture->Desc()->Width &&
           m_activePresentParams->BackBufferHeight == renderTargetTexture->Desc()->Height);
      }

      // A pre-post color target already holds the upscaled scene.
      // The gamma draw stays, and applies its curve once.
      if (!triggerInjection && !m_ngxDeferPrePostInjection && isPrimary &&
          m_ngxColorTargetImage == nullptr) {
        // ScreenPercentage upscale replacement: only engages when this frame's scene provably
        // rendered into a subrect meaningfully smaller than the backbuffer in both dimensions
        // (a reduced height alone would match letterboxed cinematics)
        const bool sceneIsSubrect =
          m_ngxSceneViewportValid &&
          ngxSceneViewportIsSubrect(m_ngxSceneViewport, backBufferWidth, backBufferHeight);

        // The composite has to execute before DLSS can read what it wrote, so the draw that
        // produced it must never be the one that triggers the injection.
        bool recordedRuntimeComposite = false;

        if (sceneIsSubrect && ngxRuntimeOwnsUpscale()) {
          // The engine believes it renders natively, so there is no upscale stretch to replace:
          // its post chain composites the reduced view rect straight into the backbuffer (at the
          // centred position ScaleScreenCoords produced) and stops. Note where that landed; the
          // upscale happens once the scene is finished, in emitNgxPassthroughFrameData.
          //
          // The composite's own viewport is not a usable signature for finding it. UE3 may leave
          // the viewport at full size and size the quad instead, so requiring the viewport to
          // match the scene rect misses the draw entirely in those titles, and the reduced image
          // is then presented unstretched with borders around it. Either shape is accepted.
          const D3DVIEWPORT9& vp = d3d9State().viewport;
          const bool depthTestDisabled = d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_FALSE ||
                                         d3d9State().renderStates[D3DRS_ZFUNC] == D3DCMP_ALWAYS;
          const bool compositeViewportIsSceneSized =
            vp.Width == m_ngxSceneViewport.Width && vp.Height == m_ngxSceneViewport.Height;
          const bool viewportIsSceneOrFull =
            compositeViewportIsSceneSized ||
            (vp.Width + 1 >= backBufferWidth && vp.Height + 1 >= backBufferHeight);

          const bool likelySubrectComposite =
            drawContext.PrimitiveCount <= 4 &&
            depthTestDisabled &&
            d3d9State().renderStates[D3DRS_ZWRITEENABLE] == FALSE &&
            viewportIsSceneOrFull &&
            (m_parent->GetActiveRTTextures() & m_parent->m_psShaderMasks.samplerMask) != 0;

          if (likelySubrectComposite) {
            // Where the scene landed in the backbuffer, which is not where it landed in scene
            // color. A composite that sets a reduced viewport states its position directly; one
            // that leaves the viewport full size does not, and the scene viewport's own offset
            // cannot stand in for it - that is a scene color space position, and a title may
            // render at its origin while compositing centred. UE3's ScaleScreenCoords centres
            // the reduced rect in the target, so the remaining case is half the difference.
            m_ngxRuntimeUpscaleRect = m_ngxSceneViewport;

            if (compositeViewportIsSceneSized) {
              m_ngxRuntimeUpscaleRect.X = vp.X;
              m_ngxRuntimeUpscaleRect.Y = vp.Y;
            } else {
              m_ngxRuntimeUpscaleRect.X = backBufferWidth > m_ngxSceneViewport.Width
                ? (backBufferWidth - m_ngxSceneViewport.Width) / 2 : 0;
              m_ngxRuntimeUpscaleRect.Y = backBufferHeight > m_ngxSceneViewport.Height
                ? (backBufferHeight - m_ngxSceneViewport.Height) / 2 : 0;
            }

            m_ngxRuntimeUpscaleRectValid = true;
            recordedRuntimeComposite = true;

            ONCE(Logger::info(str::format(
              "[RTX NGX Passthrough] Runtime-owned Super Resolution: the engine composited its ",
              m_ngxRuntimeUpscaleRect.Width, "x", m_ngxRuntimeUpscaleRect.Height, " view rect at (",
              m_ngxRuntimeUpscaleRect.X, ", ", m_ngxRuntimeUpscaleRect.Y,
              ") in the backbuffer, ", (compositeViewportIsSceneSized ? "stated by its viewport" : "centred by ScaleScreenCoords"),
              "; DLSS will upscale it to ", backBufferWidth, "x", backBufferHeight, ".")));
          }
        } else if (sceneIsSubrect) {
          const D3DVIEWPORT9& vp = d3d9State().viewport;
          const bool depthTestDisabled = d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_FALSE ||
                                         d3d9State().renderStates[D3DRS_ZFUNC] == D3DCMP_ALWAYS;
          const bool zWriteEnabled = d3d9State().renderStates[D3DRS_ZWRITEENABLE] != FALSE;

          // The stretch is the composite onto the primary target after the scene.
          // NeedsUpscale() uses a fullscreen viewport.
          // Without that flag the same quad keeps the reduced view rect, which is the black border.
          const bool viewportCoversBackbuffer =
            vp.Width + 1 >= backBufferWidth && vp.Height + 1 >= backBufferHeight;
          const bool viewportIsScene =
            m_ngxSceneViewport.Width != 0 && m_ngxSceneViewport.Height != 0 &&
            vp.Width + 1 >= m_ngxSceneViewport.Width &&
            vp.Height + 1 >= m_ngxSceneViewport.Height &&
            vp.Width <= m_ngxSceneViewport.Width + 8 &&
            vp.Height <= m_ngxSceneViewport.Height + 8;
          const bool likelyUpscaleStretch =
            drawContext.PrimitiveCount <= 4 &&
            depthTestDisabled && !zWriteEnabled &&
            (viewportCoversBackbuffer || viewportIsScene);

          // Only a plain composite may be stood in for. Some UE3 licensee builds fold the whole
          // post chain into the upscaling draw - depth of field, bloom, the material grade and
          // atmospheric fog on the way out; replacing that discards all of it, so the runtime
          // takes ownership of the upscale instead and lets the engine's own pass composite
          // into the reduced rect untouched.
          bool compositeSafeToReplace = true;
          if (likelyUpscaleStretch && m_parent->UseProgrammablePS() &&
              d3d9State().pixelShader.ptr() != nullptr) {
            const Ue3ShaderFeatureInfo psInfo =
              getUe3ShaderFeatureInfo(d3d9State().pixelShader->GetCommonShader());
            compositeSafeToReplace =
              psInfo.gammaInverseReg != Ue3ShaderFeatureInfo::kNgxNoRegister ||
              (!psInfo.hasToneMapConstants && !psInfo.hasExposureOrToneSampler);
          }

          // With pre-post on, the draw still runs over the upscaled image.
          // A plain gamma pass with pre-post off is replaced.
          // An uber pass with pre-post off draws first, and DLSS follows it before the HUD.
          if (likelyUpscaleStretch) {
            const Rc<DxvkImage> sourceImage = findNgxSceneColorSource();

            if (sourceImage != nullptr && ngxPrePostInjectionAllowed()) {
              // DLSS writes the full scene-color image. This draw still runs.
              // Retarget opens the quad onto that image, including when the viewport was left at the view rect.
              engageNgxPrePostInjection(sourceImage);
              m_ngxFrameInjectionSite = NgxInjectionSite::Stretch;
              triggerInjection = true;

              ONCE(Logger::info(str::format(
                compositeSafeToReplace
                  ? "[RTX NGX Passthrough] ScreenPercentage gamma pass is the pre-post upscale ("
                  : "[RTX NGX Passthrough] Uber post pass is the pre-post upscale (",
                m_ngxSceneViewport.Width, "x", m_ngxSceneViewport.Height, " -> ",
                backBufferWidth, "x", backBufferHeight,
                compositeSafeToReplace
                  ? "). The gamma draw still runs."
                  : "). Depth of field, bloom, and gamma still run.")));
            } else if (sourceImage != nullptr && compositeSafeToReplace) {
              // Pre-post is off.
              // This gamma stretch is the late upscale.
              m_ngxUpscaleSourceImage = sourceImage;
              m_ngxSubrect.offset = { int32_t(m_ngxSceneViewport.X), int32_t(m_ngxSceneViewport.Y) };
              m_ngxSubrect.extent = { m_ngxSceneViewport.Width, m_ngxSceneViewport.Height };
              m_ngxColorSubrectOffset = m_ngxSubrect.offset;
              captureNgxOutputTransform();

              // UE3's copy from LDR scene colour uses InverseGamma 1, so the uber pass did the encoding.
              if (!m_ngxOutputTransform.enabled && ngxImageIsUberEncoded(sourceImage.ptr())) {
                m_ngxOutputTransform = ngxLinearizingTransform(m_ngxUberGammaColorScaleAndInverse);
                if (m_ngxOutputTransform.enabled) {
                  ONCE(Logger::info("[RTX NGX Passthrough] The late input was encoded by the uber pass. It is linearized before DLSS and encoded again afterwards."));
                }
              }

              triggerInjection = true;
              suppressDraw = true;

              ONCE(Logger::info(str::format(
                "[RTX NGX Passthrough] ScreenPercentage upscale detected and replaced with DLSS Super Resolution (",
                m_ngxSceneViewport.Width, "x", m_ngxSceneViewport.Height, " -> ",
                backBufferWidth, "x", backBufferHeight, ").")));
            } else if (!compositeSafeToReplace) {
              // Pre-post is off.
              // The uber pass still draws into the centred rect.
              // DLSS runs after that draw and before the HUD.
              m_ngxRuntimeUpscaleRect = m_ngxSceneViewport;
              m_ngxRuntimeUpscaleRect.X = backBufferWidth > m_ngxSceneViewport.Width
                ? (backBufferWidth - m_ngxSceneViewport.Width) / 2 : 0;
              m_ngxRuntimeUpscaleRect.Y = backBufferHeight > m_ngxSceneViewport.Height
                ? (backBufferHeight - m_ngxSceneViewport.Height) / 2 : 0;
              m_ngxDeferLateSubrectUpscale = true;
              m_ngxDeferredPrimitiveType = drawContext.PrimitiveType;
              recordedRuntimeComposite = true;

              // GammaColorScaleAndInverse is already in this image.
              // Those values are removed before DLSS and applied again afterwards.
              m_ngxOutputTransform = NgxOutputTransform();
              Vector4 gamma;
              if (readNgxBoundUberGamma(gamma, false))
                m_ngxOutputTransform = ngxLinearizingTransform(gamma);

              ONCE(Logger::info(
                "[RTX NGX Passthrough] Uber post pass composites at the reduced rect. DLSS upscales it at the late point, before the HUD."));
            }
          }
        }

        if (!triggerInjection && !recordedRuntimeComposite &&
            ngxPrePostInjectionAllowed() &&
            targetsBackbuffer &&
            drawContext.PrimitiveCount <= 4 &&
            (d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_FALSE ||
             d3d9State().renderStates[D3DRS_ZFUNC] == D3DCMP_ALWAYS) &&
            d3d9State().renderStates[D3DRS_ZWRITEENABLE] == FALSE &&
            d3d9State().renderStates[D3DRS_STENCILENABLE] == FALSE &&
            d3d9State().renderStates[D3DRS_ALPHABLENDENABLE] == FALSE &&
            (d3d9State().viewport.Width + 1 >= backBufferWidth &&
             d3d9State().viewport.Height + 1 >= backBufferHeight ||
             (sceneViewportUsable &&
              d3d9State().viewport.Width + 1 >= m_ngxSceneViewport.Width &&
              d3d9State().viewport.Height + 1 >= m_ngxSceneViewport.Height)) &&
            m_parent->UseProgrammablePS() && d3d9State().pixelShader.ptr() != nullptr) {
          const Ue3ShaderFeatureInfo psInfo =
            getUe3ShaderFeatureInfo(d3d9State().pixelShader->GetCommonShader());

          if (ngxShaderIsFinalBackbufferComposite(psInfo)) {
            const Rc<DxvkImage> matchedTarget = findNgxSceneColorSource();

            if (matchedTarget != nullptr) {
              engageNgxPrePostInjection(matchedTarget);
              m_ngxFrameInjectionSite = NgxInjectionSite::FinalComposite;
              triggerInjection = true;

              ONCE(Logger::info(str::format(
                "[RTX NGX Passthrough] Pre-post-process injection engaged at the final backbuffer composite (DLSS runs on the ",
                (m_ngxColorMirrorImage != nullptr ? "resolved scene color" : "scene color"),
                " before that pass).")));
            }
          }
        }

        if (!triggerInjection && !recordedRuntimeComposite) {
          m_currentUe3PassType = classifyUe3Pass(drawContext);

          if (m_currentUe3PassType == Ue3PassType::UiComposite) {
            reportNgxPrePostMiss();
            triggerInjection = true;
          } else if (isRenderingUI()) {
            reportNgxPrePostMiss();
            triggerInjection = true;
          } else if (m_frameOptions.preTransformedVerticesIsUI &&
                     d3d9State().vertexDecl != nullptr &&
                     d3d9State().vertexDecl->TestFlag(D3D9VertexDeclFlag::HasPositionT)) {
            triggerInjection = true;
          }
        }
      }
    }

    if (triggerInjection)
      commitNgxPrePostInjection(drawContext.PrimitiveType);

    return suppressDraw ? PrepareDrawFlag::Ignore : PrepareDrawFlag::PreserveDrawCallAndItsState;
  }

  void D3D9Rtx::commitNgxPrePostInjection(D3DPRIMITIVETYPE primitiveType) {
    // Clears the jitter patch so the next constant upload matches the unjittered output.
    // A pre-post upscale then installs the full-buffer ScreenPositionScaleBias.
    // prepareDraw calls this before the triggering draw is queued.
    // The deferred completion calls it after the haze draw is queued.
    // That draw still samples the jittered scene.
    m_ngxSpsbPatchActive = false;
    m_ngxSpsbReplaceFull = false;
    m_parent->m_consts[DxsoProgramTypes::VertexShader].dirty = true;
    m_parent->m_consts[DxsoProgramTypes::PixelShader].dirty = true;
    m_ngxSpsbLastVsReg = -1;
    m_ngxSpsbLastPsReg = -1;

    if (m_ngxColorTargetImage == nullptr)
      m_ngxFrameInjectionSite = NgxInjectionSite::Late;

    if (m_ngxColorTargetImage != nullptr && m_ngxSceneColorImage != nullptr) {
      const VkExtent3D& extent = m_ngxSceneColorImage->info().extent;
      m_ngxPrePostUpscaled = ngxSceneViewportIsSubrect(m_ngxSceneViewport, extent.width, extent.height);
      m_ngxFrameUpscaledInPlace = m_ngxPrePostUpscaled;
      if (m_ngxPrePostUpscaled) {
        if (!m_ngxLoggedPrePostUpscale) {
          m_ngxLoggedPrePostUpscale = true;
          Logger::info(str::format(
            "[RTX NGX Passthrough] Pre-post Super Resolution: ",
            m_ngxSceneViewport.Width, "x", m_ngxSceneViewport.Height, " -> ",
            extent.width, "x", extent.height, "."));
        }
        armNgxFullBufferScreenPositionScaleBias();
      } else {
        m_ngxLoggedPrePostUpscale = false;
      }
    }

    // Late injection copies the backbuffer in dispatch, before this draw.
    // Pre-post injection copies it on the first draw after the scene blit.
    if (m_ngxColorTargetImage == nullptr) {
      m_ngxHudlessCapturedThisFrame = true;
      if (m_frameOptions.ngxDlfgHudless) {
        noteNgxHudlessCapture(NgxHudlessCaptureSite::LateInjection);
      }
    }

    m_parent->PrepareDraw(primitiveType);

    emitNgxPassthroughFrameData();

    triggerInjectRTX();

    m_rtxInjectTriggered = true;

    // The UI must not inherit the scene's sub-pixel jitter
    m_ngxViewportJitterApplied = false;
    m_ngxAppliedViewportJitter[0] = 0.0f;
    m_ngxAppliedViewportJitter[1] = 0.0f;
    m_parent->m_flags.set(D3D9DeviceFlag::DirtyViewportScissor);
    noteNgxPostUpscaleViewport();
  }

  void D3D9Rtx::FinishDeferredNgxPrePostInjection(bool drawWasIssued) {
    if (m_ngxDeferLateSubrectUpscale) {
      m_ngxDeferLateSubrectUpscale = false;
      if (!drawWasIssued)
        return;

      m_ngxRuntimeUpscaleRectValid = true;
      commitNgxPrePostInjection(m_ngxDeferredPrimitiveType);
      return;
    }

    if (!m_ngxDeferPrePostInjection)
      return;

    m_ngxDeferPrePostInjection = false;
    const Rc<DxvkImage> target = m_ngxDeferredPrePostTarget;
    m_ngxDeferredPrePostTarget = nullptr;

    if (!drawWasIssued || target == nullptr)
      return;

    engageNgxPrePostInjection(target);
    m_ngxFrameInjectionSite = NgxInjectionSite::Haze;
    commitNgxPrePostInjection(m_ngxDeferredPrimitiveType);
  }

  PrepareDrawFlags D3D9Rtx::PrepareDrawGeometryForRT(const bool indexed, const DrawContext& context) {
    (void)indexed;

    if (unlikely(!m_frameOptions.valid)) {
      refreshFrameOptionCache();
    }

    if (m_passthroughSuspended) {
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    return prepareDrawForNgxPassthrough(context);
  }

  PrepareDrawFlags D3D9Rtx::PrepareDrawUPGeometryForRT(const bool indexed,
                                                       const D3D9BufferSlice& buffer,
                                                       const D3DFORMAT indexFormat,
                                                       const uint32_t indexSize,
                                                       const uint32_t indexOffset,
                                                       const uint32_t vertexSize,
                                                       const uint32_t vertexStride,
                                                       const DrawContext& drawContext) {
    (void)indexed;
    (void)buffer;
    (void)indexFormat;
    (void)indexSize;
    (void)indexOffset;
    (void)vertexSize;
    (void)vertexStride;

    if (unlikely(!m_frameOptions.valid)) {
      refreshFrameOptionCache();
    }

    if (m_passthroughSuspended) {
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    return prepareDrawForNgxPassthrough(drawContext);
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

    // Prime settings before the first scene pass.
    if (!m_frameOptions.valid) {
      refreshFrameOptionCache();
    }
    applyNgxPassthroughScreenPercentage();

    // Inform the backend about potential presenter update
    m_parent->EmitCs([cWidth = m_activePresentParams->BackBufferWidth,
                      cHeight = m_activePresentParams->BackBufferHeight](DxvkContext* ctx) {
      static_cast<RtxContext*>(ctx)->resetScreenResolution({ cWidth, cHeight , 1 });
    });
  }





  void D3D9Rtx::EndFrame(const Rc<DxvkImage>& targetImage, bool callInjectRtx, bool countAsFrame) {
    refreshFrameOptionCache();

    // Update the effective settings before the next frame.
    applyNgxPassthroughScreenPercentage();

    const auto currentReflexFrameId = GetReflexFrameId();

    // NGX passthrough mode: no scene end trigger fired this frame (e.g. no UI drawn), so the
    // fallback injection below runs on the backbuffer; hand over this frame's data first
    if (!m_rtxInjectTriggered && callInjectRtx) {
      emitNgxPassthroughFrameData();
    }

    // A pre-post frame that draws nothing after its scene blit finishes the scene here.
    // Without an identified scene blit, the backbuffer may already hold the UI, so DLSS-NR stays off.
    if (m_rtxInjectTriggered && callInjectRtx &&
        m_frameOptions.ngxNeuralRendering && !m_ngxNeuralRenderingAppliedThisFrame &&
        m_ngxColorTargetImage != nullptr && m_ngxFrameBackbufferImage != nullptr) {
      if (m_ngxHudlessSceneCompositeSeen) {
        m_ngxNeuralRenderingAppliedThisFrame = true;
        m_parent->EmitCs([cBackbuffer = m_ngxFrameBackbufferImage](DxvkContext* ctx) {
          static_cast<RtxContext*>(ctx)->applyNgxPassthroughNeuralRendering(cBackbuffer);
        });
      } else {
        ONCE(Logger::info("[RTX NGX Passthrough] DLSS-NR is skipped on a pre-post frame whose scene blit was not identified. "
                          "The backbuffer may already hold the UI."));
      }
    }

    // The presenter already holds this frame's slot.
    // Leaving the slot unfilled repeats the previous frame.
    if (m_rtxInjectTriggered && callInjectRtx &&
        m_frameOptions.ngxDlfgHudless && !m_ngxHudlessCapturedThisFrame &&
        m_ngxFrameBackbufferImage != nullptr) {
      m_ngxHudlessCapturedThisFrame = true;
      noteNgxHudlessCapture(NgxHudlessCaptureSite::FrameEnd);

      m_parent->EmitCs([cBackbuffer = m_ngxFrameBackbufferImage](DxvkContext* ctx) {
        static_cast<RtxContext*>(ctx)->captureNgxPassthroughHudless(cBackbuffer);
      });
    }

    if (m_frameOptions.ngxDlfgHudlessPreview &&
        m_ngxHudlessCapturedThisFrame && m_ngxFrameBackbufferImage != nullptr) {
      m_parent->EmitCs([cBackbuffer = m_ngxFrameBackbufferImage](DxvkContext* ctx) {
        static_cast<RtxContext*>(ctx)->previewNgxPassthroughHudless(cBackbuffer);
      });
    }
    m_parent->Flush();

    // Inform backend of end-frame
    m_parent->EmitCs([currentReflexFrameId, targetImage, callInjectRtx](DxvkContext* ctx) { 
      static_cast<RtxContext*>(ctx)->endFrame(currentReflexFrameId, targetImage, callInjectRtx); 
    });

    // Wait for injectRTX to finish upscaling the backbuffer before the external presenter
    // (or deferred UI replay) touches it on the same thread.
    if (callInjectRtx) {
      m_parent->Flush();
    }

    // The per-draw profile zones in this file are only meaningful against the draw count, which
    // a UE3 title moves by an order of magnitude depending on its occlusion and frustum culling.
    ProfilerPlotValue("D3D9 Draw Calls", int64_t(m_drawCallID));

    // Reset for the next frame
    m_rtxInjectTriggered = false;
    m_drawCallID = 0;
    // Minimized presents still reset the accumulators below, but a frozen counter keeps the
    // object-velocity cache from aging out for the whole time the window is iconic.
    if (countAsFrame) {
      ++m_ue3FrameCounter;
    }

    // NGX passthrough per-frame state
    m_ngxSpsbPatchActive = false;
    m_ngxSpsbReplaceFull = false;
    m_ngxPrePostUpscaled = false;
    m_ngxViewportExpandApplied = false;
    m_ngxReducedSceneViewportThisFrame = false;
    m_ngxReducedSceneDraws.clear();
    m_ngxExpandedViewportWidth = 0;
    m_ngxExpandedViewportHeight = 0;
    m_ngxDeferPrePostInjection = false;
    m_ngxDeferLateSubrectUpscale = false;
    m_ngxVelocityStretchedThisFrame = false;
    m_ngxDeferredPrePostTarget = nullptr;
    m_ngxFrameJitterValid = false;
    // The new frame's jitter differs, so the bound viewport no longer matches anything
    m_ngxViewportJitterApplied = false;
    m_ngxAppliedViewportJitter[0] = 0.0f;
    m_ngxAppliedViewportJitter[1] = 0.0f;
    m_ngxFrameDataEmitted = false;
    m_ngxDepthSnapshotTakenThisFrame = false;
    m_ngxDepthClearsThisFrame = 0;
    m_ngxDiagSawPostQuad = false;
    m_ngxDiagSceneColorReady = false;
    m_ngxBackbufferDrawSeenThisFrame = false;

    logNgxUe3Profile();
    updateNgxInjectionFallbacks();
    m_ngxFrameInjectionSite = NgxInjectionSite::None;

    updateNgxPrePostInjectionAim();

    m_ngxHudlessCapturedThisFrame = false;
    m_ngxNeuralRenderingAppliedThisFrame = false;
    m_ngxHudlessSceneCompositeSeen = false;

    // Object velocity per-frame state: rotate the accepted camera, drop uncaptured
    // leftovers, and bound the identity cache (level transitions leave stale entries
    // behind; re-registration costs one frame of velocity). A minimized present that does
    // not count as a frame produced no camera, so copying it over the previous one would
    // drop the history the frozen frame counter is keeping alive.
    if (countAsFrame) {
      if (m_ngxFrameCameraValid && m_ngxPrevCameraValid &&
          m_ngxFrameCameraUsedTranspose != m_ngxPrevCameraUsedTranspose) {
        m_ngxCameraTransposeFlips++;
      }
      m_ngxPrevCameraValid = m_ngxFrameCameraValid;
      m_ngxPrevCameraUsedTranspose = m_ngxFrameCameraUsedTranspose;
    }
    m_ngxFrameCameraValid = false;
    m_ngxFrameCameraMatricesValid = false;
    // Settle on the best supported offset of the frame, which may be better supported than
    // whichever candidate led at the moment it was adopted. A frame that produced no evidence
    // at all leaves the standing offset alone rather than dropping it: zeroing it would put a
    // step the size of the camera's travel through the next frame's reprojection, and a frame
    // with nothing to measure from is equally a frame with nothing to misjudge.
    {
      const NgxTranslationDeltaVote* winner = nullptr;
      for (const NgxTranslationDeltaVote& vote : m_ngxTranslationDeltaVotes) {
        if (vote.votes > 0 && (winner == nullptr || vote.votes > winner->votes)) {
          winner = &vote;
        }
      }
      if (winner != nullptr) {
        m_ngxGlobalTransformOffset = winner->delta;
      }
      m_ngxTranslationDeltaVotes.fill(NgxTranslationDeltaVote());
      m_ngxAdoptedOffsetVotes = 0;
    }

    m_ngxVelocityStats = NgxVelocityCaptureStats();
    m_ngxVelocitySkinnedDraws = 0;
    m_ngxVelocityDynamicDraws = 0;
    m_ngxVelocityDraws.clear();
    // Instances left behind by level streaming and visibility changes accumulate, so the cache
    // drops the ones not sighted for a while: as routine upkeep, and as soon as it grows past
    // the ceiling. The ceiling bounds memory, not correctness - discarding the cache wholesale
    // takes every object's previous transform with it, so a level dense enough to sit above the
    // ceiling loses its history on every frame and the scene renders with no object velocities
    // at all. Pruning by age instead only ever drops what could not have paired anyway, and
    // tightens the age until the cache fits.
    constexpr size_t kVelocityCacheCeiling = 32768;
    constexpr uint32_t kVelocityCacheRoutinePruneAge = 1024;

    auto pruneVelocityInstancesOlderThan = [this](const uint32_t maxAge) {
      m_ngxVelocityObjectCache.erase_if([frame = m_ue3FrameCounter, maxAge](auto it) {
        auto& instances = it->second.instances;
        instances.erase(std::remove_if(instances.begin(), instances.end(),
                                       [frame, maxAge](const NgxVelocityObjectInstance& instance) {
                                         return instance.lastSeenFrame + maxAge < frame;
                                       }),
                        instances.end());
        return instances.empty();
      });
    };

    if ((m_ue3FrameCounter & 0xFF) == 0 || m_ngxVelocityObjectCache.size() > kVelocityCacheCeiling) {
      uint32_t maxAge = kVelocityCacheRoutinePruneAge;
      pruneVelocityInstancesOlderThan(maxAge);

      // Pairing only ever reaches one frame back, so tightening the age surrenders nothing that
      // was going to be used. At the floor the cache holds only what the last couple of frames
      // sighted, which the per-frame draw count bounds on its own.
      while (m_ngxVelocityObjectCache.size() > kVelocityCacheCeiling && maxAge > 2) {
        maxAge /= 4;
        pruneVelocityInstancesOlderThan(maxAge);
      }
    }
    m_ngxLastCameraConstantsHash = 0;
    m_ngxSceneViewportValid = false;
    m_ngxUpscaleSourceImage = nullptr;
    m_ngxColorTargetImage = nullptr;
    m_ngxColorMirrorImage = nullptr;
    m_ngxSubrect = { { 0, 0 }, { 0, 0 } };
    m_ngxColorSubrectOffset = { 0, 0 };
    m_ngxRuntimeUpscaleRectValid = false;
    m_ngxOutputTransform = NgxOutputTransform();
    for (uint32_t i = 0; i < m_ngxSceneColorResolveCount; i++) {
      m_ngxSceneColorResolves[i] = nullptr;
    }
    m_ngxSceneColorResolveCount = 0;
    for (uint32_t i = 0; i < m_ngxUberEncodedImageCount; i++) {
      m_ngxUberEncodedImages[i] = nullptr;
    }
    m_ngxUberEncodedImageCount = 0;
    m_ngxVelocityDumpSkinnedThisFrame = 0;
    m_ngxVelocityDumpDynamicThisFrame = 0;
    m_ngxVelocityDumpRigidThisFrame = 0;
    if (m_ngxVelocityDumpFramesLeft > 0) {
      m_ngxVelocityDumpFramesLeft--;
    }
    if (m_ngxPostChainDumpFramesLeft > 0) {
      m_ngxPostChainDumpFramesLeft--;
      Logger::info(str::format("[RTX NGX Passthrough][dump] ---- end of frame ", m_ue3FrameCounter, " ----"));
    }
    m_ngxPostChainDumpLinesThisFrame = 0;
    // Drop scene target references when unseen for a while (level transitions recreate them)
    if (m_ngxSceneColorImage != nullptr && m_ue3FrameCounter - m_ngxSceneTargetsLastSeenFrame > 60) {
      m_ngxSceneColorImage = nullptr;
      m_ngxSceneDepthImage = nullptr;
    }
  }

  void D3D9Rtx::OnPresent(const Rc<DxvkImage>& targetImage) {
    // Inform backend of present
    m_parent->EmitCs([targetImage](DxvkContext* ctx) { static_cast<RtxContext*>(ctx)->onPresent(targetImage); });
  }
}
