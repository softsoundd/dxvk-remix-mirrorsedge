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

  // Solves the view origin of a perspective world-to-projection matrix,
  // which is the point that projects to clip x = y = w = 0.
  static bool solveUe3CameraPositionFromViewProjection(const Matrix4& worldToProjection, Vector3& outPos) {
    const uint32_t rows[3] = { 0, 1, 3 };
    double m[3][4];
    for (uint32_t r = 0; r < 3; r++) {
      for (uint32_t c = 0; c < 4; c++) {
        m[r][c] = worldToProjection[c][rows[r]];
      }
    }
    auto det3 = [](const double a[3][3]) {
      return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
             a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
             a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    };
    double a[3][3];
    for (uint32_t r = 0; r < 3; r++) {
      for (uint32_t c = 0; c < 3; c++) {
        a[r][c] = m[r][c];
      }
    }
    const double det = det3(a);
    if (!std::isfinite(det) || std::abs(det) < 1e-18) {
      return false;
    }
    double solution[3];
    for (uint32_t i = 0; i < 3; i++) {
      double ai[3][3];
      for (uint32_t r = 0; r < 3; r++) {
        for (uint32_t c = 0; c < 3; c++) {
          ai[r][c] = (c == i) ? -m[r][3] : m[r][c];
        }
      }
      solution[i] = det3(ai) / det;
    }
    outPos = Vector3(static_cast<float>(solution[0]), static_cast<float>(solution[1]), static_cast<float>(solution[2]));
    return std::isfinite(outPos.x) && std::isfinite(outPos.y) && std::isfinite(outPos.z);
  }

  bool extractUe3CameraMatrices(
    const D3D9ShaderConstantsVSSoftware& vsConsts,
    const uint32_t viewProjRegisterBase,
    const uint32_t viewOriginRegister,
    const bool deriveCameraPosition,
    Matrix4& outWorldToView,
    Matrix4& outViewToProjection,
    bool* outUsedTranspose,
    float* outReconstructionError) {

    // ViewProjectionMatrix arrives as 4 column-major registers. The world-to-view and projection pair is
    // recovered by unprojecting NDC points with the camera position, then validated by decomposition.

    if (viewProjRegisterBase + 3 >= caps::MaxFloatConstantsSoftware) {
      return false;
    }
    if (!deriveCameraPosition && viewOriginRegister >= caps::MaxFloatConstantsSoftware) {
      return false;
    }

    Matrix4 worldToProjection;
    worldToProjection[0] = vsConsts.fConsts[viewProjRegisterBase + 0];
    worldToProjection[1] = vsConsts.fConsts[viewProjRegisterBase + 1];
    worldToProjection[2] = vsConsts.fConsts[viewProjRegisterBase + 2];
    worldToProjection[3] = vsConsts.fConsts[viewProjRegisterBase + 3];

    // A derived position is solved per candidate orientation inside tryBuild.
    const Vector3 registerCamPos =
      deriveCameraPosition ? Vector3(0.0f, 0.0f, 0.0f) : vsConsts.fConsts[viewOriginRegister].xyz();

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
      if (!finite) {
        return false;
      }

      if (outParams.fov < 0.001f) {
        return false;
      }
      if (std::abs(outParams.shearX) > 0.01f) {
        return false;
      }
      if (outParams.nearPlane <= 0.0f) {
        return false;
      }
      if (outParams.farPlane <= outParams.nearPlane) {
        return false;
      }

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
      Vector3 camPos = registerCamPos;
      if (deriveCameraPosition && !solveUe3CameraPositionFromViewProjection(candidateWorldToProjection, camPos)) {
        return false;
      }

      // avoid attempting to invert singular matrices
      {
        constexpr double kDetEps = 1e-24;
        const double det = determinant(candidateWorldToProjection);
        if (!std::isfinite(det) || std::abs(det) <= kDetEps) {
          return false;
        }
      }

      // invert world to projection to unproject a few NDC points
      Matrix4 invWorldToProjection;
      invWorldToProjection = inverse(candidateWorldToProjection);

      auto unprojectNdc = [&](float ndcX, float ndcY, float ndcZ, Vector3& outWorldPos) -> bool {
        const Vector4 clip(ndcX, ndcY, ndcZ, 1.0f);
        const Vector4 worldH = invWorldToProjection * clip;
        if (!std::isfinite(worldH.w) || std::abs(worldH.w) < kEps) {
          return false;
        }
        const float invW = 1.0f / worldH.w;
        outWorldPos = worldH.xyz() * invW;
        return std::isfinite(outWorldPos.x) && std::isfinite(outWorldPos.y) && std::isfinite(outWorldPos.z);
      };

      // reference - D3D NDC: x/y in [-1, 1], z in [0, 1]
      constexpr float ndcZ = 0.5f;
      Vector3 worldCenter, worldUp, worldRight;
      if (!unprojectNdc(0.0f, 0.0f, ndcZ, worldCenter)) {
        return false;
      }
      if (!unprojectNdc(0.0f, 1.0f, ndcZ, worldUp)) {
        return false;
      }
      if (!unprojectNdc(1.0f, 0.0f, ndcZ, worldRight)) {
        return false;
      }

      Vector3 forward = worldCenter - camPos;
      if (lengthSqr(forward) < kEps) {
        return false;
      }
      forward = normalize(forward);

      Vector3 upHint = worldUp - worldCenter;
      if (lengthSqr(upHint) < kEps) {
        upHint = Vector3(0.0f, 1.0f, 0.0f);
      } else {
        upHint = normalize(upHint);
      }

      // construct an orthonormal basis, we use the unprojected "up" direction as a hint to fix roll
      Vector3 right = cross(upHint, forward);
      if (lengthSqr(right) < kEps) {
        return false;
      }
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
        if (lengthSqr(right) < kEps) {
          return false;
        }
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

      if (!found) {
        return false;
      }

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
      if (!tryBuild(candidateWorldToProjection, w2v, v2p)) {
        return r;
      }
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
    if (outUsedTranspose != nullptr) {
      *outUsedTranspose = (best == &transposed);
    }
    if (outReconstructionError != nullptr) {
      *outReconstructionError = best->error;
    }
    return true;
  }

  // Camera and object-to-world from the vertex shader's CTAB. False when the draw is a
  // SceneCapture view to ignore.
  bool D3D9Rtx::applyUe3ShaderConstantTransforms(const DrawContext& drawContext, DrawCallTransforms& transformData) {
    const bool isUe3Mode = m_frameOptions.ue3EngineMode;
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
          if (bytecode.size() < sizeof(uint32_t) || (bytecode.size() % sizeof(uint32_t)) != 0) {
            return info;
          }

          const uint32_t* tokens = reinterpret_cast<const uint32_t*>(bytecode.data());
          const uint32_t headerToken = tokens[0];
          const uint32_t headerTypeMask = headerToken & 0xffff0000u;

          DxsoProgramType programType;
          if (headerTypeMask == 0xffff0000u) {
            programType = DxsoProgramTypes::PixelShader;
          } else if (headerTypeMask == 0xfffe0000u) {
            programType = DxsoProgramTypes::VertexShader;
          } else {
            return info;
          }

          const uint32_t majorVersion = (headerToken >> 8) & 0xffu;
          const uint32_t minorVersion = headerToken & 0xffu;
          DxsoProgramInfo programInfo { programType, minorVersion, majorVersion };

          DxsoDecodeContext decoder(programInfo);
          DxsoCodeIter iter(tokens + 1);

          while (decoder.decodeInstruction(iter)) {
            if (decoder.getCtabInfo().m_size != 0) {
              break;
            }
          }

          const DxsoCtab& ctab = decoder.getCtabInfo();
          if (ctab.m_size == 0 || ctab.m_constantData.empty()) {
            return info;
          }

          auto lower = [](const std::string& s) {
            std::string out;
            out.reserve(s.size());
            for (const char c : s) {
              out.push_back(char(std::tolower(static_cast<unsigned char>(c))));
            }
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
                    contains(name, "prev_local_to_world")) {
                  continue;
                }

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
        if (ue3CtabInfoPtr->hasViewProjectionMatrix) {
          viewProjReg = ue3CtabInfoPtr->viewProjectionMatrixRegisterIndex;
        }
        if (ue3CtabInfoPtr->hasCameraPosition) {
          viewOriginReg = ue3CtabInfoPtr->cameraPositionRegisterIndex;
        }
      }

      // Shaders that name ViewProjectionMatrix but no camera position get the position from the matrix.
      const bool deriveCameraPosition =
        m_frameOptions.ue3DeriveCameraPositionFromViewProjection &&
        ue3CtabInfoPtr != nullptr &&
        ue3CtabInfoPtr->hasViewProjectionMatrix &&
        !ue3CtabInfoPtr->hasCameraPosition;

      // cache by raw constant values to avoid repeated heavy extraction work per draw call
      struct Ue3CameraConstsKey {
        uint32_t viewProjReg;
        uint32_t viewOriginReg;
        Vector4 regs[5];
      };

      auto tryApplyFromConstants = [&](Matrix4& outWorldToView, Matrix4& outViewToProjection, bool& outUsedTranspose, float& outReconstructionError) -> bool {
        if (viewProjReg + 3 >= caps::MaxFloatConstantsSoftware ||
            (!deriveCameraPosition && viewOriginReg >= caps::MaxFloatConstantsSoftware)) {
          return false;
        }

        Ue3CameraConstsKey key {};
        key.viewProjReg = viewProjReg;
        // A derived position depends on the matrix alone, so it is keyed apart from register-based entries.
        key.viewOriginReg = deriveCameraPosition ? UINT32_MAX : viewOriginReg;
        key.regs[0] = d3d9State().vsConsts.fConsts[viewProjReg + 0];
        key.regs[1] = d3d9State().vsConsts.fConsts[viewProjReg + 1];
        key.regs[2] = d3d9State().vsConsts.fConsts[viewProjReg + 2];
        key.regs[3] = d3d9State().vsConsts.fConsts[viewProjReg + 3];
        key.regs[4] = deriveCameraPosition ? Vector4(0.0f, 0.0f, 0.0f, 0.0f) : d3d9State().vsConsts.fConsts[viewOriginReg];

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
        const bool extracted = extractUe3CameraMatrices(
            d3d9State().vsConsts, viewProjReg, viewOriginReg, deriveCameraPosition,
            ue3WorldToView, ue3ViewToProjection, &usedTranspose, &reconstructionError);

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
      // (e.g. shadow depth) that still reconstruct as a plausible camera. With
      // ue3DeriveCameraPositionFromViewProjection a named ViewProjectionMatrix alone is enough, since the
      // position then comes from that matrix rather than from an unverified register.
      const bool ctabVerifiedCamera =
        ue3CtabInfoPtr != nullptr &&
        ue3CtabInfoPtr->hasViewProjectionMatrix &&
        (ue3CtabInfoPtr->hasCameraPosition || deriveCameraPosition);

      // Reflect and portal probes render through mirrored or obliquely clipped views the viewport heuristic
      // cannot see, and their geometry is unusable. CTAB-verified cameras only: fallback registers can hold
      // arbitrary data (see "Skipped passes" in UE3Compatibility.md).
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
        // A mirror premultiplied into the view flips the sign of the ViewProjection 3x3 determinant. Tested on
        // the raw registers, since the extraction rebuilds an orthonormal basis.
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
        // A world-geometry draw whose declared ViewProjectionMatrix cannot be extracted is a reflect or portal
        // probe (oblique near plane, mirrored view); the main view always extracts.
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

          // Every input to the disambiguation below is in the key, so a hit equals a recomputation.
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

    return true;
  }

  // UE3 draws two-sided translucency as a back-face pass then a front-face pass; the second is skipped.
  // The match must be strict, since back-to-front sorting makes consecutive draws sharing a material common.
  bool D3D9Rtx::isUe3SecondTwoSidedTranslucentPass(const DrawContext& drawContext) {
    const bool usesProgrammableVs = m_parent->UseProgrammableVS();
    if (m_frameOptions.ue3EngineMode && usesProgrammableVs && d3d9State().vertexShader.ptr() != nullptr &&
        d3d9State().pixelShader.ptr() != nullptr) {
      const XXH64_hash_t vsHash = d3d9State().vertexShader->GetCommonShader()->GetBytecodeHash();
      const XXH64_hash_t psHash = d3d9State().pixelShader->GetCommonShader()->GetBytecodeHash();
      const XXH64_hash_t vsPsHash = vsHash ^ (psHash * 0x9E3779B97F4A7C15ull);

      XXH64_hash_t boundTextureHash = 0;
      {
        const BoundTextureSnapshot& boundTextures = ensureBoundTextureSnapshot();
        for (const uint32_t s : bit::BitMask(boundTextures.mask & 0xFu)) {
          if (boundTextures.entries[s].hasImage) {
            boundTextureHash ^= boundTextures.entries[s].imageHash;
          }
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
      if (d3d9State().renderStates[D3DRS_ALPHABLENDENABLE]) {
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
        d3d9State().renderStates[D3DRS_ALPHABLENDENABLE];

      m_prevDrawVsPsHash = vsPsHash;
      m_prevDrawTextureHash = boundTextureHash;
      m_prevDrawGeometryHash = geometryIdentityHash;
      m_prevDrawCullMode = cullMode;

      if (isSecondTwoSidedPass) {
        ONCE(Logger::info("[RTX-Compatibility-Info] Skipped UE3 two-pass translucent draw (second cull-mode pass)."));
        return true;
      }
    } else {
      m_prevDrawVsPsHash = 0;
      m_prevDrawTextureHash = 0;
      m_prevDrawGeometryHash = 0;
      m_prevDrawCullMode = 0;
    }

    return false;
  }

}
