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

  bool tryExtractUe3WorldToViewAndProjectionFromShaderConstants(
    const D3D9ShaderConstantsVSSoftware& vsConsts,
    const uint32_t viewProjRegisterBase,
    const uint32_t viewOriginRegister,
    Matrix4& outWorldToView,
    Matrix4& outViewToProjection,
    bool* outUsedTranspose,
    float* outReconstructionError) {

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
