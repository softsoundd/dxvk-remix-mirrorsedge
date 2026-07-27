/*
* Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
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
#include <cmath>

#include "rtx_context.h"
#include "rtx_postFx.h"
#include "rtx_camera.h"
#include "rtx_scene_manager.h"
#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_render/rtx_shader_manager.h"
#include "rtx/pass/post_fx/post_fx.h"

#include <rtx_shaders/post_fx.h>
#include <rtx_shaders/post_fx_highlight.h>
#include <rtx_shaders/post_fx_motion_blur.h>
#include <rtx_shaders/post_fx_motion_blur_prefilter.h>
#include <rtx_shaders/post_fx_motion_blur_cine_setup.h>
#include <rtx_shaders/post_fx_motion_blur_cine_tilemax.h>
#include <rtx_shaders/post_fx_motion_blur_cine_neighbormax.h>
#include <rtx_shaders/post_fx_motion_blur_cine_gather.h>
#include <pxr/base/arch/math.h>
#include "rtx_imgui.h"

#include "../util/util_global_time.h"

namespace dxvk {
  std::array<uint8_t, 3> g_customHighlightColor = { 118, 185, 0 };

  // Defined within an unnamed namespace to ensure unique definition across binary
  namespace {
    class PostFxShader : public ManagedShader
    {
      SHADER_SOURCE(PostFxShader, VK_SHADER_STAGE_COMPUTE_BIT, post_fx)

      PUSH_CONSTANTS(PostFxArgs)

      BEGIN_PARAMETER()
        SAMPLER2D(POST_FX_INPUT)
        RW_TEXTURE2D(POST_FX_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(PostFxShader);

    class PostFxMotionBlurShader : public ManagedShader
    {
      SHADER_SOURCE(PostFxMotionBlurShader, VK_SHADER_STAGE_COMPUTE_BIT, post_fx_motion_blur)

      PUSH_CONSTANTS(PostFxArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(POST_FX_MOTION_BLUR_PRIMARY_SCREEN_SPACE_MOTION_INPUT)
        TEXTURE2D(POST_FX_MOTION_BLUR_PRIMARY_SURFACE_FLAGS_INPUT)
        TEXTURE2D(POST_FX_MOTION_BLUR_PRIMARY_LINEAR_VIEW_Z_INPUT)
        TEXTURE2DARRAY(POST_FX_MOTION_BLUR_BLUE_NOISE_TEXTURE_INPUT)
        TEXTURE2D(POST_FX_MOTION_BLUR_INPUT)
        SAMPLER(POST_FX_MOTION_BLUR_NEAREST_SAMPLER)
        SAMPLER(POST_FX_MOTION_BLUR_LINEAR_SAMPLER)
        RW_TEXTURE2D(POST_FX_MOTION_BLUR_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(PostFxMotionBlurShader);

    class PostFxMotionBlurPrefilterShader : public ManagedShader
    {
      SHADER_SOURCE(PostFxMotionBlurPrefilterShader, VK_SHADER_STAGE_COMPUTE_BIT, post_fx_motion_blur_prefilter)

      PUSH_CONSTANTS(PostFxMotionBlurPrefilterArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(POST_FX_MOTION_BLUR_PREFILTER_PRIMARY_SURFACE_FLAGS_INPUT)
        RW_TEXTURE2D(POST_FX_MOTION_BLUR_PREFILTER_PRIMARY_SURFACE_FLAGS_FILTERED_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(PostFxMotionBlurPrefilterShader);

    class PostFxMotionBlurCineSetupShader : public ManagedShader
    {
      SHADER_SOURCE(PostFxMotionBlurCineSetupShader, VK_SHADER_STAGE_COMPUTE_BIT, post_fx_motion_blur_cine_setup)

      BEGIN_PARAMETER()
        TEXTURE2D(POST_FX_MB_CINE_SETUP_MOTION_VECTOR_INPUT)
        TEXTURE2D(POST_FX_MB_CINE_SETUP_SURFACE_FLAGS_INPUT)
        TEXTURE2D(POST_FX_MB_CINE_SETUP_LINEAR_VIEW_Z_INPUT)
        TEXTURE2D(POST_FX_MB_CINE_SETUP_PREV_MOTION_VECTOR_INPUT)
        RW_TEXTURE2D(POST_FX_MB_CINE_SETUP_VELOCITY_DEPTH_OUTPUT)
        RW_TEXTURE2D(POST_FX_MB_CINE_SETUP_CURVATURE_OUTPUT)
        CONSTANT_BUFFER(POST_FX_MB_CINE_SETUP_CONSTANTS)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(PostFxMotionBlurCineSetupShader);

    class PostFxMotionBlurCineTileMaxShader : public ManagedShader
    {
      SHADER_SOURCE(PostFxMotionBlurCineTileMaxShader, VK_SHADER_STAGE_COMPUTE_BIT, post_fx_motion_blur_cine_tilemax)

      PUSH_CONSTANTS(PostFxMotionBlurCineTileArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(POST_FX_MB_CINE_TILEMAX_INPUT)
        RW_TEXTURE2D(POST_FX_MB_CINE_TILEMAX_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(PostFxMotionBlurCineTileMaxShader);

    class PostFxMotionBlurCineNeighborMaxShader : public ManagedShader
    {
      SHADER_SOURCE(PostFxMotionBlurCineNeighborMaxShader, VK_SHADER_STAGE_COMPUTE_BIT, post_fx_motion_blur_cine_neighbormax)

      PUSH_CONSTANTS(PostFxMotionBlurCineTileArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(POST_FX_MB_CINE_NEIGHBORMAX_INPUT)
        RW_TEXTURE2D(POST_FX_MB_CINE_NEIGHBORMAX_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(PostFxMotionBlurCineNeighborMaxShader);

    class PostFxMotionBlurCineGatherShader : public ManagedShader
    {
      SHADER_SOURCE(PostFxMotionBlurCineGatherShader, VK_SHADER_STAGE_COMPUTE_BIT, post_fx_motion_blur_cine_gather)

      PUSH_CONSTANTS(PostFxMotionBlurCineGatherArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(POST_FX_MB_CINE_GATHER_VELOCITY_DEPTH_INPUT)
        TEXTURE2D(POST_FX_MB_CINE_GATHER_CURVATURE_INPUT)
        TEXTURE2D(POST_FX_MB_CINE_GATHER_TILE_MAX_INPUT)
        TEXTURE2D(POST_FX_MB_CINE_GATHER_NEIGHBOR_MAX_INPUT)
        TEXTURE2D(POST_FX_MB_CINE_GATHER_COLOR_INPUT)
        RW_TEXTURE2D(POST_FX_MB_CINE_GATHER_OUTPUT)
        SAMPLER(POST_FX_MB_CINE_GATHER_LINEAR_SAMPLER)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(PostFxMotionBlurCineGatherShader);

    class PostFxHighlightShader : public ManagedShader {
      SHADER_SOURCE(PostFxHighlightShader, VK_SHADER_STAGE_COMPUTE_BIT, post_fx_highlight)

        PUSH_CONSTANTS(PostFxHighlightingArgs)

        BEGIN_PARAMETER()
        TEXTURE2D(POST_FX_HIGHLIGHT_INPUT)
        RW_TEXTURE2D(POST_FX_HIGHLIGHT_OBJECT_PICKING_INPUT)
        TEXTURE2D(POST_FX_HIGHLIGHT_PRIMARY_CONE_RADIUS_INPUT)
        RW_TEXTURE2D(POST_FX_HIGHLIGHT_OUTPUT)
        STRUCTURED_BUFFER(POST_FX_HIGHLIGHT_VALUES)
        END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(PostFxHighlightShader);
  }

  namespace {
    // Camera pose interpolation for the curved sample paths.
    //
    // A rotating camera sweeps a static world point along an arc in screen space even at
    // constant angular velocity, because the perspective divide is non-linear - a point near
    // the edge of frame travels further than one near the centre, and the path bends. That
    // is the dominant curvature in a first person game, and recovering it needs nothing more
    // than the current and previous orientation. Position additionally uses the frame before
    // that, so acceleration (landing from a fall, being launched) is captured too.
    struct CameraPose {
      Vector3d position;
      Vector4d rotation; // xyz = imaginary, w = real
    };

    Vector4d rotationFromViewToWorld(const Matrix4d& viewToWorld) {
      // Shepperd's method on the upper 3x3. Unlike matrixToQuaternion in util_quat.h this
      // must not fold handedness into the sign, since the result gets interpolated.
      const double m00 = viewToWorld[0][0], m01 = viewToWorld[1][0], m02 = viewToWorld[2][0];
      const double m10 = viewToWorld[0][1], m11 = viewToWorld[1][1], m12 = viewToWorld[2][1];
      const double m20 = viewToWorld[0][2], m21 = viewToWorld[1][2], m22 = viewToWorld[2][2];

      const double trace = m00 + m11 + m22;
      Vector4d q;

      if (trace > 0.0) {
        const double s = std::sqrt(trace + 1.0) * 2.0;
        q.w = 0.25 * s;
        q.x = (m21 - m12) / s;
        q.y = (m02 - m20) / s;
        q.z = (m10 - m01) / s;
      } else if (m00 > m11 && m00 > m22) {
        const double s = std::sqrt(1.0 + m00 - m11 - m22) * 2.0;
        q.w = (m21 - m12) / s;
        q.x = 0.25 * s;
        q.y = (m01 + m10) / s;
        q.z = (m02 + m20) / s;
      } else if (m11 > m22) {
        const double s = std::sqrt(1.0 + m11 - m00 - m22) * 2.0;
        q.w = (m02 - m20) / s;
        q.x = (m01 + m10) / s;
        q.y = 0.25 * s;
        q.z = (m12 + m21) / s;
      } else {
        const double s = std::sqrt(1.0 + m22 - m00 - m11) * 2.0;
        q.w = (m10 - m01) / s;
        q.x = (m02 + m20) / s;
        q.y = (m12 + m21) / s;
        q.z = 0.25 * s;
      }

      const double length = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
      return length > 1e-12 ? Vector4d(q.x / length, q.y / length, q.z / length, q.w / length)
                            : Vector4d(0.0, 0.0, 0.0, 1.0);
    }

    // Spherical interpolation that also accepts s outside [0,1], which is how the half frame
    // forward pose is produced from the frame's own rotation.
    Vector4d slerpRotation(const Vector4d& from, Vector4d to, const double s) {
      double cosHalfTheta = from.x * to.x + from.y * to.y + from.z * to.z + from.w * to.w;

      if (cosHalfTheta < 0.0) {
        to = Vector4d(-to.x, -to.y, -to.z, -to.w);
        cosHalfTheta = -cosHalfTheta;
      }

      // Nearly co-linear: fall back to a normalized linear blend to avoid dividing by a
      // vanishing sine. Also the common case, since a frame of rotation is a small angle.
      if (cosHalfTheta > 0.9995) {
        const Vector4d result(from.x + (to.x - from.x) * s,
                              from.y + (to.y - from.y) * s,
                              from.z + (to.z - from.z) * s,
                              from.w + (to.w - from.w) * s);
        const double length = std::sqrt(result.x * result.x + result.y * result.y +
                                        result.z * result.z + result.w * result.w);
        return length > 1e-12 ? Vector4d(result.x / length, result.y / length, result.z / length, result.w / length)
                              : from;
      }

      const double halfTheta = std::acos(std::min(std::max(cosHalfTheta, -1.0), 1.0));
      const double sinHalfTheta = std::sin(halfTheta);
      const double fromScale = std::sin((1.0 - s) * halfTheta) / sinHalfTheta;
      const double toScale = std::sin(s * halfTheta) / sinHalfTheta;

      return Vector4d(from.x * fromScale + to.x * toScale,
                      from.y * fromScale + to.y * toScale,
                      from.z * fromScale + to.z * toScale,
                      from.w * fromScale + to.w * toScale);
    }

    Matrix4d viewToWorldFromPose(const CameraPose& pose) {
      const double x = pose.rotation.x, y = pose.rotation.y, z = pose.rotation.z, w = pose.rotation.w;

      Matrix4d result;
      result[0] = Vector4d(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w), 0.0);
      result[1] = Vector4d(2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w), 0.0);
      result[2] = Vector4d(2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y), 0.0);
      result[3] = Vector4d(pose.position.x, pose.position.y, pose.position.z, 1.0);
      return result;
    }

    Vector3d positionFromViewToWorld(const Matrix4d& viewToWorld) {
      return Vector3d(viewToWorld[3].x, viewToWorld[3].y, viewToWorld[3].z);
    }
  }

  // Maps current frame unjittered NDC plus depth to clip space half a frame either side of
  // now, which is what the setup pass fits its curved sample path through.
  DxvkPostFx::MotionBlurCurveMatrices buildMotionBlurCurveMatrices(
    const RtCamera& camera,
    const Matrix4d& previousSpaceFromCurrent)
  {
    DxvkPostFx::MotionBlurCurveMatrices result;

    const Matrix4d& viewToWorld = camera.getViewToWorld();
    const Matrix4d& previousViewToWorld = camera.getPreviousViewToWorld();
    const Matrix4d& previousPreviousViewToWorld = camera.getPreviousPreviousViewToWorld();

    const Vector3d p0 = positionFromViewToWorld(viewToWorld);
    const Vector3d p1 = positionFromViewToWorld(previousViewToWorld);
    const Vector3d p2 = positionFromViewToWorld(previousPreviousViewToWorld);

    const Vector4d q0 = rotationFromViewToWorld(viewToWorld);
    const Vector4d q1 = rotationFromViewToWorld(previousViewToWorld);

    // Lagrange quadratic through the samples at t = -2, -1, 0, evaluated at t = h.
    const auto positionAt = [&](const double h) {
      const double c2 = h * (h + 1.0) * 0.5;
      const double c1 = -h * (h + 2.0);
      const double c0 = (h + 2.0) * (h + 1.0) * 0.5;
      return Vector3d(p2.x * c2 + p1.x * c1 + p0.x * c0,
                      p2.y * c2 + p1.y * c1 + p0.y * c0,
                      p2.z * c2 + p1.z * c1 + p0.z * c0);
    };

    // slerp runs from the previous orientation at s = 0 to the current one at s = 1, so the
    // frame parameter maps as s = t + 1 and s = 1.5 extrapolates half a frame ahead.
    CameraPose previousHalf;
    previousHalf.position = positionAt(-0.5);
    previousHalf.rotation = slerpRotation(q1, q0, 0.5);

    CameraPose nextHalf;
    nextHalf.position = positionAt(0.5);
    nextHalf.rotation = slerpRotation(q1, q0, 1.5);

    // Where the game's world space shifts between frames the two cameras describe different
    // spaces; a half frame back is half of that shift. Identity for titles uploading true
    // world space, leaving the chain exactly as it would otherwise be.
    const auto scaledSpaceOffset = [&](const double scale) {
      Matrix4d offset = Matrix4d();
      offset[3] = Vector4d(previousSpaceFromCurrent[3].x * scale,
                           previousSpaceFromCurrent[3].y * scale,
                           previousSpaceFromCurrent[3].z * scale,
                           1.0);
      return offset;
    };

    const Matrix4d& viewToProjection = camera.getViewToProjection();
    const Matrix4d& projectionToView = camera.getProjectionToView();

    const Matrix4d toPrevHalf =
      viewToProjection * inverse(viewToWorldFromPose(previousHalf)) * scaledSpaceOffset(0.5) *
      viewToWorld * projectionToView;

    const Matrix4d toNextHalf =
      viewToProjection * inverse(viewToWorldFromPose(nextHalf)) * scaledSpaceOffset(-0.5) *
      viewToWorld * projectionToView;

    result.toPrevHalfClip = Matrix4(toPrevHalf);
    result.toNextHalfClip = Matrix4(toNextHalf);
    result.valid = true;

    return result;
  }

  DxvkPostFx::DxvkPostFx(DxvkDevice* device)
  : m_vkd(device->vkd())
  {
  }

  DxvkPostFx::~DxvkPostFx()
  {
  }

  void DxvkPostFx::prewarmShaders(DxvkPipelineManager& pipelineManager) const
  {
    PostFxShader::getShader();
    PostFxMotionBlurShader::getShader();
    PostFxMotionBlurPrefilterShader::getShader();
    PostFxMotionBlurCineSetupShader::getShader();
    PostFxMotionBlurCineTileMaxShader::getShader();
    PostFxMotionBlurCineNeighborMaxShader::getShader();
    PostFxMotionBlurCineGatherShader::getShader();
  }

  void DxvkPostFx::showImguiSettings()
  {
    RemixGui::Checkbox("Post Effect Enabled", &enableObject());
    if (enable())
    {
      RemixGui::Checkbox("Motion Blur Enabled", &enableMotionBlurObject());
      if (enableMotionBlur()) {
        RemixGui::Combo("Motion Blur Mode", &motionBlurModeObject(), "Legacy\0Cinematic (Feature-Aware)\0");

        if (motionBlurMode() == MotionBlurMode::Cinematic) {
          RemixGui::DragInt("Sample Count (N)", &motionBlurCineSampleCountObject(), 0.2f, 1, POST_FX_MB_MAX_SAMPLE_COUNT, "%d", ImGuiSliderFlags_AlwaysClamp);
          RemixGui::DragFloat("Shutter Angle (degrees)", &motionBlurShutterAngleObject(), 1.0f, 0.0f, 360.0f, "%.0f", ImGuiSliderFlags_AlwaysClamp);
          RemixGui::DragFloat("Shutter Reference FPS (0 = follow frame rate)", &motionBlurShutterReferenceFpsObject(), 1.0f, 0.0f, 240.0f, "%.0f", ImGuiSliderFlags_AlwaysClamp);
          RemixGui::DragFloat("Max Blur Radius (fraction of width)", &motionBlurCineMaxRadiusFractionObject(), 0.001f, 0.005f, 0.25f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
          RemixGui::Combo("Direction Split", &motionBlurCineDirectionSplitObject(), "Dominant only\0Even\0Tile variance\0");
          RemixGui::Checkbox("Curved Sample Paths", &motionBlurCineCurvedPathsObject());
          if (motionBlurCineCurvedPaths()) {
            RemixGui::Checkbox("Per-Object Curvature", &motionBlurCineObjectCurvatureObject());
          }
          RemixGui::Checkbox("Emissive Surface Enabled", &enableMotionBlurEmissiveObject());
          RemixGui::DragFloat("Minimum Velocity Threshold (pixels)", &motionBlurMinimumVelocityThresholdInPixelObject(), 0.01f, 0.01f, 3.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
          RemixGui::DragFloat("Dynamic Object Deduction", &motionBlurDynamicDeductionObject(), 0.001f, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);

          if (ImGui::CollapsingHeader("Filter Tuning (Guertin 2014)", ImGuiTreeNodeFlags_None)) {
            ImGui::Indent();
            RemixGui::DragFloat("Centre Weight Bias (k)", &motionBlurCineCenterWeightBiasObject(), 0.5f, 1.0f, 200.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
            RemixGui::DragFloat("Jitter Scale (phi)", &motionBlurCineJitterScaleObject(), 0.5f, 0.0f, 100.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
            RemixGui::DragFloat("Tile Blend Slope (tau)", &motionBlurCineTileBlendSlopeObject(), 0.01f, 0.0f, 4.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
            RemixGui::DragFloat("Centre Direction Threshold (gamma)", &motionBlurCineGammaThresholdObject(), 0.01f, 0.1f, 10.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
            RemixGui::Combo("Debug View", &motionBlurCineDebugViewObject(), "Off\0Velocity\0TileMax\0NeighborMax\0Tile Variance\0Blur Amount\0");
            ImGui::Unindent();
          }
        } else {
          RemixGui::Checkbox("Motion Blur Noise Sample Enabled", &enableMotionBlurNoiseSampleObject());
          RemixGui::Checkbox("Motion Blur Emissive Surface Enabled", &enableMotionBlurEmissiveObject());
          RemixGui::DragInt("Motion Blur Sample Count", &motionBlurSampleCountObject(), 0.1f, 1, 10, "%d", ImGuiSliderFlags_AlwaysClamp);
          RemixGui::DragFloat("Exposure Fraction", &exposureFractionObject(), 0.01f, 0.01f, 3.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
          RemixGui::DragFloat("Blur Diameter Fraction", &blurDiameterFractionObject(), 0.001f, 0.001f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
          RemixGui::DragFloat("Motion Blur Minimum Velocity Threshold (unit: pixel)", &motionBlurMinimumVelocityThresholdInPixelObject(), 0.01f, 0.01f, 3.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
          RemixGui::DragFloat("Motion Blur Dynamic Deduction", &motionBlurDynamicDeductionObject(), 0.001f, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
          RemixGui::DragFloat("Motion Blur Jitter Strength", &motionBlurJitterStrengthObject(), 0.001f, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        }
      }

      RemixGui::Checkbox("Chromatic Aberration Enabled", &enableChromaticAberrationObject());
      if (enableChromaticAberration()) {
        RemixGui::DragFloat("Fringe Intensity", &chromaticAberrationAmountObject(), 0.01f, 0.0f, 5.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
        RemixGui::DragFloat("Fringe Center Attenuation Amount", &chromaticCenterAttenuationAmountObject(), 0.001f, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
      }

      RemixGui::Checkbox("Vignette Enabled", &enableVignetteObject());
      if (enableVignette()) {
        RemixGui::DragFloat("Vignette Intensity", &vignetteIntensityObject(), 0.01f, 0.0f, 5.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
        RemixGui::DragFloat("Vignette Radius", &vignetteRadiusObject(), 0.001f, 0.0f, 1.4f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        RemixGui::DragFloat("Vignette Softness", &vignetteSoftnessObject(), 0.001f, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
      }
    }
  }

  void dispatchMotionBlurPrefilterPass(
    Rc<RtxContext> ctx,
    const Resources::Resource& primarySurfaceFlags,
    const Resources::Resource& primarySurfaceFlagsFilteredOutput,
    const bool isVertical)
  {
    ScopedGpuProfileZone(ctx, "PostFx Motion Blur Prefilter");

    const VkExtent3D& inputSize = primarySurfaceFlags.image->info().extent;
    const VkExtent3D workgroups = util::computeBlockCount(inputSize, VkExtent3D { POST_FX_TILE_SIZE , POST_FX_TILE_SIZE, 1 });

    PostFxMotionBlurPrefilterArgs postFxMotionBlurPrefilterArgs = {};
    postFxMotionBlurPrefilterArgs.imageSize = { (uint) inputSize.width, (uint) inputSize.height };
    if (isVertical)
    {
      postFxMotionBlurPrefilterArgs.pixelStep = { 0, 1 };
    }
    else
    {
      postFxMotionBlurPrefilterArgs.pixelStep = { 1, 0 };
    }

    ctx->pushConstants(0, sizeof(PostFxMotionBlurPrefilterArgs), &postFxMotionBlurPrefilterArgs);

    ctx->bindResourceView(POST_FX_MOTION_BLUR_PREFILTER_PRIMARY_SURFACE_FLAGS_INPUT, primarySurfaceFlags.view, nullptr);
    ctx->bindResourceView(POST_FX_MOTION_BLUR_PREFILTER_PRIMARY_SURFACE_FLAGS_FILTERED_OUTPUT, primarySurfaceFlagsFilteredOutput.view, nullptr);

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, PostFxMotionBlurPrefilterShader::getShader());

    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  // Cinematic motion blur (Guertin/McGuire/Nowrouzezahrai, HPG 2014). The prefiltered surface
  // flags are already in place when this runs; it adds the four passes the tile based filter
  // needs on top.
  void dispatchMotionBlurCine(
    Rc<RtxContext> ctx,
    Rc<DxvkSampler> linearSampler,
    const Rc<DxvkBuffer>& setupConstants,
    const PostFxMotionBlurCineSetupArgs& setupArgs,
    const PostFxMotionBlurCineGatherArgs& gatherArgs,
    const uint32_t tileSize,
    const DxvkPostFx::MotionBlurInputs& inputs)
  {
    const VkExtent3D imageExtent = { setupArgs.imageSize.x, setupArgs.imageSize.y, 1 };
    const VkExtent3D fullResWorkgroups =
      util::computeBlockCount(imageExtent, VkExtent3D { POST_FX_TILE_SIZE, POST_FX_TILE_SIZE, 1 });

    const uint32_t tileCountX = gatherArgs.tileCount.x;
    const uint32_t tileCountY = gatherArgs.tileCount.y;

    ctx->updateBuffer(setupConstants, 0, sizeof(setupArgs), &setupArgs);

    {
      ScopedGpuProfileZone(ctx, "PostFx Motion Blur Cine Setup");

      ctx->bindResourceView(POST_FX_MB_CINE_SETUP_MOTION_VECTOR_INPUT, inputs.screenSpaceMotionVector->view, nullptr);
      ctx->bindResourceView(POST_FX_MB_CINE_SETUP_SURFACE_FLAGS_INPUT, inputs.surfaceFlagsScratch2->view(Resources::AccessType::Read), nullptr);
      ctx->bindResourceView(POST_FX_MB_CINE_SETUP_LINEAR_VIEW_Z_INPUT, inputs.linearViewZ->view, nullptr);
      // The previous frame's field is optional; binding the current one keeps the descriptor
      // valid and the shader ignores it because enableObjectCurvature is cleared.
      ctx->bindResourceView(POST_FX_MB_CINE_SETUP_PREV_MOTION_VECTOR_INPUT,
                            inputs.previousScreenSpaceMotionVector != nullptr
                              ? inputs.previousScreenSpaceMotionVector->view
                              : inputs.screenSpaceMotionVector->view,
                            nullptr);
      ctx->bindResourceView(POST_FX_MB_CINE_SETUP_VELOCITY_DEPTH_OUTPUT, inputs.cineVelocityDepth->view, nullptr);
      ctx->bindResourceView(POST_FX_MB_CINE_SETUP_CURVATURE_OUTPUT, inputs.cineCurvature->view, nullptr);
      ctx->bindResourceBuffer(POST_FX_MB_CINE_SETUP_CONSTANTS, DxvkBufferSlice(setupConstants, 0, setupConstants->info().size));

      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, PostFxMotionBlurCineSetupShader::getShader());
      ctx->dispatch(fullResWorkgroups.width, fullResWorkgroups.height, 1);
    }

    // TileMax, split per axis. The paper notes the separable form is substantially cheaper
    // than a single 2D reduction for the cost of one intermediate.
    {
      ScopedGpuProfileZone(ctx, "PostFx Motion Blur Cine TileMax");

      PostFxMotionBlurCineTileArgs tileArgs = {};
      tileArgs.srcSize = setupArgs.imageSize;
      tileArgs.dstSize = { tileCountX, setupArgs.imageSize.y };
      tileArgs.pixelStep = { 1, 0 };
      tileArgs.tileSize = tileSize;

      ctx->pushConstants(0, sizeof(tileArgs), &tileArgs);
      ctx->bindResourceView(POST_FX_MB_CINE_TILEMAX_INPUT, inputs.cineVelocityDepth->view, nullptr);
      ctx->bindResourceView(POST_FX_MB_CINE_TILEMAX_OUTPUT, inputs.cineTileMaxX->view, nullptr);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, PostFxMotionBlurCineTileMaxShader::getShader());

      const VkExtent3D workgroupsX = util::computeBlockCount(
        VkExtent3D { tileCountX, setupArgs.imageSize.y, 1 }, VkExtent3D { POST_FX_TILE_SIZE, POST_FX_TILE_SIZE, 1 });
      ctx->dispatch(workgroupsX.width, workgroupsX.height, 1);

      tileArgs.srcSize = { tileCountX, setupArgs.imageSize.y };
      tileArgs.dstSize = { tileCountX, tileCountY };
      tileArgs.pixelStep = { 0, 1 };

      ctx->pushConstants(0, sizeof(tileArgs), &tileArgs);
      ctx->bindResourceView(POST_FX_MB_CINE_TILEMAX_INPUT, inputs.cineTileMaxX->view, nullptr);
      ctx->bindResourceView(POST_FX_MB_CINE_TILEMAX_OUTPUT, inputs.cineTileMax->view, nullptr);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, PostFxMotionBlurCineTileMaxShader::getShader());

      const VkExtent3D workgroupsY = util::computeBlockCount(
        VkExtent3D { tileCountX, tileCountY, 1 }, VkExtent3D { POST_FX_TILE_SIZE, POST_FX_TILE_SIZE, 1 });
      ctx->dispatch(workgroupsY.width, workgroupsY.height, 1);
    }

    {
      ScopedGpuProfileZone(ctx, "PostFx Motion Blur Cine NeighborMax");

      PostFxMotionBlurCineTileArgs neighborArgs = {};
      neighborArgs.srcSize = { tileCountX, tileCountY };
      neighborArgs.dstSize = { tileCountX, tileCountY };
      neighborArgs.pixelStep = { 0, 0 };
      neighborArgs.tileSize = tileSize;

      ctx->pushConstants(0, sizeof(neighborArgs), &neighborArgs);
      ctx->bindResourceView(POST_FX_MB_CINE_NEIGHBORMAX_INPUT, inputs.cineTileMax->view, nullptr);
      ctx->bindResourceView(POST_FX_MB_CINE_NEIGHBORMAX_OUTPUT, inputs.cineNeighborMax->view, nullptr);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, PostFxMotionBlurCineNeighborMaxShader::getShader());

      const VkExtent3D workgroups = util::computeBlockCount(
        VkExtent3D { tileCountX, tileCountY, 1 }, VkExtent3D { POST_FX_TILE_SIZE, POST_FX_TILE_SIZE, 1 });
      ctx->dispatch(workgroups.width, workgroups.height, 1);
    }

    {
      ScopedGpuProfileZone(ctx, "PostFx Motion Blur Cine Gather");

      ctx->pushConstants(0, sizeof(gatherArgs), &gatherArgs);
      ctx->bindResourceView(POST_FX_MB_CINE_GATHER_VELOCITY_DEPTH_INPUT, inputs.cineVelocityDepth->view, nullptr);
      ctx->bindResourceView(POST_FX_MB_CINE_GATHER_CURVATURE_INPUT, inputs.cineCurvature->view, nullptr);
      ctx->bindResourceView(POST_FX_MB_CINE_GATHER_TILE_MAX_INPUT, inputs.cineTileMax->view, nullptr);
      ctx->bindResourceView(POST_FX_MB_CINE_GATHER_NEIGHBOR_MAX_INPUT, inputs.cineNeighborMax->view, nullptr);
      ctx->bindResourceView(POST_FX_MB_CINE_GATHER_COLOR_INPUT, inputs.inOutColor->view, nullptr);
      ctx->bindResourceView(POST_FX_MB_CINE_GATHER_OUTPUT, inputs.intermediateColor->view, nullptr);
      ctx->bindResourceSampler(POST_FX_MB_CINE_GATHER_LINEAR_SAMPLER, linearSampler);

      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, PostFxMotionBlurCineGatherShader::getShader());
      ctx->dispatch(fullResWorkgroups.width, fullResWorkgroups.height, 1);
    }
  }

  void dispatchMotionBlurInternal(
    Rc<RtxContext> ctx,
    Rc<DxvkSampler> nearestSampler,
    Rc<DxvkSampler> linearSampler,
    const PostFxArgs& postFxArgs,
    const VkExtent3D& workgroups,
    const DxvkPostFx::MotionBlurInputs& inputs)
  {
    ScopedGpuProfileZone(ctx, "PostFx Motion Blur");

    dispatchMotionBlurPrefilterPass(ctx,
                                    *inputs.surfaceFlags,
                                    inputs.surfaceFlagsScratch1->resource(Resources::AccessType::Write),
                                    false);

    dispatchMotionBlurPrefilterPass(ctx,
                                    inputs.surfaceFlagsScratch1->resource(Resources::AccessType::Read),
                                    inputs.surfaceFlagsScratch2->resource(Resources::AccessType::Write),
                                    true);

    ctx->pushConstants(0, sizeof(postFxArgs), &postFxArgs);

    ctx->bindResourceView(POST_FX_MOTION_BLUR_PRIMARY_SCREEN_SPACE_MOTION_INPUT, inputs.screenSpaceMotionVector->view, nullptr);
    ctx->bindResourceView(POST_FX_MOTION_BLUR_PRIMARY_SURFACE_FLAGS_INPUT, inputs.surfaceFlagsScratch2->view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(POST_FX_MOTION_BLUR_PRIMARY_LINEAR_VIEW_Z_INPUT, inputs.linearViewZ->view, nullptr);
    ctx->bindResourceView(POST_FX_MOTION_BLUR_BLUE_NOISE_TEXTURE_INPUT, ctx->getResourceManager().getBlueNoiseTexture(ctx), nullptr);
    ctx->bindResourceView(POST_FX_MOTION_BLUR_INPUT, inputs.inOutColor->view, nullptr);
    ctx->bindResourceView(POST_FX_MOTION_BLUR_OUTPUT, inputs.intermediateColor->view, nullptr);
    ctx->bindResourceSampler(POST_FX_MOTION_BLUR_NEAREST_SAMPLER, nearestSampler);
    ctx->bindResourceSampler(POST_FX_MOTION_BLUR_LINEAR_SAMPLER, linearSampler);

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, PostFxMotionBlurShader::getShader());

    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void dispatchPostLensEffects(
    Rc<DxvkContext> ctx,
    Rc<DxvkSampler> linearSampler,
    const PostFxArgs& postFxArgs,
    const VkExtent3D& workgroups,
    const Resources::Resource& postFxLensEffectInput,
    const Resources::Resource& postFxLensEffectOutput)
  {
    ScopedGpuProfileZone(ctx, "PostFx Lens Effect");

    ctx->pushConstants(0, sizeof(postFxArgs), &postFxArgs);

    ctx->bindResourceView(POST_FX_INPUT, postFxLensEffectInput.view, nullptr);
    ctx->bindResourceSampler(POST_FX_INPUT, linearSampler);
    ctx->bindResourceView(POST_FX_OUTPUT, postFxLensEffectOutput.view, nullptr);

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, PostFxShader::getShader());

    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  namespace {
    // Simulate chromatic aberration offset scale by calculating the focal length differences of 3 Fraunhofer lines,
    // the wavelength of these lines are used for measuring chromatic aberrations
    // https://www.rp-photonics.com/chromatic_aberrations.html
    float2 calculateChromaticAberrationScale(const float chromaticAberrationAmount) {
      constexpr float lambdaC = 656.3f; // [nm] blue Fraunhofer F line from hydrogen
      constexpr float lambdaD = 589.2f; // [nm] orange Fraunhofer D line from sodium, in the region of maximum sensitivity of the human eye
      constexpr float lambdaF = 486.1f; // [nm] red Fraunhofer C line from hydrogen

      // https://www.rp-photonics.com/abbe_number.html
      constexpr float abbeNumber = 40.0f; // Use typical glass abbe number
      constexpr float focalD = 0.05f; // Use typical camera lens focal to represent focal of D line
      constexpr float fcFocalDiff = focalD / abbeNumber * 0.5f;

      const float2 scale = float2(fcFocalDiff * (lambdaC - lambdaD), fcFocalDiff * (lambdaD - lambdaF));

      return float2(scale.x * chromaticAberrationAmount, scale.y * chromaticAberrationAmount);
    }
  }

  void DxvkPostFx::dispatchMotionBlur(
    Rc<RtxContext> ctx,
    Rc<DxvkSampler> nearestSampler,
    Rc<DxvkSampler> linearSampler,
    const uvec2& mainCameraResolution,
    const uint32_t frameIdx,
    const Resources::RaytracingOutput& rtOutput,
    const bool cameraCutDetected)
  {
    MotionBlurInputs inputs = {};
    inputs.inOutColor = &rtOutput.m_finalOutput.resource(Resources::AccessType::ReadWrite);
    inputs.intermediateColor = &rtOutput.m_postFxIntermediateTexture;
    inputs.screenSpaceMotionVector = &rtOutput.m_primaryScreenSpaceMotionVector;
    inputs.surfaceFlags = &rtOutput.m_primarySurfaceFlags;
    inputs.surfaceFlagsScratch1 = &rtOutput.m_primarySurfaceFlagsIntermediateTexture1;
    inputs.surfaceFlagsScratch2 = &rtOutput.m_primarySurfaceFlagsIntermediateTexture2;
    inputs.linearViewZ = &rtOutput.m_primaryLinearViewZ;
    inputs.cineVelocityDepth = &rtOutput.m_motionBlurCineVelocityDepth;
    inputs.cineCurvature = &rtOutput.m_motionBlurCineCurvature;
    inputs.cineTileMaxX = &rtOutput.m_motionBlurCineTileMaxX;
    inputs.cineTileMax = &rtOutput.m_motionBlurCineTileMax;
    inputs.cineNeighborMax = &rtOutput.m_motionBlurCineNeighborMax;
    inputs.previousScreenSpaceMotionVector =
      rtOutput.m_primaryScreenSpaceMotionVectorQueue.hasDistinctPrevious()
        ? &rtOutput.m_primaryScreenSpaceMotionVectorQueue.getPrevious()
        : nullptr;

    // Only the cinematic filter consumes these, and fitting the curve costs a handful of
    // double precision inversions.
    if (motionBlurMode() == MotionBlurMode::Cinematic) {
      const RtCamera& camera = ctx->getSceneManager().getCamera();
      const auto nearFarPlanes = camera.calculateNearFarPlanes();
      inputs.nearPlane = nearFarPlanes.first;
      inputs.farPlane = nearFarPlanes.second;
      inputs.curves = buildMotionBlurCurveMatrices(camera);
    }

    dispatchMotionBlur(ctx, nearestSampler, linearSampler, mainCameraResolution, frameIdx, inputs, cameraCutDetected);
  }

  void DxvkPostFx::dispatchMotionBlurCinematic(
    Rc<RtxContext> ctx,
    Rc<DxvkSampler> linearSampler,
    const uvec2& mainCameraResolution,
    const uint32_t frameIdx,
    const MotionBlurInputs& inputs)
  {
    const VkExtent3D& imageExtent = inputs.inOutColor->image->info().extent;

    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);

    // Same surface flag prefilter the legacy path uses: expands the view model and emissive
    // flags by a pixel so their edges do not leak into the blur.
    dispatchMotionBlurPrefilterPass(ctx,
                                    *inputs.surfaceFlags,
                                    inputs.surfaceFlagsScratch1->resource(Resources::AccessType::Write),
                                    false);

    dispatchMotionBlurPrefilterPass(ctx,
                                    inputs.surfaceFlagsScratch1->resource(Resources::AccessType::Read),
                                    inputs.surfaceFlagsScratch2->resource(Resources::AccessType::Write),
                                    true);

    if (m_motionBlurCineConstants == nullptr) {
      DxvkBufferCreateInfo info = {};
      info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
      info.access = VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      info.size = align(sizeof(PostFxMotionBlurCineSetupArgs), kBufferAlignment);
      m_motionBlurCineConstants = ctx->getDevice()->createBuffer(
        info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer,
        "PostFx cinematic motion blur constants");
    }

    // The blur radius and the tile size are the same quantity: a pixel may only be reached
    // from within its one ring tile neighbourhood, so the tile has to be as wide as the
    // longest trail. Floored at the allocation granularity of the tile textures so that
    // changing the radius at runtime never needs a reallocation.
    const uint32_t maxRadiusPixels = std::max(
      static_cast<uint32_t>(motionBlurCineMaxRadiusFraction() * static_cast<float>(imageExtent.width) + 0.5f),
      static_cast<uint32_t>(POST_FX_MB_TILE_SIZE_MIN));
    const uint32_t tileSize = maxRadiusPixels;
    const uint32_t tileCountX = (imageExtent.width + tileSize - 1) / tileSize;
    const uint32_t tileCountY = (imageExtent.height + tileSize - 1) / tileSize;

    const float dlfgDeduction = ctx->isDLFGEnabled()
      ? 1.0f / static_cast<float>(ctx->dlfgInterpolatedFrameCount() + 1)
      : 1.0f;

    const float invWidth = 1.0f / static_cast<float>(imageExtent.width);
    const float invHeight = 1.0f / static_cast<float>(imageExtent.height);

    PostFxMotionBlurCineSetupArgs setupArgs = {};
    setupArgs.curveToPrevHalfClip = inputs.curves.toPrevHalfClip;
    setupArgs.curveToNextHalfClip = inputs.curves.toNextHalfClip;
    setupArgs.imageSize = { imageExtent.width, imageExtent.height };
    setupArgs.invImageSize = { invWidth, invHeight };
    setupArgs.inputOverOutputViewSize = {
      static_cast<float>(mainCameraResolution.x) * invWidth,
      static_cast<float>(mainCameraResolution.y) * invHeight
    };
    setupArgs.gbufferResolution = {
      static_cast<float>(mainCameraResolution.x),
      static_cast<float>(mainCameraResolution.y)
    };
    // Film convention: 180 degrees means the shutter is open for half of each frame. Motion
    // vectors already describe one frame of displacement, so the blur length tracks the frame
    // rate on its own - the same real motion blurs half as far at twice the frame rate, which
    // is exactly the motion lost between those two frames.
    //
    // A reference frame rate opts out of that, holding the blur at a fixed length the way
    // Unreal's r.MotionBlur.TargetFPS does. Off by default: pinned to a reference, the blur
    // overshoots the actual frame-to-frame displacement once the frame rate climbs past it,
    // so trails run ahead of the motion that produced them.
    float shutterFraction = std::max(motionBlurShutterAngle(), 0.0f) / 360.0f;
    if (motionBlurShutterReferenceFps() > 0.0f) {
      const float frameSeconds = GlobalTime::get().realDeltaTime();
      if (frameSeconds > 0.0f) {
        shutterFraction /= frameSeconds * motionBlurShutterReferenceFps();
      }
    }
    setupArgs.shutterFraction = shutterFraction;
    setupArgs.maxBlurRadiusPixels = static_cast<float>(maxRadiusPixels);
    setupArgs.dynamicDeduction = motionBlurDynamicDeduction();
    setupArgs.dlfgDeduction = dlfgDeduction;
    setupArgs.nearPlane = inputs.nearPlane;
    setupArgs.farPlane = inputs.farPlane;
    setupArgs.minVelocityPixels = motionBlurMinimumVelocityThresholdInPixel();
    setupArgs.enableEmissive = enableMotionBlurEmissive() ? 1u : 0u;

    const bool curvesUsable = inputs.curves.valid && inputs.farPlane > inputs.nearPlane && inputs.nearPlane > 0.0f;
    setupArgs.enableCurvedPaths = motionBlurCineCurvedPaths() ? 1u : 0u;
    setupArgs.cameraValid = curvesUsable ? 1u : 0u;
    setupArgs.enableObjectCurvature =
      (motionBlurCineObjectCurvature() && inputs.previousScreenSpaceMotionVector != nullptr) ? 1u : 0u;

    PostFxMotionBlurCineGatherArgs gatherArgs = {};
    gatherArgs.imageSize = setupArgs.imageSize;
    gatherArgs.invImageSize = setupArgs.invImageSize;
    gatherArgs.tileCount = { tileCountX, tileCountY };
    gatherArgs.tileSize = tileSize;
    gatherArgs.sampleCount = std::clamp(motionBlurCineSampleCount(), 1u, static_cast<uint32_t>(POST_FX_MB_MAX_SAMPLE_COUNT));
    gatherArgs.centerWeightBias = motionBlurCineCenterWeightBias();
    gatherArgs.jitterScale = motionBlurCineJitterScale();
    gatherArgs.tileBlendSlope = motionBlurCineTileBlendSlope();
    gatherArgs.gammaThreshold = motionBlurCineGammaThreshold();
    gatherArgs.frameIdx = frameIdx;
    gatherArgs.directionSplit = static_cast<uint32_t>(motionBlurCineDirectionSplit());
    gatherArgs.enableCurvedPaths = setupArgs.enableCurvedPaths;
    gatherArgs.minVelocityPixels = setupArgs.minVelocityPixels;
    gatherArgs.debugView = static_cast<uint32_t>(motionBlurCineDebugView());

    dispatchMotionBlurCine(ctx, linearSampler,
                           m_motionBlurCineConstants, setupArgs, gatherArgs, tileSize, inputs);
  }

  void DxvkPostFx::dispatchMotionBlur(
    Rc<RtxContext> ctx,
    Rc<DxvkSampler> nearestSampler,
    Rc<DxvkSampler> linearSampler,
    const uvec2& mainCameraResolution,
    const uint32_t frameIdx,
    const MotionBlurInputs& inputs,
    const bool cameraCutDetected)
  {
    if (!enable()) {
      return;
    }
    if (cameraCutDetected || !isMotionBlurEnabled()) {
      return;
    }

    ScopedGpuProfileZone(ctx, "PostFx Motion Blur");
    ctx->setFramePassStage(RtxFramePassStage::PostFX);

    const Resources::Resource& inOutColorTexture = *inputs.inOutColor;
    const VkExtent3D& inputSize = inOutColorTexture.image->info().extent;
    const VkExtent3D workgroups = util::computeBlockCount(inputSize, VkExtent3D { POST_FX_TILE_SIZE , POST_FX_TILE_SIZE, 1 } );

    // Cinematic mode needs its own intermediates. A caller that has not provided them falls
    // back rather than failing, which keeps paths that have not been extended working.
    const bool cineResourcesAvailable =
      inputs.cineVelocityDepth != nullptr &&
      inputs.cineCurvature != nullptr &&
      inputs.cineTileMaxX != nullptr &&
      inputs.cineTileMax != nullptr &&
      inputs.cineNeighborMax != nullptr;

    const bool useCinematic = motionBlurMode() == MotionBlurMode::Cinematic && cineResourcesAvailable;

    if (motionBlurMode() == MotionBlurMode::Cinematic && !cineResourcesAvailable) {
      ONCE(Logger::warn("[RTX PostFx] Cinematic motion blur requested but its intermediates are unavailable on this path; using the legacy filter."));
    }

    if (useCinematic) {
      dispatchMotionBlurCinematic(ctx, linearSampler, mainCameraResolution, frameIdx, inputs);

      ctx->copyImage(
        inOutColorTexture.image,
        { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        { 0, 0, 0 },
        inputs.intermediateColor->image,
        { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        { 0, 0, 0 },
        inputSize);
      return;
    }

    assert(motionBlurSampleCount() <= 10);

    PostFxArgs postFxArgs = {};
    postFxArgs.imageSize = { (uint)inputSize.width, (uint)inputSize.height };
    postFxArgs.invImageSize = { 1.0f / (float) inputSize.width, 1.0f / (float) inputSize.height };
    postFxArgs.invMainCameraResolution = float2(1.0f / (float)mainCameraResolution.x, 1.0f / (float)mainCameraResolution.y);
    postFxArgs.inputOverOutputViewSize = float2((float)mainCameraResolution.x * postFxArgs.invImageSize.x, (float)mainCameraResolution.y * postFxArgs.invImageSize.y);
    postFxArgs.frameIdx = frameIdx;
    postFxArgs.enableMotionBlurNoiseSample = enableMotionBlurNoiseSample();
    postFxArgs.enableMotionBlurEmissive = enableMotionBlurEmissive();
    postFxArgs.motionBlurSampleCount = motionBlurSampleCount();
    postFxArgs.exposureFraction = exposureFraction();
    postFxArgs.blurDiameterFraction = blurDiameterFraction();
    postFxArgs.motionBlurMinimumVelocityThresholdInPixel = motionBlurMinimumVelocityThresholdInPixel();
    postFxArgs.motionBlurDynamicDeduction = motionBlurDynamicDeduction();
    postFxArgs.jitterStrength = motionBlurJitterStrength();
    postFxArgs.motionBlurDlfgDeduction = ctx->isDLFGEnabled() ?
      1.0f / static_cast<float>(ctx->dlfgInterpolatedFrameCount() + 1) : 1.0f;

    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);

    dispatchMotionBlurInternal(
      ctx,
      nearestSampler, linearSampler,
      postFxArgs,
      workgroups,
      inputs);

    // Copy the blurred result back into the final output so downstream passes can read it.
    ctx->copyImage(
      inOutColorTexture.image,
      { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      { 0, 0, 0 },
      inputs.intermediateColor->image,
      { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      { 0, 0, 0 },
      inputSize);
  }

  void DxvkPostFx::dispatchLensEffects(
    Rc<RtxContext> ctx,
    Rc<DxvkSampler> linearSampler,
    const uvec2& mainCameraResolution,
    const uint32_t frameIdx,
    const Resources::RaytracingOutput& rtOutput)
  {
    dispatchLensEffects(ctx, linearSampler, mainCameraResolution, frameIdx,
                        rtOutput.m_finalOutput.resource(Resources::AccessType::ReadWrite),
                        rtOutput.m_postFxIntermediateTexture);
  }

  void DxvkPostFx::dispatchLensEffects(
    Rc<RtxContext> ctx,
    Rc<DxvkSampler> linearSampler,
    const uvec2& mainCameraResolution,
    const uint32_t frameIdx,
    const Resources::Resource& inOutColor,
    const Resources::Resource& intermediateColor)
  {
    if (!enable()) {
      return;
    }
    if (!isChromaticAberrationEnabled() && !isVignetteEnabled()) {
      return;
    }

    ScopedGpuProfileZone(ctx, "PostFx Lens Effects");
    ctx->setFramePassStage(RtxFramePassStage::PostFX);

    const Resources::Resource& inOutColorTexture = inOutColor;
    const VkExtent3D& inputSize = inOutColorTexture.image->info().extent;
    const VkExtent3D workgroups = util::computeBlockCount(inputSize, VkExtent3D { POST_FX_TILE_SIZE , POST_FX_TILE_SIZE, 1 } );

    PostFxArgs postFxArgs = {};
    postFxArgs.imageSize = { (uint)inputSize.width, (uint)inputSize.height };
    postFxArgs.invImageSize = { 1.0f / (float) inputSize.width, 1.0f / (float) inputSize.height };
    postFxArgs.invMainCameraResolution = float2(1.0f / (float)mainCameraResolution.x, 1.0f / (float)mainCameraResolution.y);
    postFxArgs.inputOverOutputViewSize = float2((float)mainCameraResolution.x * postFxArgs.invImageSize.x, (float)mainCameraResolution.y * postFxArgs.invImageSize.y);
    postFxArgs.frameIdx = frameIdx;
    postFxArgs.chromaticCenterAttenuationAmount = chromaticCenterAttenuationAmount();
    postFxArgs.chromaticAberrationScale = calculateChromaticAberrationScale(isChromaticAberrationEnabled() ? chromaticAberrationAmount() : 0.0f);
    postFxArgs.vignetteIntensity = isVignetteEnabled() ? vignetteIntensity() : 0.0f;
    postFxArgs.vignetteRadius = vignetteRadius();
    postFxArgs.vignetteSoftness = vignetteSoftness();

    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);

    dispatchPostLensEffects(ctx, linearSampler, postFxArgs, workgroups,
                            inOutColorTexture, intermediateColor);

    // The lens-effect shader uses a Sampler2D input and an RWTexture2D output. To keep the
    // input/output decoupled (and avoid sampling-while-writing hazards) we write into the
    // intermediate texture and copy it back into the final output.
    ctx->copyImage(
      inOutColorTexture.image,
      { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      { 0, 0, 0 },
      intermediateColor.image,
      { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      { 0, 0, 0 },
      inputSize);
  }

  namespace {
    uint32_t bitCeilPow2(uint32_t v) {
      if (v == 0) {
        return 0;
      }

      // most significant bit
      unsigned long msb = 0;
      static_assert(sizeof(unsigned long) == sizeof(uint32_t));
      if (_BitScanReverse(&msb, v) == 0) {
        assert(0);
        return 0;
      }

      // if pow of 2, return itself
      if ((v & (v - 1)) == 0) {
        return msb;
      }
      return msb + 1;
    }

    uint32_t packColor(uint8_t r, uint8_t g, uint8_t  b) {
      return (r << 0) | (g << 8) | (b << 16);
    }
  }

  void DxvkPostFx::dispatchHighlighting(
    Rc<RtxContext> ctx,
    const Resources::RaytracingOutput& rtOutput,
    std::vector<uint32_t>&& objectPickingValuesToHighlight,
    const std::optional<Vector2i>& pixelToHighlight,
    HighlightColor color) {
    static_assert(sizeof(ObjectPickingValue) == sizeof(objectPickingValuesToHighlight[0]));
    if (!rtOutput.m_primaryObjectPicking.isValid()) {
      return;
    }
    if (objectPickingValuesToHighlight.empty() && !pixelToHighlight) {
      return;
    }
    ScopedGpuProfileZone(ctx, "PostFx Highlight");

    const Resources::Resource& inOutColorTexture = rtOutput.m_compositeOutput.resource(Resources::AccessType::ReadWrite);
    const VkExtent3D& inputSize = inOutColorTexture.image->info().extent;

    const auto workgroups = util::computeBlockCount(inputSize, VkExtent3D { POST_FX_TILE_SIZE , POST_FX_TILE_SIZE, 1 });

    uint32_t valuesToHighlightCountPow;
    {
      // deduplicate and sort to perform binary search in the shader
      std::vector<uint32_t>& sorted = objectPickingValuesToHighlight;
      {
        if (sorted.size() > POST_FX_HIGHLIGHTING_MAX_VALUES) {
          sorted.resize(POST_FX_HIGHLIGHTING_MAX_VALUES);
          ONCE(Logger::warn("Too many values to highlight, some objects will be omitted."));
        }
        auto newEnd = std::unique(sorted.begin(), sorted.end());
        sorted.erase(newEnd, sorted.end());
        std::sort(sorted.begin(), sorted.end());
      }

      valuesToHighlightCountPow = bitCeilPow2(static_cast<uint32_t>(sorted.size()));

      // fill invalid values as POST_FX_HIGHLIGHTING_INVALID_VALUE
      {
        const size_t validCount = sorted.size();
        sorted.resize(1 << valuesToHighlightCountPow);
        for (size_t i = validCount; i < sorted.size(); i++) {
          sorted[i] = POST_FX_HIGHLIGHTING_INVALID_VALUE;
        }
      }

      if (m_highlightingValues == nullptr) {
        auto info = DxvkBufferCreateInfo {};
        {
          info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
          info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
          info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
          info.size = align(POST_FX_HIGHLIGHTING_MAX_VALUES * sizeof(ObjectPickingValue), kBufferAlignment);
        }
        m_highlightingValues = ctx->getDevice()->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Highlight Buffer");
      }

      if (!sorted.empty()) {
        ctx->writeToBuffer(m_highlightingValues, 0, sorted.size() * sizeof(ObjectPickingValue), sorted.data());
      }
    }

    auto args = PostFxHighlightingArgs {};
    {
      args.imageSize = { inputSize.width, inputSize.height };
      args.desaturateNonHighlighted = desaturateOthersOnHighlight() ? 1 : 0;
      args.timeSinceStartMS = (float)GlobalTime::get().absoluteTimeMs();
      args.pixel = pixelToHighlight ? int2 { pixelToHighlight->x, pixelToHighlight->y } : int2 { -1, -1 };
      args.highlightColorPacked =
        color == HighlightColor::World ? packColor(118, 185, 0) :
        color == HighlightColor::UI ? packColor(66, 150, 250) :
        color == HighlightColor::FromVariable ? packColor(g_customHighlightColor[0], g_customHighlightColor[1], g_customHighlightColor[2]) :
        packColor(255, 255, 255);
      args.valuesToHighlightCountPow = valuesToHighlightCountPow;
    }

    ctx->pushConstants(0, sizeof(args), &args);

    const Resources::Resource* lastOutput = &rtOutput.m_postFxIntermediateTexture;

    ctx->bindResourceView(POST_FX_HIGHLIGHT_INPUT, inOutColorTexture.view, nullptr);
    ctx->bindResourceView(POST_FX_HIGHLIGHT_OBJECT_PICKING_INPUT, rtOutput.m_primaryObjectPicking.view, nullptr);
    ctx->bindResourceView(POST_FX_HIGHLIGHT_PRIMARY_CONE_RADIUS_INPUT, rtOutput.m_primaryConeRadius.view, nullptr);
    ctx->bindResourceView(POST_FX_HIGHLIGHT_OUTPUT, lastOutput->view, nullptr);
    ctx->bindResourceBuffer(POST_FX_HIGHLIGHT_VALUES, DxvkBufferSlice(m_highlightingValues, 0, m_highlightingValues->info().size));

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, PostFxHighlightShader::getShader());
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);

    // Copy to the output texture if the final output is not the input texture
    if (lastOutput->image != inOutColorTexture.image) {
      ctx->copyImage(
        inOutColorTexture.image,
        { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        { 0, 0, 0 },
        rtOutput.m_postFxIntermediateTexture.image,
        { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        { 0, 0, 0 },
        inputSize);
    }
  }
}
