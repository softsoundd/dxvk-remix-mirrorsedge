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
#include <algorithm>
#include <cmath>

#include "rtx_sun_probe.h"
#include "rtx_context.h"
#include "rtx_atmosphere.h"
#include "rtx_camera.h"
#include "rtx_imgui.h"
#include "rtx_options.h"
#include "rtx_utils.h"
#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_render/rtx_shader_manager.h"
#include "rtx/pass/sun_probe/sun_probe.h"

#include <rtx_shaders/sun_probe.h>
#include <rtx_shaders/sun_visibility.h>

namespace dxvk {

  // Defined within an unnamed namespace to ensure unique definition across binary
  namespace {
    class SunProbeShader : public ManagedShader {
      SHADER_SOURCE(SunProbeShader, VK_SHADER_STAGE_COMPUTE_BIT, sun_probe)

      PUSH_CONSTANTS(SunProbeArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(SUN_PROBE_COLOR_INPUT)
        RW_TEXTURE2D(SUN_PROBE_VISIBILITY_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(SunProbeShader);

    class SunVisibilityShader : public ManagedShader {
      SHADER_SOURCE(SunVisibilityShader, VK_SHADER_STAGE_COMPUTE_BIT, sun_visibility)

      BINDLESS_ENABLED()

      PUSH_CONSTANTS(SunVisibilityArgs)

      BEGIN_PARAMETER()
        COMMON_RAYTRACING_BINDINGS
        RW_TEXTURE2D(SUN_VISIBILITY_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(SunVisibilityShader);

    // The radiance the atmosphere clamps the rendered disc to, kMaxSunDiscRadiance in atmosphere_common.slangh.
    constexpr float kMaxSunDiscRadiance = 16384.0f;

    // Radial samples of the disc's clamped radiance integral.
    constexpr uint32_t kDiscIntegrationSamples = 32;

    // The visibility rays start this far from the camera, in metres, clear of its near plane, and reach this far.
    constexpr float kRayStartMeters = 0.02f;
    constexpr float kRayLengthMeters = 20000.0f;

    float maxComponent(const Vector3& v) {
      return std::max(v.x, std::max(v.y, v.z));
    }

    // The share of [center - radius, center + radius] inside [0, extent].
    float overlap(float center, float radius, float extent) {
      if (radius <= 0.0f) {
        return center >= 0.0f && center <= extent ? 1.0f : 0.0f;
      }

      const float low = std::max(center - radius, 0.0f);
      const float high = std::min(center + radius, extent);
      return std::clamp((high - low) / (2.0f * radius), 0.0f, 1.0f);
    }

    RemixGui::ComboWithKey<SunVisibilitySource> sourceCombo {
      "Sun Visibility##sunProbe",
      RemixGui::ComboWithKey<SunVisibilitySource>::ComboEntries { {
        { SunVisibilitySource::RayTraced, "Ray Traced" },
        { SunVisibilitySource::Image, "Image (debug)" },
      } }
    };
  }

  RtxSunProbe::RtxSunProbe(DxvkDevice* device)
    : CommonDeviceObject(device) {
  }

  Vector3 RtxSunProbe::getRestoredIlluminance() const {
    if (!m_state.active) {
      return Vector3(0.0f);
    }

    const Vector3 rendered = m_state.renderedIlluminance * m_state.onScreenFraction;

    return Vector3(
      std::max(m_state.illuminance.x - rendered.x, 0.0f),
      std::max(m_state.illuminance.y - rendered.y, 0.0f),
      std::max(m_state.illuminance.z - rendered.z, 0.0f));
  }

  void RtxSunProbe::updateState(RtxContext& ctx, const Resources::RaytracingOutput& rtOutput) {
    const bool wasMeasured = m_state.measured;
    m_state = State {};

    const AtmosphereArgs& args = rtOutput.m_raytraceArgs.atmosphereArgs;

    if (RtxOptions::skyMode() != SkyMode::PhysicalAtmosphere || !args.sunDiscEnabled) {
      return;
    }

    Vector3 illuminance;
    Vector3 sunDirection;
    RtxAtmosphere::estimateUnoccludedVolumeLighting(args, RtxOptions::zUp(), illuminance, sunDirection);

    if (maxComponent(illuminance) <= 0.0f) {
      return;
    }

    const RtCamera& camera = ctx.getSceneManager().getCamera();
    const Matrix4d worldToProjection = camera.getViewToProjection() * camera.getWorldToView();
    const Vector4d clip = worldToProjection * Vector4d(sunDirection.x, sunDirection.y, sunDirection.z, 0.0);

    // Points at infinity behind the camera project with a negative w.
    if (clip.w <= 0.0) {
      return;
    }

    const VkExtent3D extent = rtOutput.m_finalOutput.resource(Resources::AccessType::Read).image->info().extent;
    const float width = float(extent.width);
    const float height = float(extent.height);
    const Vector2 ndc(float(clip.x / clip.w), float(clip.y / clip.w));

    m_sunDirection = sunDirection;
    m_state.tanHalfFovY = std::tan(0.5f * camera.getFov());
    m_state.aspectRatio = camera.getAspectRatio();
    m_state.pixel = Vector2((ndc.x * 0.5f + 0.5f) * width, (0.5f - ndc.y * 0.5f) * height);
    m_state.ndcAspect = Vector2(ndc.x * m_state.aspectRatio, ndc.y);

    // A rectilinear projection stretches the disc by 1 / cos across the view and 1 / cos^2 along it. The image probe
    // samples a circle of the narrower radius, and keeps its annulus clear of the wider one.
    const float cosOffAxis = std::clamp(dot(camera.getDirection(), sunDirection), 1e-3f, 1.0f);
    const float pixelsPerTangent = 0.5f * height / std::max(m_state.tanHalfFovY, 1e-6f);
    m_state.cosOffAxis = cosOffAxis;
    m_state.angularRadius = args.sunAngularRadius;
    m_state.discRadiusPixels = std::tan(args.sunAngularRadius) * pixelsPerTangent / cosOffAxis;
    m_annulusInnerRadiusPixels = 1.3f * m_state.discRadiusPixels / cosOffAxis;

    const float margin = m_state.discRadiusPixels;
    m_state.onScreen =
      m_state.pixel.x > -margin && m_state.pixel.x < width + margin &&
      m_state.pixel.y > -margin && m_state.pixel.y < height + margin;
    m_state.onScreenFraction = m_state.onScreen
      ? overlap(m_state.pixel.x, m_state.discRadiusPixels, width) * overlap(m_state.pixel.y, m_state.discRadiusPixels, height)
      : 0.0f;

    // The illuminance the rendered disc carries: evalSunDisk's limb darkened radiance, clamped, over the disc.
    const float sinRadius = std::max(std::sin(args.sunAngularRadius), 1e-6f);
    const float discSolidAngle = kPi * sinRadius * sinRadius;
    const float limb[3] = { args.sunLimbDarkeningExponent.x, args.sunLimbDarkeningExponent.y, args.sunLimbDarkeningExponent.z };
    const float total[3] = { illuminance.x, illuminance.y, illuminance.z };
    float rendered[3] = {};

    for (uint32_t c = 0; c < 3; c++) {
      const float alpha = std::max(limb[c], 0.0f);
      const float centerRadiance = total[c] / discSolidAngle * (alpha + 2.0f) * 0.5f;
      float meanRadiance = 0.0f;

      for (uint32_t k = 0; k < kDiscIntegrationSamples; k++) {
        const float rho = (float(k) + 0.5f) / float(kDiscIntegrationSamples);
        const float mu = std::sqrt(std::max(1.0f - rho * rho, 1e-8f));
        const float areaWeight = 2.0f * rho / float(kDiscIntegrationSamples);

        meanRadiance += areaWeight * std::min(centerRadiance * std::pow(mu, alpha), kMaxSunDiscRadiance);
      }

      rendered[c] = std::min(meanRadiance * discSolidAngle, total[c]);
    }

    m_state.illuminance = illuminance;
    m_state.renderedIlluminance = Vector3(rendered[0], rendered[1], rendered[2]);
    m_state.limbDarkeningExponent = Vector3(std::max(limb[0], 0.0f), std::max(limb[1], 0.0f), std::max(limb[2], 0.0f));
    m_state.active = true;
    m_state.measured = wasMeasured;
  }

  void RtxSunProbe::dispatch(RtxContext& ctx, const Resources::RaytracingOutput& rtOutput, bool measure) {
    ScopedCpuProfileZone();

    updateState(ctx, rtOutput);

    // A measurement goes stale while the sun is gone or unwatched.
    if (!m_state.active || !measure) {
      m_state.measured = false;
      return;
    }

    const SunVisibilitySource visibilitySource = source();

    // The image shows the disc only while it is on the screen. Off it, the last measurement holds.
    if (visibilitySource == SunVisibilitySource::Image && !m_state.onScreen) {
      return;
    }

    ScopedGpuProfileZone(&ctx, "Sun Visibility");
    ctx.setFramePassStage(RtxFramePassStage::Bloom);

    if (!m_visibility.isValid()) {
      Rc<DxvkContext> baseCtx = &ctx;
      m_visibility = Resources::createImageResource(baseCtx, "sun visibility", { 1, 1, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT);
    }

    if (visibilitySource == SunVisibilitySource::RayTraced) {
      dispatchRayTraced(ctx, rtOutput);
    } else {
      dispatchImage(ctx, rtOutput);
    }

    m_state.measured = true;
  }

  void RtxSunProbe::dispatchRayTraced(RtxContext& ctx, const Resources::RaytracingOutput& rtOutput) {
    const RtCamera& camera = ctx.getSceneManager().getCamera();
    const Vector3 direction = normalize(m_sunDirection);
    const Vector3 helper = std::abs(direction.y) < 0.9f ? Vector3(0.0f, 1.0f, 0.0f) : Vector3(1.0f, 0.0f, 0.0f);
    const Vector3 right = normalize(cross(helper, direction));
    const Vector3 up = cross(direction, right);
    const float tanRadius = std::tan(m_state.angularRadius);
    const float metersToWorld = RtxOptions::getMeterToWorldUnitScale();

    SunVisibilityArgs args = {};
    args.origin = camera.getPosition();
    args.tMin = kRayStartMeters * metersToWorld;
    args.sunDirection = direction;
    args.tMax = kRayLengthMeters * metersToWorld;
    args.discRight = right * tanRadius;
    args.discUp = up * tanRadius;
    args.limbDarkeningExponent = m_state.limbDarkeningExponent;
    args.flags = SUN_VISIBILITY_FLAG_CLOUDS | SUN_VISIBILITY_FLAG_FOG;

    ctx.bindCommonRayTracingResources(rtOutput);
    ctx.setPushConstantBank(DxvkPushConstantBank::RTX);
    ctx.pushConstants(0, sizeof(args), &args);
    ctx.bindResourceView(SUN_VISIBILITY_OUTPUT, m_visibility.view, nullptr);
    ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, SunVisibilityShader::getShader());
    ctx.dispatch(1, 1, 1);
  }

  void RtxSunProbe::dispatchImage(RtxContext& ctx, const Resources::RaytracingOutput& rtOutput) {
    const AtmosphereArgs& atmosphereArgs = rtOutput.m_raytraceArgs.atmosphereArgs;
    const Resources::Resource& color = rtOutput.m_finalOutput.resource(Resources::AccessType::Read);
    const VkExtent3D extent = color.image->info().extent;

    const float sinRadius = std::max(std::sin(atmosphereArgs.sunAngularRadius), 1e-6f);
    const float discSolidAngle = kPi * sinRadius * sinRadius;
    const Vector3 limb = m_state.limbDarkeningExponent;

    SunProbeArgs args = {};
    args.sunPixel = m_state.pixel;
    args.discRadiusPixels = m_state.discRadiusPixels;
    args.annulusInnerRadiusPixels = m_annulusInnerRadiusPixels;
    args.centerRadiance = Vector3(
      m_state.illuminance.x / discSolidAngle * (limb.x + 2.0f) * 0.5f,
      m_state.illuminance.y / discSolidAngle * (limb.y + 2.0f) * 0.5f,
      m_state.illuminance.z / discSolidAngle * (limb.z + 2.0f) * 0.5f);
    args.maxDiscRadiance = kMaxSunDiscRadiance;
    args.limbDarkeningExponent = limb;
    args.imageSize = { extent.width, extent.height };

    ctx.setPushConstantBank(DxvkPushConstantBank::RTX);
    ctx.pushConstants(0, sizeof(args), &args);
    ctx.bindResourceView(SUN_PROBE_COLOR_INPUT, color.view, nullptr);
    ctx.bindResourceView(SUN_PROBE_VISIBILITY_OUTPUT, m_visibility.view, nullptr);
    ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, SunProbeShader::getShader());
    ctx.dispatch(1, 1, 1);
  }

  void RtxSunProbe::showImguiSettings() {
    ImGui::Indent();
    sourceCombo.getKey(&sourceObject());
    RemixGui::SetTooltipToLastWidgetOnHover("Ray Traced: rays from the camera to the sun's disc through the scene, clouds and fog, on or off\nthe screen. Image: the disc's pixels in the rendered image, for debugging.");

    if (m_state.active) {
      ImGui::Text("Sun %s, %.0f%% of its disc on the screen", m_state.onScreen ? "on screen" : "off screen", m_state.onScreenFraction * 100.0f);
    }

    ImGui::Unindent();
  }

}
