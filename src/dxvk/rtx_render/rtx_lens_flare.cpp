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
#include <limits>

#include "rtx_lens_flare.h"
#include "rtx_lens_aperture.h"
#include "rtx_lens_prescriptions.h"
#include "rtx_sun_probe.h"
#include "rtx_context.h"
#include "rtx_imgui.h"
#include "rtx_options.h"
#include "dxvk_device.h"
#include "dxvk_context_state.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_render/rtx_shader_manager.h"
#include "rtx/pass/lens_flare/lens_flare.h"
#include "../../util/util_color.h"

#include <rtx_shaders/lens_flare.h>
#include <rtx_shaders/lens_flare_search.h>
#include <rtx_shaders/lens_flare_trace.h>
#include <rtx_shaders/lens_flare_irradiance.h>
#include <rtx_shaders/lens_flare_extend.h>
#include <rtx_shaders/lens_flare_extend_outer.h>
#include <rtx_shaders/lens_flare_ghost_vertex.h>
#include <rtx_shaders/lens_flare_ghost_fragment.h>
#include <rtx_shaders/lens_flare_composite.h>

namespace dxvk {

  // Defined within an unnamed namespace to ensure unique definition across binary
  namespace {
    class LensFlareShader : public ManagedShader {
      SHADER_SOURCE(LensFlareShader, VK_SHADER_STAGE_COMPUTE_BIT, lens_flare)

      PUSH_CONSTANTS(LensFlareArgs)

      BEGIN_PARAMETER()
        STRUCTURED_BUFFER(LENS_FLARE_GHOSTS_INPUT)
        TEXTURE2D(LENS_FLARE_SUN_VISIBILITY_INPUT)
        RW_TEXTURE2D(LENS_FLARE_COLOR_INPUT_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(LensFlareShader);

#define LENS_FLARE_TRACE_SHADER(className, shaderName)                           \
    class className : public ManagedShader {                                     \
      SHADER_SOURCE(className, VK_SHADER_STAGE_COMPUTE_BIT, shaderName)          \
      PUSH_CONSTANTS(LensFlareTraceArgs)                                         \
      BEGIN_PARAMETER()                                                          \
        STRUCTURED_BUFFER(LENS_FLARE_TRACE_SURFACES_INPUT)                       \
        STRUCTURED_BUFFER(LENS_FLARE_TRACE_INSTANCES_INPUT)                      \
        RW_STRUCTURED_BUFFER(LENS_FLARE_TRACE_DOMAINS_INPUT_OUTPUT)              \
        RW_STRUCTURED_BUFFER(LENS_FLARE_TRACE_VERTICES_INPUT_OUTPUT)             \
      END_PARAMETER()                                                            \
    };                                                                           \
    PREWARM_SHADER_PIPELINE(className);

    LENS_FLARE_TRACE_SHADER(LensFlareSearchShader, lens_flare_search)
    LENS_FLARE_TRACE_SHADER(LensFlareTraceShader, lens_flare_trace)
    LENS_FLARE_TRACE_SHADER(LensFlareIrradianceShader, lens_flare_irradiance)
    LENS_FLARE_TRACE_SHADER(LensFlareExtendShader, lens_flare_extend)
    LENS_FLARE_TRACE_SHADER(LensFlareExtendOuterShader, lens_flare_extend_outer)

#undef LENS_FLARE_TRACE_SHADER

    class LensFlareGhostVertexShader : public ManagedShader {
      SHADER_SOURCE(LensFlareGhostVertexShader, VK_SHADER_STAGE_VERTEX_BIT, lens_flare_ghost_vertex)

      PUSH_CONSTANTS(LensFlareRasterArgs)

      BEGIN_PARAMETER()
        STRUCTURED_BUFFER(LENS_FLARE_RASTER_VERTICES_INPUT)
        STRUCTURED_BUFFER(LENS_FLARE_RASTER_INSTANCES_INPUT)
        TEXTURE2D(LENS_FLARE_RASTER_SUN_VISIBILITY_INPUT)
      END_PARAMETER()

      // The stop coordinate, clip ratio, radiance, their changes across the sun's disc, the carrier's whole spectrum,
      // the loss margin, the triangle's corners and whether it is fully lit take eleven varyings.
      INTERFACE_OUTPUT_SLOTS(0x7FF);
    };

    class LensFlareGhostFragmentShader : public ManagedShader {
      SHADER_SOURCE(LensFlareGhostFragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT, lens_flare_ghost_fragment)

      PUSH_CONSTANTS(LensFlareRasterArgs)

      BEGIN_PARAMETER()
      END_PARAMETER()

      INTERFACE_INPUT_SLOTS(0x7FF);
      INTERFACE_OUTPUT_SLOTS(1);
    };

    class LensFlareCompositeShader : public ManagedShader {
      SHADER_SOURCE(LensFlareCompositeShader, VK_SHADER_STAGE_COMPUTE_BIT, lens_flare_composite)

      PUSH_CONSTANTS(LensFlareCompositeArgs)

      BEGIN_PARAMETER()
        SAMPLER2D(LENS_FLARE_COMPOSITE_HALF_INPUT)
        SAMPLER2D(LENS_FLARE_COMPOSITE_QUARTER_INPUT)
        RW_TEXTURE2D(LENS_FLARE_COMPOSITE_COLOR_INPUT_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(LensFlareCompositeShader);

    constexpr double kPiDouble = 3.14159265358979323846;

    // Ghosts focused near the sensor would be points of unbounded radiance, so they are spread over at least this
    // fraction of the screen's half height, keeping their energy.
    constexpr double kMinGhostRadius = 0.005;

    // Ghosts are skipped below this luminance, in radiance per unit of sun illuminance, and fade in over the next few
    // times it so they do not pop. A clear sky's radiance is around a tenth of the sun's illuminance per steradian, so
    // they would be a thousandth of it.
    constexpr float kMinGhostRadiance = 1e-4f;
    constexpr float kGhostFadeRange = 4.0f;

    // Below this a ghost's colour fringes are too faint to see, so one wavelength carries its whole spectrum.
    constexpr float kMinSpectralGhostRadiance = 1e-3f;

    // The search for each ghost's rays spans this many times the stop's paraxial image along the sun's meridian on the
    // entrance plane, and at least this share of the front element, for the pupil's aberrations far from the axis.
    constexpr double kSearchMargin = 4.0;
    constexpr double kMinSearchShare = 0.25;

    // A ghost drawn across the spectrum goes back to one wavelength only once it is this far inside the thresholds.
    constexpr double kSpectralHysteresis = 0.7;

    // A ghost drawn across the spectrum is split at a band along its edges, within which each wavelength draws its own
    // light: twice its colour fringe and half its blur wide and a few pixels more, so that the band holds every
    // wavelength's edge however the real edges' dispersion strays from the paraxial circles'. Past it the middle
    // wavelength takes over the whole spectrum, easing in over a few fringe widths so that the move from each
    // wavelength's own image to the carrier's stays smooth. A ghost is split only where its passing region is this
    // many times as deep as the band and the hand-off, so that an interior is left to save.
    constexpr double kBandFringes = 2.0;
    constexpr float kBandMarginPixels = 3.0f;
    constexpr double kBandTransitionFringes = 4.0;
    constexpr float kMinBandTransitionPixels = 8.0f;
    constexpr double kSplitBands = 2.0;

    // A ray traced ghost is drawn at half or quarter resolution when the sun's disc blurs its sharpest edge over this
    // many of that resolution's pixels and its edges lie this many of them from its middle. Drawn there and upsampled
    // bilinearly, the edges' profiles then stray from the full resolution ones by a few percent of the ghost's level.
    constexpr double kReducedBlurPixels = 4.0;
    constexpr double kReducedRadiusPixels = 8.0;

    RemixGui::ComboWithKey<LensFlareQuality> qualityCombo {
      "Quality##lensFlare",
      RemixGui::ComboWithKey<LensFlareQuality>::ComboEntries { {
        { LensFlareQuality::Fast, "Fast" },
        { LensFlareQuality::RayTraced, "Ray Traced" },
      } }
    };

    RemixGui::ComboWithKey<LensFlareDebugView> debugViewCombo {
      "Debug View##lensFlare",
      RemixGui::ComboWithKey<LensFlareDebugView>::ComboEntries { {
        { LensFlareDebugView::None, "None" },
        { LensFlareDebugView::FlareOnly, "Flare Only" },
        { LensFlareDebugView::GhostBounds, "Ghost Bounds" },
      } }
    };

    void setupGhostRasterState(RtxContext& ctx, const Resources::Resource& output) {
      DxvkInputAssemblyState iaState;
      iaState.primitiveTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      iaState.primitiveRestart = VK_FALSE;
      iaState.patchVertexCount = 0;
      ctx.setInputAssemblyState(iaState);

      DxvkRasterizerState rsState;
      rsState.polygonMode = VK_POLYGON_MODE_FILL;
      rsState.cullMode = VK_CULL_MODE_NONE;
      rsState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
      rsState.depthClipEnable = VK_FALSE;
      rsState.depthBiasEnable = VK_FALSE;
      rsState.conservativeMode = VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT;
      rsState.sampleCount = VK_SAMPLE_COUNT_1_BIT;
      ctx.setRasterizerState(rsState);

      DxvkMultisampleState msState;
      msState.sampleMask = 0xffffffff;
      msState.enableAlphaToCoverage = VK_FALSE;
      ctx.setMultisampleState(msState);

      VkStencilOpState stencilOp;
      stencilOp.failOp = VK_STENCIL_OP_KEEP;
      stencilOp.passOp = VK_STENCIL_OP_KEEP;
      stencilOp.depthFailOp = VK_STENCIL_OP_KEEP;
      stencilOp.compareOp = VK_COMPARE_OP_ALWAYS;
      stencilOp.compareMask = 0xFFFFFFFF;
      stencilOp.writeMask = 0xFFFFFFFF;
      stencilOp.reference = 0;

      DxvkDepthStencilState dsState;
      dsState.enableDepthTest = VK_FALSE;
      dsState.enableDepthWrite = VK_FALSE;
      dsState.enableStencilTest = VK_FALSE;
      dsState.depthCompareOp = VK_COMPARE_OP_ALWAYS;
      dsState.stencilOpFront = stencilOp;
      dsState.stencilOpBack = stencilOp;
      ctx.setDepthStencilState(dsState);

      DxvkLogicOpState loState;
      loState.enableLogicOp = VK_FALSE;
      loState.logicOp = VK_LOGIC_OP_NO_OP;
      ctx.setLogicOpState(loState);

      // The ghosts add their light, leaving alpha as it was.
      DxvkBlendMode blendMode;
      blendMode.enableBlending = VK_TRUE;
      blendMode.colorSrcFactor = VK_BLEND_FACTOR_ONE;
      blendMode.colorDstFactor = VK_BLEND_FACTOR_ONE;
      blendMode.colorBlendOp = VK_BLEND_OP_ADD;
      blendMode.alphaSrcFactor = VK_BLEND_FACTOR_ZERO;
      blendMode.alphaDstFactor = VK_BLEND_FACTOR_ONE;
      blendMode.alphaBlendOp = VK_BLEND_OP_ADD;
      blendMode.writeMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      ctx.setBlendMode(0, blendMode);

      VkViewport viewport;
      viewport.x = 0.0f;
      viewport.y = 0.0f;
      viewport.width = float(output.image->info().extent.width);
      viewport.height = float(output.image->info().extent.height);
      viewport.minDepth = 0.0f;
      viewport.maxDepth = 1.0f;

      VkRect2D scissor;
      scissor.offset = { 0, 0 };
      scissor.extent = { output.image->info().extent.width, output.image->info().extent.height };
      ctx.setViewports(1, &viewport, &scissor);

      DxvkRenderTargets renderTargets;
      renderTargets.color[0].view = output.view;
      renderTargets.color[0].layout = VK_IMAGE_LAYOUT_GENERAL;
      ctx.bindRenderTargets(renderTargets);

      ctx.setInputLayout(0, nullptr, 0, nullptr);

      ctx.bindShader(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, nullptr);
      ctx.bindShader(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, nullptr);
      ctx.bindShader(VK_SHADER_STAGE_GEOMETRY_BIT, nullptr);
    }

    template<typename T>
    void ensureBuffer(DxvkDevice* device, Rc<DxvkBuffer>& buffer, size_t count, VkBufferUsageFlags usage,
                      VkPipelineStageFlags stages, VkAccessFlags access, const char* name) {
      const VkDeviceSize size = std::max<VkDeviceSize>(sizeof(T) * count, sizeof(T));

      if (buffer != nullptr && buffer->info().size >= size) {
        return;
      }

      DxvkBufferCreateInfo info = {};
      info.usage = usage;
      info.stages = stages;
      info.access = access;
      info.size = size;
      buffer = device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, name);
    }

    void setComponent(vec4& v, uint32_t index, float value) {
      switch (index) {
      case 0: v.x = value; break;
      case 1: v.y = value; break;
      case 2: v.z = value; break;
      default: v.w = value; break;
      }
    }

    void uploadChunked(RtxContext& ctx, const Rc<DxvkBuffer>& buffer, const void* data, size_t size) {
      // vkCmdUpdateBuffer takes at most 64 KB at a time.
      constexpr size_t kChunk = 65536;
      const uint8_t* bytes = static_cast<const uint8_t*>(data);

      for (size_t offset = 0; offset < size; offset += kChunk) {
        ctx.updateBuffer(buffer, offset, std::min(kChunk, size - offset), bytes + offset);
      }
    }
  }

  RtxLensFlare::RtxLensFlare(DxvkDevice* device)
    : CommonDeviceObject(device) {
  }

  bool RtxLensFlare::isEnabled() {
    return enable() && intensity() > 0.0f && RtxOptions::skyMode() == SkyMode::PhysicalAtmosphere && !LensAperture::isEye();
  }

  void RtxLensFlare::selectGhosts(double minSensorFromPupil) {
    std::vector<LensSystem::Ghost> candidates = m_lensSystem.traceGhosts(minSensorFromPupil);
    const size_t kept = std::min<size_t>(candidates.size(), size_t(std::clamp(maxGhosts(), 1, LENS_FLARE_MAX_GHOSTS)));

    m_candidateCount = static_cast<uint32_t>(candidates.size());
    m_ghosts.assign(candidates.begin(), candidates.begin() + kept);
    m_spectralGhosts.assign(m_ghosts.size(), 0);
  }

  void RtxLensFlare::updateSurfaceBuffer(RtxContext& ctx) {
    if (m_surfaceVersion == m_lensVersion && m_surfaceBuffer != nullptr) {
      return;
    }

    const LensPrescription& prescription = kLensPrescriptions[m_lensSystem.getPrescription()];
    const uint32_t count = std::min(m_lensSystem.getSurfaceCount(), uint32_t(LENS_FLARE_MAX_SURFACES));
    std::vector<LensFlareSurface> surfaces(count);

    for (uint32_t k = 0; k < count; k++) {
      LensFlareSurface& surface = surfaces[k];
      surface.z = float(m_lensSystem.getVertexZ(k));
      surface.radius = float(m_lensSystem.getRadius(k));
      surface.clearRadius = float(m_lensSystem.getClearRadius(k));

      // Only the air to glass surfaces are coated.
      const double before = m_lensSystem.getIndexBefore(k, kFocus);
      const double after = m_lensSystem.getIndexAfter(k, kFocus);
      surface.coatedGlassIndex = std::min(before, after) <= 1.0 && std::max(before, after) > 1.0
        ? float(std::max(prescription.surfaces[k].indexD, k > 0 ? prescription.surfaces[k - 1].indexD : 1.0))
        : 0.0f;

      for (uint32_t w = 0; w < kWavelengthCount; w++) {
        setComponent(surface.indexAfter[w / 4], w % 4, float(m_lensSystem.getIndexAfter(k, w)));
      }
    }

    ensureBuffer<LensFlareSurface>(m_device, m_surfaceBuffer, LENS_FLARE_MAX_SURFACES,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, "Lens flare surfaces");
    ctx.updateBuffer(m_surfaceBuffer, 0, sizeof(LensFlareSurface) * surfaces.size(), surfaces.data());
    m_surfaceVersion = m_lensVersion;
  }

  void RtxLensFlare::dispatch(RtxContext& ctx, DxvkContextState& state, const Resources::RaytracingOutput& rtOutput, const RtxSunProbe& sunProbe) {
    ScopedCpuProfileZone();

    const RtxSunProbe::State& sun = sunProbe.getState();

    if (!isEnabled() || !sun.active || !sun.measured || sunProbe.getVisibilityView() == nullptr) {
      m_drawnGhostCount = 0;
      m_levelGhostCounts.fill(0);
      return;
    }

    if (LensAperture::updateLensSystem(m_lensSystem)) {
      m_lensVersion++;
    }

    // The stop opens to the requested f-number up to its clear aperture.
    const double focalLength = m_lensSystem.getFocalLength();
    const double requestedPupilRadius = focalLength / (2.0 * std::max(double(LensAperture::fNumber()), 0.5));
    m_stopRadius = std::min(requestedPupilRadius * m_lensSystem.getStopFromPupil(), m_lensSystem.getMaxStopRadius());
    m_entrancePupilRadius = std::min(m_stopRadius / m_lensSystem.getStopFromPupil(), m_lensSystem.getFrontRadius());

    // The sensor's half height in mm: the full frame cropped further to the game's field of view while the lens covers
    // it, so the sun's rays enter at their true angle, and otherwise the whole full frame crop.
    const double sensorScale = LensAperture::getImageHalfHeightMm(sun.aspectRatio, float(focalLength), sun.tanHalfFovY);
    const double minSensorFromPupil = kMinGhostRadius * sensorScale / m_entrancePupilRadius;

    // The f-number sets the iris's area, so its corners lie further out than the stop radius.
    const float curvature = LensAperture::getCurvature(float(focalLength / (2.0 * m_entrancePupilRadius)));
    const double irisCornerRadius = m_stopRadius / double(LensAperture::getShape(curvature).areaRadius);
    float apertureDistanceJitter;
    float apertureAngleJitter;
    LensAperture::getJitter(m_stopRadius, m_lensSystem.getMaxStopRadius(), apertureDistanceJitter, apertureAngleJitter);

    // The ghosts are ranked on the whole full frame crop rather than the view, which the game widens with the player's
    // speed, so that the chosen ghosts do not change from frame to frame.
    const double selectionMinSensorFromPupil =
      kMinGhostRadius * double(LensAperture::getSensorHalfHeightMm(sun.aspectRatio)) / m_entrancePupilRadius;

    if (m_selectionVersion != m_lensVersion || m_selectionMinSensorFromPupil != float(selectionMinSensorFromPupil) ||
        m_selectionMaxGhosts != maxGhosts()) {
      selectGhosts(selectionMinSensorFromPupil);
      m_selectionVersion = m_lensVersion;
      m_selectionMinSensorFromPupil = float(selectionMinSensorFromPupil);
      m_selectionMaxGhosts = maxGhosts();
    }

    const Resources::Resource& color = rtOutput.m_finalOutput.resource(Resources::AccessType::ReadWrite);
    const VkExtent3D extent = color.image->info().extent;
    const bool traced = quality() == LensFlareQuality::RayTraced;
    const uint32_t gridSize = uint32_t(std::clamp(traceResolution(), 8, 128)) + 1;
    const auto& wavelengthWeights = m_lensSystem.getWavelengthWeights();

    // The sensor's image is inverted on the screen, so the rays of a sun up and to the right come in heading down and
    // to the left.
    const double rayAngleX = -double(sun.ndcAspect.x) * sensorScale / focalLength;
    const double rayAngleY = -double(sun.ndcAspect.y) * sensorScale / focalLength;
    const double rayAngle = std::sqrt(rayAngleX * rayAngleX + rayAngleY * rayAngleY);
    // The direction the rays tilt in, the sun's meridian, in which the ghosts' rays are traced and fitted.
    const Vector2 meridian = rayAngle > 0.0 ? Vector2(float(rayAngleX / rayAngle), float(rayAngleY / rayAngle)) : Vector2(1.0f, 0.0f);
    const double pupilArea = kPiDouble * m_entrancePupilRadius * m_entrancePupilRadius;
    const double pixelMm = 2.0 * sensorScale / double(std::max(extent.height, 1u));
    constexpr uint32_t kMiddle = LENS_FLARE_WAVELENGTHS / 2;
    // The sun's diameter in the lens's angles, which shrink with the screen's when the view is wider than the lens's.
    const double sunDiameter = 2.0 * double(sun.angularRadius) * (sensorScale / focalLength) / std::max(double(sun.tanHalfFovY), 1e-6);

    // A circle on the sensor in mm, or in aspect corrected NDC.
    struct Circle {
      double x;
      double y;
      double radius;
    };

    // A clear aperture's paraxial image on the entrance plane: a disc whose centre moves along the sun's meridian with
    // the sun's angle, in mm.
    struct EntranceDisc {
      double centrePerSlope;
      double radius;
    };

    std::vector<LensFlareGhost> fastGhosts;
    std::array<std::vector<LensFlareTraceInstance>, kRasterLevels> levelInstances;
    std::vector<EntranceDisc> discs;
    std::vector<std::pair<double, size_t>> cuts;
    m_drawnGhostCount = 0;
    m_levelGhostCounts.fill(0);

    for (const LensSystem::Ghost& ghost : m_ghosts) {
      LensFlareGhost out = {};
      Vector3 total(0.0f);
      std::array<double, LENS_FLARE_WAVELENGTHS> reflectances {};
      // The middle wavelength's central ray, the half width of what passes about it along the meridian, and its paraxial
      // magnification onto the sensor.
      double referenceHeight = 0.0;
      double referenceHalfWidth = 0.0;
      double middleSensorFromPupil = 1.0;
      // The paraxial images of the iris and of the circle the path's clear apertures pass on the axis, which estimate the
      // ghost's colour fringe and its edges' blur.
      std::array<Circle, LENS_FLARE_WAVELENGTHS> irisCircles;
      std::array<Circle, LENS_FLARE_WAVELENGTHS> pupilCircles;
      double angularMagnification = 0.0;
      double sharpestMagnification = 0.0;

      // The discs of the clear apertures on the ghost's path. An aperture every ray meets at the same height has none,
      // and passes the sun only up to an angle.
      discs.clear();
      bool slopesPass = true;

      for (const LensSystem::Aperture& aperture : ghost.apertures) {
        if (std::abs(aperture.a) >= 1e-9) {
          discs.push_back({ -aperture.b / aperture.a, aperture.radius / std::abs(aperture.a) });
        } else {
          slopesPass = slopesPass && std::abs(aperture.b) * rayAngle <= aperture.radius;
        }
      }

      for (uint32_t w = 0; w < LENS_FLARE_WAVELENGTHS; w++) {
        const RayTransferMatrix& toSensor = ghost.toSensor[w];
        const RayTransferMatrix& toStop = ghost.toStop[w];

        const double sensorFromPupil = std::abs(toSensor.a) < minSensorFromPupil
          ? std::copysign(minSensorFromPupil, toSensor.a)
          : toSensor.a;

        // The coatings are evaluated along a ray through the middle of what the ghost passes: where the stop's image
        // on the entrance plane overlaps every clear aperture's, in the plane of the sun's rays. The ray through the
        // stop's centre can lie far off the lens, where it would meet the surfaces at angles no passing ray does, past
        // the critical angle.
        const double stopCenter = -toStop.b * rayAngle / toStop.a;
        const double stopImage = irisCornerRadius / std::abs(toStop.a);
        double passLow = stopCenter - stopImage;
        double passHigh = stopCenter + stopImage;

        for (const EntranceDisc& disc : discs) {
          passLow = std::max(passLow, disc.centrePerSlope * rayAngle - disc.radius);
          passHigh = std::min(passHigh, disc.centrePerSlope * rayAngle + disc.radius);
        }

        const double centralHeight = 0.5 * (passLow + passHigh);
        const double incidenceSecond = LensSystem::incidenceAngle(ghost.toSecond[w], centralHeight, rayAngle, m_lensSystem.getRadius(ghost.second));
        const double incidenceFirst = LensSystem::incidenceAngle(ghost.toFirst[w], centralHeight, rayAngle, m_lensSystem.getRadius(ghost.first));
        const double reflectance =
          m_lensSystem.computeReflectance(ghost.second, w, incidenceSecond, true) *
          m_lensSystem.computeReflectance(ghost.first, w, incidenceFirst, false) *
          ghost.transmission[w];
        reflectances[w] = reflectance;

        // The ghost's irradiance on the sensor, reflectance * E / Mf11^2, as the radiance that would produce it
        // through the main path, which reaches the sensor through the entrance pupil seen from f away.
        const double radiance = reflectance * focalLength * focalLength / (sensorFromPupil * sensorFromPupil * pupilArea);
        const Vector3 weighted = wavelengthWeights[w] * float(radiance);
        total = total + weighted;

        const double irisFactor = toSensor.b - sensorFromPupil * toStop.b / toStop.a;
        irisCircles[w] = { irisFactor * rayAngleX, irisFactor * rayAngleY, std::abs(sensorFromPupil / toStop.a) * irisCornerRadius };
        pupilCircles[w] = { toSensor.b * rayAngleX, toSensor.b * rayAngleY, std::abs(sensorFromPupil) * ghost.apertureLimit };

        if (w == kMiddle) {
          referenceHeight = centralHeight;
          referenceHalfWidth = 0.5 * (passHigh - passLow);
          middleSensorFromPupil = sensorFromPupil;

          // Each point of the sun's disc casts the ghost, the iris's image moving by irisFactor and the passing
          // circle's by sensorFromAngle per radian. The smaller of the two draws the ghost's edge. The light within
          // moves with the passing circle, so the slower of the two motions blurs the ghost's sharpest features.
          angularMagnification = irisCircles[w].radius < pupilCircles[w].radius ? std::abs(irisFactor) : std::abs(toSensor.b);
          sharpestMagnification = std::min(std::abs(irisFactor), std::abs(toSensor.b));
        }
      }

      const float luminance = sRGBLuminance(total) * intensity();

      if (luminance < kMinGhostRadiance) {
        continue;
      }

      const float fadeT = std::clamp((luminance - kMinGhostRadiance) / ((kGhostFadeRange - 1.0f) * kMinGhostRadiance), 0.0f, 1.0f);
      const float fade = fadeT * fadeT * (3.0f - 2.0f * fadeT);

      // How far the ghost's edge, the smaller of its two circles, moves from one wavelength to the next, and across
      // the spectrum.
      const auto circleStep = [](const Circle& a, const Circle& b) {
        return std::hypot(b.x - a.x, b.y - a.y) + std::abs(b.radius - a.radius);
      };
      const auto bounding = [&](uint32_t w) -> const Circle& {
        return irisCircles[w].radius < pupilCircles[w].radius ? irisCircles[w] : pupilCircles[w];
      };
      double spread = 0.0;
      double travel = 0.0;

      for (uint32_t w = 0; w + 1 < LENS_FLARE_WAVELENGTHS; w++) {
        const double step = circleStep(bounding(w), bounding(w + 1));
        spread = std::max(spread, step);
        travel += step;
      }

      // The edges blur across the sun's disc, scaled by the softness.
      const double sunBlur = sunDiameter * angularMagnification * double(edgeSoftness());

      // A fringe within a pixel or the sun's blur, or too faint to see, takes one wavelength for the spectrum. A ghost
      // drawn across the spectrum stays so until it is well inside both, so it does not flicker between the two.
      const size_t ghostIndex = size_t(&ghost - m_ghosts.data());
      const bool wasSpectral = ghostIndex < m_spectralGhosts.size() && m_spectralGhosts[ghostIndex];
      const double hysteresis = wasSpectral ? kSpectralHysteresis : 1.0;
      const bool narrowFringe = travel <= std::max(pixelMm, 0.5 * sunBlur) * hysteresis;
      const bool singleSample = narrowFringe || luminance < kMinSpectralGhostRadiance * float(hysteresis);

      if (ghostIndex < m_spectralGhosts.size()) {
        m_spectralGhosts[ghostIndex] = !singleSample;
      }

      if (traced) {
        m_drawnGhostCount++;

        // The ghost is drawn at the coarsest resolution whose pixels the blur of its sharpest features hides.
        uint32_t level = 0;

        if (reducedResolution()) {
          const double blurPixels = sunDiameter * sharpestMagnification * double(edgeSoftness()) / pixelMm;
          const double radiusPixels = bounding(kMiddle).radius / pixelMm;

          while (level + 1 < kRasterLevels && blurPixels >= kReducedBlurPixels * double(2u << level) &&
                 radiusPixels >= kReducedRadiusPixels * double(2u << level)) {
            level++;
          }
        }

        m_levelGhostCounts[level]++;
        // A split ghost's instances lie together, counted from the start of their resolution's.
        std::vector<LensFlareTraceInstance>& instances = levelInstances[level];

        // The search spans the stop's paraxial image along the sun's meridian with a wide margin, kept on the front
        // element, or the whole front element when that is smaller.
        const RayTransferMatrix& toStop = ghost.toStop[kMiddle];
        const double stopImageRadius = irisCornerRadius / std::abs(toStop.a);
        const double frontRadius = m_lensSystem.getFrontRadius();
        double searchCenter = 0.0;
        double searchHalfSize = 1.05 * frontRadius;

        if (stopImageRadius < frontRadius) {
          searchCenter = std::clamp(-toStop.b * rayAngle / toStop.a, -frontRadius, frontRadius);
          searchHalfSize = std::min(std::max(kSearchMargin * stopImageRadius, kMinSearchShare * frontRadius), searchHalfSize);
        }

        if (singleSample) {
          // The middle wavelength is traced, carrying the whole spectrum relative to its reflectance.
          const double reference = std::max(reflectances[kMiddle], 1e-12);
          Vector3 weight(0.0f);

          for (uint32_t w = 0; w < LENS_FLARE_WAVELENGTHS; w++) {
            weight = weight + wavelengthWeights[w] * float(reflectances[w] / reference);
          }

          LensFlareTraceInstance instance = {};
          instance.first = ghost.first;
          instance.second = ghost.second;
          instance.wavelength = kMiddle;
          instance.transmission = float(ghost.transmission[kMiddle]);
          instance.searchCenter = float(searchCenter);
          instance.searchHalfSize = float(searchHalfSize);
          instance.kind = LENS_FLARE_INSTANCE_WHOLE;
          instance.ghostFirst = static_cast<uint32_t>(instances.size());
          instance.weight = weight * fade;
          instance.edgeBlur = float(sunBlur);
          instances.push_back(instance);
        } else {
          const uint32_t ghostFirst = static_cast<uint32_t>(instances.size());
          const float bandInner = float((kBandFringes * travel + 0.5 * sunBlur) / pixelMm) + kBandMarginPixels;
          const float bandWidth = std::max(float(kBandTransitionFringes * travel / pixelMm), kMinBandTransitionPixels);

          // The grid's cells on the screen and the passing region's inradius, where the carrier's iris image and
          // passing circle overlap, both paraxially.
          const double passRadius = std::min(stopImageRadius, ghost.apertureLimit);
          const double cellPixels = 2.0 * passRadius * std::abs(middleSensorFromPupil) / pixelMm / double(gridSize - 1);
          const Circle& iris = irisCircles[kMiddle];
          const Circle& pupil = pupilCircles[kMiddle];
          const double centres = std::hypot(iris.x - pupil.x, iris.y - pupil.y);
          const double inradius = centres <= std::abs(iris.radius - pupil.radius)
            ? std::min(iris.radius, pupil.radius)
            : std::max(0.5 * (iris.radius + pupil.radius - centres), 0.0);
          const double bandPixels = std::max(double(bandInner), double(LENS_FLARE_BAND_CELLS) * cellPixels) +
                                    std::max(double(bandWidth), double(LENS_FLARE_HAND_OFF_CELLS) * cellPixels);
          const bool split = inradius / pixelMm >= kSplitBands * bandPixels;

          for (uint32_t w = 0; w < LENS_FLARE_WAVELENGTHS; w++) {
            const uint32_t splitKind = w == LENS_FLARE_CARRIER_OFFSET ? LENS_FLARE_INSTANCE_CARRIER : LENS_FLARE_INSTANCE_BANDED;
            LensFlareTraceInstance instance = {};
            instance.first = ghost.first;
            instance.second = ghost.second;
            instance.wavelength = w;
            instance.transmission = float(ghost.transmission[w]);
            instance.searchCenter = float(searchCenter);
            instance.searchHalfSize = float(searchHalfSize);
            instance.kind = split ? splitKind : LENS_FLARE_INSTANCE_WHOLE;
            instance.ghostFirst = split ? ghostFirst : static_cast<uint32_t>(instances.size());
            instance.weight = wavelengthWeights[w] * fade;
            instance.bandInner = bandInner;
            instance.bandWidth = bandWidth;
            instance.edgeBlur = float(sunBlur);
            instances.push_back(instance);
          }
        }

        continue;
      }

      out.sampleCount = singleSample ? 1u : LENS_FLARE_WAVELENGTHS;
      spread = std::max(singleSample ? 0.0 : spread, sunBlur);
      out.edgeSpread = float(spread);
      const double edgeMm = std::max(pixelMm, spread);

      // The apertures whose discs cut some wavelength's iris image on the entrance plane within the edges' blend, the
      // deepest cuts first. A disc that misses every wavelength's iris image leaves nothing to draw.
      bool passes = slopesPass;
      cuts.clear();

      for (size_t k = 0; k < discs.size() && passes; k++) {
        const double centre = discs[k].centrePerSlope * rayAngle;
        double depth = std::numeric_limits<double>::max();
        bool misses = true;

        for (uint32_t w = 0; w < LENS_FLARE_WAVELENGTHS; w++) {
          const RayTransferMatrix& toStop = ghost.toStop[w];
          const double irisRadius = irisCornerRadius / std::abs(toStop.a);
          const double offset = std::abs(centre + toStop.b * rayAngle / toStop.a);
          const double blend = 0.5 * edgeMm / std::max(std::abs(ghost.toSensor[w].a), minSensorFromPupil);
          depth = std::min(depth, discs[k].radius - offset - irisRadius - blend);
          misses = misses && offset - irisRadius >= discs[k].radius + blend;
        }

        passes = !misses;

        if (depth < 0.0) {
          cuts.push_back({ depth, k });
        }
      }

      if (!passes) {
        continue;
      }

      // What passes is found on the entrance plane paraxially, and its rays' landing on the sensor fitted to the ghost's
      // real rays about the middle of it, which carries its aberrations near there. The ghost fades out as those rays
      // near being lost, as the real lens loses its light. A ghost focused near the sensor would be a point of
      // unbounded radiance, so each axis keeps at least minSensorFromPupil of magnification about where the middle ray
      // lands, as the paraxial ghosts do.
      std::array<LensSystem::GhostMap, LENS_FLARE_WAVELENGTHS> maps;
      const float survival = float(m_lensSystem.mapGhost(ghost, rayAngle, referenceHeight, referenceHalfWidth, maps));

      if (survival <= 0.0f) {
        continue;
      }

      for (LensSystem::GhostMap& map : maps) {
        const double landing = map.scaleX * referenceHeight + map.offset;
        map.scaleX = std::abs(map.scaleX) < minSensorFromPupil ? std::copysign(minSensorFromPupil, map.scaleX) : map.scaleX;
        map.scaleY = std::abs(map.scaleY) < minSensorFromPupil ? std::copysign(minSensorFromPupil, map.scaleY) : map.scaleY;
        map.offset = landing - map.scaleX * referenceHeight;
      }

      std::sort(cuts.begin(), cuts.end());
      out.passDiscCount = static_cast<uint32_t>(std::min<size_t>(cuts.size(), LENS_FLARE_PASS_DISCS));

      for (uint32_t k = 0; k < out.passDiscCount; k++) {
        const EntranceDisc& disc = discs[cuts[k].second];
        out.passDiscs[k] = vec2(float(disc.centrePerSlope * rayAngle), float(disc.radius));
      }

      for (uint32_t k = 0; k < out.sampleCount; k++) {
        const uint32_t w = singleSample ? kMiddle : k;
        const LensSystem::GhostMap& map = maps[w];
        const double areaScale = std::abs(map.scaleX * map.scaleY);
        LensFlareGhostSample& sample = out.samples[k];
        sample.entranceFromSensor = vec2(float(1.0 / map.scaleX), float(1.0 / map.scaleY));
        sample.sensorOffset = float(map.offset);
        sample.entrancePerSensor = float(1.0 / std::sqrt(areaScale));
        sample.stopFromEntrance = float(ghost.toStop[w].a);
        sample.stopOffset = float(ghost.toStop[w].b * rayAngle);

        // The ghost's irradiance on the sensor, reflectance * E over its area magnification, as the radiance that would
        // produce it through the main path. A single sample carries the whole spectrum.
        const uint32_t firstWavelength = singleSample ? 0 : w;
        const uint32_t endWavelength = singleSample ? LENS_FLARE_WAVELENGTHS : w + 1;
        Vector3 radiance(0.0f);

        for (uint32_t v = firstWavelength; v < endWavelength; v++) {
          radiance = radiance + wavelengthWeights[v] * float(reflectances[v] * focalLength * focalLength / (areaScale * pupilArea));
        }

        sample.radiance = radiance * (fade * survival);
      }

      // The bound is one circle around every wavelength's smallest of its iris image's and cutting discs' images, grown
      // by the edges' blend and a couple of pixels. The images lie in the meridian's frame, turned into aspect corrected
      // NDC, in which the sensor's image is inverted.
      const auto screenBound = [&](uint32_t w) {
        const LensSystem::GhostMap& map = maps[w];
        const RayTransferMatrix& toStop = ghost.toStop[w];
        const double scale = std::max(std::abs(map.scaleX), std::abs(map.scaleY));
        double centre = map.offset - map.scaleX * toStop.b * rayAngle / toStop.a;
        double radius = scale * irisCornerRadius / std::abs(toStop.a);

        for (uint32_t k = 0; k < out.passDiscCount; k++) {
          const EntranceDisc& disc = discs[cuts[k].second];

          if (scale * disc.radius < radius) {
            centre = map.scaleX * disc.centrePerSlope * rayAngle + map.offset;
            radius = scale * disc.radius;
          }
        }

        return Circle { -centre * meridian.x / sensorScale, -centre * meridian.y / sensorScale, radius / sensorScale };
      };
      const Circle center = screenBound(kMiddle);
      double boundRadius = 0.0;

      for (uint32_t w = 0; w < LENS_FLARE_WAVELENGTHS; w++) {
        const Circle circle = screenBound(w);
        boundRadius = std::max(boundRadius, std::hypot(circle.x - center.x, circle.y - center.y) + circle.radius);
      }

      out.boundCenter = vec2(float(center.x), float(center.y));
      out.boundRadius = float(boundRadius + (2.0 * pixelMm + spread) / sensorScale);

      m_drawnGhostCount++;
      fastGhosts.push_back(out);
    }

    // Each resolution's instances are listed in turn, so that each resolution is drawn at once.
    std::vector<LensFlareTraceInstance> instances;
    std::array<uint32_t, kRasterLevels + 1> levelStarts {};

    for (uint32_t level = 0; level < kRasterLevels; level++) {
      levelStarts[level] = static_cast<uint32_t>(instances.size());

      for (LensFlareTraceInstance instance : levelInstances[level]) {
        instance.ghostFirst += levelStarts[level];
        instances.push_back(instance);
      }
    }

    levelStarts[kRasterLevels] = static_cast<uint32_t>(instances.size());
    m_drawnInstanceCount = static_cast<uint32_t>(traced ? instances.size() : fastGhosts.size());

    if (m_drawnInstanceCount == 0) {
      return;
    }

    ScopedGpuProfileZone(&ctx, "Lens Flare");
    ctx.setFramePassStage(RtxFramePassStage::LensFlare);

    const Vector3 sourceIlluminance = sun.illuminance * intensity();

    if (traced) {
      // The rays are traced in the frame of the sun's meridian, turned from the lens's by the direction they tilt in.
      LensFlareTraceArgs traceArgs = {};
      traceArgs.meridian = meridian;
      traceArgs.sunAngle = float(std::atan(rayAngle));
      traceArgs.surfaceCount = std::min(m_lensSystem.getSurfaceCount(), uint32_t(LENS_FLARE_MAX_SURFACES));
      traceArgs.stopIndex = m_lensSystem.getStopIndex();
      traceArgs.sensorZ = float(m_lensSystem.getSensorZ());
      traceArgs.coating = static_cast<uint32_t>(LensAperture::coating());
      traceArgs.coatingWavelengthNm = LensAperture::coatingWavelengthNm();
      traceArgs.sensorReflectance = LensAperture::sensorReflectance();
      traceArgs.irisCornerRadius = float(irisCornerRadius);
      traceArgs.gridSize = gridSize;
      traceArgs.instanceCount = static_cast<uint32_t>(instances.size());
      traceArgs.pixelMm = float(pixelMm);
      traceArgs.sunRadius = float(0.5 * sunDiameter);
      traceArgs.edgeSoftness = edgeSoftness();
      traceArgs.apertureBlades = LensAperture::getBladeCount();
      traceArgs.apertureCurvature = curvature;
      traceArgs.apertureRotation = LensAperture::getRotationRadians();
      traceArgs.apertureDistanceJitter = apertureDistanceJitter;
      traceArgs.apertureAngleJitter = apertureAngleJitter;

      const auto& wavelengths = m_lensSystem.getWavelengths();

      for (uint32_t w = 0; w < kWavelengthCount; w++) {
        setComponent(traceArgs.wavelengthsNm[w / 4], w % 4, float(wavelengths[w] * 1000.0));
      }

      LensFlareRasterArgs rasterArgs = {};
      rasterArgs.sensorScale = float(sensorScale);
      rasterArgs.aspectRatio = sun.aspectRatio;
      rasterArgs.gridSize = traceArgs.gridSize;
      rasterArgs.pixelMm = float(pixelMm);
      rasterArgs.sourceRadiance = sourceIlluminance * float(focalLength * focalLength / pupilArea);
      rasterArgs.apertureBlades = LensAperture::getBladeCount();
      rasterArgs.apertureCurvature = curvature;
      rasterArgs.apertureRotation = LensAperture::getRotationRadians();
      rasterArgs.apertureDistanceJitter = apertureDistanceJitter;
      rasterArgs.apertureAngleJitter = apertureAngleJitter;
      rasterArgs.apertureCornerReach = LensAperture::getSides(curvature, apertureDistanceJitter, apertureAngleJitter).cornerReach;
      rasterArgs.imageSize = Vector2(float(extent.width), float(extent.height));

      dispatchRayTraced(ctx, state, color, traceArgs, rasterArgs, instances, levelStarts, sunProbe);
      return;
    }

    LensFlareArgs args = {};
    args.imageSize = { extent.width, extent.height };
    args.invImageSize = { 1.0f / float(extent.width), 1.0f / float(extent.height) };
    args.sunRayAngle = { float(rayAngleX), float(rayAngleY) };
    args.sensorScale = float(sensorScale);
    args.aspectRatio = sun.aspectRatio;
    args.sourceIlluminance = sourceIlluminance;
    args.ghostCount = static_cast<uint32_t>(fastGhosts.size());
    args.stopRadius = float(irisCornerRadius);
    args.apertureBlades = LensAperture::getBladeCount();
    args.apertureCurvature = curvature;
    args.apertureRotation = LensAperture::getRotationRadians();
    args.apertureDistanceJitter = apertureDistanceJitter;
    args.apertureAngleJitter = apertureAngleJitter;
    args.debugView = static_cast<uint32_t>(debugView());

    dispatchFast(ctx, color, args, fastGhosts, sunProbe);
  }

  void RtxLensFlare::dispatchFast(RtxContext& ctx, const Resources::Resource& color, const LensFlareArgs& args,
                                  const std::vector<LensFlareGhost>& ghosts, const RtxSunProbe& sunProbe) {
    ensureBuffer<LensFlareGhost>(m_device, m_ghostBuffer, LENS_FLARE_MAX_GHOSTS,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, "Lens flare ghosts");
    uploadChunked(ctx, m_ghostBuffer, ghosts.data(), sizeof(LensFlareGhost) * ghosts.size());

    ctx.setPushConstantBank(DxvkPushConstantBank::RTX);
    ctx.pushConstants(0, sizeof(args), &args);
    ctx.bindResourceBuffer(LENS_FLARE_GHOSTS_INPUT, DxvkBufferSlice(m_ghostBuffer, 0, m_ghostBuffer->info().size));
    ctx.bindResourceView(LENS_FLARE_SUN_VISIBILITY_INPUT, sunProbe.getVisibilityView(), nullptr);
    ctx.bindResourceView(LENS_FLARE_COLOR_INPUT_OUTPUT, color.view, nullptr);
    ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, LensFlareShader::getShader());

    const VkExtent3D workgroups = util::computeBlockCount(color.image->info().extent, VkExtent3D { LENS_FLARE_TILE_SIZE, LENS_FLARE_TILE_SIZE, 1 });
    ctx.dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void RtxLensFlare::updateReducedTargets(RtxContext& ctx, const VkExtent3D& extent) {
    for (uint32_t level = 1; level < kRasterLevels; level++) {
      const uint32_t scale = 1u << level;
      const VkExtent3D reduced = { (extent.width + scale - 1) / scale, (extent.height + scale - 1) / scale, 1 };
      Resources::Resource& target = m_reducedTargets[level - 1];

      if (target.isValid() && target.image->info().extent.width == reduced.width &&
          target.image->info().extent.height == reduced.height) {
        continue;
      }

      Rc<DxvkContext> baseCtx = &ctx;
      const char* name = level == 1 ? "lens flare half resolution ghosts" : "lens flare quarter resolution ghosts";
      target = Resources::createImageResource(baseCtx, name, reduced, VK_FORMAT_R16G16B16A16_SFLOAT, 1, VK_IMAGE_TYPE_2D,
                                              VK_IMAGE_VIEW_TYPE_2D, 0, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    }
  }

  void RtxLensFlare::dispatchRayTraced(RtxContext& ctx, DxvkContextState& state, const Resources::Resource& color,
                                       const LensFlareTraceArgs& traceArgs, const LensFlareRasterArgs& rasterArgs,
                                       const std::vector<LensFlareTraceInstance>& instances,
                                       const std::array<uint32_t, kRasterLevels + 1>& levelStarts, const RtxSunProbe& sunProbe) {
    const uint32_t gridVertices = traceArgs.gridSize * traceArgs.gridSize;

    updateSurfaceBuffer(ctx);
    ensureBuffer<LensFlareTraceInstance>(m_device, m_instanceBuffer, LENS_FLARE_MAX_GHOSTS * LENS_FLARE_WAVELENGTHS,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, "Lens flare trace instances");
    ensureBuffer<LensFlareTracedVertex>(m_device, m_vertexBuffer, size_t(gridVertices) * instances.size(),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, "Lens flare traced vertices");
    ensureBuffer<LensFlareTraceDomain>(m_device, m_domainBuffer, LENS_FLARE_MAX_GHOSTS * LENS_FLARE_WAVELENGTHS,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, "Lens flare trace domains");
    uploadChunked(ctx, m_instanceBuffer, instances.data(), sizeof(LensFlareTraceInstance) * instances.size());

    {
      ScopedGpuProfileZone(&ctx, "Lens Flare Trace");
      ctx.setPushConstantBank(DxvkPushConstantBank::RTX);
      ctx.pushConstants(0, sizeof(traceArgs), &traceArgs);
      ctx.bindResourceBuffer(LENS_FLARE_TRACE_SURFACES_INPUT, DxvkBufferSlice(m_surfaceBuffer));
      ctx.bindResourceBuffer(LENS_FLARE_TRACE_INSTANCES_INPUT, DxvkBufferSlice(m_instanceBuffer));
      ctx.bindResourceBuffer(LENS_FLARE_TRACE_DOMAINS_INPUT_OUTPUT, DxvkBufferSlice(m_domainBuffer));
      ctx.bindResourceBuffer(LENS_FLARE_TRACE_VERTICES_INPUT_OUTPUT, DxvkBufferSlice(m_vertexBuffer));

      // One group searches each instance.
      ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, LensFlareSearchShader::getShader());
      ctx.dispatch(1, traceArgs.instanceCount, 1);

      const uint32_t gridGroups = (gridVertices + LENS_FLARE_TRACE_GROUP_SIZE - 1) / LENS_FLARE_TRACE_GROUP_SIZE;
      ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, LensFlareTraceShader::getShader());
      ctx.dispatch(gridGroups, traceArgs.instanceCount, 1);

      ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, LensFlareIrradianceShader::getShader());
      ctx.dispatch(gridGroups, traceArgs.instanceCount, 1);

      ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, LensFlareExtendShader::getShader());
      ctx.dispatch(gridGroups, traceArgs.instanceCount, 1);

      ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, LensFlareExtendOuterShader::getShader());
      ctx.dispatch(gridGroups, traceArgs.instanceCount, 1);
    }

    const VkClearColorValue black = {};
    VkImageSubresourceRange range = {};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = 1;
    range.layerCount = 1;

    if (debugView() == LensFlareDebugView::FlareOnly) {
      ctx.clearColorImage(color.image, black, range);
    }

    ScopedGpuProfileZone(&ctx, "Lens Flare Raster");

    const VkExtent3D extent = color.image->info().extent;
    const bool reduced = levelStarts[kRasterLevels] > levelStarts[1];

    if (reduced) {
      updateReducedTargets(ctx, extent);

      for (const Resources::Resource& target : m_reducedTargets) {
        ctx.clearColorImage(target.image, black, range);
      }
    }

    const DxvkContextState stateCopy = state;

    ctx.bindResourceBuffer(LENS_FLARE_RASTER_VERTICES_INPUT, DxvkBufferSlice(m_vertexBuffer));
    ctx.bindResourceBuffer(LENS_FLARE_RASTER_INSTANCES_INPUT, DxvkBufferSlice(m_instanceBuffer));
    ctx.bindResourceView(LENS_FLARE_RASTER_SUN_VISIBILITY_INPUT, sunProbe.getVisibilityView(), nullptr);
    ctx.bindShader(VK_SHADER_STAGE_VERTEX_BIT, LensFlareGhostVertexShader::getShader());
    ctx.bindShader(VK_SHADER_STAGE_FRAGMENT_BIT, LensFlareGhostFragmentShader::getShader());
    ctx.setPushConstantBank(DxvkPushConstantBank::RTX);

    // Each cell is two triangles, each drawn over its cover as two more.
    const uint32_t quadsPerSide = traceArgs.gridSize - 1;

    for (uint32_t level = 0; level < kRasterLevels; level++) {
      const uint32_t count = levelStarts[level + 1] - levelStarts[level];

      if (count == 0) {
        continue;
      }

      const Resources::Resource& target = level == 0 ? color : m_reducedTargets[level - 1];
      LensFlareRasterArgs levelArgs = rasterArgs;
      levelArgs.firstInstance = levelStarts[level];

      if (level > 0) {
        const VkExtent3D targetExtent = target.image->info().extent;
        levelArgs.pixelMm = float(2.0 * double(rasterArgs.sensorScale) / double(targetExtent.height));
        levelArgs.imageSize = Vector2(float(targetExtent.width), float(targetExtent.height));
      }

      setupGhostRasterState(ctx, target);
      ctx.pushConstants(0, sizeof(levelArgs), &levelArgs);
      ctx.draw(quadsPerSide * quadsPerSide * 12, count, 0, 0);
    }

    if (reduced) {
      LensFlareCompositeArgs compositeArgs = {};
      compositeArgs.imageSize = { extent.width, extent.height };
      compositeArgs.invImageSize = { 1.0f / float(extent.width), 1.0f / float(extent.height) };
      compositeArgs.drawHalf = levelStarts[2] > levelStarts[1] ? 1u : 0u;
      compositeArgs.drawQuarter = levelStarts[3] > levelStarts[2] ? 1u : 0u;
      ctx.pushConstants(0, sizeof(compositeArgs), &compositeArgs);

      const Rc<DxvkSampler> linearSampler =
        ctx.getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
      ctx.bindResourceView(LENS_FLARE_COMPOSITE_HALF_INPUT, m_reducedTargets[0].view, nullptr);
      ctx.bindResourceSampler(LENS_FLARE_COMPOSITE_HALF_INPUT, linearSampler);
      ctx.bindResourceView(LENS_FLARE_COMPOSITE_QUARTER_INPUT, m_reducedTargets[1].view, nullptr);
      ctx.bindResourceSampler(LENS_FLARE_COMPOSITE_QUARTER_INPUT, linearSampler);
      ctx.bindResourceView(LENS_FLARE_COMPOSITE_COLOR_INPUT_OUTPUT, color.view, nullptr);
      ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, LensFlareCompositeShader::getShader());

      const VkExtent3D workgroups = util::computeBlockCount(extent, VkExtent3D { LENS_FLARE_TILE_SIZE, LENS_FLARE_TILE_SIZE, 1 });
      ctx.dispatch(workgroups.width, workgroups.height, workgroups.depth);
    }

    state = stateCopy;
  }

  void RtxLensFlare::showImguiSettings() {
    ImGui::Indent();
    RemixGui::Checkbox("Lens Flare Enabled", &enableObject());
    RemixGui::SetTooltipToLastWidgetOnHover("Ghosts of the Physical Atmosphere sun, traced through the camera's lens. The starburst comes from the\nconvolution bloom's diffraction kernel.");

    if (RtxOptions::skyMode() != SkyMode::PhysicalAtmosphere) {
      ImGui::TextWrapped("The lens flare follows the Physical Atmosphere sun, so it needs that sky mode.");
    }

    if (LensAperture::isEye()) {
      ImGui::TextWrapped("The eye observer has no lens, and so no ghosts.");
    }

    RemixGui::DragFloat("Intensity##lensFlare", &intensityObject(), 0.05f, 0.0f, 1000.0f, "%.2f");
    RemixGui::SetTooltipToLastWidgetOnHover("Scale on the ghosts' physically derived brightness. 1 = physical.");
    qualityCombo.getKey(&qualityObject());
    RemixGui::SliderInt("Ghosts##lensFlare", &maxGhostsObject(), 1, LENS_FLARE_MAX_GHOSTS);

    if (quality() == LensFlareQuality::RayTraced) {
      RemixGui::SliderInt("Trace Resolution##lensFlare", &traceResolutionObject(), 8, 128);
      RemixGui::SetTooltipToLastWidgetOnHover("Cells per side of each ghost's grid of rays.");
      RemixGui::Checkbox("Reduced Resolution##lensFlare", &reducedResolutionObject());
      RemixGui::SetTooltipToLastWidgetOnHover("Draw the ghosts whose edges the sun's disc blurs over several pixels at half or quarter resolution.");
    }

    RemixGui::DragFloat("Edge Softness##lensFlare", &edgeSoftnessObject(), 0.01f, 0.0f, 10.0f, "%.2f");
    RemixGui::SetTooltipToLastWidgetOnHover("Scale on the blur of the ghosts' edges by the sun's disc. 1 = physical.");
    debugViewCombo.getKey(&debugViewObject());

    if (m_lensSystem.getPrescription() != ~0u && m_entrancePupilRadius > 0.0) {
      ImGui::Text("EFL %.1f mm, f/%.1f with a %.1f mm entrance pupil", m_lensSystem.getFocalLength(),
                  m_lensSystem.getFocalLength() / (2.0 * m_entrancePupilRadius), 2.0 * m_entrancePupilRadius);
      ImGui::Text("%u ghosts drawn of the brightest %zu of %u (%u passes)", m_drawnGhostCount, m_ghosts.size(), m_candidateCount,
                  m_drawnInstanceCount);

      if (quality() == LensFlareQuality::RayTraced && reducedResolution()) {
        ImGui::Text("%u drawn at half resolution and %u at quarter", m_levelGhostCounts[1], m_levelGhostCounts[2]);
      }
    }

    ImGui::Separator();
    LensAperture::showImguiSettings();
    ImGui::Unindent();
  }

}
