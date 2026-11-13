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
#include "rtx_fsr.h"

#include <algorithm>
#include <array>

#include "rtx_context.h"
#include "rtx_fsr_api.h"
#include "../dxvk_device.h"
#include "../dxvk_scoped_annotation.h"
#include "../util/util_once.h"

namespace dxvk {

  namespace {

    std::string describeFlags(uint32_t flags) {
      std::string description = (flags & FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE) != 0 ? "HDR colour" : "SDR colour";

      if ((flags & FFX_UPSCALE_ENABLE_DEPTH_INVERTED) != 0) {
        description += ", inverted depth";
      }

      if ((flags & FFX_UPSCALE_ENABLE_DEPTH_INFINITE) != 0) {
        description += ", infinite far plane";
      }

      if ((flags & FFX_UPSCALE_ENABLE_DEBUG_CHECKING) != 0) {
        description += ", debug checker";
      }

      return description;
    }

  } // namespace

  DxvkFSR::DxvkFSR(DxvkDevice* device)
    : CommonDeviceObject(device) { }

  DxvkFSR::~DxvkFSR() {
    // onDestroy has released the contexts by now, unless the device went away without it.
    // The device is idle at this point either way.
    for (Context& context : m_contexts) {
      destroyContext(context);
    }
  }

  void DxvkFSR::onDestroy() {
    releaseContexts();
  }

  bool DxvkFSR::hasRuntimeFiles() {
    return FsrApi::hasRuntimeFiles(FsrEffect::Upscaler);
  }

  bool DxvkFSR::isAvailable() const {
    return FsrApi::get().isAvailable(FsrEffect::Upscaler);
  }

  std::vector<std::string> DxvkFSR::versionNames() const {
    std::vector<std::string> names;

    for (const FsrVersion& version : FsrApi::get().versions(FsrEffect::Upscaler, *m_device)) {
      names.push_back(version.name);
    }

    return names;
  }

  std::string DxvkFSR::activeVersion() const {
    std::lock_guard<std::mutex> lock(m_activeVersionMutex);

    return m_activeVersion;
  }

  void DxvkFSR::releaseContexts() {
    const bool hasContext = std::any_of(std::begin(m_contexts), std::end(m_contexts), [] (const Context& context) {
      return context.handle != nullptr;
    });

    if (hasContext) {
      // Frames that are still in flight may be using the contexts' resources.
      m_device->waitForIdle();
    }

    for (Context& context : m_contexts) {
      destroyContext(context);
      context = Context();
    }

    std::lock_guard<std::mutex> lock(m_activeVersionMutex);
    m_activeVersion.clear();
  }

  void DxvkFSR::destroyContext(Context& context) {
    if (context.handle == nullptr) {
      return;
    }

    const ffxReturnCode_t result = FsrApi::get().destroyContext(&context.handle);

    if (result != FFX_API_RETURN_OK) {
      Logger::warn(str::format("FSR: Failed to destroy the upscaling context: ", FsrApi::returnCodeName(result), "."));
    }

    context.handle = nullptr;
  }

  bool DxvkFSR::createContext(Context& context, const ContextConfig& config) {
    FsrApi& api = FsrApi::get();

    ffx::CreateContextDescUpscale createDesc;
    createDesc.flags = config.flags;
    createDesc.maxRenderSize = { config.renderExtent.width, config.renderExtent.height };
    createDesc.maxUpscaleSize = { config.displayExtent.width, config.displayExtent.height };
    createDesc.fpMessage = FsrApi::messageCallback();

    // The runtime keeps a context from using anything newer than the version it was built against, unless told.
    ffx::CreateContextDescUpscaleVersion versionDesc;
    versionDesc.version = FFX_UPSCALER_VERSION;

    ffx::CreateBackendVKDesc backendDesc;
    backendDesc.deviceContext = FsrApi::deviceContext(*m_device);

    ffx::CreateContextDescOverrideVersion overrideDesc;

    createDesc.header.pNext = &versionDesc.header;
    versionDesc.header.pNext = &backendDesc.header;

    // Zero leaves the choice to the runtime, which picks the newest version the GPU supports.
    // That also lets a graphics driver supply a newer upscaler than the one that ships here.
    const uint64_t requestedVersion = api.findVersion(FsrEffect::Upscaler, *m_device, config.versionOverride);

    if (!config.versionOverride.empty() && requestedVersion == 0) {
      Logger::warn(str::format("FSR: The runtime does not offer upscaler version \"", config.versionOverride,
                               "\" on this GPU. The runtime picks the version instead."));
    }

    // When the first choice cannot be created, every version the runtime lists is tried in turn, newest first.
    std::vector<uint64_t> candidates = { requestedVersion };

    for (const FsrVersion& version : api.versions(FsrEffect::Upscaler, *m_device)) {
      if (version.id != requestedVersion) {
        candidates.push_back(version.id);
      }
    }

    ffxReturnCode_t result = FFX_API_RETURN_NO_PROVIDER;

    for (const uint64_t candidate : candidates) {
      overrideDesc.versionId = candidate;
      backendDesc.header.pNext = candidate != 0 ? &overrideDesc.header : nullptr;

      context.handle = nullptr;
      result = api.createContext(FsrEffect::Upscaler, &context.handle, &createDesc.header);

      if (result == FFX_API_RETURN_OK && context.handle != nullptr) {
        break;
      }

      context.handle = nullptr;
    }

    if (context.handle == nullptr) {
      Logger::err(str::format("FSR: Failed to create the upscaling context for ",
                              config.renderExtent.width, "x", config.renderExtent.height, " -> ",
                              config.displayExtent.width, "x", config.displayExtent.height, ": ",
                              FsrApi::returnCodeName(result), "."));
      return false;
    }

    api.routeMessages(FsrEffect::Upscaler, &context.handle, (config.flags & FFX_UPSCALE_ENABLE_DEBUG_CHECKING) != 0);

    context.version = api.contextVersion(&context.handle);
    context.tuning = TuningValues();

    Logger::info(str::format("FSR: Created the upscaling context: ",
                             config.renderExtent.width, "x", config.renderExtent.height, " -> ",
                             config.displayExtent.width, "x", config.displayExtent.height,
                             ", FSR ", context.version.empty() ? "of an unknown version" : context.version,
                             ", ", describeFlags(config.flags), "."));

    return true;
  }

  bool DxvkFSR::ensureContext(Context& context, const ContextConfig& config) {
    if (context.config == config) {
      if (context.handle != nullptr) {
        return true;
      }

      // Creation is not retried until something about the request changes.
      if (context.failed) {
        return false;
      }
    }

    if (context.handle != nullptr) {
      // Frames that are still in flight may be using the context's resources.
      m_device->waitForIdle();
      destroyContext(context);
    }

    context.config = config;
    context.failed = !createContext(context, config);

    return !context.failed;
  }

  void DxvkFSR::applyTuning(Context& context) {
    TuningValues values;
    values.velocityFactor = velocityFactor();
    values.reactivenessScale = reactivenessScale();
    values.shadingChangeScale = shadingChangeScale();
    values.accumulationAddedPerFrame = accumulationAddedPerFrame();
    values.minDisocclusionAccumulation = minDisocclusionAccumulation();

    // A new context starts from the defaults, so nothing is sent until an option moves away from them.
    if (context.tuning == values) {
      return;
    }

    context.tuning = values;

    const std::array<std::pair<uint64_t, float>, 5> keys = { {
      { FFX_API_CONFIGURE_UPSCALE_KEY_FVELOCITYFACTOR, values.velocityFactor },
      { FFX_API_CONFIGURE_UPSCALE_KEY_FREACTIVENESSSCALE, values.reactivenessScale },
      { FFX_API_CONFIGURE_UPSCALE_KEY_FSHADINGCHANGESCALE, values.shadingChangeScale },
      { FFX_API_CONFIGURE_UPSCALE_KEY_FACCUMULATIONADDEDPERFRAME, values.accumulationAddedPerFrame },
      { FFX_API_CONFIGURE_UPSCALE_KEY_FMINDISOCCLUSIONACCUMULATION, values.minDisocclusionAccumulation },
    } };

    for (const std::pair<uint64_t, float>& key : keys) {
      float value = key.second;

      ffx::ConfigureDescUpscaleKeyValue keyValueDesc;
      keyValueDesc.key = key.first;
      keyValueDesc.ptr = &value;

      // FSR 3.1 has these keys. A version without them rejects the call, which changes nothing.
      FsrApi::get().configure(&context.handle, &keyValueDesc.header);
    }
  }

  bool DxvkFSR::dispatch(RtxContext* ctx, DxvkBarrierSet& barriers, const DispatchInputs& inputs) {
    ScopedCpuProfileZone();

    if (ctx == nullptr || inputs.pColor == nullptr || inputs.pDepth == nullptr ||
        inputs.pMotionVectors == nullptr || inputs.pOutput == nullptr) {
      return false;
    }

    if (inputs.pColor->image == nullptr || inputs.pDepth->image == nullptr ||
        inputs.pMotionVectors->image == nullptr || inputs.pOutput->image == nullptr) {
      return false;
    }

    if (inputs.renderExtent.width == 0 || inputs.renderExtent.height == 0 ||
        inputs.displayExtent.width == 0 || inputs.displayExtent.height == 0) {
      return false;
    }

    FsrApi& api = FsrApi::get();

    if (!api.isAvailable(FsrEffect::Upscaler)) {
      return false;
    }

    const std::array<const Resources::Resource*, 3> readInputs = { inputs.pColor, inputs.pDepth, inputs.pMotionVectors };

    for (const Resources::Resource* pInput : readInputs) {
      if (FsrApi::resourceState(pInput->image->info().layout) == 0) {
        ONCE(Logger::err("FSR: An upscaling input is in an image layout the FSR runtime cannot read."));
        return false;
      }
    }

    if (inputs.pOutput->image->info().layout != VK_IMAGE_LAYOUT_GENERAL) {
      ONCE(Logger::err("FSR: The upscaling output is in an image layout the FSR runtime cannot write."));
      return false;
    }

    // Exposure is left to FSR, which derives it from the colour input.
    ContextConfig config;
    config.renderExtent = inputs.renderExtent;
    config.displayExtent = inputs.displayExtent;
    config.flags = FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
    config.versionOverride = upscalerVersion();

    if (inputs.colorIsHdr) {
      config.flags |= FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE;
    }

    if (inputs.depthParameters.inverted) {
      config.flags |= FFX_UPSCALE_ENABLE_DEPTH_INVERTED;
    }

    if (inputs.depthParameters.infinite) {
      config.flags |= FFX_UPSCALE_ENABLE_DEPTH_INFINITE;
    }

    if (enableDebugChecker()) {
      config.flags |= FFX_UPSCALE_ENABLE_DEBUG_CHECKING;
    }

    Context& context = m_contexts[inputs.colorIsHdr ? 1 : 0];

    if (!ensureContext(context, config)) {
      return false;
    }

    applyTuning(context);

    {
      std::lock_guard<std::mutex> lock(m_activeVersionMutex);
      m_activeVersion = context.version;
    }

    // What wrote the inputs has to be done before FSR's compute passes read them.
    for (const Resources::Resource* pInput : readInputs) {
      barriers.accessImage(
        pInput->image,
        pInput->view->imageSubresources(),
        pInput->image->info().layout,
        pInput->image->info().stages,
        pInput->image->info().access,
        pInput->image->info().layout,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);
    }

    barriers.accessImage(
      inputs.pOutput->image,
      inputs.pOutput->view->imageSubresources(),
      inputs.pOutput->image->info().layout,
      inputs.pOutput->image->info().stages,
      inputs.pOutput->image->info().access,
      inputs.pOutput->image->info().layout,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT);

    barriers.recordCommands(ctx->getCommandList());

    float cameraNear = 0.0f;
    float cameraFar = 0.0f;
    fsr::cameraPlanes(inputs.depthParameters, cameraNear, cameraFar);

    const float sharpnessValue = std::clamp(sharpness(), 0.0f, 1.0f);

    ffx::DispatchDescUpscale dispatchDesc;
    dispatchDesc.commandList = ctx->getCommandList()->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);
    dispatchDesc.color = FsrApi::resource(inputs.pColor->image, FsrApi::resourceState(inputs.pColor->image->info().layout), false);
    dispatchDesc.depth = FsrApi::resource(inputs.pDepth->image, FsrApi::resourceState(inputs.pDepth->image->info().layout), false);
    dispatchDesc.motionVectors = FsrApi::resource(inputs.pMotionVectors->image, FsrApi::resourceState(inputs.pMotionVectors->image->info().layout), false);
    dispatchDesc.output = FsrApi::resource(inputs.pOutput->image, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS, true);
    dispatchDesc.jitterOffset = { inputs.jitter[0], inputs.jitter[1] };
    // The motion vectors are already in render pixels.
    dispatchDesc.motionVectorScale = { 1.0f, 1.0f };
    dispatchDesc.renderSize = { inputs.renderExtent.width, inputs.renderExtent.height };
    dispatchDesc.upscaleSize = { inputs.displayExtent.width, inputs.displayExtent.height };
    dispatchDesc.enableSharpening = sharpnessValue > 0.0f;
    dispatchDesc.sharpness = sharpnessValue;
    dispatchDesc.frameTimeDelta = fsr::frameTimeMs(inputs.frameTimeMs);
    dispatchDesc.preExposure = 1.0f;
    dispatchDesc.reset = inputs.resetHistory;
    dispatchDesc.cameraNear = cameraNear;
    dispatchDesc.cameraFar = cameraFar;
    dispatchDesc.cameraFovAngleVertical = inputs.verticalFov;
    dispatchDesc.viewSpaceToMetersFactor = viewSpaceToMetersFactor();
    dispatchDesc.flags = 0;

    const ffxReturnCode_t result = api.dispatch(&context.handle, &dispatchDesc.header);

    // FSR leaves every image in the layout it found it in.
    barriers.accessImage(
      inputs.pOutput->image,
      inputs.pOutput->view->imageSubresources(),
      inputs.pOutput->image->info().layout,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT,
      inputs.pOutput->image->info().layout,
      inputs.pOutput->image->info().stages,
      inputs.pOutput->image->info().access);

    barriers.recordCommands(ctx->getCommandList());

    for (const Resources::Resource* pInput : readInputs) {
      ctx->getCommandList()->trackResource<DxvkAccess::Read>(pInput->image);
    }

    ctx->getCommandList()->trackResource<DxvkAccess::Write>(inputs.pOutput->image);

    if (result != FFX_API_RETURN_OK) {
      ONCE(Logger::err(str::format("FSR: Upscaling failed: ", FsrApi::returnCodeName(result), ".")));
      return false;
    }

    return true;
  }

} // namespace dxvk
