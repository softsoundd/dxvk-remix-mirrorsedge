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
#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "rtx_common_object.h"
#include "rtx_fsr_helpers.h"
#include "rtx_options.h"
#include "rtx_resources.h"

// Declared by the FSR headers, which only the implementation includes.
typedef void* ffxContext;

namespace dxvk {

  class DxvkBarrierSet;
  class DxvkDevice;
  class RtxContext;

  /**
   * \brief AMD FSR upscaling
   *
   * Upscales the NGX passthrough's scene colour with the AMD FSR runtime that ships next to this DLL.
   * The runtime picks the newest FSR version the GPU supports, unless rtx.fsr.upscalerVersion names one.
   */
  class DxvkFSR : public CommonDeviceObject {
  public:
    struct DispatchInputs {
      // Holds linear colour at render resolution.
      const Resources::Resource* pColor = nullptr;
      // Holds the scene depth at render resolution, in the convention depthParameters describes.
      const Resources::Resource* pDepth = nullptr;
      // Holds the motion from each pixel to its position in the previous frame, in render pixels.
      const Resources::Resource* pMotionVectors = nullptr;
      // Receives the upscaled colour at display resolution.
      const Resources::Resource* pOutput = nullptr;

      VkExtent2D renderExtent = { 0, 0 };
      VkExtent2D displayExtent = { 0, 0 };

      // Holds the sub-pixel offset the frame was rasterized with, in render pixels.
      float jitter[2] = { 0.0f, 0.0f };

      // Set when the colour can exceed 1, as scene colour does ahead of tone mapping.
      bool colorIsHdr = false;

      fsr::DepthParameters depthParameters;
      // Holds the vertical field of view in radians.
      float verticalFov = 0.0f;

      float frameTimeMs = 0.0f;
      bool resetHistory = false;
    };

    explicit DxvkFSR(DxvkDevice* device);
    ~DxvkFSR() override;

    void onDestroy() override;

    // Returns whether the FSR runtime's upscaler is installed next to this DLL.
    // This reads the disk only, so it is safe to call before a device exists.
    static bool hasRuntimeFiles();

    // Returns whether FSR upscaling can run on this system.
    bool isAvailable() const;

    // Upscales the colour input into the output.
    // The contexts are created on first use, and recreated when the sizes or the depth convention change.
    bool dispatch(RtxContext* ctx, DxvkBarrierSet& barriers, const DispatchInputs& inputs);

    // Destroys the upscaling contexts, which hold several display-sized images each.
    void releaseContexts();

    // Returns the names of the FSR versions the runtime offers on this device, newest first.
    // The first call loads the runtime. Any thread may call this.
    std::vector<std::string> versionNames() const;

    // Returns the name of the FSR version the last dispatch ran.
    // Returns an empty string while there is none.
    // Any thread may call this.
    std::string activeVersion() const;

    static float calcScreenPercentageForPreset(FSRPreset preset) {
      return fsr::presetScreenPercentage(preset);
    }

    RTX_OPTION_ARGS("rtx.fsr", FSRPreset, preset, FSRPreset::NativeAA,
                    "Sets the AMD FSR quality mode, which trades image quality for performance.\n"
                    "The game renders at the mode's fraction of the output resolution, and FSR upscales the result.\n"
                    "0: Ultra Performance (3.0x), 1: Performance (2.0x), 2: Balanced (1.7x), 3: Quality (1.5x), 4: Native AA (1.0x).",
                    args.minValue = FSRPreset::UltraPerformance, args.maxValue = FSRPreset::NativeAA,
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.fsr", float, sharpness, 0.25f,
                    "Sets the strength of the sharpening pass FSR runs on its output, from 0 to 1.\n"
                    "0 turns the pass off.",
                    args.minValue = 0.0f, args.maxValue = 1.0f,
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.fsr", float, mipBiasOffset, -1.0f,
                    "Offsets the texture LOD bias of the scene while FSR is the upscaler, from -2 to 0.\n"
                    "The bias is log2(render width / display width) plus this offset, and AMD recommends an offset of -1.\n"
                    "A lower value sharpens textures, and can add shimmer on fine detail. 0 leaves only the resolution term.",
                    args.minValue = -2.0f, args.maxValue = 0.0f);
    RTX_OPTION_ARGS("rtx.fsr", float, viewSpaceToMetersFactor, 0.01f,
                    "Sets how many meters one unit of the game's view space is.\n"
                    "Unreal Engine 3 units are centimeters, which makes this 0.01.",
                    args.minValue = 0.0001f, args.maxValue = 100.0f);
    RTX_OPTION("rtx.fsr", bool, enableDebugChecker, false,
               "Makes FSR validate the inputs of every upscale, and writes what it finds to the log.\n"
               "This is a development aid, and changing it recreates the upscaler.");
    RTX_OPTION("rtx.fsr", std::string, upscalerVersion, "",
               "Names the FSR upscaler version to use, as the developer menu lists them, for example \"3.1.6\".\n"
               "Leave this empty to let the runtime pick the newest version the GPU supports.\n"
               "A named version also keeps the runtime from using a newer upscaler supplied by the graphics driver.");
    RTX_OPTION_ARGS("rtx.fsr", float, velocityFactor, 1.0f,
                    "Scales how much motion reduces the weight of the history, from 0 to 1.\n"
                    "A lower value can steady bright pixels in motion. Applies to FSR 3.1.",
                    args.minValue = 0.0f, args.maxValue = 1.0f);
    RTX_OPTION_ARGS("rtx.fsr", float, reactivenessScale, 1.0f,
                    "Scales how quickly FSR replaces the history where it detects reactive content.\n"
                    "A higher value reduces ghosting, and can add flicker. Applies to FSR 3.1.",
                    args.minValue = 0.0f, args.maxValue = 8.0f);
    RTX_OPTION_ARGS("rtx.fsr", float, shadingChangeScale, 1.0f,
                    "Scales how strongly FSR reacts to shading that changes between frames.\n"
                    "A higher value reduces ghosting of moving lights and shadows. Applies to FSR 3.1.",
                    args.minValue = 0.0f, args.maxValue = 8.0f);
    RTX_OPTION_ARGS("rtx.fsr", float, accumulationAddedPerFrame, 0.333f,
                    "Sets how much history a newly revealed pixel gains per frame, from 0 to 1.\n"
                    "A lower value reduces ghosting, and can make thin features flicker. Applies to FSR 3.1.",
                    args.minValue = 0.0f, args.maxValue = 1.0f);
    RTX_OPTION_ARGS("rtx.fsr", float, minDisocclusionAccumulation, -0.333f,
                    "Sets the history a pixel starts from when it is revealed, from -1 to 1.\n"
                    "A higher value reduces flicker around thin moving objects, and can add ghosting. Applies to FSR 3.1.",
                    args.minValue = -1.0f, args.maxValue = 1.0f);

  private:
    // Describes what a context was created for.
    // A change in any of these needs a new context.
    struct ContextConfig {
      VkExtent2D renderExtent = { 0, 0 };
      VkExtent2D displayExtent = { 0, 0 };
      uint32_t flags = 0;
      std::string versionOverride;

      bool operator==(const ContextConfig& other) const {
        return renderExtent.width == other.renderExtent.width && renderExtent.height == other.renderExtent.height &&
               displayExtent.width == other.displayExtent.width && displayExtent.height == other.displayExtent.height &&
               flags == other.flags && versionOverride == other.versionOverride;
      }
    };

    // Holds the values of the tuning options a context was last given.
    struct TuningValues {
      float velocityFactor = 1.0f;
      float reactivenessScale = 1.0f;
      float shadingChangeScale = 1.0f;
      float accumulationAddedPerFrame = 0.333f;
      float minDisocclusionAccumulation = -0.333f;

      bool operator==(const TuningValues& other) const {
        return velocityFactor == other.velocityFactor && reactivenessScale == other.reactivenessScale &&
               shadingChangeScale == other.shadingChangeScale &&
               accumulationAddedPerFrame == other.accumulationAddedPerFrame &&
               minDisocclusionAccumulation == other.minDisocclusionAccumulation;
      }
    };

    struct Context {
      ffxContext handle = nullptr;
      ContextConfig config;
      // Set once creation has failed for the config, so it is not retried every frame.
      bool failed = false;
      std::string version;
      TuningValues tuning;
    };

    bool ensureContext(Context& context, const ContextConfig& config);
    bool createContext(Context& context, const ContextConfig& config);
    void destroyContext(Context& context);
    void applyTuning(Context& context);

    // HDR and SDR colour need contexts of their own, and the passthrough can alternate between the two.
    // Each one keeps its own history, so switching does not cost a reset.
    Context m_contexts[2];

    // The dispatch writes the version on the render thread, and the menus read it on the application's thread.
    mutable std::mutex m_activeVersionMutex;
    std::string m_activeVersion;
  };

} // namespace dxvk
