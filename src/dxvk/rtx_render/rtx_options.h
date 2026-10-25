/*
* Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
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

#include <algorithm>
#include <unordered_set>
#include <cassert>
#include <limits>
#include <sstream>
#include <iomanip>

#include "../util/util_keybind.h"
#include "../util/config/config.h"
#include "../util/xxHash/xxhash.h"
#include "../util/util_math.h"
#include "../util/util_env.h"
#include "rtx_utils.h"
#include "rtx_dlss.h"
#include "rtx_gpu_overrides.h"
#include "rtx_gpu_profile.h"
#include "rtx_graphics_preset.h"
#include "rtx_option.h"
#include "rtx_option_manager.h"
enum _NV_GPU_ARCHITECTURE_ID;
typedef enum _NV_GPU_ARCHITECTURE_ID NV_GPU_ARCHITECTURE_ID;
enum _NV_GPU_ARCH_IMPLEMENTATION_ID;
typedef enum _NV_GPU_ARCH_IMPLEMENTATION_ID NV_GPU_ARCH_IMPLEMENTATION_ID;

// RTX specific options

namespace dxvk {
  class DxvkDevice;

  enum class UpscalerType : int {
    None = 0,
    DLSS,
    NIS,
    TAAU,
    XeSS
  };

  enum class DlssPreset : int {
    Off = 0,
    On,
    Custom
  };

  enum class XeSSPreset : int {
    UltraPerf = 0,
    Performance,
    Balanced,
    Quality,
    UltraQuality,
    UltraQualityPlus,
    NativeAA,
    Custom,
    Invalid
  };

  enum class NisPreset : int {
    Performance = 0,
    Balanced,
    Quality,
    Fullscreen
  };

  enum class TaauPreset : int {
    UltraPerformance = 0,
    Performance,
    Balanced,
    Quality,
    Fullscreen
  };

  enum class UIType : int {
    None = 0,
    Basic,
    Advanced,
    FirstUseGuide,
    Count
  };

  enum class ReflexMode : int {
    None = 0,
    LowLatency,
    LowLatencyBoost
  };

  enum class EnableVsync : int {
    Off = 0,
    On = 1,
    WaitingForImplicitSwapchain = 2   // waiting for the app to create the device + implicit swapchain, we latch the vsync setting from there
  };

  class RtxOptions {

    RTX_OPTION("rtx", fast_unordered_set, uiTextures, {},
                  "Textures on draw calls that should be treated as screenspace UI elements.\n"
                  "When no pre-post-process injection point is found, the first draw using one of these textures triggers the late NGX injection, which runs before the UI.");
    
  public:
    // Note: This time is in milliseconds, should be named something like millisecondDeltaBetweenFrames ideally, but keeping it as it is for now.
    RTX_OPTION_ENV("rtx", float, timeDeltaBetweenFrames, 0.f, "RTX_FRAME_TIME_DELTA_MS",
                   "Frame time delta in milliseconds to use for rendering.\n"
                   "Setting this to 0 will use actual frame time delta for a given frame. Non-zero value allows the actual time delta to be overridden and is primarily used for automation to ensure determinism run to run without variance due to frame time fluctuations.");

    RTX_OPTION_ARGS("rtx", DlssPreset, dlssPreset, DlssPreset::On, "Combined DLSS Preset for quickly controlling Upscaling, Frame Interpolation and Latency Reduction.",
                    args.environment = "RTX_DLSS_PRESET",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx", NisPreset, nisPreset, NisPreset::Balanced, "Adjusts NIS scaling factor, trades quality for performance.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx", TaauPreset, taauPreset, TaauPreset::Balanced,  "Adjusts TAA-U scaling factor, trades quality for performance.",
                    args.flags = RtxOptionFlags::UserSetting);
    static void graphicsPresetOnChange(DxvkDevice* device);
    RTX_OPTION_ARGS("rtx", GraphicsPreset, graphicsPreset, GraphicsPreset::Auto, "Overall rendering preset, higher presets result in higher image quality, lower presets result in better performance.",
                    args.environment = "DXVK_GRAPHICS_PRESET_TYPE",
                    args.onChangeCallback = &graphicsPresetOnChange,
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION("rtx", bool, allowFSE, false,
               "A flag indicating if the application should be able to utilize exclusive full screen mode when set to true, otherwise force it to be disabled when set to false.\n"
               "Exclusive full screen may see performance benefits over other fullscreen modes at the cost of stability in some cases.\n"
               "Do note that on modern Windows full screen optimizations will likely be used regardless which in most cases results in performance similar to exclusive full screen even when it is not in use.");
    RTX_OPTION("rtx", std::string, baseGameModRegex, "", "Regex used to determine if the base game is running a mod, like a sourcemod.");
    RTX_OPTION("rtx", std::string, baseGameModPathRegex, "", "Regex used to redirect RTX Remix Runtime to another path for rtx.conf.");
    RTX_OPTION("rtx", bool, disableAMDSwitchableGraphics, true,
               "A flag indicating if Remix should attempt to disable AMD's switchable graphics Vulkan layer (VK_LAYER_AMD_swichable_graphics).\n"
               "Due to how some older AMD drivers filter devices exposed to Vulkan it is possible for Remix to see no valid GPUs on a machine when using an integerated AMD GPU with a dedicated Nvidia GPU (for instance a laptop).\n"
               "This is because on such machines both Nvidia Optimus and AMD switchable graphics attempt to filter the device list to promote their respective GPUs, but rather than leaving at least one device all end up filtered out.\n"
               "To work around this issue, Remix can attempt to disable the AMD switchable graphics layer which should eliminate this buggy filtering. As such, this option should generally remain enabled.\n"
               "If this causes an undesired GPU to be selected (e.g. if for some reason you want to force Remix to run on an integerated AMD GPU via the switchable graphics layer), then this option should be disabled.");


    // Shader Compilation
    struct Shader {
      friend class ShaderManager;

      // Note: Shader recompilation is only useful with a development setup for the most part and is disabled when REMIX_DEVELOPMENT is not defined,
      // so these options will not take effect in such builds. They are however still included rather than ifdeffed out to keep consistent options documentation
      // across builds.
      RTX_OPTION("rtx.shader", bool, asyncSpirVRecompilation, true,
                 "When set to true runtime shader recompilation will recompile shaders to SPIR-V asynchronously rather than blocking until complete.\n"
                 "Do note that despite setting this option the actual compilation of the shader from SPIR-V to the ISA will still be blocking as only the prewarming process can handle this step asynchronously for now.\n"
                 "Generally this option should remain enabled, though disabling it may be useful for CI where deterministic behavior is needed, and may be useful to maximize performance at the cost of blocking (by not having application running while compiling to SPIR-V).\n"
                 "This option is mainly meant for development use and should not be set for user-facing operation.");
      RTX_OPTION("rtx.shader", bool, recompileOnLaunch, false,
                 "When set to true runtime shader recompilation will execute on the first frame after launch.\n"
                 "This option is mainly meant for development use and should not be set for user-facing operation. Also see rtx.useLiveShaderEditMode for a similar option which auto-detects shader changes instead.");
      RTX_OPTION("rtx.shader", bool, useLiveEditMode, false,
                 "When set to true shaders will be automatically recompiled when any shader file is updated (saved for instance) in addition to the usual manual recompilation trigger.\n"
                 "This option is mainly meant for development use and should not be set for user-facing operation.");

      RTX_OPTION_ENV("rtx.shader", bool, enableAsyncCompilation, true, "RTX_ENABLE_ASYNC_COMPILATION",
                 "When set to true shader compilation (especially that of prewarming) will be done asynchronously rather than blocking.\n"
                 "Typically shader prewarming with async finalization is done to attempt to compile all required shader variants before they are used, often by overlapping this work with a startup sequence (e.g. a game's loading screen). Often times however this prewarming takes longer than the time available, or an application may not have a startup sequence to begin with and immediately begin using Remix shaders.\n"
                 "To accomodate this, async shader compilation allows for this work to be done asynchronously to avoid blocking the application at the cost of being unable to render anything until the process is complete.\n"
                 "This is typically better choice than blocking however and is recommended to be enabled as on Windows Remix blocking will cause the application to stop responding, making it seem as if the application has crashed if shader compilation takes a long time. Additionally, when combined with rtx.shader.enableAsyncCompilationUI the progress of the compilation process can be shown to the user as a UI, improving user experience.\n"
                 "The main downside to this approach is that when blocking shader compilation is allowed to take up more of the CPU, whereas async shader compilation will have to compete with the application which can make compilation take slightly longer than it would otherwise (especially true if the application's framerate is uncapped).\n"
                 "Finally, a more minor downside is that when async shader compilation is in use Remix currently has no way of keeping the application in a startup sequence (e.g. keeping a game on its loading screen) while it waits for shaders to compile.\n"
                 "This will mean for instance a game's menu may be active but not be able to render until the compilation is complete, rather than blocking on the loading screen and transitioning to the menu only once all shaders are loaded. Not blocking the application is typically better for user experience regardless though as long as some sort of progress UI is displayed to indicate what is happening.");
      RTX_OPTION("rtx.shader", bool, enableAsyncCompilationUI, true,
                 "Enables a UI message when async shader compilation is in progress to indicate the current compilation progress. Only takes effect when rtx.shader.enableAsyncCompilation is true.\n"
                 "This should usually be enabled as providing information to the user about the current progress of compilation is useful. May be disabled however for automated testing purposes if the nondeterministic behavior of the UI's rendered text interferes with testing.");
    } shader;

    // Camera
    RTX_OPTION("rtx", bool, enableNearPlaneOverride, false,
               "A flag to enable or disable the Camera's near plane override feature.\n"
               "Since the camera is not used directly for ray tracing the near plane the application uses typically does not matter, but for certain matrix-based operations (such as temporal reprojection or voxel grid projection) it is still relevant.\n"
               "The issue arises when geometry is ray traced that is behind where the chosen Camera's near plane is located, typically common on viewmodels especially with how they are ray traced, causing graphical artifacts and other issues.\n"
               "This option helps correct this issue by overriding the near plane value to else (usually smaller) to sit behind the objects in question (such as the view model). As such this option should usually be enabled on games with viewmodels.\n"
               "Do note that when adjusting the near plane the larger the relative magnitude gap between the near and far plane the worse the precision of matrix operations will be, so the near plane should be set as high as possible even when overriding.");
    RTX_OPTION("rtx", float, nearPlaneOverride, 0.1f,
               "The near plane value to use for the Camera when the near plane override is enabled.\n"
               "Only takes effect when rtx.enableNearPlaneOverride is enabled, see that option for more information about why this is useful.");

    RTX_OPTION_ARGS("rtx", UpscalerType, upscalerType, UpscalerType::DLSS, "Upscaling boosts performance with varying degrees of image quality tradeoff depending on the type of upscaler and the quality mode/preset.",
                    args.environment = "DXVK_UPSCALER_TYPE",
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx", float, resolutionScale, 0.75f, "",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION("rtx", bool, forceCameraJitter, false, "Force enables camera jitter frame to frame.");
    RTX_OPTION("rtx", uint32_t, cameraJitterSequenceLength, 64, "Sets a camera jitter sequence length [number of frames]. It will loop around once the length is reached.");

    // Needs to be > 0
    RTX_OPTION_ARGS("rtx", float, uniqueObjectDistance, 300.f, "The distance (in game units) the camera can move in a single frame before the move is treated as a camera cut, which resets temporal history.",
                    args.minValue = 0.f);

    RTX_OPTION("rtx", bool, useNewGuiInputMethod, true, "Disables the previous method for getting mouse/keyboard input and enables a new method which should be more reliable.  If successful the old method will be deprecated.  This setting can't be changed at runtime, so it must be set in a .conf file.");

    RTX_OPTION_ARGS("rtx", UIType, showUI, UIType::None, "0 = Don't Show, 1 = Show Simple, 2 = Show Advanced, 3 = First Use Guide.",
                    args.environment = "RTX_GUI_DISPLAY_UI",
                    args.flags = RtxOptionFlags::NoSave | RtxOptionFlags::NoReset);
    RTX_OPTION_ARGS("rtx", bool, defaultToAdvancedUI, false, "Whether to default to the Advanced UI when opening the developer menu.", 
                    args.flags = RtxOptionFlags::UserSetting | RtxOptionFlags::NoReset);

    public: static void showUICursorOnChange(DxvkDevice* device);
    RTX_OPTION_ARGS("rtx", bool, showUICursor, true, "If true, the ImGUI mouse cursor will be shown when the UI is active.\n"
                    "Can be toggled with Alt + Delete.", args.onChangeCallback = &showUICursorOnChange);
    
    public: static void blockInputToGameInUIOnChange(DxvkDevice* device);
    RTX_OPTION_ARGS("rtx", bool, blockInputToGameInUI, true,
                    "If true, input will not be passed to the game when the UI is active.\n"
                    "Can be toggled with Alt + Backspace", args.onChangeCallback = &blockInputToGameInUIOnChange, args.flags = RtxOptionFlags::NoSave);

    RTX_OPTION_ARGS("rtx", bool, restoreCursorPosition, false,
                    "If true, the game's mouse cursor position will be restored when the Remix UI is closed.\n"
                    "This should fix the issue where the game camera suddenly turns when closing the UI.\n",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx", bool, autoUnblockOptionEdits, false,
                    "If true, editing an RtxOption in the Remix UI that is overridden by a stronger config layer clears the stronger value immediately instead of showing a confirmation dialog.",
                    args.environment = "RTX_IMGUI_AUTO_UNBLOCK_OPTION_EDITS",
                    args.flags = RtxOptionFlags::UserSetting);

    inline static const VirtualKeys kDefaultRemixMenuKeyBinds{ VirtualKey{VK_MENU},VirtualKey{'X'} };
    RTX_OPTION("rtx", VirtualKeys, remixMenuKeyBinds, kDefaultRemixMenuKeyBinds,
               "Hotkey to open the Remix menu.\n"
               "example override: 'rtx.remixMenuKeyBinds = CTRL, SHIFT, Z'.\n"
               "Full list of key names available in `src/util/util_keybind.h`.");

    RTX_OPTION_ARGS("rtx", DLSSProfile, qualityDLSS, DLSSProfile::FullResolution, "Adjusts internal DLSS scaling factor, trades quality for performance.",
                    args.environment = "RTX_QUALITY_DLSS",
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION("rtx", bool, rngSeedWithFrameIndex, true,
               "Indicates that pseudo-random number generator should be seeded with the frame number of the application every frame, otherwise seed with 0.\n"
               "This should generally always be enabled as without the frame index each frame will typically be identical in the random values that are produced which will result in incorrect rendering. Only meant as a debugging tool.");
    // declare onAdvanceTimeChanged
    static void onAdvanceTimeChanged(DxvkDevice* device);
    RTX_OPTION_ARGS("rtx", bool, advanceTime, true,
                    "A flag to enable or disable advancing time used by Remix subsystems (particle effects, animations, etc.).\n",
                    args.environment = "RTX_ADVANCE_TIME",
                    args.onChangeCallback = &RtxOptions::onAdvanceTimeChanged);

    RTX_OPTION("rtx", float, maxAnisotropySamples, 8.0f,
               "The maximum number of samples to use when anisotropic filtering is enabled.\n"
               "The actual max anisotropy used will be the minimum between this value and the hardware's maximum. Higher values increase quality but will likely reduce performance.");

    // Developer Options
    RTX_OPTION_FLAG_ENV("rtx", bool, enableBreakIntoDebuggerOnPressingB, false, RtxOptionFlags::NoSave, "RTX_BREAK_INTO_DEBUGGER_ON_PRESSING_B",
                    "Enables a break into a debugger at the start of InjectRTX() on a press of key \'B\'.\n"
                    "If debugger is not attached at the time, it will wait until a debugger is attached and break into it then.");
    
    // Crash Hotkey Feature (Development builds only)
    // When enabled via checkbox in the Development tab, pressing the configured hotkey will trigger a deliberate crash.
    // This is useful for testing crash handling, crash dumps, and crash reporting systems.
    // The feature is only available in REMIX_DEVELOPMENT builds and defaults to disabled.
    // The checkbox state is not saved to config files (NoSave), but can be pre-enabled by setting rtx.enableCrashHotkey = True in rtx.conf.
    RTX_OPTION_FLAG_ENV("rtx", bool, enableCrashHotkey, false, RtxOptionFlags::NoSave, "RTX_ENABLE_CRASH_HOTKEY",
                    "Arms the crash hotkey feature. When enabled, pressing the crash hotkey combination (Ctrl+Shift+Alt+K by default) will trigger a deliberate crash.\n"
                    "This option is only available in development builds and is intended for testing crash handling and crash dump generation.\n"
                    "The armed state is indicated by a red warning overlay on screen. This setting is not saved to config files but can be set manually in rtx.conf.");
    inline static const VirtualKeys kDefaultCrashHotkey{ VirtualKey{VK_CONTROL}, VirtualKey{VK_SHIFT}, VirtualKey{VK_MENU}, VirtualKey{'K'} };
    RTX_OPTION_FLAG("rtx", VirtualKeys, crashHotkey, kDefaultCrashHotkey, RtxOptionFlags::NoSave,
                    "The hotkey combination that triggers a deliberate crash when the crash hotkey feature is armed.\n"
                    "Default is Ctrl+Shift+Alt+K. Only takes effect when rtx.enableCrashHotkey is True.\n"
                    "This setting is not saved to config files but can be set manually in rtx.conf.");
    // GPU crash hotkey - same "armed" state as crashHotkey. When armed, this key triggers a GPU crash (dialog + Sentry).
    inline static const VirtualKeys kDefaultGpuCrashHotkey{ VirtualKey{VK_CONTROL}, VirtualKey{VK_SHIFT}, VirtualKey{VK_MENU}, VirtualKey{'G'} };
    RTX_OPTION_FLAG("rtx", VirtualKeys, gpuCrashHotkey, kDefaultGpuCrashHotkey, RtxOptionFlags::NoSave,
                    "The hotkey that triggers a GPU crash when the crash hotkey feature is armed.\n"
                    "Default is Ctrl+Shift+Alt+G.");

    RTX_OPTION("rtx", bool, enablePresentThrottle, false,
               "A flag to enable or disable present throttling, when set to true a sleep for a time specified by the throttle delay will be inserted into the DXVK presentation thread.\n"
               "Useful to manually reduce the framerate if the application is running too fast or to reduce GPU power usage during development to keep temperatures down.\n"
               "Should not be enabled in anything other than development situations.");
    RTX_OPTION("rtx", std::uint32_t, presentThrottleDelay, 16U,
               "A time in milliseconds that the DXVK presentation thread should sleep for. Requires present throttling to be enabled to take effect.\n"
               "Note that the application may sleep for longer than the specified time as is expected with sleep functions in general.");
    RTX_OPTION_ARGS("rtx", bool, recomputeTextureHashOnWrite, false,
                "When true, Remix computes the hash of a texture when the game writes to the resource. Some games manage their own pool of\n"
                "textures and shuffle data around those resources resulting in incorrect textures being displayed. Recomputing the hash\n"
                "when the game writes to the resource can resolve this issue, however this can have unintended side effects when replacing\n"
                "animated game textures.");
    // Note: Use use areValidationLayersEnabled helper function rather than accessing this option directly as additional logic must be done to determine if validation layers should be used or not.
    RTX_OPTION_FLAG_ENV("rtx", bool, enableValidationLayers, false, RtxOptionFlags::NoSave, "DXVK_ENABLE_VALIDATION_LAYERS",
                        "A flag to enable validation layers in Vulkan. Note that in Debug builds validation layers will always be enabled and this flag will have no effect.\n"
                        "Enabling validation layers is useful for debugging and development to catch common issues in Vulkan, but will reduce overall performance.\n"
                        "Should only be enabled by developers during development and not put into production builds of any project.\n"
                        "Additionally, this setting must be set at startup and changing it will not take effect at runtime.");
    RTX_OPTION_FLAG_ENV("rtx", bool, enableValidationLayerExtendedValidation, false, RtxOptionFlags::NoSave, "DXVK_ENABLE_VALIDATION_LAYER_EXTENDED_VALIDATION",
                        "A flag to enable extended validation to validation layers in Vulkan. Only takes effect if validation layers are enabled already.\n"
                        "This flag enables GPU assisted and synchronization validation along with best practices within the Vulkan validation layers which allow for greater error-checking capability at the cost of significant performance impact.\n"
                        "Much like the rtx.enableValidationLayers option, this option should only be enabled by developers during development and not be put into production builds of any project.\n"
                        "Additionally, this setting must be set at startup and changing it will not take effect at runtime.");
    RTX_OPTION_FLAG_ENV("rtx", bool, logCallstacksOnValidationLayerErrors, true, RtxOptionFlags::NoSave, "DXVK_LOG_CALLSTACKS_ON_VALIDATION_LAYER_ERRORS",
                        "A flag to enable logging of callstacks when validation layer errors occur.\n"
                        "This is useful for debugging and development to help track down the source of validation layer errors more easily.\n"
                        "Requires pdb symbols to be present next to Remix's d3d9 dll and/or in the working directory to resolve symbols.");

    RTX_OPTION_ARGS("rtx", ReflexMode, reflexMode, ReflexMode::LowLatency,
               "Reflex mode selection, enabling it helps minimize input latency, boost mode may further reduce latency by boosting GPU clocks in CPU-bound cases.\n"
               "Supported enum values are 0 = None (Disabled), 1 = LowLatency (Enabled), 2 = LowLatencyBoost (Enabled + Boost).\n"
               "Note that even when using the \"None\" Reflex mode Reflex will attempt to be initialized. Use rtx.isReflexEnabled to fully disable to skip this initialization if needed.",
               args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_FLAG("rtx", bool, isReflexEnabled, true, RtxOptionFlags::NoSave,
                    "Enables or disables Reflex globally.\n"
                    "Note that this option when set to false will prevent Reflex from even attempting to initialize, unlike setting the Reflex mode to \"None\" which simply tells an initialized Reflex not to take effect.\n"
                    "Additionally, this setting must be set at startup and changing it will not take effect at runtime.");

    // Store the computed value separately from the user preference.  This enables changing it immediately when needed,
    // and lets us store the final value to be used by the game.
    public: inline static EnableVsync enableVsyncState = EnableVsync::WaitingForImplicitSwapchain;
    public: static void EnableVsyncOnChange(DxvkDevice* device) {
      if (enableVsync() != EnableVsync::WaitingForImplicitSwapchain) {
        enableVsyncState = enableVsync();
      }
      // If the option is changed to WaitingForImplicitSwapchain, just leave the computed state as it was.
    }
    RTX_OPTION_ARGS("rtx", EnableVsync, enableVsync, EnableVsync::WaitingForImplicitSwapchain,
                    "Controls V-Sync: 0 forces it off and 1 forces it on. The default (2) adopts the game's setting when it "
                    "creates its device; only D3DPRESENT_INTERVAL_IMMEDIATE counts as off, and d3d9.presentInterval takes "
                    "precedence when set. Later changes to the game's setting are ignored.",
                    args.flags = RtxOptionFlags::NoSave | RtxOptionFlags::UserSetting,
                    args.onChangeCallback = &EnableVsyncOnChange);

    // Automation Options
    struct Automation {
      RTX_OPTION_FLAG_ENV("rtx.automation", bool, disableBlockingDialogBoxes, false, RtxOptionFlags::NoSave, "RTX_AUTOMATION_DISABLE_BLOCKING_DIALOG_BOXES",
                          "Disables various blocking blocking dialog boxes (such as popup windows) requiring user interaction when set to true, otherwise uses default behavior when set to false.\n"
                          "This option is typically meant for automation-driven execution of Remix where such dialog boxes if present may cause the application to hang due to blocking waiting for user input.");
      RTX_OPTION_FLAG_ENV("rtx.automation", bool, disableDisplayMemoryStatistics, false, RtxOptionFlags::NoSave, "RTX_AUTOMATION_DISABLE_DISPLAY_MEMORY_STATISTICS",
                          "Disables display of memory statistics in the Remix window.\n"
                          "This option is typically meant for automation of tests for which we don't want non-deterministic runtime memory statistics to be shown in GUI that is included as part of test image output.");
      RTX_OPTION_FLAG_ENV("rtx.automation", bool, disableUpdateUpscaleFromDlssPreset, false, RtxOptionFlags::NoSave, "RTX_AUTOMATION_DISABLE_UPDATE_UPSCALER_FROM_DLSS_PRESET",
                          "Disables updating upscaler from DLSS preset.\n"
                          "This option is typically meant for automation of tests for which we don't want upscaler to be updated based on a DLSS preset.");
      RTX_OPTION_FLAG_ENV("rtx.automation", bool, enableTestTrace, false, RtxOptionFlags::NoSave, "RTX_TEST_TRACE",
                          "Enables opt-in frame trace artifacts for automation-driven image tests.\n"
                          "When enabled, Remix records a bounded frame window around the configured screenshot frame, writes frame_trace.jsonl, and appends dxvk_trace_* summary fields to metrics.txt.");
      RTX_OPTION_FLAG_ENV("rtx.automation", uint32_t, metricsWarmupFrames, 5, RtxOptionFlags::NoSave, "RTX_AUTOMATION_METRICS_WARMUP_FRAMES",
                          "Number of frames at the start of the run excluded as warmup from the averaged frame time, GPU idle time and memory usage in metrics.txt.\n"
                          "0 disables warmup, so every frame is averaged. dxvk_warmup_time_ms reports the dxvk_total_time_ms clock at the last warmup sample.");
    };

    RTX_OPTION("rtx", uint32_t, applicationId, 102100511, "Used to uniquely identify the application to DLSS. Generally should not be changed without good reason.");

    static RtxOptions* s_instance;

  public:

    RtxOptions(const RtxOptions&) = delete;
    RtxOptions(RtxOptions&&) = delete;
    RtxOptions& operator=(const RtxOptions&) = delete;
    RtxOptions& operator=(RtxOptions&&) = delete;

  private:

    RtxOptions() {
      Logger::info("Initializing RtxOptions...");

      // Optionally write documentation (captures code-defined defaults from RTX_OPTION macros)
      if (env::getEnvVar("DXVK_DOCUMENTATION_WRITE_RTX_OPTIONS_MD") == "1") {
        RtxOptionManager::writeMarkdownDocumentation("RtxOptions.md");
      }

      // Registered before layers load so their keys are created as the layers are applied.
      GpuOverrides::registerNamespaces();

      // Initialize all system layers (creates layers from config files)
      RtxOptionLayer::initializeSystemLayers();

      // Need to set this to true after conf files are parsed, but before any options are accessed.
      RtxOptionImpl::setInitialized(true);

      // Mark all options with onChange callbacks as dirty, so settings derived in the callbacks
      // are initialized even when the options keep their default values.
      RtxOptionManager::markOptionsWithCallbacksDirty();

      // Ensure all of the above values are promoted before the first frame starts.
      // DxvkDevice hasn't been created yet, so pass nullptr here.
      RtxOptionManager::applyPendingValues(nullptr, /* forceOnChange */ true);

      // Log effective RtxOption values after all initialization and migrations are complete
      RtxOptionManager::logEffectiveValues();
    }

  public:
    static void updateUpscalerFromDlssPreset();
    static void updateUpscalerFromNisPreset();
    static void updateUpscalerFromTaauPreset();
    static void updatePresetFromUpscaler();
    static NV_GPU_ARCHITECTURE_ID getNvidiaArch();
    static void updateGraphicsPresets(DxvkDevice* device);

    static void resetUpscaler();

    static void Create() {
      if (s_instance == nullptr) {
        s_instance = new RtxOptions();
      }
      // If called a second time, nothing to do - singleton already exists with all options initialized.
    }

    // Returns the merged configuration for DXKV Options. This includes all config files loaded from DXVK_CONFIG_FILE and DXVK_RTX_CONFIG_FILE.
    // This is available after Create() is called and can be used for DxvkOptions, etc.
    static const Config& getMergedConfig() {
      return RtxOptionLayer::getMergedConfig();
    }

    static bool isDLSSEnabled() {
      return upscalerType() == UpscalerType::DLSS;
    }

    static bool isNISEnabled() { return upscalerType() == UpscalerType::NIS; }
    static bool isTAAEnabled() { return upscalerType() == UpscalerType::TAAU; }
    static bool isXeSSEnabled() { return upscalerType() == UpscalerType::XeSS; }
    
    static float getUniqueObjectDistanceSqr() { return uniqueObjectDistance() * uniqueObjectDistance(); }

    // Developer Options
    static bool areValidationLayersEnabled() {
#ifndef _DEBUG
      return enableValidationLayers();
#else
      return true;
#endif
    }
  };
}
