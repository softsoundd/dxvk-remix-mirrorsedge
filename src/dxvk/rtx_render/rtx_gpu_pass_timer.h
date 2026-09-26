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

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "../dxvk_include.h"
#include "../dxvk_gpu_query.h"
#include "../../util/thread.h"
#include "../../util/util_keybind.h"
#include "rtx_common_object.h"
#include "rtx_option.h"

namespace dxvk {

  class DxvkContext;
  class DxvkDevice;
  class RtxOptionImpl;

  /**
   * \brief Per-pass GPU timings
   *
   * Lightweight built-in GPU profiler: when enabled, every ScopedGpuProfileZone
   * (via __ScopedAnnotation) brackets its command range with a pair of timestamp
   * queries. Completed frames are resolved on the CS thread a few frames later and
   * aggregated per (depth, parent, name) into a rolling window, which is shown in
   * the developer menu and can be dumped to the log. Off by default; when disabled
   * the per-zone cost is a single branch.
   */
  class RtxGpuPassTimer : public CommonDeviceObject {
  public:
    explicit RtxGpuPassTimer(DxvkDevice* device);
    ~RtxGpuPassTimer() override = default;

    // Called from __ScopedAnnotation on the command recording thread.
    void beginZone(DxvkContext* ctx, const char* name);
    void endZone(DxvkContext* ctx);

    // Called from DxvkContext::beginRecording / endRecording: brackets every command list with
    // timestamps so the frame's total GPU time (game draws, vertex capture, uploads and anything
    // else outside the RTX profile zones) is known, not just the zoned passes.
    void onCommandListBegin(DxvkContext* ctx);
    void onCommandListEnd(DxvkContext* ctx);

    // Called once per frame from RtxContext::injectRTX: resolves completed frames,
    // updates the rolling statistics and drives the periodic log dump.
    void onFrameBegin(DxvkContext* ctx);

    void showImguiSettings();

    // Writes the current statistics table to the log.
    void logTimings(const char* reason) const;

    // Automated A/B sweep: applies each step of rtx.gpuPassTimings.sweepSteps in turn (into the
    // user option layer), holds it for sweepHoldSeconds, logs the timing table, restores the
    // previous value and moves on. A baseline table is logged before the first and after the last step.
    void startSweep();
    void stopSweep();
    bool isSweepActive() const {
      return m_sweep.active;
    }

    // Cheap test for the __ScopedAnnotation hook.
    static bool isEnabled() {
      return enable();
    }

    // CPU-side frame breakdown. Sampled from the application and CS threads, folded into the same
    // rolling window as the GPU timings on each new frame, and reported next to the frame interval
    // so that a CPU-bound frame can be attributed to the right thread without a profiler build.
    enum class CpuCounter : std::uint8_t {
      AppPresent = 0,   // application thread: whole D3D9SwapChainEx::Present call
      AppPresentWait,   // application thread: frame-latency signal wait inside Present (SyncFrameLatency)
      AppPrevPresentWait,// application thread: wait for the previous present to have been processed by the submit thread (SynchronizePresent)
      AppAcquireWait,   // application thread: vkAcquireNextImage inside Present (waits for a free swap chain image)
      AppReflexSleep,   // application thread: Reflex sleep inside Present
      AppCsSync,        // application thread: blocked in DxvkCsThread::synchronize (resource readbacks, CS back-pressure)
      AppEventQueryWait,// application thread: polling a pending D3DQUERYTYPE_EVENT (the game's own GPU throttle)
      AppResourceWait,  // application thread: D3D9DeviceEx::WaitForResource (Lock on a GPU-busy resource)
      AppDraw,          // application thread: inside the D3D9 Draw* entry points (classification, geometry hashing, capture setup, state binding, CS enqueue)
      AppDrawPrepare,   // application thread: of which D3D9Rtx::PrepareDraw*GeometryForRT (draw classification and geometry processing)
      // Phases of D3D9Rtx::internalPrepareDraw, in order; their sum is the bulk of AppDrawPrepare.
      AppPrepClassify,  //   vertex factory and instancing classification, makeDrawCallType
      AppPrepIndices,   //   index buffer processing (min/max scan or memoized lookup)
      AppPrepRenderState,//  legacy material, fog and render state (textures, transforms, material hash)
      AppPrepVertices,  //   vertex stream processing, UE3 instance transforms, skinning anchor, capture position source
      AppPrepIdentity,  //   geometry identity keys (stable VS hash, cache keys, geometry hash and bounding box scheduling)
      AppPrepCapture,   //   skinning data, static vertex-capture cache reuse and capture setup
      CsBusy,           // CS thread: executing command stream chunks (includes injectRTX)
      CsInjectRtx,      // CS thread: RtxContext::injectRTX
      CsSubmitBackpressure, // CS thread: blocked in DxvkSubmissionQueue::submit because MaxNumQueuedCommandBuffers lists are in flight
      SubmitQueueSubmit,// submit thread: inside vkQueueSubmit (blocks when the driver's queue is full)
      SubmitQueuePresent,// submit thread: inside vkQueuePresentKHR (blocks when the present queue is full)
      Count
    };

    // Thread-safe: one relaxed atomic add.
    void addCpuSample(CpuCounter counter, std::int64_t durationNs) {
      m_cpuAccumNs[static_cast<std::size_t>(counter)].fetch_add(durationNs, std::memory_order_relaxed);
    }

    // RAII helper: accumulates the scope's wall time into a counter when timings are enabled.
    class CpuScope {
    public:
      CpuScope(RtxGpuPassTimer* timer, CpuCounter counter)
        : m_timer(timer), m_counter(counter) {
        if (m_timer != nullptr && isEnabled()) {
          m_start = std::chrono::steady_clock::now();
        } else {
          m_timer = nullptr;
        }
      }
      ~CpuScope() {
        if (m_timer != nullptr) {
          m_timer->addCpuSample(m_counter, std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - m_start).count());
        }
      }
      CpuScope(const CpuScope&) = delete;
      CpuScope& operator=(const CpuScope&) = delete;
      // The timer this scope reports to, or null when timings are disabled; lets nested scopes skip the enable check.
      RtxGpuPassTimer* timer() const { return m_timer; }
    private:
      RtxGpuPassTimer* m_timer;
      CpuCounter m_counter;
      std::chrono::steady_clock::time_point m_start;
    };

    // Attributes consecutive phases of one code path to counters with a single timestamp per boundary:
    // lap() adds the time since the previous lap (or construction) to the given counter.
    class CpuPhaseTimer {
    public:
      explicit CpuPhaseTimer(RtxGpuPassTimer* timer) : m_timer(timer) {
        if (m_timer != nullptr) {
          m_last = std::chrono::steady_clock::now();
        }
      }
      void lap(CpuCounter counter) {
        if (m_timer != nullptr) {
          const auto now = std::chrono::steady_clock::now();
          m_timer->addCpuSample(counter, std::chrono::duration_cast<std::chrono::nanoseconds>(now - m_last).count());
          m_last = now;
        }
      }
    private:
      RtxGpuPassTimer* m_timer;
      std::chrono::steady_clock::time_point m_last;
    };

    RTX_OPTION("rtx.gpuPassTimings", std::string, sweepSteps, "",
               "Steps for the automated GPU pass timing A/B sweep. Steps are separated by ';'. Each step is one or more 'option=value' "
               "assignments separated by '&', applied together while the step is held (an empty step measures the unmodified baseline). "
               "Example: 'rtx.volumetrics.enable=False;rtx.skyMode=0&rtx.atmosphere.aerialPerspective=False'. "
               "Values are written to the user option layer for the duration of the step and restored afterwards; options that the active "
               "graphics preset controls cannot be overridden this way.");
    RTX_OPTION_ARGS("rtx.gpuPassTimings", float, sweepHoldSeconds, 25.0f,
                    "Seconds each sweep step is held before its timing table is logged and the next step starts.",
                    args.minValue = 1.0f, args.maxValue = 600.0f);
    inline static const VirtualKeys kDefaultSweepHotkey{ VirtualKey{VK_CONTROL}, VirtualKey{VK_SHIFT}, VirtualKey{VK_MENU}, VirtualKey{'P'} };
    RTX_OPTION("rtx.gpuPassTimings", VirtualKeys, sweepHotkey, kDefaultSweepHotkey,
               "Hotkey that starts (or, while running, stops) the automated GPU pass timing sweep. Default is Ctrl+Shift+Alt+P.");

    RTX_OPTION("rtx.gpuPassTimings", bool, enable, false,
               "Enables built-in per-pass GPU timings. Every GPU profile zone (the same markers Tracy and Nsight see) is bracketed with timestamp queries "
               "and aggregated over a rolling window of frames. The table is shown under Developer Settings and can be dumped to the log; "
               "use it to attribute GPU frame time to individual render passes when A/B testing options. Off by default; the disabled path costs a single branch per zone.");
    RTX_OPTION("rtx.gpuPassTimings", bool, gpuZones, true,
               "Bracket every GPU profile zone with timestamp queries (the per-pass table). Each timestamp is a small GPU-side "
               "serialisation point, so with a few hundred zones per frame the table itself costs measurable GPU time. Disable to keep "
               "only the per-command-list totals, the frame interval and the CPU breakdown, which is the least intrusive way to read "
               "frame time and total GPU time.");
    RTX_OPTION_ARGS("rtx.gpuPassTimings", uint32_t, averagingFrames, 120,
                    "Number of resolved frames the GPU pass timings are averaged over.",
                    args.minValue = 1u, args.maxValue = 1024u);
    RTX_OPTION_ARGS("rtx.gpuPassTimings", float, logIntervalSeconds, 0.0f,
                    "When greater than zero, the GPU pass timings table is written to the log at this interval (in seconds) while timings are enabled. 0 disables periodic logging.",
                    args.minValue = 0.0f, args.maxValue = 3600.0f);

  private:
    // Zone names are copied so that dynamically built names stay valid until the frame resolves.
    static constexpr std::size_t kMaxNameLength = 63;
    static constexpr std::uint32_t kMaxPendingFrames = 8;
    static constexpr std::uint32_t kMaxSamples = 1024;

    struct ZoneRecord {
      std::array<char, kMaxNameLength + 1> name = {};
      std::uint32_t parentIndex = UINT32_MAX;
      // Recording stream (one per DxvkContext): the CS thread's RtxContext and the swapchain's
      // presentation context record zones concurrently, each with its own nesting.
      std::uint16_t streamId = 0;
      std::uint16_t depth = 0;
      Rc<DxvkGpuQuery> beginQuery;
      Rc<DxvkGpuQuery> endQuery;
    };

    // One recorded command list: begin/end timestamps independent of the zone tree.
    struct ListRecord {
      Rc<DxvkGpuQuery> beginQuery;
      Rc<DxvkGpuQuery> endQuery;
    };

    struct FrameRecord {
      std::uint32_t frameId = 0;
      // Zones that have begun but not ended yet; the frame is not resolved while this is non-zero.
      std::uint32_t openCount = 0;
      std::vector<ZoneRecord> zones;
      std::vector<ListRecord> lists;
    };

    struct OpenZone {
      std::uint32_t frameId = 0;
      std::uint32_t zoneIndex = 0;
    };

    struct StreamState {
      const DxvkContext* context = nullptr;
      std::vector<OpenZone> openZones;
      // Command list currently being recorded on this context (a context records one list at a time).
      bool listOpen = false;
      std::uint32_t listFrameId = 0;
      std::uint32_t listIndex = 0;
    };

    struct PassStat {
      std::string name;
      std::string parentName;
      std::uint16_t streamId = 0;
      std::uint16_t depth = 0;
      // Position of the pass in the most recent frame it appeared in; used to keep tree order.
      std::uint32_t lastOrder = 0;
      std::uint32_t lastFrameId = 0;
      // Rolling window of per-frame durations in milliseconds (zero when the pass did not run that frame).
      std::vector<float> samples;
      std::uint32_t sampleCursor = 0;
      std::uint32_t sampleCount = 0;
      std::uint32_t hits = 0;
      float lastMs = 0.0f;
      float maxMs = 0.0f;
      float sumMs = 0.0f;
      // Scratch used while folding a resolved frame into the window.
      float frameMs = 0.0f;
      bool seenThisFrame = false;
    };

    struct FrameSummary {
      float busyMs = 0.0f;   // sum of top-level zones
      float spanMs = 0.0f;   // last top-level end - first top-level begin (includes idle gaps)
      float listTotalMs = 0.0f; // sum of all command lists recorded for the frame (everything the GPU executed)
      float listSpanMs = 0.0f;  // last list end - first list begin
    };

    struct DisplayRow {
      std::uint16_t depth;
      std::string name;
      float avgMs;
      float maxMs;
      float lastMs;
      std::uint32_t hits;
    };

    struct SweepAssignment {
      std::string option;
      std::string value;
      // State of the user layer before the step, used to restore it afterwards.
      bool hadUserValue = false;
      std::string originalValue;
      RtxOptionImpl* impl = nullptr;
    };

    struct SweepStep {
      std::string label;
      std::vector<SweepAssignment> assignments;
    };

    struct SweepState {
      bool active = false;
      std::size_t stepIndex = 0;
      std::vector<SweepStep> steps;
      std::chrono::steady_clock::time_point stepStart;
      bool stepApplied = false;
    };

    bool parseSweepSteps(const std::string& text, std::vector<SweepStep>& outSteps) const;
    void applySweepStep(SweepStep& step);
    void restoreSweepStep(SweepStep& step);
    void advanceSweepLocked();
    void finishSweepLocked(const char* reason);

    Rc<DxvkGpuQuery> allocQuery();
    void recycleFrame(FrameRecord& frame);
    void dropAllFrames();
    FrameRecord& findOrCreateFrame(std::uint32_t frameId);
    FrameRecord* findFrame(std::uint32_t frameId);
    StreamState& findOrCreateStream(const DxvkContext* ctx, std::uint16_t& outStreamId);
    void resolvePendingFrames(std::uint32_t currentFrameId);
    bool resolveFrame(FrameRecord& frame);
    void foldFrameIntoStats(const FrameRecord& frame, const std::vector<float>& durationsMs, const FrameSummary& summary);
    void resizeWindow(std::uint32_t frames);
    std::vector<DisplayRow> buildRows() const;
    void resetStats();
    void logTimingsLocked(const char* reason) const;
    std::string buildContextLine() const;

    mutable dxvk::mutex m_mutex;

    // Frames currently recording or waiting for their queries, ordered by frame id.
    std::vector<FrameRecord> m_frames;
    std::vector<StreamState> m_streams;
    std::vector<Rc<DxvkGpuQuery>> m_freeQueries;

    std::vector<PassStat> m_stats;
    std::vector<float> m_busySamples;
    std::vector<float> m_spanSamples;
    std::vector<float> m_listTotalSamples;
    std::vector<float> m_listSpanSamples;
    std::uint32_t m_windowFrames = 0;
    std::uint32_t m_windowCursor = 0;
    std::uint32_t m_windowCount = 0;
    std::uint32_t m_resolvedFrames = 0;
    std::uint32_t m_droppedFrames = 0;
    std::uint32_t m_lastResolvedFrameId = 0;
    double m_timestampPeriodNs = 1.0;
    bool m_wasEnabled = false;

    // CPU-side frame interval over the same window (from consecutive onFrameBegin calls with a new frame id).
    std::vector<float> m_frameIntervalSamples;
    std::uint32_t m_frameIntervalCursor = 0;
    std::uint32_t m_frameIntervalCount = 0;
    std::uint32_t m_lastFrameBeginId = UINT32_MAX;
    std::chrono::steady_clock::time_point m_lastFrameBeginTime = std::chrono::steady_clock::now();

    // CPU frame breakdown: per-counter accumulators filled by the threads between two frame begins,
    // and the per-frame samples they are folded into (same cursor as the frame interval).
    static constexpr std::size_t kCpuCounterCount = static_cast<std::size_t>(CpuCounter::Count);
    std::array<std::atomic<std::int64_t>, kCpuCounterCount> m_cpuAccumNs = {};
    std::array<std::vector<float>, kCpuCounterCount> m_cpuSamples;

    struct CpuSummary {
      float frameIntervalMs = 0.0f;
      std::array<float, kCpuCounterCount> avgMs = {};
      float appWorkMs = 0.0f;   // frame interval - Present - CS sync - event/resource waits: the application's own frame work incl. the D3D9 -> Remix bridge
      float csIdleMs = 0.0f;    // frame interval - CS busy
    };
    CpuSummary computeCpuSummary() const;

    SweepState m_sweep;
    bool m_sweepHotkeyWasDown = false;

    std::chrono::steady_clock::time_point m_lastLogTime = std::chrono::steady_clock::now();
  };

}
