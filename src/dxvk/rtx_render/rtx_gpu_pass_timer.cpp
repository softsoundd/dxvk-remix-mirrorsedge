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
#include "rtx_gpu_pass_timer.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <iomanip>

#include "dxvk_device.h"
#include "dxvk_context.h"
#include "dxvk_objects.h"
#include "rtx_imgui.h"
#include "rtx_options.h"
#include "rtx_option_layer.h"
#include "rtx_global_volumetrics.h"
#include "rtx_neural_radiance_cache.h"
#include "rtx_scene_manager.h"
#include "rtx_accel_manager.h"
#include "../imgui/dxvk_imgui.h"
#include "../../util/config/config.h"
#include "../../util/log/log.h"

namespace dxvk {

  namespace {
    void copyZoneName(std::array<char, 64>& dst, const char* src) {
      if (src == nullptr) {
        src = "<unnamed>";
      }
      const std::size_t length = std::min(std::strlen(src), dst.size() - 1);
      std::memcpy(dst.data(), src, length);
      dst[length] = '\0';
    }

    std::string trimCopy(const std::string& s) {
      const std::size_t begin = s.find_first_not_of(" \t\r\n");
      if (begin == std::string::npos) {
        return std::string();
      }
      const std::size_t end = s.find_last_not_of(" \t\r\n");
      return s.substr(begin, end - begin + 1);
    }

    std::vector<std::string> splitString(const std::string& s, char separator) {
      std::vector<std::string> parts;
      std::size_t start = 0;
      while (start <= s.size()) {
        const std::size_t pos = s.find(separator, start);
        if (pos == std::string::npos) {
          parts.push_back(s.substr(start));
          break;
        }
        parts.push_back(s.substr(start, pos - start));
        start = pos + 1;
      }
      return parts;
    }
  }

  RtxGpuPassTimer::RtxGpuPassTimer(DxvkDevice* device)
    : CommonDeviceObject(device) {
    m_timestampPeriodNs = device->adapter()->deviceProperties().limits.timestampPeriod;
    if (!(m_timestampPeriodNs > 0.0)) {
      m_timestampPeriodNs = 1.0;
    }
  }

  Rc<DxvkGpuQuery> RtxGpuPassTimer::allocQuery() {
    if (!m_freeQueries.empty()) {
      Rc<DxvkGpuQuery> query = std::move(m_freeQueries.back());
      m_freeQueries.pop_back();
      return query;
    }

    return m_device->createGpuQuery(VK_QUERY_TYPE_TIMESTAMP, 0, 0);
  }

  void RtxGpuPassTimer::recycleFrame(FrameRecord& frame) {
    for (ZoneRecord& zone : frame.zones) {
      if (zone.beginQuery != nullptr) {
        m_freeQueries.push_back(std::move(zone.beginQuery));
      }
      if (zone.endQuery != nullptr) {
        m_freeQueries.push_back(std::move(zone.endQuery));
      }
    }
    for (ListRecord& list : frame.lists) {
      if (list.beginQuery != nullptr) {
        m_freeQueries.push_back(std::move(list.beginQuery));
      }
      if (list.endQuery != nullptr) {
        m_freeQueries.push_back(std::move(list.endQuery));
      }
    }
    frame.zones.clear();
    frame.lists.clear();
    frame.openCount = 0;
  }

  void RtxGpuPassTimer::dropAllFrames() {
    for (FrameRecord& frame : m_frames) {
      recycleFrame(frame);
    }
    m_frames.clear();
    for (StreamState& stream : m_streams) {
      stream.openZones.clear();
      stream.listOpen = false;
    }
  }

  void RtxGpuPassTimer::onCommandListBegin(DxvkContext* ctx) {
    if (ctx == nullptr) {
      return;
    }

    std::lock_guard<dxvk::mutex> lock(m_mutex);

    const std::uint32_t frameId = m_device->getCurrentFrameId();
    std::uint16_t streamId = 0;
    StreamState& stream = findOrCreateStream(ctx, streamId);
    FrameRecord& frame = findOrCreateFrame(frameId);

    ListRecord list;
    list.beginQuery = allocQuery();
    ctx->writeTimestamp(list.beginQuery);

    stream.listOpen = true;
    stream.listFrameId = frameId;
    stream.listIndex = static_cast<std::uint32_t>(frame.lists.size());
    frame.lists.push_back(std::move(list));
    ++frame.openCount;
  }

  void RtxGpuPassTimer::onCommandListEnd(DxvkContext* ctx) {
    if (ctx == nullptr) {
      return;
    }

    std::lock_guard<dxvk::mutex> lock(m_mutex);

    std::uint16_t streamId = 0;
    StreamState& stream = findOrCreateStream(ctx, streamId);
    if (!stream.listOpen) {
      return;
    }
    stream.listOpen = false;

    FrameRecord* frame = findFrame(stream.listFrameId);
    if (frame == nullptr || stream.listIndex >= frame->lists.size()) {
      return;
    }

    ListRecord& list = frame->lists[stream.listIndex];
    list.endQuery = allocQuery();
    ctx->writeTimestamp(list.endQuery);
    if (frame->openCount > 0) {
      --frame->openCount;
    }
  }

  RtxGpuPassTimer::FrameRecord& RtxGpuPassTimer::findOrCreateFrame(std::uint32_t frameId) {
    // Frames are appended in (almost) increasing id order; keep the vector sorted by id so
    // resolution walks oldest first.
    auto it = std::lower_bound(m_frames.begin(), m_frames.end(), frameId,
      [](const FrameRecord& frame, std::uint32_t id) { return frame.frameId < id; });
    if (it != m_frames.end() && it->frameId == frameId) {
      return *it;
    }
    FrameRecord fresh;
    fresh.frameId = frameId;
    it = m_frames.insert(it, std::move(fresh));
    return *it;
  }

  RtxGpuPassTimer::FrameRecord* RtxGpuPassTimer::findFrame(std::uint32_t frameId) {
    auto it = std::lower_bound(m_frames.begin(), m_frames.end(), frameId,
      [](const FrameRecord& frame, std::uint32_t id) { return frame.frameId < id; });
    if (it != m_frames.end() && it->frameId == frameId) {
      return &*it;
    }
    return nullptr;
  }

  RtxGpuPassTimer::StreamState& RtxGpuPassTimer::findOrCreateStream(const DxvkContext* ctx, std::uint16_t& outStreamId) {
    for (std::size_t i = 0; i < m_streams.size(); ++i) {
      if (m_streams[i].context == ctx) {
        outStreamId = static_cast<std::uint16_t>(i);
        return m_streams[i];
      }
    }
    StreamState stream;
    stream.context = ctx;
    m_streams.push_back(std::move(stream));
    outStreamId = static_cast<std::uint16_t>(m_streams.size() - 1);
    return m_streams.back();
  }

  void RtxGpuPassTimer::beginZone(DxvkContext* ctx, const char* name) {
    if (ctx == nullptr) {
      return;
    }

    std::lock_guard<dxvk::mutex> lock(m_mutex);

    const std::uint32_t frameId = m_device->getCurrentFrameId();

    std::uint16_t streamId = 0;
    StreamState& stream = findOrCreateStream(ctx, streamId);
    FrameRecord& frame = findOrCreateFrame(frameId);

    ZoneRecord zone;
    copyZoneName(zone.name, name);
    zone.streamId = streamId;
    zone.depth = static_cast<std::uint16_t>(std::min<std::size_t>(stream.openZones.size(), UINT16_MAX));
    // Parent links only within the same frame record; a zone straddling a frame boundary starts a new tree.
    if (!stream.openZones.empty() && stream.openZones.back().frameId == frameId) {
      zone.parentIndex = stream.openZones.back().zoneIndex;
    }
    zone.beginQuery = allocQuery();

    ctx->writeTimestamp(zone.beginQuery);

    OpenZone open;
    open.frameId = frameId;
    open.zoneIndex = static_cast<std::uint32_t>(frame.zones.size());
    stream.openZones.push_back(open);
    frame.zones.push_back(std::move(zone));
    ++frame.openCount;
  }

  void RtxGpuPassTimer::endZone(DxvkContext* ctx) {
    if (ctx == nullptr) {
      return;
    }

    std::lock_guard<dxvk::mutex> lock(m_mutex);

    std::uint16_t streamId = 0;
    StreamState& stream = findOrCreateStream(ctx, streamId);
    if (stream.openZones.empty()) {
      return;
    }

    const OpenZone open = stream.openZones.back();
    stream.openZones.pop_back();

    FrameRecord* frame = findFrame(open.frameId);
    if (frame == nullptr || open.zoneIndex >= frame->zones.size()) {
      // The frame was already dropped as too old; nothing to close.
      return;
    }

    ZoneRecord& zone = frame->zones[open.zoneIndex];
    zone.endQuery = allocQuery();
    ctx->writeTimestamp(zone.endQuery);
    if (frame->openCount > 0) {
      --frame->openCount;
    }
  }

  void RtxGpuPassTimer::onFrameBegin(DxvkContext* ctx) {
    // Hotkey toggles the sweep; starting it also enables the timings. Edge-detect the combination
    // here because this can run more than once per ImGui frame, and ImGui reports a fresh press for
    // the whole frame.
    const bool hotkeyDown = ImGUI::checkHotkeyState(sweepHotkey(), true);
    if (hotkeyDown && !m_sweepHotkeyWasDown) {
      if (isSweepActive()) {
        stopSweep();
      } else {
        startSweep();
      }
    }
    m_sweepHotkeyWasDown = hotkeyDown;

    const bool isEnabled = enable();

    std::lock_guard<dxvk::mutex> lock(m_mutex);

    if (!isEnabled) {
      if (m_wasEnabled) {
        // Drop everything so a later re-enable starts from a clean window.
        dropAllFrames();
        m_freeQueries.clear();
        resetStats();
        m_wasEnabled = false;
      }
      return;
    }

    if (!m_wasEnabled) {
      m_wasEnabled = true;
      m_lastLogTime = std::chrono::steady_clock::now();
      m_lastFrameBeginId = UINT32_MAX;
    }

    resizeWindow(std::clamp(averagingFrames(), 1u, kMaxSamples));

    const std::uint32_t currentFrameId = m_device->getCurrentFrameId();
    const auto now = std::chrono::steady_clock::now();

    if (currentFrameId != m_lastFrameBeginId) {
      if (m_lastFrameBeginId != UINT32_MAX) {
        const float intervalMs = std::chrono::duration<float, std::milli>(now - m_lastFrameBeginTime).count();
        m_frameIntervalSamples[m_frameIntervalCursor] = intervalMs;
        // The CPU counters accumulated since the previous frame begin belong to the same interval.
        for (std::size_t i = 0; i < kCpuCounterCount; ++i) {
          const std::int64_t ns = m_cpuAccumNs[i].exchange(0, std::memory_order_relaxed);
          m_cpuSamples[i][m_frameIntervalCursor] = static_cast<float>(ns) * 1e-6f;
        }
        m_frameIntervalCursor = (m_frameIntervalCursor + 1) % m_windowFrames;
        m_frameIntervalCount = std::min(m_frameIntervalCount + 1, m_windowFrames);
      } else {
        // First frame after enabling: discard whatever accumulated while disabled.
        for (std::size_t i = 0; i < kCpuCounterCount; ++i) {
          m_cpuAccumNs[i].store(0, std::memory_order_relaxed);
        }
      }
      m_lastFrameBeginId = currentFrameId;
      m_lastFrameBeginTime = now;
    }

    resolvePendingFrames(currentFrameId);

    if (m_sweep.active) {
      advanceSweepLocked();
    }

    const float interval = logIntervalSeconds();
    if (interval > 0.0f) {
      const float elapsed = std::chrono::duration<float>(now - m_lastLogTime).count();
      if (elapsed >= interval) {
        m_lastLogTime = now;
        logTimingsLocked("periodic");
      }
    }
  }

  bool RtxGpuPassTimer::parseSweepSteps(const std::string& text, std::vector<SweepStep>& outSteps) const {
    outSteps.clear();

    for (const std::string& rawStep : splitString(text, ';')) {
      const std::string stepText = trimCopy(rawStep);
      if (stepText.empty()) {
        continue;
      }

      SweepStep step;
      step.label = stepText;

      if (stepText != "baseline") {
        for (const std::string& rawAssignment : splitString(stepText, '&')) {
          const std::string assignmentText = trimCopy(rawAssignment);
          if (assignmentText.empty()) {
            continue;
          }
          const std::size_t eq = assignmentText.find('=');
          if (eq == std::string::npos) {
            Logger::warn(str::format("[GPU Pass Timings] Ignoring sweep assignment without '=': ", assignmentText));
            continue;
          }
          SweepAssignment assignment;
          assignment.option = trimCopy(assignmentText.substr(0, eq));
          assignment.value = trimCopy(assignmentText.substr(eq + 1));
          assignment.impl = RtxOptionImpl::getOptionByFullName(assignment.option);
          if (assignment.impl == nullptr) {
            Logger::warn(str::format("[GPU Pass Timings] Ignoring sweep assignment for unknown option: ", assignment.option));
            continue;
          }
          if (assignment.value.empty()) {
            // Config lookups treat an empty value as absent, so it could neither be applied nor restored.
            Logger::warn(str::format("[GPU Pass Timings] Ignoring sweep assignment with an empty value: ", assignment.option));
            continue;
          }
          step.assignments.push_back(std::move(assignment));
        }
        if (step.assignments.empty()) {
          Logger::warn(str::format("[GPU Pass Timings] Sweep step has no valid assignments, skipping: ", stepText));
          continue;
        }
      }

      outSteps.push_back(std::move(step));
    }

    return !outSteps.empty();
  }

  void RtxGpuPassTimer::applySweepStep(SweepStep& step) {
    const RtxOptionLayer* userLayer = RtxOptionLayer::getUserLayer();
    for (SweepAssignment& assignment : step.assignments) {
      assignment.hadUserValue = assignment.impl->hasValueInLayer(userLayer);
      if (assignment.hadUserValue) {
        const GenericValue* current = assignment.impl->getGenericValue(userLayer);
        assignment.originalValue = current != nullptr ? assignment.impl->genericValueToString(*current) : std::string();
      }

      Config config;
      config.setOptionMove(std::string(assignment.option), std::string(assignment.value));
      assignment.impl->readOption(config, userLayer);
    }
  }

  void RtxGpuPassTimer::restoreSweepStep(SweepStep& step) {
    const RtxOptionLayer* userLayer = RtxOptionLayer::getUserLayer();
    for (SweepAssignment& assignment : step.assignments) {
      if (assignment.hadUserValue && !assignment.originalValue.empty()) {
        Config config;
        config.setOptionMove(std::string(assignment.option), std::string(assignment.originalValue));
        assignment.impl->readOption(config, userLayer);
      } else {
        assignment.impl->disableLayerValue(userLayer);
      }
    }
  }

  void RtxGpuPassTimer::startSweep() {
    std::vector<SweepStep> steps;
    if (!parseSweepSteps(sweepSteps(), steps)) {
      Logger::warn("[GPU Pass Timings] Sweep not started: rtx.gpuPassTimings.sweepSteps has no valid steps.");
      return;
    }

    // Baseline before and after so drift over the sweep is visible.
    SweepStep baselineStart;
    baselineStart.label = "baseline";
    SweepStep baselineEnd;
    baselineEnd.label = "baseline (end)";
    steps.insert(steps.begin(), std::move(baselineStart));
    steps.push_back(std::move(baselineEnd));

    std::lock_guard<dxvk::mutex> lock(m_mutex);

    if (m_sweep.active) {
      return;
    }

    if (!enable()) {
      enable.setDeferred(true);
    }

    m_sweep = SweepState();
    m_sweep.active = true;
    m_sweep.steps = std::move(steps);
    m_sweep.stepIndex = 0;
    m_sweep.stepApplied = false;

    std::ostringstream out;
    out << "[GPU Pass Timings] Sweep started: " << m_sweep.steps.size() << " steps, " << std::fixed << std::setprecision(1)
        << sweepHoldSeconds() << " s each\n";
    for (std::size_t i = 0; i < m_sweep.steps.size(); ++i) {
      out << "  step " << (i + 1) << ": " << m_sweep.steps[i].label << '\n';
    }
    Logger::info(out.str());
  }

  void RtxGpuPassTimer::stopSweep() {
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    if (!m_sweep.active) {
      return;
    }
    finishSweepLocked("stopped");
  }

  void RtxGpuPassTimer::finishSweepLocked(const char* reason) {
    if (m_sweep.stepApplied && m_sweep.stepIndex < m_sweep.steps.size()) {
      restoreSweepStep(m_sweep.steps[m_sweep.stepIndex]);
    }
    Logger::info(str::format("[GPU Pass Timings] Sweep ", reason, " (", m_sweep.stepIndex, "/", m_sweep.steps.size(), " steps completed)"));
    m_sweep = SweepState();
  }

  void RtxGpuPassTimer::advanceSweepLocked() {
    if (m_sweep.stepIndex >= m_sweep.steps.size()) {
      finishSweepLocked("finished");
      return;
    }

    SweepStep& step = m_sweep.steps[m_sweep.stepIndex];
    const auto now = std::chrono::steady_clock::now();

    if (!m_sweep.stepApplied) {
      applySweepStep(step);
      m_sweep.stepApplied = true;
      m_sweep.stepStart = now;
      // Fresh window so the table at the end of the hold only contains this step's frames.
      resetStats();
      resizeWindow(std::clamp(averagingFrames(), 1u, kMaxSamples));
      Logger::info(str::format("[GPU Pass Timings] Sweep step ", m_sweep.stepIndex + 1, "/", m_sweep.steps.size(), " applied: ", step.label));
      return;
    }

    const float held = std::chrono::duration<float>(now - m_sweep.stepStart).count();
    if (held < sweepHoldSeconds()) {
      return;
    }

    const std::string reason = str::format("sweep step ", m_sweep.stepIndex + 1, "/", m_sweep.steps.size(), ": ", step.label);
    logTimingsLocked(reason.c_str());

    restoreSweepStep(step);
    m_sweep.stepApplied = false;
    ++m_sweep.stepIndex;

    if (m_sweep.stepIndex >= m_sweep.steps.size()) {
      finishSweepLocked("finished");
    }
  }

  void RtxGpuPassTimer::resolvePendingFrames(std::uint32_t currentFrameId) {
    // Oldest first so the rolling window keeps frame order. A frame is only eligible once the
    // device frame counter has moved past it and every zone recorded into it has ended; the
    // presentation context can still be closing its zones for the previous frame id.
    std::size_t writeIndex = 0;
    for (std::size_t i = 0; i < m_frames.size(); ++i) {
      FrameRecord& frame = m_frames[i];

      const std::uint32_t age = currentFrameId - frame.frameId;
      const bool tooOld = age > kMaxPendingFrames;
      const bool eligible = age >= 2 && frame.openCount == 0 && (!frame.zones.empty() || !frame.lists.empty());

      bool done = false;
      if (eligible && resolveFrame(frame)) {
        done = true;
      } else if (tooOld) {
        ++m_droppedFrames;
        done = true;
      }

      if (done) {
        recycleFrame(frame);
        continue;
      }

      if (writeIndex != i) {
        m_frames[writeIndex] = std::move(frame);
      }
      ++writeIndex;
    }
    m_frames.resize(writeIndex);
  }

  bool RtxGpuPassTimer::resolveFrame(FrameRecord& frame) {
    std::vector<float> durationsMs(frame.zones.size(), 0.0f);
    std::vector<std::uint64_t> beginTicks(frame.zones.size(), 0);
    std::vector<std::uint64_t> endTicks(frame.zones.size(), 0);

    for (std::size_t i = 0; i < frame.zones.size(); ++i) {
      const ZoneRecord& zone = frame.zones[i];
      if (zone.beginQuery == nullptr || zone.endQuery == nullptr) {
        // Unbalanced zone: treat as zero length.
        continue;
      }

      DxvkQueryData beginData = {};
      DxvkQueryData endData = {};
      const DxvkGpuQueryStatus beginStatus = zone.beginQuery->getData(beginData);
      const DxvkGpuQueryStatus endStatus = zone.endQuery->getData(endData);

      if (beginStatus == DxvkGpuQueryStatus::Pending || endStatus == DxvkGpuQueryStatus::Pending) {
        return false;
      }
      if (beginStatus != DxvkGpuQueryStatus::Available || endStatus != DxvkGpuQueryStatus::Available) {
        // Invalid/failed queries: skip this zone but still consume the frame.
        continue;
      }

      beginTicks[i] = beginData.timestamp.time;
      endTicks[i] = endData.timestamp.time;
      if (endTicks[i] > beginTicks[i]) {
        durationsMs[i] = static_cast<float>(static_cast<double>(endTicks[i] - beginTicks[i]) * m_timestampPeriodNs * 1.0e-6);
      }
    }

    FrameSummary summary;
    std::uint64_t firstBegin = UINT64_MAX;
    std::uint64_t lastEnd = 0;
    for (std::size_t i = 0; i < frame.zones.size(); ++i) {
      if (frame.zones[i].depth != 0 || endTicks[i] == 0) {
        continue;
      }
      summary.busyMs += durationsMs[i];
      firstBegin = std::min(firstBegin, beginTicks[i]);
      lastEnd = std::max(lastEnd, endTicks[i]);
    }
    if (lastEnd > firstBegin && firstBegin != UINT64_MAX) {
      summary.spanMs = static_cast<float>(static_cast<double>(lastEnd - firstBegin) * m_timestampPeriodNs * 1.0e-6);
    }

    // Whole command lists: everything the GPU executed for this frame, zoned or not.
    std::uint64_t firstListBegin = UINT64_MAX;
    std::uint64_t lastListEnd = 0;
    for (const ListRecord& list : frame.lists) {
      if (list.beginQuery == nullptr || list.endQuery == nullptr) {
        continue;
      }
      DxvkQueryData beginData = {};
      DxvkQueryData endData = {};
      const DxvkGpuQueryStatus beginStatus = list.beginQuery->getData(beginData);
      const DxvkGpuQueryStatus endStatus = list.endQuery->getData(endData);
      if (beginStatus == DxvkGpuQueryStatus::Pending || endStatus == DxvkGpuQueryStatus::Pending) {
        return false;
      }
      if (beginStatus != DxvkGpuQueryStatus::Available || endStatus != DxvkGpuQueryStatus::Available) {
        continue;
      }
      const std::uint64_t begin = beginData.timestamp.time;
      const std::uint64_t end = endData.timestamp.time;
      if (end > begin) {
        summary.listTotalMs += static_cast<float>(static_cast<double>(end - begin) * m_timestampPeriodNs * 1.0e-6);
        firstListBegin = std::min(firstListBegin, begin);
        lastListEnd = std::max(lastListEnd, end);
      }
    }
    if (lastListEnd > firstListBegin && firstListBegin != UINT64_MAX) {
      summary.listSpanMs = static_cast<float>(static_cast<double>(lastListEnd - firstListBegin) * m_timestampPeriodNs * 1.0e-6);
    }

    foldFrameIntoStats(frame, durationsMs, summary);
    m_lastResolvedFrameId = frame.frameId;
    ++m_resolvedFrames;
    return true;
  }

  void RtxGpuPassTimer::resizeWindow(std::uint32_t frames) {
    if (frames == m_windowFrames) {
      return;
    }
    m_windowFrames = frames;
    m_windowCursor = 0;
    m_windowCount = 0;
    m_busySamples.assign(frames, 0.0f);
    m_spanSamples.assign(frames, 0.0f);
    m_listTotalSamples.assign(frames, 0.0f);
    m_listSpanSamples.assign(frames, 0.0f);
    m_frameIntervalSamples.assign(frames, 0.0f);
    m_frameIntervalCursor = 0;
    m_frameIntervalCount = 0;
    for (std::vector<float>& samples : m_cpuSamples) {
      samples.assign(frames, 0.0f);
    }
    for (PassStat& stat : m_stats) {
      stat.samples.assign(frames, 0.0f);
      stat.sampleCursor = 0;
      stat.sampleCount = 0;
      stat.hits = 0;
      stat.sumMs = 0.0f;
      stat.maxMs = 0.0f;
    }
  }

  void RtxGpuPassTimer::resetStats() {
    m_stats.clear();
    m_windowFrames = 0;
    m_windowCursor = 0;
    m_windowCount = 0;
    m_busySamples.clear();
    m_spanSamples.clear();
    m_listTotalSamples.clear();
    m_listSpanSamples.clear();
    m_frameIntervalSamples.clear();
    m_frameIntervalCursor = 0;
    m_frameIntervalCount = 0;
    for (std::vector<float>& samples : m_cpuSamples) {
      samples.clear();
    }
    m_resolvedFrames = 0;
    m_droppedFrames = 0;
  }

  RtxGpuPassTimer::CpuSummary RtxGpuPassTimer::computeCpuSummary() const {
    CpuSummary summary;
    if (m_frameIntervalCount == 0) {
      return summary;
    }

    const float invCount = 1.0f / static_cast<float>(m_frameIntervalCount);
    for (std::uint32_t i = 0; i < m_frameIntervalCount; ++i) {
      summary.frameIntervalMs += m_frameIntervalSamples[i];
      for (std::size_t c = 0; c < kCpuCounterCount; ++c) {
        summary.avgMs[c] += m_cpuSamples[c][i];
      }
    }
    summary.frameIntervalMs *= invCount;
    for (float& avg : summary.avgMs) {
      avg *= invCount;
    }

    const float present = summary.avgMs[static_cast<std::size_t>(CpuCounter::AppPresent)];
    const float csSync = summary.avgMs[static_cast<std::size_t>(CpuCounter::AppCsSync)];
    const float eventWait = summary.avgMs[static_cast<std::size_t>(CpuCounter::AppEventQueryWait)];
    const float resourceWait = summary.avgMs[static_cast<std::size_t>(CpuCounter::AppResourceWait)];
    const float csBusy = summary.avgMs[static_cast<std::size_t>(CpuCounter::CsBusy)];
    // CS syncs mostly happen inside WaitForResource, so only count the part that exceeds the resource wait.
    const float csSyncOutsideResourceWait = std::max(0.0f, csSync - resourceWait);
    summary.appWorkMs = std::max(0.0f, summary.frameIntervalMs - present - csSyncOutsideResourceWait - eventWait - resourceWait);
    summary.csIdleMs = std::max(0.0f, summary.frameIntervalMs - csBusy);
    return summary;
  }

  void RtxGpuPassTimer::foldFrameIntoStats(const FrameRecord& frame, const std::vector<float>& durationsMs, const FrameSummary& summary) {
    for (PassStat& stat : m_stats) {
      stat.frameMs = 0.0f;
      stat.seenThisFrame = false;
    }

    // Accumulate this frame's durations per (stream, depth, parent, name).
    for (std::size_t i = 0; i < frame.zones.size(); ++i) {
      const ZoneRecord& zone = frame.zones[i];
      const char* parentName = zone.parentIndex == UINT32_MAX ? "" : frame.zones[zone.parentIndex].name.data();

      PassStat* stat = nullptr;
      for (PassStat& candidate : m_stats) {
        if (candidate.streamId == zone.streamId &&
            candidate.depth == zone.depth &&
            candidate.name == zone.name.data() &&
            candidate.parentName == parentName) {
          stat = &candidate;
          break;
        }
      }

      if (stat == nullptr) {
        PassStat fresh;
        fresh.name = zone.name.data();
        fresh.parentName = parentName;
        fresh.streamId = zone.streamId;
        fresh.depth = zone.depth;
        fresh.samples.assign(m_windowFrames, 0.0f);
        m_stats.push_back(std::move(fresh));
        stat = &m_stats.back();
      }

      if (!stat->seenThisFrame) {
        stat->seenThisFrame = true;
        stat->lastOrder = static_cast<std::uint32_t>(i);
        stat->lastFrameId = frame.frameId;
        stat->lastMs = 0.0f;
      }
      stat->frameMs += durationsMs[i];
      stat->lastMs += durationsMs[i];
      if (durationsMs[i] > 0.0f) {
        ++stat->hits;
      }
    }

    // Push a sample (zero when the pass did not run) into every stat's window.
    for (PassStat& stat : m_stats) {
      float& slot = stat.samples[stat.sampleCursor];
      stat.sumMs -= slot;
      slot = stat.frameMs;
      stat.sumMs += slot;
      stat.sampleCursor = (stat.sampleCursor + 1) % m_windowFrames;
      stat.sampleCount = std::min(stat.sampleCount + 1, m_windowFrames);
      stat.maxMs = 0.0f;
      for (float sample : stat.samples) {
        stat.maxMs = std::max(stat.maxMs, sample);
      }
    }

    m_busySamples[m_windowCursor] = summary.busyMs;
    m_spanSamples[m_windowCursor] = summary.spanMs;
    m_listTotalSamples[m_windowCursor] = summary.listTotalMs;
    m_listSpanSamples[m_windowCursor] = summary.listSpanMs;
    m_windowCursor = (m_windowCursor + 1) % m_windowFrames;
    m_windowCount = std::min(m_windowCount + 1, m_windowFrames);

    // Prune passes that have not been seen for a full window, and recount hits over the window.
    for (PassStat& stat : m_stats) {
      std::uint32_t hits = 0;
      for (float sample : stat.samples) {
        if (sample > 0.0f) {
          ++hits;
        }
      }
      stat.hits = hits;
    }
    m_stats.erase(std::remove_if(m_stats.begin(), m_stats.end(), [&](const PassStat& stat) {
      return frame.frameId - stat.lastFrameId > m_windowFrames;
    }), m_stats.end());
  }

  std::vector<RtxGpuPassTimer::DisplayRow> RtxGpuPassTimer::buildRows() const {
    // Order by the most recently observed position in the frame so the table reads as a tree.
    std::vector<const PassStat*> ordered;
    ordered.reserve(m_stats.size());
    for (const PassStat& stat : m_stats) {
      ordered.push_back(&stat);
    }
    std::stable_sort(ordered.begin(), ordered.end(), [](const PassStat* a, const PassStat* b) {
      if (a->streamId != b->streamId) {
        return a->streamId < b->streamId;
      }
      return a->lastOrder < b->lastOrder;
    });

    const float invFrames = m_windowCount > 0 ? 1.0f / static_cast<float>(m_windowCount) : 0.0f;

    std::vector<DisplayRow> rows;
    rows.reserve(ordered.size());
    for (const PassStat* stat : ordered) {
      DisplayRow row;
      row.depth = stat->depth;
      row.name = stat->name;
      row.avgMs = stat->sumMs * invFrames;
      row.maxMs = stat->maxMs;
      row.lastMs = stat->lastMs;
      row.hits = stat->hits;
      rows.push_back(std::move(row));
    }
    return rows;
  }

  void RtxGpuPassTimer::logTimings(const char* reason) const {
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    logTimingsLocked(reason);
  }

  std::string RtxGpuPassTimer::buildContextLine() const {
    // A compact record of the settings that dominate GPU cost, so a logged table is self-describing
    // when several option toggles are compared in one session.
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);

    DxvkObjects* common = m_device->getCommon();
    if (common != nullptr) {
      const VkExtent3D& internal = common->getResources().getDownscaleDimensions();
      const VkExtent3D& target = common->getResources().getTargetDimensions();
      out << "render=" << internal.width << "x" << internal.height
          << " output=" << target.width << "x" << target.height;
    }

    out << " | upscaler=" << static_cast<int>(RtxOptions::upscalerType())
        << " dlssProfile=" << static_cast<int>(RtxOptions::qualityDLSS())
        << " rayReconstruction=" << (RtxOptions::enableRayReconstruction() ? 1 : 0)
        << " graphicsPreset=" << static_cast<int>(RtxOptions::graphicsPreset())
        << " integrateIndirectMode=" << static_cast<int>(RtxOptions::integrateIndirectMode())
        << " nrcPreset=" << static_cast<int>(NeuralRadianceCache::NrcOptions::qualityPreset())
        << " pathMaxBounces=" << static_cast<int>(RtxOptions::pathMaxBounces())
        << " psrMaxDistanceMeters=" << RtxOptions::psrMaxDistanceMeters()
        << " displacementMode=" << static_cast<int>(RtxOptions::Displacement::mode())
        << " tonemappingMode=" << static_cast<int>(RtxOptions::tonemappingMode())
        << " cullBackfacesInShadowGeometries=" << RtxOptions::cullBackfacesInShadowGeometries().size()
        << " cullBackfacesInShadowTextures=" << RtxOptions::cullBackfacesInShadowTextures().size()
        << " | volumetrics=" << (RtxGlobalVolumetrics::enable() ? 1 : 0)
        << " froxelScale=" << RtxGlobalVolumetrics::froxelGridResolutionScale()
        << " froxelSlices=" << RtxGlobalVolumetrics::froxelDepthSlices()
        << " atmosphereShell=" << (RtxGlobalVolumetrics::enableAtmosphere() ? 1 : 0);

    return out.str();
  }

  void RtxGpuPassTimer::logTimingsLocked(const char* reason) const {
    if (m_windowCount == 0) {
      Logger::info(str::format("[GPU Pass Timings] (", reason, ") no resolved frames yet"));
      return;
    }

    float busyAvg = 0.0f;
    float spanAvg = 0.0f;
    float busyMax = 0.0f;
    float listTotalAvg = 0.0f;
    float listSpanAvg = 0.0f;
    for (std::uint32_t i = 0; i < m_windowCount; ++i) {
      busyAvg += m_busySamples[i];
      spanAvg += m_spanSamples[i];
      busyMax = std::max(busyMax, m_busySamples[i]);
      listTotalAvg += m_listTotalSamples[i];
      listSpanAvg += m_listSpanSamples[i];
    }
    busyAvg /= static_cast<float>(m_windowCount);
    spanAvg /= static_cast<float>(m_windowCount);
    listTotalAvg /= static_cast<float>(m_windowCount);
    listSpanAvg /= static_cast<float>(m_windowCount);

    float frameIntervalAvg = 0.0f;
    for (std::uint32_t i = 0; i < m_frameIntervalCount; ++i) {
      frameIntervalAvg += m_frameIntervalSamples[i];
    }
    if (m_frameIntervalCount > 0) {
      frameIntervalAvg /= static_cast<float>(m_frameIntervalCount);
    }
    const float fps = frameIntervalAvg > 0.0f ? 1000.0f / frameIntervalAvg : 0.0f;

    const std::vector<DisplayRow> rows = buildRows();

    std::ostringstream out;
    out << "[GPU Pass Timings] (" << reason << ") window=" << m_windowCount << " frames"
        << ", last frame id=" << m_lastResolvedFrameId
        << ", dropped=" << m_droppedFrames
        << std::fixed << std::setprecision(3)
        << ", gpu busy avg=" << busyAvg << " ms (max " << busyMax << " ms)"
        << ", gpu span avg=" << spanAvg << " ms"
        << ", gpu cmdlists total avg=" << listTotalAvg << " ms (span " << listSpanAvg << " ms)"
        << ", frame interval avg=" << frameIntervalAvg << " ms (" << std::setprecision(1) << fps << " fps)"
        << std::setprecision(3) << '\n';
    out << "  context: " << buildContextLine() << '\n';
    {
      const CpuSummary cpu = computeCpuSummary();
      const auto avg = [&cpu](CpuCounter c) { return cpu.avgMs[static_cast<std::size_t>(c)]; };
      out << "  cpu: app thread work=" << cpu.appWorkMs << " ms"
          << " (Present total=" << avg(CpuCounter::AppPresent)
          << ": frame latency wait=" << avg(CpuCounter::AppPresentWait)
          << ", prev present wait=" << avg(CpuCounter::AppPrevPresentWait)
          << ", acquire wait=" << avg(CpuCounter::AppAcquireWait)
          << ", reflex sleep=" << avg(CpuCounter::AppReflexSleep)
          << "; cs sync=" << avg(CpuCounter::AppCsSync)
          << "; event query wait=" << avg(CpuCounter::AppEventQueryWait)
          << "; resource wait=" << avg(CpuCounter::AppResourceWait) << ")"
          << " | cs thread busy=" << avg(CpuCounter::CsBusy) << " ms"
          << " (injectRTX " << avg(CpuCounter::CsInjectRtx)
          << ", submit back-pressure " << avg(CpuCounter::CsSubmitBackpressure) << ")"
          << ", idle=" << cpu.csIdleMs << " ms"
          << " | submit thread: vkQueueSubmit=" << avg(CpuCounter::SubmitQueueSubmit)
          << ", vkQueuePresent=" << avg(CpuCounter::SubmitQueuePresent) << " ms\n";
    }
    out << "  " << std::setw(9) << "avg ms" << ' ' << std::setw(9) << "max ms" << ' '
        << std::setw(9) << "last ms" << ' ' << std::setw(6) << "%busy" << ' ' << std::setw(5) << "hits" << "  pass\n";

    const float invBusy = busyAvg > 0.0f ? 100.0f / busyAvg : 0.0f;
    for (const DisplayRow& row : rows) {
      out << "  " << std::setw(9) << row.avgMs << ' ' << std::setw(9) << row.maxMs << ' '
          << std::setw(9) << row.lastMs << ' ' << std::setw(6) << std::setprecision(1) << (row.avgMs * invBusy)
          << std::setprecision(3) << ' ' << std::setw(5) << row.hits << "  ";
      for (std::uint16_t d = 0; d < row.depth; ++d) {
        out << "  ";
      }
      out << row.name << '\n';
    }

    Logger::info(out.str());
  }

  void RtxGpuPassTimer::showImguiSettings() {
    RemixGui::Checkbox("Enable GPU Pass Timings", &enableObject());
    RemixGui::SetTooltipToLastWidgetOnHover("Brackets every GPU profile zone with timestamp queries and averages the results over a rolling window. "
                                            "Passes are listed in frame order and indented by nesting depth.");
    RemixGui::DragInt("Averaging Frames", &averagingFramesObject(), 1.0f, 1, static_cast<int>(kMaxSamples), "%d", ImGuiSliderFlags_AlwaysClamp);
    RemixGui::DragFloat("Log Interval (s)", &logIntervalSecondsObject(), 0.5f, 0.0f, 3600.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
    RemixGui::SetTooltipToLastWidgetOnHover("When non-zero, the table below is written to the Remix log at this interval. 0 disables periodic logging.");

    if (ImGui::Button("Dump GPU Pass Timings To Log")) {
      logTimings("manual");
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset Window")) {
      std::lock_guard<dxvk::mutex> lock(m_mutex);
      resetStats();
      resizeWindow(std::clamp(averagingFrames(), 1u, kMaxSamples));
    }

    ImGui::Separator();
    ImGui::TextUnformatted("Automated A/B sweep");
    RemixGui::InputText("Sweep Steps", &sweepStepsObject());
    RemixGui::SetTooltipToLastWidgetOnHover("Steps separated by ';', assignments within a step by '&', e.g. rtx.volumetrics.enable=False;rtx.bloom.enable=False. "
                                            "Each step is held, logged, then restored. Baseline tables are logged before and after.");
    RemixGui::DragFloat("Hold Seconds Per Step", &sweepHoldSecondsObject(), 1.0f, 1.0f, 600.0f, "%.0f", ImGuiSliderFlags_AlwaysClamp);

    {
      bool sweepActive = false;
      std::size_t sweepIndex = 0;
      std::size_t sweepCount = 0;
      std::string sweepLabel;
      float sweepHeld = 0.0f;
      {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        sweepActive = m_sweep.active;
        if (sweepActive) {
          sweepIndex = m_sweep.stepIndex;
          sweepCount = m_sweep.steps.size();
          if (sweepIndex < sweepCount) {
            sweepLabel = m_sweep.steps[sweepIndex].label;
          }
          if (m_sweep.stepApplied) {
            sweepHeld = std::chrono::duration<float>(std::chrono::steady_clock::now() - m_sweep.stepStart).count();
          }
        }
      }

      if (!sweepActive) {
        if (ImGui::Button("Run A/B Sweep")) {
          startSweep();
        }
        ImGui::SameLine();
        ImGui::Text("(hotkey: %s)", buildKeyBindDescriptorStringForDisplay(sweepHotkey()).c_str());
      } else {
        if (ImGui::Button("Stop Sweep")) {
          stopSweep();
        }
        ImGui::SameLine();
        ImGui::Text("step %u/%u: %s (%.0f / %.0f s)", static_cast<unsigned>(sweepIndex + 1), static_cast<unsigned>(sweepCount),
                    sweepLabel.c_str(), sweepHeld, sweepHoldSeconds());
      }
    }
    ImGui::Separator();

    if (!enable()) {
      ImGui::TextUnformatted("GPU pass timings are disabled.");
      return;
    }

    std::lock_guard<dxvk::mutex> lock(m_mutex);

    if (m_windowCount == 0) {
      ImGui::TextUnformatted("Waiting for resolved frames...");
      return;
    }

    float busyAvg = 0.0f;
    float spanAvg = 0.0f;
    float listTotalAvg = 0.0f;
    for (std::uint32_t i = 0; i < m_windowCount; ++i) {
      busyAvg += m_busySamples[i];
      spanAvg += m_spanSamples[i];
      listTotalAvg += m_listTotalSamples[i];
    }
    busyAvg /= static_cast<float>(m_windowCount);
    spanAvg /= static_cast<float>(m_windowCount);
    listTotalAvg /= static_cast<float>(m_windowCount);

    ImGui::Text("GPU busy %.3f ms/frame (sum of top-level zones), span %.3f ms/frame, all command lists %.3f ms/frame, %u frames averaged, %u dropped",
                busyAvg, spanAvg, listTotalAvg, m_windowCount, m_droppedFrames);
    {
      const CpuSummary cpu = computeCpuSummary();
      const auto avg = [&cpu](CpuCounter c) { return cpu.avgMs[static_cast<std::size_t>(c)]; };
      ImGui::Text("Frame %.3f ms (%.1f fps) | app thread: work %.3f, Present %.3f (latency %.3f, prev present %.3f, acquire %.3f, Reflex sleep %.3f), CS sync %.3f, event query wait %.3f, resource wait %.3f | CS thread: busy %.3f (injectRTX %.3f, submit back-pressure %.3f), idle %.3f | submit thread: vkQueueSubmit %.3f, vkQueuePresent %.3f",
                  cpu.frameIntervalMs, cpu.frameIntervalMs > 0.0f ? 1000.0f / cpu.frameIntervalMs : 0.0f,
                  cpu.appWorkMs, avg(CpuCounter::AppPresent), avg(CpuCounter::AppPresentWait), avg(CpuCounter::AppPrevPresentWait), avg(CpuCounter::AppAcquireWait), avg(CpuCounter::AppReflexSleep), avg(CpuCounter::AppCsSync),
                  avg(CpuCounter::AppEventQueryWait), avg(CpuCounter::AppResourceWait),
                  avg(CpuCounter::CsBusy), avg(CpuCounter::CsInjectRtx), avg(CpuCounter::CsSubmitBackpressure), cpu.csIdleMs,
                  avg(CpuCounter::SubmitQueueSubmit), avg(CpuCounter::SubmitQueuePresent));
      RemixGui::SetTooltipToLastWidgetOnHover("CPU frame breakdown over the same window. 'work' is the application's own time between Presents including the D3D9 -> Remix "
                                              "draw processing; a frame that is longer than the GPU busy time with a small Present wait is bound by that thread. "
                                              "CS thread 'busy' is the time spent executing the command stream (injectRTX included).");
    }

    const std::vector<DisplayRow> rows = buildRows();
    const float invBusy = busyAvg > 0.0f ? 100.0f / busyAvg : 0.0f;

    constexpr ImGuiTableFlags tableFlags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit;
    if (ImGui::BeginTable("GpuPassTimingsTable", 6, tableFlags)) {
      ImGui::TableSetupColumn("Pass", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("avg ms");
      ImGui::TableSetupColumn("max ms");
      ImGui::TableSetupColumn("last ms");
      ImGui::TableSetupColumn("% busy");
      ImGui::TableSetupColumn("hits");
      ImGui::TableHeadersRow();

      for (const DisplayRow& row : rows) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%*s%s", static_cast<int>(row.depth) * 2, "", row.name.c_str());
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%8.3f", row.avgMs);
        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%8.3f", row.maxMs);
        ImGui::TableSetColumnIndex(3);
        ImGui::Text("%8.3f", row.lastMs);
        ImGui::TableSetColumnIndex(4);
        ImGui::Text("%5.1f", row.avgMs * invBusy);
        ImGui::TableSetColumnIndex(5);
        ImGui::Text("%u", row.hits);
      }

      ImGui::EndTable();
    }
  }

}
