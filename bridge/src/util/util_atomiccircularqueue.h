/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "util_common.h"

#include "../tracy/tracy.hpp"

#include <cstdio>
#include <atomic>
#include <assert.h>
#include <vector>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace bridge_util {

  // Intra/Inter-process thread safe, shared circular queue.
  // Constructed from a shared pool of memory - and synchronized using static atomics
  // Single Producer, Single Consumer ONLY!
  template<typename T, bridge_util::Accessor Accessor>
  class AtomicCircularQueue {
    // Next slot the reader takes; only the reader stores it.
    std::atomic<uint32_t>* m_write;
    // Next slot the writer fills; only the writer stores it.
    std::atomic<uint32_t>* m_read;

    T* m_data;
    T m_default;
    T m_pulled;

    const size_t m_queueSize;

    // Each side's last seen value of the other side's index, so the shared one, which the other
    // process keeps storing to, is only re-read when this says the queue is full (writer) or empty
    // (reader). A stale value is always behind the real one: the queue just looks fuller or emptier.
    uint32_t m_cachedWrite = 0;
    mutable uint32_t m_cachedRead = 0;

    static const size_t kAlignment = 128;
    static const size_t kWriteAtomicOffset = 0;
    static const size_t kReadAtomicOffset = kAlignment + kWriteAtomicOffset;
    static const size_t kMemoryPoolOffset = kAlignment + kReadAtomicOffset;

  public:
    static size_t getExtraMemoryRequirements() {
      return kMemoryPoolOffset;
    }

    AtomicCircularQueue(const std::string& name, void* pMemory, const size_t memSize, const size_t queueSize)
      : m_queueSize(queueSize)
      , m_data(nullptr) // INIT
    {
      // Ensure we have enough memory
      assert(memSize > kMemoryPoolOffset);
      assert(memSize - kMemoryPoolOffset >= queueSize * sizeof(T));

      // Writers own the memory, Readers are consumers
      if constexpr(IS_READER(Accessor)) {
        m_data = (T*) ((uintptr_t) pMemory + kMemoryPoolOffset);
        m_write = (std::atomic<uint32_t>*)((uintptr_t) pMemory + kWriteAtomicOffset);
        m_read = (std::atomic<uint32_t>*)((uintptr_t) pMemory + kReadAtomicOffset);
      } else if constexpr (IS_WRITER(Accessor)) {
        m_data = new((void*) ((uintptr_t) pMemory + kMemoryPoolOffset)) T[m_queueSize];
        m_write = new((void*) ((uintptr_t) pMemory + kWriteAtomicOffset)) std::atomic<uint32_t>(0);
        m_read = new((void*) ((uintptr_t) pMemory + kReadAtomicOffset)) std::atomic<uint32_t>(0);
      }

      assert(m_read->is_lock_free() && m_write->is_lock_free()); // Must be runtime check as it's CPU specific

      // Both start at the reader's index: current for the writer, and "empty" for the reader, which
      // then fetches m_read on first use.
      m_cachedWrite = m_write->load(std::memory_order_acquire);
      m_cachedRead = m_write->load(std::memory_order_acquire);
    }

    AtomicCircularQueue(const AtomicCircularQueue& q) = delete;

    ~AtomicCircularQueue() {
    }

    // Push object to queue
    Result push(const T& obj) {
      const auto currentRead = m_read->load(std::memory_order_relaxed);
      const auto nextRead = queueIdxInc(currentRead);
      if (nextRead == m_cachedWrite) {
        // Full as of the last seen reader index: fetch the current one, and wait while it really is full.
        // Acquire, so the reader is done with a slot before it is overwritten.
        ULONGLONG start = 0, curTick;
        const DWORD timeoutMS = GlobalOptions::getCommandTimeout();
        while ((m_cachedWrite = m_write->load(std::memory_order_acquire)) == nextRead) {
          std::this_thread::yield();

          curTick = GetTickCount64();
          start = start > 0 ? start : curTick;
          if (timeoutMS != 0 && start + timeoutMS <= curTick) {
            return Result::Failure;
          }
        }
      }

      m_data[currentRead] = obj;
      // Release: the slot's contents become visible to the reader before the index that publishes it.
      m_read->store(nextRead, std::memory_order_release);
      return Result::Success;
    }

    // Does nothing but wait for the next command to come in
    Result try_peek(const DWORD timeoutMS = 0) {
      return Result::Success;
    }

    // Returns a ref to the first element in the queue
    // Note: Blocks if the queue is empty
    const T& peek(Result& result, const DWORD timeoutMS = 0, std::atomic<bool>* const pbEarlyOutSignal = nullptr) const {
      const auto currentWrite = m_write->load(std::memory_order_relaxed);
      if (!waitForData(currentWrite, timeoutMS, pbEarlyOutSignal)) {
        result = Result::Timeout;
        return m_default;
      }
      result = Result::Success;
      return m_data[currentWrite];
    }

    // Returns a copy to the first element in queue, AND removes it
    // Note: Blocks if queue is empty
    const T& pull(Result& result, const DWORD timeoutMS = 0, std::atomic<bool>* const pbEarlyOutSignal = nullptr) {
      const auto currentWrite = m_write->load(std::memory_order_relaxed);
      if (!waitForData(currentWrite, timeoutMS, pbEarlyOutSignal)) {
        result = Result::Timeout;
        return m_default;
      }
      // Copy the slot out before handing it back to the writer.
      m_pulled = m_data[currentWrite];
      m_write->store(queueIdxInc(currentWrite), std::memory_order_release);
      result = Result::Success;
      return m_pulled;
    }

    // Check for queue emptiness. The function may guarantee a correct result
    // ONLY when the queue is stalled on either end and only a single index
    // is advancing.
    bool isEmpty() const {
      const auto currentWrite = m_write->load(std::memory_order_relaxed);
      return currentWrite == m_read->load(std::memory_order_acquire);
    }

    std::vector<Commands::D3D9Command> buildQueueData(int maxQueueElements, int currentIndex) {
      std::vector<Commands::D3D9Command> commandHistory;
      int itemCount = 0;
      while (itemCount < m_queueSize && itemCount < maxQueueElements) {
        // To prevent adding default commands in the Queue to the command list
        if (m_data[currentIndex].command == Commands::Bridge_Invalid)
          break;
        commandHistory.push_back(m_data[currentIndex].command);
        currentIndex = queueIdxDec(currentIndex);
        ++itemCount;
      }
      return commandHistory;
    }

    std::vector<Commands::D3D9Command> getWriterQueueData(int maxQueueElements=10) {
      const auto currentRead = m_read->load(std::memory_order_relaxed);
      int currentIndex = queueIdxDec(currentRead);
      return buildQueueData(maxQueueElements, currentIndex);
    }

    std::vector<Commands::D3D9Command> getReaderQueueData(int maxQueueElements = 10) {
      const auto currentWrite = m_write->load(std::memory_order_relaxed);
      int currentIndex = queueIdxDec(currentWrite);
      return buildQueueData(maxQueueElements, currentIndex);
    }

    uint32_t queueIdxInc(uint32_t idx) const {
      return idx + 1 < m_queueSize ? idx + 1 : 0;
    }

    uint32_t queueIdxDec(uint32_t idx) const {
      return idx == 0 ?  m_queueSize - 1 : idx - 1 ;
    }

  private:
    // Waits until the slot at currentWrite (the reader's index) has been filled. False on timeout or
    // early out. Acquire, so the slot's contents are visible once its index is.
    bool waitForData(const uint32_t currentWrite, const DWORD timeoutMS, std::atomic<bool>* const pbEarlyOutSignal) const {
      if (currentWrite != m_cachedRead) {
        return true;
      }

      ULONGLONG start = 0, curTick;
      while ((m_cachedRead = m_read->load(std::memory_order_acquire)) == currentWrite) {
        std::this_thread::yield();

        curTick = GetTickCount64();
        start = start > 0 ? start : curTick;

        if (pbEarlyOutSignal && pbEarlyOutSignal->load()) {
          return false;
        }
        if (timeoutMS != 0 && start + timeoutMS <= curTick) {
          return false;
        }
      }
      return true;
    }
  };

}