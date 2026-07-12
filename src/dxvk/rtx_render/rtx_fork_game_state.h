#pragma once

// rtx_fork_game_state.h — fork-owned generic key/value store for plugin-
// injected runtime state that graph components read by name. The store is a
// header-only singleton guarded by an internal mutex. Writes happen on the
// plugin thread via remixapi_SetGameValue; reads happen on the render thread
// from GameValueRead* graph components.
//
// Lifetime is deliberately API-lifetime (function-local static): the store
// survives Remix Shutdown / re-init so plugins do not have to re-populate
// their state across device resets.
//
// See docs/fork-touchpoints.md for the upstream file touches
// associated with the remixapi_SetGameValue entrypoint.

#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace dxvk {
  namespace fork_game_state {

    class GameStateStore {
    public:
      static GameStateStore& get() {
        static GameStateStore s_instance;
        return s_instance;
      }

      // Overwrites the value under key. Thread-safe.
      void set(const std::string& key, std::string value) {
        std::lock_guard<std::mutex> lock{ m_lock };
        m_values[key] = std::move(value);
      }

      // Copies the current value under key into out. Returns false if the
      // key is not present. Thread-safe.
      bool tryGet(const std::string& key, std::string& out) const {
        std::lock_guard<std::mutex> lock{ m_lock };
        auto it = m_values.find(key);
        if (it == m_values.end()) {
          return false;
        }
        out = it->second;
        return true;
      }

    private:
      GameStateStore() = default;
      GameStateStore(const GameStateStore&) = delete;
      GameStateStore& operator=(const GameStateStore&) = delete;

      mutable std::mutex m_lock;
      std::unordered_map<std::string, std::string> m_values;
    };

  }  // namespace fork_game_state
}  // namespace dxvk
