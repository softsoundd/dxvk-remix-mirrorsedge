#pragma once

// game_value_read_number.h — fork-owned Sense component that reads a plugin-
// injected game-state value (written via remixapi_SetGameValue) and emits it
// as a Float. See rtx_fork_game_state.h for the backing store and
// docs/fork-touchpoints.md for the associated upstream touches.

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <string>

#include "../rtx_graph_component_macros.h"
#include "../../rtx_fork_game_state.h"

namespace dxvk {
namespace components {

#define LIST_INPUTS(X) \
  X(RtComponentPropertyType::String, "", key, "Key", "The game-state key to read (e.g. 'windSpeed'). Plugins publish values under this key via remixapi_SetGameValue in remix_c.h.") \
  X(RtComponentPropertyType::Float, 0.0f, defaultValue, "Default", "Value emitted when the key has not been set by any plugin, or when the stored value does not parse as a number.")

#define LIST_STATES(X)

#define LIST_OUTPUTS(X) \
  X(RtComponentPropertyType::Float, 0.0f, value, "Value", "The current value of the game-state key as a float. Accepts any string parseable as a C floating-point literal (trailing whitespace permitted); otherwise the default is emitted.")

REMIX_COMPONENT( \
  /* the Component name */ GameValueReadNumber, \
  /* the UI name */        "Game Value Read Number", \
  /* the UI categories */  "Sense", \
  /* the doc string */     "Reads a plugin-injected game-state value as a float.\n\n" \
    "Plugins write values via remixapi_SetGameValue(key, value) in remix_c.h. " \
    "This component reads the value under the given key and parses it as a float. " \
    "Accepts any string parseable as a C floating-point literal; otherwise the default is emitted.", \
  /* the version number */ 1, \
  LIST_INPUTS, LIST_STATES, LIST_OUTPUTS);

#undef LIST_INPUTS
#undef LIST_STATES
#undef LIST_OUTPUTS

void GameValueReadNumber::updateRange(const Rc<DxvkContext>& context, const size_t start, const size_t end) {
  auto& store = fork_game_state::GameStateStore::get();

  for (size_t i = start; i < end; i++) {
    float value = m_defaultValue[i];

    const std::string& key = m_key[i];
    if (!key.empty()) {
      std::string raw;
      if (store.tryGet(key, raw)) {
        char* endPtr = nullptr;
        errno = 0;
        const float parsed = std::strtof(raw.c_str(), &endPtr);

        // strtof returns 0 and leaves endPtr == c_str() if no conversion was
        // possible; errno is set to ERANGE on over/underflow. Require at
        // least one consumed char, no errno, and only trailing whitespace
        // (or nul) past the parsed prefix.
        const bool consumedSomething = (endPtr != raw.c_str());
        bool trailingOk = false;
        if (consumedSomething && errno == 0) {
          const char* tail = endPtr;
          while (tail && *tail && std::isspace(static_cast<unsigned char>(*tail))) {
            ++tail;
          }
          trailingOk = (tail && *tail == '\0');
        }

        if (trailingOk) {
          value = parsed;
        } else {
          ONCE(Logger::warn(str::format("GameValueReadNumber: value for key '", key, "' is not a number ('", raw, "').")));
        }
      }
    }

    m_value[i] = value;
  }
}

}  // namespace components
}  // namespace dxvk
