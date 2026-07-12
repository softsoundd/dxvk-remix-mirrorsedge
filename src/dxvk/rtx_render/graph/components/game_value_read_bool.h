#pragma once

// game_value_read_bool.h — fork-owned Sense component that reads a plugin-
// injected game-state value (written via remixapi_SetGameValue) and emits it
// as a Bool. See rtx_fork_game_state.h for the backing store and
// docs/fork-touchpoints.md for the associated upstream touches.

#include <cctype>
#include <string>

#include "../rtx_graph_component_macros.h"
#include "../../rtx_fork_game_state.h"

namespace dxvk {
namespace components {

#define LIST_INPUTS(X) \
  X(RtComponentPropertyType::String, "", key, "Key", "The game-state key to read (e.g. 'isRaining'). Plugins publish values under this key via remixapi_SetGameValue in remix_c.h.") \
  X(RtComponentPropertyType::Bool, false, defaultValue, "Default", "Value emitted when the key has not been set by any plugin, or when the stored value does not parse as a bool.")

#define LIST_STATES(X)

#define LIST_OUTPUTS(X) \
  X(RtComponentPropertyType::Bool, false, value, "Value", "The current value of the game-state key as a bool. Accepts '1'/'0'/'true'/'false' (case-insensitive); any other content yields the default.")

REMIX_COMPONENT( \
  /* the Component name */ GameValueReadBool, \
  /* the UI name */        "Game Value Read Bool", \
  /* the UI categories */  "Sense", \
  /* the doc string */     "Reads a plugin-injected game-state value as a bool.\n\n" \
    "Plugins write values via remixapi_SetGameValue(key, value) in remix_c.h. " \
    "This component reads the value under the given key and interprets it as a bool. " \
    "Accepts '1'/'0'/'true'/'false' (case-insensitive); any other content emits the default.", \
  /* the version number */ 1, \
  LIST_INPUTS, LIST_STATES, LIST_OUTPUTS);

#undef LIST_INPUTS
#undef LIST_STATES
#undef LIST_OUTPUTS

void GameValueReadBool::updateRange(const Rc<DxvkContext>& context, const size_t start, const size_t end) {
  auto& store = fork_game_state::GameStateStore::get();

  for (size_t i = start; i < end; i++) {
    bool value = m_defaultValue[i];

    const std::string& key = m_key[i];
    if (!key.empty()) {
      std::string raw;
      if (store.tryGet(key, raw)) {
        // Case-insensitive comparison against a small accepted set.
        std::string lower;
        lower.reserve(raw.size());
        for (char c : raw) {
          lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }

        if (lower == "1" || lower == "true") {
          value = true;
        } else if (lower == "0" || lower == "false") {
          value = false;
        } else {
          ONCE(Logger::warn(str::format("GameValueReadBool: value for key '", key, "' is not a bool ('", raw, "').")));
        }
      }
    }

    m_value[i] = value;
  }
}

}  // namespace components
}  // namespace dxvk
