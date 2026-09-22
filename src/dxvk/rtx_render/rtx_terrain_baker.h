#pragma once

#include "rtx_option.h"
#include "rtx_replacement_material_texture_type.h"

namespace dxvk {

  class TerrainBaker {
  public:
    struct Material {
      RTX_OPTION("rtx.terrainBaker.material", bool, replacementSupportInPS_fixedFunction, true,
                 "Enables reading of secondary PBR replacement textures in pixel shaders for fixed function pipelines.\n"
                 "Must be set at launch to apply.");
      RTX_OPTION_ENV("rtx.terrainBaker.material", bool, replacementSupportInPS_programmableShaders, true, "RTX_TERRAIN_BAKER_PS_REPLACEMENT_SUPPORT_IN_PROGRAMMABLE_SHADERS",
                     "Enables reading of secondary PBR replacement textures in pixel shaders for programmable pipelines.\n"
                     "Must be set at launch to apply. Limited to Shader Model 1.0; Shader Model 2.0 and above skip this path.");
    };
  };

}
