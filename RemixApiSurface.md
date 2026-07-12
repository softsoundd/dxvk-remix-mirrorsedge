# Remix C API Surface

> Auto-generated from [`public/include/remix/remix_c.h`](public/include/remix/remix_c.h)
> by `scripts-common/generate_remix_api_md.py`. **Do not hand-edit.**
> Regenerate via `scripts/regen-docs.ps1` from the repo root.

This page is a flat exhaustive reference of every type, function pointer,
enum, struct, and macro that a plugin or host application compiles
against. For the conceptual / mental-model walkthrough of how to USE
the API, see [`docs/RemixApi.md`](docs/RemixApi.md).

---

## Macros and version

| Name | Value |
| :-- | :-- |
| `REMIX_C_H_` | `` |
| `REMIXAPI_CALL` | `__stdcall` |
| `REMIXAPI_PTR` | `REMIXAPI_CALL` |
| `REMIXAPI` | `__declspec(dllexport)` |
| `REMIXAPI` | `__declspec(dllimport)` |
| `REMIXAPI_VERSION_MAKE(major, minor, patch)` | `(      (((uint64_t)(major)) << 48) \|      (((uint64_t)(minor)) << 16) \|      (((uint64_t)(patch))      ) )` |
| `REMIXAPI_VERSION_GET_MAJOR(version)` | `(((uint64_t)(version) >> 48) & (uint64_t)0xFFFF)` |
| `REMIXAPI_VERSION_GET_MINOR(version)` | `(((uint64_t)(version) >> 16) & (uint64_t)0xFFFFFFFF)` |
| `REMIXAPI_VERSION_GET_PATCH(version)` | `(((uint64_t)(version)      ) & (uint64_t)0xFFFF)` |
| `REMIXAPI_VERSION_MAJOR` | `0` |
| `REMIXAPI_VERSION_MINOR` | `1000` |
| `REMIXAPI_VERSION_PATCH` | `0` |
| `REMIX_WINAPI_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` | `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` |
| `REMIX_WINAPI_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS` | `LOAD_LIBRARY_SEARCH_DEFAULT_DIRS` |
| `REMIX_WINAPI_MAX_PATH` | `MAX_PATH` |
| `REMIX_WINAPI_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` | `0x00000100` |
| `REMIX_WINAPI_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS` | `0x00001000` |
| `REMIX_WINAPI_MAX_PATH` | `260` |
| `REMIXAPI_INSTANCE_INFO_MAX_BONES_COUNT` | `256` |

---

## Typedefs and handles

| Name | Aliased type |
| :-- | :-- |
| `IDirect3D9Ex` | `struct IDirect3D9Ex` |
| `IDirect3DDevice9Ex` | `struct IDirect3DDevice9Ex` |
| `IDirect3DSurface9` | `struct IDirect3DSurface9` |
| `IDirect3DTexture9` | `struct IDirect3DTexture9` |
| `remixapi_HWND` | `HWND` |
| `remixapi_HMODULE` | `HMODULE` |
| `remixapi_loader_PROC` | `FARPROC` |
| `remixapi_HWND` | `struct HWND__*` |
| `remixapi_HMODULE` | `struct HINSTANCE__*` |
| `remixapi_Bool` | `uint32_t` |
| `remixapi_MaterialHandle` | `struct remixapi_MaterialHandle_T*` |
| `remixapi_MeshHandle` | `struct remixapi_MeshHandle_T*` |
| `remixapi_LightHandle` | `struct remixapi_LightHandle_T*` |
| `remixapi_TextureHandle` | `struct remixapi_TextureHandle_T*` |
| `remixapi_Path` | `const wchar_t*` |
| `remixapi_InstanceCategoryFlags` | `uint32_t` |

---

## Enums

### `remixapi_StructType`

| Member | Value | Notes |
| :-- | :-: | :-- |
| `REMIXAPI_STRUCT_TYPE_NONE` | `0` |  |
| `REMIXAPI_STRUCT_TYPE_INITIALIZE_LIBRARY_INFO` | `1` |  |
| `REMIXAPI_STRUCT_TYPE_MATERIAL_INFO` | `2` |  |
| `REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_PORTAL_EXT` | `3` |  |
| `REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_TRANSLUCENT_EXT` | `4` |  |
| `REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_OPAQUE_EXT` | `5` |  |
| `REMIXAPI_STRUCT_TYPE_LIGHT_INFO` | `6` |  |
| `REMIXAPI_STRUCT_TYPE_LIGHT_INFO_DISTANT_EXT` | `7` |  |
| `REMIXAPI_STRUCT_TYPE_LIGHT_INFO_CYLINDER_EXT` | `8` |  |
| `REMIXAPI_STRUCT_TYPE_LIGHT_INFO_DISK_EXT` | `9` |  |
| `REMIXAPI_STRUCT_TYPE_LIGHT_INFO_RECT_EXT` | `10` |  |
| `REMIXAPI_STRUCT_TYPE_LIGHT_INFO_SPHERE_EXT` | `11` |  |
| `REMIXAPI_STRUCT_TYPE_MESH_INFO` | `12` |  |
| `REMIXAPI_STRUCT_TYPE_INSTANCE_INFO` | `13` |  |
| `REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_BONE_TRANSFORMS_EXT` | `14` |  |
| `REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_BLEND_EXT` | `15` |  |
| `REMIXAPI_STRUCT_TYPE_CAMERA_INFO` | `16` |  |
| `REMIXAPI_STRUCT_TYPE_CAMERA_INFO_PARAMETERIZED_EXT` | `17` |  |
| `REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_OPAQUE_SUBSURFACE_EXT` | `18` |  |
| `REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_OBJECT_PICKING_EXT` | `19` |  |
| `REMIXAPI_STRUCT_TYPE_LIGHT_INFO_DOME_EXT` | `20` |  |
| `REMIXAPI_STRUCT_TYPE_LIGHT_INFO_USD_EXT` | `21` |  |
| `REMIXAPI_STRUCT_TYPE_STARTUP_INFO` | `22` |  |
| `REMIXAPI_STRUCT_TYPE_PRESENT_INFO` | `23` |  |
| `REMIXAPI_STRUCT_TYPE_DEPRECATED_LEGACY_PARTICLE_SYSTEM` | `24` |  |
| `REMIXAPI_STRUCT_TYPE_TEXTURE_INFO` | `25` |  |
| `REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_PARTICLE_SYSTEM_EXT` | `26` |  |
| `REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_GPU_INSTANCING_EXT` | `27` |  |
| `REMIXAPI_STRUCT_TYPE_CAMERA_MEDIUM_INFO` | `28` |  |

### `remixapi_ErrorCode`

| Member | Value | Notes |
| :-- | :-: | :-- |
| `REMIXAPI_ERROR_CODE_SUCCESS` | `0` |  |
| `REMIXAPI_ERROR_CODE_GENERAL_FAILURE` | `1` |  |
| `REMIXAPI_ERROR_CODE_LOAD_LIBRARY_FAILURE` | `2` | WinAPI's LoadLibrary has failed |
| `REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS` | `3` |  |
| `REMIXAPI_ERROR_CODE_GET_PROC_ADDRESS_FAILURE` | `4` | Couldn't find 'remixInitialize' function in the .dll |
| `REMIXAPI_ERROR_CODE_ALREADY_EXISTS` | `5` | CreateD3D9 / RegisterD3D9Device can be called only once |
| `REMIXAPI_ERROR_CODE_REGISTERING_NON_REMIX_D3D9_DEVICE` | `6` | RegisterD3D9Device requires the device that was created with IDirect3DDevice9Ex, returned by CreateD3D9 |
| `REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED` | `7` | RegisterD3D9Device was not called |
| `REMIXAPI_ERROR_CODE_INCOMPATIBLE_VERSION` | `8` |  |
| `REMIXAPI_ERROR_CODE_SET_DLL_DIRECTORY_FAILURE` | `9` | WinAPI's SetDllDirectory has failed |
| `REMIXAPI_ERROR_CODE_GET_FULL_PATH_NAME_FAILURE` | `10` | WinAPI's GetFullPathName has failed |
| `REMIXAPI_ERROR_CODE_NOT_INITIALIZED` | `11` |  |
| `REMIXAPI_ERROR_CODE_HRESULT_NO_REQUIRED_GPU_FEATURES` | `0x88960001` | Error codes that are encoded as HRESULT, i.e. returned from D3D9 functions. Look MAKE_D3DHRESULT, but with _FACD3D=0x896, instead of D3D9's 0x876 |
| `REMIXAPI_ERROR_CODE_HRESULT_DRIVER_VERSION_BELOW_MINIMUM` | `0x88960002` |  |
| `REMIXAPI_ERROR_CODE_HRESULT_DXVK_INSTANCE_EXTENSION_FAIL` | `0x88960003` |  |
| `REMIXAPI_ERROR_CODE_HRESULT_VK_CREATE_INSTANCE_FAIL` | `0x88960004` |  |
| `REMIXAPI_ERROR_CODE_HRESULT_VK_CREATE_DEVICE_FAIL` | `0x88960005` |  |
| `REMIXAPI_ERROR_CODE_HRESULT_GRAPHICS_QUEUE_FAMILY_MISSING` | `0x88960006` |  |

### `remixapi_CameraType`

| Member | Value | Notes |
| :-- | :-: | :-- |
| `REMIXAPI_CAMERA_TYPE_WORLD` | `(implicit)` |  |
| `REMIXAPI_CAMERA_TYPE_SKY` | `(implicit)` |  |
| `REMIXAPI_CAMERA_TYPE_VIEW_MODEL` | `(implicit)` |  |

### `remixapi_UIState`

| Member | Value | Notes |
| :-- | :-: | :-- |
| `REMIXAPI_UI_STATE_NONE` | `0` |  |
| `REMIXAPI_UI_STATE_BASIC` | `1` |  |
| `REMIXAPI_UI_STATE_ADVANCED` | `2` |  |

### `remixapi_InstanceCategoryBit`

Bit values match upstream NVIDIA dxvk-remix exactly. These are a published
ABI contract — do NOT reorder or reassign. (An earlier fork build shifted
IGNORE_ALPHA_CHANNEL to bit 8 to mirror the internal InstanceCategories
order; that was reverted because the C<->internal mapping is by-name in
toRtCategories(), so the bit values are free to match upstream and must.)

| Member | Value | Notes |
| :-- | :-: | :-- |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_WORLD_UI` | `1 << 0` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_WORLD_MATTE` | `1 << 1` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_SKY` | `1 << 2` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE` | `1 << 3` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE_LIGHTS` | `1 << 4` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE_ANTI_CULLING` | `1 << 5` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE_MOTION_BLUR` | `1 << 6` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE_OPACITY_MICROMAP` | `1 << 7` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_HIDDEN` | `1 << 8` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_PARTICLE` | `1 << 9` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_BEAM` | `1 << 10` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_DECAL_STATIC` | `1 << 11` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_DECAL_DYNAMIC` | `1 << 12` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_DECAL_SINGLE_OFFSET` | `1 << 13` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_DECAL_NO_OFFSET` | `1 << 14` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_ALPHA_BLEND_TO_CUTOUT` | `1 << 15` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_TERRAIN` | `1 << 16` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_ANIMATED_WATER` | `1 << 17` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_THIRD_PERSON_PLAYER_MODEL` | `1 << 18` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_THIRD_PERSON_PLAYER_BODY` | `1 << 19` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE_BAKED_LIGHTING` | `1 << 20` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE_ALPHA_CHANNEL` | `1 << 21` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE_TRANSPARENCY_LAYER` | `1 << 22` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_PARTICLE_EMITTER` | `1 << 23` |  |
| `REMIXAPI_INSTANCE_CATEGORY_BIT_SMOOTH_NORMALS` | `1 << 24` |  |

### `remixapi_dxvk_CopyRenderingOutputType`

| Member | Value | Notes |
| :-- | :-: | :-- |
| `REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_FINAL_COLOR` | `0` |  |
| `REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_DEPTH` | `1` |  |
| `REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_NORMALS` | `2` |  |
| `REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_OBJECT_PICKING` | `3` |  |

### `remixapi_Format`

Texture upload API

| Member | Value | Notes |
| :-- | :-: | :-- |
| `REMIXAPI_FORMAT_R8G8B8A8_UNORM` | `37,` |  |
| `REMIXAPI_FORMAT_R8G8B8A8_SRGB` | `43,` |  |
| `REMIXAPI_FORMAT_B8G8R8A8_UNORM` | `44,` |  |
| `REMIXAPI_FORMAT_B8G8R8A8_SRGB` | `50,` |  |
| `REMIXAPI_FORMAT_BC1_RGB_UNORM` | `131,` |  |
| `REMIXAPI_FORMAT_BC1_RGB_SRGB` | `132,` |  |
| `REMIXAPI_FORMAT_BC3_UNORM` | `135,` |  |
| `REMIXAPI_FORMAT_BC3_SRGB` | `136,` |  |
| `REMIXAPI_FORMAT_BC5_UNORM` | `139,` |  |
| `REMIXAPI_FORMAT_BC7_UNORM` | `145,` |  |
| `REMIXAPI_FORMAT_BC7_SRGB` | `146,` |  |

---

## Structs

### `remixapi_Rect2D`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `int32_t` | `left` |  |
| `int32_t` | `top` |  |
| `int32_t` | `right` |  |
| `int32_t` | `bottom` |  |

### `remixapi_Float2D`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `float` | `x` |  |
| `float` | `y` |  |

### `remixapi_Float3D`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `float` | `x` |  |
| `float` | `y` |  |
| `float` | `z` |  |

### `remixapi_Float4D`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `float` | `x` |  |
| `float` | `y` |  |
| `float` | `z` |  |
| `float` | `w` |  |

### `remixapi_Transform`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `float` | `matrix[3][4]` |  |

### `remixapi_StartupInfo`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `remixapi_HWND` | `hwnd` |  |
| `remixapi_Bool` | `disableSrgbConversionForOutput` |  |
| `remixapi_Bool` | `forceNoVkSwapchain` | If true, 'dxvk_GetExternalSwapchain' can be used to retrieve a raw VkImage, so the application can present it, for example by using OpenGL interop: converting VkImage to OpenGL, and presenting it via OpenGL. Default: false. Use VkSwapchainKHR to present frame into HWND. |
| `remixapi_Bool` | `editorModeEnabled` |  |

### `remixapi_MaterialInfoOpaqueEXT`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `remixapi_Path` | `roughnessTexture` |  |
| `remixapi_Path` | `metallicTexture` |  |
| `float` | `anisotropy` |  |
| `remixapi_Float3D` | `albedoConstant` |  |
| `float` | `opacityConstant` |  |
| `float` | `roughnessConstant` |  |
| `float` | `metallicConstant` |  |
| `remixapi_Bool` | `thinFilmThickness_hasvalue` |  |
| `float` | `thinFilmThickness_value` |  |
| `remixapi_Bool` | `alphaIsThinFilmThickness` |  |
| `remixapi_Path` | `heightTexture` |  |
| `float` | `displaceIn` |  |
| `remixapi_Bool` | `useDrawCallAlphaState` | If true, InstanceInfoBlendEXT is used as a source for alpha state |
| `remixapi_Bool` | `blendType_hasvalue` |  |
| `int` | `blendType_value` |  |
| `remixapi_Bool` | `invertedBlend` |  |
| `int` | `alphaTestType` |  |
| `uint8_t` | `alphaReferenceValue` |  |
| `float` | `displaceOut` |  |

### `remixapi_MaterialInfoOpaqueSubsurfaceEXT`

Valid only if remixapi_MaterialInfo contains remixapi_MaterialInfoOpaqueEXT in pNext chain

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `remixapi_Path` | `subsurfaceTransmittanceTexture` |  |
| `remixapi_Path` | `subsurfaceThicknessTexture` |  |
| `remixapi_Path` | `subsurfaceSingleScatteringAlbedoTexture` |  |
| `remixapi_Float3D` | `subsurfaceTransmittanceColor` |  |
| `float` | `subsurfaceMeasurementDistance` |  |
| `remixapi_Float3D` | `subsurfaceSingleScatteringAlbedo` |  |
| `float` | `subsurfaceVolumetricAnisotropy` |  |
| `remixapi_Bool` | `subsurfaceDiffusionProfile` |  |
| `remixapi_Float3D` | `subsurfaceRadius` |  |
| `float` | `subsurfaceRadiusScale` |  |
| `float` | `subsurfaceMaxSampleRadius` |  |
| `remixapi_Path` | `subsurfaceRadiusTexture` |  |

### `remixapi_MaterialInfoTranslucentEXT`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `remixapi_Path` | `transmittanceTexture` |  |
| `float` | `refractiveIndex` |  |
| `remixapi_Float3D` | `transmittanceColor` |  |
| `float` | `transmittanceMeasurementDistance` |  |
| `remixapi_Bool` | `thinWallThickness_hasvalue` |  |
| `float` | `thinWallThickness_value` |  |
| `remixapi_Bool` | `useDiffuseLayer` |  |

### `remixapi_MaterialInfoPortalEXT`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `uint8_t` | `rayPortalIndex` |  |
| `float` | `rotationSpeed` |  |

### `remixapi_MaterialInfo`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `uint64_t` | `hash` |  |
| `remixapi_Path` | `albedoTexture` |  |
| `remixapi_Path` | `normalTexture` |  |
| `remixapi_Path` | `tangentTexture` |  |
| `remixapi_Path` | `emissiveTexture` |  |
| `float` | `emissiveIntensity` |  |
| `remixapi_Float3D` | `emissiveColorConstant` |  |
| `uint8_t` | `spriteSheetRow` |  |
| `uint8_t` | `spriteSheetCol` |  |
| `uint8_t` | `spriteSheetFps` |  |
| `uint8_t` | `filterMode` |  |
| `uint8_t` | `wrapModeU` |  |
| `uint8_t` | `wrapModeV` |  |

### `remixapi_HardcodedVertex`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `float` | `position[3]` |  |
| `float` | `normal[3]` |  |
| `float` | `texcoord[2]` |  |
| `uint32_t` | `color` |  |
| `uint32_t` | `_pad0` |  |
| `uint32_t` | `_pad1` |  |
| `uint32_t` | `_pad2` |  |
| `uint32_t` | `_pad3` |  |
| `uint32_t` | `_pad4` |  |
| `uint32_t` | `_pad5` |  |
| `uint32_t` | `_pad6` |  |

### `remixapi_MeshInfoSkinning`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `uint32_t` | `bonesPerVertex` |  |
| `const float*` | `blendWeights_values` | Each tuple of 'bonesPerVertex' float-s defines a vertex. I.e. the size must be (bonesPerVertex * vertexCount). |
| `uint32_t` | `blendWeights_count` |  |
| `const uint32_t*` | `blendIndices_values` | Each tuple of 'bonesPerVertex' uint32_t-s defines a vertex. I.e. the size must be (bonesPerVertex * vertexCount). |
| `uint32_t` | `blendIndices_count` |  |

### `remixapi_MeshInfoSurfaceTriangles`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `const remixapi_HardcodedVertex*` | `vertices_values` |  |
| `uint64_t` | `vertices_count` |  |
| `const uint32_t*` | `indices_values` |  |
| `uint64_t` | `indices_count` |  |
| `remixapi_Bool` | `skinning_hasvalue` |  |
| `remixapi_MeshInfoSkinning` | `skinning_value` |  |
| `remixapi_MaterialHandle` | `material` |  |

### `remixapi_MeshInfo`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `uint64_t` | `hash` |  |
| `const remixapi_MeshInfoSurfaceTriangles*` | `surfaces_values` |  |
| `uint32_t` | `surfaces_count` |  |

### `remixapi_CameraInfoParameterizedEXT`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `remixapi_Float3D` | `position` |  |
| `remixapi_Float3D` | `forward` |  |
| `remixapi_Float3D` | `up` |  |
| `remixapi_Float3D` | `right` |  |
| `float` | `fovYInDegrees` |  |
| `float` | `aspect` |  |
| `float` | `nearPlane` |  |
| `float` | `farPlane` |  |

### `remixapi_CameraInfo`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `remixapi_CameraType` | `type` |  |
| `float` | `view[4][4]` |  |
| `float` | `projection[4][4]` |  |

### `remixapi_CameraMediumInfo`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `remixapi_MaterialHandle` | `medium` |  |

### `remixapi_InstanceInfoBoneTransformsEXT`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `const remixapi_Transform*` | `boneTransforms_values` |  |
| `uint32_t` | `boneTransforms_count` |  |

### `remixapi_InstanceInfoBlendEXT`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `remixapi_Bool` | `alphaTestEnabled` |  |
| `uint8_t` | `alphaTestReferenceValue` |  |
| `uint32_t` | `alphaTestCompareOp` |  |
| `remixapi_Bool` | `alphaBlendEnabled` |  |
| `uint32_t` | `srcColorBlendFactor` |  |
| `uint32_t` | `dstColorBlendFactor` |  |
| `uint32_t` | `colorBlendOp` |  |
| `uint32_t` | `textureColorArg1Source` |  |
| `uint32_t` | `textureColorArg2Source` |  |
| `uint32_t` | `textureColorOperation` |  |
| `uint32_t` | `textureAlphaArg1Source` |  |
| `uint32_t` | `textureAlphaArg2Source` |  |
| `uint32_t` | `textureAlphaOperation` |  |
| `uint32_t` | `tFactor` |  |
| `remixapi_Bool` | `isTextureFactorBlend` |  |
| `uint32_t` | `srcAlphaBlendFactor` |  |
| `uint32_t` | `dstAlphaBlendFactor` |  |
| `uint32_t` | `alphaBlendOp` |  |
| `uint32_t` | `writeMask` |  |
| `remixapi_Bool` | `isVertexColorBakedLighting` |  |

### `remixapi_InstanceInfoObjectPickingEXT`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `uint32_t` | `objectPickingValue` | A value to write for 'RequestObjectPicking' |

### `remixapi_AnimatedFloat1D`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `float*` | `pData` |  |
| `uint32_t` | `numberElements` |  |

### `remixapi_AnimatedFloat2D`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_Float2D*` | `pData` |  |
| `uint32_t` | `numberElements` |  |

### `remixapi_AnimatedFloat3D`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_Float3D*` | `pData` |  |
| `uint32_t` | `numberElements` |  |

### `remixapi_AnimatedFloat4D`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_Float4D*` | `pData` |  |
| `uint32_t` | `numberElements` |  |

### `remixapi_InstanceInfoParticleSystemEXT`

New particle system struct with animated curve support

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `uint32_t` | `maxNumParticles` |  |
| `remixapi_Bool` | `useTurbulence` |  |
| `remixapi_Bool` | `alignParticlesToVelocity` |  |
| `remixapi_Bool` | `useSpawnTexcoords` |  |
| `remixapi_Bool` | `enableCollisionDetection` |  |
| `remixapi_Bool` | `enableMotionTrail` |  |
| `remixapi_Bool` | `hideEmitter` |  |
| `remixapi_Bool` | `restrictVelocityX` |  |
| `remixapi_Bool` | `restrictVelocityY` |  |
| `remixapi_Bool` | `restrictVelocityZ` |  |
| `remixapi_AnimatedFloat4D` | `minColor` |  |
| `remixapi_AnimatedFloat4D` | `maxColor` |  |
| `remixapi_AnimatedFloat1D` | `minRotationSpeed` |  |
| `remixapi_AnimatedFloat1D` | `maxRotationSpeed` |  |
| `remixapi_AnimatedFloat2D` | `minSize` |  |
| `remixapi_AnimatedFloat2D` | `maxSize` |  |
| `remixapi_AnimatedFloat3D` | `maxVelocity` |  |
| `remixapi_Float3D` | `attractorPosition` |  |
| `float` | `minTimeToLive` |  |
| `float` | `maxTimeToLive` |  |
| `float` | `initialVelocityFromNormal` |  |
| `float` | `initialVelocityConeAngleDegrees` |  |
| `float` | `dragCoefficient` |  |
| `float` | `initialRotationDeviationDegrees` |  |
| `float` | `gravityForce` |  |
| `float` | `turbulenceFrequency` |  |
| `float` | `turbulenceForce` |  |
| `float` | `spawnRatePerSecond` |  |
| `float` | `collisionThickness` |  |
| `float` | `collisionRestitution` |  |
| `float` | `motionTrailMultiplier` |  |
| `float` | `initialVelocityFromMotion` |  |
| `float` | `spawnBurstDuration` |  |
| `float` | `attractorRadius` |  |
| `float` | `attractorForce` |  |
| `uint8_t` | `billboardType` |  |
| `uint8_t` | `spriteSheetMode` |  |
| `uint8_t` | `collisionMode` |  |
| `uint8_t` | `randomFlipAxis` |  |

### `remixapi_InstanceInfoGpuInstancingEXT`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `const remixapi_Transform*` | `instanceTransforms_values` |  |
| `uint32_t` | `instanceTransforms_count` |  |

### `remixapi_InstanceInfo`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `remixapi_InstanceCategoryFlags` | `categoryFlags` |  |
| `remixapi_MeshHandle` | `mesh` |  |
| `remixapi_Transform` | `transform` |  |
| `remixapi_Bool` | `doubleSided` |  |

### `remixapi_LightInfoLightShaping`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_Float3D` | `direction` | The direction the Light Shaping is pointing in. Must be normalized. |
| `float` | `coneAngleDegrees` |  |
| `float` | `coneSoftness` |  |
| `float` | `focusExponent` |  |

### `remixapi_LightInfoSphereEXT`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `remixapi_Float3D` | `position` |  |
| `float` | `radius` |  |
| `remixapi_Bool` | `shaping_hasvalue` |  |
| `remixapi_LightInfoLightShaping` | `shaping_value` |  |
| `float` | `volumetricRadianceScale` |  |

### `remixapi_LightInfoRectEXT`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `remixapi_Float3D` | `position` |  |
| `remixapi_Float3D` | `xAxis` | The X axis of the Rect Light. Must be normalized and orthogonal to the Y and direction axes. |
| `float` | `xSize` |  |
| `remixapi_Float3D` | `yAxis` | The Y axis of the Rect Light. Must be normalized and orthogonal to the X and direction axes. |
| `float` | `ySize` |  |
| `remixapi_Float3D` | `direction` | The direction the Rect Light is pointing in, should match the Shaping direction if present. Must be normalized and orthogonal to the X and Y axes. |
| `remixapi_Bool` | `shaping_hasvalue` |  |
| `remixapi_LightInfoLightShaping` | `shaping_value` |  |
| `float` | `volumetricRadianceScale` |  |

### `remixapi_LightInfoDiskEXT`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `remixapi_Float3D` | `position` |  |
| `remixapi_Float3D` | `xAxis` | The X axis of the Disk Light. Must be normalized and orthogonal to the Y and direction axes. |
| `float` | `xRadius` |  |
| `remixapi_Float3D` | `yAxis` | The Y axis of the Disk Light. Must be normalized and orthogonal to the X and direction axes. |
| `float` | `yRadius` |  |
| `remixapi_Float3D` | `direction` | The direction the Disk Light is pointing in, should match the Shaping direction if present Must be normalized and orthogonal to the X and Y axes. |
| `remixapi_Bool` | `shaping_hasvalue` |  |
| `remixapi_LightInfoLightShaping` | `shaping_value` |  |
| `float` | `volumetricRadianceScale` |  |

### `remixapi_LightInfoCylinderEXT`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `remixapi_Float3D` | `position` |  |
| `float` | `radius` |  |
| `remixapi_Float3D` | `axis` | The "center" axis of the Cylinder Light. Must be normalized. |
| `float` | `axisLength` |  |
| `float` | `volumetricRadianceScale` |  |

### `remixapi_LightInfoDistantEXT`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `remixapi_Float3D` | `direction` | The direction the Distant Light is pointing in. Must be normalized. |
| `float` | `angularDiameterDegrees` |  |
| `float` | `volumetricRadianceScale` |  |

### `remixapi_LightInfoDomeEXT`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `remixapi_Transform` | `transform` |  |
| `remixapi_Path` | `colorTexture` |  |

### `remixapi_LightInfoUSDEXT`

Attachable to remixapi_LightInfo.
If attached, 'remixapi_LightInfo::radiance' is ignored.
Any other attached 'remixapi_LightInfo*EXT' are ignored.
Most fields correspond to a usd token. Set to null, if no value.

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `remixapi_StructType` | `lightType` |  |
| `remixapi_Transform` | `transform` |  |
| `const float*` | `pRadius` | "radius" |
| `const float*` | `pWidth` | "width" |
| `const float*` | `pHeight` | "height" |
| `const float*` | `pLength` | "length" |
| `const float*` | `pAngleRadians` | "angle" |
| `const remixapi_Bool*` | `pEnableColorTemp` | "enableColorTemperature" |
| `const remixapi_Float3D*` | `pColor` | "color" |
| `const float*` | `pColorTemp` | "colorTemperature" |
| `const float*` | `pExposure` | "exposure" |
| `const float*` | `pIntensity` | "intensity" |
| `const float*` | `pConeAngleRadians` | "shaping:cone:angle" |
| `const float*` | `pConeSoftness` | "shaping:cone:softness" |
| `const float*` | `pFocus` | "shaping:focus" |
| `const float*` | `pVolumetricRadianceScale` | "volumetric_radiance_scale" |

### `remixapi_LightInfo`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `uint64_t` | `hash` |  |
| `remixapi_Float3D` | `radiance` |  |
| `remixapi_Bool` | `isDynamic` |  |
| `remixapi_Bool` | `ignoreViewModel` |  |

### `remixapi_PresentInfo`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `remixapi_HWND` | `hwndOverride` | Can be NULL |

### `remixapi_TextureInfo`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `uint64_t` | `hash` | Unique identifier for this texture |
| `uint32_t` | `width` |  |
| `uint32_t` | `height` |  |
| `uint32_t` | `depth` | Set to 1 for 2D textures |
| `uint32_t` | `mipLevels` | Set to 1 for no mipmaps |
| `remixapi_Format` | `format` |  |
| `const void*` | `data` | Pointer to pixel data (all mips sequential) |
| `uint64_t` | `dataSize` | Total size in bytes |

### `remixapi_InitializeLibraryInfo`

| Type | Field | Notes |
| :-- | :-- | :-- |
| `remixapi_StructType` | `sType` |  |
| `void*` | `pNext` |  |
| `uint64_t` | `version` |  |

### `remixapi_VramStats`

Per-category VRAM usage snapshot aggregated across all device-local heaps.
Sizes are in bytes. poolRetainedBytes == totalAllocatedBytes - totalUsedBytes
and represents memory owned by the DXVK allocator's empty-chunk pool that
has not been returned to the driver -- the number that RequestVramCompaction
moves. Category breakdown mirrors DxvkMemoryStats::Category and covers only
RTX-owned suballocations; app-owned buffers/textures (D3D9 app traffic) are
not included.

| Type | Field | Notes |
| :-- | :-- | :-- |
| `uint64_t` | `totalAllocatedBytes` |  |
| `uint64_t` | `totalUsedBytes` |  |
| `uint64_t` | `poolRetainedBytes` |  |
| `uint64_t` | `usedReplacementGeometryBytes` |  |
| `uint64_t` | `usedBufferBytes` |  |
| `uint64_t` | `usedAccelerationStructureBytes` |  |
| `uint64_t` | `usedOpacityMicromapBytes` |  |
| `uint64_t` | `usedMaterialTextureBytes` |  |
| `uint64_t` | `usedRenderTargetBytes` |  |
| `uint64_t` | `driverAllocatedBytes` | Driver-reported allocation on device-local heaps. Sourced from VK_EXT_memory_budget (adapter->getMemoryHeapInfo -> memoryAllocated). This is the Task-Manager / nvidia-smi view of the process's Vulkan footprint. totalAllocatedBytes only counts DXVK's own allocator path, so (driverAllocatedBytes - totalAllocatedBytes) is the non-DXVK overhead: DLSS/NGX internal buffers, raytracing pipeline driver state, bindless descriptor pools, NRC CUDA, pipeline caches, etc. |
| `uint64_t` | `driverBudgetBytes` |  |
| `uint32_t` | `forkTextureCacheCount` | Size of the fork-side RtxTextureManager's SparseUniqueCache. Grows with both D3D9-native and plugin-created textures tracked by Remix. If this climbs while plugin's own texture count is stable, the growth is in fork-side streaming/native textures, not plugin uploads. |

---

## Function-pointer typedefs

### `PFN_remixapi_Startup`

Returns: `remixapi_ErrorCode`

Parameters:

- `const remixapi_StartupInfo* info`

### `PFN_remixapi_Shutdown`

Returns: `remixapi_ErrorCode`

Parameters:

- *(none)*

### `PFN_remixapi_CreateMaterial`

Returns: `remixapi_ErrorCode`

Parameters:

- `const remixapi_MaterialInfo* info`
- `remixapi_MaterialHandle* out_handle`

### `PFN_remixapi_DestroyMaterial`

Returns: `remixapi_ErrorCode`

Parameters:

- `remixapi_MaterialHandle handle`

### `PFN_remixapi_CreateMesh`

Returns: `remixapi_ErrorCode`

Parameters:

- `const remixapi_MeshInfo* info`
- `remixapi_MeshHandle* out_handle`

### `PFN_remixapi_CreateMeshBatched`

Returns: `remixapi_ErrorCode`

Parameters:

- `const remixapi_MeshInfo* info`
- `remixapi_MeshHandle* out_handle`

### `PFN_remixapi_DestroyMesh`

Returns: `remixapi_ErrorCode`

Parameters:

- `remixapi_MeshHandle handle`

### `PFN_remixapi_SetupCamera`

Returns: `remixapi_ErrorCode`

Parameters:

- `const remixapi_CameraInfo* info`

### `PFN_remixapi_SetCameraMediumMaterial`

Returns: `remixapi_ErrorCode`

Parameters:

- `const remixapi_CameraMediumInfo* info`

### `PFN_remixapi_DrawInstance`

Returns: `remixapi_ErrorCode`

Parameters:

- `const remixapi_InstanceInfo* info`

### `PFN_remixapi_CreateLight`

Returns: `remixapi_ErrorCode`

Parameters:

- `const remixapi_LightInfo* info`
- `remixapi_LightHandle* out_handle`

### `PFN_remixapi_CreateLightBatched`

Returns: `remixapi_ErrorCode`

Parameters:

- `const remixapi_LightInfo* info`
- `remixapi_LightHandle* out_handle`

### `PFN_remixapi_DestroyLight`

Returns: `remixapi_ErrorCode`

Parameters:

- `remixapi_LightHandle handle`

### `PFN_remixapi_DrawLightInstance`

Returns: `remixapi_ErrorCode`

Parameters:

- `remixapi_LightHandle lightHandle`

### `PFN_remixapi_SetConfigVariable`

Returns: `remixapi_ErrorCode`

Parameters:

- `const char* key`
- `const char* value`

### `PFN_remixapi_SetGameValue`

Returns: `remixapi_ErrorCode`

Parameters:

- `const char* key`
- `const char* value`

### `PFN_remixapi_GetGameValue`

Returns: `remixapi_ErrorCode`

Parameters:

- `const char* key`
- `char* out_buffer`
- `uint32_t in_buffer_size`
- `uint32_t* out_actual_size`

### `PFN_remixapi_AddTextureHash`

Returns: `remixapi_ErrorCode`

Parameters:

- `const char* textureCategory`
- `const char* textureHash`

### `PFN_remixapi_RemoveTextureHash`

Returns: `remixapi_ErrorCode`

Parameters:

- `const char* textureCategory`
- `const char* textureHash`

### `PFN_remixapi_Present`

Returns: `remixapi_ErrorCode`

Parameters:

- `const remixapi_PresentInfo* info`

### `PFN_remixapi_BridgeCallback`

Returns: `void`

Parameters:

- *(none)*

### `PFN_remixapi_RegisterCallbacks`

Returns: `remixapi_ErrorCode`

Parameters:

- `PFN_remixapi_BridgeCallback beginSceneCallback`
- `PFN_remixapi_BridgeCallback endSceneCallback`
- `PFN_remixapi_BridgeCallback presentCallback`

### `PFN_remixapi_AutoInstancePersistentLights`

Returns: `remixapi_ErrorCode`

Parameters:

- *(none)*

### `PFN_remixapi_UpdateLightDefinition`

Returns: `remixapi_ErrorCode`

Parameters:

- `remixapi_LightHandle handle`
- `const remixapi_LightInfo* info`

### `PFN_remixapi_pick_RequestObjectPickingUserCallback`

Returns: `void`

Parameters:

- `const uint32_t* objectPickingValues_values`
- `uint32_t objectPickingValues_count`
- `void* callbackUserData`

### `PFN_remixapi_pick_RequestObjectPicking`

Returns: `remixapi_ErrorCode`

Parameters:

- `const remixapi_Rect2D* pixelRegion`
- `PFN_remixapi_pick_RequestObjectPickingUserCallback callback`
- `void* callbackUserData`

### `PFN_remixapi_pick_HighlightObjects`

Returns: `remixapi_ErrorCode`

Parameters:

- `const uint32_t* objectPickingValues_values`
- `uint32_t objectPickingValues_count`
- `uint8_t colorR`
- `uint8_t colorG`
- `uint8_t colorB`

### `PFN_remixapi_dxvk_CreateD3D9`

Returns: `remixapi_ErrorCode`

Parameters:

- `remixapi_Bool editorModeEnabled`
- `IDirect3D9Ex** out_pD3D9`

### `PFN_remixapi_dxvk_RegisterD3D9Device`

Returns: `remixapi_ErrorCode`

Parameters:

- `IDirect3DDevice9Ex* d3d9Device`

### `PFN_remixapi_dxvk_GetExternalSwapchain`

Returns: `remixapi_ErrorCode`

Parameters:

- `uint64_t* out_vkImage`
- `uint64_t* out_vkSemaphoreRenderingDone`
- `uint64_t* out_vkSemaphoreResumeSemaphore`

### `PFN_remixapi_dxvk_GetSharedD3D11TextureHandle`

Returns: `remixapi_ErrorCode`

Parameters:

- `void** out_sharedHandle`
- `uint32_t* out_width`
- `uint32_t* out_height`

### `PFN_remixapi_dxvk_GetVkImage`

Returns: `remixapi_ErrorCode`

Parameters:

- `IDirect3DSurface9* source`
- `uint64_t* out_vkImage`

### `PFN_remixapi_dxvk_CopyRenderingOutput`

Returns: `remixapi_ErrorCode`

Parameters:

- `IDirect3DSurface9* destination`
- `remixapi_dxvk_CopyRenderingOutputType type`

### `PFN_remixapi_dxvk_SetDefaultOutput`

Returns: `remixapi_ErrorCode`

Parameters:

- `remixapi_dxvk_CopyRenderingOutputType type`
- `const remixapi_Float4D* color`

### `PFN_remixapi_dxvk_GetTextureHash`

Returns: `remixapi_ErrorCode`

Parameters:

- `IDirect3DTexture9* texture`
- `uint64_t* out_hash`

### `PFN_remixapi_CreateTexture`

Returns: `remixapi_ErrorCode`

Parameters:

- `const remixapi_TextureInfo* info`
- `remixapi_TextureHandle* out_handle`

### `PFN_remixapi_DestroyTexture`

Returns: `remixapi_ErrorCode`

Parameters:

- `remixapi_TextureHandle handle`

### `PFN_remixapi_DrawScreenOverlay`

Returns: `remixapi_ErrorCode`

Parameters:

- `const void* pPixelData`
- `uint32_t width`
- `uint32_t height`
- `remixapi_Format format`
- `float opacity`

### `PFN_remixapi_RequestTextureVramFree`

Returns: `remixapi_ErrorCode`

Parameters:

- *(none)*

### `PFN_remixapi_RequestVramCompaction`

Returns: `remixapi_ErrorCode`

Parameters:

- *(none)*

### `PFN_remixapi_GetVramStats`

Returns: `remixapi_ErrorCode`

Parameters:

- `remixapi_VramStats* out_stats`

---

## `remixapi_Interface` table

The function-pointer table that `remixapi_InitializeLibrary` fills out. Each field points at the matching `PFN_*` typedef documented in the previous section.

### `remixapi_Interface`

NOTE: If adding a new function, append it at the END of the struct.
Reordering or inserting in the middle breaks backwards compatibility.

| Type | Field | Notes |
| :-- | :-- | :-- |
| `PFN_remixapi_Shutdown` | `Shutdown` |  |
| `PFN_remixapi_CreateMaterial` | `CreateMaterial` |  |
| `PFN_remixapi_DestroyMaterial` | `DestroyMaterial` |  |
| `PFN_remixapi_CreateMesh` | `CreateMesh` |  |
| `PFN_remixapi_CreateMeshBatched` | `CreateMeshBatched` |  |
| `PFN_remixapi_DestroyMesh` | `DestroyMesh` |  |
| `PFN_remixapi_SetupCamera` | `SetupCamera` |  |
| `PFN_remixapi_DrawInstance` | `DrawInstance` |  |
| `PFN_remixapi_CreateLight` | `CreateLight` |  |
| `PFN_remixapi_CreateLightBatched` | `CreateLightBatched` |  |
| `PFN_remixapi_DestroyLight` | `DestroyLight` |  |
| `PFN_remixapi_DrawLightInstance` | `DrawLightInstance` |  |
| `PFN_remixapi_SetConfigVariable` | `SetConfigVariable` |  |
| `PFN_remixapi_AddTextureHash` | `AddTextureHash` |  |
| `PFN_remixapi_RemoveTextureHash` | `RemoveTextureHash` |  |
| `PFN_remixapi_CreateTexture` | `CreateTexture` |  |
| `PFN_remixapi_DestroyTexture` | `DestroyTexture` |  |
| `PFN_remixapi_dxvk_CreateD3D9` | `dxvk_CreateD3D9` | DXVK interoperability |
| `PFN_remixapi_dxvk_RegisterD3D9Device` | `dxvk_RegisterD3D9Device` |  |
| `PFN_remixapi_dxvk_GetExternalSwapchain` | `dxvk_GetExternalSwapchain` |  |
| `PFN_remixapi_dxvk_GetVkImage` | `dxvk_GetVkImage` |  |
| `PFN_remixapi_dxvk_CopyRenderingOutput` | `dxvk_CopyRenderingOutput` |  |
| `PFN_remixapi_dxvk_SetDefaultOutput` | `dxvk_SetDefaultOutput` |  |
| `PFN_remixapi_dxvk_GetTextureHash` | `dxvk_GetTextureHash` |  |
| `PFN_remixapi_dxvk_GetSharedD3D11TextureHandle` | `dxvk_GetSharedD3D11TextureHandle` |  |
| `PFN_remixapi_pick_RequestObjectPicking` | `pick_RequestObjectPicking` | Object picking utils |
| `PFN_remixapi_pick_HighlightObjects` | `pick_HighlightObjects` |  |
| `PFN_remixapi_Startup` | `Startup` |  |
| `PFN_remixapi_Present` | `Present` |  |
| `PFN_remixapi_SetCameraMediumMaterial` | `SetCameraMediumMaterial` | Camera-in-medium material. Kept adjacent to Present to mirror the canonical upstream layout (upstream commit 2bac8874 moved it here to fix backwards-compat); fork extension functions are appended below. |
| `PFN_remixapi_RegisterCallbacks` | `RegisterCallbacks` | Optional extension functions (present starting in v0.5.1+) |
| `PFN_remixapi_AutoInstancePersistentLights` | `AutoInstancePersistentLights` |  |
| `PFN_remixapi_UpdateLightDefinition` | `UpdateLightDefinition` |  |
| `PFN_remixapi_DrawScreenOverlay` | `DrawScreenOverlay` |  |
| `PFN_remixapi_SetGameValue` | `SetGameValue` |  |
| `PFN_remixapi_RequestVramCompaction` | `RequestVramCompaction` |  |
| `PFN_remixapi_GetVramStats` | `GetVramStats` |  |
| `PFN_remixapi_RequestTextureVramFree` | `RequestTextureVramFree` |  |
| `PFN_remixapi_GetGameValue` | `GetGameValue` |  |
