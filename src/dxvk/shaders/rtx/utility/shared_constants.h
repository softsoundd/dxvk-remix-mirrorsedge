/*
* Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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
#ifndef SHARED_CONSTANTS_H
#define SHARED_CONSTANTS_H

// contains constants shared between shader and host code

static const uint8_t surfaceMaterialTypeOpaque = uint8_t(0u);
static const uint8_t surfaceMaterialTypeTranslucent = uint8_t(1u);
static const uint8_t surfaceMaterialTypeRayPortal = uint8_t(2u);
static const uint8_t surfaceMaterialTypeMask = uint8_t(0x3u);

#define COMMON_MATERIAL_FLAG_TYPE_MASK surfaceMaterialTypeMask
#define COMMON_MATERIAL_FLAG_TYPE_OFFSET(X) (2 + X)

// NOTE: Each material memory structure contains a set of flags.  The first 2 bits in that flag identify the material type (opaque, etc).
//       We must ensure all other material flags are written to byte addresses after these first two bits.  Use the COMMON_MATERIAL_FLAG_TYPE_OFFSET(x) 
//       macro, and ensure there is enough storage in the flags to represent desired bits accordingly.

// maximum value for thin film thickness in nanometers
#define OPAQUE_SURFACE_MATERIAL_THIN_FILM_MAX_THICKNESS (1500.0f)
// bits for flags field in OpaqueSurfaceMaterial
#define OPAQUE_SURFACE_MATERIAL_FLAG_USE_THIN_FILM_LAYER (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(0))
#define OPAQUE_SURFACE_MATERIAL_FLAG_ALPHA_IS_THIN_FILM_THICKNESS (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(1))
#define OPAQUE_SURFACE_MATERIAL_FLAG_IGNORE_ALPHA_CHANNEL (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(2))
#define OPAQUE_SURFACE_MATERIAL_FLAG_IS_RAYTRACED_RENDER_TARGET (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(3))
#define OPAQUE_SURFACE_MATERIAL_FLAG_HAS_DISPLACEMENT (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(4))
#define OPAQUE_SURFACE_MATERIAL_FLAG_IS_HAIR_CARD (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(5))
// NV-DXVK start: Raytraced camera material emission.
#define OPAQUE_SURFACE_MATERIAL_FLAG_ENABLE_RENDER_TARGET_EMISSION (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(6))
#define OPAQUE_SURFACE_MATERIAL_FLAG_ALBEDO_IS_RENDER_TARGET (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(7))
// NV-DXVK end
// Is set when maskLayerIndex names a MaskLayerMaterial in the surface material extensions.
#define OPAQUE_SURFACE_MATERIAL_FLAG_HAS_MASK_LAYER (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(8))
// Is set when flowLayerIndex names a FlowLayerMaterial in the surface material extensions.
#define OPAQUE_SURFACE_MATERIAL_FLAG_HAS_FLOW_LAYER (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(9))

// FlowLayerMaterial::flags holds the sample count in its low bits and these flags above it.
#define FLOW_LAYER_MATERIAL_SAMPLE_COUNT_MASK 0x3
// Is set when a sample's normal map is octahedral, as the Toolkit ingests them, rather than RGB decoded as 2 * t - 1.
#define FLOW_LAYER_MATERIAL_FLAG_OCTAHEDRAL_0 (1 << 2)
#define FLOW_LAYER_MATERIAL_FLAG_OCTAHEDRAL_1 (1 << 3)
#define FLOW_LAYER_MATERIAL_FLAG_HAS_MASK (1 << 4)
#define FLOW_LAYER_MATERIAL_MASK_CHANNEL_SHIFT 5
#define FLOW_LAYER_MATERIAL_FLAG_MASK_SRGB (1 << 7)
// Is set when the layer blends the surface's normal towards sample 0 rather than adding its samples up. The blend's weight is
// weights.y times the texel channel sample 1's x component names, times the mask.
#define FLOW_LAYER_MATERIAL_FLAG_BLEND (1 << 8)
// Is set when the game samples the blend's weight as sRGB.
#define FLOW_LAYER_MATERIAL_FLAG_BLEND_SRGB (1 << 9)
// Are set when sample 0, sample 1 or the mask reads the surface's second UV set rather than its texture coordinates.
#define FLOW_LAYER_MATERIAL_FLAG_SECONDARY_UV_0 (1 << 10)
#define FLOW_LAYER_MATERIAL_FLAG_SECONDARY_UV_1 (1 << 11)
#define FLOW_LAYER_MATERIAL_FLAG_SECONDARY_UV_MASK (1 << 12)
// FlowLayerMaterial::components holds 8 bits per sample, sample 0 lowest: the texel channels of the normal's x, y and z in
// 2 bits each, this bit when z is reconstructed from x and y,
#define FLOW_LAYER_MATERIAL_COMPONENT_Z_ABSENT (1 << 6)
// and this bit when y is negated, for an RGB normal map with an OpenGL style green channel.
#define FLOW_LAYER_MATERIAL_COMPONENT_NEGATE_Y (1 << 7)

// MaskLayerMaterial::maskChannelAndFlags holds the mask channel in its low bits and these flags above it.
#define MASK_LAYER_MATERIAL_CHANNEL_MASK 0x3
// Is set when the game samples the mask as sRGB.
#define MASK_LAYER_MATERIAL_FLAG_SRGB (1 << 2)
// Is set when the colour is quadratic in the mask, so the surface interpolates through the bake at 1/2.
#define MASK_LAYER_MATERIAL_FLAG_QUADRATIC (1 << 3)


#define OPAQUE_SURFACE_MATERIAL_INTERACTION_FLAG_HAS_HEIGHT_TEXTURE (1 << 0)
#define OPAQUE_SURFACE_MATERIAL_INTERACTION_FLAG_USE_THIN_FILM_LAYER (1 << 1)
// flags overlap with type field when in gbuffer, which occupies last 2 bits.
#define OPAQUE_SURFACE_MATERIAL_INTERACTION_FLAG_MASK 0x3F


// Note: Bits for flags field in TranslucentSurfaceMaterial and TranslucentSurfaceMaterialInteraction
// If set, then the texture bound to transmittanceOrDiffuseTextureIndex is an albedo map for the diffuse layer
#define TRANSLUCENT_SURFACE_MATERIAL_FLAG_USE_DIFFUSE_LAYER (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(0))

#endif // ifndef SHARED_CONSTANTS_H
