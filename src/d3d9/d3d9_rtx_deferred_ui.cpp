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
#include "d3d9_rtx.h"
#include "d3d9_rtx_ue3_helpers.h"

#include "d3d9_include.h"
#include "d3d9_state.h"
#include "d3d9_util.h"
#include "d3d9_buffer.h"
#include "d3d9_device.h"
#include "d3d9_initializer.h"
#include "../util/util_fastops.h"
#include "../util/util_game_patches.h"
#include "../util/util_math.h"
#include "d3d9_rtx_utils.h"
#include "d3d9_texture.h"
#include "../dxso/dxso_color_terms.h"
#include "../dxso/dxso_highlight_tints.h"
#include "../dxso/dxso_material_fades.h"
#include "../dxso/dxso_sampler_inference.h"
#include "../dxso/dxso_ue3_material_identity.h"
#include "../dxso/dxso_uv_dataflow.h"
#include "../dxso/dxso_tables.h"
#include "../dxvk/rtx_render/rtx_bridge_message_channel.h"
#include "../dxvk/rtx_render/rtx_terrain_baker.h"
#include "../dxvk/rtx_render/rtx_ue3_tone_mapping.h"
#include "../dxvk/rtx_render/rtx_gpu_pass_timer.h"
#include "../dxvk/imgui/dxvk_imgui.h"
#include <algorithm>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <shared_mutex>
#include <sstream>
#include <system_error>

namespace dxvk {

  bool D3D9Rtx::isDeferredUiTaggedDraw(XXH64_hash_t* pMatchedTextureHash) const {
    // Pixel shader tag: stable across texture streaming and render target recreation
    if (!m_frameOptions.deferredUiPixelShaders->empty() &&
        m_parent->UseProgrammablePS() && d3d9State().pixelShader != nullptr) {
      const XXH64_hash_t psHash = d3d9State().pixelShader->GetCommonShader()->GetBytecodeHash();
      if (lookupHash(*m_frameOptions.deferredUiPixelShaders, psHash)) {
        return true;
      }
    }

    if (m_frameOptions.deferredUiTextures->empty()) {
      return false;
    }

    const uint32_t usedSamplerMask = m_parent->m_psShaderMasks.samplerMask | m_parent->m_vsShaderMasks.samplerMask;
    const BoundTextureSnapshot& boundTextures = ensureBoundTextureSnapshot();
    const uint32_t usedTextureMask = boundTextures.mask & usedSamplerMask;
    for (const uint32_t idx : bit::BitMask(usedTextureMask)) {
      const BoundTextureSnapshotEntry& entry = boundTextures.entries[idx];
      if (!entry.hasSampleView) {
        continue;
      }

      const auto reportMatch = [&](XXH64_hash_t matchedHash) {
        if (pMatchedTextureHash) {
          *pMatchedTextureHash = matchedHash;
        }
        return true;
      };

      // Non-RT overlays carry no descriptor hash, so they can only match by image hash.
      if (!entry.isRenderTarget) {
        const XXH64_hash_t texHash = entry.imageHash;
        if (texHash != kEmptyHash && lookupHash(*m_frameOptions.deferredUiTextures, texHash)) {
          return reportMatch(texHash);
        }
        continue;
      }

      const XXH64_hash_t matchedHash =
        matchAuthoredRenderTargetTag(*m_frameOptions.deferredUiTextures,
                                     entry.rtDescriptorHash,
                                     entry.rtResolutionAgnosticDescriptorHash);
      if (matchedHash != kEmptyHash) {
        return reportMatch(matchedHash);
      }
    }

    return false;
  }

  bool D3D9Rtx::isSceneLinearRenderTarget() const {
    switch (getCurrentRenderTargetFormat()) {
    case D3D9Format::R16F:
    case D3D9Format::G16R16F:
    case D3D9Format::A16B16G16R16F:
    case D3D9Format::R32F:
    case D3D9Format::G32R32F:
    case D3D9Format::A32B32G32R32F:
      return true;
    default:
      return false;
    }
  }

  // Snapshots a deferred UI draw for replay after injection. Vertex and index ranges are copied, as the
  // game may re-lock its buffers, and multi-stream draws are interleaved into stream 0.
  bool D3D9Rtx::captureDeferredUiDraw(const IndexContext& indexContext,
                                      const VertexContext vertexContext[caps::MaxStreams],
                                      const DrawContext& drawContext) {
    if (m_deferredUiDraws.size() >= kMaxDeferredUiDrawsPerFrame) {
      ONCE(Logger::warn("[RTX-DeferredUI] Too many deferred UI overlay draws in one frame; suppressing the rest. Check the rtx.deferredUiTextures tagging."));
      return false;
    }

    // Only the programmable pipeline is supported (UE3 MaterialEffect overlays are always
    // shader draws); fixed-function overlays would additionally need transform and texture
    // stage state capture.
    if (!m_parent->UseProgrammableVS() || !m_parent->UseProgrammablePS() ||
        d3d9State().vertexShader == nullptr || d3d9State().pixelShader == nullptr) {
      ONCE(Logger::warn("[RTX-DeferredUI] Fixed-function draw tagged as deferred UI overlay is not supported for replay; suppressing."));
      return false;
    }

    if (d3d9State().vertexDecl == nullptr) {
      return false;
    }

    // Instanced draws cannot be replayed through the UP path
    if ((d3d9State().streamFreq[0] & 0x7FFFFFu) > 1) {
      ONCE(Logger::warn("[RTX-DeferredUI] Instanced draw tagged as deferred UI overlay is not supported for replay; suppressing."));
      return false;
    }

    uint32_t usedStreamMask = 0;
    for (const auto& element : d3d9State().vertexDecl->GetElements()) {
      if (element.Stream == 0xFF) {
        continue; // D3DDECL_END
      }
      if (element.Stream >= caps::MaxStreams) {
        return false;
      }
      usedStreamMask |= 1u << element.Stream;
    }

    if (usedStreamMask == 0) {
      return false;
    }

    // Per-stream layout of the interleaved stream-0 vertex record used for replay
    uint32_t streamBase[caps::MaxStreams] = {};
    uint32_t combinedStride = 0;
    for (const uint32_t s : bit::BitMask(usedStreamMask)) {
      const VertexContext& v = vertexContext[s];
      if (v.stride == 0 || v.mappedSlice.mapPtr == nullptr) {
        return false;
      }
      if ((d3d9State().streamFreq[s] & D3DSTREAMSOURCE_INSTANCEDATA) != 0) {
        ONCE(Logger::warn("[RTX-DeferredUI] Instance-data stream on a draw tagged as deferred UI overlay is not supported for replay; suppressing."));
        return false;
      }
      streamBase[s] = combinedStride;
      combinedStride += v.stride;
    }

    if (combinedStride == 0 || combinedStride > 0xFFFF) {
      return false;
    }

    DeferredUiDraw draw;
    draw.primitiveType = drawContext.PrimitiveType;
    draw.primitiveCount = drawContext.PrimitiveCount;
    draw.indexed = drawContext.Indexed;
    draw.vertexStride = combinedStride;

    int64_t firstVertex = 0;
    uint32_t vertexCount = 0;

    if (draw.indexed) {
      const uint32_t indexCount = GetVertexCount(drawContext.PrimitiveType, drawContext.PrimitiveCount);
      if (indexCount == 0 || indexCount > kMaxDeferredUiIndices) {
        ONCE(Logger::warn("[RTX-DeferredUI] Draw tagged as deferred UI overlay has too many indices for replay; suppressing."));
        return false;
      }

      if (indexContext.indexType == VK_INDEX_TYPE_NONE_KHR || indexContext.indexBuffer.mapPtr == nullptr) {
        return false;
      }

      const bool is16Bit = indexContext.indexType == VK_INDEX_TYPE_UINT16;
      const uint32_t indexStride = is16Bit ? 2 : 4;
      const size_t indexByteOffset = size_t(indexStride) * drawContext.StartIndex;
      if (indexByteOffset + size_t(indexStride) * indexCount > indexContext.indexBuffer.length) {
        return false;
      }

      const uint8_t* pIndexBase = static_cast<const uint8_t*>(indexContext.indexBuffer.mapPtr) + indexByteOffset;

      uint32_t minIndex = std::numeric_limits<uint32_t>::max();
      uint32_t maxIndex = 0;

      // Scan the used index range, then rebase the copied indices onto the copied vertex
      // window (widened to 32-bit for the replay draw)
      draw.indexData.resize(indexCount);
      const auto scanAndRebase = [&](const auto* pSrc) {
        for (uint32_t i = 0; i < indexCount; i++) {
          minIndex = std::min<uint32_t>(minIndex, pSrc[i]);
          maxIndex = std::max<uint32_t>(maxIndex, pSrc[i]);
        }
        for (uint32_t i = 0; i < indexCount; i++) {
          draw.indexData[i] = uint32_t(pSrc[i]) - minIndex;
        }
      };
      if (is16Bit) {
        scanAndRebase(reinterpret_cast<const uint16_t*>(pIndexBase));
      } else {
        scanAndRebase(reinterpret_cast<const uint32_t*>(pIndexBase));
      }

      vertexCount = maxIndex - minIndex + 1;
      firstVertex = int64_t(drawContext.BaseVertexIndex) + minIndex;
    } else {
      vertexCount = GetVertexCount(drawContext.PrimitiveType, drawContext.PrimitiveCount);
      firstVertex = drawContext.BaseVertexIndex; // StartVertex for DrawPrimitive, 0 for the UP path
    }

    if (vertexCount == 0 || firstVertex < 0) {
      return false;
    }

    const size_t vertexBytes = size_t(vertexCount) * combinedStride;
    if (vertexBytes > kMaxDeferredUiVertexBytes ||
        m_deferredUiFrameVertexBytes + vertexBytes > kMaxDeferredUiFrameVertexBytes) {
      ONCE(Logger::warn("[RTX-DeferredUI] Draw tagged as deferred UI overlay exceeds the vertex data replay budget; suppressing. Check the rtx.deferredUiTextures tagging."));
      return false;
    }

    // Validate source ranges for every referenced stream before copying anything
    for (const uint32_t s : bit::BitMask(usedStreamMask)) {
      const VertexContext& v = vertexContext[s];
      const size_t streamByteOffset = size_t(v.offset) + size_t(firstVertex) * v.stride;
      if (streamByteOffset + size_t(vertexCount) * v.stride > v.mappedSlice.length) {
        return false;
      }
    }

    draw.vertexCount = vertexCount;
    draw.vertexData.resize(vertexBytes);
    for (const uint32_t s : bit::BitMask(usedStreamMask)) {
      const VertexContext& v = vertexContext[s];
      const uint8_t* pSrc = static_cast<const uint8_t*>(v.mappedSlice.mapPtr) + v.offset + size_t(firstVertex) * v.stride;
      uint8_t* pDst = draw.vertexData.data() + streamBase[s];
      for (uint32_t i = 0; i < vertexCount; i++) {
        std::memcpy(pDst + size_t(i) * combinedStride, pSrc + size_t(i) * v.stride, v.stride);
      }
    }

    // Vertex declaration for the replay: the original when everything already lives on
    // stream 0, otherwise an internally created remap onto the interleaved stream-0 layout
    if (usedStreamMask == 1u) {
      draw.replayDecl = d3d9State().vertexDecl.ptr();
    } else {
      std::vector<D3DVERTEXELEMENT9> remappedElements;
      remappedElements.reserve(d3d9State().vertexDecl->GetElements().size() + 1);
      for (const auto& element : d3d9State().vertexDecl->GetElements()) {
        if (element.Stream == 0xFF) {
          continue;
        }
        D3DVERTEXELEMENT9 remapped = element;
        remapped.Stream = 0;
        remapped.Offset = WORD(streamBase[element.Stream] + element.Offset);
        remappedElements.push_back(remapped);
      }
      remappedElements.push_back(D3DDECL_END());

      Com<IDirect3DVertexDeclaration9> remappedDecl;
      if (FAILED(m_parent->CreateVertexDeclaration(remappedElements.data(), &remappedDecl)) || remappedDecl == nullptr) {
        ONCE(Logger::warn("[RTX-DeferredUI] Failed to create the remapped vertex declaration for a deferred UI overlay draw; suppressing."));
        return false;
      }
      draw.replayDecl = remappedDecl;
    }

    m_deferredUiFrameVertexBytes += uint32_t(vertexBytes);

    draw.vertexShader = d3d9State().vertexShader;
    draw.pixelShader = d3d9State().pixelShader;

    // Constants: only the ranges the shaders actually declare
    const auto& vsMeta = d3d9State().vertexShader->GetCommonShader()->GetMeta();
    const auto& psMeta = d3d9State().pixelShader->GetCommonShader()->GetMeta();

    const uint32_t vsFloatCount = std::min<uint32_t>(vsMeta.maxConstIndexF, caps::MaxFloatConstantsVS);
    const uint32_t psFloatCount = std::min<uint32_t>(psMeta.maxConstIndexF, caps::MaxFloatConstantsPS);
    const uint32_t vsIntCount = std::min<uint32_t>(vsMeta.maxConstIndexI, caps::MaxOtherConstants);
    const uint32_t psIntCount = std::min<uint32_t>(psMeta.maxConstIndexI, caps::MaxOtherConstants);
    const uint32_t vsBoolDwords = (std::min<uint32_t>(vsMeta.maxConstIndexB, caps::MaxOtherConstants) + 31u) / 32u;
    const uint32_t psBoolDwords = (std::min<uint32_t>(psMeta.maxConstIndexB, caps::MaxOtherConstants) + 31u) / 32u;

    draw.vsFloatConsts.assign(d3d9State().vsConsts.fConsts, d3d9State().vsConsts.fConsts + vsFloatCount);
    draw.psFloatConsts.assign(d3d9State().psConsts.fConsts, d3d9State().psConsts.fConsts + psFloatCount);
    draw.vsIntConsts.assign(d3d9State().vsConsts.iConsts, d3d9State().vsConsts.iConsts + vsIntCount);
    draw.psIntConsts.assign(d3d9State().psConsts.iConsts, d3d9State().psConsts.iConsts + psIntCount);
    draw.vsBoolConsts.assign(d3d9State().vsConsts.bConsts, d3d9State().vsConsts.bConsts + vsBoolDwords);
    draw.psBoolConsts.assign(d3d9State().psConsts.bConsts, d3d9State().psConsts.bConsts + psBoolDwords);

    // Texture bindings for every sampler the shaders use (including used-but-unbound slots so
    // the replay never samples whatever the app happens to have bound at replay time)
    const uint32_t usedSamplerMask = m_parent->m_psShaderMasks.samplerMask | m_parent->m_vsShaderMasks.samplerMask;
    for (const uint32_t idx : bit::BitMask(usedSamplerMask)) {
      if (idx >= SamplerCount) {
        continue;
      }

      DeferredUiDraw::TextureBinding binding;
      binding.slot = idx;
      binding.texture = d3d9State().textures[idx];
      binding.samplerStates = d3d9State().samplerStates[idx];

      // Track sampled render targets (scene color candidates for the refresh blit)
      if (d3d9State().textures[idx] != nullptr && (m_parent->GetActiveRTTextures() & (1u << idx)) != 0) {
        if (D3D9CommonTexture* texInfo = GetCommonTexture(d3d9State().textures[idx])) {
          binding.renderTargetImage = texInfo->GetImage();
        }
      }

      draw.textures.push_back(std::move(binding));
    }

    for (size_t i = 0; i < kDeferredUiRenderStates.size(); i++) {
      draw.renderStates[i] = d3d9State().renderStates[kDeferredUiRenderStates[i]];
    }

    draw.viewport = d3d9State().viewport;
    draw.scissorRect = d3d9State().scissorRect;

    if (d3d9State().renderTargets[kRenderTargetIndex] != nullptr) {
      const auto rtExtent = d3d9State().renderTargets[kRenderTargetIndex]->GetSurfaceExtent();
      draw.sourceRenderTargetWidth = rtExtent.width;
      draw.sourceRenderTargetHeight = rtExtent.height;
    }
    draw.sceneLinear = isSceneLinearRenderTarget();

    m_deferredUiDraws.push_back(std::move(draw));

    ONCE(Logger::info("[RTX-DeferredUI] Captured a deferred UI overlay draw for post-injection replay."));
    return true;
  }

  // Replays captured deferred UI overlay draws on top of the ray-traced image. Called right
  // after RTX injection is queued (at the first UI draw, or at EndFrame) so the overlays land
  // between the ray-traced image and the game's UI rasterization.
  void D3D9Rtx::replayDeferredUiDraws(std::vector<DeferredUiDraw> draws,
                                      IDirect3DSurface9* pOverrideRenderTarget,
                                      const Rc<DxvkImage>& sceneSourceImage) {
    if (draws.empty()) {
      return;
    }

    if (!m_frameOptions.deferredUiReplay) {
      return;
    }

    if (m_parent->ShouldRecord()) {
      // Mid state-block recording: internal Set* calls would be recorded instead of applied
      ONCE(Logger::warn("[RTX-DeferredUI] Skipping deferred UI overlay replay while a state block is being recorded."));
      return;
    }

    ScopedCpuProfileZone();

    // Copies sceneSourceImage into the scene colour textures an overlay samples before every replayed
    // draw, so each overlay reads the ray-traced image with the earlier overlays applied.
    auto refreshSampledSceneTargets = [&](const DeferredUiDraw& draw) {
      if (!m_frameOptions.deferredUiRefreshSceneColor || sceneSourceImage == nullptr) {
        return;
      }

      for (const auto& binding : draw.textures) {
        const Rc<DxvkImage>& sceneImage = binding.renderTargetImage;
        if (sceneImage == nullptr || sceneImage == sceneSourceImage) {
          continue;
        }

        // Only refresh plausible scene-color targets (aspect ratio matching the final
        // image); small utility render targets keep their game-rendered content.
        const VkExtent3D dstExtent = sceneImage->info().extent;
        const VkExtent3D srcExtent = sceneSourceImage->info().extent;
        const double a = double(dstExtent.width) * double(srcExtent.height);
        const double b = double(dstExtent.height) * double(srcExtent.width);
        const double denom = std::max(a, b);
        if (denom <= 0.0 || (std::abs(a - b) / denom) >= 0.01) {
          continue;
        }

        m_parent->EmitCs([cSrcImage = sceneSourceImage, cDstImage = sceneImage](DxvkContext* ctx) {
          RtxContext::blitImageHelper(ctx, cSrcImage, cDstImage, VkFilter::VK_FILTER_NEAREST);
        });
      }
    };

    // ---- save every piece of application state the replay overrides ----

    uint32_t maxVsFloat = 0, maxPsFloat = 0, maxVsInt = 0, maxPsInt = 0, maxVsBool = 0, maxPsBool = 0;
    uint32_t touchedTextureSlots = 0;
    for (const auto& draw : draws) {
      maxVsFloat = std::max<uint32_t>(maxVsFloat, uint32_t(draw.vsFloatConsts.size()));
      maxPsFloat = std::max<uint32_t>(maxPsFloat, uint32_t(draw.psFloatConsts.size()));
      maxVsInt = std::max<uint32_t>(maxVsInt, uint32_t(draw.vsIntConsts.size()));
      maxPsInt = std::max<uint32_t>(maxPsInt, uint32_t(draw.psIntConsts.size()));
      maxVsBool = std::max<uint32_t>(maxVsBool, uint32_t(draw.vsBoolConsts.size()));
      maxPsBool = std::max<uint32_t>(maxPsBool, uint32_t(draw.psBoolConsts.size()));
      for (const auto& binding : draw.textures) {
        touchedTextureSlots |= 1u << binding.slot;
      }
    }

    Com<IDirect3DVertexDeclaration9> savedDecl(d3d9State().vertexDecl.ptr());
    Com<IDirect3DVertexShader9> savedVertexShader(d3d9State().vertexShader.ptr());
    Com<IDirect3DPixelShader9> savedPixelShader(d3d9State().pixelShader.ptr());
    Com<IDirect3DSurface9> savedRenderTarget(pOverrideRenderTarget != nullptr ? d3d9State().renderTargets[kRenderTargetIndex].ptr() : nullptr);
    Com<IDirect3DSurface9> savedDepthStencil(d3d9State().depthStencil.ptr());
    Com<IDirect3DVertexBuffer9> savedStream0(d3d9State().vertexBuffers[0].vertexBuffer.ptr());
    const UINT savedStream0Offset = d3d9State().vertexBuffers[0].offset;
    const UINT savedStream0Stride = d3d9State().vertexBuffers[0].stride;
    Com<IDirect3DIndexBuffer9> savedIndices(d3d9State().indices.ptr());
    const UINT savedStream0Freq = d3d9State().streamFreq[0];

    std::vector<Vector4> savedVsFloat(d3d9State().vsConsts.fConsts, d3d9State().vsConsts.fConsts + maxVsFloat);
    std::vector<Vector4> savedPsFloat(d3d9State().psConsts.fConsts, d3d9State().psConsts.fConsts + maxPsFloat);
    std::vector<Vector4i> savedVsInt(d3d9State().vsConsts.iConsts, d3d9State().vsConsts.iConsts + maxVsInt);
    std::vector<Vector4i> savedPsInt(d3d9State().psConsts.iConsts, d3d9State().psConsts.iConsts + maxPsInt);
    std::vector<uint32_t> savedVsBool(d3d9State().vsConsts.bConsts, d3d9State().vsConsts.bConsts + maxVsBool);
    std::vector<uint32_t> savedPsBool(d3d9State().psConsts.bConsts, d3d9State().psConsts.bConsts + maxPsBool);

    struct SavedTextureSlot {
      uint32_t slot;
      Com<IDirect3DBaseTexture9> texture;
      std::array<DWORD, SamplerStateCount> samplerStates;
    };
    std::vector<SavedTextureSlot> savedTextureSlots;
    for (const uint32_t idx : bit::BitMask(touchedTextureSlots)) {
      SavedTextureSlot saved;
      saved.slot = idx;
      saved.texture = d3d9State().textures[idx];
      saved.samplerStates = d3d9State().samplerStates[idx];
      savedTextureSlots.push_back(std::move(saved));
    }

    std::array<DWORD, kDeferredUiRenderStates.size()> savedRenderStates;
    for (size_t i = 0; i < kDeferredUiRenderStates.size(); i++) {
      savedRenderStates[i] = d3d9State().renderStates[kDeferredUiRenderStates[i]];
    }

    const D3DVIEWPORT9 savedViewport = d3d9State().viewport;
    const RECT savedScissor = d3d9State().scissorRect;

    // ---- replay ----

    m_replayingDeferredUiDraws = true;

    if (pOverrideRenderTarget != nullptr) {
      m_parent->SetRenderTarget(0, pOverrideRenderTarget);
    }

    // Overlays composite over the final image: depth/stencil contents at this point in the
    // frame are meaningless, and an incompatible depth surface must not clip the render area.
    m_parent->SetDepthStencilSurface(nullptr);
    m_parent->SetStreamSourceFreq(0, 1);

    uint32_t replayTargetWidth = 0, replayTargetHeight = 0;
    if (d3d9State().renderTargets[kRenderTargetIndex] != nullptr) {
      const auto rtExtent = d3d9State().renderTargets[kRenderTargetIndex]->GetSurfaceExtent();
      replayTargetWidth = rtExtent.width;
      replayTargetHeight = rtExtent.height;
    }

    for (const auto& draw : draws) {
      // Sync the overlay's scene inputs with the target's current content (ray-traced blit
      // plus any previously replayed overlays) before it draws
      refreshSampledSceneTargets(draw);

      m_parent->SetVertexDeclaration(draw.replayDecl.ptr());
      m_parent->SetVertexShader(draw.vertexShader.ptr());
      m_parent->SetPixelShader(draw.pixelShader.ptr());

      if (!draw.vsFloatConsts.empty()) {
        m_parent->SetVertexShaderConstantF(0, reinterpret_cast<const float*>(draw.vsFloatConsts.data()), UINT(draw.vsFloatConsts.size()));
      }
      if (!draw.psFloatConsts.empty()) {
        m_parent->SetPixelShaderConstantF(0, reinterpret_cast<const float*>(draw.psFloatConsts.data()), UINT(draw.psFloatConsts.size()));
      }
      if (!draw.vsIntConsts.empty()) {
        m_parent->SetVertexShaderConstantI(0, reinterpret_cast<const int*>(draw.vsIntConsts.data()), UINT(draw.vsIntConsts.size()));
      }
      if (!draw.psIntConsts.empty()) {
        m_parent->SetPixelShaderConstantI(0, reinterpret_cast<const int*>(draw.psIntConsts.data()), UINT(draw.psIntConsts.size()));
      }
      for (uint32_t i = 0; i < draw.vsBoolConsts.size(); i++) {
        m_parent->SetVertexBoolBitfield(i, ~0u, draw.vsBoolConsts[i]);
      }
      for (uint32_t i = 0; i < draw.psBoolConsts.size(); i++) {
        m_parent->SetPixelBoolBitfield(i, ~0u, draw.psBoolConsts[i]);
      }

      for (const auto& binding : draw.textures) {
        m_parent->SetStateTexture(binding.slot, binding.texture.ptr());
        for (uint32_t type = D3DSAMP_ADDRESSU; type < SamplerStateCount; type++) {
          m_parent->SetStateSamplerState(binding.slot, D3DSAMPLERSTATETYPE(type), binding.samplerStates[type]);
        }
      }

      for (size_t i = 0; i < kDeferredUiRenderStates.size(); i++) {
        m_parent->SetRenderState(kDeferredUiRenderStates[i], draw.renderStates[i]);
      }
      m_parent->SetRenderState(D3DRS_ZENABLE, D3DZB_FALSE);
      m_parent->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
      m_parent->SetRenderState(D3DRS_STENCILENABLE, FALSE);

      // Rescale the captured viewport/scissor when the capture-time render target and the
      // replay target differ in size (e.g. overlays captured on a scaled scene target)
      D3DVIEWPORT9 viewport = draw.viewport;
      RECT scissor = draw.scissorRect;
      if (replayTargetWidth != 0 && replayTargetHeight != 0 &&
          draw.sourceRenderTargetWidth != 0 && draw.sourceRenderTargetHeight != 0 &&
          (draw.sourceRenderTargetWidth != replayTargetWidth || draw.sourceRenderTargetHeight != replayTargetHeight)) {
        const double scaleX = double(replayTargetWidth) / double(draw.sourceRenderTargetWidth);
        const double scaleY = double(replayTargetHeight) / double(draw.sourceRenderTargetHeight);
        viewport.X = DWORD(viewport.X * scaleX);
        viewport.Y = DWORD(viewport.Y * scaleY);
        viewport.Width = std::max<DWORD>(1, DWORD(viewport.Width * scaleX));
        viewport.Height = std::max<DWORD>(1, DWORD(viewport.Height * scaleY));
        scissor.left = LONG(scissor.left * scaleX);
        scissor.right = LONG(scissor.right * scaleX);
        scissor.top = LONG(scissor.top * scaleY);
        scissor.bottom = LONG(scissor.bottom * scaleY);
      }
      m_parent->SetViewport(&viewport);
      m_parent->SetScissorRect(&scissor);

      if (draw.indexed) {
        m_parent->DrawIndexedPrimitiveUP(draw.primitiveType, 0, draw.vertexCount, draw.primitiveCount,
                                         draw.indexData.data(), D3DFMT_INDEX32,
                                         draw.vertexData.data(), draw.vertexStride);
      } else {
        m_parent->DrawPrimitiveUP(draw.primitiveType, draw.primitiveCount,
                                  draw.vertexData.data(), draw.vertexStride);
      }
    }

    // ---- restore the application state ----

    if (pOverrideRenderTarget != nullptr && savedRenderTarget != nullptr) {
      m_parent->SetRenderTarget(0, savedRenderTarget.ptr());
    }
    m_parent->SetDepthStencilSurface(savedDepthStencil.ptr());

    m_parent->SetVertexDeclaration(savedDecl.ptr());
    m_parent->SetVertexShader(savedVertexShader.ptr());
    m_parent->SetPixelShader(savedPixelShader.ptr());

    if (maxVsFloat != 0) {
      m_parent->SetVertexShaderConstantF(0, reinterpret_cast<const float*>(savedVsFloat.data()), maxVsFloat);
    }
    if (maxPsFloat != 0) {
      m_parent->SetPixelShaderConstantF(0, reinterpret_cast<const float*>(savedPsFloat.data()), maxPsFloat);
    }
    if (maxVsInt != 0) {
      m_parent->SetVertexShaderConstantI(0, reinterpret_cast<const int*>(savedVsInt.data()), maxVsInt);
    }
    if (maxPsInt != 0) {
      m_parent->SetPixelShaderConstantI(0, reinterpret_cast<const int*>(savedPsInt.data()), maxPsInt);
    }
    for (uint32_t i = 0; i < maxVsBool; i++) {
      m_parent->SetVertexBoolBitfield(i, ~0u, savedVsBool[i]);
    }
    for (uint32_t i = 0; i < maxPsBool; i++) {
      m_parent->SetPixelBoolBitfield(i, ~0u, savedPsBool[i]);
    }

    for (const auto& saved : savedTextureSlots) {
      m_parent->SetStateTexture(saved.slot, saved.texture.ptr());
      for (uint32_t type = D3DSAMP_ADDRESSU; type < SamplerStateCount; type++) {
        m_parent->SetStateSamplerState(saved.slot, D3DSAMPLERSTATETYPE(type), saved.samplerStates[type]);
      }
    }

    for (size_t i = 0; i < kDeferredUiRenderStates.size(); i++) {
      m_parent->SetRenderState(kDeferredUiRenderStates[i], savedRenderStates[i]);
    }

    m_parent->SetViewport(&savedViewport);
    m_parent->SetScissorRect(&savedScissor);

    m_parent->SetStreamSource(0, savedStream0.ptr(), savedStream0Offset, savedStream0Stride);
    m_parent->SetIndices(savedIndices.ptr());
    m_parent->SetStreamSourceFreq(0, savedStream0Freq);

    m_replayingDeferredUiDraws = false;

    ONCE(Logger::info(str::format("[RTX-DeferredUI] Replayed ", draws.size(), " deferred UI overlay draw(s) after RTX injection.")));
  }

  void D3D9Rtx::injectRtxWithOverlays(const Rc<DxvkImage>& targetImage, IDirect3DSurface9* pDisplayOverlayTarget) {
    const bool hasSceneLinearDraws = std::any_of(m_deferredUiDraws.begin(), m_deferredUiDraws.end(),
                                                 [](const DeferredUiDraw& draw) { return draw.sceneLinear; });
    const bool staged = hasSceneLinearDraws && m_frameOptions.deferredUiReplay && m_frameOptions.deferredUiHdrReplay &&
                        targetImage != nullptr && !m_parent->ShouldRecord() &&
                        ensureDeferredUiHdrCanvas(targetImage->info().extent);

    // Take the draws up front: every path must drop them rather than leave them queued for a
    // later, incorrectly ordered replay point
    std::vector<DeferredUiDraw> draws = std::move(m_deferredUiDraws);
    m_deferredUiDraws.clear();

    if (!staged) {
      triggerInjectRTX(targetImage);
      replayDeferredUiDraws(std::move(draws), pDisplayOverlayTarget, targetImage);
      return;
    }

    // See UE3Compatibility.md, "Deferred overlays"
    std::vector<DeferredUiDraw> sceneLinearDraws;
    std::vector<DeferredUiDraw> displayDraws;
    for (DeferredUiDraw& draw : draws) {
      if (draw.sceneLinear) {
        sceneLinearDraws.push_back(std::move(draw));
      } else {
        displayDraws.push_back(std::move(draw));
      }
    }

    ONCE(Logger::info(str::format("[RTX-DeferredUI] Staging RTX injection: ", sceneLinearDraws.size(),
                                  " scene-linear overlay draw(s) replay on the HDR image before tone mapping.")));

    const Rc<DxvkImage> canvasImage = m_deferredUiHdrCanvas->GetCommonTexture()->GetImage();

    triggerInjectRTX(targetImage, canvasImage);
    replayDeferredUiDraws(std::move(sceneLinearDraws), m_deferredUiHdrCanvas.ptr(), canvasImage);
    m_parent->EmitCs([](DxvkContext* ctx) {
      static_cast<RtxContext*>(ctx)->finishInjectRTX();
    });
    replayDeferredUiDraws(std::move(displayDraws), pDisplayOverlayTarget, targetImage);
  }

  bool D3D9Rtx::ensureDeferredUiHdrCanvas(const VkExtent3D& extent) {
    if (m_deferredUiHdrCanvas != nullptr) {
      const VkExtent2D canvasExtent = m_deferredUiHdrCanvas->GetSurfaceExtent();
      if (canvasExtent.width == extent.width && canvasExtent.height == extent.height) {
        return true;
      }
      m_deferredUiHdrCanvas = nullptr;
    }

    D3D9_COMMON_TEXTURE_DESC desc;
    desc.Width              = extent.width;
    desc.Height             = extent.height;
    desc.Depth              = 1;
    desc.ArraySize          = 1;
    desc.MipLevels          = 1;
    desc.Usage              = D3DUSAGE_RENDERTARGET;
    desc.Format             = EnumerateFormat(D3DFMT_A16B16G16R16F);
    desc.Pool               = D3DPOOL_DEFAULT;
    desc.Discard            = FALSE;
    desc.MultiSample        = D3DMULTISAMPLE_NONE;
    desc.MultisampleQuality = 0;
    desc.IsBackBuffer       = FALSE;
    desc.IsAttachmentOnly   = TRUE;

    if (SUCCEEDED(D3D9CommonTexture::NormalizeTextureProperties(m_parent, &desc))) {
      try {
        m_deferredUiHdrCanvas = new D3D9Surface(m_parent, &desc, nullptr, nullptr);
        m_parent->m_initializer->InitTexture(m_deferredUiHdrCanvas->GetCommonTexture());
      } catch (const DxvkError& e) {
        Logger::err(e.message());
        m_deferredUiHdrCanvas = nullptr;
      }
    }

    if (m_deferredUiHdrCanvas == nullptr) {
      ONCE(Logger::warn("[RTX-DeferredUI] Could not create the HDR overlay canvas; scene-linear overlays replay on the tone-mapped output."));
      return false;
    }
    return true;
  }

  bool D3D9Rtx::DeferredUiTagQuery::isTagged() {
    if (m_state < 0) {
      m_state = m_rtx.isDeferredUiTaggedDraw(&m_matchedTextureHash) ? 1 : 0;
    }
    return m_state == 1;
  }

  // The emptiness test keeps post-process draws from building a bound-texture snapshot just to
  // discover that no pixel shader tag exists.
  bool D3D9Rtx::DeferredUiTagQuery::isPixelShaderTagged() {
    return !m_rtx.m_frameOptions.deferredUiPixelShaders->empty() &&
           isTagged() &&
           m_matchedTextureHash == kEmptyHash;
  }

  // See "Deferred overlays" in UE3Compatibility.md. Runs after the pass switch, so only a pixel shader tag can
  // defer a composite or video pass, and never defers world geometry or depth-writing draws, which
  // can share textures with tagged overlays.
  std::optional<D3D9Rtx::DrawCallType> D3D9Rtx::decideDeferredUiDraw(const DrawContext& drawContext,
                                                                     DeferredUiTagQuery& deferredUiTag) {
    if (!m_frameOptions.deferredUiTextures->empty() || !m_frameOptions.deferredUiPixelShaders->empty()) {
      const XXH64_hash_t& matchedTextureHash = deferredUiTag.matchedTextureHash();

      if (deferredUiTag.isTagged()) {
        const bool matchedByPixelShaderTag = deferredUiTag.isPixelShaderTagged();
        const bool zWriteEnabled = d3d9State().renderStates[D3DRS_ZWRITEENABLE];
        const bool isWorldGeometryVertexFactory = isUe3WorldGeometryVertexFactory(m_currentUe3VertexFactory);

        // Engine post-process shaders sample the scene target, but replaying them would brighten or paint over
        // the ray-traced image, so a texture tag never defers them; a pixel shader tag still does.
        bool isEnginePostProcessShader = false;
        if (!matchedByPixelShaderTag && m_parent->UseProgrammablePS() && d3d9State().pixelShader != nullptr) {
          const Ue3ShaderFeatureInfo psInfo = getUe3ShaderFeatureInfo(d3d9State().pixelShader->GetCommonShader());
          isEnginePostProcessShader = psInfo.hasGammaConstants ||
                                      psInfo.hasToneMapConstants ||
                                      psInfo.hasExposureOrToneSampler ||
                                      psInfo.hasMotionBlurConstants ||
                                      psInfo.hasVelocitySampler ||
                                      psInfo.hasDistortionSampler ||
                                      psInfo.hasFogConstants ||
                                      psInfo.hasHazeConstants ||
                                      psInfo.looksLikeDofAndBloomPostProcess();
        }

        // Overlay tiles have a Local-style declaration but never depth test, unlike even small world quads.
        const bool depthTestDisabled = d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_FALSE ||
                                       d3d9State().renderStates[D3DRS_ZFUNC] == D3DCMP_ALWAYS;
        const bool looksLikeOverlayTile = drawContext.PrimitiveCount <= 4 && depthTestDisabled && !zWriteEnabled;

        const bool eligible = !zWriteEnabled && !isEnginePostProcessShader &&
                              (!isWorldGeometryVertexFactory || looksLikeOverlayTile);
        const char* refusalReason = isEnginePostProcessShader
                                    ? "engine post-process shader"
                                    : "world geometry or depth write";

        // One-shot diagnostics per (pixel shader, decision): prints the stable pixel shader
        // hash so tags on unstable render-target textures can be moved to
        // rtx.d3d9.deferredUiPixelShaders.
        const XXH64_hash_t psHash = (m_parent->UseProgrammablePS() && d3d9State().pixelShader != nullptr)
                                    ? d3d9State().pixelShader->GetCommonShader()->GetBytecodeHash() : 0;
        const XXH64_hash_t vsHash = (m_parent->UseProgrammableVS() && d3d9State().vertexShader != nullptr)
                                    ? d3d9State().vertexShader->GetCommonShader()->GetBytecodeHash() : 0;
        const XXH64_hash_t logKey = psHash ^ (eligible ? 0xD1B54A32D192ED03ull
                                                       : (isEnginePostProcessShader ? 0x2545F4914F6CDD1Dull
                                                                                    : 0x9E3779B97F4A7C15ull));
        if (m_deferredUiLoggedDecisions.insert(logKey).second) {
          const std::string matchedDescription = matchedTextureHash != 0
            ? str::format(" matchedTexture=0x", std::hex, matchedTextureHash, std::dec)
            : std::string(" matchedBy=pixelShaderTag");

          Logger::info(str::format(
            "[RTX-DeferredUI] ",
            eligible ? std::string("Deferring overlay draw")
                     : str::format("Tagged draw NOT deferred (", refusalReason, ")"),
            ": ps=0x", std::hex, psHash,
            " vs=0x", vsHash, std::dec,
            " vertexFactory=", describeUe3VertexFactory(m_currentUe3VertexFactory),
            " pass=", describeUe3PassType(m_currentUe3PassType),
            " prims=", drawContext.PrimitiveCount,
            " ztest=", depthTestDisabled ? 0 : 1,
            " zwrite=", zWriteEnabled ? 1 : 0,
            " target=", getCurrentRenderTargetFormat(),
            " domain=", isSceneLinearRenderTarget() ? "sceneLinear" : "display",
            matchedDescription));
        }

        if (eligible) {
          logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Rasterized, "deferred UI overlay");
          return DrawCallType { RtxGeometryStatus::Rasterized, false, true };
        }

        // Screen-space contribution passes only reached this branch through a pixel shader tag;
        // a refusal here restores the pass switch's decision rather than promoting a DoF/fog
        // draw to normal classification.
        if (m_currentUe3PassType == Ue3PassType::FullscreenPostProcess ||
            m_currentUe3PassType == Ue3PassType::FogOrDistortion) {
          logUe3Classification(drawContext, m_currentUe3PassType, RtxGeometryStatus::Ignored, "native UE3 screen-space contribution pass");
          return DrawCallType { RtxGeometryStatus::Ignored, false };
        }
        // Other ineligible tagged draws fall through to normal classification - never suppressed.
      }
    }

    return std::nullopt;
  }

}
