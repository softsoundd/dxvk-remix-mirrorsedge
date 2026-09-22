/*
* Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
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

#include "dxvk_gpu_event.h"
#include "dxvk_gpu_query.h"
#include "dxvk_memory.h"
#include "dxvk_meta_blit.h"
#include "dxvk_meta_clear.h"
#include "dxvk_meta_copy.h"
#include "dxvk_meta_mipgen.h"
#include "dxvk_meta_pack.h"
#include "dxvk_meta_resolve.h"
#include "dxvk_pipemanager.h"
#include "dxvk_renderpass.h"
#include "dxvk_unbound.h"
#include "rtx_render/rtx_resources.h"
#include "rtx_render/rtx_ngx_wrapper.h"
#include "rtx_render/rtx_ngx_passthrough.h"
#include "rtx_render/rtx_dlfg.h"
#include "rtx_render/rtx_dlss.h"
#include "rtx_render/rtx_nis.h"
#include "rtx_render/rtx_taa.h"
#include "rtx_render/rtx_auto_exposure.h"
#include "rtx_render/rtx_tone_mapping.h"
#include "rtx_render/rtx_bloom.h"
#include "rtx_render/rtx_image_utils.h"
#include "rtx_render/rtx_postFx.h"
#include "rtx_render/rtx_srgb_dither.h"
#include "rtx_render/rtx_initializer.h"
#include "rtx_render/rtx_scene_manager.h"
#include "rtx_render/rtx_reflex.h"
#include "rtx_render/rtx_game_capturer.h"
#include "rtx_render/rtx_gpu_crash.h"
#include "rtx_render/rtx_dlss_neural_rendering.h"
#include "rtx_render/rtx_gpu_pass_timer.h"

#include "rtx_render/rtx_denoise_type.h"
#include "../util/util_lazy.h"
#include "../util/util_active.h"

namespace dxvk {

  class DxvkDevice;
  class DxvkRayReconstruction;
  class DxvkRtxdiRayQuery;
  class DxvkReSTIRGIRayQuery;
  class DxvkToneMapping;
  class DxvkBloom;
  class RtxGeometryUtils;
  class CompositePass;
  class DebugView;
  class DxvkPostFx;
  class DxvkSRGBDither;
  class OpacityMicromapManager;
  class ImGUI;
  class RtxTextureManager;
  class NeuralRadianceCache;
  class DxvkXeSS;
  class SparseRendering;

  class NGXContext;

  class DxvkObjects {

  public:

    explicit DxvkObjects(DxvkDevice* device);

    DxvkMemoryAllocator& memoryManager() {
      return m_memoryManager;
    }

    DxvkRenderPassPool& renderPassPool() {
      return m_renderPassPool;
    }

    DxvkPipelineManager& pipelineManager() {
      return m_pipelineManager;
    }

    DxvkGpuEventPool& eventPool() {
      return m_eventPool;
    }

    DxvkGpuQueryPool& queryPool() {
      return m_queryPool;
    }

    DxvkUnboundResources& dummyResources() {
      return m_dummyResources;
    }

    DxvkMetaBlitObjects& metaBlit() {
      return m_metaBlit.get(m_device);
    }

    DxvkMetaClearObjects& metaClear() {
      return m_metaClear.get(m_device);
    }

    DxvkMetaCopyObjects& metaCopy() {
      return m_metaCopy.get(m_device);
    }

    DxvkMetaResolveObjects& metaResolve() {
      return m_metaResolve.get(m_device);
    }
    
    DxvkMetaPackObjects& metaPack() {
      return m_metaPack.get(m_device);
    }

    NGXContext& metaNGXContext() {
      return m_ngxContext.get();
    }

    RtxNgxPassthrough& metaNgxPassthrough() {
      return m_ngxPassthrough.get();
    }

    DxvkDLSS& metaDLSS() {
      return m_dlss.get();
    }

    DxvkRayReconstruction& metaRayReconstruction() {
      return m_rayReconstruction.get();
    }

    DxvkDLFG& metaDLFG() {
      return m_dlfg.get();
    }

    DlssNeuralRendering& metaDlssNeuralRendering() {
      return m_dlssNeuralRendering.get();
    }

    DxvkNIS& metaNIS() {
      return m_nis.get();
    }

    DxvkTemporalAA& metaTAA() {
      return m_taa.get();
    }

    DxvkXeSS& metaXeSS() {
      return m_xess.get();
    }

    GpuCrashPass& metaGpuCrash() {
      return m_gpuCrash.get();
    }

    DebugView& metaDebugView() {
      return m_debug_view.get();
    }

    DxvkAutoExposure& metaAutoExposure() {
      return m_autoExposure.get();
    }

    DxvkToneMapping& metaToneMapping() {
      return m_toneMapping.get();
    }

    DxvkBloom& metaBloom() {
      return m_bloom.get();
    }

    RtxImageUtils& metaImageUtils() {
      return m_imageUtils.get();
    }

    DxvkPostFx& metaPostFx() {
      return m_postFx.get();
    }

    DxvkSRGBDither& metaSRGBDither() {
      return m_srgbDither.get();
    }

    RtxReflex& metaReflex() {
      return m_reflex.get(m_device);
    }

    SceneManager& getSceneManager() {
      return m_sceneManager;
    }

    Resources& getResources() {
      return m_rtResources;
    }

    RtxInitializer& getRtxInitializer() {
      return m_rtInitializer;
    }

    RtxTextureManager& getTextureManager();
    
    ImGUI& getImgui() {
      return m_imgui;
    }

    AssetExporter& metaExporter() {
      return m_exporter.get();
    }

    Rc<GameCapturer> capturer() {
      return m_capturer;
    }

    RtxGpuPassTimer& metaGpuPassTimer() {
      return m_gpuPassTimer.get(m_device);
    }

    void onDestroy();

    void setWindowHandle(const HWND hwnd) {
      m_lastKnownWindowHandle.store(hwnd);
    }
    HWND getLastKnownWindowHandle() const {
      return m_lastKnownWindowHandle.load();
    }

  private:

    DxvkDevice*                       m_device;

    DxvkMemoryAllocator               m_memoryManager;
    DxvkRenderPassPool                m_renderPassPool;
    DxvkPipelineManager               m_pipelineManager;

    DxvkGpuEventPool                  m_eventPool;
    DxvkGpuQueryPool                  m_queryPool;

    DxvkUnboundResources              m_dummyResources;

    Lazy<DxvkMetaBlitObjects>         m_metaBlit;
    Lazy<DxvkMetaClearObjects>        m_metaClear;
    Lazy<DxvkMetaCopyObjects>         m_metaCopy;
    Lazy<DxvkMetaResolveObjects>      m_metaResolve;
    Lazy<DxvkMetaPackObjects>         m_metaPack;


    // Note: SceneManager(...) retrieves m_exporter from DxvkObjects(), so m_exporter has to be initialized prior to m_sceneManager
    Lazy<AssetExporter>               m_exporter;

    // RTX Management
    SceneManager       m_sceneManager;
    Resources          m_rtResources;
    RtxInitializer     m_rtInitializer;
    std::unique_ptr<RtxTextureManager> m_textureManager;
    ImGUI              m_imgui;
    Rc<GameCapturer>   m_capturer;

    // RTX Shaders
    Active<NGXContext>                      m_ngxContext;
    Active<RtxNgxPassthrough>               m_ngxPassthrough;
    Active<DxvkDLFG>                        m_dlfg;
    Active<DxvkDLSS>                        m_dlss;
    Active<DxvkRayReconstruction>           m_rayReconstruction;
    Active<DlssNeuralRendering>             m_dlssNeuralRendering;
    Active<DxvkNIS>                         m_nis;
    Active<DxvkTemporalAA>                  m_taa;
    Active<DxvkXeSS>                        m_xess;
    Active<GpuCrashPass>                    m_gpuCrash;
    Active<DebugView>                       m_debug_view;
    Active<DxvkAutoExposure>                m_autoExposure;
    Active<DxvkToneMapping>                 m_toneMapping;
    Active<DxvkBloom>                       m_bloom;
    Active<RtxImageUtils>                   m_imageUtils;
    Active<DxvkPostFx>                      m_postFx;
    Active<DxvkSRGBDither>                  m_srgbDither;
    Lazy<RtxReflex>                         m_reflex;
    Lazy<RtxGpuPassTimer>                   m_gpuPassTimer;

    std::atomic<HWND>                       m_lastKnownWindowHandle;
  };
}
