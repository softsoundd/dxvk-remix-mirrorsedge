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
#pragma once

#include "rtx_dlss.h"

namespace dxvk {

  class NGXNeuralRenderingContext;

  class DlssNeuralRendering : public DxvkDLSS {
  public:
    explicit DlssNeuralRendering(DxvkDevice* device);

    bool supportsDlssNeuralRendering() const;

    void showDlssNeuralRenderingImguiSettings();
    void showDlssNeuralRenderingImguiTuning();

    // Returns true when NGX produced a valid output image.
    // Color, depth, and motion vectors must share the output resolution. motionVectorScale
    // converts stored motion-vector units into that resolution's pixels.
    bool dispatch(
      Rc<RtxContext> ctx,
      DxvkBarrierSet& barriers,
      const Resources::Resource& inColor,
      const Resources::Resource& outColor,
      const Resources::Resource& motionVectors,
      const Resources::Resource& depth,
      const Resources::Resource& controlMask,
      bool resetHistory,
      float motionVectorScaleX,
      float motionVectorScaleY);

    void release();

    bool useDlssNeuralRendering() const;

    void setDlssNeuralRenderingSettings(const uint32_t displaySize[2]);

    virtual void onDestroy();

    enum class Model : int {
      Model0 = 0,
      Model1 = 1,
      Model2 = 2,
    };

    RTX_OPTION_ARGS("rtx.dlssNeuralRendering", bool, enable, false,
                    "Enables DLSS 3D-Guided Neural Rendering, which adjusts detail, shadows, materials and tone in the finished frame.",
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx.dlssNeuralRendering", float, intensity, 1.0f,
                    "Controls the overall DLSS 3D-Guided Neural Rendering effect. Lower values reduce the effect; higher values increase it.",
                    args.minValue = 0.0f, args.maxValue = 1.0f);
    RTX_OPTION_ARGS("rtx.dlssNeuralRendering", float, structuralStrength, 0.7f,
                    "Strength of adjustments to detail, shadows and materials. Lower values keep more of the original image.",
                    args.minValue = 0.0f, args.maxValue = 1.0f,
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dlssNeuralRendering", float, toneStrength, 0.3f,
                    "Strength of adjustments to lightness, darkness and colour. Lower values keep more of the original tone.",
                    args.minValue = 0.0f, args.maxValue = 1.0f,
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dlssNeuralRendering", Model, model, Model::Model0,
                    "Model variant. Results vary by scene, so compare them in game. 0: Model A, 1: Model B, 2: Model C.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dlssNeuralRendering", bool, useAutoMask, true,
                    "Detects characters automatically so Character Intensity controls how strongly they are adjusted.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.dlssNeuralRendering", float, skinStructureStrength, 0.5f,
                    "Strength of structural adjustments to characters found by Auto Mask. Lower values keep more of their original look.",
                    args.minValue = 0.0f, args.maxValue = 1.0f,
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION("rtx.dlssNeuralRendering", bool, showVramUsage, false,
               "Shows SDK VRAM allocation in the DLSS 3D-Guided Neural Rendering settings. Queries aggregate SDK allocation at feature creation; excludes Remix textures. Enable before activating the feature.");

  protected:
    bool isEnabled() const override;
    bool onActivation(Rc<DxvkContext>& ctx) override;
    void onDeactivation() override;

  private:
    bool initializeDlssNeuralRendering(Rc<DxvkContext> pRenderContext);
    void disableAfterFailure(const char* pReason);

    uint32_t m_displaySize[2] = {};

    std::unique_ptr<NGXNeuralRenderingContext> m_dlssNeuralRenderingContext;
  };
} // namespace dxvk
