Installation instructions:

1. Install a Remix release from https://github.com/NVIDIAGameWorks/rtx-remix
(Generally, the last release that came out before the build you are installing)

2. Ensure you have the correct artifact.  Most d3d9 games were compiled for x86
computers, and will need the `rtx-remix-for-x86-games-...` package.  If you know
your game was compiled for x64, use the `rtx-remix-for-x64-games-...` package.

3. Copy and paste the files directly into your game directory.

Notes:

- You should be prompted to overwrite existing files - if you aren't, you're
probably putting the files in the wrong place.


===========================================================================
THIRD-PARTY NOTICES
===========================================================================

This distribution embeds data and code from the following third-party
sources. Full license texts and per-file attribution headers live in the
dxvk-remix source tree.

---------------------------------------------------------------------------
EA Importance-Sampled FAST Noise
---------------------------------------------------------------------------

Source:    https://github.com/electronicarts/importance-sampled-FAST-noise
Embedded:  128x128x32 RG8 noise data, compiled into d3d9.dll for use by
           the cloud / atmosphere raymarching jitter (see
           src/dxvk/rtx_render/rtx_fast_noise.{h,cpp}).
License:   BSD-3-Clause variant (see data/ea-fast-noise/LICENSE.txt in
           the source tree).

  Copyright (c) 2025 Electronic Arts Inc. All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  3. Neither the name of Electronic Arts, Inc. ("EA") nor the names of
     its contributors may be used to endorse or promote products derived
     from this software without specific prior written permission.
  4. EA's marks or logos, including SEED logos, are distributed with
     this software solely for demonstration purposes and may not be
     displayed or shared other than as part of a redistribution of this
     software, provided they are redistributed without modification. No
     other rights are provided for the use of these marks and logos,
     other than for their intended demonstration purposes.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---------------------------------------------------------------------------
Blender sky_multiple_scattering.cpp
---------------------------------------------------------------------------

Source:    https://projects.blender.org/blender/blender/src/branch/main/intern/sky/source/sky_multiple_scattering.cpp
Used by:   computeGroundReflection and computeAnalyticalMultiscattering
           in src/dxvk/shaders/rtx/pass/atmosphere/multiscattering_lut.comp.slang
           (plus the analytical transmittance branch in
           src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh).
License:   MIT.

  SPDX-License-Identifier: MIT
  SPDX-FileCopyrightText: 2022 Fernando García Liñán
  SPDX-FileCopyrightText: 2011-2025 Blender Authors

  Permission is hereby granted, free of charge, to any person obtaining
  a copy of this software and associated documentation files (the
  "Software"), to deal in the Software without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to
  the following conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---------------------------------------------------------------------------
AgX display rendering transform
---------------------------------------------------------------------------

Source:    Troy Sobotka's AgX (https://github.com/sobotka/AgX), with
           HLSL/OCIO reference implementation by MrLixm
           (https://github.com/MrLixm/AgXc).
Used by:   src/dxvk/shaders/rtx/pass/tonemap/AgX.hlsl.
License:   License-free per Troy Sobotka's deliberate position; MrLixm
           inherits the same status. Attribution-only is the appropriate
           posture for open-source / non-commercial use per MrLixm's
           public statement on the AgXc repository (2024-03-19).

  Credit:
    Troy Sobotka — AgX creator and spec author.
    MrLixm       — HLSL/OCIO reference implementation.