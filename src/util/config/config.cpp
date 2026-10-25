/*
* Copyright (c) 2019-2026, NVIDIA CORPORATION. All rights reserved.
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
#include <array>
#include <fstream>
#include <sstream>
#include <iostream>
#include <regex>
#include <utility>
#include <filesystem>
#include <algorithm>

#include "config.h"

#include "../log/log.h"

#include "../util_env.h"
// NV-DXVK start: Fix some circular inclusion stuff
#include "../util_string.h"
// NV-DXVK end

namespace dxvk {

  const static std::vector<std::pair<const char*, Config>> g_appDefaults = {{
    /* Assassin's Creed Syndicate: amdags issues  */
    { R"(\\ACS\.exe$)", {{
      { "dxgi.customVendorId",              "10de" },
    }} },
    /* Dissidia Final Fantasy NT Free Edition */
    { R"(\\dffnt\.exe$)", {{
      { "dxgi.deferSurfaceCreation",        "True" },
    }} },
    /* Elite Dangerous: Compiles weird shaders    *
     * when running on AMD hardware               */
    { R"(\\EliteDangerous64\.exe$)", {{
      { "dxgi.customVendorId",              "10de" },
    }} },
    /* The Vanishing of Ethan Carter Redux        */
    { R"(\\EthanCarter-Win64-Shipping\.exe$)", {{
      { "dxgi.customVendorId",              "10de" },
    }} },
    /* The Evil Within: Submits command lists     * 
     * multiple times                             */
    { R"(\\EvilWithin(Demo)?\.exe$)", {{
      { "d3d11.dcSingleUseMode",            "False" },
    }} },
    /* Far Cry 3: Assumes clear(0.5) on an UNORM  *
     * format to result in 128 on AMD and 127 on  *
     * Nvidia. We assume that the Vulkan drivers  *
     * match the clear behaviour of D3D11.        */
    { R"(\\(farcry3|fc3_blooddragon)_d3d11\.exe$)", {{
      { "dxgi.nvapiHack",                   "False" },
    }} },
    /* Far Cry 4: Same as Far Cry 3               */
    { R"(\\FarCry4\.exe$)", {{
      { "dxgi.nvapiHack",                   "False" },
    }} },
    /* Frostpunk: Renders one frame with D3D9     *
     * after creating the DXGI swap chain         */
    { R"(\\Frostpunk\.exe$)", {{
      { "dxgi.deferSurfaceCreation",        "True" },
    }} },
    /* Nioh: See Frostpunk, apparently?           */
    { R"(\\nioh\.exe$)", {{
      { "dxgi.deferSurfaceCreation",        "True" },
    }} },
    /* Quantum Break: Mever initializes shared    *
     * memory in one of its compute shaders       */
    { R"(\\QuantumBreak\.exe$)", {{
      { "d3d11.zeroInitWorkgroupMemory",    "True" },
    }} },
    /* Anno 2205: Random crashes with state cache */
    { R"(\\anno2205\.exe$)", {{
      { "dxvk.enableStateCache",            "False" },
    }} },
    /* Fifa '19+: Binds typed buffer SRV to shader *
     * that expects raw/structured buffer SRV     */
    { R"(\\FIFA(19|[2-9][0-9])(_demo)?\.exe$)", {{
      { "dxvk.useRawSsbo",                  "True" },
    }} },
    /* Resident Evil 2/3: Ignore WaW hazards      */
    { R"(\\re(2|3|3demo)\.exe$)", {{
      { "d3d11.relaxedBarriers",            "True" },
    }} },
    /* Devil May Cry 5                            */
    { R"(\\DevilMayCry5\.exe$)", {{
      { "d3d11.relaxedBarriers",            "True" },
    }} },
    /* Call of Duty WW2                           */
    { R"(\\s2_sp64_ship\.exe$)", {{
      { "dxgi.nvapiHack",                   "False" },
    }} },
    /* Need for Speed 2015                        */
    { R"(\\NFS16\.exe$)", {{
      { "dxgi.nvapiHack",                   "False" },
    }} },
    /* Mass Effect Andromeda                      */
    { R"(\\MassEffectAndromeda\.exe$)", {{
      { "dxgi.nvapiHack",                   "False" },
    }} },
    /* Mirror`s Edge Catalyst: Crashes on AMD     */
    { R"(\\MirrorsEdgeCatalyst(Trial)?\.exe$)", {{
      { "dxgi.customVendorId",              "10de" },
    }} },
    /* Mirror's Edge (2008)                       */
    { R"(\\MirrorsEdge\.exe$)", {{
      { "rtx.d3d9.ue3EngineMode",                                    "True" },
    }} },
    /* Star Wars Battlefront (2015)               */
    { R"(\\starwarsbattlefront(trial)?\.exe$)", {{
      { "dxgi.nvapiHack",                   "False" },
    }} },
    /* Dark Souls Remastered                      */
    { R"(\\DarkSoulsRemastered\.exe$)", {{
      { "d3d11.constantBufferRangeCheck",   "True" },
    }} },
    /* Grim Dawn                                  */
    { R"(\\Grim Dawn\.exe$)", {{
      { "d3d11.constantBufferRangeCheck",   "True" },
    }} },
    /* NieR:Automata                              */
    { R"(\\NieRAutomata\.exe$)", {{
      { "d3d11.constantBufferRangeCheck",   "True" },
    }} },
    /* NieR Replicant                             */
    { R"(\\NieR Replicant ver\.1\.22474487139\.exe)", {{
      { "dxgi.syncInterval",                "1"   },
      { "dxgi.maxFrameRate",                "60"  },
    }} },
    /* SteamVR performance test                   */
    { R"(\\vr\.exe$)", {{
      { "d3d11.dcSingleUseMode",            "False" },
    }} },
    /* Hitman 2 and 3 - requires AGS library      */
    { R"(\\HITMAN(2|3)\.exe$)", {{
      { "dxgi.customVendorId",              "10de" },
    }} },
    /* Modern Warfare Remastered                  */
    { R"(\\h1_[ms]p64_ship\.exe$)", {{
      { "dxgi.customVendorId",              "10de" },
    }} },
    /* Titan Quest                                */
    { R"(\\TQ\.exe$)", {{
      { "d3d11.constantBufferRangeCheck",   "True" },
    }} },
    /* Saints Row IV                              */
    { R"(\\SaintsRowIV\.exe$)", {{
      { "d3d11.constantBufferRangeCheck",   "True" },
    }} },
    /* Saints Row: The Third                      */
    { R"(\\SaintsRowTheThird_DX11\.exe$)", {{
      { "d3d11.constantBufferRangeCheck",   "True" },
    }} },
    /* Crysis 3 - slower if it notices AMD card     *
     * Apitrace mode helps massively in cpu bound   *
     * game parts                                   */
    { R"(\\Crysis3\.exe$)", {{
      { "dxgi.customVendorId",              "10de" },
      { "d3d11.apitraceMode",               "True" },
    }} },
    /* Crysis 3 Remastered                          *
     * Apitrace mode helps massively in cpu bound   *
     * game parts                                   */
    { R"(\\Crysis3Remastered\.exe$)", {{
      { "d3d11.apitraceMode",               "True" },
    }} },
    /* Atelier series - games try to render video *
     * with a D3D9 swap chain over the DXGI swap  *
     * chain, which breaks D3D11 presentation     */
    { R"(\\Atelier_(Ayesha|Escha_and_Logy|Shallie)(_EN)?\.exe$)", {{
      { "d3d9.deferSurfaceCreation",        "True" },
    }} },
    /* Atelier Rorona/Totori/Meruru               */
    { R"(\\A(11R|12V|13V)_x64_Release(_en)?\.exe$)", {{
      { "d3d9.deferSurfaceCreation",        "True" },
    }} },
    /* Just how many of these games are there?    */
    { R"(\\Atelier_(Lulua|Lydie_and_Suelle|Ryza(_2)?)\.exe$)", {{
      { "d3d9.deferSurfaceCreation",        "True" },
    }} },
    /* ...                                        */
    { R"(\\Atelier_(Lydie_and_Suelle|Firis|Sophie)_DX\.exe$)", {{
      { "d3d9.deferSurfaceCreation",        "True" },
    }} },
    /* Fairy Tail                                 */
    { R"(\\FAIRY_TAIL\.exe$)", {{
      { "d3d9.deferSurfaceCreation",        "True" },
    }} },
    /* Nights of Azure                            */
    { R"(\\CNN\.exe$)", {{
      { "d3d9.deferSurfaceCreation",        "True" },
    }} },
    /* Star Wars Battlefront II: amdags issues    */
    { R"(\\starwarsbattlefrontii\.exe$)", {{
      { "dxgi.customVendorId",              "10de" },
    }} },
    /* F1 games - do not synchronize TGSM access  *
     * in a compute shader, causing artifacts     */
    { R"(\\F1_20(1[89]|[2-9][0-9])\.exe$)", {{
      { "d3d11.forceTgsmBarriers",          "True" },
    }} },
    /* Blue Reflection                            */
    { R"(\\BLUE_REFLECTION\.exe$)", {{
      { "d3d11.constantBufferRangeCheck",   "True" },
    }} },
    /* Secret World Legends                       */
    { R"(\\SecretWorldLegendsDX11\.exe$)", {{
      { "d3d11.constantBufferRangeCheck",   "True" },
    }} },
    /* Darksiders Warmastered - apparently reads  *
     * from write-only mapped buffers             */
    { R"(\\darksiders1\.exe$)", {{
      { "d3d11.apitraceMode",               "True" },
    }} },
    /* Monster Hunter World                       */
    { R"(\\MonsterHunterWorld\.exe$)", {{
      { "d3d11.apitraceMode",               "True" },
    }} },
    /* Kingdome Come: Deliverance                 */
    { R"(\\KingdomCome\.exe$)", {{
      { "d3d11.apitraceMode",               "True" },
    }} },
    /* Homefront: The Revolution                  */
    { R"(\\Homefront2_Release\.exe$)", {{
      { "d3d11.apitraceMode",               "True" },
    }} },
    /* Sniper Ghost Warrior Contracts             */
    { R"(\\SGWContracts\.exe$)", {{
      { "d3d11.apitraceMode",               "True" },
    }} },
    /* Shadow of the Tomb Raider - invariant      *
     * position breaks character rendering on NV  */
    { R"(\\SOTTR\.exe$)", {{
      { "d3d11.invariantPosition",          "False" },
      { "d3d11.floatControls",              "False" },
    }} },
    /* Nioh 2                                     */
    { R"(\\nioh2\.exe$)", {{
      { "dxgi.deferSurfaceCreation",        "True" },
    }} },
    /* DIRT 5 - uses amd_ags_x64.dll when it      *
     * detects an AMD GPU                         */
    { R"(\\DIRT5\.exe$)", {{
      { "dxgi.customVendorId",              "10de" },
    }} },
    /* Crazy Machines 3 - crashes on long device  *
     * descriptions                               */
    { R"(\\cm3\.exe$)", {{
      { "dxgi.customDeviceDesc",            "DXVK Adapter" },
    }} },
    /* World of Final Fantasy: Broken and useless *
     * use of 4x MSAA throughout the renderer     */
    { R"(\\WOFF\.exe$)", {{
      { "d3d11.disableMsaa",                "True" },
    }} },
    /* Final Fantasy XIV - Stuttering on NV       */
    { R"(\\ffxiv_dx11\.exe$)", {{
      { "dxvk.shrinkNvidiaHvvHeap",         "True" },
    }} },
    /* God of War - relies on NVAPI/AMDAGS for    *
     * barrier stuff, needs nvapi for DLSS        */
    { R"(\\GoW\.exe$)", {{
      { "d3d11.ignoreGraphicsBarriers",     "True" },
      { "d3d11.relaxedBarriers",            "True" },
      { "dxgi.nvapiHack",                   "False" },
    }} },
    /* AoE 2 DE - runs poorly for some users      */
    { R"(\\AoE2DE_s\.exe$)", {{
      { "d3d11.apitraceMode",               "True" },
    }} },

    /**********************************************/
    /* D3D9 GAMES                                 */
    /**********************************************/

    /* A Hat in Time                              */
    { R"(\\HatinTimeGame\.exe$)", {{
      { "d3d9.strictPow",                   "False" },
      { "d3d9.lenientClear",                "True" },
    }} },
    /* Anarchy Online                             */
    { R"(\\anarchyonline\.exe$)", {{
      { "d3d9.memoryTrackTest",             "True" },
    }} },
    /* Borderlands 2 and The Pre Sequel!           */
    { R"(\\Borderlands(2|PreSequel)\.exe$)", {{
      { "d3d9.lenientClear",                "True" },
      { "d3d9.supportDFFormats",            "False" },
    }} },
    /* Borderlands                                */
    { R"(\\Borderlands\.exe$)", {{
      { "d3d9.lenientClear",                "True" },
    }} },
    /* Gothic 3                                   */
    { R"(\\Gothic(3|3Final| III Forsaken Gods)\.exe$)", {{
      { "d3d9.supportDFFormats",            "False" },
    }} },
    /* Risen                                      */
    { R"(\\Risen[23]?\.exe$)", {{
      { "d3d9.invariantPosition",           "True" },
    }} },
    /* Sonic Adventure 2                          */
    { R"(\\Sonic Adventure 2\\(launcher|sonic2app)\.exe$)", {{
      { "d3d9.floatEmulation",              "False" },
    }} },
    /* The Sims 2,
       Body Shop,
       The Sims Life Stories,
       The Sims Pet Stories,
       and The Sims Castaway Stories             */
    { R"(\\(Sims2.*|TS2BodyShop|SimsLS|SimsPS|SimsCS)\.exe$)", {{
      { "d3d9.customVendorId",              "10de" },
      { "d3d9.customDeviceId",              "0091" },
      { "d3d9.customDeviceDesc",            "GeForce 7800 GTX" },
      { "d3d9.disableA8RT",                 "True" },
      { "d3d9.supportX4R4G4B4",             "False" },
      { "d3d9.maxAvailableMemory",          "2048" },
      { "d3d9.memoryTrackTest",             "True" },
    }} },
    /* Dead Space uses the a NULL render target instead
       of a 1x1 one if DF24 is NOT supported      */
    { R"(\\Dead Space\.exe$)", {{
      { "d3d9.supportDFFormats",                 "False" },
    }} },
    /* Halo 2                                     */
    { R"(\\halo2\.exe$)", {{
      { "d3d9.invariantPosition",           "True" },
    }} },
    /* Halo CE/HaloPC                             */
    { R"(\\halo(ce)?\.exe$)", {{
      { "d3d9.invariantPosition",           "True" },
      // Game enables minor decal layering fixes
      // specifically when it detects AMD.
      // Avoids chip being detected as unsupported
      // when on intel. Avoids possible path towards
      // invalid texture addressing methods.
      { "d3d9.customVendorId",              "1002" },
      // Avoids card not recognized error.
      // Keeps game's rendering methods consistent
      // for optimal compatibility.
      { "d3d9.customDeviceId",              "4172" },
      // The game uses incorrect sampler types in
      // the shaders for glass rendering which
      // breaks it on native + us if we don't
      // spec-constantly chose the sampler type
      // automagically.
      { "d3d9.forceSamplerTypeSpecConstants", "True" },
      { "rtx.uiTextures",        "" },
    }} },
    /* Counter Strike: Global Offensive
       Needs NVAPI to avoid a forced AO + Smoke
       exploit so we must force AMD vendor ID.    */
    { R"(\\csgo\.exe$)", {{
      { "d3d9.customVendorId",              "1002" },
    }} },
    /* Vampire - The Masquerade Bloodlines        */
    { R"(\\vampire\.exe$)", {{
      { "d3d9.deferSurfaceCreation",        "True" },
      { "d3d9.memoryTrackTest",             "True" },
      { "d3d9.maxAvailableMemory",          "1024" },
    }} },
    /* Senran Kagura Shinovi Versus               */
    { R"(\\SKShinoviVersus\.exe$)", {{
      { "d3d9.forceAspectRatio",            "16:9" },
    }} },
    /* Metal Slug X                               */
    { R"(\\mslugx\.exe$)", {{
      { "d3d9.supportD32",                  "False" },
    }} },
    /* Skyrim (NVAPI)                             */
    { R"(\\TESV\.exe$)", {{
      { "d3d9.customVendorId",              "1002" },
    }} },
    /* RTHDRIBL Demo                              
       Uses DONOTWAIT after GetRenderTargetData
       then goes into an infinite loop if it gets
       D3DERR_WASSTILLDRAWING.
       This is a better solution than penalizing
       other apps that use this properly.         */
    { R"(\\rthdribl\.exe$)", {{
      { "d3d9.allowDoNotWait",              "False" },
    }} },
    /* Hyperdimension Neptunia U: Action Unleashed */
    { R"(\\Neptunia\.exe$)", {{
      { "d3d9.forceAspectRatio",            "16:9" },
    }} },
    /* D&D - The Temple Of Elemental Evil          */
    { R"(\\ToEE\.exe$)", {{
      { "d3d9.allowDiscard",                "False" },
    }} },
    /* ZUSI 3 - Aerosoft Edition                  */
    { R"(\\ZusiSim\.exe$)", {{
      { "d3d9.noExplicitFrontBuffer",       "True" },
    }} },
    /* GTA IV (NVAPI)                             */
    /* Also thinks we're always on Intel          *
     * and will report/use bad amounts of VRAM.   */
    { R"(\\GTAIV\.exe$)", {{
      { "d3d9.customVendorId",              "1002" },
      { "dxgi.emulateUMA",                  "True" },
    }} },
    /* Battlefield 2 (bad z-pass)                 */
    { R"(\\BF2\.exe$)", {{
      { "d3d9.longMad",                     "True" },
      { "d3d9.invariantPosition",           "True" },
    }} },
    /* SpellForce 2 Series                        */
    { R"(\\SpellForce2.*\.exe$)", {{
      { "d3d9.forceSamplerTypeSpecConstants", "True" },
    }} },
    /* Everquest 2                                */
    { R"(\\EverQuest2.*\.exe$)", {{
      { "d3d9.alphaTestWiggleRoom", "True" },
    }} },
    /* Tomb Raider: Legend                       */
    { R"(\\trl\.exe$)", {{
      { "d3d9.apitraceMode",                "True" },
    }} },
    /* Everquest                                 */
    { R"(\\eqgame\.exe$)", {{
      { "d3d9.apitraceMode",                "True" },
    }} },
    /* Dark Messiah of Might & Magic             */
    { R"(\\mm\.exe$)", {{
      { "d3d9.deferSurfaceCreation",        "True" },
      { "d3d9.memoryTrackTest",             "True" },
    }} },
    /* Mafia 2                                   */
    { R"(\\mafia2\.exe$)", {{
      { "d3d9.customVendorId",              "10de" },
      { "d3d9.customDeviceId",              "0402" },
    }} },
    /* Warhammer: Online                         */
    { R"(\\WAR(-64)?\.exe$)", {{
      { "d3d9.customVendorId",              "1002" },
    }} },
    /* Dragon Nest                               */
    { R"(\\DragonNest_x64\.exe$)", {{
      { "d3d9.memoryTrackTest ",            "True" },
    }} },
    /* Dal Segno                                 */
    { R"(\\DST\.exe$)", {{
      { "d3d9.deferSurfaceCreation",        "True" },
    }} },
    /* HL2 engine                              */
    { R"(\\hl2\.exe$)", {{
      { "rtx.baseGameModRegex", std::string("sourcemods") },
      { "rtx.baseGameModPathRegex", std::string("-game \"([a-zA-Z]:.*sourcemods.*)\"") },
      { "rtx.uiTextures",
        "7C47908363E9FB46,"
        "49C49D3F95609C9D,"
        "0ACC5AF8C7A6A72C,"
        "25F9AA6D11F0F1E0,"
        "0F707B765176FA99,"
        "A0B13F306011D748,"
        "71C560B061683B20,"
        "b1efb6a865b3082c,"
        "aedb4949ec308638," 
        "BC840D956C24C33C,"
        "010500D8F9BC71A1,"
        "65C067A6504C559E,"
        "7E20D6C917522EE4,"
        "1F06EE8596B7DD41,"
        "1FB0EBA5FEBD1B5A,"
        "C0F138D79131F8C1" // cake splash screen (after beating the game)
      },
      { "rtx.uniqueObjectDistance",  "300.0" }, // Game is 1unit=1cm - picking up objects can move them very quickly, 3m should be sufficient.
      { "rtx.temporalAA.maximumRadiance",                       "10000.0" },
      { "rtx.temporalAA.colorClampingFactor",                   "1.0"  },
      { "rtx.temporalAA.newFrameWeight",                        "0.05" },
      { "rtx.postfx.motionBlurSampleCount",                     "4"     },
      { "rtx.postfx.exposureFraction",                          "0.4"   },
      { "rtx.postfx.blurDiameterFraction",                      "0.02"  },
      { "rtx.postfx.motionBlurMinimumVelocityThresholdInPixel", "1.5"   },
      { "rtx.postfx.motionBlurDynamicDeduction",                "0.075" },
      { "rtx.postfx.motionBlurJitterStrength",                  "0.6"   },
      { "rtx.postfx.enableMotionBlurNoiseSample",               "True"  },
      { "rtx.postfx.chromaticAberrationAmount",                 "0.0"   },
      { "rtx.postfx.chromaticCenterAttenuationAmount",          "0.975" },
      { "rtx.postfx.vignetteIntensity",                         "0.8"   },
      { "rtx.postfx.vignetteRadius",                            "0.8"   },
      { "rtx.postfx.vignetteSoftness",                          "0.1"   },
      { "rtx.enableNearPlaneOverride",                          "True"  },
    }} },
    /* Kohan II                                  */
    { R"(\\k2\.exe$)", {{
      { "d3d9.memoryTrackTest",             "True" },
    }} },
    /* Ninja Gaiden Sigma 1/2                    */
    { R"(\\NINJA GAIDEN SIGMA(2)?\.exe$)", {{
      { "d3d9.deferSurfaceCreation",        "True" },
    }} },
    /* Demon Stone breaks at frame rates > 60fps */
    { R"(\\Demonstone\.exe$)", {{
      { "d3d9.maxFrameRate",                "60" },
    }} },
    /* Far Cry 1 has worse water rendering when it detects AMD GPUs */
    { R"(\\FarCry\.exe$)", {{
      { "d3d9.customVendorId",              "10de" },
    }} },
    /* Earth Defense Force 5 */
    { R"(\\EDF5\.exe$)", {{
      { "dxgi.tearFree",                    "False" },
      { "dxgi.syncInterval",                "1"     },
    }} },
    /* Sine Mora EX */
    { R"(\\SineMoraEX\.exe$)", {{
      { "d3d9.maxFrameRate",                "60" },
    }} },
    /* Fantasy Grounds                           */
    { R"(\\FantasyGrounds\.exe$)", {{
      { "d3d9.noExplicitFrontBuffer",       "True" },
    }} },
    /* Red Orchestra 2                           */
    { R"(\\ROGame\.exe$)", {{
      { "d3d9.floatEmulation",              "Strict" },
    }} },
    /* Driver: Parallel Lines crash prevention  */
    { R"(\\DriverParallelLines\.exe$)",{{
      { "d3d9.deferSurfaceCreation",       "False"   },
    }} },
    /* Sword and Fairy 4 flickering fix         */
    { R"(\\PAL4\.exe$)",{{
    { "d3d9.noExplicitFrontBuffer",        "True"    },
    }} },
    /* Dark Souls II                            */
    { R"(\\DarkSoulsII\.exe$)", {{
      { "d3d9.floatEmulation",              "Strict" },
    }} },
    /* Dogfight 1942                            */
    { R"(\\Dogfight1942\.exe$)", {{
      { "d3d9.floatEmulation",              "Strict" },
    }} },
    /* Bayonetta                                */
    { R"(\\Bayonetta\.exe$)", {{
      { "d3d9.floatEmulation",              "Strict" },
    }} },
    /* Rayman Origins                           */
    { R"(\\Rayman Origins\.exe$)", {{
      { "d3d9.floatEmulation",              "Strict" },
    }} },
    /* Guilty Gear Xrd -Relevator-              */
    { R"(\\GuiltyGearXrd\.exe$)", {{
      { "d3d9.floatEmulation",              "Strict" },
    }} },
    /* Richard Burns Rally                      */
    { R"(\\RichardBurnsRally_SSE\.exe$)", {{
      { "d3d9.floatEmulation",              "Strict" },
    }} },
    /* BlazBlue Centralfiction                  */
    { R"(\\BBCF\.exe$)", {{
      { "d3d9.floatEmulation",              "Strict" },
    }} },
    /* James Cameron's Avatar needs invariantPosition to fix black flickering vegetation */
    { R"(\\Avatar\.exe$)", {{
      { "d3d9.invariantPosition",              "True" },
    }} },
  }};


  static bool isWhitespace(char ch) {
    return ch == ' ' || ch == '\x9' || ch == '\r';
  }

  
  static bool isValidKeyChar(char ch) {
    return (ch >= '0' && ch <= '9')
        || (ch >= 'A' && ch <= 'Z')
        || (ch >= 'a' && ch <= 'z')
        || (ch == '.' || ch == '_');
  }


  static size_t skipWhitespace(const std::string& line, size_t n) {
    while (n < line.size() && isWhitespace(line[n]))
      n += 1;
    return n;
  }


  struct ConfigContext {
    bool active;
  };


  static void parseUserConfigLine(Config& config, ConfigContext& ctx, const std::string& line) {
    std::stringstream key;
    std::stringstream value;

    // Extract the key
    size_t n = skipWhitespace(line, 0);

    if (n < line.size() && line[n] == '[') {
      n += 1;

      size_t e = line.size() - 1;
      while (e > n && line[e] != ']')
        e -= 1;

      while (n < e)
        key << line[n++];
      
      ctx.active = key.str() == env::getExeName();
    } else {
      while (n < line.size() && isValidKeyChar(line[n]))
        key << line[n++];
      
      // Check whether the next char is a '='
      n = skipWhitespace(line, n);
      if (n >= line.size() || line[n] != '=')
        return;

      // Extract the value
      bool insideString = false;
      n = skipWhitespace(line, n + 1);

      while (n < line.size()) {
        // NV-DXVK start: allow white-space in vector entries within a line
        // NV-DXVK end
        if (line[n] == '"') {
          insideString = !insideString;
          n++;
        } else
          value << line[n++];
      }
      
      if (ctx.active)
        config.setOption(key.str(), value.str());
    }
  }

  // NV-DXVK start: Configuration parsing logic moved out for sharing between multiple configuration loading functions
  static Config parseConfigFile(std::string filePath) {
    Config config;
    
    // Open the file if it exists
    std::ifstream stream(str::tows(filePath.c_str()).c_str());

    if (!stream) {
      Logger::info(str::format("No config file found at: ", filePath));
      return config;
    }
    
    // Inform the user that we loaded a file, might
    // help when debugging configuration issues
    Logger::info(str::format("Found config file: ", filePath));

    // Initialize parser context
    ConfigContext ctx;
    ctx.active = true;

    // Parse the file line by line
    std::string line;

    while (std::getline(stream, line))
      parseUserConfigLine(config, ctx, line);
    
    Logger::info("Parsed config file.");
    return config;
  }
  // NV-DXVK end

  Config::Config() { }
  Config::~Config() { }


  Config::Config(OptionMap&& options)
  : m_options(std::move(options)) { }


  void Config::merge(const Config& other) {
    for (auto& pair : other.m_options)
      m_options[pair.first] = pair.second;
  }

  // NV-DXVK start: new methods
  std::string Config::generateOptionString(const bool& value) {
    return value ? std::string("True") : std::string("False");
  }

  std::string Config::generateOptionString(const int32_t& value) {
    return std::to_string(value);
  }

  std::string Config::generateOptionString(const uint32_t& value) {
    return std::to_string(value);
  }

  std::string Config::generateOptionString(const float& value) {
    std::stringstream ss;
    ss << value;
    return ss.str();
  }

  std::string Config::generateOptionString(const Vector2i& value) {
    std::stringstream ss;
    ss << value.x << ", " << value.y;
    return ss.str();
  }

  // NV-DXVK start: added a variant
  std::string Config::generateOptionString(const Vector2& value) {
    std::stringstream ss;
    ss << value.x << ", " << value.y;
    return ss.str();
  }
  // NV-DXVK end

  std::string Config::generateOptionString(const Vector3& value) {
    std::stringstream ss;
    ss << value.x << ", " << value.y << ", " << value.z;
    return ss.str();
  }

  // NV-DXVK start: added a variant
  std::string Config::generateOptionString(const Vector4& value) {
    std::stringstream ss;
    ss << value.x << ", " << value.y << ", " << value.z << ", " << value.w;
    return ss.str();
  }
  // NV-DXVK end

  std::string Config::generateOptionString(const Tristate& value) {
    switch (value) {
    default:
    case Tristate::Auto: return "Auto";
    case Tristate::False: return "False";
    case Tristate::True: return "True";
    }
  }
  // NV-DXVK end

  void Config::setOption(const std::string& key, const std::string& value) {
    m_options.insert_or_assign(key, value);
  }

  // NV-DXVK start: rvalue variant for less allocations
  void Config::setOptionMove(std::string&& key, std::string&& value) {
    m_options.insert_or_assign(std::move(key), std::move(value));
  }
  // NV-DXVK end

  void Config::setOption(const std::string& key, const bool& value) {
    setOption(key, generateOptionString(value));
  }

  void Config::setOption(const std::string& key, const int32_t& value) {
    setOption(key, generateOptionString(value));
  }

  void Config::setOption(const std::string& key, const uint32_t& value) {
    setOption(key, generateOptionString(value));
  }

  void Config::setOption(const std::string& key, const float& value) {
    setOption(key, generateOptionString(value));
  }

  void Config::setOption(const std::string& key, const Vector2i& value) {
    setOption(key, generateOptionString(value));
  }

  // NV-DXVK start: added a variant
  void Config::setOption(const std::string& key, const Vector2& value) {
    setOption(key, generateOptionString(value));
  }

  void Config::setOption(const std::string& key, const Vector4& value) {
    setOption(key, generateOptionString(value));
  }
  // NV-DXVK end

  void Config::setOption(const std::string& key, const Vector3& value) {
    setOption(key, generateOptionString(value));
  }

  void Config::setOption(const std::string& key, const Tristate& value) {
    setOption(key, generateOptionString(value));
  }

  std::string Config::getOptionValue(const char* option) const {
    auto iter = m_options.find(option);

    return iter != m_options.end()
      ? iter->second : std::string();
  }

  bool Config::parseOptionValue(
    const std::string&  value,
          std::string&  result) {
    if (value.size() == 0)
      return false;
    result = value;
    return true;
  }

  bool Config::parseOptionValue(
    const std::string& value,
    std::vector<std::string>& result) {
    std::stringstream ss(value);
    std::string s;
    while (std::getline(ss, s, ',')) {
      result.push_back(s);
    }
    return true;
  }

  bool Config::parseOptionValue(
    const std::string&  value,
          bool&         result) {
    // NV-DXVK start: Allow 1 and 0 for true/false options
    static const std::array<std::pair<const char*, bool>, 4> s_lookup = {{
      { "true",  true  },
      { "false", false },
      { "1",  true  },
      { "0", false },
    }};
    // NV-DXVK end

    return parseStringOption(value,
      s_lookup.begin(), s_lookup.end(), result);
  }

  bool Config::parseOptionValue(
    const std::string&  value,
          int32_t&      result) {
    if (value.size() == 0)
      return false;

    // NV-DXVK start: skip whitespaces at start of number strings
    try {
      result = std::stoi(value);
      return true;
    }
    catch (...) {
      return false;
    }
    // NV-DXVK end
  }

  bool Config::parseOptionValue(
    const std::string& value,
          uint32_t&    result) {
    if (value.size() == 0)
      return false;

    // NV-DXVK start: skip whitespaces at start of number strings
    try {
      result = std::stol(value);
      return true;
    }
    catch (...) {
      return false;
    }
    // NV-DXVK end
  }

  bool Config::parseOptionValue( 
    const std::string&  value,
          float&        result) {
    if (value.size() == 0)
      return false;

    // NV-DXVK start: handle invalid inputs
    try {
      result = std::stof(value);
      return true;
    }
    catch (...) {
      return false;
    }
    // NV-DXVK end
  }

  bool Config::parseOptionValue(
    const std::string& value,
    Vector2i& result) {
    std::stringstream ss(value);
    std::string s;

    // NV-DXVK start: promote scalar to vectors
    if (!std::getline(ss, s, ',') || !parseOptionValue(s, result[0])) {
      return false;
    }

    // If there is only a single value in the config for this vector, copy it to the other channels
    if (!std::getline(ss, s, ',')) {
      result = Vector2i(result[0]);
      return true;
    }

    if (!parseOptionValue(s, result[1])) {
      return false;
    }

    return true;
    // NV-DXVK end
  }

  // NV-DXVK start: added a variant
  bool Config::parseOptionValue(
    const std::string& value,
    Vector2& result) {
    std::stringstream ss(value);
    std::string s;

    if (!std::getline(ss, s, ',') || !parseOptionValue(s, result[0])) {
      return false;
    }

    // If there is only a single value in the config for this vector, copy it to the other channels
    if (!std::getline(ss, s, ',')) {
      result = Vector2(result[0]);
      return true;
    }
    if (!parseOptionValue(s, result[1])) {
      return false;
    }

    return true;
  }

  bool Config::parseOptionValue(
    const std::string& value,
    Vector4& result) {
    std::stringstream ss(value);
    std::string s;

    if (!std::getline(ss, s, ',') || !parseOptionValue(s, result[0])) {
      return false;
    }

    // If there is only a single value in the config for this vector, copy it to the other channels
    if (!std::getline(ss, s, ',')) {
      result = Vector4(result[0]);
      return true;
    }

    if (!parseOptionValue(s, result[1]) ||
        !std::getline(ss, s, ',') || !parseOptionValue(s, result[2]) ||
        !std::getline(ss, s, ',') || !parseOptionValue(s, result[3])) {
      return false;
    }

    return true;
  }
  // NV-DXVK end

  bool Config::parseOptionValue(
    const std::string& value,
    Vector3& result) {
    std::stringstream ss(value);
    std::string s;

    // NV-DXVK start: promote scalar to vectors
    if (!std::getline(ss, s, ',') || !parseOptionValue(s, result[0])) {
      return false;
    }

    // If there is only a single value in the config for this vector, copy it to the other channels
    if (!std::getline(ss, s, ',')) {
      result = Vector3(result[0]);
      return true;
    }

    if (!parseOptionValue(s, result[1]) ||
        !std::getline(ss, s, ',') || !parseOptionValue(s, result[2])) {
      return false;
    }

    return true;
    // NV-DXVK end
  }

  bool Config::parseOptionValue(
    const std::string& value,
    VirtualKeys& result) {
    std::stringstream ss(value);
    std::string s;
    bool bFoundValidConfig = false;
    VirtualKeys virtKeys;
    while (std::getline(ss, s, kVirtualKeyDelimiter)) {
      VirtualKey vk;
      try {
        // Strip whitespace from s
        s.erase(std::remove_if(s.begin(), s.end(), isWhitespace), s.end());
        
        if(s.find("0x") != std::string::npos) {
          VkValue vkVal = std::stoul(s, nullptr, 16);
          vk.val = vkVal;
        } else {
          vk = KeyBind::getVk(toUpper(s));
        }
        if(!KeyBind::isValidVk(vk)) {
          Logger::err(str::format("Failed to parse virtual key string: '", s, "' string does not map to valid Keybind."));
          return false;
        }
        virtKeys.push_back(vk);
        bFoundValidConfig = true;
      } catch (const std::invalid_argument& e) {
        Logger::err(str::format("Failed to parse virtual key hex code: '", s, "' - Invalid format."));
        return false;
      } catch (const std::out_of_range& e) {
        Logger::err(str::format("Failed to parse virtual key hex code: '", s, "' - Value out of range."));
        return false;
      }
    }
    if(bFoundValidConfig) {
      result = std::move(virtKeys);
    }
    return bFoundValidConfig;
  }
  
  bool Config::parseOptionValue(
    const std::string&  value,
          Tristate&     result) {
    static const std::array<std::pair<const char*, Tristate>, 3> s_lookup = {{
      { "true",  Tristate::True  },
      { "false", Tristate::False },
      { "auto",  Tristate::Auto  },
    }};

    return parseStringOption(value,
      s_lookup.begin(), s_lookup.end(), result);
  }


  template<typename I, typename V>
  bool Config::parseStringOption(
          std::string   str,
          I             begin,
          I             end,
          V&            value) {
    str = Config::toLower(str);

    for (auto i = begin; i != end; i++) {
      if (str == i->first) {
        value = i->second;
        return true;
      }
    }

    return false;
  }

  // NV-DXVK start: Config file loading
  Config Config::getOptionLayerConfig(const std::string& configPath) {
    Logger::info(str::format("Attempting to parse option layer: ", configPath, "..."));
    return parseConfigFile(configPath);
  }
  // NV-DXVK end 

  Config Config::getAppConfig(const std::string& appName) {
    auto appConfig = std::find_if(g_appDefaults.begin(), g_appDefaults.end(),
      [&appName] (const std::pair<const char*, Config>& pair) {
        std::regex expr(pair.first, std::regex::extended | std::regex::icase);
        return std::regex_search(appName, expr);
      });
    
    if (appConfig != g_appDefaults.end()) {
      // NV-DXVK change: Update getAppConfig logging
      Logger::info(str::format("Found app config for executable: ", appName));
      return appConfig->second;
    }

    // NV-DXVK addition: Update getAppConfig logging
    Logger::info(str::format("Did not find app config for executable: ", appName));
    return Config();
  }

  void Config::serializeCustomConfig(const Config& config, std::string filePath, std::string filterStr) {
    // Open the file if it exists
    std::ofstream stream(str::tows(filePath.c_str()).c_str());

    if (!stream)
      return;

    Logger::info(str::format("Serializing config file: ", filePath));

    for (const auto& line : config.m_options) {
      // Write if no filter specified, or if key matches the filter
      if (filterStr.empty() || line.first.find(filterStr) != std::string::npos)
        stream << line.first << " = " << line.second << std::endl;
    }
  }

  // NV-DXVK start: Extend logOptions function
  void Config::logOptions(const char* configName) const {
    if (!m_options.empty()) {
      Logger::info(str::format(configName, " configuration:"));

      for (auto& pair : m_options)
        Logger::info(str::format("  ", pair.first, " = ", pair.second));
    }
  }
  // NV-DXVK end

  std::string Config::toLower(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(),
      [] (unsigned char c) { return (c >= 'A' && c <= 'Z') ? (c + 'a' - 'A') : c; });
    return str;
  }

  std::string Config::toUpper(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(),
      [] (unsigned char c) { return (c >= 'a' && c <= 'z') ? (c - 'a' + 'A') : c; });
    return str;
  }

}
