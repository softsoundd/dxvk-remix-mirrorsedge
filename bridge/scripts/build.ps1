#requires -version 5.1
<#
.SYNOPSIS
  Build the Remix Bridge (32-bit client + 64-bit server) reliably.

.DESCRIPTION
  Replaces upstream bridge/build_bridge_*.bat for fork-side builds.
  Discovers Visual Studio via vswhere, calls vcvarsall.bat by full
  path, runs meson setup + compile per arch in isolated cmd.exe shells
  (no env contamination between x64 and x86). Streams output to the
  PowerShell pipeline so progress and errors are visible. Throws on
  any non-zero exit.

  Upstream's build_bridge_release.bat lies about success when
  vcvarsall lookup fails inside SetupVS — this script does not.

.PARAMETER Flavor
  release (default), debug, debugoptimized.

.PARAMETER Arch
  x64, x86, or both (default).

.PARAMETER Clean
  Wipe the matching _comp* dir(s) before building.

.EXAMPLE
  .\scripts\build.ps1
  .\scripts\build.ps1 -Arch x86 -Clean
  .\scripts\build.ps1 -Flavor debugoptimized -Arch both
#>
[CmdletBinding()]
param(
    [ValidateSet('release','debug','debugoptimized')]
    [string]$Flavor = 'release',

    [ValidateSet('x64','x86','both')]
    [string]$Arch = 'both',

    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

$BridgeRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$flavorTag = switch ($Flavor) {
    'release'        { 'Release' }
    'debug'          { 'Debug' }
    'debugoptimized' { 'DebugOptimized' }
}
$buildDirPrefix = "_comp$flavorTag"

# --- Discover Visual Studio (no PATH dependency, no SetupVS lying) ---
$vsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vsWhere)) {
    throw "vswhere.exe not found at '$vsWhere'. Install the Visual Studio Installer (ships with VS2017+)."
}
$vsPath = & $vsWhere -latest -version '[16.0,18.0)' -property installationPath
if ([string]::IsNullOrWhiteSpace($vsPath)) {
    throw "No Visual Studio 2019 or 2022 installation with v142 toolchain found via vswhere."
}
$vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvarsall.bat'
if (-not (Test-Path $vcvars)) {
    throw "vcvarsall.bat not found at '$vcvars'. The detected VS install may be incomplete."
}

Write-Host "VS install:  $vsPath"  -ForegroundColor DarkGray
Write-Host "vcvars:      $vcvars"   -ForegroundColor DarkGray
Write-Host "Bridge root: $BridgeRoot" -ForegroundColor DarkGray
Write-Host "Flavor:      $Flavor / $Arch" -ForegroundColor DarkGray

$arches = if ($Arch -eq 'both') { @('x64','x86') } else { @($Arch) }

Set-Location $BridgeRoot

foreach ($a in $arches) {
    $buildDir = "${buildDirPrefix}_$a"
    Write-Host ""
    Write-Host "=== Bridge $a $Flavor → $buildDir ===" -ForegroundColor Cyan

    if ($Clean -and (Test-Path $buildDir)) {
        Write-Host "Wiping $buildDir for clean rebuild..." -ForegroundColor Yellow
        Remove-Item -Path $buildDir -Recurse -Force
    }

    $alreadyConfigured = Test-Path (Join-Path $buildDir 'meson-private')
    if ($alreadyConfigured) {
        # meson auto-reconfigures from inside the build dir if meson.build changed
        $chain = "call `"$vcvars`" $a >nul 2>&1 && cd $buildDir && meson compile"
    } else {
        $chain = "call `"$vcvars`" $a >nul 2>&1 && meson setup --buildtype $Flavor --backend ninja $buildDir && cd $buildDir && meson compile"
    }

    # Native-command stderr triggers PowerShell's NativeCommandError under
    # EAP=Stop even on exit 0. Exit code is the authoritative signal here.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        cmd.exe /c $chain
        $exit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEAP
    }
    if ($exit -ne 0) {
        throw "Bridge $a $Flavor build failed (exit $exit)."
    }
}

# Sanity-check artifacts so the caller can trust exit 0
$expected = @{}
if ($arches -contains 'x64') {
    $expected["${buildDirPrefix}_x64/src/server/NvRemixBridge.exe"] = 'x64 server'
}
if ($arches -contains 'x86') {
    $expected["${buildDirPrefix}_x86/src/client/d3d9.dll"]            = 'x86 client'
    $expected["${buildDirPrefix}_x86/src/launcher/NvRemixLauncher32.exe"] = 'x86 launcher'
}

$missing = @()
foreach ($rel in $expected.Keys) {
    $abs = Join-Path $BridgeRoot $rel
    if (-not (Test-Path $abs)) { $missing += "$($expected[$rel]) ($rel)" }
}
if ($missing.Count -gt 0) {
    throw "Build reported success but artifacts are missing: $($missing -join ', ')"
}

Write-Host ""
Write-Host "=== Bridge build OK ===" -ForegroundColor Green
foreach ($rel in $expected.Keys) {
    $abs = Join-Path $BridgeRoot $rel
    $info = Get-Item $abs
    Write-Host ("  {0,-14} {1:yyyy-MM-dd HH:mm}  {2,10:N0} bytes  {3}" -f $expected[$rel], $info.LastWriteTime, $info.Length, $rel) -ForegroundColor Green
}
