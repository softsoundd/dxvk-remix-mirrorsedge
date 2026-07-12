#Requires -Version 5.1
$ErrorActionPreference = 'Stop'

# Fetches the pinned DLSS / NGX DLL bundle into dlss_override/ at repo root.
# meson.build prefers dlss_override/ over the packman-shipped NGX DLLs when
# this dir is populated, so re-run this after editing scripts/dlss-pins.json.

$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot    = Split-Path -Parent $scriptDir
$pinFile     = Join-Path $scriptDir 'dlss-pins.json'
$overrideDir = Join-Path $repoRoot  'dlss_override'
$cacheDir    = Join-Path $overrideDir '.cache'
$stampFile   = Join-Path $overrideDir '.installed-pin'

$pin         = Get-Content -Raw -LiteralPath $pinFile | ConvertFrom-Json
$expectedSha = $pin.sha256.ToUpperInvariant()

# Idempotent: skip if the stamp matches and every expected DLL is present.
if ((Test-Path -LiteralPath $stampFile) -and
    ((Get-Content -Raw -LiteralPath $stampFile).Trim() -eq $expectedSha)) {
    $allPresent = $true
    foreach ($name in $pin.dlls) {
        if (-not (Test-Path -LiteralPath (Join-Path $overrideDir $name))) {
            $allPresent = $false; break
        }
    }
    if ($allPresent) {
        Write-Host "DLSS override already at $($pin.release_tag) - nothing to do."
        return
    }
}

New-Item -ItemType Directory -Force -Path $overrideDir | Out-Null
New-Item -ItemType Directory -Force -Path $cacheDir    | Out-Null

$cachedZip = Join-Path $cacheDir "$($pin.release_tag).zip"
$needFetch = $true
if (Test-Path -LiteralPath $cachedZip) {
    $cachedSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $cachedZip).Hash.ToUpperInvariant()
    if ($cachedSha -eq $expectedSha) { $needFetch = $false }
}

if ($needFetch) {
    Write-Host "Fetching $($pin.url)"
    $pp = $ProgressPreference; $ProgressPreference = 'SilentlyContinue'
    try {
        Invoke-WebRequest -Uri $pin.url -OutFile $cachedZip -UseBasicParsing `
            -MaximumRedirection 5 -UserAgent 'dxvk-remix-dlss-bump/1.0'
    } finally { $ProgressPreference = $pp }

    $actualSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $cachedZip).Hash.ToUpperInvariant()
    if ($actualSha -ne $expectedSha) {
        Remove-Item -LiteralPath $cachedZip -Force
        throw "SHA-256 mismatch for $($pin.release_tag): expected $expectedSha, got $actualSha. Bundle deleted."
    }
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($cachedZip)
try {
    foreach ($name in $pin.dlls) {
        $entry = $zip.Entries | Where-Object { $_.FullName -eq $name } | Select-Object -First 1
        if (-not $entry) {
            $found = ($zip.Entries | ForEach-Object FullName) -join ', '
            throw "Bundle missing expected root-level entry '$name'. Found: $found"
        }
        $dest = Join-Path $overrideDir $name
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dest, $true)
    }
} finally { $zip.Dispose() }

Set-Content -LiteralPath $stampFile -Value $expectedSha -Encoding ASCII

Write-Host "Installed DLSS bundle $($pin.release_tag) -> $overrideDir"
Write-Host "Re-run meson configure (or rtx-build) to pick up the new DLLs."
