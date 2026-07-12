[CmdletBinding()]
param(
  [string]$GameDir,
  [string]$GameExe,
  [switch]$SkipRtxOptions,
  [switch]$SkipApiSurface
)

$ErrorActionPreference = 'Stop'

$RepoRoot   = Split-Path -Parent $PSScriptRoot
$ScriptsDir = $PSScriptRoot
$LocalFile  = Join-Path $ScriptsDir 'regen-docs.local.ps1'
$ExampleFile = Join-Path $ScriptsDir 'regen-docs.local.example.ps1'

# Resolve $GameDir / $GameExe from the local file if not given as args.
if (-not $GameDir -or -not $GameExe) {
  if (Test-Path $LocalFile) {
    . $LocalFile
  } else {
    Write-Error @"
No game directory configured.

Either pass -GameDir <path> -GameExe <exe>, or copy
  $ExampleFile
to
  $LocalFile
and edit the `$GameDir` / `$GameExe` lines.
"@
    exit 1
  }
}

if (-not $GameDir -or -not (Test-Path $GameDir)) {
  Write-Error "GameDir '$GameDir' does not exist."
  exit 1
}
if (-not $GameExe) {
  Write-Error "GameExe is not set."
  exit 1
}

$GameExePath = Join-Path $GameDir $GameExe
if (-not (Test-Path $GameExePath)) {
  Write-Error "Game executable not found at: $GameExePath"
  exit 1
}

Write-Host "regen-docs: using GameDir = $GameDir" -ForegroundColor Cyan
Write-Host "regen-docs: using GameExe = $GameExe" -ForegroundColor Cyan

function Invoke-RegenRtxOptions {
  param(
    [string]$GameDir,
    [string]$GameExePath,
    [string]$RepoRoot,
    [int]$TimeoutSeconds = 60
  )

  $outputFile = Join-Path $GameDir 'RtxOptions.md'
  $targetFile = Join-Path $RepoRoot 'RtxOptions.md'

  # Wipe any pre-existing output so we can detect a fresh write.
  if (Test-Path $outputFile) {
    Remove-Item -LiteralPath $outputFile -Force
  }

  Write-Host "regen-docs: launching $GameExePath with DXVK_DOCUMENTATION_WRITE_RTX_OPTIONS_MD=1" -ForegroundColor Cyan

  $env:DXVK_DOCUMENTATION_WRITE_RTX_OPTIONS_MD = '1'
  try {
    $proc = Start-Process -FilePath $GameExePath -WorkingDirectory $GameDir -PassThru
  } finally {
    Remove-Item Env:DXVK_DOCUMENTATION_WRITE_RTX_OPTIONS_MD -ErrorAction SilentlyContinue
  }

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  $lastSize = -1
  $stableSince = $null

  while ((Get-Date) -lt $deadline) {
    if (Test-Path $outputFile) {
      $size = (Get-Item $outputFile).Length
      if ($size -eq $lastSize -and $size -gt 0) {
        if (-not $stableSince) { $stableSince = Get-Date }
        if (((Get-Date) - $stableSince).TotalMilliseconds -ge 500) {
          break
        }
      } else {
        $lastSize = $size
        $stableSince = $null
      }
    }
    Start-Sleep -Milliseconds 100
  }

  # Kill the game process tree, whether we succeeded or timed out.
  try {
    Get-Process -Id $proc.Id -ErrorAction SilentlyContinue | ForEach-Object {
      # Recursively kill child processes too (some launchers spawn subprocesses).
      Get-CimInstance Win32_Process -Filter "ParentProcessId = $($_.Id)" -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
      Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
  } catch { }

  if (-not (Test-Path $outputFile) -or (Get-Item $outputFile).Length -eq 0) {
    Write-Error "RtxOptions.md did not appear in '$GameDir' within ${TimeoutSeconds}s. Likely a crash before RtxOptions ctor; check Remix logs in '$GameDir'."
    exit 1
  }

  Copy-Item -LiteralPath $outputFile -Destination $targetFile -Force

  $rowCount = (Get-Content $targetFile | Where-Object { $_ -match '^\|rtx\.' }).Count
  Write-Host "regen-docs: regenerated RtxOptions.md (${rowCount} rows)" -ForegroundColor Green
}

if (-not $SkipRtxOptions) {
  Invoke-RegenRtxOptions -GameDir $GameDir -GameExePath $GameExePath -RepoRoot $RepoRoot
}

function Invoke-RegenApiSurface {
  param(
    [string]$RepoRoot
  )

  $script = Join-Path $RepoRoot 'scripts-common\generate_remix_api_md.py'
  $output = Join-Path $RepoRoot 'RemixApiSurface.md'

  Write-Host "regen-docs: running $script" -ForegroundColor Cyan
  & python $script
  if ($LASTEXITCODE -ne 0) {
    Write-Error "generate_remix_api_md.py exited with code $LASTEXITCODE"
    exit $LASTEXITCODE
  }

  if (-not (Test-Path $output)) {
    Write-Error "RemixApiSurface.md was not produced at $output"
    exit 1
  }

  $size = (Get-Item $output).Length
  Write-Host "regen-docs: regenerated RemixApiSurface.md ($size bytes)" -ForegroundColor Green
}

if (-not $SkipApiSurface) {
  Invoke-RegenApiSurface -RepoRoot $RepoRoot
}
