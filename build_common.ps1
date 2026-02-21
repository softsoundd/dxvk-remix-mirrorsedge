<#
  Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
 
  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following conditions:
 
  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.
 
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  DEALINGS IN THE SOFTWARE.
#>

function SetupVS {
	param(
		[Parameter(Mandatory)]
		[string]$Platform
	)
	If ($vsWhere = Get-Command "vswhere.exe" -ErrorAction SilentlyContinue) {
	  $vsWhere = $vsWhere.Path
	} ElseIf (Test-Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe") {
	  $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
	}
	 Else {
	  Write-Error "vswhere not found. Aborting." -ErrorAction Stop
	}
	Write-Host "vswhere found at: $vsWhere" -ForegroundColor Yellow


	$vsPath = &$vsWhere -latest -version "[16.0,18.0)" -products * `
	 -requires Microsoft.Component.MSBuild `
	 -property installationPath
	If ([string]::IsNullOrEmpty("$vsPath")) {
	  Write-Error "Failed to find a supported Visual Studio installation. Aborting." -ErrorAction Stop
	}
	Write-Host "Using Visual Studio installation at: ${vsPath}" -ForegroundColor Yellow


	Push-Location "${vsPath}\VC\Auxiliary\Build"
	cmd /c "vcvarsall.bat $Platform&set" |
		ForEach-Object {
		  If ($_ -match "=") {
			  If (-not ($_.Contains('==='))) {
				  $v = $_.split("="); Set-Item -Force -Path "ENV:\$($v[0])" -Value "$($v[1])"
			  }
		  }
		}
	Pop-Location
	Write-Host "Visual Studio Command Prompt variables set ($Platform)." -ForegroundColor Yellow

	try {
		if ($pyCmd = Get-Command "py" -ErrorAction SilentlyContinue) {
			$userScriptsDir = & $pyCmd.Source -3 -c "import sysconfig; print(sysconfig.get_path('scripts', scheme='nt_user'))"
			if ($userScriptsDir -and (Test-Path $userScriptsDir)) {
				if (-not ($env:PATH -like "*$userScriptsDir*")) {
					$env:PATH = "$env:PATH;$userScriptsDir"
				}
			}
		}
	} catch {
	}
}

function Invoke-Meson {
	param(
		[Parameter(Mandatory)]
		[string[]]$MesonArgs
	)

	if ($mesonCmd = Get-Command "meson" -ErrorAction SilentlyContinue) {
		& $mesonCmd.Source @MesonArgs
		return
	}

	if ($pyCmd = Get-Command "py" -ErrorAction SilentlyContinue) {
		& $pyCmd.Source -3 -m mesonbuild.mesonmain @MesonArgs
		return
	}

	Write-Error "meson not found, install meson  or add it to PATH." -ErrorAction Stop
}

function PerformBuild {
	param(
		[Parameter(Mandatory)]
		[string]$Backend,

		[string]$Platform = "x64",

		[Parameter(Mandatory)]
		[string]$BuildFlavour,
		
		[Parameter(Mandatory)]
		[string]$BuildSubDir,

		[Parameter(Mandatory)]
		[string]$EnableTracy,

		[string]$BuildTarget,

		[string[]]$InstallTags,

		[bool]$ConfigureOnly = $false,

		[bool]$ShadersOnly = $false
	)

	SetupVS -Platform $Platform

	$CurrentDir = Get-Location
	$BuildDir = [IO.Path]::Combine($CurrentDir, $BuildSubDir)

	Push-Location $CurrentDir
		$setupArgs = @(
			"setup",
			"--buildtype", $BuildFlavour,
			"--backend", $Backend,
			("-Denable_tracy=$EnableTracy"),
			$BuildSubDir
		)
		if ( $ShadersOnly ) {
			$setupArgs += "-Ddownload_apics=False"
		}
		Invoke-Meson -MesonArgs $setupArgs
	Pop-Location

	if ( $LASTEXITCODE -ne 0 ) {
		Write-Output "Failed to run meson setup"
		exit $LASTEXITCODE
	}

	if ($ShadersOnly) {
		Push-Location $BuildDir
		Invoke-Meson -MesonArgs @("compile", "rtx_shaders")
		Pop-Location
		exit $LASTEXITCODE
	}

	if (!$ConfigureOnly) {
		Push-Location $BuildDir
			Invoke-Meson -MesonArgs @("compile", "-v")

			if ($InstallTags -and $InstallTags.Count -gt 0) {
				# join array into comma-separated list
				$tagList = $InstallTags -join ','
				Invoke-Meson -MesonArgs @("install", "--tags", $tagList)
			}
			else {
				Invoke-Meson -MesonArgs @("install")
			}
		Pop-Location

		if ( $LASTEXITCODE -ne 0 ) {
			Write-Output "Failed to run build step"
			exit $LASTEXITCODE
		}
	}
}