param(
    [string]$OutputName = 'forza-painter-geometrize-go.exe'
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

if ([System.IO.Path]::IsPathRooted($OutputName)) {
    $binaryPath = $OutputName
} else {
    $binaryPath = Join-Path $root $OutputName
}

$openclSdk = Join-Path $root 'OpenCL-SDK'
$openclInclude = Join-Path $openclSdk 'include'
$openclLib = Join-Path $openclSdk 'lib'

if (!(Test-Path (Join-Path $openclInclude 'CL\cl.h'))) {
    throw "OpenCL header not found: $openclInclude\CL\cl.h"
}
if (!(Test-Path (Join-Path $openclLib 'OpenCL.lib'))) {
    throw "OpenCL.lib not found: $openclLib\OpenCL.lib"
}

$vulkanSdk = $env:VULKAN_SDK
if (-not $vulkanSdk) {
    $vulkanSdk = 'C:\VulkanSDK\1.4.350.0'
}

$vulkanInclude = Join-Path $vulkanSdk 'Include'
$vulkanLib = Join-Path $vulkanSdk 'Lib'
$vulkanBin = Join-Path $vulkanSdk 'Bin'
$shaderDir = Join-Path $root 'shaders'

if (!(Test-Path (Join-Path (Join-Path $vulkanInclude 'vulkan') 'vulkan.h'))) {
	throw "Vulkan header not found: $vulkanInclude\vulkan\vulkan.h"
}
if (!(Test-Path (Join-Path $vulkanLib 'vulkan-1.lib'))) {
    throw "vulkan-1.lib not found: $vulkanLib\vulkan-1.lib"
}

$env:PATH = "$vulkanBin;$env:PATH"

Write-Host "=== Building shader assets ==="
& (Join-Path $root 'build-shaders.ps1')

Write-Host ""
Write-Host "=== Building single binary with OpenCL + Vulkan support ==="
Write-Host "  OpenCL SDK: $openclSdk"
Write-Host "  Vulkan SDK: $vulkanSdk"

$env:CGO_CFLAGS = "-DCL_TARGET_OPENCL_VERSION=120 -DCL_DEPTH_STENCIL=0x10BE -DCL_UNORM_INT24=0x10DF -I$openclInclude -I$vulkanInclude"
$env:CGO_LDFLAGS = "-L$openclLib -lOpenCL -L$vulkanLib -lvulkan-1"
$env:VULKAN_SDK = $vulkanSdk

Push-Location $root
try {
    go build -o $binaryPath ./cmd/forza-painter-geometrize
    if ($LASTEXITCODE -ne 0) {
        throw "go build failed with exit code $LASTEXITCODE"
    }
    Write-Host ""
    Write-Host "Built: $binaryPath"
    $distDir = Join-Path $root 'dist'
    if (!(Test-Path $distDir)) {
        New-Item -ItemType Directory -Path $distDir | Out-Null
    }
    Copy-Item $binaryPath (Join-Path $distDir (Split-Path $binaryPath -Leaf)) -Force
    $vulkanDllCandidates = @(
        (Join-Path $vulkanBin 'vulkan-1.dll'),
        'C:\Windows\System32\vulkan-1.dll',
        'C:\Windows\SysWOW64\vulkan-1.dll'
    )
    $vulkanDll = $null
    foreach ($candidate in $vulkanDllCandidates) {
        if (Test-Path $candidate) {
            $vulkanDll = $candidate
            break
        }
    }
    if (-not $vulkanDll) {
        throw "vulkan-1.dll not found in SDK bin or system directories"
    }
    Copy-Item $vulkanDll (Join-Path $distDir 'vulkan-1.dll') -Force
    if (Test-Path $shaderDir) {
        $shaderOut = Join-Path $distDir 'shaders'
        if (!(Test-Path $shaderOut)) {
            New-Item -ItemType Directory -Path $shaderOut | Out-Null
        }
        Copy-Item (Join-Path $shaderDir '*.spv') $shaderOut -Force
    }
    Write-Host ""
    Write-Host "Release package created under: $distDir"
    Write-Host ""
    Write-Host "Run with OpenCL:"
    Write-Host "  .\dist\forza-painter-geometrize-go.exe --backend opencl <image.png>"
    Write-Host ""
    Write-Host "Run with Vulkan:"
    Write-Host "  .\dist\forza-painter-geometrize-go.exe --backend vulkan <image.png>"
}
finally {
    Pop-Location
}
