$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$shaderDir = Join-Path $root 'shaders'

if (!(Test-Path $shaderDir)) {
	throw "Shader directory not found: $shaderDir"
}

$glslc = Get-Command glslc -ErrorAction SilentlyContinue
if (-not $glslc) {
    Write-Host "glslc not found on PATH."
    Write-Host ""
    Write-Host "To compile GLSL shaders to SPIR-V, install the Vulkan SDK:"
    Write-Host "  https://vulkan.lunarg.com/sdk/home"
    Write-Host ""
    Write-Host "After installing, add the SDK bin directory to your PATH,"
    Write-Host "e.g. C:\VulkanSDK\1.3.x.x\Bin"
    Write-Host ""
    Write-Host "Then re-run this script."
    exit 1
}

Write-Host "Compiling GLSL shaders to SPIR-V..."
$shaders = @(
    @{src='eval_v3.comp';     spv='eval_v3.comp.spv'},
    @{src='eval_v4.comp';     spv='eval_v4.comp.spv'},
    @{src='apply.comp';       spv='apply.comp.spv'},
    @{src='error_grid.comp';  spv='error_grid.comp.spv'}
)

foreach ($s in $shaders) {
    $srcPath = Join-Path $shaderDir $s.src
    $spvPath = Join-Path $shaderDir $s.spv
    Write-Host "  $($s.src) -> $($s.spv)"
    & glslc -fshader-stage=compute $srcPath -o $spvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to compile $($s.src)"
    }
}

Write-Host "Done. SPIR-V files are in $shaderDir"
