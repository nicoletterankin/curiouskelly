# ============================================================================
# Unity WebGL Build Compressor (PowerShell)
# Compresses Unity WebGL build files using Brotli for optimal delivery
# ============================================================================

param(
    [string]$BuildDir = "digital-kelly\engines\Kelly_Engine_V2\onlykelly\Builds\WebGL\Build",
    [string]$OutputDir = "",
    [string]$BuildName = "Kelly_Web_Build"
)

$ErrorActionPreference = "Stop"

if (-not $OutputDir) {
    $OutputDir = $BuildDir
}

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Unity WebGL Build Compressor" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Build Name: $BuildName"
Write-Host "Input Dir:  $BuildDir"
Write-Host "Output Dir: $OutputDir"
Write-Host ""

# Check if build directory exists
if (-not (Test-Path $BuildDir)) {
    Write-Host "ERROR: Build directory not found: $BuildDir" -ForegroundColor Red
    exit 1
}

# Check if brotli is available (via npm package or system)
$brotliCmd = $null

# Try npx brotli-cli
try {
    $null = npx --yes brotli --help 2>$null
    $brotliCmd = "npx"
    Write-Host "Using: npx brotli" -ForegroundColor Gray
} catch {
    # Try system brotli
    $systemBrotli = Get-Command brotli -ErrorAction SilentlyContinue
    if ($systemBrotli) {
        $brotliCmd = "system"
        Write-Host "Using: system brotli" -ForegroundColor Gray
    }
}

if (-not $brotliCmd) {
    Write-Host "ERROR: Brotli not found." -ForegroundColor Red
    Write-Host "Install with: npm install -g brotli-cli" -ForegroundColor Yellow
    Write-Host "         or: scoop install brotli / choco install brotli" -ForegroundColor Yellow
    exit 1
}

# Create output directory if different
if ($OutputDir -ne $BuildDir) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
}

# Function to compress a file
function Compress-File {
    param(
        [string[]]$SourcePatterns,
        [string]$DestName
    )
    
    # Find source file
    $srcFile = $null
    foreach ($pattern in $SourcePatterns) {
        $fullPath = Join-Path $BuildDir $pattern
        if (Test-Path $fullPath) {
            $srcFile = $fullPath
            break
        }
    }
    
    if (-not $srcFile) {
        Write-Host "  Not found: $($SourcePatterns -join ', ')" -ForegroundColor Yellow
        return $false
    }
    
    $destPath = Join-Path $OutputDir $DestName
    $srcSize = (Get-Item $srcFile).Length
    
    Write-Host -NoNewline "  Compressing $([System.IO.Path]::GetFileName($srcFile))..."
    
    if ($brotliCmd -eq "npx") {
        npx --yes brotli-cli --quality 11 --output $destPath $srcFile 2>$null
    } else {
        brotli -q 11 -f $srcFile -o $destPath
    }
    
    if (Test-Path $destPath) {
        $dstSize = (Get-Item $destPath).Length
        $ratio = [math]::Round(100 - ($dstSize * 100 / $srcSize), 1)
        Write-Host " OK (${ratio}% reduction)" -ForegroundColor Green
        return $true
    } else {
        Write-Host " FAILED" -ForegroundColor Red
        return $false
    }
}

# Function to copy loader
function Copy-Loader {
    param(
        [string[]]$SourcePatterns,
        [string]$DestName
    )
    
    foreach ($pattern in $SourcePatterns) {
        $fullPath = Join-Path $BuildDir $pattern
        if (Test-Path $fullPath) {
            $destPath = Join-Path $OutputDir $DestName
            Copy-Item $fullPath $destPath -Force
            Write-Host "  Copied $pattern -> $DestName" -ForegroundColor Green
            return $true
        }
    }
    
    Write-Host "  Loader not found: $($SourcePatterns -join ', ')" -ForegroundColor Yellow
    return $false
}

Write-Host "=== Compressing Unity Build Files ===" -ForegroundColor Yellow
Write-Host ""

# Compress each file type
Compress-File -SourcePatterns @("WebGL.wasm", "$BuildName.wasm") -DestName "$BuildName.wasm.br"
Compress-File -SourcePatterns @("WebGL.data", "$BuildName.data") -DestName "$BuildName.data.br"
Compress-File -SourcePatterns @("WebGL.framework.js", "$BuildName.framework.js") -DestName "$BuildName.framework.js.br"
Copy-Loader -SourcePatterns @("WebGL.loader.js", "$BuildName.loader.js") -DestName "$BuildName.loader.js"

Write-Host ""
Write-Host "=== Compression Summary ===" -ForegroundColor Yellow
Write-Host ""

# List output files
Get-ChildItem -Path $OutputDir -Filter "$BuildName.*" | ForEach-Object {
    $sizeKB = [math]::Round($_.Length / 1KB, 1)
    $sizeMB = [math]::Round($_.Length / 1MB, 2)
    if ($sizeMB -gt 1) {
        Write-Host "  $($_.Name): ${sizeMB} MB"
    } else {
        Write-Host "  $($_.Name): ${sizeKB} KB"
    }
}

Write-Host ""
Write-Host "Compression complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Files ready for upload to R2:" -ForegroundColor Cyan
Get-ChildItem -Path $OutputDir -Filter "$BuildName.*" | ForEach-Object {
    Write-Host "  - $($_.Name)"
}



