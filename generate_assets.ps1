# This script is used to generate game assets by calling the Google AI Studio Image Generation API.

# ----------------- CONFIGURATION -----------------
$modelName = "imagen-3.0-generate-002"  # Imagen 3.0 model for image generation
# Reference images directory - check both relative and absolute paths
$referenceImagePath = if (Test-Path "C:\iLearnStudio\projects\Kelly\Ref") { 
    "C:\iLearnStudio\projects\Kelly\Ref" 
} else { 
    "iLearnStudio/projects/Kelly/Ref" 
}
# Primary character references: headshot2-kelly-base169 101225.png, kelly_directors_chair_8k_light (2).png
# Wardrobe references: reinmaker kelly outfit base.png (clothing only)

# Vertex AI Configuration
$vertexAIProjectId = "gen-lang-client-0005524332"
$vertexAILocation = "us-central1"
$vertexAIEndpoint = "https://aiplatform.googleapis.com/v1/projects/$vertexAIProjectId/locations/$vertexAILocation/publishers/google/models/$modelName`:predict"

# Resolve script root (required for locating helper scripts)
if (-not $PSScriptRoot) {
    $script:PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
} else {
    $script:PSScriptRoot = $PSScriptRoot
}

# Detect optional Python helper for reference-image generation
$script:PythonHelperPath = Join-Path $script:PSScriptRoot "tools\generate_vertex_image_with_references.py"
$script:PythonExecutable = $null
$script:PythonPrefixArgs = @()
$script:PythonHelperAvailable = $false

$pythonCandidates = @("python", "python3", "py")
foreach ($candidate in $pythonCandidates) {
    try {
        $cmd = Get-Command $candidate -ErrorAction Stop
        if ($cmd) {
            if ($cmd.Name -eq "py") {
                $script:PythonExecutable = $cmd.Source
                $script:PythonPrefixArgs = @("-3")
            } else {
                $script:PythonExecutable = $cmd.Source
            }
            break
        }
    } catch {
        continue
    }
}

if ((Test-Path $script:PythonHelperPath) -and $script:PythonExecutable) {
    $script:PythonHelperAvailable = $true
    Write-Host "Python reference helper detected ($script:PythonExecutable)." -ForegroundColor Gray
} else {
    Write-Host "Python reference helper unavailable; defaulting to REST API for image generation." -ForegroundColor DarkGray
}

# Function to get OAuth2 access token using gcloud
function Get-AccessToken {
    # Try to find gcloud in common locations
    $gcloudPaths = @(
        "gcloud",
        "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
        "$env:ProgramFiles\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
        "$env:ProgramFiles(x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
        "$env:USERPROFILE\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
    )
    
    $gcloudExe = $null
    foreach ($path in $gcloudPaths) {
        if ($path -eq "gcloud") {
            # Try direct command
            try {
                $null = Get-Command gcloud -ErrorAction Stop
                $gcloudExe = "gcloud"
                break
            } catch {
                continue
            }
        } elseif (Test-Path $path) {
            $gcloudExe = $path
            break
        }
    }
    
    if (-not $gcloudExe) {
        Write-Host "ERROR: gcloud not found. Please install Google Cloud SDK." -ForegroundColor Red
        Write-Host "Download from: https://cloud.google.com/sdk/docs/install" -ForegroundColor Yellow
        Write-Host "Or add gcloud to your PATH" -ForegroundColor Yellow
        return $null
    }
    
    try {
        $token = & $gcloudExe auth print-access-token 2>&1
        if ($LASTEXITCODE -eq 0 -and $token -and $token -notmatch "ERROR") {
            return $token.Trim()
        } else {
            Write-Host "ERROR: Failed to get access token. Make sure gcloud is authenticated." -ForegroundColor Red
            Write-Host "Run: $gcloudExe auth login" -ForegroundColor Yellow
            return $null
        }
    } catch {
        Write-Host "ERROR: Failed to execute gcloud: $($_.Exception.Message)" -ForegroundColor Red
        return $null
    }
}

# Get access token at script start
$accessToken = Get-AccessToken
if (-not $accessToken) {
    Write-Host "`nCannot proceed without access token. Please authenticate with gcloud." -ForegroundColor Red
    exit 1
}
Write-Host "Successfully obtained OAuth2 access token" -ForegroundColor Green

# ----------------- CHARACTER REFERENCE -----------------
# Kelly's canonical appearance - ALWAYS include in prompts featuring Kelly

$kellyCharacterBase = @"
Kelly Rein, photorealistic digital human, modern timeless "Apple Genius" aesthetic. 
Oval face shape with soft, rounded contours (NOT angular, NOT square, NOT sharp jawline, NOT angular cheekbones), smooth rounded jawline, soft cheek contours, gentle facial curves, soft rounded chin, no sharp angles or hard edges on face, clear smooth complexion with natural glow, warm light-medium skin tone, healthy radiant skin. 
Warm brown almond-shaped eyes, bright and engaging, well-defined dark brown eyebrows with natural arch, long dark eyelashes. 
Medium brown hair with subtle caramel/honey-blonde highlights, soft waves that fall together cohesively in unified sections (hair strands stay together, NOT frizzy, NOT whispy, NOT curly), smooth polished texture, parted slightly off-center or down the middle, long hair that extends well past shoulders, cascades down to mid-back or lower chest area, hair reaches at least halfway down the upper back when standing, clearly visible hair length extending well beyond shoulder line, long flowing hair, rich and voluminous. Hair moves as cohesive wave sections - no separate curls, no frizz, no flyaways. NOT short hair, NOT shoulder-length hair, NOT bob cut, NOT chin-length hair.
Full lips with natural rosy-pink color, genuine warm smile showing straight white teeth, natural smile lines (nasolabial folds), slight crinkles at outer corners of eyes when smiling. 
Late 20s to early 30s, athletic build, strong capable presence, approachable and professional demeanor.
"@

$kellyReinmakerWardrobe = @"
Wearing Reinmaker armor: dark gray ribbed turtleneck base layer, 
form-fitting dark charcoal-gray tactical garment with structured seams and panels, 
metallic dark steel-colored shoulder pauldrons (multi-layered, riveted, curved protective design), 
wide dark gray fabric sash draped diagonally from left shoulder to right hip secured by dark metallic straps, 
wide dark metallic horizontal strap across chest, multiple dark utilitarian belts around waist with rectangular metallic buckle, 
long form-fitting sleeves with fingerless glove-like covering on left hand, textured wrapped detailing on right forearm, 
dark gray tactical pants matching upper garment. 
Color palette: dark grays, charcoal, metallic steel, dark browns. NO bright colors, NO reds, NO yellows, NO light browns, NO Roman/ancient elements.
"@

$kellyDailyLessonWardrobe = @"
Wearing modern professional attire: light blue ribbed knit sweater with crew neck, soft muted blue color, clean contemporary wardrobe. 
Seated in classic director's chair with dark brown wooden frame and black canvas seat/backrest, visible armrests. 
Clean bright white or very light gray studio background, plain uncluttered environment, soft even studio lighting with subtle shadows, professional photography setup. 
Same facial features and hair as Reinmaker variant - warm brown eyes, medium brown hair with caramel highlights, oval face, genuine smile.
"@

$mandatoryNegativePrompts = @"
cartoon, stylized, anime, illustration, drawing, sketch, fantasy, medieval, Roman, ancient, historical, 
exaggerated features, unrealistic proportions, memes, internet humor, casual style, 
second person, extra people, multiple faces, bright colors, red, yellow, orange, 
light browns, tan, beige, leather straps, Roman armor, ornate decorations, jewelry, 
frizzy hair, whispy hair, separate curls, tight curls, curly hair, flyaways, unkempt hair, messy hair, separate hair strands, hair strands flying apart,
angular face, square face, sharp jawline, angular cheekbones, strong angular jaw, angular features, hard facial edges, sharp chin, short hair, shoulder-length hair, bob cut, pixie cut, short bob, chin-length hair, hair above shoulders,
low quality, blurry, pixelated, compression artifacts, oversaturated colors, 
unrealistic lighting, watermark, text overlay, logo, CGI, 3D render, game asset, sprite
"@

# Function to build a character-consistent prompt
function Build-KellyPrompt {
    param (
        [string]$SceneDescription,
        [string]$WardrobeVariant = "Reinmaker",  # "Reinmaker" or "DailyLesson"
        [string]$Pose = "",
        [string]$Lighting = "professional photography quality lighting, soft key light at 45 degrees with subtle fill, realistic shadows and highlights",
        [string]$AdditionalNegatives = "",
        [array]$ReferenceImages = @()
    )
    
    $wardrobe = if ($WardrobeVariant -eq "DailyLesson") { $kellyDailyLessonWardrobe } else { $kellyReinmakerWardrobe }
    
    # CRITICAL: If reference images are provided, use MINIMAL text and let images do the work
    # Reference images are PRIMARY for character likeness - text is secondary
    if ($ReferenceImages.Count -gt 0) {
        # With reference images: Use minimal prompt focused on scene/wardrobe
        # Let reference images handle ALL character likeness (face, hair, features)
        $prompt = "$SceneDescription, featuring Kelly Rein, $wardrobe"
        Write-Host "Using reference images for character likeness (minimal text prompt)" -ForegroundColor Green
    } else {
        # Without reference images: Fall back to detailed text description
        $prompt = "$SceneDescription, featuring $kellyCharacterBase, $wardrobe"
        Write-Host "No reference images - using detailed text description for character likeness" -ForegroundColor Yellow
    }
    
    if ($Pose) {
        $prompt += ", $Pose"
    }
    
    $prompt += ", $Lighting, photorealistic digital human, modern timeless aesthetic, professional photography quality, high detail, realistic skin textures, realistic fabric textures, realistic metallic surfaces."
    
    $negative = $mandatoryNegativePrompts
    if ($AdditionalNegatives) {
        $negative += ", $AdditionalNegatives"
    }
    
    return @{
        Prompt = $prompt
        Negative = $negative
    }
}

# ----------------- REFERENCE IMAGE HANDLING -----------------

# Function to get CHARACTER reference images (face, hair, skin tone - NOT wardrobe)
function Get-CharacterReferences {
    param (
        [string]$ReferenceDir = $referenceImagePath
    )
    
    $characterRefs = @()
    $supportedFormats = @(".png", ".jpg", ".jpeg")
    
    if (-not (Test-Path $ReferenceDir)) {
        Write-Host "Reference image directory not found: $ReferenceDir" -ForegroundColor Yellow
        return $characterRefs
    }
    
    Write-Host "Scanning for CHARACTER reference images (face, hair, skin tone)..." -ForegroundColor Cyan
    
    # PRIMARY CHARACTER REFERENCES (face, hair, skin tone - use for ALL variants)
    $primaryCharacterRefs = @(
        "headshot2-kelly-base169 101225.png",  # Primary Headshot 2 reference
        "kelly_directors_chair_8k_light (2).png",  # 8K quality character reference
        "kelly square.jpg",  # Square format headshot
        "Kelly Source.jpeg",  # Original source photo
        "cd3a3ce0-45f4-40bc-b941-4b0b13ba1cc1.png"  # Additional character reference
    )
    
    # SECONDARY CHARACTER REFERENCES (multiple angles)
    $secondaryCharacterRefs = @("3.jpeg", "3 (1).jpeg", "8.png", "9.png", "12.png", "24.png", "32.png")
    
    # Load primary character references first
    foreach ($refName in $primaryCharacterRefs) {
        $fullPath = Join-Path $ReferenceDir $refName
        if (Test-Path $fullPath) {
            Write-Host "Found PRIMARY character reference: $refName" -ForegroundColor Green
            try {
                $imageBytes = [System.IO.File]::ReadAllBytes($fullPath)
                $base64String = [System.Convert]::ToBase64String($imageBytes)
                $characterRefs += @{
                    Path = $fullPath
                    Name = $refName
                    Base64 = $base64String
                    MimeType = if ($refName -match "\.png$") { "image/png" } else { "image/jpeg" }
                    Type = "character"
                    Priority = "primary"
                }
            } catch {
                Write-Host "ERROR: Failed to read $refName - $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
    
    # Load secondary character references (multi-angle)
    foreach ($refName in $secondaryCharacterRefs) {
        $fullPath = Join-Path $ReferenceDir $refName
        if (Test-Path $fullPath) {
            Write-Host "Found SECONDARY character reference: $refName" -ForegroundColor Gray
            try {
                $imageBytes = [System.IO.File]::ReadAllBytes($fullPath)
                $base64String = [System.Convert]::ToBase64String($imageBytes)
                $characterRefs += @{
                    Path = $fullPath
                    Name = $refName
                    Base64 = $base64String
                    MimeType = if ($refName -match "\.png$") { "image/png" } else { "image/jpeg" }
                    Type = "character"
                    Priority = "secondary"
                }
            } catch {
                Write-Host "ERROR: Failed to read $refName - $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
    
    if ($characterRefs.Count -eq 0) {
        Write-Host "No character reference images found. Character consistency will rely on enhanced prompts only." -ForegroundColor Yellow
    } else {
        Write-Host "Loaded $($characterRefs.Count) CHARACTER reference image(s) for face/hair/skin consistency" -ForegroundColor Green
    }
    
    return $characterRefs
}

# Function to get WARDROBE reference images (clothing/outfit only)
function Get-WardrobeReferences {
    param (
        [string]$ReferenceDir = $referenceImagePath,
        [string]$Variant = "Reinmaker"
    )
    
    $wardrobeRefs = @()
    
    if (-not (Test-Path $ReferenceDir)) {
        return $wardrobeRefs
    }
    
    Write-Host "Scanning for WARDROBE reference images (clothing/outfit)..." -ForegroundColor Cyan
    
    if ($Variant -eq "Reinmaker") {
        $wardrobeFile = "reinmaker kelly outfit base.png"
        $fullPath = Join-Path $ReferenceDir $wardrobeFile
        if (Test-Path $fullPath) {
            Write-Host "Found Reinmaker wardrobe reference: $wardrobeFile" -ForegroundColor Green
            try {
                $imageBytes = [System.IO.File]::ReadAllBytes($fullPath)
                $base64String = [System.Convert]::ToBase64String($imageBytes)
                $wardrobeRefs += @{
                    Path = $fullPath
                    Name = $wardrobeFile
                    Base64 = $base64String
                    MimeType = "image/png"
                    Type = "wardrobe"
                    Variant = "Reinmaker"
                }
            } catch {
                Write-Host "ERROR: Failed to read $wardrobeFile - $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
    
    # Daily Lesson variant can use directors_chair image for wardrobe context
    # but we'll use character references for the actual character appearance
    
    return $wardrobeRefs
}

# Legacy function for backward compatibility (now uses character references)
function Get-ReferenceImages {
    param (
        [string]$ReferenceDir = $referenceImagePath
    )
    # Return character references only (not wardrobe)
    return Get-CharacterReferences -ReferenceDir $ReferenceDir
}

# ----------------- SCRIPT LOGIC -----------------

# A function to call the Google AI image generation API
function Generate-Google-Asset {
    param (
        [string]$AssetName,
        [string]$Prompt,
        [string]$NegativePrompt = "",
        [array]$ReferenceImages = @(),
        [int]$MasterWidth,
        [int]$MasterHeight,
        [string]$OutputFile
    )

    Write-Host "Preparing to generate asset: $AssetName..." -ForegroundColor Cyan

    # Map dimensions to closest supported aspect ratio
    # Supported: 1:1, 9:16, 16:9, 4:3, 3:4
    $ratio = $MasterWidth / $MasterHeight
    $aspectRatio = if ($ratio -gt 1.5) { "16:9" }      # Wide landscape
                  elseif ($ratio -gt 1.2) { "4:3" }     # Landscape
                  elseif ($ratio -gt 0.8) { "1:1" }    # Square
                  elseif ($ratio -gt 0.7) { "3:4" }     # Portrait
                  else { "9:16" }                       # Tall portrait
    
    Write-Host "Using aspect ratio: $aspectRatio (requested: $($MasterWidth)x$($MasterHeight))" -ForegroundColor Gray
    
    if ($ReferenceImages.Count -gt 0) {
        Write-Host "Google AI Studio API: Reference images detected but may not be supported. Will attempt to include." -ForegroundColor Yellow
        Write-Host "Note: If this fails, use Vertex AI API which has full reference image support." -ForegroundColor Yellow
    }
    
    $payload = @{
        "instances" = @(
            @{
                "prompt" = $Prompt
            }
        )
        "parameters" = @{
            "sampleCount" = 1
            "aspectRatio" = $aspectRatio
        }
    }
    
    # Note: Imagen 3.0 REST API doesn't support negativePrompt parameter
    # Instead, we'll append negative prompts to the main prompt text
    if ($NegativePrompt) {
        $fullPrompt = "$Prompt. Avoid: $NegativePrompt"
    } else {
        $fullPrompt = $Prompt
    }
    
    $payload.instances[0].prompt = $fullPrompt
    
    # Attempt to add reference images if provided (may not be supported)
    if ($ReferenceImages.Count -gt 0) {
        $referenceImageArray = @()
        foreach ($refImg in $ReferenceImages) {
            $referenceImageArray += @{
                "bytesBase64Encoded" = $refImg.Base64
                "mimeType" = $refImg.MimeType
            }
        }
        try {
            $payload.instances[0]["referenceImages"] = $referenceImageArray
            Write-Host "Attempting to include reference images (may fail if not supported)" -ForegroundColor Gray
        } catch {
            Write-Host "Warning: Could not add reference images to payload: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
    
    $payload = $payload | ConvertTo-Json -Depth 5

    $tempPayloadFile = "payload_$(Get-Random).json"
    $payload | Out-File -FilePath $tempPayloadFile -Encoding utf8

    $responseFile = "response_$(Get-Random).json"
    
    Write-Host "Calling API..." -ForegroundColor Yellow
    # Use OAuth2 access token for Vertex AI endpoint
    curl.exe -X POST -H "Authorization: Bearer $accessToken" -H "Content-Type: application/json" -d "@$tempPayloadFile" "$vertexAIEndpoint" -o $responseFile
    
    # Check for errors
    $response = Get-Content -Raw -Path $responseFile | ConvertFrom-Json
    
    if ($response.error) {
        Write-Host "ERROR: $($response.error.message)" -ForegroundColor Red
        if ($response.error.code -eq 429) {
            $retryDelay = 60
            if ($response.error.details) {
                $retryInfo = $response.error.details | Where-Object { $_.'@type' -eq 'type.googleapis.com/google.rpc.RetryInfo' }
                if ($retryInfo -and $retryInfo.retryDelay) {
                    $retryDelay = [math]::Ceiling([double]$retryInfo.retryDelay.Replace('s', ''))
                }
            }
            Write-Host "Quota exceeded. Waiting $retryDelay seconds before retry..." -ForegroundColor Yellow
            Start-Sleep -Seconds $retryDelay
            
            # Retry once
            Write-Host "Retrying..." -ForegroundColor Yellow
            curl.exe -X POST -H "Authorization: Bearer $accessToken" -H "Content-Type: application/json" -d "@$tempPayloadFile" "$vertexAIEndpoint" -o $responseFile
            $response = Get-Content -Raw -Path $responseFile | ConvertFrom-Json
            
            if ($response.error) {
                Write-Host "ERROR: Still hitting quota limits after retry. Please upgrade your plan or wait longer." -ForegroundColor Red
                Remove-Item $tempPayloadFile -ErrorAction SilentlyContinue
                Remove-Item $responseFile -ErrorAction SilentlyContinue
                return
            }
        } else {
            Remove-Item $tempPayloadFile -ErrorAction SilentlyContinue
            Remove-Item $responseFile -ErrorAction SilentlyContinue
            return
        }
    }
    
    # Extract image data from response
    # The structure is: response.predictions[0].bytesBase64Encoded
    if ($response.predictions -and $response.predictions[0].bytesBase64Encoded) {
        $base64String = $response.predictions[0].bytesBase64Encoded
        
        Write-Host "Found image data. Decoding..." -ForegroundColor Green
        
        # Decode and save
        $imageBytes = [System.Convert]::FromBase64String($base64String)
        
        # Ensure output directory exists
        $outputDir = Split-Path -Parent $OutputFile
        if (-not (Test-Path $outputDir)) {
            New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
        }
        
        [System.IO.File]::WriteAllBytes($OutputFile, $imageBytes)
        
        Write-Host "Successfully generated: $OutputFile" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Could not find image data in response." -ForegroundColor Red
        Write-Host "Response structure:" -ForegroundColor Yellow
        $response | ConvertTo-Json -Depth 10 | Write-Host
    }
    
    # Cleanup
    Remove-Item $tempPayloadFile -ErrorAction SilentlyContinue
    Remove-Item $responseFile -ErrorAction SilentlyContinue
}

# Helper: invoke Python-based reference image generator (uses Vertex AI SDK)
function Invoke-PythonReferenceGeneration {
    param (
        [string]$Prompt,
        [string]$NegativePrompt,
        [array]$ReferenceImages,
        [string]$AspectRatio,
        [int]$MasterWidth,
        [int]$MasterHeight,
        [string]$OutputFile,
        [string]$ProjectId,
        [string]$Location,
        [string]$ModelName
    )

    if (-not $script:PythonHelperAvailable) {
        return $false
    }

    if (-not (Test-Path $script:PythonHelperPath)) {
        return $false
    }

    $argsList = @()
    if ($script:PythonPrefixArgs -and $script:PythonPrefixArgs.Count -gt 0) {
        $argsList += $script:PythonPrefixArgs
    }
    $argsList += $script:PythonHelperPath
    $argsList += @("--prompt", $Prompt, "--output", $OutputFile, "--aspect-ratio", $AspectRatio, "--model", $ModelName, "--location", $Location)

    if ($ProjectId) {
        $argsList += @("--project", $ProjectId)
    }
    if ($NegativePrompt) {
        $argsList += @("--negative-prompt", $NegativePrompt)
    }
    if ($MasterWidth -gt 0 -and $MasterHeight -gt 0) {
        $argsList += @("--width", $MasterWidth.ToString(), "--height", $MasterHeight.ToString())
    }

    foreach ($ref in $ReferenceImages) {
        if ($ref.Path -and (Test-Path $ref.Path)) {
            $argsList += @("--reference", $ref.Path)
        }
    }

    if ($argsList -notcontains "--reference") {
        # No usable reference paths available
        return $false
    }

    Write-Host "Attempting reference-controlled generation via Python helper..." -ForegroundColor Cyan
    try {
        $output = & $script:PythonExecutable @argsList 2>&1
        $exitCode = $LASTEXITCODE
    } catch {
        Write-Host "Python helper execution failed: $($_.Exception.Message)" -ForegroundColor Yellow
        return $false
    }

    if ($exitCode -eq 0 -and (Test-Path $OutputFile)) {
        if ($output) {
            Write-Host $output -ForegroundColor Gray
        }
        Write-Host "Python helper succeeded. Output: $OutputFile" -ForegroundColor Green
        return $true
    }

    if ($output) {
        Write-Host $output -ForegroundColor Yellow
    }
    Write-Host "Python helper failed (exit code $exitCode). Falling back to REST API." -ForegroundColor Yellow
    return $false
}

# Function to call Vertex AI image generation API with reference image support
function Generate-VertexAI-Asset {
    param (
        [string]$AssetName,
        [string]$Prompt,
        [string]$NegativePrompt = "",
        [array]$ReferenceImages = @(),
        [int]$MasterWidth,
        [int]$MasterHeight,
        [string]$OutputFile,
        [string]$ProjectId = $vertexAIProjectId,
        [string]$Location = $vertexAILocation
    )

    Write-Host "Preparing to generate asset via Vertex AI: $AssetName..." -ForegroundColor Cyan
    
    # Map dimensions to closest supported aspect ratio
    # Supported: 1:1, 9:16, 16:9, 4:3, 3:4
    $ratio = $MasterWidth / $MasterHeight
    $aspectRatio = if ($ratio -gt 1.5) { "16:9" }      # Wide landscape
                  elseif ($ratio -gt 1.2) { "4:3" }     # Landscape
                  elseif ($ratio -gt 0.8) { "1:1" }    # Square
                  elseif ($ratio -gt 0.7) { "3:4" }     # Portrait
                  else { "9:16" }                       # Tall portrait
    
    Write-Host "Using aspect ratio: $aspectRatio (requested: $($MasterWidth)x$($MasterHeight))" -ForegroundColor Gray

    $originalPrompt = $Prompt
    $promptRefs = ""
    if ($ReferenceImages.Count -gt 0) {
        Write-Host "Including $($ReferenceImages.Count) reference image(s) for character consistency" -ForegroundColor Green
        for ($i = 1; $i -le $ReferenceImages.Count; $i++) {
            $promptRefs += "[$i]"
        }
    }

    $promptWithReferences = if ([string]::IsNullOrWhiteSpace($promptRefs)) { $originalPrompt } else { "$originalPrompt $promptRefs" }

    $pythonGenerated = $false
    if ($ReferenceImages.Count -gt 0) {
        $pythonGenerated = Invoke-PythonReferenceGeneration -Prompt $promptWithReferences -NegativePrompt $NegativePrompt -ReferenceImages $ReferenceImages -AspectRatio $aspectRatio -MasterWidth $MasterWidth -MasterHeight $MasterHeight -OutputFile $OutputFile -ProjectId $ProjectId -Location $Location -ModelName $modelName
        if ($pythonGenerated) {
            return
        }
    }
    
    # Build Vertex AI API payload
    $endpoint = "https://aiplatform.googleapis.com/v1/projects/$ProjectId/locations/$Location/publishers/google/models/$modelName`:predict"
    
    $payload = @{
        "instances" = @(
            @{
                "prompt" = $Prompt
            }
        )
        "parameters" = @{
            "sampleCount" = 1
            "aspectRatio" = $aspectRatio
        }
    }
    
    # Add negative prompt if provided
    if ($NegativePrompt) {
        $payload.parameters["negativePrompt"] = $NegativePrompt
    }
    
    # Add reference images if provided
    if ($ReferenceImages.Count -gt 0) {
        $referenceImageArray = @()
        $referenceId = 1
        
        foreach ($refImg in $ReferenceImages) {
            # REFERENCE IMAGE FORMAT (Expected structure - currently not supported by REST API /predict endpoint)
            # NOTE: Python SDK works with VertexImage.load_from_file() - use Python SDK for reference images
            # This format is preserved for future API support or documentation updates
            $referenceImageArray += @{
                "referenceType" = "REFERENCE_TYPE_SUBJECT"
                "referenceId" = $referenceId
                "referenceImage" = @{
                    "bytesBase64Encoded" = $refImg.Base64
                    "mimeType" = $refImg.MimeType
                }
                "subjectImageConfig" = @{
                    "subjectDescription" = "Kelly Rein, photorealistic digital human, oval face with soft rounded contours, long hair extending well past shoulders"
                    "subjectType" = "SUBJECT_TYPE_PERSON"
                }
            }
            $referenceId++
        }
        
        # Add reference images to instance (not parameters)
        $payload.instances[0]["referenceImages"] = $referenceImageArray
        $payload.instances[0].prompt = $promptWithReferences
        
        Write-Host "Added $($referenceImageArray.Count) reference image(s) - NOTE: Reference images may not be supported by REST API /predict endpoint" -ForegroundColor Yellow
        Write-Host "If generation fails, reference images will be skipped and detailed text prompts will be used instead" -ForegroundColor Yellow
    }
    else {
        $payload.instances[0].prompt = $originalPrompt
    }
    
    $payloadParameters = $payload.parameters
    $payload = $payload | ConvertTo-Json -Depth 10
    
    $tempPayloadFile = "payload_vertex_$(Get-Random).json"
    $payload | Out-File -FilePath $tempPayloadFile -Encoding utf8
    
    $responseFile = "response_vertex_$(Get-Random).json"
    
    Write-Host "Calling Vertex AI API..." -ForegroundColor Yellow
    
    # Vertex AI uses OAuth2 Bearer token authentication
    curl.exe -X POST `
        -H "Authorization: Bearer $accessToken" `
        -H "Content-Type: application/json" `
        -d "@$tempPayloadFile" `
        "$endpoint" `
        -o $responseFile
    
    # Check for errors
    $responseContent = Get-Content -Raw -Path $responseFile -ErrorAction SilentlyContinue
    if (-not $responseContent) {
        Write-Host "ERROR: Empty response from Vertex AI API" -ForegroundColor Red
        Remove-Item $tempPayloadFile -ErrorAction SilentlyContinue
        Remove-Item $responseFile -ErrorAction SilentlyContinue
        return
    }
    
    try {
        $response = $responseContent | ConvertFrom-Json
    } catch {
        Write-Host "ERROR: Failed to parse response JSON" -ForegroundColor Red
        Write-Host "Response content: $responseContent" -ForegroundColor Yellow
        Remove-Item $tempPayloadFile -ErrorAction SilentlyContinue
        Remove-Item $responseFile -ErrorAction SilentlyContinue
        return
    }
    
    if ($response.error) {
        $errorMsg = $response.error.message
        Write-Host "ERROR: $errorMsg" -ForegroundColor Red
        
        # If error is about reference images, retry without them
        if ($ReferenceImages.Count -gt 0 -and ($errorMsg -match "uri|raw bytes|image bytes|media content")) {
            Write-Host "Reference images not supported by REST API endpoint. Retrying without reference images..." -ForegroundColor Yellow
            
            # Remove reference images and retry
            $payloadNoRefs = @{
                "instances" = @(
                    @{
                        "prompt" = $originalPrompt
                    }
                )
                "parameters" = $payloadParameters
            }
            
            $payloadNoRefsJson = $payloadNoRefs | ConvertTo-Json -Depth 10
            $tempPayloadFileNoRefs = "payload_vertex_$(Get-Random).json"
            $payloadNoRefsJson | Out-File -FilePath $tempPayloadFileNoRefs -Encoding utf8
            
            $responseFileNoRefs = "response_vertex_$(Get-Random).json"
            
            curl.exe -X POST `
                -H "Authorization: Bearer $accessToken" `
                -H "Content-Type: application/json" `
                -d "@$tempPayloadFileNoRefs" `
                "$endpoint" `
                -o $responseFileNoRefs | Out-Null
            
            $responseContentNoRefs = Get-Content -Raw -Path $responseFileNoRefs -ErrorAction SilentlyContinue
            if ($responseContentNoRefs) {
                try {
                    $response = $responseContentNoRefs | ConvertFrom-Json
                    if (-not $response.error) {
                        Write-Host "Successfully generated without reference images (using detailed text prompts)" -ForegroundColor Green
                        # Continue with normal processing below - update file paths
                        $responseFile = $responseFileNoRefs
                        $tempPayloadFile = $tempPayloadFileNoRefs
                    } else {
                        Write-Host "ERROR: Still failing after removing reference images: $($response.error.message)" -ForegroundColor Red
                        Remove-Item $tempPayloadFileNoRefs -ErrorAction SilentlyContinue
                        Remove-Item $responseFileNoRefs -ErrorAction SilentlyContinue
                        Remove-Item $tempPayloadFile -ErrorAction SilentlyContinue
                        Remove-Item $responseFile -ErrorAction SilentlyContinue
                        return
                    }
                } catch {
                    Write-Host "ERROR: Failed to parse retry response" -ForegroundColor Red
                    Remove-Item $tempPayloadFileNoRefs -ErrorAction SilentlyContinue
                    Remove-Item $responseFileNoRefs -ErrorAction SilentlyContinue
                    Remove-Item $tempPayloadFile -ErrorAction SilentlyContinue
                    Remove-Item $responseFile -ErrorAction SilentlyContinue
                    return
                }
            } else {
                Write-Host "ERROR: Empty response from retry" -ForegroundColor Red
                Remove-Item $tempPayloadFile -ErrorAction SilentlyContinue
                Remove-Item $responseFile -ErrorAction SilentlyContinue
                return
            }
        } else {
            # Non-reference-image error - just return
            if ($response.error.details) {
                Write-Host "Details: $($response.error.details | ConvertTo-Json)" -ForegroundColor Yellow
            }
            Remove-Item $tempPayloadFile -ErrorAction SilentlyContinue
            Remove-Item $responseFile -ErrorAction SilentlyContinue
            return
        }
    }
    
    # Extract image data from response
    # Vertex AI response structure: response.predictions[0].bytesBase64Encoded
    if ($response.predictions -and $response.predictions[0].bytesBase64Encoded) {
        $base64String = $response.predictions[0].bytesBase64Encoded
        
        Write-Host "Found image data. Decoding..." -ForegroundColor Green
        
        # Decode and save
        $imageBytes = [System.Convert]::FromBase64String($base64String)
        
        # Ensure output directory exists
        $outputDir = Split-Path -Parent $OutputFile
        if (-not (Test-Path $outputDir)) {
            New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
        }
        
        [System.IO.File]::WriteAllBytes($OutputFile, $imageBytes)
        
        Write-Host "Successfully generated via Vertex AI: $OutputFile" -ForegroundColor Green
    } elseif ($response.predictions -and $response.predictions[0].mimeType -and $response.predictions[0].bytesBase64Encoded) {
        # Alternative response structure
        $base64String = $response.predictions[0].bytesBase64Encoded
        
        Write-Host "Found image data (alternative format). Decoding..." -ForegroundColor Green
        
        $imageBytes = [System.Convert]::FromBase64String($base64String)
        
        $outputDir = Split-Path -Parent $OutputFile
        if (-not (Test-Path $outputDir)) {
            New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
        }
        
        [System.IO.File]::WriteAllBytes($OutputFile, $imageBytes)
        
        Write-Host "Successfully generated via Vertex AI: $OutputFile" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Could not find image data in Vertex AI response." -ForegroundColor Red
        Write-Host "Response structure:" -ForegroundColor Yellow
        $response | ConvertTo-Json -Depth 10 | Write-Host
    }
    
    # Cleanup
    Remove-Item $tempPayloadFile -ErrorAction SilentlyContinue
    Remove-Item $responseFile -ErrorAction SilentlyContinue
}

# Unified function to generate assets with reference images using either API
function Generate-Asset-With-Reference {
    param (
        [string]$AssetName,
        [string]$Prompt,
        [string]$NegativePrompt = "",
        [int]$MasterWidth,
        [int]$MasterHeight,
        [string]$OutputFile,
        [bool]$UseVertexAI = $false,
        [array]$ReferenceImages = @()
    )
    
    Write-Host "`n========== Generating Asset: $AssetName ==========" -ForegroundColor Cyan
    Write-Host "API: $(if ($UseVertexAI) { 'Vertex AI' } else { 'Google AI Studio' })" -ForegroundColor Cyan
    Write-Host "Reference Images: $($ReferenceImages.Count)" -ForegroundColor Cyan
    Write-Host "Output: $OutputFile" -ForegroundColor Cyan
    Write-Host "="*60 -ForegroundColor Cyan
    
    # Both APIs now use Vertex AI endpoint with OAuth2 token
    # UseVertexAI parameter still useful for logging/clarity
    if ($UseVertexAI) {
        Generate-VertexAI-Asset `
            -AssetName $AssetName `
            -Prompt $Prompt `
            -NegativePrompt $NegativePrompt `
            -ReferenceImages $ReferenceImages `
            -MasterWidth $MasterWidth `
            -MasterHeight $MasterHeight `
            -OutputFile $OutputFile
    } else {
        # Use Vertex AI endpoint for Google AI Studio as well (same endpoint)
        Generate-VertexAI-Asset `
            -AssetName $AssetName `
            -Prompt $Prompt `
            -NegativePrompt $NegativePrompt `
            -ReferenceImages $ReferenceImages `
            -MasterWidth $MasterWidth `
            -MasterHeight $MasterHeight `
            -OutputFile $OutputFile
    }
    
    Write-Host "="*60 -ForegroundColor Cyan
    Write-Host ""
}

# ----------------- ASSET GENERATION CALLS -----------------
# Uncomment to generate assets once quota is available

# --- A. Core Gameplay Sprites ---
$playerPrompt = Build-KellyPrompt -SceneDescription "Full-body photorealistic view of Kelly in neutral running pose, facing right, maintaining complete character consistency" -WardrobeVariant "Reinmaker" -Pose "running pose, dynamic movement, readable silhouette, action-ready stance" -Lighting "orthographic camera view, soft forge key light from lower-left, cool rim light upper-right, transparent background, game-ready asset format" -AdditionalNegatives "stylized, cel-shaded, painterly texture, pixel art, low resolution, game sprite aesthetic, non-photorealistic, cartoon rendering"
Generate-Google-Asset -AssetName "A1. Player: Kelly (Runner)" -Prompt $playerPrompt.Prompt -NegativePrompt $playerPrompt.Negative -MasterWidth 1024 -MasterHeight 1280 -OutputFile "assets/player.png"
Generate-Google-Asset -AssetName "A2. Obstacle: Knowledge Shard" -Prompt "Stylized game obstacle: vertical circuit-rein shard totem, forged brass and steel with a glowing teal inlay, simple sharp geometry, clean edges, orthographic, cel-shaded, painterly texture, warm forge key light, cool rim light, transparent background, 512px tall master, no base shadow, no background. Negative: complex, detailed, rounded, photo." -MasterWidth 512 -MasterHeight 512 -OutputFile "assets/obstacle.png"

# --- B. Background & Environment ---
Generate-Google-Asset -AssetName "B1. Parallax Skyline" -Prompt "Tiling horizontal skyline for a side-scrolling runner game, silhouettes of a mythic academy / Hall of the Seven Tribes in the far distance, faint vertical banners in seven colors (red, blue, green, yellow, purple, orange, white), subtle painterly clouds, dusk palette matching a #0B1020 night steel sky with ember-colored accents on the horizon, seamless horizontally, 1024x256px, soft parallax readability, no foreground details, no text." -MasterWidth 1024 -MasterHeight 256 -OutputFile "assets/bg.png"

# --- C. UI & Meta ---
Generate-Google-Asset -AssetName "C1. Logo / Title Card" -Prompt "Key art logo cinematic title card, text reads 'The Rein Maker's Daughter', elegant serif combined with a clean geometric sans-serif font, features a central emblem of a broken leather rein seamlessly reshaping into a glowing circuit loop, brass and ember (#D8A24A) accents on the emblem, deep blue steel (#495057) background vignette, cinematic composition, crisp, high contrast, transparent background, export at 1280x720." -MasterWidth 1280 -MasterHeight 720 -OutputFile "marketing/cover-1280x720.png"
Generate-Google-Asset -AssetName "C2. Favicon" -Prompt "Icon only: a glowing circuit-rein emblem, simple 2-tone (teal #0BB39C and graphite #1B1E22), centered, flat background, clean and readable silhouette, 256x256px, no background." -MasterWidth 256 -MasterHeight 256 -OutputFile "assets/favicon.png"

# --- D. Lore Collectibles (Stones) ---
Generate-Google-Asset -AssetName "D1. Stone: Light" -Prompt "Stylized gem icon, for the Light Tribe, color #F2F7FA, subtle etched symbol of an eye, soft inner glow, clean silhouette, 64x64, transparent background." -MasterWidth 64 -MasterHeight 64 -OutputFile "assets/stones/stone_light.png"
Generate-Google-Asset -AssetName "D1. Stone: Stone" -Prompt "Stylized gem icon, for the Stone Tribe, color #8E9BA7, subtle etched symbol of a mountain, soft inner glow, clean silhouette, 64x64, transparent background." -MasterWidth 64 -MasterHeight 64 -OutputFile "assets/stones/stone_stone.png"
Generate-Google-Asset -AssetName "D1. Stone: Metal" -Prompt "Stylized gem icon, for the Metal Tribe, color #adb5bd, subtle etched symbol of a gear, soft inner glow, clean silhouette, 64x64, transparent background." -MasterWidth 64 -MasterHeight 64 -OutputFile "assets/stones/stone_metal.png"
Generate-Google-Asset -AssetName "D1. Stone: Code" -Prompt "Stylized gem icon, for the Code Tribe, color #0BB39C, subtle etched symbol of code brackets '<>', soft inner glow, clean silhouette, 64x64, transparent background." -MasterWidth 64 -MasterHeight 64 -OutputFile "assets/stones/stone_code.png"
Generate-Google-Asset -AssetName "D1. Stone: Air" -Prompt "Stylized gem icon, for the Air Tribe, color #aed9e0, subtle etched symbol of a feather, soft inner glow, clean silhouette, 64x64, transparent background." -MasterWidth 64 -MasterHeight 64 -OutputFile "assets/stones/stone_air.png"
Generate-Google-Asset -AssetName "D1. Stone: Water" -Prompt "Stylized gem icon, for the Water Tribe, color #4dabf7, subtle etched symbol of a wave, soft inner glow, clean silhouette, 64x64, transparent background." -MasterWidth 64 -MasterHeight 64 -OutputFile "assets/stones/stone_water.png"
Generate-Google-Asset -AssetName "D1. Stone: Fire" -Prompt "Stylized gem icon, for the Fire Tribe, color #F25F5C, subtle etched symbol of a flame, soft inner glow, clean silhouette, 64x64, transparent background." -MasterWidth 64 -MasterHeight 64 -OutputFile "assets/stones/stone_fire.png"

# --- E. Narrative Inserts ---
$splashPrompt = Build-KellyPrompt -SceneDescription "Cinematic splash art: Kelly stepping out from a grand dark forge into the light, on the dark side old leather reins hang like chains, on the light side the air is clean, she carries a small glowing circuit-rein token that illuminates her determined face, high contrast between shadow and light, dusk palette, the glowing token is the focal point" -WardrobeVariant "Reinmaker" -Pose "stepping forward, determined expression, holding glowing circuit-rein token" -Lighting "cinematic lighting, high contrast between dark forge interior and bright exterior, token provides warm key light on face" -AdditionalNegatives "painterly, vector style, stylized, illustration, non-photorealistic"
Generate-Google-Asset -AssetName "E1. Opening Splash" -Prompt $splashPrompt.Prompt -NegativePrompt $splashPrompt.Negative -MasterWidth 1280 -MasterHeight 720 -OutputFile "marketing/splash_intro.png"
Generate-Google-Asset -AssetName "E2. Game Over Panel" -Prompt "Soft, dark, low-contrast vignette background panel, suggesting shattered rein fragments slowly reforming into a whole circuit loop in the center, 960x540px, for background usage behind text. The imagery should be very subtle and not distracting." -MasterWidth 960 -MasterHeight 540 -OutputFile "assets/gameover_bg.png"

# --- F. Marketing ---
$bannerPrompt = Build-KellyPrompt -SceneDescription "Wide marketing banner montage: on the right, photorealistic Kelly in full running pose; on the left, the game's bold logo ('The Rein Maker's Daughter'); in the background is the parallax skyline of the Hall of the Seven Tribes at dusk" -WardrobeVariant "Reinmaker" -Pose "full running pose, dynamic action, facing right" -Lighting "dynamic composition lighting, dusk palette, cinematic banner aesthetic" -AdditionalNegatives "stylized sprite, illustration, non-photorealistic character, cartoon, game asset style"
Generate-Google-Asset -AssetName "F1. Itch.io Banner" -Prompt $bannerPrompt.Prompt -NegativePrompt $bannerPrompt.Negative -MasterWidth 1920 -MasterHeight 480 -OutputFile "marketing/itch-banner-1920x480.png"

Write-Host "`nScript execution complete." -ForegroundColor Cyan
Write-Host "NOTE: If you see quota errors, you may need to wait for quota reset or upgrade your Google AI Studio plan." -ForegroundColor Yellow
