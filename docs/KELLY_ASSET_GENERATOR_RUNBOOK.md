## Kelly Asset Generator Runbook

Purpose: One‑click generation of Kelly assets from YAML presets, optional Google Vertex Imagen backend, 8K outputs, and reproducible manifests.

### Prerequisites
- Python 3.9+
- Install deps:
  ```bash
  pip install -r requirements.txt
  ```

### Optional: Enable Google Vertex Imagen
If environment variables are set, the generator calls Imagen 3; otherwise it produces a placeholder for dry runs.

```powershell
$env:GOOGLE_CLOUD_PROJECT = "your-project-id"
$env:VERTEX_LOCATION = "us-central1"  # optional, default
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\service-account.json"  # optional when using gcloud auth
```

Notes
- Uses `imagen-3.0-generate-001` via the Vertex AI Python SDK.
- Reference images listed in presets are attached when present on disk.

### Run commands
- Single preset:
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts/generate_kelly_asset.ps1 -Preset presets/identity_front.yaml -OpenFolder
  ```
- Batch all presets in `presets/`:
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts/generate_kelly_batch.ps1
  ```

### Helpers
- Identity contact sheet:
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts/make_identity_contact_sheet.ps1
  ```
- CC5 identity-set JSON (paths to front/3‑4/profile for quick reference):
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts/write_cc5_identity_set.ps1 -Open
  ```

### Environment setup (Vertex/ESRGAN)
Configure env vars for generation in the current shell:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_generation_env.ps1 -ProjectId "your-project-id" -ServiceAccountJson "C:\path\service-account.json" -VertexLocation "us-central1" -RealEsrganExe "C:\tools\realesrgan-ncnn-vulkan.exe"
```
Verify:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/vertex_self_test.ps1
```

### Output locations
- Renders: `projects/Kelly/assets/renders/`
- Manifests: `projects/Kelly/assets/manifests/`

Each run writes a JSON manifest with schema `kelly.asset.manifest/v1` including prompt, refs, parameters, seed, and file checksum.

### Preset schema (minimal)
```yaml
asset_type: identity | expression | hair_plate | texture | background
view: front | three_quarter | profile | ...
lighting: studio_neutral | backlit_edge | ...
prompt: | # required
  text prompt
negative_prompt: optional text
seed: 123456  # optional
output:
  # choose one of:
  profile: 8k_square | 8k_16x9  # exact sizes: 8192×8192 or 7680×4320
  # or explicit sizes
  width: 2048
  height: 2048
  version: 1
backend:
  provider: google-vertex
  model: imagen-3.0-generate-001
  guidance_scale: 7.5
  steps: 40
refs:
  images:
    - path: iLearnStudio/projects/Kelly/ref/front.png
    - path: iLearnStudio/projects/Kelly/ref/three_quarter.png
    - path: iLearnStudio/projects/Kelly/ref/profile.png
```

### Reference images
Place identity references at:
`iLearnStudio\projects\Kelly\ref\front.png`, `three_quarter.png`, `profile.png`.

### CC5 (Character Creator 5) – Click‑by‑click
1) Open CC5 → File > New Project.
2) Modify panel → Character → Load CC5 HD Neutral Base.
3) Plugins → Headshot 2 → Open.
4) Photo Fitting → Load `projects/Kelly/assets/renders/kelly_identity_front_v001.png` → Auto Detect → Preserve Identity = On.
5) Profile Refinement → Load `kelly_identity_profile_v001.png` → Apply.
6) 3/4 Refinement → Load `kelly_identity_three_quarter_v001.png` → Apply.
7) Click Generate Head → Wait for bake.
8) Optional Hair: apply style; use `kelly_hair_plate_v001.png` only as visual reference.
9) Materials (optional): assign neutral diffuse textures if available.
10) Save: `iLearnStudio\projects\Kelly\CC5\Kelly_8K_Identity.ccProject`.

### iClone 8 – Click‑by‑click
1) In CC5: use Send to iClone.
2) Lighting: Create > Light > 3‑Point (or your studio rig).
3) Cameras: Create > Camera; set front/3‑4/profile positions; save.
4) Render Settings: Resolution 7680×4320 for tests.
5) Render Image → save under `renders\Kelly\`.

### QA checklist (quick)
- Identity consistent across front/3‑4/profile (eyes, nose, jawline).
- Eye integrity: no sclera artifacts; iris roundness preserved.
- Hair edge continuity; no halo on dark, graceful on light.
- Exposure/white balance consistent; no heavy skin banding.

### Roadmap (next)
- Real‑ESRGAN upscale: If `backend.upscale.enabled=true` in a preset, the generator attempts an upscaling pass. It first tries `realesrgan-ncnn-vulkan` if found in PATH; otherwise it tries the Python package `realesrgan` (if installed) with `weights/RealESRGAN_x4.pth`. Final images are resized exactly to 8K profile sizes.
- QC metrics: The manifest now includes lightweight metrics — resolution, brightness mean, contrast std, colorfulness index, and sharpness variance (Laplacian). These help spot exposure/blur issues at a glance.
- QC thresholds & verdict: Defaults are `brightness_min=0.10`, `brightness_max=0.90`, `contrast_min=0.05`, `sharpness_min=50.0`. Override per preset with:
  ```yaml
  backend:
    qc_thresholds:
      brightness_min: 0.12
      sharpness_min: 60
  ```
  Manifest example:
  ```json
  {
    "qc": {
      "metrics": { "brightness_mean": 0.36, ... },
      "verdict": { "pass": true, "reasons": [], "thresholds": { ... } }
    }
  }
  ```
- Optional: Additional lighting and background presets.


