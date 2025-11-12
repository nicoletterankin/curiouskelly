## Kelly Batch Run — Cheat Card

Goal: Generate Kelly’s identity assets (8K renders + QC manifests) so we can import into CC5 and lip‑sync in iClone.

0) Open the right terminal
- Inside VS Code: Terminal → New Terminal (PowerShell)
- Or: Start → PowerShell (Run as Administrator), then:
  ```powershell
  cd "C:\Users\user\UI-TARS-desktop"
  ```

1) Quick pre‑flight checks
```powershell
Get-Location
Test-Path .\scripts\generate_kelly_batch.ps1
Test-Path .\presets
python --version
```

2) Choose your run mode
- Standard (simplest):
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts/generate_kelly_batch.ps1 *>&1 | Tee-Object -FilePath .\runlog.txt
  ```
- Full (Vertex + ESRGAN):
  ```powershell
  $env:GOOGLE_CLOUD_PROJECT="your-project-id"
  $env:VERTEX_LOCATION="us-central1"
  $env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\service-account.json"
  $env:REAL_ESRGAN_EXE="C:\tools\realesrgan-ncnn-vulkan.exe"
  Test-Path $env:GOOGLE_APPLICATION_CREDENTIALS
  Test-Path $env:REAL_ESRGAN_EXE
  powershell -ExecutionPolicy Bypass -File scripts/generate_kelly_batch.ps1 *>&1 | Tee-Object -FilePath .\runlog.txt
  ```

3) What success looks like
- Summary:
  ```
  === Kelly Batch Summary ===
  PASS: N   FAIL: M
  ```
- Renders: `projects/Kelly/assets/renders/`
- Manifests: `projects/Kelly/assets/manifests/`

Check last summary line from log:
```powershell
Select-String -Path .\runlog.txt -Pattern "PASS|FAIL" | Select-Object -Last 1
```

4) If there’s a FAIL, grab one manifest’s qc section
```powershell
$mf = Get-ChildItem ".\projects\Kelly\assets\manifests" -Filter *.json | Select-Object -First 1
(Get-Content $mf.FullName -Raw | ConvertFrom-Json).qc | ConvertTo-Json -Depth 8
```

5) Visual consistency checklist
- Front / ¾ / Profile must look like the same person
- No blur/stretch; lighting consistent

6) Common gotchas
- Script execution disabled → Use the provided command with `-ExecutionPolicy Bypass`
- Python not found → Ensure Python/venv in PATH; `pip install -r requirements.txt`
- Missing credentials/tools → Fix paths or run Standard mode

7) Next two clicks toward lip‑sync
- CC5: Import the passed identity set via Headshot 2
- iClone: Send to iClone → AccuLips → load .wav → Generate


