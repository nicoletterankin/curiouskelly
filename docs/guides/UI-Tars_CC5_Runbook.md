---
title: "UI‑Tars Runbook — CC5/iClone Studio Bootstrap (Kelly + 11)"
version: "2.0.0"
date: "2025-10-13"
updated: "CURRENT SESSION - Kelly Talking Pipeline"
root_dir: "C:\\iLearnStudio"
primary_audio: "projects\\Kelly\\Audio\\kelly25_audio.wav"
kelly_headshot: "projects\\Kelly\\Ref\\headshot2-kelly-base169 101225.png"
kelly_cc5_project: "projects\\Kelly\\CC5\\Kelly_8K_Production.ccProject"
kelly_voice_id: "wAdymQH5YucAkXwmrdL0"
elevenlabs_api: "sk_17b7a1d5b54e992c687a165646ddf84dd3997cd748127568"
contact_sheet_output: "analytics\\Kelly\\kelly_contact_sheet_6x5.png"
---

# UI‑Tars Runbook — CC5/iClone Studio Bootstrap (Kelly + 11)

## 🎯 CURRENT STATUS (October 13, 2025)

**✅ ASSETS WE HAVE:**
- Kelly headshot photo: `C:\iLearnStudio\projects\Kelly\Ref\headshot2-kelly-base169 101225.png`
- CC5 Projects: `Kelly_G3Plus_Base.ccProject` & `Kelly_8K_Production.ccProject`
- Audio file: `C:\iLearnStudio\projects\Kelly\Audio\kelly25_audio.wav`
- ElevenLabs API: Voice ID `wAdymQH5YucAkXwmrdL0` (Kelly voice trained and ready)
- API Key: `sk_17b7a1d5b54e992c687a165646ddf84dd3997cd748127568`

**🎯 TODAY'S GOAL: Get Kelly Talking in Video**

**📍 WHERE WE ARE:**
- CC5 headshot workflow "kinda working" (last attempt)
- Need to complete the pipeline: CC5 → iClone → AccuLips → Render

**🚀 IMMEDIATE ACTION PLAN:**
1. **Generate fresh Kelly lipsync audio** (5 min) - Use ElevenLabs API
2. **Load Kelly_8K_Production.ccProject in CC5** (2 min) - Review character
3. **Export to iClone 8** (5 min) - Send character from CC5
4. **Apply AccuLips lipsync** (10 min) - Import audio, run AccuLips
5. **Render test video** (20-60 min) - Get Kelly talking!

**📂 Key File Locations:**
- Workspace: `C:\Users\user\UI-TARS-desktop`
- iLearnStudio: `C:\iLearnStudio`
- Audio generator: `synthetic_tts\generate_kelly_lipsync.py`
- CC5 workflow: `C:\iLearnStudio\projects\Kelly\Ref\Kelly_Simple_Steps.txt`

---

> **Purpose**: A single, automation‑friendly plan that **UI‑Tars** (or any agent) can execute to install, configure, and operate a **Character Creator 5 (CC5) + iClone 8** studio on Windows with an RTX GPU (e.g., **RTX 5090**), producing a photoreal talking avatar **"Kelly"** and scaffolding **11 additional characters**.
>
> **This file is designed for AIs to read and follow.** All paths and filenames are real and consistent.

---

## 0) Success Criteria (what "done" means)

- **Software installed & licensed**: CC5, Headshot 2, iClone 8 (+ Motion LIVE hub), *(optional)* AccuFACE / Live Face.
- **Studio template saved**: `projects\\_Shared\\iClone\\DirectorsChair_Template.iProject` (85 mm, DOF on eyes, studio lights).
- **Kelly built**: `projects\\Kelly\\CC5\\Kelly_HD_Head.ccProject` (CC5 HD character), sent to iClone.
- **Voice integrated**: `projects\\Kelly\\Audio\\kelly25_audio.wav` mapped via AccuLips.
- **Optional facial nuance**: AccuFACE (Video mode) applied from `projects\\Kelly\\Ref\\kelly_ref_video.mp4` with mouth/jaw disabled (AccuLips retains control).
- **Rendered test**: `renders\\Kelly\\kelly_test_talk_v1.mp4` (1080p or 4K H.264).
- **Analytics generated** under `analytics\\Kelly\\`:
  - Storyboard contact sheet: `kelly_contact_sheet_6x5.png`  *(facial/pose continuity)*
  - Audio plots: `kelly25_waveform.png`, `kelly25_pitch.png`  *(objective pacing & pitch)*
  - Frame metrics: `kelly_test_frame_metrics.csv`  *(framewise luminance & motion)*
- **11 additional characters scaffolded** from `config\\characters.yml` with folders and TODOs created.
- **Machine‑readable status**: `metrics\\install_status.json` contains PASS flags; `docs\\VERSIONS.md` lists installed versions.

---

## 1) Repository & Folders (automated)

**Root directory (edit if needed):** `D:\\iLearnStudio`  
**Characters list file:** `config\\characters.yml`

**Task 1.1 — Initialize repo + folders**

Create the following tree:

```
D:\iLearnStudio\
  config\
  installers\
  scripts\
  tools\
  docs\receipts\
  metrics\
  analytics\Kelly\
  projects\_Shared\{HDRI,Fonts,Logos}\
  projects\_Shared\iClone\
  projects\Kelly\{Audio,Ref,CC5,iClone,Renders}\
  renders\Kelly\
```

**Automation artifacts to create:**

- `scripts\00_bootstrap.ps1`  
- `.gitattributes` (with LFS)  
- `.gitignore`  
- `README.md` (seed)  
- `config\characters.yml` (seed 12 names)

**Acceptance:** Folder tree exists; Git repo initialized; Git‑LFS enabled; `metrics\install_status.json` created.

---

## 2) System Dependencies (automated via PowerShell)

**Task 2.1 — Core tools (winget)**  
Install: Git, Git LFS, Python 3.11, FFmpeg, 7‑Zip.

**Task 2.2 — Python libs**  
Install: `numpy pandas matplotlib librosa soundfile opencv-python`

**Automation:** `scripts\01_install_deps.ps1`

**Acceptance:** `python --version`, `ffmpeg -version`, `git lfs env` succeed; `metrics\install_status.json` marks `deps: PASS`.

---

## 3) Reallusion Stack (purchase + install)

> **HUMAN ACTION REQUIRED** (purchase/sign‑in in Reallusion Hub)
> - **Character Creator 5 (CC5)**
> - **Headshot 2** (CC5 compatible)
> - **iClone 8**
> - **Motion LIVE** hub
> - *(Optional)* **AccuFACE** (webcam/video facial capture)
> - *(Optional)* **Live Face** (iPhone TrueDepth capture)

**Task 3.1 — Download installers**  
Save installers to `installers\` and receipts/PDF invoices to `docs\receipts\`.

**Task 3.2 — Install & sign in**  
- Install **Reallusion Hub**.
- From Hub: install CC5, iClone 8, Headshot 2, Motion LIVE, (AccuFACE), (Live Face).
- Sign in to activate licenses.

**Automation assist:** `scripts\02_detect_reallusion.ps1` (records versions to `docs\VERSIONS.md`, status to `metrics\install_status.json`).

**Acceptance:** CC5 and iClone launch; "About" shows valid licenses; versions recorded.

---

## 4) Project Config & Shared Assets

**Task 4.1 — Content roots** (inside CC5/iClone)  
Set **Custom Content** paths to:
- `D:\iLearnStudio\projects\_Shared\iClone`
- `D:\iLearnStudio\projects\_Shared\HDRI`

**Task 4.2 — Save studio template**  
In iClone:
- Camera: **85 mm**, **DOF focus on eyes**  
- Lights: soft **3‑point** or neutral studio **HDRI**  
- Neutral seated pose + gentle idle (breathing, blinks)  
Save as `projects\_Shared\iClone\DirectorsChair_Template.iProject`

**Acceptance:** Template opens without missing assets.

---

## 5) Kelly: Audio Ingest & Analytics (automated)

**Place your audio:** `projects\Kelly\Audio\kelly25_audio.wav`

**Task 5.1 — Analyze audio**  
Outputs to `analytics\Kelly\`:
- `kelly25_waveform.png`  
- `kelly25_pitch.png`  
- `kelly25_audio_metrics.csv` (columns: time_s, rms_db, f0_hz, pitch_conf)

**Automation:** `tools\analyze_audio.py`, `scripts\10_audio_analyze.ps1`

**Acceptance:** 3 artifacts exist; CSV has >100 rows (unless audio is very short).

---

## 6) Kelly: Build the HD Head in CC5 (guided)

**Task 6.1 — CC5 HD character**  
Create an **HD Character** and save as `projects\Kelly\CC5\Kelly_HD_Head.ccProject`.

**Task 6.2 — Headshot 2**  
Use **Photo to 3D (Pro)** with a synthetic, front‑facing Kelly still. Adjust features (jaw, lips, nose, eyes). Keep **CC5 HD skin/eyes/lashes**. Save.

**Task 6.3 — (Optional) ActorMIXER**  
If available (CC5 Deluxe), rough‑in the face/body with **ActorMIXER PRO** before Headshot 2 for speed.

**Acceptance:** `Kelly_HD_Head.ccProject` exists and loads as HD with proper eyes/lashes.

---

## 7) Send Kelly to iClone & Set Stage (guided)

**Task 7.1 — Send to iClone**  
From CC5 → iClone. Open `projects\_Shared\iClone\DirectorsChair_Template.iProject` and swap in Kelly.

**Task 7.2 — AccuLips**  
Import `projects\Kelly\Audio\kelly25_audio.wav` → **AccuLips** → correct text → **Apply** (mouth only).

**Acceptance:** Timeline shows viseme keys; M/B/P close; F/V lower‑lip on teeth.

---

## 8) (Optional) Facial nuance from video (guided)

**Place reference video:** `projects\Kelly\Ref\kelly_ref_video.mp4`

**Task 8.1 — AccuFACE (Video Mode)**  
Motion LIVE → AccuFACE → **Load Offline Video** = `kelly_ref_video.mp4`.  
**Disable mouth/jaw** (AccuLips controls lips). **Enable brows/eyes/cheeks/head**. **Record**.

**Acceptance:** Subtle brows/eyes/cheeks overlay without mouth conflicts.

---

## 9) Render & QC (automated + guided)

**Task 9.1 — Render test**  
Output: `renders\Kelly\kelly_test_talk_v1.mp4` (1080p or 4K H.264).

**Task 9.2 — Contact sheet (storyboard continuity)**  
Output: `analytics\Kelly\kelly_contact_sheet_6x5.png` (6×5 = 30 frames).

**Task 9.3 — Frame metrics CSV**  
Output: `analytics\Kelly\kelly_test_frame_metrics.csv` (frame, time_s, mean_luma, motion_diff).

**Automation:** `scripts\20_contact_sheet.ps1`, `tools\frame_metrics.py`, `scripts\21_frame_metrics.ps1`

**Acceptance:** MP4 plays; grid/CSV exist; no missing codec errors.

---

## 10) Scale to 11 more characters (automated scaffolding)

**Task 10.1 — Define characters**  
Edit `config\\characters.yml` (seeded example below).

**Task 10.2 — Scaffold folders**  
Creates `projects\\<Name>\\{Audio,Ref,CC5,iClone,Renders}` and `TODO.md` per character; copies studio template.

**Automation:** `scripts\30_new_character.ps1`

**Acceptance:** 12 character folders exist with TODOs and template scenes.

---

## 11) VS Code / Cursor Tasks (automated)

**Task 11.1 — One‑click tasks**  
Create `.vscode\\tasks.json` exposing:
- Bootstrap repo
- Install dependencies
- Analyze audio
- Contact sheet
- Frame metrics
- Scaffold all characters

**Automation:** `scripts\40_write_tasksjson.ps1`

**Acceptance:** Cursor/VS Code **Run Task** shows entries; each runs without path errors.

---

## 12) Metrics & Logs (automated)

- `metrics\\install_status.json` — rolling PASS/FAIL updates by scripts.
- `docs\\VERSIONS.md` — installed versions.
- `metrics\\artifact_index.csv` — key outputs per character.

**Acceptance:** Files exist with timestamps and PASS flags.

---

# Automation Payloads (create verbatim)

> **Note**: Paths assume `D:\iLearnStudio`. If you change the root, update the scripts (search/replace `D:\\iLearnStudio`).

## `scripts\00_bootstrap.ps1`

```powershell
$ErrorActionPreference = "Stop"
$root = "D:\iLearnStudio"
$dirs = @(
  "$root","$root\config","$root\installers","$root\scripts","$root\tools",
  "$root\docs\receipts","$root\metrics","$root\analytics\Kelly",
  "$root\projects\_Shared\HDRI","$root\projects\_Shared\Fonts","$root\projects\_Shared\Logos",
  "$root\projects\_Shared\iClone",
  "$root\projects\Kelly\Audio","$root\projects\Kelly\Ref","$root\projects\Kelly\CC5","$root\projects\Kelly\iClone","$root\projects\Kelly\Renders",
  "$root\renders\Kelly"
)
$dirs | ForEach-Object { if(-not (Test-Path $_)) { New-Item -ItemType Directory -Path $_ | Out-Null } }

# Git repo + LFS
if(-not (Test-Path "$root\.git")) { git init $root | Out-Null }
Set-Content -Path "$root\.gitignore" -Value @"
# Build / cache / binary
*.pyc
__pycache__/
.vscode/
*.tmp
*.log
installers/
"@
Set-Content -Path "$root\.gitattributes" -Value @"
*.wav filter=lfs diff=lfs merge=lfs -text
*.mp4 filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
*.tif filter=lfs diff=lfs merge=lfs -text
*.iProject filter=lfs diff=lfs merge=lfs -text
*.ccProject filter=lfs diff=lfs merge=lfs -text
"@
git -C $root lfs install

# Seed README
if(-not (Test-Path "$root\README.md")) {
Set-Content -Path "$root\README.md" -Value "# iLearn Studio — CC5/iClone Pipeline`nThis repo is managed by UI‑Tars."
}

# Seed characters.yml if missing
if(-not (Test-Path "$root\config\characters.yml")) {
Set-Content -Path "$root\config\characters.yml" -Value @"
characters:
  - name: Kelly
    voice_wav: projects/Kelly/Audio/kelly25_audio.wav
  - name: Ken
  - name: Amina
  - name: Leo
  - name: Priya
  - name: Mateo
  - name: Sofia
  - name: Grace
  - name: Omar
  - name: Hana
  - name: Liam
  - name: Mei
"@
}

# Metrics file
$ms = "$root\metrics\install_status.json"
if(-not (Test-Path $ms)) { Set-Content -Path $ms -Value "{`"created`": `"$((Get-Date).ToString("s"))`"}" }

Write-Host "Bootstrap folders + repo complete."
```

## `scripts\01_install_deps.ps1`

```powershell
$ErrorActionPreference = "Stop"
function MarkStatus($k,$v){
  $f="D:\iLearnStudio\metrics\install_status.json"
  if(Test-Path $f){$j=Get-Content $f -Raw | ConvertFrom-Json}else{$j=@{}}
  $j | Add-Member -NotePropertyName $k -NotePropertyValue $v -Force
  ($j | ConvertTo-Json -Depth 4) | Set-Content $f
}

# Winget installs
$apps = @(
  @{id="Git.Git"; name="Git"},
  @{id="GitHub.GitLFS"; name="Git LFS"},
  @{id="Python.Python.3.11"; name="Python"},
  @{id="Gyan.FFmpeg"; name="FFmpeg"},
  @{id="7zip.7zip"; name="7-Zip"}
)

foreach($a in $apps){
  try { winget install --id $a.id --silent --accept-package-agreements --accept-source-agreements -e }
  catch { Write-Warning "Failed $($a.name). Try: winget search $($a.name)"; }
}

# Python libs
py -3.11 -m pip install --upgrade pip
py -3.11 -m pip install numpy pandas matplotlib librosa soundfile opencv-python

MarkStatus "deps" "PASS"
Write-Host "Dependencies installed."
```

## `scripts\02_detect_reallusion.ps1`

```powershell
$ErrorActionPreference = "SilentlyContinue"
$root="D:\iLearnStudio"
$versPath="$root\docs\VERSIONS.md"
$status="$root\metrics\install_status.json"

function GetExeVersion($path){
  if(Test-Path $path){
    $v = (Get-Item $path).VersionInfo.ProductVersion
    return $v
  } else { return $null }
}

$paths = @{
  "Character Creator 5" = "C:\Program Files\Reallusion\Character Creator 5\Bin64\CharacterCreator.exe"
  "iClone 8"            = "C:\Program Files\Reallusion\iClone 8\Bin64\iClone.exe"
}

$versions = @()
foreach($k in $paths.Keys){
  $v = GetExeVersion $paths[$k]
  $versions += @{ name=$k; version=($v?$v:"(not found)") }
}

# Write VERSIONS.md
$md = "# Installed Versions`r`n"
foreach($rec in $versions){
  $md += "- {0}: {1}`r`n" -f $rec.name, $rec.version
}
Set-Content -Path $versPath -Value $md

# Update status JSON
if(Test-Path $status){ $j=Get-Content $status -Raw | ConvertFrom-Json } else { $j=@{} }
$ok = ($versions | Where-Object { $_.version -ne "(not found)" }).Count -ge 1
$j | Add-Member -NotePropertyName "reallusion_detect" -NotePropertyValue ($ok?"PASS":"WARN") -Force
($j | ConvertTo-Json -Depth 4) | Set-Content $status

Write-Host "Detection complete. See $versPath"
```

## `tools\analyze_audio.py`

```python
import os, numpy as np, pandas as pd, soundfile as sf, librosa, librosa.display, matplotlib.pyplot as plt

audio_path = r"D:\iLearnStudio\projects\Kelly\Audio\kelly25_audio.wav"
out_dir = r"D:\iLearnStudio\analytics\Kelly"
os.makedirs(out_dir, exist_ok=True)

y, sr = librosa.load(audio_path, sr=None, mono=True)

# RMS (dB)
frame_len = 2048; hop = 512
rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
rms_db = librosa.amplitude_to_db(rms, ref=1.0)
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

# Pitch (f0) with confidence
f0, voiced_flag, voiced_probs = librosa.pyin(
    y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
)
# Align sizes
minN = min(len(times), len(f0), len(voiced_probs))
times, f0, voiced_probs = times[:minN], f0[:minN], voiced_probs[:minN]
rms_db = rms_db[:minN]

df = pd.DataFrame({
    "time_s": times,
    "rms_db": rms_db,
    "f0_hz": np.nan_to_num(f0, nan=0.0),
    "pitch_conf": voiced_probs
})
df.to_csv(os.path.join(out_dir, "kelly25_audio_metrics.csv"), index=False)

# Waveform plot
plt.figure(figsize=(12,3))
librosa.display.waveshow(y, sr=sr)
plt.title("Kelly25 Waveform")
plt.tight_layout()
plt.savefig(os.path.join(out_dir,"kelly25_waveform.png"), dpi=150)
plt.close()

# Pitch plot
plt.figure(figsize=(12,3))
plt.plot(times, df["f0_hz"])
plt.title("Kelly25 Pitch (Hz)")
plt.xlabel("Time (s)"); plt.ylabel("f0 (Hz)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir,"kelly25_pitch.png"), dpi=150)
plt.close()

print("Audio analytics complete.")
```

## `scripts\10_audio_analyze.ps1`

```powershell
py -3.11 "D:\iLearnStudio\tools\analyze_audio.py"
if($LASTEXITCODE -eq 0){ Write-Host "Audio analysis OK." } else { throw "Audio analysis failed." }
```

## `scripts\20_contact_sheet.ps1`

```powershell
param(
  [string]$InputVideo = "D:\iLearnStudio\renders\Kelly\kelly_test_talk_v1.mp4",
  [string]$OutPng     = "D:\iLearnStudio\analytics\Kelly\kelly_contact_sheet_6x5.png",
  [int]$Cols = 6, [int]$Rows = 5, [int]$Width = 640
)
# 30 frames total (Cols x Rows), sample ~1 fps (adjust as needed)
$tile = "$Cols"x"$Rows"
ffmpeg -y -i "$InputVideo" -vf "fps=1,scale=$Width:-1,tile=$tile" -frames:v 1 "$OutPng"
if($LASTEXITCODE -ne 0){ throw "Contact sheet failed." } else { Write-Host "Contact sheet created: $OutPng" }
```

## `tools\frame_metrics.py`

```python
import cv2, os, pandas as pd

inp = r"D:\iLearnStudio\renders\Kelly\kelly_test_talk_v1.mp4"
out_csv = r"D:\iLearnStudio\analytics\Kelly\kelly_test_frame_metrics.csv"

cap = cv2.VideoCapture(inp)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame=0; rows=[]
ret = True
prev_gray=None

while ret:
    ret, img = cap.read()
    if not ret: break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_luma = float(gray.mean())
    motion = 0.0
    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        motion = float(diff.mean())
    rows.append({"frame":frame,"time_s":frame/float(fps),"mean_luma":mean_luma,"motion_diff":motion})
    prev_gray = gray; frame+=1

cap.release()
pd.DataFrame(rows).to_csv(out_csv, index=False)
print("Frame metrics CSV saved:", out_csv)
```

## `scripts\21_frame_metrics.ps1`

```powershell
py -3.11 "D:\iLearnStudio\tools\frame_metrics.py"
if($LASTEXITCODE -ne 0){ throw "Frame metrics failed." } else { Write-Host "Frame metrics OK." }
```

## `scripts\30_new_character.ps1`

```powershell
param([string]$Config="D:\iLearnStudio\config\characters.yml")
$root = "D:\iLearnStudio"
$yaml = Get-Content $Config -Raw
# Naive parse for names (swap to a YAML parser if available)
$names = ($yaml -split "`n") | Where-Object { $_ -match "^\s*-\s*name:\s*(.+)$" } | ForEach-Object { ($_ -replace "^\s*-\s*name:\s*","").Trim() }
foreach($n in $names){
  $base = Join-Path $root "projects\$n"
  foreach($sub in @("Audio","Ref","CC5","iClone","Renders")) {
    $p = Join-Path $base $sub; if(-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p | Out-Null }
  }
  $todo = Join-Path $base "TODO.md"
  if(-not (Test-Path $todo)){
    Set-Content -Path $todo -Value @"
# $n — Build TODO
1) CC5: Create HD head (Headshot 2 or ActorMIXER), save to projects/$n/CC5
2) Send to iClone. Load DirectorsChair_Template.iProject.
3) Voice: drop WAV into projects/$n/Audio and run AccuLips.
4) (Optional) AccuFACE VIDEO with mouth disabled.
5) Render test to renders/$n/${n}_test_talk_v1.mp4
6) Run contact sheet + frame metrics scripts for $n.
"@
  }
}
Write-Host "Character scaffolds ready."
```

## `scripts\40_write_tasksjson.ps1`

```powershell
$root="D:\iLearnStudio"
$vscode="$root\.vscode"
if(-not (Test-Path $vscode)) { New-Item -ItemType Directory -Path $vscode | Out-Null }
$tasks=@"
{
  "version": "2.0.0",
  "tasks": [
    { "label": "Bootstrap Repo", "type":"shell", "command":"powershell", "args":["-ExecutionPolicy","Bypass","-File","$root\\scripts\\00_bootstrap.ps1"] },
    { "label": "Install Deps", "type":"shell", "command":"powershell", "args":["-ExecutionPolicy","Bypass","-File","$root\\scripts\\01_install_deps.ps1"] },
    { "label": "Detect Reallusion Installs", "type":"shell", "command":"powershell", "args":["-ExecutionPolicy","Bypass","-File","$root\\scripts\\02_detect_reallusion.ps1"] },
    { "label": "Analyze Audio (Kelly)", "type":"shell", "command":"powershell", "args":["-ExecutionPolicy","Bypass","-File","$root\\scripts\\10_audio_analyze.ps1"] },
    { "label": "Contact Sheet (Kelly)", "type":"shell", "command":"powershell", "args":["-ExecutionPolicy","Bypass","-File","$root\\scripts\\20_contact_sheet.ps1"] },
    { "label": "Frame Metrics (Kelly)", "type":"shell", "command":"powershell", "args":["-ExecutionPolicy","Bypass","-File","$root\\scripts\\21_frame_metrics.ps1"] },
    { "label": "Scaffold All Characters", "type":"shell", "command":"powershell", "args":["-ExecutionPolicy","Bypass","-File","$root\\scripts\\30_new_character.ps1","-Config","$root\\config\\characters.yml"] }
  ]
}
"@
Set-Content -Path "$vscode\tasks.json" -Value $tasks
Write-Host "VSCode/Cursor tasks written."
```

## `docs\VERSIONS.md` (stub; will be overwritten)

```markdown
# Installed Versions
- Character Creator 5: (pending detection)
- Headshot 2: (pending detection)
- iClone 8: (pending detection)
- Motion LIVE: (pending detection)
- AccuFACE: (optional, pending detection)
```

---

## 13) Execution Order (for UI‑Tars)

1. Run `scripts\00_bootstrap.ps1`
2. Run `scripts\01_install_deps.ps1`
3. **Prompt human**: purchase + install CC5/iClone/Headshot2/Motion LIVE/(AccuFACE) via Reallusion Hub.
4. Run `scripts\02_detect_reallusion.ps1` → fill `docs\VERSIONS.md`, update `metrics\install_status.json`.
5. Place `projects\Kelly\Audio\kelly25_audio.wav` and (optional) `projects\Kelly\Ref\kelly_ref_video.mp4`.
6. Run `scripts\10_audio_analyze.ps1` → create waveform, pitch, metrics CSV.
7. **Human in CC5/iClone**: build Kelly HD head → send to iClone → open DirectorsChair_Template.iProject → AccuLips on WAV → (optional) AccuFACE Video (brows/eyes/cheeks only) → render `renders\Kelly\kelly_test_talk_v1.mp4`.
8. Run `scripts\20_contact_sheet.ps1` and `scripts\21_frame_metrics.ps1`.
9. Edit `config\characters.yml` with the other 11 names and future voice_wav paths.
10. Run `scripts\30_new_character.ps1` to scaffold everyone.
11. Run `scripts\40_write_tasksjson.ps1` to expose tasks in Cursor/VS Code.

---

## 14) QC Checklist (copyable to `metrics\QC_Kelly.md`)

- [ ] **Mouth**: M/B/P fully closed; F/V lower lip touches teeth.
- [ ] **Blinks**: natural, not mirrored; rate ~3–5s.
- [ ] **Eyes**: catchlight visible; DOF focus on eyes.
- [ ] **Skin**: no waxy SSS; appropriate roughness in T‑zone.
- [ ] **Hairline**: no scalp peek; no plastic specular.
- [ ] **Audio**: no clipping; consistent loudness.
- [ ] **Contact sheet**: `kelly_contact_sheet_6x5.png` shows framing continuity.
- [ ] **Frame metrics**: `kelly_test_frame_metrics.csv` shows expected motion peaks on emphasized words.

---

## 15) Characters YAML (example)

```yaml
characters:
  - name: Kelly
    voice_wav: projects/Kelly/Audio/kelly25_audio.wav
  - name: Ken
  - name: Amina
  - name: Leo
  - name: Priya
  - name: Mateo
  - name: Sofia
  - name: Grace
  - name: Omar
  - name: Hana
  - name: Liam
  - name: Mei
```

---

## 16) Notes & Constraints

- **Purchases and account sign‑ins cannot be fully automated**; UI‑Tars should pause and prompt.
- **If AccuFACE is unavailable**, skip Section 8 and rely on AccuLips + manual facial key edits.
- **Paths assume `D:\iLearnStudio`**. If you change the root, search/replace in all scripts.
- **Prefer consistent camera/lighting** for all 12 characters to enhance trust and continuity across lessons.

---

## 17) Deliverables Summary

- `projects\_Shared\iClone\DirectorsChair_Template.iProject`
- `projects\Kelly\CC5\Kelly_HD_Head.ccProject`
- `renders\Kelly\kelly_test_talk_v1.mp4`
- `analytics\Kelly\kelly_contact_sheet_6x5.png`
- `analytics\Kelly\kelly25_waveform.png`
- `analytics\Kelly\kelly25_pitch.png`
- `analytics\Kelly\kelly_test_frame_metrics.csv`
- `config\characters.yml` (12 names)
- `metrics\install_status.json`, `docs\VERSIONS.md`

---

## 18) TODAY'S WORKFLOW - Get Kelly Talking (Step-by-Step)

### ✅ STEP 1: Generate Kelly's Voice for Lipsync (5 minutes)

**Option A: Use Interactive Generator (Recommended)**
```powershell
cd C:\Users\user\UI-TARS-desktop\synthetic_tts
python generate_kelly_lipsync.py
```
- Select option 1-5 for pre-written lesson text
- OR select 6 to enter custom text
- Audio saves to: `C:\iLearnStudio\projects\Kelly\Audio\kelly_[name]_lipsync.wav`

**Option B: Quick Command Line**
```powershell
cd C:\Users\user\UI-TARS-desktop\synthetic_tts
python -c "from generate_kelly_lipsync import generate_kelly_lipsync; generate_kelly_lipsync('Hello! I am Kelly, your learning companion. Today we are going to explore something amazing together!', 'kelly_quick_test.wav')"
```

**Output:** WAV file ready for iClone (22,050 Hz, Mono, normalized)

---

### ✅ STEP 2: Open Kelly Character in CC5 (2 minutes)

1. Launch **Character Creator 5**
2. **File → Open Project**
3. Navigate to: `C:\iLearnStudio\projects\Kelly\CC5\Kelly_8K_Production.ccProject`
4. Click **Open**
5. Wait for character to load
6. **Review the character:**
   - Check facial features
   - Verify skin quality
   - Confirm eyes and hair are present

**Status Check:** Character should be CC5 HD format with high detail

---

### ✅ STEP 3: Export Kelly to iClone 8 (5 minutes)

**In Character Creator 5:**
1. **File → Send Character to iClone**
2. In dialog box:
   - ✅ Check "Export with Facial Profile"
   - ✅ Check "Export with Expression Wrinkle"
   - ✅ Quality: **Ultra High** or **High**
3. Click **Send to iClone**
4. Wait for iClone 8 to launch and receive character (2-3 min)

**iClone 8 should open automatically with Kelly imported**

---

### ✅ STEP 4: Set Up Scene in iClone (5 minutes)

**In iClone 8:**
1. **Position Kelly:**
   - Click character in scene
   - Move to center if needed
   - Ensure facing camera

2. **Set Camera:**
   - Select camera
   - **Focal Length: 85mm** (portrait lens)
   - Frame Kelly's head and shoulders
   - Center composition

3. **Set Lighting:**
   - Use default studio lighting, OR
   - Add 3-point light setup:
     - Key light (main, 45° front-left)
     - Fill light (softer, 45° front-right)  
     - Rim light (back-top for hair definition)

4. **Save Scene Template:**
   - File → Save As
   - Name: `Kelly_Director_Chair_Template.iProject`
   - Location: `C:\iLearnStudio\projects\Kelly\iClone\`

---

### ✅ STEP 5: Apply AccuLips Lipsync (10 minutes)

**In iClone 8:**

1. **Import Audio:**
   - Timeline panel (bottom)
   - Right-click in audio track area
   - **Import Audio File**
   - Select your generated audio: `C:\iLearnStudio\projects\Kelly\Audio\kelly_[name]_lipsync.wav`
   - Audio waveform appears in timeline

2. **Apply AccuLips:**
   - Select Kelly character in scene
   - Menu: **Animation → Facial Animation → AccuLips**
   - In AccuLips window:
     - **Audio:** Select your imported audio track
     - **Language:** English
     - **Quality:** High or Ultra High
     - **Method:** Accu3D (best quality)
   - Click **Apply**
   - Wait 1-3 minutes for processing

3. **Preview Lipsync:**
   - Press **SPACEBAR** or click Play button
   - Watch Kelly's mouth sync with audio
   - Scrub timeline to check specific words
   - Look for: M/B/P closing lips, F/V lower lip to teeth

4. **Fine-Tune (if needed):**
   - If sync is off: adjust audio track timing slightly
   - If mouth movements too subtle: increase strength in AccuLips settings
   - Re-apply if major changes needed

---

### ✅ STEP 6: Render Test Video (20-60 minutes)

**In iClone 8:**

1. **Set Render Range:**
   - Timeline: drag range selector to cover audio length
   - Or use full timeline

2. **Configure Render Settings:**
   - Menu: **File → Export → Video**
   - **Format:** MP4 (H.264)
   - **Resolution:** 
     - Quick test: 1920×1080 (Full HD) - ~20-30 min
     - High quality: 3840×2160 (4K) - ~45-60 min
   - **Frame Rate:** 30 FPS (or 60 for smooth)
   - **Quality:** High
   - **Output Path:** `C:\iLearnStudio\renders\Kelly\kelly_talking_test_v1.mp4`

3. **Start Render:**
   - Click **Export** button
   - Monitor progress bar
   - RTX 5090 should render quickly

4. **Wait for Completion:**
   - Progress window shows percentage
   - Do not close iClone during render
   - Render completes when progress reaches 100%

---

### ✅ STEP 7: Review & Quality Check (5 minutes)

**After render completes:**

1. **Play Video:**
   - Navigate to: `C:\iLearnStudio\renders\Kelly\kelly_talking_test_v1.mp4`
   - Double-click to play in default player
   - Watch full video

2. **Quality Checklist:**
   - ✅ Lipsync accuracy: mouth matches words?
   - ✅ Audio quality: clear, no distortion?
   - ✅ Facial expressions: natural, realistic?
   - ✅ Lighting: professional, good skin tones?
   - ✅ Camera: good framing, focus on eyes?
   - ✅ Overall: believable talking avatar?

3. **If Issues Found:**
   - Poor lipsync: Re-run AccuLips with different settings
   - Bad lighting: Adjust lights in iClone, re-render
   - Audio issues: Generate new audio with different settings
   - Character quality: Return to CC5, adjust features

4. **If Successful:**
   - 🎉 **KELLY IS TALKING!**
   - Save iClone project for future use
   - Document settings that worked
   - Ready for production lessons

---

## 19) TROUBLESHOOTING

### Issue: ElevenLabs API Error
**Solution:**
- Check internet connection
- Verify API key is valid: `sk_17b7a1d5b54e992c687a165646ddf84dd3997cd748127568`
- Check ElevenLabs account credits/quota
- Try different voice settings (stability, similarity)

### Issue: CC5 Project Won't Open
**Solution:**
- Ensure CC5 is fully updated
- Check file path is correct
- Try opening a fresh CC5 project first
- If corrupt: use `Kelly_G3Plus_Base.ccProject` instead

### Issue: iClone Won't Receive Character
**Solution:**
- Restart both CC5 and iClone
- Send character again from CC5
- Check firewall isn't blocking communication
- Manually export from CC5, import in iClone

### Issue: AccuLips Not Working
**Solution:**
- Verify audio format: WAV, 22,050 Hz or 44,100 Hz, Mono
- Check audio track is properly imported in timeline
- Select character before running AccuLips
- Try different quality settings
- Re-import audio if waveform doesn't show

### Issue: Render Takes Too Long
**Solution:**
- Reduce resolution to 1080p for test
- Lower quality setting to Medium
- Reduce frame range to first 10 seconds only
- Check GPU drivers are updated
- Close other GPU-intensive programs

### Issue: Lipsync Quality Poor
**Solution:**
- Re-generate audio with higher quality settings
- Increase AccuLips quality to Ultra High
- Check audio has clear pronunciation
- Manually adjust mouth keys in timeline if needed
- Try Accu3D method instead of Accu2D

---

## 20) NEXT STEPS AFTER SUCCESS

Once Kelly is talking:

1. **Create Lesson Library:**
   - Generate 5-10 lesson audio files for different topics
   - Create age-variant versions (child, teen, adult, elder)
   - Organize in `C:\iLearnStudio\projects\Kelly\Audio\lessons\`

2. **Expand to Other Characters:**
   - Use same pipeline for Ken, Amina, Leo, etc.
   - Clone ElevenLabs voices for each character
   - Follow same CC5 → iClone → AccuLips workflow

3. **Optimize Production:**
   - Create iClone scene templates for each lesson type
   - Batch render multiple lessons overnight
   - Automate with Python scripts where possible

4. **Integrate with Lesson Player:**
   - Export videos to `lesson-player/videos/`
   - Update `lesson-dna-schema.json` with video paths
   - Test in browser lesson player
   - Deploy to Cloudflare

5. **Quality & Analytics:**
   - Run contact sheet generator on all renders
   - Generate frame metrics for quality tracking
   - Document successful settings
   - Create production runbook

---

**End of Runbook.**

---

## Appendix — Folder alignment checklist (from current session)

- Create `C:\iLearnStudio\Kelly\` structure with: `01_CC5_Source\`, `02_iClone_Animation\`, `03_Audio\`, `04_Render\Previews\`, `04_Render\Final\`.
- Move `.ccProject` into `01_CC5_Source\` and textures into `01_CC5_Source\Textures\`.
- Move `.iProject`/`.iAvatar` into `02_iClone_Animation\`.
- Open and re‑save CC5 project from `01_CC5_Source\` so CC5 remembers the path.
- Send to iClone, then save avatar and scene into `02_iClone_Animation\`.
- Set iClone render outputs: previews → `04_Render\Previews\`, finals → `04_Render\Final\`.
- Place test WAV in `03_Audio\`, apply AccuLips, and render a short preview.

