# 🎬 Kelly HD Pipeline (ActorMIXER → Headshot 2 → iClone 8.62)

### Kelly Avatar Production Pipeline © 2025 | The Daily Lesson Project  
**From concept to millions of daily learners**  

---

## 🖥️ System Requirements & Current Setup

**✅ Your Current System (Verified October 12, 2025):**
- **GPU:** NVIDIA GeForce RTX 5090 (32GB VRAM)
- **Driver:** 581.29 WHQL (Latest for RTX 50-series)
- **CUDA:** 13.0
- **Status:** Fully optimized for 8K rendering & real-time processing

**Performance Advantages:**
- 32GB VRAM enables simultaneous 8K texture processing
- Hardware-accelerated ray tracing for photorealistic lighting
- Real-time viewport performance with Digital Human Shader
- Headshot 2 processing: ~5-8 min (vs. ~10-15 min on older GPUs)
- 4K/8K render times: 2-3x faster than RTX 40-series

💡 **Tip:** Your RTX 5090 can handle maximum quality settings across all stages. Don't compromise on resolution or detail levels.

---

## 🧭 Step 1 — Start HD Character (ActorMIXER → Base Face)

### 1️⃣ Launch & Prepare
1. **Open Character Creator 5**  
   - Start Menu → Type “Character Creator 5” → Enter  
   - Wait ~30 sec for UI to fully load.  
2. **Create a New Project**  
   - Menu Bar → `File → New Project`  
   - Save location: `projects/Kelly/CC5/`  
   - Name: `Kelly_HD_Base.ccProject`

### 2️⃣ Create Base Model with ActorMIXER
1. In the **Content** panel (left), open:  
   `Actor → Character → HD Human Anatomy Set`  
2. Pick **Female Athletic (CC5 HD)** for balanced proportions.  
3. Click **Apply** → wait for model to load in viewport.  
4. **Refine Base Face Approximation**  
   - Top Toolbar → `ActorMIXER → Head`  
   - Adjust sliders:  
     - Nose Bridge ≈ Kelly photo angle (−5 to −10)  
     - Jawline Width ≈ Kelly (+4)  
     - Mouth Corner Up ≈ +3 (smile lines)  
5. Save checkpoint → `Ctrl+S` → confirm path: `projects/Kelly/CC5/Kelly_HD_Base.ccProject`

---

## 🧭 Step 2 — Headshot 2 → Photo to 3D (HD Pipeline)

### 1️⃣ Activate Headshot 2
1. Top Menu → `Plugins → Headshot 2 → Photo to 3D (Pro)`  
2. In the dialog box, click **Load Photo**.  
3. Navigate to your synthetic reference:  
   `projects/Kelly/Ref/headshot2-kelly-base169_101225.png`

### 2️⃣ Configure Settings
| Setting | Recommended Value |
|:--|:--|
| **Resolution** | Ultra High (8K) |
| **Mesh Density** | Maximum |
| **Detail Level** | Maximum |
| **Processing Quality** | Ultra High |
| **Character Gender** | Female |
| **Age Range** | 25–35 |
| **Lighting Mode** | Balanced |

Click **Generate** → Processing (~10 min).  
When finished, click **Apply to Character** → wait 2–3 min → click **Accept**.

### 3️⃣ Polish Face Sculpt
1. Switch to **Modify → Morph → Headshot 2 Sculpt Sliders**.  
2. Adjust subtle features:  
   - Jaw Roundness −5 to narrow profile  
   - Lip Thickness +3 (lower lip fuller)  
   - Eye Outer Tilt +1  
3. Bake Normals if prompted.  
4. Save as `Kelly_HS2_HD.ccProject`.

💡 **Tip:** If you ever obtain a multi-angle scan, reopen Headshot 2 → Mesh to 3D and wrap/bake onto this head for max pore detail.

---

## 🧭 Step 3 — Skin / Eyes / Hair (15–30 min)

### 1️⃣ High-Detail Skin
1. In **Modify → Material**, verify **Digital Human Shader** is active.  
2. Adjust roughness ≈ 0.45 → 0.40 for soft sheen.  
3. Subsurface Scattering (SSS) ≈ 0.25–0.30.  
4. Confirm 8K maps loaded (Base Color, Normal, AO, Roughness).  

### 2️⃣ Eyes & Lashes
- Content → `Actor → Eye → HD Eyes` → Apply.  
- Add **HD Lashes** package.  
- Tip: Catchlight will be added later in iClone.

### 3️⃣ Hair Selection
- Use `Actor → Hair → Female Long → Brown Gloss` (closest match).  
- Adjust specular to 0.25 for natural look.  

💾 Save progress → `Kelly_HS2_HD_SkinHair.ccProject`.

---

## 🧭 Step 4 — Send to iClone (15–30 min)

1. **File → Send Character to iClone.**  
   - Confirm transfer as “Digital Human (CC5 HD)”.  
2. In iClone, click `File → Save Project As` → `DirectorsChair_Template.iProject`.

### Create Director’s Chair Scene
1. Camera → Focal Length 85 mm.  
2. Enable DOF (focus on eyes).  
3. Lighting:  
   - Key Light (soft white, 45° right)  
   - Fill Light (−45°, half intensity)  
   - Rim Light (back warm tone) or neutral HDRI.  
4. Add Idle Motion: Animation → Idle → “Gentle Breathing + Blink”.  
5. Save as `Kelly_DirectorsChair.iProject`.

---

## 🧭 Step 5 — Lip-Sync with AccuLips (10–20 min)

1. Drag your ElevenLabs audio (`kelly25_audio.wav`) into timeline.  
2. Select Audio Track → Right-Click → `AccuLips → Generate Text`.  
3. Verify transcription accuracy and edit misheard words.  
4. Click **Apply to Viseme Track** → preview mouth motion.  
5. Optional: Import `.txt` or `.srt` for perfect timing.  
6. Save as `Kelly_LipSync_Test.iProject`.

---

## 🧭 Step 6 — Layer Facial Nuance from HeyGen Video (20–30 min)

1. **Plugins → AccuFACE → Video Mode.**  
2. Load your HeyGen reference video (“Kelly_talking.mp4”).  
3. Choose a neutral frame → click **Calibrate Neutral**.  
4. Open `Plugins → Motion LIVE`.  
5. In the Facial channel:  
   - Enable AccuFACE (Video).  
   - Disable mouth/jaw channels (so AccuLips drives them).  
   - Enable brow/lid/cheek/head.  
6. Click **Preview** → then **Record** once satisfied.  
7. Fine-tune in `Modify → Facial Control (HD)` for micro-adjustments.  
8. Save as `Kelly_Hybrid_FacialPass.iProject`.

🧠 **Why it works:**  
AccuFACE delivers natural brow/eye/head motion; AccuLips maintains perfect phoneme accuracy. CC5’s HD corrective morphs keep expressions lifelike without artifacting.

---

## 🧭 Step 7 — Render Test (5–10 min)

1. Menu → `Render → Render Video`.  
2. Settings:  
   - Format: H.264 MP4  
   - Resolution: 1920×1080 or 3840×2160  
   - Bitrate: 20 Mbps  
   - Frame Rate: 30 fps  
3. Output → `projects/Kelly/Renders/kelly.l1.short.v1.mp4`  
4. Click **Render** and preview result.

---

## ✅ Checklist Summary

| Phase | Output File | Duration |
|:--|:--|:--:|
| Base Face Creation | `Kelly_HD_Base.ccProject` | 10 min |
| Headshot 2 Projection | `Kelly_HS2_HD.ccProject` | 15 min |
| Skin/Eye/Hair Polish | `Kelly_HS2_HD_SkinHair.ccProject` | 20 min |
| Scene Setup in iClone | `DirectorsChair_Template.iProject` | 20 min |
| Lip-Sync Pass | `Kelly_LipSync_Test.iProject` | 15 min |
| AccuFACE Blend Pass | `Kelly_Hybrid_FacialPass.iProject` | 25 min |
| Render Output | `kelly.l1.short.v1.mp4` | 10 min |
