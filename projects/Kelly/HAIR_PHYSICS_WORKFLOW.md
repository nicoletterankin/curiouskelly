# Kelly Hair Physics Workflow
## Natural & Weighted Hair Simulation for CC5/iClone8

### Overview
Kelly's hair physics preset provides realistic, weighted hair movement optimized for educational content and natural daily motion (not floaty cinematic hair). The preset uses a gradient weight map that locks roots and allows progressive freedom toward tips.

---

## 📦 Asset Files

**Location:** `projects/Kelly/CC5/HairPhysics/`

| File | Purpose | Details |
|------|---------|---------|
| `Kelly_Hair_Physics.json` | Main physics preset | SoftCloth simulation with tuned parameters |
| `Kelly_Hair_PhysicsMap.png` | Gradient weight map | Black (roots, locked) → White (tips, free) |
| `Fine_Strand_Noise.png` | Micro-detail texture | Secondary normal map for strand-level detail |
| `README.txt` | Import instructions | Quick reference guide |

---

## 🔧 Character Creator 5 — Import Process

### Step 1: Copy Files
```
Documents\Reallusion\Custom\HairPhysics\Kelly\
```
Or any preferred location. The JSON references PNGs by filename.

### Step 2: Load Preset
1. Open Kelly's CC5 project
2. Select **Hair mesh** in Scene Manager
3. Navigate to **Modify Panel → Physics**
4. Click **Load Preset**
5. Choose `Kelly_Hair_Physics.json`
6. If prompted, point to:
   - `Kelly_Hair_PhysicsMap.png` (weight map)
   - `Fine_Strand_Noise.png` (texture detail)

### Step 3: Preview & Test
- Press **Alt + Space** for real-time simulation preview
- Observe root lock and tip movement
- Check for natural sway and weight distribution

### Step 4: Fine-Tune Parameters

| Parameter | Default | Heavier Feel | More Movement |
|-----------|---------|--------------|---------------|
| **Damping** | 0.45 | 0.50–0.55 | 0.35–0.40 |
| **Elasticity** | 0.65 | 0.70–0.75 | 0.55–0.60 |
| **Air Resistance** | 0.08 | 0.10–0.12 | 0.04–0.06 |
| **Substeps** | 6 | 8 (smoother) | 4 (faster) |

---

## 🎬 iClone 8 — Production Use

### Step 1: Send to iClone
- In CC5: **Shift + F12** (Send to iClone)
- Hair physics preset transfers automatically

### Step 2: Enable Soft Cloth
1. Select character in iClone Scene Manager
2. Enable **Soft Cloth** for Hair mesh
3. Verify weight map is applied

### Step 3: Add Environmental Forces
**Wind Setup for Natural Daily Motion:**
```
Create → Wind
Direction: -10° horizontal, 5° vertical
Strength: 1.5
Turbulence: 0.08
Gust Frequency: 0.15 Hz
```

### Step 4: Cache Simulation
**Before Final Render:**
1. Set frame range (typically 0–180 for lesson segments)
2. **Animation → Soft Cloth → Bake Simulation**
3. Cache ensures consistent playback and faster rendering

---

## 🧬 Weight Map Logic

The `Kelly_Hair_PhysicsMap.png` uses a gradient approach:

| UV Region | Grayscale Value | Behavior |
|-----------|-----------------|----------|
| **Roots** (top 30% of UV) | 0–32 (black) | Fully locked to scalp |
| **Mid** (30%–80% of UV) | 32–224 (gradient) | Smooth falloff transition |
| **Tips** (bottom 20% of UV) | 224–255 (white) | Maximum freedom/movement |

**UV Orientation Note:**  
The map assumes hair UVs are oriented with **roots at top, tips at bottom**. If your model uses a different layout, rotate or flip the PNG in Photoshop/GIMP so black aligns with roots.

---

## 🔗 Integration with Existing Avatar Pipeline

This hair physics workflow integrates with Kelly's existing production pipeline:

### Pre-Requisite Steps:
1. ✅ **8K Textures** — From `demo_output/` (diffuse, alpha, normal)
2. ✅ **CC5 Character** — HD head created (see `TODO.md` step 1)
3. ✅ **iClone Scene** — Directors Chair template loaded

### Physics Workflow Position:
```
Character Creation (CC5)
    ↓
Hair Physics Application  ← YOU ARE HERE
    ↓
Send to iClone (Shift+F12)
    ↓
Voice Sync (AccuLips)
    ↓
Render & Export
```

### Related Files:
- **Avatar Textures:** `demo_output/kelly_*.png`
- **Physics Reference:** `demo_output/kelly_physics_reference_sheet.pdf`
- **Main Workflow:** `KELLY_AVATAR_WORKFLOW.md`
- **Todo Tracker:** `projects/Kelly/TODO.md`

---

## 📊 Technical Specifications

### Physics Parameters (from JSON):
```json
{
  "gravity": 9.81,
  "elasticity": 0.65,
  "damping": 0.45,
  "airResistance": 0.08,
  "selfCollision": true,
  "collisionMargin": 0.6,
  "substeps": 6,
  "solverIterations": 16
}
```

### Performance Notes:
- **Substeps: 6** provides good balance between accuracy and speed
- **Self-collision enabled** prevents hair strands from intersecting
- **Solver iterations: 16** ensures stable simulation at 30 FPS

### Material Overrides:
The preset also includes hair material enhancements:
- **Anisotropy:** 0.95 (strong directional specular)
- **Specular Tint:** 0.18 (subtle color tint)
- **Secondary Normal:** Fine strand noise at 0.25 strength
- **SSS Color:** [0.24, 0.12, 0.1] (warm subsurface scattering)
- **Flyaway Alpha:** 0.06 (micro-detail at edges)

---

## 🎯 Expected Results

### Natural Daily Motion:
- ✅ Roots stay locked to scalp (no floating hair)
- ✅ Mid-lengths respond to subtle head movements
- ✅ Tips show natural weight and inertia
- ✅ Gentle sway from ambient wind (not hyperactive)

### Use Cases:
- **Educational content** — non-distracting, realistic
- **Talking head lessons** — natural movement during speech
- **Character presentations** — polished, professional appearance

### Quality Validation:
Test Kelly's hair physics by:
1. **Head turn test** — Hair should lag naturally behind head rotation
2. **Settle test** — Hair should settle to rest position smoothly
3. **Wind test** — Tips react more than roots
4. **Collision test** — No hair penetrating shoulders/neck

---

## 🛠️ Troubleshooting

### Hair Floats Away from Scalp
- **Cause:** Weight map not loaded or inverted
- **Fix:** Reload `Kelly_Hair_PhysicsMap.png`, ensure black = roots

### Hair Too Stiff / Not Moving
- **Cause:** Damping too high or elasticity too high
- **Fix:** Lower damping to 0.35, lower elasticity to 0.55

### Hair Too Bouncy / Floaty
- **Cause:** Damping too low, gravity insufficient
- **Fix:** Raise damping to 0.55, verify gravity = 9.81

### Simulation Unstable / Jittery
- **Cause:** Substeps too low or collision margin too small
- **Fix:** Increase substeps to 8, increase collision margin to 0.8

### Hair Penetrates Shoulders/Neck
- **Cause:** Self-collision disabled or margin too small
- **Fix:** Enable self-collision, increase collision margin to 0.8–1.0

---

## 📅 Maintenance & Updates

### Version History:
- **v1.0** (Oct 12, 2025) — Initial preset pack release
  - Natural weighted physics
  - Gradient weight map
  - Fine strand noise texture

### Future Enhancements:
- [ ] Dynamic wind presets (calm/breezy/windy)
- [ ] Motion-specific caches (head turn left/right/nod)
- [ ] Fast-motion override (running/quick movements)

---

## 📚 Additional Resources

- **8K Avatar Guide:** `8K_PHOTOREALISTIC_AVATAR_GUIDE.md`
- **CC5 Lipsync Guide:** `CC5_LIPSYNC_TTS_GUIDE.md`
- **Kelly Workflow:** `KELLY_AVATAR_WORKFLOW.md`
- **Physics Reference PDF:** `demo_output/kelly_physics_reference_sheet.pdf`

---

**Last Updated:** October 12, 2025  
**Author:** Production Pipeline  
**Status:** ✅ Production Ready


