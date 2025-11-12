# Kelly Video Generation Guide

## Goal
Create 6 video files showing Kelly teaching "Why Do Leaves Change Color?" with lipsync - one for each age bucket (2-5, 6-12, 13-17, 18-35, 36-60, 61-102).

## What You Have Ready ✅

- ✅ Audio files in `lesson-player/videos/audio/` (6 MP3 files with Kelly's voice)
- ✅ Hair physics preset in `projects/Kelly/CC5/HairPhysics/`
- ✅ Headshot reference: `projects/Kelly/Ref/headshot2-kelly-base169 101225.png`
- ✅ Director's Chair backgrounds (8K renders)
- ✅ Audio file: `projects/Kelly/Audio/kelly25_audio.wav`

## Complete Workflow

### PHASE 1: Create Kelly Avatar in CC5 (1 hour)

#### Step 1: Launch Character Creator 5
1. Open **Character Creator 5**
2. **File → New Project**
3. Save as: `projects/Kelly/CC5/Kelly_8K_Production.ccProject`

#### Step 2: Use Headshot 2
1. Go to **Headshot 2** tab
2. **Click "Load Photo"**
3. Navigate to: `projects/Kelly/Ref/headshot2-kelly-base169 101225.png`
4. Settings:
   - Resolution: **Ultra High**
   - Mesh Density: **Maximum**
   - Detail Level: **Maximum**
   - Quality: **Ultra High**
5. **Generate** (wait ~10 minutes)

#### Step 3: Apply to Character
1. Click **"Apply to Character"**
2. Wait 2-3 minutes
3. Click **"Accept"**

#### Step 4: Set Maximum SubD
1. Go to **Modify** tab
2. Find **SubD Levels** section
3. Set Viewport: **4** (maximum)
4. Set Render: **4** (maximum)
5. Click **"Subdivide"**
6. Wait 5-10 minutes

#### Step 5: Add Hair
1. Go to **Modify → Hair**
2. Browse Hair HD library
3. Choose **long wavy dark brown hair**
4. Apply to character

#### Step 6: Apply Hair Physics
1. Select hair mesh in scene
2. **Modify Panel → Physics**
3. **Load Preset** → select `projects/Kelly/CC5/HairPhysics/Kelly_Hair_Physics.json`
4. If prompted, point to:
   - Weight map: `Kelly_Hair_PhysicsMap.png`
   - Detail texture: `Fine_Strand_Noise.png`
5. Test with **Alt + Space** (preview simulation)

#### Step 7: Save Project
1. **Ctrl + S** to save
2. Verify file: `projects/Kelly/CC5/Kelly_8K_Production.ccProject`

---

### PHASE 2: Export to iClone (15 minutes)

#### Step 1: Send to iClone
1. In CC5: **File → Send Character to iClone**
2. Check these settings:
   - ✅ Export with Facial Profile
   - ✅ Export with Expression Wrinkle
   - ✅ Quality: Ultra High
   - ✅ Include Hair Physics
3. Click **"Send to iClone"**
4. Wait for iClone 8 to open (2-3 minutes)

#### Step 2: Import Director's Chair Scene
1. **File → Open Project**
2. Load: `projects/_Shared/iClone/DirectorsChair_Template.iProject`
   (If this doesn't exist, create a new scene with director's chair)

#### Step 3: Setup Camera & Lighting
1. **Camera Settings:**
   - FOV: 38
   - Position: Tight head close-up
   - Background: Black or director's chair

2. **Lighting:**
   - 3-point studio lighting
   - Soft shadows
   - Rim light for definition

3. **Save Scene** as: `projects/Kelly/iClone/Kelly_Talking_Scene.iProject`

---

### PHASE 3: Generate Lipsync for All 6 Age Variants (1.5 hours)

For EACH audio file (6 total):

#### For Age 2-5: kelly_leaves_2-5.mp3

1. **Import Audio:**
   - Copy `lesson-player/videos/audio/kelly_leaves_2-5.mp3` to `projects/Kelly/Audio/`
   - In iClone: Right-click audio track → Import Audio File
   - Select the MP3

2. **Run AccuLips:**
   - Select Kelly character
   - **Animation → Facial Animation → AccuLips**
   - Audio: select the track
   - Language: English
   - Quality: High
   - Click **"Apply"** (wait 1-3 min)

3. **Preview:**
   - Press SPACEBAR
   - Check mouth syncs with words

4. **Create Age-Appropriate Kelly:**
   - Use **Morph Editor** to adjust Kelly's apparent age:
     - Age 2-5: Larger eyes, rounder face, younger features
     - Age 6-12: Slightly more mature features
     - Age 13-17: Teen features
     - Age 18-35: Young adult (default)
     - Age 36-60: More mature, refined features
     - Age 61-102: Elderly features, softer skin

5. **Save Animation** as: `Kelly_leaves_2-5_animation.ibar`

6. **Repeat for all 6 age variants**

#### Audio Files to Process:
1. `kelly_leaves_2-5.mp3` → Youngest Kelly
2. `kelly_leaves_6-12.mp3` → Child Kelly
3. `kelly_leaves_13-17.mp3` → Teen Kelly
4. `kelly_leaves_18-35.mp3` → Young Adult Kelly (default)
5. `kelly_leaves_36-60.mp3` → Adult Kelly
6. `kelly_leaves_61-102.mp3` → Elder Kelly

---

### PHASE 4: Render Videos (30-60 minutes per video)

For EACH age variant:

1. **Set Render Settings:**
   - Resolution: 1920x1080 (1080p) or 3840x2160 (4K)
   - Frame Rate: 30 FPS
   - Quality: High
   - Format: MP4 (H.264)
   - Audio: Keep original

2. **Set Frame Range:**
   - Start Frame: 0
   - End Frame: Based on audio duration
   - Example: 10-second audio = 300 frames (30fps × 10s)

3. **Render Video:**
   - **File → Export Video**
   - Choose output location: `renders/Kelly/kelly_leaves_2-5.mp4`
   - Click **"Render"**
   - Wait 20-60 minutes depending on length

4. **Repeat for all 6 variants**

#### Output Files:
```
renders/Kelly/
├── kelly_leaves_2-5.mp4
├── kelly_leaves_6-12.mp4
├── kelly_leaves_13-17.mp4
├── kelly_leaves_18-35.mp4
├── kelly_leaves_36-60.mp4
└── kelly_leaves_61-102.mp4
```

---

### PHASE 5: Copy Videos to Lesson Player (5 minutes)

1. Copy all 6 video files to:
   ```
   lesson-player/videos/
   ```

2. Verify file structure:
   ```
   lesson-player/videos/
   ├── kelly_leaves_2-5.mp4
   ├── kelly_leaves_6-12.mp4
   ├── kelly_leaves_13-17.mp4
   ├── kelly_leaves_18-35.mp4
   ├── kelly_leaves_36-60.mp4
   └── kelly_leaves_61-102.mp4
   ```

---

## Testing the Complete Prototype

1. Open `lesson-player/index.html` in browser
2. Move age slider - videos should play instead of placeholders
3. Test audio plays synchronously with video
4. Verify age-appropriate content displays
5. Check teaching moments trigger correctly

---

## Time Breakdown

- **CC5 Creation**: 1 hour
- **iClone Setup**: 15 minutes
- **Lipsync (6 variants)**: 1.5 hours
- **Rendering (6 videos)**: 3-6 hours (background processing)
- **Integration**: 5 minutes

**Total Active Work**: ~4-5 hours  
**Total Time** (including render): ~8-10 hours

---

## Quick Start Checklist

Before starting:
- [ ] CC5 is installed and running
- [ ] iClone 8 is installed and running
- [ ] Headshot reference file exists
- [ ] Audio files are ready in `lesson-player/videos/audio/`
- [ ] Hair physics preset is ready

Start with:
1. **Open CC5** → Create new project
2. **Use Headshot 2** → Load photo
3. **Generate head** → Wait 10 min
4. **Apply to character** → Wait 2-3 min
5. **Add hair** → Browse and select
6. **Apply hair physics** → Load preset
7. **Send to iClone** → Wait for transfer

Then proceed with lipsync for each audio file.

---

## Troubleshooting

### Headshot 2 Generation Failed
- Use lower quality settings first
- Try different photo angles
- Check photo resolution (should be 4K+)

### Lipsync Not Working
- Check audio file format (MP3/WAV)
- Verify character has facial rig
- Try different AccuLips settings

### Video Rendering Slow
- Lower resolution for testing (720p)
- Disable ray tracing for faster renders
- Reduce frame rate for test renders

### Age Morphing Not Working
- Use Morph Editor in iClone
- Adjust eye size, face roundness, skin tone
- Save separate character files for each age variant

---

## Next Steps After Videos Are Done

1. **Test in lesson player** - Verify all 6 videos play correctly
2. **Optimize video files** - Compress for web delivery
3. **Add to CDN** - Upload to cloud storage
4. **Create more lessons** - Generate 2-3 more complete lessons
5. **Expand to 30 lessons** - One month of daily lessons

---

**Status**: Ready to execute  
**Start**: Open CC5 and follow Phase 1  
**Estimate**: 8-10 hours total (4-5 hours active work)

