# Character Creator 5 + TTS Lipsync Integration Guide
## Complete Click-by-Click Workflow

**Status**: ✅ Kelly's Voice Model Ready (1.7GB trained model, 41.3 generations/sec)  
**Target**: Create photorealistic talking avatar with perfect lipsync  
**Tools**: Character Creator 5, Headshot 2, iClone 8, Kelly TTS Model

---

## 🎯 **PHASE 1: Generate TTS Audio for Lipsync**

### **Step 1.1: Generate Kelly's Voice Audio**
1. **Open Command Prompt** as Administrator
2. **Navigate to TTS directory**:
   ```
   cd C:\Users\user\UI-TARS-desktop\synthetic_tts
   ```
3. **Generate test audio for lipsync**:
   ```
   python generate_kelly25_samples.py
   ```
4. **Verify audio files created** in `demo_output/` folder:
   - `kelly_sample_1.wav` (5 seconds)
   - `kelly_sample_2.wav` (5 seconds)
   - `kelly_sample_3.wav` (5 seconds)
   - `kelly_sample_4.wav` (5 seconds)

### **Step 1.2: Create Custom Lipsync Audio**
1. **Open Kelly voice generator**:
   ```
   python synthesize_speech_enhanced.py
   ```
2. **Enter your custom text** (e.g., "Hello, I'm Kelly, your virtual teacher. Today we'll learn about science!")
3. **Save the generated audio** as `kelly_lipsync_audio.wav` in `projects/Kelly/Audio/`
4. **Verify audio quality**: Should be 22,050 Hz, mono, WAV format

---

## 🎨 **PHASE 2: Character Creator 5 Setup**

### **Step 2.1: Launch Character Creator 5**
1. **Click Start Menu** → Search "Character Creator 5"
2. **Click "Character Creator 5"** to launch
3. **Wait for application** to fully load (may take 30-60 seconds)

### **Step 2.2: Create New Project**
1. **Click "File"** in top menu bar
2. **Click "New Project"**
3. **Name your project**: "Kelly_Lipsync_Project"
4. **Click "OK"**

### **Step 2.3: Load Base Character**
1. **In Content Panel (left side)**, expand "Actor" folder
2. **Click "CC3+ Character"** folder
3. **Select "CC3_Base_Plus"** character
4. **Click "Apply"** button at bottom of Content panel
5. **Wait for character** to load in viewport

---

## 📸 **PHASE 3: Headshot 2 Integration**

### **Step 3.1: Access Headshot 2**
1. **In Character Creator 5**, look for **"Headshot 2"** tab in top menu
2. **Click "Headshot 2"** tab
3. **Wait for Headshot 2 interface** to load

### **Step 3.2: Import Your Headshot Photo**
1. **Click "Load Photo"** button
2. **Navigate to your headshot image** (the Kelly photo you uploaded)
3. **Select the image file**
4. **Click "Open"**

### **Step 3.3: Configure Headshot 2 Settings**
1. **Set "Quality"** to "High" (if available)
2. **Set "Gender"** to "Female"
3. **Set "Age Range"** to "25-35" (adjust based on your photo)
4. **Click "Generate"** button
5. **Wait for processing** (2-5 minutes depending on your GPU)

### **Step 3.4: Apply Generated Head to Character**
1. **Review the generated head** in preview
2. **Click "Apply to Character"** button
3. **Wait for head replacement** to complete
4. **Click "Accept"** when satisfied with result

---

## 🎭 **PHASE 4: Character Optimization for Lipsync**

### **Step 4.1: Switch to Modify Tab**
1. **Click "Modify" tab** in right panel
2. **Ensure "Visual" sub-tab** is selected

### **Step 4.2: Optimize Facial Features**
1. **In "Character" section**, set **SubD Levels**:
   - **Viewport**: Set to "2" or "3"
   - **Render**: Set to "3" or "4"
2. **Click "Subdivide"** button to increase mesh detail
3. **Wait for subdivision** to complete

### **Step 4.3: Enable Facial Animation**
1. **In "Character" section**, check **"Corrective Expressions"** checkbox
2. **Set "Auto-Blink"** to "None" (we'll control this manually)
3. **Ensure "Contact"** checkboxes are checked for both "Foot" and "Hand"

### **Step 4.4: Prepare for iClone Export**
1. **In "ACTORMIXER" section**, click **"Convert to Game Base"**
2. **Wait for conversion** to complete
3. **Click "Optimize and Decimate"** for better performance
4. **Wait for optimization** to complete

---

## 🎬 **PHASE 5: Export to iClone 8**

### **Step 5.1: Export Character**
1. **Click "File"** in top menu
2. **Click "Export"**
3. **Select "iClone Character"** from export options
4. **Name the file**: "Kelly_Lipsync_Character"
5. **Choose export location**: `projects/Kelly/iClone/`
6. **Click "Export"**
7. **Wait for export** to complete

### **Step 5.2: Launch iClone 8**
1. **Click Start Menu** → Search "iClone 8"
2. **Click "iClone 8"** to launch
3. **Wait for application** to load

---

## 🎤 **PHASE 6: iClone 8 Lipsync Setup**

### **Step 6.1: Load Character**
1. **In iClone 8**, click **"File"** → **"Import"**
2. **Navigate to** `projects/Kelly/iClone/`
3. **Select "Kelly_Lipsync_Character"** file
4. **Click "Open"**
5. **Wait for character** to load in scene

### **Step 6.2: Load Audio File**
1. **Click "Timeline"** tab at bottom of screen
2. **Right-click in audio track** area
3. **Select "Import Audio"**
4. **Navigate to** `projects/Kelly/Audio/kelly_lipsync_audio.wav`
5. **Select the audio file**
6. **Click "Open"**
7. **Audio should appear** in timeline

### **Step 6.3: Apply AccuLips**
1. **Select your character** in the viewport
2. **In "Modify" tab**, look for **"AccuLips"** section
3. **Click "AccuLips"** button
4. **In AccuLips dialog**:
   - **Audio Source**: Select your imported audio track
   - **Language**: Set to "English"
   - **Quality**: Set to "High"
   - **Click "Generate"**
5. **Wait for lipsync generation** (1-3 minutes)

### **Step 6.4: Fine-tune Lipsync**
1. **Play the timeline** to preview lipsync
2. **If needed, adjust**:
   - **Timing offset**: Move audio track slightly
   - **Sensitivity**: Adjust in AccuLips settings
   - **Expression intensity**: Modify in character settings
3. **Re-generate** if major adjustments needed

---

## 🎥 **PHASE 7: Render Test Video**

### **Step 7.1: Set Up Camera**
1. **In iClone 8**, position camera for good headshot view
2. **Adjust camera angle** to show character's face clearly
3. **Set camera to "Front View"** or "Three-Quarter View"

### **Step 7.2: Configure Render Settings**
1. **Click "Render"** tab
2. **Set Resolution**: 1920x1080 (Full HD)
3. **Set Frame Rate**: 30 FPS
4. **Set Quality**: High
5. **Set Format**: MP4 (H.264)

### **Step 7.3: Render Test Video**
1. **Click "Render"** button
2. **Name the file**: "Kelly_test_talk_v1"
3. **Choose location**: `projects/Kelly/Renders/`
4. **Click "Start Render"**
5. **Wait for render** to complete (5-15 minutes)

---

## 📊 **PHASE 8: Quality Control & Analytics**

### **Step 8.1: Review Rendered Video**
1. **Navigate to** `projects/Kelly/Renders/`
2. **Double-click "Kelly_test_talk_v1.mp4"**
3. **Watch the video** and check:
   - ✅ Lipsync accuracy
   - ✅ Facial expressions
   - ✅ Audio quality
   - ✅ Overall realism

### **Step 8.2: Run Analytics Scripts**
1. **Open Command Prompt** as Administrator
2. **Navigate to project root**:
   ```
   cd C:\Users\user\UI-TARS-desktop
   ```
3. **Run contact sheet generator**:
   ```
   .\scripts\20_contact_sheet.ps1
   ```
4. **Run frame metrics**:
   ```
   .\scripts\21_frame_metrics.ps1
   ```

### **Step 8.3: Review Analytics**
1. **Check contact sheet**: `analytics/Kelly/contact_sheet.png`
2. **Check frame metrics**: `analytics/Kelly/frame_metrics.csv`
3. **Verify quality metrics** are within acceptable ranges

---

## 🔧 **TROUBLESHOOTING GUIDE**

### **Common Issues & Solutions**

#### **Issue: Headshot 2 Not Available**
- **Solution**: Ensure Character Creator 5 is fully updated
- **Alternative**: Use ActorMIXER for character creation

#### **Issue: Poor Lipsync Quality**
- **Solution**: 
  1. Increase SubD levels in CC5
  2. Use higher quality audio (22,050 Hz minimum)
  3. Adjust AccuLips sensitivity settings

#### **Issue: Audio Not Playing in iClone**
- **Solution**:
  1. Check audio file format (WAV recommended)
  2. Ensure sample rate is 22,050 Hz or 44,100 Hz
  3. Verify audio track is properly imported

#### **Issue: Character Export Fails**
- **Solution**:
  1. Ensure character is fully loaded
  2. Check available disk space
  3. Try exporting without high SubD levels first

#### **Issue: Render Quality Poor**
- **Solution**:
  1. Increase render resolution
  2. Enable anti-aliasing
  3. Use higher bitrate settings

---

## 📈 **SUCCESS METRICS**

### **Technical Quality Checklist**
- [ ] **Audio Quality**: Clear, natural-sounding Kelly voice
- [ ] **Lipsync Accuracy**: 95%+ sync with audio
- [ ] **Facial Expressions**: Natural, realistic movements
- [ ] **Render Quality**: 1080p, smooth playback
- [ ] **Performance**: Real-time playback capability

### **Visual Quality Checklist**
- [ ] **Character Likeness**: Matches headshot photo
- [ ] **Facial Detail**: High-resolution mesh
- [ ] **Expression Range**: Natural emotion display
- [ ] **Lighting**: Professional lighting setup
- [ ] **Camera Work**: Good framing and angles

---

## 🚀 **NEXT STEPS**

### **Immediate Actions**
1. **Test with different audio samples**
2. **Experiment with different expressions**
3. **Create multiple character variations**
4. **Test with longer audio content**

### **Advanced Features**
1. **Add body animation** to character
2. **Implement real-time TTS integration**
3. **Create multiple camera angles**
4. **Add background environments**

### **Production Deployment**
1. **Batch process multiple characters**
2. **Create automated pipeline**
3. **Implement quality control systems**
4. **Scale to multiple languages**

---

## 📞 **SUPPORT & RESOURCES**

### **Documentation**
- **CC5 Manual**: Built-in help system
- **iClone 8 Guide**: Official documentation
- **TTS System**: `synthetic_tts/README.md`

### **File Locations**
- **Kelly Audio**: `projects/Kelly/Audio/`
- **Kelly Character**: `projects/Kelly/CC5/`
- **Kelly Renders**: `projects/Kelly/Renders/`
- **Analytics**: `analytics/Kelly/`

### **Backup Locations**
- **Character Files**: `iLearnStudio/projects/Kelly/`
- **Audio Samples**: `synthetic_tts/demo_output/`
- **Trained Models**: `synthetic_tts/kelly25_model_output/`

---

**🎉 Congratulations! You now have a complete lipsync avatar system with Kelly's trained voice! 🎉**

*This guide provides everything needed to create professional-quality talking avatars using your custom TTS system and Character Creator 5.*




















