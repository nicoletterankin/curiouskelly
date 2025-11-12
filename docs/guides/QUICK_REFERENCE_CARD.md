# 🎯 Quick Reference Card - CC5 + TTS Lipsync

## ✅ **READY TO GO!**

Your Kelly voice model is **fully trained and working** (1.7GB model, 41.3 generations/sec)

---

## 🚀 **START HERE - 3 Simple Steps**

### **Step 1: Generate Audio** ⏱️ 2 minutes
```bash
cd C:\Users\user\UI-TARS-desktop\synthetic_tts
python generate_kelly25_samples.py
```
**Result**: 40 high-quality Kelly voice samples ready

### **Step 2: Launch Character Creator 5** ⏱️ 1 minute
1. **Start Menu** → "Character Creator 5"
2. **File** → "New Project" → Name: "Kelly_Lipsync"
3. **Content Panel** → "Actor" → "CC3+ Character" → "CC3_Base_Plus" → **Apply**

### **Step 3: Headshot 2 Integration** ⏱️ 5 minutes
1. **Click "Headshot 2" tab**
2. **Load Photo** → Select your Kelly headshot
3. **Set Quality: High, Gender: Female, Age: 25-35**
4. **Click "Generate"** → Wait 2-5 minutes
5. **Click "Apply to Character"** → **Accept**

---

## 🎬 **LIPSYNC SETUP - 4 Quick Steps**

### **Step 4: Optimize Character** ⏱️ 2 minutes
1. **Modify Tab** → **SubD Levels**: Viewport=2, Render=3
2. **Click "Subdivide"** → Wait for completion
3. **Check "Corrective Expressions"**
4. **ACTORMIXER** → "Convert to Game Base" → "Optimize and Decimate"

### **Step 5: Export to iClone** ⏱️ 1 minute
1. **File** → **Export** → **"iClone Character"**
2. **Name**: "Kelly_Lipsync_Character"
3. **Location**: `projects/Kelly/iClone/`
4. **Click "Export"**

### **Step 6: iClone 8 Setup** ⏱️ 3 minutes
1. **Launch iClone 8**
2. **File** → **Import** → Select "Kelly_Lipsync_Character"
3. **Timeline** → **Right-click Audio Track** → **Import Audio**
4. **Select**: `projects/Kelly/Audio/kelly_lipsync_audio.wav`

### **Step 7: Apply AccuLips** ⏱️ 2 minutes
1. **Select Character** → **Modify Tab** → **AccuLips**
2. **Audio Source**: Select imported audio
3. **Language**: English, **Quality**: High
4. **Click "Generate"** → Wait 1-3 minutes

---

## 🎥 **RENDER & TEST - 2 Final Steps**

### **Step 8: Render Test Video** ⏱️ 10 minutes
1. **Position Camera** for good headshot view
2. **Render Tab** → **Resolution**: 1920x1080, **Quality**: High
3. **Click "Render"** → **Name**: "Kelly_test_talk_v1"
4. **Location**: `projects/Kelly/Renders/`
5. **Wait for completion** (5-15 minutes)

### **Step 9: Quality Check** ⏱️ 2 minutes
1. **Open**: `projects/Kelly/Renders/Kelly_test_talk_v1.mp4`
2. **Check**: ✅ Lipsync accuracy, ✅ Facial expressions, ✅ Audio quality
3. **Run Analytics**: `.\scripts\20_contact_sheet.ps1` + `.\scripts\21_frame_metrics.ps1`

---

## 📁 **KEY FILE LOCATIONS**

| Purpose | Location |
|---------|----------|
| **Kelly Audio** | `projects/Kelly/Audio/kelly_lipsync_audio.wav` |
| **Kelly Character** | `projects/Kelly/CC5/` |
| **Kelly Renders** | `projects/Kelly/Renders/` |
| **Voice Samples** | `synthetic_tts/kelly25_voice_samples/` |
| **Trained Model** | `synthetic_tts/kelly25_model_output/best_model.pth` |

---

## 🔧 **TROUBLESHOOTING**

| Problem | Quick Fix |
|---------|-----------|
| **Headshot 2 not available** | Use ActorMIXER instead |
| **Poor lipsync** | Increase SubD levels, check audio quality |
| **Audio not playing** | Verify WAV format, 22,050 Hz sample rate |
| **Export fails** | Check disk space, try lower SubD first |
| **Poor render quality** | Increase resolution, enable anti-aliasing |

---

## 🎯 **SUCCESS CHECKLIST**

- [ ] **Audio**: Clear Kelly voice (✅ Ready)
- [ ] **Character**: Matches headshot photo
- [ ] **Lipsync**: 95%+ accuracy with audio
- [ ] **Expressions**: Natural facial movements
- [ ] **Render**: 1080p, smooth playback
- [ ] **Analytics**: Contact sheet + frame metrics generated

---

## ⚡ **NEXT LEVEL FEATURES**

### **Advanced Options**
- **Real-time TTS**: Integrate live voice generation
- **Multiple Characters**: Scale to 11+ avatars
- **Body Animation**: Add full-body movement
- **Multiple Angles**: Create multi-camera setups
- **Backgrounds**: Add professional environments

### **Production Pipeline**
- **Batch Processing**: Multiple characters at once
- **Quality Control**: Automated QC systems
- **Multi-language**: Support for ES/FR voices
- **Cloud Deployment**: Scalable infrastructure

---

## 📞 **SUPPORT**

- **Full Guide**: `CC5_LIPSYNC_TTS_GUIDE.md`
- **TTS System**: `synthetic_tts/README.md`
- **Analytics**: `analytics/Kelly/` folder
- **Backup**: `iLearnStudio/projects/Kelly/`

---

**🎉 You're ready to create professional talking avatars! 🎉**

*Total time: ~30 minutes from start to finished video*




















