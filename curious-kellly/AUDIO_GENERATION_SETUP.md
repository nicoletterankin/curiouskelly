# Audio Generation Setup - Quick Start

## 🎙️ **Generate Audio for Leaves and Water Lessons**

### **Step 1: Create `.env` File** (2 minutes)

**Windows:**
```powershell
# Navigate to backend
cd curious-kellly\backend

# Create .env file
notepad .env
```

**Paste this into `.env` and add your API key:**
```env
# OpenAI API
OPENAI_API_KEY=sk-proj-YOUR-ACTUAL-KEY-HERE

# Server
NODE_ENV=development
PORT=3000
LOG_LEVEL=debug
```

**Replace `sk-proj-YOUR-ACTUAL-KEY-HERE` with your real OpenAI API key!**

Then save and close Notepad.

---

### **Step 2: Install Content Tools Dependencies** (1 minute)

```powershell
cd ..\content-tools
npm install
```

This installs: `ajv`, `dotenv`, `openai`

---

### **Step 3: Generate Audio!** (5-10 minutes)

#### **Option A: Generate for ALL age groups** (Recommended)

```powershell
# Leaves lesson
node generate-audio.js ..\backend\config\lessons\leaves-change-color.json

# Water lesson  
node generate-audio.js ..\backend\config\lessons\water-cycle.json
```

**Output:** 
- Creates `curious-kellly/backend/config/audio/` folder
- Generates 3 audio files per age group (welcome, mainContent, wisdomMoment)
- 6 age groups × 3 files = 18 MP3 files per lesson

#### **Option B: Generate for ONE age group** (Faster testing)

```powershell
# Just adults (age 18-35)
node generate-audio.js ..\backend\config\lessons\leaves-change-color.json --age-group 18-35
```

**Output:**
- 3 MP3 files: `18-35-welcome.mp3`, `18-35-mainContent.mp3`, `18-35-wisdomMoment.mp3`

---

### **Step 4: Listen to Results!** (2 minutes)

```powershell
# Audio files are in:
cd ..\backend\config\audio\leaves-change-color
dir

# Open in default player:
start 18-35-welcome.mp3
```

**You should hear:** Kelly (age 27) saying the welcome message!

---

## 🎯 **Voice Mapping**

The generator automatically uses these OpenAI TTS voices:

| Kelly Age | Learner Age | Voice | Character |
|-----------|-------------|-------|-----------|
| 3 | 2-5 | nova | Warm, friendly |
| 9 | 6-12 | alloy | Balanced |
| 15 | 13-17 | echo | Natural |
| 27 | 18-35 | shimmer | Clear |
| 48 | 36-60 | onyx | Authoritative |
| 82 | 61-102 | fable | Expressive |

---

## 🐛 **Troubleshooting**

### Issue: "OpenAI API Error"

**Solution:** Check your API key is correct in `.env`:
```powershell
cd ..\backend
notepad .env
# Verify OPENAI_API_KEY=sk-proj-...
```

### Issue: "Cannot find module"

**Solution:** Install dependencies:
```powershell
cd ..\content-tools
npm install
```

### Issue: "File not found"

**Solution:** Check file path is correct:
```powershell
# Use relative path from content-tools directory
node generate-audio.js ..\backend\config\lessons\leaves-change-color.json
```

---

## 🎉 **Expected Results**

After running both lessons, you should have:

```
curious-kellly/backend/config/audio/
├── leaves-change-color/
│   ├── 2-5-welcome.mp3
│   ├── 2-5-mainContent.mp3
│   ├── 2-5-wisdomMoment.mp3
│   ├── 6-12-welcome.mp3
│   ├── 6-12-mainContent.mp3
│   ├── 6-12-wisdomMoment.mp3
│   ├── 13-17-welcome.mp3
│   ├── 13-17-mainContent.mp3
│   ├── 13-17-wisdomMoment.mp3
│   ├── 18-35-welcome.mp3
│   ├── 18-35-mainContent.mp3
│   ├── 18-35-wisdomMoment.mp3
│   ├── 36-60-welcome.mp3
│   ├── 36-60-mainContent.mp3
│   ├── 36-60-wisdomMoment.mp3
│   ├── 61-102-welcome.mp3
│   ├── 61-102-mainContent.mp3
│   └── 61-102-wisdomMoment.mp3
└── water-cycle/
    └── (same 18 files)
```

**Total:** 36 audio files (18 per lesson)

---

## 💡 **Cost Estimate**

**OpenAI TTS Pricing:** $15 per 1 million characters

**Leaves lesson:** ~5,000 characters total  
**Water lesson:** ~8,000 characters total  
**Total:** ~13,000 characters = **~$0.20** 💰

**Very affordable for testing!**

---

## ⏭️ **Next Steps**

After audio is generated:

1. **Listen to each age variant** - Are they age-appropriate?
2. **Test in Unity** - Sync with avatar lip-sync
3. **Test in Flutter** - Play during lesson
4. **Iterate on content** - Adjust text based on audio quality
5. **Generate for remaining 28 lessons** - After content is written

---

## 🚀 **Quick Commands**

```powershell
# Setup (one time)
cd curious-kellly\backend
notepad .env  # Add API key
cd ..\content-tools
npm install

# Generate audio
node generate-audio.js ..\backend\config\lessons\leaves-change-color.json
node generate-audio.js ..\backend\config\lessons\water-cycle.json

# Listen to results
cd ..\backend\config\audio\leaves-change-color
start 18-35-welcome.mp3
```

---

**🎉 Ready to hear Kelly's voice for the first time!** 🎙️

**Questions?** The audio generator will show progress and any errors.

**Want ElevenLabs instead?** Add `ELEVENLABS_API_KEY` to `.env` and it will use that automatically!















