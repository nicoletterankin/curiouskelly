# Complete Claude.ai Automation Guide

## 🎯 Three Automation Solutions

I've created **three different approaches** for automating file downloads from Claude.ai:

---

## **Solution 1: API-Based Downloader** (Requires Claude API)

**Files:**
- `scripts/claude_file_downloader.py` - Direct API access
- `scripts/claude_file_downloader_alternative.py` - Messages API approach

**Status:** ⚠️ May need API endpoint adjustments based on Claude's actual API

**Best for:** Fully automated, scheduled downloads

---

## **Solution 2: Export File Processor** ⭐ **MOST RELIABLE**

**File:** `scripts/claude_export_helper.py`

**How it works:**
1. Export Claude conversation (Claude.ai → Export)
2. Run script on export file
3. Script extracts all lesson files automatically
4. Saves to `lesson-player/` directory

**Pros:**
- ✅ Works with Claude's current features
- ✅ No API limitations
- ✅ Reliable file extraction
- ✅ Can process multiple files at once

**Cons:**
- ⚠️ Requires manual export step (but can be automated with browser automation)

**Usage:**
```bash
# Export conversation from Claude.ai, then:
python scripts/claude_export_helper.py claude_export.txt

# Or paste text directly:
python scripts/claude_export_helper.py < paste_here.txt
```

---

## **Solution 3: Browser Automation** (Future Enhancement)

**Status:** Can be created if needed

**How it works:**
- Uses Selenium/Playwright to automate Claude.ai browser
- Monitors project files tab
- Automatically downloads new files
- Saves to codebase

**Best for:** Fully automated without API access

---

## 🚀 Recommended Workflow

### **Option A: Claude API (Fully Automated)** ⭐ **NEW**

1. **Get API Key** → [console.anthropic.com](https://console.anthropic.com)
2. **Set environment variable:**
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-api03-your-key"
   ```
3. **Run API downloader:**
   ```bash
   python scripts/claude_api_downloader.py
   ```
4. **Files automatically downloaded** → Saved to `lesson-player/`

**Time:** Fully automated, runs on schedule

### **Option B: Export Processor (Semi-Automated)**

1. **Claude creates lesson** → Files saved in Claude.ai project
2. **You export conversation** → Claude.ai → Export conversation
3. **Run export processor:**
   ```bash
   python scripts/claude_export_helper.py exported_conversation.txt
   ```
4. **Files automatically extracted** → Saved to `lesson-player/`

**Time:** ~30 seconds per lesson (mostly export time)

---

### **Option C: Scheduled API Automation**

1. **Set up API script** → Configure API key (see `CLAUDE_API_COMPLETE_GUIDE.md`)
2. **Schedule script** → Run every hour via Task Scheduler/cron
3. **Script automatically downloads** → New files appear in `lesson-player/`

**Time:** Fully automated, no manual steps

---

## 📋 Quick Start: Export Processor (Solution 2)

### **Step 1: Install Dependencies**

```bash
pip install -r requirements.txt
# Or just: pip install (no extra dependencies needed for export processor)
```

### **Step 2: Export Claude Conversation**

1. In Claude.ai, go to your project
2. Find the conversation where Claude created lesson files
3. Click **"Export"** or **"Download"** conversation
4. Save as text file (e.g., `claude_lesson_export.txt`)

### **Step 3: Run Script**

```bash
python scripts/claude_export_helper.py claude_lesson_export.txt
```

### **Step 4: Files Extracted**

Files are automatically saved to `lesson-player/` directory:
- `{lesson-id}-dna.json`
- `{lesson-id}-visual-prompts.json`
- `{lesson-id}-knowledge-base.md`
- etc.

---

## 🔄 Automation Options

### **A. Manual Export + Script** (Current)
- Export conversation manually
- Run script to extract files
- **Time:** ~1 minute per lesson

### **B. Browser Automation** (Can Create)
- Automate browser to export conversations
- Run script automatically
- **Time:** Fully automated

### **C. API Integration** (If Available)
- Direct API access to Claude project files
- Fully automated downloads
- **Time:** Fully automated

---

## 💡 Recommended Next Steps

1. **Try Solution 2 first** (Export processor) - Most reliable
2. **Test with one lesson** - Verify it works
3. **Set up automation** - If you want fully automated:
   - Try API approach (Solution 1)
   - Or request browser automation script (Solution 3)

---

## 🎯 Which Solution Should You Use?

- **Need reliability?** → Use Solution 2 (Export processor)
- **Want full automation?** → Try Solution 1 (API) or request Solution 3 (Browser)
- **Just starting?** → Use Solution 2, then upgrade to automation later

---

## 📝 File Organization

All solutions save files to:
```
lesson-player/
├── {lesson-id}-dna.json
├── {lesson-id}-visual-prompts.json
├── {lesson-id}-knowledge-base.md
├── {lesson-id}-asset-manifest.json
├── {lesson-id}-teaching-moments.json
├── {lesson-id}-interactive-specs.json
├── {lesson-id}-animation-sequences.json
└── {lesson-id}-export-package.md
```

---

## ✅ Success Checklist

- [ ] Export processor script works
- [ ] Files extracted correctly
- [ ] Files saved to correct directory
- [ ] No duplicates (state tracking works)
- [ ] Ready to automate further (if desired)

---

**Ready to automate!** Start with Solution 2 (export processor) - it's the most reliable and works immediately.

