# Claude.ai Setup Quick Start - Action Plan

## 🎯 Goal
Get Claude.ai project "The Daily Lesson" configured with file access so Claude can:
- ✅ Read the 365-day calendar
- ✅ Create lesson artifacts
- ✅ Save files back to the project

---

## ⚡ Quick Setup (5 Minutes)

### **Step 1: Open Claude.ai Project**
1. Go to [claude.ai](https://claude.ai)
2. Click **"Projects"** in left sidebar
3. Open project: **"The Daily Lesson"** (or create it)
4. Ensure model is set to **"Opus 4.1"**

### **Step 2: Upload Critical Files** (Do This First!)

**In Claude.ai project, go to "Files" tab and upload:**

1. **`lessons/365_day_calendar.json`** ⭐⭐⭐
   - **Location:** `C:\Users\user\UI-TARS-desktop\lessons\365_day_calendar.json`
   - **Why:** Claude's source of truth - without this, Claude can't know what lessons exist

2. **`CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md`** ⭐⭐
   - **Location:** `C:\Users\user\UI-TARS-desktop\CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md`
   - **Why:** Topic selection guide

3. **`content-agent-base/lesson-dna-schema.json`** ⭐⭐
   - **Location:** `C:\Users\user\UI-TARS-desktop\content-agent-base\lesson-dna-schema.json`
   - **Why:** JSON validation schema

4. **`content-agent-base/lesson-template.json`** ⭐⭐
   - **Location:** `C:\Users\user\UI-TARS-desktop\content-agent-base\lesson-template.json`
   - **Why:** Lesson template

### **Step 3: Set Project Instructions**

**In Claude.ai project, go to "Instructions" or "Project Settings":**

1. Copy **ENTIRE content** of: `CLAUDE_DAILY_LESSON_EXPANDED_PROMPT.md`
2. Paste into project instructions
3. This gives Claude its role and workflow

### **Step 4: Test File Access**

**Ask Claude:**
```
"List all files in this project"
```

**Should show:**
- `365_day_calendar.json`
- `CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md`
- `lesson-dna-schema.json`
- `lesson-template.json`

**Then ask:**
```
"What lesson is scheduled for November 16?"
```

**Should respond:**
- "Infinity - The Mathematics of the Endless" (Day 320)
- Should reference `365_day_calendar.json`

---

## ✅ Success Checklist

- [ ] Claude.ai project "The Daily Lesson" is open
- [ ] `365_day_calendar.json` uploaded to project Files
- [ ] `CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md` uploaded
- [ ] `lesson-dna-schema.json` uploaded
- [ ] `lesson-template.json` uploaded
- [ ] Project instructions contain `CLAUDE_DAILY_LESSON_EXPANDED_PROMPT.md` content
- [ ] Claude can list files in project
- [ ] Claude can read November 16 lesson from calendar

---

## 🚨 If Claude Can't See Files

**Problem:** Claude says "I don't see the 365-day calendar file"

**Solution:**
1. Verify file is in Claude.ai project "Files" tab
2. Check file name matches exactly: `365_day_calendar.json`
3. Ask Claude: "List all files in this project"
4. If file missing, re-upload it
5. If file exists but Claude can't see it, try:
   - "Read the file `365_day_calendar.json`"
   - "Open `365_day_calendar.json`"
   - Reference it explicitly: "In the file `365_day_calendar.json`, what is day 320?"

---

## 📁 Optional: Upload Reference Files

**These help Claude with examples:**

- `content-agent-base/the-sun-dna.json` - Example lesson
- `content-agent-base/CONTENT_AGENT_ONBOARDING.md` - Full guide
- `lesson-player/balance-visual-prompts.json` - Visual prompts example

---

## 💾 How Claude Saves Files

**When Claude creates artifacts:**

1. Claude creates files in chat response
2. Claude can save files to project "Files" tab
3. You download files from Claude.ai project
4. You save to codebase: `lesson-player/{lesson-id}-*.json`

**File naming:**
- `{lesson-id}-dna.json`
- `{lesson-id}-visual-prompts.json`
- `{lesson-id}-knowledge-base.md`
- etc.

---

## 📚 Reference Documents

**Full guides available:**
- `CLAUDE_FILE_ACCESS_SETUP.md` - Complete setup guide
- `CLAUDE_FILES_TO_UPLOAD.md` - File list with locations
- `CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md` - Topic selection guide
- `CLAUDE_DAILY_LESSON_EXPANDED_PROMPT.md` - Role definition

---

## 🎯 Next Steps After Setup

1. **Test:** Ask Claude "What lesson is on November 16?"
2. **Create:** Ask Claude "Create the lesson for November 16"
3. **Verify:** Claude should create all artifacts and save files
4. **Download:** Get files from Claude.ai project
5. **Integrate:** Save to codebase `lesson-player/` directory

---

**Ready!** Once files are uploaded and instructions are set, Claude can create lessons with full file access.








