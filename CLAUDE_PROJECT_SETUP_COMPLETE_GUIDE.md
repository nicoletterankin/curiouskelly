# Complete Claude.ai Project Setup Guide

## 🎯 Goal
Set up a brand new Claude.ai project "The Daily Lesson" with all files and instructions needed for Claude to create comprehensive lesson artifacts.

---

## 📋 Pre-Setup Checklist

Before starting, ensure you have access to these files in your codebase:

- [ ] `lessons/365_day_calendar.json` (source of truth)
- [ ] `CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md` (topic selection guide)
- [ ] `content-agent-base/lesson-dna-schema.json` (validation schema)
- [ ] `content-agent-base/lesson-template.json` (starting template)
- [ ] `content-agent-base/the-sun-dna.json` (example lesson)
- [ ] `content-agent-base/CONTENT_AGENT_ONBOARDING.md` (workflow guide)
- [ ] `CLAUDE_DAILY_LESSON_UNIFIED_PROMPT.md` (master prompt - just created)
- [ ] `lesson-player/balance-visual-prompts.json` (if exists - visual prompts example)

---

## 🚀 Step-by-Step Setup Instructions

### **STEP 1: Create Claude.ai Project**

1. **Open Claude.ai**
   - Go to [claude.ai](https://claude.ai)
   - Sign in to your account

2. **Create New Project**
   - Click **"Projects"** in the left sidebar
   - Click **"New Project"** button (or "+" icon)
   - Name it: **"The Daily Lesson"**
   - Set model to: **"Opus 4.1"** (or latest available)
   - Click **"Create"** or **"Save"**

3. **Verify Project Created**
   - You should see "The Daily Lesson" in your projects list
   - Click on it to open the project

---

### **STEP 2: Upload Essential Files**

**In Claude.ai project, click the "Files" tab (or "Upload Files" button):**

Upload these files **in this order** (drag-and-drop or click Upload):

#### **File 1: 365-Day Calendar** ⭐⭐⭐ MOST IMPORTANT
- **File:** `lessons/365_day_calendar.json`
- **Location:** `C:\Users\user\UI-TARS-desktop\lessons\365_day_calendar.json`
- **Why:** Claude's source of truth - without this, Claude can't know what lessons exist
- **Action:** Upload to Claude.ai project "Files" tab
- **Verify:** File appears in project file list

#### **File 2: Topic Selection Guide** ⭐⭐
- **File:** `CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md`
- **Location:** `C:\Users\user\UI-TARS-desktop\CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md`
- **Why:** Complete guide for selecting universal topics
- **Action:** Upload to Claude.ai project

#### **File 3: Lesson DNA Schema** ⭐⭐
- **File:** `content-agent-base/lesson-dna-schema.json`
- **Location:** `C:\Users\user\UI-TARS-desktop\content-agent-base\lesson-dna-schema.json`
- **Why:** JSON schema for validation - Claude needs this to create valid lesson files
- **Action:** Upload to Claude.ai project

#### **File 4: Lesson Template** ⭐⭐
- **File:** `content-agent-base/lesson-template.json`
- **Location:** `C:\Users\user\UI-TARS-desktop\content-agent-base\lesson-template.json`
- **Why:** Starting template for new lessons - Claude can copy this structure
- **Action:** Upload to Claude.ai project

#### **File 5: Example Lesson** ⭐
- **File:** `content-agent-base/the-sun-dna.json`
- **Location:** `C:\Users\user\UI-TARS-desktop\content-agent-base\the-sun-dna.json`
- **Why:** Complete example lesson showing structure
- **Action:** Upload to Claude.ai project

#### **File 6: Content Agent Onboarding** ⭐
- **File:** `content-agent-base/CONTENT_AGENT_ONBOARDING.md`
- **Location:** `C:\Users\user\UI-TARS-desktop\content-agent-base\CONTENT_AGENT_ONBOARDING.md`
- **Why:** Complete workflow guide with detailed steps
- **Action:** Upload to Claude.ai project

#### **File 7: Visual Prompts Example** ⭐ (Optional - if exists)
- **File:** `lesson-player/balance-visual-prompts.json`
- **Location:** `C:\Users\user\UI-TARS-desktop\lesson-player\balance-visual-prompts.json`
- **Why:** Example of visual asset prompt structure
- **Action:** Upload to Claude.ai project (if file exists)

**After uploading all files:**
- Verify all 7 files appear in the "Files" tab
- Check file names match exactly (case-sensitive)

---

### **STEP 3: Set Project Instructions**

**In Claude.ai project, go to "Instructions" or "Project Settings":**

1. **Open Unified Prompt File**
   - Navigate to: `C:\Users\user\UI-TARS-desktop\CLAUDE_DAILY_LESSON_UNIFIED_PROMPT.md`
   - Open in a text editor (VS Code, Notepad, etc.)

2. **Copy Entire Content**
   - Press `Ctrl+A` (select all)
   - Press `Ctrl+C` (copy)
   - This copies the entire unified prompt

3. **Paste into Claude.ai Instructions**
   - In Claude.ai project, find **"Instructions"** or **"Project Settings"** section
   - Click in the instructions text field
   - Press `Ctrl+V` (paste)
   - The entire unified prompt should now be in the instructions field

4. **Save Instructions**
   - Click **"Save"** or **"Update"** button
   - Instructions are now saved to the project

**What this does:**
- Gives Claude its complete role and workflow
- Defines all 8 artifact types to create
- Provides detailed DNA JSON creation steps
- Sets quality standards and validation requirements

---

### **STEP 4: Test File Access**

**In Claude.ai project chat, ask Claude these test questions:**

#### **Test 1: List Files**
**Ask Claude:**
```
List all files in this project
```

**Expected Response:**
Claude should list all 7 uploaded files:
- `365_day_calendar.json`
- `CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md`
- `lesson-dna-schema.json`
- `lesson-template.json`
- `the-sun-dna.json`
- `CONTENT_AGENT_ONBOARDING.md`
- `balance-visual-prompts.json` (if uploaded)

**If files are missing:** Re-upload missing files.

#### **Test 2: Read Calendar**
**Ask Claude:**
```
What lesson is scheduled for November 16?
```

**Expected Response:**
Claude should:
- Reference `365_day_calendar.json`
- Give Day 320 lesson details
- Show lesson title and details

**If Claude can't read calendar:** Verify `365_day_calendar.json` is uploaded correctly.

#### **Test 3: Understand Schema**
**Ask Claude:**
```
Show me the structure of a lesson DNA file
```

**Expected Response:**
Claude should:
- Reference `lesson-dna-schema.json` or `lesson-template.json`
- Show the structure with age variants, languages, interactions

**If Claude doesn't understand:** Verify schema and template files are uploaded.

#### **Test 4: Topic Selection**
**Ask Claude:**
```
What makes a good universal topic?
```

**Expected Response:**
Claude should:
- Reference `CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md`
- Explain universal topic criteria
- Give examples of good topics

**If Claude doesn't reference guide:** Verify topic selection guide is uploaded.

#### **Test 5: Verify Role**
**Ask Claude:**
```
What is your role in this project?
```

**Expected Response:**
Claude should describe itself as:
- Master lesson creator for "The Daily Lesson"
- Creates 8 types of artifacts per lesson:
  1. Lesson DNA JSON
  2. Visual Asset Prompts JSON
  3. Knowledge Base Articles
  4. Asset Manifests
  5. Teaching Moment Visualizations
  6. Interactive Element Specs
  7. Animation Sequences
  8. Export Package Structure
- Uses `365_day_calendar.json` as source of truth
- Creates comprehensive production-ready resources

**If Claude doesn't understand its role:** Verify unified prompt is in project instructions.

---

### **STEP 5: Verify Complete Setup**

**Final Verification Checklist:**

- [ ] Claude.ai project "The Daily Lesson" is created and open
- [ ] Model is set to Opus 4.1 (or latest)
- [ ] All 7 essential files uploaded to "Files" tab
- [ ] Unified prompt copied to "Instructions"
- [ ] Claude can list all files in project
- [ ] Claude can read November 16 lesson from calendar
- [ ] Claude understands lesson DNA structure
- [ ] Claude understands topic selection criteria
- [ ] Claude understands its expanded role (8 artifact types)

**If all checks pass:** ✅ **Setup is complete!**

---

## 🎯 Next Steps: Creating Your First Lesson

Once setup is complete, you can create lessons:

### **Example: Create a Lesson**

**Ask Claude:**
```
Create the lesson for [topic-name] from the 365-day calendar. Use the lesson ID and date from the calendar.
```

**Claude will:**
1. Check `365_day_calendar.json` for the topic
2. Create all 8 artifact types:
   - Lesson DNA JSON
   - Visual Asset Prompts JSON
   - Knowledge Base Article
   - Asset Manifest JSON
   - Teaching Moment Visualizations
   - Interactive Element Specs
   - Animation Sequences
   - Export Package Structure
3. Save all files to Claude.ai project "Files" tab

### **Download Files**

1. **In Claude.ai project, go to "Files" tab**
2. **Download files Claude created:**
   - `{lesson-id}-dna.json`
   - `{lesson-id}-visual-prompts.json`
   - `{lesson-id}-knowledge-base.md`
   - `{lesson-id}-asset-manifest.json`
   - `{lesson-id}-teaching-moments.json`
   - `{lesson-id}-interactive-specs.json`
   - `{lesson-id}-animation-sequences.json`
   - `{lesson-id}-export-package.md`

3. **Save to codebase:**
   - Place files in `lesson-player/` directory
   - Or follow the export package structure

---

## 🚨 Troubleshooting

### **Problem: Claude says "I don't see the 365-day calendar file"**

**Solution:**
1. Verify file is uploaded to Claude.ai project "Files" tab
2. Check file name is exactly: `365_day_calendar.json` (case-sensitive)
3. Ask Claude: "List all files in this project" to verify
4. Re-upload file if missing
5. Try asking: "Read the file `365_day_calendar.json`"

### **Problem: Claude references 30-day curriculum instead**

**Solution:**
1. Verify `CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md` is uploaded
2. Verify unified prompt is in project instructions (includes warning about 30-day curriculum)
3. Remind Claude: "Use `365_day_calendar.json` as source of truth, not the 30-day curriculum"
4. Re-copy unified prompt to instructions if needed

### **Problem: Claude doesn't create all 8 artifacts**

**Solution:**
1. Verify unified prompt is in project instructions
2. Remind Claude: "Create all 8 artifact types per lesson"
3. Reference the artifact creation workflow section
4. Ask: "What artifacts should you create for each lesson?" (should list all 8)

### **Problem: Files are too large to upload**

**Solution:**
1. `365_day_calendar.json` might be large - Claude can still read it
2. If upload fails, try:
   - Uploading specific day ranges OR
   - Providing specific days when needed OR
   - Splitting into smaller files (not recommended)

### **Problem: Claude can't save files**

**Solution:**
1. Check Claude.ai project has file upload permissions
2. Ask Claude to save files explicitly: "Save this as {filename}.json"
3. Use copy-paste workflow as backup:
   - Claude creates artifacts in chat response
   - You copy file contents from Claude's response
   - You create files in codebase manually

---

## 📚 Reference Documents

**Setup Guides:**
- `CLAUDE_AGENT_COMPARISON_AND_UNIFIED_SETUP.md` - Comparison of both agents
- `CLAUDE_DAILY_LESSON_UNIFIED_PROMPT.md` - Master prompt (copy to instructions)
- `CLAUDE_FILES_TO_UPLOAD.md` - File upload checklist
- `CLAUDE_SETUP_QUICK_START.md` - Quick setup guide
- `CLAUDE_FILE_ACCESS_SETUP.md` - Detailed setup guide

**Content Guides:**
- `CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md` - Topic selection guide
- `content-agent-base/CONTENT_AGENT_ONBOARDING.md` - Detailed workflow guide

**Original Agents:**
- `CLAUDE_DAILY_LESSON_EXPANDED_PROMPT.md` - Original expanded prompt
- `content-agent-base/CONTENT_AGENT_ONBOARDING.md` - Original onboarding guide

---

## ✅ Success Criteria

Setup is successful when:

1. ✅ Claude can read `365_day_calendar.json`
2. ✅ Claude references calendar, not 30-day curriculum
3. ✅ Claude can access schema and templates
4. ✅ Claude can save files it creates
5. ✅ Claude follows topic selection guidelines
6. ✅ Claude creates all 8 required artifacts per lesson
7. ✅ Claude understands its expanded role

---

## 🎉 You're Ready!

Once all steps are complete and tests pass, Claude is ready to create comprehensive, production-ready lessons with all 8 artifact types.

**Start creating:** Ask Claude to create a lesson from the 365-day calendar!

---

**Questions?** Refer to troubleshooting section or check reference documents.

