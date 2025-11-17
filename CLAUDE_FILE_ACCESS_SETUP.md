# Claude.ai File Access & Artifact Saving Setup Guide

## 🎯 Goal
Configure Claude.ai to:
1. **Read** essential project files (365-day calendar, schemas, templates)
2. **Save** lesson artifacts it creates (JSON, markdown, manifests)
3. **Access** reference files for topic selection and validation

---

## 📁 Essential Files Claude Needs Access To

### **CRITICAL: Must Upload to Claude.ai Project**

#### 1. **365-Day Calendar** (PRIMARY SOURCE OF TRUTH)
- **File:** `lessons/365_day_calendar.json`
- **Why:** Claude needs this to know what lessons exist and what topics to create
- **Action:** Upload this file to Claude.ai project "The Daily Lesson"

#### 2. **Topic Selection Guide**
- **File:** `CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md`
- **Why:** Complete guide for selecting universal topics
- **Action:** Upload to Claude.ai project

#### 3. **Lesson DNA Schema**
- **File:** `content-agent-base/lesson-dna-schema.json`
- **Why:** Validation schema for lesson JSON structure
- **Action:** Upload to Claude.ai project

#### 4. **Lesson Template**
- **File:** `content-agent-base/lesson-template.json`
- **Why:** Starting template for new lessons
- **Action:** Upload to Claude.ai project

#### 5. **Content Agent Onboarding**
- **File:** `content-agent-base/CONTENT_AGENT_ONBOARDING.md`
- **Why:** Complete onboarding guide with workflow
- **Action:** Upload to Claude.ai project

#### 6. **Example Lesson** (Reference)
- **File:** `lessons/the-sun-dna.json` OR `content-agent-base/the-sun-dna.json`
- **Why:** Complete example lesson showing structure
- **Action:** Upload to Claude.ai project

#### 7. **Visual Prompts Example** (Reference)
- **File:** `lesson-player/balance-visual-prompts.json`
- **Why:** Example of visual asset prompt structure
- **Action:** Upload to Claude.ai project

---

## 🔧 Step-by-Step Setup Instructions

### **Step 1: Create/Open Claude.ai Project**

1. Go to [claude.ai](https://claude.ai)
2. Click **"Projects"** in left sidebar
3. Create new project OR open existing: **"The Daily Lesson"**
4. Project should be set to **"Opus 4.1"** model

### **Step 2: Upload Essential Files**

**Upload these files in this order:**

1. **`lessons/365_day_calendar.json`** ⭐ MOST IMPORTANT
   - This is Claude's source of truth for all lessons
   - Without this, Claude can't know what topics exist

2. **`CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md`**
   - Complete topic selection guide
   - Explains what topics are universal and why

3. **`content-agent-base/lesson-dna-schema.json`**
   - JSON schema for validation
   - Claude needs this to create valid lesson files

4. **`content-agent-base/lesson-template.json`**
   - Starting template
   - Claude can copy this structure

5. **`content-agent-base/CONTENT_AGENT_ONBOARDING.md`**
   - Complete workflow guide
   - Includes all steps and requirements

6. **`lessons/the-sun-dna.json`** (if exists) OR **`content-agent-base/the-sun-dna.json`**
   - Example lesson
   - Shows complete structure

7. **`lesson-player/balance-visual-prompts.json`**
   - Example visual prompts
   - Shows structure for visual asset generation

**How to Upload:**
- In Claude.ai project, click **"Files"** tab
- Click **"Upload"** or drag-and-drop files
- Files will appear in project file list

### **Step 3: Set Project Instructions**

**Copy the entire content of `CLAUDE_DAILY_LESSON_EXPANDED_PROMPT.md` into Claude.ai project instructions:**

1. In Claude.ai project, go to **"Project Settings"** or **"Instructions"**
2. Paste the full content of `CLAUDE_DAILY_LESSON_EXPANDED_PROMPT.md`
3. This gives Claude its role and workflow

### **Step 4: Configure File Saving**

**Claude.ai can save files directly to the project:**

When Claude creates artifacts, it will:
1. **Save files to Claude.ai project** (accessible via "Files" tab)
2. **You can download** files from Claude.ai project
3. **You can copy** file contents from Claude's responses

**File Naming Convention Claude Should Use:**
- `{lesson-id}-dna.json` - Lesson content
- `{lesson-id}-visual-prompts.json` - Visual generation prompts
- `{lesson-id}-knowledge-base.md` - Educational article
- `{lesson-id}-asset-manifest.json` - Asset inventory
- `{lesson-id}-teaching-moments.json` - Visual moment descriptions
- `{lesson-id}-interactive-specs.json` - Interactive component specs
- `{lesson-id}-animation-sequences.json` - Animation descriptions
- `{lesson-id}-export-package.md` - File structure guide

---

## 📋 Verification Checklist

After setup, verify Claude can:

- [ ] **Read** `365_day_calendar.json` (ask: "What lesson is on November 16?")
- [ ] **Understand** topic selection criteria (ask: "Is 'Clouds' a good universal topic?")
- [ ] **Access** schema for validation (ask: "What's the structure of a lesson DNA file?")
- [ ] **Reference** example lesson (ask: "Show me an example lesson structure")
- [ ] **Save** files it creates (ask Claude to create a test lesson and save it)

---

## 🔄 Workflow After Setup

### **When Claude Creates a Lesson:**

1. **Claude creates all artifacts** (JSON, markdown, manifests)
2. **Claude saves files** to Claude.ai project "Files" tab
3. **You download files** from Claude.ai project
4. **You save to codebase:**
   - `lesson-player/{lesson-id}-dna.json`
   - `lesson-player/{lesson-id}-visual-prompts.json`
   - `lesson-player/{lesson-id}-knowledge-base.md`
   - etc.

### **Alternative: Copy-Paste Workflow**

If file saving doesn't work:
1. Claude creates artifacts in chat response
2. You copy file contents from Claude's response
3. You create files in codebase manually
4. Files are ready for integration

---

## 🚨 Troubleshooting

### **Problem: Claude can't find 365_day_calendar.json**

**Solution:**
1. Verify file is uploaded to Claude.ai project
2. Check file name matches exactly: `365_day_calendar.json`
3. Ask Claude: "List all files in this project" to verify
4. Re-upload file if missing

### **Problem: Claude references 30-day curriculum instead**

**Solution:**
1. Verify `CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md` is uploaded
2. Verify project instructions include warning about 30-day curriculum
3. Remind Claude: "Use 365_day_calendar.json as source of truth, not the 30-day curriculum"

### **Problem: Claude can't save files**

**Solution:**
1. Check Claude.ai project has file upload permissions
2. Ask Claude to save files explicitly: "Save this as {filename}.json"
3. Use copy-paste workflow as backup
4. Check Claude.ai project storage limits

### **Problem: Files are too large**

**Solution:**
1. `365_day_calendar.json` might be large - Claude can still read it
2. If upload fails, split into smaller files OR
3. Provide specific day ranges when needed

---

## 📝 Quick Start Commands for Claude

After setup, test with these commands:

1. **"What lesson is scheduled for November 16?"**
   - Should reference `365_day_calendar.json`
   - Should give Day 320 lesson details

2. **"Is 'Clouds' a good universal topic? Why?"**
   - Should reference topic selection guide
   - Should explain universal topic criteria

3. **"Create a lesson for [topic] following the template"**
   - Should use `lesson-template.json`
   - Should validate against `lesson-dna-schema.json`
   - Should save all artifacts

4. **"List all files in this project"**
   - Should show uploaded files
   - Verifies file access

---

## ✅ Success Criteria

Setup is successful when:

1. ✅ Claude can read `365_day_calendar.json`
2. ✅ Claude references calendar, not 30-day curriculum
3. ✅ Claude can access schema and templates
4. ✅ Claude can save files it creates
5. ✅ Claude follows topic selection guidelines
6. ✅ Claude creates all required artifacts per lesson

---

## 🔗 File Locations Reference

**In Codebase:**
- `lessons/365_day_calendar.json` - Main calendar
- `content-agent-base/` - All templates and schemas
- `lesson-player/` - Where new lessons should be saved

**In Claude.ai Project:**
- Uploaded files accessible via "Files" tab
- Files Claude creates also saved here
- Download from Claude.ai to integrate into codebase

---

**Next Step:** Upload all essential files to Claude.ai project "The Daily Lesson" and test file access.




