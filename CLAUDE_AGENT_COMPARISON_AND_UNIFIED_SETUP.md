# Claude Agent Comparison & Unified Setup Guide

## 🔍 Analysis: Two Agent Systems

You've created two overlapping systems for Claude to write lessons. Here's what each does:

---

## **Agent #1: CLAUDE_DAILY_LESSON_EXPANDED_PROMPT.md**

**Purpose:** Comprehensive lesson production system for Claude.ai  
**Location:** Root directory  
**Use:** Copy entire content into Claude.ai project "Instructions"

### What It Does:
Creates **8 types of artifacts** per lesson:
1. ✅ **Lesson DNA JSON** (primary - same as Agent #2)
2. ✅ **Visual Asset Prompts JSON** (NEW - 150+ asset descriptions)
3. ✅ **Knowledge Base Articles** (NEW - educational deep-dives)
4. ✅ **Asset Manifests** (NEW - complete inventory)
5. ✅ **Teaching Moment Visualizations** (NEW - visual descriptions)
6. ✅ **Interactive Element Specs** (NEW - HTML5 components)
7. ✅ **Animation Sequences** (NEW - animation descriptions)
8. ✅ **Export Package Structure** (NEW - file organization)

### Scope:
- **Expanded role:** Not just JSON creator, but complete production system
- **Time per lesson:** ~12-15 hours (comprehensive)
- **Output:** Production-ready knowledge base resources

### Key Features:
- References `lessons/365_day_calendar.json` as source of truth
- Warns against using 30-day curriculum (example only)
- Creates visual generation prompts for images/3D models
- Produces educational articles for knowledge base
- Generates complete asset manifests

---

## **Agent #2: CONTENT_AGENT_ONBOARDING.md**

**Purpose:** Onboarding guide for content agents creating lesson DNA files  
**Location:** `content-agent-base/CONTENT_AGENT_ONBOARDING.md`  
**Use:** Reference guide (can be uploaded to Claude.ai project Files)

### What It Does:
Focuses on creating **Lesson DNA JSON files**:
1. ✅ **Lesson DNA JSON** (primary focus)
2. ❌ Visual prompts (not included)
3. ❌ Knowledge base articles (not included)
4. ❌ Other artifacts (not included)

### Scope:
- **Focused role:** Lesson DNA JSON creator
- **Time per lesson:** ~12-15 hours (for DNA JSON only)
- **Output:** Validated lesson DNA files

### Key Features:
- Detailed workflow for DNA JSON creation
- Age variant guidelines (6 ages: 2-5, 6-12, 13-17, 18-35, 36-60, 61-102)
- Multilingual requirements (EN/ES/FR)
- Quality standards and validation
- References `lessons/365_day_calendar.json` as source of truth
- Warns against using 30-day curriculum

---

## 📊 Comparison Table

| Feature | Agent #1 (Expanded) | Agent #2 (Onboarding) |
|---------|-------------------|---------------------|
| **Lesson DNA JSON** | ✅ Yes | ✅ Yes |
| **Visual Asset Prompts** | ✅ Yes | ❌ No |
| **Knowledge Base Articles** | ✅ Yes | ❌ No |
| **Asset Manifests** | ✅ Yes | ❌ No |
| **Teaching Moments** | ✅ Yes | ❌ No |
| **Interactive Specs** | ✅ Yes | ❌ No |
| **Animation Sequences** | ✅ Yes | ❌ No |
| **Export Packages** | ✅ Yes | ❌ No |
| **Age Variants (6)** | ✅ Yes | ✅ Yes |
| **Multilingual (EN/ES/FR)** | ✅ Yes | ✅ Yes |
| **365-Day Calendar Reference** | ✅ Yes | ✅ Yes |
| **Schema Validation** | ✅ Yes | ✅ Yes |
| **Time per Lesson** | ~12-15 hours | ~12-15 hours |
| **Use Case** | Complete production | DNA JSON only |

---

## 🎯 Recommendation: Use Agent #1 (Expanded)

**Why:** Agent #1 includes everything Agent #2 does, plus additional production artifacts. It's a superset.

**However:** Agent #2 has better detailed workflow explanations for DNA JSON creation that could enhance Agent #1.

---

## ✅ Unified Solution: Combined Master Prompt

I'll create a unified master prompt that:
1. ✅ Includes all 8 artifact types from Agent #1
2. ✅ Incorporates detailed DNA JSON workflow from Agent #2
3. ✅ Has clear file access requirements
4. ✅ Includes all necessary setup instructions

---

## 📁 Files Needed for Claude.ai Project

### **Priority 1: Essential Files (Upload to Claude.ai Project "Files" tab)**

1. **`lessons/365_day_calendar.json`** ⭐⭐⭐ MOST IMPORTANT
   - Claude's source of truth for all 365 lessons
   - Location: `C:\Users\user\UI-TARS-desktop\lessons\365_day_calendar.json`

2. **`CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md`** ⭐⭐
   - Complete topic selection guide
   - Location: `C:\Users\user\UI-TARS-desktop\CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md`

3. **`content-agent-base/lesson-dna-schema.json`** ⭐⭐
   - JSON schema for validation
   - Location: `C:\Users\user\UI-TARS-desktop\content-agent-base\lesson-dna-schema.json`

4. **`content-agent-base/lesson-template.json`** ⭐⭐
   - Starting template for new lessons
   - Location: `C:\Users\user\UI-TARS-desktop\content-agent-base\lesson-template.json`

5. **`content-agent-base/the-sun-dna.json`** ⭐
   - Complete example lesson
   - Location: `C:\Users\user\UI-TARS-desktop\content-agent-base\the-sun-dna.json`

6. **`content-agent-base/CONTENT_AGENT_ONBOARDING.md`** ⭐
   - Detailed workflow guide (reference)
   - Location: `C:\Users\user\UI-TARS-desktop\content-agent-base\CONTENT_AGENT_ONBOARDING.md`

7. **`lesson-player/balance-visual-prompts.json`** ⭐ (if exists)
   - Example visual prompts structure
   - Location: `C:\Users\user\UI-TARS-desktop\lesson-player\balance-visual-prompts.json`

### **Priority 2: Project Instructions (Copy to Claude.ai Project "Instructions")**

8. **`CLAUDE_DAILY_LESSON_UNIFIED_PROMPT.md`** ⭐⭐⭐
   - **NEW:** Unified master prompt (combines both agents)
   - Copy entire content into Claude.ai project "Instructions"
   - Location: `C:\Users\user\UI-TARS-desktop\CLAUDE_DAILY_LESSON_UNIFIED_PROMPT.md` (will be created)

---

## 🚀 Step-by-Step Setup Instructions

### **Step 1: Create Claude.ai Project**

1. Go to [claude.ai](https://claude.ai)
2. Click **"Projects"** in left sidebar
3. Click **"New Project"** or open existing **"The Daily Lesson"**
4. Set model to **"Opus 4.1"** (or latest available)
5. Name project: **"The Daily Lesson"**

### **Step 2: Upload Essential Files**

**In Claude.ai project, click "Files" tab:**

Upload these files in order (drag-and-drop or click Upload):

1. ✅ `lessons/365_day_calendar.json`
2. ✅ `CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md`
3. ✅ `content-agent-base/lesson-dna-schema.json`
4. ✅ `content-agent-base/lesson-template.json`
5. ✅ `content-agent-base/the-sun-dna.json`
6. ✅ `content-agent-base/CONTENT_AGENT_ONBOARDING.md`
7. ✅ `lesson-player/balance-visual-prompts.json` (if exists)

**Verify upload:** Ask Claude "List all files in this project" - should show all 7 files.

### **Step 3: Set Project Instructions**

**In Claude.ai project, go to "Instructions" or "Project Settings":**

1. Open `CLAUDE_DAILY_LESSON_UNIFIED_PROMPT.md` (will be created next)
2. Copy **ENTIRE content** (Ctrl+A, Ctrl+C)
3. Paste into Claude.ai project "Instructions" field
4. Save instructions

### **Step 4: Test File Access**

**Ask Claude these test questions:**

1. **"List all files in this project"**
   - Should show all 7 uploaded files

2. **"What lesson is scheduled for November 16?"**
   - Should reference `365_day_calendar.json`
   - Should give Day 320 lesson details

3. **"Show me the structure of a lesson DNA file"**
   - Should reference `lesson-dna-schema.json` or `lesson-template.json`

4. **"What makes a good universal topic?"**
   - Should reference `CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md`

### **Step 5: Verify Claude's Role**

**Ask Claude:**

**"What is your role in this project?"**

**Expected response:** Claude should describe itself as:
- Master lesson creator for "The Daily Lesson"
- Creates 8 types of artifacts per lesson
- Uses `365_day_calendar.json` as source of truth
- Creates comprehensive production-ready resources

---

## ✅ Success Checklist

- [ ] Claude.ai project "The Daily Lesson" created/opened
- [ ] Model set to Opus 4.1 (or latest)
- [ ] All 7 essential files uploaded to "Files" tab
- [ ] Unified prompt copied to "Instructions"
- [ ] Claude can list all files
- [ ] Claude can read November 16 lesson from calendar
- [ ] Claude understands its expanded role

---

## 🎯 Next Steps After Setup

Once setup is complete, you can:

1. **Create a lesson:** Ask Claude "Create the lesson for [topic] from the 365-day calendar"
2. **Claude will create:** All 8 artifact types automatically
3. **Download files:** Get files from Claude.ai project "Files" tab
4. **Save to codebase:** Place in `lesson-player/` directory

---

## 📝 File Naming Convention

Claude will create files with these names:
- `{lesson-id}-dna.json` - Lesson content
- `{lesson-id}-visual-prompts.json` - Visual generation prompts
- `{lesson-id}-knowledge-base.md` - Educational article
- `{lesson-id}-asset-manifest.json` - Asset inventory
- `{lesson-id}-teaching-moments.json` - Visual moment descriptions
- `{lesson-id}-interactive-specs.json` - Interactive component specs
- `{lesson-id}-animation-sequences.json` - Animation descriptions
- `{lesson-id}-export-package.md` - File structure guide

---

## 🚨 Troubleshooting

### Claude says "I don't see the 365-day calendar file"
**Solution:**
1. Verify file is uploaded to Claude.ai project "Files" tab
2. Check file name is exactly: `365_day_calendar.json`
3. Ask Claude: "List all files in this project"
4. Re-upload if missing

### Claude references 30-day curriculum instead
**Solution:**
1. Verify `CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md` is uploaded
2. Verify unified prompt includes warning about 30-day curriculum
3. Remind Claude: "Use 365_day_calendar.json as source of truth"

### Claude doesn't create all 8 artifacts
**Solution:**
1. Verify unified prompt is in project instructions
2. Remind Claude: "Create all 8 artifact types per lesson"
3. Reference the artifact creation workflow section

---

## 📚 Reference Documents

- `CLAUDE_DAILY_LESSON_EXPANDED_PROMPT.md` - Original expanded prompt
- `content-agent-base/CONTENT_AGENT_ONBOARDING.md` - Detailed workflow guide
- `CLAUDE_FILES_TO_UPLOAD.md` - File upload checklist
- `CLAUDE_SETUP_QUICK_START.md` - Quick setup guide
- `CLAUDE_FILE_ACCESS_SETUP.md` - Detailed setup guide

---

**Ready to proceed?** I'll now create the unified master prompt that combines both agents into one comprehensive system.

