# Files to Upload to Claude.ai Project "The Daily Lesson"

## ⚠️ CRITICAL: Upload These Files First

### **Priority 1: Essential Files (Upload These Immediately)**

1. **`lessons/365_day_calendar.json`** ⭐⭐⭐ MOST IMPORTANT
   - **Why:** Claude's source of truth for all 365 lessons
   - **Size:** Large file (~5000+ lines)
   - **Action:** Upload to Claude.ai project "Files" tab

2. **`CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md`** ⭐⭐
   - **Why:** Complete topic selection guide
   - **Action:** Upload to Claude.ai project

3. **`content-agent-base/lesson-dna-schema.json`** ⭐⭐
   - **Why:** JSON schema for validation
   - **Action:** Upload to Claude.ai project

4. **`content-agent-base/lesson-template.json`** ⭐⭐
   - **Why:** Starting template for new lessons
   - **Action:** Upload to Claude.ai project

5. **`CLAUDE_DAILY_LESSON_EXPANDED_PROMPT.md`** ⭐⭐
   - **Why:** Complete role definition and workflow
   - **Action:** Copy content into Claude.ai project "Instructions" (not Files)

### **Priority 2: Reference Files (Upload for Examples)**

6. **`content-agent-base/the-sun-dna.json`** ⭐
   - **Why:** Complete example lesson
   - **Action:** Upload to Claude.ai project

7. **`lesson-player/balance-visual-prompts.json`** ⭐
   - **Why:** Example visual prompts structure
   - **Action:** Upload to Claude.ai project (if exists)

8. **`content-agent-base/CONTENT_AGENT_ONBOARDING.md`** ⭐
   - **Why:** Complete onboarding guide
   - **Action:** Upload to Claude.ai project

---

## 📋 Upload Checklist

- [ ] `lessons/365_day_calendar.json` uploaded
- [ ] `CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md` uploaded
- [ ] `content-agent-base/lesson-dna-schema.json` uploaded
- [ ] `content-agent-base/lesson-template.json` uploaded
- [ ] `content-agent-base/the-sun-dna.json` uploaded
- [ ] `CLAUDE_DAILY_LESSON_EXPANDED_PROMPT.md` copied to project instructions
- [ ] Test: Ask Claude "What lesson is on November 16?"
- [ ] Test: Ask Claude "List all files in this project"

---

## 🔍 File Locations in Codebase

**From workspace root:** `C:\Users\user\UI-TARS-desktop\`

- `lessons/365_day_calendar.json`
- `CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md`
- `CLAUDE_DAILY_LESSON_EXPANDED_PROMPT.md`
- `content-agent-base/lesson-dna-schema.json`
- `content-agent-base/lesson-template.json`
- `content-agent-base/the-sun-dna.json`
- `content-agent-base/CONTENT_AGENT_ONBOARDING.md`
- `lesson-player/balance-visual-prompts.json` (if exists)

---

## ✅ Verification Commands

After uploading, test Claude with:

1. **"List all files in this project"**
   - Should show uploaded files

2. **"What lesson is scheduled for November 16?"**
   - Should read from `365_day_calendar.json`
   - Should say: "Infinity - The Mathematics of the Endless" (Day 320)

3. **"Show me the structure of a lesson DNA file"**
   - Should reference `lesson-dna-schema.json` or `lesson-template.json`

4. **"What makes a good universal topic?"**
   - Should reference `CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md`

---

## 💾 How Claude Saves Files

**Claude.ai can save files directly:**

1. **Claude creates artifacts** in chat
2. **Claude saves to project** via "Files" tab
3. **You download** from Claude.ai project
4. **You save to codebase** at `lesson-player/`

**File naming Claude should use:**
- `{lesson-id}-dna.json`
- `{lesson-id}-visual-prompts.json`
- `{lesson-id}-knowledge-base.md`
- `{lesson-id}-asset-manifest.json`
- `{lesson-id}-teaching-moments.json`
- `{lesson-id}-interactive-specs.json`
- `{lesson-id}-animation-sequences.json`
- `{lesson-id}-export-package.md`

---

## 🚨 Troubleshooting

### Claude says "I don't see the 365-day calendar file"
**Solution:**
1. Verify file is uploaded to Claude.ai project "Files" tab
2. Check file name is exactly: `365_day_calendar.json`
3. Ask Claude: "List all files in this project" to verify
4. Re-upload if missing

### Claude references 30-day curriculum instead
**Solution:**
1. Verify `CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md` is uploaded
2. Verify project instructions include warning about 30-day curriculum
3. Remind Claude: "Use 365_day_calendar.json as source of truth"

### Files are too large to upload
**Solution:**
- `365_day_calendar.json` is large but Claude can read it
- If upload fails, try uploading specific day ranges OR
- Ask Claude to read specific days when needed

---

**Next Step:** Upload Priority 1 files to Claude.ai project "The Daily Lesson"








