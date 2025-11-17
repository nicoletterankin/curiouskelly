# Required Files for Content Management Agent

This document lists all files the content management agent needs access to in addition to `CONTENT_AGENT_ONBOARDING.md`.

---

## 📋 Essential Files (Required)

### 1. Template & Starting Point
- **`curious-kellly/content-tools/lesson-template.json`**
  - Starting template for new lessons
  - Contains complete structure with all 6 age variants
  - Copy this to create new lessons

### 2. Authoring Guide
- **`curious-kellly/content-tools/lesson-authoring-guide.md`**
  - Complete writing guide with examples for each age group
  - Topic selection criteria
  - 30-lesson curriculum plan
  - Writing tips and common pitfalls

### 3. Schema Definition
- **`curious-kellly/backend/config/lesson-dna-schema.json`**
  - JSON schema for validation
  - Defines required fields and structure
  - Referenced by validation tools

### 4. Validation Tools
- **`curious-kellly/content-tools/validate-lesson.js`**
  - Validates lesson JSON against schema
  - Checks content quality rules
  - Reports errors with line numbers
  - **Usage:** `node validate-lesson.js your-lesson.json`

- **`curious-kellly/content-tools/preview-lesson.js`**
  - Shows formatted lesson content for any age
  - Useful for reviewing before submission
  - **Usage:** `node preview-lesson.js your-lesson.json --age 35`

- **`curious-kellly/content-tools/generate-audio.js`**
  - Generates TTS audio via ElevenLabs/OpenAI
  - Optional but recommended
  - **Usage:** `node generate-audio.js your-lesson.json`

### 5. Example Lessons (Reference)
- **`lessons/the-sun-dna.json`**
  - Complete production example
  - Shows all 6 age variants with full content
  - Demonstrates proper structure and formatting
  - **Best reference for understanding DNA structure**

- **`curious-kellly/backend/config/lessons/the-sun-dna.json`**
  - Alternative location (may be duplicate)
  - Use as additional reference if needed

### 6. Tool Dependencies
- **`curious-kellly/content-tools/package.json`**
  - Node.js dependencies for validation/preview tools
  - Must run `npm install` in this directory before using tools
  - Contains required packages (ajv, etc.)

- **`curious-kellly/content-tools/package-lock.json`**
  - Lock file for consistent dependency versions
  - Generated automatically by npm

### 7. Quick Reference
- **`curious-kellly/content-tools/README.md`**
  - Quick start guide
  - Command reference
  - Workflow overview
  - 30-lesson curriculum list

---

## 📚 Additional Reference Files (Optional but Helpful)

### Documentation
- **`CURIOUS_KELLLY_EXECUTION_PLAN.md`**
  - Full project roadmap and architecture
  - Useful for understanding broader context

- **`LESSON_SYSTEM_EXPERTISE.md`**
  - Technical details about lesson player system
  - Interaction flow and phase progression
  - Helpful for understanding how lessons are consumed

- **`CLAUDE.md`**
  - Operating rules and constraints
  - Critical invariants and workflows
  - Safety rails and approvals

### Other Example Lessons (If Available)
- **`lessons/poetry-dna.json`**
- **`lessons/nutrition-science-dna.json`**
- **`lessons/negotiation-skills-dna.json`**
- **`lessons/molecular-biology-dna.json`**
- **`lessons/dance-expression-dna.json`**
- **`lessons/creative-writing-dna.json`**
- **`lessons/applied-mathematics-math-in-the-real-world-dna.json`**

*Note: These may have varying quality/completeness. Use `the-sun-dna.json` as the primary reference.*

---

## 🗂️ Directory Structure Summary

```
curious-kellly/
├── content-tools/
│   ├── lesson-template.json          ⭐ START HERE
│   ├── lesson-authoring-guide.md     ⭐ READ THIS
│   ├── validate-lesson.js            ⭐ VALIDATION TOOL
│   ├── preview-lesson.js             ⭐ PREVIEW TOOL
│   ├── generate-audio.js             ⭐ AUDIO TOOL
│   ├── package.json                  ⭐ DEPENDENCIES
│   ├── package-lock.json
│   └── README.md                     ⭐ QUICK REFERENCE
│
└── backend/
    └── config/
        └── lesson-dna-schema.json    ⭐ SCHEMA

lessons/
└── the-sun-dna.json                  ⭐ BEST EXAMPLE

CONTENT_AGENT_ONBOARDING.md           ⭐ ONBOARDING GUIDE
```

---

## 🚀 Setup Checklist

Before starting work, ensure the agent has:

1. ✅ **`CONTENT_AGENT_ONBOARDING.md`** - Read first
2. ✅ **`curious-kellly/content-tools/lesson-template.json`** - Copy to start
3. ✅ **`curious-kellly/content-tools/lesson-authoring-guide.md`** - Reference while writing
4. ✅ **`curious-kellly/backend/config/lesson-dna-schema.json`** - For validation
5. ✅ **`lessons/the-sun-dna.json`** - Study as example
6. ✅ **`curious-kellly/content-tools/validate-lesson.js`** - Run after writing
7. ✅ **`curious-kellly/content-tools/preview-lesson.js`** - Review output
8. ✅ **`curious-kellly/content-tools/package.json`** - Install dependencies

### Installation Steps

```bash
# Navigate to content-tools directory
cd curious-kellly/content-tools

# Install dependencies
npm install

# Verify tools work
node validate-lesson.js ../backend/config/lessons/the-sun-dna.json
```

---

## 📝 Workflow File Usage

### Creating a New Lesson

1. **Copy template:**
   ```bash
   cp curious-kellly/content-tools/lesson-template.json \
      curious-kellly/backend/config/lessons/your-topic.json
   ```

2. **Reference while writing:**
   - `lesson-authoring-guide.md` - Writing guidelines
   - `the-sun-dna.json` - Structure examples

3. **Validate:**
   ```bash
   node curious-kellly/content-tools/validate-lesson.js \
     curious-kellly/backend/config/lessons/your-topic.json
   ```

4. **Preview:**
   ```bash
   node curious-kellly/content-tools/preview-lesson.js \
     curious-kellly/backend/config/lessons/your-topic.json --age 35
   ```

5. **Generate audio (optional):**
   ```bash
   node curious-kellly/content-tools/generate-audio.js \
     curious-kellly/backend/config/lessons/your-topic.json
   ```

---

## ⚠️ Important Notes

- **Schema location:** Tools reference `../backend/config/lesson-dna-schema.json` relative to content-tools directory
- **Output location:** Save completed lessons to either:
  - `curious-kellly/backend/config/lessons/*-dna.json` (preferred)
  - `lessons/*-dna.json` (alternative)
- **Dependencies:** Must run `npm install` in `content-tools/` directory before using validation/preview tools
- **Example quality:** Use `the-sun-dna.json` as primary reference; other examples may be incomplete

---

**Total Essential Files: 8**  
**Total Optional Reference Files: 10+**




