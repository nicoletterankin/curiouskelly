# 🛡️ Request Validator

**"Stop Before You Wreck"** — A communication discipline tool for working with your AI assistant.

## Why This Exists

You have patterns that cause havoc:
- Creating new things without checking if they already exist
- Using wacky/inconsistent naming
- Jumping to implementation without proper validation
- Not cross-referencing the extensive scaffolding docs

This tool forces discipline. It's a **gate** between your impulse and the AI taking action.

## How It Works

### Phase 1: Classify Your Request
Select what you're trying to do:
- 🔍 **Investigate/Understand** (LOW RISK) - Just learning, exploring
- 🔧 **Fix/Debug** (MEDIUM RISK) - Something's broken
- ✨ **Enhance Existing** (MEDIUM RISK) - Improving what exists
- 🆕 **Create New** (HIGH RISK) - Making something new
- 📝 **Rename/Move** (HIGH RISK) - Restructuring
- 🗑️ **Delete/Remove** (HIGH RISK) - Removing things

### Phase 2: Validate Against Existing Patterns
The tool checks your request against:
- Existing implementations in the codebase
- Naming conventions (SCREAMING_SNAKE for .md, kebab-case for .ts, snake_case for .py)
- Known dangerous patterns

### Phase 3: CLAUDE.md Checklist
The 7-point pre-flight checklist from CLAUDE.md:
1. ✅ Is there an approved plan reference?
2. ✅ Are languages/schemas precomputed?
3. ⚠️ Will this increase costs or touch production?
4. ✅ GPU/driver requirements satisfied?
5. ✅ Caching/batching in place?
6. ✅ Tests/linters passing?
7. ✅ Does this help daily use?

### Phase 4: Generate Brief
Once validated, generates a structured brief you paste to the AI.

## "Picky Nicky" Mode

For HIGH RISK actions (create, rename, delete), the tool activates **Picky Nicky** mode — named after the principle that the AI should protect you from breaking things you don't want to break.

In this mode, the AI will:
- Verify against existing patterns rigorously
- Check for duplicates
- Validate naming strictly
- Require explicit approval

## Usage

1. Open `tools/request-validator/index.html` in your browser
2. Select your request type
3. Describe what you want in one sentence
4. If creating something new, enter the proposed name
5. Complete the checklist
6. Reference the authoritative doc for this work
7. Generate the brief
8. Copy and paste the brief at the START of your message to the AI

## Example Flow

**Bad (before):**
> "Create a new script to generate thumbnails"

**Good (after):**
```
═══════════════════════════════════════════════════════════════════════
VALIDATED REQUEST BRIEF
Generated: 12/21/2025, 2:30:00 PM
═══════════════════════════════════════════════════════════════════════

📋 REQUEST SUMMARY
──────────────────────────────────────────────────────────────────────
Type: CREATE
Risk Level: HIGH
Description: Create a new script to generate lesson thumbnails with Kelly's face
Target: generate_lesson_thumbnails_v2.py

⚠️ VALIDATION NOTES
──────────────────────────────────────────────────────────────────────
EXISTING: Check generate_lesson_thumbnails.py, scripts/kelly-phase-visuals/ before proceeding

✅ CHECKLIST STATUS
──────────────────────────────────────────────────────────────────────
Reference: BUILD_PLAN.md → Phase 3 thumbnail generation
[✓] Approved plan reference
[ ] Languages/schemas validated
[✓] Cost/production impact reviewed
...

🎯 INSTRUCTIONS FOR AI
──────────────────────────────────────────────────────────────────────
1. VERIFY no existing implementation serves this purpose
2. Follow naming conventions strictly
...
═══════════════════════════════════════════════════════════════════════
```

## Integration with Mission Control

You can also access this from the Mission Control dashboard. The link is in the Quick Actions section.

## File Location

```
tools/
└── request-validator/
    ├── index.html    # The validation tool
    └── README.md     # This file
```

## Remember

Every time you want to do something that modifies the codebase:
1. **Stop** — Don't just tell the AI
2. **Validate** — Use this tool
3. **Brief** — Generate and paste the brief
4. **Proceed** — Now the AI has proper context

The 30 seconds this takes will save hours of cleanup from havoc.

