# 🎯 Cursor Router

**Your AI Conversation Command Center** — Route every request to the right model, pipeline, and context.

## Why This Exists

You have patterns that cause havoc:
- Asking questions without knowing which model is best
- Not referencing the right docs for the task
- Forgetting to check existing implementations
- Starting work without proper context

**Cursor Router** fixes this by:
1. Routing to the optimal model based on task complexity
2. Auto-detecting relevant pipelines from your request
3. Pre-loading context docs before I start
4. Generating a structured prompt with everything I need

## How It Works

### Step 1: Select Your Task Type

| Task | Complexity | Recommended Model |
|------|------------|-------------------|
| 🔍 Understand | Low | Haiku 3.5 |
| 🐛 Debug | Medium | Sonnet 4 |
| 🔧 Fix Bug | Medium | Sonnet 4 |
| ✨ Enhance | Medium | Sonnet 4 |
| ⚡ Generate | High | Opus 4.5 |
| 🆕 Create New | High | Opus 4.5 |
| 🔄 Refactor | High | Opus 4.5 |
| 🚀 Deploy | High | Opus 4.5 |

### Step 2: Model Auto-Selection

The tool recommends a model based on:
- **Task complexity** (high-risk → Opus/GPT-5.2)
- **Detected pipeline** (video generation → Opus, simple fixes → Sonnet/GPT-4o)
- **Your constraints** (cost-sensitive → Haiku, need deep reasoning → o1)

#### Anthropic Models

| Model | Best For | Cost |
|-------|----------|------|
| 🧠 Opus 4.5 | Complex reasoning, architecture, multi-step tasks | $$$ |
| ⚡ Sonnet 4 | Balanced power and speed, standard development | $$ |
| 🌿 Haiku 3.5 | Quick lookups, simple fixes, fast responses | $ |

#### OpenAI Models

| Model | Best For | Cost |
|-------|----------|------|
| 🚀 GPT-5.2 | Advanced tool use, multimodal, latest capabilities | $$$ |
| 🔮 o1 / o1-pro | Deep reasoning, complex problem decomposition | $$$$ |
| 💚 GPT-4o | Fast multimodal, good at code, cost-effective | $$ |

### Step 3: Pipeline Detection

As you type your request, the tool detects keywords and shows relevant:
- **Docs** to read first
- **Scripts** to use
- **Existing implementations** to check

Example keyword mappings:
- "video" → HD Golden Lesson Pipeline
- "audio" → ElevenLabs/Audio generation
- "lesson" → Lesson Player architecture
- "deploy" → Deployment checklist
- "stripe" → Billing/payments

### Step 4: Context Flags

Quick toggles to add constraints:
- 💰 **Cost-sensitive** — Minimize API calls
- ✨ **Quality-critical** — Must be perfect
- ⏰ **Need it fast** — Time-sensitive
- 🚫 **Don't break existing** — Careful changes only
- ✅ **Ask before acting** — Get approval first

### Step 5: Generate Prompt

Click "Generate Cursor Prompt" to create a structured brief with:
- Task type and model recommendation
- Pipeline and relevant docs
- Validation notes (existing implementations)
- Constraints and pre-flight checks
- Task-specific instructions

**Copy and paste this at the START of your Cursor chat.**

## Model Selection Guide

### Anthropic Models

#### Use Opus 4.5 (🧠) for:
- Multi-file refactors
- Architecture decisions
- Complex debugging
- Creating new features
- Understanding complex systems
- Deployment workflows

#### Use Sonnet 4 (⚡) for:
- Standard bug fixes
- Code reviews
- Single-file changes
- Documentation updates
- API integrations
- UI adjustments

#### Use Haiku 3.5 (🌿) for:
- Quick questions
- Syntax lookups
- Simple formatting
- Single-line fixes
- Checking file locations

### OpenAI Models

#### Use GPT-5.2 (🚀) for:
- Tasks requiring advanced tool use
- Multimodal tasks (analyzing screenshots)
- Novel problems requiring latest capabilities
- When you want OpenAI's newest reasoning

#### Use o1 / o1-pro (🔮) for:
- Deep reasoning problems
- Complex debugging that needs step-by-step thinking
- Mathematical or logical proofs
- When you need the model to "think longer"

#### Use GPT-4o (💚) for:
- Fast iterative development
- Analyzing UI screenshots
- General coding tasks
- Cost-effective multimodal work

## Pipeline Reference

The router knows about these pipelines:

| Pipeline | Keywords | Primary Docs |
|----------|----------|--------------|
| Video | video, lipsync, motion | kelly-video-factory/ |
| Audio | audio, elevenlabs, tts | ELEVENLABS_OPTIMAL_SETUP.md |
| Lesson | lesson, phase, dna | LESSON_PLAYER_ARCHITECTURE.md |
| Player | player, learn.html | UI_AUDIT_2025-12-20.md |
| Social | social, twitter | SOCIAL_MEDIA_STRATEGY.md |
| Deploy | deploy, vercel, production | DEPLOYMENT_CHECKLIST.md |
| Billing | stripe, payment, checkout | billing/GLOBAL_ROADMAP.md |
| Database | supabase, database, api | SUPABASE_SCHEMA.md |
| Infographic | infographic, thumbnail | batch-infographics-from-db.ts |

## Integration

Access from:
- Direct: `tools/cursor-router/index.html`
- Mission Control: Purple "Cursor Router" button in Quick Actions

## Example Generated Prompt

```
╔══════════════════════════════════════════════════════════════════╗
║  CURSOR ROUTER — Validated Request Brief                         ║
╚══════════════════════════════════════════════════════════════════╝

🎯 TASK: Fix Bug
⚡ MODEL: Claude Sonnet 4 (Anthropic)
📁 PIPELINE: Lesson Player
💪 WHY THIS MODEL: Balanced speed/quality, Bug fixes

📚 RELEVANT CONTEXT (read these first):
────────────────────────────────────────────────────────────
  • LESSON_PLAYER_ARCHITECTURE.md
  • docs/UI_AUDIT_2025-12-20.md

🔧 RELEVANT SCRIPTS/FILES:
  • public/learn.html

⚠️ VALIDATION NOTES:
────────────────────────────────────────────────────────────
  CHECK EXISTING: public/learn.html, content/

🚦 CONSTRAINTS:
────────────────────────────────────────────────────────────
  • 🚫 Do not break existing functionality

═══════════════════════════════════════════════════════════════
REQUEST:
═══════════════════════════════════════════════════════════════

The video player isn't loading Kelly's response videos after the user 
selects an answer. The console shows a 404 for the video URL.

═══════════════════════════════════════════════════════════════

INSTRUCTIONS FOR AI:
────────────────────────────────────────────────────────────
1. Reproduce/understand the issue first
2. Check existing solutions in codebase
3. Fix root cause, not symptoms
4. Verify fix doesn't break other functionality

💡 CLAUDE SONNET 4 TIPS:
────────────────────────────────────────────────────────────
• Balanced speed and quality - good for most dev tasks
• Fast enough for iterative work
• Ask for explanations if the solution is complex

────────────────────────────────────────────────────────────
Generated by Cursor Router • 12/21/2025, 8:30:00 PM
```

## The Workflow

1. **Before every Cursor conversation:**
   - Open Cursor Router
   - Select task type
   - Describe your request
   - Check any constraints
   - Generate the prompt

2. **In Cursor:**
   - Start new chat
   - Paste the generated prompt
   - The AI now has full context

3. **Result:**
   - Right model for the job
   - Relevant docs pre-loaded
   - Existing implementations checked
   - Clear instructions
   - No havoc

## File Location

```
tools/
└── cursor-router/
    ├── index.html    # The router tool
    └── README.md     # This file
```

---

**Every conversation should start with a routed request. This is how we level up.**

