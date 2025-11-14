# Quick Start Conversation Template

**Purpose**: Use this template to quickly align with your AI assistant on any task or question.

---

## 🎯 The 5-Minute Check-In

Copy and fill out this template when starting a new conversation:

```
IMMEDIATE GOAL:
[What do you want to accomplish right now?]

URGENCY:
[ ] Critical (today)
[ ] High (this week)
[ ] Medium (this sprint)
[ ] Low (exploring)

CURRENT STATE:
- Working: [What's functioning well?]
- Broken: [What's not working?]
- Tried: [What have you already attempted?]

PRIORITY:
[ ] P0 - Critical/blocking
[ ] P1 - Important
[ ] P2 - Nice to have

COMMUNICATION STYLE:
[ ] Direct & concise
[ ] Detailed explanations
[ ] Step-by-step walkthrough
[ ] Autonomous (just do it)

DECISION MAKING:
[ ] Ask first
[ ] Suggest & wait
[ ] Decide & inform

ADDITIONAL CONTEXT:
[Any files to review, constraints, or other info]
```

---

## 📋 Example Usage

### Example 1: Quick Fix
```
IMMEDIATE GOAL: Fix audio generation failing with ElevenLabs API

URGENCY: High (this week)

CURRENT STATE:
- Working: Lesson player renders correctly
- Broken: generate_lesson_audio_for_iclone.py throws 401 error
- Tried: Checked API key in .env, verified it's set correctly

PRIORITY: P0

COMMUNICATION STYLE: Direct & concise

DECISION MAKING: Decide & inform

ADDITIONAL CONTEXT: API key format might have changed, check ElevenLabs docs
```

### Example 2: New Feature
```
IMMEDIATE GOAL: Add Spanish translation support to PhaseDNA schema

URGENCY: Medium (this sprint)

CURRENT STATE:
- Working: English lessons render correctly
- Broken: No multilingual support yet
- Tried: Read CLAUDE.md about precomputed languages requirement

PRIORITY: P1

COMMUNICATION STYLE: Step-by-step walkthrough

DECISION MAKING: Suggest & wait

ADDITIONAL CONTEXT: Need to maintain backward compatibility with existing lessons
```

### Example 3: Understanding
```
IMMEDIATE GOAL: Understand how the Unity avatar integrates with Flutter app

URGENCY: Low (exploring)

CURRENT STATE:
- Working: Both Unity and Flutter apps run independently
- Broken: Don't understand the bridge/communication layer
- Tried: Looked at digital-kelly/apps/kelly_app_flutter/lib/bridge/

PRIORITY: P2

COMMUNICATION STYLE: Detailed explanations

DECISION MAKING: Ask first

ADDITIONAL CONTEXT: Planning to modify avatar rendering, need to understand architecture first
```

---

## 🔄 Daily Standup Format

For regular check-ins, use this even shorter format:

```
TODAY'S FOCUS: [One main thing]
BLOCKERS: [What's stopping you]
NEED HELP WITH: [Specific question or task]
CONTEXT: [Any relevant info]
```

---

## 🎨 Context-Aware Questions

Based on your project, here are common questions to consider:

### For Backend Tasks
- [ ] Is this a new endpoint or modifying existing?
- [ ] Does this need safety router integration?
- [ ] Are there rate limits or cost concerns?
- [ ] Does this affect session state management?

### For Mobile Tasks
- [ ] iOS, Android, or both?
- [ ] Does this need IAP integration?
- [ ] Unity bridge implications?
- [ ] Performance considerations (60fps target)?

### For Content Tasks
- [ ] Which age buckets (2-102)?
- [ ] Which languages (EN/ES/FR)?
- [ ] PhaseDNA schema compliance?
- [ ] Teaching moments included?

### For Avatar/Audio Tasks
- [ ] GPU available for Audio2Face?
- [ ] ElevenLabs vs local TTS?
- [ ] Viseme sync requirements?
- [ ] 60fps rendering target?

### For Deployment Tasks
- [ ] Which environment (staging/prod)?
- [ ] Cloudflare/Vercel/Netlify?
- [ ] Branch protection rules?
- [ ] Environment variables needed?

---

## 💡 Pro Tips

1. **Be Specific**: "Fix audio generation" → "Fix ElevenLabs 401 error in generate_lesson_audio_for_iclone.py"

2. **Share Context**: Mention relevant files, error messages, or previous attempts

3. **Set Expectations**: If you need step-by-step, say so. If you want autonomy, specify.

4. **Reference Documents**: Point to specific sections in CLAUDE.md, BUILD_PLAN.md, etc.

5. **Constraints Matter**: Mention cost, time, quality, or compliance concerns upfront

---

## 🚀 Quick Reference: Project Context

**What This Project Is:**
- Curious Kelly: Multi-platform AI learning companion
- Components: Web player, mobile apps (iOS/Android), Unity avatar, audio pipeline
- Goal: Production launch in 12 weeks

**Key Documents:**
- `CLAUDE.md` - Operating rules (READ FIRST)
- `START_HERE.md` - Onboarding guide
- `CURIOUS_KELLLY_EXECUTION_PLAN.md` - 12-week roadmap
- `TECHNICAL_ALIGNMENT_MATRIX.md` - Component mapping
- `BUILD_PLAN.md` - Prototype lineage

**Critical Rules:**
- Languages must be precomputed (EN/ES/FR) - no runtime generation
- Minimum 60 minutes training audio per voice model
- Never use browser TTS (prefer ElevenLabs)
- Never create new lesson players or pages
- Target 60 FPS for avatar rendering

**Common Commands:**
- Setup: `setup_local.ps1` → verify with `verify_installation.py`
- Audio: `generate_lesson_audio_for_iclone.py`
- Training: `gpu_optimized_trainer.py`
- Deployment: `deployment/setup-cloud.sh`

---

**Remember**: The better context you provide, the better assistance you'll receive! 🎯

