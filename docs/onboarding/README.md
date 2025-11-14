# Onboarding Communication Tools

This directory contains tools and templates designed to help you communicate effectively with your AI assistant when working on the Curious Kelly project.

## 🎯 Purpose

When starting a new task or conversation, these tools help:
- **Align on goals** - What do you want to accomplish?
- **Share context** - What's working? What's broken?
- **Set expectations** - How do you want to work together?
- **Assess state** - What's the current project status?

---

## 📁 Files in This Directory

### 1. `QUICK_START_CONVERSATION.md`
**Best for**: Quick check-ins, daily standups, simple questions

A lightweight template you can copy/paste into any conversation. Takes 2-3 minutes to fill out.

**Usage**: Copy the template, fill it out, paste into your conversation with the AI assistant.

### 2. `onboarding-communication-tool.html` (in project root)
**Best for**: Comprehensive onboarding, complex tasks, first-time setup

An interactive HTML tool that guides you through a detailed questionnaire and generates a communication protocol.

**Usage**: 
1. Open `onboarding-communication-tool.html` in your browser
2. Fill out the form
3. Click "Generate Communication Protocol"
4. Copy the generated protocol and share it with your AI assistant

### 3. `tools/project_diagnostic.py`
**Best for**: Understanding current project state, identifying gaps

A Python script that analyzes your project structure and generates a diagnostic report.

**Usage**:
```bash
# Run diagnostic
python tools/project_diagnostic.py

# Save report to specific file
python tools/project_diagnostic.py --output my_report.json

# Don't save JSON (just print)
python tools/project_diagnostic.py --no-save
```

---

## 🚀 Quick Start Guide

### For Your First Conversation

1. **Run the diagnostic** to understand current state:
   ```bash
   python tools/project_diagnostic.py
   ```

2. **Open the interactive tool**:
   - Open `onboarding-communication-tool.html` in your browser
   - Fill out the form (takes ~5 minutes)
   - Copy the generated protocol

3. **Start your conversation** with the AI assistant:
   ```
   [Paste the generated protocol here]
   
   I'd like help with: [your specific task]
   ```

### For Daily Check-Ins

Use the quick template from `QUICK_START_CONVERSATION.md`:

```
IMMEDIATE GOAL: [What you want to do]
URGENCY: [Critical/High/Medium/Low]
CURRENT STATE: [What's working/broken]
PRIORITY: [P0/P1/P2]
COMMUNICATION STYLE: [Your preference]
```

### For Complex Tasks

1. Run diagnostic to assess current state
2. Use interactive tool to generate detailed protocol
3. Share protocol + specific task description with AI assistant

---

## 💡 When to Use Each Tool

| Scenario | Tool | Time Investment |
|----------|------|----------------|
| Quick question | Quick Start Template | 1-2 min |
| Daily standup | Quick Start Template | 2-3 min |
| New feature | Interactive Tool | 5-10 min |
| First-time setup | Diagnostic + Interactive Tool | 10-15 min |
| Debugging issue | Quick Start Template + Diagnostic | 5-8 min |
| Architecture question | Interactive Tool | 5-10 min |

---

## 📋 Example Workflows

### Example 1: "I need to fix audio generation"

```bash
# Step 1: Run diagnostic
python tools/project_diagnostic.py

# Step 2: Use quick template
IMMEDIATE GOAL: Fix ElevenLabs API 401 error in audio generation
URGENCY: High
CURRENT STATE: 
- Working: Lesson player renders
- Broken: generate_lesson_audio_for_iclone.py throws 401
- Tried: Checked .env file, API key looks correct
PRIORITY: P0
COMMUNICATION STYLE: Direct & concise
```

### Example 2: "I want to add a new lesson"

```bash
# Step 1: Check content status
python tools/project_diagnostic.py | grep -A 10 content

# Step 2: Use interactive tool for detailed protocol
# (Open onboarding-communication-tool.html, fill out form)

# Step 3: Share protocol + task
[Generated protocol]

I want to create a new lesson about "The Solar System" following the PhaseDNA schema.
```

### Example 3: "Help me understand the architecture"

```bash
# Step 1: Run full diagnostic
python tools/project_diagnostic.py --output architecture_report.json

# Step 2: Use interactive tool with "exploring" urgency
# (Fill out form with communication style = "Detailed explanations")

# Step 3: Share protocol + question
[Generated protocol]

I want to understand how the Unity avatar integrates with Flutter. 
Can you walk me through the bridge/communication layer?
```

---

## 🎨 Customization

### Adding Your Own Questions

Edit `onboarding-communication-tool.html` to add project-specific questions:

1. Find the relevant section (e.g., "SECTION 2: CURRENT STATE")
2. Add a new `question-group` div
3. Update the JavaScript `generateProtocol()` function to include your new field

### Extending the Diagnostic

Edit `tools/project_diagnostic.py` to add new checks:

1. Add a new `check_*()` method
2. Call it in `run_diagnostic()`
3. Update `generate_recommendations()` if needed

---

## 🔗 Related Documents

- `CLAUDE.md` - Operating rules for AI assistant
- `START_HERE.md` - Project onboarding guide
- `CURIOUS_KELLLY_EXECUTION_PLAN.md` - 12-week roadmap
- `TECHNICAL_ALIGNMENT_MATRIX.md` - Component mapping

---

## ❓ FAQ

**Q: Do I need to use these tools every time?**  
A: No! Use them when:
- Starting a new complex task
- First-time setup
- When communication isn't working well
- When you want to ensure alignment

**Q: Can I skip the diagnostic?**  
A: Yes, but it's helpful for understanding current state. The interactive tool and quick template work standalone.

**Q: What if I don't know my communication preferences?**  
A: Start with "Direct & concise" and "Suggest & wait" - you can always adjust based on what works.

**Q: Can I save the generated protocols?**  
A: Yes! Copy and save them for reference. The diagnostic tool saves JSON reports automatically.

---

## 🎯 Success Metrics

You'll know these tools are working when:
- ✅ AI assistant understands your context immediately
- ✅ Fewer back-and-forth clarification questions
- ✅ Solutions match your expectations
- ✅ Faster task completion

---

**Remember**: The goal is efficient, aligned collaboration. Use these tools as much or as little as needed! 🚀

