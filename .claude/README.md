## .claude – Claude Desktop Helper Folder

This folder mirrors the **exact files Claude needs** for lesson creation, gathered from different parts of the repo, so you can add everything to a Claude project in one click.

When you connect the GitHub repo `nicolettterankin/curiouskelly` to Claude Desktop and choose **“Add content from GitHub”**, simply select the **`.claude`** folder and you’ll get all of the core context files at once.

### Files in this folder

- `365_day_calendar.json` → **pointer** file that tells Claude where to find the full calendar (`lessons/365_day_calendar.json` in the repo). The full calendar is too large to duplicate here.
- `CLAUDE_365_DAY_CALENDAR_TOPIC_SELECTION.md` → copy of the topic selection guide
- `lesson-dna-schema.json` → copy of `content-agent-base/lesson-dna-schema.json`
- `lesson-template.json` → copy of `content-agent-base/lesson-template.json`
- `the-sun-dna.json` → copy of `content-agent-base/the-sun-dna.json`
- `CONTENT_AGENT_ONBOARDING.md` → copy of `content-agent-base/CONTENT_AGENT_ONBOARDING.md`
- `balance-visual-prompts.json` → copy of `lesson-player/balance-visual-prompts.json`
- `CLAUDE_DAILY_LESSON_UNIFIED_PROMPT.md` → master unified prompt for project instructions

> **Note:** Except for the calendar pointer, these are **copies** of the canonical files. The source of truth still lives in their original locations (`lessons/`, `content-agent-base/`, `lesson-player/`, and root). Keep them in sync if you make changes; this folder is just a convenience bundle for Claude Desktop.



