# 🚀 AGENT QUICK REFERENCE CARD
## One-Page Operational Guide

**Read this first. Then read `BOSS_OPERATIONAL_MANUAL.md` for details.**

---

## ⚡ THE PRIME DIRECTIVE

```
STOP → READ SPEC → CHECK IN-SPEC → IF NO → ESCALATE
```

**Never:**
- Create new things without approval
- Refactor unrelated code
- Skip testing
- Deploy without approval

---

## 🎯 CURRENT TASK

**Infrastructure Agent:** Complete Vercel Edge setup per `SETUP_VERCEL_EDGE_NOW.md`

**All Other Agents:** Stand by. Do not start new work.

---

## 🛡️ GUARDRAILS (All Agents)

| Rule | Action |
|------|--------|
| **Scope** | One task at a time. No "while I'm here" changes. |
| **Testing** | Test before moving on. Document failures. |
| **Deployment** | No production deploys without Boss approval. |
| **Cost** | No new paid services. Batch operations. Cache everything. |
| **Quality** | Lint/type checks pass. Tests pass. No breaking changes. |

---

## 👥 AGENT DOMAINS

| Agent | Domain | Current Task |
|-------|--------|--------------|
| **Infrastructure** | Vercel, Edge Config, Blob Storage, Deployment | Vercel Edge Setup |
| **Content** | Lessons, Assets, Database Content | Stand by |
| **Frontend** | `learn.html`, UI/UX, Client-side JS | Stand by |
| **Backend** | API endpoints, Serverless, Database | Stand by |
| **Quality** | Tests, Validation, Quality Gates | Monitor Infrastructure |

---

## 🚨 ESCALATION TRIGGERS

**STOP and escalate if:**
- Plan contradicts reality
- Missing credentials/config
- Stuck >10 min
- Tempted to create new thing
- Tempted to refactor unrelated code
- Cost implications unclear
- Production impact uncertain

**Format:**
```
🚨 ESCALATION REQUEST
Agent: [Role]
Task: [What]
Step: [Where]
Issue: [Problem]
Options: [A/B/C]
Question: [What you need]
```

---

## ✅ CHECKLIST (Before Starting)

- [ ] Read relevant plan document
- [ ] Verify you're in correct domain
- [ ] Check for existing solutions
- [ ] Test locally first
- [ ] Get Boss approval if needed

---

## 📞 REFERENCE

- **Boss Manual:** `BOSS_OPERATIONAL_MANUAL.md`
- **Current Task:** `SETUP_VERCEL_EDGE_NOW.md`
- **Operating Rules:** `CLAUDE.md`

---

**Remember:** Follow the Boss Manual. Escalate when stuck. One task at a time.


