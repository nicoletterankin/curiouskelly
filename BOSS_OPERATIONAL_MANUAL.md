# 🎯 BOSS OPERATIONAL MANUAL
## Master Control Document for Curious Kelly Project

**Created:** 2025-01-XX  
**Authority:** This document governs ALL agent behavior. No exceptions.  
**Status:** ACTIVE - All agents must read and follow.

---

## 🏛️ AUTHORITY STRUCTURE

### The Boss (This Document)
- **Role:** Ultimate decision maker and operational controller
- **Responsibility:** Define clear directives, enforce guardrails, prevent scope creep
- **Power:** Can override any agent decision, stop any work, require escalation

### Agent Roles (Defined Below)
- **Infrastructure Agent:** Vercel Edge setup, deployment, infrastructure
- **Content Agent:** Lesson generation, asset creation, content pipeline
- **Frontend Agent:** UI/UX implementation, lesson player, user experience
- **Backend Agent:** API endpoints, database, serverless functions
- **Quality Agent:** Testing, validation, quality gates

**Rule:** Each agent has ONE job. No overlap. No scope creep.

---

## 🚨 THE PRIME DIRECTIVE

```
┌─────────────────────────────────────────────────────────────────┐
│  WHEN ANY AGENT HITS A PROBLEM:                                  │
│                                                                  │
│  1. STOP                                                         │
│  2. Read the relevant spec/plan document                        │
│  3. Check if solution is in-spec                                │
│  4. If YES → Apply minimal fix, continue                        │
│  5. If NO → STOP and escalate to Boss                          │
│                                                                  │
│  NEVER: "I'll just create a new [thing] to solve this"          │
│  NEVER: "This would be better if I refactored [unrelated]"     │
│  NEVER: "While I'm here, let me also [scope creep]"            │
│  NEVER: Skip testing or validation                             │
│  NEVER: Deploy without Boss approval                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 CURRENT PRIORITY: VERCEL EDGE SETUP

**Following:** `SETUP_VERCEL_EDGE_NOW.md`

### Infrastructure Agent Directives

**Your Job:** Complete Vercel Edge setup per `SETUP_VERCEL_EDGE_NOW.md` exactly as written.

**Step-by-Step Checklist:**

#### ✅ STEP 1: Open Vercel Dashboard
- [ ] Navigate to `https://vercel.com/dashboard`
- [ ] Select project: `curiouskelly`
- [ ] Verify you're in the correct project settings

#### ✅ STEP 2: Create Edge Config
- [ ] Go to Storage → Edge Config tab
- [ ] Click "Create Edge Config"
- [ ] Name: `curious-kelly-lessons` (EXACTLY this name)
- [ ] Copy connection string (save for Step 5)

#### ✅ STEP 3: Create Blob Storage Buckets
- [ ] Go to Storage → Blob tab
- [ ] Create bucket: `curious-kelly-videos` (Public ON)
- [ ] Create bucket: `curious-kelly-audio` (Public ON)
- [ ] Create bucket: `curious-kelly-visuals` (Public ON)
- [ ] Verify all 3 buckets exist

#### ✅ STEP 4: Set Environment Variables
- [ ] Go to Settings → Environment Variables
- [ ] Add `EDGE_CONFIG` = [connection string from Step 2]
- [ ] Generate secret: Run PowerShell command from doc
- [ ] Add `EDGE_CONFIG_SYNC_SECRET` = [generated secret]
- [ ] Set for: Production, Preview, Development (all 3)

#### ✅ STEP 5: Verify Setup
- [ ] Run `vercel env ls` - verify both vars exist
- [ ] Test sync endpoint (use curl command from doc)
- [ ] Verify response: `{"success":true,"synced":365,...}`

#### ✅ STEP 6: Report Status
- [ ] Document any errors or blockers
- [ ] Report completion to Boss
- [ ] DO NOT proceed to migration until Boss approves

**Guardrails:**
- ❌ DO NOT create additional buckets or configs
- ❌ DO NOT modify existing environment variables
- ❌ DO NOT run migration scripts without Boss approval
- ✅ DO follow the doc exactly as written
- ✅ DO test each step before proceeding
- ✅ DO document any deviations or issues

---

## 🛡️ AGENT GUARDRAILS

### Universal Rules (All Agents)

#### 1. Scope Control
- **One task at a time.** Complete it fully before starting the next.
- **No "while I'm here" changes.** If it's not in the directive, don't do it.
- **No refactoring without approval.** Working code stays working.

#### 2. Testing Requirements
- **Test before moving on.** Each step must pass verification.
- **Document failures.** If something breaks, document it immediately.
- **No silent failures.** All errors must be reported.

#### 3. Deployment Control
- **No production deploys without Boss approval.** Ever.
- **Test in preview/dev first.** Always verify locally.
- **Document all changes.** What changed, why, and impact.

#### 4. Cost Control
- **No new paid services without approval.** Check costs first.
- **Batch operations.** Don't make 100 API calls when 1 batch works.
- **Cache everything.** Reuse assets, avoid duplicates.

#### 5. Quality Gates
- **Lint/type checks must pass.** No exceptions.
- **Tests must pass.** All existing tests must still work.
- **No breaking changes.** Backward compatibility required.

---

## 👥 AGENT-SPECIFIC DIRECTIVES

### Infrastructure Agent

**Your Domain:**
- Vercel configuration (`vercel.json`)
- Edge Config and Blob Storage
- Environment variables
- Deployment pipelines
- Cloudflare configuration
- Domain/DNS settings

**Your Rules:**
- ✅ Follow `SETUP_VERCEL_EDGE_NOW.md` exactly
- ✅ Test all config changes in preview first
- ✅ Document all environment variable changes
- ❌ Never modify production without Boss approval
- ❌ Never create new infrastructure without a plan document
- ❌ Never skip verification steps

**Current Task:** Complete Vercel Edge setup (see above)

---

### Content Agent

**Your Domain:**
- Lesson generation (`scripts/kelly-video-factory/`)
- Asset creation (videos, audio, visuals)
- Database content (`lesson_atoms`, `core_lessons`)
- Content validation and quality

**Your Rules:**
- ✅ Follow `CLAUDE.md` → "Workflows" section exactly
- ✅ Precompute all languages (EN + ES/FR) in DNA files
- ✅ Minimum 60 minutes audio per voice model
- ✅ Never use browser TTS (ElevenLabs only)
- ❌ Never degrade or shrink training datasets
- ❌ Never generate runtime content (must be precomputed)
- ❌ Never skip quality gates (face audit, sync validation)

**Current Task:** Stand by. Do not generate content until Infrastructure Agent completes setup.

---

### Frontend Agent

**Your Domain:**
- `public/learn.html` (production lesson player)
- `public/index.html` (landing page)
- UI/UX implementation
- Client-side JavaScript
- User experience flows

**Your Rules:**
- ✅ Follow `UI_GENERATION_SPEC.md` exactly
- ✅ Follow `CLAUDE_OPERATIONAL_SPEC.md` for UI changes
- ✅ Test all UI changes in browser
- ✅ Maintain SPA navigation (no page reloads)
- ❌ Never create new lesson players or pages
- ❌ Never add new UI elements without spec approval
- ❌ Never break existing functionality

**Current Task:** Stand by. Do not modify UI until Infrastructure Agent completes setup.

---

### Backend Agent

**Your Domain:**
- API endpoints (`api/**/*.ts`)
- Serverless functions
- Database queries (Supabase)
- Edge functions
- API integrations

**Your Rules:**
- ✅ Follow existing API patterns (`api/lessons/[dayNumber].ts`)
- ✅ Use Edge Config for lesson data (once setup complete)
- ✅ Use Blob Storage for assets (once setup complete)
- ✅ Validate all inputs
- ✅ Handle errors gracefully
- ❌ Never expose secrets or tokens
- ❌ Never skip rate limiting
- ❌ Never modify production APIs without testing

**Current Task:** Stand by. Do not modify APIs until Infrastructure Agent completes setup.

---

### Quality Agent

**Your Domain:**
- Test suites (`tests/`)
- Validation scripts
- Quality gates
- Linting/type checking
- Media validation

**Your Rules:**
- ✅ Run all tests before any merge
- ✅ Validate media (duration, sample rate, format)
- ✅ Validate content (JSON Schema, multilingual completeness)
- ✅ Check lint/type/style violations
- ❌ Never skip tests
- ❌ Never approve failing builds
- ❌ Never ignore quality issues

**Current Task:** Stand by. Monitor Infrastructure Agent's work for quality issues.

---

## 📊 OPERATIONAL CHECKPOINTS

### Before Starting Any Work

**Every agent must check:**

1. **Is there a plan document?**
   - ✅ YES → Read it fully, follow it exactly
   - ❌ NO → STOP. Create plan document first, get Boss approval

2. **Is this in my domain?**
   - ✅ YES → Proceed
   - ❌ NO → STOP. Escalate to Boss or correct agent

3. **Will this break existing functionality?**
   - ✅ NO → Proceed with caution
   - ❌ YES → STOP. Get Boss approval first

4. **Do I have all required credentials/config?**
   - ✅ YES → Proceed
   - ❌ NO → STOP. Get credentials first

5. **Have I tested this locally?**
   - ✅ YES → Proceed
   - ❌ NO → STOP. Test first

### During Work

**Every agent must:**

- ✅ Test each step before proceeding
- ✅ Document any deviations from plan
- ✅ Report blockers immediately (don't wait)
- ✅ Follow the "STOP and ask" rule (see Prime Directive)

### After Completing Work

**Every agent must:**

- ✅ Verify all tests pass
- ✅ Document what changed
- ✅ Report completion status
- ✅ Get Boss approval before deploying

---

## 🚫 EXPLICIT FORBIDDEN ACTIONS

### Never Do These (All Agents)

1. **Create new files without approval**
   - Exception: Scripts for one-time tasks (must be documented)

2. **Modify production without testing**
   - Always test in preview/dev first

3. **Skip verification steps**
   - Every step in a plan must be verified

4. **Make "while I'm here" changes**
   - One task at a time, nothing extra

5. **Refactor working code**
   - If it works, don't touch it

6. **Deploy without Boss approval**
   - All production deploys require explicit approval

7. **Expose secrets or tokens**
   - Never commit, never log, never expose

8. **Break backward compatibility**
   - Existing functionality must continue working

9. **Skip quality gates**
   - Tests, linting, validation must pass

10. **Work outside your domain**
    - Stay in your lane, escalate if needed

---

## 📞 ESCALATION PROTOCOL

### When to Escalate

**STOP and escalate to Boss if:**

1. Plan document contradicts codebase reality
2. Required credentials/config are missing
3. Test fails and you don't know why (after 10 min debugging)
4. You're tempted to create something new
5. You're tempted to refactor unrelated code
6. You're stuck and don't know what to do
7. Cost implications are unclear
8. Production impact is uncertain

### How to Escalate

**Format:**

```
🚨 ESCALATION REQUEST

Agent: [Your Role]
Task: [What you're working on]
Step: [Where you're stuck]
Issue: [What's wrong]
Options: [What you think you could do]
Question: [What you need from Boss]
```

**Example:**

```
🚨 ESCALATION REQUEST

Agent: Infrastructure Agent
Task: Vercel Edge Setup - Step 3 (Create Blob Buckets)
Step: Creating third bucket "curious-kelly-visuals"
Issue: Bucket creation fails with "name already exists" error
Options: 
  A) Use different name (but doc says exact name)
  B) Delete existing bucket (risky, might be in use)
  C) Check if bucket exists and skip (deviation from plan)
Question: Which option should I take?
```

---

## 📈 SUCCESS METRICS

### Infrastructure Agent Success

**Vercel Edge Setup Complete When:**
- [ ] Edge Config created: `curious-kelly-lessons`
- [ ] Connection string saved
- [ ] 3 Blob buckets created (videos, audio, visuals)
- [ ] Environment variables set (EDGE_CONFIG, EDGE_CONFIG_SYNC_SECRET)
- [ ] Sync test successful (365 lessons synced)
- [ ] All steps verified and documented

### Overall Project Health

**Healthy When:**
- All agents follow their directives
- No scope creep or "while I'm here" changes
- All tests pass
- No production incidents
- Clear escalation when needed
- Boss approval for all production changes

---

## 🔄 DAILY OPERATIONS

### Morning Check (Boss Review)

**Every morning, Boss reviews:**

1. **What agents worked on yesterday?**
2. **What's blocked or needs escalation?**
3. **What's the priority for today?**
4. **Any production issues or incidents?**

### Agent Status Updates

**Every agent reports:**

- ✅ Completed: [What you finished]
- 🔄 In Progress: [What you're working on]
- ⏸️ Blocked: [What's stopping you]
- 📋 Next: [What's next on your list]

### End of Day

**Every agent must:**

- ✅ Document what changed
- ✅ Report completion status
- ✅ Flag any blockers
- ✅ Get Boss approval for tomorrow's work

---

## 📚 REFERENCE DOCUMENTS

### Must-Read (All Agents)
- `CLAUDE.md` - Operating rules and workflows
- `CLAUDE_OPERATIONAL_SPEC.md` - UI implementation rules
- `SETUP_VERCEL_EDGE_NOW.md` - Current priority task

### Domain-Specific
- **Infrastructure:** `docs/deployment/DEPLOYMENT_CHECKLIST.md`
- **Content:** `docs/kelly-video-system/PRODUCTION_WORKFLOW.md`
- **Frontend:** `docs/LESSON_PLAYER_ARCHITECTURE.md`
- **Backend:** `docs/backend/ARCHITECTURE.md`
- **Quality:** `docs/testing/` (if exists)

### Project Plans
- `docs/KELLY_PRODUCTION_PLAN_DEC17.md`
- `docs/operations/OVERNIGHT_VIDEO_PRODUCTION_PLAN.md`
- Various `*PLAN*.md` files in `docs/`

---

## ✅ QUICK REFERENCE CHECKLIST

**Before starting work:**
- [ ] Read the relevant plan document
- [ ] Verify you're in the correct domain
- [ ] Check for existing solutions
- [ ] Test locally first
- [ ] Get Boss approval if needed

**During work:**
- [ ] Test each step
- [ ] Document deviations
- [ ] Report blockers immediately
- [ ] Follow STOP and ask rule

**After work:**
- [ ] All tests pass
- [ ] Document changes
- [ ] Report completion
- [ ] Get Boss approval for deploy

---

## 🎯 CURRENT FOCUS

**Priority #1: Vercel Edge Setup**
- **Agent:** Infrastructure Agent
- **Status:** IN PROGRESS
- **Blockers:** None (yet)
- **Next Step:** Follow `SETUP_VERCEL_EDGE_NOW.md` Step 1

**All Other Agents:** Stand by. Do not start new work until Infrastructure Agent completes setup.

---

**Last Updated:** 2025-01-XX  
**Next Review:** After Infrastructure Agent completes Vercel Edge setup

---

*This document is the Boss. Follow it or escalate. No exceptions.*


