# Items Requiring Human Review

**Last Updated:** 2025-12-07T23:45:00Z  
**Reviewer:** Nicolette  
**CAO Agent:** Chief Academic Officer

---

## Critical Issues (Block Launch)

<!-- Items that MUST be resolved before Dec 17 -->

### 🚨 ARCHETYPE COUNT DISCREPANCY

**Discovery:** 2025-12-07T23:50:00Z

**Issue:** The database contains **12 archetypes** but documentation (GOLDEN_THREE_ARCHETYPES.md, CHIEF_ACADEMIC_OFFICER_AGENT_PROMPT.md) specifies only **3 archetypes** (Explorer, Scientist, Rebel).

**Database Counts:**
| Archetype | Count |
|-----------|-------|
| The Explorer | 1,825 |
| The Rebel | 1,825 |
| The Scientist | 1,825 |
| The Architect | 1,820 |
| The Diplomat | 1,820 |
| The Empath | 1,820 |
| The MacGyver | 1,820 |
| The Mystic | 1,820 |
| The Provider | 1,820 |
| The Storyteller | 1,820 |
| The Strategist | 1,820 |
| The Survivor | 1,820 |

**Per-Lesson Pattern:**
- Day 1: 15 atoms, 3 archetypes ✅ (matches Golden Three spec)
- Days 2-30: 60 atoms, 12 archetypes ❌ (inconsistent with spec)

**Impact:** 
- Frontend player must handle 12 archetypes, not 3
- Content volume is 4x what spec requires
- User archetype selection UX needs clarification

**CAO Recommendation:**
Given launch is Dec 17 (10 days away), I recommend:
1. **KEEP all 12 archetypes** - content is already created, deleting = lost work
2. **Update documentation** to reflect 12 archetypes
3. **Ensure frontend** can handle all 12 (or fallback to Golden Three)
4. **Post-launch:** Analyze which archetypes drive engagement, consider consolidation

**Awaiting:** Nicolette's decision on archetype strategy before proceeding

---

## High Priority (Should Fix)

<!-- Items that significantly impact quality -->

### 🔧 SLOP CONTAMINATION IN DAYS 1-30

**Discovery:** 2025-12-07T23:55:00Z

**Issue:** ~250 lesson atoms in Days 1-30 contain banned slop phrases that contradict Kelly's authentic voice.

**Slop Pattern Distribution (Days 1-30):**
| Pattern | Count | Severity |
|---------|-------|----------|
| "absolutely" | ~150 | HIGH - pervasive |
| "incredible" | ~35 | HIGH - empty enthusiasm |
| "certainly" | ~20 | MEDIUM - assistant-speak |
| "amazing" | ~15 | HIGH - empty enthusiasm |
| "delve" | ~10 | MEDIUM - AI-speak |
| "let's dive" | 2 | MEDIUM - filler transition |
| "great question" | 2 | HIGH - assistant-speak |

**Impact:**
- Kelly sounds like a generic AI assistant instead of a warm friend
- Violates the "No Slop" quality standard in CAO prompt
- Will feel inauthentic to learners

**CAO Recommendation:**
1. **Create slop_fixes.sql** migration script with all rewrites
2. **Prioritize Day 1** - it launches first (only 15 atoms, 1 slop instance)
3. **Batch process by day** - fix one day at a time
4. **Run anti-slop check** after each fix batch

**Rewrite Guidelines (per CHIEF_ACADEMIC_OFFICER_AGENT_PROMPT.md):**
| Slop | Kelly Voice Alternative |
|------|------------------------|
| "Absolutely!" | "Yes!" / "Indeed!" / Just the answer |
| "Certainly!" | Remove entirely |
| "Incredible!" | "Fascinating" / "Curious" / Show don't tell |
| "Amazing!" | Cut it, or describe what makes it interesting |
| "Let's dive in" | Just start the content |
| "Delve" | "Explore" / "Look at" / "Consider" |

**Status:** Awaiting decision on bulk update approach

---

## Medium Priority (Nice to Have)

<!-- Quality improvements, not blockers -->

### ℹ️ "251 Mismatch" Claim Investigation

**Discovery:** 2025-12-08T00:00:00Z

**Issue:** The CAO prompt stated "251 of 365 lessons have topic/headline mismatches" but audit found **no mismatches** in Days 1-30.

**Investigation Results:**
- ✅ Days 1-30 topics align with headlines and universal truths
- ✅ All 365 lessons have content atoms
- ✅ No missing lessons detected

**Possible Explanations:**
1. The 251 number may refer to a previous database state that has since been fixed
2. It may refer to a different definition of "mismatch" (e.g., marketing vs lesson content)
3. It may have been an estimate that proved incorrect

**CAO Recommendation:**
- Mark this as RESOLVED unless Nicolette has specific examples to investigate
- Focus resources on slop cleanup rather than mismatch hunting

---

## Notes for Nicolette

<!-- Context, recommendations, decisions I would make if I had authority -->

### Executive Summary (CAO Session 2025-12-07)

**Good News First:**
- All 365 lessons have content ✅
- Topic/headline alignment is solid ✅  
- Database structure is functional ✅
- Launch-blocking content gaps: NONE

**Decisions Needed:**

1. **Archetype Strategy** (CRITICAL)
   - Database has 12 archetypes but docs specify Golden Three
   - Options:
     - A) Keep all 12 (preserves work, richer content)
     - B) Consolidate to 3 (matches docs, simpler UX)
   - **My recommendation: Option A** - The content exists, don't delete it. Update docs instead.

2. **Slop Cleanup Approach** (HIGH)
   - ~250 atoms in Days 1-30 have banned phrases
   - Options:
     - A) Batch SQL update (fast but risky)
     - B) Individual rewrites via Supabase admin (slow but safe)
     - C) Create slop_fixes.sql migration (trackable, reviewable)
   - **My recommendation: Option C** - Create migration script, review before applying

3. **Priority Order** (MEDIUM)
   - Day 1 is cleanest (only 1 slop instance in 15 atoms)
   - Days 2-10 should be next (most visible to early users)
   - Days 11-30 can follow
   - Days 31-365 can be ongoing post-launch cleanup

**If I Had Full Authority, I Would:**
1. Fix Day 1's single "Absolutely!" instance right now
2. Generate slop_fixes.sql for Days 1-30
3. Review and apply in batches
4. Update GOLDEN_THREE_ARCHETYPES.md to document actual 12 archetypes
5. Add archetype descriptions for the 9 non-Golden archetypes

---

## Resolution Log

| Date | Issue | Resolution | Resolved By |
|------|-------|------------|-------------|
| - | - | - | - |


