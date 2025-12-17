# 🚨 LESSON SHARDS CORRUPTION REPORT

**Date:** December 6, 2025  
**Auditor:** Picky Nicky Shard Audit v1  
**Scope:** All 56,134 lesson_shards across 365 lessons

---

## Executive Summary

**CRITICAL FINDING:** 51% of all lessons have corrupted personalization shards. This affects 31,700 shards (56% of the database).

### Impact
- **User-Facing:** YES - `learn.html` actively uses shard content for personalization
- **Severity:** CRITICAL - Users see wrong lesson content based on age/tone preferences
- **Scale:** 186 of 365 lessons affected

---

## Corruption Statistics

| Metric | Value |
|--------|-------|
| **Total Lessons** | 365 |
| **Corrupted Lessons** | 186 (51%) |
| **Total Shards** | 56,134 |
| **Corrupted Shards** | 31,700 (56%) |

### Corruption Types

| Type | Lessons Affected | Description |
|------|------------------|-------------|
| **WIND** | 94 lessons | Shards contain wind turbine/renewable energy content |
| **LEAF** | 80 lessons | Shards contain photosynthesis/leaf content |
| **MICROSCOPE** | 11 lessons | Shards contain microscope/cell content |
| **PULLEY** | 1 lesson | Shards contain pulley/mechanical advantage content |

---

## How Shards Are Used

From `public/learn.html` line 1661-1662:

```javascript
if (personalization.currentShard?.script_content) {
  const shardContent = personalization.currentShard.script_content;
  // Uses shard content for personalized lesson delivery
}
```

**This means:**
- When a user selects an age range or tone preference
- `learn.html` fetches matching shards from `lesson_shards` table
- If the shard is corrupted, user sees WRONG CONTENT
- Example: Day 114 "Lifting Heavy Things Easily" shows microscope content

---

## Sample Corrupted Lessons (Top 20)

| Day | Topic | Total Shards | Corruption | Type |
|-----|-------|--------------|------------|------|
| 1 | Starting Fresh | 546 | 61 leaf + 6 wind | LEAF |
| 3 | Where Clouds Come From | 546 | 12 wind | WIND |
| 4 | How Light Travels | 456 | 21 wind | WIND |
| 19 | How Energy Changes Form | 456 | 28 wind | WIND |
| 71 | What's In the Air You Breathe | 156 | 11 microscope | MICROSCOPE |
| 114 | Lifting Heavy Things Easily | 156 | ALL microscope | MICROSCOPE |
| 209 | Why We Keep Doing Things | 156 | 38 wind | WIND |
| 272 | Things That Run Out | 156 | 1 leaf + 33 wind | MIXED |

---

## Root Cause Analysis

### Pattern Detected
The corruption follows the SAME pattern as `lesson_atoms`:
- Content from one lesson was systematically copied to another
- This suggests a **data generation/migration error**
- Likely occurred during initial shard population

### Affected Content Layers
1. ✅ **core_lessons** - FIXED (11 lessons, extended_explanation)
2. ⚠️ **lesson_atoms** - PARTIALLY FIXED (22 of 243 lessons)
3. 🚨 **lesson_shards** - NOT FIXED (186 of 365 lessons)
4. ✅ **lesson_age_hooks** - VERIFIED CLEAN

---

## Fix Strategy

### Option A: Delete All Corrupted Shards (FAST)
- **Time:** 5 minutes
- **Pros:** Immediate fix, forces fallback to atoms
- **Cons:** Loses personalization for 51% of lessons
- **Risk:** LOW - atoms will be used as fallback

### Option B: Regenerate All Corrupted Shards (SLOW)
- **Time:** 2-4 hours (31,700 shards × 12 archetypes × 6 age buckets)
- **Pros:** Restores full personalization
- **Cons:** Requires AI generation, costs money, takes time
- **Risk:** MEDIUM - generation could introduce new issues

### Option C: Hybrid Approach (RECOMMENDED)
1. **Immediate:** Delete all corrupted shards (5 min)
2. **Short-term:** Fix the 22 lessons we've already fixed in atoms (regenerate their shards)
3. **Long-term:** Systematically regenerate remaining shards lesson-by-lesson

---

## Recommended Action

**IMMEDIATE (Today):**
```sql
-- Delete all corrupted shards to stop serving wrong content
DELETE FROM lesson_shards
WHERE core_lesson_id IN (
  SELECT id FROM core_lessons WHERE day_number IN (
    -- List of 186 corrupted lesson day numbers
  )
);
```

**SHORT-TERM (This Week):**
- Regenerate shards for the 22 lessons we've already fixed
- Verify they work correctly in `learn.html`

**LONG-TERM (Next Month):**
- Systematically regenerate all 186 lessons' shards
- Build automated shard validation into CI/CD

---

## Testing Plan

1. **Delete corrupted shards**
2. **Test Day 1, 114, 245, 254, 274 in browser**
3. **Verify fallback to atoms works**
4. **Check that NO corrupted content appears**
5. **Regenerate shards for fixed lessons**
6. **Re-test with personalization enabled**

---

## Audit Trail

All findings recorded in `lesson_audits` table:
- `audit_type`: 'content_completeness'
- `field_name`: 'lesson_shards'
- `status`: 'fail'
- `audited_by`: 'shard_audit_v1'

---

**Next Steps:** Await user decision on fix strategy (A, B, or C).



