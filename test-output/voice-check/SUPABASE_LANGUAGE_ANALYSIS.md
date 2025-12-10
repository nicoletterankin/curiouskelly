# 🗄️ Supabase Database Language Analysis

**Analysis Date:** December 9, 2025  
**Focus:** How the database refers to people using the system

---

## Summary: ⚠️ Mixed Usage - "Learner" in System Context

The Supabase database uses **"learner"** terminology in **system/internal contexts** (tier names, table names) but **NOT in user-facing content** (lesson scripts).

**Key Finding:** The word "learner" appears in:
- ✅ **System tier names** (internal classification)
- ✅ **Table names** (developer context)
- ❌ **NOT in lesson content** (user-facing)

---

## Database Tables & Columns

### Core Tables (Content)

#### ✅ `core_lessons` - Neutral, topic-focused
```sql
- id, day_number, topic, universal_truth
- marketing_headline, marketing_tagline
```
**No "student/user/learner" terminology**

#### ✅ `lesson_atoms` - Content pieces
```sql
- id, core_lesson_id, archetype, phase, content
```
**No "student/user/learner" terminology**

#### ✅ `lesson_shards` - Demographic variants
```sql
- id, core_lesson_id, age, region, tone, birth_year
```
**No "student/user/learner" terminology**

### User Management Tables

#### ⚠️ `users` - Standard naming convention
```sql
CREATE TABLE users (
  id UUID,
  email TEXT,
  subscription_tier TEXT,
  current_day INTEGER,
  streak_days INTEGER,
  ...
)
```
**Uses "users" (standard database convention)**
- This is **acceptable** - "users" is the universal database term
- Alternative would be awkward ("people" table?)
- Context: Technical/system level, not user-facing

#### ✅ `user_progress` - Neutral
```sql
- user_id, lesson_id, completed, progress_percent
```
**Standard technical naming**

---

## Commission/Earnings System

### ⚠️ Commission Tiers - Uses "Learner" in Display Names

From `docs/backend/migrations/20251207_earn_to_learn.sql`:

```sql
INSERT INTO commission_tiers (tier_name, display_name, ...)
VALUES 
  ('new_learner', 'New Learner', 0, 0.10, ...),
  ('active_learner', 'Active Learner', 7, 0.15, ...),
  ('committed_learner', 'Committed Learner', 30, 0.20, ...),
  ('dedicated_learner', 'Dedicated Learner', 100, 0.25, ...),
  ('complete_learner', 'Complete Learner', 365, 0.30, ...),
  ('legendary_learner', 'Legendary Learner', 1000, 0.35, ...)
```

**Context:** These are **user-facing tier names** shown in the earnings dashboard.

**Issue:** "Learner" appears in UI context here!

---

## Learner Commons Tables

From `docs/backend/migrations/001_learner_commons.sql`:

### ⚠️ Table Names with "Learner"
```sql
- commons_proposals
- commons_votes  
- commons_discussions
- commons_lesson_notes
- commons_contributor_stats
```

**File name:** `001_learner_commons.sql`

**Table comments:**
```sql
-- LEARNER COMMONS TABLES
-- What learners often misunderstand
```

**Context:** These are **developer comments and file names**, not user-facing.

---

## Kids Compliance System

From `docs/backend/migrations/20251207_kids_compliance.sql`:

### ✅ No "Student/Learner" Language
```sql
- earnings_compliance_log
- minor_consent_records
- users.parent_account_id
- users.is_family_admin
```

**Uses neutral terms:** "minor," "parent," "family"

---

## Analysis by Context

### 1. User-Facing Content (Lesson Scripts)
**Status:** ✅ **EXCELLENT**
- No "students" (0 occurrences)
- No "users" (0 occurrences)
- No "learners" (0 occurrences)
- Uses "you," "we," "friend"

### 2. User-Facing UI (Commission Tiers)
**Status:** ⚠️ **NEEDS REVIEW**
- "New Learner" tier
- "Active Learner" tier
- "Committed Learner" tier
- etc.

**Question:** Should these be renamed?

### 3. System/Technical (Table Names, Columns)
**Status:** ✅ **ACCEPTABLE**
- `users` table (standard convention)
- `user_progress` (technical context)
- `learner_commons` (file/table name)

**Rationale:** Technical naming conventions are fine for backend.

### 4. Developer Comments/Documentation
**Status:** ✅ **ACCEPTABLE**
- "What learners often misunderstand" (comment)
- "LEARNER COMMONS TABLES" (header)

**Rationale:** Developer documentation can use technical terms.

---

## Recommendations

### 🔴 HIGH PRIORITY: Commission Tier Display Names

**Current:**
- "New Learner"
- "Active Learner"
- "Committed Learner"
- "Dedicated Learner"
- "Complete Learner"
- "Legendary Learner"

**Recommended Alternatives:**

#### Option 1: Friend-Based (Warmest)
- "New Friend" → Too casual?
- "Active Friend" → Weird
- ❌ Not scalable

#### Option 2: Journey-Based (Natural)
- "New Explorer" ✅
- "Active Explorer" ✅
- "Committed Explorer" ✅
- "Dedicated Explorer" ✅
- "Complete Explorer" ✅
- "Legendary Explorer" ✅

#### Option 3: Achievement-Based (Gamified)
- "Curious Starter" ✅
- "Daily Discoverer" ✅
- "Committed Companion" ✅
- "Dedicated Adventurer" ✅
- "Complete Champion" ✅
- "Legendary Leader" ✅

#### Option 4: Role-Based (Empowering)
- "New Member" ✅
- "Active Member" ✅
- "Committed Member" ✅
- "Dedicated Member" ✅
- "Complete Member" ✅
- "Legendary Member" ✅

#### Option 5: Kelly-Centric (Brand-Aligned)
- "Kelly's Friend" ✅
- "Kelly's Companion" ✅
- "Kelly's Partner" ✅
- "Kelly's Champion" ✅
- "Kelly's Hero" ✅
- "Kelly's Legend" ✅

**Recommendation:** Use **Option 2 (Explorer)** or **Option 5 (Kelly-Centric)**

### 🟡 MEDIUM PRIORITY: Bonus Program Names

From the same migration:

```sql
('community_builder', 'Community Builder', 
 'Bonus for referring 10+ learners', ...)
```

**Issue:** Description says "referring 10+ learners"

**Recommended:**
- "Bonus for referring 10+ friends" ✅
- "Bonus for sharing with 10+ people" ✅
- "Bonus for building community" ✅

### 🟢 LOW PRIORITY: Keep As-Is

#### Table Names
- `learner_commons` → Keep (technical)
- `users` → Keep (standard)
- `user_progress` → Keep (standard)

#### Developer Comments
- "What learners often misunderstand" → Keep (internal docs)
- "LEARNER COMMONS TABLES" → Keep (developer context)

---

## SQL Migration Needed?

### Yes - Update Commission Tier Display Names

```sql
-- Update commission tier display names
UPDATE commission_tiers SET display_name = 'New Explorer' WHERE tier_name = 'new_learner';
UPDATE commission_tiers SET display_name = 'Active Explorer' WHERE tier_name = 'active_learner';
UPDATE commission_tiers SET display_name = 'Committed Explorer' WHERE tier_name = 'committed_learner';
UPDATE commission_tiers SET display_name = 'Dedicated Explorer' WHERE tier_name = 'dedicated_learner';
UPDATE commission_tiers SET display_name = 'Complete Explorer' WHERE tier_name = 'complete_learner';
UPDATE commission_tiers SET display_name = 'Legendary Explorer' WHERE tier_name = 'legendary_learner';

-- Update bonus program descriptions
UPDATE bonus_programs 
SET description = REPLACE(description, 'learners', 'friends')
WHERE description LIKE '%learners%';
```

**OR** (Kelly-Centric Option):

```sql
UPDATE commission_tiers SET display_name = 'Kelly''s Friend' WHERE tier_name = 'new_learner';
UPDATE commission_tiers SET display_name = 'Kelly''s Companion' WHERE tier_name = 'active_learner';
UPDATE commission_tiers SET display_name = 'Kelly''s Partner' WHERE tier_name = 'committed_learner';
UPDATE commission_tiers SET display_name = 'Kelly''s Champion' WHERE tier_name = 'dedicated_learner';
UPDATE commission_tiers SET display_name = 'Kelly''s Hero' WHERE tier_name = 'complete_learner';
UPDATE commission_tiers SET display_name = 'Kelly''s Legend' WHERE tier_name = 'legendary_learner';
```

---

## Comparison: Database vs. Content

| Context | Current Usage | Status |
|---------|---------------|--------|
| **Lesson scripts** | "you," "we," "friend" | ✅ Perfect |
| **Commission tiers** | "New Learner," etc. | ⚠️ Needs update |
| **Bonus descriptions** | "referring learners" | ⚠️ Needs update |
| **Table names** | `users`, `learner_commons` | ✅ Acceptable (technical) |
| **Column names** | `user_id`, `referred_user_id` | ✅ Standard (technical) |
| **Developer comments** | "learners often misunderstand" | ✅ Acceptable (internal) |

---

## Trust & Safety Alignment

### Current Issues:
1. **Commission tier names** use "learner" (clinical, educational jargon)
2. **Bonus descriptions** refer to "learners" (not warm/personal)

### Alignment with Principles:
- ✅ **Radical Transparency** - System is honest about tiers
- ⚠️ **User Control** - Language could be more empowering
- ✅ **No Manipulation** - Tiers are merit-based
- ⚠️ **Authentic AI** - "Learner" feels institutional, not Kelly-like

---

## Final Recommendation

### Immediate Action:
1. ✅ **Update commission tier display names** to "Explorer" or "Kelly's [Role]"
2. ✅ **Update bonus program descriptions** to use "friends" or "people"
3. ✅ **Keep technical names** (`users` table, `learner_commons`, etc.)

### Migration Script Priority:
**HIGH** - These are user-facing in the earnings dashboard

### Voice Check Impact:
**NONE** - Lesson content is already perfect. This is a database UI issue only.

---

## Conclusion

**Database Status:** ⚠️ **MOSTLY GOOD, ONE USER-FACING ISSUE**

- ✅ **Lesson content:** Perfect (no "learner" language)
- ⚠️ **Commission tiers:** Uses "learner" (user-facing, should update)
- ✅ **Technical names:** Standard conventions (acceptable)
- ✅ **Developer docs:** Internal context (acceptable)

**Action Required:** Update commission tier display names before launch.

**Voice Check Status:** Still ✅ **READY FOR BULK GENERATION** (lesson content is perfect)

---

**Files Analyzed:**
- `docs/backend/SUPABASE_SCHEMA.md`
- `docs/backend/migrations/20251207_earn_to_learn.sql`
- `docs/backend/migrations/20251207_kids_compliance.sql`
- `docs/backend/migrations/001_learner_commons.sql`







