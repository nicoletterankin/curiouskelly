# Curious Kelly Evaluation Suite

Quality gates for Kelly's voice and the lifetime experience system.

## Quick Start

```bash
# Run all evals
npm run eval

# Or run individually
npx ts-node evals/kelly-voice-eval.ts
npx ts-node evals/lifetime-experience-eval.ts
```

## Eval Suites

### 1. Kelly Voice Eval (`kelly-voice-eval.ts`)

Tests all Kelly communications against the sacred voice guide.

**Scores (1-5 each, 30 total):**
- Humility - Is Kelly WITH the learner, not above?
- Warmth - Does it feel like a friend?
- Simplicity - Clear, not clever?
- Invitation - Welcoming, not demanding?
- Richness - Understated, not cheap?
- Collaboration - About learning TOGETHER?

**Pass Criteria:**
- All individual scores ≥ 4
- Total score ≥ 25/30

**Automatic Fail Triggers:**
- Uses "user" instead of "learner"
- Contains "don't miss", "act now", etc.
- More than 2 emojis
- Multiple exclamation marks
- All-caps shouting

### 2. Lifetime Experience Eval (`lifetime-experience-eval.ts`)

Tests the spiral learning system.

**Database Tests:**
- `users` table has lifetime fields
- `lesson_history` table exists
- `milestones` table exists
- `commons_answers` table exists
- `increment_commons_answer()` function works

**API Tests:**
- `GET /api/lesson-history?day=X` - Accessible
- `POST /api/lesson-complete` - Accessible
- `GET /api/reflection?day=X` - Accessible
- `GET /api/commons?day=X` - Returns correct structure

**Logic Tests:**
- Layer recommendation (foundation → teaching)
- Milestone definitions (streaks, years, lessons)
- Birthday detection
- Day of year calculation

## Running in CI

Add to your CI pipeline:

```yaml
- name: Run Evals
  run: npm run eval
  env:
    PUBLIC_SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
    SUPABASE_SERVICE_ROLE_KEY: ${{ secrets.SUPABASE_SERVICE_KEY }}
```

## Adding New Test Cases

### Kelly Voice

Add to `TEST_CASES` array in `kelly-voice-eval.ts`:

```typescript
{
  name: 'My New Test',
  text: `The text to evaluate...`,
  expectedPass: true, // or false
}
```

### Lifetime Experience

Add test functions following the pattern:

```typescript
async function testMyFeature() {
  log('\n🔍 MY FEATURE TESTS');
  log('─'.repeat(50));
  
  // Test logic here
  if (condition) {
    pass('Test name', 'Success message');
  } else {
    fail('Test name', 'Failure reason');
  }
}
```

Then call it in `runEvals()`.

## Philosophy

> "Every word Kelly speaks must earn its place."

These evals exist to protect Kelly's voice and ensure the lifetime experience works correctly. They should run:

1. **Before every deploy** - CI gate
2. **Before merging PRs** - Quality check
3. **During development** - Rapid feedback

If evals fail, **fix the issue before shipping**.

---

*"I don't have all the answers. But I love finding them."*

