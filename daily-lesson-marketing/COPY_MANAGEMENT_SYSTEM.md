# Marketing Copy Management System
## The Daily Lesson by Curious Kelly

**Created:** November 17, 2025  
**Purpose:** AI-powered copy governance to maintain consistent, compelling marketing across all touchpoints

---

## 🎯 What This System Does

This system ensures all marketing copy (website, emails, ads) follows brand guidelines, pricing accuracy, and messaging consistency across English, Spanish, and Brazilian Portuguese.

**Key Benefits:**
- ✅ Prevents use of forbidden language (free, concierge, cohort, etc.)
- ✅ Ensures pricing is always accurate ($4.99/month, $49.99/year)
- ✅ Maintains consistent brand voice inspired by OpenEnglish.com
- ✅ AI-assisted copy generation and translation
- ✅ Automated validation before deployment

---

## 📁 System Components

### 1. **MARKETING_COPY_AGENT.md** (The Brain)
Complete brand guidelines, messaging framework, and copywriting rules.

**Contains:**
- Brand identity & positioning
- Target audiences (Adults, Children, Teachers)
- Pricing & offers
- Forbidden language list
- Approved messaging patterns
- FAQ templates
- Translation guidelines

### 2. **tools/copy-agent.js** (The Tool)
Node.js CLI that validates copy, generates new copy via Claude API, and translates.

**Commands:**
```bash
# Validate copy against guidelines
node tools/copy-agent.js validate

# Generate new copy section
node tools/copy-agent.js generate hero adults

# Translate copy to ES & PT
node tools/copy-agent.js translate copy.txt
```

### 3. **src/lib/i18n/en-us-UPDATED.ts** (Updated Copy)
New marketing copy following all guidelines. Removes:
- ❌ "Concierge" language
- ❌ "Cohort" language
- ❌ Group/school sales focus

Adds:
- ✅ Clear pricing: $4.99/month or $49.99/year
- ✅ 7 days to explore (no credit card)
- ✅ Christmas gifting angle
- ✅ "The Daily Lesson by Curious Kelly" branding
- ✅ 365 universal lessons for 2026

---

## 🚀 Quick Start

### Step 1: Review the Agent Guidelines
```bash
cd daily-lesson-marketing
cat MARKETING_COPY_AGENT.md
```

**Key sections to know:**
- **Forbidden Language:** What never to use
- **Pricing:** Always $4.99/month or $49.99/year
- **Target Audiences:** Adults, Children (via parents), Teachers
- **Brand Voice:** Warm, curious, simple (like OpenEnglish)

### Step 2: Validate Existing Copy
```bash
node tools/copy-agent.js validate src/lib/i18n/en-us.ts
```

**What it checks:**
- ✓ No forbidden words (free, concierge, cohort, etc.)
- ✓ Pricing mentioned correctly
- ✓ Trial period mentioned (7 days to explore)
- ✓ All three languages mentioned

### Step 3: Generate New Copy (Optional)
```bash
# Set your Claude API key
export ANTHROPIC_API_KEY=your-key-here

# Generate hero section for adults
node tools/copy-agent.js generate hero adults

# Generate FAQ section
node tools/copy-agent.js generate faq children
```

### Step 4: Replace Old Copy
```bash
# Backup current copy
cp src/lib/i18n/en-us.ts src/lib/i18n/en-us-OLD.ts

# Use updated copy
cp src/lib/i18n/en-us-UPDATED.ts src/lib/i18n/en-us.ts

# Update Spanish & Portuguese similarly
```

---

## 📝 Copy Guidelines Summary

### ✅ Always Include

**Pricing:**
- $4.99/month
- $49.99/year (saves $10)
- 7 days to explore, no credit card

**Features:**
- 8-minute daily lessons
- Age-adaptive (2-102 years)
- 3 languages: English, Spanish, Portuguese
- Up to 5 family profiles
- Privacy-first, ad-free (note: "ad-free" is acceptable—it describes absence of ads, not price)

**Call-to-Actions:**
- "Start Learning"
- "Begin Your Journey"
- "Take 7 Days to Explore"
- "Give 365 Days of Curiosity for 2026"

### ❌ Never Use

- **"Free"** → Use "yours," "included," "7 days to explore" (see `docs/brand/FORBIDDEN_WORD_FREE.md`)
- **"Free trial"** → Use "7 days to explore" or "your first week"
- **"Start free"** → Use "Start learning" or "Begin"
- "Concierge" → Use "support team" or nothing
- "Cohort" → Use "learners" or "subscribers"
- "Onboarding" → Use "getting started"
- "Founding members" → No exclusivity language
- "Pilot partners" → Implies unfinished
- "District-level" → No B2B school focus yet

### 🎨 Tone & Voice

**Think OpenEnglish.com:**
- Direct, not flowery
- Benefit-focused, not feature-focused
- "You" language, not "we"
- Active verbs: Start, Learn, Discover, Explore
- Short sentences. Easy to scan.

**Examples:**

❌ **Bad:** "Our concierge team will facilitate an onboarding experience tailored to your cohort's unique learning objectives."

✅ **Good:** "Start your first lesson in minutes. No setup required."

---

## 🌍 Translation Workflow

### English → Spanish → Portuguese (Simultaneous)

1. **Write English first** (en-us.ts)
2. **Validate** with copy-agent.js
3. **Translate** using AI or human translator
4. **Maintain consistency:**
   - Keep "Kelly" in English
   - Keep pricing in USD
   - Preserve emotional tone
   - Use warm, accessible language (tú/você)

### Using the Translation Tool

```bash
# Create a text file with English copy
echo "Start your 7-day free trial" > copy-to-translate.txt

# Translate to both languages
node tools/copy-agent.js translate copy-to-translate.txt
```

---

## 🎯 Usage by Team

### Marketing Team
**Before creating any ad, email, or landing page:**
1. Review MARKETING_COPY_AGENT.md guidelines
2. Draft copy following the framework
3. Run `node tools/copy-agent.js validate your-copy.txt`
4. Fix any errors/warnings
5. Get final human approval

### Developers
**When updating website copy:**
1. Edit src/lib/i18n/en-us.ts
2. Run validation: `node tools/copy-agent.js validate`
3. Update ES/PT translations
4. Test locally: `npm run dev`
5. Deploy only after validation passes

### Content Creators
**When writing new lesson descriptions:**
1. Follow brand voice from agent guidelines
2. Emphasize daily habit, curiosity, age-adaptive
3. Always mention 3 languages available
4. Keep it under 100 words

---

## 🛠️ Technical Setup

### Prerequisites
- Node.js 18+
- Anthropic Claude API key (for AI features)

### Installation
```bash
cd daily-lesson-marketing

# Install dependencies (if not already)
npm install

# Make copy-agent executable
chmod +x tools/copy-agent.js

# Set up API key (optional, for AI generation)
export ANTHROPIC_API_KEY=your-key-here
# Or add to .env file
```

### Running Validations in CI/CD

Add to your GitHub Actions or deployment pipeline:

```yaml
# .github/workflows/validate-copy.yml
name: Validate Marketing Copy
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: cd daily-lesson-marketing
      - run: node tools/copy-agent.js validate
```

---

## 📊 Success Metrics

### Copy Performance Tracking

**Conversion Goals:**
- Homepage → Trial signup: **5%**
- Trial → Paid subscriber: **40%**
- Gift purchases (December): **20% of revenue**

**A/B Testing:**
- Test headlines monthly
- Track which pain points resonate:
  - Time-constrained parents?
  - Curiosity-driven adults?
  - Resource-strapped teachers?

### Monthly Copy Review

**First week of each month:**
1. Review conversion data
2. Identify underperforming copy
3. Generate 2-3 headline variants with AI
4. A/B test for 2 weeks
5. Deploy winner

---

## 🎁 Christmas 2025 Campaign

### Timing
**Black Friday (Nov 29) → New Year's Day (Jan 1)**

### Messaging
**Hero:** "Give 365 days of curiosity for 2026"
**CTA:** "Buy gift subscription - $49.99"
**Benefits:**
- Email delivery for last-minute gifting
- Personalized message included
- Starts January 1, 2026
- Perfect for: parents, teachers, lifelong learners

### Copy Variants

**For Adults:**
"Give the gift that keeps them curious all year."

**For Parents:**
"Give your child a year of wonder. 365 lessons they'll actually look forward to."

**For Teachers:**
"Gift your teacher tribe a year of ready-to-use lessons."

---

## 🚨 Emergency Copy Updates

If pricing, features, or offers change urgently:

1. **Update MARKETING_COPY_AGENT.md first**
2. **Run AI generation** for affected sections
3. **Validate** all 3 languages
4. **Deploy simultaneously** (EN/ES/PT)
5. **Test checkout flow** end-to-end
6. **Notify** marketing team of changes

---

## 📚 Resources & References

### Inspiration Sites
- **OpenEnglish.com** - Hero structure, lead form placement, social proof
- **Duolingo.com** - Simple pricing, gamification mentions
- **Headspace.com** - Warm tone, daily habit messaging

### Brand Voice Examples

**We sound like:**
- A curious friend, not a salesperson
- A helpful teacher, not a lecturer
- A daily habit, not a course

**We don't sound like:**
- Academic journals
- Corporate software
- Luxury brands

---

## 🔄 Maintenance Schedule

### Weekly
- Monitor signup conversion rates
- Review support emails for copy confusion
- Update FAQ if new questions emerge

### Monthly
- A/B test hero headlines
- Refresh testimonials (as they come in)
- Validate all copy files

### Quarterly
- Update agent guidelines based on learnings
- Refresh social proof / trust badges
- Audit competitor messaging

### Annually
- Update year references (2026 → 2027)
- Refresh entire homepage design
- Major brand voice review

---

## 📞 Support

### For Copy Questions
- Review: MARKETING_COPY_AGENT.md
- Validate: `node tools/copy-agent.js validate`
- Generate: `node tools/copy-agent.js generate [section] [audience]`

### For Technical Issues
- Check Node.js version: `node --version` (need 18+)
- Verify API key: `echo $ANTHROPIC_API_KEY`
- Test validation: `node tools/copy-agent.js validate`

### For Strategy Questions
- Review: OpenEnglish.com for inspiration
- Consider: Target audience pain points
- Test: Always A/B test big changes

---

**Last Updated:** November 17, 2025  
**Maintained By:** Marketing Team + AI Copy Agent  
**Version:** 1.0

**Next Steps:**
1. Review updated copy (en-us-UPDATED.ts)
2. Replace old copy in production
3. Generate Spanish & Portuguese translations
4. Deploy to curiouskelly.com
5. Start A/B testing headlines

