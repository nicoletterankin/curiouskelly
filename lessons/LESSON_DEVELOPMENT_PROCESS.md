# Lesson Development Process & Workflow

**Purpose:** Standardized process for creating comprehensive, research-backed lessons for Curious Kelly
**Created:** 2025-11-17
**Status:** Active standard for all lesson development
**First Implementation:** 3D Printing lesson (Day 321)

---

## Overview

This document defines the complete workflow for developing high-quality, age-appropriate, multilingual lessons that can answer ANY learner question about the topic through comprehensive background research and artifact creation.

**Core Philosophy:**
Each lesson should include a complete knowledge base that enables Kelly to answer detailed questions, provide examples, and engage learners at any depth appropriate to their age. We're not just creating a single lesson script - we're building a mini-encyclopedia for each topic.

---

## The Five Artifact System

Every lesson requires these five core artifacts:

1. **Brainstorming Document** - Initial planning and conceptual framework
2. **Knowledge Base** - Comprehensive research and authoritative reference
3. **Case Studies Library** - Real-world stories adapted for all age groups
4. **Interactive Elements Specification** - Visual, interactive, and hands-on components
5. **DNA File** - Complete lesson content with all age variants and multilingual support

---

## Step-by-Step Workflow

### Phase 1: Planning & Brainstorming

**Objective:** Establish the lesson's core concepts, age-appropriate approaches, and learning objectives.

**Deliverable:** `{topic}-lesson-brainstorm.md`

**Process:**
1. Identify the universal concept (what is fundamentally true about this topic?)
2. Define the core principle (what does this teach about democracy, justice, science, community?)
3. Map age-specific approaches for all 6 age groups:
   - Ages 2-5: Sensory, wonder-based, simple cause-effect
   - Ages 6-12: Exploration, hands-on, problem-solving
   - Ages 13-17: Critical thinking, social justice, technical depth
   - Ages 18-35: Systems thinking, economic analysis, implementation
   - Ages 36-60: Strategic deployment, community building, mentorship
   - Ages 61-102: Reflection, wisdom sharing, practical empowerment

4. Identify key tensions or questions to explore
5. Outline phase structure (Welcome → Questions → Wisdom)
6. Note multilingual considerations
7. Define Kelly's personality and voice for each age group
8. List open questions for further research

**Key Sections:**
- Universal concept & core principles
- Age-specific approaches (all 6 groups)
- Phase structure
- Multilingual considerations
- Interactive elements (initial ideas)
- Kelly's voice & personality
- Success metrics
- Open questions

**Example:** `lessons/3d-printing-lesson-brainstorm.md`

**Time Estimate:** 1-2 hours

---

### Phase 2: Comprehensive Background Research

**Objective:** Gather authoritative, current information to create a complete knowledge base that can answer any learner question.

**Deliverable:** `{topic}-knowledge-base.md`

**Process:**
1. **Web Research** (use WebSearch tool extensively):
   - How the technology/concept works (technical details)
   - Real-world applications and examples
   - Historical context and evolution
   - Current state (2024/2025 data)
   - Social, political, economic implications
   - Environmental impact (if relevant)
   - Key statistics and data points
   - Notable organizations, projects, or movements

2. **Document Everything:**
   - Core definitions
   - Technology overview (how it works)
   - Historical context
   - Real-world applications
   - Social & political dimensions
   - Environmental impact
   - Key statistics & data
   - Terminology & translations (EN/ES/FR)
   - **Sources & references** (critical - document all sources!)

3. **Synthesize Key Takeaways:**
   - What are the most important points?
   - What tensions or complexities exist?
   - What examples will resonate across age groups?

**Key Sections:**
- Table of Contents
- Core Definitions
- Technology/Concept Overview
- Historical Context
- Real-World Applications (detailed)
- Social & Political Dimensions
- Environmental Impact (if applicable)
- Key Statistics & Data
- Terminology & Translations (EN/ES/FR)
- Sources & References (all web searches, articles, data)
- Key Takeaways for Lesson Design

**Research Depth:**
- Minimum 6-8 web searches covering different aspects
- Gather specific numbers, dates, names, organizations
- Find 2-3 compelling real-world stories/case studies
- Document costs, scale, impact data
- Find both positive and challenging aspects (balanced view)

**Example:** `lessons/3d-printing-knowledge-base.md`

**Time Estimate:** 2-3 hours

---

### Phase 3: Case Studies & Storytelling Library

**Objective:** Compile detailed, emotionally resonant stories that can be adapted for all age groups.

**Deliverable:** `{topic}-case-studies.md`

**Process:**
1. **Select Primary Case Studies** (2-4 core stories):
   - Each should illustrate a key concept or tension
   - Should have human element (real people, real impact)
   - Should work globally (not overly US-centric)
   - Should allow for critical thinking at appropriate ages

2. **Develop Each Case Study:**
   - Full story with details (names, numbers, quotes if available)
   - Emotional arc (problem → solution → impact)
   - Age-specific adaptations for all 6 groups
   - Key learning from each story

3. **Create Quick Reference:**
   - Examples organized by theme
   - Age-appropriate selection guidelines
   - Storytelling guidelines by age group

**Key Sections:**
- Core Case Studies (2-4 detailed stories)
  - Full story (adaptable)
  - Emotional arc
  - Age-specific adaptations (all 6 groups)
  - Key insights
- Age-Specific Example Selection
- Quick Reference: Examples by Theme
- Storytelling Guidelines
- Emotional Arcs by Case Study
- Usage Notes for DNA Development

**Case Study Requirements:**
- Real people or organizations (not hypothetical)
- Specific data (costs, scale, impact)
- Emotional resonance (care about the people involved)
- Demonstrates the core principle/tension
- Works across cultures and languages

**Example:** `lessons/3d-printing-case-studies.md`

**Time Estimate:** 2-3 hours

---

### Phase 4: Interactive Elements & Engagement Specification

**Objective:** Define all visual assets, interactive components, and hands-on activities to maximize engagement.

**Deliverable:** `{topic}-interactive-elements.md`

**Process:**
1. **Visual Assets:**
   - Essential animations (how it works)
   - Photo galleries (real examples)
   - Infographics (data visualization)
   - Before/after comparisons
   - Process documentation

2. **Interactive Components:**
   - Questions and polls
   - Design challenges
   - Decision games
   - Exploration tools (maps, virtual tours)

3. **Hands-On Activities:**
   - For those without specialized equipment
   - For those with access to tools
   - Age-appropriate variations

4. **Engagement Prompts:**
   - Specific questions for each age group
   - Critical thinking prompts
   - Creative prompts
   - Action prompts

5. **Kelly's Direction:**
   - Avatar behavior and expressions
   - Voice characteristics by age
   - Signature moments and phrases

**Key Sections:**
- Visual Assets (Priority 1, 2, 3)
  - Specifications for each asset
  - Age adaptations
  - Kelly's narration notes
- Interactive Components
  - Full specifications for each activity
  - User flows
  - Age appropriateness
- Hands-On Activities
  - With and without equipment access
  - Materials needed
  - Learning goals
- Engagement Prompts by Age
- Kelly's Avatar & Voice Direction
- Implementation Notes
  - Priority for development
  - Accessibility considerations
  - Multilingual assets
  - Technical specifications

**Example:** `lessons/3d-printing-interactive-elements.md`

**Time Estimate:** 2-3 hours

---

### Phase 5: DNA File Development

**Objective:** Create the complete, multilingual lesson DNA file with all age variants and phase content.

**Deliverable:** `{topic}-dna.json`

**Process:**
1. **Top-Level Structure:**
   - ID, title, version, dates, author
   - Calendar placement
   - Universal concept (EN/ES/FR)
   - Core principle (EN/ES/FR)
   - Learning essence (EN/ES/FR)
   - Metadata

2. **Age Variants** (all 6):
   For each age group (2-5, 6-12, 13-17, 18-35, 36-60, 61-102):

   a. **Variant Metadata:**
   - Title, description, video reference
   - Kelly age and persona
   - Voice profile
   - Core metaphor (EN/ES/FR)
   - Attention span, cognitive focus
   - Examples list

   b. **Language Content (EN/ES/FR):**
   - Title
   - Welcome message
   - Main content (full lesson text)
   - Key points
   - Interaction prompts
   - Wisdom moment
   - Core metaphor
   - Abstract concepts
   - Call to action
   - Summary

   c. **Supplementary:**
   - Objectives
   - Vocabulary
   - Abstract concepts & translations
   - Pacing
   - Teaching moments
   - Expression cues
   - Tone & voice patterns

3. **Quality Checks:**
   - All 6 age variants complete
   - All 3 languages (EN/ES/FR) for each variant
   - Consistent universal concepts across variants
   - Age-appropriate complexity and examples
   - Tone matches Kelly persona for each age

**DNA File Structure:**
```json
{
  "id": "topic-slug",
  "title": "Lesson Title",
  "version": "1.0.0",
  "createdAt": "timestamp",
  "updatedAt": "timestamp",
  "author": "Curious Kelly Lesson Development Team",
  "description": "Brief description",
  "calendar": {"day": X, "date": "Month Day"},
  "universal_concept": "key",
  "universal_concept_translations": {"en": "", "es": "", "fr": ""},
  "core_principle": "key",
  "core_principle_translations": {"en": "", "es": "", "fr": ""},
  "learning_essence": "description",
  "learning_essence_translations": {"en": "", "es": "", "fr": ""},
  "metadata": {
    "category": "category",
    "difficulty": "beginner|intermediate|advanced",
    "duration": {"min": X, "max": Y},
    "tags": [],
    "prerequisites": [],
    "learningOutcomes": []
  },
  "ageVariants": {
    "2-5": { ... },
    "6-12": { ... },
    "13-17": { ... },
    "18-35": { ... },
    "36-60": { ... },
    "61-102": { ... }
  }
}
```

**Example:** `curious-kellly/backend/config/lessons/3d-printing-manufacturing-layer-by-layer-dna.json`

**Time Estimate:** 4-6 hours (all 6 age variants with 3 languages each)

---

### Phase 6: Validation & Quality Assurance

**Objective:** Ensure all artifacts meet quality standards and requirements.

**Checklist:**

**Content Completeness:**
- [ ] All 5 artifacts created
- [ ] All 6 age variants in DNA file
- [ ] All 3 languages (EN/ES/FR) for all content
- [ ] Sources documented in knowledge base
- [ ] Case studies have specific data and real names
- [ ] Interactive elements fully specified

**Quality Standards:**
- [ ] Universal concept consistent across all variants
- [ ] Age-appropriate complexity and language
- [ ] Balances technical accuracy with accessibility
- [ ] Includes both positive aspects and tensions/challenges
- [ ] Empowers learner agency (even without access to tools/tech)
- [ ] Culturally sensitive, globally relevant examples

**CLAUDE.md Compliance:**
- [ ] Precomputed multilingual content (EN/ES/FR)
- [ ] No runtime language generation
- [ ] Follows approved plan references
- [ ] No interest-driven selection
- [ ] Maintains daily habit reinforcement approach
- [ ] Sub-second UI paths (optimized assets)

**Technical Validation:**
- [ ] JSON syntax valid
- [ ] Matches DNA schema
- [ ] File paths correct
- [ ] All referenced assets specified
- [ ] Accessibility notes included

**Time Estimate:** 1 hour

---

### Phase 7: Documentation & Knowledge Transfer

**Objective:** Ensure the lesson and process are fully documented for future reference and iteration.

**Process:**
1. Update this process document if improvements identified
2. Create summary notes on what worked well and what to improve
3. Update calendar/index with new lesson
4. Document any new patterns or insights for future lessons

**Time Estimate:** 30 minutes

---

## Total Time Investment Per Lesson

**Estimated Time:**
- Phase 1 (Brainstorming): 1-2 hours
- Phase 2 (Research): 2-3 hours
- Phase 3 (Case Studies): 2-3 hours
- Phase 4 (Interactive Elements): 2-3 hours
- Phase 5 (DNA File): 4-6 hours
- Phase 6 (Validation): 1 hour
- Phase 7 (Documentation): 30 minutes

**Total: 13-18 hours per complete lesson**

This investment creates a lesson that can:
- Answer ANY learner question about the topic
- Serve learners across 6 age groups
- Work in 3 languages
- Provide multiple entry points (visual, interactive, story-based)
- Stand the test of time with research-backed content

---

## File Naming Conventions

**All artifacts stored in:** `/lessons/`

**Naming Pattern:**
- Brainstorm: `{topic}-lesson-brainstorm.md`
- Knowledge Base: `{topic}-knowledge-base.md`
- Case Studies: `{topic}-case-studies.md`
- Interactive Elements: `{topic}-interactive-elements.md`
- DNA File: `curious-kellly/backend/config/lessons/{topic}-dna.json`

**Topic Slug Format:**
- Lowercase
- Hyphens for spaces
- Descriptive but concise
- Example: `3d-printing-manufacturing-layer-by-layer`

---

## Git Workflow

**Branch Naming:**
- Use claude/{session-id} format as required
- Branch created automatically by system

**Commit Strategy:**
1. **First Commit:** Brainstorm artifact
2. **Second Commit:** All research artifacts (knowledge base, case studies, interactive elements, partial/complete DNA)
3. **Third Commit (if needed):** DNA file completion and validation
4. **Final Commit:** Any fixes, documentation updates

**Commit Message Format:**
```
Add [comprehensive|complete] {topic} lesson [development artifacts|completion]

[Detailed bullet points describing:]
- Knowledge base scope and sources
- Case studies included
- Interactive elements defined
- DNA file status
- Any notable insights or innovations

Ready for: [next steps]
```

---

## Quality Metrics

**A High-Quality Lesson Includes:**

1. **Depth:** Can answer questions at multiple levels of complexity
2. **Breadth:** Covers technical, social, environmental, historical aspects
3. **Humanity:** Centers real people and their stories
4. **Criticality:** Explores tensions and complexities, not just benefits
5. **Empowerment:** Gives learners agency to engage, even without resources
6. **Accessibility:** Works for all ages and languages
7. **Currency:** Uses recent data (2024/2025) and current examples
8. **Authenticity:** Real names, real numbers, real organizations
9. **Global Relevance:** Works across cultures, not US-centric
10. **Research Integrity:** All claims sourced and documented

---

## Common Pitfalls to Avoid

**❌ Don't:**
- Skip comprehensive research (knowledge base must be thorough)
- Use hypothetical examples when real ones exist
- Oversimplify complex tensions (acknowledge nuance)
- Forget to document sources
- Create content for only some age groups
- Make up statistics or examples
- Use only US-centric case studies
- Ignore environmental or social implications
- Write patronizing content for any age group
- Create DNA content without multilingual versions

**✅ Do:**
- Invest time in research upfront
- Find and tell real stories with real names
- Acknowledge both benefits and challenges
- Cite sources for all data and claims
- Develop all 6 age variants completely
- Use recent, verifiable statistics
- Include global examples and perspectives
- Address social justice and equity dimensions
- Write respectfully for all ages
- Complete all 3 language translations

---

## When to Use This Process

**Use this full process for:**
- New daily lessons (365-day curriculum)
- Major topic overhauls or updates
- Lessons requiring deep technical accuracy
- Topics with significant social/political dimensions
- Lessons where comprehensive Q&A is expected

**Can abbreviate for:**
- Simple review lessons
- Iterative improvements to existing lessons (update one artifact at a time)
- Time-sensitive emergency content

---

## Future Enhancements

**Potential Improvements to This Process:**
- Automated validation scripts for DNA schema compliance
- Template generators for each artifact type
- Shared glossary/translation database for common terms
- Peer review workflow
- Learner feedback integration loop
- Analytics on which interactive elements perform best
- A/B testing framework for different approaches

---

## Example: 3D Printing Lesson

**Implementation:** Day 321 (November 17, 2025)
**Topic:** 3D Printing - Manufacturing Layer by Layer
**Files Created:**
- `/lessons/3d-printing-lesson-brainstorm.md` ✅
- `/lessons/3d-printing-knowledge-base.md` ✅
- `/lessons/3d-printing-case-studies.md` ✅
- `/lessons/3d-printing-interactive-elements.md` ✅
- `/curious-kellly/backend/config/lessons/3d-printing-manufacturing-layer-by-layer-dna.json` ⏳ (Ages 2-5, 6-12 complete; 13-17, 18-35, 36-60, 61-102 pending)

**Time Investment:** ~12 hours (so far)

**Outcomes:**
- Knowledge base can answer questions about FDM/SLA/SLS, e-NABLE, ICON homes, right-to-repair, environmental impact, makerspaces, history, bioprinting, and more
- Three detailed case studies with age adaptations
- Complete interactive element specifications
- Partial DNA file with multilingual support

**Next Steps:** Complete remaining DNA age variants, validate, integrate with lesson player

---

## Conclusion

This process transforms lesson development from "writing a script" to "building a comprehensive knowledge system." The upfront investment pays dividends in:

1. **Quality:** Lessons are research-backed and authoritative
2. **Flexibility:** Can answer unexpected questions
3. **Scalability:** Works across ages and languages
4. **Longevity:** Content stays relevant with proper sourcing
5. **Engagement:** Multiple entry points for different learning styles

Every lesson following this process becomes a mini-encyclopedia, empowering Kelly to be a true guide and educator across the full spectrum of learner needs.

---

**Document Status:** Active standard
**Next Review:** After 5 lessons completed using this process
**Maintained By:** Curious Kelly Lesson Development Team
**Last Updated:** 2025-11-17
