# Curriculum Data Quality Audit Report

**Generated:** December 16, 2025
**Source:** lessons/365_day_calendar.json (Supabase sync)

---

## Summary

| Metric | Count |
|--------|-------|
| Total lessons | 365 |
| Potential mismatches | 57 |
| Duplicate titles | 3 |
| Missing fields | 0 |

---

## Potential Title/Objective Mismatches

These days have titles that don't obviously match their learning objectives:

| Day | Title | Learning Objective |
|-----|-------|-------------------|
| 1 | Starting Fresh | New beginnings offer opportunities for growth and change. |
| 6 | What's Inside a Seed | Seeds hold the potential for entire forests and gardens. |
| 8 | What Makes a Real Friend | Friendship connects people across differences and distances. |
| 14 | Why Curious People Learn More | Curiosity drives deeper learning and understanding. |
| 21 | What Makes Things Grow | Growth requires change, challenge, and time. |
| 52 | What's Inside a Volcano | Volcanoes release molten rock from deep within Earth. |
| 55 | The Deep Ocean Mystery | Oceans cover over 70% of Earth's surface and teem with life. |
| 58 | Life in the Desert | Deserts receive very little rainfall but host adapted life f... |
| 61 | The Power of Grass | Grasses cover vast areas and support many ecosystems. |
| 74 | The Gas You Don't Notice | Nitrogen makes up most of the air we breathe. |
| 91 | What Makes a Family | Families come in many forms and provide support and belongin... |
| 92 | How We Understand Each Other | Understanding how we understand each other helps us make sen... |
| 100 | Splitting Things Fairly | Division distributes quantities equally among groups. |
| 104 | How We Measure Things | Measurement uses standard units to quantify the world. |
| 107 | How We Know Which Way | Navigation tools and landmarks help us find our way. |
| 123 | How TV Shows Pictures | Television transmits moving images through electronic signal... |
| 132 | Why We Dream | Dreams may help process emotions and consolidate memories. |
| 134 | How Farming Changed Everything | Agriculture allowed humans to settle and build civilizations... |
| 157 | Why Every Culture Dances | Dance is a universal form of human expression and celebratio... |
| 161 | What Clothes Communicate | Clothing communicates identity, status, and culture. |
| 163 | Why We Compete | Competition drives improvement and reveals our capabilities. |
| 166 | When Sounds Agree | Harmony occurs when musical notes sound pleasing together. |
| 170 | Finding What Was Already There | Discovery reveals what exists but was previously unknown. |
| 173 | The Strongest Shape | Triangles distribute force efficiently and are structurally ... |
| 178 | What's Beyond Earth | Space contains planets, stars, galaxies, and vast distances. |
| 180 | Why Difference Catches Your Eye | Contrast makes things stand out and captures attention. |
| 191 | Trusting Yourself | Self-trust comes from knowing your values and capabilities. |
| 192 | Knowing Who You Are | Identity encompasses our values, experiences, and sense of s... |
| 197 | Making Things Right | Repairing harm restores relationships and trust. |
| 198 | Working Together | Collaboration combines different strengths for better outcom... |
| 199 | Why We Compete | Competition drives improvement and reveals our capabilities. |
| 211 | When Everything Changes Fast | Rapid change requires adaptation and resilience. |
| 224 | Speeding Up and Slowing Down | Acceleration is the rate of change in speed. |
| 232 | When Things Come Apart | Decomposition breaks matter into simpler components. |
| 233 | Using Things Again | Recycling transforms waste into new useful materials. |
| 250 | Hiding in Plain Sight | Camouflage helps animals blend into their environment. |
| 263 | How Life Makes More Life | Reproduction creates new generations of living things. |
| 271 | Things That Grow Back | Regeneration allows some organisms to regrow lost parts. |
| 280 | Using Things Again | Recycling transforms waste into new useful materials. |
| 287 | Learning From Getting It Wrong | Mistakes provide valuable feedback for improvement. |
| 292 | Different Ways of Being Smart | Intelligence takes many forms beyond traditional measures. |
| 307 | Brain and Body Working Together | Collaboration combines different strengths for better outcom... |
| 314 | Ready Before It Happens | Preparation helps us respond effectively to challenges. |
| 319 | Stopping Hurt Before It Starts | Prevention avoids problems before they occur. |
| 323 | Who You Are | Identity encompasses our values, experiences, and sense of s... |
| 324 | Showing Who You Are | Identity encompasses our values, experiences, and sense of s... |
| 325 | What Makes You You | Individuality comes from our unique combination of traits an... |
| 332 | Being Part of a Place | Belonging connects us to communities and locations. |
| 344 | Accepting What's Given | Receiving gracefully honors the generosity of others. |
| 349 | The Voice in Your Head | Self-talk shapes how we think and feel about ourselves. |
| 350 | Words That Shape Beliefs | Affirmations are positive statements that influence our mind... |
| 353 | Being Where You Are | Presence means focusing attention on the current moment. |
| 357 | Deciding Right From Wrong | Moral reasoning guides ethical decision-making. |
| 359 | Who You Are When No One's Looking | Identity encompasses our values, experiences, and sense of s... |
| 361 | Looking Back to Learn | Reflection examines past experiences to gain wisdom. |
| 364 | Starting Fresh | New beginnings offer opportunities for growth and change. |
| 365 | 365 Days of Growing | A year of daily learning adds up to remarkable growth. |

---

## Recommended Actions

1. Review each mismatch in Supabase `core_lessons` table
2. Determine if title or objective needs updating
3. After fixes, re-run sync scripts
