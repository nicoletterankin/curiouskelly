# UX Audit — December 2025
## Dual Perspective: Learner + Creative Agency

---

## 🎯 HOMEPAGE (`/`)

### As a Learner
| Aspect | Observation | Severity |
|--------|-------------|----------|
| **First Impression** | "Curious? Always." is compelling but vague - what AM I getting? | 🟡 Medium |
| **Value Prop** | "Learn something new every day" - clear but generic | 🟡 Medium |
| **Sign-up Friction** | Immediately asks for Google/Apple/email - feels pushy before showing value | 🔴 High |
| **Free Access** | "No account? Start learning now" link is tiny and easy to miss | 🔴 High |
| **Trust Signals** | No social proof, testimonials, or "used by X learners" | 🔴 High |
| **What is Kelly?** | Never clearly explains Kelly is an AI teacher until About page | 🟡 Medium |
| **Generation Slider** | Silent Gen/Boomer/etc. buttons - unclear what they do | 🟡 Medium |
| **Pricing Section** | "Basic personalization" in Free tier - what does that mean? | 🟡 Medium |
| **Affiliate Section** | "Earnings estimate" slider on homepage feels premature/salesy | 🟢 Low |

### As a Creative Agency
| Aspect | Observation | Severity |
|--------|-------------|----------|
| **Typography** | Instrument Sans + Newsreader is elegant. Consistent hierarchy. | ✅ Good |
| **Color Palette** | Dark mode with blue accents - cohesive but very "tech startup" | 🟡 Consider |
| **Hero Visual** | Kelly image exists but no animation/video on homepage hero | 🔴 High |
| **White Space** | Generally good spacing, breathable layout | ✅ Good |
| **CTA Hierarchy** | "Start Free" and "Sign In" both prominent - unclear primary action | 🟡 Medium |
| **Motion** | Minimal to none. Page feels static. | 🔴 High |
| **Footer Redundancy** | "Support" heading appears twice with different content | 🟢 Low |
| **Holiday Banner** | "Shop Holiday Gifts" feels generic, could be more festive | 🟢 Low |
| **Card Design** | Pricing cards are clean but lack visual differentiation beyond copy | 🟡 Medium |

---

## 📚 LEARN PAGE (`/learn`)

### As a Learner
| Aspect | Observation | Severity |
|--------|-------------|----------|
| **Cognitive Load** | 15+ toolbar buttons across top - overwhelming | 🔴 High |
| **Where's the Lesson?** | Main content area appears empty/loading on first visit | 🔴 High |
| **Age/Tone/Language** | Controls visible but no guidance on what they change | 🟡 Medium |
| **Phase Progress** | "Hook - Question 1 - 2 - 3 - Complete" visible but small | 🟡 Medium |
| **Previous/Next Day** | Can browse other days - great for exploration | ✅ Good |
| **Chat Input** | "Say something to Kelly" - is this live? What happens? | 🟡 Medium |
| **Share & Earn Modal** | Shows "Loading..." for referral link - feels broken | 🔴 High |
| **Search** | "No lessons found for ''" - empty search shows error state | 🟡 Medium |

### As a Creative Agency
| Aspect | Observation | Severity |
|--------|-------------|----------|
| **Layout** | TikTok-style full-screen is modern and immersive | ✅ Good |
| **Toolbar Density** | Too many icons, needs progressive disclosure | 🔴 High |
| **Kelly Avatar** | Static 2D image, no animation during lessons | 🔴 High |
| **Bottom Navigation** | Home/Calendar/Learn/Me/Settings - clean mobile nav | ✅ Good |
| **Mode Toggle** | 2D/3D toggle exists but 3D doesn't work | 🟡 Medium |
| **Language Flags** | Flag emojis work well for quick recognition | ✅ Good |
| **Modals** | Multiple overlapping modals possible - z-index issues? | 🟡 Medium |
| **Accessibility** | "Skip to main content" link exists - good | ✅ Good |

---

## 💰 PRICING PAGE (`/pricing`)

### As a Learner
| Aspect | Observation | Severity |
|--------|-------------|----------|
| **Clarity** | "Simple, transparent pricing" header - delivers on promise | ✅ Good |
| **Family Access** | "Whole family access" prominent - good family value | ✅ Good |
| **Language Support** | "English, Spanish & Portuguese" - could expand | 🟢 Low |
| **Age Range** | "Ages 2-102" - memorable but the 2-year-old claim feels... ambitious | 🟡 Medium |
| **Lifetime Value** | Lifetime tier clearly positioned as premium | ✅ Good |
| **Missing Details** | No FAQs, no money-back guarantee visible | 🟡 Medium |
| **Trial Length** | 7-day trial prominent - industry standard | ✅ Good |

### As a Creative Agency
| Aspect | Observation | Severity |
|--------|-------------|----------|
| **Card Layout** | 3 tiers side-by-side, standard SaaS pattern | ✅ Good |
| **Price Visibility** | Actual prices not shown in snapshot - are they visible? | 🔴 Check |
| **Recommended Tier** | No visual indicator of "most popular" tier | 🟡 Medium |
| **Footer Duplication** | Same "Support" section appears twice | 🟢 Low |
| **Header Nav** | Different nav items than homepage (no Commons) | 🟡 Medium |

---

## 🎁 GIFTS PAGE (`/gifts`)

### As a Learner
| Aspect | Observation | Severity |
|--------|-------------|----------|
| **Seasonal Tie-in** | "Make This Christmas Unforgettable" - good timing | ✅ Good |
| **Gift Options** | 3mo, 6mo, 12mo, Lifetime - clear progression | ✅ Good |
| **Scheduling** | "Schedule delivery" for 12mo is thoughtful | ✅ Good |
| **Form Fields** | Clear sender/recipient separation | ✅ Good |
| **Stripe Badge** | "Secure checkout powered by Stripe" builds trust | ✅ Good |
| **Personal Message** | Optional message field - nice touch | ✅ Good |

### As a Creative Agency
| Aspect | Observation | Severity |
|--------|-------------|----------|
| **Holiday Theme** | 🎄 emoji use is festive but minimal | 🟡 Medium |
| **Card Design** | Clean gift cards but no preview of what recipient sees | 🟡 Medium |
| **Form UX** | Modal form works but could be full-page for gift flow | 🟢 Low |
| **Visual Hierarchy** | "Gift 12 Months" has 🎁 emoji, others don't - inconsistent | 🟢 Low |

---

## 🔄 CROSS-PAGE ISSUES

### Navigation Inconsistencies
| Page | Nav Items |
|------|-----------|
| Homepage | Curriculum, Commons, Pricing, Compare, About |
| Pricing | Curriculum, Compare, Gifts, About |
| Gifts | About, Curriculum, Pricing, Compare |
| Learn | (Full app nav - different paradigm) |

**Issue**: Navigation varies per page. "Commons" appears/disappears. No "Share" in main nav.

### Copy Alignment Issues
| Location | Current | Should Be |
|----------|---------|-----------|
| Pricing | "Daily lessons for ages 2-102" | "Designed for any learner" |
| Various | "personalized to your age" | "adapts to how you learn" |
| Homepage | "Basic personalization" | Unclear what this means |

### Missing Elements
- ❌ No testimonials anywhere
- ❌ No "X learners joined this week" social proof
- ❌ No sample lesson preview (video/screenshot)
- ❌ No "What you'll learn this week" teaser
- ❌ No clear Kelly introduction on homepage
- ❌ No FAQ section on pricing
- ❌ No visible progress/streak for returning users

---

## 🎨 AGENCY RECOMMENDATIONS

### Immediate (Week 1)
1. **Add hero video** showing Kelly in action (committed but not deployed)
2. **Add social proof section** with trust badges
3. **Reduce learn.html toolbar** - use progressive disclosure
4. **Fix "Loading..." states** with branded messages

### Short-term (Week 2-3)
5. **Unify navigation** across all pages
6. **Add testimonials** or "Learner Stories" section
7. **Create onboarding flow** before dumping users into learn.html
8. **Add sample lesson preview** on homepage (30-sec teaser)

### Medium-term (Month 1)
9. **Motion design system** - micro-interactions, page transitions
10. **Kelly personality showcase** - homepage should show her teaching
11. **Progress/streak visibility** for returning users
12. **A/B test hero copy** - "Curious? Always" vs clearer value prop

### Brand Alignment Checklist
- [ ] Update all "personalized for age" copy *(committed, awaiting deploy)*
- [ ] Ensure Kelly is introduced before asking for sign-up
- [ ] Add "Safe for any learner" messaging prominently
- [ ] Remove "ages 2-102" if Kelly doesn't truly adapt for toddlers
- [ ] Consider light mode option for accessibility/preference

---

## 📊 PRIORITY MATRIX

| Impact | Effort | Item |
|--------|--------|------|
| 🔴 High | 🟢 Low | Fix copy inconsistencies |
| 🔴 High | 🟡 Med | Add social proof/trust signals |
| 🔴 High | 🟡 Med | Simplify learn.html toolbar |
| 🔴 High | 🔴 High | Add hero video (in progress) |
| 🟡 Med | 🟢 Low | Unify navigation |
| 🟡 Med | 🟡 Med | Add testimonials |
| 🟡 Med | 🔴 High | Onboarding flow |
| 🟢 Low | 🟢 Low | Fix footer duplication |

---

## 💡 THE BIG INSIGHT

**The product (learn.html) is impressive. The marketing (homepage) undersells it.**

A first-time visitor sees:
- Generic "learn something new" messaging
- Immediate sign-up prompts
- No preview of what they're getting
- No Kelly personality showcase

They should see:
- Kelly teaching a snippet
- Clear "365 lessons, any age, any mood, any language"
- Social proof from real learners
- One-click "Try today's lesson free" without account

**The gap isn't design quality—it's story. Tell the story of daily learning before asking for commitment.**

---

*Audit completed December 9, 2025*


