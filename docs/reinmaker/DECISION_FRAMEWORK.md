# ReinMaker Game Portfolio - Decision Framework & Roadmap

**Document Type:** Strategic Decision Document  
**Date:** November 2025  
**Status:** Decision Framework - Ready for Review

---

## 🎯 The Core Question

**Should we build the Mobile RPG alongside the Runner Game, or instead of it?**

This is a strategic decision that affects:
- Resource allocation (time, money, team)
- Market positioning
- Brand identity
- Go-to-market strategy
- Technical complexity

---

## 📊 Understanding the Tradeoffs

### Tradeoff 1: Target Audience Overlap

#### Runner Game Audience
- **Age:** All ages (2-102)
- **Income:** Not specified (broader appeal)
- **Psychographics:** Educational focus, family-friendly
- **Use Case:** Daily micro-learning, habit formation
- **Play Style:** Quick sessions (5-15 min), casual

#### Mobile RPG Audience
- **Age:** 25-44 years old
- **Income:** $65,000+/year
- **Psychographics:** Nostalgic, analytical, traditional RPG fans
- **Use Case:** Immersive story experience, meaningful progression
- **Play Style:** Longer sessions (30-60+ min), mid-core

**Key Insight:** Different audiences = different markets. They don't compete directly, but they don't share resources either.

---

### Tradeoff 2: Development Complexity

#### Runner Game Complexity
- **Art Assets:** 25+ 2D sprites (mostly done - 13/25 complete)
- **Code:** Simple game loop (run, jump, collect)
- **Systems:** Basic quest completion, XP, Knowledge Stones
- **Backend:** Quest API, manifest system (already built)
- **Time Estimate:** 2-4 weeks to finish remaining assets, 4-6 weeks for gameplay polish
- **Cost:** Low ($1,000-3,000 for remaining assets)

#### Mobile RPG Complexity
- **Art Assets:** 3D character models, environments, animations (from scratch)
- **Code:** Complex systems (combat, Strife, Followers, Disciplines)
- **Systems:** Non-kill combat, Zone Strife, Follower buffs, Discipline trees
- **Backend:** New systems (combat engine, zone state, follower AI)
- **Time Estimate:** 12-18 months for full RPG
- **Cost:** High ($50,000-200,000+ depending on team size)

**Key Insight:** Runner is 90% done. RPG is 0% done. Starting RPG means abandoning runner progress.

---

### Tradeoff 3: Market Positioning

#### Option A: Runner Game First
- **Brand Identity:** Educational, accessible, family-friendly
- **Market Entry:** Fast (2-3 months to launch)
- **Revenue Potential:** Lower LTV but broader reach
- **Differentiation:** Story-driven runner (niche but clear)
- **Risk:** Lower (simpler game, less investment)

#### Option B: Mobile RPG First
- **Brand Identity:** Premium, story-driven, adult-focused
- **Market Entry:** Slow (12-18 months)
- **Revenue Potential:** Higher LTV but narrower reach
- **Differentiation:** Empathy/compassion RPG (unique positioning)
- **Risk:** Higher (complex game, large investment)

#### Option C: Both Games
- **Brand Identity:** Confusing (two different audiences)
- **Market Entry:** Slower (6-9 months for runner, 18+ for RPG)
- **Revenue Potential:** Highest (two revenue streams)
- **Differentiation:** ReinMaker as a franchise
- **Risk:** Highest (resource split, brand confusion)

---

### Tradeoff 4: Resource Allocation

#### Current Resources (Based on Codebase)
- **1 Developer** (you, based on solo-operator model)
- **Existing Assets:** Runner game 52% complete
- **Backend:** Quest system, manifest system built
- **Budget:** Unknown (likely limited for solo operation)

#### Resource Requirements

**Runner Game to Completion:**
- Art: 12 missing assets (ground stripe, texture, banners, etc.)
- Code: 4-6 weeks gameplay polish
- Testing: 2 weeks
- **Total:** 6-8 weeks, ~$2,000-3,000

**Mobile RPG to MVP:**
- Art: 3D models, environments, UI (6+ months)
- Code: Combat system, Strife system, Disciplines (6+ months)
- Design: Game design docs, balance testing (3+ months)
- **Total:** 12-18 months, $50,000-200,000+

**Key Insight:** Solo developer + limited budget = Runner Game is feasible, RPG is not without team/funding.

---

### Tradeoff 5: Strategic Value

#### Runner Game Value
- ✅ **Quick Launch:** Get to market fast
- ✅ **Validate Brand:** Test ReinMaker concept
- ✅ **Learn:** Build audience, gather feedback
- ✅ **Asset Reuse:** Knowledge Stones, Tribes, Quests can be reused
- ✅ **Foundation:** Can evolve into RPG later
- ❌ **Lower Revenue:** Casual games monetize less
- ❌ **Age Range:** Too broad (harder to target)

#### Mobile RPG Value
- ✅ **Premium Positioning:** Higher revenue potential
- ✅ **Clear Audience:** 25-44 with disposable income
- ✅ **Unique:** Empathy/compassion RPG is differentiated
- ✅ **Expert Backing:** Dalai Lama partnership adds credibility
- ❌ **Long Timeline:** 12-18 months before launch
- ❌ **High Risk:** Complex game, high investment
- ❌ **Missed Opportunity:** Runner game assets wasted

---

## 🎲 Decision Framework

### Decision Matrix

Rate each option on a scale of 1-5 for each criterion:

| Criterion | Runner Game | Mobile RPG | Both Games |
|-----------|------------|------------|------------|
| **Time to Market** | ⭐⭐⭐⭐⭐ (Fast) | ⭐⭐ (Slow) | ⭐⭐ (Very Slow) |
| **Resource Required** | ⭐⭐⭐⭐⭐ (Low) | ⭐⭐ (High) | ⭐ (Very High) |
| **Revenue Potential** | ⭐⭐⭐ (Moderate) | ⭐⭐⭐⭐⭐ (High) | ⭐⭐⭐⭐ (High) |
| **Risk Level** | ⭐⭐⭐⭐⭐ (Low) | ⭐⭐ (High) | ⭐ (Very High) |
| **Brand Clarity** | ⭐⭐⭐⭐ (Clear) | ⭐⭐⭐⭐ (Clear) | ⭐⭐ (Confusing) |
| **Feasibility (Solo)** | ⭐⭐⭐⭐⭐ (Yes) | ⭐⭐ (No) | ⭐ (No) |
| **Asset Reuse** | ⭐⭐⭐⭐⭐ (Can evolve) | ⭐⭐⭐ (Can reuse) | ⭐⭐⭐⭐ (Both) |

**Scoring:**
- Runner Game: 35/35 points
- Mobile RPG: 21/35 points  
- Both Games: 19/35 points

---

## ✅ Recommended Decision Path

### **PHASE 1: Launch Runner Game (Months 1-3)**

**Why:**
1. ✅ **90% Complete:** Most assets done, systems built
2. ✅ **Low Risk:** Simple game, small investment
3. ✅ **Fast to Market:** Validate ReinMaker brand quickly
4. ✅ **Learn:** Gather user feedback, test monetization
5. ✅ **Foundation:** Can evolve into RPG later

**What to Build:**
- Complete remaining 12 assets
- Polish gameplay (4-6 weeks)
- Add monetization (IAP for skins, knowledge stones)
- Launch on iOS/Android/Itch.io
- Market to families, educators, casual gamers

**Success Metrics:**
- 1,000+ downloads in first month
- 10%+ conversion to paid
- User feedback on story/engagement
- Brand recognition

---

### **PHASE 2: Evaluate & Decide (Months 4-6)**

**Options After Runner Game Launch:**

#### Option A: Iterate on Runner Game
- Add more levels, tribes, story content
- Build audience to 10,000+ users
- Refine monetization
- **Timeline:** Ongoing
- **Cost:** Low ($500-1,000/month)

#### Option B: Start RPG Development
- **IF** Runner Game succeeds (5,000+ users, positive feedback)
- **AND** You have resources (team, funding, or 18 months)
- **THEN** Begin RPG development
- **Timeline:** 12-18 months
- **Cost:** $50,000-200,000+

#### Option C: Hybrid Approach
- **Keep Runner Game** as "ReinMaker Lite" (free, educational)
- **Build RPG** as "ReinMaker Pro" (premium, story-driven)
- **Shared Lore:** Seven Tribes, Knowledge Stones, Kelly Rein
- **Timeline:** RPG in parallel (18+ months)
- **Cost:** High ($100,000+)

---

## 🚀 Recommended Roadmap

### **DECISION: Launch Runner Game First, Evaluate RPG Later**

**Rationale:**
1. Runner Game is 90% complete - finish what you started
2. Solo developer + limited budget = Runner Game is feasible
3. Fast launch = validate brand, learn from users
4. RPG can be built later if Runner Game succeeds
5. Asset reuse: Tribes, Quests, Knowledge Stones work for both

---

## 📋 Implementation Roadmap

### **MONTH 1: Complete Runner Game Assets**

**Week 1-2: Missing Core Assets**
- [ ] Generate A3. Ground Stripe (60x6px)
- [ ] Generate B2. Ground Texture (512x64px)
- [ ] Generate C1. Logo square variant (600x600px)
- **Cost:** ~$200-400 (AI generation or contractor)

**Week 3-4: Missing Lore Assets**
- [ ] Generate D2. Tribe Banners (7 banners, 128x256px each)
- **Cost:** ~$300-600 (AI generation or contractor)

**Week 5-6: Gameplay Polish**
- [ ] Implement core runner mechanics (run, jump, collect)
- [ ] Integrate Knowledge Stones collection
- [ ] Add quest completion system
- [ ] Build basic UI (HUD, menus, quest tracker)
- **Cost:** $0 (development time)

**Week 7-8: Testing & Polish**
- [ ] Playtesting sessions
- [ ] Bug fixes
- [ ] Performance optimization
- [ ] Asset optimization
- **Cost:** $0 (development time)

---

### **MONTH 2: Monetization & Launch Prep**

**Week 1-2: Monetization**
- [ ] Add IAP system (Apple IAP + Google Play Billing)
- [ ] Cosmetic items (character skins, Knowledge Stone variants)
- [ ] Optional: Remove ads purchase
- [ ] Analytics integration (downloads, retention, revenue)
- **Cost:** $0 (development time)

**Week 3-4: Platform Preparation**
- [ ] iOS App Store listing (screenshots, description, age rating)
- [ ] Google Play Store listing
- [ ] Itch.io page (marketing assets ready)
- [ ] Privacy policy, terms of service
- [ ] App Store compliance (COPPA if targeting kids)
- **Cost:** $124 (Apple $99/year + Google $25 one-time)

**Week 5-6: Marketing Prep**
- [ ] Trailer video (use existing cinematic script)
- [ ] Screenshots, GIFs
- [ ] Press kit
- [ ] Social media accounts
- [ ] Community building (Discord, Reddit)
- **Cost:** $0-500 (depending on video production)

**Week 7-8: Soft Launch**
- [ ] Beta test with 50-100 users
- [ ] Gather feedback
- [ ] Fix critical bugs
- [ ] Final polish
- **Cost:** $0 (development time)

---

### **MONTH 3: Launch & Post-Launch**

**Week 1-2: Launch**
- [ ] App Store submission
- [ ] Google Play submission
- [ ] Itch.io publication
- [ ] Marketing campaign (social media, influencers)
- [ ] Monitor reviews, respond to feedback
- **Cost:** $0-1,000 (marketing budget)

**Week 3-4: Post-Launch Support**
- [ ] Bug fixes based on user feedback
- [ ] Content updates (new levels, quests)
- [ ] Community engagement
- [ ] Analytics review (retention, revenue)
- **Cost:** $0 (development time)

**Week 5-6: Evaluation Period**
- [ ] Analyze metrics:
  - Downloads (target: 1,000+)
  - Retention (D1: 30%+, D7: 15%+, D30: 5%+)
  - Revenue (target: $500-1,000/month)
  - User feedback (target: 4+ stars)
- [ ] Make decision: Iterate, pivot, or start RPG

**Week 7-8: Decision Point**
- [ ] **IF successful:** Continue Runner Game iteration OR start RPG planning
- [ ] **IF not successful:** Pivot strategy OR sunset game
- [ ] Document learnings
- [ ] Plan next phase

---

## 💰 Budget Estimate

### **Phase 1: Runner Game Launch (Months 1-3)**

| Item | Cost |
|------|------|
| Missing Assets (12 assets) | $500-1,000 |
| Development Time (8 weeks) | $0 (your time) |
| App Store Fees | $124 |
| Marketing (optional) | $0-1,000 |
| **Total** | **$624-2,124** |

### **Phase 2: RPG Development (If Proceeding)**

| Item | Cost |
|------|------|
| 3D Art Assets | $20,000-50,000 |
| Game Development (12-18 months) | $30,000-100,000 |
| Sound/Music | $5,000-15,000 |
| Marketing | $10,000-50,000 |
| **Total** | **$65,000-215,000** |

---

## 🎯 Success Criteria for RPG Decision

**Proceed with RPG IF:**
- ✅ Runner Game has 5,000+ downloads
- ✅ Retention: D1 ≥ 30%, D7 ≥ 15%, D30 ≥ 5%
- ✅ Revenue: $1,000+/month or positive user feedback
- ✅ You have resources: Team (2-3 people) OR funding ($50,000+) OR 18 months
- ✅ User feedback confirms story/educational value

**Do NOT proceed with RPG IF:**
- ❌ Runner Game fails to gain traction (<1,000 downloads)
- ❌ Low retention (<5% D30)
- ❌ No revenue or negative user feedback
- ❌ No resources available (solo, no funding, <6 months)

---

## 📝 Decision Log

### Decision Date: [TO BE FILLED]
### Decision Maker: [TO BE FILLED]

**Decision:**
- [ ] Launch Runner Game First (Recommended)
- [ ] Start Mobile RPG Development
- [ ] Build Both Games in Parallel
- [ ] Other: _________________

**Rationale:**
[To be filled based on discussion]

**Next Steps:**
[To be filled based on decision]

**Review Date:** [Set for 3 months after launch]

---

## 📚 References

- **Runner Game Assets:** `REINMAKER_COMPLETE_ASSET_LIST.md`
- **RPG Positioning:** `docs/reinmaker/MOBILE_RPG_POSITIONING.md`
- **Quest System:** `curious-kellly/backend/config/reinmaker/quests/`
- **Tribe Definitions:** `curious-kellly/backend/config/reinmaker/tribes/`

---

**Document Status:** ✅ Ready for Decision  
**Next Action:** Review this framework, make decision, fill in Decision Log section









