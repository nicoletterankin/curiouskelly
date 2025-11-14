# ReinMaker Game Portfolio - Implementation Roadmap

**Document Type:** Implementation Roadmap  
**Date:** November 2025  
**Based On:** Decision Framework (`docs/reinmaker/DECISION_FRAMEWORK.md`)

---

## 🎯 Strategic Decision

**Decision:** Launch Runner Game First, Evaluate RPG Later

**Rationale:**
1. Runner Game is 90% complete (13/25 assets done)
2. Low risk, fast to market (2-3 months vs 12-18 months)
3. Validates ReinMaker brand quickly
4. Gathers user feedback to inform RPG development
5. Asset reuse: Tribes, Quests, Knowledge Stones work for both games

**Timeline:** 3 months to launch Runner Game, then evaluate RPG

---

## 📅 3-Month Launch Roadmap

### **MONTH 1: Complete Assets & Core Gameplay**

#### Week 1-2: Missing Core Assets
**Goal:** Generate remaining 12 missing assets

**Tasks:**
- [ ] **A3. Ground Stripe** (`assets/ground_stripe.png`)
  - Specs: 60x6px, rounded ends, off-white (#F2F7FA)
  - Tool: AI generation or simple graphic design
  - Time: 1 hour
  - Cost: $0-50

- [ ] **B2. Ground Texture** (`assets/ground_tex.png`)
  - Specs: 512x64px, seamless tile, dark steel-stone texture
  - Tool: AI generation or texture creation
  - Time: 2-3 hours
  - Cost: $0-100

- [ ] **C1. Logo Square** (`marketing/square-600.png`)
  - Specs: 600x600px, transparent BG
  - Tool: Resize existing logo or regenerate
  - Time: 1 hour
  - Cost: $0-50

**Deliverables:**
- ✅ 3 core assets generated
- ✅ Assets validated (dimensions, format)

**Success Criteria:**
- All assets match specifications
- Assets pass quality validation

---

#### Week 3-4: Missing Lore Assets
**Goal:** Generate 7 Tribe Banners

**Tasks:**
- [ ] **D2. Tribe Banners** (7 total)
  - Light, Stone, Metal, Code, Air, Water, Fire
  - Specs: 128x256px each, tileable vertically, transparent BG
  - Tool: AI generation (use prompts from `runner_game_asset_prompts.md`)
  - Time: 3-4 hours per banner = 21-28 hours total
  - Cost: $300-600 (AI generation credits)

**Deliverables:**
- ✅ 7 Tribe Banners generated
- ✅ Assets validated and catalogued

**Success Criteria:**
- All banners match tribe color schemes
- Vertical tiling works correctly
- Assets are optimized for mobile

---

#### Week 5-6: Core Gameplay Implementation
**Goal:** Implement runner game mechanics

**Tasks:**
- [ ] **Core Runner Mechanics**
  - Character movement (run, jump)
  - Obstacle collision detection
  - Knowledge Stone collection
  - Score/XP system
  - Time: 20-30 hours
  - Cost: $0 (development time)

- [ ] **Quest Integration**
  - Connect quest system to gameplay
  - Quest completion tracking
  - Rewards system (XP, Knowledge Stones)
  - Time: 10-15 hours
  - Cost: $0 (development time)

- [ ] **UI Foundation**
  - HUD (score, XP, Knowledge Stones)
  - Pause menu
  - Quest tracker overlay
  - Settings menu
  - Time: 15-20 hours
  - Cost: $0 (development time)

**Deliverables:**
- ✅ Playable runner game prototype
- ✅ Quest system integrated
- ✅ Basic UI implemented

**Success Criteria:**
- Game runs smoothly (60 FPS)
- Core mechanics feel responsive
- Quest system works end-to-end

---

#### Week 7-8: Testing & Polish
**Goal:** Polish gameplay and fix bugs

**Tasks:**
- [ ] **Playtesting Sessions**
  - Internal testing (yourself)
  - Beta testers (5-10 people)
  - Gather feedback
  - Time: 10-15 hours
  - Cost: $0

- [ ] **Bug Fixes**
  - Fix critical bugs
  - Fix major gameplay issues
  - Performance optimization
  - Time: 15-20 hours
  - Cost: $0 (development time)

- [ ] **Asset Optimization**
  - Compress assets for mobile
  - Optimize sprite sheets
  - Reduce file sizes
  - Time: 5-10 hours
  - Cost: $0

**Deliverables:**
- ✅ Stable, polished game
- ✅ Performance optimized
- ✅ Feedback documented

**Success Criteria:**
- No critical bugs
- Performance: 60 FPS on mid-range devices
- Positive feedback from testers

---

### **MONTH 2: Monetization & Platform Prep**

#### Week 1-2: Monetization System
**Goal:** Add IAP and analytics

**Tasks:**
- [ ] **IAP Integration**
  - Apple IAP (StoreKit)
  - Google Play Billing
  - Product catalog:
    - Remove ads ($2.99)
    - Character skins ($0.99-$4.99)
    - Knowledge Stone variants ($0.99-$1.99)
    - Starter pack ($4.99)
  - Time: 20-25 hours
  - Cost: $0 (development time)

- [ ] **Analytics Integration**
  - Downloads tracking
  - Retention metrics (D1, D7, D30)
  - Revenue tracking
  - User behavior events
  - Tool: Firebase Analytics or similar
  - Time: 10-15 hours
  - Cost: $0 (free tier)

**Deliverables:**
- ✅ IAP working on iOS and Android
- ✅ Analytics dashboard set up

**Success Criteria:**
- IAP purchases work end-to-end
- Analytics capture key events
- Revenue tracking accurate

---

#### Week 3-4: Store Listings Preparation
**Goal:** Prepare app store listings

**Tasks:**
- [ ] **iOS App Store**
  - App name, description, keywords
  - Screenshots (all required sizes)
  - App icon (1024x1024px)
  - Age rating (E for Everyone or E10+)
  - Privacy policy URL
  - Support URL
  - Time: 5-10 hours
  - Cost: $99 (Apple Developer Program)

- [ ] **Google Play Store**
  - App name, description, short description
  - Screenshots (phone, tablet, TV)
  - Feature graphic (1024x500px)
  - App icon (512x512px)
  - Age rating (Everyone or Everyone 10+)
  - Privacy policy URL
  - Data safety form
  - Time: 5-10 hours
  - Cost: $25 (one-time Google Play fee)

- [ ] **Itch.io Page**
  - Game description
  - Trailer video
  - Screenshots
  - Download links
  - Time: 3-5 hours
  - Cost: $0

**Deliverables:**
- ✅ App Store listings complete
- ✅ All required assets prepared

**Success Criteria:**
- Listings approved by stores
- All assets meet requirements
- Store pages look professional

---

#### Week 5-6: Marketing Preparation
**Goal:** Create marketing assets and strategy

**Tasks:**
- [ ] **Trailer Video**
  - Use existing cinematic script (`RMD_Opening/10_scripts/`)
  - 30-60 second trailer
  - Show gameplay + story
  - Tool: Screen recording + editing
  - Time: 10-15 hours
  - Cost: $0-500 (editing software or contractor)

- [ ] **Marketing Assets**
  - Screenshots (10-15 images)
  - Animated GIFs (3-5)
  - Press kit (logo, screenshots, description)
  - Social media graphics
  - Time: 5-10 hours
  - Cost: $0

- [ ] **Community Building**
  - Set up Discord server
  - Create Reddit subreddit (r/reinmaker)
  - Social media accounts (Twitter, Instagram)
  - Launch week content calendar
  - Time: 5-10 hours
  - Cost: $0

**Deliverables:**
- ✅ Trailer video ready
- ✅ Marketing assets complete
- ✅ Community channels set up

**Success Criteria:**
- Trailer captures game essence
- Assets are high quality
- Community channels active

---

#### Week 7-8: Soft Launch (Beta)
**Goal:** Beta test with limited audience

**Tasks:**
- [ ] **Beta Test Setup**
  - TestFlight (iOS) or Google Play Beta (Android)
  - 50-100 beta testers
  - Feedback form/survey
  - Time: 5-10 hours
  - Cost: $0

- [ ] **Beta Testing Period**
  - 2-week beta period
  - Monitor feedback
  - Fix critical bugs
  - Gather analytics
  - Time: 10-15 hours
  - Cost: $0

- [ ] **Pre-Launch Checklist**
  - Final bug fixes
  - Performance optimization
  - Store listing updates
  - Marketing campaign prep
  - Time: 10-15 hours
  - Cost: $0

**Deliverables:**
- ✅ Beta test complete
- ✅ Feedback documented
- ✅ Pre-launch checklist complete

**Success Criteria:**
- No critical bugs found
- Positive feedback from beta testers
- Performance metrics acceptable

---

### **MONTH 3: Launch & Post-Launch**

#### Week 1-2: Launch Week
**Goal:** Launch game on all platforms

**Tasks:**
- [ ] **Store Submissions**
  - Submit to App Store (review: 1-7 days)
  - Submit to Google Play (review: 1-3 days)
  - Publish to Itch.io (immediate)
  - Time: 2-3 hours
  - Cost: $0 (already paid fees)

- [ ] **Marketing Campaign**
  - Launch day social media posts
  - Press outreach (10-20 outlets)
  - Influencer outreach (5-10 influencers)
  - Reddit posts (r/gaming, r/iosgaming, r/androidgaming)
  - Time: 10-15 hours
  - Cost: $0-1,000 (influencer fees optional)

- [ ] **Launch Monitoring**
  - Monitor downloads
  - Track reviews
  - Respond to user feedback
  - Monitor crashes/errors
  - Time: 5-10 hours
  - Cost: $0

**Deliverables:**
- ✅ Game live on all platforms
- ✅ Marketing campaign active
- ✅ Launch metrics tracked

**Success Criteria:**
- 500+ downloads in first week
- 4+ star average rating
- No critical crashes

---

#### Week 3-4: Post-Launch Support
**Goal:** Fix issues and engage community

**Tasks:**
- [ ] **Bug Fixes**
  - Fix critical bugs from launch
  - Address user feedback
  - Performance improvements
  - Time: 15-20 hours
  - Cost: $0

- [ ] **Content Updates**
  - Add new quests (2-3 quests)
  - New Knowledge Stone variants
  - Seasonal events (optional)
  - Time: 10-15 hours
  - Cost: $0

- [ ] **Community Engagement**
  - Respond to reviews
  - Discord/Reddit engagement
  - Social media posts
  - Time: 5-10 hours
  - Cost: $0

**Deliverables:**
- ✅ Version 1.1 update released
- ✅ Community actively engaged
- ✅ User feedback incorporated

**Success Criteria:**
- Positive user sentiment
- Active community
- Retention improving

---

#### Week 5-6: Metrics Evaluation
**Goal:** Analyze launch performance

**Tasks:**
- [ ] **Metrics Analysis**
  - Downloads (target: 1,000+)
  - Retention: D1 (target: 30%+), D7 (target: 15%+), D30 (target: 5%+)
  - Revenue (target: $500-1,000/month)
  - User ratings (target: 4+ stars)
  - Time: 5-10 hours
  - Cost: $0

- [ ] **User Feedback Review**
  - App Store reviews
  - Google Play reviews
  - Community feedback (Discord, Reddit)
  - Survey responses
  - Time: 5-10 hours
  - Cost: $0

- [ ] **Competitive Analysis**
  - Compare to similar games
  - Identify strengths/weaknesses
  - Market positioning
  - Time: 5-10 hours
  - Cost: $0

**Deliverables:**
- ✅ Metrics dashboard complete
- ✅ Feedback analysis document
- ✅ Competitive analysis report

**Success Criteria:**
- Metrics meet or exceed targets
- Clear understanding of user needs
- Clear market positioning

---

#### Week 7-8: Decision Point
**Goal:** Decide next phase based on results

**Tasks:**
- [ ] **Review Success Criteria**
  - Compare metrics to targets
  - Assess user feedback
  - Evaluate resource availability
  - Time: 5-10 hours
  - Cost: $0

- [ ] **Make Decision**
  - **Option A:** Continue Runner Game iteration
  - **Option B:** Start RPG development planning
  - **Option C:** Pivot strategy
  - Document decision and rationale
  - Time: 2-3 hours
  - Cost: $0

- [ ] **Plan Next Phase**
  - If continuing Runner: Roadmap for v2.0
  - If starting RPG: Create RPG development roadmap
  - If pivoting: Define new strategy
  - Time: 5-10 hours
  - Cost: $0

**Deliverables:**
- ✅ Decision documented
- ✅ Next phase roadmap created
- ✅ Team aligned on direction

**Success Criteria:**
- Clear decision made
- Roadmap for next 3-6 months
- Team alignment achieved

---

## 📊 Success Metrics & Decision Criteria

### Launch Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Downloads** | 1,000+ | Analytics dashboard |
| **D1 Retention** | 30%+ | Analytics dashboard |
| **D7 Retention** | 15%+ | Analytics dashboard |
| **D30 Retention** | 5%+ | Analytics dashboard |
| **Revenue** | $500-1,000/month | IAP analytics |
| **User Rating** | 4+ stars | App Store/Play Store |
| **Reviews** | 50+ reviews | App Store/Play Store |
| **Community** | 100+ members | Discord/Reddit |

### RPG Development Decision Criteria

**Proceed with RPG IF:**
- ✅ Runner Game has 5,000+ downloads
- ✅ Retention: D1 ≥ 30%, D7 ≥ 15%, D30 ≥ 5%
- ✅ Revenue: $1,000+/month OR positive user feedback (4.5+ stars)
- ✅ Resources available: Team (2-3 people) OR funding ($50,000+) OR 18 months solo
- ✅ User feedback confirms story/educational value

**Do NOT proceed with RPG IF:**
- ❌ Runner Game fails to gain traction (<1,000 downloads)
- ❌ Low retention (<5% D30)
- ❌ No revenue AND negative user feedback
- ❌ No resources available (solo, no funding, <6 months)

---

## 💰 Budget Summary

### Phase 1: Runner Game Launch (Months 1-3)

| Category | Item | Cost |
|----------|------|------|
| **Assets** | Missing 12 assets | $500-1,000 |
| **Development** | Your time (8 weeks) | $0 |
| **Platform Fees** | Apple Developer ($99/year) | $99 |
| **Platform Fees** | Google Play ($25 one-time) | $25 |
| **Marketing** | Trailer video (optional) | $0-500 |
| **Marketing** | Influencer outreach (optional) | $0-1,000 |
| **Total** | | **$624-2,624** |

### Phase 2: RPG Development (If Proceeding)

| Category | Item | Cost |
|----------|------|------|
| **Art Assets** | 3D models, environments | $20,000-50,000 |
| **Development** | Game systems (12-18 months) | $30,000-100,000 |
| **Audio** | Music, sound effects | $5,000-15,000 |
| **Marketing** | Launch campaign | $10,000-50,000 |
| **Total** | | **$65,000-215,000** |

---

## 🎯 Key Milestones

### Month 1 Milestones
- ✅ All 25 assets complete
- ✅ Core gameplay implemented
- ✅ Quest system integrated
- ✅ Basic UI complete

### Month 2 Milestones
- ✅ IAP system working
- ✅ Analytics integrated
- ✅ Store listings approved
- ✅ Beta test complete

### Month 3 Milestones
- ✅ Game launched on all platforms
- ✅ 1,000+ downloads achieved
- ✅ Metrics evaluated
- ✅ Decision on next phase made

---

## 📝 Tracking & Updates

**Update Frequency:** Weekly status updates

**Status Tracking:**
- ✅ Complete
- 🟡 In Progress
- 🔴 Blocked
- ⚪ Not Started

**Risks & Mitigations:**
- **Risk:** Asset generation delays
  - **Mitigation:** Start early, use multiple tools
- **Risk:** App Store rejection
  - **Mitigation:** Follow guidelines, test thoroughly
- **Risk:** Low downloads
  - **Mitigation:** Marketing campaign, influencer outreach
- **Risk:** Resource constraints
  - **Mitigation:** Focus on MVP, prioritize core features

---

## 📚 References

- **Decision Framework:** `docs/reinmaker/DECISION_FRAMEWORK.md`
- **RPG Positioning:** `docs/reinmaker/MOBILE_RPG_POSITIONING.md`
- **Asset List:** `REINMAKER_COMPLETE_ASSET_LIST.md`
- **Asset Prompts:** `runner_game_asset_prompts.md`
- **Quest System:** `curious-kellly/backend/config/reinmaker/quests/`

---

**Document Status:** ✅ Ready for Execution  
**Next Action:** Review roadmap, confirm timeline, begin Month 1 tasks









