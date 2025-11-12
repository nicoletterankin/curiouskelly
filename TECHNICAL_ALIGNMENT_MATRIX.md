# Technical Alignment Matrix
**Mapping UI-TARS Assets → Curious Kellly Requirements**

---

## 🎯 Overview

This document maps your existing working components to the Curious Kellly production requirements, identifies gaps, and provides concrete migration paths.

---

## 📊 Component Mapping

### 1. Voice & Audio Pipeline

| **Existing Component** | **Location** | **CK Requirement** | **Migration Path** | **Effort** |
|----------------------|--------------|-------------------|-------------------|-----------|
| ElevenLabs TTS | `lesson-player/generate_audio.py` | Realtime voice (R1) | Keep as fallback; add OpenAI Realtime API | **3 days** |
| Audio playback | `lesson-player/script.js` | Audio sync | Migrate to Flutter `just_audio` | **1 day** |
| Audio2Face integration | `kelly_audio2face/` | Lip-sync (R2) | Map A2F blendshapes → OpenAI visemes | **2 days** |

**Status**: 🟡 Partial - ElevenLabs works, need realtime voice  
**Priority**: **P0** - Critical for launch  
**Next Action**: Add OpenAI Realtime API client to backend

---

### 2. Avatar Rendering

| **Existing Component** | **Location** | **CK Requirement** | **Migration Path** | **Effort** |
|----------------------|--------------|-------------------|-------------------|-----------|
| Unity avatar | `digital-kelly/engines/kelly_unity_player/` | 60fps avatar (R2) | Add gaze tracking + micro-expressions | **5 days** |
| Blendshape driver | `digital-kelly/engines/.../BlendshapeDriver.cs` | Viseme sync | Update to handle realtime stream | **2 days** |
| Flutter embed | `digital-kelly/apps/kelly_app_flutter/` | Mobile app | Migrate to `curious-kellly/mobile/` | **1 day** |
| Kelly 3D model | `iLearnStudio/` | Avatar mesh | Export FBX with blendshapes | **2 days** |

**Status**: 🟢 Good foundation - Unity + Flutter working  
**Priority**: **P1** - Important but can iterate  
**Next Action**: Profile frame rate on target devices

---

### 3. Lesson System

| **Existing Component** | **Location** | **CK Requirement** | **Migration Path** | **Effort** |
|----------------------|--------------|-------------------|-------------------|-----------|
| PhaseDNA schema | `lesson-player/lesson-dna-schema.json` | Lesson planner (R3) | Migrate to backend | **1 day** |
| Age adaptation | `lesson-player/script.js` | Age buckets | Keep logic, expose via API | **2 days** |
| Teaching moments | `lessons/leaves-change-color.json` | Teaching moments | Use existing format | **0 days** |
| Lesson player UI | `lesson-player/index.html` | Dev tool | Keep for content authoring | **0 days** |

**Status**: 🟢 Excellent - Fully working prototype  
**Priority**: **P0** - Use as reference implementation  
**Next Action**: Migrate schema to backend, create API endpoints

---

### 4. Content Pipeline

| **Existing Component** | **Location** | **CK Requirement** | **Migration Path** | **Effort** |
|----------------------|--------------|-------------------|-------------------|-----------|
| "Leaves Change Color" | `lessons/leaves-change-color.json` | 90 lessons (3 tracks) | Use as template for 89 more | **40 days** |
| Audio generation | `lesson-player/generate_audio.py` | Audio for all lessons | Batch generate for all lessons | **3 days** |
| Kelly avatar assets | `kelly_pack/` | Avatar images | Generate variants as needed | **1 day** |

**Status**: 🟡 1 complete lesson, need 89 more  
**Priority**: **P0** - Content is the product  
**Next Action**: Hire content writers or use AI-assisted authoring

---

### 5. Safety & Moderation

| **Existing Component** | **Location** | **CK Requirement** | **Migration Path** | **Effort** |
|----------------------|--------------|-------------------|-------------------|-----------|
| ❌ None | N/A | Safety router (R6) | Build from scratch | **3 days** |
| ❌ None | N/A | Moderation API | Integrate OpenAI Moderation | **1 day** |
| ❌ None | N/A | Safe completion | Custom rewrite logic | **2 days** |

**Status**: 🔴 Missing - Critical security component  
**Priority**: **P0** - Must have before any user testing  
**Next Action**: Build safety router as first backend component

---

### 6. Backend Services

| **Existing Component** | **Location** | **CK Requirement** | **Migration Path** | **Effort** |
|----------------------|--------------|-------------------|-------------------|-----------|
| ❌ None | N/A | Orchestration service | Build Node.js/Python backend | **5 days** |
| ❌ None | N/A | Vector DB (RAG) | Set up Pinecone/Qdrant | **2 days** |
| ❌ None | N/A | Session state | Redis + PostgreSQL | **2 days** |
| ❌ None | N/A | Analytics pipeline | Integrate Mixpanel/Amplitude | **1 day** |

**Status**: 🔴 Missing - No backend infrastructure  
**Priority**: **P0** - Required for all features  
**Next Action**: Scaffold backend, deploy to staging

---

### 7. Mobile Platform

| **Existing Component** | **Location** | **CK Requirement** | **Migration Path** | **Effort** |
|----------------------|--------------|-------------------|-------------------|-----------|
| Flutter app | `digital-kelly/apps/kelly_app_flutter/` | iOS/Android apps | Clean up, add IAP | **3 days** |
| ❌ No billing | N/A | Apple IAP (R8) | Integrate StoreKit | **2 days** |
| ❌ No billing | N/A | Play Billing (R8) | Integrate Play Billing v7+ | **2 days** |
| ❌ No privacy labels | N/A | Privacy compliance (R11, R12) | Complete forms | **2 days** |

**Status**: 🟡 App structure exists, missing monetization  
**Priority**: **P0** - Required for store submission  
**Next Action**: Add IAP, test in sandbox

---

### 8. Platform Extensions

| **Existing Component** | **Location** | **CK Requirement** | **Migration Path** | **Effort** |
|----------------------|--------------|-------------------|-------------------|-----------|
| ❌ None | N/A | MCP server (R10) | Build HTTP streamable endpoint | **2 days** |
| ❌ None | N/A | Apps SDK widget (R9) | Create React component | **2 days** |
| ❌ None | N/A | Claude Artifacts | Reuse lesson player as demo | **1 day** |

**Status**: 🔴 Missing - No GPT Store integration  
**Priority**: **P1** - Nice to have but not launch-critical  
**Next Action**: Build after core mobile app works

---

## 🔄 Migration Strategies

### Strategy 1: Keep & Enhance
**Use when**: Existing component works well and aligns with CK requirements

| **Component** | **Action** | **Timeline** |
|--------------|-----------|-------------|
| Lesson player | Keep as dev tool for content authoring | Keep forever |
| PhaseDNA schema | Migrate to backend, keep same structure | Week 1 |
| Age adaptation logic | Extract to library, share backend/mobile | Week 2 |
| Teaching moments | Use existing format, no changes needed | Day 1 |
| Kelly avatar assets | Use for splash screens, profile pictures | Ongoing |

### Strategy 2: Migrate & Upgrade
**Use when**: Component needs adaptation but core logic is solid

| **Component** | **Current State** | **Target State** | **Timeline** |
|--------------|------------------|-----------------|-------------|
| ElevenLabs TTS | Pre-generated MP3s | Realtime voice + fallback | Week 3 |
| Unity avatar | Basic blendshapes | 60fps + gaze + expressions | Week 4 |
| Audio2Face | Offline processing | Realtime viseme stream | Week 3 |
| Flutter app | Dev prototype | Production with IAP | Week 7-8 |

### Strategy 3: Build From Scratch
**Use when**: No existing component or requirements fundamentally different

| **Component** | **Why New?** | **Priority** | **Timeline** |
|--------------|------------|-------------|-------------|
| Backend service | No backend exists | P0 | Week 1-2 |
| Safety router | Security-critical, no shortcut | P0 | Week 1 |
| MCP server | New protocol, specific spec | P1 | Week 9 |
| IAP integration | Platform-specific, complex | P0 | Week 7 |

---

## 📈 Technical Debt & Refactoring

### Current Technical Debt

| **Issue** | **Impact** | **Refactor Needed** | **When** |
|----------|-----------|-------------------|---------|
| No TypeScript in lesson player | Low | Optional, works fine | Post-launch |
| Hardcoded API keys in examples | Medium | Move to backend .env | Week 1 |
| No unit tests for lesson schema | Medium | Add schema validation tests | Week 2 |
| Manual audio generation | High | Automate batch generation | Week 5 |
| No error boundaries in Flutter | Medium | Add error handling | Week 8 |
| No monitoring/logging | High | Add Sentry + structured logs | Week 10 |

### Refactoring Priority

**Week 1-2 (Sprint 0):**
- Remove hardcoded credentials
- Add environment variable management
- Set up linting/formatting

**Week 3-6 (Sprint 1-2):**
- Add unit tests for critical paths
- Implement error boundaries
- Add retry logic for API calls

**Week 7-10 (Sprint 3-5):**
- Add integration tests
- Set up monitoring
- Performance profiling

**Post-Launch:**
- Refactor to TypeScript (optional)
- Extract shared libraries
- Optimize bundle size

---

## 🎯 Priority Matrix (What to Build First)

### Phase 0: Infrastructure (Week 1-2)
**Goal**: Get backend running with basic functionality

```
┌─────────────────────────────────────────┐
│ 1. Backend API skeleton                 │ ← Start here
│ 2. OpenAI Realtime API integration      │
│ 3. Safety router + moderation           │
│ 4. Deploy to staging                    │
│ 5. Migrate PhaseDNA schema              │
└─────────────────────────────────────────┘
```

**Why this order?**
- Backend is foundation for everything
- Safety must be in place before user testing
- Schema migration enables content creation

**Deliverable**: Backend responds to API calls, blocks unsafe content

---

### Phase 1: Core Experience (Week 3-4)
**Goal**: Get voice + avatar working end-to-end

```
┌─────────────────────────────────────────┐
│ 1. Add Realtime voice to Flutter app    │
│ 2. Upgrade Unity avatar to 60fps        │
│ 3. Map visemes to blendshapes           │
│ 4. Test on physical devices             │
│ 5. Measure latency & frame rate         │
└─────────────────────────────────────────┘
```

**Why this order?**
- Voice is the core interaction
- Avatar brings Kelly to life
- Early testing catches performance issues

**Deliverable**: Can have a voice conversation with Kelly avatar

---

### Phase 2: Content (Week 5-6)
**Goal**: Create all 90 lessons

```
┌─────────────────────────────────────────┐
│ 1. Spanish A1 track (30 lessons)        │
│ 2. Study Skills track (30 lessons)      │
│ 3. Career Storytelling (30 lessons)     │
│ 4. Generate audio for all               │
│ 5. Populate vector DB with content      │
└─────────────────────────────────────────┘
```

**Why this order?**
- Content creation is the longest task
- Parallel work with mobile development
- RAG needs content to test against

**Deliverable**: 90 complete lessons ready to teach

---

### Phase 3: Monetization (Week 7-8)
**Goal**: Add billing & privacy compliance

```
┌─────────────────────────────────────────┐
│ 1. Apple IAP integration                │
│ 2. Google Play Billing                  │
│ 3. Privacy labels & data safety         │
│ 4. Age gate + parent consent            │
│ 5. Test in sandbox environments         │
└─────────────────────────────────────────┘
```

**Why this order?**
- IAP required for store approval
- Privacy compliance is mandatory
- Sandbox testing before production

**Deliverable**: Can purchase subscriptions in sandbox

---

### Phase 4: Extensions (Week 9)
**Goal**: Add GPT Store & Claude support

```
┌─────────────────────────────────────────┐
│ 1. Build MCP server                     │
│ 2. Create Apps SDK widget               │
│ 3. Test in ChatGPT dev mode             │
│ 4. Publish to GPT Store                 │
│ 5. Create Claude Artifact demo          │
└─────────────────────────────────────────┘
```

**Why this order?**
- MCP server is reusable infrastructure
- Widget depends on MCP server
- GPT Store submission after testing

**Deliverable**: Curious Kellly working in ChatGPT

---

### Phase 5: Quality (Week 10)
**Goal**: Analytics, testing, security

```
┌─────────────────────────────────────────┐
│ 1. Analytics pipeline + dashboards      │
│ 2. Security review                      │
│ 3. Device matrix testing                │
│ 4. Performance profiling                │
│ 5. Fix critical issues                  │
└─────────────────────────────────────────┘
```

**Why this order?**
- Analytics needed before beta launch
- Security review before user data
- Testing finds issues before launch

**Deliverable**: App meets all performance targets

---

### Phase 6: Beta (Week 11)
**Goal**: Launch beta, gather feedback

```
┌─────────────────────────────────────────┐
│ 1. TestFlight distribution (300 users)  │
│ 2. Play Internal track (300 users)      │
│ 3. Monitor metrics & crashes            │
│ 4. Avatar polish pass                   │
│ 5. UI/UX refinement                     │
└─────────────────────────────────────────┘
```

**Why this order?**
- Beta validates product-market fit
- Early feedback shapes final polish
- Crash reports guide bug fixes

**Deliverable**: Beta users loving the product

---

### Phase 7: Launch (Week 12)
**Goal**: Submit to stores & go live

```
┌─────────────────────────────────────────┐
│ 1. Create store assets                  │
│ 2. Submit to App Store                  │
│ 3. Submit to Google Play                │
│ 4. Set up support & monitoring          │
│ 5. 🚀 LAUNCH!                           │
└─────────────────────────────────────────┘
```

**Why this order?**
- Assets take time to create
- Submissions can take 1-3 days review
- Monitoring catches issues fast

**Deliverable**: Curious Kellly is LIVE! 🎉

---

## 🔧 Development Workflow

### Week 1 (Today → Day 7)

**Monday-Tuesday: Backend Setup**
```bash
# Day 1
- Scaffold Node.js backend
- Add OpenAI SDK
- Create /health endpoint
- Deploy to Render/Railway

# Day 2
- Build safety router
- Add moderation API
- Create test suite
- Validate precision/recall
```

**Wednesday-Thursday: Lesson System**
```bash
# Day 3
- Migrate PhaseDNA schema
- Create lesson loader
- Build /session/start endpoint

# Day 4
- Set up Pinecone/Qdrant
- Populate with sample content
- Create /rag/search endpoint
```

**Friday: Integration Testing**
```bash
# Day 5
- Test all backend endpoints
- Load test with 100 concurrent users
- Fix critical bugs
- Document API
```

### Week 2 (Day 8 → Day 14)

**Monday-Wednesday: Voice Integration**
```bash
# Day 8-10
- Add OpenAI Realtime API to Flutter
- Implement WebRTC client
- Test barge-in/barge-out
- Measure latency
```

**Thursday-Friday: Avatar Upgrade**
```bash
# Day 11-12
- Add gaze tracking to Unity
- Implement micro-saccades
- Update blendshape driver
- Profile frame rate
```

### Week 3-12: Continue according to phases above

---

## 📊 Success Metrics by Phase

| **Phase** | **Key Metric** | **Target** | **Current** |
|----------|---------------|-----------|------------|
| Phase 0 | Backend uptime | 99.9% | N/A |
| Phase 1 | Voice RTT p50 | <600ms | N/A |
| Phase 2 | Lessons complete | 90/90 | 1/90 |
| Phase 3 | IAP success rate | >99% | N/A |
| Phase 4 | GPT Store listing | Live | N/A |
| Phase 5 | Crash-free rate | >99.5% | N/A |
| Phase 6 | Beta D7 retention | >30% | N/A |
| Phase 7 | App Store approval | ✅ | N/A |

---

## ✅ Migration Checklist

### Before Starting Development
- [ ] Read complete execution plan
- [ ] Understand all 17 requirements
- [ ] Set up development environment
- [ ] Register developer accounts
- [ ] Get API keys (OpenAI, ElevenLabs, etc.)

### Week 1 Checklist
- [ ] Backend API responding
- [ ] Safety router working
- [ ] Schema migrated
- [ ] Deployed to staging

### Week 4 Checkpoint
- [ ] Voice conversation working
- [ ] Avatar at 60fps
- [ ] Can test on real device

### Week 8 Checkpoint
- [ ] IAP working in sandbox
- [ ] Privacy labels complete
- [ ] 60+ lessons created

### Week 12 - Launch Day!
- [ ] Apps submitted and approved
- [ ] Monitoring dashboards live
- [ ] Support channels ready
- [ ] 🚀 LAUNCH!

---

**Status**: 📋 Planning Complete - Ready to Execute  
**Next Action**: Start Week 1 backend setup  
**Timeline**: 12 weeks to launch  
**Last Updated**: October 29, 2025

**You have everything you need. Let's ship this! 🚀**














