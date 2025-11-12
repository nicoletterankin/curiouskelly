# Curious Kellly - Complete Execution Plan
**Merged from: UI-TARS workspace + CK Production Requirements**  
**Date:** October 29, 2025

---

## 🎯 Executive Summary

**Mission**: Transform the working Kelly lesson player prototype into a production-ready, multi-platform learning companion that launches on iOS, Android, GPT Store, and Claude Artifacts.

**Current State**: 
- ✅ Working web lesson player with age adaptation (2-102)
- ✅ ElevenLabs voice synthesis integration
- ✅ Audio2Face lip-sync pipeline (NVIDIA)
- ✅ Unity + Flutter cross-platform avatar rendering
- ✅ Kelly asset generation pipeline (8K, matting, compositing)
- ✅ PhaseDNA lesson schema with teaching moments
- ✅ 6 audio files generated for "Leaves Change Color" lesson

**Goal State**: 
- ✅ Real-time voice with OpenAI Realtime API (<600ms RTT)
- ✅ 60fps avatar with lip-sync, gaze tracking, micro-expressions
- ✅ Daily Lesson pipeline: 30 launch topics (universal calendar) with Kelly aging across six personas, scaling to 365 topics post-launch
- ✅ Mobile apps on App Store & Play Store with IAP
- ✅ GPT Store listing with MCP server integration
- ✅ Safety router with moderation (precision ≥0.98)
- ✅ D1≥45%, D30≥20% retention metrics

---

## 📊 Gap Analysis

### Assets We Have → What CK Needs

| **Current Asset** | **Status** | **CK Requirement** | **Gap** | **Priority** |
|------------------|-----------|-------------------|---------|-------------|
| Web lesson player | ✅ Working | Mobile apps (iOS/Android) | Flutter app exists but needs IAP integration | **P0** |
| ElevenLabs TTS | ✅ Working | OpenAI Realtime API (speech-in/speech-out) | Need to add realtime voice path | **P0** |
| Audio2Face lip-sync | ✅ Working | 60fps viseme sync | Need to map A2F → CK blendshapes | **P1** |
| Unity avatar | ✅ Basic | Avatar with gaze, expressions, blinks | Need gaze tracking + micro-expressions | **P1** |
| 1 lesson (Leaves) | ✅ Complete | Daily Lesson calendar (30 launch topics) | Need 29 more universal topics + multilingual variants | **P0** |
| No safety layer | ❌ Missing | Input/output moderation + safety router | Build from scratch | **P0** |
| No billing | ❌ Missing | Apple IAP + Play Billing | Integrate StoreKit/Play Billing | **P0** |
| No backend | ❌ Missing | Session state, lesson planner, RAG, analytics | Build orchestration service | **P0** |
| Local only | ❌ Missing | Multi-platform deployment | Add Cloudflare/AWS hosting | **P1** |
| No GPT integration | ❌ Missing | MCP server + Apps SDK | Build MCP endpoint + widget | **P1** |

---

## 🏗️ Architecture Alignment

### Current Architecture (UI-TARS)
```
[Web Browser] → [Lesson Player HTML/JS] → [JSON Lessons]
                          ↓
                    [ElevenLabs API] → [Audio MP3]
                          ↓
                    [Audio2Face] → [Blendshapes]
                          ↓
[Flutter App] ← → [Unity Engine] → [Kelly Avatar 3D]
```

### Target Architecture (Curious Kellly)
```
[Mobile App iOS/Android] ← → [OpenAI Realtime API (WebRTC)]
         ↓                              ↓
   [Unity Avatar]                [Orchestration Service]
    - 60fps visemes                    ↓
    - Gaze tracking              ┌─────┴─────┐
    - Expressions           [Safety]  [RAG]  [Planner]
         ↓                       ↓       ↓       ↓
   [Audio Sync]            [Moderation] [Vector DB] [Lesson JSON]
                                 ↓
                          [Analytics Pipeline]
                                 ↓
                      [User/Subscriptions Service]
                            ↓         ↓
                    [Apple IAP] [Play Billing]
```

### Integration Strategy
1. **Keep the lesson player** as development/testing tool
2. **Migrate Unity avatar** to support realtime viseme streams
3. **Add backend layer** for orchestration, safety, RAG
4. **Wrap Flutter app** with billing + auth
5. **Deploy** to stores and cloud

---

## 📋 Execution Roadmap (12 Weeks to Launch)

### **SPRINT 0: Foundation (Week 1-2)** - *Setup & Architecture*

#### **0.1 Backend Infrastructure** (5 days)
- [ ] Set up Node.js/Python backend service (choose one)
- [ ] Configure OpenAI Realtime API client
- [ ] Set up vector database (Pinecone/Qdrant) for RAG
- [ ] Create session state manager
- [ ] Deploy to staging environment (Render/Railway)

**Deliverable**: Backend API responding to health checks

#### **0.2 Safety Router** (3 days)
- [ ] Integrate OpenAI Moderation API
- [ ] Create custom content filter rules
- [ ] Build safe-completion rewrite logic
- [ ] Create test suite with policy violations
- [ ] Target: Precision ≥0.98, Recall ≥0.95

**Deliverable**: Safety endpoint blocks unsafe content

#### **0.3 Lesson Planner Migration** (2 days)
- [ ] Migrate PhaseDNA schema to backend
- [ ] Create lesson JSON loader
- [ ] Build daily lesson selector
- [ ] Validate against schema
- [ ] Create Daily Lesson calendar structure (30-day launch window scaffold)

**Deliverable**: API endpoint returns daily lesson JSON

---

### **SPRINT 1: Voice & Avatar (Week 3-4)** - *LC-010, LC-011, R1, R2*

#### **1.1 Realtime Voice Integration** (5 days)
- [ ] Add OpenAI Realtime API WebRTC client to Flutter
- [ ] Implement ephemeral key fetch from backend
- [ ] Add barge-in/barge-out support
- [ ] Test median RTT <600ms
- [ ] Add fallback to ElevenLabs if realtime unavailable

**Deliverable**: Voice conversation working end-to-end

#### **1.2 Avatar Upgrade to 60fps** (5 days)
- [ ] Update Unity blendshape driver for realtime visemes
- [ ] Map OpenAI viseme stream → A2F blendshapes
- [ ] Add gaze tracking (screen-space targets)
- [ ] Implement micro-saccades (2-4/s)
- [ ] Add blink system (8-12/min)
- [ ] Profile and optimize to 60fps on iPhone 12/Pixel 6

**Deliverable**: Avatar syncs with realtime speech at 60fps

#### **1.3 Audio Sync Calibration** (2 days)
- [ ] Add delay calibration slider (±60ms)
- [ ] Test on 5 devices (2 iOS, 3 Android)
- [ ] Measure lip-sync error <5%
- [ ] Add frame metrics logging

**Deliverable**: Frame-accurate sync validated

---

### **SPRINT 2: Content Creation (Week 5-6)** - *LC-001, LC-012, R3*

#### **2.1 Daily Lesson Proof Set** (7 days)
- [ ] Author **10 universal daily topics** (proof-of-concept set) covering nature/science/wisdom beats
  - Map each topic to the daily calendar and teaching moment cues
- [ ] Populate **6 age personas per topic** with `kellyAge`, `kellyPersona`, voice pacing, and expression triggers
- [ ] Precompute multilingual payloads (EN live, ES/FR cached) for welcome/main/wisdom sections
- [ ] Generate ElevenLabs audio for all 60 variants (10 topics × 6 ages) and store VO metadata
- [ ] Validate with updated schema + automated validator, then spot-check in lesson player across the age slider

**Deliverable**: 10 topics (60 variants) production-ready with multilingual text + audio

#### **2.2 RAG Content Population** (3 days)
- [ ] Create curated content corpus
- [ ] Generate embeddings for lessons
- [ ] Populate vector DB
- [ ] Test retrieval quality
- [ ] Add citation system

**Deliverable**: RAG retrieves relevant content

---

### **SPRINT 3: Mobile Apps (Week 7-8)** - *LC-017, R8, R11, R12*

#### **3.1 Apple IAP Integration** (3 days)
- [ ] Enroll in Apple Developer Program ($99)
- [ ] Create IAP products (monthly, annual, family)
- [ ] Integrate StoreKit in Flutter
- [ ] Test in sandbox
- [ ] Add receipt validation server-side
- [ ] Target: Unlock <2s after purchase

**Deliverable**: IAP working in TestFlight

#### **3.2 Google Play Billing** (3 days)
- [ ] Register Play Console ($25)
- [ ] Target API 35 (required by Aug 31, 2025)
- [ ] Integrate Play Billing Library v7+
- [ ] Create subscription products
- [ ] Test in internal track

**Deliverable**: Billing working in Play Internal Testing

#### **3.3 Privacy Compliance** (4 days)
- [ ] **iOS**: Complete App Privacy labels
  - Data collection inventory
  - Privacy manifest files
  - ATT prompt if needed
- [ ] **Android**: Complete Data Safety form
  - Data types collected
  - Account deletion URL
- [ ] Add age gate (13+ default)
- [ ] Create parent consent flow for <13
- [ ] Add privacy dashboard (view/delete/export)

**Deliverable**: Apps pass privacy precheck

---

### **SPRINT 4: GPT Store & Claude (Week 9)** - *LC-020, LC-021, R9, R10*

#### **4.1 MCP Server Implementation** (3 days)
- [ ] Build HTTP streamable `/mcp` endpoint
- [ ] Register 3 tools:
  - `get_daily_lesson(topic, level)` → returns lesson JSON
  - `start_session(age)` → initializes lesson state
  - `submit_answer(choice)` → progresses lesson
- [ ] Make tools introspectable
- [ ] Test with MCP Inspector

**Deliverable**: MCP server working locally

#### **4.2 Apps SDK Widget** (2 days)
- [ ] Create "Lesson Board" React component
- [ ] Wire `callTool()` to MCP endpoints
- [ ] Add refresh on state change
- [ ] Test in ChatGPT Dev Mode

**Deliverable**: Widget renders in ChatGPT

#### **4.3 GPT Store Submission** (2 days)
- [ ] Verify OpenAI Builder Profile
- [ ] Publish GPT with "Everyone" visibility
- [ ] Write store description
- [ ] Add demo conversation script
- [ ] Submit for review

**Deliverable**: GPT live in store

---

### **SPRINT 5: Analytics & Testing (Week 10)** - *LC-016, R7, R15*

#### **5.1 Analytics Pipeline** (3 days)
- [ ] Integrate Mixpanel or Amplitude
- [ ] Define event schema:
  - `session_start`, `session_complete`
  - `teaching_moment_viewed`
  - `interaction_answered`
  - `age_changed`
  - `lesson_completed`
- [ ] Create dashboards:
  - D1/D7/D30 retention
  - Session length p50/p95
  - Latency metrics
  - Moderation triggers

**Deliverable**: Dashboards tracking KPIs

#### **5.2 Security Review** (2 days)
- [ ] Map PII flows
- [ ] Review API scopes
- [ ] Check storage encryption
- [ ] Scan for vulnerabilities
- [ ] Close all criticals

**Deliverable**: Security sign-off

#### **5.3 Device Matrix Testing** (2 days)
- [ ] Test on iPhone 12, 13, 14, 15 (4 devices)
- [ ] Test on Pixel 6, 7, 8 (3 devices)
- [ ] Test on iPad/tablet
- [ ] Log crashes, latency, lip-sync errors
- [ ] Target: Crash-free ≥99.7%

**Deliverable**: Test report with metrics

---

### **SPRINT 6: Beta & Polish (Week 11)** - *LC-030, LC-042*

#### **6.1 Beta Distribution** (2 days)
- [ ] TestFlight: Invite 300 users
- [ ] Play Internal Track: Invite 300 users
- [ ] Collect feedback
- [ ] Monitor crash reports
- [ ] Track retention D1/D7

**Deliverable**: Beta cohort active

#### **6.2 Avatar Polish** (3 days)
- [ ] Add micro-expressions from teaching moment cues
- [ ] Tune gaze saccade frequency
- [ ] Adjust blink timing for naturalness
- [ ] Record 90fps sample for review
- [ ] Target: CSAT ≥4.6 in beta

**Deliverable**: Avatar feels lifelike

#### **6.3 UI/UX Refinement** (2 days)
- [ ] Add loading states
- [ ] Improve error messages
- [ ] Polish animations
- [ ] Test accessibility (captions, speech rate)
- [ ] Add dark mode support

**Deliverable**: UI polish complete

---

### **SPRINT 7: Store Submission & Launch (Week 12)** - *LC-050, LC-051*

#### **7.1 Store Assets** (2 days)
- [ ] **Screenshots**: 6.7" and 6.1" for iOS
- [ ] **Preview video**: 15-30s with Kelly speaking
- [ ] **Store copy**: Short/full descriptions
- [ ] **Localization**: EN (ES/FR optional for v1)
- [ ] **Age rating**: Complete questionnaires
- [ ] **Review notes**: Explain AI/mic/camera usage

**Deliverable**: All store assets ready

#### **7.2 App Store Submission** (1 day)
- [ ] Submit to App Store Connect
- [ ] Provide review notes
- [ ] Monitor review status
- [ ] Respond to questions within 24h

**Deliverable**: App in review

#### **7.3 Google Play Submission** (1 day)
- [ ] Submit to Play Console
- [ ] Complete IARC rating
- [ ] Verify target API 35
- [ ] Monitor review status

**Deliverable**: App in review

#### **7.4 Launch Preparation** (2 days)
- [ ] Set up customer support (email/chat)
- [ ] Prepare launch announcements
- [ ] Set up monitoring dashboards
- [ ] Create incident response runbook
- [ ] Plan rollback strategy

**Deliverable**: Ready for launch

#### **7.5 LAUNCH! 🚀** (Day 84)
- [ ] Apps go live on App Store & Play Store
- [ ] GPT Store listing activated
- [ ] Monitor metrics hour-by-hour
- [ ] Track D1 retention
- [ ] Fix critical issues <15min

**Deliverable**: Curious Kellly is LIVE

---

## 📁 File & Directory Structure

### New Directories to Create
```
UI-TARS-desktop/
├── curious-kellly/                    # NEW: Production app
│   ├── backend/                       # NEW: Orchestration service
│   │   ├── src/
│   │   │   ├── api/                   # Express/FastAPI routes
│   │   │   ├── services/
│   │   │   │   ├── realtime.js        # OpenAI Realtime API client
│   │   │   │   ├── safety.js          # Moderation + safe-completion
│   │   │   │   ├── planner.js         # Lesson planner
│   │   │   │   ├── rag.js             # Vector DB retrieval
│   │   │   │   └── subscriptions.js   # IAP webhook handlers
│   │   │   ├── models/                # Data models
│   │   │   └── utils/                 # Helpers
│   │   ├── config/
│   │   │   └── env.example
│   │   ├── package.json
│   │   └── README.md
│   │
│   ├── mobile/                        # NEW: Flutter production app
│   │   ├── lib/
│   │   │   ├── screens/
│   │   │   │   ├── onboarding.dart
│   │   │   │   ├── lesson_session.dart
│   │   │   │   ├── subscription.dart
│   │   │   │   └── settings.dart
│   │   │   ├── services/
│   │   │   │   ├── realtime_client.dart
│   │   │   │   ├── iap_service.dart
│   │   │   │   └── analytics.dart
│   │   │   ├── widgets/
│   │   │   │   ├── kelly_avatar_view.dart
│   │   │   │   └── lesson_controls.dart
│   │   │   └── main.dart
│   │   ├── android/                   # Play Billing config
│   │   ├── ios/                       # StoreKit config
│   │   ├── pubspec.yaml
│   │   └── README.md
│   │
│   ├── mcp-server/                    # NEW: GPT Store integration
│   │   ├── index.ts
│   │   ├── tools/
│   │   │   ├── get_daily_lesson.ts
│   │   │   ├── start_session.ts
│   │   │   └── submit_answer.ts
│   │   ├── package.json
│   │   └── README.md
│   │
│   ├── apps-sdk-widget/               # NEW: ChatGPT widget
│   │   ├── src/
│   │   │   └── LessonBoard.tsx
│   │   ├── package.json
│   │   └── README.md
│   │
│   ├── content/                       # NEW: Daily Lesson content
│   │   ├── daily-topics/
│   │   │   ├── 2026-01-01-leaves-change-color/
│   │   │   ├── 2026-01-02-where-does-the-sun-go/
│   │   │   └── ...                    # One folder per calendar day
│   │   └── rag-corpus/                # Curated content for RAG
│   │
│   └── docs/
│       ├── ARCHITECTURE.md
│       ├── API.md
│       ├── DEPLOYMENT.md
│       └── TESTING.md
│
├── digital-kelly/                     # KEEP: Development/testing app
├── lesson-player/                     # KEEP: Web dev tool
├── kelly_pack/                        # KEEP: Asset pipeline
├── kelly_audio2face/                  # KEEP: Lip-sync pipeline
└── lessons/                           # KEEP: Legacy lessons
```

---

## 🔧 Technology Stack

### Frontend (Mobile)
- **Framework**: Flutter 3.x
- **3D Engine**: Unity 2022.3 LTS (embedded)
- **State**: Provider or Riverpod
- **Audio**: Just_audio plugin
- **Billing**: in_app_purchase + purchases_flutter
- **Analytics**: Mixpanel or Amplitude SDK

### Backend
- **Language**: Node.js (Express) or Python (FastAPI)
- **Voice**: OpenAI Realtime API (WebRTC)
- **LLM**: GPT-4 Turbo (text) + Realtime (voice)
- **Vector DB**: Pinecone or Qdrant
- **Database**: PostgreSQL (user data, sessions)
- **Cache**: Redis (session state)
- **Storage**: S3/CloudFlare R2 (audio/video assets)

### Infrastructure
- **Hosting**: Render, Railway, or AWS ECS
- **CDN**: CloudFlare
- **Monitoring**: Sentry (errors) + Mixpanel (product)
- **CI/CD**: GitHub Actions
- **Secrets**: Doppler or AWS Secrets Manager

### Development Tools
- **Design**: Figma
- **API Testing**: Postman
- **MCP Testing**: MCP Inspector
- **Device Testing**: BrowserStack or AWS Device Farm

---

## 🎓 Success Criteria (90-Day Post-Launch)

### Product Metrics
- [ ] D1 retention ≥ 45%
- [ ] D30 retention ≥ 20%
- [ ] Average session length ≥ 8 minutes
- [ ] Session completion rate ≥ 70%
- [ ] CSAT ≥ 4.6/5
- [ ] NPS ≥ +40

### Technical Metrics
- [ ] Median voice RTT ≤ 600ms (p95 ≤ 900ms)
- [ ] Lip-sync error < 5%
- [ ] 60fps on iPhone 12/Pixel 6
- [ ] Crash-free sessions ≥ 99.5%
- [ ] Safety precision ≥ 0.98, recall ≥ 0.95

### Business Metrics
- [ ] 10,000+ app downloads
- [ ] 1,000+ paying subscribers
- [ ] Trial → paid conversion ≥ 15%
- [ ] Refund rate < 5%
- [ ] CAC payback < 6 months

---

## 🚨 Risk Register & Mitigations

| **Risk** | **Impact** | **Probability** | **Mitigation** |
|---------|-----------|----------------|---------------|
| **OpenAI Realtime API downtime** | High | Medium | Keep ElevenLabs fallback path |
| **App Store rejection** | High | Low | Follow IAP guidelines strictly; pre-flight checklist |
| **Latency >1s on LTE** | High | Medium | Regional endpoints; edge caching; optimize payload |
| **Safety bypass (hallucinations)** | Critical | Medium | Multi-layer moderation; citation system; critic pass |
| **Poor retention** | High | Medium | A/B test onboarding; daily nudges; streaks |
| **Unity embedding crashes** | Medium | Low | Extensive device testing; graceful fallback |
| **Content creation bottleneck** | Medium | High | Use AI-assisted authoring; hire writers |
| **API cost overrun** | Medium | Medium | Cache aggressively; use cheaper models for non-voice |

---

## 📞 Next Actions (RIGHT NOW)

### Week 1 Sprint Planning
1. **Today (Day 1)**:
   - [ ] Review this plan with team
   - [ ] Set up project management board (Linear/Jira)
   - [ ] Create GitHub repo for `curious-kellly/`
   - [ ] Provision staging environment
   - [ ] Register Apple Developer + Google Play accounts

2. **Day 2-3**:
   - [ ] Scaffold backend service
   - [ ] Integrate OpenAI Realtime API (test with curl)
   - [ ] Set up vector database
   - [ ] Create first API endpoint: `POST /session/start`

3. **Day 4-5**:
   - [ ] Build safety router
   - [ ] Run moderation test suite
   - [ ] Migrate PhaseDNA schema to backend
   - [ ] Deploy to staging

4. **Week 1 Deliverable**: 
   - Backend API responding
   - Safety router blocking bad content
   - First lesson loading via API

---

## 📚 Resources & References

### Documentation
- [OpenAI Realtime API Docs](https://platform.openai.com/docs/guides/realtime)
- [Apple App Store Guidelines](https://developer.apple.com/app-store/review/guidelines/)
- [Google Play Policy](https://play.google.com/about/developer-content-policy/)
- [MCP Protocol Spec](https://spec.modelcontextprotocol.io/)

### Internal Docs
- `Curious-Kellly_PRD.md` - Product requirements
- `Curious-Kellly_Technical_Blueprint.md` - Architecture
- `CK_Requirements-Matrix.csv` - All 17 requirements
- `CK_Launch-Checklist.csv` - 62 launch tasks

### Existing Assets
- `lesson-player/` - Working web prototype
- `digital-kelly/` - Unity + Flutter base
- `lessons/leaves-change-color.json` - Complete lesson DNA
- `kelly_pack/` - Avatar asset generation

---

## ✅ Approval & Sign-Off

**This plan is ready for execution when:**
- [ ] Team has reviewed and agreed
- [ ] Budget approved for tools/services (~$500-1000/mo)
- [ ] Developer accounts registered (Apple $99, Google $25)
- [ ] OpenAI API credits funded
- [ ] Sprint 0 tasks assigned

---

**Status**: 📝 DRAFT - Ready for Review  
**Last Updated**: October 29, 2025  
**Estimated Timeline**: 12 weeks (84 days)  
**Estimated Cost**: $1,200-2,500 (excl. labor)  
**Team Size**: 3-5 people (1 backend, 1 mobile, 1 content, 1 PM)

---

*Let's build something insanely great.* 🚀

