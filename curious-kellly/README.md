# Curious Kellly - Production Application
**Multimodal AI Teacher-Avatar for iOS, Android, GPT Store, and Claude Artifacts**

---

## 🎯 Project Overview

Curious Kellly is a daily, avatar-led learning companion delivering personalized 5-15 minute sessions with:
- Real-time voice conversation (OpenAI Realtime API)
- 60fps avatar with lip-sync, gaze tracking, and micro-expressions
- Age-adaptive content for ages 2-102
- Daily Lesson calendar: 30 universal launch topics (one per day) with Kelly aging across six personas (roadmap to 365 topics)
- Precomputed multilingual content (EN live, ES/FR cached per topic)
- Multi-platform: iOS, Android, GPT Store, Claude Artifacts

**Launch Target**: 12 weeks from today (84 days)

---

## 📂 Directory Structure

```
curious-kellly/
├── backend/                 # Orchestration service (Node.js/Python)
│   ├── src/
│   │   ├── api/            # Express/FastAPI routes
│   │   ├── services/       # Business logic (realtime, safety, RAG, planner)
│   │   ├── models/         # Data models
│   │   └── utils/          # Helpers
│   ├── tests/
│   ├── config/
│   └── README.md
│
├── mobile/                  # Flutter production app
│   ├── lib/
│   │   ├── screens/        # UI screens
│   │   ├── services/       # API clients, IAP, analytics
│   │   ├── widgets/        # Reusable components
│   │   └── main.dart
│   ├── ios/                # iOS specific (StoreKit)
│   ├── android/            # Android specific (Play Billing)
│   ├── unity/              # Unity avatar engine
│   └── README.md
│
├── mcp-server/              # GPT Store MCP integration
│   ├── index.ts
│   ├── tools/              # MCP tools
│   └── README.md
│
├── apps-sdk-widget/         # ChatGPT Apps SDK widget
│   ├── src/
│   │   └── LessonBoard.tsx
│   └── README.md
│
├── content/                 # Daily Lesson content
│   ├── daily-topics/
│   │   ├── 2026-01-01-leaves-change-color.json
│   │   ├── 2026-01-02-where-does-the-sun-go.json
│   │   └── ...              # One JSON per calendar day
│   └── rag-corpus/         # Curated content for RAG
│
└── docs/
    ├── ARCHITECTURE.md
    ├── API.md
    ├── DEPLOYMENT.md
    └── TESTING.md
```

---

## 🚀 Quick Start

### Prerequisites
- Flutter 3.x+
- Unity 2022.3 LTS
- Node.js 18+ or Python 3.10+
- OpenAI API account
- ElevenLabs API key
- Apple Developer account ($99)
- Google Play Console account ($25)

### Setup (10 minutes)

```bash
# 1. Navigate to project
cd UI-TARS-desktop/curious-kellly

# 2. Setup backend
cd backend
npm install  # or pip install -r requirements.txt
cp .env.example .env  # Add your API keys
npm run dev

# 3. Setup mobile (new terminal)
cd ../mobile
flutter pub get
flutter run

# 4. Verify lesson player works
cd ../../lesson-player
python -m http.server 8000
# Open http://localhost:8000
```

See **[GETTING_STARTED_CK.md](../GETTING_STARTED_CK.md)** for detailed setup.

---

## 🏗️ Architecture

### High-Level Flow
```
[Mobile App] ←WebRTC→ [OpenAI Realtime API]
      ↓                        ↓
[Unity Avatar]           [Backend Service]
   - 60fps                     ↓
   - Visemes            ┌──────┴──────┐
   - Gaze          [Safety] [RAG] [Planner]
                        ↓       ↓       ↓
                   [Moderate] [Vector] [Lessons]
                                DB
```

### Key Components

1. **Backend (`backend/`)**
   - Session management
   - OpenAI Realtime API client
   - Safety router (moderation)
   - RAG (vector search)
   - Lesson planner

2. **Mobile (`mobile/`)**
   - Flutter UI shell
   - Unity 3D avatar engine
   - Apple IAP / Google Play Billing
   - Analytics integration
   - Offline support

3. **MCP Server (`mcp-server/`)**
   - GPT Store integration
   - HTTP streamable endpoint
   - 3 tools: get_daily_lesson, start_session, submit_answer

4. **Content (`content/`)**
   - Daily Lesson calendar JSON (30 launch topics → 365) with six age personas per topic
   - PhaseDNA structure with persona, voice, expression metadata
   - Precomputed EN/ES/FR payloads + scripts
   - Teaching moments and interaction flow

---

## 📋 Requirements Tracking

See **[CK_Requirements-Matrix.csv](../CK_Requirements-Matrix.csv)** for all 17 requirements.

### Critical Requirements (Must Have)
- **R1**: Realtime voice loop (RTT ≤600ms)
- **R2**: Avatar lip-sync & gaze (60fps, <5% error)
- **R3**: Lesson planner (JSON schema validated)
- **R6**: Safety router (precision ≥0.98)
- **R7**: Analytics (D1/D7/D30 dashboards)
- **R8**: Billing (IAP working)

### Current Status
- 🟢 **Complete (0/17)**: None yet
- 🟡 **In Progress (1/17)**: R3 (Lesson planner - base schema exists)
- 🔴 **Not Started (16/17)**: Most components
- ⚫ **Blocked (0/17)**: None

---

## 🧪 Testing

### Run Backend Tests
```bash
cd backend
npm test  # or pytest
```

### Run Mobile Tests
```bash
cd mobile
flutter test
```

### Device Testing
Target devices:
- **iOS**: iPhone 12, 13, 14, 15
- **Android**: Pixel 6, 7, 8

### Performance Targets
- Voice RTT: p50 ≤600ms, p95 ≤900ms
- Lip-sync error: <5%
- Frame rate: 60fps on iPhone 12/Pixel 6
- Crash-free: ≥99.7%

---

## 📦 Deployment

### Backend
```bash
# Deploy to Render/Railway
npm run build
npm run deploy
```

### Mobile Apps
```bash
# iOS (requires Mac)
cd mobile/ios
bundle exec fastlane release

# Android
cd mobile/android
./gradlew bundleRelease
```

### GPT Store
```bash
cd mcp-server
npm run deploy
# Then publish GPT with Builder Profile
```

See **[docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md)** for details.

---

## 🎓 Learning Resources

### Essential Reading
1. **[CURIOUS_KELLLY_EXECUTION_PLAN.md](../CURIOUS_KELLLY_EXECUTION_PLAN.md)** - Complete roadmap
2. **[GETTING_STARTED_CK.md](../GETTING_STARTED_CK.md)** - Setup guide
3. **[Curious-Kellly_PRD.md](../Curious-Kellly_PRD.md)** - Product requirements
4. **[Curious-Kellly_Technical_Blueprint.md](../Curious-Kellly_Technical_Blueprint.md)** - Architecture

### API Documentation
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
- [Flutter Unity Widget](https://pub.dev/packages/flutter_unity_widget)
- [Apple IAP](https://developer.apple.com/in-app-purchase/)
- [Google Play Billing](https://developer.android.com/google/play/billing)
- [MCP Protocol](https://spec.modelcontextprotocol.io/)

---

## 📊 Success Metrics (90-Day Post-Launch)

### Product KPIs
- D1 retention ≥ 45%
- D30 retention ≥ 20%
- Session length ≥ 8 minutes
- Completion rate ≥ 70%
- CSAT ≥ 4.6/5
- NPS ≥ +40

### Technical KPIs
- Voice RTT p50 ≤ 600ms
- Lip-sync error < 5%
- 60fps on target devices
- Crash-free ≥ 99.5%
- Safety precision ≥ 0.98

### Business KPIs
- 10,000+ downloads
- 1,000+ paid subscribers
- Trial → paid ≥ 15%
- Refund rate < 5%

---

## 🚨 Current Sprint

### Sprint 0: Foundation (Week 1-2)
**Goal**: Backend responding + Safety router working

**This Week's Tasks**:
- [ ] Scaffold backend service
- [ ] Integrate OpenAI Realtime API
- [ ] Build safety router with moderation
- [ ] Set up vector database
- [ ] Deploy to staging

**Next Week**:
- Sprint 1: Voice & Avatar integration

See **[CK_Launch-Checklist.csv](../CK_Launch-Checklist.csv)** for all 62 launch tasks.

---

## 🤝 Contributing

### Code Standards
- **TypeScript/Dart**: Strict type checking
- **Testing**: 80%+ coverage on critical paths
- **Documentation**: Every public API documented
- **Linting**: Run before commit

### PR Process
1. Create feature branch
2. Write tests
3. Update docs
4. Submit PR with description
5. Get 1+ approvals
6. Merge to main

---

## 📄 License

Proprietary - UI-TARS Team

---

## 📞 Support

### Internal Team
- **PM**: Product questions, priorities
- **Backend Lead**: API, safety, RAG
- **Mobile Lead**: Flutter, Unity, IAP
- **Content Lead**: Lesson creation

### External Resources
- OpenAI API support
- ElevenLabs support
- Apple Developer support
- Google Play support

---

**Status**: 🚧 In Development (Sprint 0)  
**Launch Target**: 12 weeks from today  
**Last Updated**: October 29, 2025

**Let's build something insanely great.** 🚀

