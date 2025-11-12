# Getting Started with Curious Kellly Development
**Quick Start Guide for Team Members**

---

## 📋 Prerequisites Checklist

Before you begin development, ensure you have:

### Accounts & Access
- [ ] Apple Developer Program enrollment ($99/year) - [Sign up](https://developer.apple.com/programs/)
- [ ] Google Play Console registration ($25 one-time) - [Sign up](https://play.google.com/console/signup)
- [ ] OpenAI API account with credits - [Sign up](https://platform.openai.com/)
- [ ] ElevenLabs API key - [Sign up](https://elevenlabs.io/)
- [ ] Vector DB account (Pinecone or Qdrant) - [Pinecone](https://www.pinecone.io/) | [Qdrant](https://qdrant.tech/)
- [ ] Analytics account (Mixpanel or Amplitude) - [Mixpanel](https://mixpanel.com/) | [Amplitude](https://amplitude.com/)
- [ ] GitHub access to UI-TARS-desktop repository

### Development Tools
- [ ] Flutter SDK 3.x+ - [Install](https://docs.flutter.dev/get-started/install)
- [ ] Unity 2022.3 LTS - [Download](https://unity.com/download)
- [ ] Node.js 18+ or Python 3.10+ - [Node.js](https://nodejs.org/) | [Python](https://www.python.org/)
- [ ] Git
- [ ] VS Code or Android Studio
- [ ] Xcode (Mac only, for iOS development)
- [ ] Android SDK (for Android development)

### Hardware
- [ ] Mac (required for iOS builds)
- [ ] iPhone 12+ for testing (physical device recommended)
- [ ] Android device Pixel 6+ for testing

---

## 🚀 Quick Setup (30 Minutes)

### Step 1: Clone and Setup Workspace
```bash
# Clone the repository (if you haven't already)
cd UI-TARS-desktop

# Create the Curious Kellly directory structure
mkdir -p curious-kellly/{backend,mobile,mcp-server,apps-sdk-widget,content}
mkdir -p curious-kellly/content/{daily-topics,rag-corpus}
mkdir -p curious-kellly/content/daily-topics/2026-01

# Verify existing assets
ls -la lesson-player/        # Should see working web player
ls -la digital-kelly/        # Should see Flutter + Unity app
ls -la lessons/              # Should see leaves-change-color.json
```

### Step 2: Install Dependencies

**For Backend (Node.js example):**
```bash
cd curious-kellly/backend
npm init -y
npm install express cors dotenv openai pinecone-client body-parser ws
npm install --save-dev nodemon typescript @types/node @types/express
```

**For Mobile (Flutter):**
```bash
cd curious-kellly/mobile
flutter create .
flutter pub add provider http just_audio in_app_purchase uuid
flutter pub add flutter_unity_widget
cd ios && pod install && cd ..
```

**For MCP Server:**
```bash
cd curious-kellly/mcp-server
npm init -y
npm install @modelcontextprotocol/sdk express cors
```

### Step 3: Configure Environment Variables

**Backend `.env`:**
```bash
cd curious-kellly/backend
cat > .env << 'EOF'
# OpenAI
OPENAI_API_KEY=sk-proj-...
OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview-2024-10-01

# ElevenLabs (fallback)
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=wAdymQH5YucAkXwmrdL0

# Vector DB (choose one)
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east-1
# OR
QDRANT_URL=https://...
QDRANT_API_KEY=...

# Database
DATABASE_URL=postgresql://localhost:5432/curious_kellly

# Redis
REDIS_URL=redis://localhost:6379

# Environment
NODE_ENV=development
PORT=3000
EOF
```

**Mobile `.env`:**
```bash
cd curious-kellly/mobile
cat > .env << 'EOF'
API_BASE_URL=http://localhost:3000
ENABLE_ANALYTICS=false
EOF
```

### Step 4: Verify Existing Assets Work

**Test the Lesson Player:**
```bash
cd lesson-player
python -m http.server 8000
# Open http://localhost:8000 in browser
# Move age slider and verify content changes
```

**Test Digital Kelly:**
```bash
cd digital-kelly/apps/kelly_app_flutter
flutter pub get
flutter run
# Should launch the Flutter app with Unity embed
```

**Test Kelly Avatar Generation:**
```bash
cd ../..  # Back to root
python -m kelly_pack.cli --help
# Should show Kelly asset generator CLI
```

---

## 📂 Project Structure Overview

```
UI-TARS-desktop/
├── curious-kellly/              # 🆕 Production Curious Kellly app
│   ├── backend/                 # Backend orchestration service
│   ├── mobile/                  # Flutter production app
│   ├── mcp-server/              # GPT Store MCP integration
│   ├── apps-sdk-widget/         # ChatGPT widget component
│   └── content/                 # Lesson content (90 lessons)
│
├── lesson-player/               # ✅ Existing web dev tool (keep)
├── digital-kelly/               # ✅ Existing Flutter+Unity base (migrate from)
├── kelly_pack/                  # ✅ Avatar asset generation pipeline
├── kelly_audio2face/            # ✅ Audio2Face lip-sync integration
├── lessons/                     # ✅ Sample lesson JSON files
└── docs/                        # Documentation
```

### What Goes Where?

| **Component** | **Location** | **Purpose** |
|--------------|-------------|------------|
| Backend API | `curious-kellly/backend/` | Session management, safety, RAG, lesson planner |
| Production mobile app | `curious-kellly/mobile/` | iOS/Android apps with IAP |
| Development/testing | `digital-kelly/` | Use for rapid testing before prod migration |
| Web lesson player | `lesson-player/` | Dev tool for content authoring and testing |
| Avatar assets | `kelly_pack/` | Generate avatar images, sprites |
| Lip-sync | `kelly_audio2face/` | Generate blendshape animations |
| Lesson content | `curious-kellly/content/daily-topics/` | Daily Lesson calendar (30 launch topics → 365) |

---

## 🎯 Your First Task (Choose Your Role)

### **Backend Engineer**
**Goal**: Get backend responding with first API endpoint

1. Scaffold the Express/FastAPI backend
2. Add `/health` endpoint
3. Integrate OpenAI SDK (test with a simple chat completion)
4. Add `/session/start` endpoint that returns lesson JSON
5. Test with Postman/curl

**Expected Time**: 2-3 hours  
**Deliverable**: `curl http://localhost:3000/health` returns `{"status": "ok"}`

**Files to create**:
- `curious-kellly/backend/src/index.js` (or `main.py`)
- `curious-kellly/backend/src/api/sessions.js`
- `curious-kellly/backend/src/services/realtime.js`

### **Mobile Engineer**
**Goal**: Migrate digital-kelly to curious-kellly/mobile and add IAP

1. Copy `digital-kelly/` as base for `curious-kellly/mobile/`
2. Add `in_app_purchase` Flutter package
3. Configure StoreKit (iOS) and Play Billing (Android)
4. Create subscription products in sandbox
5. Test purchase flow

**Expected Time**: 3-4 hours  
**Deliverable**: Can test-purchase a subscription in sandbox

**Files to create**:
- `curious-kellly/mobile/lib/services/iap_service.dart`
- `curious-kellly/mobile/ios/Runner/Info.plist` (with SKProducts)
- `curious-kellly/mobile/android/app/src/main/AndroidManifest.xml` (with billing permission)

### **Content Creator**
**Goal**: Author first 10 Daily Lesson topics for the launch calendar (Jan 1–10)

1. Copy `lessons/leaves-change-color.json` as the baseline skeleton.
2. For each calendar day (2026-01-01 → 2026-01-10), draft a universal topic outline with 6 age personas (`kellyAge`, `kellyPersona`, voice pacing, expression cues).
3. Populate the `language` map with fully written English content and precomputed Spanish/French variants (no runtime generation).
4. Align teaching moments and expression cues across all personas; ensure timestamps map to the audio plan.
5. Run `node curious-kellly/content-tools/validate-lesson.js <file>` to confirm schema + quality guardrails.

**Expected Time**: 4-6 hours  
**Deliverable**: 10 validated Daily Lesson JSON files (60 persona variants) ready for audio rendering

**Files to create**:
- `curious-kellly/content/daily-topics/2026-01-01-leaves-change-color.json`
- `curious-kellly/content/daily-topics/2026-01-02-where-does-the-sun-go.json`
- ... (8 more)

### **AI/ML Engineer**
**Goal**: Set up RAG pipeline and test retrieval

1. Set up Pinecone or Qdrant vector database
2. Create embeddings for lesson content
3. Build retrieval API endpoint
4. Test with sample queries
5. Add citation system

**Expected Time**: 3-4 hours  
**Deliverable**: Query returns relevant lesson snippets

**Files to create**:
- `curious-kellly/backend/src/services/rag.js`
- `curious-kellly/backend/scripts/populate_vector_db.js`

---

## 🧪 Testing Your Setup

### Test Backend
```bash
cd curious-kellly/backend
npm run dev
curl http://localhost:3000/health
# Should return: {"status": "ok"}
```

### Test Mobile App
```bash
cd curious-kellly/mobile
flutter doctor  # Verify no issues
flutter run
# Should launch app on connected device/simulator
```

### Test Lesson Player (Already Working)
```bash
cd lesson-player
python -m http.server 8000
# Open browser to http://localhost:8000
# Move age slider - content should update
```

### Test Avatar Pipeline (Already Working)
```bash
python -m kelly_pack.cli build --help
# Should show usage
```

---

## 📖 Key Documents to Read

### Essential Reading (Do this first!)
1. **`CURIOUS_KELLLY_EXECUTION_PLAN.md`** - Complete roadmap (this directory)
2. **`Curious-Kellly_PRD.md`** - Product requirements (Downloads/)
3. **`Curious-Kellly_Technical_Blueprint.md`** - Architecture (Downloads/)
4. **`CK_Requirements-Matrix.csv`** - All 17 requirements (Downloads/)

### Reference Materials
5. **`lesson-player/README.md`** - How the lesson player works
6. **`lesson-player/lesson-dna-schema.json`** - Lesson structure
7. **`digital-kelly/README.md`** - Flutter + Unity architecture
8. **`README.md`** - Overall project overview

### Compliance & Legal
9. **`Curious-Kellly_GTM_Checklists.md`** - Store submission checklists
10. **`Curious-Kellly_Tools_and_Fees.csv`** - Costs and services

---

## 🤝 Team Communication

### Daily Standup Format
- **What I did yesterday**: (specific tasks)
- **What I'm doing today**: (specific tasks)
- **Blockers**: (if any)
- **Demo**: (if you have something to show)

### Sprint Planning (Every 2 Weeks)
- Review completed requirements from `CK_Requirements-Matrix.csv`
- Demo working features
- Plan next sprint tasks
- Update `CK_Launch-Checklist.csv` status

### Code Review Guidelines
- PR must pass all tests
- At least 1 approval required
- Update relevant documentation
- Add tests for new features

---

## ❓ Common Questions

### Q: Should I work in `digital-kelly/` or `curious-kellly/mobile/`?
**A**: Start in `digital-kelly/` for quick prototyping and testing. Once features work, migrate to `curious-kellly/mobile/` for production.

### Q: Which backend framework should I use?
**A**: Choose what you're comfortable with:
- **Node.js + Express**: Best for quick iteration, JavaScript end-to-end
- **Python + FastAPI**: Best if you need ML/AI libraries, async support

### Q: Do I need a Mac for iOS development?
**A**: Yes, unfortunately. iOS builds require Xcode which only runs on macOS. Android can be developed on any OS.

### Q: How do I test IAP without spending money?
**A**: Both Apple and Google have sandbox environments:
- **iOS**: Use TestFlight with sandbox accounts
- **Android**: Use test purchases in Play Console

### Q: Where do I get the Kelly avatar 3D model?
**A**: It's created in the `iLearnStudio/` workflow using Character Creator 5. See `docs/guides/CC5_HEADSHOT_BEGINNER_GUIDE.md`.

### Q: How do I generate audio for lessons?
**A**: Use `lesson-player/generate_audio.py` which calls ElevenLabs API. See `lesson-player/README.md`.

---

## 🆘 Getting Help

### Stuck on setup?
1. Check the file you're working on has a README
2. Look in `docs/guides/` for setup guides
3. Search existing issues in GitHub
4. Ask in team chat

### Found a bug?
1. Check if it's already reported
2. Create minimal reproduction steps
3. File an issue with details
4. Tag with appropriate label

### Need clarification on requirements?
1. Check `CK_Requirements-Matrix.csv` for detailed acceptance criteria
2. Review the PRD (`Curious-Kellly_PRD.md`)
3. Ask PM or team lead

---

## ✅ Setup Complete Checklist

Before moving to your first sprint task, verify:

- [ ] All prerequisite accounts created
- [ ] Development tools installed and working
- [ ] Repository cloned and directory structure created
- [ ] Environment variables configured
- [ ] Dependencies installed for your area (backend/mobile/etc.)
- [ ] Can run existing lesson player and see it work
- [ ] Read essential documents (PRD, Technical Blueprint)
- [ ] Completed "Your First Task" for your role
- [ ] Tested your setup successfully

---

**Estimated Setup Time**: 2-4 hours  
**Status**: Ready to start Sprint 0  
**Questions?**: Check docs/ or ask team

**Let's ship Curious Kellly! 🚀**

