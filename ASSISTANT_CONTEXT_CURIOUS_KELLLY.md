# Assistant Context - Curious Kellly Project
**For AI Assistant: Complete project understanding and guidelines**

---

## ✅ **CRITICAL: THE CORRECT PRODUCT VISION**

### **"The Daily Lesson" Model**
- **365 universal daily topics** (one per day of year)
- **Everyone gets the same topic** each day (community experience)
- **Kelly ages with the learner's age slider**:
  - Age 2 slider = 2-year-old Kelly (toddler voice, simple words)
  - Age 35 slider = 35-year-old Kelly (adult voice, professional tone)
  - Age 102 slider = 102-year-old Kelly (elder voice, wisdom)
- **No course tracks, no learning paths, no choice**
- **Daily ritual**: "What's today's topic?"

### **❌ NEVER Say These (Wrong Concepts)**
- "3 lesson tracks"
- "90 lessons"
- "Choose your learning path"
- "Spanish A1, Study Skills, Career Storytelling"
- "Age adaptation is just content complexity"

### **✅ ALWAYS Say These (Correct Concepts)**
- "365 universal daily topics"
- "The Daily Lesson"
- "Kelly ages with you"
- "Everyone on the same topic today"
- "Launch with 30 topics, scale to 365"

---

## 📊 **Current Project State**

### **What Exists (Working Assets)**
```
✅ lesson-player/                 → Web prototype with age adaptation
✅ digital-kelly/                 → Flutter + Unity base app
✅ kelly_pack/                    → Avatar asset generation (8K)
✅ kelly_audio2face/              → NVIDIA lip-sync pipeline
✅ lessons/leaves-change-color.json → 1 complete universal topic
✅ 6 audio files                  → Generated with ElevenLabs
✅ Age slider (2-102)             → Works in lesson player
✅ Teaching moments system        → Implemented
✅ Interactive choices            → Working
```

### **What Needs to Be Built**
```
🚧 curious-kellly/backend/        → Orchestration service
🚧 curious-kellly/mobile/         → Production Flutter app
🚧 curious-kellly/mcp-server/     → GPT Store integration
🚧 29 more universal topics       → For launch (30 total)
🚧 6 Kelly age variants           → 3D models for ages 3,9,15,27,48,82
🚧 Safety router                  → Moderation (≥98% precision)
🚧 OpenAI Realtime API            → Voice (<600ms RTT)
🚧 Apple IAP + Google Play Billing → Subscriptions
🚧 Analytics pipeline             → Retention dashboards
```

### **Timeline**
- **Weeks 1-2**: Backend + Safety
- **Weeks 3-4**: Voice + Avatar
- **Weeks 5-6**: 30 universal topics
- **Weeks 7-8**: Mobile + IAP
- **Week 9**: GPT Store
- **Week 10**: Testing
- **Week 11**: Beta (600 users)
- **Week 12**: Store submission → **LAUNCH**
- **Post-launch**: Scale to 365 topics

---

## 🎯 **Key Requirements (17 Total)**

### **P0 - Launch Blockers (9)**
1. Backend API responding
2. Safety router (≥98% precision)
3. Realtime voice (<600ms)
4. 60fps avatar
5. **30 universal topics** (not 90 lessons!)
6. Apple IAP
7. Google Play Billing
8. Privacy compliance
9. Store submission

### **P1 - Important (3)**
10. MCP server
11. Analytics
12. Device testing

### **P2 - Nice to Have (5)**
13. Advanced analytics
14. AR mode (post-launch)
15. Offline mode
16. Family dashboard
17. Multi-language UI

---

## 📁 **Project Structure**

```
UI-TARS-desktop/
├── 📚 Documentation (Read These)
│   ├── START_HERE.md                          ← Entry point
│   ├── CRITICAL_UPDATE_DAILY_LESSON_MODEL.md  ← The correction
│   ├── CURIOUS_KELLLY_EXECUTION_PLAN.md       ← 12-week roadmap
│   ├── TECHNICAL_ALIGNMENT_MATRIX.md          ← Asset mapping
│   └── GETTING_STARTED_CK.md                  ← Role setup
│
├── ✅ Working Assets (Keep & Use)
│   ├── lesson-player/            → Web dev tool
│   ├── digital-kelly/            → Flutter+Unity base
│   ├── kelly_pack/               → Avatar generation
│   ├── kelly_audio2face/         → Lip-sync
│   └── lessons/                  → Sample topics
│
└── 🚧 Production Build (To Create)
    └── curious-kellly/
        ├── backend/              → Node.js/Python
        ├── mobile/               → Flutter production
        ├── mcp-server/           → GPT Store
        └── content/              → 365 topics
```

---

## 🎓 **User's Preferences (From Memories)**

1. **Precompute everything** - No runtime AI models, all content authored
2. **Never use browser TTS** - ElevenLabs or OpenAI voice only
3. **Step-by-step instructions** - Not multiple options
4. **Review plans before executing** - Stay on track per CLAUDE.md
5. **Complete solutions** - No simplified summaries
6. **Think through plans first** - Correct on first try
7. **Use existing codebase** - Don't create new designs
8. **At least 60min audio** - For voice training (Kelly/Ken)

---

## 💡 **What Makes This Product Special**

1. **Age-morphing Kelly** - Not just content, Kelly herself ages
2. **Universal topics** - Work for toddler through centenarian
3. **Daily community** - Everyone on same topic creates shared experience
4. **No choice paralysis** - One topic per day, come back tomorrow
5. **Viral potential** - "What's today's topic?" becomes cultural moment

---

## 🎯 **Success Metrics (90-day post-launch)**

### **Product KPIs**
- D1 retention: ≥45%
- D30 retention: ≥20%
- Session length: ≥8 minutes
- CSAT: ≥4.6/5
- NPS: ≥+40

### **Technical KPIs**
- Voice RTT p50: ≤600ms
- Lip-sync error: <5%
- Frame rate: 60fps
- Crash-free: ≥99.5%
- Safety precision: ≥98%

### **Business KPIs**
- Downloads: 10,000+
- Paid subscribers: 1,000+
- Trial → paid: ≥15%
- Refund rate: <5%

---

## 🛠️ **Common Tasks & How to Help**

### **Content Creation**
- Template: `lessons/leaves-change-color.json`
- Schema: `lesson-player/lesson-dna-schema.json`
- Audio: `lesson-player/generate_audio.py`
- **Remember**: Universal topic, not course lesson!

### **Backend Development**
- Language: Node.js or Python (user's choice)
- Voice: OpenAI Realtime API (WebRTC)
- Safety: OpenAI Moderation API
- Vector DB: Pinecone or Qdrant

### **Mobile Development**
- Framework: Flutter 3.x
- 3D Engine: Unity 2022.3 LTS
- Test app: `digital-kelly/`
- Production: `curious-kellly/mobile/`

### **Avatar Work**
- Generation: `kelly_pack/cli.py`
- Lip-sync: `kelly_audio2face/`
- Ages needed: 3, 9, 15, 27, 48, 82

---

## 📋 **When User Asks for Help**

### **Content Questions**
- Point to: `lessons/leaves-change-color.json`
- Emphasize: Universal topic, not lesson track
- Remind: 6 age variants (2-5, 6-12, 13-17, 18-35, 36-60, 61-102)
- Kelly ages with slider (not just content)

### **Technical Questions**
- Check: TECHNICAL_ALIGNMENT_MATRIX.md
- Reference: Curious-Kellly_Technical_Blueprint.md
- Existing code: Map to their current assets

### **Planning Questions**
- Primary: CURIOUS_KELLLY_EXECUTION_PLAN.md
- Quick ref: QUICK_REFERENCE.md
- Tasks: CK_Launch-Checklist.csv

### **Product Questions**
- Vision: CRITICAL_UPDATE_DAILY_LESSON_MODEL.md
- Requirements: Curious-Kellly_PRD.md
- Acceptance: CK_Requirements-Matrix.csv

---

## 🚨 **Critical Reminders**

### **Always Remember**
1. The Daily Lesson = one topic for everyone each day
2. Kelly ages with the learner (appearance + voice + content)
3. 365 universal topics (not 90 lessons in tracks)
4. Launch with 30 topics, scale to 365 post-launch
5. User prefers precomputed content, no runtime AI

### **Never Suggest**
1. Course tracks or learning paths
2. User choice in lesson selection
3. Browser text-to-speech
4. Creating new lesson player (use existing)
5. Simplified summaries (user wants complete solutions)

### **Always Do**
1. Think through the plan first
2. Provide step-by-step instructions
3. Reference existing codebase
4. Check memories for user preferences
5. Update TODOs as tasks complete

---

## 🎯 **Most Likely Next Requests**

Based on current state, user will probably ask for:

1. **Content Creation**
   - Design 10 universal topics
   - Write age-adaptive scripts
   - Generate audio files

2. **Backend Setup**
   - Scaffold Node.js/Python service
   - Integrate OpenAI Realtime API
   - Build safety router

3. **Avatar Work**
   - Create 6 Kelly age variants
   - Set up age-switching logic
   - Test avatar rendering

4. **Mobile Development**
   - Migrate digital-kelly to production
   - Add IAP integration
   - Test on devices

5. **Documentation Updates**
   - Fix remaining "track" references
   - Update Sprint 2 content plan
   - Create topic creation guide

---

## ✅ **Ready to Serve Checklist**

- [x] Understand The Daily Lesson model
- [x] Know correct vs incorrect terminology
- [x] Mapped existing assets to requirements
- [x] 12-week timeline internalized
- [x] User preferences from memories noted
- [x] Common tasks documented
- [x] Key documents indexed
- [x] Success metrics clear
- [x] Critical reminders listed
- [x] Ready for any question

---

## 📞 **How to Respond**

### **When User Asks "What should I do?"**
→ Point to current sprint (Week 1-2: Backend + Safety)
→ Reference START_HERE.md or GETTING_STARTED_CK.md

### **When User Asks "How do I build X?"**
→ Check TECHNICAL_ALIGNMENT_MATRIX.md for existing assets
→ Reference CURIOUS_KELLLY_EXECUTION_PLAN.md for implementation
→ Provide step-by-step instructions (user preference)

### **When User Questions the Plan**
→ Acknowledge feedback
→ Update documents immediately
→ Provide corrected summary

### **When User Shows Progress**
→ Update TODO list
→ Celebrate the win
→ Suggest next task

---

**Status**: ✅ **FULLY PREPARED**  
**Understanding**: Complete and correct  
**Readiness**: 100%  
**Awaiting**: User's direction

**Ready to build Curious Kellly!** 🚀














