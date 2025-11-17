# Curious Kellly - Quick Reference Card
**One-Page Cheat Sheet for Daily Use**

---

## 🎯 The Mission
**Launch Curious Kellly on iOS, Android, and GPT Store in 12 weeks**

---

## 📚 Essential Documents (Bookmark These)

| **Document** | **When to Use** | **Reading Time** |
|-------------|----------------|-----------------|
| **START_HERE.md** | First time setup | 5 min |
| **CURIOUS_KELLLY_EXECUTION_PLAN.md** | Weekly sprint planning | 30 min |
| **TECHNICAL_ALIGNMENT_MATRIX.md** | Architecture decisions | 10 min |
| **GETTING_STARTED_CK.md** | Role-specific setup | 15 min |
| **CK_Requirements-Matrix.csv** | Acceptance criteria | Ref only |
| **CK_Launch-Checklist.csv** | Daily task tracking | Ref only |

---

## 🗓️ 12-Week Timeline (At a Glance)

```
┌─────────────────────────────────────────────────────────────┐
│ Week 1-2:  Backend + Safety              [Sprint 0]         │
│ Week 3-4:  Voice + Avatar                [Sprint 1]         │
│ Week 5-6:  Content (90 lessons)          [Sprint 2]         │
│ Week 7-8:  Mobile + IAP                  [Sprint 3]         │
│ Week 9:    GPT Store + Claude            [Sprint 4]         │
│ Week 10:   Analytics + Testing           [Sprint 5]         │
│ Week 11:   Beta (600 users)              [Sprint 6]         │
│ Week 12:   Store Submission → LAUNCH! 🚀 [Sprint 7]         │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ Critical Path (The 9 Launch Blockers)

**Must complete to launch:**

1. ✅ Backend API (Week 1-2)
2. ✅ Safety Router (Week 1)
3. ✅ Realtime Voice (Week 3)
4. ✅ 60fps Avatar (Week 3-4)
5. ✅ 90 Lessons (Week 5-6)
6. ✅ Apple IAP (Week 7)
7. ✅ Google Play Billing (Week 7)
8. ✅ Privacy Compliance (Week 8)
9. ✅ Store Submission (Week 12)

**Skip first, add later:**
- MCP Server (Week 9)
- Advanced analytics (Week 10)
- AR mode (post-launch)

---

## 📂 Key Directories

| **Directory** | **Purpose** | **Status** |
|--------------|------------|-----------|
| `curious-kellly/backend/` | Production backend | 🔴 To build |
| `curious-kellly/mobile/` | Production app | 🔴 To build |
| `curious-kellly/content/` | 90 lessons | 🟡 1/90 done |
| `lesson-player/` | Dev tool | ✅ Working |
| `digital-kelly/` | Test app | ✅ Working |
| `kelly_pack/` | Avatar assets | ✅ Working |
| `kelly_audio2face/` | Lip-sync | ✅ Working |

---

## 🔧 Common Commands

### **Backend Development**
```bash
# Start backend
cd curious-kellly/backend
npm run dev  # or: python main.py

# Test safety router
curl -X POST http://localhost:3000/moderate \
  -H "Content-Type: application/json" \
  -d '{"text": "test message"}'
```

### **Mobile Development**
```bash
# Run Flutter app
cd curious-kellly/mobile
flutter run

# Run on specific device
flutter devices
flutter run -d <device-id>

# Build for production
flutter build ios --release
flutter build apk --release
```

### **Content Development**
```bash
# Test lesson player
cd lesson-player
python -m http.server 8000
# Open http://localhost:8000

# Generate audio for lesson
python generate_audio.py
```

### **Asset Generation**
```bash
# Generate Kelly avatar assets
python -m kelly_pack.cli build --outdir ./output --device cuda
```

---

## 🎯 Success Metrics (Quick Check)

### **Technical KPIs**
- Voice RTT: p50 ≤600ms ✅ or ❌
- Frame rate: 60fps ✅ or ❌
- Lip-sync error: <5% ✅ or ❌
- Crash-free: ≥99.5% ✅ or ❌
- Safety precision: ≥98% ✅ or ❌

### **Product KPIs (90-day post-launch)**
- D1 retention: ≥45%
- D30 retention: ≥20%
- Session length: ≥8 min
- CSAT: ≥4.6/5
- Paid subs: 1,000+

---

## 🚨 Emergency Contacts

### **Technical Issues**
- Backend broken? Check logs: `docker logs backend`
- Flutter crash? Run: `flutter clean && flutter pub get`
- Unity not responding? Restart Unity Hub

### **Store Submission Issues**
- Apple rejection? Check App Review Guidelines
- Google Play rejection? Check Developer Policy

### **API Issues**
- OpenAI rate limit? Check billing/usage
- ElevenLabs quota? Check plan limits

---

## 💡 Quick Wins (When You're Stuck)

**Need motivation?**
- Test lesson player (it works!)
- Generate audio for a lesson
- Move age slider and see adaptation

**Need progress?**
- Update CK_Launch-Checklist.csv
- Mark off completed tasks
- Demo what you built

**Need clarity?**
- Re-read relevant section of execution plan
- Check requirements matrix for acceptance criteria
- Ask team lead

---

## 📋 Daily Standup Template

```
Yesterday: [What I completed]
Today: [What I'm working on]
Blockers: [None / specific issue]
Demo: [Link / screenshot if available]
```

---

## 🎓 Learning Resources (5-Min Reads)

**Real-time Voice**
→ https://platform.openai.com/docs/guides/realtime

**Flutter Unity Widget**
→ https://pub.dev/packages/flutter_unity_widget

**Apple IAP**
→ https://developer.apple.com/in-app-purchase/

**Google Play Billing**
→ https://developer.android.com/google/play/billing

**MCP Protocol**
→ https://spec.modelcontextprotocol.io/

---

## 🔍 Troubleshooting (Common Issues)

### **"OpenAI API not responding"**
```bash
# Check API key
echo $OPENAI_API_KEY

# Test with curl
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### **"Flutter app won't build"**
```bash
flutter clean
flutter pub get
flutter doctor -v  # Check for issues
```

### **"Unity embed not showing"**
```bash
# iOS: Check podfile
cd ios && pod install

# Android: Check gradle
cd android && ./gradlew clean
```

### **"Lesson player audio won't play"**
- Use local server (not file://)
- Check browser console for CORS errors
- Verify audio files exist in videos/audio/

---

## ✅ Pre-Launch Checklist (Week 12)

**Before submitting to stores:**
- [ ] All 90 lessons complete and tested
- [ ] IAP working in production
- [ ] Privacy labels accurate
- [ ] Beta tested with 300+ users
- [ ] Crash-free rate ≥99.5%
- [ ] Voice latency <600ms p50
- [ ] Avatar at 60fps
- [ ] Safety router active
- [ ] Analytics tracking
- [ ] Customer support ready

---

## 📞 Who to Ask

| **Question About...** | **Ask...** |
|----------------------|-----------|
| Product priorities | PM |
| Backend architecture | Backend Lead |
| Mobile app issues | Mobile Lead |
| Content creation | Content Lead |
| Store submission | PM |
| Technical blockers | Team Lead |

---

## 🎉 Celebrating Wins

### **Week 1 Win**: Backend responding
### **Week 4 Win**: Voice conversation works
### **Week 8 Win**: Can purchase subscription
### **Week 11 Win**: Beta users loving it
### **Week 12 Win**: Apps submitted
### **Day 84 Win**: 🚀 **LAUNCH!**

---

## 📊 Where We Are (Update Weekly)

**Current Sprint**: Sprint ___ (Week ___)  
**Completion**: ___% (__ of 17 requirements done)  
**Blockers**: ___ (None / List if any)  
**Next Milestone**: ___ (Due: ____)

---

## 🚀 Next Actions (Always Know These 3)

1. **Today**: ___________________________
2. **This Week**: ___________________________
3. **This Sprint**: ___________________________

---

**Print this out. Keep it visible. Update it weekly.** 📌

**You've got this! 🚀**


















