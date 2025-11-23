# Curious Kellly - Visual Roadmap
**12-Week Journey from Prototype to Production**

---

## 🎯 The Journey

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  TODAY                                                 DAY 84      │
│    │                                                      │        │
│    ▼                                                      ▼        │
│  Prototype ──────────────────────────────────────► PRODUCTION     │
│  Working                                              Launched!    │
│                                                                    │
│  • Lesson player                                  • iOS App       │
│  • 1 lesson                                       • Android App   │
│  • ElevenLabs                                     • GPT Store     │
│  • Basic avatar                                   • 90 lessons    │
│                                                   • 1000+ users   │
└────────────────────────────────────────────────────────────────────┘
```

---

## 📅 12-Week Sprint Timeline

```
SPRINT 0       SPRINT 1       SPRINT 2       SPRINT 3       SPRINT 4
Week 1-2       Week 3-4       Week 5-6       Week 7-8       Week 9
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│Backend │ => │ Voice  │ => │Content │ => │Mobile  │ => │  GPT   │
│Safety  │    │Avatar  │    │90 Less │    │  IAP   │    │ Store  │
└────────┘    └────────┘    └────────┘    └────────┘    └────────┘
    ↓             ↓             ↓             ↓             ↓

SPRINT 5       SPRINT 6       SPRINT 7       🚀 LAUNCH
Week 10        Week 11        Week 12        Day 84
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│Analytics   │ Beta   │    │Submit  │    │CURIOUS │
│Testing │    │600 usr │    │Stores  │    │KELLLY  │
└────────┘    └────────┘    └────────┘    └────────┘
```

---

## 🏗️ Build Phases (Detailed)

### **PHASE 0: Foundation** (Week 1-2)
```
┌─────────────────────────────────────┐
│ Day 1-2:  Backend scaffold          │ ← YOU ARE HERE
│ Day 3:    Safety router             │
│ Day 4:    Lesson system             │
│ Day 5:    Integration tests         │
│                                     │
│ Day 8-10: OpenAI Realtime API       │
│ Day 11-12: Vector DB setup          │
│ Day 13-14: Deploy staging           │
└─────────────────────────────────────┘

OUTPUT: ✅ Backend API responding
        ✅ Safety blocking bad content
        ✅ Lesson endpoints working
```

### **PHASE 1: Core Experience** (Week 3-4)
```
┌─────────────────────────────────────┐
│ Day 15-17: Realtime voice Flutter   │
│ Day 18-19: Test latency <600ms      │
│                                     │
│ Day 22-24: Unity 60fps upgrade      │
│ Day 25-26: Gaze tracking            │
│ Day 27-28: Device testing           │
└─────────────────────────────────────┘

OUTPUT: ✅ Voice conversation works
        ✅ Avatar at 60fps
        ✅ Lip-sync <5% error
```

### **PHASE 2: Content** (Week 5-6)
```
┌─────────────────────────────────────┐
│ Week 5:   Spanish A1 (30 lessons)   │
│ Week 6:   Study Skills (30)         │
│           Career Storytelling (30)  │
│                                     │
│ Parallel: Audio generation          │
│           Vector DB population      │
└─────────────────────────────────────┘

OUTPUT: ✅ 90 complete lessons
        ✅ Audio for all ages
        ✅ RAG corpus populated
```

### **PHASE 3: Monetization** (Week 7-8)
```
┌─────────────────────────────────────┐
│ Week 7:   Apple IAP                 │
│           Google Play Billing       │
│           Test subscriptions        │
│                                     │
│ Week 8:   Privacy labels            │
│           Data safety forms         │
│           Age gate + consent        │
└─────────────────────────────────────┘

OUTPUT: ✅ IAP working in sandbox
        ✅ Privacy compliant
        ✅ Ready for submission
```

### **PHASE 4-7: Polish & Launch** (Week 9-12)
```
┌─────────────────────────────────────┐
│ Week 9:   GPT Store + MCP           │
│ Week 10:  Analytics + Testing       │
│ Week 11:  Beta (600 users)          │
│ Week 12:  Store submission          │
│           🚀 LAUNCH DAY! 🎉         │
└─────────────────────────────────────┘

OUTPUT: ✅ Apps on stores
        ✅ GPT Store live
        ✅ Users learning!
```

---

## 📊 Component Dependencies

```
                    ┌──────────────┐
                    │   Backend    │
                    │   (Week 1)   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌──────────┐ ┌──────────┐ ┌──────────┐
       │  Safety  │ │   RAG    │ │ Planner  │
       │ (Week 1) │ │(Week 1-2)│ │(Week 1-2)│
       └──────────┘ └──────────┘ └──────────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼───────┐
                    │ Realtime API │
                    │   (Week 3)   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌──────────┐ ┌──────────┐ ┌──────────┐
       │  Voice   │ │  Avatar  │ │ Content  │
       │(Week 3-4)│ │(Week 3-4)│ │(Week 5-6)│
       └──────────┘ └──────────┘ └──────────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼───────┐
                    │  Mobile App  │
                    │  (Week 7-8)  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌──────────┐ ┌──────────┐ ┌──────────┐
       │   IAP    │ │ Privacy  │ │  Beta    │
       │(Week 7-8)│ │(Week 7-8)│ │(Week 11) │
       └──────────┘ └──────────┘ └──────────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼───────┐
                    │   LAUNCH!    │
                    │   (Week 12)  │
                    └──────────────┘
```

---

## 🎯 Critical Path (The 9 Gates)

```
GATE 1      GATE 2      GATE 3      GATE 4      GATE 5
Backend  => Safety   => Voice    => Avatar   => Content
✅ API      ✅ Moderate ✅ <600ms   ✅ 60fps    ✅ 90 lessons
   │           │           │           │           │
   └───────────┴───────────┴───────────┴───────────┘
                           │
GATE 6      GATE 7      GATE 8      GATE 9      🏁 DONE
IAP      => Privacy  => Testing  => Submit   => Launch
✅ Works    ✅ Labels   ✅ Pass     ✅ Approved  🚀 Live
```

**If any gate fails, you cannot proceed to launch.**

---

## 📈 Team Capacity Planning

```
BACKEND ENGINEER
Week 1-2: ████████████████████████ Backend + Safety
Week 3-4: ████████████████░░░░░░░░ Voice integration
Week 5-6: ████░░░░░░░░░░░░░░░░░░░░ RAG support
Week 7-8: ████░░░░░░░░░░░░░░░░░░░░ API polish
Week 9+:  ████████████████████████ Analytics + ops

MOBILE ENGINEER
Week 1-2: ░░░░░░░░░░░░░░░░░░░░░░░░ (Preparing)
Week 3-4: ████████████████████████ Voice + Avatar
Week 5-6: ████░░░░░░░░░░░░░░░░░░░░ Content testing
Week 7-8: ████████████████████████ IAP + Privacy
Week 9+:  ████████████████████████ Beta + submission

CONTENT CREATOR
Week 1-2: ████░░░░░░░░░░░░░░░░░░░░ Planning
Week 3-4: ████░░░░░░░░░░░░░░░░░░░░ Sample lessons
Week 5-6: ████████████████████████ 90 LESSONS!
Week 7-8: ████████████████░░░░░░░░ Polish + audio
Week 9+:  ████░░░░░░░░░░░░░░░░░░░░ Beta content

AI/ML ENGINEER (Part-time)
Week 1-2: ████████████████████████ Safety + RAG
Week 3-4: ████████░░░░░░░░░░░░░░░░ Voice testing
Week 5-6: ████████████████░░░░░░░░ RAG population
Week 7+:  ████░░░░░░░░░░░░░░░░░░░░ Monitoring
```

---

## 💰 Budget Timeline

```
WEEK 1-2 (Setup)
- Apple Dev Program:        $99
- Google Play Console:      $25
- OpenAI credits:          $100
- Vector DB:               $50
TOTAL:                     $274

WEEK 3-8 (Development)
- OpenAI API:              $600  ($100/week × 6)
- ElevenLabs:              $300  ($50/week × 6)
- Hosting:                 $180  ($30/week × 6)
- Vector DB:               $300  ($50/week × 6)
TOTAL:                   $1,380

WEEK 9-12 (Launch)
- OpenAI API:              $400
- Analytics:               $200
- Support tools:           $100
- Misc:                    $100
TOTAL:                     $800

GRAND TOTAL:             $2,454
```

---

## 🎯 Success Metrics Over Time

```
RETENTION TARGET
100%│
    │  ◀─ D1 Target: 45%
 80%│   ●
    │    ╲
 60%│     ╲
    │      ●
 40%│       ╲  ◀─ D7 Target: 30%
    │        ╲
 20%│         ● ◀─ D30 Target: 20%
    │          ╲___________
  0%└──────────────────────────►
    D1    D7         D30      D90


DOWNLOAD GROWTH
10K│                        ● ← Target
   │                       ╱
 8K│                      ╱
   │                     ╱
 6K│                    ╱
   │                   ╱
 4K│                  ╱
   │                 ╱
 2K│        ● ← Week 11 Beta
   │       ╱
  0└──────┴──────────────────────►
    Launch  Week 1-4  Week 5-8  Week 9-12
```

---

## 🚀 Launch Checklist Visual

```
WEEK 12 - FINAL COUNTDOWN

Monday        Tuesday       Wednesday     Thursday      Friday
┌───────┐    ┌───────┐    ┌───────┐    ┌───────┐    ┌───────┐
│Assets │ => │Submit │ => │Review │ => │Monitor│ => │LAUNCH!│
│Ready  │    │ iOS   │    │Status │    │Apps   │    │  🎉   │
└───────┘    └───────┘    └───────┘    └───────┘    └───────┘
    │            │            │            │            │
    ▼            ▼            ▼            ▼            ▼
Screenshots  App Store  Wait for     Fix any      GO LIVE
Videos       Connect    approval     issues       Celebrate!
Copy         Submission  (1-3 days)  (<15 min)    🚀🎊🎉
```

---

## 🎓 Learning Curve

```
KNOWLEDGE REQUIRED
High│
    │                        ┌─────────
    │                       ╱   Expert
    │                      ╱
Med │            ┌────────┘
    │           ╱  Confident
    │          ╱
Low │ ┌───────┘
    │ │Learning
    └─┴──────────────────────────────►
      Week    Week    Week    Week
      1-3     4-6     7-9     10-12

By Week 12, you'll be an expert in:
✅ OpenAI Realtime API
✅ Flutter + Unity integration
✅ Avatar rendering & lip-sync
✅ App Store submission
✅ IAP implementation
```

---

## 📊 Deliverables by Week

```
Week 1:  ✅ Backend API responding
Week 2:  ✅ Safety router working
Week 3:  ✅ Voice conversation works
Week 4:  ✅ Avatar at 60fps
Week 5:  ✅ 30 Spanish lessons
Week 6:  ✅ 90 total lessons
Week 7:  ✅ IAP in sandbox
Week 8:  ✅ Privacy compliant
Week 9:  ✅ GPT Store live
Week 10: ✅ Testing complete
Week 11: ✅ Beta launched
Week 12: 🚀 PRODUCTION LAUNCH!
```

---

## 🎯 Your Position on the Map

```
                  YOU ARE HERE
                       ↓
┌─────────────────────●─────────────────────────────────┐
│                     │                                  │
│  Prototype          │                   Production     │
│  (Done)             │                   (12 weeks)     │
│                     │                                  │
│  ✅ Lesson player    │    🚧 Backend                    │
│  ✅ 1 lesson         │    🚧 Realtime voice             │
│  ✅ ElevenLabs       │    🚧 60fps avatar               │
│  ✅ Basic Unity      │    🚧 90 lessons                 │
│  ✅ Audio2Face       │    🚧 Mobile apps                │
│                     │    🚧 IAP                        │
│                     │    🚧 GPT Store                  │
│                     │                                  │
└─────────────────────┴─────────────────────────────────┘
```

**Next Step**: Read [START_HERE.md](./START_HERE.md) (5 minutes)

---

**Print this out. Put it on your wall. Track your progress!** 📍

**You're 10% done. Let's do the other 90%!** 🚀






















