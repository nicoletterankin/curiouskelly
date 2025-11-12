# Curious Kellly — Product Requirements Document (PRD)
**Date:** October 29, 2025  
**Author:** Steve Jobs (GPT-5 Pro), in collaboration with the Curious Kellly team  
**One‑line:** *An insanely great, multimodal teacher-avatar that greets every learner each day with a personalized, face‑to‑face micro‑lesson, project coaching, and a sense of wonder.*

## 1. Purpose
Curious Kellly (CK) is a daily, avatar‑led learning companion. It blends expressive real‑time voice, facial animation, and adaptive pedagogy to deliver **5–15 minute** personalized sessions that build skills over time. The avatar is **present, helpful, and human‑centred**—never gimmicky. CK runs on iOS, Android, and as a storefront experience in the GPT Store and Claude Artifacts.

## 2. North‑star outcomes (12 months)
- **D1 retention ≥ 45%, D30 ≥ 20%** for new cohorts (organic + paid).
- **Average daily session ≥ 8 minutes**, completion rate ≥ 70%.
- **Learning impact:** +15% improvement between baseline and week‑4 formative check-ins across shared Daily Lesson arcs.
- **CSAT ≥ 4.6/5**, **NPS ≥ +40**.
- **Safety:** 0 critical safety incidents; policy violation rate < 0.1% of sessions.

## 3. Target users & needs
- **Learners (13+)** who want compact, guided practice (languages, study skills, career skills, wellness micro‑habits).  
- **Parents & educators** seeking safe, structured daily routines.  
- **Prosumers** wanting an animated AI tutor that can look at notes, images, or whiteboards and explain with voice + visuals.

> **Compliance note:** If you target users **under 13**, COPPA/GDPR‑Kids obligations apply (see Compliance) — default age gate at onboarding; parent consent workflow.

## 4. Experience pillars (“feel”)
1) **Instant presence** — wake word or tap; <300 ms perceived start; avatar looks at you, tracks gaze, and speaks naturally.  
2) **Personal, not creepy** — memory is explicit and editable; consent on camera/voice use; private by default.  
3) **Progress you can feel** — tiny wins daily; weekly arc, monthly milestone; visible skill graph.  
4) **Multimodal clarity** — CK can see (camera, uploads), speak, show steps, and draw.  
5) **Trust & control** — session transcripts, sources, opt‑outs, and “Why this recommendation?” always available.

## 5. Scope (MVP, 90 days)
- **Daily avatar session** (5–15 min) with scripted scaffolds: warm‑up → explain → apply → reflect.
- **Daily Lesson calendar**: 30 universal launch topics (one per calendar day) with Kelly aging across six personas; roadmap to 365 topics.
- **Multilingual content**: English live delivery with Spanish/French variants precomputed in every PhaseDNA file (no runtime generation).
- **Multimodal I/O**: live voice both ways; image in (worksheet, whiteboard); screen cards out.
- **Starter personalization**: placement quiz → pace + difficulty; lightweight memory (goals, schedule, pronouns, tone).
- **Privacy dashboard**: data types, retention windows, delete/export.
- **Mobile apps**: iOS (App Store), Android (Google Play).  
- **Storefront**: GPT Store listing (routes into web experience) + Claude Artifact demo.
- **Billing**: Apple IAP + Google Play Billing for subscription; web waitlist for updates.

Out‑of‑scope MVP: full AR glasses mode; classroom multi‑user; offline synthesis; child (<13) accounts.

## 6. Detailed requirements
### 6.1 Avatar & voice
- Real‑time voice with low latency; barge‑in (interrupt) and barge‑out.
- Lifelike facial animation; mouth shape visemes synced to TTS; eye contact + idle micro‑motions.
- Styles: “warm coach,” “curious guide,” “celebratory.” User sets tone/pace.
- Safety filters before speaking (no unsafe output aloud).

### 6.2 Learning engine
- Session planner generates **lesson plan JSON** per day (objectives, steps, checks).  
- Knowledge sources: vetted curricula + retrieval from owned content; cite when using external facts.
- Formative checks: inline questions, image‑based problems, quick reflection prompts.
- End‑of‑week recap; end‑of‑month milestone (certificate/badge).

### 6.3 Multimodal understanding
- Vision: read photos of notes/worksheets; extract text; reason; overlay explanations.  
- Whiteboard mode: CK draws steps (cards/canvas) while talking; shareable transcript + board.

### 6.4 Personalization & memory
- Opt‑in memory with categories: profile, goals, progress, preferred feedback style.  
- Controls: *forget*, *export*, *pause memory*; per‑field retention (e.g., session notes 90 days).

### 6.5 Safety & compliance
- Age gate (13+ default), parental consent flow if <13 detected/selected.  
- Sensitive topics guarded; escalation to text‑only for certain queries; always cite.  
- Regional data residency toggle (US/EU). Content moderation on user input and model output.

### 6.6 Monetization & pricing
- Free trial: 7 days (3 sessions/day).  
- Monthly subscription via App Store / Google Play; annual plan with 2 months free.  
- Family sharing SKU (2–4 seats). edu discount later.

### 6.7 Analytics & ops
- Event schema: session start/end, interruptions, quiz items, hints, streaks, cancellations, errors.  
- KPIs: retention (D1/D7/D30), session length, skill delta, CSAT, refund rate, latency p50/p95.

## 7. Acceptance criteria (MVP)
- Start to first spoken word < **1.0 s** median on Wi‑Fi; < **1.8 s** LTE.  
- Word‑to‑viseme latency < **120 ms**; lip‑sync error rate (phoneme timing) < **5%**.  
- Average STT WER ≤ **6%** in quiet, ≤ **12%** moderate noise.  
- Safety: block rate precision ≥ **0.98**, recall ≥ **0.95** on policy test set.  
- App reviews ≥ **4.5★** post‑launch (first 1,000 ratings).

## 8. Dependencies
- Realtime LLM with speech‑in/speech‑out; on‑device audio capture; billing SDKs; vector DB; analytics; crash reporting.

## 9. Risks & mitigations
- **Latency spikes** → regional endpoints + edge caching; graceful degrade to text captions.  
- **Hallucinations** → retrieval, citations, answer‑checking critic pass for facts.  
- **App review rejections** → follow IAP and privacy rules; pre‑flight App Store/Play checklists.  
- **Privacy** → explicit consent, data minimization, privacy manifests.

## 10. Launch checklist (store‑specific in “Go‑to‑Market” section below)
- App privacy labels / Data safety form complete.
- Screenshots/video; localized descriptions; age rating; contact info; support URL.
- Beta flight/internal testing (TestFlight/Play testing) with ≥300 users.

---
