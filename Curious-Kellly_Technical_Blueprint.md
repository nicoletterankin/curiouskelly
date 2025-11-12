# Curious Kellly — Technical Blueprint
**Date:** October 29, 2025

## 0. Architecture (high level)
**Client (iOS/Android/Web):**
- WebRTC audio capture/playback; lip‑sync via viseme stream; local wake‑word; on‑device VAD/endpointing.
- Canvas/SceneKit (iOS) / Filament (Android) 2.5D avatar; captioning and accessibility.
- Feature flags, remote config; crash reporting.

**Backend:**
- **Realtime LLM** (speech‑in/speech‑out) with tool calling (OpenAI Realtime API preferred). citeturn21search6turn21search0  
- **Orchestration service**: session state, lesson planner, safety router.  
- **RAG**: vector DB (Pinecone/Qdrant/Weaviate) with curated content; signed URLs to object storage. citeturn9search0turn9search3turn9search5  
- **User/Subscriptions**: App Store / Play Billing webhooks; entitlement service.  
- **Analytics pipeline**: event ingestion → warehouse → BI.  
- **Moderation**: input/output moderation before speech; audit logs.

## 1. Realtime & voice
- Use **OpenAI Realtime API (WebRTC/SIP)**. Client retrieves ephemeral key from backend; never ship standard API keys to clients. citeturn21search3  
- Barge‑in support using endpointing; text + audio stream back.  
- Fall back path: STT (Whisper/Cloud Speech) → text LLM → TTS, if realtime model unavailable. (See pricing links.) citeturn14search0

## 2. Avatar
- **Lip‑sync**: viseme stream → blendshape targets; 60 fps update.  
- **Gaze**: screen‑space target tracking; micro‑saccades 2–4/s; blinks 8–12/min.  
- **Accessibility**: live captions; speech rate slider; high‑contrast UI.

## 3. Data & privacy
- Consent flows; privacy manifests (iOS) and Data safety (Android). citeturn6search2turn5search2  
- App Tracking Transparency on iOS if tracking/ads; offer “Sign in with Apple.” citeturn6search1  
- Regional storage controls; encryption at rest; data minimization; logs with redaction.

## 4. Mobile platform specifics
- **iOS:** AVAudioEngine, Speech framework, AVSpeechSynthesizer; App Review & Small Business Program. citeturn7search0turn7search1  
- **Android:** AudioRecord/AudioTrack; SpeechRecognizer/TextToSpeech; target **API 35** by Aug 31, 2025; Play Billing v7+. citeturn7search2turn7search3turn11search0turn16search2

## 5. Billing & subscriptions
- **Apple IAP:** 30% standard; 15% Small Business; 85% net after year‑1 for auto‑renewing subscriptions. EU DMA terms may vary; see Apple’s DMA page. citeturn17search3turn17search1turn17search5turn17search2  
- **Google Play Billing:** 15% for first $1M/year; subs 15% day one; Billing Library v7+ and target SDK deadlines. citeturn23search0

## 6. Safety stack
- **Moderation**: provider moderation API + custom rules; block before speech; safe‑completion rewrites.  
- **Telemetry**: flag categories; reviewer tools; weekly audits.

## 7. Observability
- Metrics: latency, error buckets, crashes, tool success, moderation triggers.  
- Tracing across client ↔ realtime ↔ tools.

## 8. Test plan
- Unit tests for lesson planner; golden audio/viseme tests; device matrix (iPhone 12–current; Pixel 6–current).  
- Store pre‑flight (screenshots, privacy labels, IARC rating on Play).

---
