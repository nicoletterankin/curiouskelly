# 🚀 SOTA Video Pipeline - API Setup Guide

> **Goal:** Make Kelly the best digital human teacher on the planet using state-of-the-art AI video generation APIs.

---

## Quick Start

```bash
# Install dependencies
npm install replicate @fal-ai/client

# Run the pipeline
npx tsx scripts/kelly-video-factory/sota-video-pipeline.ts --tier best-available --text "Hello learners!"
```

---

## API Keys Required

Add these to your `.env` file:

```bash
# REQUIRED - Core APIs
REPLICATE_API_TOKEN=r8_xxxxxxxxxxxxx          # Image gen, LivePortrait, SadTalker
ELEVENLABS_API_KEY=xxxxxxxxxxxxxxxx           # Kelly's voice

# OPTIONAL - Premium APIs (choose based on budget)
SYNC_LABS_API_KEY=sync_xxxxxxxxxxxxx          # Premium lip-sync (recommended!)
HEDRA_API_KEY=hedra_xxxxxxxxxxxxx             # Full face animation
FAL_KEY=xxxxxxxxxxxxxxxx                       # OmniHuman, advanced models
```

---

## API Signup Links & Pricing

### 1. Sync Labs (HIGHEST PRIORITY)
**Why:** Best lip-sync quality available (95%+ accuracy, 4K support)

| Plan | Price | Features |
|------|-------|----------|
| Free | $0 | 5 min/month, 720p |
| Pro | $29/mo | 60 min, 1080p |
| Business | $99/mo | 300 min, 4K |

**Sign Up:** https://sync.so
**Docs:** https://docs.sync.so

```bash
# Test Sync Labs API
curl -X POST https://api.sync.so/v2/lipsync \
  -H "Authorization: Bearer $SYNC_LABS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "lipsync-2", "input": {"video_url": "...", "audio_url": "..."}}'
```

---

### 2. Hedra (Character-1)
**Why:** Full facial animation, not just lips (eyes, brows, head movement)

| Plan | Price | Features |
|------|-------|----------|
| Free | $0 | Limited generations |
| Pro | $24/mo | More generations |
| Enterprise | Custom | Unlimited |

**Sign Up:** https://www.hedra.com
**API Docs:** https://docs.hedra.com

---

### 3. Replicate (Already Using)
**Why:** Access to dozens of models - LivePortrait, SadTalker, FLUX, upscalers

| Plan | Price | Features |
|------|-------|----------|
| Pay-as-you-go | ~$0.001-0.50/run | Per-model pricing |

**Sign Up:** https://replicate.com
**Dashboard:** https://replicate.com/account/api-tokens

---

### 4. fal.ai (OmniHuman, Advanced Models)
**Why:** Cutting-edge models, fast inference, OmniHuman support

| Plan | Price | Features |
|------|-------|----------|
| Free | $0 | $10 credits |
| Pay-as-you-go | Variable | Per-model pricing |

**Sign Up:** https://fal.ai
**Dashboard:** https://fal.ai/dashboard

---

### 5. ElevenLabs (Already Using)
**Why:** Kelly's voice - premium quality TTS

| Plan | Price | Features |
|------|-------|----------|
| Free | $0 | 10k chars/month |
| Starter | $5/mo | 30k chars |
| Creator | $22/mo | 100k chars |
| Pro | $99/mo | 500k chars |

**Sign Up:** https://elevenlabs.io
**Voice ID:** `wAdymQH5YucAkXwmrdL0` (Kelly's voice)

---

## Tier Comparison

| Tier | Lip-Sync | Eyes/Brows | Head Motion | Body | Quality | Cost/30s |
|------|----------|------------|-------------|------|---------|----------|
| **Sync Labs** | 95% | ⚠️ Via base | ⚠️ Via base | ❌ | 🌟🌟🌟🌟🌟 | ~$0.20 |
| **Hedra** | 90% | ✅ Full | ✅ Natural | ❌ | 🌟🌟🌟🌟 | ~$0.15 |
| **LivePortrait** | 85% | ✅ Some | ✅ Driven | ❌ | 🌟🌟🌟 | ~$0.05 |
| **OmniHuman** | 90% | ✅ Full | ✅ Natural | ✅ | 🌟🌟🌟🌟🌟 | ~$0.30 |
| **SadTalker** | 70% | ❌ Static | ⚠️ Basic | ❌ | 🌟🌟 | ~$0.02 |

---

## Recommended Setup by Budget

### 💰 Budget Tier ($0-10/month)
```bash
REPLICATE_API_TOKEN=...   # LivePortrait, SadTalker
ELEVENLABS_API_KEY=...    # Kelly voice

# Use: --tier best-available (will use LivePortrait + SadTalker)
```

### 💎 Professional Tier ($30-50/month)
```bash
REPLICATE_API_TOKEN=...   
ELEVENLABS_API_KEY=...    
SYNC_LABS_API_KEY=...     # ⭐ Add this for premium lip-sync

# Use: --tier sync
```

### 🚀 Premium Tier ($100+/month)
```bash
REPLICATE_API_TOKEN=...   
ELEVENLABS_API_KEY=...    
SYNC_LABS_API_KEY=...     
HEDRA_API_KEY=...         # Full face animation
FAL_KEY=...               # OmniHuman for full body

# Use: --tier omnihuman for hero content
```

---

## Testing Your Setup

```bash
# 1. Test basic pipeline (LivePortrait + SadTalker)
npx tsx scripts/kelly-video-factory/sota-video-pipeline.ts \
  --text "Hello! Testing the basic pipeline." \
  --pose excited

# 2. Test Sync Labs (if configured)
npx tsx scripts/kelly-video-factory/sota-video-pipeline.ts \
  --tier sync \
  --text "Testing Sync Labs premium lip-sync!" \
  --upscale

# 3. Test Hedra (if configured)  
npx tsx scripts/kelly-video-factory/sota-video-pipeline.ts \
  --tier hedra \
  --text "Testing Hedra full face animation!" \
  --pose curious

# 4. Test OmniHuman (if configured)
npx tsx scripts/kelly-video-factory/sota-video-pipeline.ts \
  --tier omnihuman \
  --text "Testing OmniHuman full body animation!" \
  --full-body
```

---

## Quality Comparison Test

Run all tiers and compare:

```bash
npx tsx scripts/kelly-video-factory/quality-comparison-test.ts
```

This generates videos from all available tiers with the same audio and creates a comparison HTML page.

---

## Troubleshooting

### "SYNC_LABS_API_KEY not configured"
Sign up at https://sync.so and add the key to `.env`

### "Hedra API error"
Check your Hedra credits at https://www.hedra.com/dashboard

### "Rate limit exceeded"
Add delays between requests or upgrade your plan

### "Video quality not as expected"
1. Try `--upscale` flag for 4K output
2. Check source image quality (higher = better)
3. Try different tiers

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SOTA KELLY VIDEO PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    │
│   │  Kelly   │    │ElevenLabs│    │  Video   │    │ Upscaler │    │
│   │  LoRA    │───▶│   TTS    │───▶│Generator │───▶│  (opt)   │    │
│   │  Image   │    │  Audio   │    │          │    │  4K      │    │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘    │
│        │                               │                            │
│        └───────────────────────────────┤                            │
│                                        ▼                            │
│                            ┌─────────────────────┐                  │
│                            │   TIER SELECTION    │                  │
│                            └─────────────────────┘                  │
│                                        │                            │
│          ┌────────────┬────────────┬───┴───┬────────────┐          │
│          ▼            ▼            ▼       ▼            ▼          │
│    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐             │
│    │  Sync    │ │  Hedra   │ │ LivePort │ │OmniHuman │             │
│    │  Labs    │ │Character1│ │   rait   │ │          │             │
│    │  (95%)   │ │  (90%)   │ │  (85%)   │ │  (90%)   │             │
│    └──────────┘ └──────────┘ └──────────┘ └──────────┘             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Next Steps

1. **Sign up for Sync Labs** - Biggest quality jump for the money
2. **Run comparison test** - See the quality difference yourself
3. **Set up batch processing** - Generate all lesson videos
4. **Enable 4K upscaling** - For hero content

---

## Files Created

- `scripts/kelly-video-factory/sota-video-pipeline.ts` - Main pipeline
- `scripts/kelly-video-factory/quality-comparison-test.ts` - Compare all tiers
- `docs/SOTA_VIDEO_API_SETUP.md` - This guide

---

**Goal:** By December 17, Kelly will have the best digital human video quality available through API-based generation.

When the artist delivers the iClone avatar, we'll layer that in as **Tier 0 (Cinema Quality)** for the absolute best content.



