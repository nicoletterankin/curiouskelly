# 🔑 BYOK Flywheel: The Magic Paintbrush

## Vision

**Pool the internet's AI resources to power Kelly** - transforming Curious Kelly into a learner-created platform where community contributions fuel everyone's learning experience.

---

## The Flywheel

```
Student adds free HeyGen key
         ↓
System generates 1 video
         ↓
Video available to ALL students
         ↓
Community grows, more keys added
         ↓
More videos generated faster
         ↓
Premium experience at scale
         ↓
         ↺ REPEAT
```

---

## Supported Providers

| Provider | Capability | Free Tier | Use Case |
|----------|------------|-----------|----------|
| 🎬 **HeyGen** | Video avatars | Yes! | Kelly lesson videos |
| 🤖 **OpenAI** | Chat, TTS, Images | $5 credit | Live chat, DALL-E visuals |
| 🧠 **Anthropic** | Chat, Vision | $5 credit | Deep conversations |
| ✨ **Google AI** | Chat, Images | Generous | Gemini chat, Imagen visuals |
| 🎙️ **ElevenLabs** | TTS, Voice | 10k chars | Premium voice synthesis |

---

## Database Schema

### `byok_keys` - User API Keys
- Encrypted client-side before storage
- One key per provider per user
- Tracks usage and validation status

### `kelly_keys` - Platform Credits
- Platform-provided pooled credits
- Daily/monthly limits
- Fair distribution to all users

### `generation_queue` - Batch Processing
- Community-driven content generation
- Priority-based processing
- Tracks who contributed what

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    BYOK MANAGER                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ OpenAI   │  │Anthropic │  │ Google   │  │ HeyGen   │ │
│  │  Chat    │  │  Chat    │  │ Images   │  │ Video    │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘ │
│       │             │             │             │       │
│       └─────────────┴─────────────┴─────────────┘       │
│                          │                               │
│              ┌───────────▼───────────┐                  │
│              │  GENERATION QUEUE     │                  │
│              │  - Batch processing   │                  │
│              │  - Community pooling  │                  │
│              │  - Fair distribution  │                  │
│              └───────────┬───────────┘                  │
│                          │                               │
│              ┌───────────▼───────────┐                  │
│              │   CONTENT LIBRARY     │                  │
│              │  - Videos (kelly_video_assets)           │
│              │  - Visuals (visual_commons)              │
│              │  - Audio (lesson_audio)                  │
│              └───────────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

---

## API Reference

### BYOKManager

```javascript
// Initialize
BYOKManager.init();

// Check if provider is configured
BYOKManager.hasProvider('heygen'); // true/false

// Save key (with validation)
await BYOKManager.saveKey('heygen', 'your-api-key');

// Generate content
await BYOKManager.generate('video', {
  script: 'Hello learners!',
  avatarId: 'kelly-avatar'
});

// Get capabilities
BYOKManager.getAvailableCapabilities();
// ['chat', 'video', 'image', 'tts']
```

### KellyGenerationQueue

```javascript
// Initialize with Supabase
KellyGenerationQueue.init(supabaseClient);

// Queue a video
await KellyGenerationQueue.enqueue({
  dayNumber: 17,
  phase: 'hook',
  generationType: 'video',
  inputData: { script: 'Welcome to Day 17!' }
});

// Queue next week's lessons
await KellyGenerationQueue.queueNextWeek();

// Start automatic processing
KellyGenerationQueue.startProcessing();

// Get community stats
const stats = await KellyGenerationQueue.getCommunityStats();
// { totalGenerated: 150, contributors: 23, byType: { video: 100, visual: 50 } }
```

---

## UI Integration

Settings → AI Keys Hub shows:
- 5 provider cards with connect buttons
- Capability indicators (unlocked/locked)
- Privacy notice (keys stored locally)

---

## Security

1. **Keys encrypted client-side** before any storage
2. **Never sent to our servers** - only used client-to-provider
3. **RLS policies** ensure users only access their own keys
4. **Key validation** before saving

---

## Future Enhancements

1. **KELLY_KEYS gifting** - Earn credits by contributing
2. **Affiliate tracking** - HeyGen referral revenue
3. **Key sharing circles** - Trusted groups pool credits
4. **Usage analytics** - Show community contribution impact
5. **One-click OAuth** - Skip manual key entry

---

## Impact

This transforms Kelly from a **content-consumption** platform to a **content-creation** platform:

- **Students become contributors** by adding their free AI credits
- **Platform scales** without linear cost increase
- **Community owns** the learning experience
- **Everyone benefits** from pooled resources

---

*The Magic Paintbrush is now in the hands of the community.* 🎨✨
