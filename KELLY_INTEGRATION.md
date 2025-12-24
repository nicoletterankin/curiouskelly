# Kelly Avatar Integration Guide

**Date:** December 2025  
**Purpose:** How to embed and control Kelly avatar in Kelly OS  
**Status:** Production-ready (2D videos), Legacy (Unity WebGL)

---

## Overview

Kelly appears in two forms:
1. **2D HD Videos** (Production) - Pre-generated MP4 videos with lip-sync
2. **Unity WebGL** (Legacy) - 3D avatar in browser (not in production)

**Recommendation:** Use 2D videos for production. Unity WebGL is legacy and not actively maintained.

---

## 2D HD Videos (Production)

### Architecture

**Storage:** Supabase Storage (`kelly-videos` bucket)  
**Format:** MP4, 1080p, H.264  
**Naming:** `videos/day-{dayNumber}/{archetype}/{phase}_{type}.mp4`

**Types:**
- `main` - Main script video
- `response_a` - Option A response
- `response_b` - Option B response
- `response_c` - Option C response

### Integration

**1. Get Video URL from Database:**

```typescript
// From lesson_atoms table
const atom = await supabase
  .from('lesson_atoms')
  .select('hd_video_url, content')
  .eq('core_lesson_id', lessonId)
  .eq('archetype', archetype)
  .eq('phase', phase)
  .single();

const videoUrl = atom.data.hd_video_url;
```

**2. Display Video:**

```html
<video 
  id="kelly-video" 
  src="{{ videoUrl }}" 
  autoplay 
  muted
  playsinline
></video>
```

**3. Control Playback:**

```javascript
const video = document.getElementById('kelly-video');

// Play
video.play();

// Pause
video.pause();

// Seek
video.currentTime = 10; // seconds

// Events
video.addEventListener('ended', () => {
  // Video finished
});

video.addEventListener('timeupdate', () => {
  // Update progress
});
```

**4. Lip-Sync Integration:**

```javascript
// Use kelly-lipsync.js for real-time lip-sync
import { KellyLipSync } from './js/kelly-lipsync.js';

const lipSync = new KellyLipSync({
  videoElement: document.getElementById('kelly-video'),
  audioElement: document.getElementById('kelly-audio'),
  onLipSync: (amplitude) => {
    // Update visual based on amplitude
  }
});

lipSync.start();
```

### File: `public/js/kelly-lipsync.js`

**Purpose:** Real-time lip-sync animation based on audio amplitude

**Usage:**
```javascript
const lipSync = new KellyLipSync({
  videoElement: videoEl,
  audioElement: audioEl,
  canvasElement: canvasEl, // Optional: overlay canvas
  onLipSync: (amplitude) => {
    // Amplitude 0-1, use to animate mouth
  }
});

lipSync.start();
lipSync.stop();
```

**How It Works:**
1. Analyzes audio amplitude in real-time
2. Maps amplitude to mouth shape
3. Updates visual overlay or triggers animation
4. Smooth transitions between shapes

---

## Unity WebGL (Legacy - Not Recommended)

### Status

**⚠️ Legacy:** Unity WebGL integration exists but is not in production.  
**Reason:** Too heavy, 2D videos preferred.

### Location

**Files:** `public/unity/` directory  
**Build:** Unity WebGL export

### Communication

**Bidirectional:** Uses `postMessage` API

**Web App → Unity:**
```javascript
const unityInstance = // Unity instance

unityInstance.SendMessage('KellyController', 'Speak', 'Hello!');
unityInstance.SendMessage('KellyController', 'SetExpression', 'happy');
unityInstance.SendMessage('KellyController', 'SetPose', 'teaching');
```

**Unity → Web App:**
```javascript
window.addEventListener('message', (event) => {
  const { type, data } = event.data;
  
  if (type === 'speaking') {
    // Kelly started speaking
  } else if (type === 'expression') {
    // Expression changed
  }
});
```

### Parameters

**Available Commands:**
- `Speak(text)` - Trigger speech animation
- `SetExpression(expression)` - Change facial expression
  - Values: `happy`, `thoughtful`, `encouraging`, `celebrating`, `listening`
- `SetPose(pose)` - Change body pose
  - Values: `teaching`, `listening`, `thinking`, `celebrating`
- `PlayAnimation(animation)` - Play specific animation
  - Values: `wave`, `point`, `nod`, `gesture`

**Example:**
```javascript
unityInstance.SendMessage('KellyController', 'Speak', 'Welcome to today\'s lesson!');
unityInstance.SendMessage('KellyController', 'SetExpression', 'happy');
unityInstance.SendMessage('KellyController', 'SetPose', 'teaching');
```

### Integration

**1. Load Unity Build:**

```html
<script src="Build/UnityLoader.js"></script>
<script>
  var unityInstance = UnityLoader.instantiate("unityContainer", "Build/your-build.json");
</script>
<div id="unityContainer"></div>
```

**2. Wait for Load:**

```javascript
unityInstance.on('progress', (progress) => {
  console.log(`Loading: ${progress * 100}%`);
});

unityInstance.on('loaded', () => {
  console.log('Unity loaded!');
  // Start controlling Kelly
});
```

**3. Control Kelly:**

```javascript
// Speak
unityInstance.SendMessage('KellyController', 'Speak', 'Hello!');

// Change expression
unityInstance.SendMessage('KellyController', 'SetExpression', 'happy');

// Listen for events
window.addEventListener('message', (event) => {
  console.log('Unity event:', event.data);
});
```

---

## Audio Integration

### Kelly's Voice

**Provider:** ElevenLabs  
**Voice ID:** `wAdymQH5YucAkXwmrdL0`  
**Model:** `eleven_multilingual_v2`

### File: `public/js/kelly-audio.js`

**Purpose:** Audio playback system for Kelly's voice

**Usage:**
```javascript
import { KellyAudio } from './js/kelly-audio.js';

const audio = new KellyAudio({
  kellyVoiceId: 'wAdymQH5YucAkXwmrdL0',
  onSpeakingStart: () => {
    // Kelly started speaking
  },
  onSpeakingEnd: () => {
    // Kelly finished speaking
  },
  lipSyncEnabled: true
});

// Speak text
await audio.speak('Hello! Welcome to today\'s lesson.');

// Pause
audio.pause();

// Resume
audio.resume();

// Stop
audio.stop();
```

**Features:**
- ✅ ElevenLabs TTS only (no browser TTS)
- ✅ Pre-generated audio support
- ✅ Lip-sync integration
- ✅ Play/pause/mute controls
- ✅ Audio caching

**⚠️ Important:** Browser TTS is NEVER used. If no ElevenLabs key, Kelly shows text only (silent mode).

---

## Conversational AI Integration

### File: `public/js/kelly-conversation.js`

**Purpose:** Real-time voice conversation with Kelly

**Usage:**
```javascript
import { KellyConversation } from './js/kelly-conversation.js';

const conversation = KellyConversation.init({
  agentId: 'your-agent-id', // Or use signed URL
  lessonContext: {
    topic: 'Today\'s lesson topic',
    currentPhase: 'hook',
    summary: 'Lesson summary'
  },
  onExpression: (expression) => {
    // Update Kelly's expression
  },
  onSpeakingStart: () => {
    // Kelly started speaking
  },
  onSpeakingEnd: () => {
    // Kelly finished speaking
  }
});

// Start conversation
await conversation.start();

// Stop conversation
await conversation.stop();
```

**Features:**
- ✅ Real-time voice conversation
- ✅ Lesson context awareness
- ✅ Expression callbacks
- ✅ WebSocket-based
- ✅ Automatic reconnection

**Setup:**
1. Get signed URL from `/api/elevenlabs-signed-url`
2. Initialize conversation with URL
3. Start listening/speaking
4. Handle expressions and events

---

## Best Practices

### Performance

1. **Preload Videos:** Load next phase video while current plays
2. **Cache Audio:** Cache generated audio to avoid regeneration
3. **Lazy Load:** Only load videos when needed
4. **Optimize Size:** Use compressed videos (H.264, optimized)

### User Experience

1. **Smooth Transitions:** Fade between videos
2. **Loading States:** Show loading indicator while video loads
3. **Error Handling:** Fallback to static image if video fails
4. **Accessibility:** Provide captions/subtitles

### Code Organization

1. **Separate Concerns:** Audio, video, lip-sync in separate modules
2. **Event-Driven:** Use events for communication
3. **Error Boundaries:** Handle errors gracefully
4. **Type Safety:** Use TypeScript for type safety

---

## Troubleshooting

### Video Not Loading

**Check:**
- Video URL is correct
- Supabase Storage bucket exists
- CORS is configured
- User has access (RLS policies)

### Audio Not Playing

**Check:**
- ElevenLabs API key is set
- Voice ID is correct
- Audio format is supported
- Browser permissions (autoplay)

### Lip-Sync Not Working

**Check:**
- Audio and video are synced
- `kelly-lipsync.js` is loaded
- Canvas element exists (if using overlay)
- Audio analysis is working

---

## Migration Notes

**From Unity to 2D Videos:**
1. Replace Unity instance with `<video>` element
2. Use `hd_video_url` from database
3. Remove Unity loader code
4. Update event handlers

**Benefits:**
- ✅ Faster loading
- ✅ Lower bandwidth
- ✅ Better compatibility
- ✅ Easier maintenance

---

**Status:** ✅ Production-ready (2D videos)  
**Legacy:** Unity WebGL (not recommended)  
**Next Steps:** Continue using 2D videos, deprecate Unity


