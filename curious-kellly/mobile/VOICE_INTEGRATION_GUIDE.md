# Curious Kellly - Voice Integration Guide

## 🎙️ **Overview**

This guide covers the OpenAI Realtime API integration for voice-to-voice conversation with Kelly.

---

## 🏗️ **Architecture**

```
FLUTTER APP
├─ VoiceController (state management)
│  ├─ OpenAIRealtimeService (WebRTC + WebSocket)
│  ├─ VoiceActivityDetector (speech detection)
│  ├─ AudioPlayerService (Kelly's voice output)
│  └─ PermissionService (microphone access)
│
↕ WebSocket (signaling)
│
BACKEND (Node.js)
├─ /api/realtime/ws (WebSocket endpoint)
│  ├─ SafetyService (moderation)
│  ├─ RealtimeService (Kelly responses)
│  └─ SessionService (conversation tracking)
│
↕ HTTPS
│
OPENAI REALTIME API
├─ WebRTC voice streaming
├─ Speech-to-text
├─ LLM processing
└─ Text-to-speech
```

---

## 📦 **Flutter Components**

### 1. **OpenAIRealtimeService** (`lib/services/openai_realtime_service.dart`)

Main service for voice streaming.

**Key Methods:**
```dart
// Connect to OpenAI Realtime API
Future<bool> connect({required int learnerAge});

// Start listening to user
void startListening();

// Stop listening
void stopListening();

// Barge-in (interrupt Kelly)
void bargeIn();

// Send text message
void sendMessage(String text);

// Disconnect
Future<void> disconnect();
```

**Streams:**
```dart
Stream<Uint8List> audioInputStream;   // User audio
Stream<Uint8List> audioOutputStream;  // Kelly audio
Stream<String> transcriptStream;      // User transcripts
Stream<RealtimeEvent> eventStream;    // All events
```

### 2. **VoiceActivityDetector** (`lib/services/voice_activity_detector.dart`)

Detects when user starts/stops speaking.

**Configuration:**
```dart
VoiceActivityDetector(
  silenceThreshold: 0.02,           // Energy threshold
  silenceDuration: Duration(milliseconds: 500),  // Silence before speech ends
  speechDuration: Duration(milliseconds: 300),   // Speech before activation
);
```

**Callbacks:**
```dart
vad.onSpeechStart = () => print('User started speaking');
vad.onSpeechEnd = () => print('User stopped speaking');
vad.onEnergyUpdate = (energy) => print('Energy: $energy');
```

### 3. **VoiceController** (`lib/controllers/voice_controller.dart`)

Coordinates all voice services.

**Usage:**
```dart
// Initialize
final voiceController = VoiceController(
  apiKey: 'your-openai-key',
  backendUrl: 'https://your-backend.onrender.com',
);

// Connect
await voiceController.connect(learnerAge: 35);

// Start conversation
voiceController.startListening();

// Interrupt Kelly
voiceController.bargeIn();

// Disconnect
await voiceController.disconnect();
```

**State:**
```dart
enum VoiceState {
  disconnected,   // Not connected
  connecting,     // Connecting
  connected,      // Ready
  listening,      // Listening for user
  userSpeaking,   // User speaking
  processing,     // Processing speech
  kellySpeaking,  // Kelly responding
  error,          // Error
}
```

### 4. **UI Widgets**

#### **VoiceControlButton** (`lib/widgets/voice_control_button.dart`)
Main voice button with animated pulse effect.

```dart
VoiceControlButton(
  size: 80.0,
  activeColor: Colors.green,
  inactiveColor: Colors.grey,
)
```

**Actions:**
- **Tap:** Start/stop listening, or barge-in
- **Long press:** Text input dialog

#### **VoiceVisualizer** (`lib/widgets/voice_visualizer.dart`)
Animated waveform showing audio energy.

```dart
VoiceVisualizer(
  height: 80.0,
  color: Colors.green,
)
```

#### **VoiceStatusIndicator**
Shows current voice state with color coding.

```dart
VoiceStatusIndicator()
```

#### **LatencyIndicator**
Shows current and average latency.

```dart
LatencyIndicator()
```

### 5. **ConversationScreen** (`lib/screens/conversation_screen.dart`)

Full-screen conversation UI with Kelly avatar.

**Features:**
- Unity avatar integration
- Voice control button
- Voice visualizer
- Conversation history
- Barge-in button
- Settings panel

---

## 🔧 **Backend WebSocket Handler**

### **Endpoint:** `ws://localhost:3000/api/realtime/ws`

### **Message Types**

#### From Flutter → Backend:

**1. Offer (WebRTC)**
```json
{
  "type": "offer",
  "sdp": "webrtc-sdp-string",
  "learnerAge": 35,
  "apiKey": "your-api-key"
}
```

**2. ICE Candidate**
```json
{
  "type": "ice_candidate",
  "candidate": {
    "candidate": "...",
    "sdpMLineIndex": 0,
    "sdpMid": "audio"
  }
}
```

**3. Start Listening**
```json
{
  "type": "start_listening"
}
```

**4. Stop Listening**
```json
{
  "type": "stop_listening"
}
```

**5. User Message**
```json
{
  "type": "user_message",
  "text": "Why do leaves change color?"
}
```

**6. Barge-in**
```json
{
  "type": "barge_in"
}
```

#### From Backend → Flutter:

**1. Answer (WebRTC)**
```json
{
  "type": "answer",
  "sdp": "webrtc-sdp-answer"
}
```

**2. Transcript**
```json
{
  "type": "transcript",
  "text": "Why do leaves change color?",
  "isFinal": true,
  "timestamp": "2025-10-30T12:00:00.000Z"
}
```

**3. Kelly Response**
```json
{
  "type": "kelly_response",
  "text": "Great question! Leaves change color because...",
  "kellyAge": 27,
  "kellyPersona": "knowledgeable-adult",
  "audio": "base64-encoded-audio",
  "timestamp": "2025-10-30T12:00:01.234Z"
}
```

**4. Error**
```json
{
  "type": "error",
  "message": "Content blocked by safety filter",
  "categories": ["inappropriate"],
  "timestamp": "2025-10-30T12:00:00.000Z"
}
```

---

## 🚀 **Setup Instructions**

### 1. **Install Flutter Dependencies**

```bash
cd curious-kellly/mobile
flutter pub get
```

**Dependencies added:**
- `flutter_webrtc`: ^0.9.48
- `web_socket_channel`: ^2.4.0
- `permission_handler`: ^11.0.1
- `record`: ^5.0.4
- `just_audio`: ^0.9.36
- `provider`: ^6.1.1
- `logger`: ^2.0.2

### 2. **Install Backend Dependencies**

```bash
cd curious-kellly/backend
npm install express-ws ajv
```

### 3. **Configure Environment**

**Backend `.env`:**
```env
OPENAI_API_KEY=sk-proj-...
OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview-2024-10-01
```

**Flutter config:**
```dart
const String backendUrl = 'https://your-backend.onrender.com';
const String openaiApiKey = 'your-key'; // For direct connection (optional)
```

### 4. **Run Backend**

```bash
npm run dev
```

Backend starts on `http://localhost:3000`

WebSocket available at: `ws://localhost:3000/api/realtime/ws`

### 5. **Run Flutter App**

```bash
flutter run
```

---

## 🧪 **Testing**

### 1. **Test WebSocket Connection**

```bash
# Install wscat
npm install -g wscat

# Connect to WebSocket
wscat -c ws://localhost:3000/api/realtime/ws

# Send test message
> {"type":"user_message","text":"Hello Kelly!"}
```

### 2. **Test Voice in Flutter**

```dart
// In your app
final voiceController = VoiceController(
  apiKey: 'your-key',
  backendUrl: 'http://localhost:3000',
);

// Connect
await voiceController.connect(learnerAge: 35);

// Start listening
voiceController.startListening();

// Check state
print(voiceController.state); // Should be VoiceState.listening
```

### 3. **Test Permissions**

```dart
final permService = PermissionService();

// Request microphone
final granted = await permService.requestMicrophonePermission();
print('Microphone: $granted');
```

### 4. **Test Voice Activity Detection**

```dart
final vad = VoiceActivityDetector();

vad.onSpeechStart = () => print('Speech started!');
vad.onSpeechEnd = () => print('Speech ended!');

// Process audio buffer
vad.processAudio(audioBytes);
```

---

## 📊 **Performance Targets**

| Metric | Target | Notes |
|--------|--------|-------|
| **Round-trip latency (RTT)** | <600ms | User speech → Kelly response |
| **Speech detection latency** | <300ms | VAD activation time |
| **Audio buffer size** | 20-50ms | Balance latency vs quality |
| **WebSocket reconnect** | <2s | Auto-reconnect on disconnect |
| **Memory usage** | <50MB | Voice services only |

### Latency Breakdown

```
User speaks → 0-300ms (VAD detection)
          → 50-150ms (Network upload)
          → 200-400ms (OpenAI processing)
          → 50-150ms (Network download)
          → 0-50ms (Audio playback)
Total: ~300-1050ms (target: <600ms avg)
```

---

## 🐛 **Troubleshooting**

### Issue: Microphone permission denied

**Solution:**
```dart
// Check permission
final hasPermission = await Permission.microphone.status.isGranted;

// Request if denied
if (!hasPermission) {
  await Permission.microphone.request();
}

// If permanently denied, open settings
if (await Permission.microphone.isPermanentlyDenied) {
  await openAppSettings();
}
```

### Issue: WebSocket connection failed

**Check:**
1. Backend running? `curl http://localhost:3000/health`
2. WebSocket endpoint accessible? `wscat -c ws://localhost:3000/api/realtime/ws`
3. CORS enabled? Check backend `cors()` middleware
4. Firewall blocking WebSocket?

### Issue: High latency (>600ms)

**Optimize:**
1. Reduce audio buffer size (trade-off: quality)
2. Use faster OpenAI model (gpt-4o-mini for tests)
3. Check network quality (WiFi vs 4G/5G)
4. Enable audio compression
5. Use CDN for backend (closer to user)

### Issue: Voice Activity Detection not working

**Tune parameters:**
```dart
VoiceActivityDetector(
  silenceThreshold: 0.01,  // Lower = more sensitive
  silenceDuration: Duration(milliseconds: 300),  // Shorter = faster cutoff
  speechDuration: Duration(milliseconds: 200),   // Shorter = faster activation
);
```

**Debug energy levels:**
```dart
vad.onEnergyUpdate = (energy) {
  print('Energy: $energy (threshold: 0.02)');
  // Adjust threshold based on typical values
};
```

### Issue: Audio playback stuttering

**Solutions:**
1. Increase buffer size
2. Use `audioplayers` instead of `just_audio` for raw PCM
3. Pre-buffer audio before playback
4. Check device CPU usage

---

## 🔐 **Security**

### 1. **API Key Protection**

**Never expose API keys in Flutter code!**

```dart
// BAD: Hardcoded key
const apiKey = 'sk-proj-...';

// GOOD: Fetch from backend
final apiKey = await http.get('$backendUrl/api/auth/session-key');
```

### 2. **Content Moderation**

All user input and Kelly output is moderated:

```javascript
// Backend safety checks
const moderationResult = await safetyService.moderateContent(userText);
if (!moderationResult.safe) {
  return { error: 'Content blocked' };
}

const ageCheck = safetyService.isAgeAppropriate(userText, learnerAge);
if (!ageCheck.appropriate) {
  return { error: 'Not age-appropriate' };
}
```

### 3. **Rate Limiting**

```javascript
// Add rate limiter (express-rate-limit)
const rateLimit = require('express-rate-limit');

const voiceLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 30, // 30 messages per minute
  message: 'Too many requests, please slow down',
});

app.use('/api/realtime/ws', voiceLimiter);
```

---

## 📚 **API Reference**

### VoiceController

```dart
class VoiceController extends ChangeNotifier {
  // Connect to service
  Future<bool> connect({required int learnerAge});
  
  // Start/stop listening
  void startListening();
  void stopListening();
  
  // Barge-in
  void bargeIn();
  
  // Send text
  void sendMessage(String text);
  
  // Disconnect
  Future<void> disconnect();
  
  // Getters
  VoiceState get state;
  int get learnerAge;
  String? get lastUserText;
  String? get lastKellyText;
  int get latencyMs;
  double get averageLatencyMs;
}
```

### OpenAIRealtimeService

```dart
class OpenAIRealtimeService {
  // Initialize
  OpenAIRealtimeService({
    required String apiKey,
    required String backendUrl,
  });
  
  // Streams
  Stream<Uint8List> get audioInputStream;
  Stream<Uint8List> get audioOutputStream;
  Stream<String> get transcriptStream;
  Stream<RealtimeEvent> get eventStream;
  
  // Callbacks
  Function(String text)? onTranscriptReceived;
  Function(Uint8List audio)? onAudioReceived;
  Function(String kellyText)? onKellyResponse;
  Function(int latencyMs)? onLatencyUpdate;
}
```

---

## 🎉 **Next Steps**

1. **Test voice integration** in Flutter app
2. **Optimize latency** to <600ms average
3. **Add viseme support** for lip-sync
4. **Implement audio caching** for common responses
5. **Add offline mode** with local TTS fallback

---

**Questions?** See `curious-kellly/mobile/lib/services/` for implementation details or `curious-kellly/backend/src/api/realtime_ws.js` for backend code.

**Performance issues?** Enable debug logging:
```dart
final logger = Logger(level: Level.debug);
```

**Need help?** Check the conversation screen example in `lib/screens/conversation_screen.dart`.















