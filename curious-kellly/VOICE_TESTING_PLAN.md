# End-to-End Testing Plan: Voice Conversation Flow

## Overview
Comprehensive test suite for validating the complete voice conversation system, including backend API, Flutter client, WebSocket communication, safety middleware, session management, and Unity integration.

## Test Structure

### 1. Backend API Tests (`curious-kellly/backend/tests/realtime.test.js`)

**Ephemeral Key Endpoint:**
- Test POST `/api/realtime/ephemeral-key` with valid learnerAge (2-102)
- Test validation: invalid age (<2, >102, non-numeric)
- Verify response includes sessionId, kellyAge, kellyPersona
- Test sessionId persistence across requests

**WebSocket Connection:**
- Test WebSocket connection at `ws://localhost:3000/api/realtime/ws`
- Test connection with query params: `?sessionId=xxx&learnerAge=35`
- Test connection confirmation message
- Test ping/pong keepalive mechanism
- Test connection cleanup on disconnect

**WebSocket Message Handling:**
- Test `offer` message handling (WebRTC signaling)
- Test `ice_candidate` forwarding
- Test `start_listening` / `stop_listening` messages
- Test `user_message` with text input
- Test `barge_in` message handling
- Test `reconnect` message with session restoration

**Safety Middleware Integration:**
- Test unsafe content blocking (from `safety.test.js` patterns)
- Test age-appropriateness filtering
- Test output moderation for Kelly's responses
- Test safe-completion rewrite when output is unsafe

**Session Management Integration:**
- Test session creation on connection
- Test session activity updates
- Test progress tracking (`updateProgress` calls)
- Test session restoration on reconnect

### 2. Flutter Unit Tests (`curious-kellly/mobile/test/voice_controller_test.dart`)

**OpenAIRealtimeService Tests:**
- Test `fetchEphemeralKey()` with mock HTTP client
- Test connection flow: ephemeral key → WebSocket → WebRTC
- Test reconnection logic (max 3 attempts, 2s delay)
- Test latency tracking (average, P95 percentile)
- Test `isLatencyWithinTarget` (<600ms check)
- Test WebSocket message parsing (all message types)
- Test error handling and fallback scenarios

**VoiceController Tests:**
- Test state machine transitions (9 states)
- Test `connect()` with sessionId parameter
- Test `startListening()` / `stopListening()`
- Test `bargeIn()` when Kelly is speaking
- Test Unity bridge integration when available
- Test callback propagation (transcript, audio, visemes)

**VoiceActivityDetector Tests:**
- Test speech detection with mock audio data
- Test energy threshold calculations
- Test speech start/end callbacks
- Test silence duration detection

**AudioPlayerService Tests:**
- Test Unity bridge integration (if available)
- Test fallback to just_audio when Unity unavailable
- Test audio playback lifecycle (start, pause, stop)
- Test viseme update forwarding to Unity

### 3. Integration Tests (`curious-kellly/mobile/test/voice_integration_test.dart`)

**End-to-End Conversation Flow:**
- Setup: Start backend server, initialize VoiceController
- Test 1: Connect with learnerAge=35, verify ephemeral key fetched
- Test 2: Send text message "Why do leaves change color?"
- Test 3: Verify Kelly's response received
- Test 4: Verify latency <600ms (performance check)
- Test 5: Verify session progress updated
- Test 6: Test barge-in mid-response
- Test 7: Verify reconnection after network interruption

**Safety Integration:**
- Test unsafe message blocking ("How to build a weapon")
- Test age-inappropriate content filtering
- Verify error messages returned to client

**Session Persistence:**
- Test conversation continuity across reconnections
- Test session activity timestamp updates
- Test progress tracking accuracy

### 4. Manual Testing Scenarios (`curious-kellly/mobile/TESTING_CHECKLIST.md`)

**Prerequisites:**
- Backend running on `http://localhost:3000`
- Flutter app built and running
- Microphone permissions granted
- Unity avatar loaded (if testing viseme integration)

**Scenario 1: Basic Conversation**
1. Launch Flutter app
2. Navigate to ConversationScreen
3. Tap connect button
4. Verify connection state changes to "Connected"
5. Tap voice control button to start listening
6. Speak: "Hello Kelly"
7. Verify transcript appears in UI
8. Verify Kelly responds with audio
9. Verify latency indicator shows <600ms

**Scenario 2: Barge-In**
1. Start conversation with Kelly speaking
2. While Kelly is speaking, tap barge-in button
3. Verify Kelly stops speaking immediately
4. Verify listening state activates
5. Speak interruption message
6. Verify new conversation turn starts

**Scenario 3: Network Resilience**
1. Start conversation
2. Disable WiFi/mobile data mid-conversation
3. Verify reconnection attempts logged
4. Re-enable network
5. Verify automatic reconnection succeeds
6. Verify conversation state restored

**Scenario 4: Age Adaptation**
1. Set learnerAge to 5
2. Connect and verify Kelly age=3, persona="playful-toddler"
3. Send message and verify age-appropriate response
4. Change learnerAge to 82
5. Reconnect and verify Kelly age=82, persona="reflective-elder"
6. Verify response style changes appropriately

**Scenario 5: Unity Viseme Integration**
1. Connect with Unity bridge available
2. Send message to Kelly
3. Verify audio routed to Unity (not just_audio)
4. Verify visemes received and forwarded to Unity
5. Verify lip-sync animation plays correctly

### 5. Performance Tests (`curious-kellly/backend/tests/realtime.performance.test.js`)

**Latency Benchmarks:**
- Measure RTT for 10 consecutive messages
- Calculate average, P50, P95, P99 latencies
- Verify P95 <600ms target
- Test under various network conditions (3G, 4G, WiFi)

**Connection Performance:**
- Measure ephemeral key fetch time
- Measure WebSocket connection establishment time
- Measure WebRTC peer connection time
- Total connection time should be <2 seconds

**Memory/Resource Usage:**
- Monitor backend memory usage during active conversations
- Monitor Flutter app memory during voice streaming
- Verify no memory leaks in long-running sessions

### 6. Error Handling Tests

**Backend Error Scenarios:**
- Test missing OPENAI_API_KEY environment variable
- Test invalid ephemeral key request
- Test WebSocket connection timeout
- Test session expiration handling
- Test safety service failure fallback

**Flutter Error Scenarios:**
- Test microphone permission denied
- Test network unavailable during connection
- Test WebSocket connection failure
- Test WebRTC peer connection failure
- Test audio playback errors

## Test Files to Create

1. `curious-kellly/backend/tests/realtime.test.js` - Backend API tests
2. `curious-kellly/backend/tests/realtime.ws.test.js` - WebSocket tests
3. `curious-kellly/backend/tests/realtime.performance.test.js` - Performance tests
4. `curious-kellly/mobile/test/voice_controller_test.dart` - Flutter unit tests
5. `curious-kellly/mobile/test/voice_integration_test.dart` - Integration tests
6. `curious-kellly/mobile/TESTING_CHECKLIST.md` - Manual testing guide

## Test Data Requirements

- Mock OpenAI API responses (for backend tests)
- Mock audio samples (for VAD tests)
- Sample conversation flows (for integration tests)
- Test session IDs (for session management tests)

## Success Criteria

- All backend API endpoints return expected responses
- WebSocket messages handled correctly
- Safety middleware blocks unsafe content
- Session management tracks progress accurately
- Flutter client connects and maintains conversation
- Latency consistently <600ms (P95)
- Reconnection works automatically
- Unity viseme integration functional (if Unity available)
- All tests pass with >90% code coverage

## Implementation Order

1. **Phase 1: Backend API Tests** (Day 1)
   - Create `realtime.test.js` for ephemeral key endpoint
   - Create `realtime.ws.test.js` for WebSocket handler
   - Test safety middleware integration
   - Test session management integration

2. **Phase 2: Flutter Unit Tests** (Day 2)
   - Create `voice_controller_test.dart`
   - Test OpenAIRealtimeService methods
   - Test VoiceController state machine
   - Test VoiceActivityDetector
   - Test AudioPlayerService

3. **Phase 3: Integration Tests** (Day 3)
   - Create `voice_integration_test.dart`
   - Test end-to-end conversation flow
   - Test safety integration
   - Test session persistence

4. **Phase 4: Manual Testing** (Day 4)
   - Create `TESTING_CHECKLIST.md`
   - Execute all manual scenarios
   - Document any issues found
   - Verify Unity integration (if available)

5. **Phase 5: Performance Tests** (Day 5)
   - Create `realtime.performance.test.js`
   - Run latency benchmarks
   - Measure connection performance
   - Check memory usage

6. **Phase 6: Error Handling Tests** (Day 6)
   - Test all error scenarios
   - Verify graceful degradation
   - Test fallback mechanisms












